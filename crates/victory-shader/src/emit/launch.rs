//! Amplification shader (`as_6_6`) that launches a single direct or indirect
//! mesh draw directly on the GPU (`victory-gpu`'s `launch`, `VICTORY_MESH_LAUNCH`).
//!
//! Ported from fxceless/victory-fork (`b1d777a`, `0e006f0`) and kelvaroq
//! (`0041-mesh-launch-wide-hash.patch`), emitted directly through our DXIL
//! builder so no binary or hex shader blob lives in the repository and the
//! clamping logic matches `victory-gpu::indirect` exactly:
//!
//! - `b0, space9`: `uint direct` (`0xffff_ffff` for indirect record, else CPU
//!   group count clamped to 65,535 and unbounded by budget).
//! - `t0, space9`: `ByteAddressBuffer record` (`uint3` group counts).
//!   Records with `x > 65535` (unfilled argument slots) run empty (`0, 0, 0`),
//!   matching `victory-gpu`'s indirect clamp; `y` and `z` clamp to `min(_, 1)`.
//! - `u0, space9`: `RWByteAddressBuffer budget` (5 dwords: `[0]` work charged
//!   against `BUDGET = 1 << 27` at 128/group, `[1]` max asked X, `[2]` total
//!   asked X clamped to `1 << 24` per record, `[3]` refused count, `[4]` max
//!   cost `>= 1 << 22`).

use super::bitcode::{Attr, Bin, Term, pred};
use super::shader::{LAUNCH_SPACE, Shader};

/// Per-list work budget (matches `victory-gpu::indirect`'s `BUDGET`).
const BUDGET: u32 = 1 << 27;
/// Sentinel value in `b0, space9` selecting indirect record launch.
const DIRECT_NONE: u32 = u32::MAX;
/// Payload size in bytes declared for `dx.op.dispatchMesh` and matched by
/// translated mesh shaders when `VICTORY_MESH_LAUNCH` is enabled.
const PAYLOAD_BYTES: u32 = 4;

/// Builds a validated DXIL (`as_6_6`) container for the mesh-launch
/// amplification shader.
///
/// When `rest` is `false` (default, matching `VICTORY_INDIRECT_REST=0`), a
/// record whose cost does not fit in the remaining budget runs empty without
/// consuming the remaining budget so smaller later draws can still run. When
/// `rest` is `true`, an oversized record saturates the budget to `BUDGET`.
pub fn mesh_launch_as(rest: bool) -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([1, 1, 1]);
    s.amplification = Some(PAYLOAD_BYTES);
    s.root_space = LAUNCH_SPACE;
    s.root_cbv_size = 4;

    let entry = s.entry;
    let b_direct = s.m.new_block();
    let b_record = s.m.new_block();
    let b_charge = s.m.new_block();
    let b_loop_hdr = s.m.new_block();
    let b_loop_body = s.m.new_block();
    let b_cas = (!rest).then(|| s.m.new_block());
    let b_cas_done = rest.then(|| s.m.new_block());
    let b_fitted = s.m.new_block();
    let b_biggest = s.m.new_block();
    let b_refused = s.m.new_block();
    let b_refused_big = s.m.new_block();
    let b_stats = s.m.new_block();
    let b_dispatch = s.m.new_block();

    let i32_ty = s.t.i32;
    let c0 = s.m.i32(0);
    let c1 = s.m.i32(1);
    let c4 = s.m.i32(4);
    let c7 = s.m.i32(7);
    let c8 = s.m.i32(8);
    let c12 = s.m.i32(12);
    let c16 = s.m.i32(16);
    let c65535 = s.m.i32(65_535);
    let c_direct_none = s.m.i32(DIRECT_NONE);
    let c_budget = s.m.i32(BUDGET);
    let c_big_thresh = s.m.i32(1 << 22);
    let c_asked_cap = s.m.i32(1 << 24);
    let undef_i32 = s.m.undef(i32_ty);

    // Resource handles in `LAUNCH_SPACE`: `b0` (direct), `t0` (record), `u0` (budget).
    let cb0 = s.root_cbv_handle(0);
    let t0 = s.root_srv_handle(0);
    let u0 = s.root_uav_handle(0);

    // entry: load `direct` from `b0, space9`.
    let cb = s.op(entry, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, &[cb0, c0]);
    let direct = s.m.extract(entry, cb, 0);
    let is_record = s.m.cmp(entry, pred::EQ, direct, c_direct_none);
    s.m.terminate(entry, Term::CondBr(is_record, b_record, b_direct));

    // b_direct: CPU-known dispatch count clamped to D3D12 limit, unbounded by budget.
    let direct_x = s.op(b_direct, "binary.i32", Attr::ReadNone, i32_ty, 40, &[direct, c65535]);
    s.m.terminate(b_direct, Term::Br(b_dispatch));

    // b_record: load 3 dwords (x, y, z) from `record` (`t0, space9`).
    let res_ret_i32 = s
        .m
        .named_struct("dx.types.ResRet.i32", vec![i32_ty, i32_ty, i32_ty, i32_ty, i32_ty]);
    let mask7 = s.m.i8(7);
    let loaded = s.op(
        b_record,
        "rawBufferLoad.i32",
        Attr::ReadOnly,
        res_ret_i32,
        139,
        &[t0, c0, undef_i32, mask7, c4],
    );
    let rx = s.m.extract(b_record, loaded, 0);
    let ry = s.m.extract(b_record, loaded, 1);
    let rz = s.m.extract(b_record, loaded, 2);

    // Unfilled argument slots (`x > 65535`, e.g. all ones) run empty (0, 0, 0).
    let over_max = s.m.cmp(b_record, pred::IUGT, rx, c65535);
    let asked = s.m.select(b_record, over_max, c0, rx);
    let ry_valid = s.m.select(b_record, over_max, c0, ry);
    let rz_valid = s.m.select(b_record, over_max, c0, rz);
    let gx0 = asked;
    let gy0 = s.op(b_record, "binary.i32", Attr::ReadNone, i32_ty, 40, &[ry_valid, c1]);
    let gz0 = s.op(b_record, "binary.i32", Attr::ReadNone, i32_ty, 40, &[rz_valid, c1]);
    let cost_x = s.m.bin(b_record, Bin::Shl, gx0, c7);
    let cost_xy = s.m.bin(b_record, Bin::Mul, cost_x, gy0);
    let cost = s.m.bin(b_record, Bin::Mul, cost_xy, gz0);
    let zero_cost = s.m.cmp(b_record, pred::EQ, cost, c0);
    s.m.terminate(b_record, Term::CondBr(zero_cost, b_stats, b_charge));

    // b_charge: initial atomic read of `budget[0]` via `InterlockedOr(0, 0)`.
    let atomic_or = s.m.i32(2);
    let atomic_add = s.m.i32(0);
    let atomic_umax = s.m.i32(7);
    let init_before = s.op(
        b_charge,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_or, c0, undef_i32, undef_i32, c0],
    );
    s.m.terminate(b_charge, Term::Br(b_loop_hdr));

    // b_loop_hdr: CAS loop checking remaining budget against `cost`.
    let before = s.m.phi(b_loop_hdr, i32_ty);
    let under_budget = s.m.cmp(b_loop_hdr, pred::IULT, before, c_budget);
    s.m.terminate(b_loop_hdr, Term::CondBr(under_budget, b_loop_body, b_refused));

    // b_loop_body: compute remaining budget and attempt CAS.
    let rem = s.m.bin(b_loop_body, Bin::Sub, c_budget, before);
    let fits_now = s.m.cmp(b_loop_body, pred::IUGE, rem, cost);
    if let Some(cas_blk) = b_cas {
        // `REST == 0`: if it does not fit, leave remaining budget untouched.
        s.m.terminate(b_loop_body, Term::CondBr(fits_now, cas_blk, b_refused));
        let next = s.m.bin(cas_blk, Bin::Add, before, cost);
        let prev = s.op(
            cas_blk,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32_ty,
            79,
            &[u0, c0, undef_i32, undef_i32, before, next],
        );
        let won = s.m.cmp(cas_blk, pred::EQ, prev, before);
        s.m.terminate(cas_blk, Term::CondBr(won, b_fitted, b_loop_hdr));
        s.m.set_phi(before, vec![(init_before, b_charge), (prev, cas_blk)]);
    } else {
        // `REST == 1`: an oversized record saturates the budget to `BUDGET`.
        let cas_done = b_cas_done.expect("b_cas_done exists when rest is true");
        let sum = s.m.bin(b_loop_body, Bin::Add, before, cost);
        let next = s.m.select(b_loop_body, fits_now, sum, c_budget);
        let prev = s.op(
            b_loop_body,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32_ty,
            79,
            &[u0, c0, undef_i32, undef_i32, before, next],
        );
        let won = s.m.cmp(b_loop_body, pred::EQ, prev, before);
        s.m.terminate(b_loop_body, Term::CondBr(won, cas_done, b_loop_hdr));
        s.m.set_phi(before, vec![(init_before, b_charge), (prev, b_loop_body)]);
        s.m.terminate(cas_done, Term::CondBr(fits_now, b_fitted, b_refused));
    }

    // b_fitted / b_biggest: track largest cost >= 2^22 in `budget[4]`.
    let big_fit = s.m.cmp(b_fitted, pred::IUGE, cost, c_big_thresh);
    s.m.terminate(b_fitted, Term::CondBr(big_fit, b_biggest, b_stats));
    let _ = s.op(
        b_biggest,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_umax, c16, undef_i32, undef_i32, cost],
    );
    s.m.terminate(b_biggest, Term::Br(b_stats));

    // b_refused / b_refused_big: record refusal in `budget[3]` and track largest cost.
    let _ = s.op(
        b_refused,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_add, c12, undef_i32, undef_i32, c1],
    );
    let big_ref = s.m.cmp(b_refused, pred::IUGE, cost, c_big_thresh);
    s.m.terminate(b_refused, Term::CondBr(big_ref, b_refused_big, b_stats));
    let _ = s.op(
        b_refused_big,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_umax, c16, undef_i32, undef_i32, cost],
    );
    s.m.terminate(b_refused_big, Term::Br(b_stats));

    // b_stats: update `budget[1]` (largest asked) and `budget[2]` (total asked).
    let rec_gx = s.m.phi(b_stats, i32_ty);
    let rec_gy = s.m.phi(b_stats, i32_ty);
    let rec_gz = s.m.phi(b_stats, i32_ty);
    s.m.set_phi(
        rec_gx,
        vec![
            (gx0, b_record),
            (gx0, b_fitted),
            (gx0, b_biggest),
            (c0, b_refused),
            (c0, b_refused_big),
        ],
    );
    s.m.set_phi(
        rec_gy,
        vec![
            (gy0, b_record),
            (gy0, b_fitted),
            (gy0, b_biggest),
            (c0, b_refused),
            (c0, b_refused_big),
        ],
    );
    s.m.set_phi(
        rec_gz,
        vec![
            (gz0, b_record),
            (gz0, b_fitted),
            (gz0, b_biggest),
            (c0, b_refused),
            (c0, b_refused_big),
        ],
    );
    let _ = s.op(
        b_stats,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_umax, c4, undef_i32, undef_i32, asked],
    );
    let clamped_asked = s.op(b_stats, "binary.i32", Attr::ReadNone, i32_ty, 40, &[asked, c_asked_cap]);
    let _ = s.op(
        b_stats,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32_ty,
        78,
        &[u0, atomic_add, c8, undef_i32, undef_i32, clamped_asked],
    );
    s.m.terminate(b_stats, Term::Br(b_dispatch));

    // b_dispatch: initialize groupshared payload and call `dx.op.dispatchMesh`.
    let gx = s.m.phi(b_dispatch, i32_ty);
    let gy = s.m.phi(b_dispatch, i32_ty);
    let gz = s.m.phi(b_dispatch, i32_ty);
    s.m.set_phi(gx, vec![(direct_x, b_direct), (rec_gx, b_stats)]);
    s.m.set_phi(gy, vec![(c1, b_direct), (rec_gy, b_stats)]);
    s.m.set_phi(gz, vec![(c1, b_direct), (rec_gz, b_stats)]);

    let payload_ty = s.m.named_struct("struct.Payload", vec![i32_ty]);
    let payload = s.m.global("payload", payload_ty, 3);
    let payload_field = s.m.field(b_dispatch, payload, 0);
    s.m.store(b_dispatch, payload_field, c0);
    let _ = s.op(
        b_dispatch,
        "dispatchMesh.struct.Payload",
        Attr::NoUnwind,
        s.t.void,
        173,
        &[gx, gy, gz, payload],
    );
    s.m.terminate(b_dispatch, Term::Ret);

    s.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxbc::{Container, FourCC};

    #[test]
    fn mesh_launch_amplification_shader_is_valid_dxil() {
        for rest in [false, true] {
            let mut dxil = mesh_launch_as(rest);
            let c = Container::parse(&dxil).expect("valid DXBC container");
            // Program kind 14 (amplification), shader model 6.6.
            assert_eq!(c.part(FourCC::DXIL).unwrap().data[..4], [0x66, 0, 14, 0]);
            let psv = c.part(FourCC::PSV0).unwrap().data;
            // PSV0 info size 48, payload bytes 4, stage 14, threads (1, 1, 1).
            assert_eq!(&psv[0..4], &48u32.to_le_bytes());
            assert_eq!(&psv[4..8], &4u32.to_le_bytes());
            assert_eq!(psv[4 + 24], 14);
            assert_eq!(&psv[4 + 36..4 + 48], &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
            #[cfg(windows)]
            if let Ok(mut v) = super::super::validator::Validator::load(
                super::super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut dxil)
                    .unwrap_or_else(|e| panic!("dxil.dll rejected mesh_launch_as(rest={rest}): {e}"));
            }
        }
    }
}
