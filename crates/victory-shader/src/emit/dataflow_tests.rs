//! End-to-end hardware D3D12 + WARP parity tests and `ID3D12QueryHeap` GPU
//! timestamp benchmarks for the whole-function `DATAFLOW` super-optimizer
//! (`bitcode::dataflow`).

use super::bitcode::census::{self, Census, KINDS};
use super::bitcode::opt::{self, DATAFLOW, DEAD, FACTS, FOLD, LOADS, REUSE};
use super::bitcode::{Attr, Bin, Cast, Term, pred};
use super::host::{Dispatch, Host, Root, View};
use super::shader::{Shader, flags};
use super::{HostShader, lower, validator, with_shader_opt};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const BASELINE_OPT: u32 = DEAD | FOLD | REUSE | FACTS | LOADS | opt::FIRST;
const DATAFLOW_OPT: u32 = BASELINE_OPT | DATAFLOW;

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;

const fn inline_u(n: u8) -> u16 {
    0x80 + n as u16
}
const fn sgpr(n: u8) -> u16 {
    n as u16
}
const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32 & 0x7F) << 16 | 0x03 << 8 | src0 as u32
}

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32 & 0x7F) << 16 | (op as u32) << 8 | src0 as u32
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}

const fn sopp(op: u8, simm16: i16) -> u32 {
    0xBF80_0000 | (op as u32 & 0x7F) << 16 | (simm16 as u16) as u32
}

const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

fn validate_dxil(dxil: &[u8]) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = dxil.to_vec();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }
}

fn census_kind(c: &Census, name: &str) -> u32 {
    let idx = KINDS.iter().position(|k| *k == name).expect("census kind");
    c.counts[idx]
}

/// Lowers a synthetic RDNA2 wave32 compute shader with 4 root constants in
/// `s0..s3`, raw UAV at heap slot 0 (loaded into `s[8:11]`), `threads` lanes
/// in X, and `lds_bytes` of LDS under explicit `VICTORY_SHADER_OPT` bitmask.
fn lower_rdna2_cs(
    body: &[u32],
    threads: u32,
    lds_bytes: u32,
    passes: u32,
) -> (HostShader, Census) {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s5 = 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15),
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");

    let (lowered, cen) = with_shader_opt(passes, || {
        census::with_census(|| {
            let lowered = lower::lower(&lower::Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
            })
            .expect("lower");
            let cen = census::take(&lowered.dxil).expect("census recorded");
            (lowered, cen)
        })
    });
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    validate_dxil(&shader.dxil);
    (shader, cen)
}

/// RDNA2 compute loop stressing `v_bfe_u32`, `v_bfe_i32`, `v_mul_u32_u24`,
/// `v_mad_u32_u24`, `v_bfi_b32`, and shift/mask address arithmetic.
///
/// Inputs: `s0` = loop iterations, `s1` = seed A, `s2` = seed B, `s3` = mask.
/// Thread `v0` writes its final accumulator to `uav[v0]`.
fn rdna2_bitfield_loop_body() -> Vec<u32> {
    let mut w = Vec::new();
    // v1 = v0 ^ s1, v2 = (v0 * 3 + s2), s12 = 0 (loop counter).
    w.push(vop2(0x1D, 1, sgpr(1), 0)); // v_xor_b32 v1, s1, v0
    w.push(vop2(0x25, 2, sgpr(2), 0)); // v_add_nc_u32 v2, s2, v0
    w.push(s_mov_b32(12, 0x80)); // s12 = 0

    let loop_start = w.len();
    // Dynamic-width `v_bfe_u32` (v3 = bfe_u32(v1, v4_off, v5_w)) where v4, v5 come from loop state:
    w.push(vop2(0x1B, 4, inline_u(15), 1)); // v_and_b32 v4, 15, v1  (off in 0..15)
    w.push(vop2(0x1B, 5, inline_u(7), 2)); // v_and_b32 v5, 7, v2   (width in 0..7)
    w.push(vop2(0x1C, 5, inline_u(1), 5)); // v_or_b32 v5, 1, v5    (width in 1..7)
    w.extend_from_slice(&vop3(0x148, 3, vgpr(1), vgpr(4), vgpr(5))); // v_bfe_u32 v3, v1, v4, v5
    // Second `v_bfe_i32` with dynamic offset and width in 1..7:
    w.extend_from_slice(&vop3(0x149, 6, vgpr(2), vgpr(4), vgpr(5))); // v_bfe_i32 v6, v2, v4, v5
    // `v_mul_u32_u24` and `v_mad_u32_u24` on values with provably zero upper bits:
    w.push(vop2(0x1B, 7, inline_u(31), 3)); // v_and_b32 v7, 31, v3 (known <= 31)
    w.push(vop2(0x0B, 8, vgpr(3), 7)); // v_mul_u32_u24 v8, v3, v7 (both < 2^8 -> 4 shifts collapse!)
    w.extend_from_slice(&vop3(0x143, 9, vgpr(7), vgpr(5), vgpr(8))); // v_mad_u32_u24 v9, v7, v5, v8
    // `v_bfi_b32` + shift-mask packing:
    w.extend_from_slice(&vop3(0x14A, 10, sgpr(3), vgpr(9), vgpr(6))); // v_bfi_b32 v10, s3, v9, v6
    w.push(vop2(0x1A, 11, vgpr(5), 10)); // v_lshlrev_b32 v11, v5, v10 (v5 <= 7 -> `& 31` redundant!)
    w.push(vop2(0x16, 12, vgpr(4), 10)); // v_lshrrev_b32 v12, v4, v10 (v4 <= 15 -> `& 31` redundant!)
    // Mix back into v1, v2:
    w.push(vop2(0x25, 1, vgpr(11), 1)); // v_add_nc_u32 v1, v11, v1
    w.push(vop2(0x1D, 1, vgpr(12), 1)); // v_xor_b32 v1, v12, v1
    w.push(vop2(0x25, 2, vgpr(9), 2)); // v_add_nc_u32 v2, v9, v2
    w.push(vop2(0x25, 2, inline_u(1), 2)); // v_add_nc_u32 v2, 1, v2

    // Loop counter: s12 += 1; if s12 <u s0 goto loop_start.
    w.push(sop2(0x00, 12, 12, 0x81)); // s_add_u32 s12, s12, 1
    w.push(sopc(0x0A, 12, 0)); // s_cmp_lt_u32 s12, s0
    let back_off = (loop_start as isize - (w.len() as isize + 1)) as i16;
    w.push(sopp(0x05, back_off)); // s_cbranch_scc1 loop_start

    // Store v1 ^ v2 to uav[v0 * 4].
    w.push(vop2(0x1D, 1, vgpr(2), 1)); // v_xor_b32 v1, v2, v1
    w.push(vop2(0x1A, 13, inline_u(2), 0)); // v_lshlrev_b32 v13, 2, v0
    w.extend_from_slice(&buffer_store_dword_offen(1, 13, 8));
    w
}

/// RDNA2 compute loop with nested `EXEC`-mask branches, `v_cndmask_b32` chains,
/// and cross-block known-bits propagation into `v_mul_u32_u24` + `v_bfe_u32`.
fn rdna2_exec_cndmask_loop_body() -> Vec<u32> {
    let mut w = Vec::new();
    w.push(vop2(0x1D, 1, sgpr(1), 0)); // v1 = v0 ^ s1
    w.push(vop2(0x25, 2, sgpr(2), 0)); // v2 = v0 + s2
    w.push(s_mov_b32(12, 0x80)); // s12 = 0

    let loop_start = w.len();
    // Outer mask: v3 = v1 & 31; vcc = (v3 <u 16)
    w.push(vop2(0x1B, 3, inline_u(31), 1)); // v_and_b32 v3, 31, v1
    w.push(vopc(0xC1, vgpr(3), 0)); // v_cmp_lt_u32 vcc, v3, v0 (save in s14)
    w.push(s_mov_b32(14, VCC_LO));
    w.push(vopc(0xC1, vgpr(3), 2)); // v_cmp_lt_u32 vcc, v3, v2
    w.push(sop2(0x0E, 15, 14, VCC_LO)); // s15 = s14 & vcc (implied by s14!)

    // Nested `v_cndmask_b32` cascade:
    // v4 = select(s15, v2, v1)
    // v5 = select(s14, v3, v4)   <-- when !s14, s15 is provably false => v4 == v1!
    w.extend_from_slice(&vop3(0x101, 4, vgpr(1), vgpr(2), sgpr(15))); // v_cndmask v4, v1, v2, s15
    w.extend_from_slice(&vop3(0x101, 5, vgpr(4), vgpr(3), sgpr(14))); // v_cndmask v5, v4, v3, s14

    // Divergent EXEC branch: if (v3 < 16) { ... }
    w.push(vop1(0x001, 6, inline_u(16))); // v_mov_b32 v6, 16
    w.push(vopc(0xC1, vgpr(3), 6)); // v_cmp_lt_u32 vcc, v3, v6
    w.push(sop1(0x3C, 16, VCC_LO)); // s_and_saveexec_b32 s16, vcc
    let br_idx = w.len();
    w.push(0); // placeholder for s_cbranch_execz

    // Inside EXEC branch: v3 is provably < 16!
    w.push(vop2(0x1B, 7, inline_u(7), 2)); // v7 = v2 & 7
    w.extend_from_slice(&vop3(0x148, 8, vgpr(5), vgpr(3), vgpr(7))); // v_bfe_u32 v8, v5, v3, v7
    w.push(vop2(0x0B, 9, vgpr(3), 8)); // v_mul_u32_u24 v9, v3, v8 (v3 < 16!)
    w.push(vop2(0x25, 1, vgpr(9), 1)); // v1 += v9 (guarded write!)

    let after_then = w.len();
    w[br_idx] = sopp(0x08, (after_then as isize - (br_idx as isize + 1)) as i16);
    w.push(sop2(0x10, EXEC_LO, EXEC_LO, 16)); // s_or_b32 exec_lo, exec_lo, s16

    // Join & mix:
    w.push(vop2(0x1D, 2, vgpr(5), 2)); // v2 ^= v5
    w.push(vop2(0x25, 1, inline_u(3), 1)); // v1 += 3

    w.push(sop2(0x00, 12, 12, 0x81)); // s12 += 1
    w.push(sopc(0x0A, 12, 0)); // s_cmp_lt_u32 s12, s0
    let back_off = (loop_start as isize - (w.len() as isize + 1)) as i16;
    w.push(sopp(0x05, back_off));

    w.push(vop2(0x1D, 1, vgpr(2), 1));
    w.push(vop2(0x1A, 13, inline_u(2), 0));
    w.extend_from_slice(&buffer_store_dword_offen(1, 13, 8));
    w
}

/// Direct SM 6.6 DXIL compute kernel looping `iters` times over:
/// - `dx.op.binary.i32` (`UMin` / `Tag::Clamp`) on bounded values
/// - 64-bit `zext`/`shl`/`or`/`lshr`/`trunc` pack/unpack roundtrips
/// - nested `exec`-guard `select` chains (`select(g, a, select(g & h, b, c))`)
/// - `v_bfe_u32` dynamic-width extraction + `v_mul_u32_u24` shift pairs
fn build_dxil_dataflow_cs(iters: u32, passes: u32) -> (HostShader, Census, bool) {
    let mut s = Shader::new(false);
    s.threads = Some([64, 1, 1]);
    s.user_data = 4;
    s.flags |= flags::RESOURCE_HEAP | flags::INT64;
    let ud_h = s.user_data_handle();
    let entry = s.entry;
    let (i1t, i32t, i64t) = (s.t.i1, s.t.i32, s.m.int(64));

    let k0 = s.m.i32(0);
    let k1 = s.m.i32(1);
    let k7 = s.m.i32(7);
    let k8 = s.m.i32(8);
    let k15 = s.m.i32(15);
    let k16 = s.m.i32(16);
    let k31 = s.m.i32(31);
    let k32 = s.m.i32(32);
    let k2047 = s.m.i32(2047);
    let kffff = s.m.i32(0xFFFF);
    let ones = s.m.i32(u32::MAX);
    let limit = s.m.i32(iters);
    let sh32_64 = s.m.int_const(i64t, 32);

    let tid = s.op(entry, "threadId.i32", Attr::ReadNone, i32t, 93, &[k0]);
    let cb = s.op(
        entry,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        s.t.cbuf_ret,
        59,
        &[ud_h, k0],
    );
    let seed0 = s.m.extract(entry, cb, 0);
    let seed1 = s.m.extract(entry, cb, 1);
    let init_a = s.m.bin(entry, Bin::Xor, tid, seed0);
    let init_b = s.m.bin(entry, Bin::Add, tid, seed1);

    let head = s.m.new_block();
    let then_b = s.m.new_block();
    let else_b = s.m.new_block();
    let latch = s.m.new_block();
    let exit = s.m.new_block();
    s.m.terminate(entry, Term::Br(head));

    let cnt = s.m.phi(head, i32t);
    let acc_a = s.m.phi(head, i32t);
    let acc_b = s.m.phi(head, i32t);

    // Branch on `slot < 16` where `slot = acc_a & 31`:
    let slot = s.m.bin(head, Bin::And, acc_a, k31);
    let g_outer = s.m.cmp(head, pred::IULT, slot, k16);
    let b_nz = s.m.cmp(head, pred::NE, acc_b, k0);
    let g_inner = s.m.bin(head, Bin::And, g_outer, b_nz);
    s.m.terminate(head, Term::CondBr(g_outer, then_b, else_b));

    // In `then_b`, `slot < 16` holds! Redundant clamp `UMin(slot, 2047)` + redundant `slot < 32`.
    let clamp_t = s.op(then_b, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, k2047]);
    let chk32 = s.m.cmp(then_b, pred::IULT, clamp_t, k32);
    let val_t = s.m.select(then_b, chk32, clamp_t, acc_b);
    s.m.terminate(then_b, Term::Br(latch));

    // In `else_b`, value is masked to 8 bits (`<= 255`).
    let kff = s.m.i32(0xFF);
    let val_e = s.m.bin(else_b, Bin::And, acc_b, kff);
    s.m.terminate(else_b, Term::Br(latch));

    // Latch: edge-sensitive SCKP proves `joined <= 255`!
    let joined = s.m.phi(latch, i32t);
    s.m.set_phi(joined, vec![(val_t, then_b), (val_e, else_b)]);

    // Redundant `v_mul_u32_u24` shifts `(joined << 8) >>u 8` and `Tag::Clamp` `UMin(joined, 2047)`:
    let u24_s = s.m.bin(latch, Bin::Shl, joined, k8);
    let u24 = s.m.bin(latch, Bin::LShr, u24_s, k8);
    let clamped = s.op(latch, "binary.i32", Attr::ReadNone, i32t, 40, &[u24, k2047]);

    // Redundant 64-bit pack + unpack (collapses completely, clearing INT64 flag!):
    let zlo = s.m.cast(latch, Cast::ZExt, clamped, i64t);
    let zhi = s.m.cast(latch, Cast::ZExt, acc_a, i64t);
    let hi64 = s.m.bin(latch, Bin::Shl, zhi, sh32_64);
    let pack64 = s.m.bin(latch, Bin::Or, zlo, hi64);
    let unp_lo = s.m.cast(latch, Cast::Trunc, pack64, i32t);
    let unp_hi64 = s.m.bin(latch, Bin::LShr, pack64, sh32_64);
    let unp_hi = s.m.cast(latch, Cast::Trunc, unp_hi64, i32t);

    // Dynamic `v_bfe_u32` expansion:
    let w_raw = s.m.bin(latch, Bin::And, unp_lo, k7);
    let w_b = s.m.bin(latch, Bin::Or, w_raw, k1);
    let off_b = s.m.bin(latch, Bin::And, unp_lo, k15);
    let full = s.m.cmp(latch, pred::IUGE, w_b, k32);
    let w_m = s.m.bin(latch, Bin::And, w_b, k31);
    let bfe_sh = s.m.bin(latch, Bin::LShr, unp_hi, off_b);
    let bfe_m = s.m.bin(latch, Bin::Shl, k1, w_m);
    let bfe_m = s.m.bin(latch, Bin::Sub, bfe_m, k1);
    let bfe_m = s.m.select(latch, full, ones, bfe_m);
    let bfe_r = s.m.bin(latch, Bin::And, bfe_sh, bfe_m);

    // Nested `exec`-guard select cascade + dead upper pack bits under `& 0xFFFF`:
    let inner_sel = s.m.select(latch, g_inner, acc_b, bfe_r);
    let outer_sel = s.m.select(latch, g_outer, unp_lo, inner_sel);
    let hi_tag = s.m.bin(latch, Bin::Shl, acc_a, k16);
    let tagged = s.m.bin(latch, Bin::Or, hi_tag, outer_sel);
    let low_only = s.m.bin(latch, Bin::And, tagged, kffff);

    let next_a = s.m.bin(latch, Bin::Add, unp_hi, low_only);
    let next_b = s.m.bin(latch, Bin::Xor, acc_b, clamped);
    let next_b = s.m.bin(latch, Bin::Add, next_b, k1);
    let next_cnt = s.m.bin(latch, Bin::Add, cnt, k1);
    let more = s.m.cmp(latch, pred::IULT, next_cnt, limit);
    let _ = i1t;
    s.m.terminate(latch, Term::CondBr(more, head, exit));
    s.m.set_phi(cnt, vec![(k0, entry), (next_cnt, latch)]);
    s.m.set_phi(acc_a, vec![(init_a, entry), (next_a, latch)]);
    s.m.set_phi(acc_b, vec![(init_b, entry), (next_b, latch)]);

    // Store final `next_a ^ next_b` to heap UAV 0 at byte offset `tid * 4`
    // (one unique dword per thread across the full dispatch, zero race).
    let res_val = s.m.bin(exit, Bin::Xor, next_a, next_b);
    let k2 = s.m.i32(2);
    let byte_off = s.m.bin(exit, Bin::Shl, tid, k2);
    let uav_h = s.heap_handle_at(exit, k0, false, false, (11 | 0x1000, 0), true);
    let iu = s.m.undef(i32t);
    let mask1 = s.m.i8(1);
    let align4 = s.m.i32(4);
    let void = s.t.void;
    s.op(
        exit,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        void,
        140,
        &[uav_h, byte_off, iu, res_val, iu, iu, iu, mask1, align4],
    );
    s.m.terminate(exit, Term::Ret);

    let (dxil, cen, has_int64) = with_shader_opt(passes, || {
        census::with_census(|| {
            let dxil = s.finish();
            let cen = census::take(&dxil).expect("census recorded");
            // Read SFI0 feature flags to check if INT64 stayed set.
            let c = crate::dxbc::Container::parse(&dxil).expect("dxbc");
            let sfi0 = c.part(crate::dxbc::FourCC(*b"SFI0")).expect("SFI0").data;
            let feat = u64::from_le_bytes(sfi0[..8].try_into().unwrap());
            (dxil, cen, feat & 0x8000 != 0)
        })
    });
    validate_dxil(&dxil);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: vec![
            super::UserData {
                index: 0,
                slot: 0,
                root: RootType::Constants,
                dword: 0,
            },
            super::UserData {
                index: 1,
                slot: 0,
                root: RootType::Constants,
                dword: 1,
            },
            super::UserData {
                index: 2,
                slot: 0,
                root: RootType::Constants,
                dword: 2,
            },
            super::UserData {
                index: 3,
                slot: 0,
                root: RootType::Constants,
                dword: 3,
            },
        ],
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    (shader, cen, has_int64)
}

/// Runs a compute shader with D3D12 `ID3D12QueryHeap` GPU timestamps and
/// returns `(uav_output_first_run, median_gpu_us)`.
#[allow(clippy::too_many_arguments)]
fn dispatch_timed(
    host: &Host,
    shader: &HostShader,
    root_consts: &[u32; 4],
    uav_dwords: u32,
    groups: [u32; 3],
    warmups: usize,
    samples: usize,
) -> Result<(Vec<u32>, f64), String> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::Interface;

    let err = |e: windows::core::Error| e.message().to_string();
    let (root, pso) = host.compute_pipeline(shader)?;
    let dev = &host.device;
    // SAFETY: standard D3D12 resource/query lifetimes; every GPU submission is
    // fenced before mapping readback buffers.
    unsafe {
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let freq = queue.GetTimestampFrequency().map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        list.Close().map_err(err)?;

        let make_buf = |heap: D3D12_HEAP_TYPE, bytes: u64, flags, state| {
            let mut r: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut r,
            )
            .map_err(err)?;
            r.ok_or_else(|| "no buffer".to_owned())
        };

        let uav_bytes = u64::from(uav_dwords) * 4;
        let uav = make_buf(
            D3D12_HEAP_TYPE_DEFAULT,
            uav_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let uav_rb = make_buf(
            D3D12_HEAP_TYPE_READBACK,
            uav_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let ts_rb = make_buf(
            D3D12_HEAP_TYPE_READBACK,
            16,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let view_heap: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(err)?;
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: uav_dwords,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        dev.CreateUnorderedAccessView(
            &uav,
            None,
            Some(&uav_desc),
            view_heap.GetCPUDescriptorHandleForHeapStart(),
        );

        let mut qheap: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: 2,
                NodeMask: 0,
            },
            &mut qheap,
        )
        .map_err(err)?;
        let qheap = qheap.ok_or_else(|| "no query heap".to_owned())?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        let mut fence_val = 0u64;
        let mut times_us = Vec::with_capacity(samples);
        let mut first_out = Vec::new();

        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => root_consts[u.dword as usize],
                (RootType::Table, _) => 0,
                _ => 0,
            };
        }

        for run_idx in 0..(warmups + samples) {
            alloc.Reset().map_err(err)?;
            list.Reset(&alloc, &pso).map_err(err)?;
            list.SetDescriptorHeaps(&[Some(view_heap.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
            list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.Dispatch(groups[0], groups[1], groups[2]);
            list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);
            if run_idx == 0 {
                let to_copy = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(&uav),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                            },
                        ),
                    },
                };
                list.ResourceBarrier(&[to_copy]);
                list.CopyBufferRegion(&uav_rb, 0, &uav, 0, uav_bytes);
                let to_uav = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(&uav),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_COPY_SOURCE,
                                StateAfter: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            },
                        ),
                    },
                };
                list.ResourceBarrier(&[to_uav]);
            }
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            fence_val += 1;
            queue.Signal(&fence, fence_val).map_err(err)?;
            while fence.GetCompletedValue() < fence_val {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            if run_idx == 0 {
                let mut p = std::ptr::null_mut();
                uav_rb.Map(0, None, Some(&mut p)).map_err(err)?;
                first_out =
                    std::slice::from_raw_parts(p.cast::<u32>(), uav_dwords as usize).to_vec();
                uav_rb.Unmap(0, None);
            }
            if run_idx >= warmups {
                let mut p = std::ptr::null_mut();
                ts_rb.Map(0, None, Some(&mut p)).map_err(err)?;
                let ts = *p.cast::<[u64; 2]>();
                ts_rb.Unmap(0, None);
                let us = (ts[1].saturating_sub(ts[0]) as f64) * 1e6 / (freq as f64);
                times_us.push(us);
            }
        }
        times_us.sort_by(|a, b| a.total_cmp(b));
        Ok((first_out, times_us[times_us.len() / 2]))
    }
}

fn dispatch_simple(
    host: &Host,
    shader: &HostShader,
    root_consts: &[u32; 4],
    uav_dwords: u32,
    groups: [u32; 3],
) -> Vec<u32> {
    let (root, pso) = host.compute_pipeline(shader).expect("compute pipeline");
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => root_consts[u.dword as usize],
            (RootType::Table, _) => 0,
            _ => 0,
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(uav_dwords)],
                samplers: &[],
                groups,
            },
        )
        .expect("dispatch");
    out[0].clone()
}

#[test]
fn rdna2_bitfield_and_u24_loop_hw_parity() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let body = rdna2_bitfield_loop_body();
    let (sh_none, c_none) = lower_rdna2_cs(&body, 64, 0, 0);
    let (sh_base, c_base) = lower_rdna2_cs(&body, 64, 0, BASELINE_OPT);
    let (sh_df, c_df) = lower_rdna2_cs(&body, 64, 0, DATAFLOW_OPT);

    let (n_none, n_base, n_df) = (
        census_kind(&c_none, "total"),
        census_kind(&c_base, "total"),
        census_kind(&c_df, "total"),
    );
    assert!(
        n_df < n_base && n_base < n_none,
        "expected strict DXIL instruction reduction: opt=0 ({n_none}) -> base ({n_base}) -> dataflow ({n_df})"
    );
    assert!(
        sh_df.dxil.len() < sh_base.dxil.len(),
        "expected smaller DXIL container: {} -> {}",
        sh_base.dxil.len(),
        sh_df.dxil.len()
    );

    let cases = [
        [8, 0, 0, 0],
        [16, 0x1234_5678, 0x9ABC_DEF0, 0x0F0F_0F0F],
        [24, u32::MAX, 0x8000_0000, 0xAAAA_5555],
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        for c in &cases {
            let out_none = dispatch_simple(&host, &sh_none, c, 64, [1, 1, 1]);
            let out_base = dispatch_simple(&host, &sh_base, c, 64, [1, 1, 1]);
            let out_df = dispatch_simple(&host, &sh_df, c, 64, [1, 1, 1]);
            assert_eq!(out_df, out_none, "{label} case {c:?} vs opt=0");
            assert_eq!(out_df, out_base, "{label} case {c:?} vs baseline");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn rdna2_exec_mask_and_cndmask_loop_hw_parity() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let body = rdna2_exec_cndmask_loop_body();
    let (sh_none, c_none) = lower_rdna2_cs(&body, 64, 0, 0);
    let (sh_base, c_base) = lower_rdna2_cs(&body, 64, 0, BASELINE_OPT);
    let (sh_df, c_df) = lower_rdna2_cs(&body, 64, 0, DATAFLOW_OPT);

    let (n_none, n_base, n_df) = (
        census_kind(&c_none, "total"),
        census_kind(&c_base, "total"),
        census_kind(&c_df, "total"),
    );
    assert!(
        n_df < n_base && n_base < n_none,
        "expected strict DXIL instruction reduction: opt=0 ({n_none}) -> base ({n_base}) -> dataflow ({n_df})"
    );

    let cases = [
        [12, 0, 1, 0],
        [20, 0xCAFE_BABE, 0x0000_0007, 0],
        [20, 0xFFFF_FFFF, 0x8000_0010, 0],
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        for c in &cases {
            let out_none = dispatch_simple(&host, &sh_none, c, 64, [1, 1, 1]);
            let out_base = dispatch_simple(&host, &sh_base, c, 64, [1, 1, 1]);
            let out_df = dispatch_simple(&host, &sh_df, c, 64, [1, 1, 1]);
            assert_eq!(out_df, out_none, "{label} case {c:?} vs opt=0");
            assert_eq!(out_df, out_base, "{label} case {c:?} vs baseline");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn wave_ordered_barrierless_lds_soundness_hw_parity() {
    // Negative control for soundness rule #1 (#678):
    // 1) Same-pointer barrierless `ds_write_b32 lds[tid], tid` followed by
    //    `ds_read_b32 lds[tid]` (disjoint per-lane slots 0..64 so 64-thread
    //    groups have zero cross-warp data race) MUST keep the `addrspace(3)`
    //    load in emitted DXIL (zero space-3 store-to-load forwarding).
    // 2) Cross-lane LDS exchange (`lds[64 + tid] = tid ^ s1`, read
    //    `lds[64 + (tid ^ 1)]` after `s_barrier`) must match bit-exact across
    //    `opt=0`, `opt=63`, and `opt=575` on both HW & WARP.
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    // Same-pointer barrierless write + read on race-free per-lane `lds[tid]`:
    body.push(vop2(0x1A, 1, inline_u(2), 0)); // v1 = tid * 4
    body.extend_from_slice(&ds(0x0D, 0, 1, 0)); // ds_write_b32 v1, v0
    body.extend_from_slice(&ds(0x36, 2, 1, 0)); // ds_read_b32 v2, v1 (NO barrier!)
    // Cross-lane exchange on `lds[64 + tid]` -> `lds[64 + (tid ^ 1)]` after `s_barrier`:
    body.push(vop2(0x25, 3, inline_u(64), 0)); // v3 = 64 + tid
    body.push(vop2(0x1A, 3, inline_u(2), 3)); // v3 = (64 + tid) * 4
    body.push(vop2(0x1D, 4, sgpr(1), 0)); // v4 = tid ^ s1
    body.extend_from_slice(&ds(0x0D, 0, 3, 4)); // ds_write_b32 v3, v4
    body.push(0xBF8A_0000); // s_barrier
    body.push(vop2(0x1D, 5, inline_u(1), 0)); // v5 = tid ^ 1
    body.push(vop2(0x25, 5, inline_u(64), 5)); // v5 = 64 + (tid ^ 1)
    body.push(vop2(0x1A, 5, inline_u(2), 5)); // v5 = (64 + (tid ^ 1)) * 4
    body.extend_from_slice(&ds(0x36, 6, 5, 0)); // ds_read_b32 v6, v5
    // Mix both reads and store per-lane:
    body.push(vop2(0x1D, 6, vgpr(2), 6)); // v6 ^= v2 (keeps same-slot load live!)
    body.push(vop2(0x1A, 7, inline_u(2), 0)); // v7 = tid * 4
    body.extend_from_slice(&buffer_store_dword_offen(6, 7, 8));

    let (sh_none, c_none) = lower_rdna2_cs(&body, 64, 512, 0);
    let (sh_base, c_base) = lower_rdna2_cs(&body, 64, 512, BASELINE_OPT);
    let (sh_df, c_df) = lower_rdna2_cs(&body, 64, 512, DATAFLOW_OPT);
    assert_eq!(
        census_kind(&c_df, "lds_load"),
        2,
        "both LDS loads (same-slot barrierless + cross-lane) must remain in DXIL (none={}, base={}, df={})",
        census_kind(&c_none, "lds_load"),
        census_kind(&c_base, "lds_load"),
        census_kind(&c_df, "lds_load"),
    );
    assert_eq!(census_kind(&c_df, "lds_store"), 2, "both LDS stores must remain");

    let consts = [0, 0x5A5A_1234, 0, 0];
    let want: Vec<u32> = (0..64u32).map(|tid| tid ^ ((tid ^ 1) ^ consts[1])).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let out_none = dispatch_simple(&host, &sh_none, &consts, 64, [1, 1, 1]);
        let out_base = dispatch_simple(&host, &sh_base, &consts, 64, [1, 1, 1]);
        let out_df = dispatch_simple(&host, &sh_df, &consts, 64, [1, 1, 1]);
        assert_eq!(out_df, want, "{label}: DATAFLOW LDS exchange output mismatch");
        assert_eq!(out_df, out_none, "{label}: DATAFLOW vs opt=0 mismatch on LDS exchange");
        assert_eq!(out_df, out_base, "{label}: DATAFLOW vs baseline mismatch on LDS exchange");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn dxil_whole_function_bitfield_clamp_exec_hw_parity() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (sh_base, c_base, int64_base) = build_dxil_dataflow_cs(32, BASELINE_OPT);
    let (sh_df, c_df, int64_df) = build_dxil_dataflow_cs(32, DATAFLOW_OPT);

    let (n_base, n_df) = (census_kind(&c_base, "total"), census_kind(&c_df, "total"));
    assert!(int64_base, "baseline keeps redundant i64 pack/unpack");
    assert!(!int64_df, "DATAFLOW eliminates all i64 instructions and clears INT64 flag");
    assert!(
        n_df * 3 <= n_base * 2,
        "expected >=33% total DXIL instruction reduction on bitfield+clamp+exec kernel: {n_base} -> {n_df}"
    );

    let cases = [
        [0, 0, 0, 0],
        [0x1357_9BDF, 0x2468_ACE0, 0, 0],
        [u32::MAX, 0x8000_0001, 0, 0],
    ];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        for c in &cases {
            let out_base = dispatch_simple(&host, &sh_base, c, 1024, [16, 1, 1]);
            let out_df = dispatch_simple(&host, &sh_df, c, 1024, [16, 1, 1]);
            assert_eq!(out_df, out_base, "{label} case {c:?}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
#[ignore = "GPU timestamp benchmark: cargo test --release -p victory-shader --locked bench_dataflow_hw_gpu -- --ignored --nocapture"]
fn bench_dataflow_hw_gpu() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(false) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("no SM 6.6 hardware GPU ({e}); benchmark skipped");
            return;
        }
    };
    println!(
        "Hardware adapter: {}",
        host.adapter_name().unwrap_or_else(|| "unknown".into())
    );

    // 1. RDNA2 Bitfield (`v_bfe_u32`/`v_bfe_i32`/`v_mul_u32_u24`/`v_mad_u32_u24`/`v_bfi_b32`) loop:
    {
        let body = rdna2_bitfield_loop_body();
        let (_, c_none) = lower_rdna2_cs(&body, 64, 0, 0);
        let (sh_base, c_base) = lower_rdna2_cs(&body, 64, 0, BASELINE_OPT);
        let (sh_df, c_df) = lower_rdna2_cs(&body, 64, 0, DATAFLOW_OPT);
        let consts = [512, 0x1234_5678, 0x9ABC_DEF0, 0x0F0F_F0F0];
        let (out_base, us_base) =
            dispatch_timed(&host, &sh_base, &consts, 64, [4096, 1, 1], 5, 19).expect("base");
        let (out_df, us_df) =
            dispatch_timed(&host, &sh_df, &consts, 64, [4096, 1, 1], 5, 19).expect("dataflow");
        assert_eq!(out_df, out_base, "bitfield loop parity mismatch");
        let (n0, nb, nd) = (
            census_kind(&c_none, "total"),
            census_kind(&c_base, "total"),
            census_kind(&c_df, "total"),
        );
        println!(
            "[1] RDNA2 bitfield+u24 loop (4096x64, 512 iters): insts {n0} -> {nb} (base) -> {nd} (dataflow, {:.1}%), bytes {} -> {}, GPU median {us_base:.2} -> {us_df:.2} us ({:.1}%)",
            (nd as f64 - nb as f64) * 100.0 / (nb as f64),
            sh_base.dxil.len(),
            sh_df.dxil.len(),
            (us_df - us_base) * 100.0 / us_base,
        );
    }

    // 2. RDNA2 Exec-Mask + `v_cndmask_b32` + cross-block KnownBits loop:
    {
        let body = rdna2_exec_cndmask_loop_body();
        let (_, c_none) = lower_rdna2_cs(&body, 64, 0, 0);
        let (sh_base, c_base) = lower_rdna2_cs(&body, 64, 0, BASELINE_OPT);
        let (sh_df, c_df) = lower_rdna2_cs(&body, 64, 0, DATAFLOW_OPT);
        let consts = [512, 0xCAFE_BABE, 0x0000_0017, 0];
        let (out_base, us_base) =
            dispatch_timed(&host, &sh_base, &consts, 64, [4096, 1, 1], 5, 19).expect("base");
        let (out_df, us_df) =
            dispatch_timed(&host, &sh_df, &consts, 64, [4096, 1, 1], 5, 19).expect("dataflow");
        assert_eq!(out_df, out_base, "exec+cndmask loop parity mismatch");
        let (n0, nb, nd) = (
            census_kind(&c_none, "total"),
            census_kind(&c_base, "total"),
            census_kind(&c_df, "total"),
        );
        println!(
            "[2] RDNA2 exec+cndmask loop (4096x64, 512 iters): insts {n0} -> {nb} (base) -> {nd} (dataflow, {:.1}%), bytes {} -> {}, GPU median {us_base:.2} -> {us_df:.2} us ({:.1}%)",
            (nd as f64 - nb as f64) * 100.0 / (nb as f64),
            sh_base.dxil.len(),
            sh_df.dxil.len(),
            (us_df - us_base) * 100.0 / us_base,
        );
    }

    // 3. Whole-function DXIL bitfield + Tag::Clamp + i64 pack/unpack + exec-select loop:
    {
        let (_, c_none, _) = build_dxil_dataflow_cs(512, 0);
        let (sh_base, c_base, _) = build_dxil_dataflow_cs(512, BASELINE_OPT);
        let (sh_df, c_df, int64_df) = build_dxil_dataflow_cs(512, DATAFLOW_OPT);
        assert!(!int64_df);
        let consts = [0x1357_9BDF, 0x2468_ACE0, 0, 0];
        let uav_dwords = 4096 * 64;
        let (out_base, us_base) =
            dispatch_timed(&host, &sh_base, &consts, uav_dwords, [4096, 1, 1], 5, 19)
                .expect("base");
        let (out_df, us_df) =
            dispatch_timed(&host, &sh_df, &consts, uav_dwords, [4096, 1, 1], 5, 19)
                .expect("dataflow");
        assert_eq!(out_df, out_base, "dxil whole-function loop parity mismatch");
        let (n0, nb, nd) = (
            census_kind(&c_none, "total"),
            census_kind(&c_base, "total"),
            census_kind(&c_df, "total"),
        );
        println!(
            "[3] Whole-fn bitfield+clamp+i64+exec (4096x64, 512 iters): insts {n0} -> {nb} (base) -> {nd} (dataflow, {:.1}%), bytes {} -> {}, GPU median {us_base:.2} -> {us_df:.2} us ({:.1}%)",
            (nd as f64 - nb as f64) * 100.0 / (nb as f64),
            sh_base.dxil.len(),
            sh_df.dxil.len(),
            (us_df - us_base) * 100.0 / us_base,
        );
    }
}
