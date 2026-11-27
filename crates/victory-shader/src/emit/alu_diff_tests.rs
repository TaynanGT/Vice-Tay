//! Differential tests of the integer scalar and vector ALU lowering.
//!
//! Each case is a one-thread synthetic RDNA2 compute shader built around one
//! instruction, run on the host GPU and WARP for a deterministic set of
//! inputs (edge values and a seeded generator), and compared with a model of
//! the instruction written from AMD's RDNA2 ISA reference: the result and the
//! flags it leaves (SCC, VCC, a VOP3 carry-out). Integer results are exact,
//! so any difference is a translation error (or an error in the model, which
//! the case's note names).
//!
//! The SCC of `s_add_i32`, `s_sub_i32` and `s_addk_i32` is the signed
//! overflow, as the ISA's pseudocode has it (the machine-readable ISA's
//! one-line descriptions call it the carry-out).
//!
//! `VICTORY_ALU_DIFF_ROUNDS=<n>` runs `n` times as many random inputs.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

// ---------------------------------------------------------------------------
// Encodings (RDNA2 ISA, chapter 13: microcode formats).
// ---------------------------------------------------------------------------

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constants and special operands.
const ZERO: u16 = 128;
const ONE: u16 = 129;
const LITERAL: u16 = 255;
const VCC_LO: u16 = 106;
/// The registers the inputs are in: `s0..s3` and `v1..v4`.
const fn s(n: u16) -> u16 {
    n
}
const fn v(n: u16) -> u16 {
    256 + n
}

const fn sop2(op: u32, sdst: u32, src0: u16, src1: u16) -> u32 {
    0x8000_0000
        | (op & 0x7F) << 23
        | (sdst & 0x7F) << 16
        | (src1 as u32 & 0xFF) << 8
        | (src0 as u32 & 0xFF)
}
const fn sop1(op: u32, sdst: u32, src0: u16) -> u32 {
    0xBE80_0000 | (sdst & 0x7F) << 16 | (op & 0xFF) << 8 | (src0 as u32 & 0xFF)
}
const fn sopc(op: u32, src0: u16, src1: u16) -> u32 {
    0xBF00_0000 | (op & 0x7F) << 16 | (src1 as u32 & 0xFF) << 8 | (src0 as u32 & 0xFF)
}
const fn sopk(op: u32, sdst: u32, simm16: u16) -> u32 {
    0xB000_0000 | (op & 0x1F) << 23 | (sdst & 0x7F) << 16 | simm16 as u32
}
const fn vop1(op: u32, vdst: u32, src0: u16) -> u32 {
    0x7E00_0000 | (vdst & 0xFF) << 17 | (op & 0xFF) << 9 | (src0 as u32 & 0x1FF)
}
/// VOP2 `op vdst, src0, vsrc1` (`vsrc1` a VGPR number).
const fn vop2(op: u32, vdst: u32, src0: u16, vsrc1: u32) -> u32 {
    (op & 0x3F) << 25 | (vdst & 0xFF) << 17 | (vsrc1 & 0xFF) << 9 | (src0 as u32 & 0x1FF)
}
/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u32, src0: u16, vsrc1: u32) -> u32 {
    0x7C00_0000 | (op & 0xFF) << 17 | (vsrc1 & 0xFF) << 9 | (src0 as u32 & 0x1FF)
}
/// VOP3A (or VOP3B with `sdst` in bits 14:8) without modifiers.
const fn vop3(op: u32, vdst: u32, sdst: u32, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0xD400_0000 | (op & 0x3FF) << 16 | (sdst & 0x7F) << 8 | (vdst & 0xFF),
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}
/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0`.
const fn s_load_dwordx4(dst: u32, base: u32) -> [u32; 2] {
    [0x3D << 26 | 0x02 << 18 | dst << 6 | base >> 1, 125 << 25]
}
/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u32, srsrc: u32, byte_off: u32) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off & 0xFFF),
        0x80 << 24 | (srsrc >> 2 & 0x1F) << 16 | vdata << 8,
    ]
}

/// The outputs of a case: `v0`, `v5` and `v6`, stored at dwords 0, 1, 2.
const OUTS: [u32; 3] = [0, 5, 6];

/// A one-thread compute shader: 4 root constants in `s0..s3` (copied to
/// `v1..v4`), `body`, then [`OUTS`] stored. `None` with the stage that
/// refused it.
fn build(body: &[u32]) -> Result<HostShader, String> {
    build_for(body, 1)
}

/// [`build`] for groups of `threads` threads.
fn build_for(body: &[u32], threads: u32) -> Result<HostShader, String> {
    let mut words = vec![sop1(0x03, 5, ZERO)]; // s_mov_b32 s5, 0: the table at s[4:5]
    words.extend(s_load_dwordx4(8, 4));
    for i in 0..4u16 {
        words.push(vop1(0x01, u32::from(i) + 1, s(i))); // v_mov_b32 v[i+1], s[i]
    }
    // Outputs not written by the body read 0.
    words.push(vop1(0x01, 5, ZERO));
    words.push(vop1(0x01, 6, ZERO));
    words.extend_from_slice(body);
    for (k, &out) in OUTS.iter().enumerate() {
        words.extend(buffer_store_dword(out, 8, 4 * k as u32));
    }
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
            rsrc2: 5 << 1,
            threads: [threads, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)
        .map_err(|e| format!("ir::build: {e:?}"))?;
    let st = structure::structure(&p).map_err(|e| format!("structure: {e:?}"))?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))
        .map_err(|e| format!("resource: {e:?}"))?;
    let io = io::raise(&p, &native).map_err(|e| format!("io: {e:?}"))?;
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
    .map_err(|e| format!("lower: {e:?}"))?;
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
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy)
            .map_err(|e| format!("DXIL validation: {e}"))?;
    }
    Ok(shader)
}

// ---------------------------------------------------------------------------
// The cases.
// ---------------------------------------------------------------------------

/// What one output dword must be: the bits of `mask` equal those of the value.
type Want = [Option<(u32, u32)>; 3];

/// In a [`Want`]: the dword is a single-precision NaN, any of them.
const NAN32: (u32, u32) = (0, 0);
/// In a [`Want`]: the low half is a half-precision NaN, any of them.
const NAN16: (u32, u32) = (0, 1);

fn is_nan32(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

fn is_nan16(x: u32) -> bool {
    x & 0x7FFF > 0x7C00
}

/// Whether `got` is what `want` asks for.
fn meets(got: u32, want: (u32, u32)) -> bool {
    match want {
        NAN32 => is_nan32(got),
        NAN16 => is_nan16(got),
        (mask, value) => got & mask == value & mask,
    }
}

const ALL: u32 = u32::MAX;
const fn w1(r: u32) -> Want {
    [Some((ALL, r)), None, None]
}
const fn w2(r: u32, f: bool) -> Want {
    [Some((ALL, r)), Some((ALL, f as u32)), None]
}
/// A result and a lane-mask flag (bit 0: lane 0's).
const fn wlane(r: u32, f: bool) -> Want {
    [Some((ALL, r)), Some((1, f as u32)), None]
}
const fn w16(r: u32) -> Want {
    [Some((0xFFFF, r & 0xFFFF)), None, None]
}
fn w64(r: u64) -> Want {
    [Some((ALL, r as u32)), Some((ALL, (r >> 32) as u32)), None]
}
fn w64f(r: u64, f: bool) -> Want {
    [
        Some((ALL, r as u32)),
        Some((ALL, (r >> 32) as u32)),
        Some((ALL, f as u32)),
    ]
}

struct Case {
    name: String,
    body: Vec<u32>,
    model: Box<dyn Fn([u32; 4]) -> Want>,
    /// How the inputs are drawn.
    inputs: Inputs,
}

#[derive(Clone, Copy)]
enum Inputs {
    /// Any dwords.
    Any,
    /// `c` and `d` are 0 or 1 (a carry in).
    CarryIn,
    /// Single-precision floats: edge values (zeros, halves, the ends of the
    /// integer and half-precision ranges, infinities, NaNs) and normal
    /// numbers of every magnitude; no denormals.
    Float,
    /// Half-precision floats in the low halves (denormals included), and in
    /// the high halves.
    Half,
    /// A double in `a:b` (low dword first) and another in `c:d`.
    Double,
}

/// SCC from `s_cselect_b32 s17, 1, 0`, into `v5`.
fn scc_to_v5() -> [u32; 2] {
    [sop2(0x0A, 17, ONE, ZERO), vop1(0x01, 5, s(17))]
}

fn sop2_case(
    name: &str,
    op: u32,
    f: fn(u32, u32, bool) -> (u32, Option<bool>),
    inputs: Inputs,
) -> Case {
    // SCC in from d (s3 == 1), for the ops that read it.
    let mut body = vec![
        sopc(0x06, s(3), ONE),
        sop2(op, 16, s(0), s(1)),
        vop1(0x01, 0, s(16)),
    ];
    body.extend(scc_to_v5());
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, scc) = f(i[0], i[1], i[3] == 1);
            // An op that writes no SCC leaves the one set beforehand.
            w2(r, scc.unwrap_or(i[3] == 1))
        }),
        inputs,
    }
}

/// A SOP2 op with a literal or an inline constant as its second operand.
fn sop2_const_case(
    name: &str,
    op: u32,
    src1: u16,
    literal: Option<u32>,
    value: u32,
    f: fn(u32, u32, bool) -> (u32, Option<bool>),
) -> Case {
    let mut body = vec![sopc(0x06, s(3), ONE), sop2(op, 16, s(0), src1)];
    body.extend(literal);
    body.push(vop1(0x01, 0, s(16)));
    body.extend(scc_to_v5());
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, scc) = f(i[0], value, false);
            w2(r, scc.unwrap_or(i[3] == 1))
        }),
        inputs: Inputs::Any,
    }
}

fn sop2_64_case(name: &str, op: u32, shift: bool, f: fn(u64, u64) -> (u64, Option<bool>)) -> Case {
    // `s2` is the shift count, or the second operand's low half.
    let mut body = vec![
        sop2(op, 16, s(0), s(2)),
        vop1(0x01, 0, s(16)),
        vop1(0x01, 5, s(17)),
    ];
    body.extend([sop2(0x0A, 18, ONE, ZERO), vop1(0x01, 6, s(18))]);
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let a = u64::from(i[0]) | u64::from(i[1]) << 32;
            let b = if shift {
                u64::from(i[2])
            } else {
                u64::from(i[2]) | u64::from(i[3]) << 32
            };
            match f(a, b) {
                (r, Some(c)) => w64f(r, c),
                (r, None) => w64(r),
            }
        }),
        inputs: Inputs::Any,
    }
}

fn sop1_case(name: &str, op: u32, f: fn(u32, u32, bool) -> (u32, Option<bool>)) -> Case {
    // sdst starts as b (for the ops that read it), SCC as d == 1.
    let mut body = vec![
        sop1(0x03, 16, s(1)),
        sopc(0x06, s(3), ONE),
        sop1(op, 16, s(0)),
        vop1(0x01, 0, s(16)),
    ];
    body.extend(scc_to_v5());
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, scc) = f(i[0], i[1], i[3] == 1);
            w2(r, scc.unwrap_or(i[3] == 1))
        }),
        inputs: Inputs::Any,
    }
}

/// A SOP1 op of 64-bit source `s[0:1]`; its result is 64 bits wide (`wide`)
/// or 32.
fn sop1_64_case(name: &str, op: u32, wide: bool, f: fn(u64) -> (u64, Option<bool>)) -> Case {
    let mut body = vec![
        sopc(0x06, s(3), ONE),
        sop1(op, 16, s(0)),
        vop1(0x01, 0, s(16)),
    ];
    if wide {
        body.push(vop1(0x01, 5, s(17)));
    }
    body.extend([sop2(0x0A, 18, ONE, ZERO), vop1(0x01, 6, s(18))]);
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, scc) = f(u64::from(i[0]) | u64::from(i[1]) << 32);
            let mut want = if wide { w64(r) } else { w1(r as u32) };
            want[2] = Some((ALL, scc.unwrap_or(i[3] == 1) as u32));
            want
        }),
        inputs: Inputs::Any,
    }
}

fn sopc_case(name: &str, op: u32, wide: bool, f: fn(u64, u64) -> bool) -> Case {
    let (a, b) = if wide { (s(0), s(2)) } else { (s(0), s(1)) };
    // A flag that is set beforehand, so that a compare that leaves SCC alone shows.
    let mut body = vec![sopc(0x06, s(3), s(3)), sopc(op, a, b)];
    body.extend(scc_to_v5());
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (a, b) = if wide {
                (
                    u64::from(i[0]) | u64::from(i[1]) << 32,
                    u64::from(i[2]) | u64::from(i[3]) << 32,
                )
            } else {
                (u64::from(i[0]), u64::from(i[1]))
            };
            [None, Some((ALL, f(a, b) as u32)), None]
        }),
        inputs: Inputs::Any,
    }
}

fn sopk_case(name: &str, op: u32, imm: u16, f: fn(u32, u16) -> (u32, Option<bool>)) -> Case {
    let mut body = vec![
        sop1(0x03, 16, s(0)),
        sopc(0x06, s(3), ONE),
        sopk(op, 16, imm),
        vop1(0x01, 0, s(16)),
    ];
    body.extend(scc_to_v5());
    Case {
        name: format!("{name} {imm:#06x}"),
        body,
        model: Box::new(move |i| {
            let (r, scc) = f(i[0], imm);
            w2(r, scc.unwrap_or(i[3] == 1))
        }),
        inputs: Inputs::Any,
    }
}

/// How the first operand of a vector op is given.
#[derive(Clone, Copy)]
enum Src0 {
    Vgpr,
    Sgpr,
    /// A literal (`a` of the first input set is used for all of them).
    Literal(u32),
}

impl Src0 {
    fn label(self) -> &'static str {
        match self {
            Src0::Vgpr => "v",
            Src0::Sgpr => "s",
            Src0::Literal(_) => "lit",
        }
    }
    fn operand(self) -> u16 {
        match self {
            Src0::Vgpr => v(1),
            Src0::Sgpr => s(0),
            Src0::Literal(_) => LITERAL,
        }
    }
    fn a(self, i: [u32; 4]) -> u32 {
        match self {
            Src0::Literal(k) => k,
            _ => i[0],
        }
    }
    fn literal(self) -> Option<u32> {
        match self {
            Src0::Literal(k) => Some(k),
            _ => None,
        }
    }
}

fn vop2_case(name: &str, op: u32, src0: Src0, f: fn(u32, u32) -> u32) -> Case {
    let mut body = vec![vop2(op, 0, src0.operand(), 2)];
    body.extend(src0.literal());
    Case {
        name: format!("{name} ({})", src0.label()),
        body,
        model: Box::new(move |i| w1(f(src0.a(i), i[1]))),
        inputs: Inputs::Any,
    }
}

/// A VOP2 op in its VOP3 encoding (`op + 0x100`), with `src1` from an SGPR.
fn vop2_as3_case(name: &str, op: u32, f: fn(u32, u32) -> u32) -> Case {
    let body = vop3(0x100 + op, 0, 0, v(1), s(1), ZERO).to_vec();
    Case {
        name: format!("{name} (vop3)"),
        body,
        model: Box::new(move |i| w1(f(i[0], i[1]))),
        inputs: Inputs::Any,
    }
}

/// A VOP2 op that reads VCC as a carry in (`c != 0`) and writes it.
fn vop2_carry_case(name: &str, op: u32, f: fn(u32, u32, bool) -> (u32, bool)) -> Case {
    let body = vec![
        vopc(0xC5, ZERO, 3),    // v_cmp_ne_u32 vcc, 0, v3
        vop2(op, 0, v(1), 2),   // op v0, vcc, v1, v2, vcc
        vop1(0x01, 7, ONE),     // v7 = 1
        vop2(0x01, 5, ZERO, 7), // v_cndmask_b32 v5, 0, v7, vcc
    ];
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, c) = f(i[0], i[1], i[2] != 0);
            w2(r, c)
        }),
        inputs: Inputs::CarryIn,
    }
}

fn vop1_case(name: &str, op: u32, src0: Src0, f: fn(u32) -> u32) -> Case {
    let mut body = vec![vop1(op, 0, src0.operand())];
    body.extend(src0.literal());
    Case {
        name: format!("{name} ({})", src0.label()),
        body,
        model: Box::new(move |i| w1(f(src0.a(i)))),
        inputs: Inputs::Any,
    }
}

fn vop3_case(name: &str, op: u32, f: fn(u32, u32, u32) -> u32) -> Case {
    let body = vop3(op, 0, 0, v(1), v(2), v(3)).to_vec();
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| w1(f(i[0], i[1], i[2]))),
        inputs: Inputs::Any,
    }
}

/// A VOP3 op with SGPR operands, and an inline constant for `src2`.
fn vop3_sgpr_case(name: &str, op: u32, f: fn(u32, u32, u32) -> u32) -> Case {
    let body = vop3(op, 0, 0, s(0), v(2), ONE).to_vec();
    Case {
        name: format!("{name} (s, v, 1)"),
        body,
        model: Box::new(move |i| w1(f(i[0], i[1], 1))),
        inputs: Inputs::Any,
    }
}

fn vop3_16_case(name: &str, op: u32, f: fn(u16, u16) -> u16) -> Case {
    let body = vop3(op, 0, 0, v(1), v(2), ZERO).to_vec();
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| w16(u32::from(f(i[0] as u16, i[1] as u16)))),
        inputs: Inputs::Any,
    }
}

/// A VOP3B add or subtract: the carry-out to `s20`, read as a lane mask.
fn vop3b_case(name: &str, op: u32, f: fn(u32, u32) -> (u32, bool)) -> Case {
    let body = [
        vop3(op, 0, 20, v(1), v(2), ZERO).as_slice(),
        &[vop1(0x01, 5, s(20))],
    ]
    .concat();
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (r, c) = f(i[0], i[1]);
            wlane(r, c)
        }),
        inputs: Inputs::Any,
    }
}

/// A 64-bit vector shift: `op v[6:7], v3, v[1:2]`.
fn vop3_shift64_case(name: &str, op: u32, f: fn(u64, u32) -> u64) -> Case {
    let body = [
        vop3(op, 6, 0, v(3), v(1), ZERO).as_slice(),
        &[
            vop1(0x01, 0, v(6)),
            vop1(0x01, 5, v(7)),
            vop1(0x01, 6, ZERO),
        ],
    ]
    .concat();
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| w64(f(u64::from(i[0]) | u64::from(i[1]) << 32, i[2]))),
        inputs: Inputs::Any,
    }
}

/// `v_mad_u64_u32 v[6:7], s20, v1, v2, v[3:4]`.
fn mad_u64_case() -> Case {
    let body = [
        vop3(0x176, 6, 20, v(1), v(2), v(3)).as_slice(),
        &[
            vop1(0x01, 0, v(6)),
            vop1(0x01, 5, v(7)),
            vop1(0x01, 6, s(20)),
        ],
    ]
    .concat();
    Case {
        name: "v_mad_u64_u32".into(),
        body,
        model: Box::new(|i| {
            let c = u64::from(i[2]) | u64::from(i[3]) << 32;
            let (r, carry) = (u64::from(i[0]) * u64::from(i[1])).overflowing_add(c);
            let mut want = w64(r);
            want[2] = Some((1, carry as u32));
            want
        }),
        inputs: Inputs::Any,
    }
}

/// A VOPC compare into VCC (`e32`), or into `s20` (`e64`); 64-bit compares
/// read `v[1:2]` and `v[3:4]`.
fn vopc_case(name: &str, op: u32, bits: u32, e64: bool, f: fn(u64, u64, u32) -> bool) -> Case {
    let (a, b) = if bits == 64 { (v(1), 3) } else { (v(1), 2) };
    let mut body = if e64 {
        vop3(op, 20, 0, a, v(b as u16), ZERO).to_vec()
    } else {
        vec![vopc(op, a, b)]
    };
    body.push(vop1(0x01, 5, if e64 { s(20) } else { VCC_LO }));
    Case {
        name: format!("{name}{}", if e64 { " (e64)" } else { "" }),
        body,
        model: Box::new(move |i| {
            let (a, b) = if bits == 64 {
                (
                    u64::from(i[0]) | u64::from(i[1]) << 32,
                    u64::from(i[2]) | u64::from(i[3]) << 32,
                )
            } else {
                (u64::from(i[0]), u64::from(i[1]))
            };
            [None, Some((1, f(a, b, bits) as u32)), None]
        }),
        inputs: Inputs::Any,
    }
}

/// A VOPC `v_cmpx_*`: EXEC narrowed to lanes that pass, so only they set `v0` to 1.
fn cmpx_case(name: &str, op: u32, bits: u32, f: fn(u64, u64, u32) -> bool) -> Case {
    let (a, b) = if bits == 64 { (v(1), 3) } else { (v(1), 2) };
    let body = vec![vop1(0x01, 0, ZERO), vopc(op, a, b), vop1(0x01, 0, ONE)];
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let (a, b) = if bits == 64 {
                (
                    u64::from(i[0]) | u64::from(i[1]) << 32,
                    u64::from(i[2]) | u64::from(i[3]) << 32,
                )
            } else {
                (u64::from(i[0]), u64::from(i[1]))
            };
            w1(f(a, b, bits) as u32)
        }),
        inputs: Inputs::Any,
    }
}

// The models. Names and operand orders follow the ISA reference.

fn bfe(x: u32, off: u32, width: u32, signed: bool) -> u32 {
    // As the hardware does it (and LLVM's constant folding of BFE): a field
    // that reaches bit 31 is the source shifted right, arithmetically for the
    // signed forms.
    if width == 0 {
        return 0;
    }
    if off + width < 32 {
        let field = (x >> off) & ((1u32 << width) - 1);
        return if signed && field >> (width - 1) & 1 != 0 {
            field | !((1u32 << width) - 1)
        } else {
            field
        };
    }
    if signed {
        ((x as i32) >> off) as u32
    } else {
        x >> off
    }
}

fn sad_bytes(a: u32, b: u32, masked: bool) -> u32 {
    (0..4)
        .map(|k| {
            let (x, y) = ((a >> (8 * k)) & 0xFF, (b >> (8 * k)) & 0xFF);
            if masked && y == 0 { 0 } else { x.abs_diff(y) }
        })
        .sum()
}

fn perm(a: u32, b: u32, sel: u32) -> u32 {
    let both = u64::from(a) << 32 | u64::from(b);
    (0..4)
        .map(|k| {
            let s = (sel >> (8 * k)) & 0xFF;
            let byte = match s {
                0..=7 => (both >> (8 * s)) as u32 & 0xFF,
                8 => {
                    if b >> 15 & 1 != 0 {
                        0xFF
                    } else {
                        0
                    }
                }
                9 => {
                    if b >> 31 & 1 != 0 {
                        0xFF
                    } else {
                        0
                    }
                }
                10 => {
                    if a >> 15 & 1 != 0 {
                        0xFF
                    } else {
                        0
                    }
                }
                11 => {
                    if a >> 31 & 1 != 0 {
                        0xFF
                    } else {
                        0
                    }
                }
                12 => 0,
                _ => 0xFF,
            };
            byte << (8 * k)
        })
        .fold(0, |m, b| m | b)
}

fn sext24(x: u32) -> i64 {
    i64::from(((x << 8) as i32) >> 8)
}

/// The leading bits that are the same as the sign bit (the sign bit among
/// them); -1 if all are.
fn flbit_signed(x: u32) -> u32 {
    let y = if (x as i32) < 0 { !x } else { x };
    if y == 0 { u32::MAX } else { y.leading_zeros() }
}

fn cmp(kind: &str, a: u64, b: u64, bits: u32) -> bool {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let (ua, ub) = (a & mask, b & mask);
    let sign = |x: u64| -> i64 { ((x << (64 - bits)) as i64) >> (64 - bits) };
    let (sa, sb) = (sign(ua), sign(ub));
    match kind {
        "f" => false,
        "t" => true,
        "lt_i" => sa < sb,
        "eq_i" | "eq_u" => ua == ub,
        "le_i" => sa <= sb,
        "gt_i" => sa > sb,
        "ne_i" | "ne_u" => ua != ub,
        "ge_i" => sa >= sb,
        "lt_u" => ua < ub,
        "le_u" => ua <= ub,
        "gt_u" => ua > ub,
        "ge_u" => ua >= ub,
        _ => unreachable!("{kind}"),
    }
}

macro_rules! cmp_fn {
    ($kind:literal) => {{
        fn f(a: u64, b: u64, bits: u32) -> bool {
            cmp($kind, a, b, bits)
        }
        f
    }};
}

fn cases() -> Vec<Case> {
    let mut c = Vec::new();

    // SOP2, 32 bits.
    c.push(sop2_case(
        "s_add_u32",
        0x00,
        |a, b, _| {
            let (r, o) = a.overflowing_add(b);
            (r, Some(o))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_sub_u32",
        0x01,
        |a, b, _| {
            let (r, o) = a.overflowing_sub(b);
            (r, Some(o))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_add_i32",
        0x02,
        |a, b, _| {
            let (r, o) = (a as i32).overflowing_add(b as i32);
            (r as u32, Some(o))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_sub_i32",
        0x03,
        |a, b, _| {
            let (r, o) = (a as i32).overflowing_sub(b as i32);
            (r as u32, Some(o))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_addc_u32",
        0x04,
        |a, b, c| {
            let r = u64::from(a) + u64::from(b) + u64::from(c);
            (r as u32, Some(r >> 32 != 0))
        },
        Inputs::CarryIn,
    ));
    c.push(sop2_case(
        "s_subb_u32",
        0x05,
        |a, b, c| {
            let r = u64::from(a)
                .wrapping_sub(u64::from(b))
                .wrapping_sub(u64::from(c));
            (r as u32, Some(u64::from(b) + u64::from(c) > u64::from(a)))
        },
        Inputs::CarryIn,
    ));
    c.push(sop2_case(
        "s_min_i32",
        0x06,
        |a, b, _| {
            (
                (a as i32).min(b as i32) as u32,
                Some((a as i32) < (b as i32)),
            )
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_min_u32",
        0x07,
        |a, b, _| (a.min(b), Some(a < b)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_max_i32",
        0x08,
        |a, b, _| {
            (
                (a as i32).max(b as i32) as u32,
                Some((a as i32) > (b as i32)),
            )
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_max_u32",
        0x09,
        |a, b, _| (a.max(b), Some(a > b)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_cselect_b32",
        0x0A,
        |a, b, c| (if c { a } else { b }, Some(c)),
        Inputs::CarryIn,
    ));
    c.push(sop2_case(
        "s_and_b32",
        0x0E,
        |a, b, _| (a & b, Some(a & b != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_or_b32",
        0x10,
        |a, b, _| (a | b, Some(a | b != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_xor_b32",
        0x12,
        |a, b, _| (a ^ b, Some(a ^ b != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_andn2_b32",
        0x14,
        |a, b, _| (a & !b, Some(a & !b != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_orn2_b32",
        0x16,
        |a, b, _| (a | !b, Some(a | !b != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_nand_b32",
        0x18,
        |a, b, _| (!(a & b), Some(!(a & b) != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_nor_b32",
        0x1A,
        |a, b, _| (!(a | b), Some(!(a | b) != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_xnor_b32",
        0x1C,
        |a, b, _| (!(a ^ b), Some(!(a ^ b) != 0)),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_lshl_b32",
        0x1E,
        |a, b, _| {
            let r = a << (b & 31);
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_lshr_b32",
        0x20,
        |a, b, _| {
            let r = a >> (b & 31);
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_ashr_i32",
        0x22,
        |a, b, _| {
            let r = ((a as i32) >> (b & 31)) as u32;
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_bfm_b32",
        0x24,
        |a, b, _| ((((1u64 << (a & 31)) - 1) as u32) << (b & 31), None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_mul_i32",
        0x26,
        |a, b, _| (a.wrapping_mul(b), None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_bfe_u32",
        0x27,
        |a, b, _| {
            let r = bfe(a, b & 31, (b >> 16) & 0x7F, false);
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_bfe_i32",
        0x28,
        |a, b, _| {
            let r = bfe(a, b & 31, (b >> 16) & 0x7F, true);
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_absdiff_i32",
        0x2C,
        |a, b, _| {
            let r = (i64::from(a as i32) - i64::from(b as i32)).unsigned_abs() as u32;
            (r, Some(r != 0))
        },
        Inputs::Any,
    ));
    for (k, op) in [(1u32, 0x2E), (2, 0x2F), (3, 0x30), (4, 0x31)] {
        let f: fn(u32, u32, bool) -> (u32, Option<bool>) = match k {
            1 => |a, b, _| {
                let r = (u64::from(a) << 1) + u64::from(b);
                (r as u32, Some(r >> 32 != 0))
            },
            2 => |a, b, _| {
                let r = (u64::from(a) << 2) + u64::from(b);
                (r as u32, Some(r >> 32 != 0))
            },
            3 => |a, b, _| {
                let r = (u64::from(a) << 3) + u64::from(b);
                (r as u32, Some(r >> 32 != 0))
            },
            _ => |a, b, _| {
                let r = (u64::from(a) << 4) + u64::from(b);
                (r as u32, Some(r >> 32 != 0))
            },
        };
        c.push(sop2_case(&format!("s_lshl{k}_add_u32"), op, f, Inputs::Any));
    }
    c.push(sop2_case(
        "s_pack_ll_b32_b16",
        0x32,
        |a, b, _| (a & 0xFFFF | b << 16, None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_pack_lh_b32_b16",
        0x33,
        |a, b, _| (a & 0xFFFF | b & 0xFFFF_0000, None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_pack_hh_b32_b16",
        0x34,
        |a, b, _| (a >> 16 | b & 0xFFFF_0000, None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_mul_hi_u32",
        0x35,
        |a, b, _| (((u64::from(a) * u64::from(b)) >> 32) as u32, None),
        Inputs::Any,
    ));
    c.push(sop2_case(
        "s_mul_hi_i32",
        0x36,
        |a, b, _| {
            (
                ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
                None,
            )
        },
        Inputs::Any,
    ));
    // The second operand as an inline constant and as a literal.
    c.push(sop2_const_case(
        "s_add_u32 (inline 1)",
        0x00,
        ONE,
        None,
        1,
        |a, b, _| {
            let (r, o) = a.overflowing_add(b);
            (r, Some(o))
        },
    ));
    c.push(sop2_const_case(
        "s_add_u32 (inline -1)",
        0x00,
        193,
        None,
        u32::MAX,
        |a, b, _| {
            let (r, o) = a.overflowing_add(b);
            (r, Some(o))
        },
    ));
    c.push(sop2_const_case(
        "s_and_b32 (literal)",
        0x0E,
        LITERAL,
        Some(0x00FF_F00F),
        0x00FF_F00F,
        |a, b, _| (a & b, Some(a & b != 0)),
    ));
    c.push(sop2_const_case(
        "s_bfe_u32 (literal 8@4)",
        0x27,
        LITERAL,
        Some(0x0008_0004),
        0x0008_0004,
        |a, b, _| {
            let r = bfe(a, b & 31, (b >> 16) & 0x7F, false);
            (r, Some(r != 0))
        },
    ));
    c.push(sop2_const_case(
        "s_bfe_i32 (literal 12@20)",
        0x28,
        LITERAL,
        Some(0x000C_0014),
        0x000C_0014,
        |a, b, _| {
            let r = bfe(a, b & 31, (b >> 16) & 0x7F, true);
            (r, Some(r != 0))
        },
    ));
    c.push(sop2_const_case(
        "s_lshl_b32 (inline 4)",
        0x1E,
        132,
        None,
        4,
        |a, b, _| {
            let r = a << (b & 31);
            (r, Some(r != 0))
        },
    ));

    // SOP2, 64 bits.
    c.push(sop2_64_case("s_and_b64", 0x0F, false, |a, b| {
        (a & b, Some(a & b != 0))
    }));
    c.push(sop2_64_case("s_or_b64", 0x11, false, |a, b| {
        (a | b, Some(a | b != 0))
    }));
    c.push(sop2_64_case("s_xor_b64", 0x13, false, |a, b| {
        (a ^ b, Some(a ^ b != 0))
    }));
    c.push(sop2_64_case("s_andn2_b64", 0x15, false, |a, b| {
        (a & !b, Some(a & !b != 0))
    }));
    c.push(sop2_64_case("s_orn2_b64", 0x17, false, |a, b| {
        (a | !b, Some(a | !b != 0))
    }));
    c.push(sop2_64_case("s_nand_b64", 0x19, false, |a, b| {
        (!(a & b), Some(!(a & b) != 0))
    }));
    c.push(sop2_64_case("s_nor_b64", 0x1B, false, |a, b| {
        (!(a | b), Some(!(a | b) != 0))
    }));
    c.push(sop2_64_case("s_xnor_b64", 0x1D, false, |a, b| {
        (!(a ^ b), Some(!(a ^ b) != 0))
    }));
    c.push(sop2_64_case("s_lshl_b64", 0x1F, true, |a, b| {
        let r = a << (b & 63);
        (r, Some(r != 0))
    }));
    c.push(sop2_64_case("s_lshr_b64", 0x21, true, |a, b| {
        let r = a >> (b & 63);
        (r, Some(r != 0))
    }));
    c.push(sop2_64_case("s_ashr_i64", 0x23, true, |a, b| {
        let r = ((a as i64) >> (b & 63)) as u64;
        (r, Some(r != 0))
    }));

    // SOP1.
    c.push(sop1_case("s_mov_b32", 0x03, |a, _, _| (a, None)));
    c.push(sop1_case("s_cmov_b32", 0x05, |a, b, c| {
        (if c { a } else { b }, None)
    }));
    c.push(sop1_case("s_not_b32", 0x07, |a, _, _| (!a, Some(!a != 0))));
    c.push(sop1_case("s_brev_b32", 0x0B, |a, _, _| {
        (a.reverse_bits(), None)
    }));
    c.push(sop1_case("s_bcnt0_i32_b32", 0x0D, |a, _, _| {
        (a.count_zeros(), Some(a.count_zeros() != 0))
    }));
    c.push(sop1_case("s_bcnt1_i32_b32", 0x0F, |a, _, _| {
        (a.count_ones(), Some(a.count_ones() != 0))
    }));
    c.push(sop1_case("s_ff0_i32_b32", 0x11, |a, _, _| {
        (
            if a == u32::MAX {
                u32::MAX
            } else {
                a.trailing_ones()
            },
            None,
        )
    }));
    c.push(sop1_case("s_ff1_i32_b32", 0x13, |a, _, _| {
        (if a == 0 { u32::MAX } else { a.trailing_zeros() }, None)
    }));
    c.push(sop1_case("s_flbit_i32_b32", 0x15, |a, _, _| {
        (if a == 0 { u32::MAX } else { a.leading_zeros() }, None)
    }));
    c.push(sop1_case("s_flbit_i32", 0x17, |a, _, _| {
        (flbit_signed(a), None)
    }));
    c.push(sop1_case("s_sext_i32_i8", 0x19, |a, _, _| {
        (a as u8 as i8 as i32 as u32, None)
    }));
    c.push(sop1_case("s_sext_i32_i16", 0x1A, |a, _, _| {
        (a as u16 as i16 as i32 as u32, None)
    }));
    c.push(sop1_case("s_bitset0_b32", 0x1B, |a, b, _| {
        (b & !(1 << (a & 31)), None)
    }));
    c.push(sop1_case("s_bitset1_b32", 0x1D, |a, b, _| {
        (b | 1 << (a & 31), None)
    }));
    c.push(sop1_case("s_abs_i32", 0x34, |a, _, _| {
        let r = (a as i32).unsigned_abs();
        (r, Some(r != 0))
    }));
    c.push(sop1_64_case("s_mov_b64", 0x04, true, |a| (a, None)));
    c.push(sop1_64_case("s_not_b64", 0x08, true, |a| {
        (!a, Some(!a != 0))
    }));
    c.push(sop1_64_case("s_brev_b64", 0x0C, true, |a| {
        (a.reverse_bits(), None)
    }));
    c.push(sop1_64_case("s_bcnt1_i32_b64", 0x10, false, |a| {
        (u64::from(a.count_ones()), Some(a != 0))
    }));
    c.push(sop1_64_case("s_ff1_i32_b64", 0x14, false, |a| {
        (
            if a == 0 {
                u64::from(u32::MAX)
            } else {
                u64::from(a.trailing_zeros())
            },
            None,
        )
    }));
    c.push(sop1_64_case("s_flbit_i32_b64", 0x16, false, |a| {
        (
            if a == 0 {
                u64::from(u32::MAX)
            } else {
                u64::from(a.leading_zeros())
            },
            None,
        )
    }));

    // SOPC.
    let sopcs: [(&str, u32, fn(u64, u64) -> bool); 12] = [
        ("s_cmp_eq_i32", 0x00, |a, b| a == b),
        ("s_cmp_lg_i32", 0x01, |a, b| a != b),
        ("s_cmp_gt_i32", 0x02, |a, b| {
            (a as u32 as i32) > (b as u32 as i32)
        }),
        ("s_cmp_ge_i32", 0x03, |a, b| {
            (a as u32 as i32) >= (b as u32 as i32)
        }),
        ("s_cmp_lt_i32", 0x04, |a, b| {
            (a as u32 as i32) < (b as u32 as i32)
        }),
        ("s_cmp_le_i32", 0x05, |a, b| {
            (a as u32 as i32) <= (b as u32 as i32)
        }),
        ("s_cmp_eq_u32", 0x06, |a, b| a == b),
        ("s_cmp_lg_u32", 0x07, |a, b| a != b),
        ("s_cmp_gt_u32", 0x08, |a, b| a > b),
        ("s_cmp_ge_u32", 0x09, |a, b| a >= b),
        ("s_cmp_lt_u32", 0x0A, |a, b| a < b),
        ("s_cmp_le_u32", 0x0B, |a, b| a <= b),
    ];
    for (name, op, f) in sopcs {
        c.push(sopc_case(name, op, false, f));
    }
    c.push(sopc_case("s_bitcmp0_b32", 0x0C, false, |a, b| {
        a >> (b & 31) & 1 == 0
    }));
    c.push(sopc_case("s_bitcmp1_b32", 0x0D, false, |a, b| {
        a >> (b & 31) & 1 == 1
    }));
    c.push(sopc_case("s_cmp_eq_u64", 0x12, true, |a, b| a == b));
    c.push(sopc_case("s_cmp_lg_u64", 0x13, true, |a, b| a != b));

    // SOPK.
    for imm in [1u16, 0x7FFF, 0x8000, 0xFFFF] {
        c.push(sopk_case("s_addk_i32", 0x0F, imm, |a, k| {
            let (r, o) = (a as i32).overflowing_add(i32::from(k as i16));
            (r as u32, Some(o))
        }));
        c.push(sopk_case("s_mulk_i32", 0x10, imm, |a, k| {
            ((a as i32).wrapping_mul(i32::from(k as i16)) as u32, None)
        }));
    }

    // VOP2.
    type F2 = fn(u32, u32) -> u32;
    let vop2s: [(&str, u32, F2); 18] = [
        ("v_mul_i32_i24", 0x09, |a, b| (sext24(a) * sext24(b)) as u32),
        ("v_mul_hi_i32_i24", 0x0A, |a, b| {
            ((sext24(a) * sext24(b)) >> 32) as u32
        }),
        ("v_mul_u32_u24", 0x0B, |a, b| {
            (u64::from(a & 0xFF_FFFF) * u64::from(b & 0xFF_FFFF)) as u32
        }),
        ("v_mul_hi_u32_u24", 0x0C, |a, b| {
            ((u64::from(a & 0xFF_FFFF) * u64::from(b & 0xFF_FFFF)) >> 32) as u32
        }),
        ("v_min_i32", 0x11, |a, b| (a as i32).min(b as i32) as u32),
        ("v_max_i32", 0x12, |a, b| (a as i32).max(b as i32) as u32),
        ("v_min_u32", 0x13, |a, b| a.min(b)),
        ("v_max_u32", 0x14, |a, b| a.max(b)),
        ("v_lshrrev_b32", 0x16, |a, b| b >> (a & 31)),
        ("v_ashrrev_i32", 0x18, |a, b| {
            ((b as i32) >> (a & 31)) as u32
        }),
        ("v_lshlrev_b32", 0x1A, |a, b| b << (a & 31)),
        ("v_and_b32", 0x1B, |a, b| a & b),
        ("v_or_b32", 0x1C, |a, b| a | b),
        ("v_xor_b32", 0x1D, |a, b| a ^ b),
        ("v_xnor_b32", 0x1E, |a, b| !(a ^ b)),
        ("v_add_nc_u32", 0x25, |a, b| a.wrapping_add(b)),
        ("v_sub_nc_u32", 0x26, |a, b| a.wrapping_sub(b)),
        ("v_subrev_nc_u32", 0x27, |a, b| b.wrapping_sub(a)),
    ];
    for (name, op, f) in vop2s {
        c.push(vop2_case(name, op, Src0::Vgpr, f));
        c.push(vop2_case(name, op, Src0::Sgpr, f));
        c.push(vop2_case(name, op, Src0::Literal(0x8000_7F81), f));
        c.push(vop2_as3_case(name, op, f));
    }
    c.push(vop2_carry_case("v_add_co_ci_u32", 0x28, |a, b, c| {
        let r = u64::from(a) + u64::from(b) + u64::from(c);
        (r as u32, r >> 32 != 0)
    }));
    c.push(vop2_carry_case("v_sub_co_ci_u32", 0x29, |a, b, c| {
        let r = u64::from(a).wrapping_sub(u64::from(b) + u64::from(c));
        (r as u32, u64::from(b) + u64::from(c) > u64::from(a))
    }));
    c.push(vop2_carry_case("v_subrev_co_ci_u32", 0x2A, |a, b, c| {
        let r = u64::from(b).wrapping_sub(u64::from(a) + u64::from(c));
        (r as u32, u64::from(a) + u64::from(c) > u64::from(b))
    }));

    // VOP1.
    type F1 = fn(u32) -> u32;
    let vop1s: [(&str, u32, F1); 6] = [
        ("v_mov_b32", 0x01, |a| a),
        ("v_not_b32", 0x37, |a| !a),
        ("v_bfrev_b32", 0x38, |a| a.reverse_bits()),
        ("v_ffbh_u32", 0x39, |a| {
            if a == 0 { u32::MAX } else { a.leading_zeros() }
        }),
        ("v_ffbl_b32", 0x3A, |a| {
            if a == 0 { u32::MAX } else { a.trailing_zeros() }
        }),
        ("v_ffbh_i32", 0x3B, flbit_signed),
    ];
    for (name, op, f) in vop1s {
        c.push(vop1_case(name, op, Src0::Vgpr, f));
        c.push(vop1_case(name, op, Src0::Sgpr, f));
        c.push(vop1_case(name, op, Src0::Literal(0x0001_8000), f));
    }

    // VOP3.
    type F3 = fn(u32, u32, u32) -> u32;
    let vop3s: [(&str, u32, F3); 33] = [
        ("v_mad_i32_i24", 0x142, |a, b, c| {
            ((sext24(a) * sext24(b)) as u32).wrapping_add(c)
        }),
        ("v_mad_u32_u24", 0x143, |a, b, c| {
            ((u64::from(a & 0xFF_FFFF) * u64::from(b & 0xFF_FFFF)) as u32).wrapping_add(c)
        }),
        ("v_bfe_u32", 0x148, |a, b, c| bfe(a, b & 31, c & 31, false)),
        ("v_bfe_i32", 0x149, |a, b, c| bfe(a, b & 31, c & 31, true)),
        ("v_bfi_b32", 0x14A, |a, b, c| (a & b) | (!a & c)),
        ("v_alignbit_b32", 0x14E, |a, b, c| {
            ((u64::from(a) << 32 | u64::from(b)) >> (c & 31)) as u32
        }),
        ("v_alignbyte_b32", 0x14F, |a, b, c| {
            ((u64::from(a) << 32 | u64::from(b)) >> (8 * (c & 3))) as u32
        }),
        ("v_min3_i32", 0x152, |a, b, c| {
            (a as i32).min(b as i32).min(c as i32) as u32
        }),
        ("v_min3_u32", 0x153, |a, b, c| a.min(b).min(c)),
        ("v_max3_i32", 0x155, |a, b, c| {
            (a as i32).max(b as i32).max(c as i32) as u32
        }),
        ("v_max3_u32", 0x156, |a, b, c| a.max(b).max(c)),
        ("v_med3_i32", 0x158, |a, b, c| {
            let mut v = [a as i32, b as i32, c as i32];
            v.sort_unstable();
            v[1] as u32
        }),
        ("v_med3_u32", 0x159, |a, b, c| {
            let mut v = [a, b, c];
            v.sort_unstable();
            v[1]
        }),
        ("v_sad_u8", 0x15A, |a, b, c| {
            sad_bytes(a, b, false).wrapping_add(c)
        }),
        ("v_sad_hi_u8", 0x15B, |a, b, c| {
            (sad_bytes(a, b, false) << 16).wrapping_add(c)
        }),
        ("v_sad_u16", 0x15C, |a, b, c| {
            ((a & 0xFFFF).abs_diff(b & 0xFFFF) + (a >> 16).abs_diff(b >> 16)).wrapping_add(c)
        }),
        ("v_sad_u32", 0x15D, |a, b, c| a.abs_diff(b).wrapping_add(c)),
        ("v_mul_lo_u32", 0x169, |a, b, _| a.wrapping_mul(b)),
        ("v_mul_hi_u32", 0x16A, |a, b, _| {
            ((u64::from(a) * u64::from(b)) >> 32) as u32
        }),
        ("v_mul_lo_i32", 0x16B, |a, b, _| a.wrapping_mul(b)),
        ("v_mul_hi_i32", 0x16C, |a, b, _| {
            ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32
        }),
        ("v_msad_u8", 0x171, |a, b, c| {
            sad_bytes(a, b, true).wrapping_add(c)
        }),
        ("v_xor3_b32", 0x178, |a, b, c| a ^ b ^ c),
        ("v_perm_b32", 0x344, perm),
        ("v_xad_u32", 0x345, |a, b, c| (a ^ b).wrapping_add(c)),
        ("v_lshl_add_u32", 0x346, |a, b, c| {
            (a << (b & 31)).wrapping_add(c)
        }),
        ("v_add_lshl_u32", 0x347, |a, b, c| {
            a.wrapping_add(b) << (c & 31)
        }),
        ("v_bfm_b32", 0x363, |a, b, _| {
            (((1u64 << (a & 31)) - 1) as u32) << (b & 31)
        }),
        ("v_bcnt_u32_b32", 0x364, |a, b, _| {
            a.count_ones().wrapping_add(b)
        }),
        ("v_add3_u32", 0x36D, |a, b, c| {
            a.wrapping_add(b).wrapping_add(c)
        }),
        ("v_lshl_or_b32", 0x36F, |a, b, c| (a << (b & 31)) | c),
        ("v_and_or_b32", 0x371, |a, b, c| (a & b) | c),
        ("v_or3_b32", 0x372, |a, b, c| a | b | c),
    ];
    for (name, op, f) in vop3s {
        c.push(vop3_case(name, op, f));
        c.push(vop3_sgpr_case(name, op, f));
    }
    c.push(vop3_case("v_add_nc_i32", 0x37F, |a, b, _| {
        a.wrapping_add(b)
    }));
    c.push(vop3_case("v_sub_nc_i32", 0x376, |a, b, _| {
        a.wrapping_sub(b)
    }));
    c.push(vop3_case("v_mad_u32_u16", 0x373, |a, b, c| {
        ((a & 0xFFFF) * (b & 0xFFFF)).wrapping_add(c)
    }));
    c.push(vop3_case("v_mad_i32_i16", 0x375, |a, b, c| {
        ((i32::from(a as i16) * i32::from(b as i16)) as u32).wrapping_add(c)
    }));
    c.push(vop3_case("v_lerp_u8", 0x14D, |a, b, c| {
        (0..4)
            .map(|k| {
                let (x, y, r) = (
                    (a >> (8 * k)) & 0xFF,
                    (b >> (8 * k)) & 0xFF,
                    (c >> (8 * k)) & 1,
                );
                ((x + y + r) >> 1) << (8 * k)
            })
            .sum()
    }));
    // 16-bit (the low half compared).
    type F16 = fn(u16, u16) -> u16;
    let vop3_16s: [(&str, u32, F16); 12] = [
        ("v_add_nc_u16", 0x303, |a, b| a.wrapping_add(b)),
        ("v_sub_nc_u16", 0x304, |a, b| a.wrapping_sub(b)),
        ("v_mul_lo_u16", 0x305, |a, b| a.wrapping_mul(b)),
        ("v_lshrrev_b16", 0x307, |a, b| b >> (a & 15)),
        ("v_ashrrev_i16", 0x308, |a, b| {
            ((b as i16) >> (a & 15)) as u16
        }),
        ("v_max_u16", 0x309, |a, b| a.max(b)),
        ("v_max_i16", 0x30A, |a, b| (a as i16).max(b as i16) as u16),
        ("v_min_u16", 0x30B, |a, b| a.min(b)),
        ("v_min_i16", 0x30C, |a, b| (a as i16).min(b as i16) as u16),
        ("v_add_nc_i16", 0x30D, |a, b| a.wrapping_add(b)),
        ("v_sub_nc_i16", 0x30E, |a, b| a.wrapping_sub(b)),
        ("v_lshlrev_b16", 0x314, |a, b| b << (a & 15)),
    ];
    for (name, op, f) in vop3_16s {
        c.push(vop3_16_case(name, op, f));
    }
    c.push(vop3b_case("v_add_co_u32", 0x30F, |a, b| {
        a.overflowing_add(b)
    }));
    c.push(vop3b_case("v_sub_co_u32", 0x310, |a, b| {
        a.overflowing_sub(b)
    }));
    c.push(vop3b_case("v_subrev_co_u32", 0x319, |a, b| {
        b.overflowing_sub(a)
    }));
    c.push(vop3_shift64_case("v_lshlrev_b64", 0x2FF, |x, s| {
        x << (s & 63)
    }));
    c.push(vop3_shift64_case("v_lshrrev_b64", 0x300, |x, s| {
        x >> (s & 63)
    }));
    c.push(vop3_shift64_case("v_ashrrev_i64", 0x301, |x, s| {
        ((x as i64) >> (s & 63)) as u64
    }));
    c.push(mad_u64_case());

    // VOPC: the integer compares of each width, and their v_cmpx forms.
    let kinds: [(&str, u32, fn(u64, u64, u32) -> bool); 8] = [
        ("f", 0, cmp_fn!("f")),
        ("lt", 1, cmp_fn!("lt_i")),
        ("eq", 2, cmp_fn!("eq_i")),
        ("le", 3, cmp_fn!("le_i")),
        ("gt", 4, cmp_fn!("gt_i")),
        ("ne", 5, cmp_fn!("ne_i")),
        ("ge", 6, cmp_fn!("ge_i")),
        ("t", 7, cmp_fn!("t")),
    ];
    let ukinds: [fn(u64, u64, u32) -> bool; 8] = [
        cmp_fn!("f"),
        cmp_fn!("lt_u"),
        cmp_fn!("eq_u"),
        cmp_fn!("le_u"),
        cmp_fn!("gt_u"),
        cmp_fn!("ne_u"),
        cmp_fn!("ge_u"),
        cmp_fn!("t"),
    ];
    // (type, base opcode of v_cmp, bits, signed); 16-bit ones have no f/t.
    for (ty, base, bits, signed) in [
        ("i32", 0x80, 32, true),
        ("u32", 0xC0, 32, false),
        ("i64", 0xA0, 64, true),
        ("u64", 0xE0, 64, false),
        ("i16", 0x88, 16, true),
        ("u16", 0xA8, 16, false),
    ] {
        for (k, &(kind, n, sf)) in kinds.iter().enumerate() {
            if bits == 16 && (kind == "f" || kind == "t") {
                continue;
            }
            let f = if signed { sf } else { ukinds[k] };
            let op = base + n;
            c.push(vopc_case(&format!("v_cmp_{kind}_{ty}"), op, bits, false, f));
            if bits != 16 {
                c.push(vopc_case(&format!("v_cmp_{kind}_{ty}"), op, bits, true, f));
            }
            c.push(cmpx_case(
                &format!("v_cmpx_{kind}_{ty}"),
                op + 0x10,
                bits,
                f,
            ));
        }
    }
    c
}

/// Edge values, then a seeded generator.
fn input_sets(kind: Inputs, rounds: usize) -> Vec<[u32; 4]> {
    const EDGES: [u32; 22] = [
        0,
        1,
        2,
        3,
        15,
        16,
        31,
        32,
        33,
        63,
        0x7F,
        0x80,
        0xFF,
        0x7FFF,
        0x8000,
        0xFFFF,
        0x007F_FFFF,
        0x0080_0000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
    ];
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let pick = |next: &mut dyn FnMut() -> u64| -> u32 {
        let r = next();
        match r % 8 {
            0 | 1 => EDGES[(r >> 8) as usize % EDGES.len()],
            2 => (r >> 16) as u32 & 0x3F,
            3 => ((r >> 16) as u32 & 0x1F) | ((r >> 32) as u32 & 0x3F) << 16,
            4 => (r >> 20) as u32 & 0xFFFF,
            _ => (r >> 16) as u32,
        }
    };
    let mut sets = Vec::new();
    for &a in &[0u32, 1, 0x8000_0000, 0xFFFF_FFFF] {
        for &b in &[0u32, 1, 31, 0xFFFF_FFFF] {
            sets.push([a, b, b ^ 0x8000_0000, 1]);
        }
    }
    for _ in 0..(24 * rounds) {
        let mut set = [
            pick(&mut next),
            pick(&mut next),
            pick(&mut next),
            pick(&mut next),
        ];
        if next() % 4 == 0 {
            set[1] = set[0];
        }
        if next() % 6 == 0 {
            set[2] = set[0];
            set[3] = set[1];
        }
        sets.push(set);
    }
    match kind {
        Inputs::CarryIn => {
            for (n, set) in sets.iter_mut().enumerate() {
                set[2] = (n & 1) as u32;
                set[3] = (n & 1) as u32;
            }
        }
        Inputs::Float => {
            let mut floats = FLOATS.to_vec();
            for _ in 0..(48 * rounds) {
                let r = next();
                // A normal number: any sign and mantissa, an exponent
                // weighted towards the integer and half-precision ranges.
                let exp = match r % 4 {
                    0 => 1 + (r >> 8) as u32 % 254,
                    1 => 112 + (r >> 8) as u32 % 32,
                    _ => 126 + (r >> 8) as u32 % 34,
                };
                floats.push(((r >> 40) as u32 & 0x8000_0000) | exp << 23 | (r >> 16) as u32 & 0x7F_FFFF);
            }
            sets = floats
                .chunks(2)
                .enumerate()
                .map(|(n, f)| [f[0], *f.get(1).unwrap_or(&f[0]), FLOATS[n % FLOATS.len()], (n % 4) as u32])
                .collect();
        }
        Inputs::Half => {
            let mut halves: Vec<u32> = HALVES.iter().map(|&h| u32::from(h)).collect();
            for _ in 0..(48 * rounds) {
                halves.push((next() >> 24) as u32 & 0xFFFF);
            }
            sets = halves
                .iter()
                .enumerate()
                .map(|(n, &h)| {
                    let other = u32::from(HALVES[(n * 7 + 3) % HALVES.len()]);
                    [h | other << 16, other | h << 16, (n % 4) as u32, 0]
                })
                .collect();
        }
        Inputs::Double => {
            let mut doubles: Vec<u64> = DOUBLES.iter().map(|d| d.to_bits()).collect();
            for _ in 0..(24 * rounds) {
                let r = next();
                let exp = 1023 - 40 + (r >> 8) % 100;
                doubles.push((r & 1 << 63) | exp << 52 | (next() & ((1 << 52) - 1)));
            }
            sets = doubles
                .iter()
                .enumerate()
                .map(|(n, &d)| {
                    let e = doubles[(n * 5 + 1) % doubles.len()];
                    [d as u32, (d >> 32) as u32, e as u32, (e >> 32) as u32]
                })
                .collect();
        }
        Inputs::Any => {}
    }
    sets
}

/// Single-precision edge values.
const FLOATS: [u32; 40] = [
    0x0000_0000, 0x8000_0000, 0x3F00_0000, 0xBF00_0000, 0x3F80_0000, 0xBF80_0000, 0x3FC0_0000,
    0xBFC0_0000, 0x4020_0000, 0xC020_0000, 0x3EFF_FFFF, 0x3F7F_FFFF, 0x4B00_0000, 0x4AFF_FFFF,
    0x4B7F_FFFF, 0x4EFF_FFFF, 0x4F00_0000, 0xCF00_0000, 0xCF00_0001, 0x4F7F_FFFF, 0x4F80_0000,
    0x477F_E000, 0x477F_EFFF, 0x477F_F000, 0x4780_0000, 0x3880_0000, 0x387F_C000, 0x3380_0000,
    0x3300_0000, 0x3F7F_F000, 0x3F80_1000, 0x3F80_1001, 0x3F80_0FFF, 0x7F80_0000, 0xFF80_0000,
    0x7FC0_0000, 0xFFC0_0000, 0x7F80_0001, 0x7F7F_FFFF, 0xC2F6_E979,
];

/// Half-precision edge values.
const HALVES: [u16; 24] = [
    0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3C00, 0xBC00, 0x3E00, 0x4100, 0x5BFF,
    0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x57FF, 0x5800, 0xD800, 0x7800, 0x3BFF, 0x3C01,
];

/// Double-precision edge values.
const DOUBLES: [f64; 16] = [
    0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 2147483647.0, 2147483647.5, 2147483648.0, -2147483648.5,
    4294967295.5, 4294967296.0, 1.0e300, f64::INFINITY, f64::NAN,
];

/// Runs `cases` on the hardware adapter and WARP; the wrong results, said.
fn run(title: &str, cases: &[Case]) -> Vec<String> {
    let rounds = std::env::var("VICTORY_ALU_DIFF_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize)
        .max(1);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let hosts: Vec<(&str, Host)> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| match Host::new(warp) {
            Ok(h) => Some((label, h)),
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                None
            }
        })
        .collect();
    assert!(!hosts.is_empty(), "no SM 6.6 device");
    let mut refused = Vec::new();
    let mut skipped = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut ran = 0usize;
    for case in cases {
        let shader = match build(&case.body) {
            Ok(s) => s,
            Err(e) => {
                refused.push(format!("{}: {e}", case.name));
                continue;
            }
        };
        let inputs = input_sets(case.inputs, rounds);
        for (label, host) in &hosts {
            let (root, pso) = match host.compute_pipeline(&shader) {
                Ok(p) => p,
                // WARP refuses some programs the hardware takes (those that
                // need a wave of 32 lanes); the hardware's run is the check.
                Err(e) if *label == "warp" && hosts.len() > 1 => {
                    skipped.push(format!("{}: {}", case.name, e.trim()));
                    continue;
                }
                Err(e) => {
                    wrong.push(format!("{label} {}: no pipeline ({e})", case.name));
                    continue;
                }
            };
            let mut shown = 0;
            for set in &inputs {
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Constants, 0) => set[u.dword as usize],
                        (RootType::Table, 1) => 0,
                        other => panic!("unexpected user data {other:?}"),
                    };
                }
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[View::RawUav(4)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label} {}: dispatch failed: {e}", case.name));
                ran += 1;
                let want = (case.model)(*set);
                for (k, w) in want.iter().enumerate() {
                    let Some(w) = *w else { continue };
                    let got = out[0][k];
                    if !meets(got, w) {
                        if shown < 3 {
                            let what = match w {
                                NAN32 => "a NaN".to_string(),
                                NAN16 => "a half NaN".to_string(),
                                (mask, value) => format!("{:#010x} (mask {mask:#010x})", value & mask),
                            };
                            wrong.push(format!(
                                "{label} {}: inputs {:08x?}: output {k} is {got:#010x}, the ISA gives {what}",
                                case.name, set
                            ));
                        }
                        shown += 1;
                    }
                }
            }
            if shown > 3 {
                wrong.push(format!(
                    "{label} {}: {} more wrong input sets",
                    case.name,
                    shown - 3
                ));
            }
        }
    }
    eprintln!(
        "{title}: {} cases, {ran} runs, {} refused by the translator, {} not made on WARP, {} wrong",
        cases.len(),
        refused.len(),
        skipped.len(),
        wrong.len()
    );
    for r in &refused {
        eprintln!("refused: {r}");
    }
    for r in &skipped {
        eprintln!("not on WARP: {r}");
    }
    for w in &wrong {
        eprintln!("wrong: {w}");
    }
    wrong
}

#[test]
fn integer_alu_matches_the_isa_on_the_host_gpu_and_warp() {
    let wrong = run("integer ALU", &cases());
    assert!(wrong.is_empty(), "{} wrong results (see above)", wrong.len());
}

// ---------------------------------------------------------------------------
// Conversions and rounding.
// ---------------------------------------------------------------------------

fn f32_of(x: u32) -> f32 {
    f32::from_bits(x)
}

/// The half-precision bits of `mag` (>= 0), a value a half holds exactly.
fn encode_f16(mag: f64) -> u32 {
    if mag == 0.0 {
        return 0;
    }
    let exp = ((mag.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    if exp < -14 {
        return (mag * f64::powi(2.0, 24)) as u32;
    }
    let mant = ((mag / f64::powi(2.0, exp) - 1.0) * 1024.0) as u32;
    ((exp + 15) as u32) << 10 | mant
}

/// A single-precision value as a half: rounded to the nearest, ties to even
/// (`rtz`: towards zero, and too large a value is the largest half rather
/// than an infinity). `None` for a NaN.
fn f32_to_f16(x: u32, rtz: bool) -> Option<u32> {
    let f = f32_of(x);
    if f.is_nan() {
        return None;
    }
    let sign = (x >> 16) & 0x8000;
    let v = f64::from(f.abs());
    if v.is_infinite() {
        return Some(sign | 0x7C00);
    }
    // The spacing of halves around `v`.
    let quantum = if v < f64::powi(2.0, -14) {
        f64::powi(2.0, -24)
    } else {
        let exp = ((v.to_bits() >> 52) & 0x7FF) as i32 - 1023;
        f64::powi(2.0, exp - 10)
    };
    let n = v / quantum;
    let n = if rtz { n.trunc() } else { n.round_ties_even() };
    let mag = n * quantum;
    if mag > 65504.0 {
        return Some(sign | if rtz { 0x7BFF } else { 0x7C00 });
    }
    Some(sign | encode_f16(mag))
}

/// A half as a single-precision value (exact).
fn f16_to_f32(h: u32) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let (exp, mant) = ((h >> 10) & 0x1F, h & 0x3FF);
    let mag = match exp {
        0 => f64::from(mant) * f64::powi(2.0, -24),
        0x1F if mant == 0 => f64::INFINITY,
        0x1F => f64::NAN,
        _ => (1.0 + f64::from(mant) / 1024.0) * f64::powi(2.0, exp as i32 - 15),
    };
    (sign * mag) as f32
}

/// A float result: its bits, or any NaN.
fn wf(f: f32) -> Want {
    [Some(if f.is_nan() { NAN32 } else { (ALL, f.to_bits()) }), None, None]
}

/// A half result in the low half: its bits, or any NaN.
fn wh(h: Option<u32>) -> Want {
    [Some(h.map_or(NAN16, |h| (0xFFFF, h))), None, None]
}

/// Two halves packed: each compared where it is not a NaN.
fn wpack(lo: Option<u32>, hi: Option<u32>) -> Want {
    let mask = lo.map_or(0, |_| 0xFFFF) | hi.map_or(0, |_| 0xFFFF_0000);
    let value = lo.unwrap_or(0) | hi.unwrap_or(0) << 16;
    [(mask != 0).then_some((mask, value)), None, None]
}

/// Normalised 16-bit integers of a float: clamped, scaled, rounded to the
/// nearest (ties to even); a NaN is 0.
fn snorm16(f: f32) -> u32 {
    if f.is_nan() { 0 } else { (f.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i32 as u32 & 0xFFFF }
}
fn unorm16(f: f32) -> u32 {
    if f.is_nan() { 0 } else { (f.clamp(0.0, 1.0) * 65535.0).round_ties_even() as u32 }
}

fn frexp_exp(f: f32) -> u32 {
    if !f.is_finite() || f == 0.0 { 0 } else { ((f.to_bits() >> 23) & 0xFF) as i32 as u32 - 126 }
}
fn frexp_mant(f: f32) -> f32 {
    if !f.is_finite() || f == 0.0 { f } else { f32::from_bits(f.to_bits() & 0x807F_FFFF | 126 << 23) }
}

fn vop1_f_case(name: &str, op: u32, inputs: Inputs, f: fn(u32) -> Want) -> Case {
    Case { name: name.into(), body: vec![vop1(op, 0, v(1))], model: Box::new(move |i| f(i[0])), inputs }
}

fn vop3_f_case(name: &str, op: u32, src: [u16; 3], inputs: Inputs, f: fn([u32; 4]) -> Want) -> Case {
    Case { name: name.into(), body: vop3(op, 0, 0, src[0], src[1], src[2]).to_vec(), model: Box::new(f), inputs }
}

/// A conversion to a double: `op v[6:7], v1`, the halves into `v0` and `v5`.
fn to_f64_case(name: &str, op: u32, f: fn(u32) -> f64) -> Case {
    let body = vec![vop1(op, 6, v(1)), vop1(0x01, 0, v(6)), vop1(0x01, 5, v(7)), vop1(0x01, 6, ZERO)];
    Case {
        name: name.into(),
        body,
        model: Box::new(move |i| {
            let d = f(i[0]);
            if d.is_nan() {
                [None, Some((0x7FF8_0000, 0x7FF8_0000)), None]
            } else {
                w64(d.to_bits())
            }
        }),
        inputs: Inputs::Float,
    }
}

/// A conversion of the double in `v[1:2]`: `op v0, v[1:2]`.
fn from_f64_case(name: &str, op: u32, f: fn(f64) -> Want) -> Case {
    Case {
        name: name.into(),
        body: vec![vop1(op, 0, v(1))],
        model: Box::new(move |i| f(f64::from_bits(u64::from(i[0]) | u64::from(i[1]) << 32))),
        inputs: Inputs::Double,
    }
}

fn conversion_cases() -> Vec<Case> {
    use Inputs::{Any, Float, Half};
    let mut c = Vec::new();
    c.push(vop1_f_case("v_cvt_f32_i32", 0x05, Any, |a| wf(a as i32 as f32)));
    c.push(vop1_f_case("v_cvt_f32_u32", 0x06, Any, |a| wf(a as f32)));
    c.push(vop1_f_case("v_cvt_u32_f32", 0x07, Float, |a| w1(f32_of(a) as u32)));
    c.push(vop1_f_case("v_cvt_i32_f32", 0x08, Float, |a| w1(f32_of(a) as i32 as u32)));
    c.push(vop1_f_case("v_cvt_f16_f32", 0x0A, Float, |a| wh(f32_to_f16(a, false))));
    c.push(vop1_f_case("v_cvt_f32_f16", 0x0B, Half, |a| wf(f16_to_f32(a & 0xFFFF))));
    c.push(vop1_f_case("v_cvt_rpi_i32_f32", 0x0C, Float, |a| w1((f32_of(a) + 0.5).floor() as i32 as u32)));
    c.push(vop1_f_case("v_cvt_flr_i32_f32", 0x0D, Float, |a| w1(f32_of(a).floor() as i32 as u32)));
    c.push(vop1_f_case("v_cvt_off_f32_i4", 0x0E, Any, |a| wf(f32::from(((a << 28) as i32 >> 28) as i8) * 0.0625)));
    c.push(vop1_f_case("v_cvt_f32_ubyte0", 0x11, Any, |a| wf((a & 0xFF) as f32)));
    c.push(vop1_f_case("v_cvt_f32_ubyte1", 0x12, Any, |a| wf((a >> 8 & 0xFF) as f32)));
    c.push(vop1_f_case("v_cvt_f32_ubyte2", 0x13, Any, |a| wf((a >> 16 & 0xFF) as f32)));
    c.push(vop1_f_case("v_cvt_f32_ubyte3", 0x14, Any, |a| wf((a >> 24) as f32)));
    c.push(vop1_f_case("v_trunc_f32", 0x21, Float, |a| wf(f32_of(a).trunc())));
    c.push(vop1_f_case("v_ceil_f32", 0x22, Float, |a| wf(f32_of(a).ceil())));
    c.push(vop1_f_case("v_rndne_f32", 0x23, Float, |a| wf(f32_of(a).round_ties_even())));
    c.push(vop1_f_case("v_floor_f32", 0x24, Float, |a| wf(f32_of(a).floor())));
    c.push(vop1_f_case("v_frexp_exp_i32_f32", 0x3F, Float, |a| w1(frexp_exp(f32_of(a)))));
    c.push(vop1_f_case("v_frexp_mant_f32", 0x40, Float, |a| wf(frexp_mant(f32_of(a)))));
    c.push(vop1_f_case("v_cvt_f16_u16", 0x50, Any, |a| wh(f32_to_f16(((a & 0xFFFF) as f32).to_bits(), false))));
    c.push(vop1_f_case("v_cvt_f16_i16", 0x51, Any, |a| wh(f32_to_f16(f32::from(a as u16 as i16).to_bits(), false))));
    c.push(vop1_f_case("v_cvt_u16_f16", 0x52, Half, |a| w16(f16_to_f32(a & 0xFFFF) as u16 as u32)));
    c.push(vop1_f_case("v_cvt_i16_f16", 0x53, Half, |a| w16(f16_to_f32(a & 0xFFFF) as i16 as u16 as u32)));
    c.push(vop1_f_case("v_cvt_norm_i16_f16", 0x63, Half, |a| w16(snorm16(f16_to_f32(a & 0xFFFF)))));
    c.push(vop1_f_case("v_cvt_norm_u16_f16", 0x64, Half, |a| w16(unorm16(f16_to_f32(a & 0xFFFF)))));
    // Packed.
    c.push(Case {
        name: "v_cvt_pkrtz_f16_f32".into(),
        body: vec![vop2(0x2F, 0, v(1), 2)],
        model: Box::new(|i| wpack(f32_to_f16(i[0], true), f32_to_f16(i[1], true))),
        inputs: Float,
    });
    c.push(vop3_f_case("v_cvt_pkrtz_f16_f32 (vop3)", 0x12F, [v(1), v(2), ZERO], Float, |i| {
        wpack(f32_to_f16(i[0], true), f32_to_f16(i[1], true))
    }));
    c.push(vop3_f_case("v_cvt_pknorm_i16_f32", 0x368, [v(1), v(2), ZERO], Float, |i| {
        w1(snorm16(f32_of(i[0])) | snorm16(f32_of(i[1])) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pknorm_u16_f32", 0x369, [v(1), v(2), ZERO], Float, |i| {
        w1(unorm16(f32_of(i[0])) | unorm16(f32_of(i[1])) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pknorm_i16_f16", 0x312, [v(1), v(2), ZERO], Half, |i| {
        w1(snorm16(f16_to_f32(i[0] & 0xFFFF)) | snorm16(f16_to_f32(i[1] & 0xFFFF)) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pknorm_u16_f16", 0x313, [v(1), v(2), ZERO], Half, |i| {
        w1(unorm16(f16_to_f32(i[0] & 0xFFFF)) | unorm16(f16_to_f32(i[1] & 0xFFFF)) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pk_u16_u32", 0x36A, [v(1), v(2), ZERO], Any, |i| {
        w1(i[0].min(0xFFFF) | i[1].min(0xFFFF) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pk_i16_i32", 0x36B, [v(1), v(2), ZERO], Any, |i| {
        let sat = |x: u32| (x as i32).clamp(-32768, 32767) as u32 & 0xFFFF;
        w1(sat(i[0]) | sat(i[1]) << 16)
    }));
    c.push(vop3_f_case("v_cvt_pk_u8_f32", 0x15E, [v(1), v(4), v(3)], Float, |i| {
        let f = f32_of(i[0]);
        let byte = if f.is_nan() { 0 } else { f.clamp(0.0, 255.0).round_ties_even() as u32 };
        let at = 8 * (i[3] & 3);
        w1(i[2] & !(0xFF << at) | byte << at)
    }));
    c.push(vop3_f_case("v_ldexp_f32", 0x362, [v(1), v(4), ZERO], Float, |i| {
        wf(f32_of(i[0]) * f32::powi(2.0, i[3] as i32))
    }));
    // Doubles.
    c.push(to_f64_case("v_cvt_f64_f32", 0x10, |a| f64::from(f32_of(a))));
    c.push(to_f64_case("v_cvt_f64_i32", 0x04, |a| f64::from(a as i32)));
    c.push(to_f64_case("v_cvt_f64_u32", 0x16, f64::from));
    c.push(from_f64_case("v_cvt_f32_f64", 0x0F, |d| wf(d as f32)));
    c.push(from_f64_case("v_cvt_i32_f64", 0x03, |d| w1(d as i32 as u32)));
    c.push(from_f64_case("v_cvt_u32_f64", 0x15, |d| w1(d as u32)));
    c
}

/// The conversions, with 16-bit float results rounded as the console rounds
/// them (`VICTORY_F16_NEAREST=1`). By default the host's `f32tof16` gives
/// them, which rounds toward zero on NVIDIA and WARP: one step of the last
/// place lower where the result is not exact, and the largest half where the
/// console's is an infinity.
#[test]
fn conversions_match_the_isa_on_the_host_gpu_and_warp() {
    super::lower::TEST_F16_NEAREST.set(Some(true));
    let wrong = run("conversions", &conversion_cases());
    super::lower::TEST_F16_NEAREST.set(None);
    // WARP's own rounding of large unsigned integers to floats is off by one
    // step now and then; the hardware's is checked.
    let wrong: Vec<_> = wrong.into_iter().filter(|w| !w.starts_with("warp v_cvt_f32_u32")).collect();
    assert!(wrong.is_empty(), "{} wrong results (see above)", wrong.len());
}

/// What the console's rounding of 16-bit float results costs on the host
/// GPU (`VICTORY_F16_NEAREST`): a loop of dependent `v_add_f16`s, four
/// chains side by side, timed with the host's conversion and with the exact
/// one. Not run by default:
/// `cargo test --release -p victory-shader --lib bench_f16_rounding -- --ignored --nocapture`.
#[test]
#[ignore = "benchmark"]
fn bench_f16_rounding_cost() {
    const ITERATIONS: u16 = 256;
    const CHAIN: usize = 16;
    const THREADS: u32 = 64;
    const GROUPS: u32 = 4096;
    let mut body = vec![
        vop1(0x01, 5, v(1)),
        vop1(0x01, 6, v(2)),
        vop1(0x01, 7, v(3)),
        sopk(0x00, 20, ITERATIONS), // s_movk_i32 s20, ITERATIONS (s[8:11] is the buffer)
    ];
    let start = body.len();
    for _ in 0..CHAIN {
        for r in [1u32, 5, 6, 7] {
            body.push(vop2(0x32, r, v(r as u16), 4)); // v_add_f16 v[r], v[r], v4
        }
    }
    body.push(sop2(0x01, 20, s(20), ONE)); // s_sub_u32 s20, s20, 1
    body.push(sopc(0x07, s(20), ZERO)); // s_cmp_lg_u32 s20, 0
    let back = -((body.len() - start + 1) as i32);
    body.push(0xBF85_0000 | (back as u16 as u32)); // s_cbranch_scc1 -> the loop
    body.extend([
        vop2(0x1D, 0, v(1), 5), // v_xor_b32 v0, v1, v5
        vop2(0x1D, 0, v(6), 0),
        vop2(0x1D, 0, v(7), 0),
    ]);
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device");
        return;
    };
    // (1.0, 1.5) and (0.25, 0.75) as packed halves: sums that are not exact
    // in the last place soon.
    let inputs = [0x3E00_3C00u32, 0x3A00_3400, 0x3C00_3E00, 0x1400_1C00];
    for nearest in [false, true] {
        super::lower::TEST_F16_NEAREST.set(Some(nearest));
        let shader = build_for(&body, THREADS).expect("the loop translates");
        super::lower::TEST_F16_NEAREST.set(None);
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0) = (u.root, u.slot) {
                ud[u.index as usize] = inputs[u.dword as usize];
            }
        }
        let dispatch = Dispatch {
            roots: vec![(0, Root::Constants(ud))],
            views: &[View::RawUav(4)],
            samplers: &[],
            groups: [GROUPS, 1, 1],
        };
        let mut samples = Vec::new();
        for sample in 0..12 {
            let (_, ns) = host.dispatch_repeated(&root, &pso, &dispatch, 64).expect("timed dispatch");
            if sample >= 2 {
                samples.push(ns);
            }
        }
        samples.sort_by(f64::total_cmp);
        let median = (samples[4] + samples[5]) / 2.0;
        let ops = f64::from(THREADS * GROUPS) * f64::from(ITERATIONS) * (CHAIN * 4) as f64;
        eprintln!(
            "f16 adds, {}: {} bytes of DXIL, {:.1} us a dispatch (min {:.1}, max {:.1}), {:.2} ns per thousand adds",
            if nearest { "rounded to nearest (VICTORY_F16_NEAREST=1)" } else { "the host's conversion (default)" },
            shader.dxil.len(),
            median / 1e3,
            samples[0] / 1e3,
            samples[9] / 1e3,
            median / ops * 1e3,
        );
    }
}
