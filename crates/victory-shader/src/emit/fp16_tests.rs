//! Native half format and real Microsoft validator checks, with synthetic guest shaders.

use super::bitcode::{Attr, Bin, Term};
use super::container::{SigElement, signature};
use super::shader::Shader;
use super::validator::Validator;
use super::{HostShader, TranslationOptions, emit, emit_with_options};
use crate::dxbc::{self, Container, FourCC};
use crate::native::{ComputeIo, HwStage, NativeProgram, StageIo};
use crate::{DxilProgram, ShaderKind, io, ir, resource, structure};

const NATIVE: TranslationOptions = TranslationOptions { native_fp16: true };
const WIDENED: TranslationOptions = TranslationOptions { native_fp16: false };
const S_ENDPGM: u32 = 0xbf81_0000;
const NATIVE_LOW_PRECISION: u64 = 1 << 18;

fn validate(label: &str, bytes: &[u8], model: (u8, u8), kind: ShaderKind) {
    let container = Container::parse(bytes).unwrap();
    let part = container.part(FourCC::DXIL).unwrap();
    let dxil = DxilProgram::parse(part.data).unwrap();
    let sfi = u64::from_le_bytes(container.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
    assert_eq!(dxil.kind, kind);
    assert_eq!(dxil.shader_model, model);
    assert_eq!(dxil.dxil_version, 0x100 | u32::from(model.1));
    assert_ne!(sfi & NATIVE_LOW_PRECISION, 0);
    assert_eq!(sfi & (1 << 23), 0, "no 64-bit groupshared atomics");

    let path = Validator::default_path();
    let mut validator = Validator::load(path.as_deref()).expect("native-half tests require dxil.dll");
    let mut valid = bytes.to_vec();
    validator.validate(&mut valid).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(valid, bytes, "our container signature differs from the validator's");
    // Prove the call is not an accept-all stub. Damage the header of the same shader.
    valid[0] ^= 1;
    assert!(validator.validate(&mut valid).is_err(), "validator accepted invalid DXBC magic");
    eprintln!("validated {label} ({kind:?} {}.{}) with {:?}", model.0, model.1, path);
}

fn fixture(native: &NativeProgram, parts: &[(FourCC, &[u8])]) -> (HostShader, HostShader) {
    let bytes = dxbc::build(parts);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    let default = emit(&c, native, &p, &st, &io, &res).unwrap();
    let widened = emit_with_options(&c, native, &p, &st, &io, &res, WIDENED).unwrap();
    assert_eq!(default.dxil, widened.dxil, "default must equal explicit OFF byte for byte");
    let native = emit_with_options(&c, native, &p, &st, &io, &res, NATIVE).unwrap();
    (widened, native)
}

const fn vop1(op: u32, dst: u32, src: u32) -> u32 {
    0x7e00_0000 | dst << 17 | op << 9 | src
}

const fn vop2(op: u32, dst: u32, src: u32, src1: u32) -> u32 {
    op << 25 | dst << 17 | src1 << 9 | src
}

#[test]
fn fp16_sm62_compute_uses_typed_arithmetic_conversions_and_widened_sqrt() {
    let words = [
        vop2(0x1a, 2, 130, 0), // v_lshlrev_b32 v2, 2, v0 (byte address in LDS)
        vop1(6, 1, 256),       // v_cvt_f32_u32 v1, v0
        vop1(0x0a, 1, 257),    // v_cvt_f16_f32 v1, v1
        vop2(0x32, 1, 242, 1), // v_add_f16 v1, 1.0, v1
        vop2(0x34, 1, 240, 1), // v_subrev_f16 v1, 0.5, v1
        vop2(0x35, 1, 244, 1), // v_mul_f16 v1, 2.0, v1
        // v_fma_f16 v1, v1, 2.0, 1.0 (VOP3)
        0xd74b_0001, 257 | (244 << 9) | (242 << 18),
        vop2(0x39, 1, 242, 1), // v_max_f16 v1, 1.0, v1
        vop1(0x55, 1, 257),    // v_sqrt_f16 v1, v1 (call must stay f32)
        0xd4c9_0014, 242 | (257 << 9), // v_cmp_lt_f16_e64 s20, 1.0, v1
        0xd501_0001, 128 | (257 << 9) | (20 << 18), // v_cndmask_b32_e64 v1, 0, v1, s20
        vop1(0x0b, 1, 257),    // v_cvt_f32_f16 v1, v1
        0xd834_0000, 0x0000_0102, // ds_write_b32 v2, v1
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        // Two guest waves, so this does not require a fixed host WaveSize.
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 15, threads: [64, 1, 1] }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let (widened, native) = fixture(&native, &[]);
    assert_ne!(widened.dxil, native.dxil);
    validate("half arithmetic + LDS", &native.dxil, (6, 2), ShaderKind::Compute);
}

#[test]
fn fp16_sm62_pixel_keeps_signature_f32() {
    let words = [
        vop1(0x0a, 0, 256),    // v_cvt_f16_f32 v0, v0 (SV_Position.x)
        vop2(0x32, 0, 240, 0), // v_add_f16 v0, 0.5, v0
        vop1(0x0b, 0, 256),    // v_cvt_f32_f16 v0, v0
        0xf800_180f, 0,       // exp mrt0 v0, v0, v0, v0 done vm
        S_ENDPGM,
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: vec![],
        used: vec![],
        io: StageIo::Ps {
            inputs: vec![], input_addr: 1 << 8, db_shader_control: 0, patches: vec![],
        },
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let input = signature(&[SigElement {
        name: "SV_Position".into(), index: 0, kind: 3, comp: 9, interp: 4,
        row: Some(0), col: 0, cols: 4, usage: 1, half: false,
    }], true);
    let output = signature(&[SigElement {
        name: "SV_Target".into(), index: 0, kind: 16, comp: 9, interp: 0,
        row: Some(0), col: 0, cols: 4, usage: 15, half: false,
    }], false);
    let (widened, native) = fixture(&native, &[(FourCC::ISG1, &input), (FourCC::OSG1, &output)]);
    assert_ne!(widened.dxil, native.dxil);
    validate("half PS with f32 input/output", &native.dxil, (6, 2), ShaderKind::Pixel);
}

#[test]
fn fp16_sm62_static_raw_uav_load_extract_and_store_validate() {
    let mut s = Shader::new(false);
    s.threads = Some([1, 1, 1]);
    s.enable_native_fp16();
    let (half, _) = s.native_fp16_types().unwrap();
    let ret = s.m.named_struct("dx.types.ResRet.f16", vec![half, half, half, half, s.t.i32]);
    let b = s.entry;
    // A binding at u7 has range ID 0 but absolute register index 7.
    let h = s.root_uav_handle(7);
    let byte = s.m.i32(0);
    let offset = s.m.undef(s.t.i32);
    let mask = s.m.i8(1);
    let align = s.m.i32(2);
    let load = s.op(b, "rawBufferLoad.f16", Attr::ReadOnly, ret, 139, &[h, byte, offset, mask, align]);
    let a = s.m.extract(b, load, 0);
    let two = s.m.half_const(half, 0x4000);
    let product = s.m.bin(b, Bin::Mul, a, two);
    let unused = s.m.undef(half);
    let out = s.m.i32(2);
    s.op(b, "rawBufferStore.f16", Attr::NoUnwind, s.t.void, 140,
        &[h, out, offset, product, unused, unused, unused, mask, align]);
    s.m.terminate(b, Term::Ret);
    validate("half raw buffer + static binding", &s.finish(), (6, 2), ShaderKind::Compute);
}

#[test]
fn fp16_raw_aligned_load_extract_and_store_validate_with_heap_bindings() {
    let body = [
        0xe028_0002, 0x8002_0100, // buffer_load_ushort v1, off, s[8:11], 0 offset:2
        vop2(0x32, 0, 242, 1),   // v_add_f16 v0, 1.0, v1
        0xe068_0006, 0x8002_0000, // buffer_store_short v0, off, s[8:11], 0 offset:6
    ];
    let shader = super::valu_tests::build_cs_with_options(&body, 1, 0, NATIVE);
    // This shader needs a directly indexed descriptor heap, which cannot be downgraded to 6.2.
    validate("half raw buffer + heap", &shader.dxil, (6, 6), ShaderKind::Compute);
}
