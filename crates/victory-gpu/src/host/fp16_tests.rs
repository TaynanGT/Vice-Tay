//! Native FP16 opt-in, device qualification and translated compute readback.

use std::ffi::OsStr;
use std::io::Write as _;

use victory_shader::dxbc::{Container, FourCC};
use victory_shader::emit::{self, HostShader, TranslationOptions};
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use victory_shader::{DxilProgram, io, ir, resource, structure};

use super::*;

#[test]
fn fp16_requires_exact_opt_in_and_a_successful_device_probe() {
    for value in [None, Some(""), Some("0"), Some("true"), Some("01"), Some("1 "), Some(" 1")] {
        let requested = native_fp16_requested(value.map(OsStr::new));
        assert!(!requested, "unexpected FP16 opt-in for {value:?}");
        assert!(!resolve_native_fp16(requested, || panic!("probed without opt-in")));
    }
    let requested = native_fp16_requested(Some(OsStr::new("1")));
    assert!(requested);
    assert!(resolve_native_fp16(requested, || Ok(true)));
    assert!(!resolve_native_fp16(requested, || Ok(false)));
    assert!(!resolve_native_fp16(requested, || Err(windows::core::Error::empty())));
}

fn fp16_shader(native_fp16: bool) -> HostShader {
    // s0..s2 are runtime constants, s[4:5] the root UAV address. Using a
    // root UAV keeps this fixture independent of the runtime view heap layout.
    let words: [u32; 14] = [
        0xbe86_03c1, // s_mov_b32 s6, -1
        0xbe87_03ff, 0x1101_4fac, // s_mov_b32 s7, 0x11014fac
        0x7e02_0200, // v_mov_b32 v1, s0
        0x7e04_0201, // v_mov_b32 v2, s1
        0x7e06_0202, // v_mov_b32 v3, s2 (upper half must be preserved)
        0x6406_0501, // v_add_f16 v3, v1, v2
        0xcc0f_4004, 0x1a02_0501, // v_pk_add_f16 v4, v1, v2
        0xe070_0000, 0x8001_0300, // buffer_store_dword v3, off, s[4:7], 0
        0xe070_0004, 0x8001_0400, // buffer_store_dword v4, off, s[4:7], 0 offset:4
        0xbf81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo { rsrc2: 6 << 1, threads: [1, 1, 1] }),
        code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    // Match the header-only DXBC produced by victory-shader's test helper.
    let bytes: [u8; 32] = [
        b'D', b'X', b'B', b'C', // magic
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // digest
        1, 0, 0, 0, // version 1.0
        32, 0, 0, 0, // total size in bytes
        0, 0, 0, 0, // part count
    ];
    let c = Container::parse(&bytes).unwrap();
    emit::emit_with_options(&c, &native, &p, &st, &io, &res, TranslationOptions { native_fp16 })
        .expect("translate FP16 compute fixture")
}

fn requires_native_fp16(shader: &HostShader) -> bool {
    let c = Container::parse(&shader.dxil).unwrap();
    let features = u64::from_le_bytes(c.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
    features & u64::from(D3D_SHADER_REQUIRES_NATIVE_16BIT_OPS) != 0
}

fn dispatch_fp16(
    host: &Host,
    shader: &HostShader,
    root: &ID3D12RootSignature,
    pso: &ID3D12PipelineState,
    inputs: [u32; 4],
) -> [u32; 2] {
    const BYTES: u64 = size_of::<[u32; 2]>() as u64;
    let output = host.uav_buffer(BYTES).expect("output UAV buffer");
    let (upload, initial) = host.upload_buffer(BYTES).expect("UAV initializer");
    let readback = host.read_back_buffer(BYTES).expect("UAV readback buffer");
    assert!(shader.root_cbvs.is_empty() && shader.root_srvs.is_empty());
    assert_eq!(shader.root_uavs, [1]);
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => inputs[u.dword as usize],
            (RootType::Other(emit::USER_VIEW_BIAS), 1) => 0,
            other => panic!("unexpected FP16 shader user data: {other:?}"),
        };
    }

    let heaps = [
        Some(host.fallback_view_heap().expect("FP16 view heap").clone()),
        Some(host.fallback_sampler_heap().expect("FP16 sampler heap").clone()),
    ];
    let list = host.new_list().expect("FP16 command list");
    // SAFETY: the buffers and descriptor heaps outlive the synchronized list;
    // the upload and readback mappings hold exactly the two output dwords.
    unsafe {
        initial.cast::<[u32; 2]>().write([u32::MAX; 2]);
        let gpu = &list.list;
        gpu.ResourceBarrier(&[transition(&output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
        gpu.CopyBufferRegion(&output, 0, &upload, 0, BYTES);
        gpu.ResourceBarrier(&[transition(&output, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        gpu.SetDescriptorHeaps(&heaps);
        gpu.SetComputeRootSignature(root);
        gpu.SetPipelineState(pso);
        gpu.SetComputeRoot32BitConstants(0, ud.len() as u32, ud.as_ptr().cast(), 0);
        gpu.SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
        gpu.Dispatch(1, 1, 1);
        gpu.ResourceBarrier(&[transition(&output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        gpu.CopyBufferRegion(&readback, 0, &output, 0, BYTES);
        list.close();
        assert_eq!(host.log_messages(), None, "D3D12 command recording");
        host.execute_sync_internal(&list);
        host.device.GetDeviceRemovedReason().expect("FP16 compute did not remove device");
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, Some(&D3D12_RANGE { Begin: 0, End: BYTES as usize }), Some(&mut mapped))
            .expect("map FP16 compute readback");
        assert!(!mapped.is_null());
        let words = mapped.cast::<[u32; 2]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        words
    }
}

#[test]
fn fp16_compute_parity_or_widened_fallback() {
    let warp = crate::switch::on("VICTORY_D3D12_WARP");
    let debug = crate::switch::on("VICTORY_D3D12_DEBUG");
    // Bypass libtest's output capture so an unavailable backend is visible
    // even when this test is invoked without --nocapture.
    let report = |text: &str| writeln!(std::io::stderr().lock(), "fp16 compute: {text}").unwrap();
    let Some(host) = host() else {
        assert!(!warp && !debug, "the explicitly requested D3D12 backend could not be created");
        report("SKIPPED: no D3D12 device; no compute dispatch ran");
        return;
    };
    // SAFETY: a plain adapter query.
    let desc = host.adapter.as_ref().and_then(|a| unsafe { a.GetDesc1().ok() });
    let desc = desc.expect("identify the adapter that executes FP16 compute");
    let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c);
    if software && !warp && !debug {
        report("SKIPPED: no hardware adapter; use VICTORY_D3D12_WARP=1 to test WARP");
        return;
    }
    assert_eq!(software, warp, "the selected device must match the requested hardware/WARP mode");
    if debug {
        assert!(host.messages.is_some(), "requested D3D12 debug layer was not enabled");
    }
    let name = String::from_utf16_lossy(&desc.Description);
    let name = name.trim_end_matches('\0');
    let backend = if warp { "WARP" } else { "hardware" };
    let mut model = D3D12_FEATURE_DATA_SHADER_MODEL { HighestShaderModel: D3D_SHADER_MODEL_6_6 };
    // SAFETY: the feature query writes the corresponding structure of the given size.
    let sm66 = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_SHADER_MODEL,
            (&raw mut model).cast(),
            size_of_val(&model) as u32,
        )
    }.is_ok() && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
    if !sm66 {
        let reason = format!("{backend} ({name}) does not support SM 6.6; no compute dispatch ran");
        assert!(!warp && !debug, "{reason}");
        report(&format!("SKIPPED: {reason}"));
        return;
    }
    report(&format!("running on {backend} ({name}), debug layer: {}", host.messages.is_some()));

    // Resolve an explicit request without changing process-wide environment.
    let resolved = resolve_native_fp16(true, || native_fp16_supported(&host.device));
    let opted_in = native_fp16_requested(std::env::var_os("VICTORY_NATIVE_FP16").as_deref());
    assert_eq!(host.native_fp16, opted_in && resolved);
    let baseline = fp16_shader(false);
    let requested = fp16_shader(resolved);
    assert!(!requires_native_fp16(&baseline));
    assert_eq!(requires_native_fp16(&requested), resolved);
    let c = Container::parse(&requested.dxil).unwrap();
    let program = DxilProgram::parse(c.part(FourCC::DXIL).unwrap().data).unwrap();
    assert_eq!(program.shader_model, if resolved { (6, 2) } else { (6, 6) });
    if resolved {
        assert_ne!(baseline.dxil, requested.dxil);
    } else {
        assert_eq!(baseline.dxil, requested.dxil);
    }
    let shader_host = emit::host::Host { device: host.device.clone() };
    let (base_root, base_pso) = shader_host.compute_pipeline(&baseline).expect("widened FP16 compute PSO");
    let (request_root, request_pso) = shader_host.compute_pipeline(&requested).expect("requested FP16 compute PSO");
    for (inputs, expected) in [
        // Low half: 1 + 2 = 3. High half: 2 + 0.5 = 2.5.
        ([0x4000_3c00, 0x3800_4000, 0x5a5a_0000, 0], [0x5a5a_4200, 0x4100_4200]),
        // Low half: -0.5 + 1.5 = 1. High half: -2 + 4 = 2.
        ([0xc000_b800, 0x4400_3e00, 0xa5a5_0000, 0], [0xa5a5_3c00, 0x4000_3c00]),
    ] {
        let widened_bits = dispatch_fp16(host, &baseline, &base_root, &base_pso, inputs);
        let requested_bits = dispatch_fp16(host, &requested, &request_root, &request_pso, inputs);
        assert_eq!(widened_bits, expected, "{backend} widened FP16 output buffer");
        assert_eq!(requested_bits, widened_bits, "{backend} FP16 output buffer parity");
    }
    if resolved {
        report(&format!("SM 6.2 native and widened output buffers matched in 2 cases on {backend} ({name})"));
    } else {
        report(&format!("native parity NOT RUN on {backend} ({name}): Options4 did not report support; widened fallback output verified"));
    }
}
