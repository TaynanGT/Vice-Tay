//! Two presented planes as one image: the console's display puts the second
//! (the interface) over the first (the scene), each scaled to the screen.
//! Grand Theft Auto VI presents so on Series S.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{PCSTR, s};

use crate::display_output::ColorSpace;
use crate::host::Host;

/// Whether a second presented plane is composed over the first
/// (`VICTORY_PLANES`; on by default, `0` turns it off). It acts only when the
/// title presents a second plane.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLANES").is_none_or(|v| v != "0"))
}

const SOURCE: &str = r#"
Texture2D<float4> scene : register(t0);
Texture2D<float4> over : register(t1);
SamplerState smooth_ : register(s0);
cbuffer Plane : register(b0) {
    uint mode;
    uint scene_space;
    uint output_space;
    float paper_white;
};
struct V { float4 at : SV_Position; float2 uv : TEXCOORD0; };
V vs(uint id : SV_VertexID) {
    V v;
    v.uv = float2((id << 1) & 2, id & 2);
    v.at = float4(v.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return v;
}
float3 safe_pow(float3 x, float e) {
    return step(1e-30, x) * pow(max(x, 1e-30), e);
}
float3 srgb_to_linear(float3 c) {
    c = max(c, 0.0);
    float3 lo = c / 12.92;
    float3 hi = pow(max((c + 0.055) / 1.055, 1e-30), 2.4);
    return lerp(lo, hi, step(0.04045, c));
}
float3 linear_to_srgb(float3 c) {
    c = max(c, 0.0);
    float3 lo = c * 12.92;
    float3 hi = 1.055 * safe_pow(c, 1.0 / 2.4) - 0.055;
    return lerp(lo, hi, step(0.0031308, c));
}
float3 rec709_to_rec2020(float3 rgb) {
    return float3(
        dot(rgb, float3(0.6274040, 0.3292820, 0.0433140)),
        dot(rgb, float3(0.0690973, 0.9195404, 0.0113623)),
        dot(rgb, float3(0.0163914, 0.0880133, 0.8955953))
    );
}
float3 rec2020_to_rec709(float3 rgb) {
    return float3(
        dot(rgb, float3(1.6604910, -0.5876411, -0.0728499)),
        dot(rgb, float3(-0.1245505, 1.1328999, -0.0083494)),
        dot(rgb, float3(-0.0181508, -0.1005789, 1.1187297))
    );
}
float3 pq_to_linear_nits(float3 pq) {
    float3 p = safe_pow(max(pq, 0.0), 1.0 / 78.84375);
    float3 num = max(p - 0.8359375, 0.0);
    float3 den = max(18.8515625 - 18.6875 * p, 1e-6);
    return safe_pow(num / den, 1.0 / 0.1593017578125) * 10000.0;
}
float3 linear_nits_to_pq(float3 nits) {
    float3 y = saturate(nits / 10000.0);
    float3 ym1 = safe_pow(y, 0.1593017578125);
    float3 num = 0.8359375 + 18.8515625 * ym1;
    float3 den = 1.0 + 18.6875 * ym1;
    return safe_pow(num / den, 78.84375);
}
float3 ui_linear_premul(float4 o) {
    float a = saturate(o.a);
    float3 straight = (a > 1e-5) ? (max(o.rgb, 0.0) / a) : max(o.rgb, 0.0);
    float3 lin = srgb_to_linear(straight);
    return (a > 1e-5) ? (lin * a) : lin;
}
float4 ps(V v) : SV_Target {
    float4 s = scene.SampleLevel(smooth_, v.uv, 0);
    if (mode == 2) {
        return (output_space == 0) ? saturate(s) : max(s, 0.0);
    }
    if (output_space == 0) {
        // SDR swap-chain output (preserve exact sRGB blending for SDR scenes,
        // and gracefully convert HDR scenes down to SDR when falling back).
        float3 base = s.rgb;
        if (scene_space == 1) {
            base = linear_to_srgb(saturate(rec2020_to_rec709(pq_to_linear_nits(s.rgb)) / max(paper_white, 80.0)));
        } else if (scene_space == 2) {
            base = linear_to_srgb(saturate(s.rgb * (80.0 / max(paper_white, 80.0))));
        } else {
            base = saturate(s.rgb);
        }
        if (mode == 1) return float4(base, 1);
        float4 o = over.SampleLevel(smooth_, v.uv, 0);
        return float4(saturate(o.rgb + base * (1 - saturate(o.a))), 1);
    }
    if (output_space == 2) {
        // scRGB extended linear Rec.709 output (1.0 == 80 nits): NEVER saturate()
        // so specular highlights > 1.0 and wide-gamut values stay intact.
        float3 base;
        if (scene_space == 2) {
            base = s.rgb;
        } else if (scene_space == 1) {
            base = rec2020_to_rec709(pq_to_linear_nits(s.rgb)) / 80.0;
        } else {
            base = srgb_to_linear(s.rgb) * (paper_white / 80.0);
        }
        if (mode == 1) return float4(base, 1);
        float4 o = over.SampleLevel(smooth_, v.uv, 0);
        float3 ui_scrgb = ui_linear_premul(o) * (paper_white / 80.0);
        return float4(ui_scrgb + base * (1 - saturate(o.a)), 1);
    }
    // output_space == 1: HDR10 ST.2084 PQ Rec.2020 output.
    if (mode == 1 && scene_space == 1) {
        return float4(max(s.rgb, 0.0), 1);
    }
    float3 scene_nits;
    if (scene_space == 1) {
        scene_nits = pq_to_linear_nits(s.rgb);
    } else if (scene_space == 2) {
        scene_nits = max(rec709_to_rec2020(s.rgb), 0.0) * 80.0;
    } else {
        scene_nits = rec709_to_rec2020(srgb_to_linear(s.rgb)) * paper_white;
    }
    if (mode == 1) {
        return float4(linear_nits_to_pq(scene_nits), 1);
    }
    float4 o = over.SampleLevel(smooth_, v.uv, 0);
    float3 ui_nits = rec709_to_rec2020(ui_linear_premul(o)) * paper_white;
    float3 blended_nits = ui_nits + scene_nits * (1 - saturate(o.a));
    return float4(linear_nits_to_pq(blended_nits), 1);
}
"#;

type View<'a> = (&'a ID3D12Resource, DXGI_FORMAT);

pub(crate) enum Picture<'a> {
    Composed(View<'a>, View<'a>),
    Scene(View<'a>),
    Overlay(View<'a>),
}

pub(crate) const UI_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
pub(crate) const UI_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

/// Two images allow the proxy to read the previous UI asynchronously while
/// the current Present records another. Present waits for the previous UI
/// composition before returning, so its image can be reused next time.
pub(crate) struct UiFrames {
    pub(crate) size: (u32, u32),
    pub(crate) images: [ID3D12Resource; 2],
}

impl UiFrames {
    pub(crate) fn new(host: &Host, size: (u32, u32)) -> Option<Self> {
        let create = || {
            let mut texture = None;
            // SAFETY: a default-heap RGBA render target for one UI image.
            unsafe {
                host.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(size.0),
                        Height: size.1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: UI_FORMAT,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    UI_READ,
                    None,
                    &mut texture,
                ).ok()?;
            }
            texture
        };
        Some(Self { size, images: [create()?, create()?] })
    }
}

pub(crate) struct Composer {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Secondary PSO compiled for [`UI_FORMAT`] so `Picture::Overlay` rendering
    /// into [`UiFrames`] matches RTV format even when the swap chain is HDR10/scRGB.
    ui_pso: ID3D12PipelineState,
    views: Vec<ID3D12DescriptorHeap>,
    step: u32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Composer {}
// SAFETY: as above.
unsafe impl Sync for Composer {}

fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names live through the call; the out pointers are
    // ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("compose"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        tracing::warn!(error = %e, log = %blob_text(errors.as_ref()), "display composer shader not compiled");
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn blob_text(b: Option<&ID3DBlob>) -> String {
    // SAFETY: the blob's own bytes, alive through the call.
    b.map(|b| unsafe {
        String::from_utf8_lossy(std::slice::from_raw_parts(
            b.GetBufferPointer().cast::<u8>(),
            b.GetBufferSize(),
        ))
        .into_owned()
    })
    .unwrap_or_default()
}

/// Parameters controlling HDR/SDR conversion and UI paper-white scaling during
/// display composition.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ComposeParams {
    pub(crate) scene_space: ColorSpace,
    pub(crate) output_space: ColorSpace,
    pub(crate) paper_white_nits: f32,
}

impl Default for ComposeParams {
    fn default() -> Self {
        Self {
            scene_space: ColorSpace::Sdr,
            output_space: ColorSpace::Sdr,
            paper_white_nits: crate::display_output::DEFAULT_PAPER_WHITE_NITS,
        }
    }
}

/// The composer's root signature, built here rather than with an HLSL
/// `[RootSignature]` attribute, which Wine's d3dcompiler does not compile:
/// t0-t1 in a table and four 32-bit constants at b0 for the pixel shader, and
/// a linear clamping sampler at s0.
fn root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
    let range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MaxAnisotropy: 16,
        ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
        MaxLOD: f32::MAX,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let (mut blob, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the description and what it points at live through the calls.
    unsafe {
        if let Err(e) =
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut errors))
        {
            tracing::warn!(error = %e, log = %blob_text(errors.as_ref()), "display composer root signature not made");
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        device.CreateRootSignature(
            0,
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
        )
    }
}

impl Composer {
    pub(crate) fn new(host: &Host, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        let (vs, ps) = (
            compile(s!("vs"), s!("vs_5_0"))?,
            compile(s!("ps"), s!("ps_5_0"))?,
        );
        // SAFETY: the blobs' own bytes, alive through the calls.
        unsafe {
            let device = &host.device;
            let root = root_signature(device)?;
            let code = |b: &ID3DBlob| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.GetBufferPointer(),
                BytecodeLength: b.GetBufferSize(),
            };
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            desc.RTVFormats[0] = format;
            desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
                D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            let ui_pso: ID3D12PipelineState = if format == UI_FORMAT {
                pso.clone()
            } else {
                desc.RTVFormats[0] = UI_FORMAT;
                device.CreateGraphicsPipelineState(&desc)?
            };
            Ok(Composer {
                format,
                root,
                pso,
                ui_pso,
                views: Vec::new(),
                step: device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            })
        }
    }

    /// Swap-chain RTV format this composer's primary PSO was compiled for.
    pub(crate) fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    /// Each presenter command-list slot owns its own pair of SRVs. The swap
    /// chain may have several outstanding presents; the slot is selected by
    /// the completion fence, and its descriptors are only overwritten on reuse.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        host: &Host,
        slot: usize,
        list: &ID3D12GraphicsCommandList,
        picture: Picture<'_>,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        params: ComposeParams,
    ) -> windows::core::Result<()> {
        // SAFETY: querying the destination resource's static format description.
        let target_format = unsafe { back.GetDesc().Format };
        let (expected_format, use_ui_pso) = match picture {
            Picture::Overlay(_) => (UI_FORMAT, true),
            Picture::Composed(..) | Picture::Scene(_) => (self.format, false),
        };
        if target_format != expected_format
            || !crate::display_output::format_matches_color_space(
                target_format.0,
                params.output_space,
            )
        {
            tracing::warn!(
                target_format = target_format.0,
                expected_format = expected_format.0,
                output_space = ?params.output_space,
                "composer target format does not match active display mode"
            );
            return Err(windows::core::Error::from(
                windows::Win32::Foundation::E_INVALIDARG,
            ));
        }
        let (mode, planes) = match picture {
            Picture::Composed(scene, overlay) => (0u32, [scene, overlay]),
            Picture::Scene(scene) => (1u32, [scene, scene]),
            Picture::Overlay(overlay) => (2u32, [overlay, overlay]),
        };
        let constants: [u32; 4] = [
            mode,
            params.scene_space as u32,
            params.output_space as u32,
            params.paper_white_nits.to_bits(),
        ];
        while self.views.len() <= slot {
            let heap = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 2,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })?
            };
            self.views.push(heap);
        }
        let views = &self.views[slot];
        // SAFETY: an open list; the views are made in slots no frame in flight
        // reads; live textures.
        unsafe {
            let (cpu, gpu) = (
                views.GetCPUDescriptorHandleForHeapStart(),
                views.GetGPUDescriptorHandleForHeapStart(),
            );
            for (i, plane) in planes.iter().enumerate() {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + (i * self.step as usize),
                };
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: plane.1,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(Some(plane.0), Some(&desc), at);
            }
            let rtv = host.rtv(Some(back), None);
            let pso = if use_ui_pso { &self.ui_pso } else { &self.pso };
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(pso);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.SetGraphicsRoot32BitConstants(1, 4, constants.as_ptr().cast(), 0);
            list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.DrawInstanced(3, 1, 0, 0);
        }
        Ok(())
    }
}

/// Select a legal float-sampled view of a display plane, preserving a typed
/// resource's format. Typeless resources use the corresponding display type.
pub(crate) fn view(d: &D3D12_RESOURCE_DESC) -> Option<DXGI_FORMAT> {
    if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || d.DepthOrArraySize != 1
        || d.SampleDesc.Count != 1
        || d.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
    {
        return None;
    }
    match d.Format.0 {
        9 => Some(DXGI_FORMAT_R16G16B16A16_FLOAT),
        23 => Some(DXGI_FORMAT_R10G10B10A2_UNORM),
        27 => Some(DXGI_FORMAT_R8G8B8A8_UNORM),
        90 => Some(DXGI_FORMAT_B8G8R8A8_UNORM),
        10 | 11 | 13 | 24 | 26 | 28 | 29 | 31 | 67 | 87 | 91 => Some(d.Format),
        _ => None,
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;

    #[test]
    fn composes_scaled_premultiplied_plane_on_gpu() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping display-plane readback: no D3D12 device");
            return;
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut compositor = Composer::new(host, format).unwrap();
        let list = host.new_list().unwrap();
        let texture = |width: u64| -> ID3D12Resource {
            let mut resource = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut resource,
                    )
                    .unwrap()
            };
            resource.unwrap()
        };
        // A half-red base at one resolution. The overlay at output resolution
        // contains transparent, half-opaque green and fully opaque blue pixels.
        let base = texture(1);
        let overlay = texture(3);
        let output = texture(3);
        let scene_output = texture(3);
        let ui_output = texture(3);
        let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let cmds = &list.list;
        unsafe {
            cmds.ClearRenderTargetView(host.rtv(Some(&base), None), &[0.5, 0.0, 0.0, 1.0], None);
            let overlay_rtv = host.rtv(Some(&overlay), None);
            let colors = [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.5],
                [0.0, 0.0, 1.0, 1.0],
            ];
            for (i, color) in colors.iter().enumerate() {
                let rect = windows::Win32::Foundation::RECT {
                    left: i as i32,
                    right: i as i32 + 1,
                    top: 0,
                    bottom: 1,
                };
                cmds.ClearRenderTargetView(overlay_rtv, color, Some(&[rect]));
            }
            cmds.ResourceBarrier(&[
                crate::host::transition(&base, target, read),
                crate::host::transition(&overlay, target, read),
            ]);
            compositor
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Composed((&base, format), (&overlay, format)),
                    &output,
                    (3, 1),
                    ComposeParams::default(),
                )
                .unwrap();
            compositor
                .record(
                    host,
                    1,
                    cmds,
                    Picture::Scene((&base, format)),
                    &scene_output,
                    (3, 1),
                    ComposeParams::default(),
                )
                .unwrap();
            compositor
                .record(
                    host,
                    2,
                    cmds,
                    Picture::Overlay((&overlay, format)),
                    &ui_output,
                    (3, 1),
                    ComposeParams::default(),
                )
                .unwrap();
            let outputs = [&output, &scene_output, &ui_output];
            cmds.ResourceBarrier(&outputs.map(|output| crate::host::transition(
                output, target, D3D12_RESOURCE_STATE_COPY_SOURCE,
            )));
            let readbacks = outputs.map(|output| crate::capture::read_back(&host.device, cmds, output).unwrap());
            list.close();
            host.execute(&[&list]);
            host.device.GetDeviceRemovedReason().unwrap();
            let expected = [
                [128, 0, 0, 255, 64, 128, 0, 255, 0, 0, 255, 255],
                [128, 0, 0, 255, 128, 0, 0, 255, 128, 0, 0, 255],
                [0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 255, 255],
            ];
            for ((buffer, footprint), expected) in readbacks.into_iter().zip(expected) {
                let mut mapped = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut mapped)).unwrap();
                let start = mapped.cast::<u8>().add(footprint.Offset as usize);
                let pixels: [u8; 12] = std::slice::from_raw_parts(start, 12).try_into().unwrap();
                buffer.Unmap(0, None);
                for (actual, expected) in pixels.iter().zip(expected) {
                    assert!(actual.abs_diff(expected) <= 1, "unexpected plane {pixels:?}");
                }
                eprintln!("D3D12 display-plane readback RGBA: {pixels:?}");
            }
        }
    }
}

#[cfg(test)]
mod hdr_tests {
    use super::*;

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = u32::from((bits >> 15) & 1) << 31;
        let exp = i32::from((bits >> 10) & 0x1f);
        let frac = u32::from(bits & 0x3ff);
        let out = if exp == 0 {
            if frac == 0 {
                sign
            } else {
                let mut f = frac;
                let mut e = -14i32;
                while (f & 0x400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                sign | (((e + 127) as u32) << 23) | ((f & 0x3ff) << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (frac << 13)
        } else {
            sign | (((exp - 15 + 127) as u32) << 23) | (frac << 13)
        };
        f32::from_bits(out)
    }

    fn make_rt(host: &Host, width: u64, format: DXGI_FORMAT) -> ID3D12Resource {
        let mut resource = None;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        // SAFETY: valid default-heap 2D render target creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut resource,
                )
                .unwrap();
        }
        resource.unwrap()
    }

    fn record_readback(
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
    ) -> (ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: querying copyable footprint and creating readback buffer.
        unsafe {
            let desc = texture.GetDesc();
            device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total.max(1),
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            (buffer, footprint)
        }
    }

    #[test]
    fn hdr_scrgb_preserves_highlights_above_one_and_scales_paper_white_ui() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping HDR scRGB composer test: no D3D12 device");
            return;
        };
        let hdr_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let ui_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut compositor = Composer::new(host, hdr_fmt).unwrap();
        let list = host.new_list().unwrap();

        let base = make_rt(host, 1, hdr_fmt);
        let overlay = make_rt(host, 3, ui_fmt);
        let composed_out = make_rt(host, 3, hdr_fmt);
        let scene_out = make_rt(host, 1, hdr_fmt);

        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let sr = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let cs = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let cmds = &list.list;
        let params = ComposeParams {
            scene_space: ColorSpace::ScRgb,
            output_space: ColorSpace::ScRgb,
            paper_white_nits: 240.0, // 240 / 80 = 3.0 scRGB
        };

        // SAFETY: valid D3D12 recording and readback on test resources.
        unsafe {
            // 1000-nit red highlight (12.5 * 80 nits), 160-nit green, 40-nit blue.
            cmds.ClearRenderTargetView(host.rtv(Some(&base), None), &[12.5, 2.0, 0.5, 1.0], None);
            let overlay_rtv = host.rtv(Some(&overlay), None);
            let ui_pixels = [
                [0.0, 0.0, 0.0, 0.0], // transparent -> preserves 12.5 highlight
                [0.0, 0.5, 0.0, 0.5], // 50% green premul -> R=6.25, G=0.5*3.0 + 2.0*0.5 = 2.5
                [1.0, 1.0, 1.0, 1.0], // opaque white -> 3.0 scRGB (240 nits)
            ];
            for (i, color) in ui_pixels.iter().enumerate() {
                let rect = windows::Win32::Foundation::RECT {
                    left: i as i32,
                    right: i as i32 + 1,
                    top: 0,
                    bottom: 1,
                };
                cmds.ClearRenderTargetView(overlay_rtv, color, Some(&[rect]));
            }
            cmds.ResourceBarrier(&[
                crate::host::transition(&base, rt, sr),
                crate::host::transition(&overlay, rt, sr),
            ]);
            compositor
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Composed((&base, hdr_fmt), (&overlay, ui_fmt)),
                    &composed_out,
                    (3, 1),
                    params,
                )
                .unwrap();
            compositor
                .record(
                    host,
                    1,
                    cmds,
                    Picture::Scene((&base, hdr_fmt)),
                    &scene_out,
                    (1, 1),
                    params,
                )
                .unwrap();
            cmds.ResourceBarrier(&[
                crate::host::transition(&composed_out, rt, cs),
                crate::host::transition(&scene_out, rt, cs),
            ]);
            let (rb_comp, fp_comp) = record_readback(&host.device, cmds, &composed_out);
            let (rb_scene, fp_scene) = record_readback(&host.device, cmds, &scene_out);
            list.close();
            host.execute(&[&list]);
            host.device.GetDeviceRemovedReason().unwrap();

            let read_fp16 = |buf: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, count: usize| -> Vec<[f32; 4]> {
                let mut ptr = std::ptr::null_mut();
                buf.Map(0, None, Some(&mut ptr)).unwrap();
                let base_ptr = ptr.cast::<u8>().add(fp.Offset as usize).cast::<u16>();
                let raw = std::slice::from_raw_parts(base_ptr, count * 4);
                let pixels = raw
                    .chunks_exact(4)
                    .map(|c| [f16_to_f32(c[0]), f16_to_f32(c[1]), f16_to_f32(c[2]), f16_to_f32(c[3])])
                    .collect();
                buf.Unmap(0, None);
                pixels
            };

            let comp = read_fp16(&rb_comp, &fp_comp, 3);
            let scene = read_fp16(&rb_scene, &fp_scene, 1);

            // Pixel 0: >1.0 HDR specular highlight preserved uncrushed.
            assert!((comp[0][0] - 12.5).abs() < 0.05, "pixel 0 crushed: {:?}", comp[0]);
            assert!((comp[0][1] - 2.0).abs() < 0.05, "pixel 0 G: {:?}", comp[0]);
            assert!((comp[0][2] - 0.5).abs() < 0.02, "pixel 0 B: {:?}", comp[0]);
            // Pixel 1: half-opaque green UI at 240 nits over HDR scene.
            assert!((comp[1][0] - 6.25).abs() < 0.1, "pixel 1 R: {:?}", comp[1]);
            assert!((comp[1][1] - 2.5).abs() < 0.1, "pixel 1 G: {:?}", comp[1]);
            assert!((comp[1][2] - 0.25).abs() < 0.05, "pixel 1 B: {:?}", comp[1]);
            // Pixel 2: opaque white sRGB UI scaled cleanly to 240 nits (3.0 scRGB).
            for ch in 0..3 {
                assert!((comp[2][ch] - 3.0).abs() < 0.05, "pixel 2 ch {ch}: {:?}", comp[2]);
            }
            // Single-plane scene pass also preserves >1.0 uncrushed.
            assert!((scene[0][0] - 12.5).abs() < 0.05, "scene crushed: {:?}", scene[0]);
            assert_eq!(host.log_messages(), None, "D3D12 debug layer reported an error");
        }
    }

    #[test]
    fn hdr10_pq_preserves_peak_luminance_and_maps_paper_white_ui() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping HDR10 PQ composer test: no D3D12 device");
            return;
        };
        let scrgb_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
        let pq_fmt = DXGI_FORMAT_R10G10B10A2_UNORM;
        let ui_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut compositor = Composer::new(host, pq_fmt).unwrap();
        let list = host.new_list().unwrap();

        let base_scrgb = make_rt(host, 1, scrgb_fmt);
        let base_pq = make_rt(host, 1, pq_fmt);
        let overlay = make_rt(host, 3, ui_fmt);
        let out_from_scrgb = make_rt(host, 3, pq_fmt);
        let out_from_pq = make_rt(host, 3, pq_fmt);

        let rt = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let sr = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        let cs = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let cmds = &list.list;

        // SAFETY: valid D3D12 recording and readback on test resources.
        unsafe {
            // 1000 nits white in scRGB (12.5 * 80 nits) and in ST.2084 PQ (~0.7518).
            cmds.ClearRenderTargetView(host.rtv(Some(&base_scrgb), None), &[12.5, 12.5, 12.5, 1.0], None);
            cmds.ClearRenderTargetView(host.rtv(Some(&base_pq), None), &[0.7518, 0.7518, 0.7518, 1.0], None);
            let overlay_rtv = host.rtv(Some(&overlay), None);
            let ui_pixels = [
                [0.0, 0.0, 0.0, 0.0], // transparent -> 1000 nits PQ (~0.7518)
                [0.0, 0.0, 0.0, 0.5], // 50% dim -> 500 nits PQ (~0.6766)
                [1.0, 1.0, 1.0, 1.0], // opaque white -> 200 nits paper white PQ (~0.5807)
            ];
            for (i, color) in ui_pixels.iter().enumerate() {
                let rect = windows::Win32::Foundation::RECT {
                    left: i as i32,
                    right: i as i32 + 1,
                    top: 0,
                    bottom: 1,
                };
                cmds.ClearRenderTargetView(overlay_rtv, color, Some(&[rect]));
            }
            cmds.ResourceBarrier(&[
                crate::host::transition(&base_scrgb, rt, sr),
                crate::host::transition(&base_pq, rt, sr),
                crate::host::transition(&overlay, rt, sr),
            ]);
            compositor
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Composed((&base_scrgb, scrgb_fmt), (&overlay, ui_fmt)),
                    &out_from_scrgb,
                    (3, 1),
                    ComposeParams {
                        scene_space: ColorSpace::ScRgb,
                        output_space: ColorSpace::Hdr10Pq,
                        paper_white_nits: 200.0,
                    },
                )
                .unwrap();
            compositor
                .record(
                    host,
                    1,
                    cmds,
                    Picture::Composed((&base_pq, pq_fmt), (&overlay, ui_fmt)),
                    &out_from_pq,
                    (3, 1),
                    ComposeParams {
                        scene_space: ColorSpace::Hdr10Pq,
                        output_space: ColorSpace::Hdr10Pq,
                        paper_white_nits: 200.0,
                    },
                )
                .unwrap();
            cmds.ResourceBarrier(&[
                crate::host::transition(&out_from_scrgb, rt, cs),
                crate::host::transition(&out_from_pq, rt, cs),
            ]);
            let (rb_a, fp_a) = record_readback(&host.device, cmds, &out_from_scrgb);
            let (rb_b, fp_b) = record_readback(&host.device, cmds, &out_from_pq);
            list.close();
            host.execute(&[&list]);
            host.device.GetDeviceRemovedReason().unwrap();

            let unpack_r10g10b10a2 = |buf: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| -> [[f32; 3]; 3] {
                let mut ptr = std::ptr::null_mut();
                buf.Map(0, None, Some(&mut ptr)).unwrap();
                let words = std::slice::from_raw_parts(
                    ptr.cast::<u8>().add(fp.Offset as usize).cast::<u32>(),
                    3,
                );
                let mut out = [[0.0f32; 3]; 3];
                for (i, &w) in words.iter().enumerate() {
                    out[i] = [
                        (w & 0x3ff) as f32 / 1023.0,
                        ((w >> 10) & 0x3ff) as f32 / 1023.0,
                        ((w >> 20) & 0x3ff) as f32 / 1023.0,
                    ];
                }
                buf.Unmap(0, None);
                out
            };

            for (label, pixels) in [
                ("scRGB->PQ", unpack_r10g10b10a2(&rb_a, &fp_a)),
                ("PQ->PQ", unpack_r10g10b10a2(&rb_b, &fp_b)),
            ] {
                // 1000 nits uncrushed (~0.7518 PQ, NOT crushed to 80 nits ~0.485).
                for ch in 0..3 {
                    assert!(
                        (pixels[0][ch] - 0.7518).abs() < 0.01,
                        "{label} pixel 0 (1000 nits) ch {ch}: {:?}",
                        pixels[0]
                    );
                    // 500 nits dimmed (~0.6766 PQ).
                    assert!(
                        (pixels[1][ch] - 0.6766).abs() < 0.01,
                        "{label} pixel 1 (500 nits) ch {ch}: {:?}",
                        pixels[1]
                    );
                    // 200 nits paper-white UI (~0.5807 PQ).
                    assert!(
                        (pixels[2][ch] - 0.5807).abs() < 0.01,
                        "{label} pixel 2 (200 nits UI) ch {ch}: {:?}",
                        pixels[2]
                    );
                }
            }
            assert_eq!(host.log_messages(), None, "D3D12 debug layer reported an error");
        }
    }

    #[test]
    fn composer_rejects_mismatched_target_format_or_display_color_space() {
        let Some(host) = crate::host::host() else {
            eprintln!("skipping composer format-pairing test: no D3D12 device");
            return;
        };
        let sdr_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pq_fmt = DXGI_FORMAT_R10G10B10A2_UNORM;
        let scrgb_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;

        let mut sdr_comp = Composer::new(host, sdr_fmt).unwrap();
        let mut pq_comp = Composer::new(host, pq_fmt).unwrap();
        assert_eq!(sdr_comp.format(), sdr_fmt);
        assert_eq!(pq_comp.format(), pq_fmt);

        let src = make_rt(host, 1, scrgb_fmt);
        let sdr_target = make_rt(host, 1, sdr_fmt);
        let pq_target = make_rt(host, 1, pq_fmt);
        let list = host.new_list().unwrap();
        let cmds = &list.list;

        // 1. Target RTV format does not match composer PSO format -> rejected.
        assert!(
            pq_comp
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Scene((&src, scrgb_fmt)),
                    &sdr_target,
                    (1, 1),
                    ComposeParams {
                        scene_space: ColorSpace::ScRgb,
                        output_space: ColorSpace::Hdr10Pq,
                        paper_white_nits: 200.0,
                    },
                )
                .is_err()
        );
        // 2. Display color space incompatible with swap-chain format -> rejected.
        for bad_space in [ColorSpace::Hdr10Pq, ColorSpace::ScRgb] {
            assert!(
                sdr_comp
                    .record(
                        host,
                        0,
                        cmds,
                        Picture::Scene((&src, scrgb_fmt)),
                        &sdr_target,
                        (1, 1),
                        ComposeParams {
                            scene_space: ColorSpace::ScRgb,
                            output_space: bad_space,
                            paper_white_nits: 200.0,
                        },
                    )
                    .is_err(),
                "SDR target must reject {bad_space:?}"
            );
        }
        assert!(
            pq_comp
                .record(
                    host,
                    0,
                    cmds,
                    Picture::Scene((&src, scrgb_fmt)),
                    &pq_target,
                    (1, 1),
                    ComposeParams {
                        scene_space: ColorSpace::ScRgb,
                        output_space: ColorSpace::ScRgb,
                        paper_white_nits: 200.0,
                    },
                )
                .is_err(),
            "R10G10B10A2 target must reject ScRgb"
        );
        // 3. Valid SDR control fallback on the same HDR scene succeeds cleanly.
        // SAFETY: transitioning test source texture to SRV and recording valid SDR draw.
        unsafe {
            cmds.ResourceBarrier(&[crate::host::transition(
                &src,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )]);
        }
        sdr_comp
            .record(
                host,
                0,
                cmds,
                Picture::Scene((&src, scrgb_fmt)),
                &sdr_target,
                (1, 1),
                ComposeParams {
                    scene_space: ColorSpace::ScRgb,
                    output_space: ColorSpace::Sdr,
                    paper_white_nits: 200.0,
                },
            )
            .unwrap();
        list.close();
        host.execute(&[&list]);
        // SAFETY: checking device health after execution.
        unsafe {
            host.device.GetDeviceRemovedReason().unwrap();
        }
        assert_eq!(host.log_messages(), None, "D3D12 debug layer reported an error");
    }
}
