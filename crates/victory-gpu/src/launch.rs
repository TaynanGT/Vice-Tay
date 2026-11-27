//! Mesh draws launched on the GPU by an amplification stage
//! (`victory_shader::emit::mesh_launch_as`), eliminating the compute clamp
//! pass and barriers before single-record indirect mesh draws
//! (`VICTORY_MESH_LAUNCH`, on by default; `0` off; `alt` alternates every 20 s).
//!
//! Ported from fxceless/victory-fork (`b1d777a`, `0e006f0`) and kelvaroq
//! (`0041-mesh-launch-wide-hash.patch`), generating the `as_6_6` amplification
//! shader programmatically through `victory-shader` so no precompiled DXIL
//! blob is committed and runtime clamping matches `crate::indirect` (including
//! unfilled-slot zeroing above 65,535 groups, `VICTORY_INDIRECT_REST`, and
//! budget diagnostics).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;

/// Root constant value instructing the amplification stage to read the indirect
/// argument record from `t0, space9` instead of using a CPU-known count.
pub(crate) const DIRECT_NONE: u32 = u32::MAX;

/// Register space where the amplification stage binds `b0`, `t0`, `u0`.
const SPACE: u32 = victory_shader::emit::LAUNCH_SPACE;

/// `VICTORY_MESH_LAUNCH`: 0 off, 1 on (default), 2 alternating 20 s on/off (`alt`).
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_MESH_LAUNCH").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        Some("alt") => 2,
        _ => 1,
    })
}

/// Whether mesh pipelines should be built with the amplification launch stage.
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::ab::bit("VICTORY_MESH_LAUNCH").is_some() {
        return true;
    }
    mode() != 0
}

/// Keeps `victory_shader::emit::mesh_payload_bytes()` aligned with [`on`]
/// (including `VICTORY_AB` overrides under diagnostics builds).
pub(crate) fn sync_shader_payload() {
    victory_shader::emit::set_mesh_launch_payload(on());
}

/// Whether a mesh draw should currently use the amplification launch stage
/// (supports `VICTORY_MESH_LAUNCH=alt` and `VICTORY_AB` toggling by falling
/// back to `HostPipeline::launch.plain()` in off phases).
pub(crate) fn now() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(active) = crate::ab::pick("VICTORY_MESH_LAUNCH") {
        return active;
    }
    match mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let active = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(active), Ordering::Relaxed) != u8::from(active) {
                tracing::info!(
                    "mesh draws from now (VICTORY_MESH_LAUNCH=alt): {}",
                    if active {
                        "ON, launched by the amplification stage"
                    } else {
                        "off, clamped before each record"
                    }
                );
            }
            active
        }
    }
}

/// Cached DXIL bytes for the amplification launch stage.
pub(crate) fn bytes() -> &'static [u8] {
    static DXIL: OnceLock<Vec<u8>> = OnceLock::new();
    DXIL.get_or_init(|| victory_shader::emit::mesh_launch_as(crate::indirect::rest_on()))
}

/// D3D12 shader bytecode descriptor for [`bytes`].
pub(crate) fn code() -> D3D12_SHADER_BYTECODE {
    let b = bytes();
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: b.as_ptr().cast(),
        BytecodeLength: b.len(),
    }
}

/// Root parameters for the amplification launch stage (5 root dwords):
/// `b0` (direct count constant), `t0` (indirect record root SRV), and
/// `u0` (list budget root UAV), all in [`SPACE`].
pub(crate) fn params() -> [D3D12_ROOT_PARAMETER1; 3] {
    let descriptor = |kind| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: 0,
                RegisterSpace: SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
    };
    [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: SPACE,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_UAV),
    ]
}

/// Mesh draws launched since the last frame summary: `[indirect_records, direct_counts]`.
static LAUNCHED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What a single amplification-stage launch dispatches.
#[derive(Clone, Copy)]
pub(crate) enum Launch {
    /// Indirect 3-dword dispatch record at this GPU virtual address.
    Record(u64),
    /// Direct X group count known on the CPU.
    Direct(u32),
}

/// Small read-only upload buffer kept permanently in `GENERIC_READ` so
/// [`Launch::Direct`] binds a valid non-null SRV distinct from the list's
/// `UNORDERED_ACCESS` budget buffer (avoiding a D3D12 SRV+UAV state conflict).
struct DummyRecord {
    buffer: ID3D12Resource,
    gpu: u64,
}

// SAFETY: D3D12 COM resources are free-threaded and this buffer is read-only after init.
unsafe impl Send for DummyRecord {}
// SAFETY: as above.
unsafe impl Sync for DummyRecord {}

fn dummy_record() -> Option<&'static DummyRecord> {
    static DUMMY: OnceLock<Option<DummyRecord>> = OnceLock::new();
    DUMMY
        .get_or_init(|| {
            let host = crate::host::host()?;
            let (buffer, data) = host.upload_buffer(16)?;
            crate::host::name(&buffer, || "mesh launch dummy record".into());
            // SAFETY: 16 mapped bytes of a live upload buffer.
            let gpu = unsafe {
                std::ptr::write_bytes(data, 0, 16);
                buffer.GetGPUVirtualAddress()
            };
            (gpu != 0).then_some(DummyRecord { buffer, gpu })
        })
        .as_ref()
}

/// Records a single amplification-launched mesh dispatch on `list` using the
/// stage's root parameter indices `roots`. Returns `false` if `list` lacks
/// `ID3D12GraphicsCommandList6` or live non-zero GPU addresses for the bound
/// root descriptors.
pub(crate) fn record(list: &HostList, roots: [u32; 3], what: Launch) -> bool {
    use windows::core::Interface;
    let l6 = match list.list.cast::<ID3D12GraphicsCommandList6>() {
        Ok(l6) => l6,
        Err(_) => return false,
    };
    let Some(budget) = list.budget().filter(|&va| va != 0) else {
        return false;
    };
    let (direct, record_addr) = match what {
        Launch::Record(address) => {
            if address == 0 {
                return false;
            }
            (DIRECT_NONE, address)
        }
        // Direct count never dereferences `t0` or `u0`, but D3D12 validates
        // bound root descriptors unconditionally: bind a dedicated read-only
        // `GENERIC_READ` dummy buffer to `roots[1]` so it never conflicts with
        // `budget`'s `UNORDERED_ACCESS` state on `roots[2]`.
        Launch::Direct(x) => {
            let Some(dummy) = dummy_record() else {
                return false;
            };
            list.keep(&dummy.buffer);
            (x.min(65_535), dummy.gpu)
        }
    };
    // SAFETY: open command list with the amplification-enabled mesh pipeline
    // and matching root signature bound; both buffers are live non-zero GPU
    // addresses kept alive by `list` in compatible states (`SRV` vs `UAV`).
    unsafe {
        list.list.SetGraphicsRoot32BitConstant(roots[0], direct, 0);
        list.list.SetGraphicsRootShaderResourceView(roots[1], record_addr);
        list.list.SetGraphicsRootUnorderedAccessView(roots[2], budget);
        l6.DispatchMesh(1, 1, 1);
    }
    LAUNCHED[usize::from(matches!(what, Launch::Direct(_)))].fetch_add(1, Ordering::Relaxed);
    true
}

/// Logs per-frame launched mesh draw counts for the periodic frame summary.
pub(crate) fn line(frames: f64) {
    let [records, direct] = [0, 1].map(|i| LAUNCHED[i].swap(0, Ordering::Relaxed));
    if records + direct > 0 {
        tracing::info!(
            "mesh draws launched by amplification stage per frame (VICTORY_MESH_LAUNCH): {:.0} indirect records, {:.0} direct counts",
            records as f64 / frames,
            direct as f64 / frames
        );
    }
}
