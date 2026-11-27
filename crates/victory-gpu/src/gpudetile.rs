//! Texture detiling on the GPU (`VICTORY_GPU_DETILE`).
//!
//! Without it, a title texture's memory is detiled on the CPU into its staging
//! buffer (`crate::texture`, `Staging::detile` via `victory_formats::detile`).
//! With it, the staging buffer holds the texture's raw tiled bytes (one plain
//! CPU copy, followed by the held-level bitmask and the layout's pattern
//! tables), and a compute shader in the command list's prelude
//! (`HostList::prelude`) detiles them on the GPU into a device-local linear
//! buffer from which the list's `CopyTextureRegion` commands read.
//!
//! Supported layouts match [`victory_formats::texture::Layout::detile`]:
//! linear and every thin/thick single-sampled non-XOR swizzle mode, including
//! mip tails, arrays, 3D textures, held-level sub-ranges (`first > 0` /
//! `level0 > 0`), and on-demand zeroing of unheld mip levels. Unsupported
//! modes (such as XOR swizzles), background copy-queue uploads (`ahead`), and
//! allocation-pressured cases fall back automatically to CPU detiling.
//!
//! Adapted from the GPU prelude detiling design in `fxceless/victory-fork`
//! (`3461f52`), extended for pooled staging base offsets, held-level mip
//! clamping (`first`/`level0`), exact 4-byte-rounded UAV sizing, and
//! division-free power-of-two shift/mask addressing.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use victory_formats::texture::Layout;
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(test)]
thread_local! {
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Temporarily overrides `VICTORY_GPU_DETILE` on the current thread for tests.
#[cfg(test)]
pub(crate) fn with_enabled<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_OVERRIDE.set(self.0);
        }
    }
    let _reset = Reset(TEST_OVERRIDE.replace(Some(enabled)));
    f()
}

/// `VICTORY_GPU_DETILE`: unset/`0` off; `1` (or other non-zero) on; `alt`
/// toggles off/on every 20 seconds, logging each transition. Checked when a
/// texture's staging state is created; each staging keeps the mode it was
/// created with.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(forced) = TEST_OVERRIDE.get() {
        return forced;
    }
    // 0: off, 1: on, 2: alternating 20 s windows.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_GPU_DETILE").as_deref() {
        Ok("0") | Err(_) => 0,
        Ok("alt") => 2,
        Ok(_) => 1,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let active = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(active), Relaxed) != u8::from(active) {
                tracing::info!(
                    "title textures made from now on are detiled {} (VICTORY_GPU_DETILE=alt)",
                    if active { "on the GPU" } else { "on the CPU" }
                );
            }
            active
        }
    }
}

/// Number of 32-bit root constants per subresource dispatch (matches [`SOURCE`]).
pub(crate) const CONSTANTS: usize = 24;

/// Compute shader (`cs_5_1`) with embedded root signature.
///
/// Each thread writes one 32-bit word of a row in the linear destination
/// buffer: either one word of a 4/8/16-byte element or 2/4 packed 1/2-byte
/// elements. Root SRV `t0` starts at the staging piece's GPU virtual address
/// (raw tiled bytes at offset 0, held-level bitmask at `mask_at`, x/y/z
/// pattern tables at `tables_x`/`tables_y`/`tables_z`); root UAV `u0` is the
/// device-local linear buffer.
///
/// All block dimensions, slabs, and element sizes are powers of two, so every
/// coordinate quotient and remainder uses shift/mask with zero integer divides.
pub(crate) const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=24, b0), SRV(t0), UAV(u0)"
cbuffer P : register(b0) {
    uint flags;        // bit 0: linear
    uint n_log2;       // log2(element_bytes): 0..4
    uint width;        // subresource width in elements
    uint height;       // subresource height in elements
    uint depth;        // subresource depth (3D) or 1 (2D/array)
    uint first;        // array slice (0 for 3D)
    uint slab_log2;    // log2(depth slices per layout slice)
    uint slice_size;   // bytes per layout slice
    uint mip_offset;   // mip offset within slice
    uint pitch;        // linear: padded width in elements; tiled: pitch / block_w
    uint bw_log2;      // log2(block_w)
    uint bh_log2;      // log2(block_h)
    uint block_log2;   // log2(block_bytes)
    uint tx;           // tail x origin in elements
    uint ty;           // tail y origin in elements
    uint tables_x;     // byte offset of x pattern table in src
    uint tables_y;     // byte offset of y pattern table in src
    uint tables_z;     // byte offset of z pattern table in src
    uint dst;          // byte offset of subresource in out_buf
    uint row_pitch;    // destination row pitch in bytes
    uint depth_pitch;  // destination depth-slice pitch in bytes
    uint words;        // 32-bit words per row (ceil(width * n / 4))
    uint mask_at;      // byte offset of held-level bitmask in src
    uint mip;          // console mip index
};
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer out_buf : register(u0);

uint element(uint x, uint y, uint z) {
    uint z_abs = first + z;
    uint s = z_abs >> slab_log2;
    uint inner = z_abs & ((1u << slab_log2) - 1u);
    uint base = s * slice_size + mip_offset;
    if ((flags & 1u) != 0u) {
        return base + ((y * pitch + x) << n_log2);
    }
    x += tx;
    y += ty;
    uint row = ((y >> bh_log2) * pitch) << block_log2;
    uint col = (x >> bw_log2) << block_log2;
    uint px = src.Load(tables_x + ((x & ((1u << bw_log2) - 1u)) << 2u));
    uint py = src.Load(tables_y + ((y & ((1u << bh_log2) - 1u)) << 2u));
    uint pz = src.Load(tables_z + (inner << 2u));
    return base + row + col + (py | pz) + px;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= words || id.y >= height || id.z >= depth) return;
    uint at = dst + id.z * depth_pitch + id.y * row_pitch + (id.x << 2u);
    uint held = src.Load(mask_at);
    if (mip < 32u && ((held >> mip) & 1u) == 0u) {
        out_buf.Store(at, 0u);
        return;
    }
    uint v = 0u;
    if (n_log2 >= 2u) {
        uint per_log2 = n_log2 - 2u;
        uint elem_x = id.x >> per_log2;
        uint word_off = (id.x & ((1u << per_log2) - 1u)) << 2u;
        v = src.Load(element(elem_x, id.y, id.z) + word_off);
    } else {
        uint per = 4u >> n_log2;
        uint keep = n_log2 == 0u ? 0xffu : 0xffffu;
        uint shift_step = 8u << n_log2;
        [unroll]
        for (uint k = 0u; k < 4u; k++) {
            uint x = id.x * per + k;
            if (k < per && x < width) {
                uint a = element(x, id.y, id.z);
                uint w = src.Load(a & ~3u);
                v |= ((w >> ((a & 3u) << 3u)) & keep) << (k * shift_step);
            }
        }
    }
    out_buf.Store(at, v);
}
"#;

/// Compiled root signature and compute PSO for texture detiling.
pub(crate) struct Detiler {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) pso: ID3D12PipelineState,
}

// SAFETY: immutable D3D12 objects, which are free-threaded.
unsafe impl Send for Detiler {}
// SAFETY: immutable D3D12 objects, which are free-threaded.
unsafe impl Sync for Detiler {}

/// Compiles [`SOURCE`] (`cs_5_1`, with embedded root signature) on `device`.
pub(crate) fn make(device: &ID3D12Device) -> windows::core::Result<Detiler> {
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: source and static names outlive `D3DCompile`; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            windows::core::s!("texture detile"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob owns its diagnostic bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("the texture detile shader: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: bytecode blob owns its bytes and embeds the root signature.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Detiler { root, pso })
    }
}

/// Host-wide detiler, built on first use; `None` (logged once) if compilation
/// or PSO creation fails, in which case textures fall back to CPU detiling.
pub(crate) fn detiler() -> Option<&'static Detiler> {
    static MADE: OnceLock<Option<Detiler>> = OnceLock::new();
    MADE.get_or_init(|| {
        let host = crate::host::host()?;
        match make(&host.device) {
            Ok(d) => {
                tracing::info!("title textures are detiled on the GPU (VICTORY_GPU_DETILE)");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "VICTORY_GPU_DETILE: compute detile pipeline could not be created; falling back to CPU detiling"
                );
                None
            }
        }
    })
    .as_ref()
}

/// Layout inside a GPU-detiled staging piece: raw tiled bytes `[0..tiled_bytes)`,
/// 16-byte-aligned held-level mask at `mask_at`, pattern tables at `tables_at`,
/// and total staging bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Raw {
    pub(crate) tiled_bytes: u64,
    pub(crate) mask_at: u64,
    pub(crate) tables_at: u64,
    pub(crate) bytes: u64,
}

/// Computes the upload staging layout for `tiled_bytes` of raw tiled data plus
/// the held-level bitmask and `layout`'s pattern tables.
pub(crate) fn raw_layout(layout: &Layout, tiled_bytes: u64) -> Raw {
    let mask_at = tiled_bytes.next_multiple_of(16);
    let tables_at = mask_at + 16;
    let words: usize = layout.pattern_tables().iter().map(|t| t.len()).sum();
    Raw {
        tiled_bytes,
        mask_at,
        tables_at,
        bytes: tables_at + 4 * words as u64,
    }
}

/// Writes `layout`'s pattern tables and an all-held mask into the mapped
/// staging piece `data`.
///
/// # Safety
/// `data` points to at least `raw.bytes` writable bytes with exclusive CPU access.
pub(crate) unsafe fn write_tables(layout: &Layout, raw: &Raw, data: *mut u8) {
    let mut at = raw.tables_at as usize;
    // SAFETY: bounds guaranteed by caller matching `raw_layout`.
    unsafe {
        write_mask(raw, data, u32::MAX);
        for table in layout.pattern_tables() {
            for &v in table {
                data.add(at).cast::<u32>().write_unaligned(v);
                at += 4;
            }
        }
    }
}

/// Writes the held-level bitmask (bit `m` = console mip `m` held; levels >= 32
/// are always treated as held).
///
/// # Safety
/// `data` points to at least `raw.bytes` writable bytes with exclusive CPU access.
pub(crate) unsafe fn write_mask(raw: &Raw, data: *mut u8, held: u32) {
    // SAFETY: `raw.mask_at + 4 <= raw.bytes`.
    unsafe { data.add(raw.mask_at as usize).cast::<u32>().write_unaligned(held) };
}

/// One subresource's compute dispatch parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Job {
    pub(crate) constants: [u32; CONSTANTS],
    pub(crate) groups: [u32; 3],
}

/// All subresource dispatches for a texture plus the minimum 4-byte-aligned
/// byte size the destination UAV buffer must have so every `Store` stays
/// strictly in bounds (including 1-byte / 2-byte last rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) jobs: Vec<Job>,
    pub(crate) linear_bytes: u64,
}

/// Builds compute dispatches for every subresource in `footprints`/`rows`,
/// where entry `i` covers console mip `first + i % (mips - first)` and slice
/// `i / (mips - first)`. Returns `None` if any subresource cannot be detiled
/// safely on the GPU (causing automatic fallback to CPU detiling).
pub(crate) fn jobs(
    layout: &Layout,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    first: u32,
    mips: u32,
    raw: &Raw,
) -> Option<Plan> {
    if footprints.is_empty()
        || footprints.len() != rows.len()
        || first >= mips
        || mips as usize > layout.mips.len()
        || raw.bytes > u64::from(u32::MAX)
    {
        return None;
    }
    let tables_at = u32::try_from(raw.tables_at).ok()?;
    let mask_at = u32::try_from(raw.mask_at).ok()?;
    if !tables_at.is_multiple_of(4) || !mask_at.is_multiple_of(4) {
        return None;
    }
    let span = mips - first;
    let mut out = Vec::with_capacity(footprints.len());
    let mut max_written_end = 0u64;
    for (i, (footprint, &row_count)) in footprints.iter().zip(rows).enumerate() {
        let mip = first + (i as u32 % span);
        let slice = i as u32 / span;
        if layout.bytes_from(mip) > raw.tiled_bytes {
            return None;
        }
        let p = layout.detile_params(mip, slice)?;
        let f = footprint.Footprint;
        if !footprint.Offset.is_multiple_of(4) || !f.RowPitch.is_multiple_of(4) {
            return None;
        }
        let row_bytes = u64::from(p.width).checked_mul(u64::from(p.element_bytes))?;
        let words = row_bytes.div_ceil(4);
        let depth_pitch = u64::from(f.RowPitch).checked_mul(u64::from(row_count))?;
        let last_row = footprint
            .Offset
            .checked_add(u64::from(p.depth - 1).checked_mul(depth_pitch)?)?
            .checked_add(u64::from(p.height - 1).checked_mul(u64::from(f.RowPitch))?)?;
        let written_end = last_row.checked_add(words.checked_mul(4)?)?;
        let footprint_end = footprint
            .Offset
            .checked_add(depth_pitch.checked_mul(u64::from(f.Depth))?)?;
        let groups_x = words.div_ceil(64);
        if row_bytes > u64::from(f.RowPitch)
            || words * 4 > u64::from(f.RowPitch)
            || p.height > row_count
            || p.depth > f.Depth
            || written_end > u64::from(u32::MAX)
            || footprint_end > u64::from(u32::MAX)
            || words == 0
            || groups_x > 65535
            || p.height > 65535
            || p.depth > 65535
        {
            return None;
        }
        let (pitch_const, bw_log2, bh_log2, tables_x, tables_y, tables_z) = if p.linear {
            (p.pitch, 0, 0, tables_at, tables_at, tables_at)
        } else {
            let ty = tables_at.checked_add(p.block_w.checked_mul(4)?)?;
            let tz = ty.checked_add(p.block_h.checked_mul(4)?)?;
            let tables_end = tz.checked_add(p.slab.checked_mul(4)?)?;
            if u64::from(tables_end) > raw.bytes {
                return None;
            }
            (
                p.pitch / p.block_w,
                p.block_w.trailing_zeros(),
                p.block_h.trailing_zeros(),
                tables_at,
                ty,
                tz,
            )
        };
        max_written_end = max_written_end.max(written_end);
        out.push(Job {
            constants: [
                u32::from(p.linear),
                p.element_bytes.trailing_zeros(),
                p.width,
                p.height,
                p.depth,
                p.first,
                p.slab.trailing_zeros(),
                p.slice_size,
                p.mip_offset,
                pitch_const,
                bw_log2,
                bh_log2,
                p.block_log2,
                p.tail_x,
                p.tail_y,
                tables_x,
                tables_y,
                tables_z,
                footprint.Offset as u32,
                f.RowPitch,
                depth_pitch as u32,
                words as u32,
                mask_at,
                mip,
            ],
            groups: [groups_x as u32, p.height, p.depth],
        });
    }
    Some(Plan {
        jobs: out,
        linear_bytes: max_written_end.next_multiple_of(4),
    })
}

/// Records the compute detiling pass from `raw_gpu_va` (an upload-heap staging
/// slice in `GENERIC_READ`, aligned to at least 4 bytes) into `linear` (a
/// device-local UAV buffer resting in `COMMON` between submissions).
///
/// Transitions `linear` `COMMON -> UNORDERED_ACCESS`, runs all subresource
/// dispatches, and transitions `linear` back `UNORDERED_ACCESS -> COMMON` so
/// subsequent `CopyTextureRegion` commands promote it implicitly to
/// `COPY_SOURCE` and decay it back to `COMMON` at the end of the execution.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    d: &Detiler,
    raw_gpu_va: u64,
    linear: &ID3D12Resource,
    jobs: &[Job],
) {
    use crate::host::transition;
    debug_assert!(raw_gpu_va.is_multiple_of(4));
    // SAFETY: open direct command list; live buffers kept alive through execution;
    // root signature and PSO match the 24 root constants + root SRV + root UAV.
    unsafe {
        list.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        list.SetComputeRootSignature(&d.root);
        list.SetPipelineState(&d.pso);
        list.SetComputeRootShaderResourceView(1, raw_gpu_va);
        list.SetComputeRootUnorderedAccessView(2, linear.GetGPUVirtualAddress());
        for job in jobs {
            list.SetComputeRoot32BitConstants(0, CONSTANTS as u32, job.constants.as_ptr().cast(), 0);
            list.Dispatch(job.groups[0], job.groups[1], job.groups[2]);
        }
        list.ResourceBarrier(&[transition(
            linear,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
}

/// Since the last frame-rate line:
/// `[0]` textures detiled on GPU, `[1]` compute dispatches recorded,
/// `[2]` raw tiled bytes copied to upload staging on CPU,
/// `[3]` CPU copy microseconds, `[4]` stagings created for GPU detile,
/// `[5]` stagings that fell back to CPU detile.
pub(crate) static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Emits the per-frame texture detiling meter line when either CPU or GPU
/// detiling moved bytes since the last frame-rate report.
pub(crate) fn meter_line(frames: f64) {
    let [cpu_us, cpu_bytes, cpu_textures] =
        [0, 1, 2].map(|i| crate::texture::DETILED[i].swap(0, Relaxed) as f64 / frames);
    let [gpu_textures, gpu_dispatches, gpu_bytes, gpu_us, gpu_made, gpu_fallback] =
        std::array::from_fn::<f64, 6, _>(|i| STATS[i].swap(0, Relaxed) as f64 / frames);
    let modes: Vec<String> = crate::texture::BY_MODE
        .iter()
        .enumerate()
        .filter_map(|(mode, bytes)| {
            let mb = bytes.swap(0, Relaxed) as f64 / frames / 1e6;
            (mb >= 0.05).then(|| format!("{} {mb:.1}", victory_formats::texture::Swizzle(mode as u32).name()))
        })
        .collect();
    if cpu_bytes + gpu_bytes > 0.0 {
        tracing::info!(
            "texture detiling per frame: CPU {:.2} ms for {:.1} MB ({cpu_textures:.1} textures); GPU \
             (VICTORY_GPU_DETILE={}) {gpu_textures:.1} textures in {gpu_dispatches:.0} dispatches, CPU raw copy \
             {:.1} MB in {:.2} ms, {gpu_made:.1} GPU stagings, {gpu_fallback:.1} CPU fallbacks; \
             MB by swizzle mode: {}",
            cpu_us / 1e3,
            cpu_bytes / 1e6,
            std::env::var("VICTORY_GPU_DETILE").unwrap_or_else(|_| "0".into()),
            gpu_bytes / 1e6,
            gpu_us / 1e3,
            modes.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Dimension, Swizzle, Texture, TextureDesc};
    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter, IDXGIFactory1,
    };
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;

    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        event: HANDLE,
        submitted: u64,
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            // SAFETY: event created by `Rig::from_adapter`.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }

    impl Rig {
        fn warp() -> Option<Rig> {
            let adapter = crate::host::warp_adapter().ok()?;
            Self::from_adapter(&adapter)
        }

        fn hardware() -> Option<Rig> {
            // SAFETY: standard DXGI factory & adapter enumeration.
            unsafe {
                let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
                let mut i = 0;
                while let Ok(adapter) = factory.EnumAdapters1(i) {
                    i += 1;
                    let Ok(desc) = adapter.GetDesc1() else {
                        continue;
                    };
                    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                        continue;
                    }
                    let base: IDXGIAdapter = adapter.cast().ok()?;
                    if let Some(rig) = Self::from_adapter(&base) {
                        return Some(rig);
                    }
                }
            }
            None
        }

        fn from_adapter(adapter: &IDXGIAdapter) -> Option<Rig> {
            // SAFETY: standalone D3D12 test device and objects.
            unsafe {
                if crate::switch::on("VICTORY_D3D12_DEBUG") {
                    let mut debug: Option<ID3D12Debug> = None;
                    if D3D12GetDebugInterface(&mut debug).is_ok()
                        && let Some(debug) = debug
                    {
                        debug.EnableDebugLayer();
                    }
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Rig {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    event,
                    submitted: 0,
                })
            }
        }

        fn run(&mut self, record_fn: impl FnOnce(&ID3D12GraphicsCommandList)) {
            record_fn(&self.list);
            self.submitted += 1;
            // SAFETY: owned list, queue, fence, event; allocator reset only after wait.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                assert_eq!(WaitForSingleObject(self.event, 60_000), WAIT_OBJECT_0, "GPU detile timed out");
                let removed = self.device.GetDeviceRemovedReason();
                assert!(removed.is_ok(), "device lost: {removed:?}");
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        fn buffer(
            &self,
            kind: D3D12_HEAP_TYPE,
            flags: D3D12_RESOURCE_FLAGS,
            state: D3D12_RESOURCE_STATES,
            bytes: u64,
        ) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes.max(1),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: plain committed buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                    .unwrap();
            }
            out.unwrap()
        }
    }

    fn map(buffer: &ID3D12Resource) -> *mut u8 {
        let mut data = std::ptr::null_mut();
        // SAFETY: CPU-visible upload or readback buffer owned by the test.
        unsafe { buffer.Map(0, None, Some(&mut data)).unwrap() };
        data.cast()
    }

    struct Case {
        dimension: Dimension,
        width: u32,
        height: u32,
        depth: u32,
        mips: u32,
        format: DXGI_FORMAT,
        swizzle: Swizzle,
        /// Simulated `copy.base` (first console mip present in host texture).
        level0: u32,
        /// First console mip uploaded (`>= level0`).
        first: u32,
        /// Non-zero staging pool base offset to verify `raw_gpu_va` offsetting.
        raw_base: u64,
    }

    fn case2d(width: u32, height: u32, mips: u32, format: DXGI_FORMAT, swizzle: Swizzle) -> Case {
        Case {
            dimension: Dimension::Tex2d,
            width,
            height,
            depth: 1,
            mips,
            format,
            swizzle,
            level0: 0,
            first: 0,
            raw_base: 0,
        }
    }

    /// Compares GPU compute detiling against CPU `Layout::detile` byte for byte
    /// across all active row bytes of every uploaded subresource.
    fn verify_case(rig: &mut Rig, d: &Detiler, case: &Case, held: u32) -> bool {
        let texture = Texture::new(&TextureDesc {
            dimension: case.dimension,
            width: case.width,
            height: case.height,
            depth_or_array: case.depth,
            mips: case.mips,
            format: case.format.0 as u32,
            samples: 1,
            swizzle: case.swizzle,
        })
        .unwrap();
        let layout = texture.planes[0].layout.clone();
        let three_d = case.dimension == Dimension::Tex3d;
        let host_w = (case.width >> case.level0).max(1);
        let host_h = (case.height >> case.level0).max(1);
        let host_mips = case.mips - case.level0;
        let host_desc = D3D12_RESOURCE_DESC {
            Dimension: if three_d {
                D3D12_RESOURCE_DIMENSION_TEXTURE3D
            } else {
                D3D12_RESOURCE_DIMENSION_TEXTURE2D
            },
            Width: u64::from(host_w),
            Height: host_h,
            DepthOrArraySize: case.depth as u16,
            MipLevels: host_mips as u16,
            Format: case.format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let span = case.mips - case.first;
        let count = if three_d { span } else { span * case.depth };
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
        let mut rows = vec![0u32; count as usize];
        let mut total = 0u64;
        // SAFETY: output slices hold `count` entries.
        unsafe {
            rig.device.GetCopyableFootprints(
                &host_desc,
                case.first - case.level0,
                count,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                None,
                Some(&mut total),
            );
        }
        let read_bytes = layout.bytes_from(case.first);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ u64::from(case.width * 31 + case.height + case.first * 17);
        let tiled: Vec<u8> = (0..read_bytes)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 24) as u8
            })
            .collect();
        // CPU reference (`victory-formats` AVX2/SSE2/portable/thick detiler).
        let mut cpu = vec![0xA5u8; total as usize + 4096];
        for (i, (footprint, &r)) in footprints.iter().zip(&rows).enumerate() {
            let mip = case.first + (i as u32 % span);
            let slice = i as u32 / span;
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * r as usize * f.Depth as usize;
            let linear = &mut cpu[footprint.Offset as usize..][..len];
            if mip < 32 && held & (1 << mip) == 0 {
                linear.fill(0);
                continue;
            }
            layout
                .detile(&tiled, mip, slice, linear, pitch, pitch * r as usize)
                .unwrap();
        }
        // GPU compute detile.
        let raw = raw_layout(&layout, read_bytes);
        let plan = jobs(&layout, &footprints, &rows, case.first, case.mips, &raw)
            .expect("addressable case must produce GPU detile jobs");
        let upload_bytes = case.raw_base + raw.bytes;
        let upload = rig.buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            upload_bytes,
        );
        let base_ptr = map(&upload);
        // SAFETY: `upload` has `case.raw_base + raw.bytes` mapped bytes.
        unsafe {
            // Fill prefix before `raw_base` with poison bytes to prove offsetting.
            std::ptr::write_bytes(base_ptr, 0xCC, case.raw_base as usize);
            let data = base_ptr.add(case.raw_base as usize);
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), data, tiled.len());
            write_tables(&layout, &raw, data);
            write_mask(&raw, data, held);
        }
        let linear_bytes = total.max(plan.linear_bytes).next_multiple_of(4);
        let linear = rig.buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            linear_bytes,
        );
        let back = rig.buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            linear_bytes,
        );
        // SAFETY: live upload buffer.
        let raw_gpu_va = unsafe { upload.GetGPUVirtualAddress() } + case.raw_base;
        rig.run(|list| {
            record(list, d, raw_gpu_va, &linear, &plan.jobs);
            // SAFETY: open list; buffers live through execution; `linear` promotes
            // from `COMMON` to `COPY_SOURCE` for the readback copy.
            unsafe { list.CopyBufferRegion(&back, 0, &linear, 0, linear_bytes) };
        });
        let read = map(&back);
        // SAFETY: readback buffer holds `linear_bytes` bytes written by the copy.
        let gpu = unsafe { std::slice::from_raw_parts(read.cast_const(), linear_bytes as usize) };
        let mut same = true;
        for (i, (footprint, &r)) in footprints.iter().zip(&rows).enumerate() {
            let mip = case.first + (i as u32 % span);
            let m = &layout.mips[mip as usize];
            let n = layout.desc.element_bytes as usize;
            let f = footprint.Footprint;
            for z in 0..m.depth as usize {
                for y in 0..m.height as usize {
                    let at = footprint.Offset as usize
                        + z * f.RowPitch as usize * r as usize
                        + y * f.RowPitch as usize;
                    let own = m.width as usize * n;
                    if cpu[at..at + own] != gpu[at..at + own] {
                        if same {
                            eprintln!(
                                "{:?} {}x{}x{} {:?} (level0={}, first={}, raw_base={}): subresource {i} (mip {mip}) row {y} depth {z} differs",
                                case.swizzle,
                                case.width,
                                case.height,
                                case.depth,
                                case.format,
                                case.level0,
                                case.first,
                                case.raw_base
                            );
                        }
                        same = false;
                    }
                }
            }
        }
        same
    }

    fn run_matrix(rig: &mut Rig, label: &str) {
        let d = make(&rig.device).expect("detile compute shader compiles and creates PSO");
        let cases = [
            // BC7 POT & NPOT
            case2d(1024, 1024, 11, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_S),
            case2d(140, 84, 6, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_D),
            // BC1 POT, NPOT, array
            case2d(512, 256, 10, DXGI_FORMAT_BC1_UNORM, Swizzle::SW_64KB_S),
            case2d(100, 60, 5, DXGI_FORMAT_BC1_UNORM, Swizzle::SW_64KB_D),
            Case { depth: 6, ..case2d(128, 128, 8, DXGI_FORMAT_BC1_UNORM, Swizzle::SW_4KB_S) },
            // BC3 POT & NPOT
            case2d(256, 256, 9, DXGI_FORMAT_BC3_UNORM, Swizzle::SW_4KB_S),
            case2d(132, 68, 6, DXGI_FORMAT_BC3_UNORM, Swizzle::SW_64KB_S),
            // BC5 NPOT
            case2d(100, 60, 7, DXGI_FORMAT_BC5_UNORM, Swizzle::SW_4KB_S),
            // RGBA8 POT, NPOT, array, linear, pooled raw_base offset
            case2d(256, 256, 9, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S),
            case2d(300, 200, 9, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S),
            case2d(129, 65, 8, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_4KB_S),
            Case { depth: 3, ..case2d(70, 70, 4, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S) },
            case2d(130, 70, 8, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::LINEAR),
            // 1B, 2B, 8B, 16B plain elements + 256B micro-tile modes + unaligned 1x1 tail
            case2d(97, 61, 7, DXGI_FORMAT_R8_UNORM, Swizzle::SW_64KB_S),
            case2d(33, 17, 6, DXGI_FORMAT_R8_UNORM, Swizzle::SW_256B_D),
            case2d(61, 3, 6, DXGI_FORMAT_R8_UNORM, Swizzle::LINEAR),
            case2d(77, 33, 7, DXGI_FORMAT_R16_UNORM, Swizzle::SW_4KB_D),
            case2d(64, 64, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, Swizzle::SW_64KB_D),
            case2d(50, 30, 6, DXGI_FORMAT_R32G32B32A32_FLOAT, Swizzle::SW_256B_S),
            // 3D thick & linear
            Case { dimension: Dimension::Tex3d, depth: 16, ..case2d(64, 32, 7, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_4KB_S) },
            Case { dimension: Dimension::Tex3d, depth: 40, ..case2d(33, 20, 6, DXGI_FORMAT_R16_UNORM, Swizzle::SW_64KB_S) },
            Case { dimension: Dimension::Tex3d, depth: 5, ..case2d(20, 12, 3, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::LINEAR) },
            // Mip-clamped / held-level sub-ranges (`level0 > 0`, `first > level0`) + pooled `raw_base`
            Case { level0: 2, first: 2, raw_base: 512, ..case2d(256, 256, 9, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_S) },
            Case { level0: 2, first: 3, raw_base: 4096, ..case2d(256, 256, 9, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_S) },
            Case { level0: 1, first: 2, raw_base: 1024, ..case2d(300, 200, 9, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S) },
        ];
        let mut all = true;
        for case in &cases {
            all &= verify_case(rig, &d, case, u32::MAX);
        }
        // On-demand partial held masks: unheld large mips zeroed, held mips byte-exact.
        all &= verify_case(rig, &d, &cases[0], !0b11);
        all &= verify_case(rig, &d, &cases[9], !0b1);
        assert!(all, "{label}: GPU compute detiling differed from CPU reference");
    }

    #[test]
    fn gpu_detile_matches_cpu_on_warp() {
        let mut rig = Rig::warp().expect("WARP adapter must be available on Windows");
        run_matrix(&mut rig, "WARP");
    }

    #[test]
    fn gpu_detile_matches_cpu_on_hardware() {
        let Some(mut rig) = Rig::hardware() else {
            eprintln!("no hardware D3D12 adapter available");
            return;
        };
        run_matrix(&mut rig, "HW");
    }

    #[test]
    fn xor_swizzle_modes_fall_back_to_cpu() {
        let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Width: 64,
                Height: 64,
                Depth: 1,
                RowPitch: 256,
            },
        };
        for swizzle in [
            Swizzle::SW_64KB_S_T,
            Swizzle::SW_64KB_D_T,
            Swizzle::SW_4KB_S_X,
            Swizzle::SW_4KB_D_X,
            Swizzle::SW_4KB_R_X,
            Swizzle::SW_64KB_Z_X,
            Swizzle::SW_64KB_S_X,
            Swizzle::SW_64KB_D_X,
            Swizzle::SW_64KB_R_X,
        ] {
            let tex = Texture::new(&TextureDesc {
                dimension: Dimension::Tex2d,
                width: 64,
                height: 64,
                depth_or_array: 1,
                mips: 1,
                format: DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let layout = &tex.planes[0].layout;
            let raw = raw_layout(layout, layout.size);
            assert!(
                jobs(layout, &[footprint], &[64], 0, 1, &raw).is_none(),
                "{swizzle:?} must return None so Staging falls back to CPU"
            );
        }
    }

    /// End-to-end integration test through `HostList::prelude` +
    /// `texture::upload_from` + `Host::execute` + GPU texture readback,
    /// covering full mip chains (mip 0 + tail mip), NPOT RGBA8, BC3, BC7,
    /// mip-clamped (`level0 == first == 2` and `level0 == 0, first == 2`),
    /// multi-list submission deduplication, and dynamic memory updates.
    #[test]
    fn prelude_uploads_detiled_textures_byte_exact() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let Some(host) = crate::host::host() else { return };
        if detiler().is_none() {
            return;
        }
        // (width, height, mips, dxgi_format, swizzle_layout_word, want_mip, upload_first_mip)
        let specs: &[(u32, u32, u32, u32, u32, u32, u32)] = &[
            (256, 256, 9, 98, 0x109, 0, 0), // BC7_UNORM POT SW_64KB_S, all mips
            (129, 65, 8, 28, 0x105, 0, 0),  // R8G8B8A8_UNORM NPOT SW_4KB_S, all mips
            (256, 256, 9, 98, 0x109, 2, 2), // BC7_UNORM POT SW_64KB_S, level0=2, first=2
            (256, 256, 9, 98, 0x109, 0, 2), // BC7_UNORM POT SW_64KB_S, level0=0, first=2
            (132, 68, 6, 77, 0x109, 0, 0),  // BC3_UNORM NPOT SW_64KB_S, all mips
        ];
        let fill_pattern = |at: *mut u8, whole: usize, mut s: u64| {
            // SAFETY: caller ensures `[at..at + whole)` is writable.
            unsafe {
                for i in 0..whole {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    at.add(i).write((s >> 16) as u8);
                }
            }
        };
        let check_subresource = |resource: crate::object::This,
                                 copy: &crate::resource::HostTexture,
                                 at: *const u8,
                                 whole: usize,
                                 check_console_mip: u32,
                                 tag: &str| {
            let host_subresource = check_console_mip - copy.base;
            // SAFETY: live host texture.
            let d = unsafe { copy.texture.GetDesc() };
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut num_rows = 0u32;
            let mut bytes = 0u64;
            // SAFETY: single-subresource footprint query.
            unsafe {
                host.device.GetCopyableFootprints(
                    &d,
                    host_subresource,
                    1,
                    0,
                    Some(&mut footprint),
                    Some(&mut num_rows),
                    None,
                    Some(&mut bytes),
                );
            }
            let back = host.read_back_buffer(bytes).unwrap();
            let read_list = host.new_list().unwrap();
            crate::list::ensure_state(&read_list, resource, 0x800); // COPY_SOURCE
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed COM pointer valid for list execution.
                pResource: unsafe { std::mem::transmute_copy(&back) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: borrowed COM pointer valid for list execution.
                pResource: unsafe { std::mem::transmute_copy(&copy.texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: host_subresource },
            };
            // SAFETY: open list, valid resources in COPY_SOURCE / COPY_DEST.
            unsafe { read_list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
            read_list.close();
            host.execute(&[&read_list]);
            read_list.reset();

            let layout = crate::resource::texture_memory(resource)
                .and_then(|(_, rd)| crate::resource::texture_layout(&rd).map(|t| t.planes[0].layout.clone()))
                .unwrap();
            let pitch = footprint.Footprint.RowPitch as usize;
            let mut expected = vec![0u8; pitch * num_rows as usize];
            // SAFETY: committed slice of `whole` bytes.
            let tiled = unsafe { std::slice::from_raw_parts(at, whole) };
            layout
                .detile(tiled, check_console_mip, 0, &mut expected, pitch, pitch * num_rows as usize)
                .unwrap();

            let mut mapped = std::ptr::null_mut();
            // SAFETY: readback completed before mapping.
            unsafe {
                back.Map(0, None, Some(&mut mapped)).unwrap();
                let seen = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize);
                let m = &layout.mips[check_console_mip as usize];
                let row_bytes = m.width as usize * layout.desc.element_bytes as usize;
                for y in 0..m.height as usize {
                    let off = y * pitch;
                    assert_eq!(
                        &seen[off..off + row_bytes],
                        &expected[off..off + row_bytes],
                        "{tag} mip={check_console_mip} row={y}"
                    );
                }
                back.Unmap(0, None);
            }
        };

        for &(width, height, mips, format, layout_word, want_mip, first_mip) in specs {
            let desc = [
                3u32,
                0,
                0,
                0,
                width,
                0,
                height,
                (mips << 16) | 1,
                format,
                1,
                0,
                layout_word,
                0,
                0,
            ];
            let whole = crate::resource::placed_bytes(&desc).unwrap() as usize;
            // SAFETY: private committed allocation freed before leaving the loop.
            let at = unsafe { VirtualAlloc(std::ptr::null(), whole, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
            assert!(!at.is_null());
            let seed = 0x517C_C1B7_2722_0A95u64 ^ u64::from(width * 97 + height * 13 + format + first_mip * 7);
            fill_pattern(at, whole, seed);
            let resource = crate::resource::texture_at(at as u64, &desc).unwrap();
            crate::resource::want_level(resource, want_mip);
            let copy = crate::resource::host_copy(resource).unwrap();

            // Record upload on TWO lists in one submission to test once-per-submission
            // prelude deduplication and D3D12 resource state transitions under debug layer.
            let gpu_before = STATS[0].load(Relaxed);
            let list_a = host.new_list().unwrap();
            let list_b = host.new_list().unwrap();
            with_enabled(true, || {
                crate::texture::upload_from(&list_a, resource, first_mip);
                crate::texture::upload_from(&list_b, resource, first_mip);
            });
            list_a.close();
            list_b.close();
            host.execute(&[&list_a, &list_b]);
            list_a.reset();
            list_b.reset();
            assert_eq!(
                STATS[0].load(Relaxed) - gpu_before,
                1,
                "{width}x{height} f{format} want={want_mip} first={first_mip}: expected exactly 1 deduplicated GPU prelude detile"
            );

            let first_uploaded = first_mip.max(copy.base);
            let tag = format!("{width}x{height} f{format} want={want_mip} first={first_mip} v1");
            check_subresource(resource, &copy, at, whole, first_uploaded, &tag);
            if mips - 1 > first_uploaded {
                check_subresource(resource, &copy, at, whole, mips - 1, &tag);
            }

            // Mutate title memory in place and verify dynamic re-upload via prelude.
            if want_mip == 0 && first_mip == 0 && format == 28 {
                fill_pattern(at, whole, seed ^ 0xDEAD_BEEF_CAFE_BABEu64);
                let gpu_before_v2 = STATS[0].load(Relaxed);
                let refresh_list = host.new_list().unwrap();
                with_enabled(true, || {
                    crate::texture::BAKING.set(true);
                    crate::texture::upload_from(&refresh_list, resource, 0);
                    crate::texture::BAKING.set(false);
                });
                refresh_list.close();
                host.execute(&[&refresh_list]);
                refresh_list.reset();
                assert_eq!(
                    STATS[0].load(Relaxed) - gpu_before_v2,
                    1,
                    "dynamic update must re-run GPU prelude detile"
                );
                check_subresource(resource, &copy, at, whole, 0, "dynamic v2");
                check_subresource(resource, &copy, at, whole, mips - 1, "dynamic v2");
            }

            crate::texture::forget(resource as usize);
            // SAFETY: freeing test allocation after `forget`.
            unsafe {
                assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }

    /// Copy-queue uploads (`ahead::claim`) cannot run compute preludes, so they
    /// must always build CPU-detiled stagings even when `VICTORY_GPU_DETILE=1`.
    #[test]
    fn ahead_copy_queue_never_uses_gpu_staging() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let Some(_host) = crate::host::host() else { return };
        let desc = [3u32, 0, 0, 0, 128, 0, 128, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let whole = crate::resource::placed_bytes(&desc).unwrap() as usize;
        // SAFETY: private test allocation.
        let at = unsafe { VirtualAlloc(std::ptr::null(), whole, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!at.is_null());
        // SAFETY: initialized within `[0..whole)`.
        unsafe { std::ptr::write_bytes(at, 0x3c, whole) };
        let resource = crate::resource::texture_at(at as u64, &desc).unwrap();
        crate::resource::want_level(resource, 0);
        let copy = crate::resource::host_copy(resource).unwrap();
        let staging = with_enabled(true, || crate::texture::staging_for_test(resource, &copy, 0, false));
        assert!(staging.is_some_and(|s| !s.is_gpu()), "copy-queue staging must remain CPU-detiled");
        crate::texture::forget(resource as usize);
        // SAFETY: freeing test allocation.
        unsafe {
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Release benchmark comparing CPU `Layout::detile` vs raw staging copy +
    /// GPU compute detile dispatch on hardware (`cargo test --release -p victory-gpu
    /// --lib gpudetile::tests::bench_gpu_vs_cpu_detile -- --ignored --nocapture`).
    #[test]
    #[ignore = "a timing benchmark, not a functional check"]
    fn bench_gpu_vs_cpu_detile() {
        let Some(mut rig) = Rig::hardware() else {
            eprintln!("no hardware adapter");
            return;
        };
        let d = make(&rig.device).expect("detile shader compiles");
        for (width, mips, format, swizzle) in [
            (2048u32, 12u32, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_S),
            (2048, 12, DXGI_FORMAT_BC1_UNORM, Swizzle::SW_64KB_S),
            (1024, 11, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S),
            (512, 10, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_4KB_S),
        ] {
            let texture = Texture::new(&TextureDesc {
                dimension: Dimension::Tex2d,
                width,
                height: width,
                depth_or_array: 1,
                mips,
                format: format.0 as u32,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let layout = &texture.planes[0].layout;
            let host_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(width),
                Height: width,
                DepthOrArraySize: 1,
                MipLevels: mips as u16,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mips as usize];
            let mut rows = vec![0u32; mips as usize];
            let mut total = 0u64;
            // SAFETY: output buffers match `mips`.
            unsafe {
                rig.device.GetCopyableFootprints(
                    &host_desc,
                    0,
                    mips,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    None,
                    Some(&mut total),
                );
            }
            let tiled: Vec<u8> = (0..layout.size).map(|i| (i * 7 + (i >> 9)) as u8).collect();
            let raw = raw_layout(layout, layout.size);
            let plan = jobs(layout, &footprints, &rows, 0, mips, &raw).unwrap();
            let upload = rig.buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                raw.bytes,
            );
            let upload_cpu = rig.buffer(
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                total,
            );
            let data = map(&upload);
            let data_cpu = map(&upload_cpu);
            // SAFETY: valid mappings.
            unsafe { write_tables(layout, &raw, data) };
            let linear = rig.buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
                total.max(plan.linear_bytes).next_multiple_of(4),
            );
            // SAFETY: live upload buffer.
            let raw_gpu_va = unsafe { upload.GetGPUVirtualAddress() };
            // Warm up
            rig.run(|list| record(list, &d, raw_gpu_va, &linear, &plan.jobs));
            let iters = 32usize;
            let cpu_t0 = std::time::Instant::now();
            for _ in 0..iters {
                for (i, (fp, &r)) in footprints.iter().zip(&rows).enumerate() {
                    let pitch = fp.Footprint.RowPitch as usize;
                    let len = (pitch * r as usize).min((total - fp.Offset) as usize);
                    // SAFETY: within `upload_cpu`.
                    let out = unsafe { std::slice::from_raw_parts_mut(data_cpu.add(fp.Offset as usize), len) };
                    layout.detile(&tiled, i as u32, 0, out, pitch, pitch * r as usize).unwrap();
                }
            }
            let cpu_ms = cpu_t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let copy_t0 = std::time::Instant::now();
            for _ in 0..iters {
                // SAFETY: within `upload`.
                unsafe {
                    std::ptr::copy_nonoverlapping(tiled.as_ptr(), data, tiled.len());
                    write_mask(&raw, data, u32::MAX);
                }
            }
            let copy_ms = copy_t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let gpu_t0 = std::time::Instant::now();
            for _ in 0..iters {
                rig.run(|list| record(list, &d, raw_gpu_va, &linear, &plan.jobs));
            }
            let gpu_ms = gpu_t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let mb = layout.size as f64 / 1e6;
            eprintln!(
                "{} {width}x{width} {format:?} ({mb:.2} MB): CPU detile->WC {:.3} ms ({:.1} GB/s), CPU raw copy->WC {:.3} ms ({:.1} GB/s), GPU compute round-trip {:.3} ms",
                swizzle.name(),
                cpu_ms,
                mb / (cpu_ms / 1e3) / 1e3,
                copy_ms,
                mb / (copy_ms / 1e3) / 1e3,
                gpu_ms,
            );
        }
    }
}
