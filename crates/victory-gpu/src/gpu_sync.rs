//! GPU-direct zero-CPU-copy bidirectional 3-way sync compute engine over
//! `OpenExistingHeapFromAddress1` (`VICTORY_GPU_SYNC=1`, off by default).
//!
//! Title buffers in local VRAM (`D3D12_HEAP_TYPE_DEFAULT`, pool L1) run shaders
//! at full VRAM bandwidth, while title memory lives in system RAM (pool L0).
//! Instead of CPU-side staging copies and CPU readback merges, this engine
//! opens the title memory's 64 KiB pages directly as an L0 write-back D3D12
//! heap (`OpenExistingHeapFromAddress1`), keeps a device-local `before`
//! baseline twin alongside the `shadow` buffer in L1, and resolves byte-exact
//! 3-way sync in compute shaders:
//!
//! - For every byte inside a synced range `[range_start, range_end)`:
//!   - `T != B` (CPU/title wrote, including concurrent CPU+GPU conflict):
//!     **CPU/title wins** -> `S <- T`, `B <- T` (no write to `T`).
//!   - `T == B && S != B` (GPU-only write):
//!     **GPU write propagates** -> `T <- S`, `B <- S` (no write to `S`).
//!   - `T == B && S == B` (clean): zero writes to `T`, `S`, or `B`.
//!
//! Adaptive hardware capability modes (probed once per device):
//! - [`SyncMode::BidirectionalUav`]: L0 `OpenExistingHeapFromAddress1` supports
//!   both compute SRV reads and compute UAV writes — full zero-CPU-copy
//!   bidirectional 3-way sync in a single compute dispatch.
//! - [`SyncMode::SrvDirectUpload`]: L0 supports compute SRV reads while L0
//!   compute UAV writes are rejected — zero-CPU-staging `L0 -> L1` upload and
//!   L1 3-way diff/merge emitting compact GPU-only dword deltas.
//! - [`SyncMode::Unsupported`]: automatic fallback to the existing CPU path.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};
use windows_sys::Win32::System::Memory as mem;

use crate::host::{Host, transition};
use crate::inplace::{From as OpenFrom, open, private_whole, shown_runs};

/// Heap & placed-buffer alignment granularity (64 KiB).
const K64: u64 = 64 << 10;

/// Maximum logical buffer size supported by 32-bit byte-offset shader math (1 GiB).
const MAX_REGION_BYTES: u64 = 1 << 30;

/// Threads per compute group; each thread handles 4 coalesced 16-byte strides = 16 KiB/group.
const GROUP_THREADS: u32 = 256;
const DWORDS_PER_THREAD: u32 = 16;
const DWORDS_PER_GROUP: u32 = GROUP_THREADS * DWORDS_PER_THREAD;

/// Compact GPU-only delta ring capacity for [`SyncMode::SrvDirectUpload`] (262,144 dwords = 1 MiB
/// of pure GPU mutations per pass; larger mutations automatically loop until drained).
const SRV_MAX_DELTAS: u32 = 262_144;
const SRV_DELTA_HEADER_BYTES: u64 = 16;
const SRV_DELTA_STRIDE_BYTES: u64 = 16;
const SRV_DELTA_BUFFER_BYTES: u64 = SRV_DELTA_HEADER_BYTES + (SRV_MAX_DELTAS as u64) * SRV_DELTA_STRIDE_BYTES;

/// Whether `VICTORY_GPU_SYNC=1` is opted in (off by default).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_SYNC"))
}

/// Optional mode override via `VICTORY_GPU_SYNC_MODE=auto|bidir|srv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModeOverride {
    Auto,
    Bidir,
    Srv,
}

fn parse_mode_override(val: Option<&str>) -> ModeOverride {
    match val.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("bidir" | "uav" | "bidirectional") => ModeOverride::Bidir,
        Some("srv" | "upload") => ModeOverride::Srv,
        _ => ModeOverride::Auto,
    }
}

fn configured_mode() -> ModeOverride {
    static MODE: OnceLock<ModeOverride> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode_override(std::env::var("VICTORY_GPU_SYNC_MODE").ok().as_deref()))
}

/// Optional override forcing L0 `InterlockedCompareExchange` atomics even on discrete PCIe GPUs
/// (`VICTORY_GPU_SYNC_ATOMICS=1`, off by default; CacheCoherentUMA/WARP always use atomics).
fn force_atomics() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| crate::switch::on("VICTORY_GPU_SYNC_ATOMICS"))
}

/// Atomic mode bitflags passed to `SYNC_BIDIR_HLSL`:
/// - `ATOMIC_PARTIAL`: route partial-dword writes (`gpu_only != 0xFFFFFFFFu`) through
///   `InterlockedCompareExchange` CAS so concurrent CPU writes to sibling bytes in the same
///   32-bit dword are never clobbered by a non-atomic RMW.
/// - `ATOMIC_FULL`: also route full-dword writes (`gpu_only == 0xFFFFFFFFu`) through CAS
///   (enabled on cache-coherent UMA / WARP / `VICTORY_GPU_SYNC_ATOMICS=1`; disabled on discrete
///   PCIe GPUs so full-dword writes emit fast coalesced posted `Store4`/`Store`).
const ATOMIC_PARTIAL: u32 = 1;
const ATOMIC_FULL: u32 = 2;

fn select_mode(report: &ProbeReport, override_mode: ModeOverride) -> SyncMode {
    if !report.debug_errors.is_empty() {
        return SyncMode::Unsupported;
    }
    let bidir_safe = report.l0_uav_write_ok && report.l0_srv_read_ok && report.l0_atomics_ok;
    match override_mode {
        ModeOverride::Auto => {
            if bidir_safe {
                SyncMode::BidirectionalUav
            } else if report.l0_srv_read_ok {
                SyncMode::SrvDirectUpload
            } else {
                SyncMode::Unsupported
            }
        }
        ModeOverride::Bidir => {
            if bidir_safe {
                SyncMode::BidirectionalUav
            } else {
                SyncMode::Unsupported
            }
        }
        ModeOverride::Srv => {
            if report.l0_srv_read_ok {
                SyncMode::SrvDirectUpload
            } else {
                SyncMode::Unsupported
            }
        }
    }
}

fn compute_atomic_flags(report: &ProbeReport, force_full: bool) -> u32 {
    if !report.l0_atomics_ok {
        return 0;
    }
    let use_full = report.cache_coherent_uma || report.is_software || force_full;
    ATOMIC_PARTIAL | if use_full { ATOMIC_FULL } else { 0 }
}

/// Hardware sync capability tier determined by the live D3D12 probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SyncMode {
    /// Full zero-CPU-copy bidirectional 3-way sync via L0 compute UAV.
    BidirectionalUav,
    /// Zero-CPU-staging L0 compute SRV upload + L1 3-way merge with compact delta readback.
    SrvDirectUpload,
    /// Hardware/driver does not support L0 compute access; fall back to CPU path.
    Unsupported,
}

/// Reason a specific title memory range fell back to the standard CPU path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FallbackReason {
    DeviceUnsupported,
    EmptyOrTooLarge,
    UnalignedBase,
    ViewMemory,
    NotCommittedReadWrite,
    HeapOpenFailed,
    BufferCreateFailed,
    BeforeAllocFailed,
}

/// Live hardware/WARP D3D12 capability probe result.
#[derive(Clone, Debug)]
pub(crate) struct ProbeReport {
    pub(crate) adapter_name: String,
    pub(crate) is_software: bool,
    pub(crate) cache_coherent_uma: bool,
    pub(crate) has_device13: bool,
    pub(crate) l0_open_ok: bool,
    pub(crate) l0_srv_read_ok: bool,
    pub(crate) l0_uav_write_ok: bool,
    pub(crate) l0_copy_to_l1_ok: bool,
    pub(crate) l0_copy_from_l1_ok: bool,
    pub(crate) l0_atomics_ok: bool,
    pub(crate) l0_heap_flags: u32,
    pub(crate) mode: SyncMode,
    pub(crate) debug_errors: Vec<String>,
    pub(crate) notes: Vec<String>,
}

/// Operation mode passed to the 3-way sync compute shaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum SyncOp {
    /// Initial bulk copy: `S <- T`, `B <- T` for every byte inside `[range_start, range_end)`.
    Init = 0,
    /// CPU-to-GPU upload only: where `T != B`, `S <- T` and `B <- T`. Never writes `T`.
    Upload3Way = 1,
    /// Full byte-exact 3-way sync: CPU wins on conflict (`T != B -> S, B`; `T == B && S != B -> T, B`).
    Bidirectional3Way = 2,
    /// L1-first GPU write-back (`post_list`): reads `S` and `B` in local L1 VRAM first and skips
    /// clean dwords (`S == B`) with ZERO PCIe L0 reads or writes. Only where `S != B` does it load
    /// `T` over PCIe and resolve 3-way (`T != B` -> CPU wins `S, B`; `T == B` -> GPU wins `T, B`).
    GpuWriteBackOnly = 3,
}

/// Number of `SrvDirectUpload` submissions currently in flight requiring `srv_rig` drain.
static SRV_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

/// Returns true if an `SrvDirectUpload` sync flight is still in flight and must settle before
/// submitting a new flight (`false` always in `BidirectionalUav` mode).
pub(crate) fn needs_settle() -> bool {
    SRV_IN_FLIGHT.load(Ordering::Acquire) > 0
}

/// Counters:
/// `[0]` total attached regions, `[1]` total fallback count, `[2]` interval upload dispatches,
/// `[3]` interval upload bytes, `[4]` interval sync-back dispatches, `[5]` interval sync-back bytes,
/// `[6]` interval SRV delta passes, `[7]` total detached regions.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Live byte count of title buffers currently attached to GPU-direct sync (`eligible_bytes`).
static ELIGIBLE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Interval CPU microseconds spent waiting on `gpu_sync` fences (`fence_wait_us`).
static FENCE_WAIT_US: AtomicU64 = AtomicU64::new(0);

/// Interval on-card GPU microseconds spent in `gpu_sync` dispatches/copies (`gpu_dispatch_us`).
static GPU_DISPATCH_US: AtomicU64 = AtomicU64::new(0);

/// Tracks live eligible bytes when a GPU-synced shadow attaches.
pub(crate) fn note_attached(bytes: u64) {
    ELIGIBLE_BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// Tracks live eligible bytes when a GPU-synced shadow drops.
pub(crate) fn note_dropped(bytes: u64) {
    ELIGIBLE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

/// Records CPU microseconds spent waiting on a `gpu_sync` submission or `SrvDirectUpload` fence.
pub(crate) fn note_fence_wait_us(us: u64) {
    FENCE_WAIT_US.fetch_add(us, Ordering::Relaxed);
}

/// Returns true if any GPU-synced shadow is currently live (zero-cost fast check for `watch` / `inplace`).
pub(crate) fn has_eligible() -> bool {
    ELIGIBLE_BYTES.load(Ordering::Relaxed) != 0
}

/// Detaches and settles any `VICTORY_GPU_SYNC` L0 heaps overlapping `[at, at + len)` (or the whole
/// allocation at `at` when `len == 0`).
///
/// Called by the runtime (`xmem.rs`) BEFORE acquiring `watch::title_memory_remapping()`
/// (`MAPPING.write()`) so settling GPU work under `host.submitted` never deadlocks against
/// `Host::submit` acquiring `MAPPING.read()`. Single relaxed atomic no-op when off.
pub fn memory_freeing(at: usize, len: usize) {
    if !on() && !has_eligible() {
        return;
    }
    let detached = crate::shadow::gpu_sync_let_go(at as u64, len as u64);
    if detached > 0 {
        note_detached(detached);
    }
}

#[cfg(test)]
pub(crate) fn eligible_bytes() -> u64 {
    ELIGIBLE_BYTES.load(Ordering::Relaxed)
}

/// Full bidirectional 3-way sync compute shader (`BidirectionalUav` mode).
///
/// Uses branchless SWAR per-byte inequality masks (`byte_ne_mask` / `byte_ne_mask4`) and a
/// 128-bit (`uint4`) fast path for interior blocks while guarding per-dword stores to `title_buf`
/// and `shadow_buf` so unmodified dwords are never touched (preserving concurrent CPU writes on
/// adjacent dwords). Boundary and tail dwords use exact per-byte range masking.
const SYNC_BIDIR_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=10, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Params : register(b0) {
    uint title_bias;
    uint shadow_bias;
    uint before_bias;
    uint range_start;
    uint range_end;
    uint first_word;
    uint word_count;
    uint mode;
    uint atomic_flags;
    uint dirty_words;
};
RWByteAddressBuffer title_buf : register(u0);
RWByteAddressBuffer shadow_buf : register(u1);
RWByteAddressBuffer before_buf : register(u2);
RWByteAddressBuffer dirty_pages_buf : register(u3);
groupshared uint grp_dirty_mask;

void note_dirty_offset(uint off, uint grp_p0) {
    if (dirty_words == 0u) return;
    uint p = off >> 16u;
    uint bit = (p == grp_p0) ? 1u : 2u;
    uint prev = 0u;
    InterlockedOr(grp_dirty_mask, bit, prev);
}

uint byte_ne_mask(uint x, uint y) {
    uint d = x ^ y;
    d = (d | (d >> 4u)) & 0x0F0F0F0Fu;
    d = (d | (d >> 2u)) & 0x03030303u;
    d = (d | (d >> 1u)) & 0x01010101u;
    return d * 0xFFu;
}

uint4 byte_ne_mask4(uint4 x, uint4 y) {
    uint4 d = x ^ y;
    d = (d | (d >> 4u)) & 0x0F0F0F0Fu;
    d = (d | (d >> 2u)) & 0x03030303u;
    d = (d | (d >> 1u)) & 0x01010101u;
    return d * 0xFFu;
}

// Writes GPU-exclusive bytes (`gpu_only`) into `title_buf` at 4-byte-aligned `addr`.
// Returns the byte mask of GPU-exclusive bytes that were committed.
// `atomic_flags` bit 0 (`1u`): CAS for partial-dword writes (`gpu_only != 0xFFFFFFFFu`) so
// concurrent CPU writes to sibling bytes are never clobbered by a non-atomic RMW.
// `atomic_flags` bit 1 (`2u`): CAS for full-dword writes (`gpu_only == 0xFFFFFFFFu`) on
// cache-coherent UMA / WARP / explicit `VICTORY_GPU_SYNC_ATOMICS=1`; discrete PCIe leaves bit 1
// clear so full-dword writes emit fast non-atomic posted `Store4`/`Store`.
uint write_title_masked(uint addr, uint t_seen, uint s_val, uint gpu_only) {
    bool need_cas = (gpu_only == 0xFFFFFFFFu)
        ? ((atomic_flags & 2u) != 0u)
        : ((atomic_flags & 1u) != 0u);
    if (need_cas) {
        uint expected = t_seen;
        uint mask = gpu_only;
        [loop]
        for (uint attempt = 0u; attempt < 64u; ++attempt) {
            uint desired = (expected & ~mask) | (s_val & mask);
            uint actual = 0u;
            title_buf.InterlockedCompareExchange(addr, expected, desired, actual);
            if (actual == expected) {
                return mask;
            }
            uint cpu_raced = byte_ne_mask(actual, expected);
            mask &= ~cpu_raced;
            if (mask == 0u) {
                return 0u;
            }
            expected = actual;
        }
        return 0u;
    }
    if (gpu_only == 0xFFFFFFFFu) {
        title_buf.Store(addr, s_val);
        return 0xFFFFFFFFu;
    }
    title_buf.Store(addr, (t_seen & ~gpu_only) | (s_val & gpu_only));
    return gpu_only;
}

uint dword_range_mask(uint off) {
    if (off >= range_start && off + 4u <= range_end) {
        return 0xFFFFFFFFu;
    }
    uint m = 0u;
    if (off + 0u >= range_start && off + 0u < range_end) m |= 0x000000FFu;
    if (off + 1u >= range_start && off + 1u < range_end) m |= 0x0000FF00u;
    if (off + 2u >= range_start && off + 2u < range_end) m |= 0x00FF0000u;
    if (off + 3u >= range_start && off + 3u < range_end) m |= 0xFF000000u;
    return m;
}

void sync_vec4(uint off, uint4 t, uint4 b, uint4 s, uint grp_p0) {
    uint t_addr = title_bias + off;
    uint s_addr = shadow_bias + off;
    uint b_addr = before_bias + off;
    uint4 cpu_diff = byte_ne_mask4(t, b);
    uint4 gpu_diff = byte_ne_mask4(s, b);
    uint4 gpu_only = gpu_diff & ~cpu_diff;
    if (any(cpu_diff != 0u)) {
        uint4 new_s = (s & ~cpu_diff) | (t & cpu_diff);
        if (all(cpu_diff != 0u)) {
            shadow_buf.Store4(s_addr, new_s);
        } else {
            if (cpu_diff.x != 0u) shadow_buf.Store(s_addr + 0u, new_s.x);
            if (cpu_diff.y != 0u) shadow_buf.Store(s_addr + 4u, new_s.y);
            if (cpu_diff.z != 0u) shadow_buf.Store(s_addr + 8u, new_s.z);
            if (cpu_diff.w != 0u) shadow_buf.Store(s_addr + 12u, new_s.w);
        }
    }
    uint4 committed_gpu = uint4(0u, 0u, 0u, 0u);
    if (any(gpu_only != 0u)) {
        if ((atomic_flags & 2u) == 0u && all(gpu_only == 0xFFFFFFFFu)) {
            title_buf.Store4(t_addr, s);
            committed_gpu = uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
        } else {
            if (gpu_only.x != 0u) committed_gpu.x = write_title_masked(t_addr + 0u, t.x, s.x, gpu_only.x);
            if (gpu_only.y != 0u) committed_gpu.y = write_title_masked(t_addr + 4u, t.y, s.y, gpu_only.y);
            if (gpu_only.z != 0u) committed_gpu.z = write_title_masked(t_addr + 8u, t.z, s.z, gpu_only.z);
            if (gpu_only.w != 0u) committed_gpu.w = write_title_masked(t_addr + 12u, t.w, s.w, gpu_only.w);
        }
        if (committed_gpu.x != 0u) note_dirty_offset(off + 0u, grp_p0);
        if (committed_gpu.y != 0u) note_dirty_offset(off + 4u, grp_p0);
        if (committed_gpu.z != 0u) note_dirty_offset(off + 8u, grp_p0);
        if (committed_gpu.w != 0u) note_dirty_offset(off + 12u, grp_p0);
    }
    uint4 any_diff = cpu_diff | committed_gpu;
    if (any(any_diff != 0u)) {
        uint4 new_b = (b & ~any_diff) | (t & cpu_diff) | (s & committed_gpu);
        before_buf.Store4(b_addr, new_b);
    }
}

void upload_vec4(uint off, uint4 t, uint4 b) {
    uint4 cpu_diff = byte_ne_mask4(t, b);
    if (any(cpu_diff != 0u)) {
        uint s_addr = shadow_bias + off;
        uint b_addr = before_bias + off;
        uint4 s = shadow_buf.Load4(s_addr);
        uint4 new_s = (s & ~cpu_diff) | (t & cpu_diff);
        uint4 new_b = (b & ~cpu_diff) | (t & cpu_diff);
        if (all(cpu_diff != 0u)) {
            shadow_buf.Store4(s_addr, new_s);
        } else {
            if (cpu_diff.x != 0u) shadow_buf.Store(s_addr + 0u, new_s.x);
            if (cpu_diff.y != 0u) shadow_buf.Store(s_addr + 4u, new_s.y);
            if (cpu_diff.z != 0u) shadow_buf.Store(s_addr + 8u, new_s.z);
            if (cpu_diff.w != 0u) shadow_buf.Store(s_addr + 12u, new_s.w);
        }
        before_buf.Store4(b_addr, new_b);
    }
}

void writeback_dword(uint off, uint b, uint s, uint gpu_diff, uint grp_p0) {
    uint t_addr = title_bias + off;
    uint s_addr = shadow_bias + off;
    uint b_addr = before_bias + off;
    uint t = title_buf.Load(t_addr);
    uint cpu_diff = byte_ne_mask(t, b) & gpu_diff;
    uint gpu_only = gpu_diff & ~cpu_diff;
    if (cpu_diff != 0u) {
        shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
    }
    uint committed_gpu = 0u;
    if (gpu_only != 0u) {
        committed_gpu = write_title_masked(t_addr, t, s, gpu_only);
        if (committed_gpu != 0u) {
            note_dirty_offset(off, grp_p0);
        }
    }
    uint any_diff = cpu_diff | committed_gpu;
    if (any_diff != 0u) {
        before_buf.Store(b_addr, (b & ~any_diff) | (t & cpu_diff) | (s & committed_gpu));
    }
}

void writeback_vec4(uint off, uint4 b, uint4 s, uint grp_p0) {
    uint4 gpu_diff = byte_ne_mask4(s, b);
    if (!any(gpu_diff != 0u)) {
        return;
    }
    if (all(gpu_diff != 0u)) {
        uint t_addr = title_bias + off;
        uint s_addr = shadow_bias + off;
        uint b_addr = before_bias + off;
        uint4 t = title_buf.Load4(t_addr);
        uint4 cpu_diff = byte_ne_mask4(t, b) & gpu_diff;
        uint4 gpu_only = gpu_diff & ~cpu_diff;
        if (any(cpu_diff != 0u)) {
            uint4 new_s = (s & ~cpu_diff) | (t & cpu_diff);
            if (all(cpu_diff != 0u)) {
                shadow_buf.Store4(s_addr, new_s);
            } else {
                if (cpu_diff.x != 0u) shadow_buf.Store(s_addr + 0u, new_s.x);
                if (cpu_diff.y != 0u) shadow_buf.Store(s_addr + 4u, new_s.y);
                if (cpu_diff.z != 0u) shadow_buf.Store(s_addr + 8u, new_s.z);
                if (cpu_diff.w != 0u) shadow_buf.Store(s_addr + 12u, new_s.w);
            }
        }
        uint4 committed_gpu = uint4(0u, 0u, 0u, 0u);
        if (any(gpu_only != 0u)) {
            if ((atomic_flags & 2u) == 0u && all(gpu_only == 0xFFFFFFFFu)) {
                title_buf.Store4(t_addr, s);
                committed_gpu = uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
            } else {
                if (gpu_only.x != 0u) committed_gpu.x = write_title_masked(t_addr + 0u, t.x, s.x, gpu_only.x);
                if (gpu_only.y != 0u) committed_gpu.y = write_title_masked(t_addr + 4u, t.y, s.y, gpu_only.y);
                if (gpu_only.z != 0u) committed_gpu.z = write_title_masked(t_addr + 8u, t.z, s.z, gpu_only.z);
                if (gpu_only.w != 0u) committed_gpu.w = write_title_masked(t_addr + 12u, t.w, s.w, gpu_only.w);
            }
            if (committed_gpu.x != 0u) note_dirty_offset(off + 0u, grp_p0);
            if (committed_gpu.y != 0u) note_dirty_offset(off + 4u, grp_p0);
            if (committed_gpu.z != 0u) note_dirty_offset(off + 8u, grp_p0);
            if (committed_gpu.w != 0u) note_dirty_offset(off + 12u, grp_p0);
        }
        uint4 any_diff = cpu_diff | committed_gpu;
        if (any(any_diff != 0u)) {
            uint4 new_b = (b & ~any_diff) | (t & cpu_diff) | (s & committed_gpu);
            before_buf.Store4(b_addr, new_b);
        }
    } else {
        if (gpu_diff.x != 0u) writeback_dword(off + 0u, b.x, s.x, gpu_diff.x, grp_p0);
        if (gpu_diff.y != 0u) writeback_dword(off + 4u, b.y, s.y, gpu_diff.y, grp_p0);
        if (gpu_diff.z != 0u) writeback_dword(off + 8u, b.z, s.z, gpu_diff.z, grp_p0);
        if (gpu_diff.w != 0u) writeback_dword(off + 12u, b.w, s.w, gpu_diff.w, grp_p0);
    }
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (tid == 0u) grp_dirty_mask = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint grp_base = gid.x * 4096u;
    uint grp_off0 = (first_word + grp_base) * 4u;
    uint grp_p0 = grp_off0 >> 16u;

    if (grp_base < word_count) {
        if (grp_base + 4096u <= word_count && grp_off0 >= range_start && grp_off0 + 16384u <= range_end) {
            uint o0 = grp_off0 + tid * 16u;
            uint o1 = o0 + 4096u;
            uint o2 = o0 + 8192u;
            uint o3 = o0 + 12288u;
            if (mode == 3u) {
                // L1-first fast path: read ONLY local L1 VRAM (`shadow` + `before`).
                // Clean blocks exit immediately with ZERO PCIe L0 reads or writes.
                uint4 s0 = shadow_buf.Load4(shadow_bias + o0);
                uint4 s1 = shadow_buf.Load4(shadow_bias + o1);
                uint4 s2 = shadow_buf.Load4(shadow_bias + o2);
                uint4 s3 = shadow_buf.Load4(shadow_bias + o3);
                uint4 b0 = before_buf.Load4(before_bias + o0);
                uint4 b1 = before_buf.Load4(before_bias + o1);
                uint4 b2 = before_buf.Load4(before_bias + o2);
                uint4 b3 = before_buf.Load4(before_bias + o3);
                if (any(s0 != b0)) writeback_vec4(o0, b0, s0, grp_p0);
                if (any(s1 != b1)) writeback_vec4(o1, b1, s1, grp_p0);
                if (any(s2 != b2)) writeback_vec4(o2, b2, s2, grp_p0);
                if (any(s3 != b3)) writeback_vec4(o3, b3, s3, grp_p0);
            } else {
                // Issue 4 coalesced 128-byte PCIe bursts simultaneously to saturate bus MLP.
                uint4 t0 = title_buf.Load4(title_bias + o0);
                uint4 t1 = title_buf.Load4(title_bias + o1);
                uint4 t2 = title_buf.Load4(title_bias + o2);
                uint4 t3 = title_buf.Load4(title_bias + o3);
                if (mode == 0u) {
                    shadow_buf.Store4(shadow_bias + o0, t0);
                    shadow_buf.Store4(shadow_bias + o1, t1);
                    shadow_buf.Store4(shadow_bias + o2, t2);
                    shadow_buf.Store4(shadow_bias + o3, t3);
                    before_buf.Store4(before_bias + o0, t0);
                    before_buf.Store4(before_bias + o1, t1);
                    before_buf.Store4(before_bias + o2, t2);
                    before_buf.Store4(before_bias + o3, t3);
                } else {
                    uint4 b0 = before_buf.Load4(before_bias + o0);
                    uint4 b1 = before_buf.Load4(before_bias + o1);
                    uint4 b2 = before_buf.Load4(before_bias + o2);
                    uint4 b3 = before_buf.Load4(before_bias + o3);
                    if (mode == 1u) {
                        upload_vec4(o0, t0, b0);
                        upload_vec4(o1, t1, b1);
                        upload_vec4(o2, t2, b2);
                        upload_vec4(o3, t3, b3);
                    } else {
                        uint4 s0 = shadow_buf.Load4(shadow_bias + o0);
                        uint4 s1 = shadow_buf.Load4(shadow_bias + o1);
                        uint4 s2 = shadow_buf.Load4(shadow_bias + o2);
                        uint4 s3 = shadow_buf.Load4(shadow_bias + o3);
                        sync_vec4(o0, t0, b0, s0, grp_p0);
                        sync_vec4(o1, t1, b1, s1, grp_p0);
                        sync_vec4(o2, t2, b2, s2, grp_p0);
                        sync_vec4(o3, t3, b3, s3, grp_p0);
                    }
                }
            }
        } else {
            [loop]
            for (uint stride = 0u; stride < 4u; ++stride) {
                uint base = grp_base + stride * 1024u + tid * 4u;
                if (base >= word_count) break;
                uint limit = min(base + 4u, word_count);
                [loop]
                for (uint i = base; i < limit; ++i) {
                    uint w_off = (first_word + i) * 4u;
                    uint rmask = dword_range_mask(w_off);
                    if (rmask == 0u) continue;
                    uint t_addr = title_bias + w_off;
                    uint s_addr = shadow_bias + w_off;
                    uint b_addr = before_bias + w_off;
                    if (mode == 3u) {
                        uint s = shadow_buf.Load(s_addr);
                        uint b = before_buf.Load(b_addr);
                        uint gpu_diff = byte_ne_mask(s, b) & rmask;
                        if (gpu_diff != 0u) {
                            writeback_dword(w_off, b, s, gpu_diff, grp_p0);
                        }
                        continue;
                    }
                    uint t = title_buf.Load(t_addr);
                    if (mode == 0u) {
                        if (rmask == 0xFFFFFFFFu) {
                            shadow_buf.Store(s_addr, t);
                            before_buf.Store(b_addr, t);
                        } else {
                            uint s = shadow_buf.Load(s_addr);
                            uint b = before_buf.Load(b_addr);
                            shadow_buf.Store(s_addr, (s & ~rmask) | (t & rmask));
                            before_buf.Store(b_addr, (b & ~rmask) | (t & rmask));
                        }
                        continue;
                    }
                    uint b = before_buf.Load(b_addr);
                    uint cpu_diff = byte_ne_mask(t, b) & rmask;
                    if (mode == 1u) {
                        if (cpu_diff != 0u) {
                            uint s = shadow_buf.Load(s_addr);
                            shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
                            before_buf.Store(b_addr, (b & ~cpu_diff) | (t & cpu_diff));
                        }
                        continue;
                    }
                    uint s = shadow_buf.Load(s_addr);
                    uint gpu_diff = byte_ne_mask(s, b) & rmask;
                    uint gpu_only = gpu_diff & ~cpu_diff;
                    if (cpu_diff != 0u) {
                        shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
                    }
                    uint committed_gpu = 0u;
                    if (gpu_only != 0u) {
                        committed_gpu = write_title_masked(t_addr, t, s, gpu_only);
                        if (committed_gpu != 0u) {
                            note_dirty_offset(w_off, grp_p0);
                        }
                    }
                    uint any_diff = cpu_diff | committed_gpu;
                    if (any_diff != 0u) {
                        before_buf.Store(b_addr, (b & ~any_diff) | (t & cpu_diff) | (s & committed_gpu));
                    }
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();
    if (dirty_words != 0u && tid == 0u && grp_dirty_mask != 0u) {
        uint prev = 0u;
        if ((grp_dirty_mask & 1u) != 0u) {
            uint w = min(grp_p0 >> 5u, dirty_words - 1u);
            uint b = (grp_p0 >> 5u) < dirty_words ? (1u << (grp_p0 & 31u)) : 0x80000000u;
            dirty_pages_buf.InterlockedOr(w * 4u, b, prev);
        }
        if ((grp_dirty_mask & 2u) != 0u) {
            uint p1 = grp_p0 + 1u;
            uint w = min(p1 >> 5u, dirty_words - 1u);
            uint b = (p1 >> 5u) < dirty_words ? (1u << (p1 & 31u)) : 0x80000000u;
            dirty_pages_buf.InterlockedOr(w * 4u, b, prev);
        }
    }
}
"#;

/// SRV-only adaptive compute shader (`SrvDirectUpload` mode).
///
/// Reads `title_buf` strictly through an SRV (`ByteAddressBuffer`, never writing to L0 from
/// compute). Handles `Init`, `Upload3Way`, and `Bidirectional3Way` (updating `shadow_buf` and
/// `before_buf` in L1 and emitting compact `(byte_off, gpu_word, gpu_only_mask)` records into an
/// L1 delta buffer; overflowing dwords keep their old `before_buf` bytes so subsequent passes drain
/// them cleanly without data loss or extra allocation).
const SYNC_SRV_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=9, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Params : register(b0) {
    uint title_bias;
    uint shadow_bias;
    uint before_bias;
    uint range_start;
    uint range_end;
    uint first_word;
    uint word_count;
    uint mode;
    uint max_deltas;
};
ByteAddressBuffer title_buf : register(t0);
RWByteAddressBuffer shadow_buf : register(u0);
RWByteAddressBuffer before_buf : register(u1);
RWByteAddressBuffer deltas_buf : register(u2);

uint byte_ne_mask(uint x, uint y) {
    uint d = x ^ y;
    d = (d | (d >> 4u)) & 0x0F0F0F0Fu;
    d = (d | (d >> 2u)) & 0x03030303u;
    d = (d | (d >> 1u)) & 0x01010101u;
    return d * 0xFFu;
}

uint4 byte_ne_mask4(uint4 x, uint4 y) {
    uint4 d = x ^ y;
    d = (d | (d >> 4u)) & 0x0F0F0F0Fu;
    d = (d | (d >> 2u)) & 0x03030303u;
    d = (d | (d >> 1u)) & 0x01010101u;
    return d * 0xFFu;
}

uint dword_range_mask(uint off) {
    if (off >= range_start && off + 4u <= range_end) {
        return 0xFFFFFFFFu;
    }
    uint m = 0u;
    if (off + 0u >= range_start && off + 0u < range_end) m |= 0x000000FFu;
    if (off + 1u >= range_start && off + 1u < range_end) m |= 0x0000FF00u;
    if (off + 2u >= range_start && off + 2u < range_end) m |= 0x00FF0000u;
    if (off + 3u >= range_start && off + 3u < range_end) m |= 0xFF000000u;
    return m;
}

bool emit_delta(uint w_off, uint gpu_word, uint gpu_mask, uint base_t) {
    uint slot = 0u;
    deltas_buf.InterlockedAdd(0u, 1u, slot);
    if (slot < max_deltas) {
        uint dst = 16u + slot * 16u;
        deltas_buf.Store4(dst, uint4(w_off, gpu_word, gpu_mask, base_t));
        return true;
    }
    deltas_buf.Store(4u, 1u);
    return false;
}

void upload_vec4_srv(uint off, uint4 t, uint4 b) {
    uint4 cpu_diff = byte_ne_mask4(t, b);
    if (any(cpu_diff != 0u)) {
        uint s_addr = shadow_bias + off;
        uint b_addr = before_bias + off;
        uint4 s = shadow_buf.Load4(s_addr);
        uint4 new_s = (s & ~cpu_diff) | (t & cpu_diff);
        uint4 new_b = (b & ~cpu_diff) | (t & cpu_diff);
        if (all(cpu_diff != 0u)) {
            shadow_buf.Store4(s_addr, new_s);
        } else {
            if (cpu_diff.x != 0u) shadow_buf.Store(s_addr + 0u, new_s.x);
            if (cpu_diff.y != 0u) shadow_buf.Store(s_addr + 4u, new_s.y);
            if (cpu_diff.z != 0u) shadow_buf.Store(s_addr + 8u, new_s.z);
            if (cpu_diff.w != 0u) shadow_buf.Store(s_addr + 12u, new_s.w);
        }
        before_buf.Store4(b_addr, new_b);
    }
}

void writeback_srv_dword(uint w_off, uint b, uint s, uint gpu_diff) {
    uint t_addr = title_bias + w_off;
    uint s_addr = shadow_bias + w_off;
    uint b_addr = before_bias + w_off;
    uint t = title_buf.Load(t_addr);
    uint cpu_diff = byte_ne_mask(t, b) & gpu_diff;
    uint gpu_only = gpu_diff & ~cpu_diff;
    if (cpu_diff != 0u) {
        shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
    }
    uint commit_gpu = 0u;
    if (gpu_only != 0u) {
        if (emit_delta(w_off, s, gpu_only, t)) {
            commit_gpu = gpu_only;
        }
    }
    uint commit_b = cpu_diff | commit_gpu;
    if (commit_b != 0u) {
        before_buf.Store(b_addr, (b & ~commit_b) | (t & cpu_diff) | (s & commit_gpu));
    }
}

void writeback_srv_vec4(uint off, uint4 b, uint4 s) {
    uint4 gpu_diff = byte_ne_mask4(s, b);
    if (gpu_diff.x != 0u) writeback_srv_dword(off + 0u, b.x, s.x, gpu_diff.x);
    if (gpu_diff.y != 0u) writeback_srv_dword(off + 4u, b.y, s.y, gpu_diff.y);
    if (gpu_diff.z != 0u) writeback_srv_dword(off + 8u, b.z, s.z, gpu_diff.z);
    if (gpu_diff.w != 0u) writeback_srv_dword(off + 12u, b.w, s.w, gpu_diff.w);
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint grp_base = gid.x * 4096u;
    if (grp_base >= word_count) return;
    uint grp_off0 = (first_word + grp_base) * 4u;

    if (grp_base + 4096u <= word_count && grp_off0 >= range_start && grp_off0 + 16384u <= range_end && mode != 2u) {
        uint o0 = grp_off0 + tid * 16u;
        uint o1 = o0 + 4096u;
        uint o2 = o0 + 8192u;
        uint o3 = o0 + 12288u;
        if (mode == 3u) {
            uint4 s0 = shadow_buf.Load4(shadow_bias + o0);
            uint4 s1 = shadow_buf.Load4(shadow_bias + o1);
            uint4 s2 = shadow_buf.Load4(shadow_bias + o2);
            uint4 s3 = shadow_buf.Load4(shadow_bias + o3);
            uint4 b0 = before_buf.Load4(before_bias + o0);
            uint4 b1 = before_buf.Load4(before_bias + o1);
            uint4 b2 = before_buf.Load4(before_bias + o2);
            uint4 b3 = before_buf.Load4(before_bias + o3);
            if (any(s0 != b0)) writeback_srv_vec4(o0, b0, s0);
            if (any(s1 != b1)) writeback_srv_vec4(o1, b1, s1);
            if (any(s2 != b2)) writeback_srv_vec4(o2, b2, s2);
            if (any(s3 != b3)) writeback_srv_vec4(o3, b3, s3);
            return;
        }
        uint4 t0 = title_buf.Load4(title_bias + o0);
        uint4 t1 = title_buf.Load4(title_bias + o1);
        uint4 t2 = title_buf.Load4(title_bias + o2);
        uint4 t3 = title_buf.Load4(title_bias + o3);
        if (mode == 0u) {
            shadow_buf.Store4(shadow_bias + o0, t0);
            shadow_buf.Store4(shadow_bias + o1, t1);
            shadow_buf.Store4(shadow_bias + o2, t2);
            shadow_buf.Store4(shadow_bias + o3, t3);
            before_buf.Store4(before_bias + o0, t0);
            before_buf.Store4(before_bias + o1, t1);
            before_buf.Store4(before_bias + o2, t2);
            before_buf.Store4(before_bias + o3, t3);
            return;
        }
        uint4 b0 = before_buf.Load4(before_bias + o0);
        uint4 b1 = before_buf.Load4(before_bias + o1);
        uint4 b2 = before_buf.Load4(before_bias + o2);
        uint4 b3 = before_buf.Load4(before_bias + o3);
        upload_vec4_srv(o0, t0, b0);
        upload_vec4_srv(o1, t1, b1);
        upload_vec4_srv(o2, t2, b2);
        upload_vec4_srv(o3, t3, b3);
        return;
    }

    [loop]
    for (uint stride = 0u; stride < 4u; ++stride) {
        uint base = grp_base + stride * 1024u + tid * 4u;
        if (base >= word_count) break;
        uint limit = min(base + 4u, word_count);
        [loop]
        for (uint i = base; i < limit; ++i) {
            uint w_off = (first_word + i) * 4u;
            uint rmask = dword_range_mask(w_off);
            if (rmask == 0u) continue;
            uint t_addr = title_bias + w_off;
            uint s_addr = shadow_bias + w_off;
            uint b_addr = before_bias + w_off;
            if (mode == 3u) {
                uint s = shadow_buf.Load(s_addr);
                uint b = before_buf.Load(b_addr);
                uint gpu_diff = byte_ne_mask(s, b) & rmask;
                if (gpu_diff != 0u) {
                    writeback_srv_dword(w_off, b, s, gpu_diff);
                }
                continue;
            }
            uint t = title_buf.Load(t_addr);
            if (mode == 0u) {
                if (rmask == 0xFFFFFFFFu) {
                    shadow_buf.Store(s_addr, t);
                    before_buf.Store(b_addr, t);
                } else {
                    uint s = shadow_buf.Load(s_addr);
                    uint b = before_buf.Load(b_addr);
                    shadow_buf.Store(s_addr, (s & ~rmask) | (t & rmask));
                    before_buf.Store(b_addr, (b & ~rmask) | (t & rmask));
                }
                continue;
            }
            uint b = before_buf.Load(b_addr);
            uint cpu_diff = byte_ne_mask(t, b) & rmask;
            if (mode == 1u) {
                if (cpu_diff != 0u) {
                    uint s = shadow_buf.Load(s_addr);
                    shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
                    before_buf.Store(b_addr, (b & ~cpu_diff) | (t & cpu_diff));
                }
                continue;
            }
            uint s = shadow_buf.Load(s_addr);
            uint gpu_diff = byte_ne_mask(s, b) & rmask;
            uint gpu_only = gpu_diff & ~cpu_diff;
            if (cpu_diff != 0u) {
                shadow_buf.Store(s_addr, (s & ~cpu_diff) | (t & cpu_diff));
            }
            uint commit_gpu = 0u;
            if (gpu_only != 0u) {
                if (emit_delta(w_off, s, gpu_only, t)) {
                    commit_gpu = gpu_only;
                }
            }
            uint commit_b = cpu_diff | commit_gpu;
            if (commit_b != 0u) {
                before_buf.Store(b_addr, (b & ~commit_b) | (t & cpu_diff) | (s & commit_gpu));
            }
        }
    }
}
"#;

/// Small atomic probe shader to test whether `InterlockedCompareExchange` works on L0 UAV.
const PROBE_ATOMIC_HLSL: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer buf : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint orig = 0u;
    buf.InterlockedCompareExchange(0u, 0x11223344u, 0x55667788u, orig);
    buf.Store(4u, orig);
}
"#;

/// Compiled compute pipelines + reusable L1/readback delta buffers for `SrvDirectUpload`.
pub(crate) struct EnginePipelines {
    pub(crate) mode: SyncMode,
    pub(crate) atomic_flags: u32,
    pub(crate) l0_copy_to_l1: bool,
    pub(crate) bidir_root: ID3D12RootSignature,
    pub(crate) bidir_pso: ID3D12PipelineState,
    pub(crate) srv_root: ID3D12RootSignature,
    pub(crate) srv_pso: ID3D12PipelineState,
    pub(crate) deltas_gpu: ID3D12Resource,
    pub(crate) deltas_readback: ID3D12Resource,
    pub(crate) deltas_readback_ptr: usize,
    pub(crate) deltas_zero: ID3D12Resource,
    pub(crate) dirty_dummy_gpu: ID3D12Resource,
}

// SAFETY: D3D12 interfaces are free-threaded; `deltas_readback_ptr` is accessed under lock.
unsafe impl Send for EnginePipelines {}
// SAFETY: same as above.
unsafe impl Sync for EnginePipelines {}

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(4),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

fn compile_cs(source: &str) -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: valid pointers and lifetimes through the call.
    let res = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("gpu_sync"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = res {
        if let Some(err_blob) = errors {
            // SAFETY: valid DXBC compiler error blob.
            let msg = unsafe {
                std::slice::from_raw_parts(err_blob.GetBufferPointer().cast::<u8>(), err_blob.GetBufferSize())
            };
            tracing::warn!("gpu_sync shader compile failed: {}", String::from_utf8_lossy(msg));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: valid compiled bytecode blob.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    Ok(bytes.to_vec())
}

fn make_compute_pso(device: &ID3D12Device, source: &str) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    let bytes = compile_cs(source)?;
    // SAFETY: valid device and shader bytecode with embedded root signature.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytes)?;
        let pso: ID3D12PipelineState = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok((root, pso))
    }
}

fn make_committed(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    initial_state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES { Type: heap_type, ..Default::default() };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: valid D3D12 committed resource creation.
    unsafe {
        device.CreateCommittedResource(
            &props,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(size, flags),
            initial_state,
            None,
            &mut resource,
        )?;
    }
    resource.ok_or_else(windows::core::Error::empty)
}

fn place_l0_buffer(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    size: u64,
    allow_uav: bool,
) -> windows::core::Result<ID3D12Resource> {
    let flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER
        | if allow_uav {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: live heap, offset 0, size <= heap size, valid flags.
    unsafe {
        device.CreatePlacedResource(
            heap,
            0,
            &buffer_desc(size, flags),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(windows::core::Error::empty)
}

impl EnginePipelines {
    pub(crate) fn new(
        device: &ID3D12Device,
        mode: SyncMode,
        atomic_flags: u32,
        l0_copy_to_l1: bool,
    ) -> windows::core::Result<Self> {
        let (bidir_root, bidir_pso) = make_compute_pso(device, SYNC_BIDIR_HLSL)?;
        let (srv_root, srv_pso) = make_compute_pso(device, SYNC_SRV_HLSL)?;
        let deltas_gpu = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            SRV_DELTA_BUFFER_BYTES,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let deltas_readback = make_committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            SRV_DELTA_BUFFER_BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let deltas_zero = make_committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            256,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let dirty_dummy_gpu = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let mut rb_ptr = std::ptr::null_mut();
        let mut zero_ptr = std::ptr::null_mut();
        // SAFETY: persistent CPU mappings of readback and upload scratch buffers.
        unsafe {
            deltas_readback.Map(0, None, Some(&mut rb_ptr))?;
            deltas_zero.Map(0, None, Some(&mut zero_ptr))?;
            std::ptr::write_bytes(zero_ptr.cast::<u8>(), 0, 256);
            deltas_zero.Unmap(0, None);
        }
        Ok(Self {
            mode,
            atomic_flags,
            l0_copy_to_l1,
            bidir_root,
            bidir_pso,
            srv_root,
            srv_pso,
            deltas_gpu,
            deltas_readback,
            deltas_readback_ptr: rb_ptr as usize,
            deltas_zero,
            dirty_dummy_gpu,
        })
    }
}

/// Coalesces unsorted/overlapping/adjacent byte ranges `[from, to)` clamped to `[0, limit)`.
pub(crate) fn coalesce_ranges(ranges: &[(usize, usize)], limit: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = ranges
        .iter()
        .filter_map(|&(a, b)| {
            let lo = a.min(limit);
            let hi = b.min(limit);
            (lo < hi).then_some((lo, hi))
        })
        .collect();
    if out.len() < 2 {
        return out;
    }
    out.sort_unstable();
    let mut write = 0usize;
    for i in 1..out.len() {
        let (lo, hi) = out[i];
        if lo <= out[write].1 {
            out[write].1 = out[write].1.max(hi);
        } else {
            write += 1;
            out[write] = (lo, hi);
        }
    }
    out.truncate(write + 1);
    out
}

/// Records one or more `SYNC_BIDIR_HLSL` dispatches covering `ranges` (pre-coalesced, non-empty).
///
/// Inserts a UAV barrier between consecutive disjoint ranges only when they share a 4-byte dword
/// boundary (e.g. `[0, 1)` and `[2, 3)`), preventing any intra-dword write race.
///
/// # Safety
/// `list` must be open; all three buffers must be valid, large enough for `bias + range_end`
/// (rounded up to 4 bytes), and in a state compatible with `UNORDERED_ACCESS` (or `COMMON`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_bidir_ranges(
    list: &ID3D12GraphicsCommandList,
    pipelines: &EnginePipelines,
    title_va: u64,
    title_bias: u32,
    shadow_va: u64,
    shadow_bias: u32,
    before_va: u64,
    before_bias: u32,
    ranges: &[(usize, usize)],
    op: SyncOp,
) {
    // SAFETY: delegates to tracked helper with dirty-page bitmask disabled.
    unsafe {
        record_bidir_ranges_tracked(
            list,
            pipelines,
            title_va,
            title_bias,
            shadow_va,
            shadow_bias,
            before_va,
            before_bias,
            ranges,
            op,
            None,
        );
    }
}

/// Records `SYNC_BIDIR_HLSL` dispatches with an optional per-64 KiB-page dirty bitmask UAV
/// `(dirty_va, dirty_words)` populated whenever GPU bytes are committed to `title_buf`.
///
/// # Safety
/// Same as [`record_bidir_ranges`]; if `dirty` is `Some((va, words))`, `va` must point to
/// `words * 4` bytes of UAV-accessible memory.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_bidir_ranges_tracked(
    list: &ID3D12GraphicsCommandList,
    pipelines: &EnginePipelines,
    title_va: u64,
    title_bias: u32,
    shadow_va: u64,
    shadow_bias: u32,
    before_va: u64,
    before_bias: u32,
    ranges: &[(usize, usize)],
    op: SyncOp,
    dirty: Option<(u64, u32)>,
) {
    if ranges.is_empty() {
        return;
    }
    const MAX_WORDS_PER_DISPATCH: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION * DWORDS_PER_GROUP;
    // SAFETY: caller upholds command list and resource invariants.
    unsafe {
        let (dirty_va, dirty_words) =
            dirty.unwrap_or_else(|| (pipelines.dirty_dummy_gpu.GetGPUVirtualAddress(), 0));
        list.SetComputeRootSignature(&pipelines.bidir_root);
        list.SetPipelineState(&pipelines.bidir_pso);
        list.SetComputeRootUnorderedAccessView(1, title_va);
        list.SetComputeRootUnorderedAccessView(2, shadow_va);
        list.SetComputeRootUnorderedAccessView(3, before_va);
        list.SetComputeRootUnorderedAccessView(4, dirty_va);
        let mut prev_last_word: Option<u32> = None;
        for &(from, to) in ranges {
            if from >= to {
                continue;
            }
            let first_word = (from as u32) >> 2;
            let last_word = ((to as u32) - 1) >> 2;
            if prev_last_word == Some(first_word) {
                list.ResourceBarrier(&[uav_barrier()]);
            }
            let total_words = last_word - first_word + 1;
            let mut done = 0u32;
            while done < total_words {
                let n = (total_words - done).min(MAX_WORDS_PER_DISPATCH);
                let groups = n.div_ceil(DWORDS_PER_GROUP);
                let constants = [
                    title_bias,
                    shadow_bias,
                    before_bias,
                    from as u32,
                    to as u32,
                    first_word + done,
                    n,
                    op as u32,
                    pipelines.atomic_flags,
                    dirty_words,
                ];
                list.SetComputeRoot32BitConstants(0, 10, constants.as_ptr().cast(), 0);
                list.Dispatch(groups, 1, 1);
                done += n;
            }
            prev_last_word = Some(last_word);
        }
    }
}

/// Records `SYNC_SRV_HLSL` dispatches (`SrvDirectUpload` mode) covering `ranges`.
///
/// For `SyncOp::Bidirectional3Way`, callers must first zero the 8-byte header of
/// `pipelines.deltas_gpu` and after dispatch copy `pipelines.deltas_gpu` to
/// `pipelines.deltas_readback` before applying deltas with [`apply_srv_deltas_to_title`].
///
/// # Safety
/// `list` must be open; all buffers must be valid and in compatible states (`title_va` SRV,
/// `shadow_va`/`before_va`/`deltas_gpu` UAV).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn record_srv_ranges(
    list: &ID3D12GraphicsCommandList,
    pipelines: &EnginePipelines,
    title_va: u64,
    title_bias: u32,
    shadow_va: u64,
    shadow_bias: u32,
    before_va: u64,
    before_bias: u32,
    ranges: &[(usize, usize)],
    op: SyncOp,
    max_deltas: u32,
) {
    if ranges.is_empty() {
        return;
    }
    const MAX_WORDS_PER_DISPATCH: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION * DWORDS_PER_GROUP;
    // SAFETY: caller upholds command list and resource invariants.
    unsafe {
        list.SetComputeRootSignature(&pipelines.srv_root);
        list.SetPipelineState(&pipelines.srv_pso);
        list.SetComputeRootShaderResourceView(1, title_va);
        list.SetComputeRootUnorderedAccessView(2, shadow_va);
        list.SetComputeRootUnorderedAccessView(3, before_va);
        // Bind `deltas_gpu` exclusively during write-back passes so `Init`/`Upload3Way` on
        // `host.queue` never alias `deltas_gpu` with `srv_rig`'s queue.
        let deltas_va = if matches!(op, SyncOp::Bidirectional3Way | SyncOp::GpuWriteBackOnly) {
            pipelines.deltas_gpu.GetGPUVirtualAddress()
        } else {
            before_va
        };
        list.SetComputeRootUnorderedAccessView(4, deltas_va);
        let mut prev_last_word: Option<u32> = None;
        for &(from, to) in ranges {
            if from >= to {
                continue;
            }
            let first_word = (from as u32) >> 2;
            let last_word = ((to as u32) - 1) >> 2;
            if prev_last_word == Some(first_word) {
                list.ResourceBarrier(&[uav_barrier()]);
            }
            let total_words = last_word - first_word + 1;
            let mut done = 0u32;
            while done < total_words {
                let n = (total_words - done).min(MAX_WORDS_PER_DISPATCH);
                let groups = n.div_ceil(DWORDS_PER_GROUP);
                let constants = [
                    title_bias,
                    shadow_bias,
                    before_bias,
                    from as u32,
                    to as u32,
                    first_word + done,
                    n,
                    op as u32,
                    max_deltas,
                ];
                list.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
                list.Dispatch(groups, 1, 1);
                done += n;
            }
            prev_last_word = Some(last_word);
        }
    }
}

/// Records DMA-engine `CopyBufferRegion` `L0 -> L1` (`shadow`) followed by full-VRAM-bandwidth
/// `L1 -> L1` (`shadow -> before`) for `SyncOp::Init` ranges.
///
/// Saturates the PCIe DMA engine (`~17 GB/s` on RTX 4070 SUPER) and supports arbitrary 1-byte
/// aligned ranges. If `then_compute` is true, transitions all three buffers into the states
/// required by a subsequent `record_srv_ranges(..., SyncOp::Upload3Way)` in the same command list.
///
/// # Safety
/// `list` must be open; all three buffers must be in `COMMON` state and large enough for `ranges`.
pub(crate) unsafe fn record_init_dma(
    list: &ID3D12GraphicsCommandList,
    title_buf: &ID3D12Resource,
    title_bias: u32,
    shadow_buf: &ID3D12Resource,
    before_buf: &ID3D12Resource,
    ranges: &[(usize, usize)],
    then_compute: bool,
) {
    if ranges.is_empty() {
        return;
    }
    // SAFETY: caller upholds command list and resource invariants.
    unsafe {
        for &(from, to) in ranges {
            if from < to {
                list.CopyBufferRegion(
                    shadow_buf,
                    from as u64,
                    title_buf,
                    title_bias as u64 + from as u64,
                    (to - from) as u64,
                );
            }
        }
        list.ResourceBarrier(&[transition(
            shadow_buf,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for &(from, to) in ranges {
            if from < to {
                list.CopyBufferRegion(before_buf, from as u64, shadow_buf, from as u64, (to - from) as u64);
            }
        }
        if then_compute {
            list.ResourceBarrier(&[
                transition(
                    title_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
                transition(
                    shadow_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                transition(
                    before_buf,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
        }
    }
}

/// Applies compact GPU-only dword deltas emitted by `SYNC_SRV_HLSL` into title memory at `title_ptr`.
///
/// Each 16-byte record holds `(w_off, gpu_word, gpu_only_mask, base_t_word)`. Before writing any
/// byte, reads current live title memory safely (`host::read_into`), checks that the byte still
/// matches `base_t_word` (so concurrent CPU writes between GPU dispatch and CPU apply always win),
/// notifies write-watch/guards (`watch::landing`), and writes ONLY the surviving GPU bytes safely
/// via `safecopy` (or `host::write_mapped`), never crashing on `PAGE_READONLY` guards or
/// decommitted pages. Returns `(emitted_dwords, overflow)`.
///
/// # Safety
/// `readback_ptr` must point to a mapped `SRV_DELTA_BUFFER_BYTES` buffer populated by `SYNC_SRV_HLSL`;
/// `title_ptr` must be the base address of the `region_len`-byte title region.
pub(crate) unsafe fn apply_srv_deltas_to_title(
    readback_ptr: *const u8,
    max_deltas: u32,
    title_ptr: *mut u8,
    region_len: usize,
) -> (u32, bool) {
    let safe_copy = crate::safecopy::on();
    // SAFETY: header is 4 x u32 at offset 0 of mapped readback buffer.
    let (raw_count, overflow_flag) = unsafe {
        let hdr = readback_ptr.cast::<u32>();
        (hdr.read_volatile(), hdr.add(1).read_volatile())
    };
    let count = raw_count.min(max_deltas);
    // SAFETY: records start at byte offset 16, each 4 x u32.
    let records = unsafe { readback_ptr.add(SRV_DELTA_HEADER_BYTES as usize).cast::<u32>() };
    for i in 0..count as usize {
        // SAFETY: `i < count <= max_deltas`.
        let (w_off, gpu_word, mask, base_t) = unsafe {
            let rec = records.add(i * 4);
            (rec.read() as usize, rec.add(1).read(), rec.add(2).read(), rec.add(3).read())
        };
        if w_off >= region_len || mask == 0 {
            continue;
        }
        let span = (region_len - w_off).min(4);
        // SAFETY: `w_off < region_len`; title memory access is guarded via `read_into` / `safecopy`.
        let dst = unsafe { title_ptr.add(w_off) };
        let mut cur = [0u8; 4];
        if !crate::host::read_into(dst.cast_const(), &mut cur[..span]) {
            continue;
        }
        let base_bytes = base_t.to_le_bytes();
        let gpu_bytes = gpu_word.to_le_bytes();
        let mut merged = cur;
        let mut any_change = false;
        for k in 0..span {
            if ((mask >> (k * 8)) & 0xFF) != 0 && cur[k] == base_bytes[k] && gpu_bytes[k] != cur[k] {
                merged[k] = gpu_bytes[k];
                any_change = true;
            }
        }
        if !any_change {
            continue;
        }
        crate::watch::landing(dst as usize, span);
        if safe_copy {
            let since = crate::watch::changes();
            for k in 0..span {
                if merged[k] != cur[k] {
                    // SAFETY: within `region_len`; faults on RO/unmapped pages are caught by `safecopy`.
                    let byte_ptr = unsafe { dst.add(k) };
                    let put = || unsafe { crate::safecopy::copy(byte_ptr, &merged[k], 1) };
                    if !put() {
                        let _ = crate::watch::again_after_change(since, put);
                    }
                }
            }
        } else {
            for k in 0..span {
                if merged[k] != cur[k] {
                    // SAFETY: within `region_len`; `write_mapped` handles RO/unmapped pages safely.
                    let byte_ptr = unsafe { dst.add(k) };
                    crate::host::write_mapped(byte_ptr, &merged[k], 1);
                }
            }
        }
    }
    (count, overflow_flag != 0 || raw_count > max_deltas)
}

/// Reference CPU 3-way merge oracle for verification and benchmarks.
///
/// Semantics (byte-exact): for every byte `i` inside `ranges`:
/// - `title[i] != before[i]`: CPU/title wins -> `shadow[i] = title[i]`, `before[i] = title[i]`.
/// - `title[i] == before[i] && shadow[i] != before[i]`: GPU wins -> `title[i] = shadow[i]`, `before[i] = shadow[i]`.
/// - Otherwise: untouched.
#[cfg(test)]
pub(crate) fn cpu_three_way_merge_oracle(
    title: &mut [u8],
    shadow: &mut [u8],
    before: &mut [u8],
    ranges: &[(usize, usize)],
    op: SyncOp,
) {
    let limit = title.len().min(shadow.len()).min(before.len());
    for (from, to) in coalesce_ranges(ranges, limit) {
        match op {
            SyncOp::Init => {
                shadow[from..to].copy_from_slice(&title[from..to]);
                before[from..to].copy_from_slice(&title[from..to]);
            }
            SyncOp::Upload3Way => {
                for i in from..to {
                    if title[i] != before[i] {
                        shadow[i] = title[i];
                        before[i] = title[i];
                    }
                }
            }
            SyncOp::Bidirectional3Way => {
                for i in from..to {
                    let t = title[i];
                    let s = shadow[i];
                    let b = before[i];
                    if t != b {
                        shadow[i] = t;
                        before[i] = t;
                    } else if s != b {
                        title[i] = s;
                        before[i] = s;
                    }
                }
            }
            SyncOp::GpuWriteBackOnly => {
                for i in from..to {
                    let s = shadow[i];
                    let b = before[i];
                    if s != b {
                        let t = title[i];
                        if t != b {
                            shadow[i] = t;
                            before[i] = t;
                        } else {
                            title[i] = s;
                            before[i] = s;
                        }
                    }
                }
            }
        }
    }
}

/// Drains D3D12 debug layer messages from `info_queue`, returning any errors/corruptions.
pub(crate) fn drain_debug_errors(info_queue: Option<&ID3D12InfoQueue>) -> Vec<String> {
    let Some(q) = info_queue else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    // SAFETY: standard D3D12 InfoQueue read & clear.
    unsafe {
        let count = q.GetNumStoredMessages();
        for i in 0..count {
            let mut size = 0usize;
            if q.GetMessage(i, None, &mut size).is_err() || size == 0 {
                continue;
            }
            let mut buf = vec![0u64; size.div_ceil(8)];
            let ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if q.GetMessage(i, Some(ptr), &mut size).is_ok() {
                let m = &*ptr;
                let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                let text = String::from_utf8_lossy(text).trim_end_matches('\0').to_owned();
                if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                    errors.push(format!("[sev={} id={}] {text}", m.Severity.0, m.ID.0));
                }
            }
        }
        q.ClearStoredMessages();
    }
    errors
}

/// Lightweight standalone D3D12 execution rig (its own queue, list, fence, timestamps) for probing
/// and benchmarking without touching the main host submission state.
pub(crate) struct ProbeRig {
    pub(crate) device: ID3D12Device,
    pub(crate) queue: ID3D12CommandQueue,
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) fence: ID3D12Fence,
    pub(crate) event: HANDLE,
    pub(crate) fence_val: u64,
    pub(crate) query_heap: ID3D12QueryHeap,
    pub(crate) query_readback: ID3D12Resource,
    pub(crate) query_ptr: *const u64,
    pub(crate) timestamp_freq: u64,
    pub(crate) info_queue: Option<ID3D12InfoQueue>,
}

impl Drop for ProbeRig {
    fn drop(&mut self) {
        // SAFETY: own event handle and mapped readback buffer.
        unsafe {
            self.query_readback.Unmap(0, None);
            let _ = CloseHandle(self.event);
        }
    }
}

impl ProbeRig {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain D3D12 command/query/fence creation on live device.
        unsafe {
            let info_queue = device.cast::<ID3D12InfoQueue>().ok();
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let mut query_heap: Option<ID3D12QueryHeap> = None;
            device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 4,
                    NodeMask: 0,
                },
                &mut query_heap,
            )?;
            let query_heap = query_heap.ok_or_else(windows::core::Error::empty)?;
            let query_readback = make_committed(
                device,
                D3D12_HEAP_TYPE_READBACK,
                64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let mut ptr = std::ptr::null_mut();
            query_readback.Map(0, None, Some(&mut ptr))?;
            let timestamp_freq = queue.GetTimestampFrequency().unwrap_or(0);
            Ok(Self {
                device: device.clone(),
                queue,
                allocator,
                list,
                fence,
                event,
                fence_val: 0,
                query_heap,
                query_readback,
                query_ptr: ptr.cast::<u64>(),
                timestamp_freq,
                info_queue,
            })
        }
    }

    /// Executes `record`, waits for GPU completion, and returns `(gpu_us, cpu_submit_us)`.
    pub(crate) fn run_timed(
        &mut self,
        record: impl FnOnce(&ID3D12GraphicsCommandList),
    ) -> windows::core::Result<(f64, f64)> {
        // SAFETY: own allocator/list/queue/query heap, synchronized one run at a time.
        unsafe {
            let cpu_t0 = std::time::Instant::now();
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            self.list.EndQuery(&self.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            record(&self.list);
            self.list.EndQuery(&self.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            self.list.ResolveQueryData(
                &self.query_heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                2,
                &self.query_readback,
                0,
            );
            self.list.Close()?;
            let cmd: ID3D12CommandList = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            let target_fence = self.fence_val + 1;
            self.queue.Signal(&self.fence, target_fence)?;
            self.fence_val = target_fence;
            let cpu_submit_us = cpu_t0.elapsed().as_secs_f64() * 1e6;
            self.fence.SetEventOnCompletion(self.fence_val, self.event)?;
            if WaitForSingleObject(self.event, 15_000) != windows::Win32::Foundation::WAIT_OBJECT_0 {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            self.device.GetDeviceRemovedReason()?;
            let t0 = self.query_ptr.read_volatile();
            let t1 = self.query_ptr.add(1).read_volatile();
            let gpu_us = if self.timestamp_freq > 0 && t1 >= t0 {
                (t1 - t0) as f64 * 1e6 / self.timestamp_freq as f64
            } else {
                0.0
            };
            Ok((gpu_us, cpu_submit_us))
        }
    }

    pub(crate) fn read_l1(&mut self, src: &ID3D12Resource, offset: u64, len: usize) -> windows::core::Result<Vec<u8>> {
        let rb = make_committed(
            &self.device,
            D3D12_HEAP_TYPE_READBACK,
            len.max(4) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        // SAFETY: valid live buffers.
        self.run_timed(|list| unsafe {
            list.CopyBufferRegion(&rb, 0, src, offset, len as u64);
        })?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: mapped readback buffer.
        unsafe {
            rb.Map(0, None, Some(&mut ptr))?;
            let out = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
            rb.Unmap(0, None);
            Ok(out)
        }
    }

    pub(crate) fn write_l1(&mut self, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) -> windows::core::Result<()> {
        let up = make_committed(
            &self.device,
            D3D12_HEAP_TYPE_UPLOAD,
            bytes.len().max(4) as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut();
        // SAFETY: mapped upload buffer.
        unsafe {
            up.Map(0, None, Some(&mut ptr))?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            up.Unmap(0, None);
        }
        // SAFETY: valid live buffers.
        self.run_timed(|list| unsafe {
            list.CopyBufferRegion(dst, offset, &up, 0, bytes.len() as u64);
        })?;
        Ok(())
    }
}

fn adapter_info(device: &ID3D12Device) -> (String, bool) {
    // SAFETY: plain DXGI LUID lookup.
    unsafe {
        let luid = device.GetAdapterLuid();
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return ("unknown".to_owned(), false);
        };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid) else {
            return ("unknown".to_owned(), false);
        };
        let Ok(desc) = adapter.GetDesc1() else {
            return ("unknown".to_owned(), false);
        };
        let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_owned();
        let sw = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        (name, sw)
    }
}

/// Runs a real D3D12 hardware/WARP capability probe for `OpenExistingHeapFromAddress1` L0
/// compute SRV reads, compute UAV writes, L0<->L1 copies, and L0 shader atomics.
pub(crate) fn probe_device(device: &ID3D12Device) -> ProbeReport {
    let (adapter_name, is_software) = adapter_info(device);
    let mut arch1 = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    // SAFETY: plain D3D12 feature query.
    let cache_coherent_uma = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut arch1).cast(),
                std::mem::size_of::<D3D12_FEATURE_DATA_ARCHITECTURE1>() as u32,
            )
            .is_ok()
            && arch1.CacheCoherentUMA.as_bool()
    };
    let has_device13 = device.cast::<ID3D12Device13>().is_ok();
    let mut report = ProbeReport {
        adapter_name,
        is_software,
        cache_coherent_uma,
        has_device13,
        l0_open_ok: false,
        l0_srv_read_ok: false,
        l0_uav_write_ok: false,
        l0_copy_to_l1_ok: false,
        l0_copy_from_l1_ok: false,
        l0_atomics_ok: false,
        l0_heap_flags: 0,
        mode: SyncMode::Unsupported,
        debug_errors: Vec::new(),
        notes: Vec::new(),
    };
    if !has_device13 {
        report.notes.push("ID3D12Device13 unavailable".to_owned());
        return report;
    }
    let mut rig = match ProbeRig::new(device) {
        Ok(r) => r,
        Err(e) => {
            report.notes.push(format!("probe rig failed: {e}"));
            return report;
        }
    };
    let _ = drain_debug_errors(rig.info_queue.as_ref());
    let pipelines = match EnginePipelines::new(device, SyncMode::BidirectionalUav, 0, false) {
        Ok(p) => p,
        Err(e) => {
            report.notes.push(format!("pipeline compile failed: {e}"));
            report.debug_errors = drain_debug_errors(rig.info_queue.as_ref());
            return report;
        }
    };

    const PROBE_BYTES: usize = 4 * (K64 as usize);
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    // SAFETY: probe-owned VirtualAlloc block, freed before returning.
    unsafe {
        let base = mem::VirtualAlloc(
            std::ptr::null(),
            PROBE_BYTES,
            mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
            mem::PAGE_READWRITE,
        )
        .cast::<u8>();
        if base.is_null() {
            report.notes.push("VirtualAlloc failed".to_owned());
            return report;
        }
        let part_ptr = base.add(K64 as usize);
        let part_bytes = 2 * (K64 as usize);
        let title_slice = std::slice::from_raw_parts_mut(part_ptr, part_bytes);
        for (i, b) in title_slice.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(37) + 13) & 0xFF) as u8;
        }

        let heap = match open(device, OpenFrom::Part(part_ptr as usize, part_bytes)) {
            Ok(h) => h,
            Err(e) => {
                report.notes.push(format!("OpenExistingHeapFromAddress1 failed: {e}"));
                mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
                report.debug_errors = drain_debug_errors(rig.info_queue.as_ref());
                return report;
            }
        };
        report.l0_open_ok = true;
        report.l0_heap_flags = heap.GetDesc().Flags.0 as u32;

        let Ok(shadow_l1) = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            part_bytes as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        ) else {
            drop(heap);
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
            return report;
        };
        let Ok(before_l1) = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            part_bytes as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        ) else {
            drop((shadow_l1, heap));
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
            return report;
        };

        // 1. Place L0 buffer (try ALLOW_UNORDERED_ACCESS first, else drain expected debug error & retry SRV-only).
        let (l0_buf, l0_has_uav) = match place_l0_buffer(device, &heap, part_bytes as u64, true) {
            Ok(buf) => (buf, true),
            Err(e) => {
                let _ = drain_debug_errors(rig.info_queue.as_ref());
                report.notes.push(format!("L0 placed buffer with UAV refused ({e}); retrying SRV-only"));
                match place_l0_buffer(device, &heap, part_bytes as u64, false) {
                    Ok(buf) => (buf, false),
                    Err(e2) => {
                        report.notes.push(format!("L0 SRV-only placed buffer refused: {e2}"));
                        drop((before_l1, shadow_l1, heap));
                        mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
                        report.debug_errors = drain_debug_errors(rig.info_queue.as_ref());
                        return report;
                    }
                }
            }
        };

        // 2. Probe CopyBufferRegion L0 -> L1.
        if rig
            .run_timed(|list| {
                list.CopyBufferRegion(&shadow_l1, 0, &l0_buf, 0, part_bytes as u64);
            })
            .is_ok()
            && rig.read_l1(&shadow_l1, 0, part_bytes).is_ok_and(|got| got == title_slice)
        {
            report.l0_copy_to_l1_ok = true;
        }

        // 3. Probe L0 compute SRV read -> L1 shadow & before (including live CPU mutation visibility).
        title_slice[0] ^= 0xA5;
        title_slice[4097] ^= 0x5A;
        title_slice[part_bytes - 1] ^= 0x3C;
        let l0_va = l0_buf.GetGPUVirtualAddress();
        let s_va = shadow_l1.GetGPUVirtualAddress();
        let b_va = before_l1.GetGPUVirtualAddress();
        let srv_ok = rig
            .run_timed(|list| {
                record_srv_ranges(
                    list,
                    &pipelines,
                    l0_va,
                    0,
                    s_va,
                    0,
                    b_va,
                    0,
                    &[(0, part_bytes)],
                    SyncOp::Init,
                    SRV_MAX_DELTAS,
                );
            })
            .is_ok()
            && rig.read_l1(&shadow_l1, 0, part_bytes).is_ok_and(|got| got == title_slice)
            && rig.read_l1(&before_l1, 0, part_bytes).is_ok_and(|got| got == title_slice);
        report.l0_srv_read_ok = srv_ok;

        // 4. Probe L0 CopyBufferRegion L1 -> L0.
        let mut copy_back_test = title_slice[..4096].to_vec();
        copy_back_test[12] ^= 0x77;
        if rig.write_l1(&shadow_l1, 0, &copy_back_test).is_ok()
            && rig
                .run_timed(|list| {
                    list.CopyBufferRegion(&l0_buf, 0, &shadow_l1, 0, 4096);
                })
                .is_ok()
            && title_slice[..4096] == copy_back_test[..]
        {
            report.l0_copy_from_l1_ok = true;
        }

        // 5. Probe L0 compute UAV 3-way sync write + CPU conflict wins + CPU visibility.
        if l0_has_uav && srv_ok {
            // Re-baseline cleanly.
            let _ = rig.run_timed(|list| {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    l0_va,
                    0,
                    s_va,
                    0,
                    b_va,
                    0,
                    &[(0, part_bytes)],
                    SyncOp::Init,
                );
            });
            // Mutate GPU-only bytes at [16..20), CPU+GPU conflict at byte 20, CPU-only at byte 21.
            let mut s_mut = title_slice[..64].to_vec();
            s_mut[16] ^= 0x11;
            s_mut[17] ^= 0x22;
            s_mut[18] ^= 0x33;
            s_mut[19] ^= 0x44;
            s_mut[20] ^= 0x99; // GPU writes 20
            let _ = rig.write_l1(&shadow_l1, 0, &s_mut);
            let cpu_win_byte = title_slice[20] ^ 0x55;
            let cpu_only_byte = title_slice[21] ^ 0x66;
            title_slice[20] = cpu_win_byte;
            title_slice[21] = cpu_only_byte;

            let bidir_ran = rig.run_timed(|list| {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    l0_va,
                    0,
                    s_va,
                    0,
                    b_va,
                    0,
                    &[(0, 64)],
                    SyncOp::Bidirectional3Way,
                );
            });
            if bidir_ran.is_ok()
                && title_slice[16..20] == s_mut[16..20]
                && title_slice[20] == cpu_win_byte
                && title_slice[21] == cpu_only_byte
                && let (Ok(s_after), Ok(b_after)) = (rig.read_l1(&shadow_l1, 0, 64), rig.read_l1(&before_l1, 0, 64))
                && s_after == title_slice[..64]
                && b_after == title_slice[..64]
            {
                report.l0_uav_write_ok = true;
            }

            // 6. Probe L0 shader atomics (diagnostic).
            if let Ok((a_root, a_pso)) = make_compute_pso(device, PROBE_ATOMIC_HLSL) {
                title_slice[0..8].copy_from_slice(&[0x44, 0x33, 0x22, 0x11, 0, 0, 0, 0]);
                if rig
                    .run_timed(|list| {
                        list.SetComputeRootSignature(&a_root);
                        list.SetPipelineState(&a_pso);
                        list.SetComputeRootUnorderedAccessView(0, l0_va);
                        list.Dispatch(1, 1, 1);
                    })
                    .is_ok()
                {
                    let new_val = u32::from_le_bytes(title_slice[0..4].try_into().unwrap());
                    let orig_val = u32::from_le_bytes(title_slice[4..8].try_into().unwrap());
                    report.l0_atomics_ok = new_val == 0x5566_7788 && orig_val == 0x1122_3344;
                }
            }
        }

        report.debug_errors = drain_debug_errors(rig.info_queue.as_ref());
        report.mode = if !report.debug_errors.is_empty() {
            SyncMode::Unsupported
        } else if report.l0_uav_write_ok && report.l0_srv_read_ok && report.l0_atomics_ok {
            SyncMode::BidirectionalUav
        } else if report.l0_srv_read_ok {
            SyncMode::SrvDirectUpload
        } else {
            SyncMode::Unsupported
        };

        drop((l0_buf, before_l1, shadow_l1, heap));
        mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
    }
    report
}

/// Pending upload range for a GPU-synced shadow.
#[derive(Clone, Copy, Debug)]
struct UploadReq {
    from: usize,
    to: usize,
    init: bool,
}

/// Per-shadow GPU-direct sync state attached to a `Shadow` when `VICTORY_GPU_SYNC=1`.
///
/// Replaces both `Card::staging` (CPU RAM copy) and `Card::before` (hash buffer + readback/merge):
/// zero CPU staging allocations, shaders stay in L1 VRAM, and CPU/GPU sync runs directly on GPU.
pub(crate) struct SyncedShadow {
    _title_heap: ID3D12Heap,
    pub(crate) title_buffer: ID3D12Resource,
    pub(crate) title_bias: u32,
    pub(crate) before_buffer: ID3D12Resource,
    pub(crate) size: u64,
    pub(crate) mode: SyncMode,
    pending_uploads: Mutex<Vec<UploadReq>>,
    has_uploads: AtomicBool,
    /// Per-64 KiB page initialization bitset (`1` = both `shadow.buffer` and `before_buffer`
    /// initialized from title RAM via `SyncOp::Init`).
    initialized_pages: Mutex<Vec<u64>>,
    page_count: usize,
}

// SAFETY: D3D12 resources are free-threaded; mutable vectors are behind Mutex.
unsafe impl Send for SyncedShadow {}
// SAFETY: same as above.
unsafe impl Sync for SyncedShadow {}

impl SyncedShadow {
    /// Queues `[from, to)` (byte offsets relative to `Shadow::start`) for zero-CPU-copy upload.
    pub(crate) fn queue_upload(&self, from: usize, to: usize, init: bool) {
        if from >= to {
            return;
        }
        let mut guard = self.pending_uploads.lock().unwrap_or_else(|e| e.into_inner());
        guard.push(UploadReq { from, to, init });
        self.has_uploads.store(true, Ordering::Release);
    }

    pub(crate) fn has_uploads(&self) -> bool {
        self.has_uploads.load(Ordering::Acquire)
    }

    /// Returns whether the 64 KiB page containing relative byte offset `rel_off` has been initialized.
    pub(crate) fn is_page_initialized(&self, rel_off: usize) -> bool {
        let page = rel_off >> 16;
        if page >= self.page_count {
            return true;
        }
        let bits = self.initialized_pages.lock().unwrap_or_else(|e| e.into_inner());
        (bits[page >> 6] & (1u64 << (page & 63))) != 0
    }

    /// Returns coalesced full-64 KiB-page ranges (clamped to `limit`) for every uninitialized
    /// page intersecting `ranges`.
    pub(crate) fn uninitialized_page_ranges(&self, ranges: &[(usize, usize)], limit: usize) -> Vec<(usize, usize)> {
        if ranges.is_empty() || limit == 0 {
            return Vec::new();
        }
        let bits = self.initialized_pages.lock().unwrap_or_else(|e| e.into_inner());
        let mut missing = Vec::new();
        for &(a, b) in ranges {
            let lo = a.min(limit);
            let hi = b.min(limit);
            if lo >= hi {
                continue;
            }
            let first_page = lo >> 16;
            let last_page = ((hi - 1) >> 16).min(self.page_count.saturating_sub(1));
            for p in first_page..=last_page {
                if (bits[p >> 6] & (1u64 << (p & 63))) == 0 {
                    let p_lo = p << 16;
                    let p_hi = ((p + 1) << 16).min(limit);
                    if p_lo < p_hi {
                        missing.push((p_lo, p_hi));
                    }
                }
            }
        }
        coalesce_ranges(&missing, limit)
    }

    /// Marks every 64 KiB page wholly covered (up to `limit`) by `ranges` as initialized.
    pub(crate) fn mark_pages_initialized(&self, ranges: &[(usize, usize)], limit: usize) {
        if ranges.is_empty() || limit == 0 {
            return;
        }
        let mut bits = self.initialized_pages.lock().unwrap_or_else(|e| e.into_inner());
        for &(a, b) in ranges {
            let lo = a.min(limit);
            let hi = b.min(limit);
            if lo >= hi {
                continue;
            }
            let first_page = lo >> 16;
            let last_page = ((hi - 1) >> 16).min(self.page_count.saturating_sub(1));
            for p in first_page..=last_page {
                let p_lo = p << 16;
                let p_hi = ((p + 1) << 16).min(limit);
                if lo <= p_lo && hi >= p_hi {
                    bits[p >> 6] |= 1u64 << (p & 63);
                }
            }
        }
    }

    /// Drains pending upload requests, coalesced into `(init_ranges, upload3way_ranges)`.
    pub(crate) fn take_uploads(&self, limit: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        if !self.has_uploads.load(Ordering::Acquire) {
            return (Vec::new(), Vec::new());
        }
        let reqs = {
            let mut guard = self.pending_uploads.lock().unwrap_or_else(|e| e.into_inner());
            let taken = std::mem::take(&mut *guard);
            self.has_uploads.store(false, Ordering::Release);
            taken
        };
        let mut init_raw = Vec::new();
        let mut up_raw = Vec::new();
        for r in reqs {
            if r.init {
                init_raw.push((r.from, r.to));
            } else {
                up_raw.push((r.from, r.to));
            }
        }
        (coalesce_ranges(&init_raw, limit), coalesce_ranges(&up_raw, limit))
    }

    pub(crate) fn mode(&self) -> SyncMode {
        self.mode
    }

    pub(crate) fn title_va(&self) -> u64 {
        // SAFETY: live D3D12 buffer.
        unsafe { self.title_buffer.GetGPUVirtualAddress() }
    }

    pub(crate) fn before_va(&self) -> u64 {
        // SAFETY: live D3D12 buffer.
        unsafe { self.before_buffer.GetGPUVirtualAddress() }
    }
}

/// Validates alignment and backing memory eligibility for `OpenExistingHeapFromAddress1` GPU sync.
pub(crate) fn check_eligibility(start: u64, len: u64) -> Result<(u64, u64, u32), FallbackReason> {
    if len == 0 || len > MAX_REGION_BYTES {
        return Err(FallbackReason::EmptyOrTooLarge);
    }
    if !start.is_multiple_of(4) {
        return Err(FallbackReason::UnalignedBase);
    }
    let aligned_start = start & !(K64 - 1);
    let end = start.checked_add(len).ok_or(FallbackReason::EmptyOrTooLarge)?;
    let aligned_end = end.next_multiple_of(K64);
    let heap_bytes = aligned_end.saturating_sub(aligned_start);
    if heap_bytes == 0 || heap_bytes > MAX_REGION_BYTES + K64 {
        return Err(FallbackReason::EmptyOrTooLarge);
    }
    if !shown_runs(aligned_start as usize, heap_bytes as usize).is_empty() {
        return Err(FallbackReason::ViewMemory);
    }
    if !private_whole(aligned_start as usize, heap_bytes as usize) {
        return Err(FallbackReason::NotCommittedReadWrite);
    }
    Ok((aligned_start, heap_bytes, (start - aligned_start) as u32))
}

/// Opens a `SyncedShadow` on `device` for title range `[start, start + len)` with L1 `before_buffer`
/// of `l1_size` bytes, or returns an explicit [`FallbackReason`] so the caller falls back cleanly.
pub(crate) fn try_create_region(
    device: &ID3D12Device,
    mode: SyncMode,
    start: u64,
    len: u64,
    l1_size: u64,
) -> Result<SyncedShadow, FallbackReason> {
    if mode == SyncMode::Unsupported {
        return Err(FallbackReason::DeviceUnsupported);
    }
    let (aligned_start, heap_bytes, title_bias) = check_eligibility(start, len)?;
    let heap = open(device, OpenFrom::Part(aligned_start as usize, heap_bytes as usize))
        .map_err(|_| FallbackReason::HeapOpenFailed)?;
    let allow_uav = mode == SyncMode::BidirectionalUav;
    let title_buffer = place_l0_buffer(device, &heap, heap_bytes, allow_uav)
        .map_err(|_| FallbackReason::BufferCreateFailed)?;
    let before_size = l1_size.max(len.next_multiple_of(16));
    let before_buffer = make_committed(
        device,
        D3D12_HEAP_TYPE_DEFAULT,
        before_size,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    )
    .map_err(|_| FallbackReason::BeforeAllocFailed)?;
    let page_count = (len as usize).div_ceil(K64 as usize).max(1);
    Ok(SyncedShadow {
        _title_heap: heap,
        title_buffer,
        title_bias,
        before_buffer,
        size: before_size,
        mode,
        pending_uploads: Mutex::new(Vec::new()),
        has_uploads: AtomicBool::new(false),
        initialized_pages: Mutex::new(vec![0u64; page_count.div_ceil(64)]),
        page_count,
    })
}

/// Maximum concurrent submissions timed by [`TimestampPool`] (4 timestamp queries per slot:
/// `[pre_begin, pre_end, post_begin, post_end]`).
const TS_SLOTS: u32 = 64;

/// Non-stalling paired `ID3D12QueryHeap` pool measuring on-card `pre_list` + `post_list` GPU
/// dispatch microseconds without adding extra fences or CPU waits.
struct TimestampPool {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    ptr: usize,
    freq: u64,
    free_slots: Mutex<Vec<u32>>,
}

// SAFETY: D3D12 query heap & readback buffer are free-threaded; slots are exclusively owned per flight.
unsafe impl Send for TimestampPool {}
// SAFETY: same as above.
unsafe impl Sync for TimestampPool {}

impl Drop for TimestampPool {
    fn drop(&mut self) {
        // SAFETY: persistently mapped readback buffer owned by this pool.
        unsafe {
            self.readback.Unmap(0, None);
        }
    }
}

impl TimestampPool {
    fn new(device: &ID3D12Device, queue: &ID3D12CommandQueue) -> Option<Self> {
        // SAFETY: plain D3D12 timestamp query heap & readback buffer creation.
        unsafe {
            let freq = queue.GetTimestampFrequency().unwrap_or(0);
            let mut heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: TS_SLOTS * 4,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
            let heap = heap?;
            let readback = make_committed(
                device,
                D3D12_HEAP_TYPE_READBACK,
                (TS_SLOTS as u64) * 4 * 8,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .ok()?;
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).ok()?;
            let free_slots = (0..TS_SLOTS).rev().collect();
            Some(Self {
                heap,
                readback,
                ptr: ptr as usize,
                freq,
                free_slots: Mutex::new(free_slots),
            })
        }
    }

    fn alloc_slot(&self) -> Option<u32> {
        if self.freq == 0 {
            return None;
        }
        self.free_slots.lock().unwrap_or_else(|e| e.into_inner()).pop()
    }

    fn release_slot(&self, slot: u32) {
        let mut free = self.free_slots.lock().unwrap_or_else(|e| e.into_inner());
        if free.len() < TS_SLOTS as usize {
            free.push(slot);
        }
    }

    unsafe fn begin_query(&self, list: &ID3D12GraphicsCommandList, slot: u32, is_post: bool) {
        let idx = slot * 4 + if is_post { 2 } else { 0 };
        // SAFETY: caller ensures `list` is open and `slot < TS_SLOTS`.
        unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, idx) };
    }

    unsafe fn end_and_resolve(&self, list: &ID3D12GraphicsCommandList, slot: u32, is_post: bool) {
        let base = slot * 4 + if is_post { 2 } else { 0 };
        // SAFETY: caller ensures `list` is open and `slot < TS_SLOTS`.
        unsafe {
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, base + 1);
            list.ResolveQueryData(
                &self.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                base,
                2,
                &self.readback,
                (base as u64) * 8,
            );
        }
    }

    /// Reads completed timestamps for `slot` (must only be called after the submission fence has
    /// signalled!), returns elapsed GPU microseconds, and recycles `slot`.
    fn collect_and_free(&self, slot: u32, has_pre: bool, has_post: bool) -> u64 {
        let us = if self.freq > 0 {
            let base = (slot as usize) * 4;
            let p = self.ptr as *const u64;
            let mut ticks = 0u64;
            // SAFETY: submission fence completed; readback memory is valid for `TS_SLOTS * 4` u64s.
            unsafe {
                if has_pre {
                    let t0 = p.add(base).read_volatile();
                    let t1 = p.add(base + 1).read_volatile();
                    ticks = ticks.saturating_add(t1.saturating_sub(t0));
                }
                if has_post {
                    let t2 = p.add(base + 2).read_volatile();
                    let t3 = p.add(base + 3).read_volatile();
                    ticks = ticks.saturating_add(t3.saturating_sub(t2));
                }
            }
            ticks_to_us(ticks, self.freq)
        } else {
            0
        };
        self.release_slot(slot);
        us
    }
}

fn ticks_to_us(ticks: u64, freq: u64) -> u64 {
    if freq == 0 {
        0
    } else {
        ((ticks as u128).saturating_mul(1_000_000) / (freq as u128)) as u64
    }
}

/// Maximum concurrent submissions tracked by [`DirtyPagePool`].
const DIRTY_SLOTS: u32 = 64;
/// Bytes per submission slot in [`DirtyPagePool`] (512 u32s = 16,384 x 64 KiB pages = 1 GiB).
const DIRTY_SLOT_BYTES: u64 = 2048;

/// Per-target dirty-page slice recorded inside a [`DirtyPagePool`] submission slot.
struct DirtyTarget {
    title_start: usize,
    region_len: usize,
    /// Byte offset inside the slot (`0..DIRTY_SLOT_BYTES`).
    slot_off: usize,
    /// Number of u32 bitmask words (`1` when `overflow_flag == true`).
    words: u32,
    /// If true, any non-zero u32 at `slot_off` triggers notifying `fallback_ranges`.
    overflow_flag: bool,
    fallback_ranges: Vec<(usize, usize)>,
}

/// Non-stalling 64-slot GPU->CPU bitmask pool recording which 64 KiB pages `BidirectionalUav`
/// `post_list` actually wrote over PCIe so `finish_submission` can notify `watch::landed_beside`
/// and `descriptor::host::remapped` with zero false positives on clean pages.
struct DirtyPagePool {
    gpu: ID3D12Resource,
    gpu_va: u64,
    readback: ID3D12Resource,
    ptr: usize,
    zero: ID3D12Resource,
    free_slots: Mutex<Vec<u32>>,
}

// SAFETY: D3D12 resources are free-threaded; slots are exclusively owned per flight.
unsafe impl Send for DirtyPagePool {}
// SAFETY: same as above.
unsafe impl Sync for DirtyPagePool {}

impl Drop for DirtyPagePool {
    fn drop(&mut self) {
        // SAFETY: persistently mapped readback buffer owned by this pool.
        unsafe {
            self.readback.Unmap(0, None);
        }
    }
}

impl DirtyPagePool {
    fn new(device: &ID3D12Device) -> Option<Self> {
        let total_bytes = (DIRTY_SLOTS as u64) * DIRTY_SLOT_BYTES;
        let gpu = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            total_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .ok()?;
        let readback = make_committed(
            device,
            D3D12_HEAP_TYPE_READBACK,
            total_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
        .ok()?;
        let zero = make_committed(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            DIRTY_SLOT_BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .ok()?;
        let mut rb_ptr = std::ptr::null_mut();
        let mut zero_ptr = std::ptr::null_mut();
        // SAFETY: newly allocated buffers owned by this pool.
        let gpu_va = unsafe {
            readback.Map(0, None, Some(&mut rb_ptr)).ok()?;
            zero.Map(0, None, Some(&mut zero_ptr)).ok()?;
            std::ptr::write_bytes(zero_ptr.cast::<u8>(), 0, DIRTY_SLOT_BYTES as usize);
            zero.Unmap(0, None);
            gpu.GetGPUVirtualAddress()
        };
        Some(Self {
            gpu,
            gpu_va,
            readback,
            ptr: rb_ptr as usize,
            zero,
            free_slots: Mutex::new((0..DIRTY_SLOTS).rev().collect()),
        })
    }

    fn alloc_slot(&self) -> Option<u32> {
        self.free_slots.lock().unwrap_or_else(|e| e.into_inner()).pop()
    }

    fn release_slot(&self, slot: u32) {
        let mut free = self.free_slots.lock().unwrap_or_else(|e| e.into_inner());
        if free.len() < DIRTY_SLOTS as usize {
            free.push(slot);
        }
    }

    /// Reads completed dirty-page bitmasks for `slot` (must only be called after the submission
    /// fence has signalled!), notifies `watch::landed_beside` + `descriptor::host::remapped` for
    /// every 64 KiB page written by GPU, and recycles `slot`. Returns the number of notified spans.
    fn notify_and_free(&self, slot: u32, targets: &[DirtyTarget]) -> usize {
        const PAGE4K: usize = 4096;
        const PAGE64K: usize = K64 as usize;
        let slot_base = (slot as usize) * (DIRTY_SLOT_BYTES as usize);
        let mut pages_4k: Vec<usize> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for target in targets {
            let word_ptr = (self.ptr + slot_base + target.slot_off) as *const u32;
            if target.overflow_flag {
                // SAFETY: fence completed; `slot_off + 4 <= DIRTY_SLOT_BYTES`.
                let flag = unsafe { word_ptr.read_volatile() };
                if flag != 0 {
                    for &(rel_lo, rel_hi) in &target.fallback_ranges {
                        let lo = target.title_start + rel_lo;
                        let hi = target.title_start + rel_hi;
                        if lo < hi {
                            spans.push((lo, hi - lo));
                            for p4 in lo / PAGE4K..hi.div_ceil(PAGE4K) {
                                pages_4k.push(p4);
                            }
                        }
                    }
                }
                continue;
            }
            for wi in 0..target.words as usize {
                // SAFETY: fence completed; `slot_off + target.words * 4 <= DIRTY_SLOT_BYTES`.
                let mut bits = unsafe { word_ptr.add(wi).read_volatile() };
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let page64 = wi * 32 + bit;
                    let rel_lo = page64 * PAGE64K;
                    if rel_lo >= target.region_len {
                        continue;
                    }
                    let rel_hi = (rel_lo + PAGE64K).min(target.region_len);
                    let lo = target.title_start + rel_lo;
                    let hi = target.title_start + rel_hi;
                    spans.push((lo, hi - lo));
                    for p4 in lo / PAGE4K..hi.div_ceil(PAGE4K) {
                        pages_4k.push(p4);
                    }
                }
            }
        }
        self.release_slot(slot);
        let notified = spans.len();
        if !spans.is_empty() {
            pages_4k.sort_unstable();
            pages_4k.dedup();
            // `watch::landed_beside` also calls `descriptor::host::remapped` for every span.
            crate::watch::landed_beside(&pages_4k, &spans);
        }
        notified
    }
}

/// Process-wide GPU sync runtime state initialized lazily on first use when `VICTORY_GPU_SYNC=1`.
struct HostSyncState {
    pipelines: EnginePipelines,
    /// Pool of command list + allocator pairs for pre-submission upload and post-submission sync.
    free_lists: Mutex<Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
    /// Cached execution rig for `SrvDirectUpload` delta drain passes (created lazily only if needed).
    srv_rig: Mutex<Option<ProbeRig>>,
    /// Non-stalling paired timestamp pool for on-card `gpu_dispatch_us` telemetry.
    timestamps: Option<TimestampPool>,
    /// Non-stalling per-64 KiB-page dirty bitmask pool for `BidirectionalUav` write-back notifications.
    dirty_pages: Option<DirtyPagePool>,
}

// SAFETY: D3D12 command objects are free-threaded; pool and rig access are serialized by Mutex.
unsafe impl Send for HostSyncState {}
// SAFETY: same as above.
unsafe impl Sync for HostSyncState {}

static HOST_SYNC: OnceLock<Option<HostSyncState>> = OnceLock::new();

fn maybe_log_stats(state: &HostSyncState) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static LAST_LOG_MS: AtomicU64 = AtomicU64::new(0);
    const INTERVAL_MS: u64 = 5_000;
    let now_ms = START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64;
    let prev = LAST_LOG_MS.load(Ordering::Relaxed);
    if now_ms.saturating_sub(prev) < INTERVAL_MS {
        return;
    }
    if LAST_LOG_MS.compare_exchange(prev, now_ms, Ordering::AcqRel, Ordering::Relaxed).is_err() {
        return;
    }
    let attached = STATS[0].load(Ordering::Relaxed);
    let fallbacks = STATS[1].load(Ordering::Relaxed);
    let upload_dispatches = STATS[2].swap(0, Ordering::Relaxed);
    let upload_bytes = STATS[3].swap(0, Ordering::Relaxed);
    let sync_back_dispatches = STATS[4].swap(0, Ordering::Relaxed);
    let sync_back_bytes = STATS[5].swap(0, Ordering::Relaxed);
    let srv_passes = STATS[6].swap(0, Ordering::Relaxed);
    let detached = STATS[7].load(Ordering::Relaxed);
    let eligible_bytes = ELIGIBLE_BYTES.load(Ordering::Relaxed);
    let gpu_dispatch_us = GPU_DISPATCH_US.swap(0, Ordering::Relaxed);
    let fence_wait_us = FENCE_WAIT_US.swap(0, Ordering::Relaxed);
    tracing::info!(
        mode = ?state.pipelines.mode,
        atomic_flags = state.pipelines.atomic_flags,
        eligible_bytes,
        eligible_mb = format_args!("{:.1}", eligible_bytes as f64 / (1024.0 * 1024.0)),
        attached,
        fallbacks,
        detached,
        upload_dispatches,
        upload_bytes,
        sync_back_dispatches,
        sync_back_bytes,
        srv_passes,
        gpu_dispatch_us,
        fence_wait_us,
        "VICTORY_GPU_SYNC telemetry"
    );
}

fn host_state(host: &Host) -> Option<&'static HostSyncState> {
    if !on() {
        return None;
    }
    HOST_SYNC
        .get_or_init(|| {
            let report = probe_device(&host.device);
            let mode_override = configured_mode();
            let chosen_mode = select_mode(&report, mode_override);
            let atomic_flags = compute_atomic_flags(&report, force_atomics());
            tracing::info!(
                adapter = %report.adapter_name,
                software = report.is_software,
                cache_coherent_uma = report.cache_coherent_uma,
                device13 = report.has_device13,
                probed_mode = ?report.mode,
                mode_override = ?mode_override,
                mode = ?chosen_mode,
                atomic_flags,
                l0_srv = report.l0_srv_read_ok,
                l0_uav = report.l0_uav_write_ok,
                l0_copy_to_l1 = report.l0_copy_to_l1_ok,
                l0_copy_from_l1 = report.l0_copy_from_l1_ok,
                l0_atomics = report.l0_atomics_ok,
                heap_flags = format_args!("{:#x}", report.l0_heap_flags),
                debug_errors = report.debug_errors.len(),
                "VICTORY_GPU_SYNC=1 capability probe"
            );
            if chosen_mode == SyncMode::Unsupported {
                tracing::warn!(notes = ?report.notes, ?mode_override, "VICTORY_GPU_SYNC=1 unsupported on this adapter; falling back to CPU path");
                return None;
            }
            match EnginePipelines::new(&host.device, chosen_mode, atomic_flags, report.l0_copy_to_l1_ok) {
                Ok(pipelines) => Some(HostSyncState {
                    pipelines,
                    free_lists: Mutex::new(Vec::new()),
                    srv_rig: Mutex::new(None),
                    timestamps: TimestampPool::new(&host.device, host.queue()),
                    dirty_pages: DirtyPagePool::new(&host.device),
                }),
                Err(e) => {
                    tracing::warn!(error = %e, "VICTORY_GPU_SYNC=1 pipeline creation failed; falling back to CPU path");
                    None
                }
            }
        })
        .as_ref()
}

/// Attempts to attach a GPU-direct sync engine to a newly created L1 `Shadow` buffer.
pub(crate) fn try_attach(host: &Host, start: u64, len: u64, l1_size: u64) -> Option<SyncedShadow> {
    let state = host_state(host)?;
    match try_create_region(&host.device, state.pipelines.mode, start, len, l1_size) {
        Ok(synced) => {
            STATS[0].fetch_add(1, Ordering::Relaxed);
            Some(synced)
        }
        Err(reason) => {
            let count = STATS[1].fetch_add(1, Ordering::Relaxed) + 1;
            if count <= 8 || count.is_power_of_two() {
                tracing::debug!(
                    start = format_args!("{start:#x}"),
                    len,
                    ?reason,
                    "VICTORY_GPU_SYNC fallback to CPU path for buffer"
                );
            }
            None
        }
    }
}

pub(crate) fn note_detached(count: u64) {
    STATS[7].fetch_add(count, Ordering::Relaxed);
}

fn take_cmd_list(host: &Host, state: &HostSyncState) -> Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
    if let Some(pair) = state.free_lists.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(pair);
    }
    // SAFETY: plain D3D12 command allocator & direct list creation.
    unsafe {
        let alloc: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
        let list: ID3D12GraphicsCommandList =
            host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).ok()?;
        list.Close().ok()?;
        Some((alloc, list))
    }
}

fn return_cmd_list(state: &HostSyncState, pair: (ID3D12CommandAllocator, ID3D12GraphicsCommandList)) {
    let mut free = state.free_lists.lock().unwrap_or_else(|e| e.into_inner());
    if free.len() < 64 {
        free.push(pair);
    }
}

/// Subtracts pre-coalesced `remove` intervals from pre-coalesced `ranges`.
pub(crate) fn subtract_ranges(ranges: &[(usize, usize)], remove: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if ranges.is_empty() || remove.is_empty() {
        return ranges.to_vec();
    }
    let mut out = Vec::with_capacity(ranges.len());
    let mut r_idx = 0usize;
    for &(mut lo, hi) in ranges {
        while r_idx < remove.len() && remove[r_idx].1 <= lo {
            r_idx += 1;
        }
        let mut j = r_idx;
        while j < remove.len() && remove[j].0 < hi {
            let (r_lo, r_hi) = remove[j];
            if r_lo > lo {
                out.push((lo, r_lo));
            }
            lo = lo.max(r_hi);
            if lo >= hi {
                break;
            }
            j += 1;
        }
        if lo < hi {
            out.push((lo, hi));
        }
    }
    out
}

/// Placed in-list copy snapshot captured at submission preparation (`host.rs`).
///
/// `pre_bytes` (populated only for `gpu_synced` shadows before `Placed::prepare(true)` mutates
/// live title memory) preserves the pre-`Placed` bytes for `shadow.buffer` so draws earlier in
/// the same command list never observe the post-`Placed` value written into live title RAM.
pub(crate) struct PlacedCopy {
    pub(crate) shadow: Arc<crate::shadow::Shadow>,
    pub(crate) at: usize,
    pub(crate) bytes: Vec<u8>,
    pub(crate) pre_bytes: Option<Vec<u8>>,
}

struct StagedPlaced {
    pre_copies: Vec<(usize, u64, usize)>,
    after_copies: Vec<(usize, u64, usize)>,
}

/// Clips `(rel_lo, up_off, copy_len)` upload copies to only the sub-intervals covered by `mask`
/// (sorted, disjoint `(from, to)` ranges).
fn clip_copies_to_ranges(copies: &[(usize, u64, usize)], mask: &[(usize, usize)]) -> Vec<(usize, u64, usize)> {
    let mut out = Vec::new();
    for &(rel_lo, up_off, copy_len) in copies {
        let rel_hi = rel_lo.saturating_add(copy_len);
        for &(m_lo, m_hi) in mask {
            if m_hi <= rel_lo {
                continue;
            }
            if m_lo >= rel_hi {
                break;
            }
            let lo = rel_lo.max(m_lo);
            let hi = rel_hi.min(m_hi);
            if lo < hi {
                out.push((lo, up_off + (lo - rel_lo) as u64, hi - lo));
            }
        }
    }
    out
}

/// Allocates one transient `UPLOAD` buffer containing both `pre_bytes` (for `shadow.buffer` ahead
/// of in-list draws) and `bytes` (for `before_buffer` post-list baseline) for GPU-synced shadows.
fn stage_placed_snapshots(
    device: &ID3D12Device,
    placed: &[PlacedCopy],
) -> Option<(ID3D12Resource, Vec<(Arc<crate::shadow::Shadow>, StagedPlaced)>)> {
    struct Entry {
        idx: usize,
        rel_lo: usize,
        after_up_off: u64,
        pre_up_off: Option<u64>,
        copy_len: usize,
    }
    let mut per_shadow: Vec<(Arc<crate::shadow::Shadow>, Vec<Entry>)> = Vec::new();
    let mut total_bytes = 0u64;
    for (idx, item) in placed.iter().enumerate() {
        if !item.shadow.gpu_synced() || item.bytes.is_empty() {
            continue;
        }
        let (start, len) = item.shadow.span();
        let lo = item.at.saturating_sub(start as usize).min(len as usize);
        let hi = item.at.saturating_add(item.bytes.len()).saturating_sub(start as usize).min(len as usize);
        if lo >= hi {
            continue;
        }
        let src_off = (start as usize + lo).saturating_sub(item.at);
        let copy_len = hi - lo;
        if src_off + copy_len > item.bytes.len() {
            continue;
        }
        let after_up_off = total_bytes;
        total_bytes += copy_len as u64;
        let pre_up_off = if item.pre_bytes.as_ref().is_some_and(|p| src_off + copy_len <= p.len()) {
            let off = total_bytes;
            total_bytes += copy_len as u64;
            Some(off)
        } else {
            None
        };
        let entry = Entry { idx, rel_lo: lo, after_up_off, pre_up_off, copy_len };
        match per_shadow.iter_mut().find(|(s, _)| Arc::ptr_eq(s, &item.shadow)) {
            Some((_, entries)) => entries.push(entry),
            None => per_shadow.push((item.shadow.clone(), vec![entry])),
        }
    }
    if total_bytes == 0 {
        return None;
    }
    let upload = make_committed(
        device,
        D3D12_HEAP_TYPE_UPLOAD,
        total_bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )
    .ok()?;
    let mut ptr = std::ptr::null_mut();
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: newly created upload buffer of `total_bytes`; write-only mapping.
    unsafe {
        upload.Map(0, Some(&no_read), Some(&mut ptr)).ok()?;
        let dst = ptr.cast::<u8>();
        for (shadow, entries) in &per_shadow {
            let start = shadow.span().0 as usize;
            for e in entries {
                let item = &placed[e.idx];
                let src_off = (start + e.rel_lo).saturating_sub(item.at);
                std::ptr::copy_nonoverlapping(
                    item.bytes.as_ptr().add(src_off),
                    dst.add(e.after_up_off as usize),
                    e.copy_len,
                );
                if let (Some(pre_off), Some(pre)) = (e.pre_up_off, item.pre_bytes.as_ref()) {
                    std::ptr::copy_nonoverlapping(
                        pre.as_ptr().add(src_off),
                        dst.add(pre_off as usize),
                        e.copy_len,
                    );
                }
            }
        }
        upload.Unmap(0, None);
    }
    let mapped = per_shadow
        .into_iter()
        .map(|(s, entries)| {
            let mut pre_copies = Vec::new();
            let mut after_copies = Vec::new();
            for e in entries {
                after_copies.push((e.rel_lo, e.after_up_off, e.copy_len));
                if let Some(pre_off) = e.pre_up_off {
                    pre_copies.push((e.rel_lo, pre_off, e.copy_len));
                }
            }
            (s, StagedPlaced { pre_copies, after_copies })
        })
        .collect();
    Some((upload, mapped))
}

/// Plan returned by [`prepare_submission`] holding pre/post sync command lists and
/// metadata needed when the submission completes ([`finish_submission`]).
pub(crate) struct SubmissionPlan {
    /// Pre-submission upload / auto-Init / placed-snapshot list (spliced at the front of `host.rs`'s
    /// submission `lists`; transitions every modified buffer back to `COMMON` before closing).
    pub(crate) pre_list: Option<ID3D12CommandList>,
    /// Post-submission L1-first GPU write-back list (run in the trailing `compare_at` batch).
    pub(crate) post_list: Option<ID3D12CommandList>,
    /// Command lists owned by this plan to recycle in [`finish_submission`].
    lists: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// Copies kept alive until submission finishes.
    _kept: Vec<Arc<crate::shadow::Shadow>>,
    /// Transient upload buffers (e.g. `placed` snapshots) kept alive until submission finishes.
    _transient: Vec<ID3D12Resource>,
    /// Pages to mark initialized once `host.rs` actually submits `pre_list`.
    pending_init: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>,
    /// Declared ranges requiring SRV delta application after fence completion (empty in `BidirectionalUav` mode).
    srv_ranges: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>,
    /// Optional paired timestamp slot `(slot, has_pre, has_post)` resolved in [`finish_submission`].
    ts_slot: Option<(u32, bool, bool)>,
    /// Optional dirty-page bitmask slot `(slot, targets)` notified in [`finish_submission`].
    dirty_slot: Option<(u32, Vec<DirtyTarget>)>,
    /// Whether `host.rs` actually submitted `lists` containing this plan.
    submitted: bool,
    /// Whether this plan incremented `SRV_IN_FLIGHT`.
    counted_srv_flight: bool,
}

impl SubmissionPlan {
    /// Marks this plan as submitted to `host.queue` and records its `Init` pages as initialized.
    pub(crate) fn note_submitted(&mut self) {
        self.submitted = true;
        for (shadow, ranges) in std::mem::take(&mut self.pending_init) {
            if let Some(gs) = shadow.gpu_sync_ref() {
                gs.mark_pages_initialized(&ranges, shadow.span().1 as usize);
            }
        }
    }
}

impl Drop for SubmissionPlan {
    fn drop(&mut self) {
        if self.counted_srv_flight {
            SRV_IN_FLIGHT.fetch_sub(1, Ordering::Release);
            self.counted_srv_flight = false;
        }
        // Only recycle unsubmitted pool slots directly; submitted slots are recycled in
        // `finish_submission` after the submission's fence completes.
        if !self.submitted
            && let Some(Some(state)) = HOST_SYNC.get()
        {
            if let Some((slot, _, _)) = self.ts_slot.take()
                && let Some(ts) = state.timestamps.as_ref()
            {
                ts.release_slot(slot);
            }
            if let Some((slot, _)) = self.dirty_slot.take()
                && let Some(dp) = state.dirty_pages.as_ref()
            {
                dp.release_slot(slot);
            }
        }
    }
}

// SAFETY: D3D12 lists/allocators are free-threaded; plan has single ownership per flight.
unsafe impl Send for SubmissionPlan {}

struct PreShadowWork {
    shadow: Arc<crate::shadow::Shadow>,
    init_pages: Vec<(usize, usize)>,
    init_ranges: Vec<(usize, usize)>,
    up_ranges: Vec<(usize, usize)>,
    placed_pre_copies: Vec<(usize, u64, usize)>,
    placed_after_copies: Vec<(usize, u64, usize)>,
}

/// Records one shadow's pre-list upload/Init/placed-snapshot commands and transitions every
/// touched buffer explicitly back to `D3D12_RESOURCE_STATE_COMMON` so `pre_list` can be spliced
/// directly into `host.rs`'s `lists` without an extra `ExecuteCommandLists` call.
///
/// `placed_pre_copies` are written into `shadow.buffer` in reverse order (so the earliest in-list
/// `Placed` copy's pre-bytes stand ahead of prior draws), while `placed_after_copies` are written
/// into `before_buffer` in forward order (so `post_list` diffs against the final `Placed` bytes).
///
/// # Safety
/// `list` must be open; all referenced buffers must start in `COMMON`.
#[allow(clippy::too_many_arguments)]
unsafe fn record_pre_shadow_to_common(
    list: &ID3D12GraphicsCommandList,
    pipelines: &EnginePipelines,
    shadow: &crate::shadow::Shadow,
    gs: &SyncedShadow,
    init_ranges: &[(usize, usize)],
    up_ranges: &[(usize, usize)],
    placed_pre_copies: &[(usize, u64, usize)],
    placed_after_copies: &[(usize, u64, usize)],
    placed_upload: Option<&ID3D12Resource>,
) -> u64 {
    let mut total_bytes = 0u64;
    let title_va = gs.title_va();
    let shadow_va = unsafe { shadow.buffer.GetGPUVirtualAddress() };
    let before_va = gs.before_va();
    let mut did_dma_init = false;
    let mut title_state = D3D12_RESOURCE_STATE_COMMON;
    let mut shadow_state = D3D12_RESOURCE_STATE_COMMON;
    let mut before_state = D3D12_RESOURCE_STATE_COMMON;

    // SAFETY: caller ensures `list` is open and buffers are live in `COMMON`.
    unsafe {
        if !init_ranges.is_empty() {
            total_bytes += init_ranges.iter().map(|&(a, b)| (b - a) as u64).sum::<u64>();
            if pipelines.l0_copy_to_l1 {
                let then_compute = !up_ranges.is_empty();
                record_init_dma(
                    list,
                    &gs.title_buffer,
                    gs.title_bias,
                    &shadow.buffer,
                    &gs.before_buffer,
                    init_ranges,
                    then_compute,
                );
                did_dma_init = true;
                if then_compute {
                    title_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    shadow_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    before_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                } else {
                    title_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    shadow_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    before_state = D3D12_RESOURCE_STATE_COPY_DEST;
                }
            } else if gs.mode() != SyncMode::Unsupported {
                record_srv_ranges(
                    list,
                    pipelines,
                    title_va,
                    gs.title_bias,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    init_ranges,
                    SyncOp::Init,
                    SRV_MAX_DELTAS,
                );
                title_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                shadow_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                before_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
        }
        if !up_ranges.is_empty() {
            if !init_ranges.is_empty() && !did_dma_init {
                list.ResourceBarrier(&[uav_barrier()]);
            }
            total_bytes += up_ranges.iter().map(|&(a, b)| (b - a) as u64).sum::<u64>();
            if gs.mode() != SyncMode::Unsupported {
                record_srv_ranges(
                    list,
                    pipelines,
                    title_va,
                    gs.title_bias,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    up_ranges,
                    SyncOp::Upload3Way,
                    SRV_MAX_DELTAS,
                );
                title_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                shadow_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                before_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
        }
        if !placed_pre_copies.is_empty()
            && let Some(upload) = placed_upload
        {
            if shadow_state != D3D12_RESOURCE_STATE_COMMON && shadow_state != D3D12_RESOURCE_STATE_COPY_DEST {
                list.ResourceBarrier(&[transition(
                    &shadow.buffer,
                    shadow_state,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            // Apply in reverse order so the earliest in-list `Placed` copy's pre-bytes prevail on
            // overlapping ranges ahead of earlier draws; serialize any overlapping copies.
            let mut written_spans: Vec<(usize, usize)> = Vec::new();
            for &(rel_lo, up_off, copy_len) in placed_pre_copies.iter().rev() {
                let rel_hi = rel_lo.saturating_add(copy_len);
                if written_spans.iter().any(|&(a, b)| rel_lo < b && a < rel_hi) {
                    list.ResourceBarrier(&[transition(
                        &shadow.buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    list.ResourceBarrier(&[transition(
                        &shadow.buffer,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    written_spans.clear();
                }
                list.CopyBufferRegion(
                    &shadow.buffer,
                    rel_lo as u64,
                    upload,
                    up_off,
                    copy_len as u64,
                );
                written_spans.push((rel_lo, rel_hi));
            }
            shadow_state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        if !placed_after_copies.is_empty()
            && let Some(upload) = placed_upload
        {
            // Flush any prior writes to `before_buffer` and put it in `COPY_DEST`.
            if before_state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                list.ResourceBarrier(&[transition(
                    &gs.before_buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            } else if before_state == D3D12_RESOURCE_STATE_COPY_DEST {
                list.ResourceBarrier(&[transition(
                    &gs.before_buffer,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                list.ResourceBarrier(&[transition(
                    &gs.before_buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            let mut written_spans: Vec<(usize, usize)> = Vec::new();
            for &(rel_lo, up_off, copy_len) in placed_after_copies {
                let rel_hi = rel_lo.saturating_add(copy_len);
                if written_spans.iter().any(|&(a, b)| rel_lo < b && a < rel_hi) {
                    list.ResourceBarrier(&[transition(
                        &gs.before_buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    list.ResourceBarrier(&[transition(
                        &gs.before_buffer,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    written_spans.clear();
                }
                list.CopyBufferRegion(
                    &gs.before_buffer,
                    rel_lo as u64,
                    upload,
                    up_off,
                    copy_len as u64,
                );
                written_spans.push((rel_lo, rel_hi));
            }
            before_state = D3D12_RESOURCE_STATE_COPY_DEST;
        }
        // Transition every touched buffer explicitly back to `COMMON` so subsequent title lists
        // in the same `ExecuteCommandLists` call can implicitly promote them to any state.
        let mut end_barriers = Vec::with_capacity(3);
        if title_state != D3D12_RESOURCE_STATE_COMMON {
            end_barriers.push(transition(&gs.title_buffer, title_state, D3D12_RESOURCE_STATE_COMMON));
        }
        if shadow_state != D3D12_RESOURCE_STATE_COMMON {
            end_barriers.push(transition(&shadow.buffer, shadow_state, D3D12_RESOURCE_STATE_COMMON));
        }
        if before_state != D3D12_RESOURCE_STATE_COMMON {
            end_barriers.push(transition(&gs.before_buffer, before_state, D3D12_RESOURCE_STATE_COMMON));
        }
        if !end_barriers.is_empty() {
            list.ResourceBarrier(&end_barriers);
        }
    }
    total_bytes
}

/// Prepares zero-CPU-copy GPU sync for a submission:
/// 1. Records `pre_list` (spliced at the front of `host.rs`'s `lists` without an extra
///    `ExecuteCommandLists`) uploading dirty/init ranges (`L0 -> L1`), auto-initializing any
///    uninitialized 64 KiB pages covered by `uploads`/`declared`/`placed`, and applying pre/post
///    `placed` byte snapshots without clobbering draws ahead of in-list copies.
/// 2. Records `post_list` running L1-first `SyncOp::GpuWriteBackOnly` (zero PCIe reads/writes on
///    clean GPU dwords, coalesced non-atomic posted `Store4`/`Store` for full dwords and CAS for
///    partial dwords on discrete PCIe GPUs, plus per-64 KiB-page dirty-bitmask tracking).
pub(crate) fn prepare_submission(
    host: &Host,
    declared: &[(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)],
    placed: &[PlacedCopy],
) -> Option<SubmissionPlan> {
    let state = host_state(host)?;
    let uploads = crate::shadow::gpu_sync_dirty();
    let placed_staged = stage_placed_snapshots(&host.device, placed);

    let mut sync_targets: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
    for (shadow, ranges) in declared {
        if shadow.gpu_synced() {
            let (start, len) = shadow.span();
            let rel: Vec<(usize, usize)> = ranges
                .iter()
                .filter_map(|&(a, b)| {
                    let lo = a.saturating_sub(start as usize).min(len as usize);
                    let hi = b.saturating_sub(start as usize).min(len as usize);
                    (lo < hi).then_some((lo, hi))
                })
                .collect();
            let coalesced = coalesce_ranges(&rel, len as usize);
            if !coalesced.is_empty() {
                sync_targets.push((shadow.clone(), coalesced));
            }
        }
    }

    // Build unified per-shadow pre-list plan covering `uploads`, uninitialized pages in
    // `sync_targets`, and `placed` snapshots.
    let mut pre_work: Vec<PreShadowWork> = Vec::new();
    let find_or_add = |vec: &mut Vec<PreShadowWork>, s: &Arc<crate::shadow::Shadow>| -> usize {
        if let Some(i) = vec.iter().position(|w| Arc::ptr_eq(&w.shadow, s)) {
            i
        } else {
            vec.push(PreShadowWork {
                shadow: s.clone(),
                init_pages: Vec::new(),
                init_ranges: Vec::new(),
                up_ranges: Vec::new(),
                placed_pre_copies: Vec::new(),
                placed_after_copies: Vec::new(),
            });
            vec.len() - 1
        }
    };
    for (shadow, init_r, up_r) in uploads {
        let idx = find_or_add(&mut pre_work, &shadow);
        pre_work[idx].init_ranges.extend(init_r);
        pre_work[idx].up_ranges.extend(up_r);
    }
    let mut transient = Vec::new();
    let placed_upload: Option<ID3D12Resource> = if let Some((upload, per_shadow)) = placed_staged {
        for (shadow, staged) in per_shadow {
            let idx = find_or_add(&mut pre_work, &shadow);
            pre_work[idx].placed_pre_copies.extend(staged.pre_copies);
            pre_work[idx].placed_after_copies.extend(staged.after_copies);
        }
        transient.push(upload.clone());
        Some(upload)
    } else {
        None
    };
    // Ensure any uninitialized 64 KiB pages touched by `sync_targets` (e.g. `Shadow::writing`
    // without a prior `refresh`) are initialized in the pre-list BEFORE title lists execute!
    for (shadow, ranges) in &sync_targets {
        let Some(gs) = shadow.gpu_sync_ref() else {
            continue;
        };
        let len = shadow.span().1 as usize;
        let missing = gs.uninitialized_page_ranges(ranges, len);
        if !missing.is_empty() {
            let idx = find_or_add(&mut pre_work, shadow);
            pre_work[idx].init_ranges.extend(missing);
        }
    }
    // Expand uninitialized pages, subtract `placed` spans from `init_ranges`/`up_ranges` so live
    // post-`Placed` title RAM never overwrites `shadow.buffer` ahead of earlier draws, and clip
    // `placed_pre_copies` to the union of `init_pages` and `orig_up`.
    for work in &mut pre_work {
        let Some(gs) = work.shadow.gpu_sync_ref() else {
            continue;
        };
        let len = work.shadow.span().1 as usize;
        let placed_spans: Vec<(usize, usize)> = coalesce_ranges(
            &work
                .placed_after_copies
                .iter()
                .map(|&(rel_lo, _, copy_len)| (rel_lo, rel_lo + copy_len))
                .collect::<Vec<_>>(),
            len,
        );
        let orig_up = coalesce_ranges(&work.up_ranges, len);
        let mut all_touched = Vec::new();
        all_touched.extend_from_slice(&work.init_ranges);
        all_touched.extend_from_slice(&orig_up);
        all_touched.extend_from_slice(&placed_spans);
        let missing = gs.uninitialized_page_ranges(&all_touched, len);
        work.init_ranges.extend(missing);
        let full_init = coalesce_ranges(&work.init_ranges, len);
        work.init_pages = full_init.clone();
        work.init_ranges = subtract_ranges(&full_init, &placed_spans);
        let up_minus_init = subtract_ranges(&orig_up, &full_init);
        work.up_ranges = subtract_ranges(&up_minus_init, &placed_spans);
        if !work.placed_pre_copies.is_empty() {
            let mut shadow_dirty = Vec::with_capacity(full_init.len() + orig_up.len());
            shadow_dirty.extend_from_slice(&full_init);
            shadow_dirty.extend_from_slice(&orig_up);
            let dirty_mask = coalesce_ranges(&shadow_dirty, len);
            work.placed_pre_copies = clip_copies_to_ranges(&work.placed_pre_copies, &dirty_mask);
        }
    }
    pre_work.retain(|w| {
        !w.init_ranges.is_empty()
            || !w.up_ranges.is_empty()
            || !w.placed_pre_copies.is_empty()
            || !w.placed_after_copies.is_empty()
    });

    if pre_work.is_empty() && sync_targets.is_empty() {
        maybe_log_stats(state);
        return None;
    }

    let mut lists = Vec::new();
    let mut kept: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
    let mut pending_init: Vec<(Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
    let ts_alloc = state.timestamps.as_ref().and_then(TimestampPool::alloc_slot);
    let mut ts_has_pre = false;
    let mut ts_has_post = false;
    let mut dirty_slot: Option<(u32, Vec<DirtyTarget>)> = None;

    // 1. Pre-submission upload + auto-Init + placed-snapshot list (spliced into `host.rs`'s `lists`).
    let mut pre_list = None;
    if !pre_work.is_empty()
        && let Some((alloc, list)) = take_cmd_list(host, state)
    {
        // SAFETY: own allocator/list; buffers kept alive in `kept` and `transient`.
        let recorded = unsafe {
            (|| -> windows::core::Result<ID3D12CommandList> {
                alloc.Reset()?;
                list.Reset(&alloc, None)?;
                if let (Some(ts), Some(slot)) = (state.timestamps.as_ref(), ts_alloc) {
                    ts.begin_query(&list, slot, false);
                }
                let mut total_bytes = 0u64;
                for work in &pre_work {
                    let Some(gs) = work.shadow.gpu_sync_ref() else {
                        continue;
                    };
                    total_bytes += record_pre_shadow_to_common(
                        &list,
                        &state.pipelines,
                        &work.shadow,
                        gs,
                        &work.init_ranges,
                        &work.up_ranges,
                        &work.placed_pre_copies,
                        &work.placed_after_copies,
                        placed_upload.as_ref(),
                    );
                    if !work.init_pages.is_empty() {
                        pending_init.push((work.shadow.clone(), work.init_pages.clone()));
                    }
                    kept.push(work.shadow.clone());
                }
                if let (Some(ts), Some(slot)) = (state.timestamps.as_ref(), ts_alloc) {
                    ts.end_and_resolve(&list, slot, false);
                }
                list.Close()?;
                STATS[2].fetch_add(1, Ordering::Relaxed);
                STATS[3].fetch_add(total_bytes, Ordering::Relaxed);
                list.cast()
            })()
        };
        match recorded {
            Ok(cmd) => {
                pre_list = Some(cmd);
                ts_has_pre = ts_alloc.is_some();
            }
            Err(e) => {
                pending_init.clear();
                tracing::warn!(error = %e, "VICTORY_GPU_SYNC pre-upload list failed");
            }
        }
        lists.push((alloc, list));
    }

    // 2. Post-submission L1-first write-back list (`SyncOp::GpuWriteBackOnly`, runs in `compare_at`).
    let mut post_list = None;
    let mut srv_ranges = Vec::new();
    if !sync_targets.is_empty() {
        match state.pipelines.mode {
            SyncMode::BidirectionalUav => {
                let dp_alloc = state.dirty_pages.as_ref().and_then(DirtyPagePool::alloc_slot);
                let mut dp_targets: Vec<DirtyTarget> = Vec::new();
                let mut dp_used_bytes: usize = 0;
                if dp_alloc.is_some() {
                    const OVERFLOW_OFF: usize = (DIRTY_SLOT_BYTES as usize) - 4;
                    for (shadow, ranges) in &sync_targets {
                        let (start, len) = shadow.span();
                        let pages = (len as usize).div_ceil(K64 as usize).max(1);
                        let words = pages.div_ceil(32) as u32;
                        let need = (words as usize) * 4;
                        if dp_used_bytes + need <= OVERFLOW_OFF {
                            dp_targets.push(DirtyTarget {
                                title_start: start as usize,
                                region_len: len as usize,
                                slot_off: dp_used_bytes,
                                words,
                                overflow_flag: false,
                                fallback_ranges: Vec::new(),
                            });
                            dp_used_bytes += need;
                        } else {
                            dp_targets.push(DirtyTarget {
                                title_start: start as usize,
                                region_len: len as usize,
                                slot_off: OVERFLOW_OFF,
                                words: 1,
                                overflow_flag: true,
                                fallback_ranges: ranges.clone(),
                            });
                            dp_used_bytes = DIRTY_SLOT_BYTES as usize;
                        }
                    }
                }
                if let Some((alloc, list)) = take_cmd_list(host, state) {
                    // SAFETY: own allocator/list; buffers kept alive in `kept`.
                    let recorded = unsafe {
                        (|| -> windows::core::Result<ID3D12CommandList> {
                            alloc.Reset()?;
                            list.Reset(&alloc, None)?;
                            if let (Some(ts), Some(slot)) = (state.timestamps.as_ref(), ts_alloc) {
                                ts.begin_query(&list, slot, true);
                            }
                            let dp_ref = state.dirty_pages.as_ref();
                            let dp_slot_base = dp_alloc.map(|s| (s as u64) * DIRTY_SLOT_BYTES).unwrap_or(0);
                            if let (Some(dp), Some(_)) = (dp_ref, dp_alloc)
                                && dp_used_bytes > 0
                            {
                                list.CopyBufferRegion(&dp.gpu, dp_slot_base, &dp.zero, 0, dp_used_bytes as u64);
                                list.ResourceBarrier(&[
                                    transition(
                                        &dp.gpu,
                                        D3D12_RESOURCE_STATE_COPY_DEST,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    ),
                                    uav_barrier(),
                                ]);
                            } else {
                                list.ResourceBarrier(&[uav_barrier()]);
                            }
                            let mut total_bytes = 0u64;
                            for (i, (shadow, ranges)) in sync_targets.iter().enumerate() {
                                let Some(gs) = shadow.gpu_sync_ref() else {
                                    continue;
                                };
                                total_bytes += ranges.iter().map(|&(a, b)| (b - a) as u64).sum::<u64>();
                                let dirty_binding = dp_ref.zip(dp_alloc).and_then(|(dp, _)| {
                                    dp_targets.get(i).map(|t| (dp.gpu_va + dp_slot_base + t.slot_off as u64, t.words))
                                });
                                record_bidir_ranges_tracked(
                                    &list,
                                    &state.pipelines,
                                    gs.title_va(),
                                    gs.title_bias,
                                    shadow.buffer.GetGPUVirtualAddress(),
                                    0,
                                    gs.before_va(),
                                    0,
                                    ranges,
                                    SyncOp::GpuWriteBackOnly,
                                    dirty_binding,
                                );
                                kept.push(shadow.clone());
                            }
                            if let (Some(dp), Some(_)) = (dp_ref, dp_alloc)
                                && dp_used_bytes > 0
                            {
                                list.ResourceBarrier(&[
                                    uav_barrier(),
                                    transition(
                                        &dp.gpu,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    ),
                                ]);
                                list.CopyBufferRegion(
                                    &dp.readback,
                                    dp_slot_base,
                                    &dp.gpu,
                                    dp_slot_base,
                                    dp_used_bytes as u64,
                                );
                                list.ResourceBarrier(&[transition(
                                    &dp.gpu,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                                    D3D12_RESOURCE_STATE_COMMON,
                                )]);
                            } else {
                                list.ResourceBarrier(&[uav_barrier()]);
                            }
                            if let (Some(ts), Some(slot)) = (state.timestamps.as_ref(), ts_alloc) {
                                ts.end_and_resolve(&list, slot, true);
                            }
                            list.Close()?;
                            STATS[4].fetch_add(1, Ordering::Relaxed);
                            STATS[5].fetch_add(total_bytes, Ordering::Relaxed);
                            list.cast()
                        })()
                    };
                    match recorded {
                        Ok(cmd) => {
                            post_list = Some(cmd);
                            ts_has_post = ts_alloc.is_some();
                            if let Some(slot) = dp_alloc {
                                dirty_slot = Some((slot, dp_targets));
                            }
                        }
                        Err(e) => {
                            if let (Some(dp), Some(slot)) = (state.dirty_pages.as_ref(), dp_alloc) {
                                dp.release_slot(slot);
                            }
                            tracing::warn!(error = %e, "VICTORY_GPU_SYNC post-sync list failed");
                        }
                    }
                    lists.push((alloc, list));
                } else if let (Some(dp), Some(slot)) = (state.dirty_pages.as_ref(), dp_alloc) {
                    dp.release_slot(slot);
                }
            }
            SyncMode::SrvDirectUpload => {
                for (shadow, ranges) in sync_targets {
                    kept.push(shadow.clone());
                    srv_ranges.push((shadow, ranges));
                }
            }
            SyncMode::Unsupported => {}
        }
    }

    let ts_slot = match (ts_alloc, ts_has_pre || ts_has_post) {
        (Some(slot), true) => Some((slot, ts_has_pre, ts_has_post)),
        (Some(slot), false) => {
            if let Some(ts) = state.timestamps.as_ref() {
                ts.release_slot(slot);
            }
            None
        }
        (None, _) => None,
    };

    let counted_srv_flight = !srv_ranges.is_empty();
    if counted_srv_flight {
        SRV_IN_FLIGHT.fetch_add(1, Ordering::Release);
    }

    Some(SubmissionPlan {
        pre_list,
        post_list,
        lists,
        _kept: kept,
        _transient: transient,
        pending_init,
        srv_ranges,
        ts_slot,
        dirty_slot,
        submitted: false,
        counted_srv_flight,
    })
}

/// Finishes a submission's GPU sync plan after its fence has completed.
///
/// Collects non-stalling `ID3D12QueryHeap` GPU dispatch time, notifies `watch::landed_beside` +
/// `descriptor::host::remapped` for any GPU-written 64 KiB pages, recycles command lists, and in
/// `SrvDirectUpload` mode drains compact L1-first GPU-only delta passes into title memory.
pub(crate) fn finish_submission(host: &Host, mut plan: SubmissionPlan) {
    let Some(state) = host_state(host) else {
        return;
    };
    if let (Some(ts), Some((slot, has_pre, has_post))) = (state.timestamps.as_ref(), plan.ts_slot.take()) {
        let us = if plan.submitted {
            ts.collect_and_free(slot, has_pre, has_post)
        } else {
            ts.release_slot(slot);
            0
        };
        if us > 0 {
            GPU_DISPATCH_US.fetch_add(us, Ordering::Relaxed);
        }
    }
    if let (Some(dp), Some((slot, targets))) = (state.dirty_pages.as_ref(), plan.dirty_slot.take()) {
        if plan.submitted {
            let _ = dp.notify_and_free(slot, &targets);
        } else {
            dp.release_slot(slot);
        }
    }
    if !plan.srv_ranges.is_empty() {
        let mut rig_slot = state.srv_rig.lock().unwrap_or_else(|e| e.into_inner());
        if rig_slot.is_none() {
            *rig_slot = ProbeRig::new(&host.device).ok();
        }
        if let Some(rig) = rig_slot.as_mut() {
            for (shadow, ranges) in &plan.srv_ranges {
                let Some(gs) = shadow.gpu_sync_ref() else {
                    continue;
                };
                let (start, len) = shadow.span();
                let range_bytes: u64 = ranges.iter().map(|&(a, b)| (b - a) as u64).sum();
                STATS[4].fetch_add(1, Ordering::Relaxed);
                STATS[5].fetch_add(range_bytes, Ordering::Relaxed);
                loop {
                    let wait_t0 = std::time::Instant::now();
                    // SAFETY: live buffers; single-owner delta scratch under `srv_rig` lock.
                    let ran = rig.run_timed(|list| unsafe {
                        list.CopyBufferRegion(
                            &state.pipelines.deltas_gpu,
                            0,
                            &state.pipelines.deltas_zero,
                            0,
                            SRV_DELTA_HEADER_BYTES,
                        );
                        list.ResourceBarrier(&[transition(
                            &state.pipelines.deltas_gpu,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )]);
                        record_srv_ranges(
                            list,
                            &state.pipelines,
                            gs.title_va(),
                            gs.title_bias,
                            shadow.buffer.GetGPUVirtualAddress(),
                            0,
                            gs.before_va(),
                            0,
                            ranges,
                            SyncOp::GpuWriteBackOnly,
                            SRV_MAX_DELTAS,
                        );
                        list.ResourceBarrier(&[
                            uav_barrier(),
                            transition(
                                &state.pipelines.deltas_gpu,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                            ),
                        ]);
                        list.CopyBufferRegion(
                            &state.pipelines.deltas_readback,
                            0,
                            &state.pipelines.deltas_gpu,
                            0,
                            SRV_DELTA_BUFFER_BYTES,
                        );
                        list.ResourceBarrier(&[transition(
                            &state.pipelines.deltas_gpu,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_COMMON,
                        )]);
                    });
                    FENCE_WAIT_US.fetch_add(wait_t0.elapsed().as_micros() as u64, Ordering::Relaxed);
                    match ran {
                        Ok((gpu_us, _)) => {
                            GPU_DISPATCH_US.fetch_add(gpu_us.max(0.0).round() as u64, Ordering::Relaxed);
                        }
                        Err(_) => break,
                    }
                    STATS[6].fetch_add(1, Ordering::Relaxed);
                    // SAFETY: readback buffer populated by completed GPU run; title memory live.
                    let (_, overflow) = unsafe {
                        apply_srv_deltas_to_title(
                            state.pipelines.deltas_readback_ptr as *const u8,
                            SRV_MAX_DELTAS,
                            start as *mut u8,
                            len as usize,
                        )
                    };
                    if !overflow {
                        break;
                    }
                }
            }
        }
    }
    for pair in std::mem::take(&mut plan.lists) {
        return_cmd_list(state, pair);
    }
    maybe_log_stats(state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;

    fn create_debug_device(warp: bool) -> Option<ID3D12Device> {
        // SAFETY: standard D3D12 debug layer, GPU-based validation & device creation.
        unsafe {
            if crate::switch::on("VICTORY_D3D12_DEBUG") {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok()
                    && let Some(d) = &debug
                {
                    d.EnableDebugLayer();
                    if crate::switch::on("VICTORY_D3D12_GBV")
                        && let Ok(d1) = d.cast::<ID3D12Debug1>()
                    {
                        d1.SetEnableGPUBasedValidation(true);
                    }
                }
            }
            if warp {
                let adapter = crate::host::warp_adapter().ok()?;
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                device
            } else {
                victory_shader::emit::host::create_preferred_device().ok()
            }
        }
    }

    struct OwnedAlloc {
        base: *mut u8,
        len: usize,
    }

    impl OwnedAlloc {
        fn new(len: usize) -> Self {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            // SAFETY: test-owned VirtualAlloc buffer.
            let base = unsafe {
                mem::VirtualAlloc(
                    std::ptr::null(),
                    len,
                    mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
                    mem::PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!base.is_null(), "VirtualAlloc failed for {len} bytes");
            Self { base, len }
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            // SAFETY: valid test allocation.
            unsafe { std::slice::from_raw_parts_mut(self.base, self.len) }
        }
    }

    impl Drop for OwnedAlloc {
        fn drop(&mut self) {
            // SAFETY: test-owned VirtualAlloc release.
            unsafe {
                let _ = mem::VirtualFree(self.base.cast(), 0, mem::MEM_RELEASE);
            }
        }
    }

    /// Test compute shader that mutates ONLY GPU-owned lanes directly in L1 `shadow_buf`,
    /// without clobbering CPU-owned sibling dwords or sibling bytes:
    /// - `kind == 0`: writes `(round << 16) | (d * 2)` to even dwords `0, 2, 4, ...`.
    /// - `kind == 1`: updates ONLY byte 2 (`bits 16..24`) of every dword to `round & 0xFF`.
    /// - `kind == 2`: writes all 4 bytes of every dword with `(round & 0x7F) | 0x01` replicated
    ///   (all high bits clear, forcing `gpu_only == 0xFFFFFFFFu` against a `0` baseline).
    const TEST_GPU_MUTATE_HLSL: &str = r#"
    #define RS "RootConstants(num32BitConstants=3, b0), UAV(u0)"
    cbuffer Args : register(b0) { uint count; uint round; uint kind; };
    RWByteAddressBuffer shadow_buf : register(u0);
    [RootSignature(RS)]
    [numthreads(256, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {
        if (id.x >= count) return;
        if (kind == 0u) {
            uint d = id.x * 2u;
            shadow_buf.Store(d * 4u, (round << 16u) | d);
        } else if (kind == 1u) {
            uint addr = id.x * 4u;
            uint cur = shadow_buf.Load(addr);
            uint next = (cur & 0xFF00FFFFu) | ((round & 0xFFu) << 16u);
            shadow_buf.Store(addr, next);
        } else {
            uint b = (round & 0x7Fu) | 0x01u;
            uint v = b | (b << 8u) | (b << 16u) | (b << 24u);
            shadow_buf.Store(id.x * 4u, v);
        }
    }
    "#;

    fn run_srv_bidir_drain(
        rig: &mut ProbeRig,
        pipelines: &EnginePipelines,
        synced: &SyncedShadow,
        shadow_l1: &ID3D12Resource,
        ranges: &[(usize, usize)],
        target: (*mut u8, usize, u32, SyncOp),
    ) -> u32 {
        let (title_ptr, region_len, max_deltas, op) = target;
        let title_va = synced.title_va();
        let shadow_va = unsafe { shadow_l1.GetGPUVirtualAddress() };
        let before_va = synced.before_va();
        let mut passes = 0u32;
        let copy_bytes = SRV_DELTA_HEADER_BYTES + (max_deltas as u64) * SRV_DELTA_STRIDE_BYTES;
        loop {
            passes += 1;
            assert!(passes <= 128, "SRV delta drain did not converge");
            rig.run_timed(|list| unsafe {
                list.CopyBufferRegion(&pipelines.deltas_gpu, 0, &pipelines.deltas_zero, 0, SRV_DELTA_HEADER_BYTES);
                list.ResourceBarrier(&[transition(
                    &pipelines.deltas_gpu,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
                record_srv_ranges(
                    list,
                    pipelines,
                    title_va,
                    synced.title_bias,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    ranges,
                    op,
                    max_deltas,
                );
                list.ResourceBarrier(&[
                    uav_barrier(),
                    transition(
                        &pipelines.deltas_gpu,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list.CopyBufferRegion(&pipelines.deltas_readback, 0, &pipelines.deltas_gpu, 0, copy_bytes);
                list.ResourceBarrier(&[transition(
                    &pipelines.deltas_gpu,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            })
            .expect("SRV delta drain run failed");
            // SAFETY: readback populated, title memory valid.
            let (_, overflow) = unsafe {
                apply_srv_deltas_to_title(
                    pipelines.deltas_readback_ptr as *const u8,
                    max_deltas,
                    title_ptr,
                    region_len,
                )
            };
            if !overflow {
                break;
            }
        }
        passes
    }

    fn verify_device_against_oracle(device: &ID3D12Device, label: &str) {
        let report = probe_device(device);
        eprintln!(
            "[{label}] probe: adapter='{}' sw={} dev13={} open={} srv={} uav={} copy_to_l1={} copy_from_l1={} atomics={} flags={:#x} mode={:?} debug_errs={}",
            report.adapter_name,
            report.is_software,
            report.has_device13,
            report.l0_open_ok,
            report.l0_srv_read_ok,
            report.l0_uav_write_ok,
            report.l0_copy_to_l1_ok,
            report.l0_copy_from_l1_ok,
            report.l0_atomics_ok,
            report.l0_heap_flags,
            report.mode,
            report.debug_errors.len()
        );
        assert!(
            report.debug_errors.is_empty(),
            "[{label}] D3D12 debug layer errors during probe: {:?}",
            report.debug_errors
        );
        if report.mode == SyncMode::Unsupported {
            eprintln!("[{label}] device unsupported ({:?}); skipping GPU sync oracle test", report.notes);
            return;
        }

        let mut rig = ProbeRig::new(device).expect("probe rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());
        let pipelines = EnginePipelines::new(
            device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");

        // Test both 64 KiB aligned base and non-64KiB (4-byte) aligned base with unaligned length.
        let configs = [
            (0usize, 2 * (K64 as usize) + 13),
            (0x1230usize, (K64 as usize) + 4099),
            (0xFFFCusize, (K64 as usize) + 77),
        ];

        let mut seed = 0x51A7_9E37_79B9_7F4Au64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        for &(base_offset, region_len) in &configs {
            let alloc_len = (base_offset + region_len).next_multiple_of(K64 as usize) + (K64 as usize);
            let mut alloc = OwnedAlloc::new(alloc_len);
            let start = alloc.base as u64 + base_offset as u64;
            let l1_size = (region_len as u64).next_multiple_of(K64);

            // Test both BidirectionalUav (if supported) and forced SrvDirectUpload (including overflow).
            let mut modes = vec![(SyncMode::SrvDirectUpload, 64u32)]; // small max_deltas forces overflow loop!
            if report.mode == SyncMode::BidirectionalUav {
                modes.insert(0, (SyncMode::BidirectionalUav, SRV_MAX_DELTAS));
            }

            for (test_mode, max_deltas) in modes {
                let synced = try_create_region(device, test_mode, start, region_len as u64, l1_size)
                    .unwrap_or_else(|r| panic!("[{label}] try_create_region failed: {r:?}"));
                assert_eq!(synced.title_bias as usize, base_offset % (K64 as usize));
                let shadow_l1 = make_committed(
                    device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    l1_size,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )
                .expect("shadow_l1");

                let title_va = synced.title_va();
                let shadow_va = unsafe { shadow_l1.GetGPUVirtualAddress() };
                let before_va = synced.before_va();

                // Sentinel fill outside region to prove out-of-bounds bytes are never touched!
                alloc.as_mut_slice().fill(0xCC);
                let title_init: Vec<u8> = (0..region_len).map(|_| (next() & 0xFF) as u8).collect();
                alloc.as_mut_slice()[base_offset..base_offset + region_len].copy_from_slice(&title_init);
                let mut oracle_t = title_init.clone();
                let mut oracle_s = vec![0xAAu8; region_len];
                let mut oracle_b = vec![0xBBu8; region_len];
                rig.write_l1(&shadow_l1, 0, &oracle_s).unwrap();
                rig.write_l1(&synced.before_buffer, 0, &oracle_b).unwrap();

                // Step 1: Init whole region.
                let init_ranges = [(0, region_len)];
                cpu_three_way_merge_oracle(&mut oracle_t, &mut oracle_s, &mut oracle_b, &init_ranges, SyncOp::Init);
                rig.run_timed(|list| unsafe {
                    match test_mode {
                        SyncMode::BidirectionalUav => record_bidir_ranges(
                            list,
                            &pipelines,
                            title_va,
                            synced.title_bias,
                            shadow_va,
                            0,
                            before_va,
                            0,
                            &init_ranges,
                            SyncOp::Init,
                        ),
                        SyncMode::SrvDirectUpload => record_srv_ranges(
                            list,
                            &pipelines,
                            title_va,
                            synced.title_bias,
                            shadow_va,
                            0,
                            before_va,
                            0,
                            &init_ranges,
                            SyncOp::Init,
                            max_deltas,
                        ),
                        SyncMode::Unsupported => unreachable!(),
                    }
                })
                .unwrap();

                // Step 2: Multiple adversarial rounds of unaligned, overlapping, same-dword disjoint,
                // CPU-only, GPU-only, and CPU+GPU conflict mutations covering Bidirectional3Way,
                // Upload3Way, and L1-first GpuWriteBackOnly!
                let round_ranges: Vec<Vec<(usize, usize)>> = vec![
                    // Round 0 (Bidirectional3Way): same-dword disjoint + single-byte + unaligned boundaries!
                    vec![(0, 1), (2, 3), (5, 13), (65533, (65541).min(region_len)), (region_len - 5, region_len)],
                    // Round 1 (Upload3Way): overlapping and unsorted ranges!
                    vec![(300, 900), (100, 500), (0, 101), (899, 1403), (4093, 8201)],
                    // Round 2 (Bidirectional3Way): full region with unaligned tail (exercises SRV overflow)!
                    vec![(0, region_len)],
                    // Round 3 (L1-first GpuWriteBackOnly): unaligned boundaries + interior + full region!
                    vec![(1, 7), (15, 65540.min(region_len)), (0, region_len)],
                ];

                for (round, raw_ranges) in round_ranges.iter().enumerate() {
                    let coalesced = coalesce_ranges(raw_ranges, region_len);
                    let mut gpu_shadow = rig.read_l1(&shadow_l1, 0, region_len).unwrap();
                    assert_eq!(gpu_shadow, oracle_s, "[{label}] mode={test_mode:?} pre-round {round} shadow");

                    // Inject mutations across all 4 conflict quadrants (including OUTSIDE ranges to prove clamping!).
                    for k in 0..400usize {
                        let pos = (next() as usize) % region_len;
                        let span = [1usize, 2, 3, 4, 7, 16][(next() as usize) % 6].min(region_len - pos);
                        let kind = k % 4;
                        for j in 0..span {
                            let idx = pos + j;
                            match kind {
                                0 => {
                                    // CPU-only write
                                    let v = (next() & 0xFF) as u8;
                                    alloc.as_mut_slice()[base_offset + idx] = v;
                                    oracle_t[idx] = v;
                                }
                                1 => {
                                    // GPU-only write
                                    let v = (next() & 0xFF) as u8;
                                    gpu_shadow[idx] = v;
                                    oracle_s[idx] = v;
                                }
                                2 => {
                                    // BOTH CPU and GPU write the EXACT same byte -> CPU/title MUST win!
                                    let v_gpu = oracle_b[idx].wrapping_add(17).wrapping_add((k & 0x3F) as u8);
                                    let v_cpu = v_gpu.wrapping_add(31);
                                    gpu_shadow[idx] = v_gpu;
                                    oracle_s[idx] = v_gpu;
                                    alloc.as_mut_slice()[base_offset + idx] = v_cpu;
                                    oracle_t[idx] = v_cpu;
                                }
                                _ => {
                                    // Both write identical value (still must converge cleanly)
                                    let v = oracle_b[idx] ^ 0xA5;
                                    gpu_shadow[idx] = v;
                                    oracle_s[idx] = v;
                                    alloc.as_mut_slice()[base_offset + idx] = v;
                                    oracle_t[idx] = v;
                                }
                            }
                        }
                    }
                    // Also guarantee an explicit byte-conflict on an unaligned boundary byte!
                    if let Some(&(lo, hi)) = coalesced.first() {
                        let b_conflict = lo;
                        let vg = oracle_b[b_conflict].wrapping_add(0x41);
                        let vc = oracle_b[b_conflict].wrapping_add(0x82);
                        gpu_shadow[b_conflict] = vg;
                        oracle_s[b_conflict] = vg;
                        alloc.as_mut_slice()[base_offset + b_conflict] = vc;
                        oracle_t[b_conflict] = vc;
                        if hi > lo + 1 {
                            let b_gpu = hi - 1;
                            let vg2 = oracle_b[b_gpu].wrapping_add(0x53);
                            gpu_shadow[b_gpu] = vg2;
                            oracle_s[b_gpu] = vg2;
                            // Ensure CPU matches baseline on b_gpu
                            alloc.as_mut_slice()[base_offset + b_gpu] = oracle_b[b_gpu];
                            oracle_t[b_gpu] = oracle_b[b_gpu];
                        }
                    }
                    rig.write_l1(&shadow_l1, 0, &gpu_shadow).unwrap();

                    let op = match round {
                        1 => SyncOp::Upload3Way,
                        3 => SyncOp::GpuWriteBackOnly,
                        _ => SyncOp::Bidirectional3Way,
                    };
                    cpu_three_way_merge_oracle(&mut oracle_t, &mut oracle_s, &mut oracle_b, &coalesced, op);

                    match (test_mode, op) {
                        (SyncMode::BidirectionalUav, _) => {
                            rig.run_timed(|list| unsafe {
                                record_bidir_ranges(
                                    list,
                                    &pipelines,
                                    title_va,
                                    synced.title_bias,
                                    shadow_va,
                                    0,
                                    before_va,
                                    0,
                                    &coalesced,
                                    op,
                                );
                            })
                            .unwrap();
                        }
                        (SyncMode::SrvDirectUpload, SyncOp::Bidirectional3Way | SyncOp::GpuWriteBackOnly) => {
                            let passes = run_srv_bidir_drain(
                                &mut rig,
                                &pipelines,
                                &synced,
                                &shadow_l1,
                                &coalesced,
                                (unsafe { alloc.base.add(base_offset) }, region_len, max_deltas, op),
                            );
                            if round >= 2 {
                                assert!(passes >= 2, "expected delta overflow loop to exercise multiple passes (got {passes})");
                            }
                        }
                        (SyncMode::SrvDirectUpload, _) => {
                            rig.run_timed(|list| unsafe {
                                record_srv_ranges(
                                    list,
                                    &pipelines,
                                    title_va,
                                    synced.title_bias,
                                    shadow_va,
                                    0,
                                    before_va,
                                    0,
                                    &coalesced,
                                    op,
                                    max_deltas,
                                );
                            })
                            .unwrap();
                        }
                        (SyncMode::Unsupported, _) => unreachable!(),
                    }

                    let got_t = &alloc.as_mut_slice()[base_offset..base_offset + region_len];
                    let got_s = rig.read_l1(&shadow_l1, 0, region_len).unwrap();
                    let got_b = rig.read_l1(&synced.before_buffer, 0, region_len).unwrap();
                    assert_eq!(got_t, oracle_t.as_slice(), "[{label}] base={base_offset:#x} mode={test_mode:?} round={round} op={op:?} title mismatch");
                    assert_eq!(got_s, oracle_s, "[{label}] base={base_offset:#x} mode={test_mode:?} round={round} op={op:?} shadow mismatch");
                    assert_eq!(got_b, oracle_b, "[{label}] base={base_offset:#x} mode={test_mode:?} round={round} op={op:?} before mismatch");

                    // Verify sentinels outside [base_offset, base_offset + region_len) untouched!
                    assert!(
                        alloc.as_mut_slice()[..base_offset].iter().all(|&b| b == 0xCC),
                        "[{label}] leading sentinel corrupted"
                    );
                    assert!(
                        alloc.as_mut_slice()[base_offset + region_len..].iter().all(|&b| b == 0xCC),
                        "[{label}] trailing sentinel corrupted"
                    );
                }

                // Step 3: Concurrent CPU thread writing disjoint dwords while GPU runs Bidirectional3Way!
                if test_mode == SyncMode::BidirectionalUav {
                    let sync_len = (K64 as usize).min(region_len & !3);
                    let ranges = [(0, sync_len)];
                    rig.run_timed(|list| unsafe {
                        record_bidir_ranges(
                            list,
                            &pipelines,
                            title_va,
                            synced.title_bias,
                            shadow_va,
                            0,
                            before_va,
                            0,
                            &ranges,
                            SyncOp::Init,
                        );
                    })
                    .unwrap();
                    let (mut_root, mut_pso) = make_compute_pso(device, TEST_GPU_MUTATE_HLSL).expect("gpu mutate pso");
                    let title_base_addr = alloc.base as usize + base_offset;
                    let started = AtomicBool::new(false);
                    let stop = AtomicBool::new(false);
                    let even_count = (sync_len / 4).div_ceil(2) as u32;
                    std::thread::scope(|scope| {
                        let writer = scope.spawn(|| {
                            let mut val = 1u32;
                            loop {
                                // CPU owns odd dwords (1, 3, 5, ...).
                                for d in (1..sync_len / 4).step_by(2) {
                                    // SAFETY: 4-byte aligned committed memory; disjoint from GPU's even dwords.
                                    let slot = unsafe { &*((title_base_addr + d * 4) as *const AtomicU32) };
                                    slot.store(val, Ordering::Relaxed);
                                }
                                started.store(true, Ordering::Release);
                                if stop.load(Ordering::Relaxed) {
                                    return val;
                                }
                                val = val.wrapping_add(1).max(1);
                            }
                        });
                        while !started.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                        // GPU shader writes ONLY even dwords (0, 2, 4, ...) in L1 VRAM for 30 rounds!
                        for round in 1..=30u32 {
                            rig.run_timed(|list| unsafe {
                                let args = [even_count, round, 0u32];
                                list.SetComputeRootSignature(&mut_root);
                                list.SetPipelineState(&mut_pso);
                                list.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
                                list.SetComputeRootUnorderedAccessView(1, shadow_va);
                                list.Dispatch(even_count.div_ceil(256), 1, 1);
                                list.ResourceBarrier(&[uav_barrier()]);
                                record_bidir_ranges(
                                    list,
                                    &pipelines,
                                    title_va,
                                    synced.title_bias,
                                    shadow_va,
                                    0,
                                    before_va,
                                    0,
                                    &ranges,
                                    SyncOp::Bidirectional3Way,
                                );
                            })
                            .unwrap();
                            for d in (0..sync_len / 4).step_by(2) {
                                let expect = (round << 16) | (d as u32);
                                // SAFETY: aligned read of GPU-owned even dword after GPU fence wait.
                                let got = unsafe { &*((title_base_addr + d * 4) as *const AtomicU32) }.load(Ordering::Relaxed);
                                assert_eq!(got, expect, "[{label}] concurrent GPU even dword {d} round {round}");
                            }
                        }
                        stop.store(true, Ordering::Relaxed);
                        let last_cpu = writer.join().expect("CPU writer thread");
                        // Verify NOT A SINGLE CPU-owned odd dword was clobbered back to an old value!
                        for d in (1..sync_len / 4).step_by(2) {
                            // SAFETY: aligned read after CPU thread joined.
                            let got = unsafe { &*((title_base_addr + d * 4) as *const AtomicU32) }.load(Ordering::Relaxed);
                            assert_eq!(got, last_cpu, "[{label}] concurrent CPU odd dword {d} clobbered!");
                        }
                    });
                }

                let errs = drain_debug_errors(rig.info_queue.as_ref());
                assert!(
                    errs.is_empty(),
                    "[{label}] D3D12 debug layer errors after base={base_offset:#x} mode={test_mode:?}: {errs:?}"
                );
                drop((shadow_l1, synced));
            }
        }
    }

    #[test]
    fn hw_capability_probe_and_timestamp_benchmark() {
        let Some(device) = create_debug_device(false) else {
            eprintln!("no hardware D3D12 device available; skipping HW probe");
            return;
        };
        let report = probe_device(&device);
        eprintln!("=== LIVE HW GPU CAPABILITY PROBE ===");
        eprintln!(
            "adapter='{}' software={} device13={} open={} srv={} uav={} copy_l0_to_l1={} copy_l1_to_l0={} atomics={} heap_flags={:#x} mode={:?}",
            report.adapter_name,
            report.is_software,
            report.has_device13,
            report.l0_open_ok,
            report.l0_srv_read_ok,
            report.l0_uav_write_ok,
            report.l0_copy_to_l1_ok,
            report.l0_copy_from_l1_ok,
            report.l0_atomics_ok,
            report.l0_heap_flags,
            report.mode
        );
        for note in &report.notes {
            eprintln!("  note: {note}");
        }
        assert!(
            report.debug_errors.is_empty(),
            "HW probe hit D3D12 debug errors: {:?}",
            report.debug_errors
        );
        assert!(report.has_device13 && report.l0_open_ok && report.l0_srv_read_ok, "expected live HW L0 SRV support");

        // Measure exact GB/s and CPU microseconds via ID3D12QueryHeap GPU timestamps,
        // comparing OLD `Bidirectional3Way` (with L0 atomics) vs NEW L1-first `GpuWriteBackOnly`
        // (with non-atomic posted PCIe stores on discrete GPUs)!
        let mut rig = ProbeRig::new(&device).expect("probe rig");
        let pipelines_old_atomic = EnginePipelines::new(
            &device,
            report.mode,
            if report.l0_atomics_ok { ATOMIC_PARTIAL | ATOMIC_FULL } else { 0 },
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines_old");
        let pipelines_fast = EnginePipelines::new(
            &device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines_fast");

        let gbv = std::env::var_os("VICTORY_D3D12_GBV").is_some_and(|v| v == "1");
        let sizes_mb: &[usize] = if cfg!(debug_assertions) || gbv { &[4] } else { &[4, 16, 64] };
        let iters: u8 = if cfg!(debug_assertions) || gbv { 2 } else { 5 };

        for &size_mb in sizes_mb {
            let bytes = size_mb << 20;
            let contig_bytes = (bytes / 10) & !15;
            let mut alloc = OwnedAlloc::new(bytes);
            for (i, b) in alloc.as_mut_slice().iter_mut().enumerate() {
                *b = ((i.wrapping_mul(31) + 7) & 0xFF) as u8;
            }
            let synced = try_create_region(&device, report.mode, alloc.base as u64, bytes as u64, bytes as u64)
                .expect("benchmark region");
            let shadow_l1 = make_committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                bytes as u64,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
            .expect("shadow_l1");
            let title_va = synced.title_va();
            let shadow_va = unsafe { shadow_l1.GetGPUVirtualAddress() };
            let before_va = synced.before_va();
            let ranges = [(0, bytes)];

            // 1. DMA Init (`L0 -> L1 shadow` + `L1 shadow -> L1 before`) at full bus speed.
            let mut best_dma = (f64::MAX, f64::MAX);
            for _ in 0..iters {
                let (gpu_us, cpu_us) = rig
                    .run_timed(|list| unsafe {
                        record_init_dma(list, &synced.title_buffer, 0, &shadow_l1, &synced.before_buffer, &ranges, false);
                    })
                    .unwrap();
                if gpu_us < best_dma.0 {
                    best_dma = (gpu_us, cpu_us);
                }
            }

            let mut best_old_clean = (f64::MAX, f64::MAX);
            let mut best_old_contig10 = (f64::MAX, f64::MAX);
            let mut best_old_scatter10 = (f64::MAX, f64::MAX);
            let mut best_new_clean = (f64::MAX, f64::MAX);
            let mut best_new_contig10 = (f64::MAX, f64::MAX);
            let mut best_new_scatter10 = (f64::MAX, f64::MAX);
            if report.mode == SyncMode::BidirectionalUav {
                // 2. OLD path: `Bidirectional3Way` with `use_atomics = true` (0% clean, 10% contig, 10% scattered).
                for _ in 0..iters {
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_old_atomic,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::Bidirectional3Way,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_old_clean.0 {
                        best_old_clean = (gpu_us, cpu_us);
                    }
                }
                let mut gpu_data = alloc.as_mut_slice().to_vec();
                for iter in 0..iters {
                    for b in &mut gpu_data[..contig_bytes] {
                        *b = b.wrapping_add(13).wrapping_add(iter);
                    }
                    rig.write_l1(&shadow_l1, 0, &gpu_data).unwrap();
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_old_atomic,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::Bidirectional3Way,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_old_contig10.0 {
                        best_old_contig10 = (gpu_us, cpu_us);
                    }
                }
                gpu_data.copy_from_slice(alloc.as_mut_slice());
                for iter in 0..iters {
                    for d in (0..bytes / 4).step_by(10) {
                        gpu_data[d * 4] = gpu_data[d * 4].wrapping_add(17).wrapping_add(iter);
                    }
                    rig.write_l1(&shadow_l1, 0, &gpu_data).unwrap();
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_old_atomic,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::Bidirectional3Way,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_old_scatter10.0 {
                        best_old_scatter10 = (gpu_us, cpu_us);
                    }
                }

                // 3. NEW path: L1-first `GpuWriteBackOnly` with `use_atomics = false` on discrete PCIe!
                rig.run_timed(|list| unsafe {
                    record_init_dma(list, &synced.title_buffer, 0, &shadow_l1, &synced.before_buffer, &ranges, false);
                })
                .unwrap();
                for _ in 0..iters {
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_fast,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::GpuWriteBackOnly,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_new_clean.0 {
                        best_new_clean = (gpu_us, cpu_us);
                    }
                }
                gpu_data.copy_from_slice(alloc.as_mut_slice());
                for iter in 0..iters {
                    for b in &mut gpu_data[..contig_bytes] {
                        *b = b.wrapping_add(19).wrapping_add(iter);
                    }
                    rig.write_l1(&shadow_l1, 0, &gpu_data).unwrap();
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_fast,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::GpuWriteBackOnly,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_new_contig10.0 {
                        best_new_contig10 = (gpu_us, cpu_us);
                    }
                }
                gpu_data.copy_from_slice(alloc.as_mut_slice());
                for iter in 0..iters {
                    for d in (0..bytes / 4).step_by(10) {
                        gpu_data[d * 4] = gpu_data[d * 4].wrapping_add(23).wrapping_add(iter);
                    }
                    rig.write_l1(&shadow_l1, 0, &gpu_data).unwrap();
                    let (gpu_us, cpu_us) = rig
                        .run_timed(|list| unsafe {
                            record_bidir_ranges(
                                list,
                                &pipelines_fast,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::GpuWriteBackOnly,
                            );
                        })
                        .unwrap();
                    if gpu_us < best_new_scatter10.0 {
                        best_new_scatter10 = (gpu_us, cpu_us);
                    }
                }
            }

            // 4. Apples-to-apples CPU 3-way merge oracle baseline on 10% contiguous GPU-dirty pattern.
            let mut cpu_t = alloc.as_mut_slice().to_vec();
            let mut cpu_s = cpu_t.clone();
            let mut cpu_b = cpu_t.clone();
            for b in &mut cpu_s[..contig_bytes] {
                *b ^= 0x55;
            }
            let cpu_t0 = std::time::Instant::now();
            cpu_three_way_merge_oracle(&mut cpu_t, &mut cpu_s, &mut cpu_b, &ranges, SyncOp::GpuWriteBackOnly);
            let cpu_oracle_us = cpu_t0.elapsed().as_secs_f64() * 1e6;

            let gbps = |us: f64| if us > 0.0 { (bytes as f64 / 1e9) / (us / 1e6) } else { 0.0 };
            eprintln!(
                "[BENCH {size_mb:>2} MiB] Init DMA: GPU {:7.1} us ({:5.2} GB/s), CPU {:5.1} us | \
                 OLD Bidir+Atomics clean: {:7.1} us ({:5.2} GB/s), contig10: {:7.1} us ({:5.2} GB/s), scatter10: {:7.1} us ({:5.2} GB/s) | \
                 NEW L1-First WriteBack clean: {:6.1} us ({:6.2} GB/s), contig10: {:7.1} us ({:5.2} GB/s), scatter10: {:7.1} us ({:5.2} GB/s), CPU {:5.1} us | \
                 CPU oracle: {:7.1} us ({:5.2} GB/s)",
                best_dma.0,
                gbps(best_dma.0),
                best_dma.1,
                best_old_clean.0,
                gbps(best_old_clean.0),
                best_old_contig10.0,
                gbps(best_old_contig10.0),
                best_old_scatter10.0,
                gbps(best_old_scatter10.0),
                best_new_clean.0,
                gbps(best_new_clean.0),
                best_new_contig10.0,
                gbps(best_new_contig10.0),
                best_new_scatter10.0,
                gbps(best_new_scatter10.0),
                best_new_contig10.1,
                cpu_oracle_us,
                gbps(cpu_oracle_us),
            );

            let errs = drain_debug_errors(rig.info_queue.as_ref());
            assert!(errs.is_empty(), "HW benchmark D3D12 debug errors ({size_mb} MiB): {errs:?}");
        }
    }

    #[test]
    fn hw_three_way_merge_matches_cpu_oracle() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        verify_device_against_oracle(&device, "HW");
    }

    #[test]
    fn warp_three_way_merge_matches_cpu_oracle() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = create_debug_device(true) else {
            return;
        };
        verify_device_against_oracle(&device, "WARP");
    }

    #[test]
    fn fallback_when_unaligned_or_unopenable() {
        // 1. Unaligned base address -> FallbackReason::UnalignedBase
        let alloc = OwnedAlloc::new(4 * (K64 as usize));
        for misalign in [1u64, 2, 3, 5] {
            assert_eq!(
                check_eligibility(alloc.base as u64 + misalign, 4096),
                Err(FallbackReason::UnalignedBase)
            );
        }
        // 2. Empty or > 1 GiB -> FallbackReason::EmptyOrTooLarge
        assert_eq!(check_eligibility(alloc.base as u64, 0), Err(FallbackReason::EmptyOrTooLarge));
        assert_eq!(
            check_eligibility(alloc.base as u64, MAX_REGION_BYTES + 4),
            Err(FallbackReason::EmptyOrTooLarge)
        );
        // 3. Reserved (uncommitted) memory -> NotCommittedReadWrite
        // SAFETY: test-owned reservation.
        unsafe {
            let res = mem::VirtualAlloc(std::ptr::null(), 2 * (K64 as usize), mem::MEM_RESERVE, mem::PAGE_NOACCESS);
            assert!(!res.is_null());
            assert_eq!(
                check_eligibility(res as u64, K64),
                Err(FallbackReason::NotCommittedReadWrite)
            );
            mem::VirtualFree(res, 0, mem::MEM_RELEASE);
        }
        // 4. Guarded (PAGE_READONLY) page -> NotCommittedReadWrite
        // SAFETY: test-owned allocation, restored before drop.
        unsafe {
            let mut old = 0u32;
            assert_ne!(mem::VirtualProtect(alloc.base.cast(), K64 as usize, mem::PAGE_READONLY, &mut old), 0);
            assert_eq!(
                check_eligibility(alloc.base as u64, 2 * K64),
                Err(FallbackReason::NotCommittedReadWrite)
            );
            mem::VirtualProtect(alloc.base.cast(), K64 as usize, mem::PAGE_READWRITE, &mut old);
        }
        // 5. View memory -> ViewMemory
        let fake_view = 0x7ff2_0000_0000usize;
        crate::inplace::title_memory_view(fake_view, 2 * (K64 as usize), 999, 0);
        assert_eq!(check_eligibility(fake_view as u64, K64), Err(FallbackReason::ViewMemory));
        crate::inplace::views_gone(fake_view, 2 * (K64 as usize));
        // 6. Unsupported device mode -> DeviceUnsupported
        let _turn = crate::indirect::tests::warp_turn();
        if let Some(device) = create_debug_device(true) {
            assert_eq!(
                try_create_region(&device, SyncMode::Unsupported, alloc.base as u64, K64, K64).err(),
                Some(FallbackReason::DeviceUnsupported)
            );
        }
    }

    #[test]
    fn coalesce_ranges_handles_overlaps_unsorted_and_clamping() {
        assert_eq!(coalesce_ranges(&[], 100), vec![]);
        assert_eq!(
            coalesce_ranges(&[(50, 80), (10, 30), (30, 55), (0, 0), (90, 200), (15, 25)], 100),
            vec![(10, 80), (90, 100)]
        );
        // Same-dword disjoint ranges remain separate in coalesce (UAV barrier handles them in dispatch).
        assert_eq!(coalesce_ranges(&[(2, 3), (0, 1)], 16), vec![(0, 1), (2, 3)]);
    }

    fn verify_same_dword_atomic_concurrency(device: &ID3D12Device, label: &str) {
        let report = probe_device(device);
        assert!(report.debug_errors.is_empty(), "[{label}] probe debug errors: {:?}", report.debug_errors);
        if report.mode != SyncMode::BidirectionalUav || !report.l0_atomics_ok {
            eprintln!("[{label}] L0 UAV atomics unavailable; skipping same-dword atomic test");
            return;
        }
        let gbv = std::env::var_os("VICTORY_D3D12_GBV").is_some_and(|v| v == "1");
        let len: usize = if gbv { 4096 } else { K64 as usize };
        let partial_rounds: u8 = if gbv { 4 } else { 25 };
        let full_rounds: u32 = if gbv { 3 } else { 15 };
        let mut alloc = OwnedAlloc::new(K64 as usize);
        alloc.as_mut_slice().fill(0);
        let synced = try_create_region(device, SyncMode::BidirectionalUav, alloc.base as u64, K64, K64)
            .expect("synced region");
        assert_eq!(synced.mode(), SyncMode::BidirectionalUav);
        let shadow_l1 = make_committed(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            K64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let mut rig = ProbeRig::new(device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());
        let mut pipelines = EnginePipelines::new(
            device,
            SyncMode::BidirectionalUav,
            ATOMIC_PARTIAL,
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        let title_va = synced.title_va();
        let shadow_va = unsafe { shadow_l1.GetGPUVirtualAddress() };
        let before_va = synced.before_va();
        let ranges = [(0, len)];
        let (mut_root, mut_pso) = make_compute_pso(device, TEST_GPU_MUTATE_HLSL).expect("gpu mutate pso");
        let dword_count = (len / 4) as u32;
        let base_addr = alloc.base as usize;

        for (tag, flags) in [
            ("partial-only (discrete)", ATOMIC_PARTIAL),
            ("full+partial (UMA/WARP)", ATOMIC_PARTIAL | ATOMIC_FULL),
        ] {
            pipelines.atomic_flags = flags;
            let pipe = &pipelines;
            alloc.as_mut_slice().fill(0);
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(list, pipe, title_va, 0, shadow_va, 0, before_va, 0, &ranges, SyncOp::Init);
            })
            .unwrap();
            if report.is_software || report.cache_coherent_uma {
                // On coherent adapters (WARP / CacheCoherentUMA APUs), D3D12 `InterlockedCompareExchange`
                // shares the CPU cache hierarchy, so tight concurrent CPU byte-0 CAS + GPU byte-2 CAS
                // across every dword must preserve 100% of both CPU and GPU bytes.
                let started = AtomicBool::new(false);
                let stop = AtomicBool::new(false);
                std::thread::scope(|scope| {
                    let writer = scope.spawn(|| {
                        let mut counter = 1u8;
                        loop {
                            for d in 0..(len / 4) {
                                // SAFETY: aligned 32-bit atomic update to byte 0 of each dword.
                                let slot = unsafe { &*((base_addr + d * 4) as *const AtomicU32) };
                                let mut cur = slot.load(Ordering::Relaxed);
                                loop {
                                    let next = (cur & 0xFFFF_FF00) | u32::from(counter);
                                    match slot.compare_exchange_weak(cur, next, Ordering::Relaxed, Ordering::Relaxed) {
                                        Ok(_) => break,
                                        Err(actual) => cur = actual,
                                    }
                                }
                            }
                            started.store(true, Ordering::Release);
                            if stop.load(Ordering::Relaxed) {
                                return counter;
                            }
                            counter = counter.wrapping_add(1).max(1);
                        }
                    });
                    while !started.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }

                    for round in 1..=partial_rounds {
                        rig.run_timed(|list| unsafe {
                            let args = [dword_count, u32::from(round), 1u32];
                            list.SetComputeRootSignature(&mut_root);
                            list.SetPipelineState(&mut_pso);
                            list.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
                            list.SetComputeRootUnorderedAccessView(1, shadow_va);
                            list.Dispatch(dword_count.div_ceil(256), 1, 1);
                            list.ResourceBarrier(&[uav_barrier()]);
                            record_bidir_ranges(
                                list,
                                pipe,
                                title_va,
                                0,
                                shadow_va,
                                0,
                                before_va,
                                0,
                                &ranges,
                                SyncOp::GpuWriteBackOnly,
                            );
                        })
                        .unwrap();
                        for d in 0..(len / 4) {
                            // SAFETY: aligned atomic load of byte 2 written by GPU.
                            let word = unsafe { &*((base_addr + d * 4) as *const AtomicU32) }.load(Ordering::Relaxed);
                            let got_gpu_byte = ((word >> 16) & 0xFF) as u8;
                            assert_eq!(
                                got_gpu_byte, round,
                                "[{label}/{tag}] same-dword atomic CAS lost GPU byte 2 at dword {d} round {round} (word={word:#010x})"
                            );
                        }
                    }
                    stop.store(true, Ordering::Relaxed);
                    let last_cpu = writer.join().expect("CPU writer joined");
                    for d in 0..(len / 4) {
                        // SAFETY: aligned atomic load after CPU thread joined.
                        let word = unsafe { &*((base_addr + d * 4) as *const AtomicU32) }.load(Ordering::Relaxed);
                        assert_eq!((word & 0xFF) as u8, last_cpu, "[{label}/{tag}] CPU byte 0 clobbered at dword {d}");
                        assert_eq!(((word >> 16) & 0xFF) as u8, partial_rounds, "[{label}/{tag}] GPU byte 2 lost at dword {d}");
                    }
                });
            } else {
                // On discrete non-coherent PCIe GPUs (`CacheCoherentUMA == false`), D3D12 L0 CAS is
                // GPU-L2 scoped (not cross-PCIe coherent with a spinning CPU CAS loop). Verify
                // deterministic same-dword disjoint-byte 3-way merge via partial-dword CAS on HW.
                for round in 1..=partial_rounds.min(5) {
                    let cpu_byte = round.wrapping_mul(17).max(1);
                    for d in 0..(len / 4) {
                        alloc.as_mut_slice()[d * 4] = cpu_byte;
                    }
                    rig.run_timed(|list| unsafe {
                        let args = [dword_count, u32::from(round), 1u32];
                        list.SetComputeRootSignature(&mut_root);
                        list.SetPipelineState(&mut_pso);
                        list.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
                        list.SetComputeRootUnorderedAccessView(1, shadow_va);
                        list.Dispatch(dword_count.div_ceil(256), 1, 1);
                        list.ResourceBarrier(&[uav_barrier()]);
                        record_bidir_ranges(
                            list,
                            pipe,
                            title_va,
                            0,
                            shadow_va,
                            0,
                            before_va,
                            0,
                            &ranges,
                            SyncOp::GpuWriteBackOnly,
                        );
                    })
                    .unwrap();
                    for d in 0..(len / 4) {
                        let word = u32::from_le_bytes(alloc.as_mut_slice()[d * 4..(d + 1) * 4].try_into().unwrap());
                        assert_eq!((word & 0xFF) as u8, cpu_byte, "[{label}/{tag}] HW partial CAS clobbered CPU byte 0 at dword {d}");
                        assert_eq!(((word >> 16) & 0xFF) as u8, round, "[{label}/{tag}] HW partial CAS lost GPU byte 2 at dword {d}");
                    }
                }
            }
        }

        // Full-dword (`gpu_only == 0xFFFFFFFFu`) concurrent CPU+GPU frontier CAS race:
        if report.is_software || report.cache_coherent_uma {
            for round in 1..=full_rounds {
                alloc.as_mut_slice().fill(0);
                rig.run_timed(|list| unsafe {
                    record_bidir_ranges(list, &pipelines, title_va, 0, shadow_va, 0, before_va, 0, &ranges, SyncOp::Init);
                })
                .unwrap();
                let cpu_word = 0x8080_8080u32 | ((round & 0x7F) * 0x0101_0101u32);
                let go = AtomicBool::new(false);
                std::thread::scope(|scope| {
                    let sweeper = scope.spawn(|| {
                        while !go.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                        for d in 0..(len / 4) {
                            // SAFETY: aligned 32-bit atomic store to test buffer.
                            let slot = unsafe { &*((base_addr + d * 4) as *const AtomicU32) };
                            slot.store(cpu_word, Ordering::Relaxed);
                        }
                    });
                    rig.run_timed(|list| unsafe {
                        let args = [dword_count, round, 2u32];
                        list.SetComputeRootSignature(&mut_root);
                        list.SetPipelineState(&mut_pso);
                        list.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
                        list.SetComputeRootUnorderedAccessView(1, shadow_va);
                        list.Dispatch(dword_count.div_ceil(256), 1, 1);
                        list.ResourceBarrier(&[uav_barrier()]);
                        record_bidir_ranges(
                            list,
                            &pipelines,
                            title_va,
                            0,
                            shadow_va,
                            0,
                            before_va,
                            0,
                            &ranges,
                            SyncOp::Bidirectional3Way,
                        );
                        go.store(true, Ordering::Release);
                    })
                    .unwrap();
                    sweeper.join().expect("CPU full-dword sweeper joined");
                });
                for d in 0..(len / 4) {
                    // SAFETY: aligned atomic load after both GPU and CPU sweeper completed.
                    let word = unsafe { &*((base_addr + d * 4) as *const AtomicU32) }.load(Ordering::Relaxed);
                    assert_eq!(
                        word, cpu_word,
                        "[{label}] full-dword L0 CAS clobbered concurrent CPU write at dword {d} round {round} (got {word:#010x}, expected {cpu_word:#010x})"
                    );
                }
            }
        } else {
            // Discrete PCIe GPU (`CacheCoherentUMA == false`): verify full-dword (`gpu_only == 0xFFFFFFFFu`)
            // fast posted Store4 + pre-dispatch CPU win.
            pipelines.atomic_flags = ATOMIC_PARTIAL;
            alloc.as_mut_slice().fill(0);
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(list, &pipelines, title_va, 0, shadow_va, 0, before_va, 0, &ranges, SyncOp::Init);
            })
            .unwrap();
            let cpu_win = 0x89AB_CDEFu32;
            alloc.as_mut_slice()[0..4].copy_from_slice(&cpu_win.to_le_bytes());
            rig.run_timed(|list| unsafe {
                let args = [dword_count, 7u32, 2u32];
                list.SetComputeRootSignature(&mut_root);
                list.SetPipelineState(&mut_pso);
                list.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, shadow_va);
                list.Dispatch(dword_count.div_ceil(256), 1, 1);
                list.ResourceBarrier(&[uav_barrier()]);
                record_bidir_ranges(
                    list,
                    &pipelines,
                    title_va,
                    0,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    &ranges,
                    SyncOp::GpuWriteBackOnly,
                );
            })
            .unwrap();
            let got0 = u32::from_le_bytes(alloc.as_mut_slice()[0..4].try_into().unwrap());
            assert_eq!(got0, cpu_win, "[{label}] CPU write must win full-dword conflict");
            for d in 1..(len / 4) {
                let got = u32::from_le_bytes(alloc.as_mut_slice()[d * 4..(d + 1) * 4].try_into().unwrap());
                assert_eq!(got, 0x0707_0707, "[{label}] full-dword Store4 propagation failed at dword {d}");
            }
        }

        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "[{label}] D3D12 debug errors in same-dword atomic test: {errs:?}");
    }

    #[test]
    fn hw_and_warp_same_dword_concurrent_byte_writes_with_atomics() {
        if let Some(hw) = create_debug_device(false) {
            verify_same_dword_atomic_concurrency(&hw, "HW");
        }
        let _turn = crate::indirect::tests::warp_turn();
        if let Some(warp) = create_debug_device(true) {
            verify_same_dword_atomic_concurrency(&warp, "WARP");
        }
    }

    #[test]
    fn end_to_end_shadow_zero_staging_and_detach_lifecycle() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        const LEN: usize = 2 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        for (i, b) in alloc.as_mut_slice().iter_mut().enumerate() {
            *b = ((i.wrapping_mul(19) + 3) & 0xFF) as u8;
        }
        let start = alloc.base as u64;
        let synced = try_create_region(&device, report.mode, start, LEN as u64, LEN as u64).expect("synced region");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1.clone(), start, LEN as u64, synced);

        // Verify ZERO CPU staging buffer allocated and clean classification!
        assert!(shadow.gpu_synced());
        assert!(!shadow.carded());
        assert!(!shadow.in_place());
        assert!(shadow.card_data().is_null(), "gpu_sync shadow must allocate zero CPU staging memory");

        // 1. Shadow::update queues GPU-direct upload with zero CPU memcpy!
        let bytes_queued = shadow.update(start as usize, LEN);
        assert_eq!(bytes_queued, LEN);
        let (init_ranges, up_ranges) = {
            let dirty = crate::shadow::gpu_sync_dirty();
            let (_, init_r, up_r) = dirty
                .iter()
                .find(|(s, _, _)| Arc::ptr_eq(s, &shadow))
                .expect("queued upload for gpu_sync shadow");
            (init_r.clone(), up_r.clone())
        };
        assert!(!init_ranges.is_empty() || !up_ranges.is_empty());

        let mut rig = ProbeRig::new(&device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());
        let pipelines = EnginePipelines::new(
            &device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        let gs = shadow.gpu_sync_ref().unwrap();
        let title_va = gs.title_va();
        let shadow_va = unsafe { shadow.buffer.GetGPUVirtualAddress() };
        let before_va = gs.before_va();

        // Dispatch queued initial upload via DMA L0->L1 + L1->L1 (`record_init_dma`) followed
        // directly by a compute `Upload3Way` in the same command list (verifying all state
        // transitions under D3D12 Debug Layer + GBV!).
        rig.run_timed(|list| unsafe {
            if !init_ranges.is_empty() && pipelines.l0_copy_to_l1 {
                record_init_dma(
                    list,
                    &gs.title_buffer,
                    gs.title_bias,
                    &shadow.buffer,
                    &gs.before_buffer,
                    &init_ranges,
                    true,
                );
            } else if !init_ranges.is_empty() {
                record_srv_ranges(
                    list, &pipelines, title_va, gs.title_bias, shadow_va, 0, before_va, 0, &init_ranges, SyncOp::Init, SRV_MAX_DELTAS,
                );
                list.ResourceBarrier(&[uav_barrier()]);
            }
            let check_up = if up_ranges.is_empty() { vec![(0, 64)] } else { up_ranges.clone() };
            record_srv_ranges(
                list, &pipelines, title_va, gs.title_bias, shadow_va, 0, before_va, 0, &check_up, SyncOp::Upload3Way, SRV_MAX_DELTAS,
            );
        })
        .unwrap();
        assert_eq!(rig.read_l1(&shadow.buffer, 0, LEN).unwrap(), alloc.as_mut_slice());
        assert_eq!(rig.read_l1(&gs.before_buffer, 0, LEN).unwrap(), alloc.as_mut_slice());

        // 2. Declare pending GPU write (zero CPU snapshot!), simulate GPU write + CPU conflict write!
        shadow.writing(start as usize, LEN);
        assert!(shadow.writes_pending());
        let mut s_mut = alloc.as_mut_slice().to_vec();
        s_mut[100] ^= 0x55; // GPU-only write
        s_mut[101] ^= 0x77; // Conflict: both GPU and CPU write byte 101 -> CPU wins!
        rig.write_l1(&shadow.buffer, 0, &s_mut).unwrap();
        let cpu_win = alloc.as_mut_slice()[101] ^ 0x33;
        alloc.as_mut_slice()[101] = cpu_win;
        let cpu_only = alloc.as_mut_slice()[102] ^ 0x99;
        alloc.as_mut_slice()[102] = cpu_only;

        // 3. Bidirectional 3-way sync resolves all three buffers byte-for-byte.
        if report.mode == SyncMode::BidirectionalUav {
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    title_va,
                    gs.title_bias,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    &[(0, LEN)],
                    SyncOp::Bidirectional3Way,
                );
            })
            .unwrap();
        } else {
            run_srv_bidir_drain(
                &mut rig,
                &pipelines,
                gs,
                &shadow.buffer,
                &[(0, LEN)],
                (alloc.base, LEN, SRV_MAX_DELTAS, SyncOp::Bidirectional3Way),
            );
        }
        shadow.copy_back_done(&[(start as usize, start as usize + LEN)]);
        assert!(!shadow.writes_pending());
        assert_eq!(alloc.as_mut_slice()[100], s_mut[100], "GPU-only write propagated to title RAM");
        assert_eq!(alloc.as_mut_slice()[101], cpu_win, "CPU/title write won conflict on byte 101");
        assert_eq!(alloc.as_mut_slice()[102], cpu_only, "CPU-only write preserved");
        assert_eq!(rig.read_l1(&shadow.buffer, 0, LEN).unwrap(), alloc.as_mut_slice());
        assert_eq!(rig.read_l1(&gs.before_buffer, 0, LEN).unwrap(), alloc.as_mut_slice());

        // 3b. Placed-copy snapshot (`stage_placed_snapshots`): title resets counter [200..204)
        // to 0, then CPU races and updates title RAM [200..204) to 99 before pre-list runs;
        // copying the exact `placed` snapshot (`0`) into `before_buffer` preserves CPU's 99!
        // Meanwhile, a non-racing placed copy at [208..212) (`0` -> shader increments to `42`)
        // lands `42` cleanly without leaving stale `pending_uploads`.
        let pre_shadow_200 = rig.read_l1(&shadow.buffer, 200, 4).unwrap();
        let pre_shadow_208 = rig.read_l1(&shadow.buffer, 208, 4).unwrap();
        alloc.as_mut_slice()[200..204].copy_from_slice(&0u32.to_le_bytes());
        alloc.as_mut_slice()[208..212].copy_from_slice(&0u32.to_le_bytes());
        let placed_batch = vec![
            PlacedCopy {
                shadow: shadow.clone(),
                at: start as usize + 200,
                bytes: 0u32.to_le_bytes().to_vec(),
                pre_bytes: Some(pre_shadow_200.clone()),
            },
            PlacedCopy {
                shadow: shadow.clone(),
                at: start as usize + 208,
                bytes: 0u32.to_le_bytes().to_vec(),
                pre_bytes: Some(pre_shadow_208.clone()),
            },
        ];
        shadow.placed_copy(start as usize + 200, &0u32.to_le_bytes());
        shadow.placed_copy(start as usize + 208, &0u32.to_le_bytes());
        assert!(!gs.has_uploads(), "placed_copy must not queue stale uploads for the next submission");
        let (placed_upload, staged_copies) =
            stage_placed_snapshots(&device, &placed_batch).expect("staged placed snapshots");
        // Simulate CPU racing on [200..204) AFTER `Placed::prepare` snapshot, BEFORE pre-list executes!
        alloc.as_mut_slice()[200..204].copy_from_slice(&99u32.to_le_bytes());
        rig.run_timed(|list| unsafe {
            for (_, staged) in &staged_copies {
                for &(rel_lo, up_off, copy_len) in &staged.after_copies {
                    list.CopyBufferRegion(&gs.before_buffer, rel_lo as u64, &placed_upload, up_off, copy_len as u64);
                }
            }
        })
        .unwrap();
        assert_eq!(
            rig.read_l1(&shadow.buffer, 200, 4).unwrap(),
            pre_shadow_200,
            "placed snapshot copy must NOT overwrite shadow.buffer ahead of in-list draws"
        );
        assert_eq!(
            rig.read_l1(&shadow.buffer, 208, 4).unwrap(),
            pre_shadow_208,
            "placed snapshot copy must NOT overwrite shadow.buffer ahead of in-list draws"
        );
        assert_eq!(rig.read_l1(&gs.before_buffer, 200, 4).unwrap(), 0u32.to_le_bytes());
        assert_eq!(rig.read_l1(&gs.before_buffer, 208, 4).unwrap(), 0u32.to_le_bytes());
        // Simulate in-list placed copies (`0` at 200, `0 -> 42` at 208), then Bidirectional3Way.
        rig.write_l1(&shadow.buffer, 200, &0u32.to_le_bytes()).unwrap();
        rig.write_l1(&shadow.buffer, 208, &42u32.to_le_bytes()).unwrap();
        if report.mode == SyncMode::BidirectionalUav {
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    title_va,
                    gs.title_bias,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    &[(200, 212)],
                    SyncOp::Bidirectional3Way,
                );
            })
            .unwrap();
        } else {
            run_srv_bidir_drain(
                &mut rig,
                &pipelines,
                gs,
                &shadow.buffer,
                &[(200, 212)],
                (alloc.base, LEN, SRV_MAX_DELTAS, SyncOp::Bidirectional3Way),
            );
        }
        assert_eq!(
            u32::from_le_bytes(alloc.as_mut_slice()[200..204].try_into().unwrap()),
            99,
            "TOCTOU CPU write (99) after Placed::prepare snapshot must win over stale placed 0"
        );
        assert_eq!(
            u32::from_le_bytes(rig.read_l1(&shadow.buffer, 200, 4).unwrap().try_into().unwrap()),
            99,
            "TOCTOU CPU write (99) must also propagate to shadow.buffer"
        );
        assert_eq!(
            u32::from_le_bytes(alloc.as_mut_slice()[208..212].try_into().unwrap()),
            42,
            "GPU increment (42) after non-racing placed_copy counter reset must land cleanly"
        );
        drop((placed_batch, staged_copies, placed_upload));

        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "end-to-end shadow lifecycle debug errors: {errs:?}");

        // 4. Drop shadow & rig & verify OS unlocks the opened pages so VirtualFree(MEM_DECOMMIT) succeeds!
        assert_eq!(Arc::strong_count(&shadow), 1, "no leaked Arc<Shadow> references");
        drop(rig);
        drop(shadow);
        drop(shadow_l1);
        let decommitted = (0..500).any(|_| {
            // SAFETY: test-owned allocation, heap released.
            let ok = unsafe { mem::VirtualFree(alloc.base.cast(), K64 as usize, mem::MEM_DECOMMIT) != 0 };
            if !ok {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            ok
        });
        assert!(decommitted, "dropping gpu_sync Shadow must release OpenExistingHeapFromAddress1 page lock");
    }

    #[test]
    fn uninitialized_shadow_writing_auto_inits_before_bidir_sync() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        const LEN: usize = 2 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        alloc.as_mut_slice().fill(0x42);
        let start = alloc.base as u64;
        let synced = try_create_region(&device, report.mode, start, LEN as u64, LEN as u64).expect("synced region");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1, start, LEN as u64, synced);
        let gs = shadow.gpu_sync_ref().unwrap();

        // Simulate `Shadow::writing` on a fresh shadow with ZERO prior `refresh`/`update`.
        // `before_buffer` starts zeroed while title RAM holds non-zero `0x42`.
        assert!(!gs.is_page_initialized(0));
        assert!(!gs.is_page_initialized(K64 as usize));
        shadow.writing(start as usize, 4096);
        let auto_init = gs.uninitialized_page_ranges(&[(0, 4096)], LEN);
        assert_eq!(auto_init, vec![(0, K64 as usize)]);
        assert_eq!(subtract_ranges(&[(0, 8192), (K64 as usize, LEN)], &auto_init), vec![(K64 as usize, LEN)]);

        let mut rig = ProbeRig::new(&device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());
        let pipelines = EnginePipelines::new(
            &device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        // Pre-list auto-Init runs on page 0 before title list executes.
        rig.run_timed(|list| unsafe {
            if pipelines.l0_copy_to_l1 {
                record_init_dma(list, &gs.title_buffer, gs.title_bias, &shadow.buffer, &gs.before_buffer, &auto_init, false);
            } else {
                record_srv_ranges(
                    list,
                    &pipelines,
                    gs.title_va(),
                    gs.title_bias,
                    shadow.buffer.GetGPUVirtualAddress(),
                    0,
                    gs.before_va(),
                    0,
                    &auto_init,
                    SyncOp::Init,
                    SRV_MAX_DELTAS,
                );
            }
        })
        .unwrap();
        gs.mark_pages_initialized(&auto_init, LEN);
        assert!(gs.is_page_initialized(0));
        assert!(!gs.is_page_initialized(K64 as usize));
        assert!(gs.uninitialized_page_ranges(&[(0, 4096)], LEN).is_empty());

        // Title shader writes `[100..104) = 0xDEAD_BEEF`, then post-list `Bidirectional3Way` runs.
        rig.write_l1(&shadow.buffer, 100, &0xDEAD_BEEFu32.to_le_bytes()).unwrap();
        if report.mode == SyncMode::BidirectionalUav {
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    gs.title_va(),
                    gs.title_bias,
                    shadow.buffer.GetGPUVirtualAddress(),
                    0,
                    gs.before_va(),
                    0,
                    &[(0, 4096)],
                    SyncOp::Bidirectional3Way,
                );
            })
            .unwrap();
        } else {
            run_srv_bidir_drain(
                &mut rig,
                &pipelines,
                gs,
                &shadow.buffer,
                &[(0, 4096)],
                (alloc.base, LEN, SRV_MAX_DELTAS, SyncOp::GpuWriteBackOnly),
            );
        }
        assert_eq!(
            u32::from_le_bytes(alloc.as_mut_slice()[100..104].try_into().unwrap()),
            0xDEAD_BEEF,
            "GPU write on freshly written shadow must NOT be discarded by zeroed before_buffer"
        );
        assert!(alloc.as_mut_slice()[0..100].iter().all(|&b| b == 0x42));
        assert!(alloc.as_mut_slice()[104..4096].iter().all(|&b| b == 0x42));
        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "auto-init test debug errors: {errs:?}");
    }

    #[test]
    fn srv_delta_apply_preserves_concurrent_cpu_writes_and_guards_readonly() {
        const LEN: usize = 2 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        alloc.as_mut_slice().fill(0x11);
        // Simulate baseline `[0x11, 0x11, 0x11, 0x11]` at offset 0 and at page 1 (`K64`).
        // Suppose CPU concurrently changed byte 1 of offset 0 from `0x11 -> 0xCC` AFTER
        // `SYNC_SRV_HLSL` read `base_t = 0x1111_1111`, while GPU emitted `0x4433_2299` across all 4 bytes.
        alloc.as_mut_slice()[1] = 0xCC;

        // Protect page 1 as `PAGE_READONLY` so applying a delta there MUST NOT crash (safecopy guards it).
        // SAFETY: test-owned allocation, restored before drop.
        let mut old_prot = 0u32;
        unsafe {
            assert_ne!(
                mem::VirtualProtect(alloc.base.add(K64 as usize).cast(), K64 as usize, mem::PAGE_READONLY, &mut old_prot),
                0
            );
        }

        // Build a synthetic delta readback buffer: 2 records (offset 0 and offset K64).
        let mut rb = [0u32; 4 + 2 * 4];
        rb[0] = 2; // raw_count
        rb[1] = 0; // overflow
        // Record 0: w_off = 0, gpu_word = 0x4433_2299, mask = 0xFFFF_FFFF, base_t = 0x1111_1111
        rb[4] = 0;
        rb[5] = 0x4433_2299;
        rb[6] = 0xFFFF_FFFF;
        rb[7] = 0x1111_1111;
        // Record 1: w_off = K64 (PAGE_READONLY), gpu_word = 0x5566_7788, mask = 0xFFFF_FFFF, base_t = 0x1111_1111
        rb[8] = K64 as u32;
        rb[9] = 0x5566_7788;
        rb[10] = 0xFFFF_FFFF;
        rb[11] = 0x1111_1111;

        // SAFETY: synthetic readback buffer and test allocation.
        let (applied, overflow) = unsafe { apply_srv_deltas_to_title(rb.as_ptr().cast::<u8>(), 64, alloc.base, LEN) };
        assert_eq!(applied, 2);
        assert!(!overflow);
        // Byte 1 MUST keep CPU's concurrent `0xCC`; bytes 0, 2, 3 get GPU's `0x99`, `0x33`, `0x44`!
        assert_eq!(&alloc.as_mut_slice()[0..4], &[0x99, 0xCC, 0x33, 0x44]);
        // Page 1 stayed `0x11` and did NOT AV-crash!
        assert_eq!(&alloc.as_mut_slice()[K64 as usize..K64 as usize + 4], &[0x11, 0x11, 0x11, 0x11]);

        // SAFETY: restore RW protection before drop.
        unsafe {
            let _ = mem::VirtualProtect(alloc.base.add(K64 as usize).cast(), K64 as usize, mem::PAGE_READWRITE, &mut old_prot);
        }

        // Verify `SRV_IN_FLIGHT` / `needs_settle()` lifecycle.
        assert!(!needs_settle());
        SRV_IN_FLIGHT.fetch_add(1, Ordering::Release);
        assert!(needs_settle());
        let plan = SubmissionPlan {
            pre_list: None,
            post_list: None,
            lists: Vec::new(),
            _kept: Vec::new(),
            _transient: Vec::new(),
            pending_init: Vec::new(),
            srv_ranges: Vec::new(),
            ts_slot: None,
            dirty_slot: None,
            submitted: false,
            counted_srv_flight: true,
        };
        drop(plan);
        assert!(!needs_settle());
    }

    #[test]
    fn gpu_sync_let_go_detaches_on_64k_rounded_boundary_frees() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        // Allocate 3 x 64 KiB and place a buffer at [base + 0x8000, base + 0x18000),
        // which locks [base, base + 0x20000) under `OpenExistingHeapFromAddress1`!
        let alloc = OwnedAlloc::new(3 * (K64 as usize));
        let base = alloc.base as u64;
        let buf_start = base + 0x8000;
        let buf_len = K64;

        // Also verify `check_eligibility` rejects a view overlapping the 64 KiB-rounded padding!
        let fake_view_base = 0x7ff3_0000_0000usize;
        crate::inplace::title_memory_view(fake_view_base, K64 as usize, 998, 0);
        // Buffer starts at +0x8000 past the view's start (same 64 KiB page) and ends at +0x18000:
        assert_eq!(check_eligibility((fake_view_base + 0x8000) as u64, K64), Err(FallbackReason::ViewMemory));
        // View only in the trailing 64 KiB padding page [fake_view_base + K64, fake_view_base + 2*K64):
        crate::inplace::views_gone(fake_view_base, K64 as usize);
        crate::inplace::title_memory_view(fake_view_base + K64 as usize, K64 as usize, 997, 0);
        assert_eq!(check_eligibility((fake_view_base + 0x8000) as u64, K64), Err(FallbackReason::ViewMemory));
        crate::inplace::views_gone(fake_view_base + K64 as usize, K64 as usize);

        for (case, free_at, free_len, resource_id) in [
            ("leading half-page", base, 0x8000u64, 0xA11E_0001usize),
            ("trailing half-page", base + 0x1_8000, 0x8000u64, 0xA11E_0002usize),
        ] {
            let synced = try_create_region(&device, report.mode, buf_start, buf_len, buf_len).expect("synced region");
            let shadow_l1 = make_committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                buf_len,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
            .expect("shadow_l1");
            let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1, buf_start, buf_len, synced);
            crate::shadow::register_gpu_sync_for_test(resource_id, shadow);

            // Disjoint next 64 KiB page [base + 0x20000, base + 0x30000) must NOT detach it.
            assert_eq!(crate::shadow::gpu_sync_let_go(base + 2 * K64, K64), 0, "{case}: disjoint free");
            // Freeing only the leading or trailing 32 KiB padding half-page MUST detach it!
            assert_eq!(
                crate::shadow::gpu_sync_let_go(free_at, free_len),
                1,
                "{case}: 64 KiB-rounded lock overlap must detach"
            );
            crate::shadow::unregister(resource_id, buf_start);
        }
    }

    #[test]
    fn spliced_pre_list_and_title_list_single_execute_gbv_clean() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        const LEN: usize = 2 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        for (i, b) in alloc.as_mut_slice().iter_mut().enumerate() {
            *b = ((i.wrapping_mul(29) + 5) & 0xFF) as u8;
        }
        let start = alloc.base as u64;
        let synced = try_create_region(&device, report.mode, start, LEN as u64, LEN as u64).expect("synced");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1, start, LEN as u64, synced);
        let gs = shadow.gpu_sync_ref().unwrap();
        let pipelines = EnginePipelines::new(
            &device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        let mut rig = ProbeRig::new(&device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());

        // Stage a placed snapshot at [256..260), plus Init on page 0 and Upload3Way on page 1.
        let pre_256 = alloc.as_mut_slice()[256..260].to_vec();
        let placed_batch = vec![PlacedCopy {
            shadow: shadow.clone(),
            at: start as usize + 256,
            bytes: 0xCAFE_BABEu32.to_le_bytes().to_vec(),
            pre_bytes: Some(pre_256),
        }];
        let (placed_upload, staged) = stage_placed_snapshots(&device, &placed_batch).expect("staged");
        let staged_placed = &staged[0].1;

        // Build TWO separate command lists executed in ONE single `ExecuteCommandLists(&[pre, title])`:
        // - `pre_list` runs `record_pre_shadow_to_common` (DMA Init + Upload3Way + placed Copy + end -> COMMON).
        // - `title_list` immediately promotes `shadow.buffer` from COMMON to COPY_DEST (copying 4 bytes
        //   from a scratch upload buffer into offset 512) and UAV (dispatching `TEST_GPU_MUTATE_HLSL`).
        let title_patch = make_committed(
            &device,
            D3D12_HEAP_TYPE_UPLOAD,
            16,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .expect("title_patch");
        let (mut_root, mut_pso) = make_compute_pso(&device, TEST_GPU_MUTATE_HLSL).expect("mut pso");
        unsafe {
            let mut ptr = std::ptr::null_mut();
            title_patch.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut ptr)).unwrap();
            ptr.cast::<u32>().write(0x1234_5678);
            title_patch.Unmap(0, None);

            let alloc_pre: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list_pre: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc_pre, None).unwrap();
            let alloc_title: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list_title: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc_title, None).unwrap();

            record_pre_shadow_to_common(
                &list_pre,
                &pipelines,
                &shadow,
                gs,
                &[(0, 256), (260, K64 as usize)],
                &[(K64 as usize, LEN)],
                &staged_placed.pre_copies,
                &staged_placed.after_copies,
                Some(&placed_upload),
            );
            list_pre.Close().unwrap();

            // `title_list` implicitly promotes `shadow.buffer` from COMMON -> COPY_DEST, then -> UAV!
            list_title.CopyBufferRegion(&shadow.buffer, 512, &title_patch, 0, 4);
            list_title.ResourceBarrier(&[transition(
                &shadow.buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            let args = [1u32, 9u32, 2u32];
            list_title.SetComputeRootSignature(&mut_root);
            list_title.SetPipelineState(&mut_pso);
            list_title.SetComputeRoot32BitConstants(0, 3, args.as_ptr().cast(), 0);
            list_title.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
            list_title.Dispatch(1, 1, 1);
            list_title.Close().unwrap();

            let cmd_pre: ID3D12CommandList = list_pre.cast().unwrap();
            let cmd_title: ID3D12CommandList = list_title.cast().unwrap();
            rig.queue.ExecuteCommandLists(&[Some(cmd_pre), Some(cmd_title)]);
            let target = rig.fence_val + 1;
            rig.queue.Signal(&rig.fence, target).unwrap();
            rig.fence_val = target;
            rig.fence.SetEventOnCompletion(target, rig.event).unwrap();
            assert_eq!(WaitForSingleObject(rig.event, 15_000), windows::Win32::Foundation::WAIT_OBJECT_0);
        }
        assert_eq!(rig.read_l1(&shadow.buffer, 0, 4).unwrap(), 0x0909_0909u32.to_le_bytes());
        assert_eq!(rig.read_l1(&shadow.buffer, 512, 4).unwrap(), 0x1234_5678u32.to_le_bytes());
        assert_eq!(rig.read_l1(&gs.before_buffer, 256, 4).unwrap(), 0xCAFE_BABEu32.to_le_bytes());
        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "spliced [pre_list, title_list] ExecuteCommandLists GBV errors: {errs:?}");
    }

    #[test]
    fn telemetry_counters_timestamp_pool_and_mode_override() {
        // 1. Mode override parser + mode selector + discrete vs UMA atomic gating.
        assert_eq!(parse_mode_override(None), ModeOverride::Auto);
        assert_eq!(parse_mode_override(Some("auto")), ModeOverride::Auto);
        assert_eq!(parse_mode_override(Some("BIDIR")), ModeOverride::Bidir);
        assert_eq!(parse_mode_override(Some("srv")), ModeOverride::Srv);

        let discrete_report = ProbeReport {
            adapter_name: "Discrete".to_owned(),
            is_software: false,
            cache_coherent_uma: false,
            has_device13: true,
            l0_open_ok: true,
            l0_srv_read_ok: true,
            l0_uav_write_ok: true,
            l0_copy_to_l1_ok: true,
            l0_copy_from_l1_ok: true,
            l0_atomics_ok: true,
            l0_heap_flags: 0x421,
            mode: SyncMode::BidirectionalUav,
            debug_errors: Vec::new(),
            notes: Vec::new(),
        };
        assert_eq!(select_mode(&discrete_report, ModeOverride::Auto), SyncMode::BidirectionalUav);
        assert_eq!(select_mode(&discrete_report, ModeOverride::Srv), SyncMode::SrvDirectUpload);
        assert_eq!(select_mode(&discrete_report, ModeOverride::Bidir), SyncMode::BidirectionalUav);
        // Discrete PCIe (`cache_coherent_uma == false`) disables full-dword CAS (`ATOMIC_FULL`) while
        // keeping partial-dword CAS (`ATOMIC_PARTIAL`) so sibling CPU bytes are never clobbered!
        assert_eq!(compute_atomic_flags(&discrete_report, false), ATOMIC_PARTIAL);
        assert_eq!(compute_atomic_flags(&discrete_report, true), ATOMIC_PARTIAL | ATOMIC_FULL);

        let mut uma_report = discrete_report.clone();
        uma_report.cache_coherent_uma = true;
        assert_eq!(compute_atomic_flags(&uma_report, false), ATOMIC_PARTIAL | ATOMIC_FULL);
        let mut no_atomics = discrete_report.clone();
        no_atomics.l0_atomics_ok = false;
        assert_eq!(select_mode(&no_atomics, ModeOverride::Auto), SyncMode::SrvDirectUpload);
        assert_eq!(select_mode(&no_atomics, ModeOverride::Bidir), SyncMode::Unsupported);
        assert_eq!(compute_atomic_flags(&no_atomics, true), 0);
        let mut srv_only = discrete_report;
        srv_only.l0_uav_write_ok = false;
        srv_only.mode = SyncMode::SrvDirectUpload;
        assert_eq!(select_mode(&srv_only, ModeOverride::Bidir), SyncMode::Unsupported);
        assert_eq!(select_mode(&srv_only, ModeOverride::Srv), SyncMode::SrvDirectUpload);

        // 2. Zero-frequency guard + live non-stalling `TimestampPool` + `eligible_bytes` / `fence_wait_us`.
        assert_eq!(ticks_to_us(123_456, 0), 0);
        assert_eq!(ticks_to_us(1_000_000, 1_000_000), 1_000_000);

        let before_wait = FENCE_WAIT_US.load(Ordering::Relaxed);
        note_fence_wait_us(42);
        assert_eq!(FENCE_WAIT_US.load(Ordering::Relaxed) - before_wait, 42);

        let before_elig = eligible_bytes();
        note_attached(K64);
        assert_eq!(eligible_bytes(), before_elig + K64);
        note_dropped(K64);
        assert_eq!(eligible_bytes(), before_elig);

        if let Some(device) = create_debug_device(false) {
            let mut rig = ProbeRig::new(&device).expect("rig");
            let pool = TimestampPool::new(&device, &rig.queue).expect("timestamp pool");
            let slot = pool.alloc_slot().expect("slot");
            rig.run_timed(|list| unsafe {
                pool.begin_query(list, slot, false);
                pool.end_and_resolve(list, slot, false);
                pool.begin_query(list, slot, true);
                pool.end_and_resolve(list, slot, true);
            })
            .unwrap();
            let _us = pool.collect_and_free(slot, true, true);
            assert_eq!(pool.free_slots.lock().unwrap().len(), TS_SLOTS as usize);
            let errs = drain_debug_errors(rig.info_queue.as_ref());
            assert!(errs.is_empty(), "TimestampPool debug errors: {errs:?}");
        }
    }

    #[test]
    fn placed_copy_refresh_preserves_pre_placed_bytes_for_prior_draws_and_post_baseline() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        const LEN: usize = 2 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        alloc.as_mut_slice().fill(0x11);
        // Page 0 (`0..64 KiB`): initialized (`8` at `[200..204)`), then CPU updates `[256] = 1`
        // (same 4 KiB page!) and `[200..204]` `7 -> 8` before `Placed::prepare(true)` writes `0`
        // (and a second overlapping `Placed` writes `5` at `[200..204)`).
        // Page 1 (`64 KiB..128 KiB`): UNINITIALIZED, holds `0x7777_7777` at `[K64+200..K64+204)`
        // before `Placed::prepare(true)` writes `0` there.
        alloc.as_mut_slice()[200..204].copy_from_slice(&7u32.to_le_bytes());
        alloc.as_mut_slice()[K64 as usize + 200..K64 as usize + 204].copy_from_slice(&0x7777_7777u32.to_le_bytes());

        let start = alloc.base as u64;
        let synced = try_create_region(&device, report.mode, start, LEN as u64, LEN as u64).expect("synced");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1, start, LEN as u64, synced);
        let gs = shadow.gpu_sync_ref().unwrap();
        let pipelines = EnginePipelines::new(
            &device,
            report.mode,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        let mut rig = ProbeRig::new(&device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());

        // Initialize ONLY page 0 (`[0..K64)`); leave page 1 uninitialized!
        rig.run_timed(|list| unsafe {
            record_pre_shadow_to_common(list, &pipelines, &shadow, gs, &[(0, K64 as usize)], &[], &[], &[], None);
        })
        .unwrap();
        gs.mark_pages_initialized(&[(0, K64 as usize)], LEN);

        // Now CPU modifies page 0 `[200..204]` `7 -> 8` and `[256] = 0x5A`, `Refresh` queues `[0..4096)`.
        alloc.as_mut_slice()[200..204].copy_from_slice(&8u32.to_le_bytes());
        alloc.as_mut_slice()[256] = 0x5A;
        let orig_up = vec![(0usize, 4096usize)];

        // Capture `pre_bytes` right BEFORE `Placed::prepare(true)` overwrites live title RAM:
        // Placed #1 (page 0 `[200..204)`): `8 -> 0`
        // Placed #2 (page 0 `[200..204)`): `0 -> 5` (double overlapping Placed!)
        // Placed #3 (uninitialized page 1 `[K64+200..K64+204)`): `0x7777_7777 -> 0`
        let p1_pre = alloc.as_mut_slice()[200..204].to_vec();
        alloc.as_mut_slice()[200..204].copy_from_slice(&0u32.to_le_bytes());
        let p2_pre = alloc.as_mut_slice()[200..204].to_vec();
        alloc.as_mut_slice()[200..204].copy_from_slice(&5u32.to_le_bytes());
        let p3_pre = alloc.as_mut_slice()[K64 as usize + 200..K64 as usize + 204].to_vec();
        alloc.as_mut_slice()[K64 as usize + 200..K64 as usize + 204].copy_from_slice(&0u32.to_le_bytes());

        let placed_batch = vec![
            PlacedCopy {
                shadow: shadow.clone(),
                at: start as usize + 200,
                bytes: 0u32.to_le_bytes().to_vec(),
                pre_bytes: Some(p1_pre),
            },
            PlacedCopy {
                shadow: shadow.clone(),
                at: start as usize + 200,
                bytes: 5u32.to_le_bytes().to_vec(),
                pre_bytes: Some(p2_pre),
            },
            PlacedCopy {
                shadow: shadow.clone(),
                at: start as usize + K64 as usize + 200,
                bytes: 0u32.to_le_bytes().to_vec(),
                pre_bytes: Some(p3_pre),
            },
        ];
        let (placed_upload, staged) = stage_placed_snapshots(&device, &placed_batch).expect("staged");
        let staged_placed = &staged[0].1;

        // Apply the exact `prepare_submission` range math:
        let placed_spans = coalesce_ranges(
            &staged_placed.after_copies.iter().map(|&(lo, _, n)| (lo, lo + n)).collect::<Vec<_>>(),
            LEN,
        );
        let mut all_touched = orig_up.clone();
        all_touched.extend_from_slice(&placed_spans);
        let full_init = gs.uninitialized_page_ranges(&all_touched, LEN);
        assert_eq!(full_init, vec![(K64 as usize, LEN)]);
        let init_ranges = subtract_ranges(&full_init, &placed_spans);
        let up_ranges = subtract_ranges(&subtract_ranges(&orig_up, &full_init), &placed_spans);
        let mut dirty_mask = full_init.clone();
        dirty_mask.extend_from_slice(&orig_up);
        let dirty_mask = coalesce_ranges(&dirty_mask, LEN);
        let pre_copies = clip_copies_to_ranges(&staged_placed.pre_copies, &dirty_mask);

        // Execute `pre_list` (`record_pre_shadow_to_common`):
        rig.run_timed(|list| unsafe {
            record_pre_shadow_to_common(
                list,
                &pipelines,
                &shadow,
                gs,
                &init_ranges,
                &up_ranges,
                &pre_copies,
                &staged_placed.after_copies,
                Some(&placed_upload),
            );
        })
        .unwrap();

        // Draw 0 (BEFORE in-list CopyBufferRegion):
        // - page 0 `[200..204]` MUST still see `8` (earliest pre-Placed value, NOT `0` or `5`!)
        // - page 0 `[256]` MUST see CPU's updated `0x5A`
        // - page 1 `[K64+200..K64+204]` MUST see `0x7777_7777` (pre-Placed value, NOT `0`!)
        // Meanwhile `before_buffer` MUST hold the final post-Placed values (`5` on page 0, `0` on page 1)!
        assert_eq!(rig.read_l1(&shadow.buffer, 200, 4).unwrap(), 8u32.to_le_bytes(), "Draw 0 on page 0 must read pre-Placed 8");
        assert_eq!(rig.read_l1(&shadow.buffer, 256, 1).unwrap(), &[0x5A], "Draw 0 must see sibling CPU byte 0x5A");
        assert_eq!(
            rig.read_l1(&shadow.buffer, K64 + 200, 4).unwrap(),
            0x7777_7777u32.to_le_bytes(),
            "Draw 0 on uninitialized page 1 must read pre-Placed 0x7777_7777"
        );
        assert_eq!(rig.read_l1(&gs.before_buffer, 200, 4).unwrap(), 5u32.to_le_bytes(), "before_buffer holds final Placed 5");
        assert_eq!(
            rig.read_l1(&gs.before_buffer, K64 + 200, 4).unwrap(),
            0u32.to_le_bytes(),
            "before_buffer on page 1 holds Placed 0"
        );

        // Simulate in-list CopyBufferRegion (`5` on page 0, `0` on page 1) + Draw 1 incrementing page 1 to `42`.
        rig.write_l1(&shadow.buffer, 200, &5u32.to_le_bytes()).unwrap();
        rig.write_l1(&shadow.buffer, K64 + 200, &42u32.to_le_bytes()).unwrap();
        let sync_ranges = [(0, 4096), (K64 as usize, K64 as usize + 4096)];
        if report.mode == SyncMode::BidirectionalUav {
            rig.run_timed(|list| unsafe {
                record_bidir_ranges(
                    list,
                    &pipelines,
                    gs.title_va(),
                    gs.title_bias,
                    shadow.buffer.GetGPUVirtualAddress(),
                    0,
                    gs.before_va(),
                    0,
                    &sync_ranges,
                    SyncOp::GpuWriteBackOnly,
                );
            })
            .unwrap();
        } else {
            run_srv_bidir_drain(
                &mut rig,
                &pipelines,
                gs,
                &shadow.buffer,
                &sync_ranges,
                (alloc.base, LEN, SRV_MAX_DELTAS, SyncOp::GpuWriteBackOnly),
            );
        }
        assert_eq!(u32::from_le_bytes(alloc.as_mut_slice()[200..204].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(alloc.as_mut_slice()[K64 as usize + 200..K64 as usize + 204].try_into().unwrap()), 42);
        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "placed+refresh temporal test debug errors: {errs:?}");
    }

    #[test]
    fn private_virtual_free_detaches_gpu_sync_and_unlocks_heap() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode == SyncMode::Unsupported {
            return;
        }
        const LEN: usize = 2 * (K64 as usize);
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: test-owned VirtualAlloc block freed via `VirtualFree(MEM_RELEASE)` below.
        let base = unsafe {
            mem::VirtualAlloc(
                std::ptr::null(),
                LEN,
                mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
                mem::PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0, "VirtualAlloc failed");

        let synced = try_create_region(&device, report.mode, base as u64, LEN as u64, LEN as u64).expect("synced");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let shadow = crate::shadow::new_gpu_sync_for_test(shadow_l1, base as u64, LEN as u64, synced);
        let resource_id = 0xA11E_F00Dusize;
        crate::shadow::register_gpu_sync_for_test(resource_id, shadow);

        let before_detached = STATS[7].load(Ordering::Relaxed);
        // Simulate exact `xmem.rs` (`XMemFreeDefault` / `VirtualFree(base, 0, MEM_RELEASE)`) sequence:
        // 1. Detach & settle BEFORE acquiring `title_memory_remapping()` (`MAPPING.write()`)
        // 2. Acquire `title_memory_remapping()` + call `title_memory_freed(base, 0)`
        // 3. Call `VirtualFree(base, 0, MEM_RELEASE)` and verify immediate success!
        crate::gpu_sync_memory_freeing(base, 0);
        assert!(
            STATS[7].load(Ordering::Relaxed) > before_detached,
            "gpu_sync_memory_freeing(base, 0) must resolve allocation length and detach gpu_sync shadow"
        );
        {
            let _changing = crate::watch::title_memory_remapping();
            assert!(crate::watch::is_changing());
            crate::watch::title_memory_freed(base, 0);
        }
        crate::shadow::unregister(resource_id, base as u64);

        // `OpenExistingHeapFromAddress1` handle MUST already be dropped synchronously so
        // `VirtualFree(base, 0, MEM_RELEASE)` succeeds immediately!
        // SAFETY: `base` is our test-owned allocation, now unlocked.
        let freed = unsafe { mem::VirtualFree(base as *mut _, 0, mem::MEM_RELEASE) };
        assert_ne!(freed, 0, "VirtualFree(MEM_RELEASE) must succeed immediately after gpu_sync_memory_freeing(base, 0)");
    }

    #[test]
    fn bidir_dirty_page_pool_notifies_only_gpu_written_pages() {
        let Some(device) = create_debug_device(false) else {
            return;
        };
        let report = probe_device(&device);
        assert!(report.debug_errors.is_empty(), "probe debug errors: {:?}", report.debug_errors);
        if report.mode != SyncMode::BidirectionalUav {
            return;
        }
        const LEN: usize = 3 * (K64 as usize);
        let mut alloc = OwnedAlloc::new(LEN);
        alloc.as_mut_slice().fill(0x22);
        let start = alloc.base as u64;
        let synced = try_create_region(&device, SyncMode::BidirectionalUav, start, LEN as u64, LEN as u64).expect("synced");
        let shadow_l1 = make_committed(
            &device,
            D3D12_HEAP_TYPE_DEFAULT,
            LEN as u64,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
        .expect("shadow_l1");
        let pipelines = EnginePipelines::new(
            &device,
            SyncMode::BidirectionalUav,
            compute_atomic_flags(&report, false),
            report.l0_copy_to_l1_ok,
        )
        .expect("pipelines");
        let pool = DirtyPagePool::new(&device).expect("dirty pool");
        let mut rig = ProbeRig::new(&device).expect("rig");
        let _ = drain_debug_errors(rig.info_queue.as_ref());

        let title_va = synced.title_va();
        let shadow_va = unsafe { shadow_l1.GetGPUVirtualAddress() };
        let before_va = synced.before_va();

        // Init all 3 x 64 KiB pages, then GPU mutates ONLY page 1 (`[K64 + 64 .. K64 + 68)`).
        rig.run_timed(|list| unsafe {
            record_bidir_ranges(list, &pipelines, title_va, 0, shadow_va, 0, before_va, 0, &[(0, LEN)], SyncOp::Init);
        })
        .unwrap();
        rig.write_l1(&shadow_l1, K64 + 64, &0xBEEF_CAFEu32.to_le_bytes()).unwrap();

        let slot = pool.alloc_slot().expect("slot");
        let slot_base = (slot as u64) * DIRTY_SLOT_BYTES;
        let target = DirtyTarget {
            title_start: start as usize,
            region_len: LEN,
            slot_off: 0,
            words: 1,
            overflow_flag: false,
            fallback_ranges: Vec::new(),
        };
        rig.run_timed(|list| unsafe {
            list.CopyBufferRegion(&pool.gpu, slot_base, &pool.zero, 0, 4);
            list.ResourceBarrier(&[
                transition(&pool.gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                uav_barrier(),
            ]);
            record_bidir_ranges_tracked(
                list,
                &pipelines,
                title_va,
                0,
                shadow_va,
                0,
                before_va,
                0,
                &[(0, LEN)],
                SyncOp::GpuWriteBackOnly,
                Some((pool.gpu_va + slot_base, 1)),
            );
            list.ResourceBarrier(&[
                uav_barrier(),
                transition(&pool.gpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            ]);
            list.CopyBufferRegion(&pool.readback, slot_base, &pool.gpu, slot_base, 4);
            list.ResourceBarrier(&[transition(
                &pool.gpu,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        })
        .unwrap();

        let raw_bits = unsafe { ((pool.ptr + slot_base as usize) as *const u32).read_volatile() };
        assert_eq!(raw_bits, 0b010, "only page 1 bit must be set in dirty_pages_buf (got {raw_bits:#b})");
        let notified = pool.notify_and_free(slot, &[target]);
        assert_eq!(notified, 1);
        assert_eq!(pool.free_slots.lock().unwrap().len(), DIRTY_SLOTS as usize);
        assert!(crate::watch::landed_lately(start as usize + K64 as usize + 64, 4, 2), "page 1 must be noted in watch");
        assert!(!crate::watch::landed_lately(start as usize, 4, 2), "clean page 0 must NOT be falsely noted");
        assert!(!crate::watch::landed_lately(start as usize + 2 * K64 as usize, 4, 2), "clean page 2 must NOT be falsely noted");

        // Cross-64 KiB-page `uint4` fast-path check (`[0xFFFC, 0x13FFC)`): `tid = 0` processes
        // `o0 = 0xFFFC` where `.x` lies in page 0 (`0xFFFC`) and `.y/.z/.w` lie in page 1 (`0x10000..0x1000C`).
        // Mutating ONLY `.y` at `0x10000` MUST set page 1 (`0b010`) and NOT page 0; mutating BOTH
        // `.x` (`0xFFFC`) and `.y` (`0x10000`) MUST set BOTH page 0 and page 1 (`0b011`)!
        let cross_range = [(0xFFFCusize, 0x13FFCusize)];
        for (mutate_p0, mutate_p1, expect_bits) in [(false, true, 0b010u32), (true, true, 0b011u32)] {
            if mutate_p0 {
                rig.write_l1(&shadow_l1, 0xFFFC, &0xA1A2_A3A4u32.to_le_bytes()).unwrap();
            }
            if mutate_p1 {
                let val = if mutate_p0 { 0xB5B6_B7B8u32 } else { 0xB1B2_B3B4u32 };
                rig.write_l1(&shadow_l1, 0x10000, &val.to_le_bytes()).unwrap();
            }
            let slot2 = pool.alloc_slot().expect("slot2");
            let slot2_base = (slot2 as u64) * DIRTY_SLOT_BYTES;
            rig.run_timed(|list| unsafe {
                list.CopyBufferRegion(&pool.gpu, slot2_base, &pool.zero, 0, 4);
                list.ResourceBarrier(&[
                    transition(&pool.gpu, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                    uav_barrier(),
                ]);
                record_bidir_ranges_tracked(
                    list,
                    &pipelines,
                    title_va,
                    0,
                    shadow_va,
                    0,
                    before_va,
                    0,
                    &cross_range,
                    SyncOp::GpuWriteBackOnly,
                    Some((pool.gpu_va + slot2_base, 1)),
                );
                list.ResourceBarrier(&[
                    uav_barrier(),
                    transition(&pool.gpu, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
                ]);
                list.CopyBufferRegion(&pool.readback, slot2_base, &pool.gpu, slot2_base, 4);
                list.ResourceBarrier(&[transition(
                    &pool.gpu,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            })
            .unwrap();
            let got_bits = unsafe { ((pool.ptr + slot2_base as usize) as *const u32).read_volatile() };
            pool.release_slot(slot2);
            assert_eq!(
                got_bits, expect_bits,
                "cross-64KiB uint4 range [0xFFFC..0x13FFC) (p0={mutate_p0}, p1={mutate_p1}) expected bits {expect_bits:#b}, got {got_bits:#b}"
            );
        }
        let errs = drain_debug_errors(rig.info_queue.as_ref());
        assert!(errs.is_empty(), "DirtyPagePool debug errors: {errs:?}");
    }
}
