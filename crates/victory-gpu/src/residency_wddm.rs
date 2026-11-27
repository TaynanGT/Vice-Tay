//! Tiered WDDM 2.0+ video memory residency orchestrator.
//!
//! Protects hot render targets, depth-stencil surfaces, and active GPU buffers
//! from OS eviction on 8-12 GiB discrete GPUs by coordinating three D3D12/DXGI
//! mechanisms:
//! 1. `IDXGIAdapter3::SetVideoMemoryReservation` to reserve the core RT/depth
//!    and working budget cleanly without exceeding OS `AvailableForReservation`
//!    or `Budget`.
//! 2. `ID3D12Device1::SetResidencyPriority` tagging committed resources and
//!    heaps by tier (`HIGH` = RT/depth/hot buffers, `NORMAL` = active textures
//!    and active pool blocks, `LOW`/`MINIMUM` = idle `pool.rs` spare blocks).
//! 3. Explicit fence-safe `ID3D12Device::Evict` / `ID3D12Device::MakeResident`
//!    on aged idle `pool.rs` texture heaps once every submission referencing
//!    them has completed.
//!
//! Opt-in switch: `VICTORY_WDDM_RESIDENCY=1` enables WDDM reservation,
//! priority tagging, and explicit pool eviction (default `0` leaves baseline
//! residency unchanged). Every OS call is error-tolerant on WARP and older
//! drivers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const MIB: u64 = 1 << 20;

/// WDDM residency priority tier for committed resources and texture heaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ResidencyTier {
    Minimum = 0,
    Low = 1,
    Normal = 2,
    High = 3,
}

impl ResidencyTier {
    pub(crate) const fn d3d12_priority(self) -> D3D12_RESIDENCY_PRIORITY {
        match self {
            ResidencyTier::Minimum => D3D12_RESIDENCY_PRIORITY_MINIMUM,
            ResidencyTier::Low => D3D12_RESIDENCY_PRIORITY_LOW,
            ResidencyTier::Normal => D3D12_RESIDENCY_PRIORITY_NORMAL,
            ResidencyTier::High => D3D12_RESIDENCY_PRIORITY_HIGH,
        }
    }
}

/// Action recommended for an empty, resident texture pool heap during sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IdleBlockAction {
    /// Keep resident at `LOW` priority (hot spare or not yet aged/fence-safe).
    KeepResidentLow,
    /// Aged or under pressure, but GPU fence has not yet retired `idle_fence`:
    /// lower priority to `MINIMUM` without calling `Evict()`.
    DemoteMinimum,
    /// Aged and fence-safe (`completed_submission >= idle_fence`): demote to
    /// `MINIMUM` and explicitly call `ID3D12Device::Evict`.
    Evict,
}

/// Snapshot of WDDM residency telemetry counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WddmResidencyStats {
    pub(crate) reserved_bytes: u64,
    pub(crate) reservation_updates: u64,
    pub(crate) reservation_failures: u64,
    pub(crate) priority_high: u64,
    pub(crate) priority_normal: u64,
    pub(crate) priority_low: u64,
    pub(crate) priority_minimum: u64,
    pub(crate) priority_failures: u64,
    pub(crate) evictions: u64,
    pub(crate) restores: u64,
    pub(crate) evicted_blocks_now: u64,
}

static RESERVED_BYTES: AtomicU64 = AtomicU64::new(0);
static RESERVATION_UPDATES: AtomicU64 = AtomicU64::new(0);
static RESERVATION_FAILURES: AtomicU64 = AtomicU64::new(0);
static RESERVATION_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

static PRIORITY_COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static PRIORITY_FAILURES: AtomicU64 = AtomicU64::new(0);
static PRIORITY_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

static EVICTIONS: AtomicU64 = AtomicU64::new(0);
static RESTORES: AtomicU64 = AtomicU64::new(0);
static EVICTED_BLOCKS_NOW: AtomicU64 = AtomicU64::new(0);
static EVICT_UNSUPPORTED: AtomicBool = AtomicBool::new(false);

/// Parse `VICTORY_WDDM_RESIDENCY` (default `false` unless explicitly `"1"`).
pub(crate) fn parse_wddm_residency_env(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Whether WDDM tiered residency orchestration is enabled (opt-in; default `0`
/// preserves baseline residency unless `VICTORY_WDDM_RESIDENCY=1`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_wddm_residency_env(std::env::var("VICTORY_WDDM_RESIDENCY").ok().as_deref()))
}

/// Probe `D3D12_FEATURE_DATA_ARCHITECTURE1::UMA` (falling back to
/// `D3D12_FEATURE_DATA_ARCHITECTURE`) so WDDM reservation caps distinguish UMA
/// APUs from discrete GPUs.
pub(crate) fn probe_uma(device: &ID3D12Device) -> bool {
    let mut arch1 = D3D12_FEATURE_DATA_ARCHITECTURE1 { NodeIndex: 0, ..Default::default() };
    // SAFETY: valid feature query struct and exact byte size.
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1,
            (&raw mut arch1).cast(),
            size_of_val(&arch1) as u32,
        )
    }
    .is_ok()
    {
        return arch1.UMA.as_bool();
    }
    let mut arch0 = D3D12_FEATURE_DATA_ARCHITECTURE { NodeIndex: 0, ..Default::default() };
    // SAFETY: valid legacy feature query struct and exact byte size.
    if unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE,
            (&raw mut arch0).cast(),
            size_of_val(&arch0) as u32,
        )
    }
    .is_ok()
    {
        return arch0.UMA.as_bool();
    }
    false
}

/// Read current WDDM residency telemetry counters.
pub(crate) fn stats() -> WddmResidencyStats {
    WddmResidencyStats {
        reserved_bytes: RESERVED_BYTES.load(Ordering::Relaxed),
        reservation_updates: RESERVATION_UPDATES.load(Ordering::Relaxed),
        reservation_failures: RESERVATION_FAILURES.load(Ordering::Relaxed),
        priority_minimum: PRIORITY_COUNTS[0].load(Ordering::Relaxed),
        priority_low: PRIORITY_COUNTS[1].load(Ordering::Relaxed),
        priority_normal: PRIORITY_COUNTS[2].load(Ordering::Relaxed),
        priority_high: PRIORITY_COUNTS[3].load(Ordering::Relaxed),
        priority_failures: PRIORITY_FAILURES.load(Ordering::Relaxed),
        evictions: EVICTIONS.load(Ordering::Relaxed),
        restores: RESTORES.load(Ordering::Relaxed),
        evicted_blocks_now: EVICTED_BLOCKS_NOW.load(Ordering::Relaxed),
    }
}

/// Decrement the live evicted-block gauge when an already-evicted pool block
/// is permanently dropped by pool reclamation.
pub(crate) fn note_evicted_block_dropped() {
    let _ = EVICTED_BLOCKS_NOW.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(1))
    });
}

/// Compute a safe WDDM local video memory reservation in bytes.
///
/// Guarantees:
/// - Never exceeds `available_for_reservation` (avoiding `DXGI_ERROR_INVALID_CALL`).
/// - Never exceeds `budget`.
/// - Returns `0` when `budget == 0` or `available_for_reservation == 0` (e.g. WARP).
/// - On discrete GPUs, protects the core RT/depth working set (at least 35% of
///   budget up to 3 GiB floor, scaling up to 60% of budget as working set grows).
/// - On UMA APUs, keeps reservation modest (`<= 25%` of budget) so host CPU and
///   OS allocations retain ample shared DRAM.
pub(crate) fn compute_reservation(
    budget: u64,
    available_for_reservation: u64,
    core_working_bytes: u64,
    uma: bool,
) -> u64 {
    if budget == 0 || available_for_reservation == 0 {
        return 0;
    }
    let max_share = if uma {
        budget / 4
    } else {
        (budget / 100).saturating_mul(60)
    };
    let min_floor = if uma {
        (budget / 8).min(512 * MIB)
    } else {
        ((budget / 100).saturating_mul(35)).min(3072 * MIB)
    };
    let desired = core_working_bytes.max(min_floor);
    let clamped = desired
        .min(max_share)
        .min(available_for_reservation)
        .min(budget);
    // Align down to 1 MiB so rounding never pushes above AvailableForReservation.
    clamped & !(MIB - 1)
}

/// Apply `IDXGIAdapter3::SetVideoMemoryReservation` on `DXGI_MEMORY_SEGMENT_GROUP_LOCAL`
/// with hysteresis and error tolerance. Returns the accepted reservation in bytes
/// when updated or verified, or `None` when disabled/unsupported.
pub(crate) fn update_video_memory_reservation(
    adapter: &IDXGIAdapter3,
    core_working_bytes: u64,
    uma: bool,
) -> Option<u64> {
    if !enabled() {
        return None;
    }
    apply_video_memory_reservation(adapter, core_working_bytes, uma, true)
}

fn apply_video_memory_reservation(
    adapter: &IDXGIAdapter3,
    core_working_bytes: u64,
    uma: bool,
    record_stats: bool,
) -> Option<u64> {
    if RESERVATION_UNSUPPORTED.load(Ordering::Relaxed) {
        return None;
    }
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: plain DXGI query filling our struct.
    if unsafe {
        adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)
    }
    .is_err()
    {
        return None;
    }
    let target = compute_reservation(
        info.Budget,
        info.AvailableForReservation,
        core_working_bytes,
        uma,
    );
    if target == 0 && info.CurrentReservation == 0 {
        if record_stats {
            RESERVED_BYTES.store(0, Ordering::Relaxed);
        }
        return Some(0);
    }

    if record_stats {
        let previous = RESERVED_BYTES.load(Ordering::Relaxed);
        let must_shrink = info.CurrentReservation > info.AvailableForReservation
            || previous > info.AvailableForReservation;
        let delta = previous.abs_diff(target);
        if !must_shrink && previous != 0 && delta < 64 * MIB {
            return Some(previous);
        }
    }

    // SAFETY: target is strictly <= info.AvailableForReservation and info.Budget.
    match unsafe {
        adapter.SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, target)
    } {
        Ok(()) => {
            if record_stats {
                RESERVED_BYTES.store(target, Ordering::Relaxed);
                let n = RESERVATION_UPDATES.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 {
                    tracing::info!(
                        reserved_mb = target >> 20,
                        budget_mb = info.Budget >> 20,
                        available_mb = info.AvailableForReservation >> 20,
                        uma,
                        "WDDM local video memory reservation set"
                    );
                }
            }
            Some(target)
        }
        Err(e) => {
            if record_stats {
                let fails = RESERVATION_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
                if fails == 1 {
                    tracing::debug!(
                        error = %e,
                        target_mb = target >> 20,
                        available_mb = info.AvailableForReservation >> 20,
                        "SetVideoMemoryReservation unavailable on this adapter/driver; continuing without reservation"
                    );
                }
            }
            if e.code() == DXGI_ERROR_UNSUPPORTED
                || e.code() == windows::Win32::Foundation::E_NOTIMPL
            {
                RESERVATION_UNSUPPORTED.store(true, Ordering::Relaxed);
            }
            None
        }
    }
}

/// Apply `ID3D12Device1::SetResidencyPriority` to a single committed resource or
/// `ID3D12Heap` (never call on placed or reserved resources).
///
/// Returns `true` if the driver accepted the priority hint. Error-tolerant on
/// WARP and drivers lacking `ID3D12Device1`.
pub(crate) fn set_residency_priority(
    device: &ID3D12Device,
    pageable: &impl Interface,
    tier: ResidencyTier,
) -> bool {
    if !enabled() {
        return false;
    }
    apply_residency_priority(device, pageable, tier, true)
}

fn apply_residency_priority(
    device: &ID3D12Device,
    pageable: &impl Interface,
    tier: ResidencyTier,
    record_stats: bool,
) -> bool {
    if PRIORITY_UNSUPPORTED.load(Ordering::Relaxed) {
        return false;
    }
    let Ok(device1) = device.cast::<ID3D12Device1>() else {
        PRIORITY_UNSUPPORTED.store(true, Ordering::Relaxed);
        return false;
    };
    let Ok(pageable) = pageable.cast::<ID3D12Pageable>() else {
        return false;
    };
    let priority = tier.d3d12_priority();
    let objects = [Some(pageable)];
    // SAFETY: `objects` and `priority` are 1-element arrays of a live pageable
    // belonging to `device1`.
    match unsafe { device1.SetResidencyPriority(1, objects.as_ptr(), &priority) } {
        Ok(()) => {
            if record_stats {
                PRIORITY_COUNTS[tier as usize].fetch_add(1, Ordering::Relaxed);
            }
            true
        }
        Err(e) => {
            if record_stats {
                let fails = PRIORITY_FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
                if fails == 1 {
                    tracing::debug!(
                        error = %e,
                        tier = ?tier,
                        "ID3D12Device1::SetResidencyPriority unsupported or refused; continuing"
                    );
                }
            }
            if e.code() == windows::Win32::Foundation::E_NOTIMPL
                || e.code() == DXGI_ERROR_UNSUPPORTED
            {
                PRIORITY_UNSUPPORTED.store(true, Ordering::Relaxed);
            }
            false
        }
    }
}

/// Tag a committed host texture with `HIGH` priority for render targets /
/// depth-stencil surfaces, or `NORMAL` for active sampled/general textures.
/// Must NOT be called on placed or reserved resources.
pub(crate) fn set_committed_texture_priority(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    is_rt_or_depth: bool,
) -> bool {
    let tier = if is_rt_or_depth {
        ResidencyTier::High
    } else {
        ResidencyTier::Normal
    };
    set_residency_priority(device, resource, tier)
}

/// Tag a hot device-local committed buffer (e.g. UAV / card buffer) as `HIGH`.
pub(crate) fn set_hot_buffer_priority(device: &ID3D12Device, resource: &ID3D12Resource) -> bool {
    set_residency_priority(device, resource, ResidencyTier::High)
}

/// Tag a `pool.rs` texture `ID3D12Heap` block with its current residency tier.
pub(crate) fn set_heap_priority(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    tier: ResidencyTier,
) -> bool {
    set_residency_priority(device, heap, tier)
}

/// Decide how to manage an idle, resident texture pool heap during the periodic sweep.
///
/// - `age_ms`: milliseconds elapsed since the block became completely empty.
/// - `idle_fence`: highest submission number begun/queued when the block became empty.
/// - `completed_submission`: highest submission number confirmed completed on the GPU fence.
/// - `vram_used_pct`: current local VRAM usage percentage (`0..=100+`), or `None`.
/// - `pressured`: host RAM / commit pressure flag (`crate::residency::pressured()`).
/// - `critical`: critical host commit flag (`crate::residency::critical_commit()`).
/// - `normal_hot_spares`: hot resident spare count preserved under normal VRAM
///   headroom (e.g. `ahead().min(limit).max(2)` so `VICTORY_RIPE_BLOCKS` is never
///   cannibalized while VRAM is unpressured).
/// - `resident_spare_rank`: 0-based index among currently resident empty blocks
///   (newest-idle blocks receive lowest ranks so hot spares stay resident).
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdleBlockQuery {
    pub(crate) age_ms: u64,
    pub(crate) idle_fence: u64,
    pub(crate) completed_submission: u64,
    pub(crate) vram_used_pct: Option<u64>,
    pub(crate) pressured: bool,
    pub(crate) critical: bool,
    pub(crate) normal_hot_spares: usize,
    pub(crate) resident_spare_rank: usize,
}

pub(crate) fn classify_idle_block(q: IdleBlockQuery) -> IdleBlockAction {
    let used_pct = q.vram_used_pct.unwrap_or(0);
    let high_vram = used_pct >= 85 || q.critical;
    let moderate_vram = used_pct >= 75 || q.pressured;

    // Preserve the full ripe ahead window under normal headroom; collapse to
    // 1 hot spare under moderate pressure and 0 under high/critical pressure.
    let hot_spares_to_keep = if high_vram {
        0
    } else if moderate_vram {
        1
    } else {
        q.normal_hot_spares.max(2)
    };

    if q.resident_spare_rank < hot_spares_to_keep {
        return IdleBlockAction::KeepResidentLow;
    }

    let evict_after_ms = if high_vram {
        1_500
    } else if moderate_vram {
        3_000
    } else {
        15_000
    };

    if q.age_ms < evict_after_ms {
        return IdleBlockAction::KeepResidentLow;
    }

    // Fence-safety invariant: never call Evict() until every submission begun or
    // queued before the block became empty has retired on the GPU fence.
    if q.completed_submission < q.idle_fence {
        return IdleBlockAction::DemoteMinimum;
    }

    IdleBlockAction::Evict
}

/// Explicitly evict a fence-safe, idle `ID3D12Heap` (residency refcount 1 -> 0).
///
/// Caller MUST ensure:
/// 1. No live placed resources exist in `heap` (`free.whole(BLOCK)`).
/// 2. No GPU submission referencing `heap` is in flight (`finished >= idle_fence`).
/// 3. `heap` is currently resident (refcount == 1) and reserved against concurrent allocation.
pub(crate) fn evict_heap(device: &ID3D12Device, heap: &ID3D12Heap) -> windows::core::Result<()> {
    if !enabled() {
        return Err(windows::core::Error::from_hresult(DXGI_ERROR_UNSUPPORTED));
    }
    evict_heap_impl(device, heap, true)
}

fn evict_heap_impl(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    record_stats: bool,
) -> windows::core::Result<()> {
    if EVICT_UNSUPPORTED.load(Ordering::Relaxed) {
        return Err(windows::core::Error::from_hresult(DXGI_ERROR_UNSUPPORTED));
    }
    let _ = apply_residency_priority(device, heap, ResidencyTier::Minimum, record_stats);
    let pageable: ID3D12Pageable = heap.cast()?;
    let objects = [Some(pageable)];
    // SAFETY: caller guarantees `heap` is currently resident with refcount 1 and
    // no in-flight GPU work references it.
    match unsafe { device.Evict(&objects) } {
        Ok(()) => {
            if record_stats {
                EVICTIONS.fetch_add(1, Ordering::Relaxed);
                EVICTED_BLOCKS_NOW.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        }
        Err(e) => {
            if e.code() == DXGI_ERROR_UNSUPPORTED
                || e.code() == windows::Win32::Foundation::E_NOTIMPL
            {
                EVICT_UNSUPPORTED.store(true, Ordering::Relaxed);
            }
            Err(e)
        }
    }
}

/// Restore a previously evicted `ID3D12Heap` back to residency (refcount 0 -> 1),
/// wait for GPU page table mapping (`crate::host::mapped`), and restore `NORMAL` priority.
///
/// Preserves the residency refcount invariant on error: if `MakeResident`
/// succeeds (0 -> 1) but `mapped(device)` subsequently fails, the heap is
/// immediately rolled back via `Evict` (1 -> 0) before returning `Err`.
pub(crate) fn make_heap_resident(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
) -> windows::core::Result<()> {
    make_heap_resident_impl(device, heap, true)
}

fn make_heap_resident_impl(
    device: &ID3D12Device,
    heap: &ID3D12Heap,
    record_stats: bool,
) -> windows::core::Result<()> {
    let pageable: ID3D12Pageable = heap.cast()?;
    let objects = [Some(pageable)];
    // SAFETY: `heap` was previously evicted (residency refcount 0) and is still alive.
    unsafe {
        device.MakeResident(&objects)?;
    }
    if let Err(e) = crate::host::mapped(device) {
        // SAFETY: `MakeResident` just incremented refcount 0 -> 1 and no caller
        // has received or referenced this heap yet; roll back to 0 cleanly.
        let _ = unsafe { device.Evict(&objects) };
        return Err(e);
    }
    let _ = apply_residency_priority(device, heap, ResidencyTier::Normal, record_stats);
    if record_stats {
        RESTORES.fetch_add(1, Ordering::Relaxed);
        note_evicted_block_dropped();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn reservation_never_exceeds_os_available_or_budget() {
        // 8 GiB discrete GPU: budget ~7.2 GiB, OS allows reserving ~3.6 GiB.
        let r8 = compute_reservation(7372 * MIB, 3686 * MIB, 2500 * MIB, false);
        assert!(r8 > 0);
        assert!(r8 <= 3686 * MIB);
        assert!(r8 <= 7372 * MIB);
        assert_eq!(r8 % MIB, 0);

        // 12 GiB discrete GPU: floor protects ~3 GiB core RT/depth working set.
        let r12 = compute_reservation(11_200 * MIB, 5_600 * MIB, 1_000 * MIB, false);
        assert_eq!(r12, 3072 * MIB);
        assert!(r12 <= 5_600 * MIB);

        // Large core working set clamps cleanly to AvailableForReservation.
        let r_clamped = compute_reservation(11_200 * MIB, 4_000 * MIB, 9_000 * MIB, false);
        assert_eq!(r_clamped, 4_000 * MIB);

        // Unaligned AvailableForReservation rounds down (never up!).
        let odd_avail = 2048 * MIB + 123_456;
        let r_odd = compute_reservation(8 * GIB, odd_avail, 6 * GIB, false);
        assert!(r_odd <= odd_avail);
        assert_eq!(r_odd, 2048 * MIB);

        // UMA stays conservative (<= 25% of budget).
        let r_uma = compute_reservation(16 * GIB, 8 * GIB, 8 * GIB, true);
        assert_eq!(r_uma, 4 * GIB);

        // Zero budget or zero available (WARP / restricted driver) yields 0.
        assert_eq!(compute_reservation(0, 4 * GIB, GIB, false), 0);
        assert_eq!(compute_reservation(8 * GIB, 0, GIB, false), 0);
    }

    #[test]
    fn idle_block_classification_enforces_fence_safety_and_pressure_bands() {
        // Under normal headroom, the full ripe ahead window (e.g. 12) stays resident at LOW.
        for rank in [0, 1, 11] {
            assert_eq!(
                classify_idle_block(IdleBlockQuery {
                    age_ms: 30_000,
                    idle_fence: 10,
                    completed_submission: 20,
                    vram_used_pct: Some(50),
                    pressured: false,
                    critical: false,
                    normal_hot_spares: 12,
                    resident_spare_rank: rank,
                }),
                IdleBlockAction::KeepResidentLow
            );
        }
        // Aged excess spares beyond the ripe window (rank 12) evict once fence-safe.
        assert_eq!(
            classify_idle_block(IdleBlockQuery {
                age_ms: 15_000,
                idle_fence: 10,
                completed_submission: 10,
                vram_used_pct: Some(50),
                pressured: false,
                critical: false,
                normal_hot_spares: 12,
                resident_spare_rank: 12,
            }),
            IdleBlockAction::Evict
        );
        // Same spare when GPU fence has NOT yet retired idle_fence -> DemoteMinimum only (NEVER Evict!).
        assert_eq!(
            classify_idle_block(IdleBlockQuery {
                age_ms: 60_000,
                idle_fence: 10,
                completed_submission: 9,
                vram_used_pct: Some(95),
                pressured: true,
                critical: true,
                normal_hot_spares: 12,
                resident_spare_rank: 2,
            }),
            IdleBlockAction::DemoteMinimum
        );
        // Under 75% VRAM pressure: collapses hot spares to 1; rank >= 1 evicts after 3s once fence-safe.
        assert_eq!(
            classify_idle_block(IdleBlockQuery {
                age_ms: 3_000,
                idle_fence: 5,
                completed_submission: 5,
                vram_used_pct: Some(75),
                pressured: false,
                critical: false,
                normal_hot_spares: 12,
                resident_spare_rank: 1,
            }),
            IdleBlockAction::Evict
        );
        // Under 85%+ VRAM pressure: collapses hot spares to 0; even rank 0 evicts after 1.5s once fence-safe.
        assert_eq!(
            classify_idle_block(IdleBlockQuery {
                age_ms: 1_500,
                idle_fence: 5,
                completed_submission: 5,
                vram_used_pct: Some(85),
                pressured: false,
                critical: false,
                normal_hot_spares: 12,
                resident_spare_rank: 0,
            }),
            IdleBlockAction::Evict
        );
    }

    /// Live D3D12 round-trip test on hardware GPU and WARP (with optional
    /// `VICTORY_D3D12_DEBUG=1`): exercises `SetVideoMemoryReservation`,
    /// `SetResidencyPriority` across HIGH/NORMAL/LOW/MINIMUM tiers, and
    /// fence-safe `Evict()` -> `MakeResident()` -> placed resource use ->
    /// `Evict()` -> drop with zero D3D12 debug errors.
    #[test]
    fn live_wddm_reservation_priority_and_evict_make_resident_roundtrip() {
        let Some(host) = crate::host::host() else {
            return;
        };

        // Drain any preexisting info-queue messages before our test begins.
        let info_queue = host.device.cast::<ID3D12InfoQueue>().ok();
        if let Some(iq) = &info_queue {
            // SAFETY: clear existing queue on this test's device.
            unsafe { iq.ClearStoredMessages() };
        }

        let active = enabled();

        // 1. Exercise SetVideoMemoryReservation on the live adapter.
        // When VICTORY_WDDM_RESIDENCY=0 (default), verify the gated runtime API
        // is a strict no-op, then exercise the raw D3D12/DXGI mechanism cleanly
        // and restore OS reservation to 0 so baseline process state stays clean.
        let mut live_reserved_mb = 0u64;
        if let Some(adapter) = host.adapter_ref() {
            let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: plain query.
            let queried = unsafe {
                adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
            }
            .is_ok();
            if !active {
                assert!(
                    update_video_memory_reservation(adapter, 256 * MIB, host.is_uma()).is_none(),
                    "gated reservation must be a no-op when VICTORY_WDDM_RESIDENCY=0"
                );
                assert_eq!(stats().reservation_updates, 0);
                assert_eq!(stats().reserved_bytes, 0);
            }
            let res = if active {
                update_video_memory_reservation(adapter, 256 * MIB, host.is_uma())
            } else {
                apply_video_memory_reservation(adapter, 256 * MIB, host.is_uma(), false)
            };
            if queried && local.Budget > 0 && local.AvailableForReservation >= MIB {
                let reserved = res.expect("SetVideoMemoryReservation must succeed on capable adapter");
                assert!(reserved > 0);
                assert!(reserved <= local.AvailableForReservation);
                assert!(reserved <= local.Budget);
                live_reserved_mb = reserved >> 20;
                if !active {
                    // Restore baseline 0 reservation when testing under default-off mode.
                    // SAFETY: 0 reservation is always valid on LOCAL segment.
                    let _ = unsafe {
                        adapter.SetVideoMemoryReservation(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, 0)
                    };
                }
            }
        }

        // 2. Create committed RT (HIGH) and texture heap (NORMAL -> LOW -> MINIMUM).
        let rt_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let default_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut rt: Option<ID3D12Resource> = None;
        // SAFETY: plain committed render target creation.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &rt_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut rt,
                )
                .expect("committed RT creation");
        }
        let rt = rt.unwrap();
        if !active {
            assert!(!set_committed_texture_priority(&host.device, &rt, true));
        }
        let _ = apply_residency_priority(&host.device, &rt, ResidencyTier::High, active);

        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: 4 * MIB,
            Properties: default_heap,
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | crate::host::not_resident(),
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain texture heap creation.
        unsafe {
            host.device
                .CreateHeap(&heap_desc, &mut heap)
                .expect("texture heap creation");
        }
        let heap = heap.unwrap();
        crate::host::resident(&host.device, &heap).expect("initial MakeResident");
        crate::host::mapped(&host.device).expect("initial mapped fence");

        // Exercise priority transitions on the heap.
        let _ = apply_residency_priority(&host.device, &heap, ResidencyTier::Normal, active);
        let _ = apply_residency_priority(&host.device, &heap, ResidencyTier::Low, active);

        if !active {
            assert!(
                evict_heap(&host.device, &heap).is_err(),
                "gated evict_heap must be a no-op when VICTORY_WDDM_RESIDENCY=0"
            );
            assert_eq!(stats().evictions, 0);
        }

        // 3. Explicit Evict() -> MakeResident() round-trip on the idle heap.
        let evicted = if active {
            evict_heap(&host.device, &heap)
        } else {
            evict_heap_impl(&host.device, &heap, false)
        };
        if evicted.is_ok() {
            if active {
                make_heap_resident(&host.device, &heap).expect("MakeResident after Evict");
            } else {
                make_heap_resident_impl(&host.device, &heap, false).expect("MakeResident after Evict");
            }

            // Place a texture into the restored heap and verify it creates cleanly.
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 128,
                Height: 128,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let mut placed: Option<ID3D12Resource> = None;
            // SAFETY: heap is resident with room at offset 0.
            unsafe {
                host.device
                    .CreatePlacedResource(
                        &heap,
                        0,
                        &tex_desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut placed,
                    )
                    .expect("placed texture in restored heap");
            }
            drop(placed);

            // Evict again and drop the evicted heap cleanly (refcount 0 drop must be valid).
            if active {
                evict_heap(&host.device, &heap).expect("second Evict");
                note_evicted_block_dropped();
            } else {
                evict_heap_impl(&host.device, &heap, false).expect("second Evict");
            }
        }

        drop(heap);
        drop(rt);

        let s = stats();
        if !active {
            assert_eq!(s.reservation_updates, 0);
            assert_eq!(s.priority_high, 0);
            assert_eq!(s.priority_normal, 0);
            assert_eq!(s.priority_low, 0);
            assert_eq!(s.priority_minimum, 0);
            assert_eq!(s.evictions, 0);
            assert_eq!(s.restores, 0);
        } else {
            assert!(s.evictions >= 2 || EVICT_UNSUPPORTED.load(Ordering::Relaxed));
            assert_eq!(s.evicted_blocks_now, 0);
        }
        eprintln!(
            "live_wddm_roundtrip: enabled={} live_reserved_mb={} stats_reserved_mb={} updates={} prio(H={},N={},L={},M={}) evictions={} restores={} evicted_now={}",
            active,
            live_reserved_mb,
            s.reserved_bytes >> 20,
            s.reservation_updates,
            s.priority_high,
            s.priority_normal,
            s.priority_low,
            s.priority_minimum,
            s.evictions,
            s.restores,
            s.evicted_blocks_now
        );

        // SAFETY: plain device health check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // 4. Assert zero D3D12 debug layer errors/corruptions when debug layer is enabled.
        if let Some(iq) = &info_queue {
            // SAFETY: inspect stored messages on live info queue.
            unsafe {
                let count = iq.GetNumStoredMessages();
                let mut errors = Vec::new();
                for i in 0..count {
                    let mut len = 0usize;
                    if iq.GetMessage(i, None, &mut len).is_ok() && len > 0 {
                        let mut buf = vec![0u8; len];
                        let ptr = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if iq.GetMessage(i, Some(ptr), &mut len).is_ok() {
                            let msg = &*ptr;
                            if msg.Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                                || msg.Severity == D3D12_MESSAGE_SEVERITY_ERROR
                            {
                                let text = if msg.pDescription.is_null() {
                                    String::new()
                                } else {
                                    std::ffi::CStr::from_ptr(msg.pDescription.cast())
                                        .to_string_lossy()
                                        .into_owned()
                                };
                                errors.push(format!("id={:?}: {text}", msg.ID));
                            }
                        }
                    }
                }
                assert!(
                    errors.is_empty(),
                    "D3D12 debug layer reported error(s) during WDDM residency round-trip: {}",
                    errors.join(" | ")
                );
            }
        }
    }
}
