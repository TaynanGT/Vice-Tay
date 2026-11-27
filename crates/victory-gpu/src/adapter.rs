//! Smart multi-GPU adapter selection, native UMA/APU architecture probing,
//! numeric model filter matching, and display-output LUID diagnostics.
//!
//! All three behavioral paths are independently opt-in (default `0`, preserving
//! exact baseline behavior until opted in) so each can be measured solo on the
//! same physical GPU via `VICTORY_D3D12_ADAPTER`:
//! - `VICTORY_UMA_FAST_PATH=0|1` (default `0`): diagnostic override disabling
//!   card buffer twins and bounce copy-queue staging (`1` forces on for A/B
//!   testing on any GPU; `VICTORY_CARD_BUFFERS=0|1` overrides card buffers
//!   independently). Automatic UMA selection (`"auto"`) is intentionally
//!   disabled after Ryzen AI Max+ 395 / Radeon 8060S measurements showed
//!   disabling card buffers on UMA APUs regresses median FPS from 17 to 3.
//! - `VICTORY_SMART_ADAPTER=0|1` (default `0`): enables multi-criteria
//!   `IDXGIFactory6` (`DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE`) scoring of SM 6.6+,
//!   dedicated VRAM (`>= 4 GiB`), `TiledResourcesTier`, and output LUID affinity.
//!   Explicit `VICTORY_D3D12_ADAPTER=<name|index>` filtering (including numeric
//!   GPU model substrings like `"4090"` and `"6700"`) is always active when set.

use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU8, Ordering};

use windows::Win32::Foundation::{HWND, LUID, POINT};
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::{
    HMONITOR, MONITOR_DEFAULTTOPRIMARY, MONITOR_DEFAULTTONEAREST, MonitorFromPoint,
    MonitorFromWindow,
};
use windows::core::Interface;

const GIB: u64 = 1 << 30;

/// Stable, hashable representation of a DXGI adapter `LUID`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct AdapterLuid {
    pub(crate) low: u32,
    pub(crate) high: i32,
}

impl AdapterLuid {
    pub(crate) const fn from_win(luid: LUID) -> Self {
        Self { low: luid.LowPart, high: luid.HighPart }
    }

    pub(crate) fn format_hex(self) -> String {
        format!("{:#010x}:{:#010x}", self.high as u32, self.low)
    }
}

/// Hardware memory architecture flags from `D3D12_FEATURE_DATA_ARCHITECTURE1`
/// (with fallback to `D3D12_FEATURE_DATA_ARCHITECTURE`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ArchitectureInfo {
    pub(crate) uma: bool,
    pub(crate) cache_coherent_uma: bool,
    pub(crate) tile_based_renderer: bool,
    pub(crate) isolated_mmu: bool,
    pub(crate) probed: bool,
}

/// Runtime fast-path policy derived from [`ArchitectureInfo`] and user switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UmaPolicy {
    /// True when UMA fast path is active (detected UMA or forced on via
    /// `VICTORY_UMA_FAST_PATH=1`, unless killed via `VICTORY_UMA_FAST_PATH=0`).
    pub(crate) fast_path: bool,
    /// Whether `Card` buffer twins (`DEFAULT` + `staging` + `before`) are allowed.
    pub(crate) allow_card_buffers: bool,
    /// Whether separate copy-queue bounce staging copies (`upq` / `relay_copy`)
    /// are allowed.
    pub(crate) allow_bounce_copy_queue: bool,
}

impl Default for UmaPolicy {
    fn default() -> Self {
        Self {
            fast_path: false,
            allow_card_buffers: true,
            allow_bounce_copy_queue: true,
        }
    }
}

/// Packed runtime UMA telemetry bitmask for lock-free hot-path queries.
/// Bit 0: initialized
/// Bit 1: arch.uma
/// Bit 2: arch.cache_coherent_uma
/// Bit 3: arch.tile_based_renderer
/// Bit 4: arch.isolated_mmu
/// Bit 5: policy.fast_path
/// Bit 6: policy.allow_card_buffers
/// Bit 7: policy.allow_bounce_copy_queue
static UMA_TELEMETRY_BITS: AtomicU8 = AtomicU8::new(0);

/// Record the active host's architecture and UMA policy for lock-free queries
/// and telemetry inspection.
pub(crate) fn set_runtime_telemetry(arch: ArchitectureInfo, policy: UmaPolicy) {
    let mut bits = 1u8;
    if arch.uma {
        bits |= 1 << 1;
    }
    if arch.cache_coherent_uma {
        bits |= 1 << 2;
    }
    if arch.tile_based_renderer {
        bits |= 1 << 3;
    }
    if arch.isolated_mmu {
        bits |= 1 << 4;
    }
    if policy.fast_path {
        bits |= 1 << 5;
    }
    if policy.allow_card_buffers {
        bits |= 1 << 6;
    }
    if policy.allow_bounce_copy_queue {
        bits |= 1 << 7;
    }
    UMA_TELEMETRY_BITS.store(bits, Ordering::Release);
}

/// Snapshot of the active host's UMA architecture and policy, if initialized.
pub(crate) fn runtime_telemetry() -> Option<(ArchitectureInfo, UmaPolicy)> {
    let bits = UMA_TELEMETRY_BITS.load(Ordering::Acquire);
    if bits & 1 == 0 {
        return None;
    }
    let arch = ArchitectureInfo {
        uma: bits & (1 << 1) != 0,
        cache_coherent_uma: bits & (1 << 2) != 0,
        tile_based_renderer: bits & (1 << 3) != 0,
        isolated_mmu: bits & (1 << 4) != 0,
        probed: true,
    };
    let policy = UmaPolicy {
        fast_path: bits & (1 << 5) != 0,
        allow_card_buffers: bits & (1 << 6) != 0,
        allow_bounce_copy_queue: bits & (1 << 7) != 0,
    };
    Some((arch, policy))
}

/// True when the active host's UMA policy suppresses redundant card buffer twins.
pub(crate) fn uma_disables_card_buffers() -> bool {
    runtime_telemetry().is_some_and(|(_, policy)| !policy.allow_card_buffers)
}

/// True when the active host's UMA policy suppresses redundant copy-queue bounce copies.
pub(crate) fn uma_disables_bounce_copy_queue() -> bool {
    runtime_telemetry().is_some_and(|(_, policy)| !policy.allow_bounce_copy_queue)
}

/// Probe `D3D12_FEATURE_ARCHITECTURE1` on `device`, falling back to
/// `D3D12_FEATURE_ARCHITECTURE` on older drivers.
pub(crate) fn probe_architecture(device: &ID3D12Device) -> ArchitectureInfo {
    let mut arch1 = D3D12_FEATURE_DATA_ARCHITECTURE1 { NodeIndex: 0, ..Default::default() };
    // SAFETY: valid feature query struct and exact byte size.
    let ok1 = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut arch1).cast(),
                size_of_val(&arch1) as u32,
            )
            .is_ok()
    };
    if ok1 {
        return ArchitectureInfo {
            uma: arch1.UMA.as_bool(),
            cache_coherent_uma: arch1.CacheCoherentUMA.as_bool(),
            tile_based_renderer: arch1.TileBasedRenderer.as_bool(),
            isolated_mmu: arch1.IsolatedMMU.as_bool(),
            probed: true,
        };
    }

    let mut arch0 = D3D12_FEATURE_DATA_ARCHITECTURE { NodeIndex: 0, ..Default::default() };
    // SAFETY: valid legacy feature query struct and exact byte size.
    let ok0 = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE,
                (&raw mut arch0).cast(),
                size_of_val(&arch0) as u32,
            )
            .is_ok()
    };
    if ok0 {
        return ArchitectureInfo {
            uma: arch0.UMA.as_bool(),
            cache_coherent_uma: arch0.CacheCoherentUMA.as_bool(),
            tile_based_renderer: arch0.TileBasedRenderer.as_bool(),
            isolated_mmu: false,
            probed: true,
        };
    }

    ArchitectureInfo::default()
}

/// Opt-in mode for `VICTORY_UMA_FAST_PATH`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum UmaFastPathMode {
    /// Default (`unset`, `"0"`, or `"auto"`): baseline behavior, card twins and copy queues unchanged.
    #[default]
    Off,
    /// Explicit opt-in (`"1"`): diagnostic A/B override disabling card twins and copy-queue staging.
    On,
}

/// Parse `VICTORY_UMA_FAST_PATH` (default `Off`; only explicit `"1"` enables it;
/// `"auto"` stays `Off` to prevent UMA card-buffer disable regressions).
pub(crate) fn parse_uma_fast_path_env(value: Option<&str>) -> UmaFastPathMode {
    match value.map(str::trim) {
        Some("1") => UmaFastPathMode::On,
        _ => UmaFastPathMode::Off,
    }
}

/// Parse an explicit boolean override env value (`"0"` -> `Some(false)`,
/// `"1"` -> `Some(true)`, unset/empty/other -> `None`).
pub(crate) fn parse_tri_state_env(value: Option<&str>) -> Option<bool> {
    match value.map(str::trim) {
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}

/// Parse `VICTORY_SMART_ADAPTER` (default `false` unless explicitly `"1"`).
pub(crate) fn parse_smart_adapter_env(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

/// Whether the multi-criteria `IDXGIFactory6` automatic adapter scorer is opted in
/// (`VICTORY_SMART_ADAPTER=1`; default `0` preserves baseline automatic selection).
pub(crate) fn smart_adapter_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_smart_adapter_env(std::env::var("VICTORY_SMART_ADAPTER").ok().as_deref()))
}

/// Resolve [`UmaPolicy`] from probed [`ArchitectureInfo`] and optional overrides.
///
/// Precedence:
/// - `VICTORY_UMA_FAST_PATH` unset, `"0"`, or `"auto"` (`Off`, default): baseline behavior
///   (`fast_path = false`, card buffers and copy queues unchanged unless overridden).
/// - `VICTORY_UMA_FAST_PATH=1` (`On`): explicit diagnostic override for same-GPU A/B comparison.
/// - `VICTORY_CARD_BUFFERS=0` / `=1`: explicit override for card buffer twins.
pub(crate) fn resolve_uma_policy(
    _arch: ArchitectureInfo,
    uma_mode: UmaFastPathMode,
    card_buffers_override: Option<bool>,
) -> UmaPolicy {
    let fast_path = matches!(uma_mode, UmaFastPathMode::On);
    let allow_card_buffers = match card_buffers_override {
        Some(false) => false,
        Some(true) => true,
        None => !fast_path,
    };
    let allow_bounce_copy_queue = !fast_path;
    UmaPolicy {
        fast_path,
        allow_card_buffers,
        allow_bounce_copy_queue,
    }
}

/// Resolve [`UmaPolicy`] reading `VICTORY_UMA_FAST_PATH` and `VICTORY_CARD_BUFFERS`.
pub(crate) fn resolve_uma_policy_from_env(arch: ArchitectureInfo) -> UmaPolicy {
    let uma_env = std::env::var("VICTORY_UMA_FAST_PATH").ok();
    let card_env = std::env::var("VICTORY_CARD_BUFFERS").ok();
    resolve_uma_policy(
        arch,
        parse_uma_fast_path_env(uma_env.as_deref()),
        parse_tri_state_env(card_env.as_deref()),
    )
}

/// Identify software/WARP adapters so hardware selection never picks them.
pub(crate) fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

fn trim_utf16(buf: &[u16]) -> String {
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16_lossy(&buf[..len])
}

/// Candidate metadata collected for deterministic multi-GPU scoring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AdapterCandidate {
    /// Canonical index from `IDXGIFactory1::EnumAdapters1`.
    pub(crate) dxgi_index: u32,
    /// Rank from `IDXGIFactory6::EnumAdapterByGpuPreference(HIGH_PERFORMANCE)`.
    pub(crate) high_perf_rank: Option<u32>,
    pub(crate) name: String,
    pub(crate) vendor_id: u32,
    pub(crate) device_id: u32,
    pub(crate) luid: AdapterLuid,
    pub(crate) dedicated_vram: u64,
    pub(crate) shared_system_memory: u64,
    pub(crate) software: bool,
    /// Raw `D3D_SHADER_MODEL` value (`0x66` = SM 6.6), `None` if device creation failed.
    pub(crate) shader_model: Option<u32>,
    /// Raw `D3D12_TILED_RESOURCES_TIER` (`0..=4`), `None` if unavailable.
    pub(crate) tiled_resources_tier: Option<u32>,
    pub(crate) arch: ArchitectureInfo,
    /// True when this adapter drives the target display monitor (`MonitorFromWindow`).
    pub(crate) matches_output_luid: bool,
    /// True when this adapter owns at least one DXGI display output.
    pub(crate) has_display_output: bool,
}

/// Breakdown of a scored adapter candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AdapterScore {
    pub(crate) total: i64,
    pub(crate) meets_sm66: bool,
    pub(crate) meets_vram_4gib: bool,
}

/// Score a hardware adapter candidate deterministically.
///
/// Returns `None` for software adapters or candidates that cannot create a
/// D3D12 device (`shader_model.is_none()`).
pub(crate) fn score_adapter(candidate: &AdapterCandidate) -> Option<AdapterScore> {
    if candidate.software {
        return None;
    }
    let sm = candidate.shader_model?;

    let mut total: i64 = 0;

    // 1. Shader Model 6.6+ is required by translated host shaders.
    let sm66 = D3D_SHADER_MODEL_6_6.0 as u32;
    let meets_sm66 = sm >= sm66;
    if meets_sm66 {
        total += 100_000;
        let extra = sm.saturating_sub(sm66).min(5) as i64;
        total += extra * 1_000;
    } else if sm >= D3D_SHADER_MODEL_6_0.0 as u32 {
        total += 10_000;
    }

    // 2. Dedicated VRAM (>= 4 GiB strong preference + gradual scaling up to 32 GiB).
    let meets_vram_4gib = candidate.dedicated_vram >= 4 * GIB;
    if meets_vram_4gib {
        total += 40_000;
        if !candidate.arch.uma {
            total += 15_000;
        }
    }
    let vram_gib = (candidate.dedicated_vram / GIB).min(32) as i64;
    total += vram_gib * 2_000;

    // 3. TiledResourcesTier (Tier 2+ enables sparse/tiled texture streaming; Tier 3+ preferred).
    total += match candidate.tiled_resources_tier.unwrap_or(0) {
        0 => 0,
        1 => 2_000,
        2 => 6_000,
        3 => 10_000,
        _ => 12_000,
    };

    // 4. IDXGIFactory6 HIGH_PERFORMANCE order tiebreaker.
    total += match candidate.high_perf_rank {
        Some(0) => 8_000,
        Some(1) => 4_000,
        Some(2) => 2_000,
        _ => 0,
    };

    // 5. Display output affinity (breaks ties among comparable GPUs without
    // overriding a discrete GPU on a hybrid laptop whose panel is wired to an iGPU).
    if candidate.matches_output_luid {
        total += 5_000;
    } else if candidate.has_display_output {
        total += 500;
    }

    Some(AdapterScore {
        total,
        meets_sm66,
        meets_vram_4gib,
    })
}

/// Match an explicit `VICTORY_D3D12_ADAPTER` filter against enumerated hardware
/// candidates, returning the index inside `candidates` of the winning match.
///
/// Rules:
/// 1. Empty/whitespace filter matches nothing.
/// 2. If `requested` is an explicit canonical DXGI index (`"0"`, `"1"`, etc.,
///    without leading zeros except `"0"`) matching a hardware candidate's
///    `dxgi_index`, return that exact candidate.
/// 3. Otherwise, if `requested` is non-numeric OR has `>= 3` ASCII digits
///    (numeric GPU model numbers like `"4090"`, `"6700"`, `"780"`, `"4070"`,
///    or oversized digit strings that overflow `u32`), perform a case-insensitive
///    substring match against candidate names and pick the highest-scoring match
///    (breaking ties by lowest `dxgi_index`).
/// 4. Short 1-2 digit numeric strings (e.g. `"2"` on a 1-GPU machine) that did
///    not match any hardware `dxgi_index` do NOT fall back to substring search,
///    preventing accidental matches against digits inside model names like `"2080"`.
pub(crate) fn select_explicit_candidate(
    requested: &str,
    candidates: &[AdapterCandidate],
) -> Option<usize> {
    let trimmed = requested.trim();
    if trimmed.is_empty() {
        return None;
    }

    let all_ascii_digits = trimmed.bytes().all(|b| b.is_ascii_digit());
    let canonical_index_syntax =
        all_ascii_digits && (trimmed == "0" || !trimmed.starts_with('0'));

    if canonical_index_syntax
        && let Ok(wanted_index) = trimmed.parse::<u32>()
        && let Some((pos, _)) = candidates
            .iter()
            .enumerate()
            .find(|(_, c)| !c.software && c.dxgi_index == wanted_index)
    {
        return Some(pos);
    }

    // 1-2 digit out-of-range indices should not spuriously substring-match GPU names.
    if all_ascii_digits && trimmed.len() < 3 {
        return None;
    }

    let needle = trimmed.to_ascii_lowercase();
    candidates
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.software && c.name.to_ascii_lowercase().contains(&needle))
        .max_by(|(_, a), (_, b)| {
            let sa = score_adapter(a).map_or(i64::MIN, |s| s.total);
            let sb = score_adapter(b).map_or(i64::MIN, |s| s.total);
            sa.cmp(&sb).then_with(|| b.dxgi_index.cmp(&a.dxgi_index))
        })
        .map(|(pos, _)| pos)
}

/// Pick the best automatic hardware candidate by [`score_adapter`], breaking
/// ties by `high_perf_rank` then `dxgi_index`.
pub(crate) fn select_best_candidate(candidates: &[AdapterCandidate]) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(pos, c)| score_adapter(c).map(|score| (pos, score, c)))
        .max_by(|(_, sa, a), (_, sb, b)| {
            sa.total
                .cmp(&sb.total)
                .then_with(|| {
                    b.high_perf_rank
                        .unwrap_or(u32::MAX)
                        .cmp(&a.high_perf_rank.unwrap_or(u32::MAX))
                })
                .then_with(|| b.dxgi_index.cmp(&a.dxgi_index))
        })
        .map(|(pos, _, _)| pos)
}

/// Information about the DXGI adapter driving a window's display monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OutputAdapterInfo {
    pub(crate) adapter_luid: AdapterLuid,
    pub(crate) adapter_name: String,
    pub(crate) output_name: String,
}

/// Result of comparing the display output adapter LUID against the chosen D3D12
/// device's adapter LUID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OutputLuidCheck {
    pub(crate) device_luid: AdapterLuid,
    pub(crate) device_name: String,
    pub(crate) output_luid: Option<AdapterLuid>,
    pub(crate) output_adapter_name: Option<String>,
    pub(crate) output_device_name: Option<String>,
    pub(crate) mismatched: bool,
}

/// Pure evaluation of chosen device LUID vs target display output adapter LUID.
pub(crate) fn evaluate_luid_mismatch(
    device_luid: AdapterLuid,
    device_name: String,
    output: Option<OutputAdapterInfo>,
) -> OutputLuidCheck {
    match output {
        Some(out) => {
            let mismatched = out.adapter_luid != device_luid;
            OutputLuidCheck {
                device_luid,
                device_name,
                output_luid: Some(out.adapter_luid),
                output_adapter_name: Some(out.adapter_name),
                output_device_name: Some(out.output_name),
                mismatched,
            }
        }
        None => OutputLuidCheck {
            device_luid,
            device_name,
            output_luid: None,
            output_adapter_name: None,
            output_device_name: None,
            mismatched: false,
        },
    }
}

/// Resolve the `HMONITOR` for `hwnd` via `MonitorFromWindow`, falling back to
/// the primary monitor if `hwnd` is `None`.
fn target_monitor(hwnd: Option<HWND>) -> Option<HMONITOR> {
    // SAFETY: plain GDI monitor queries; zero handle means no monitor.
    let monitor = unsafe {
        match hwnd {
            Some(hwnd) if !hwnd.0.is_null() => MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
            _ => MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY),
        }
    };
    (!monitor.0.is_null()).then_some(monitor)
}

/// Find the DXGI adapter that owns the display monitor `monitor`.
fn output_adapter_for_monitor(
    factory: &IDXGIFactory4,
    monitor: HMONITOR,
) -> Option<OutputAdapterInfo> {
    // SAFETY: plain DXGI enumeration of live adapters and outputs.
    unsafe {
        for adapter_idx in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(adapter_idx) else {
                break;
            };
            let Ok(adapter_desc) = adapter.GetDesc1() else {
                continue;
            };
            for output_idx in 0.. {
                let Ok(output) = adapter.EnumOutputs(output_idx) else {
                    break;
                };
                if let Ok(out_desc) = output.GetDesc()
                    && out_desc.Monitor == monitor
                {
                    return Some(OutputAdapterInfo {
                        adapter_luid: AdapterLuid::from_win(adapter_desc.AdapterLuid),
                        adapter_name: trim_utf16(&adapter_desc.Description),
                        output_name: trim_utf16(&out_desc.DeviceName),
                    });
                }
            }
        }
    }
    None
}

/// Resolve the display output adapter owning `hwnd`'s monitor (or primary monitor).
pub(crate) fn resolve_output_adapter(hwnd: Option<HWND>) -> Option<OutputAdapterInfo> {
    let monitor = target_monitor(hwnd)?;
    // SAFETY: plain DXGI factory creation.
    let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok()?;
    output_adapter_for_monitor(&factory, monitor)
}

/// Check `MonitorFromWindow` output adapter LUID against `device.GetAdapterLuid()`
/// and emit an explicit warning if mismatched (hybrid iGPU/dGPU display trap).
///
/// Deduplicates repeated identical checks so startup + presenter verification
/// only re-logs when the window moves to a different output adapter.
pub(crate) fn check_output_luid(device: &ID3D12Device, hwnd: Option<HWND>) -> OutputLuidCheck {
    // SAFETY: plain LUID query on a live device.
    let device_luid = AdapterLuid::from_win(unsafe { device.GetAdapterLuid() });
    let device_name = adapter_name_for_luid(device_luid).unwrap_or_else(|| "unknown adapter".into());
    let output = resolve_output_adapter(hwnd);
    let check = evaluate_luid_mismatch(device_luid, device_name, output);

    static LAST_LOGGED: Mutex<Option<(AdapterLuid, Option<AdapterLuid>)>> = Mutex::new(None);
    let key = (check.device_luid, check.output_luid);
    let should_log = {
        let mut last = LAST_LOGGED.lock().unwrap_or_else(|e| e.into_inner());
        if *last == Some(key) {
            false
        } else {
            *last = Some(key);
            true
        }
    };

    if should_log {
        if check.mismatched {
            tracing::warn!(
                device_adapter = %check.device_name,
                device_luid = %check.device_luid.format_hex(),
                output_adapter = %check.output_adapter_name.as_deref().unwrap_or("unknown"),
                output_luid = %check.output_luid.map(AdapterLuid::format_hex).unwrap_or_default(),
                display = %check.output_device_name.as_deref().unwrap_or("unknown"),
                "display monitor is driven by a different GPU adapter than the chosen D3D12 device \
                 (hybrid display path adds cross-adapter present copies; connect the display to the \
                 discrete GPU, enable discrete MUX mode, or override with VICTORY_D3D12_ADAPTER)"
            );
        } else if let Some(out_luid) = check.output_luid {
            tracing::info!(
                device_adapter = %check.device_name,
                luid = %out_luid.format_hex(),
                display = %check.output_device_name.as_deref().unwrap_or("unknown"),
                "display output adapter matches chosen D3D12 device"
            );
        }
    }

    check
}

fn adapter_name_for_luid(luid: AdapterLuid) -> Option<String> {
    // SAFETY: plain DXGI query.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter1 =
            factory.EnumAdapterByLuid(LUID { LowPart: luid.low, HighPart: luid.high }).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        Some(trim_utf16(&desc.Description))
    }
}

fn probe_shader_model(device: &ID3D12Device) -> Option<u32> {
    let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_8,
    };
    // SAFETY: valid feature query struct and exact byte size.
    let ok = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .is_ok()
    };
    if ok {
        return Some(sm.HighestShaderModel.0 as u32);
    }
    let mut sm66 = D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL_6_6,
    };
    // SAFETY: valid fallback feature query struct.
    unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm66).cast(),
                size_of_val(&sm66) as u32,
            )
            .ok()
            .map(|()| sm66.HighestShaderModel.0 as u32)
    }
}

fn probe_tiled_resources_tier(device: &ID3D12Device) -> Option<u32> {
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: valid feature query struct and exact byte size.
    unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut opts).cast(),
                size_of_val(&opts) as u32,
            )
            .ok()
            .map(|()| opts.TiledResourcesTier.0 as u32)
    }
}

fn create_device_on(adapter: Option<&IDXGIAdapter1>) -> windows::core::Result<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: plain D3D12 device creation.
    unsafe {
        match adapter {
            Some(adapter) => D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?,
            None => D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?,
        }
    }
    device.ok_or_else(windows::core::Error::empty)
}

struct ProbedCandidate {
    meta: AdapterCandidate,
    device: Option<ID3D12Device>,
}

#[cfg(test)]
fn select_baseline_candidate(candidates: &[AdapterCandidate]) -> Option<usize> {
    let sm66 = D3D_SHADER_MODEL_6_6.0 as u32;
    candidates
        .iter()
        .enumerate()
        .find(|(_, c)| !c.software && c.shader_model.is_some_and(|sm| sm >= sm66))
        .map(|(pos, _)| pos)
}

/// Create the preferred hardware D3D12 device.
///
/// - When `VICTORY_D3D12_ADAPTER` is set, explicit selection (DXGI index,
///   numeric model substring like `"4090"` / `"6700"`, or name substring) is
///   always active.
/// - Otherwise, when `VICTORY_SMART_ADAPTER=1`, uses multi-criteria scoring
///   across SM 6.6+, dedicated VRAM (`>= 4 GiB`), `TiledResourcesTier`, and
///   display output LUID affinity.
/// - When `VICTORY_SMART_ADAPTER` is unset or `0` (default), preserves the
///   baseline first-SM-6.6+ `HIGH_PERFORMANCE` adapter selection.
pub(crate) fn create_preferred_device(hwnd: Option<HWND>) -> windows::core::Result<ID3D12Device> {
    let requested = std::env::var("VICTORY_D3D12_ADAPTER").unwrap_or_default();
    let requested = requested.trim();
    let use_smart_scorer = smart_adapter_enabled();

    // SAFETY: plain DXGI enumeration and D3D12 device creation.
    unsafe {
        let factory4: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;

        // Baseline automatic path (no explicit filter and VICTORY_SMART_ADAPTER=0):
        // pick the first non-software SM 6.6+ adapter in HIGH_PERFORMANCE order
        // without creating extra probe devices on secondary adapters.
        if requested.is_empty() && !use_smart_scorer {
            if let Ok(factory6) = factory4.cast::<IDXGIFactory6>() {
                for rank in 0.. {
                    let Ok(adapter) = factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                        rank,
                        DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    ) else {
                        break;
                    };
                    if adapter.GetDesc1().is_ok_and(|desc| !is_software_adapter(&desc))
                        && let Ok(device) = create_device_on(Some(&adapter))
                        && probe_shader_model(&device)
                            .is_some_and(|sm| sm >= D3D_SHADER_MODEL_6_6.0 as u32)
                    {
                        return Ok(device);
                    }
                }
            }
            return create_device_on(None);
        }

        let target_output_luid = target_monitor(hwnd)
            .and_then(|m| output_adapter_for_monitor(&factory4, m))
            .map(|info| info.adapter_luid);

        // Build canonical EnumAdapters1 index map keyed by LUID.
        let mut canonical_indices: Vec<(AdapterLuid, u32)> = Vec::new();
        for index in 0.. {
            let Ok(adapter) = factory4.EnumAdapters1(index) else {
                break;
            };
            if let Ok(desc) = adapter.GetDesc1() {
                canonical_indices.push((AdapterLuid::from_win(desc.AdapterLuid), index));
            }
        }
        let lookup_dxgi_index = |luid: AdapterLuid, fallback: u32| -> u32 {
            canonical_indices
                .iter()
                .find(|&&(k, _)| k == luid)
                .map_or(fallback, |&(_, idx)| idx)
        };

        // Enumerate in IDXGIFactory6 HIGH_PERFORMANCE order when available.
        let mut enumerated: Vec<(IDXGIAdapter1, Option<u32>, u32)> = Vec::new();
        if let Ok(factory6) = factory4.cast::<IDXGIFactory6>() {
            for rank in 0.. {
                let Ok(adapter) = factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    rank,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                if let Ok(desc) = adapter.GetDesc1() {
                    let luid = AdapterLuid::from_win(desc.AdapterLuid);
                    let dxgi_index = lookup_dxgi_index(luid, rank);
                    enumerated.push((adapter, Some(rank), dxgi_index));
                }
            }
        }
        if enumerated.is_empty() {
            for index in 0.. {
                let Ok(adapter) = factory4.EnumAdapters1(index) else {
                    break;
                };
                enumerated.push((adapter, None, index));
            }
        }

        let mut probed: Vec<ProbedCandidate> = Vec::with_capacity(enumerated.len());
        for (adapter, high_perf_rank, dxgi_index) in enumerated {
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            let software = is_software_adapter(&desc);
            if software {
                continue;
            }
            let luid = AdapterLuid::from_win(desc.AdapterLuid);
            let name = trim_utf16(&desc.Description);
            let has_display_output = adapter.EnumOutputs(0).is_ok();
            let matches_output_luid = target_output_luid == Some(luid);

            let device = create_device_on(Some(&adapter)).ok();
            let (shader_model, tiled_resources_tier, arch) = match device.as_ref() {
                Some(dev) => (
                    probe_shader_model(dev),
                    probe_tiled_resources_tier(dev),
                    probe_architecture(dev),
                ),
                None => (None, None, ArchitectureInfo::default()),
            };

            probed.push(ProbedCandidate {
                meta: AdapterCandidate {
                    dxgi_index,
                    high_perf_rank,
                    name,
                    vendor_id: desc.VendorId,
                    device_id: desc.DeviceId,
                    luid,
                    dedicated_vram: desc.DedicatedVideoMemory as u64,
                    shared_system_memory: desc.SharedSystemMemory as u64,
                    software: false,
                    shader_model,
                    tiled_resources_tier,
                    arch,
                    matches_output_luid,
                    has_display_output,
                },
                device,
            });
        }

        let metas: Vec<AdapterCandidate> = probed.iter().map(|p| p.meta.clone()).collect();

        if !requested.is_empty() {
            if let Some(pos) = select_explicit_candidate(requested, &metas) {
                let chosen = &mut probed[pos];
                let score = score_adapter(&chosen.meta).map_or(0, |s| s.total);
                tracing::info!(
                    requested,
                    adapter = %chosen.meta.name,
                    dxgi_index = chosen.meta.dxgi_index,
                    score,
                    vram_mb = chosen.meta.dedicated_vram >> 20,
                    shader_model = ?chosen.meta.shader_model.map(|sm| format!("{sm:#x}")),
                    tiled_tier = ?chosen.meta.tiled_resources_tier,
                    uma = chosen.meta.arch.uma,
                    matches_output = chosen.meta.matches_output_luid,
                    "selected explicit VICTORY_D3D12_ADAPTER"
                );
                if let Some(device) = chosen.device.take() {
                    return Ok(device);
                }
            }
            return Err(windows::core::Error::new(
                DXGI_ERROR_NOT_FOUND,
                format!("VICTORY_D3D12_ADAPTER={requested} matched no hardware adapter"),
            ));
        }

        if let Some(pos) = select_best_candidate(&metas) {
            let chosen = &mut probed[pos];
            let score = score_adapter(&chosen.meta).map_or(0, |s| s.total);
            tracing::info!(
                adapter = %chosen.meta.name,
                dxgi_index = chosen.meta.dxgi_index,
                high_perf_rank = ?chosen.meta.high_perf_rank,
                smart_adapter = use_smart_scorer,
                score,
                vram_mb = chosen.meta.dedicated_vram >> 20,
                shader_model = ?chosen.meta.shader_model.map(|sm| format!("{sm:#x}")),
                tiled_tier = ?chosen.meta.tiled_resources_tier,
                uma = chosen.meta.arch.uma,
                cache_coherent_uma = chosen.meta.arch.cache_coherent_uma,
                matches_output = chosen.meta.matches_output_luid,
                candidates = metas.len(),
                "selected D3D12 hardware adapter"
            );
            if let Some(device) = chosen.device.take() {
                return Ok(device);
            }
        }

        create_device_on(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SampleOpts {
        dedicated_gib: u64,
        sm: u32,
        tiled_tier: u32,
        uma: bool,
        matches_output_luid: bool,
        high_perf_rank: Option<u32>,
    }

    fn sample_candidate(dxgi_index: u32, name: &str, opts: SampleOpts) -> AdapterCandidate {
        AdapterCandidate {
            dxgi_index,
            high_perf_rank: opts.high_perf_rank,
            name: name.to_owned(),
            vendor_id: 0x10de,
            device_id: 0x2700 + dxgi_index,
            luid: AdapterLuid { low: dxgi_index + 1, high: 0 },
            dedicated_vram: opts.dedicated_gib * GIB,
            shared_system_memory: 16 * GIB,
            software: false,
            shader_model: Some(opts.sm),
            tiled_resources_tier: Some(opts.tiled_tier),
            arch: ArchitectureInfo {
                uma: opts.uma,
                cache_coherent_uma: opts.uma,
                tile_based_renderer: false,
                isolated_mmu: false,
                probed: true,
            },
            matches_output_luid: opts.matches_output_luid,
            has_display_output: opts.matches_output_luid,
        }
    }

    #[test]
    fn numeric_model_substrings_and_indices_match_cleanly_without_misparsing() {
        let candidates = vec![
            sample_candidate(
                0,
                "Intel(R) UHD Graphics",
                SampleOpts { dedicated_gib: 0, sm: 0x66, tiled_tier: 3, uma: true, matches_output_luid: true, high_perf_rank: Some(1) },
            ),
            sample_candidate(
                1,
                "NVIDIA GeForce RTX 4090",
                SampleOpts { dedicated_gib: 24, sm: 0x67, tiled_tier: 3, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
            sample_candidate(
                2,
                "AMD Radeon RX 6700 XT",
                SampleOpts { dedicated_gib: 12, sm: 0x66, tiled_tier: 3, uma: false, matches_output_luid: false, high_perf_rank: Some(2) },
            ),
            sample_candidate(
                3,
                "NVIDIA GeForce GTX 1080",
                SampleOpts { dedicated_gib: 8, sm: 0x65, tiled_tier: 2, uma: false, matches_output_luid: false, high_perf_rank: Some(3) },
            ),
        ];

        // Numeric GPU model filters (#131) match model substrings, never index-fail.
        assert_eq!(select_explicit_candidate("4090", &candidates), Some(1));
        assert_eq!(select_explicit_candidate("6700", &candidates), Some(2));
        assert_eq!(select_explicit_candidate("1080", &candidates), Some(3));
        assert_eq!(select_explicit_candidate("  4090  ", &candidates), Some(1));

        // Canonical DXGI indices still select the exact adapter index.
        assert_eq!(select_explicit_candidate("0", &candidates), Some(0));
        assert_eq!(select_explicit_candidate("1", &candidates), Some(1));
        assert_eq!(select_explicit_candidate("2", &candidates), Some(2));

        // Short out-of-range numeric index does not spuriously match digits in "GTX 1080" or "4090".
        assert_eq!(select_explicit_candidate("8", &candidates), None);
        assert_eq!(select_explicit_candidate("9", &candidates), None);

        // Overflowing numeric strings, signs, and empty strings never panic.
        assert_eq!(select_explicit_candidate("99999999999999999999999999", &candidates), None);
        assert_eq!(select_explicit_candidate("-1", &candidates), None);
        assert_eq!(select_explicit_candidate("", &candidates), None);
        assert_eq!(select_explicit_candidate("   ", &candidates), None);

        // Name substring matching picks the highest-scoring match when multiple match.
        assert_eq!(select_explicit_candidate("nvidia", &candidates), Some(1));
        assert_eq!(select_explicit_candidate("radeon", &candidates), Some(2));
    }

    #[test]
    fn scorer_prefers_high_vram_sm66_tiled_tier_and_output_luid_tiebreak() {
        // Hybrid laptop: discrete RTX 4070 beats iGPU even though iGPU owns the panel.
        let hybrid = vec![
            sample_candidate(
                0,
                "AMD Radeon(TM) Graphics",
                SampleOpts { dedicated_gib: 0, sm: 0x66, tiled_tier: 3, uma: true, matches_output_luid: true, high_perf_rank: Some(1) },
            ),
            sample_candidate(
                1,
                "NVIDIA GeForce RTX 4070 Laptop GPU",
                SampleOpts { dedicated_gib: 8, sm: 0x67, tiled_tier: 3, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
        ];
        assert_eq!(select_best_candidate(&hybrid), Some(1));

        // Dual equal discrete GPUs: display-connected output LUID wins the tie.
        let dual_equal = vec![
            sample_candidate(
                0,
                "NVIDIA GeForce RTX 4070 SUPER",
                SampleOpts { dedicated_gib: 12, sm: 0x67, tiled_tier: 3, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
            sample_candidate(
                1,
                "NVIDIA GeForce RTX 4070 SUPER",
                SampleOpts { dedicated_gib: 12, sm: 0x67, tiled_tier: 3, uma: false, matches_output_luid: true, high_perf_rank: Some(0) },
            ),
        ];
        assert_eq!(select_best_candidate(&dual_equal), Some(1));

        // SM 6.6+ beats higher VRAM on SM 6.5.
        let sm_gate = vec![
            sample_candidate(
                0,
                "Legacy 16G SM65",
                SampleOpts { dedicated_gib: 16, sm: 0x65, tiled_tier: 3, uma: false, matches_output_luid: true, high_perf_rank: Some(0) },
            ),
            sample_candidate(
                1,
                "Modern 8G SM66",
                SampleOpts { dedicated_gib: 8, sm: 0x66, tiled_tier: 2, uma: false, matches_output_luid: false, high_perf_rank: Some(1) },
            ),
        ];
        assert_eq!(select_best_candidate(&sm_gate), Some(1));

        // Higher TiledResourcesTier wins among equal SM/VRAM cards.
        let tier_gate = vec![
            sample_candidate(
                0,
                "Card Tier 1",
                SampleOpts { dedicated_gib: 8, sm: 0x66, tiled_tier: 1, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
            sample_candidate(
                1,
                "Card Tier 3",
                SampleOpts { dedicated_gib: 8, sm: 0x66, tiled_tier: 3, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
        ];
        assert_eq!(select_best_candidate(&tier_gate), Some(1));

        // Software adapters are never scored as hardware.
        let mut sw = sample_candidate(
            0,
            "Microsoft Basic Render Driver",
            SampleOpts { dedicated_gib: 0, sm: 0x67, tiled_tier: 3, uma: true, matches_output_luid: true, high_perf_rank: Some(0) },
        );
        sw.software = true;
        assert!(score_adapter(&sw).is_none());
        assert_eq!(select_best_candidate(&[sw]), None);
    }

    #[test]
    fn uma_policy_defaults_off_and_toggles_independently_with_safe_overrides() {
        let discrete = ArchitectureInfo {
            uma: false,
            cache_coherent_uma: false,
            tile_based_renderer: false,
            isolated_mmu: false,
            probed: true,
        };
        let apu = ArchitectureInfo {
            uma: true,
            cache_coherent_uma: true,
            tile_based_renderer: false,
            isolated_mmu: false,
            probed: true,
        };

        // Default (unset, "0", or legacy "auto"): baseline unchanged on both discrete GPUs and UMA APUs.
        assert_eq!(parse_uma_fast_path_env(None), UmaFastPathMode::Off);
        assert_eq!(parse_uma_fast_path_env(Some("0")), UmaFastPathMode::Off);
        assert_eq!(parse_uma_fast_path_env(Some("1")), UmaFastPathMode::On);
        assert_eq!(parse_uma_fast_path_env(Some("auto")), UmaFastPathMode::Off);
        assert_eq!(parse_uma_fast_path_env(Some(" AUTO ")), UmaFastPathMode::Off);

        let p_discrete_default = resolve_uma_policy(discrete, UmaFastPathMode::Off, None);
        let p_apu_default = resolve_uma_policy(apu, UmaFastPathMode::Off, None);
        assert_eq!(
            p_discrete_default,
            UmaPolicy {
                fast_path: false,
                allow_card_buffers: true,
                allow_bounce_copy_queue: true,
            }
        );
        assert_eq!(p_apu_default, p_discrete_default);

        // Even legacy "auto" never disables card buffers automatically on UMA APUs.
        let p_apu_auto = resolve_uma_policy(apu, parse_uma_fast_path_env(Some("auto")), None);
        assert!(!p_apu_auto.fast_path);
        assert!(p_apu_auto.allow_card_buffers);
        assert!(p_apu_auto.allow_bounce_copy_queue);

        // Opt-in diagnostic VICTORY_UMA_FAST_PATH=1 disables card buffers on both APUs and discrete GPUs.
        for arch in [discrete, apu] {
            let p_on = resolve_uma_policy(arch, UmaFastPathMode::On, None);
            assert_eq!(
                p_on,
                UmaPolicy {
                    fast_path: true,
                    allow_card_buffers: false,
                    allow_bounce_copy_queue: false,
                }
            );
        }

        // Explicit VICTORY_CARD_BUFFERS=0 / =1 overrides remain honored independently.
        assert!(!resolve_uma_policy(discrete, UmaFastPathMode::Off, Some(false)).allow_card_buffers);
        assert!(resolve_uma_policy(apu, UmaFastPathMode::On, Some(true)).allow_card_buffers);
    }

    #[test]
    fn both_adapter_and_uma_subsystems_default_off_and_toggle_independently() {
        // Default (unset): both subsystems are OFF.
        assert_eq!(parse_uma_fast_path_env(None), UmaFastPathMode::Off);
        assert!(!parse_smart_adapter_env(None));

        // Explicit "0" keeps each OFF; explicit "1" turns only that switch ON.
        assert_eq!(parse_uma_fast_path_env(Some("0")), UmaFastPathMode::Off);
        assert!(!parse_smart_adapter_env(Some("0")));
        assert_eq!(parse_uma_fast_path_env(Some("1")), UmaFastPathMode::On);
        assert!(parse_smart_adapter_env(Some("1")));
        assert!(!parse_smart_adapter_env(Some("true")));

        // Verify baseline vs smart automatic selector differ cleanly when first
        // HIGH_PERFORMANCE candidate is weaker than a second SM 6.6+ candidate,
        // while explicit VICTORY_D3D12_ADAPTER still works regardless of VICTORY_SMART_ADAPTER.
        let dual = vec![
            sample_candidate(
                0,
                "Card A 8G Tier1",
                SampleOpts { dedicated_gib: 8, sm: 0x66, tiled_tier: 1, uma: false, matches_output_luid: false, high_perf_rank: Some(0) },
            ),
            sample_candidate(
                1,
                "NVIDIA GeForce RTX 4090",
                SampleOpts { dedicated_gib: 24, sm: 0x67, tiled_tier: 3, uma: false, matches_output_luid: true, high_perf_rank: Some(1) },
            ),
        ];
        assert_eq!(select_baseline_candidate(&dual), Some(0));
        assert_eq!(select_best_candidate(&dual), Some(1));
        assert_eq!(select_explicit_candidate("4090", &dual), Some(1));
    }

    #[test]
    fn luid_mismatch_diagnostics_detects_hybrid_display_trap() {
        let dgpu = AdapterLuid { low: 0x1111, high: 0 };
        let igpu = AdapterLuid { low: 0x2222, high: 0 };

        let mismatched = evaluate_luid_mismatch(
            dgpu,
            "NVIDIA GeForce RTX 4070 Laptop GPU".into(),
            Some(OutputAdapterInfo {
                adapter_luid: igpu,
                adapter_name: "Intel(R) Iris(R) Xe Graphics".into(),
                output_name: r"\\.\DISPLAY1".into(),
            }),
        );
        assert!(mismatched.mismatched);
        assert_eq!(mismatched.device_luid, dgpu);
        assert_eq!(mismatched.output_luid, Some(igpu));

        let matched = evaluate_luid_mismatch(
            dgpu,
            "NVIDIA GeForce RTX 4070 SUPER".into(),
            Some(OutputAdapterInfo {
                adapter_luid: dgpu,
                adapter_name: "NVIDIA GeForce RTX 4070 SUPER".into(),
                output_name: r"\\.\DISPLAY1".into(),
            }),
        );
        assert!(!matched.mismatched);

        let headless = evaluate_luid_mismatch(dgpu, "RTX 4070 SUPER".into(), None);
        assert!(!headless.mismatched);
    }

    #[test]
    fn live_device_probes_architecture1_and_luid_cleanly() {
        let explicit_env = std::env::var("VICTORY_D3D12_ADAPTER").ok();
        let host = match crate::host::host() {
            Some(host) => host,
            None => {
                assert!(
                    explicit_env.is_none(),
                    "explicit VICTORY_D3D12_ADAPTER={explicit_env:?} failed to create host device"
                );
                return;
            }
        };
        let arch = probe_architecture(&host.device);
        assert!(arch.probed, "ARCHITECTURE1 or ARCHITECTURE query must succeed on D3D12");
        assert_eq!(host.architecture(), arch);

        let expected_policy = resolve_uma_policy_from_env(arch);
        assert_eq!(host.uma_policy(), expected_policy);
        assert_eq!(runtime_telemetry(), Some((arch, expected_policy)));
        assert_eq!(
            smart_adapter_enabled(),
            parse_smart_adapter_env(std::env::var("VICTORY_SMART_ADAPTER").ok().as_deref())
        );
        if !expected_policy.allow_card_buffers {
            assert!(
                !host.card_ready(),
                "UMA fast path must disable card buffer twins automatically"
            );
            assert!(
                !crate::shadow::cards(),
                "shadow::cards() must return false when UMA disables card buffers"
            );
            assert_eq!(
                crate::shadow::card_room_kept(),
                0,
                "shadow::card_room_kept() must not withhold VRAM on UMA"
            );
        } else {
            assert!(
                host.card_ready(),
                "discrete default policy must keep card buffers ready"
            );
            assert!(
                crate::shadow::cards(),
                "discrete default policy must keep shadow::cards() enabled"
            );
        }

        // Verify synthetic UMA policy derivation from live probe under On / Off / auto.
        let synthetic_uma_on = resolve_uma_policy(
            ArchitectureInfo { uma: true, cache_coherent_uma: true, ..arch },
            UmaFastPathMode::On,
            None,
        );
        assert!(synthetic_uma_on.fast_path);
        assert!(!synthetic_uma_on.allow_card_buffers);
        assert!(!synthetic_uma_on.allow_bounce_copy_queue);
        let synthetic_uma_auto = resolve_uma_policy(
            ArchitectureInfo { uma: true, cache_coherent_uma: true, ..arch },
            parse_uma_fast_path_env(Some("auto")),
            None,
        );
        assert!(!synthetic_uma_auto.fast_path);
        assert!(synthetic_uma_auto.allow_card_buffers);
        let synthetic_uma_off = resolve_uma_policy(
            ArchitectureInfo { uma: true, cache_coherent_uma: true, ..arch },
            UmaFastPathMode::Off,
            None,
        );
        assert!(!synthetic_uma_off.fast_path);
        assert!(synthetic_uma_off.allow_card_buffers);

        // Verify output LUID check runs cleanly without panicking or device errors.
        let check = check_output_luid(&host.device, None);
        assert_eq!(
            check.device_luid,
            AdapterLuid::from_win(unsafe { host.device.GetAdapterLuid() })
        );
        eprintln!(
            "live_device_probe: adapter={:?} uma={} cc_uma={} fast_path={} cards_allowed={} smart_adapter={} luid={} output_luid={:?} mismatched={}",
            check.device_name,
            arch.uma,
            arch.cache_coherent_uma,
            expected_policy.fast_path,
            expected_policy.allow_card_buffers,
            smart_adapter_enabled(),
            check.device_luid.format_hex(),
            check.output_luid.map(AdapterLuid::format_hex),
            check.mismatched
        );
        if let Some(filter) = explicit_env.as_deref().map(str::trim).filter(|s| !s.is_empty())
            && !filter.bytes().all(|b| b.is_ascii_digit() && filter.len() < 3)
        {
            assert!(
                check.device_name.to_ascii_lowercase().contains(&filter.to_ascii_lowercase()),
                "expected chosen adapter {:?} to match explicit filter {:?}",
                check.device_name,
                filter
            );
        }
        // SAFETY: plain query on live device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
