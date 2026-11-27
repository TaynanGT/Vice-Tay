//! Host display output probing, HDR10/scRGB swap-chain configuration, VRR
//! tearing & waitable-latency presentation, high-refresh display mode
//! enumeration, Per-Monitor V2 DPI awareness, and deferred resize/occlusion
//! state management.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Default reference white for SDR UI composited onto an HDR surface, in nits.
pub const DEFAULT_PAPER_WHITE_NITS: f32 = 200.0;
/// Minimum allowed UI paper-white luminance, in nits (Windows scRGB 1.0 = 80 nits).
pub const MIN_PAPER_WHITE_NITS: f32 = 80.0;
/// Maximum allowed UI paper-white luminance, in nits.
pub const MAX_PAPER_WHITE_NITS: f32 = 1000.0;
/// Baseline flip-model swap-chain buffer count when `VICTORY_VRR=0` (default).
pub const BASELINE_SWAP_BUFFERS: u32 = 2;
/// VRR flip-model swap-chain buffer count when `VICTORY_VRR=1` (triple buffered for latency 1).
pub const DEFAULT_SWAP_BUFFERS: u32 = 3;
/// Waitable swap-chain maximum frame latency for minimal input-to-photon delay.
pub const MAX_FRAME_LATENCY: u32 = 1;
/// Bounded wait timeout (ms) on the DXGI frame-latency waitable object so a
/// minimized or occluded window can never stall the presenting thread.
pub const FRAME_LATENCY_WAIT_MS: u32 = 100;

/// Win32 `WM_SIZE` `wParam` constant indicating a minimized window.
pub const SIZE_MINIMIZED_TYPE: usize = 1;

/// User preference for HDR presentation (`VICTORY_HDR`, default `0` = SDR control).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HdrPreference {
    /// Follow the target display's Windows HDR state (HDR10 PQ / scRGB when active, SDR otherwise).
    Auto,
    /// Prefer HDR10 ST.2084 PQ (`R10G10B10A2_UNORM` + `G2084_NONE_P2020`) when HDR is active.
    Hdr10,
    /// Prefer extended-linear scRGB (`R16G16B16A16_FLOAT` + `G10_NONE_P709`) when HDR is active.
    ScRgb,
    /// SDR control presentation (`R8G8B8A8_UNORM` + `G22_NONE_P709`, default).
    Off,
}

impl HdrPreference {
    /// Parses a `VICTORY_HDR` environment variable string.
    ///
    /// Unset or empty defaults to [`HdrPreference::Off`] (SDR control) so HDR
    /// can be A/B compared cleanly against SDR on the same output and scene.
    pub fn parse(raw: Option<&str>) -> Self {
        let Some(raw) = raw else {
            return Self::Off;
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" | "false" | "sdr" => Self::Off,
            "auto" | "1" | "on" | "true" => Self::Auto,
            "hdr10" | "pq" | "st2084" | "10" => Self::Hdr10,
            "scrgb" | "fp16" | "linear" | "16" => Self::ScRgb,
            _ => Self::Off,
        }
    }
}

/// Independent opt-in switches for HDR, VRR, and native DPI/resize handling.
/// All three default to off (`0`) so baseline SDR/swap-chain/window behavior
/// stays unchanged unless explicitly opted in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplaySwitches {
    pub hdr: HdrPreference,
    pub vrr: bool,
    pub native_resize: bool,
}

impl DisplaySwitches {
    pub fn parse_flag(raw: Option<&str>) -> bool {
        raw.is_some_and(|s| matches!(s.trim().to_ascii_lowercase().as_str(), "1" | "on" | "true"))
    }

    pub fn from_env_values(
        hdr: Option<&str>,
        vrr: Option<&str>,
        native_resize: Option<&str>,
    ) -> Self {
        Self {
            hdr: HdrPreference::parse(hdr),
            vrr: Self::parse_flag(vrr),
            native_resize: Self::parse_flag(native_resize),
        }
    }
}

/// Whether the flip-model VRR / high-refresh path is opted in (`VICTORY_VRR=1`,
/// default `0` = baseline 2-buffer swap chain and 60 Hz display mode).
pub fn vrr_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("VICTORY_VRR").ok();
        let on = DisplaySwitches::parse_flag(raw.as_deref());
        if raw.is_some() {
            tracing::info!(on, "VRR / high-refresh mode (VICTORY_VRR)");
        }
        on
    })
}

/// Whether Per-Monitor V2 DPI awareness, deferred `WM_SIZE`/`WM_DPICHANGED`
/// resize tracking, and `DXGI_STATUS_OCCLUDED` `PRESENT_TEST` probing are opted
/// in (`VICTORY_NATIVE_RESIZE=1`, default `0` = baseline window/resize path).
pub fn native_resize_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("VICTORY_NATIVE_RESIZE").ok();
        let on = DisplaySwitches::parse_flag(raw.as_deref());
        if raw.is_some() {
            tracing::info!(on, "native DPI / resize / occlusion path (VICTORY_NATIVE_RESIZE)");
        }
        on
    })
}

/// Logical color space used by the composer and swap chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum ColorSpace {
    /// Standard dynamic range sRGB / Rec.709 gamma ~2.2 (`G22_NONE_P709`).
    Sdr = 0,
    /// HDR10 SMPTE ST.2084 PQ with Rec.2020 primaries (`G2084_NONE_P2020`).
    Hdr10Pq = 1,
    /// Extended linear scRGB with Rec.709 primaries where `1.0 = 80 nits` (`G10_NONE_P709`).
    ScRgb = 2,
}

/// Capabilities read from `IDXGIOutput6::GetDesc1()` for a host display output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputHdrCaps {
    /// True when the OS output color space is HDR10 (`G2084_NONE_P2020`).
    pub hdr_active: bool,
    /// Output color channel depth (typically 8 or 10).
    pub bits_per_color: u32,
    /// Minimum display luminance in nits.
    pub min_luminance: f32,
    /// Peak display luminance in nits.
    pub max_luminance: f32,
    /// Sustained full-frame peak luminance in nits.
    pub max_full_frame_luminance: f32,
    /// CIE 1931 xy chromaticity coordinates for red, green, blue, and white point.
    pub red_primary: [f32; 2],
    pub green_primary: [f32; 2],
    pub blue_primary: [f32; 2],
    pub white_point: [f32; 2],
}

impl Default for OutputHdrCaps {
    fn default() -> Self {
        Self::sdr_default()
    }
}

impl OutputHdrCaps {
    /// Standard SDR Rec.709 fallback capabilities when no HDR output is active.
    pub const fn sdr_default() -> Self {
        Self {
            hdr_active: false,
            bits_per_color: 8,
            min_luminance: 0.05,
            max_luminance: 200.0,
            max_full_frame_luminance: 200.0,
            red_primary: [0.64, 0.33],
            green_primary: [0.30, 0.60],
            blue_primary: [0.15, 0.06],
            white_point: [0.3127, 0.3290],
        }
    }

    /// Returns sanitized `(min_nits, max_nits, max_full_frame_nits)` safe to
    /// hand to a title's HDR tonemapper or DXGI metadata.
    pub fn sanitized_luminance(&self) -> (f32, f32, f32) {
        let max_nits = if self.max_luminance.is_finite() && self.max_luminance > 80.0 {
            self.max_luminance.min(10_000.0)
        } else {
            1000.0
        };
        let min_nits = if self.min_luminance.is_finite()
            && self.min_luminance >= 0.0
            && self.min_luminance < max_nits
        {
            self.min_luminance
        } else {
            0.01
        };
        let full_nits = if self.max_full_frame_luminance.is_finite()
            && self.max_full_frame_luminance > min_nits
        {
            self.max_full_frame_luminance.min(max_nits)
        } else {
            max_nits.min(400.0).max(min_nits)
        };
        (min_nits, max_nits, full_nits)
    }
}

/// Result of resolving whether HDR should be advertised to the title.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrModeDecision {
    pub enabled: bool,
    pub min_tone_map_luminance: f32,
    pub max_tone_map_luminance: f32,
    pub max_full_frame_tone_map_luminance: f32,
}

/// Pure policy deciding whether `try_enable_hdr` reports enabled (`1`) or
/// disabled (`2`) and what luminance metadata it returns.
pub fn resolve_hdr_mode(
    caps: &OutputHdrCaps,
    _preference: u32,
    override_pref: HdrPreference,
) -> HdrModeDecision {
    let (min_nits, max_nits, full_nits) = caps.sanitized_luminance();
    let enabled = match override_pref {
        HdrPreference::Off => false,
        HdrPreference::Auto | HdrPreference::Hdr10 | HdrPreference::ScRgb => caps.hdr_active,
    };
    HdrModeDecision {
        enabled,
        min_tone_map_luminance: min_nits,
        max_tone_map_luminance: max_nits,
        max_full_frame_tone_map_luminance: full_nits,
    }
}

/// Configured `VICTORY_HDR` preference (read once).
pub fn hdr_preference() -> HdrPreference {
    static PREF: OnceLock<HdrPreference> = OnceLock::new();
    *PREF.get_or_init(|| {
        let raw = std::env::var("VICTORY_HDR").ok();
        let pref = HdrPreference::parse(raw.as_deref());
        if raw.is_some() {
            tracing::info!(?pref, "HDR presentation preference (VICTORY_HDR)");
        }
        pref
    })
}

/// Parses `VICTORY_HDR_PAPER_WHITE_NITS`, clamping to `[80.0, 1000.0]`.
pub fn parse_paper_white_nits(raw: Option<&str>) -> f32 {
    raw.and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .map_or(DEFAULT_PAPER_WHITE_NITS, |v| {
            v.clamp(MIN_PAPER_WHITE_NITS, MAX_PAPER_WHITE_NITS)
        })
}

/// Configured UI paper-white luminance in nits (`VICTORY_HDR_PAPER_WHITE_NITS`).
pub fn paper_white_nits() -> f32 {
    static NITS: OnceLock<f32> = OnceLock::new();
    *NITS.get_or_init(|| {
        let raw = std::env::var("VICTORY_HDR_PAPER_WHITE_NITS").ok();
        let nits = parse_paper_white_nits(raw.as_deref());
        if raw.is_some() {
            tracing::info!(nits, "HDR UI paper white (VICTORY_HDR_PAPER_WHITE_NITS)");
        }
        nits
    })
}

/// Configured swap-chain buffer count (`VICTORY_SWAP_BUFFERS`; defaults to `3`
/// when `VICTORY_VRR=1`, else baseline `2`).
pub fn swap_buffer_count() -> u32 {
    static COUNT: OnceLock<u32> = OnceLock::new();
    *COUNT.get_or_init(|| {
        let raw = std::env::var("VICTORY_SWAP_BUFFERS").ok();
        let count = parse_swap_buffers_for(raw.as_deref(), vrr_enabled());
        if raw.is_some() {
            tracing::info!(count, "swap chain buffer count (VICTORY_SWAP_BUFFERS)");
        }
        count
    })
}

pub fn parse_swap_buffers_for(raw: Option<&str>, vrr: bool) -> u32 {
    let default = if vrr {
        DEFAULT_SWAP_BUFFERS
    } else {
        BASELINE_SWAP_BUFFERS
    };
    raw.and_then(|s| s.trim().parse::<u32>().ok())
        .filter(|&n| (2..=4).contains(&n))
        .unwrap_or(default)
}

/// Pure policy for `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING`: explicit `VICTORY_TEARING`
/// wins when set, otherwise follows `vrr` (off by default).
pub fn tearing_requested_for(vrr: bool, raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => vrr,
    }
}

/// Whether `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING` is requested (`VICTORY_VRR=1`,
/// overridable with `VICTORY_TEARING=0|1`; off by default).
pub fn tearing_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("VICTORY_TEARING").ok();
        tearing_requested_for(vrr_enabled(), raw.as_deref())
    })
}

/// Pure policy for `DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT`: explicit
/// `VICTORY_WAITABLE_SWAP` wins when set, otherwise follows `vrr` (off by default).
pub fn waitable_requested_for(vrr: bool, raw: Option<&str>) -> bool {
    match raw.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => vrr,
    }
}

/// Whether `DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT` is requested
/// (`VICTORY_VRR=1`, overridable with `VICTORY_WAITABLE_SWAP=0|1`; off by default).
pub fn waitable_requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let raw = std::env::var("VICTORY_WAITABLE_SWAP").ok();
        waitable_requested_for(vrr_enabled(), raw.as_deref())
    })
}

/// Returns true iff `swap_format` is a legal DXGI format for `space`, keeping
/// swap-chain format and composer color-space checks paired with the display mode.
pub const fn format_matches_color_space(swap_format: i32, space: ColorSpace) -> bool {
    match space {
        ColorSpace::Sdr => matches!(
            swap_format,
            FMT_R8G8B8A8_UNORM | FMT_B8G8R8A8_UNORM | FMT_R10G10B10A2_UNORM | FMT_R16G16B16A16_FLOAT
        ),
        ColorSpace::Hdr10Pq => {
            matches!(swap_format, FMT_R10G10B10A2_UNORM | FMT_R16G16B16A16_FLOAT)
        }
        ColorSpace::ScRgb => swap_format == FMT_R16G16B16A16_FLOAT,
    }
}

/// Present sync interval (`VICTORY_VSYNC=1` -> 1, default 0 for VRR / tearing).
pub fn sync_interval() -> u32 {
    static INTERVAL: OnceLock<u32> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        if std::env::var_os("VICTORY_VSYNC").is_some_and(|v| v == "1") {
            1
        } else {
            0
        }
    })
}

static TITLE_HDR_ENABLED: AtomicBool = AtomicBool::new(false);

/// Records whether the title enabled HDR through `try_enable_hdr`.
pub fn set_title_hdr_enabled(enabled: bool) {
    TITLE_HDR_ENABLED.store(enabled, Ordering::Release);
}

/// Whether the title has negotiated HDR mode.
pub fn title_hdr_enabled() -> bool {
    TITLE_HDR_ENABLED.load(Ordering::Acquire)
}

/// Raw DXGI format constants used by pure format-selection helpers.
pub const FMT_R16G16B16A16_FLOAT: i32 = 10;
pub const FMT_R10G10B10A2_UNORM: i32 = 24;
pub const FMT_R8G8B8A8_UNORM: i32 = 28;
pub const FMT_B8G8R8A8_UNORM: i32 = 87;

/// Pure selection of swap-chain format and composer color spaces for a frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentPlan {
    pub swap_format: i32,
    pub scene_space: ColorSpace,
    pub output_space: ColorSpace,
    pub needs_compose: bool,
}

/// Chooses the swap-chain format, scene color space, output color space, and
/// whether GPU shader composition/conversion is required.
pub fn plan_present(
    native_swap_format: Option<i32>,
    has_overlay: bool,
    hdr_active: bool,
    title_hdr: bool,
    pref: HdrPreference,
) -> Option<PresentPlan> {
    let scene_space = match native_swap_format {
        Some(FMT_R16G16B16A16_FLOAT) => ColorSpace::ScRgb,
        Some(FMT_R10G10B10A2_UNORM) if title_hdr => ColorSpace::Hdr10Pq,
        _ => ColorSpace::Sdr,
    };

    let (swap_format, output_space) = if !hdr_active || pref == HdrPreference::Off {
        let fmt = if has_overlay {
            FMT_R8G8B8A8_UNORM
        } else {
            native_swap_format?
        };
        (fmt, ColorSpace::Sdr)
    } else {
        match pref {
            HdrPreference::ScRgb => (FMT_R16G16B16A16_FLOAT, ColorSpace::ScRgb),
            HdrPreference::Hdr10 => (FMT_R10G10B10A2_UNORM, ColorSpace::Hdr10Pq),
            HdrPreference::Auto => match scene_space {
                ColorSpace::ScRgb => (FMT_R16G16B16A16_FLOAT, ColorSpace::ScRgb),
                ColorSpace::Hdr10Pq => (FMT_R10G10B10A2_UNORM, ColorSpace::Hdr10Pq),
                ColorSpace::Sdr if title_hdr => (FMT_R10G10B10A2_UNORM, ColorSpace::Hdr10Pq),
                ColorSpace::Sdr => {
                    let fmt = if has_overlay {
                        FMT_R8G8B8A8_UNORM
                    } else {
                        native_swap_format?
                    };
                    (fmt, ColorSpace::Sdr)
                }
            },
            HdrPreference::Off => unreachable!(),
        }
    };

    let needs_compose =
        has_overlay || native_swap_format != Some(swap_format) || scene_space != output_space;

    Some(PresentPlan {
        swap_format,
        scene_space,
        output_space,
        needs_compose,
    })
}

/// DXGI `Present` flag constants (identical to Win32 `DXGI_PRESENT_*`).
pub const PRESENT_TEST_FLAG: u32 = 0x0000_0001;
pub const PRESENT_ALLOW_TEARING_FLAG: u32 = 0x0000_0200;

/// Pure decision for `IDXGISwapChain::Present(sync_interval, flags)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PresentFlagsDecision {
    pub sync_interval: u32,
    pub flags: u32,
}

/// Computes legal DXGI `Present` arguments.
///
/// `DXGI_PRESENT_ALLOW_TEARING` is strictly forbidden when:
/// - `sync_interval != 0`,
/// - the swap chain was not created with `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING`,
/// - the swap chain is in exclusive fullscreen, or
/// - `probe_only` (`DXGI_PRESENT_TEST`) is requested.
pub fn decide_present_flags(
    sync_interval: u32,
    swap_has_tearing: bool,
    exclusive_fullscreen: bool,
    probe_only: bool,
) -> PresentFlagsDecision {
    if probe_only {
        return PresentFlagsDecision {
            sync_interval: 0,
            flags: PRESENT_TEST_FLAG,
        };
    }
    let flags = if sync_interval == 0 && swap_has_tearing && !exclusive_fullscreen {
        PRESENT_ALLOW_TEARING_FLAG
    } else {
        0
    };
    PresentFlagsDecision {
        sync_interval,
        flags,
    }
}

/// One raw display mode reported by the host monitor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawDisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_num: u32,
    pub refresh_den: u32,
}

impl RawDisplayMode {
    /// Rounded integer refresh rate in Hz (0 if denominator is zero).
    pub fn rounded_hz(&self) -> u32 {
        if self.refresh_den == 0 || self.refresh_num == 0 {
            return 0;
        }
        (f64::from(self.refresh_num) / f64::from(self.refresh_den)).round() as u32
    }
}

/// Converts a rational refresh rate (`num / den` Hz) into a frame interval in
/// microseconds, clamped to a sane range (`1_000..=33_333` us, i.e. 30–1000 Hz).
pub fn refresh_interval_us(num: u32, den: u32) -> Option<u64> {
    if num == 0 || den == 0 {
        return None;
    }
    let hz = f64::from(num) / f64::from(den);
    if !hz.is_finite() || hz < 24.0 || hz > 1000.0 {
        return None;
    }
    let us = (1_000_000.0 / hz).round() as u64;
    Some(us.clamp(1_000, 41_667))
}

/// Builds a clean, deduplicated progressive display mode list for `dxgi.rs`.
///
/// - When `requested_size` (`VICTORY_DISPLAY_MODE`) is set:
///   - Returns real monitor modes matching that resolution if present;
///   - Otherwise pairs `requested_size` with the distinct refresh rates found on
///     the monitor's highest resolution (preserving real 60/120/144/180/240 Hz rates).
/// - When `requested_size` is unset:
///   - Returns deduplicated real host modes (width >= 640, height >= 360), sorted
///     by `(width, height, hz)`.
/// - Falls back to `(requested_size or 3840x2160) @ 60 Hz` if `raw` has no valid entries.
pub fn select_display_modes(
    raw: &[RawDisplayMode],
    requested_size: Option<(u32, u32)>,
) -> Vec<RawDisplayMode> {
    let valid: Vec<RawDisplayMode> = raw
        .iter()
        .copied()
        .filter(|m| {
            (640..=7680).contains(&m.width)
                && (360..=4320).contains(&m.height)
                && m.refresh_den > 0
                && m.refresh_num > 0
                && (24..=1000).contains(&m.rounded_hz())
        })
        .collect();

    if valid.is_empty() {
        let (w, h) = requested_size.unwrap_or((3840, 2160));
        return vec![RawDisplayMode {
            width: w,
            height: h,
            refresh_num: 60,
            refresh_den: 1,
        }];
    }

    let dedup = |modes: &[RawDisplayMode]| -> Vec<RawDisplayMode> {
        let mut by_key: BTreeMap<(u32, u32, u32), RawDisplayMode> = BTreeMap::new();
        for &m in modes {
            let hz = m.rounded_hz();
            let key = (m.width, m.height, hz);
            by_key
                .entry(key)
                .and_modify(|existing| {
                    // Prefer the rate closest to the integer Hz (e.g. 120/1 over 119.88).
                    let old_err = (f64::from(existing.refresh_num)
                        / f64::from(existing.refresh_den)
                        - f64::from(hz))
                    .abs();
                    let new_err =
                        (f64::from(m.refresh_num) / f64::from(m.refresh_den) - f64::from(hz)).abs();
                    if new_err < old_err {
                        *existing = m;
                    }
                })
                .or_insert(m);
        }
        by_key.into_values().collect()
    };

    if let Some((rw, rh)) = requested_size {
        let exact: Vec<RawDisplayMode> = valid
            .iter()
            .copied()
            .filter(|m| m.width == rw && m.height == rh)
            .collect();
        if !exact.is_empty() {
            return dedup(&exact);
        }
        // Take the real refresh rates supported at the monitor's largest area.
        let max_area = valid
            .iter()
            .map(|m| u64::from(m.width) * u64::from(m.height))
            .max()
            .unwrap_or(0);
        let mapped: Vec<RawDisplayMode> = valid
            .iter()
            .copied()
            .filter(|m| u64::from(m.width) * u64::from(m.height) == max_area)
            .map(|m| RawDisplayMode {
                width: rw,
                height: rh,
                refresh_num: m.refresh_num,
                refresh_den: m.refresh_den,
            })
            .collect();
        return dedup(&mapped);
    }

    dedup(&valid)
}

/// Deferred per-window resize, DPI, and occlusion state machine.
///
/// Window messages (`WM_SIZE`, `WM_DPICHANGED`, `WM_DISPLAYCHANGE`) arrive on
/// the Win32 message thread, while DXGI `ResizeBuffers` / `Present` run on the
/// presenting thread. Recording state transitions here avoids cross-thread
/// deadlocks, zero-sized `ResizeBuffers(0, 0)` (`0x887A0001`), and redundant
/// GPU work while minimized or occluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowResizeState {
    pub minimized: bool,
    pub occluded: bool,
    pub client_size: Option<(u32, u32)>,
    pub dpi: (u32, u32),
    pub resize_epoch: u64,
    pub applied_resize_epoch: u64,
    pub display_epoch: u64,
    pub applied_display_epoch: u64,
}

impl Default for WindowResizeState {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowResizeState {
    pub const fn new() -> Self {
        Self {
            minimized: false,
            occluded: false,
            client_size: None,
            dpi: (96, 96),
            resize_epoch: 0,
            applied_resize_epoch: 0,
            display_epoch: 0,
            applied_display_epoch: 0,
        }
    }

    /// Handles a `WM_SIZE` message (`size_type = wParam`, `width`/`height` from `lParam`).
    pub fn on_wm_size(&mut self, size_type: usize, width: u32, height: u32) {
        // Win32 sends SIZE_MAXSHOW (3) / SIZE_MAXHIDE (4) with lParam == 0 to
        // popup windows when unrelated desktop windows maximize or restore.
        if size_type > 2 {
            return;
        }
        if size_type == SIZE_MINIMIZED_TYPE || width == 0 || height == 0 {
            self.minimized = true;
            return;
        }
        let was_minimized = self.minimized;
        self.minimized = false;
        let next = Some((width, height));
        if self.client_size != next || was_minimized {
            self.client_size = next;
            self.resize_epoch = self.resize_epoch.wrapping_add(1);
        }
    }

    /// Handles a `WM_DPICHANGED` message (`dpi_x`, `dpi_y` from `wParam`).
    pub fn on_wm_dpi_changed(&mut self, dpi_x: u32, dpi_y: u32) {
        let next = (dpi_x.max(1), dpi_y.max(1));
        if self.dpi != next {
            self.dpi = next;
        }
        // Moving across monitors can also change HDR capability or luminance.
        self.display_epoch = self.display_epoch.wrapping_add(1);
    }

    /// Handles a `WM_DISPLAYCHANGE` message or external monitor topology change.
    pub fn on_display_change(&mut self) {
        self.display_epoch = self.display_epoch.wrapping_add(1);
    }

    /// Updates occlusion state from a `Present` status (`DXGI_STATUS_OCCLUDED`).
    pub fn on_present_occluded(&mut self, occluded: bool) {
        self.occluded = occluded;
    }

    /// True when presenting and resizing must be deferred (minimized window).
    pub const fn should_skip_present(&self) -> bool {
        self.minimized
    }

    /// True when the swap chain should probe recovery via `DXGI_PRESENT_TEST`
    /// before recording GPU presentation work.
    pub const fn needs_occlusion_probe(&self) -> bool {
        self.occluded && !self.minimized
    }

    /// True when `ResizeBuffers` is safe to execute for `(width, height)`.
    pub const fn can_resize_buffers(&self, width: u32, height: u32) -> bool {
        !self.minimized && width > 0 && height > 0
    }

    /// True when a new DPI or display change requires re-probing output HDR/refresh caps.
    pub const fn needs_display_reprobe(&self) -> bool {
        self.display_epoch != self.applied_display_epoch
    }

    /// Marks the current display-change epoch as processed.
    pub fn mark_display_probed(&mut self) {
        self.applied_display_epoch = self.display_epoch;
    }

    /// True when a pending window resize has not yet been acknowledged.
    pub const fn has_pending_resize(&self) -> bool {
        self.resize_epoch != self.applied_resize_epoch
    }

    /// Marks the current resize epoch as acknowledged.
    pub fn mark_resized(&mut self) {
        self.applied_resize_epoch = self.resize_epoch;
    }
}

static WINDOW_STATES: Mutex<BTreeMap<usize, WindowResizeState>> = Mutex::new(BTreeMap::new());

fn window_states() -> std::sync::MutexGuard<'static, BTreeMap<usize, WindowResizeState>> {
    WINDOW_STATES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Records `WM_SIZE` for `hwnd` (called from `WndProc` on the UI thread).
pub fn note_wm_size(hwnd: usize, size_type: usize, width: u32, height: u32) {
    if hwnd == 0 {
        return;
    }
    window_states()
        .entry(hwnd)
        .or_default()
        .on_wm_size(size_type, width, height);
}

/// Records `WM_DPICHANGED` for `hwnd` (called from `WndProc` on the UI thread).
pub fn note_wm_dpi_changed(hwnd: usize, dpi_x: u32, dpi_y: u32) {
    if hwnd == 0 {
        return;
    }
    window_states()
        .entry(hwnd)
        .or_default()
        .on_wm_dpi_changed(dpi_x, dpi_y);
}

/// Records `WM_DISPLAYCHANGE` for `hwnd` (or all tracked windows if `hwnd == 0`).
pub fn note_display_change(hwnd: usize) {
    let mut map = window_states();
    if hwnd == 0 {
        for state in map.values_mut() {
            state.on_display_change();
        }
    } else {
        map.entry(hwnd).or_default().on_display_change();
    }
}

/// Removes tracked state for a destroyed window (`WM_NCDESTROY`).
pub fn note_window_destroyed(hwnd: usize) {
    if hwnd != 0 {
        window_states().remove(&hwnd);
    }
}

/// Returns a snapshot of the window's deferred resize/DPI state.
pub fn window_state(hwnd: usize) -> WindowResizeState {
    if hwnd == 0 {
        return WindowResizeState::new();
    }
    window_states().get(&hwnd).copied().unwrap_or_default()
}

/// Marks the display-change epoch as processed on the presenting thread.
pub fn mark_window_display_probed(hwnd: usize) {
    if hwnd != 0
        && let Some(state) = window_states().get_mut(&hwnd)
    {
        state.mark_display_probed();
    }
}

/// Marks the resize epoch as acknowledged on the presenting thread.
pub fn mark_window_resized(hwnd: usize) {
    if hwnd != 0
        && let Some(state) = window_states().get_mut(&hwnd)
    {
        state.mark_resized();
    }
}

/// Outcome of initializing Per-Monitor V2 DPI awareness for this process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DpiAwarenessOutcome {
    Disabled,
    PerMonitorV2,
    AlreadyConfigured,
    LegacyFallback,
    Unavailable,
}

/// Converts normalized CIE 1931 chromaticity (`0.0..=1.0`) to HDR10 static
/// metadata units (`1/50000` per SMPTE ST.2086 / `DXGI_HDR_METADATA_HDR10`).
pub fn chromaticity_to_hdr10(coord: f32) -> u16 {
    if !coord.is_finite() {
        return 0;
    }
    (coord.clamp(0.0, 1.0) * 50_000.0).round() as u16
}

/// Pure HDR10 static metadata tuple matching `DXGI_HDR_METADATA_HDR10` units:
/// - primaries & white point in `1/50000`,
/// - `max_mastering_luminance` in whole nits,
/// - `min_mastering_luminance` in `0.0001` nits,
/// - `max_content_light_level` & `max_frame_average_light_level` in whole nits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hdr10MetadataValues {
    pub red_primary: [u16; 2],
    pub green_primary: [u16; 2],
    pub blue_primary: [u16; 2],
    pub white_point: [u16; 2],
    pub max_mastering_luminance: u32,
    pub min_mastering_luminance: u32,
    pub max_content_light_level: u16,
    pub max_frame_average_light_level: u16,
}

pub fn build_hdr10_metadata_values(caps: &OutputHdrCaps) -> Hdr10MetadataValues {
    let (min_nits, max_nits, full_nits) = caps.sanitized_luminance();
    let valid_primary = |p: [f32; 2], fallback: [f32; 2]| {
        if p[0].is_finite() && p[1].is_finite() && p[0] > 0.0 && p[1] > 0.0 {
            [chromaticity_to_hdr10(p[0]), chromaticity_to_hdr10(p[1])]
        } else {
            [
                chromaticity_to_hdr10(fallback[0]),
                chromaticity_to_hdr10(fallback[1]),
            ]
        }
    };
    // BT.2020 fallback primaries and D65 white point.
    Hdr10MetadataValues {
        red_primary: valid_primary(caps.red_primary, [0.708, 0.292]),
        green_primary: valid_primary(caps.green_primary, [0.170, 0.797]),
        blue_primary: valid_primary(caps.blue_primary, [0.131, 0.046]),
        white_point: valid_primary(caps.white_point, [0.3127, 0.3290]),
        max_mastering_luminance: max_nits.round().clamp(1.0, 10_000.0) as u32,
        min_mastering_luminance: (min_nits * 10_000.0).round().clamp(0.0, 1_000_000.0) as u32,
        max_content_light_level: max_nits.round().clamp(1.0, 10_000.0) as u16,
        max_frame_average_light_level: full_nits.round().clamp(1.0, 10_000.0) as u16,
    }
}

#[cfg(not(windows))]
pub fn ensure_dpi_awareness() -> DpiAwarenessOutcome {
    DpiAwarenessOutcome::Unavailable
}

#[cfg(not(windows))]
pub fn probe_primary_hdr() -> OutputHdrCaps {
    OutputHdrCaps::sdr_default()
}

#[cfg(not(windows))]
pub fn enumerate_host_display_modes() -> Vec<RawDisplayMode> {
    Vec::new()
}

#[cfg(not(windows))]
pub fn active_refresh_interval() -> Option<std::time::Duration> {
    None
}

#[cfg(windows)]
mod win {
    use super::*;
    use std::ffi::c_void;
    use std::time::Duration;
    use windows::Win32::Foundation::{
        CloseHandle, ERROR_ACCESS_DENIED, GetLastError, HANDLE, HWND, POINT, WAIT_OBJECT_0,
    };
    use windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::Graphics::Gdi::{
        DEVMODEW, DM_INTERLACED, ENUM_CURRENT_SETTINGS, EnumDisplaySettingsW, GetMonitorInfoW,
        HMONITOR, MONITOR_DEFAULTTONEAREST, MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MONITORINFOEXW,
        MonitorFromPoint, MonitorFromWindow,
    };
    use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
    use windows::Win32::System::Threading::WaitForSingleObjectEx;
    use windows::core::{BOOL, Interface, PCWSTR, s, w};

    /// Enables Per-Monitor V2 DPI awareness when `VICTORY_NATIVE_RESIZE=1` is
    /// opted in; returns [`DpiAwarenessOutcome::Disabled`] under the baseline
    /// default (`VICTORY_NATIVE_RESIZE=0`).
    pub fn ensure_dpi_awareness() -> DpiAwarenessOutcome {
        if !native_resize_enabled() {
            return DpiAwarenessOutcome::Disabled;
        }
        ensure_dpi_awareness_now()
    }

    /// Unconditionally enables Per-Monitor V2 DPI awareness once per process,
    /// safely degrading on older Windows builds or compatibility layers.
    pub fn ensure_dpi_awareness_now() -> DpiAwarenessOutcome {
        static OUTCOME: OnceLock<DpiAwarenessOutcome> = OnceLock::new();
        *OUTCOME.get_or_init(|| {
            // `DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2` == (DPI_AWARENESS_CONTEXT)-4.
            const PER_MONITOR_AWARE_V2: isize = -4;
            type SetCtx = unsafe extern "system" fn(isize) -> i32;
            type SetLegacy = unsafe extern "system" fn() -> i32;
            type SetShcore = unsafe extern "system" fn(i32) -> i32;

            // SAFETY: dynamically probing standard Win32 user32/shcore exports.
            let outcome = unsafe {
                if let Ok(user32) = GetModuleHandleW(w!("user32.dll")) {
                    if let Some(proc) =
                        GetProcAddress(user32, s!("SetProcessDpiAwarenessContext"))
                    {
                        let func: SetCtx = std::mem::transmute(proc);
                        if func(PER_MONITOR_AWARE_V2) != 0 {
                            DpiAwarenessOutcome::PerMonitorV2
                        } else if GetLastError() == ERROR_ACCESS_DENIED {
                            DpiAwarenessOutcome::AlreadyConfigured
                        } else {
                            DpiAwarenessOutcome::LegacyFallback
                        }
                    } else if let Ok(shcore) = LoadLibraryW(w!("shcore.dll"))
                        && let Some(proc) = GetProcAddress(shcore, s!("SetProcessDpiAwareness"))
                    {
                        let func: SetShcore = std::mem::transmute(proc);
                        // PROCESS_PER_MONITOR_DPI_AWARE = 2
                        let hr = func(2);
                        if hr == 0 {
                            DpiAwarenessOutcome::LegacyFallback
                        } else if hr == 0x8007_0005u32 as i32 {
                            DpiAwarenessOutcome::AlreadyConfigured
                        } else {
                            DpiAwarenessOutcome::Unavailable
                        }
                    } else if let Some(proc) = GetProcAddress(user32, s!("SetProcessDPIAware")) {
                        let func: SetLegacy = std::mem::transmute(proc);
                        if func() != 0 {
                            DpiAwarenessOutcome::LegacyFallback
                        } else {
                            DpiAwarenessOutcome::Unavailable
                        }
                    } else {
                        DpiAwarenessOutcome::Unavailable
                    }
                } else {
                    DpiAwarenessOutcome::Unavailable
                }
            };
            tracing::info!(?outcome, "process DPI awareness initialized");
            outcome
        })
    }

    fn target_monitor(hwnd: Option<HWND>) -> HMONITOR {
        // SAFETY: Win32 monitor queries with valid flags.
        unsafe {
            match hwnd.filter(|h| !h.0.is_null()) {
                Some(h) => MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST),
                None => MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTOPRIMARY),
            }
        }
    }

    fn find_output_for_monitor(target: HMONITOR) -> Option<IDXGIOutput> {
        // SAFETY: DXGI factory/adapter/output enumeration.
        unsafe {
            let factory: IDXGIFactory4 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let mut fallback: Option<IDXGIOutput> = None;
            let mut adapter_idx = 0u32;
            while let Ok(adapter) = factory.EnumAdapters1(adapter_idx) {
                adapter_idx += 1;
                let mut output_idx = 0u32;
                while let Ok(output) = adapter.EnumOutputs(output_idx) {
                    output_idx += 1;
                    if let Ok(desc) = output.GetDesc() {
                        if !target.0.is_null() && desc.Monitor == target {
                            return Some(output);
                        }
                        if fallback.is_none() && desc.AttachedToDesktop.as_bool() {
                            fallback = Some(output);
                        }
                    }
                }
            }
            fallback
        }
    }

    /// Probes HDR capabilities and luminance/primaries via `IDXGIOutput6::GetDesc1()`
    /// for the output hosting `hwnd` (or the primary display when `None`).
    pub fn probe_output_hdr(hwnd: Option<HWND>) -> OutputHdrCaps {
        let monitor = target_monitor(hwnd);
        let Some(output) = find_output_for_monitor(monitor) else {
            return OutputHdrCaps::sdr_default();
        };
        let Ok(output6) = output.cast::<IDXGIOutput6>() else {
            return OutputHdrCaps::sdr_default();
        };
        // SAFETY: valid `IDXGIOutput6` query.
        let Ok(desc1) = (unsafe { output6.GetDesc1() }) else {
            return OutputHdrCaps::sdr_default();
        };
        OutputHdrCaps {
            hdr_active: desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            bits_per_color: desc1.BitsPerColor,
            min_luminance: desc1.MinLuminance,
            max_luminance: desc1.MaxLuminance,
            max_full_frame_luminance: desc1.MaxFullFrameLuminance,
            red_primary: desc1.RedPrimary,
            green_primary: desc1.GreenPrimary,
            blue_primary: desc1.BluePrimary,
            white_point: desc1.WhitePoint,
        }
    }

    /// Probes the title window's display (or the primary display if no window
    /// exists yet) via `IDXGIOutput6::GetDesc1()`.
    pub fn probe_primary_hdr() -> OutputHdrCaps {
        probe_output_hdr(crate::host::title_window())
    }

    fn monitor_device_name(monitor: HMONITOR) -> Option<[u16; 32]> {
        if monitor.0.is_null() {
            return None;
        }
        // SAFETY: `MONITORINFOEXW` starts with `MONITORINFO` whose `cbSize` is initialized.
        unsafe {
            let mut info: MONITORINFOEXW = std::mem::zeroed();
            info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
            let ok = GetMonitorInfoW(monitor, (&raw mut info).cast::<MONITORINFO>()).as_bool();
            ok.then_some(info.szDevice)
        }
    }

    /// Enumerates real progressive display modes (including 120/144/180/240 Hz)
    /// from the host monitor via `IDXGIOutput::GetDisplayModeList`, falling back
    /// to `EnumDisplaySettingsW` when DXGI exposes no desktop outputs.
    pub fn enumerate_host_display_modes_for(hwnd: Option<HWND>) -> Vec<RawDisplayMode> {
        let monitor = target_monitor(hwnd);
        let mut modes = Vec::new();
        if let Some(output) = find_output_for_monitor(monitor) {
            // SAFETY: two-pass DXGI `GetDisplayModeList` query.
            unsafe {
                let mut count = 0u32;
                if output
                    .GetDisplayModeList(
                        DXGI_FORMAT_R8G8B8A8_UNORM,
                        DXGI_ENUM_MODES(0),
                        &mut count,
                        None,
                    )
                    .is_ok()
                    && count > 0
                {
                    let mut raw = vec![DXGI_MODE_DESC::default(); count as usize];
                    if output
                        .GetDisplayModeList(
                            DXGI_FORMAT_R8G8B8A8_UNORM,
                            DXGI_ENUM_MODES(0),
                            &mut count,
                            Some(raw.as_mut_ptr()),
                        )
                        .is_ok()
                    {
                        for d in raw.into_iter().take(count as usize) {
                            if d.ScanlineOrdering == DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED
                                || d.ScanlineOrdering == DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE
                            {
                                modes.push(RawDisplayMode {
                                    width: d.Width,
                                    height: d.Height,
                                    refresh_num: d.RefreshRate.Numerator,
                                    refresh_den: d.RefreshRate.Denominator,
                                });
                            }
                        }
                    }
                }
            }
        }
        if !modes.is_empty() {
            return modes;
        }
        // Fallback: Win32 `EnumDisplaySettingsW`.
        let device = monitor_device_name(monitor);
        let device_ptr = device
            .as_ref()
            .map_or(PCWSTR::null(), |name| PCWSTR(name.as_ptr()));
        // SAFETY: plain `EnumDisplaySettingsW` iteration with sized `DEVMODEW`.
        unsafe {
            let mut idx = 0u32;
            loop {
                let mut dm: DEVMODEW = std::mem::zeroed();
                dm.dmSize = size_of::<DEVMODEW>() as u16;
                if !EnumDisplaySettingsW(device_ptr, windows::Win32::Graphics::Gdi::ENUM_DISPLAY_SETTINGS_MODE(idx), &mut dm).as_bool() {
                    break;
                }
                idx += 1;
                let interlaced = (dm.Anonymous2.dmDisplayFlags & DM_INTERLACED.0) != 0;
                if !interlaced && dm.dmPelsWidth > 0 && dm.dmPelsHeight > 0 && dm.dmDisplayFrequency > 1 {
                    modes.push(RawDisplayMode {
                        width: dm.dmPelsWidth,
                        height: dm.dmPelsHeight,
                        refresh_num: dm.dmDisplayFrequency,
                        refresh_den: 1,
                    });
                }
            }
        }
        modes
    }

    /// Enumerates real progressive display modes for the primary desktop monitor.
    pub fn enumerate_host_display_modes() -> Vec<RawDisplayMode> {
        enumerate_host_display_modes_for(None)
    }

    /// Returns the active display refresh interval of the primary desktop monitor.
    pub fn active_refresh_interval() -> Option<Duration> {
        let monitor = target_monitor(None);
        let device = monitor_device_name(monitor);
        let device_ptr = device
            .as_ref()
            .map_or(PCWSTR::null(), |name| PCWSTR(name.as_ptr()));
        // SAFETY: querying current settings into a sized `DEVMODEW`.
        unsafe {
            let mut dm: DEVMODEW = std::mem::zeroed();
            dm.dmSize = size_of::<DEVMODEW>() as u16;
            if EnumDisplaySettingsW(device_ptr, ENUM_CURRENT_SETTINGS, &mut dm).as_bool() {
                let us = refresh_interval_us(dm.dmDisplayFrequency, 1)?;
                return Some(Duration::from_micros(us));
            }
        }
        None
    }

    /// Queries hardware `DXGI_FEATURE_PRESENT_ALLOW_TEARING` capability regardless
    /// of env switches.
    pub fn check_tearing_support_unconditional(factory: &IDXGIFactory2) -> bool {
        let Ok(factory5) = factory.cast::<IDXGIFactory5>() else {
            return false;
        };
        let mut allow = BOOL(0);
        // SAFETY: writes a `BOOL` of exact size `size_of::<BOOL>()`.
        let ok = unsafe {
            factory5
                .CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    (&raw mut allow).cast::<c_void>(),
                    size_of::<BOOL>() as u32,
                )
                .is_ok()
        };
        ok && allow.as_bool()
    }

    /// Queries `IDXGIFactory5::CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING)`
    /// gated by `VICTORY_VRR` / `VICTORY_TEARING`.
    pub fn check_tearing_support(factory: &IDXGIFactory2) -> bool {
        tearing_requested() && check_tearing_support_unconditional(factory)
    }

    /// Converts a [`ColorSpace`] to its DXGI color-space enum value.
    pub fn dxgi_color_space(space: ColorSpace) -> DXGI_COLOR_SPACE_TYPE {
        match space {
            ColorSpace::Sdr => DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
            ColorSpace::Hdr10Pq => DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
            ColorSpace::ScRgb => DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
        }
    }

    /// Builds a `DXGI_HDR_METADATA_HDR10` struct from [`OutputHdrCaps`].
    pub fn build_hdr10_metadata(caps: &OutputHdrCaps) -> DXGI_HDR_METADATA_HDR10 {
        let v = build_hdr10_metadata_values(caps);
        DXGI_HDR_METADATA_HDR10 {
            RedPrimary: v.red_primary,
            GreenPrimary: v.green_primary,
            BluePrimary: v.blue_primary,
            WhitePoint: v.white_point,
            MaxMasteringLuminance: v.max_mastering_luminance,
            MinMasteringLuminance: v.min_mastering_luminance,
            MaxContentLightLevel: v.max_content_light_level,
            MaxFrameAverageLightLevel: v.max_frame_average_light_level,
        }
    }

    /// Applies color space and HDR10 static metadata to `swap_chain`, returning
    /// the color space actually activated (degrading cleanly to `Sdr` if unsupported).
    pub fn apply_color_space_and_metadata(
        swap_chain: &IDXGISwapChain4,
        format: DXGI_FORMAT,
        desired: ColorSpace,
        caps: &OutputHdrCaps,
    ) -> ColorSpace {
        let effective_desired = match (desired, format) {
            (ColorSpace::ScRgb, DXGI_FORMAT_R16G16B16A16_FLOAT) => ColorSpace::ScRgb,
            (ColorSpace::Hdr10Pq, DXGI_FORMAT_R10G10B10A2_UNORM)
            | (ColorSpace::Hdr10Pq, DXGI_FORMAT_R16G16B16A16_FLOAT) => ColorSpace::Hdr10Pq,
            _ => ColorSpace::Sdr,
        };
        let target_cs = dxgi_color_space(effective_desired);
        // SAFETY: valid `IDXGISwapChain4` color-space queries and setters.
        unsafe {
            let supported = swap_chain
                .CheckColorSpaceSupport(target_cs)
                .map(|mask| (mask & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32) != 0)
                .unwrap_or(false);
            let active = if supported && swap_chain.SetColorSpace1(target_cs).is_ok() {
                effective_desired
            } else {
                if effective_desired != ColorSpace::Sdr {
                    tracing::info!(
                        ?effective_desired,
                        format = format.0,
                        "swap chain color space unsupported; falling back to SDR"
                    );
                }
                let _ = swap_chain.SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
                ColorSpace::Sdr
            };
            match active {
                ColorSpace::Hdr10Pq | ColorSpace::ScRgb => {
                    let meta = build_hdr10_metadata(caps);
                    // SAFETY: `meta` is a plain `#[repr(C)]` struct with no padding bytes read uninitialized.
                    let meta_bytes = std::slice::from_raw_parts(
                        (&raw const meta).cast::<u8>(),
                        size_of::<DXGI_HDR_METADATA_HDR10>(),
                    );
                    if let Err(err) = swap_chain.SetHDRMetaData(
                        DXGI_HDR_METADATA_TYPE_HDR10,
                        Some(meta_bytes),
                    ) {
                        tracing::debug!(%err, "SetHDRMetaData(HDR10) skipped or unsupported");
                    }
                }
                ColorSpace::Sdr => {
                    let _ = swap_chain.SetHDRMetaData(DXGI_HDR_METADATA_TYPE_NONE, None);
                }
            }
            active
        }
    }

    /// Owned handle to a DXGI frame-latency waitable object (`CloseHandle` on drop).
    pub struct WaitableLatency {
        handle: HANDLE,
        pending: bool,
    }

    // SAFETY: Win32 kernel waitable handles are safe to send/wait across threads
    // when access is externally serialized (held behind `Host::presenter`).
    unsafe impl Send for WaitableLatency {}

    impl WaitableLatency {
        pub fn new(handle: HANDLE) -> Option<Self> {
            (!handle.0.is_null()).then_some(Self {
                handle,
                pending: false,
            })
        }

        /// Waits once (bounded by `FRAME_LATENCY_WAIT_MS`) before presenting a frame.
        /// If a prior wait was consumed but `Present` was aborted early, avoids
        /// double-waiting until a `Present` actually executes.
        pub fn wait_before_present(&mut self) -> bool {
            if self.pending {
                return true;
            }
            // SAFETY: owned valid DXGI waitable handle.
            let status = unsafe { WaitForSingleObjectEx(self.handle, FRAME_LATENCY_WAIT_MS, false) };
            let signaled = status == WAIT_OBJECT_0;
            if signaled {
                self.pending = true;
            }
            signaled
        }

        /// Called once `Present` has been invoked so the next frame waits again.
        pub fn note_presented(&mut self) {
            self.pending = false;
        }
    }

    impl Drop for WaitableLatency {
        fn drop(&mut self) {
            if !self.handle.0.is_null() {
                // SAFETY: owned handle returned by `GetFrameLatencyWaitableObject`.
                unsafe {
                    let _ = CloseHandle(self.handle);
                }
                self.handle = HANDLE::default();
            }
        }
    }

    /// Flip-model swap chain created with VRR tearing, waitable latency, and
    /// HDR color-space/metadata configuration.
    pub struct CreatedSwapChain {
        pub swap_chain: IDXGISwapChain4,
        pub buffer_count: u32,
        pub flags: DXGI_SWAP_CHAIN_FLAG,
        pub tearing_enabled: bool,
        pub waitable: Option<WaitableLatency>,
        pub color_space: ColorSpace,
    }

    /// Explicit swap-chain feature options (decoupled from process env vars so
    /// baseline SDR/2-buffer and VRR/3-buffer paths can be tested independently).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SwapChainOptions {
        pub buffer_count: u32,
        pub request_tearing: bool,
        pub request_waitable: bool,
        pub disable_alt_enter: bool,
    }

    impl SwapChainOptions {
        /// Options derived from the live `VICTORY_VRR` / `VICTORY_SWAP_BUFFERS` /
        /// `VICTORY_TEARING` / `VICTORY_WAITABLE_SWAP` switches.
        pub fn from_env(allow_waitable: bool, has_title_window: bool) -> Self {
            let request_tearing = tearing_requested();
            Self {
                buffer_count: swap_buffer_count(),
                request_tearing,
                request_waitable: allow_waitable && waitable_requested(),
                // Always disable DXGI Alt+Enter when tearing is requested or for
                // the title's own borderless toggle, keeping baseline behavior on
                // the fallback window when tearing is off.
                disable_alt_enter: has_title_window || request_tearing,
            }
        }
    }

    /// Creates a flip-model `IDXGISwapChain4` configured by `opts` and paired
    /// with `desired_space`, gracefully degrading optional VRR/waitable flags on
    /// drivers/WARP that refuse them.
    #[allow(clippy::too_many_arguments)]
    pub fn create_swap_chain_with_options(
        factory: &IDXGIFactory2,
        queue: &ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        desired_space: ColorSpace,
        hdr_caps: &OutputHdrCaps,
        opts: SwapChainOptions,
    ) -> windows::core::Result<CreatedSwapChain> {
        let tearing_supported = opts.request_tearing && check_tearing_support_unconditional(factory);
        let waitable_wanted = opts.request_waitable;
        let buffer_count = opts.buffer_count.clamp(2, 4);

        let tearing_bit = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32;
        let waitable_bit = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT.0 as u32;

        // Progressive fallback chain: requested flags -> strip tearing -> strip waitable -> plain.
        let mut candidates = Vec::with_capacity(4);
        let initial = (if tearing_supported { tearing_bit } else { 0 })
            | (if waitable_wanted { waitable_bit } else { 0 });
        for f in [
            initial,
            initial & !tearing_bit,
            initial & !waitable_bit,
            0u32,
        ] {
            if !candidates.contains(&f) {
                candidates.push(f);
            }
        }

        let mut created: Option<(IDXGISwapChain4, u32)> = None;
        let mut last_err = None;
        for flags in candidates {
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width.max(1),
                Height: height.max(1),
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: buffer_count,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                Flags: flags,
                ..Default::default()
            };
            // SAFETY: valid D3D12 queue, live window, and valid swap-chain description.
            match unsafe { factory.CreateSwapChainForHwnd(queue, hwnd, &desc, None, None) }
                .and_then(|sc| sc.cast::<IDXGISwapChain4>())
            {
                Ok(swap_chain) => {
                    created = Some((swap_chain, flags));
                    break;
                }
                Err(err) => {
                    tracing::debug!(
                        %err,
                        flags = format_args!("{flags:#x}"),
                        buffer_count,
                        "CreateSwapChainForHwnd attempt rejected; trying fallback flags"
                    );
                    last_err = Some(err);
                }
            }
        }

        let (swap_chain, active_flags) = match created {
            Some(pair) => pair,
            None => return Err(last_err.unwrap_or_else(windows::core::Error::empty)),
        };

        if opts.disable_alt_enter || (active_flags & tearing_bit) != 0 {
            // SAFETY: valid factory and live HWND.
            unsafe {
                let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
            }
        }

        let tearing_enabled = (active_flags & tearing_bit) != 0;
        let waitable = if (active_flags & waitable_bit) != 0 {
            // SAFETY: swap chain was created with `FRAME_LATENCY_WAITABLE_OBJECT`.
            unsafe {
                if let Err(err) = swap_chain.SetMaximumFrameLatency(MAX_FRAME_LATENCY) {
                    tracing::warn!(%err, "SetMaximumFrameLatency(1) failed");
                }
                WaitableLatency::new(swap_chain.GetFrameLatencyWaitableObject())
            }
        } else {
            None
        };

        let color_space = if desired_space == ColorSpace::Sdr {
            ColorSpace::Sdr
        } else {
            apply_color_space_and_metadata(&swap_chain, format, desired_space, hdr_caps)
        };

        tracing::info!(
            width,
            height,
            format = format.0,
            buffer_count,
            tearing_enabled,
            waitable = waitable.is_some(),
            ?color_space,
            "flip-model swap chain ready"
        );

        Ok(CreatedSwapChain {
            swap_chain,
            buffer_count,
            flags: DXGI_SWAP_CHAIN_FLAG(active_flags as i32),
            tearing_enabled,
            waitable,
            color_space,
        })
    }

    /// Creates a flip-model `IDXGISwapChain4` using the active env switches.
    #[allow(clippy::too_many_arguments)]
    pub fn create_swap_chain(
        factory: &IDXGIFactory2,
        queue: &ID3D12CommandQueue,
        hwnd: HWND,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        desired_space: ColorSpace,
        hdr_caps: &OutputHdrCaps,
        allow_waitable: bool,
        has_title_window: bool,
    ) -> windows::core::Result<CreatedSwapChain> {
        create_swap_chain_with_options(
            factory,
            queue,
            hwnd,
            width,
            height,
            format,
            desired_space,
            hdr_caps,
            SwapChainOptions::from_env(allow_waitable, has_title_window),
        )
    }

    /// Presents `swap_chain` using legal VRR tearing or occlusion-test flags,
    /// falling back once without tearing if the runtime rejects tearing due to
    /// an unexpected fullscreen transition.
    pub fn present_swap_chain(
        swap_chain: &IDXGISwapChain4,
        tearing_enabled: bool,
        interval: u32,
        probe_only: bool,
    ) -> windows::core::HRESULT {
        let mut fullscreen = BOOL(0);
        // SAFETY: querying current fullscreen state on a live swap chain.
        let is_fullscreen = unsafe { swap_chain.GetFullscreenState(Some(&mut fullscreen), None) }
            .is_ok()
            && fullscreen.as_bool();
        let decision = decide_present_flags(interval, tearing_enabled, is_fullscreen, probe_only);
        // SAFETY: caller holds exclusive presenter access and submitted the back buffer.
        let status = unsafe { swap_chain.Present(decision.sync_interval, DXGI_PRESENT(decision.flags)) };
        if status == DXGI_ERROR_INVALID_CALL && (decision.flags & PRESENT_ALLOW_TEARING_FLAG) != 0 {
            tracing::debug!("Present(ALLOW_TEARING) returned INVALID_CALL; retrying without tearing");
            // SAFETY: same as above, without tearing flag.
            return unsafe { swap_chain.Present(decision.sync_interval, DXGI_PRESENT(0)) };
        }
        status
    }
}

#[cfg(windows)]
pub use win::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdr_preference_parses_all_modes() {
        assert_eq!(HdrPreference::parse(None), HdrPreference::Off);
        assert_eq!(HdrPreference::parse(Some("")), HdrPreference::Off);
        assert_eq!(HdrPreference::parse(Some("bogus")), HdrPreference::Off);
        assert_eq!(HdrPreference::parse(Some("auto")), HdrPreference::Auto);
        assert_eq!(HdrPreference::parse(Some("1")), HdrPreference::Auto);
        assert_eq!(HdrPreference::parse(Some("on")), HdrPreference::Auto);
        assert_eq!(HdrPreference::parse(Some("true")), HdrPreference::Auto);
        assert_eq!(HdrPreference::parse(Some("0")), HdrPreference::Off);
        assert_eq!(HdrPreference::parse(Some("SDR")), HdrPreference::Off);
        assert_eq!(HdrPreference::parse(Some("hdr10")), HdrPreference::Hdr10);
        assert_eq!(HdrPreference::parse(Some("PQ")), HdrPreference::Hdr10);
        assert_eq!(HdrPreference::parse(Some("scrgb")), HdrPreference::ScRgb);
        assert_eq!(HdrPreference::parse(Some("fp16")), HdrPreference::ScRgb);
    }

    #[test]
    fn independent_switches_default_off_and_toggle_without_crosstalk() {
        let baseline = DisplaySwitches::from_env_values(None, None, None);
        assert_eq!(
            baseline,
            DisplaySwitches {
                hdr: HdrPreference::Off,
                vrr: false,
                native_resize: false,
            }
        );
        assert_eq!(parse_swap_buffers_for(None, baseline.vrr), BASELINE_SWAP_BUFFERS);
        assert!(!tearing_requested_for(baseline.vrr, None));
        assert!(!waitable_requested_for(baseline.vrr, None));

        // Enabling HDR alone leaves VRR and native resize at baseline off.
        let hdr_only = DisplaySwitches::from_env_values(Some("hdr10"), None, None);
        assert_eq!(hdr_only.hdr, HdrPreference::Hdr10);
        assert!(!hdr_only.vrr);
        assert!(!hdr_only.native_resize);

        // Enabling VRR alone leaves HDR (SDR control) and native resize off.
        let vrr_only = DisplaySwitches::from_env_values(None, Some("1"), None);
        assert_eq!(vrr_only.hdr, HdrPreference::Off);
        assert!(vrr_only.vrr);
        assert!(!vrr_only.native_resize);
        assert_eq!(parse_swap_buffers_for(None, vrr_only.vrr), DEFAULT_SWAP_BUFFERS);
        assert!(tearing_requested_for(vrr_only.vrr, None));
        assert!(waitable_requested_for(vrr_only.vrr, None));

        // Enabling native resize alone leaves HDR and VRR off.
        let resize_only = DisplaySwitches::from_env_values(None, None, Some("1"));
        assert_eq!(resize_only.hdr, HdrPreference::Off);
        assert!(!resize_only.vrr);
        assert!(resize_only.native_resize);

        // Explicit sub-overrides override the VRR default in both directions.
        assert_eq!(parse_swap_buffers_for(Some("3"), false), 3);
        assert_eq!(parse_swap_buffers_for(Some("2"), true), 2);
        assert!(tearing_requested_for(false, Some("1")));
        assert!(!tearing_requested_for(true, Some("0")));
        assert!(waitable_requested_for(false, Some("1")));
        assert!(!waitable_requested_for(true, Some("0")));
    }

    #[test]
    fn plan_present_always_pairs_swap_format_with_color_space() {
        assert!(!format_matches_color_space(FMT_R8G8B8A8_UNORM, ColorSpace::Hdr10Pq));
        assert!(!format_matches_color_space(FMT_R8G8B8A8_UNORM, ColorSpace::ScRgb));
        assert!(!format_matches_color_space(FMT_R10G10B10A2_UNORM, ColorSpace::ScRgb));
        assert!(format_matches_color_space(FMT_R10G10B10A2_UNORM, ColorSpace::Hdr10Pq));
        assert!(format_matches_color_space(FMT_R16G16B16A16_FLOAT, ColorSpace::ScRgb));

        let formats = [
            None,
            Some(FMT_R8G8B8A8_UNORM),
            Some(FMT_B8G8R8A8_UNORM),
            Some(FMT_R10G10B10A2_UNORM),
            Some(FMT_R16G16B16A16_FLOAT),
        ];
        let prefs = [
            HdrPreference::Off,
            HdrPreference::Auto,
            HdrPreference::Hdr10,
            HdrPreference::ScRgb,
        ];
        for native in formats {
            for has_overlay in [false, true] {
                for hdr_active in [false, true] {
                    for title_hdr in [false, true] {
                        for pref in prefs {
                            if let Some(plan) =
                                plan_present(native, has_overlay, hdr_active, title_hdr, pref)
                            {
                                assert!(
                                    format_matches_color_space(plan.swap_format, plan.output_space),
                                    "unpaired format={} space={:?} for native={native:?} overlay={has_overlay} hdr={hdr_active} title_hdr={title_hdr} pref={pref:?}",
                                    plan.swap_format,
                                    plan.output_space
                                );
                                if pref == HdrPreference::Off {
                                    assert_eq!(plan.output_space, ColorSpace::Sdr);
                                    if !has_overlay {
                                        assert_eq!(Some(plan.swap_format), native);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paper_white_clamps_to_safe_nits_range() {
        assert_eq!(parse_paper_white_nits(None), 200.0);
        assert_eq!(parse_paper_white_nits(Some("250")), 250.0);
        assert_eq!(parse_paper_white_nits(Some("40")), 80.0);
        assert_eq!(parse_paper_white_nits(Some("5000")), 1000.0);
        assert_eq!(parse_paper_white_nits(Some("NaN")), 200.0);
        assert_eq!(parse_paper_white_nits(Some("-10")), 200.0);
    }

    #[test]
    fn resolve_hdr_mode_reports_output_luminance_and_respects_off_override() {
        let hdr_caps = OutputHdrCaps {
            hdr_active: true,
            bits_per_color: 10,
            min_luminance: 0.005,
            max_luminance: 1000.0,
            max_full_frame_luminance: 450.0,
            red_primary: [0.68, 0.32],
            green_primary: [0.265, 0.69],
            blue_primary: [0.15, 0.06],
            white_point: [0.3127, 0.3290],
        };
        let decision = resolve_hdr_mode(&hdr_caps, 0, HdrPreference::Auto);
        assert!(decision.enabled);
        assert!((decision.min_tone_map_luminance - 0.005).abs() < 1e-6);
        assert!((decision.max_tone_map_luminance - 1000.0).abs() < 1e-4);
        assert!((decision.max_full_frame_tone_map_luminance - 450.0).abs() < 1e-4);

        let off = resolve_hdr_mode(&hdr_caps, 0, HdrPreference::Off);
        assert!(!off.enabled);

        let sdr = resolve_hdr_mode(&OutputHdrCaps::sdr_default(), 0, HdrPreference::Auto);
        assert!(!sdr.enabled);
    }

    #[test]
    fn hdr10_metadata_uses_exact_dxgi_units() {
        let caps = OutputHdrCaps {
            hdr_active: true,
            bits_per_color: 10,
            min_luminance: 0.005,
            max_luminance: 1000.0,
            max_full_frame_luminance: 400.0,
            red_primary: [0.708, 0.292],
            green_primary: [0.170, 0.797],
            blue_primary: [0.131, 0.046],
            white_point: [0.3127, 0.3290],
        };
        let m = build_hdr10_metadata_values(&caps);
        assert_eq!(m.red_primary, [35_400, 14_600]);
        assert_eq!(m.green_primary, [8_500, 39_850]);
        assert_eq!(m.blue_primary, [6_550, 2_300]);
        assert_eq!(m.white_point, [15_635, 16_450]);
        assert_eq!(m.max_mastering_luminance, 1000);
        assert_eq!(m.min_mastering_luminance, 50);
        assert_eq!(m.max_content_light_level, 1000);
        assert_eq!(m.max_frame_average_light_level, 400);
    }

    #[test]
    fn plan_present_preserves_hdr_with_overlay_and_falls_back_to_sdr() {
        // Dual-plane HDR10 PQ on HDR display must NOT downgrade to RGBA8 SDR.
        let pq_dual = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            true,
            true,
            true,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(pq_dual.swap_format, FMT_R10G10B10A2_UNORM);
        assert_eq!(pq_dual.scene_space, ColorSpace::Hdr10Pq);
        assert_eq!(pq_dual.output_space, ColorSpace::Hdr10Pq);
        assert!(pq_dual.needs_compose);

        // Single-plane HDR10 PQ on HDR display uses direct zero-copy path.
        let pq_single = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            false,
            true,
            true,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(pq_single.swap_format, FMT_R10G10B10A2_UNORM);
        assert!(!pq_single.needs_compose);

        // scRGB FP16 scene on HDR display preserves FP16 scRGB even with UI overlay.
        let scrgb_dual = plan_present(
            Some(FMT_R16G16B16A16_FLOAT),
            true,
            true,
            true,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(scrgb_dual.swap_format, FMT_R16G16B16A16_FLOAT);
        assert_eq!(scrgb_dual.scene_space, ColorSpace::ScRgb);
        assert_eq!(scrgb_dual.output_space, ColorSpace::ScRgb);
        assert!(scrgb_dual.needs_compose);

        // SDR display always falls back cleanly to SDR.
        let sdr_fallback = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            true,
            false,
            false,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(sdr_fallback.swap_format, FMT_R8G8B8A8_UNORM);
        assert_eq!(sdr_fallback.output_space, ColorSpace::Sdr);
        assert!(sdr_fallback.needs_compose);

        // 10-bit SDR title (`title_hdr == false`) on an HDR desktop must NOT be
        // misclassified as ST.2084 PQ BT.2020.
        let sdr10_auto_single = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            false,
            true,
            false,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(sdr10_auto_single.scene_space, ColorSpace::Sdr);
        assert_eq!(sdr10_auto_single.output_space, ColorSpace::Sdr);
        assert_eq!(sdr10_auto_single.swap_format, FMT_R10G10B10A2_UNORM);
        assert!(!sdr10_auto_single.needs_compose);

        let sdr10_auto_dual = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            true,
            true,
            false,
            HdrPreference::Auto,
        )
        .unwrap();
        assert_eq!(sdr10_auto_dual.scene_space, ColorSpace::Sdr);
        assert_eq!(sdr10_auto_dual.output_space, ColorSpace::Sdr);
        assert_eq!(sdr10_auto_dual.swap_format, FMT_R8G8B8A8_UNORM);
        assert!(sdr10_auto_dual.needs_compose);

        // Explicit HDR10 / scRGB overrides still convert 10-bit SDR cleanly via Composer.
        let sdr10_forced_pq = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            false,
            true,
            false,
            HdrPreference::Hdr10,
        )
        .unwrap();
        assert_eq!(sdr10_forced_pq.scene_space, ColorSpace::Sdr);
        assert_eq!(sdr10_forced_pq.output_space, ColorSpace::Hdr10Pq);
        assert_eq!(sdr10_forced_pq.swap_format, FMT_R10G10B10A2_UNORM);
        assert!(sdr10_forced_pq.needs_compose);

        let sdr10_forced_scrgb = plan_present(
            Some(FMT_R10G10B10A2_UNORM),
            false,
            true,
            false,
            HdrPreference::ScRgb,
        )
        .unwrap();
        assert_eq!(sdr10_forced_scrgb.scene_space, ColorSpace::Sdr);
        assert_eq!(sdr10_forced_scrgb.output_space, ColorSpace::ScRgb);
        assert_eq!(sdr10_forced_scrgb.swap_format, FMT_R16G16B16A16_FLOAT);
        assert!(sdr10_forced_scrgb.needs_compose);
    }

    #[test]
    fn present_flags_enforce_dxgi_tearing_invariants() {
        // Windowed + sync_interval 0 + tearing flag -> ALLOW_TEARING.
        assert_eq!(
            decide_present_flags(0, true, false, false),
            PresentFlagsDecision {
                sync_interval: 0,
                flags: PRESENT_ALLOW_TEARING_FLAG,
            }
        );
        // Non-zero sync interval -> never ALLOW_TEARING.
        assert_eq!(
            decide_present_flags(1, true, false, false),
            PresentFlagsDecision {
                sync_interval: 1,
                flags: 0,
            }
        );
        // Exclusive fullscreen -> never ALLOW_TEARING.
        assert_eq!(
            decide_present_flags(0, true, true, false),
            PresentFlagsDecision {
                sync_interval: 0,
                flags: 0,
            }
        );
        // Occlusion probe -> PRESENT_TEST only, never combined with ALLOW_TEARING.
        assert_eq!(
            decide_present_flags(0, true, false, true),
            PresentFlagsDecision {
                sync_interval: 0,
                flags: PRESENT_TEST_FLAG,
            }
        );
    }

    #[test]
    fn high_refresh_mode_selection_preserves_120_144_180_240_hz() {
        let raw = vec![
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 60_000, refresh_den: 1000 },
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 119_998, refresh_den: 1000 },
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 120_000, refresh_den: 1000 },
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 144_000, refresh_den: 1000 },
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 180_000, refresh_den: 1000 },
            RawDisplayMode { width: 2560, height: 1440, refresh_num: 240_000, refresh_den: 1000 },
        ];
        let modes = select_display_modes(&raw, None);
        let rates: Vec<u32> = modes.iter().map(RawDisplayMode::rounded_hz).collect();
        assert_eq!(rates, vec![60, 120, 144, 180, 240]);
        // Exact 120/1000 wins over 119.998/1000 during deduplication.
        assert_eq!(modes[1].refresh_num, 120_000);

        // Requesting 3840x2160 on a 1440p monitor maps its real refresh rates cleanly.
        let scaled = select_display_modes(&raw, Some((3840, 2160)));
        assert!(scaled.iter().all(|m| (m.width, m.height) == (3840, 2160)));
        let scaled_rates: Vec<u32> = scaled.iter().map(RawDisplayMode::rounded_hz).collect();
        assert_eq!(scaled_rates, vec![60, 120, 144, 180, 240]);
    }

    #[test]
    fn refresh_interval_math_covers_high_refresh_rates() {
        assert_eq!(refresh_interval_us(60, 1), Some(16_667));
        assert_eq!(refresh_interval_us(120, 1), Some(8_333));
        assert_eq!(refresh_interval_us(144, 1), Some(6_944));
        assert_eq!(refresh_interval_us(180, 1), Some(5_556));
        assert_eq!(refresh_interval_us(240, 1), Some(4_167));
        assert_eq!(refresh_interval_us(0, 1), None);
        assert_eq!(refresh_interval_us(120, 0), None);
    }

    #[test]
    fn resize_dpi_and_occlusion_state_machine_prevents_zero_resize_and_races() {
        let mut s = WindowResizeState::new();
        assert!(s.can_resize_buffers(1920, 1080));
        assert!(!s.can_resize_buffers(0, 1080));
        assert!(!s.should_skip_present());

        // Minimizing blocks ResizeBuffers and skips presenting cleanly.
        s.on_wm_size(SIZE_MINIMIZED_TYPE, 0, 0);
        assert!(s.should_skip_present());
        assert!(!s.can_resize_buffers(1920, 1080));
        assert!(!s.needs_occlusion_probe());

        // Restoring unblocks resize and flags a pending resize epoch.
        s.on_wm_size(0, 2560, 1440);
        assert!(!s.should_skip_present());
        assert!(s.can_resize_buffers(2560, 1440));
        assert!(s.has_pending_resize());
        s.mark_resized();
        assert!(!s.has_pending_resize());

        // SIZE_MAXSHOW (3) and SIZE_MAXHIDE (4) arrive with lParam == 0 on popup
        // windows when another desktop window maximizes/restores; never freeze!
        s.on_wm_size(3, 0, 0);
        assert!(!s.should_skip_present());
        assert!(s.can_resize_buffers(2560, 1440));
        assert_eq!(s.client_size, Some((2560, 1440)));
        s.on_wm_size(4, 0, 0);
        assert!(!s.should_skip_present());
        assert!(s.can_resize_buffers(2560, 1440));
        assert_eq!(s.client_size, Some((2560, 1440)));

        // DPI change triggers display reprobe without losing client size.
        s.on_wm_dpi_changed(144, 144);
        assert_eq!(s.dpi, (144, 144));
        assert!(s.needs_display_reprobe());
        s.mark_display_probed();
        assert!(!s.needs_display_reprobe());

        // Occlusion triggers PRESENT_TEST probing until cleared.
        s.on_present_occluded(true);
        assert!(s.needs_occlusion_probe());
        s.on_present_occluded(false);
        assert!(!s.needs_occlusion_probe());
    }

    #[cfg(windows)]
    #[test]
    fn swap_chain_tearing_waitable_resize_and_present_on_gpu() {
        use windows::Win32::Foundation::DXGI_STATUS_OCCLUDED;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        };
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory2,
        };
        use windows::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WINDOW_EX_STYLE, WS_POPUP,
        };
        use windows::core::w;

        let Some(host) = crate::host::host() else {
            eprintln!("skipping swap chain test: no D3D12 device");
            return;
        };
        if !native_resize_enabled() {
            assert_eq!(ensure_dpi_awareness(), DpiAwarenessOutcome::Disabled);
        }
        let dpi_outcome = ensure_dpi_awareness_now();
        assert_ne!(dpi_outcome, DpiAwarenessOutcome::Unavailable);

        let caps = probe_primary_hdr();
        let host_modes = enumerate_host_display_modes();
        eprintln!(
            "DXGI output probe: hdr_active={} bpc={} max_nits={} modes={}",
            caps.hdr_active,
            caps.bits_per_color,
            caps.max_luminance,
            host_modes.len()
        );

        // SAFETY: off-screen hidden popup window owned by this test thread.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE(0),
                w!("STATIC"),
                w!("victory hdr vrr test"),
                WS_POPUP,
                0,
                0,
                128,
                128,
                None,
                None,
                None,
                None,
            )
            .unwrap()
        };

        let factory: IDXGIFactory2 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap() };

        // Part A: baseline SDR control swap chain (VICTORY_HDR=0, VICTORY_VRR=0):
        // 2 buffers, Flags == 0, no tearing, no waitable object, SDR color space.
        let baseline = create_swap_chain_with_options(
            &factory,
            host.queue(),
            hwnd,
            128,
            128,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            ColorSpace::Sdr,
            &caps,
            SwapChainOptions {
                buffer_count: BASELINE_SWAP_BUFFERS,
                request_tearing: false,
                request_waitable: false,
                disable_alt_enter: false,
            },
        )
        .unwrap();
        let base_desc = unsafe { baseline.swap_chain.GetDesc1().unwrap() };
        assert_eq!(base_desc.BufferCount, BASELINE_SWAP_BUFFERS);
        assert_eq!(base_desc.Flags, 0);
        assert!(!baseline.tearing_enabled);
        assert!(baseline.waitable.is_none());
        assert_eq!(baseline.color_space, ColorSpace::Sdr);
        let base_hr = present_swap_chain(&baseline.swap_chain, false, 0, false);
        assert!(
            base_hr.is_ok() || base_hr == DXGI_STATUS_OCCLUDED,
            "baseline Present failed: {:#010x}",
            base_hr.0 as u32
        );
        host.drain_queue();
        drop(baseline);

        // Part B: opted-in VRR + HDR swap chain (3 buffers, tearing + waitable,
        // preserved flags across ResizeBuffers, HDR10 PQ + scRGB format pairing).
        let mut created = create_swap_chain_with_options(
            &factory,
            host.queue(),
            hwnd,
            128,
            128,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            ColorSpace::Sdr,
            &caps,
            SwapChainOptions {
                buffer_count: DEFAULT_SWAP_BUFFERS,
                request_tearing: true,
                request_waitable: true,
                disable_alt_enter: true,
            },
        )
        .unwrap();
        let init_desc = unsafe { created.swap_chain.GetDesc1().unwrap() };
        assert_eq!(init_desc.BufferCount, DEFAULT_SWAP_BUFFERS);
        assert_eq!(init_desc.Flags, created.flags.0 as u32);
        if created.waitable.is_some() {
            assert_eq!(
                unsafe { created.swap_chain.GetMaximumFrameLatency().unwrap() },
                MAX_FRAME_LATENCY
            );
        }
        eprintln!(
            "swap chain created: buffers={} tearing={} waitable={} space={:?}",
            created.buffer_count,
            created.tearing_enabled,
            created.waitable.is_some(),
            created.color_space
        );

        // Present twice using the waitable handle + legal tearing flags.
        for _ in 0..2 {
            if let Some(waitable) = created.waitable.as_mut() {
                let _ = waitable.wait_before_present();
            }
            let hr = present_swap_chain(&created.swap_chain, created.tearing_enabled, 0, false);
            if let Some(waitable) = created.waitable.as_mut() {
                waitable.note_presented();
            }
            assert!(
                hr.is_ok() || hr == DXGI_STATUS_OCCLUDED,
                "unexpected Present HRESULT {:#010x}",
                hr.0 as u32
            );
        }

        // Occlusion probe with PRESENT_TEST must never return DXGI_ERROR_INVALID_CALL.
        let probe_hr = present_swap_chain(&created.swap_chain, created.tearing_enabled, 0, true);
        assert!(
            probe_hr.is_ok() || probe_hr == DXGI_STATUS_OCCLUDED,
            "unexpected PRESENT_TEST HRESULT {:#010x}",
            probe_hr.0 as u32
        );

        // Drain the direct queue completely before ResizeBuffers so prior Presents finish.
        host.drain_queue();
        // SAFETY: no back buffers are held and the queue is idle.
        unsafe {
            created
                .swap_chain
                .ResizeBuffers(
                    created.buffer_count,
                    256,
                    192,
                    DXGI_FORMAT_R10G10B10A2_UNORM,
                    created.flags,
                )
                .expect("ResizeBuffers with preserved tearing/waitable flags must succeed");
        }
        let pq_desc = unsafe { created.swap_chain.GetDesc1().unwrap() };
        assert_eq!(
            (pq_desc.Width, pq_desc.Height, pq_desc.Format, pq_desc.Flags),
            (256, 192, DXGI_FORMAT_R10G10B10A2_UNORM, created.flags.0 as u32)
        );
        let applied_pq = apply_color_space_and_metadata(
            &created.swap_chain,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            ColorSpace::Hdr10Pq,
            &caps,
        );
        // On SDR/WARP displays DXGI may accept or fall back cleanly; either way
        // the active color space stays strictly paired with the swap format.
        assert!(matches!(applied_pq, ColorSpace::Hdr10Pq | ColorSpace::Sdr));
        assert!(format_matches_color_space(
            DXGI_FORMAT_R10G10B10A2_UNORM.0,
            applied_pq
        ));

        // Resize again to scRGB FP16 preserving flags.
        unsafe {
            created
                .swap_chain
                .ResizeBuffers(
                    created.buffer_count,
                    320,
                    240,
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    created.flags,
                )
                .expect("ResizeBuffers to R16G16B16A16_FLOAT must succeed");
        }
        let scrgb_desc = unsafe { created.swap_chain.GetDesc1().unwrap() };
        assert_eq!(
            (scrgb_desc.Width, scrgb_desc.Height, scrgb_desc.Format, scrgb_desc.Flags),
            (320, 240, DXGI_FORMAT_R16G16B16A16_FLOAT, created.flags.0 as u32)
        );
        let applied_scrgb = apply_color_space_and_metadata(
            &created.swap_chain,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            ColorSpace::ScRgb,
            &caps,
        );
        assert!(matches!(applied_scrgb, ColorSpace::ScRgb | ColorSpace::Sdr));
        assert!(format_matches_color_space(
            DXGI_FORMAT_R16G16B16A16_FLOAT.0,
            applied_scrgb
        ));

        // Present once more after resizing and verify zero device errors.
        if let Some(waitable) = created.waitable.as_mut() {
            let _ = waitable.wait_before_present();
        }
        let post_hr = present_swap_chain(&created.swap_chain, created.tearing_enabled, 0, false);
        if let Some(waitable) = created.waitable.as_mut() {
            waitable.note_presented();
        }
        assert!(
            post_hr.is_ok() || post_hr == DXGI_STATUS_OCCLUDED,
            "post-resize Present failed: {:#010x}",
            post_hr.0 as u32
        );

        host.drain_queue();
        drop(created);
        // SAFETY: test window owned by this thread.
        unsafe {
            let _ = DestroyWindow(hwnd);
            host.device.GetDeviceRemovedReason().unwrap();
        }
        assert_eq!(host.log_messages(), None, "D3D12 debug layer reported an error");
    }
}
