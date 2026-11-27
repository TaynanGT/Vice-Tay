//! NVIDIA DLSS Frame Generation / Multi Frame Generation through Streamline.
//! Runtime/proxy/ABI portions adapted from trustno1-coder/VIctory-NVIDIA
//! c437b0594bde23780e15c24969258ad9d70bb7be (GPL-3.0-or-later), itself
//! based on readmet3xt-lab/foxhound. Existing NGX upscaling stays independent.
//! NVIDIA DLLs are loaded only from an explicitly configured external folder;
//! no downloads, machine-global settings, or OTA updates occur here.
#![allow(dead_code)] // Public Streamline ABI mirrors optional SDK features.
pub(crate) mod abi;
pub mod plan;

pub use plan::Mode;

use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const INTERPOSER: &str = "sl.interposer.dll";
/// The project id NGX is given (engine "custom", no NVIDIA application id).
const PROJECT_ID: &CStr = c"6f0d7b2e-3a41-4c1e-9b57-2d8e4f1a9c30";
const ENGINE_VERSION: &CStr = c"VIctory 0.1";
/// The viewport of the upscaling; frame generation wants 0.
const VIEWPORT_DLSS: u32 = 1;
const VIEWPORT_FG: u32 = 0;

/// What the user asked for, read from the environment once.
#[derive(Clone, Debug)]
pub struct Asked {
    pub dlss: bool,
    pub framegen: bool,
    pub mode: Mode,
    pub preset: u32,
    pub dir: PathBuf,
    /// `VICTORY_DLSS_JITTER` and `VICTORY_DLSS_MV`: sign or scale fixes.
    pub jitter: [f32; 2],
    pub mvec: [f32; 2],
    /// `VICTORY_DLSS_REACTIVE=0`: the reactive mask is not given to DLSS.
    pub reactive: bool,
    /// `VICTORY_STREAMLINE_LOG=<folder>`: Streamline's own log there, verbose.
    pub log_dir: Option<PathBuf>,
    /// `VICTORY_FRAMEGEN_HUDLESS=0`: with frame generation, no colour
    /// without the interface is copied or tagged where the title draws its
    /// interface into the frame (Series X).
    pub hudless: bool,
    /// `VICTORY_STREAMLINE_PRESENT=1`: with DLSS and no frame generation,
    /// our swap chain is Streamline's proxy of it, so that Streamline sees
    /// the presents (its bookkeeping per frame).
    pub present_hook: bool,
}

impl Asked {
    /// The switches; an unknown mode or preset name is an error.
    pub fn from_env() -> Result<Self, String> {
        let var = |name: &str| std::env::var(name).ok();
        let framegen = requested();
        Ok(Asked {
            dlss: false,
            framegen,
            mode: Mode::Dlaa,
            preset: 0,
            dir: var("VICTORY_STREAMLINE_DIR")
                .filter(|p| !p.trim().is_empty())
                .map(PathBuf::from)
                .ok_or("set VICTORY_STREAMLINE_DIR to the signed NVIDIA SDK runtime folder")?,
            jitter: plan::pair(var("VICTORY_DLSS_JITTER").as_deref()),
            mvec: plan::pair(var("VICTORY_DLSS_MV").as_deref()),
            reactive: false,
            log_dir: var("VICTORY_STREAMLINE_LOG")
                .filter(|v| !v.trim().is_empty() && v.trim() != "0")
                .map(PathBuf::from),
            hudless: true,
            present_hook: false,
        })
    }

    pub fn any(&self) -> bool {
        self.dlss || self.framegen
    }
}

/// What [`init`] made available: the upscaling (the title's renderer may be
/// patched to the FidelityFX path for it) and frame generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Available {
    pub dlss: bool,
    pub framegen: bool,
}

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

struct Library(windows_sys::Win32::Foundation::HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this owns the reference LoadLibraryExW took.
        unsafe { windows_sys::Win32::Foundation::FreeLibrary(self.0) };
    }
}

// SAFETY: a module handle.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// The library's exported functions.
struct Sl {
    _library: Library,
    init: abi::SlInit,
    is_feature_supported: abi::SlIsFeatureSupported,
    evaluate_feature: abi::SlEvaluateFeature,
    free_resources: abi::SlFreeResources,
    set_tag_for_frame: abi::SlSetTagForFrame,
    get_feature_requirements: abi::SlGetFeatureRequirements,
    get_feature_version: abi::SlGetFeatureVersion,
    upgrade_interface: abi::SlUpgradeInterface,
    set_constants: abi::SlSetConstants,
    get_native_interface: abi::SlGetNativeInterface,
    get_feature_function: abi::SlGetFeatureFunction,
    get_new_frame_token: abi::SlGetNewFrameToken,
    set_d3d_device: abi::SlSetD3DDevice,
    shutdown: abi::SlShutdown,
}

static SL: OnceLock<Sl> = OnceLock::new();

/// What is on once the library is in: the switches and the features found.
struct Plan {
    asked: Asked,
    dlss: bool,
    framegen: bool,
}

static PLAN: OnceLock<Plan> = OnceLock::new();

/// The per-feature functions, once the device is set.
#[derive(Default)]
struct Functions {
    dlss_optimal: Option<abi::SlDlssGetOptimalSettings>,
    dlss_set_options: Option<abi::SlDlssSetOptions>,
    dlssg_set_options: Option<abi::SlDlssgSetOptions>,
    dlssg_get_state: Option<abi::SlDlssgGetState>,
    reflex_set_options: Option<abi::SlReflexSetOptions>,
    pcl_set_marker: Option<abi::SlPclSetMarker>,
}

static FUNCTIONS: OnceLock<Functions> = OnceLock::new();
static DLSS_READY: AtomicBool = AtomicBool::new(false);
static FG_READY: AtomicBool = AtomicBool::new(false);

/// Streamline's proxies kept for frame generation: the device that made the
/// direct queue and the queue the swap chain is made on.
struct Proxies {
    _device: ID3D12Device,
    queue: ID3D12CommandQueue,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Proxies {}
// SAFETY: as above.
unsafe impl Sync for Proxies {}

static PROXIES: OnceLock<Proxies> = OnceLock::new();
/// Our present goes through Streamline's swap chain.
static SWAP_CHAIN_PROXIED: AtomicBool = AtomicBool::new(false);
/// Our swap chain is Streamline's proxy of it ([`present_hooked`]).
static PRESENT_HOOKED: AtomicBool = AtomicBool::new(false);

/// Library calls that are not thread safe, and what the upscaling keeps
/// between frames, one caller at a time.
#[derive(Default)]
struct Calls {
    /// Optimal render sizes by output size and mode.
    optimal: Vec<([u32; 2], u32, Option<[u32; 2]>)>,
    /// The mode, sizes and preset the last options named.
    last: Option<(u32, [u32; 2], [u32; 2])>,
    /// Frame generation: on, the frames for each rendered one last asked of it (0 off), and
    /// the frame whose simulation began.
    framegen_on: bool,
    framegen_times: u32,
    simulating: Option<u32>,
}

static CALLS: Mutex<Calls> = Mutex::new(Calls {
    optimal: Vec::new(),
    last: None,
    framegen_on: false,
    framegen_times: 0,
    simulating: None,
});

fn calls() -> std::sync::MutexGuard<'static, Calls> {
    CALLS.lock().unwrap_or_else(|e| e.into_inner())
}

// Counters since the last frame-rate line: upscalings done by DLSS, failed,
// left to the title; presents with frame generation on and off, frames it
// reported shown; and in all.
static EVALUATED: AtomicU64 = AtomicU64::new(0);
static FAILED: AtomicU64 = AtomicU64::new(0);
static FORWARDED: AtomicU64 = AtomicU64::new(0);
static FG_ON: AtomicU64 = AtomicU64::new(0);
static FG_OFF: AtomicU64 = AtomicU64::new(0);
static FG_SHOWN: AtomicU64 = AtomicU64::new(0);
static EVALUATED_ALL: AtomicU64 = AtomicU64::new(0);
static FG_STATUS: AtomicU32 = AtomicU32::new(0);
/// Presents with frame generation on, and frames it showed, in all.
static FG_ON_ALL: AtomicU64 = AtomicU64::new(0);
static FG_SHOWN_ALL: AtomicU64 = AtomicU64::new(0);
/// Upscalings noted so far (their ids).
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
/// [`check`] presents from its one thread, with nothing else submitting.
static CHECKING: AtomicBool = AtomicBool::new(false);

/// Loads Streamline from `asked.dir` and asks for what `asked` names; call
/// once, before the host device is made (`crate::host::host`) and before the
/// title's imports are bound. Fails soft: what is not there is logged and
/// left off.
pub fn init(asked: &Asked) -> Available {
    if !asked.any() {
        return Available::default();
    }
    match load_and_init(asked, true) {
        Ok(available) => available,
        Err(why) => {
            tracing::warn!(
                folder = %asked.dir.display(),
                "Streamline: {why}; DLSS and frame generation stay off, the title's own upscaling and our present run"
            );
            Available::default()
        }
    }
}

fn load_and_init(asked: &Asked, require_line: bool) -> Result<Available, String> {
    // SDK logging and generated runtime data obey the same asset policy as
    // host caches, including when the user leaves logging at its default.
    victory_core::ensure_outside_repo(&asked.dir).map_err(|e| e.to_string())?;
    let log_dir = asked.log_dir.as_deref().unwrap_or(&asked.dir);
    victory_core::ensure_outside_repo(log_dir).map_err(|e| e.to_string())?;
    let path = std::path::absolute(asked.dir.join(INTERPOSER)).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!(
            "no {INTERPOSER} in that folder (VICTORY_STREAMLINE_DIR)"
        ));
    }
    verify_signature(&path)?;
    let sl = load(&path)?;
    let sl = SL.get_or_init(|| sl);
    let dir_wide: Vec<u16> = wide(&asked.dir);
    let paths = [dir_wide.as_ptr()];
    let log_wide = wide(log_dir);
    // Frame generation presents from the line's thread, in order with the
    // submissions: without the line it is not loaded at all.
    let framegen_asked = asked.framegen && (!require_line || line_asked());
    if asked.framegen && !framegen_asked {
        tracing::warn!(
            "DLSS Frame Generation needs the line of submissions (VICTORY_THIN=1 or VICTORY_ASYNC_SUBMIT=1): frame generation stays off"
        );
    }
    let mut features = Vec::new();
    if asked.dlss {
        features.push(abi::FEATURE_DLSS);
    }
    if framegen_asked {
        features.extend([abi::FEATURE_DLSS_G, abi::FEATURE_REFLEX, abi::FEATURE_PCL]);
    }
    let preferences = abi::Preferences {
        base: abi::BaseStructure::new(abi::PREFERENCES, 1),
        show_console: false,
        log_level: if asked.log_dir.is_some() {
            abi::LOG_VERBOSE
        } else {
            abi::LOG_DEFAULT
        },
        paths_to_plugins: paths.as_ptr(),
        num_paths_to_plugins: 1,
        path_to_logs_and_data: log_wide.as_ptr(),
        allocate_callback: std::ptr::null(),
        release_callback: std::ptr::null(),
        log_message_callback: Some(log_message),
        // No downloads (OTA): only what is in the folder runs.
        flags: abi::DISABLE_CL_STATE_TRACKING
            | abi::DISABLE_DEBUG_TEXT
            | abi::USE_MANUAL_HOOKING
            | abi::USE_DXGI_FACTORY_PROXY
            | abi::USE_FRAME_BASED_RESOURCE_TAGGING,
        features_to_load: features.as_ptr(),
        num_features_to_load: features.len() as u32,
        application_id: 0,
        engine: abi::ENGINE_CUSTOM,
        engine_version: ENGINE_VERSION.as_ptr(),
        project_id: PROJECT_ID.as_ptr(),
        render_api: abi::RENDER_API_D3D12,
    };
    // SAFETY: a filled preferences structure; the library copies what it keeps.
    let code = unsafe { (sl.init)(&preferences, abi::SDK_VERSION) };
    if code != 0 {
        return Err(format!(
            "slInit failed: {} ({code})",
            abi::result_name(code)
        ));
    }
    let adapter = default_adapter();
    let luid = ADAPTER_LUID
        .get()
        .copied()
        .or_else(|| adapter.as_ref().map(|a| a.1));
    let supported = |feature: u32| -> Result<(), String> {
        let Some(luid) = luid else {
            return Err("no adapter".into());
        };
        let info = abi::AdapterInfo {
            base: abi::BaseStructure::new(abi::ADAPTER_INFO, 1),
            device_luid: std::ptr::from_ref(&luid).cast(),
            device_luid_size: size_of_val(&luid) as u32,
            vk_physical_device: std::ptr::null_mut(),
        };
        // SAFETY: an adapter description that lives through the call.
        let code = unsafe { (sl.is_feature_supported)(feature, &info) };
        if code == 0 {
            Ok(())
        } else {
            Err(format!("{} ({code})", abi::result_name(code)))
        }
    };
    let dlss = asked.dlss
        && report(
            "DLSS Super Resolution",
            abi::FEATURE_DLSS,
            supported(abi::FEATURE_DLSS),
            sl,
        );
    let framegen = framegen_asked
        && report(
            "DLSS Frame Generation",
            abi::FEATURE_DLSS_G,
            supported(abi::FEATURE_DLSS_G),
            sl,
        )
        && report(
            "Reflex",
            abi::FEATURE_REFLEX,
            supported(abi::FEATURE_REFLEX),
            sl,
        );
    tracing::info!(
        library = %path.display(),
        sdk = "2.14.16",
        adapter = adapter.as_ref().map_or("?", |a| a.0.as_str()),
        dlss,
        framegen,
        mode = asked.mode.name(),
        preset = asked.preset,
        "Streamline loaded (manual hooking, no downloads)"
    );
    let _ = PLAN.set(Plan {
        asked: asked.clone(),
        dlss,
        framegen,
    });
    Ok(Available { dlss, framegen })
}

/// Logs whether a feature is supported, with its requirements; whether it is.
fn report(name: &str, feature: u32, supported: Result<(), String>, sl: &Sl) -> bool {
    let mut requirements: abi::FeatureRequirements = unsafe_zeroed_requirements();
    // SAFETY: a structure to fill.
    let asked = unsafe { (sl.get_feature_requirements)(feature, &mut requirements) };
    let needs = (asked == 0).then(|| {
        format!(
            "OS {} (needs {}), driver {} (needs {}){}{}",
            requirements.os_version_detected,
            requirements.os_version_required,
            requirements.driver_version_detected,
            requirements.driver_version_required,
            if requirements.flags & abi::REQUIRES_HARDWARE_SCHEDULING != 0 {
                ", hardware GPU scheduling"
            } else {
                ""
            },
            if requirements.flags & abi::REQUIRES_VSYNC_OFF != 0 {
                ", v-sync off"
            } else {
                ""
            },
        )
    });
    match supported {
        Ok(()) => {
            tracing::info!(
                requirements = needs.as_deref().unwrap_or("?"),
                "Streamline: {name} is supported on this GPU"
            );
            true
        }
        Err(why) => {
            tracing::warn!(
                requirements = needs.as_deref().unwrap_or("?"),
                "Streamline: {name} is not supported here: {why}"
            );
            false
        }
    }
}

fn unsafe_zeroed_requirements() -> abi::FeatureRequirements {
    // SAFETY: plain data; pointers null, counts zero.
    let mut r: abi::FeatureRequirements = unsafe { std::mem::zeroed() };
    r.base = abi::BaseStructure::new(abi::FEATURE_REQUIREMENTS, 2);
    r
}

/// Whether the switches turn the line of submissions on (frame generation
/// presents from its thread, in order with the submissions).
fn line_asked() -> bool {
    let on = |name: &str| std::env::var_os(name).is_some_and(|v| v == "1");
    on("VICTORY_THIN") || on("VICTORY_ASYNC_SUBMIT")
}

unsafe extern "C" fn log_message(kind: u32, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: a NUL-terminated message for the call.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let text = text.trim_end();
    // (Every plugin in the folder that was not asked for says so.)
    let noise = text.contains("since it is was not requested by the host");
    match kind {
        abi::LOG_ERROR => tracing::error!(target: "streamline", "{text}"),
        abi::LOG_WARN if !noise => tracing::warn!(target: "streamline", "{text}"),
        _ => tracing::debug!(target: "streamline", "{text}"),
    }
}

fn wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain([0]).collect()
}

/// The adapter a device made with no adapter named is on (the first), its
/// name and LUID.
fn default_adapter() -> Option<(String, windows::Win32::Foundation::LUID)> {
    // SAFETY: plain DXGI queries.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let adapter = factory.EnumAdapters1(0).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        let len = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        Some((
            String::from_utf16_lossy(&desc.Description[..len]),
            desc.AdapterLuid,
        ))
    }
}

/// The Authenticode signature of `path` (NVIDIA signs its DLLs), checked
/// without going online.
fn verify_signature(path: &Path) -> Result<(), String> {
    use windows_sys::Win32::Security::WinTrust::*;
    let file = wide(path);
    let mut info = WINTRUST_FILE_INFO {
        cbStruct: size_of::<WINTRUST_FILE_INFO>() as u32,
        pcwszFilePath: file.as_ptr(),
        hFile: std::ptr::null_mut(),
        pgKnownSubject: std::ptr::null_mut(),
    };
    let mut data = WINTRUST_DATA {
        cbStruct: size_of::<WINTRUST_DATA>() as u32,
        dwUIChoice: WTD_UI_NONE,
        fdwRevocationChecks: WTD_REVOKE_NONE,
        dwUnionChoice: WTD_CHOICE_FILE,
        Anonymous: WINTRUST_DATA_0 { pFile: &mut info },
        dwStateAction: WTD_STATEACTION_VERIFY,
        dwProvFlags: WTD_CACHE_ONLY_URL_RETRIEVAL | WTD_REVOCATION_CHECK_NONE,
        ..Default::default()
    };
    let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    // SAFETY: filled structures that live through both calls.
    let code = unsafe { WinVerifyTrust(std::ptr::null_mut(), &mut action, (&raw mut data).cast()) };
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    // SAFETY: as above; closes the state the first call opened.
    unsafe { WinVerifyTrust(std::ptr::null_mut(), &mut action, (&raw mut data).cast()) };
    if code == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} is not validly signed ({:#x})",
            path.display(),
            code as u32
        ))
    }
}

fn load(path: &Path) -> Result<Sl, String> {
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    let file = wide(path);
    // Its own folder and System32 only, never the current folder.
    // SAFETY: a NUL-terminated full path.
    let module = unsafe {
        LoadLibraryExW(
            file.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if module.is_null() {
        return Err(format!(
            "{INTERPOSER} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let library = Library(module);
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{INTERPOSER} has no {}", name.to_string_lossy()))
    };
    macro_rules! get {
        ($name:literal, $t:ty) => {
            // SAFETY: the export of that name has the header's signature.
            unsafe { std::mem::transmute::<*const c_void, $t>(find($name)?) }
        };
    }
    Ok(Sl {
        init: get!(c"slInit", abi::SlInit),
        is_feature_supported: get!(c"slIsFeatureSupported", abi::SlIsFeatureSupported),
        evaluate_feature: get!(c"slEvaluateFeature", abi::SlEvaluateFeature),
        free_resources: get!(c"slFreeResources", abi::SlFreeResources),
        set_tag_for_frame: get!(c"slSetTagForFrame", abi::SlSetTagForFrame),
        get_feature_requirements: get!(c"slGetFeatureRequirements", abi::SlGetFeatureRequirements),
        get_feature_version: get!(c"slGetFeatureVersion", abi::SlGetFeatureVersion),
        upgrade_interface: get!(c"slUpgradeInterface", abi::SlUpgradeInterface),
        set_constants: get!(c"slSetConstants", abi::SlSetConstants),
        get_native_interface: get!(c"slGetNativeInterface", abi::SlGetNativeInterface),
        get_feature_function: get!(c"slGetFeatureFunction", abi::SlGetFeatureFunction),
        get_new_frame_token: get!(c"slGetNewFrameToken", abi::SlGetNewFrameToken),
        set_d3d_device: get!(c"slSetD3DDevice", abi::SlSetD3DDevice),
        shutdown: get!(c"slShutdown", abi::SlShutdown),
        _library: library,
    })
}

/// Whether the title's FidelityFX calls are to be served (DLSS or frame
/// generation is on): `victory_hle` then puts its own in front of them.
pub fn serving() -> bool {
    PLAN.get().is_some_and(|p| p.dlss || p.framegen)
}

/// Whether DLSS is to take the title's upscale dispatches (`VICTORY_DLSS=1`
/// and Streamline loaded).
pub fn upscaling() -> bool {
    PLAN.get().is_some_and(|p| p.dlss)
}

/// The host device is made: Streamline is given it (`slSetD3DDevice`), the
/// feature functions are fetched, and, for frame generation, the direct
/// queue is made through Streamline's device proxy. The native queue to use
/// for everything, if it was made here.
pub(crate) fn device_made(device: &ID3D12Device) -> Option<ID3D12CommandQueue> {
    if !requested() {
        return None;
    }
    let asked = Asked::from_env()
        .map_err(|e| tracing::warn!(%e, "NVIDIA frame generation configuration invalid"))
        .ok()?;
    // SAFETY: plain query on the host device being handed over.
    let _ = ADAPTER_LUID.set(unsafe { device.GetAdapterLuid() });
    init(&asked);
    let plan = PLAN.get()?;
    let sl = SL.get()?;
    // SAFETY: the native device.
    let code = unsafe { (sl.set_d3d_device)(device.as_raw()) };
    if code != 0 {
        tracing::warn!(
            "Streamline: slSetD3DDevice failed: {} ({code}); DLSS and frame generation stay off",
            abi::result_name(code)
        );
        return None;
    }
    let function = |feature: u32, name: &CStr| -> Option<*mut c_void> {
        let mut f = std::ptr::null_mut();
        // SAFETY: a NUL-terminated name and an out pointer.
        let code = unsafe { (sl.get_feature_function)(feature, name.as_ptr(), &mut f) };
        if code != 0 || f.is_null() {
            tracing::warn!(
                "Streamline: no {} ({}, {code})",
                name.to_string_lossy(),
                abi::result_name(code)
            );
            return None;
        }
        Some(f)
    };
    macro_rules! f {
        ($feature:expr, $name:literal, $t:ty) => {
            // SAFETY: the function of that name has the header's signature.
            function($feature, $name).map(|f| unsafe { std::mem::transmute::<*mut c_void, $t>(f) })
        };
    }
    let mut functions = Functions::default();
    if plan.dlss {
        functions.dlss_optimal = f!(
            abi::FEATURE_DLSS,
            c"slDLSSGetOptimalSettings",
            abi::SlDlssGetOptimalSettings
        );
        functions.dlss_set_options = f!(
            abi::FEATURE_DLSS,
            c"slDLSSSetOptions",
            abi::SlDlssSetOptions
        );
        let ready = functions.dlss_optimal.is_some() && functions.dlss_set_options.is_some();
        log_version(sl, abi::FEATURE_DLSS, "DLSS Super Resolution");
        DLSS_READY.store(ready, Release);
    }
    let mut queue = None;
    if plan.framegen {
        functions.dlssg_set_options = f!(
            abi::FEATURE_DLSS_G,
            c"slDLSSGSetOptions",
            abi::SlDlssgSetOptions
        );
        functions.dlssg_get_state = f!(
            abi::FEATURE_DLSS_G,
            c"slDLSSGGetState",
            abi::SlDlssgGetState
        );
        functions.reflex_set_options = f!(
            abi::FEATURE_REFLEX,
            c"slReflexSetOptions",
            abi::SlReflexSetOptions
        );
        functions.pcl_set_marker = f!(abi::FEATURE_PCL, c"slPCLSetMarker", abi::SlPclSetMarker);
        log_version(sl, abi::FEATURE_DLSS_G, "DLSS Frame Generation");
        log_version(sl, abi::FEATURE_REFLEX, "Reflex");
        let have = functions.dlssg_set_options.is_some()
            && functions.dlssg_get_state.is_some()
            && functions.reflex_set_options.is_some()
            && functions.pcl_set_marker.is_some();
        if have {
            if let Some(reflex) = functions.reflex_set_options {
                let options = abi::ReflexOptions {
                    base: abi::BaseStructure::new(abi::REFLEX_OPTIONS, 1),
                    mode: abi::REFLEX_LOW_LATENCY,
                    frame_limit_us: 0,
                    use_markers_to_optimize: false,
                    virtual_key: 0,
                    id_thread: 0,
                };
                // SAFETY: filled options.
                let code = unsafe { reflex(&options) };
                tracing::info!(
                    code,
                    result = abi::result_name(code),
                    "Streamline: Reflex low latency on (needed by frame generation)"
                );
            }
            queue = proxied_queue(sl, device);
        }
        FG_READY.store(have && queue.is_some(), Release);
        if !(have && queue.is_some()) {
            tracing::warn!(
                "DLSS Frame Generation stays off: the device hand-over to Streamline did not complete"
            );
        }
    }
    let _ = FUNCTIONS.set(functions);
    queue
}

fn log_version(sl: &Sl, feature: u32, name: &str) {
    let mut version = abi::FeatureVersion {
        base: abi::BaseStructure::new(abi::FEATURE_VERSION, 1),
        version_sl: abi::Version::default(),
        version_ngx: abi::Version::default(),
    };
    // SAFETY: a structure to fill.
    let code = unsafe { (sl.get_feature_version)(feature, &mut version) };
    if code == 0 {
        tracing::info!(streamline = %version.version_sl, ngx = %version.version_ngx, "Streamline: {name} loaded");
    } else {
        tracing::warn!(
            "Streamline: {name} has no version: {} ({code})",
            abi::result_name(code)
        );
    }
}

/// The direct queue, made through Streamline's proxy of `device` (frame
/// generation hooks queue creation); the native queue is returned for our
/// use and the proxy kept for the swap chain.
fn proxied_queue(sl: &Sl, device: &ID3D12Device) -> Option<ID3D12CommandQueue> {
    let mut raw = device.as_raw();
    // SAFETY: an owned reference of ours; on success `raw` is a new proxy
    // that holds its own reference to the device.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == device.as_raw() {
        tracing::warn!(
            "Streamline: the device was not upgraded: {} ({code})",
            abi::result_name(code)
        );
        return None;
    }
    // SAFETY: the proxy's one reference, now ours.
    let proxy_device = unsafe { ID3D12Device::from_raw(raw) };
    let desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };
    // SAFETY: plain queue creation through the proxy.
    let proxy_queue: ID3D12CommandQueue = match unsafe { proxy_device.CreateCommandQueue(&desc) } {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!(error = %e, "Streamline: no queue from the device proxy");
            return None;
        }
    };
    let mut native = std::ptr::null_mut();
    // SAFETY: a proxy queue; the native one comes back with a reference.
    let code = unsafe { (sl.get_native_interface)(proxy_queue.as_raw(), &mut native) };
    if code != 0 || native.is_null() {
        tracing::warn!(
            "Streamline: no native queue behind the proxy: {} ({code})",
            abi::result_name(code)
        );
        return None;
    }
    // SAFETY: the reference slGetNativeInterface added.
    let native = unsafe { ID3D12CommandQueue::from_raw(native) };
    tracing::info!(
        "Streamline: the direct queue was made through the device proxy (frame generation)"
    );
    let _ = PROXIES.set(Proxies {
        _device: proxy_device,
        queue: proxy_queue,
    });
    Some(native)
}

/// For frame generation: Streamline's proxy of `factory` and the queue
/// proxy to make the swap chain with; `None` to make it as before.
pub(crate) fn swap_chain_factory(
    factory: &IDXGIFactory2,
) -> Option<(IDXGIFactory2, ID3D12CommandQueue)> {
    if !FG_READY.load(Acquire) {
        return None;
    }
    let sl = SL.get()?;
    let queue = PROXIES.get()?.queue.clone();
    let mut raw = factory.as_raw();
    // SAFETY: as for the device: a new proxy with its own reference.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == factory.as_raw() {
        tracing::warn!(
            "Streamline: the DXGI factory was not upgraded: {} ({code})",
            abi::result_name(code)
        );
        FG_READY.store(false, Release);
        return None;
    }
    // SAFETY: the proxy's reference, now ours.
    let proxy = unsafe { IDXGIFactory2::from_raw(raw) };
    Some((proxy, queue))
}

/// The swap chain was made through the proxy (or not, after all: then
/// frame generation stays off).
pub(crate) fn swap_chain_made(proxied: bool, error: Option<&windows::core::Error>) {
    SWAP_CHAIN_PROXIED.store(proxied, Release);

    if proxied {
        tracing::info!(
            "Streamline: our swap chain is Streamline's proxy; DLSS Frame Generation can run on its presents"
        );
    } else {
        FG_READY.store(false, Release);
        tracing::warn!(
            error = error.map(|e| e.to_string()).unwrap_or_default(),
            "Streamline: the swap chain could not be made through the proxy; made as before, frame generation off"
        );
    }
}

/// `VICTORY_STREAMLINE_PRESENT=1`, DLSS without frame generation: Streamline's
/// proxy of `swap_chain`, whose Present, ResizeBuffers and GetBuffer run
/// Streamline's hooks first (its common plugin frees what it retired and
/// keeps its video memory budget at each present; without it, it logs
/// "presentCommon() was not observed" after 60 upscalings and frees nothing
/// it retired). `swap_chain` itself if the switch is off, frame generation
/// owns the swap chain, or the upgrade fails.
pub(crate) fn present_hooked(swap_chain: IDXGISwapChain3) -> IDXGISwapChain3 {
    let (Some(plan), Some(sl)) = (PLAN.get(), SL.get()) else {
        return swap_chain;
    };
    if !plan.asked.present_hook
        || !plan.dlss
        || !DLSS_READY.load(Acquire)
        || SWAP_CHAIN_PROXIED.load(Acquire)
    {
        return swap_chain;
    }
    let mut raw = swap_chain.as_raw();
    // SAFETY: an owned reference of ours; on success `raw` is a new proxy
    // that holds its own reference to the swap chain.
    let code = unsafe { (sl.upgrade_interface)(&mut raw) };
    if code != 0 || raw.is_null() || raw == swap_chain.as_raw() {
        tracing::warn!(
            "Streamline: the swap chain was not upgraded: {} ({code}); presents stay unseen by Streamline",
            abi::result_name(code)
        );
        return swap_chain;
    }
    tracing::info!(
        "Streamline: our swap chain is Streamline's proxy of it (VICTORY_STREAMLINE_PRESENT=1): it sees every present"
    );
    PRESENT_HOOKED.store(true, Release);
    // SAFETY: the proxy's one reference, now ours; it implements the swap
    // chain interfaces up to IDXGISwapChain4 and holds the swap chain.
    unsafe { IDXGISwapChain3::from_raw(raw) }
}

fn frame_token(sl: &Sl, index: u32) -> Option<*mut abi::FrameToken> {
    let mut token = std::ptr::null_mut();
    // SAFETY: an out pointer and an index.
    let code = unsafe { (sl.get_new_frame_token)(&mut token, &index) };
    (code == 0 && !token.is_null()).then_some(token)
}

static ADAPTER_LUID: OnceLock<windows::Win32::Foundation::LUID> = OnceLock::new();
static CURRENT_TOKEN: AtomicU64 = AtomicU64::new(u64::MAX);
const COPY_DEST: u32 = 0x400;

pub(crate) fn requested() -> bool {
    std::env::var_os("VICTORY_FRAMEGEN").is_some_and(|v| v == "1")
        && std::env::var_os("VICTORY_FRAMEGEN_PROVIDER").is_some_and(|v| v == "nvidia")
}

#[derive(Clone)]
enum Source {
    Host(Arc<crate::resource::HostTexture>),
    Fixed(ID3D12Resource, u32),
}
impl Source {
    fn tagged(&self) -> abi::Resource {
        match self {
            Source::Host(t) => abi::Resource::texture(t.texture.as_raw(), t.state.load(Relaxed)),
            Source::Fixed(t, state) => abi::Resource::texture(t.as_raw(), *state),
        }
    }
}
// SAFETY: retained D3D12 resources are free-threaded and host state is atomic.
unsafe impl Send for Source {}
// SAFETY: as above; the pointed-to state is synchronized by the queue and atomics.
unsafe impl Sync for Source {}

const UNBOUND_TOKEN: u64 = u64::MAX;

/// Submission identity is kept separate from GPU resources so discarded
/// recordings can be pruned without retaining their textures.
struct Submission {
    id: u64,
    list: usize,
    token: AtomicU64,
}
impl Submission {
    fn bind(&self, lists: &[crate::object::This], token: u64) {
        if lists.iter().any(|&list| list as usize == self.list) {
            let _ = self
                .token
                .compare_exchange(UNBOUND_TOKEN, token, AcqRel, Acquire);
        }
    }

    fn matches_present(&self, last_presented: u64, token: u64) -> bool {
        let bound = self.token.load(Acquire);
        self.id > last_presented && bound != UNBOUND_TOKEN && bound == token
    }
}

pub(crate) struct Preparation {
    submission: Arc<Submission>,
    depth: Source,
    motion: Source,
    camera: plan::Camera,
}
static PENDING: Mutex<Vec<std::sync::Weak<Submission>>> = Mutex::new(Vec::new());
static SUBMITTED: Mutex<Option<Arc<Preparation>>> = Mutex::new(None);
static PRESENTED_ID: AtomicU64 = AtomicU64::new(0);
impl Preparation {
    pub(crate) fn submitted(self: &Arc<Self>) {
        let mut latest = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
        if self.submission.id <= PRESENTED_ID.load(Acquire) {
            return;
        }
        if latest
            .as_ref()
            .is_none_or(|n| n.submission.id < self.submission.id)
        {
            *latest = Some(self.clone());
        }
    }
}
fn bind_pending(
    pending: &mut Vec<std::sync::Weak<Submission>>,
    lists: &[crate::object::This],
    token: u64,
) {
    pending.retain(|entry| {
        let Some(submission) = entry.upgrade() else {
            return false;
        };
        submission.bind(lists, token);
        submission.token.load(Acquire) == UNBOUND_TOKEN
    });
}
pub(crate) fn bind_submission(lists: &[crate::object::This], token: u64) {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    bind_pending(&mut pending, lists, token);
}
pub(crate) fn prepare(
    list: crate::object::This,
    input: &crate::framegen::Inputs<'_>,
) -> Option<Arc<Preparation>> {
    if !FG_READY.load(Acquire) {
        return None;
    }
    let resource = |i: usize, fixed: &ID3D12Resource| {
        let object = input.title_resources[i] as crate::object::This;
        if !crate::resource::is_resource(object) {
            return None;
        }
        crate::resource::made_copy(object)
            .filter(|copy| copy.texture.as_raw() == fixed.as_raw())
            .map(Source::Host)
    };
    // Pair the size snapshot with its identity under the same lock used by
    // resize invalidation; no old-size input can receive a post-resize ID.
    let (output, id) = {
        let display = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
        display.input_identity(&NEXT_ID)?
    };
    let preparation = Arc::new(Preparation {
        submission: Arc::new(Submission {
            id,
            list: list as usize,
            token: AtomicU64::new(UNBOUND_TOKEN),
        }),
        depth: resource(0, input.depth)?,
        motion: resource(1, input.motion)?,
        camera: plan::Camera {
            jitter: input.jitter,
            motion_scale: input.motion_scale,
            render: input.render_size,
            output,
            near: input.camera[0],
            far: input.camera[1],
            fov_y: input.camera[2],
            reset: input.reset,
            flags: input.upscale_flags,
        },
    });
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    pending.retain(|entry| entry.strong_count() != 0);
    pending.push(Arc::downgrade(&preparation.submission));
    Some(preparation)
}
#[derive(Default)]
struct Display {
    size: [u32; 2],
    scene: Option<(ID3D12Resource, u32)>,
    ui: Option<ID3D12Resource>,
    sample: Option<(std::time::Instant, u64, u64)>,
    real: u64,
    output_frames: u64,
}
impl Display {
    fn input_identity(&self, next_id: &AtomicU64) -> Option<([u32; 2], u64)> {
        if self.size.contains(&0) {
            return None;
        }
        Some((self.size, next_id.fetch_add(1, Relaxed) + 1))
    }

    fn retire_inputs(&mut self, size: [u32; 2], next_id: &AtomicU64, retired: &AtomicU64) {
        retired.fetch_max(next_id.load(Acquire), AcqRel);
        self.size = size;
        self.scene = None;
        self.ui = None;
    }
}
// SAFETY: COM resources are free-threaded; Display is accessed only under a mutex.
unsafe impl Send for Display {}
static DISPLAY: std::sync::LazyLock<Mutex<Display>> =
    std::sync::LazyLock::new(|| Mutex::new(Display::default()));
static FRAME_RATES: AtomicU64 = AtomicU64::new(0);
fn invalidate_inputs(display: &mut Display, size: [u32; 2]) {
    // A recording made before the swap chain changed must not become fresh
    // merely because its command list was submitted after the resize.
    display.retire_inputs(size, &NEXT_ID, &PRESENTED_ID);
    *SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()) = None;
}
pub(crate) fn frame_rates() -> Option<(f32, f32)> {
    let v = FRAME_RATES.load(Relaxed);
    (v != 0).then_some((f32::from_bits((v >> 32) as u32), f32::from_bits(v as u32)))
}
pub(crate) fn set_scene(scene: &ID3D12Resource, state: u32) {
    DISPLAY.lock().unwrap_or_else(|e| e.into_inner()).scene = Some((scene.clone(), state));
}
pub(crate) fn set_ui(ui: Option<&ID3D12Resource>) -> bool {
    DISPLAY.lock().unwrap_or_else(|e| e.into_inner()).ui = ui.cloned();
    true
}
pub(crate) fn swap_chain(
    hwnd: windows::Win32::Foundation::HWND,
    desc: &DXGI_SWAP_CHAIN_DESC1,
    factory: &IDXGIFactory2,
) -> Option<IDXGISwapChain3> {
    let (proxy, queue) = swap_chain_factory(factory)?;
    // SAFETY: valid live window, factory, and Streamline queue proxy.
    let result = unsafe { proxy.CreateSwapChainForHwnd(&queue, hwnd, desc, None, None) }
        .and_then(|swap| swap.cast::<IDXGISwapChain3>());
    swap_chain_made(result.is_ok(), result.as_ref().err());
    let swap = result.ok()?;
    DISPLAY.lock().unwrap_or_else(|e| e.into_inner()).size = [desc.Width, desc.Height];
    Some(swap)
}
pub(crate) fn present(
    swap: &IDXGISwapChain3,
    frame: u64,
    token: Option<u64>,
    allow: bool,
) -> windows::core::HRESULT {
    CURRENT_TOKEN.store(token.unwrap_or(u64::MAX - 1), Release);
    let (scene, ui, size) = {
        let d = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
        (d.scene.clone(), d.ui.clone(), d.size)
    };
    let presenting = scene.as_ref().and_then(|(scene, state)| {
        present_begins(
            frame,
            Planes {
                base: (scene, *state),
                overlay: if allow {
                    ui.as_ref().map(|ui| (ui, crate::compose::UI_READ.0 as u32))
                } else {
                    None
                },
                size,
            },
        )
    });
    let called = present_called(&presenting);
    // SAFETY: all composing commands are submitted; Streamline tags and options are set.
    let status = unsafe { swap.Present(0, DXGI_PRESENT(0)) };
    present_returned(called);
    present_ends(presenting);
    let mut d = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
    d.real += 1;
    d.output_frames += u64::from(shown_per_frame());
    let shown = d.output_frames;
    let now = std::time::Instant::now();
    if let Some((at, real, then)) = d.sample {
        let seconds = (now - at).as_secs_f64();
        if seconds >= 1.0 {
            let rendered = (d.real - real) as f32 / seconds as f32;
            let presented = (shown - then) as f32 / seconds as f32;
            tracing::info!(
                rendered_fps = rendered,
                presented_fps = presented,
                requested = framegen_times(),
                actual_last = shown_per_frame(),
                status = FG_STATUS.load(Relaxed),
                "NVIDIA frame generation: driver-reported output"
            );
            FRAME_RATES.store(
                (u64::from(rendered.to_bits()) << 32) | u64::from(presented.to_bits()),
                Relaxed,
            );
            d.sample = Some((now, d.real, shown));
        }
    } else {
        d.sample = Some((now, d.real, shown));
    }
    status
}
pub(crate) fn resize(
    width: u32,
    height: u32,
    resize: impl FnOnce() -> windows::core::Result<()>,
) -> windows::core::Result<()> {
    resizing();
    let result = resize();
    if result.is_ok() {
        let mut d = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
        invalidate_inputs(&mut d, [width, height]);
    }
    result
}
pub(crate) fn release(release: impl FnOnce()) -> bool {
    resizing();
    release();
    SWAP_CHAIN_PROXIED.store(false, Release);
    let mut display = DISPLAY.lock().unwrap_or_else(|e| e.into_inner());
    invalidate_inputs(&mut display, [0; 2]);
    *display = Display::default();
    true
}

/// Called after all title submissions, proxy presents and presenter destruction.
pub(crate) fn shut_down() -> i32 {
    static RESULT: Mutex<Option<i32>> = Mutex::new(None);
    let mut result = RESULT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(code) = *result {
        return code;
    }
    FG_READY.store(false, Release);
    DLSS_READY.store(false, Release);
    let Some(sl) = SL.get() else {
        return 0;
    };
    // SAFETY: caller finished submitted work and released our only swap chain
    // while the host device is still alive. No more title SDK calls can occur.
    let code = unsafe { (sl.shutdown)() };
    tracing::info!(code, result = abi::result_name(code), "Streamline shutdown");
    *result = Some(code);
    code
}

pub(crate) struct Planes<'a> {
    pub base: (&'a ID3D12Resource, u32),
    pub overlay: Option<(&'a ID3D12Resource, u32)>,
    pub size: [u32; 2],
}

/// What [`present_begins`] began, for [`present_ends`].
pub(crate) struct Presenting {
    token: *mut abi::FrameToken,
    frame: u32,
}

/// Before our swap chain's Present of title frame `frame`: with frame
/// generation, the frame's latency markers, constants and tags, and whether
/// frame generation runs on it (it does when an upscaling was submitted
/// since the last present).
pub(crate) fn present_begins(frame: u64, planes: Planes<'_>) -> Option<Presenting> {
    if !FG_READY.load(Acquire) || !SWAP_CHAIN_PROXIED.load(Acquire) {
        return None;
    }
    // The tagged textures' states are those the submissions before this
    // present left: only the line of submissions orders the two.
    if !crate::queue::line_on() && !CHECKING.load(Relaxed) {
        once!(
            warn,
            "DLSS Frame Generation: the line of submissions did not start; frame generation stays off"
        );
        return None;
    }
    // Where the interface is no plane of its own (Series X): the copy of
    // this texture taken in front of the title's interface composite.
    let hudless: Option<ID3D12Resource> = None;
    let (sl, functions) = (SL.get()?, FUNCTIONS.get()?);
    let (set_options, marker) = (functions.dlssg_set_options?, functions.pcl_set_marker?);
    let began = std::time::Instant::now();
    let frame = frame as u32;
    let token = frame_token(sl, frame)?;
    let mark = |m: u32| {
        let began = std::time::Instant::now();
        // SAFETY: a marker for a token of ours.
        unsafe { marker(m, token) };
        spent(T_MARKERS, began);
    };
    let mut calls = calls();
    if calls.simulating != Some(frame) {
        mark(abi::SIMULATION_START);
    }
    calls.simulating = None;
    mark(abi::SIMULATION_END);
    mark(abi::RENDER_SUBMIT_START);
    mark(abi::RENDER_SUBMIT_END);
    let fresh = {
        let latest = SUBMITTED.lock().unwrap_or_else(|e| e.into_inner());
        latest
            .as_ref()
            .filter(|n| {
                n.submission
                    .matches_present(PRESENTED_ID.load(Relaxed), CURRENT_TOKEN.load(Acquire))
            })
            .map(|n| (n.submission.id, n.depth.clone(), n.motion.clone(), n.camera))
    };
    let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
    let on = if let Some((id, depth, motion, camera)) = fresh {
        PRESENTED_ID.fetch_max(id, AcqRel);
        let plan = PLAN.get()?;
        // (Frame generation reads the camera's planes and matrices: the
        // nearer plane is the near one, whichever way the title names them.)
        let camera = camera.planes_in_order();
        let constants = camera.constants(plan.asked.jitter, plan.asked.mvec);
        // SAFETY: constants for this frame and viewport.
        let code = unsafe { (sl.set_constants)(&constants, token, &viewport) };
        if code != 0 {
            once!(
                warn,
                "frame generation: slSetConstants: {} ({code})",
                abi::result_name(code)
            );
        }
        let constants_ok = code == 0 && !camera.reset;
        let mut resources = vec![
            (depth.tagged(), abi::BUFFER_DEPTH, camera.render),
            (
                motion.tagged(),
                abi::BUFFER_MOTION_VECTORS,
                camera.motion_size(),
            ),
        ];
        // Two planes: the scene is the colour without the interface, the
        // interface plane its colour and alpha (premultiplied, as `compose`
        // blends it).
        if let Some((over, over_state)) = planes.overlay {
            resources.push((
                abi::Resource::texture(planes.base.0.as_raw(), planes.base.1),
                abi::BUFFER_HUDLESS_COLOR,
                planes.size,
            ));
            resources.push((
                abi::Resource::texture(over.as_raw(), over_state),
                abi::BUFFER_UI_COLOR_AND_ALPHA,
                planes.size,
            ));
        } else if let Some(copy) = &hudless {
            // One plane with the interface drawn into it: the copy taken in
            // front of the interface composite (ours, always a copy's
            // destination; Streamline holds it until the tags of the frame
            // go).
            resources.push((
                abi::Resource::texture(copy.as_raw(), COPY_DEST),
                abi::BUFFER_HUDLESS_COLOR,
                planes.size,
            ));
        }
        let tags: Vec<abi::ResourceTag> = resources
            .iter_mut()
            .map(|(r, kind, size)| {
                abi::ResourceTag::new(
                    r,
                    *kind,
                    abi::VALID_UNTIL_PRESENT,
                    abi::Extent::sized(*size),
                )
            })
            .collect();
        // SAFETY: tags of live textures in the states they are in at the
        // present (the line of submissions sends the present after them).
        let code = unsafe {
            (sl.set_tag_for_frame)(
                token,
                &viewport,
                tags.as_ptr(),
                tags.len() as u32,
                std::ptr::null_mut(),
            )
        };
        if code != 0 {
            once!(
                warn,
                "frame generation: slSetTagForFrame: {} ({code})",
                abi::result_name(code)
            );
        }
        once!(
            info,
            render = ?camera.render,
            output = ?camera.output,
            back_buffer = ?planes.size,
            interface_plane = planes.overlay.is_some(),
            interface_free_copy = hudless.is_some(),
            depth_state = format_args!("{:#x}", resources[0].0.state),
            motion_state = format_args!("{:#x}", resources[1].0.state),
            near = constants.camera_near,
            far = constants.camera_far,
            fov = constants.camera_fov,
            depth_inverted = camera.depth_inverted(),
            jitter = ?constants.jitter_offset,
            mvec_scale = ?constants.mvec_scale,
            "DLSS Frame Generation: first frame with depth and motion vectors tagged"
        );
        constants_ok && code == 0 && planes.overlay.is_some()
    } else {
        if calls.framegen_on {
            // No upscaling this frame (a menu, a load): what was tagged may
            // no longer hold what it did.
            let tags = [
                abi::BUFFER_DEPTH,
                abi::BUFFER_MOTION_VECTORS,
                abi::BUFFER_HUDLESS_COLOR,
                abi::BUFFER_UI_COLOR_AND_ALPHA,
            ]
            .map(|kind| {
                abi::ResourceTag::new(
                    std::ptr::null_mut(),
                    kind,
                    abi::VALID_UNTIL_PRESENT,
                    abi::Extent::default(),
                )
            });
            // SAFETY: null tags.
            unsafe {
                (sl.set_tag_for_frame)(
                    token,
                    &viewport,
                    tags.as_ptr(),
                    tags.len() as u32,
                    std::ptr::null_mut(),
                )
            };
        }
        false
    };
    // What is asked right now (the F1 menu changes it while the title runs): off, or the frames
    // shown for each one rendered. The swap chain stays Streamline's proxy either way.
    let wanted = framegen_wanted();
    let on = on && wanted != 0;
    let times = if on { wanted } else { 0 };
    if on != calls.framegen_on || times != calls.framegen_times {
        let mut options = abi::DlssgOptions::new(if on { abi::DLSSG_ON } else { abi::DLSSG_OFF });
        // `VICTORY_FRAMEGEN_X=2..6`: frames shown for each one rendered (2 unless set), no more
        // than this card and driver generate (the state says how many: 5 on an RTX 50).
        let mut generated = wanted.max(2) - 1;
        if on
            && generated > 1
            && let Some(get_state) = functions.dlssg_get_state
        {
            let mut state = abi::DlssgState::default();
            // SAFETY: a structure to fill, no options.
            let code = unsafe { get_state(&viewport, &mut state, std::ptr::null()) };
            let most = if code == 0 {
                state.num_frames_to_generate_max.max(1)
            } else {
                1
            };
            if generated > most {
                tracing::warn!(
                    asked = generated + 1,
                    given = most + 1,
                    "DLSS Frame Generation: more frames asked than this card generates"
                );
                generated = most;
            }
        }
        options.num_frames_to_generate = generated;
        FG_TIMES.store(generated + 1, Relaxed);
        options.flags = abi::DLSSG_RETAIN_RESOURCES_WHEN_OFF;
        options.num_back_buffers = 2;
        options.color_width = planes.size[0];
        options.color_height = planes.size[1];
        options.on_error_callback = Some(present_error);
        // SAFETY: filled options for viewport 0, on the presenting thread.
        let code = unsafe { set_options(&viewport, &options) };
        tracing::info!(
            frame,
            on,
            times = generated + 1,
            code,
            result = abi::result_name(code),
            "DLSS Frame Generation switched"
        );
        // 39, "warning: out of VRAM": the options are set all the same, with the card's memory
        // past its budget. Taken as set: asked again at every present (1,828 times in one
        // run, after the menu's 3X on a full 12 GB card) it showed no generated frame at all.
        if code == 39 {
            tracing::warn!(
                times = generated + 1,
                "DLSS Frame Generation is on with the card's memory past its budget: frames may come unevenly (the launcher leaves it room when it is on from the start)"
            );
        }
        if code == 0 || code == 39 {
            calls.framegen_on = on;
            // What was asked (the driver may give fewer): not set again until that changes.
            calls.framegen_times = times;
            if !on {
                FG_PER.store(1, Relaxed);
            }
        }
    }
    if calls.framegen_on {
        FG_ON.fetch_add(1, Relaxed);
        // What it takes of the card's memory: before its first present and
        // some seconds of presents later.
        let presents = FG_ON_ALL.fetch_add(1, Relaxed) + 1;
        if (presents == 1 || presents == 240)
            && !CHECKING.load(Relaxed)
            && let Some(host) = crate::host::host()
        {
            host.log_video_memory(if presents == 1 {
                "right before the first present with frame generation on"
            } else {
                "240 presents with frame generation on later"
            });
        }
    } else {
        FG_OFF.fetch_add(1, Relaxed);
    }
    drop(calls);
    mark(abi::PRESENT_START);
    spent(T_BEFORE, began);
    Some(Presenting { token, frame })
}

// Since the last frame-rate line, microseconds: on the presenting thread
// before the present (of them in the latency markers), in the present call
// (and the longest), after it (of them asking frame generation's state); on
// the threads that record upscalings, recording one (of them waiting for our
// lock); recording the copy of the colour without the interface.
static TIMES: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const T_BEFORE: usize = 0;
const T_MARKERS: usize = 1;
const T_PRESENT: usize = 2;
const T_PRESENT_LONGEST: usize = 3;
const T_AFTER: usize = 4;
const T_STATE: usize = 5;
const T_UPSCALE: usize = 6;
const T_UPSCALE_LOCK: usize = 7;
const T_COPY: usize = 8;

fn spent(slot: usize, since: std::time::Instant) {
    TIMES[slot].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
}

/// When the present call of a frame [`present_begins`] began begins: `None`
/// without frame generation (nothing is timed then).
pub(crate) fn present_called(presenting: &Option<Presenting>) -> Option<std::time::Instant> {
    presenting.is_some().then(std::time::Instant::now)
}

/// The present call that began at `called` returned.
pub(crate) fn present_returned(called: Option<std::time::Instant>) {
    if let Some(called) = called {
        let took = called.elapsed().as_micros() as u64;
        TIMES[T_PRESENT].fetch_add(took, Relaxed);
        TIMES[T_PRESENT_LONGEST].fetch_max(took, Relaxed);
    }
}

unsafe extern "C" fn present_error(error: *const i32) {
    // SAFETY: the error for the call.
    let code = unsafe { error.as_ref() }.copied().unwrap_or(0);
    once!(
        warn,
        hresult = format_args!("{:#x}", code as u32),
        "DLSS Frame Generation: a present failed"
    );
}

/// Frames shown for each one rendered with frame generation on, as asked at the start:
/// `VICTORY_FRAMEGEN_X=2..6` (2 unless set); 0: ready but off until the settings menu asks.
fn framegen_times() -> u32 {
    static TIMES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *TIMES.get_or_init(|| {
        std::env::var("VICTORY_FRAMEGEN_X")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map_or(2, |n| if n == 0 { 0 } else { n.clamp(2, 6) })
    })
}

/// Frames shown for each one rendered, as last given to frame generation.
static FG_TIMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(2);

/// What is asked of frame generation now: 0 off, 2..6 frames shown for each one rendered;
/// `u32::MAX`: not asked yet (then [`framegen_times`], the start's).
static FG_WANTED: AtomicU32 = AtomicU32::new(u32::MAX);
/// Frames frame generation showed for the last frame presented (1 while it is off).
static FG_PER: AtomicU32 = AtomicU32::new(1);

fn framegen_wanted() -> u32 {
    match FG_WANTED.load(Relaxed) {
        u32::MAX => framegen_times(),
        n => n,
    }
}

/// Whether frame generation can be switched and changed while the title runs: it was planned
/// at the start, so our swap chain is Streamline's proxy (the F1 menu asks).
pub(crate) fn framegen_live() -> bool {
    FG_READY.load(Acquire) && SWAP_CHAIN_PROXIED.load(Acquire)
}

/// Asks frame generation off (0) or for `times` frames shown for each one rendered (2..6),
/// from the next present on.
pub(crate) fn set_framegen(times: u32) {
    FG_WANTED.store(if times == 0 { 0 } else { times.clamp(2, 6) }, Relaxed);
    tracing::info!(times, "DLSS Frame Generation: asked from the settings menu");
}

/// For the frame-rate texts: frames shown for each one rendered (1 without frame generation).
pub(crate) fn shown_per_frame() -> u32 {
    FG_PER.load(Relaxed).max(1)
}

/// After our swap chain's Present: the frame's last marker, the next
/// frame's first, and what frame generation reports.
pub(crate) fn present_ends(presenting: Option<Presenting>) {
    let Some(Presenting { token, frame }) = presenting else {
        return;
    };
    let (Some(sl), Some(functions)) = (SL.get(), FUNCTIONS.get()) else {
        return;
    };
    let (Some(marker), Some(get_state)) = (functions.pcl_set_marker, functions.dlssg_get_state)
    else {
        return;
    };
    let began = std::time::Instant::now();
    // SAFETY: a marker for this frame's token.
    unsafe { marker(abi::PRESENT_END, token) };
    let mut calls = calls();
    if let Some(next) = frame_token(sl, frame.wrapping_add(1)) {
        // SAFETY: as above, for the next frame.
        unsafe { marker(abi::SIMULATION_START, next) };
        calls.simulating = Some(frame.wrapping_add(1));
    }
    spent(T_MARKERS, began);
    if calls.framegen_on {
        let mut state = abi::DlssgState::default();
        let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
        let asked = std::time::Instant::now();
        // SAFETY: a structure to fill, no options (no estimate asked).
        let code = unsafe { get_state(&viewport, &mut state, std::ptr::null()) };
        spent(T_STATE, asked);
        if code == 0 {
            FG_PER.store(state.num_frames_actually_presented.max(1), Relaxed);
            FG_SHOWN.fetch_add(u64::from(state.num_frames_actually_presented), Relaxed);
            FG_SHOWN_ALL.fetch_add(u64::from(state.num_frames_actually_presented), Relaxed);
            let before = FG_STATUS.swap(state.status, Relaxed);
            if state.status != before {
                tracing::warn!(
                    status = format_args!("{:#x}", state.status),
                    "DLSS Frame Generation status changed (0 ok, 1 output too small, 2 Reflex not seen, 4 HDR format, 8 constants invalid, 16 back buffer index not asked)"
                );
            }
        }
    }
    spent(T_AFTER, began);
}

/// Our swap chain's buffers are about to be made again: frame generation is
/// off until the next present with an upscaling (Streamline asks so).
pub(crate) fn resizing() {
    if !FG_READY.load(Acquire) || !SWAP_CHAIN_PROXIED.load(Acquire) {
        return;
    }
    let Some(set_options) = FUNCTIONS.get().and_then(|f| f.dlssg_set_options) else {
        return;
    };
    let mut calls = calls();
    if calls.framegen_on {
        let options = abi::DlssgOptions::new(abi::DLSSG_OFF);
        let viewport = abi::ViewportHandle::new(VIEWPORT_FG);
        // SAFETY: filled options.
        unsafe { set_options(&viewport, &options) };
        calls.framegen_on = false;
        calls.framegen_times = 0;
        FG_PER.store(1, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn submission(id: u64, list: usize) -> Arc<Submission> {
        Arc::new(Submission {
            id,
            list,
            token: AtomicU64::new(UNBOUND_TOKEN),
        })
    }

    #[test]
    fn only_the_matching_list_binds_and_the_first_token_wins() {
        let input = submission(7, 0x1000);
        input.bind(&[0x2000usize as crate::object::This], 10);
        assert!(!input.matches_present(0, 10));
        input.bind(&[0x1000usize as crate::object::This], 10);
        assert!(input.matches_present(0, 10));
        input.bind(&[0x1000usize as crate::object::This], 11);
        assert!(input.matches_present(0, 10));
        assert!(!input.matches_present(0, 11));
    }

    #[test]
    fn a_present_never_reuses_consumed_or_mismatched_inputs() {
        let input = submission(7, 0x1000);
        input.bind(&[0x1000usize as crate::object::This], 10);
        assert!(input.matches_present(6, 10));
        assert!(!input.matches_present(7, 10));
        assert!(!input.matches_present(8, 10));
        assert!(!input.matches_present(6, 11));
        let newer_unbound = submission(8, 0x1000);
        assert!(!newer_unbound.matches_present(7, UNBOUND_TOKEN));
        assert!(!newer_unbound.matches_present(7, 10));
    }

    #[test]
    fn discarded_recordings_are_pruned_without_becoming_submitted() {
        let discarded = submission(7, 0x1000);
        let surviving = submission(8, 0x2000);
        let mut pending = vec![Arc::downgrade(&discarded), Arc::downgrade(&surviving)];
        drop(discarded);
        bind_pending(&mut pending, &[0x1000usize as crate::object::This], 10);
        assert_eq!(pending.len(), 1);
        assert!(!surviving.matches_present(0, 10));
        bind_pending(&mut pending, &[0x2000usize as crate::object::This], 11);
        assert!(pending.is_empty());
        assert!(surviving.matches_present(0, 11));
    }

    #[test]
    fn resize_serializes_the_size_identity_pair_and_rejects_late_old_recordings() {
        use std::sync::mpsc;
        let display = Arc::new(Mutex::new(Display {
            size: [1920, 1080],
            ..Display::default()
        }));
        let next = Arc::new(AtomicU64::new(0));
        let retired = Arc::new(AtomicU64::new(0));
        let (captured, captured_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let prepare = {
            let (display, next) = (display.clone(), next.clone());
            std::thread::spawn(move || {
                let guard = display.lock().unwrap();
                let identity = guard.input_identity(&next).unwrap();
                captured.send(identity).unwrap();
                release_rx.recv().unwrap();
                drop(guard);
                identity
            })
        };
        let old = captured_rx.recv().unwrap();
        let late = submission(old.1, 0x1000);
        let (attempting, attempting_rx) = mpsc::channel();
        let resize = {
            let (display, next, retired) = (display.clone(), next.clone(), retired.clone());
            std::thread::spawn(move || {
                attempting.send(()).unwrap();
                display
                    .lock()
                    .unwrap()
                    .retire_inputs([1280, 720], &next, &retired);
            })
        };
        attempting_rx.recv().unwrap();
        release.send(()).unwrap();
        assert_eq!(prepare.join().unwrap(), old);
        resize.join().unwrap();

        // The old recording survives the resize and only binds when submitted
        // afterward. Its old dimensions and identity were retired together.
        late.bind(&[0x1000usize as crate::object::This], 12);
        assert_eq!(old.0, [1920, 1080]);
        assert!(!late.matches_present(retired.load(Acquire), 12));
        let fresh = display.lock().unwrap().input_identity(&next).unwrap();
        assert_eq!(fresh.0, [1280, 720]);
        assert!(fresh.1 > retired.load(Acquire));
        let current = submission(fresh.1, 0x2000);
        current.bind(&[0x2000usize as crate::object::This], 13);
        assert!(current.matches_present(retired.load(Acquire), 13));

        // A late completion of the older present cannot lower the barrier.
        retired.fetch_max(old.1 - 1, AcqRel);
        assert!(!late.matches_present(retired.load(Acquire), 12));
    }

    fn runtime_at(dir: PathBuf) -> Asked {
        Asked {
            dlss: false,
            framegen: true,
            mode: Mode::Dlaa,
            preset: 0,
            dir,
            jitter: [1.0; 2],
            mvec: [1.0; 2],
            reactive: false,
            log_dir: None,
            hudless: true,
            present_hook: false,
        }
    }

    #[test]
    fn sdk_and_log_paths_are_checked_before_loading_any_library() {
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut asked = runtime_at(source.clone());
        assert!(load_and_init(&asked, false).unwrap_err().contains("inside"));
        let external = tempfile::tempdir().unwrap();
        asked.dir = external.path().to_path_buf();
        asked.log_dir = Some(source);
        assert!(load_and_init(&asked, false).unwrap_err().contains("inside"));
    }

    #[test]
    fn missing_optional_sdk_returns_an_error_without_initializing_streamline() {
        let external = tempfile::tempdir().unwrap();
        let asked = runtime_at(external.path().to_path_buf());
        assert!(
            load_and_init(&asked, false)
                .unwrap_err()
                .contains(INTERPOSER)
        );
    }

    #[test]
    #[ignore = "Requires official NVIDIA SDK DLLs and an RTX GPU; opt in explicitly"]
    fn nvidia_adapter_requirements() {
        assert!(
            requested(),
            "set VICTORY_FRAMEGEN=1 and VICTORY_FRAMEGEN_PROVIDER=nvidia"
        );
        let asked = Asked::from_env().expect("runtime configuration");
        let available = load_and_init(&asked, false).expect("signed Streamline SDK initialises");
        println!(
            "NVIDIA adapter support: frame_generation={}",
            available.framegen
        );
        assert!(
            available.framegen,
            "driver, GPU and HAGS must support frame generation"
        );
        let sl = SL.get().unwrap();
        // SAFETY: no device or swap chain has been handed to Streamline.
        assert_eq!(unsafe { (sl.shutdown)() }, 0);
    }

    fn texture(
        device: &ID3D12Device,
        size: [u32; 2],
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(size[0]),
            Height: size[1],
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: flags,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut out = None;
        // SAFETY: valid plain descriptors and an out pointer.
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut out,
            )
        }
        .unwrap();
        out.unwrap()
    }

    #[test]
    #[ignore = "Opens a synthetic GPU test window; close the game and opt in explicitly"]
    fn nvidia_synthetic_present() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        CHECKING.store(true, Relaxed);
        assert!(requested());
        let host = crate::host::host().expect("host D3D12 device");
        assert!(
            FG_READY.load(Acquire),
            "NVIDIA frame-generation device hand-over"
        );
        let output = [1920, 1080];
        let render = [1280, 720];
        let device = &host.device;
        let base = texture(
            device,
            output,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let ui = texture(
            device,
            output,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let motion = texture(
            device,
            render,
            DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let depth = texture(
            device,
            render,
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        // SAFETY: ordinary descriptor-heap creation, descriptor writes and known clear values.
        unsafe {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 3,
                    ..Default::default()
                })
                .unwrap();
            let step =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            let list = host.new_list().unwrap();
            for (i, (texture, color)) in [
                (&base, [0.1, 0.3, 0.6, 1.0]),
                (&ui, [0.0; 4]),
                (&motion, [0.0; 4]),
            ]
            .into_iter()
            .enumerate()
            {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + i * step,
                };
                device.CreateRenderTargetView(texture, None, handle);
                list.list.ClearRenderTargetView(handle, &color, None);
            }
            let dsv: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let handle = dsv.GetCPUDescriptorHandleForHeapStart();
            device.CreateDepthStencilView(&depth, None, handle);
            list.list
                .ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
            list.list.ResourceBarrier(&[
                crate::host::transition(
                    &base,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                crate::host::transition(
                    &ui,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                crate::host::transition(
                    &motion,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATES(0xc0),
                ),
                crate::host::transition(
                    &depth,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    D3D12_RESOURCE_STATES(0xc0),
                ),
            ]);
            list.close();
            host.execute_sync_internal(&list);
        }
        let camera = plan::Camera {
            jitter: [0.0; 2],
            motion_scale: [render[0] as f32, render[1] as f32],
            render,
            output,
            near: 10000.0,
            far: 0.15,
            fov_y: 1.0,
            reset: false,
            flags: plan::FFX_DEPTH_INVERTED | plan::FFX_AUTO_EXPOSURE,
        };
        let before = FG_SHOWN_ALL.load(Relaxed);
        for frame in 1..=180u64 {
            *SUBMITTED.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(Preparation {
                submission: Arc::new(Submission {
                    id: NEXT_ID.fetch_add(1, Relaxed) + 1,
                    list: 0,
                    token: AtomicU64::new(frame),
                }),
                depth: Source::Fixed(depth.clone(), 0xc0),
                motion: Source::Fixed(motion.clone(), 0xc0),
                camera,
            }));
            assert!(
                host.present_planes(&base, 0, Some((&ui, 0)), frame, Some(frame))
                    .is_some()
            );
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        crate::host::settle_for_fence();
        let shown = FG_SHOWN_ALL.load(Relaxed) - before;
        let on = FG_ON_ALL.load(Relaxed);
        println!(
            "NVIDIA synthetic test: rendered=180 enabled={on} presented={shown} last_multiplier={} status={} requested={}",
            shown_per_frame(),
            FG_STATUS.load(Relaxed),
            framegen_times()
        );
        // Shut down while the device is still live, including on an assertion failure below.
        crate::framegen::shut_down();
        let shutdown = shut_down();
        println!("Streamline shutdown={shutdown}");
        assert_eq!(shutdown, 0);
        assert_eq!(FG_STATUS.load(Relaxed), 0);
        assert!(
            shown > 180 * 2,
            "actual NVIDIA output must exceed 2x, including warm-up"
        );
        // SAFETY: plain device health query after the submitted GPU work completed.
        unsafe { device.GetDeviceRemovedReason() }.unwrap();
    }
}
