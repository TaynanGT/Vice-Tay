//! Opt-in startup profile for the device VIctory actually selected. The host
//! already owns the DXGI factory and D3D12 device; don't make a second device.

mod cpu;
mod gpu;
mod policy;
#[cfg(test)]
mod tests;

use std::ffi::{OsStr, OsString};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use victory_core::autotune::AutotuneProfile;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter3, IDXGIFactory4};

use self::policy::Hardware;

const FILE_NAME: &str = "victory.toml";

#[cfg(test)]
pub(crate) fn test_directory() -> tempfile::TempDir {
    let target = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
    fs::create_dir_all(&target).unwrap();
    tempfile::Builder::new().prefix("autotune-").tempdir_in(target).unwrap()
}

struct Request {
    path: Option<PathBuf>,
}

impl Request {
    fn from_env(env: impl Fn(&str) -> Option<OsString>) -> Option<Self> {
        let path = env("VICTORY_AUTOTUNE_FILE")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        (path.is_some() || env("VICTORY_AUTOTUNE").is_some_and(|value| value == "1"))
            .then_some(Self { path })
    }

    fn profile_path(self) -> io::Result<PathBuf> {
        if let Some(path) = self.path {
            return Ok(path);
        }
        default_profile_path(PathBuf::from(FILE_NAME), || {
            std::env::current_exe().ok().map(|exe| exe.with_file_name(FILE_NAME))
        })
    }
}

fn default_profile_path(
    working: PathBuf,
    installed: impl FnOnce() -> Option<PathBuf>,
) -> io::Result<PathBuf> {
    if !working.try_exists()?
        && let Some(installed) = installed()
        && installed.try_exists()?
    {
        return Ok(installed);
    }
    Ok(working)
}

/// Called once when making the host. No hardware queries, path lookups or
/// writes when neither opt-in is set. Returns the profile's loose-window cap
/// (the explicit env override is still read by the window sweep).
pub(crate) fn startup(
    device: &ID3D12Device,
    factory: Option<&IDXGIFactory4>,
    adapter: Option<&IDXGIAdapter3>,
) -> Option<u64> {
    let request = Request::from_env(|key| std::env::var_os(key))?;
    enabled_startup(request, device, factory, adapter)
}

#[cold]
fn enabled_startup(
    request: Request,
    device: &ID3D12Device,
    factory: Option<&IDXGIFactory4>,
    adapter: Option<&IDXGIAdapter3>,
) -> Option<u64> {
    let began = Instant::now();
    let hardware = Hardware {
        gpu: gpu::probe(device, factory, adapter),
        cpu: cpu::probe(),
        ram_total_mib: crate::residency::memory_status().map(|status| status.ullTotalPhys >> 20),
    };
    let probe_ms = began.elapsed().as_millis() as u64;
    let defaults = hardware.defaults();
    let profile = match request.profile_path() {
        Ok(path) => load_or_create(&path, &hardware, defaults, probe_ms).unwrap_or_else(|error| {
            tracing::warn!(
                path = %path.display(),
                error = %format_args!("{error:#}"),
                "autotune profile unavailable; keeping built-in settings"
            );
            AutotuneProfile::default()
        }),
        Err(error) => {
            tracing::warn!(%error, "autotune profile path unavailable; keeping built-in settings");
            AutotuneProfile::default()
        }
    };
    let uma_fast_path_requested = policy::uma_requested(
        std::env::var_os("VICTORY_UMA_FAST_PATH").as_deref(),
        profile.uma_fast_path,
    );
    if uma_fast_path_requested {
        tracing::warn!("UMA fast path requested, but this build has no UMA fast-path backend");
    }
    if let (Some(true), Some(display_luid)) = (hardware.gpu.display_luid_mismatch, hardware.gpu.display_luid) {
        tracing::warn!(
            render_luid = format_args!("{:016x}", hardware.gpu.render_luid),
            display_luid = format_args!("{display_luid:016x}"),
            "autotune: the primary display is on another adapter; cross-adapter presentation may cost more"
        );
    }
    tracing::info!(
        ?hardware,
        ?profile,
        probe_ms,
        startup_ms = began.elapsed().as_millis() as u64,
        uma_fast_path_requested,
        "autotune startup profile"
    );
    profile.ram_budget_mb
}

fn load_or_create(
    path: &Path,
    hardware: &Hardware,
    defaults: AutotuneProfile,
    probe_ms: u64,
) -> Result<AutotuneProfile> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let text = hardware.profile_text(defaults, probe_ms)?;
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    file.write_all(text.as_bytes()).context("writing startup profile")?;
                    tracing::info!(path = %path.display(), "autotune wrote a new profile");
                    return Ok(defaults);
                }
                // Someone created it after our read; their profile wins.
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    fs::read_to_string(path).context("reading startup profile")?
                }
                Err(error) => return Err(error).context("creating startup profile"),
            }
        }
        Err(error) => return Err(error).context("reading startup profile"),
    };
    let file = AutotuneProfile::from_toml(&text).context("parsing [autotune]")?;
    if file.is_some() {
        tracing::info!(path = %path.display(), "autotune loaded the profile");
    } else {
        tracing::info!(path = %path.display(), "no [autotune] table; using probed defaults without editing the file");
    }
    let file = file.unwrap_or_default();
    Ok(AutotuneProfile {
        ram_budget_mb: file.ram_budget_mb.or(defaults.ram_budget_mb),
        vram_tier: file.vram_tier.or(defaults.vram_tier),
        hybrid_cpu: file.hybrid_cpu.or(defaults.hybrid_cpu),
        uma_fast_path: file.uma_fast_path.or(defaults.uma_fast_path),
    })
}

/// Even a zero, empty or invalid explicit value keeps its pre-autotune
/// meaning (no cap); only a missing value falls back to the file.
pub(crate) fn window_budget_bytes(explicit: Option<&OsStr>, profile_mb: Option<u64>) -> Option<u64> {
    let mb = match explicit {
        Some(value) => value.to_str()?.trim().parse().ok(),
        None => profile_mb,
    }?;
    (mb > 0).then(|| mb.saturating_mul(1 << 20))
}
