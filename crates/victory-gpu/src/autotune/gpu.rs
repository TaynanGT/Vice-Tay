//! D3D12, DXGI and primary monitor queries for the opt-in autotune report.
//! Uses the host's device and DXGI factory without changing adapter selection.

use std::mem::size_of_val;

use windows::Win32::Foundation::{LUID, POINT};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_ARCHITECTURE1, D3D12_FEATURE_D3D12_OPTIONS16,
    D3D12_FEATURE_DATA_ARCHITECTURE1, D3D12_FEATURE_DATA_D3D12_OPTIONS16, ID3D12Device,
};
use windows::Win32::Graphics::Dxgi::{
    DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
    DXGI_MEMORY_SEGMENT_GROUP, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory4,
};
use windows::Win32::Graphics::Gdi::{HMONITOR, MONITOR_DEFAULTTONULL, MonitorFromPoint};
use windows::core::Interface;

// Use the same node for the architecture and memory budget queries.
const NODE_INDEX: u32 = 0;

/// Startup GPU snapshot. `None` is unknown, not zero or false.
#[derive(Debug, Default)]
pub(super) struct GpuProbe {
    pub(super) render_luid: u64,
    /// Hardware adapter owning the primary desktop output, not a window's output.
    pub(super) display_luid: Option<u64>,
    /// Only meaningful when both the display owner and a hardware renderer are known.
    pub(super) display_luid_mismatch: Option<bool>,
    pub(super) hardware_adapter_count: Option<u32>,
    pub(super) dedicated_vram_mib: Option<u64>,
    pub(super) local_budget_mib: Option<u64>,
    pub(super) non_local_budget_mib: Option<u64>,
    /// D3D12 runtime capability, a ReBAR proxy on discrete GPUs, not a BIOS reading.
    pub(super) gpu_upload_heap_supported: Option<bool>,
    pub(super) uma_detected: Option<bool>,
    pub(super) cache_coherent_uma: Option<bool>,
    pub(super) software_adapter: Option<bool>,
}

/// `adapter`, if supplied, is the Host's adapter for `device`.
pub(super) fn probe(
    device: &ID3D12Device,
    factory: Option<&IDXGIFactory4>,
    adapter: Option<&IDXGIAdapter3>,
) -> GpuProbe {
    // SAFETY: a plain identity query on the live Host device.
    let render_luid = pack_luid(unsafe { device.GetAdapterLuid() });
    let mut result = GpuProbe {
        render_luid,
        ..GpuProbe::default()
    };
    probe_device(device, &mut result);

    let matching = factory.and_then(|factory| {
        scan_adapters(factory, &mut result).or_else(|| {
            if adapter.is_none() {
                warp_by_luid(factory, render_luid)
            } else {
                None
            }
        })
    });
    // Reuse the description from the scan if available. An adapter supplied
    // by the Host (e.g. WARP) need not be present in EnumAdapters1.
    let desc = matching.as_ref().map(|(_, desc)| *desc).or_else(|| {
        // SAFETY: a plain description query on the borrowed adapter.
        adapter.and_then(|adapter| unsafe { adapter.GetDesc1() }.ok())
    });
    if let Some(desc) = desc {
        result.dedicated_vram_mib = Some((desc.DedicatedVideoMemory as u64) >> 20);
        result.software_adapter = Some(is_software_adapter(&desc));
    }
    let recovered_adapter = match (&matching, adapter) {
        (Some((candidate, _)), None) => candidate.cast::<IDXGIAdapter3>().ok(),
        _ => None,
    };
    if let Some(adapter) = adapter.or(recovered_adapter.as_ref()) {
        result.local_budget_mib = memory_budget_mib(adapter, DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
        result.non_local_budget_mib =
            memory_budget_mib(adapter, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
    }
    result.display_luid_mismatch =
        display_mismatch(render_luid, result.display_luid, result.software_adapter);
    result
}

fn pack_luid(luid: LUID) -> u64 {
    (u64::from(luid.HighPart as u32) << 32) | u64::from(luid.LowPart)
}

fn display_mismatch(render: u64, display: Option<u64>, software: Option<bool>) -> Option<bool> {
    match (display, software) {
        (Some(display), Some(false)) => Some(render != display),
        _ => None,
    }
}

fn probe_device(device: &ID3D12Device, result: &mut GpuProbe) {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let mut architecture = D3D12_FEATURE_DATA_ARCHITECTURE1 {
        NodeIndex: NODE_INDEX,
        ..Default::default()
    };
    // SAFETY: both feature queries receive a writable structure of the matching type and size.
    unsafe {
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
        {
            result.gpu_upload_heap_supported = Some(options.GPUUploadHeapSupported.as_bool());
        }
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE1,
                (&raw mut architecture).cast(),
                size_of_val(&architecture) as u32,
            )
            .is_ok()
        {
            result.uma_detected = Some(architecture.UMA.as_bool());
            result.cache_coherent_uma = Some(architecture.CacheCoherentUMA.as_bool());
        }
    }
}

fn scan_adapters(
    factory: &IDXGIFactory4,
    result: &mut GpuProbe,
) -> Option<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    // The primary monitor contains the virtual desktop origin. A NULL result
    // is not evidence of a render/display difference.
    // SAFETY: this queries a desktop coordinate and does not require a window.
    let primary = unsafe { MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTONULL) };
    let mut matching = None;
    let mut hardware_count = Some(0u32);
    for index in 0..=u32::MAX {
        // SAFETY: DXGI enumerates on the live factory; invalid indices are errors.
        let candidate = match unsafe { factory.EnumAdapters1(index) } {
            Ok(candidate) => candidate,
            Err(err) => {
                if err.code() == DXGI_ERROR_NOT_FOUND {
                    result.hardware_adapter_count = hardware_count;
                }
                break;
            }
        };
        // SAFETY: a plain query on the enumerated adapter.
        let Ok(desc) = (unsafe { candidate.GetDesc1() }) else {
            hardware_count = None;
            continue;
        };
        let luid = pack_luid(desc.AdapterLuid);
        if !is_software_adapter(&desc) {
            hardware_count = hardware_count.and_then(|count| count.checked_add(1));
            // DXGI normally enumerates the primary owner first. Stop asking
            // about outputs as soon as we have verified the monitor handle.
            if result.display_luid.is_none()
                && !primary.is_invalid()
                && owns_monitor(&candidate, primary)
            {
                result.display_luid = Some(luid);
            }
        }
        if luid == result.render_luid && matching.is_none() {
            matching = Some((candidate, desc));
        }
    }
    matching
}

fn owns_monitor(adapter: &IDXGIAdapter1, monitor: HMONITOR) -> bool {
    for index in 0..=u32::MAX {
        // SAFETY: the live adapter returns an owned DXGI output or an error.
        let Ok(output) = (unsafe { adapter.EnumOutputs(index) }) else {
            break;
        };
        // SAFETY: a plain description query on this output.
        if let Ok(desc) = unsafe { output.GetDesc() }
            && desc.AttachedToDesktop.as_bool()
            && desc.Monitor == monitor
        {
            return true;
        }
    }
    false
}

fn warp_by_luid(
    factory: &IDXGIFactory4,
    render_luid: u64,
) -> Option<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    // SAFETY: adapter queries only. EnumWarpAdapter is not proof of identity
    // or that the result is software (some DXGI implementations return hardware).
    unsafe {
        let candidate: IDXGIAdapter1 = factory.EnumWarpAdapter().ok()?;
        let desc = candidate.GetDesc1().ok()?;
        (pack_luid(desc.AdapterLuid) == render_luid).then_some((candidate, desc))
    }
}

fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    // Same test as victory-shader's device selection, including the Microsoft
    // Basic Render Driver if the DXGI software flag is absent.
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

fn memory_budget_mib(adapter: &IDXGIAdapter3, group: DXGI_MEMORY_SEGMENT_GROUP) -> Option<u64> {
    let mut memory = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: the query fills the writable memory info structure for the supplied segment.
    unsafe { adapter.QueryVideoMemoryInfo(NODE_INDEX, group, &mut memory) }.ok()?;
    Some(memory.Budget >> 20)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_both_luid_halves_without_losing_bits() {
        let luid = LUID {
            HighPart: -0x0123_4568,
            LowPart: 0x89ab_cdef,
        };
        assert_eq!(pack_luid(luid), 0xfedc_ba98_89ab_cdef);
    }

    #[test]
    fn compares_only_known_hardware_renderers_with_a_resolved_desktop() {
        let render = 0x0000_0001_0000_1234;
        let other = 0x0000_0002_0000_1234;
        assert_eq!(display_mismatch(render, Some(render), Some(false)), Some(false));
        assert_eq!(display_mismatch(render, Some(other), Some(false)), Some(true));
        assert_eq!(display_mismatch(render, None, Some(false)), None);
        assert_eq!(display_mismatch(render, Some(other), Some(true)), None);
        assert_eq!(display_mismatch(render, Some(other), None), None);
    }
}
