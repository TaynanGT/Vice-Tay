//! Conservative defaults and the inspectable startup snapshot. These
//! calculations don't read the machine, process environment or filesystem.

use std::ffi::OsStr;
use std::fmt::{Display, Write};

use anyhow::Result;
use victory_core::autotune::{AutotuneProfile, VramTier};

use super::{cpu::CpuTopology, gpu::GpuProbe};

#[derive(Debug, Default)]
pub(super) struct Hardware {
    pub(super) gpu: GpuProbe,
    pub(super) cpu: Option<CpuTopology>,
    pub(super) ram_total_mib: Option<u64>,
}

impl Hardware {
    pub(super) fn defaults(&self) -> AutotuneProfile {
        // On a measured 16 GiB laptop, paging hurt the title and desktop
        // (docs/laptop-16gb.md). Trim *idle* loose copies only under pressure;
        // this is a headroom heuristic, not a measured optimum or a RAM limit.
        // Those copies already reached 4.3 GB in run 24 (shadow::let_go_idle).
        let ram_budget_mb = self.ram_total_mib.map(|total| (total / 8).min(4096));
        let vram_tier = if self.gpu.software_adapter == Some(true) {
            VramTier::Software
        } else if self.gpu.uma_detected == Some(true) {
            VramTier::Shared
        } else {
            match self.gpu.dedicated_vram_mib {
                None | Some(0) => VramTier::Unknown,
                Some(mib) if mib < 12 * 1024 => VramTier::Low,
                Some(mib) if mib < 16 * 1024 => VramTier::Mid,
                Some(_) => VramTier::High,
            }
        };
        AutotuneProfile {
            ram_budget_mb,
            vram_tier: Some(vram_tier),
            hybrid_cpu: self.cpu.as_ref().map(|cpu| cpu.classes.len() > 1),
            // Radeon 8060S regressed on the experimental fast path. Even
            // cache-coherent UMA is telemetry, never an optimization opt-in.
            uma_fast_path: Some(false),
        }
    }

    pub(super) fn profile_text(&self, profile: AutotuneProfile, probe_ms: u64) -> Result<String> {
        let gpu = &self.gpu;
        let display_luid = gpu.display_luid.map(|luid| format!("{luid:016x}"));
        let mut text = format!(
            "# VIctory autotune profile. Existing files are never rewritten.\n\
             # Explicit environment variables override the profile for this run only.\n\
             # ram_budget_mb: soft cap on loose RAM copies under pressure (MiB);\n\
             #   default: 1/8 of RAM, at most 4096 MiB. 0 disables the cap.\n\
             #   Runtime override: VICTORY_WINDOW_BUDGET_MB.\n\
             # vram_tier: dedicated VRAM <12 (low), 12–<16 (mid), >=16 GiB (high),\n\
             #   or shared, software, unknown. DXGI budgets still govern residency.\n\
             # hybrid_cpu records mixed core classes; it doesn't change affinity.\n\
             # uma_fast_path: reserved for explicit opt-in (VICTORY_UMA_FAST_PATH=1);\n\
             #   off by default even on UMA (Radeon 8060S regressions).\n\
             # Hardware snapshot from the run that created this file. GPU budgets\n\
             # and UMA describe node 0 of the selected render adapter.\n\
             # probe_ms = {probe_ms}\n\
             # ram_total_mb = {ram}\n\
             # render_luid = {render:016x}\n\
             # primary_display_luid = {display}\n\
             # display_luid_mismatch = {mismatch}\n\
             # hardware_adapters = {adapters}\n\
             # software_adapter = {software}\n\
             # dedicated_vram_mb = {dedicated}\n\
             # dxgi_local_budget_mb = {local}\n\
             # dxgi_non_local_budget_mb = {non_local}\n\
             # gpu_upload_heap_supported = {rebar} (discrete ReBAR proxy, not BIOS state)\n\
             # uma_detected = {uma}\n\
             # cache_coherent_uma = {coherent}\n",
            ram = known(self.ram_total_mib),
            render = gpu.render_luid,
            display = known(display_luid),
            mismatch = known(gpu.display_luid_mismatch),
            adapters = known(gpu.hardware_adapter_count),
            software = known(gpu.software_adapter),
            dedicated = known(gpu.dedicated_vram_mib),
            local = known(gpu.local_budget_mib),
            non_local = known(gpu.non_local_budget_mib),
            rebar = known(gpu.gpu_upload_heap_supported),
            uma = known(gpu.uma_detected),
            coherent = known(gpu.cache_coherent_uma),
        );
        if let Some(cpu) = &self.cpu {
            writeln!(text, "# logical_cores = {}", cpu.logical_cores).unwrap();
            writeln!(text, "# CPU core classes (on a hybrid CPU the highest class is fastest):").unwrap();
            for class in &cpu.classes {
                writeln!(
                    text,
                    "#   class {}: {} physical / {} logical",
                    class.efficiency_class, class.physical_cores, class.logical_cores
                )
                .unwrap();
            }
            writeln!(text, "# numa_nodes = {:?}", cpu.numa_nodes).unwrap();
            writeln!(
                text,
                "# l3_groups = {} (shared L3 domains, a CCD proxy, not a CCD count)",
                cpu.l3_domains.len()
            )
            .unwrap();
            for l3 in &cpu.l3_domains {
                writeln!(text, "#   {} MiB across {} logical cores", l3.size_mib, l3.logical_cores).unwrap();
            }
        } else {
            text.push_str("# CPU topology unavailable\n");
        }
        text.push('\n');
        text.push_str(&profile.to_toml()?);
        Ok(text)
    }
}

fn known(value: Option<impl Display>) -> String {
    value.map_or_else(|| "unknown".to_owned(), |value| value.to_string())
}

pub(super) fn uma_requested(explicit: Option<&OsStr>, profile: Option<bool>) -> bool {
    explicit.map_or(profile.unwrap_or(false), |value| value == "1")
}
