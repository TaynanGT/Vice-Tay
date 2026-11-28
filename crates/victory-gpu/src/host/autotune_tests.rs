use std::fs;
use std::path::Path;
use std::process::Command;

use victory_core::autotune::{AutotuneProfile, VramTier};

#[test]
fn startup_uses_the_selected_device_and_writes_only_when_opted_in() {
    const CHILD: &str = "VICTORY_AUTOTUNE_TEST_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let host = super::host().expect("a live D3D12 host is required for this test");
        if crate::switch::on("VICTORY_D3D12_DEBUG") {
            assert!(host.messages.is_some(), "the D3D12 debug layer was requested");
        }
        assert!(!Path::new("victory.toml").try_exists().unwrap());
        if mode == "off" {
            assert_eq!(host.autotune_ram_budget_mb, None);
            return;
        }
        assert_eq!(mode, "emit");

        let snapshot = fs::read_to_string("probe.toml").expect("an explicit profile was generated");
        let profile = AutotuneProfile::from_toml(&snapshot).unwrap().unwrap();
        let field = |prefix: &str| {
            snapshot.lines().find_map(|line| line.strip_prefix(prefix)).unwrap()
        };
        let number = |prefix: &str| field(prefix).parse::<u64>().unwrap();
        assert_eq!(profile.uma_fast_path, Some(false));
        assert_eq!(host.autotune_ram_budget_mb, profile.ram_budget_mb);
        assert!(profile.ram_budget_mb.is_some_and(|mb| (1..=4096).contains(&mb)));
        assert!(number("# ram_total_mb = ") > 0);
        assert!(number("# logical_cores = ") > 0);
        assert!(profile.hybrid_cpu.is_some());
        assert!(matches!(field("# uma_detected = "), "true" | "false" | "unknown"));
        // SAFETY: these are identity queries on the live device and adapter.
        let luid = unsafe { host.device.GetAdapterLuid() };
        assert_eq!(field("# render_luid = "), format!("{:08x}{:08x}", luid.HighPart as u32, luid.LowPart));
        let warp = crate::switch::on("VICTORY_D3D12_WARP");
        assert_eq!(field("# software_adapter = "), if warp { "true" } else { "false" });
        if warp {
            assert_eq!(profile.vram_tier, Some(VramTier::Software));
            assert_eq!(field("# display_luid_mismatch = "), "unknown");
        } else {
            let adapter = host.adapter.as_ref().expect("DXGI adapter for the hardware device");
            let desc = unsafe { adapter.GetDesc1() }.unwrap();
            assert_eq!(number("# dedicated_vram_mb = "), desc.DedicatedVideoMemory as u64 >> 20);
            assert!(number("# hardware_adapters = ") >= 1);
            let _ = number("# dxgi_local_budget_mb = ");
        }
        println!("autotune {}: {} ms hardware probe", if warp { "WARP" } else { "hardware" }, number("# probe_ms = "));
        return;
    }

    // A fresh process per startup, without changing process-global env in
    // the parallel test runner. Keep the generated file under target/.
    let dir = crate::autotune::test_directory();
    // Windows CI need not have a hardware adapter. The debug gate asks for
    // hardware; the WARP gate (and an ordinary test run) uses software.
    let default_to_warp = std::env::var_os("VICTORY_D3D12_WARP").is_none()
        && !crate::switch::on("VICTORY_D3D12_DEBUG");
    for mode in ["off", "emit"] {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "host::autotune_tests::startup_uses_the_selected_device_and_writes_only_when_opted_in",
                "--nocapture",
            ])
            .current_dir(dir.path())
            .env(CHILD, mode)
            .env("VICTORY_AUTOTUNE", "0")
            .env("VICTORY_UMA_FAST_PATH", "1")
            .env_remove("VICTORY_AUTOTUNE_FILE");
        if mode == "emit" {
            child.env("VICTORY_AUTOTUNE_FILE", "probe.toml");
        }
        if default_to_warp {
            child.env("VICTORY_D3D12_WARP", "1");
        }
        let status = child.status().expect("launching the isolated startup test");
        assert!(status.success(), "autotune startup {mode}: {status}");
    }
    // `--exact` exits successfully even if the child test name didn't match.
    assert!(dir.path().join("probe.toml").is_file());
}
