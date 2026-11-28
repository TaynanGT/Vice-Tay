use super::policy::uma_requested;
use super::*;
use victory_core::autotune::VramTier;

#[test]
fn default_off_requires_an_exact_opt_in_or_a_nonempty_explicit_path() {
    for (flag, file) in [(None, None), (Some("0"), None), (Some("yes"), Some(""))] {
        assert!(Request::from_env(|key| match key {
            "VICTORY_AUTOTUNE" => flag.map(Into::into),
            "VICTORY_AUTOTUNE_FILE" => file.map(Into::into),
            _ => unreachable!(),
        })
        .is_none());
    }
    let on = Request::from_env(|key| (key == "VICTORY_AUTOTUNE").then(|| "1".into())).unwrap();
    assert!(on.path.is_none());
    let explicit = Request::from_env(|key| match key {
        "VICTORY_AUTOTUNE" => Some("0".into()),
        "VICTORY_AUTOTUNE_FILE" => Some("profiles/laptop.toml".into()),
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(explicit.profile_path().unwrap(), PathBuf::from("profiles/laptop.toml"));
}

#[test]
fn generated_ram_caps_and_vram_tiers_are_conservative() {
    for (ram, dedicated, dxgi_budget, cap, tier) in [
        (4, 4, 3600, 512, VramTier::Low),
        (8, 8, 7000, 1024, VramTier::Low),
        (16, 12, 11000, 2048, VramTier::Mid),
        (32, 16, 14700, 4096, VramTier::High),
        (64, 24, 22000, 4096, VramTier::High),
    ] {
        let hardware = Hardware {
            ram_total_mib: Some(ram * 1024),
            gpu: gpu::GpuProbe {
                dedicated_vram_mib: Some(dedicated * 1024),
                local_budget_mib: Some(dxgi_budget),
                ..Default::default()
            },
            ..Default::default()
        };
        let profile = hardware.defaults();
        assert_eq!(profile.ram_budget_mb, Some(cap));
        assert_eq!(profile.vram_tier, Some(tier));
    }
    let unknown = Hardware::default().defaults();
    assert_eq!(unknown.ram_budget_mb, None);
    assert_eq!(unknown.vram_tier, Some(VramTier::Unknown));
    assert_eq!(unknown.hybrid_cpu, None);
}

#[test]
fn uma_detection_never_generates_an_acceleration_opt_in() {
    let hardware = Hardware {
        gpu: gpu::GpuProbe {
            uma_detected: Some(true),
            cache_coherent_uma: Some(true),
            gpu_upload_heap_supported: Some(true),
            dedicated_vram_mib: Some(16 * 1024),
            ..Default::default()
        },
        ..Default::default()
    };
    let profile = hardware.defaults();
    assert_eq!(profile.vram_tier, Some(VramTier::Shared));
    assert!(!uma_requested(None, profile.uma_fast_path));
    assert!(uma_requested(Some(OsStr::new("1")), profile.uma_fast_path));
    let text = hardware.profile_text(profile, 1).unwrap();
    assert!(text.contains("# uma_detected = true\n"));
    assert!(text.contains("# cache_coherent_uma = true\n"));
    assert!(text.lines().any(|line| line == "uma_fast_path = false"));
}

#[test]
fn explicit_values_win_even_when_zero_empty_or_invalid() {
    let budget = |value: Option<&str>| window_budget_bytes(value.map(OsStr::new), Some(2048));
    assert_eq!(budget(None), Some(2048 << 20));
    assert_eq!(budget(Some(" 512 ")), Some(512 << 20));
    for value in ["0", "", "invalid"] {
        assert_eq!(budget(Some(value)), None);
        assert!(!uma_requested(Some(OsStr::new(value)), Some(true)));
    }
    assert!(!uma_requested(None, None));
    assert!(uma_requested(None, Some(true)));
    assert_eq!(window_budget_bytes(None, Some(0)), None);
    assert_eq!(window_budget_bytes(None, Some(u64::MAX)), Some(u64::MAX));
}

#[test]
fn toml_round_trip_explains_hybrid_and_multi_l3_without_tuning_affinity() {
    let mut hardware = Hardware {
        cpu: Some(cpu::CpuTopology {
            logical_cores: 20,
            classes: vec![
                cpu::CoreClass { efficiency_class: 0, physical_cores: 8, logical_cores: 8 },
                cpu::CoreClass { efficiency_class: 1, physical_cores: 6, logical_cores: 12 },
            ],
            l3_domains: vec![
                cpu::L3Domain { size_mib: 32, logical_cores: 8 },
                cpu::L3Domain { size_mib: 96, logical_cores: 12 },
            ],
            numa_nodes: vec![0, 1],
        }),
        ..Default::default()
    };
    let profile = hardware.defaults();
    assert_eq!(profile.hybrid_cpu, Some(true));
    let text = hardware.profile_text(profile, 45).unwrap();
    assert_eq!(AutotuneProfile::from_toml(&text).unwrap(), Some(profile));
    for snapshot in [
        "# logical_cores = 20",
        "class 1: 6 physical / 12 logical",
        "numa_nodes = [0, 1]",
        "l3_groups = 2",
        "96 MiB across 12 logical cores",
    ] {
        assert!(text.contains(snapshot), "missing {snapshot}");
    }
    // Multiple L3 domains alone don't imply hybrid P/E core classes.
    hardware.cpu.as_mut().unwrap().classes = vec![cpu::CoreClass {
        efficiency_class: 0,
        physical_cores: 10,
        logical_cores: 20,
    }];
    assert_eq!(hardware.defaults().hybrid_cpu, Some(false));
}

#[test]
fn working_file_wins_then_executable_file_then_create_in_working_dir() {
    let dir = test_directory();
    let working = dir.path().join("working.toml");
    let installed = dir.path().join("installed.toml");
    let choose = || default_profile_path(working.clone(), || Some(installed.clone())).unwrap();
    assert_eq!(choose(), working);
    fs::write(&installed, "[autotune]\n").unwrap();
    assert_eq!(choose(), installed);
    fs::write(&working, "[paths]\n").unwrap();
    assert_eq!(choose(), working);
}

#[test]
fn edits_override_defaults_and_existing_files_are_never_rewritten() {
    let dir = test_directory();
    let path = dir.path().join(FILE_NAME);
    let hardware = Hardware { ram_total_mib: Some(16 * 1024), ..Default::default() };
    let defaults = hardware.defaults();
    let load = || load_or_create(&path, &hardware, defaults, 1);
    assert_eq!(load().unwrap(), defaults);
    let generated = fs::read_to_string(&path).unwrap();
    assert_eq!(AutotuneProfile::from_toml(&generated).unwrap(), Some(defaults));

    for (existing, mb) in [
        ("[paths]\ngame_dir = 'D:/title'\n", Some(2048)),
        ("[paths]\ngame_dir = 'D:/title'\n[autotune]\nram_budget_mb = 0 # opt out\n", Some(0)),
        ("[autotune]\nram_budget_mb = 768\n", Some(768)),
    ] {
        fs::write(&path, existing).unwrap();
        let effective = load().unwrap();
        assert_eq!(effective.ram_budget_mb, mb);
        assert_eq!(effective.vram_tier, defaults.vram_tier);
        assert_eq!(fs::read_to_string(&path).unwrap(), existing);
    }
    let invalid = "[autotune]\nram_budget_mb = 'invalid'\n";
    fs::write(&path, invalid).unwrap();
    assert!(load().is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), invalid);
}
