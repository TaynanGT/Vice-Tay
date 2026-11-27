use std::path::Path;

use anyhow::{Context, Result};

use crate::{Console, Unimplemented};

#[cfg(any(windows, test))]
const SKIP_CPU_CHECK_ENV: &str = "VICTORY_SKIP_CPU_CHECK";

#[cfg(any(windows, test))]
fn skip_cpu_check(value: Option<&str>) -> bool {
    value == Some("1")
}

#[cfg(any(windows, test))]
fn check_cpu_features_with(has_avx2: bool, skip_env: Option<&str>) -> Result<()> {
    if has_avx2 || skip_cpu_check(skip_env) {
        return Ok(());
    }
    anyhow::bail!(
        "this CPU does not support AVX2, which Xbox Series titles require; \
         set {SKIP_CPU_CHECK_ENV}=1 to bypass this check (for example under CPU emulation such as Intel SDE)"
    );
}

/// Pre-flight check before executing title code: Xbox Series binaries are
/// compiled for Zen 2 and require AVX2.
#[cfg(windows)]
fn check_cpu_features() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = true;
    let skip = std::env::var(SKIP_CPU_CHECK_ENV).ok();
    check_cpu_features_with(has_avx2, skip.as_deref())
}

#[cfg(windows)]
pub fn run(
    exe: &Path,
    unimplemented: Unimplemented,
    strict: bool,
    execute: bool,
    console: Console,
) -> Result<()> {
    use std::sync::Arc;
    use victory_hle::{HleResolver, HostPolicy};
    use victory_loader::{BindReport, UnimplementedPolicy, WithTitleDlls};

    if execute {
        check_cpu_features()?;
    }
    // A title that relaunched itself: the old process must be gone first.
    victory_hle::modules::lifecycle::wait_for_previous();
    let cfg = victory_core::Config::load()?;
    let original = std::fs::read(exe).with_context(|| format!("reading {}", exe.display()))?;
    let prepared = victory_loader::prepare(&original)?;
    let hash = prepared.original_hash.to_hex();
    let build = victory_game::identify(&hash);
    match build {
        Some(build) => println!("build          {} ({hash})", build.label),
        None => println!("build          unknown ({hash})"),
    }
    // The known build's Series S upscaler choice (victory-loader's
    // `resolution`): both pickers return the title's own upscaler.
    let upscaler_default = build.is_some()
        && known_patches_on(console, std::env::var("VICTORY_TITLE_PATCHES").ok().as_deref());

    let images = cfg.cache_subdir("images")?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let image = victory_loader::load(&prepared, &images)?;
    // The program is mapped from the copy `load` wrote: the file's bytes
    // and their patched copy are not read again, and were held for the
    // whole run (two blocks of 114.7 MB each for the title, in
    // physical memory: run 65's memory map).
    let image_name = prepared.file_name();
    drop(prepared);
    drop(original);
    victory_hle::runtime::set_main_image(image.base());
    victory_hle::runtime::set_image_path(image.base(), exe);
    println!(
        "mapped at      {:#x} (preferred {:#x})",
        image.base(),
        image.info().image_base
    );
    let resolver = HleResolver::new(if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    });
    let wanted = |dll: &str| title_dll(&resolver.host, dll);
    let title_dir = exe.parent().unwrap_or(Path::new("."));
    // A missing manifest leaves id 0 and no storage quota.
    let manifest =
        std::fs::read_to_string(title_dir.join("MicrosoftGame.config")).unwrap_or_default();
    let title = victory_hle::runtime::Title::from_manifest(&manifest);
    let storage = cfg.data_subdir(&format!("{:08X}/storage", title.id))?;
    println!(
        "title          {:08X}, storage {} ({} MB)",
        title.id,
        storage.display(),
        title.storage_bytes >> 20
    );
    // The console's fixed drives: the package as `G:`, temporary files `T:`.
    victory_hle::modules::files::mount_at('G', title_dir);
    let temp = cfg.data_subdir(&format!("{:08X}/temp", title.id))?;
    victory_hle::modules::files::mount_at('T', &temp);
    victory_hle::runtime::set_title(victory_hle::runtime::Title { storage, ..title });
    victory_hle::console::set(match console {
        Console::SeriesX => victory_hle::console::Console::SeriesX,
        Console::SeriesS => victory_hle::console::Console::SeriesS,
    });
    println!("console        {}", victory_hle::console::current().name());
    let dlls = victory_loader::load_title_dlls(image.info(), title_dir, &images, &wanted)?;
    for (name, dll) in &dlls {
        println!("title dll      {name} at {:#x}", dll.base());
        victory_hle::runtime::set_image_path(dll.base(), &title_dir.join(name));
    }
    let imported: Vec<&str> = image
        .info()
        .imports
        .iter()
        .map(|m| m.dll.as_str())
        .collect();
    let mapped: Vec<&str> = dlls.iter().map(|(name, _)| name.as_str()).collect();
    let absent = absent_title_dlls(&imported, &mapped, &wanted);
    if !absent.is_empty() {
        let message = format!(
            "{} not in {}, and neither the host nor VIctory provides {}: the package is \
             incomplete; copy every file of the package next to the executable",
            absent.join(", "),
            title_dir.display(),
            if absent.len() == 1 { "it" } else { "them" },
        );
        println!("missing dlls   {message}");
        // None of their imports can be bound, so the title stops at its first
        // call into one (Grand Theft Auto VI 1.0.2.1 at +0x2237178 during
        // System Init, #930) with nothing saying why. Stubs only if asked for.
        if execute && matches!(unimplemented, Unimplemented::Trap) {
            anyhow::bail!(
                "the title's own DLLs are missing: {message} (`--unimplemented zero` runs it anyway)"
            );
        }
    }

    let policy = match unimplemented {
        Unimplemented::Trap => UnimplementedPolicy::Trap,
        Unimplemented::Zero => UnimplementedPolicy::ReturnZero,
    };
    let with_dlls = WithTitleDlls {
        inner: &resolver,
        dlls: &dlls,
    };
    let show = |name: &str, report: BindReport| {
        println!(
            "bound          {name}: {} hle, {} host, {} title, {} missing",
            report.hle,
            report.host,
            report.title,
            report.missing.len()
        );
        for m in &report.missing {
            println!("  missing      {}!{}", m.dll, m.symbol);
        }
    };
    for (name, dll) in &dlls {
        show(name, dll.bind_imports(&with_dlls, policy)?);
    }
    let exe_name = exe.file_name().unwrap_or_default().to_string_lossy();
    show(&exe_name, image.bind_imports(&with_dlls, policy)?);
    if execute {
        let size = image.info().size_of_image as usize;
        // SAFETY: the image stays mapped, and nothing of it has run yet.
        unsafe { victory_loader::write_sites::apply_asked(&image_name, image.base(), size) };
    }
    // Renderer port #2 (VICTORY_PORT_DESCRIPTORS=1): the title's stores of
    // descriptor records tell the layer their slots.
    #[cfg(windows)]
    if execute && victory_hle::port_descriptors::on() {
        use victory_loader::descriptor_sites::{Hooked, SITES, apply};
        let size = image.info().size_of_image as usize;
        // SAFETY: the image stays mapped, nothing of it has run yet, and the
        // marks are a static.
        let done = unsafe { apply(image.base(), size, victory_hle::port_descriptors::marks_address(), &SITES) };
        let patched = done.iter().filter(|d| **d == Hooked::Done).count();
        println!("renderer port 2: {patched} of {} descriptor record stores of the title patched", SITES.len());
        victory_hle::port_descriptors::hooked(image.base(), size, patched);
        // VICTORY_PORT_COPIES=1: the frame allocator's table copies the six
        // stores miss (its own memcpy, a 128-byte copy) mark their slots too.
        // A third fewer records left to the fallback (runs pop-4/pop-5), no
        // change seen in the picture (POPIN.md): off by default.
        if std::env::var("VICTORY_PORT_COPIES").is_ok_and(|v| v.trim() == "1") {
            use victory_loader::descriptor_sites::{TITLE_MEMCPY, apply_copies};
            let hook = victory_hle::port_descriptors::title_copy_hook(image.base() + TITLE_MEMCPY);
            // SAFETY: as above; the hook is a function of ours.
            let done = unsafe { apply_copies(image.base(), size, victory_hle::port_descriptors::marks_address(), hook) };
            let copies = done.iter().filter(|d| **d == Hooked::Done).count();
            println!("renderer port 2: {copies} of {} table copies of the title patched", done.len());
        }
    }
    // The frame rate (VICTORY_FRAME_INTERVAL_MS, off without it): our frame
    // clock ticks at that interval (victory-gpu), and the title is given
    // the matching rate in place of its setting's 30 (VICTORY_TITLE_FPS
    // overrides it; `off` leaves the title alone).
    #[cfg(windows)]
    if execute && let Some(ms) = victory_loader::frame_rate::asked_interval_ms() {
        use victory_loader::frame_rate::{Patched, SITES, apply, title_interval_us, title_rate};
        let asked = std::env::var("VICTORY_TITLE_FPS").ok();
        match title_rate(ms, asked.as_deref()) {
            Some(rate) => {
                let size = image.info().size_of_image as usize;
                // SAFETY: the image stays mapped and nothing of it has run yet.
                let done = unsafe { apply(image.base(), size, rate, &SITES) };
                let patched = done.iter().filter(|d| **d == Patched::Done).count();
                println!(
                    "frame rate: our clock {ms} ms; the title's rate {rate} ({patched} of {} reads of its setting patched; it asks for {} us at 60 Hz)",
                    SITES.len(),
                    title_interval_us(rate, false)
                );
            }
            None => println!("frame rate: our clock {ms} ms; the title's own rate (30) left as it is"),
        }
    }
    // The output size and render scale (VICTORY_OUTPUT, VICTORY_RENDER_SCALE,
    // VICTORY_UPSCALER; each off without it).
    #[cfg(windows)]
    if execute {
        let size = image.info().size_of_image as usize;
        // SAFETY: the image stays mapped and nothing of it has run yet.
        unsafe { victory_loader::resolution::apply_asked(image.base(), size, upscaler_default) };
    }
    // VICTORY_MAX_STEP_MS=<ms> (off without it): the title's longest
    // simulation step (1/15 s) raised, so the world keeps real speed below
    // 15 frames a second (victory-game's `step`). Only on the known build,
    // every site compared first (all or nothing).
    #[cfg(windows)]
    if execute && let Ok(spec) = std::env::var("VICTORY_MAX_STEP_MS") {
        match (victory_game::step::parse_ms(&spec), build) {
            (Err(why), _) => eprintln!("VIctory: {why}; the title's step is left as it is"),
            (Ok(_), None) => eprintln!("VIctory: VICTORY_MAX_STEP_MS: not a known build of the title; its step is left as it is"),
            (Ok(ms), Some(_)) => {
                let size = image.info().size_of_image as usize;
                let patches = victory_game::step::patches(ms);
                // SAFETY: the image stays mapped and nothing of it has run yet.
                match unsafe { victory_loader::resolution::write_all_checked(image.base(), size, &patches) } {
                    Ok(()) => {
                        for (rva, old, new) in &patches {
                            println!(
                                "title patch    +{rva:#x} longest step {ms} ms (VICTORY_MAX_STEP_MS): {} -> {}",
                                hex_str(old),
                                hex_str(new)
                            );
                        }
                        tracing::info!(ms, "title step: longest simulation step raised (VICTORY_MAX_STEP_MS)");
                    }
                    Err(why) => {
                        eprintln!("VIctory: VICTORY_MAX_STEP_MS: {why}; the title's step is left as it is");
                    }
                }
            }
        }
    }
    if !execute {
        println!("not executing (pass --execute to start the title)");
        return Ok(());
    }
    victory_loader::report_crashes();
    victory_loader::set_before_thread_shutdown(victory_hle::known_pipelines::flush);
    victory_loader::log_crashes_to(temp.with_file_name("crashes.log"));
    victory_hle::runtime::use_console_timers();
    #[cfg(windows)]
    step_log(image.base(), image.info().size_of_image as usize, build.is_some());
    #[cfg(feature = "diagnostics")]
    diagnostics(&image, &cfg);
    let late = LateDlls {
        host: resolver.host.clone(),
        title_dir: title_dir.to_owned(),
        cache_dir: images,
        policy,
        mapped: std::sync::Mutex::new(dlls.clone()),
        lock: LoaderLock::default(),
    };
    victory_hle::runtime::set_title_dll_loader(Box::new(move |file| {
        late.load(file, show).map_err(|e| format!("{e:#}"))
    }));
    // SAFETY: imports are bound; from here on we run title code.
    let main = unsafe { Arc::new(image).spawn_main(dlls)? };
    let code = main
        .join()
        .map_err(|_| anyhow::anyhow!("title main thread panicked"))??;
    victory_hle::known_pipelines::flush();
    println!("title exited with {code}");
    Ok(())
}

/// The debugging aids (README): scheduled stack dumps, watched values and
/// memory saved on a crash.
#[cfg(all(windows, feature = "diagnostics"))]
fn diagnostics(image: &victory_loader::LoadedImage, cfg: &victory_core::Config) {
    victory_loader::dump_stacks_on_schedule();
    victory_loader::profile_on_schedule(victory_hle::frame_phase);
    victory_hle::modules::audio::capture::start();
    watch(image.base(), image.info().size_of_image as usize);
    timers(image.base(), image.info().size_of_image as usize);
    poke(image.base(), image.info().size_of_image as usize);
    // Debugging aid: `VICTORY_CRASH_SAVE=<rva>:<len>,...` (hex) writes those
    // title ranges to `crash-<rva>.bin` in the cache dir on a crash.
    let size = image.info().size_of_image as usize;
    let ranges = std::env::var("VICTORY_CRASH_SAVE").unwrap_or_default();
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    victory_loader::save_on_crash(
        ranges
            .split(',')
            .filter_map(|r| r.split_once(':'))
            .filter_map(|(rva, len)| Some((hex(rva)?, hex(len)?)))
            .filter(|&(rva, len)| rva.checked_add(len).is_some_and(|end| end <= size))
            .map(|(rva, len)| {
                let file = cfg.cache_dir.join(format!("crash-{rva:x}.bin"));
                (image.base() + rva, len, file)
            })
            .collect(),
    );
}

/// Debugging aid: `VICTORY_WATCH=<rva>,...` (hex) logs the u32 at each
/// title+rva whenever it changes, polled every 5 ms (an engine's state, say).
#[cfg(all(windows, feature = "diagnostics"))]
fn watch(base: usize, size: usize) {
    let Ok(list) = std::env::var("VICTORY_WATCH") else {
        return;
    };
    let rvas: Vec<usize> = list
        .split(',')
        .filter_map(|r| usize::from_str_radix(r.trim().trim_start_matches("0x"), 16).ok())
        .filter(|&rva| rva + 4 <= size)
        .collect();
    std::thread::spawn(move || {
        let mut seen = vec![None; rvas.len()];
        loop {
            for (&rva, seen) in rvas.iter().zip(&mut seen) {
                // SAFETY: four bytes inside the mapped image, checked above.
                let value = unsafe { ((base + rva) as *const u32).read_volatile() };
                if *seen != Some(value) {
                    eprintln!("VIctory: title+{rva:#x} = {value:#x}");
                    *seen = Some(value);
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    });
}

/// Debugging aid: `VICTORY_TIMER_LOG=1` logs the title's timers once a
/// second (`victory_loader::frame_rate`): its frame time and limiter flag,
/// and each timer's two times, step in seconds and in milliseconds. Whether
/// game time keeps to the wall clock at any frame rate is read from these.
#[cfg(all(windows, feature = "diagnostics"))]
fn timers(base: usize, size: usize) {
    use victory_loader::frame_rate::{FRAME_US, LIMITER_ON, TIMERS};
    if std::env::var("VICTORY_TIMER_LOG").ok().as_deref() != Some("1") {
        return;
    }
    if TIMERS.iter().chain([&FRAME_US, &LIMITER_ON]).any(|&rva| rva + 0x14 > size) {
        return;
    }
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let u32_at = |rva: usize| {
            // SAFETY: inside the mapped image (checked above), title data.
            unsafe { ((base + rva) as *const u32).read_volatile() }
        };
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let timers: Vec<String> = TIMERS
                .iter()
                .map(|&t| {
                    let step = f32::from_bits(u32_at(t + 0xc));
                    format!("{} {} {step:.4} {}", u32_at(t), u32_at(t + 4), u32_at(t + 0x10))
                })
                .collect();
            eprintln!(
                "VIctory: the title's timers at {} ms: frame_us {} limiter {} frames {}; timers (time ms, frames, step s, step ms): {}",
                start.elapsed().as_millis(),
                u32_at(FRAME_US),
                u32_at(LIMITER_ON) & 0xff,
                victory_hle::frames_begun(),
                timers.join(" | ")
            );
        }
    });
}

/// Debugging aid: `VICTORY_POKE=<rva>:<byte>,...` (hex) writes the byte at
/// each title+rva twice a second. A title that ends itself when its main
/// loop has not ticked for a while (Grand Theft Auto VI: 30 s) runs on
/// through slower frames with its tick flag poked. The address must be
/// writable title data.
#[cfg(all(windows, feature = "diagnostics"))]
fn poke(base: usize, size: usize) {
    let Ok(list) = std::env::var("VICTORY_POKE") else {
        return;
    };
    let hex = |s: &str| usize::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    let pokes: Vec<(usize, u8)> = list
        .split(',')
        .filter_map(|p| p.split_once(':'))
        .filter_map(|(rva, byte)| Some((hex(rva)?, u8::try_from(hex(byte)?).ok()?)))
        .filter(|&(rva, _)| rva < size)
        .collect();
    std::thread::spawn(move || {
        loop {
            for &(rva, byte) in &pokes {
                // SAFETY: a byte inside the mapped image, checked above.
                unsafe { ((base + rva) as *mut u8).write_volatile(byte) };
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    });
}

/// DLLs the host or our runtime provide are never taken from the title.
#[cfg(windows)]
fn title_dll(host: &victory_hle::HostPolicy, dll: &str) -> bool {
    !host.allows(dll) && !victory_hle::registry().iter().any(|m| m.serves(dll))
}

/// The DLLs among `imported` (the main image's static imports) that must
/// come from the title's folder (`wanted`) but were not mapped from it:
/// none of their imports can be bound. Each named once, case ignored.
#[cfg(any(windows, test))]
fn absent_title_dlls(
    imported: &[&str],
    mapped: &[&str],
    wanted: &dyn Fn(&str) -> bool,
) -> Vec<String> {
    let mut absent: Vec<String> = Vec::new();
    for &dll in imported {
        let known = |list: &[&str]| list.iter().any(|m| m.eq_ignore_ascii_case(dll));
        if wanted(dll) && !known(mapped) && !absent.iter().any(|a| a.eq_ignore_ascii_case(dll)) {
            absent.push(dll.to_owned());
        }
    }
    absent
}

/// Title DLLs the title loads itself (`LoadLibrary`) that nothing imports,
/// e.g. The title's upscaler behind its FidelityFX loader.
#[cfg(windows)]
struct LateDlls {
    host: victory_hle::HostPolicy,
    title_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    policy: victory_loader::UnimplementedPolicy,
    /// Every title DLL mapped so far, to bind new ones against.
    mapped: std::sync::Mutex<Vec<(String, victory_loader::LoadedImage)>>,
    lock: LoaderLock,
}

#[cfg(windows)]
impl LateDlls {
    /// Every `LoadLibrary` of the title comes here first. Returns a title
    /// image called `file`, mapping it and the title DLLs it needs first,
    /// binding them and running their `DllMain` on the calling thread, as
    /// the Windows loader would; `None` if `file` is no title image.
    fn load(
        &self,
        file: &str,
        show: impl Fn(&str, victory_loader::BindReport),
    ) -> Result<Option<usize>> {
        use victory_hle::runtime::{forget_image, image_named, set_image_path};
        // Waits while another thread initialises a DLL; a `DllMain` that
        // loads another DLL re-enters.
        self.lock.hold(|| {
            if let Some(base) = image_named(file) {
                return Ok(Some(base));
            }
            let new = {
                let mut mapped = self.mapped.lock().unwrap_or_else(|e| e.into_inner());
                let wanted = |dll: &str| title_dll(&self.host, dll) && image_named(dll).is_none();
                let new = victory_loader::load_title_dll(
                    file,
                    &self.title_dir,
                    &self.cache_dir,
                    &wanted,
                )?;
                let hle = victory_hle::HleResolver::new(self.host.clone());
                let old = victory_loader::WithTitleDlls {
                    inner: &hle,
                    dlls: &mapped,
                };
                let resolver = victory_loader::WithTitleDlls {
                    inner: &old,
                    dlls: &new,
                };
                // All of them bound before any is known as an image: a load
                // that fails here leaves none behind, unattached, for the
                // next `LoadLibrary` of its name to return.
                let bound = new.iter().try_for_each(|(name, dll)| {
                    println!(
                        "title dll      {name} at {:#x} (loaded by the title)",
                        dll.base()
                    );
                    show(name, dll.bind_imports(&resolver, self.policy)?);
                    anyhow::Ok(())
                });
                if let Err(error) = bound {
                    for (_, dll) in new {
                        // SAFETY: never attached, and no title code holds it.
                        unsafe { dll.unload() };
                    }
                    return Err(error);
                }
                for (name, dll) in &new {
                    set_image_path(dll.base(), &self.title_dir.join(name));
                }
                // Visible before their `DllMain`s run, which may load more.
                mapped.extend(new.iter().cloned());
                new
            };
            let failed = new.iter().enumerate().find_map(|(i, (name, dll))| {
                // SAFETY: bound above, dependencies attached first.
                let attached = unsafe { dll.attach() };
                attached
                    .err()
                    .map(|e| (i, e.context(format!("attaching {name}"))))
            });
            let Some((i, error)) = failed else {
                return Ok(new.last().map(|(_, dll)| dll.base()));
            };
            // As `LoadLibrary` does when a `DllMain` fails: detach what this
            // load attached (the failed one too) and unmap all of it.
            let bases: Vec<_> = new.iter().map(|(_, d)| d.base()).collect();
            self.mapped
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(_, d)| !bases.contains(&d.base()));
            for (_, done) in new[..=i].iter().rev() {
                // SAFETY: attached (or failing to) on this thread just now.
                unsafe { done.detach() };
            }
            for (_, dll) in new {
                forget_image(dll.base());
                // SAFETY: detached, and no title code holds it yet.
                unsafe { dll.unload() };
            }
            Err(error)
        })
    }
}

/// The Windows loader lock's shape: one thread at a time, re-entered by
/// the thread that holds it.
#[cfg(windows)]
#[derive(Default)]
struct LoaderLock {
    owner: std::sync::Mutex<(Option<std::thread::ThreadId>, usize)>,
    free: std::sync::Condvar,
}

#[cfg(windows)]
impl LoaderLock {
    fn hold<T>(&self, f: impl FnOnce() -> T) -> T {
        let me = std::thread::current().id();
        let lock = || self.owner.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut owner = lock();
            while owner.0.is_some_and(|o| o != me) {
                owner = self.free.wait(owner).unwrap_or_else(|e| e.into_inner());
            }
            *owner = (Some(me), owner.1 + 1);
        }
        let result = f();
        let mut owner = lock();
        owner.1 -= 1;
        if owner.1 == 0 {
            owner.0 = None;
            self.free.notify_one();
        }
        result
    }
}

/// `VICTORY_STEP_LOG=1` (off without it): every 5 s, how far the title's
/// game time (victory-game's `step::GAME_TIMER`) moved against the wall
/// clock. `game/wall` 1.00 is real speed; below 1 the title's longest step
/// (1/15 s, `VICTORY_MAX_STEP_MS`) holds it back. Only on the known build
/// (`known`: the timer's offset is that build's). Read only, from a thread of
/// its own, sampling words the title writes without a lock: a sample counts
/// only if the time word reads the same before and after the other two (a
/// sample torn by the title's update is taken again). Works with or without
/// the patch. A diagnostic, not part of the patch.
#[cfg(windows)]
fn step_log(base: usize, size: usize, known: bool) {
    if std::env::var_os("VICTORY_STEP_LOG").is_none_or(|v| v != "1") {
        return;
    }
    if !known {
        eprintln!("VIctory: VICTORY_STEP_LOG: not a known build of the title; no step log");
        return;
    }
    let at = victory_game::step::GAME_TIMER as usize;
    if at + 0x10 > size {
        return;
    }
    let timer = base + at;
    let spawned = std::thread::Builder::new()
        .name("victory-step-log".into())
        .spawn(move || {
            // SAFETY: inside the mapped image (checked above), which stays
            // mapped for the process; plain reads of words the title writes.
            let read = || loop {
                // SAFETY: as above.
                let sample = unsafe {
                    let time = std::ptr::read_volatile(timer as *const u32);
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                    let frames = std::ptr::read_volatile((timer + 4) as *const u32);
                    let step = std::ptr::read_volatile((timer + 0xc) as *const f32);
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                    (time, frames, step, std::ptr::read_volatile(timer as *const u32))
                };
                if sample.0 == sample.3 {
                    break (sample.0, sample.1, sample.2);
                }
                std::thread::yield_now();
            };
            let mut since = (std::time::Instant::now(), read());
            let mut longest = 0f32;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let now = read();
                longest = longest.max(now.2);
                let wall = since.0.elapsed();
                if wall < std::time::Duration::from_secs(5) {
                    continue;
                }
                let wall_ms = wall.as_secs_f64() * 1000.0;
                let game_ms = now.0.wrapping_sub(since.1.0);
                let frames = now.1.wrapping_sub(since.1.1);
                tracing::info!(
                    "title step: game/wall {:.2} (game time {} ms in {:.0} ms), {} frames by its counter, mean step {:.1} ms, longest {:.1} ms (VICTORY_STEP_LOG)",
                    f64::from(game_ms) / wall_ms,
                    game_ms,
                    wall_ms,
                    frames,
                    if frames > 0 { f64::from(game_ms) / f64::from(frames) } else { 0.0 },
                    longest * 1000.0,
                );
                since = (std::time::Instant::now(), now);
                longest = 0.0;
            }
        });
    if let Err(e) = spawned {
        eprintln!("VIctory: VICTORY_STEP_LOG: no thread for it ({e})");
    }
}

/// Bytes as lower-case hex.
#[cfg(windows)]
fn hex_str(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether the known build's Series S upscaler choice applies, given
/// `VICTORY_TITLE_PATCHES` (`var`): `0` never, `1` on any console; unset
/// (or anything else), only with `--console series-s`. On a Series X the
/// pickers already return what is written.
fn known_patches_on(console: Console, var: Option<&str>) -> bool {
    match var.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => matches!(console, Console::SeriesS),
    }
}

#[cfg(test)]
mod absent_tests {
    use super::absent_title_dlls;

    /// The known build's upscaler choice: on with `--console series-s`, off
    /// on a Series X unless asked for, and off everywhere with `0`.
    #[test]
    fn title_patches_follow_the_console_and_the_switch() {
        use super::{Console, known_patches_on};
        assert!(known_patches_on(Console::SeriesS, None));
        assert!(!known_patches_on(Console::SeriesX, None));
        assert!(!known_patches_on(Console::SeriesS, Some("0")));
        assert!(known_patches_on(Console::SeriesX, Some("1")));
    }

    #[test]
    fn title_dlls_missing_from_the_folder_are_named_once() {
        // The host provides kernel32, our runtime d3d12_xs; the rest are the
        // title's own (#930: the four DLLs beside the executable missing).
        let wanted = |dll: &str| {
            !["KERNEL32.dll", "d3d12_xs.dll"]
                .iter()
                .any(|h| h.eq_ignore_ascii_case(dll))
        };
        let imported = [
            "KERNEL32.dll",
            "d3d12_xs.dll",
            "XboxBvhBuilder.dll",
            "MSVCP140.dll",
            "Bink2Scarlett.dll",
            "xboxbvhbuilder.dll",
        ];
        let mapped = ["msvcp140.dll"];
        assert_eq!(
            absent_title_dlls(&imported, &mapped, &wanted),
            ["XboxBvhBuilder.dll", "Bink2Scarlett.dll"]
        );
    }

    #[test]
    fn a_complete_package_has_nothing_absent() {
        let wanted = |dll: &str| dll.starts_with("Title");
        assert!(absent_title_dlls(&["Title.dll", "Host.dll"], &["TITLE.DLL"], &wanted).is_empty());
        assert!(absent_title_dlls(&[], &[], &wanted).is_empty());
    }
}

#[cfg(all(test, windows))]
mod tests {
    use std::sync::{Arc, Mutex};

    #[test]
    fn loader_lock_is_reentrant_and_exclusive() {
        let lock = Arc::new(super::LoaderLock::default());
        assert_eq!(lock.hold(|| lock.hold(|| 7)), 7);
        let order = Arc::new(Mutex::new(Vec::new()));
        let other = lock.hold(|| {
            let (theirs, their_order) = (lock.clone(), order.clone());
            let other = std::thread::spawn(move || {
                theirs.hold(|| their_order.lock().unwrap().push(2));
            });
            std::thread::sleep(std::time::Duration::from_millis(50));
            order.lock().unwrap().push(1);
            other
        });
        other.join().unwrap();
        assert_eq!(*order.lock().unwrap(), [1, 2]);
    }

    #[test]
    fn cpu_feature_check_matches_host() {
        let skip = std::env::var(super::SKIP_CPU_CHECK_ENV).ok();
        #[cfg(target_arch = "x86_64")]
        let expected_ok =
            std::arch::is_x86_feature_detected!("avx2") || skip.as_deref() == Some("1");
        #[cfg(not(target_arch = "x86_64"))]
        let expected_ok = true;
        assert_eq!(super::check_cpu_features().is_ok(), expected_ok);
    }
}

#[cfg(test)]
mod cpu_check_tests {
    use super::{SKIP_CPU_CHECK_ENV, check_cpu_features_with};

    #[test]
    fn avx2_present_passes() {
        assert!(check_cpu_features_with(true, None).is_ok());
        assert!(check_cpu_features_with(true, Some("0")).is_ok());
        assert!(check_cpu_features_with(true, Some("1")).is_ok());
    }

    #[test]
    fn avx2_missing_fails_with_clear_error() {
        for skip in [None, Some(""), Some("0"), Some("true")] {
            let err = check_cpu_features_with(false, skip).unwrap_err();
            let msg = err.to_string();
            assert!(msg.contains("AVX2"), "unexpected error: {msg}");
            assert!(msg.contains("Xbox Series"), "unexpected error: {msg}");
            assert!(msg.contains(SKIP_CPU_CHECK_ENV), "unexpected error: {msg}");
        }
    }

    #[test]
    fn skip_env_var_bypasses_missing_avx2() {
        assert!(check_cpu_features_with(false, Some("1")).is_ok());
    }
}

#[cfg(not(windows))]
pub fn run(_: &Path, _: Unimplemented, _: bool, _: bool, _: Console) -> Result<()> {
    anyhow::bail!("running titles requires Windows")
}
