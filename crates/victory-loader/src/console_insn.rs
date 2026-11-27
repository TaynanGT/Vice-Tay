//! Instructions the console's CPU (Zen 2) has and an Intel PC's lacks, emulated: the title's
//! code raises STATUS_ILLEGAL_INSTRUCTION on them (the title zeroes memory with a `clzero` loop at
//! its first frames), and the vectored handler steps over them here.
//!
//! Ported from XCL (github.com/ivall/xcl, GPL-2.0-only): `insn.rs` (decoder, as
//! `console_insn_decode.rs`, unchanged) and `eh.rs` `emulate_console_only`.

#[path = "console_insn_decode.rs"]
mod decode;

use decode::ConsoleOnly;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;

const CONTINUE_EXECUTION: i32 = -1;

/// Whether what Windows said about memory is remembered per thread (the regions it found
/// writable, the code bytes it found readable) and not asked again at every emulated
/// instruction. A query walks the pages of the region it answers for: asked from outside, it
/// took 2-9 ms in the title's heaps of 0.6-2.2 GB and 89 ms in its 6.2 GB page pool (run 30),
/// and the title zeroes blocks of them with `clzero` loops all through a frame: six of its
/// threads spent 25-78 % of their time inside that query (runs 29 and 30), about 27 s of
/// processor time in Windows per 9 s frame. On by default; `VICTORY_CLZERO_ASK=1` asks every
/// time, as before.
fn remembers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLZERO_ASK").is_none_or(|v| v != "1"))
}

/// Answers a remembered region gives before Windows is asked about it again: one freed or
/// protected meanwhile is forgotten then.
const ANSWERS: u32 = 4096;

thread_local! {
    /// The regions Windows last said are committed and writable (start, end), newest first,
    /// and the answers they gave since they were emptied.
    static WRITABLE: Cell<([(usize, usize); 4], u32)> = const { Cell::new(([(0, 0); 4], 0)) };
    /// Faulting addresses and the code bytes found readable there, newest first.
    static CODE: Cell<[(usize, usize); 8]> = const { Cell::new([(0, 0); 8]) };
}

/// Instructions emulated, the times Windows was asked about memory for them, the bytes
/// `clzero` loops zeroed, and lines zeroed through a fault, since the start.
static EMULATED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);
static THROUGH_FAULT: AtomicU64 = AtomicU64::new(0);

/// Whether `monitorx` and `mwaitx` are replaced in the title's code the first time they fault
/// (`monitorx` by a 3-byte no-op, `mwaitx` by `pause` and a no-op), so that they never fault
/// again. Emulated, each pass of such a wait loop is two trips into Windows and back, through
/// every vectored handler, and no lighter a wait for it: the emulation does no more than
/// `pause` either, and the loop looks at its value again after it. On by default;
/// `VICTORY_PATCH_WAITS=0` emulates each time, as before.
fn patches_waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_WAITS").is_none_or(|v| v != "0"))
}

/// What replaces the 3 bytes of a wait instruction.
fn wait_patch(what: ConsoleOnly) -> Option<[u8; 3]> {
    match what {
        ConsoleOnly::Monitorx => Some([0x0f, 0x1f, 0x00]),
        ConsoleOnly::Mwaitx => Some([0xf3, 0x90, 0x90]),
        _ => None,
    }
}

/// Whether a `clzero` loop ([`ConsoleOnly::ClzeroLoop`]) is replaced the first time it faults
/// by a jump to the same loop with plain stores ([`clzero_stub`]). Run cpu-5 (the dorm): 25,000
/// exceptions a second, 2,100 a frame, at five such loops of one function of the title, for
/// some kilobytes each; ten times what the wait instructions raised. On by default;
/// `VICTORY_PATCH_CLZERO=0` emulates each time, as before.
fn patches_clzero() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_CLZERO").is_none_or(|v| v != "0"))
}

/// `jmp rel32` at `at` to `to`, if it reaches.
pub(crate) fn jump(at: usize, to: usize) -> Option<[u8; 5]> {
    let rel = i32::try_from(to as i64 - (at as i64 + 5)).ok()?;
    let [a, b, c, d] = rel.to_le_bytes();
    Some([0xe9, a, b, c, d])
}

/// The `clzero` loop of 12 bytes at `at` with plain stores, as code at `stub`: every register
/// and the flags come out as the loop leaves them. `cmp` is its `cmp rax, reg` (3 bytes, the
/// register not `rax`), `None` for another form. (The scratch register is saved on the stack
/// and back before the compare, whose register it may be.)
fn clzero_stub(at: usize, stub: usize, cmp: [u8; 3]) -> Option<Vec<u8>> {
    if !matches!(cmp[0], 0x48 | 0x4c) || cmp[1] != 0x39 || cmp[2] & 0xc7 != 0xc0 || (cmp[0] == 0x48 && cmp[2] == 0xc0) {
        return None;
    }
    let mut code = vec![
        0x51, // push rcx
        0x48, 0x89, 0xc1, // mov rcx, rax
        0x48, 0x83, 0xe1, 0xc0, // and rcx, -64
    ];
    for k in 0..8 {
        code.extend([0x48, 0xc7, 0x41, k * 8, 0, 0, 0, 0]); // mov qword [rcx + 8k], 0
    }
    code.push(0x59); // pop rcx
    code.extend([0x48, 0x83, 0xc0, 0x40]); // add rax, 0x40
    code.extend(cmp);
    let back = -(code.len() as i64 + 2);
    code.extend([0x72, i8::try_from(back).ok()? as u8]); // jb the start
    let end = stub + code.len();
    code.extend(jump(end, at + 12)?);
    Some(code)
}

/// Room for [`clzero_stub`]s near the title's code: where it is and the bytes used.
static STUBS: std::sync::Mutex<Option<(usize, usize)>> = std::sync::Mutex::new(None);
pub(crate) const STUBS_LEN: usize = 64 << 10;

/// Memory for code within reach of a `jmp rel32` at `at`.
pub(crate) fn near(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
    let at = at & !0xffff;
    (0x10_0000..0x7000_0000).step_by(0x10_0000).flat_map(|d| [at.checked_sub(d), at.checked_add(d)]).flatten().find_map(|place| {
        // SAFETY: reserves free memory at that address or fails.
        let got = unsafe { VirtualAlloc(place as *const _, STUBS_LEN, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) } as usize;
        (got != 0).then_some(got)
    })
}

/// Replaces the `clzero` loop at `at`, whose 12 bytes are `code`: whether it did.
///
/// No thread is ever inside the loop's other instructions (the emulation does the whole loop
/// in one), so the five bytes of the jump may take the place of two of them.
///
/// # Safety
/// `at` is in the title's code, which stays mapped.
unsafe fn patch_clzero(at: usize, code: &[u8]) -> bool {
    let &[0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, x, y, z, 0x72, 0xf4] = code else {
        return false;
    };
    if (at & 15) + 5 > 16 {
        return false;
    }
    let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
    if stubs.is_none() {
        *stubs = near(at).map(|place| (place, 0));
    }
    let Some((place, used)) = stubs.as_mut() else {
        return false;
    };
    let stub = *place + *used;
    let (Some(body), Some(jump)) = (clzero_stub(at, stub, [x, y, z]), jump(at, stub)) else {
        return false;
    };
    if *used + body.len() > STUBS_LEN {
        return false;
    }
    // SAFETY: inside the memory made for it, which nothing runs yet; then the title's code.
    unsafe {
        std::ptr::copy_nonoverlapping(body.as_ptr(), stub as *mut u8, body.len());
        *used += body.len().next_multiple_of(16);
        patch(at, [0x0f, 0x01, 0xfc, 0x48, 0x83], jump)
    }
}

/// The aligned 16 bytes `word` around address `at` with the bytes at `at` replaced; `None`
/// where they do not all lie in those 16. (16 and not 8: the title's busiest one lies over the
/// end of its 8 and took 28,000 exceptions a second, run cpu-6.)
fn with_patch(word: u128, at: usize, patch: &[u8]) -> Option<u128> {
    let offset = at & 15;
    if offset + patch.len() > 16 {
        return None;
    }
    let mut bytes = word.to_le_bytes();
    bytes[offset..offset + patch.len()].copy_from_slice(patch);
    Some(u128::from_le_bytes(bytes))
}

/// One thread at a time changes code: a second one would take the first one's loosened
/// protection for the page's own.
static PATCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Replaces the bytes at `at`, which are `was`, by `patch`, in one store that other threads
/// see whole: whether they are replaced now (not where they cross 16 aligned bytes, or the page
/// cannot be written).
///
/// # Safety
/// `at` is in code that stays mapped, and `patch` does for it what `was` does.
unsafe fn patch<const N: usize>(at: usize, was: [u8; N], patch: [u8; N]) -> bool {
    use std::sync::atomic::Ordering::SeqCst;
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if (at & 15) + N > 16 || !std::is_x86_feature_detected!("cmpxchg16b") {
        return false;
    }
    let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
    let word = (at & !15) as *mut u128;
    let here = (at & 15)..(at & 15) + N;
    let mut old = 0u32;
    // SAFETY: the aligned 16 bytes around the instruction lie in one page of mapped code, made
    // writable for the store and put back; the processor has the instruction (asked above).
    unsafe {
        if VirtualProtect(word.cast(), 16, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        // What is there (zeros are replaced by zeros).
        let mut now = std::arch::x86_64::cmpxchg16b(word, 0, 0, SeqCst, SeqCst);
        let done = loop {
            let bytes = &now.to_le_bytes()[here.clone()];
            if bytes == patch {
                // (Another thread's fault at the same place: done already.)
                break true;
            }
            let Some(new) = (bytes == was).then(|| with_patch(now, at, &patch)).flatten() else {
                break false;
            };
            let there = std::arch::x86_64::cmpxchg16b(word, now, new, SeqCst, SeqCst);
            if there == now {
                break true;
            }
            now = there;
        };
        VirtualProtect(word.cast(), 16, old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), word.cast(), 16);
        done
    }
}

/// Diagnostics: how often each place in the title's code raises one of these exceptions, the
/// busiest logged every five seconds.
#[cfg(feature = "diagnostics")]
mod sites {
    use super::{AtomicU64, Relaxed};
    use std::sync::atomic::AtomicUsize;

    /// Address, what is there (an index into [`NAMES`]), exceptions since the last line.
    static SITES: [(AtomicUsize, AtomicUsize, AtomicU64); 512] =
        [const { (AtomicUsize::new(0), AtomicUsize::new(0), AtomicU64::new(0)) }; 512];
    const NAMES: [&str; 8] = ["other", "clwb", "monitorx", "mwaitx", "clzero", "clzero loop", "rdpru", "extrq/insertq"];
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    static PATCHED: AtomicU64 = AtomicU64::new(0);

    pub(super) fn patched() {
        PATCHED.fetch_add(1, Relaxed);
    }

    pub(super) fn count(at: usize, what: super::ConsoleOnly) {
        use super::ConsoleOnly as C;
        let kind = match what {
            C::Clwb => 1,
            C::Monitorx => 2,
            C::Mwaitx => 3,
            C::Clzero => 4,
            C::ClzeroLoop { .. } => 5,
            C::Rdpru => 6,
            C::Extrq { .. } | C::Insertq { .. } => 7,
            _ => 0,
        };
        let mut i = (at ^ at >> 9) % SITES.len();
        for _ in 0..8 {
            let s = &SITES[i];
            let there = s.0.compare_exchange(0, at, Relaxed, Relaxed).unwrap_or_else(|there| there);
            if there == 0 || there == at {
                s.1.store(kind, Relaxed);
                s.2.fetch_add(1, Relaxed);
                break;
            }
            i = (i + 1) % SITES.len();
        }
        // Now and then: is a line due? Never waited for.
        if let Ok(mut last) = LAST.try_lock() {
            let since = *last.get_or_insert_with(std::time::Instant::now);
            if since.elapsed().as_secs() >= 5 {
                let s = since.elapsed().as_secs_f64();
                *last = Some(std::time::Instant::now());
                drop(last);
                line(s);
            }
        }
    }

    fn line(seconds: f64) {
        let mut all: Vec<(u64, usize, usize)> =
            SITES.iter().map(|s| (s.2.swap(0, Relaxed), s.0.load(Relaxed), s.1.load(Relaxed))).filter(|s| s.0 > 0).collect();
        all.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
        let mut kinds = [0u64; NAMES.len()];
        for s in &all {
            kinds[s.2] += s.0;
        }
        let kinds: Vec<String> = NAMES
            .iter()
            .zip(kinds)
            .filter(|(_, n)| *n > 0)
            .map(|(name, n)| format!("{name} {:.0}", n as f64 / seconds))
            .collect();
        let top: Vec<String> =
            all.iter().take(8).map(|s| format!("{:#x} {} {:.0}", s.1, NAMES[s.2], s.0 as f64 / seconds)).collect();
        tracing::info!(
            "console CPU instructions that fault, a second: {}; places replaced so far: {}; the busiest places: {}",
            kinds.join(", "),
            PATCHED.load(Relaxed),
            top.join(", ")
        );
    }
}

fn readable(va: usize, bytes: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    found && m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

fn writable(va: usize, bytes: usize) -> bool {
    writable_to(va, va + bytes) == Some(va + bytes)
}

/// Where the committed, writable memory that `va` lies in ends, at most `end`; None if `va`
/// is not in such memory. A region found is remembered ([`remembers`]).
/// shortcut: a remembered region that was freed or protected since is written all the same
/// for up to [`ANSWERS`] answers: the write then faults inside the handler, which the guard
/// on view memory opens as it does any write, and which is a crash otherwise (as the
/// unchecked single-line emulation behind this one already is).
fn writable_to(va: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE, VirtualQuery,
    };
    let remember = remembers();
    if remember {
        let (regions, answers) = WRITABLE.with(Cell::get);
        if let Some(&(_, to)) = regions.iter().find(|&&(from, to)| from <= va && va < to) {
            let next = if answers >= ANSWERS { ([(0, 0); 4], 0) } else { (regions, answers + 1) };
            WRITABLE.with(|w| w.set(next));
            return Some(to.min(end));
        }
    }
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    if !found || m.State != MEM_COMMIT || m.Protect & PAGE_GUARD != 0 || m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 {
        return None;
    }
    let (from, to) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
    if remember {
        WRITABLE.with(|w| {
            let (mut regions, answers) = w.get();
            regions.rotate_right(1);
            regions[0] = (from, to);
            w.set((regions, answers));
        });
    }
    Some(to.min(end))
}

/// Zeroes the 64-byte line at `line` with ordinary stores, for a line [`writable`] refused that
/// is committed or reserved all the same: write-protected on purpose (`PAGE_READONLY`,
/// `PAGE_NOACCESS`, `PAGE_GUARD`) or reserved for demand commit, as the write-watch guards over
/// the host copies of view memory make it (`watch.rs`), whose vectored handler is first in the
/// chain and opens a guarded block for the title's own stores. The console's `clzero` faults into
/// such a handler as any store does; here it is an illegal instruction the handler never sees, so
/// refusing the line left `STATUS_ILLEGAL_INSTRUCTION` (`0xc000001d`) unhandled and ended the run.
/// These stores fault inside this handler instead, the runtime's VEH opens or commits the block,
/// and the stores complete (a fault nothing resolves ends the run, as the title's own store
/// would). Returns `false`, with nothing written, for `MEM_FREE` / unmapped memory: the exception
/// stays the title's.
///
/// # Safety
/// `line` is 64-byte aligned and owned by the title, as the instruction being emulated targets it.
unsafe fn zero_through_fault(line: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(line as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    if !found || (m.State != MEM_COMMIT && m.State != MEM_RESERVE) {
        return false;
    }
    let words = line as *mut u64;
    for i in 0..8 {
        // SAFETY: see the function's contract; a fault here is resolved by the runtime's
        // handlers or ends the run, as the title's own store would.
        unsafe { words.add(i).write_volatile(0) };
    }
    THROUGH_FAULT.fetch_add(1, Relaxed);
    true
}

/// An illegal instruction at `ctx.Rip`: a console-only one is emulated and stepped over
/// (`CONTINUE_EXECUTION`); anything else returns 0 (the crash report and the title's handlers
/// see it).
///
/// # Safety
/// `ctx` is the faulting thread's context, from the exception being dispatched.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    // The same few instructions fault again and again: their bytes are asked about once.
    let known = if remembers() { CODE.with(Cell::get) } else { [(0, 0); 8] };
    let n = match known.iter().find(|&&(at, n)| at == rip && n > 0) {
        Some(&(_, n)) => n,
        None => {
            if !readable(rip, 1) {
                return 0;
            }
            let n = (1..=16).rev().find(|&n| readable(rip, n)).unwrap_or(0);
            if remembers() {
                let mut known = known;
                known.rotate_right(1);
                known[0] = (rip, n);
                CODE.with(|c| c.set(known));
            }
            n
        }
    };
    // SAFETY: `n` bytes at `rip` are readable.
    let code = unsafe { std::slice::from_raw_parts(rip as *const u8, n) };
    let Some((what, len)) = decode::decode(code) else { return 0 };
    static SEEN: std::sync::Mutex<Vec<std::mem::Discriminant<ConsoleOnly>>> = std::sync::Mutex::new(Vec::new());
    if let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&std::mem::discriminant(&what))
    {
        seen.push(std::mem::discriminant(&what));
        if what != ConsoleOnly::Ud2 {
            tracing::warn!(instruction = ?what, at = format_args!("{rip:#x}"), "a console CPU instruction this CPU lacks: emulated (logged once)");
        }
    }
    #[cfg(feature = "diagnostics")]
    if what != ConsoleOnly::Ud2 {
        sites::count(rip, what);
    }
    match what {
        ConsoleOnly::Ud2 => return 0,
        ConsoleOnly::Clwb => {
            // nothing to write back on the PC: patched to NOPs, no exception next time
            use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
            let mut old = 0u32;
            let _one = PATCHING.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: the instruction's own bytes, made writable for the patch and restored.
            unsafe {
                if VirtualProtect(rip as *const _, len, PAGE_EXECUTE_READWRITE, &mut old) != 0 {
                    std::slice::from_raw_parts_mut(rip as *mut u8, len).fill(0x90);
                    VirtualProtect(rip as *const _, len, old, &mut old);
                }
            }
        }
        ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {
            // Without a prefix, so 3 bytes: replaced, the next pass does not fault.
            if len == 3
                && patches_waits()
                && let Some(new) = wait_patch(what)
                // SAFETY: the title's code, where the instruction just faulted; both
                // replacements are waits that wait for nothing, as the emulation is.
                && unsafe { patch(rip, [code[0], code[1], code[2]], new) }
            {
                #[cfg(feature = "diagnostics")]
                sites::patched();
            }
            std::hint::spin_loop();
        }
        ConsoleOnly::Clzero => {
            // with an address-size prefix (0x67) the line is at EAX, zero-extended
            let a32 = code[..len - 3].contains(&0x67);
            let line = (if a32 { ctx.Rax & 0xFFFF_FFFF } else { ctx.Rax } as usize) & !63;
            if writable(line, 64) {
                // SAFETY: the 64-byte line is committed and writable.
                unsafe { std::ptr::write_bytes(line as *mut u8, 0, 64) };
            } else {
                // SAFETY: `line` is 64-byte aligned and targeted by the emulated `clzero`.
                if !unsafe { zero_through_fault(line) } {
                    return 0; // free memory: the real instruction faults there too
                }
            }
        }
        ConsoleOnly::ClzeroLoop { end_reg } => {
            let regs: [u64; 16] = [
                ctx.Rax, ctx.Rcx, ctx.Rdx, ctx.Rbx, ctx.Rsp, ctx.Rbp, ctx.Rsi, ctx.Rdi, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.R12, ctx.R13, ctx.R14, ctx.R15,
            ];
            let end = regs[end_reg as usize & 15] as usize;
            let start = ctx.Rax as usize;
            // A pass zeroes the whole line `rax` is in, wherever in it, and the test comes
            // after it: lines are zeroed past an `end` inside one, and one is for an `end`
            // not above the start. (The bytes from `start` to `end` were, `rax` came out as
            // `end`, and the flags stayed what they were.)
            const MOST: usize = (2 << 30) / 64;
            let wanted = if end > start { (end - start).div_ceil(64) } else { 1 };
            let passes = wanted.min(MOST);
            let first = start & !63;
            let Some(stop) = first.checked_add(passes * 64) else { return 0 };
            // Region by region: a block may lie across several (the title's pages are
            // mapped one view each), and one answer for all of it was a refusal, after
            // which every 64-byte line took an exception of its own.
            let mut at = first;
            while at < stop {
                let Some(to) = writable_to(at, stop) else { break };
                // SAFETY: committed and writable from `at` to `to`.
                unsafe { std::ptr::write_bytes(at as *mut u8, 0, to - at) };
                at = to;
            }
            let mut done = (at - first) / 64;
            if done == 0 {
                // Its first line is not writable: zero that line through a fault so the
                // runtime's write-watch / demand-commit handler can open it, then the loop
                // goes on after it (or finishes if it was the only line).
                // SAFETY: the line the instruction zeroes.
                if !unsafe { zero_through_fault(first) } {
                    return 0;
                }
                done = 1;
            }
            // The whole `clzero; add rax,40h; cmp rax,end; jb` loop at once (and not again:
            // the next time it is plain stores). Done after confirming the target is mapped so
            // a declined loop on free memory leaves the title's code untouched.
            if len == 12
                && patches_clzero()
                // SAFETY: the title's code, where the loop's first instruction just faulted.
                && unsafe { patch_clzero(rip, &code[..12]) }
            {
                #[cfg(feature = "diagnostics")]
                sites::patched();
            }
            ZEROED.fetch_add((done * 64) as u64, Relaxed);
            ctx.Rax = ctx.Rax.wrapping_add((done * 64) as u64);
            if done < wanted {
                // What follows is not writable (or the block is huge): the loop goes on
                // there (the same instruction again), where the real one faults too.
                return CONTINUE_EXECUTION;
            }
            ctx.EFlags = ctx.EFlags & !crate::crash::CMP_FLAGS | crate::crash::cmp_flags(ctx.Rax, end as u64);
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            // SAFETY: a valid out pointer.
            unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            // SAFETY: no preconditions.
            let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u64;
            let regs: [&mut u64; 16] = [
                &mut ctx.Rax, &mut ctx.Rcx, &mut ctx.Rdx, &mut ctx.Rbx, &mut ctx.Rsp, &mut ctx.Rbp, &mut ctx.Rsi, &mut ctx.Rdi, &mut ctx.R8, &mut ctx.R9, &mut ctx.R10, &mut ctx.R11, &mut ctx.R12, &mut ctx.R13,
                &mut ctx.R14, &mut ctx.R15,
            ];
            *regs[r as usize & 15] = cpu;
        }
        // the low quadword changes; the high one is zeroed, as a Zen 5 does
        ConsoleOnly::Extrq { dst, src, len, idx } => {
            // SAFETY: the context's XMM registers (CONTEXT_FLOATING_POINT is captured for exceptions).
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let (len, idx) = match src {
                Some(s) => {
                    let c = x[s as usize & 15].Low;
                    (c as u8, (c >> 8) as u8)
                }
                None => (len, idx),
            };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::extrq(d.Low, len, idx);
            d.High = 0;
        }
        ConsoleOnly::Insertq { dst, src, reg, len, idx } => {
            // SAFETY: as above.
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let s = x[src as usize & 15];
            let (len, idx) = if reg { (s.High as u8, (s.High >> 8) as u8) } else { (len, idx) };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::insertq(d.Low, s.Low, len, idx);
            d.High = 0;
        }
    }
    ctx.Rip += len as u64;
    // How often this runs at all (each one is an exception: a trip into Windows and back).
    let emulated = EMULATED.fetch_add(1, Relaxed) + 1;
    if emulated >= 1 << 16 && emulated.is_power_of_two() {
        tracing::info!(
            emulated,
            asked = ASKED.load(Relaxed),
            zeroed_mb = ZEROED.load(Relaxed) >> 20,
            through_fault = THROUGH_FAULT.load(Relaxed),
            "console CPU instructions emulated so far, the times Windows was asked about memory for them, and what clzero loops zeroed"
        );
    }
    CONTINUE_EXECUTION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_instructions_are_replaced_inside_sixteen_aligned_bytes() {
        let word = u128::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 0x0f, 0x01, 0xfb, 11, 12, 13, 14, 15, 16]);
        let new = with_patch(word, 0x1007, &[0xf3, 0x90, 0x90]).unwrap();
        assert_eq!(new.to_le_bytes(), [1, 2, 3, 4, 5, 6, 7, 0xf3, 0x90, 0x90, 11, 12, 13, 14, 15, 16]);
        assert!(with_patch(word, 0x100d, &[0; 3]).is_some());
        assert!(with_patch(word, 0x100e, &[0; 3]).is_none(), "crosses into the next sixteen");
    }

    /// `monitorx; mwaitx; ret` in a page of its own: each is replaced at its first fault,
    /// and the page is protected as before.
    #[test]
    fn a_wait_faults_once_and_the_page_keeps_its_protection() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            VirtualProtect, VirtualQuery,
        };
        // SAFETY: memory of our own, read, written and run within its page.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>();
            assert!(!page.is_null());
            let code = [0x0f, 0x01, 0xfa, 0x0f, 0x01, 0xfb, 0xc3];
            // The first over the end of its aligned eight bytes, as the title's is; and one
            // over the end of its sixteen (at 46), which is left alone.
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(6), code.len());
            std::ptr::copy_nonoverlapping(code.as_ptr(), page.add(46), code.len());
            let mut old = 0;
            assert_ne!(VirtualProtect(page.cast(), 4096, PAGE_EXECUTE_READ, &mut old), 0);
            let at = page as usize;
            let mut context: CONTEXT = std::mem::zeroed();
            for rip in [at + 6, at + 9, at + 46] {
                context.Rip = rip as u64;
                assert_eq!(emulate(&mut context), CONTINUE_EXECUTION);
                assert_eq!(context.Rip, rip as u64 + 3);
            }
            if patches_waits() {
                let now = std::slice::from_raw_parts(page.add(6), 7);
                assert_eq!(now, [0x0f, 0x1f, 0x00, 0xf3, 0x90, 0x90, 0xc3]);
                // They run now.
                let run: extern "system" fn() = std::mem::transmute(page.add(6));
                run();
                // A second thread that faulted at the same place meanwhile.
                assert!(patch(at + 6, [0x0f, 0x01, 0xfa], [0x0f, 0x1f, 0x00]));
                assert!(!patch(at + 46, [0x0f, 0x01, 0xfa], [0x0f, 0x1f, 0x00]), "crosses its sixteen bytes");
            }
            assert_eq!(std::slice::from_raw_parts(page.add(46), 3), [0x0f, 0x01, 0xfa]);
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(page.cast(), &mut m, size_of_val(&m));
            assert_eq!(m.Protect, PAGE_EXECUTE_READ);
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }
    }

    /// `clzero` loops run as plain stores: the lines they name are zero and no other byte,
    /// and the pointer, the scratch register and the flags come out as from the loop.
    #[test]
    fn a_clzero_loop_becomes_plain_stores() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // The native loop test can initialise the shared stub bank first.
        // Reserve the fixture within reach of that bank on either CPU.
        let stub_base = {
            let mut stubs = STUBS.lock().unwrap_or_else(|e| e.into_inner());
            stubs
                .get_or_insert_with(|| {
                    let anchor = a_clzero_loop_becomes_plain_stores as *const () as usize;
                    (near(anchor).expect("CLZERO stub storage near the test image"), 0)
                })
                .0
        };
        // Against rdx (rm rax, reg rdx) and against rcx, the scratch register.
        for (cmp, by) in [([0x48, 0x39, 0xd0], 2), ([0x48, 0x39, 0xc8], 1), ([0x4c, 0x39, 0xc0], 8)] {
            // SAFETY: memory of our own, written and run inside it.
            unsafe {
                let page = near(stub_base).expect("CLZERO test loop within jump range");
                // A function of (start, end): the start in rax, the end in its register, the
                // loop at 16, and what rax became.
                let mut head = vec![0x48, 0x89, 0xc8]; // mov rax, rcx
                match by {
                    1 => head.extend([0x48, 0x89, 0xd1]),  // mov rcx, rdx
                    8 => head.extend([0x49, 0x89, 0xd0]),  // mov r8, rdx
                    _ => {}
                }
                head.resize(16, 0x90);
                let mut code = head;
                code.extend([0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40]);
                code.extend(cmp);
                code.extend([0x72, 0xf4]);
                code.push(0xc3);
                std::ptr::copy_nonoverlapping(code.as_ptr(), page as *mut u8, code.len());
                let mut context: CONTEXT = std::mem::zeroed();
                let mut block = vec![0xaau8; 64 * 8 + 64];
                let base = block.as_mut_ptr() as usize;
                let first = base.next_multiple_of(64);
                // The fault itself: emulated whole, and replaced.
                context.Rip = page as u64 + 16;
                context.Rax = first as u64;
                context.Rdx = first as u64 + 128;
                context.Rcx = context.Rdx;
                context.R8 = context.Rdx;
                assert_eq!(emulate(&mut context), CONTINUE_EXECUTION);
                assert_eq!((context.Rip, context.Rax), (page as u64 + 28, first as u64 + 128));
                if patches_clzero() {
                    assert_eq!(*((page + 16) as *const u8), 0xe9, "a jump");
                    block.fill(0xaa);
                    let run: extern "system" fn(usize, usize) -> usize = std::mem::transmute(page);
                    // From inside a line to inside the third: whole lines, one past the end.
                    assert_eq!(run(first + 8, first + 130), first + 8 + 128);
                    let at = first - base;
                    assert!(block[at..at + 128].iter().all(|&b| b == 0));
                    assert!(block[..at].iter().chain(&block[at + 128..]).all(|&b| b == 0xaa));
                    // An end at the start: one line.
                    block.fill(0xaa);
                    assert_eq!(run(first, first), first + 64);
                    assert!(block[at..at + 64].iter().all(|&b| b == 0) && block[at + 64] == 0xaa);
                }
                VirtualFree(page as *mut _, 0, MEM_RELEASE);
            }
        }
        assert!(clzero_stub(0x1000, 0x2000, [0x48, 0x39, 0xc0]).is_none(), "against rax itself");
        assert!(jump(0x1000, 0x9000_0000_0000).is_none(), "out of reach");
    }

    /// Memory kept from the title's stores until one faults (`PAGE_READONLY`, `PAGE_NOACCESS`,
    /// `PAGE_GUARD`, or `MEM_RESERVE`), with a scoped VEH handler at the front of the chain that
    /// opens or commits a page of this reservation on a write.
    /// `clzero` on such a line, alone or as the 12-byte loop, zeroes it through that fault;
    /// `MEM_FREE` memory is still declined (`return 0`).
    #[test]
    fn clzero_on_guarded_or_uncommitted_lines_faults_into_veh_and_declines_free() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::atomic::Ordering::{Acquire, Release};
        use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE};
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY,
            PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
        };

        const PAGES: usize = 5;
        static BASE: AtomicUsize = AtomicUsize::new(0);
        static OPENED: AtomicUsize = AtomicUsize::new(0);
        static GUARD_HITS: AtomicUsize = AtomicUsize::new(0);

        struct VehGuard {
            base: usize,
            handler: *mut std::ffi::c_void,
        }

        impl Drop for VehGuard {
            fn drop(&mut self) {
                BASE.store(0, Release);
                WRITABLE.with(|w| w.set(([(0, 0); 4], 0)));
                // SAFETY: unregisters the handler and releases the reservation owned by this test.
                unsafe {
                    if !self.handler.is_null() {
                        RemoveVectoredExceptionHandler(self.handler);
                    }
                    if self.base != 0 {
                        VirtualFree(self.base as *mut _, 0, MEM_RELEASE);
                    }
                }
            }
        }

        unsafe extern "system" fn open(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes the exception being dispatched.
            let r = unsafe { &*(*info).ExceptionRecord };
            let base = BASE.load(Acquire);
            if base == 0
                || r.NumberParameters < 2
                || r.ExceptionInformation[0] != 1
                || !(base..base + PAGES * 0x1000).contains(&r.ExceptionInformation[1])
            {
                return 0; // EXCEPTION_CONTINUE_SEARCH
            }
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION {
                let page = r.ExceptionInformation[1] & !0xfff;
                let mut old = 0;
                // SAFETY: a page of the test's own reservation; never panic across the FFI VEH boundary.
                let ok = unsafe {
                    !VirtualAlloc(page as _, 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null()
                        && VirtualProtect(page as _, 0x1000, PAGE_READWRITE, &mut old) != 0
                };
                if !ok {
                    return 0;
                }
                OPENED.fetch_add(1, Relaxed);
                return CONTINUE_EXECUTION;
            }
            if r.ExceptionCode == EXCEPTION_GUARD_PAGE {
                GUARD_HITS.fetch_add(1, Relaxed);
                return CONTINUE_EXECUTION;
            }
            0 // EXCEPTION_CONTINUE_SEARCH
        }

        OPENED.store(0, Relaxed);
        GUARD_HITS.store(0, Relaxed);

        // SAFETY: the test's own reservation; `guard` always unregisters the VEH and releases it,
        // even if an assertion panics.
        let mut guard = unsafe {
            let base = VirtualAlloc(std::ptr::null(), PAGES * 0x1000, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert_ne!(base, 0);
            VehGuard { base, handler: std::ptr::null_mut() }
        };
        let base = guard.base;

        // Page 0: committed, filled 0xAA, made PAGE_READONLY.
        // Page 1 & 2: MEM_RESERVE only.
        // Page 3: committed, filled 0xBB, made PAGE_NOACCESS.
        // Page 4: committed, filled 0xCC, made PAGE_READWRITE | PAGE_GUARD.
        // SAFETY: all pages lie within `guard.base..guard.base + PAGES * 0x1000`.
        unsafe {
            let mut old = 0;
            assert!(!VirtualAlloc(base as _, 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes(base as *mut u8, 0xAA, 0x1000);
            assert_ne!(VirtualProtect(base as _, 0x1000, PAGE_READONLY, &mut old), 0);

            let p3 = base + 3 * 0x1000;
            assert!(!VirtualAlloc(p3 as _, 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes(p3 as *mut u8, 0xBB, 0x1000);
            assert_ne!(VirtualProtect(p3 as _, 0x1000, PAGE_NOACCESS, &mut old), 0);

            let p4 = base + 4 * 0x1000;
            assert!(!VirtualAlloc(p4 as _, 0x1000, MEM_COMMIT, PAGE_READWRITE).is_null());
            std::ptr::write_bytes(p4 as *mut u8, 0xCC, 0x1000);
            assert_ne!(VirtualProtect(p4 as _, 0x1000, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);

            BASE.store(base, Release);
            guard.handler = AddVectoredExceptionHandler(1, Some(open));
            assert!(!guard.handler.is_null());
        }

        const INIT_FLAGS: u32 = 0x246;
        let run = |code: &[u8], rax: usize, r10: usize| {
            // SAFETY: plain data.
            let mut ctx: CONTEXT = unsafe { std::mem::zeroed() };
            ctx.Rip = code.as_ptr() as u64;
            ctx.Rax = rax as u64;
            ctx.R10 = r10 as u64;
            ctx.EFlags = INIT_FLAGS;
            // SAFETY: the context points at `code`; the target lines belong to this test.
            let r = unsafe { emulate(&mut ctx) };
            (r, ctx.Rip - code.as_ptr() as u64, ctx.Rax as usize, ctx.EFlags)
        };
        let page = |p: usize| {
            // SAFETY: opened (committed read-write) by the handler before this reads it.
            unsafe { std::slice::from_raw_parts((base + p * 0x1000) as *const u8, 0x1000) }
        };

        let lone = [0x0F, 0x01, 0xFC, 0x90];
        // Place the 12-byte loop at offset 12 of a 16-byte-aligned buffer so `(at & 15) + 5 > 16`:
        // `patch_clzero` leaves the bytes intact and we exercise the emulate re-entry path cleanly.
        #[repr(C, align(16))]
        struct AlignedCode([u8; 32]);
        let mut lp_buf = AlignedCode([0x90; 32]);
        lp_buf.0[12..24].copy_from_slice(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]);
        let lp = &lp_buf.0[12..25];

        // 1. Committed PAGE_READONLY line: zeroed through the fault, rest of page untouched.
        assert_eq!(run(&lone, base + 0x45, 0), (CONTINUE_EXECUTION, 3, base + 0x45, INIT_FLAGS));
        assert_eq!(OPENED.load(Relaxed), 1);
        assert!(page(0)[0x40..0x80].iter().all(|&b| b == 0));
        assert!(page(0)[..0x40].iter().chain(&page(0)[0x80..]).all(|&b| b == 0xAA));

        // 2. Reserved (uncommitted) line: committed by VEH and zeroed.
        assert_eq!(run(&lone, base + 0x1080, 0), (CONTINUE_EXECUTION, 3, base + 0x1080, INIT_FLAGS));
        assert_eq!(OPENED.load(Relaxed), 2);
        assert!(page(1)[0x80..0xC0].iter().all(|&b| b == 0));

        // 3. 12-byte loop starting on reserved memory: first line through fault (EFlags unchanged),
        // then the rest whole (EFlags updated from `cmp rax, end`).
        let (start, end) = (base + 0x2000, base + 0x2400);
        assert_eq!(
            run(lp, start, end),
            (CONTINUE_EXECUTION, 0, start + 64, INIT_FLAGS),
            "the next line, the same instruction"
        );
        assert_eq!(OPENED.load(Relaxed), 3);
        let expected_end_flags = INIT_FLAGS & !crate::crash::CMP_FLAGS | crate::crash::cmp_flags(end as u64, end as u64);
        assert_eq!(
            run(lp, start + 64, end),
            (CONTINUE_EXECUTION, 12, end, expected_end_flags),
            "the rest, then past the loop"
        );
        assert!(page(2)[..0x400].iter().all(|&b| b == 0));

        // 4. Single-line 12-byte loop (`end <= start`) on committed PAGE_NOACCESS: completes in one step.
        let p3_line = base + 0x3080;
        let expected_p3_flags = INIT_FLAGS & !crate::crash::CMP_FLAGS | crate::crash::cmp_flags((p3_line + 64) as u64, p3_line as u64);
        assert_eq!(run(lp, p3_line, p3_line), (CONTINUE_EXECUTION, 12, p3_line + 64, expected_p3_flags));
        assert_eq!(OPENED.load(Relaxed), 4);
        assert!(page(3)[0x80..0xC0].iter().all(|&b| b == 0));
        assert!(page(3)[..0x80].iter().chain(&page(3)[0xC0..]).all(|&b| b == 0xBB));

        // 5. Committed PAGE_READWRITE | PAGE_GUARD: declined by `writable`, zeroed via guard fault.
        let p4_line = base + 0x4040;
        assert!(!writable(p4_line, 64), "PAGE_GUARD must not be treated as plain writable");
        let before_tf = THROUGH_FAULT.load(Relaxed);
        assert_eq!(run(&lone, p4_line, 0), (CONTINUE_EXECUTION, 3, p4_line, INIT_FLAGS));
        assert_eq!(GUARD_HITS.load(Relaxed), 1);
        assert!(THROUGH_FAULT.load(Relaxed) > before_tf);
        assert!(page(4)[0x40..0x80].iter().all(|&b| b == 0));
        assert!(page(4)[..0x40].iter().chain(&page(4)[0x80..]).all(|&b| b == 0xCC));

        // Unregister VEH, clear the thread-local `WRITABLE` cache, and release the reservation.
        drop(guard);

        // 6. Permanently unallocatable null-guard MEM_FREE address: both lone `clzero` and
        // the 12-byte loop decline emulation (`return 0`) safely under parallel tests.
        let free_line = 0x40usize;
        // SAFETY: plain data filled in by VirtualQuery.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer of the right size.
        assert_ne!(unsafe { VirtualQuery(free_line as *const _, &mut mbi, size_of_val(&mbi)) }, 0);
        assert_eq!(mbi.State, MEM_FREE);
        assert_eq!(run(&lone, free_line, 0).0, 0);
        assert_eq!(run(lp, free_line, free_line + 0x400).0, 0);
    }
}
