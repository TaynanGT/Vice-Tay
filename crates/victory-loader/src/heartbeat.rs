//! The title's heartbeat flag, for the RAM throttle (`VICTORY_RAM_KEEP_ALIVE=1`;
//! off without it).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets): its
//! main loop calls +0x2214a90, `mov byte [rip+disp32], 1; ret`, which sets a
//! flag at +0x71b6330; a watchdog thread clears the flag and ends the title
//! when it finds it still clear about 30 s later (+0x23cf683, a write to
//! address 0). While the RAM throttle holds the title's submissions back
//! (`VICTORY_RAM_FRAME_WAIT_MS`), the main loop waits for them, and a world
//! load the throttle slows can go more than 30 s without a frame: on a 16 GB
//! laptop the watchdog ended 2 of 2 such loads.
//!
//! With the switch on, the throttle sets the flag as the title's own function
//! does, only while it is the one holding the title back (in its waits, which
//! `VICTORY_RAM_WAIT_MS` and `VICTORY_RAM_FRAME_WAIT_MS` bound). A title that
//! hangs anywhere else still meets its watchdog.
//!
//! What is checked before the flag is handed to the throttle ([`asked`]):
//! - the 8 bytes of the tick function lie inside the image, in one committed
//!   region the process may read; they are copied out (nothing else of the
//!   image is read) and must be the 1.0.2.1 ones, from which the flag's
//!   address is taken (another build is left alone, said so);
//! - the flag lies inside the image, in a committed region the process may
//!   write.
//!
//! The image is the prepared title DLL, loaded for the rest of the process
//! and never unloaded, so the flag's address stays valid. The title writes
//! the flag with plain byte stores and its watchdog reads and clears it; the
//! throttle's threads store to it atomically (a byte store, which x86 never
//! tears), so they do not race one another, and whichever store lands last,
//! the flag says "ticked" or is cleared by the watchdog as on the console.

/// The tick function: `mov byte [rip+disp32], 1` then `ret`, and where it is.
pub const TICK_AT: usize = 0x221_4a90;
/// Its bytes with the build's displacement (the flag at +0x71b6330).
pub const TICK: [u8; 8] = [0xc6, 0x05, 0x99, 0x18, 0xfa, 0x04, 0x01, 0xc3];

/// The flag's offset in the image from the tick function's bytes, if they are
/// the 1.0.2.1 ones.
pub fn flag_offset(code: &[u8; 8]) -> Option<usize> {
    if *code != TICK {
        return None;
    }
    let disp = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
    // RIP-relative: from the end of the 7-byte `mov`.
    (TICK_AT + 7).checked_add_signed(disp as isize)
}

/// Whether `[at, at + len)` lies in one committed region of this process's
/// memory that may be read (and written, with `write`): not a guard page, not
/// `PAGE_NOACCESS`.
#[cfg(windows)]
pub fn region_allows(at: usize, len: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
        VirtualQuery,
    };
    let Some(end) = at.checked_add(len) else {
        return false;
    };
    // SAFETY: a query about an address, into a structure of its own size.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let asked = size_of::<MEMORY_BASIC_INFORMATION>();
    if unsafe { VirtualQuery(at as *const _, &mut m, asked) } != asked {
        return false;
    }
    let region_end = m.BaseAddress as usize + m.RegionSize;
    let protect = m.Protect & !PAGE_GUARD;
    let readable = [
        PAGE_READONLY,
        PAGE_READWRITE,
        PAGE_WRITECOPY,
        PAGE_EXECUTE_READ,
        PAGE_EXECUTE_READWRITE,
        PAGE_EXECUTE_WRITECOPY,
    ];
    let writable = [
        PAGE_READWRITE,
        PAGE_WRITECOPY,
        PAGE_EXECUTE_READWRITE,
        PAGE_EXECUTE_WRITECOPY,
    ];
    m.State == MEM_COMMIT
        && m.Protect & PAGE_GUARD == 0
        && end <= region_end
        && if write {
            writable.contains(&protect)
        } else {
            readable.contains(&protect)
        }
}

/// The flag's address in the title image mapped at `base` (`size` bytes, its
/// `SizeOfImage`) if `VICTORY_RAM_KEEP_ALIVE=1` asks for it and every check
/// above holds; says what it found.
///
/// # Safety
/// `base` is the title image, `size` bytes, loaded for the rest of the process.
#[cfg(windows)]
pub unsafe fn asked(base: usize, size: usize) -> Option<*const std::sync::atomic::AtomicU8> {
    if std::env::var_os("VICTORY_RAM_KEEP_ALIVE").is_none_or(|v| v != "1") {
        return None;
    }
    let leave = |why: &str| {
        tracing::warn!(
            why,
            "VICTORY_RAM_KEEP_ALIVE: the title's heartbeat is left alone"
        );
        None
    };
    if TICK_AT.checked_add(TICK.len()).is_none_or(|end| end > size) {
        return leave("the tick function would lie past the image");
    }
    let tick = base + TICK_AT;
    if !region_allows(tick, TICK.len(), false) {
        return leave("the tick function's bytes are not readable");
    }
    // SAFETY: 8 bytes inside the image, in readable committed memory (checked).
    let code: [u8; 8] = unsafe { std::ptr::read_volatile(tick as *const [u8; 8]) };
    let Some(at) = flag_offset(&code) else {
        return leave("not the build whose heartbeat is known (+0x2214a90 differs)");
    };
    if at >= size || !region_allows(base + at, 1, true) {
        return leave("the flag is not a writable byte of the image");
    }
    tracing::info!(
        flag = format_args!("+{at:#x}"),
        "the RAM throttle sets the title's heartbeat flag while it holds the title back (VICTORY_RAM_KEEP_ALIVE)"
    );
    Some((base + at) as *const std::sync::atomic::AtomicU8)
}

#[cfg(test)]
mod tests {
    use super::{TICK, TICK_AT, flag_offset};

    #[test]
    fn the_heartbeat_flag_is_found_only_behind_the_known_tick_function() {
        assert_eq!(flag_offset(&TICK), Some(0x71b_6330));
        assert_eq!(flag_offset(&[0; 8]), None);
        // Another displacement is another build.
        let mut other = TICK;
        other[2] ^= 1;
        assert_eq!(flag_offset(&other), None);
        assert!(TICK_AT + 7 < 0x71b_6330);
    }

    #[cfg(windows)]
    #[test]
    fn a_region_is_checked_for_commit_access_and_bounds() {
        use super::region_allows;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualProtect,
        };
        const PAGE: usize = 4096;
        // SAFETY: memory of the test's own, freed at the end.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS) as usize;
            assert!(base != 0);
            assert!(
                !VirtualAlloc(base as *const _, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null()
            );
            let mut old = 0;
            assert!(VirtualProtect((base + PAGE) as *const _, PAGE, PAGE_READONLY, &mut old) != 0);
            assert!(region_allows(base, 8, true));
            assert!(region_allows(base + PAGE, 8, false));
            assert!(
                !region_allows(base + PAGE, 1, true),
                "read-only is not writable"
            );
            assert!(
                !region_allows(base + PAGE - 4, 8, false),
                "a range across two regions"
            );
            assert!(
                !region_allows(base + 2 * PAGE, 1, false),
                "reserved, not committed"
            );
            assert!(
                !region_allows(usize::MAX - 2, 8, false),
                "an end past the address space"
            );
            VirtualFree(base as *mut _, 0, MEM_RELEASE);
        }
    }
}
