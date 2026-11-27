//! The title's heartbeat flag, for the RAM throttle (`VICTORY_RAM_KEEP_ALIVE=1`;
//! off without it).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets): its
//! main loop calls +0x2214a90, `mov byte [rip+disp32], 1; ret`, which sets a
//! flag at +0x71b6330; a watchdog thread clears the flag and ends the title
//! when it finds it still clear about 30 s later (+0x23cf683). While the RAM
//! throttle holds the title's submissions back (`VICTORY_RAM_FRAME_WAIT_MS`),
//! the main loop waits for them, and a world load the throttle slows can go
//! more than 30 s without a frame.
//!
//! With the switch on, and only if those 8 bytes are there (another build of
//! the title is left alone, said so), the throttle sets the flag as the title's
//! own function does, **only while it is the one holding the title back** (in
//! its waits, which `VICTORY_RAM_WAIT_MS` and `VICTORY_RAM_FRAME_WAIT_MS`
//! bound). A title that hangs anywhere else still meets its watchdog.

/// The tick function: `mov byte [rip+disp32], 1` then `ret`, and where it is.
pub const TICK_AT: usize = 0x221_4a90;
/// Its bytes with the build's displacement (the flag at +0x71b6330).
pub const TICK: [u8; 8] = [0xc6, 0x05, 0x99, 0x18, 0xfa, 0x04, 0x01, 0xc3];

/// The flag's offset in the image, from the tick function's bytes there, if
/// they are the 1.0.2.1 ones.
pub fn flag_offset(image: &[u8]) -> Option<usize> {
    let code = image.get(TICK_AT..TICK_AT + TICK.len())?;
    if code != TICK {
        return None;
    }
    let disp = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
    // RIP-relative: from the end of the 7-byte `mov`.
    (TICK_AT + 7).checked_add_signed(disp as isize)
}

/// The flag's address in the image mapped at `base` (`size` bytes) if
/// `VICTORY_RAM_KEEP_ALIVE=1` asks for it and the build is the one above;
/// says what it found.
///
/// # Safety
/// `base` is a mapped image of `size` bytes that stays mapped.
pub unsafe fn asked(base: usize, size: usize) -> Option<usize> {
    if std::env::var_os("VICTORY_RAM_KEEP_ALIVE").is_none_or(|v| v != "1") {
        return None;
    }
    // SAFETY: the caller's.
    let image = unsafe { std::slice::from_raw_parts(base as *const u8, size) };
    match flag_offset(image) {
        Some(at) if at < size => {
            tracing::info!(
                flag = format_args!("+{at:#x}"),
                "the RAM throttle sets the title's heartbeat flag while it holds the title back (VICTORY_RAM_KEEP_ALIVE)"
            );
            Some(base + at)
        }
        _ => {
            tracing::warn!(
                "VICTORY_RAM_KEEP_ALIVE: not the build whose heartbeat is known (+0x2214a90 differs); the watchdog stays as it is"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TICK, TICK_AT, flag_offset};

    #[test]
    fn the_heartbeat_flag_is_found_only_behind_the_known_tick_function() {
        let mut image = vec![0u8; TICK_AT + 16];
        assert_eq!(flag_offset(&image), None);
        image[TICK_AT..TICK_AT + TICK.len()].copy_from_slice(&TICK);
        assert_eq!(flag_offset(&image), Some(0x71b_6330));
        // Another displacement is another build.
        image[TICK_AT + 2] ^= 1;
        assert_eq!(flag_offset(&image), None);
        assert_eq!(flag_offset(&image[..TICK_AT + 4]), None);
    }
}
