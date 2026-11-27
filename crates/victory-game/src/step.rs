//! The title's longest simulation step.
//!
//! What the title does (the known Series master build, image offsets):
//!
//! - Each frame its timer update (+0x22264d0) takes the real time since the
//!   last frame from `QueryPerformanceCounter` (passed through to the host:
//!   real time here) and hands it to the step function +0x2226040 for each
//!   of its timers (game time is the one at +0xabbef40). That function makes
//!   the step `max(min(dt, 1/15 s), 1/30 s)` (times a scale): the 1/15 s is
//!   the float at +0x6159ec0, loaded at +0x2226045.
//! - The paused branch of +0x22264d0 clamps one timer (+0xa958358) the same
//!   way (load at +0x2226833).
//! - +0x8dec0 takes the game timer's step (+0xabbef4c) and clamps it to
//!   1/15 s once more (load at +0x8df76) for the world update after it.
//!
//! So a frame longer than 66.7 ms moves the world only 66.7 ms: at 8 frames
//! a second (125 ms) it runs at about half speed, at 4 at a quarter. The
//! constant has 64 readers (most multiply by it), so it stays as it is;
//! the three loads are pointed at a float of ours, written into the `int3`
//! padding after +0x2226040's `ret` (+0x2226125). Each site's whole
//! instruction is compared first.

/// The switch's range in milliseconds: from the title's own 66.7 up to
/// 333 (3 frames a second); longer steps mean more of the title's 1/15 s
/// physics substeps in one frame.
pub const MIN_MS: f32 = 66.7;
pub const MAX_MS: f32 = 333.0;

/// `VICTORY_MAX_STEP_MS=<ms>` parsed: `None` when unset, `Err` (the text)
/// when malformed or out of range.
pub fn parse_ms(spec: &str) -> Result<f32, String> {
    match spec.trim().parse::<f32>() {
        Ok(ms) if ms.is_finite() && (MIN_MS..=MAX_MS).contains(&ms) => Ok(ms),
        _ => Err(format!(
            "VICTORY_MAX_STEP_MS={spec}: expected milliseconds from {MIN_MS} to {MAX_MS} (e.g. 200)"
        )),
    }
}

/// Where our float goes: `int3` padding after +0x2226040's `ret`.
pub const CONSTANT_AT: u32 = 0x222_6128;
// The replacement float must fit after the return and before the next function.
const _: () = assert!(CONSTANT_AT > 0x222_6125 && CONSTANT_AT + 4 <= 0x222_6130);
/// The title's 1/15 s.
pub const TITLE_CONSTANT_AT: u32 = 0x615_9ec0;

/// A load of the 1/15 s: `vmovss xmmN, [rip+disp32]`, 8 bytes, the
/// displacement at +4.
pub struct Load {
    pub rva: u32,
    pub bytes: [u8; 8],
}

pub const LOADS: [Load; 3] = [
    // +0x2226040, the step of every timer: vmovss xmm3, [+0x6159ec0]
    Load { rva: 0x222_6045, bytes: [0xc5, 0xfa, 0x10, 0x1d, 0x73, 0x3e, 0xf3, 0x03] },
    // +0x22264d0, the paused branch's clamp: vmovss xmm0, [+0x6159ec0]
    Load { rva: 0x222_6833, bytes: [0xc5, 0xfa, 0x10, 0x05, 0x85, 0x36, 0xf3, 0x03] },
    // +0x8dec0, the world update's own clamp: vmovss xmm0, [+0x6159ec0]
    Load { rva: 0x8_df76, bytes: [0xc5, 0xfa, 0x10, 0x05, 0x42, 0xbf, 0x0c, 0x06] },
];

/// The address a load's displacement leads to.
pub fn load_target(rva: u32, bytes: &[u8; 8]) -> u32 {
    let disp = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    (rva as i64 + 8 + disp as i64) as u32
}

/// The patches for `ms`: (rva, bytes expected, bytes written), the float
/// first.
pub fn patches(ms: f32) -> Vec<(u32, Vec<u8>, Vec<u8>)> {
    let mut out = vec![(CONSTANT_AT, vec![0xcc; 4], (ms / 1000.0).to_le_bytes().to_vec())];
    for load in &LOADS {
        let disp = (CONSTANT_AT as i64 - (load.rva as i64 + 8)) as i32;
        let mut new = load.bytes.to_vec();
        new[4..].copy_from_slice(&disp.to_le_bytes());
        out.push((load.rva, load.bytes.to_vec(), new));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loads_read_the_titles_fifteenth_and_then_ours() {
        for load in &LOADS {
            assert_eq!(load_target(load.rva, &load.bytes), TITLE_CONSTANT_AT, "{:#x}", load.rva);
        }
        let p = patches(200.0);
        assert_eq!(p[0].0, CONSTANT_AT);
        assert_eq!(f32::from_le_bytes(p[0].2[..].try_into().unwrap()), 0.2);
        for (rva, old, new) in &p[1..] {
            assert_eq!(old.len(), new.len());
            assert_eq!(old[..4], new[..4], "same instruction, other displacement");
            let new: [u8; 8] = new[..].try_into().unwrap();
            assert_eq!(load_target(*rva, &new), CONSTANT_AT);
        }
    }

    #[test]
    fn the_switch_range() {
        assert_eq!(parse_ms(" 200 "), Ok(200.0));
        assert_eq!(parse_ms("66.7"), Ok(66.7));
        assert!(parse_ms("50").is_err());
        assert!(parse_ms("1000").is_err());
        assert!(parse_ms("fast").is_err());
        assert!(parse_ms("NaN").is_err());
    }
}
