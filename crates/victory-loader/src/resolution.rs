//! The title's output size and render scale (`VICTORY_OUTPUT`,
//! `VICTORY_RENDER_SCALE`, `VICTORY_FFX_SCALE`, `VICTORY_UPSCALER`; each off
//! without it): places
//! in the title's code changed at load, each compared byte for byte first
//! and left alone, said so, if it differs (another build of the title).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets):
//!
//! - **Output size.** Its settings (`settings.xml` in its storage,
//!   `video/resolutionFullscreen`, 3840x2160 as a Series X wrote it; read
//!   into the live settings at +0x204/+0x208, the defaults' 1920x1080 at
//!   +0x5b37a20 overwritten) are applied by +0x22434a0: from +0x224352b it
//!   hands them to +0x2478130, the one writer of the screen size
//!   (+0xa9438cc/+0xa9438d0), and when the height changes calls +0xd8c3a0.
//!   The back buffers (+0x247ad10, by +0x501f4a0 and +0xcc2780) take that
//!   size through the copy +0x5eb24f0 makes (+0x75e6fe0). The display mode
//!   list it matches against is its own (+0x5a38910: 1280x720, 1664x936,
//!   1920x1080, 2560x1440, 3840x2160 at 60 Hz, table +0x615c0c0), not the
//!   DXGI output's, which is why `VICTORY_DISPLAY_MODE` changes nothing.
//!   Here that block becomes: write the size asked for into the settings
//!   (+0x204/+0x208), then the title's own else-path (screen size set,
//!   +0xd8c3a0 if the height changed). Everything that reads the settings
//!   after it (the device parameters +0x1e0fb30, the render size +0x7b4e10)
//!   sees the same size. The back buffers' pool (64 MB on Series X) is
//!   checked to be used up but for 2 MB (+0x247af25, an `int3`); for another
//!   size that check is jumped over.
//! - **Render scale.** The upscaler the title picks (+0x7e93f0: 6, its own,
//!   except on Series S with a flag, 7, AMD FidelityFX) gives the render
//!   size from the output size (+0x7e92f0 → the upscaler's +0x50 method):
//!   its own (+0x7e9440) multiplies by a scale at +0x90 of it, set to 2/3 at
//!   +0x28f1e78 (so 2560x1440 for 4K, 1280x720 for 1080p);
//!   `VICTORY_RENDER_SCALE` replaces that constant (Series X; run res-4 at
//!   1.0). FidelityFX's (+0x28ef680, Series S) multiplies by 1.0 on Series X,
//!   2/3 on Series S; `VICTORY_FFX_SCALE` points its two loads at a constant
//!   in its padding (as pull request #53 did). That one is not safe: at 0.5
//!   on Series S (run res-6) FidelityFX's jitter phase count came back 0 and
//!   the title divided by it (+0x28f03e8) 49 s in.
//! - **Upscaler.** `VICTORY_UPSCALER=<kind>` makes the picker and its
//!   per-frame copy (+0x5400190) return `kind` (pull request #53): 6 the
//!   title's own, 7 FidelityFX. On a Series S of the known build both return
//!   6 without it (`VICTORY_TITLE_PATCHES=0` leaves them; `=1` on any console).

use crate::frame_rate::Patched;

/// Sizes the title has a display mode for (its own list, +0x615c0c0).
pub const MODES: [(u32, u32); 5] = [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)];

/// `VICTORY_OUTPUT=<w>x<h>`: one of [`MODES`]. `None` if unset or not one
/// (said why).
pub fn asked_output() -> Option<(u32, u32)> {
    let spec = std::env::var("VICTORY_OUTPUT").ok()?;
    let size = parse_output(&spec);
    if size.is_none() {
        eprintln!("VIctory: VICTORY_OUTPUT={spec}: expected one of the title's modes: 1920x1080, 2560x1440, 1280x720, 1664x936, 3840x2160");
    }
    size
}

pub fn parse_output(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.trim().split_once(['x', 'X'])?;
    let size = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

/// `VICTORY_RENDER_SCALE=<s>`, 0.5 to 1.0 (0.5: half the output's width and
/// height): the title's own upscaler's scale (Series X; 2/3 without it).
/// Tested at 1.0 (run res-4); below 2/3 the title's largest render size
/// (+0x94, max(2/3, scale)) stays at 2/3. `None` if unset or out of range
/// (said why).
pub fn asked_render_scale() -> Option<f32> {
    let spec = std::env::var("VICTORY_RENDER_SCALE").ok()?;
    let scale = parse_render_scale(&spec);
    if scale.is_none() {
        eprintln!("VIctory: VICTORY_RENDER_SCALE={spec}: expected 0.5 to 1.0");
    }
    scale
}

pub fn parse_render_scale(spec: &str) -> Option<f32> {
    spec.trim().parse::<f32>().ok().filter(|s| (0.5..=1.0).contains(s))
}

/// `VICTORY_FFX_SCALE=<s>`, 0.5 to 1.0: FidelityFX's render scale (Series
/// S). Crashed at 0.5 (run res-6); see the module's notes.
pub fn asked_ffx_scale() -> Option<f32> {
    let spec = std::env::var("VICTORY_FFX_SCALE").ok()?;
    let scale = parse_render_scale(&spec);
    if scale.is_none() {
        eprintln!("VIctory: VICTORY_FFX_SCALE={spec}: expected 0.5 to 1.0");
    }
    scale
}

/// `VICTORY_UPSCALER=<kind>`: 6 (the title's own) or 7 (FidelityFX).
pub fn asked_upscaler() -> Option<u32> {
    let spec = std::env::var("VICTORY_UPSCALER").ok()?;
    let kind = spec.trim().parse::<u32>().ok().filter(|k| matches!(k, 6 | 7));
    if kind.is_none() {
        eprintln!("VIctory: VICTORY_UPSCALER={spec}: expected 6 (the title's own) or 7 (FidelityFX)");
    }
    kind
}

/// A place to change: its offset in the image and the bytes there
/// (compared first; the replacement is at most as long).
pub struct Site {
    pub at: usize,
    pub what: &'static str,
    pub bytes: &'static [u8],
}

/// The screen size setter, its screen height, and what +0x22434a0 calls when
/// the height changed.
pub const SET_SCREEN_SIZE: usize = 0x2478130;
pub const SCREEN_HEIGHT: usize = 0xa9438d0;
pub const HEIGHT_CHANGED: usize = 0xd8c3a0;

/// +0x22434a0 from +0x224352b to +0x2243568: the settings' size handed to
/// the screen size.
pub const OUTPUT: Site = Site {
    at: 0x224352b,
    what: "settings' output size applied",
    bytes: &[
        0x8b, 0x0d, 0x9b, 0x03, 0x70, 0x08, // mov ecx, [+0xa9438cc]
        0x8b, 0x86, 0x04, 0x02, 0x00, 0x00, // mov eax, [rsi+204h]
        0x39, 0xc1, 0x75, 0x0d, // cmp ecx, eax ; jne
        0x8b, 0x96, 0x08, 0x02, 0x00, 0x00, // mov edx, [rsi+208h]
        0xe8, 0xea, 0x4b, 0x23, 0x00, 0xeb, 0x20, // call +0x2478130 ; jmp +0x2243568
        0x8b, 0xbe, 0x08, 0x02, 0x00, 0x00, // mov edi, [rsi+208h]
        0x8b, 0x2d, 0x7c, 0x03, 0x70, 0x08, // mov ebp, [+0xa9438d0]
        0x89, 0xc1, 0x89, 0xfa, // mov ecx, eax ; mov edx, edi
        0xe8, 0xd3, 0x4b, 0x23, 0x00, // call +0x2478130
        0x39, 0xfd, 0x74, 0x07, 0xb1, 0x01, // cmp ebp, edi ; je ; mov cl, 1
        0xe8, 0x38, 0x8e, 0xb4, 0xfe, // call +0xd8c3a0
    ],
};

/// +0x247ad10, after the back buffers are made in their pool (64 MB on
/// Series X, 82 MB on Series S, from +0x247ad45): `int3` unless less than
/// 2 MB of the pool is left over. Sized for 3840x2160; at 1920x1080 49 MB
/// are left (run res-1 stopped here). With another output size the check
/// becomes a jump over the `int3`; the pool stays as large.
pub const POOL_CHECK: Site = Site {
    at: 0x247af25,
    what: "back buffer pool's left-over check",
    bytes: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc], // cmp rax, 1FFFFFh ; jbe +1 ; int3
};

/// What goes at [`POOL_CHECK`]: `jmp +1` in place of `jbe +1`.
pub fn pool_check_code() -> Vec<u8> {
    let mut c = POOL_CHECK.bytes[..8].to_vec();
    c[6] = 0xeb;
    c
}

/// The title's own upscaler, its method +0x28f1e70: `mov dword [rcx+90h],
/// 2/3` (the scale +0x7e9440 renders at).
pub const OWN_SCALE: Site = Site {
    at: 0x28f1e78,
    what: "the title's own upscaler's render scale",
    bytes: &[0xc7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x2a, 0x3f],
};

/// FidelityFX's render size method, +0x28ef680, and its padding: the scale
/// is 1.0 (+0x5fc6ed8) or, on Series S, 2/3 (+0x615d95c), loaded at +0x28
/// and +0x32.
pub const FFX_SCALE: Site = Site {
    at: 0x28ef680,
    what: "FidelityFX's render scale",
    bytes: &[
        0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x28, 0x44, 0x89, 0xcb, 0x4c, 0x89, 0xc6, 0x48, 0x89, 0xd7, 0x49, 0x89,
        0xce, 0xe8, 0xa6, 0x5a, 0x78, 0xfd, 0x0f, 0xb6, 0xc0, 0x41, 0x89, 0x86, 0x90, 0x00, 0x00, 0x00, 0x84, 0xc0, 0x75, 0x0a,
        0xc5, 0xfa, 0x10, 0x05, 0x28, 0x78, 0x6d, 0x03, 0xeb, 0x08, 0xc5, 0xfa, 0x10, 0x05, 0xa2, 0xe2, 0x86, 0x03, 0xc5, 0xf2,
        0x2a, 0x54, 0x24, 0x70, 0xc5, 0xf2, 0x2a, 0xcb, 0xc5, 0xfa, 0x59, 0xc9, 0xc5, 0xfa, 0x2c, 0xc1, 0x89, 0x07, 0xc5, 0xfa,
        0x59, 0xc2, 0xc5, 0xfa, 0x2c, 0xc0, 0x89, 0x06, 0x48, 0x83, 0xc4, 0x28, 0x5b, 0x5f, 0x5e, 0x41, 0x5e, 0xc3, 0xcc, 0xcc,
        0xcc, 0xcc, 0xcc, 0xcc,
    ],
};

/// The upscaler picker, +0x7e93f0, to its first `ret`.
pub const PICKER: Site = Site {
    at: 0x7e93f0,
    what: "upscaler picker",
    bytes: &[
        0x48, 0x83, 0xec, 0x28, 0x80, 0x3d, 0x6d, 0xf0, 0x3a, 0x07, 0x00, 0x75, 0x09, 0x83, 0x3d, 0x68, 0xf0, 0x3a, 0x07, 0x00,
        0x74, 0x28, 0xe8, 0x35, 0xbd, 0x88, 0xff, 0x8a, 0x0d, 0x5b, 0xf0, 0x3a, 0x07, 0x80, 0xe1, 0x40, 0x80, 0xf9, 0x01, 0xb9,
        0x07, 0x00, 0x00, 0x00, 0x83, 0xd9, 0x00, 0x84, 0xc0, 0xb8, 0x06, 0x00, 0x00, 0x00, 0x0f, 0x45, 0xc1, 0x48, 0x83, 0xc4,
        0x28, 0xc3,
    ],
};

/// The picker's per-frame copy, +0x5400190 (its first 32 bytes): the
/// title asks it again every frame, so with only [`PICKER`] changed a
/// Series S still switches to FidelityFX. (victory-fork #798, by KyleS98.)
pub const PICKER_FRAME: Site = Site {
    at: 0x540_0190,
    what: "upscaler picker (the per-frame copy)",
    bytes: &[
        0x48, 0x83, 0xec, 0x28, 0x48, 0x8b, 0x05, 0xb5, 0xe2, 0x1d, 0x02, 0x48, 0x8d, 0x0d, 0x5e, 0x00, 0x00, 0x00, 0x48, 0x8b,
        0x80, 0xa8, 0x00, 0x00, 0x00, 0x48, 0x39, 0xc8, 0x75, 0x3c, 0xe8, 0x4d,
    ],
};

/// The upscaler both pickers are made to return: `VICTORY_UPSCALER` if set;
/// else, on a Series S of the known build (`series_s_default`), 6, the
/// title's own: FidelityFX's (7) output never reaches the screen here, and
/// its DLL is refused by the host policy, so the world stays black.
/// `VICTORY_TITLE_PATCHES=0` leaves the title's choice alone.
pub fn upscaler_kind(asked: Option<u32>, series_s_default: bool) -> Option<u32> {
    asked.or(series_s_default.then_some(6))
}

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at [`OUTPUT`] for `width` x `height` (the same length).
pub fn output_code(width: u32, height: u32) -> Vec<u8> {
    let at = OUTPUT.at;
    let mut c = Vec::with_capacity(OUTPUT.bytes.len());
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [screen height]
    c.extend(rel(SCREEN_HEIGHT, at + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the setter
    c.extend(rel(SET_SCREEN_SIZE, at + c.len() + 4));
    let end = OUTPUT.bytes.len();
    c.extend([0x39, 0xfd, 0x74, (end - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call
    c.extend(rel(HEIGHT_CHANGED, at + c.len() + 4));
    c.extend([0xeb, (end - (c.len() + 2)) as u8]); // jmp end
    c.resize(end, 0xcc);
    c
}

/// What goes at [`OWN_SCALE`] for `scale`.
pub fn own_scale_code(scale: f32) -> Vec<u8> {
    let mut c = OWN_SCALE.bytes[..6].to_vec();
    c.extend(scale.to_le_bytes());
    c
}

/// What goes at [`FFX_SCALE`] for `scale`: both loads read `scale`, written
/// into the padding at +0x64.
pub fn ffx_scale_code(scale: f32) -> Vec<u8> {
    const LOADS: [usize; 2] = [0x28, 0x32];
    const SLOT: usize = 0x64;
    let mut c = FFX_SCALE.bytes.to_vec();
    c[SLOT..SLOT + 4].copy_from_slice(&scale.to_le_bytes());
    for l in LOADS {
        c[l + 4..l + 8].copy_from_slice(&((SLOT - (l + 8)) as i32).to_le_bytes());
    }
    c
}

/// `mov eax, kind ; ret`.
pub fn picker_code(kind: u32) -> Vec<u8> {
    let mut c = vec![0xb8];
    c.extend(kind.to_le_bytes());
    c.push(0xc3);
    c
}

/// Compares `site`'s bytes in the image of `size` bytes at `base`.
///
/// # Safety
/// The image is mapped.
pub unsafe fn matches(base: usize, size: usize, site: &Site) -> Patched {
    if site.at.checked_add(site.bytes.len()).is_none_or(|end| end > size) {
        return Patched::Failed("outside the image");
    }
    // SAFETY: inside the mapped image (checked above).
    let there = unsafe { std::slice::from_raw_parts((base + site.at) as *const u8, site.bytes.len()) };
    if there == site.bytes { Patched::Done } else { Patched::Differs }
}

/// Writes `code` over `site` if its bytes are the expected ones.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn write(base: usize, size: usize, site: &Site, code: &[u8]) -> Patched {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if code.len() > site.bytes.len() {
        return Patched::Failed("replacement longer than the bytes compared");
    }
    // SAFETY: the caller's promise.
    let found = unsafe { matches(base, size, site) };
    if found != Patched::Done {
        return found;
    }
    let at = base + site.at;
    let mut old = 0;
    // SAFETY: inside the image (checked); made writable for this and put
    // back; nothing runs it yet.
    unsafe {
        if VirtualProtect(at as *const _, code.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Patched::Failed("Windows would not make it writable");
        }
        std::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
        VirtualProtect(at as *const _, code.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as *const _, code.len());
    }
    Patched::Done
}

/// Writes every (offset, old, new) as one group. Every page is prepared
/// and every signature compared before writing any replacement bytes.
/// Err names the first site that stopped the group and why.
///
/// # Safety
/// The image remains mapped and no thread runs its code during the patch.
#[cfg(windows)]
pub unsafe fn write_all(base: usize, size: usize, sites: &[(usize, &[u8], &[u8])]) -> Result<(), (usize, Patched)> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS,
        VirtualProtect, VirtualQuery,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut protections = Vec::with_capacity(sites.len());
    // SAFETY: the caller keeps the range mapped and its code inactive.
    let result = unsafe {
        write_all_with(base, size, sites, |mut at, len| {
            let end = at + len;
            while at < end {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if VirtualQuery(at as *const _, &mut info, std::mem::size_of_val(&info)) == 0
                    || info.State != MEM_COMMIT
                    || info.Protect & (PAGE_GUARD | PAGE_NOACCESS) != 0
                {
                    return Err("Windows did not report readable committed memory");
                }
                // VirtualProtect only reports the old protection of its first
                // page. Limit each call to a region of uniform protection.
                let Some(region_end) = (info.BaseAddress as usize)
                    .checked_add(info.RegionSize)
                    .filter(|&region_end| region_end > at)
                else {
                    return Err("Windows reported invalid region bounds");
                };
                let next = end.min(region_end);
                let mut old = 0;
                if VirtualProtect(at as *const _, next - at, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    return Err("Windows would not make it writable");
                }
                protections.push((at, next - at, old));
                at = next;
            }
            Ok(())
        })
    };
    // Reverse order also restores the right permissions when two sites touch
    // one page and the later one temporarily inherited its writable state.
    for (at, len, protection) in protections.into_iter().rev() {
        let mut old = 0;
        // SAFETY: these are the still-mapped regions recorded above.
        if unsafe { VirtualProtect(at as *const _, len, protection, &mut old) } == 0 {
            tracing::error!(at, len, "could not restore image page protection");
        }
    }
    if result.is_ok() {
        for &(at, _, new) in sites {
            if !new.is_empty() {
                // SAFETY: the written bytes are still in the mapped image.
                if unsafe { FlushInstructionCache(GetCurrentProcess(), (base + at) as *const _, new.len()) } == 0 {
                    tracing::error!(at, "could not flush the image instruction cache");
                }
            }
        }
    }
    result
}

/// Prepares a group of byte ranges, verifies all signatures, then writes.
///
/// # Safety
/// A successful prepare call leaves the range readable and writable until
/// the function returns. Nothing runs or changes the image during the patch.
#[cfg(any(windows, test))]
pub(crate) unsafe fn write_all_with(
    base: usize,
    size: usize,
    sites: &[(usize, &[u8], &[u8])],
    mut prepare: impl FnMut(usize, usize) -> Result<(), &'static str>,
) -> Result<(), (usize, Patched)> {
    for (i, &(at, old, new)) in sites.iter().enumerate() {
        if new.len() > old.len()
            || at.checked_add(old.len()).is_none_or(|end| end > size || base.checked_add(end).is_none())
        {
            return Err((i, Patched::Failed("outside the image or longer than the bytes compared")));
        }
        if !old.is_empty() {
            prepare(base + at, old.len()).map_err(|why| (i, Patched::Failed(why)))?;
        }
    }
    for (i, &(at, old, _)) in sites.iter().enumerate() {
        if !old.is_empty() {
            // SAFETY: prepare validated this readable range.
            let there = unsafe { std::slice::from_raw_parts((base + at) as *const u8, old.len()) };
            if there != old {
                return Err((i, Patched::Differs));
            }
        }
    }
    for &(at, _, new) in sites {
        if !new.is_empty() {
            // SAFETY: prepare made the range writable and new is a separate buffer.
            unsafe { std::ptr::copy_nonoverlapping(new.as_ptr(), (base + at) as *mut u8, new.len()) };
        }
    }
    Ok(())
}

/// Applies what the switches ask for; a line each in the log and on
/// standard output.
///
/// # Safety
/// The image of `size` bytes at `base` stays mapped and no thread runs its
/// code yet.
#[cfg(windows)]
pub unsafe fn apply_asked(base: usize, size: usize, series_s_default: bool) {
    let mut asked: Vec<(&Site, Vec<u8>, String)> = Vec::new();
    if let Some((w, h)) = asked_output() {
        asked.push((&OUTPUT, output_code(w, h), format!("{w}x{h}")));
        if (w, h) != (3840, 2160) {
            asked.push((&POOL_CHECK, pool_check_code(), format!("{w}x{h}")));
        }
    }
    if let Some(s) = asked_render_scale() {
        asked.push((&OWN_SCALE, own_scale_code(s), format!("{s}")));
    }
    if let Some(s) = asked_ffx_scale() {
        asked.push((&FFX_SCALE, ffx_scale_code(s), format!("{s}")));
    }
    // The two upscaler choices as a pair: both sites are compared first, and
    // neither is written unless both are the expected bytes (as in
    // fxceless/victory-fork #798), so the title never runs with one choice
    // changed and the other not.
    if let Some(kind) = upscaler_kind(asked_upscaler(), series_s_default) {
        let code = picker_code(kind);
        let pair = [(PICKER.at, PICKER.bytes, &code[..]), (PICKER_FRAME.at, PICKER_FRAME.bytes, &code[..])];
        // SAFETY: the caller's promise.
        match unsafe { write_all(base, size, &pair) } {
            Ok(()) => {
                for site in [&PICKER, &PICKER_FRAME] {
                    let offset = format!("{:#x}", site.at);
                    tracing::info!(site = %offset, what = site.what, value = %format!("returns {kind}"), "resolution: patched");
                    println!("resolution: {} (returns {kind}) at title+{offset}: Done", site.what);
                }
            }
            Err((i, why)) => {
                let site = [&PICKER, &PICKER_FRAME][i];
                let offset = format!("{:#x}", site.at);
                tracing::warn!(site = %offset, what = site.what, ?why, "resolution: upscaler choice not patched at either site (another build?)");
                println!("resolution: upscaler choice (returns {kind}) not patched: title+{offset} {why:?}; neither site written");
            }
        }
    }
    for (site, code, value) in asked {
        // SAFETY: the caller's promise.
        let done = unsafe { write(base, size, site, &code) };
        let offset = format!("{:#x}", site.at);
        match done {
            Patched::Done => tracing::info!(site = %offset, what = site.what, value, "resolution: patched"),
            Patched::Differs => tracing::warn!(site = %offset, what = site.what, "resolution: not patched, the title's bytes there are not the ones expected (another build?)"),
            Patched::Failed(why) => tracing::warn!(site = %offset, what = site.what, why, "resolution: not patched"),
        }
        println!("resolution: {} ({value}) at title+{offset}: {done:?}", site.what);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse() {
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_output("1600x900"), None);
        assert_eq!(parse_output("1080p"), None);
        assert_eq!(parse_render_scale("0.67"), Some(0.67));
        assert_eq!(parse_render_scale("0.5"), Some(0.5));
        assert_eq!(parse_render_scale("1"), Some(1.0));
        assert_eq!(parse_render_scale("0.4"), None);
        assert_eq!(parse_render_scale("1.5"), None);
    }

    /// The calls and loads in the bytes compared are the ones the
    /// replacement keeps.
    #[test]
    fn the_output_block_is_the_titles() {
        let b = OUTPUT.bytes;
        assert_eq!(b.len(), 61);
        let target = |op_end: usize| (OUTPUT.at + op_end) as i64 + i64::from(i32::from_le_bytes(b[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(target(27), SET_SCREEN_SIZE as i64);
        assert_eq!(target(50), SET_SCREEN_SIZE as i64);
        assert_eq!(target(61), HEIGHT_CHANGED as i64);
        assert_eq!(target(41), SCREEN_HEIGHT as i64);
        assert_eq!(target(6), (SCREEN_HEIGHT - 4) as i64, "the width before it");
        // jne to the else path at 29; jmp +0x20 and je +7 both land at the end.
        assert_eq!((b[14], 16 + b[15] as usize), (0x75, 29));
        assert_eq!((b[27], 29 + b[28] as usize), (0xeb, 61));
        assert_eq!((b[52], 54 + b[53] as usize), (0x74, 61));
    }

    #[test]
    fn the_output_code_keeps_the_titles_else_path() {
        let c = output_code(1920, 1080);
        assert_eq!(c.len(), OUTPUT.bytes.len());
        assert_eq!(c[0], 0xb8);
        assert_eq!(c[1..5], 1920u32.to_le_bytes());
        assert_eq!(c[5..11], [0x89, 0x86, 0x04, 0x02, 0x00, 0x00]);
        assert_eq!(c[12..16], 1080u32.to_le_bytes());
        assert_eq!(c[16..22], [0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]);
        let target = |op_end: usize| (OUTPUT.at + op_end) as i64 + i64::from(i32::from_le_bytes(c[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(c[22..24], [0x8b, 0x2d]);
        assert_eq!(target(28), SCREEN_HEIGHT as i64);
        assert_eq!(c[32], 0xe8);
        assert_eq!(target(37), SET_SCREEN_SIZE as i64);
        assert_eq!(c[37..40], [0x39, 0xfd, 0x74]);
        assert_eq!(41 + c[40] as usize, 61, "je lands at the end");
        assert_eq!(c[43], 0xe8);
        assert_eq!(target(48), HEIGHT_CHANGED as i64);
        assert_eq!(c[48], 0xeb);
        assert_eq!(50 + c[49] as usize, 61, "jmp lands at the end");
        assert!(c[50..].iter().all(|&x| x == 0xcc));
    }

    #[test]
    fn the_scale_codes() {
        let own = own_scale_code(0.5);
        assert_eq!(own[..6], OWN_SCALE.bytes[..6]);
        assert_eq!(own[6..], 0.5f32.to_le_bytes());
        assert_eq!(OWN_SCALE.bytes[6..], (2.0f32 / 3.0).to_le_bytes());
        let ffx = ffx_scale_code(0.75);
        assert_eq!(ffx.len(), FFX_SCALE.bytes.len());
        assert_eq!(ffx[0x64..0x68], 0.75f32.to_le_bytes());
        for at in [0x28, 0x32] {
            assert_eq!(ffx[at..at + 4], [0xc5, 0xfa, 0x10, 0x05], "vmovss xmm0, [rip+d]");
            let disp = i32::from_le_bytes(ffx[at + 4..at + 8].try_into().unwrap());
            assert_eq!(at as i32 + 8 + disp, 0x64);
        }
        // Only the two displacements and the padding change.
        let changed: Vec<usize> = (0..ffx.len()).filter(|&i| ffx[i] != FFX_SCALE.bytes[i]).collect();
        assert!(changed.iter().all(|&i| (0x2c..0x30).contains(&i) || (0x36..0x3a).contains(&i) || (0x64..0x68).contains(&i)));
        assert_eq!(picker_code(7), [0xb8, 7, 0, 0, 0, 0xc3]);
        assert!(picker_code(6).len() <= PICKER_FRAME.bytes.len());
        assert_eq!(PICKER_FRAME.bytes[..6], [0x48, 0x83, 0xec, 0x28, 0x48, 0x8b]);
        assert_eq!(pool_check_code(), [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0xeb, 0x01]);
    }

    #[test]
    fn series_s_takes_the_titles_own_upscaler_unless_asked() {
        assert_eq!(upscaler_kind(None, true), Some(6));
        assert_eq!(upscaler_kind(None, false), None);
        assert_eq!(upscaler_kind(Some(7), true), Some(7));
        assert_eq!(upscaler_kind(Some(6), false), Some(6));
    }

    /// A synthetic image with the picker pair at 0x100 and 0x200.
    fn pair_image(second: &[u8]) -> Vec<u8> {
        let mut image = vec![0xccu8; 0x300];
        image[0x100..0x100 + PICKER.bytes.len()].copy_from_slice(PICKER.bytes);
        image[0x200..0x200 + second.len()].copy_from_slice(second);
        image
    }

    fn write_pair(image: &mut [u8], fail_at: Option<usize>) -> Result<(), (usize, Patched)> {
        let code = picker_code(6);
        let pair = [(0x100, PICKER.bytes, &code[..]), (0x200, PICKER_FRAME.bytes, &code[..])];
        let base = image.as_mut_ptr() as usize;
        let mut calls = 0;
        // SAFETY: the image belongs to the test and is already writable.
        unsafe {
            write_all_with(base, image.len(), &pair, |_, _| {
                calls += 1;
                if Some(calls) == fail_at {
                    return Err("refused");
                }
                Ok(())
            })
        }
    }

    /// Both upscaler choices are written, or neither: a second site that is
    /// not the expected bytes leaves the first as it was.
    #[test]
    fn the_upscaler_pair_is_written_whole_or_not_at_all() {
        let mut image = pair_image(PICKER_FRAME.bytes);
        assert_eq!(write_pair(&mut image, None), Ok(()));
        assert_eq!(image[0x100..0x106], picker_code(6)[..]);
        assert_eq!(image[0x200..0x206], picker_code(6)[..]);

        let mut other = PICKER_FRAME.bytes.to_vec();
        other[10] ^= 0xff;
        let mut image = pair_image(&other);
        let before = image.clone();
        assert_eq!(write_pair(&mut image, None), Err((1, Patched::Differs)));
        assert_eq!(image, before, "nothing written when the second site differs");
    }

    /// A failure preparing the second site leaves both original sequences intact.
    #[test]
    fn a_failed_second_preparation_leaves_both_sites_unchanged() {
        let mut image = pair_image(PICKER_FRAME.bytes);
        let before = image.clone();
        assert_eq!(write_pair(&mut image, Some(2)), Err((1, Patched::Failed("refused"))));
        assert_eq!(image, before);
    }

    #[test]
    fn refusing_any_later_page_leaves_all_four_step_sites_unchanged() {
        let mut image = [0u8; 32];
        let before = image;
        let sites: [(usize, &[u8], &[u8]); 4] = [(4, &[0], &[1]), (8, &[0], &[2]), (12, &[0], &[3]), (16, &[0], &[4])];
        for fail_at in 2..=4 {
            let mut calls = 0;
            // SAFETY: all sites belong to the writable image owned by this test.
            let result = unsafe {
                write_all_with(image.as_mut_ptr() as usize, image.len(), &sites, |_, _| {
                    calls += 1;
                    if calls == fail_at { Err("refused") } else { Ok(()) }
                })
            };
            assert_eq!(result, Err((fail_at - 1, Patched::Failed("refused"))));
            assert_eq!(image, before);
        }
    }

    #[cfg(windows)]
    #[test]
    fn cross_page_group_restores_protection_after_success_and_failure() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
            PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree,
            VirtualProtect, VirtualQuery,
        };
        const PAGE: usize = 0x1000;
        // SAFETY: the test owns the reserved pages and writes code-shaped
        // data into them without executing it.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), 2 * PAGE, MEM_COMMIT, PAGE_READWRITE).cast::<u8>(), base);
            base.add(PAGE - 1).write(1);
            base.add(PAGE).write(2);
            let mut old = 0;
            assert_ne!(VirtualProtect(base.cast(), PAGE, PAGE_EXECUTE_READ, &mut old), 0);
            assert_ne!(VirtualProtect(base.add(PAGE).cast(), PAGE, PAGE_READONLY, &mut old), 0);

            let original = || (base.add(PAGE - 1).read(), base.add(PAGE).read());
            let protection = |offset| {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(base.add(offset).cast(), &mut info, std::mem::size_of_val(&info)), 0);
                info.Protect
            };
            let patch: (usize, &[u8], &[u8]) = (PAGE - 1, &[1, 2], &[3, 4]);
            assert!(matches!(write_all(base as usize, 3 * PAGE, &[patch, (2 * PAGE, &[0], &[1])]), Err((1, Patched::Failed(_)))));
            assert_eq!(original(), (1, 2));
            assert_eq!((protection(0), protection(PAGE)), (PAGE_EXECUTE_READ, PAGE_READONLY));

            assert_eq!(write_all(base as usize, 3 * PAGE, &[patch]), Ok(()));
            assert_eq!(original(), (3, 4));
            assert_eq!((protection(0), protection(PAGE)), (PAGE_EXECUTE_READ, PAGE_READONLY));
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_site_is_written_only_over_its_bytes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, VirtualAlloc, VirtualFree};
        // SAFETY: fresh memory of our own, read-only to code as the image's is.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, 0x40) as *mut u8;
            assert!(!place.is_null());
            let site = Site { at: 0x100, what: "test", bytes: OWN_SCALE.bytes };
            std::ptr::copy_nonoverlapping(site.bytes.as_ptr(), place.add(0x100), site.bytes.len());
            let mut old = 0;
            windows_sys::Win32::System::Memory::VirtualProtect(place.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old);
            assert_eq!(write(place as usize, 0x1000, &site, &own_scale_code(0.5)), Patched::Done);
            assert_eq!(std::slice::from_raw_parts(place.add(0x106), 4), 0.5f32.to_le_bytes());
            // Already changed: its bytes differ now, nothing is written again.
            assert_eq!(write(place as usize, 0x1000, &site, &own_scale_code(0.75)), Patched::Differs);
            assert_eq!(std::slice::from_raw_parts(place.add(0x106), 4), 0.5f32.to_le_bytes());
            let far = Site { at: 0xffa, what: "test", bytes: OWN_SCALE.bytes };
            assert!(matches!(write(place as usize, 0x1000, &far, &own_scale_code(0.5)), Patched::Failed(_)));
            VirtualFree(place.cast(), 0, MEM_RELEASE);
        }
    }
}
