//! Byte-masked GPU patch shader for [`crate::shadow::masked_patch_now`]
//! (after `fxceless/victory-fork` commits `3221ac4`, `0e006f0`, `50c79c3`).
//!
//! Each 12-byte entry `(offset, mask, value)` updates only the masked bytes
//! in the card-side buffer while preserving untouched sibling bytes written
//! by in-flight shaders. Shares [`super::CARD_COMPARE`]'s root signature.

/// Byte-masked patch shader ([`crate::shadow::masked_patch_now`]; after
/// fxceless/victory-fork `3221ac4`): for each changed dword `(offset, mask,
/// value)` (12 bytes), updates only the masked bytes in `copy` and preserves
/// all other bytes written by in-flight shaders. Shares [`super::CARD_COMPARE`]'s
/// root signature.
pub(crate) const MASKED_PATCH: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Patch : register(b0) {
    uint count;
    uint first;
    uint unused0;
    uint unused1;
    uint unused2;
    uint unused3;
};
ByteAddressBuffer words : register(t0);
RWByteAddressBuffer copy : register(u0);
RWByteAddressBuffer unused : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint3 w = words.Load3((first + id.x) * 12u);
    copy.Store(w.x, (copy.Load(w.x) & ~w.y) | (w.z & w.y));
}
"#;

/// Compiles [`MASKED_PATCH`] as `cs_5_1` (matching [`super::CARD_COMPARE`]).
pub(crate) fn masked_patch_code() -> windows::core::Result<windows::Win32::Graphics::Direct3D::ID3DBlob> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    let mut code = None;
    // SAFETY: the source and names live through the call; the out pointer is ours.
    unsafe {
        D3DCompile(
            MASKED_PATCH.as_ptr().cast(),
            MASKED_PATCH.len(),
            windows::core::s!("masked patch"),
            None,
            None,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?;
    }
    code.ok_or_else(windows::core::Error::empty)
}

#[cfg(test)]
mod tests {
    use super::super::*;

    /// `MASKED_PATCH` compiles as `cs_5_1` with the compare's root signature
    /// embedded (no device needed).
    #[test]
    fn the_masked_patch_compiles() {
        let code = super::masked_patch_code().expect("MASKED_PATCH compiles");
        // SAFETY: the blob's own bytes.
        let size = unsafe { code.GetBufferSize() };
        assert!(size > 100);
    }

    fn assert_no_host_d3d12_errors(host: &Host, phase: &str) {
        let Some(iq) = &host.messages else {
            return;
        };
        // SAFETY: live D3D12 info queue owned by `host`.
        unsafe {
            let count = iq.GetNumStoredMessages();
            let mut errors = Vec::new();
            for idx in 0..count {
                let mut len = 0usize;
                let _ = iq.GetMessage(idx, None, &mut len);
                if len == 0 {
                    continue;
                }
                let mut buf = vec![0u8; len];
                let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if iq.GetMessage(idx, Some(msg), &mut len).is_ok() {
                    let sev = (*msg).Severity;
                    if sev == D3D12_MESSAGE_SEVERITY_CORRUPTION || sev == D3D12_MESSAGE_SEVERITY_ERROR {
                        let desc = std::slice::from_raw_parts(
                            (*msg).pDescription.cast::<u8>(),
                            (*msg).DescriptionByteLength.saturating_sub(1),
                        );
                        errors.push(String::from_utf8_lossy(desc).into_owned());
                    }
                }
            }
            iq.ClearStoredMessages();
            assert!(errors.is_empty(), "D3D12 debug errors during {phase}:\n{}", errors.join("\n"));
        }
    }

    /// Runs the real `Host::card_ready` + `Host::card_before` + `Host::card_done`
    /// pipeline with `VICTORY_MASKED_PATCH=1`, exercising:
    /// - overlapping masked patches (`p1` then `p2`) + untouched flight bytes,
    /// - unaligned `card.dirty` (`copy.runs`) and aligned 4-byte `card.dirty`
    ///   (`copy.pairs` `SCATTER` under `VICTORY_SEND_OWN=1`),
    /// - sub-dword byte overlaps within the SAME 32-bit dword (earlier masked
    ///   patch on bytes `0..=1` + newer dirty copy on bytes `1..=3`; earlier
    ///   patch `0..=1` + newer dirty `1..=2` + flight byte `3`; earlier patch
    ///   `1..=3` + newer dirty `0..=1`), and
    /// - `CARD_COMPARE` piece hashing under `VICTORY_D3D12_DEBUG=1`.
    fn run_host_card_before_ordering_check() {
        let Some(host) = host() else {
            eprintln!("no D3D12 device; skipping real Host::card_before masked-patch test");
            return;
        };
        assert!(
            crate::shadow::masked_patch_enabled(),
            "child test process must have VICTORY_MASKED_PATCH=1"
        );
        if !host.card_ready() {
            return;
        }
        assert!(crate::shadow::PATCH_READY.load(Ordering::Relaxed));

        const LEN: usize = 64 << 10;
        let (staging, data) = host.shared_buffer(LEN as u64).expect("staging buffer");
        let card_buf = host.uav_buffer(LEN as u64).expect("card UAV buffer");
        let readback = host.read_back_buffer(LEN as u64).expect("readback buffer");
        // SAFETY: `LEN` mapped bytes.
        unsafe { std::ptr::write_bytes(data, 0x10, LEN) };

        let mut title = vec![0x10u8; LEN];
        let shadow = crate::shadow::Shadow::backed_card_test(
            title.as_ptr() as u64,
            card_buf.clone(),
            staging.clone(),
            data,
            LEN as u64,
        );
        let _tracked = shadow.track_carded_test();

        // Prime card buffer with initial 0x10 plus simulated in-flight GPU writes
        // (0xEE) on bytes 1 & 3 (dword 0), byte 5 (dword 1), byte 9 (dword 2),
        // and byte 23 (dword 5).
        {
            let (init_buf, init_ptr) = host.shared_buffer(LEN as u64).expect("init buffer");
            // SAFETY: `LEN` mapped bytes.
            unsafe {
                std::ptr::write_bytes(init_ptr, 0x10, LEN);
                *init_ptr.add(1) = 0xEE;
                *init_ptr.add(3) = 0xEE;
                *init_ptr.add(5) = 0xEE;
                *init_ptr.add(9) = 0xEE;
                *init_ptr.add(23) = 0xEE;
            }
            let list = host.new_list().expect("init list");
            // SAFETY: open list and live buffers.
            unsafe {
                list.list.CopyBufferRegion(&card_buf, 0, &init_buf, 0, LEN as u64);
            }
            list.close();
            host.execute(&[&list]);
        }

        // Queue TWO overlapping masked patches (`p1` then `p2`) covering:
        // - dword 0 (0..4): p1 sets bytes 0 & 2, p2 overrides byte 2 -> [0x21, 0xEE, 0x42, 0xEE]
        // - dword 2 (8..12): p2 sets byte 8 = 0x55, later overridden by dirty (8, 9) -> 0x99
        // - dword 3 (12..16): p2 sets byte 12 = 0x66, later overridden by aligned SCATTER dirty (12, 16)
        // - dword 4 (16..20): earlier patch sets bytes 0..=1 ([0x31, 0x32]), newer dirty overwrites bytes 1..=3 ([0xB1, 0xB2, 0xB3]) -> [0x31, 0xB1, 0xB2, 0xB3]
        // - dword 5 (20..24): earlier patch sets bytes 0..=1 ([0x41, 0x42]), newer dirty overwrites bytes 1..=2 ([0xC1, 0xC2]), byte 3 stays flight 0xEE -> [0x41, 0xC1, 0xC2, 0xEE]
        // - dword 6 (24..28): earlier patch sets bytes 1..=3 ([0x51, 0x52, 0x53]), newer dirty overwrites bytes 0..=1 ([0xD0, 0xD1]) -> [0xD0, 0xD1, 0x52, 0x53]
        let mut p1 = Vec::new();
        title[0] = 0x21;
        title[2] = 0x31;
        title[16] = 0x31;
        title[17] = 0x32;
        title[20] = 0x41;
        title[21] = 0x42;
        title[25] = 0x51;
        title[26] = 0x52;
        title[27] = 0x53;
        crate::shadow::patch_words(
            0,
            &title[..32],
            &[(0, 1), (2, 1), (16, 2), (20, 2), (25, 3)],
            &mut p1,
        );

        let mut p2 = Vec::new();
        title[2] = 0x42; // overlaps byte 2 from p1 -> 0x42 must win
        title[8] = 0x55; // later overwritten by unaligned dirty -> 0x99 must win
        title[12] = 0x66; // later overwritten by aligned 4-byte SCATTER dirty -> 0xA1..0xA4 must win
        crate::shadow::patch_words(0, &title[..32], &[(2, 1), (8, 1), (12, 1)], &mut p2);

        // SAFETY: `data` holds `LEN` bytes.
        unsafe {
            *data.add(0) = 0x21;
            *data.add(2) = 0x42;
            *data.add(4) = 0x77;
            *data.add(8) = 0x99;
            *data.add(12) = 0xA1;
            *data.add(13) = 0xA2;
            *data.add(14) = 0xA3;
            *data.add(15) = 0xA4;
            // Dword 4: byte 16 keeps earlier patch 0x31; bytes 17..20 overwritten by newer dirty
            *data.add(16) = 0x31;
            *data.add(17) = 0xB1;
            *data.add(18) = 0xB2;
            *data.add(19) = 0xB3;
            // Dword 5: byte 20 keeps earlier patch 0x41; bytes 21..23 overwritten by newer dirty; byte 23 stays flight 0xEE
            *data.add(20) = 0x41;
            *data.add(21) = 0xC1;
            *data.add(22) = 0xC2;
            // Dword 6: bytes 24..26 overwritten by newer dirty; bytes 26..28 keep earlier patch [0x52, 0x53]
            *data.add(24) = 0xD0;
            *data.add(25) = 0xD1;
            *data.add(26) = 0x52;
            *data.add(27) = 0x53;
        }
        shadow.record_masked_patch_test(p1, vec![(0, 1), (2, 1), (16, 18), (20, 22), (25, 28)]);
        shadow.record_masked_patch_test(p2, vec![(2, 1), (8, 1), (12, 1)]);
        shadow.record_dirty_test(&[
            (4, 5),
            (8, 9),
            (12, 16),
            (17, 20), // sub-dword overlap with patch (16, 18) on byte 17
            (21, 23), // sub-dword overlap with patch (20, 22) on byte 21, preserving flight byte 23
            (24, 26), // sub-dword overlap with patch (25, 28) on byte 25
        ]);

        // Run `Host::card_before` (which records & executes the sync list on `host.queue`),
        // then copy the card buffer to `readback` and wait.
        let plan = host
            .card_before(
                &[(shadow.clone(), vec![(title.as_ptr() as usize, title.as_ptr() as usize + LEN)])],
                &crate::picked::Ranges::default(),
                false,
            )
            .expect("card_before must return a plan");
        let list = host.new_list().expect("readback list");
        // SAFETY: open list, live buffers, card buffer returns to COMMON between submissions.
        unsafe {
            list.list.CopyBufferRegion(&readback, 0, &card_buf, 0, 32);
        }
        list.close();
        host.execute(&[&list]);
        host.card_done(plan);
        assert_no_host_d3d12_errors(host, "Host::card_before masked-patch + sub-dword dirty ordering");

        let mut ptr = std::ptr::null_mut();
        let mut got = [0u8; 32];
        // SAFETY: `execute` waited for completion; 32 bytes copied.
        unsafe {
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), got.as_mut_ptr(), 32);
            readback.Unmap(0, None);
        }
        assert_eq!(&got[0..4], &[0x21, 0xEE, 0x42, 0xEE], "dword 0 overlapping patch + flight bytes");
        assert_eq!(&got[4..8], &[0x77, 0xEE, 0x10, 0x10], "dword 1 dirty byte + flight byte");
        assert_eq!(&got[8..12], &[0x99, 0xEE, 0x10, 0x10], "dword 2 dirty-after-patch ordering");
        assert_eq!(&got[12..16], &[0xA1, 0xA2, 0xA3, 0xA4], "dword 3 aligned SCATTER dirty-after-patch ordering");
        assert_eq!(
            &got[16..20],
            &[0x31, 0xB1, 0xB2, 0xB3],
            "dword 4 sub-dword overlap: byte 0 keeps earlier patch, bytes 1..=3 take newer dirty"
        );
        assert_eq!(
            &got[20..24],
            &[0x41, 0xC1, 0xC2, 0xEE],
            "dword 5 sub-dword overlap: byte 0 keeps patch, bytes 1..=2 take newer dirty, byte 3 keeps flight"
        );
        assert_eq!(
            &got[24..28],
            &[0xD0, 0xD1, 0x52, 0x53],
            "dword 6 reverse sub-dword overlap: bytes 0..=1 take newer dirty, bytes 2..=3 keep earlier patch"
        );
    }

    /// Verifies `VICTORY_MASKED_PATCH` defaults OFF and exercises the real
    /// `Host::card_ready` + `Host::card_before` + `Host::card_done` ordering
    /// pipeline in an isolated `VICTORY_MASKED_PATCH=1` subprocess (both HW
    /// and, under `diagnostics`, WARP) under `VICTORY_D3D12_DEBUG=1`.
    #[test]
    fn hw_host_card_before_applies_masked_patch_and_dirty_in_order() {
        const CHILD: &str = "VICTORY_MASKED_PATCH_HOST_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            run_host_card_before_ordering_check();
            return;
        }
        if std::env::var_os("VICTORY_MASKED_PATCH").is_none() {
            assert_eq!(crate::shadow::masked_patch_mode(), 0, "VICTORY_MASKED_PATCH must be opt-in");
            assert!(!crate::shadow::masked_patch_enabled());
            assert!(!crate::shadow::masked_patch_now());
        }
        let run_child = |warp: bool| {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args([
                "hw_host_card_before_applies_masked_patch_and_dirty_in_order",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("VICTORY_MASKED_PATCH", "1");
            if warp {
                cmd.env("VICTORY_D3D12_WARP", "1");
            } else {
                cmd.env_remove("VICTORY_D3D12_WARP");
            }
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "masked-patch Host::card_before child failed (warp={warp}):\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        };
        run_child(false);
        #[cfg(feature = "diagnostics")]
        run_child(true);
    }
}
