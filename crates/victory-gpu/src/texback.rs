//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else { return false };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let _mapping = crate::watch::title_memory_reading();
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: crate::host::TextureReadback,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        )
    };
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let buffer = host.texture_readback_buffer(total)?;
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(buffer.resource()) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back { buffer, footprints, rows, mips, layout: target.layout, ticket: target.ticket })
}

/// Temporary pixels and the layouts reused by one thread returning textures.
#[derive(Default)]
pub(crate) struct Scratch {
    pixels: Vec<u8>,
    tiling: victory_formats::texture::TilingCache,
}

/// Whether flight texture readbacks use parallel landing (`VICTORY_TEXBACK_PARALLEL=1`,
/// opt-in, off by default; `VICTORY_AB` can flip it live in `diagnostics` builds).
fn parallel_enabled() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_TEXBACK_PARALLEL") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_parallel_switch(std::env::var("VICTORY_TEXBACK_PARALLEL").ok().as_deref()))
}

fn parse_parallel_switch(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v.trim() == "1")
}

/// Once the list has run, tile its texels back into the same title allocation.
///
/// The `TICKETS` lock is narrowed to the final validity check and copy into
/// title memory (after XaviFortes/VIctory-RDNA `e4250eb`), preserving the
/// `MAPPING -> TICKETS` lock order so parallel workers can read, map, and tile
/// disjoint textures concurrently without serializing on `TICKETS`.
pub(crate) fn land(back: &Back, scratch: &mut Scratch) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    let _mapping = crate::watch::title_memory_reading();
    let (at, end) = back.range();
    let size = end - at;
    if scratch.pixels.len() < size {
        scratch.pixels.reserve_exact(size - scratch.pixels.len());
        scratch.pixels.resize(size, 0);
    }
    let tiled = &mut scratch.pixels[..size];
    // Keep title bytes the subresources do not cover.
    if !crate::host::read_into(at as *const u8, tiled) {
        return;
    }
    let buffer = back.buffer.resource();
    let mut data = std::ptr::null_mut();
    // SAFETY: our read-back buffer, written by the list that has run.
    if unsafe { buffer.Map(0, None, Some(&mut data)) }.is_err() {
        return;
    }
    if data.is_null() {
        // SAFETY: Map succeeded, even though it supplied no pointer.
        unsafe { buffer.Unmap(0, None) };
        return;
    }
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: the subresource's part of the mapped buffer.
        let linear = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len) };
        let pitches = (pitch, pitch * rows as usize);
        if let Err(e) = scratch.tiling.tile(&back.layout, tiled, (mip, slice), linear, pitches) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
            // SAFETY: mapped above.
            unsafe { buffer.Unmap(0, None) };
            return;
        }
    }
    // SAFETY: mapped above.
    unsafe { buffer.Unmap(0, None) };
    let ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        drop(ticket_guard);
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
        }
        return;
    }
    crate::host::copy_mapped(at as *mut u8, tiled.as_ptr(), size, true);
    drop(ticket_guard);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        let stats = back.buffer.stats();
        tracing::info!(
            copied,
            bytes = size,
            created = stats.created,
            reused = stats.reused,
            allocation_failures = stats.allocation_failures,
            allocation_us = stats.allocation_us,
            peak_allocation_us = stats.peak_allocation_us,
            active_buffers = stats.active_buffers,
            active_bytes = stats.active_bytes,
            peak_active_bytes = stats.peak_active_bytes,
            idle_bytes = stats.idle_bytes,
            peak_idle_bytes = stats.peak_idle_bytes,
            "texture readback landed"
        );
    }
}

/// Lands one texture readback using a thread-local [`Scratch`] buffer and
/// tiling cache, for worker threads in [`land_many`].
fn land_local(back: &Back) {
    thread_local! {
        static LOCAL: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
    }
    LOCAL.with(|scratch| land(back, &mut scratch.borrow_mut()));
}

/// Decides which readback indices to execute and whether they are pairwise
/// disjoint (safe to tile and land in parallel).
///
/// Each entry is `(owner, at, end, valid)`. An earlier readback is skipped only
/// when a later still-valid readback shares the exact same `(owner, at, end)`,
/// so both tickets have identical invalidation predicates (`released`,
/// `replaced`, `remapped`) and cover the same texture subresources. Any
/// overlapping remaining entries fall back to strict sequential submission
/// order (`parallel == false`).
fn plan_land_many(entries: &[(usize, usize, usize, bool)]) -> (Vec<usize>, bool) {
    let mut active = Vec::with_capacity(entries.len());
    for (i, &(owner, at, end, _)) in entries.iter().enumerate() {
        let superseded = entries[i + 1..]
            .iter()
            .any(|&(lowner, lat, lend, lvalid)| lowner == owner && lat == at && lend == end && lvalid);
        if !superseded {
            active.push(i);
        }
    }
    if active.len() <= 1 {
        return (active, false);
    }
    let mut sorted: Vec<(usize, usize)> = active.iter().map(|&i| (entries[i].1, entries[i].2)).collect();
    sorted.sort_unstable();
    let overlaps = sorted.windows(2).any(|w| w[0].1 > w[1].0);
    (active, !overlaps)
}

/// Lands a flight's texture readbacks. When `VICTORY_TEXBACK_PARALLEL=1` is
/// enabled, skips superseded duplicates for the same `(owner, at, end)` and
/// tiles pairwise-disjoint readbacks in parallel across `crate::land::each_view`
/// (after XaviFortes/VIctory-RDNA `e4250eb` and `af65a79`). When off (the
/// default), lands sequentially in exact submission order.
pub(crate) fn land_many(backs: &[&Back], scratch: &mut Scratch) {
    land_many_with(backs, scratch, parallel_enabled());
}

fn land_many_with(backs: &[&Back], scratch: &mut Scratch, parallel_on: bool) {
    if backs.is_empty() {
        return;
    }
    if !parallel_on || backs.len() == 1 {
        for &back in backs {
            land(back, scratch);
        }
        return;
    }
    let entries: Vec<(usize, usize, usize, bool)> = backs
        .iter()
        .map(|b| {
            let (at, end) = b.range();
            (b.ticket.owner, at, end, b.ticket.valid.load(Ordering::Acquire))
        })
        .collect();
    let (active, parallel) = plan_land_many(&entries);
    if active.is_empty() {
        return;
    }
    if !parallel {
        for i in active {
            land(backs[i], scratch);
        }
        return;
    }
    crate::land::each_view(false, active.len(), &|slot| land_local(backs[active[slot]]));
}

/// Register the initialized upload row copied by the host's test list as a
/// linear 1×1 readback, so it goes through the real landing/reset path.
#[cfg(test)]
pub(crate) fn buffer_back_for_test(buffer: crate::host::TextureReadback, dst: &mut [u8; 256]) -> Back {
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    let layout = Layout::new(&Desc {
        dimension: Dimension::Tex2d,
        width: 1,
        height: 1,
        depth: 1,
        mips: 1,
        element_bytes: 4,
        block: 1,
        samples: 1,
        swizzle: Swizzle::LINEAR,
    }).unwrap();
    assert_eq!(layout.size, dst.len() as u64);
    let at = dst.as_mut_ptr() as usize;
    Back {
        buffer,
        footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                Width: 1,
                Height: 1,
                Depth: 1,
                RowPitch: 256,
            },
        }],
        rows: vec![1],
        mips: 1,
        layout,
        ticket: track(at, at, at + dst.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn parallel_switch_defaults_off_and_accepts_only_one() {
        assert!(!parse_parallel_switch(None));
        assert!(!parse_parallel_switch(Some("0")));
        assert!(!parse_parallel_switch(Some("")));
        assert!(!parse_parallel_switch(Some("false")));
        assert!(!parse_parallel_switch(Some("2")));
        assert!(parse_parallel_switch(Some("1")));
        assert!(parse_parallel_switch(Some(" 1 ")));
    }

    #[test]
    fn plan_land_many_handles_disjoint_dedup_overlap_and_stale_tickets() {
        // Pairwise disjoint ranges run in parallel.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1100, true),
                (2, 0x2000, 0x2100, true),
                (3, 0x1100, 0x1200, true),
            ]),
            (vec![0, 1, 2], true)
        );
        // Same-owner exact duplicate keeps only the last valid entry; remaining disjoint entries stay parallel.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1100, true),
                (2, 0x2000, 0x2100, true),
                (1, 0x1000, 0x1100, true),
            ]),
            (vec![1, 2], true)
        );
        // Different-owner aliases covering the exact same range never suppress each other and run sequentially.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1100, true),
                (9, 0x1000, 0x1100, true),
            ]),
            (vec![0, 1], false)
        );
        // An invalidated later duplicate never suppresses an earlier still-valid entry.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1100, true),
                (1, 0x1000, 0x1100, false),
            ]),
            (vec![0, 1], false)
        );
        // Partially overlapping ranges fall back to sequential submission order.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1200, true),
                (2, 0x1100, 0x1300, true),
            ]),
            (vec![0, 1], false)
        );
        // Contained sub-range overlap falls back to sequential submission order.
        assert_eq!(
            plan_land_many(&[
                (1, 0x1000, 0x1400, true),
                (2, 0x1100, 0x1200, true),
            ]),
            (vec![0, 1], false)
        );
    }

    /// Full 256-byte linear 64x1 R32_UINT readback targeting `[dst_ptr, dst_ptr + 256)`
    /// with an explicit `owner` identity, so every byte of the 256-byte range is
    /// written on landing for overlap-order and stale-ticket verification.
    fn full_row_back_for_test(buffer: crate::host::TextureReadback, dst_ptr: *mut u8, owner: usize) -> Back {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 1,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::LINEAR,
        })
        .unwrap();
        assert_eq!(layout.size, 256);
        let at = dst_ptr as usize;
        Back {
            buffer,
            footprints: vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                    Width: 64,
                    Height: 1,
                    Depth: 1,
                    RowPitch: 256,
                },
            }],
            rows: vec![1],
            mips: 1,
            layout,
            ticket: track(owner, at, at + 256),
        }
    }

    fn record_row_back(
        host: &'static crate::host::Host,
        fill: u8,
        dst_ptr: *mut u8,
        owner: usize,
    ) -> (Back, ID3D12Resource, crate::host::HostList) {
        const ROW: u64 = 256;
        let (src, ptr) = host.upload_buffer(ROW).expect("upload buffer");
        // SAFETY: freshly allocated 256-byte upload mapping.
        unsafe { ptr.write_bytes(fill, ROW as usize) };
        let lease = host.texture_readback_buffer(ROW).expect("readback lease");
        let list = host.new_list().expect("host list");
        // SAFETY: open list, live upload and readback buffers of ROW bytes.
        unsafe { list.list.CopyBufferRegion(lease.resource(), 0, &src, 0, ROW) };
        list.close();
        host.execute(&[&list]);
        (full_row_back_for_test(lease, dst_ptr, owner), src, list)
    }

    /// Real D3D12 (HW and WARP) test verifying both baseline sequential and
    /// opt-in parallel `land_many_with`: disjoint textures land concurrently,
    /// same-owner duplicates keep the latest write, and partially overlapping
    /// destination ranges strictly preserve submission/ticket order in both
    /// forward and reverse directions.
    #[test]
    fn land_many_preserves_overlap_order_and_disjoint_parallel_on_gpu() {
        let Some(host) = crate::host::host() else {
            return;
        };
        const ROW: usize = 256;
        const COUNT: usize = 8;

        for parallel_on in [false, true] {
            // 1. Disjoint targets + same-owner duplicate latest-wins.
            let mut dsts = [[0u8; ROW]; COUNT];
            let mut uploads = Vec::new();
            let mut lists = Vec::new();
            let mut backs = Vec::new();
            for (i, dst) in dsts.iter_mut().enumerate() {
                let owner = 0x1000 + i;
                let (back, src, list) = record_row_back(host, 0x20 + i as u8, dst.as_mut_ptr(), owner);
                backs.push(back);
                uploads.push(src);
                lists.push(list);
            }
            let (ovr, src_ovr, list_ovr) = record_row_back(host, 0xEE, dsts[0].as_mut_ptr(), 0x1000);
            backs.push(ovr);
            uploads.push(src_ovr);
            lists.push(list_ovr);

            let refs: Vec<&Back> = backs.iter().collect();
            let mut scratch = Scratch::default();
            land_many_with(&refs, &mut scratch, parallel_on);
            assert_eq!(dsts[0], [0xEE; ROW], "latest valid write wins (parallel_on={parallel_on})");
            for (i, dst) in dsts.iter().enumerate().skip(1) {
                assert_eq!(*dst, [0x20 + i as u8; ROW], "disjoint slot {i} (parallel_on={parallel_on})");
            }
            drop((backs, uploads, lists));

            // 2. Partial forward overlap: [0..256) = 0xAA then [128..384) = 0xBB.
            let mut shared = [0x55u8; 512];
            let base = shared.as_mut_ptr();
            // SAFETY: base+0..256 and base+128..384 both lie inside `shared[..512]`.
            let (b1, u1, l1) = record_row_back(host, 0xAA, base, 0x2001);
            let (b2, u2, l2) = record_row_back(host, 0xBB, unsafe { base.add(128) }, 0x2002);
            land_many_with(&[&b1, &b2], &mut scratch, parallel_on);
            assert_eq!(&shared[0..128], &[0xAA; 128], "forward overlap head");
            assert_eq!(&shared[128..256], &[0xBB; 128], "forward overlap shared middle");
            assert_eq!(&shared[256..384], &[0xBB; 128], "forward overlap tail");
            assert_eq!(&shared[384..512], &[0x55; 128], "untouched suffix");
            drop((b1, b2, u1, u2, l1, l2));

            // 3. Reverse overlap: [128..384) = 0xCC then [0..256) = 0xDD.
            shared.fill(0x55);
            let base = shared.as_mut_ptr();
            // SAFETY: base+128..384 and base+0..256 both lie inside `shared[..512]`.
            let (b3, u3, l3) = record_row_back(host, 0xCC, unsafe { base.add(128) }, 0x2003);
            let (b4, u4, l4) = record_row_back(host, 0xDD, base, 0x2004);
            land_many_with(&[&b3, &b4], &mut scratch, parallel_on);
            assert_eq!(&shared[0..256], &[0xDD; 256], "reverse overlap later write wins");
            assert_eq!(&shared[256..384], &[0xCC; 128], "reverse overlap earlier tail preserved");
            assert_eq!(&shared[384..512], &[0x55; 128], "untouched suffix");
            drop((b3, b4, u3, u4, l3, l4));
        }
        assert_eq!(host.log_messages(), None);
    }

    /// Real D3D12 (HW and WARP) stale-ticket isolation test: cancelled
    /// (`released`), superseded (`replaced`), and partially remapped
    /// (`remapped`) tickets never overwrite destination memory, out-of-range
    /// invalidations leave valid tickets intact, and invalidated later
    /// duplicates never suppress an earlier valid ticket.
    #[test]
    fn land_many_blocks_stale_cancelled_superseded_and_remapped_tickets_on_gpu() {
        let Some(host) = crate::host::host() else {
            return;
        };
        const ROW: usize = 256;

        for parallel_on in [false, true] {
            let mut dst_released = [0x11u8; ROW];
            let mut dst_replaced = [0x22u8; ROW];
            let mut dst_remapped = [0x33u8; ROW];
            // Reserve a trailing guard region so `remapped(at + ROW, 64)` cannot alias
            // another local buffer on the stack.
            let mut dst_valid = [0x44u8; ROW * 2];
            let mut dst_stale_dup = [0x55u8; ROW];

            let (b_rel, u_rel, l_rel) = record_row_back(host, 0xF1, dst_released.as_mut_ptr(), 0x3001);
            let (b_rep_old, u_rep_old, l_rep_old) = record_row_back(host, 0xF2, dst_replaced.as_mut_ptr(), 0x3002);
            let (b_remap, u_remap, l_remap) = record_row_back(host, 0xF3, dst_remapped.as_mut_ptr(), 0x3003);
            let (b_val, u_val, l_val) = record_row_back(host, 0x99, dst_valid.as_mut_ptr(), 0x3004);
            let (b_dup_good, u_dg, l_dg) = record_row_back(host, 0x7A, dst_stale_dup.as_mut_ptr(), 0x3005);
            let (b_dup_stale, u_ds, l_ds) = record_row_back(host, 0xFA, dst_stale_dup.as_mut_ptr(), 0x3006);

            // Cancel by owner, supersede by base address, invalidate by partial remap,
            // and cancel only the later alias on `dst_stale_dup`.
            released(0x3001);
            replaced(dst_replaced.as_ptr() as usize);
            let (b_rep_new, u_rep_new, l_rep_new) = record_row_back(host, 0x88, dst_replaced.as_mut_ptr(), 0x3007);
            remapped(dst_remapped.as_ptr() as usize + 128, 4);
            released(0x3006);
            // Out-of-range invalidation right after `dst_valid[..ROW]` must not cancel `b_val`.
            remapped(dst_valid.as_ptr() as usize + ROW, 64);

            let mut scratch = Scratch::default();
            land_many_with(
                &[&b_rel, &b_rep_old, &b_rep_new, &b_remap, &b_val, &b_dup_good, &b_dup_stale],
                &mut scratch,
                parallel_on,
            );

            assert_eq!(dst_released, [0x11; ROW], "released ticket must not write");
            assert_eq!(dst_replaced, [0x88; ROW], "superseded old ticket skipped, newer replacement lands");
            assert_eq!(dst_remapped, [0x33; ROW], "partially remapped ticket must not write");
            assert_eq!(&dst_valid[..ROW], &[0x99; ROW], "out-of-range remap must not cancel valid ticket");
            assert_eq!(&dst_valid[ROW..], &[0x44; ROW], "guard region untouched");
            assert_eq!(dst_stale_dup, [0x7A; ROW], "cancelled later ticket must not suppress earlier valid write");

            drop((
                b_rel, b_rep_old, b_rep_new, b_remap, b_val, b_dup_good, b_dup_stale,
                u_rel, u_rep_old, u_rep_new, u_remap, u_val, u_dg, u_ds,
                l_rel, l_rep_old, l_rep_new, l_remap, l_val, l_dg, l_ds,
            ));
        }
        assert_eq!(host.log_messages(), None);
    }
}
