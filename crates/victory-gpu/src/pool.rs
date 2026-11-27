//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from blocks made once.
//!
//! Each such texture had an allocation of its own and its staging buffer
//! another. On an RX 7900 XTX an allocation takes about 310 microseconds
//! whatever its size, on one thread (`texture::allocation_costs`), and a
//! texture cut from a block 6: the pass that makes the textures shaders
//! pick by index (`draw.rs` `heap_textures`) got through 350-500 of the
//! 24,000 queued when the world loads in each of its 250 ms turns, some
//! forty seconds of a world without its textures.
//! `VICTORY_TEXTURE_POOL=0` makes every one an allocation of its own again.
//!
//! A piece given back is used again only two seconds later (nothing the card
//! was told before that is still running: Windows resets a card that takes
//! that long over one submission), and by then no list names it (the owners' rule, as for allocations of their own): a
//! texture's piece comes back when the last reference to the host texture
//! goes (an object of ours the texture holds says so), a staging buffer's
//! when the last list that recorded its copy is reset.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

/// Bytes of a block; pieces of more than half of one are not pooled (with
/// blocks made ahead, [`ahead`]: of more than one).
const BLOCK: u64 = 64 << 20;
/// Blocks with nothing in them that are kept.
const SPARE: usize = 2;
/// The same while the card has room ([`room`]): 4 GiB of them.
const SPARE_KEPT: usize = 64;
/// The same of blocks of staging buffers, which are the PC's memory: run
/// crash-1 held 5.2 GB of them.
const STAGING_KEPT: usize = 8;
/// How many are kept now.
static SPARES: AtomicUsize = AtomicUsize::new(SPARE);

/// Sets how many empty blocks stay, by the card's memory (budget and use):
/// [`SPARE_KEPT`] below 85 % ([`crate::host::keep_freed`]), else [`SPARE`].
/// Also demotes idle texture spare blocks and performs fence-safe WDDM
/// `Evict()` on aged idle heaps outside the pool lock. Asked once a second.
pub(crate) fn room(memory: Option<[u64; 4]>) {
    let pressure = crate::residency::pressured();
    let limit = spares(memory, crate::host::keep_freed() && !pressure);
    SPARES.store(limit, Ordering::Relaxed);
    let tick = now();
    if !crate::residency_wddm::enabled() {
        // Strict default-0 baseline: untouched reclaim behavior and zero WDDM bookkeeping.
        if let Ok(mut pool) = TEXTURES.try_lock() {
            pool.reclaim(tick, limit, true);
        }
        if let Ok(mut pool) = STAGING.try_lock() {
            let staging_limit = if crate::residency::critical_commit() {
                0
            } else if pressure {
                1
            } else {
                limit
            };
            pool.reclaim(tick, staging_limit, true);
        }
        return;
    }

    let critical = crate::residency::critical_commit();
    let (started, finished) = crate::host::submissions();
    let idle_fence = started.saturating_add(crate::queue::queued());
    let host = crate::host::made_host();

    let mut priority_updates: Vec<(ID3D12Heap, crate::residency_wddm::ResidencyTier)> = Vec::new();
    let mut evict_candidates: Vec<(usize, ID3D12Heap)> = Vec::new();
    let mut dropped_evicted = 0usize;
    // A full pool can become idle while another subsystem needs the VRAM.
    // A busy allocator observes the new limit on its next allocation or tick.
    if let Ok(mut pool) = TEXTURES.try_lock() {
        dropped_evicted = pool.reclaim_with_fence(tick, limit, true, idle_fence);
        if host.is_some() {
            let vram_used_pct = memory.and_then(|[budget, used, ..]| {
                (budget > 0).then(|| ((u128::from(used) * 100) / u128::from(budget)) as u64)
            });
            let normal_hot_spares = ahead().min(limit).max(SPARE);
            pool.plan_idle_residency(
                IdleResidencyCtx {
                    now: tick,
                    completed_submission: finished,
                    vram_used_pct,
                    pressured: pressure,
                    critical,
                    normal_hot_spares,
                },
                &mut priority_updates,
                &mut evict_candidates,
            );
        }
    }
    for _ in 0..dropped_evicted {
        crate::residency_wddm::note_evicted_block_dropped();
    }

    if let Some(host) = host {
        for (heap, tier) in priority_updates {
            let _ = crate::residency_wddm::set_heap_priority(&host.device, &heap, tier);
        }
        if !evict_candidates.is_empty() {
            let mut outcomes = Vec::with_capacity(evict_candidates.len());
            for (idx, heap) in evict_candidates {
                let evicted = crate::residency_wddm::evict_heap(&host.device, &heap).is_ok();
                outcomes.push((idx, evicted));
            }
            let mut pool = lock(&TEXTURES);
            for (idx, evicted) in outcomes {
                if let Some(Some(block)) = pool.blocks.get_mut(idx)
                    && block.residency == BlockResidency::Evicting
                {
                    if evicted {
                        block.residency = BlockResidency::Evicted;
                        block.priority = crate::residency_wddm::ResidencyTier::Minimum;
                    } else {
                        block.residency = BlockResidency::Resident;
                        // `evict_heap` lowers priority to `Minimum` before attempting `Evict`;
                        // mirror that floor here so the next allocation always re-promotes to `Normal`.
                        block.priority = crate::residency_wddm::ResidencyTier::Minimum;
                    }
                }
            }
        }
    }

    if let Ok(mut pool) = STAGING.try_lock() {
        let staging_limit = if critical {
            0
        } else if pressure {
            1
        } else {
            limit
        };
        pool.reclaim_with_fence(tick, staging_limit, true, idle_fence);
    }
}

fn spares(memory: Option<[u64; 4]>, keep: bool) -> usize {
    spares_for(memory, keep, crate::resource::room_percent().saturating_sub(5))
}

fn spares_for(memory: Option<[u64; 4]>, keep: bool, target_pct: u64) -> usize {
    match memory {
        Some([budget, used, ..]) if keep && u128::from(used) * 100 < u128::from(budget) * u128::from(target_pct) => SPARE_KEPT,
        _ => SPARE,
    }
}

/// Milliseconds until a piece given back is used again. (It was three
/// frames, a tenth of a second while the title shows a loading screen at 30
/// a second, with up to 18 submissions in the air.)
const REST: u64 = 2500;

fn now() -> u64 {
    crate::host::now_ms()
}

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_POOL").is_none_or(|v| v != "0"))
}

/// How many blocks of textures are made ahead, so that one has been there
/// for a while when its first piece is taken (`VICTORY_RIPE_BLOCKS`, 12;
/// 0: none, and new blocks are used at once, as before).
///
/// A page fault of lists already on the card, in a texture made the same
/// tenth of a second as the first piece of a block made for it, lost the
/// device in runs n08 and n11 (2026-11-25): the title gives a slot that
/// those lists read by index a new texture. Why the card does not have the
/// new block's addresses yet is not known (n11 waited for `MakeResident`;
/// `indirect::tests::new_blocks_under_readers` never loses it). A block that
/// every submission begun before its making has left behind ([`Ripe`]) has
/// no such lists: pieces are cut from those first.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| std::env::var("VICTORY_RIPE_BLOCKS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(12))
}

/// Submissions begun and finished ([`crate::host::submissions`]), and how
/// many blocks to make ahead (0: every block is ripe at once).
#[derive(Clone, Copy)]
struct Ripe {
    started: u64,
    finished: u64,
    ahead: usize,
}

impl Ripe {
    const ALL: Ripe = Ripe { started: 0, finished: 0, ahead: 0 };

    fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        // Not more than are kept ([`room`]): they would go and be made again.
        Ripe { started, finished, ahead: ahead().min(SPARES.load(Ordering::Relaxed)) }
    }

    fn is<B>(self, block: &Block<B>) -> bool {
        self.ahead == 0 || block.made <= self.finished
    }
}

/// Pieces cut from blocks that were not ripe, for want of a ripe one.
static UNRIPE: AtomicU64 = AtomicU64::new(0);

/// The free ranges of a block: start and length, never two that touch.
struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    fn new(size: u64) -> Ranges {
        Ranges(BTreeMap::from([(0, size)]))
    }

    /// The lowest place for `len` bytes at a multiple of `align`.
    fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.0.iter().find_map(|(&start, &size)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + size).then_some((start, size, at))
        })?;
        self.0.remove(&start);
        if at > start {
            self.0.insert(start, at - start);
        }
        if at + len < start + size {
            self.0.insert(at + len, start + size - at - len);
        }
        Some(at)
    }

    fn give(&mut self, mut start: u64, mut len: u64) {
        if let Some((&before, &size)) = self.0.range(..start).next_back()
            && before + size == start
        {
            self.0.remove(&before);
            (start, len) = (before, len + size);
        }
        if let Some(size) = self.0.remove(&(start + len)) {
            len += size;
        }
        self.0.insert(start, len);
    }

    fn whole(&self, size: u64) -> bool {
        self.0.get(&0) == Some(&size)
    }
}

/// A part of a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    block: usize,
    start: u64,
    len: u64,
}

/// WDDM residency lifecycle state of a pooled block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockResidency {
    /// Currently resident (D3D12 residency refcount 1) and eligible for allocation.
    Resident,
    /// Reserved by the sweep thread for lock-free `Evict()`; excluded from allocation/reclaim.
    Evicting,
    /// Explicitly evicted via `ID3D12Device::Evict` (refcount 0); must `MakeResident` before use.
    Evicted,
    /// Reserved by an allocator thread for lock-free `MakeResident()`; excluded from others.
    Restoring,
}

#[derive(Clone, Copy)]
struct TakeReq {
    len: u64,
    align: u64,
    now: u64,
    ripe: Ripe,
    idle_fence: u64,
    allow_restore: bool,
}

#[derive(Clone, Copy)]
struct IdleResidencyCtx {
    now: u64,
    completed_submission: u64,
    vram_used_pct: Option<u64>,
    pressured: bool,
    critical: bool,
    normal_hot_spares: usize,
}

enum TakeOutcome<B> {
    Taken {
        piece: Piece,
        memory: B,
        again: bool,
        promote_normal: bool,
        dropped_evicted: usize,
    },
    Restore {
        block: usize,
        memory: B,
        dropped_evicted: usize,
    },
    None {
        dropped_evicted: usize,
    },
}

struct Block<B> {
    memory: B,
    free: Ranges,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
    /// When that was (ms).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    since: u64,
    /// Explicit WDDM residency state for fence-safe `Evict()` / `MakeResident()`.
    residency: BlockResidency,
    /// Last applied WDDM residency priority tier.
    priority: crate::residency_wddm::ResidencyTier,
    /// Millisecond timestamp when this block last became completely empty (`whole(BLOCK)`).
    idle_since_ms: Option<u64>,
    /// Submission fence number that must complete before `Evict()` is legal.
    idle_fence: u64,
}

struct Pool<B> {
    /// `None`: a block let go, its place to be used again.
    blocks: Vec<Option<Block<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty blocks it was last told to keep, and the most it keeps.
    spare: usize,
    most: usize,
    retry_growth_at: u64,
}

impl<B: Clone> Pool<B> {
    const fn new(most: usize) -> Pool<B> {
        Pool { blocks: Vec::new(), resting: VecDeque::new(), spare: 0, most, retry_growth_at: 0 }
    }

    /// A piece of `len` bytes and the memory it lies in, and whether that
    /// part was ever used before; `make` makes a block when none has room.
    fn take(&mut self, len: u64, align: u64, now: u64, make: impl FnMut() -> Option<B>) -> Option<(Piece, B, bool)> {
        self.take_ripe(len, align, now, Ripe::ALL, make)
    }

    /// The same from a block that is ripe, if one has room; `make` makes one
    /// ahead too, when few are empty. Baseline default path (no WDDM residency).
    fn take_ripe(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return None;
        }
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
        let mut found = None;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if let Some(block) = block
                && ripe.is(block)
                && let Some(start) = block.free.take(len, align)
            {
                found = Some((i, start));
                break;
            }
        }
        if found.is_none() && ripe.ahead > 0 {
            // The one that has been there the longest.
            let mut unripe: Vec<(u64, usize)> =
                self.blocks.iter().enumerate().filter_map(|(i, b)| Some((b.as_ref()?.made, i))).collect();
            unripe.sort_unstable();
            found = unripe.into_iter().find_map(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        }
        // A failed 64 MiB block must not trigger another driver call for each
        // small texture. Existing free ranges remain usable during backoff.
        let (i, start) = match found {
            Some(found) => found,
            None => {
                let memory = self.grow(now, &mut make)?;
                let i = self.add_at(memory, ripe.started, ripe.started, now);
                (i, self.blocks[i].as_mut()?.free.take(len, align)?)
            }
        };
        if ripe.ahead > 0 {
            if !ripe.is(self.blocks[i].as_ref()?) {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
                }
            }
            let empty = self.blocks.iter().flatten().filter(|b| b.free.whole(BLOCK)).count();
            if empty < ripe.ahead
                && let Some(memory) = self.grow(now, &mut make)
            {
                self.add_at(memory, ripe.started, ripe.started, now);
            }
        }
        let block = self.blocks[i].as_mut()?;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        Some((Piece { block: i, start, len }, block.memory.clone(), again))
    }

    /// Like [`take_ripe`], skipping non-resident blocks and optionally yielding an
    /// `Evicted` block reserved in [`BlockResidency::Restoring`] so the caller can
    /// invoke `MakeResident` outside the pool lock before growing a brand-new heap.
    fn take_ripe_or_restore(
        &mut self,
        req: TakeReq,
        mut make: impl FnMut() -> Option<B>,
    ) -> TakeOutcome<B> {
        let TakeReq { len, align, now, ripe, idle_fence, allow_restore } = req;
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return TakeOutcome::None { dropped_evicted: 0 };
        }
        let dropped_evicted = self.reclaim_with_fence(now, SPARES.load(Ordering::Relaxed), false, idle_fence);
        let mut found = None;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if let Some(block) = block
                && block.residency == BlockResidency::Resident
                && ripe.is(block)
                && let Some(start) = block.free.take(len, align)
            {
                found = Some((i, start));
                break;
            }
        }
        if found.is_none() && ripe.ahead > 0 {
            // The resident block that has been there the longest.
            let mut unripe: Vec<(u64, usize)> = self
                .blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    let b = b.as_ref()?;
                    (b.residency == BlockResidency::Resident).then_some((b.made, i))
                })
                .collect();
            unripe.sort_unstable();
            found = unripe.into_iter().find_map(|(_, i)| {
                let b = self.blocks[i].as_mut()?;
                (b.residency == BlockResidency::Resident)
                    .then(|| b.free.take(len, align).map(|start| (i, start)))
                    .flatten()
            });
        }
        if found.is_none()
            && allow_restore
            && let Some((idx, block)) = self.blocks.iter_mut().enumerate().find(|(_, b)| {
                b.as_ref().is_some_and(|b| b.residency == BlockResidency::Evicted && b.free.whole(BLOCK))
            })
        {
            let block = block.as_mut().unwrap();
            block.residency = BlockResidency::Restoring;
            return TakeOutcome::Restore {
                block: idx,
                memory: block.memory.clone(),
                dropped_evicted,
            };
        }
        // A failed 64 MiB block must not trigger another driver call for each
        // small texture. Existing free ranges remain usable during backoff.
        let (i, start) = match found {
            Some(found) => found,
            None => {
                let Some(memory) = self.grow(now, &mut make) else {
                    return TakeOutcome::None { dropped_evicted };
                };
                let i = self.add_at(memory, ripe.started, idle_fence, now);
                let Some(start) = self.blocks[i].as_mut().and_then(|b| b.free.take(len, align)) else {
                    return TakeOutcome::None { dropped_evicted };
                };
                (i, start)
            }
        };
        if ripe.ahead > 0 {
            if self.blocks[i].as_ref().is_some_and(|b| !ripe.is(b)) {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!("textures cut from blocks made while lists ran that still run: {n}");
                }
            }
            // Count both resident and evicted empty blocks so evicting aged idle
            // spares does not trigger unbounded replacement `CreateHeap` churn;
            // evicted spares are restored via `MakeResident` on demand instead.
            let empty = self
                .blocks
                .iter()
                .flatten()
                .filter(|b| {
                    matches!(b.residency, BlockResidency::Resident | BlockResidency::Evicted)
                        && b.free.whole(BLOCK)
                })
                .count();
            if empty < ripe.ahead
                && let Some(memory) = self.grow(now, &mut make)
            {
                self.add_at(memory, ripe.started, idle_fence, now);
            }
        }
        let Some(block) = self.blocks[i].as_mut() else {
            return TakeOutcome::None { dropped_evicted };
        };
        block.idle_since_ms = None;
        let promote_normal = block.priority < crate::residency_wddm::ResidencyTier::Normal;
        block.priority = crate::residency_wddm::ResidencyTier::Normal;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        TakeOutcome::Taken {
            piece: Piece { block: i, start, len },
            memory: block.memory.clone(),
            again,
            promote_normal,
            dropped_evicted,
        }
    }

    /// Complete a lock-free `MakeResident` on block `idx` and immediately carve
    /// `(len, align)` from it while holding the pool lock.
    fn finish_restore_and_take(
        &mut self,
        idx: usize,
        req: TakeReq,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        let TakeReq { len, align, now, ripe, idle_fence, .. } = req;
        let block = self.blocks.get_mut(idx)?.as_mut()?;
        if block.residency != BlockResidency::Restoring {
            return None;
        }
        block.residency = BlockResidency::Resident;
        block.priority = crate::residency_wddm::ResidencyTier::Normal;
        block.made = ripe.started;
        block.idle_since_ms = Some(now);
        block.idle_fence = idle_fence.max(ripe.started);
        let start = block.free.take(len, align)?;
        block.idle_since_ms = None;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        let memory = block.memory.clone();
        if ripe.ahead > 0 {
            let empty = self
                .blocks
                .iter()
                .flatten()
                .filter(|b| {
                    matches!(b.residency, BlockResidency::Resident | BlockResidency::Evicted)
                        && b.free.whole(BLOCK)
                })
                .count();
            if empty < ripe.ahead
                && let Some(extra) = self.grow(now, &mut make)
            {
                self.add_at(extra, ripe.started, idle_fence, now);
            }
        }
        Some((Piece { block: idx, start, len }, memory, again))
    }

    fn cancel_restore(&mut self, idx: usize) {
        if let Some(Some(block)) = self.blocks.get_mut(idx)
            && block.residency == BlockResidency::Restoring
        {
            block.residency = BlockResidency::Evicted;
        }
    }

    fn grow(&mut self, now: u64, make: &mut impl FnMut() -> Option<B>) -> Option<B> {
        if now < self.retry_growth_at {
            return None;
        }
        let block = make();
        if block.is_none() {
            self.retry_growth_at = now.saturating_add(500);
        }
        block
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn add(&mut self, memory: B, made: u64) -> usize {
        self.add_at(memory, made, made, 0)
    }

    fn add_at(&mut self, memory: B, made: u64, idle_fence: u64, now: u64) -> usize {
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[i] = Some(Block {
            memory,
            free: Ranges::new(BLOCK),
            used_to: 0,
            made,
            since: crate::host::now_ms(),
            residency: BlockResidency::Resident,
            priority: crate::residency_wddm::ResidencyTier::Normal,
            idle_since_ms: Some(now),
            idle_fence: idle_fence.max(made),
        });
        i
    }

    fn give(&mut self, piece: Piece, now: u64) {
        self.resting.push_back((now + REST, piece));
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// blocks past [`SPARE`].
    #[cfg_attr(not(test), allow(dead_code))]
    fn rested(&mut self, now: u64) {
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
    }

    /// Baseline reclaim: makes rested pieces free and drops excess empty blocks
    /// past `limit` in ascending block order with no residency/fence bookkeeping.
    fn reclaim(&mut self, now: u64, limit: usize, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                any |= block.free.whole(BLOCK);
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        let previous = std::mem::replace(&mut self.spare, most);
        if sweep || any || most < previous {
            let mut spare = 0;
            for block in &mut self.blocks {
                if block.as_ref().is_some_and(|b| b.free.whole(BLOCK)) {
                    spare += 1;
                    if spare > most {
                        *block = None;
                    }
                }
            }
        }
    }

    /// Reclaim rested pieces and trim excess empty blocks past `limit`.
    /// Never drops blocks mid-transition (`Evicting`/`Restoring`), and prefers
    /// dropping already-`Evicted` empty blocks before hot `Resident` spares.
    /// Returns the number of `Evicted` blocks dropped.
    fn reclaim_with_fence(&mut self, now: u64, limit: usize, sweep: bool, idle_fence: u64) -> usize {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                if block.free.whole(BLOCK) {
                    if block.idle_since_ms.is_none() {
                        block.idle_since_ms = Some(now);
                        block.idle_fence = idle_fence.max(block.made);
                    }
                    any = true;
                }
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        let previous = std::mem::replace(&mut self.spare, most);
        let mut dropped_evicted = 0usize;
        if sweep || any || most < previous {
            let mut evicted_empty = Vec::new();
            let mut resident_empty = Vec::new();
            for (i, block) in self.blocks.iter().enumerate() {
                if let Some(b) = block
                    && b.free.whole(BLOCK)
                {
                    match b.residency {
                        BlockResidency::Evicted => evicted_empty.push(i),
                        BlockResidency::Resident => resident_empty.push(i),
                        BlockResidency::Evicting | BlockResidency::Restoring => {}
                    }
                }
            }
            let total_empty = evicted_empty.len() + resident_empty.len();
            if total_empty > most {
                let mut to_drop = total_empty - most;
                for i in evicted_empty {
                    if to_drop == 0 {
                        break;
                    }
                    self.blocks[i] = None;
                    dropped_evicted += 1;
                    to_drop -= 1;
                }
                for i in resident_empty {
                    if to_drop == 0 {
                        break;
                    }
                    self.blocks[i] = None;
                    to_drop -= 1;
                }
            }
        }
        dropped_evicted
    }

    /// Plan WDDM priority demotions (`LOW`/`MINIMUM`) and fence-safe `Evict()`
    /// transitions for resident empty blocks. Transitions eviction targets to
    /// [`BlockResidency::Evicting`] so the caller can invoke D3D12 outside the lock.
    fn plan_idle_residency(
        &mut self,
        ctx: IdleResidencyCtx,
        priority_updates: &mut Vec<(B, crate::residency_wddm::ResidencyTier)>,
        evict_candidates: &mut Vec<(usize, B)>,
    ) {
        let IdleResidencyCtx {
            now,
            completed_submission,
            vram_used_pct,
            pressured,
            critical,
            normal_hot_spares,
        } = ctx;
        let mut resident_idle: Vec<(u64, usize)> = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let b = b.as_ref()?;
                (b.residency == BlockResidency::Resident && b.free.whole(BLOCK))
                    .then_some((b.idle_since_ms.unwrap_or(now), i))
            })
            .collect();
        // Newest-idle first so hot spares receive ranks 0..hot_spares_to_keep
        // and the oldest idle blocks are evicted first.
        resident_idle.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        for (rank, (idle_since, idx)) in resident_idle.into_iter().enumerate() {
            let Some(block) = self.blocks[idx].as_mut() else {
                continue;
            };
            let age_ms = now.saturating_sub(idle_since);
            let eff_fence = block.idle_fence.max(block.made);
            match crate::residency_wddm::classify_idle_block(crate::residency_wddm::IdleBlockQuery {
                age_ms,
                idle_fence: eff_fence,
                completed_submission,
                vram_used_pct,
                pressured,
                critical,
                normal_hot_spares,
                resident_spare_rank: rank,
            }) {
                crate::residency_wddm::IdleBlockAction::KeepResidentLow => {
                    if block.priority != crate::residency_wddm::ResidencyTier::Low {
                        block.priority = crate::residency_wddm::ResidencyTier::Low;
                        priority_updates.push((
                            block.memory.clone(),
                            crate::residency_wddm::ResidencyTier::Low,
                        ));
                    }
                }
                crate::residency_wddm::IdleBlockAction::DemoteMinimum => {
                    if block.priority != crate::residency_wddm::ResidencyTier::Minimum {
                        block.priority = crate::residency_wddm::ResidencyTier::Minimum;
                        priority_updates.push((
                            block.memory.clone(),
                            crate::residency_wddm::ResidencyTier::Minimum,
                        ));
                    }
                }
                crate::residency_wddm::IdleBlockAction::Evict => {
                    block.residency = BlockResidency::Evicting;
                    evict_candidates.push((idx, block.memory.clone()));
                }
            }
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    fn resident_blocks(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .filter(|b| b.residency == BlockResidency::Resident)
            .count()
    }
}

/// A mapped upload buffer's memory.
#[derive(Clone)]
struct Upload(ID3D12Resource, *mut u8);
// SAFETY: the mapping is written by the owners of its pieces, each its own.
unsafe impl Send for Upload {}

static TEXTURES: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SPARE_KEPT));
static STAGING: Mutex<Pool<Upload>> = Mutex::new(Pool::new(STAGING_KEPT));

fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bytes of the pieces textures hold now.
static HELD: AtomicU64 = AtomicU64::new(0);

/// Bytes of the resident texture blocks no texture holds: the card's memory in
/// use by us that new textures take first (evicted heaps are excluded so VRAM
/// budget math does not double-subtract them).
pub(crate) fn texture_room() -> u64 {
    (lock(&TEXTURES).resident_blocks() as u64 * BLOCK).saturating_sub(HELD.load(Ordering::Relaxed))
}

/// For the memory line: blocks of textures and of staging buffers, in MiB.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 2] {
    [lock(&TEXTURES).blocks() as u64 * (BLOCK >> 20), lock(&STAGING).blocks() as u64 * (BLOCK >> 20)]
}

/// Part of a block of mapped upload memory, back in the pool when dropped.
pub(crate) struct StagingPiece(Piece);

impl Drop for StagingPiece {
    fn drop(&mut self) {
        lock(&STAGING).give(self.0, now());
    }
}

/// `bytes` of upload memory, all zeros: the buffer it lies in, where in
/// that buffer, and its mapping. `None`: not pooled.
pub(crate) fn staging(host: &crate::host::Host, bytes: u64) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
    if !on() {
        return None;
    }
    // Copies from a buffer begin at multiples of 512.
    let bytes = bytes.next_multiple_of(512);
    let (piece, memory, again) = lock(&STAGING).take(bytes, 512, now(), || {
        // Only gate speculative bindless prefetch on staging-pool growth
        // headroom; table-bound textures and decoration bakes are sampled
        // immediately and would otherwise read uninitialized/reused VRAM.
        let speculative = crate::texture::PICKING.get().is_some() && !crate::texture::BAKING.get();
        if speculative && !crate::residency::allow_staging_growth(false) {
            return None;
        }
        let (buffer, data) = host.upload_buffer(BLOCK)?;
        crate::host::name(&buffer, || "staging of textures (a block)".into());
        Some(Upload(buffer, data))
    })?;
    let owned = StagingPiece(piece);
    // SAFETY: the piece lies in the block's mapping.
    let data = unsafe { memory.1.add(piece.start as usize) };
    if again {
        // What is not detiled into it (an unmapped texture) uploads zeros,
        // as from a buffer of its own.
        // SAFETY: the piece is ours alone.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    }
    Some((owned, memory.0, piece.start, data))
}

/// An object the host texture holds and releases when it goes: its piece
/// is free then.
#[repr(C)]
struct Reclaim {
    methods: &'static [*const (); 3],
    references: AtomicU32,
    piece: Piece,
}

struct Methods([*const (); 3]);
// SAFETY: function addresses only.
unsafe impl Sync for Methods {}

static RECLAIM: Methods = Methods([query as *const (), add_ref as *const (), release as *const ()]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    // SAFETY: COM's arguments.
    unsafe {
        if out.is_null() {
            return HRESULT(0x8000_4003_u32 as i32);
        }
        if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
            add_ref(this);
            *out = this;
            return HRESULT(0);
        }
        *out = std::ptr::null_mut();
    }
    E_NOINTERFACE
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: one of ours, alive while a reference to it is.
    unsafe { (*this.cast::<Reclaim>()).references.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last reference frees the box it was made as.
    unsafe {
        let left = (*this.cast::<Reclaim>()).references.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            let gone = Box::from_raw(this.cast::<Reclaim>());
            HELD.fetch_sub(gone.piece.len, Ordering::Relaxed);
            lock(&TEXTURES).give(gone.piece, now());
        }
        left
    }
}

/// Whether textures of 64 KiB and less are placed at multiples of 4 KiB in
/// their block, where the host allows it, rather than at multiples of 64 KiB
/// as every other texture: on any card but an AMD one by default;
/// `VICTORY_POOL_SMALL=1` or `=0` picks either.
///
/// On an RTX 4090 (Series S, 30 minutes) the 4 KiB placement held 3.2 GB less
/// of the PC's memory and 3.4 GB less of the card's than 64 KiB, with no
/// device loss; the losses below were on an RX 7900 XTX.
///
/// The two device losses of the night of 2026-11-25 with the pool in the
/// report (runs steps-ab2 and start-3) faulted in the same kind of texture:
/// 256x256, block-compressed, 7 levels, placed at a 4 KiB multiple that is no
/// 64 KiB one (offsets 0x406000 and 0x3fe6000 of their blocks), never used
/// before, read by the card the moment its slot was made; the descriptor's
/// base of both lies 0xa000 into the piece, on a 64 KiB multiple, and its
/// first level of 32 KiB from there ends 0x2000 past the piece. Why the card
/// faulted there is not proven; textures at 64 KiB multiples have none of
/// that, for some memory (at most 60 KiB a small texture).
fn small_placement() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_POOL_SMALL") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_POOL_SMALL") {
        Some(v) => v != "0",
        None => !crate::host::host().is_some_and(|h| crate::indirect::amd(&h.device)),
    })
}

/// Diagnostics: whether the card reads a pooled texture inside the piece it
/// was placed in. A block's address is taken as the lowest of its textures'
/// descriptor bases less their pieces' starts (a texture's first level may
/// lie past its piece's start, never before it); a texture whose first level
/// the card would read from past its piece's end is said loudly.
#[cfg(feature = "diagnostics")]
fn where_the_card_reads(heap: &ID3D12Heap, texture: &ID3D12Resource, piece: Piece) {
    static BLOCKS: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());
    static SAID: AtomicU64 = AtomicU64::new(0);
    let Some(base) = crate::book::address(texture) else { return };
    let Some(seen) = base.checked_sub(piece.start) else { return };
    let raw = heap.as_raw() as usize;
    let block = {
        let mut blocks = BLOCKS.lock().unwrap_or_else(|e| e.into_inner());
        match blocks.iter_mut().find(|b| b.0 == raw) {
            Some(b) => {
                b.1 = b.1.min(seen);
                b.1
            }
            None => {
                if blocks.len() >= 4096 {
                    blocks.remove(0);
                }
                blocks.push((raw, seen));
                seen
            }
        }
    };
    let into = base - block - piece.start;
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    // The first level's bytes, as a copy would have them.
    let mut level = 0u64;
    if let Some(host) = crate::host::made_host() {
        // SAFETY: plain query of a valid description.
        unsafe { host.device.GetCopyableFootprints(&desc, 0, 1, 0, None, None, None, Some(&raw mut level)) };
    }
    if into + level > piece.len {
        let n = SAID.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 50 || n.is_power_of_two() {
            // (Debug only: it assumes a texture's address is its block's
            // start and its piece's offset, which this driver does not
            // keep: 56 false alarms in run pool-1.)
            tracing::debug!(
                times = n,
                base = format_args!("{base:#x}"),
                piece_start = format_args!("{:#x}", piece.start),
                piece_len = format_args!("{:#x}", piece.len),
                into = format_args!("{into:#x}"),
                first_level_bytes = format_args!("{level:#x}"),
                width = desc.Width,
                height = desc.Height,
                levels = desc.MipLevels,
                format = desc.Format.0,
                "POOL: the card reads a pooled texture's first level past the end of the piece it was placed in"
            );
        }
    }
    // Every placement offset and where the first level lies, a few of them.
    static SHOWN: AtomicU64 = AtomicU64::new(0);
    if into > 0 && SHOWN.fetch_add(1, Ordering::Relaxed) < 40 {
        tracing::info!(
            into = format_args!("{into:#x}"),
            piece_start = format_args!("{:#x}", piece.start),
            piece_len = format_args!("{:#x}", piece.len),
            width = desc.Width,
            height = desc.Height,
            levels = desc.MipLevels,
            format = desc.Format.0,
            alignment = desc.Alignment,
            "pool: a texture's first level lies past its piece's start"
        );
    }
}

/// Textures made in the last second ([`fresh`]), and when.
static RECENT: Mutex<VecDeque<(usize, u64)>> = Mutex::new(VecDeque::new());

/// Whether `texture` was made in a block in the last second.
pub(crate) fn fresh(texture: &ID3D12Resource) -> bool {
    let raw = texture.as_raw() as usize;
    let since = now().saturating_sub(1000);
    RECENT.lock().unwrap_or_else(|e| e.into_inner()).iter().rev().take_while(|r| r.1 >= since).any(|r| r.0 == raw)
}

fn made_now(texture: &ID3D12Resource) {
    let now = now();
    let raw = texture.as_raw() as usize;
    {
        let mut recent = RECENT.lock().unwrap_or_else(|e| e.into_inner());
        while recent.front().is_some_and(|r| r.1 + 1000 < now) {
            recent.pop_front();
        }
        recent.push_back((raw, now));
    }
    let begun = crate::host::submissions().0;
    NEWEST_BEGUN.fetch_max(begun, Ordering::Relaxed);
    let mut unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    let finished = crate::host::submissions().1;
    // Those every submission begun before them has left behind are no
    // one's concern any more (and a raw pointer may come again).
    unseen.retain(|_, b| *b > finished);
    unseen.insert(raw, begun);
}

/// Pooled textures made while submissions begun before them may still run:
/// the submissions begun when each was made.
static UNSEEN: std::sync::LazyLock<Mutex<std::collections::HashMap<usize, u64>>> = std::sync::LazyLock::new(Default::default);
/// The newest of those numbers (nothing to look up once it has finished).
static NEWEST_BEGUN: AtomicU64 = AtomicU64::new(0);

/// Whether a texture was made in a block while a submission begun before
/// it may still run ([`begun_before`]).
pub(crate) fn any_unseen() -> bool {
    NEWEST_BEGUN.load(Ordering::Relaxed) > crate::host::submissions().1
}

/// The submissions begun when `texture` was made in a block, if one of
/// those may still be on the card: lists already sent then must not find it
/// in a slot they read by index ([`crate::descriptor::host::show_new_later`]).
/// Every device loss with the pool in the report (runs n08, n11, n15,
/// steps-ab2, start-3 and the user's play-1125-2107) was a page fault at the
/// first byte of a texture made that very millisecond, its title slot
/// changed "at once" while lists sent before it read that slot by index.
pub(crate) fn begun_before(texture: &ID3D12Resource) -> Option<u64> {
    if !any_unseen() {
        return None;
    }
    let finished = crate::host::submissions().1;
    let raw = texture.as_raw() as usize;
    let unseen = UNSEEN.lock().unwrap_or_else(|e| e.into_inner());
    unseen.get(&raw).copied().filter(|&begun| begun > finished)
}

/// Makes `texture` give `piece` back when the last reference to it goes.
fn give_back_with(texture: &ID3D12Resource, piece: Piece) {
    const KEY: GUID = GUID::from_u128(0x6f0e2f0c_5a59_4b1e_9a3f_7f4d2c9be713);
    let made = Box::into_raw(Box::new(Reclaim { methods: &RECLAIM.0, references: AtomicU32::new(1), piece }));
    // SAFETY: a COM object of ours with one reference, which this takes and
    // drops at the end; the texture takes its own.
    unsafe {
        let object = IUnknown::from_raw(made.cast());
        if texture.SetPrivateDataInterface(&KEY, &object).is_err() {
            // Never given back: the piece stays taken, nothing is reused
            // under a texture that may be alive.
            std::mem::forget(object);
        }
    }
}

/// A texture of `desc` (no target, depth or unordered access) in a block;
/// `None`: not pooled, or the host refused.
///
/// # Safety
/// `desc` is a valid description.
pub(crate) unsafe fn texture(
    device: &ID3D12Device,
    desc: *const D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
    first_attempt: bool,
) -> Option<ID3D12Resource> {
    if !on() {
        return None;
    }
    // Small textures lie at multiples of 4 KiB where the host allows it
    // (the top mip is 64 KiB at most) and [`small_placement`] says so: 64 KiB
    // each otherwise.
    // SAFETY: the caller's description.
    let mut d = unsafe { *desc };
    d.Alignment = if small_placement() { u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT) } else { 0 };
    // SAFETY: plain queries.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != d.Alignment {
        d.Alignment = 0;
        info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    }
    if info.SizeInBytes == u64::MAX {
        return None;
    }
    let tick = now();
    let ripe = Ripe::now();
    let wddm_residency = crate::residency_wddm::enabled();
    let idle_fence = if wddm_residency {
        ripe.started.saturating_add(crate::queue::queued())
    } else {
        0
    };
    let mut make_heap = || {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: BLOCK,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | crate::host::not_resident(),
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
            // The pool still backs off its own growth when this is the
            // texture's last attempt, without renewing the global pause.
            if first_attempt {
                crate::residency::allocation_error(&e);
            }
            tracing::warn!(error = %e, "no block of texture memory");
        }
        if let Some(h) = &heap {
            if let Err(e) = crate::host::resident(device, h).and_then(|()| crate::host::mapped(device)) {
                tracing::warn!(error = %e, "a block of texture memory is not resident");
                return None;
            }
            if wddm_residency {
                let _ = crate::residency_wddm::set_heap_priority(
                    device,
                    h,
                    crate::residency_wddm::ResidencyTier::Normal,
                );
            }
        }
        heap
    };

    let align = info.Alignment.max(1);
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (piece, heap, again) = if !wddm_residency {
        lock(&TEXTURES).take_ripe(info.SizeInBytes, align, tick, ripe, &mut make_heap)?
    } else {
        let outcome = lock(&TEXTURES).take_ripe_or_restore(
            TakeReq { len: info.SizeInBytes, align, now: tick, ripe, idle_fence, allow_restore: true },
            &mut make_heap,
        );
        match outcome {
            TakeOutcome::Taken { piece, memory, again, promote_normal, dropped_evicted } => {
                for _ in 0..dropped_evicted {
                    crate::residency_wddm::note_evicted_block_dropped();
                }
                if promote_normal {
                    let _ = crate::residency_wddm::set_heap_priority(
                        device,
                        &memory,
                        crate::residency_wddm::ResidencyTier::Normal,
                    );
                }
                (piece, memory, again)
            }
            TakeOutcome::Restore { block: idx, memory: evicted_heap, dropped_evicted } => {
                for _ in 0..dropped_evicted {
                    crate::residency_wddm::note_evicted_block_dropped();
                }
                // Restore the evicted heap OUTSIDE the TEXTURES lock so concurrent
                // returns and queries never stall behind MakeResident / mapped().
                match crate::residency_wddm::make_heap_resident(device, &evicted_heap) {
                    Ok(()) => {
                        let ripe_now = Ripe::now();
                        let restore_fence = ripe_now.started.saturating_add(crate::queue::queued());
                        lock(&TEXTURES).finish_restore_and_take(
                            idx,
                            TakeReq {
                                len: info.SizeInBytes,
                                align,
                                now: now(),
                                ripe: ripe_now,
                                idle_fence: restore_fence,
                                allow_restore: false,
                            },
                            &mut make_heap,
                        )?
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, block = idx, "could not make evicted texture pool block resident");
                        let fallback_ripe = Ripe::now();
                        let fallback_fence = fallback_ripe.started.saturating_add(crate::queue::queued());
                        let mut pool = lock(&TEXTURES);
                        pool.cancel_restore(idx);
                        match pool.take_ripe_or_restore(
                            TakeReq {
                                len: info.SizeInBytes,
                                align,
                                now: now(),
                                ripe: fallback_ripe,
                                idle_fence: fallback_fence,
                                allow_restore: false,
                            },
                            &mut make_heap,
                        ) {
                            TakeOutcome::Taken { piece, memory, again, promote_normal, dropped_evicted } => {
                                drop(pool);
                                for _ in 0..dropped_evicted {
                                    crate::residency_wddm::note_evicted_block_dropped();
                                }
                                if promote_normal {
                                    let _ = crate::residency_wddm::set_heap_priority(
                                        device,
                                        &memory,
                                        crate::residency_wddm::ResidencyTier::Normal,
                                    );
                                }
                                (piece, memory, again)
                            }
                            TakeOutcome::Restore { dropped_evicted, .. }
                            | TakeOutcome::None { dropped_evicted } => {
                                drop(pool);
                                for _ in 0..dropped_evicted {
                                    crate::residency_wddm::note_evicted_block_dropped();
                                }
                                return None;
                            }
                        }
                    }
                }
            }
            TakeOutcome::None { dropped_evicted } => {
                for _ in 0..dropped_evicted {
                    crate::residency_wddm::note_evicted_block_dropped();
                }
                return None;
            }
        }
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: the piece is no one else's; the heap lives while a piece of
    // it is taken.
    let result = unsafe { device.CreatePlacedResource(&heap, piece.start, &d, state, None, &mut made) };
    match (result, made) {
        (Ok(()), Some(made)) => {
            HELD.fetch_add(piece.len, Ordering::Relaxed);
            give_back_with(&made, piece);
            made_now(&made);
            #[cfg(feature = "diagnostics")]
            {
                crate::book::made(&made, || "a texture in a block".into());
                where_the_card_reads(&heap, &made, piece);
                crate::book::block(&heap, &made, piece.start, BLOCK);
                crate::book::note(made.as_raw() as usize, || {
                    let block = lock(&TEXTURES).blocks.get(piece.block).and_then(Option::as_ref).map(|b| (b.made, b.since));
                    let (submission, since) = block.unwrap_or_default();
                    format!(
                        "; piece {:#x}+{:#x} of block {} {:#x} made {:.1} s before it at submission {submission}, used before: {again}, made at submissions {:?}",
                        piece.start,
                        piece.len,
                        piece.block,
                        heap.as_raw() as usize,
                        now().saturating_sub(since) as f64 / 1e3,
                        crate::host::submissions()
                    )
                });
            }
            Some(made)
        }
        _ => {
            lock(&TEXTURES).give(piece, now());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_cut_aligned_and_join_again() {
        let mut r = Ranges::new(1024);
        assert_eq!(r.take(100, 64), Some(0));
        assert_eq!(r.take(100, 64), Some(128));
        assert_eq!(r.take(28, 4), Some(100));
        assert_eq!(r.take(2000, 4), None);
        r.give(128, 100);
        r.give(0, 100);
        assert!(!r.whole(1024));
        r.give(100, 28);
        assert!(r.whole(1024));
    }

    #[test]
    fn empty_blocks_stay_while_the_card_has_room() {
        assert_eq!(spares(Some([100, 84, 0, 0]), true), SPARE_KEPT);
        assert_eq!(spares(Some([100, 85, 0, 0]), true), SPARE);
        assert_eq!(spares(Some([100, 10, 0, 0]), false), SPARE);
        assert_eq!(spares(None, true), SPARE);
    }

    #[test]
    fn a_failed_block_is_not_retried_for_each_small_upload() {
        let mut pool: Pool<u32> = Pool::new(2);
        let calls = std::cell::Cell::new(0);
        let refuse = || { calls.set(calls.get() + 1); None };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pool.take(size, 512, 12, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        assert!(pool.take(24080, 512, 1012, || Some(1)).is_some());
    }

    #[test]
    fn budget_tick_reclaims_idle_blocks_without_reusing_live_or_resting_pieces() {
        let mut pool = Pool::new(STAGING_KEPT);
        let (held, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (resting, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(2)).unwrap();
        pool.give(resting, 0);
        for i in 0..12 {
            pool.add(i, 0);
        }
        pool.reclaim(0, SPARE_KEPT, true);
        assert_eq!(pool.blocks(), STAGING_KEPT + 1);
        pool.reclaim(REST - 1, SPARE, true);
        assert_eq!(pool.blocks(), SPARE + 1);
        assert_eq!(pool.resting.len(), 1);
        pool.reclaim(REST, 0, true);
        assert_eq!(pool.blocks(), 1);
        assert!(pool.resting.is_empty());
        pool.give(held, REST);
        pool.reclaim(2 * REST, 0, true);
        assert_eq!(pool.blocks(), 0);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pool.take(65536 * (1 + i % 7), 65536, 0, make).unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.block != p.block || p.start + p.len <= q.start || q.start + q.len <= p.start));
            taken.push(p);
        }
        let blocks = pool.blocks();
        assert!(blocks >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pool.give(first, 10);
        let (p, _, _) = pool.take(first.len, 65536, 10 + REST - 1, make).unwrap();
        assert_ne!((p.block, p.start), (first.block, first.start));
        let (q, _, again) = pool.take(first.len, 65536, 10 + REST, make).unwrap();
        assert_eq!((q.block, q.start, again), (first.block, first.start, true));
        // Everything back: the empty blocks past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pool.give(p, 20);
        }
        pool.rested(20 + REST);
        assert_eq!(pool.blocks(), blocks.min(SPARE));
        // Too big for a block, and a block that cannot be made.
        assert!(pool.take(BLOCK, 65536, 30, make).is_none());
        let mut none: Pool<u32> = Pool::new(SPARE);
        assert!(none.take(65536, 65536, 0, || None).is_none());
    }

    #[test]
    fn pieces_come_from_blocks_older_than_the_lists_that_run() {
        let mut pool: Pool<u64> = Pool::new(SPARE_KEPT);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let at = |started, finished| Ripe { started, finished, ahead: 2 };
        let take = |pool: &mut Pool<u64>, ripe| pool.take_ripe(BLOCK / 2, 65536, 0, ripe, make).unwrap().1;
        // Nothing runs: at once, and two are made ahead, one a time.
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs.
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        // None is ripe: the oldest; it is ripe once 5 has finished.
        assert_eq!(take(&mut pool, at(7, 4)), 4);
        assert_eq!(take(&mut pool, at(7, 5)), 4);
        // A whole block's worth too.
        assert_eq!(pool.take_ripe(BLOCK, 65536, 0, at(8, 7), make).unwrap().1, 5);
        // Off: the first with room, whatever its age, none ahead.
        let n = made.get();
        assert_eq!(pool.take(BLOCK / 2, 65536, 0, make).unwrap().1, 6);
        assert_eq!(made.get(), n);
        assert!(pool.take(BLOCK, 65536, 0, make).is_none());
    }

    /// A texture placed in a heap holds no reference to it that its count
    /// shows (this host, 2026-11-25: 1 before placing, 1 with the texture):
    /// the pool must not count on one, and does not (a block goes only with
    /// every piece back and rested, [`Pool::rested`]). Said here, and the
    /// count is the same again once the texture went.
    #[test]
    fn a_placed_texture_holds_its_heap() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: 1 << 20,
            Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        unsafe { host.device.CreateHeap(&desc, &mut heap) }.unwrap();
        let heap = heap.unwrap();
        let before = crate::host::references(&heap);
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a live heap, a valid description.
        unsafe { host.device.CreatePlacedResource(&heap, 0, &texture_desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made) }.unwrap();
        let made = made.unwrap();
        let held = crate::host::references(&heap);
        println!("heap references: {before} before placing, {held} with a texture in it");
        assert!(held >= before);
        drop(made);
        assert_eq!(crate::host::references(&heap), before);
    }

    /// A pooled texture's piece comes back when its last reference goes, not
    /// before; its staging piece begins as zeros.
    #[test]
    fn a_texture_gives_its_piece_back_when_it_goes() {
        let Some(host) = crate::host::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: a valid description.
        let Some(t) = (unsafe { texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON, true) }) else { return };
        assert_eq!(crate::host::references(&t), 1, "what the texture holds is no reference to it");
        let resting = || lock(&TEXTURES).resting.len();
        let before = resting();
        let other = t.clone();
        drop(t);
        assert_eq!(resting(), before);
        drop(other);
        assert_eq!(resting(), before + 1);
        // SAFETY: the piece's mapping, 4096 bytes of it ours.
        unsafe {
            let (piece, _, at, data) = staging(host, 4096).unwrap();
            assert!(at.is_multiple_of(512));
            std::ptr::write_bytes(data, 0xAB, 4096);
            drop(piece);
            lock(&STAGING).rested(u64::MAX);
            let (_piece, _, _, data) = staging(host, 4096).unwrap();
            assert!((0..4096).all(|i| *data.add(i) == 0));
            // SAFETY: plain query.
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    #[test]
    fn spares_scales_with_room_target() {
        let gib = 1 << 30;
        assert_eq!(spares_for(Some([100 * gib, 84 * gib, 0, 0]), true, 85), SPARE_KEPT);
        assert_eq!(spares_for(Some([100 * gib, 85 * gib, 0, 0]), true, 85), SPARE);
        assert_eq!(spares_for(Some([100 * gib, 74 * gib, 0, 0]), true, 75), SPARE_KEPT);
        assert_eq!(spares_for(Some([100 * gib, 75 * gib, 0, 0]), true, 75), SPARE);
        assert_eq!(spares_for(Some([100 * gib, 10 * gib, 0, 0]), false, 85), SPARE);
    }

    #[test]
    fn default_disabled_wddm_take_ripe_and_reclaim_preserve_baseline_order() {
        assert!(!crate::residency_wddm::parse_wddm_residency_env(None));
        let mut pool: Pool<u32> = Pool::new(8);
        let b0 = pool.add(10, 0);
        let b1 = pool.add(11, 0);
        let b2 = pool.add(12, 0);
        pool.blocks[b0].as_mut().unwrap().residency = BlockResidency::Evicted;
        // Baseline `reclaim` trims empty blocks strictly past `limit` in ascending block order
        // without residency-based reordering or WDDM evict accounting.
        pool.reclaim(1_000, 1, true);
        assert!(pool.blocks[b0].is_some());
        assert!(pool.blocks[b1].is_none());
        assert!(pool.blocks[b2].is_none());
        // Baseline `take_ripe` does not enter `take_ripe_or_restore`.
        let mut next = 100u32;
        let (_piece, mem, again) = pool
            .take_ripe(BLOCK / 4, 65536, 2_000, Ripe::ALL, || {
                next += 1;
                Some(next)
            })
            .unwrap();
        assert_eq!(mem, 10);
        assert!(!again);
    }

    #[test]
    fn idle_blocks_demote_evict_fence_safely_and_restore_without_races() {
        let mut pool: Pool<u32> = Pool::new(8);
        // Three idle blocks aged differently, all carrying idle_fence = 10.
        let b0 = pool.add_at(100, 10, 10, 1_000); // oldest idle
        let b1 = pool.add_at(101, 10, 10, 18_000);
        let b2 = pool.add_at(102, 10, 10, 20_000); // newest idle
        assert_eq!(pool.resident_blocks(), 3);

        // Before GPU fence reaches 10: oldest aged block demotes to MINIMUM, NEVER Evict.
        let mut prios = Vec::new();
        let mut evicts = Vec::new();
        pool.plan_idle_residency(
            IdleResidencyCtx {
                now: 21_000,
                completed_submission: 9,
                vram_used_pct: Some(50),
                pressured: false,
                critical: false,
                normal_hot_spares: 2,
            },
            &mut prios,
            &mut evicts,
        );
        assert!(evicts.is_empty(), "must never evict while GPU fence < idle_fence");
        assert_eq!(pool.blocks[b0].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Minimum);
        assert_eq!(pool.blocks[b1].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Low);
        assert_eq!(pool.blocks[b2].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Low);

        // With normal_hot_spares = 12 (full ripe window), all 3 stay resident under normal VRAM.
        let mut prios_wide = Vec::new();
        let mut evicts_wide = Vec::new();
        pool.plan_idle_residency(
            IdleResidencyCtx {
                now: 21_000,
                completed_submission: 10,
                vram_used_pct: Some(50),
                pressured: false,
                critical: false,
                normal_hot_spares: 12,
            },
            &mut prios_wide,
            &mut evicts_wide,
        );
        assert!(evicts_wide.is_empty(), "normal headroom preserves the ripe ahead window");
        assert_eq!(pool.blocks[b0].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Low);

        // Once GPU fence reaches 10 and aged block exceeds normal_hot_spares: transitions to Evicting.
        prios.clear();
        pool.plan_idle_residency(
            IdleResidencyCtx {
                now: 21_000,
                completed_submission: 10,
                vram_used_pct: Some(50),
                pressured: false,
                critical: false,
                normal_hot_spares: 2,
            },
            &mut prios,
            &mut evicts,
        );
        assert_eq!(evicts, vec![(b0, 100)]);
        assert_eq!(pool.blocks[b0].as_ref().unwrap().residency, BlockResidency::Evicting);
        // Reclaim MUST NOT drop a block currently in Evicting state.
        assert_eq!(pool.reclaim_with_fence(21_000, 0, true, 10), 0);
        assert!(pool.blocks[b0].is_some());
        assert_eq!(pool.resident_blocks(), 0); // b1 & b2 dropped when limit = 0; b0 is Evicting.

        // Finalize eviction of b0.
        pool.blocks[b0].as_mut().unwrap().residency = BlockResidency::Evicted;
        assert_eq!(pool.resident_blocks(), 0);
        assert_eq!(pool.blocks(), 1);

        // Allocating with allow_restore reserves b0 in Restoring instead of growing a new block.
        let mut grew = 0;
        let outcome = pool.take_ripe_or_restore(
            TakeReq {
                len: BLOCK / 4,
                align: 65536,
                now: 22_000,
                ripe: Ripe::ALL,
                idle_fence: 12,
                allow_restore: true,
            },
            || {
                grew += 1;
                Some(999)
            },
        );
        assert_eq!(grew, 0);
        assert!(matches!(outcome, TakeOutcome::Restore { block, memory: 100, .. } if block == b0));
        assert_eq!(pool.blocks[b0].as_ref().unwrap().residency, BlockResidency::Restoring);

        // Finishing restore makes b0 resident, promotes priority to Normal, and hands out the piece.
        // Taking BLOCK fills b0 completely so the next allocation tests b4 + evicted b3.
        let (piece, mem, _) = pool
            .finish_restore_and_take(
                b0,
                TakeReq {
                    len: BLOCK,
                    align: 65536,
                    now: 22_000,
                    ripe: Ripe::ALL,
                    idle_fence: 12,
                    allow_restore: false,
                },
                || None,
            )
            .unwrap();
        assert_eq!((piece.block, mem), (b0, 100));
        assert_eq!(pool.blocks[b0].as_ref().unwrap().residency, BlockResidency::Resident);
        assert_eq!(pool.blocks[b0].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Normal);
        assert_eq!(pool.resident_blocks(), 1);

        // Dropping excess empty blocks prefers Evicted blocks first and returns dropped_evicted count.
        let b3 = pool.add_at(103, 12, 12, 22_000);
        let b4 = pool.add_at(104, 12, 12, 22_000);
        pool.blocks[b3].as_mut().unwrap().residency = BlockResidency::Evicted;
        // Evicted spare b3 counts toward ahead reserve so taking from b4 (leaving only b3 empty)
        // does not churn-grow a brand new heap.
        let mut churn_grew = 0;
        let take_res = pool.take_ripe_or_restore(
            TakeReq {
                len: BLOCK / 4,
                align: 65536,
                now: 22_000,
                ripe: Ripe { started: 13, finished: 12, ahead: 1 },
                idle_fence: 13,
                allow_restore: true,
            },
            || {
                churn_grew += 1;
                Some(888)
            },
        );
        assert_eq!(churn_grew, 0, "evicted spare satisfies ahead count without CreateHeap churn");
        assert!(matches!(take_res, TakeOutcome::Taken { piece, .. } if piece.block == b4));
        // Return the piece and verify reclaim drops Evicted b3 before Resident hot spare b4.
        pool.give(Piece { block: b4, start: 0, len: BLOCK / 4 }, 22_000);
        assert_eq!(pool.reclaim_with_fence(22_000 + REST, 1, true, 14), 1);
        assert!(pool.blocks[b3].is_none(), "evicted empty block dropped before hot resident spare");
        // Failed Evict() rollback mirrors Minimum priority so next allocation promotes back to Normal.
        pool.blocks[b4].as_mut().unwrap().priority = crate::residency_wddm::ResidencyTier::Minimum;
        let promote_res = pool.take_ripe_or_restore(
            TakeReq {
                len: BLOCK / 4,
                align: 65536,
                now: 22_000 + REST,
                ripe: Ripe::ALL,
                idle_fence: 15,
                allow_restore: false,
            },
            || None,
        );
        assert!(matches!(
            promote_res,
            TakeOutcome::Taken { piece, promote_normal: true, .. } if piece.block == b4
        ));
        assert_eq!(pool.blocks[b4].as_ref().unwrap().priority, crate::residency_wddm::ResidencyTier::Normal);
    }
}
