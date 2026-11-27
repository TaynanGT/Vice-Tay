//! Hardware virtual page table for loose title-memory windows
//! (`VICTORY_SPARSE_WINDOWS=1`; off by default).
//!
//! Loose windows outside registered buffer objects previously committed whole
//! 4 MiB or 8 MiB host buffers for every distinct `(start, len)` ask. Untouched
//! gaps stayed committed, and overlapping sized asks, tail asks, and root-view
//! asks duplicated the same 64 KiB title pages across separate host buffers.
//!
//! When enabled on an adapter with D3D12 Tiled Resources Tier 2 or higher:
//! - Each loose window is a `CreateReservedResource` buffer (`0` bytes
//!   physically committed on creation).
//! - Physical memory comes from pooled 64 KiB tiles (`watch::PAGE` == 64 KiB,
//!   exact 1:1 match) carved out of `D3D12_HEAP_TYPE_DEFAULT` heaps with
//!   `D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS`.
//! - A canonical page table maps each touched 64 KiB title page to at most ONE
//!   physical `(heap, tile)` slot and binds that exact slot into every live
//!   overlapping reserved window via `ID3D12CommandQueue::UpdateTileMappings`.
//! - Untouched gap tiles stay NULL-mapped (`0` physical bytes; Tier 2+ reads
//!   zero and drops writes).
//! - Pending upload payloads stay in normal cached CPU memory until `flush()`
//!   writes them sequentially into write-only `D3D12_HEAP_TYPE_UPLOAD` buffers
//!   (`D3D12_RANGE { Begin: 0, End: 0 }` strictly respected, zero WC reads).
//! - Active UAV byte ranges and pre-shader snapshots are tracked canonically
//!   per `SparseVa` across all overlapping windows so cross-window refreshes,
//!   overlapping UAV declarations, and `placed_copy` stay coherent with zero
//!   TOCTOU races.
//! - Unmapping and window teardown emit `D3D12_TILE_RANGE_FLAG_NULL` mappings,
//!   signal a queue fence, and hold retired tiles and reserved resources until
//!   `ID3D12Fence::GetCompletedValue` passes that fence before recycling tiles
//!   or releasing empty heaps.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::watch::PAGE;

/// Physical tile size in bytes: exact 1:1 match to `watch::PAGE` (64 KiB).
pub(crate) const TILE_BYTES: u64 = PAGE as u64;

const _: () = assert!(PAGE == 65_536, "D3D12 tile size must match watch::PAGE (64 KiB)");

/// Default tiles per backing `ID3D12Heap` (256 tiles = 16 MiB).
const DEFAULT_HEAP_TILES: u32 = 256;

/// Starting size for transient upload buffers (1 MiB, doubling to 4 MiB).
const MIN_UPLOAD_CHUNK_BYTES: u64 = 1 << 20;

/// Soft cap for pooled upload chunks kept warm between submissions (16 MiB).
const MAX_CACHED_UPLOAD_BYTES: u64 = 16 << 20;

/// Alignment of slices packed into a write-only upload buffer.
const UPLOAD_ALIGN: u64 = 16;

/// Process-global count of active declared UAV ranges on global sparse windows.
/// Checked lock-free by `Host::submit` to serialize rare sparse UAV flights in
/// `late`/`relay` mode before subsequent flights touch `host.queue()`.
static GLOBAL_ACTIVE_UAV: AtomicU32 = AtomicU32::new(0);

/// Returns whether any global sparse-VA window currently has an active UAV write
/// pending copy-back. Lock-free (`0 ns` when `VICTORY_SPARSE_WINDOWS` is off or
/// no loose window is UAV-written).
pub(crate) fn has_active_uav() -> bool {
    GLOBAL_ACTIVE_UAV.load(Ordering::Acquire) != 0
}

/// Queries `D3D12_FEATURE_DATA_D3D12_OPTIONS::TiledResourcesTier` on `device`.
pub(crate) fn tiled_resources_tier(device: &ID3D12Device) -> i32 {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: plain feature query writing a struct of its exact size.
    let ok = unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
    };
    if ok { options.TiledResourcesTier.0 } else { 0 }
}

/// Whether `device` supports Tier 2+ tiled resources (guaranteed zero-read and
/// write-drop semantics on NULL-mapped sparse tiles).
pub(crate) fn supported(device: &ID3D12Device) -> bool {
    tiled_resources_tier(device) >= D3D12_TILED_RESOURCES_TIER_2.0
}

/// Whether `VICTORY_SPARSE_WINDOWS=1` is requested and supported by the live host.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !std::env::var_os("VICTORY_SPARSE_WINDOWS").is_some_and(|v| v == "1") {
            return false;
        }
        let Some(host) = crate::host::host() else {
            return false;
        };
        let tier = tiled_resources_tier(&host.device);
        let on = tier >= D3D12_TILED_RESOURCES_TIER_2.0;
        tracing::info!(
            tier,
            on,
            tile_kb = TILE_BYTES >> 10,
            heap_mb = (u64::from(DEFAULT_HEAP_TILES) * TILE_BYTES) >> 20,
            "sparse virtual-address loose windows (VICTORY_SPARSE_WINDOWS=1)"
        );
        on
    })
}

/// Configuration for a [`SparseVa`] instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseConfig {
    /// Number of 64 KiB tiles in each pooled `ID3D12Heap`.
    pub(crate) heap_tiles: u32,
    /// Keep one empty heap warm to avoid churn on steady workloads.
    pub(crate) keep_one_heap: bool,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            heap_tiles: DEFAULT_HEAP_TILES,
            keep_one_heap: true,
        }
    }
}

/// Snapshot of live and cumulative [`SparseVa`] accounting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SparseStats {
    pub(crate) live_windows: u64,
    pub(crate) reserved_va_bytes: u64,
    pub(crate) committed_tiles: u64,
    pub(crate) committed_bytes: u64,
    pub(crate) mapped_window_tiles: u64,
    pub(crate) deduplicated_bytes: u64,
    pub(crate) heap_count: u64,
    pub(crate) heap_pool_bytes: u64,
    pub(crate) retired_tiles_pending: u64,
    pub(crate) update_tile_mappings_calls: u64,
    pub(crate) update_tile_mappings_us: u64,
    pub(crate) tiles_allocated_total: u64,
    pub(crate) tiles_reclaimed_total: u64,
    pub(crate) heaps_reclaimed_total: u64,
    pub(crate) upload_bytes_total: u64,
}

/// Per-local-tile mapping state inside one reserved window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum TileState {
    Unmapped = 0,
    PendingMap = 1,
    MappedOnGpu = 2,
}

/// One pooled `ID3D12Heap` and its sorted, coalesced free runs `(first_tile, count)`.
struct HeapChunk {
    heap: ID3D12Heap,
    free: Vec<(u32, u32)>,
    total_tiles: u32,
    /// Tiles currently live OR retired-pending-fence. A heap is dropped only
    /// when `in_use_tiles == 0`, guaranteeing zero GPU references remain.
    in_use_tiles: u32,
}

// SAFETY: `ID3D12Heap` is free-threaded; metadata is guarded by `SparseVa::inner`.
unsafe impl Send for HeapChunk {}

impl HeapChunk {
    fn alloc_tile(&mut self) -> Option<u32> {
        let &(start, count) = self.free.first()?;
        if count == 1 {
            self.free.remove(0);
        } else {
            self.free[0] = (start + 1, count - 1);
        }
        self.in_use_tiles += 1;
        Some(start)
    }

    fn free_tile(&mut self, tile_idx: u32) {
        self.in_use_tiles = self.in_use_tiles.saturating_sub(1);
        let pos = self.free.partition_point(|&(start, _)| start < tile_idx);
        let merge_prev = pos > 0 && self.free[pos - 1].0 + self.free[pos - 1].1 == tile_idx;
        let merge_next = pos < self.free.len() && tile_idx + 1 == self.free[pos].0;
        match (merge_prev, merge_next) {
            (true, true) => {
                let next_count = self.free.remove(pos).1;
                self.free[pos - 1].1 += 1 + next_count;
            }
            (true, false) => {
                self.free[pos - 1].1 += 1;
            }
            (false, true) => {
                self.free[pos].0 = tile_idx;
                self.free[pos].1 += 1;
            }
            (false, false) => {
                self.free.insert(pos, (tile_idx, 1));
            }
        }
    }
}

/// Canonical backing entry for one 64 KiB page of title virtual address space.
struct PageEntry {
    heap_idx: usize,
    tile_idx: u32,
    /// Number of live windows whose tile table binds this canonical page.
    mapped_windows: u32,
    /// Preferred live window ID covering this page for O(1) upload targeting.
    canonical_window: u64,
    /// True once all 64 KiB of the physical tile have been deterministically
    /// written (preventing stale bytes from a previous recycled owner).
    initialized: bool,
    /// Shared write-watch sequence across all overlapping windows (`0` = never
    /// copied whole via write-watch).
    copied_at: u64,
    /// Latest signaled fence value associated with GPU access to this page.
    last_fence: u64,
}

/// One live reserved buffer window registered in [`SparseVa`].
struct WindowRecord {
    id: u64,
    start_page: u64,
    page_count: u32,
    resource: ID3D12Resource,
    /// Per-local-tile mapping state (`length == page_count`).
    tiles: Vec<TileState>,
}

// SAFETY: `ID3D12Resource` is free-threaded; record is guarded by `SparseVa::inner`.
unsafe impl Send for WindowRecord {}

impl WindowRecord {
    fn contains_page(&self, title_page: u64) -> Option<u32> {
        let local = title_page.checked_sub(self.start_page)?;
        (local < u64::from(self.page_count)).then_some(local as u32)
    }
}

/// Write-only CPU-mapped upload buffer (`D3D12_HEAP_TYPE_UPLOAD`).
/// Mapped with `D3D12_RANGE { Begin: 0, End: 0 }`: CPU NEVER reads from `ptr`.
struct UploadChunk {
    resource: ID3D12Resource,
    ptr: *mut u8,
    capacity: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded; `ptr` is persistently mapped for
// the lifetime of `resource` and written only under `SparseVa::flush_lock`.
unsafe impl Send for UploadChunk {}

impl UploadChunk {
    fn create(device: &ID3D12Device, capacity: u64) -> windows::core::Result<Self> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: capacity.max(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
            ..Default::default()
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain committed upload buffer creation on a live device.
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(windows::core::Error::empty)?;
        let mut ptr = std::ptr::null_mut();
        let read_none = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: upload buffers are permanently CPU-mappable; we only write to `ptr`.
        unsafe {
            resource.Map(0, Some(&read_none), Some(&mut ptr))?;
        }
        Ok(Self {
            resource,
            ptr: ptr.cast(),
            capacity: desc.Width,
        })
    }
}

/// Reusable direct command allocator + list for sparse upload flushes and readbacks.
struct UploadFrame {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

// SAFETY: D3D12 allocator/list used by one submission at a time under `flush_lock`.
unsafe impl Send for UploadFrame {}

impl UploadFrame {
    fn create(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain command allocator/list creation on a live device.
        unsafe {
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
            list.Close()?;
            Ok(Self { allocator, list })
        }
    }
}

/// Pending upload slice in normal cached CPU RAM targeting a canonical 64 KiB
/// title page. Packed sequentially into a write-only [`UploadChunk`] on `flush`.
struct PendingUpload {
    title_page: u64,
    intra_offset: u32,
    /// Cached CPU bytes (`empty` marks a tombstoned entry superseded by a merge).
    data: Vec<u8>,
}

/// Pending tile mapping `(window_id, local_tile) -> (heap_idx, heap_tile)`.
struct PendingMap {
    window_id: u64,
    local_tile: u32,
    heap_idx: usize,
    heap_tile: u32,
}

/// Pending NULL tile unmap for a reserved resource before tile retirement.
struct PendingUnmap {
    resource: ID3D12Resource,
    local_tile: u32,
}

// SAFETY: `ID3D12Resource` is free-threaded.
unsafe impl Send for PendingUnmap {}

/// Mutable state behind [`SparseVa::inner`].
struct State {
    config: SparseConfig,
    is_global: bool,
    next_window_id: u64,
    /// Highest fence value actually signaled on `queue` via `queue.Signal`.
    /// Never incremented until `queue.Signal` returns `Ok(())`.
    signaled_fence: u64,
    heaps: Vec<Option<HeapChunk>>,
    pages: BTreeMap<u64, PageEntry>,
    windows: HashMap<u64, WindowRecord>,
    /// Canonical active UAV byte ranges `(from_addr, to_addr)` per window ID.
    active_uav_ranges: HashMap<u64, Vec<(usize, usize)>>,
    /// Canonical 64 KiB pre-shader reference snapshots for pages with active UAV writes.
    uav_pages: BTreeMap<u64, Vec<u8>>,
    /// Emits a global aliasing barrier on next `flush()` after UAV writes.
    needs_aliasing_barrier: bool,
    /// Pending tile mappings waiting for the next `flush`.
    pending_maps: Vec<PendingMap>,
    /// Pending NULL unmaps waiting for the next `flush`.
    pending_unmaps: Vec<PendingUnmap>,
    /// Pending uploads in cached CPU RAM waiting for the next `flush`.
    pending_uploads: Vec<PendingUpload>,
    /// O(1) index from `title_page` to indices in `pending_uploads` (cleared on `flush`).
    pending_by_page: HashMap<u64, Vec<usize>>,
    /// Recycled write-only upload chunks ready for reuse.
    free_upload_chunks: Vec<UploadChunk>,
    /// Upload chunks in flight on the GPU, gated on their flush fence.
    in_flight_uploads: VecDeque<(u64, UploadChunk)>,
    /// Recycled upload command frames.
    free_frames: Vec<UploadFrame>,
    /// Upload command frames in flight on the GPU, gated on their flush fence.
    in_flight_frames: VecDeque<(u64, UploadFrame)>,
    /// Frames/resources kept alive if `Signal` failed after GPU submission (device lost).
    abandoned_frames: Vec<UploadFrame>,
    abandoned_uploads: Vec<UploadChunk>,
    abandoned_resources: Vec<ID3D12Resource>,
    /// Physical tiles detached from all windows, waiting for `flush` to attach
    /// a post-unmap fence value.
    pending_retire_tiles: Vec<(usize, u32)>,
    /// Reserved resources detached from live windows, waiting for `flush` to
    /// attach a post-unmap fence value.
    pending_retire_resources: Vec<ID3D12Resource>,
    /// Physical tiles gated on GPU fence completion before returning to pool.
    retired_tiles: VecDeque<(u64, usize, u32)>,
    /// Reserved resources gated on GPU fence completion before final drop.
    retired_resources: VecDeque<(u64, ID3D12Resource)>,
    /// Cached transient readback buffer for UAV copy-back / verification.
    readback: Option<(ID3D12Resource, u64)>,
    /// Cumulative counters.
    reserved_va_bytes: u64,
    mapped_window_tiles: u64,
    update_tile_mappings_calls: u64,
    update_tile_mappings_us: u64,
    tiles_allocated_total: u64,
    tiles_reclaimed_total: u64,
    heaps_reclaimed_total: u64,
    upload_bytes_total: u64,
}

// SAFETY: every D3D12 COM object and mapped buffer is owned by `State` and
// accessed exclusively under `SparseVa::inner`.
unsafe impl Send for State {}

impl State {
    fn new(config: SparseConfig, is_global: bool) -> Self {
        Self {
            config: SparseConfig {
                heap_tiles: config.heap_tiles.max(1),
                keep_one_heap: config.keep_one_heap,
            },
            is_global,
            next_window_id: 1,
            signaled_fence: 0,
            heaps: Vec::new(),
            pages: BTreeMap::new(),
            windows: HashMap::new(),
            active_uav_ranges: HashMap::new(),
            uav_pages: BTreeMap::new(),
            needs_aliasing_barrier: false,
            pending_maps: Vec::new(),
            pending_unmaps: Vec::new(),
            pending_uploads: Vec::new(),
            pending_by_page: HashMap::new(),
            free_upload_chunks: Vec::new(),
            in_flight_uploads: VecDeque::new(),
            free_frames: Vec::new(),
            in_flight_frames: VecDeque::new(),
            abandoned_frames: Vec::new(),
            abandoned_uploads: Vec::new(),
            abandoned_resources: Vec::new(),
            pending_retire_tiles: Vec::new(),
            pending_retire_resources: Vec::new(),
            retired_tiles: VecDeque::new(),
            retired_resources: VecDeque::new(),
            readback: None,
            reserved_va_bytes: 0,
            mapped_window_tiles: 0,
            update_tile_mappings_calls: 0,
            update_tile_mappings_us: 0,
            tiles_allocated_total: 0,
            tiles_reclaimed_total: 0,
            heaps_reclaimed_total: 0,
            upload_bytes_total: 0,
        }
    }

    /// Allocates one 64 KiB physical tile from the heap pool, creating a new
    /// `ID3D12Heap` when all existing heaps are full.
    fn alloc_physical_tile(&mut self, device: &ID3D12Device) -> windows::core::Result<(usize, u32)> {
        for (idx, slot) in self.heaps.iter_mut().enumerate() {
            if let Some(chunk) = slot
                && let Some(tile) = chunk.alloc_tile()
            {
                self.tiles_allocated_total += 1;
                return Ok((idx, tile));
            }
        }
        let tiles = self.config.heap_tiles;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: u64::from(tiles) * TILE_BYTES,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain default buffer heap creation on a live device.
        unsafe {
            device.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut heap)?;
        }
        let heap = heap.ok_or_else(windows::core::Error::empty)?;
        let mut fresh = HeapChunk {
            heap,
            free: vec![(0, tiles)],
            total_tiles: tiles,
            in_use_tiles: 0,
        };
        let tile = fresh.alloc_tile().ok_or_else(windows::core::Error::empty)?;
        self.tiles_allocated_total += 1;
        let idx = match self.heaps.iter().position(Option::is_none) {
            Some(i) => {
                self.heaps[i] = Some(fresh);
                i
            }
            None => {
                self.heaps.push(Some(fresh));
                self.heaps.len() - 1
            }
        };
        Ok((idx, tile))
    }

    /// Ensures canonical page `title_page` has a physical tile and is bound
    /// (at least `PendingMap`) into every live window overlapping `title_page`.
    /// Returns whether the page needs full 64 KiB initialization.
    fn ensure_page(&mut self, device: &ID3D12Device, title_page: u64) -> windows::core::Result<bool> {
        // Invariant: every `title_page` in `self.pages` was bound to all live
        // overlapping windows on insertion, and `create_window` binds all
        // existing pages in any newly created window. Existing pages hit O(1).
        if let Some(entry) = self.pages.get(&title_page) {
            return Ok(!entry.initialized);
        }

        // Only allocate a physical tile if at least one live window spans it.
        let mut targets: Vec<(u64, u32)> = Vec::new();
        for window in self.windows.values() {
            if let Some(local) = window.contains_page(title_page) {
                targets.push((window.id, local));
            }
        }
        if targets.is_empty() {
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_INVALIDARG,
            ));
        }

        let (heap_idx, tile_idx) = self.alloc_physical_tile(device)?;
        let mut canonical_window = u64::MAX;
        for &(window_id, local) in &targets {
            canonical_window = canonical_window.min(window_id);
            if let Some(window) = self.windows.get_mut(&window_id) {
                window.tiles[local as usize] = TileState::PendingMap;
                self.pending_maps.push(PendingMap {
                    window_id,
                    local_tile: local,
                    heap_idx,
                    heap_tile: tile_idx,
                });
            }
        }
        let count = targets.len() as u32;
        self.mapped_window_tiles += u64::from(count);
        self.pages.insert(
            title_page,
            PageEntry {
                heap_idx,
                tile_idx,
                mapped_windows: count,
                canonical_window,
                initialized: false,
                copied_at: 0,
                last_fence: self.signaled_fence,
            },
        );
        Ok(true)
    }

    /// Stages cached CPU bytes `[intra_offset, intra_offset + bytes.len())` for
    /// `title_page`, coalescing overlapping sub-ranges in cached RAM so `flush()`
    /// issues strictly disjoint `CopyBufferRegion` commands with zero reads from
    /// write-combined upload buffers.
    fn stage_bytes(&mut self, title_page: u64, intra_offset: u32, bytes: Vec<u8>) {
        let len = bytes.len() as u32;
        if len == 0 {
            return;
        }
        let end = intra_offset + len;
        debug_assert!(u64::from(end) <= TILE_BYTES);

        if let Some(indices) = self.pending_by_page.get(&title_page) {
            // Fast O(1) hit: an existing staged slice covers `[intra_offset, end)`.
            for &idx in indices.iter().rev() {
                let existing = &mut self.pending_uploads[idx];
                if !existing.data.is_empty()
                    && existing.intra_offset <= intra_offset
                    && existing.intra_offset + existing.data.len() as u32 >= end
                {
                    let rel = (intra_offset - existing.intra_offset) as usize;
                    existing.data[rel..rel + bytes.len()].copy_from_slice(&bytes);
                    return;
                }
            }

            // Check overlap among slices of this single `title_page`.
            let overlaps = indices.iter().any(|&idx| {
                let u = &self.pending_uploads[idx];
                !u.data.is_empty()
                    && u.intra_offset < end
                    && u.intra_offset + u.data.len() as u32 > intra_offset
            });
            if overlaps {
                let indices = indices.clone();
                let mut merged_lo = intra_offset;
                let mut merged_hi = end;
                loop {
                    let (prev_lo, prev_hi) = (merged_lo, merged_hi);
                    for &idx in &indices {
                        let u = &self.pending_uploads[idx];
                        if !u.data.is_empty()
                            && u.intra_offset < merged_hi
                            && u.intra_offset + u.data.len() as u32 > merged_lo
                        {
                            merged_lo = merged_lo.min(u.intra_offset);
                            merged_hi = merged_hi.max(u.intra_offset + u.data.len() as u32);
                        }
                    }
                    if (merged_lo, merged_hi) == (prev_lo, prev_hi) {
                        break;
                    }
                }
                let merged_len = (merged_hi - merged_lo) as usize;
                let mut merged = vec![0u8; merged_len];
                let mut surviving = Vec::new();
                for &idx in &indices {
                    let u = &mut self.pending_uploads[idx];
                    if !u.data.is_empty()
                        && u.intra_offset < merged_hi
                        && u.intra_offset + u.data.len() as u32 > merged_lo
                    {
                        let rel = (u.intra_offset - merged_lo) as usize;
                        merged[rel..rel + u.data.len()].copy_from_slice(&u.data);
                        u.data.clear();
                    } else if !u.data.is_empty() {
                        surviving.push(idx);
                    }
                }
                let rel = (intra_offset - merged_lo) as usize;
                merged[rel..rel + bytes.len()].copy_from_slice(&bytes);
                let new_idx = self.pending_uploads.len();
                self.pending_uploads.push(PendingUpload {
                    title_page,
                    intra_offset: merged_lo,
                    data: merged,
                });
                surviving.push(new_idx);
                self.pending_by_page.insert(title_page, surviving);
                self.upload_bytes_total += u64::from(len);
                return;
            }
        }

        let new_idx = self.pending_uploads.len();
        self.pending_uploads.push(PendingUpload {
            title_page,
            intra_offset,
            data: bytes,
        });
        self.pending_by_page.entry(title_page).or_default().push(new_idx);
        self.upload_bytes_total += u64::from(len);
    }

    /// Acquires a recycled or newly created write-only `UploadChunk` holding at
    /// least `total_bytes`. Called BEFORE draining pending queues in `flush()`.
    fn take_upload_chunk(
        &mut self,
        device: &ID3D12Device,
        total_bytes: u64,
    ) -> windows::core::Result<UploadChunk> {
        let fit = self
            .free_upload_chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.capacity >= total_bytes)
            .min_by_key(|(_, c)| c.capacity)
            .map(|(i, _)| i);
        match fit {
            Some(i) => Ok(self.free_upload_chunks.swap_remove(i)),
            None => {
                let capacity = total_bytes.max(MIN_UPLOAD_CHUNK_BYTES).next_power_of_two();
                UploadChunk::create(device, capacity)
            }
        }
    }

    /// Returns all currently active UAV byte ranges across ALL overlapping
    /// windows that intersect `[from, to)`, sorted and coalesced.
    fn active_uav_ranges_in(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        if from >= to || self.active_uav_ranges.is_empty() {
            return Vec::new();
        }
        let mut ranges: Vec<(usize, usize)> = self
            .active_uav_ranges
            .values()
            .flat_map(|v| v.iter().copied())
            .filter(|&(a, e)| a < to && e > from)
            .map(|(a, e)| (a.max(from), e.min(to)))
            .collect();
        if ranges.is_empty() {
            return ranges;
        }
        ranges.sort_unstable();
        let mut joined: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
        for (a, e) in ranges {
            match joined.last_mut() {
                Some(last) if a <= last.1 => last.1 = last.1.max(e),
                _ => joined.push((a, e)),
            }
        }
        joined
    }

    /// Returns `[from, to)` minus all currently active UAV byte ranges across
    /// ALL windows.
    fn subtract_active_uav(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        if from >= to {
            return Vec::new();
        }
        let busy = self.active_uav_ranges_in(from, to);
        if busy.is_empty() {
            return vec![(from, to)];
        }
        let mut out = Vec::new();
        let mut at = from;
        for (a, e) in busy {
            if a > at {
                out.push((at, a));
            }
            at = at.max(e);
        }
        if at < to {
            out.push((at, to));
        }
        out
    }

    /// Drops `uav_pages` snapshots for pages no longer covered by any active UAV range.
    fn prune_unused_uav_pages(&mut self) {
        if self.active_uav_ranges.is_empty() {
            self.uav_pages.clear();
            return;
        }
        let mut live_ranges: Vec<(usize, usize)> = self
            .active_uav_ranges
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        if live_ranges.is_empty() {
            self.uav_pages.clear();
            return;
        }
        live_ranges.sort_unstable();
        self.uav_pages.retain(|&page, _| {
            let p_lo = (page as usize) * PAGE;
            let p_hi = p_lo + PAGE;
            live_ranges.iter().any(|&(a, e)| a < p_hi && e > p_lo)
        });
    }

    /// Reclaims all retired tiles, resources, upload chunks, and command frames
    /// whose GPU fence `<= completed`.
    fn reclaim_completed(&mut self, completed: u64) -> usize {
        while let Some(&(f, _)) = self.in_flight_frames.front() {
            if f > completed {
                break;
            }
            if let Some((_, frame)) = self.in_flight_frames.pop_front() {
                self.free_frames.push(frame);
            }
        }
        while let Some(&(f, _)) = self.in_flight_uploads.front() {
            if f > completed {
                break;
            }
            if let Some((_, chunk)) = self.in_flight_uploads.pop_front() {
                self.free_upload_chunks.push(chunk);
            }
        }
        let mut cached: u64 = self.free_upload_chunks.iter().map(|c| c.capacity).sum();
        if cached > MAX_CACHED_UPLOAD_BYTES {
            self.free_upload_chunks
                .sort_by_key(|c| std::cmp::Reverse(c.capacity));
            while cached > MAX_CACHED_UPLOAD_BYTES && self.free_upload_chunks.len() > 1 {
                if let Some(dropped) = self.free_upload_chunks.pop() {
                    cached = cached.saturating_sub(dropped.capacity);
                }
            }
        }
        while let Some(&(f, _)) = self.retired_resources.front() {
            if f > completed {
                break;
            }
            let _ = self.retired_resources.pop_front();
        }
        let mut reclaimed = 0usize;
        let mut touched_heaps = BTreeSet::new();
        while let Some(&(f, _, _)) = self.retired_tiles.front() {
            if f > completed {
                break;
            }
            if let Some((_, heap_idx, tile_idx)) = self.retired_tiles.pop_front() {
                if let Some(Some(chunk)) = self.heaps.get_mut(heap_idx) {
                    chunk.free_tile(tile_idx);
                    touched_heaps.insert(heap_idx);
                }
                reclaimed += 1;
                self.tiles_reclaimed_total += 1;
            }
        }
        if !touched_heaps.is_empty() {
            let min_keep = usize::from(self.config.keep_one_heap);
            let mut live_heaps = self.heaps.iter().flatten().count();
            for heap_idx in touched_heaps {
                if live_heaps <= min_keep {
                    break;
                }
                if self
                    .heaps
                    .get(heap_idx)
                    .and_then(|s| s.as_ref())
                    .is_some_and(|c| c.in_use_tiles == 0)
                {
                    self.heaps[heap_idx] = None;
                    live_heaps -= 1;
                    self.heaps_reclaimed_total += 1;
                }
            }
        }
        reclaimed
    }

    fn stats(&self) -> SparseStats {
        let committed_tiles = self.pages.len() as u64;
        let committed_bytes = committed_tiles * TILE_BYTES;
        let deduplicated_tiles = self.mapped_window_tiles.saturating_sub(committed_tiles);
        let heap_count = self.heaps.iter().flatten().count() as u64;
        let heap_pool_bytes: u64 = self
            .heaps
            .iter()
            .flatten()
            .map(|h| u64::from(h.total_tiles) * TILE_BYTES)
            .sum();
        SparseStats {
            live_windows: self.windows.len() as u64,
            reserved_va_bytes: self.reserved_va_bytes,
            committed_tiles,
            committed_bytes,
            mapped_window_tiles: self.mapped_window_tiles,
            deduplicated_bytes: deduplicated_tiles * TILE_BYTES,
            heap_count,
            heap_pool_bytes,
            retired_tiles_pending: (self.retired_tiles.len() + self.pending_retire_tiles.len()) as u64,
            update_tile_mappings_calls: self.update_tile_mappings_calls,
            update_tile_mappings_us: self.update_tile_mappings_us,
            tiles_allocated_total: self.tiles_allocated_total,
            tiles_reclaimed_total: self.tiles_reclaimed_total,
            heaps_reclaimed_total: self.heaps_reclaimed_total,
            upload_bytes_total: self.upload_bytes_total,
        }
    }
}

/// Handle to one reserved-resource window backed by [`SparseVa`].
#[derive(Clone)]
pub(crate) struct SparseWindowHandle {
    pub(crate) id: u64,
    pub(crate) resource: ID3D12Resource,
    pub(crate) gpu_va: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded.
unsafe impl Send for SparseWindowHandle {}
// SAFETY: `ID3D12Resource` is free-threaded.
unsafe impl Sync for SparseWindowHandle {}

/// Device-scoped hardware virtual page table managing reserved buffer windows
/// backed by pooled 64 KiB `ID3D12Heap` tiles via `UpdateTileMappings`.
pub(crate) struct SparseVa {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: HANDLE,
    /// Per-instance count of active declared UAV ranges (lock-free fast path).
    active_uav_count: AtomicU32,
    /// Serializes queue-order flush sequences (`UpdateTileMappings` -> upload
    /// `ExecuteCommandLists` -> `Signal`). Always acquired before `inner`.
    flush_lock: Mutex<()>,
    inner: Mutex<State>,
}

// SAFETY: D3D12 device, queue, and fence are free-threaded; `event` is waited
// on strictly under `flush_lock`.
unsafe impl Send for SparseVa {}
// SAFETY: as above.
unsafe impl Sync for SparseVa {}

impl Drop for SparseVa {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        // SAFETY: `self.event` is owned exclusively by this `SparseVa`.
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

impl SparseVa {
    /// Creates a new sparse virtual-address backing engine for `(device, queue)`.
    #[cfg(test)]
    pub(crate) fn new(
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        config: SparseConfig,
    ) -> windows::core::Result<Self> {
        Self::new_with_mode(device, queue, config, false)
    }

    fn new_with_mode(
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        config: SparseConfig,
        is_global: bool,
    ) -> windows::core::Result<Self> {
        if !supported(&device) {
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_NOTIMPL,
            ));
        }
        // SAFETY: plain fence and Win32 auto-reset event creation.
        let (fence, event) = unsafe {
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            (fence, event)
        };
        Ok(Self {
            device,
            queue,
            fence,
            event,
            active_uav_count: AtomicU32::new(0),
            flush_lock: Mutex::new(()),
            inner: Mutex::new(State::new(config, is_global)),
        })
    }

    fn adjust_uav_count(&self, is_global: bool, add: u32, sub: u32) {
        if add > 0 {
            self.active_uav_count.fetch_add(add, Ordering::Release);
            if is_global {
                GLOBAL_ACTIVE_UAV.fetch_add(add, Ordering::Release);
            }
        }
        if sub > 0 {
            self.active_uav_count.fetch_sub(sub, Ordering::Release);
            if is_global {
                GLOBAL_ACTIVE_UAV.fetch_sub(sub, Ordering::Release);
            }
        }
    }

    /// Creates a reserved buffer window covering `[start_addr, start_addr + len)`.
    /// Commits `0` physical tiles on creation; any title pages in the span that
    /// are ALREADY committed by overlapping windows are scheduled to bind to the
    /// exact same physical tiles before the next GPU submission.
    pub(crate) fn create_window(
        &self,
        start_addr: u64,
        len: u64,
    ) -> windows::core::Result<SparseWindowHandle> {
        if len == 0 || !start_addr.is_multiple_of(TILE_BYTES) || !len.is_multiple_of(TILE_BYTES) {
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_INVALIDARG,
            ));
        }
        let end_addr = start_addr.checked_add(len).ok_or_else(|| {
            windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG)
        })?;
        let start_page = start_addr / TILE_BYTES;
        let end_page = end_addr / TILE_BYTES;
        let page_count = u32::try_from(end_page - start_page).map_err(|_| {
            windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG)
        })?;

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        };
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain reserved buffer creation; buffers always begin in COMMON.
        unsafe {
            self.device.CreateReservedResource(
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: live reserved buffer.
        let gpu_va = unsafe { resource.GetGPUVirtualAddress() };

        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let id = state.next_window_id;
        state.next_window_id += 1;

        let mut tiles = vec![TileState::Unmapped; page_count as usize];
        let mut bound = 0u64;
        let State {
            pages,
            pending_maps,
            mapped_window_tiles,
            reserved_va_bytes,
            windows,
            ..
        } = &mut *state;
        // Bind any canonical pages already committed by overlapping windows so
        // this new view aliases the exact same physical tiles with zero copies.
        for (&title_page, entry) in pages.range_mut(start_page..end_page) {
            let local = (title_page - start_page) as u32;
            tiles[local as usize] = TileState::PendingMap;
            entry.mapped_windows += 1;
            bound += 1;
            pending_maps.push(PendingMap {
                window_id: id,
                local_tile: local,
                heap_idx: entry.heap_idx,
                heap_tile: entry.tile_idx,
            });
        }
        *mapped_window_tiles += bound;
        *reserved_va_bytes += len;
        windows.insert(
            id,
            WindowRecord {
                id,
                start_page,
                page_count,
                resource: resource.clone(),
                tiles,
            },
        );
        Ok(SparseWindowHandle {
            id,
            resource,
            gpu_va,
        })
    }

    /// Commits tiles covering `[title_addr, title_addr + data.len())` and stages
    /// `data` in cached CPU memory for upload. Any brand-new (possibly recycled)
    /// 64 KiB tile has its uncovered bytes zero-initialized before upload.
    #[cfg(test)]
    pub(crate) fn upload_range(&self, title_addr: u64, data: &[u8]) -> windows::core::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut offset = 0usize;
        while offset < data.len() {
            let cur_addr = title_addr + offset as u64;
            let title_page = cur_addr / TILE_BYTES;
            let intra_off = (cur_addr % TILE_BYTES) as u32;
            let chunk_len = ((TILE_BYTES as usize) - intra_off as usize).min(data.len() - offset);
            let slice = &data[offset..offset + chunk_len];

            let needs_full_init = state.ensure_page(&self.device, title_page)?;
            if needs_full_init && (intra_off != 0 || chunk_len != PAGE) {
                let mut full = vec![0u8; PAGE];
                full[intra_off as usize..intra_off as usize + chunk_len].copy_from_slice(slice);
                state.stage_bytes(title_page, 0, full);
            } else {
                state.stage_bytes(title_page, intra_off, slice.to_vec());
            }
            if let Some(entry) = state.pages.get_mut(&title_page) {
                entry.initialized = true;
            }
            offset += chunk_len;
        }
        Ok(())
    }

    /// Commits physical tiles for `[title_addr, title_addr + len)` and
    /// zero-initializes any newly allocated tiles (for GPU-written buffers).
    #[cfg(test)]
    pub(crate) fn ensure_committed_zeroed(
        &self,
        title_addr: u64,
        len: u64,
    ) -> windows::core::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let end = title_addr.checked_add(len).ok_or_else(|| {
            windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG)
        })?;
        let first_page = title_addr / TILE_BYTES;
        let last_page = (end - 1) / TILE_BYTES;
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        for title_page in first_page..=last_page {
            let needs_init = state.ensure_page(&self.device, title_page)?;
            if needs_init {
                state.stage_bytes(title_page, 0, vec![0u8; PAGE]);
                if let Some(entry) = state.pages.get_mut(&title_page) {
                    entry.initialized = true;
                }
            }
        }
        Ok(())
    }

    /// Refreshes title memory `[from, to)` into canonical 64 KiB sparse tiles
    /// using the shared `copied_at` sequence across all overlapping windows.
    /// Keeps any co-resident `uav_pages` reference on disjoint sub-ranges in sync
    /// from the same single read. Returns bytes staged for upload.
    pub(crate) fn stage_title_range(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: Option<&[(usize, u64)]>,
    ) -> windows::core::Result<usize> {
        if from >= to {
            return Ok(0);
        }
        let first_page = from / PAGE;
        let last_page = (to - 1) / PAGE;
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut staged = 0usize;

        let mut wi = written.map_or(0, |w| w.partition_point(|&(p, _)| p < first_page));
        for page_usize in first_page..=last_page {
            let title_page = page_usize as u64;
            let page_lo = page_usize * PAGE;
            let page_hi = page_lo + PAGE;
            let sub_lo = page_lo.max(from);
            let sub_hi = page_hi.min(to);
            let whole_page = sub_lo == page_lo && sub_hi == page_hi;

            let latest_write = if let Some(w) = written {
                let mut latest = 0u64;
                while wi < w.len() && w[wi].0 < page_usize {
                    wi += 1;
                }
                while wi < w.len() && w[wi].0 == page_usize {
                    latest = latest.max(w[wi].1);
                    wi += 1;
                }
                Some(latest)
            } else {
                None
            };

            if let Some(latest) = latest_write
                && let Some(entry) = state.pages.get(&title_page)
                && entry.initialized
                && entry.copied_at != 0
                && latest <= entry.copied_at
            {
                continue;
            }

            let needs_full_init = state.ensure_page(&self.device, title_page)?;
            if needs_full_init {
                let mut full = vec![0u8; PAGE];
                crate::host::copy_mapped(full.as_mut_ptr(), page_lo as *const u8, PAGE, false);
                if let Some(snap) = state.uav_pages.get_mut(&title_page) {
                    snap.copy_from_slice(&full);
                }
                state.stage_bytes(title_page, 0, full);
                if let Some(entry) = state.pages.get_mut(&title_page) {
                    entry.initialized = true;
                    if now != 0 {
                        entry.copied_at = now;
                    }
                }
                staged += PAGE;
            } else {
                let intra_off = (sub_lo - page_lo) as u32;
                let sub_len = sub_hi - sub_lo;
                let mut slice = vec![0u8; sub_len];
                crate::host::copy_mapped(slice.as_mut_ptr(), sub_lo as *const u8, sub_len, false);
                if let Some(snap) = state.uav_pages.get_mut(&title_page) {
                    snap[intra_off as usize..intra_off as usize + sub_len].copy_from_slice(&slice);
                }
                state.stage_bytes(title_page, intra_off, slice);
                if whole_page
                    && now != 0
                    && let Some(entry) = state.pages.get_mut(&title_page)
                {
                    entry.copied_at = now;
                }
                staged += sub_len;
            }
        }
        Ok(staged)
    }

    /// Declares an active GPU UAV write on `[from_addr, to_addr)` for `window_id`.
    /// Subtracts all already-active UAV ranges across ALL overlapping windows so
    /// in-flight/active GPU bytes are never overwritten, captures newly covered
    /// bytes in a SINGLE title-memory read used for BOTH GPU upload and the
    /// canonical `uav_pages` snapshot (zero TOCTOU), and flushes immediately.
    pub(crate) fn declare_uav_write(
        &self,
        window_id: u64,
        from_addr: usize,
        to_addr: usize,
    ) -> windows::core::Result<()> {
        if from_addr >= to_addr {
            return Ok(());
        }
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let uncovered = state.subtract_active_uav(from_addr, to_addr);
            let is_global = state.is_global;
            state
                .active_uav_ranges
                .entry(window_id)
                .or_default()
                .push((from_addr, to_addr));
            self.adjust_uav_count(is_global, 1, 0);
            state.needs_aliasing_barrier = true;

            for (u_lo, u_hi) in uncovered {
                let first_page = u_lo / PAGE;
                let last_page = (u_hi - 1) / PAGE;
                for page_usize in first_page..=last_page {
                    let title_page = page_usize as u64;
                    let page_lo = page_usize * PAGE;
                    let sub_lo = page_lo.max(u_lo);
                    let sub_hi = (page_lo + PAGE).min(u_hi);
                    let intra_off = sub_lo - page_lo;
                    let sub_len = sub_hi - sub_lo;

                    let needs_full_init = state.ensure_page(&self.device, title_page)?;
                    if needs_full_init || !state.uav_pages.contains_key(&title_page) {
                        // Single title-memory read feeds BOTH GPU upload and `uav_pages`
                        // snapshot, eliminating any TOCTOU window between the two.
                        let mut page_buf = vec![0u8; PAGE];
                        crate::host::copy_mapped(
                            page_buf.as_mut_ptr(),
                            page_lo as *const u8,
                            PAGE,
                            false,
                        );
                        if needs_full_init {
                            state.stage_bytes(title_page, 0, page_buf.clone());
                            if let Some(entry) = state.pages.get_mut(&title_page) {
                                entry.initialized = true;
                            }
                        } else {
                            state.stage_bytes(
                                title_page,
                                intra_off as u32,
                                page_buf[intra_off..intra_off + sub_len].to_vec(),
                            );
                        }
                        state.uav_pages.insert(title_page, page_buf);
                    } else {
                        // Disjoint sub-range on a page already holding a `uav_pages`
                        // snapshot: single read updates both snapshot and GPU upload.
                        let mut slice = vec![0u8; sub_len];
                        crate::host::copy_mapped(
                            slice.as_mut_ptr(),
                            sub_lo as *const u8,
                            sub_len,
                            false,
                        );
                        if let Some(snap) = state.uav_pages.get_mut(&title_page) {
                            snap[intra_off..intra_off + sub_len].copy_from_slice(&slice);
                        }
                        state.stage_bytes(title_page, intra_off as u32, slice);
                    }
                }
            }
        }
        let _ = self.flush()?;
        Ok(())
    }

    /// Updates ONLY the canonical pre-shader reference snapshots for placed
    /// copy bytes `[address, address + bytes.len())`. Does NOT stage a GPU
    /// upload (the command list already executes the placed `CopyBufferRegion`
    /// in order; staging an upload here would clobber the next submission).
    pub(crate) fn update_placed_reference(&self, address: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut off = 0usize;
        while off < bytes.len() {
            let cur = address + off;
            let page = (cur / PAGE) as u64;
            let intra = cur % PAGE;
            let n = (PAGE - intra).min(bytes.len() - off);
            if let Some(snap) = state.uav_pages.get_mut(&page) {
                snap[intra..intra + n].copy_from_slice(&bytes[off..off + n]);
            }
            if let Some(entry) = state.pages.get_mut(&page) {
                entry.copied_at = 0;
            }
            off += n;
        }
    }

    /// Finishes `freed_ranges` (`(from_addr, to_addr)`) for `window_id` after
    /// copy-back completes, pruning `uav_pages` snapshots no longer covered.
    pub(crate) fn finish_uav_ranges(&self, window_id: u64, freed_ranges: &[(usize, usize)]) {
        if freed_ranges.is_empty() {
            return;
        }
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let is_global = state.is_global;
        let mut removed = 0u32;
        if let Some(list) = state.active_uav_ranges.get_mut(&window_id) {
            for &target in freed_ranges {
                if let Some(pos) = list.iter().position(|&r| r == target) {
                    list.swap_remove(pos);
                    removed += 1;
                }
            }
            if list.is_empty() {
                state.active_uav_ranges.remove(&window_id);
            }
        }
        if removed > 0 {
            self.adjust_uav_count(is_global, 0, removed);
            state.prune_unused_uav_pages();
        }
    }

    /// Lock-free check whether ANY window in this `SparseVa` has an active UAV write.
    pub(crate) fn has_any_active_uav(&self) -> bool {
        self.active_uav_count.load(Ordering::Acquire) != 0
    }

    /// Whether ANY overlapping window in this `SparseVa` has an active UAV write
    /// intersecting `[from, to)`.
    pub(crate) fn has_active_uav_in(&self, from: usize, to: usize) -> bool {
        if from >= to || !self.has_any_active_uav() {
            return false;
        }
        let state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state
            .active_uav_ranges
            .values()
            .flat_map(|v| v.iter())
            .any(|&(a, e)| a < to && e > from)
    }

    /// Returns coalesced active UAV byte ranges intersecting `[from, to)` across
    /// ALL overlapping windows.
    pub(crate) fn active_uav_ranges_in(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        if from >= to || !self.has_any_active_uav() {
            return Vec::new();
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_uav_ranges_in(from, to)
    }

    /// Returns per-page `copied_at` timestamps for `[from, to)` iff every page
    /// in the range is already committed and initialized.
    pub(crate) fn copied_at_range(&self, from: usize, to: usize) -> Option<(usize, Vec<u64>)> {
        if from >= to {
            return Some((from / PAGE, Vec::new()));
        }
        let first_page = from / PAGE;
        let last_page = (to - 1) / PAGE;
        let state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(last_page - first_page + 1);
        for page_usize in first_page..=last_page {
            let entry = state.pages.get(&(page_usize as u64))?;
            if !entry.initialized || entry.copied_at == 0 {
                return None;
            }
            out.push(entry.copied_at);
        }
        Some((first_page, out))
    }

    /// Reads back `[from, to)` from `resource`, diffs each 64 KiB page slice
    /// against `uav_pages`, trims `read_into`/`put` strictly to
    /// `first_diff..=last_diff` (preserving writes even when untouched 4 KiB
    /// sub-pages are unmapped), and resolves same-byte CPU+GPU conflicts in
    /// favor of title memory while updating the GPU reference to the actual GPU
    /// byte (`*o = g`) so subsequent copy-backs never clobber title's byte.
    pub(crate) fn copy_back_range(
        &self,
        resource: &ID3D12Resource,
        start: u64,
        from: usize,
        to: usize,
        put: impl Fn(usize, &[u8], &[u8]) -> usize,
    ) -> usize {
        if from >= to {
            return 0;
        }
        let gpu_bytes = match self.readback_range(resource, from as u64, to - from) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, from, to, "sparse VA readback_range failed");
                return 0;
            }
        };
        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0usize;
        let base_addr = start as usize;
        let mut off = 0usize;
        let mut title = Vec::new();
        while off < gpu_bytes.len() {
            let addr = base_addr + from + off;
            let page = (addr / PAGE) as u64;
            let intra = addr % PAGE;
            let n = (PAGE - intra).min(gpu_bytes.len() - off);
            if let Some(snap) = state.uav_pages.get_mut(&page) {
                let gpu_slice = &gpu_bytes[off..off + n];
                let old_slice = &mut snap[intra..intra + n];
                if let Some(first) = gpu_slice
                    .iter()
                    .zip(old_slice.iter())
                    .position(|(g, o)| g != o)
                {
                    let last = gpu_slice
                        .iter()
                        .zip(old_slice.iter())
                        .rposition(|(g, o)| g != o)
                        .unwrap_or(first);
                    let trim_len = last + 1 - first;
                    let trim_addr = addr + first;
                    let g_trim = &gpu_slice[first..=last];
                    let o_trim = &mut old_slice[first..=last];
                    title.clear();
                    title.resize(trim_len, 0);
                    if crate::host::read_into(trim_addr as *const u8, &mut title) {
                        let was = title.clone();
                        let mut changed = false;
                        for ((t, &g), o) in
                            title.iter_mut().zip(g_trim).zip(o_trim.iter_mut())
                        {
                            let gpu_wrote = g != *o;
                            if gpu_wrote {
                                if *t == *o {
                                    *t = g;
                                    changed = true;
                                }
                                // Keep reference synchronized with the GPU tile's
                                // actual byte `g` even when title wins a conflict.
                                *o = g;
                            }
                        }
                        if changed {
                            written += put(trim_addr, &was, &title);
                        }
                    } else {
                        o_trim.copy_from_slice(g_trim);
                    }
                }
            }
            off += n;
        }
        let first_page = ((base_addr + from) / PAGE) as u64;
        let last_page = ((base_addr + to - 1) / PAGE) as u64;
        for (_, entry) in state.pages.range_mut(first_page..=last_page) {
            entry.copied_at = 0;
        }
        state.needs_aliasing_barrier = true;
        written
    }

    /// Submits all pending NULL unmaps, tile mappings (`UpdateTileMappings`),
    /// and staged `CopyBufferRegion` uploads on `self.queue`, followed by an
    /// aliasing barrier and queue fence signal.
    ///
    /// Error-path invariants:
    /// - Fallible frame/upload allocations happen BEFORE draining pending state
    ///   or flipping `TileState::PendingMap -> MappedOnGpu`.
    /// - `signaled_fence` advances ONLY AFTER `queue.Signal` succeeds, so
    ///   `wait_idle()` / `Drop` can never deadlock on an unsignaled fence.
    /// - Retired resources and tiles are never dropped without a signaled fence
    ///   (falling back to `abandoned_*` on device removal).
    pub(crate) fn flush(&self) -> windows::core::Result<u64> {
        let _flush_guard = self.flush_lock.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: plain fence query on a live fence.
        let completed = unsafe { self.fence.GetCompletedValue() };

        let (
            unmaps,
            map_batches,
            uploads,
            upload_chunk,
            upload_frame,
            emit_barrier_only,
            retire_tiles,
            retire_resources,
            touched_pages,
            candidate_fence,
        ) = {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            state.reclaim_completed(completed);

            let has_work = !state.pending_unmaps.is_empty()
                || !state.pending_maps.is_empty()
                || !state.pending_uploads.is_empty()
                || !state.pending_retire_tiles.is_empty()
                || !state.pending_retire_resources.is_empty()
                || state.needs_aliasing_barrier;
            if !has_work {
                return Ok(state.signaled_fence);
            }

            // Compute required upload capacity BEFORE mutating any queue or tile state.
            let mut total_upload_bytes = 0u64;
            for u in &state.pending_uploads {
                if !u.data.is_empty() && state.pages.contains_key(&u.title_page) {
                    total_upload_bytes = total_upload_bytes.next_multiple_of(UPLOAD_ALIGN);
                    total_upload_bytes += u.data.len() as u64;
                }
            }
            let need_cmd_list = total_upload_bytes > 0 || state.needs_aliasing_barrier;

            // Pre-allocate fallible resources before draining pending queues so
            // any allocation error leaves `State` 100% intact.
            let upload_frame = if need_cmd_list {
                Some(match state.free_frames.pop() {
                    Some(f) => f,
                    None => UploadFrame::create(&self.device)?,
                })
            } else {
                None
            };
            let upload_chunk = if total_upload_bytes > 0 {
                match state.take_upload_chunk(&self.device, total_upload_bytes) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        if let Some(f) = upload_frame {
                            state.free_frames.push(f);
                        }
                        return Err(e);
                    }
                }
            } else {
                None
            };

            let unmaps = std::mem::take(&mut state.pending_unmaps);
            let raw_maps = std::mem::take(&mut state.pending_maps);
            let raw_uploads = std::mem::take(&mut state.pending_uploads);
            state.pending_by_page.clear();
            let retire_tiles = std::mem::take(&mut state.pending_retire_tiles);
            let retire_resources = std::mem::take(&mut state.pending_retire_resources);
            let emit_barrier_only = state.needs_aliasing_barrier && total_upload_bytes == 0;
            state.needs_aliasing_barrier = false;

            // Group valid pending maps by `(window_id, heap_idx)` and transition
            // surviving local tiles `PendingMap -> MappedOnGpu`.
            let mut grouped_maps: BTreeMap<(u64, usize), Vec<(u32, u32)>> = BTreeMap::new();
            for pm in raw_maps {
                if let Some(window) = state.windows.get_mut(&pm.window_id)
                    && let Some(slot) = window.tiles.get_mut(pm.local_tile as usize)
                    && *slot == TileState::PendingMap
                {
                    *slot = TileState::MappedOnGpu;
                    grouped_maps
                        .entry((pm.window_id, pm.heap_idx))
                        .or_default()
                        .push((pm.local_tile, pm.heap_tile));
                }
            }

            let mut map_batches: Vec<(ID3D12Resource, ID3D12Heap, Vec<(u32, u32)>)> =
                Vec::with_capacity(grouped_maps.len());
            for ((window_id, heap_idx), pairs) in grouped_maps {
                if let (Some(window), Some(Some(heap_chunk))) =
                    (state.windows.get(&window_id), state.heaps.get(heap_idx))
                {
                    map_batches.push((window.resource.clone(), heap_chunk.heap.clone(), pairs));
                }
            }

            // Pack surviving uploads sequentially into the write-only `UploadChunk`
            // and resolve each destination window in O(1) via `canonical_window`.
            let mut resolved_uploads: Vec<(ID3D12Resource, u64, u64, u64)> =
                Vec::with_capacity(raw_uploads.len());
            let mut touched_pages: Vec<u64> = Vec::with_capacity(raw_uploads.len());
            if let Some(ref chunk) = upload_chunk {
                let mut cursor = 0u64;
                for u in raw_uploads {
                    if u.data.is_empty() {
                        continue;
                    }
                    let Some(entry) = state.pages.get(&u.title_page) else {
                        continue;
                    };
                    let canon_id = entry.canonical_window;
                    let fast = state.windows.get(&canon_id).and_then(|w| {
                        let local = w.contains_page(u.title_page)?;
                        (w.tiles[local as usize] == TileState::MappedOnGpu)
                            .then_some((w.resource.clone(), u64::from(local) * TILE_BYTES))
                    });
                    let target = if let Some(hit) = fast {
                        Some(hit)
                    } else {
                        let chosen = state
                            .windows
                            .values()
                            .filter_map(|w| {
                                let local = w.contains_page(u.title_page)?;
                                (w.tiles[local as usize] == TileState::MappedOnGpu)
                                    .then_some((w.id, w.resource.clone(), u64::from(local) * TILE_BYTES))
                            })
                            .min_by_key(|&(id, _, _)| id);
                        if let Some((new_id, res, base)) = chosen {
                            if let Some(e) = state.pages.get_mut(&u.title_page) {
                                e.canonical_window = new_id;
                            }
                            Some((res, base))
                        } else {
                            None
                        }
                    };
                    if let Some((dst_res, tile_base)) = target {
                        let aligned = cursor.next_multiple_of(UPLOAD_ALIGN);
                        let bytes = u.data.len() as u64;
                        debug_assert!(aligned + bytes <= chunk.capacity);
                        // SAFETY: write-only sequential copy into mapped upload chunk.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                u.data.as_ptr(),
                                chunk.ptr.add(aligned as usize),
                                u.data.len(),
                            );
                        }
                        cursor = aligned + bytes;
                        touched_pages.push(u.title_page);
                        resolved_uploads.push((
                            dst_res,
                            tile_base + u64::from(u.intra_offset),
                            aligned,
                            bytes,
                        ));
                    }
                }
            }

            let candidate_fence = state.signaled_fence + 1;
            (
                unmaps,
                map_batches,
                resolved_uploads,
                upload_chunk,
                upload_frame,
                emit_barrier_only,
                retire_tiles,
                retire_resources,
                touched_pages,
                candidate_fence,
            )
        };

        let mut utm_calls = 0u64;
        let mut utm_us = 0u64;

        // 1. Submit NULL unmaps ahead of subsequent mappings/uploads.
        if !unmaps.is_empty() {
            let mut by_res: HashMap<usize, (ID3D12Resource, Vec<u32>)> = HashMap::new();
            for u in unmaps {
                let key = u.resource.as_raw() as usize;
                by_res
                    .entry(key)
                    .or_insert_with(|| (u.resource, Vec::new()))
                    .1
                    .push(u.local_tile);
            }
            for (_, (resource, mut locals)) in by_res {
                let runs = coalesce_local_runs(&mut locals);
                let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = runs
                    .iter()
                    .map(|&(start, _)| D3D12_TILED_RESOURCE_COORDINATE {
                        X: start,
                        Y: 0,
                        Z: 0,
                        Subresource: 0,
                    })
                    .collect();
                let sizes: Vec<D3D12_TILE_REGION_SIZE> = runs
                    .iter()
                    .map(|&(_, count)| D3D12_TILE_REGION_SIZE {
                        NumTiles: count,
                        ..Default::default()
                    })
                    .collect();
                let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; runs.len()];
                let counts: Vec<u32> = runs.iter().map(|&(_, count)| count).collect();
                let began = std::time::Instant::now();
                // SAFETY: `resource` stays alive in `retire_resources` or `windows`;
                // NULL ranges pass `pHeap = None` and `pHeapRangeStartOffsets = None`.
                unsafe {
                    self.queue.UpdateTileMappings(
                        &resource,
                        runs.len() as u32,
                        Some(coords.as_ptr()),
                        Some(sizes.as_ptr()),
                        None,
                        runs.len() as u32,
                        Some(flags.as_ptr()),
                        None,
                        Some(counts.as_ptr()),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
                utm_calls += 1;
                utm_us += began.elapsed().as_micros() as u64;
            }
        }

        // 2. Submit batched tile mappings grouped by `(resource, heap)`.
        for (resource, heap, mut pairs) in map_batches {
            let runs = coalesce_tile_pairs(&mut pairs);
            if runs.is_empty() {
                continue;
            }
            let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = runs
                .iter()
                .map(|&(local, _, _)| D3D12_TILED_RESOURCE_COORDINATE {
                    X: local,
                    Y: 0,
                    Z: 0,
                    Subresource: 0,
                })
                .collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> = runs
                .iter()
                .map(|&(_, _, count)| D3D12_TILE_REGION_SIZE {
                    NumTiles: count,
                    ..Default::default()
                })
                .collect();
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; runs.len()];
            let starts: Vec<u32> = runs.iter().map(|&(_, heap_start, _)| heap_start).collect();
            let counts: Vec<u32> = runs.iter().map(|&(_, _, count)| count).collect();
            let began = std::time::Instant::now();
            // SAFETY: live resource and heap; 1:1 region-to-range counts.
            unsafe {
                self.queue.UpdateTileMappings(
                    &resource,
                    runs.len() as u32,
                    Some(coords.as_ptr()),
                    Some(sizes.as_ptr()),
                    &heap,
                    runs.len() as u32,
                    Some(flags.as_ptr()),
                    Some(starts.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            utm_calls += 1;
            utm_us += began.elapsed().as_micros() as u64;
        }

        // 3. Record and submit uploads + global aliasing barrier if needed.
        let mut cmd_executed = false;
        let mut record_err: Option<windows::core::Error> = None;
        if let Some(ref frame) = upload_frame
            && (!uploads.is_empty() || emit_barrier_only)
        {
            // SAFETY: `frame` came from completed `free_frames` or fresh creation.
            let res = unsafe {
                (|| -> windows::core::Result<()> {
                    frame.allocator.Reset()?;
                    frame.list.Reset(&frame.allocator, None)?;
                    if let Some(ref chunk) = upload_chunk {
                        for &(ref dst_res, dst_off, src_off, bytes) in &uploads {
                            frame.list.CopyBufferRegion(
                                dst_res,
                                dst_off,
                                &chunk.resource,
                                src_off,
                                bytes,
                            );
                        }
                    }
                    frame.list.ResourceBarrier(&[aliasing_barrier(None, None)]);
                    frame.list.Close()?;
                    let cmd: ID3D12CommandList = frame.list.cast()?;
                    cmd_executed = true;
                    self.queue.ExecuteCommandLists(&[Some(cmd)]);
                    Ok(())
                })()
            };
            if let Err(e) = res {
                record_err = Some(e);
            }
        }

        // 4. Signal the queue fence BEFORE committing `signaled_fence` and
        //    attaching retired tiles/resources/frames. Even if upload recording
        //    failed, we still signal so any queued unmaps/retires stay fence-safe.
        let need_signal =
            utm_calls > 0 || cmd_executed || !retire_tiles.is_empty() || !retire_resources.is_empty();
        let signal_res = if need_signal {
            // SAFETY: live queue and fence.
            unsafe { self.queue.Signal(&self.fence, candidate_fence) }
        } else {
            Ok(())
        };

        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        state.update_tile_mappings_calls += utm_calls;
        state.update_tile_mappings_us += utm_us;

        match signal_res {
            Ok(()) if need_signal => {
                state.signaled_fence = candidate_fence;
                for page in touched_pages {
                    if let Some(entry) = state.pages.get_mut(&page) {
                        entry.last_fence = candidate_fence;
                    }
                }
                if let Some(frame) = upload_frame {
                    if cmd_executed {
                        state.in_flight_frames.push_back((candidate_fence, frame));
                    } else {
                        state.free_frames.push(frame);
                    }
                }
                if let Some(chunk) = upload_chunk {
                    if cmd_executed && !uploads.is_empty() {
                        state.in_flight_uploads.push_back((candidate_fence, chunk));
                    } else {
                        state.free_upload_chunks.push(chunk);
                    }
                }
                for (heap_idx, tile_idx) in retire_tiles {
                    state.retired_tiles.push_back((candidate_fence, heap_idx, tile_idx));
                }
                for res in retire_resources {
                    state.retired_resources.push_back((candidate_fence, res));
                }
            }
            Ok(()) => {
                if let Some(frame) = upload_frame {
                    state.free_frames.push(frame);
                }
                if let Some(chunk) = upload_chunk {
                    state.free_upload_chunks.push(chunk);
                }
            }
            Err(sig_err) => {
                // `queue.Signal` failed (device removed): do NOT advance
                // `signaled_fence` (prevents `wait_idle` deadlock) and do NOT
                // drop submitted frames/resources prematurely.
                if let Some(frame) = upload_frame {
                    if cmd_executed {
                        state.abandoned_frames.push(frame);
                    } else {
                        state.free_frames.push(frame);
                    }
                }
                if let Some(chunk) = upload_chunk {
                    if cmd_executed {
                        state.abandoned_uploads.push(chunk);
                    } else {
                        state.free_upload_chunks.push(chunk);
                    }
                }
                state.abandoned_resources.extend(retire_resources);
                // Leave `retire_tiles` in `pending_retire_tiles` (keeps `in_use_tiles > 0`
                // so heaps cannot drop while GPU state is unknown).
                state.pending_retire_tiles.extend(retire_tiles);
                return Err(record_err.unwrap_or(sig_err));
            }
        }

        if let Some(err) = record_err {
            return Err(err);
        }
        Ok(state.signaled_fence)
    }

    /// Explicitly unmaps `title_pages` from every live overlapping window
    /// (`D3D12_TILE_RANGE_FLAG_NULL`) and schedules their physical tiles for
    /// fence-safe reclamation.
    #[cfg(test)]
    pub(crate) fn unmap_pages(&self, title_pages: &[u64]) -> windows::core::Result<u64> {
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            for &title_page in title_pages {
                let Some(entry) = state.pages.remove(&title_page) else {
                    continue;
                };
                let mut detached = 0u64;
                let mut unmaps = Vec::new();
                for window in state.windows.values_mut() {
                    if let Some(local) = window.contains_page(title_page) {
                        match window.tiles[local as usize] {
                            TileState::MappedOnGpu => {
                                window.tiles[local as usize] = TileState::Unmapped;
                                unmaps.push(PendingUnmap {
                                    resource: window.resource.clone(),
                                    local_tile: local,
                                });
                                detached += 1;
                            }
                            TileState::PendingMap => {
                                window.tiles[local as usize] = TileState::Unmapped;
                                detached += 1;
                            }
                            TileState::Unmapped => {}
                        }
                    }
                }
                state.pending_unmaps.extend(unmaps);
                state.mapped_window_tiles = state.mapped_window_tiles.saturating_sub(detached);
                state.pending_retire_tiles.push((entry.heap_idx, entry.tile_idx));
            }
        }
        self.flush()
    }

    /// Releases a reserved window: detaches it from every canonical page and
    /// active UAV list, queues `NULL` tile unmaps for all GPU-mapped tiles, and
    /// schedules orphaned physical tiles and the reserved resource for
    /// fence-safe reclamation.
    pub(crate) fn release_window(&self, window_id: u64) -> windows::core::Result<u64> {
        {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let is_global = state.is_global;
            if let Some(leftover_uav) = state.active_uav_ranges.remove(&window_id) {
                self.adjust_uav_count(is_global, 0, leftover_uav.len() as u32);
                state.prune_unused_uav_pages();
            }
            let Some(window) = state.windows.remove(&window_id) else {
                return Ok(state.signaled_fence);
            };
            let len = u64::from(window.page_count) * TILE_BYTES;
            state.reserved_va_bytes = state.reserved_va_bytes.saturating_sub(len);
            let mut detached = 0u64;
            for (local, &tile_state) in window.tiles.iter().enumerate() {
                if tile_state == TileState::Unmapped {
                    continue;
                }
                detached += 1;
                if tile_state == TileState::MappedOnGpu {
                    state.pending_unmaps.push(PendingUnmap {
                        resource: window.resource.clone(),
                        local_tile: local as u32,
                    });
                }
                let title_page = window.start_page + local as u64;
                let retire = if let Some(entry) = state.pages.get_mut(&title_page) {
                    entry.mapped_windows = entry.mapped_windows.saturating_sub(1);
                    entry.mapped_windows == 0
                } else {
                    false
                };
                if retire && let Some(entry) = state.pages.remove(&title_page) {
                    state.pending_retire_tiles.push((entry.heap_idx, entry.tile_idx));
                }
            }
            state.mapped_window_tiles = state.mapped_window_tiles.saturating_sub(detached);
            state.pending_retire_resources.push(window.resource);
        }
        self.flush()
    }

    /// Reclaims any retired tiles, resources, upload chunks, and empty heaps
    /// whose GPU fence has completed. Returns the number of reclaimed tiles.
    pub(crate) fn reclaim(&self) -> usize {
        // SAFETY: plain fence query on a live fence.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            return 0;
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reclaim_completed(completed)
    }

    /// Waits for all signaled sparse-VA operations to finish on `self.queue`,
    /// then reclaims every completed tile, resource, and empty heap. Never
    /// deadlocks on device removal or unsignaled candidate fences.
    pub(crate) fn wait_idle(&self) -> windows::core::Result<()> {
        let _ = self.flush();
        let _flush_guard = self.flush_lock.lock().unwrap_or_else(|e| e.into_inner());
        let target = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .signaled_fence;
        if target == 0 {
            return Ok(());
        }
        // SAFETY: live device, fence, and event owned by `self` under `flush_lock`.
        unsafe {
            self.device.GetDeviceRemovedReason()?;
            let completed = self.fence.GetCompletedValue();
            if completed == u64::MAX {
                return Err(windows::core::Error::from_hresult(
                    windows::Win32::Foundation::E_FAIL,
                ));
            }
            if completed < target {
                self.fence.SetEventOnCompletion(target, self.event)?;
                let waited = WaitForSingleObject(self.event, INFINITE);
                if waited != WAIT_OBJECT_0 {
                    return Err(windows::core::Error::from_hresult(
                        windows::Win32::Foundation::E_FAIL,
                    ));
                }
            }
        }
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reclaim_completed(target);
        Ok(())
    }

    /// Reads back `[offset, offset + len)` bytes from `resource` via a transient
    /// readback copy on `self.queue`. Used for rare UAV copy-backs on sparse
    /// loose windows and for GPU verification tests.
    pub(crate) fn readback_range(
        &self,
        resource: &ID3D12Resource,
        offset: u64,
        len: usize,
    ) -> windows::core::Result<Vec<u8>> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let _flush_guard = self.flush_lock.lock().unwrap_or_else(|e| e.into_inner());
        let len_u64 = len as u64;
        let (readback_buf, frame, candidate_fence) = {
            let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: plain fence query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if completed != u64::MAX {
                state.reclaim_completed(completed);
            }

            let need_new = state.readback.as_ref().is_none_or(|(_, cap)| *cap < len_u64);
            if need_new {
                let cap = len_u64.next_multiple_of(TILE_BYTES);
                let heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: cap,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                    ..Default::default()
                };
                let mut buf: Option<ID3D12Resource> = None;
                // SAFETY: plain committed readback buffer creation.
                unsafe {
                    self.device.CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut buf,
                    )?;
                }
                state.readback = Some((buf.ok_or_else(windows::core::Error::empty)?, cap));
            }
            let readback_buf = state.readback.as_ref().expect("just created").0.clone();
            let frame = match state.free_frames.pop() {
                Some(f) => f,
                None => UploadFrame::create(&self.device)?,
            };
            let candidate_fence = state.signaled_fence + 1;
            (readback_buf, frame, candidate_fence)
        };

        let mut executed = false;
        let mut signaled = false;
        let mut waited_ok = false;
        // SAFETY: live buffers, allocator, list, queue, fence, and event under `flush_lock`.
        let result = unsafe {
            (|| -> windows::core::Result<Vec<u8>> {
                frame.allocator.Reset()?;
                frame.list.Reset(&frame.allocator, None)?;
                frame.list.CopyBufferRegion(&readback_buf, 0, resource, offset, len_u64);
                frame.list.Close()?;
                let cmd: ID3D12CommandList = frame.list.cast()?;
                executed = true;
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, candidate_fence)?;
                signaled = true;
                self.fence.SetEventOnCompletion(candidate_fence, self.event)?;
                let waited = WaitForSingleObject(self.event, INFINITE);
                if waited != WAIT_OBJECT_0 {
                    return Err(windows::core::Error::from_hresult(
                        windows::Win32::Foundation::E_FAIL,
                    ));
                }
                waited_ok = true;
                let mut ptr = std::ptr::null_mut();
                let range = D3D12_RANGE { Begin: 0, End: len };
                readback_buf.Map(0, Some(&range), Some(&mut ptr))?;
                let bytes = std::slice::from_raw_parts(ptr.cast::<u8>().cast_const(), len).to_vec();
                let write_none = D3D12_RANGE { Begin: 0, End: 0 };
                readback_buf.Unmap(0, Some(&write_none));
                Ok(bytes)
            })()
        };

        let mut state = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if signaled {
            state.signaled_fence = candidate_fence;
        }
        if !executed || waited_ok {
            state.free_frames.push(frame);
        } else if signaled {
            state.in_flight_frames.push_back((candidate_fence, frame));
        } else {
            state.abandoned_frames.push(frame);
        }
        result
    }

    /// Returns a snapshot of live and cumulative [`SparseStats`].
    pub(crate) fn stats(&self) -> SparseStats {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).stats()
    }
}

/// Builds a `D3D12_RESOURCE_BARRIER_TYPE_ALIASING` barrier.
pub(crate) fn aliasing_barrier(
    before: Option<&ID3D12Resource>,
    after: Option<&ID3D12Resource>,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                // SAFETY: borrowed pointers never released by the barrier struct.
                pResourceBefore: ManuallyDrop::new(
                    before.map(|r| unsafe { std::mem::transmute_copy(r) }),
                ),
                pResourceAfter: ManuallyDrop::new(
                    after.map(|r| unsafe { std::mem::transmute_copy(r) }),
                ),
            }),
        },
    }
}

/// Coalesces `(local_tile, heap_tile)` pairs into sorted contiguous runs
/// `(local_start, heap_start, count)` where both indices advance by +1.
fn coalesce_tile_pairs(pairs: &mut Vec<(u32, u32)>) -> Vec<(u32, u32, u32)> {
    pairs.sort_unstable();
    pairs.dedup();
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for &(local, heap) in pairs.iter() {
        match runs.last_mut() {
            Some(last) if last.0 + last.2 == local && last.1 + last.2 == heap => {
                last.2 += 1;
            }
            _ => runs.push((local, heap, 1)),
        }
    }
    runs
}

/// Coalesces local tile indices into sorted contiguous runs `(start, count)`.
fn coalesce_local_runs(locals: &mut Vec<u32>) -> Vec<(u32, u32)> {
    locals.sort_unstable();
    locals.dedup();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &local in locals.iter() {
        match runs.last_mut() {
            Some(last) if last.0 + last.1 == local => {
                last.1 += 1;
            }
            _ => runs.push((local, 1)),
        }
    }
    runs
}

// ---------------------------------------------------------------------------
// Global host integration for `shadow.rs` (`VICTORY_SPARSE_WINDOWS=1`)
// ---------------------------------------------------------------------------

static GLOBAL: OnceLock<Option<Arc<SparseVa>>> = OnceLock::new();
static CREATED_WINDOWS: AtomicU64 = AtomicU64::new(0);

/// Returns the process-wide [`SparseVa`] attached to [`crate::host::host`], if
/// `VICTORY_SPARSE_WINDOWS=1` is active and the adapter supports Tier 2+.
fn global() -> Option<&'static Arc<SparseVa>> {
    if !wanted() {
        return None;
    }
    GLOBAL
        .get_or_init(|| {
            let host = crate::host::host()?;
            match SparseVa::new_with_mode(
                host.device.clone(),
                host.queue().clone(),
                SparseConfig::default(),
                true,
            ) {
                Ok(engine) => Some(Arc::new(engine)),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "sparse VA engine could not start; falling back to standard loose windows"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Per-`Shadow` state for a sparse virtual-address loose window.
/// Deliberately does NOT hold a duplicate `ID3D12Resource` COM reference so the
/// resource refcount stays at `2` (`Shadow::buffer` + `State::windows[id].resource`)
/// when idle, allowing `shadow::let_go_idle` to detect unreferenced windows cleanly.
pub(crate) struct SparseShadow {
    engine: Arc<SparseVa>,
    window_id: u64,
    gpu_va: u64,
}

impl SparseShadow {
    #[cfg(test)]
    pub(crate) fn from_handle(engine: Arc<SparseVa>, handle: SparseWindowHandle) -> (ID3D12Resource, Self) {
        (
            handle.resource,
            Self {
                engine,
                window_id: handle.id,
                gpu_va: handle.gpu_va,
            },
        )
    }

    /// Host GPU virtual address for `Shadow::gpu`.
    pub(crate) fn gpu_va(&self) -> u64 {
        self.gpu_va
    }

    /// Refreshes `[from, to)` from title memory into canonical shared 64 KiB tiles.
    pub(crate) fn update(
        &self,
        from: usize,
        to: usize,
        now: u64,
        written: Option<&[(usize, u64)]>,
    ) -> usize {
        match self.engine.stage_title_range(from, to, now, written) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!(error = %e, from, to, "sparse VA stage_title_range failed");
                0
            }
        }
    }

    /// Whether every 64 KiB page in `[from, to)` is backed, not active-UAV in
    /// any overlapping window, and clean in the write-watch journal.
    pub(crate) fn nothing_new(&self, from: usize, to: usize) -> bool {
        if self.engine.has_active_uav_in(from, to) {
            return false;
        }
        let Some((base, copied_at)) = self.engine.copied_at_range(from, to) else {
            return false;
        };
        crate::watch::nothing_new(from, to, base, &copied_at, 0)
    }

    /// Declares an active GPU UAV write on `[from, to)` with single-read
    /// snapshot capture and cross-window active-range subtraction.
    pub(crate) fn declare_uav_write(&self, from: usize, to: usize) {
        if let Err(e) = self.engine.declare_uav_write(self.window_id, from, to) {
            tracing::warn!(error = %e, from, to, "sparse VA declare_uav_write failed");
        }
    }

    /// Updates ONLY canonical pre-shader reference snapshots for placed copy
    /// bytes (never queues a GPU upload).
    pub(crate) fn placed_copy(&self, address: usize, bytes: &[u8]) {
        self.engine.update_placed_reference(address, bytes);
    }

    /// Reads back `[from, to)` from `resource` and merges GPU-written bytes
    /// into title memory (title wins on concurrent same-byte writes; reference
    /// updated to actual GPU byte `g`).
    pub(crate) fn copy_back_range(
        &self,
        resource: &ID3D12Resource,
        start: u64,
        from: usize,
        to: usize,
        put: impl Fn(usize, &[u8], &[u8]) -> usize,
    ) -> usize {
        self.engine.copy_back_range(resource, start, from, to, put)
    }

    /// Releases completed UAV ranges for this window.
    pub(crate) fn finish_uav_ranges(&self, freed_ranges: &[(usize, usize)]) {
        self.engine.finish_uav_ranges(self.window_id, freed_ranges);
    }

    /// Whether any overlapping window has an active UAV write intersecting `[from, to)`.
    pub(crate) fn has_active_uav_in(&self, from: usize, to: usize) -> bool {
        self.engine.has_active_uav_in(from, to)
    }

    /// Coalesced active UAV byte ranges intersecting `[from, to)` across all windows.
    pub(crate) fn active_uav_ranges_in(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.engine.active_uav_ranges_in(from, to)
    }
}

impl Drop for SparseShadow {
    fn drop(&mut self) {
        let _ = self.engine.release_window(self.window_id);
    }
}

/// Allocates a sparse virtual-address loose window for `[start, start + len)` if
/// `VICTORY_SPARSE_WINDOWS=1` is enabled and supported. Returns `None` to fall
/// back cleanly to standard loose-window allocation.
pub(crate) fn create_loose_window(start: u64, len: u64) -> Option<(ID3D12Resource, SparseShadow)> {
    let engine = global()?;
    let handle = match engine.create_window(start, len) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(
                error = %e,
                start = format_args!("{start:#x}"),
                len,
                "CreateReservedResource failed for sparse window; falling back"
            );
            return None;
        }
    };
    crate::host::name(
        &handle.resource,
        || format!("title memory {start:#x}+{len:#x} sparse VA"),
    );
    let n = CREATED_WINDOWS.fetch_add(1, Ordering::Relaxed) + 1;
    if n == 1 || n.is_multiple_of(256) {
        let s = engine.stats();
        tracing::info!(
            windows = s.live_windows,
            reserved_mb = s.reserved_va_bytes >> 20,
            committed_kb = s.committed_bytes >> 10,
            dedup_kb = s.deduplicated_bytes >> 10,
            heaps = s.heap_count,
            "sparse VA loose windows active"
        );
    }
    Some((
        handle.resource,
        SparseShadow {
            engine: engine.clone(),
            window_id: handle.id,
            gpu_va: handle.gpu_va,
        },
    ))
}

/// Flushes pending tile mappings and staged uploads ahead of submission lists.
pub(crate) fn flush_pending() {
    let Some(engine) = global() else {
        return;
    };
    if let Err(e) = engine.flush() {
        tracing::warn!(error = %e, "sparse VA flush_pending failed");
    }
}

/// Reclaims fence-completed retired tiles and empty heaps during idle sweeps.
pub(crate) fn reclaim_idle() {
    let Some(engine) = global() else {
        return;
    };
    engine.reclaim();
}

/// Returns current process-wide sparse VA stats, if enabled and initialized.
#[cfg(any(test, feature = "diagnostics"))]
#[allow(dead_code)]
pub(crate) fn global_stats() -> Option<SparseStats> {
    global().map(|e| e.stats())
}

// ---------------------------------------------------------------------------
// Hardware + WARP verification tests & benchmark
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{compile, transition, warp_adapter};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::core::s;

    /// Compute shader reading three independent byte-address SRVs (Window A,
    /// Window B, Window C) at configurable offsets and writing their loaded
    /// dwords to an output UAV for bit-exact hardware verification.
    const READ_THREE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), SRV(t2), UAV(u0)"
cbuffer Params : register(b0) {
    uint off_a;
    uint off_b;
    uint off_c;
    uint count;
};
ByteAddressBuffer win_a : register(t0);
ByteAddressBuffer win_b : register(t1);
ByteAddressBuffer win_c : register(t2);
RWByteAddressBuffer out_buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint i = id.x * 4u;
    uint a = win_a.Load(off_a + i);
    uint b = win_b.Load(off_b + i);
    uint c = win_c.Load(off_c + i);
    out_buf.Store(id.x * 12u + 0u, a);
    out_buf.Store(id.x * 12u + 4u, b);
    out_buf.Store(id.x * 12u + 8u, c);
}
"#;

    /// Compute shader writing a deterministic pattern via UAV to one view.
    const WRITE_UAV_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0)"
cbuffer Params : register(b0) {
    uint dst_off;
    uint count;
    uint seed;
};
RWByteAddressBuffer target : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint v = seed ^ (id.x * 2654435761u) ^ 0xA55A1337u;
    target.Store(dst_off + id.x * 4u, v);
}
"#;

    /// Compute shader transforming dwords from one view for throughput benchmarks.
    const SUM_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Params : register(b0) {
    uint off;
    uint count;
};
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint v = src.Load(off + id.x * 4u);
    dst.Store(id.x * 4u, v ^ 0x55AA55AAu);
}
"#;

    struct TestRig {
        label: &'static str,
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        event: HANDLE,
        submitted: u64,
        info: Option<ID3D12InfoQueue>,
        tier: i32,
    }

    impl Drop for TestRig {
        fn drop(&mut self) {
            // SAFETY: owned Win32 event handle.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }

    impl TestRig {
        fn new(hardware: bool) -> Option<Self> {
            // SAFETY: isolated test device creation with optional debug layer.
            unsafe {
                if std::env::var_os("VICTORY_D3D12_DEBUG").is_some_and(|v| v != "0") {
                    let mut layer: Option<ID3D12Debug> = None;
                    if D3D12GetDebugInterface(&mut layer).is_ok()
                        && let Some(l) = layer
                    {
                        l.EnableDebugLayer();
                    }
                }
                let device: ID3D12Device = if hardware {
                    let mut dev: Option<ID3D12Device> = None;
                    if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                        return None;
                    }
                    dev?
                } else {
                    let adapter = warp_adapter().ok()?;
                    let mut dev: Option<ID3D12Device> = None;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).ok()?;
                    dev?
                };
                let tier = tiled_resources_tier(&device);
                let info = device.cast::<ID3D12InfoQueue>().ok();
                eprintln!(
                    "[sparse_va] {}: TiledResourcesTier={}, debug_layer={}",
                    if hardware { "HW" } else { "WARP" },
                    tier,
                    info.is_some()
                );
                if tier < D3D12_TILED_RESOURCES_TIER_2.0 {
                    return None;
                }
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Self {
                    label: if hardware { "HW" } else { "WARP" },
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    event,
                    submitted: 0,
                    info,
                    tier,
                })
            }
        }

        fn check_debug(&self, step: &str) {
            // SAFETY: plain device-removed and InfoQueue checks.
            unsafe {
                let removed = self.device.GetDeviceRemovedReason();
                let mut first_err: Option<String> = None;
                if let Some(q) = &self.info {
                    for i in 0..q.GetNumStoredMessages() {
                        let mut size = 0;
                        if q.GetMessage(i, None, &mut size).is_err() || size == 0 {
                            continue;
                        }
                        let mut buf = vec![0u64; size.div_ceil(8)];
                        let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if q.GetMessage(i, Some(msg), &mut size).is_ok() {
                            let m = &*msg;
                            let bytes =
                                std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                            let text = String::from_utf8_lossy(bytes)
                                .trim_end_matches('\0')
                                .to_owned();
                            if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                                eprintln!(
                                    "[sparse_va {}] D3D12 ERROR (sev {}, id {}): {text}",
                                    self.label, m.Severity.0, m.ID.0
                                );
                                if first_err.is_none() {
                                    first_err = Some(text);
                                }
                            }
                        }
                    }
                    q.ClearStoredMessages();
                }
                assert!(
                    removed.is_ok(),
                    "[{}] {step}: device removed: {removed:?}",
                    self.label
                );
                assert!(
                    first_err.is_none(),
                    "[{}] {step}: D3D12 debug error: {}",
                    self.label,
                    first_err.unwrap_or_default()
                );
            }
        }

        fn make_compute_pso(&self, hlsl: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
            let code = compile(hlsl, s!("main"), s!("cs_5_1")).expect("compile compute shader");
            // SAFETY: valid compiled DXBC bytecode with embedded root signature.
            unsafe {
                let root: ID3D12RootSignature =
                    self.device.CreateRootSignature(0, &code).expect("root signature");
                let pso: ID3D12PipelineState = self
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .expect("compute PSO");
                (root, pso)
            }
        }

        fn make_buffer(
            &self,
            heap_type: D3D12_HEAP_TYPE,
            flags: D3D12_RESOURCE_FLAGS,
            state: D3D12_RESOURCE_STATES,
            bytes: u64,
        ) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: plain committed buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                    .expect("create buffer");
            }
            out.expect("buffer")
        }

        fn run(&mut self, step: &str, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
            record(&self.list);
            self.submitted += 1;
            // SAFETY: isolated test list, queue, fence, and event.
            unsafe {
                self.list.Close().expect("close list");
                let cmd: ID3D12CommandList = self.list.cast().expect("cast list");
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).expect("signal");
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .expect("set event");
                let timeout = if self.label == "HW" { 5_000 } else { 30_000 };
                let waited = WaitForSingleObject(self.event, timeout);
                assert_eq!(waited, WAIT_OBJECT_0, "[{}] {step}: GPU wait timed out", self.label);
                self.allocator.Reset().expect("reset allocator");
                self.list.Reset(&self.allocator, None).expect("reset list");
            }
            self.check_debug(step);
        }

        fn read_u32s(&self, buf: &ID3D12Resource, count: usize) -> Vec<u32> {
            let bytes = count * 4;
            let mut out = vec![0u32; count];
            // SAFETY: readback buffer populated by a completed GPU submission.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                let range = D3D12_RANGE { Begin: 0, End: bytes };
                buf.Map(0, Some(&range), Some(&mut ptr)).expect("map readback");
                std::ptr::copy_nonoverlapping(ptr.cast::<u32>(), out.as_mut_ptr(), count);
                let write_none = D3D12_RANGE { Begin: 0, End: 0 };
                buf.Unmap(0, Some(&write_none));
            }
            out
        }
    }

    fn pattern_bytes(len: usize, salt: u32) -> Vec<u8> {
        let mut out = vec![0u8; len];
        for (i, chunk) in out.chunks_exact_mut(4).enumerate() {
            let v = (i as u32).wrapping_mul(0x9E37_79B1) ^ salt.wrapping_mul(0x85EB_CA6B);
            chunk.copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn run_overlapping_views_suite(rig: &mut TestRig) {
        let engine = SparseVa::new(
            rig.device.clone(),
            rig.queue.clone(),
            SparseConfig {
                heap_tiles: 32,
                keep_one_heap: false,
            },
        )
        .expect("SparseVa::new");
        rig.check_debug("init SparseVa");

        // Window A: 4 MiB root view [0x1000_0000, 0x1040_0000) (64 tiles)
        let win_a = engine
            .create_window(0x1000_0000, 4 << 20)
            .expect("create win_a");
        // Window B: 256 KiB sized view [0x1002_0000, 0x1006_0000) (4 tiles, overlaps A[2..6])
        let win_b = engine
            .create_window(0x1002_0000, 256 << 10)
            .expect("create win_b");
        assert_ne!(win_a.gpu_va, 0);
        assert_eq!(win_a.gpu_va % TILE_BYTES, 0);
        assert_ne!(win_b.gpu_va, 0);
        assert_eq!(win_b.gpu_va % TILE_BYTES, 0);

        let s0 = engine.stats();
        assert_eq!(s0.committed_tiles, 0);
        assert_eq!(s0.heap_count, 0);
        assert_eq!(s0.reserved_va_bytes, (4 << 20) + (256 << 10));

        let pat_2 = pattern_bytes(PAGE, 0x1111);
        let pat_5 = pattern_bytes(PAGE, 0x2222);
        let pat_63 = pattern_bytes(PAGE, 0x3333);
        engine.upload_range(0x1002_0000, &pat_2).expect("upload tile 2");
        engine.upload_range(0x1005_0000, &pat_5).expect("upload tile 5");
        engine.upload_range(0x103F_0000, &pat_63).expect("upload tile 63");
        engine.flush().expect("flush uploads");
        rig.check_debug("after initial flush");

        // Window C created AFTER upload: 8 MiB crossing [0x103E_0000, 0x10BE_0000).
        let win_c = engine
            .create_window(0x103E_0000, 8 << 20)
            .expect("create win_c");
        engine.flush().expect("flush win_c bind");
        rig.check_debug("after win_c bind");

        let s1 = engine.stats();
        assert_eq!(s1.committed_tiles, 3, "must allocate exactly 3 physical tiles");
        assert_eq!(s1.committed_bytes, 3 * TILE_BYTES);
        assert_eq!(s1.mapped_window_tiles, 6);
        assert_eq!(s1.deduplicated_bytes, 3 * TILE_BYTES);

        let (read_root, read_pso) = rig.make_compute_pso(READ_THREE_HLSL);
        let dwords = 256u32;
        let out_bytes = u64::from(dwords) * 12;
        let out_uav = rig.make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            out_bytes,
        );
        let out_rb = rig.make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            out_bytes,
        );

        rig.run("shader read overlapping views A/B/C", |list| {
            let params = [0x2_0000u32, 0u32, 0x1_0000u32, dwords];
            // SAFETY: live resources, root signature, and PSO.
            unsafe {
                list.SetComputeRootSignature(&read_root);
                list.SetPipelineState(&read_pso);
                list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, win_a.gpu_va);
                list.SetComputeRootShaderResourceView(2, win_b.gpu_va);
                list.SetComputeRootShaderResourceView(3, win_c.gpu_va);
                list.SetComputeRootUnorderedAccessView(4, out_uav.GetGPUVirtualAddress());
                list.Dispatch(dwords.div_ceil(64), 1, 1);
                list.ResourceBarrier(&[transition(
                    &out_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&out_rb, 0, &out_uav, 0, out_bytes);
                list.ResourceBarrier(&[transition(
                    &out_uav,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        });
        let read = rig.read_u32s(&out_rb, (dwords * 3) as usize);
        for i in 0..dwords as usize {
            let expect_2 = u32::from_le_bytes(pat_2[i * 4..i * 4 + 4].try_into().unwrap());
            let expect_63 = u32::from_le_bytes(pat_63[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(read[i * 3], expect_2, "A@tile2 mismatch at dword {i}");
            assert_eq!(read[i * 3 + 1], expect_2, "B@tile0 alias mismatch at dword {i}");
            assert_eq!(read[i * 3 + 2], expect_63, "C@tile1 post-upload alias mismatch at dword {i}");
        }

        let tail_a = engine
            .readback_range(&win_a.resource, 5 * TILE_BYTES + 65_024, 512)
            .expect("readback A tile 5");
        let tail_b = engine
            .readback_range(&win_b.resource, 3 * TILE_BYTES + 65_024, 512)
            .expect("readback B tile 3");
        assert_eq!(tail_a, pat_5[65_024..65_536]);
        assert_eq!(tail_b, pat_5[65_024..65_536]);

        // Hardware-coherent GPU UAV write via Window A -> SRV read via Window B.
        let (write_root, write_pso) = rig.make_compute_pso(WRITE_UAV_HLSL);
        let seed = 0xCAFE_BABEu32;
        rig.run("GPU UAV write on A -> coherent SRV read on B", |list| {
            let w_params = [0x2_0000u32, dwords, seed];
            let r_params = [0x2_0000u32, 0u32, 0x1_0000u32, dwords];
            // SAFETY: live reserved resources aliasing physical tile 0x1002.
            unsafe {
                list.SetComputeRootSignature(&write_root);
                list.SetPipelineState(&write_pso);
                list.SetComputeRoot32BitConstants(0, 3, w_params.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, win_a.gpu_va);
                list.Dispatch(dwords.div_ceil(64), 1, 1);
                list.ResourceBarrier(&[
                    D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: ManuallyDrop::new(Some(std::mem::transmute_copy(
                                    &win_a.resource,
                                ))),
                            }),
                        },
                    },
                    transition(
                        &win_a.resource,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    ),
                    aliasing_barrier(Some(&win_a.resource), Some(&win_b.resource)),
                ]);
                list.SetComputeRootSignature(&read_root);
                list.SetPipelineState(&read_pso);
                list.SetComputeRoot32BitConstants(0, 4, r_params.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, win_a.gpu_va);
                list.SetComputeRootShaderResourceView(2, win_b.gpu_va);
                list.SetComputeRootShaderResourceView(3, win_c.gpu_va);
                list.SetComputeRootUnorderedAccessView(4, out_uav.GetGPUVirtualAddress());
                list.Dispatch(dwords.div_ceil(64), 1, 1);
                list.ResourceBarrier(&[transition(
                    &out_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&out_rb, 0, &out_uav, 0, out_bytes);
            }
        });
        let aliased = rig.read_u32s(&out_rb, (dwords * 3) as usize);
        for i in 0..dwords as usize {
            let expect = seed ^ ((i as u32).wrapping_mul(2_654_435_761)) ^ 0xA55A_1337u32;
            assert_eq!(aliased[i * 3], expect, "A GPU write mismatch at {i}");
            assert_eq!(
                aliased[i * 3 + 1],
                expect,
                "B hardware-coherent alias failed to observe A's UAV write at {i}"
            );
        }

        engine.release_window(win_b.id).expect("release win_b");
        engine.release_window(win_c.id).expect("release win_c");
        engine.release_window(win_a.id).expect("release win_a");
        engine.wait_idle().expect("wait_idle");
        rig.check_debug("after overlapping suite teardown");
        let s_end = engine.stats();
        assert_eq!(s_end.live_windows, 0);
        assert_eq!(s_end.committed_tiles, 0);
        assert_eq!(s_end.heap_count, 0);
        assert_eq!(s_end.retired_tiles_pending, 0);
    }

    fn run_sparse_gaps_suite(rig: &mut TestRig) {
        let engine = SparseVa::new(
            rig.device.clone(),
            rig.queue.clone(),
            SparseConfig {
                heap_tiles: 16,
                keep_one_heap: false,
            },
        )
        .expect("SparseVa::new");

        let win = engine
            .create_window(0x2000_0000, 8 << 20)
            .expect("create 8 MiB window");
        assert_eq!(engine.stats().committed_tiles, 0);
        assert_eq!(engine.stats().heap_count, 0);

        let pat_first = pattern_bytes(PAGE, 0xFACE);
        let pat_last = pattern_bytes(PAGE, 0xBEEF);
        engine.upload_range(0x2000_0000, &pat_first).expect("upload tile 0");
        engine.upload_range(0x207F_0000, &pat_last).expect("upload tile 127");
        engine.flush().expect("flush");
        rig.check_debug("sparse gaps upload");

        let s = engine.stats();
        assert_eq!(s.committed_tiles, 2);
        assert_eq!(s.committed_bytes, 2 * TILE_BYTES);
        assert_eq!(s.reserved_va_bytes, 8 << 20);

        let (read_root, read_pso) = rig.make_compute_pso(READ_THREE_HLSL);
        let dwords = 128u32;
        let out_bytes = u64::from(dwords) * 12;
        let out_uav = rig.make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            out_bytes,
        );
        let out_rb = rig.make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            out_bytes,
        );
        rig.run("shader read tile 0, gap tile 64, tile 127", |list| {
            let params = [0u32, (64 * TILE_BYTES) as u32, (127 * TILE_BYTES) as u32, dwords];
            // SAFETY: live resources; Tier 2+ reads NULL tile 64 as zero.
            unsafe {
                list.SetComputeRootSignature(&read_root);
                list.SetPipelineState(&read_pso);
                list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, win.gpu_va);
                list.SetComputeRootShaderResourceView(2, win.gpu_va);
                list.SetComputeRootShaderResourceView(3, win.gpu_va);
                list.SetComputeRootUnorderedAccessView(4, out_uav.GetGPUVirtualAddress());
                list.Dispatch(dwords.div_ceil(64), 1, 1);
                list.ResourceBarrier(&[transition(
                    &out_uav,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&out_rb, 0, &out_uav, 0, out_bytes);
            }
        });
        let read = rig.read_u32s(&out_rb, (dwords * 3) as usize);
        for i in 0..dwords as usize {
            let expect_0 = u32::from_le_bytes(pat_first[i * 4..i * 4 + 4].try_into().unwrap());
            let expect_127 = u32::from_le_bytes(pat_last[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(read[i * 3], expect_0, "tile 0 mismatch at {i}");
            assert_eq!(read[i * 3 + 1], 0, "NULL gap tile 64 must read zero at {i}");
            assert_eq!(read[i * 3 + 2], expect_127, "tile 127 mismatch at {i}");
        }

        let cross = vec![0x5Au8; 256];
        engine
            .upload_range(0x2000_FF80, &cross)
            .expect("cross-page upload");
        engine.flush().expect("flush cross-page");
        assert_eq!(engine.stats().committed_tiles, 3);
        let rb = engine
            .readback_range(&win.resource, 0xFF80, 512)
            .expect("readback cross-page");
        assert!(rb[..256].iter().all(|&b| b == 0x5A), "cross-page written bytes");
        assert!(
            rb[256..].iter().all(|&b| b == 0),
            "uncovered part of newly committed tile 1 must be zeroed"
        );

        engine
            .ensure_committed_zeroed(0x2002_0000, TILE_BYTES)
            .expect("ensure_committed_zeroed tile 2");
        engine.flush().expect("flush zeroed tile 2");
        assert_eq!(engine.stats().committed_tiles, 4);
        let zeroed_rb = engine
            .readback_range(&win.resource, 2 * TILE_BYTES, 128)
            .expect("readback zeroed tile 2");
        assert!(zeroed_rb.iter().all(|&b| b == 0));

        engine.release_window(win.id).expect("release");
        engine.wait_idle().expect("wait_idle");
        rig.check_debug("sparse gaps teardown");
    }

    fn run_reclamation_and_recycling_suite(rig: &mut TestRig) {
        let engine = SparseVa::new(
            rig.device.clone(),
            rig.queue.clone(),
            SparseConfig {
                heap_tiles: 4,
                keep_one_heap: false,
            },
        )
        .expect("SparseVa::new");

        let win_a = engine
            .create_window(0x3000_0000, 4 * TILE_BYTES)
            .expect("win_a");
        let win_b = engine
            .create_window(0x3000_0000, 2 * TILE_BYTES)
            .expect("win_b");
        let win_c = engine
            .create_window(0x3004_0000, 2 * TILE_BYTES)
            .expect("win_c");

        for p in 0..6u64 {
            let fill = vec![(0xA0 + p) as u8; PAGE];
            engine
                .upload_range(0x3000_0000 + p * TILE_BYTES, &fill)
                .expect("upload tile");
        }
        engine.flush().expect("flush 6 tiles");
        rig.check_debug("after 6 tiles across 2 heaps");
        let s = engine.stats();
        assert_eq!(s.committed_tiles, 6);
        assert_eq!(s.heap_count, 2);

        engine.release_window(win_b.id).expect("release win_b");
        engine.wait_idle().expect("wait_idle after win_b");
        rig.check_debug("after releasing win_b");
        assert_eq!(engine.stats().committed_tiles, 6);
        let check_a0 = engine.readback_range(&win_a.resource, 0, 64).expect("read A0");
        assert!(check_a0.iter().all(|&b| b == 0xA0));

        engine
            .unmap_pages(&[0x3002, 0x3003])
            .expect("unmap 2 pages");
        engine.wait_idle().expect("wait_idle after unmap");
        rig.check_debug("after explicit unmap_pages");
        let s_unmap = engine.stats();
        assert_eq!(s_unmap.committed_tiles, 4);
        assert_eq!(s_unmap.tiles_reclaimed_total, 2);
        let unmapped_rb = engine
            .readback_range(&win_a.resource, 2 * TILE_BYTES, 128)
            .expect("read unmapped");
        assert!(unmapped_rb.iter().all(|&b| b == 0));

        let fresh = vec![0x3Cu8; 32];
        engine
            .upload_range(0x3002_0000, &fresh)
            .expect("reuse tile 2");
        engine
            .upload_range(0x3003_0000, &fresh)
            .expect("reuse tile 3");
        engine.flush().expect("flush reused tiles");
        rig.check_debug("after tile recycling");
        assert_eq!(engine.stats().heap_count, 2, "must recycle freed tiles without a 3rd heap");
        let reused_rb = engine
            .readback_range(&win_a.resource, 2 * TILE_BYTES, 128)
            .expect("read reused");
        assert!(reused_rb[..32].iter().all(|&b| b == 0x3C));
        assert!(
            reused_rb[32..].iter().all(|&b| b == 0),
            "recycled tile leaked previous owner bytes: {:?}",
            &reused_rb[32..48]
        );

        engine.release_window(win_a.id).expect("release win_a");
        engine.release_window(win_c.id).expect("release win_c");
        engine.wait_idle().expect("wait_idle final");
        rig.check_debug("after full reclamation");
        let s_final = engine.stats();
        assert_eq!(s_final.live_windows, 0);
        assert_eq!(s_final.committed_tiles, 0);
        assert_eq!(s_final.retired_tiles_pending, 0);
        assert_eq!(s_final.heap_count, 0);
        assert_eq!(s_final.tiles_reclaimed_total, 8);
        assert_eq!(s_final.heaps_reclaimed_total, 2);
    }

    fn run_coalesced_upload_and_early_drop_suite(rig: &mut TestRig) {
        let engine = SparseVa::new(
            rig.device.clone(),
            rig.queue.clone(),
            SparseConfig {
                heap_tiles: 8,
                keep_one_heap: false,
            },
        )
        .expect("SparseVa::new");

        let win_a = engine
            .create_window(0x5000_0000, 2 * TILE_BYTES)
            .expect("win_a");
        let win_b = engine
            .create_window(0x5000_0000, 2 * TILE_BYTES)
            .expect("win_b");

        engine
            .upload_range(0x5000_0000, &vec![0x01u8; PAGE])
            .expect("init tile 0");
        engine.flush().expect("flush init");

        // Coalesce overlapping sub-ranges purely in cached CPU Vec<u8> (never WC read):
        // [0..128]=0xAA, [200..260]=0xDD, [64..220]=0xBB (bridges both), [16..48]=0xCC.
        engine.upload_range(0x5000_0000, &[0xAAu8; 128]).expect("p1");
        engine.upload_range(0x5000_0000 + 200, &[0xDDu8; 60]).expect("p2");
        engine.upload_range(0x5000_0000 + 64, &[0xBBu8; 156]).expect("p3 bridge");
        engine.upload_range(0x5000_0000 + 16, &[0xCCu8; 32]).expect("p4 overwrite");

        // Drop canonical Window A BEFORE flush: surviving Window B must receive all bytes.
        engine.release_window(win_a.id).expect("early release win_a");
        engine.flush().expect("flush into surviving win_b");
        rig.check_debug("coalesced upload + early canonical window drop");

        let rb = engine
            .readback_range(&win_b.resource, 0, 300)
            .expect("readback win_b");
        assert!(rb[0..16].iter().all(|&b| b == 0xAA), "0..16 mismatch");
        assert!(rb[16..48].iter().all(|&b| b == 0xCC), "16..48 overwrite mismatch");
        assert!(rb[48..64].iter().all(|&b| b == 0xAA), "48..64 mismatch");
        assert!(rb[64..220].iter().all(|&b| b == 0xBB), "64..220 bridge mismatch");
        assert!(rb[220..260].iter().all(|&b| b == 0xDD), "220..260 tail mismatch");
        assert!(rb[260..300].iter().all(|&b| b == 0x01), "260..300 untouched mismatch");

        engine.release_window(win_b.id).expect("release win_b");
        engine.wait_idle().expect("wait_idle coalesced");
        rig.check_debug("coalesced teardown");
    }

    fn run_benchmark_vs_loose_windows(rig: &mut TestRig) {
        const CELLS: u64 = 16;
        const CELL_SIZE: u64 = 4 << 20;
        const BASE: u64 = 0x4000_0000;
        const TOUCHED_PER_CELL: u64 = 4;
        const FRAMES: u32 = 5;
        let read_passes: u32 = if rig.label == "HW" { 16 } else { 2 };

        let mut windows_spec: Vec<(u64, u64)> = Vec::new();
        for c in 0..CELLS {
            windows_spec.push((BASE + c * CELL_SIZE, CELL_SIZE));
            windows_spec.push((BASE + c * CELL_SIZE, 512 << 10));
            windows_spec.push((BASE + c * CELL_SIZE + (3 << 20) + (512 << 10), 512 << 10));
        }
        for c in (0..CELLS - 1).step_by(2) {
            windows_spec.push((BASE + c * CELL_SIZE, 2 * CELL_SIZE));
        }
        let touched_pages: Vec<u64> = (0..CELLS)
            .flat_map(|c| {
                let cell_page = (BASE + c * CELL_SIZE) / TILE_BYTES;
                [cell_page, cell_page + 1, cell_page + 62, cell_page + 63]
            })
            .collect();
        let page_payload = pattern_bytes(PAGE, 0x7777);
        let (sum_root, sum_pso) = rig.make_compute_pso(SUM_HLSL);
        let out_uav = rig.make_buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            TILE_BYTES,
        );
        let out_rb = rig.make_buffer(
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            TILE_BYTES,
        );

        // --- Baseline: separate committed L0 write-back shared buffers ---
        let base_create_start = std::time::Instant::now();
        let mut base_windows: Vec<(u64, u64, ID3D12Resource, *mut u8)> = Vec::new();
        let mut base_committed_bytes = 0u64;
        for &(start, len) in &windows_spec {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: len,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            let mut ptr = std::ptr::null_mut();
            // SAFETY: baseline committed L0 buffer matching `Host::shared_buffer`.
            unsafe {
                rig.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut res,
                    )
                    .expect("baseline create");
                let r = res.expect("baseline res");
                r.Map(0, None, Some(&mut ptr)).expect("baseline map");
                base_windows.push((start, len, r, ptr.cast()));
            }
            base_committed_bytes += len;
        }
        let base_create_us = base_create_start.elapsed().as_micros() as u64;

        let base_run_start = std::time::Instant::now();
        let mut base_bytes_copied = 0u64;
        for frame in 0..FRAMES {
            let dirty_pages: &[u64] = if frame == 0 {
                &touched_pages
            } else {
                &touched_pages[..16]
            };
            for &(start, len, _, ptr) in &base_windows {
                let end = start + len;
                for &page in dirty_pages {
                    let addr = page * TILE_BYTES;
                    if addr >= start && addr + TILE_BYTES <= end {
                        // SAFETY: inside mapped baseline window.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                page_payload.as_ptr(),
                                ptr.add((addr - start) as usize),
                                PAGE,
                            );
                        }
                        base_bytes_copied += TILE_BYTES;
                    }
                }
            }
            rig.run("baseline shader read", |list| {
                // SAFETY: live baseline buffers and compute pipeline.
                unsafe {
                    list.SetComputeRootSignature(&sum_root);
                    list.SetPipelineState(&sum_pso);
                    list.SetComputeRootUnorderedAccessView(2, out_uav.GetGPUVirtualAddress());
                    for _ in 0..read_passes {
                        for (_, _, res, _) in &base_windows {
                            let params = [0u32, (PAGE / 4) as u32];
                            list.SetComputeRoot32BitConstants(0, 2, params.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, res.GetGPUVirtualAddress());
                            list.Dispatch((PAGE as u32 / 4).div_ceil(64), 1, 1);
                        }
                    }
                    list.ResourceBarrier(&[transition(
                        &out_uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    list.CopyBufferRegion(&out_rb, 0, &out_uav, 0, TILE_BYTES);
                    list.ResourceBarrier(&[transition(
                        &out_uav,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            });
        }
        let base_run_us = base_run_start.elapsed().as_micros() as u64;
        let base_checksum = rig.read_u32s(&out_rb, 64);
        drop(base_windows);

        // --- Sparse VA: unified hardware virtual page table ---
        let engine = SparseVa::new(
            rig.device.clone(),
            rig.queue.clone(),
            SparseConfig {
                heap_tiles: 256,
                keep_one_heap: false,
            },
        )
        .expect("SparseVa::new");
        let sparse_create_start = std::time::Instant::now();
        let sparse_windows: Vec<SparseWindowHandle> = windows_spec
            .iter()
            .map(|&(start, len)| engine.create_window(start, len).expect("sparse create"))
            .collect();
        let sparse_create_us = sparse_create_start.elapsed().as_micros() as u64;

        let sparse_run_start = std::time::Instant::now();
        for frame in 0..FRAMES {
            let dirty_pages: &[u64] = if frame == 0 {
                &touched_pages
            } else {
                &touched_pages[..16]
            };
            for &page in dirty_pages {
                engine
                    .upload_range(page * TILE_BYTES, &page_payload)
                    .expect("sparse upload");
            }
            engine.flush().expect("sparse flush");
            rig.run("sparse shader read", |list| {
                // SAFETY: live reserved windows and compute pipeline.
                unsafe {
                    list.SetComputeRootSignature(&sum_root);
                    list.SetPipelineState(&sum_pso);
                    list.SetComputeRootUnorderedAccessView(2, out_uav.GetGPUVirtualAddress());
                    for _ in 0..read_passes {
                        for w in &sparse_windows {
                            let params = [0u32, (PAGE / 4) as u32];
                            list.SetComputeRoot32BitConstants(0, 2, params.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, w.gpu_va);
                            list.Dispatch((PAGE as u32 / 4).div_ceil(64), 1, 1);
                        }
                    }
                    list.ResourceBarrier(&[transition(
                        &out_uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    list.CopyBufferRegion(&out_rb, 0, &out_uav, 0, TILE_BYTES);
                    list.ResourceBarrier(&[transition(
                        &out_uav,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            });
        }
        let sparse_run_us = sparse_run_start.elapsed().as_micros() as u64;
        let sparse_checksum = rig.read_u32s(&out_rb, 64);
        assert_eq!(
            sparse_checksum, base_checksum,
            "sparse VA shader output must match baseline bit-for-bit"
        );

        let stats = engine.stats();
        assert_eq!(stats.committed_tiles, CELLS * TOUCHED_PER_CELL);
        let mem_reduction = base_committed_bytes as f64 / stats.committed_bytes as f64;
        let copy_reduction = base_bytes_copied as f64 / stats.upload_bytes_total as f64;
        let base_total_us = base_create_us + base_run_us;
        let sparse_total_us = sparse_create_us + sparse_run_us;
        let total_speedup = base_total_us as f64 / sparse_total_us.max(1) as f64;
        eprintln!(
            "[sparse_va {} BENCH] tier={} windows={} passes={read_passes} reserved_mib={} | \
             committed: baseline={}MiB vs sparse={}KiB ({mem_reduction:.1}x less, heap_pool={}MiB) | \
             copies: baseline={}KiB vs sparse={}KiB ({copy_reduction:.2}x less) | \
             create: baseline={}us vs sparse={}us | {FRAMES}-frame run: baseline={}us vs sparse={}us | \
             total: baseline={}us vs sparse={}us ({total_speedup:.2}x) | \
             UpdateTileMappings: {} calls in {}us",
            rig.label,
            rig.tier,
            windows_spec.len(),
            stats.reserved_va_bytes >> 20,
            base_committed_bytes >> 20,
            stats.committed_bytes >> 10,
            stats.heap_pool_bytes >> 20,
            base_bytes_copied >> 10,
            stats.upload_bytes_total >> 10,
            base_create_us,
            sparse_create_us,
            base_run_us,
            sparse_run_us,
            base_total_us,
            sparse_total_us,
            stats.update_tile_mappings_calls,
            stats.update_tile_mappings_us,
        );

        for w in sparse_windows {
            engine.release_window(w.id).expect("release bench window");
        }
        engine.wait_idle().expect("bench wait_idle");
        rig.check_debug("benchmark teardown");
    }

    #[test]
    fn coalescing_and_heap_free_list_logic() {
        let mut pairs = vec![(5, 10), (1, 2), (2, 3), (3, 7), (6, 11), (1, 2)];
        assert_eq!(
            coalesce_tile_pairs(&mut pairs),
            vec![(1, 2, 2), (3, 7, 1), (5, 10, 2)]
        );
        let mut locals = vec![10, 2, 3, 4, 11, 2, 0];
        assert_eq!(coalesce_local_runs(&mut locals), vec![(0, 1), (2, 3), (10, 2)]);
    }

    #[test]
    fn alignment_and_bounds_rejected_cleanly() {
        let Some(rig) = TestRig::new(false) else {
            return;
        };
        let engine = SparseVa::new(rig.device.clone(), rig.queue.clone(), SparseConfig::default())
            .expect("SparseVa::new");
        assert!(engine.create_window(0x1000_0001, TILE_BYTES).is_err());
        assert!(engine.create_window(0x1000_0000, 0).is_err());
        assert!(engine.create_window(0x1000_0000, 4096).is_err());
        assert!(
            engine
                .create_window(u64::MAX - TILE_BYTES + 1, TILE_BYTES * 2)
                .is_err()
        );
        rig.check_debug("alignment rejection");
    }

    #[test]
    fn sparse_va_warp_full_verification_and_benchmark() {
        let Some(mut rig) = TestRig::new(false) else {
            panic!("WARP Tier 2+ device unavailable");
        };
        run_overlapping_views_suite(&mut rig);
        run_sparse_gaps_suite(&mut rig);
        run_reclamation_and_recycling_suite(&mut rig);
        run_coalesced_upload_and_early_drop_suite(&mut rig);
        run_benchmark_vs_loose_windows(&mut rig);
    }

    #[test]
    fn sparse_va_hardware_full_verification_and_benchmark() {
        let Some(mut rig) = TestRig::new(true) else {
            eprintln!("[sparse_va] no Tier 2+ hardware D3D12 adapter available; skipped");
            return;
        };
        run_overlapping_views_suite(&mut rig);
        run_sparse_gaps_suite(&mut rig);
        run_reclamation_and_recycling_suite(&mut rig);
        run_coalesced_upload_and_early_drop_suite(&mut rig);
        run_benchmark_vs_loose_windows(&mut rig);
    }

    /// Regression suite covering:
    /// (1) `Shadow::own_buffer_refs()` COM refcount & `let_go_idle` eviction + tile reclamation,
    /// (2) `placed_copy` stages 0 GPU uploads while updating reference snapshots,
    /// (3) single-read `declare_uav_write` & overlapping UAV declaration not clobbering GPU bytes,
    /// (4) cross-window active-UAV visibility (`writes_pending`/`stale_under_writes`/`nothing_new`/`update`),
    /// (5a) repeated copy-back after same-byte conflict preserves title's winning byte,
    /// (5b) trimmed `read_into`/`put` succeeds when untouched 4 KiB sub-pages are unmapped,
    /// (5c) `shadow::find_all` splits multi-page sparse ranges on 64 KiB `PIECE` boundaries.
    #[test]
    fn sparse_shadow_integration_regressions_all_seven_invariants() {
        use windows_sys::Win32::System::Memory as mem;
        let Some(mut rig) = TestRig::new(false) else {
            return;
        };
        let engine = Arc::new(
            SparseVa::new(
                rig.device.clone(),
                rig.queue.clone(),
                SparseConfig {
                    heap_tiles: 16,
                    keep_one_heap: false,
                },
            )
            .expect("SparseVa::new"),
        );

        // Allocate 256 KiB (4 x 64 KiB pages) of synthetic title memory.
        let len = 4 * PAGE;
        // SAFETY: isolated VirtualAlloc test buffer freed at end of test.
        let base = unsafe {
            mem::VirtualAlloc(
                std::ptr::null(),
                len,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: committed 256 KiB test buffer.
        let title_slice = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, len) };
        title_slice.fill(0x11);

        // Create TWO overlapping sparse Shadows: A covers [0..256 KiB), B covers [0..128 KiB).
        let (res_a, sp_a) = SparseShadow::from_handle(
            engine.clone(),
            engine
                .create_window(base as u64, len as u64)
                .expect("win A"),
        );
        let shadow_a = Arc::new(crate::shadow::Shadow::sparse_backed_test(
            res_a,
            sp_a,
            base as u64,
            len as u64,
        ));
        let (res_b, sp_b) = SparseShadow::from_handle(
            engine.clone(),
            engine
                .create_window(base as u64, (2 * PAGE) as u64)
                .expect("win B"),
        );
        let shadow_b = Arc::new(crate::shadow::Shadow::sparse_backed_test(
            res_b,
            sp_b,
            base as u64,
            (2 * PAGE) as u64,
        ));

        // --- Regression (1): COM refcount matches `own_buffer_refs()` (2 when idle, >2 when pinned) ---
        assert_eq!(shadow_a.own_buffer_refs(), 2);
        assert_eq!(
            crate::host::references(&shadow_a.buffer),
            shadow_a.own_buffer_refs(),
            "idle sparse shadow must have exact own_buffer_refs() so let_go_idle can evict it"
        );
        {
            let _descriptor_pin = shadow_a.buffer.clone();
            assert!(
                crate::host::references(&shadow_a.buffer) > shadow_a.own_buffer_refs(),
                "descriptor-pinned sparse shadow must exceed own_buffer_refs()"
            );
        }
        assert_eq!(crate::host::references(&shadow_a.buffer), shadow_a.own_buffer_refs());

        // --- Regression (5c): `find_all` splits multi-page sparse ranges on 64 KiB PIECE boundaries ---
        let pieces = crate::shadow::find_all(&[(shadow_a.clone(), base + 128, base + 3 * PAGE + 256)]);
        assert_eq!(pieces.len(), 1);
        assert_eq!(
            pieces[0],
            vec![
                (128, PAGE - 128),
                (PAGE, PAGE),
                (2 * PAGE, PAGE),
                (3 * PAGE, 256),
            ],
            "find_all must split sparse spans strictly on 64 KiB PIECE boundaries for land::jobs"
        );

        // --- Regression (3) + (4): UAV write on A + cross-window visibility on B + no clobber ---
        assert!(!engine.has_any_active_uav());
        shadow_a.writing(base, 4096);
        assert!(engine.has_any_active_uav());
        assert_eq!(engine.stats().committed_tiles, 1, "only page 0 committed");

        // Overlapping Window B (whose own `has_pending` is false!) must observe A's active UAV write!
        assert!(
            shadow_b.writes_pending(),
            "overlapping sparse shadow B must report writes_pending while A has active UAV"
        );
        assert!(
            shadow_b.writes_pending_in(base, base + 4096),
            "overlapping sparse shadow B must report writes_pending_in on active range"
        );
        assert!(
            !shadow_b.writes_pending_in(base + PAGE, base + 2 * PAGE),
            "overlapping sparse shadow B must not report writes_pending_in on untouched page"
        );
        assert!(
            shadow_b.stale_under_writes(base, 4096),
            "overlapping sparse shadow B must report stale_under_writes on shared page"
        );
        let mut runs_left = 64;
        assert!(
            shadow_b.merge_under_writes(base, 4096, &mut runs_left),
            "overlapping sparse shadow B must force settle via merge_under_writes"
        );
        assert!(
            !shadow_b.nothing_new(base, 4096),
            "overlapping sparse shadow B must not report nothing_new during active UAV"
        );

        // GPU writes dwords 0..16 (bytes 0..64) through Window A.
        let (write_root, write_pso) = rig.make_compute_pso(WRITE_UAV_HLSL);
        let seed = 0x1234_5678u32;
        rig.run("sparse shadow UAV write", |list| {
            let params = [0u32, 16u32, seed];
            // SAFETY: live reserved window with page 0 mapped.
            unsafe {
                list.SetComputeRootSignature(&write_root);
                list.SetPipelineState(&write_pso);
                list.SetComputeRoot32BitConstants(0, 3, params.as_ptr().cast(), 0);
                list.SetComputeRootUnorderedAccessView(1, shadow_a.gpu(base as u64));
                list.Dispatch(1, 1, 1);
            }
        });

        // Regression (3b) & (4): declaring an overlapping UAV range [0..8192) on A or
        // refreshing [0..4096) through overlapping B MUST NOT clobber GPU's bytes [0..64)!
        title_slice[0..64].fill(0x99);
        shadow_b.update(base, 4096);
        shadow_a.writing(base, 8192);
        engine.flush().expect("flush after overlapping update/writing");
        let gpu_after_overlap = engine
            .readback_range(&shadow_a.buffer, 0, 64)
            .expect("readback GPU bytes");
        for i in 0..16usize {
            let expect = seed ^ ((i as u32).wrapping_mul(2_654_435_761)) ^ 0xA55A_1337u32;
            let got = u32::from_le_bytes(gpu_after_overlap[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(
                got, expect,
                "overlapping update or writing clobbered active GPU UAV dword {i}"
            );
        }

        // Restore title_slice[0..64] to initial 0x11, then create:
        // - same-byte conflict at [0..4) -> 0xDEAD_BEEF (title MUST win!)
        // - CPU-only write at [128..192) -> 0x22
        title_slice[0..64].fill(0x11);
        title_slice[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        title_slice[128..192].fill(0x22);

        // --- Regression (2): `placed_copy` must NOT stage any GPU upload bytes ---
        let uploads_before_placed = engine.stats().upload_bytes_total;
        let placed_bytes = [0x44u8; 16];
        shadow_a.placed_copy(base + 512, &placed_bytes);
        assert_eq!(
            engine.stats().upload_bytes_total,
            uploads_before_placed,
            "placed_copy must only update reference snapshot, never stage GPU uploads"
        );
        // Verify pending_uploads is empty (no stale upload queued for next submission).
        assert!(
            engine
                .inner
                .lock()
                .unwrap()
                .pending_uploads
                .is_empty(),
            "placed_copy left stale pending_uploads for next submission"
        );

        // --- Regression (5b): Decommit an untouched 4 KiB sub-page in page 0 (`[4096..8192)`) ---
        // Trimmed `read_into` must STILL merge GPU writes in `[0..64)` instead of failing!
        // SAFETY: decommitting a 4 KiB sub-page inside our own test VirtualAlloc region.
        let decommit_ok = unsafe {
            mem::VirtualFree((base + 4096) as *mut _, 4096, mem::MEM_DECOMMIT) != 0
        };
        assert!(decommit_ok, "decommit untouched 4 KiB sub-page");

        // --- Regression (5a): Copy-back TWICE while `uav_pages` stays live ---
        // First merge: [0..4) keeps title's 0xDEAD_BEEF; [4..64) takes GPU bytes;
        // [128..192) keeps title's 0x22.
        let written_1 = shadow_a.copy_back(base, 8192);
        assert!(written_1 >= 60);
        assert_eq!(
            u32::from_le_bytes(title_slice[0..4].try_into().unwrap()),
            0xDEAD_BEEFu32,
            "title must win on first copy-back same-byte conflict"
        );
        for i in 1..16usize {
            let expect = seed ^ ((i as u32).wrapping_mul(2_654_435_761)) ^ 0xA55A_1337u32;
            let got = u32::from_le_bytes(title_slice[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(got, expect, "GPU byte missing at dword {i}");
        }
        assert!(title_slice[128..192].iter().all(|&b| b == 0x22));

        // Second merge while UAV snapshot is still live: MUST NOT clobber title's 0xDEAD_BEEF!
        let written_2 = shadow_a.copy_back(base, 8192);
        assert_eq!(written_2, 0, "second copy-back with no new GPU writes must write 0 bytes");
        assert_eq!(
            u32::from_le_bytes(title_slice[0..4].try_into().unwrap()),
            0xDEAD_BEEFu32,
            "second copy-back clobbered title's winning byte with rejected GPU byte"
        );

        // Complete all UAV ranges -> `has_any_active_uav()` returns to false.
        shadow_a.copy_back_done(&[(base, base + 4096), (base, base + 8192)]);
        assert!(!engine.has_any_active_uav());

        // --- Regression (1) end-to-end: real `LOOSE` + `let_go_idle` eviction & tile reclamation ---
        let key_a = (base as u64, len as u64);
        let key_b = (base as u64, (2 * PAGE) as u64);
        // Pin `shadow_a.buffer` as if held by a descriptor slot (`HostHeap::kept`) and
        // move both windows into `shadow::LOOSE` so `Arc::strong_count == 1`.
        let descriptor_pin = shadow_a.buffer.clone();
        crate::shadow::insert_loose_test(shadow_a);
        crate::shadow::insert_loose_test(shadow_b);
        assert!(crate::shadow::loose_contains_test(key_a.0, key_a.1));
        assert!(crate::shadow::loose_contains_test(key_b.0, key_b.1));

        // First sweep: unpinned `shadow_b` MUST be evicted; descriptor-pinned `shadow_a` MUST stay.
        crate::shadow::let_go_idle_now_test();
        assert!(
            !crate::shadow::loose_contains_test(key_b.0, key_b.1),
            "idle unpinned sparse shadow B must be evicted by let_go_idle"
        );
        assert!(
            crate::shadow::loose_contains_test(key_a.0, key_a.1),
            "descriptor-pinned sparse shadow A must NOT be evicted while buffer refcount > 2"
        );

        // Release descriptor pin -> second sweep MUST evict `shadow_a` and reclaim every tile & heap.
        drop(descriptor_pin);
        crate::shadow::let_go_idle_now_test();
        assert!(
            !crate::shadow::loose_contains_test(key_a.0, key_a.1),
            "unpinned sparse shadow A must be evicted by let_go_idle once refcount == 2"
        );
        engine.wait_idle().expect("wait_idle");
        rig.check_debug("sparse shadow integration regressions teardown");
        let end_stats = engine.stats();
        assert_eq!(end_stats.live_windows, 0);
        assert_eq!(end_stats.committed_tiles, 0);
        assert_eq!(end_stats.heap_count, 0);

        // --- Regression (7): global `has_active_uav()` counter tracks declare / drop ---
        {
            assert!(!has_active_uav());
            let global_eng = Arc::new(
                SparseVa::new_with_mode(
                    rig.device.clone(),
                    rig.queue.clone(),
                    SparseConfig {
                        heap_tiles: 8,
                        keep_one_heap: false,
                    },
                    true,
                )
                .expect("global-mode SparseVa"),
            );
            let h = global_eng
                .create_window(base as u64, PAGE as u64)
                .expect("global win");
            global_eng
                .declare_uav_write(h.id, base, base + 4096)
                .expect("declare global uav");
            assert!(
                has_active_uav(),
                "Host::submit settle gate must observe active sparse UAV writes"
            );
            global_eng.release_window(h.id).expect("release global win");
            assert!(
                !has_active_uav(),
                "releasing window must decrement global active sparse UAV counter back to 0"
            );
            global_eng.wait_idle().expect("global wait_idle");
        }
        rig.check_debug("global active uav gate check");

        // SAFETY: release our test VirtualAlloc buffer.
        unsafe {
            mem::VirtualFree(base as *mut _, 0, mem::MEM_RELEASE);
        }
    }
}
