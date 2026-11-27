//! Multi-queue hardware timeline engine (`VICTORY_ASYNC_COMPUTE=1`, off by default).
//!
//! Coordinates three D3D12 command queues — the host's `DIRECT` queue plus dedicated
//! `D3D12_COMMAND_LIST_TYPE_COMPUTE` and `D3D12_COMMAND_LIST_TYPE_COPY` queues — using
//! strictly GPU-side `ID3D12Fence` timeline synchronization (`ProducerQueue->Signal`
//! followed by `ConsumerQueue->Wait` before `ExecuteCommandLists`) with zero CPU fence
//! waits on the submission hot path.
//!
//! All cross-queue buffer handoffs adhere strictly to D3D12 common-state promotion and
//! decay rules:
//! - Buffers created in `D3D12_RESOURCE_STATE_COMMON` (`DEFAULT` or `CUSTOM` heaps)
//!   implicitly promote on first use within an `ExecuteCommandLists` batch and decay
//!   back to `COMMON` at the end of the batch.
//! - `UPLOAD` heap buffers remain `D3D12_RESOURCE_STATE_GENERIC_READ` and `READBACK`
//!   heap buffers remain `D3D12_RESOURCE_STATE_COPY_DEST` (never transitioned).
//! - Copy command lists only execute copy operations on copy-legal states
//!   (`COMMON`, `COPY_SOURCE`, `COPY_DEST`, or immutable upload/readback heaps).
//! - Compute command lists never reference direct-only states (`RENDER_TARGET`,
//!   `DEPTH_WRITE`, `DEPTH_READ`, `PIXEL_SHADER_RESOURCE`, `STREAM_OUT`, `PRESENT`).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Maximum simultaneously in-flight command allocator slots per async queue before
/// applying backpressure to avoid unbounded allocator memory growth.
const MAX_SLOTS_PER_QUEUE: usize = 32;

/// Minimum allocation granularity for pooled VRAM bounce buffers used by copy uploads.
const MIN_BOUNCE_BYTES: u64 = 4 << 20;

/// Maximum single upload size routed through the async copy bounce path.
const MAX_BOUNCE_BYTES: u64 = 512 << 20;

/// Maximum pooled bounce buffers kept warm for reuse across frames.
const MAX_POOLED_BOUNCES: usize = 4;

/// Maximum tracked resource entries before completed entries are pruned.
const HAZARD_PRUNE_THRESHOLD: usize = 512;

/// Resource states strictly forbidden on `D3D12_COMMAND_LIST_TYPE_COMPUTE`.
#[cfg(test)]
const COMPUTE_FORBIDDEN_MASK: u32 = (D3D12_RESOURCE_STATE_RENDER_TARGET.0
    | D3D12_RESOURCE_STATE_DEPTH_WRITE.0
    | D3D12_RESOURCE_STATE_DEPTH_READ.0
    | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0
    | D3D12_RESOURCE_STATE_STREAM_OUT.0
    | D3D12_RESOURCE_STATE_RESOLVE_DEST.0
    | D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0
    | D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE.0) as u32;

/// Resource states legal on `D3D12_COMMAND_LIST_TYPE_COPY` (excluding immutable
/// `UPLOAD` / `READBACK` heap states, which must not be transitioned).
#[cfg(test)]
const COPY_LEGAL_MASK: u32 =
    (D3D12_RESOURCE_STATE_COPY_SOURCE.0 | D3D12_RESOURCE_STATE_COPY_DEST.0) as u32;

/// Hardware queue lane managed by [`AsyncTimeline`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub(crate) enum QueueKind {
    Direct = 0,
    Compute = 1,
    Copy = 2,
}

impl QueueKind {
    const ALL: [QueueKind; 3] = [QueueKind::Direct, QueueKind::Compute, QueueKind::Copy];

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

/// Monotonic GPU fence point on a specific hardware queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TimelineToken {
    pub(crate) queue: QueueKind,
    pub(crate) value: u64,
}

/// Returns whether `state` is legal for a barrier on `D3D12_COMMAND_LIST_TYPE_COPY`.
#[cfg(test)]
pub(crate) fn is_copy_legal_state(state: D3D12_RESOURCE_STATES) -> bool {
    let bits = state.0 as u32;
    bits == 0 || (bits & !COPY_LEGAL_MASK) == 0
}

/// Returns whether `state` is legal for a barrier on `D3D12_COMMAND_LIST_TYPE_COMPUTE`.
#[cfg(test)]
pub(crate) fn is_compute_legal_state(state: D3D12_RESOURCE_STATES) -> bool {
    let bits = state.0 as u32;
    (bits & COMPUTE_FORBIDDEN_MASK) == 0
}

/// Parse an explicit `VICTORY_ASYNC_COMPUTE` environment value (off by default).
fn parse_switch(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Whether the opt-in multi-queue hardware timeline engine is enabled
/// (`VICTORY_ASYNC_COMPUTE=1`, or A/B-toggled under `diagnostics`).
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(picked) = crate::ab::pick("VICTORY_ASYNC_COMPUTE") {
        return picked;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_switch(std::env::var_os("VICTORY_ASYNC_COMPUTE").as_deref()))
}

/// Raw identity key for a D3D12 resource inside [`HazardTracker`].
#[inline]
pub(crate) fn resource_key(resource: &ID3D12Resource) -> usize {
    resource.as_raw() as usize
}

/// Per-resource last-writer and per-queue last-reader timeline points.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResourceUsage {
    last_write: Option<TimelineToken>,
    last_read: [u64; 3],
}

/// Tracks cross-queue RAW / WAR / WAW dependencies per resource and coalesces
/// them to at most one GPU-side `Queue->Wait` per producer queue.
#[derive(Default)]
pub(crate) struct HazardTracker {
    entries: HashMap<usize, ResourceUsage>,
}

impl HazardTracker {
    /// Computes the minimal fence value required on each queue before `target`
    /// executes a batch reading `reads` and writing `writes`.
    ///
    /// The slot for `target` itself is always `0` because same-queue submission
    /// order is already FIFO-ordered by D3D12 hardware.
    pub(crate) fn deps_for(
        &self,
        target: QueueKind,
        reads: &[usize],
        writes: &[usize],
    ) -> [u64; 3] {
        let mut waits = [0u64; 3];
        let target_idx = target.index();

        for &key in reads {
            if let Some(usage) = self.entries.get(&key)
                && let Some(writer) = usage.last_write
                && writer.queue != target
            {
                let idx = writer.queue.index();
                waits[idx] = waits[idx].max(writer.value);
            }
        }

        for &key in writes {
            if let Some(usage) = self.entries.get(&key) {
                if let Some(writer) = usage.last_write
                    && writer.queue != target
                {
                    let idx = writer.queue.index();
                    waits[idx] = waits[idx].max(writer.value);
                }
                for q in QueueKind::ALL {
                    let idx = q.index();
                    if idx != target_idx && usage.last_read[idx] > 0 {
                        waits[idx] = waits[idx].max(usage.last_read[idx]);
                    }
                }
            }
        }

        waits[target_idx] = 0;
        waits
    }

    /// Records that `token` read `reads` and wrote `writes`.
    pub(crate) fn record_submission(
        &mut self,
        token: TimelineToken,
        reads: &[usize],
        writes: &[usize],
    ) {
        let q_idx = token.queue.index();
        for &key in reads {
            let entry = self.entries.entry(key).or_default();
            entry.last_read[q_idx] = entry.last_read[q_idx].max(token.value);
        }
        for &key in writes {
            let entry = self.entries.entry(key).or_default();
            entry.last_write = Some(token);
            // Cross-queue waits inserted before this write ensure earlier reads
            // on other queues have completed; reset them so later writers do not
            // re-wait on superseded reads.
            entry.last_read = [0; 3];
        }
    }

    /// Drops resource entries whose tracked fence values have all completed on GPU.
    pub(crate) fn prune(&mut self, completed: [u64; 3]) {
        if self.entries.len() < HAZARD_PRUNE_THRESHOLD {
            return;
        }
        self.entries.retain(|_, usage| {
            let write_live = usage
                .last_write
                .is_some_and(|tok| tok.value > completed[tok.queue.index()]);
            let read_live = QueueKind::ALL
                .iter()
                .any(|&q| usage.last_read[q.index()] > completed[q.index()]);
            write_live || read_live
        });
    }
}

/// Reusable command allocator + closed command list pair for an async queue.
struct CommandSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

// SAFETY: D3D12 command allocators and lists are free-threaded objects owned by
// one slot at a time and protected by the queue engine mutex.
unsafe impl Send for CommandSlot {}

/// An in-flight async command list together with resources kept alive until its
/// GPU fence value completes.
struct InFlight {
    fence_value: u64,
    slot: CommandSlot,
    _keep: Vec<ID3D12Resource>,
}

// SAFETY: All COM handles are free-threaded and dropped only after GPU completion.
unsafe impl Send for InFlight {}

/// One dedicated async queue (`COMPUTE` or `COPY`) with its non-blocking slot ring.
struct QueueEngine {
    kind: D3D12_COMMAND_LIST_TYPE,
    queue: ID3D12CommandQueue,
    sent: u64,
    idle: Vec<CommandSlot>,
    in_flight: VecDeque<InFlight>,
    /// Slots whose `ExecuteCommandLists` ran but whose trailing `Signal` failed;
    /// pinned until teardown and never waited on via `SetEventOnCompletion`.
    abandoned: Vec<InFlight>,
    total_slots: usize,
}

// SAFETY: Access to `queue`, `idle`, `in_flight`, and `abandoned` is serialized
// by `Mutex<QueueEngine>`.
unsafe impl Send for QueueEngine {}

impl QueueEngine {
    fn new(
        device: &ID3D12Device,
        kind: D3D12_COMMAND_LIST_TYPE,
        priority: D3D12_COMMAND_QUEUE_PRIORITY,
        debug_name: &'static str,
    ) -> windows::core::Result<Self> {
        // SAFETY: Valid D3D12 queue creation on a live device.
        let queue: ID3D12CommandQueue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: kind,
                Priority: priority.0,
                ..Default::default()
            })?
        };
        crate::host::name(&queue, || debug_name.to_owned());
        Ok(Self {
            kind,
            queue,
            sent: 0,
            idle: Vec::new(),
            in_flight: VecDeque::new(),
            abandoned: Vec::new(),
            total_slots: 0,
        })
    }

    /// Moves all GPU-completed in-flight slots back to `idle` and releases their
    /// held resources. Never blocks the CPU.
    fn reclaim(&mut self, fence: &ID3D12Fence) {
        // SAFETY: `GetCompletedValue` is a lock-free query on a live fence.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX {
            // Device removed: avoid resetting allocators while removed.
            return;
        }
        while self
            .in_flight
            .front()
            .is_some_and(|entry| entry.fence_value <= completed)
        {
            if let Some(done) = self.in_flight.pop_front() {
                self.idle.push(done.slot);
            }
        }
    }

    /// Acquires an open, reset command slot ready for recording without blocking
    /// unless all `MAX_SLOTS_PER_QUEUE` slots are simultaneously in flight.
    fn acquire_open(
        &mut self,
        device: &ID3D12Device,
        fence: &ID3D12Fence,
    ) -> windows::core::Result<CommandSlot> {
        self.reclaim(fence);
        let slot = if let Some(slot) = self.idle.pop() {
            // SAFETY: The slot was reclaimed only after `fence.GetCompletedValue() >= last_fence`.
            unsafe {
                slot.allocator.Reset()?;
                slot.list.Reset(&slot.allocator, None)?;
            }
            slot
        } else if self.total_slots < MAX_SLOTS_PER_QUEUE {
            // SAFETY: Plain D3D12 object creation matching this queue's list type.
            let slot = unsafe {
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(self.kind)?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, self.kind, &allocator, None)?;
                CommandSlot { allocator, list }
            };
            self.total_slots += 1;
            slot
        } else {
            // Peek before waiting so a failed wait never pops and drops an
            // uncompleted in-flight slot and its live GPU resources.
            let oldest_val = self
                .in_flight
                .front()
                .ok_or_else(|| {
                    windows::core::Error::from(
                        windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED,
                    )
                })?
                .fence_value;
            wait_fence_cpu(fence, oldest_val)?;
            let oldest = self
                .in_flight
                .pop_front()
                .ok_or_else(windows::core::Error::empty)?;
            // SAFETY: The oldest submission's fence value has completed on GPU.
            unsafe {
                oldest.slot.allocator.Reset()?;
                oldest.slot.list.Reset(&oldest.slot.allocator, None)?;
            }
            oldest.slot
        };
        Ok(slot)
    }
}

/// Waits on the CPU for `fence` to reach `value` only if it has not already done so.
/// Returns `true` if a blocking wait was needed, `false` if already complete.
fn wait_fence_cpu(fence: &ID3D12Fence, value: u64) -> windows::core::Result<bool> {
    if value == 0 {
        return Ok(false);
    }
    if value == u64::MAX {
        return Err(windows::core::Error::from(
            windows::Win32::Foundation::E_INVALIDARG,
        ));
    }
    // SAFETY: Lock-free completion query on a live fence.
    let completed = unsafe { fence.GetCompletedValue() };
    if completed == u64::MAX {
        return Err(windows::core::Error::from(
            windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED,
        ));
    }
    if completed >= value {
        return Ok(false);
    }
    // SAFETY: Passing a null `HANDLE` instructs D3D12 to block this thread
    // synchronously until `value` is reached without allocating an event object.
    unsafe { fence.SetEventOnCompletion(value, HANDLE::default())? };
    Ok(true)
}

/// A closed compute command list prepared ahead of Direct execution and ready to
/// be dispatched onto the async compute queue once Direct signals its handoff fence.
pub(crate) struct PreparedCompute {
    slot: CommandSlot,
}

/// Deferred async compute compare state attached to a `Host` `CardPlan`.
pub(crate) enum PendingCheck {
    /// Recorded onto a `D3D12_COMMAND_LIST_TYPE_COMPUTE` list; awaiting Direct
    /// title list execution before dispatch.
    Prepared {
        prepared: PreparedCompute,
        reads: Vec<usize>,
        writes: Vec<usize>,
        keep: Vec<ID3D12Resource>,
    },
    /// Dispatched to the async compute queue; `token` tracks GPU completion.
    Dispatched(TimelineToken),
}

/// A pooled VRAM bounce buffer populated by the async copy queue and consumed by Direct.
pub(crate) struct BounceHandle {
    buffer: ID3D12Resource,
    size: u64,
    copy_token: TimelineToken,
}

// SAFETY: Free-threaded D3D12 resource owned exclusively by one submission until returned.
unsafe impl Send for BounceHandle {}

impl BounceHandle {
    /// The VRAM bounce buffer holding uploaded bytes.
    pub(crate) fn resource(&self) -> &ID3D12Resource {
        &self.buffer
    }
}

struct PooledBounce {
    buffer: ID3D12Resource,
    size: u64,
}

// SAFETY: Free-threaded D3D12 resource stored behind a mutex.
unsafe impl Send for PooledBounce {}

/// Telemetry counters for the frame-rate log line.
/// 0: compute submits, 1: copy uploads, 2: copy readbacks,
/// 3: GPU-side queue waits inserted, 4: CPU waits avoided (already done),
/// 5: CPU waits blocked, 6: rehash passes offloaded, 7: compare passes offloaded,
/// 8: bytes uploaded via async copy, 9: bytes read back via async copy.
static STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

const ST_COMPUTE: usize = 0;
const ST_COPY_UP: usize = 1;
const ST_COPY_RB: usize = 2;
const ST_GPU_WAITS: usize = 3;
const ST_CPU_AVOIDED: usize = 4;
const ST_CPU_BLOCKED: usize = 5;
const ST_REHASH: usize = 6;
const ST_COMPARE: usize = 7;
const ST_UP_BYTES: usize = 8;
const ST_RB_BYTES: usize = 9;

/// Multi-queue hardware timeline engine coordinating `DIRECT`, `COMPUTE`, and `COPY`
/// execution with strictly GPU-side `ID3D12Fence` handoffs.
///
/// Lock order invariant (strictly acyclic):
/// `hazards` (outer, held atomically across Wait + ExecuteCommandLists + Signal + record)
///   -> `compute` | `copy` | `direct_sent` (inner, never nested with each other).
/// `bounces` is a leaf lock. `wait_cpu` never holds any lock.
pub(crate) struct AsyncTimeline {
    device: ID3D12Device,
    direct_fence: ID3D12Fence,
    compute_fence: ID3D12Fence,
    copy_fence: ID3D12Fence,
    /// Dedicated 256-byte compute-only scratch UAV bound to unused slot 3 in
    /// mode-0 `CARD_COMPARE` rehash dispatches so async rehash never aliases
    /// `CARD_WORK.differs` across queues.
    rehash_dummy_uav: ID3D12Resource,
    direct_sent: Mutex<u64>,
    compute: Mutex<QueueEngine>,
    copy: Mutex<QueueEngine>,
    hazards: Mutex<HazardTracker>,
    bounces: Mutex<Vec<PooledBounce>>,
    copy_timestamps_supported: bool,
}

// SAFETY: All D3D12 device/fence/resource interfaces are free-threaded; mutable
// queue and tracker states live behind dedicated mutexes with a strict lock order.
unsafe impl Send for AsyncTimeline {}
// SAFETY: As above.
unsafe impl Sync for AsyncTimeline {}

impl AsyncTimeline {
    /// Creates a new multi-queue timeline engine on `device` with dedicated
    /// `D3D12_COMMAND_LIST_TYPE_COMPUTE` and `D3D12_COMMAND_LIST_TYPE_COPY` queues.
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let compute = QueueEngine::new(
            device,
            D3D12_COMMAND_LIST_TYPE_COMPUTE,
            D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
            "victory async compute queue",
        )?;
        let copy = QueueEngine::new(
            device,
            D3D12_COMMAND_LIST_TYPE_COPY,
            D3D12_COMMAND_QUEUE_PRIORITY_HIGH,
            "victory async copy queue",
        )?;
        // SAFETY: Plain fence, scratch UAV buffer, and feature query on a live device.
        let (direct_fence, compute_fence, copy_fence, rehash_dummy_uav, copy_timestamps_supported) =
            unsafe {
                let direct_fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let compute_fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let copy_fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let mut dummy: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut dummy,
                )?;
                let dummy = dummy.ok_or_else(windows::core::Error::empty)?;
                let mut opts3 = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
                let copy_ts = device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut opts3).cast(),
                        size_of_val(&opts3) as u32,
                    )
                    .is_ok()
                    && opts3.CopyQueueTimestampQueriesSupported.as_bool();
                (direct_fence, compute_fence, copy_fence, dummy, copy_ts)
            };
        crate::host::name(&direct_fence, || "victory timeline direct fence".into());
        crate::host::name(&compute_fence, || "victory timeline compute fence".into());
        crate::host::name(&copy_fence, || "victory timeline copy fence".into());
        crate::host::name(&rehash_dummy_uav, || {
            "victory async rehash dummy uav".into()
        });
        Ok(Self {
            device: device.clone(),
            direct_fence,
            compute_fence,
            copy_fence,
            rehash_dummy_uav,
            direct_sent: Mutex::new(0),
            compute: Mutex::new(compute),
            copy: Mutex::new(copy),
            hazards: Mutex::new(HazardTracker::default()),
            bounces: Mutex::new(Vec::new()),
            copy_timestamps_supported,
        })
    }

    /// Dedicated compute-only 256-byte scratch UAV for unused `differs` binding
    /// in mode-0 rehash passes.
    pub(crate) fn rehash_dummy_uav(&self) -> &ID3D12Resource {
        &self.rehash_dummy_uav
    }

    #[inline]
    fn fence_for(&self, queue: QueueKind) -> &ID3D12Fence {
        match queue {
            QueueKind::Direct => &self.direct_fence,
            QueueKind::Compute => &self.compute_fence,
            QueueKind::Copy => &self.copy_fence,
        }
    }

    /// Whether the adapter supports `D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP`.
    #[cfg(test)]
    pub(crate) fn copy_timestamps_supported(&self) -> bool {
        self.copy_timestamps_supported
    }

    /// Signals the `Direct` timeline fence on `direct_queue` (GPU-side, non-blocking).
    pub(crate) fn signal_direct(
        &self,
        direct_queue: &ID3D12CommandQueue,
    ) -> windows::core::Result<TimelineToken> {
        let mut sent = self.direct_sent.lock().unwrap_or_else(|e| e.into_inner());
        let value = *sent + 1;
        // SAFETY: Caller supplies a live Direct queue; `direct_fence` belongs to this device.
        unsafe { direct_queue.Signal(&self.direct_fence, value)? };
        *sent = value;
        Ok(TimelineToken {
            queue: QueueKind::Direct,
            value,
        })
    }

    /// Inserts a GPU-side `consumer_queue->Wait(producer_fence, token.value)`.
    /// Zero CPU stall.
    pub(crate) fn wait_on_gpu(
        &self,
        consumer_queue: &ID3D12CommandQueue,
        token: TimelineToken,
    ) -> windows::core::Result<()> {
        if token.value == 0 {
            return Ok(());
        }
        let fence = self.fence_for(token.queue);
        // SAFETY: Both queue and fence live on the same device.
        unsafe { consumer_queue.Wait(fence, token.value)? };
        STATS[ST_GPU_WAITS].fetch_add(1, Relaxed);
        Ok(())
    }

    /// Atomically checks cross-queue hazards for `reads`/`writes`, inserts any
    /// required GPU-side `direct_queue->Wait(...)`, invokes `execute` (which must
    /// only perform non-blocking `direct_queue.ExecuteCommandLists`), signals
    /// `direct_fence`, and records the Direct accesses in [`HazardTracker`].
    ///
    /// Holding `self.hazards` across Wait + Execute + Signal + Record eliminates
    /// TOCTOU races between concurrent threads submitting to Direct, Compute, and Copy.
    pub(crate) fn execute_direct_tracked<R>(
        &self,
        direct_queue: &ID3D12CommandQueue,
        reads: &[usize],
        writes: &[usize],
        execute: impl FnOnce() -> R,
    ) -> windows::core::Result<(R, TimelineToken, u32)> {
        let mut hazards = self.hazards.lock().unwrap_or_else(|e| e.into_inner());
        let waits = if reads.is_empty() && writes.is_empty() {
            [0u64; 3]
        } else {
            hazards.deps_for(QueueKind::Direct, reads, writes)
        };
        let mut wait_count = 0u32;
        for q in [QueueKind::Compute, QueueKind::Copy] {
            let val = waits[q.index()];
            if val > 0 {
                self.wait_on_gpu(direct_queue, TimelineToken { queue: q, value: val })?;
                wait_count += 1;
            }
        }
        let result = execute();
        let token = self.signal_direct(direct_queue)?;
        if !reads.is_empty() || !writes.is_empty() {
            let completed = self.completed_snapshot();
            hazards.record_submission(token, reads, writes);
            hazards.prune(completed);
        }
        Ok((result, token, wait_count))
    }

    /// Non-blocking check whether `token` has completed on the GPU.
    pub(crate) fn is_complete(&self, token: TimelineToken) -> bool {
        if token.value == 0 {
            return true;
        }
        let fence = self.fence_for(token.queue);
        // SAFETY: Lock-free query on a live fence.
        let done = unsafe { fence.GetCompletedValue() };
        done != u64::MAX && done >= token.value
    }

    /// Blocks the calling CPU thread until `token` finishes on the GPU (only for
    /// final readback consumption, landing, or shutdown — never on the submit path).
    pub(crate) fn wait_cpu(&self, token: TimelineToken) -> windows::core::Result<()> {
        let fence = self.fence_for(token.queue);
        let blocked = wait_fence_cpu(fence, token.value)?;
        if blocked {
            STATS[ST_CPU_BLOCKED].fetch_add(1, Relaxed);
        } else {
            STATS[ST_CPU_AVOIDED].fetch_add(1, Relaxed);
        }
        Ok(())
    }

    /// Reclaims completed command slots on both Compute and Copy queues without blocking.
    pub(crate) fn reclaim(&self) {
        self.compute
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reclaim(&self.compute_fence);
        self.copy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .reclaim(&self.copy_fence);
    }

    /// Waits on CPU for all currently submitted Compute and Copy work and reclaims slots.
    #[cfg(test)]
    pub(crate) fn drain_all(&self) -> windows::core::Result<()> {
        let comp_sent = self.compute.lock().unwrap_or_else(|e| e.into_inner()).sent;
        let copy_sent = self.copy.lock().unwrap_or_else(|e| e.into_inner()).sent;
        if comp_sent > 0 {
            wait_fence_cpu(&self.compute_fence, comp_sent)?;
        }
        if copy_sent > 0 {
            wait_fence_cpu(&self.copy_fence, copy_sent)?;
        }
        self.reclaim();
        Ok(())
    }

    /// Records a `D3D12_COMMAND_LIST_TYPE_COMPUTE` command list and closes it for
    /// deferred dispatch via [`Self::dispatch_prepared_compute`].
    pub(crate) fn prepare_compute(
        &self,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> windows::core::Result<()>,
    ) -> windows::core::Result<PreparedCompute> {
        let slot = {
            let mut engine = self.compute.lock().unwrap_or_else(|e| e.into_inner());
            engine.acquire_open(&self.device, &self.compute_fence)?
        };
        let rec_res = record(&slot.list).and_then(|()| {
            // SAFETY: The list was just opened and recorded by `record`.
            unsafe { slot.list.Close() }
        });
        if let Err(err) = rec_res {
            // Close best-effort before returning the unused slot to `idle`.
            // SAFETY: Closing an abandoned open list before recycling.
            let _ = unsafe { slot.list.Close() };
            self.compute
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .idle
                .push(slot);
            return Err(err);
        }
        Ok(PreparedCompute { slot })
    }

    /// Returns an unused [`PreparedCompute`] slot back to the compute pool without submitting.
    pub(crate) fn discard_prepared_compute(&self, prepared: PreparedCompute) {
        self.compute
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .idle
            .push(prepared.slot);
    }

    /// Submits a previously prepared compute list onto the async compute queue with
    /// atomic GPU-side waits for `explicit_deps` and tracked resource hazards. Zero CPU wait.
    pub(crate) fn dispatch_prepared_compute(
        &self,
        prepared: PreparedCompute,
        explicit_deps: &[TimelineToken],
        reads: &[usize],
        writes: &[usize],
        keep: Vec<ID3D12Resource>,
    ) -> windows::core::Result<TimelineToken> {
        let token = self.execute_slot_tracked(
            QueueKind::Compute,
            prepared.slot,
            explicit_deps,
            reads,
            writes,
            keep,
        )?;
        STATS[ST_COMPUTE].fetch_add(1, Relaxed);
        Ok(token)
    }

    /// Records and immediately submits a compute command list on the dedicated
    /// `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue. Zero CPU wait.
    #[cfg(test)]
    pub(crate) fn submit_compute(
        &self,
        explicit_deps: &[TimelineToken],
        reads: &[usize],
        writes: &[usize],
        keep: Vec<ID3D12Resource>,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> windows::core::Result<()>,
    ) -> windows::core::Result<TimelineToken> {
        let prepared = self.prepare_compute(record)?;
        self.dispatch_prepared_compute(prepared, explicit_deps, reads, writes, keep)
    }

    /// Records and submits a copy command list on the dedicated
    /// `D3D12_COMMAND_LIST_TYPE_COPY` queue. Zero CPU wait.
    pub(crate) fn submit_copy(
        &self,
        explicit_deps: &[TimelineToken],
        reads: &[usize],
        writes: &[usize],
        keep: Vec<ID3D12Resource>,
        record: impl FnOnce(&ID3D12GraphicsCommandList) -> windows::core::Result<()>,
    ) -> windows::core::Result<TimelineToken> {
        let slot = {
            let mut engine = self.copy.lock().unwrap_or_else(|e| e.into_inner());
            engine.acquire_open(&self.device, &self.copy_fence)?
        };
        let rec_res = record(&slot.list).and_then(|()| {
            // SAFETY: The list was opened and recorded by `record`.
            unsafe { slot.list.Close() }
        });
        if let Err(err) = rec_res {
            // SAFETY: Closing an abandoned list before returning it to `idle`.
            let _ = unsafe { slot.list.Close() };
            self.copy
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .idle
                .push(slot);
            return Err(err);
        }
        self.execute_slot_tracked(QueueKind::Copy, slot, explicit_deps, reads, writes, keep)
    }

    /// Copies staging/upload ranges into a pooled VRAM bounce buffer on the async
    /// copy queue and inserts a GPU-side `direct_queue->Wait(copy_fence)` so Direct
    /// can scatter/copy out of VRAM with zero CPU fence wait.
    pub(crate) fn submit_upload_bounce(
        &self,
        host: &crate::host::Host,
        direct_queue: &ID3D12CommandQueue,
        ranges: &[(&ID3D12Resource, u64, u64)],
        mut on_placed: impl FnMut(usize, &ID3D12Resource, u64),
    ) -> Option<BounceHandle> {
        let total: u64 = ranges
            .iter()
            .map(|&(_, _, bytes)| bytes.next_multiple_of(256))
            .sum();
        if total == 0 || total > MAX_BOUNCE_BYTES {
            return None;
        }
        let (bounce, bounce_size) = self.acquire_bounce(host, total)?;
        let mut keep = Vec::with_capacity(ranges.len() + 1);
        keep.push(bounce.clone());
        let mut reads = Vec::with_capacity(ranges.len());
        for &(src, _, _) in ranges {
            let k = resource_key(src);
            if !reads.contains(&k) {
                reads.push(k);
                keep.push(src.clone());
            }
        }
        let writes = [resource_key(&bounce)];
        let copy_res = self.submit_copy(&[], &reads, &writes, keep, |list| {
            let mut dst_at = 0u64;
            for &(src, src_offset, bytes) in ranges {
                // SAFETY: Open copy list; live staging/upload source and VRAM bounce
                // buffer starting in `COMMON` (implicitly promoted to `COPY_DEST` and
                // decayed back to `COMMON` at `ExecuteCommandLists` end).
                unsafe {
                    list.CopyBufferRegion(&bounce, dst_at, src, src_offset, bytes);
                }
                dst_at += bytes.next_multiple_of(256);
            }
            Ok(())
        });
        let copy_token = match copy_res {
            Ok(tok) => tok,
            Err(err) => {
                tracing::warn!(error = %err, "async copy queue upload failed; falling back to direct queue");
                self.push_bounce(PooledBounce {
                    buffer: bounce,
                    size: bounce_size,
                });
                return None;
            }
        };
        if let Err(err) = self.wait_on_gpu(direct_queue, copy_token) {
            tracing::warn!(error = %err, "direct queue GPU wait on async copy failed; waiting on CPU");
            let _ = self.wait_cpu(copy_token);
        }
        let mut dst_at = 0u64;
        let mut payload_bytes = 0u64;
        for (i, &(_, _, bytes)) in ranges.iter().enumerate() {
            on_placed(i, &bounce, dst_at);
            dst_at += bytes.next_multiple_of(256);
            payload_bytes += bytes;
        }
        STATS[ST_COPY_UP].fetch_add(1, Relaxed);
        STATS[ST_UP_BYTES].fetch_add(payload_bytes, Relaxed);
        Some(BounceHandle {
            buffer: bounce,
            size: bounce_size,
            copy_token,
        })
    }

    /// Returns a VRAM bounce buffer to the pool once Direct has finished reading it.
    pub(crate) fn return_bounce(&self, handle: BounceHandle) {
        if !self.is_complete(handle.copy_token) {
            let _ = self.wait_cpu(handle.copy_token);
        }
        self.push_bounce(PooledBounce {
            buffer: handle.buffer,
            size: handle.size,
        });
    }

    /// Submits a VRAM -> shared/readback DMA copy batch on the async copy queue.
    pub(crate) fn submit_copy_readback(
        &self,
        deps: &[TimelineToken],
        dst: &ID3D12Resource,
        copies: &[(&ID3D12Resource, u64, u64, u64)],
    ) -> windows::core::Result<TimelineToken> {
        let mut reads = Vec::with_capacity(copies.len());
        let mut keep = Vec::with_capacity(copies.len() + 1);
        keep.push(dst.clone());
        let mut bytes_total = 0u64;
        for &(src, _, _, bytes) in copies {
            bytes_total += bytes;
            let k = resource_key(src);
            if !reads.contains(&k) {
                reads.push(k);
                keep.push(src.clone());
            }
        }
        let writes = [resource_key(dst)];
        let token = self.submit_copy(deps, &reads, &writes, keep, |list| {
            for &(src, src_offset, dst_offset, bytes) in copies {
                // SAFETY: Open copy list; live buffers in `COMMON` / `COPY_DEST`
                // states with bounds guaranteed by caller.
                unsafe {
                    list.CopyBufferRegion(dst, dst_offset, src, src_offset, bytes);
                }
            }
            Ok(())
        })?;
        STATS[ST_COPY_RB].fetch_add(1, Relaxed);
        STATS[ST_RB_BYTES].fetch_add(bytes_total, Relaxed);
        Ok(token)
    }

    fn acquire_bounce(
        &self,
        host: &crate::host::Host,
        needed: u64,
    ) -> Option<(ID3D12Resource, u64)> {
        {
            let mut pool = self.bounces.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pos) = pool
                .iter()
                .enumerate()
                .filter(|(_, b)| b.size >= needed)
                .min_by_key(|(_, b)| b.size)
                .map(|(i, _)| i)
            {
                let chosen = pool.swap_remove(pos);
                return Some((chosen.buffer, chosen.size));
            }
        }
        let alloc_size = needed.next_power_of_two().max(MIN_BOUNCE_BYTES);
        let buffer = host.uav_buffer(alloc_size)?;
        crate::host::name(&buffer, || {
            format!("victory async copy bounce, {} MB", alloc_size >> 20)
        });
        Some((buffer, alloc_size))
    }

    fn push_bounce(&self, entry: PooledBounce) {
        let mut pool = self.bounces.lock().unwrap_or_else(|e| e.into_inner());
        pool.push(entry);
        pool.sort_by_key(|b| std::cmp::Reverse(b.size));
        while pool.len() > MAX_POOLED_BOUNCES {
            pool.pop();
        }
    }

    fn completed_snapshot(&self) -> [u64; 3] {
        // SAFETY: Lock-free queries on live fences.
        unsafe {
            [
                self.direct_fence.GetCompletedValue(),
                self.compute_fence.GetCompletedValue(),
                self.copy_fence.GetCompletedValue(),
            ]
        }
    }

    /// Atomically queries `hazards`, emits GPU-side `Queue->Wait` calls, executes
    /// and signals `target`'s queue, and records `reads`/`writes` under the same
    /// `hazards` lock (lock order: `hazards` -> `engine`).
    fn execute_slot_tracked(
        &self,
        target: QueueKind,
        slot: CommandSlot,
        explicit_deps: &[TimelineToken],
        reads: &[usize],
        writes: &[usize],
        keep: Vec<ID3D12Resource>,
    ) -> windows::core::Result<TimelineToken> {
        let (engine_mutex, own_fence) = match target {
            QueueKind::Compute => (&self.compute, &self.compute_fence),
            QueueKind::Copy => (&self.copy, &self.copy_fence),
            QueueKind::Direct => {
                return Err(windows::core::Error::from(
                    windows::Win32::Foundation::E_INVALIDARG,
                ));
            }
        };
        let mut hazards = self.hazards.lock().unwrap_or_else(|e| e.into_inner());
        let mut waits = hazards.deps_for(target, reads, writes);
        for dep in explicit_deps {
            if dep.queue != target && dep.value > 0 {
                let idx = dep.queue.index();
                waits[idx] = waits[idx].max(dep.value);
            }
        }
        waits[target.index()] = 0;

        let token = {
            let mut engine = engine_mutex.lock().unwrap_or_else(|e| e.into_inner());
            // Insert all GPU-side `ConsumerQueue->Wait` calls strictly BEFORE `ExecuteCommandLists`.
            for q in QueueKind::ALL {
                let val = waits[q.index()];
                if q != target && val > 0 {
                    let producer_fence = self.fence_for(q);
                    // SAFETY: Same-device fence wait on `engine.queue` before list execution.
                    if let Err(err) = unsafe { engine.queue.Wait(producer_fence, val) } {
                        engine.idle.push(slot);
                        return Err(err);
                    }
                    STATS[ST_GPU_WAITS].fetch_add(1, Relaxed);
                }
            }
            let cmd_list: ID3D12CommandList = match slot.list.cast() {
                Ok(list) => list,
                Err(err) => {
                    engine.idle.push(slot);
                    return Err(err);
                }
            };
            let value = engine.sent + 1;
            // SAFETY: `cmd_list` is closed and its referenced resources are held in `keep`
            // until `own_fence` reaches `value`.
            let signal_res = unsafe {
                engine.queue.ExecuteCommandLists(&[Some(cmd_list)]);
                engine.queue.Signal(own_fence, value)
            };
            if let Err(err) = signal_res {
                // Keep `slot` and `keep` pinned in `abandoned` (never waited on via
                // `SetEventOnCompletion` and never popped by `acquire_open`) and do
                // not advance `engine.sent`.
                engine.abandoned.push(InFlight {
                    fence_value: 0,
                    slot,
                    _keep: keep,
                });
                return Err(err);
            }
            engine.sent = value;
            engine.in_flight.push_back(InFlight {
                fence_value: value,
                slot,
                _keep: keep,
            });
            TimelineToken {
                queue: target,
                value,
            }
        };

        if !reads.is_empty() || !writes.is_empty() {
            let completed = self.completed_snapshot();
            hazards.record_submission(token, reads, writes);
            hazards.prune(completed);
        }
        Ok(token)
    }

    /// Exposes the compute queue for GPU timestamp calibration / benchmarks in tests.
    #[cfg(test)]
    pub(crate) fn compute_queue_clone(&self) -> ID3D12CommandQueue {
        self.compute
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue
            .clone()
    }

    /// Exposes the copy queue for GPU timestamp calibration / benchmarks in tests.
    #[cfg(test)]
    pub(crate) fn copy_queue_clone(&self) -> ID3D12CommandQueue {
        self.copy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .queue
            .clone()
    }
}

impl Drop for AsyncTimeline {
    fn drop(&mut self) {
        let comp_sent = self.compute.lock().unwrap_or_else(|e| e.into_inner()).sent;
        let copy_sent = self.copy.lock().unwrap_or_else(|e| e.into_inner()).sent;
        if comp_sent > 0 {
            let _ = wait_fence_cpu(&self.compute_fence, comp_sent);
        }
        if copy_sent > 0 {
            let _ = wait_fence_cpu(&self.copy_fence, copy_sent);
        }
        self.reclaim();
    }
}

static ENGINE: OnceLock<Option<AsyncTimeline>> = OnceLock::new();

/// Returns the global timeline engine if it has already been initialized,
/// without creating any async queues on demand.
pub(crate) fn existing() -> Option<&'static AsyncTimeline> {
    ENGINE.get().and_then(Option::as_ref)
}

/// Lazily initialized global timeline engine for the host device.
pub(crate) fn global(device: &ID3D12Device) -> Option<&'static AsyncTimeline> {
    ENGINE
        .get_or_init(|| match AsyncTimeline::new(device) {
            Ok(engine) => {
                tracing::info!(
                    copy_timestamps = engine.copy_timestamps_supported,
                    "multi-queue hardware timeline engine ready (VICTORY_ASYNC_COMPUTE=1)"
                );
                Some(engine)
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "could not create async compute/copy queues; staying on direct queue"
                );
                None
            }
        })
        .as_ref()
}

/// Counts an offloaded card-buffer rehash pass for telemetry.
pub(crate) fn count_rehash() {
    STATS[ST_REHASH].fetch_add(1, Relaxed);
}

/// Counts an offloaded card-buffer compare pass for telemetry.
pub(crate) fn count_compare() {
    STATS[ST_COMPARE].fetch_add(1, Relaxed);
}

/// Logs per-frame multi-queue timeline statistics when active.
pub(crate) fn line(frames: f64) {
    let raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|i| STATS[i].swap(0, Relaxed));
    if raw[ST_COMPUTE] + raw[ST_COPY_UP] + raw[ST_COPY_RB] == 0 {
        return;
    }
    let f = |i: usize| raw[i] as f64 / frames;
    tracing::info!(
        "multi-queue hardware timeline per frame (VICTORY_ASYNC_COMPUTE=1): \
         compute {:.1} (rehash {:.1}, compare {:.1}), \
         copy uploads {:.1} ({:.1} MB), copy readbacks {:.1} ({:.1} MB), \
         GPU fence waits {:.1}, CPU waits avoided {:.1} / blocked {:.1}",
        f(ST_COMPUTE),
        f(ST_REHASH),
        f(ST_COMPARE),
        f(ST_COPY_UP),
        f(ST_UP_BYTES) / 1048576.0,
        f(ST_COPY_RB),
        f(ST_RB_BYTES) / 1048576.0,
        f(ST_GPU_WAITS),
        f(ST_CPU_AVOIDED),
        f(ST_CPU_BLOCKED),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::time::Instant;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
        IDXGIFactory4,
    };

    // -------------------------------------------------------------------------
    // Pure unit tests (CPU-only, deterministic)
    // -------------------------------------------------------------------------

    #[test]
    fn hazard_tracker_enforces_raw_war_waw_and_coalesces_per_queue() {
        let mut tracker = HazardTracker::default();
        let (buf_a, buf_b, buf_c) = (0x1000usize, 0x2000usize, 0x3000usize);

        // Copy writes A at copy=1, B at copy=2.
        tracker.record_submission(
            TimelineToken { queue: QueueKind::Copy, value: 1 },
            &[],
            &[buf_a],
        );
        tracker.record_submission(
            TimelineToken { queue: QueueKind::Copy, value: 2 },
            &[],
            &[buf_b],
        );

        // Compute reading both A and B coalesces to single wait copy=2 (RAW).
        assert_eq!(
            tracker.deps_for(QueueKind::Compute, &[buf_a, buf_b], &[]),
            [0, 0, 2]
        );
        // Copy itself needs no wait (FIFO same-queue).
        assert_eq!(
            tracker.deps_for(QueueKind::Copy, &[buf_a, buf_b], &[buf_a]),
            [0, 0, 0]
        );
        // Unrelated resource C needs zero waits (full overlap).
        assert_eq!(
            tracker.deps_for(QueueKind::Direct, &[buf_c], &[buf_c]),
            [0, 0, 0]
        );

        // Compute reads A and writes C at compute=5.
        tracker.record_submission(
            TimelineToken { queue: QueueKind::Compute, value: 5 },
            &[buf_a],
            &[buf_c],
        );

        // Direct reading A concurrently with Compute needs ONLY Copy wait (copy=1), NOT Compute!
        assert_eq!(
            tracker.deps_for(QueueKind::Direct, &[buf_a], &[]),
            [0, 0, 1]
        );
        // Direct writing A must wait on BOTH Copy writer (WAW: copy=1) and Compute reader (WAR: compute=5)!
        assert_eq!(
            tracker.deps_for(QueueKind::Direct, &[], &[buf_a]),
            [0, 5, 1]
        );
        // Direct reading C must wait on Compute writer (RAW: compute=5).
        assert_eq!(
            tracker.deps_for(QueueKind::Direct, &[buf_c], &[]),
            [0, 5, 0]
        );

        // After Direct writes A at direct=7, old Compute read is superseded.
        tracker.record_submission(
            TimelineToken { queue: QueueKind::Direct, value: 7 },
            &[],
            &[buf_a],
        );
        assert_eq!(
            tracker.deps_for(QueueKind::Copy, &[buf_a], &[]),
            [7, 0, 0]
        );
        assert_eq!(
            tracker.deps_for(QueueKind::Compute, &[], &[buf_a]),
            [7, 0, 0]
        );
    }

    #[test]
    fn state_validation_rejects_illegal_compute_and_copy_states() {
        assert!(is_copy_legal_state(D3D12_RESOURCE_STATE_COMMON));
        assert!(is_copy_legal_state(D3D12_RESOURCE_STATE_COPY_SOURCE));
        assert!(is_copy_legal_state(D3D12_RESOURCE_STATE_COPY_DEST));
        assert!(!is_copy_legal_state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        assert!(!is_copy_legal_state(
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        ));
        assert!(!is_copy_legal_state(D3D12_RESOURCE_STATE_RENDER_TARGET));

        assert!(is_compute_legal_state(D3D12_RESOURCE_STATE_COMMON));
        assert!(is_compute_legal_state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        assert!(is_compute_legal_state(
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        ));
        assert!(is_compute_legal_state(D3D12_RESOURCE_STATE_COPY_SOURCE));
        assert!(is_compute_legal_state(D3D12_RESOURCE_STATE_COPY_DEST));
        assert!(!is_compute_legal_state(D3D12_RESOURCE_STATE_RENDER_TARGET));
        assert!(!is_compute_legal_state(D3D12_RESOURCE_STATE_DEPTH_WRITE));
        assert!(!is_compute_legal_state(
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        ));
        assert!(!is_compute_legal_state(D3D12_RESOURCE_STATE_STREAM_OUT));
    }

    #[test]
    fn switch_defaults_off_and_parses_values() {
        assert!(!parse_switch(None));
        assert!(!parse_switch(Some(OsStr::new("0"))));
        assert!(!parse_switch(Some(OsStr::new("true"))));
        assert!(parse_switch(Some(OsStr::new("1"))));
    }

    // -------------------------------------------------------------------------
    // D3D12 HW + WARP helpers
    // -------------------------------------------------------------------------

    /// Returns `(adapter_name, is_software)` for `device`.
    fn adapter_info(device: &ID3D12Device) -> (String, bool) {
        // SAFETY: Plain DXGI adapter query by LUID.
        unsafe {
            let luid = device.GetAdapterLuid();
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return ("unknown".into(), false);
            };
            let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid) else {
                return ("unknown".into(), false);
            };
            let Ok(desc) = adapter.GetDesc1() else {
                return ("unknown".into(), false);
            };
            let name = String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_owned();
            let sw = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
            (name, sw)
        }
    }

    /// Drains any pre-existing debug layer messages before a test starts.
    fn clear_debug_queue(device: &ID3D12Device) {
        if let Ok(info) = device.cast::<ID3D12InfoQueue>() {
            // SAFETY: Valid `ID3D12InfoQueue` on our test device.
            unsafe { info.ClearStoredMessages() };
        }
    }

    /// Asserts ZERO D3D12 debug layer corruptions, errors, or warnings.
    fn assert_zero_d3d12_debug_issues(device: &ID3D12Device, context: &str) {
        let Ok(info) = device.cast::<ID3D12InfoQueue>() else {
            return;
        };
        let mut issues = Vec::new();
        // SAFETY: Reading stored messages from `ID3D12InfoQueue` into properly sized buffers.
        unsafe {
            let count = info.GetNumStoredMessages();
            for i in 0..count {
                let mut size = 0usize;
                if info.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buf = vec![0u64; size.div_ceil(8)];
                let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if info.GetMessage(i, Some(msg), &mut size).is_ok() {
                    let m = &*msg;
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                        let bytes =
                            std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(bytes)
                            .trim_end_matches('\0')
                            .to_owned();
                        issues.push(format!("[sev={} id={}] {text}", m.Severity.0, m.ID.0));
                    }
                }
            }
            info.ClearStoredMessages();
        }
        assert!(
            issues.is_empty(),
            "{context}: expected 0 D3D12 debug errors/warnings, got {}:\n{}",
            issues.len(),
            issues.join("\n")
        );
    }

    fn make_committed_buffer(
        device: &ID3D12Device,
        heap_type: D3D12_HEAP_TYPE,
        bytes: u64,
        flags: D3D12_RESOURCE_FLAGS,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: Valid committed buffer descriptor on a live device.
        unsafe {
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap_type,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: bytes,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: flags,
                        ..Default::default()
                    },
                    initial_state,
                    None,
                    &mut out,
                )
                .unwrap();
        }
        out.unwrap()
    }

    fn make_compute_pso(
        device: &ID3D12Device,
        hlsl: &str,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let bytecode =
            crate::host::compile(hlsl, windows::core::s!("main"), windows::core::s!("cs_5_1"))
                .unwrap();
        // SAFETY: Compiled bytecode contains an embedded root signature and valid CS.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &bytecode).unwrap();
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Affine transform compute shader: `dst[i] = src[i] * mul + add` (with `rounds`
    /// ALU iterations for controlled GPU workload duration in overlap benchmarks).
    const TRANSFORM_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0)"
cbuffer Params : register(b0) {
    uint count;
    uint mul;
    uint add;
    uint rounds;
};
ByteAddressBuffer src_buf : register(t0);
RWByteAddressBuffer dst_buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint v = src_buf.Load(id.x * 4u);
    [loop]
    for (uint r = 0u; r < rounds; ++r) {
        v = v * mul + add;
    }
    dst_buf.Store(id.x * 4u, v);
}
"#;

    /// In-place read-modify-write compute shader for strict RAW/WAR/WAW hazard testing.
    const INPLACE_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0)"
cbuffer Params : register(b0) {
    uint count;
    uint mul;
    uint add;
};
RWByteAddressBuffer buf : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint v = buf.Load(id.x * 4u);
    buf.Store(id.x * 4u, v * mul + add);
}
"#;

    fn cpu_transform(mut v: u32, mul: u32, add: u32, rounds: u32) -> u32 {
        for _ in 0..rounds {
            v = v.wrapping_mul(mul).wrapping_add(add);
        }
        v
    }

    fn qpc_frequency() -> f64 {
        let mut freq = 0i64;
        // SAFETY: Standard Win32 query writing a single `i64`.
        unsafe {
            windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut freq);
        }
        freq.max(1) as f64
    }

    // -------------------------------------------------------------------------
    // D3D12 HW + WARP Correctness & Hazard Tests
    // -------------------------------------------------------------------------

    /// Proves 8-flight pipelined Copy -> Async Compute -> Direct -> Copy Readback
    /// executes with ZERO CPU fence waits in the submission loop, 100% bit-exact
    /// readback across all flights, and 0 D3D12 Debug Layer errors/warnings.
    #[test]
    fn multi_flight_copy_compute_direct_readback_ordering() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = AsyncTimeline::new(device).unwrap();
        let (root, pso) = make_compute_pso(device, TRANSFORM_CS);

        const FLIGHTS: usize = 8;
        const WORDS: u32 = 16_384; // 64 KiB per flight
        const BYTES: u64 = (WORDS as u64) * 4;

        // Dedicated Direct queue + per-flight direct allocators so this test never
        // blocks on `Host::submitted` and keeps zero CPU waits in the submit loop.
        // SAFETY: Standard D3D12 direct queue and command list creation.
        let (direct_queue, direct_slots): (ID3D12CommandQueue, Vec<CommandSlot>) = unsafe {
            let q: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let mut slots = Vec::with_capacity(FLIGHTS);
            for _ in 0..FLIGHTS {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                list.Close().unwrap();
                slots.push(CommandSlot { allocator, list });
            }
            (q, slots)
        };

        let mut uploads = Vec::with_capacity(FLIGHTS);
        let mut vram_src = Vec::with_capacity(FLIGHTS);
        let mut vram_mid = Vec::with_capacity(FLIGHTS);
        let mut vram_dst = Vec::with_capacity(FLIGHTS);
        let mut readbacks = Vec::with_capacity(FLIGHTS);

        for f in 0..FLIGHTS {
            let up = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            // SAFETY: Mapping fresh upload buffer of size `BYTES`.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                up.Map(0, None, Some(&mut ptr)).unwrap();
                let slice = std::slice::from_raw_parts_mut(ptr.cast::<u32>(), WORDS as usize);
                for (i, slot) in slice.iter_mut().enumerate() {
                    *slot = ((f as u32 + 1) * 0x0101_0001u32) ^ (i as u32).wrapping_mul(0x9E37_79B9);
                }
                up.Unmap(0, None);
            }
            uploads.push(up);
            vram_src.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                BYTES,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            vram_mid.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                BYTES,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            vram_dst.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                BYTES,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            readbacks.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ));
        }

        let cpu_waits_before = STATS[ST_CPU_BLOCKED].load(Relaxed);
        let gpu_waits_before = STATS[ST_GPU_WAITS].load(Relaxed);
        let mut rb_tokens = Vec::with_capacity(FLIGHTS);

        // Hot submission loop: ZERO CPU fence waits!
        for f in 0..FLIGHTS {
            let mul_c = 1_664_525u32;
            let add_c = 1_013_904_223u32 ^ (f as u32);
            let mul_d = 2_246_822_519u32;
            let add_d = 326_648_991u32 + (f as u32) * 17;

            // Stage 1: Copy queue uploads UPLOAD -> vram_src[f].
            let src_ref = vram_src[f].clone();
            let up_ref = uploads[f].clone();
            let copy_tok = timeline
                .submit_copy(
                    &[],
                    &[resource_key(&uploads[f])],
                    &[resource_key(&vram_src[f])],
                    vec![uploads[f].clone(), vram_src[f].clone()],
                    move |list| {
                        // SAFETY: Valid copy between upload buffer and COMMON default buffer.
                        unsafe { list.CopyBufferRegion(&src_ref, 0, &up_ref, 0, BYTES) };
                        Ok(())
                    },
                )
                .unwrap();

            // Stage 2: Async Compute waits on Copy on GPU and transforms vram_src -> vram_mid.
            let src_va = unsafe { vram_src[f].GetGPUVirtualAddress() };
            let mid_va = unsafe { vram_mid[f].GetGPUVirtualAddress() };
            let root_c = root.clone();
            let pso_c = pso.clone();
            let comp_tok = timeline
                .submit_compute(
                    &[copy_tok],
                    &[resource_key(&vram_src[f])],
                    &[resource_key(&vram_mid[f])],
                    vec![vram_src[f].clone(), vram_mid[f].clone()],
                    move |list| {
                        let params = [WORDS, mul_c, add_c, 3u32];
                        // SAFETY: Valid compute root signature, PSO, and COMMON-promoted buffers.
                        unsafe {
                            list.SetComputeRootSignature(&root_c);
                            list.SetPipelineState(&pso_c);
                            list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, src_va);
                            list.SetComputeRootUnorderedAccessView(2, mid_va);
                            list.Dispatch(WORDS.div_ceil(64), 1, 1);
                        }
                        Ok(())
                    },
                )
                .unwrap();

            // Stage 3: Direct queue atomically waits on Async Compute on GPU (via HazardTracker!)
            // and transforms vram_mid -> vram_dst.
            let dst_va = unsafe { vram_dst[f].GetGPUVirtualAddress() };
            let params_d = [WORDS, mul_d, add_d, 2u32];
            // SAFETY: Direct slot `f` is dedicated to flight `f` and currently idle.
            let direct_cmd: ID3D12CommandList = unsafe {
                let slot = &direct_slots[f];
                slot.allocator.Reset().unwrap();
                slot.list.Reset(&slot.allocator, Some(&pso)).unwrap();
                slot.list.SetComputeRootSignature(&root);
                slot.list
                    .SetComputeRoot32BitConstants(0, 4, params_d.as_ptr().cast(), 0);
                slot.list.SetComputeRootShaderResourceView(1, mid_va);
                slot.list.SetComputeRootUnorderedAccessView(2, dst_va);
                slot.list.Dispatch(WORDS.div_ceil(64), 1, 1);
                slot.list.Close().unwrap();
                slot.list.cast().unwrap()
            };
            let ((), direct_tok, gpu_waits) = timeline
                .execute_direct_tracked(
                    &direct_queue,
                    &[resource_key(&vram_mid[f])],
                    &[resource_key(&vram_dst[f])],
                    || {
                        // SAFETY: Closed Direct list with live resources.
                        unsafe { direct_queue.ExecuteCommandLists(&[Some(direct_cmd)]) };
                    },
                )
                .unwrap();
            assert_eq!(gpu_waits, 1, "direct must wait on compute for vram_mid");

            // Stage 4: Copy queue waits on Direct on GPU and reads back vram_dst -> readbacks[f].
            let rb_tok = timeline
                .submit_copy_readback(&[direct_tok], &readbacks[f], &[(&vram_dst[f], 0, 0, BYTES)])
                .unwrap();
            let _ = comp_tok;
            rb_tokens.push(rb_tok);
        }

        let cpu_waits_in_loop = STATS[ST_CPU_BLOCKED].load(Relaxed) - cpu_waits_before;
        let gpu_waits_inserted = STATS[ST_GPU_WAITS].load(Relaxed) - gpu_waits_before;
        assert_eq!(
            cpu_waits_in_loop, 0,
            "hot submission loop must perform ZERO CPU fence waits"
        );
        assert!(
            gpu_waits_inserted >= (FLIGHTS * 3) as u64,
            "expected >= {} GPU waits across 8x 4-stage flights, got {gpu_waits_inserted}",
            FLIGHTS * 3
        );

        // Now wait only at final readback consumption and verify 100% bit-exact parity.
        for (f, &tok) in rb_tokens.iter().enumerate() {
            timeline.wait_cpu(tok).unwrap();
            let mul_c = 1_664_525u32;
            let add_c = 1_013_904_223u32 ^ (f as u32);
            let mul_d = 2_246_822_519u32;
            let add_d = 326_648_991u32 + (f as u32) * 17;
            // SAFETY: Copy readback fence `tok` has completed on GPU.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                readbacks[f].Map(0, None, Some(&mut ptr)).unwrap();
                let out = std::slice::from_raw_parts(ptr.cast::<u32>(), WORDS as usize);
                for (i, &actual) in out.iter().enumerate() {
                    let init = ((f as u32 + 1) * 0x0101_0001u32) ^ (i as u32).wrapping_mul(0x9E37_79B9);
                    let mid = cpu_transform(init, mul_c, add_c, 3);
                    let expected = cpu_transform(mid, mul_d, add_d, 2);
                    assert_eq!(
                        actual, expected,
                        "flight {f} dword {i} mismatch (adapter={adapter_name}, sw={is_sw})"
                    );
                }
                readbacks[f].Unmap(0, None);
            }
        }

        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "multi_flight_copy_compute_direct_readback_ordering");
        eprintln!(
            "[async_timeline] multi_flight_copy_compute_direct_readback_ordering PASS \
             (adapter={adapter_name}, sw={is_sw}, flights={FLIGHTS}, gpu_waits={gpu_waits_inserted}, cpu_waits_in_submit_loop=0)"
        );
    }

    /// Hammers a SINGLE shared VRAM buffer across 16 alternating Copy / Compute / Direct
    /// mutations and readbacks, relying strictly on `HazardTracker`'s GPU-side fence waits
    /// to prevent RAW, WAR, and WAW hazards.
    #[test]
    fn shared_buffer_raw_war_waw_hazard_stress() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = AsyncTimeline::new(device).unwrap();
        let (root, pso) = make_compute_pso(device, INPLACE_CS);

        const ROUNDS: usize = 8;
        const WORDS: u32 = 4096;
        const BYTES: u64 = (WORDS as u64) * 4;

        // SAFETY: Standard Direct queue and command list creation for stress test.
        let (direct_queue, direct_slots): (ID3D12CommandQueue, Vec<CommandSlot>) = unsafe {
            let q: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let mut slots = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                list.Close().unwrap();
                slots.push(CommandSlot { allocator, list });
            }
            (q, slots)
        };

        let shared = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            BYTES,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let shared_key = resource_key(&shared);
        let shared_va = unsafe { shared.GetGPUVirtualAddress() };

        let mut uploads = Vec::with_capacity(ROUNDS);
        let mut readbacks = Vec::with_capacity(ROUNDS);
        let mut rb_tokens = Vec::with_capacity(ROUNDS);
        let mut expected_snapshots = Vec::with_capacity(ROUNDS);

        for r in 0..ROUNDS {
            let up = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut init_words = vec![0u32; WORDS as usize];
            for (i, w) in init_words.iter_mut().enumerate() {
                *w = ((r as u32 + 3) * 0x1357_9BDFu32).wrapping_add(i as u32 * 29);
            }
            // SAFETY: Mapping fresh upload buffer of `BYTES` bytes.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                up.Map(0, None, Some(&mut ptr)).unwrap();
                std::ptr::copy_nonoverlapping(
                    init_words.as_ptr(),
                    ptr.cast::<u32>(),
                    WORDS as usize,
                );
                up.Unmap(0, None);
            }
            let rb = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );

            // 1) Copy overwrites `shared` (must wait for prior round's Copy readback & Direct/Compute).
            let dst_ref = shared.clone();
            let up_ref = up.clone();
            timeline
                .submit_copy(
                    &[],
                    &[resource_key(&up)],
                    &[shared_key],
                    vec![up.clone(), shared.clone()],
                    move |list| {
                        // SAFETY: Valid copy into `COMMON` buffer.
                        unsafe { list.CopyBufferRegion(&dst_ref, 0, &up_ref, 0, BYTES) };
                        Ok(())
                    },
                )
                .unwrap();

            // 2) Compute mutates `shared` in place (automatic RAW/WAW wait on Copy!).
            let mul1 = 1_103_515_245u32;
            let add1 = 12_345u32 + r as u32 * 97;
            let root_c = root.clone();
            let pso_c = pso.clone();
            timeline
                .submit_compute(
                    &[],
                    &[shared_key],
                    &[shared_key],
                    vec![shared.clone()],
                    move |list| {
                        let params = [WORDS, mul1, add1];
                        // SAFETY: Valid in-place compute dispatch on `COMMON`-promoted UAV buffer.
                        unsafe {
                            list.SetComputeRootSignature(&root_c);
                            list.SetPipelineState(&pso_c);
                            list.SetComputeRoot32BitConstants(0, 3, params.as_ptr().cast(), 0);
                            list.SetComputeRootUnorderedAccessView(1, shared_va);
                            list.Dispatch(WORDS.div_ceil(64), 1, 1);
                        }
                        Ok(())
                    },
                )
                .unwrap();

            // 3) Direct mutates `shared` in place atomically (GPU wait on Compute!).
            let mul2 = 214_013u32;
            let add2 = 2_531_011u32 ^ (r as u32 * 13);
            let params_d = [WORDS, mul2, add2];
            // SAFETY: Direct slot `r` is dedicated and idle.
            let direct_cmd: ID3D12CommandList = unsafe {
                let slot = &direct_slots[r];
                slot.allocator.Reset().unwrap();
                slot.list.Reset(&slot.allocator, Some(&pso)).unwrap();
                slot.list.SetComputeRootSignature(&root);
                slot.list
                    .SetComputeRoot32BitConstants(0, 3, params_d.as_ptr().cast(), 0);
                slot.list.SetComputeRootUnorderedAccessView(1, shared_va);
                slot.list.Dispatch(WORDS.div_ceil(64), 1, 1);
                slot.list.Close().unwrap();
                slot.list.cast().unwrap()
            };
            let ((), _direct_tok, waits) = timeline
                .execute_direct_tracked(&direct_queue, &[shared_key], &[shared_key], || {
                    // SAFETY: Closed Direct list with live resources.
                    unsafe { direct_queue.ExecuteCommandLists(&[Some(direct_cmd)]) };
                })
                .unwrap();
            assert_eq!(waits, 1, "direct must wait on compute for shared buffer");

            // 4) Copy reads back snapshot of `shared` BEFORE next round overwrites it!
            let rb_tok = timeline
                .submit_copy_readback(&[], &rb, &[(&shared, 0, 0, BYTES)])
                .unwrap();

            let expected: Vec<u32> = init_words
                .into_iter()
                .map(|v| {
                    let s1 = v.wrapping_mul(mul1).wrapping_add(add1);
                    s1.wrapping_mul(mul2).wrapping_add(add2)
                })
                .collect();
            uploads.push(up);
            readbacks.push(rb);
            rb_tokens.push(rb_tok);
            expected_snapshots.push(expected);
        }

        // Wait for final round and verify EVERY intermediate snapshot was captured
        // in exact order without next round's Copy overwrite tearing it!
        timeline.wait_cpu(*rb_tokens.last().unwrap()).unwrap();
        for r in 0..ROUNDS {
            // SAFETY: All rounds up to final readback have completed on GPU.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                readbacks[r].Map(0, None, Some(&mut ptr)).unwrap();
                let actual = std::slice::from_raw_parts(ptr.cast::<u32>(), WORDS as usize);
                assert_eq!(
                    actual,
                    expected_snapshots[r].as_slice(),
                    "hazard stress round {r} snapshot corrupted (adapter={adapter_name}, sw={is_sw})"
                );
                readbacks[r].Unmap(0, None);
            }
        }
        drop(uploads);
        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "shared_buffer_raw_war_waw_hazard_stress");
        eprintln!(
            "[async_timeline] shared_buffer_raw_war_waw_hazard_stress PASS \
             (adapter={adapter_name}, sw={is_sw}, rounds={ROUNDS}, 0 RAW/WAR/WAW hazards)"
        );
    }

    /// Runs the real `CARD_COMPARE` hash/diff shader on the dedicated Async Compute
    /// queue chained with Copy uploads, Direct mutations, and Copy readbacks,
    /// proving bit-exact change detection and 0 D3D12 Debug Layer errors/warnings.
    #[test]
    fn async_card_compare_matches_direct_bit_for_bit() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = AsyncTimeline::new(device).unwrap();
        let (cmp_root, cmp_pso) = make_compute_pso(device, crate::host::CARD_COMPARE);

        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 64;
        const BUF_BYTES: u64 = (PIECES as u64) * (PIECE as u64); // 4 MiB

        let vram_card = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            BUF_BYTES,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let hashes = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            u64::from(PIECES + 2) * 16,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let (differs, differs_ptr) = host.shared_buffer(u64::from(PIECES) * 4).unwrap();
        let upload_init = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            BUF_BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let upload_patch = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            PIECE as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let readback = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_READBACK,
            PIECE as u64 * 4,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );

        // Populate initial 4 MiB card image and 64 KiB patch.
        // SAFETY: Fresh upload buffers mapped to their full lengths.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            upload_init.Map(0, None, Some(&mut ptr)).unwrap();
            let init_slice = std::slice::from_raw_parts_mut(ptr.cast::<u8>(), BUF_BYTES as usize);
            for (i, b) in init_slice.iter_mut().enumerate() {
                *b = ((i.wrapping_mul(31) ^ (i >> 8)) & 0xff) as u8;
            }
            upload_init.Unmap(0, None);

            let mut pptr = std::ptr::null_mut();
            upload_patch.Map(0, None, Some(&mut pptr)).unwrap();
            std::ptr::write_bytes(pptr.cast::<u8>(), 0xA5, PIECE);
            upload_patch.Unmap(0, None);
            std::ptr::write_bytes(differs_ptr, 0, PIECES as usize * 4);
        }

        // Direct queue for simulating title-list mutations.
        // SAFETY: Valid Direct queue and command list creation.
        let (direct_queue, direct_alloc, direct_list): (
            ID3D12CommandQueue,
            ID3D12CommandAllocator,
            ID3D12GraphicsCommandList,
        ) = unsafe {
            let q: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let a: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let l: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)
                .unwrap();
            l.Close().unwrap();
            (q, a, l)
        };

        // Step 1: Copy uploads initial 4 MiB -> vram_card.
        let vc = vram_card.clone();
        let ui = upload_init.clone();
        timeline
            .submit_copy(
                &[],
                &[resource_key(&upload_init)],
                &[resource_key(&vram_card)],
                vec![upload_init.clone(), vram_card.clone()],
                move |list| {
                    // SAFETY: Valid copy into `COMMON` buffer.
                    unsafe { list.CopyBufferRegion(&vc, 0, &ui, 0, BUF_BYTES) };
                    Ok(())
                },
            )
            .unwrap();

        // Step 2: Async Compute runs CARD_COMPARE mode 0 (initial hashes).
        let card_va = unsafe { vram_card.GetGPUVirtualAddress() };
        let hashes_va = unsafe { hashes.GetGPUVirtualAddress() };
        let differs_va = unsafe { differs.GetGPUVirtualAddress() };
        {
            let r = cmp_root.clone();
            let p = cmp_pso.clone();
            timeline
                .submit_compute(
                    &[],
                    &[resource_key(&vram_card)],
                    &[resource_key(&hashes)],
                    vec![vram_card.clone(), hashes.clone(), differs.clone()],
                    move |list| {
                        let constants = [0u32, 0, BUF_BYTES as u32, PIECES, 0, 0];
                        // SAFETY: Valid `CARD_COMPARE` mode 0 dispatch on compute list.
                        unsafe {
                            list.SetComputeRootSignature(&r);
                            list.SetPipelineState(&p);
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, card_va);
                            list.SetComputeRootUnorderedAccessView(2, hashes_va);
                            list.SetComputeRootUnorderedAccessView(3, differs_va);
                            list.Dispatch(PIECES, 1, 1);
                        }
                        Ok(())
                    },
                )
                .unwrap();
        }

        // Step 3: Direct mutates pieces {3, 17, 42, 63} atomically (GPU waits on Copy WAW + Compute WAR!).
        let mutated = [3u32, 17, 42, 63];
        // SAFETY: Direct allocator is idle; copying 64 KiB patch into 4 disjoint pieces.
        let direct_cmd: ID3D12CommandList = unsafe {
            direct_alloc.Reset().unwrap();
            direct_list.Reset(&direct_alloc, None).unwrap();
            for &piece_idx in &mutated {
                direct_list.CopyBufferRegion(
                    &vram_card,
                    u64::from(piece_idx) * PIECE as u64,
                    &upload_patch,
                    0,
                    PIECE as u64,
                );
            }
            direct_list.Close().unwrap();
            direct_list.cast().unwrap()
        };
        let ((), direct_tok, waits) = timeline
            .execute_direct_tracked(&direct_queue, &[], &[resource_key(&vram_card)], || {
                // SAFETY: Closed Direct list with live resources.
                unsafe { direct_queue.ExecuteCommandLists(&[Some(direct_cmd)]) };
            })
            .unwrap();
        assert_eq!(
            waits, 2,
            "direct must wait on both Copy (WAW) and Compute (WAR) before mutating vram_card"
        );

        // Step 4: Async Compute runs CARD_COMPARE mode 2 (compare + update hashes).
        let cmp_tok = {
            let r = cmp_root.clone();
            let p = cmp_pso.clone();
            timeline
                .submit_compute(
                    &[direct_tok],
                    &[resource_key(&vram_card)],
                    &[resource_key(&hashes), resource_key(&differs)],
                    vec![vram_card.clone(), hashes.clone(), differs.clone()],
                    move |list| {
                        let constants = [0u32, 0, BUF_BYTES as u32, PIECES, 0, 2];
                        // SAFETY: Valid `CARD_COMPARE` mode 2 dispatch on compute list.
                        unsafe {
                            list.SetComputeRootSignature(&r);
                            list.SetPipelineState(&p);
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, card_va);
                            list.SetComputeRootUnorderedAccessView(2, hashes_va);
                            list.SetComputeRootUnorderedAccessView(3, differs_va);
                            list.Dispatch(PIECES, 1, 1);
                        }
                        Ok(())
                    },
                )
                .unwrap()
        };

        // Wait on Async Compute before inspecting `differs` on CPU, then read back flagged pieces on Copy queue.
        timeline.wait_cpu(cmp_tok).unwrap();
        // SAFETY: `cmp_tok` has completed; `differs_ptr` holds `PIECES` u32 flags.
        let flagged: Vec<u32> = unsafe {
            let slice = std::slice::from_raw_parts(differs_ptr.cast::<u32>(), PIECES as usize);
            (0..PIECES).filter(|&p| (slice[p as usize] & 1) != 0).collect()
        };
        assert_eq!(flagged, mutated, "CARD_COMPARE on async compute must flag exact mutated pieces");

        let rb_copies: Vec<(&ID3D12Resource, u64, u64, u64)> = flagged
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                (
                    &vram_card,
                    u64::from(p) * PIECE as u64,
                    (i * PIECE) as u64,
                    PIECE as u64,
                )
            })
            .collect();
        let rb_tok = timeline
            .submit_copy_readback(&[cmp_tok], &readback, &rb_copies)
            .unwrap();
        timeline.wait_cpu(rb_tok).unwrap();

        // SAFETY: `rb_tok` has completed; verify all 4 readback pieces hold `0xA5`.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), PIECE * 4);
            assert!(bytes.iter().all(|&b| b == 0xA5), "readback pieces must match 0xA5 patch");
            readback.Unmap(0, None);
        }

        // Second compare with no further mutations must flag ZERO pieces (mode 2 updated hashes).
        let cmp_tok2 = {
            let r = cmp_root.clone();
            let p = cmp_pso.clone();
            timeline
                .submit_compute(
                    &[],
                    &[resource_key(&vram_card)],
                    &[resource_key(&hashes), resource_key(&differs)],
                    vec![vram_card.clone(), hashes.clone(), differs.clone()],
                    move |list| {
                        let constants = [0u32, 0, BUF_BYTES as u32, PIECES, 0, 2];
                        // SAFETY: Valid `CARD_COMPARE` mode 2 dispatch; outer `cmp_root` and
                        // `cmp_pso` remain alive until after `timeline.drain_all()`.
                        unsafe {
                            list.SetComputeRootSignature(&r);
                            list.SetPipelineState(&p);
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, card_va);
                            list.SetComputeRootUnorderedAccessView(2, hashes_va);
                            list.SetComputeRootUnorderedAccessView(3, differs_va);
                            list.Dispatch(PIECES, 1, 1);
                        }
                        Ok(())
                    },
                )
                .unwrap()
        };
        timeline.wait_cpu(cmp_tok2).unwrap();
        // SAFETY: `cmp_tok2` has completed.
        let flagged2: Vec<u32> = unsafe {
            let slice = std::slice::from_raw_parts(differs_ptr.cast::<u32>(), PIECES as usize);
            (0..PIECES).filter(|&p| (slice[p as usize] & 1) != 0).collect()
        };
        assert!(flagged2.is_empty(), "unchanged buffer must flag 0 pieces on second compare");

        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "async_card_compare_matches_direct_bit_for_bit");
        eprintln!(
            "[async_timeline] async_card_compare_matches_direct_bit_for_bit PASS \
             (adapter={adapter_name}, sw={is_sw}, 64 pieces verified, 0 debug issues)"
        );
    }

    // -------------------------------------------------------------------------
    // Real ID3D12QueryHeap + Wall-Clock Multi-Queue Overlap Benchmark
    // -------------------------------------------------------------------------

    /// Measures real GPU timestamp (`ID3D12QueryHeap`) and wall-clock execution time
    /// for identical multi-flight workloads (DMA uploads + Direct compute + Async Compute
    /// `CARD_COMPARE` hashing) comparing:
    /// 1) Serialized single Direct queue (baseline), vs
    /// 2) Multi-queue `AsyncTimeline` (Copy DMA + Async Compute + Direct overlapped via
    ///    GPU-side `ID3D12Fence` handoffs).
    #[test]
    fn benchmark_multi_queue_overlap_vs_serialized_direct() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = AsyncTimeline::new(device).unwrap();
        let (xform_root, xform_pso) = make_compute_pso(device, TRANSFORM_CS);
        let (cmp_root, cmp_pso) = make_compute_pso(device, crate::host::CARD_COMPARE);

        // Scale down on WARP software emulation so tests complete quickly while
        // still exercising the full 3-queue query-heap + overlap verification.
        // On discrete hardware, balance Copy DMA, Direct compute, and Async Compute
        // durations into the multi-millisecond regime so hardware overlap is clearly
        // measurable above submission latency.
        let (flights, upload_mb, direct_words, direct_rounds, cmp_pieces, cmp_reps, iterations): (
            usize,
            u64,
            u32,
            u32,
            u32,
            u32,
            usize,
        ) = if is_sw {
            (4, 2, 16_384, 32, 32, 1, 2)
        } else {
            (4, 48, 262_144, 52_000, 512, 16, 4)
        };
        let upload_bytes = upload_mb << 20;
        let direct_bytes = u64::from(direct_words) * 4;
        let cmp_bytes = u64::from(cmp_pieces) * (64 << 10);

        // SAFETY: Creating a dedicated Direct queue, fence, and timestamp query heaps.
        let (
            direct_queue,
            direct_fence,
            direct_qh,
            compute_qh,
            copy_qh,
            direct_ts_rb,
            compute_ts_rb,
            copy_ts_rb,
        ) = unsafe {
            let dq: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let df: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let mut dqh: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: (flights * 2) as u32,
                        NodeMask: 0,
                    },
                    &mut dqh,
                )
                .unwrap();
            let mut cqh: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: (flights * 2) as u32,
                        NodeMask: 0,
                    },
                    &mut cqh,
                )
                .unwrap();
            let cpqh = if timeline.copy_timestamps_supported() {
                let mut h: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP,
                            Count: (flights * 2) as u32,
                            NodeMask: 0,
                        },
                        &mut h,
                    )
                    .ok()
                    .and(h)
            } else {
                None
            };
            let ts_bytes = (flights * 2 * 8) as u64;
            let d_rb = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                ts_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let c_rb = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                ts_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let cp_rb = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                ts_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            (
                dq,
                df,
                dqh.unwrap(),
                cqh.unwrap(),
                cpqh,
                d_rb,
                c_rb,
                cp_rb,
            )
        };

        // Allocate per-flight independent resources (upload -> VRAM bounce, Direct src/dst,
        // Compute CARD_COMPARE buffer + hashes + differs).
        let mut up_bufs = Vec::with_capacity(flights);
        let mut bounce_bufs = Vec::with_capacity(flights);
        let mut direct_src = Vec::with_capacity(flights);
        let mut direct_dst = Vec::with_capacity(flights);
        let mut cmp_bufs = Vec::with_capacity(flights);
        let mut cmp_hashes = Vec::with_capacity(flights);
        let mut cmp_differs = Vec::with_capacity(flights);
        let mut direct_rb_check = Vec::with_capacity(flights);

        let seed_upload = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            direct_bytes.max(cmp_bytes),
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: Populating seed upload buffer once.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            seed_upload.Map(0, None, Some(&mut ptr)).unwrap();
            let words = std::slice::from_raw_parts_mut(
                ptr.cast::<u32>(),
                (direct_bytes.max(cmp_bytes) / 4) as usize,
            );
            for (i, w) in words.iter_mut().enumerate() {
                *w = (i as u32).wrapping_mul(0x9E37_79B1) ^ 0x5A5A_A5A5;
            }
            seed_upload.Unmap(0, None);
        }

        for f in 0..flights {
            let up = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                upload_bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            // SAFETY: Touch first & last page so upload pages are resident.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                up.Map(0, None, Some(&mut ptr)).unwrap();
                ptr.cast::<u32>().write((f as u32) + 1);
                up.Unmap(0, None);
            }
            up_bufs.push(up);
            bounce_bufs.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                upload_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            direct_src.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                direct_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            direct_dst.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                direct_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            cmp_bufs.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                cmp_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            cmp_hashes.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                u64::from(cmp_pieces + 2) * 16,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            cmp_differs.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                u64::from(cmp_pieces) * 4,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            ));
            direct_rb_check.push(make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_READBACK,
                256,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ));
        }

        // Seed `direct_src` and `cmp_bufs` once via Copy queue.
        let init_tok = timeline
            .submit_copy(&[], &[], &[], vec![seed_upload.clone()], |list| {
                for f in 0..flights {
                    // SAFETY: Valid initial seed copies into `COMMON` VRAM buffers.
                    unsafe {
                        list.CopyBufferRegion(&direct_src[f], 0, &seed_upload, 0, direct_bytes);
                        list.CopyBufferRegion(&cmp_bufs[f], 0, &seed_upload, 0, cmp_bytes);
                    }
                }
                Ok(())
            })
            .unwrap();
        timeline.wait_cpu(init_tok).unwrap();

        // Direct command allocators/lists for serialized and overlapped runs.
        // SAFETY: Creating per-flight Direct allocators/lists on live device.
        let direct_slots: Vec<CommandSlot> = unsafe {
            (0..flights)
                .map(|_| {
                    let allocator: ID3D12CommandAllocator = device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .unwrap();
                    let list: ID3D12GraphicsCommandList = device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                        .unwrap();
                    list.Close().unwrap();
                    CommandSlot { allocator, list }
                })
                .collect()
        };

        let compute_queue = timeline.compute_queue_clone();
        let copy_queue = timeline.copy_queue_clone();
        // SAFETY: Valid timestamp frequency queries on live queues.
        let (direct_freq, compute_freq, copy_freq) = unsafe {
            let df = direct_queue.GetTimestampFrequency().unwrap() as f64;
            let cf = compute_queue.GetTimestampFrequency().unwrap() as f64;
            let cpf = if copy_qh.is_some() {
                copy_queue.GetTimestampFrequency().ok().map(|v| v as f64)
            } else {
                None
            };
            (df, cf, cpf)
        };
        let qpc_freq = qpc_frequency();

        let mut direct_fence_val = 0u64;

        let read_timestamps = |rb: &ID3D12Resource, count: usize| -> Vec<u64> {
            // SAFETY: Caller has waited on the GPU fence covering `rb`.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let out = std::slice::from_raw_parts(ptr.cast::<u64>(), count).to_vec();
                rb.Unmap(0, None);
                out
            }
        };

        let calibrate_us = |q: &ID3D12CommandQueue, freq: f64| -> (u64, u64, f64) {
            let (mut gpu, mut cpu) = (0u64, 0u64);
            // SAFETY: Plain clock calibration query on a live queue.
            unsafe {
                let _ = q.GetClockCalibration(&mut gpu, &mut cpu);
            }
            (gpu, cpu, freq)
        };

        let to_qpc_us = |tick: u64, (gpu0, cpu0, freq): (u64, u64, f64)| -> f64 {
            let cpu_us = (cpu0 as f64 / qpc_freq) * 1e6;
            let delta_gpu = tick as i128 - gpu0 as i128;
            cpu_us + (delta_gpu as f64 / freq) * 1e6
        };

        let mut serial_walls_ms = Vec::new();
        let mut serial_gpu_ms = Vec::new();
        let mut overlap_walls_ms = Vec::new();
        let mut overlap_gpu_union_ms = Vec::new();
        let mut overlap_direct_busy_ms = Vec::new();
        let mut overlap_compute_busy_ms = Vec::new();
        let mut overlap_copy_busy_ms = Vec::new();
        let mut overlap_intersect_ms = Vec::new();
        let mut final_checksum = 0u32;

        // Run warmup (iter 0) + measured iterations.
        for iter in 0..=iterations {
            // -----------------------------------------------------------------
            // Mode A: Serialized Direct Queue (Copy + Direct + Compute all on Direct)
            // -----------------------------------------------------------------
            let t0 = Instant::now();
            // SAFETY: All direct slots are idle; recording serialized Copy + Direct + Compare.
            unsafe {
                for f in 0..flights {
                    let slot = &direct_slots[f];
                    slot.allocator.Reset().unwrap();
                    slot.list.Reset(&slot.allocator, None).unwrap();
                    slot.list
                        .EndQuery(&direct_qh, D3D12_QUERY_TYPE_TIMESTAMP, (f * 2) as u32);
                    // 1) PCIe Upload DMA on Direct
                    slot.list
                        .CopyBufferRegion(&bounce_bufs[f], 0, &up_bufs[f], 0, upload_bytes);
                    // 2) Direct compute workload
                    let params = [direct_words, 1_664_525u32, 1_013_904_223u32 + f as u32, direct_rounds];
                    slot.list.SetComputeRootSignature(&xform_root);
                    slot.list.SetPipelineState(&xform_pso);
                    slot.list
                        .SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                    slot.list
                        .SetComputeRootShaderResourceView(1, direct_src[f].GetGPUVirtualAddress());
                    slot.list
                        .SetComputeRootUnorderedAccessView(2, direct_dst[f].GetGPUVirtualAddress());
                    slot.list.Dispatch(direct_words.div_ceil(64), 1, 1);
                    // 3) CARD_COMPARE hash workload on Direct
                    let cmp_params = [0u32, 0, cmp_bytes as u32, cmp_pieces, 0, 0];
                    slot.list.SetComputeRootSignature(&cmp_root);
                    slot.list.SetPipelineState(&cmp_pso);
                    slot.list
                        .SetComputeRoot32BitConstants(0, 6, cmp_params.as_ptr().cast(), 0);
                    slot.list
                        .SetComputeRootShaderResourceView(1, cmp_bufs[f].GetGPUVirtualAddress());
                    slot.list
                        .SetComputeRootUnorderedAccessView(2, cmp_hashes[f].GetGPUVirtualAddress());
                    slot.list
                        .SetComputeRootUnorderedAccessView(3, cmp_differs[f].GetGPUVirtualAddress());
                    for _ in 0..cmp_reps {
                        slot.list.Dispatch(cmp_pieces, 1, 1);
                    }
                    if f == flights - 1 {
                        slot.list.ResourceBarrier(&[crate::host::transition(
                            &direct_dst[f],
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        slot.list
                            .CopyBufferRegion(&direct_rb_check[f], 0, &direct_dst[f], 0, 64);
                    }
                    slot.list.EndQuery(
                        &direct_qh,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        (f * 2 + 1) as u32,
                    );
                    slot.list.ResolveQueryData(
                        &direct_qh,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        (f * 2) as u32,
                        2,
                        &direct_ts_rb,
                        (f * 2 * 8) as u64,
                    );
                    slot.list.Close().unwrap();
                    direct_queue.ExecuteCommandLists(&[Some(slot.list.cast().unwrap())]);
                }
                direct_fence_val += 1;
                direct_queue.Signal(&direct_fence, direct_fence_val).unwrap();
            }
            wait_fence_cpu(&direct_fence, direct_fence_val).unwrap();
            let s_wall = t0.elapsed().as_secs_f64() * 1000.0;
            let s_ts = read_timestamps(&direct_ts_rb, flights * 2);
            let s_gpu = (s_ts[flights * 2 - 1].saturating_sub(s_ts[0]) as f64 / direct_freq) * 1000.0;
            // SAFETY: `direct_fence_val` completed; reading 1 dword checksum.
            let serial_checksum = unsafe {
                let mut ptr = std::ptr::null_mut();
                direct_rb_check[flights - 1].Map(0, None, Some(&mut ptr)).unwrap();
                let v = ptr.cast::<u32>().read();
                direct_rb_check[flights - 1].Unmap(0, None);
                v
            };

            // -----------------------------------------------------------------
            // Mode B: Multi-Queue Overlap (Copy DMA + Async Compute + Direct)
            // -----------------------------------------------------------------
            let cal_d = calibrate_us(&direct_queue, direct_freq);
            let cal_c = calibrate_us(&compute_queue, compute_freq);
            let cal_cp = copy_freq.map(|f| calibrate_us(&copy_queue, f));

            let t1 = Instant::now();
            let mut last_copy_tok = TimelineToken { queue: QueueKind::Copy, value: 0 };
            let mut last_comp_tok = TimelineToken { queue: QueueKind::Compute, value: 0 };

            for f in 0..flights {
                // 1) Copy Queue: DMA upload UPLOAD -> bounce_bufs[f] concurrently with Direct & Compute!
                let dst_b = bounce_bufs[f].clone();
                let src_u = up_bufs[f].clone();
                let cp_qh_opt = copy_qh.clone();
                let cp_rb_ref = copy_ts_rb.clone();
                last_copy_tok = timeline
                    .submit_copy(
                        &[],
                        &[],
                        &[],
                        vec![dst_b.clone(), src_u.clone(), cp_rb_ref.clone()],
                        move |list| {
                            // SAFETY: Valid copy + optional copy-queue timestamp query.
                            unsafe {
                                if let Some(qh) = &cp_qh_opt {
                                    list.EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, (f * 2) as u32);
                                }
                                list.CopyBufferRegion(&dst_b, 0, &src_u, 0, upload_bytes);
                                if let Some(qh) = &cp_qh_opt {
                                    list.EndQuery(
                                        qh,
                                        D3D12_QUERY_TYPE_TIMESTAMP,
                                        (f * 2 + 1) as u32,
                                    );
                                    list.ResolveQueryData(
                                        qh,
                                        D3D12_QUERY_TYPE_TIMESTAMP,
                                        (f * 2) as u32,
                                        2,
                                        &cp_rb_ref,
                                        (f * 2 * 8) as u64,
                                    );
                                }
                            }
                            Ok(())
                        },
                    )
                    .unwrap();

                // 2) Async Compute Queue: CARD_COMPARE hashing concurrently with Direct & Copy!
                let r = cmp_root.clone();
                let p = cmp_pso.clone();
                let c_qh = compute_qh.clone();
                let c_rb = compute_ts_rb.clone();
                let cb_va = unsafe { cmp_bufs[f].GetGPUVirtualAddress() };
                let ch_va = unsafe { cmp_hashes[f].GetGPUVirtualAddress() };
                let cd_va = unsafe { cmp_differs[f].GetGPUVirtualAddress() };
                last_comp_tok = timeline
                    .submit_compute(
                        &[],
                        &[],
                        &[],
                        vec![
                            cmp_bufs[f].clone(),
                            cmp_hashes[f].clone(),
                            cmp_differs[f].clone(),
                            c_rb.clone(),
                        ],
                        move |list| {
                            let cmp_params = [0u32, 0, cmp_bytes as u32, cmp_pieces, 0, 0];
                            // SAFETY: Valid compute timestamp + `CARD_COMPARE` dispatch.
                            unsafe {
                                list.EndQuery(&c_qh, D3D12_QUERY_TYPE_TIMESTAMP, (f * 2) as u32);
                                list.SetComputeRootSignature(&r);
                                list.SetPipelineState(&p);
                                list.SetComputeRoot32BitConstants(
                                    0,
                                    6,
                                    cmp_params.as_ptr().cast(),
                                    0,
                                );
                                list.SetComputeRootShaderResourceView(1, cb_va);
                                list.SetComputeRootUnorderedAccessView(2, ch_va);
                                list.SetComputeRootUnorderedAccessView(3, cd_va);
                                for _ in 0..cmp_reps {
                                    list.Dispatch(cmp_pieces, 1, 1);
                                }
                                list.EndQuery(
                                    &c_qh,
                                    D3D12_QUERY_TYPE_TIMESTAMP,
                                    (f * 2 + 1) as u32,
                                );
                                list.ResolveQueryData(
                                    &c_qh,
                                    D3D12_QUERY_TYPE_TIMESTAMP,
                                    (f * 2) as u32,
                                    2,
                                    &c_rb,
                                    (f * 2 * 8) as u64,
                                );
                            }
                            Ok(())
                        },
                    )
                    .unwrap();

                // 3) Direct Queue: Independent Direct compute workload concurrently with Copy & Compute!
                // SAFETY: Dedicated Direct slot `f` is idle; recording Direct workload + timestamps.
                unsafe {
                    let slot = &direct_slots[f];
                    slot.allocator.Reset().unwrap();
                    slot.list.Reset(&slot.allocator, None).unwrap();
                    slot.list
                        .EndQuery(&direct_qh, D3D12_QUERY_TYPE_TIMESTAMP, (f * 2) as u32);
                    let params = [direct_words, 1_664_525u32, 1_013_904_223u32 + f as u32, direct_rounds];
                    slot.list.SetComputeRootSignature(&xform_root);
                    slot.list.SetPipelineState(&xform_pso);
                    slot.list
                        .SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                    slot.list
                        .SetComputeRootShaderResourceView(1, direct_src[f].GetGPUVirtualAddress());
                    slot.list
                        .SetComputeRootUnorderedAccessView(2, direct_dst[f].GetGPUVirtualAddress());
                    slot.list.Dispatch(direct_words.div_ceil(64), 1, 1);
                    if f == flights - 1 {
                        slot.list.ResourceBarrier(&[crate::host::transition(
                            &direct_dst[f],
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        slot.list
                            .CopyBufferRegion(&direct_rb_check[f], 0, &direct_dst[f], 0, 64);
                    }
                    slot.list.EndQuery(
                        &direct_qh,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        (f * 2 + 1) as u32,
                    );
                    slot.list.ResolveQueryData(
                        &direct_qh,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        (f * 2) as u32,
                        2,
                        &direct_ts_rb,
                        (f * 2 * 8) as u64,
                    );
                    slot.list.Close().unwrap();
                    direct_queue.ExecuteCommandLists(&[Some(slot.list.cast().unwrap())]);
                }
            }

            // GPU-side join: Direct waits for final Copy & Compute tokens before signaling
            // one unified completion fence (zero intermediate CPU waits!).
            timeline.wait_on_gpu(&direct_queue, last_copy_tok).unwrap();
            timeline.wait_on_gpu(&direct_queue, last_comp_tok).unwrap();
            direct_fence_val += 1;
            // SAFETY: Valid final fence signal on Direct queue.
            unsafe {
                direct_queue.Signal(&direct_fence, direct_fence_val).unwrap();
            }
            wait_fence_cpu(&direct_fence, direct_fence_val).unwrap();
            let o_wall = t1.elapsed().as_secs_f64() * 1000.0;

            let d_ts = read_timestamps(&direct_ts_rb, flights * 2);
            let c_ts = read_timestamps(&compute_ts_rb, flights * 2);
            let cp_ts = copy_qh
                .as_ref()
                .map(|_| read_timestamps(&copy_ts_rb, flights * 2));

            let d_busy_ms: f64 = (0..flights)
                .map(|f| d_ts[f * 2 + 1].saturating_sub(d_ts[f * 2]) as f64 / direct_freq * 1000.0)
                .sum();
            let c_busy_ms: f64 = (0..flights)
                .map(|f| c_ts[f * 2 + 1].saturating_sub(c_ts[f * 2]) as f64 / compute_freq * 1000.0)
                .sum();
            let cp_busy_ms: f64 = match (&cp_ts, copy_freq) {
                (Some(ts), Some(freq)) => (0..flights)
                    .map(|f| ts[f * 2 + 1].saturating_sub(ts[f * 2]) as f64 / freq * 1000.0)
                    .sum(),
                _ => 0.0,
            };

            let d_start = to_qpc_us(d_ts[0], cal_d);
            let d_end = to_qpc_us(d_ts[flights * 2 - 1], cal_d);
            let c_start = to_qpc_us(c_ts[0], cal_c);
            let c_end = to_qpc_us(c_ts[flights * 2 - 1], cal_c);
            let mut min_start = d_start.min(c_start);
            let mut max_end = d_end.max(c_end);
            let mut intersect_us = (d_end.min(c_end) - d_start.max(c_start)).max(0.0);
            if let (Some(ts), Some(cal)) = (&cp_ts, cal_cp) {
                let cp_start = to_qpc_us(ts[0], cal);
                let cp_end = to_qpc_us(ts[flights * 2 - 1], cal);
                min_start = min_start.min(cp_start);
                max_end = max_end.max(cp_end);
                let cp_d_overlap = (d_end.min(cp_end) - d_start.max(cp_start)).max(0.0);
                intersect_us = intersect_us.max(cp_d_overlap);
            }
            let union_ms = (max_end - min_start).max(0.0) / 1000.0;

            // SAFETY: Final fence completed; verify checksum parity.
            let overlap_checksum = unsafe {
                let mut ptr = std::ptr::null_mut();
                direct_rb_check[flights - 1].Map(0, None, Some(&mut ptr)).unwrap();
                let v = ptr.cast::<u32>().read();
                direct_rb_check[flights - 1].Unmap(0, None);
                v
            };
            assert_eq!(
                overlap_checksum, serial_checksum,
                "overlapped execution must produce bit-identical checksum to serialized Direct"
            );
            final_checksum = overlap_checksum;

            if iter > 0 {
                serial_walls_ms.push(s_wall);
                serial_gpu_ms.push(s_gpu);
                overlap_walls_ms.push(o_wall);
                overlap_gpu_union_ms.push(union_ms);
                overlap_direct_busy_ms.push(d_busy_ms);
                overlap_compute_busy_ms.push(c_busy_ms);
                overlap_copy_busy_ms.push(cp_busy_ms);
                overlap_intersect_ms.push(intersect_us / 1000.0);
            }
        }

        let avg = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len().max(1) as f64;
        let s_wall_avg = avg(&serial_walls_ms);
        let s_gpu_avg = avg(&serial_gpu_ms);
        let o_wall_avg = avg(&overlap_walls_ms);
        let o_union_avg = avg(&overlap_gpu_union_ms);
        let o_d_busy = avg(&overlap_direct_busy_ms);
        let o_c_busy = avg(&overlap_compute_busy_ms);
        let o_cp_busy = avg(&overlap_copy_busy_ms);
        let o_inter = avg(&overlap_intersect_ms);
        let wall_speedup = s_wall_avg / o_wall_avg.max(1e-6);
        let gpu_speedup = s_gpu_avg / o_union_avg.max(1e-6);

        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "benchmark_multi_queue_overlap_vs_serialized_direct");

        eprintln!(
            "[async_timeline] BENCHMARK (adapter={adapter_name}, sw={is_sw}, flights={flights}, upload={} MiB/flight, copy_ts={}):\n  \
             Serialized Direct:  wall={s_wall_avg:.2} ms | GPU_span={s_gpu_avg:.2} ms\n  \
             Multi-Queue Overlap: wall={o_wall_avg:.2} ms | GPU_union={o_union_avg:.2} ms | \
             Direct_busy={o_d_busy:.2} ms, Compute_busy={o_c_busy:.2} ms, Copy_busy={o_cp_busy:.2} ms, \
             Calibrated_overlap={o_inter:.2} ms\n  \
             Speedup: wall={wall_speedup:.2}x ({:+.1}%) | GPU={gpu_speedup:.2}x ({:+.1}%) | checksum={final_checksum:#010x}",
            upload_mb,
            copy_qh.is_some(),
            (wall_speedup - 1.0) * 100.0,
            (gpu_speedup - 1.0) * 100.0,
        );

        if !is_sw {
            assert!(
                o_union_avg < s_gpu_avg,
                "expected hardware multi-queue GPU union span ({o_union_avg:.2} ms) to beat serialized Direct GPU span ({s_gpu_avg:.2} ms)"
            );
            assert!(
                o_inter > 0.0,
                "expected positive calibrated GPU timestamp overlap across queues, got {o_inter:.2} ms"
            );
            // Without D3D12 Debug Layer CPU validation overhead on ExecuteCommandLists,
            // wall-clock time must also strictly beat serialized Direct execution.
            let debug_layer_active = device.cast::<ID3D12InfoQueue>().is_ok();
            if !debug_layer_active {
                assert!(
                    o_wall_avg < s_wall_avg,
                    "expected hardware multi-queue wall-clock ({o_wall_avg:.2} ms) to beat serialized Direct ({s_wall_avg:.2} ms)"
                );
            }
        }
    }

    /// End-to-end integration test exercising `AsyncTimeline`'s VRAM bounce upload,
    /// deferred `prepare_compute` -> `Host::queue` Direct execution -> `dispatch_prepared_compute`
    /// -> `submit_copy_readback` -> `return_bounce` lifecycle directly on `Host`.
    #[test]
    fn host_async_bounce_and_prepared_compute_integration() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = AsyncTimeline::new(device).unwrap();
        let (root, pso) = make_compute_pso(device, TRANSFORM_CS);

        const WORDS: u32 = 8192;
        const BYTES: u64 = (WORDS as u64) * 4;
        let (staging, staging_ptr) = host.upload_buffer(BYTES).unwrap();
        // SAFETY: Valid upload buffer mapping of `BYTES`.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(staging_ptr.cast::<u32>(), WORDS as usize);
            for (i, w) in slice.iter_mut().enumerate() {
                *w = (i as u32).wrapping_mul(37).wrapping_add(11);
            }
        }
        let card_buf = host.uav_buffer(BYTES).unwrap();
        let out_buf = host.uav_buffer(BYTES).unwrap();
        let readback = host.read_back_buffer(BYTES).unwrap();

        // 1) Copy queue uploads `staging` -> pooled VRAM bounce, inserting GPU Wait on `Host::queue`.
        let mut placed_offset = None;
        let bounce = timeline
            .submit_upload_bounce(
                host,
                host.queue(),
                &[(&staging, 0, BYTES)],
                |_, _, offset| {
                    placed_offset = Some(offset);
                },
            )
            .unwrap();
        let bounce_off = placed_offset.unwrap();

        // 2) Prepare Async Compute transform pass BEFORE Direct executes (mirroring `card_before`).
        let card_va = unsafe { card_buf.GetGPUVirtualAddress() };
        let out_va = unsafe { out_buf.GetGPUVirtualAddress() };
        let prepared = timeline
            .prepare_compute(|clist| {
                let params = [WORDS, 1_664_525u32, 1_013_904_223u32, 4u32];
                // SAFETY: Valid compute pass reading `card_buf` and writing `out_buf`.
                unsafe {
                    clist.SetComputeRootSignature(&root);
                    clist.SetPipelineState(&pso);
                    clist.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
                    clist.SetComputeRootShaderResourceView(1, card_va);
                    clist.SetComputeRootUnorderedAccessView(2, out_va);
                    clist.Dispatch(WORDS.div_ceil(64), 1, 1);
                }
                Ok(())
            })
            .unwrap();

        // 3) Direct (`Host::queue` via `HostList`) copies bounce -> card_buf (VRAM -> VRAM).
        let direct_list = host.new_list().unwrap();
        // SAFETY: Open HostList; `bounce` is ready after GPU wait on `Host::queue`.
        unsafe {
            direct_list
                .list
                .CopyBufferRegion(&card_buf, 0, bounce.resource(), bounce_off, BYTES);
        }
        direct_list.close();
        let _ = host.execute(&[&direct_list]);

        // 4) Dispatch prepared Async Compute behind Direct via atomic tracked handoff, then readback on Copy.
        let ((), direct_tok, _) = timeline
            .execute_direct_tracked(host.queue(), &[], &[resource_key(&card_buf)], || {})
            .unwrap();
        let comp_tok = timeline
            .dispatch_prepared_compute(
                prepared,
                &[direct_tok],
                &[resource_key(&card_buf)],
                &[resource_key(&out_buf)],
                vec![card_buf.clone(), out_buf.clone()],
            )
            .unwrap();
        let rb_tok = timeline
            .submit_copy_readback(&[comp_tok], &readback, &[(&out_buf, 0, 0, BYTES)])
            .unwrap();

        timeline.return_bounce(bounce);
        timeline.wait_cpu(rb_tok).unwrap();

        // SAFETY: `rb_tok` completed on GPU; verify all `WORDS` dwords bit-for-bit.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut ptr)).unwrap();
            let actual = std::slice::from_raw_parts(ptr.cast::<u32>(), WORDS as usize);
            for (i, &got) in actual.iter().enumerate() {
                let init = (i as u32).wrapping_mul(37).wrapping_add(11);
                let expected = cpu_transform(init, 1_664_525u32, 1_013_904_223u32, 4);
                assert_eq!(got, expected, "dword {i} mismatch");
            }
            readback.Unmap(0, None);
        }

        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "host_async_bounce_and_prepared_compute_integration");
        eprintln!(
            "[async_timeline] host_async_bounce_and_prepared_compute_integration PASS \
             (adapter={adapter_name}, sw={is_sw}, 0 debug issues)"
        );
    }

    /// Proves `wait_fence_cpu` immediately rejects `u64::MAX` without ever blocking
    /// on `SetEventOnCompletion(u64::MAX, ...)`.
    #[test]
    fn wait_fence_cpu_rejects_u64_max() {
        let Some(host) = crate::host::host() else {
            return;
        };
        // SAFETY: Plain fence creation on live test device.
        let fence: ID3D12Fence = unsafe {
            host.device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .unwrap()
        };
        let err = wait_fence_cpu(&fence, u64::MAX);
        assert!(err.is_err(), "u64::MAX fence wait must return Err immediately");
    }

    /// Multi-threaded stress test proving zero TOCTOU hazard between a Direct
    /// writer thread (`execute_direct_tracked`) and a concurrent Copy readback
    /// thread (`submit_copy_readback`) operating on the SAME shared VRAM buffer.
    /// Every readback snapshot must observe one uniform round value (zero torn
    /// cross-queue reads) and zero D3D12 Debug Layer errors/warnings.
    #[test]
    fn concurrent_direct_write_and_copy_readback_stress() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        clear_debug_queue(device);
        let (adapter_name, is_sw) = adapter_info(device);

        let timeline = std::sync::Arc::new(AsyncTimeline::new(device).unwrap());
        const ROUNDS: usize = 24;
        const WORDS: u32 = 16_384; // 64 KiB
        const BYTES: u64 = (WORDS as u64) * 4;

        let shared = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            BYTES,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        let shared_key = resource_key(&shared);

        // Pre-fill per-round upload buffers, each filled uniformly with a distinct tag.
        let mut uploads = Vec::with_capacity(ROUNDS);
        for r in 0..ROUNDS {
            let up = make_committed_buffer(
                device,
                D3D12_HEAP_TYPE_UPLOAD,
                BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let tag = 0x1000_0000u32 + (r as u32) + 1;
            // SAFETY: Valid full mapping of fresh upload buffer.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                up.Map(0, None, Some(&mut ptr)).unwrap();
                let slice = std::slice::from_raw_parts_mut(ptr.cast::<u32>(), WORDS as usize);
                slice.fill(tag);
                up.Unmap(0, None);
            }
            uploads.push(up);
        }

        // Seed `shared` with initial uniform tag `0x1000_0000`.
        let seed_up = make_committed_buffer(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            BYTES,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: Valid full mapping of seed upload buffer.
        unsafe {
            let mut ptr = std::ptr::null_mut();
            seed_up.Map(0, None, Some(&mut ptr)).unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr.cast::<u32>(), WORDS as usize);
            slice.fill(0x1000_0000u32);
            seed_up.Unmap(0, None);
        }
        let seed_dst = shared.clone();
        let seed_src = seed_up.clone();
        let seed_tok = timeline
            .submit_copy(
                &[],
                &[resource_key(&seed_up)],
                &[shared_key],
                vec![seed_up.clone(), shared.clone()],
                move |list| {
                    // SAFETY: Valid seed copy into `COMMON` VRAM buffer.
                    unsafe { list.CopyBufferRegion(&seed_dst, 0, &seed_src, 0, BYTES) };
                    Ok(())
                },
            )
            .unwrap();
        timeline.wait_cpu(seed_tok).unwrap();

        // Direct queue + per-round Direct lists for the writer thread.
        // SAFETY: Creating dedicated Direct queue and per-round allocators/lists.
        let (direct_queue, direct_slots): (ID3D12CommandQueue, Vec<CommandSlot>) = unsafe {
            let q: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let mut slots = Vec::with_capacity(ROUNDS);
            for _ in 0..ROUNDS {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap();
                list.Close().unwrap();
                slots.push(CommandSlot { allocator, list });
            }
            (q, slots)
        };

        // Pre-allocate readback buffers for the concurrent Copy reader thread.
        let readbacks: Vec<ID3D12Resource> = (0..ROUNDS)
            .map(|_| {
                make_committed_buffer(
                    device,
                    D3D12_HEAP_TYPE_READBACK,
                    BYTES,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )
            })
            .collect();

        // Wrapper to pass free-threaded COM handles across `std::thread::scope`.
        struct SendCom<T>(T);
        // SAFETY: D3D12 resources and queues are free-threaded; each queue is used
        // exclusively by one thread in this test.
        unsafe impl<T> Send for SendCom<T> {}
        // SAFETY: As above.
        unsafe impl<T> Sync for SendCom<T> {}

        let shared_send = SendCom(shared.clone());
        let direct_q_send = SendCom(direct_queue);
        let uploads_send = SendCom(uploads);
        let readbacks_send = SendCom(readbacks);
        let start_barrier = std::sync::Barrier::new(2);

        let tl_reader = timeline.clone();
        let rb_tokens = std::thread::scope(|s| {
            let reader = s.spawn(|| {
                let mut tokens = Vec::with_capacity(ROUNDS);
                start_barrier.wait();
                for r in 0..ROUNDS {
                    let tok = tl_reader
                        .submit_copy_readback(
                            &[],
                            &readbacks_send.0[r],
                            &[(&shared_send.0, 0, 0, BYTES)],
                        )
                        .unwrap();
                    tokens.push(tok);
                }
                tokens
            });

            start_barrier.wait();
            let mut last_direct_tok = TimelineToken {
                queue: QueueKind::Direct,
                value: 0,
            };
            for r in 0..ROUNDS {
                // SAFETY: Dedicated slot `r` is idle; copies uniform tag into `shared`.
                let cmd: ID3D12CommandList = unsafe {
                    let slot = &direct_slots[r];
                    slot.allocator.Reset().unwrap();
                    slot.list.Reset(&slot.allocator, None).unwrap();
                    slot.list
                        .CopyBufferRegion(&shared_send.0, 0, &uploads_send.0[r], 0, BYTES);
                    slot.list.Close().unwrap();
                    slot.list.cast().unwrap()
                };
                let ((), dtok, _) = timeline
                    .execute_direct_tracked(&direct_q_send.0, &[], &[shared_key], || {
                        // SAFETY: Closed Direct list with live buffers.
                        unsafe { direct_q_send.0.ExecuteCommandLists(&[Some(cmd)]) };
                    })
                    .unwrap();
                last_direct_tok = dtok;
            }
            let tokens = reader.join().unwrap();
            timeline.wait_cpu(last_direct_tok).unwrap();
            tokens
        });

        timeline.wait_cpu(*rb_tokens.last().unwrap()).unwrap();
        let mut last_seen = 0u32;
        for (r, rb) in readbacks_send.0.iter().enumerate() {
            // SAFETY: All Copy readbacks have completed on GPU.
            unsafe {
                let mut ptr = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let words = std::slice::from_raw_parts(ptr.cast::<u32>(), WORDS as usize);
                let first = words[0];
                assert!(
                    (0x1000_0000u32..=0x1000_0000u32 + ROUNDS as u32).contains(&first),
                    "round {r}: unexpected tag {first:#x}"
                );
                assert!(
                    words.iter().all(|&w| w == first),
                    "round {r}: torn cross-queue read detected! first={first:#x}"
                );
                assert!(
                    first >= last_seen,
                    "round {r}: non-monotonic snapshot {first:#x} < {last_seen:#x}"
                );
                last_seen = first;
                rb.Unmap(0, None);
            }
        }

        timeline.drain_all().unwrap();
        assert_zero_d3d12_debug_issues(device, "concurrent_direct_write_and_copy_readback_stress");
        eprintln!(
            "[async_timeline] concurrent_direct_write_and_copy_readback_stress PASS \
             (adapter={adapter_name}, sw={is_sw}, rounds={ROUNDS}, 0 torn reads, 0 debug issues)"
        );
    }
}
