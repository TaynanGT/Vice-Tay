//! Event-driven Windows Raw Input mouse capture, lock-free timestamped ring,
//! and cursor clip/visibility lifecycle for GameInput.

use std::cell::UnsafeCell;
use std::io;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, HWND, LPARAM, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::ClientToScreen;
use windows_sys::Win32::System::Threading::GetCurrentThreadId;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON, VK_XBUTTON1, VK_XBUTTON2,
};
use windows_sys::Win32::UI::Input::{
    DefRawInputProc, GetRawInputBuffer, GetRawInputData, GetRegisteredRawInputDevices,
    MOUSE_MOVE_ABSOLUTE, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RAWMOUSE, RID_INPUT,
    RIDEV_PAGEONLY, RIDEV_REMOVE, RIM_TYPEMOUSE, RegisterRawInputDevices,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    ClipCursor, GetClientRect, GetForegroundWindow, PostMessageW, RI_MOUSE_BUTTON_4_DOWN,
    RI_MOUSE_BUTTON_4_UP, RI_MOUSE_BUTTON_5_DOWN, RI_MOUSE_BUTTON_5_UP, RI_MOUSE_HWHEEL,
    RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP, RI_MOUSE_MIDDLE_BUTTON_DOWN,
    RI_MOUSE_MIDDLE_BUTTON_UP, RI_MOUSE_RIGHT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_UP,
    RI_MOUSE_WHEEL, RIM_INPUT, ShowCursor, WM_NULL,
};

use super::{MouseBatch, MouseSample, MouseState, lock, now, switches};

// HID Generic Desktop page, Mouse usage.
const USAGE_PAGE: u16 = 1;
const USAGE: u16 = 2;
const DEVICE_SIZE: u32 = size_of::<RAWINPUTDEVICE>() as u32;
const HEADER_SIZE: u32 = size_of::<RAWINPUTHEADER>() as u32;

const _: () = assert!(std::mem::offset_of!(RAWINPUT, data) == size_of::<RAWINPUTHEADER>());

/// Power-of-two capacity for the lock-free microsecond-timestamped mouse ring.
/// Sized for 8 kHz bursts (~128 ms of uncoalesced events without overflow).
pub(super) const RING_CAP: usize = 1024;
const RING_MASK: usize = RING_CAP - 1;
const _: () = assert!(RING_CAP.is_power_of_two());

/// Stack buffer size for batched `GetRawInputBuffer` drains (8 KiB, 8-byte aligned).
const RAW_BATCH_U64S: usize = 1024;
/// Upper bound on `GetRawInputBuffer` drain loops per `WM_INPUT` message.
const MAX_DRAIN_ITERS: usize = 64;
/// Safety cap on `ShowCursor` stepping to avoid runaway loops on host quirks.
const MAX_SHOW_CURSOR_STEPS: u32 = 16;

/// GameInput mouse button bits paired with their Raw Input edge flags.
const BUTTON_EDGES: [(u32, u32, u32); 5] = [
    (RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP, 1 << 0),
    (RI_MOUSE_RIGHT_BUTTON_DOWN, RI_MOUSE_RIGHT_BUTTON_UP, 1 << 1),
    (RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_MIDDLE_BUTTON_UP, 1 << 2),
    (RI_MOUSE_BUTTON_4_DOWN, RI_MOUSE_BUTTON_4_UP, 1 << 3),
    (RI_MOUSE_BUTTON_5_DOWN, RI_MOUSE_BUTTON_5_UP, 1 << 4),
];

struct RingSlot {
    seq: AtomicUsize,
    sample: UnsafeCell<MouseSample>,
}

// SAFETY: access to `sample` is strictly sequenced by `seq` Acquire/Release transitions.
unsafe impl Sync for RingSlot {}

/// Lock-free bounded MPMC ring of microsecond-timestamped mouse samples.
pub(super) struct MouseRing {
    slots: Box<[RingSlot]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    connected: AtomicBool,
    has_overflow: AtomicBool,
    /// Fallback slot used only if the ring temporarily fills during a stall,
    /// guaranteeing cumulative axes and held button state never drift.
    overflow: Mutex<Option<MouseSample>>,
    /// Optional wake callback notifying the GameInput hub on state changes.
    wake: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl MouseRing {
    pub(super) fn new() -> Self {
        let slots = (0..RING_CAP)
            .map(|i| RingSlot {
                seq: AtomicUsize::new(i),
                sample: UnsafeCell::new(MouseSample::default()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            connected: AtomicBool::new(false),
            has_overflow: AtomicBool::new(false),
            overflow: Mutex::new(None),
            wake: Mutex::new(None),
        }
    }

    pub(super) fn set_wake(&self, wake: Arc<dyn Fn() + Send + Sync>) {
        *lock(&self.wake) = Some(wake);
    }

    fn notify(&self) {
        let wake = lock(&self.wake).clone();
        if let Some(wake) = wake {
            wake();
        }
    }

    pub(super) fn set_connected(&self, connected: bool) {
        let was = self.connected.swap(connected, Ordering::AcqRel);
        if was != connected {
            self.notify();
        }
    }

    pub(super) fn push(&self, sample: MouseSample) -> bool {
        if self.try_push(sample) {
            if self.has_overflow.swap(false, Ordering::AcqRel) {
                *lock(&self.overflow) = None;
            }
            return true;
        }
        *lock(&self.overflow) = Some(sample);
        self.has_overflow.store(true, Ordering::Release);
        false
    }

    fn try_push(&self, sample: MouseSample) -> bool {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[pos & RING_MASK];
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(pos) as isize;
            if diff == 0 {
                match self.tail.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: we exclusively claimed `slot` for cycle `pos` (seq == pos).
                        unsafe { *slot.sample.get() = sample };
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release);
                        return true;
                    }
                    Err(next) => pos = next,
                }
            } else if diff < 0 {
                return false;
            } else {
                pos = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    fn try_pop(&self) -> Option<MouseSample> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = &self.slots[pos & RING_MASK];
            let seq = slot.seq.load(Ordering::Acquire);
            let expected = pos.wrapping_add(1);
            let diff = seq.wrapping_sub(expected) as isize;
            if diff == 0 {
                match self.head.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: producer released `slot` with `seq == pos + 1`; we own it.
                        let sample = unsafe { *slot.sample.get() };
                        slot.seq.store(pos.wrapping_add(RING_CAP), Ordering::Release);
                        return Some(sample);
                    }
                    Err(next) => pos = next,
                }
            } else if diff < 0 {
                return None;
            } else {
                pos = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Drains all pending timestamped samples and reports connection status.
    pub(super) fn drain(&self) -> MouseBatch {
        let mut samples = Vec::new();
        while let Some(sample) = self.try_pop() {
            samples.push(sample);
        }
        if self.has_overflow.swap(false, Ordering::AcqRel)
            && let Some(latest) = lock(&self.overflow).take()
            && samples
                .last()
                .is_none_or(|last| latest.timestamp >= last.timestamp)
        {
            samples.push(latest);
        }
        if self.connected.load(Ordering::Acquire) {
            MouseBatch::Connected(samples)
        } else {
            MouseBatch::Disconnected
        }
    }
}

/// Producer-side accumulator translating `RAWMOUSE` batches into timestamped
/// ring samples without dropping button edges or quantizing relative deltas.
#[derive(Default)]
struct MouseIngestor {
    state: MouseState,
    last_timestamp: u64,
}

impl MouseIngestor {
    fn monotonic_timestamp(&mut self, ts: u64) -> u64 {
        let next = ts.max(self.last_timestamp);
        self.last_timestamp = next;
        next
    }

    fn next_edge_timestamp(&mut self, ts: u64) -> u64 {
        let next = if ts > self.last_timestamp {
            ts
        } else {
            self.last_timestamp.saturating_add(1)
        };
        self.last_timestamp = next;
        next
    }

    /// Processes a batch of `RAWMOUSE` packets. Motion-only packets within the
    /// same batch coalesce losslessly; every button down/up edge produces a
    /// distinct timestamped reading (including simultaneous down+up in one packet).
    fn ingest_batch(&mut self, packets: &[RAWMOUSE], timestamp: u64, ring: &MouseRing) -> bool {
        let mut dirty_motion = false;
        let mut saw_relative_motion = false;
        let mut ts = self.monotonic_timestamp(timestamp);

        for &mouse in packets {
            if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
                if mouse.lLastX != 0 || mouse.lLastY != 0 {
                    saw_relative_motion = true;
                    dirty_motion = true;
                }
                self.state.position_x = self.state.position_x.saturating_add(i64::from(mouse.lLastX));
                self.state.position_y = self.state.position_y.saturating_add(i64::from(mouse.lLastY));
            }
            // SAFETY: documented layout of the mouse button union.
            let buttons = unsafe { mouse.Anonymous.Anonymous };
            let flags = u32::from(buttons.usButtonFlags);
            let delta = i64::from(buttons.usButtonData as i16);
            if flags & RI_MOUSE_HWHEEL != 0 && delta != 0 {
                self.state.wheel_x = self.state.wheel_x.saturating_add(delta);
                dirty_motion = true;
            }
            if flags & RI_MOUSE_WHEEL != 0 && delta != 0 {
                self.state.wheel_y = self.state.wheel_y.saturating_add(delta);
                dirty_motion = true;
            }

            let mut down_mask = 0u32;
            let mut up_mask = 0u32;
            for (down, up, bit) in BUTTON_EDGES {
                if flags & down != 0 {
                    down_mask |= bit;
                }
                if flags & up != 0 {
                    up_mask |= bit;
                }
            }

            if down_mask == 0 && up_mask == 0 {
                continue;
            }

            let both = down_mask & up_mask;
            if both != 0 {
                // Fast click contained inside a single RAWMOUSE packet: emit DOWN then UP.
                self.state.buttons |= down_mask;
                ring.push(MouseSample {
                    timestamp: ts,
                    state: self.state,
                });
                ts = self.next_edge_timestamp(ts);
                self.state.buttons &= !up_mask;
                ring.push(MouseSample {
                    timestamp: ts,
                    state: self.state,
                });
                dirty_motion = false;
            } else {
                let next_buttons = (self.state.buttons | down_mask) & !up_mask;
                if next_buttons != self.state.buttons || dirty_motion {
                    self.state.buttons = next_buttons;
                    ring.push(MouseSample {
                        timestamp: ts,
                        state: self.state,
                    });
                    ts = self.next_edge_timestamp(ts);
                    dirty_motion = false;
                }
            }
        }

        if dirty_motion {
            ring.push(MouseSample {
                timestamp: ts,
                state: self.state,
            });
        }
        saw_relative_motion
    }

    /// Releases any held buttons (e.g., on focus loss, overlay, or pause) so no
    /// button can remain stuck after leaving capture.
    fn release_buttons(&mut self, timestamp: u64, ring: &MouseRing) {
        if self.state.buttons != 0 {
            self.state.buttons = 0;
            let ts = self.next_edge_timestamp(timestamp);
            ring.push(MouseSample {
                timestamp: ts,
                state: self.state,
            });
        }
    }

    /// Publishes the current baseline state when capture becomes active.
    fn emit_baseline(&mut self, timestamp: u64, ring: &MouseRing) {
        let ts = self.monotonic_timestamp(timestamp);
        ring.push(MouseSample {
            timestamp: ts,
            state: self.state,
        });
    }
}

/// Host abstraction for cursor clipping and visibility so transition logic is
/// deterministically testable without altering the desktop pointer in tests.
trait CursorHost {
    fn clip(&mut self, rect: Option<RECT>) -> bool;
    fn hide_cursor(&mut self) -> u32;
    fn restore_cursor(&mut self, steps: u32);
    fn client_screen_rect(&self, hwnd: usize) -> Option<RECT>;
}

struct Win32CursorHost;

impl CursorHost for Win32CursorHost {
    fn clip(&mut self, rect: Option<RECT>) -> bool {
        let ptr = rect.as_ref().map_or(std::ptr::null(), |r| r as *const RECT);
        // SAFETY: either a valid screen-space RECT or NULL to release clipping.
        unsafe { ClipCursor(ptr) != 0 }
    }

    fn hide_cursor(&mut self) -> u32 {
        let mut steps = 0;
        while steps < MAX_SHOW_CURSOR_STEPS {
            steps += 1;
            // SAFETY: decrements the calling thread's cursor display counter.
            if unsafe { ShowCursor(0) } < 0 {
                break;
            }
        }
        steps
    }

    fn restore_cursor(&mut self, steps: u32) {
        for _ in 0..steps {
            // SAFETY: restores the exact number of decrements applied by `hide_cursor`.
            unsafe { ShowCursor(1) };
        }
    }

    fn client_screen_rect(&self, hwnd: usize) -> Option<RECT> {
        if hwnd == 0 {
            return None;
        }
        let hwnd = hwnd as HWND;
        let mut client = RECT::default();
        // SAFETY: plain query of the title window's client rectangle.
        if unsafe { GetClientRect(hwnd, &mut client) } == 0 {
            return None;
        }
        if client.right <= client.left || client.bottom <= client.top {
            return None;
        }
        let mut tl = POINT {
            x: client.left,
            y: client.top,
        };
        let mut br = POINT {
            x: client.right,
            y: client.bottom,
        };
        // SAFETY: converts client-area corners to screen coordinates.
        if unsafe { ClientToScreen(hwnd, &mut tl) == 0 || ClientToScreen(hwnd, &mut br) == 0 } {
            return None;
        }
        Some(RECT {
            left: tl.x,
            top: tl.y,
            right: br.x,
            bottom: br.y,
        })
    }
}

/// Deterministic focus/clip/overlay/pause state machine for cursor capture
/// (`VICTORY_CURSOR_CAPTURE=1`) and raw mouse ring activity (`VICTORY_RAW_MOUSE=1`).
/// The two flags are strictly independent so each subsystem can be toggled alone.
struct CursorCapture {
    raw_mouse_enabled: bool,
    cursor_capture_enabled: bool,
    window: usize,
    owner_thread: u32,
    focused: bool,
    minimized: bool,
    in_size_move: bool,
    overlay_depth: u32,
    paused: bool,
    clipped: bool,
    hide_steps: u32,
    clipped_rect: Option<RECT>,
    pending_visibility_sync: bool,
}

impl CursorCapture {
    fn new(raw_mouse_enabled: bool, cursor_capture_enabled: bool) -> Self {
        Self {
            raw_mouse_enabled,
            cursor_capture_enabled,
            window: 0,
            owner_thread: 0,
            focused: false,
            minimized: false,
            in_size_move: false,
            overlay_depth: 0,
            paused: false,
            clipped: false,
            hide_steps: 0,
            clipped_rect: None,
            pending_visibility_sync: false,
        }
    }

    fn input_active(&self) -> bool {
        self.raw_mouse_enabled
            && self.window != 0
            && self.focused
            && !self.minimized
            && self.overlay_depth == 0
    }

    fn wants_capture(&self) -> bool {
        self.cursor_capture_enabled
            && self.window != 0
            && self.focused
            && !self.minimized
            && !self.in_size_move
            && self.overlay_depth == 0
            && !self.paused
    }

    #[cfg(test)]
    fn is_clipped(&self) -> bool {
        self.clipped
    }

    #[cfg(test)]
    fn is_cursor_hidden(&self) -> bool {
        self.hide_steps > 0
    }

    fn attach(&mut self, hwnd: usize, owner_thread: u32, focused: bool, host: &mut dyn CursorHost) {
        self.window = hwnd;
        self.owner_thread = owner_thread;
        self.focused = focused;
        self.minimized = false;
        self.in_size_move = false;
        self.reconcile(owner_thread, false, host);
    }

    fn detach(&mut self, hwnd: usize, caller_thread: u32, host: &mut dyn CursorHost) -> bool {
        if self.window == 0 || self.window != hwnd {
            return false;
        }
        self.window = 0;
        self.focused = false;
        self.minimized = false;
        self.in_size_move = false;
        self.reconcile(caller_thread, false, host);
        self.owner_thread = 0;
        true
    }

    fn set_focused(&mut self, focused: bool, caller_thread: u32, host: &mut dyn CursorHost) {
        self.focused = focused;
        self.reconcile(caller_thread, false, host);
    }

    fn set_size_move(&mut self, entering: bool, caller_thread: u32, host: &mut dyn CursorHost) {
        self.in_size_move = entering;
        self.reconcile(caller_thread, !entering, host);
    }

    fn set_geometry(&mut self, minimized: Option<bool>, caller_thread: u32, host: &mut dyn CursorHost) {
        if let Some(minimized) = minimized {
            self.minimized = minimized;
        }
        self.reconcile(caller_thread, !self.minimized, host);
    }

    fn enter_overlay(&mut self, caller_thread: u32, host: &mut dyn CursorHost) {
        self.overlay_depth = self.overlay_depth.saturating_add(1);
        self.reconcile(caller_thread, false, host);
    }

    fn leave_overlay(&mut self, caller_thread: u32, host: &mut dyn CursorHost) {
        self.overlay_depth = self.overlay_depth.saturating_sub(1);
        self.reconcile(caller_thread, false, host);
    }

    fn set_paused(&mut self, paused: bool, caller_thread: u32, host: &mut dyn CursorHost) {
        self.paused = paused;
        self.reconcile(caller_thread, false, host);
    }

    fn sync_owner_thread(&mut self, caller_thread: u32, host: &mut dyn CursorHost) {
        if self.pending_visibility_sync && caller_thread == self.owner_thread {
            self.reconcile(caller_thread, false, host);
        }
    }

    /// Reconciles `ClipCursor` (desktop-wide) and `ShowCursor` (owner-thread-only)
    /// against desired capture state. Never leaks clip or drifts `ShowCursor` counts.
    fn reconcile(&mut self, caller_thread: u32, force_reclip: bool, host: &mut dyn CursorHost) {
        let on_owner = self.owner_thread != 0 && caller_thread == self.owner_thread;
        if self.wants_capture() {
            if let Some(rect) = host.client_screen_rect(self.window) {
                let rect_changed = self.clipped_rect.is_none_or(|prev| {
                    prev.left != rect.left
                        || prev.top != rect.top
                        || prev.right != rect.right
                        || prev.bottom != rect.bottom
                });
                if (!self.clipped || force_reclip || rect_changed) && host.clip(Some(rect)) {
                    self.clipped = true;
                    self.clipped_rect = Some(rect);
                }
                if self.hide_steps == 0 {
                    if on_owner {
                        self.hide_steps = host.hide_cursor();
                        self.pending_visibility_sync = false;
                    } else {
                        self.pending_visibility_sync = true;
                    }
                } else {
                    self.pending_visibility_sync = false;
                }
            } else {
                // Empty client rect (minimized or zero-sized): do not trap the cursor.
                self.release_inner(on_owner, host);
            }
        } else {
            self.release_inner(on_owner, host);
        }
    }

    fn release_inner(&mut self, on_owner: bool, host: &mut dyn CursorHost) {
        if self.clipped {
            let _ = host.clip(None);
            self.clipped = false;
            self.clipped_rect = None;
        }
        if self.hide_steps > 0 {
            if on_owner {
                host.restore_cursor(self.hide_steps);
                self.hide_steps = 0;
                self.pending_visibility_sync = false;
            } else {
                self.pending_visibility_sync = true;
            }
        } else {
            self.pending_visibility_sync = false;
        }
    }
}

struct Capture {
    ingestor: MouseIngestor,
    cursor: CursorCapture,
    batch_drain_safe: bool,
    raw_input_registered: bool,
    /// Baseline `VICTORY_GAMEPAD_MOUSE=1` state when `VICTORY_RAW_MOUSE=0`.
    legacy_window: usize,
    legacy_state: MouseState,
}

static LEGACY_MOUSE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VICTORY_GAMEPAD_MOUSE").is_some_and(|v| v == "1"));
static RING: LazyLock<Arc<MouseRing>> = LazyLock::new(|| Arc::new(MouseRing::new()));
static CAPTURE: LazyLock<Mutex<Capture>> = LazyLock::new(|| {
    let sw = switches();
    Mutex::new(Capture {
        ingestor: MouseIngestor::default(),
        cursor: CursorCapture::new(sw.raw_mouse, sw.cursor_capture),
        batch_drain_safe: false,
        raw_input_registered: false,
        legacy_window: 0,
        legacy_state: MouseState::default(),
    })
});

fn lifecycle_hooks_enabled() -> bool {
    let sw = switches();
    sw.raw_mouse || sw.cursor_capture
}

fn legacy_mouse_only() -> bool {
    *LEGACY_MOUSE_ENABLED && !switches().raw_mouse
}

fn accumulate_legacy(state: &mut MouseState, mouse: RAWMOUSE) {
    if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
        state.position_x = state.position_x.saturating_add(i64::from(mouse.lLastX));
        state.position_y = state.position_y.saturating_add(i64::from(mouse.lLastY));
    }
    // SAFETY: documented layout of the mouse button union.
    let buttons = unsafe { mouse.Anonymous.Anonymous };
    let flags = u32::from(buttons.usButtonFlags);
    let delta = i64::from(buttons.usButtonData as i16);
    if flags & RI_MOUSE_HWHEEL != 0 {
        state.wheel_x = state.wheel_x.saturating_add(delta);
    }
    if flags & RI_MOUSE_WHEEL != 0 {
        state.wheel_y = state.wheel_y.saturating_add(delta);
    }
}

/// Baseline `VICTORY_GAMEPAD_MOUSE=1` sampler used only when `VICTORY_RAW_MOUSE=0`.
pub(super) fn legacy_sample() -> Option<MouseState> {
    if !legacy_mouse_only() {
        return None;
    }
    // SAFETY: plain foreground window query.
    let window = unsafe { GetForegroundWindow() } as usize;
    let mut state = {
        let capture = lock(&CAPTURE);
        if window == 0 || window != capture.legacy_window {
            return None;
        }
        capture.legacy_state
    };
    state.buttons = [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2]
        .into_iter()
        .enumerate()
        // SAFETY: each value is a mouse button key code.
        .filter(|&(_, button)| unsafe { GetAsyncKeyState(i32::from(button)) } < 0)
        .fold(0, |bits, (index, _)| bits | 1 << index);
    // SAFETY: plain query. Discard if focus changed during the button reads.
    (unsafe { GetForegroundWindow() } as usize == window).then_some(state)
}

pub(super) fn global_ring() -> Arc<MouseRing> {
    Arc::clone(&RING)
}

/// Raw Input is process-wide. List current registrations before modifying ours.
fn registered_devices() -> io::Result<Vec<RAWINPUTDEVICE>> {
    let mut slots = 0;
    // SAFETY: the null pointer asks for the required count in `slots`.
    let result = unsafe { GetRegisteredRawInputDevices(null_mut(), &mut slots, DEVICE_SIZE) };
    if result == u32::MAX {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            return Err(error);
        }
    }
    if slots == 0 {
        return Ok(Vec::new());
    }
    let mut devices = vec![RAWINPUTDEVICE::default(); slots as usize];
    // SAFETY: the buffer holds `slots` device registrations.
    let count =
        unsafe { GetRegisteredRawInputDevices(devices.as_mut_ptr(), &mut slots, DEVICE_SIZE) };
    if count == u32::MAX {
        return Err(io::Error::last_os_error());
    }
    devices.truncate(count as usize);
    Ok(devices)
}

fn find_mouse_registration(devices: &[RAWINPUTDEVICE]) -> Option<RAWINPUTDEVICE> {
    devices.iter().copied().find(|d| {
        d.usUsagePage == USAGE_PAGE && (d.usUsage == USAGE || d.dwFlags & RIDEV_PAGEONLY != 0)
    })
}

fn has_other_registrations(devices: &[RAWINPUTDEVICE]) -> bool {
    devices
        .iter()
        .any(|d| !(d.usUsagePage == USAGE_PAGE && d.usUsage == USAGE && d.dwFlags == 0))
}

fn register(window: HWND, flags: u32) -> io::Result<()> {
    let device = RAWINPUTDEVICE {
        usUsagePage: USAGE_PAGE,
        usUsage: USAGE,
        dwFlags: flags,
        hwndTarget: window,
    };
    // SAFETY: one initialized device registration; removal uses a null window.
    if unsafe { RegisterRawInputDevices(&device, 1, DEVICE_SIZE) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Parses `count` aligned `RAWINPUT` structures produced by `GetRawInputBuffer`,
/// extracting all `RIM_TYPEMOUSE` payloads safely within buffer bounds.
/// Returns `true` if any non-mouse `RAWINPUT` packet was encountered.
fn parse_raw_input_buffer(bytes: &[u8], count: u32, out: &mut Vec<RAWMOUSE>) -> bool {
    let align = align_of::<usize>();
    let header_size = size_of::<RAWINPUTHEADER>();
    let min_mouse_size = header_size + size_of::<RAWMOUSE>();
    let mut offset = 0usize;
    let mut saw_non_mouse = false;

    for _ in 0..count {
        if offset.saturating_add(header_size) > bytes.len() {
            break;
        }
        // SAFETY: bounds checked above; `read_unaligned` handles arbitrary byte slices cleanly.
        let header = unsafe {
            bytes
                .as_ptr()
                .add(offset)
                .cast::<RAWINPUTHEADER>()
                .read_unaligned()
        };
        let size = header.dwSize as usize;
        if size < header_size || offset.checked_add(size).is_none_or(|end| end > bytes.len()) {
            break;
        }
        if header.dwType == RIM_TYPEMOUSE {
            if size >= min_mouse_size {
                // SAFETY: `offset + header_size + size_of::<RAWMOUSE>() <= offset + size <= bytes.len()`.
                let mouse = unsafe {
                    bytes
                        .as_ptr()
                        .add(offset + header_size)
                        .cast::<RAWMOUSE>()
                        .read_unaligned()
                };
                out.push(mouse);
            }
        } else {
            saw_non_mouse = true;
        }
        let Some(next) = offset.checked_add(size) else {
            break;
        };
        let Some(aligned) = next.checked_add(align - 1).map(|v| v & !(align - 1)) else {
            break;
        };
        if aligned <= offset {
            break;
        }
        offset = aligned;
    }
    saw_non_mouse
}

/// Drains queued `WM_INPUT` packets from the calling thread's message queue in
/// batches via `GetRawInputBuffer` (8 kHz friendly, zero heap allocation on the
/// fast path). Returns `true` if any non-mouse packet was encountered so the
/// caller can disable batch draining and preserve title `WndProc` delivery.
fn drain_raw_input_buffer(out: &mut Vec<RAWMOUSE>) -> bool {
    let mut stack_buf = [0u64; RAW_BATCH_U64S];
    let mut heap_buf: Vec<u64> = Vec::new();
    let mut saw_non_mouse = false;

    for _ in 0..MAX_DRAIN_ITERS {
        let words = if heap_buf.is_empty() {
            stack_buf.as_mut_slice()
        } else {
            heap_buf.as_mut_slice()
        };
        let capacity_bytes = std::mem::size_of_val(words);
        let mut cb_size = capacity_bytes as u32;
        // SAFETY: `words` is 8-byte aligned and holds `cb_size` bytes for `GetRawInputBuffer`.
        let count = unsafe {
            GetRawInputBuffer(words.as_mut_ptr().cast::<RAWINPUT>(), &mut cb_size, HEADER_SIZE)
        };
        if count == 0 {
            break;
        }
        if count == u32::MAX {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(ERROR_INSUFFICIENT_BUFFER as i32)
                && (cb_size as usize) > capacity_bytes
                && (cb_size as usize) <= 256 * 1024
            {
                let needed_u64s = (cb_size as usize).div_ceil(size_of::<u64>());
                heap_buf.resize(needed_u64s, 0);
                continue;
            }
            break;
        }
        // SAFETY: `words` is initialized for `capacity_bytes` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), capacity_bytes) };
        if parse_raw_input_buffer(bytes, count, out) {
            saw_non_mouse = true;
            let mut ptrs: Vec<*const RAWINPUT> = Vec::new();
            let align = align_of::<usize>();
            let header_size = size_of::<RAWINPUTHEADER>();
            let mut offset = 0usize;
            for _ in 0..count {
                if offset + header_size > bytes.len() {
                    break;
                }
                // SAFETY: bounds checked above.
                let header = unsafe {
                    bytes
                        .as_ptr()
                        .add(offset)
                        .cast::<RAWINPUTHEADER>()
                        .read_unaligned()
                };
                let size = header.dwSize as usize;
                if size < header_size || offset + size > bytes.len() {
                    break;
                }
                if header.dwType != RIM_TYPEMOUSE {
                    // SAFETY: `offset` is 8-byte aligned within `words`.
                    ptrs.push(unsafe { bytes.as_ptr().add(offset).cast::<RAWINPUT>() });
                }
                let Some(next) = offset
                    .checked_add(size)
                    .and_then(|v| v.checked_add(align - 1))
                    .map(|v| v & !(align - 1))
                else {
                    break;
                };
                if next <= offset {
                    break;
                }
                offset = next;
            }
            if !ptrs.is_empty() {
                // SAFETY: pointers reference valid `RAWINPUT` headers inside `words`.
                unsafe { DefRawInputProc(ptrs.as_ptr(), ptrs.len() as i32, HEADER_SIZE) };
            }
            // Stop draining immediately so further non-mouse messages reach the title WndProc.
            break;
        }
    }
    saw_non_mouse
}

fn wake_owner_window(hwnd: usize) {
    if hwnd != 0 {
        // SAFETY: posting WM_NULL safely wakes the owner thread's message loop to sync ShowCursor.
        unsafe { PostMessageW(hwnd as HWND, WM_NULL, 0, 0) };
    }
}

pub(in crate::modules) fn attach(window: HWND) {
    let sw = switches();
    let legacy_only = legacy_mouse_only();
    let want_raw_input = sw.raw_mouse || legacy_only;
    if !want_raw_input && !sw.cursor_capture {
        return;
    }

    let mut registered_raw_input = false;
    let mut batch_drain_safe = false;
    if want_raw_input {
        match registered_devices() {
            Ok(existing) => {
                if find_mouse_registration(&existing).is_some() {
                    tracing::warn!(
                        "the process already receives mouse Raw Input; keeping its registration"
                    );
                } else if let Err(error) = register(window, 0) {
                    tracing::warn!(%error, "could not register mouse Raw Input for the title window");
                } else {
                    registered_raw_input = true;
                    batch_drain_safe = sw.raw_mouse && !has_other_registrations(&existing);
                }
            }
            Err(error) => {
                tracing::warn!(%error, "could not check existing Raw Input registrations");
            }
        }
        if !registered_raw_input && !sw.cursor_capture {
            return;
        }
    }

    // SAFETY: plain queries on the title window thread.
    let (focused, owner_thread) = unsafe { (GetForegroundWindow() == window, GetCurrentThreadId()) };
    let mut capture = lock(&CAPTURE);
    capture.raw_input_registered = registered_raw_input;
    capture.batch_drain_safe = batch_drain_safe;
    if legacy_only && registered_raw_input {
        capture.legacy_window = window as usize;
    }
    if sw.raw_mouse || sw.cursor_capture {
        capture
            .cursor
            .attach(window as usize, owner_thread, focused, &mut Win32CursorHost);
        if sw.raw_mouse && registered_raw_input {
            let active = capture.cursor.input_active();
            if active {
                capture.ingestor.emit_baseline(now(), &RING);
            }
            RING.set_connected(active);
        }
    }
}

pub(in crate::modules) fn detach(window: HWND) {
    let sw = switches();
    let legacy_only = legacy_mouse_only();
    if !sw.raw_mouse && !sw.cursor_capture && !legacy_only {
        return;
    }
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let (remove_raw_input, disconnect_ring) = {
        let mut capture = lock(&CAPTURE);
        let legacy_match =
            capture.legacy_window != 0 && capture.legacy_window == window as usize;
        if legacy_match {
            capture.legacy_window = 0;
        }
        let cursor_match = capture.cursor.detach(window as usize, caller, &mut Win32CursorHost);
        if !legacy_match && !cursor_match {
            return;
        }
        let was_registered = capture.raw_input_registered;
        capture.raw_input_registered = false;
        capture.batch_drain_safe = false;
        if sw.raw_mouse {
            capture.ingestor.release_buttons(now(), &RING);
        }
        (was_registered, sw.raw_mouse)
    };
    if disconnect_ring {
        RING.set_connected(false);
    }
    if remove_raw_input
        && registered_devices().is_ok_and(|devices| {
            find_mouse_registration(&devices)
                .is_some_and(|d| d.hwndTarget == window && d.usUsage == USAGE && d.dwFlags == 0)
        })
        && let Err(error) = register(null_mut(), RIDEV_REMOVE)
    {
        tracing::warn!(%error, "could not remove mouse Raw Input registration");
    }
}

/// Focus change (`WM_SETFOCUS`, `WM_KILLFOCUS`, `WM_ACTIVATE`, `WM_ACTIVATEAPP`).
pub(in crate::modules) fn on_focus(window: HWND, focused: bool) {
    if !lifecycle_hooks_enabled() {
        return;
    }
    let raw_mouse = switches().raw_mouse;
    // Re-check process-wide Raw Input registrations on focus gain so post-attach
    // keyboard/HID registrations disable batch draining before the next WM_INPUT.
    let refreshed_batch_safe = if focused && raw_mouse {
        registered_devices()
            .ok()
            .map(|devices| !has_other_registrations(&devices))
    } else {
        None
    };
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let mut capture = lock(&CAPTURE);
    if capture.cursor.window != window as usize {
        return;
    }
    if let Some(safe) = refreshed_batch_safe {
        capture.batch_drain_safe = safe;
    }
    let was_active = capture.cursor.input_active();
    capture
        .cursor
        .set_focused(focused, caller, &mut Win32CursorHost);
    if raw_mouse && capture.raw_input_registered {
        let now_active = capture.cursor.input_active();
        let ts = now();
        if !now_active {
            capture.ingestor.release_buttons(ts, &RING);
        } else if !was_active {
            capture.ingestor.emit_baseline(ts, &RING);
        }
        RING.set_connected(now_active);
    }
}

/// Window move/resize loop transitions (`WM_ENTERSIZEMOVE`, `WM_EXITSIZEMOVE`).
pub(in crate::modules) fn on_size_move(window: HWND, entering: bool) {
    if !lifecycle_hooks_enabled() {
        return;
    }
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let mut capture = lock(&CAPTURE);
    if capture.cursor.window != window as usize {
        return;
    }
    capture
        .cursor
        .set_size_move(entering, caller, &mut Win32CursorHost);
    if entering && switches().raw_mouse {
        capture.ingestor.release_buttons(now(), &RING);
    }
}

/// Window size, move, minimize, or fullscreen toggle updates (`WM_SIZE`, `WM_MOVE`).
/// Pass `Some(minimized)` on `WM_SIZE` and `None` on `WM_MOVE` so a minimized
/// window's synthetic `WM_MOVE(-32000, -32000)` never clears `minimized`.
pub(in crate::modules) fn on_geometry_change(window: HWND, minimized: Option<bool>) {
    if !lifecycle_hooks_enabled() {
        return;
    }
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let mut capture = lock(&CAPTURE);
    if capture.cursor.window != window as usize {
        return;
    }
    let was_active = capture.cursor.input_active();
    capture
        .cursor
        .set_geometry(minimized, caller, &mut Win32CursorHost);
    if switches().raw_mouse && capture.raw_input_registered {
        let now_active = capture.cursor.input_active();
        let ts = now();
        if !now_active {
            capture.ingestor.release_buttons(ts, &RING);
        } else if !was_active {
            capture.ingestor.emit_baseline(ts, &RING);
        }
        RING.set_connected(now_active);
    }
}

/// Reconciles deferred `ShowCursor` visibility on the title window's owning thread.
pub(in crate::modules) fn sync_cursor(window: HWND) {
    if !switches().cursor_capture {
        return;
    }
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let mut capture = lock(&CAPTURE);
    if capture.cursor.window != window as usize {
        return;
    }
    capture.cursor.sync_owner_thread(caller, &mut Win32CursorHost);
}

/// Pauses or resumes cursor capture (e.g., explicit pause or shutdown).
pub(in crate::modules) fn set_paused(paused: bool) {
    if !lifecycle_hooks_enabled() {
        return;
    }
    // SAFETY: plain query of calling thread id.
    let caller = unsafe { GetCurrentThreadId() };
    let wake_hwnd = {
        let mut capture = lock(&CAPTURE);
        capture.cursor.set_paused(paused, caller, &mut Win32CursorHost);
        if paused && switches().raw_mouse {
            capture.ingestor.release_buttons(now(), &RING);
        }
        if capture.cursor.pending_visibility_sync {
            capture.cursor.window
        } else {
            0
        }
    };
    wake_owner_window(wake_hwnd);
}

/// RAII guard that releases cursor clipping/hiding while a host overlay or
/// modal system dialog is open, restoring capture cleanly on drop.
pub(in crate::modules) struct OverlayGuard {
    active: bool,
}

impl OverlayGuard {
    pub(in crate::modules) fn enter() -> Self {
        if !lifecycle_hooks_enabled() {
            return Self { active: false };
        }
        // SAFETY: plain query of calling thread id.
        let caller = unsafe { GetCurrentThreadId() };
        let wake_hwnd = {
            let mut capture = lock(&CAPTURE);
            capture.cursor.enter_overlay(caller, &mut Win32CursorHost);
            if switches().raw_mouse && capture.raw_input_registered {
                capture.ingestor.release_buttons(now(), &RING);
                RING.set_connected(capture.cursor.input_active());
            }
            if capture.cursor.pending_visibility_sync {
                capture.cursor.window
            } else {
                0
            }
        };
        wake_owner_window(wake_hwnd);
        Self { active: true }
    }
}

impl Drop for OverlayGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // SAFETY: plain query of calling thread id.
        let caller = unsafe { GetCurrentThreadId() };
        let wake_hwnd = {
            let mut capture = lock(&CAPTURE);
            let was_active = capture.cursor.input_active();
            capture.cursor.leave_overlay(caller, &mut Win32CursorHost);
            if switches().raw_mouse && capture.raw_input_registered {
                let now_active = capture.cursor.input_active();
                if now_active && !was_active {
                    capture.ingestor.emit_baseline(now(), &RING);
                }
                RING.set_connected(now_active);
            }
            if capture.cursor.pending_visibility_sync {
                capture.cursor.window
            } else {
                0
            }
        };
        wake_owner_window(wake_hwnd);
    }
}

/// `WM_INPUT` handler. Preserves baseline single-packet `GetRawInputData`
/// ingestion for `VICTORY_GAMEPAD_MOUSE=1` when `VICTORY_RAW_MOUSE=0`; when
/// `VICTORY_RAW_MOUSE=1`, reads `lparam` and drains queued high-rate bursts via
/// batched `GetRawInputBuffer` into the lock-free `MouseRing`.
pub(in crate::modules) fn receive(window: HWND, wparam: WPARAM, lparam: LPARAM) {
    let raw_mouse = switches().raw_mouse;
    let legacy_only = legacy_mouse_only();
    if (!raw_mouse && !legacy_only) || wparam & 0xff != RIM_INPUT as usize {
        return;
    }

    if legacy_only {
        let mut capture = lock(&CAPTURE);
        if capture.legacy_window != window as usize {
            return;
        }
        let mut packet = RAWINPUT::default();
        let mut buffer_size = size_of::<RAWINPUT>() as u32;
        // SAFETY: the window system provides the packet handle; the buffer is aligned and sized.
        let copied = unsafe {
            GetRawInputData(
                lparam as _,
                RID_INPUT,
                (&raw mut packet).cast(),
                &mut buffer_size,
                HEADER_SIZE,
            )
        };
        if copied == u32::MAX
            || copied < HEADER_SIZE + size_of::<RAWMOUSE>() as u32
            || packet.header.dwType != RIM_TYPEMOUSE
        {
            return;
        }
        // SAFETY: plain query of the foreground window.
        if unsafe { GetForegroundWindow() } == window {
            // SAFETY: header identifies the mouse union member.
            let mouse = unsafe { packet.data.mouse };
            accumulate_legacy(&mut capture.legacy_state, mouse);
            #[cfg(feature = "diagnostics")]
            if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 && (mouse.lLastX != 0 || mouse.lLastY != 0)
            {
                static REPORTED: std::sync::Once = std::sync::Once::new();
                REPORTED.call_once(|| {
                    tracing::info!("received relative mouse movement in the title window")
                });
            }
        }
        return;
    }

    let mut packets: Vec<RAWMOUSE> = Vec::with_capacity(16);
    let mut packet = RAWINPUT::default();
    let mut buffer_size = size_of::<RAWINPUT>() as u32;
    // SAFETY: the window system provides the packet handle; the buffer is aligned and sized.
    let copied = unsafe {
        GetRawInputData(
            lparam as _,
            RID_INPUT,
            (&raw mut packet).cast(),
            &mut buffer_size,
            HEADER_SIZE,
        )
    };
    let mut lparam_non_mouse = false;
    if copied != u32::MAX && copied >= HEADER_SIZE {
        if packet.header.dwType == RIM_TYPEMOUSE {
            if copied >= HEADER_SIZE + size_of::<RAWMOUSE>() as u32 {
                // SAFETY: header identifies the mouse union member.
                packets.push(unsafe { packet.data.mouse });
            }
        } else {
            lparam_non_mouse = true;
        }
    }

    // SAFETY: plain queries on the title window thread.
    let (foreground, caller) = unsafe { (GetForegroundWindow() == window, GetCurrentThreadId()) };
    let batch_drain_safe = {
        let mut capture = lock(&CAPTURE);
        if !capture.raw_input_registered || capture.cursor.window != window as usize {
            return;
        }
        if lparam_non_mouse {
            capture.batch_drain_safe = false;
        }
        capture.cursor.sync_owner_thread(caller, &mut Win32CursorHost);
        if capture.cursor.focused != foreground {
            capture
                .cursor
                .set_focused(foreground, caller, &mut Win32CursorHost);
        }
        capture.batch_drain_safe
    };

    let mut saw_non_mouse = false;
    if batch_drain_safe {
        saw_non_mouse = drain_raw_input_buffer(&mut packets);
    }

    let ts = now();
    let mut capture = lock(&CAPTURE);
    if saw_non_mouse {
        capture.batch_drain_safe = false;
    }
    if packets.is_empty()
        || !capture.raw_input_registered
        || capture.cursor.window != window as usize
    {
        return;
    }
    let active = capture.cursor.input_active();
    RING.set_connected(active);
    if !active {
        capture.ingestor.release_buttons(ts, &RING);
        return;
    }

    let saw_relative = capture.ingestor.ingest_batch(&packets, ts, &RING);
    #[cfg(feature = "diagnostics")]
    if saw_relative {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| {
            tracing::info!("received relative mouse movement in the title window")
        });
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = saw_relative;
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::UI::Input::{RAWMOUSE_0, RAWMOUSE_0_0};

    use super::*;

    fn raw_mouse(flags: u16, dx: i32, dy: i32, btn_flags: u32, btn_data: i16) -> RAWMOUSE {
        RAWMOUSE {
            usFlags: flags,
            lLastX: dx,
            lLastY: dy,
            Anonymous: RAWMOUSE_0 {
                Anonymous: RAWMOUSE_0_0 {
                    usButtonFlags: btn_flags as u16,
                    usButtonData: btn_data as u16,
                },
            },
            ..Default::default()
        }
    }

    #[derive(Default)]
    struct FakeCursorHost {
        rect: Option<RECT>,
        clipped_to: Option<RECT>,
        clip_calls: Vec<Option<RECT>>,
        cursor_counter: i32,
        hide_calls: u32,
        restore_calls: u32,
    }

    impl FakeCursorHost {
        fn with_rect(left: i32, top: i32, right: i32, bottom: i32) -> Self {
            Self {
                rect: Some(RECT {
                    left,
                    top,
                    right,
                    bottom,
                }),
                ..Default::default()
            }
        }
    }

    impl CursorHost for FakeCursorHost {
        fn clip(&mut self, rect: Option<RECT>) -> bool {
            self.clipped_to = rect;
            self.clip_calls.push(rect);
            true
        }

        fn hide_cursor(&mut self) -> u32 {
            let mut steps = 0;
            while steps < MAX_SHOW_CURSOR_STEPS {
                steps += 1;
                self.hide_calls += 1;
                self.cursor_counter -= 1;
                if self.cursor_counter < 0 {
                    break;
                }
            }
            steps
        }

        fn restore_cursor(&mut self, steps: u32) {
            for _ in 0..steps {
                self.restore_calls += 1;
                self.cursor_counter += 1;
            }
        }

        fn client_screen_rect(&self, hwnd: usize) -> Option<RECT> {
            (hwnd != 0).then_some(self.rect?)
        }
    }

    #[test]
    fn mouse_accumulates_relative_motion_and_signed_wheels() {
        let ring = MouseRing::new();
        ring.set_connected(true);
        let mut ingestor = MouseIngestor::default();
        let batch = [
            raw_mouse(0, 20, -6, RI_MOUSE_WHEEL, -120),
            raw_mouse(0, -8, 3, RI_MOUSE_HWHEEL, 120),
            raw_mouse(MOUSE_MOVE_ABSOLUTE, 65535, 65535, RI_MOUSE_WHEEL, 60),
        ];
        assert!(ingestor.ingest_batch(&batch, 1000, &ring));
        let MouseBatch::Connected(samples) = ring.drain() else {
            panic!("expected connected mouse batch");
        };
        assert_eq!(samples.len(), 1);
        let state = samples[0].state;
        assert_eq!((state.position_x, state.position_y), (12, -3));
        assert_eq!((state.wheel_x, state.wheel_y), (120, -60));
        assert_eq!(samples[0].timestamp, 1000);
    }

    #[test]
    fn high_rate_8khz_batching_and_sub_millisecond_click_edges_are_preserved() {
        let ring = MouseRing::new();
        ring.set_connected(true);
        let mut ingestor = MouseIngestor::default();

        // Simulate 64 microsecond-spaced 8 kHz motion deliveries (125 us apart)
        // interleaved with a 250 us fast tap and a single-packet down+up tap.
        for i in 0..32_u64 {
            let p = [raw_mouse(0, 3, -2, 0, 0)];
            ingestor.ingest_batch(&p, 10_000 + i * 125, &ring);
        }
        // Fast left-button tap: DOWN at 14_000 us, motion at 14_125 us, UP at 14_250 us.
        ingestor.ingest_batch(
            &[raw_mouse(0, 1, 0, RI_MOUSE_LEFT_BUTTON_DOWN, 0)],
            14_000,
            &ring,
        );
        ingestor.ingest_batch(&[raw_mouse(0, 2, 1, 0, 0)], 14_125, &ring);
        ingestor.ingest_batch(
            &[raw_mouse(0, 1, 0, RI_MOUSE_LEFT_BUTTON_UP, 0)],
            14_250,
            &ring,
        );
        // Simultaneous RIGHT down+up inside ONE RAWMOUSE packet + XBUTTON1 down.
        ingestor.ingest_batch(
            &[
                raw_mouse(
                    0,
                    5,
                    4,
                    RI_MOUSE_RIGHT_BUTTON_DOWN | RI_MOUSE_RIGHT_BUTTON_UP,
                    0,
                ),
                raw_mouse(0, 0, 0, RI_MOUSE_BUTTON_4_DOWN, 0),
                raw_mouse(0, 0, 0, RI_MOUSE_BUTTON_4_UP, 0),
            ],
            14_375,
            &ring,
        );

        let MouseBatch::Connected(samples) = ring.drain() else {
            panic!("expected connected");
        };
        // 32 motion + LDOWN + motion + LUP + RDOWN + RUP + B4DOWN + B4UP = 39 samples.
        assert_eq!(samples.len(), 39);
        assert!(samples.windows(2).all(|w| w[1].timestamp >= w[0].timestamp));
        assert_eq!(samples[32].state.buttons, 1 << 0);
        assert_eq!(samples[33].state.buttons, 1 << 0);
        assert_eq!(samples[34].state.buttons, 0);
        assert_eq!(samples[35].state.buttons, 1 << 1);
        assert_eq!(samples[36].state.buttons, 0);
        assert_eq!(samples[37].state.buttons, 1 << 3);
        assert_eq!(samples[38].state.buttons, 0);

        let last = samples.last().unwrap().state;
        // Exact unquantized delta sum: 32*3 + 1 + 2 + 1 + 5 = 105, 32*(-2) + 1 + 4 = -59.
        assert_eq!((last.position_x, last.position_y), (105, -59));
    }

    #[test]
    fn ring_overflow_never_loses_cumulative_position_or_final_button_state() {
        let ring = MouseRing::new();
        ring.set_connected(true);
        let mut ingestor = MouseIngestor::default();

        // Push more events than RING_CAP without draining.
        for i in 0..(RING_CAP + 200) {
            let btn = if i == RING_CAP + 199 {
                RI_MOUSE_MIDDLE_BUTTON_DOWN
            } else {
                0
            };
            ingestor.ingest_batch(&[raw_mouse(0, 1, -1, btn, 0)], 100 + i as u64, &ring);
        }
        let MouseBatch::Connected(samples) = ring.drain() else {
            panic!("expected connected");
        };
        assert_eq!(samples.len(), RING_CAP + 1);
        let final_state = samples.last().unwrap().state;
        let expected = (RING_CAP + 200) as i64;
        assert_eq!((final_state.position_x, final_state.position_y), (expected, -expected));
        assert_eq!(final_state.buttons, 1 << 2);
    }

    #[test]
    fn parse_raw_input_buffer_handles_alignment_mixed_types_and_truncation() {
        let align = align_of::<usize>();
        let header_size = size_of::<RAWINPUTHEADER>();
        let mouse_block_size = header_size + size_of::<RAWMOUSE>();
        let stride = (mouse_block_size + align - 1) & !(align - 1);

        let mut buf = vec![0u8; stride * 3 + header_size];
        // Block 0: mouse (dx=10, dy=-4, left down)
        let h0 = RAWINPUTHEADER {
            dwType: RIM_TYPEMOUSE,
            dwSize: mouse_block_size as u32,
            ..Default::default()
        };
        let m0 = raw_mouse(0, 10, -4, RI_MOUSE_LEFT_BUTTON_DOWN, 0);
        // Block 1: non-mouse (dwType = 1 = keyboard), should be skipped cleanly.
        let h1 = RAWINPUTHEADER {
            dwType: 1,
            dwSize: mouse_block_size as u32,
            ..Default::default()
        };
        // Block 2: mouse (dx=-3, dy=7, left up)
        let h2 = RAWINPUTHEADER {
            dwType: RIM_TYPEMOUSE,
            dwSize: mouse_block_size as u32,
            ..Default::default()
        };
        let m2 = raw_mouse(0, -3, 7, RI_MOUSE_LEFT_BUTTON_UP, 0);

        // SAFETY: `buf` is sized for three full aligned blocks plus trailing malformed header.
        unsafe {
            buf.as_mut_ptr().cast::<RAWINPUTHEADER>().write_unaligned(h0);
            buf.as_mut_ptr()
                .add(header_size)
                .cast::<RAWMOUSE>()
                .write_unaligned(m0);

            buf.as_mut_ptr()
                .add(stride)
                .cast::<RAWINPUTHEADER>()
                .write_unaligned(h1);

            buf.as_mut_ptr()
                .add(stride * 2)
                .cast::<RAWINPUTHEADER>()
                .write_unaligned(h2);
            buf.as_mut_ptr()
                .add(stride * 2 + header_size)
                .cast::<RAWMOUSE>()
                .write_unaligned(m2);

            // Block 3: invalid dwSize = 0 must terminate cleanly without looping.
            buf.as_mut_ptr()
                .add(stride * 3)
                .cast::<RAWINPUTHEADER>()
                .write_unaligned(RAWINPUTHEADER {
                    dwType: RIM_TYPEMOUSE,
                    dwSize: 0,
                    ..Default::default()
                });
        }

        let mut out = Vec::new();
        assert!(
            parse_raw_input_buffer(&buf, 4, &mut out),
            "mixed buffer with keyboard header must report non-mouse packet"
        );
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].lLastX, out[0].lLastY), (10, -4));
        assert_eq!((out[1].lLastX, out[1].lLastY), (-3, 7));

        // Pure mouse-only buffer returns `saw_non_mouse == false`.
        let mut mouse_only = Vec::new();
        assert!(!parse_raw_input_buffer(&buf[..stride], 1, &mut mouse_only));
        assert_eq!(mouse_only.len(), 1);
    }

    #[test]
    fn cursor_capture_releases_on_killfocus_overlay_pause_and_balances_show_cursor() {
        let mut host = FakeCursorHost::with_rect(100, 50, 1380, 770);
        // Simulate a title that already called ShowCursor(TRUE) twice before capture.
        host.cursor_counter = 2;
        let owner = 42;
        let worker = 99;
        let hwnd = 0x1234;

        let mut cap = CursorCapture::new(true, true);
        cap.attach(hwnd, owner, true, &mut host);
        assert!(cap.is_clipped());
        assert!(cap.is_cursor_hidden());
        assert_eq!(host.cursor_counter, -1);
        assert_eq!(host.clipped_to.unwrap().left, 100);

        // WM_KILLFOCUS: immediately unclips and restores exact ShowCursor counter.
        cap.set_focused(false, owner, &mut host);
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        assert!(host.clipped_to.is_none());
        assert_eq!(host.cursor_counter, 2);

        // WM_SETFOCUS: re-clips and re-hides cleanly.
        cap.set_focused(true, owner, &mut host);
        assert!(cap.is_clipped());
        assert!(cap.is_cursor_hidden());
        assert_eq!(host.cursor_counter, -1);

        // Overlay opened from a non-owner worker thread: unclips immediately,
        // defers ShowCursor restoration to owner-thread sync.
        cap.enter_overlay(worker, &mut host);
        assert!(!cap.is_clipped());
        assert!(host.clipped_to.is_none());
        assert!(cap.pending_visibility_sync);
        cap.sync_owner_thread(owner, &mut host);
        assert!(!cap.is_cursor_hidden());
        assert_eq!(host.cursor_counter, 2);

        // Closing overlay from worker thread queues re-capture, completed on owner thread.
        cap.leave_overlay(worker, &mut host);
        assert!(cap.is_clipped());
        assert!(cap.pending_visibility_sync);
        cap.sync_owner_thread(owner, &mut host);
        assert!(cap.is_cursor_hidden());
        assert_eq!(host.cursor_counter, -1);

        // Pause transition releases clip & cursor, resume restores both.
        cap.set_paused(true, owner, &mut host);
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        assert_eq!(host.cursor_counter, 2);
        cap.set_paused(false, owner, &mut host);
        assert!(cap.is_clipped());
        assert!(cap.is_cursor_hidden());

        // Move/size loop releases during drag and re-clips to the moved rectangle.
        cap.set_size_move(true, owner, &mut host);
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        host.rect = Some(RECT {
            left: 200,
            top: 100,
            right: 1480,
            bottom: 820,
        });
        cap.set_size_move(false, owner, &mut host);
        assert!(cap.is_clipped());
        assert_eq!(host.clipped_to.unwrap().left, 200);
        assert_eq!(host.cursor_counter, -1);

        // Minimize (SC_MINIMIZE / WM_SIZE(SIZE_MINIMIZED)) + focus loss + synthetic
        // WM_MOVE(-32000, -32000) must keep capture released until restored via
        // WM_SIZE(SIZE_RESTORED) + WM_SETFOCUS (even without SC_RESTORE, e.g. Alt-Tab).
        cap.set_geometry(Some(true), owner, &mut host);
        cap.set_focused(false, owner, &mut host);
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        cap.set_geometry(None, owner, &mut host);
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        cap.set_geometry(Some(false), owner, &mut host);
        cap.set_focused(true, owner, &mut host);
        assert!(cap.is_clipped());
        assert!(cap.is_cursor_hidden());
        assert!(!cap.paused);

        // Detach always leaves desktop unclipped and ShowCursor balanced.
        assert!(cap.detach(hwnd, owner, &mut host));
        assert!(!cap.is_clipped());
        assert!(!cap.is_cursor_hidden());
        assert!(host.clipped_to.is_none());
        assert_eq!(host.cursor_counter, 2);
    }

    #[test]
    fn raw_mouse_and_cursor_capture_switches_are_strictly_independent() {
        let owner = 7;
        let hwnd = 0x5678;

        // Both off (default): neither raw mouse ring nor cursor clipping activates.
        let mut host = FakeCursorHost::with_rect(0, 0, 1280, 720);
        let mut off = CursorCapture::new(false, false);
        off.attach(hwnd, owner, true, &mut host);
        assert!(!off.input_active());
        assert!(!off.is_clipped());
        assert!(!off.is_cursor_hidden());
        assert!(host.clip_calls.is_empty());
        assert_eq!(host.hide_calls, 0);

        // VICTORY_RAW_MOUSE=1 alone: raw mouse ring active, zero ClipCursor/ShowCursor calls.
        let mut host = FakeCursorHost::with_rect(0, 0, 1280, 720);
        let mut raw_only = CursorCapture::new(true, false);
        raw_only.attach(hwnd, owner, true, &mut host);
        assert!(raw_only.input_active());
        assert!(!raw_only.is_clipped());
        assert!(!raw_only.is_cursor_hidden());
        assert!(host.clip_calls.is_empty());
        assert_eq!(host.hide_calls, 0);

        // VICTORY_CURSOR_CAPTURE=1 alone: ClipCursor/ShowCursor active without enabling raw mouse ring.
        let mut host = FakeCursorHost::with_rect(0, 0, 1280, 720);
        let mut clip_only = CursorCapture::new(false, true);
        clip_only.attach(hwnd, owner, true, &mut host);
        assert!(!clip_only.input_active());
        assert!(clip_only.is_clipped());
        assert!(clip_only.is_cursor_hidden());
        assert_eq!(host.hide_calls, 1);
        // Null HWND must never spuriously detach an active or idle window.
        assert!(!clip_only.detach(0, owner, &mut host));
        assert!(clip_only.is_clipped());
        assert!(clip_only.detach(hwnd, owner, &mut host));
        assert!(!clip_only.detach(0, owner, &mut host));
        assert!(!clip_only.is_clipped());
        assert_eq!(host.cursor_counter, 0);

        // Baseline legacy accumulator preserves relative motion and wheels when VICTORY_RAW_MOUSE=0.
        let mut legacy = MouseState::default();
        accumulate_legacy(&mut legacy, raw_mouse(0, 14, -5, RI_MOUSE_WHEEL, 120));
        accumulate_legacy(&mut legacy, raw_mouse(0, -4, 2, RI_MOUSE_HWHEEL, -60));
        accumulate_legacy(&mut legacy, raw_mouse(MOUSE_MOVE_ABSOLUTE, 65535, 65535, 0, 0));
        assert_eq!((legacy.position_x, legacy.position_y), (10, -3));
        assert_eq!((legacy.wheel_x, legacy.wheel_y), (-60, 120));
    }
}
