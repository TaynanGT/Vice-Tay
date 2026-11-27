//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3), with PS5 HID input when XInput reports no pads or both
//! sources are explicitly requested. With no pad the keyboard drives one
//! stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard is a device of its own
//! too. `VICTORY_GAMEPAD_MOUSE=1` adds a mouse while the title has focus.
//! A poll thread records changes and runs device callbacks.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

#[cfg(any(windows, test))]
mod dualsense;
#[cfg(windows)]
pub(super) mod mouse;

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const MOUSE_KIND: u32 = 0x20;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, keyboard and mouse.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const MOUSE: usize = PADS + 2;
pub(super) const DEVICES: usize = MOUSE + 1;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
/// Extended history depth for high-rate (1000-8000 Hz) raw mouse readings.
const MOUSE_HISTORY: usize = 1024;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// GameInput v0 mouse state. The four axes are cumulative relative values.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct MouseState {
    buttons: u32,
    position_x: i64,
    position_y: i64,
    wheel_x: i64,
    wheel_y: i64,
}
const _: () =
    assert!(size_of::<MouseState>() == 40 && std::mem::offset_of!(MouseState, position_x) == 8);

/// One microsecond-timestamped mouse reading sample from the raw input ring.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct MouseSample {
    timestamp: u64,
    state: MouseState,
}

/// Drained raw mouse batch for one poll or read synchronization.
#[derive(Debug, PartialEq)]
enum MouseBatch {
    Disconnected,
    Connected(Vec<MouseSample>),
}

/// Four-channel gamepad rumble/trigger vibration levels (`0.0..=1.0`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Rumble {
    low: f32,
    high: f32,
    left_trigger: f32,
    right_trigger: f32,
}

impl Rumble {
    fn from_params(params: [f32; 4]) -> Self {
        let clamp = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        Self {
            low: clamp(params[0]),
            high: clamp(params[1]),
            left_trigger: clamp(params[2]),
            right_trigger: clamp(params[3]),
        }
    }

    fn low_u16(self) -> u16 {
        (self.low * f32::from(u16::MAX)) as u16
    }

    fn high_u16(self) -> u16 {
        (self.high * f32::from(u16::MAX)) as u16
    }
}

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down: the keyboard's only while the title's window has
    /// focus, scripted ones (`VICTORY_KEYS_FILE`, `VICTORY_KEYS`) always.
    fn keys(&mut self) -> Vec<KeyState>;
    /// The mouse while the title window has focus, if enabled.
    fn mouse(&mut self) -> Option<MouseState> {
        None
    }
    /// Drains timestamped mouse samples (defaults to wrapping `mouse()`).
    fn mouse_batch(&mut self) -> MouseBatch {
        match self.mouse() {
            Some(state) => MouseBatch::Connected(vec![MouseSample {
                timestamp: now(),
                state,
            }]),
            None => MouseBatch::Disconnected,
        }
    }
    /// Four-channel vibration: low/high main motors and left/right trigger motors.
    fn rumble(&mut self, slot: usize, rumble: Rumble);
}

fn fallback(keys: &[KeyState]) -> PadState {
    let on = |vk: u8| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: FALLBACK_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z'),
        right_trigger: on(b'C'),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

/// Two pads as one: buttons of both, and of each axis and trigger the one
/// pushed further.
fn merged(a: PadState, b: PadState) -> PadState {
    let further = |x: f32, y: f32| if y.abs() > x.abs() { y } else { x };
    PadState {
        buttons: a.buttons | b.buttons,
        left_trigger: further(a.left_trigger, b.left_trigger),
        right_trigger: further(a.right_trigger, b.right_trigger),
        left_x: further(a.left_x, b.left_x),
        left_y: further(a.left_y, b.left_y),
        right_x: further(a.right_x, b.right_x),
        right_y: further(a.right_y, b.right_y),
    }
}

/// Independent opt-in switches for native input and haptics features.
/// All four default to `false` (`0`) so baseline behavior stays unchanged
/// unless a subsystem is explicitly enabled for testing or use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct InputSwitches {
    /// `VICTORY_RAW_MOUSE=1`: event-driven 8 kHz `WM_INPUT`/`GetRawInputBuffer` mouse ring.
    pub(super) raw_mouse: bool,
    /// `VICTORY_CURSOR_CAPTURE=1`: `ClipCursor` + balanced `ShowCursor` focus/minimize ownership.
    pub(super) cursor_capture: bool,
    /// `VICTORY_KBM_PAD=1`: bind logical mouse, keyboard, and pads simultaneously to local XUID.
    pub(super) kbm_pad: bool,
    /// `VICTORY_HAPTICS=1`: four-channel trigger haptics (`Windows.Gaming.Input` + DualSense HID).
    pub(super) haptics: bool,
}

impl InputSwitches {
    pub(super) fn from_lookup<F>(get: F) -> Self
    where
        F: Fn(&str) -> Option<std::ffi::OsString>,
    {
        let on = |name: &str| get(name).is_some_and(|v| v == "1");
        Self {
            raw_mouse: on("VICTORY_RAW_MOUSE"),
            cursor_capture: on("VICTORY_CURSOR_CAPTURE"),
            kbm_pad: on("VICTORY_KBM_PAD"),
            haptics: on("VICTORY_HAPTICS"),
        }
    }

    pub(super) fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var_os(name))
    }
}

pub(super) fn switches() -> InputSwitches {
    static SWITCHES: LazyLock<InputSwitches> = LazyLock::new(InputSwitches::from_env);
    *SWITCHES
}

/// Whether the title sees one pad that is always there, fed by the keyboard
/// and by every host pad (on by default; `VICTORY_ONE_PAD=0` shows each host
/// pad as a device of its own, and the stand-in only while there is none).
/// The title keeps to the pad it started with: with the stand-in taken away
/// when a pad was switched on, it showed its controller-disconnected screen
/// until the pad was off again. It also keeps a pad the host lists twice (its
/// own slot and a remapper's) from being two players.
fn one_pad() -> bool {
    std::env::var_os("VICTORY_ONE_PAD").is_none_or(|v| v != "0")
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    match index {
        KEYBOARD => KEYBOARD_KIND,
        MOUSE => MOUSE_KIND,
        _ => GAMEPAD_KIND,
    }
}

/// Canonical 32-byte `APP_LOCAL_DEVICE_ID` for logical device `index`
/// (`PAD[0..4]`, stand-in pad, `KEYBOARD`, `MOUSE`).
pub(super) const fn device_id(index: usize) -> [u8; 32] {
    let mut id = [0u8; 32];
    id[0] = b'v';
    id[1] = b'i';
    id[2] = b'c';
    id[3] = b't';
    id[4] = b'o';
    id[5] = b'r';
    id[6] = b'y';
    id[31] = index as u8 + 1;
    id
}

/// Whether `id` matches one of our logical GameInput devices (`PAD[0..4]`,
/// stand-in pad, `KEYBOARD`, or `MOUSE`), used by `xuser::find_for_device`.
pub(super) fn is_known_device_id(id: &[u8; 32]) -> bool {
    id[..7] == *b"victory"
        && id[7..31].iter().all(|&b| b == 0)
        && (1..=(DEVICES as u8)).contains(&id[31])
}

impl Device {
    fn new(index: usize, haptics: bool) -> Device {
        let id = device_id(index);
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency by default; add left/right triggers when `VICTORY_HAPTICS=1`.
            rumble_motors: if index <= FALLBACK {
                if haptics { 0xF } else { 0x3 }
            } else {
                0
            },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
    Mouse(MouseState),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
            Value::Mouse(_) => MOUSE_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

struct State {
    slots: [Slot; DEVICES],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    /// Lock-free raw mouse ring drained directly on read/poll without blocking on `source`.
    #[cfg(windows)]
    mouse_ring: Option<Arc<mouse::MouseRing>>,
    /// The title sees one pad ([`one_pad`]).
    one_pad: bool,
    /// `VICTORY_HAPTICS=1`: forward trigger channels (`params[2..4]`) alongside main rumble.
    haptics: bool,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
    /// Wakes the background poll loop immediately on event-driven mouse status changes.
    poll_wake: (Mutex<u64>, Condvar),
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    #[cfg(any(test, not(windows)))]
    fn new_with_options(source: Box<dyn Source>, one_pad: bool, haptics: bool) -> &'static Hub {
        #[cfg(windows)]
        {
            Self::new_with_mouse_ring(source, one_pad, haptics, None)
        }
        #[cfg(not(windows))]
        {
            Self::new_inner(source, one_pad, haptics)
        }
    }

    #[cfg(windows)]
    fn new_with_mouse_ring(
        source: Box<dyn Source>,
        one_pad: bool,
        haptics: bool,
        mouse_ring: Option<Arc<mouse::MouseRing>>,
    ) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(|index| Device::new(index, haptics)),
            source: Mutex::new(source),
            mouse_ring,
            one_pad,
            haptics,
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
            poll_wake: (Mutex::new(0), Condvar::new()),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        let hub = unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        };
        if let Some(ring) = &hub.mouse_ring {
            ring.set_wake(Arc::new(move || hub.wake_poll()));
        }
        hub
    }

    #[cfg(not(windows))]
    fn new_inner(source: Box<dyn Source>, one_pad: bool, haptics: bool) -> &'static Hub {
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(|index| Device::new(index, haptics)),
            source: Mutex::new(source),
            one_pad,
            haptics,
            state: Mutex::new(State {
                slots: Default::default(),
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
            }),
            returned: Condvar::new(),
            poll_wake: (Mutex::new(0), Condvar::new()),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    fn wake_poll(&self) {
        let mut seq = lock(&self.poll_wake.0);
        *seq = seq.wrapping_add(1);
        self.poll_wake.1.notify_all();
    }

    fn has_mouse_ring(&self) -> bool {
        #[cfg(windows)]
        {
            self.mouse_ring.is_some()
        }
        #[cfg(not(windows))]
        {
            false
        }
    }

    /// Drains event-driven raw mouse samples directly into `state.slots[MOUSE]`
    /// without touching `self.source`. When `deliver_now` is false (read paths),
    /// connect/disconnect callbacks are deferred to `state.pending` so title read
    /// calls never invoke callbacks re-entrantly.
    fn sync_mouse_locked(&self, state: &mut State, deliver_now: bool) -> Vec<Event> {
        #[cfg(windows)]
        {
            let Some(ring) = &self.mouse_ring else {
                return Vec::new();
            };
            let batch = ring.drain();
            let mut events = Vec::new();
            state.update_mouse_batch(&self.devices, batch, now(), &mut events);
            if !deliver_now && !events.is_empty() {
                state.pending.append(&mut events);
                self.wake_poll();
            }
            events
        }
        #[cfg(not(windows))]
        {
            let _ = (state, deliver_now);
            Vec::new()
        }
    }

    /// A hub over the host's input, polled on pad/keyboard cadence and woken
    /// immediately on event-driven mouse status transitions when `VICTORY_RAW_MOUSE=1`.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let sw = switches();
        #[cfg(windows)]
        let hub = Hub::new_with_mouse_ring(
            source,
            one_pad(),
            sw.haptics,
            sw.raw_mouse.then(mouse::global_ring),
        );
        #[cfg(not(windows))]
        let hub = Hub::new_with_options(source, one_pad(), sw.haptics);
        hub.poll();
        std::thread::spawn(move || {
            let mut seen = *lock(&hub.poll_wake.0);
            loop {
                {
                    let guard = lock(&hub.poll_wake.0);
                    let (guard, _) = hub
                        .poll_wake
                        .1
                        .wait_timeout_while(guard, POLL, |s| *s == seen)
                        .unwrap_or_else(PoisonError::into_inner);
                    seen = *guard;
                }
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys, fallback_mouse) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            let mouse = (!self.has_mouse_ring()).then(|| source.mouse_batch());
            (pads, source.keys(), mouse)
        };
        let events = {
            let mut state = self.state();
            let mut events = self.sync_mouse_locked(&mut state, true);
            events.extend(state.poll(&self.devices, self.one_pad, pads, keys, fallback_mouse));
            events
        };
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

impl State {
    fn poll(
        &mut self,
        devices: &[Device; DEVICES],
        one_pad: bool,
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        mouse: Option<MouseBatch>,
    ) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        if let Some(mouse) = mouse {
            self.update_mouse_batch(devices, mouse, now, &mut events);
        }
        if one_pad {
            let pad = pads.into_iter().flatten().fold(fallback(&keys), merged);
            self.update(devices, FALLBACK, Some(Value::Pad(pad)), now, &mut events);
            self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
            return events;
        }
        let any_pad = pads.iter().any(Option::is_some);
        for (slot, pad) in pads.into_iter().enumerate() {
            self.update(devices, slot, pad.map(Value::Pad), now, &mut events);
        }
        let fallback = (!any_pad).then(|| Value::Pad(fallback(&keys)));
        self.update(devices, FALLBACK, fallback, now, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(keys)), now, &mut events);
        events
    }

    /// Appends every microsecond-timestamped mouse sample into `slots[MOUSE]`
    /// without dropping fast click edges or coalescing across ring entries.
    fn update_mouse_batch(
        &mut self,
        devices: &[Device; DEVICES],
        batch: MouseBatch,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[MOUSE];
        let was = slot.connected;
        match batch {
            MouseBatch::Disconnected => {
                slot.connected = false;
            }
            MouseBatch::Connected(samples) => {
                slot.connected = true;
                let mut had_initial = was;
                for sample in samples {
                    let value = Value::Mouse(sample.state);
                    if !had_initial || slot.history.back().is_none_or(|r| r.value != value) {
                        slot.sequence += 1;
                        if slot.history.len() == MOUSE_HISTORY {
                            slot.history.pop_front();
                        }
                        slot.history.push_back(Ref::new(Reading {
                            vtable: &READING,
                            refs: AtomicU32::new(1),
                            device: &devices[MOUSE],
                            sequence: slot.sequence,
                            timestamp: sample.timestamp,
                            value,
                        }));
                        had_initial = true;
                    }
                }
            }
        }
        if was != slot.connected {
            tracing::info!(device = MOUSE, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, MOUSE) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(MOUSE, now, current, previous)),
            );
        }
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if let Some(value) = value
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.sequence += 1;
            if slot.history.len() == HISTORY {
                slot.history.pop_front();
            }
            slot.history.push_back(Ref::new(Reading {
                vtable: &READING,
                refs: AtomicU32::new(1),
                device: &devices[index],
                sequence: slot.sequence,
                timestamp: now,
                value,
            }));
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    Box::new(host::Host::default())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: Rumble) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let mut state = hub.state();
    hub.sync_mouse_locked(&mut state, false);
    let newest = |i: usize| {
        state.slots[i]
            .history
            .back()
            .filter(|r| r.kind() & kind != 0)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(newest)
            .max_by_key(|r| r.timestamp)
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let mut state = hub.state();
    if index == MOUSE {
        hub.sync_mouse_locked(&mut state, false);
    }
    let history = &state.slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let mut state = hub.state();
    if index == MOUSE {
        hub.sync_mouse_locked(&mut state, false);
    }
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        hub.sync_mouse_locked(&mut state, false);
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub
                .returned
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    let hub = hub(this);
    let mouse_connected = {
        let mut state = hub.state();
        hub.sync_mouse_locked(&mut state, false);
        state.slots[MOUSE].connected
    };
    match hub.devices.iter().find(|d| {
        d.info.id == *id && (d.index != MOUSE || mouse_connected)
    }) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    let mut state = device.hub().state();
    if device.index == MOUSE {
        device.hub().sync_mouse_locked(&mut state, false);
    }
    if state.slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
/// Routes all four motor channels (low/high + left/right triggers) to the
/// active host backend (DualSense HID output report, `Windows.Gaming.Input`
/// trigger vibration, or XInput fallback).
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    let one_pad = device.hub().one_pad;
    if device.index >= PADS && !(one_pad && device.index == FALLBACK) {
        return;
    }
    let raw = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let mut rumble = Rumble::from_params(raw);
    if !device.hub().haptics {
        rumble.left_trigger = 0.0;
        rumble.right_trigger = 0.0;
    }
    let mut source = lock(&device.hub().source);
    if one_pad {
        // The one pad is every host pad.
        for slot in 0..PADS {
            if source.pad(slot).is_some() {
                source.rumble(slot, rumble);
            }
        }
    } else {
        source.rumble(device.index, rumble);
    }
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        _ => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
extern "system" fn mouse_state(this: *const Reading, out: *mut MouseState) -> bool {
    if out.is_null() {
        return false;
    }
    let Value::Mouse(state) = &reading(this).value else {
        // SAFETY: the title passes room for a v0 mouse state.
        unsafe { out.write(MouseState::default()) };
        return false;
    };
    // SAFETY: the same output contract as above.
    unsafe { out.write(*state) };
    #[cfg(feature = "diagnostics")]
    {
        static REPORTED: std::sync::Once = std::sync::Once::new();
        REPORTED.call_once(|| tracing::info!("the title read a GameInput mouse state"));
    }
    true
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    mouse_state as *const (),       // 0x80
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: XInput pads, then DualSense HID pads, plus focused keyboard and mouse.
#[cfg(windows)]
mod host {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Condvar, LazyLock, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    use windows::Gaming::Input::{Gamepad, GamepadVibration};
    use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, GetKeyState, GetKeyboardLayout, HKL, KLF_NOTELLSHELL,
        LoadKeyboardLayoutW, MAPVK_VK_TO_VSC_EX, MapVirtualKeyExW, ToUnicodeEx, VK_CAPITAL,
        VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_MENU, VK_NUMLOCK, VK_RCONTROL, VK_RMENU,
        VK_RSHIFT, VK_SHIFT,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowThreadProcessId,
    };
    use windows_sys::core::w;

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    /// Windows 10 1607+ flag telling `ToUnicodeEx` not to alter kernel dead-key state.
    const TOUNICODE_NO_STATE_CHANGE: u32 = 0x04;

    /// Fallback US-English HKL when `LoadKeyboardLayoutW` is unavailable.
    const US_ENGLISH_HKL_FALLBACK: usize = 0x0409_0409;

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Builds a 256-byte Win32 keyboard-state table from currently held virtual
    /// keys and lock toggles (`GetKeyboardState` is thread-queue local, so the
    /// poll thread synthesizes the modifier table directly).
    pub(super) fn build_keyboard_state(down: &[u8], caps_lock: bool, num_lock: bool) -> [u8; 256] {
        let mut state = [0u8; 256];
        for &vk in down {
            state[usize::from(vk)] |= 0x80;
        }
        if (state[usize::from(VK_LSHIFT)] | state[usize::from(VK_RSHIFT)]) & 0x80 != 0 {
            state[usize::from(VK_SHIFT)] |= 0x80;
        }
        if (state[usize::from(VK_LCONTROL)] | state[usize::from(VK_RCONTROL)]) & 0x80 != 0 {
            state[usize::from(VK_CONTROL)] |= 0x80;
        }
        if (state[usize::from(VK_LMENU)] | state[usize::from(VK_RMENU)]) & 0x80 != 0 {
            state[usize::from(VK_MENU)] |= 0x80;
        }
        if caps_lock {
            state[usize::from(VK_CAPITAL)] |= 0x01;
        }
        if num_lock {
            state[usize::from(VK_NUMLOCK)] |= 0x01;
        }
        state
    }

    /// Interprets the return code and UTF-16 output buffer from `ToUnicodeEx`
    /// into `(code_point, dead_key)`, supporting BMP characters, UTF-16
    /// surrogate pairs, and dead keys.
    pub(super) fn decode_to_unicode_result(rc: i32, buf: &[u16]) -> (u32, bool) {
        if rc < 0 {
            let dead_char = buf.first().copied().unwrap_or(0);
            return (u32::from(dead_char), true);
        }
        if rc == 0 {
            return (0, false);
        }
        let len = (rc as usize).min(buf.len());
        let code_point = char::decode_utf16(buf[..len].iter().copied())
            .find_map(Result::ok)
            .map_or(0, |c| c as u32);
        (code_point, false)
    }

    fn ime_english_enabled() -> bool {
        static ENABLED: LazyLock<bool> = LazyLock::new(|| {
            std::env::var_os("VICTORY_IME_ENGLISH").is_some_and(|v| v == "1")
        });
        *ENABLED
    }

    fn english_layout() -> HKL {
        static HKL_US: LazyLock<usize> = LazyLock::new(|| {
            // SAFETY: constant NUL-terminated US-English layout id; KLF_NOTELLSHELL
            // avoids altering the user's taskbar language indicator.
            let loaded = unsafe { LoadKeyboardLayoutW(w!("00000409"), KLF_NOTELLSHELL) };
            if loaded.is_null() {
                US_ENGLISH_HKL_FALLBACK
            } else {
                loaded as usize
            }
        });
        *HKL_US as HKL
    }

    fn active_keyboard_layout(ime_english: bool) -> HKL {
        if ime_english {
            return english_layout();
        }
        // SAFETY: plain queries of the foreground window's thread and its keyboard layout.
        unsafe {
            let fg = GetForegroundWindow();
            let thread = if fg.is_null() {
                0
            } else {
                GetWindowThreadProcessId(fg, std::ptr::null_mut())
            };
            let hkl = GetKeyboardLayout(thread);
            if hkl.is_null() {
                GetKeyboardLayout(0)
            } else {
                hkl
            }
        }
    }

    fn translate_key(vk: u8, key_state: &[u8; 256], hkl: HKL) -> KeyState {
        // SAFETY: plain layout-aware scan code query.
        let scan_code = unsafe { MapVirtualKeyExW(u32::from(vk), MAPVK_VK_TO_VSC_EX, hkl) };
        let mut buf = [0u16; 8];
        // Mask scan_code to low 8 bits so extended 0xE0xx codes never set bit 15 (key-up).
        // SAFETY: valid 256-byte key state, writable 8-code-unit UTF-16 buffer, and
        // `TOUNICODE_NO_STATE_CHANGE` so kernel dead-key state is untouched.
        let rc = unsafe {
            ToUnicodeEx(
                u32::from(vk),
                scan_code & 0xFF,
                key_state.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
                TOUNICODE_NO_STATE_CHANGE,
                hkl,
            )
        };
        let (code_point, dead_key) = decode_to_unicode_result(rc, &buf);
        KeyState {
            scan_code,
            code_point,
            virtual_key: vk,
            dead_key,
        }
    }

    /// Dedicated MTA worker routing four-channel gamepad vibration (including
    /// impulse trigger motors) through `Windows.Gaming.Input` without altering
    /// any title thread's COM apartment or blocking indefinitely on WinRT.
    mod wgi {
        use super::*;

        const CALL_TIMEOUT: Duration = Duration::from_millis(15);

        struct Request {
            seq: u64,
            slot: usize,
            rumble: Rumble,
            done_seq: u64,
            handled: bool,
        }

        struct Worker {
            available: AtomicBool,
            mailbox: Mutex<Request>,
            wake_worker: Condvar,
            wake_caller: Condvar,
        }

        pub(super) fn to_vibration(rumble: Rumble) -> GamepadVibration {
            GamepadVibration {
                LeftMotor: f64::from(rumble.low),
                RightMotor: f64::from(rumble.high),
                LeftTrigger: f64::from(rumble.left_trigger),
                RightTrigger: f64::from(rumble.right_trigger),
            }
        }

        static WORKER: LazyLock<Worker> = LazyLock::new(|| {
            let worker = Worker {
                available: AtomicBool::new(true),
                mailbox: Mutex::new(Request {
                    seq: 0,
                    slot: 0,
                    rumble: Rumble::default(),
                    done_seq: 0,
                    handled: false,
                }),
                wake_worker: Condvar::new(),
                wake_caller: Condvar::new(),
            };
            if let Err(error) = std::thread::Builder::new()
                .name("WGI haptics".into())
                .spawn(run_worker)
            {
                tracing::debug!(%error, "could not spawn Windows.Gaming.Input worker");
                worker.available.store(false, Ordering::Release);
            }
            worker
        });

        fn run_worker() {
            // SAFETY: initializing WinRT MTA exclusively on our dedicated worker thread.
            let _ = unsafe { RoInitialize(RO_INIT_MULTITHREADED) };
            // Probe activation once; if WinRT/WGI is absent (e.g., Wine), disable cleanly.
            if Gamepad::Gamepads().is_err() {
                WORKER.available.store(false, Ordering::Release);
                let mut req = WORKER.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
                req.done_seq = req.seq;
                req.handled = false;
                WORKER.wake_caller.notify_all();
                return;
            }
            let mut processed = 0u64;
            loop {
                let (seq, slot, rumble) = {
                    let mut req = WORKER.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
                    while req.seq == processed {
                        req = WORKER
                            .wake_worker
                            .wait(req)
                            .unwrap_or_else(PoisonError::into_inner);
                    }
                    (req.seq, req.slot, req.rumble)
                };
                let handled = apply_vibration(slot, rumble);
                processed = seq;
                let mut req = WORKER.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
                req.done_seq = seq;
                req.handled = handled;
                WORKER.wake_caller.notify_all();
            }
        }

        fn apply_vibration(slot: usize, rumble: Rumble) -> bool {
            let Ok(pads) = Gamepad::Gamepads() else {
                return false;
            };
            let Ok(count) = pads.Size() else {
                return false;
            };
            let Ok(index) = u32::try_from(slot) else {
                return false;
            };
            if index >= count {
                return false;
            }
            let Ok(pad) = pads.GetAt(index) else {
                return false;
            };
            pad.SetVibration(to_vibration(rumble)).is_ok()
        }

        /// Starts WinRT gamepad discovery early when a host XInput pad appears.
        pub(super) fn warm_up() {
            LazyLock::force(&WORKER);
        }

        /// Attempts to drive all four vibration motors via `Windows.Gaming.Input`.
        /// Returns `true` if handled; `false` instructs the caller to fall back to XInput.
        pub(super) fn try_set_rumble(slot: usize, rumble: Rumble) -> bool {
            let worker = &*WORKER;
            if !worker.available.load(Ordering::Acquire) {
                return false;
            }
            let mut req = worker.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
            let target = req.seq.wrapping_add(1).max(1);
            req.seq = target;
            req.slot = slot;
            req.rumble = rumble;
            worker.wake_worker.notify_one();

            let deadline = Instant::now() + CALL_TIMEOUT;
            while (req.done_seq.wrapping_sub(target) as i64) < 0 {
                if !worker.available.load(Ordering::Acquire) {
                    return false;
                }
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = worker.wake_caller.wait_timeout(req, left);
                req = waited.unwrap_or_else(PoisonError::into_inner).0;
            }
            req.handled
        }
    }

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        /// The input snapshot for this poll and the HID connection owning each slot.
        pads: [Option<PadState>; PADS],
        hid_slots: [Option<u64>; PADS],
    }

    impl Host {
        fn xinput(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            if switches().haptics {
                wgi::warm_up();
            }
            Some(convert(&state.Gamepad))
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if slot == 0 {
                let xinput = std::array::from_fn(|slot| self.xinput(slot));
                self.pads = super::dualsense::slots(
                    super::dualsense::mode(),
                    xinput,
                    &mut self.hid_slots,
                    super::dualsense::pads,
                );
            }
            let mut pad = self.pads[slot]?;
            // The scripted keys (`VICTORY_KEYS_FILE` and `VICTORY_KEYS`)
            // also press the first pad's buttons, by the stand-in pad's
            // table: with a real pad connected the stand-in is not there,
            // and a title that listens to the pad (the title's start screen)
            // saw nothing.
            if slot == 0 {
                let mut asked = asked_keys();
                asked.extend(scripted());
                for vk in asked {
                    if let Some(&(_, button)) =
                        super::FALLBACK_BUTTONS.iter().find(|(key, _)| *key == vk)
                    {
                        pad.buttons |= button;
                    }
                }
            }
            Some(pad)
        }

        fn keys(&mut self) -> Vec<KeyState> {
            let mut down = asked_keys();
            down.extend(scripted());
            if focused() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            if down.is_empty() {
                return Vec::new();
            }
            // SAFETY: plain queries of lock toggle states.
            let caps_lock = unsafe { GetKeyState(i32::from(VK_CAPITAL)) & 1 != 0 };
            let num_lock = unsafe { GetKeyState(i32::from(VK_NUMLOCK)) & 1 != 0 };
            let key_state = build_keyboard_state(&down, caps_lock, num_lock);
            let hkl = active_keyboard_layout(ime_english_enabled());
            down.into_iter()
                .map(|vk| translate_key(vk, &key_state, hkl))
                .collect()
        }

        fn mouse_batch(&mut self) -> MouseBatch {
            if switches().raw_mouse {
                super::mouse::global_ring().drain()
            } else {
                match super::mouse::legacy_sample() {
                    Some(state) => MouseBatch::Connected(vec![MouseSample {
                        timestamp: now(),
                        state,
                    }]),
                    None => MouseBatch::Disconnected,
                }
            }
        }

        fn rumble(&mut self, slot: usize, rumble: Rumble) {
            if let Some(hid_id) = self.hid_slots.get(slot).copied().flatten() {
                if switches().haptics {
                    if super::dualsense::set_rumble(hid_id, rumble) {
                        return;
                    }
                } else {
                    // Preserve #314 default USB DualSense HID rumble when VICTORY_HAPTICS=0.
                    super::dualsense::set_motors(hid_id, rumble.low_u16(), rumble.high_u16());
                    return;
                }
            }
            if switches().haptics {
                // Route impulse trigger + main motors through Windows.Gaming.Input when available;
                // avoid calling XInputSetState afterward as XInput overwrites trigger rumble with zero.
                if wgi::try_set_rumble(slot, rumble) {
                    return;
                }
            }
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: rumble.low_u16(),
                wRightMotorSpeed: rumble.high_u16(),
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// A key's virtual-key code: a letter, a digit, or `Enter`, `Esc`,
    /// `Tab`, `Back`, `Up`, `Down`, `Left`, `Right` (the stand-in pad's
    /// table maps them to buttons).
    fn key_code(key: &str) -> Option<u8> {
        Some(match key {
            "Enter" => 0x0D,
            "Esc" => 0x1B,
            "Tab" => 0x09,
            "Back" => 0x08,
            "Up" => 0x26,
            "Down" => 0x28,
            "Left" => 0x25,
            "Right" => 0x27,
            k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                k.as_bytes()[0].to_ascii_uppercase()
            }
            _ => return None,
        })
    }

    /// Test aid: `VICTORY_KEYS_FILE=<file>`. A script writes lines
    /// `<key> <seconds>` into the file (keys as [`key_code`] names them);
    /// the next look, at most 50 ms later, holds each key that long, focus
    /// or not, and removes the file. A measuring script can then answer
    /// what is on screen (the account screen's prompt, a turn of the
    /// camera) while someone else uses the desktop. Unset: nothing is read.
    fn asked_keys() -> Vec<u8> {
        /// The file, and the name it is taken to before it is read ([`take`]).
        static FILE: LazyLock<Option<(std::path::PathBuf, std::path::PathBuf)>> =
            LazyLock::new(|| {
                let file = std::env::var_os("VICTORY_KEYS_FILE")?;
                let mut taken = file.clone();
                taken.push(".taken");
                Some((file.into(), taken.into()))
            });
        /// Keys held, until when; when the file was last looked at.
        type Held = (Vec<(Instant, u8)>, Option<Instant>);
        static HELD: Mutex<Held> = Mutex::new((Vec::new(), None));
        let Some((file, taken)) = &*FILE else {
            return Vec::new();
        };
        let now = Instant::now();
        let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
        let due = held
            .1
            .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(50));
        if due {
            held.1 = Some(now);
            if let Some(text) = take(file, taken) {
                held.0.extend(asked(&text, now));
            }
        }
        held.0.retain(|&(until, _)| until > now);
        held.0.iter().map(|&(_, vk)| vk).collect()
    }

    /// What the keys file holds, taken away: it is first renamed to `taken`
    /// (one step), then read and removed there. A batch the script moves in
    /// after the rename is a new file at `file`, read by the next look; read
    /// and removed by name, it could be removed unread between the two.
    fn take(file: &std::path::Path, taken: &std::path::Path) -> Option<String> {
        std::fs::rename(file, taken).ok()?;
        let text = std::fs::read_to_string(taken);
        let _ = std::fs::remove_file(taken);
        text.ok()
    }

    /// The keys of a [`asked_keys`] file read at `now`: each with the time
    /// it is held until. Lines that do not read as `<key> <seconds>` are
    /// left out.
    fn asked(text: &str, now: Instant) -> Vec<(Instant, u8)> {
        text.lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                let vk = key_code(words.next()?)?;
                let long = Duration::try_from_secs_f64(words.next()?.parse().ok()?).ok()?;
                Some((now + long, vk))
            })
            .collect()
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts (`<from>-<to>:<key>`: all that
    /// time), with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the stand-in pad's table maps them to buttons).
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = key_code(key.trim())?;
                let time = |t: &str| Duration::try_from_secs_f64(t.trim().parse().ok()?).ok();
                let (from, to) = match at.split_once('-') {
                    Some((from, to)) => (time(from)?, time(to)?),
                    None => (time(at)?, time(at)? + Duration::from_millis(200)),
                };
                Some((from, to, vk))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(from, to, _)| (from..to).contains(&now))
            .map(|&(_, _, vk)| vk)
            .collect()
    }

    /// The foreground window is one of ours (the title's or our fallback).
    fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn a_keys_file_holds_each_key_for_its_seconds() {
        let now = Instant::now();
        let asked = asked("Enter 0.2\nJ 1.5\nnot a key 3\nW -1\nX\n", now);
        assert_eq!(
            asked,
            [
                (now + Duration::from_millis(200), 0x0D),
                (now + Duration::from_millis(1500), b'J')
            ]
        );
    }

    #[test]
    fn a_batch_written_after_the_file_is_taken_waits_for_the_next_look() {
        let dir = tempfile::tempdir().unwrap();
        let (file, taken) = (
            dir.path().join("keys.txt"),
            dir.path().join("keys.txt.taken"),
        );
        assert_eq!(take(&file, &taken), None);
        std::fs::write(&file, "Enter 0.3\n").unwrap();
        assert_eq!(take(&file, &taken).as_deref(), Some("Enter 0.3\n"));
        assert!(!file.exists() && !taken.exists());
        // the script's next batch, after the take: still there, and the next look's
        std::fs::write(&file, "L 1.5\n").unwrap();
        assert!(file.exists());
        assert_eq!(take(&file, &taken).as_deref(), Some("L 1.5\n"));
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn to_unicode_ex_populates_code_points_modifiers_dead_keys_and_ime_english() {
        // Modifier propagation and lock toggles.
        let ks = build_keyboard_state(&[VK_LSHIFT as u8, b'A'], false, true);
        assert_eq!(ks[usize::from(VK_SHIFT)] & 0x80, 0x80);
        assert_eq!(ks[usize::from(VK_CAPITAL)] & 0x01, 0);
        assert_eq!(ks[usize::from(VK_NUMLOCK)] & 0x01, 1);

        // Surrogate pair decoding (U+1F600 = D83D DE00), dead key, and non-char.
        assert_eq!(decode_to_unicode_result(2, &[0xD83D, 0xDE00]), (0x1F600, false));
        assert_eq!(decode_to_unicode_result(-1, &[b'^' as u16]), (b'^' as u32, true));
        assert_eq!(decode_to_unicode_result(0, &[b'x' as u16]), (0, false));

        // Real Windows ToUnicodeEx using US-English layout (`VICTORY_IME_ENGLISH` path).
        let hkl = active_keyboard_layout(true);
        let plain = build_keyboard_state(b"A", false, false);
        let key_a = translate_key(b'A', &plain, hkl);
        assert_eq!(key_a.virtual_key, b'A');
        assert_eq!(key_a.code_point, u32::from(b'a'));
        assert!(!key_a.dead_key);
        assert_ne!(key_a.scan_code, 0);

        let shifted = build_keyboard_state(&[VK_LSHIFT as u8, b'A', b'1'], false, false);
        assert_eq!(translate_key(b'A', &shifted, hkl).code_point, u32::from(b'A'));
        assert_eq!(translate_key(b'1', &shifted, hkl).code_point, u32::from(b'!'));

        let caps = build_keyboard_state(b"A", true, false);
        assert_eq!(translate_key(b'A', &caps, hkl).code_point, u32::from(b'A'));

        // Extended non-character key (VK_UP -> 0xE048): must not misread bit 15 or emit garbage.
        let arrow = build_keyboard_state(&[0x26], false, false);
        let up = translate_key(0x26, &arrow, hkl);
        assert_eq!(up.code_point, 0);
        assert!(!up.dead_key);
    }

    #[test]
    fn wgi_vibration_mapping_and_missing_pad_fallback_are_safe() {
        let r = Rumble::from_params([0.25, 0.75, 0.5, 1.0]);
        let vib = wgi::to_vibration(r);
        assert!((vib.LeftMotor - 0.25).abs() < 1e-6);
        assert!((vib.RightMotor - 0.75).abs() < 1e-6);
        assert!((vib.LeftTrigger - 0.5).abs() < 1e-6);
        assert!((vib.RightTrigger - 1.0).abs() < 1e-6);

        // Non-finite and out-of-range rumble values clamp safely.
        let clamped = Rumble::from_params([f32::NAN, -1.0, 2.0, f32::INFINITY]);
        assert_eq!(
            clamped,
            Rumble {
                low: 0.0,
                high: 0.0,
                left_trigger: 1.0,
                right_trigger: 0.0,
            }
        );

        // High slot index must safely report unhandled (falling back to XInput) without panicking.
        assert!(!wgi::try_set_rumble(99, r));
    }

    /// Hardware probe, not a check: prints what each host slot holds
    /// whenever that changes, for `VICTORY_PROBE_SECS` (default 60), so a
    /// pad can be unplugged, switched to Bluetooth or put behind a remapper
    /// while it runs. `cargo test -p victory-hle probe_host_pads -- --ignored --nocapture`
    #[test]
    #[ignore = "needs real pads and someone to plug them"]
    fn probe_host_pads() {
        let secs = std::env::var("VICTORY_PROBE_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        let end = Instant::now() + Duration::from_secs(secs);
        let _ = tracing_subscriber::fmt()
            .with_env_filter("victory_hle::modules::gameinput=debug")
            .with_test_writer()
            .try_init();
        let mut host = Host::default();
        let mut last = String::new();
        let start = Instant::now();
        while Instant::now() < end {
            let mut line = String::new();
            for slot in 0..PADS {
                let xinput = host.xinput(slot).is_some();
                let pad = host.pad(slot);
                let from = match (pad, xinput) {
                    (None, _) => "-",
                    (Some(_), true) => "xinput",
                    (Some(_), false) => "hid",
                };
                let pressed = pad.map(|p| p.buttons).unwrap_or(0);
                line += &format!("[{slot}: {from} {pressed:#06x}] ");
            }
            line += &format!("hid pads: {}", super::dualsense::pads().len());
            if line != last {
                println!("{:6.1}s {line}", start.elapsed().as_secs_f32());
                last = line;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        mouse: Option<MouseState>,
        rumble: Vec<(usize, Rumble)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn mouse(&mut self) -> Option<MouseState> {
            self.0.lock().unwrap().mouse
        }
        fn rumble(&mut self, slot: usize, rumble: Rumble) {
            self.0.lock().unwrap().rumble.push((slot, rumble));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with(false)
    }

    fn fake_with(one_pad: bool) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new_with_options(Box::new(Fake(Arc::clone(&host))), one_pad, false);
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn mouse_readings_connect_and_disconnect_in_both_pad_modes() {
        for one_pad in [false, true] {
            let (hub, host) = fake_with(one_pad);
            let log = Log::default();
            register(hub, MOUSE_KIND, 2, &log);
            assert!(log.lock().unwrap().is_empty());
            assert_eq!(current(hub, MOUSE_KIND, MOUSE), Err(DEVICE_DISCONNECTED));

            let sample = MouseState {
                buttons: 0x9,
                position_x: 30,
                position_y: -12,
                wheel_y: -120,
                ..Default::default()
            };
            host.lock().unwrap().mouse = Some(sample);
            hub.poll();
            let r = current(hub, MOUSE_KIND, MOUSE).unwrap();
            let get =
                slot::<extern "system" fn(*const Reading, *mut MouseState) -> bool, _>(r, 0x80);
            let mut state = MouseState::default();
            assert!(get(r, &mut state));
            assert_eq!(state, sample);
            assert!(!get(r, std::ptr::null_mut()));
            assert_eq!(key_count(r), 0);
            release(r);

            let p = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
            assert!(!get(p, &mut state));
            assert_eq!(state, MouseState::default());
            release(p);

            host.lock().unwrap().mouse = None;
            hub.poll();
            assert_eq!(current(hub, MOUSE_KIND, MOUSE), Err(DEVICE_DISCONNECTED));
            assert_eq!(
                *log.lock().unwrap(),
                [(MOUSE, STATUS_CONNECTED, 0), (MOUSE, 0, STATUS_CONNECTED)]
            );
        }
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn a_host_pad_feeds_the_one_pad_which_never_goes() {
        let (hub, host) = fake_with(true);
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);
        let pad = |hub: &Hub| match &hub.state().slots[FALLBACK].history.back().unwrap().value {
            Value::Pad(p) => *p,
            _ => unreachable!(),
        };
        host.lock().unwrap().keys = vec![key(b'D')];
        hub.poll();
        assert_eq!(pad(hub).left_x, 1.0);
        // A pad is switched on, in any slot: its input arrives, beside the
        // keyboard's, and no device comes or goes.
        host.lock().unwrap().pads[2] = Some(PadState {
            buttons: A,
            left_y: -0.5,
            ..Default::default()
        });
        hub.poll();
        let p = pad(hub);
        assert_eq!((p.buttons, p.left_x, p.left_y), (A, 1.0, -0.5));
        let device: *const Device = &hub.devices[FALLBACK];
        set_rumble(device, &[1.0, 0.0, 0.5, 0.25]);
        // Default `VICTORY_HAPTICS=0` preserves baseline 2-motor rumble (triggers zeroed).
        assert_eq!(
            host.lock().unwrap().rumble,
            [(
                2,
                Rumble {
                    low: 1.0,
                    high: 0.0,
                    left_trigger: 0.0,
                    right_trigger: 0.0,
                }
            )]
        );
        // And off again.
        host.lock().unwrap().pads[2] = None;
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(pad(hub), PadState::default());
        assert_eq!(log.lock().unwrap().len(), 1);
        assert!(!hub.state().slots[2].connected);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(
                &hub.object,
                std::ptr::null(),
                GAMEPAD_KIND,
                1,
                0,
                context,
                Some(slow),
                &mut token
            ),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(
            !unregister(&hub.object, token, 5000),
            "times out while it runs"
        );
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type RumbleFn = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<RumbleFn, _>(device, 0x50);
        // Baseline default (`VICTORY_HAPTICS=0`): advertises low/high (0x3) and zeroes triggers.
        // SAFETY: `device` is our valid Device struct.
        assert_eq!(unsafe { (*device).info.rumble_motors }, 0x3);
        rumble(device, &[0.5, 1.0, 0.4, 0.8]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        let recorded = host.lock().unwrap().rumble.clone();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].0, 2);
        assert_eq!((recorded[0].1.low_u16(), recorded[0].1.high_u16()), (32767, 65535));
        assert_eq!((recorded[0].1.left_trigger, recorded[0].1.right_trigger), (0.0, 0.0));
        assert_eq!(recorded[1], (2, Rumble::default()));

        // Opted in (`VICTORY_HAPTICS=1`): advertises all 4 motors (0xF) and forwards triggers.
        let host_haptics = Arc::new(Mutex::new(FakeHost::default()));
        let hub_haptics = Hub::new_with_options(Box::new(Fake(Arc::clone(&host_haptics))), false, true);
        host_haptics.lock().unwrap().pads[2] = Some(PadState::default());
        hub_haptics.poll();
        let hdev = &hub_haptics.devices[2] as *const Device;
        // SAFETY: `hdev` is our valid Device struct.
        assert_eq!(unsafe { (*hdev).info.rumble_motors }, 0xF);
        let hrumble = slot::<RumbleFn, _>(hdev, 0x50);
        hrumble(hdev, &[0.5, 1.0, 0.4, 0.8]);
        hrumble(hdev, std::ptr::null());
        let hrec = host_haptics.lock().unwrap().rumble.clone();
        assert_eq!(hrec.len(), 2);
        assert_eq!(hrec[0].0, 2);
        assert_eq!((hrec[0].1.low_u16(), hrec[0].1.high_u16()), (32767, 65535));
        assert_eq!((hrec[0].1.left_trigger, hrec[0].1.right_trigger), (0.4, 0.8));
        assert_eq!(hrec[1], (2, Rumble::default()));
    }

    #[test]
    fn all_four_input_switches_default_off_and_toggle_independently() {
        let none = InputSwitches::from_lookup(|_| None);
        assert_eq!(none, InputSwitches::default());
        assert!(!none.raw_mouse && !none.cursor_capture && !none.kbm_pad && !none.haptics);

        let zeros = InputSwitches::from_lookup(|_| Some(std::ffi::OsString::from("0")));
        assert_eq!(zeros, InputSwitches::default());

        let invalid = InputSwitches::from_lookup(|_| Some(std::ffi::OsString::from("true")));
        assert_eq!(invalid, InputSwitches::default());

        let only_raw = InputSwitches::from_lookup(|k| (k == "VICTORY_RAW_MOUSE").then(|| "1".into()));
        assert_eq!(
            only_raw,
            InputSwitches {
                raw_mouse: true,
                cursor_capture: false,
                kbm_pad: false,
                haptics: false,
            }
        );

        let only_cursor =
            InputSwitches::from_lookup(|k| (k == "VICTORY_CURSOR_CAPTURE").then(|| "1".into()));
        assert_eq!(
            only_cursor,
            InputSwitches {
                raw_mouse: false,
                cursor_capture: true,
                kbm_pad: false,
                haptics: false,
            }
        );

        let only_kbm = InputSwitches::from_lookup(|k| (k == "VICTORY_KBM_PAD").then(|| "1".into()));
        assert_eq!(
            only_kbm,
            InputSwitches {
                raw_mouse: false,
                cursor_capture: false,
                kbm_pad: true,
                haptics: false,
            }
        );

        let only_haptics =
            InputSwitches::from_lookup(|k| (k == "VICTORY_HAPTICS").then(|| "1".into()));
        assert_eq!(
            only_haptics,
            InputSwitches {
                raw_mouse: false,
                cursor_capture: false,
                kbm_pad: false,
                haptics: true,
            }
        );

        let all = InputSwitches::from_lookup(|_| Some("1".into()));
        assert!(all.raw_mouse && all.cursor_capture && all.kbm_pad && all.haptics);
    }

    #[cfg(windows)]
    #[test]
    fn event_driven_mouse_ring_delivers_high_rate_edges_without_poll_delay() {
        let ring = Arc::new(mouse::MouseRing::new());
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let hub = Hub::new_with_mouse_ring(
            Box::new(Fake(Arc::clone(&host))),
            true,
            false,
            Some(Arc::clone(&ring)),
        );
        hub.poll();
        assert_eq!(current(hub, MOUSE_KIND, MOUSE), Err(DEVICE_DISCONNECTED));

        // Push a baseline + fast sub-millisecond tap directly into the ring WITHOUT calling hub.poll().
        ring.set_connected(true);
        let s0 = MouseSample {
            timestamp: 50_000,
            state: MouseState {
                buttons: 0,
                position_x: 10,
                position_y: -5,
                wheel_x: 0,
                wheel_y: 0,
            },
        };
        let s1 = MouseSample {
            timestamp: 50_125,
            state: MouseState {
                buttons: 0x1, // left down
                position_x: 14,
                position_y: -7,
                wheel_x: 0,
                wheel_y: 0,
            },
        };
        let s2 = MouseSample {
            timestamp: 50_250,
            state: MouseState {
                buttons: 0x0, // left up 125 us later
                position_x: 17,
                position_y: -8,
                wheel_x: 0,
                wheel_y: 120,
            },
        };
        assert!(ring.push(s0));
        assert!(ring.push(s1));
        assert!(ring.push(s2));

        // Immediately readable via GameInput vtable with zero 8 ms poll wait.
        let latest = current(hub, MOUSE_KIND, MOUSE).unwrap();
        let read_mouse = |r: *const Reading| -> (u64, MouseState) {
            let mut st = MouseState::default();
            let get = slot::<extern "system" fn(*const Reading, *mut MouseState) -> bool, _>(r, 0x80);
            assert!(get(r, &mut st));
            let ts = slot::<extern "system" fn(*const Reading) -> u64, _>(r, 0x28)(r);
            (ts, st)
        };
        assert_eq!(read_mouse(latest), (50_250, s2.state));

        // Walk backward to prove the 125 us button-down edge was never dropped.
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let prev_fn = slot::<Walk, _>(&hub.object, 0x30);
        let mut prev_out = std::ptr::null();
        assert_eq!(
            prev_fn(&hub.object, latest, MOUSE_KIND, &hub.devices[MOUSE], &mut prev_out),
            S_OK
        );
        let r1: *const Reading = prev_out.cast();
        assert_eq!(read_mouse(r1), (50_125, s1.state));

        let mut prev0_out = std::ptr::null();
        assert_eq!(
            prev_fn(&hub.object, r1, MOUSE_KIND, &hub.devices[MOUSE], &mut prev0_out),
            S_OK
        );
        let r0: *const Reading = prev0_out.cast();
        assert_eq!(read_mouse(r0), (50_000, s0.state));

        for r in [r0, r1, latest] {
            release(r);
        }
    }
}
