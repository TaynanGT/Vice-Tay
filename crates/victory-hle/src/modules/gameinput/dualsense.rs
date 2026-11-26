//! PlayStation 5 pads (DualSense, DualSense Edge) read straight from HID:
//! they are not XInput devices, so without a remapper (Steam Input,
//! DS4Windows) XInput never lists them. Reports can be USB or Bluetooth. A scan
//! thread looks for new pads every [`SCAN`]; each pad has a reader thread
//! that keeps its latest state until the pad goes away. By default XInput
//! has priority over HID, since a remapper can present the same pad twice.

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

#[cfg(windows)]
const SONY: u16 = 0x054C;
/// DualSense, DualSense Edge.
#[cfg(windows)]
const PRODUCTS: [u16; 2] = [0x0CE6, 0x0DF2];

/// The D-pad hat (low nibble of the first button byte): 0 up, clockwise in
/// eighths, 8 and above centred.
fn hat(value: u8) -> u32 {
    match value & 0xF {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => RIGHT | DOWN,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => LEFT | UP,
        _ => 0,
    }
}

/// Face buttons in the first button byte, the rest in the second and third
/// (the PS button and the mute button have no console pad button).
const BUTTONS: [(usize, u8, u32); 11] = [
    (0, 0x10, X), // square
    (0, 0x20, A), // cross
    (0, 0x40, B), // circle
    (0, 0x80, Y), // triangle
    (1, 0x01, LEFT_SHOULDER),
    (1, 0x02, RIGHT_SHOULDER),
    (1, 0x10, VIEW), // create
    (1, 0x20, MENU), // options
    (1, 0x40, LEFT_STICK),
    (1, 0x80, RIGHT_STICK),
    (2, 0x02, VIEW), // touchpad click
];

/// The pad in one input report, `None` for a report of another kind.
/// `bluetooth`: the pad is connected over Bluetooth, where the full report
/// is 0x31 (one byte further in) and 0x01 is the short report the pad sends
/// before it was asked for the full one.
pub(super) fn parse(report: &[u8], bluetooth: bool) -> Option<PadState> {
    // Offsets of the sticks (LX LY RX RY), the triggers (L2 R2) and the
    // three button bytes.
    let (sticks, triggers, buttons) = match (report.first()?, bluetooth) {
        (0x01, false) => (1, 5, 8),
        (0x31, true) => (2, 6, 9),
        (0x01, true) => (1, 8, 5),
        _ => return None,
    };
    if report.len() < sticks + 4 || report.len() < triggers + 2 || report.len() < buttons + 3 {
        return None;
    }
    let bytes = &report[buttons..buttons + 3];
    // 0 to 255 with 128 centred; down is positive on the pad, up here.
    let axis = |v: u8| ((f32::from(v) - 127.5) / 127.5).clamp(-1.0, 1.0);
    Some(PadState {
        buttons: BUTTONS
            .iter()
            .filter(|&&(byte, mask, _)| bytes[byte] & mask != 0)
            .fold(hat(bytes[0]), |all, &(_, _, bit)| all | bit),
        left_trigger: f32::from(report[triggers]) / 255.0,
        right_trigger: f32::from(report[triggers + 1]) / 255.0,
        left_x: axis(report[sticks]),
        left_y: -axis(report[sticks + 1]),
        right_x: axis(report[sticks + 2]),
        right_y: -axis(report[sticks + 3]),
    })
}

#[cfg(windows)]
pub(super) use host::pads;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    Auto,
    Both,
}

#[cfg(windows)]
pub(super) fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> =
        std::sync::LazyLock::new(|| match std::env::var("VICTORY_DUALSENSE").as_deref() {
            Ok("0") => Mode::Off,
            Ok("1") => Mode::Both,
            _ => Mode::Auto,
        });
    *MODE
}

/// XInput does not identify the corresponding physical HID device. Prefer
/// it in automatic mode. If both sources are requested, retain the slot of
/// every connected HID pad unless that slot has been claimed by XInput.
pub(super) fn slots(
    mode: Mode,
    mut xinput: [Option<PadState>; super::PADS],
    assigned: &mut [Option<u64>; super::PADS],
    hid: impl FnOnce() -> Vec<(u64, PadState)>,
) -> [Option<PadState>; super::PADS] {
    if mode == Mode::Off || mode == Mode::Auto && xinput.iter().any(Option::is_some) {
        assigned.fill(None);
        return xinput;
    }
    let pads = hid();
    for (slot, owner) in xinput.iter_mut().zip(assigned.iter_mut()) {
        if slot.is_none()
            && let Some((_, state)) = pads.iter().find(|(id, _)| Some(*id) == *owner)
        {
            *slot = Some(*state);
        } else {
            *owner = None;
        }
    }
    for (id, state) in pads {
        if assigned.contains(&Some(id)) {
            continue;
        }
        let Some(slot) = xinput.iter().position(Option::is_none) else {
            break;
        };
        xinput[slot] = Some(state);
        assigned[slot] = Some(id);
    }
    xinput
}

#[cfg(windows)]
mod host {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{LazyLock, Mutex, PoisonError};
    use std::time::Duration;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA,
        SP_DEVICE_INTERFACE_DETAIL_DATA_W, SetupDiDestroyDeviceInfoList,
        SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile,
    };

    use super::{PRODUCTS, PadState, SONY, parse};

    /// How often to look for newly connected pads.
    const SCAN: Duration = Duration::from_secs(2);

    struct Pad {
        id: u64,
        path: Vec<u16>,
        /// `None` until the first report.
        state: Option<PadState>,
    }

    /// Pads in the order they were found.
    static PADS: Mutex<Vec<Pad>> = Mutex::new(Vec::new());

    fn lock() -> std::sync::MutexGuard<'static, Vec<Pad>> {
        PADS.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The connected pads' latest states, oldest connection first; the first
    /// call starts looking for them.
    pub(in super::super) fn pads() -> Vec<(u64, PadState)> {
        static STARTED: LazyLock<()> = LazyLock::new(|| {
            std::thread::spawn(|| {
                loop {
                    scan();
                    std::thread::sleep(SCAN);
                }
            });
        });
        LazyLock::force(&STARTED);
        lock()
            .iter()
            .filter_map(|pad| Some((pad.id, pad.state?)))
            .collect()
    }

    /// Opens the pads not open yet, each with a reader thread.
    fn scan() {
        for path in hid_paths() {
            // Cheap filter before opening anything: Sony's vendor id is in
            // the path, as `vid_054c` (USB) or `vid&0002054c` (Bluetooth).
            let lower = String::from_utf16_lossy(&path).to_ascii_lowercase();
            if !lower.contains("054c") {
                continue;
            }
            if lock().iter().any(|pad| pad.path == path) {
                continue;
            }
            let Some(handle) = open(&path) else {
                continue;
            };
            let bluetooth = lower.contains("{00001124-");
            tracing::info!(path = %lower, bluetooth, "DualSense connected");
            // Asking for the calibration feature report switches a pad on
            // Bluetooth to full reports (0x31); over USB it changes nothing.
            let mut feature = [0u8; 41];
            feature[0] = 0x05;
            // SAFETY: the buffer is as long as we say.
            unsafe { HidD_GetFeature(handle, feature.as_mut_ptr().cast(), feature.len() as u32) };
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            lock().push(Pad {
                id,
                path,
                state: None,
            });
            let raw = handle as usize;
            if let Err(error) = std::thread::Builder::new()
                .name("DualSense reader".into())
                .spawn(move || read(raw as HANDLE, id, bluetooth))
            {
                remove(handle, id);
                tracing::warn!(%error, "could not start a DualSense reader");
            }
        }
    }

    /// Reads reports until the pad goes away, then forgets it.
    fn read(handle: HANDLE, id: u64, bluetooth: bool) {
        let mut report = [0u8; 128];
        loop {
            let mut got = 0;
            // SAFETY: a synchronous read into a buffer as long as we say.
            let ok = unsafe {
                ReadFile(
                    handle,
                    report.as_mut_ptr(),
                    report.len() as u32,
                    &mut got,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 || got == 0 {
                break;
            }
            if let Some(state) = parse(&report[..got as usize], bluetooth) {
                let mut pads = lock();
                if let Some(pad) = pads.iter_mut().find(|pad| pad.id == id) {
                    pad.state = Some(state);
                }
            }
        }
        tracing::info!("DualSense disconnected");
        remove(handle, id);
    }

    fn remove(handle: HANDLE, id: u64) {
        // SAFETY: ours, and nothing uses it any more.
        unsafe { CloseHandle(handle) };
        lock().retain(|pad| pad.id != id);
    }

    /// The pad at `path` opened for reading, `None` when it is something else
    /// or cannot be opened.
    fn open(path: &[u16]) -> Option<HANDLE> {
        let mut name = path.to_vec();
        name.push(0);
        // SAFETY: a terminated path; no security attributes or template.
        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                0,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut attributes = HIDD_ATTRIBUTES {
            Size: size_of::<HIDD_ATTRIBUTES>() as u32,
            ..Default::default()
        };
        // SAFETY: an open device and room for its attributes.
        let ours = unsafe { HidD_GetAttributes(handle, &mut attributes) }
            && attributes.VendorID == SONY
            && PRODUCTS.contains(&attributes.ProductID);
        if !ours {
            // SAFETY: just opened.
            unsafe { CloseHandle(handle) };
            return None;
        }
        Some(handle)
    }

    /// The device paths of the HID devices present, without terminators.
    fn hid_paths() -> Vec<Vec<u16>> {
        let mut paths = Vec::new();
        // SAFETY: plain queries into buffers we own; the list is destroyed
        // before returning.
        unsafe {
            let mut guid = std::mem::zeroed();
            HidD_GetHidGuid(&mut guid);
            let set = SetupDiGetClassDevsW(
                &guid,
                std::ptr::null(),
                std::ptr::null_mut(),
                DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
            );
            if set == INVALID_HANDLE_VALUE as isize {
                return paths;
            }
            for index in 0.. {
                let mut interface = SP_DEVICE_INTERFACE_DATA {
                    cbSize: size_of::<SP_DEVICE_INTERFACE_DATA>() as u32,
                    ..std::mem::zeroed()
                };
                if SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &guid, index, &mut interface)
                    == 0
                {
                    break;
                }
                let mut size = 0;
                SetupDiGetDeviceInterfaceDetailW(
                    set,
                    &interface,
                    std::ptr::null_mut(),
                    0,
                    &mut size,
                    std::ptr::null_mut(),
                );
                if size == 0 {
                    continue;
                }
                // u64s keep the detail's alignment.
                let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
                let detail = buffer
                    .as_mut_ptr()
                    .cast::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>();
                (*detail).cbSize = size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
                if SetupDiGetDeviceInterfaceDetailW(
                    set,
                    &interface,
                    detail,
                    size,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ) == 0
                {
                    continue;
                }
                let start = std::ptr::addr_of!((*detail).DevicePath).cast::<u16>();
                let room = (size as usize
                    - std::mem::offset_of!(SP_DEVICE_INTERFACE_DETAIL_DATA_W, DevicePath))
                    / 2;
                let wide = std::slice::from_raw_parts(start, room);
                let end = wide.iter().position(|&c| c == 0).unwrap_or(room);
                paths.push(wide[..end].to_vec());
            }
            SetupDiDestroyDeviceInfoList(set);
        }
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_input_prefers_the_remapper_in_any_slot_and_falls_back_to_hid() {
        let physical = PadState {
            buttons: A,
            ..PadState::default()
        };
        let remapped = PadState {
            buttons: Y,
            ..PadState::default()
        };
        let mut assigned = [None; super::super::PADS];
        let none = [None; super::super::PADS];
        let xinput = [None, None, Some(remapped), None];

        assert_eq!(
            slots(Mode::Auto, none, &mut assigned, || vec![(1, physical)])[0],
            Some(physical)
        );
        assert_eq!(
            slots(Mode::Auto, xinput, &mut assigned, || panic!(
                "HID not needed"
            )),
            xinput
        );
        assert_eq!(
            slots(Mode::Off, none, &mut assigned, || panic!("HID disabled")),
            none
        );
        assert_eq!(
            slots(Mode::Auto, none, &mut assigned, || vec![(2, physical)])[0],
            Some(physical)
        );
    }

    #[test]
    fn a_surviving_hid_pad_keeps_its_slot_through_disconnect_and_reconnect() {
        let a = PadState {
            buttons: A,
            ..PadState::default()
        };
        let b = PadState {
            buttons: B,
            ..PadState::default()
        };
        let x = PadState {
            buttons: X,
            ..PadState::default()
        };
        let xinput = [None, Some(x), None, None];
        let mut assigned = [None; super::super::PADS];
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(7, a), (8, b)]),
            [Some(a), Some(x), Some(b), None]
        );
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(8, b)]),
            [None, Some(x), Some(b), None]
        );
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(8, b), (9, a)]),
            [Some(a), Some(x), Some(b), None]
        );
    }

    #[test]
    fn a_usb_report_reads_as_a_pad() {
        let mut report = [0u8; 64];
        report[0] = 0x01;
        report[1..5].copy_from_slice(&[0, 255, 128, 128]); // left stick: left, down
        report[5] = 255; // L2
        report[8] = 0x20 | 2; // cross, D-pad right
        report[9] = 0x20 | 0x01; // options, L1
        let pad = parse(&report, false).unwrap();
        assert_eq!(pad.buttons, A | RIGHT | MENU | LEFT_SHOULDER);
        assert_eq!(
            (pad.left_x, pad.left_y, pad.left_trigger),
            (-1.0, -1.0, 1.0)
        );
        assert!(pad.right_x.abs() < 0.01 && pad.right_y.abs() < 0.01);
    }

    #[test]
    fn bluetooth_reports_read_one_byte_further_in() {
        let mut report = [0u8; 78];
        report[0] = 0x31;
        report[2..6].copy_from_slice(&[128, 0, 255, 128]); // left up, right right
        report[7] = 255; // R2
        report[9] = 0x80 | 8; // triangle, D-pad centred
        let pad = parse(&report, true).unwrap();
        assert_eq!(pad.buttons, Y);
        assert_eq!(
            (pad.left_y, pad.right_x, pad.right_trigger),
            (1.0, 1.0, 1.0)
        );
        // the same bytes over USB are not a pad report
        assert_eq!(parse(&report, false), None);
    }

    #[test]
    fn the_short_bluetooth_report_reads_too() {
        let mut report = [0u8; 78];
        report[0] = 0x01;
        report[1..5].copy_from_slice(&[128; 4]);
        report[5] = 0x40 | 4; // circle, D-pad down
        report[7] = 0x02; // touchpad click
        report[8] = 128; // L2 halfway
        let pad = parse(&report, true).unwrap();
        assert_eq!(pad.buttons, B | DOWN | VIEW);
        assert!((pad.left_trigger - 0.5).abs() < 0.01);
    }
}
