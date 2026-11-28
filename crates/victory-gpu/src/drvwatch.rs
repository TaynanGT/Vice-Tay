//! Watches the Windows System event log for display driver warnings and errors.
//!
//! Some driver events precede device removal. The first matching event stops
//! indirect draw windows for the rest of the process; subsequent events are
//! still counted and logged. The watcher is off by default. Set
//! `VICTORY_DRIVER_WATCH` to `1`, `true`, or `on` before host creation to opt in.

use std::ffi::OsStr;
use std::io;
use std::ptr::{null, null_mut};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows_sys::Win32::System::EventLog::{
    EVT_HANDLE, EVT_SUBSCRIBE_CALLBACK, EVT_SUBSCRIBE_NOTIFY_ACTION, EvtClose, EvtSubscribe, EvtSubscribeActionDeliver,
    EvtSubscribeToFutureEvents,
};

const ENV_VAR: &str = "VICTORY_DRIVER_WATCH";

fn enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v == "1" || v == "true" || v == "on")
}

#[derive(Default)]
struct DriverErrors(AtomicU64);

impl DriverErrors {
    fn count(&self) -> u64 {
        self.0.load(Relaxed)
    }

    fn tripped(&self) -> bool {
        self.count() != 0
    }

    fn record(&self) -> u64 {
        // The count also serves as a latch, so never wrap back to zero.
        let mut count = self.count();
        loop {
            let next = count.saturating_add(1);
            match self.0.compare_exchange_weak(count, next, Relaxed, Relaxed) {
                Ok(_) => return next,
                Err(current) => count = current,
            }
        }
    }
}

static ERRORS: DriverErrors = DriverErrors(AtomicU64::new(0));

/// Whether a matching driver event has arrived since startup.
pub(crate) fn tripped() -> bool {
    ERRORS.tripped()
}

/// The number of matching driver events received since startup.
pub(crate) fn errors() -> u64 {
    ERRORS.count()
}

unsafe extern "system" fn on_event(
    action: EVT_SUBSCRIBE_NOTIFY_ACTION,
    _: *const core::ffi::c_void,
    _: EVT_HANDLE,
) -> u32 {
    if action == EvtSubscribeActionDeliver {
        let n = ERRORS.record();
        if n == 1 {
            tracing::error!(
                n,
                "display driver event recorded in the System log; indirect windows disabled"
            );
        } else {
            tracing::error!(n, "display driver event recorded in the System log");
        }
    }
    0
}

// Critical, error, and warning entries only. In particular, informational
// Display events such as Auto HDR notices must not trip the watcher.
const QUERY: &str = "*[System[Provider[@Name='nvlddmkm' or @Name='amdkmdag' or @Name='amdwddmg' or @Name='igfx' \
                     or @Name='igfxn' or @Name='Display'] and (Level=1 or Level=2 or Level=3)]]";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

struct Subscription(EVT_HANDLE);

impl Subscription {
    fn open(channel: &str, query: &str, callback: EVT_SUBSCRIBE_CALLBACK) -> io::Result<Self> {
        let channel = wide(channel);
        let query = wide(query);
        // SAFETY: The strings are NUL-terminated and valid for the call. The
        // callback needs no context and does not retain or close event handles.
        let handle = unsafe {
            EvtSubscribe(
                0,
                null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                null(),
                callback,
                EvtSubscribeToFutureEvents,
            )
        };
        if handle == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(Self(handle))
        }
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        // SAFETY: This owns a live subscription handle returned by EvtSubscribe.
        unsafe { EvtClose(self.0) };
    }
}

fn subscribe_if_enabled(value: Option<&OsStr>) -> Option<Subscription> {
    if !enabled(value) {
        return None;
    }
    match Subscription::open("System", QUERY, Some(on_event)) {
        Ok(subscription) => {
            tracing::info!("watching the System event log for display driver warnings and errors");
            Some(subscription)
        }
        Err(error) => {
            tracing::warn!(%error, "could not subscribe to display driver events in the System log");
            None
        }
    }
}

/// Reads the opt-in once and retains any subscription for the process lifetime.
/// Only events published after subscribing are delivered.
pub(crate) fn start() {
    static SUBSCRIPTION: OnceLock<Option<Subscription>> = OnceLock::new();
    SUBSCRIPTION.get_or_init(|| subscribe_if_enabled(std::env::var_os(ENV_VAR).as_deref()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_off() {
        for value in [None, Some(OsStr::new("0"))] {
            assert!(!enabled(value));
            assert!(subscribe_if_enabled(value).is_none());
        }
    }

    #[test]
    fn only_explicit_values_enable_the_watcher() {
        for value in ["1", "true", "on"] {
            assert!(enabled(Some(OsStr::new(value))), "{value:?}");
        }
        for value in ["", "0", "false", "off", "yes", "2", "TRUE", " on", "1 "] {
            assert!(!enabled(Some(OsStr::new(value))), "{value:?}");
        }
    }

    #[test]
    fn the_first_event_trips_the_latch_and_every_one_is_counted() {
        let state = DriverErrors::default();
        assert!(!state.tripped());
        assert_eq!(state.count(), 0);
        assert_eq!(state.record(), 1);
        assert!(state.tripped());
        assert_eq!(state.record(), 2);
        assert_eq!(state.count(), 2);
        assert!(state.tripped());
    }

    #[test]
    fn the_system_accepts_the_subscription_query() {
        unsafe extern "system" fn ignore_event(
            _: EVT_SUBSCRIBE_NOTIFY_ACTION,
            _: *const core::ffi::c_void,
            _: EVT_HANDLE,
        ) -> u32 {
            0
        }
        // Use a separate callback so real driver events cannot affect other tests.
        let _subscription =
            Subscription::open("System", QUERY, Some(ignore_event)).expect("subscribe to the System log");
    }

    #[test]
    #[ignore = "writes one entry to the Windows Application event log"]
    fn an_entry_written_reaches_a_subscription() {
        use std::sync::mpsc::{Sender, channel};
        use std::time::Duration;
        use windows_sys::Win32::System::EventLog::{
            DeregisterEventSource, EVENTLOG_INFORMATION_TYPE, RegisterEventSourceW, ReportEventW,
        };

        static DELIVERED: OnceLock<Sender<()>> = OnceLock::new();
        unsafe extern "system" fn notify(
            action: EVT_SUBSCRIBE_NOTIFY_ACTION,
            _: *const core::ffi::c_void,
            _: EVT_HANDLE,
        ) -> u32 {
            if action == EvtSubscribeActionDeliver
                && let Some(sender) = DELIVERED.get()
            {
                let _ = sender.send(());
            }
            0
        }

        let (sender, receiver) = channel();
        DELIVERED.set(sender).unwrap();
        let provider = format!("DriverWatchTest-{}", std::process::id());
        let query = format!("*[System[Provider[@Name='{provider}'] and EventID=1]]");
        let _subscription =
            Subscription::open("Application", &query, Some(notify)).expect("subscribe to the Application log");
        let name = wide(&provider);
        let text = wide("Display driver watcher test event");
        let strings = [text.as_ptr()];
        // SAFETY: All strings are NUL-terminated; there is one insertion
        // string and no optional SID or binary data.
        let result = unsafe {
            let source = RegisterEventSourceW(null(), name.as_ptr());
            assert!(
                !source.is_null(),
                "register event source: {}",
                io::Error::last_os_error()
            );
            let written = ReportEventW(
                source,
                EVENTLOG_INFORMATION_TYPE,
                0,
                1,
                null_mut(),
                1,
                0,
                strings.as_ptr(),
                null(),
            );
            let result = if written == 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(())
            };
            DeregisterEventSource(source);
            result
        };
        result.expect("write the test event");
        receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("receive the test event");
    }
}
