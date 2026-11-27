//! Diagnostics: breadcrumbs of our own (`VICTORY_GPU_CRUMBS=1`; off by
//! default), for a host whose runtime keeps none. Under vkd3d-proton D3D12's
//! device-removed data is a stub, and a card that faults is often never
//! reported lost to us at all: the title just stops presenting (issue #221).
//!
//! Every command a host list notes ([`HostList::note_op`]) is told to the
//! card in two words of memory the PC reads, one pair per list: before the
//! command, "reached" (`MARKER_IN`: written when the card comes to it) and
//! "done" (`MARKER_OUT`: written once everything before it has finished). The
//! list's close writes "done" for all of it. When no frame has been presented
//! for 3 s (and again every 10 s while none is, [`due`]), or the device is
//! reported removed, the lists whose "reached" is
//! ahead of their "done" are logged with the commands between the two, as the
//! list noted them.
//!
//! With `VICTORY_GPU_TIMING=1` each group of lists is sent and waited for
//! alone, and the report also names the group that was sent last: one list in
//! flight, so the command named is the one the card stopped in.
//!
//! shortcut: 32-bit words, so a slot holds counts only; a slot is handed out
//! again after [`SLOTS`] other lists, long after its last list ran.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use super::*;

/// Lists that have slots at a time.
const SLOTS: u32 = 1 << 14;

/// `VICTORY_GPU_CRUMBS=1`.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_CRUMBS").is_some_and(|v| v == "1"))
}

struct Words {
    _buffer: ID3D12Resource,
    /// Two words per slot: reached, done.
    map: *mut u32,
    gpu: u64,
}

// SAFETY: a mapping of a free-threaded buffer that lives as long as the
// process; the words are read and zeroed with volatile accesses.
unsafe impl Send for Words {}
// SAFETY: as above.
unsafe impl Sync for Words {}

static WORDS: OnceLock<Option<Words>> = OnceLock::new();

fn words() -> Option<&'static Words> {
    WORDS
        .get_or_init(|| {
            let device = &host()?.device;
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain object creation and mapping.
            unsafe {
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: u64::from(SLOTS) * 8,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut made,
                    )
                    .ok()?;
                let buffer = made?;
                let mut map = std::ptr::null_mut();
                buffer.Map(0, None, Some(&mut map)).ok()?;
                let gpu = buffer.GetGPUVirtualAddress();
                tracing::info!(
                    "our own breadcrumbs: on (VICTORY_GPU_CRUMBS=1), {SLOTS} lists at a time"
                );
                watch();
                Some(Words {
                    _buffer: buffer,
                    map: map.cast(),
                    gpu,
                })
            }
        })
        .as_ref()
}

/// A list's slot and what it noted.
struct Noted {
    slot: u32,
    texts: Vec<String>,
    closed: bool,
}

/// By list (its address).
static LISTS: Mutex<Option<HashMap<usize, Noted>>> = Mutex::new(None);
static NEXT: AtomicU32 = AtomicU32::new(0);

fn read(w: &Words, at: u32) -> u32 {
    // SAFETY: inside the mapping.
    unsafe { w.map.add(at as usize).read_volatile() }
}

/// Before the command `text` describes, on `list`.
pub(crate) fn mark(list: &HostList, l: &ID3D12GraphicsCommandList, text: &str) {
    let Some(w) = words() else {
        return;
    };
    let key = std::ptr::from_ref(list) as usize;
    let (slot, index) = {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        let lists = lists.get_or_insert_with(HashMap::new);
        let noted = lists.entry(key).or_insert_with(|| {
            let slot = NEXT.fetch_add(1, Ordering::Relaxed) % SLOTS;
            // SAFETY: inside the mapping; the slot's last list ran long ago.
            unsafe {
                w.map.add(2 * slot as usize).write_volatile(0);
                w.map.add(2 * slot as usize + 1).write_volatile(0);
            }
            Noted {
                slot,
                texts: Vec::new(),
                closed: false,
            }
        });
        noted.texts.push(text.to_owned());
        (noted.slot, noted.texts.len() as u32 - 1)
    };
    // "Done" is a count plus one: 0 is a word never written.
    write(w, l, slot, index + 1, index + 1);
}

/// Before `list` is closed: all of it is done once this is.
pub(crate) fn closing(list: &HostList, l: &ID3D12GraphicsCommandList) {
    let Some(w) = words() else {
        return;
    };
    let key = std::ptr::from_ref(list) as usize;
    let (slot, all) = {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(noted) = lists.as_mut().and_then(|l| l.get_mut(&key)) else {
            return;
        };
        noted.closed = true;
        (noted.slot, noted.texts.len() as u32)
    };
    write(w, l, slot, all, all + 1);
}

/// `list` is recorded again: what it noted is gone.
pub(crate) fn reset(list: &HostList) {
    let key = std::ptr::from_ref(list) as usize;
    if let Some(lists) = LISTS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        lists.remove(&key);
    }
}

fn write(w: &Words, l: &ID3D12GraphicsCommandList, slot: u32, reached: u32, done: u32) {
    let Ok(l2) = l.cast::<ID3D12GraphicsCommandList2>() else {
        return;
    };
    let at = w.gpu + u64::from(slot) * 8;
    // "Done" first: everything before this command has finished; then
    // "reached": the card came to it.
    let params = [
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + 4,
            Value: done,
        },
        D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at,
            Value: reached,
        },
    ];
    let modes = [
        D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT,
        D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN,
    ];
    // SAFETY: an open list; our buffer, in the copy destination state it
    // was made in and never leaves.
    unsafe { l2.WriteBufferImmediate(2, params.as_ptr(), Some(modes.as_ptr())) };
}

/// Where the card is in a list, from its two words.
#[derive(Debug, PartialEq, Eq)]
enum State {
    /// The card has not come to it.
    NotBegun,
    /// All of it has finished.
    Done,
    /// Commands `from..to` were begun and not known to be finished;
    /// `known`: some "done" word was written (else not even the work before
    /// the first command is known to be finished).
    Stopped { from: u32, to: u32, known: bool },
    /// Begun, nothing in the middle (between commands, or still open).
    Between,
}

/// `reached` and `done` as the card wrote them, of a list of `all` commands.
fn state(reached: u32, done: u32, all: u32, closed: bool) -> State {
    // (Done is written as a count plus one; 0: nothing known to be done.)
    let finished = done.saturating_sub(1);
    if reached == 0 {
        State::NotBegun
    } else if closed && finished >= all {
        State::Done
    } else if reached > finished {
        State::Stopped {
            from: finished,
            to: reached.min(all),
            known: done != 0,
        }
    } else {
        State::Between
    }
}

/// The last group of lists given to the drawing queue and waited for alone
/// (`VICTORY_GPU_TIMING`), when, and the title's list in it, if one is.
static SENDING: Mutex<Option<(String, std::time::Instant, Option<usize>)>> = Mutex::new(None);

/// Before a group of lists is sent and waited for; `list`: the title's host
/// list it is, if it is one.
pub(crate) fn sending(what: String, list: Option<&HostList>) {
    let list = list.map(|l| std::ptr::from_ref(l) as usize);
    *SENDING.lock().unwrap_or_else(|e| e.into_inner()) =
        Some((what, std::time::Instant::now(), list));
}

/// Logs the lists the card was in the middle of: whether any.
pub(crate) fn report(why: &str) -> bool {
    let Some(w) = words() else {
        return false;
    };
    let sent = SENDING.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some((what, at, _)) = &sent {
        tracing::error!(
            sent_ago = format_args!("{:.1} s", at.elapsed().as_secs_f64()),
            "our own breadcrumbs: the last group sent alone: {what}"
        );
    }
    let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(lists) = lists.as_ref() else {
        return false;
    };
    // The title's list of that group, whatever its words say: a list the
    // card stopped in before its first noted command (copies, clears, a
    // list that notes nothing) is named here and nowhere else.
    if let Some((_, _, Some(key))) = &sent {
        match lists.get(key) {
            None => tracing::error!(
                list = format_args!("{key:#x}"),
                "our own breadcrumbs: the list sent alone noted no command"
            ),
            Some(noted) => {
                let (reached, finished) = (read(w, 2 * noted.slot), read(w, 2 * noted.slot + 1));
                let first: Vec<String> = noted
                    .texts
                    .iter()
                    .take(3)
                    .enumerate()
                    .map(|(i, t)| format!("#{i} {}", t.chars().take(300).collect::<String>()))
                    .collect();
                tracing::error!(
                    list = format_args!("{key:#x}"),
                    commands = noted.texts.len(),
                    reached,
                    done_word = finished,
                    closed = noted.closed,
                    state = ?state(reached, finished, noted.texts.len() as u32, noted.closed),
                    "our own breadcrumbs: the list sent alone; its first commands: {}",
                    first.join(" || ")
                );
            }
        }
    }
    let (mut done, mut not_begun, mut stopped) = (0, 0, Vec::new());
    for (&key, noted) in lists {
        let reached = read(w, 2 * noted.slot);
        let finished = read(w, 2 * noted.slot + 1);
        match state(reached, finished, noted.texts.len() as u32, noted.closed) {
            State::NotBegun => not_begun += 1,
            State::Done => done += 1,
            State::Stopped { from, to, known } => stopped.push((key, noted, from, to, known)),
            State::Between => {}
        }
    }
    tracing::error!(
        why,
        lists = lists.len(),
        done,
        not_begun,
        stopped = stopped.len(),
        "our own breadcrumbs: the lists the card was in the middle of"
    );
    for (key, noted, from, to, known) in &stopped {
        let shown: Vec<String> = (*from as usize..*to as usize)
            .take(8)
            .map(|i| {
                format!(
                    "#{i} {}",
                    noted.texts[i].chars().take(3000).collect::<String>()
                )
            })
            .collect();
        tracing::error!(
            list = format_args!("{key:#x}"),
            commands = noted.texts.len(),
            done_before = from,
            done_known = known,
            reached = to,
            closed = noted.closed,
            "our own breadcrumbs: stopped among {}",
            shown.join(" || ")
        );
    }
    !stopped.is_empty()
}

/// Whether to report again after presents have stood for `stood`, having
/// reported `said` times since they last moved: at 3 s, then every 10 s, four
/// times at most. (A slow stretch can stop presents for 3 s before the card
/// faults: one run for issue #221 reported 0.6 s before its fault, and said
/// nothing of the fault itself when it reported only once.)
fn due(stood: std::time::Duration, said: u32) -> bool {
    said < 4 && stood >= std::time::Duration::from_secs(3 + 10 * u64::from(said))
}

/// A thread that reports while presents stand ([`due`]).
fn watch() {
    let _ = std::thread::Builder::new()
        .name("victory-crumbs-watch".into())
        .spawn(|| {
            let (mut last, mut said) = ((0, std::time::Instant::now()), 0);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(500));
                let now = crate::queue::presented();
                if now != last.0 {
                    last = (now, std::time::Instant::now());
                    said = 0;
                } else if now > 0 && due(last.1.elapsed(), said) {
                    said += 1;
                    let stood = last.1.elapsed().as_secs();
                    report(&format!("no frame presented for {stood} s"));
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::{State, due, state};
    use std::time::Duration;

    /// The words as a list leaves them: not come to, stopped in a command,
    /// between commands, finished.
    #[test]
    fn the_words_say_where_the_card_is() {
        // Never come to (a fresh slot is zeroed).
        assert_eq!(state(0, 0, 5, true), State::NotBegun);
        // Came to command 0, and nothing is known to have finished: the
        // work before it (other lists on the queue) or command 0 stopped.
        assert_eq!(
            state(1, 0, 5, true),
            State::Stopped {
                from: 0,
                to: 1,
                known: false
            }
        );
        // Came to command 0 after all before it finished: command 0 stopped.
        assert_eq!(
            state(1, 1, 5, true),
            State::Stopped {
                from: 0,
                to: 1,
                known: true
            }
        );
        // Commands 0..8 finished, 8 begun (run 16 of issue #221).
        assert_eq!(
            state(9, 9, 13, true),
            State::Stopped {
                from: 8,
                to: 9,
                known: true
            }
        );
        // Several begun, the first two finished.
        assert_eq!(
            state(4, 3, 13, true),
            State::Stopped {
                from: 2,
                to: 4,
                known: true
            }
        );
        // All finished: the close wrote "done" for all of it.
        assert_eq!(state(5, 6, 5, true), State::Done);
        // An open list whose commands so far all finished.
        assert_eq!(state(3, 4, 3, false), State::Between);
        // A reached count past the noted commands is cut to them.
        assert_eq!(
            state(7, 3, 5, false),
            State::Stopped {
                from: 2,
                to: 5,
                known: true
            }
        );
    }

    /// Reports at 3 s of presents standing, then every 10 s, four at most.
    #[test]
    fn reports_while_presents_stand() {
        let s = Duration::from_secs;
        assert!(!due(s(2), 0));
        assert!(due(s(3), 0));
        assert!(!due(s(12), 1));
        assert!(due(s(13), 1));
        assert!(due(s(33), 3));
        assert!(!due(s(600), 4));
    }
}
