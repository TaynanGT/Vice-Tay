//! Opt-in measurements of successful host presents (VICTORY_FRAME_CSV).
//!
//! The presenting thread sends fixed-size samples to a bounded queue. Formatting,
//! file writes and periodic flushes happen on its worker. A complete marker is
//! written only after the producer is closed and the CSV is flushed and synced.
//! These are CPU observations between rendered-frame presents, not GPU timestamp
//! queries or measurements of when a frame reached the display.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Once, OnceLock, TryLockError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CAPACITY: usize = 8192;
const FLUSH_INTERVAL: Duration = Duration::from_secs(1);
const FINISH_TIMEOUT: Duration = Duration::from_secs(3);
const CLOSED: u64 = 1 << 63;
const HEADER: &str = "schema_version,process_id,stream_id,frame_index,timestamp_unix_us,elapsed_ns,interval_valid,frame_ms,draws,dispatches,prepare_ms,gpu_wait_ms,readback_ms,readback_bytes,pipelines_pending,shader_cache_misses,textures_uploaded,texture_upload_bytes,generated_frames_total,swapchain_present_count,frame_generation";

static CAPTURE: OnceLock<Option<Capture>> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);
static WORK: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Independent lifetime totals: the log meter must not reset these.
#[derive(Clone, Copy)]
pub(crate) enum Work {
    PrepareUs,
    GpuWaitUs,
    ReadbackUs,
    ReadbackBytes,
}

pub(crate) fn add(work: Work, amount: u64) {
    if ACTIVE.load(Ordering::Relaxed) {
        WORK[work as usize].fetch_add(amount, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Counters {
    pub(crate) draws: u64,
    pub(crate) dispatches: u64,
    pub(crate) pipelines_pending: u64,
    pub(crate) shader_cache_misses: u64,
    pub(crate) textures_uploaded: u64,
    pub(crate) texture_upload_bytes: u64,
    pub(crate) generated_frames_total: u64,
    pub(crate) swapchain_present_count: Option<u32>,
    pub(crate) frame_generation: bool,
}

/// One presenter lifetime. Recreating the swap chain starts a new baseline;
/// there is no artificial long frame between different presenters.
pub(crate) struct Stream {
    capture: &'static Capture,
    id: u64,
}

impl Stream {
    pub(crate) fn new() -> Option<Self> {
        let capture = CAPTURE.get_or_init(|| {
            let path = std::env::var_os("VICTORY_FRAME_CSV").filter(|p| !p.is_empty())?;
            let path = PathBuf::from(path);
            let opened = (|| {
                if !path.is_absolute() {
                    return Err(io::Error::other("VICTORY_FRAME_CSV must be an absolute new file path"));
                }
                victory_core::ensure_outside_repo(&path).map_err(io::Error::other)?;
                Capture::open(&path, CAPACITY)
            })();
            match opened {
                Ok(capture) => {
                    ACTIVE.store(true, Ordering::Relaxed);
                    tracing::info!(path = %path.display(), "per-frame CSV capture enabled");
                    Some(capture)
                }
                Err(error) => {
                    tracing::error!(%error, path = %path.display(), "per-frame CSV capture could not start");
                    None
                }
            }
        }).as_ref()?;
        Some(Self {
            capture,
            id: capture.streams.fetch_add(1, Ordering::Relaxed) + 1,
        })
    }

    pub(crate) fn present(&self, counters: Counters) {
        self.capture.submit(Sample {
            stream_id: self.id,
            elapsed_ns: self.capture.since.elapsed().as_nanos() as u64,
            counters,
            work: WORK.each_ref().map(|c| c.load(Ordering::Relaxed)),
            frame_index: 0, // Assigned while enqueueing, including refused samples.
        });
    }
}

#[derive(Clone, Copy, Default)]
struct Sample {
    stream_id: u64,
    frame_index: u64,
    elapsed_ns: u64,
    counters: Counters,
    work: [u64; 4],
}

struct Capture {
    since: Instant,
    streams: AtomicU64,
    accepted: AtomicU64,
    dropped: Arc<AtomicU64>,
    sender: Mutex<Option<SyncSender<Sample>>>,
    done: Mutex<Receiver<io::Result<()>>>,
    // One atomic combines admission closure with the active producer count.
    producers: AtomicU64,
    finish_once: Once,
}

struct InFlight<'a>(&'a AtomicU64);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}

fn completion_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".complete.json");
    name.into()
}

impl Capture {
    fn open(path: &Path, capacity: usize) -> io::Result<Self> {
        // Never replace another run's measurements or completion evidence.
        if completion_path(path).try_exists()? {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "completion file already exists",
            ));
        }
        let output = OpenOptions::new().create_new(true).write(true).open(path)?;
        let unix_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_micros() as u64;
        let since = Instant::now();
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (finished, done) = mpsc::channel();
        let dropped = Arc::new(AtomicU64::new(0));
        let worker_dropped = Arc::clone(&dropped);
        let worker_path = path.to_path_buf();
        std::thread::Builder::new()
            .name("victory-frame-csv".into())
            .spawn(move || {
                let result =
                    write_capture(output, &worker_path, unix_us, receiver, &worker_dropped);
                if let Err(error) = &result {
                    tracing::error!(%error, "per-frame CSV writer failed; capture is incomplete");
                }
                let _ = finished.send(result);
            })?;
        Ok(Self {
            since,
            streams: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            dropped,
            sender: Mutex::new(Some(sender)),
            done: Mutex::new(done),
            producers: AtomicU64::new(0),
            finish_once: Once::new(),
        })
    }

    fn enter(&self) -> Option<InFlight<'_>> {
        self.producers
            .fetch_update(Ordering::Acquire, Ordering::Relaxed, |state| {
                (state & CLOSED == 0).then_some(state + 1)
            })
            .ok()
            .map(|_| InFlight(&self.producers))
    }

    fn submit(&self, mut sample: Sample) {
        let Some(_in_flight) = self.enter() else {
            return;
        };
        // No I/O, formatting, unbounded allocation, or blocking send on this path.
        let sender = match self.sender.try_lock() {
            Ok(sender) => sender,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let Some(sender) = sender.as_ref() else {
            return;
        };
        sample.frame_index = self.accepted.fetch_add(1, Ordering::Relaxed) + 1;
        match sender.try_send(sample) {
            Ok(()) => {}
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn finish(&self) {
        self.finish_once.call_once(|| {
            let started = Instant::now();
            self.producers.fetch_or(CLOSED, Ordering::AcqRel);
            // Pre-close enqueue/drop accounting finishes before disconnection
            // can let the writer publish its completion marker.
            while self.producers.load(Ordering::Acquire) & !CLOSED != 0 {
                if started.elapsed() >= FINISH_TIMEOUT {
                    tracing::error!("per-frame CSV producers did not stop within three seconds; capture is incomplete");
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            // This lock guards only try_send, never a disk operation.
            self.sender.lock().unwrap_or_else(|e| e.into_inner()).take();
            let result = self
                .done
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .recv_timeout(FINISH_TIMEOUT.saturating_sub(started.elapsed()));
            match result {
                Ok(Ok(())) => {
                    let dropped = self.dropped.load(Ordering::Relaxed);
                    if dropped != 0 {
                        tracing::warn!(
                            dropped,
                            "per-frame CSV queue lost samples; benchmark is invalid"
                        );
                    }
                }
                Ok(Err(error)) => tracing::error!(%error, "per-frame CSV capture is incomplete"),
                Err(error) => {
                    tracing::error!(%error, "per-frame CSV did not finish within three seconds")
                }
            }
        });
    }
}

/// Must run before TerminateProcess/thread shutdown. A killed process has no
/// completion marker; ordinary process exit alone is not completion evidence.
pub(crate) fn finish() {
    if let Some(Some(capture)) = CAPTURE.get() {
        capture.finish();
    }
}

fn write_capture(
    file: File,
    path: &Path,
    unix_us: u64,
    samples: Receiver<Sample>,
    dropped: &AtomicU64,
) -> io::Result<()> {
    let mut output = BufWriter::with_capacity(64 * 1024, file);
    writeln!(output, "{HEADER}")?;
    let mut previous = HashMap::new();
    let mut written = 0_u64;
    let mut flushed = Instant::now();
    loop {
        match samples.recv_timeout(FLUSH_INTERVAL) {
            Ok(sample) => {
                write_sample(
                    &mut output,
                    unix_us,
                    sample,
                    previous.insert(sample.stream_id, sample),
                )?;
                written += 1;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if flushed.elapsed() >= FLUSH_INTERVAL {
            output.flush()?;
            flushed = Instant::now();
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    let csv_bytes = output.get_ref().metadata()?.len();
    let mut completion = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(completion_path(path))?;
    writeln!(
        completion,
        "{{\"schema_version\":1,\"status\":\"complete\",\"process_id\":{},\"written_frames\":{written},\"dropped_frames\":{},\"csv_bytes\":{csv_bytes}}}",
        std::process::id(),
        dropped.load(Ordering::Relaxed),
    )?;
    completion.sync_all()
}

fn write_sample(
    output: &mut impl Write,
    unix_us: u64,
    now: Sample,
    before: Option<Sample>,
) -> io::Result<()> {
    let c = now.counters;
    let timestamp = unix_us + now.elapsed_ns / 1000;
    write!(
        output,
        "1,{},{},{},{timestamp},{}",
        std::process::id(),
        now.stream_id,
        now.frame_index,
        now.elapsed_ns
    )?;
    if let Some(before) = before {
        let elapsed = now
            .elapsed_ns
            .checked_sub(before.elapsed_ns)
            .filter(|&ns| ns != 0)
            .ok_or_else(|| io::Error::other("frame timestamps did not increase"))?;
        let delta = |now: u64, before: u64| {
            now.checked_sub(before)
                .ok_or_else(|| io::Error::other("a cumulative frame counter went backwards"))
        };
        let work = std::array::from_fn::<_, 4, _>(|i| delta(now.work[i], before.work[i]));
        let [prepare, wait, back, bytes] = work;
        write!(
            output,
            ",1,{:.6},{},{},{:.6},{:.6},{:.6},{}",
            elapsed as f64 / 1e6,
            delta(c.draws, before.counters.draws)?,
            delta(c.dispatches, before.counters.dispatches)?,
            prepare? as f64 / 1000.0,
            wait? as f64 / 1000.0,
            back? as f64 / 1000.0,
            bytes?,
        )?;
    } else {
        // First accepted frame establishes a baseline, never a zero-duration frame.
        write!(output, ",0,,,,,,,")?;
    }
    write!(
        output,
        ",{},{},{},{},{},",
        c.pipelines_pending,
        c.shader_cache_misses,
        c.textures_uploaded,
        c.texture_upload_bytes,
        c.generated_frames_total,
    )?;
    if let Some(count) = c.swapchain_present_count {
        write!(output, "{count}")?;
    }
    writeln!(output, ",{}", u8::from(c.frame_generation))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(stream: u64, ns: u64, draws: u64, work: [u64; 4]) -> Sample {
        Sample {
            stream_id: stream,
            elapsed_ns: ns,
            counters: Counters {
                draws,
                dispatches: draws / 2,
                ..Counters::default()
            },
            work,
            ..Sample::default()
        }
    }

    #[test]
    fn capture_drains_and_keeps_first_sample_as_baseline() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("frames.csv");
        let capture = Capture::open(&path, 16).unwrap();
        capture.submit(sample(1, 1_000_000, 100, [10, 20, 30, 40]));
        capture.submit(sample(1, 34_333_333, 150, [1010, 2020, 3030, 4040]));
        capture.finish();
        capture.finish(); // Idempotent even when orderly shutdown is called twice.
        let text = std::fs::read_to_string(&path).unwrap();
        let rows: Vec<_> = text
            .lines()
            .map(|line| line.split(',').collect::<Vec<_>>())
            .collect();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.len() == 21));
        assert_eq!(rows[1][6], "0");
        assert_eq!(rows[1][7], "");
        assert_eq!(rows[2][6], "1");
        assert_eq!(rows[2][7], "33.333333");
        assert_eq!(
            &rows[2][8..14],
            &["50", "25", "1.000000", "2.000000", "3.000000", "4000"]
        );
        let completion = std::fs::read_to_string(completion_path(&path)).unwrap();
        assert!(completion.contains("\"written_frames\":2,\"dropped_frames\":0"));
        assert!(completion.contains(&format!("\"csv_bytes\":{}", text.len())));
    }

    #[test]
    fn recreating_a_presenter_does_not_measure_time_between_streams() {
        let mut output = Vec::new();
        write_sample(&mut output, 0, sample(2, 90_000_000_000, 300, [0; 4]), None).unwrap();
        let text = String::from_utf8(output).unwrap();
        let row: Vec<_> = text.trim_end().split(',').collect();
        assert_eq!(row[2], "2");
        assert_eq!(row[6], "0");
        assert_eq!(row[7], "");
    }

    #[test]
    fn overflow_is_nonblocking_and_reported_as_loss() {
        // No worker: force a full queue without relying on scheduler timing.
        let (sender, receiver) = mpsc::sync_channel(1);
        let (_done_sender, done) = mpsc::channel();
        let capture = Capture {
            since: Instant::now(),
            streams: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            dropped: Arc::new(AtomicU64::new(0)),
            sender: Mutex::new(Some(sender)),
            done: Mutex::new(done),
            producers: AtomicU64::new(0),
            finish_once: Once::new(),
        };
        capture.submit(sample(1, 1, 0, [0; 4]));
        capture.submit(sample(1, 2, 0, [0; 4]));
        assert_eq!(capture.dropped.load(Ordering::Relaxed), 1);
        assert_eq!(receiver.try_recv().unwrap().frame_index, 1);
        capture.submit(sample(1, 3, 0, [0; 4]));
        assert_eq!(receiver.try_recv().unwrap().frame_index, 3);
        // Sender-lock contention is also observable loss, never a render-thread wait.
        let _held = capture.sender.lock().unwrap();
        capture.submit(sample(1, 4, 0, [0; 4]));
        assert_eq!(capture.dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn existing_capture_is_never_truncated() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("frames.csv");
        std::fs::write(&path, "prior evidence").unwrap();
        assert!(Capture::open(&path, 1).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "prior evidence");
        std::fs::remove_file(&path).unwrap();
        std::fs::write(completion_path(&path), "prior completion").unwrap();
        assert!(Capture::open(&path, 1).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn finish_waits_for_in_flight_drop_accounting() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("frames.csv");
        let capture = Arc::new(Capture::open(&path, 8).unwrap());
        capture.submit(sample(1, 10, 0, [0; 4]));
        // Pause an admitted producer before its final drop bookkeeping.
        let in_flight = capture.enter().unwrap();
        let ending = Arc::clone(&capture);
        let finish = std::thread::spawn(move || ending.finish());
        let deadline = Instant::now() + Duration::from_secs(1);
        while capture.producers.load(Ordering::Acquire) & CLOSED == 0 {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(!completion_path(&path).exists());
        assert!(capture.enter().is_none());
        capture.dropped.fetch_add(1, Ordering::Relaxed);
        drop(in_flight);
        finish.join().unwrap();
        let complete = std::fs::read_to_string(completion_path(&path)).unwrap();
        assert!(complete.contains("\"written_frames\":1,\"dropped_frames\":1"));
    }

    #[test]
    fn worker_error_never_creates_a_completion_marker() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("frames.csv");
        let capture = Capture::open(&path, 8).unwrap();
        capture.submit(sample(1, 30, 20, [0; 4]));
        capture.submit(sample(1, 40, 10, [0; 4])); // Corrupt cumulative counter.
        capture.finish();
        assert!(!completion_path(&path).exists());
    }

    #[test]
    fn disk_write_failure_never_creates_a_completion_marker() {
        let folder = tempfile::tempdir().unwrap();
        let path = folder.path().join("frames.csv");
        File::create(&path).unwrap();
        let read_only = File::open(&path).unwrap();
        let (sender, receiver) = mpsc::sync_channel(1);
        sender.send(sample(1, 10, 0, [0; 4])).unwrap();
        drop(sender);
        let result = write_capture(read_only, &path, 0, receiver, &AtomicU64::new(0));
        assert!(result.is_err());
        assert!(!completion_path(&path).exists());
    }

    #[test]
    fn generated_count_is_not_substituted_for_rendered_intervals() {
        let before = sample(1, 10_000_000, 0, [0; 4]);
        let mut now = sample(1, 50_000_000, 0, [0; 4]);
        now.counters.generated_frames_total = 123;
        now.counters.swapchain_present_count = Some(246);
        now.counters.frame_generation = true;
        let mut output = Vec::new();
        write_sample(&mut output, 0, now, Some(before)).unwrap();
        let text = String::from_utf8(output).unwrap();
        let row: Vec<_> = text.trim_end().split(',').collect();
        assert_eq!(row[7], "40.000000");
        assert_eq!(&row[18..21], &["123", "246", "1"]);
    }
}
