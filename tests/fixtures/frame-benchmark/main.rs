use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HEADER: &str = "schema_version,process_id,stream_id,frame_index,timestamp_unix_us,elapsed_ns,interval_valid,frame_ms,draws,dispatches,prepare_ms,gpu_wait_ms,readback_ms,readback_bytes,pipelines_pending,shader_cache_misses,textures_uploaded,texture_upload_bytes,generated_frames_total,swapchain_present_count,frame_generation";

fn main() -> std::io::Result<()> {
    let csv = PathBuf::from(env::var("VICTORY_FRAME_CSV").expect("VICTORY_FRAME_CSV"));
    let mailbox = PathBuf::from(env::var("VICTORY_KEYS_FILE").expect("VICTORY_KEYS_FILE"));
    let run_dir = csv.parent().unwrap();
    let ready = run_dir.join("ready.marker");
    let mut output = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(&csv)?);
    writeln!(output, "{HEADER}")?;
    output.flush()?;
    File::create_new(ready)?;
    let start = Instant::now();
    let origin_us = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
    let mut previous_ns = None;
    let mut frames = 0u64;
    println!("synthetic runtime pid={}", std::process::id());
    eprintln!("synthetic stderr marker");
    while start.elapsed() < Duration::from_millis(1500) {
        if mailbox.exists() { let _ = fs::remove_file(&mailbox); }
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let timestamp_us = origin_us + elapsed_ns / 1000;
        frames += 1;
        if let Some(prior) = previous_ns {
            let frame_ms = (elapsed_ns - prior) as f64 / 1_000_000.0;
            writeln!(output, "1,{},{},{},{},{},1,{:.6},1,0,0.000000,0.000000,0.000000,0,0,0,0,0,0,,0",
                std::process::id(), 1, frames, timestamp_us, elapsed_ns, frame_ms)?;
        } else {
            writeln!(output, "1,{},{},{},{},{},0,,,,,,,,0,0,0,0,0,,0",
                std::process::id(), 1, frames, timestamp_us, elapsed_ns)?;
        }
        previous_ns = Some(elapsed_ns);
        output.flush()?;
        thread::sleep(Duration::from_millis(10));
    }
    drop(output);
    let file = OpenOptions::new().write(true).open(&csv)?;
    file.sync_all()?;
    let bytes = file.metadata()?.len();
    let dropped = if env::var("VICTORY_FAKE_DROP").as_deref() == Ok("1") { 1 } else { 0 };
    let completion = PathBuf::from(format!("{}.complete.json", csv.display()));
    fs::write(completion, format!("{{\"schema_version\":1,\"status\":\"complete\",\"process_id\":{},\"written_frames\":{},\"dropped_frames\":{},\"csv_bytes\":{}}}",
        std::process::id(), frames, dropped, bytes))?;
    println!("synthetic runtime completed frames={frames}");
    Ok(())
}
