//! Wire-format fixtures and cold-process checks of the native pipeline cache.

use std::fs::{self, FileTimes, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12Device, ID3D12RootSignature,
};
use windows::core::s;

use super::{COUNT, HEADER_LEN, MAX_PAYLOAD, payload, prefix, route};
use crate::pipeline::library;

#[test]
fn routes_the_first_hex_byte_to_one_of_64_slots() {
    assert_eq!(COUNT, 64);
    for (name, expected) in [
        ("00", 0),
        ("3f", 63),
        ("40", 0),
        ("81ignored", 1),
        ("c2", 2),
        ("aF", 47),
        ("FF", 63),
    ] {
        assert_eq!(route(name), Some(expected), "{name:?}");
    }
    for name in ["", "f", "+1", "g0", "0g", "０f", "0é"] {
        assert_eq!(route(name), None, "{name:?}");
    }
}

#[test]
fn shard_header_checks_format_length_and_digest() {
    const PREFIX: [u8; 16] = *b"VTPSOLIB\x01\x40\x12\x00\x03\x00\x00\x00";
    const BODY: &[u8] = b"abc";
    assert_eq!(HEADER_LEN, 48);
    assert_eq!(MAX_PAYLOAD, 128 << 20);
    assert_eq!(prefix(0x12, BODY.len()), PREFIX);
    assert_eq!(prefix(0x12, 0x0123_4567)[12..], [0x67, 0x45, 0x23, 0x01]);

    let valid = [&PREFIX[..], blake3::hash(BODY).as_bytes(), BODY].concat();
    assert_eq!(payload(0x12, &valid).unwrap(), BODY);
    assert_eq!(payload(0x13, &valid).unwrap_err().kind(), ErrorKind::InvalidData);

    for (field, offset) in [
        ("magic", 7),
        ("version", 8),
        ("shard count", 9),
        ("shard index", 10),
        ("reserved byte", 11),
        ("declared length", 12),
        ("digest", 16),
        ("payload", 48),
    ] {
        let mut wrong = valid.clone();
        wrong[offset] ^= 1;
        assert_eq!(payload(0x12, &wrong).expect_err(field).kind(), ErrorKind::InvalidData);
    }

    let mut trailing = valid.clone();
    trailing.push(0);
    let mut empty_prefix = PREFIX;
    empty_prefix[12] = 0;
    let empty = [&empty_prefix[..], blake3::hash(b"").as_bytes()].concat();
    for (defect, bytes) in [
        ("short header", &valid[..47]),
        ("short payload", &valid[..50]),
        ("trailing byte", &trailing[..]),
        ("empty payload with matching header and digest", &empty[..]),
    ] {
        assert_eq!(payload(0x12, bytes).expect_err(defect).kind(), ErrorKind::InvalidData);
    }
}

const TEST_DIR: &str = "VICTORY_PSO_TEST_CHILD_DIR";
const TEST_NAMES: &str = "VICTORY_PSO_TEST_CHILD_NAMES";
const TEST_RETRY: &str = "VICTORY_PSO_TEST_CHILD_RETRY";
const RESULTS: &str = "tested-lookups.txt";

// Fixed hex keys: 81 routes to slot 01, and c2 and 42 both route to slot 02.
const CLEAN: &str = "81000000000000000000000000000001";
const DIRTY: &str = "c2000000000000000000000000000002";
const EXTRA: &str = "42000000000000000000000000000003";

#[derive(Debug)]
enum Mode {
    Default,
    Disabled,
    Sharded,
    ShardedWithBlockedSave,
}

#[test]
fn real_library_saves_and_reopens_sharded_and_monolithic_caches() {
    // Only the exact child test sets the OnceLocks for the cache and the host device.
    if let Some(dir) = std::env::var_os(TEST_DIR) {
        return exercise_cache(Path::new(&dir));
    }
    let target = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("target");
    fs::create_dir_all(&target).unwrap();
    let temp = tempfile::Builder::new().prefix("pso-shards-").tempdir_in(target).unwrap();
    let dir = temp.path();
    let mono = dir.join("host-pipelines.bin");
    let clean = shard_file(dir, 1);
    let dirty = shard_file(dir, 2);
    let probes = [CLEAN, DIRTY, EXTRA];

    run_cache(dir, Mode::Default, &[CLEAN], "M");
    assert!(!dir.join("host-pipelines.shards").try_exists().unwrap());
    let old_mono = stamp_old(&mono);
    assert!(!old_mono.is_empty());

    // A directory at the first shard's destination prevents its rename; the
    // child verifies that the second shard still publishes and retries the first.
    fs::create_dir_all(&clean).unwrap();
    run_cache(dir, Mode::ShardedWithBlockedSave, &[CLEAN, DIRTY], "MM");
    let old_clean = stamp_old(&clean);
    let old_dirty = read_shard(&dirty, 2);

    // Read both shards but add an entry only to the second one.
    run_cache(dir, Mode::Sharded, &probes, "HHM");
    assert_untouched(&clean, &old_clean);
    let mut corrupted = read_shard(&dirty, 2);
    assert!(corrupted != old_dirty, "adding a PSO did not change the dirty shard");

    // Leave a valid abandoned stage file. Even when the canonical file is
    // invalid, it must be rebuilt instead of being recovered from that stage.
    fs::write(dirty.with_extension("tmp-0"), &corrupted).unwrap();
    corrupted[16] ^= 1; // Wrong digest, but the native blob itself is still valid.
    // Conversely, this envelope is valid but its native blob must be refused.
    let bad_native = b"not a pipeline library".repeat(16);
    let header = prefix(2, bad_native.len());
    let digest = blake3::hash(&bad_native);
    let bad_library = [&header[..], digest.as_bytes(), bad_native.as_slice()].concat();
    assert_eq!(payload(2, &bad_library).unwrap(), bad_native.as_slice());
    for invalid in [&corrupted[..], &corrupted[..47], &bad_library[..]] {
        fs::write(&dirty, invalid).unwrap();
        run_cache(dir, Mode::Sharded, &probes, "HMM");
        assert_untouched(&clean, &old_clean);
        let rebuilt = stamp_old(&dirty);
        run_cache(dir, Mode::Sharded, &probes, "HHH");
        assert_untouched(&clean, &old_clean);
        assert_untouched(&dirty, &rebuilt);
    }

    // Both opt-out settings still share the same monolithic cache. Neither
    // backend can see PSOs that were stored only by the other backend.
    assert_untouched(&mono, &old_mono);
    assert!(!dir.join("host-pipelines.next").try_exists().unwrap());
    let rebuilt_dirty = read_shard(&dirty, 2);
    run_cache(dir, Mode::Disabled, &[CLEAN, DIRTY], "HM");
    run_cache(dir, Mode::Default, &probes, "HHM");
    assert_untouched(&clean, &old_clean);
    assert_untouched(&dirty, &rebuilt_dirty);
}

fn run_cache(dir: &Path, mode: Mode, names: &[&str], expected: &str) {
    // libtest can exit successfully with zero matching tests. Start with an
    // empty report, and require the selected test to write its actual outcomes.
    let results = dir.join(RESULTS);
    fs::write(&results, b"").unwrap();
    let (_, module) = module_path!().split_once("::").unwrap();
    let test = format!("{module}::real_library_saves_and_reopens_sharded_and_monolithic_caches");
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", test.as_str(), "--nocapture"])
        .env(TEST_DIR, dir)
        .env(TEST_NAMES, names.join(","))
        .env_remove(TEST_RETRY);
    match mode {
        Mode::Default => child.env_remove("VICTORY_SHARDED_PSO"),
        Mode::Disabled => child.env("VICTORY_SHARDED_PSO", "0"),
        Mode::Sharded | Mode::ShardedWithBlockedSave => child.env("VICTORY_SHARDED_PSO", "1"),
    };
    if matches!(mode, Mode::ShardedWithBlockedSave) {
        child.env(TEST_RETRY, "1");
    }
    // Inherit the child's diagnostics without pipe capture in the Windows sandbox.
    let status = child.status().expect("start the isolated pipeline cache test");
    assert!(status.success(), "{mode:?} for {names:?}: child exited with {status}");
    assert_eq!(fs::read_to_string(results).unwrap(), expected, "{mode:?} for {names:?}");
}

fn exercise_cache(dir: &Path) {
    library::use_pipeline_cache(dir.to_owned());
    let host = crate::host::host().expect("pipeline cache test requires a host D3D12 device");
    #[cfg(feature = "diagnostics")]
    if crate::switch::on("VICTORY_D3D12_WARP") {
        let warp = crate::host::warp_adapter().expect("get the requested WARP adapter");
        // SAFETY: plain queries on a live device and adapter.
        let (actual, wanted) = unsafe {
            (host.device.GetAdapterLuid(), warp.GetDesc().expect("query WARP adapter").AdapterLuid)
        };
        assert_eq!(
            (actual.HighPart, actual.LowPart),
            (wanted.HighPart, wanted.LowPart),
            "VICTORY_D3D12_WARP requested, but the host used another adapter"
        );
    }
    let names = std::env::var(TEST_NAMES).unwrap();
    let outcomes: String = names.split(',').map(|name| probe(&host.device, name)).collect();
    library::save(true);

    if std::env::var_os(TEST_RETRY).is_some() {
        let blocked = shard_file(dir, 1);
        let healthy = shard_file(dir, 2);
        assert!(blocked.is_dir(), "first save did not encounter the blocked destination");
        let written = stamp_old(&healthy);
        fs::remove_dir(&blocked).unwrap();
        library::save(true);
        // The failed shard stayed dirty, and the published one did not.
        assert!(blocked.is_file(), "retry did not publish the failed shard");
        assert_untouched(&healthy, &written);
    }

    // SAFETY: plain query on the host device.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("pipeline cache test lost the device");
    fs::write(dir.join(RESULTS), outcomes).unwrap();
}

fn probe(device: &ID3D12Device, key: &str) -> char {
    let threads = match key {
        CLEAN => 1,
        DIRTY => 2,
        EXTRA => 3,
        _ => panic!("unknown fixture PSO: {key}"),
    };
    let source = format!("[RootSignature(\"\")] [numthreads({threads}, 1, 1)] void main() {{}}");
    let code = crate::host::compile(&source, s!("main"), s!("cs_5_1")).expect("compile fixture CS");
    // SAFETY: compiled shader with an embedded empty root signature.
    let root: ID3D12RootSignature = unsafe { device.CreateRootSignature(0, &code) }.unwrap();
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: borrowed pointer; the description does not release it and `root` outlives it.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        },
        ..Default::default()
    };
    let mut tried_library = false;
    let mut created = false;
    let _pso = library::pipeline(
        Some(key),
        |lib, name| {
            tried_library = true;
            // SAFETY: library, name, and the description's data are live.
            unsafe { lib.LoadComputePipeline(name, &desc) }
        },
        || {
            created = true;
            // SAFETY: the description points at live bytecode and root signature.
            unsafe { device.CreateComputePipelineState(&desc) }
        },
    )
    .unwrap_or_else(|error| panic!("fixture PSO {key}: {error}"));
    assert!(tried_library, "no ID3D12PipelineLibrary1 available for {key}");
    if created { 'M' } else { 'H' }
}

fn shard_file(dir: &Path, index: usize) -> PathBuf {
    dir.join("host-pipelines.shards").join(format!("shard-{index:02x}.bin"))
}

fn read_shard(path: &Path, index: usize) -> Vec<u8> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    payload(index, &bytes).unwrap_or_else(|error| panic!("validate {path:?}: {error}"));
    bytes
}

fn old_time() -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(1_000_000_000)
}

fn stamp_old(path: &Path) -> Vec<u8> {
    OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap_or_else(|error| panic!("open {path:?} to set its timestamp: {error}"))
        .set_times(FileTimes::new().set_modified(old_time()))
        .unwrap();
    assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), old_time());
    fs::read(path).unwrap()
}

fn assert_untouched(path: &Path, bytes: &[u8]) {
    assert!(fs::read(path).unwrap().as_slice() == bytes, "bytes changed for {path:?}");
    assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), old_time(), "rewrote {path:?}");
}
