//! Static samplers given to the driver in one order in every process.
//!
//! The NVIDIA driver keeps static samplers in a table of its own, in the
//! order in which a process first makes each, and a stored pipeline holds its
//! samplers' places in that table. The title's pipelines are made by several
//! threads in the order the title asks for them, so the places differ from
//! run to run, and a pipeline restored from the library could sample with
//! another sampler's filter or address mode ([`super::library::allows_static_samplers`]).
//!
//! So every distinct static sampler is written to a list beside the library
//! ([`FILE`], one line each, in the order first met), and each process makes
//! the listed ones in that order as soon as its device exists ([`prime`]),
//! in root signatures of its own that live as long as the process. A
//! sampler's place is then its line in the list, whoever asks first. One met
//! later ([`know`]) is added at the end; pipelines with it are not taken from
//! or stored in the library in that process ([`settled`]), as its place may
//! follow samplers the next process makes later.
//!
//! From trustno1-coder/VIctory-NVIDIA@64d4931 (`pipeline/samplers.rs`,
//! GPL-3.0). `VICTORY_SAMPLER_ORDER=0`: each process's own order, as before.

use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use victory_shader::emit::StaticSampler;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

/// The list, beside the pipeline library.
const FILE: &str = "static-samplers.txt";

/// A sampler as the host is given it: filter, the three address modes, the
/// LOD bias (bits), the anisotropy, the comparison, the border, the least
/// and the greatest LOD (bits).
type Words = [u32; 10];

/// Those [`prime`] made from the list: their places are the list's.
static PRIMED: OnceLock<HashSet<Words>> = OnceLock::new();

/// Those made so far in this process.
static SEEN: RwLock<Option<HashSet<Words>>> = RwLock::new(None);

struct Made {
    known: HashSet<Words>,
    /// The root signatures that hold them: kept, so that the driver keeps
    /// their places.
    kept: Vec<ID3D12RootSignature>,
}

// SAFETY: D3D12 objects are free-threaded; these are only held.
unsafe impl Send for Made {}

static MADE: Mutex<Option<Made>> = Mutex::new(None);

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SAMPLER_ORDER").is_none_or(|v| v != "0"))
}

/// Makes the samplers of the list, in its order: called when the device is
/// made, before anything else has a static sampler.
pub(crate) fn prime(device: &ID3D12Device) {
    if !on() {
        return;
    }
    let made = sync(device, &[]);
    let primed = SEEN.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default();
    let _ = PRIMED.set(primed);
    tracing::info!(samplers = made, "static samplers made in the order of the cache's list (VICTORY_SAMPLER_ORDER)");
}

/// Whether all of `samplers` hold the places the list gives them in this
/// process: made by [`prime`].
pub(super) fn settled(samplers: &[StaticSampler], aniso_point_mip_supported: bool) -> bool {
    let Some(primed) = PRIMED.get() else { return false };
    samplers.iter().all(|s| primed.contains(&words(s, aniso_point_mip_supported)))
}

/// A pipeline with these static samplers is about to be made: those not yet
/// in the list are added to it and made.
pub(super) fn know(device: &ID3D12Device, samplers: &[StaticSampler], aniso_point_mip_supported: bool) {
    if samplers.is_empty() || !on() {
        return;
    }
    let words: Vec<Words> = samplers.iter().map(|s| words(s, aniso_point_mip_supported)).collect();
    {
        let seen = SEEN.read().unwrap_or_else(|e| e.into_inner());
        if seen.as_ref().is_some_and(|seen| words.iter().all(|w| seen.contains(w))) {
            return;
        }
    }
    sync(device, &words);
}

fn words(s: &StaticSampler, aniso_point_mip_supported: bool) -> Words {
    let [u, v, w] = s.address;
    [
        s.filter_for_host(aniso_point_mip_supported),
        u,
        v,
        w,
        s.mip_lod_bias.to_bits(),
        s.max_anisotropy,
        s.comparison,
        s.border,
        s.min_lod.to_bits(),
        s.max_lod.to_bits(),
    ]
}

fn line(w: &Words) -> String {
    w.iter().map(|v| format!("{v:x}")).collect::<Vec<_>>().join(" ")
}

fn parse(line: &str) -> Option<Words> {
    let mut words = [0u32; 10];
    let mut given = line.split_ascii_whitespace();
    for w in &mut words {
        *w = u32::from_str_radix(given.next()?, 16).ok()?;
    }
    given.next().is_none().then_some(words)
}

/// The list's file, open for this process alone (others wait in their own
/// call of this), or `None` without a folder or after ten seconds.
fn open_alone() -> Option<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt;
    let dir = super::library::folder()?;
    let _ = std::fs::create_dir_all(&dir);
    let began = Instant::now();
    loop {
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(dir.join(FILE))
        {
            Ok(file) => return Some(file),
            Err(_) if began.elapsed() < Duration::from_secs(10) => std::thread::sleep(Duration::from_millis(5)),
            Err(e) => {
                tracing::warn!(error = %e, "the list of static samplers could not be opened; their order is this process's own");
                return None;
            }
        }
    }
}

/// Makes what the list holds and this process has not made yet, in the
/// list's order, then `mine` where new (added to the list). How many this
/// process has made in all.
fn sync(device: &ID3D12Device, mine: &[Words]) -> usize {
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let made = made.get_or_insert_with(|| Made { known: HashSet::new(), kept: Vec::new() });
    let mut file = open_alone();
    let mut listed = String::new();
    if let Some(file) = &mut file {
        let _ = file.read_to_string(&mut listed);
    }
    let mut new = Vec::new();
    for w in listed.lines().filter_map(parse) {
        if made.known.insert(w) {
            new.push(w);
        }
    }
    let mut added = String::new();
    for w in mine {
        if made.known.insert(*w) {
            new.push(*w);
            added.push_str(&line(w));
            added.push('\n');
        }
    }
    // A refused sampler shifts the places of every one after it: none of them
    // count as made, so pipelines using them stay out of the library.
    if let Some(at) = new.iter().position(|w| match root_with(device, w) {
        Ok(root) => {
            made.kept.push(root);
            false
        }
        Err(e) => {
            tracing::warn!(error = %e, sampler = line(w), "a static sampler of the list was refused");
            true
        }
    }) {
        new.truncate(at);
    }
    if let Some(file) = &mut file
        && !added.is_empty()
    {
        if !listed.is_empty() && !listed.ends_with('\n') {
            added.insert(0, '\n');
        }
        if let Err(e) = file.seek(SeekFrom::End(0)).and_then(|_| file.write_all(added.as_bytes())) {
            tracing::warn!(error = %e, "the list of static samplers was not written");
        }
    }
    drop(file);
    let mut seen = SEEN.write().unwrap_or_else(|e| e.into_inner());
    seen.get_or_insert_default().extend(new);
    made.known.len()
}

/// A root signature with the one static sampler `w`.
fn root_with(device: &ID3D12Device, w: &Words) -> windows::core::Result<ID3D12RootSignature> {
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER(w[0] as i32),
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(w[1] as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(w[2] as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(w[3] as i32),
        MipLODBias: f32::from_bits(w[4]),
        MaxAnisotropy: w[5],
        ComparisonFunc: D3D12_COMPARISON_FUNC(w[6] as i32),
        BorderColor: D3D12_STATIC_BORDER_COLOR(w[7] as i32),
        MinLOD: f32::from_bits(w[8]),
        MaxLOD: f32::from_bits(w[9]),
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 { NumStaticSamplers: 1, pStaticSamplers: &sampler, ..Default::default() },
        },
    };
    // SAFETY: the description points at `sampler`, which lives through the
    // call; the blob is read in full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_reads_back_as_its_sampler() {
        let w: Words = [0x15, 3, 3, 3, 0, 1, 1, 0, 0, 0x4140_0000];
        assert_eq!(parse(&line(&w)), Some(w));
        assert_eq!(parse("15 3 3"), None);
        assert_eq!(parse("15 3 3 3 0 1 1 0 0 41400000 7"), None);
        assert_eq!(parse(""), None);
    }
}
