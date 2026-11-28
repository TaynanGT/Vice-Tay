//! `VICTORY_SHARDED_PSO=1` uses `host-pipelines.shards` instead of the
//! monolithic file. The first byte of the PSO's hex key selects one of 64
//! slots. Each file identifies its slot and format, so there is no shared
//! index to publish or recover. The driver still validates adapter and
//! driver compatibility of each native library blob. Mapped inputs always
//! keep their file handles open across replacement; saves stage at most one
//! bounded shard in memory at a time.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::Ordering;

use super::{BLOB, Blob, ID3D12Device1, Library, Mapped, create_library, mapped, one_call};

const DIR_NAME: &str = "host-pipelines.shards";
const COUNT: usize = 64;
const HEADER_LEN: usize = 48;
/// Bound the bytes read or written for one shard. If it grows past this,
/// keep its last on-disk snapshot; PSOs can still be used in this run.
const MAX_PAYLOAD: usize = 128 << 20;

pub(super) fn enabled() -> bool {
    std::env::var_os("VICTORY_SHARDED_PSO").is_some_and(|value| value == "1")
}

fn route(name: &str) -> Option<usize> {
    let [high, low, ..] = name.as_bytes() else { return None };
    let hex = |c: u8| char::from(c).to_digit(16).map(|digit| digit as usize);
    Some((hex(*high)? * 16 + hex(*low)?) % COUNT)
}

pub(super) struct Shards {
    device: ID3D12Device1,
    slots: [OnceLock<Option<Library>>; COUNT],
}

impl Shards {
    pub(super) fn new(device: ID3D12Device1) -> Self {
        Self { device, slots: [const { OnceLock::new() }; COUNT] }
    }

    pub(super) fn for_name(&self, name: &str, base: Option<&Path>) -> Option<&Library> {
        let index = route(name)?;
        self.slots[index].get_or_init(|| self.open(index, base)).as_ref()
    }

    fn opened(&self) -> impl Iterator<Item = (usize, &Library)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| Some((i, slot.get()?.as_ref()?)))
    }

    pub(super) fn mappings(&self) -> impl Iterator<Item = &Mapped> {
        self.opened().filter_map(|(_, lib)| match &lib._blob {
            Blob::Mapped(file) => Some(file),
            _ => None,
        })
    }

    fn load(&self, index: usize, path: &Path) -> io::Result<Library> {
        let blob = read(path)?;
        let bytes = payload(index, blob.bytes())?;
        // SAFETY: `Library` drops the driver library before `blob`.
        let library = unsafe { create_library(&self.device, bytes) }.map_err(|e| io::Error::other(e.to_string()))?;
        BLOB.fetch_add(blob.bytes().len() as u64, Ordering::Relaxed);
        tracing::debug!(shard = index, bytes = bytes.len(), "host pipeline shard loaded");
        Ok(Library::new(library, blob))
    }

    fn open(&self, index: usize, base: Option<&Path>) -> Option<Library> {
        if let Some(base) = base {
            match self.load(index, &path(&base.join(DIR_NAME), index)) {
                Ok(library) => return Some(library),
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => tracing::warn!(shard = index, error = %e, "host pipeline shard rejected; starting an empty one"),
            }
        }
        // Discard the rejected bytes, but don't unlink the path: a concurrent
        // process could have published a good snapshot there in the meantime.
        // The next successful save replaces only this shard.
        // SAFETY: no backing bytes are passed to the driver.
        match unsafe { create_library(&self.device, &[]) } {
            Ok(library) => Some(Library::new(library, Blob::None)),
            Err(e) => {
                tracing::warn!(shard = index, error = %e, "no host pipeline shard; pipelines are not kept");
                None
            }
        }
    }

    /// Try every dirty, opened shard, even if publishing another one fails.
    /// The shared cache save scheduler serializes flushes. Each shard's store
    /// lock keeps its native size and dirty count stable until publication.
    pub(super) fn save(&self, base: &Path) -> usize {
        let mut dirty = self.opened().filter(|(_, lib)| lib.added.load(Ordering::Relaxed) != 0).peekable();
        if dirty.peek().is_none() {
            return 0;
        }
        let dir = base.join(DIR_NAME);
        if let Err(e) = fs::create_dir_all(&dir) {
            tracing::warn!(error = %e, "host pipeline shards not saved");
            return 0;
        }
        let mut written = 0;
        for (index, lib) in dirty {
            let _all = one_call();
            let _alone = lib.saving.write().unwrap_or_else(|e| e.into_inner());
            match save_one(&dir, index, lib) {
                Ok(bytes) => {
                    written += bytes;
                    lib.added.store(0, Ordering::Relaxed);
                }
                Err(e) => tracing::warn!(shard = index, error = %e, "host pipeline shard not saved"),
            }
        }
        written
    }
}

fn path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("shard-{index:02x}.bin"))
}

fn save_one(dir: &Path, index: usize, lib: &Library) -> io::Result<usize> {
    // SAFETY: a plain query; stores are excluded until the save ends.
    let size = unsafe { lib.library.GetSerializedSize() };
    if !(1..=MAX_PAYLOAD).contains(&size) {
        return Err(io::Error::other(format!("pipeline library payload is {size} bytes, outside the shard limit")));
    }
    let label = format!("shard-{index:02x}");
    let _call = super::call("serialise", &label);
    let _writing = crate::timeline::span("pipeline: serialize a library shard to disk");
    let mut bytes = vec![0; HEADER_LEN + size];
    let (header, data) = bytes.split_at_mut(HEADER_LEN);
    // SAFETY: the buffer holds the size the library asks for.
    unsafe { lib.library.Serialize(data) }.map_err(|e| io::Error::other(e.to_string()))?;
    header[..16].copy_from_slice(&prefix(index, size));
    header[16..].copy_from_slice(blake3::hash(data).as_bytes());
    publish(&path(dir, index), &bytes)?;
    Ok(bytes.len())
}

/// Magic, layout version (including routing), shard count, shard index,
/// reserved byte, and LE payload length. A BLAKE3 digest and payload follow.
fn prefix(index: usize, len: usize) -> [u8; 16] {
    let mut header = [0; 16];
    header[..8].copy_from_slice(b"VTPSOLIB");
    header[8] = 1;
    header[9] = COUNT as u8;
    header[10] = index as u8;
    header[12..].copy_from_slice(&(len as u32).to_le_bytes());
    header
}

fn payload(index: usize, bytes: &[u8]) -> io::Result<&[u8]> {
    let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid pipeline shard length, header or checksum");
    let (header, data) = bytes.split_at_checked(HEADER_LEN).ok_or_else(invalid)?;
    if !(1..=MAX_PAYLOAD).contains(&data.len())
        || header[..16] != prefix(index, data.len())
        || header[16..] != *blake3::hash(data).as_bytes()
    {
        return Err(invalid());
    }
    Ok(data)
}

fn read(path: &Path) -> io::Result<Blob> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    let limit = (HEADER_LEN + MAX_PAYLOAD) as u64;
    if !(HEADER_LEN as u64 + 1..=limit).contains(&len) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid pipeline shard length"));
    }
    if mapped()
        && let Some(mut view) = Mapped::map(&file, len as usize, false)
    {
        // A renamed Windows mapping needs its old file to stay open.
        view._file = Some(file);
        return Ok(Blob::Mapped(view));
    }
    let mut bytes = Vec::with_capacity(len as usize);
    file.take(len + 1).read_to_end(&mut bytes)?;
    if bytes.len() != len as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "pipeline shard size changed while reading"));
    }
    Ok(Blob::Held(bytes))
}

/// Stage on the same filesystem and sync before publication. The reader
/// only knows the final filename, never a partial or abandoned temp file.
fn publish(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new().write(true).create(true).truncate(true).open(&tmp)?;
    let result = file.write_all(bytes).and_then(|()| file.sync_all());
    drop(file);
    let result = result.and_then(|()| fs::rename(&tmp, path));
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests;
