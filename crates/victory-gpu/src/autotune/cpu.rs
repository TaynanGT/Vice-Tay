//! Windows CPU topology for the opt-in autotune report. An L3 sharing domain
//! is a proxy for a CCD, not a measurement of physical chiplets.

use std::collections::BTreeMap;
use std::mem::{align_of, offset_of};
use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
use windows_sys::Win32::System::SystemInformation::{
    CACHE_RELATIONSHIP, GROUP_AFFINITY, GROUP_RELATIONSHIP, GetLogicalProcessorInformationEx,
    NUMA_NODE_RELATIONSHIP, PROCESSOR_GROUP_INFO, PROCESSOR_RELATIONSHIP, RelationAll, RelationCache,
    RelationGroup, RelationNumaNode, RelationNumaNodeEx, RelationProcessorCore,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CpuTopology {
    pub(super) logical_cores: u32,
    pub(super) classes: Vec<CoreClass>,
    pub(super) l3_domains: Vec<L3Domain>,
    pub(super) numa_nodes: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct CoreClass {
    /// Larger values mean higher performance on this system.
    pub(super) efficiency_class: u8,
    pub(super) physical_cores: u32,
    pub(super) logical_cores: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct L3Domain {
    pub(super) size_mib: u64,
    pub(super) logical_cores: u32,
}

/// Read one topology snapshot without changing process or thread affinity.
pub(super) fn probe() -> Option<CpuTopology> {
    let mut bytes = 0;
    // SAFETY: Windows permits a null buffer with zero size for the size query.
    let sized =
        unsafe { GetLogicalProcessorInformationEx(RelationAll, std::ptr::null_mut(), &mut bytes) };
    // SAFETY: reads the calling thread's last error if the size query failed.
    if sized != 0 || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || bytes == 0 {
        return None;
    }

    const _: () =
        assert!(align_of::<u64>() >= align_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>());
    let capacity = bytes;
    let words = (capacity as usize).div_ceil(size_of::<u64>());
    let mut buffer = Vec::<u64>::new();
    buffer.try_reserve_exact(words).ok()?;
    buffer.resize(words, 0);
    // SAFETY: the buffer is aligned and has at least `capacity` writable bytes.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(RelationAll, buffer.as_mut_ptr().cast(), &mut bytes)
    };
    if ok == 0 || bytes > capacity {
        return None;
    }
    // SAFETY: the initialized allocation contains at least the returned bytes
    // (including any unwritten padding); the slice cannot outlive the buffer.
    parse(unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), bytes as usize) })
}

const RECORD_HEADER: usize = offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous);

/// Walk by each record's Size, not by the size of the largest union member.
/// The bindings supply field offsets, but byte reads never require alignment
/// or form a reference to a C struct with a variable-length trailing array.
#[allow(non_upper_case_globals)] // Win32 relationship constants
fn parse(mut records: &[u8]) -> Option<CpuTopology> {
    const KIND: usize = offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Relationship);
    const SIZE: usize = offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size);
    let mut logical_cores = None;
    let mut classes = BTreeMap::new();
    let mut l3_domains = Vec::new();
    let mut numa_nodes = Vec::new();
    let mut core_total = 0u32;
    while !records.is_empty() {
        let kind = i32::from_le_bytes(field(records, KIND)?);
        let size = u32::from_le_bytes(field(records, SIZE)?) as usize;
        let (record, rest) = records.split_at_checked(size)?;
        let body = record.get(RECORD_HEADER..)?;
        records = rest;
        match kind {
            RelationGroup => {
                let count =
                    u16::from_le_bytes(field(body, offset_of!(GROUP_RELATIONSHIP, ActiveGroupCount))?);
                let groups = array_bytes(
                    body,
                    offset_of!(GROUP_RELATIONSHIP, GroupInfo),
                    count,
                    size_of::<PROCESSOR_GROUP_INFO>(),
                )?;
                let active = offset_of!(PROCESSOR_GROUP_INFO, ActiveProcessorCount);
                let total = groups
                    .chunks_exact(size_of::<PROCESSOR_GROUP_INFO>())
                    .map(|group| u32::from(group[active]))
                    .sum();
                if logical_cores.replace(total).is_some() {
                    return None;
                }
            }
            RelationProcessorCore => {
                let class = *body.get(offset_of!(PROCESSOR_RELATIONSHIP, EfficiencyClass))?;
                let groups = offset_of!(PROCESSOR_RELATIONSHIP, GroupCount);
                let count = u16::from_le_bytes(field(body, groups)?);
                let masks = offset_of!(PROCESSOR_RELATIONSHIP, GroupMask);
                let logical = affinity_count(body, masks, count)?;
                let class = classes.entry(class).or_insert(CoreClass {
                    efficiency_class: class,
                    physical_cores: 0,
                    logical_cores: 0,
                });
                // Each core has at least one logical processor, so bounding
                // their total also bounds both of the per-class counts.
                core_total = core_total.checked_add(logical)?;
                class.physical_cores += 1;
                class.logical_cores += logical;
            }
            RelationCache => {
                let groups = offset_of!(CACHE_RELATIONSHIP, GroupCount);
                // Older cache records had a single mask and this word was reserved.
                let count = u16::from_le_bytes(field(body, groups)?).max(1);
                let masks = offset_of!(CACHE_RELATIONSHIP, Anonymous);
                if *body.get(offset_of!(CACHE_RELATIONSHIP, Level))? == 3 {
                    let bytes =
                        u32::from_le_bytes(field(body, offset_of!(CACHE_RELATIONSHIP, CacheSize))?);
                    l3_domains.push(L3Domain {
                        size_mib: u64::from(bytes) / (1024 * 1024),
                        logical_cores: affinity_count(body, masks, count)?,
                    });
                } else {
                    array_bytes(body, masks, count, size_of::<GROUP_AFFINITY>())?;
                }
            }
            RelationNumaNode | RelationNumaNodeEx => {
                let groups = offset_of!(NUMA_NODE_RELATIONSHIP, GroupCount);
                let count = u16::from_le_bytes(field(body, groups)?);
                // Older Windows reports zero in this field but carries one mask.
                let count = if kind == RelationNumaNode { count.max(1) } else { count };
                array_bytes(
                    body,
                    offset_of!(NUMA_NODE_RELATIONSHIP, Anonymous),
                    count,
                    size_of::<GROUP_AFFINITY>(),
                )?;
                let node = offset_of!(NUMA_NODE_RELATIONSHIP, NodeNumber);
                numa_nodes.push(u32::from_le_bytes(field(body, node)?));
            }
            _ => {}
        }
    }
    // Do not present a partial set of core records as the full topology.
    let logical_cores = logical_cores.filter(|&total| total != 0 && total == core_total)?;
    numa_nodes.sort_unstable();
    numa_nodes.dedup();
    Some(CpuTopology {
        logical_cores,
        classes: classes.into_values().collect(),
        l3_domains,
        numa_nodes,
    })
}

fn field<const N: usize>(body: &[u8], at: usize) -> Option<[u8; N]> {
    body.get(at..at.checked_add(N)?)?.try_into().ok()
}

fn array_bytes(body: &[u8], at: usize, count: u16, item_size: usize) -> Option<&[u8]> {
    if count == 0 {
        return None;
    }
    body.get(at..at.checked_add(usize::from(count).checked_mul(item_size)?)?)
}

fn affinity_count(body: &[u8], at: usize, count: u16) -> Option<u32> {
    let masks = array_bytes(body, at, count, size_of::<GROUP_AFFINITY>())?;
    let mut total = 0;
    for mask in masks.chunks_exact(size_of::<GROUP_AFFINITY>()) {
        // Bit positions are local to each group, so count each mask separately.
        total += usize::from_le_bytes(field(mask, offset_of!(GROUP_AFFINITY, Mask))?).count_ones();
    }
    (total != 0).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Explicit Windows ABI fixture bytes (not serialized using parser offsets).
    fn record(kind: i32, body: &[u8]) -> Vec<u8> {
        [kind.to_le_bytes().as_slice(), &((8 + body.len()) as u32).to_le_bytes(), body].concat()
    }

    fn mask(group: u16, bits: usize) -> Vec<u8> {
        [bits.to_le_bytes().as_slice(), &group.to_le_bytes(), &[0; 6]].concat()
    }

    fn groups(active: &[usize]) -> Vec<u8> {
        let mut body = vec![0; 24];
        body[..2].copy_from_slice(&(active.len() as u16).to_le_bytes());
        body[2..4].copy_from_slice(&(active.len() as u16).to_le_bytes());
        for &bits in active {
            let mut info = [0; 40];
            info[0] = usize::BITS as u8;
            info[1] = bits.count_ones() as u8;
            body.extend(info);
            body.extend(bits.to_le_bytes());
        }
        record(RelationGroup, &body)
    }

    fn core(class: u8, group: u16, bits: usize) -> Vec<u8> {
        let mut body = vec![0; 24];
        body[1] = class;
        body[22] = 1;
        body.extend(mask(group, bits));
        record(RelationProcessorCore, &body)
    }

    fn cache(level: u8, mib: u32, groups: &[(u16, usize)], legacy: bool) -> Vec<u8> {
        let mut body = vec![0; 32];
        body[0] = level;
        body[4..8].copy_from_slice(&(mib * 1024 * 1024).to_le_bytes());
        if !legacy {
            body[30..32].copy_from_slice(&(groups.len() as u16).to_le_bytes());
        }
        for &(group, bits) in groups {
            body.extend(mask(group, bits));
        }
        record(RelationCache, &body)
    }

    fn numa(kind: i32, node: u32, groups: &[(u16, usize)]) -> Vec<u8> {
        let mut body = vec![0; 24];
        body[..4].copy_from_slice(&node.to_le_bytes());
        if kind == RelationNumaNodeEx {
            body[22..24].copy_from_slice(&(groups.len() as u16).to_le_bytes());
        }
        for &(group, bits) in groups {
            body.extend(mask(group, bits));
        }
        record(kind, &body)
    }

    #[test]
    fn autotune_cpu_topology_counts_hybrid_classes_groups_l3_and_distinct_numa() {
        const HIGH: usize = 1 << (usize::BITS - 1);
        let snapshot = [
            core(7, 0, 0b011),
            numa(RelationNumaNode, 9, &[(0, 0b011)]),
            cache(2, 2, &[(0, 0b11)], true),
            groups(&[0b111, HIGH | 0b11]),
            core(1, 1, 0b01),
            cache(3, 48, &[(0, 0b111), (1, 0b01)], false),
            record(99, &[7; 3]), // Unknown size and unaligned next record.
            numa(RelationNumaNodeEx, 2, &[(0, 0b100), (1, HIGH | 0b10)]),
            core(1, 0, 0b100),
            cache(3, 32, &[(1, HIGH | 0b10)], true),
            numa(RelationNumaNode, 9, &[(1, 0b01)]),
            core(3, 1, HIGH | 0b10),
        ]
        .concat();
        let topology = parse(&snapshot).unwrap();
        assert_eq!(topology.logical_cores, 6);
        assert_eq!(
            topology.classes,
            [
                CoreClass { efficiency_class: 1, physical_cores: 2, logical_cores: 2 },
                CoreClass { efficiency_class: 3, physical_cores: 1, logical_cores: 2 },
                CoreClass { efficiency_class: 7, physical_cores: 1, logical_cores: 2 },
            ]
        );
        assert_eq!(
            topology.l3_domains,
            [
                L3Domain { size_mib: 48, logical_cores: 4 },
                L3Domain { size_mib: 32, logical_cores: 2 },
            ]
        );
        assert_eq!(topology.numa_nodes, [2, 9]);
    }

    #[test]
    fn autotune_cpu_topology_rejects_malformed_record_boundaries() {
        let valid = [groups(&[1]), core(0, 0, 1)].concat();
        assert_eq!(parse(&valid).unwrap().logical_cores, 1);
        for end in 0..valid.len() {
            assert!(parse(&valid[..end]).is_none(), "accepted prefix {end}");
        }
        for bad_size in [0u32, 7, u32::MAX] {
            let mut malformed = valid.clone();
            malformed[4..8].copy_from_slice(&bad_size.to_le_bytes());
            assert!(parse(&malformed).is_none(), "accepted record Size={bad_size}");
        }
        let mut short_cache = cache(3, 32, &[(0, 1)], false);
        short_cache[38] = 2; // Claims two masks but only has space for one.
        let malformed = [short_cache, valid].concat();
        assert!(parse(&malformed).is_none()); // Cannot take mask bytes from the next record.
    }
}
