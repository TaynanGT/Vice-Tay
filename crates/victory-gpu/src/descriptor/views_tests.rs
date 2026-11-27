use super::*;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

/// Reference single-threaded `BTreeMap`-backed oracle matching the exact
/// semantics of `ViewRegistryCore` (lossless shape interning, table entries,
/// resource invalidation, and FIFO ownerless eviction).
#[derive(Default)]
struct BTreeViewsOracle {
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<Shape, u32>,
    ownerless: std::collections::VecDeque<(u32, u32)>,
}

impl BTreeViewsOracle {
    fn shape(&mut self, shape: Shape, max_shapes: usize) -> Option<u32> {
        if let Some(&at) = self.shape_index.get(&shape) {
            return Some(at);
        }
        if self.shapes.len() >= max_shapes {
            return None;
        }
        let at = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_index.insert(shape, at);
        Some(at)
    }

    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
        let &(unordered, resource, rest) =
            self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&rest[..10]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc[12..].copy_from_slice(&rest[10..]);
        Some(View::Placed(unordered, resource, desc))
    }

    fn view_at(&self, i: u32, generation: u32) -> Option<View> {
        match generation >> KIND_SHIFT {
            0 => {
                let (view, current) = self.list.get(i as usize)?;
                (*current == generation).then(|| view.clone())
            }
            CONSTANT => Some(constant_of(i, generation)),
            PLACED => self.placed_of(i, generation),
            _ => None,
        }
    }

    fn free_entry(&mut self, i: u32, view: &View, generation: u32) {
        self.index.remove(view);
        self.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        self.free.push(i);
    }

    fn free_ownerless(&mut self, i: u32, generation: u32) {
        let Some((view, current)) = self.list.get(i as usize).cloned() else {
            return;
        };
        if current != generation || self.index.get(&view) != Some(&i) {
            return;
        }
        self.free_entry(i, &view, generation);
    }

    fn entry_kept(&mut self, view: View, cap: usize) -> [u32; 2] {
        let ownerless =
            view.resources() == [0, 0] && matches!(view, View::Constant(..) | View::Placed(..));
        let fresh = !self.index.contains_key(&view);
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.list.len() as u32 - 1
                    }
                };
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                self.index.insert(view, i);
                i
            }
        };
        let words = [i, self.list[i as usize].1];
        if ownerless && fresh {
            self.ownerless.push_back((words[0], words[1]));
            while cap != 0 && self.ownerless.len() > cap {
                let (old_i, old_gen) = self.ownerless.pop_front().expect("longer than cap");
                self.free_ownerless(old_i, old_gen);
            }
        }
        words
    }

    fn words_for_with(
        &mut self,
        view: View,
        use_inline: bool,
        cap: usize,
        max_shapes: usize,
    ) -> [u32; 2] {
        if use_inline {
            match &view {
                View::Constant(address, size) => {
                    if let Some(words) = constant_words(*address, *size) {
                        return words;
                    }
                }
                View::Placed(unordered, resource, desc) => {
                    let (shape, address) = shape_of(*unordered, resource, desc);
                    if address >> PLACED_ADDRESS_BITS == 0
                        && let Some(shape) = self.shape(shape, max_shapes)
                    {
                        return placed_words(shape, address);
                    }
                }
                _ => {}
            }
        }
        self.entry_kept(view, cap)
    }

    fn forget_resource(&mut self, resource: usize) {
        let Some(entries) = self.by_resource.remove(&resource) else {
            return;
        };
        for i in entries {
            let (view, generation) = self.list[i as usize].clone();
            if self.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
                continue;
            }
            self.free_entry(i, &view, generation);
        }
    }
}

fn make_placed_view(shape_id: u32, address: u64, unordered: bool) -> View {
    let mut resource = [0u32; 14];
    resource[0] = 3;
    resource[2] = shape_id;
    resource[4] = shape_id.wrapping_mul(0x045d_9f3b) ^ 0x1357_9bdf;
    resource[6] = shape_id.rotate_left(7);
    resource[8] = 28;
    resource[12] = shape_id & 7;
    let mut desc = [0u32; 16];
    desc[0] = 28;
    desc[1] = 4;
    desc[2] = 0x1688;
    desc[4] = shape_id ^ 0x55aa_1234;
    desc[5] = shape_id.wrapping_add(1);
    desc[6] = shape_id.wrapping_mul(3);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc[12] = shape_id.rotate_right(5);
    View::Placed(unordered, resource, desc)
}

#[test]
fn view_registry_matches_btree_oracle() {
    let mut rng = SplitMix64(0x1234_5678_9abc_def0);
    let caps = [0usize, 1, 4, 32];
    // 50,000 randomized ops across all View kinds, multiple slab boundaries
    // (>5,000 shapes), forget, ownerless eviction caps, and stale/invalid
    // handle checks.
    let ops_per_cap = [6_000usize, 6_000, 6_000, 32_000];
    for (cap_idx, (&cap, &ops)) in caps.iter().zip(ops_per_cap.iter()).enumerate() {
        let reg = ViewRegistry::new();
        let mut oracle = BTreeViewsOracle::default();
        let mut recorded: Vec<[u32; 2]> = Vec::new();
        let mut next_shape_id = 0u32;

        for step in 0..ops {
            let use_inline = (step & 15) != 15;
            let action = rng.gen_range(100);
            if action < 70 {
                let view = match rng.gen_range(10) {
                    0 => {
                        // Inline-eligible Constant
                        let addr = rng.next_u64() & ((1u64 << CONSTANT_ADDRESS_BITS) - 1);
                        let size = if rng.gen_range(2) == 0 {
                            rng.next_u32() & 0x3fff
                        } else {
                            (rng.next_u32() & 0x3fff) << 8
                        };
                        View::Constant(addr, size)
                    }
                    1 => {
                        // Overflow Constant (forces ownerless table entry)
                        let addr = (1u64 << CONSTANT_ADDRESS_BITS) | (rng.next_u64() & 0xffff);
                        let size = (1 << 14) + 1 + (rng.next_u32() & 0xff);
                        View::Constant(addr, size)
                    }
                    2 => {
                        // Shader with resource
                        let r = 0x1000 + rng.gen_range(64) * 0x10;
                        let desc = (rng.gen_range(3) != 0).then(|| {
                            std::array::from_fn(|k| rng.next_u32() & 0xf | (k as u32) << 8)
                        });
                        View::Shader(r, desc)
                    }
                    3 => {
                        // Unordered with resource + optional counter resource
                        let r = 0x1000 + rng.gen_range(64) * 0x10;
                        let counter = if rng.gen_range(2) == 0 {
                            0
                        } else {
                            0x1000 + rng.gen_range(64) * 0x10
                        };
                        let desc = (rng.gen_range(2) == 0)
                            .then(|| std::array::from_fn(|k| (rng.next_u32() & 7) + k as u32));
                        View::Unordered(r, counter, desc)
                    }
                    4 => {
                        let tag = rng.next_u32() & 31;
                        View::Sampler([tag; 13])
                    }
                    5 => {
                        let tag = rng.next_u32() & 31;
                        View::ConsoleSampler([tag; 27])
                    }
                    6..=8 => {
                        // Placed view: fresh shapes (to cross multiple slab
                        // boundaries, >5,000 shapes) mixed with repeated
                        // shapes at varying addresses.
                        let sid = if cap_idx == 3 && rng.gen_range(5) < 4 {
                            let id = next_shape_id;
                            next_shape_id += 1;
                            id
                        } else if next_shape_id > 0 && rng.gen_range(2) == 0 {
                            rng.gen_range(next_shape_id as usize) as u32
                        } else {
                            let id = next_shape_id;
                            next_shape_id += 1;
                            id
                        };
                        let addr = rng.next_u64() & ((1u64 << PLACED_ADDRESS_BITS) - 1);
                        make_placed_view(sid, addr, (sid & 1) != 0)
                    }
                    _ => {
                        // Placed view past 1 TiB (forces ownerless table entry)
                        let sid = rng.next_u32() & 0x3ff;
                        let addr = (1u64 << PLACED_ADDRESS_BITS) | (rng.next_u64() & 0xffff_ff00);
                        make_placed_view(sid, addr, rng.gen_range(2) == 0)
                    }
                };
                let got = reg.words_for_with(view.clone(), use_inline, cap, false);
                let want = oracle.words_for_with(view, use_inline, cap, ShapeArena::CAPACITY);
                assert_eq!(got, want, "cap={cap} step={step}: record words mismatch");
                assert_eq!(
                    reg.view_at(got[0], got[1]),
                    oracle.view_at(want[0], want[1]),
                    "cap={cap} step={step}: immediate view_at mismatch"
                );
                if recorded.len() < 4096 {
                    recorded.push(got);
                } else {
                    let slot = rng.gen_range(recorded.len());
                    recorded[slot] = got;
                }
            } else if action < 80 {
                // Forget a resource
                let r = 0x1000 + rng.gen_range(64) * 0x10;
                reg.forget_resource(r);
                oracle.forget_resource(r);
            } else if action < 94 && !recorded.is_empty() {
                // Query previously recorded handle (may be live, forgotten, or evicted)
                let [i, generation] = recorded[rng.gen_range(recorded.len())];
                assert_eq!(
                    reg.view_at(i, generation),
                    oracle.view_at(i, generation),
                    "cap={cap} step={step}: recorded view_at mismatch"
                );
            } else {
                // Query stale / out-of-bounds / invalid-kind handles
                let bad = match rng.gen_range(5) {
                    0 => [u32::MAX, 1],
                    1 => [0, 0],
                    2 => [0, 3 << KIND_SHIFT],
                    3 => placed_words(reg.shapes.len() as u32 + 7, 0x1000),
                    _ => [rng.next_u32() & 0xff, (rng.next_u32() & GENERATIONS).max(1)],
                };
                assert_eq!(
                    reg.view_at(bad[0], bad[1]),
                    oracle.view_at(bad[0], bad[1]),
                    "cap={cap} step={step}: invalid handle mismatch for {bad:?}"
                );
            }
        }

        if cap_idx == 3 {
            assert!(
                reg.shapes.len() > 5_000,
                "expected >5,000 shapes across multiple slabs, got {}",
                reg.shapes.len()
            );
        }
        assert_eq!(reg.shapes.len(), oracle.shapes.len());
        for &[i, generation] in &recorded {
            assert_eq!(reg.view_at(i, generation), oracle.view_at(i, generation));
        }
    }
}

#[test]
fn shape_arena_collisions_and_capacity_ceiling() {
    // Tiny arena: SLAB_LEN = 8 (1 << 3), MAX_SLABS = 4 -> exact CAPACITY = 32.
    let reg: ViewRegistryCore<3, 8, 4> = ViewRegistryCore::new();
    assert_eq!(ShapeArenaCore::<3, 8, 4>::CAPACITY, 32);

    let forced_key = ShapeKey(0xDEAD_BEEF);
    let mut expected_views = Vec::with_capacity(32);

    // Force 12 distinct shapes onto the exact same 64-bit ShapeKey, crossing slab 0 -> 1.
    for id in 0..12u32 {
        let addr = 0x1_0000 + u64::from(id) * 0x100;
        let view = make_placed_view(id, addr, (id & 1) != 0);
        let View::Placed(unordered, ref resource, ref desc) = view else {
            unreachable!()
        };
        let (shape, _) = shape_of(unordered, resource, desc);
        let idx = reg
            .intern_shape_with_key(shape, forced_key)
            .expect("fits in arena");
        assert_eq!(idx, id);
        expected_views.push((idx, addr, view, shape));
    }

    assert_eq!(reg.shapes.len(), 12);
    assert_eq!(
        reg.read_state()
            .shape_collisions
            .get(&forced_key)
            .map(Vec::len),
        Some(11)
    );

    // Re-interning all 12 colliding shapes must hit their exact existing indices.
    for &(idx, addr, ref view, shape) in &expected_views {
        assert_eq!(reg.intern_shape_with_key(shape, forced_key), Some(idx));
        let words = placed_words(idx, addr);
        assert_eq!(reg.view_at(words[0], words[1]), Some(view.clone()));
    }
    assert_eq!(reg.shapes.len(), 12);

    // Fill remaining 20 slots (12..32) across slabs 1, 2, and 3 to the exact ceiling.
    for id in 12..32u32 {
        let addr = 0x2_0000 + u64::from(id) * 0x100;
        let view = make_placed_view(id, addr, (id & 1) != 0);
        let words = reg.words_for_with(view.clone(), true, 64, false);
        assert_eq!(words[1] >> KIND_SHIFT, PLACED);
        let shape_idx = (payload(words[0], words[1]) >> PLACED_ADDRESS_BITS) as u32;
        assert_eq!(shape_idx, id);
        assert_eq!(reg.view_at(words[0], words[1]), Some(view.clone()));
        let View::Placed(unordered, ref resource, ref desc) = view else {
            unreachable!()
        };
        let (shape, _) = shape_of(unordered, resource, desc);
        expected_views.push((id, addr, view, shape));
    }

    assert_eq!(reg.shapes.len(), 32);
    assert_eq!(reg.shapes.allocated_slots(), 32);
    assert_eq!(reg.shapes.get(32), None);

    // 33rd distinct shape must hit capacity ceiling and fall back cleanly to Views::entry.
    let overflow_view = make_placed_view(999, 0x3_0000, false);
    let View::Placed(unordered, ref resource, ref desc) = overflow_view else {
        unreachable!()
    };
    let (overflow_shape, _) = shape_of(unordered, resource, desc);
    assert_eq!(reg.intern_shape(overflow_shape), None);

    let fallback = reg.words_for_with(overflow_view.clone(), true, 64, false);
    assert_eq!(
        fallback[1] >> KIND_SHIFT,
        0,
        "33rd shape must fall back to table entry"
    );
    assert_eq!(reg.shapes.len(), 32);
    assert_eq!(
        reg.view_at(fallback[0], fallback[1]),
        Some(overflow_view.clone())
    );
    // Repeat of 33rd shape reuses the same table entry.
    assert_eq!(reg.words_for_with(overflow_view, true, 64, false), fallback);

    // All 32 arena shapes remain intact and readable.
    for &(idx, addr, ref view, _) in &expected_views {
        let words = placed_words(idx, addr);
        assert_eq!(reg.placed_of(words[0], words[1]), Some(view.clone()));
    }
}

#[test]
fn shape_and_view_hash_quality() {
    use std::hash::{Hash, Hasher};

    // 1. Single-bit flip avalanche on ShapeKey::of across all fields.
    let base_shape: Shape = (
        false,
        std::array::from_fn(|i| 0x1020_3040u32.wrapping_add(i as u32 * 0x0101_0101)),
        std::array::from_fn(|i| 0x8070_6050u32.wrapping_add(i as u32 * 0x0305_070b)),
    );
    let base_hash = ShapeKey::of(&base_shape).0;
    let mut total_bits = 0u64;
    let mut flip_count = 0u64;

    let mut check_flip = |mutated: &Shape| {
        let h = ShapeKey::of(mutated).0;
        assert_ne!(h, base_hash, "single-bit flip produced identical ShapeKey");
        let diff = (h ^ base_hash).count_ones();
        assert!(
            diff >= 10,
            "weak avalanche on ShapeKey: only {diff} bits flipped"
        );
        total_bits += u64::from(diff);
        flip_count += 1;
    };

    // Flip shape.0
    {
        let mut s = base_shape;
        s.0 = !s.0;
        check_flip(&s);
    }
    // Flip every bit in shape.1[0..14] and shape.2[0..14]
    for word in 0..14 {
        for bit in 0..32 {
            let mut s1 = base_shape;
            s1.1[word] ^= 1u32 << bit;
            check_flip(&s1);

            let mut s2 = base_shape;
            s2.2[word] ^= 1u32 << bit;
            check_flip(&s2);
        }
    }
    let mean_shape_bits = total_bits as f64 / flip_count as f64;
    assert!(
        (28.0..=36.0).contains(&mean_shape_bits),
        "ShapeKey avalanche mean {mean_shape_bits:.2} outside [28, 36]"
    );

    // 2. Single-bit flip avalanche on ViewHash across View fields.
    let hash_view = |v: &View| -> u64 {
        let mut h = ViewHash::default();
        v.hash(&mut h);
        h.finish()
    };
    let base_view = View::Placed(false, base_shape.1, [0x2468_ace0; 16]);
    let base_vh = hash_view(&base_view);
    let mut v_bits = 0u64;
    let mut v_flips = 0u64;
    for word in 0..16 {
        for bit in 0..32 {
            let mut desc = [0x2468_ace0u32; 16];
            desc[word] ^= 1u32 << bit;
            let h = hash_view(&View::Placed(false, base_shape.1, desc));
            assert_ne!(h, base_vh, "single-bit flip produced identical ViewHash");
            v_bits += u64::from((h ^ base_vh).count_ones());
            v_flips += 1;
        }
    }
    let mean_view_bits = v_bits as f64 / v_flips as f64;
    assert!(
        (28.0..=36.0).contains(&mean_view_bits),
        "ViewHash avalanche mean {mean_view_bits:.2} outside [28, 36]"
    );

    // 3. Zero-array differentiation for ViewHash::write across lengths 0..=64.
    let mut zero_hashes = std::collections::HashSet::with_capacity(65);
    let zeros = [0u8; 64];
    for len in 0..=64 {
        let mut h = ViewHash::default();
        h.write(&zeros[..len]);
        assert!(
            zero_hashes.insert(h.finish()),
            "ViewHash collided on all-zero slice of length {len}"
        );
    }

    // 4. Zero collisions across 100,000 synthetic shapes.
    const COUNT: usize = 100_000;
    let mut seen = std::collections::HashSet::with_capacity(COUNT);
    let mut rng = SplitMix64(0xfedc_ba98_7654_3210);
    for i in 0..COUNT as u32 {
        let shape: Shape = match i & 3 {
            0 => {
                // Sequential single-word increments
                let mut r = [0u32; 14];
                r[0] = 3;
                r[4] = i;
                let mut d = [0u32; 14];
                d[0] = 28;
                d[1] = 4;
                ((i & 4) != 0, r, d)
            }
            1 => {
                // Differing only in last word of shape.2
                let r = [0u32; 14];
                let mut d = [0u32; 14];
                d[13] = i;
                (false, r, d)
            }
            2 => {
                // Differing in sparse high/low bits
                let mut r = [0u32; 14];
                r[(i as usize) % 14] = i.reverse_bits() ^ 0x8000_0001;
                let mut d = [0u32; 14];
                d[((i as usize) / 14) % 14] = i;
                (true, r, d)
            }
            _ => {
                // Pseudo-random realistic descriptors
                let r = std::array::from_fn(|_| rng.next_u32());
                let d = std::array::from_fn(|_| rng.next_u32());
                ((rng.next_u32() & 1) != 0, r, d)
            }
        };
        assert!(
            seen.insert(ShapeKey::of(&shape).0),
            "ShapeKey collision at synthetic shape {i}"
        );
    }
    assert_eq!(seen.len(), COUNT);
}

#[test]
fn view_registry_eight_thread_stress() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    // Self-verifying shape encoding so wait-free readers can validate
    // every published slot without sharing extra locks with writers.
    fn encoded_view(tag: u32, addr: u64) -> View {
        let unordered = (tag & 1) != 0;
        let mut resource = [0u32; 14];
        resource[0] = tag;
        resource[2] = tag.wrapping_mul(0x9e37_79b9);
        resource[7] = tag ^ 0x5a5a_a5a5;
        resource[13] = !tag;
        let mut desc = [0u32; 16];
        desc[0] = tag.rotate_left(11);
        desc[4] = tag.wrapping_add(0x1234_5678);
        (desc[10], desc[11]) = (addr as u32, (addr >> 32) as u32);
        desc[15] = tag ^ 0xf00d_cafe;
        View::Placed(unordered, resource, desc)
    }

    fn verify_placed(view: &View, expected_addr: u64) {
        let View::Placed(unordered, resource, desc) = *view else {
            panic!("expected View::Placed, got {view:?}");
        };
        let tag = resource[0];
        assert_eq!(unordered, (tag & 1) != 0);
        assert_eq!(resource[2], tag.wrapping_mul(0x9e37_79b9));
        assert_eq!(resource[7], tag ^ 0x5a5a_a5a5);
        assert_eq!(resource[13], !tag);
        assert_eq!(desc[0], tag.rotate_left(11));
        assert_eq!(desc[4], tag.wrapping_add(0x1234_5678));
        assert_eq!(desc[15], tag ^ 0xf00d_cafe);
        let got_addr = u64::from(desc[10]) | (u64::from(desc[11]) << 32);
        assert_eq!(got_addr, expected_addr);
    }

    const WRITERS: usize = 4;
    const READERS: usize = 4;
    const SHAPES_PER_WRITER: u32 = 1_280; // 4 * 1,280 = 5,120 > 4,096 (crosses 2 slab boundaries)

    let reg = Arc::new(ViewRegistry::new());
    let start = Arc::new(Barrier::new(WRITERS + READERS));
    let writers_done = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::with_capacity(WRITERS + READERS);

    for tid in 0..WRITERS as u32 {
        let reg = Arc::clone(&reg);
        let start = Arc::clone(&start);
        handles.push(std::thread::spawn(move || {
            start.wait();
            for n in 0..SHAPES_PER_WRITER {
                let tag = (tid << 16) | n;
                let addr = 0x10_0000 + u64::from(tag) * 0x40;
                let view = encoded_view(tag, addr);
                let words = reg.words_for_with(view.clone(), true, 256, false);
                assert_eq!(words[1] >> KIND_SHIFT, PLACED);
                assert_eq!(reg.placed_of(words[0], words[1]), Some(view));

                // Interleave resource-owned views and `forget_resource` to
                // exercise write-lock contention against wait-free readers.
                if (n & 7) == 0 {
                    let resource = ((tid as usize + 1) << 20) + (n as usize);
                    let srv = View::Shader(resource, Some([n; 10]));
                    let srv_words = reg.words_for_with(srv.clone(), true, 256, false);
                    assert_eq!(reg.view_at(srv_words[0], srv_words[1]), Some(srv));
                    reg.forget_resource(resource);
                    assert_eq!(reg.view_at(srv_words[0], srv_words[1]), None);
                }
            }
        }));
    }

    for rtid in 0..READERS as u64 {
        let reg = Arc::clone(&reg);
        let start = Arc::clone(&start);
        let writers_done = Arc::clone(&writers_done);
        handles.push(std::thread::spawn(move || {
            let mut rng = SplitMix64(0x9988_7766_5544_3322 ^ (rtid << 32));
            let mut reads = 0u64;
            start.wait();
            while !writers_done.load(Ordering::Acquire) || reads < 50_000 {
                let published = reg.shapes.len() as u32;
                if published == 0 {
                    std::hint::spin_loop();
                    continue;
                }
                // Hammer both the live publishing frontier and random older slabs.
                let idx = if (reads & 3) == 0 {
                    published - 1
                } else {
                    rng.gen_range(published as usize) as u32
                };
                let addr = ((reads << 8) ^ (rtid << 28)) & ((1u64 << PLACED_ADDRESS_BITS) - 1);
                let words = placed_words(idx, addr);
                let view = reg
                    .placed_of(words[0], words[1])
                    .expect("published shape must be wait-free readable");
                verify_placed(&view, addr);
                reads += 1;
            }
        }));
    }

    // Wait for all 4 writers, then signal the 4 readers.
    for handle in handles.drain(..WRITERS) {
        handle.join().expect("writer thread panicked");
    }
    writers_done.store(true, Ordering::Release);
    for handle in handles {
        handle.join().expect("reader thread panicked");
    }

    let total = (WRITERS as u32 * SHAPES_PER_WRITER) as usize;
    assert_eq!(reg.shapes.len(), total);
    for idx in 0..total as u32 {
        let addr = 0x4_0000 + u64::from(idx) * 0x20;
        let words = placed_words(idx, addr);
        let view = reg.placed_of(words[0], words[1]).expect("final check");
        verify_placed(&view, addr);
    }
}

/// Release microbenchmark comparing the pre-change baseline
/// (`RwLock<Vec<Shape>>` + SipHash + `BTreeMap`) against `ViewRegistry`
/// across all four hot regimes:
///   cargo test --release --locked -p victory-gpu views_registry_bench -- --ignored --nocapture
#[test]
#[ignore = "release microbenchmark"]
fn views_registry_bench() {
    use std::collections::BTreeMap;
    use std::hash::{BuildHasher, BuildHasherDefault, DefaultHasher};
    use std::hint::black_box;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier, RwLock};
    use std::time::Instant;

    struct BaselineViews {
        list: Vec<(View, u32)>,
        index: BTreeMap<View, u32>,
        shapes: Vec<Shape>,
        shape_index: BTreeMap<u64, u32>,
    }

    impl BaselineViews {
        fn new() -> Self {
            Self {
                list: Vec::new(),
                index: BTreeMap::new(),
                shapes: Vec::new(),
                shape_index: BTreeMap::new(),
            }
        }

        fn shape_key(shape: &Shape) -> u64 {
            BuildHasherDefault::<DefaultHasher>::default().hash_one(shape)
        }

        fn insert_shape(&mut self, shape: Shape) -> u32 {
            let key = Self::shape_key(&shape);
            if let Some(&at) = self.shape_index.get(&key)
                && self.shapes[at as usize] == shape
            {
                return at;
            }
            let at = self.shapes.len() as u32;
            self.shapes.push(shape);
            self.shape_index.insert(key, at);
            at
        }

        fn known_shape(&self, shape: &Shape) -> Option<u32> {
            let &at = self.shape_index.get(&Self::shape_key(shape))?;
            (self.shapes[at as usize] == *shape).then_some(at)
        }

        fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
            let payload = payload(i, generation);
            let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
            let &(unordered, resource, rest) =
                self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&rest[..10]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            desc[12..].copy_from_slice(&rest[10..]);
            Some(View::Placed(unordered, resource, desc))
        }

        fn insert_entry(&mut self, view: View) -> [u32; 2] {
            if let Some(&i) = self.index.get(&view) {
                return [i, self.list[i as usize].1];
            }
            let i = self.list.len() as u32;
            self.list.push((view.clone(), 1));
            self.index.insert(view, i);
            [i, 1]
        }

        fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
            let &i = self.index.get(view)?;
            Some([i, self.list[i as usize].1])
        }
    }

    // --- Regime 1: 1T view_at(PLACED) ---
    const R1_SHAPES: u32 = 4_096;
    const R1_ITERS: usize = 2_000_000;
    let base_r1 = RwLock::new(BaselineViews::new());
    let new_r1 = ViewRegistry::new();
    let mut r1_words = Vec::with_capacity(R1_SHAPES as usize);
    for id in 0..R1_SHAPES {
        let addr = 0x1_0000 + u64::from(id) * 0x100;
        let View::Placed(u, r, d) = make_placed_view(id, addr, (id & 1) != 0) else {
            unreachable!()
        };
        let (shape, _) = shape_of(u, &r, &d);
        let b_idx = base_r1.write().unwrap().insert_shape(shape);
        let n_idx = new_r1.intern_shape(shape).unwrap();
        assert_eq!(b_idx, n_idx);
        r1_words.push(placed_words(n_idx, addr));
    }

    let t0 = Instant::now();
    for i in 0..R1_ITERS {
        let [w0, w1] = r1_words[i & (R1_SHAPES as usize - 1)];
        let v = base_r1.read().unwrap().placed_of(w0, w1);
        black_box(v);
    }
    let base_r1_ns = t0.elapsed().as_nanos() as f64 / R1_ITERS as f64;

    let t0 = Instant::now();
    for i in 0..R1_ITERS {
        let [w0, w1] = r1_words[i & (R1_SHAPES as usize - 1)];
        let v = new_r1.view_at(w0, w1);
        black_box(v);
    }
    let new_r1_ns = t0.elapsed().as_nanos() as f64 / R1_ITERS as f64;

    // --- Regime 2: 8T contended view_at(PLACED) with concurrent writers (7R + 1W) ---
    const R2_READERS: usize = 7;
    const R2_ITERS_PER_READER: usize = 500_000;
    let words_arc = Arc::new(r1_words);

    let base_r2 = Arc::new(base_r1);
    let done_b = Arc::new(AtomicBool::new(false));
    let bar_b = Arc::new(Barrier::new(R2_READERS + 1));
    let mut threads = Vec::new();
    {
        let base = Arc::clone(&base_r2);
        let done = Arc::clone(&done_b);
        let bar = Arc::clone(&bar_b);
        threads.push(std::thread::spawn(move || {
            let mut id = R1_SHAPES;
            bar.wait();
            while !done.load(Ordering::Relaxed) {
                let View::Placed(u, r, d) = make_placed_view(id, 0x20_0000, (id & 1) != 0) else {
                    unreachable!()
                };
                let (shape, _) = shape_of(u, &r, &d);
                let mut guard = base.write().unwrap();
                if guard.shapes.len() < 32_768 {
                    guard.insert_shape(shape);
                } else {
                    guard.insert_entry(View::Shader(0x8000 + (id as usize & 0xfff), None));
                }
                drop(guard);
                id = id.wrapping_add(1);
            }
        }));
    }
    let t_b = Instant::now();
    let mut r_handles = Vec::new();
    for tid in 0..R2_READERS {
        let base = Arc::clone(&base_r2);
        let words = Arc::clone(&words_arc);
        let bar = Arc::clone(&bar_b);
        r_handles.push(std::thread::spawn(move || {
            bar.wait();
            let start = Instant::now();
            let mask = R1_SHAPES as usize - 1;
            let offset = tid * 131;
            for i in 0..R2_ITERS_PER_READER {
                let [w0, w1] = words[(i + offset) & mask];
                black_box(base.read().unwrap().placed_of(w0, w1));
            }
            start.elapsed()
        }));
    }
    let mut base_r2_sum_ns = 0f64;
    for h in r_handles {
        base_r2_sum_ns += h.join().unwrap().as_nanos() as f64;
    }
    let base_r2_wall_ms = t_b.elapsed().as_secs_f64() * 1e3;
    done_b.store(true, Ordering::Relaxed);
    for h in threads {
        h.join().unwrap();
    }
    let base_r2_ns = base_r2_sum_ns / (R2_READERS * R2_ITERS_PER_READER) as f64;

    let new_r2 = Arc::new(new_r1);
    let done_n = Arc::new(AtomicBool::new(false));
    let bar_n = Arc::new(Barrier::new(R2_READERS + 1));
    let mut threads = Vec::new();
    {
        let reg = Arc::clone(&new_r2);
        let done = Arc::clone(&done_n);
        let bar = Arc::clone(&bar_n);
        threads.push(std::thread::spawn(move || {
            let mut id = R1_SHAPES;
            bar.wait();
            while !done.load(Ordering::Relaxed) {
                let View::Placed(u, r, d) = make_placed_view(id, 0x20_0000, (id & 1) != 0) else {
                    unreachable!()
                };
                let (shape, _) = shape_of(u, &r, &d);
                if reg.shapes.len() < 32_768 {
                    let _ = reg.intern_shape(shape);
                } else {
                    let _ = reg.words_for_with(
                        View::Shader(0x8000 + (id as usize & 0xfff), None),
                        true,
                        1024,
                        false,
                    );
                }
                id = id.wrapping_add(1);
            }
        }));
    }
    let t_n = Instant::now();
    let mut r_handles = Vec::new();
    for tid in 0..R2_READERS {
        let reg = Arc::clone(&new_r2);
        let words = Arc::clone(&words_arc);
        let bar = Arc::clone(&bar_n);
        r_handles.push(std::thread::spawn(move || {
            bar.wait();
            let start = Instant::now();
            let mask = R1_SHAPES as usize - 1;
            let offset = tid * 131;
            for i in 0..R2_ITERS_PER_READER {
                let [w0, w1] = words[(i + offset) & mask];
                black_box(reg.view_at(w0, w1));
            }
            start.elapsed()
        }));
    }
    let mut new_r2_sum_ns = 0f64;
    for h in r_handles {
        new_r2_sum_ns += h.join().unwrap().as_nanos() as f64;
    }
    let new_r2_wall_ms = t_n.elapsed().as_secs_f64() * 1e3;
    done_n.store(true, Ordering::Relaxed);
    for h in threads {
        h.join().unwrap();
    }
    let new_r2_ns = new_r2_sum_ns / (R2_READERS * R2_ITERS_PER_READER) as f64;

    // --- Regime 3: known_shape hit latency on 65,536 shapes ---
    const R3_SHAPES: usize = 65_536;
    const R3_ITERS: usize = 500_000;
    let base_r3 = RwLock::new(BaselineViews::new());
    let new_r3 = ViewRegistry::new();
    let mut r3_shapes = Vec::with_capacity(R3_SHAPES);
    for id in 0..R3_SHAPES as u32 {
        let View::Placed(u, r, d) = make_placed_view(id, 0x1_0000, (id & 1) != 0) else {
            unreachable!()
        };
        let (shape, _) = shape_of(u, &r, &d);
        base_r3.write().unwrap().insert_shape(shape);
        new_r3.intern_shape(shape).unwrap();
        r3_shapes.push(shape);
    }
    let mut rng = SplitMix64(0x1122_3344_5566_7788);
    let r3_order: Vec<usize> = (0..R3_ITERS).map(|_| rng.gen_range(R3_SHAPES)).collect();

    let t0 = Instant::now();
    for &idx in &r3_order {
        let s = &r3_shapes[idx];
        black_box(base_r3.read().unwrap().known_shape(s));
    }
    let base_r3_ns = t0.elapsed().as_nanos() as f64 / R3_ITERS as f64;

    let t0 = Instant::now();
    for &idx in &r3_order {
        let s = &r3_shapes[idx];
        let key = ShapeKey::of(s);
        black_box(new_r3.read_state().known_shape(&new_r3.shapes, key, s));
    }
    let new_r3_ns = t0.elapsed().as_nanos() as f64 / R3_ITERS as f64;

    // --- Regime 4: entry_of hit latency on 32,768 views ---
    const R4_VIEWS: usize = 32_768;
    const R4_ITERS: usize = 500_000;
    let base_r4 = RwLock::new(BaselineViews::new());
    let new_r4 = ViewRegistry::new();
    let mut r4_views = Vec::with_capacity(R4_VIEWS);
    for id in 0..R4_VIEWS as u32 {
        let v = match id & 3 {
            0 => View::Shader(0x10_0000 + id as usize * 16, Some([id; 10])),
            1 => View::Unordered(
                0x20_0000 + id as usize * 16,
                0x30_0000 + (id as usize & 0xff) * 16,
                Some([id ^ 0x55; 10]),
            ),
            2 => View::Sampler([id; 13]),
            _ => View::ConsoleSampler([id; 27]),
        };
        base_r4.write().unwrap().insert_entry(v.clone());
        new_r4.write().entry_kept(v.clone(), 0);
        r4_views.push(v);
    }
    let r4_order: Vec<usize> = (0..R4_ITERS).map(|_| rng.gen_range(R4_VIEWS)).collect();

    let t0 = Instant::now();
    for &idx in &r4_order {
        black_box(base_r4.read().unwrap().entry_of(&r4_views[idx]));
    }
    let base_r4_ns = t0.elapsed().as_nanos() as f64 / R4_ITERS as f64;

    let t0 = Instant::now();
    for &idx in &r4_order {
        black_box(new_r4.read_state().entry_of(&r4_views[idx]));
    }
    let new_r4_ns = t0.elapsed().as_nanos() as f64 / R4_ITERS as f64;

    eprintln!("=== views_registry_bench ===");
    eprintln!(
        "1) 1T view_at(PLACED)         : baseline = {base_r1_ns:7.2} ns/op | new = {new_r1_ns:7.2} ns/op | speedup = {:.2}x",
        base_r1_ns / new_r1_ns
    );
    eprintln!(
        "2) 8T contended view_at(PLACED): baseline = {base_r2_ns:7.2} ns/op ({base_r2_wall_ms:6.1} ms wall) | new = {new_r2_ns:7.2} ns/op ({new_r2_wall_ms:6.1} ms wall) | speedup = {:.2}x",
        base_r2_ns / new_r2_ns
    );
    eprintln!(
        "3) known_shape hit (65K shapes): baseline = {base_r3_ns:7.2} ns/op | new = {new_r3_ns:7.2} ns/op | speedup = {:.2}x",
        base_r3_ns / new_r3_ns
    );
    eprintln!(
        "4) entry_of hit (32K views)    : baseline = {base_r4_ns:7.2} ns/op | new = {new_r4_ns:7.2} ns/op | speedup = {:.2}x",
        base_r4_ns / new_r4_ns
    );
}
