//! The ray-tracing node test end to end: a synthetic compute program that
//! runs `image_bvh64_intersect_ray` (or the 32-bit form) once per thread,
//! lowered to DXIL, validated by `dxil.dll` when present, run on the host
//! GPU and WARP, and compared dword for dword with the CPU reference
//! ([`crate::bvh`]).
//!
//! The nodes are in a raw buffer. Two ways to name it are run: the
//! buffer's descriptor from the program's table (a resource of the host,
//! [`crate::resource::BvhNodes::Resource`]), and the title's way, a
//! descriptor of numbers over title memory with the structure's address in
//! the node pointer or in the descriptor. These tests have no host that
//! keeps copies of title memory and their directory, so for the second
//! they put a stand-in under the node reader
//! (`lower::intersect::TEST_MEMORY`); without it the same program reads
//! through the directory, which is only checked to lower and validate.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::lower::intersect::TEST_MEMORY;
use super::{HostShader, lower, validator};
use crate::bvh::{self, Ray, tests as nodes};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::resource::BvhNodes;
use crate::{ShaderError, io, ir, resource, structure};

/// Threads of one group, each with its own record.
const THREADS: u32 = 32;
/// Bytes of a thread's record: the node pointer (two dwords), the ray's
/// extent, origin, direction and inverse direction, and whether the
/// thread runs the instruction.
const RECORD: u32 = 64;
/// Where the nodes start in the buffer.
const NODES: u32 = THREADS * RECORD;
/// What the result registers hold before the instruction.
const BEFORE: [u32; 4] = [1, 2, 3, 4];

/// How the program names the nodes' memory.
#[derive(Clone, Copy, PartialEq)]
enum Descriptor {
    /// The buffer's descriptor from the table.
    Table,
    /// Numbers, as the title builds it: dwords 0 and 1 (the base address
    /// in 256-byte units and the box sorting bit); the size and the rest of
    /// the flags as the title sets them.
    Numbers(u32, u32),
}

/// The descriptor the title builds for the 64-bit form: the whole address
/// space, sorted boxes, barycentrics returned.
const WHOLE: Descriptor = Descriptor::Numbers(0, bvh::BOX_SORT);

/// The program: per thread, the record's twelve operand dwords in v1-v12,
/// the instruction's result in v13-v16 (under a lane mask of the threads
/// whose record says so), stored to the UAV at 16 bytes a thread.
fn program(wide: bool, descriptor: Descriptor) -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    for k in 0..4u32 {
        // v_mov_b32 v13+k, 1+k
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k));
    }
    let rsrc = match descriptor {
        Descriptor::Table => 8,
        Descriptor::Numbers(word0, word1) => {
            // s_mov_b32 s24, word0
            w.extend([0xBE98_03FF, word0]);
            if word1 == bvh::BOX_SORT {
                // As the title makes it.
                w.push(0xBE99_0B81); // s_brev_b32 s25, 1
            } else {
                w.extend([0xBE99_03FF, word1]); // s_mov_b32 s25, word1
            }
            w.extend([
                0xBE9A_03C1, // s_mov_b32 s26, -1
                0xBE9B_03FF,
                0x8900_03FF, // s_mov_b32 s27, 0x890003ff
            ]);
            24
        }
    };
    w.extend([
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
    ]);
    let first = 0x0000_0D01 | (rsrc >> 2) << 16;
    if wide {
        // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[rsrc:rsrc+3]
        w.extend([0xF19C_9F07, first, 0x0504_0302, 0x0908_0706, 0x000C_0B0A]);
    } else {
        // image_bvh_intersect_ray v[13:16], [v1, v3 .. v12], s[rsrc:rsrc+3]
        w.extend([0xF198_9F07, first, 0x0605_0403, 0x0A09_0807, 0x0000_0C0B]);
    }
    w.extend([
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0x342A_0084, // v_lshlrev_b32 v21, 4, v0
        0xE078_1000,
        0x8003_0D15, // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn native(words: &[u32]) -> NativeProgram {
    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 22,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Srv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Lowers the program; the DXIL is validated when the validator is there.
fn build(
    wide: bool,
    descriptor: Descriptor,
) -> Result<(HostShader, resource::Resources), ShaderError> {
    build_compute_shader(&native(&program(wide, descriptor)))
}

/// `native`, a compute program, translated (the tests' own way: no container).
pub(super) fn build_compute_shader(
    native: &NativeProgram,
) -> Result<(HostShader, resource::Resources), ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    Ok((shader, res))
}

/// One thread's test: a node (by its index in the node list), the kind its
/// pointer carries, and the ray; `None` leaves the thread masked out.
type Case = Option<(usize, u32, Ray)>;

/// The scene: node dwords (a 32-bit box node takes 32, the others 16) and
/// where each node starts, in dwords from the first node.
struct Scene {
    words: Vec<u32>,
    starts: Vec<usize>,
}

impl Scene {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            starts: Vec::new(),
        }
    }

    /// Adds a node of `dwords` (16 or 32) and returns its index.
    fn add(&mut self, node: [u32; 28], dwords: usize) -> usize {
        self.starts.push(self.words.len());
        let mut padded = [0u32; 32];
        padded[..28].copy_from_slice(&node);
        // Bytes the instruction must not read as part of the node: what
        // follows a 64-byte node, and a box node's last 16 bytes.
        for w in padded.iter_mut().skip(if dwords == 16 { 16 } else { 28 }) {
            *w = 0x7fc0_dead;
        }
        self.words.extend(&padded[..dwords]);
        self.starts.len() - 1
    }

    /// The pointer of node `n` with `kind`, counted from the first node.
    fn pointer(&self, n: usize, kind: u32) -> u32 {
        (4 * self.starts[n] as u32) >> 3 | kind
    }

    /// What the instruction reads of node `n`.
    fn node(&self, n: usize) -> [u32; 28] {
        std::array::from_fn(|k| self.words.get(self.starts[n] + k).copied().unwrap_or(0))
    }
}

/// A small generator of repeatable numbers.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// A float in -1..1 in steps of 1/4096 (exactly representable).
    fn unit(&mut self) -> f32 {
        (self.next() % 8193) as f32 / 4096.0 - 1.0
    }

    fn vector(&mut self, scale: f32) -> [f32; 3] {
        [
            self.unit() * scale,
            self.unit() * scale,
            self.unit() * scale,
        ]
    }
}

/// The scene and batches of cases over it.
fn scene() -> (Scene, Vec<Vec<Case>>) {
    let mut s = Scene::new();
    let mut g = Numbers(0x5eed);
    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved =
        |b: [f32; 6], by: [f32; 3]| -> [f32; 6] { std::array::from_fn(|k| b[k] + by[k % 3]) };
    let off = [f32::NAN; 6];
    let mut box_nodes = Vec::new();
    // The boxes of the reference's own tests, in both widths.
    let lined = [
        moved(UNIT, [30.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 5.0, 0.0]),
        moved(UNIT, [20.0, 0.0, 0.0]),
    ];
    let ids = [0x105, 0x20c, 0x304, 0x400];
    box_nodes.push((s.add(nodes::box32(ids, lined), 32), 5));
    box_nodes.push((s.add(nodes::box16(ids, lined), 16), 4));
    // Boxes around the origin, one switched off, one child missing.
    let around = [
        UNIT,
        moved(UNIT, [0.5, 0.5, 0.5]),
        off,
        [-4.0, -0.25, -0.25, 4.0, 0.25, 0.25],
    ];
    let ids2 = [8, bvh::NONE, 24, 0x1fff_fffc];
    box_nodes.push((s.add(nodes::box32(ids2, around), 32), 5));
    box_nodes.push((s.add(nodes::box16(ids2, around), 16), 4));
    // A child that is not there with inverted infinite bounds (the title's
    // builder leaves such slots): every ray "enters" it at minus infinity.
    let mut empty = lined;
    empty[1] = [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let ids3 = [0x105, bvh::NONE, 0x304, 0x400];
    box_nodes.push((s.add(nodes::box32(ids3, empty), 32), 5));
    // Random boxes.
    for n in 0..6 {
        let boxes: [[f32; 6]; 4] = std::array::from_fn(|_| {
            let (c, h) = (g.vector(8.0), g.vector(2.0).map(f32::abs));
            std::array::from_fn(|k| c[k % 3] + if k < 3 { -h[k] } else { h[k - 3] })
        });
        let ids: [u32; 4] = std::array::from_fn(|k| 0x1000 * (n + 1) + 8 * k as u32 + 5);
        if n % 2 == 0 {
            box_nodes.push((s.add(nodes::box32(ids, boxes), 32), 5));
        } else {
            box_nodes.push((s.add(nodes::box16(ids, boxes), 16), 4));
        }
    }
    // Triangles: one of the reference's tests, a pair as the title's
    // builder stores it, all four triangles of a full node with turned
    // ids, random ones.
    let mut triangle_nodes = Vec::new();
    let front = [[0.0, 2.0, 5.0], [2.0, -2.0, 5.0], [-2.0, -2.0, 5.0]];
    triangle_nodes.push((s.add(nodes::triangles(&front, 9), 16), 0));
    let pair = s.add(
        nodes::triangles(
            &[front[0], front[1], front[2], [2.0, 2.0, 5.0]],
            9 | 0x80 | 6 << 8,
        ),
        16,
    );
    triangle_nodes.extend([(pair, 0), (pair, 1)]);
    let five = [
        [-3.0, -1.0, 6.0],
        [-1.0, 2.0, 6.5],
        [1.0, -1.0, 6.0],
        [3.0, 2.0, 6.5],
        [5.0, -1.0, 7.0],
    ];
    // I and J per triangle: (1, 2), (2, 0), (0, 1), (1, 2).
    let full = s.add(nodes::triangles(&five, 9 | 2 << 8 | 4 << 16 | 9 << 24), 16);
    triangle_nodes.extend((0..4).map(|k| (full, k)));
    for _ in 0..5 {
        let c = g.vector(6.0);
        let v: [[f32; 3]; 3] = std::array::from_fn(|_| {
            let d = g.vector(3.0);
            [c[0] + d[0], c[1] + d[1], c[2] + d[2]]
        });
        triangle_nodes.push((s.add(nodes::triangles(&v, 9), 16), 0));
    }
    let mut cases: Vec<Case> = Vec::new();
    // Rays of the reference's tests against the lined-up boxes.
    for &(n, kind) in &box_nodes[..2] {
        for (origin, direction, extent) in [
            ([0.0; 3], [1.0, 0.0, 0.0], 1000.0),
            ([0.0; 3], [1.0, 0.0, 0.0], 19.5),
            ([0.0; 3], [1.0, 0.0, 0.0], 19.0),
            ([0.0; 3], [-1.0, 0.0, 0.0], 1000.0),
            ([40.0, 0.0, 0.0], [-2.0, 0.0, 0.0], 1000.0),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], 1000.0),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1000.0),
            ([0.0, 0.5, 0.25], [1.0, -0.0, 0.0], 1000.0),
            ([0.0; 3], [0.0; 3], 1000.0),
        ] {
            cases.push(Some((n, kind, Ray::new(origin, direction, extent))));
        }
    }
    // Random rays at every box node, half of them aimed at a box.
    for &(n, kind) in &box_nodes {
        let words = s.node(n);
        let boxes = if kind == 5 {
            bvh::boxes32(&words)
        } else {
            bvh::boxes16(&words)
        };
        for k in 0..12 {
            let origin = g.vector(12.0);
            let direction = if k % 2 == 0 {
                let b = boxes[(k / 2) % 4];
                if b[0].is_nan() {
                    g.vector(1.0)
                } else {
                    std::array::from_fn(|a| (b[a] + b[3 + a]) / 2.0 - origin[a] + g.unit() / 2.0)
                }
            } else {
                g.vector(1.0)
            };
            let extent = if k % 3 == 0 { 6.0 } else { 100.0 };
            cases.push(Some((n, kind, Ray::new(origin, direction, extent))));
        }
    }
    // Rays at every triangle: aimed inside, at an edge, at a vertex, past
    // it, from both sides, along each axis.
    for &(n, kind) in &triangle_nodes {
        let p = bvh::triangle_vertices(&s.node(n), kind);
        let at = |w: [f32; 3]| -> [f32; 3] {
            std::array::from_fn(|a| w[0] * p[0][a] + w[1] * p[1][a] + w[2] * p[2][a])
        };
        for (k, w) in [
            [0.5, 0.25, 0.25],
            [0.125, 0.75, 0.125],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.75, 0.75, -0.5],
            [-0.25, 0.5, 0.75],
            [0.25, 0.25, 0.5],
        ]
        .into_iter()
        .enumerate()
        {
            let target = at(w);
            let origin = if k % 2 == 0 { [0.0; 3] } else { g.vector(16.0) };
            let mut direction: [f32; 3] = std::array::from_fn(|a| target[a] - origin[a]);
            if k == 7 {
                // The triangle lies behind.
                direction = direction.map(|d| -d);
            }
            cases.push(Some((n, kind, Ray::new(origin, direction, 100.0))));
        }
        for axis in 0..3 {
            for sign in [1.0f32, -1.0] {
                let target = at([0.25, 0.5, 0.25]);
                let mut origin = target;
                origin[axis] -= 8.0 * sign;
                let mut direction = [0.0; 3];
                direction[axis] = sign;
                cases.push(Some((n, kind, Ray::new(origin, direction, 100.0))));
            }
        }
    }
    // Kinds the instruction does not read, and threads masked out.
    let ray = Ray::new([0.0; 3], [0.0, 0.0, 1.0], 100.0);
    cases.extend([
        Some((triangle_nodes[0].0, 6, ray)),
        Some((triangle_nodes[0].0, 7, ray)),
        None,
        Some((
            box_nodes[0].0,
            5,
            Ray::new([0.0; 3], [1.0, 0.0, 0.0], 1000.0),
        )),
        None,
    ]);
    let batches = cases
        .chunks(THREADS as usize)
        .map(<[Case]>::to_vec)
        .collect();
    (s, batches)
}

/// The buffer for one batch: the threads' records, then the nodes. Node
/// pointers count from `base` before the first node, in bytes.
fn buffer(scene: &Scene, batch: &[Case], base: u64) -> Vec<u32> {
    let mut data = vec![0u32; (NODES / 4) as usize];
    for (t, case) in batch.iter().enumerate() {
        let Some((n, kind, ray)) = case else { continue };
        let r = &mut data[t * (RECORD / 4) as usize..][..13];
        let pointer = (base >> 3) + u64::from(scene.pointer(*n, *kind));
        (r[0], r[1]) = (pointer as u32, (pointer >> 32) as u32);
        r[2] = ray.extent.to_bits();
        for a in 0..3 {
            r[3 + a] = ray.origin[a].to_bits();
            r[6 + a] = ray.direction[a].to_bits();
            r[9 + a] = ray.inverse[a].to_bits();
        }
        r[12] = 1;
    }
    data.extend(&scene.words);
    data
}

fn expected(scene: &Scene, case: &Case, sort: bool) -> [u32; 4] {
    expected_for(scene, case, sort, false)
}

fn expected_for(scene: &Scene, case: &Case, sort: bool, fused: bool) -> [u32; 4] {
    match case {
        Some((n, kind, ray)) => {
            if fused {
                bvh::intersect_fused(&scene.node(*n), *kind, ray, sort)
            } else {
                bvh::intersect(&scene.node(*n), *kind, ray, sort)
            }
        }
        None => BEFORE,
    }
}

/// Sets the tests' title memory for this thread until dropped.
struct Memory;

impl Memory {
    fn at(base: u64) -> Self {
        // Heap slot 0: the buffer of records and nodes.
        TEST_MEMORY.set(Some((0, base)));
        Self
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        TEST_MEMORY.set(None);
    }
}

/// Runs every batch on the hardware adapter and WARP and compares with
/// the reference. `pointers` is what node pointers count from, in bytes
/// before the first node.
fn run(wide: bool, descriptor: Descriptor, pointers: u64, sort: bool) {
    let (shader, res) = build(wide, descriptor).expect("the program lowers");
    assert_eq!(res.bvh.len(), 1);
    let (scene, batches) = scene();
    // The reference must see a bit of everything, or the comparison
    // proves little.
    let all: Vec<&Case> = batches.iter().flatten().collect();
    let results: Vec<[u32; 4]> = all.iter().map(|c| expected(&scene, c, sort)).collect();
    let boxes = |hits: usize| {
        all.iter()
            .zip(&results)
            .filter(|(c, r)| {
                matches!(c, Some((_, 4 | 5, _)))
                    && r.iter().filter(|&&x| x != bvh::NONE).count() == hits
            })
            .count()
    };
    let triangles = |hit: bool| {
        all.iter()
            .zip(&results)
            .filter(|(c, r)| {
                matches!(c, Some((_, 0..=3, _))) && (r[0] != f32::INFINITY.to_bits()) == hit
            })
            .count()
    };
    assert!(
        boxes(0) > 5 && boxes(1) > 5 && boxes(2) > 5 && boxes(3) > 0,
        "box cases"
    );
    assert!(
        triangles(true) > 30 && triangles(false) > 30,
        "triangle cases"
    );
    // Sorting must matter to some case, or the unsorted run proves nothing.
    assert!(
        all.iter()
            .any(|c| expected(&scene, c, true) != expected(&scene, c, false)),
        "no case tells sorted from unsorted"
    );
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (b, batch) in batches.iter().enumerate() {
            let data = buffer(&scene, batch, pointers);
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0; 5]))],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Raw,
                            },
                            View::RawUav(4 * THREADS),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            for t in 0..THREADS as usize {
                let case = batch.get(t).copied().flatten();
                let want = match batch.get(t) {
                    // GPU hardware fuses DXIL `FMad` (single rounding); WARP
                    // evaluates `FMad` unfused.
                    Some(c) => expected_for(&scene, c, sort, !warp),
                    // A thread past the batch: its record is zeros, so it
                    // is masked out.
                    None => BEFORE,
                };
                let got = &out[0][4 * t..4 * t + 4];
                assert_eq!(
                    got,
                    want,
                    "{label}, batch {b} thread {t}: {case:x?}\n got {:x?} = {:?}\nwant {:x?} = {:?}",
                    got,
                    got.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>(),
                    want,
                    want.map(f32::from_bits),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn the_64_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    // Pointers count from the buffer's start.
    run(true, Descriptor::Table, u64::from(NODES), true);
    assert_eq!(
        build(true, Descriptor::Table).unwrap().1.bvh[0].1,
        BvhNodes::Resource
    );
}

#[test]
fn the_32_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    run(false, Descriptor::Table, u64::from(NODES), true);
}

/// Where the tests' title memory puts the buffer: past 4 GiB, on 256 bytes.
const BASE: u64 = 0x12_3456_7800;

#[test]
fn the_64_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's way: one descriptor for all of memory, the structure's
    // address folded into a 64-bit pointer.
    let _memory = Memory::at(BASE);
    run(true, WHOLE, BASE + u64::from(NODES), true);
}

#[test]
fn the_32_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's other way: the structure's address in the descriptor
    // (in 256-byte units, 40 bits), a 32-bit pointer from there. Here with
    // an address past 2^40 bytes, and without box sorting.
    let base = 0x312_3456_7800u64;
    let structure = base + u64::from(NODES);
    let _memory = Memory::at(base);
    let descriptor = Descriptor::Numbers((structure >> 8) as u32, (structure >> 40) as u32);
    run(false, descriptor, 0, false);
    // And sorted.
    let descriptor = Descriptor::Numbers(
        (structure >> 8) as u32,
        (structure >> 40) as u32 | bvh::BOX_SORT,
    );
    run(false, descriptor, 0, true);
}

#[test]
fn a_descriptor_of_numbers_is_title_memory_read_through_the_directory() {
    for wide in [true, false] {
        let native = native(&program(wide, WHOLE));
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("tracked");
        // The flags the title sets: sorted boxes, barycentrics, a
        // ray-tracing structure. No resource is taken for it: the buffer
        // loads' and the store's descriptors are all there is.
        assert_eq!(res.bvh.len(), 1);
        assert_eq!(
            res.bvh[0].1,
            BvhNodes::Memory {
                word1: 0x8000_0000,
                word3: 0x8900_03ff
            }
        );
        assert_eq!(res.descriptors.len(), 5);
        // Without the tests' stand-in the nodes are read as the title's
        // programs will read them: through the host's directory of title
        // memory. The program lowers and validates, and asks the host for
        // the directory and for the list of pages it found no copy of.
        let (shader, _) = build(wide, WHOLE).expect("lowers through the directory");
        assert!(shader.root_srvs.contains(&super::SLOT_MEMORY));
        assert!(shader.root_uavs.contains(&super::SLOT_FAULTS));
    }
}

#[test]
fn a_structure_that_returns_triangle_ids_is_refused_by_name() {
    // Triangle return mode 0 (bit 24 of dword 3 clear): the last two
    // result dwords would be a triangle id and a hit status.
    let mut words = program(true, WHOLE);
    let flags = words.iter().position(|&w| w == 0x8900_03FF).unwrap();
    words[flags] = 0x8800_03FF;
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("tracked");
    let io = io::raise(&p, &native).expect("io::raise");
    let r = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    match r {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a ray-tracing structure that returns triangle ids");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A buffer view picked from the table at an index the program computes,
/// whose dwords 0 and 1 the program reads as numbers (the tracer takes its
/// structure's address so, by an index from a constant buffer): they are
/// the view's title address from the host's table by view heap slot, after
/// the strides in the same root view.
#[test]
fn a_picked_views_address_comes_from_the_hosts_table() {
    use super::{SLOT_STRIDES, SLOT_STRIDES_BYTES, VIEW_ADDRESSES};
    let words = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0x8F06_8500, // s_lshl_b32 s6, s0, 5
        0xF408_0202,
        0x0C00_0000, // s_load_dwordx4 s[8:11], s[4:5], s6
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0208, // v_mov_b32 v1, s8
        0x7E04_0209, // v_mov_b32 v2, s9
        0x342A_0083, // v_lshlrev_b32 v21, 3, v0
        0xE074_1000,
        0x8003_0115, // buffer_store_dwordx2 v[1:2], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let (shader, _) = build_compute_shader(&native(&words)).expect("the program lowers");
    assert!(shader.root_srvs.contains(&SLOT_STRIDES), "{:?}", shader.root_srvs);
    assert!(
        !shader.user_data.iter().any(|u| u.root == RootType::Other(super::USER_VIEW_ADDRESS)),
        "no host word for a fixed place"
    );
    let mut table = vec![0u32; SLOT_STRIDES_BYTES as usize / 4];
    let at = VIEW_ADDRESSES as usize / 4;
    // Slot 2, the one picked; slot 0 as a decoy.
    table[at..at + 2].copy_from_slice(&[0xdead_0000, 0xdd]);
    table[at + 4..at + 6].copy_from_slice(&[0x1234_5600, 0x42]);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let decoy = [7u32; 4];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // s0: the picked view's place in the table (slot 2).
                    roots: vec![
                        (0, Root::Constants(vec![2, 0, 0, 0, 0])),
                        (1, Root::View { srv: true, data: table.clone() }),
                    ],
                    views: &[
                        View::Buffer { data: &decoy, stride: Stride::Raw },
                        View::RawUav(8 * THREADS),
                        View::Buffer { data: &decoy, stride: Stride::Raw },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        for t in 0..THREADS as usize {
            assert_eq!(out[0][2 * t..2 * t + 2], [0x1234_5600, 0x42], "{label} thread {t}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// A register that holds a lane mask on one path and a number on the other
/// (the ray tracer's SGPRs reused between saved masks and constant-buffer
/// fields), joined where the wave takes one edge, then cut with mask
/// operations and read as a number: the mask operations compute on the
/// numbers (`sop`), and the read is the console's register.
#[test]
fn mask_operations_on_a_joined_number_give_the_number() {
    let words = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x8706_7E6A, // s_and_b32 s6, vcc_lo, exec_lo (lanes 0-23)
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 +1
        0x8106_8101, // s_add_i32 s6, s1, 1 (a number)
        0x8707_FF06,
        0x00FF_00FF, // s_and_b32 s7, s6, 0x00ff00ff
        0x8808_0207, // s_or_b32 s8, s7, s2
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E02_0208, // v_mov_b32 v1, s8
        0x342A_0082, // v_lshlrev_b32 v21, 2, v0
        0xE070_1000,
        0x8003_0115, // buffer_store_dword v1, v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    // On the first try refused, as before (programs that translate so do
    // not change); `translate` tries such a program once more with mask
    // operations on numbers read as numbers.
    match build_compute_shader(&native(&words)) {
        Err(ShaderError::Unsupported { what, .. }) => assert_eq!(what, lower::MASK_ELSEWHERE),
        other => panic!("{:?}", other.map(|_| ())),
    }
    let (shader, _) = lower::with_mask_ops_on_numbers(|| build_compute_shader(&native(&words)))
        .expect("the program lowers");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // The hardware device only: the program declares the console's 32
    // lanes (its wave operations), which WARP does not take.
    let mut ran = 0;
    for (label, warp) in [("hardware", false)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the {label} device does not take this program (waves of 64?): {e}; not run");
                continue;
            }
        };
        let unused = [0u32; 4];
        for (constants, want) in [
            // The mask's path: lanes 0-23 as the number.
            ([0, 0x1234_5678, 0x0100_0000], 0x01FF_00FF),
            // The number's path.
            ([1, 0x1234_5678, 0x0100_0000], 0x0134_0079),
        ] {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants([&constants[..], &[0, 0]].concat()))],
                        views: &[View::Buffer { data: &unused, stride: Stride::Raw }, View::RawUav(4 * THREADS)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            for t in 0..THREADS as usize {
                assert_eq!(out[0][t], want, "{label} {constants:x?} thread {t}");
            }
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("not run on a device; lowered only");
    }
}

/// Encodes one thread's 16-dword record (64 bytes) for [`program`]:
/// `addr` is the 64-byte-aligned title byte address of the node and `kind`
/// is its 3-bit node type (0..=7).
fn put_record(records: &mut [u32], thread: usize, addr: u64, kind: u32, ray: &Ray, active: bool) {
    assert_eq!(addr & 63, 0, "node address must be 64-byte aligned");
    let r = &mut records[thread * (RECORD / 4) as usize..][..13];
    let pointer = (addr >> 3) | u64::from(kind & 7);
    r[0] = pointer as u32;
    r[1] = (pointer >> 32) as u32;
    r[2] = ray.extent.to_bits();
    for a in 0..3 {
        r[3 + a] = ray.origin[a].to_bits();
        r[6 + a] = ray.direction[a].to_bits();
        r[9 + a] = ray.inverse[a].to_bits();
    }
    r[12] = u32::from(active);
}

/// Builds the user-data constants vector for a shader that uses
/// `USER_MEMORY_SRV_BASE` or `USER_MEMORY_BASE` alongside the test's s0..s3
/// constants and table at slot 0.
fn memory_user_data(shader: &HostShader, memory_base: u32) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match u.root {
            RootType::Other(k)
                if k == super::USER_MEMORY_SRV_BASE || k == super::USER_MEMORY_BASE =>
            {
                memory_base
            }
            RootType::Table => 0,
            RootType::Constants => 0,
            other => panic!("unexpected user-data entry {other:?}"),
        };
    }
    ud
}

/// Real directory (`SLOT_MEMORY`) + faults (`SLOT_FAULTS`) + read-only SRV
/// (`USER_MEMORY_SRV_BASE`) end-to-end test on hardware and WARP covering:
/// - intra-page `Box32` at `0x0000` (same-page fast path, 1 directory load),
/// - intra-page `Box32` at `0xFF80` (last same-page offset; next page unmapped
///   to prove no cross-page lookup or fault occurs),
/// - cross-page `Box32` at `0xFFC0` spanning TWO DIFFERENT copies in the
///   directory (with poison bytes immediately after page P in copy 1 to prove
///   the second half actually performs the second lookup into copy 2),
/// - cross-page `Box32` at `0xFFC0` whose second page is unmapped (copy 0):
///   children 0..1 still hit while children 2..3 are masked to `NONE` and
///   `SLOT_FAULTS` records only that missing page.
#[test]
fn box32_same_page_and_cross_page_directory_on_the_card() {
    let (shader, _) = build(true, WHOLE).expect("lowers through directory");
    assert!(shader.root_srvs.contains(&super::SLOT_MEMORY));
    assert!(shader.root_uavs.contains(&super::SLOT_FAULTS));
    assert!(
        shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_SRV_BASE)),
        "read-only walker must request USER_MEMORY_SRV_BASE"
    );
    assert!(
        !shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_BASE)),
        "read-only walker must not request UAV USER_MEMORY_BASE"
    );

    const PAGE_DWORDS: usize = 65536 / 4;
    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved = |by: [f32; 3]| -> [f32; 6] { std::array::from_fn(|k| UNIT[k] + by[k % 3]) };
    let ray = Ray::new([0.0; 3], [1.0, 0.0, 0.0], 1000.0);

    // Distinct Box32 nodes whose 4 children all lie along +x at distinct distances.
    let node_a = nodes::box32(
        [0x105, 0x205, 0x305, 0x405],
        [
            moved([30.0, 0.0, 0.0]),
            moved([10.0, 0.0, 0.0]),
            moved([40.0, 0.0, 0.0]),
            moved([20.0, 0.0, 0.0]),
        ],
    );
    let node_b = nodes::box32(
        [0x505, 0x605, 0x705, 0x805],
        [
            moved([25.0, 0.0, 0.0]),
            moved([5.0, 0.0, 0.0]),
            moved([15.0, 0.0, 0.0]),
            moved([35.0, 0.0, 0.0]),
        ],
    );
    let node_c = nodes::box32(
        [0x905, 0xA05, 0xB05, 0xC05],
        [
            moved([50.0, 0.0, 0.0]),
            moved([30.0, 0.0, 0.0]),
            moved([12.0, 0.0, 0.0]),
            moved([6.0, 0.0, 0.0]),
        ],
    );
    let node_d = nodes::box32(
        [0xD05, 0xE05, 0xF05, 0x1005],
        [
            moved([22.0, 0.0, 0.0]),
            moved([8.0, 0.0, 0.0]),
            moved([4.0, 0.0, 0.0]),
            moved([2.0, 0.0, 0.0]),
        ],
    );

    // Copy 1 (4 pages = 256 KiB):
    //   offset 0x0_0000: page 1 -> node_a at 0x0000 (intra-page)
    //   offset 0x1_0000: page 2 -> node_b at 0xFF80 (last intra-page Box32)
    //   offset 0x2_0000: page 4 -> node_c first half [0..16] at 0xFFC0
    //   offset 0x3_0000: POISON (all NaN boxes so skipping 2nd lookup fails!)
    let mut copy1 = vec![0u32; 4 * PAGE_DWORDS];
    copy1[..28].copy_from_slice(&node_a);
    let b_at = PAGE_DWORDS + (0xFF80 / 4);
    copy1[b_at..b_at + 28].copy_from_slice(&node_b);
    let c_first_at = 2 * PAGE_DWORDS + (0xFFC0 / 4);
    copy1[c_first_at..c_first_at + 16].copy_from_slice(&node_c[..16]);
    for w in &mut copy1[3 * PAGE_DWORDS..3 * PAGE_DWORDS + 16] {
        *w = f32::NAN.to_bits();
    }

    // Copy 2 (1 page = 64 KiB):
    //   offset 0x0_0000: page 5 -> node_c second half [16..28] at 0x0000
    let mut copy2 = vec![0u32; PAGE_DWORDS];
    copy2[..12].copy_from_slice(&node_c[16..28]);

    // Copy 3 (2 pages = 128 KiB):
    //   offset 0x0_0000: page 6 -> node_d first half [0..16] at 0xFFC0
    //   offset 0x1_0000: valid boxes (decoy: page 7 is unmapped so children 2..3
    //                    MUST be masked to NONE, not read from copy 3 + 0x10000!)
    let mut copy3 = vec![0u32; 2 * PAGE_DWORDS];
    let d_first_at = 0xFFC0 / 4;
    copy3[d_first_at..d_first_at + 16].copy_from_slice(&node_d[..16]);
    copy3[PAGE_DWORDS..PAGE_DWORDS + 12].copy_from_slice(&node_d[16..28]);

    // Directory (16 pages):
    //   page 1 -> (copy 1, 0x0_0000)
    //   page 2 -> (copy 1, 0x1_0000)
    //   page 3 -> unmapped (0, 0) — proves 0x2_FF80 does NOT look up page 3
    //   page 4 -> (copy 1, 0x2_0000)
    //   page 5 -> (copy 2, 0x0_0000) — different copy for 0x4_FFC0 second half
    //   page 6 -> (copy 3, 0x0_0000)
    //   page 7 -> unmapped (0, 0) — 0x6_FFC0 second half unmapped & faulted
    let mut directory = vec![0u32; 16 * 2];
    let set_dir = |dir: &mut [u32], page: usize, copy: u32, byte_off: u32| {
        dir[2 * page] = copy;
        dir[2 * page + 1] = byte_off;
    };
    set_dir(&mut directory, 1, 1, 0x0_0000);
    set_dir(&mut directory, 2, 1, 0x1_0000);
    set_dir(&mut directory, 4, 1, 0x2_0000);
    set_dir(&mut directory, 5, 2, 0x0_0000);
    set_dir(&mut directory, 6, 3, 0x0_0000);

    let mut records = vec![0u32; (NODES / 4) as usize];
    let want_a = bvh::intersect(&node_a, 5, &ray, true);
    let want_b = bvh::intersect(&node_b, 5, &ray, true);
    let want_c = bvh::intersect(&node_c, 5, &ray, true);
    // Page 7 is unmapped -> only children 0 & 1 of node_d (at x=22 and x=8) hit, sorted.
    let want_d = [0xE05, 0xD05, bvh::NONE, bvh::NONE];

    let mut expected_threads = vec![BEFORE; THREADS as usize];
    for t in 0..THREADS as usize {
        match t {
            0..8 => {
                put_record(&mut records, t, 0x1_0000, 5, &ray, true);
                expected_threads[t] = want_a;
            }
            8..16 => {
                put_record(&mut records, t, 0x2_FF80, 5, &ray, true);
                expected_threads[t] = want_b;
            }
            16..24 => {
                put_record(&mut records, t, 0x4_FFC0, 5, &ray, true);
                expected_threads[t] = want_c;
            }
            24..28 => {
                put_record(&mut records, t, 0x6_FFC0, 5, &ray, true);
                expected_threads[t] = want_d;
            }
            // Inactive lanes pointing at an unmapped page: must not fault!
            _ => {
                put_record(&mut records, t, 0x9_0000, 5, &ray, false);
                expected_threads[t] = BEFORE;
            }
        }
    }

    // Heap layout: 0=records SRV, 1=output UAV, 2=null SRV (copy 0),
    // 3=copy 1 SRV, 4=copy 2 SRV, 5=copy 3 SRV.
    let null_srv = [0u32; 4];
    let ud = memory_user_data(&shader, 2);

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // Root signature order: param 0 = user_data constants,
        // param 1 = SLOT_FAULTS (253, UAV), param 2 = SLOT_MEMORY (254, SRV).
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud.clone())),
                        (1, Root::Uav(super::MEMORY_FAULTS)),
                        (2, Root::View { srv: true, data: directory.clone() }),
                    ],
                    views: &[
                        View::Buffer { data: &records, stride: Stride::Raw },
                        View::RawUav(4 * THREADS),
                        View::Buffer { data: &null_srv, stride: Stride::Raw },
                        View::Buffer { data: &copy1, stride: Stride::Raw },
                        View::Buffer { data: &copy2, stride: Stride::Raw },
                        View::Buffer { data: &copy3, stride: Stride::Raw },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        for t in 0..THREADS as usize {
            let got = &out[0][4 * t..4 * t + 4];
            assert_eq!(
                got, expected_threads[t],
                "{label}: thread {t} mismatch (got {got:x?}, want {:x?})",
                expected_threads[t]
            );
        }
        let faults = &out[1];
        let faulted_pages: Vec<u32> = faults.iter().copied().filter(|&p| p != 0).collect();
        assert_eq!(
            faulted_pages,
            vec![7],
            "{label}: expected only unmapped page 7 in SLOT_FAULTS (no fault for 0x0000, 0xFF80, mapped 0xFFC0, or inactive lanes)"
        );
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// Tests BOTH coherent waves (all 32 active lanes hit the same copy ->
/// `WaveReadLaneFirst` + `WaveAllTrue` uniform handle fast path) and divergent
/// waves (lanes interleaved across 3 copies + unmapped copy 0 ->
/// `NonUniformResourceIndex` fallback), for BOTH:
/// 1. Read-only BVH walkers (`USER_MEMORY_SRV_BASE`, raw SRV `ByteAddressBuffer`),
/// 2. Shaders that also contain a `global_store_dword` (`USER_MEMORY_BASE`,
///    raw UAV `RWByteAddressBuffer`).
#[test]
fn wave_uniform_and_divergent_srv_vs_uav_on_the_card() {
    let (ro_shader, _) = build(true, WHOLE).expect("ro walker lowers");
    assert!(
        ro_shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_SRV_BASE))
    );

    // Same program with a never-taken (when s0 == 0) `global_store_dword` to
    // raw address v[1:2], forcing `stores_by_address = true` so the shader
    // must use `USER_MEMORY_BASE` raw UAVs instead of read-only SRVs.
    let mut rw_words = program(true, WHOLE);
    let end = rw_words.pop().unwrap(); // s_endpgm
    rw_words.extend([
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 +2 (skips 2-dword global_store_dword)
        0xDC70_8000,
        0x007D_0D01, // global_store_dword v[1:2], v13, off
        end,
    ]);
    let (rw_shader, _) = build_compute_shader(&native(&rw_words)).expect("rw shader lowers");
    assert!(
        rw_shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_BASE)),
        "shader with global_store must use UAV USER_MEMORY_BASE"
    );
    assert!(
        !rw_shader
            .user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_SRV_BASE)),
        "shader with global_store must not use read-only USER_MEMORY_SRV_BASE"
    );

    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved = |by: [f32; 3]| -> [f32; 6] { std::array::from_fn(|k| UNIT[k] + by[k % 3]) };
    let ray_x = Ray::new([0.0; 3], [1.0, 0.0, 0.0], 1000.0);
    let ray_z = Ray::new([0.0; 3], [0.0, 0.0, 1.0], 100.0);

    // Copy 1: Box32 at +0x000, Box16 at +0x100, Triangle at +0x200.
    let box32_c1 = nodes::box32(
        [0x115, 0x225, 0x335, 0x445],
        [
            moved([40.0, 0.0, 0.0]),
            moved([10.0, 0.0, 0.0]),
            moved([20.0, 0.0, 0.0]),
            moved([30.0, 0.0, 0.0]),
        ],
    );
    let box16_c1 = nodes::box16(
        [0x514, 0x624, 0x734, 0x844],
        [
            moved([12.0, 0.0, 0.0]),
            moved([24.0, 0.0, 0.0]),
            moved([6.0, 0.0, 0.0]),
            moved([18.0, 0.0, 0.0]),
        ],
    );
    let tri_c1 = nodes::triangles(&[[0.0, 2.0, 5.0], [2.0, -2.0, 5.0], [-2.0, -2.0, 5.0]], 9);

    // Copy 2: distinct Box32 at +0x000, distinct Triangle at +0x200.
    let box32_c2 = nodes::box32(
        [0xA15, 0xB25, 0xC35, 0xD45],
        [
            moved([15.0, 0.0, 0.0]),
            moved([5.0, 0.0, 0.0]),
            moved([25.0, 0.0, 0.0]),
            moved([35.0, 0.0, 0.0]),
        ],
    );
    let tri_c2 = nodes::triangles(&[[0.0, 3.0, 8.0], [3.0, -3.0, 8.0], [-3.0, -3.0, 8.0]], 9);

    // Copy 3: distinct Box16 at +0x100.
    let box16_c3 = nodes::box16(
        [0xE14, 0xF24, 0x1034, 0x1144],
        [
            moved([32.0, 0.0, 0.0]),
            moved([16.0, 0.0, 0.0]),
            moved([8.0, 0.0, 0.0]),
            moved([4.0, 0.0, 0.0]),
        ],
    );

    let mut copy1 = vec![0u32; 256];
    copy1[..28].copy_from_slice(&box32_c1);
    copy1[64..80].copy_from_slice(&box16_c1[..16]);
    copy1[128..144].copy_from_slice(&tri_c1[..16]);

    let mut copy2 = vec![0u32; 256];
    copy2[..28].copy_from_slice(&box32_c2);
    copy2[128..144].copy_from_slice(&tri_c2[..16]);

    let mut copy3 = vec![0u32; 256];
    copy3[64..80].copy_from_slice(&box16_c3[..16]);

    // Directory: page 1 -> copy 1, page 2 -> copy 2, page 3 -> copy 3, page 4 -> unmapped.
    let mut directory = vec![0u32; 16 * 2];
    directory[2] = 1;
    directory[4] = 2;
    directory[6] = 3;

    // Batch 0: Coherent wave (all 32 lanes hit copy 1: mix of Box32, Box16, Triangle).
    let mut coherent_rec = vec![0u32; (NODES / 4) as usize];
    let mut coherent_want_hw = Vec::with_capacity(THREADS as usize);
    let mut coherent_want_warp = Vec::with_capacity(THREADS as usize);
    for t in 0..THREADS as usize {
        let (addr, kind, node, ray) = match t % 3 {
            0 => (0x1_0000, 5, &box32_c1, &ray_x),
            1 => (0x1_0100, 4, &box16_c1, &ray_x),
            _ => (0x1_0200, 0, &tri_c1, &ray_z),
        };
        put_record(&mut coherent_rec, t, addr, kind, ray, true);
        coherent_want_hw.push(bvh::intersect_fused(node, kind, ray, true));
        coherent_want_warp.push(bvh::intersect(node, kind, ray, true));
    }

    // Batch 1: Divergent wave (lanes interleaved across copies 1, 2, 3 and unmapped page 4).
    let mut divergent_rec = vec![0u32; (NODES / 4) as usize];
    let mut divergent_want_hw = Vec::with_capacity(THREADS as usize);
    let mut divergent_want_warp = Vec::with_capacity(THREADS as usize);
    for t in 0..THREADS as usize {
        match t % 4 {
            0 => {
                put_record(&mut divergent_rec, t, 0x1_0000, 5, &ray_x, true);
                divergent_want_hw.push(bvh::intersect_fused(&box32_c1, 5, &ray_x, true));
                divergent_want_warp.push(bvh::intersect(&box32_c1, 5, &ray_x, true));
            }
            1 => {
                put_record(&mut divergent_rec, t, 0x2_0000, 5, &ray_x, true);
                divergent_want_hw.push(bvh::intersect_fused(&box32_c2, 5, &ray_x, true));
                divergent_want_warp.push(bvh::intersect(&box32_c2, 5, &ray_x, true));
            }
            2 => {
                put_record(&mut divergent_rec, t, 0x3_0100, 4, &ray_x, true);
                divergent_want_hw.push(bvh::intersect_fused(&box16_c3, 4, &ray_x, true));
                divergent_want_warp.push(bvh::intersect(&box16_c3, 4, &ray_x, true));
            }
            _ => {
                put_record(&mut divergent_rec, t, 0x4_0000, 5, &ray_x, true);
                divergent_want_hw.push([bvh::NONE; 4]);
                divergent_want_warp.push([bvh::NONE; 4]);
            }
        }
    }

    // Batch 2: Divergent triangles across copies 1 and 2 (tests triangle wave-uniform fallback).
    let mut div_tri_rec = vec![0u32; (NODES / 4) as usize];
    let mut div_tri_want_hw = Vec::with_capacity(THREADS as usize);
    let mut div_tri_want_warp = Vec::with_capacity(THREADS as usize);
    for t in 0..THREADS as usize {
        let (addr, node) = if t % 2 == 0 {
            (0x1_0200, &tri_c1)
        } else {
            (0x2_0200, &tri_c2)
        };
        put_record(&mut div_tri_rec, t, addr, 0, &ray_z, true);
        div_tri_want_hw.push(bvh::intersect_fused(node, 0, &ray_z, true));
        div_tri_want_warp.push(bvh::intersect(node, 0, &ray_z, true));
    }

    let null_buf = [0u32; 4];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (mode_name, shader, use_uav) in [
            ("ro_srv", &ro_shader, false),
            ("rw_uav", &rw_shader, true),
        ] {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}/{mode_name}: compute_pipeline failed: {e}"));
            let ud = memory_user_data(shader, 2);
            let batches = [
                ("coherent", &coherent_rec, &coherent_want_hw, &coherent_want_warp),
                ("divergent_box", &divergent_rec, &divergent_want_hw, &divergent_want_warp),
                ("divergent_tri", &div_tri_rec, &div_tri_want_hw, &div_tri_want_warp),
            ];
            for (batch_name, rec, want_hw, want_warp) in batches {
                let want = if warp { want_warp } else { want_hw };
                let views: Vec<View<'_>> = if use_uav {
                    vec![
                        View::Buffer { data: rec, stride: Stride::Raw },
                        View::RawUav(4 * THREADS),
                        View::RawUavData(&null_buf),
                        View::RawUavData(&copy1),
                        View::RawUavData(&copy2),
                        View::RawUavData(&copy3),
                    ]
                } else {
                    vec![
                        View::Buffer { data: rec, stride: Stride::Raw },
                        View::RawUav(4 * THREADS),
                        View::Buffer { data: &null_buf, stride: Stride::Raw },
                        View::Buffer { data: &copy1, stride: Stride::Raw },
                        View::Buffer { data: &copy2, stride: Stride::Raw },
                        View::Buffer { data: &copy3, stride: Stride::Raw },
                    ]
                };
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![
                                (0, Root::Constants(ud.clone())),
                                (1, Root::Uav(super::MEMORY_FAULTS)),
                                (2, Root::View { srv: true, data: directory.clone() }),
                            ],
                            views: &views,
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}/{mode_name}/{batch_name}: dispatch failed: {e}"));
                for t in 0..THREADS as usize {
                    let got = &out[0][4 * t..4 * t + 4];
                    assert_eq!(
                        got, want[t],
                        "{label}/{mode_name}/{batch_name} thread {t}: got {got:x?}, want {:x?}",
                        want[t]
                    );
                }
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

/// Pixel-shader helper lanes are excluded from `WaveReadLaneFirst` /
/// `WaveActiveAllTrue`, so `bvh_with_handle` must skip wave ops in `StageIo::Ps`
/// (falling back to `NonUniformResourceIndex`) while keeping wave ops enabled
/// for compute shaders.
#[test]
fn pixel_shader_bvh_memory_skips_wave_ops_for_helper_lane_safety() {
    let lower_native = |prog: &NativeProgram| {
        let p = ir::build(&prog.code, prog.wave64, prog.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &prog.layout, prog.stage, Some(&st)).expect("resource");
        let io = io::raise(&p, prog).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: prog,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = lowered.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        lowered
    };
    let cs_prog = native(&program(true, WHOLE));
    let cs = lower_native(&cs_prog);
    // Prepend `v_mov_b32 v0, 0` so `v0` is defined in PS (no CS thread ID).
    let mut ps_words = vec![0x7E00_0280u32];
    ps_words.extend(program(true, WHOLE));
    let mut ps_prog = native(&ps_words);
    ps_prog.stage = HwStage::Ps;
    ps_prog.io = StageIo::Ps {
        inputs: vec![],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let ps = lower_native(&ps_prog);
    assert!(
        cs.wave_ops,
        "compute shader by-address BVH walk must emit wave-uniform fast path"
    );
    assert!(
        !ps.wave_ops,
        "pixel shader by-address BVH walk must skip wave ops for helper-lane safety"
    );
    assert!(
        ps.user_data
            .iter()
            .any(|u| u.root == RootType::Other(super::USER_MEMORY_SRV_BASE)),
        "read-only PS walker must still use USER_MEMORY_SRV_BASE"
    );
}
