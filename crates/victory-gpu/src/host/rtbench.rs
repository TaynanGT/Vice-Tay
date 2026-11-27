//! Real `ID3D12QueryHeap` GPU timestamp microbenchmark of by-address 128-byte
//! `Box32` BVH node walks compiled as separate SM 6.6 (`cs_6_6`) pipelines
//! using `ResourceDescriptorHeap` (matching `victory-shader`'s emitted DXIL):
//! compares the previous 2-lookup per-lane `NonUniformResourceIndex`
//! `RWByteAddressBuffer` (`UAV`) path against the 1-lookup same-page-guarded
//! (`(addr & 0xFFFF) >= 0xFFC0`) wave-uniform (`WaveReadLaneFirst` +
//! `WaveActiveAllTrue`) read-only `ByteAddressBuffer` (`SRV`) path across
//! coherent waves, divergent waves, and VRAM vs PC-RAM backing, over a
//! full-size 16 MiB (`MEMORY_PAGES`) directory. Also verifies bit-exact output
//! parity between the old and new pipelines across intra-page and cross-page
//! (`0xFFC0`) nodes.
//!
//! Ignored by default; run with:
//! `cargo test -p victory-gpu --release --lib host::rtbench -- --ignored --nocapture`
//! Needs `dxc.exe` from the Windows SDK (`VICTORY_DXC`, or the newest SDK).

use super::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

/// 64-byte slots per copy (32 MiB = 512 Ki slots; 128-byte `Box32` nodes take
/// two consecutive 64-byte slots, including cross-page `0xFFC0` slots).
const SLOTS: u32 = 1 << 19;
const COPIES: u32 = 8;
const STEPS: u32 = 48;
/// Full 16 MiB directory (`MEMORY_PAGES` = 2^21 entries x 8 bytes), matching
/// `victory_shader::emit::MEMORY_PAGES`.
const DIR_PAGES: u32 = victory_shader::emit::MEMORY_PAGES;
/// Threads: a 1440p frame's pixels, one ray walker per thread.
const THREADS: u32 = 2560 * 1440;

const SOURCE: &str = r#"
#define RS "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(num32BitConstants=5, b0), SRV(t0), UAV(u0)"
cbuffer C : register(b0) { uint steps; uint mask; uint spread; uint uav_base; uint srv_base; };
ByteAddressBuffer dir : register(t0);
RWByteAddressBuffer outb : register(u0);

uint2 lookup(uint slot, uint lane_bias) {
    // Spread page indices across the 16 MiB directory (stride 2047) so
    // directory lookups exercise realistic L2/VRAM footprint.
    uint page = ((slot >> 10) * 2047u) & 2097151u;
    uint2 entry = dir.Load2(page * 8u);
    uint copy = (entry.x + lane_bias) & 7u;
    uint at = entry.y + (slot & 1023u) * 64u;
    return uint2(copy, at);
}

void load_first_uav_nu(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d) {
    RWByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(uav_base + copy)];
    a = buf.Load4(at);
    b = buf.Load4(at + 16u);
    c = buf.Load4(at + 32u);
    d = buf.Load4(at + 48u);
}

void load_second_uav_nu(uint copy, uint at, out uint4 e, out uint4 f, out uint4 g) {
    RWByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(uav_base + copy)];
    e = buf.Load4(at);
    f = buf.Load4(at + 16u);
    g = buf.Load4(at + 32u);
}

void load_first_uav_wf(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        RWByteAddressBuffer buf = ResourceDescriptorHeap[uav_base + first];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
    } else {
        load_first_uav_nu(copy, at, a, b, c, d);
    }
}

void load_second_uav_wf(uint copy, uint at, out uint4 e, out uint4 f, out uint4 g) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        RWByteAddressBuffer buf = ResourceDescriptorHeap[uav_base + first];
        e = buf.Load4(at);
        f = buf.Load4(at + 16u);
        g = buf.Load4(at + 32u);
    } else {
        load_second_uav_nu(copy, at, e, f, g);
    }
}

void load_first_srv_wf(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        ByteAddressBuffer buf = ResourceDescriptorHeap[srv_base + first];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
    } else {
        ByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(srv_base + copy)];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
    }
}

void load_second_srv_wf(uint copy, uint at, out uint4 e, out uint4 f, out uint4 g) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        ByteAddressBuffer buf = ResourceDescriptorHeap[srv_base + first];
        e = buf.Load4(at);
        f = buf.Load4(at + 16u);
        g = buf.Load4(at + 32u);
    } else {
        ByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(srv_base + copy)];
        e = buf.Load4(at);
        f = buf.Load4(at + 16u);
        g = buf.Load4(at + 32u);
    }
}

void load_all_uav_nu(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d, out uint4 e, out uint4 f, out uint4 g) {
    RWByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(uav_base + copy)];
    a = buf.Load4(at);
    b = buf.Load4(at + 16u);
    c = buf.Load4(at + 32u);
    d = buf.Load4(at + 48u);
    e = buf.Load4(at + 64u);
    f = buf.Load4(at + 80u);
    g = buf.Load4(at + 96u);
}

void load_all_uav_wf(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d, out uint4 e, out uint4 f, out uint4 g) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        RWByteAddressBuffer buf = ResourceDescriptorHeap[uav_base + first];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
        e = buf.Load4(at + 64u);
        f = buf.Load4(at + 80u);
        g = buf.Load4(at + 96u);
    } else {
        load_all_uav_nu(copy, at, a, b, c, d, e, f, g);
    }
}

void load_all_srv_wf(uint copy, uint at, out uint4 a, out uint4 b, out uint4 c, out uint4 d, out uint4 e, out uint4 f, out uint4 g) {
    uint first = WaveReadLaneFirst(copy);
    [branch] if (WaveActiveAllTrue(copy == first)) {
        ByteAddressBuffer buf = ResourceDescriptorHeap[srv_base + first];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
        e = buf.Load4(at + 64u);
        f = buf.Load4(at + 80u);
        g = buf.Load4(at + 96u);
    } else {
        ByteAddressBuffer buf = ResourceDescriptorHeap[NonUniformResourceIndex(srv_base + copy)];
        a = buf.Load4(at);
        b = buf.Load4(at + 16u);
        c = buf.Load4(at + 32u);
        d = buf.Load4(at + 48u);
        e = buf.Load4(at + 64u);
        f = buf.Load4(at + 80u);
        g = buf.Load4(at + 96u);
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint id : SV_DispatchThreadID) {
    uint p = ((id >> 6) * 2654435761u + (id & 63u) * 4099u) & mask;
    uint lane_bias = spread == 0u ? 0u : ((id >> spread) & 7u);
    uint acc = 0u;
    for (uint i = 0u; i < steps; i++) {
        uint s0 = p;
        uint s1 = (p + 1u) & mask;
        uint2 h0 = lookup(s0, lane_bias);
        uint4 a, b, c, d, e, f, g;

#if MODE == 0
        // Old baseline: unconditional 2nd directory lookup + NonUniform UAV.
        uint2 h1 = lookup(s1, lane_bias);
        load_first_uav_nu(h0.x, h0.y, a, b, c, d);
        load_second_uav_nu(h1.x, h1.y, e, f, g);
#elif MODE == 1
        // 1-lookup same-page guard (single handle for all 28 dwords) + NonUniform UAV.
        [branch] if ((s0 & 1023u) == 1023u) {
            uint2 h1 = lookup(s1, lane_bias);
            load_first_uav_nu(h0.x, h0.y, a, b, c, d);
            load_second_uav_nu(h1.x, h1.y, e, f, g);
        } else {
            load_all_uav_nu(h0.x, h0.y, a, b, c, d, e, f, g);
        }
#elif MODE == 2
        // 1-lookup same-page guard (single handle for all 28 dwords) + Wave-Uniform UAV.
        [branch] if ((s0 & 1023u) == 1023u) {
            uint2 h1 = lookup(s1, lane_bias);
            load_first_uav_wf(h0.x, h0.y, a, b, c, d);
            load_second_uav_wf(h1.x, h1.y, e, f, g);
        } else {
            load_all_uav_wf(h0.x, h0.y, a, b, c, d, e, f, g);
        }
#else
        // New full fast-path: 1-lookup same-page guard (single handle for all 28 dwords) + Wave-Uniform SRV.
        [branch] if ((s0 & 1023u) == 1023u) {
            uint2 h1 = lookup(s1, lane_bias);
            load_first_srv_wf(h0.x, h0.y, a, b, c, d);
            load_second_srv_wf(h1.x, h1.y, e, f, g);
        } else {
            load_all_srv_wf(h0.x, h0.y, a, b, c, d, e, f, g);
        }
#endif
        acc += (a.x ^ b.y ^ c.z ^ d.w) + (e.x ^ f.y ^ g.z);
        p = (p * 1103515245u + 12345u + ((a.y ^ e.x) & 255u)) & mask;
    }
    outb.Store(id * 4u, acc);
}
"#;

fn dxc() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("VICTORY_DXC") {
        return Some(p.into());
    }
    let bin = std::path::Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<_> = std::fs::read_dir(bin)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect();
    versions.sort();
    versions
        .into_iter()
        .rev()
        .map(|v| v.join(r"x64\dxc.exe"))
        .find(|p| p.exists())
}

#[test]
#[ignore]
fn bench_box32_zero_waterfall_srv_vs_uav() {
    let Some(host) = super::host() else {
        eprintln!("no host GPU");
        return;
    };
    let Some(dxc) = dxc() else {
        eprintln!("no dxc.exe");
        return;
    };
    let dir = std::env::temp_dir().join(format!("victory-rtbench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hlsl = dir.join("b.hlsl");
    std::fs::write(&hlsl, SOURCE).unwrap();

    let mut codes = Vec::with_capacity(4);
    for mode in 0..4u32 {
        let cso = dir.join(format!("b_{mode}.cso"));
        let define = format!("-DMODE={mode}");
        let out = std::process::Command::new(&dxc)
            .args(["-T", "cs_6_6", "-E", "main", "-O3", &define, "-Fo"])
            .arg(&cso)
            .arg(&hlsl)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "dxc mode {mode} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        codes.push(std::fs::read(&cso).unwrap());
    }

    let device = &host.device;
    // SAFETY: plain D3D12 object creation with valid descriptions; every
    // resource outlives the command list executions below.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &codes[0]).unwrap();
        let psos: Vec<ID3D12PipelineState> = codes
            .iter()
            .map(|code| {
                device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap()
            })
            .collect();
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        let frequency = queue.GetTimestampFrequency().unwrap() as f64;
        let mut done = 0u64;
        let run = |list: &ID3D12GraphicsCommandList, done: &mut u64| {
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            *done += 1;
            queue.Signal(&fence, *done).unwrap();
            fence.SetEventOnCompletion(*done, event).unwrap();
            WaitForSingleObject(event, INFINITE);
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
        };

        let pages = SLOTS / 1024;
        let dir_bytes = u64::from(DIR_PAGES) * 8;
        let node_bytes = u64::from(SLOTS) * 64;
        let (up, up_ptr) = host.upload_buffer(dir_bytes + node_bytes).unwrap();
        std::ptr::write_bytes(up_ptr, 0, dir_bytes as usize);
        let words = up_ptr.cast::<u32>();
        for p in 0..pages as usize {
            let dir_idx = (p * 2047) & (DIR_PAGES as usize - 1);
            *words.add(2 * dir_idx) = 0;
            *words.add(2 * dir_idx + 1) = p as u32 * 1024 * 64;
        }
        let mut x = 0x1234_5678u32;
        for i in 0..(node_bytes / 4) as usize {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *words.add((dir_bytes / 4) as usize + i) = x;
        }

        let directory = host.uav_buffer(dir_bytes).unwrap();
        let vram_copies: Vec<ID3D12Resource> = (0..COPIES)
            .map(|_| host.uav_buffer(node_bytes).unwrap())
            .collect();
        let pc_copies: Vec<(ID3D12Resource, *mut u8)> = (0..COPIES)
            .map(|_| host.shared_buffer(node_bytes).unwrap())
            .collect();
        for (_, ptr) in &pc_copies {
            std::ptr::copy_nonoverlapping(
                up_ptr.add(dir_bytes as usize),
                *ptr,
                node_bytes as usize,
            );
        }
        let outb = host.uav_buffer(u64::from(THREADS) * 4).unwrap();
        let out_rb = host.read_back_buffer(64 * 4).unwrap();
        list.CopyBufferRegion(&directory, 0, &up, 0, dir_bytes);
        for c in &vram_copies {
            list.CopyBufferRegion(c, 0, &up, dir_bytes, node_bytes);
        }
        run(&list, &mut done);

        let build_heap = |copies: &[&ID3D12Resource]| -> ID3D12DescriptorHeap {
            let heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2 * COPIES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            for (i, c) in copies.iter().enumerate() {
                // Slots 0..COPIES: raw UAVs (`uav_base = 0`).
                device.CreateUnorderedAccessView(
                    *c,
                    None::<&ID3D12Resource>,
                    Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_UAV {
                                FirstElement: 0,
                                NumElements: (node_bytes / 4) as u32,
                                StructureByteStride: 0,
                                CounterOffsetInBytes: 0,
                                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            },
                        },
                    }),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: start.ptr + i * step,
                    },
                );
                // Slots COPIES..2*COPIES: raw SRVs (`srv_base = COPIES`).
                device.CreateShaderResourceView(
                    *c,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: 0,
                                NumElements: (node_bytes / 4) as u32,
                                StructureByteStride: 0,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            },
                        },
                    }),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: start.ptr + (COPIES as usize + i) * step,
                    },
                );
            }
            heap
        };

        let vram_refs: Vec<&ID3D12Resource> = vram_copies.iter().collect();
        let pc_refs: Vec<&ID3D12Resource> = pc_copies.iter().map(|(r, _)| r).collect();
        let vram_heap = build_heap(&vram_refs);
        let pc_heap = build_heap(&pc_refs);

        let mut stamps: Option<ID3D12QueryHeap> = None;
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut stamps,
            )
            .unwrap();
        let stamps = stamps.unwrap();
        let back = host.read_back_buffer(16).unwrap();

        let mut run_case = |heap: &ID3D12DescriptorHeap, mode: usize, spread: u32, read_sample: bool| -> (f64, Vec<u32>) {
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&psos[mode]);
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.SetComputeRoot32BitConstants(
                0,
                5,
                [STEPS, SLOTS - 1, spread, 0, COPIES].as_ptr().cast(),
                0,
            );
            list.SetComputeRootShaderResourceView(1, directory.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, outb.GetGPUVirtualAddress());
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.Dispatch(THREADS / 64, 1, 1);
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            if read_sample {
                list.ResourceBarrier(&[transition(
                    &outb,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyBufferRegion(&out_rb, 0, &outb, 0, 64 * 4);
                list.ResourceBarrier(&[transition(
                    &outb,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            run(&list, &mut done);
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data)).unwrap();
            let t = std::slice::from_raw_parts(data.cast::<u64>(), 2);
            let us = t[1].saturating_sub(t[0]) as f64 / frequency * 1e6;
            back.Unmap(0, None);
            let sample = if read_sample {
                let mut sp = std::ptr::null_mut();
                out_rb.Map(0, None, Some(&mut sp)).unwrap();
                let words = std::slice::from_raw_parts(sp.cast::<u32>(), 64).to_vec();
                out_rb.Unmap(0, None);
                words
            } else {
                Vec::new()
            };
            (us, sample)
        };

        // Parity check: all four SM 6.6 PSOs must produce bit-exact identical
        // results across coherent and divergent waves.
        for spread in [0u32, 5, 1] {
            let (_, ref_words) = run_case(&vram_heap, 0, spread, true);
            for mode in 1..4 {
                let (_, got) = run_case(&vram_heap, mode, spread, true);
                assert_eq!(got, ref_words, "parity mismatch for mode={mode} spread={spread}");
            }
        }

        let cases: [(&str, bool, usize, u32); 10] = [
            ("VRAM old 2-lookup NonUniform UAV (coherent)", false, 0, 0),
            ("VRAM 1-lookup NonUniform UAV (coherent)", false, 1, 0),
            ("VRAM 1-lookup Wave-Uniform UAV (coherent)", false, 2, 0),
            ("VRAM new 1-lookup Wave-Uniform SRV (coherent)", false, 3, 0),
            ("VRAM old 2-lookup NonUniform UAV (2 copies/wave)", false, 0, 5),
            ("VRAM new 1-lookup Wave-Uniform SRV (2 copies/wave)", false, 3, 5),
            ("VRAM old 2-lookup NonUniform UAV (8 copies/wave)", false, 0, 1),
            ("VRAM new 1-lookup Wave-Uniform SRV (8 copies/wave)", false, 3, 1),
            ("PC-RAM old 2-lookup NonUniform UAV (coherent)", true, 0, 0),
            ("PC-RAM new 1-lookup Wave-Uniform SRV (coherent)", true, 3, 0),
        ];
        for &(_, pc, mode, spread) in &cases {
            let heap = if pc { &pc_heap } else { &vram_heap };
            run_case(heap, mode, spread, false);
        }
        for (name, pc, mode, spread) in cases {
            let heap = if pc { &pc_heap } else { &vram_heap };
            let mut t: Vec<f64> = (0..5).map(|_| run_case(heap, mode, spread, false).0).collect();
            t.sort_by(f64::total_cmp);
            println!(
                "{name}: {:.1} us ({:.2} ms) (median of 5; {STEPS} Box32 nodes/ray, {THREADS} rays)",
                t[2],
                t[2] / 1000.0
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
