# Texture pool allocation and contention

These changes target CPU allocation overhead and contention while textures are
streamed. They do not change texture quality, render resolution, upscaling,
frame generation, pool block size, or the delay before a returned piece can
be reused. They are not a measured game FPS improvement.

## Separate pool bookkeeping from driver allocation

Previously the staging and texture pools held their bookkeeping mutex while
creating a new 64 MiB block. That included driver allocation, residency,
mapping an upload buffer, or waiting for the texture heap's mapping fence.
Another thread could not take an existing free piece or return a completed
piece until that work finished.

Each pool now has a separate growth gate. Existing pieces are reserved under
the bookkeeping mutex. A miss releases that mutex, enters the growth gate,
and checks the pool again before making a block outside the bookkeeping lock.
Only a completely initialized block is published. Waiting misses can reuse a
block supplied by the previous grower instead of creating duplicates.

Optional spare growth uses a nonblocking attempt to enter the growth gate.
A caller that already reserved a piece does not wait behind another grower.
The caller that actually replenishes spare stock still performs that driver
work synchronously; this is not a background allocator or a guarantee that
every allocation is nonblocking.

Growth failure retains a 500 ms retry delay, measured from completion of the
failed attempt. Existing free pieces remain usable during that delay. The
texture block's submission stamp is sampled after growth completes so a
submission that began during allocation is not incorrectly treated as older
than the block. The separate GPU mapping fence and its serialization remain
unchanged.

## Retain the oldest-first fallback order

When no ripe texture block fits, the old fallback collected every live block's
`(submission stamp, block index)` into a temporary vector and sorted it on
each request. The pool now retains that order in a `BTreeSet`, updated when
blocks are added or reclaimed.

Fallback probes free ranges in exactly the same order, including the block
index tie-break for equal stamps. It does not use a minimum-fitting linear
scan: with fragmented ranges and reused block indices, that alternative can
probe many more ranges than the original oldest-first search.

This moves ordered-index allocation to block membership changes. It does not
remove every pool allocation or replace range bookkeeping.

## Verification scope

The focused tests use synthetic memory owners and controlled threads, not
title files. They check selection, reclamation, slot reuse, growth ordering,
retry behavior, and publication identity. Existing placed-resource and staging
tests exercise the native D3D12 integration.

A local optimized harness extracted the actual generic pool implementation
before and after the change. It counted allocator calls over 100,000 repeated
take/return operations per fixture, with all blocks unripe, 16 small nonfitting
holes followed by a fitting range, and both ascending and descending age order.
No blocks were grown during measurement.

| Live blocks | Before: allocation/reallocation calls per request | After |
| --- | ---: | ---: |
| 8 | 2 | 0 |
| 64 | 5 | 0 |
| 256 | 7 | 0 |

The harness returned ranges directly rather than waiting for the real reuse
delay. These counts establish removal of the fallback's temporary vector in
those fixtures, not a frame-time improvement. A separate source-derived
concurrency harness reproduced blocked returns with the old lock scope and
observed returns and existing allocations completing before delayed growth
was released with the new scope. Neither harness exercised a game scene.

For a game comparison, follow [the test procedure](tests.md).
Use the same save, route, camera, resolution, upscaler, frame generation and
cache seeds. Record and review the entire visual run. Compare rendered frame
time percentiles and long-frame counts alongside image quality; do not infer
gameplay gains from a pool microbenchmark.
