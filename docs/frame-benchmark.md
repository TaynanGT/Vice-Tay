# Buffered frame benchmark

To capture a normal session, create an output directory outside the checkout and
pass `-More @{VICTORY_FRAME_CSV='C:\Bench\frames.csv'}` to
`scripts/play.ps1`. The CSV path must be absolute and new; its parent must
already exist. Capture works in diagnostics and release builds, with the
performance panel shown or hidden. It is off unless this variable is set.

Each row records one successful host Present of a rendered frame. The first row
of each presenter lifetime establishes a baseline and leaves interval fields
empty. Later rows contain elapsed frame time, draws, dispatches, submission
preparation, GPU waits, read-back work and bytes, pending pipelines, shader-cache
misses, texture-upload totals, and separate frame-generation/output counters.
`gpu_wait_ms` measures CPU waiting, not GPU execution. Shader-cache misses count
translations, not all pipeline creation. Upload and generation totals are
cumulative observations; preparation and read-back timings describe work observed
between samples, which can belong to submissions crossing frame boundaries.

Samples use an 8,192-entry queue and background buffered writer. Queue overflow
counts as lost data; the benchmark rejects it. Orderly shutdown closes producer
admission, drains accepted samples, syncs the CSV, then writes
`frames.csv.complete.json`. A crash or forced stop may leave incomplete output.
The completion marker proves admitted samples were drained; it does not prove
that every attempted title Present reached the display. Use external presentation
tracing when display cadence needs investigation.



This benchmark launches each repeat as a new process, with new copies of frozen data and app cache seeds. It does not reset Windows or GPU driver caches. Keep the scene, camera, resolution, upscaler and frame generation settings the same for comparisons. Record every visual run from start to finish, review the full video, and note timestamps for scene changes and artifacts. Compare colour, lighting, textures, reflections, HUD and temporal artifacts alongside frame metrics.

1. Build the runtime. Prepare full copies of data and app cache seeds in directories outside the checkout and any `.victory-root`. Do not use live player directories as seeds. Keep these seeds unchanged. Choose a new, absent results directory for each recipe.
2. Freeze launch settings without starting the title: `pwsh scripts/play.ps1 -Exe <absolute-title-exe> -SkipShaders -ExportEnvironment <new-absolute-frozen-launch.json>`. Pass the same `-Console`, `-Machine`, upscaler, frame generation and `-More` settings intended for the benchmark. `-SkipShaders` has no effect on export; shader warmup never runs in export mode. Export includes effective `VICTORY_*` variables, including inherited path settings, plus runtime, title, console and working directory. Review the JSON before use. The runner replaces data/cache/key/CSV paths for each repeat.
3. Copy `scripts/benchmark.example.json` to a recipe. Set all absolute paths and durations. `startup_keys` times are relative to process start; `route_keys` times are relative to measurement start. Keys are mailbox lines (`W 2.0`, `Enter 0.2`), published atomically, at least 50 ms apart. Consumption is observed; gameplay action is not proved. Set `ready_timeout_seconds` long enough for manual scene setup.
4. Run `pwsh scripts/benchmark.ps1 -Recipe <absolute-recipe.json> -ValidateOnly`. This scans full seeds, rejects reparse points and overlapping paths, hashes all seed files and binaries, and creates no run directories or process.
5. Start `pwsh scripts/benchmark.ps1 -Recipe <absolute-recipe.json>`. For each repeat, inspect scene. When scene ready, create empty `<output_root>\run-NNN\ready.marker` before ready timeout. This explicit marker means operator verified readiness; draw counts alone do not prove scene. Runner waits for warmup, measures, sends route batches, waits for one post-window frame to prove coverage, closes owned process window, then requires exact CSV completion and summary. Tail wait uses `close_timeout_seconds`; tail frames stay outside metrics. Timeout, nonzero exit, forced exit, missing marker or dropped frame invalidates run. Relaunch of the runtime into a child process invalidates the run. Failed evidence stays in its run directory. A later repeat starts only after prior one succeeds.

`manifest.json` records launch and effective environment, binary/recipe/seed hashes, Git HEAD and dirty diff hash, GPU driver, CPU, UTC window and shutdown. `frames.csv.complete.json` is written by capture writer only after drain and sync. `summary.json` counts successful rendered host Present intervals fully inside UTC window, excluding each stream's baseline and boundary crossing intervals. FPS is `1000 × interval_count / sum(frame_ms)`. One percent low FPS is `1000 / mean(slowest ceil(1% of intervals) frame_ms)`. Median, p95 and p99 use sorted frame times and nearest-rank percentiles. `over_50_ms`, `over_100_ms` and `over_budget` are strict `>` counts. Frame generation counts, when present, are separate observations, not added to rendered FPS.

Frame samples describe observed work between host Presents. They are not precise GPU attribution for each rendered image. Manual readiness and timed key routing support repeatable procedure, not deterministic gameplay. No live performance or visual quality is established by synthetic script tests.
