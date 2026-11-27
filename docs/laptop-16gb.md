# A 16 GB laptop: `-Machine laptop`

`scripts\play.ps1 -Machine laptop` gives these settings. Ask for them explicitly; `-Machine auto` doesn't pick them, because they were measured on one laptop:

| Part | |
|---|---|
| CPU | Intel Core i7-13700H (6 P-cores + 8 E-cores, 20 threads) |
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop GPU, DXGI budget 5,233 MB, Resizable BAR on; the Intel iGPU is adapter 0 (the device takes the RTX on its own) |
| RAM | 16 GB, a system-managed page file |

## What runs out, and when

On the 4090 settings, the Series S profile commits 15 to 18 GB of private memory once the world streams in. On 16 GB, **free RAM runs out while the world loads**: within about 20 s, the memory the layer shares with the card goes from under 1 GB to 4–5 GB, faster than Windows writes other pages out, and the PC starts paging everything, the desktop included. Once the world is up, free RAM holds at 0.5–1.2 GB. Here the card's own memory is not what runs out first; RAM is.

## The settings

On top of the NVIDIA settings (threads, the wave-wide list, placed order), which every measured run had:

| Setting | Why |
|---|---|
| `VICTORY_WORKING_SET_MB` = RAM − 11.6 GB, at least 3 GB (4,500 on 16 GB) | A hard ceiling on the game's working set: past it Windows trims the game's own pages, not the rest of the PC's. |
| `VICTORY_DSTORAGE_IO=direct` | Game files read past Windows' file cache, which would otherwise hold gigabytes of RAM. |
| `VICTORY_INPLACE=2` | Title buffers in its private memory used in place, not copied. |
| `VICTORY_WINDOW_KB=64`, `VICTORY_WINDOW_TAILS=1` | Windows outside buffers cut on 64 KiB: less copied and locked for the card. |
| `VICTORY_WINDOWS_BAR_MB=3072` | Up to 3 GB of those windows in the card's memory (Resizable BAR) instead of RAM. |
| `VICTORY_CARD_BUFFERS_MB=256` | The card's own copies of title buffers kept small, so the windows have room on the card. |
| `VICTORY_LEVELS_ON_DEMAND=1` | Texture levels get the card's memory as they're written. |
| `VICTORY_OUTPUT=1920x1080`, `VICTORY_UPSCALER=6`, `VICTORY_RENDER_SCALE=0.5` | 1080p output drawn at half size by the title's own upscaler: the card stays at about 4.2 GB of its 5.2 GB budget, and the GPU's share of a world frame is small. |

The preset sets no RAM throttle and leaves the title's heartbeat watchdog alone: with these settings the world load passes with the watchdog on (below). The experimental changes in [#96](https://github.com/TaynanGT/Vice-Tay/pull/96) and [#126](https://github.com/TaynanGT/Vice-Tay/pull/126) are separate from the preset (see "Longer sessions").

All of them stay under `-Safe`, since without them the world load runs out of RAM. Any one can be overridden with `-More`.

## Measurements

Each run is 240 s on the Series S profile, windowed, with the title's watchdog active, under an external guard that stops the game below 400 MB of free RAM. Scripted Enter presses reach the story load in 60–100 s. "World" counts frame reports with 300 or more draws.

| Build and settings | Runs | Passed the world load | In the world | World median frame | Ended by |
|---|---|---|---|---|---|
| `main`, `-Machine 4090` | 1 | 0 | – | – | the guard, at the load (3 MB free) |
| `main` + this, `-Machine laptop` | 2 | 2 | 161 / 157 s (262 / 248 frames) | 881 / 716 ms | the guard near the end (235 / 240 s; 95 / 320 MB free) |

The game sets its own ceiling in every run (`working set ceiling 4500 MB: set`), and its working set stays at 4,500 MB; nothing outside the game sets one.

### Longer sessions

Without a throttle, free RAM in the world touches a few MB at moments, and the guard stopped both runs above after about 160 s in the world. The RAM throttle (#96) keeps free RAM above about 400 MB through 240 s runs, but its waits hold the title's main loop back. In the earlier #96 tests, the watchdog stopped all six throttled world loads, with submission waits of 1,000, 300 or 0 ms. #126 lets the throttle tick the title's heartbeat during its own waits. With #96 and #126 (and the throttle at a 1,500 MB floor, 500 ms per buffer, 900 ms a second, 1,000 ms per submission, windows on the card up to 92 % of its budget), 4 of 4 loads passed, and 3 of them ran the full 240 s (one met the guard 55 s into the world): about 176 s in the world, 112–122 frames, a median frame of 1.37–1.47 s.

### Watchdog-on runs with the list pool disabled

On the same 16 GB laptop, [seven later runs](https://github.com/TaynanGT/Vice-Tay/pull/147#issuecomment-7213344061)
used Series S and `-Machine laptop` with
`-More @{VICTORY_LIST_POOL='0'; VICTORY_OUTPUT='1280x720'; VICTORY_RENDER_SCALE='0.75'}`.
The control was `main` at `791648b`. The experimental #147 build used a 600 MiB
RAM floor, a 250 ms per-buffer wait and a 600 ms shared wait per second.

| Setting | Passed the world load | World median frame |
|---|---:|---:|
| Control | 3 of 3 | 548–881 ms |
| Buffer waits | 2 of 2 | 1,059–1,150 ms |
| Buffer waits plus a 300 ms submission wait | 1 of 2 | 687 ms in the completed run |

The last run ended at the watchdog after heavy paging, with one submission
wait and no buffer waits. The throttle remains experimental. The launcher
prints the executable path and SHA-256 so each recorded run can be tied
to the binary it used.

### Measured options and dead ends (on top of the above with #96 and #126, 2 runs each)

| Change | World median frame | Result |
|---|---|---|
| none | 1,471 / 1,455 ms | – |
| `-More @{VICTORY_OUTPUT='1280x720'; VICTORY_RENDER_SCALE='0.75'}` (the same 960x540 inside, smaller output targets) | **1,141 / 1,134 ms** | about a fifth faster, same RAM; a 720p picture |
| `VICTORY_WORKING_SET_MB=5000` | 1,545 / 1,593 ms | no faster |
| `VICTORY_D3D12_DRED=0` and `VICTORY_UP_QUEUE=1` (faster on an RTX 3080 Ti, #115) | – | the guard stopped both runs for low RAM: they cost RAM 16 GB doesn't have |
| `VICTORY_TEXTURE_ROOM_PERCENT=75` | 1,210 / 1,263 ms | no more windows on the card, less free RAM |

Identical settings vary about 2× in frame counts between runs, so these numbers show the runs survive, not how fast they are.

## Optional upload guardrails

Speculative texture uploads now leave the existing 1 GiB commit reserve **after** subtracting the requested new staging storage: a full 64 MiB pool block, or the calculated standalone upload footprint. Existing pool space can still be reused, and a smaller standalone upload can fit even when a new pool block cannot. Deferred uploads remain retryable. Table-bound textures and decoration bakes are not treated as speculative uploads, because dropping those uploads would leave required textures uninitialized.

These are sampled admission checks, not a hard cap on concurrent process or system allocations. They preserve the existing pressure hysteresis and allocation-failure cooldown; they do not shrink required title allocations, guarantee that a world load fits, or correct an unrelated null dereference.

## Validation on the RTX 2060 host

The following is a host snapshot, not a game-run measurement. On 2026-11-27 at 16:07:51 +01:00, Windows reported 16,326 MiB physical RAM, 8,157 MiB free, and system commitment of 12,254 / 25,030 MiB (12,776 MiB headroom). The system-managed page file was 8,704 MiB allocated, 308 MiB in use, and 1,586 MiB peak. The laptop preset reported `VICTORY_WORKING_SET_MB=4702`; that limits the game's resident working set, not its private commit or total system commitment. The page file was left unchanged. Earlier Series S measurements in this document are from another machine and do not establish this host's peak requirement.

The matching launch log records a `host memory pressure` warning with 3,024 MiB of available commit and 7,702 MiB of free physical RAM. That is evidence of limited system headroom, but not a failed title allocation or proof that memory pressure caused the crash. The crash report's `TileHeap` value (174 MiB used of 6,236 MiB) is allocator accounting, not proof that the same amount of physical RAM was committed. The repeated `0xc0000005` null read at `victory-image-…+0xb5a5f8` occurs in `Game Init`, frame 0; its cause remains unproven.

`victory.toml` still names the old game directory. This corrected-path preparation command was verified with `-DryRun` only:

```powershell
.\scripts\play.ps1 -Exe 'C:\Misc\Torrents\downloads\vc\game_scarlett_master.exe' -Machine laptop -More @{ VICTORY_GAME_DIR = 'C:\Misc\Torrents\downloads\vc' } -DryRun
```

The title executable hashes to `E95020B2FA4A721DB3DF59FDECF75621730AC559D8835252F6C3B658D5A08993`. Before rebuilding, `victory.exe` and its symbols were preserved under the local, ignored `target\startup-memory-evidence-20261127` directory. That executable hashes to `2D6DCB9FBEF84B5B00D216254BCB59DAE85390E9F3FB24CFAAF439266D2088BC`, consistent with the abbreviated runtime hash in `claude-context.md`. The capture has no title-executable hash, so identity with the original game binary is not established. The diagnostic executable built after incorporating upstream through `5e588e8` hashes to `CEF75DF2C89A02E3EF4D9413BB82ACFF6718440471AB2A74477DC95D19F2AFFC`.

The `-More` plumbing was also dry-run checked one variable at a time for `VICTORY_ASYNC_SUBMIT=0`, `VICTORY_THIN=0`, and `VICTORY_MAP_THREADS=1`, always retaining the corrected `VICTORY_GAME_DIR`. Those `-DryRun` checks establish launcher precedence only; they do not exercise the title.

The inherited `VICTORY_THIN=1` setting initially made `resource::tests::buffer_data_reaches_the_host_texture` fail because thin lists intentionally have no host command list to retain the host texture. With `VICTORY_THIN=0`, the focused test passed and the GPU suite passed (489 passed, 43 ignored after incorporating upstream through `5e588e8`). The following checks passed after synchronization:

```powershell
$env:VICTORY_THIN = '0'
cargo test --locked -p victory-gpu
cargo build --locked --profile diag -p victory --features diagnostics
.\scripts\test-play.ps1
```

The launcher check mocks CIM and file queries, covers unavailable/partial memory data, inherited paths and explicit overrides, and never invokes an executable. `git diff --check` passed. Rust edition 2024 formatting checks still report differences in all four touched Rust files; formatting the default-branch versions also changes those files, so unrelated whole-file formatting was not applied.

No debugger was available on PATH or in the checked Windows SDK locations. The Agility SDK support exists in `apps/victory/build.rs` and `apps/victory/src/agility.rs`, but `target\diag\D3D12\D3D12Core.dll` was absent and that route was not tested. No usable full-session recording/external low-memory-stop workflow was established. Do not remove `-DryRun` until those safeguards are ready. Repeated startup, world loading, video review, frame-time/image-quality comparison, and the diagnostic controls remain unverified. No page-file, watchdog, console-identity, or image-quality setting was changed, and no startup correction is claimed.
