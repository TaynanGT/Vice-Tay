# NVIDIA DLSS Frame Generation backport

This adds optional NVIDIA DLSS Frame Generation / Multi Frame Generation to
Vice-Tay's existing upscaler and presentation path. The existing AMD provider and
defaults remain unchanged. DLSS Super Resolution continues to use the existing
NGX integration; frame generation can also observe the title's own upscaler.

## Source and adaptation

The Streamline runtime, public ABI and camera conversions are adapted from
[trustno1-coder/VIctory-NVIDIA at c437b059](https://github.com/trustno1-coder/VIctory-NVIDIA/tree/c437b0594bde23780e15c24969258ad9d70bb7be),
which builds on
[readmet3xt-lab/foxhound at 02fc18a2](https://github.com/readmet3xt-lab/foxhound/tree/02fc18a20a1e1771374babcb44084f8de3275b96).
The original files are `crates/victory-gpu/src/streamline.rs`,
`streamline/abi.rs` and `streamline/plan.rs`, under GPL-3.0-or-later.

The backport adapts those interfaces to Vice-Tay's external recordings and
submission tokens. It does not import the source fork's renderer, image patches,
launcher defaults, settings menu, or lighting approximations.

## Setup and controls

NVIDIA's runtime is optional and is not redistributed in this repository. Run
`scripts/get-streamline.ps1 -Directory <absolute-external-directory>` to obtain
the official Streamline SDK v2.14.16 release DLLs and their licences. The helper
requires a directory outside this checkout and verifies valid NVIDIA signatures.
Do not replace runtime DLLs while the game is using them.
The engine also enforces the external-directory rule for SDK files and logs;
SDK-generated data defaults to the selected external runtime directory.

Use the existing `scripts/play.ps1` and its `-More` settings:

| Setting | Meaning |
|---|---|
| `VICTORY_FRAMEGEN=1` | Request frame generation; zero disables it |
| `VICTORY_FRAMEGEN_PROVIDER=nvidia` | Select NVIDIA; unset or `amd` retains the AMD provider |
| `VICTORY_FRAMEGEN_X=4` | Request four shown frames per rendered frame; 2–6 accepted, limited by the GPU and driver |
| `VICTORY_STREAMLINE_DIR` | Absolute external directory containing the signed NVIDIA runtime |
| `VICTORY_STREAMLINE_LOG` | Optional external directory for verbose SDK logs |
| `VICTORY_TEXTURE_ROOM_PERCENT=80` | Optional room for generation on memory-constrained cards; not a new default |

`play.ps1` clears most inherited `VICTORY_*` environment variables. Put provider,
multiplier, SDK directory and other overrides in `-More`. Keep its existing
Series S console and machine selections. `-Dlss quality` selects the existing
NGX upscaler; `VICTORY_DLSS_DIR` can point at the same SDK directory.
`VICTORY_DLSS_NO_UPDATE=1` makes that existing NGX path use the selected runtime
without its updater, useful for reproducible comparisons.

NVIDIA generation requires an RTX 40-class or newer GPU, a supporting driver,
and hardware-accelerated GPU scheduling. Multipliers above 2 require supporting
RTX 50-class hardware and driver. Capability queries report the actual adapter's
support. Nothing here enables scheduling, changes driver settings or downloads
files during play. An unavailable runtime logs the reason and presents ordinary
frames; choosing `amd` remains an explicit fallback.

## Resource and UI handling

The direct queue and swap chain are created through Streamline's manual proxies;
the renderer retains the native queue. Upscaler inputs are associated with their
first submission token and only become eligible after their command list is
submitted. Dropped recordings, mismatched tokens and already consumed inputs
cannot supply another present. Resource references survive GPU submission, and
tags use the host states present at presentation.

This port supports the separate Series S scene and UI planes. It creates a
postprocessed HUD-free scene and an independently composited UI mask matching
the output dimensions, while presenting a fully composed final image. The
performance panel is copied into both the real image and the UI mask. Existing
stock postprocessing is used; no shadow or ambient-light change is required.

Frames without fresh valid upscaler inputs, camera cuts, or missing separate UI
disable interpolation. Resize disables it and clears retained inputs. Shutdown
drains submitted work, releases the proxy presenter, then calls `slShutdown`
while the D3D12 device is still alive. No Series X single-plane HUD extraction
or F1 live NVIDIA settings menu is included.

## Counters and validation

`NVIDIA frame generation: driver-reported output` reports rendered FPS, total
presented FPS, the startup request, the last actual multiplier and SDK status.
Total presented FPS includes original and generated frames. Status zero and
actual multiplier are needed to distinguish active interpolation from a startup
request. These are SDK counters, not physical display measurements or evidence
of faster simulation or reduced input latency.

An existing combined-build RTX 5070 run had 240 active one-second samples with
actual multiplier 4 and status 0, averaging **19.08 rendered / 75.69 total
driver-reported presented FPS**. It included separate local lighting changes and
is an observed run, not a matched before/after benchmark. Its 60 FPS recording
cannot show every output frame. A combined-build synthetic test reported 180
rendered / 696 presented frames including warm-up, a healthy device, status 0
and SDK shutdown 0. Those observations do not replace clean-backport testing.

A later clean-backport synthetic check reported **180 rendered / 672 presented
frames**, with 180 generation-enabled presents, last multiplier 4, status 0,
a healthy device and SDK shutdown 0. The signed SDK's adapter capability check
also passed. These synthetic inputs establish SDK operation, not game image
quality or a performance benchmark.

A separate stock-lighting game check used the same executable for AMD 2x and
NVIDIA 4x, DLSS Quality at **1280x720 to 1920x1080**, and an 80% texture budget.
In the NVIDIA recording, 22 one-second samples whose last multiplier was 4 and
status was 0 averaged **17.34 rendered / 66.35 total driver-reported presented
FPS**, including mode transitions. Individual steady samples included
**21.67 / 86.70** and **20.15 / 80.59**. SDK shutdown returned 0. These are
observed counters from an RTX 5070 with driver 610.88. Exact camera and route
equivalence has not been verified, and full temporal review is pending; no
before/after speedup is established.

That recording requested 120 FPS but contains 7,806 captured frames in
103.391 seconds, approximately **75.50 captured FPS**. Capture sampling cannot
show every generated output frame or independently establish display cadence,
input latency or simulation rate. The comparison recordings remain local for
review; no visual-quality result follows from the counter checks alone.

The hardware checks are opt-in tests in `streamline::tests`:

- `nvidia_adapter_requirements` loads the signed SDK and checks support without
  opening a presentation window.
- `nvidia_synthetic_present` opens a synthetic presentation window. With
  `VICTORY_FRAMEGEN=1`, provider `nvidia`, multiplier `4` and the external SDK
  directory, it requires more than 360 shown frames for 180 rendered frames,
  status 0, healthy device state and successful shutdown.

Run those explicitly with `--ignored --nocapture --test-threads=1`. The ordinary
unit tests cover ABI layouts, camera conversions, input freshness, token binding
and dropped-recording bookkeeping without requiring NVIDIA's DLLs.

Camera projection matrices are reconstructed from upscaler parameters; the
title's original view matrices are unavailable. A matched recorded comparison
must still check moving edges, disocclusion, phone/UI brightness, subtitles,
map/HUD stability, loading/menu transitions, resize, rendered frame times and
output cadence. Raw captures and other derived title data stay outside this
repository under the asset policy.
