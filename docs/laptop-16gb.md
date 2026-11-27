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
| `VICTORY_RAM_FLOOR_MB=1500`, `VICTORY_RAM_WAIT_MS=500`, `VICTORY_RAM_WAIT_SHARE_MS=900`, `VICTORY_RAM_FRAME_WAIT_MS=1000`, `VICTORY_WINDOWS_BAR_PERCENT=92` | The RAM throttle (pull request #96; without it these do nothing). Under 1.5 GB free, new buffers in RAM and the title's submissions wait a little. |

The title's heartbeat watchdog stays on. A throttled world load can go 40–50 s without a frame, past the watchdog's 30 s; if it trips, add `-More @{VICTORY_RAM_KEEP_ALIVE='1'}` (#96). The throttle then sets the title's heartbeat flag during its own waits only, after checking the 1.0.2.1 code that sets it, so a hang anywhere else still ends the title.

All of them stay under `-Safe`, since without them the world load runs out of RAM. Any one can be overridden with `-More`.

## Measurements

Each run is 240 s on the Series S profile, windowed, with an external guard that stops the game below 400 MB of free RAM. Scripted Enter presses reach the story load in 60–100 s. "World" counts frame reports with 300 or more draws.

| Settings | Runs | Passed the world load | Lowest free RAM | World median frame |
|---|---|---|---|---|
| `-Machine 4090` | 1 | 0 (out of RAM) | 3 MB | – |
| the memory settings without the throttle | 2 | 1 (out of RAM once) | 114 / 2 MB | 819 ms |
| the memory settings with the throttle (#96) | 2 | 2 | 465 / 428 MB | 1,764 / 1,304 ms |
| `-Machine laptop` with #96 | 2 | 0: **the title's watchdog ended both** during the world load (`+0x23cf683`) | 679 / 588 MB | – |
| `-Machine laptop -More @{VICTORY_RAM_KEEP_ALIVE='1'}` with #96 | 2 | 2 (the full 240 s) | 31 / 41 MB, for less than a second | 1,471 / 1,455 ms |

Every run with `-Machine laptop` set only the game's own ceiling (`working set ceiling 4500 MB: set`); its working set stayed at 4,500 MB, and nothing outside the game set one. So on this laptop the throttled world load needs `VICTORY_RAM_KEEP_ALIVE=1`, and the preset leaves that to you.

With the throttle the world is slower but stays up: it trades speed for not running out of RAM. Identical settings vary about 2× in frame counts between runs, so these numbers say "it survives", not how fast.
