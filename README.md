# Vice-Tay

[VIctory](https://github.com/Marvisak/VIctory), a compatibility layer that runs
an Xbox Series title natively on Windows from files the user supplies, tuned and
fixed on one PC: an **RTX 4090**, a **Ryzen 9 9950X3D** and **48 GB of
RAM**, with a 4K screen. The aim is the world at 30 frames a second with the
picture right, fully playable.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

It is published on its own, not as a GitHub fork, with the whole history of the
trees it is built from: it started from [codec-d/cool_fox](#credits) (now taken
down), itself [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)
with work for NVIDIA cards on top. Our fixes come over in batches; what each one
changes is in [docs/this-fork.md](docs/this-fork.md).

## Prison gameplay test (maintainer report)

On the PC below, I observed **20–25 fps** while walking through the prison and
navigating the in-game menus. I ended the session manually after **31 minutes
and 22 seconds**, with no crashes.

### Test PC

| Component | Configuration |
| --- | --- |
| CPU | AMD Ryzen 9 9950X3D, 16 cores and 32 threads |
| Motherboard | MSI MAG X870 TOMAHAWK WIFI (MS-7E51), revision 1.0 |
| BIOS | MSI 1.A93 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB GDDR6X |
| NVIDIA driver | 621.40 |
| Integrated graphics | AMD Radeon Graphics |
| RAM | 48 GiB DDR5 (2 × 24 GiB), configured at 8,000 MT/s |
| System SSD | Samsung 990 PRO 4 TB NVMe |
| Additional SSDs | Samsung 990 PRO 2 TB NVMe and Samsung 970 PRO 1 TB NVMe |
| Monitor | Samsung Odyssey OLED G8 (G81SF), 3840 × 2160, 240 Hz |
| OS | Windows 11 Pro 26H2 x64, build 26300.10075 |

## Earlier Series X measurements (2026-11-26, 3840 × 2160)

| | |
|---|---|
| Menus, videos, loading screens | 30 fps |
| Prologue, indoors and the yard | 12-15 fps (median 12.9-13.2 over 6-8 minute sessions) |
| A frame in the world | about 75-80 ms: 38-44 ms on the GPU, about 28 ms preparing (12.6 ms of copies, ~105 MB), 23-25 ms copying back |
| Stability runs | 8-10 minutes in the world, no crash, the character and camera answering every minute |
| Memory | about 37 GB of private memory: 48 GB of RAM with a large page file works |

These earlier runs shared the PC with concurrent builds and tests.

What is known to be wrong:

- Below 30 fps in the world; the GPU time and the copies back are the next targets.
- The ray-traced structures are not built by default (`-RayTracing` turns the
  builder on), so a few hundred dispatches a frame are skipped and some light is missing.
- Some faces show scrambled colour patches, some eyes are lit white, and a few
  windows sparkle.
- The first run with a new build translates every shader while it plays
  (`victory shaders` does it ahead, see below).

## How to run it

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders, Rust (the toolchain is pinned), the MSVC build tools
with a Windows SDK, and your own dump of the title.

```powershell
git clone https://github.com/TaynanGT/Vice-Tay
cd Vice-Tay
copy victory.example.toml victory.toml      # then set your paths, outside this folder
cargo build --locked --profile diag -p victory --features diagnostics
scripts\play.ps1 -Exe <path of your copy's executable>
```

`Play-VIctory.cmd` does the same: `Play-VIctory.cmd -Exe <path>`.

- **Full screen** is the default (frameless, the whole monitor). **F11** or
  **Alt+Enter** switches to a window and back while it runs; `-Windowed` starts in
  a window.
- `-Machine` picks the settings by the graphics card: `4090` for any NVIDIA card
  (thread counts and switches measured on the RTX 4090 / Ryzen 9 9950X3D),
  `7900xtx` otherwise.
- `-Console series-s` makes the title take the PC for a Series S: a smaller
  picture, lighter on the card. `series-x` is the default and looks better.
- `-Safe` leaves out the ways still on trial; try it first when something is wrong.
- `-RayTracing` turns the title's ray-traced indirect light on.
- `-Build release` runs `target\release\victory.exe` (`cargo build --release --locked -p victory`).

Before the first run, ahead of time (both can be stopped and go on later):

```powershell
target\diag\victory.exe shaders <the shader archives>   # every shader translated, minutes
target\diag\victory.exe warm --used                     # pipelines earlier runs drew with
```

Commands, debugging switches and the upstream descriptions:
[docs/README-fxceless.md](docs/README-fxceless.md) and
[docs/findings.md](docs/findings.md). Every `VICTORY_...` switch is described in a
comment where it is read.

## Credits

This is a tree of other people's work with more on top. The history is kept, and
commits taken from elsewhere name the commit and its author. Everyone with a
commit in it is listed in [docs/credits.md](docs/credits.md).

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end and
  the shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)** and the
  people whose pull requests it carries.
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)**:
  list calls recorded on the title's threads and translated by workers,
  submissions prepared on threads of their own, cheaper write guards, textures on
  a copy queue, frame record and replay.
- **codec-d/cool_fox** (taken down by its author): two-plane composition, fallback
  heaps, table plans, the residency guard, fence wakes on `WaitOnAddress`, wave32
  lowering for NVIDIA, movie-plane uploads and XMA audio fixes.
- **[DexNom/VIctory](https://github.com/DexNom/VIctory)**: ray tracing fixes,
  cube gradients, sparse textures.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  the ambient light idea, pipeline workers, and work by romainhedouin.
- **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)**: the guard over
  every view of a page (with Spexivus, joroyt and tightfall), and Joshua Odom's
  reads into title memory.
- **Geckomacabre** and **cxy0072019-sketch**: the texture read-back.
- **[XCL](https://github.com/ivall/xcl)** (GPL-2.0-only) was read for how the same
  title behaves under another layer; no code of it is here.

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE); the one file that is not
covered is named in [docs/asset-policy.md](docs/asset-policy.md).
