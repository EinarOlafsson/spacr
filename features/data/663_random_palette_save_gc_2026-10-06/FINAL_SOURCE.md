# Final-source Random-palette Save replay, 2026-10-06

The same `reproduce.py` was replayed in a new isolated checkout of
`930455861c7b6f9fbac9820c8edfe44e496c40cc`, after the Random-cache,
no-backdrop and Preferences row-help changes. Its complete process output is
`xvfb-4k-930-final.log`. The source SHA-256 values were identical before and
after the process:

| File | SHA-256 |
| --- | --- |
| `spacr/qt/widgets/ambient.py` | `a324d395c4029fdeb1e307595fcc4d2353cb4fd7f15589f51e456c67f19b417a` |
| `spacr/qt/preferences.py` | `733f9af071745b2d170ed743fe96130b6b346d5f101c8a87d0e3b3b758ff21ee` |
| `spacr/qt/app.py` | `5fc990e6904525432af3e6fb2b419342af2dabd2776cc371e0cd55ed3dc543f7` |

The Xvfb screen was 3840 × 2160 with a hard 4 GiB cgroup, CUDA hidden,
PySide6 6.11.2 and a new private `QSettings` file. The real MainWindow
visited Home, Make Masks, Measure and Annotate. Three real modal Preferences
Saves persisted detail `200 → 100 → 200`, gravity radius `0 → 0.65 → 0`,
and palette `Random → spaCR → Random`. At radius 0.65, the actual cursor poll
and running engine pointer both read `(0.49986979, 0.49976236)`; the engine
radius was 0.65. At zero radius the engine pointer was cleared.

The process exited zero. Six natural GC callbacks all ran on the GUI thread.
After MainWindow close and deferred deletion, there were zero Qt widgets,
zero AmbientWidgets and zero shading workers. Peak `VmHWM` was
1,182,180 KiB. `RssAnon` after close was 819,008 KiB; this is remaining
process/allocator state, not evidence of reclaimed heap.

The installed-app Detail/Save segmentation fault remains **open**. This
bounded run did not reproduce it or provide its missing native stack.
