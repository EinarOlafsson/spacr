# Random-palette owned-frame MainWindow replay, 2026-10-06

`reproduce-alpha.py` and `xvfb-4k-9ed-alpha.log` capture the actual bounded
run on clean isolated source
`9ed9688b0b714993b9954c584664e49df9adf5eb`. Product SHA-256 values
were identical before and after the process:

| File | SHA-256 |
| --- | --- |
| `spacr/qt/widgets/ambient.py` | `ac146d3c61bb73def9a08857eeb1fca8e7163de30b1b8892ba3720857244ec20` |
| `spacr/qt/preferences.py` | `733f9af071745b2d170ed743fe96130b6b346d5f101c8a87d0e3b3b758ff21ee` |
| `spacr/qt/app.py` | `5fc990e6904525432af3e6fb2b419342af2dabd2776cc371e0cd55ed3dc543f7` |

The process used Xvfb at 3840 × 2160, a hard 4 GiB cgroup, hidden CUDA,
PySide6 6.11.2 and a new private `QSettings` file. The real MainWindow
visited Home, Make Masks, Measure and Annotate. Three real modal Preferences
Saves persisted detail `200 → 100 → 200`, radius `0 → 0.65 → 0`, and palette
`Random → spaCR → Random`. At radius 0.65, the actual cursor poll and the
running engine pointer both read `(0.49986979, 0.49976236)` and the engine
radius was 0.65; at zero radius its pointer was cleared.

The owned published producer frame was held as a `QImage` while its bits were
read. Its alpha was `FF` at every raster word: zero nonopaque words in
8,079,360 pixels at the first Random Save (3840 × 2104), and zero in
5,580,800 pixels at the final Random Save (3200 × 1744). The intervening
spaCR-palette frame was also fully opaque.

The process exited zero. Seven natural GC callbacks all ran on the GUI
thread. After MainWindow close and deferred deletion, Qt widgets,
AmbientWidgets and live shading workers were each zero. Peak `VmHWM` was
1,323,672 KiB. After-close `RssAnon` was 1,026,144 KiB, which remains
process/allocator state rather than reclaimed-memory evidence.

The installed-app Detail/Save segmentation fault remains **open**: this
bounded source-bound run did not reproduce it or establish its cause.
