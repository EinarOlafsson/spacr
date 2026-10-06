# N663 Random-palette MainWindow Save and shutdown probe, 2026-10-06

`reproduce.py` and `xvfb-4k-final.log` are the exact final script and process
output. The isolated checkout was
`a8874e89baf80169755d1c81c0e5a3e8e1f485e0`. The three product files
had the same SHA-256 before and after the process:

| File | SHA-256 |
| --- | --- |
| `spacr/qt/widgets/ambient.py` | `7a35b8e63e05ea786f2fe1f11602111d85aaf881b871685ed86da21dada0f138` |
| `spacr/qt/preferences.py` | `1f0fbab1888823a643fb7b1e60eae82e17ec3a4b1aa674c163c0a69bc2a8c5f0` |
| `spacr/qt/app.py` | `991dcaaa4cbf331b76216d6fb816e57340b055b541e838dfbd0bc0e625e4be82` |

The process used Xvfb at 3840 × 2160, PySide6 6.11.2, a hard 4 GiB cgroup,
hidden CUDA, and a new private `QSettings` INI. It opened the real MainWindow,
visited Home, Make Masks, Measure and Annotate, then saved the real modal
Preferences dialog three times. Animation detail was 200 → 100 → 200,
mouse-gravity radius 0 → 0.65 → 0, and palette Random → spaCR → Random.
All saved settings matched the selections. With the window activated, the
actual Xvfb cursor poll yielded approximately `(0.49987, 0.49976)`; at radius
0.65 the running engine's pointer matched and its radius was 0.65. At radius
zero the engine pointer was cleared.

The process exited zero without a native fault. Eight natural GC callbacks
all ran on the GUI thread. After closing the MainWindow and delivering
deferred deletes, `QApplication.allWidgets()`, AmbientWidget count and live
shading-worker count were all zero. Peak process `VmHWM` was 1,293,192 KiB.
After close, `RssAnon` remained 1,005,656 KiB, which is allocator/process
state and is not reported as reclaimed memory.

The reported installed-app Detail/Save segmentation fault remains **open**:
this bounded Xvfb process did not reproduce it and supplies no native stack
or causal fix. This is a targeted source-bound lifetime check, not a full Qt
or GPU acceptance run.
