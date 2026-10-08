# N663/N679 CPU flow and field frame-stage profile

The probe executed the accepted private Home `64de91a0ff9c3330dc9799125a13b6ad235ecfe1` ambient source, Git blob `daec12de9f91f0408eb8919c8dcaf5e055253412`, with base Python 3.12.4, PySide6 6.10.0 and NumPy 1.26.4. It used `CUDA_VISIBLE_DEVICES=''`, `QT_QPA_PLATFORM=offscreen`, an isolated `XDG_CONFIG_HOME`, and `tools/run_capped.sh 8G`. The script first waited for the asynchronous packed-scatter compiler. Each configuration discarded three warm frames, then timed five successive `engine.shade()` frames with the animation clock advancing by 1/24 second. A separate `cProfile` frame followed selected 4K cases and is not part of the reported medians.

These are synchronous QImage generation timings, including NumPy calculation and rasterization. They exclude QWidget composition, screen presentation, event-loop cadence, GPU work, and application startup. They cannot establish live FPS or 24-FPS acceptance. The offscreen process peak VmHWM was 386,696 KiB for the ordinary-theme sequence and 408,784 KiB for the Spaceout sequence.

| Theme and active state | Display/detail/density | Median shade | Point material | Other frame work |
| --- | --- | ---: | ---: | ---: |
| Advection | 4K/1/1 | 18.5 ms | 14.3 ms | about 4.2 ms |
| Advection | 4K/1/3 | 31.9 ms | 23.7 ms | about 8.2 ms |
| Ordinary field | 4K/1/1 | 11.4 ms | 9.8 ms | about 1.5 ms |
| Ordinary field | 4K/1/3 | 20.6 ms | 16.2 ms | about 4.3 ms |
| Spaceout field, density wave plus vortex at clock 35 s | 4K/1/1 | 21.6 ms | 11.6 ms | deformation 8.6 ms |
| Spaceout field, same events | 4K/1/3 | 72.5 ms | 28.3 ms | deformation 33.5 ms |

At 4K density 1, changing Detail from 1 to 0.5 changed the buffer from 3840×2160 to 1920×1080 and advection/ordinary-field medians from 18.5/11.4 ms to 11.0/2.5 ms. Native 1080p advection Density 0.1/1/3 medians were 1.6/12.9/31.1 ms. Density and Detail therefore change actual CPU workload. Stage medians are calculated independently, so stage sums need not equal the frame median.

The point-material work includes full-image fill and scattering; the advection cProfile sample includes `numpy.ufunc.at`. With Spaceout effects active, `_bend_pointer`'s NumPy deformation becomes a second major cost at high density. These source-bound observations identify CPU stages to compare in workstation GPU profiling; no GPU implementation or acceptance is claimed.

The compressed payloads retain the exact source file, probe, complete JSON results, and raw stdout logs. `receipt.json` records hashes of both compressed and original bytes. Run `python verify.py` on files, or `python verify.py --git` after committing, to check the payloads and source identity.
