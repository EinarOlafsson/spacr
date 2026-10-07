# N663 native Aurora surge paint, 2026-10-07

The Aurora optimization is the two-commit source chain `1e05cd4e0e` then
`079591b7dd`, based on `ae7269a08c`. It draws the cached surge image through
the original antialiased sheet with `Qt.IntersectClip`, preserving an incoming
rectangle or path clip and the caller's painter state. The final class AST in
the later integrated `11fc2f2a1b` tree is identical to the class at
`079591b7dd`; unrelated growth source changed the full module blob.

`source-abba.json` is the original native 3840×2160 ABBA run on committed
`1e05cd4e0e`. `source-abba-final-079.json` repeats it on the integrated
source with the exact final Aurora class: default old medians 42.31/45.29 ms,
new 39.26/39.76 ms; Density 3, Size 3, Detail 2 old 213.68/221.25 ms,
new 192.54/196.05 ms. These are direct `shade` timings under one 4 GiB
CUDA-hidden, offscreen CPU process, not a live-widget FPS claim. The final
high-control case remains far slower than the requested native 24 FPS.

`parity-final-079.json` compares the baseline `ae7269a08c` Aurora method to
the integrated final class in 36 complete images. All 24 native 4K images
are byte-identical. Twelve 1280×720 images differ in at most 47 pixels by
one byte. `parity-clip.json` is the earlier prototype comparison against the
same baseline and is retained separately; it is not the final-source receipt.
The exact source and class hashes, interpreter versions, counts and timing
limits are in `receipt.json`. `clip-negative.log` records two meaningful
incoming-clip failures before the final correction; `clip-positive.log`
records both passing afterward. `clip-and-ownership.log` covers those plus
frame ownership. `focused-tests.log` records 44 passing Aurora cases. The
branch coverage JSON and profile scripts/raw receipts are preserved as
separate payloads. No pixel, control, density, geometry or material limit was
reduced.

To reproduce in an isolated worktree containing the three source commits,
set `PYTHONPATH=.` and run `benchmark_source_abba.py` and `parity_final.py`
with CUDA hidden, `QT_QPA_PLATFORM=offscreen`, and `tools/run_capped.sh 4G`.
The scripts use `git show` to load the original method into the exact current
engine. The archived measurements used Python 3.12.13, NumPy 2.5.2 and
PySide6 6.11.2. Run `verify_manifest.py` from the repository root to check
every payload and the Git source/class identities. Human visual acceptance,
installed Save-crash resolution and hard native 24 FPS remain open.
