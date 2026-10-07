2026-10-07: spaCR field explicit background hold-drag CPU checkpoint

Production source: 397c8acb5d, parent ce58f867e4.
ambient.py SHA256: 07225441bdce84a1347bdcab2b7c611f1dff108372c25db10edcc36f49c6b452.
This individual checkpoint precedes integration with other agents' theme edits.

Left press/hold/drag on the field's host or plain backdrop containers offers
an immutable local material handle. Target and material displacement are bounded
to 18% of the shorter screen edge; local influence fades to zero at 34%.
The analytic critically damped spring uses frequency 6 per animation second;
release retains displacement/velocity, and material returns without reseeding.
Hover gravity is independent, including radius zero and a live change to zero.
Hide, theme change, restart, resize, deactivation and missing-button movement
cannot leave a stale held handle. Leaving while held preserves the handle until
release, including release outside the host. Application events are observed,
never consumed. The actual hit child prevents ignored QLabel/canvas events
propagating into an accidental background grab.

93 tests in seven affected files pass in 20.43s. Four new private helpers cover
62 executable statements and every branch destination. Separate package
callable/nested-doc guards pass 90/90 in 25.29s. These cohorts overlap other
receipts; do not sum them into unique acceptance or claim complete ambient
coverage. Compiler/frame queue contracts are retained. The tests include actual
button/text input and a real MaskCanvas box gesture, custom Screen host, DPR1/2,
partition-independent constant-target spring, same-clock continuity, owned native
frames, bounded latest-state worker offers and nonblocking busy-worker input.

Eight complete native 3840x2160 BEFORE/AFTER idle pairs preserve every byte:
dark/light, default/Random/custom and Random Density3 with pointer/click gravity.
Native held state changes 375871 pixels versus the ungrabbed same-clock field;
release does not jump; four animation seconds restore exact same-clock seeded
pixels. Three lossless native stills are actual engine renders. Default grain
count, palette, native Detail and density are retained. The optimization kernel
is deliberately bypassed for deterministic NumPy scatter fallback parity.

Actual native-size Random widget uses its real producer and Qt event loop,
requested24 cap, Detail2/Density1/hover radius0. Cold synchronous show67.44ms;
press/move delivery0.394ms; 469 heartbeat samples p95/max14.84/24.11ms; retained
initial QImage remains independent/alphaFF and producer retires on close.
Peak proof process RSS387272KiB under4GiB cap. This is input/lifetime proof,
not a controlled native cadence benchmark or hard24FPS acceptance. Cold show
is separate from steady heartbeat. Offscreen cannot establish desktop stacking,
human aesthetics or the historical native Save SIGSEGV. GPU remains owner-only.

An initial harness used QTest.qWait, which retains the GIL between its Python
callbacks and prevented the producer advancing. The accepted harness uses a
real QEventLoop; that harness correction is not an application clock fix.
A first script launch lacked explicit PYTHONPATH and imported the installed
checkout; that rejected run is not accepted source-current parity. The accepted
script asserts its actual module path and records exact source hashes.

Reproduction, from an isolated worktree at397c8acb5d with the accepted environment:
  mkdir -p /mnt/wd4tb/scratch/field-grab-replay-20261007
  cp features/data/663_field_grab_cpu_2026-10-07/native_probe.py /mnt/wd4tb/scratch/field-grab-replay-20261007/
  CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python /mnt/wd4tb/scratch/field-grab-replay-20261007/native_probe.py
The script writes only beside its scratch copy; do not run it in this archive.
Focused pytest command: use the seven files in acceptance.json with
  CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest -q -p no:randomly --cov=spacr.qt.widgets.ambient --cov-branch --cov-report=json:/mnt/wd4tb/scratch/field-grab-replay-20261007/coverage.json --cov-report= <files>

Verify payload hashes from filesystem or committed Git blobs:
  python features/data/663_field_grab_cpu_2026-10-07/verify_manifest.py
  python features/data/663_field_grab_cpu_2026-10-07/verify_manifest.py --git HEAD
