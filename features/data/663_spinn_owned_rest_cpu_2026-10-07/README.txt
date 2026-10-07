2026-10-07 — exact native spinn owned/resting frame CPU checkpoint

Source254f3dd39f, parent368e2cdf1d. BEFORE ambient40e7f77b88febd8589c6cab2232c539194bf13d9019b53f7ee67e1663ecb9445;
AFTER ambienta3d1fed1d18b323fec6859e6ea72bad4ce273c8b60e8e87823ed751d7965eaff.
The measured scratch candidate is byte-identical to this committed production
file. Frozen before/after source and separately labelled rejected format source
are compressed here. No other engine's drawing algorithm changes.

Profile first: actual3840x2160, Density3, Detail2, Size1, local radius.5:
2652 facets/1082 spinning,42,081,168 cached native tile bytes. Four profiled
frames attribute39–44ms/frame to QPainter.drawImage, about8ms to the painter's
Python geometry,2.5–3.5ms to the deep publication copy. Eight uninstrumented
baseline samples have medians54–57ms/p95≈65–66ms. This is direct shader profiling,
not a real display cadence promise. Actual native resolution was asserted;
Density remains3 independently of Detail.

Accepted optimization: spinn is intentionally static at pointer leave/zero
hover radius. Reuse its already-owned resting buffer, return a distinct shallow
QImage wrapper with Qt copy-on-write, and update the rotation timestamp on every
resting shade. Active spin paints into a new owned native QImage. All input/time
acknowledgement remains active. Controls invalidate existing material caches;
one boolean marker adds no raster or NumPy storage. The original tile count,
AA, bilinear spin, drawing order, colors and local physics remain exact. Caller
fill() and writable bits() detach safely; retained publications survive entry,
rest, failures, resize and subsequent shades. No additional resting full-screen
image is retained beyond the existing engine buffer.

60 complete native before/after transition pairs are exact: dark/light ×
default/Random/custom × idle/100s-later/entry/movement/leave/Detail/Density/Size/
palette/background changes. The 24 alternating benchmark pairs are another
phase, not24 additional unique control states. Warm dark shader medians at
Density3/Detail2: rest16.576→0.029ms; active53.233→47.827ms. Entry after a long
rest retains the original tiny step and angle, with no catch-up rotation.

Actual balanced BEFORE/AFTER/AFTER/BEFORE native widget probes use requested24,
Density3/Detail2, fixed normalized pointer offered on each active tick, real Qt
event loops and real engine worker. Rest producer samples before23.69/24.28 vs
after23.67/23.68FPS remain cap-paced; resting pixels intentionally do not move.
Distinct published-slot paints improve22.85–23.16→24.20–24.21FPS; this is slot
identity, not animated scene-change FPS. Active before9.84/11.40 vs after10.81/
11.67FPS is only a modest gain. Heartbeat tails are mixed; do not claim smooth
native24FPS. All8 workers retire, retained frames remain opaque/owned, and
natural deferred deletion reaches0widgets after each case without forced GC.
Mixed8-case process peakRSS466940KiB is not a paired total-RAM improvement.
Cold initial tile construction remains about584–684ms in these probes.

86 affected tests pass21.80s, including COW mutations, original buffered parity,
long resting timestamp/entry/reentry, material invalidation, failure-owned
painter cleanup, field queued-input/lifecycle and legacy material fallback.
Both changed dispatcher methods cover all37 executable statements and every
branch destination. Package callable/nested-doc guards90pass24.09s. Cohorts
are separate/overlapping; no unique aggregate or whole-suite claim.

Rejected distinct diagnostic: once-rendered opaque tiles converted to
ARGB32_Premultiplied preserve24 full native pairs, but dark shader median
65.81→76.10ms worsens; light53.55→53.56ms is unchanged. This format is NOT in
production. No nearest interpolation/vector/raster quality reductions introduced.

Reproduce paired probes in an isolated worktree at BEFORE368e2cdf1d:
copy profile_native.py, owned_probe.py, live_pair.py, format_probe.py into one
private /mnt/wd4tb/scratch/spinn-replay-20261007 directory. First run owned_probe.py
with that worktree path; it writes owned_candidate.py into its scratch directory.
Then run live_pair.py there. Each command uses:
  CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python <scratch-script> <before-worktree>
Do not run probes within this immutable archive. On AFTER254f3dd39f, reproduce
focused tests using files in acceptance.json and --cov=spacr.qt.widgets.ambient
--cov-branch with a private scratch coverage output. No GPU/full suite needed.

Filesystem and Git-blob verification:
  python features/data/663_spinn_owned_rest_cpu_2026-10-07/verify_manifest.py
  python features/data/663_spinn_owned_rest_cpu_2026-10-07/verify_manifest.py --git HEAD

Hard24FPS, historical native Save SIGSEGV, GPU, actual desktop and human aesthetic
acceptance remain OPEN. This individual renderer checkpoint precedes integration
with other agents' source edits; hashes must be checked again after integration.

Integrated-root acceptance: d6cbd checkpoint includes source8aae with83
meaningful field/spinn/cache/ownership cases PASS12.83s. Raw root log and
root-integration.json are archived separately. Both changed DataArt dispatcher
ASTs match the individual source exactly; the complete root ambient SHA differs
because other owners' changes are present. This does not bind their unrelated
code to this individual native performance proof. Root83 overlaps individual86;
no unique aggregate is claimed.
