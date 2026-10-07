Wave density quantity correction — CPU proof, 2026-10-07
=======================================================

Source commit `a57965350c`, parent `57544560ec`; accepted ambient SHA256
`95d55c192d6d7bde70fa671666d3a2138fe30e5cb7e9bb0805c526c85b57f0e2`.
Frozen parent SHA256 is
`63786b70d3082148db9d53a616957c4ad9c294b61ee5df31824a2433123f0437`.
Both complete renderers are compressed here. No other production function,
UI setting, native raster path, palette, gravity or wave shape was changed.

The old native 4K sampler hit its 900×520 upper grid limit by Density 1.
Density 1 and 3 therefore had the same 257,047 safely culled samples and,
after existing alpha compensation, identical Random frames. The preceding
read-only integration review established this actual control limitation.

The correction constructs the original maximum-density grid first, then
scales both axes by sqrt(requested density / maximum density). This reserves
headroom throughout the .01–3 range within the existing 900×520 maximum pool.
A 3×3 floor retains a central point on tiny canvases; quantized finite counts
can still plateau at very small canvases/settings. The original maximum-grid
48×32 minimum remains part of maximum-grid construction. Native raster pixels,
round stamps, wave deformation/repulsion, overscan bounds and cache lifetime
are preserved. Detail above native still neither supersamples nor trims density.

**Default population changes intentionally.** Native 3840×2160, size 1,
radius .5 now uses 85,871 / 171,675 / 257,047 sampled grains at Density 1 / 2 /
3. Default Density 1 has roughly one-third of the previously saturated pool.
This is an explicit quantity-control correction authorized after the review,
not an exact-frame performance optimization or an undisclosed quality cut.
Density 3 retains every previous maximum-detail pixel in the checked cases.

Evidence and limits
-------------------

* 50 focused tests pass in 10.52 seconds under capped 4 GiB, CUDA-hidden,
  offscreen execution. Two final recorder-callback cases pass separately in
  2.10 seconds after Ruff closure binding; these overlap the 50 and are not
  counted as additional unique tests. The atlas method's 46 executable
  statements and branch destinations have no gaps in this focused run.
* Four full native cropped-vs-uncropped wave-grid pairs still match exactly.
  Graded .01/.1/.5/1/2/3 sample counts are tested at 640×360 and 3840×2160.
  Native default and Random Density 1→2→3 frames differ; Detail 1/2 preserve
  the same samples and complete native frames at each density.
* `native_quantity.json` records twelve actual native density/detail frames
  and eight complete old/new Density 3 pairs: default/Random, dark/light,
  radius 0/.5. All eight maximum pairs have zero changed pixels. Default
  successive density changes affect 1,195,610 and 1,855,412 pixels; Random
  changes affect 1,221,185 and 1,925,225. Retained frames remain independent
  and all output alpha bytes are 255. Peak process RSS is 400,136 KiB.
* Three PNGs are actual full native default-palette frames at densities 1,
  2 and 3. They are renderer captures, not generated art. No timings or
  worker FPS are asserted by this quantity proof. Hard 24 FPS, human visual
  approval, installed crash acceptance and GPU acceptance remain open.
* Raw prior failed cohorts are retained explicitly: the generic Thore
  `(rain,lightning)` tuple length assumption was reported and fixed in parent
  source tests; a retired chromatin factory pin was reported to the catalog
  owner. Neither failure was hidden by a production guard or ceiling change.

The separately labelled `review_before_*` records/scripts are from earlier
integrated root `712a38c37e`, ambient hash `02779760...`, before this correction.
They show native Detail 1/2 parity, facet counts 920→2652 at Density 1→3,
bounded angles/native tile memory, and the atlas saturation discovery. They
also verify actual dialog `exec()` modality, modeless interaction, native
parent teardown and two availability popup reopen/Escape cycles. Popup flags
and ownership checks passed; offscreen tests do not prove desktop WM z-order.
These earlier records are not relabelled as final candidate acceptance.

Reproduction
------------

Run `python reproduce.py` to verify every payload and frozen renderer hash.
`python reproduce.py --git HEAD --repo /path/to/repo` verifies committed blobs.
For a fresh native replay, copy `native_quantity.py` and unpack
`ambient.before.py.gz` as `before.py` in a new directory under
`/mnt/wd4tb/scratch/`. Invoke the copied script with the bound source worktree
as its argument, using that worktree's `tools/run_capped.sh 4G`, CPU Python,
`CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`. It creates only local receipts/images in that scratch
directory. Use commit `a57965350c` for candidate replay and `712a38c37e` only
for the explicitly earlier read-only review scripts. No full suite is needed.
