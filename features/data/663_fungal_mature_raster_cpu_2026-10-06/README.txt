This archive binds the guarded mature-mycelium raster candidate to parent22413
ambient08815be8 and source/test555323f6db ambientb2a19147. Only the existing
_FungalGrowthEngine changes; source_boundary.json proves every byte outside
that class unchanged. It preserves geometry, original group/pen insertion
order, alpha, native detail, density, colour and blur. The optional fast path
requires this shade's private RGB32 image with identity transform, full
opacity, no clip, dark Plus mode, size>=2 and effective density<=1. External,
light and higher-density contexts retain the original Qt painter.

Owned sparse indices/RGB words cache unchanged mature groups. A conservative
headroom test keeps saturated groups on their original Qt path. Cache arrays
are limited to8MiB/64entries; observed paths64; at most one warm group/frame
after three unchanged observations. Qt path metadata is entry-bounded and
not included in the8MiB array figure. A full-native32MiB staging QImage is
frame-local; no native QImage or borrowed output array is retained by cache.
Strong local image owners protect diagnostic traceback views. A cache error
restores painter state, clears/disables the optional cache, resets the owned
frame and repaints identical original groups/tips before returning. Direct
field errors retain their original propagate/end-painter behavior.

native_parity.json records144 full3840x2160 byte-equal pairs, including
spacr/Random, custom white and16-bit-channel custom colours, dark/light,
density.01/1/3, detail1/2 and three clocks.40frames exercise166cache hits;
alphaFF, native dimensions and identical geometry are checked. The new
focused tests cover old returned-frame independence/collection, owned cache
arrays, partial-write failure recovery including transform/opacity changes,
readable retained exception pixels/eventual collection, positive native and
twelve invalid contexts, saturation,8MiB/64entry/history limits and resize.
Source-targeted branch coverage reaches154statements in12touched methods
without missing statements or reported branch destinations. It is not a
package coverage claim or MC/DC statement about each short-circuit predicate.

The final source-current actual widget/producer pairs request24FPS with
3840x2160, density3/detail2/size2.5, seed42, speed1/blur0/radius0.
Default:12.929->16.128 distinct publishedFPS; shade median74.06->61.07ms,
p9596.24->74.44ms;10ms heartbeat p9554.25->24.54ms.
Random, reversed launch order:16.539->19.564FPS; shade median60.29->50.83ms,
p9571.62->63.42ms;heartbeat p9549.15->24.57ms.
Machine load and full cold/steady records are retained in each receipt.
CurrentRSS rises from308792->324928KiB (default) and308732->325172KiB
(Random); this is a speed/latency tradeoff, not whole-RSS improvement.
All four producers stop immediately/200ms after hide. Hard24FPS remainsOPEN.
Prior scratch variants and their faster absolute FPS are not accepted final
source measurements. No GPU, whole-suite, aesthetic acceptance, crash fix,
or generated API/docs/translation acceptance is claimed here.

Separate tests-only fixes: d93816f44e preserves adjacent-lineage assertions
onPython3.9 viazip instead ofitertools.pairwise.4b1c86bda5 replaces an obsolete
fungal style census with early advancing/common-origin and mature/hour-later
render phases, <=30%screen ink,>=half painted-ink motion over7seconds, and
injected empty/frozen/mature-frozen/overfilled defect rejection. Original
linked-parent/fork/bright-tip/density/<=25%geometry/raster guards remain.

Reproduce against a checkout containing 555323f6db (and 4b1c86bda5 for phases):

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python -m pytest -q -p no:randomly tests/qt/test_fungal_mature_raster_cache.py tests/qt/test_data_art_owned_branch_frames.py tests/qt/test_fungal_growth_engine.py

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python ARCHIVE/native_parity.py CHECKOUT_PATH

SPACR_PROOF_REPO=CHECKOUT_PATH CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=xcb QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s '-screen 0 3840x2160x24 -nolisten tcp' tools/run_capped.sh 4G python ARCHIVE/worker_probe.py data_art_fungal_growth before --palette spacr

Replace before with after and spacr with random for the other three worker
cases. ARCHIVE is this directory; CHECKOUT_PATH is the absolute repository
checkout. Frozen before.py/after.py are imported explicitly, so both worker
variants use identical supplied checkout dependencies. Actual phases: 2s
cold at clock0, set_time95, 2.2s warmup, 5s steady. Run processes sequentially;
full native grab/hash happens after steady. Use the original repo fatal Ruff
and six Xenon targets from .github/quality-targets.txt. Counts overlap;
focused-tests.log is a raw archived pytest log. Other phase results are
faithfully summarized from this agent's tool transcript, not raw log files.
