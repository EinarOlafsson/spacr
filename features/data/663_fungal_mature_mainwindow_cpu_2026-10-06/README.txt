This bounded integrated MainWindow check uses exact ambient b2a19147,
Preferences733f9af0 and app5fc990e6. Root HEAD at process start was
1534cf84185c9877a872e7772773f3d7e561fbee; all three actual imported file hashes
are asserted stable at process end. IO/object/core commit file hashes remain
unchanged and are additional provenance, not claims of processing coverage.

Fresh CPU-only software Xvfb has a3840x2160 native display. Home, Mask,
Measure and Annotate are actually opened. Three real modal Preferences Saves
set density3, size2.5, blur0 and detail200/100/200, Random/spacr/Random,
gravity preference0/65%/0. Each selected growth widget jumps to clock95.
First maximum-control scene shades its native3840x2104viewport on the
8294400physicalpixel budget and performs16real successful mature-cache hits,
with6entries/66136ownedarraybytes. All recorded output images match the
native current viewport. The final3200x1744viewport follows an intentional
windowresize between Saves, rather than a renderer resolution reduction.

Later hit counts414/623 are cumulative per-engine counters. They do not
claim the higher-effective-density/detail1 context is cache-eligible. Cache
entries are capped64/8MiB; maximum recorded arraybytes1063448. Existing cached
contributions may remain while a context uses the unchanged originalQt path.
All three produced/retained frames have alphaFF. Raw SHA256 of earlier owned
frames stays unchanged through palette/detail changes, navigation and window
resizes. This is raw-frame independence rather than a screenshot impression.

11natural garbage collections occur, all on the GUI thread. The probe never
calls gc.collect. Close/deferred deletes leave0QWidgets/0AmbientWidgets/
0spacr-ambient-shade threads. Peak1264156KiB; aftercloseRSS1130688KiB and
RssAnon976416KiB include three intentionally retained native ownedframes.
No comparative RSS improvement, hard24FPS, aesthetic acceptance, installed
SaveSIGSEGV fix, GPU, inference or platform-wide acceptance is claimed.
The full original log retains postcompletion GTK/Xvfb monitor diagnostics;
process exit0, every assertion and final source-stability check passed.

receipt.raw.json is unmodified probe output; receipt.json adds only provenance
and counter/scope/limitation explanations. Source probe and whole log are
archived. Adapted from immutable663_unchanged_preferences CPU stress script.

Run with the spaCR environment and absolute CHECKOUT/ARCHIVE/SCRATCH paths:
SPACR_SOURCE_ROOT=CHECKOUT SPACR_PROOF_DIR=SCRATCH PYTHONPATH=CHECKOUT CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=xcb QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s '-screen 0 3840x2160x24 -nolisten tcp' CHECKOUT/tools/run_capped.sh 4G python ARCHIVE/repro_mainwindow_fungal_cache.py final

The script requires the exact archived renderer SHA. No source changes or
push were made by this audit. Complete terminal output is in probe.log.
