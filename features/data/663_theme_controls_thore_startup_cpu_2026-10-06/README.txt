2026-10-06 accepted CPU renderer controls, Thore, and startup boundaries.

Source checkpoints are individual renderer branches, not the combined app SHA.
manifest.json binds exact source snapshots, original measurements and tests.
The separate root app/timing integration must close before Home construction and
release after its actual post-paint readiness observers; hosted CI is separate.

Density9e9: 88 focused cases pass. Actual3840x2160 display frames from all ten
offered themes differ at1/10/50%; classic themes retain their existing buffer
sizes. Thirty complete dark/light .5/1/2-density fresh-engine frames match the
pre-change renderer byte-for-byte. Aurora's bounded ray/surge cache keys now
include peak alpha. Fractional weighting applies only below one coarse shape;
default/higher density alpha compensation is unchanged. Rain counts1/10/52,
paper grid population and advection counts scale with low density.

Radius has no hidden minimum: actual4K changed pixels at1/10/50% are atlas
116/16426/766921, field131/16824/302784, lens429/39503/798633. Centre-sampled
paper gives0/21759/676372 at cursor(.5,.5): a1% region can miss every tile centre.
This discrete paper interaction limitation is recorded, not marked complete.
Pixel occupancy and intensity sums are descriptive, not aesthetic thresholds.

Thore7fd53: tapered analytic rain wind/fade and narrower pale32-step branching
bolts with horizontal and vertical indexed events. Thirty-nine targeted cases
pass, including tail/head pixels, subframe movement, connection, seek/cache,
restraint, native frame ownership and all-theme low density rasters. Six native
dark/light frames were inspected; four lossless frames/crops are retained here.
Actual software-Xvfb3840x2160 requested24FPS worker before/after, fresh processes:
published23.962/23.997FPS; worker median6.396/8.411ms, p9514.304/18.923ms,
max19.193/24.278ms. Both runs publish120 distinct clocks; repeats3/4. Publication
p9548.33/51.23ms exceeds the41.67ms target despite near24 average. GUI heartbeat
p9516.70/16.24ms. Rain taper increases shader cost; universal smoothness stays OPEN.
Each receipt includes cold show, RSS, actual load and immediate-hide worker stop.

Startup9aef: both optional compiler helpers defer scheduling/import while the
explicit application gate is closed, retaining exact NumPy frames and one later
warmup attempt. Worker-entry race resets defer without marking failure. Existing
compiled kernels are reusable while closed; standalone renderer gate defaults
open. Fifty focused compiled/wave/gate cases pass; six ambient callable docstrings
are repaired. Package audit in the older isolated base had only unrelated
core.py nested analyse_pool omitted; no ambient callable omissions remained.
Fresh process rendered actual1920x1080 lens+satin while both NumBa and SciPy were
absent, then released and compiled in1.021s; complete frames matched fallback.
This proves renderer boundary, not installed-wheel Home startup independently.

Reproduce with the spacr CPU Python:
python reproduce.py density --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
python reproduce.py startup --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
python reproduce.py worker --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
The wrapper validates originals, restores explicit sources only under scratch,
caps each Python child4G with CUDA hidden, and uses software Xvfb for live worker.
Sources are gzip snapshots; no new app modules, third-party assets, GPU work,
user aesthetic approval, Save crash closure or hosted-CI-green claim is included.
