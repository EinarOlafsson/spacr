2026-10-06 CPU native common-origin mycelium revision and paper-radius audit.

Individual renderer checkpoint204f819938, SHA256bb0dc363259a186db869846094cbbc7b
067215dd0fc43035ad37f33c51ef4318; parent integration SHA is separate. Source before
is9aefdd563a, after original owned-frame improvement and compiler startup gate.
No GPU, reference asset/code bundling, new app modules or Save-crash claim.

Fine irregular connected filaments branch progressively from common origins.
Each finite tree has240 ancestor-first six-segment branches; default density
selects90. Colonies start30s apart, fade over110s and retain at most8 cached trees.
Established branches dim while growing tips stay bright. Clock seeks rebuild
deterministically rather than retaining simulated history. Compatible exact-alpha
paths are batched. Continuous partial Beziers, finite raster footprint budget,
native detail and fresh independent output frames are preserved.

53 focused cases pass. Recorded branch trace covers all51 added executable
statements and26 touching arcs, with zero gaps. Tests verify parent-tip joins and
birth order, advancing-tip brightness, density population, fractional growth,
continuous colony boundary, two-hour seeks/cache eviction, ownership/lifecycle,
and real dark/light25% occupancy at size/density/blur/detail extremes including4K.
Ten actual native dark/light frames at5/18/47/97/3600.33s were inspected; four
lossless frames/crops are retained. Maximum default sampled ink0.007591, below
25%. Actual1/10/50% density populations at80s are18/136/550 visible edges.
Appearance review remains the user's; this does not declare aesthetic acceptance.

Fresh sequential software-Xvfb3840x2160 actual widget/producer request24FPS:
before/after published23.997/23.779FPS, distinct clocks120/119, shade median
21.500/16.784ms, p9534.471/26.569ms, max41.835/33.582ms. Repeated paints19/5;
GUI heartbeat p9521.459/16.915ms. Publication intervals p9564.211/50.985ms still
exceed41.67ms, so smoothness is OPEN despite shader samples within budget.
Original receipts include cold first show, actual CPU load/RSS and immediate
hide worker stop. Earlier same-source shared-load samples had27.44ms median,
46.62ms p95 and23.38FPS (agent transcript only, not archived raw receipts).
No hard performance guarantee or silent detail/density reduction is made.

Paper audit test-only checkpoint0fb0c875a3: existing influence samples primitive
centres. A1% region between centres may target zero cells. Native seed42 actual
3840x2160 test positions pointer(.509249,.492937) on a real seeded centre:21.6px
physical reach targets exactly1 cell, changes5452 native pixels and clears back
to identical resting pixels. This is an intentional geometric sampling boundary;
no expanded radius, minimum or new physics was added. No source defect remains
in the bounded primitive-centre contract, while the limitation stays explicit.

Portable capped reproduction:
python reproduce.py frames --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
python reproduce.py paper --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
python reproduce.py worker --repo /path/to/spacr --scratch /mnt/wd4tb/scratch/replay
The wrapper validates immutable files and sources, writes only scratch, hides
CUDA and caps each Python child4G; live workers use CPU software Xvfb.
Coverage JSON is recorded from the individual branch's53-case cohort; scripts
and source.diff bind the changed application source without raising any gate.
