Complete/rejected CPU experiment. Production source, render quality, cache
guards and memory/work limits are unchanged; hard 24 FPS remains open.

Candidate SHA256 `74d1de82ee8d0d8d82b37258364cc0752c66d97426a0d3cef25d34a99ddd82c4`
hoists one frame-invariant fractional-alpha calculation outside the visible
edge loop. Frozen baseline is
`f36b33c9714b26a49de0e9d362aadaeee5d110a725e022c622d7eb659cf35225`.
The decompressed sources, original scripts, receipts and raw logs are bound
by the manifest. `decision.json` preserves the original scratch payload
hashes, which include duplicate uncompressed before/after files not retained
here; the archive manifest describes the actual compact payload set.

Twenty-eight geometry cases and twelve complete native 4K pairs are exactly
equal across default/Random/custom and a light 1% density control. Direct
geometry medians improve about 28% (0.51–0.96 ms), but direct complete shader
timings are mixed. No per-depth cache was implemented.

Eight fresh actual software-Xvfb native 3840×2160 widget/producer processes
use default and Random, each in A1/B1/B2/A2 order. Each process retains the
original cold two seconds, warm 2.2 seconds, steady five seconds, requested
24 FPS, detail 2, density 3, size 2.5 and blur 0. The production shade path is
unprofiled; publication/paint/heartbeat measurement wrappers remain. Default
mean publication FPS is 19.950→19.691 (-1.30%); Random is 23.568→23.596
(+0.12%). Mean worker-shade medians save 0.394/0.691 ms. Both recipe mean p95
values remain above 41.7 ms. All eight workers retire after hide. Source,
load, cold/steady timings and dependency bindings remain in each raw JSON.
These mixed results do not justify production integration or hard FPS closure.

Verify without rerunning:

```
tools/run_capped.sh 2G python features/data/663_rejected_fungal_alpha_hoist_cpu_2026-10-07/verify_proof.py
```

Add `--git HEAD` after commit to read committed blobs. Original scripts use
their recorded scratch directory. Optional replay requires decompressing
before/candidate into a private directory under `/mnt/wd4tb/scratch`, copying
candidate to `after.py`, and placing `profile_worker.py` alongside. Set
`SPACR_PROOF_REPO` to the calling checkout and use CUDA-hidden capped 4G with
`QT_QPA_PLATFORM=xcb`, OMP/MKL threads 1 and private native-4K Xvfb. Package
dependencies come from that checkout. No replay was performed for this archive.
No GPU, aesthetic acceptance, full-suite acceptance or crash fix is claimed.
