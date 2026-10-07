This investigation is complete and rejected. It changes no application source,
tests, cache guards, native resolution, density, colours, or alpha. Native hard
24 FPS remains open.

The frozen production renderer has SHA256
`f36b33c9714b26a49de0e9d362aadaeee5d110a725e022c622d7eb659cf35225`;
the private stroke-cache candidate has SHA256
`708b4f91e909d7d003154ec7af9f5053a12bac7e0630c2075b03b2e1187fa52f`.
Both are stored compressed. Scripts and receipts from the original scratch
experiments are copied unchanged. No separate initial/cached raw logs were
saved; their JSON receipts are the available original numerical record.
`admission.log` is the actual later capped counter-probe output.

The unrestricted Qt stroked-outline fill matches a complete native 4K frame at
size 2.5 but changes 61,750 pixels at size 1. It is rejected for that mismatch.
The subsequent candidate uses the existing eligible painter path and bypasses
strokes below 1 pixel. Its 12 complete native comparisons match: nine exercise
stroke conversion and three are thin-stroke bypass controls. These are dark
default/Random presets at one seed and three clocks, not every possible context
or a full custom/light qualification. Balanced direct-shader medians worsen
7.17%, 9.86%, and 4.73% in the three eligible recipes. The thin control changes
-0.43%. Direct-shader measurements are not live widget/producer FPS.

The admission probe uses 72 evolving native 3840×2160 frames per case,
detail 2, density 3, size 2.5, seed 42, clocks 95 to 97.9583333333. Original
default/Random sparse caches have 19/28 entries, 760,976/766,952 owned array
bytes and 1,005/1,378 successful reuses; neither evicts nor refuses headroom.
Observation caches also retain only 19/28 entries. The candidate stroke cache
has 12,964 lookups, zero hits and 12,900 evictions. Approximately 194 evolving
groups per frame overwhelm its 64 entries. These instrumented counters are
not a new timing or parity measurement. Limiting admission to mature unchanged
groups would duplicate existing sparse reuse; skipping unstable outlines
returns the original Qt work. No further improvement is justified by this
bounded diagnostic. Extra native stroke-path storage is not accounted for by
the original sparse-array 8 MiB guard, so the candidate is not a qualified
memory-safe extension of that budget either.

Verification uses only the standard library and does not rerender:

```
tools/run_capped.sh 2G python features/data/663_rejected_fungal_stroker_cpu_2026-10-07/verify_proof.py
```

After committing, add `--git HEAD` to verify committed Git blobs. The manifest
binds all payloads except itself. The verifier also checks decompressed source
hashes, the rejected thin result, all 12 cached comparisons, and counter/source
bindings.

Optional replay runs one preserved probe in a private scratch directory:

```
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/run_capped.sh 4G python features/data/663_rejected_fungal_stroker_cpu_2026-10-07/reproduce.py admission
```

Other phases are `unrestricted` and `cached`. The runner loads frozen renderer
bytes and changes only the original admission script's scratch-path literal;
other package dependencies come from the calling checkout/environment. It
prints generated receipts before removing its private scratch directory.
No rerun is required to verify this archived evidence. No GPU, GUI aesthetics,
platform-wide acceptance, startup/lifecycle acceptance, or hard FPS closure
is claimed.
