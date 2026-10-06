# N663 advection point gravity — 2026-10-06

The source change `bf4a96528064628ff00f5ed8a7c7796451656c12` replaces the
advection painter's pointer-local rotation with inward pull. It uses the
already computed trail positions at three recent samples to estimate exposure
to the cursor's compact field. The strength is highest near the center and
smoothly reaches zero at the selected radius. Radius zero takes the original
path; every outside-radius trail coordinate remains unchanged.

This is a bounded finite-history approximation to moving particle trajectories,
not a physical time integrator. Background wind and vortices continue, so a
particle's *total* instantaneous velocity need not point inward. The
attraction contribution points inward. Reverse seeking is deterministic and
does not retain particle history across frames.

Source identity:

- Parent `d576c0333f25f705343bfd5717d1a27250a30b9d` renderer SHA-256:
  `ca7cc348dc4e99e0413b2954839e309a6795c12f4ff9450af298b04baeadba8c`.
- Changed renderer SHA-256:
  `2917d3be9db693075c160aedb3ed233d32eb92e5e7ef7cd234eb3e100d926d87`.
- Source changed only `_DataArtEngine._frame_genetic_advection`, plus
  `tests/qt/test_data_art_advection_gravity.py`. No detail, point count,
  buffer, FPS or memory limit was changed.

`radius_metrics.json` comes from `radius_probe.py` at seed 42,
1920×1080, times 2.0 and 2.2. At 1%, 10% and 50% reach, respectively,
19–24, 3684–3693 and 155062–155753 of 408000 trail samples changed.
All 230345–407951 samples at least two pixels outside the respective
radius remained coordinate-identical. Within 80% of the radius, 67–71% of
samples moved inward at 1%; all samples moved inward at 10% and 50%.
The small 1% count reflects pixel rounding, not a hidden minimum-radius
clamp. Tests also verify exact radius-zero restoration, finite center,
relocation of the cursor, evolving clocks and reverse seeking. Critically,
two trail sizes have identical original head positions but over 1200
different attracted head positions: the pull depends on recent path,
rather than only the current point's radial location.

`worker_probe.py` wraps the actual offscreen `AmbientWidget` timer and
`_FrameProducer` under a 4 GiB process cap with CUDA hidden. It warms
600 ms, then measures three seconds at each native 1920×1080 and
3840×2160 viewport, fixed center pointer and 24 FPS request. The
source-bound JSON receipts are:

| Renderer and reach | 1080p published FPS | 4K published FPS | 4K shade median |
| --- | ---: | ---: | ---: |
| Parent, 25% | 23.86 | 23.81 | 24.53 ms |
| Pull, 25% | 24.14 | 23.81 | 20.78 ms |
| Pull, 50% | 23.94 | 23.94 | 25.61 ms |

Every 4K run published 75 distinct animation clocks and stopped its
worker on hide. These short local measurements characterize this machine;
they are not an FPS guarantee on every host.

`capture.py` rendered owned native 1920×1080 images at time 8, radius
25%, centered pointer. The exact PNGs remain in
`/mnt/wd4tb/scratch/advection_gravity_probe_20261006/final/`:
parent SHA-256 `682d1f9800855dc12f5470222ba94c13efa8e251f678aa4745fff2d7f9e995ba`,
pull SHA-256 `b101fd8e20fe0a6c04681e52308322427f75fd2ad950bac3788d9487a4123097`.
The two full-dimension WebP images here are size-bounded lossy review
previews, not pixel parity artifacts.

Focused validation, using the conda `spacr` Python 3.12 interpreter:
```sh
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen \
  tools/run_capped.sh 4G python -m pytest -q \
  tests/qt/test_data_art_advection_gravity.py \
  tests/qt/test_data_art_advection_evolution.py \
  tests/qt/test_data_art_gravity_radius.py \
  tests/qt/test_data_art_renderer.py \
  tests/qt/test_data_art_owned_point_frames.py \
  tests/qt/test_data_art_crisp_controls.py
```
Result: 92 passed. The archive scripts pass Ruff; the changed source has
no whitespace errors. Existing unrelated import-order findings elsewhere
in `ambient.py` were not altered.

For replay, extract the parent renderer with
`git show d576c0333f25f705343bfd5717d1a27250a30b9d:spacr/qt/widgets/ambient.py > /tmp/ambient_parent.py`.
Set `AMBIENT_SOURCE` to that file or the changed `ambient.py`, then run
`worker_probe.py` or `capture.py` from the repository under the same
4 GiB/CUDA-hidden/offscreen wrapper. `radius_probe.py` imports the current
checkout, so first check its renderer hash against the changed identity.
