# ITEM DRAFT (no number yet): govern the whole-site Cellpose mask step of item 921 (VRAM cap, CPU fallback)

- **Number:** 945 (reserved 2026-11-14).

- **Code basis:** origin/nightly ad161c4242. Affects `spacr/ops_large_sites.py` `_cellpose_masks` and `spacr/_gpu_governor.py`.
- **Found by:** PERISCOPE P1 benchmark (S3), 2026-11-13. This blocks S3 on Home under the 6 GB per-job GPU rule.

## Gap
- Item 921 stitches the tile flows over the whole 5,500² site and then runs Cellpose's mask step once on the whole site, on the model's device:
  - `resize_and_compute_masks` → `remove_bad_flow_masks` → `flow_error` → `masks_to_flows_gpu`.
- Unlike the network calls (`_eval_labels` / `_eval_flows` go through `_governed_eval`), this call is outside the GPU governor:
  - no batch shrinking;
  - no out-of-memory retry;
  - no CPU fallback;
  - `gpu_memory_fraction` is not respected.
- With a 6 GB cap (`torch.cuda.set_per_process_memory_fraction`, maintainer GPU rule), site 3 (742 cells) finished. Site 15 (6,328 cells) failed with `torch.OutOfMemoryError: Tried to allocate 1.47 GiB` in `_extend_centers_gpu`, and the run stopped (rc 1). Log: `periscope_benchmark/logs/run_seg_s3.log`.
- The Ready 921 runs used an uncapped 24 GB work-machine card, so this was never hit.

## Spec
- Route `_cellpose_masks` through the governor:
  - estimate its need from the site's pixel count and the share of pixels with cellprob > 0, as `SITE_FLOW_BYTES_PER_PIXEL` does for RAM;
  - run on the GPU when it fits under `gpu_memory_fraction` / the free memory, else on the CPU;
  - catch OOM: release, retry once, then use the CPU.
- Record `mask_step_device` per site in `sites.csv`.
- New setting `seg_mask_device` (`auto` | `gpu` | `cpu`, default `auto`).
- `gpu` keeps today's behaviour exactly.

## Proof check (pre-set)
- On sites 76 and 15, labels from `seg_mask_device='cpu'` are identical to `'gpu'`, or IoU ≥ 0.9 for every object, as in the 921 Ready rule for GPU float order.
- A 6 GB-capped run of the 10 P1 sites finishes with no OOM stop.
- With `gpu`, labels are byte-identical to today's.
