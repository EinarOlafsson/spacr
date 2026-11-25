# Item 1011: VRAM-filling, even, seeded-random mask batches as the default (Mask Generation and OPS)

- **Number:** 1011 (master, 2026-11-21). **Settings schema:** 955 (`DEFAULT_CHANGES[955]`, `SETTINGS_SCHEMA_VERSION = 955`).
- **Decided by:** the maintainer, 2026-11-21 ~14:20, for the final Funk runs (A1 pilot, plate A, all plates). It becomes spaCR's default.
- **Lifecycle:** the behaviour is the default for every run (API, CLI, GUI). The three new GUI controls (below) are registered in `ALPHA_FEATURES[1011]`, so they show only with "Show alpha features" on (decision 1 above). With alpha off, a GUI run still uses the new defaults.
## Maintainer decisions (2026-11-21)

1. **Batching is ON for everyone.** The new batching (`batch_size` and `mask_eval_batch_size` default `'auto_vram'`, even sizes, seeded random assembly, VRAM-aware tile batch) is the default for every new file and every GUI run, with "Show alpha features" off or on. Its three new controls, `mask_eval_batch_cap`, `mask_batch_seed` and `gpu_vram_budget_gb`, are alpha-only: they are registered in `ALPHA_FEATURES[1011]` and show only behind "Show alpha features" (off by default). Old settings files stamped before 955 keep the old values through `DEFAULT_CHANGES[955]` (see §3), so their masks do not change.
2. **Mask Generation gets exactly the same batching as OPS, through the shared plan.** One code path and one planner (`_mask_batches` plan, `_mask_array_batches`, `_gpu_governor`, with `_calibrate`, `_even_sizes` and `_field_order` as defined here). There is no separate Mask Generation batching, and no OPS-only batcher. It is accepted by the Mask Generation route in §4 F.

- **Code basis:** origin/nightly 19a95cd799. Spec only; nothing here is implemented.
- **Repo rule 43 (one operation, one code path):** this item EXTENDS item 987b's one batch plan (`spacr/_mask_batches.py`), its array-input twin (`spacr/_mask_array_batches.py`) and the GPU memory governor (`spacr/_gpu_governor.py`). Mask Generation and OPS already share all three. No OPS-only batcher, no second VRAM probe, no second shuffle.

## 1. What exists today (origin/nightly 19a95cd799)

| Concern | Existing code | What it does now | Gap for 1011 |
|---|---|---|---|
| `batch_size` request | `_mask_batches._batch_size_request`, `_legacy_batch_size` | 0, -1, None, `'none'`, `'auto'` mean automatic (None). The archive routes (timelapse, volumetric, plot, illumination/PSF sessions, item 987c) get the legacy 50. | No token for the new mode. |
| Automatic field count | `_mask_batches._largest_fitting_size` via `_resolve_plan` (Mask) and `_mask_array_batches._batch_limit` (OPS) | Largest batch that fits **free** VRAM and RAM. It uses fixed constants (`_MODEL_GIB` 1.3, `_TILE_GIB` 0.2) and `_field_costs`, not a measurement. On a 24 GB card it usually fits the whole plate, or the whole well for OPS, as **one** batch. That is the behaviour the maintainer rejected (§0.3/§2.3 of `20261119_final_funk_runs.md`: "987b's new default 0 normalises the whole plate as one batch"). | It is not measured, it reads the whole card instead of the declared budget, and it does not split evenly. |
| Batch sizes | `_mask_batches._batch_sizes` + `_borrowing_indices` (Mask). OPS: greedy per-group `limit` in `_iter_array_batches`. | Full batches first, then the remainder. A last batch below half the median borrows the median percentiles of the others. | Gives small end batches. |
| Field order | `io.py:2686` `if settings['randomize']: random.shuffle(paths)` (also `io.py:1721`, `_concatenate_channel`). `randomize` defaults to True in Mask (`settings.py:1234`); timelapse forces it to False (`core.py:673`). OPS has no shuffle: it uses acquisition order (`order` in `ops_engine.py` ~5655). | The shuffle is already random, but it is **unseeded** (global `random`), so a settings rerun gets different batches. | It needs a seed and a record. |
| Batch normalisation | `io._normalize_img_batch` called from `_mask_batches._load_batch`. Per-batch percentiles go into `batch_plan.json` (`_record_normalisation`) and per field into `spacr_resolved_parameters` (run journal). | Percentiles are pooled over the fields of one batch (Mask). **OPS does not pool:** `_ops_mask_eval` hands each field to Cellpose's own per-image normalisation. | None (reuse). |
| Tile batch | `object._eval_tile_batch` (`mask_eval_batch_size`, 0 = one tile per field in the batch). `_mask_array_batches._tile_override` + `ops_engine._ops_tile_batch` (`ops_ph_tile_batch`, `ops_sbs_tile_batch`, `spatial_tile_batch`; 0 = use the shared key). Schema 948 gives old files 8/16/16. | Fixed numbers. The governor shrinks the batch, or waits for memory (mask sites), at run time. | No VRAM-aware default, and no cap. |
| VRAM probing | `_mask_batches._free_vram_bytes` (`torch.cuda.mem_get_info`), `_mask_batches._store_on_gpu`, `_model_resident`. Governor `_memory`, `_allocated`, `_reserved`, `_budget`, `_BatchController._learn` (it measures `per_unit` = max(reserved growth, 1.5 × allocated growth) per tile, on the real data), the ledger, and `gpu_memory_fraction`, `gpu_memory_margin_gb`. `gpu_admission.GpuAdmission` (`ops_gpu_vram_cap_gb` worker halving). `ops_large_sites` `seg_mask_vram_gb` (item 945). | `_learn` already makes the per-tile measurement 1011 needs, but only after the batch size is fixed. | Nothing reads gpu_turn's declared `--vram-gb`: `tools/gpu_turn.sh` does not pass it to the child. The Funk scripts cap the card themselves (`run_step.py --gpu-cap-gb` → `set_per_process_memory_fraction`). |
| Resume / grouping record | `batch_plan.json` (Mask; resolved plans are kept for the same request). `mask_batch_plan.json` (OPS `_batch_plan`, refuses a different `batch_size`/`mask_eval_batch_size`). | Resume keeps the recorded groups. | It must also hold the seed, the calibration and the resolved tile batch. |

## 2. Behaviour

### 2.1 Calibration pass (once per run, per device)

New function `_mask_batches._calibrate(model, sample, settings, *, site)`. It is used by both apps.

- **Sample:** the first field of the seeded order (§2.3). For Mask it is the normalised kept channels. For OPS it is the field as `load_field`/`load_window` returns it, with the same `ops_ph_downsample` and channel stacking as the real call.
- **Probes:** two `_mask_eval` calls on that field, at tile batches `t1 = 2` and `t2 = min(8, cap)`. Both go through the governor's `_BatchController` for `site`, so the measurement is the governor's own: the growth in reserved memory, never below 1.5 × the growth in allocated memory (`_learn`, `_HEADROOM`). The results are **discarded**. The two probes cost two field evals, about 20–30 s per run (ph_segment is about 10–15 s per field, against 3.6–5.3 h per well).
- **Fit:** `per_tile = (need(t2) − need(t1)) / (t2 − t1)` and `fixed = need(t1) − t1 · per_tile`, where `fixed` covers the weights, the image and flow buffers, and the network's fixed working set. Both are floored at the 987b constants' tenth (`0.1 × _TILE_GIB`, `0.1 × _MODEL_GIB`) so a noisy probe cannot claim the tile is free. If `t2` itself does not fit, it falls back to `t1` alone with `per_tile = need(t1)/t1`, `fixed = _MODEL_GIB`.
- **Per field (resident batch):** the bytes are known exactly: the packed array that `_store_on_gpu` copies (`array.nbytes` per field). No probe is needed.
- **Budget B:** the governor's budget, which is now the one source (`_free_vram_bytes` is rewritten to call it):
  1. the declared budget, when one is set: `gpu_vram_budget_gb` (new governor key; empty → env `SPACR_GPU_VRAM_GB`, which `tools/gpu_turn.sh --vram-gb N` and `gpu_turn.ps1` now export to the child). The governor applies it as an effective `gpu_memory_fraction = min(fraction, B/total)` through the existing `_apply_fraction`/`_keep_headroom` path. The scripts' `--gpu-cap-gb` then becomes this setting;
  2. otherwise the free VRAM at calibration (today's `mem_get_info`), less what other spaCR processes have announced in the ledger (`_pending_elsewhere`);
  
  then less `gpu_memory_margin_gb` (1.5) and the 0.4 GB CUDA context (`_CONTEXT_BYTES`). With `mask_workers_per_gpu` > 1, B is divided by the number of workers on that GPU.
- **Resolve, tile batch first** (it sets the speed): `tiles = clamp(floor((B − fixed − 1 field) / per_tile), 1, cap)`. Then `max_fields = floor((B − fixed − tiles · per_tile) / per_field)`. Then that is limited by RAM through the existing RAM term of `_largest_fitting_size` (one batch + 2 GiB spare; plus one look-ahead batch when `_BatchFeeder` builds the next batch while the current one runs; the implementing lane checks whether the look-ahead copy is on the GPU and, if so, counts it in B too).
- **Record:** the calibration (`per_tile`, `fixed`, `per_field`, B and its source `declared|free`, `tiles`, `max_fields`, device name, total VRAM, Cellpose and torch versions) goes into `batch_plan.json` (Mask) or `mask_batch_plan.json` (OPS), into the run journal (§2.3), and into one console line: `Mask batches: 1,281 fields → 6 batches of 213–214 (VRAM budget 6.0 GB declared; tile batch 24 of cap 32; 0.14 GB per tile measured).`
- **Resume / replay:** a resolved plan with the same request, seed and budget source is kept, so the run is not calibrated again. `spacr repro` and a journal replay use the recorded `tiles` and sizes, with no calibration. On a smaller card the governor then waits, and then shrinks with its existing "a few pixels" warning. The run record notes that this happened.
- **No CUDA device** (CPU, MPS, or a backend served from another environment, where `_model_device` is None): no calibration. `tiles = min(cap, 8)` (Cellpose's own default). `max_fields` = the RAM limit, capped at `_LEGACY_BATCH_SIZE` (50) so the percentile pools stay the size the CPU route always had. Even split and seeded order still apply. The record says `budget source: cpu`.

### 2.2 Even batching

New `_mask_batches._even_sizes(n, max_fields)`: `k = ceil(n / max_fields)`, `sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]`. Sizes differ by at most 1, and none is larger than `max_fields`. It replaces `_batch_sizes` for `'auto_vram'` only. Explicit numbers and the 987b automatic keep `_batch_sizes` + `_borrowing_indices` unchanged. With even sizes, `_borrowing_indices` never fires for `'auto_vram'`, except when `n` is 1 (no change needed: one batch, nothing to borrow).

OPS: `_iter_array_batches` no longer derives `limit` greedily per group under `'auto_vram'`. It takes the precomputed `_even_sizes(len(records), max_fields)` (capped by `max_fields` of the stager as today). Resume keeps the recorded groups from `mask_batch_plan.json` as today.

### 2.3 Seeded random assembly

- **One shuffle, seeded:** `io.py:2686` and `io.py:1721` change from `random.shuffle(paths)` to `random.Random(seed).shuffle(paths)`. The seed is the new shared key `mask_batch_seed` (int, default 42). It sits in the Mask defaults and in `_OPS_MASK_SETTINGS`/OPS defaults, and both apps read it through one helper `_mask_batches._field_order(names, settings)`. `randomize` stays the on/off switch. Timelapse still forces it off. `watch_folder` fixed_map still requires `randomize=False`.
- **OPS:** phenotype fields (`ph_segment`) use `_field_order` too, with `randomize` added to the OPS mask settings (default True). The `FieldStager` is built from the same permuted order, so look-ahead still matches. The **stitched SBS windows** (`objects`) and Spatial Mask Generation keep acquisition order: windows overlap and are not independent fields. They get §2.1/§2.2 sizing only.
- **Recorded in the run journal:** per field, `_record_resolved_parameter(settings, field, 'mask_batch', source=<field fingerprint the normaliser already takes>, configuration={'request', 'mask_batch_seed', 'randomize', 'budget_source', 'cap'}, value={'batch': i, 'position': j, 'size': n_i, 'tiles': tiles})`. A replay then refuses a different draw, as it already does for normalisation records. Also, at run level, `batch_plan.json` gets `seed`, `order_sha256` (SHA-256 of the newline-joined permuted names) and `sizes`. A settings rerun with the same seed, field set and budget gives the same batches. A rerun on a different card with no declared budget can get different **sizes**: the console and the record say so, and the declared budget is the way to pin them.
- **Normalisation consequence:**
  - **Mask Generation:** each batch's percentiles are pooled over a **random sample of the plate's fields**, not over neighbouring fields in acquisition order. Every batch is then a draw from the same population, so plate-, well- and time-of-acquisition drift is spread across batches instead of being absorbed per batch. Batches of 213–214 random fields should give near-identical percentiles. That is the maintainer's "everything normalised the same way, or as similarly as possible", and it is tested (§4).
  - **OPS:** each field is normalised alone by Cellpose, so batch composition cannot change an OPS mask. For OPS the gains are speed (a full tile batch, VRAM-filling resident batches) and one shared code path. The test is byte-identity of masks between random and acquisition order at a fixed tile batch.

### 2.4 Tile batch: VRAM-aware with a cap of 32

- `mask_eval_batch_size` gets the new default `'auto_vram'`: the largest tile batch that fits B (§2.1), up to the new key `mask_eval_batch_cap` (int, default **32**, ≥ 1). Explicit positive numbers are unchanged: the governor still checks them, waits for memory, and then shrinks. 0 keeps "one tile per field in the batch".
- `ops_ph_tile_batch`, `ops_sbs_tile_batch` and `spatial_tile_batch` keep default 0 (= use the shared key), so they follow `'auto_vram'` with no change of their own. `_ops_tile_batch` and `_eval_tile_batch` resolve `'auto_vram'` through the one calibration result held on the plan, so all of them read one number.
- A field with fewer tiles than the resolved batch runs at its tile count. Cellpose cannot batch more tiles than an image has, so memory stops growing there. The record shows the resolved number and the tile count.
- Masks need not match the old frozen ones; the maintainer wants the best masks. Evidence for the cap: bf16 at 32 tiles gives F1 0.997 and 0.18 % of mask pixels changed vs 8 tiles on Home's Cellpose 4.2.1.1, and identical results on the WS's 4.0.9 (`20261119_final_funk_runs.md` §0.3, S2).

### 2.5 Explicit old behaviours

| `batch_size` value | Meaning after 1011 |
|---|---|
| `'auto_vram'` (**new default**) | §2.1 + §2.2 + §2.3 |
| 0, -1, None, `'none'`, `'auto'`, and the new alias `'whole'` | item 987b unchanged: largest batch that fits the free memory by the constant estimate (the whole plate or well on a large card), full batches + borrowed tiny last batch |
| positive number | unchanged (reduced with a warning when it does not fit) |

`_batch_size_request` returns the string sentinel `'auto_vram'` for the new mode. `_legacy_batch_size` maps it to 50 for the 987c archive routes, as it does for 0 today. `spacr.validate` (`_AUTOMATIC_BATCH_APPS`, item 1003) accepts it. `mask_eval_batch_size` gets type `(int, str)` and accepts `'auto_vram'`.

## 3. Settings and schema 955

New keys (Mask defaults; the shared ones are also in OPS defaults and `_OPS_MASK_SETTINGS`):

| Key | Default | Type | Where |
|---|---|---|---|
| `batch_size` | `'auto_vram'` (was 0) | `(int, str)` | Mask, OPS |
| `mask_eval_batch_size` | `'auto_vram'` (was 0) | `(int, str)` | Mask, OPS |
| `mask_eval_batch_cap` | 32 | int ≥ 1 | Mask, OPS (alpha control) |
| `mask_batch_seed` | 42 | int | Mask, OPS (alpha control) |
| `gpu_vram_budget_gb` | None (→ `SPACR_GPU_VRAM_GB`, then the card) | float or None | governor `_DEFAULTS`, Mask, OPS (alpha control) |
| `randomize` | True (Mask unchanged) | bool | now also OPS |

- `DEFAULT_CHANGES[955] = {'batch_size': 0, 'mask_eval_batch_size': 0}`. A settings file or run journal stamped before 955 (or unstamped) that does not set them keeps 987b's automatic batch and one tile per field, so its masks are unchanged. Old OPS files keep their per-path 8/16/16 through 948 as now. `mask_eval_batch_cap`, `mask_batch_seed` and `gpu_vram_budget_gb` need no rows: they are read only under `'auto_vram'`, which an old file never gets.
- **Seeding the existing shuffle:** for Mask files with `randomize` True, the old shuffle was unseeded, so it was never reproducible. Seeding it with 42 changes no reproducible output, so it needs no row. The reproducibility manifest notes it.
- `SETTINGS_SCHEMA_VERSION = 955`. The `_fold_older_defaults` docstring gets the 955 paragraph. `docs/source/reproducibility_manifests.rst` gets the row.
- Tooltips (`settings.py` descriptions) for all six keys. `batch_size` and `mask_eval_batch_size` say what `'auto_vram'` does and how to get the old behaviour (`batch_size: 0` or `'whole'`, `mask_eval_batch_size: 0`). The OPS descriptions (`ops_settings.py` 1449, 1772) say that 0 now follows the shared VRAM-aware tile batch.
- `ALPHA_FEATURES[1011] = {'settings': ('mask_eval_batch_cap', 'mask_batch_seed', 'gpu_vram_budget_gb')}`. `batch_size` and `mask_eval_batch_size` stay visible: they are not new.
- Arrival contract: `_calibrate`, `_even_sizes` and `_field_order` are internal (`_`-prefixed), so they are not documented callables. The new **settings** ship with their `setting_api_targets`/`help_api_index`/`settings_flow_index` entries in the same commit (the documented-callable and settings-index ratchets). If the implementation adds any public callable, it ships with its arrival contract and 9-locale records in that commit.
- Translations: the new and changed tooltip and console strings go to a Codex task, `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_1011_task.md` (sources with SHA-256, 9 locales), written at implementation. This item writes no translations.

## 4. Tests and acceptance

**Unit (CPU, `CUDA_VISIBLE_DEVICES=''`, `pytest -n 4` at most). New file `tests/test_mask_batches_vram_auto_1011.py`:**
1. `_even_sizes`: for n in 1..2000 and max in 1..300, `sum == n`, `max − min ≤ 1`, `max(sizes) ≤ max_fields`, `len == ceil(n/max)`. Example: 1,281 fields with max 300 gives 5 × 256–257, not 4 × 300 + 81.
2. Determinism: the same `mask_batch_seed` and field set give the same `order_sha256` and batches across two processes (`random.Random` per call, not global state). A different seed gives a different order. `randomize=False` gives acquisition order. Timelapse forces it off.
3. Calibration fit on a fake governor (injected `_allocated`/`_reserved` with a known linear model) recovers `per_tile`/`fixed`. The resolved `tiles` respects `mask_eval_batch_cap`, B and the 1-tile floor. A declared `gpu_vram_budget_gb` or `SPACR_GPU_VRAM_GB` wins over a larger free VRAM. The ledger reservations of other processes are subtracted.
4. CPU fallback: with no CUDA, there is no probe call, tiles = min(cap, 8), fields are RAM-limited and capped at 50, and the record says `cpu`.
5. Request parsing: `'auto_vram'`, 0, `'whole'`, positive numbers. `_legacy_batch_size` maps `'auto_vram'` to 50 for the 987c routes. The validator accepts the new defaults (extend `tests/test_default_settings_validate_1003.py`).
6. Schema: a file stamped 954, and an unstamped file, get `batch_size` 0 and `mask_eval_batch_size` 0. A 955 file gets `'auto_vram'`. An explicit value is never touched.
7. Journal: a replay with a different seed or field set is refused with the existing resolved-parameter message. Resume keeps the plan without calibrating again (the probe count is 0 on resume).
8. OPS: under `'auto_vram'`, `_iter_array_batches` groups follow `_even_sizes`, the stager order equals the segmentation order, and stitched windows keep acquisition order.
9. Existing 987b tests (`test_mask_batch_size_auto_987b.py`, `test_mask_batches_stay_in_memory_987b.py`, `test_mask_batch_first_987b.py`, `test_gpu_memory_governor.py`) pass unchanged when they pin `batch_size: 0`. Any test that relied on the old default pins 0 explicitly.

**GPU acceptance (paper lane, through `tools/gpu_turn.sh --vram-gb`, 280 W cap; design card and sign-off first):**
- **A. Funk A1, the pilot's fields** (the 4 check fields of the pilot's seed 20261121, plus 32 more drawn with the same seed for counts): `'auto_vram'` vs the frozen reference (`<pilot root>/frozen_ref_masks/`), with the pilot's own measures (`pilot_acceptance.py` C1, `spacr.ops_large_sites.mask_agreement`): per-field F1 at IoU 0.5 and 0.9, % of foreground pixels changed, identity and displaced controls, and nuclei counts per field position (site index in acquisition order). **Criterion: within 0.2 % of mask pixels of the tile-batch-8 masks at the same precision, F1@0.5 ≥ 0.99**, with no position trend in the count difference (slope CI includes 0).
- **B. Small-GPU case:** the same fields under `--vram-gb 6`, which gives `SPACR_GPU_VRAM_GB=6` and the governor fraction. The run must not OOM or fall back to the CPU. The resolved tile batch is recorded (expected below 32 if the §6 estimate holds). The masks meet the same 0.2 % criterion vs A. A second case with `--vram-gb 3` shows the floor behaviour.
- **C. Normalisation (Mask Generation route, `spacr.core.preprocess_generate_masks` on the A1 fields exported as a Mask project):** random vs sequential (`randomize=False`) batches at equal sizes. Report the per-batch percentile spread (lower/upper per channel: SD and range across batches, with 500 seed permutations, mean and 95 % CI), and mask quality vs the frozen reference (F1@0.5, nuclei per field position). Pre-set criterion: random batches have a percentile spread ≤ sequential, and F1 ≥ sequential minus 0.001. Report the sequential row as the control.
- **D. OPS order invariance:** ph_segment masks with random vs acquisition order at a fixed tile batch are byte-identical.
- **E. Time:** ph_segment seconds per field, `'auto_vram'` vs 8 tiles, on the same fields and the same card state. Report it, with no gate.
- **F. Mask Generation route (same plan as OPS, decision 2):** Funk plate A well A1 run through `spacr.core.preprocess_generate_masks` with the new batching (`'auto_vram'`) vs the current path (`batch_size: 0`, `mask_eval_batch_size: 0`), on the same fields, the same seed and the same card state. Pre-set criterion: masks agree within 0.2 % of mask pixels and object F1 at IoU 0.5 >= 0.99 (same measures as A: `spacr.ops_large_sites.mask_agreement`, identity and displaced controls). Run the two memory cases of B: `--vram-gb 6` and `--vram-gb 3`. No OOM and no CPU fallback; the resolved tile batch and batch sizes are recorded, and the masks meet the same criterion. Percentile spread, as in C: report the spread of the normalisation percentiles across batches (lower/upper per channel: SD and range across batches, 500 seed permutations with mean and 95 % CI), for the new batching and for the current path, with the sequential row as the control. Pre-set criterion: the new batches' percentile spread <= the current path's. The OPS route (A, B, D) and this route must show the same resolved plan for the same inputs, since both read one planner.

## 5. Size and lanes

About 450–600 changed lines in `spacr/` plus about 400 test lines. 1.5–2 days of lane time, plus one GPU session for §4 A–E.

| Lane | Owner | Files (exclusive) |
|---|---|---|
| **L1 core plan + governor** | WS Claude (Opus) | `spacr/_mask_batches.py` (`_calibrate`, `_even_sizes`, `_field_order`, `_batch_size_request`/`_legacy_batch_size`, `_resolve_plan`, plan record), `spacr/_gpu_governor.py` (`gpu_vram_budget_gb`, env, fraction), `spacr/_mask_array_batches.py` (`_batch_limit`/`_iter_array_batches`/`_batch_plan` under `'auto_vram'`), `tools/gpu_turn.sh` + `tools/gpu_turn.ps1` (export `SPACR_GPU_VRAM_GB`), the new test file |
| **L2 wiring** | Codex | `spacr/io.py` (the two shuffles → `_field_order`), `spacr/core.py` (calibrate after the model loads, before `_resolve_plan` fixes sizes), `spacr/object.py` (`_eval_tile_batch` reads the resolved tiles), `spacr/ops_engine.py` (`_OPS_MASK_SETTINGS`, `_ops_tile_batch`, ph_segment order + stager, calibration at the first resident batch) |
| **L3 settings/schema/docs** | Codex | `spacr/settings.py` (defaults, types, tooltips, `DEFAULT_CHANGES[955]`, schema 955, `ALPHA_FEATURES[1011]`), `spacr/ops_settings.py`, `spacr/validate.py`, the `spacr/cli` type mirror, `spacr/qt/screens/settings_flow_index.py`, `setting_api_targets.py`, `spacr/qt/help_api_index.py`, `docs/source/reproducibility_manifests.rst`, extended 1003 test |
| **L4 translations** | Codex tr | `tr_1011_task.md` → `spacr/qt/i18n_catalogs/*.py` after L3 freezes the strings |
| **L5 acceptance** | Home paper session (GPU) | scripts and outputs under `BMCspacr/reanalysis/`; no spaCR files |

L1 lands first: L2 and L3 import its names, and L2 and L3 can start against its signatures (`_calibrate(model, sample, settings, *, site) -> dict`, `_even_sizes(n, max_fields) -> list[int]`, `_field_order(names, settings) -> list[str]`). Before handover, run the raw-reader, nested-function, documented-callable and hash-comment ratchets, and warm the API catalogs.

## 6. Does a tile batch of 32 depend on VRAM?

Yes. The network's working set grows with the number of tiles per pass, so 32 tiles need more card memory than 8.

- **Measured:** 1.716 GB torch allocated peak at tile batch 8, half resolution (`ops_ph_downsample` 2), cpsam bf16, spaCR 4792e9d57e. That was the frozen_ref 2-field test of 2026-11-21 (`final_funk_runs_1120/scripts/common.sh`, frozen_ref line).
- **Extrapolated (not measured):** about 0.6 GB of that is cpsam's bf16 weights. This is an estimate from the model size, about 305 M parameters, not a measurement. If all of the remaining ~1.1 GB scales with tiles (an upper bound), one tile costs about 0.14 GB, and 32 tiles give about **5.0 GB** torch peak. If half of it is fixed image and flow buffers, 32 tiles give about **2.9 GB**. So the range is roughly 3–5 GB allocated at 32 tiles.
- **Card need:** the governor sizes the card need as max(reserved, 1.5 × allocated) + 0.4 GB context + 1.5 GB margin. That is about 6.5–9.4 GB at the top of the range. spaCR's own constants give 1.3 + 32 × 0.2 = 7.7 GB plus the margin.
- **Consequence:** 32 fits the maintainer's 24 GB card easily. It is borderline or too much under the Funk runs' 6 GB per-job cap, and too much for a 4 GB card. The governor would not crash such a card, but it would wait and then shrink mid-run, so the masks would depend on what else was on the card. `'auto_vram'` with `mask_eval_batch_cap` 32 resolves the largest batch that fits once, up front, from a measurement on the run's own data, and records it.
- §4 B replaces this extrapolation with a measurement.
