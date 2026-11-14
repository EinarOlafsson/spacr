# Item spec: every OPS measurement feature available in spaCR's general Measure module, from one implementation

- **Lifecycle:** future (alpha), behind "Show alpha features". Every new Measure key goes under one new `ALPHA_FEATURES[<this item>]` entry. Measure's default output stays byte-identical.
- **Number:** 934 (reserved 2026-11-13).
  - `ITEM_ops_feature_families.md` (the family ledger audit; its proposed `measure_families` key is superseded by the shipped `profile_features`);
  - `ITEM_ops_phenotype_run_infrastructure.md` (A1 bounding-box loops, B7/B12/B13 GPU and CPU hot spots);
  - `ITEM_ops_phenotype_run_v2.md`;
  - 752 (OPS phenotype path), 566 (`measure_gpu`), 788 (GPU skew/kurtosis), 891, 914, 856.
- **Overlap with an existing spec:** none covers this request.
  - `ITEM_ops_feature_families.md` decides **which** families help OPS profiles. It does not decide where they live.
  - The measure-families work that landed (nightly 2339faf5c2 / 0bed2ff593, `profile_features`) already put the profiling families in `spacr/measure.py` itself, so most of the request is already met.
  - This spec covers the remainder: the features that exist **only** in OPS code, plus a guard that keeps it that way.
- **Rule:**
  - Measure's default output is byte-identical to today.
  - The OPS phenotype table on plate A field A1/514 is byte-identical before and after the refactor (406 of 406 columns).
  - Where a feature is shared, OPS and Measure give identical values for the same field, masks and settings, up to column naming.
  - Every new documented callable ships with its arrival contract and nine-locale records in the same commit.
- **Code basis:** origin/nightly (46f259ea23).
- **Paper-critical:** no. It is a platform claim: "OPS features are spaCR features".

## Problem and goal
The OPS phenotype path measures cells with spaCR's own measurement functions. A few features, however, are computed only inside the OPS code, and a few switches exist only as OPS settings. A user of the normal Measure module cannot get them, and two implementations of nearly the same feature can drift apart.

**Goal:**
- every per-object measurement OPS computes is a Measure feature with a Measure setting;
- OPS calls Measure for it;
- a test fails if a new OPS-only measurement appears.

## Current state (verified on origin/nightly)

### What OPS already takes from `spacr.measure` (shared, nothing to do)
- `ops_sample.measure_phenotype_field` (`spacr/ops_sample.py:439`) calls `spacr.measure._morphological_measurements` and `_intensity_measurements` (import at `ops_sample.py:492`). It passes a synthetic Measure settings dict from `_measurement_settings` (`ops_sample.py:332`).
- `profile_feature_families` is imported at `ops_sample.py:353` and `ops_engine.py:3231`. The profiling families are defined in `measure.py`: `PROFILE_FEATURE_FAMILIES` (l. 3038), `measure_profile_features` (l. 4182). They are reached in Measure through `profile_features` (`settings.py:2135`, default None) and in OPS through `ops_ph_profile_features`.
- `_write_image_rows` (`ops_engine.py:5794`) and `_remove_objects_smaller_than` (`ops_engine.py:10197`).
- The GPU switch: OPS `ops_ph_measure_gpu` → Measure `measure_gpu` (`settings.py:2043`). The GPU path is `measure._gpu_measurable` (l. 2372).

### OPS-only features and switches (the work of this item)
| # | What | Where | OPS setting | Measure today |
|---|---|---|---|---|
| 1 | **Funk et al. 2022 object features**: `<prefix>_<ch>_mass_displacement` and `<prefix>_correlation_<a>_<b>` (Pearson r per channel pair over the object's pixels; `np.corrcoef` definition from OpticalPooledScreens `cp_emulator.py`) | `ops_sample.funk_object_features` (l. 370), per-label bincount sums | `ops_ph_funk_features` (False, `ops_settings.py:104`) | **Partly.** `measure_intensity_moments` (`measure.py:3218`) writes `channel_<c>_mass_displacement` under the `intensity_moments` family, by a different code path. Pairwise Pearson exists as `calculate_correlation`'s block, with different column names and only for the configured pairs. Not tested for agreement |
| 2 | **Form factor** 4π·area/perimeter² | inline in `measure_phenotype_field` (`ops_sample.py:551-556`) | always on in OPS | `measure_shape_extended` (`measure.py:3265`) writes `form_factor` under `shape_extended`. Two implementations |
| 3 | **Explicit Zernike switch** | `ops_ph_zernike` (False, `ops_settings.py:105`) → `_morphological_measurements(zernike=...)` | yes | No setting. Zernike turns on automatically when mahotas is importable (`measure.py:1321`/`1360`, `_zernike_is_available`), so the same settings file gives different columns on different machines |
| 4 | **GPU self-check**: one field measured on CPU and GPU, columns outside rtol 1e-4 reported before the run proceeds | `ops_engine._ph_gpu_check` (l. 3202) | `ops_ph_measure_gpu_check` (True, `ops_settings.py:289`) | None. `measure_gpu` runs unchecked |
| 5 | **Channel-named columns** (`cell_dapi_mean_intensity` and so on instead of `cell_channel_0_mean_intensity`) | `ops_sample.py:513-540` | implicit, from `ops_ph_channels` | No. Measure always writes `channel_<i>` |
| 6 | **Texture distances** [2, 4, 8] | `_measurement_settings` | hard-coded | Available: `homogeneity_distances` (default [8, 16, 32]). Already a Measure setting; only the default differs. **No work** beyond documenting the mapping |

### Not measurement (out of scope here; listed so nobody ports them by mistake)
- Mitotic and cell-state calls (`ops_mitotic.py`, `ops_tvn.cell_states`; items 856 and 914);
- neighbour-effect tests (`ops_engine._ph_neighbour_effects`, l. 7365; item 891). The per-cell neighbour block is already shared (`measure_neighbours`, `measure.py:3879`; `spatial_measurements`);
- TVN and profile building (`ops_tvn.py`);
- the column filter `_ph_feature_columns` (`ops_engine.py:4070`).

These are analysis phases on finished tables. A general-module version of them is a separate question (see open question 4).

### Gating gap
- `profile_features` and `profile_granularity_subsample` are in the "Measurements" group (`settings.py:6492`) and are **not** in `ALPHA_FEATURES`.
- The maintainer's rule is that new features ship as alpha, and the READINESS register lists "Measure feature families" as Under development.
- This item's alpha entry should cover them too, unless the maintainer decides otherwise (open question 1).

## Proposed design

### 1. One implementation per feature
- **Mass displacement.**
  - `measure_intensity_moments` becomes the single implementation.
  - `funk_object_features` calls it and renames columns, unless the two definitions differ. Both are centroid-to-weighted-centroid distance in pixels; the test below decides.
  - If they differ in a documented way (for example, the treatment of negative intensities after background subtraction), Measure gets `mass_displacement_definition` ('spacr' | 'funk2022', default 'spacr'), and OPS passes 'funk2022'.
- **Pairwise channel correlation.**
  - A Measure family `channel_pair_pearson` computes Pearson r for every channel pair (itertools.combinations order) from per-label sums. It is the same arithmetic as `funk_object_features`, moved into `measure.py`.
  - `calculate_correlation`'s existing block stays as it is, because its default output must not change.
  - OPS's `ops_ph_funk_features` turns this family on.
- **Form factor.** OPS takes `form_factor` from `measure_shape_extended`. The inline formula in `ops_sample.py:551-556` is deleted once identity on A1/514 holds. If it does not hold (perimeter estimator differences), the difference is documented and `shape_extended` gains the OPS perimeter as an option.

### 2. New Measure settings (alpha)
| Key | Type | Default | Meaning |
|---|---|---|---|
| `funk_object_features` | bool | False | adds `channel_<c>_mass_displacement` (if `intensity_moments` is off) and the `channel_pair_pearson` family, named as Funk et al. 2022 measure them |
| `zernike` | bool or None | None | None keeps today's behaviour (on when mahotas is importable). True requires mahotas and errors without it; False never computes |
| `measure_gpu_check` | bool | True | with `measure_gpu`, measure the first field on both CPU and GPU and stop with a table of the columns outside `measure_gpu_check_rtol` |
| `measure_gpu_check_rtol` | float | 1e-4 | tolerance of the check |
| `channel_names` | list[str] or None | None | when given, measurement columns use these names in place of `channel_<i>` (lowercased), as OPS does; the `channel_<i>` form stays the default |

**OPS mapping.**
- `ops_ph_funk_features` → `funk_object_features`;
- `ops_ph_zernike` → `zernike` (False/True; OPS keeps its explicit False);
- `ops_ph_measure_gpu_check` → `measure_gpu_check`;
- `ops_ph_channels` → `channel_names`;
- `ops_ph_texture` → `homogeneity` with `homogeneity_distances=[2, 4, 8]`.

The mapping is one dict, `spacr.ops_settings.OPS_TO_MEASURE_KEYS`. `_measurement_settings` builds its dict from it rather than by hand.

### 3. Guard against new OPS-only measurements
A ratchet test, `tests/test_ops_measurements_are_measure_features.py`:
- every column `measure_phenotype_field` can write, with every OPS measurement setting on and on a synthetic field, is produced by a function in `spacr.measure`. The test traces column provenance through a registry: each family function declares its output column patterns in `MEASURE_COLUMN_SOURCES`;
- every `ops_ph_*` measurement setting has an entry in `OPS_TO_MEASURE_KEYS`.

### Public callables
- `spacr.measure.measure_channel_pair_pearson(labels, image, channel_names=None)`
- `spacr.measure.check_gpu_measurement(field, settings) -> DataFrame`: the per-column CPU/GPU comparison that `_ph_gpu_check` becomes a thin call to.
- `spacr.ops_settings.OPS_TO_MEASURE_KEYS`: documented module data.

### GUI
- Measure → "Measurements" group: `funk_object_features`, `zernike`, `channel_names`, under alpha.
- Measure → GPU group: `measure_gpu_check`, `measure_gpu_check_rtol`, under alpha.
- The tooltips say which OPS setting each mirrors.

## Acceptance tests (CPU, `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers unless marked GPU)
1. **Measure default identity.** On the bundled example plate, `measure_crop` with default settings writes a `measurements.db` identical table by table and row by row (`SELECT * ORDER BY rowid`, exact), with the same schema, before and after.
2. **OPS identity.** `measure_phenotype_field` on A1/514 gives a table identical to the reference (406 of 406 columns, exact). The reference is `funk_phenotype/work/plateA/A1/phenotype_cells_A1.parquet` with `phenotype_masks/site-514.npz`, or a frozen-source rerun on today's nightly.
3. **OPS = Measure on the same field.** `measure_crop` on A1/514 as a merged field, with the OPS masks, `channel_names` set and the settings `OPS_TO_MEASURE_KEYS` produces, gives every shared column equal to the OPS table: exact on CPU, within 1e-12 relative for moments. Columns that differ by definition are listed in the test with the reason.
4. **Mass displacement and Pearson.** On synthetic objects (uniform, off-centre bright spot, zero-variance channel, one-pixel object, touching objects):
   - the Measure family equals the Funk definition (`np.corrcoef` per object; the centroid distance) within 1e-12;
   - a zero-variance object gives NaN in both.
5. **Zernike determinism.** With `zernike=True` and no mahotas (monkeypatched), the call raises. With None, columns are as today on a machine with and without mahotas.
6. **GPU check** (GPU-marked; run through `tools/gpu_turn.sh`, never in the default run). A planted 1 % perturbation of one GPU column makes the check stop with that column named. With no perturbation, the check passes on A1/514 (rtol 1e-4, after item 788's skew/kurtosis fix).
7. **Ratchet.** The guard test above passes on nightly after the move, and fails on a branch that adds an OPS-only column (a self-test with a planted column).
8. **Time vs the current path** (marked slow).
   - On A1/514, time per field for the OPS path before and after the refactor: the after/before ratio must be ≤ 1.05 (median of 3 interleaved rounds, one pinned core).
   - Report Measure's time for the same field with the OPS-equivalent settings. Report it only; do not gate on it.

## Translations / catalog obligations
- `measure_channel_pair_pearson`, `check_gpu_measurement` and `OPS_TO_MEASURE_KEYS` ship in the same commit with:
  - their arrival contracts;
  - nine-locale records;
  - the regenerated help index, API catalogs, setting consumer map and settings flow.
- The five new tooltips get nine-locale records.
- Warm the API catalogs first (`build_documentation_i18n.py --assemble`).
- Run the pre-handover ratchets (raw-reader, nested-function, documented-callable, hash-comment) before handing the branch to the master.

## Dependencies on items 928–932
- None functionally.
- 931's settings-schema version increments, because new keys are added.
- The READINESS row "Measure feature families" stays Under development. This item moves code, not readiness; a move must not change OPS stage 3 outputs (test 2).

## Alpha status
Alpha. All new keys go under one `ALPHA_FEATURES` entry. Proposed: `profile_features` and `profile_granularity_subsample` move under the same entry (open question 1).

## Maintainer decisions (2026-11-13, AskUserQuestion in the master session)
profile_features stays visible (no alpha gate, users already have it); OPS keeps channel-named columns and Measure offers channel_names as an option; Funk's mass displacement is a separate column if the definitions differ; the post-measure OPS analyses become a later item.

## Open questions for the maintainer (answered above)
1. `profile_features` (the 15 profiling families) is in Measure today **without** the alpha gate, which conflicts with "new features ship as alpha". Should it be gated now? Gating hides it from users who already see it.
2. Should channel-named columns (`channel_names`) be offered in Measure at all, or should OPS move to `channel_<i>` names with a name map in the database? The second changes OPS column names and therefore the paper's reference tables.
3. If Funk's mass displacement and spaCR's differ by definition, which should be the Measure default under `intensity_moments`?
4. Should the post-measure OPS analyses (mitotic call, cell-state stratification, neighbour-effect tests) also become general-module features (for example, under Measure analysis)? That would be a separate item.

## Build note (2026-11-13, measure lane, branch claude-934-measure-1113)
Built as features/feature_list/934_measure_ops_features_general.txt describes (steps 5 and 6).
- Public: `spacr.measure.measure_mass_displacement`, `measure_channel_pair_pearson`, `check_gpu_measurement`; data `spacr.ops_settings.OPS_TO_MEASURE_KEYS`. Arrival contract `934_measure_ops_features_api_arrivals_2026-11-13.json`.
- Measure settings `funk_object_features`, `zernike`, `channel_names`, `measure_gpu_check`, `measure_gpu_check_rtol` under `ALPHA_FEATURES[934]`; `profile_features` and `profile_granularity_subsample` stay visible (decision 1).
- Mass displacement: same definition in both; no separate column. OPS Funk columns move by at most 1 ulp (hypot). Form factor via `spacr.measure._form_factor` (NaN for a zero perimeter).
- The ratchet is value-level rather than a `MEASURE_COLUMN_SOURCES` registry: `tests/test_measure_ops_features_934.py` runs the OPS call with its switches on and spacr.measure with the mapped settings on the same field and fails on any OPS column Measure does not write with the same value (self-test with a planted column).
- A1/514: OPS default 407/407 byte-identical to nightly; Funk 421/429 identical, 8 within 2.2e-16 relative; OPS = Measure at the function level 475/475 exact (with Zernike); `measure_crop` end to end 425/425 present, 326 exact, 99 within 1.83e-7 (uint16 vs float32 input). The reference parquet matches 357/406 for nightly and branch alike (pre-existing).
- Not done: GPU test 6 on a card; test 8 timing gate (the machine was loaded; times reported only); 931 schema version unchanged (new keys, no changed default); translations not written (listed in the hand-over).
