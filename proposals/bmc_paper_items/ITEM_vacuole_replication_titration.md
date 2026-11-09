# Item spec: HRMAn-style infection and replication readout: vacuole grouping by gap, replication classifier from an external training table, size calibration and uninfected-signal floor

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related: 829 (parasites per vacuole from shape, open), 823 (size-proxy calibration, open), 815 (replication report, done).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P18).

## Why
A Toxoplasma IFNγ titration readout (% infected, % replicating, relative vacuole size, the large-vacuole tail) on the published HRMAn data (Fisch et al. 2019) is computed by `titration_summary.py` and helpers outside spaCR. Segmentation and measurement were spaCR (Make Masks and Measure through `run_masks.py` / `run_measure.py`; settings saved); the per-well % infected came from spaCR's `infection_report`, reshaped ad hoc.

## Reference implementation
`/mnt/firecuda2/hrman_fig6/scripts/` (copy in `R/hrman/code_archive`):
- **`vacuoles.py::vacuoles_from_parasites`:** binary closing with a gap of `VAC_GAP` px (4; sensitivity 2), then label. Links via `spacr.host_pathogen.vacuole_links`. This replaces `utils.merge_touching_objects`, which is O(labels × pixels).
- **`titration_objects.py::one/main`:** regionprops per vacuole; host cell by majority overlap; MIN_CELL 300, MIN_VAC 15.
- **`titration_summary.py::main/plot`:**
  - `HistGradientBoostingClassifier` trained on HRMAn's published `Training_dataset.csv` (classes 1/2/4/8 parasites per vacuole), with cross-validation, an area-only model and a majority control;
  - `modal_area` and k = hr_single / sp_mode (the HRMAn single-parasite area over spaCR's modal single-vacuole area) → `titration_calibration.txt`;
  - TOX_FLOOR = 1000 (set from the 95th percentile of uninfected wells);
  - `well_meta`;
  - hard-coded HRMAn reference values (move them to a CSV as data, not code).
- **Inputs:** `work/gpu5/plate103_m/measurements/` (spaCR measurement db and `infection_report.csv`); `R/hrman/settings/titration/plate103*`.
- **Outputs:** `R/hrman/titration/{titration_per_well.csv, titration_per_condition.csv, titration_vacuoles.csv, spacr_infection_report_per_well.csv}`, `R/hrman/qc/replication_classifier_cv_on_hrman_table.txt`.

## What to build
In `spacr.host_pathogen` / `analyze_replication` (alpha):
1. **Vacuole grouping:** `vacuole_from_parasites` (bool), `vacuole_gap_px` (4), `vacuole_min_size` (15), `host_min_size` (300), `host_assignment` ('majority_overlap'). It runs inside `measure_crop` and writes a `vacuole` table.
2. **Replication classifier:**
   - `replication_classifier` ('none' | 'hist_gbm');
   - `replication_training_table` (path, e.g. HRMAn's `Training_dataset.csv`);
   - `replication_training_columns` (mapping to spaCR features);
   - `replication_classes` ('1,2,4,8');
   - `replication_seed`;
   - writes the CV report (accuracy, area-only, majority).
   - Build together with item 829 (shape-based counts), which this generalises.
3. **Size calibration:** `size_calibration` ('none' | 'modal_single'), `size_calibration_reference_px` (57.0 for HRMAn) → k (item 823 territory: calibration against an external reference value).
4. **Uninfected floor** in `infection_report`: `infection_signal_floor` ('none' | 'uninfected_percentile' | number), `infection_floor_percentile` (95), `infection_uninfected_wells` (list).
5. **Per-well / per-condition outputs** (item 815 report): the four CSVs above with the same columns, including the vacuole size distribution × k.

## Validation data and acceptance
- **Data:** HRMAn plate103, the IFNγ titration (`/mnt/firecuda2/hrman_fig6/` raw; spaCR db under `work/gpu5/plate103_m`).
- **Acceptance:** identical output, or within the stated tolerance, to the reference outputs in `R/hrman/titration/` and `R/hrman/qc/`:
  - From the existing masks: `vacuole` table identical to `titration_objects.py`; per-well and per-condition CSVs identical; CV report identical (same seed); k identical to `titration_calibration.txt`; the floor and the uninfected percentile identical to the reference.
  - End to end from the saved settings (`R/hrman/settings/titration/plate103`, `plate103_m`): masks within the GPU self-diff floor; downstream counts identical when the masks are identical.

## Effort
- **Effort:** M.
