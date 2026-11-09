# Item spec: plaque figure mode: measured-vs-reported scoring, well merge, and public entry points for the private plaque functions

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related: 843–849 (plaque figure mode, open, alpha), 850.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P20).

## Why
Measuring plaque effects (size, number) from published plaque-assay figures and scoring them against the effects the papers report, plus merging the measured wells into one meta-analysis table, is done by custom scripts. These scripts:
- import private spaCR functions (`plaque_papers._zoo_path`, `_model_well_resize`, `_cellpose_segmenter`, `spacr.plaque._segment_resized_wells`, `model_zoo._model_precision_kwargs`);
- run from a detached **origin/main** worktree (`/mnt/wd4tb/spacr-worktrees/plaque-meta-main`), not nightly.

## Reference implementation
`R/plaque_meta/code/`:
- **Detection:** `detect_all.py` (`find_plaque_regions`), `redetect.py` (`spacr.plaque.detect_wells`), `snap_tiles.py`.
- **Segmentation:** `seg_wells.py`, `gpu_seg_job.py`, `run_gpu_seg.sh` (plaque v8; `fetch_v8.py`).
- **Figure mode:** `run_figure_mode.py` / `run_fm_text.py` (`plaque_papers.measure_figure_folder`).
- **Scoring:** `score_figure_mode.py` (`iou`, `measured_by_condition(f, metric, labeler)`, `spearman`, the permutation control), `score_lib.py` (`norm`, `alnum`, `strain_key`, `is_wt`, `treat_tokens`, `strain_match`, `treat_match`, `cond_match`) → `results_measured_vs_reported*.csv`.
- **Merge:** `merge_wells.py` (`add`).
- **Labels:** `gold.py`; the stored model readings of figure conditions in `R/plaque_meta/vlm_agents/responses/` (**data, not code**: store them, do not re-derive them).
- **Design:** `R/plaque_meta/DESIGN.md`.

## What to build
1. **Public entries** in `spacr.plaque_papers` / `spacr.plaque`: `segment_wells(images, model, settings)` (wrapping `_segment_resized_wells`, `_model_well_resize`, `_cellpose_segmenter`) and `resolve_plaque_model(name)` (wrapping `_zoo_path` / `_ensure_model_file`), with documented settings:
   - `plaque_model` ('toxoplasma_plaque_v8');
   - `plaque_well_resize`;
   - `plaque_precision`.
2. **Per-figure effect table** (846): `measured_by_condition` as a public function. Settings:
   - `figure_metric` ('area' | 'number' | 'cleared_fraction');
   - `figure_condition_labels` (path to a condition table: gold or model-read);
   - `figure_reference_condition` ('parent/WT');
   - `figure_effect` ('log2_ratio').
3. **Measured-vs-reported scoring:** `compare_reported(measured, reported, settings)` with `compare_permutations` (as in the reference) and `compare_seed`. It writes `results_measured_vs_reported.csv` and `summary.json` (Spearman ρ per readout + permuted controls). The condition matcher is `score_lib.cond_match` as a documented function.
4. **Well merge** (844/849): `merge_figure_wells(...)`, as `merge_wells.py`, writing the meta-analysis table.
5. Run all of this from a plaque figure-mode settings file in the Plaque module. The LLM provider stays injected, with no default and no API (item 843 rule).

## Validation data and acceptance
- **Data:** the gold figure set and the contrasts in `R/plaque_meta/results_measured_vs_reported.csv`, plus the stored model readings.
- **Acceptance:** identical output, or within the stated tolerance, to the reference outputs in `R/plaque_meta/`:
  - identical per-condition measured values from the same masks;
  - identical ρ per readout and identical permuted controls with the same seed (`results_measured_vs_reported*.csv`);
  - identical merge table (well, effect and reference-well counts) to the `merge_wells.py` output from the same inputs.
- The masks themselves must come from `segment_wells` on nightly. Rerun the reference wells with v8 and compare against the origin/main-worktree masks; areas rtol 1e-3, label counts exact. Record the commit.

## Effort
- **Effort:** M.
