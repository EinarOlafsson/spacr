# Item spec: division prediction: scoring unit and minimum-positives guard, shape-and-motion arm, single-movie grouping, reference features, learner and label options

- **Lifecycle:** future (alpha), behind "Show alpha features". New settings go under ALPHA_FEATURES[866] (or the new item number, registered beside 866).
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - 866: division prediction (`spacr.division_prediction`), done, alpha;
  - 867: per-movie bootstrap, done, alpha;
  - 771: division-aligned trajectories, done, alpha;
  - 864 (units), 865 (budding cycle model), open;
  - `ITEM_division_onsets.md` (onset step; no overlap).
  This item extends 866/867. It does not replace the module; every new setting defaults to today's behaviour.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data, and within the tolerances below only where the reference drew from a random stream that a settings file cannot replay (row subsample, label shuffle). Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **CPU only.**

## Why
Predicting whether a tracked cell divides within N frames, with the controls that make the number interpretable, is done for 13 public time-lapse datasets by an analysis that is not yet in spaCR. Item 866 added `spacr.division_prediction`; run on the same datasets with its defaults it computes a different quantity, for five reasons, and each is an option spaCR lacks:
1. **Scoring unit.** spaCR keeps one labelled frame per track. The reference scores every frame, with a random row cap of 120,000. The two estimate different things: per-track vs per-cell-frame AUPRC. Per track, several small datasets keep fewer than 30 positive tracks, so their AUPRCs are not informative, and prevalence shifts. `one_per_track` is a function argument today, not a setting, and there is no row cap and no guard.
2. **Arms.** spaCR arms are model, age only, shuffled labels, majority. The key comparison, "shape and motion only" (all features except age and the born-in-view flag) against "age only" and "full", can only be built by passing `feature_columns` by hand.
3. **Single movie.** spaCR refuses a dataset with one movie (grouped folds need ≥ 2 movies). The reference groups folds by lineage tree there.
4. **Features and learner.** spaCR: current value, change over `history` 3 frames (log ratio for size), age. Reference: value, 1-frame difference, lag-frame difference, rolling mean over lag frames (lag = min(W, 5), W = alignment window), age and a born-in-view flag. Learner: spaCR xgboost depth 3, lr 0.1, default max_bin; reference depth 4, lr 0.05, max_bin 64.
5. **Labels.** spaCR counts every fission event and calls a frame negative when the lineage end lies past the horizon. The reference uses two-daughter divisions only and the last frame the track is seen.

## Reference implementation
- `stage2_analyse.py`:
  - `prediction_table(fovs, mode, horizon, lag, rng)`: per frame, for each measure in `da.COMPARED_MEASURES`: value, `_d1`, `_d{lag}`, `_roll{lag}` (rolling mean, `min_periods=1`); `age = t − t_start`; `born_observed = parent > 0 or t_start > first frame`. Positive when the next two-daughter division of the track is < horizon frames ahead. Kept when positive or `last_seen ≥ t + horizon − 1`. Cap `MAX_PRED_ROWS` = 120,000 (random subsample). Guard: run only if ≥ 20 positive rows.
  - `run_prediction(df, rng)`: `GroupKFold(min(5, n_groups))` by movie, by lineage root when there is one movie; arms full / age_only (`age`, `born_observed`) / morphology_motion_only / full_shuffled_labels (labels permuted within group); `XGBClassifier(n_estimators=200, max_depth=4, max_bin=64, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=1, random_state=20261104)`; out-of-fold AUPRC/AUROC; PR curves at 120 points.
  - Horizon N = max(2, round(0.15 × median cycle)).
- `run_spacr_division_prediction.py`: the same datasets run through `spacr.division_prediction` (feature sets passed by hand; lineage tree renamed to `movie` for the single-movie case; row cap applied outside spaCR), with its `settings.json`.
- **Outputs:** `panelE_prediction_summary.csv`, `panelE_pr_curves.csv`; `comparison_script_vs_spacr_module.csv` (number, reference value, spaCR default, spaCR every frame, rows, positives, grouping).

## What to build
In `spacr.division_prediction` (alpha), settings read by `_prediction_options` and passed through the Make Masks timelapse step. Defaults keep today's output.

1. **Scoring unit and minimum-positives guard**
   - `timelapse_division_prediction_unit` ('track' | 'frame'; default 'track'): 'track' = one labelled frame per track (today), 'frame' = every labelled frame.
   - `timelapse_division_prediction_sample_seed` (0): seed of the one-per-track draw (today's `sample_seed`).
   - `timelapse_division_prediction_max_rows` (None; 120000 in the reference settings): random row cap, applied after labelling and before folds.
   - `timelapse_division_prediction_row_seed` (0): seed of the cap.
   - `timelapse_division_prediction_min_positive_tracks` (30): fewer distinct positive tracks than this → the dataset is flagged, not dropped. `summary.json` gets `informative: false`, `n_positive_tracks`, and a plain-language note; the figure greys the bars; the metrics CSV gets an `informative` column.
   - `timelapse_division_prediction_min_positive_rows` (20): below this no model is fitted (the reference guard); the summary says why.
   - Always report both `n_positive_tracks` and `n_positive_rows`, and the unit, in `summary.json` and the metrics CSV.
2. **Shape-and-motion arm**
   - `timelapse_division_prediction_arms` (default 'model,age_only,shuffled,majority'; reference: 'model,age_only,shape_motion,shuffled,majority').
   - 'shape_motion' = every feature except `age` and `born_in_view`; 'age_only' = `age` (+ `born_in_view` when that feature is on); 'model' = full.
   - Labels in tables and figure: "full", "age only", "shape and motion only", "shuffled labels", "majority". The paired margin CIs (866) gain "full − shape and motion" and "shape and motion − age only".
3. **Single-movie datasets**
   - `timelapse_division_prediction_group` ('movie' | 'lineage_tree' | 'auto'; default 'movie').
   - 'auto' = movie when there are ≥ 2 movies, else lineage tree (root of the parent chain), with a warning in the log, `summary.json` (`grouping`) and the figure subtitle that lineage-tree folds are weaker than movie folds.
   - The bootstrap (867) resamples the same unit as the folds; below `min_movies` the existing row-resampling note stays.
   - With 'movie' and one movie, keep today's refusal, but name the 'auto' / 'lineage_tree' setting in the error.
4. **Reference features and learner**
   - `timelapse_division_prediction_features` ('spacr' | 'reference'; default 'spacr'). 'reference' = value, `d1`, `d{lag}`, `roll{lag}` per measure, plus `age` and `born_in_view`.
   - `timelapse_division_prediction_lag` ('auto' | int; default 'auto' = min(alignment window W, 5), W from `division_analysis` = 0.3 × median cycle clipped to 4–30).
   - `timelapse_division_prediction_born_flag` (bool; default False with 'spacr', True with 'reference').
   - `timelapse_division_prediction_history` (3): today's change window.
   - Learner: `timelapse_division_prediction_max_depth` (3; reference 4), `_learning_rate` (0.1; reference 0.05), `_max_bin` (256 = xgboost default; reference 64), `_n_estimators` (200), `_subsample` (0.8), `_colsample` (0.8), `_n_jobs` (2; 1 for bit-identity with the reference), `_seed` (0 today; reference 20261104). The sklearn fallback takes the same depth, rate and seed and records that it was used.
5. **Label definitions**
   - `timelapse_division_prediction_events` ('all' | 'two_daughters'; default 'all'): fission events counted; budding unchanged.
   - `timelapse_division_prediction_negative_rule` ('lineage_end' | 'last_seen'; default 'lineage_end'): a negative frame needs the lineage end, or the last frame the track is seen, to lie at least `horizon − 1` frames ahead.
   - `timelapse_division_prediction_frames` also accepts 'auto' = max(2, round(0.15 × median cycle)), as the reference.
6. **Reference preset:** one saved settings file (`division_prediction_reference.json`) with unit 'frame', max_rows 120000, arms incl. shape_motion, group 'auto', features 'reference', learner depth 4 / lr 0.05 / max_bin 64 / seed 20261104 / n_jobs 1, events 'two_daughters', negative_rule 'last_seen', and the per-dataset horizons of the reference `settings.json`.
7. **Outputs:** existing tables (`metrics`, `folds`, `importances`, `predictions`) plus `pr_curves.csv` (120 points per arm, as the reference) and the new summary fields.

## Controls (kept, and reported in every run)
- Prevalence (majority / random classifier AUPRC), labels shuffled within the grouping unit, age only, and now shape and motion only.
- The flag from the minimum-positives guard, and the grouping and interval method actually used.

## Validation data and acceptance
- **Data:** 13 public time-lapse datasets with lineage (E. coli mother machine, C. glutamicum, S. cerevisiae bck2, LYN, Cell-ACDC and BABY, S. pombe, DynamicNuclearNet, DeepSea, and the Cell Tracking Challenge sets PhC-C2DL-PSC, Fluo-N2DL-HeLa, BF-C2DL-HSC, BF-C2DL-MuSC), train + valid splits only, from the per-frame measurement cache, with the reference horizons. Reference: `comparison_script_vs_spacr_module.csv` (13 datasets × prevalence, shuffled labels, age only, shape and motion only, full).
- **Acceptance:** identical output, or within the stated tolerance, to the reference outputs:
  1. **No regression:** with default settings, a rerun reproduces the `spacr_default_*` columns (value, rows, positives, CI) identically.
  2. **Every-frame diagnostic:** unit 'frame', max_rows 120000, row_seed 0, other settings default: identical to the `all_frames_diagnostic_*` columns.
  3. **Reference preset vs the reference implementation:** with xgboost pinned to the reference version and `_n_jobs` 1:
     - the 10 datasets under the row cap (all but C. glutamicum, bck2, DynamicNuclearNet): `n_rows` and `n_positive` identical to `script_n_rows` / `script_n_positive`; prevalence, age only, shape and motion only and full AUPRC identical to `script_value` (|Δ| ≤ 1e-9);
     - shuffled-labels arm (all 13) and every arm of the 3 capped datasets: the reference drew those seeds from a per-dataset stream shared with earlier steps, which a settings file cannot replay. Tolerance |Δ| ≤ 0.02 AUPRC, or ≤ 2 × the spread of the reference rerun under 5 row/shuffle seeds, whichever is larger (record that spread with the acceptance run);
     - S. pombe runs with group 'auto' → lineage tree, and the warning appears.
  4. **Guard:** in unit 'track', exactly Cell-ACDC, BABY, MuSC, S. pombe, HSC and LYN are flagged (< 30 positive tracks); in unit 'frame' none is skipped (all ≥ 20 positive rows).
  5. **Synthetic test** (unit tests, CPU): planted pre-division shape change → shape_motion beats age_only and shuffled; planted timer → age_only ≈ full > shape_motion; one movie with 'auto' → lineage-tree folds that never split a tree.

## Literature
- XGBoost: Chen & Guestrin 2016 (KDD).
- AUPRC under class imbalance: Saito & Rehmsmeier 2015 (PLoS One).
- Grouped cross-validation for clustered data: Roberts et al. 2017 (Ecography).
- Cluster bootstrap: Field & Welsh 2007 (JRSS B).
- Cell-size control and the adder: Taheri-Araghi et al. 2015 (Curr Biol).
- Datasets: Cell Tracking Challenge (Maška et al. 2023, Nat Methods); DeepSea (Zargari et al. 2024, Cell Rep Methods); Cell-ACDC (Padovani et al. 2022, BMC Biol); BABY (Pietsch et al. 2023, Genome Biol); DynamicNuclearNet (Schwartz et al. 2023, bioRxiv).

## Effort
- **Effort:** S–M (settings and arms S; reference features and label options M; tests and catalogue records for any new public callable, e.g. `division_prediction_pr_curves`, and the arrival contracts of changed signatures).
