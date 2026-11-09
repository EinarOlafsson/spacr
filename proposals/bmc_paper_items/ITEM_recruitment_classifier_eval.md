# Item spec: recruitment classifier with leave-one-well-out evaluation and controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related: 832 (multiclass ML scores, done), 816 (recruitment report, done), 830 (rim recruitment with a null threshold, open).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P19).

## Why
A GBP1-recruitment classifier on vacuoles (the readout of the published HRMAn work, Fisch et al. 2019) needs an evaluation with leave-one-well-out splits and controls. The script that will produce it is custom:
- a feature route with sklearn HGB, leave-one-well-out, majority and shuffled controls;
- a CNN route through spaCR internals.

Manual labels are still to be made (`R/hrman/ANNOTATE_ME.md`). Port the evaluation **before** labels arrive, so that the first result comes from a settings run.

## Reference implementation
`/mnt/firecuda2/hrman_fig6/scripts/train_eval_recruitment.py`:
- `load(src)`: reads the measurement db + `png_list.annotate`;
- `metrics(y, p, name)`;
- `lowo(df, shuffle_seed=None)`: leave-one-well-out;
- `feature_route(df, out)`: HistGradientBoosting on the vacuole features, plus majority and shuffled-label controls;
- `cnn_route(src, out)`: `spacr.io.generate_training_dataset` + `spacr.deep_spacr.train_test_model`.

The annotation set comes from `export_annotation_set.py`, which uses a raw sqlite `ALTER TABLE png_list`; replace that with spaCR's Annotate DB writer.

## What to build
In the Classify/ML module (`spacr.ml.generate_ml_scores` and the train module):
- `ml_eval_split` ('random' | 'leave_one_well_out' | 'leave_one_plate_out');
- `ml_eval_controls` ('majority, shuffled');
- `ml_eval_shuffle_seed` (0);
- `ml_model_type` ('hist_gbm' added if missing);
- `ml_label_column` ('annotate');
- `ml_eval_report` (bool; writes `results/ml_eval/metrics.csv` with accuracy, balanced accuracy, F1 and AUROC per fold, plus the controls).

For the CNN route, the train module gets the same split and controls (`train_split='leave_one_well_out'`).

Also add an Annotate export of a stratified sample of N objects to `png_list.annotate` (`annotate_sample_n`, `annotate_sample_seed`, `annotate_sample_strata`). This replaces `export_annotation_set.py`.

## Validation data and acceptance
- **Before labels:** run the reference script and the settings run on a fixed synthetic label column (e.g. labels from the rim score threshold, seed 0) on GBP plate77. Acceptance: metrics and control rows identical to the reference script's output on the same labels.
- **After labels:** the settings run is the result. The reference script is run once on the same labels as a cross-check, with identical metrics required.
- Save the GBP plate77 settings, which were never saved.

## Effort
- **Effort:** S–M.
