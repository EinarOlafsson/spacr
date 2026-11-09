# Item spec: OPS false-call model: what makes a genotype call a decoy (SHAP + controls)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P5).

## Why
The model of which call evidence separates decoy calls from library calls is produced by custom scripts only. They train a gradient-boosted classifier on decoy versus library calls, with:
- a held-out grouped split;
- a sequence-only control model (it measures how much of the separation comes from how the decoys were constructed);
- chance as a floor;
- mean |SHAP| and permutation importance per feature (Lundberg & Lee 2017).

## Reference implementation
`R/decode_yield_2026-11-04/fp_characterisation/code/`:
- `n9_fp_model.py` (feature table per call: exact reads, mismatches, read quality, spot shape, position, sequence features; uses `ops_sbs.called_bases`);
- `n9_train.py` (XGBoost, grouped held-out split, sequence-only control, permutation importance);
- `n9_document.py` (SHAP, `shap_mean_abs.csv`, `model_metrics.csv`).

## What to build
A new QC phase `decode_qc` in `spacr.ops_engine.run_ops` (alpha), run after `decode` when `ops_decode_qc` is True. It writes `<dst_root>/results/ops_decode_qc/`:
- `shap_mean_abs.csv`
- `model_metrics.csv` (main model, sequence-only control, chance)
- `permutation_importance.csv`
- `model.json`

Settings:
- `ops_decode_qc` (False);
- `ops_decode_qc_model` ('xgboost' | 'hist_gbm'; xgboost is an optional dependency, imported lazily with a clear install message);
- `ops_decode_qc_features` (default: the reference list);
- `ops_decode_qc_split` ('field', grouped);
- `ops_decode_qc_seed` (0);
- `ops_decode_qc_max_calls` (sample size as in the reference).

The decode stage must store the per-call evidence the model needs (`ops_store_reads` from 842).

## Validation data and acceptance
- **Input:** the plate A final decode (`R/decode_yield_2026-11-04/plates/plateA_final/`), or the cache the reference used (see `n9_fp_model.py` paths).
- **Acceptance** (identical, or within the stated tolerance, to the reference outputs in `R/decode_yield_2026-11-04/fp_characterisation/`):
  - the feature table is identical (row for row);
  - with the same seed and `n_jobs=1`: AUROC of the main and the sequence-only model equal to the reference `model_metrics.csv` to 1e-6, and `shap_mean_abs.csv` equal to 1e-6;
  - if XGBoost threading is not deterministic, record the self-diff of two reference runs and accept within it.

## Effort
- **Effort:** M.
