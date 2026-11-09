# Item spec: calibrated per-cell genotype confidence

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P6).

## Why
A per-cell calibrated P(correct) for each genotype call, its reliability curve (predicted vs decoy-observed) and the expected calibration error (ECE) are computed by custom scripts only.

## Reference implementation
- `R/funk_plateA_full/strategies/scripts/decode_diag.py`: per-call evidence table.
- `R/funk_plateA_full/strategies/scripts/s1a_confidence.py`: HistGradientBoosting on decoy vs real, calibration, reliability bins, ECE, shuffled-label control.

## What to build
In the decode stage of `spacr.ops_engine` (alpha), after the cell rule:
- a column `genotype_confidence` (calibrated P(correct)) in the genotype table;
- a report `results/ops_decode_qc/confidence_reliability.csv` (bin, predicted, observed, n) + `confidence_metrics.json` (AUC, ECE, shuffled AUC).

Settings:
- `ops_genotype_confidence` (False);
- `ops_confidence_model` ('hist_gbm');
- `ops_confidence_calibration` ('isotonic' or as in the reference);
- `ops_confidence_bins` (as in the reference);
- `ops_confidence_seed` (0);
- `ops_confidence_min` (None: a filter threshold that is off by default).

## Validation data and acceptance
- **Data:** the plate A final decode; reference outputs in `R/funk_plateA_full/strategies/` (tables written by `s1a_confidence.py`).
- **Acceptance** (identical, or within the stated tolerance, to those reference outputs):
  - the same set of calls, row for row, as the reference tables;
  - `genotype_confidence` identical to the reference to 1e-9 (CPU, fixed seed);
  - reliability table and AUC/ECE identical.

## Effort
- **Effort:** M.
