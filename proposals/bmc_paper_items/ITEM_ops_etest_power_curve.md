# Item spec: per-gene E-distance and E-test against non-targeting cells, with a cells-per-gene power curve

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_ops_distribution_similarity.md` (energy distance as a gene–gene similarity for retrieval; this item is a per-gene test against NT cells plus a power analysis, not a similarity);
  - `ITEM_ops_cell_state_profiles.md` (supplies the per-cell aligned embedding), `ITEM_ops_profile_pipeline.md`.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
A gene median hides penetrance and spread changes. scPerturb (Peidli et al. 2024, Nat Methods 21:531) scores each perturbation by its E-distance to control cells, tests it with the E-test and subsamples cells per perturbation to show how power falls. No OPS paper reports such a power curve, and spaCR has neither the test nor the subsampling report; together they answer "how many cells per guide does an OPS screen need".

## Reference implementation
1. **E-distance** of gene cells X against an NT reference Y (2,000 NT cells, `default_rng(0)`), in the per-cell aligned (TVN) embedding: E = 2·mean d(x, y) − mean_{x≠x'} d(x, x') − mean_{y≠y'} d(y, y'), d Euclidean (energy distance) and squared Euclidean (scPerturb default).
2. **E-test:** empirical p against 1,000 NT pseudo-genes of the same cell count drawn from NT cells outside the reference (`default_rng(1 + n)`), p = (1 + #null ≥ E)/1,001, BH over genes; for "all cells" the null uses the median cell count. Optional label-permutation mode (scPerturb style, `ops_etest_permutations` 1000).
3. **Controls:** 500 held-out NT pseudo-genes (`default_rng(2 + n)`) and each real NT guide scored the same way (false-positive rate at p < 0.05).
4. **Hit-rate power curve:** n ∈ {10, 25, 50, 100, 200, 500, all}; cells per gene capped at n (`default_rng(100 + seed)`, seeds 0–2); hit = q < 0.05.
5. **Retrieval power curve:** per n and seed keep at most n cells per targeting gene (`default_rng(1000·n + seed)`) and every NT cell; guide and gene medians as the profile aggregate step; the final profile weighting on top; retrieval mAP and top-10 partner rate on CORUM, STRING ≥ 0.7 and BioGRID report halves over one gene list.

## What to build
`spacr.gene_measurement_compare.energy_distance_test(cells, labels, control, n_ref, n_null, seed)` and `subsample_power(cells, labels, ns, seeds, scorer)`; a settings-file phase `etest` after the profile phase (alpha).

## Settings (alpha)
`ops_etest` (bool), `ops_etest_metric` ('euclidean' | 'sqeuclidean' | 'both'), `ops_etest_reference_cells` (2000), `ops_etest_null` (1000), `ops_etest_holdout` (500), `ops_etest_mode` ('empirical_null' | 'permutation'), `ops_power_cells` ([10, 25, 50, 100, 200, 500, null]), `ops_power_seeds` ([0, 1, 2]), `ops_power_retrieval` (bool).

## Outputs
`results/ops_etest/edist_per_gene_n<n>_s<seed>.csv.gz` (gene, cells, E both metrics, p, q; NT guides as `NTguide_<barcode>` rows), `power_hits.csv`, `power_retrieval_summary.csv` + paired tests, one figure.

## Validation data and acceptance
- **Data:** the per-cell aligned embedding of Funk 2022 plate A (classifier-interphase cells) held with the reference outputs.
- **Acceptance:** identical E values (float64 ≤ 1e-9), p, q and hit counts, and identical power tables for the same seeds.
- **Tests (CPU):** a shifted Gaussian gene is called at large n and missed at n = 10; NT pseudo-genes give ≈ 5 % at p < 0.05; E(X, X) ≈ 0.

## Literature
- Peidli et al. 2024 (Nat Methods), scPerturb; Székely & Rizzo 2013 (J Stat Plan Inference), energy statistics.

## Effort
- **Effort:** M.
