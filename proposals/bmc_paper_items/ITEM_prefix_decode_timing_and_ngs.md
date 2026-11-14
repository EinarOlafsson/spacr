# ITEM DRAFT (no number yet): per-site resource log and NGS representation in the prefix decode

- **Number:** 944 (reserved 2026-11-14).

- **Code basis:** origin/nightly ad161c4242 (`spacr/ops_prefix.py`; `spacr/resource_log.py`).
- **Found by:** PERISCOPE P1 benchmark (card §6 timing; §5 S2 d), 2026-11-13.

## Gap
1. `run_prefix_decode` reports one `seconds` for the whole run. The card needs CPU s, wall s and peak RAM per
   site and per step (read/load, spot detection, base calling, matching, cell calls). Item 921's
   `run_large_site_segmentation` already logs this with `ResourceSampler`; the decode does not. The benchmark
   ran one job per site under `/usr/bin/time -v` instead.
2. There is no setting for a library NGS count table, so S2 (d) (R² of log cells per guide vs
   `A549_NGS_Counts.csv`) is computed in the scorer. Card §8 already lists this as "new, small".

## Spec
- Wrap each site in `ResourceSampler(level="summary")`; `site_summary.csv` gains `wall_s`, `cpu_s`,
  `peak_rss_gb` and per-step `*_s` columns; `resource_log/site_<S>.jsonl`.
- Settings `ops_ngs_counts` (CSV path), `ops_ngs_guide_col` (`Barcode`), `ops_ngs_count_col` (`Count`).
  Blank rows are dropped (the published file ends with 4,656 blank rows). Output `ngs_representation.json`:
  R² of log10(cells + 1) vs log10(NGS + 1) over library guides, guides with 0 cells, and a shuffled-label
  control row.

## Proof check (pre-set)
- R² equals the P1 scorer's value on the 10 sites to 1e-12; off by default, byte-identical outputs otherwise.
