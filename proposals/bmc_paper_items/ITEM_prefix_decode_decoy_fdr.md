# ITEM DRAFT (no number yet): decoy FDR and decoy-tuned threshold for the spacer-prefix decode

- **Number:** 942 (reserved 2026-11-14).

- **Code basis:** origin/nightly ad161c4242 (`spacr/ops_prefix.py`, `spacr/ops_sbs.py`, `spacr/ops_decode.py`).
- **Found by:** PERISCOPE P1 benchmark (S2 a, c), 2026-11-13.

## Gap
`run_prefix_decode` (item 920) has only a shuffled-whitelist control: each read is also matched against
`shuffled_library(library, seed)` and the site summary reports `control_frac_matched`. It does not:
1. match reads against a **decoy library** jointly with the real prefixes (a read equally close to a guide and
   a decoy maps to neither), as the barcode decode does through `ops_decoy_library_k` and `LibraryMatcher`;
2. call **decoy cells** under the cell rule, and report the decoy FDR per site / well / plate with
   `spacr.ops_decode.fdr` and `fdr_report` (the S2 (a) and (f) measure);
3. tune `ops_read_threshold` per well **on decoys only** (the maintainer's 2026-11-13 rule for stage 2). The
   frozen 920 threshold (80) was chosen on concordance with PERISCOPE, i.e. on the comparator.

The P1 benchmark therefore had to compute the FDR in a scoring script that re-matched each method's reads with
`match_prefix_reads` and `decoy_library` (k = 4, seed 1). That is scoring, not a spaCR result.

## Spec
- New settings: `ops_prefix_decoy_k` (int, default 0 = off; 4 recommended), `ops_prefix_decoy_seed` (1),
  `ops_prefix_fdr_scale` (1.0), `ops_prefix_threshold_tuning` (`none` | `per_well_decoy`),
  `ops_prefix_fdr_target_pct` (0.1).
- Build decoys with `spacr.ops_sbs.decoy_library(library.keys, k, seed)` (12-mers ≥ 2 edits from every prefix,
  collision prefixes included in the exclusion set). Add them to the matcher as callable keys of gene `DECOY`.
- Cell calls: a cell whose top barcode is a decoy is a decoy call; never written as a genotype.
- Outputs: `site_summary.csv` gains `cells_real`, `cells_decoy`, `fdr_pct`; new `fdr_well.json`
  (`fdr_report` layout) per well.
- `per_well_decoy`: grid over `ops_read_threshold` (the 920 grid), pick per well the threshold with most real
  cells subject to FDR ≤ target (ties → higher threshold); record the grid in `threshold_tuning_<well>.csv`.
  No reference calls may be read by the tuner.
- Off by default: with `ops_prefix_decoy_k = 0` outputs are byte-identical to today.

## Proof check (pre-set)
- Identity: decoy calls and FDR equal the P1 scorer's common-rule numbers on the 10 iterate sites
  (`periscope_benchmark/out/s2_10sites/`) to the cell.
- Calibration: barcode-sequence shuffle call rate ≤ 1.5 × decoy FDR (S2 c).
- Off-switch byte identity on 2 sites.

## Maintainer requirement (2026-11-14): auto-tune precision AND recall
- The auto mode must tune detection and calling together, not FDR alone: spot-detection thresholds (e.g. read/peak thresholds) and read-quality/calling thresholds are searched jointly.
- Objective options: (a) maximise recall (cells genotyped) subject to decoy FDR ≤ max (default max 0.1 %; if unreachable, report the lowest reachable FDR and use it); (b) maximise an F-score of precision (1 − FDR) and recall. Default (a).
- Output a precision–recall (genotyped % vs decoy FDR) curve per well and the chosen point; decoy-only tuning, tuning/report split as in 939.
- Same requirement applies to the Funk decode's auto mode (939, which currently keeps spot thresholds manual): extend it with joint spot-detection tuning.


## Scope decision (master, 2026-11-14)
The joint precision-and-recall tuning (spot detection and calling thresholds together) also replaces 939's auto mode, which tunes only the calling threshold. It is built here, in 942, as one shared tuner used by both the prefix decode and the OPS decode (939's ops_decode_threshold_mode='auto'); no separate item.
