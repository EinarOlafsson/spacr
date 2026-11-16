# Item 949: decoy FDR calibration for short barcode prefixes

- **Number:** 949 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features). Each new setting gets 9-locale strings and an alpha deck label in `tools/refresh_alpha_deck.py`.
- **Maintainer decision:** 2026-11-15, "Report both": every PERISCOPE comparison reports the matched decoy FDR and the barcode-shuffle call rate.
- **Found by:** the item 942 proof on PERISCOPE CP186A, 85 sites (`periscope_benchmark/out/autotune_942/fast_search/prefix_85/`).
  - With 12-nt prefixes and 80,862 guides, the barcode-shuffle call rate stays 2–4× above 1.5 × decoy FDR after tuning: 0.38 % against a limit of 0.134 % (tuned on 85 sites), and 0.26 % against 0.063 % (tuned on 10 sites).
  - This holds for every method, PERISCOPE included.
  - Cause: the decoys sit ≥ 2 edits from every prefix, so they miss errors that land 1 edit from a real guide.
- **Workflow:** the 8-step feature workflow. Steps 1–4 (script, small iteration, test, dev slide) come before any spaCR code.

## Spec
1. **Near-miss decoy class** (optional; `ops_prefix_nearmiss_decoy_k`, default 0 = off). Decoys are 1 edit from a real prefix and never callable. A read nearest to a near-miss decoy is a decoy call, and is reported separately from the far decoys.
2. **FDR-scale calibration** (optional; `ops_prefix_fdr_calibration`: `none` | `shuffle`). The FDR scale (`ops_prefix_fdr_scale`, and the barcode decode's FDR scale) is set from the barcode-shuffle control, so that the shuffle call rate ≈ 1.5 × the calibrated FDR. This is decoy- and shuffle-only: no reference calls are read.
3. **Reporting:** `site_summary.csv` and the per-well FDR report gain the shuffle call rate next to the decoy FDR (far and near-miss), with bootstrap CIs.
4. **Defaults:** off, so outputs stay byte-identical unless the settings are turned on. Schema bump only if a default changes.

## Proof check (pre-set)
- **Calibration:** on CP186A held-out sites, after tuning with the calibrated scale, the shuffle call rate ≤ 1.5 × calibrated FDR (S2 c).
- **Funk:** plate A A1 stays within its current S2 (c) pass.
- **Off-switch:** byte-identical on 2 sites.
- **The calibrated FDR is reported for spaCR and PERISCOPE** with the same scorer.

## Implementation (2026-11-16, branch claude-949-decoy-calibration)
- `spacr.ops_prefix.prefix_nearmiss_decoys` (gene `DECOY_NEARMISS`), `spacr.ops_decode.calibrate_fdr_scale`, `spacr.ops_decode.shuffle_call_rate`; settings `ops_prefix_nearmiss_decoy_k`, `ops_prefix_fdr_calibration`, `ops_decode_fdr_calibration` (alpha, off by default, no schema bump).
- Proof outputs: BMCspacr/reanalysis/periscope_benchmark/out/fdr_calibration_949/ (off switch byte-identical on sites 27 and 98; calibrated arms pass on all CP186A held-out splits).
- Funk plate A A1 does not pass S2 (c) today under this shuffle (0.90 % vs limit 0.09 %, report half); with the calibration it passes at 70.5 % genotyped instead of 76.4 %. The pre-set "stays within its current S2 (c) pass" therefore could not be met as written.
