# Item 966: spacer-prefix decode default = 952 auto read mode with per-well decoy tuning at a 1 % FDR target (genotyping freeze)

- **Number:** 966, reserved by master on 2026-11-16. **Schema:** 944.
- **Maintainer decision (2026-11-16, before sleeping):** "Leak-free rerun, then freeze", then "Yes, land tonight". The rule: land on nightly only (not main, no release), and only if the pre-set criterion in the proof section holds.
- **Status:** BORDERLINE, held for the maintainer. The leak-free rerun (`periscope_benchmark/out/fdr_sweep_955/compare_952.md`, the `*_952auto_leakfree` rows) passes reads-only and Funk. End to end at spaCR's default shuffle level, PERISCOPE leads by 0.22 points [0.12, 0.32], which is inside the ±0.3-point tie band but not ≥ 0. Build and gate tonight; push only with the maintainer's OK.

## Gap
The spacer-prefix decode reads the LoG-filtered pixels by default (`ops_read_mode='log'`) with fixed thresholds (`ops_prefix_threshold_tuning='none'`).

On PERISCOPE CP186A Well1 (75 held-out sites):
- **With the defaults, spaCR is 3.9–6.8 points behind PERISCOPE** at matched decoy FDR (≤ 1 % / ≤ 0.1 %).
- **With 952 auto and per-well decoy tuning, it ties or leads.** The auto mode picks raw reads on CP186A and log reads on Funk A1.

## Spec
1. **New defaults in `spacr/ops_settings.py` and `settings_spec.py`, spacer-prefix path only:**
   - `ops_read_mode`: `'log'` → `'auto'`
   - `ops_prefix_threshold_tuning`: `'none'` → `'per_well_decoy'`
   - `ops_prefix_decoy_k`: `0` → `4`
   - `ops_prefix_fdr_target_pct`: `0.1` → `1.0`
   - `ops_prefix_tuning_grid`: unchanged (None = the built-in grid).
2. **Scope:** only the spacer-prefix decode (`ops_library_mode='spacer_prefix'`) gets these defaults.
   - The full-barcode OPS decode (`ops_decode_threshold_mode`) keeps its current read mode, because tonight's proof does not cover it.
   - If `ops_read_mode` is shared between the two paths, apply the default per path: resolve `'auto'` only when the prefix path runs, and keep `'log'` for the OPS decode unless the user sets it.
   - Document in the tooltip which path reads which default.
3. **Migration:** `DEFAULT_CHANGES[944] = {'ops_read_mode': 'log', 'ops_prefix_threshold_tuning': 'none', 'ops_prefix_decoy_k': 0, 'ops_prefix_fdr_target_pct': 0.1}`. Settings files older than 944 reproduce today's output byte for byte.
4. **Reporting:** the run report says, per well, which read mode and thresholds auto chose, the tuning-half decoy FDR and the report-half decoy FDR. These already exist in `threshold_tuning_<plate>_<well>.json`; surface one line per well in the run summary.
5. **Docs and strings:**
   - 9-locale tooltips updated with the new defaults;
   - a release-note line in simple terms;
   - arrival contract and catalogue records in the same commit;
   - the readiness register: spacer-prefix genotyping → Ready (frozen).

## Proof check (pre-set)
1. **Off-switch:** with the pre-944 values, the prefix decode output on 4 CP186A sites (3, 45, 64, 86) and 2 Funk A1 tiles is byte-identical to nightly before the change.
2. **Default reproduces the leak-free rerun:** the landed default on the CP186A selection sites, reported on the 75 held-out sites, matches the leak-free proof rows (same chosen mode and thresholds; genotyped % within 0.01 points).
3. **Criterion, met by the leak-free rerun before landing.** spaCR is ≥ PERISCOPE at ≤ 1 % decoy FDR (paired diff CI upper bound ≤ 0, or a tie within ±0.3 points), both reads-only and end to end. spaCR is > Brieflow on Funk A1 at matched FDR and shuffle rate. Values:

| Dataset / comparison | spaCR | Comparator, our run | Comparator, published | Diff [95 % CI] |
|---|---|---|---|---|
| CP186A reads-only, ≤ 1 % decoy FDR (75 held-out sites, 490,923 cells) | 43.70 % (FDR 0.095 %, shuffle 0.498 %) | PERISCOPE reads 43.46 % (FDR 0.064 %, shuffle 0.462 %) | n/a in paper | spaCR +0.23 [0.14, 0.33] ✅ |
| CP186A reads-only, ≤ 0.1 % decoy FDR | 42.64 % | 43.46 % | n/a in paper | spaCR −0.82 [−0.96, −0.68] (reported, not part of the criterion) |
| CP186A end to end, at ≤ spaCR default shuffle (0.531 %); recipe "simple" perfect+great rule for both | 34.54 % | PERISCOPE own calls 34.76 % (score ≥ 0.917) | n/a in paper | spaCR −0.22 [−0.32, −0.12]: tie band, borderline |
| CP186A end to end, ≤ 0.1 % shuffle | 21.06 % | 15.71 % | n/a in paper | spaCR +5.35 [4.34, 6.43] |
| CP186A end to end, PERISCOPE's own default (500-permutation shuffle 32.12 % [32.08, 32.17] vs spaCR 0.531 % [0.444, 0.656]) | 34.54 % | 35.71 % | n/a in paper | spaCR −1.16 [−2.42, +0.14] at a 60× higher PERISCOPE false-call rate |
| Funk A1 reads-only, all targets (25 report tiles, 114,325 cells) | 71.70 % (FDR 0.024 %, shuffle 0.609 %) | Brieflow reads 69.53 % (FDR 0.051 %, shuffle 0.711 %) | n/a | spaCR +2.17 [1.80, 2.57] ✅ |
| Funk A1/A1–A3 end to end, matched shuffle (shuffle_control_brieflow; Funk uses the full-barcode decode, unaffected by this item) | 74.56 / 72.27 / 73.73 % | 63.36 / 60.78 / 62.27 % | 63.54 % (Nat Commun 2026, Source Data Fig. 2b, plate 1 A1, own rule, unmatched) | +11.2 / +11.5 / +11.5 ✅ |

- **Leak-free settings chosen by auto:** CP186A raw read, threshold 40, quality 0, 0 mismatches (tuned on the 10 selection sites); Funk A1 log read, threshold 160, quality 0.05, 0 mismatches. These are the same as the leaky run.
- **Caveat:** PERISCOPE's CallBarcodes tie-break could not be reproduced in its shuffle control (10–14 % of foci are ties; ties go to the first row). Its real calls are untouched.

4. **Controls:**
   - 3-seed barcode shuffle call rate reported next to decoy FDR in every row;
   - identity rerun byte-identical.
5. **Speed:** per-site decode time against the pre-944 default on the same machine. Leaky-rerun reference: 58 s per CP186A site, 7.5 s per Funk tile.
