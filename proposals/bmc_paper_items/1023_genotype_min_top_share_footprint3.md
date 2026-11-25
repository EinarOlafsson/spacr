# Item 1023: stricter genotype calls by default (minimum top-guide share 0.9, read footprint 3 px)

- **Status:** spec, CONFIRMED on an independent well with a pre-set rule. Proposed for L6 as a default, with a DEFAULT_CHANGES row. The maintainer approves the default before the final A1 rerun (that rerun waits for his OK anyway).
- **Number:** 1023 (claimed 2026-11-25). Settings schema: next free after 965; ask the lander.
- **Builds on:** 964 (`ops_pixel_read_gate`, the B1 read gate), 956 (neighbour-bleed discount), 957 (never-barcode FDR calibration).

## Problem
The decoy false-call rate cannot see wrong calls that are real guides. An independent accuracy reference can: **split-read concordance**, i.e. the same nucleus read in two overlapping sequencing tiles, scored against itself, independent of both spaCR and Brieflow.
- Plate A (LaC024A) well A1 at Home: spaCR's current calls are 90.1 % concordant; Brieflow's 97.3 %.
- spaCR's extra yield came at the cost of accuracy.

## Change
Two settings, both applied in the one decode/vote path (ops_decode `_assemble_calls` / the read vote), not in a second one:
1. `ops_min_top_share` (new, default **0.9**, old behaviour 0.0 / off): a majority call is kept only when the top guide holds at least this share of the nucleus's reads.
2. `ops_footprint` default **3 px** (from 10 px): reads are assigned to a nucleus within 3 px of its mask.
Together with the 964 read gate. DEFAULT_CHANGES row: `{ops_min_top_share: 0.0, ops_footprint: 10}`, so settings files saved earlier replay the old calls.

## Evidence (development; both wells; nothing tuned on the second)
| Well | Arm | Yield (nuclei called) | Split-read concordance | Paired gain vs current, 95 % CI |
|---|---|---|---|---|
| LaC024A A1 (Home; cutoffs chosen here) | current (F10, all) | 76.76 % | 90.1 % | — |
| LaC024A A1 | candidate (F3 + B1 + share ≥ 0.9) | 67.63 % | 97.8 % | +1.63 pts [1.48, 1.78] |
| LaC024A A1 | Brieflow (reference) | 63.6 % | 97.3 % | — |
| LaC024E A1 (WS; independent) | current | 74.28 % | 91.11 % | — |
| LaC024E A1 | candidate | 66.09 % | 97.62 % | +1.46 pts [1.30, 1.62] |

- Pre-set rule (written before the second run): CONFIRMED if the paired concordance-gain CI excludes 0 AND the yield loss is ≤ A1's 9.13 pts. LaC024E: CI [1.30, 1.62], yield loss 8.19 pts → **CONFIRMED**. At ≥ 2 reads per nucleus: +0.26 [0.11, 0.43].
- Tile bootstrap: 2,000 resamples, seed 20261125; report half = 167 tiles, 630,552 nuclei. Decoy FDR barely moves (0.090 → 0.104 %), which is why the decoy rate alone missed this.
- Note: LaC024E was decoded with the LaC024A signed settings; no SBS flat-field is applied in the decode path, so the plate difference does not confound the reads.
- Package: paper repo `reanalysis/genotype_confirm_1125/` (main 474c3606); the full 32-arm grid is `confirm_grid.csv`.

## Also proposed
Report split-read concordance as a standard decode QC metric (per well, with its bootstrap CI) next to the decoy FDR, so that this failure mode is visible in every run.

## Tests
- Unit: share cutoff and footprint on a synthetic read table; the DEFAULT_CHANGES replay gives the old calls exactly.
- Proof: on LaC024A A1, decode-only from the stored caches, the old settings reproduce the production calls; the new defaults reproduce the candidate arm's calls.

## Cost
Decode-only; no extra compute. About 8–9 points less yield, at Brieflow-level accuracy or better, and still above Brieflow's yield.
