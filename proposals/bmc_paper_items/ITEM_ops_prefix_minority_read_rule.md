# Item 967: spacer-prefix cell call ignores minority stray barcodes (rule F1)

- **Number:** 967, reserved by master on 2026-11-17.
- **Status:** final spec, 2026-11-17. Maintainer decision: "Land 966 now, F1 next". The setting ships OFF by default (alpha). No default change without a later acceptance run. In the paper, any comparison that uses F1 applies it to every method and says so.
- **Acceptance before a default change:** measure the decoy cost on Funk production (rule (a) + S2 + pixel fallback, A1–A3), run the k grid {2, 3, 4} on more wells, and get a readiness-register entry.
- **Build acceptance (alpha):** with k = 0, output is byte-identical to nightly; with k = 3, the CP186A held-out numbers in the table below are reproduced within 0.01 points.
- **Evidence:** `periscope_benchmark/out/fdr_sweep_955/e2e_gap_diagnosis/`. Proof check only; knobs chosen on the 10 CP186A selection sites, scored on the 75 held-out sites (490,923 cells).

## Gap
- **Cell rule:** the spacer-prefix cell call (recipe "simple"-style class) marks a cell "bad" when more than one exact barcode is present.
- **Size of the loss:** a single stray exact read of a second barcode removes the call. On CP186A, 4,897 cells are lost this way that PERISCOPE calls.
- **Cause:** in 2,585 of the 2,643 cells where the main barcode leads, the competitor is one stray read.
- **Why a quality floor doesn't help:** stray reads are only slightly weaker (median quality 0.22 vs 0.26).

## Spec (proposal)
- **New setting** `ops_prefix_minority_min_reads` (int, default 0 = off, which gives today's behaviour).
- **The rule:** inside a cell, a barcode with fewer than k reads is ignored when another barcode has strictly more reads. Ties still block the call. Decoy-top cells are still excluded.
- The cell report states how many cells were rescued by the rule.

## Proof-check numbers (k = 3, 75 held-out sites)

| Arm | Target shuffle call rate | Genotyped % | Shuffle, 500-perm mean [95 %] | vs PERISCOPE baseline | vs PERISCOPE with the same rule |
|---|---|---|---|---|---|
| spaCR baseline (control) | ≤ 0.531 % | 34.54 | 0.531 [0.444, 0.656] | −0.22 [−0.32, −0.12] | — |
| spaCR + F1 | ≤ 0.531 % | 42.99 | 0.532 [0.444, 0.657] | +8.23 [+6.48, +10.10] | +0.07 [−0.03, +0.17] |
| spaCR + F1 | ≤ 0.1 % | 24.30 | 0.169 [0.133, 0.217] | +8.60 [+6.80, +10.58] | +0.44 [+0.34, +0.56] |

- **Decoy-top FDR:** 0.096 % with F1, against 0.115 % baseline.
- **Same barcode as PERISCOPE** on 99.86 % of shared cells.
- **PERISCOPE published:** n/a in paper.

## Fairness rule for the paper
- F1 is a cell-rule change that helps every method about equally.
- Any comparison that uses it applies it to all methods and says so.
- The spaCR-vs-PERISCOPE gap with the shared rule is reported as is: −0.22 at spaCR's default, attributed to the cell rule. Detection (+0.12) and read calling (+0.08) favour spaCR.

## Still to test before it can become a default
1. Funk A1–A3 through the full-barcode decode (does the same failure exist there?), and Funk A1 via the prefix decode.
2. Downstream: does the rescued cells' gene signal match single-barcode cells? Compare guide-replicate agreement of rescued cells vs normally called cells (step 2b of the acceptance standard).
3. The k grid on more wells; whether k should scale with the cell's total read count.

## Follow-up results (2026-11-17 night; `e2e_gap_diagnosis/followup_funk_rescued/`)
- **Funk A1, 955 sweep (reads-only):** F1 is never picked over maxcount, which already ignores minority barcodes. spaCR leads Brieflow by +2.17 [1.80, 2.57] with or without F1. Against a single-barcode rule (purity100), F1 adds +6.4 points for spaCR and +5.9 for Brieflow.
- **Funk production decode (rule (a) + S2 + pixel fallback), A1, 92,512 nuclei:**
  - 1,358 nuclei (1.47 %) are lost to a single stray second barcode after the fallback. 1,100 of them are genuine 1-vs-1 ties.
  - F1 (k = 2) would add 549 nuclei (+0.59 points). Its decoy cost is not yet measured.
- **Rescued cells are real (CP186A, 41,457 rescued vs 169,588 normal):**
  - NT share 16.97 % vs 17.21 %.
  - Decoy-top FDR 0.017 % vs 0.115 %.
  - Gene signal (correlation with sibling-guide published profiles, permutation-null corrected): +0.0040 [0.0024, 0.0058] vs size-matched normal +0.0035 [0.0022, 0.0049]. Single-cell signal is weak for both groups.
- **Revised scope:** F1 applies to cell rules that require a single barcode: recipe perfect/great, purity100, rule (a)'s 0.6 vote fraction. It adds nothing to maxcount. Expected gain:
  - +8.4 points on the PERISCOPE recipe rule;
  - about +0.6 on Funk production, decoy cost to measure.
