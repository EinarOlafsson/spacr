# Item 954: the complexes null of the calibrated cluster benchmark (929) uses 1,000 permutations

- **Number:** 954 (assigned by master 2026-11-15). **Maintainer decision:** 2026-11-15, "Raise to 1,000".
- **Found by:** card C+ (`BMCspacr/reanalysis/funk_published_rows/REPORT.md` and `tables/partcp_rows_and_controls.csv`).
  - The 929 complexes null is a 200-permutation Monte Carlo estimate, which leaves a bias of about 0.2 points.
  - Under the maintainer's control rule of 2026-11-15 (the mean of 500 permuted-label runs must have a CI including 0), 3 complexes control rows fail: Funk-published mitotic −0.21 [−0.32, −0.10], and the plate B count-matched row −0.17 [−0.32, −0.03].
- **Related:** 929 (Ready); the permuted-control rule (maintainer, 2026-11-15).

## Spec
1. A new setting for the number of permutations in the 929 complexes null (size-matched), default 1,000.
2. **Schema:** a bump plus a DEFAULT_CHANGES row; files stamped before it rerun with 200.
3. Seeds are as now, so the first 200 permutations are the old ones (a reproducible prefix).

## Proof check (pre-set)
- **929 proof recheck:** spaCR reproduces cal_lib and the Design D / Design F gates with the new null. The calls are unchanged; report the numeric differences.
- **Bias:** on the card C+ inputs, the 500-permutation control means for all complexes rows have CIs including 0.
- **Back-compatibility:** a pre-954 file reruns byte-identically (200 permutations).
- **Time:** report the added CPU time per benchmark call.
