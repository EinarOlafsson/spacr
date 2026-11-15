# Item 948: speed-up of the gene-pairs consensus option (B1 + B2, optional B4 workers)

- **Number:** 948 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features). Each new setting gets 9-locale strings and an alpha deck label in `tools/refresh_alpha_deck.py`.
- **Maintainer decision:** 2026-11-15, "Accept B1+B2, keep going". A cheaper-search card continues separately (`experiment_designs/20261115_pairs_fast_search_DRAFT.md`).
- **Evidence:** `BMCspacr/reanalysis/design_f_strategies/pairs_speed/REPORT.md`, section "IMPLEMENTATION-READY".
  - 2,358 vs 7,326 CPU s per replicate in the same pinned run (3.1× faster).
  - The plate A held-out accuracy gate is met: pairs +30.0 [25.1, 34.7], complexes +10.7 [7.5, 14.1]; the difference from the current option is about 0.
- **Workflow:** the 8-step feature workflow. The script stage is done (card + report). This item covers steps 5–8: spaCR implementation, spaCR tests, dev slide, spaCR production.

## Spec
1. **B1, one resolution search per replicate.**
   - Inner redraw 0 runs the existing bisection: geometric over [1e-3, 1e3], ≤ 22 steps, mid = sqrt(lo·hi); keep the best |count − k| (first-found tie); stop on equality.
   - Redraws 1 … B−1 call Leiden once at that resolution, with seed (100 + r) × 100 + b.
   - The final partition is the average-linkage cut of 1 − co-association to k, as now.
   - The resolution and the per-redraw counts are stored in the metadata.
2. **B2, no discarded final SNF fusion.** After the redraws, build only the rows, mask and profile for the score matrix Aco + 1e-3·cos.
3. **B4, optional `ops_consensus_workers`** (default 1).
   - Redraws 1 … B−1 run in workers.
   - The co-association is summed in redraw order, never in completion order.
   - Seeds stay per redraw. About 2 GB per worker.
4. **Schema:**
   - the option's output changes (gate-equivalent, not bit-identical), so bump `SETTINGS_SCHEMA_VERSION` to the next free number at merge;
   - add a DEFAULT_CHANGES row so older settings files rerun the pre-948 consensus exactly;
   - no separate version setting.

## Proof check (pre-set)
- **B2 and B4 are byte-identical:**
  - md5 of the targeting labels after the count adjustment, and of the per-gene AP or the float32 co-association;
  - workers = 1 vs workers = 3.
- **B1 uses the card's gate as its tolerance:**
  - plate A, R = 20: pairs inside [24.95, 34.78], complexes inside [7.97, 14.63], difference from the current option ≈ 0, and k after the cut;
  - permuted controls at chance.
- **Regression fixture:** the card's 20 B1+B2 label vectors (`res/eval/ps_b1b2/`), md5-identical with igraph 1.0.0 / leidenalg 0.12 / PHATE 2.0.0. Fall back to the gate if the library versions differ.
- **Back-compatibility:** a pre-948 settings file reproduces the stored Design F labels md5-identically.
- **Time:** ≤ 2,400 CPU s per replicate on the card's timing set, same machine.
