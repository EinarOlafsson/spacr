# Pipeline readiness register

Every pipeline step is either **Ready** or **Under development**.

- **Ready**: proven to work against reference data. It may be used to generate results.
- **Under development**: it must not be used to generate results. Resources go into iterating on it. The only runs allowed are its proof checks: identity or reproduction tests against reference data, with pre-set criteria and controls.

A step moves to Ready only when its proof check passes. The evidence is then recorded here.

An analysis may start only when every step it uses is Ready. If a proof fails, the step stays under development.

Every session (Home, work machine, paper) reads this file before running any analysis.

Maintainer rule, 2026-11-12.

## Register

| Step | Item | Status | Evidence | Proof check that makes it Ready | Depends on |
|---|---|---|---|---|---|
| OPS barcode decode: recipe_simple classes and tuned threshold | 920 | **Ready** (2026-11-12) | recipe_simple reproduces PERISCOPE's classes on 196,153 cells with 0 mismatches. All four pre-set criteria pass on 79 held-out sites (`/nas_mnt/data/archive/periscope/item920_threshold/`) | — | — |
| Large-site segmentation: tiling, stitched flows, nucleus-diameter estimate | 921 | Under development | Tiling fixed 2026-11-12. On CPU, 141/141 objects away from seams are identical. Not yet proven on GPU | 921 final run on the work-machine GPU: the tiling control passes and the identity rerun is byte-identical | — |
| spaCR vs PERISCOPE segmentation comparison | 922 | Under development | — | Runs after 921 is Ready | 921 |
| Effector translocation (pzfx reader, own Measure database, infection report) | 910 | Under development | 3 pilot fields are identical to the reference scripts (≤ 1e-9) | A: end-to-end on 3 fields, identical. Then B: full plate, tables identical to the reference on the same masks | — |
| Interphase-fitted TVN for cell-cycle profiles | 890 | Under development | Synthetic tests. Matches a port of build_cells.py to ≈ 3e-5 | Card A: plate A phase tables identical | — |
| Neighbour effects | 891 | Under development | Synthetic tests | Card B: plate A neighbour tables and tests identical | 890 |
| Published comparator rows in the Leiden sweep | 916 | Under development | Synthetic tests | Card C: comparator rows identical to the published rows | — |
| Cell-state stratified profiles and mitotic index | 914 | Under development | Synthetic tests | Card E: per-state tables and mitotic-index tables identical | 856 |
| Retrieval power curve | 917 | Under development | Synthetic tests | Card D: mAP, top-10 rate and test p values identical to the reference | 907, 887 (X04 reproduction) |
| Recruitment CNN held-out route | 899 | Under development | Stand-in model tests | Card C: reproducible with synthetic labels. A real result needs real labels | GBP plate 77 settings |
| Measure feature families | measure 1112 | Under development | Default output byte-identical to legacy. Well-A1 evaluation in progress | Paper session's well-A1 result | — |
| Vacuole counts together with per-cell export | 898 | Under development (bug) | Export fails when both are on (`n_vacuoles` column) | Fix plus a test with both on | — |

## Proof order

1. 921
2. 910 A, then 910 B
3. 890
4. 891
5. 916
6. 914
7. 917 (after 907 and 887)
8. 899

Each proof uses the design card signed off for it.
