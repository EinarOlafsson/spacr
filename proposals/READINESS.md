# Pipeline readiness register

Every pipeline step is either **Ready** or **Under development**.

- **Ready**: proven to work against reference data. It may be used to generate results.
- **Under development**: it must not be used to generate results. Resources go into iterating on it. The only runs allowed are its proof checks: identity or reproduction tests against reference data, with pre-set criteria and controls.

A step moves to Ready only when its proof check passes. The evidence is then recorded here.

An analysis may start only when every step it uses is Ready. If a proof fails, the step stays under development.

Every session (Home, work machine, paper) reads this file before running any analysis.

Maintainer rule, 2026-11-12.

## Feature workflow (maintainer, 2026-11-12, all lanes)

Every feature goes through these steps:

1. Build it in scripts.
2. Iterate fast on a small but robust dataset.
3. Test it in scripts.
4. Make a development-deck slide of that test.
5. Implement it in spaCR, with its new settings.
6. Test it in spaCR.
7. Make a development-deck slide of that test.
8. Do a production run with spaCR, saving all settings and files as source data.

- At every step, run Brieflow's code on the same data. "Does our output match or beat Brieflow?" is always one of the tests, but never the only one.
- Once the Brieflow level is reached, keep iterating. Move to step 5 only after three principally different new approaches have failed to improve further.
- Paper data come only from step 8, and only when the step and everything upstream of it are Ready.
- No TB-scale production runs with scripts.
- Work through the OPS stages in order, 1 to 5. A plate gets a production run only after every stage it uses is Ready.

## OPS: five stages, all under development

Maintainer, 2026-11-12: every OPS stage is optimised for **quality first, then analysis time**. A stage is Ready only after its quality proof passes. Speed work comes after that, and must not change the stage's output beyond the stated tolerance.

| Stage | Status | Components (item / owner) | Quality measure | Gap before a proof can run |
|---|---|---|---|---|
| 1. Stitching and aligning | Under development | Funk first pass `run_stage.py stitch` (WS, spaCR df9d9d9d8); 921 large-site segmentation tiling (Home) | Registration error between cycles and phenotype (px), seam artefacts, fraction of cells lost at tile and site borders, against the reference (Funk/Brieflow and PERISCOPE alignment) | Proof card S1 run 2026-11-12: (b), (d) and the duplicates half of (c) pass. (e) rerun identity passes with the deterministic solve (5797246603, on nightly). (a) plate B (94.7 % within 2 px, need 95 %) and the (c) seams fail. (c) seams criterion amended by the maintainer 2026-11-13: at most 5 % of seams above the tile-centre p95 (upper CI included) and median seam step no larger than the tile-centre median (1-point margin); flat-field per tile still fails (164/1,248 and 157/1,248 seams above p95 on plates A and B), and feathered blending, BaSiC, Voronoi one-tile-per-pixel and per-tile gain matching are queued; iteration card S1-it1 (arms i–iv, flat-field seams, (v) Sinkhorn soft point matching, (vi) mutual-information registration) signed off and running |
| 2. Genotyping | Under development | 920 recipe_simple decode (proven against PERISCOPE, see below); Funk rule (a) + S2 final decode (paper session, WS); 919 | Cells genotyped, concordance with the reference genotype, shuffled-barcode rate, guide representation against the library (R² vs NGS) | Proof card S2 run 2026-11-12: (a), (d), (e) pass; (c) passes as the restated calibration test (0.79× the decoy prediction). (b) plate C concordance 96.4 % (need 97 %) and (f) plate D FDR 0.121 % (need ≤ 0.1 %) fail; iteration card S2-it1 signed off and running |
| 3. Phenotyping from tabular features | Under development | Measure feature families (paper session); hybrid profile with 10 % trimmed mean (paper session); 856 per-cell phase classifier; 890; 914; 891 | Reproduction of the reference profile matrices; retrieval of known complexes and pairs (CORUM, STRING) on held-out genes, with shuffled-label and non-targeting controls | Cards A, B and E (890, 891, 914); paper-session proofs for the measure families, hybrid profile and 856 |
| 4. Phenotyping from single-cell images | Under development | OPS image-model embedding V1 `spacr_ops_transformer` (paper session, E2); 896 crop embeddings to well profiles | Retrieval of complexes and pairs from embeddings against the tabular profile and against controls, on held-out genes; E2 QC v2 checks | E2 QC v2 (due Fri); 896 has no proof card yet |
| 5. Genotype–phenotype analysis | Under development | Guide-bootstrap consensus clustering, calibrated cluster benchmark (paper session); 916 Leiden sweep with published comparator rows; 917 retrieval power curve | Complexes and pairs recovered against Brieflow on the same data, false-positive rate under the jitter gate, calibrated p and CI | Consensus clustering (928) and the calibrated benchmark (929) are Ready (2026-11-13). Still needed: cards C and D (916, 917) |

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
| Measure feature families | measure 1112 | Under development | On nightly 0bed2ff593 (2026-11-12). Default output byte-identical to legacy. Well-A1 evaluation in progress | Paper session's well-A1 result | — |
| Vacuole counts together with per-cell export | 898 | **Ready** (2026-11-12) | Fixed in 2fd62a155d. The export now writes `Metadata_n_vacuoles` and `Metadata_n_pathogens`; a regression test runs both on a synthetic plate | — | — |
| Funk first pass: stitch, phenotype, GPU objects, decode (WS) | OPS stage 1–2 | Under development | Produced the Home plates B–D. No recorded proof. LaC024E and LaC024F outputs are labelled not paper data | Stage 1 and stage 2 proof cards | — |
| Funk rule (a) + S2 final decode | OPS stage 2 | Under development | Used for Home plates A–D. No recorded proof | Proof card against the reference genotype calls | — |
| Guide-bootstrap consensus clustering | 928, OPS stage 5 | **Ready** (2026-11-13) | On nightly da2f7ce0f1. Part 2 passed: spaCR reproduces the script partitions in 80/80 replicates (four arms × 20, cut and final partition identical; `BMCspacr/reanalysis/spacr_stage5_proof/stage5_proof_result.json`). Held-out genes on plate A: +10.7 complexes and +14.1 pairs vs Brieflow on the same data. Plate B: +14.3 and +15.3. Proof part 1 passed 2026-11-12: the consensus-stage jitter gate gave 0/20 false positives on complexes, pairs and capped pairs (search side, R = 20). Part 2 is waiting for the spaCR implementation | Jitter gate passes (≤ 5 % false positives), and spaCR reproduces the Design F partitions on plate A (identical labels, fixed seeds) | — |
| Hybrid profile, 10 % trimmed mean | OPS stage 3 (paper session) | Under development | STRING pairs improve on 3 datasets | spaCR reproduces the Design D gene-profile matrix on plate A (max abs difference ≤ 1e-6) | — |
| Calibrated cluster benchmark test | 929, OPS stage 5 | **Ready** (2026-11-13) | On nightly da2f7ce0f1. spaCR reproduces cal_lib and the Design D and Design F gates to ≤ 1.8e-15 and the Design D ledger to ≤ 4.4e-15 with all 34 calls equal; gate false-positive share equal to the reference (same file) | spaCR reproduces cal_lib on the plate A test set (identical p and CI to 1e-9), and the gate passes | — |
| OPS image-model embedding V1 (`spacr_ops_transformer`) | OPS stage 4 (paper session) | Under development | Learning gate passed: +0.012 [0.007, 0.017] | E2 QC v2 checks pass | — |
| Per-cell cell-cycle phase classifier | 856, OPS stage 3 (paper session) | Under development | F1 0.949 against hand labels; CDK1 and KIF11 4N controls | spaCR reproduces the phase calls on plate A (identical labels) | — |

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
