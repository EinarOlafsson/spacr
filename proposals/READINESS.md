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
| 1. Stitching and aligning | Ready (2026-11-14) | Stitch, compose and phenotype↔sequencing alignment in spaCR (930 per-tile affine, 931 feather blend, 932 float32 registration; 921 large-site tiling) | Registration error between cycles and phenotype (px), seam artefacts, fraction of cells lost at tile and site borders, against the reference (Funk/Brieflow and PERISCOPE alignment) | Card S1 (a)–(e) pass on Funk plate A A1 and held-out plate B A1 with spaCR nightly 35ca0c47f0 shipped defaults (ops_align_model tile_affine, ops_compose_blend feather, float32 registration, 4 reader threads), including sequencing nuclei segmented from the feathered mosaic: within 2 px 97.57 / 97.05 % (A / B), median residual 0.646 / 0.689 px, 0 failed fields, duplicates 0.067 / 0.068 %, border keep ≥ 99.44 %, seams +0.32 / −0.11 under the restated (c) scorer (maintainer sign-off), stitch reruns byte-identical. Evidence: `BMCspacr/reanalysis/proof_stage1_stage2/REPORT_stage1_ready_check.md`. Open follow-up: item 947 (phenotype folders with two acquisitions sharing a channel) |
| 2. Genotyping | Ready (2026-11-15) | Funk rule (a) + S2 decode with per-well decoy thresholds, spaCR's default decode since item 946 (873, 939, 946; 932 decode speed); 920 recipe_simple decode (proven against PERISCOPE, see below); 919 | Cells genotyped, concordance with the reference genotype, shuffled-barcode rate, guide representation against the library (R² vs NGS) | The proof card S2 criteria (a)–(f) pass with the signed-off restatements (per-well decoy tuning, plate C cycle-2 data-quality exception), and the speed card is met (932). Item 946 makes the proven decode spaCR's default (ops_decode_preset rule_a_s2, ops_decode_threshold_mode auto, calling-only), schema 940. spaCR reproduces the production script calls on one well per plate (A A1, A A2, B A1, C A1, D A1): identical thresholds and per-cell calls, FDR 0.058–0.088 %. The proof was reduced from 23 wells by maintainer decision 2026-11-15. Evidence: `BMCspacr/reanalysis/proof_stage1_stage2/item946/out/` and `/nas_mnt/data/archive/item946_proof_20261115/` |
| 3. Phenotyping from tabular features | Under development | Measure feature families (paper session); hybrid profile with 10 % trimmed mean (paper session); 856 per-cell phase classifier; 890; 914; 891 | Reproduction of the reference profile matrices; retrieval of known complexes and pairs (CORUM, STRING) on held-out genes, with shuffled-label and non-targeting controls | Cards A, B and E (890, 891, 914); paper-session proofs for the measure families, hybrid profile and 856 |
| 4. Phenotyping from single-cell images | Under development | OPS image-model embedding V1 `spacr_ops_transformer` (paper session, E2); 896 crop embeddings to well profiles | Retrieval of complexes and pairs from embeddings against the tabular profile and against controls, on held-out genes; E2 QC v2 checks | E2 QC v2 (due Fri); 896 has no proof card yet |
| 5. Genotype–phenotype analysis | Under development | Guide-bootstrap consensus clustering, calibrated cluster benchmark (paper session); 916 Leiden sweep with published comparator rows; 917 retrieval power curve | Complexes and pairs recovered against Brieflow on the same data, false-positive rate under the jitter gate, calibrated p and CI | Consensus clustering (928) and the calibrated benchmark (929) are Ready (2026-11-13). Still needed: cards C and D (916, 917) |

## Register

| Step | Item | Status | Evidence | Proof check that makes it Ready | Depends on |
|---|---|---|---|---|---|
| Cropped-site reader (cpg0021 PERISCOPE crops) | 919 | Under development | Used for PERISCOPE P1 on CP186A Well1, all 85 analysed sites (nightly ad161c4242). Decode outputs on the 79 sites outside 920 tuning reproduce the item-920 Ready result exactly (34.14 % vs 34.69 %, concordance 96.18 %); reruns byte-identical (sites 27, 98). Report: BMCspacr/reanalysis/periscope_benchmark/REPORT.md | Proof card of its own: pixel identity vs the published crops on a site sample, and handling of the 04Align shift (CorrDNA in the SBS frame) | — |
| OPS barcode decode: recipe_simple classes and tuned threshold | 920 | **Ready** (2026-11-12) | recipe_simple reproduces PERISCOPE's classes on 196,153 cells with 0 mismatches. All four pre-set criteria pass on 79 held-out sites (`/nas_mnt/data/archive/periscope/item920_threshold/`) | — | — |
| Large-site segmentation: tiling, stitched flows, nucleus-diameter estimate | 921 | **Ready** (2026-11-13) | Final run c on the work-machine GPU (spaCR d6c2184b15): c and c_rerun labels byte-identical (200/200). Tiling control (site 75, 782 objects): at seams 519/519 at IoU ≥ 0.9; away from seams 249/263 identical (run b: 2/450) and the other 14 at IoU 0.995–0.9998, a few boundary pixels each, consistent with GPU float order. Maintainer rule 2026-11-13: Ready if all 14 are at IoU ≥ 0.9. Agreement with PERISCOPE unchanged (cell F1 0.751, nucleus F1 0.852); the cell-count shortfall is the separate B04 lane (`item921/rescore_c/`) | — | — |
| spaCR vs PERISCOPE segmentation comparison | 922 | Under development | — | Runs after 921 is Ready | 921 |
| Effector translocation (pzfx reader, own Measure database, infection report) | 910 | Under development | 3 pilot fields are identical to the reference scripts (≤ 1e-9) | A: end-to-end on 3 fields, identical. Then B: full plate, tables identical to the reference on the same masks | — |
| Interphase-fitted TVN for cell-cycle profiles | 890 | Under development | Synthetic tests. Matches a port of build_cells.py to ≈ 3e-5 | Card A: plate A phase tables identical | — |
| Neighbour effects | 891 | Under development | Synthetic tests | Card B: plate A neighbour tables and tests identical | 890 |
| Published comparator rows in the Leiden sweep | 916, 953 | **Ready** (2026-11-15) | Branch claude-953-954-1115 (item 953: comparator control rows in the Fisher background). C3: score_published on Funk 2022's published interphase clusters (5,072 genes + 250 non-targeting rows, repaired table) equals the published row exactly: 222 clusters, CORUM 27.5 / 24.3 %, KEGG 15.8 / 13.5 %, STRING precision 0.0498, recall 0.1222, tp 3,359 / fp 64,046 / fn 24,137. C2: the reference funk_benchmark.score on the same rows is identical (raw differences 0). C5: a spaCR settings-file run (cluster_sweep, ops_cluster_published_controls True) gives the same row; a file stamped 940 reruns at the old 23.9 %. Controls: 500 shuffled-label permutations at chance (CORUM may2025 mean 0.15 %, STRING precision 0.0021), the single-seed row inside every 95 % range. Report: BMCspacr/reanalysis/item953_954_proof/REPORT.md. Item 959 (2026-11-17, branch claude-959-1116, alpha): spaCR's own sweep rows now count the run's NT rows in the same Fisher background (ops_cluster_sweep_controls_background, default True), so each sweep table has one background size (5,322 on the Funk data, 5,281 on plate A X04); the new rows equal the comparator path and the reference funk_benchmark.score, STRING and complex metrics are unchanged, and False or a file stamped 944 is byte-identical to nightly 084bac13fa. Open: Brieflow Nat Commun 2026 publishes Funk CORUM 23.9 %, not the 24.3 % item 953 targeted (maintainer decision). Report: BMCspacr/reanalysis/item959_proof/REPORT.md | Card C: comparator rows identical to the published rows | — |
| Cell-state stratified profiles and mitotic index | 914 | Under development | Synthetic tests | Card E: per-state tables and mitotic-index tables identical | 856 |
| Retrieval power curve | 917 | Under development | Synthetic tests | Card D: mAP, top-10 rate and test p values identical to the reference | 907, 887 (X04 reproduction) |
| Recruitment CNN held-out route | 899 | Under development | Stand-in model tests | Card C: reproducible with synthetic labels. A real result needs real labels | GBP plate 77 settings |
| Measure feature families | measure 1112 | **Ready** (2026-11-13) | On nightly 0bed2ff593 (2026-11-12). Default output byte-identical to legacy. Well A1: features matched (no detectable difference vs Brieflow on A1). Maintainer decision 2026-11-13: matching Brieflow is enough; no further family iterations. Computer-vision phenotyping continues as the next development of this strategy | — | — |
| Vacuole counts together with per-cell export | 898 | **Ready** (2026-11-12) | Fixed in 2fd62a155d. The export now writes `Metadata_n_vacuoles` and `Metadata_n_pathogens`; a regression test runs both on a synthetic plate | — | — |
| Funk first pass: stitch, phenotype, GPU objects, decode (WS) | OPS stage 1–2 | Under development | Produced the Home plates B–D. No recorded proof. LaC024E and LaC024F outputs are labelled not paper data | Stage 1 and stage 2 proof cards | — |
| Funk rule (a) + S2 final decode | OPS stage 2 | **Ready** (2026-11-15) | Item 946 makes it spaCR's default decode (ops_decode_preset rule_a_s2, ops_decode_threshold_mode auto, calling-only), schema 940. spaCR reproduces the production script calls on one well per plate (A A1, A A2, B A1, C A1, D A1): identical thresholds and per-cell calls, FDR 0.058–0.088 %. The proof was reduced from 23 wells by maintainer decision 2026-11-15; maintainer confirmed stage 2 Ready 2026-11-15. Evidence: `BMCspacr/reanalysis/proof_stage1_stage2/item946/out/` | — | — |
| Guide-bootstrap consensus clustering | 928, OPS stage 5 | **Ready** (2026-11-13) | On nightly da2f7ce0f1. Part 2 passed: spaCR reproduces the script partitions in 80/80 replicates (four arms × 20, cut and final partition identical; `BMCspacr/reanalysis/spacr_stage5_proof/stage5_proof_result.json`). Held-out genes on plate A: +10.7 complexes and +14.1 pairs vs Brieflow on the same data. Plate B: +14.3 and +15.3. Proof part 1 passed 2026-11-12: the consensus-stage jitter gate gave 0/20 false positives on complexes, pairs and capped pairs (search side, R = 20). Part 2 is waiting for the spaCR implementation | Jitter gate passes (≤ 5 % false positives), and spaCR reproduces the Design F partitions on plate A (identical labels, fixed seeds) | — |
| Hybrid profile, 10 % trimmed mean | OPS stage 3 (paper session) | **Ready** (2026-11-14) | On nightly 6dd095efb2 (spaCR 905c823cdf + 2b5f502c39, translations 37518534). spaCR rebuilds the Design D plate A gene-profile matrix exactly (5,299 × 95, max abs difference 0; guide rows 20,439 × 95, 0), with ops_blas_threads 2 (default) applied by spaCR and no environment variables. 928 consensus through spaCR's phase code (ops_consensus_space hybrid): 20/20 identical partitions, ARI 1. Trim option (gene_trim) equal to H2_trim on Design D's TVN cells. Report: reanalysis/spacr_stage3_hybrid_proof/REPORT.md | — | — |
| Calibrated cluster benchmark test | 929, OPS stage 5 | **Ready** (2026-11-13) | On nightly da2f7ce0f1. spaCR reproduces cal_lib and the Design D and Design F gates to ≤ 1.8e-15 and the Design D ledger to ≤ 4.4e-15 with all 34 calls equal; gate false-positive share equal to the reference (same file) Recheck 2026-11-15 with the 1,000-permutation null (item 954, branch claude-953-954-1115): the same calls. Design D gate and Design F consensus gate pass with the same false-positive shares (complexes 0.05 and 0.00, pairs 0.00); Design D ledger 34/34 calls equal (27 no detectable effect, 4 decreases, 3 increases). Numeric differences 1,000 vs 200: complexes differences and CI bounds up to 0.24 points, complexes p up to 0.11; pairs and capped pairs unchanged (they use the exact pair probability). At 200 spaCR still equals the reference to ≤ 4.4e-15. Report: BMCspacr/reanalysis/item953_954_proof/REPORT.md | spaCR reproduces cal_lib on the plate A test set (identical p and CI to 1e-9), and the gate passes | — |
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
