# spaCR item specs from the reanalysis work: index in priority order

These specs describe spaCR features to build so that analyses now done by custom scripts run from a spaCR settings file.

**What these specs share:**
- Each spec stands on its own.
- Every port is judged by one rule: a spaCR settings-file run gives identical output, or output within a stated tolerance, to the reference outputs at the NAS or local path named in the spec, on the named validation data.
- Build on **origin/nightly**.
- None of these specs has an item number yet. Number them in this order and check `features/feature_list/` and the claims files first. 869 is reserved.
- The source inventory of reference scripts is `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/SPACR_PROVENANCE.md`; read its section 12 before building anything.
- "Paper-critical" marks specs whose outputs a manuscript in preparation depends on. Build those first.

| Rank | Spec | What it builds | spaCR module | Existing items | Effort | Paper-critical |
|---|---|---|---|---|---|---|
| 1 | [ITEM_ops_phenotype_run_v2.md](ITEM_ops_phenotype_run_v2.md) | The OPS phenotype run from one settings file: `merged/` per field (stacked channels + masks, flat-field corrected, compressed, NAS destination), standard `measurements.db` tables + `png_list`, owned cells, raw sharded crops, second-acquisition registration | ops_engine (`ph_merged`, `ph_tables`, `ph_crops`) | 809–813, 852, 853, 857, 860–862 | L | yes |
| 2 | [ITEM_ops_decode_rule_a_s2.md](ITEM_ops_decode_rule_a_s2.md) | Decode rule (a) + S2 + pixel fallback + decoy FDR + skip-unplaced-site as engine settings | ops_engine / ops_sbs / ops_cycles | 842 | L | yes |
| 3 | [ITEM_ops_cross_field_ownership.md](ITEM_ops_cross_field_ownership.md) | Mask-based ownership of cells cut across overlapping fields (`ph_ownership`) | ops_engine | 852, 813 | M | yes |
| 4 | [ITEM_ops_run_report_funnel_counts.md](ITEM_ops_run_report_funnel_counts.md) | Per-well yield funnel; per-guide/gene counts vs a published table with a shuffled control | ops_engine `report` | — | S | yes |
| 5 | [ITEM_ops_false_call_model.md](ITEM_ops_false_call_model.md) | Decoy-vs-real false-call model with SHAP and a sequence-only control | ops_engine `decode_qc` | — | M | yes |
| 6 | [ITEM_ops_genotype_confidence.md](ITEM_ops_genotype_confidence.md) | Calibrated per-cell genotype confidence + reliability report | ops_engine decode | — | M | yes |
| 7 | [ITEM_ops_genotype_three_way_compare.md](ITEM_ops_genotype_three_way_compare.md) | Per-cell comparison against external call tables with per-category decoy FDR | ops_engine `compare_calls` | 842 | M | yes |
| 8 | *(not included: a run task with no code)* | — | — | — | S | yes |
| 9 | [ITEM_ops_profile_pipeline.md](ITEM_ops_profile_pipeline.md) | One settings-driven profile pipeline (cell tables → profiles → metrics) | ops_engine profile phase | 854, 855 | M | yes |
| 10 | [ITEM_ops_retrieval_map.md](ITEM_ops_retrieval_map.md) | Gene–gene retrieval mAP with multi-label references | sp_stats | — | S–M | yes |
| 11 | [ITEM_ops_reference_sets.md](ITEM_ops_reference_sets.md) | Versioned CORUM/KEGG/STRING/BioGRID sets, SELECT/REPORT halves, time split | ops_reference (new) | — | S–M | yes |
| 12 | [ITEM_ops_specificity_metrics.md](ITEM_ops_specificity_metrics.md) | Fitness-matched retrieval, complex identification, held-out complexes, time split, pair AUPRC | sp_stats | — | M | yes |
| 13 | [ITEM_ops_gene_level_aggregation.md](ITEM_ops_gene_level_aggregation.md) | Gene profile = median over all cells (Brieflow aggregation, credited) | gene_measurement_compare | 858 | S | yes |
| 14 | [ITEM_ops_modules_precomputed.md](ITEM_ops_modules_precomputed.md) | Module atlas (consensus + gates) incl. precomputed embeddings | ops_modules (new) | 855 | L | yes |
| 15 | [ITEM_gene_class_training.md](ITEM_gene_class_training.md) | Gene-class multi-label training: class building, grouped folds, decoy heads, shuffled controls, LR baseline, discovery tables, benchmark arms, spaCR CNN fix | gene_classes (new) + train/classify | NEW-1, 772 | L | yes |
| 16 | [ITEM_ops_cell_cycle_profiles.md](ITEM_ops_cell_cycle_profiles.md) | Cell-cycle-resolved gene profiles | ops_engine aggregate | 856 | M | yes |
| 17 | [ITEM_ops_penetrance_neighbours.md](ITEM_ops_penetrance_neighbours.md) | Penetrance, escapers, phase tests, neighbour table and neighbour effects | ops_engine / gene_measurement_compare | 856 | M–L | yes |
| 18 | [ITEM_cross_screen_similarity.md](ITEM_cross_screen_similarity.md) | OPS ↔ JUMP cross-screen gene similarity (Mantel, top pairs, named pairs) | sp_stats | — | M | yes |
| 19 | [ITEM_screen_staging_export_aggregate.md](ITEM_screen_staging_export_aggregate.md) | One settings file per plate for import + prep + masks + measure; parquet export; DMSO aggregate + heterogeneity; timing | core / measure / new `screen` entry | NEW-4, 814 | M | yes |
| 20 | [ITEM_profiling_moa_evaluation.md](ITEM_profiling_moa_evaluation.md) | NSC/NSCB MoA evaluation with CPA-equivalent classifiers, 1-NN, logistic, controls, seeds, per-MoA table | sp_stats / Measure analysis | 807, 808 | L | yes |
| 21 | [ITEM_profile_retrieval_entry.md](ITEM_profile_retrieval_entry.md) | Public replicate-mAP / compound–gene matching / plate-effect entry | sp_stats / Measure analysis | 863 | M | yes |
| 22 | [ITEM_crop_embedding_well_profiles.md](ITEM_crop_embedding_well_profiles.md) | Raw crop shards from merged fields, embeddings from settings, well embedding profiles, UMAP | measure_crop / embeddings | 809, 772 | M | yes |
| 23 | [ITEM_spacr_architectures.md](ITEM_spacr_architectures.md) | Channel-adaptive CNN/ViT, well set-transformer, MAE/DINO, missing-stain evaluation | utils / deep_spacr | — | L | yes |
| 24 | [ITEM_vacuole_replication_titration.md](ITEM_vacuole_replication_titration.md) | Vacuole grouping by gap, replication classifier from an external table, size calibration, uninfected floor, per-well tables | host_pathogen / analyze_replication | 815, 823, 829 | M | yes |
| 25 | [ITEM_recruitment_classifier_eval.md](ITEM_recruitment_classifier_eval.md) | Leave-one-well-out ML evaluation with controls; Annotate sample export | ml / train | 832, 816 | S–M | yes |
| 26 | [ITEM_plaque_measured_vs_reported.md](ITEM_plaque_measured_vs_reported.md) | Public plaque segmentation entry, per-figure effect table, measured-vs-reported scoring, well merge | plaque_papers / plaque | 843–849 | M | yes |
| 27 | [ITEM_division_onsets.md](ITEM_division_onsets.md) | Onset of pre-division changes | division_analysis | 771 (864–867 cover the rest) | S | yes |
| 28 | [ITEM_ops_guide_shrinkage.md](ITEM_ops_guide_shrinkage.md) | Guide-reliability empirical-Bayes shrinkage | gene_measurement_compare | — | S | no |
| 29 | [ITEM_ops_generic_axes.md](ITEM_ops_generic_axes.md) | Fitness / generic-axis removal | gene_measurement_compare | — | S | no |
| 30 | [ITEM_ops_guide_level_similarity.md](ITEM_ops_guide_level_similarity.md) | Guide-level similarity with calibrated pair p values | gene_measurement_compare | — | S–M | no |
| 31 | [ITEM_ops_distribution_similarity.md](ITEM_ops_distribution_similarity.md) | Responder fraction / profiles, sliced Wasserstein, energy distance | gene_measurement_compare | — | M | no |
| 32 | [ITEM_ops_embedding_profiles.md](ITEM_ops_embedding_profiles.md) | Gene profiles from frozen cell embeddings and fusion | embeddings → ops_engine | — | M | no |
| 33 | [ITEM_ops_contrastive_weighting.md](ITEM_ops_contrastive_weighting.md) | Contrastive generalised-eigen Wiener weighting (S02) | gene_measurement_compare | — | S | no |
| 34 | [ITEM_ops_mito_benchmark.md](ITEM_ops_mito_benchmark.md) | MitoCarta sub-module benchmark against an external published module table | ops_clusters | — | S | no |

**Ordering rule.**
- Ranks 1–8 cover the OPS genotype and phenotype run. Rank 1 comes first: once it lands, the first run of any plate not yet processed is a spaCR run.
- Ranks 9–18 cover OPS gene profiles and their evaluation, 19–22 the BBBC021/JUMP profiling path, 23 the model architectures, and 24–27 host–pathogen, plaque and division analyses.
- Ranks 28–34 are supplement-only strategies.
- Where a later spec needs an earlier one, the earlier rank is listed first (e.g. 19 and 22 feed 23).

**Not ported, by design:**
- external comparator tools and their code (other OPS pipelines, CellProfiler, CellProfiler Analyst as a tool, copairs, HRMAn's own values); their outputs are inputs only;
- one-off corpus builds: the plaque held-out set and the literature checks;
- the gene graph;
- figure drawing (item 818 later);
- annotation readings (cluster annotation, mitotic labels, plaque gold labels, figure readings): these are stored as data, never re-derived.
