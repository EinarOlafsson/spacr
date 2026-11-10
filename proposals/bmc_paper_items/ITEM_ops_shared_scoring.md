# Item spec: shared community-standard scoring of OPS gene profiles and clusterings (copairs activity and consistency mAP, EFAAR known-relationship recall) with controls

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - `ITEM_ops_retrieval_map.md`: spaCR's own copairs-equivalent gene–gene retrieval mAP. **This item does not replace it**; it adds the field's reference packages as a second, independent scoring step, and its per-gene CORUM/STRING mAP must agree with `pair_map` on the same inputs.
  - `ITEM_ops_reference_sets.md`: the versioned references and SELECT/REPORT halves used here (CORUM@REPORT, STRING>=0.7@REPORT).
  - `ITEM_ops_profile_pipeline.md`: produces the gene and guide profiles this item scores.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.

## Why
Gene-level OPS results are compared across analyses (spaCR, Brieflow, Funk et al. 2022) on the same genes. The profiling field scores such results with two shared tools: copairs mAP (Kalinin et al. 2025) for phenotypic activity and consistency, and the EFAAR benchmark (Celik et al. 2024) for recall of known gene relationships. The scoring has to be one step, applied identically to every analysis on the same gene set, with controls, and it has to state which analysis could be scored on which metric: activity needs guide-level profiles, profile-level consistency and EFAAR need a profile per gene, and some published analyses release only clusters.

## Reference implementation
The reference analysis scripts:
- **`prep_inputs.py`** → one bundle (`inputs.npz`): shared genes (5,031), spaCR gene rows (69 dims) and guide rows (20,117 targeting + 250 NT guides, n_cells), Funk's 32 published interphase `*_score` columns (z-scored on the shared genes; 250 NT rows), cluster ids of spaCR (221), Brieflow (227) and Funk (222).
- **`score_activity.py`**: copairs `map.average_precision` with `assign_reference_index` on NT guides, `pos_sameby=[gene, ref_index]`, `neg_diffby=[ctrl]`, cosine; `map.mean_average_precision(sameby=gene, null_size=10000, threshold=0.05, seed=0)` (permutation null, BH).
- **`score_consistency.py`**: REPORT-half CORUM complexes with ≥ 3 members among the shared genes (444): `map.multilabel.average_precision` (`multilabel_col=complexes`, `neg_diffby=[complexes]`), per-complex mAP, null 10,000, BH; per-gene AP of CORUM@REPORT and STRING>=0.7@REPORT partners with `compute.ap_contiguous` and `compute.get_null_dists` (null 10,000, seed 0), BH.
- **`score_efaar.py`**: `efaar_benchmarking.benchmarking.known_relationship_benchmark` unchanged, bundled CORUM, hu.MAP, Reactome, SIGNOR, StringDB, thresholds (0.05, 0.95) and (0.1, 0.9).
- **`stats_and_figures.py`**: Fisher exact, exact McNemar over the same complexes or known pairs, paired Wilcoxon over the same genes; Holm within each panel; `summary_main.csv`, `stats.json`, figures. **`make_report_tables.py`**: report tables.
- Packages: copairs 0.6.0 (BSD-3-Clause), efaar_benchmarking 1.0 (Apache-2.0; its bundled annotation files keep their own licences, e.g. CORUM CC-BY-NC 4.0).

## What to build
In a new `spacr.ops_scoring` module (alpha), entry `score_ops_profiles(settings)`, run after the profile pipeline:
1. **Inputs**, `ops_scoring_inputs` (dict: name → gene profile file | guide profile file | cluster table). Shared genes = `ops_scoring_genes` or the intersection of all inputs. Non-targeting rows recognised as in `ops_reference` (`nontargeting*`).
2. **Scorability table** (`scorable.csv`): per input and metric, yes / "not scorable" with the reason (no guide profiles, no gene profile, clusters only). Nothing is dropped silently.
3. **Activity** (guide inputs only): as `score_activity.py`; `ops_scoring_null_size` (10000).
4. **Consistency**: complexes (`ops_scoring_min_complex`, 3) and per-gene partner retrieval for `ops_scoring_references` (['CORUM@REPORT', 'STRING>=0.7@REPORT']).
5. **Clusters as rows**: one-hot rows + N(0, `ops_scoring_cluster_jitter` = 1e-3) noise for copairs (random tie order; `ops_scoring_cluster_seeds` = [0, 1, 2], averaged); exact one-hot rows for EFAAR, where recall = share of known pairs co-clustered (reported separately from profile-level recall).
6. **EFAAR**: `ops_scoring_efaar_sources` (package defaults) and thresholds (package defaults).
7. **Controls** (`ops_scoring_controls`):
   - shuffled gene labels (activity seed 7; consistency seed 7; EFAAR seeds 70–74);
   - NT pseudo-genes: activity = NT guides in fours vs the other NT guides (seeds 1000–1004); partner retrieval = an NT row given a real gene's partner list (seeds 900–902); EFAAR = each known pair (a, b) replaced by (NT row, b) (seeds 300–302);
   - NT pseudo-complexes (seeds 500–502), reported with the caveat that NT rows share the no-knockout phenotype, so the shuffled control is the null for complexes.
8. **Outputs**: `activity_map.csv`, `consistency_complex_map.csv`, `consistency_pairs_pergene.csv.gz`, `consistency_pairs_summary_by_seed.csv`, `efaar_recall_all.csv`, `efaar_recall_summary.csv`, `efaar_mcnemar.csv`, `summary_main.csv`, `stats.json`, two figures (bars + jittered points, spaCR palette, Holm-corrected brackets).
9. copairs and efaar_benchmarking are optional dependencies (`spacr[scoring]`), pinned, never vendored; the copairs null cache goes to the run folder, not the home directory.

## Validation data and acceptance
- **Data:** the reference input bundle `inputs.npz` (+ `inputs_meta.json`) and the reference output tables, supplied with the validation data.
- **Acceptance (copairs 0.6.0, efaar_benchmarking 1.0 pinned):**
  - every row of the reference tables identical (mAP, p, q, flags; recall per source and threshold);
  - `stats.json` p and Holm q identical;
  - reference values: activity 2,649 of 5,031 genes q < 0.05 (NT pseudo-genes 0 of 155); CORUM complexes q < 0.05: profile 39.9 % (spaCR) / 33.1 % (Funk), clusters 26.1 % / 39.8 % (Brieflow) / 26.4 %; per-gene mAP CORUM 0.110 / 0.093, STRING 0.041 / 0.032; EFAAR 5 % mean 0.323 / 0.262 (profiles), 0.105 / 0.134 / 0.080 (clusters);
  - the per-gene CORUM and STRING mAP equal `pair_map` (`ITEM_ops_retrieval_map.md`) on the same profile (0.110 and 0.041).
- **Synthetic test** (unit test, CPU): planted gene groups + NT rows → planted genes active at q < 0.05, NT pseudo-genes not; shuffled labels fall to the null for consistency and EFAAR; an input with clusters only is listed as "not scorable" for activity.

## Literature
- copairs: Kalinin et al. 2025, Nat Commun 16:5181.
- EFAAR: Celik et al. 2024, PLoS Comput Biol.
- References: CORUM (Tsitsiridis et al. 2023), STRING v12 (Szklarczyk et al. 2023), hu.MAP, Reactome, SIGNOR as bundled with EFAAR.
- Screen: Funk et al. 2022, Cell 185:4634; Brieflow: Di Bernardo et al. 2026.

## Effort
- **Effort:** S–M (wrappers, controls, one table and two figures; the metrics come from the two packages).
