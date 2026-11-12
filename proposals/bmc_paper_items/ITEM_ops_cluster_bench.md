# Item spec: fair clustering test bench for OPS gene profiles (methods × seeds at matched cluster counts, development / held-out gene split, size-matched null)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Extends** `ITEM_ops_leiden_resolution_sweep.md`: build this as a second phase of the same `spacr.ops_clusters` module and reuse its `gene_graph`, `leiden_partitions` and `score_partitions` unchanged. Do not re-implement the graphs, Leiden or the scorer. Related: `ITEM_ops_reference_sets.md` (CORUM / KEGG / STRING sets and halves), `ITEM_ops_shared_scoring.md`, `ITEM_ops_modules_precomputed.md` (consensus clustering).
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data: the same split, the same partitions for the same seeds, the same metric values. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (development analysis of the OPS gene clustering).

## Why
The resolution sweep answers how many clusters to use for one method. Choosing a clustering method for a profile, and comparing profiles under several methods, needs more:
- the same genes and the same cluster count;
- the same tuning budget per method;
- a size-matched null for every metric;
- a held-out gene set that no tuning touches.

## Behaviour
- **Gene split, written once (seed 20261110).** Universe = the fixed shared gene list. CORUM complexes (≥ 2 members in the universe) are grown into regions: start from a random complex of ≤ 40 genes, then keep adding the complex with the largest share of its members already in the region, up to 150 genes. Regions are added until 650 complex-member genes are on the development side, which is then filled with random non-member genes to 1,000. The rest are test genes. Non-targeting rows are clustered with every set and never scored.
- **Complex metric.** Counts only report-half complexes that lie entirely on the scored side. The output records which complexes are split.
- **Target cluster count.** round(genes × 227 / 5,031): 45 on development genes, 182 on test genes, 227 on all genes. A count more than ± 10 % off target is split or merged to the target (largest cluster split by cosine 2-means; smallest merged into its nearest centroid) and flagged.
- **Methods.** 4 variants each; seeds 42, 1, 2:
  - PHATE + Leiden, PHATE knn 5 / 10 / 15 / 30;
  - Leiden on a cosine kNN graph, k 5 / 10 / 15 / 30;
  - hierarchical average linkage: cosine / correlation / Euclidean on unit rows / Euclidean on z-scored features;
  - Ward: raw / unit rows / z-scored / z-scored then unit;
  - HDBSCAN on unit rows, min_samples 1 / 3 / 5 / 10; min_cluster_size bisected to the target; noise goes to the nearest centroid;
  - spectral clustering, cosine kNN affinity k 5 / 10 / 15 / 30, kmeans label assignment, n_init 10;
  - k-means: raw / unit / z / z-unit, n_init 10;
  - Markov clustering (MCL), cosine kNN k 5 / 10 / 15 / 30, expansion 2, prune 1e-4; inflation bisected in [1.05, 8];
  - consensus: 10 kNN-Leiden runs (seeds 1000·seed + i), average linkage on 1 − co-association.
- **Control.** PHATE + Leiden knn 10 on the profile with its gene rows permuted.
- **Null.** Size-matched. For each distinct size vector, gene labels are permuted with the exact cluster sizes kept (50 permutations on development genes, 200 on test genes). Excess = observed − null mean, in points. Every metric is reported as excess over this null, never as a raw value alone.
- **Selection, on development genes only.** Composite = the mean over complexes, CORUM-enriched clusters and STRING F1 of excess / the PHATE + Leiden knn 10 excess.
- **Selection half vs report half.** Genes are split once into a selection half (all choices: method, variant, resolution, recipe step) and a report half plus held-out genes (reported only, never tuned on). Complexes and STRING pairs are assigned to the half in which all their genes lie.
- **One final test evaluation per profile.** Runs the reference, the selected method and the control. A second run is refused unless forced, and a forced run is recorded.
- **Plate replication.** A winning comparison is repeated with the same recipe on a second plate. A result that does not replicate on the second plate is reported as not replicated, whatever its first-plate q.

## Statistics (calibrated tests)
Each metric family uses the test that passed its own calibration gate. The test used is recorded per metric in the output.

**Clustering metrics (complexes, CORUM-enriched clusters, STRING pairs, STRING F1).**
- **Replicates.** R = 20 paired replicates per profile. Replicate r = a guide bootstrap (seed 7000 + r: guides resampled within each gene, then the gene profile rebuilt) plus its own clustering seed (100 + r). Replicates are paired by index between the two profiles being compared.
- **Effect.** The mean over replicates of (arm − comparator), in points of excess over the size-matched null.
- **p-value.** Exact sign-flip test over the 20 replicate differences (2^20 sign patterns, enumerated or exactly sampled with a fixed seed).
- **Confidence interval.** Two-level bootstrap: replicates are resampled, and within each resample genes are resampled in blocks (STRING / CORUM graph components), so the interval reflects both clustering noise and gene sampling noise. Gene blocks are needed because a single graph component can hold more than 90 % of the STRING pairs.
- **Sensitivity metric.** A per-gene-capped pair metric (each gene counts once) is reported next to STRING pairs.
- **Multiplicity.** Holm across all tests of one comparison family.
- **Calibration gate (run before a test is used for a new clustering setup).** Build jitter copies of a profile (row noise of about 1.7e-3 relative size, 20 to 40 copies), run the full replicate test of each copy against the original, and count rejections at alpha 0.05. The gate passes if no more than 5 % of copies are rejected for every metric. A setup whose gate has not been run is flagged "not gated" in the output.
- **Noise-floor margin when the gate fails.** For each (clustering stage × cluster count × metric), delta = the 95th percentile of |effect| over the jitter copies. An effect is called only if q < 0.05 AND |effect| > delta. The margin rule is itself checked by cross-validation (fit delta on half of the copies, count false positives on the other half); a cell that still exceeds 5 % false positives is reported as "not calibrated", and a call that rests only on such a cell is flagged.
- **R is fixed.** R = 20 is the gated value. R must not be raised (or lowered) without re-running the gate: at R = 40 the same test detected the real, tiny shift that jitter causes and failed the gate (7.5 % false positives). Any change of R, of the jitter size, or of the clustering stage requires a new gate record.

**Retrieval metrics (mean average precision, AUPRC; no clustering seed).** The paired-replicate test does not pass its gate here. Use the noise-floor interval test: a difference counts only if its 95 % confidence interval lies entirely outside +/- the 95th percentile of |difference| over jitter and shuffled-variable controls. Floors are computed leave-one-out for the controls themselves. Development benchmark gate: 0 of 35 control comparisons called.

**Why the calibrated tests.** Per-pair and per-complex sign-flip tests treat pairs and complexes as independent. They are not: they share genes and graph components, and one clustering seed can split a large module. On noise copies of one profile, the per-pair test called a difference in about half of the comparisons, and the per-complex test in about a fifth. Replicate-level tests with gene-block intervals remove this dependence and pass the gate.

## What to build
In `spacr.ops_clusters` (alpha):
- `gene_split(...)`;
- a `cluster_methods()` registry (name → callable(rows, variant, seed, k));
- `match_count(...)`;
- `size_matched_null(...)`;
- `bench(...)`, `select_on_dev(...)`, `final_test(...)`;
- a settings-file phase `cluster_bench` after `cluster_sweep`.

HDBSCAN and spectral clustering come from scikit-learn, and MCL is written with scipy, so there are no new dependencies.

## Settings (alpha)
`ops_bench_methods`, `ops_bench_variants`, `ops_bench_seeds` ([42, 1, 2]), `ops_bench_split_file`, `ops_bench_split_seed` (20261110), `ops_bench_dev_genes` (1000), `ops_bench_dev_complex_genes` (650), `ops_bench_genes_per_cluster` (5031/227), `ops_bench_null_permutations` (50 / 200), `ops_bench_replicates` (20, gated; changing it requires a new gate record), `ops_bench_replicate_seed` (7000), `ops_bench_cluster_seed` (100), `ops_bench_gate_copies` (40), `ops_bench_gate_noise` (1.7e-3), `ops_bench_gate_alpha` (0.05), `ops_bench_set` ('dev' | 'final'), `ops_cluster_reference_dir`.

## Outputs
In `results/ops_clusters/`:
- `bench_runs.csv` (append-only, one row per partition: observed value, null mean, excess and empirical p per metric; count, adjusted flag, seconds);
- `gene_split.csv`, `gene_split_complexes.csv`;
- `dev_selection.csv`, `dev_summary.csv`, `profile_ranking_dev.csv`;
- `test_final.csv`, `test_stats.csv` (effect, CI, exact p, Holm q, noise-floor margin, call, gate status per metric);
- `calibration_gate.csv` (rejection rate per metric and setup, jitter-effect percentiles);
- one bars + jitter figure.

## Validation data and acceptance
- **Data:** the final Funk 2022 plate A gene-profile rows, the Brieflow-recipe profile of the same cells, Funk's published per-gene scores, and the fixed shared gene list, held with the reference outputs.
- **Acceptance:** with phate 2.0.0, leidenalg 0.12.0, igraph 1.0.0, scikit-learn 1.9.1 and scipy 1.18.1 pinned:
  - identical `gene_split.csv`;
  - identical partitions for every (profile, method, variant, seed);
  - identical metric columns (integers exact, floats ≤ 1e-12);
  - identical replicate tables, p-values and intervals for fixed seeds.
  A version change re-baselines the reference with the versions recorded.
- **Tests (CPU):**
  - every method recovers planted blocks at the planted count;
  - the shuffled-profile control gives an excess of about 0;
  - a second final evaluation is refused;
  - a profile missing genes warns and records the gene count;
  - the gate: jitter copies of a profile give <= 5 % rejections at R = 20 for every metric; a setup that fails is tested with the noise-floor margin and is never silently reported with the plain test;
  - changing R without a new gate record is refused;
  - the exact sign-flip p equals brute-force enumeration on small R;
  - a planted effect larger than the margin is called, one smaller is not.

## Reference result (earlier development run; re-baseline under the calibrated tests above)
On test genes, no method was clearly better than PHATE + Leiden:
- **spaCR:** spectral clustering was chosen on development genes. Against PHATE + Leiden it gave:
  - complexes excess 35.0 vs 31.1 points (Holm q 0.28);
  - CORUM-enriched clusters 19.0 vs 17.7 (q 1);
  - STRING F1 0.058 vs 0.083 (q < 1e-8).
- **The other two profiles:** the method chosen was PHATE + Leiden itself or one of its variants.

## Literature
- Traag, Waltman & van Eck 2019 (Sci Rep), Leiden; Moon et al. 2019 (Nat Biotechnol), PHATE; van Dongen 2000 (thesis), MCL; Campello et al. 2013 (PAKDD), HDBSCAN; von Luxburg 2007 (Stat Comput), spectral clustering; Monti et al. 2003 (Mach Learn), consensus clustering.
- Funk et al. 2022 (Cell); Di Bernardo et al. 2025 (bioRxiv, Brieflow).

## Effort
- **Effort:** M.
