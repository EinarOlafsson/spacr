# Item spec: guide-bootstrap consensus clustering as an OPS clustering option (on top of PHATE + Leiden)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Extends:** `spacr.ops_clusters` (PHATE + Leiden). Reuse `gene_graph` and `leiden_partitions` unchanged. Related: `ITEM_ops_cluster_bench.md` (the tests), `ITEM_ops_modules_precomputed.md`, `ITEM_ops_profile_recipe.md`.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (stability of OPS gene clusters).
- **Status:** candidate, pending held-out and second-plate confirmation. It becomes a recommended option only after both.

## Why
A single PHATE + Leiden run depends on one clustering seed and on one draw of guides per gene. In development benchmarks one seed could split a large module, and gene profiles changed visibly when guides were resampled. Clustering many guide-bootstrap profiles and keeping what stays together gives a partition that does not depend on one draw, and a co-clustering matrix that says how stable each gene's placement is.

## Behaviour
1. **Bootstraps.** For b = 1..B: resample guides within each gene with replacement (seed = base seed + b), rebuild the gene profile from the resampled guide rows with the chosen profile recipe, build the PHATE (knn k) graph, and run Leiden at the target cluster count with its own seed (base seed + b).
2. **Co-clustering matrix.** C[i, j] = the fraction of bootstraps in which genes i and j share a cluster (a gene absent from a bootstrap is not counted in that bootstrap).
3. **Consensus partition.** Cluster the co-clustering matrix (Leiden on the graph with edges C[i, j] above a minimum, at the matched cluster count; or average linkage on 1 - C). **Every gene is assigned**: no noise label and no dropped low-stability genes. Stability is reported per gene (mean co-clustering with its own cluster) so users can filter if they choose.
4. The consensus partition is compared with a single-run PHATE + Leiden on the same genes and cluster count by the calibrated tests of the cluster benchmark.

## What to build
In `spacr.ops_clusters` (alpha): `guide_bootstrap_profiles(guide_rows, n_boot, seed)`; `coclustering_matrix(partitions)`; `consensus_partition(C, k, method, seed)`; a settings-file phase `cluster_consensus` after `cluster_sweep`. No new dependencies.

## Settings (alpha)
`ops_consensus_enabled` (false), `ops_consensus_bootstraps` (number of bootstraps B; the default is set by the benchmark below), `ops_consensus_seed`, `ops_consensus_knn` (10), `ops_consensus_k` (target cluster count; default as the sweep), `ops_consensus_method` ('leiden' | 'average'), `ops_consensus_min_cooc` (0.0).

## Outputs
In `results/ops_clusters/`: `consensus_cooc.npz` (co-clustering matrix with gene order), `consensus_partition.csv` (gene, cluster, stability), `consensus_runs.csv` (per-bootstrap seed, resolution, cluster count, seconds), and one figure (co-clustering matrix ordered by cluster).

## Acceptance and tests
- **Reproduction:** for fixed settings (B, seed), identical co-clustering matrix and partition on pinned versions.
- **Tests (CPU):**
  - on planted blocks with guide noise, consensus recovers the blocks at least as well as a single run, and gives lower stability to planted ambiguous genes;
  - every input gene has a cluster in the output;
  - the matrix is symmetric, has ones on the diagonal, and values in [0, 1];
  - with B = 1 the partition equals a single run;
  - a different seed changes the partition only within the stated tolerance.
- **Quality acceptance (decides "recommended"):** on the cluster benchmark, consensus vs single-run PHATE + Leiden at matched cluster count, on the report half and held-out genes, then a second plate, with the calibrated tests: not worse, and better on at least one of complexes or STRING pairs beyond the noise-floor margin. Until both hold, it stays an alpha option.

## Literature
- Monti et al. 2003 (Mach Learn), consensus clustering; Traag, Waltman & van Eck 2019 (Sci Rep), Leiden; Moon et al. 2019 (Nat Biotechnol), PHATE.

## Effort
- **Effort:** S to M.
