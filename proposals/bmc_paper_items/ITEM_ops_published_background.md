# Item 953: published comparator rows keep their control rows in the Fisher background

- **Lifecycle:** future (alpha), the same as item 916.
- **Number:** 953 (assigned by master 2026-11-15). The default output changes (keep_controls True), so: a schema bump and a DEFAULT_CHANGES row (older files rerun with False).
- **Related:** 916 (`ITEM_ops_leiden_resolution_sweep.md`, "Published comparator rows in the Leiden sweep"; READINESS proof card C).
- **Rule:**
  - A spaCR settings-file run must reproduce the reference outputs identically.
  - Every new or changed documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly (found on 4087169bc8).
- **Paper-critical:** yes. It blocks row 916 from becoming Ready.

## Why

Card C, check C3 (`funk_published_rows/REPORT.md`, 2026-11-15):
- `spacr.ops_clusters.score_published` gives Funk 2022's published interphase clusters CORUM 23.9 % (may2025).
- The published value is 24.3 %, the value Brieflow's comparison reports.

The cause:
- `published_clusters(spec, control_prefix)` drops the comparator's non-targeting rows.
- `cluster_sweep_phase` then scores the published partition on the targeting genes only.
- So the Fisher background of `group_enrichment` has 5,072 rows instead of 5,322.
- Brieflow's code, and spaCR's own sweep rows, count control rows as background and never as cluster members. `group_enrichment` already documents this: "control rows are background, never cluster members".

The fix was checked while diagnosing. With the 250 NT rows kept, spaCR reproduces every published value exactly:
- 222 clusters, 5,072 genes;
- CORUM 27.5 / 24.3 %, KEGG 15.8 / 13.5 % (head / may2025);
- STRING precision 0.0498, recall 0.1222, tp 3,359, fp 64,046, fn 24,137.

## What to change

1. **`published_clusters(spec, control_prefix="nontargeting", keep_controls=True)`.**
   - With `keep_controls=True` (the new default) it returns the control rows too, with their published cluster ids.
   - With `False` it keeps today's behaviour (recorded in `DEFAULT_CHANGES`).
2. **`score_published(...)`.** Each comparator is scored on the sweep's scored genes it clusters, **plus its own control rows**. The control rows enter the background only:
   - `score_clusters` already excludes them from cluster counts, enrichment membership and STRING pairs;
   - complex metrics and the shuffled control use the targeting genes only, as now.
   - Record the number of control rows per comparator in `published_record` (`control_rows_in_background`).
3. **`cluster_sweep_phase`** passes `keep_controls` from a new setting `ops_cluster_published_controls` (bool, default True).
   - Settings files stamped before this schema version rerun with False.
   - The legend in `published_record` states the rule.
4. **Tests (CPU):**
   - a synthetic table with control rows: the enrichment p values change with the background size exactly as `funk_benchmark.group_enrichment` does;
   - controls are never counted as cluster members;
   - `keep_controls=False` reproduces today's output byte for byte.

## Proof check (card C, checks C3 and C5 rerun)

- **C3.**
  - Data: the repaired Funk table `/nas_mnt/data/ops/funk2022_published/repaired/vesuvius_genes_repaired.csv`, column `interphase_cluster`.
  - Run `score_published` on all 5,072 Funk genes plus the 250 NT rows.
  - Pass: every value equals the published row in `funk_plateA_full/benchmarks_summary.json` ("all genes, repaired table"): integers exact, floats at the published rounding.
- **C2 (regression).** The reference code `funk_benchmark.score` run on the same rows (NT rows included) gives identical values (raw differences ≤ 1e-12).
- **C5.**
  - The same row produced by a spaCR settings-file run (`cluster_sweep` phase, `ops_cluster_published` = the Funk spec, `ops_cluster_published_controls` True).
  - Pass: identical to C3.
- **Controls** (maintainer rule 2026-11-15):
  - the shuffled-label control, 500 permutations (cluster sizes kept), reported as a mean with its CI. For excess-type metrics the CI must include 0. The Fisher shares (CORUM/KEGG % enriched, STRING precision) are not excess-type, so no "includes 0" criterion is applied to them; their mean and 95 % range are reported against the real row;
  - the single-seed control row (seed 20261110) must fall inside the 95 % permutation range.
- When C3 and C5 pass, row 916 moves to Ready, with the evidence in `funk_published_rows/`.

## Effort
S.
