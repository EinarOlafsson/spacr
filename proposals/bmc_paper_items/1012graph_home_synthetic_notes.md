# Home synthetic findings for the workstation graph audit

2026-11-22, inbox v172 handoff. **Home analysis has stopped. Work Codex owns the original-data investigation on the workstation; no data transfer is needed.** These notes summarize completed work only. The full [audit](1012graph_audit.md), [measurements](1012graph_audit/measurements.json) and [frozen executed harness](1012graph_audit/benchmark.py.txt) were already published at `b918f57677796760ec16779c6b84f5447353b462`. This handoff adds no computation or application changes.

The original **376/520 is target-view PCA top-ten partner retrieval; 14/520 is final matched-partition partner membership**. They share a cohort but measure different outcomes. Home did not localize that original discrepancy. Do not substitute the later standardized-cosine/Bonferroni fallback arrays for the original PCA/calibrated/max-null scores.

## Fixed experiment and results

Source: `862ab34810badc4138d0aa55c97509f56652f759`, whose max-projection/PCA owners match the original external acceptance implementation. Six predetermined arms: two-view cosine, thirteen-view cosine and thirteen-view shrunk; each paired with no-plant data. The existing 160-gene fixture has two eight-member complexes, independent six-column target views, Gaussian seed 65 and amplitude eight along unit directions seeded 12/20. Thirteen-view arms append eleven independent six-column noise views, seed 66. No-plant removes only the two shifts. There are **16 query genes and 56 true unordered partner pairs**; the remaining rows are background, not one truth cluster.

Normal production scoring: column null, `control=None`, full coverage, PCA variance 0.99, three bins, 50 calibration draws, 200 joint-null draws, seed 20261119, float32 packed, one worker, block size 32. Graph: p<0.05 plus symmetric best rescue, union kNN with k=10, calibrated −log10(p) weights. Leiden seed 42; matched target `round(160*209/2647)=13`. This small homogeneous fixture does not reproduce original dimensions, missingness or candidate-universe size.

Each cell is **planted / no-plant**, directly from `cases[*]` in the measurements. Query denominators are 16 and pair denominators 56. Keep retrieval, direct edges and co-clustering separate.

| Stage | Two-view cosine | Thirteen-view cosine | Thirteen-view shrunk |
|---|---:|---:|---:|
| Embedded target-view top-ten query hits | 16 / 5 | 16 / 5 | 16 / 5 |
| Calibrated target-view top-ten query hits | 16 / 5 | 16 / 5 | 16 / 5 |
| Across-view maximum top-ten query hits | 16 / 6 | 16 / 5 | 16 / 7 |
| Max-null strength top-ten query hits | 16 / 6 | 16 / 5 | 16 / 7 |
| Retained-candidate directed top-ten query hits | 16 / 5 | 15 / 2 | 16 / 7 |
| Threshold candidate partner pairs | 51 / 4 | 29 / 1 | 56 / 4 |
| Final union-kNN partner pairs | 49 / 4 | 27 / 1 | 56 / 4 |
| Final graph direct-partner query hits | 16 / 6 | 15 / 2 | 16 / 7 |
| Weighted Leiden resolution-1 co-cluster query hits | 16 / 10 | 13 / 10 | 16 / 7 |
| Count-matched co-cluster query hits | 15 / 10 | 11 / 4 | 16 / 7 |

In thirteen-view cosine, the **first measured loss in comparable directed retrieval is threshold/rescue selection, 16→15**. Thresholding leaves 29/56 true pairs; kNN removes two further pairs without losing another any-partner query. Best rescue adds no candidates in these arms. The same graph yields thirteen co-cluster hits at resolution 1 and eleven at matched resolution. No `match_count` split/merge occurs: every arm returns `adjusted=False`. These observations establish synthetic threshold and partition losses, not the cause of the original 376→14 discrepancy.

Adding noise views leaves 54/56 true pairs winning in their target view, compared with 56/56 for two views, yet reduces significant true pairs 51→29. Target-view winner share alone therefore misses the loss in significance and community support. True-pair weight share falls from 9.57% to 5.00%. The experiment does not fully separate a changed joint null from changed competing neighbours.

## Useful follow-ups and cautions

1. **Existing shrunk similarity is the strongest measured candidate.** It retains all 56 edges and all sixteen matched-partition query hits in thirteen views, versus 27 edges and eleven hits for cosine. However, no-plant matched hits also rise four→seven. This amplitude-based fixture favours magnitude-sensitive similarity; it does not establish a real-data improvement. Refit its own calibration/null and retain the original computational `control=None`, rather than reusing returned all-False control metadata.
2. **Separate graph recovery from partition recovery.** A direct edge can cross a community boundary, and a cluster can join partners indirectly. Inspect original graph edges before changing the resolution/count objective. The original attained counts 646/642 are already within the default 10% tolerance of target 643; default posthoc split/merge would be a no-op there too. Larger clusters increase chance co-membership, so resolution 1 is not a recommended optimum.
3. **Removing pruning has a modest benefit and a background cost.** On thirteen-view cosine, unpruned kNN changes graph hits 15→16 planted and 2→7 no-plant; true edges 27→29 and 1→4; total planted-arm edges 642→939. No partition benefit was measured for that graph.
4. **Do not discard the weights on this evidence.** Uniform weights on the same thirteen-view cosine edges reduce resolution-1 co-cluster hits thirteen→seven. This is a graph-fixed comparison, not a recalibration.
5. **Ties are a reproducibility issue, not the demonstrated planted failure.** Stable ordering and best/worst tie bounds agree with production any-partner counts. Reversing row order leaves all planted edge sets unchanged; two null arms differ by one edge. No reversed-order partitions were tested. Increasing permutation count alone cannot remove fixed-grid quantization.

No-plant p<0.05 rates are 650/12,720=5.11%, 628/12,720=4.94% and 633/12,720=4.98%. These are one-fixture sanity checks with dependent pairs, not confidence intervals or a calibration acceptance study. No real-data default or original scientific verdict should change from this handoff.

For the surviving workstation outputs, preserve the original gene/coverage order and the same 520 queries with seven partners each. Compare their ranks through per-view calibration, max projection and max-null probabilities, then record threshold/rescue flags, both directions of kNN selection, retained weights and before/after matching labels. The graph is union kNN, not mutual kNN or SNN; q values are exported but p drives this graph. Original `control=None` skips TVN. Verify the original runner forwards `valid=view>=0`; Home could not inspect the complete runner and does not claim it was omitted.

## Completed validation

One P2 scheduler job, 1607, CPU 1, RAM 2 GiB, VRAM 0, native threads 1; all six arms and exact score/graph reconstruction checks passed once. Wall 86.78 s, maximum RSS 422,288 KiB. The source tree stayed unchanged. Three independent static reviews checked source mapping, provenance and the scientific interpretation. No further runs are planned at Home.

Measurements SHA-256: `40e5cc0862760969f32a505601b5c417a7fab3510c240fbc47f2e639b6bf4f32`. Harness SHA-256: `1165dde9828dbc470e43a6d8c996181017fac15e29487fcd5a087f60faea4a14`. Both linked files remain byte-identical to the completed run.
