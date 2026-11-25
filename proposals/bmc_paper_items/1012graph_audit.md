# Max-projection graph audit

Read-only specification, 2026-11-22. No application, settings or test changes.

**Ownership update, inbox v172:** further original-data investigation belongs to Work Codex on the workstation, where surviving outputs reside. Home has stopped this task; no data transfer or additional Home computation. The requested concise handoff is [1012graph_home_synthetic_notes.md](1012graph_home_synthetic_notes.md). The findings below describe the completed Home audit and its limits; the earlier request for original arrays is superseded by workstation ownership.

**The supplied 72.3% and 2.7% do not locate the loss step:** the first measures target-view PCA neighbours; the second measures final partition membership. The original intermediate arrays were discarded according to the Work diagnostic. A small, fixed synthetic experiment does locate losses at the probability threshold and partition stages. The existing shrunk similarity recovers the synthetic complexes, but that is not sufficient evidence to change the real-data default.

## Sources and scope

The publication base is nightly `4bb6cc7a3f80a77903d600a6380971468a83ade9`. The numerical audit and experiment use the separate external implementation at **`862ab34810badc4138d0aa55c97509f56652f759`**. Its max-projection and PCA code is unchanged from original acceptance commit `2c65174074ca3900a83fcc71ea07a5e179202bf4`; the later commit changes complex-recovery statistics. Nightly does not yet contain that external implementation. All source links below pin the audited revision.

Work results are attributed to the checked-in [original acceptance receipt](https://github.com/EinarOlafsson/spacr/blob/2c65174074ca3900a83fcc71ea07a5e179202bf4/reports/1012_external_maxproj/acceptance.json), [diagnostic readout](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/reports/1012fix/READOUT.md), [PCA diagnostic](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/reports/1012fix/embedding-cosine-diagnostic.json) and its [frozen script](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/reports/1012fix/source/pca_trace.py.txt). They were not rerun at Home. New Home results are in [measurements.json](1012graph_audit/measurements.json); the exact executed diagnostic harness is frozen as [benchmark.py.txt](1012graph_audit/benchmark.py.txt). These two supporting files are data and a read-only reproducer, not application changes.

## What the reported endpoints mean

The original inputs contain 8,140 genes, 651 raw features and 13 views. The 8x cohort is 65 eight-member complexes, five per view: 520 members and 1,820 unordered within-complex pairs. These are part of one arm containing all prescribed 1x/2x/4x/8x plants. The no-plant arm uses the same membership assignments.

| Metric | Planted 8x | No plant | Meaning |
|---|---:|---:|---|
| PCA top-ten any-partner | 376/520, 72.31% | 4/520, 0.77% | At least one of seven partners among the ten nearest covered genes in the assigned target view; cosine after normal per-view PCA, self excluded, stable input-order ties. |
| Saved matched-partition any-partner | 14/520, 2.69% | 2/520, 0.38% | At least one partner has the same final cluster label. This does not measure retained graph edges. |
| Secondary saved-partition complex share | 7/65, 10.77% | 1/65, 1.54% | Any two members of a complex share a cluster. This is yet another denominator and success rule. |

Sources: the linked PCA diagnostic/script; original acceptance stratum CSVs and the [saved-partition secondary script](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/reports/1012fix/source/secondary_saved_partitions.py.txt). The 14 members can also be reconstructed directly from the original real-8x CSV rows: five views contribute two members each and `structure_computed` contributes four. PCA retrieval is heterogeneous: ESM gives 2/40 hits, whereas interaction, secondary structure and 3di each give 40/40. The statement that every plant survives PCA is unsupported.

The original matched target is the code-defined `round(8140 * 209 / 2647) = 643`; actual partitions have 646 planted and 642 unplanted clusters. This describes a code constant and Work receipt, not an independently verified published comparator result. Matching cluster count does not match neighbour count, cluster sizes or biological recovery. The original zero complex-recovery result used a different rule, requiring half the complex together. The later Fisher/Bonferroni recovery correction does not change these graph edges or the distinct-member partner predicate.

## Actual construction and owners

The external caller selects cosine, variance 0.99, three gene bins, 50 calibration draws, 200 joint-null draws, seed 20261119, column null, `control=None`, packed float32, block size 1024, three workers, ten neighbours and edge p<0.05. `n_cells=None` makes the gene bins depend on view-coverage counts. Default arguments of the generic function instead retain the older OPS guide-null route; they must not be substituted for these selected external arguments.

| Stage | Actual operation | Existing owner |
|---|---|---|
| Per-view input/PCA | Select covered genes; convert nonfinite values to missing; retain columns below the missingness cutoff; median-fill, drop constant columns, standardize, then unwhitened PCA retaining 0.99 variance. Column-null mode defaults to cutoff 1.0, retaining nonempty columns. | [`ops_reference._GeneEmbedding`, line 607](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_reference.py#L607); [`ops_maxproj._view_arrays`, line 114](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L114) |
| Similarity | Cosine of embedded rows. Optional shrunk similarity adds c to each squared row norm; with no controls, c is the covered-row mean squared norm. `control=None` skips TVN entirely in this run. | [`view_similarity`, line 254](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L254) |
| Per-view null/calibration | Independently shuffle columns among covered genes and refit PCA. Seeds derive from draw seed and view index, including the observed PCA seed. Pool unordered null pairs by pair stratum on a 0.0005 similarity grid; inclusive upper tail with +1 correction becomes `-Phi^-1(p)`. Monotone within a stratum, but introduces ties and can reorder across strata. | [`_iter_view_rows`, line 142](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L142); [`calibrate_view`, line 403](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L403) |
| View combination | For each gene pair take the highest eligible calibrated view score. Ties select the first eligible view. A pair sharing no view gets score −infinity and view −1. There is no per-view kNN before this maximum. | [`max_projection`, line 437](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L437) |
| Joint max-null | Fresh independently randomized views, maximum across them, then pooled pair-stratum upper tails on a 0.005 score grid with +1 correction. Return p and BH q. **The graph uses p, not q.** The null maximum is across views for each pair, not across every gene pair. | [`max_null_pvalues`, line 466](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L466); [`maxproj_scores`, line 695](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L695) |
| Pruning/rescue | Eligible pairs with strict p<0.05 survive, plus each row's best eligible partner and the reverse of those best choices. Strength is −log10(p). Coverage-aware calls require `valid=result['view'] >= 0`. | [`_projected_strength_rows`, line 57](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L57); [`projected_graph`, line 486](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_maxproj.py#L486) |
| Neighbours and weights | `argpartition` selects ten strongest retained candidates per row; nonfinite entries are skipped. **Union** of directed choices makes an undirected graph, with weight `max(-log10(p), 1e-6)`. Degree can exceed ten through incoming edges or fall below ten after pruning. Ties have no explicit gene-identity rule. | [`_knn_rows`, line 1307, and `_knn_graph`, line 1325](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_clusters.py#L1307) |
| Partition/count matching | Weighted Leiden RB configuration objective, full optimization, selected seed/resolution. `match_resolution` uses up to 22 log-bisections over 0.001–1000 to approach the requested count, first solution on equal count error. | [`leiden_partitions`, line 1359](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_clusters.py#L1359); [`match_resolution`, line 1954](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/spacr/ops_clusters.py#L1954) |

There is no shared-neighbour/Jaccard weighting, mutual-kNN restriction or clustering of display coordinates in this route. Best-pair rescue does not guarantee a globally connected graph. A maximum cannot reduce a pair's score, but it can promote competing pairs and worsen its rank; the joint-null probability can also change with the view family.

`match_count` is a separate optional helper: outside 10% count tolerance it splits/merges using supplied profile geometry; inside tolerance it only renumbers labels. The original 646/642 counts are already within tolerance of 643, so default posthoc split/merge cannot explain this outcome. The complete original runner is represented by a path/hash in the receipt, not frozen source; exact forwarding of the graph `valid` mask cannot be confirmed. A missing validity mask would allow unsupported p=1 pairs into rescue and should be checked in the runner, not asserted as an observed bug.

The optional maxproj PHATE branch has a separate static wiring defect: it passes the `(graph, weights)` tuple returned by `fused_phate_graph` straight to Leiden (`ops_maxproj.py:957`; `ops_clusters.py:2518`). `_pairs_graph` already shows the required unpack/weight assignment. This option was not run and does not explain the selected kNN result.

## Small planted experiment

The six arms were fixed before execution: two-view cosine, thirteen-view cosine and thirteen-view shrunk, each paired with no-plant rows. Reuse the [existing external planted fixture](https://github.com/EinarOlafsson/spacr/blob/862ab34810badc4138d0aa55c97509f56652f759/tests/test_ops_maxproj_external_1012.py#L235): 160×12 standard-normal values, seed 65; add eight times a unit direction to rows 0–7/columns 0–5 and rows 8–15/columns 6–11, directions seeded 12 and 20. For thirteen views, append 66 unplanted Gaussian columns seeded 66, split all columns into six-column views. No-plant removes only the two shifts. The background rows are not a ground-truth cluster.

All runs use normal production helpers, `control=None`, full coverage, variance 0.99, three bins, 50/200 draws, seed 20261119, float32 packed, k=10 and p<0.05. Resource limits require one worker; packed block size is 32. These differ from the original worker/block sizes, and the fixture differs substantially in sample size, missingness and view dimensions. Every target view retained all six PCs. The instrumented similarity-to-calibration reconstruction matches returned scores exactly; recorded neighbour union and weights match the actual graph exactly. No source monkeypatches, test stand-ins, parameter search or favourable-seed selection.

All query counts below have denominator **16** and partner-pair counts denominator **56**. Rank rows use production `argpartition`, exclude self and reject nonfinite candidates. Stable ordering and best/worst tie bounds give the same any-partner counts in every measured arm. Partition membership and direct edges remain separate metrics. Each cell is **planted / no-plant**. Source: `cases[*]` in the accompanying measurements.

| Stage/metric | 2 views, cosine | 13 views, cosine | 13 views, shrunk |
|---|---:|---:|---:|
| Raw target-view cosine, top-ten query hits | 16 / 5 | 16 / 5 | 16 / 5 |
| Embedded target-view similarity, top-ten query hits | 16 / 5 | 16 / 5 | 16 / 5 |
| Calibrated target view, top-ten query hits | 16 / 5 | 16 / 5 | 16 / 5 |
| Across-view maximum, top-ten query hits | 16 / 6 | 16 / 5 | 16 / 7 |
| Max-null strength, top-ten query hits | 16 / 6 | 16 / 5 | 16 / 7 |
| Retained candidates, directed top-ten query hits | 16 / 5 | 15 / 2 | 16 / 7 |
| Threshold candidate graph, direct partner query hits | 16 / 6 | 15 / 2 | 16 / 7 |
| Threshold candidate graph, partner pairs | 51 / 4 | 29 / 1 | 56 / 4 |
| Threshold plus best rescue, partner pairs | 51 / 4 | 29 / 1 | 56 / 4 |
| Final union-kNN graph, direct partner query hits | 16 / 6 | 15 / 2 | 16 / 7 |
| Final union-kNN graph, partner pairs | 49 / 4 | 27 / 1 | 56 / 4 |
| Weighted Leiden at resolution 1, co-cluster query hits | 16 / 10 | 13 / 10 | 16 / 7 |
| Count-matched Leiden, co-cluster query hits | 15 / 10 | 11 / 4 | 16 / 7 |
| Count-matched Leiden, co-cluster partner pairs | 49 / 8 | 25 / 2 | 56 / 5 |

Matching used seed 42 and the unchanged ratio, `round(160*209/2647)=13`. All matched partitions attained thirteen clusters except two-view no-plant, which attained twelve; its count is inside tolerance. A separate observation call to `match_count` returned `adjusted=False` in every arm. It did not cause any measured membership loss. Resolution 1 is a fixed diagnostic comparison, not a selected optimum.

**Located loss in this fixture:** thirteen-view cosine keeps all sixteen top-ten queries through the max-null ranking. The first loss in that same directed retrieval metric occurs when p-threshold/rescue candidates are selected: sixteen to fifteen. The threshold retains 29 of the 56 labelled pairs; kNN removes two further pairs, leaving 27, without another any-partner query loss. All best-rescue candidate masks equal the threshold masks here. The partition then has thirteen partner-bearing queries at resolution 1 and eleven at the count-matched resolution. These are community-assignment losses relative to direct-edge availability, not proof that graph construction discarded every missing relationship.

**Adding views matters without stealing most winning views.** Cosine target-view wins change from 56/56 to 54/56 planted pairs, yet significant partner pairs fall from 51 to 29 and their share of total graph weight from 9.57% to 5.00%. Thus a near-perfect target-view winner share does not establish preserved edge significance or community support. A pair can keep its winning view while the multiple-view null and competing neighbourhood change. This experiment does not separate those two effects fully; a global maximum should not be described as a raw-score decrease.

**Sensitivity checks on the same fitted scores:**

| Comparison | Observed result | Interpretation |
|---|---|---|
| Remove p pruning; keep k=10 and weights | Thirteen-view cosine graph hits 15→16 planted, 2→7 no-plant; partner edges 27→29 planted, 1→4 no-plant. Total edges 642→939 planted. | Small positive-control recovery comes with many more background edges. This is not a demonstrated partition fix. |
| Replace retained edge weights by one, resolution 1 | Thirteen-view cosine co-cluster hits 13→7 planted; no-plant 10→9. Edges unchanged. | Uniform weights worsen this fixture. Do not recommend discarding calibrated weights from these data. |
| Reverse row order, restore identities, rebuild graph | All three planted edge sets unchanged (Jaccard 1). Two-view cosine null and thirteen-view shrunk null differ by one edge; thirteen-view cosine null unchanged. | Grid/tie sensitivity exists, but does not explain the planted losses observed here. No partition-reversal test was performed. |
| Existing shrunk similarity, full recalibration/null | Thirteen-view planted graph has all 56 partner edges and all 16 matched-partition hits; no-plant has 4 partner edges and 7 matched hits, versus cosine 1 and 4. Planted partner weight share becomes 19.35%. | Promising for this amplitude-based plant; the assigned null membership also rises. No real-data or across-seed advantage is established. |

No-plant p<0.05 rates over 12,720 pairs are 650/12,720=5.11%, 628/12,720=4.94% and 633/12,720=4.98%, respectively. These are single-fixture sanity checks with dependent pairs, not confidence intervals, a calibration acceptance study or estimates of final graph/cluster false discovery. The much smaller candidate universe also makes chance any-partner hits more common than in the original 8,140-gene diagnostic.

## Ranked proposals

First add a **diagnostic export through the existing owners**, with no numerical change: preserve fixed query IDs, seven partner scores/ranks, top-ten candidates, view coverage/winner, threshold/rescue flags, directed choices, retained weights and final labels. Report retrieval, edge retention and co-clustering separately. This has no expected recovery benefit by itself and low scientific risk; it is the prerequisite for deciding which original stage needs a fix. Do not implement a second graph or PCA owner.

The following ranking is for subsequent controlled experiments, not approval to change settings or claims of predicted percentage improvement on real data.

| Rank | Candidate | Expected recovery and evidence | Risk / required discriminator |
|---:|---|---|---|
| 1 | Test the existing `similarity='shrunk'` with its own calibration and joint null. | Strongest measured positive-control improvement: thirteen-view graph pairs 27→56 and matched query hits 11→16. No new similarity implementation needed. | Medium/high scientific risk: changes magnitude sensitivity and may favour this amplitude-based planting; no-plant matched hits also increase 4→7. Require original-view coverage strata, low-norm true positives, held-out references and calibration checks before considering a default. Preserve `control=None`; returned all-False control metadata is not an equivalent computational argument for shrunk c. |
| 2 | Keep graph recovery and count-matched partition recovery as distinct outputs; examine partition objective/resolution only if original edges survive. | In the toy, resolution 1 gives 13 versus 11 matched hits on the same thirteen-view graph, and 16 versus 15 with two views. Removing an unjustified count constraint could help, but is unproven. | Low risk for reporting; high risk for changing partition policy. Larger clusters inflate co-membership by chance. Do not choose resolution from REPORT labels or silently abandon the matched-count comparison; evaluate size-conditioned null recovery and held-out biological criteria. |
| 3 | Test view-aware candidate retention before the single final graph, with bounded degree and the same calibrated strength owner. | Could retain relationships that rank well in a target view but lose global competition. Original magnitude unknown; synthetic top-ten query retrieval itself survives until pruning. | High risk: a naive union of ten neighbours per each of thirteen views can greatly increase degree and background connections. Needs a prespecified budget, proper null recalibration for any changed score, and comparison at matched density. No support here for mutual-kNN, which would prune further. |
| 4 | Test a less restrictive p-pruning policy, keeping k and weights fixed. | Only one additional planted graph query and two pairs recovered in the thirteen-view toy. Partition benefit unmeasured. | High specificity risk: no-plant graph queries rise 2→7 and total planted-arm edges rise 642→939. Do not call existing p<0.05 an FDR guarantee or simply remove it in production. |
| 5 | Make top-k tie handling explicit; investigate finer/exact tail calibration only if original rank plateaus are shown. | Useful reproducibility improvement; no planted recovery gain demonstrated. All positive-arm reversal edge sets already agree. | Low/moderate risk for deterministic tie ordering, higher for changing grids or probability estimates. Recalibration and scientific checks needed if scores change; never break ties using complex labels. More permutations alone cannot remove fixed-grid quantization. |

Do not change PCA retention, add control whitening, remove weights or increase k by default on this evidence. PCA reduction, sparse/heterogeneous view coverage and larger candidate universes remain untested by this toy. The optional PHATE wiring defect is a separate maintenance proposal, not a validated recovery alternative.

## What is still required to localize the original loss

The Work readout says the original numerical results were deleted; its receipt preserves shapes/dtypes/hashes. Original packed projected/p/q arrays were float32 and had 33,125,730 unordered pairs. The later saved fallback p/view arrays use a **different** standardized-raw-cosine, empirical per-view tail and Bonferroni-min method. They cannot be inserted as an original pipeline stage. Its failed no-plant calibration, 187/7,280=2.57% against its stated 3–7% band, remains a failure; this audit does not waive it.

A request was appended to the master inbox status at 17:32 on 2026-11-22 asking whether original per-view/PCA/calibration/graph snapshots survive. The following compact existing data would suffice without transferring full pair matrices:

1. Gene order, covered-gene masks, view definitions, planted membership, exact source/config hashes, and both saved matched partitions.
2. For each of the same 520 query genes: all seven partner ranks/scores and top-ten IDs at each original score stage, evaluated against the full covered candidate universe; target/winning view, probability and candidate-retention flags.
3. Directed top-ten choices for every gene, with weights, so incoming union edges are preserved; graph labels before/after resolution matching and any actual count adjustment.

Known Work partitions are under `/home/carruthers/Documents/repo/spacr-codex-recovery-20261121/1012/acceptance/{planted,no_plant}/matched_partition.parquet`. Their reported SHA-256 values are `410f5d4ac0e7cbeb4719d2d7b833daef81447070cdac9cc77552964cc8f25bfd` and `550f3851fc910bddade93cc02ba4beeb823fbed4218e3a4fe83c711cb02bcdd4`. The documented Work input/output roots are absent on Home. No original stage snapshot has been supplied as of this audit. A new full diagnostic requires separate scheduling; none was started or requested here.

Acceptance for a later fix: reproduce the endpoint definitions first; locate loss using comparable metrics; keep the same gene/partner universe; show both planted and no-plant effects across the prescribed strengths/views; distinguish direct edges from co-membership; preserve coverage correctness and calibrated null behavior; evaluate real-data criteria independently of labels used for choosing the fix. The original 376→14 discrepancy remains **unlocalized**, while the synthetic threshold/partition losses above are measured.

## Execution and reproducibility

One scheduler job, **1607**, P2, CPU 1, RAM 2 GiB, VRAM 0, native numerical threads 1, nice 19. Actual systemd limits were checked: memory 2,147,483,648 bytes, swap 0, CPU quota one core and weight 800. The actual global scheduler uses compute-app PID accounting and job cgroups; Chrome/graphics are not classified as compute jobs. GPU gates apply only to GPU requests; this job also hid CUDA. No downloads or full test suites.

All six arms and exact reconstruction assertions passed once. Wall time **86.78 s**, maximum RSS **422,288 KiB**, scope memory peak **354,226,176 bytes** (different accounting measures). NumPy 2.5.2, SciPy 1.18.0, scikit-learn 1.9.0, igraph 1.0.0, leidenalg 0.12.0. The source tree was unchanged before/after execution. No production tests were needed for this read-only spec; the scientific measurements are the bounded validation.

Reproducer SHA-256: `1165dde9828dbc470e43a6d8c996181017fac15e29487fcd5a087f60faea4a14`. Measurements SHA-256: `40e5cc0862760969f32a505601b5c417a7fab3510c240fbc47f2e639b6bf4f32`. Local execution folder: `/mnt/wd4tb/worktrees/codex-1012graph-source/.codex-audit/B0-run`; resource/time logs and unchanged scheduler/runner pins are retained there. To reproduce, place the frozen harness under `.codex-audit/benchmark_graph.py` in a fresh checkout of the pinned diagnostic source and submit through the applicable declared-resource scheduler. It creates `.codex-audit/B0` and refuses to overwrite an existing run. Do not run against the publication tree's older numerical owners.
