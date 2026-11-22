# External gene-level callers for max projection

Item 1012 extends `spacr.ops_maxproj.maxproj_scores` for gene profiles with
multiple information views. The existing OPS call, cosine default, guide null,
settings and square float64 outputs retain their previous bytes.

## API

```python
from spacr.ops_maxproj import maxproj_scores, projected_graph

result = maxproj_scores(
    rows, names, views_ix=views_ix,
    null_mode="columns", control=None, coverage=coverage,
    dtype="float32", packed=True, block_size=1024,
    calibration=50, null=200, seed=20261119,
)
graph = projected_graph(
    result["p"], knn=10, edge_p=0.05,
    n_genes=len(names), valid=result["view"] >= 0,
)
```

`rows` contains genes by measurements, `names` gives the row order, and each
entry of `views_ix` lists measurement columns in a fixed order. `coverage` is
a genes-by-views boolean array. The guide arrays may be omitted in column mode.
`control=None` skips TVN; the existing shared embedding standardises covered
rows and fits the PCA within each view. For shrunk similarity, the shrinkage
constant is the mean squared component norm over all covered rows. A boolean
control mask remains available. The default `control="prefix"` retains OPS
control selection; column mode with no matching prefix uses no controls.

When `n_cells=None`, the pair-bin covariate is the number of covered views per
gene. Providing cell counts retains cell-count bins. The shared embedding keeps
the existing missingness cutoff of 0.05 in guide mode. Column mode defaults to
`max_missing=1.0`, retaining nonempty columns; missing values take column medians
and constant columns are removed by the existing embedding owner.

## Null and coverage

For every calibration and joint-null draw, column mode independently permutes
each column among the covered genes of that view. View `v` receives a generator from
`SeedSequence([draw_seed, v])`; columns consume that stream in their listed order.
The PCA seed is derived from the same view seed sequence. Calibration draw `p`
uses `seed+p`; joint draw `p` uses `seed+100000+p`. The view order is part of the
external run contract. Worker results are combined in the fixed draw order.

Pairs with an uncovered member receive negative infinity in that view, before
the observed maximum and every null maximum. Pairs sharing no covered view have
`view=-1`, projected score negative infinity, and `p=q=1`. BH correction includes
only pairs sharing a covered view. Passing `valid=result["view"] >= 0` to the
graph owner excludes unsupported pairs and leaves unsupported genes isolated.

## Storage and memory

`dtype="float32"` and `packed=True` are opt-in. Packed vectors list `(i,j)` in
row order for `i<j`, without diagonal entries. Their offset is
`n*i - i*(i+1)//2 + j-i-1`. The result adds `packed=True` and `n_genes`; column
mode adds `null_mode="columns"`. Other result fields retain their meanings.

Similarity tiles contain at most `block_size**2` entries. The projected graph
streams rows through the existing shared KNN owner, and view labels read the
same packed layout. Observed per-view score vectors are retained; null views
stream through the maximum. BH also needs sorting work arrays. Extra worker
processes add working sets.

At 8,140 genes, each packed float32 view uses 132,502,920 bytes; 13 retained
views use about 1.72 GB. The full result vectors and count tables retained about
2.29 GB in the full-size resource smoke. That smoke, before the exact packed
lookup acceleration, peaked at 6,025 MiB with one scoring worker. The acceptance
receipt reports the final allocation and measured memory separately.

Packed float32 histogram indices use uniform-grid estimates corrected against
the actual floating-point boundaries. They match clipped right-sided
`searchsorted`, including adjacent representable values, infinities and NaNs.
Calibrated normal quantiles are evaluated once per grid/bin value and gathered.
Frozen legacy histogram and score-byte tests verify exact agreement.

## Validation and external acceptance

The test suite pins all original OPS result arrays by shape, dtype and bytes,
with both similarities and distinct seeds. It also checks the independent
column null, null calibration, coverage, float32 agreement, worker-count
reproducibility, planted recovery, and dense/packed graph and label agreement.

The full acceptance uses the approved 8,140-gene, 651-column inputs and 13 views:
an unplanted arm, an arm containing all prescribed 1x/2x/4x/8x complexes, and an
unplanted arm without `place` for hyperLOPIT niche evaluation. Each uses the
prespecified 50 calibration and 200 joint-null draws. Planted effects use the
manifest seed plus the original complex index, a unit direction, and covered
column population standard deviations (`ddof=0`) from the unplanted data.

The `n_holes` measurement appears in transcript, fitness and interaction.
The planted adapter gives each view its own column copy, yielding 653 view
references, so each prescribed effect stays within its target view. The
unplanted arms use the original 651-column table. Missing values remain missing.

All graphs, Leiden partitions, matched-count searches and positive-control
metrics use the existing spaCR 992 owners. EC labels occupy the generic group
enrichment input, direct XL-MS/IP-MS pairs occupy the pair input, and generated
complexes use the existing complex-recovery calculation. Niche labels are scored
only without `place`. Output columns name these actual roles. All generated
complexes are evaluation controls; they do not select parameters or resolution.
The matched cluster count follows the existing 209/2,647 reference ratio.

Only the eight approved input files are read. File hashes are checked before
and after each arm. Paths in the manifest provenance section are not followed.
Acceptance outputs record every parameter, source hash, input hash, timing,
memory measure, real metric and shuffled-label control.
