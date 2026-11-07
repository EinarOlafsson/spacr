## spaCR 1.5.1.8

### Fixed

- **OPS gene P values are calibrated (alpha OPS phenotype path).** The aggregate phase tested each gene against a null built by drawing non-targeting *cells*. In an optical pooled screen the cells of one guide share wells, fields and clonal patches, so guide medians vary much more than cell sampling alone predicts. That null left this extra variance out, so its P values were far too small. On plate A of Funk et al. 2022 it called 4,053 of 5,049 genes hits, and 83 of 250 non-targeting guides tested as genes were hits too.
  - The new default null, `ops_pvalue_null='calibrated'`, adds the between-guide variance that the non-targeting guides actually show. On plate A, that variance is 50 to 63 percent of the variance of a 250-cell guide median. Non-targeting guides left out one at a time and tested as genes now come out at P < 0.05 in 3.2 to 6.0 percent of cases, and fake genes made of 4 random non-targeting guides in 2.2 to 4.6 percent; 5 percent is the target. On plate A, 2,129 genes are hits on at least one of the 10 features compared with Funk et al.
  - The earlier test is still available as `ops_pvalue_null='cell_bootstrap'`, so analyses made with older versions can be rerun exactly. In Python, `spacr.gene_measurement_compare.bootstrap_gene_pvalues` takes `null='calibrated'` (the default) or `null='cell_bootstrap'`. The calibrated null needs the non-targeting guide table as `control_guides`. A script that called it with the old arguments now raises an error that names the two choices, instead of silently returning the old numbers.
  - The report of the aggregate phase records the centre, cell variance and between-guide variance of each tested feature.
- **The OPS aggregate phase no longer runs out of memory on a full plate.** It held the whole cell table in memory, which reached 46 GB on one Funk et al. plate. It now reads, normalises and summarises 48 feature columns at a time. A whole plate now peaks at about 10 GB, which the earlier code needed for a single well. The guide and gene tables are unchanged, value for value; on one plate A well the old and new code give bit-identical tables for all 352 features.
- **Features without spread in the control cells are dropped with a reason.** Twenty measurements were empty after robust z normalisation on the Funk et al. plates, because their control cells all have the same value. They are not measurement bugs; they are constant by construction at this scale:
  - `cell_<channel>_distance_to_nucleus` is the distance from a channel's intensity centre of mass to the nucleus. In most cells that centre lies inside the nucleus, where the distance is 0. It was 0 in 96 to 99 percent of cells.
  - `euler_number` is 1 for almost every cell and nucleus (no holes), and 0 for almost every cytoplasm (one ring).
  - `*_frac_low10` is the share of an object's pixels below its field's 10th percentile, which lies in the background, so it is 0 for most objects. `cytoplasm_dapi_frac_high90` is 0 for most cytoplasms, because the field's 90th DAPI percentile is set by the nuclei and few cytoplasm pixels are that bright.
  The aggregate phase now leaves such features out of the guide and gene tables and lists each one, with its reason, in `ops_feature_qc_<plate>.csv`. Set `ops_feature_qc` to False to keep them as empty columns.

### New (alpha)

These settings belong to the alpha OPS phenotype path and are shown only with **Show alpha features** on.

- `ops_genotype_table` joins the guide calls from another table, such as a later decode rule, instead of the calls stored by the decode phase.
- `ops_pool_wells` pools only some wells, for example to see how results change with the number of wells.
- `ops_pool_interphase` summarises guides and genes from interphase cells only.
- `ops_test_features` chooses the features each gene is tested on.

Runs that pool a subset of wells or only interphase cells write their tables with a `_wells-…` or `_interphase` suffix, so they do not replace the full tables.

Rerun from these settings, spaCR reproduces the plate A pilot analysis of Funk et al. 2022. The guide and gene medians agree to within float32 rounding (largest difference 4.8e-7), which is how the pilot stored its values. The hit set is the same on all 10 features, and the same 20 features are dropped.
