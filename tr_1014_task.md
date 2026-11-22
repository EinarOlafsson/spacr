# Item 1014 translation handover

Branch: `ws-offload-codex-work-1014`; assigned base `19a95cd7993e0ce44078c779828b91098820598f`.

Home owns translation records and catalog rebuilds. This branch writes no translations. The companion `tr_1014_sources.json` contains exact canonical bodies, SHA-256 hashes, before bodies, source locations, block layouts and file hashes.

Locales: sv, de, es, zh_CN, pt, hi, ko, is, fr. Preserve identifiers, DNA sequences, settings names and format fields.

## L3 compatibility note

'none' keeps the earlier uncalibrated tables and reports no shuffles.

Both new-file defaults are `never`. Schema 957 preserves `none` for older files. The three shared alpha controls appear once in Decoys & auto thresholds; both decode paths use them. Entered scales stay 1.268 and 1.0.

## PRELIMINARY paper references

S2J reference points remain PRELIMINARY: Funk b=1, 333 fields, 1.35566 [1.2382, 1.4754]; PERISCOPE b=1, 85 fields, 1.96469 [1.7586, 2.1597]. Exact equality against the materialized D/N1/N2 sets and reference fields is deferred to Home after transfer. These values are not claimed as branch validation.

## Public API sources

11 exact public bodies: two new functions and nine revisions, 114 canonical prose blocks. Historical JSON fixtures and global inventory/signature pins are unchanged. Approved adapters bind every revised source and both new arrivals.

### `spacr.ops_decode.calibrate_fdr_scale`

`spacr/ops_decode.py:1816`; SHA-256 `2b1a788f4b372b797c49100cfc5cadda7febcd56b847fd703e26ad6a118250dc`.

```python
counts, shuffled, nuclei, sites, axes: Sequence[Tuple[str, Sequence[float]]], *, k: int, scale: float, ci: bool=True, max_rounds: int=20, method: str='shuffle', never=None, **options: Any
```

Tune one well with a scale measured by shuffle or never-barcode controls.

The existing shuffle method only raises the entered scale until its
call-rate check passes. Never calibration measures wrong library calls
from substitution-control opportunities and may lower or raise the
scale, with a floor of one. Both methods use this loop and the same
threshold tuner. A repeated never setting ends the loop; a cycle keeps
the larger tested cycling scale and its matching threshold result.
The report half is held out from never calibration.

:param counts: ``(fields, *axis lengths, 2)`` real and decoy calls.
:param shuffled: matching shuffle-call grid, or None without shuffle.
:param nuclei: cells per field.
:param sites: each field site.
:param axes: axis names and values passed to ``tune_thresholds``.
:param k: positive matched-decoy multiplicity.
:param scale: entered scale; shuffle cannot lower it, never can.
:param ci: choose the upper field-bootstrap scale and tuner intervals.
:param max_rounds: maximum tuner passes, default 20.
:param method: none, shuffle (the compatibility default), never or
    never+shuffle, which takes the larger of the two measured scales.
:param never: mapping of ``never2_calls``, ``opp2_N2``, ``opp2_L`` and
    optional ``never1_calls``, ``opp1_N1``, ``opp1_L`` field/grid arrays.
    Entry points also supply ``library_size`` for the 50-guide guard,
    ``never_sets`` metadata and optional N1/N2 real-call grids.
    Fewer than 30 tuning-half N2 calls keep the entered scale.
:param options: other ``tune_thresholds`` keywords.
:returns: the tuner result with ``fdr_scale`` and ``calibration``.
    Shuffle keeps its existing ratio, per-part call-rate checks and
    round structure. Never adds point/interval estimates for each
    half, entered-scale FDR, status, control metadata and perturbation
    changes. Production calls are never replaced by control calls.
:raises ValueError: for an unknown method, invalid grid shape, missing
    never arrays or fewer than one permitted tuner pass.

### `spacr.ops_decode.compare_summary`

`spacr/ops_decode.py:3631`; SHA-256 `c9174f08da5b11ef1a2361b78eca5be5a272f956495d838157c129e6d155dae2`.

```python
units, names: Sequence[str], *, decoy_k: int=4, fdr_scale: float=1.268, labels: Optional[Mapping[str, str]]=None
```

Category counts with a decoy FDR per table, and per-table totals.

:param units: rows from :func:`compare_field` with ``plate`` and
    ``field`` added. Optional per-row ``fdr_scale`` values carry the
    applied well scales; pooling weights decoy counts by those scales.
:param names: the table names, in order.
:param decoy_k: decoys per guide.
:param fdr_scale: the FDR calibration.
:param labels: display names for the columns.
:returns: ``(categories, tables)`` DataFrames: per plate and category,
    ``n``, ``pct_of_matched_triples``, ``fields`` and real calls for each table
    with decoy calls and FDR; per plate and table over the units every
    table holds, real calls, assigned % and FDR.

### `spacr.ops_decode.fdr_report`

`spacr/ops_decode.py:2260`; SHA-256 `04b42dbbeabee30537f20418810e6171e88fbf9e33e4a019cf84648e1813611c`.

```python
field_stats, *, plate: str, well: str, k: int, scale: float, skipped_sites: Sequence[int]=(), calibration=None
```

The decoy FDR report for one well.

:param field_stats: one row per decoded field with ``nuclei``,
    ``assigned_real`` and ``assigned_decoy``.
:param plate: the plate.
:param well: the well.
:param k: decoys per guide.
:param scale: the FDR calibration (:func:`fdr`).
:param skipped_sites: the sites not decoded because the stitch did not
    place them.
:param calibration: optional shared calibration report; supplied only
    after controls were measured. Available shuffle fields retain
    their existing names. Never calibration adds its full report,
    point estimates and confidence intervals.
:returns: the report: nuclei, assigned, assigned %, decoy calls, FDR per
    100, and the skipped sites.

### `spacr.ops_decode.never_scale`

`spacr/ops_decode.py:1725`; SHA-256 `a040cd58b688e529fc0381b83a214bf40ce537ee9dc8410eea9f5f0c6909245f`.

```python
real, decoy, never2_calls, opp2_N2, opp2_L, *, k, ci=True, reps=_AUTO_REPS, seed=_AUTO_BOOT_SEED, never1_calls=None, opp1_N1=None, opp1_L=None
```

Estimate the decoy FDR scale from calls on substitution-built controls.

The N2 call rate per distance-two opportunity predicts wrong library
calls at that distance. Libraries containing distance-one guide pairs
also contribute the N1 rate times their distance-one opportunities.
The expected wrong calls times k, divided by the observed decoy calls,
give the scale. A zero decoy sum uses ``_AUTO_ZERO_DECOYS``, as the
existing threshold tuner does. With ci, the chosen scale is the upper
end of the 95 percent field bootstrap, using the existing row sampler.
Clipping the chosen scale to one belongs to ``calibrate_fdr_scale``.

:param real: real production calls per field at one grid setting.
:param decoy: production decoy calls in the same fields.
:param never2_calls: calls on the distance-two control set.
:param opp2_N2: production guide counts times their N2 distance-two neighbours.
:param opp2_L: the corresponding other-library distance-two opportunities.
:param k: positive matched-decoy multiplicity.
:param ci: choose the upper bootstrap scale rather than its point estimate.
:param reps: bootstrap draws, default the existing ``_AUTO_REPS``.
:param seed: field-bootstrap seed, default ``_AUTO_BOOT_SEED``.
:param never1_calls: optional distance-one control calls per field.
:param opp1_N1: N1 distance-one opportunities; supplied with never1_calls.
:param opp1_L: other-library distance-one opportunities, otherwise zero.
:returns: dict with ``scale``, ``scale_point``, ``scale_ci95``, ``rho1``,
    ``rho2``, ``rho1_over_rho2``, ``fdr_cal_pct``, their ``*_ci95``
    intervals, ``expected_wrong`` and summed real/decoy/control counts.
    Intervals are None for fewer than two fields. Rates with no
    control opportunities and estimates depending on them are None;
    a ratio with zero rho2 is also unavailable.
:raises ValueError: for misaligned, negative or nonfinite counts, a
    nonpositive multiplicity, or incomplete N1 inputs.

### `spacr.ops_decode.plate_fdr_summary`

`spacr/ops_decode.py:2306`; SHA-256 `d3e4bc6058eb0e30d4704dcc036f296d7e52998eb77d96d9eaca1172185ab752`.

```python
folder: str, *, plate: str, k: int, scale: float, reps: int=1000, seed: int=0
```

Assigned percentages and FDR across all field statistics for the plate.

:param folder: the ``results/ops_decode`` folder holding
    ``field_stats_<well>.csv`` files.
:param plate: the plate.
:param k: decoys per guide.
:param scale: the FDR calibration.
:param reps: field-bootstrap replicates for the 95 % FDR interval.
:param seed: the bootstrap seed.
Per-well never reports provide their applied scales; the plate FDR
and its existing field bootstrap use those same scales. Fixed and
shuffle-only plates retain their previous summary calculation.

:returns: the summary, with per-well rows; empty wells when no stats
    were written.

### `spacr.ops_decode.run_compare_calls`

`spacr/ops_decode.py:3759`; SHA-256 `7c26a8857e68a67d354bbf67561de5f700c6ce9c82914752a8d42c222e70b813`.

```python
db: str, plate: str, wells: Sequence[str], settings: Mapping[str, Any], library: Sequence[str]
```

The compare_calls phase: spaCR's genotype calls against other pipelines'.

Each external table holds one row per cell with ``well``, ``site``,
``y`` and ``x`` in that site's tile frame (the sequencing cycle-1 tile)
and ``call`` (or ``guide``), empty for an uncalled cell, optionally
``cell``. spaCR's side is every nucleus of the decoded fields with its
call from ``results/ops_decode/calls_<well>.parquet``, decoys included.
Cells are compared where their field owns them and at least
:data:`COMPARE_EDGE_PX` from its border; matched units are categorised
and each category gets a decoy FDR per table.

:param db: the measurements database.
:param plate: the plate.
:param wells: the wells decoded.
:param settings: read for ``ops_compare_tables``,
    ``ops_compare_match_px``, ``ops_compare_fields`` (``well:site``
    pairs separated by commas, or empty for every site of the tables),
    ``ops_decoy_library_k``, ``ops_fdr_scale`` and
    ``ops_decode_fdr_calibration``. Never methods read applied scales
    from current per-well decode reports; missing reports keep the
    entered scale. Legacy methods retain their earlier tables.
:param library: the guide barcodes, for the decoy set.
:returns: the report, with the paths written under
    ``results/ops_compare``.
:raises ValueError: when a table lacks its columns or the decode
    calls are missing.

### `spacr.ops_decode.threshold_settings`

`spacr/ops_decode.py:701`; SHA-256 `172cba2e797e4a0bfe195f58f5e1c5aeb8eacb113d2bdedf9f0a5c517ef6c6b4`.

```python
settings: Mapping[str, Any], rule: Mapping[str, Any]
```

The decode threshold mode and, in auto mode, its grid and target.

:param settings: the run's settings, read for
    ``ops_decode_threshold_mode``, ``ops_decode_fdr_max``,
    ``ops_decode_auto_grid``, ``ops_decode_auto_spot_grid``,
    ``ops_decode_auto_tune_fraction``, ``ops_decode_auto_seed`` and the
    shared tuner's ``ops_decode_auto_ci``, ``ops_decode_auto_objective``,
    ``ops_decode_auto_f_beta`` and ``ops_decode_auto_unreachable``, and
    ``ops_decode_fdr_calibration`` and the shared controls
    ``ops_never_barcode_k``, ``ops_never_barcode_seed`` and
    ``ops_never_barcode_sets``.
:param rule: from :func:`decode_rule`; a grid without the pixel quality,
    or a rule without the pixel fallback, keeps ``pixel_quality``.
Manual never-barcode calibration uses a single setting, preserving the
entered thresholds. Without decoys it requests the reported fallback.

:returns: ``mode``; in auto mode or manual never calibration also ``floors`` and ``pixel_qualities``
    (sorted), ``fdr_max`` (per 100 calls), ``ci``, ``objective``,
    ``f_beta``, ``unreachable``, ``tune_fraction``, ``seed`` and
    ``spot_thresholds`` (the read thresholds tried jointly, sorted; empty
    when the read threshold stays as entered); with
    a calibration method also ``fdr_calibration``.
:raises ValueError: for an unknown mode or FDR calibration, an FDR target that is not
    above 0, a tuning fraction outside (0, 1), a grid that is not a
    mapping of the two thresholds to lists of numbers 0 or above, or
    auto mode without decoys (``ops_decoy_library_k`` 0).

### `spacr.ops_decode.write_well_reports`

`spacr/ops_decode.py:2387`; SHA-256 `b60a898d64c8391a419dfc1b06918647c658d4946bf75b80f17d9619c7d62280`.

```python
db: str, plate: str, well: str, decoded: Sequence[Mapping[str, Any]], rule: Mapping[str, Any], skipped_sites: Sequence[int]
```

Write field statistics, FDR report and plate summary for one well.

:param db: the measurements database.
:param plate: the plate.
:param well: the well.
:param decoded: the decoded fields, each with ``site`` and ``rule_a``
    from :func:`field_calls`.
:param rule: from :func:`decode_rule`; after auto thresholds with a
    shuffle calibration (item 949) its ``fdr_scale`` is the calibrated
    scale and its ``threshold_choice`` holds the calibration.
:param skipped_sites: the unplaced sites.
Never calibration writes its JSON and CSV through the same report
writer used by the prefix path, with point and interval fields folded
into this report and the per-well plate summary.

:returns: the FDR report for one well, with the paths written under ``paths``;
    with a shuffle calibration it adds ``fdr_calibration``,
    ``fdr_scale_entered``, ``shuffle_called`` for the whole well,
    ``shuffle_call_rate_pct``, ``shuffle_ci95``, ``shuffle_limit_pct``
    (1.5 times the calibrated FDR) and ``shuffle_le_limit``.

### `spacr.ops_prefix.prefix_settings`

`spacr/ops_prefix.py:2005`; SHA-256 `28dd5e00157aa92fca09e74dcbe03b4bd841aa1ac30a1a5c27dec2f53968ac17`.

```python
settings: Mapping[str, Any]
```

The item-920 settings, checked and with defaults filled in.

:param settings: the OPS settings: ``ops_library``,
    ``ops_library_sequence_col``, ``ops_library_gene_col``,
    ``ops_prefix_length``, ``ops_sbs_cycles``, ``ops_cycle_mask``,
    ``ops_prefix_collision``, ``ops_prefix_metric``,
    ``ops_prefix_max_distance``, ``ops_prefix_crosstalk``,
    ``ops_quality_min_reads``, ``ops_prefix_quality_method``,
    ``ops_random_barcode_control``, ``ops_control_seed``,
    ``ops_prefix_masks``, ``ops_read_threshold`` and the item-942 decoy
    and tuning settings (``ops_prefix_decoy_k``,
    ``ops_prefix_decoy_seed``, ``ops_prefix_fdr_scale``,
    ``ops_prefix_min_read_quality``, ``ops_prefix_threshold_tuning``,
    ``ops_prefix_fdr_target_pct``, ``ops_prefix_tuning_grid``) and the
    item-949 ``ops_prefix_nearmiss_decoy_k`` and
    ``ops_prefix_fdr_calibration``, the item-967
    ``ops_prefix_minority_min_reads``, the item-982 small-well guard
    ``ops_prefix_tuning_min_sites``, and the item-988 weighted-majority
    settings: ``ops_prefix_distance_weight``,
    ``ops_prefix_distance_decay_px``, ``ops_prefix_distance_interior_px``,
    ``ops_prefix_distance_min_weight``, ``ops_prefix_weighted_quality``,
    ``ops_prefix_weighted_min_share``, ``ops_read_neighbour_discount``,
    ``ops_prefix_nuclei_masks``, ``ops_keep_multi_guide_cells`` and
    ``ops_multi_guide_min_reads``, and shared controls
    ``ops_never_barcode_k``, ``ops_never_barcode_seed`` and
    ``ops_never_barcode_sets``. Never calibration without decoys keeps
    the entered scale and reports the fallback.
:returns: the normalised values under the same keys.
:raises ValueError: without a library, for an unknown choice, for
    tuning or a shuffle calibration without decoys, for a negative
    decoy count, minority-read floor or small-well site count, or for a
    weight setting that is out of range or set without
    ``'weighted_majority'``.

### `spacr.ops_prefix.run_prefix_decode`

`spacr/ops_prefix.py:2585`; SHA-256 `bf083411d43111ab67ce1ac5b9f51128d65adf79c91041ee433656a78a6d290b`.

```python
settings: Mapping[str, Any], report: Optional[Callable[[str], None]]=None
```

The ``prefix_decode`` phase: decode every indexed site against spacer prefixes.

Reads the cropped-site layout (item 919, ``ops_layout='cpg_cropped'``),
builds the whitelist, decodes each complete site's sequencing stack
(cycles 1 to the prefix's last cycle), takes each read's cell from
``ops_prefix_masks`` (or the site's ``merged/`` labels), calls cells and
writes, under ``<dst_root>/results/ops_prefix/``:

* ``library_prefix_collisions.csv`` and ``library_summary.json``;
* ``quality_classes.json`` (the quality method, its class definitions
  and the settings);
* ``reads_<plate>_<well_id>.parquet`` (one row per read) and
  ``cells_<plate>_<well_id>.csv`` (:func:`prefix_cell_calls`);
* ``site_summary.csv`` (per site: :func:`read_fractions`, cells with
  reads, cells with an exact read, cells per class);
* ``cells_per_guide.csv`` and ``cells_per_gene.csv`` (perfect and great
  cells);
* with decoys (``ops_prefix_decoy_k`` above 0, item 942): a ``decoy``
  column in the cell tables (a decoy call keeps its prefix but no
  genotype), ``cells_in_labels``, ``cells_real``, ``cells_decoy`` and
  ``fdr_pct`` in ``site_summary.csv``, and ``fdr_<plate>_<well_id>.json``
  (:func:`spacr.ops_decode.fdr_report`);
* with ``ops_prefix_threshold_tuning`` 'per_well_decoy': per well
  ``threshold_tuning_<plate>_<well_id>.csv`` and ``.png`` (every grid
  setting's tuning numbers and the genotyped-vs-FDR curve,
  :func:`spacr.ops_decode.write_tuning_curve`) and ``.json`` (the choice,
  :func:`spacr.ops_decode.tune_thresholds`); each site's spots are found
  once, down to the lowest read threshold, and the chosen setting's reads
  and cells are written;
* with near-miss decoys (``ops_prefix_nearmiss_decoy_k`` above 0, item
  949, :func:`prefix_nearmiss_decoys`): a ``decoy_class`` column
  (``'far'`` or ``'nearmiss'``) in the cell tables,
  ``cells_decoy_nearmiss`` and ``fdr_nearmiss_pct`` (near-miss calls per
  100 real calls over ``ops_prefix_nearmiss_decoy_k``, scale 1) in
  ``site_summary.csv``, and the near-miss count, FDR and its interval in
  ``fdr_<plate>_<well_id>.json``; ``cells_decoy`` and ``fdr_pct`` keep
  counting the far decoys only;
* with ``ops_prefix_fdr_calibration`` 'shuffle' (item 949): every read is
  also matched against :data:`spacr.ops_decode.SHUFFLE_SEEDS` shuffled
  whitelists (:func:`shuffled_library`, seeds ``ops_control_seed`` and
  up), and each well's FDR scale is raised from that control
  (:func:`spacr.ops_decode.calibrate_fdr_scale`; without tuning at the
  entered setting); ``fdr_pct`` uses the calibrated scale, and
  ``site_summary.csv`` gains ``shuffle_called`` (mean over the shuffled
  whitelists), ``shuffle_call_rate_pct`` and ``fdr_scale``; the FDR
  report gains the entered scale, the calibration, the FDR's interval
  and the shuffle call rate with its interval
  (:func:`spacr.ops_decode.shuffle_call_rate`). With both item-949
  settings off the outputs are those of item 942;
* with ``ops_prefix_minority_min_reads`` 2 or more (item 967, alpha): the
  cell calls and the tuning leave out minority stray reads
  (:func:`prefix_cell_calls`), the cell table gains
  ``minority_ignored`` and ``minority_rescued``, ``site_summary.csv``
  gains ``cells_minority_rescued`` (genotyped cells called only because
  of the rule), and one line per well reports the rescued cells. 0, the
  default, writes what the run wrote before;
* with ``ops_prefix_quality_method`` 'weighted_majority' (item 988,
  alpha): cells are called by their barcodes' summed read weights
  (:func:`prefix_cell_calls`, classes :data:`WEIGHTED_CLASSES`), the
  tuning also tries the weight axes (:data:`WEIGHT_GRID_KEYS`;
  :data:`PREFIX_TUNING_GRID_WEIGHTED` when the grid is empty), and the
  cell table gains the weight columns. With a distance term the read
  table gains ``nuc_gap_px`` and ``nuc_gap_fallback``, measured to the
  nuclei of ``ops_prefix_nuclei_masks``;
* with ``ops_keep_multi_guide_cells`` (item 988, alpha): per well
  ``guide_multiplicity_<plate>_<well_id>.csv`` (the share of cells by
  number of guides, and the share of two-guide cells whose runner-up is
  a touching cell's top guide) and ``multi_guide_cells_<plate>_<well_id>.csv``
  (the cells with two or more guides of ``ops_multi_guide_min_reads``
  reads, none of them from a neighbour). With both item-988 settings
  off the run writes what it wrote before.
* with ``ops_prefix_tuning_min_sites`` above 0 (item 982 small-well
  guard, alpha): a tuned well with fewer sites keeps the read-quality
  floor at :data:`SMALL_WELL_FLOOR` and tunes only the other settings;
  the log says so, and the well's tuning json and ``tuning`` entry gain
  ``small_well_floor`` and ``small_well_sites``. 0, the default, writes
  what the run wrote before.

Never-barcode calibration (item 1014) adds independent distance-one
and distance-two control passes through the same prefix matcher and
cell votes. Their calls never replace production calls. The shared
estimator calibrates on tuning sites and reports held-out sites;
fixed thresholds use a one-setting grid. The shuffle call rate is
reported alongside never calibration and enters its scale only with
``'never+shuffle'``. Each well writes ``fdr_calibration_<well>.json``.

:param settings: the OPS settings: ``dst_root``, the ``cpg_*`` reader
    settings, the item-920, item-942, item-949, item-967 and item-988
    settings
    (:func:`prefix_settings`)
    and, for tuning, the shared tuner's settings
    (``ops_decode_auto_objective``, ``ops_decode_auto_f_beta``,
    ``ops_decode_auto_unreachable``, ``ops_decode_auto_ci``,
    ``ops_decode_auto_tune_fraction`` and ``ops_decode_auto_seed``).
:param report: called with progress lines.
:returns: the output paths, the library summary and the run's totals;
    with any item-942 setting on also ``seconds_per_well``, ``fdr`` (per
    well) and ``tuning`` (per well: the choice and its numbers); with
    the shuffle calibration also ``calibration`` (per well); with the
    minority-read rule (item 967) also ``minority_rescued`` (genotyped
    cells rescued, per well); with ``ops_keep_multi_guide_cells`` (item
    988) also ``multi_guide`` (per well: the two table paths and the
    number of multi-guide cells); with tuning also ``well_choices``, one
    line per well (item 966): the read mode and thresholds chosen and
    the decoy FDR on the tuning and the report sites, also sent to
    ``report`` last. An empty ``ops_read_mode`` is
    ``'auto'`` with tuning and ``'log'`` without.
:raises ValueError: without ``dst_root`` or a library, when the layout
    has fewer sequencing cycles than the prefix needs, for tuning,
    near-miss decoys or a shuffle calibration without decoys, or for a
    distance term or the touching cells without the labels they need.

### `spacr.ops_sbs.never_barcodes`

`spacr/ops_sbs.py:923`; SHA-256 `c2264601fd61bdd443276f0189f22fc806516bb5ec8b98aa844676b11bfd099a`.

```python
library, *, distance, k=4, seed=1, exclude=()
```

Substitution-built barcodes that no cell in the guide library should carry.

Distance-one controls have exactly one nearest guide and use a different
substituted position for each accepted control of that guide.
Distance-two controls have no guide at distance zero or one. Every
control lies at least two substitutions from previously accepted and
excluded controls. Sorted unique guides are visited round by round,
with one seeded generator and up to 200 attempts per guide and round.
The same builder serves full-barcode and sequenced-prefix calibration.

:param library: real guide barcodes, using nonempty A/C/G/T strings.
    Duplicates and input order do not affect the result.
:param distance: nearest-guide Hamming distance, 1 (N1) or 2 (N2).
:param k: maximum controls per guide, default 4; zero gives an empty set.
:param seed: nonnegative generator seed, default 1.
:param exclude: controls already taken, such as matched decoys or N1.
:returns: dict with ``barcodes`` in acceptance order, sorted ``guides``,
    ``distance``, ``k``, ``seed``, and read-only int64 ``m_d`` and
    ``library_m_d`` arrays. Both arrays have shape ``(guides, 3)``;
    columns count control or other-library barcodes at distances 1–3.
    Exact matches are excluded. Results are memoised per process.
:raises ValueError: for an unsupported distance, negative or fractional
    k or seed, or a barcode containing letters outside A/C/G/T.

## Runtime sources

36 canonical source identities; all admitted by the actual source extractor.

`setting_labels/ops.ops_never_barcode_k`; SHA-256 `73f5076abc5acf087f424fdf8444187a65cc82a8b7cd1a0cf88635e73dc1ddb9`.

`spacr/ops_settings.py:252`.

Never barcode k

`setting_labels/ops.ops_never_barcode_seed`; SHA-256 `3c951ba4f4c9835ffe918e6345a8aed88d7c6761f217a05e6aa9bf1b051404c0`.

`spacr/ops_settings.py:253`.

Never barcode seed

`setting_labels/ops.ops_never_barcode_sets`; SHA-256 `29612fe60787feb5a019b7ec9bc97e832bcc7b970488ae811eaa23de498a1149`.

`spacr/ops_settings.py:254`.

Never barcode sets

`setting_labels/ops_never_barcode_k`; SHA-256 `3f75c17cffab705943434767558b77f10378cd46d024d25b5bec95e449b92917`.

`spacr/ops_settings.py:252`.

Ops never barcode k

`setting_labels/ops_never_barcode_seed`; SHA-256 `c345f8f8571b4bb6072becd2e302f4d4d067f096784225d70edae5a592691a17`.

`spacr/ops_settings.py:253`.

Ops never barcode seed

`setting_labels/ops_never_barcode_sets`; SHA-256 `aedecfba3e36e9a4d40114154d8c8950a1762e25f1950211dd9e2f3e1b6d5a0b`.

`spacr/ops_settings.py:254`.

Ops never barcode sets

`setting_tooltips/ops_never_barcode_k`; SHA-256 `d517b94646b3630a6be9ff59e3dba43d2612729461e6ac1f8ce4fa69d95da8bf`.

`spacr/ops_settings.py:3317`.

Maximum substitution-built controls per guide in each never-barcode set, shared by full-barcode and spacer-prefix calibration. Fewer may be accepted when sequence constraints leave no candidate; 0 leaves calibration without control evidence, so it keeps the entered scale. Default 4. Alpha.

`setting_tooltips/ops_never_barcode_seed`; SHA-256 `4a6029ae272a6575c8185945bf775f37304414856951a1effe60b23fd825b71b`.

`spacr/ops_settings.py:3319`.

Nonnegative random seed for never-barcode substitution controls in both decodes. The same library and seed reproduce the same independent control sets. Default 1. Alpha.

`setting_tooltips/ops_never_barcode_sets`; SHA-256 `a6d3ca155b31c772fa8e04bcc9e806ad15730966e95d7642eb0ef6d4dd340da5`.

`spacr/ops_settings.py:3321`.

Never-barcode passes in both decodes: 'n1_n2' measures distance-one and distance-two controls; 'n2' skips the distance-one report pass. Distance-two controls set the scale. A library with distance-one guide pairs needs N1 evidence and otherwise keeps the entered scale. Default 'n1_n2'. Alpha.

`setting_tooltips/ops_decode_fdr_calibration`; SHA-256 `423c1fe480745334e0ea3bbabf84a0346d1e87921cc060afc0cfab7789732c81`.

`spacr/ops_settings.py:3315`.

How rule (a) sets the FDR scale. 'never' measures error calls on substitution-built barcodes through the same read and pixel votes: distance-two controls set the scale and distance-one controls report a check. It can raise or lower the entered scale, down to 1. Manual thresholds stay fixed while calibration runs. 'none' keeps ops_fdr_scale; 'shuffle' only raises it from three shuffled libraries; 'never+shuffle' takes the larger control-based scale. Never methods need decoys and keep the entered scale when evidence is insufficient. Default 'never'.

`setting_tooltips/ops_fdr_scale`; SHA-256 `2435892e698c8dea52654a6f0920c3f812ffdf777d0c45b728988074bc1b44e7`.

`spacr/ops_settings.py:3249`.

Entered multiplier for the decoy FDR after accounting for decoys per guide. This is the starting scale and fallback when calibration has too little evidence; 'none' keeps it fixed. Values below 1 count as 1. The Funk library reference value is 1.268. Default 1.268.

`setting_tooltips/ops_prefix_fdr_calibration`; SHA-256 `ceda01d878fb8486cc7ac98d8c5a5afbecdbcad139a5ade54578aabcf81b50d8`.

`spacr/ops_settings.py:3907`.

Set the spacer-prefix FDR scale. 'never' measures substitution-built controls through the production matcher and votes; distance-two controls set the scale and distance-one controls provide a check. Fixed thresholds stay fixed. 'none' keeps the earlier uncalibrated tables and reports no shuffles. 'shuffle' only raises the scale; 'never+shuffle' takes the larger control-based scale. Never and shuffle methods report shuffle rates. Without decoys, never methods keep the entered scale. Default 'never'.

`setting_tooltips/ops_prefix_fdr_scale`; SHA-256 `dde1d682b1c800095640a6aa6ab14334d83609f8cf9c2cbcd0b1f1e587f52ce7`.

`spacr/ops_settings.py:3873`.

Entered multiplier for the decoy FDR after accounting for decoys per prefix. This is the starting scale and fallback when calibration has too little evidence; 'none' keeps it fixed. Values below 1 count as 1. Default 1. Alpha.

`ui/Conservative scale retained after a threshold cycle.`; SHA-256 `12df6f6af035253b6ba093d547db977ec4bdf5ef3c4c2c66b6e479b67c003b4f`.

`spacr/ops_decode.py:2098`.

Conservative scale retained after a threshold cycle.

`ui/Distance-one library neighbours require N1 calls and opportunities.`; SHA-256 `ac9f3bdab698864ec60ebf6c687f1ee79e1900a68947799e551444c9147897a1`.

`spacr/ops_decode.py:1777`.

Distance-one library neighbours require N1 calls and opportunities.

`ui/FDR calibration needs at least one tuning round.`; SHA-256 `b953fb7008b3e36fcb5fad271d6dd8b301704727e0916b1bb678ab60aeea0434`.

`spacr/ops_decode.py:1861`.

FDR calibration needs at least one tuning round.

`ui/Never-barcode calibration needs a positive decoy multiplicity.`; SHA-256 `6fba7722c50b93603fab54f680f4f7bc99ba19fecc41bc4e5b4166a7a8d045da`.

`spacr/ops_decode.py:1775`.

Never-barcode calibration needs a positive decoy multiplicity.

`ui/Never-barcode calibration needs aligned, finite, nonnegative field counts.`; SHA-256 `5061650fdaa39fecd677e705e8d899939faa65904bf35e60eb039e578c733928`.

`spacr/ops_decode.py:1773`.

Never-barcode calibration needs aligned, finite, nonnegative field counts.

`ui/Never-barcode calibration needs both N1 calls and N1 opportunities.`; SHA-256 `84c012a99d6da2c353614044423ebc277d143bc952a88d1373eb3b140c5b40ed`.

`spacr/ops_decode.py:1764`.

Never-barcode calibration needs both N1 calls and N1 opportunities.

`ui/Never-barcode calibration needs complete control arrays matching the field grid.`; SHA-256 `8beda319fad487fb70803b77bacca58c472b3b930283a670404efeae9c79ef6b`.

`spacr/ops_decode.py:1875`.

Never-barcode calibration needs complete control arrays matching the field grid.

`ui/Never-barcode count must be a nonnegative integer.`; SHA-256 `12dbe810a2ba94c4893ec7228cac5787d7f8d697782eec464a49f2175b64cc1b`.

`spacr/ops_sbs.py:953`.

Never-barcode count must be a nonnegative integer.

`ui/Never-barcode distance must be 1 or 2.`; SHA-256 `7e70d0f26c43515ed276e9e80f303fffcfaa4d3b92d72eba028ebbb186b611d8`.

`spacr/ops_sbs.py:951`.

Never-barcode distance must be 1 or 2.

`ui/Never-barcode libraries must contain nonempty DNA strings using A, C, G and T.`; SHA-256 `2005dbef3e6df7445f592799f788685293f6b4d36846b5c1bd78318d1e144427`.

`spacr/ops_sbs.py:959`.

Never-barcode libraries must contain nonempty DNA strings using A, C, G and T.

`ui/Never-barcode seed must be a nonnegative integer.`; SHA-256 `bed0b1f84c48b4c972f1b8d0518db101b2eaa380b3c6afaafcdb3122408ac425`.

`spacr/ops_sbs.py:955`.

Never-barcode seed must be a nonnegative integer.

`ui/Opportunity profiles do not contain every called guide.`; SHA-256 `129871ecbdd0368dc40a035eefcfaa4d8a2c160aa7fc2481c1e0caec4df15a55`.

`spacr/ops_decode.py:985`.

Opportunity profiles do not contain every called guide.

`ui/Opportunity profiles must align with their sorted guide library.`; SHA-256 `862262df0037588abd223e71e08fa6ef803895c9ddfd7f5d108cedba819f6010`.

`spacr/ops_decode.py:967`.

Opportunity profiles must align with their sorted guide library.

`ui/Unknown FDR calibration method: {method}.`; SHA-256 `87e4bbe6a94a8a0af3f16028c471662287f388b45d005843a0e7f0df3d83016e`.

`spacr/ops_decode.py:1859`.

Unknown FDR calibration method: {method}.

`ui/not estimated`; SHA-256 `3581b54566f17b420a1da48b35026d0696168e8061ac8cb486c35328825399b6`.

`spacr/ops_decode.py:2089`.

not estimated

`ui/ops_never_barcode_sets must be n1_n2 or n2.`; SHA-256 `49dd99cc8a54b5300cbba47eb81a5f9ae0a20d85831cad03d0911fc6eef2874c`.

`spacr/ops_decode.py:1170`.

ops_never_barcode_sets must be n1_n2 or n2.

`ui/{key} must be a nonnegative integer.`; SHA-256 `f8a6880537ef48db8c606dfd2a05d8de6260c5e0fa4866b51347c8fb54e76e68`.

`spacr/ops_decode.py:1166`.

{key} must be a nonnegative integer.

`ui/{well}: FDR scale calibrated from never-should-appear barcodes: {entered} entered -> {scale} [{interval}]; FDR at the chosen setting {fdr} % (entered scale {entered_fdr} %).`; SHA-256 `755c6736af51814f3ae1379b48b657360bdaab23241f643d84f90a35f634e047`.

`spacr/ops_decode.py:2091`.

{well}: FDR scale calibrated from never-should-appear barcodes: {entered} entered -> {scale} [{interval}]; FDR at the chosen setting {fdr} % (entered scale {entered_fdr} %).

`ui/{well}: FDR scale kept at {scale}; control opportunities are unavailable at the chosen setting.`; SHA-256 `ffbff98941a32e9ec100d5e1d562bee283929253c4fd2717c072d860d1a9f0bc`.

`spacr/ops_decode.py:2084`.

{well}: FDR scale kept at {scale}; control opportunities are unavailable at the chosen setting.

`ui/{well}: FDR scale kept at {scale}; fewer than 30 tuning-half control calls.`; SHA-256 `5c29510c974d98d11563bb7de273769c0b3c8d395a7e8b825a93cf0a9e90f598`.

`spacr/ops_decode.py:2082`.

{well}: FDR scale kept at {scale}; fewer than 30 tuning-half control calls.

`ui/{well}: FDR scale kept at {scale}; never-barcode calibration needs a positive decoy multiplicity.`; SHA-256 `6b52b27de445d52ad56daff0661ba2cd011e4890caf3e389d1b5b58daebacbe9`.

`spacr/ops_decode.py:2086`.

{well}: FDR scale kept at {scale}; never-barcode calibration needs a positive decoy multiplicity.

`ui/{well}: FDR scale kept at {scale}; the library has fewer than 50 guides.`; SHA-256 `5d8897c92de04090bd28931b023509fae7f66b1eb6a72e67b39e5f3eb4bf0ece`.

`spacr/ops_decode.py:2080`.

{well}: FDR scale kept at {scale}; the library has fewer than 50 guides.

`ui/{well}: N2 control voting changed real calls by {change} %, above the 0.5 % perturbation guard; production calls were kept.`; SHA-256 `13faf1217cd78f7be56be10aaad3b2df8d7c45664e10f8fb5f48ce532e1f3051`.

`spacr/ops_decode.py:2103`.

{well}: N2 control voting changed real calls by {change} %, above the 0.5 % perturbation guard; production calls were kept.

## Other documentation

The JSON also inventories 105 changed private, tool and test docstrings. They do not create public API or runtime translation identities.

## Validation

Final shared implementation: 141 tests passed with CUDA hidden (18.80 s) and 141 with CUDA visibility unset and 2 GB VRAM reserved (17.44 s). Final declaration/tooltip corrections and actual alpha-off prefix execution: 79 passed hidden (8.27 s), 79 passed visible (8.13 s). Final reported-scale consumer files: 64 passed hidden (6.47 s), 64 passed visible (5.58 s). Final item regression file: 76 passed hidden (7.66 s), 76 passed visible (6.23 s). The schema-956 CLI comparison against exact assigned-base code produced nine byte-identical prefix artifacts. Three protected source-only API gates passed against all eleven final bodies and added zero production imports (21.14 s, 27.97 s, 40.44 s). All six required source-rule files passed: 157 tests (211.69 s). Full items, OPS, settings and ratchet receipts are recorded in WORK_STATUS and the final commit body; raw failures remain visible.

Final metadata/source-owner cohort: 164 passed, one expected failure at the unchanged settings-flow ceiling (13,714 xrefs versus old limit 9,700). All five artifacts reproduce byte-identically from final source. Source admission support, API consumer hooks and alpha arrivals were approved by the spec owner. The settings-flow ceiling on this old base remains unchanged; current pilot has a measured larger ceiling. Home resolves source-overlap and alpha totals at merge.
