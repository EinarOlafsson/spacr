# Evaluate sequencing quality

**Purpose.** Summarize sequencing depth, read quality, consensus, and guide-count distributions.

**Recommended use.** Use after barcode mapping and before downstream association analysis.

**Primary outputs.** Sequencing quality-control figures and threshold summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.sequencing_qc.barcode_qc`](https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html#spacr.sequencing_qc.barcode_qc)

```python
barcode_qc(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.sequencing_qc.barcode_qc`](https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html#spacr.sequencing_qc.barcode_qc)


#### Essential inputs

- **`count_data`** *(required)* — (str or list) - CSV(s) of per-well gRNA read counts from the sequencing step (unique_combinations.csv); each must contain grna, count, rowID and columnID columns or the run raises ValueError. These are the regression's independent variable. Pass one path per plate, position-aligned with plates_count; results are written under the first file's folder. Default 'list of paths', a placeholder that must be replaced; the barcode QC module defaults this key to 'path to unique_combinations.csv'.
- **`target_grnas_per_well`** *(required)* — (int) - How many gRNAs a well is meant to carry. This is the biological target that replaces picking an abundance cutoff by eye: spaCR solves for the read-fraction threshold that delivers it in THIS run's data and prints the number it derived, then sweeps around it so the trade-off is visible. Raise it for statistical power (more guides per well, more wells kept), lower it for attributability (a phenotype traceable to fewer guides). A well holding more than this is counted as a collision. Default 5.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Reference &amp; Count Tables (4 settings)</summary>

#### Reference & Count Tables

- **`grna_csv`** *(optional)* — (path) - CSV mapping gRNA barcode sequences to gRNA names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so orientation must match the reads (barecodes_reverse_complement flips a file). Rows whose gRNA does not match are written as NA and dropped from the counts. Default: the bundled spacr/resources/data/grna_barcodes.csv. Default: `''`.
- **`row_csv`** *(optional)* — (path) - CSV mapping row barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim with no reverse-complementing, so the sequences must be in the same orientation as the reads - use barecodes_reverse_complement to flip the file if needed. Unmatched reads get NA for rowID. Default: the bundled spacr/resources/data/barcodes_row.csv. Default: `''`.
- **`column_csv`** *(optional)* — (path) - CSV mapping column barcodes to well names; it must have 'sequence' and 'name' columns. Reads are matched verbatim against it with no reverse-complementing, so the sequences must be in the same orientation as the reads - run barecodes_reverse_complement on the file if they are not. Unmatched reads get NA for columnID. Default the bundled spacr/resources/data/barcodes_column.csv; barcode QC (sequencing_qc) instead defaults this key to empty, where the reference is optional. Default: `''`.
- **`qc_data`** *(optional)* — (str or list) - Path(s) to the qc.csv a barcode-mapping run wrote beside its count table. Supplies the unmapped-read panel: how many reads reached barcode lookup and how many of them matched no entry in each reference. Leave empty to skip that panel; every other panel works from count_data alone. Default ''. Default: `''`.

</details>

<details>
<summary>Well Expectations (2 settings)</summary>

#### Well Expectations

- **`target_statistic`** *(optional)* — (str) - Whether target_grnas_per_well is a 'median' or a 'mean' over wells. Median is the default because a handful of wells that soaked up the whole library drag a mean far off the typical well. Default 'median'. Default: `'median'`.
- **`min_reads_per_well`** *(optional)* — (int) - Absolute read floor below which a well is called starved and left out of the threshold fit. 0 derives one from the run instead, as starved_read_fraction of the median well's depth. Starved wells are always reported either way. Default 0. Default: `0`.

</details>

<details>
<summary>Starvation &amp; Exclusion (2 settings)</summary>

#### Starvation & Exclusion

- **`starved_read_fraction`** *(optional)* — (float) - Share of the median well's read total used as the starvation cut when min_reads_per_well is 0. A well at a tenth of typical depth turns single stray reads into 10% abundances, which is why 0.1 is the default. Ignored when min_reads_per_well is set. Default: `0.1`.
- **`exclude_starved_wells`** *(optional)* — (bool) - Leave starved wells out of the population the threshold is derived and swept over. They report one gRNA at any cutoff and pull the target down onto a threshold the healthy wells never needed. They stay in the QC panels regardless. Default True. Default: `True`.

</details>

<details>
<summary>Position &amp; Collision Checks (2 settings)</summary>

#### Position & Collision Checks

- **`position_effect_ratio`** *(optional)* — (float) - Fold-change from its plate's median read depth at which a plate row or column is flagged as a position effect. 2.0 flags a row at half or double the plate. Must be above 1. Default 2.0. Default: `2.0`.
- **`collision_max_distance`** *(optional)* — (int) - How many substituted bases still count as a barcode collision. 1 catches the pairs a single miscalled base can turn into each other, which is the common event; 0 reports only exact duplicates. Above 1 the search is pairwise and slow on a full gRNA library. Default 1. Default: `1`.

</details>

<details>
<summary>Threshold Sweep (2 settings)</summary>

#### Threshold Sweep

- **`sweep_span`** *(optional)* — (float) - How far either side of the derived threshold the sweep runs, as a multiplicative factor. 4.0 sweeps a quarter to four times the derived value. Must be above 1. Default 4.0. Default: `4.0`.
- **`sweep_points`** *(optional)* — (int) - Log-spaced points on the sweep, before the derived threshold is added to them. Must be at least 3. Default 25. Default: `25`.

</details>

<details>
<summary>QC Output (3 settings)</summary>

#### QC Output

- **`dst`** *(optional)* — (str) - Folder receiving versioned tables, manifests and figures. Default '' uses a module-specific folder beside the primary input, keeping different analyses separated. Default: `''`.
- **`plot`** *(optional)* — (bool) - Render and save quality-control figures during the pipeline, including channel montages, Cellpose mask overlays, filtration comparisons, and crop grids. Figure generation increases runtime and memory use, particularly for complete plates. test_mode enables this setting automatically. Default False. Merged Classifier and Recruitment both start with plotting enabled so their diagnostic figures are produced on the first run. Default: `True`.
- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `True`.

</details>

<details>
<summary>Runtime &amp; Reliability (1 settings)</summary>

#### Runtime & Reliability

- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Sequencing quality-control figures and threshold summaries.
