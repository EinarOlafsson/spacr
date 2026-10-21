# Estimate percent positive

**Purpose.** Classify objects relative to an intensity threshold and estimate the positive fraction for each experimental unit.

**Recommended use.** Use for binary marker-expression phenotypes with an interpretable measurement threshold.

**Primary outputs.** Per-object threshold assignments and per-well positive fractions.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.analyze_percent_positive`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_percent_positive)

```python
analyze_percent_positive(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.analyze_percent_positive`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_percent_positive)


#### Essential inputs

- **`src`** *(optional)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Measurements (1 settings)</summary>

#### Measurements

- **`tables`** *(optional)* — (list) - Measurement tables read from each plate's database and merged into one analysis frame. Supported tables are registered object roles, including organelles, and png_list. For tabular Classify, choose a nonempty list of distinct supported tables with at least one object-role table. Default ['cell', 'nucleus', 'pathogen', 'cytoplasm']. Default: `['cell']`.

</details>

<details>
<summary>Additional settings (3 settings)</summary>

#### Additional settings

- **`filter_1`** *(optional)* — (list or None) - Two-element [column, minimum] filter applied before counting. Rows whose column value is not strictly greater than minimum are excluded. Use it to remove objects that are too small or dim to score so they do not affect the reported percentage. This strict &gt; filter has one lower bound and no upper bound. Default None, which disables pre-filtering. Default: `['cell_area', 1000]`.
- **`threshold`** *(optional)* — (list, list-of-lists, int or None) - Cut-off applied to 'measurement' before the annotation grid loads, so you only label the objects you care about. Accepts a number or a quantile code 'q1'-'q9' (q3 = the 30th percentile of that column), or one entry per measurement when measurement is a list. Empty or None loads every object unfiltered. Default 2000 where a numeric cutoff is used; empty where the setting is optional. Default: `2000`.
- **`value_col`** *(optional)* — (str) - Measurement column compared with threshold to classify each object as positive. Objects strictly above the threshold are annotated 'above'; the remainder are annotated 'below', and the reported percentage per well is the fraction above. Select the column representing the phenotype, such as a recruitment ratio or mean intensity. Results from different value_col settings are not directly comparable. Default None; must be supplied. Default: `'cell_channel_2_mean_intensity'`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-object threshold assignments and per-well positive fractions.
