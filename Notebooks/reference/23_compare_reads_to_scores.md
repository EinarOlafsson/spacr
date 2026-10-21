# Compare sequencing reads with phenotype scores

**Purpose.** Join per-well guide abundance with imaging-derived phenotype scores and quantify their association.

**Recommended use.** Use as a plate-registration and data-consistency check before regression analysis.

**Primary outputs.** Joined per-well data and correlation figures.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.compare_reads_to_scores`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.compare_reads_to_scores)

```python
compare_reads_to_scores(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.compare_reads_to_scores`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.compare_reads_to_scores)


#### Essential inputs

- **`reads_csv`** *(required)* — path (or list of paths) to per-gRNA read count CSVs.
- **`scores_csv`** *(required)* — path (or list of paths) to per-object classifier score CSVs.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`plate`** *(optional)* — plate ID to stamp when a single pair of CSVs is given. Default: `None`.

</details>

<details>
<summary>Additional settings (7 settings)</summary>

#### Additional settings

- **`column`** *(optional)* — column used to select a subset of wells. Default ``'columnID'``. Default: `'columnID'`.
- **`empirical_dict`** *(optional)* — mapping of ``rowID`` to ``(pc_units, nc_units)`` mixture; a 16-row default is used when ``None``. Default: `None`.
- **`nc_grna`** *(optional)* — negative-control gRNA name. Default ``'TGGT1_233460_4'``. Default: `'TGGT1_233460_4'`.
- **`pc_grna`** *(optional)* — positive-control gRNA name. Default ``'TGGT1_220950_1'``. Default: `'TGGT1_220950_1'`.
- **`save_paths`** *(optional)* — two-element list of PDF output paths (pc plot, nc plot). Default: `None`.
- **`value`** *(optional)* — value in ``column`` to keep. Default ``'c3'``. Default: `'c3'`.
- **`y_columns`** *(optional)* — Columns to plot on the y axis. ``None`` uses ``['class_1_fraction', 'TGGT1_220950_1_fraction', 'nc_fraction']``. Default: `None`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Joined per-well data and correlation figures.
