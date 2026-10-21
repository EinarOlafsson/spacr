# Analyze regression results

**Purpose.** Derive downstream guide correlations, ranked findings, and figures from a completed regression analysis.

**Recommended use.** Use after model fitting when additional result summaries are required.

**Primary outputs.** Ranked result tables and gene- or guide-level diagnostic figures.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.post_regression_analysis`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.post_regression_analysis)

```python
post_regression_analysis(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.post_regression_analysis`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.post_regression_analysis)


#### Essential inputs

- **`csv_file`** *(required)* — CSV with columns ``grna``, ``fraction`` and ``prc``.
- **`grna_dict`** *(required)* — mapping of anchor ``grna`` names to their fixed effect sizes.
- **`grna_list`** *(required)* — gRNAs to include in the correlation matrix.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`save`** *(optional)* — persist correlation matrix, effect sizes and plots. Default ``False``. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Ranked result tables and gene- or guide-level diagnostic figures.
