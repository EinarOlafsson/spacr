# Plot gene phenotypes

**Purpose.** Display selected gene-level phenotype estimates relative to the screen-wide distribution.

**Recommended use.** Use to contextualize specific genes after association analysis.

**Primary outputs.** Gene-level phenotype figures with uncertainty when available.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.toxo.plot_gene_phenotypes`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.plot_gene_phenotypes)

```python
plot_gene_phenotypes(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.toxo.plot_gene_phenotypes`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.plot_gene_phenotypes)


#### Essential inputs

- **`data`** *(required)* — Gene identifiers and phenotype/error columns. The frame is copied before numeric conversion.
- **`gene_list`** *(required)* — Gene names or ``TGGT1_&lt;id&gt;`` identifiers to highlight.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`save_path`** *(optional)* — Figure destination. The configured figure format controls the final extension. Default: `None`.

</details>

<details>
<summary>Additional settings (3 settings)</summary>

#### Additional settings

- **`data_column`** *(optional)* — Mean phenotype column plotted on the y-axis. Default: `'T.gondii GT1 CRISPR Phenotype - Mean Phenotype'`.
- **`error_column`** *(optional)* — Standard-error column used for the uncertainty band. Default: `'T.gondii GT1 CRISPR Phenotype - Standard Error'`.
- **`x_column`** *(optional)* — Column used to match gene identifiers. Default: `'Gene ID'`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Gene-level phenotype figures with uncertainty when available.
