# Analyze gene-ontology enrichment

**Purpose.** Test whether a selected gene set is enriched for annotated gene ontology categories.

**Recommended use.** Use after defining a statistically supported gene set and an appropriate background population.

**Primary outputs.** Enrichment statistics and category-level visualizations.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.toxo.go_term_enrichment_by_column`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.go_term_enrichment_by_column)

```python
go_term_enrichment_by_column(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.toxo.go_term_enrichment_by_column`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.go_term_enrichment_by_column)


#### Essential inputs

- **`significant_df`** *(required)* — DataFrame of screen hits with a ``n_gene`` column.
- **`metadata_path`** *(required)* — CSV path holding ``Gene ID`` plus GO-term columns.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Additional settings (1 settings)</summary>

#### Additional settings

- **`go_term_columns`** *(optional)* — Columns to test. Defaults to the four standard Computed/Curated GO categories. Default: `None`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Enrichment statistics and category-level visualizations.
