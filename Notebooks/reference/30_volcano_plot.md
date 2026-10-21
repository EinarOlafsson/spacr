# Generate a volcano plot

**Purpose.** Display effect estimates against negative log10-transformed p-values and annotate selected findings.

**Recommended use.** Use as a compact summary of a completed differential or regression analysis.

**Primary outputs.** A volcano plot saved with the configured dimensions and file format.

**Desktop route.** Regression → Volcano Explorer

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.toxo.custom_volcano_plot`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.custom_volcano_plot)

```python
custom_volcano_plot(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.toxo.custom_volcano_plot`](https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html#spacr.toxo.custom_volcano_plot)


#### Essential inputs

- **`data_path`** *(required)* — Regression table containing ``feature``, ``coefficient``, and ``p_value`` columns. DataFrame input is copied.
- **`metadata_path`** *(required)* — Gene metadata containing one row per ``gene_nr`` and the selected metadata column. DataFrame input is copied.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`save_path`** *(optional)* — Destination passed to :func:`spacr.figures.scene.write_figure`, which draws the scene the screen would show and falls back to :func:`spacr.plot.save_figure`. The written extension follows the configured figure format either way. Default: `None`.

</details>

<details>
<summary>Additional settings (8 settings)</summary>

#### Additional settings

- **`draw`** *(optional)* — Build, optionally save, and show the figure. If false, return the hit list before constructing a figure. Default: `True`.
- **`figsize`** *(optional)* — Width and height of the square figure in inches. Typography scales with this value. Default: `20`.
- **`highlight_location`** *(optional)* — Values from ``metadata_column`` to overlay in the highlight colour and name in the in-panel legend. Default: `None`.
- **`metadata_column`** *(optional)* — Localization or annotation column used by ``highlight_location``. Default: `'tagm_location'`.
- **`point_size`** *(optional)* — Marker area passed to ``Axes.scatter``. Default: `50`.
- **`threshold`** *(optional)* — Absolute coefficient threshold for calls. A row is returned when ``p_value &lt;= 0.05`` and ``abs(coefficient) &gt;= abs(threshold)``. Default: `0`.
- **`x_lim`** *(optional)* — Two x-axis limits. The default is ``[-0.5, 0.5]``. Default: `None`.
- **`y_lims`** *(optional)* — Use ``[low, high]`` for one axis or ``[[lower_low, lower_high], [upper_low, upper_high]]`` for a broken y-axis. By default, fit one axis to the finite values. Default: `None`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A volcano plot saved with the configured dimensions and file format.
