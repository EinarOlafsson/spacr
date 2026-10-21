# Generate score heatmaps

**Purpose.** Arrange per-well phenotype scores in microplate coordinates and compare score tables when multiple inputs are supplied.

**Recommended use.** Use to assess spatial artifacts, gradients, and plate-position effects.

**Primary outputs.** Microplate heatmaps and score-comparison summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.generate_score_heatmap`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.generate_score_heatmap)

```python
generate_score_heatmap(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.generate_score_heatmap`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.generate_score_heatmap)

> This function does not yet expose a settings factory. The list below is recovered from direct settings access in its source, so a key used only on a dynamic branch may be absent.


#### Essential inputs

- **`folders`** *(optional)* — (list) - One folder per plate holding the classification CSVs to combine into a single heatmap; each is searched for csv_name. The order given is the order the plates are laid out in. Default None.
- **`csv`** *(optional)* — (str, path) - Sequencing-derived CSV of gRNA counts per well, used to compute the mixed-condition fraction the scores are compared against. Distinct from csv_name, which names the per-plate score files; this one is a single file for the whole comparison. Default None.
- **`cv_csv`** *(optional)* — (str, path) - CSV of cross-validated scores, added to the heatmap as its own row so a held-out score can be read directly beside the in-sample ones. Leave it None to plot the score rows alone. Default None.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Plate Layout &amp; Controls (1 settings)</summary>

#### Plate Layout & Controls

- **`plateID`** *(optional)* — (str) - Plate name stamped onto count and score rows that carry no plate of their own, and used as the first field of the plate_row_column key that joins the two tables. It is ignored with a warning when the input already contains more than one distinct plate, so it matters only for single-plate inputs. Default 'plate1'. Default: `None`.

</details>

<details>
<summary>Plot (1 settings)</summary>

#### Plot

- **`cmap`** *(optional)* — (str) - Matplotlib colormap applied to single-channel image previews and plate heatmaps. Perceptually uniform maps ('viridis', 'inferno', 'magma') preserve the relative visibility of intensity differences; 'gray' resembles the raw single-channel microscope image. Any registered matplotlib name is accepted, with an '_r' suffix to reverse it. Default 'inferno' for image plots and 'viridis' for plate heatmaps. Default: `None`.

</details>

<details>
<summary>Additional settings (7 settings)</summary>

#### Additional settings

- **`columnID`** *(optional)* — (str) - Plate-map field identifying the plate column, the 1-24 axis of a well name. It is used to group wells and define the heatmap x-axis. Selecting a different field produces an incorrectly defined axis without raising an error. Default None. Default: `None`.
- **`control_sgrnas`** *(optional)* — (list) - gRNA names treated as controls when computing the mixed-condition fraction; every fraction is measured relative to these controls. An incorrect or incomplete list shifts every fraction on the plate in the same direction without raising an exception. Verify the list against the screening library. Default None. Default: `None`.
- **`csv_name`** *(optional)* — (str) - Filename searched for within each entry of folders. All plates must use the same name. A plate whose file has a different name is omitted from the heatmap. Default None. Default: `None`.
- **`data_column`** *(optional)* — (str) - Column in each per-plate score CSV that provides the heatmap values. It must exist in every file named by csv_name; a plate missing this column is omitted. Selecting an incorrect column produces an unintended visualization without necessarily raising an error. Default None. Default: `None`.
- **`data_column_cv`** *(optional)* — (str) - Column in cv_csv holding the cross-validated value. Named separately from data_column because a cross-validation file almost always labels its column differently from the score files it came from. Default None. Default: `None`.
- **`dst`** *(optional)* — (str) - Folder receiving versioned tables, manifests and figures. Default '' uses a module-specific folder beside the primary input, keeping different analyses separated. Default: `None`.
- **`fraction_grna`** *(optional)* — (str) - The single gRNA whose fraction is kept and plotted, selected out of the fraction table once it has been computed. Everything else in that table is discarded, so this picks which gRNA the comparison is about. Default None. Default: `None`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Microplate heatmaps and score-comparison summaries.
