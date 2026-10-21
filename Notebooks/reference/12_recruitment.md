# Quantify molecular recruitment

**Purpose.** Quantify fluorescent-marker enrichment at a pathogen or vacuole relative to the surrounding cellular compartment.

**Recommended use.** Use for host-pathogen experiments in which spatial redistribution of a marker is the principal phenotype.

**Primary outputs.** Per-object recruitment measurements and condition-level summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.analyze_recruitment`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment)

```python
analyze_recruitment(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.analyze_recruitment`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`cell_plate_metadata`** *(required)* — (list of lists) - Wells occupied by each entry of cell_types, with one inner list per cell type in the same order, for example [['c2','c3'],['c4']]. Each identifier must start with 'c' (column) or 'r' (row); invalid identifiers are skipped without an exception and those wells receive no host_cells label. Because 'condition' combines the labels that are present, a typographical error changes the comparison without raising an error. Default None.
- **`pathogen_plate_metadata`** *(required)* — (list of lists) - Well locations of each pathogen condition, one inner list per entry in pathogen_types. Every item must be a row or column ID string such as 'c1' or 'r3'; anything else is silently ignored and those wells stay unannotated. Ranges like 'c2-c11' are not expanded - list each row/column. Do not leave it None while pathogen_types is set: annotation is not skipped, every row is labelled with the first pathogen_types entry. Defaults: None in the plot-from-db settings, [['c1','c2','c3'],['c4','c5','c6']] for recruitment analysis.
- **`channel_of_interest`** *(required)* — (int, list, or str) - Measurements available to the model. Specify one channel to train on that channel alone, multiple channels to use their combination of measurements, or 'shape' to use outline-derived features. An empty value includes every measurement. A colocalisation feature is associated with both measured channels, so selecting either channel also includes their shared colocalisation features. This setting also selects the channel used for recruitment measurements. Default 3 in machine-learning steps and 1 or 2 elsewhere.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Channel Mapping (4 settings)</summary>

#### Channel Mapping

- **`cell_chann_dim`** *(optional)* — (int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the cell mask when drawing outline overlays, and the switch that enables the cell filters - set an integer and cell_size_range, cell_intensity_range and target_intensity_min are applied; leave it None and cells are not filtered at all. Default 3. Default: `3`.
- **`nucleus_chann_dim`** *(optional)* — (int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the nucleus mask when drawing outline overlays, and the switch that enables nucleus_size_range / nucleus_intensity_range filtering. Set it to None to skip nucleus filtering. It plays no part in segmentation - use nucleus_channel for that. Default 0. Default: `0`.
- **`pathogen_chann_dim`** *(optional)* — (int) - Recruitment analysis only (analyze_recruitment): the image-channel index paired with the pathogen mask when drawing outline overlays, and the switch that enables pathogen_size_range / pathogen_intensity_range filtering. Set it to None to skip pathogen filtering. It plays no part in segmentation - use pathogen_channel for that. Default 2. Default: `2`.
- **`channel_dims`** *(optional)* — (list) - Recruitment analysis only: image-channel indices in the merged arrays. They determine the channels used for overlays and recruitment measurements, each auxiliary recruitment ratio records its channel explicitly, so measurements from different channels remain separate. Default [0, 1, 2, 3]. Default: `[0, 1, 2, 3]`.

</details>

<details>
<summary>Object Filtering (10 settings)</summary>

#### Object Filtering

- **`cell_size_range`** *(optional)* — (list) - [min, max] bounds in pixels^2 on cell_area, used to drop rows from the measurement table during recruitment analysis; only cells strictly between the two values are kept. Both entries must be integers or that bound is silently skipped. Setting it to None widens it to [0, 1e100]. Default [0, 100000]. Default: `[0, 100000]`.
- **`cell_intensity_range`** *(optional)* — (list) - Legacy [min, max] bounds used during recruitment analysis when cell_chann_dim is set. Despite the setting name, the current _object_filter call uses index 0 from [nucleus, pathogen, cell] and therefore filters the nucleus-channel mean intensity. Review the filtered object counts when using this setting. Default None. Default: `None`.
- **`nucleus_size_range`** *(optional)* — (list) - Two-element [min, max] bound in pixels^2 on nucleus_area, used by the recruitment analysis to drop rows from the measurement table; masks are left untouched. Rows are kept only if min &lt; area &lt; max, and each bound is ignored unless it is an int. Default [0, 100000]; None widens it to [0, 1e100]. Default: `[0, 100000]`.
- **`nucleus_intensity_range`** *(optional)* — (list) - Two-element [min, max] bound on mean nucleus-channel intensity used by the recruitment analysis to drop rows from the measurement table - it filters measured objects, not masks or normalization. Rows are kept only if min &lt; mean intensity &lt; max (raw units), and each bound is ignored unless it is an int. Default [0, 100000]. Default: `[0, 100000]`.
- **`pathogen_size_range`** *(optional)* — (list) - Two-element [min, max] area filter in pixels squared applied to the pathogen table in analyze_recruitment, well after segmentation: rows with pathogen_area outside the open interval are dropped. Bounds must be ints - floats are silently ignored. None widens it to effectively unlimited. Default [0, 100000]. Use it to discard debris and merged clumps. Default: `[0, 100000]`.
- **`pathogen_intensity_range`** *(optional)* — (list) - Two-element [min, max] mean-intensity filter applied to the pathogen table in analyze_recruitment; pathogens whose mean intensity in the paired mask channel falls outside the open interval are dropped before recruitment ratios are computed. Bounds must be ints - floats are silently ignored. Default [0, 100000]. Use it to exclude dead or saturated parasites. Default: `[0, 100000]`.
- **`cells_per_well`** *(optional)* — (int) - Minimum cells a well must contribute to survive recruitment analysis; wells below it, and every cell in them, are dropped before the by-well plots and CSVs are produced. Raise it to suppress noisy, sparsely populated wells at the cost of losing those wells. Default 0, which keeps every well. Default: `0`.
- **`target_intensity_min`** *(optional)* — (float) - Recruitment-analysis cutoff on the 95th-percentile intensity of channel_of_interest inside each cell: cells at or below it are discarded before recruitment ratios are computed. Raise it to keep only strongly expressing cells; set 0 or None to disable the filter entirely. Raw intensity units, default 1. Default: `1`.
- **`nuclei_limit`** *(optional)* — (int, bool, or None) - Cap on nuclei per cell, applied when the per-object tables are merged. None disables the filter, True keeps only single-nucleus cells, and an integer N keeps cells with N or fewer. Cells over the cap are dropped from the merged table entirely. Do not pass False: it is interpreted as 0 and removes every cell, leaving an empty analysis rather than raising an error. Default None. Merged Classifier starts at True and Recruitment starts at 1, so both initially retain only single-nucleus cells. Replication starts at 10. Default: `1`.
- **`pathogen_limit`** *(optional)* — (int, bool, or None) - Maximum pathogens per cell. True or 1 = single pathogen only; None or False = no limit; int = custom limit. Default varies by module (1, 3, 10 or 1000 depending on the factory that fills it), so check the module's own settings rather than assuming one value. Default: `10`.

</details>

<details>
<summary>Plate Layout &amp; Controls (5 settings)</summary>

#### Plate Layout & Controls

- **`cell_types`** *(optional)* — (list) - Names of the host cell lines in the experiment, e.g. ['HeLa']. Each name is written into the host_cells column and folded into the combined condition label used for grouping and plotting; the list is positionally paired with cell_plate_metadata, which says which wells hold each one. Default ['HeLa']. Default: `['HeLa']`.
- **`pathogen_types`** *(optional)* — (list) - Names given to each pathogen condition on the plate, e.g. ['wt','ku80']. Element i is written into the pathogen column for every well listed in pathogen_plate_metadata[i] and folded into the combined condition label used for grouping and plotting. Must match pathogen_plate_metadata in length and order; None skips pathogen annotation. Default ['pathogen_1', 'pathogen_2'] for the dataset builders, ['pc'] for the control-based paths, None where types are not used. Default: `['pathogen_1', 'pathogen_2']`.
- **`treatments`** *(optional)* — (list or None) - Names of the drug or treatment conditions in the experiment, e.g. ['dmso','lovastatin']. Each name is written into the treatment column and folded into the combined condition label used for grouping and plotting; positionally paired with treatment_plate_metadata (or treatment_loc), which lists the wells for each. Default ['cm','lovastatin']. Invasion and Replication start at None, so they add no treatment condition until names and matching well groups are supplied. Default: `['cm', 'lovastatin']`.
- **`treatment_plate_metadata`** *(optional)* — (list of lists) - Wells that received each treatment, with one inner list per treatment in the same order, for example [['r1','r2','r3'],['r4','r5','r6']]. Entries must start with 'r' (row) or 'c' (column); other entries are ignored and receive no treatment label. Unlisted wells remain in the output, and their condition values contain only the available cell, pathogen, or treatment labels. Default None. Recruitment starts with [['r1', 'r2', 'r3'], ['r4', 'r5', 'r6']], positionally paired with its two initial treatment names. Default: `[['r1', 'r2', 'r3'], ['r4', 'r5', 'r6']]`.
- **`target`** *(optional)* — (str) - Free-text label for the protein or marker imaged in channel_of_interest, e.g. 'GRA1'. The recruitment run prints it in its banner ('channel:3 = protein') to record what the recruitment ratio is measuring; it feeds no computation, so changing it alters nothing but that log line. Default 'protein'. Default: `'protein'`.

</details>

<details>
<summary>Plots &amp; Diagnostics (4 settings)</summary>

#### Plots & Diagnostics

- **`plot`** *(optional)* — (bool) - Render and save quality-control figures during the pipeline, including channel montages, Cellpose mask overlays, filtration comparisons, and crop grids. Figure generation increases runtime and memory use, particularly for complete plates. test_mode enables this setting automatically. Default False. Merged Classifier and Recruitment both start with plotting enabled so their diagnostic figures are produced on the first run. Default: `True`.
- **`figuresize`** *(optional)* — (int) - Base figure size in inches; figures are built square as figuresize x figuresize and font sizes are derived from it (legend, axis labels and ticks at 0.75x, overlay text at 0.5x). Raise it when text is unreadable at publication scale, lower it to fit panels on screen. Default 10; cluster grids cap total width at 200 inches. Default: `10`.
- **`plot_control`** *(optional)* — (bool) - Before the recruitment plots, draw a control panel of per-compartment mean intensities (cell, nucleus, pathogen, cytoplasm) for every channel, split by condition. Use it to confirm channel assignment and that positive/negative control wells separate as expected before trusting the recruitment numbers. Turn it off to shorten the run. Default True. Default: `True`.
- **`plot_nr`** *(optional)* — (int) - Number of merged image stacks from the start of the folder displayed with cell, nucleus, and pathogen outlines before recruitment analysis. The implementation checks index &lt;= plot_nr, so plot_nr + 1 images are displayed and 0 displays one image. Increase the value to inspect segmentation across more fields. Default 3. Default: `3`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-object recruitment measurements and condition-level summaries.
