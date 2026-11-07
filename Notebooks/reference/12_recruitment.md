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

<details>
<summary>Additional Settings (8 settings)</summary>

#### Additional Settings

- **`field_qc`** *(optional)* — (bool) - Recruitment analysis: count host cells and parasites per imaged field from measurements.db and drop every object in a field that is over-crowded, too sparse or over-infected, before the recruitment ratios are grouped by well; the per-field verdicts are written to results/field_qc.csv. Replication analysis: count vacuoles (or parasites) per field and drop fields above the robust limit or below field_qc_min_objects. Default False. Default: `False`.
- **`field_qc_mad`** *(optional)* — (float) - Recruitment or replication analysis with field_qc: a field fails when any column in field_qc_columns exceeds the plate median plus this many robust standard deviations (MAD scaled by 1.4826). Lower is stricter; None disables this test. Default 3.5. Default: `3.5`.
- **`field_qc_min_cells`** *(optional)* — (int) - Recruitment analysis with field_qc: fields with fewer host cells than this are dropped as too sparse to be informative. None disables this test. Default 10. Default: `10`.
- **`field_qc_max_pathogens_per_cell`** *(optional)* — (float) - Recruitment analysis with field_qc: fields whose parasites per host cell exceed this absolute ceiling are dropped, in addition to the MAD limit. None means no absolute ceiling. Default None. Default: `None`.
- **`field_qc_columns`** *(optional)* — (list) - Recruitment analysis with field_qc: the per-field density columns tested against the MAD limit. Choose from n_cells_field, n_pathogens_field and pathogens_per_cell_field. Default ['n_pathogens_field', 'pathogens_per_cell_field']. Default: `['n_pathogens_field', 'pathogens_per_cell_field']`.
- **`confounder_covariates`** *(optional)* — (list) - Recruitment analysis only: columns to screen as possible confounders of the recruitment ratio across conditions, for example n_cells_field, n_pathogens_field, pathogens_per_cell_field, cell_area or a pathogen intensity. Each is tested for a within-condition slope and for uneven spread across conditions; the table is written to results/confounder_report.csv. Field density columns are attached automatically. None skips the screen. Default None. Default: `None`.
- **`adjust_covariates`** *(optional)* — (list) - Recruitment analysis only: columns to regress out of the recruitment ratio (ANCOVA with a pooled within-condition slope), adding a recruitment_adjusted column per object and per well. Only list variables the condition does not itself cause; a mediator such as vacuole size would remove the true effect together with the artifact. The fitted slopes are written to results/covariate_model.csv. None disables adjustment. Default None. Default: `None`.
- **`adjust_log_outcome`** *(optional)* — (bool) - Recruitment analysis with adjust_covariates or confounder_covariates: model the log of the recruitment ratio, which suits multiplicative effects, and back-transform the adjusted value. False models the raw ratio. Default True. Default: `True`.

</details>

<details>
<summary>Additional settings (35 settings)</summary>

#### Additional settings

- **`background_filters`** *(optional)* — (list) - Recruitment and replication: object quality rules that compare a vacuole measurement with the median of the mask-free pixels of its own field, so the threshold follows each plate, well and strain. Each rule is a dict with channel, statistic (mean_intensity by default, or e.g. percentile_25), and any of min_ratio (drop vacuoles without signal, e.g. 1.5), max_ratio (drop saturated uniform blobs, e.g. 20) and max_quantile (drop the top fraction per plate, e.g. Hoechst percentile_25 at 0.98 for masks filled by a host nucleus). Field backgrounds are read from merged_dir and cached in results. Default None. Default: `None`.
- **`background_mask_channel`** *(optional)* — (int) - Recruitment and replication: index of the mask channel in the merged arrays that marks objects; pixels where it is 0 count as field background for background_filters. Default -1. Default: `-1`.
- **`bracket_pairs`** *(optional)* — (str) - Recruitment and replication graphs: which significant post hoc pairs get a bracket. 'one_factor' marks pairs that differ in one factor of the x axis (same host with another strain, same strain in another host; every significant pair when the conditions vary in one variable only), 'all' marks every significant pair, and a list such as [['Me49WT_HeLa_WT', 'Me49dGRA8_HeLa_WT']] marks the listed pairs. Brackets stack with the shortest spans lowest; beyond 12 the smallest p values are drawn and the rest stay in stats.csv. Default 'one_factor'. Default: `'one_factor'`.
- **`condition_factors`** *(optional)* — (dict) - Recruitment and replication graphs: split annotated values into the factors they carry, so the +/- matrix gets one row per factor, e.g. {'Me49WT': ['Me49', 'WT'], 'Me49dGRA8': ['Me49', 'dGRA8'], 'Me49GRA8comp': ['Me49', 'Comp']}. Factors at the same position form one variable (background, genotype). Values without an entry are a factor of their own. Default None. Default: `None`.
- **`condition_labels`** *(optional)* — (dict) - Recruitment and replication panels: display labels for raw values, e.g. {'HeLa_ALG2': 'HeLa ALG2 KO'}. Values without a label are shown with spaces in place of underscores. Labels set on a panel are merged on top. Default None. Default: `None`.
- **`condition_order`** *(optional)* — (list) - Recruitment and replication panels: the x-axis order shared by every panel. Listed values come first in the given order, the rest follow in natural sort order, so a condition sits in the same place in every panel. The order set on a panel overrides it. Default None. Default: `None`.
- **`confounder_filter`** *(optional)* — (dict) - Recruitment analysis: range filters on confounding variables, {column: [min, max]} at object level, or {column: {'quantiles': [0.05, 0.95], 'level': 'well'}} to drop whole wells outside per-plate quantile bounds ('range' gives absolute bounds). The well density columns n_cells_well, n_pathogens_well and pathogens_per_cell_well are computed from measurements.db when named. Each step is reported in results/filter_summary.csv. Default None. Default: `None`.
- **`control_color_by`** *(optional)* — (str) - Recruitment control graphs: colour every condition bar by the value of this annotated column (for example treatment, so all wells of one host share a colour). Conditions are always ordered as the plate map lists them unless condition_order is set. None colours each condition separately. Default None. Default: `None`.
- **`exclude_edge_objects`** *(optional)* — (bool) - Recruitment and replication analyses: drop objects cut by the image edge, i.e. whose centre is closer to the field edge than their radius (distance_to_field_edge below half of equivalent_diameter_area). Recruitment tests the pathogen object of each host cell; replication drops a vacuole when any of its parasites is cut. Reported in filter_summary.csv. Default False. Default: `False`.
- **`factor_order`** *(optional)* — (list) - Recruitment and replication graphs: order of the +/- matrix rows, as factor or display names, e.g. ['HeLa WT', 'HeLa ALG2', 'RH', 'Me49', 'WT', 'dGRA8', 'Comp']. None groups the rows by variable in plate-map order. Default None. Default: `None`.
- **`figures`** *(optional)* — (dict) - Recruitment and replication panels: figures assembled from saved panels, {figure_name: {'panels': ['P01', 'P02'], 'ncol': 2}}, written as PDF and PNG with bold panel IDs into results/figures (recruitment) or results/analyze_replication/figures. Recruitment control panels (C01, ...) can be used as well. None assembles nothing. Default None. Default: `None`.
- **`max_vacuoles_per_cell`** *(optional)* — (int) - Recruitment analysis: keep host cells with at most this many pathogen objects (pathogen_prcfo_count), applied after the merge, which already used pathogen_limit. The removed count is reported in results/filter_summary.csv. None keeps every host cell. Default None. Default: `None`.
- **`merged_dir`** *(optional)* — (str) - Recruitment and replication: folder with the spaCR merged arrays (&lt;file_name&gt;.npy) used to measure field backgrounds for background_filters. None uses the merged folder of the project. Default None. Default: `None`.
- **`min_solidity`** *(optional)* — (float) - Recruitment and replication: drop vacuole (pathogen or replication compartment) masks whose solidity, area divided by convex hull area, is below this value; removes masks that reach beyond the vacuole and fragments of rosettes (for example 0.90). Default None. Default: `None`.
- **`object_qc`** *(optional)* — (bool) - Recruitment and replication: drop objects whose area or mean intensity in any channel lies beyond the robust fence of the objects of the same type on the same plate (log scale). Counts removed per measure are written to results/object_qc_summary.csv. Default True. Default: `True`.
- **`object_qc_mad`** *(optional)* — (float) - Recruitment and replication object_qc: robust standard deviations of the log values beyond which an object is dropped. Default 3.5. Default: `3.5`.
- **`object_qc_measures`** *(optional)* — (list) - Recruitment and replication object_qc: measures checked per object, any of 'area' and 'intensity'. None uses both. Default None. Default: `None`.
- **`outlier_points`** *(optional)* — (bool) - Recruitment and replication panels: leave points beyond a robust fence of their group out of every bar panel and its test; they stay in the data CSV marked outlier = True and are counted per group in the statistics. Default True. Default: `True`.
- **`outlier_points_method`** *(optional)* — (str) - Recruitment and replication panels: fence used by outlier_points, 'mad' (median +- 3.5 x 1.4826 x MAD) or 'iqr' (1.5 x IQR beyond the quartiles); 'mad:3' or 'iqr:2' set another fence. Default 'mad'. Default: `'mad'`.
- **`panels`** *(optional)* — (list) - Recruitment and replication analyses: publication panels, one dict per panel, e.g. {'name': 'strains_in_WT', 'where': {'host_cells': 'HeLa_WT'}, 'group': 'pathogen', 'order': [...], 'labels': {...}, 'annotations': {'GRA8': {'WT': '+', 'dGRA8': '-'}}, 'value': 'recruitment'}. group can be any annotated column. Recruitment writes one bar panel per entry; replication writes three (size or count bins with chi-square, the same with the Fisher-Freeman-Halton test, and a bar of the per-vacuole mean). Each panel folder under results holds PDF, PNG, the plotted data and statistics CSVs; the test is chosen from normality and equal-variance checks. None draws no panels. Default None. Default: `None`.
- **`qc_group`** *(optional)* — (str) - Recruitment and replication well_qc and object_qc: column whose groups are compared separately, so wells and objects are judged against those of the same condition and real differences between conditions are kept. 'condition' uses the condition without the replicate when the replicate is part of the condition name. A group with fewer than 8 wells (well_qc) or 30 objects (object_qc) on a plate is compared with the whole plate. None compares every well and object with its whole plate. Default 'condition'. Default: `'condition'`.
- **`replicate_column`** *(optional)* — (str or None) - Optional column naming the biological replicate of each well. Drug-screen mode counts it per level in drug_effects.csv and does not model it. Recruitment and replication panels use it as the statistical unit: object values are averaged per well, then well means per replicate. Default None. Default: `None`.
- **`report`** *(optional)* — (bool) - Recruitment analysis: write the standard recruitment report into results/report: representative tiles per condition, recruitment per condition before and after the opt-in filters at well and replicate level, and well-level control graphs (every channel in every object type, objects per well) before and after filtering, as A4 pages in recruitment_report.pdf with one folder per panel (PDF, PNG, data.csv, stats.csv) and an automatic figure legend (figure_legend.txt). Default False. Default: `False`.
- **`report_layout`** *(optional)* — (str) - Recruitment report: page layout of recruitment_report.pdf. compact uses 5-6 pt text, small tiles and four panels per row, and fits the whole figure on one A4 page when it can (the legend follows on its own page); spacious uses larger text and paginates freely. More conditions, channels or object types continue on further pages in both. Default 'compact'. Default: `'compact'`.
- **`report_max_conditions_per_row`** *(optional)* — (int) - Recruitment report: most conditions drawn side by side in one graph or tile row; longer condition axes continue on the next row or page. Default 12. Default: `12`.
- **`report_tile_channels`** *(optional)* — (dict) - Recruitment report: composite colours of the tiles, {channel: colour} with colours red, green, blue, magenta, cyan, yellow or grey, e.g. {2: 'red', 1: 'green', 0: 'blue'}. None shows channel_of_interest in red, the pathogen channel in green and the nucleus channel in blue. Default None. Default: `None`.
- **`report_tile_seed`** *(optional)* — (int) - Recruitment report: random seed for choosing the representative tiles, so the same tiles are chosen on every run. Default 0. Default: `0`.
- **`report_tile_size`** *(optional)* — (int) - Recruitment report: side of each representative tile in pixels, centred on the host cell mask centroid. Default 128. Default: `128`.
- **`report_tiles_per_condition`** *(optional)* — (int) - Recruitment report: number of representative host-cell tiles per condition, chosen at random from the filtered objects. Default 5. Default: `5`.
- **`representative_crops`** *(optional)* — (bool) - Recruitment and replication: write representative crops from the merged arrays: one crop per condition closest to the condition median of the main readout, and under every main graph a grid of the objects nearest each condition median, from the same data version as the graph. Saved as raw uint16 TIFF and as 16-bit PNG scaled to the 1st and 99th percentile of the crops of each figure. Default True. Default: `True`.
- **`representative_crops_n`** *(optional)* — (int) - Recruitment and replication representative_crops: number of crops per condition in the grid under each graph. Default 5. Default: `5`.
- **`well_qc`** *(optional)* — (bool) - Recruitment and replication: drop wells that differ from the other wells of their plate, e.g. a row imaged far brighter or dimmer. Per well and object type the object count, median area and median mean intensity of every channel are compared with the plate median of the wells; a well more than well_qc_mad robust standard deviations away in any measure is dropped before the other filters. Every well, its measures, robust z and the failing measures are written to results/well_qc.csv. Default True. Default: `True`.
- **`well_qc_mad`** *(optional)* — (float) - Recruitment and replication well_qc: robust standard deviations (1.4826 x median absolute deviation over the wells of the plate) beyond which a well fails. Default 3.5. Default: `3.5`.
- **`well_qc_measures`** *(optional)* — (list) - Recruitment and replication well_qc: measures compared between wells, any of 'count', 'area' and 'intensity'. None uses all three. Default None. Default: `None`.
- **`x_axis_style`** *(optional)* — (str) - Recruitment and replication graphs: how conditions are named under the x axis. 'auto' writes plain labels of the one variable that differs between the conditions of a graph, and a +/- matrix (one row per factor, + under the conditions that carry it) when two or more variables differ; 'matrix' always uses the matrix, 'labels' the full condition names. Factors shared by every condition of a graph are named in the figure legend instead. Default 'auto'. Default: `'auto'`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-object recruitment measurements and condition-level summaries.
