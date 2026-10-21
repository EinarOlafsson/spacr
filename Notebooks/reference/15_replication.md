# Quantify intracellular replication

**Purpose.** Estimate intracellular parasite number per vacuole and compare count distributions across experimental groups.

**Recommended use.** Use after parasite and parent-compartment objects have been measured.

**Primary outputs.** Per-vacuole parasite counts and group-level replication summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.analyze_replication`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication)

```python
analyze_replication(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.analyze_replication`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`cell_types`** *(required)* — (list) - Names of the host cell lines in the experiment, e.g. ['HeLa']. Each name is written into the host_cells column and folded into the combined condition label used for grouping and plotting; the list is positionally paired with cell_plate_metadata, which says which wells hold each one. Default ['HeLa'].
- **`pathogen_types`** *(required)* — (list) - Names given to each pathogen condition on the plate, e.g. ['wt','ku80']. Element i is written into the pathogen column for every well listed in pathogen_plate_metadata[i] and folded into the combined condition label used for grouping and plotting. Must match pathogen_plate_metadata in length and order; None skips pathogen annotation. Default ['pathogen_1', 'pathogen_2'] for the dataset builders, ['pc'] for the control-based paths, None where types are not used.
- **`treatments`** *(required)* — (list or None) - Names of the drug or treatment conditions in the experiment, e.g. ['dmso','lovastatin']. Each name is written into the treatment column and folded into the combined condition label used for grouping and plotting; positionally paired with treatment_plate_metadata (or treatment_loc), which lists the wells for each. Default ['cm','lovastatin']. Invasion and Replication start at None, so they add no treatment condition until names and matching well groups are supplied.
- **`group_column`** *(required)* — (str) - Column whose values become the experimental conditions compared against each other; 'condition' is the combined host-cell / pathogen / treatment label built from the plate-metadata maps. Point it at 'pathogen' or 'treatment' to compare on one factor alone. Rows with no value here are dropped before anything is counted. Default 'condition'.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Replication Method (1 settings)</summary>

#### Replication Method

- **`replication_method`** *(optional)* — (str) - Readout for spacr.submodules.analyze_replication. direct_count counts individually segmented parasites per assigned vacuole. size_proxy delegates to the existing area-derived endodyogeny analysis, which aggregates pathogen area per host cell and assumes one vacuole per host for a vacuole interpretation; it does not count parasites or measure volume. deep_learning_coming_soon is reserved for the forthcoming whole-vacuole classification model and cannot run yet. Default direct_count. Default: `'direct_count'`.

</details>

<details>
<summary>Assay Inputs (2 settings)</summary>

#### Assay Inputs

- **`parasite_table`** *(optional)* — (str) - Table in measurements/measurements.db holding one row per segmented parasite. It is read directly rather than through the usual merge, because that merge collapses pathogen rows onto their host cell and would sum several parasites' stain intensities into a single row. Change it only if measure_crop wrote the parasite objects under a non-standard name. Default 'pathogen'. Default: `'pathogen'`.
- **`compartment`** *(optional)* — (str) - Prefix used by per-object measurement columns, so 'pathogen' selects pathogen_area and pathogen_channel_1_percentile_95. It must match the object type contained in the table; otherwise the run stops and reports the unresolved area and intensity columns. Default 'pathogen'. Default: `'pathogen'`.

</details>

<details>
<summary>Size Proxy (Legacy) (9 settings)</summary>

#### Size Proxy (Legacy)

- **`tables`** *(optional)* — (list) - Measurement tables read from each plate's database and merged into one analysis frame. Supported tables are registered object roles, including organelles, and png_list. For tabular Classify, choose a nonempty list of distinct supported tables with at least one object-role table. Default ['cell', 'nucleus', 'pathogen', 'cytoplasm']. Default: `['cell', 'nucleus', 'pathogen', 'cytoplasm']`.
- **`min_area_bin`** *(optional)* — (int) - Width of the smallest area bin in the endodyogeny histogram, and therefore the resolution at which small parasites are distinguished from one another. Expressed in the same unit as max_area, so it follows um_per_px when a scale is set. Too small a value produces sparse noisy bins; too large merges real division states. Default 500. Default: `500`.
- **`max_area`** *(optional)* — (int) - Upper area cutoff applied before the endodyogeny bins are built; larger objects are excluded. The cutoff is applied after um_per_px scaling and therefore uses square micrometres when a scale is set or square pixels when it is None. The default is effectively unbounded. Default 1000000000. Default: `1000000000`.
- **`max_bins`** *(optional)* — (int or None) - Maximum number of area bins in the endodyogeny histogram. None derives the bin count from the data range and min_area_bin; an integer truncates the range so that the largest objects are pooled into the final bin. Set an integer when plates with different size ranges must share one axis. Default None. Default: `None`.
- **`um_per_px`** *(optional)* — (float or None) - Physical size of one pixel, used to convert the endodyogeny area column into square microns before binning. Set it and max_area, min_area_bin and every reported area are in microns; leave it None and they stay in pixels, which makes numbers from objectives of different magnification incomparable. Default 0.1. Default: `0.1`.
- **`pathogen_limit`** *(optional)* — (int, bool, or None) - Maximum pathogens per cell. True or 1 = single pathogen only; None or False = no limit; int = custom limit. Default varies by module (1, 3, 10 or 1000 depending on the factory that fills it), so check the module's own settings rather than assuming one value. Default: `1`.
- **`nuclei_limit`** *(optional)* — (int, bool, or None) - Cap on nuclei per cell, applied when the per-object tables are merged. None disables the filter, True keeps only single-nucleus cells, and an integer N keeps cells with N or fewer. Cells over the cap are dropped from the merged table entirely. Do not pass False: it is interpreted as 0 and removes every cell, leaving an empty analysis rather than raising an error. Default None. Merged Classifier starts at True and Recruitment starts at 1, so both initially retain only single-nucleus cells. Replication starts at 10. Default: `10`.
- **`group_by_class`** *(optional)* — (bool) - Whether the endodyogeny condition labels are split by class before proportions are computed: on, the condition string has the class_column value appended, so each condition-class combination becomes its own group; off, classes are pooled within a condition. It changes what the bars count, not how the statistics are weighted. Default False. Default: `False`.
- **`class_column`** *(optional)* — (str) - Column containing the per-object class label used by class-proportion analysis. Missing values are filled with 0 rather than dropping the corresponding rows; selecting an incorrect column can therefore assign class zero to every object without raising an error. The value is also appended to the condition when group_by_class is enabled. Default 'test'. Replication starts with the predictions column instead. Default: `'predictions'`.

</details>

<details>
<summary>Vacuole Assignment (5 settings)</summary>

#### Vacuole Assignment

- **`vacuole_key`** *(optional)* — (str) - Rule used to group individually segmented parasites into vacuoles. 'auto' prefers an explicit vacuole-ID column, otherwise spatially clusters centroids, then falls back to host cell or one parasite per vacuole with a warning. Set 'spatial', 'cell_id', 'object', or an explicit column name to make that biological assumption reproducible. Default 'auto'. Default: `'auto'`.
- **`vacuole_link_distance`** *(optional)* — (float or None) - Maximum centroid-to-centroid distance in pixels for spatially linking parasites into the same vacuole. None derives the distance from median parasite diameter times vacuole_link_factor; set a calibrated value when magnification or segmentation scale varies between plates. Too large merges separate vacuoles and too small splits one rosette. Default None. Default: `None`.
- **`vacuole_link_factor`** *(optional)* — (float) - Multiplier applied to the median segmented-parasite diameter when vacuole_link_distance is derived automatically. Increasing it joins wider rosettes but also raises the risk of merging nearby vacuoles; decreasing it does the reverse. It is ignored when an explicit link distance or vacuole-ID column is used. Default 1.5. Default: `1.5`.
- **`parasite_count_column`** *(optional)* — (str or None) - Optional column that already stores the number of parasites represented by each segmented row. When set, the assay sums that column per vacuole instead of counting rows, which is required if one row can represent several parasites. None treats every retained row as one parasite. Default None. Default: `None`.
- **`require_host_cell`** *(optional)* — (bool) - Drop parasite rows with no valid host-cell link before constructing vacuoles. This prevents extracellular debris and attached parasites from entering a replication readout, but it will also remove real infected cells when cell segmentation or parent assignment failed. The number removed is reported. Default True. Default: `True`.

</details>

<details>
<summary>Condition Metadata (5 settings)</summary>

#### Condition Metadata

- **`cell_plate_metadata`** *(optional)* — (list of lists) - Wells occupied by each entry of cell_types, with one inner list per cell type in the same order, for example [['c2','c3'],['c4']]. Each identifier must start with 'c' (column) or 'r' (row); invalid identifiers are skipped without an exception and those wells receive no host_cells label. Because 'condition' combines the labels that are present, a typographical error changes the comparison without raising an error. Default None. Default: `None`.
- **`pathogen_plate_metadata`** *(optional)* — (list of lists) - Well locations of each pathogen condition, one inner list per entry in pathogen_types. Every item must be a row or column ID string such as 'c1' or 'r3'; anything else is silently ignored and those wells stay unannotated. Ranges like 'c2-c11' are not expanded - list each row/column. Do not leave it None while pathogen_types is set: annotation is not skipped, every row is labelled with the first pathogen_types entry. Defaults: None in the plot-from-db settings, [['c1','c2','c3'],['c4','c5','c6']] for recruitment analysis. Default: `[['c1'], ['c2']]`.
- **`treatment_plate_metadata`** *(optional)* — (list of lists) - Wells that received each treatment, with one inner list per treatment in the same order, for example [['r1','r2','r3'],['r4','r5','r6']]. Entries must start with 'r' (row) or 'c' (column); other entries are ignored and receive no treatment label. Unlisted wells remain in the output, and their condition values contain only the available cell, pathogen, or treatment labels. Default None. Recruitment starts with [['r1', 'r2', 'r3'], ['r4', 'r5', 'r6']], positionally paired with its two initial treatment names. Default: `None`.
- **`level`** *(optional)* — (str) - Result level. For regression, 'both' writes results_grna.csv and results_gene.csv and corrects each family separately; 'grna' reports guide effects, and 'gene' pools guides by gene. Nonparametric inference also honours this choice. Mixed models disable it because they estimate gene effects with guides nested inside genes. For proportion plots, the same key selects 'object', 'well', or 'plate' aggregation. Default 'both' for regression and 'object' for proportions. Default: `'object'`.
- **`change_plate`** *(optional)* — (bool) - Relabel each source directory as plate1, plate2, ... instead of trusting the plate ID stored in its database. Use it when several plates were written under the same name, which would otherwise let two plates' fields pool into one threshold and one well. Default False. Default: `False`.

</details>

<details>
<summary>Object Filtering (2 settings)</summary>

#### Object Filtering

- **`min_parasite_area`** *(optional)* — (int or float) - Smallest object area in pixels retained as a parasite. Smaller objects are treated as debris because their outside-stain statistic is estimated from too few pixels for stable thresholding. Increase it when pathogen masks are over-segmented into small fragments. Default 0, which applies no area filter. Default: `0`.
- **`max_parasite_area`** *(optional)* — (float or None) - Largest object area in pixels kept as a parasite. Anything bigger is several parasites merged by the mask, whose rim statistic mixes them and whose single classification then stands for all of them. None keeps everything. Default None. Default: `None`.

</details>

<details>
<summary>Replication Scoring (3 settings)</summary>

#### Replication Scoring

- **`max_parasites_per_vacuole`** *(optional)* — (int) - Largest power-of-two parasite count given its own replication bucket. Counts above it remain visible in a '&gt;N' bucket and non-powers stay in the separate QC bucket; they are never clipped or rounded. Use a power of two large enough for the experiment's duration. Default 16. Default: `16`.
- **`non_power_of_two_warn`** *(optional)* — (float) - Fraction of a well's vacuoles allowed in the non-power-of-two bucket before the well is flagged as unreliable. Three-, five-, or seven-parasite rosettes usually indicate segmentation or vacuole-linking errors, so lowering the threshold makes QC stricter without deleting any observations. Default 0.2. Default: `0.2`.
- **`seed_wells_from_cells`** *(optional)* — (bool) - Read the cell table as well, so a well holding host cells but no parasites appears in the results with a zero denominator instead of vanishing from the plate entirely. Switch it off only when the database has no cell table. Default True. Default: `True`.

</details>

<details>
<summary>Assay Output (2 settings)</summary>

#### Assay Output

- **`cmap`** *(optional)* — (str) - Matplotlib colormap applied to single-channel image previews and plate heatmaps. Perceptually uniform maps ('viridis', 'inferno', 'magma') preserve the relative visibility of intensity differences; 'gray' resembles the raw single-channel microscope image. Any registered matplotlib name is accepted, with an '_r' suffix to reverse it. Default 'inferno' for image plots and 'viridis' for plate heatmaps. Default: `'viridis'`.
- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `True`.

</details>

<details>
<summary>Runtime &amp; Reliability (1 settings)</summary>

#### Runtime & Reliability

- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-vacuole parasite counts and group-level replication summaries.
