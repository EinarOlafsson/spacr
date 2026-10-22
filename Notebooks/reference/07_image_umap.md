# Embed object images

**Purpose.** Compute a two-dimensional embedding of single-object images or features and display representative objects in embedding space.

**Recommended use.** Use for exploratory assessment of phenotypic structure, batch effects, and outlying experimental groups.

**Primary outputs.** Embedding coordinates and an image-annotated embedding figure.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.core.generate_image_umap`](https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap)

```python
generate_image_umap(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.core.generate_image_umap`](https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Input Data (8 settings)</summary>

#### Input Data

- **`tables`** *(optional)* — (list) - Measurement tables read from each plate's database and merged into one analysis frame. Supported tables are registered object roles, including organelles, and png_list. For tabular Classify, choose a nonempty list of distinct supported tables with at least one object-role table. Default ['cell', 'nucleus', 'pathogen', 'cytoplasm']. Default: `['cell', 'cytoplasm', 'nucleus', 'pathogen']`.
- **`filter_by`** *(optional)* — (str or None) - Restricts the feature matrix before dimensionality reduction: only columns matching this channel are kept and the other channel_1-channel_4 columns are dropped. Accepts 'channel_0'-'channel_3', an int, a list of channel numbers, or 'morphology' to keep only shape features (area, eccentricity, Zernike moments, ...). None, 'None', 'all', and '*' disable filtering. Default 'channel_0'. Default: `'channel_0'`.
- **`row_limit`** *(optional)* — (int) - Randomly subsample the joined measurement table down to this many objects (fixed seed 42) before dimensionality reduction, keeping UMAP and clustering tractable. Raise it for a more faithful map at higher memory and runtime cost, or set to None to use every row. Must not exceed the available row count. Default 1000. Default: `1000`.
- **`exclude`** *(optional)* — (str or list) - Exclude measurement features before UMAP or ML. Exact columns take priority, then whole-token channel_N groups (channel_0 differs from channel_01), then full-match regexes (.*channel_0.* matches contained text). Unmatched selectors warn; invalid regexes raise a pattern-naming error. Logs give hit counts; verbose adds matched names. Applied with channel_of_interest. Features, not rows: use exclude_rows for rows. Default None keeps all features. Default: `None`.
- **`exclude_rows`** *(optional)* — (dict or None) - General UMAP row exclusions. Choose one or more database columns, then check the values whose rows should be removed. Rules are combined with OR, so a row matching any selected column/value pair is excluded. Default None keeps every row. Default: `None`.
- **`remove_highly_correlated`** *(optional)* — (bool or float) - Before dimensionality reduction, drop numeric features whose absolute Pearson correlation with an already-kept feature exceeds a cut-off. Pass a float to set the cut-off yourself, True to use 0.95, or False to keep everything. Enable it so families of near-duplicate measurements (area, perimeter, convex_area) do not dominate the embedding. Default True. Default: `True`.
- **`log_data`** *(optional)* — (bool) - Apply log(x + 1e-6) to every numeric feature, after the correlation filter and before standard scaling. Compresses heavy-tailed measurements such as intensity sums and areas so a handful of bright or huge objects stop dominating the embedding. Negative feature values become NaN and are then filled with the column mean. Default False. Default: `False`.
- **`resnet_features`** *(optional)* — (bool) - Placeholder for embedding raw crops with ResNet features instead of the measured feature table. Not implemented: when it is True, generate_image_umap raises NotImplementedError before embedding. Leave it False. Default False. Default: `False`.

</details>

<details>
<summary>Dimensionality Reduction (3 settings)</summary>

#### Dimensionality Reduction

- **`reduction_method`** *(optional)* — (str) - Dimensionality reduction run before clustering and plotting: 'umap' preserves more global structure and can be fitted on controls then applied to all data, 'tsne' emphasises local neighbourhoods and cannot reuse a fitted model. With 'tsne', min_dist is ignored and n_neighbors is used as perplexity. Anything else raises ValueError. Default 'umap'. Default: `'umap'`.
- **`random_seed`** *(optional)* — (int) - Reproducibility seed shared by labelled train/test splitting, train/validation splitting, and grouped cross-validation folds. Keep it fixed to reproduce a run; vary it to evaluate sensitivity to the sampled partition. Default 42. Default: `42`.
- **`metric`** *(optional)* — (str) - Distance metric used both by the reducer (UMAP or t-SNE) and by DBSCAN clustering, e.g. 'euclidean', 'manhattan', 'cosine' or 'correlation'. Correlation-type metrics compare feature profiles regardless of magnitude and often separate phenotypes better than euclidean on scaled data. Default 'euclidean'. Default: `'euclidean'`.

</details>

<details>
<summary>UMAP (2 settings)</summary>

#### UMAP

- **`n_neighbors`** *(optional)* — (int or float) - Size of the local neighbourhood UMAP balances against global structure, and the perplexity when reduction_method is 'tsne'. Small values (5-50) sharpen fine local structure; large values give a smoother, more global embedding. A float is read as a fraction of the number of objects, and anything below 2 is clamped to 2. Default 1000. Default: `1000`.
- **`min_dist`** *(optional)* — (float) - UMAP's minimum spacing between points in the 2-D embedding, range 0.0-1.0. Low values (0.0-0.1) let clusters pack tightly and look crisply separated; higher values spread points out and preserve more of the global layout at the cost of visible cluster structure. Ignored when reduction_method is 'tsne'. Default 0.1. Default: `0.1`.

</details>

<details>
<summary>t-SNE (4 settings)</summary>

#### t-SNE

- **`tsne_perplexity`** *(optional)* — (float) - t-SNE neighborhood scale. It must be smaller than the number of rows; values around 5-50 are typical. Low values emphasize very local structure and can fragment populations; high values smooth them together. Used only by t-SNE. Default 30. Default: `30.0`.
- **`tsne_learning_rate`** *(optional)* — (float) - t-SNE optimization step size. Too small crowds points into a dense ball; too large can scatter them. Used only by t-SNE. Default 200. Default: `200.0`.
- **`tsne_early_exaggeration`** *(optional)* — (float) - t-SNE's initial attraction multiplier, controlling how much space forms between natural groups early in optimization. Used only by t-SNE. Default 12. Default: `12.0`.
- **`tsne_max_iter`** *(optional)* — (int) - Maximum t-SNE optimization iterations. Increase it when optimization has not stabilized; every increase costs runtime. Used only by t-SNE. Default 1000. Default: `1000`.

</details>

<details>
<summary>PCA (2 settings)</summary>

#### PCA

- **`pca_whiten`** *(optional)* — (bool) - Rescale PCA components to unit variance after projection. This can help distance-based clustering but discards relative component magnitude. Used only by PCA. Default False. Default: `False`.
- **`pca_svd_solver`** *(optional)* — (str) - PCA decomposition algorithm: auto chooses from the data shape, full is exact, randomized is faster on large matrices, and covariance_eigh suits many rows with relatively few features. Used only by PCA. Default 'auto'. Default: `'auto'`.

</details>

<details>
<summary>Isomap (2 settings)</summary>

#### Isomap

- **`isomap_n_neighbors`** *(optional)* — (int) - Number of neighbors in Isomap's geodesic graph. Too few can disconnect the graph; too many make the result approach a global linear projection. Used only by Isomap. Default 15. Default: `15`.
- **`isomap_path_method`** *(optional)* — (str) - Isomap shortest-path solver: auto chooses, FW uses Floyd-Warshall, and D uses Dijkstra. Used only by Isomap. Default 'auto'. Default: `'auto'`.

</details>

<details>
<summary>Spectral Embedding (2 settings)</summary>

#### Spectral Embedding

- **`spectral_affinity`** *(optional)* — (str) - Graph construction for Spectral Embedding: nearest_neighbors builds a sparse local graph; rbf builds a dense radial-basis affinity. Used only by Spectral Embedding. Default 'nearest_neighbors'. Default: `'nearest_neighbors'`.
- **`spectral_n_neighbors`** *(optional)* — (int) - Neighbor count for Spectral Embedding when affinity is nearest_neighbors. Ignored for rbf affinity. Default 15. Default: `15`.

</details>

<details>
<summary>Clustering (6 settings)</summary>

#### Clustering

- **`clustering`** *(optional)* — (str) - Algorithm applied to the two-dimensional embedding. 'dbscan' identifies density-based clusters from eps and min_samples, labels sparse points as noise (-1), and determines the cluster count from the data. 'kmeans' forces exactly min_samples clusters and assigns every point. Use dbscan for distinct phenotypes over a diffuse background and kmeans when a fixed number of groups is required. Default 'dbscan'. Default: `'dbscan'`.
- **`eps`** *(optional)* — (float) - DBSCAN neighbourhood radius, expressed in the units of the UMAP/t-SNE embedding and measured with the 'metric' setting: two points are neighbours if they lie within this distance. Raise it to merge fragments into fewer, larger clusters and leave less noise; lower it to split clusters and push more points to noise (-1). Ignored when clustering is 'kmeans'. Default 0.9. Default: `0.9`.
- **`min_samples`** *(optional)* — (int) - Meaning depends on 'clustering': for DBSCAN it is how many points must fall within eps for a point to count as a core point, so raising it yields fewer, denser clusters and more noise; for KMeans this same value is reused as n_clusters, the exact number of clusters produced. Lower it (or raise eps) when no clusters are found. Default 100. Default: `100`.
- **`remove_cluster_noise`** *(optional)* — (bool) - Remove points that DBSCAN labels as noise (-1) before plotting the embedding, so the figure contains only clustered points. Disable it to retain all embedded points, including diffuse background. It has no effect with kmeans, which never emits -1, and is disabled automatically when color_by is set. Default True. Default: `True`.
- **`analyze_clusters`** *(optional)* — (bool) - After clustering the embedding, rank every measured feature by cluster separation using random-forest importance and a per-feature ANOVA or Kruskal-Wallis test, then write results/cluster_results.csv. Enable this setting to identify morphology or intensity features associated with each cluster. It adds a full model fit over the feature table. Default False. Default: `False`.
- **`color_by`** *(optional)* — (str) - Name of a column in the joined measurement table (e.g. 'cond', 'columnID', 'plateID') used to color embedding points instead of the cluster labels. Set it to see how a known grouping such as condition or plate column falls across the map; leave it None to color by the clustering result. Setting it also disables remove_cluster_noise, plot_outlines and smooth_lines. Default None. Default: `None`.

</details>

<details>
<summary>Plate &amp; Batch Correction (8 settings)</summary>

#### Plate & Batch Correction

- **`batch_correction`** *(optional)* — (str) - Plate/batch correction applied before Image UMAP, ML screen classification or phenotype regression. 'none' leaves measurements alone; 'center' removes each plate's mean shift; 'zscore' aligns plate means and variances; 'robust_zscore' uses median/MAD and tolerates outliers; 'combat' models the batch effect while protecting the terms named in batch_covariate_column. Correct when plates were stained or imaged separately; leave off when they were not, since every method removes real signal that happens to align with plate. See spacr.batch_correction.correct_batch_effects. Default 'none'. Default: `'none'`.
- **`batch_column`** *(optional)* — (str) - Metadata column that identifies independent acquisition batches, normally 'plateID'. Every analyzed row must have a value and at least batch_min_samples rows must occur in each batch. Use an acquisition date or instrument ID only if that is the nuisance source you intend to remove. Default 'plateID'. API: spacr.batch_correction.correct_batch_effects. Default: `'plateID'`.
- **`batch_control_column`** *(optional)* — (str or None) - Metadata column containing reference-control labels for control_center, normally 'columnID' for plate controls. It is ignored by center, zscore, robust_zscore, and none. Blank follows col_to_compare in Image UMAP or location_column in Classify (ML); regression defaults to 'columnID'. API: spacr.batch_correction.correct_batch_effects. Default: `None`.
- **`batch_control_values`** *(optional)* — (str, number, list or None) - Reference/negative-control value(s) in batch_control_column used by control_center. Each plate needs at least batch_min_samples matching rows. Image UMAP falls back to neg and Classify (ML) to negative_control_id when this field is blank; regression requires an explicit value. Default varies by module. API: spacr.batch_correction.correct_batch_effects. Default: `None`.
- **`batch_covariate_column`** *(optional)* — (str, list or None) - Metadata column(s) naming the biological effects ComBat must preserve, for example 'condition' or 'condition,timepoint'. ComBat estimates the batch effect from residuals after fitting these terms, so unlisted effects may be removed with the plate effect. Include every treatment effect that must remain in the corrected data. See spacr.batch_correction.correct_batch_effects. Default None. Default: `None`.
- **`batch_combat_mean_only`** *(optional)* — (bool) - True corrects only the additive batch shift and leaves each batch's scale alone. Use it when the plates differ in level but not in spread, or when a batch has too few rows for a stable variance estimate. False (the default) corrects both location and scale, which is standard ComBat. Ignored by every method other than combat. API: spacr.batch_correction.correct_batch_effects. Default: `False`.
- **`batch_min_samples`** *(optional)* — (int) - Minimum number of rows required in every batch, and minimum matching reference controls per batch for control_center. Correction stops with an actionable error below this threshold because a one- or two-object plate estimate is unstable. Default 3. API: spacr.batch_correction.correct_batch_effects. Default: `3`.
- **`batch_missing_control`** *(optional)* — (str) - Policy when control_center cannot find enough reference controls on a plate: 'error' stops rather than silently mixing corrected and raw plates; 'skip' leaves that plate unchanged and records a warning. Default 'error'. API: spacr.batch_correction.correct_batch_effects. Default: `'error'`.

</details>

<details>
<summary>Points &amp; Images (13 settings)</summary>

#### Points & Images

- **`dot_size`** *(optional)* — (int) - Matplotlib marker area, in points squared, for each object plotted in the UMAP/tSNE embedding. Increase it when a few hundred points make the scatter look empty; drop it to roughly 5-10 when tens of thousands of points overplot and hide cluster structure. Default 50. Default: `50`.
- **`point_color`** *(optional)* — (str) - Point color for static and interactive UMAP plots. Use 'cluster' or 'viridis' for cluster-based Viridis colors, or any Matplotlib color such as '#4cc9f0', 'orange', or 'white' for one fixed color. Default 'cluster'. Default: `'cluster'`.
- **`point_alpha`** *(optional)* — (float) - Opacity of UMAP points from 0 (invisible) to 1 (opaque), used by both static and interactive plots. Default 0.65. Default: `0.65`.
- **`outline_width`** *(optional)* — (float) - Width in points of cluster outlines and interactive selection rings. Smaller values produce thinner boundaries. Default 1.0. Default: `1.0`.
- **`img_zoom`** *(optional)* — (float) - Scale applied to each object thumbnail pasted onto the embedding: 1.0 draws the crop at native pixel size, 0.5 at half. Raise it when crops are too small to judge morphology, lower it when thumbnails overlap and bury the point cloud. Practical range about 0.1-2.0. Default 0.5. Default: `0.5`.
- **`image_nr`** *(optional)* — (int) - How many example object crops to draw on the embedding plot: that many per cluster when plot_by_cluster is on (smaller clusters show all they have), otherwise that many sampled at random overall. It also sets how many images each cluster contributes to the cluster-grid figure. Raise it for a fuller montage, lower it when thumbnails hide the points. Default 16. Default: `16`.
- **`plot_images`** *(optional)* — (bool) - Paste the actual object crops onto the embedding scatter instead of showing bare points. Turn it off for a fast, plain scatter on large datasets - doing so also forces black_background to False and skips the cluster grid figure entirely. Default True. Default: `True`.
- **`remove_image_canvas`** *(optional)* — (bool) - When object thumbnails are overlaid on the embedding plot, make zero-valued background pixels transparent so only the segmented object is visible. Enable it to remove black thumbnail backgrounds, especially with black_background. Only L, I and RGB crops are supported; other PIL modes raise an error. Default False. Default: `False`.
- **`plot_points`** *(optional)* — (bool) - Show the scatter marker for each object in the embedding. When False the markers are still drawn but at alpha 0, so cluster colors and the legend survive while only the outlines and overlaid thumbnails stay visible - handy for image-only UMAP figures. Marker size comes from dot_size. Default True. Default: `True`.
- **`plot_outlines`** *(optional)* — (bool) - Draw a boundary around each cluster in the embedding - a smoothed hull when smooth_lines is True, otherwise the raw convex hull edges. Helps show cluster extent and overlap but clutters dense maps; clusters with fewer than three points are skipped. Forced off when color_by is set. Default True. Default: `True`.
- **`smooth_lines`** *(optional)* — (bool) - Draw cluster outlines as a smoothed spline through the convex hull (2 pt wide) rather than the raw straight hull segments (4 pt). Purely cosmetic - it does not change clustering; switch it off if smoothing distorts the true cluster boundary. No effect unless plot_outlines is on, and forced off when color_by is set. Default True. Default: `True`.
- **`plot_by_cluster`** *(optional)* — (bool) - Chooses which thumbnails get overlaid on the embedding: when True, up to image_nr crops are sampled from each cluster (DBSCAN noise excluded) so every cluster is represented; when False, image_nr crops are sampled at random across the whole map. Keep True to compare cluster morphologies, False for an unbiased sample. Default True. Default: `True`.
- **`plot_cluster_grids`** *(optional)* — (bool) - Render a second figure with one colour-bordered panel per cluster, each containing up to image_nr example crops, and save it as &lt;METHOD&gt;_grid.pdf when save_figure is enabled. The cluster grid is emitted after the embedding and therefore becomes the final displayed figure. Ignored unless plot_images is True. Default False. Default: `False`.

</details>

<details>
<summary>Canvas &amp; Output (5 settings)</summary>

#### Canvas & Output

- **`figuresize`** *(optional)* — (int) - Base figure size in inches; figures are built square as figuresize x figuresize and font sizes are derived from it (legend, axis labels and ticks at 0.75x, overlay text at 0.5x). Raise it when text is unreadable at publication scale, lower it to fit panels on screen. Default 10; cluster grids cap total width at 200 inches. Default: `10`.
- **`umap_canvas_width`** *(optional)* — (int) - Initial interactive UMAP chart width in pixels. The chart/sidebar divider can also be dragged while exploring. Default 900. Default: `900`.
- **`umap_sidebar_width`** *(optional)* — (int) - Initial interactive UMAP image and annotation sidebar width in pixels. The divider remains draggable. Default 280. Default: `280`.
- **`black_background`** *(optional)* — (bool) - Choose the standalone/CLI embedding fallback: black canvas with white axes when True, white canvas with black axes when False. In the Qt app, Image UMAP automatically matches its enclosing card in the active theme and uses that theme's readable foreground color instead. Default True. Default: `True`.
- **`save_figure`** *(optional)* — (bool) - Write the embedding, plus the cluster grid when plot_cluster_grids is on, as vector PDFs to &lt;src&gt;/results/&lt;METHOD&gt;_embedding.pdf and &lt;METHOD&gt;_grid.pdf. Enable it when you want the figure for a paper or a record of the run; either way the plots are still displayed on screen. Default False. Default: `False`.

</details>

<details>
<summary>Runtime (3 settings)</summary>

#### Runtime

- **`n_jobs`** *(optional)* — (int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None. Default: `-1`.
- **`ram_guard`** *(optional)* — (bool) - Keep worker pools from filling RAM. Before any module starts its workers, spaCR estimates each worker's memory from one input (a field, mask, crop batch or table) and lowers n_jobs to the count that leaves 12.5% of RAM free, printing a warning; the GUI asks first. Off keeps your n_jobs, and Measure fields still wait while free RAM is below that reserve. Default True. Default: `True`.
- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.

</details>

<details>
<summary>Additional Settings (6 settings)</summary>

#### Additional Settings

- **`pos`** *(optional)* — (str) - Column ID marking positive-control wells in the image UMAP. Rows whose columnID equals it are labelled cond='pos', so exclude_conditions can drop them; and when embedding_by_controls is True the rows whose col_to_compare equals it help fit the reducer. Default 'c1' (note: not 'c2'). Default: `'c1'`.
- **`neg`** *(optional)* — (str) - Column ID marking negative-control wells in the image UMAP. Rows whose columnID equals it are labelled cond='neg', so exclude_conditions can drop them; and when embedding_by_controls is True the rows whose col_to_compare equals it join pos in fitting the reducer. Default 'c2' (note: not 'c1'). Default: `'c2'`.
- **`mix`** *(optional)* — (str) - Plate column ID whose wells hold a mixed positive/negative population; rows with this columnID are labelled cond='mix' for the image UMAP, so they can be coloured separately or dropped via exclude_conditions. Any column matching none of pos, neg or mix is labelled 'screen'. Default 'c3'. Default: `'c3'`.
- **`exclude_conditions`** *(optional)* — (list) - Condition labels dropped from the image UMAP input, matched against the cond column that map_condition derives from the pos, neg and mix column IDs; the only possible entries are 'neg', 'pos', 'mix' and 'screen'. A bare string is accepted and wrapped in a list. Use it to embed screen wells only. Default None. Default: `None`.
- **`embedding_by_controls`** *(optional)* — (bool) - Fit the reducer only on control wells - rows whose col_to_compare value equals pos or neg - and then project every object into that space. Use it when the axes should be defined by the control phenotypes so treatments are read relative to them; False fits on all objects. Default False. Default: `False`.
- **`col_to_compare`** *(optional)* — (str) - Metadata column that identifies the control wells when embedding_by_controls is True: rows whose value equals pos or neg are used to train the reducer, and the column is then dropped before fitting. Typically 'columnID' or 'rowID' depending on where controls sit on the plate. Ignored otherwise. Default 'columnID'. Default: `'columnID'`.

</details>

<details>
<summary>Additional settings (2 settings)</summary>

#### Additional settings

- **`crop_source`** *(optional)* — (str) - Select where image crops come from. Viewers use 'png' (LOAD IMAGES) for exported crops in data/ or 'merged' (STREAM IMAGES) to cut from merged/*.npy using the measurements database. These viewer modes correspond to training's 'load_images' and 'stream_images' sources. spaCR reports any fallback, and controls that do not apply to the selected source are disabled. Default 'png' in viewers and 'load_images' in training. Image UMAP starts at 'auto', preferring exported PNGs when available and otherwise streaming from merged arrays. Default: `'auto'`.
- **`gpu`** *(optional)* — (bool) - Request RAPIDS acceleration for the main dimensionality reduction and Image UMAP hyperparameter search. Controlled by the GPU toggle beside Hyperparameter search; supported for UMAP, t-SNE and PCA, with the actual backend recorded. Default False. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Embedding coordinates and an image-annotated embedding figure.
