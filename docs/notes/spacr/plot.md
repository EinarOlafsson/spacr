# Notes from `spacr/plot.py`

Prose lifted out of `spacr/plot.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (7 entries)
- [_chrome](#_chrome) (2 entries)
- [illegible_data_colours](#illegible_data_colours) (1 entry)
- [print_ready](#print_ready) (1 entry)
- [save_figure](#save_figure) (1 entry)
- [plot_image_mask_overlay.random_color_cmap](#plot_image_mask_overlayrandom_color_cmap) (2 entries)
- [plot_image_mask_overlay._plot_merged_plot](#plot_image_mask_overlay_plot_merged_plot) (4 entries)
- [plot_image_mask_overlay_magenta_outlines.random_color_cmap](#plot_image_mask_overlay_magenta_outlinesrandom_color_cmap) (1 entry)
- [plot_image_mask_overlay_magenta_outlines._plot_merged_plot._generate_colored_mask](#plot_image_mask_overlay_magenta_outlines_plot_merged_plot_generate_colored_mask) (1 entry)
- [plot_image_mask_overlay_magenta_outlines._plot_merged_plot._apply_contours](#plot_image_mask_overlay_magenta_outlines_plot_merged_plot_apply_contours) (1 entry)
- [plot_image_mask_overlay_magenta_outlines._plot_merged_plot](#plot_image_mask_overlay_magenta_outlines_plot_merged_plot) (7 entries)
- [plot_image_mask_overlay_magenta_outlines._filter_object](#plot_image_mask_overlay_magenta_outlines_filter_object) (6 entries)
- [plot_image_mask_overlay_magenta_outlines](#plot_image_mask_overlay_magenta_outlines) (3 entries)
- [plot_cellpose4_output](#plot_cellpose4_output) (2 entries)
- [plot_organelle_output](#plot_organelle_output) (5 entries)
- [plot_masks](#plot_masks) (4 entries)
- [_plot_4D_arrays](#_plot_4d_arrays) (4 entries)
- [_get_colours_merged](#_get_colours_merged) (5 entries)
- [plot_images_and_arrays.find_files](#plot_images_and_arraysfind_files) (1 entry)
- [plot_images_and_arrays.plot_from_file_dict](#plot_images_and_arraysplot_from_file_dict) (4 entries)
- [_filter_objects_in_plot](#_filter_objects_in_plot) (3 entries)
- [plot_arrays](#plot_arrays) (5 entries)
- [_normalize_and_outline](#_normalize_and_outline) (2 entries)
- [_plot_merged_plot](#_plot_merged_plot) (6 entries)
- [plot_merged](#plot_merged) (1 entry)
- [_plot_images_on_grid](#_plot_images_on_grid) (14 entries)
- [_plot_cropped_arrays.plot_single_array](#_plot_cropped_arraysplot_single_array) (1 entry)
- [_plot_cropped_arrays](#_plot_cropped_arrays) (2 entries)
- [_visualize_and_save_timelapse_stack_with_tracks](#_visualize_and_save_timelapse_stack_with_tracks) (5 entries)
- [_visualize_and_save_timelapse_stack_with_tracks._view_frame_with_tracks](#_visualize_and_save_timelapse_stack_with_tracks_view_frame_with_tracks) (6 entries)
- [_display_gif](#_display_gif) (1 entry)
- [_plot_recruitment](#_plot_recruitment) (4 entries)
- [_plot_controls](#_plot_controls) (2 entries)
- [_imshow](#_imshow) (1 entry)
- [_imshow_gpu](#_imshow_gpu) (7 entries)
- [_plot_histograms_and_stats](#_plot_histograms_and_stats) (4 entries)
- [_show_residules](#_show_residules) (3 entries)
- [_reg_v_plot](#_reg_v_plot) (6 entries)
- [generate_plate_heatmap](#generate_plate_heatmap) (13 entries)
- [plot_plates](#plot_plates) (1 entry)
- [print_mask_and_flows.apply_contours_on_image](#print_mask_and_flowsapply_contours_on_image) (1 entry)
- [print_mask_and_flows.normalize_to_uint8](#print_mask_and_flowsnormalize_to_uint8) (2 entries)
- [print_mask_and_flows](#print_mask_and_flows) (6 entries)
- [plot_resize.prepare_image](#plot_resizeprepare_image) (3 entries)
- [plot_resize](#plot_resize) (2 entries)
- [normalize_and_visualize](#normalize_and_visualize) (5 entries)
- [visualize_masks](#visualize_masks) (4 entries)
- [visualize_cellpose_masks](#visualize_cellpose_masks) (4 entries)
- [plot_comparison_results](#plot_comparison_results) (3 entries)
- [plot_object_outlines](#plot_object_outlines) (1 entry)
- [plot_histogram](#plot_histogram) (1 entry)
- [plot_lorenz_curves](#plot_lorenz_curves) (4 entries)
- [plot_lorenz_curves.gini_coefficient](#plot_lorenz_curvesgini_coefficient) (1 entry)
- [plot_permutation](#plot_permutation) (6 entries)
- [plot_feature_importance](#plot_feature_importance) (4 entries)
- [read_and_plot__vision_results](#read_and_plot__vision_results) (7 entries)
- [jitterplot_by_annotation.join_measurments_and_annotation](#jitterplot_by_annotationjoin_measurments_and_annotation) (1 entry)
- [jitterplot_by_annotation](#jitterplot_by_annotation) (8 entries)
- [jitterplot_by_annotation._resolve_well_column](#jitterplot_by_annotation_resolve_well_column) (1 entry)
- [create_grouped_plot](#create_grouped_plot) (30 entries)
- [spacrGraph.preprocess_data](#spacrgraphpreprocess_data) (6 entries)
- [spacrGraph.remove_outliers_from_plot](#spacrgraphremove_outliers_from_plot) (1 entry)
- [spacrGraph.perform_normality_tests](#spacrgraphperform_normality_tests) (3 entries)
- [spacrGraph.perform_statistical_tests](#spacrgraphperform_statistical_tests) (4 entries)
- [spacrGraph.perform_posthoc_tests](#spacrgraphperform_posthoc_tests) (7 entries)
- [spacrGraph.create_plot._generate_tabels](#spacrgraphcreate_plot_generate_tabels) (8 entries)
- [spacrGraph.create_plot._place_symbols](#spacrgraphcreate_plot_place_symbols) (12 entries)
- [spacrGraph.create_plot._get_positions](#spacrgraphcreate_plot_get_positions) (2 entries)
- [spacrGraph.create_plot](#spacrgraphcreate_plot) (8 entries)
- [spacrGraph._draw_comparison_lines](#spacrgraph_draw_comparison_lines) (2 entries)
- [spacrGraph._standerdize_figure_format](#spacrgraph_standerdize_figure_format) (13 entries)
- [spacrGraph._create_bar_plot](#spacrgraph_create_bar_plot) (5 entries)
- [spacrGraph._create_jitter_plot](#spacrgraph_create_jitter_plot) (5 entries)
- [spacrGraph._create_line_graph](#spacrgraph_create_line_graph) (4 entries)
- [spacrGraph._create_line_with_std_area](#spacrgraph_create_line_with_std_area) (6 entries)
- [spacrGraph._create_box_plot](#spacrgraph_create_box_plot) (4 entries)
- [spacrGraph._create_violin_plot](#spacrgraph_create_violin_plot) (4 entries)
- [spacrGraph._create_jitter_bar_plot](#spacrgraph_create_jitter_bar_plot) (5 entries)
- [spacrGraph._create_jitter_box_plot](#spacrgraph_create_jitter_box_plot) (4 entries)
- [spacrGraph._save_results](#spacrgraph_save_results) (2 entries)
- [plot_data_from_db](#plot_data_from_db) (15 entries)
- [plot_data_from_csv](#plot_data_from_csv) (18 entries)
- [plot_image_grid._normalize_image](#plot_image_grid_normalize_image) (5 entries)
- [plot_image_grid](#plot_image_grid) (13 entries)
- [overlay_masks_on_images](#overlay_masks_on_images) (11 entries)
- [graph_importance](#graph_importance) (1 entry)
- [proportions_per_unit](#proportions_per_unit) (2 entries)
- [proportion_test_by_unit](#proportion_test_by_unit) (1 entry)
- [proportion_mixed_model](#proportion_mixed_model) (1 entry)
- [plot_proportion_stacked_bars](#plot_proportion_stacked_bars) (7 entries)
- [create_venn_diagram](#create_venn_diagram) (8 entries)
- [volcano_plot](#volcano_plot) (11 entries)
- [volcano_plot._read_table_auto](#volcano_plot_read_table_auto) (2 entries)
- [volcano_plot._threshold_x_in_plot_units](#volcano_plot_threshold_x_in_plot_units) (1 entry)

## Module level

### lines 32-33

```python
from .errors import RunLedger, raise_if_strict
```

Fail-loud accounting: a missing annotation column silently pools every condition together, which is far worse than a plot that refuses to render.

### lines 38-43

```python
from .figures.style import (ROLES, TYPE_SCALE, WEIGHTS, Palette, descriptor,
```

THE HOUSE STYLE (`spacr/figures/style.py`), applied as a CONTEXT MANAGER around every figure this module builds. A module-level `rcParams.update` would be a process-wide mutation: spaCR draws from a long-lived GUI, so one global write styles every later figure in every other module until the process exits. That failure has already cost this repository a day, and this file alone holds 45 of the ~133 figures spaCR draws.

### lines 73-78

```python
_HOUSE_PANEL_INCHES = 5.6
```

The house type scale is anchored to a single-column DATA panel, about 5.6 inches wide. The image montages in this module are not that: `figuresize` is a panel edge in inches and the figures come out 10 to 40 inches across, so the absolute 7 pt label tier would be a title nobody can read at any size the montage is looked at. The tiers keep their RATIOS and are scaled by the canvas, which is what the skill states them as in the first place.

### line 524, trailing  _(unsure)_

```python
'colourblind': {'cell': '#D55E00',
```

vermillion

### line 525, trailing  _(unsure)_

```python
'nucleus': '#56B4E9',
```

sky blue

### line 526, trailing  _(unsure)_

```python
'pathogen': '#009E73',
```

bluish green

### line 527, trailing  _(unsure)_

```python
'organelle': '#F0E442'},
```

yellow

## _chrome

### lines 201-217

```python
def _chrome(fig, ax=None):
```

A SAVED FIGURE IS FOR PAPER, NOT FOR THE SCREEN (instruction 150).

The DECISION -- which ground, which ink, whether to repaint at all -- is `spacr.figure_style.saved_figure_appearance`, deliberately in a matplotlib- free module so the pyqtgraph exporter can ask the same question and get the same answer. What follows is only the matplotlib APPLICATION of it.

WHY IT CANNOT BE rcParams ALONE, measured rather than assumed. rcParams are read when an artist is CREATED. By the time `save_figure` runs the figure is already drawn, so `rc_context({"text.color": "black"})` changes nothing that is on it. Only `savefig.facecolor`, `savefig.edgecolor` and `savefig.transparent` are read at write time, and those are exactly the three set in the rc block below. The chrome has to be repainted artist by artist and put back, which is what makes "the plot on screen is byte-identical before and after the save" an assertion rather than a hope.

### line 233, trailing  _(unsure)_

```python
for text in fig.texts:
```

suptitle lives here too

## illegible_data_colours

### lines 324-328

```python
return illegible_colours(data_colours(fig), ground, floor)
```

THE JUDGEMENT IS SHARED, ONLY THE HARVEST IS MATPLOTLIB'S. `data_colours` knows how to find a figure's marks; deciding which of them stops working on paper is the same question the pyqtgraph exporter asks of its pens, and a second copy of it here is how the two renderers would come to warn about different colours in the same palette.

## print_ready

### lines 373-377

```python
new = export_colour(current,
```

ONE DECISION, ASKED BY BOTH RENDERERS. `_chrome` says what each artist IS; `export_colour` says what that means for a save, and the pyqtgraph exporter asks the same function of its pens. `_chrome` yields "text" for the artists that are chrome made of letters, which is the same rule as a spine.

## save_figure

### lines 482-485

```python
rc["savefig.facecolor"] = look.ground
```

The three rcParams matplotlib reads at WRITE time rather than at artist-creation time, which is why the rest of the flip cannot be done here. `savefig.facecolor` covers the figure patch; the AXES patch is a separate artist and is repainted by `print_ready`.

## plot_image_mask_overlay.random_color_cmap

### line 630  _(unsure)_

```python
hues = np.linspace(0, 1, n_labels, endpoint=False)
```

Spread colors across hue space, then shuffle so different seeds give different maps

### line 634  _(unsure)_

```python
sats = rng.uniform(0.70, 1.00, size=n_labels)
```

Keep colors vivid and bright so different objects are visually distinct

## plot_image_mask_overlay._plot_merged_plot

### lines 697-700

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 704-706

```python
ax = np.atleast_1d(ax).ravel()
```

The grid is always num_channels + 1 wide, so a single channel still yields a 2-axes array -- the old `if num_channels == 1: ax = [ax]` wrapped that array in a list and made ax[0] the array, not an Axes.

### lines 733-734  _(unsure)_

```python
outline_info = channel_to_outline[current_channel]
```

Membership is tested against this mapping's own key set, and every value is built from a concrete stack plane.

### lines 773-775

```python
combined_mask = np.zeros_like(outlines[0], dtype=np.int64)
```

Priority order is the order in which outlines were added:

cell < nucleus < pathogen < organelle Later objects overwrite earlier ones in overlapping pixels.

## plot_image_mask_overlay_magenta_outlines.random_color_cmap

### line 1041, trailing  _(unsure)_

```python
rand_colors = np.vstack([[0, 0, 0], rand_colors])
```

Ensure background is black

## plot_image_mask_overlay_magenta_outlines._plot_merged_plot._generate_colored_mask

### line 1070, trailing  _(unsure)_

```python
colored_mask[..., 3] = np.where(mask > 0, 1, 0)
```

Alpha channel

## plot_image_mask_overlay_magenta_outlines._plot_merged_plot._apply_contours

### line 1096, trailing  _(unsure)_

```python
continue
```

Skip background

## plot_image_mask_overlay_magenta_outlines._plot_merged_plot

### lines 1104-1107

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 1112-1115

```python
outlines_by_channel = {}
```

The caller constructs every non-None role channel together with its extracted stack plane. Insert in reverse priority so a shared channel retains the old cell > nucleus > pathogen dispatch order.

### line 1146  _(unsure)_

```python
channel_image_rgb = _apply_contours(
```

Use magenta color when all_on_all=False

### line 1155  _(unsure)_

```python
if all_outlines:
```

Channel without associated outlines

### lines 1157-1160

```python
for outline, color in zip(outlines, outline_colors):
```

Apply all outlines with specified colors. The colours must come from outline_colors (as the all_on_all branch above does); a hard-coded list mislabels the objects and silently truncates the zip when a fourth object type is present.

### lines 1174-1176

```python
if len(outlines) > 0:
```

Create an image combining all objects filled with colors. outlines is empty when no object channel was supplied, and outlines[0] then raises IndexError instead of drawing an empty panel.

### line 1195  _(unsure)_

```python
if save_pdf:
```

Save the figure as a PDF

## plot_image_mask_overlay_magenta_outlines._filter_object

### line 1234  _(unsure)_

```python
unique_labels = np.unique(mask_int)
```

Compute properties for each labeled object

### line 1236, trailing  _(unsure)_

```python
unique_labels = unique_labels[unique_labels != 0]
```

Exclude background

### line 1239  _(unsure)_

```python
areas = []
```

Initialize lists to store area and intensity for each object

### line 1252  _(unsure)_

```python
if (min_max_area[0] <= area <= min_max_area[1]) and (min_max_intensity[0] <= mean_intensity <= mi...
```

Check if the object meets both area and intensity criteria

### line 1256  _(unsure)_

```python
areas = np.array(areas)
```

Convert lists to numpy arrays for easier computation

### line 1260  _(unsure)_

```python
avg_area_before = areas.mean() if num_objects_before > 0 else 0
```

Compute average area and intensity before and after filtering

## plot_image_mask_overlay_magenta_outlines

### line 1289  _(unsure)_

```python
if stack.dtype in (np.uint16, np.uint8):
```

Convert to float for normalization and ensure correct handling of arrays

### line 1297  _(unsure)_

```python
cell_outlines = None
```

Define variables to hold individual outlines

### line 1341, trailing  _(unsure)_

```python
percentiles=percentiles,
```

Pass percentiles to the plotting function

## plot_cellpose4_output

### lines 1385-1388

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 1397-1398  _(unsure)_

```python
unique_objects = np.unique(mask)
```

Drop the background label explicitly: [1:] assumes 0 sorts first, so a mask with no background pixel loses a real object.

## plot_organelle_output

### line 1435  _(unsure)_

```python
diag_img, diag_title = _organelle_diagnostic(img, morphology, method, settings)
```

Generate diagnostic image based on morphology/method

### lines 1438-1441

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 1446  _(unsure)_

```python
ax[0].imshow(img, cmap=cmap, interpolation='nearest')
```

Panel 1: Raw image

### line 1450  _(unsure)_

```python
ax[1].imshow(mask, cmap=random_cmap, interpolation='nearest')
```

Panel 2: Label mask

### line 1461  _(unsure)_

```python
ax[2].imshow(diag_img, cmap='viridis', interpolation='nearest')
```

Panel 3: Diagnostic

## plot_masks

### lines 1492-1496

```python
masks = np.asarray(masks)
```

`batch` takes either one image or a stack, so `masks` has to as well (the docstring promises "list or ndarray"). Blindly wrapping made an (N, H, W) stack a single "mask" and imshow died with "Invalid shape (N, H, W) for image data"; a swallowed pytest.skip in tests/test_all_plotting_functions.py hid that for the whole batch path.

### line 1504, trailing  _(unsure)_

```python
flows = [f[0] for f in flows]
```

assuming this is what you want to do when file_type is 'png'

### lines 1519-1522

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 1531-1532  _(unsure)_

```python
unique_objects = np.unique(mask)
```

Drop the background label explicitly: [1:] assumes 0 sorts first, so a mask with no background pixel loses a real object.

## _plot_4D_arrays

### lines 1566-1569

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 1571  _(unsure)_

```python
if num_channels == 1:
```

Create subplots

### line 1574, trailing  _(unsure)_

```python
axs = [axs]
```

Make axs a list to use axs[c] later

### lines 1580-1582

```python
axs[c].set_title(f'Channel {c}',
```

24 pt regardless of the canvas: on the 2-inch panels this function is called with in a browse loop, the title was taller than the image under it.

## _get_colours_merged

### line 1651, trailing  _(unsure)_

```python
outline_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

rgb

### line 1653, trailing  _(unsure)_

```python
outline_colors = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
```

bgr

### line 1655, trailing  _(unsure)_

```python
outline_colors = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
```

gbr

### line 1657, trailing  _(unsure)_

```python
outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
```

rbg

### line 1659, trailing  _(unsure)_

```python
outline_colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
```

rbg

## plot_images_and_arrays.find_files

### line 1726

```python
filtered_dict = {k: v for k, v in file_dict.items() if len(v) == len(folders)}
```

Filter out files that don't have paths in all folders

## plot_images_and_arrays.plot_from_file_dict

### lines 1764-1767

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 1771  _(unsure)_

```python
cmap = random_cmap(num_objects=len(np.unique(mask_data)))
```

Display the mask with random colormap

### line 1777  _(unsure)_

```python
axes[1].imshow(image_data, cmap='gray')
```

Display the normalized image

### line 1785  _(unsure)_

```python
contour[:, 0] += region.bbox[0]
```

Adjust contour coordinates relative to the full image

## _filter_objects_in_plot

### lines 1831-1840

```python
_role_index = {}
```

filter_min_max is in ROLE order -- [cell, nucleus, pathogen] -- while mask_dims is the COMPACTED list of the planes that exist. Indexing one by the other's position only agrees when every role is enabled.

With cell_mask_dim=4, nucleus_mask_dim=None, pathogen_mask_dim=6, mask_dims is [4, 6]: i=0 gave the cell its own range, and i=1 gave the PATHOGEN the nucleus's range. So on any run with a disabled object, one object type was size-filtered by another's limits -- objects removed from the figure that the settings never asked to remove, and objects kept that they did.

### lines 1853-1854  _(unsure)_

```python
min_max = [0, 100000000]
```

A plane that is not one of the three named roles has no declared range. Unfiltered beats borrowing a neighbour's.

### lines 1874-1877

```python
if nuclei_limit is False and nucleus_mask_dim is not None:
```

object_dim must be the dim each flag is named after: the two were swapped, so nuclei_limit=False dropped multi-infected cells and pathogen_limit=False dropped multinucleated ones. The inversion is invisible when both flags are False, which is why it survived.

## plot_arrays

### line 1934, trailing  _(unsure)_

```python
key = list(data.keys())[0]
```

assume first key

### line 1935, trailing  _(unsure)_

```python
img = data[key][0]
```

get first image in batch

### lines 1941-1943

```python
img = normalize_to_dtype(array=img[:, :, np.newaxis], p1=q1, p2=q2)[:, :, 0]
```

normalize_to_dtype indexes array.shape[2], so a single-plane array raises IndexError; promote it for the call and drop the axis again so the 2-D display path below is unchanged.

### lines 1948-1951

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 1957, trailing  _(unsure)_

```python
axs = [axs]
```

ensure iterable

## _normalize_and_outline

### lines 1994-1995

```python
raw_masks = {d: image[:, :, d].copy() for d in mask_dims}
```

`image` is the caller's stack and the remove_background branch mutates it in place, so copy the label planes rather than aliasing them.

### lines 2014-2018

```python
for d, raw in raw_masks.items():
```

Label values are categorical, not intensities. Percentile-rescaling them clips the background up to the lowest label (so that object merges into the background) and collapses a single-object mask to a constant image, from which no contour can be found. Restore the raw labels after the RGB build so the overlay image itself is unchanged.

## _plot_merged_plot

### lines 2054-2057

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 2068  _(unsure)_

```python
for v in range(0, image.shape[-1]):
```

Normalize and plot each channel with outlines

### line 2076  _(unsure)_

```python
for outline, color in zip(outlines, outline_colors):
```

Apply the outlines onto the RGB image

### line 2082  _(unsure)_

```python
ax[v + ax_index].set_title(f'Channel {v + 1}')
```

1-based, human-friendly channel label.

### lines 2089-2090

```python
n_obj = int(len(np.unique(mask)) - 1)   # exclude background 0
```

Name the mask by its object class + live object count, e.g. "Cell Mask - 200 objects".

### line 2091, trailing  _(unsure)_

```python
n_obj = int(len(np.unique(mask)) - 1)
```

exclude background 0

## plot_merged

### lines 2134-2135  _(unsure)_

```python
fig = None
```

nr=0 takes the else-branch on the very first file, so `fig` has to exist before the loop or `return fig` raises UnboundLocalError.

## _plot_images_on_grid

### lines 2215-2221

```python
with figure_style(theme_target()):
```

squeeze=False keeps the return a 2-D array: a single image gives a 1x1 grid, which matplotlib otherwise collapses to a bare Axes with no .flatten(). AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 2223-2227

```python
fig, axes = plt.subplots(int(rows), int(cols), figsize=(20, 20), squeeze=False)
```

THE GROUND COMES FROM THE STYLE, which is transparent: the standing preference is "not black not white just transparent", and this montage carries TEXT -- filenames and channel names -- so a ground baked to black forces the text to white, and white text on a white page is the failure rule 3 exists for.

### line 2231, trailing  _(unsure)_

```python
scale_bar_length_px = int(scale_bar_length_um / um_per_pixel)
```

Convert to pixels

### line 2236  _(unsure)_

```python
if channel_indices is not None:
```

Handle different channel selections

### line 2238, trailing  _(unsure)_

```python
if len(channel_indices) == 1:
```

Single channel (grayscale)

### line 2241, trailing  _(unsure)_

```python
elif len(channel_indices) == 2:
```

Dual channels

### line 2244, trailing  _(unsure)_

```python
else:
```

RGB or more channels

### line 2249  _(unsure)_

```python
if img_array.dtype == np.uint16:
```

Normalize based on dtype

### line 2259  _(unsure)_

```python
ax.plot([10, 10 + scale_bar_length_px], [img_array.shape[0] - 10] * 2, lw=2, color='white')
```

Add scale bar

### line 2261  _(unsure)_

```python
initial_offset = 0.02  # Starting offset from the left side of the figure
```

Add channel names at the top if specified

### line 2262, trailing  _(unsure)_

```python
initial_offset = 0.02
```

Starting offset from the left side of the figure

### line 2263, trailing  _(unsure)_

```python
increment = 0.05
```

Fixed increment for each subsequent channel name, adjust based on figure width

### lines 2267-2271

```python
color = (channel_colors[ci] if ci < len(channel_colors)
```

A channel name takes ITS CHANNEL'S colour -- the skill's rule for a micrograph column header. A channel beyond the three has no colour of its own and falls back to the theme's ink, which was a hard 'white'. The black box behind each name is gone: the style has no other boxes for it to match.

### lines 2278-2280

```python
for j in range(nr_of_images, len(axes)):
```

Pad from the image count, not from a leaked loop variable: the channel_names loop above used to rebind `i`, so the unused cells were blanked starting at the wrong index whenever channel_names was given.

## _plot_cropped_arrays.plot_single_array

### lines 2444-2446

```python
num_objects = int(np.count_nonzero(unique_values))
```

The number of distinct values decides mask vs intensity, but the object count in the title must exclude the background label 0, otherwise a 3-object mask is annotated "4 (obj.)".

## _plot_cropped_arrays

### lines 2455-2458

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 2468-2469

```python
axs = np.atleast_1d(axs)
```

A single channel makes plt.subplots return a bare Axes, not an array, so axs[channel] below would raise TypeError.

## _visualize_and_save_timelapse_stack_with_tracks

### line 2497  _(unsure)_

```python
random_colors = np.random.rand(highest_label + 1, 4)
```

Generate random colors for each label, including the background

### line 2499, trailing  _(unsure)_

```python
random_colors[:, 3] = 1
```

Full opacity

### line 2500, trailing  _(unsure)_

```python
random_colors[0] = [0, 0, 0, 1]
```

Background color

### line 2502  _(unsure)_

```python
norm = plt.cm.colors.Normalize(vmin=0, vmax=highest_label)
```

Ensure the normalization range covers all labels

### lines 2504-2506

```python
geometry = _mask_movie_frame_geometry(masks)
```

The same sizing the saved movie uses: 50 x 50 inches was a whole wall of figure for a mask a few hundred pixels across, and it was redrawn on every tick of the frame slider.

## _visualize_and_save_timelapse_stack_with_tracks._view_frame_with_tracks

### line 2509  _(unsure)_

```python
def _view_frame_with_tracks(frame=0):
```

Function to plot a frame and overlay tracks

### lines 2520-2523

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 2527, trailing  _(unsure)_

```python
ax.imshow(current_mask, cmap=cmap, norm=norm)
```

Apply both colormap and normalization

### line 2530  _(unsure)_

```python
for label_value in np.unique(current_mask):
```

Directly annotate each object with its label number from the mask

### line 2532, trailing  _(unsure)_

```python
if label_value == 0: continue
```

Skip background

### line 2536  _(unsure)_

```python
for track in tracks_df['track_id'].unique():
```

Overlay tracks

## _display_gif

### lines 2568-2575

```python
with open(path, 'rb') as file:
```

`format='gif'` is stated rather than sniffed. IPython only learned to recognise the GIF87a/GIF89a magic bytes in 9.0.0; before that, raw bytes with no format fall through to 'png' and the animation is emitted with an `image/png` mime type. IPython 9 needs Python 3.11, so on the 3.9 and 3.10 ends of the range spaCR claims there is no version of IPython that would guess right -- setup.py's `IPython>=8.18.1` resolves to exactly 8.18.1 on 3.9. Saying what the file is costs nothing and is correct on every version.

## _plot_recruitment

### lines 2603-2613

```python
with mpl.rc_context({'axes.prop_cycle': mpl.cycler(color=color_list)}), \
```

The palette is set for THIS figure only. `sns.set_palette` writes matplotlib's global colour cycle, so drawing one recruitment plot used to recolour every plot the session drew afterwards -- including figures on other screens, and including the palette the user chose in figure preferences. `rc_context` keeps the colours while these axes are built and puts the cycle back when they are done. The four hues stay: here the CATEGORY IS THE DATA -- one colour per pathogen strain, held across all four panels -- which is the one case the style allows a categorical palette. Everything else about the figure is the house style, applied around the palette rather than instead of it.

### lines 2637-2640

```python
handles, labels = axes[3].get_legend_handles_labels()
```

axes[0].legend_.remove() axes[1].legend_.remove() axes[2].legend_.remove() axes[3].legend_.remove()

### lines 2646-2647  _(unsure)_

```python
rotate_ticks(axes[i])
```

Right-aligned and anchored, not merely rotated: a condition name rotated about its centre drifts off the tick it belongs to.

### lines 2678-2679

```python
hide_unused(axes[len(columns):])
```

An empty framed box reads as a panel that failed to draw, which is worse than a gap.

## _plot_controls

### lines 2722-2726

```python
color_list = [ROLES['data']] * 4
```

The four components are ALREADY the x axis of every panel, so colouring them a second time argues nothing -- this is colour-by-category where the category is not the point, which the style names as the failure. One grey for all four; the panels differ by condition and channel, and those are what the descriptors say.

### lines 2735-2738

```python
names = []
```

Build labels and colours alongside the data. The bar call used to pass all four component names unconditionally while the guard above skips missing columns, so any absent component (or a whole absent channel) made x and height different lengths and raised.

## _imshow

### lines 2786-2789

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

## _imshow_gpu

### line 2814, trailing  _(unsure)_

```python
img = img.cpu()
```

Move to CPU if the tensor is on GPU

### line 2820, trailing  _(unsure)_

```python
img_height = img.shape[2]
```

Height of the image

### line 2821, trailing  _(unsure)_

```python
img_width = img.shape[3]
```

Width of the image

### line 2823  _(unsure)_

```python
canvas = torch.zeros((img_height * n_row, img_width * n_col, 3))
```

Prepare the canvas on CPU

### line 2830  _(unsure)_

```python
canvas[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = img[idx].permute...
```

Place the image on the canvas

### line 2833, trailing  _(unsure)_

```python
canvas = canvas.numpy()
```

Convert to NumPy for plotting

### lines 2835-2838

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

## _plot_histograms_and_stats

### line 2860  _(unsure)_

```python
mean_pred = subset['pred'].mean()
```

Calculate the statistics

### line 2865  _(unsure)_

```python
print(f"Condition: {condition}")
```

Print the statistics

### lines 2875-2878

```python
with figure_style(theme_target()):
```

The distribution is the subject, so it carries the one fill colour the house style keeps for distributions; the mean is a REFERENCE and is drawn as one -- thin, dashed, grey. It was a bold red line, which reads as the finding rather than as the ruler you measure it with.

### lines 2884-2886

```python
mean_line.set_label(f"Mean = {mean_pred:.2f}")
```

The value stays in the legend rather than becoming a rotated in-plot annotation: a caller asserts on that legend entry, and a frameless legend is already what the style draws.

## _show_residules

### lines 2915-2920

```python
qq_fig, qq_ax = plt.subplots()
```

QQ plot. It gets its OWN axes, explicitly: `sm.qqplot` creates a figure and leaves ITS axes current, so the residuals-vs-fitted scatter below was landing on top of the QQ panel and its `set_title` was overwriting 'QQ Plot'. Measured on a 60-point OLS fit: two figures came back, not three, and the second held both diagnostics superimposed.

### lines 2923-2926

```python
for line in qq_ax.lines:
```

Recoloured after the fact rather than through plotkwargs: qqplot passes its own 'b' format string alongside them and matplotlib warns that the two disagree. The points are data (grey); the 45-degree line is a REFERENCE, and qqplot draws it bold red.

### line 2939  _(unsure)_

```python
resid_fig, resid_ax = plt.subplots()
```

Residuals vs. Fitted values

## _reg_v_plot

### lines 2959-2961

```python
df['-log10(p)'] = -np.log10(df['p'])
```

grouping/variable/plate_number are unused by the body but kept for call-site compatibility; they default so utils.MLR's `_reg_v_plot(df)` call works instead of raising TypeError.

### lines 2964-2969

```python
called = np.asarray(df['p'] < 0.05)
```

THE ONE RULE: everything grey except what the sentence is about. This volcano used to run `cmap='coolwarm'` over `np.sign(effect)`, which colours EVERY point by a fact the x-axis already states -- the exact failure the rule exists to prevent. Only the called genes (p < 0.05, the same rows that get a label) carry colour now, GREEN up and RUST down, and every other gene is the grey they are compared against.

### lines 2975-2977

```python
with figure_style(theme_target()):
```

40x30 inches was a poster, not a panel: at the 300 dpi the save preference asks for, that canvas is 12000x9000 px -- 108 megapixels for a scatter of a few thousand dots.

### line 2986  _(unsure)_

```python
for idx, row in df.iterrows():
```

Add text for specified points

### line 2988, trailing

```python
if row['p'] < 0.05:
```

and abs(row['effect']) > 0.1:

### line 2993

```python
reference_line(ax, y=-np.log10(0.05))
```

line for p=0.05

## generate_plate_heatmap

### lines 3069-3075

```python
prc_text = df['prc'].astype(str)
```

read the well out of prc prc is <plate>_<row>_<column>, read right to left: the last two tokens are the position and whatever precedes them is the plate. Left-to-right unpacking put the *row* in the plate slot for any identifier carrying an experiment prefix, and only ``prc.iloc[0]`` was ever probed for its length, so a frame mixing 3- and 4-token identifiers misaligned every row of the minority shape.

### lines 3078-3081

```python
plate_token = np.array(
```

A longer identifier carries an experiment prefix; the plate the caller asked for is then the authority on which plate it is, which is what the old 4-part rebuild did. Too short and there is no position at all — the rows are kept here precisely so they can be reported below.

### lines 3088-3089

```python
df = df.copy()
```

A rebuilt identifier must not be written back onto the caller's frame. The plain 3-token path always has done, and is pinned.

### lines 3092-3097

```python
row_index, row_label = _well_axis_labels(
```

THE WELL COMES FROM prc, AND ONLY FROM prc. A frame that also carries 'plate'/'plate_name' or 'column'/'column_name' columns is drawn from the identifier all the same, so two spellings of the same well can never place it in two different squares. Those columns held a copy that this function overwrites from `plate_token` and the axis labels below in every case, so reading them was a second answer nobody ever saw.

### lines 3105-3107

```python
on_plate = np.asarray(plate_token == str(plate_number), dtype=bool)
```

filter one plate, and say what could not be drawn dtype=bool explicitly: an empty frame gives an empty float array, and `~` on a float array is a TypeError rather than "nothing to report".

### lines 3125-3127

```python
df['_row_index'] = row_index[keep].astype(int)
```

Group on the integer position, not on the label: 'c10' sorts before 'c2' as text, and a Categorical of hard-coded labels was what silently deleted rows past P in the first place.

### line 3132  _(unsure)_

```python
df['_well_count'] = df.groupby(
```

Optional min_count filter on true per-well counts

### line 3142, trailing  _(unsure)_

```python
plate = grouped.size().reset_index(name='value')
```

per-well row counts

### line 3146, trailing  _(unsure)_

```python
vals = pd.to_numeric(df[variable], errors='coerce')
```

ensure numeric

### line 3151, trailing  _(unsure)_

```python
else:
```

sum

### line 3159  _(unsure)_

```python
plate_map.index = pd.Index([_schema.row_id(int(i)) for i in plate_map.index],
```

Back to the ids the rest of spaCR speaks, in numeric order.

### lines 3165-3167

```python
if plate_map.values.size == 0:
```

vmin/vmax selection. Guard against an empty pivot (e.g. a tiny plate where every well was filtered out): np.quantile / np.nanmin on a zero-size array raises, so fall back to a neutral [0, 1] range.

### line 3182

```python
if vmin == vmax:
```

avoid degenerate colormap

## plot_plates

### lines 3263-3271

```python
from .figures.plates import plate_figure_name
```

NAMED FOR WHAT IT DRAWS, and rewritten in place. The old loop searched for the first free `plate_heatmap_<n>.pdf` and never overwrote, so the real screen's results folder holds twelve byte-identical copies of one figure from twelve runs -- and the figure grid showed all twelve. A run gets its own folder (`ml._next_results_folder`), so one file per measurement in it is the whole of what belongs there. The old loop also tested for a `.pdf` that `save_figure` may well write as `.png`, in which case it never found its own previous output at all.

## print_mask_and_flows.apply_contours_on_image

### line 3346  _(unsure)_

```python
image = normalize_to_uint8(image)
```

The sole caller reduces every accepted stack to a 2-D base plane.

## print_mask_and_flows.normalize_to_uint8

### line 3366, trailing  _(unsure)_

```python
image = np.clip(image, 0, 1)
```

Ensure values are between 0 and 1

### line 3367, trailing  _(unsure)_

```python
return (image * 255).astype(np.uint8)
```

Convert to uint8

## print_mask_and_flows

### line 3370  _(unsure)_

```python
stack = resize_if_needed(stack, max_size)
```

Resize if necessary

### lines 3373-3376

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 3388  _(unsure)_

```python
if stack.ndim == 2:
```

Display original image

### line 3392, trailing  _(unsure)_

```python
original_image = stack[..., 0]
```

Use the first channel as the base

### line 3412  _(unsure)_

```python
if flows and isinstance(flows, list) and flows[0].ndim in [2, 3]:
```

Display flow image or its first channel

### line 3416, trailing  _(unsure)_

```python
flow_image = flow_image[:, :, 0]
```

Use first channel for 3D

## plot_resize.prepare_image

### line 3455, trailing  _(unsure)_

```python
return img, None
```

RGB

### line 3457, trailing  _(unsure)_

```python
return img, None
```

RGBA

### line 3459  _(unsure)_

```python
return np.mean(img, axis=-1), 'gray'
```

fallback: average across channels to show as grayscale

## plot_resize

### lines 3464-3467

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 3481  _(unsure)_

```python
lbl, cmap = prepare_image(labels[0])
```

Labels (assumed grayscale or single-channel)

## normalize_and_visualize

### lines 3503-3506

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 3509, trailing  _(unsure)_

```python
if image.ndim == 3:
```

Multi-channel image

### line 3510, trailing  _(unsure)_

```python
ax[0].imshow(np.mean(image, axis=-1), cmap='gray')
```

Display the average over channels for visualization

### line 3511, trailing  _(unsure)_

```python
else:
```

Grayscale image

### line 3517, trailing  _(unsure)_

```python
ax[1].imshow(np.mean(normalized_image, axis=-1), cmap='gray')
```

Similarly, display the average over channels

## visualize_masks

### lines 3534-3537

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### lines 3540-3541

```python
for ax, mask, panel_title in zip(axs, [mask1, mask2, mask3], ['Mask 1', 'Mask 2', 'Mask 3']):
```

The loop variable must not be named `title`: it shadowed the parameter, so the suptitle below always read 'Mask 3' instead of the caller's title.

### line 3544  _(unsure)_

```python
if np.isin(mask, [0, 1]).all():
```

If the mask is binary, we can skip normalization

### line 3548  _(unsure)_

```python
norm = plt.Normalize(vmin=0, vmax=mask.max())
```

Normalize the image for displaying purposes

## visualize_cellpose_masks

### lines 3601-3604

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 3607, trailing  _(unsure)_

```python
fig, axs = plt.subplots(1, num_masks, figsize=(10 * num_masks, 10))
```

Adjusting figure size dynamically

### lines 3608-3609

```python
axs = np.atleast_1d(axs)
```

A single mask makes plt.subplots return a bare Axes, which zip() below cannot iterate.

### line 3614  _(unsure)_

```python
norm = plt.Normalize(vmin=0, vmax=mask.max())
```

Normalize and display the mask

## plot_comparison_results

### lines 3649-3652

```python
panels = (
```

Four metrics of the same comparison, so no one panel is the claim and nothing here is coloured. The points are the data and the box is the summary drawn under them: GREY for the box, the darker grey for the marks, opaque -- overplotting is handled by point size, not by alpha.

### lines 3670-3671

```python
rotate_ticks(ax)
```

45 degrees, right-aligned and anchored: a comparison name is a pair of filenames and runs off the panel at any other angle.

### lines 3675-3676  _(unsure)_

```python
panel_letter(ax, 'ABCD'[index])
```

A four-panel figure is a figure sheet, and a sheet is read by its letters.

## plot_object_outlines

### lines 3710-3711

```python
max_nr=max_nr,
```

Forward the caller's cap; the literal 10 made the documented max_nr parameter dead.

## plot_histogram

### lines 3724-3727

```python
with figure_style(theme_target()):
```

A distribution is filled with the one pale hue the published figures keep for distributions and densities, solid. The old saturated teal at alpha 0.6 was the pattern the style names as wrong: overplotting is handled by a pale fill, not by making a strong colour translucent.

## plot_lorenz_curves

### lines 3764-3766

```python
if remove_keys is None:
```

remove_keys got the same mutable-default -> None treatment as x_lim/y_lim but never got the matching guard, so the documented default call died on `for remove in None`.

### lines 3847-3851

```python
entries = []
```

THE SENTENCE THIS FIGURE MAKES is "the library as a whole is this uneven"; the individual plates are the comparison it is made against. So the plates are grey and only the combined curve is coloured. The old figure gave every plate its own cycle colour and drew the combined curve in black -- eight arguments and no claim.

### line 3859  _(unsure)_

```python
for remove in remove_keys:
```

Remove specified keys

### lines 3897-3899

```python
text_legend(ax, entries)
```

Coloured text, no frame and no marker swatches: the curve labels already carry the Gini, and a framed box would be the only box in the figure.

## plot_lorenz_curves.gini_coefficient

### lines 3808-3810

```python
gini = 1 - np.sum((cumulative_data[:-1] + cumulative_data[1:]) * np.diff(np.linspace(0, 1, n + 1)))
```

Trapezoid rule, not a left-Riemann sum: taking only the left endpoint under-counts the area by exactly 1/n, so a perfectly equal distribution reported 1/n instead of 0.

## plot_permutation

### line 3924, trailing  _(unsure)_

```python
fig_height = max(8, num_features * 0.3)
```

Set a minimum height of 8 and adjust height based on number of features

### line 3925, trailing  _(unsure)_

```python
fig_width = 10
```

Width can be fixed or adjusted similarly

### line 3926, trailing  _(unsure)_

```python
font_size = max(10, 12 - num_features * 0.2)
```

Adjust font size dynamically

### lines 3928-3931

```python
with figure_style(theme_target()):
```

The house type scale is anchored to a single-column panel. This canvas is not one: it grows to 0.3 inch per feature, so a 100-feature figure is 30 inches tall and the 7 pt label tier would be unreadable on it. The measured dynamic size stays; everything else is the house style.

### lines 3934-3936

```python
ax.barh(permutation_df['feature'], permutation_df['importance_mean'],
```

Grey, opaque. A ranking IS the claim -- no single bar of it is so nothing is singled out with colour, and the teal at alpha 0.6 was a saturated hue made translucent, which the style forbids.

### lines 3941-3943

```python
if float(np.nanmin(np.asarray(
```

A permutation importance below zero means shuffling the feature made the model BETTER. Without a zero rule you cannot see which bars cross it, so it is drawn -- as a reference, only when it is needed.

## plot_feature_importance

### line 3966, trailing  _(unsure)_

```python
fig_height = max(8, num_features * 0.3)
```

Set a minimum height of 8 and adjust height based on number of features

### line 3967, trailing  _(unsure)_

```python
fig_width = 10
```

Width can be fixed or adjusted similarly

### line 3968, trailing  _(unsure)_

```python
font_size = max(10, 12 - num_features * 0.2)
```

Adjust font size dynamically

### lines 3970-3974

```python
with figure_style(theme_target()):
```

Same reasoning as plot_permutation: the dynamic label size is kept because the canvas is not a single-column panel, the rest is the house style. The bars were solid blue at alpha 0.6 -- BLUE is the palette's highlight and means "the one thing being argued about", which is the opposite of what a whole ranking of bars is.

## read_and_plot__vision_results

### line 4001  _(unsure)_

```python
if y_lim is None:
```

List to store data from all CSV files

### lines 4007-4008  _(unsure)_

```python
os.makedirs(dst, exist_ok=True)
```

os.mkdir has no `exists` kwarg (nor `exist_ok`); the old call raised TypeError on every invocation, before any file was ever read.

### line 4011  _(unsure)_

```python
for root, dirs, files in os.walk(base_dir):
```

Walk through the directory

### line 4016  _(unsure)_

```python
file_name = os.path.basename(file_path)
```

Extract model information from the file name

### lines 4020-4023

```python
base_folder = os.path.dirname(file_path)
```

The epoch comes from the directory name below; the dropped

`file_name.split('_time')[1]` hard-coded the separator instead of using name_split and its result was never read, so it only ever raised IndexError on non-default naming.

### line 4039  _(unsure)_

```python
avg_metric = result_df.groupby(
```

Calculate average y_axis per model

### lines 4045-4048

```python
colours = [ROLES['data']] * len(avg_metric)
```

Plotting the results. THE SENTENCE IS "this model scored best", and the rows are already sorted ascending, so the last bar is the one the figure is about and the rest are the comparison it is made against. One highlight out of N, never a bar per cycle colour.

## jitterplot_by_annotation.join_measurments_and_annotation

### lines 4108-4113

```python
merged_df = pd.merge(df, paths_df[0], on='prcfo', how='left',
```

one_to_one: _read_and_merge_data returns one row per object keyed on 'prcfo', and png_list carries at most one crop per that key. A duplicated 'prcfo' in png_list (a crop step that ran twice, or two crop_modes whose object labels collide) would multiply the measurement rows, and the jitter plot would then draw the same cell two or four times as if they were independent observations.

## jitterplot_by_annotation

### line 4118  _(unsure)_

```python
df = join_measurments_and_annotation(src, tables=['cell', 'nucleus', 'pathogen', 'cytoplasm'])
```

Read the CSV file into a DataFrame

### line 4121  _(unsure)_

```python
print(f"Generated dataframe with: {df.shape[1]} columns and {df.shape[0]} rows")
```

Print column names for debugging

### line 4125  _(unsure)_

```python
df[x_column] = df[x_column].fillna('NaN')
```

Replace NaN values with a specific label in x_column

### line 4161  _(unsure)_

```python
min_count = retained_rows[x_column].value_counts().min()
```

Determine the minimum count of examples across all groups in x_column

### lines 4170-4175

```python
groups = list(pd.unique(balanced_df[x_column]))
```

Create the jitter plot. The annotation classes are the X AXIS, so painting them a second time with a viridis ramp said nothing the axis had not already said -- and a sequential colormap over unordered categories implies an order that is not there. Every point is grey; the hue split is kept only because it gives each class its own collection, which is how callers address a class's points.

### lines 4183-4185

```python
for index, group in enumerate(groups):
```

A dot strip without its mean is a cloud. GREY_DARK is the palette's own role for a mean bar, and the bar is drawn as a Line2D so a caller counting per-class collections still counts classes.

### lines 4196-4198

```python
rotate_ticks(ax)
```

Customize the x-axis labels. Right-aligned and anchored: the old code rotated to 45 degrees and then re-centred them, so every label drifted off the tick it belonged to.

### line 4201  _(unsure)_

```python
if output_path:
```

Save the plot to a file or display it

## jitterplot_by_annotation._resolve_well_column

### lines 4137-4142

```python
def _resolve_well_column(frame, *bases):
```

Resolve the well-identifier columns instead of hard-coding plate_x/row_x/ col_x: spacr.io emits plateID/rowID/columnID, so those literals never match a current database and every call raised KeyError. The merge on 'prcfo' collides on all three, hence the _x/_y suffixes; the bare and _y forms are tried too so the lookup survives a non-colliding merge, and the pre-rename names stay accepted for older frames.

## create_grouped_plot

### lines 4267-4269

```python
from .figures.stats import _clean, check_normality, compare
```

The engine is imported inside the function, as sp_stats does it and for the same reason: `spacr.figures` eagerly builds the panel catalog, and `spacr.plot` is imported by callers that never draw a grouped plot.

### line 4273  _(unsure)_

```python
df = df.dropna(subset=[grouping_column])
```

Remove NaN rows in grouping_column

### line 4280  _(unsure)_

```python
if order:
```

Sorting and ordering

### line 4289  _(unsure)_

```python
test_results = []
```

Initialize test results

### lines 4292-4297

```python
grouped_data = {group: _clean(df.loc[df[grouping_column] == group,
```

Test normality for each group. The check is the one engine's, so this function, spacrGraph and a sp_stats results table cannot disagree about the same plate. It used to be D'Agostino per group with `all(p > 0.05)`, which reads "the test had no power to reject" as "the data are normal" -- and normaltest needs eight observations before it can say anything at all.

### lines 4303-4305

```python
for group, values in grouped_data.items():
```

Add normality test results to the results_df. 'Normality test' is the ROW TYPE, not the name of a test: the schema here is four fixed columns and the check's own name lives in spacrGraph's richer table.

### lines 4315-4330

```python
from .figures.stats import MIN_N_FOR_TEST
```

Perform pairwise statistical tests. EACH PAIR IS A TWO-GROUP

COMPARISON and is now named as one. This used to pick a single test from the group count and apply it to every pair, so three normal groups produced three rows labelled 'One-way ANOVA' that were each an ANOVA across two groups -- arithmetically a t-test, reported under a name nobody could act on. A CONTINUOUS COLUMN IS NOT A GROUPING. Asked to compare by a column of measurements, every "group" holds one observation, every pair is untestable, and the pair count is quadratic -- which is how pressing Rank produced thousands of lines reading

0.735573 vs 0.778142: these groups have fewer than 2 usable observations and cannot be tested

Refused at the door, naming the column, rather than discovered one impossible pair at a time.

### lines 4353-4359

```python
untestable.append((group1, group2, str(refusal)))
```

A group too small to test. Reported rather than raised: the caller wants the figure and the other pairs.

COLLECTED, NOT PRINTED HERE. One line per impossible pair is not a report -- it is the same sentence a quadratic number of times, and it buries whatever else the run said. Summarised once after the loop.

### lines 4372-4374

```python
if untestable:
```

ONE LINE FOR ALL OF THEM. The pairs that could not be tested are in `test_results` either way, named and marked 'not testable', so nothing is hidden -- what is not repeated is the sentence explaining why.

### lines 4376-4378

```python
thin = sorted(group for group, values in grouped_data.items()
```

THE THIN GROUPS, not every group that appears in a failed pair. A pair fails because ONE side is too small, and naming both makes a healthy group look like the problem.

### lines 4388-4389

```python
test_name = ', '.join(dict.fromkeys(chosen_names)) or 'not testable'
```

The title names every test that ran, because with the choice made per pair they need not all be the same one.

### line 4392  _(unsure)_

```python
if is_normal and len(unique_groups) > 2:
```

Post-hoc test (Tukey HSD for ANOVA)

### line 4398, trailing  _(unsure)_

```python
'Test Statistic': None,
```

Tukey does not provide a test statistic in the same way

### lines 4403-4409

```python
with figure_style(theme_target()):
```

Create plot. `figure_style` is the same kind of scope the old `mpl.rc_context()` was `sns.set` writes a whole seaborn theme (style, context, palette, fonts) into matplotlib's process-wide rcParams, so a grouped plot used to decide how every later figure of the session looked. It also brought a GRID, which the house style does not have at all: a grid is the fastest way to make a panel look like a spreadsheet.

### lines 4416-4419

```python
color_palette = [ROLES['data']] * len(unique_groups)
```

`sns.color_palette("husl", n)` is a rainbow across the groups, and the groups are the x axis. The comparison between them is what the test above reports; the bars themselves are not the argument, so they are the one grey.

### line 4422  _(unsure)_

```python
if graph_type == 'bar':
```

Choose graph type

### line 4441  _(unsure)_

```python
plt.errorbar(x=np.arange(len(summary_df)), y=summary_df[summary_func], yerr=error_bars, fmt='none...
```

Add error bars (standard deviation or standard error of the mean)

### lines 4458-4459  _(unsure)_

```python
_ink = resolve_ink(theme_target())
```

The ink follows the theme, like every sibling branch here a hard-coded black is invisible axes on the dark theme.

### lines 4461-4465

```python
sns.boxplot(
```

THE BOX IS A REFERENCE OVER THE POINTS, NOT A BLOCK COMPETING WITH THEM. Instruction 139 B and the house rule it follows: the points are the data and carry the ink; the box summarises. A filled box per group is a rainbow behind a dot strip, and the reader's eye goes to the fill rather than to the observations.

### lines 4475-4478

```python
sns.stripplot(x=grouping_column, y=data_column, data=df,
```

OUTLIERS OFF ON THE BOX, and that is not hiding them: the strip below draws EVERY observation, so seaborn's own flier markers would double-plot the extreme points and only those -- which reads as the tails being twice as dense as they are.

### lines 4483-4485

```python
_ink = resolve_ink(theme_target())
```

THE BAR IS THE SUMMARY, THE POINTS ARE THE DATA -- the same rule as jitter_box above, so the fill is dropped and the observations carry the ink.

### lines 4504-4507

```python
_ink = resolve_ink(theme_target())
```

A LINE ACROSS THE GROUPS. One data column means there is no second column to put on x, so the group is the x axis and the point on each group is the same summary the bar chart would draw -- the two pictures agree about the data.

### lines 4524-4528

```python
else:
```

THE BRANCH CHAIN COVERS EVERY TYPE THE MENU OFFERS. `line` and `jitter_bar` had no branch at all: they fell through it, drew nothing, and `plt.gcf()` below handed back an EMPTY figure. Two of the seven entries in the right-click Graph type menu blanked the plot and reported no error.

### line 4534  _(unsure)_

```python
results_df = pd.DataFrame(test_results)
```

Create a DataFrame to summarize the test results

### line 4537  _(unsure)_

```python
if isinstance(y_lim, list) and len(y_lim) == 2:
```

Set y-axis start if provided

### lines 4541-4545

```python
axis = plt.gca()
```

THE FIGURE ON SCREEN IS THE FIGURE ON DISK. The title naming the test and the rotated group labels used to be applied inside the `save` branch only, so a user who looked at the plot saw an untitled one with horizontal labels and a user who saved it got a different picture out of the same call.

### line 4551  _(unsure)_

```python
if save:
```

If save is True, save the plot and the results table

### lines 4553-4556

```python
plot_path = os.path.join(output_dir, 'grouped_plot')
```

No extension: `save_figure` appends the one the figure-format preference selects. Naming the file .png here and then writing a PDF into it was the old behaviour, and it is a file no viewer opens.

### line 4566  _(unsure)_

```python
plt.show()
```

Show the plot

### lines 4570-4579

```python
try:
```

THE RECIPE TRAVELS WITH THE FIGURE (178 A). "i should be able to right click on them and show them as: line, bar, jitter-bar, jitter-box, jitter, box, violin" -- which means something has to be able to draw the SAME data a different way, and the figure is the only thing the right-click menu has a reference to.

The frame is kept rather than re-read from the summary CSV beside it: that file is already aggregated to whatever level the plot used, so a jitter rebuilt from it would draw the means and call them cells.

## spacrGraph.preprocess_data

### line 4761  _(unsure)_

```python
df = self.df.dropna(subset=[self.grouping_column] + self.data_column)
```

1) Remove NaNs in both the grouping column and each data column

### line 4764  _(unsure)_

```python
if self.representation == 'object':
```

2) Decide how to handle grouping based on 'representation'

### lines 4766-4767  _(unsure)_

```python
group_cols = None
```

No grouping at all

We do nothing except keep df as-is after removing NaNs

### line 4775  _(unsure)_

```python
if 'plateID' not in df.columns:
```

Make sure 'plateID' exists (split from 'prc' if needed)

### line 4799  _(unsure)_

```python
if self.order and (self.grouping_column in df.columns):
```

4) Handle ordering if specified (and if the grouping_column still exists)

### lines 4807-4808  _(unsure)_

```python
df[self.grouping_column] = pd.Categorical(
```

The initial dropna and every aggregation require this column, so it is still present when no explicit order was supplied.

## spacrGraph.remove_outliers_from_plot

### lines 4819-4823

```python
filtered_df = self.df.copy()
```

self.data_column is a list, so the old code indexed with it and got a DataFrame: the bounds came out as per-column Series and the mask as a DataFrame, which cannot be combined with the group Series. Work one scalar column at a time, and collect the rows to drop instead of dropping inside the loop (that invalidates the group mask's index).

## spacrGraph.perform_normality_tests

### lines 4882-4885

```python
print(f"Skipping normality test for group '{group}' on "
```

Shapiro-Wilk needs three points to have a statistic at all. A constant group is NOT skipped any more: the engine reports it as "no spread to describe", which is a verdict, where 'Skipped' was silence.

### lines 4913-4916

```python
column_verdicts.append(
```

The verdict is the engine's own, taken across the groups together -- never re-derived from the per-group p-values above, because that would throw away the Bonferroni correction the check applies across groups.

### lines 4920-4922

```python
is_normal = bool(column_verdicts) and all(column_verdicts)
```

No column examined is not evidence of normality. `all([])` is True, and returning True there would license a parametric test off an empty call.

## spacrGraph.perform_statistical_tests

### line 4998, trailing  _(unsure)_

```python
for column in self.data_column:
```

Iterate over each data column

### lines 5004-5018

```python
first, second = arrays
```

THE ENGINE HAS NO GUARD FOR THIS ONE, so spacrGraph refuses on its behalf. `compare(paired=True)` hands the matched arrays straight to scipy, and when every pair differs by the same amount the standard error of the difference is zero: `ttest_rel` returns t = -inf and p = 0.0 with a RuntimeWarning. A figure would then carry p = 0 -- the strongest claim the software can make -- off an input that says nothing. All-identical arms break the signed-rank test the same way, with a NaN.

This refuses one case the signed-rank test could survive:

differences that are all the SAME non-zero number still carry sign information. Two arms matched to the last bit of a float is a synthetic input, and refusing it is the safe direction.

### lines 5033-5037

```python
refusal = str(engine_refusal)
```

Fewer than two groups, or a group too small to test. Refusing is the engine's design: a comparison that could not be made is not a comparison with an unknown answer. Reported as a row rather than raised, because the caller is drawing a figure and wants the other columns.

### lines 5065-5076

```python
'n_object': sum(
```

n_object FROM raw_df, n_well from self.df. Both used to come from `grouped_data`, which is built from self.df -- and self.df is what `preprocess_data` AGGREGATED. With representation='well' that made the two columns the same number: a plate of 4,382 cells in 12 wells reported n_object = 12.

The post-hoc rows in the same CSV already did it correctly

(n_object from raw_df, n_well from self.df), so the two row types disagreed about the same comparison in the same file which is how you get a Methods section citing whichever was read first.

## spacrGraph.perform_posthoc_tests

### line 5127, trailing  _(unsure)_

```python
'Test Statistic': None,
```

Tukey does not provide a test statistic

### line 5137  _(unsure)_

```python
long_data = self.df[[self.data_column[0], self.grouping_column]].dropna()
```

Prepare data for Dunn's test in long format

### line 5142  _(unsure)_

```python
dunn_result = sp.posthoc_dunn(
```

Perform Dunn's test with Bonferroni correction

### line 5156, trailing  _(unsure)_

```python
'Test Statistic': None,
```

Dunn's test does not return a specific test statistic

### line 5157, trailing  _(unsure)_

```python
'p-value': dunn_result.iloc[group_a, group_b],
```

Extract the p-value from the matrix

### line 5160, trailing  _(unsure)_

```python
'n_object': len(raw_data1) + len(raw_data2),
```

Total objects

### lines 5161-5163

```python
'n_well': len(self.df[self.df[self.grouping_column] == dunn_result.index[group_a]]) +
```

Both terms must index the frame with the mask. Without the outer self.df[...] the second term is the mask itself, so its len() is the row count of the whole frame.

## spacrGraph.create_plot._generate_tabels

### line 5192  _(unsure)_

```python
table_data = []
```

Initialize table data

### line 5195  _(unsure)_

```python
grouping_row = []
```

Create the grouping row: Alternate each group for every data column

### line 5202  _(unsure)_

```python
for column in self.data_column:
```

Create symbol rows for each data column

### line 5204, trailing  _(unsure)_

```python
column_row = []
```

Initialize a row for this column

### line 5205, trailing  _(unsure)_

```python
for data_col in self.data_column:
```

Iterate over data columns to align with the structure

### line 5207

```python
if column == data_col:
```

Assign '+' if the column matches, otherwise assign '-'

### line 5212, trailing  _(unsure)_

```python
table_data.append(column_row)
```

Add this row to the table

### line 5214  _(unsure)_

```python
transposed_table = list(map(list, zip(*table_data)))
```

Transpose the table to align with the plot layout

## spacrGraph.create_plot._place_symbols

### line 5229  _(unsure)_

```python
y_axis_min = ax.get_ylim()[0]  # Minimum y-axis value (usually 0)
```

Get plot dimensions and adjust for different plot sizes

### line 5230, trailing  _(unsure)_

```python
y_axis_min = ax.get_ylim()[0]
```

Minimum y-axis value (usually 0)

### line 5231, trailing  _(unsure)_

```python
symbol_start_y = y_axis_min - 0.05 * (ax.get_ylim()[1] - y_axis_min)
```

Adjust a bit below the x-axis

### line 5233  _(unsure)_

```python
y_spacing = 0.04  # Adjust this for better spacing between rows
```

Calculate spacing for the table rows (adjust as needed)

### line 5234, trailing  _(unsure)_

```python
y_spacing = 0.04
```

Adjust this for better spacing between rows

### line 5236  _(unsure)_

```python
label_x_pos = ax.get_xlim()[0] - 0.3  # Adjust offset from the y-axis
```

Determine the leftmost x-position for row labels (align with the y-axis)

### line 5237, trailing  _(unsure)_

```python
label_x_pos = ax.get_xlim()[0] - 0.3
```

Adjust offset from the y-axis

### line 5239  _(unsure)_

```python
for row_idx, title in enumerate(row_labels):
```

Place row labels vertically aligned with symbols

### line 5241, trailing  _(unsure)_

```python
y_pos = symbol_start_y - (row_idx * y_spacing)
```

Calculate vertical position for each label

### line 5244  _(unsure)_

```python
for idx, (x_pos, column_data) in enumerate(zip(x_positions, transposed_table)):
```

Place symbols under each bar or jitter point based on x-positions

### line 5247, trailing  _(unsure)_

```python
y_pos = symbol_start_y - (row_idx * y_spacing)
```

Adjust vertical spacing for symbols

### line 5250  _(unsure)_

```python
ax.figure.canvas.draw()
```

Redraw to apply changes

## spacrGraph.create_plot._get_positions

### lines 5262-5267

```python
x_positions = sorted({line.get_xdata().mean()
```

SORTED, not whatever a set iterates in. Every position here is consumed BY INDEX -- `_place_symbols` zips it against the symbol table -- and a set of floats has no order it promises, so the left-to-right order this happens to produce for small whole numbers is not one to rely on: a symbol under the wrong box is silent and reads as a real annotation.

### lines 5276-5277  _(unsure)_

```python
x_positions = []
```

The drawing dispatch rejects unknown graph types before positions are read; the only remaining pair is line/line_std.

## spacrGraph.create_plot

### lines 5282-5303

```python
stats_df = self.df
```

Optional: Remove outliers for plotting

THE TRIM IS FOR THE PICTURE, NOT FOR THE TEST, and it used to be for both. `remove_outliers_from_plot` drops 1.5*IQR points PER GROUP, and it ran here -- before the normality test, before the comparison and before the post-hoc, all of which then read the trimmed frame.

That inflates significance in the one direction nobody checks. Removing a group's tails shrinks its standard deviation, so the t-statistic grows for a difference in means that has not changed. Worse, trimming PER GROUP removes exactly the points that make two groups overlap. A caller asking not to have one point stretch the y-axis was silently also asking for a smaller p-value.

No shipped caller passes remove_outliers=True, so nothing published came through here -- but `spacrGraph` is public API and the parameter is documented, so this is a live trap rather than a historical one.

The statistics now run on every point, and only the drawing is trimmed. The results table says so, because a reader looking at a trimmed plot beside a p-value has to know which one used what.

### lines 5308-5313

```python
test_results = self.perform_statistical_tests(unique_groups, is_normal)
```

Both checks now happen inside `spacr.figures.stats.compare`, PER COLUMN, which is also what decides between the Student, Welch and rank forms. Levene used to be computed here into `levene_stat, levene_p` and never read again, while the test that depends on the assumption ran regardless. `perform_levene_test` stays as public API for callers that want the statistic itself.

### line 5318  _(unsure)_

```python
if self.remove_outliers:
```

Now, and only now, trim what gets drawn.

### lines 5343-5346

```python
with figure_style(theme_target()):
```

THE WHOLE BUILD IS INSIDE THE HOUSE STYLE. spacrGraph is what the GUI's graph button produces, drawn from a long-lived process, so a global style write here would follow the user through every later figure of the session.

### line 5360  _(unsure)_

```python
if self.graph_type == 'bar':
```

Handle the different plot types based on `graph_type`

### line 5384  _(unsure)_

```python
if isinstance(self.y_lim, list):
```

Set y-axis start

### lines 5407-5408  _(unsure)_

```python
rotate_ticks(ax)
```

Anchored as well as rotated: a group name turned about its centre lands beside the tick it belongs to, not under it.

### line 5415, trailing  _(unsure)_

```python
legend_ax = self.fig.add_axes([0.1, -0.2, 0.62, 0.2])
```

Position the table closer to the graph

## spacrGraph._draw_comparison_lines

### lines 5501-5506

```python
drawable = [(a, b, p) for a, b, p in pairs
```

A COMPARISON THAT COULD NOT BE MADE IS NOT A COMPARISON THAT CAME OUT NEGATIVE. `perform_statistical_tests` records a refused pair as `Test Name='not testable'` with a NaN p, and `_significance_marker` answers 'ns' for it -- so drawing it would put "no difference" on the figure where the truth is "no test", which is a stronger claim than the run made.

### lines 5526-5527  _(unsure)_

```python
ax.plot([x1, x1, x2, x2],
```

A statistics bracket is drawn in the ink, at the spine's weight: it is annotation, not a series.

## spacrGraph._standerdize_figure_format

### line 5553, trailing  _(unsure)_

```python
return
```

Skip layout adjustment for line graphs

### line 5557  _(unsure)_

```python
fig_size = max(6, num_groups * 2)  / correction_factor
```

Set figure size to ensure it remains square with a minimum size

### line 5566  _(unsure)_

```python
bar_width = min(0.8, 1.5 / num_groups) / correction_factor
```

Configure layout based on the number of groups

### line 5571  _(unsure)_

```python
ax.set_xlim(-0.5, num_groups - 0.5)
```

Adjust axis limits to ensure bars are centered with respect to group labels

### lines 5574-5578

```python
rotate_ticks(ax)
```

Set ticks to match the group labels in your DataFrame group_labels = self.df[self.grouping_column].unique() group_labels = self.order ax.set_xticks(range(len(group_labels))) ax.set_xticklabels(group_labels, rotation=45, ha='right')

### line 5581  _(unsure)_

```python
if graph_type == 'bar':
```

Customize elements based on the graph type

### line 5583  _(unsure)_

```python
for bar in ax.patches:
```

Adjust bars' width and position

### line 5589  _(unsure)_

```python
for coll in ax.collections:
```

Adjust jitter points' position and size

### line 5592, trailing  _(unsure)_

```python
offsets[:, 0] += jitter_amount
```

Shift jitter points slightly

### line 5594, trailing  _(unsure)_

```python
coll.set_sizes([jitter_size]  * len(offsets))
```

Adjust point size dynamically

### line 5597  _(unsure)_

```python
for artist in ax.artists:
```

Adjust box width for consistent spacing

### line 5606, trailing  _(unsure)_

```python
ax.get_legend().set_bbox_to_anchor((1.05, 1))
```

loc='upper left',borderaxespad=0.

### line 5609  _(unsure)_

```python
ax.figure.canvas.draw()
```

Redraw the figure to apply changes

## spacrGraph._create_bar_plot

### lines 5619-5621

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5643  _(unsure)_

```python
if len(self.data_column) > 1:
```

Adjust the bar width manually

### line 5649  _(unsure)_

```python
bar.set_x(bar.get_x() - target_width / 2)
```

Center the bar on its x-coordinate

### line 5652  _(unsure)_

```python
bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)]
```

Adjust error bars alignment with bars

### line 5659  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Set legend and labels

## spacrGraph._create_jitter_plot

### line 5673, trailing

```python
hue = None
```

Disable hue to avoid two-level grouping

### lines 5674-5676

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5688  _(unsure)_

```python
self.summary_df = self.df_melted.copy()
```

Create the jitter plot

### line 5696  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Adjust legend and labels

### line 5699  _(unsure)_

```python
handles, labels = ax.get_legend_handles_labels()
```

Manage the legend

## spacrGraph._create_line_graph

### line 5729  _(unsure)_

```python
x_axis_column = self.data_column[0]
```

Ensure epoch is used on the x-axis and accuracy on the y-axis

### line 5742  _(unsure)_

```python
required_columns = [x_axis_column, y_axis_column, self.grouping_column]
```

Check if the required columns exist in the DataFrame

### line 5748  _(unsure)_

```python
self.summary_df = self.df.copy()
```

Create the line graph with one line per group

### line 5757  _(unsure)_

```python
ax.set_xlabel(f"{x_axis_column}")
```

Adjust axis labels

## spacrGraph._create_line_with_std_area

### line 5807  _(unsure)_

```python
summary_df = self.df.pivot_table(index=x_axis_column,values=y_axis_column,aggfunc=['mean', 'std']...
```

Pivot the DataFrame to get mean and std for each epoch across plates

### line 5810  _(unsure)_

```python
summary_df.columns = [x_axis_column, y_axis_column_mean, y_axis_column_std]
```

Flatten MultiIndex columns (result of pivoting)

### line 5813  _(unsure)_

```python
self.summary_df = summary_df.copy()
```

Plot the mean accuracy as a line

### lines 5815-5818

```python
sns.lineplot(data=summary_df,x=x_axis_column,y=y_axis_column_mean,ax=ax,marker='o',linewidth=WEIG...
```

One line, so the line IS the claim and takes the highlight hue. Its SD band is the same hue at 0.25 -- the one opacity the published figures use for a band, and enough to read at all; at 0.1 the band disappeared against a dark ground.

### line 5822  _(unsure)_

```python
ax.fill_between(summary_df[x_axis_column],summary_df[y_axis_column_mean] - summary_df[y_axis_colu...
```

Fill the area representing the standard deviation

### line 5825  _(unsure)_

```python
ax.set_xlabel(f"{x_axis_column}")
```

Adjust axis labels

## spacrGraph._create_box_plot

### lines 5836-5838

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5850  _(unsure)_

```python
self.summary_df = self.df_melted.copy()
```

Create the box plot

### line 5857  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Adjust legend and labels

### line 5860  _(unsure)_

```python
handles, labels = ax.get_legend_handles_labels()
```

Manage the legend

## spacrGraph._create_violin_plot

### lines 5878-5880

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5892  _(unsure)_

```python
self.summary_df = self.df_melted.copy()
```

Create the violin plot

### line 5899  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Adjust legend and labels

### line 5903  _(unsure)_

```python
handles, labels = ax.get_legend_handles_labels()
```

Manage the legend

## spacrGraph._create_jitter_bar_plot

### lines 5921-5923

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5951  _(unsure)_

```python
if len(self.data_column) > 1:
```

Adjust the bar width manually

### line 5957  _(unsure)_

```python
bar.set_x(bar.get_x() - target_width / 2)
```

Center the bar on its x-coordinate

### lines 5960-5965

```python
ax.set_xlabel(self.grouping_column)
```

Adjust error bars alignment with bars bars = [bar for bar in ax.patches if isinstance(bar, plt.Rectangle)] for bar, (_, row) in zip(bars, summary_df.iterrows()): x_bar = bar.get_x() + bar.get_width() / 2 err = row[self.error_bar_type] ax.errorbar(x=x_bar, y=bar.get_height(), yerr=err, fmt='none', c='black', capsize=5, lw=2)

### line 5967  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Set legend and labels

## spacrGraph._create_jitter_box_plot

### lines 5982-5984

```python
plot_order = [f"{g} - {c}" for g in self.order for c in self.data_column]
```

order must name levels of the column used for x. With multiple data columns x is 'Combined Group', so passing the raw group names selected nothing and seaborn drew an empty plot.

### line 5996  _(unsure)_

```python
self.summary_df = self.df_melted.copy()
```

Create the box plot

### line 6009  _(unsure)_

```python
ax.set_xlabel(self.grouping_column)
```

Adjust legend and labels

### line 6012  _(unsure)_

```python
handles, labels = ax.get_legend_handles_labels()
```

Manage the legend

## spacrGraph._save_results

### lines 6029-6033

```python
plot_path = save_figure(self.fig, plot_path, bbox_inches='tight',
```

dpi=600 was hard-coded here, and this is exactly the figure that cannot always take it: `_standerdize_figure_format` pins the canvas to >=10 inches square and grows it with the group count. `save_figure` follows the preference and says so when the number asked for is not deliverable at this size.

### lines 6045-6047

```python
if hasattr(self, 'summary_df') and self.summary_df is not None:
```

Data: raw -> preprocessed -> melted (plot input) -> summary (if available) self.raw_df.to_csv(os.path.join(self.output_dir, f"{self.results_name}_raw.csv"), index=False) self.df.to_csv(os.path.join(self.output_dir, f"{self.results_name}_preprocessed.csv"),index=False)

## plot_data_from_db

### lines 6141-6145

```python
annotation_ledger = RunLedger('plot_data_from_db:annotation')
```

Category B, not a per-item skip: the user asked for these conditions, so a missing annotation column means every well below is pooled under the wrong label. Historically this printed one line and produced a plot that looked entirely fine. The ledger makes the damage countable and SPACR_STRICT_ERRORS turns it into a hard stop.

### line 6196, trailing  _(unsure)_

```python
df=df,
```

Your DataFrame

### line 6197, trailing  _(unsure)_

```python
grouping_column=settings['grouping_column'],
```

Column for grouping the data (x-axis)

### line 6198, trailing  _(unsure)_

```python
data_column=settings['data_column'],
```

Column for the data (y-axis)

### line 6199, trailing

```python
graph_type=settings['graph_type'],
```

Type of plot ('bar', 'box', 'violin', 'jitter')

### line 6200, trailing  _(unsure)_

```python
graph_name=settings['graph_name'],
```

Name of the plot

### line 6201, trailing

```python
summary_func='mean',
```

Function to summarize data (e.g., 'mean', 'median')

### line 6202, trailing  _(unsure)_

```python
colors=None,
```

Custom colors for the plot (optional)

### line 6203, trailing  _(unsure)_

```python
output_dir=dst,
```

Directory to save the plot and results

### line 6204, trailing  _(unsure)_

```python
save=settings['save'],
```

Whether to save the plot and results

### line 6205, trailing  _(unsure)_

```python
y_lim=settings['y_lim'],
```

Starting point for y-axis (optional)

### line 6206, trailing

```python
error_bar_type='std',
```

Type of error bar ('std' or 'sem')

### line 6208, trailing

```python
theme=settings['theme'],
```

Seaborn color palette theme (e.g., 'pastel', 'muted')

### line 6214  _(unsure)_

```python
fig = spacr_graph.get_figure()
```

Get the figure object if needed

### line 6218  _(unsure)_

```python
results_df = spacr_graph.get_results()
```

Optional: Get the results DataFrame containing statistical test results

## plot_data_from_csv

### line 6282

```python
if not all(col in df.columns for col in ['plate', 'rowID', 'columnID']):
```

Check if 'plateID', 'rowID', and 'columnID' are all missing from df.columns

### line 6285

```python
df[['plateID', 'rowID', 'columnID']] = df['prc'].str.split('_', expand=True)
```

Split 'prc' into 'plateID', 'rowID', and 'columnID'

### lines 6288-6290

```python
print(f"Could not split the prc column: {e}")
```

Category B: without plateID/rowID/columnID every downstream grouping falls back to whatever happens to be in the frame, so the plot groups by the wrong thing rather than not at all.

### line 6320, trailing  _(unsure)_

```python
df=df,
```

Your DataFrame

### line 6321, trailing  _(unsure)_

```python
grouping_column=settings['grouping_column'],
```

Column for grouping the data (x-axis)

### line 6322, trailing  _(unsure)_

```python
data_column=settings['data_column'],
```

Column for the data (y-axis)

### line 6323, trailing

```python
graph_type=settings['graph_type'],
```

Type of plot ('bar', 'box', 'violin', 'jitter')

### line 6324, trailing  _(unsure)_

```python
graph_name=settings['graph_name'],
```

Name of the plot

### line 6325, trailing

```python
summary_func='mean',
```

Function to summarize data (e.g., 'mean', 'median')

### line 6326, trailing  _(unsure)_

```python
colors=None,
```

Custom colors for the plot (optional)

### line 6327, trailing  _(unsure)_

```python
output_dir=dst,
```

Directory to save the plot and results

### line 6328, trailing  _(unsure)_

```python
save=settings['save'],
```

Whether to save the plot and results

### line 6329, trailing  _(unsure)_

```python
y_lim=settings['y_lim'],
```

Starting point for y-axis (optional)

### line 6330, trailing  _(unsure)_

```python
log_y=settings['log_y'],
```

Log-transform the y-axis

### line 6331, trailing  _(unsure)_

```python
log_x=settings['log_x'],
```

Log-transform the x-axis

### line 6332, trailing

```python
error_bar_type='std',
```

Type of error bar ('std' or 'sem')

### line 6334, trailing

```python
theme=settings['theme'],
```

Seaborn color palette theme (e.g., 'pastel', 'muted')

### line 6343  _(unsure)_

```python
results_df = spacr_graph.get_results()
```

Optional: Get the results DataFrame containing statistical test results

## plot_image_grid._normalize_image

### line 6459  _(unsure)_

```python
is_pil_image = isinstance(image, Image.Image)
```

Check if the input is a PIL image and convert it to a NumPy array

### line 6464  _(unsure)_

```python
if image.ndim == 2:
```

If the image is single-channel, normalize directly

### line 6469  _(unsure)_

```python
normalized_image = np.zeros_like(image, dtype=np.float32)
```

If multi-channel, normalize each channel independently

### line 6475  _(unsure)_

```python
if is_pil_image:
```

If the input was a PIL image, convert the result back to PIL format

### line 6477

```python
normalized_image = (normalized_image * 255).astype(np.uint8)
```

Ensure the image is converted back to 8-bit range (0-255) for PIL

## plot_image_grid

### line 6484  _(unsure)_

```python
grid_size = math.ceil(math.sqrt(N))
```

Calculate the smallest square grid size to fit all images

### lines 6487-6490

```python
with figure_style(theme_target()):
```

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

### line 6492  _(unsure)_

```python
fig, axs = plt.subplots(
```

Create the square grid of subplots with a black background

### line 6496, trailing  _(unsure)_

```python
facecolor='black',
```

Set figure background to black

### lines 6497-6498

```python
squeeze=False
```

A single image gives a 1x1 grid, which matplotlib otherwise collapses to a bare Axes with no .flatten().

### line 6502  _(unsure)_

```python
axs = axs.flatten()
```

Flatten axs in case of a 2D array

### line 6508  _(unsure)_

```python
img = Image.open(img_path)
```

Load the image

### line 6512  _(unsure)_

```python
ax.imshow(img)
```

Display the image

### line 6514, trailing  _(unsure)_

```python
ax.axis('off')
```

Hide axes

### line 6516  _(unsure)_

```python
for j in range(i + 1, len(axs)):
```

Fill any unused subplots with black

### line 6518, trailing  _(unsure)_

```python
axs[j].imshow([[0, 0, 0]], cmap='gray')
```

Black square

### line 6519, trailing  _(unsure)_

```python
axs[j].axis('off')
```

Hide axes

### line 6521  _(unsure)_

```python
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
```

Adjust layout to minimize white space

## overlay_masks_on_images

### lines 6580-6584

```python
failed = []
```

ONE BAD FILE MUST NOT COST THE WHOLE FOLDER. A name shared by an image and a mask is not a promise that both read: a truncated TIFF, a stray .db, or a mask of a different rank ends the loop, and every overlay after it is silently never written -- with no list of which ones were done. Each field is its own attempt; failures are named and counted.

### line 6589  _(unsure)_

```python
img_path = os.path.join(img_folder, filename)
```

Load image and mask

### line 6596  _(unsure)_

```python
if normalize:
```

Normalize the image if requested

### line 6600  _(unsure)_

```python
mask = (mask > 0).astype(np.uint8)
```

Ensure the mask is binary

### line 6603  _(unsure)_

```python
if mask.shape != image.shape[:2]:
```

Resize the mask if it doesn't match the image size

### line 6607  _(unsure)_

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

Generate contours from the mask

### line 6610  _(unsure)_

```python
if image.ndim == 2:
```

Convert to RGB if grayscale

### line 6616  _(unsure)_

```python
overlay = image_rgb.copy()
```

Draw contours with alpha blending

### line 6621  _(unsure)_

```python
if resize:
```

Resize the final overlay if requested

### line 6625  _(unsure)_

```python
if save:
```

Save the overlay if requested

### lines 6631-6635

```python
with figure_style(theme_target()):
```

Display the result

AN IMAGE PANEL IS NOT A DATA PANEL: there is no ink to grey out and no axes to frame. What the house style gives a montage is the ground, the type scale and the theme's own ink -- applied as a context manager, so it does not follow the session out of here.

## graph_importance

### lines 6666-6668

```python
if isinstance(settings['csvs'], (str, os.PathLike)):
```

Wrap a scalar path: the guard used to assign the value to itself, so a single path string fell through and was iterated character by character. Only str/PathLike are wrapped -- a tuple or Series of paths already works.

## proportions_per_unit

### lines 6742-6745

```python
keys = list(dict.fromkeys([group_column, unit_column, bin_column]))
```

Deduplicated, because a caller may GROUP BY the unit -- the replication tables group by `prc`, which is also the well. Passing 'prc' to groupby twice puts it in the index twice, and `reset_index` then raises "cannot insert prc, already exists" instead of choosing.

### lines 6752-6756

```python
clashing = [name for name in proportions.index.names
```

`unstack` leaves the bin values as COLUMN names, and a caller's frame can already carry a column spelled like one of the index levels `prc` is both the unit and, in the replication tables, a plain column. `reset_index` then raises "cannot insert prc, already exists" rather than choosing, so the clash is removed before it can happen.

## proportion_test_by_unit

### lines 6813-6815

```python
return pd.DataFrame([{
```

The unit of replication IS the thing being compared, so every group holds exactly one unit and there is nothing to test across. Saying so beats returning a p-value computed from one number each.

## proportion_mixed_model

### line 6895, trailing  _(unsure)_

```python
except Exception as error:
```

singular, separated, or too few clusters

## plot_proportion_stacked_bars

### lines 6931-6935

```python
if isinstance(cmap, str) and cmap.strip().lower() == LEGACY_PLATE_CMAP:
```

The bins are an ORDERED quantity (volume), so their encoding is a single-hue ramp, light to dark -- the house sequential map. `'viridis'` is the literal every internal call site was written with rather than a choice anybody made, so it is treated as unset here exactly as `plot_plates` treats it; any other colormap is a choice and is honoured.

### line 6939  _(unsure)_

```python
raw_counts = df.groupby([group_column, bin_column], observed=True).size().unstack(fill_value=0)
```

Calculate contingency table for overall chi-squared test

### line 6945  _(unsure)_

```python
pairwise_results = chi_pairwise(raw_counts, verbose=settings.get('verbose', False))
```

Perform pairwise comparisons

### lines 6948-6958

```python
_level = str(level or 'object').strip().lower()
```

Plot based on level setting.

'plate' USED TO FALL THROUGH HERE. The check read

`level in ['well', 'plateID']`, while the setting's own tooltip offers 'object', 'well' and 'plate' -- so a user who asked for plate-level bars got object-level pooling instead: every object in one bar per condition, no per-plate averaging and no SD whiskers, which is a different figure answering a different question with nothing to say it had happened.

An unknown level is now named rather than silently pooled, because falling back to 'object' is exactly what made the typo invisible.

### lines 6969-6973

```python
raise ValueError(
```

'plateID' used to group by `prc` -- the WELL column -- so a plate-level request averaged wells and called them plates. It now groups by the plate, which means the plate column has to be present, and naming the missing one beats a bare KeyError raised from inside a groupby.

### lines 7015-7025

```python
results_df = pd.DataFrame({
```

THREE NUMBERS, EACH LABELLED WITH ITS UNIT AND ITS N.

The chi-squared above is computed over OBJECTS and was the only number this function reported, at every level -- object, well and plate gave byte-identical chi2 and p, while the level tooltip promised that "the reported statistics always treat the well as the unit of replication". It never did.

The old number is kept as the first row rather than replaced: every figure already published came from it, and a reader comparing an old result with a new one has to be able to see why they differ.

### lines 7036-7038

```python
unit_column = _unit_column(level, prc_column) or prc_column
```

`level='object'` still gets the well-level tests when a well column is there. Pooling objects does not make them independent, so the honest denominator is reported whether or not it was asked for.

## create_venn_diagram

### line 7071  _(unsure)_

```python
df1 = pd.read_csv(file1)
```

Read CSV files

### line 7075  _(unsure)_

```python
if filter_coeff is not None:
```

Filter based on coefficient

### line 7080  _(unsure)_

```python
genes1 = set(df1[gene_column].dropna())
```

Extract gene columns and drop NaN values

### line 7084  _(unsure)_

```python
overlapping_genes = genes1.intersection(genes2)
```

Calculate overlapping and non-overlapping genes

### lines 7089-7093

```python
with figure_style(theme_target()):
```

Create a Venn diagram. THE SENTENCE A VENN MAKES IS THE OVERLAP, so the overlap is the only region that carries colour and the two private sets are the grey it is read against. matplotlib_venn's own defaults are a red circle and a green one at alpha 0.4 -- the one pair red-green deficiency removes, and two arguments where the figure has one.

### line 7102  _(unsure)_

```python
if patch is not None:
```

A region with no genes in it has no patch at all.

### line 7114  _(unsure)_

```python
if save:
```

Save or show the figure

### line 7124  _(unsure)_

```python
return {
```

Return the results

## volcano_plot

### line 7138, trailing

```python
x_transform: str = "none",
```

"none" | "log2" | "log10" | "ln"

### line 7139, trailing

```python
y_transform: str = "-log10",
```

"none" | "-log10" | "-ln" | "log10" | "ln"

### line 7143  _(unsure)_

```python
annotate: bool = True,
```

annotation

### line 7146  _(unsure)_

```python
point_size: float = 20.0,
```

plotting

### line 7159  _(unsure)_

```python
sheet_name: Union[int, str] = 0,
```

excel options

### lines 7341-7345

```python
with figure_style(theme_target()):
```

THE WHOLE BUILD SITS INSIDE THE HOUSE STYLE, as a context manager. A volcano is the figure spaCR shows most often and it is drawn from a long-lived GUI, so a global rcParams write here would restyle every later figure of the session in every other module until the process exits.

### lines 7357-7363

```python
if (fold_change_threshold is not None) or (p_value_threshold is not None):
```

color hits if thresholds are provided; otherwise all gray. The hues are the house roles, fixed across every spaCR panel: GREEN is upregulated / called positive, RUST is downregulated, and everything that was not called is the one grey every figure compares against. They were crimson, royalblue and lightgray -- three hues that appear in no other spaCR figure, so a reader who learned them here learned nothing they could carry to the next panel.

### line 7373  _(unsure)_

```python
xlab = fold_change_col if x_transform.lower() == "none" else f"{x_transform}({fold_change_col})"
```

labels

### lines 7381-7383

```python
line_defaults = dict(color=ROLES["reference"], linestyle=(0, (4, 3)),
```

threshold lines. A THRESHOLD IS NOT A RESULT: thin, dashed and grey, so it cannot compete with the points it is there to sort. They were black at 1.0 pt, heavier than any mark on the panel.

### lines 7400-7401

```python
ax.spines["right"].set_visible(False)
```

cosmetics. The spines are set explicitly as well as by the style, because a caller may hand in an `ax` that was built outside it.

### line 7411  _(unsure)_

```python
if (fold_change_threshold is None) and (p_value_threshold is None) and (annotate_max is None):
```

If no thresholds were set, annotate nothing unless annotate_max is provided

## volcano_plot._read_table_auto

### line 7219  _(unsure)_

```python
if lower.endswith((".tsv", ".tab")):
```

TSV-like

### lines 7227-7228

```python
with open(path, "r", encoding="utf-8", errors="ignore") as f:
```

Fallback: sniff delimiter (comma vs tab) and try CSV reader

(If it's actually Excel with a missing extension, user should pass a DataFrame or fix extension)

## volcano_plot._threshold_x_in_plot_units

### lines 7290-7291  _(unsure)_

```python
return abs(np.log(t))
```

_transform_x validates the vocabulary before this helper runs, so the only remaining accepted forms are the natural-log aliases.
