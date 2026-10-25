# Analyze a plaque assay

**Purpose.** Segment plaques and quantify plaque number and area for each image or well.

**Recommended use.** Use for lytic-cycle assays that quantify disruption of a host-cell monolayer.

**Primary outputs.** Per-plaque measurements and per-well count and area summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.analyze_plaques`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques)

```python
analyze_plaques(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.analyze_plaques`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`channels`** *(optional)* — (list of int) - Zero-indexed image channels kept in merged/*.npy and measured by measure_crop; each entry produces its own &lt;object&gt;_channel_&lt;n&gt;_* intensity columns. The list length fixes where masks land, so cell/nucleus/pathogen_mask_dim must shift if you change it. Preprocessing silently resets it to range(n) when it does not match the number of channel folders found. Default [0,1,2,3]. External Masks starts with []; there an empty list means every detected intensity channel, not no channels.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Input &amp; Channels (1 settings)</summary>

#### Input & Channels

- **`masks`** *(optional)* — (bool) - Run Cellpose segmentation for every configured object channel (cell, nucleus, pathogen, organelle) and write label stacks to masks/&lt;object&gt;_mask_stack. False performs preprocessing only, producing normalized arrays without label masks; downstream measurement therefore requires a subsequent segmentation step. Default True. Default: `True`.

</details>

<details>
<summary>Scale &amp; Time (4 settings)</summary>

#### Scale & Time

- **`plate_format`** *(optional)* — (str or None) - Standard culture-plate format used as the physical ruler for detected wells: '6-well', '12-well', '24-well', '48-well' or '96-well'. It converts plaque areas from pixels to square millimetres; None leaves physical-area columns empty unless well_diameter_mm is supplied. Default None. Default: `None`.
- **`well_diameter_mm`** *(optional)* — (float, int or None) - Known interior diameter of a detected well in millimetres, overriding plate_format when both are set. It converts the detected pixel diameter into pixels per millimetre and therefore rescales every physical plaque area; use None when the diameter is unknown. Default None. Default: `None`.
- **`plaque_pixels_per_um`** *(optional)* — (float, int or None) - Known pixels per micrometer in the analyzed image. Positive values override detected rulers. Leave blank for automatic scale-bar or well-diameter calibration. Per-well values entered in Figure preview take precedence. Default None. Default: `None`.
- **`plaque_formation_hours`** *(optional)* — (float, int or None) - Elapsed plaque formation time in hours, recorded as experimental metadata. Zero is permitted; blank means unknown. Figure preview allows per-well overrides. Default None. Default: `None`.

</details>

<details>
<summary>Colony Counting α (9 settings)</summary>

#### Colony Counting α

- **`colony_counting`** *(optional)* — (bool) - Count bacterial or fungal colonies on plate or dish photos instead of segmenting plaques. Each image is one plate, or one well per detected well when well_detection is on; the dish is found by its outline otherwise. Colonies are thresholded against the agar, touching ones are split, and the count, CFU/mL, colony areas and diameters go to colonies/colonies.db in src. No plaque model is loaded. Plaque mode only: Figure mode ignores it. Default False. Default: `False`.
- **`colony_dilution`** *(optional)* — (float, int, dict or str) - Dilution factor of the plated suspension: 10000 for a 10^-4 dilution; fractions such as 0.0001 are inverted. CFU/mL = colonies x dilution factor / plated volume. Enter one number, a dict keyed by filename or stem, or a UTF-8 CSV path with file,dilution columns. Exact filenames take priority over stems; unmatched plates get no CFU/mL. CSV factors must be positive and finite, with unique file identifiers. Default 1. Default: `1`.
- **`colony_plated_volume_ul`** *(optional)* — (float) - Volume of the dilution spread on each plate, in microlitres, the denominator of CFU/mL. Change it with the plating protocol: 100 for a standard spread plate, 1000 for a pour plate of 1 mL. Default 100. Default: `100`.
- **`colony_too_many`** *(optional)* — (int or None) - Plates with more colonies than this are flagged 'too many to count' (TNTC) in per_plate: neighbouring colonies merge and compete, so the count underestimates what was plated. Their CFU/mL is still written, so filter on the flag. Blank turns the check off. Default 300. Default: `300`.
- **`colony_too_few`** *(optional)* — (int or None) - Plates with fewer colonies than this are flagged 'too few to count' (TFTC) in per_plate: so few colonies carry a sampling error too large for the CFU/mL they imply. Their CFU/mL is still written, so filter on the flag. Blank turns the check off. Default 30. Default: `30`.
- **`colony_polarity`** *(optional)* — (str) - Whether colonies are brighter than the agar (bright: white or cream colonies on blood, chocolate or dark agar, or any plate photographed on a dark background) or darker (dark: on a light box or on pale agar). auto tries both and keeps the one whose round objects stand further above the agar. Set it when auto picks the wrong one on a sparse plate. Default auto. Default: `'auto'`.
- **`colony_threshold`** *(optional)* — (float) - How far above the agar a pixel must be to count as colony, in multiples of the agar's own noise. Lower finds faint, small or translucent colonies but also picks up agar texture, bubbles and glare; higher keeps only clear colonies. Default 4.0. Default: `4.0`.
- **`colony_min_area_px`** *(optional)* — (float, int or None) - Smallest colony counted, in pixels of the original photo. Raise it to ignore dust, bubbles and pinpoint artefacts; lower it for pinpoint colonies. Blank uses 0.4 % of the dish diameter, squared: about 36 pixels for a dish 1500 pixels across. Default None. Default: `None`.
- **`colony_detector`** *(optional)* — (str or None) - A YOLO colony detector to find the colonies with, in place of thresholding: a checkpoint path or a Model Zoo key. It needs the ultralytics package. Detection counts colonies in chains and at the dish rim that thresholding merges or loses; polarity, threshold and minimum area are then not used. Blank thresholds. Default None. Default: `None`.

</details>

<details>
<summary>Experimental Growth Estimates (3 settings)</summary>

#### Experimental Growth Estimates

- **`plaque_estimate_growth`** *(optional)* — (bool) - Optional experimental plaque time/scale suggestions, saved separately from measurements. Uses the largest quarter of plaques and an assumed linear growth relation. Defaults off. The RH/HFF seven-day reference has about 40 hours held-out endpoint error and has not been validated across times or other conditions. Without a known scale or time, the reference duration is explicitly assumed. Default: `False`.
- **`plaque_growth_reference_um`** *(optional)* — (float) - Reference median equivalent diameter in micrometers of the largest 25 percent of plaques. Default 893.8178699548309, derived from three independent RH/HFF vehicle-control experiments in Kelsen et al. 2023 S20 Data, DOI 10.1371/journal.pbio.3002110. Replace with a matched local reference when available. Default: `893.8178699548309`.
- **`plaque_growth_reference_hours`** *(optional)* — (float) - Positive formation time in hours corresponding to the growth reference diameter. Default 168 (seven days). Linear diameter growth through zero is assumed; this is not a fitted temporal growth curve. Default: `168.0`.

</details>

<details>
<summary>Model (1 settings)</summary>

#### Model

- **`diameter`** *(optional)* — (float) - Deprecated expected object diameter in pixels, passed to model.eval(diameter=...) by the mask-finetune tool and check_cellpose_models. Cellpose rescales each image by 30/diameter to match its approximately 30-pixel working size; a value below the true diameter upscales the image, whereas a larger value downscales it. Prefer the per-object diameter settings. Default 30. Default: `30`.

</details>

<details>
<summary>Detection Thresholds (6 settings)</summary>

#### Detection Thresholds

- **`CP_prob`** *(optional)* — (float) - Cellpose cellprob_threshold: the cell-probability cut-off applied to the network output when deciding which pixels belong to an object. Lower it (typically toward -6) to recover dim or partly detected objects and grow existing masks; raise it (toward 6) to drop faint false positives and shrink masks. Default 0. Default: `0`.
- **`flow_threshold`** *(optional)* — (float) - Cellpose flow_threshold: the maximum allowed error between the predicted flow field and the flows recomputed from each candidate mask; masks above it are discarded. Raise it to keep more objects, including irregularly shaped ones; lower it to reject poorly formed masks and reduce false positives. Default 0.4. Default: `0.4`.
- **`minimum_object_size`** *(optional)* — (float or int) - Minimum plaque area in square pixels of the input image or detected well crop. Objects smaller than this value are discarded. 0 keeps small objects; other segmentation filters still apply. Fractional values are accepted. Default 0. Default: `0.0`.
- **`rescale`** *(optional)* — (bool) - Let Cellpose rescale each image by 30/diameter before segmenting, so objects arrive at the size the model expects. Turn off only when the diameter is already correct for the model. Default False. Default: `False`.
- **`resample`** *(optional)* — (bool) - Passed to Cellpose model.eval: run the mask-tracking dynamics at full image resolution instead of on the downsampled network grid. Enabling it gives smoother, better-fitting object outlines at the cost of time and memory, and helps most when objects differ a lot from the model's training diameter. Default False; the object pipeline sets True for cell/nucleus and False for pathogen. Default: `False`.
- **`fill_in`** *(optional)* — (bool) - Fill the holes inside each object of every mask Cellpose Masks (Apply) writes. Each object is filled on its own and keeps its id, so touching objects stay separate and a hole never takes pixels from a neighbour. Default True in Cellpose Masks. Default: `True`.

</details>

<details>
<summary>Image Geometry (6 settings)</summary>

#### Image Geometry

- **`plaque_well_resize`** *(optional)* — (str) - Resize each detected well before segmentation so plaques reach the size the plaque model was trained at. 'auto' follows the selected model: a model that declares well resize turns it on with its own values, and other models leave it off. 'on' uses the side and enlargement settings below; 'off' segments the image as it is. When on, the diameter setting is not used. Default 'auto'. Default: `'auto'`.
- **`plaque_well_resize_side`** *(optional)* — (int) - Longest side in pixels that each detected well is resized to before segmentation when plaque well resize is on. Larger values make small plaques bigger for the model and cost more time and memory; masks are mapped back to the original pixels. Default 1500. Default: `1500`.
- **`plaque_well_resize_max_factor`** *(optional)* — (float or int) - Largest enlargement plaque well resize applies to a small well or a low-resolution image; shrinking a large well is never limited. It stops a tiny image from being magnified far beyond what its pixels can show. Default 6. Default: `6.0`.
- **`resize`** *(optional)* — (bool or float) - Resize every image to target_height x target_width before running Cellpose, then scale the returned mask back to the original dimensions with nearest-neighbour interpolation so measurements remain in original pixels. Enable this setting to match oversized fields to the model's training scale or reduce GPU memory use. Requires target_height and target_width. Default False (True for plaque analysis). Default: `True`.
- **`target_height`** *(optional)* — (int) - Height in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_height and target_width are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis. Default: `1120`.
- **`target_width`** *(optional)* — (int) - Width in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_width and target_height are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis. Default: `1120`.

</details>

<details>
<summary>Background &amp; Denoising (3 settings)</summary>

#### Background & Denoising

- **`remove_background`** *(optional)* — (bool) - Hard-clip every pixel below the 'background' value to zero before normalization and segmentation. Use it when a channel carries a bright, even haze that inflates the normalization floor; leave it off for dim or already flat-fielded data, since the clip silently deletes faint real signal. Default False. Default: `False`.
- **`background`** *(optional)* — (float) - Per-channel background level in raw intensity units. Pixels below it are zeroed when remove_background is on, and it is multiplied by Signal_to_noise to set the upper anchor for normalization. Raise it if faint haze survives; set it too high and dim real objects vanish. Default 100 (200 for Cellpose training and plaque analysis). Default: `200`.
- **`Signal_to_noise`** *(optional)* — (int) - Background multiplier used as the Cellpose normalization threshold (background * Signal_to_noise). Per channel, spaCR selects the first of the 98th, 99th, 99.9th, 99.99th and 99.999th percentiles above it and rescales to that value. Higher values reduce clipping and make output dimmer; lower values reveal faint signal but can saturate bright objects. If no percentile qualifies, the range collapses to the 2nd percentile, indicating that this value is too high. Ignored when percentiles is set. Default 10 (5 in check_cellpose_models). Default: `10`.

</details>

<details>
<summary>Output &amp; Runtime (3 settings)</summary>

#### Output & Runtime

- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `True`.
- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `50`.
- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `True`.

</details>

<details>
<summary>Additional Settings (28 settings)</summary>

#### Additional Settings

- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.
- **`percentiles`** *(optional)* — (list) - Two percentiles [low, high] used to rescale each channel of each image to 0-1 before segmentation, e.g. [2, 98]. Narrowing the window boosts contrast on dim objects but clips bright ones. Set None to derive them automatically: low fixed at 2, high the first of 98/99/99.9/99.99/99.999 exceeding background * Signal_to_noise. In Cellpose Masks (Apply), None instead lets Cellpose normalise each image itself, as the live preview does. Default None in the Cellpose steps. Default: `None`.
- **`invert`** *(optional)* — (bool) - Invert intensities as each image is loaded, pixel -&gt; dtype_max - pixel (255 - x for uint8). Switch it on for brightfield or phase-contrast data where objects are darker than the background, since Cellpose expects bright objects on a dark field; leave it off for fluorescence. Default False. Default: `False`.
- **`grayscale`** *(optional)* — (bool) - Force the Cellpose channel pair to [0, 0] so the network treats the input as a single combined channel, overriding the [cytoplasm, nucleus] pair otherwise inferred from model_name (cyto -&gt; [1,0], cyto2 -&gt; [2,1], nucleus -&gt; [0,0]). Leave it on for single-channel inputs; switch it off only when feeding a genuine two-channel stack. Default True. Default: `True`.
- **`plaque_model`** *(optional)* — (str) - Cellpose checkpoint used to segment plaques: a model-zoo key downloads a checksum-verified checkpoint on first use ('toxoplasma_plaque_v6' was trained on resized wells and turns plaque well resize on; 'toxoplasma_plaque_v5', the previous default, segments images as they are; 'toxoplasma_plaque_v3' scores higher on Gel Doc wells), a filesystem path selects a custom checkpoint, and 'bundled' is a Cellpose 3 checkpoint that needs Cellpose 3. Changing it changes plaque counts, so recorded runs should keep their value. Default 'toxoplasma_plaque_v6'. Default: `'toxoplasma_plaque_v6'`.
- **`plaque_mode`** *(optional)* — (str) - What Plaque Assay reads. 'plaque' takes images that each show one plaque field (a well or a cropped plaque image), segments the plaques and writes per_image and per_plaque tables. 'figure' takes published figures: the YOLO detector finds the plaque images in each figure, the text around them is read (panel letter, column and row labels) and keyed to the figure legend, each image is annotated with its condition and its plaques are segmented, into plaque_figures/plaque_figures.db. Default 'plaque'. Default: `'plaque'`.
- **`figure_detector`** *(optional)* — (str) - Figure mode: model-zoo key or checkpoint path of the YOLO detector that finds plaque images inside a figure. Default 'toxoplasma_well_detector_v2', which was measured to find cropped plaque panels in published figures. Default: `'toxoplasma_well_detector_v2'`.
- **`figure_imgsz`** *(optional)* — (str) - Figure mode: comma-separated detector input sizes, all of which are asked and their boxes merged. 640 finds whole faint panels, 1280 finds small dilution spots; neither alone finds both. Default '640,1280'. Default: `'640,1280'`.
- **`figure_confidence`** *(optional)* — (float) - Figure mode: minimum detector score, 0 to 1, for a box to count as a plaque image. Default 0.25. Default: `0.25`.
- **`figure_read_text`** *(optional)* — (bool) - Figure mode: read the text printed around each plaque image (needs RapidOCR, part of spacr[papers]) to annotate it with its panel and condition. False names each image by its figure, row and column only. Default True. Default: `True`.
- **`confirm_annotations`** *(optional)* — (bool) - Figure mode: measure only images whose condition a person approved in the Figure preview (saved to figure_annotations.csv in the source folder); the others are counted as waiting. Default False. Default: `False`.
- **`text_reach_above`** *(optional)* — (float) - Figure mode, text detection: How far above the grid of plaque images a column header may sit, in image heights. Raise it when headers are printed well above the images; lower it when the header of a neighbouring panel is picked up. Default 1.0. Default: `1.0`.
- **`text_reach_left`** *(optional)* — (float) - Figure mode, text detection: How far left of the grid a row label may sit, in image widths. Raise it when row labels are set far to the left; lower it when text from the panel on the left is picked up. Default 1.0. Default: `1.0`.
- **`text_reach_below`** *(optional)* — (float) - Figure mode, text detection: How far below the grid text may sit and still count as a label, in image heights. Default 0.5. Default: `0.5`.
- **`text_use_above`** *(optional)* — (bool) - Figure mode, text detection: Use the column header printed above each image as part of its condition. Default True. Default: `True`.
- **`text_use_left`** *(optional)* — (bool) - Figure mode, text detection: Use the row label printed to the left of each image as part of its condition. Default True. Default: `True`.
- **`text_use_below`** *(optional)* — (bool) - Figure mode, text detection: Use text printed under the images as part of the condition. Turn off when captions or axis labels sit under the images. Default True. Default: `True`.
- **`text_panel_reach`** *(optional)* — (float) - Figure mode, text detection: How far up and left of a grid's top-left corner the panel letter (A, B, C...) may sit, in image sizes. Raise it when the letter is set far from the images; lower it when another panel's letter is taken. Default 1.0. Default: `1.0`.
- **`text_min_confidence`** *(optional)* — (float) - Figure mode, text detection: Ignore OCR words the reader scored below this (0-1). Raise it when stray specks are read as text. Default 0.0. Default: `0.0`.
- **`text_ignore`** *(optional)* — (str) - Figure mode, text detection: Comma-separated regular expressions; a word matching any is not used as a label, for example scale bars and axis numbers: '^\\d+$, [uμ]m$'. Default ''. Default: `''`.
- **`text_order`** *(optional)* — (str) - Figure mode, text detection: Which labels come first when they are joined into one condition: a comma-separated order of above, left and below. Default 'above,left,below'. Default: `'above,left,below'`.
- **`text_separator`** *(optional)* — (str) - Figure mode, text detection: What joins the labels into one condition. Default ' / '. Default: `' / '`.
- **`text_reread`** *(optional)* — (bool) - Figure mode, text detection: Read the text around each grid of images a second time, enlarged, which finds small or rotated labels the first reading missed. Default True. Default: `True`.
- **`text_reread_scale`** *(optional)* — (int) - Figure mode, text detection: How many times to enlarge the text around each grid for the second reading. Default 3. Default: `3`.
- **`well_detection`** *(optional)* — (str or bool) - Split a plate image into detected wells before plaque segmentation. False passes each source image through whole; True selects the default YOLO detector, while a model-zoo key or checkpoint path selects another detector. Enabling it changes result rows from one per image to one per detected well. Default False. Default: `False`.
- **`well_confidence`** *(optional)* — (float) - Minimum YOLO confidence, from 0 to 1, for keeping a detected well when well_detection is enabled. Raising it removes uncertain boxes but can lose an entire condition; lowering it retains more candidates and can create spurious well crops. Default 0.25. Default: `0.25`.
- **`well_pad`** *(optional)* — (int) - Extra image pixels retained on every side of a detected well crop, clipped at the source-image boundary. Increase it when the detector box trims the well edge; excessive padding can include neighbouring wells or background. Default 0. Default: `0`.
- **`model_name`** *(optional)* — (str) - Cellpose model used for segmentation. Cellpose 4 provides one stock model, 'cpsam'. Pre-SAM names ('cyto', 'cyto2', 'cyto3', 'nuclei') remain accepted for compatibility with older settings, but they are mapped to 'cpsam' and reported. Of the three parameters that previously distinguished models, only diameter remains operational in Cellpose 4 (eval rescales the image by 30/diameter); model_type and diam_mean are logged as 'not used in v4.0.1+' and omitted. Use 'cpsam' unless loading a custom CPSAM checkpoint. Default 'cpsam'. Default: `'cpsam'`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-plaque measurements and per-well count and area summaries.
