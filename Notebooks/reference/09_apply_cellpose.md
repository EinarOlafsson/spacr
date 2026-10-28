# Apply a Cellpose model

**Purpose.** Apply a pretrained or custom Cellpose model to a microscopy image collection.

**Recommended use.** Use for direct segmentation outside the complete mask-generation pipeline or for evaluating a newly trained model on unlabeled data.

**Primary outputs.** A label image for each processed input image and associated object measurements when enabled.

**Desktop route.** Make Masks → Cellpose Workbench → Apply

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.spacr_cellpose.identify_masks_finetune`](https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html#spacr.spacr_cellpose.identify_masks_finetune)

```python
identify_masks_finetune(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.spacr_cellpose.identify_masks_finetune`](https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html#spacr.spacr_cellpose.identify_masks_finetune)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`model_name`** *(conditionally required)* — (str) - Cellpose model used for segmentation. Cellpose 4 provides one stock model, 'cpsam'. Pre-SAM names ('cyto', 'cyto2', 'cyto3', 'nuclei') remain accepted for compatibility with older settings, but they are mapped to 'cpsam' and reported. Of the three parameters that previously distinguished models, only diameter remains operational in Cellpose 4 (eval rescales the image by 30/diameter); model_type and diam_mean are logged as 'not used in v4.0.1+' and omitted. Use 'cpsam' unless loading a custom CPSAM checkpoint. Default 'cpsam'.
- **`custom_model`** *(conditionally required)* — (str) - Path to a saved Cellpose model, loaded as pretrained_model by the mask-finetune tool. When set, model_type is passed as None and diameter as diam_mean (which Cellpose 4.x ignores with a warning). model_name remains active and selects the channel pair sent to model.eval; an incompatible value therefore segments the wrong channels. Default None.
- **`channels`** *(optional)* — (list of int) - Zero-indexed image channels kept in merged/*.npy and measured by measure_crop; each entry produces its own &lt;object&gt;_channel_&lt;n&gt;_* intensity columns. The list length fixes where masks land, so cell/nucleus/pathogen_mask_dim must shift if you change it. Preprocessing silently resets it to range(n) when it does not match the number of channel folders found. Default [0,1,2,3]. External Masks starts with []; there an empty list means every detected intensity channel, not no channels.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Model (1 settings)</summary>

#### Model

- **`diameter`** *(optional)* — (float) - Deprecated expected object diameter in pixels, passed to model.eval(diameter=...) by the mask-finetune tool and check_cellpose_models. Cellpose rescales each image by 30/diameter to match its approximately 30-pixel working size; a value below the true diameter upscales the image, whereas a larger value downscales it. Prefer the per-object diameter settings. Default 30. Default: `30`.

</details>

<details>
<summary>Image Preprocessing (4 settings)</summary>

#### Image Preprocessing

- **`grayscale`** *(optional)* — (bool) - Force the Cellpose channel pair to [0, 0] so the network treats the input as a single combined channel, overriding the [cytoplasm, nucleus] pair otherwise inferred from model_name (cyto -&gt; [1,0], cyto2 -&gt; [2,1], nucleus -&gt; [0,0]). Leave it on for single-channel inputs; switch it off only when feeding a genuine two-channel stack. Default True. Default: `True`.
- **`invert`** *(optional)* — (bool) - Invert intensities as each image is loaded, pixel -&gt; dtype_max - pixel (255 - x for uint8). Switch it on for brightfield or phase-contrast data where objects are darker than the background, since Cellpose expects bright objects on a dark field; leave it off for fluorescence. Default False. Default: `False`.
- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.
- **`percentiles`** *(optional)* — (list) - Two percentiles [low, high] used to rescale each channel of each image to 0-1 before segmentation, e.g. [2, 98]. Narrowing the window boosts contrast on dim objects but clips bright ones. Set None to derive them automatically: low fixed at 2, high the first of 98/99/99.9/99.99/99.999 exceeding background * Signal_to_noise. In Cellpose Masks (Apply), None instead lets Cellpose normalise each image itself, as the live preview does. Default None in the Cellpose steps. Default: `None`.

</details>

<details>
<summary>Detection Thresholds (5 settings)</summary>

#### Detection Thresholds

- **`CP_prob`** *(optional)* — (float) - Cellpose cellprob_threshold: the cell-probability cut-off applied to the network output when deciding which pixels belong to an object. Lower it (typically toward -6) to recover dim or partly detected objects and grow existing masks; raise it (toward 6) to drop faint false positives and shrink masks. Default 0. Default: `0`.
- **`flow_threshold`** *(optional)* — (float) - Cellpose flow_threshold: the maximum allowed error between the predicted flow field and the flows recomputed from each candidate mask; masks above it are discarded. Raise it to keep more objects, including irregularly shaped ones; lower it to reject poorly formed masks and reduce false positives. Default 0.4. Default: `0.4`.
- **`rescale`** *(optional)* — (bool) - Let Cellpose rescale each image by 30/diameter before segmenting, so objects arrive at the size the model expects. Turn off only when the diameter is already correct for the model. Default False. Default: `False`.
- **`resample`** *(optional)* — (bool) - Passed to Cellpose model.eval: run the mask-tracking dynamics at full image resolution instead of on the downsampled network grid. Enabling it gives smoother, better-fitting object outlines at the cost of time and memory, and helps most when objects differ a lot from the model's training diameter. Default False; the object pipeline sets True for cell/nucleus and False for pathogen. Default: `False`.
- **`fill_in`** *(optional)* — (bool) - Fill the holes inside each object of every mask Cellpose Masks (Apply) writes. Each object is filled on its own and keeps its id, so touching objects stay separate and a hole never takes pixels from a neighbour. Default True in Cellpose Masks. Default: `True`.

</details>

<details>
<summary>Image Geometry (3 settings)</summary>

#### Image Geometry

- **`resize`** *(optional)* — (bool or float) - Resize every image to target_height x target_width before running Cellpose, then scale the returned mask back to the original dimensions with nearest-neighbour interpolation so measurements remain in original pixels. Enable this setting to match oversized fields to the model's training scale or reduce GPU memory use. Requires target_height and target_width. Default False (True for plaque analysis). Default: `False`.
- **`target_height`** *(optional)* — (int) - Height in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_height and target_width are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis. Default: `None`.
- **`target_width`** *(optional)* — (int) - Width in pixels that images are resized to before segmentation; masks are scaled back to the original dimensions afterwards. Only applied when both target_width and target_height are set (and, on the non-normalized path, when resize is True). Use it to match the field size the model was trained at. Default None, which disables resizing; 1120 for plaque analysis. Default: `None`.

</details>

<details>
<summary>Background &amp; Denoising (3 settings)</summary>

#### Background & Denoising

- **`remove_background`** *(optional)* — (bool) - Hard-clip every pixel below the 'background' value to zero before normalization and segmentation. Use it when a channel carries a bright, even haze that inflates the normalization floor; leave it off for dim or already flat-fielded data, since the clip silently deletes faint real signal. Default False. Default: `False`.
- **`background`** *(optional)* — (float) - Per-channel background level in raw intensity units. Pixels below it are zeroed when remove_background is on, and it is multiplied by Signal_to_noise to set the upper anchor for normalization. Raise it if faint haze survives; set it too high and dim real objects vanish. Default 100 (200 for Cellpose training and plaque analysis). Default: `100`.
- **`Signal_to_noise`** *(optional)* — (int) - Background multiplier used as the Cellpose normalization threshold (background * Signal_to_noise). Per channel, spaCR selects the first of the 98th, 99th, 99.9th, 99.99th and 99.999th percentiles above it and rescales to that value. Higher values reduce clipping and make output dimmer; lower values reveal faint signal but can saturate bright objects. If no percentile qualifies, the range collapses to the 2nd percentile, indicating that this value is too high. Ignored when percentiles is set. Default 10 (5 in check_cellpose_models). Default: `10`.

</details>

<details>
<summary>Output &amp; Runtime (3 settings)</summary>

#### Output & Runtime

- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `False`.
- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `50`.
- **`verbose`** *(optional)* — (bool) - Print the resolved settings table, channel and model choices per object type, row counts per table, and object counts after each filter. It only adds console output; enable it to identify which stage produced an unexpected object count. The default is True for mask, UMAP, screen analysis, barcode mapping and Cellpose training, and False for measure, plotting helpers and regression. Invasion and Replication also start with console detail disabled. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A label image for each processed input image and associated object measurements when enabled.
