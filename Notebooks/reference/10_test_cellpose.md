# Evaluate a Cellpose model

**Purpose.** Compare predicted masks with reference masks using per-image segmentation metrics.

**Recommended use.** Use to compare candidate models on a common held-out dataset before large-scale application.

**Primary outputs.** Per-image evaluation metrics and aggregate performance summaries.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.test_cellpose_model`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.test_cellpose_model)

```python
test_cellpose_model(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.test_cellpose_model`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.test_cellpose_model)


#### Essential inputs

- **`src`** *(optional)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Paths (1 settings)</summary>

#### Paths

- **`model_path`** *(optional)* — (str) - Path to a trained spaCR classifier saved as a whole PyTorch object (loaded with torch.load(weights_only=False), not a state_dict). Used when applying a model to a dataset tar and when generating activation maps. deep_spacr overwrites it with the freshly trained model whenever train is True, so set it only to score with an existing model. Default ''. Default: `'path'`.

</details>

<details>
<summary>General (1 settings)</summary>

#### General

- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.

</details>

<details>
<summary>Cellpose (2 settings)</summary>

#### Cellpose

- **`target_size`** *(optional)* — (int) - Edge length in pixels to which training images and masks are resized before Cellpose fine-tuning, applied to both axes to produce square input. Larger values preserve finer boundary detail while increasing VRAM use and computation approximately quadratically; smaller values reduce computation but may blur segmentation boundaries. Default 1000. Default: `1000`.
- **`percentiles`** *(optional)* — (list) - Two percentiles [low, high] used to rescale each channel of each image to 0-1 before segmentation, e.g. [2, 98]. Narrowing the window boosts contrast on dim objects but clips bright ones. Set None to derive them automatically: low fixed at 2, high the first of 98/99/99.9/99.99/99.999 exceeding background * Signal_to_noise. In Cellpose Masks (Apply), None instead lets Cellpose normalise each image itself, as the live preview does. Default None in the Cellpose steps. Default: `(2, 98)`.

</details>

<details>
<summary>Advanced (2 settings)</summary>

#### Advanced

- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `50`.
- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `True`.

</details>

<details>
<summary>Additional settings (2 settings)</summary>

#### Additional settings

- **`CP_probability`** *(optional)* — (int) - Cellpose cellprob_threshold used by the standalone apply/test-model submodules, where it carries this name instead of the per-object &lt;object&gt;_cellprob_threshold used by the Mask module. Only pixels whose predicted cell probability exceeds it join a mask, so raising it shrinks outlines and drops faint objects while lowering it grows them and recovers dim ones. Default 0. Default: `0`.
- **`FT`** *(optional)* — (float) - Cellpose flow_threshold for the standalone apply/test-model submodules, the counterpart of the Mask module's per-object &lt;object&gt;_flow_threshold. Masks whose recomputed flows disagree with the network's prediction by more than this are discarded, so a low value strips ragged or implausible objects and also loses real ones. Typical range 0 to 3; above 3 practically every candidate is kept. Default 0.4, Cellpose's own default. Default: `0.4`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-image evaluation metrics and aggregate performance summaries.
