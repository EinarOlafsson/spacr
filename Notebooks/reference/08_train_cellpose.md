# Train a Cellpose model

**Purpose.** Fine-tune a Cellpose segmentation model using annotated microscopy images and corresponding reference masks.

**Recommended use.** Use when available pretrained models do not adequately segment the object type or imaging condition.

**Primary outputs.** A trained Cellpose checkpoint and training diagnostics.

**Desktop route.** Make Masks → Cellpose Workbench → Train

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.submodules.train_cellpose`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.train_cellpose)

```python
train_cellpose(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.submodules.train_cellpose`](https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.train_cellpose)


#### Essential inputs

- **`src`** *(required)* — (str, path) - Folder the current step reads from and writes into: raw images for mask generation, the merged/ folder of .npy stacks for measure, the plate root for dataset/regression steps, or the folder of .fastq.gz reads for sequencing. Outputs (stack/, masks/, measurements/measurements.db, datasets/, results/) are created inside it. A list of paths, or a "['a','b']" string, processes several plates in one run. No usable default: the settings factories fill a placeholder ('path' or '/path/to/src'), so this must be supplied.
- **`model_name`** *(required)* — (str) - Cellpose model used for segmentation. Cellpose 4 provides one stock model, 'cpsam'. Pre-SAM names ('cyto', 'cyto2', 'cyto3', 'nuclei') remain accepted for compatibility with older settings, but they are mapped to 'cpsam' and reported. Of the three parameters that previously distinguished models, only diameter remains operational in Cellpose 4 (eval rescales the image by 30/diameter); model_type and diam_mean are logged as 'not used in v4.0.1+' and omitted. Use 'cpsam' unless loading a custom CPSAM checkpoint. Default 'cpsam'.
- **`channels`** *(optional)* — (list of int) - Zero-indexed image channels kept in merged/*.npy and measured by measure_crop; each entry produces its own &lt;object&gt;_channel_&lt;n&gt;_* intensity columns. The list length fixes where masks land, so cell/nucleus/pathogen_mask_dim must shift if you change it. Preprocessing silently resets it to range(n) when it does not match the number of channel folders found. Default [0,1,2,3]. External Masks starts with []; there an empty list means every detected intensity channel, not no channels.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Training Data (1 settings)</summary>

#### Training Data

- **`mask_src`** *(optional)* — (str) - Optional separate folder of integer object-label masks (background 0). Leave blank to use masks inside the image folder. Match image basenames, optionally with a _masks suffix. Missing or ambiguous pairs stop training. Default blank. Default: `''`.

</details>

<details>
<summary>Validation Data (2 settings)</summary>

#### Validation Data

- **`test_src`** *(optional)* — (str) - Optional validation image folder, separate from training images. Leave blank to train without validation losses. Split by well or experiment to avoid leakage between related fields. Default blank. Default: `''`.
- **`test_mask_src`** *(optional)* — (str) - Optional validation label-mask folder. Leave blank to use the masks subfolder inside the validation image folder. Default blank. Default: `''`.

</details>

<details>
<summary>Input &amp; Channels (1 settings)</summary>

#### Input & Channels

- **`channel_axis`** *(optional)* — (int or None) - Image channel axis: 0 for channel-first, -1 for channel-last, or blank to infer it from the mask dimensions. Ambiguous images require an explicit axis. Training supports 2-D fields with optional channels, not Z stacks. Default None (automatic). Default: `None`.

</details>

<details>
<summary>Image Preprocessing (2 settings)</summary>

#### Image Preprocessing

- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.
- **`percentiles`** *(optional)* — (list) - Two percentiles [low, high] used to rescale each channel of each image to 0-1 before segmentation, e.g. [2, 98]. Narrowing the window boosts contrast on dim objects but clips bright ones. Set None to derive them automatically: low fixed at 2, high the first of 98/99/99.9/99.99/99.999 exceeding background * Signal_to_noise. In Cellpose Masks (Apply), None instead lets Cellpose normalise each image itself, as the live preview does. Default None in the Cellpose steps. Default: `[1, 99]`.

</details>

<details>
<summary>Starting Point (1 settings)</summary>

#### Starting Point

- **`base_model`** *(optional)* — (str) - Train Cellpose: the weights training starts from. 'cpsam' is stock Cellpose-SAM; a model-zoo key (for example 'toxoplasma_plaque_v2') or a path to a checkpoint continues from that model, which is how a second fine-tuning stage builds on the first. The model that was started from is recorded with the run's settings. Default 'cpsam'. Default: `'cpsam'`.

</details>

<details>
<summary>Training Schedule (4 settings)</summary>

#### Training Schedule

- **`n_epochs`** *(optional)* — (int) - Number of training passes train_seg makes over the annotated image/mask batch. It also sets the checkpoint interval (a model is saved every n_epochs/10) and is written into the saved model filename. Raise it for a better fit on large annotation sets; lower it when a small set starts overfitting. Default 10000. Default: `100`.
- **`learning_rate`** *(optional)* — (float) - Initial optimizer step size. Values that are too high may prevent convergence; values that are too low may slow convergence or converge to a suboptimal solution. A value near 1e-3 is commonly used for training from random initialization, while 1e-4 to 1e-5 is appropriate for fine-tuning ImageNet weights (init_weights=True). The selected schedule modifies this initial value during training. Default 0.001. Default: `1e-05`.
- **`weight_decay`** *(optional)* — (float) - L2 penalty applied to the weights on every optimizer step (AdamW applies it decoupled from the gradient). Raise it, toward 1e-3 to 1e-2, when validation loss climbs while training loss keeps falling; lower it toward 0 when the model cannot fit the training set at all. Every supported optimizer honours it. Default 0.00001. Default: `0.1`.
- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `1`.

</details>

<details>
<summary>Sampling &amp; Augmentation (5 settings)</summary>

#### Sampling & Augmentation

- **`min_train_masks`** *(optional)* — (int) - Minimum labeled objects required per training image. Cellpose excludes fields below this count. Default 5. Lower it for deliberately sparse training fields. Default: `5`.
- **`max_train_images`** *(optional)* — (int or None) - Optional limit on paired training images loaded into RAM, in filename order. Blank or a nonpositive value uses every pair. This does not change the minibatch size. Default None (automatic). Default: `None`.
- **`nimg_per_epoch`** *(optional)* — (int or None) - Optional number of images sampled per training epoch. Blank uses every training image. This changes sampling, not the number of files loaded into RAM. Default None (automatic). Default: `None`.
- **`nimg_test_per_epoch`** *(optional)* — (int or None) - Optional number of validation images sampled per evaluation epoch. Blank uses all validation images. Requires a validation image source. Default None (automatic). Default: `None`.
- **`scale_range`** *(optional)* — (float) - Range of Cellpose's random training scale augmentation, from 0 to 2. Default 0.5. Cellpose also applies its native rotation, flip and crop augmentation; no eight-fold duplicate dataset is created. Default: `0.5`.

</details>

<details>
<summary>Checkpoints (3 settings)</summary>

#### Checkpoints

- **`save_path`** *(optional)* — (str) - Checkpoint output folder. Cellpose writes weights into its models subfolder. Leave blank for &lt;image source&gt;/models/cellpose_model. Use trained model reads this location too. Default blank. Default: `''`.
- **`save_every`** *(optional)* — (int) - Checkpoint interval in epochs. Default 100. Cellpose always saves the final model even when the run is shorter than this interval. Default: `100`.
- **`save_each`** *(optional)* — (bool) - Keep separate epoch checkpoints instead of replacing the periodic checkpoint. Default False. Enable to compare intermediate models; it consumes additional disk space. Default: `False`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A trained Cellpose checkpoint and training diagnostics.

## Optional dataset preparation

Prepare matching image/mask folders before training; point the essential train/validation paths at the resulting dataset.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.io.prepare_cellpose_dataset`](https://einarolafsson.github.io/spacr/api/spacr/io/index.html#spacr.io.prepare_cellpose_dataset)

```python
prepare_cellpose_dataset(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.io.prepare_cellpose_dataset`](https://einarolafsson.github.io/spacr/api/spacr/io/index.html#spacr.io.prepare_cellpose_dataset)


#### Essential inputs

- **`input_root`** *(required)* — Directory containing one subfolder per dataset.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Advanced (1 settings)</summary>

#### Advanced

- **`n_jobs`** *(optional)* — Worker count for parallel copies. Default: CPU count. Default: `None`.

</details>

<details>
<summary>Additional settings (2 settings)</summary>

#### Additional settings

- **`augment_data`** *(optional)* — If True, expand under-sized datasets by applying geometric augmentations. Default ``False``. Default: `False`.
- **`train_fraction`** *(optional)* — Fraction of pairs routed to the train split. Default ``0.8``. Default: `0.8`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.
