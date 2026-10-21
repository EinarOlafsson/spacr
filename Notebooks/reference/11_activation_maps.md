# Generate activation maps

**Purpose.** Estimate the image regions contributing to a trained classifier's predictions using saliency or activation-mapping methods.

**Recommended use.** Use after model validation to examine whether predictive evidence is spatially consistent with the intended biological phenotype.

**Primary outputs.** Per-image activation overlays and class-level aggregate maps.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.deep_spacr.generate_activation_map`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.generate_activation_map)

```python
generate_activation_map(settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.deep_spacr.generate_activation_map`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.generate_activation_map)


#### Essential inputs

- **`dataset`** *(required)* — (str) - Path to the .tar archive of single-object PNG crops produced by generate_dataset, which the activation-map step opens with TarImageDataset. The plate folder is inferred two levels above it and CAM outputs are written next to it under &lt;tar_name&gt;/&lt;cam_type&gt;/. Provide an absolute or directory-qualified path rather than a filename alone. Default ''.
- **`model_path`** *(required)* — (str) - Path to a trained spaCR classifier saved as a whole PyTorch object (loaded with torch.load(weights_only=False), not a state_dict). Used when applying a model to a dataset tar and when generating activation maps. deep_spacr overwrites it with the freshly trained model whenever train is True, so set it only to score with an existing model. Default ''.
- **`channels`** *(optional)* — (list of int) - Zero-indexed image channels kept in merged/*.npy and measured by measure_crop; each entry produces its own &lt;object&gt;_channel_&lt;n&gt;_* intensity columns. The list length fixes where masks land, so cell/nucleus/pathogen_mask_dim must shift if you change it. Preprocessing silently resets it to range(n) when it does not match the number of channel folders found. Default [0,1,2,3]. External Masks starts with []; there an empty list means every detected intensity channel, not no channels.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Model &amp; Data (3 settings)</summary>

#### Model & Data

- **`model_type`** *(optional)* — (str) - Backbone architecture for the single-object image classifier: any TorchVision classification model name (resnet50, maxvit_t, densenet121, ...). An unrecognized name does not fail during initial validation: choose_model reports 'Invalid model_type' and returns None, after which training fails. The special name 'custom' passes validation and then raises NotImplementedError. Larger backbones require more memory and generally need more labeled crops than smaller backbones. Default 'maxvit_t'. Default: `'maxvit'`.
- **`image_size`** *(optional)* — (int) - Side length in pixels of the center crop taken from each object PNG before model input. Images are cropped rather than rescaled, so larger values add zero padding and smaller values discard peripheral object pixels. This value also defines the backbone input resolution and must match the crop size used to generate the dataset. Default 224. Default: `224`.
- **`object_type`** *(optional)* — (str) - Mask used to define an object when the pointing game scores an attribution map: 'cell', 'nucleus', 'pathogen' or 'cytoplasm'. The metric checks only whether the map's maximum-valued pixel lies inside that mask. It has low computational cost but does not evaluate the rest of the map, so a method can score 1.0 while assigning spurious attribution elsewhere. Default 'cell'. Default: `'cell'`.

</details>

<details>
<summary>Attribution Method (8 settings)</summary>

#### Attribution Method

- **`cam_type`** *(optional)* — (str) - Which attribution map is computed. 'gradcam' weights target_layer feature maps by their pooled gradients; 'saliency_image' and 'saliency_channel' map the input gradient, summed or per stain. Also selectable: 'torchcam_gradcam'/'torchcam_gradcam_pp', 'hirescam', 'ablation_cam', 'gradient_shap'/'deeplift_shap', 'saliency' (SmoothGrad) and 'chefer' (ViT relevance). Methods that do not fit model_type are greyed with the reason. Default 'gradcam'. Default: `'gradcam'`.
- **`target_layer`** *(optional)* — (str) - Dotted attribute path to the convolutional layer whose activations and gradients Grad-CAM hooks, e.g. 'base_model.blocks.3.layers.1.layers.MBconv.layers.conv_b'; utils.recommend_target_layers(model) lists valid names. Later layers give class-specific but coarse maps, earlier ones finer detail. Required for 'gradcam'/'gradcam_pp' - it is auto-filled only when model_type is exactly 'maxvit', and left None it raises. Default None. Default: `None`.
- **`smoothgrad_samples`** *(optional)* — (int) - Number of noise-perturbed image copies averaged into one attribution map. Using 8-50 samples reduces local gradient variability and improves between-image comparability. A value of 0, the default, evaluates the method once and minimizes computation during method selection. Applies to every method, including the CAM family, where maps are averaged explicitly rather than through Captum. Default: `0`.
- **`smoothgrad_sigma`** *(optional)* — (float) - Standard deviation of the noise added by SmoothGrad, expressed as a fraction of the image intensity range. Values that are too small produce nearly identical samples and little averaging effect; values that are too large move samples outside the training distribution, causing the average to characterize responses to noise rather than the experimental images. Values of 0.1-0.2 are typical. Ignored when smoothgrad_samples is 0. Default 0.15. Default: `0.15`.
- **`occlusion_window`** *(optional)* — (int) - Side length in pixels of the patch moved across the image during occlusion analysis. Larger windows reduce runtime but spatial resolution and can miss features smaller than the window; smaller windows resolve finer structure with quadratically more forward passes. Occlusion provides a gradient-independent comparison for gradient-based attribution methods. Default 8. Default: `8`.
- **`occlusion_stride`** *(optional)* — (int) - How far the occlusion patch moves between evaluations. Equal to occlusion_window it tiles without overlap and is fastest; half of it doubles the passes and halves the blockiness. A stride larger than the window leaves unmeasured gaps that appear as an artificial grid in the map. Default 4. Default: `4`.
- **`ig_steps`** *(optional)* — (int) - Interpolation steps between the baseline and input image for integrated gradients. The completeness approximation improves with step count; too few steps increase the error without raising an exception. Validate the completeness error when reducing the default of 50. Computational cost is linear in this number. Default 50. Default: `50`.
- **`ig_baseline`** *(optional)* — (str) - Reference image used by integrated gradients: 'zero' is black, 'blur' is a blurred copy of the input image, and 'noise' is random. The baseline defines the attribution reference and therefore changes the result. On dark-field images, a zero baseline attributes broadly to bright object signal; a blurred baseline preserves low-frequency content and emphasizes contributions from image detail. Default 'zero'. Default: `'zero'`.

</details>

<details>
<summary>Attribution Validation (3 settings)</summary>

#### Attribution Validation

- **`attribution_steps`** *(optional)* — (int) - Points along the deletion and insertion curves used to score a map. At each step the highest-ranked remaining pixels are removed or added and the model is re-evaluated. This sets the resolution of the area under the curve used to assess whether the map identifies image features contributing to the prediction. More steps produce a smoother AUC with linearly more forward passes. Default 12. Default: `12`.
- **`attribution_baseline`** *(optional)* — (str) - Replacement used when deletion/insertion curves remove a pixel: 'blur', 'zero', or 'noise'. Replacing pixels with zero can create an out-of-distribution edge, causing part of the score change to reflect the replacement artifact rather than removed information. 'blur' generally produces the smallest distribution shift and is the default. Comparing multiple baselines quantifies the sensitivity of the AUC to this choice. Default 'blur'. Default: `'blur'`.
- **`sanity_check`** *(optional)* — (bool) - Randomize the model's weights layer by layer, recompute attribution and report the similarity between maps. A method that produces nearly the same map for a randomized model is responding to image structure rather than the trained decision function. On a small CNN, the CAM family, including spaCR's default Grad-CAM, fails this test while saliency and integrated gradients pass. The resulting similarity is reported for the selected model rather than inferred from benchmark behavior. This costs one additional attribution per randomized layer. Default True. Default: `True`.

</details>

<details>
<summary>Map Display (4 settings)</summary>

#### Map Display

- **`normalize`** *(optional)* — (bool or list) - Control percentile normalization before display, model input, or crop export. Display and activation-map tools use True for a 2nd-to-98th-percentile stretch. Measure and External Masks start at False; Measure accepts False or a two-number [low, high] percentile pair and refuses bare True because it supplies no bounds. It affects display and exported-crop scaling, not measured source intensities. Default True in the display-oriented tools. Default: `True`.
- **`normalize_input`** *(optional)* — (bool) - Apply the per-channel mean 0.5 and standard deviation 0.5 used during training before generating activation maps. Match this setting to model training; otherwise inputs are out of distribution and the resulting classes and maps are invalid. This is distinct from normalize, which percentile-stretches images for display. Default True. Default: `True`.
- **`overlay`** *(optional)* — (bool) - In the batch-grid figures, draw the activation map in the 'jet' colormap at 50 percent alpha over the source image. Turn it off and the grid tiles are left empty apart from the predicted-class label, so keep it on whenever plot is enabled. It never affects the per-object activation PNGs saved to disk, which are always the bare map. Default True. Default: `True`.
- **`plot`** *(optional)* — (bool) - Render and save quality-control figures during the pipeline, including channel montages, Cellpose mask overlays, filtration comparisons, and crop grids. Figure generation increases runtime and memory use, particularly for complete plates. test_mode enables this setting automatically. Default False. Merged Classifier and Recruitment both start with plotting enabled so their diagnostic figures are produced on the first run. Default: `False`.

</details>

<details>
<summary>Map Quantification (2 settings)</summary>

#### Map Quantification

- **`correlation`** *(optional)* — (bool) - Correlate every input channel with every activation-map channel per image and write the result to the &lt;cam_type&gt;_correlations table: a Pearson coefficient plus Manders M1/M2 at each manders_thresholds percentile (15, 50, and 75 by default). This provides quantitative evidence of stain-specific model attention beyond visual heatmap inspection. save=True is required to write the results to the database. Default True. Default: `True`.
- **`manders_thresholds`** *(optional)* — (list) - Percentiles (0-100) used by the activation-map correlation report in spacr.deep_spacr. It no longer affects a measure run: the percentile-pair columns it drove there were removed on 2026-09-02, and measure now writes the three standards-compliant Manders coefficients, which estimate each channel's background inside each object and take no percentile. Default [15, 50, 75]. Default: `[15, 50, 75]`.

</details>

<details>
<summary>Counterfactuals (10 settings)</summary>

#### Counterfactuals

- **`counterfactuals`** *(optional)* — (bool) - Also train a small class-conditional generator on the crops, guided by the loaded classifier, and morph held-out crops toward the other class in steps. Writes each crop's classifier score along its sequence, the flip rate, how far the edit moved the crop, a class-mean-shift baseline and a figure to counterfactuals/ next to the maps. The same classifier guides and scores the edits, so read the flip rate with the edit size. Default False. Default: `False`.
- **`counterfactual_crops`** *(optional)* — (int) - How many crops, taken in dataset order, train and test the counterfactual generator; a quarter is held out for scoring. More crops give a steadier estimate and a slower run. Ignored unless counterfactuals is on. Default 256. Default: `256`.
- **`counterfactual_epochs`** *(optional)* — (int) - Training passes of the counterfactual generator over its crops. Ignored unless counterfactuals is on. Default 30. Default: `30`.
- **`counterfactual_condition`** *(optional)* — (str) - What the counterfactuals morph between: 'class' uses the classifier's classes; 'plate', 'well', 'row' or 'column' uses that condition, read from each crop's file name, and pushes every edit toward the classifier's mean class probabilities for the target condition, so a control well's cells are drawn the way the classifier sees a treated well's. Needs at least two conditions among the crops. Ignored unless counterfactuals is on. Default 'class'. Default: `'class'`.
- **`counterfactual_target`** *(optional)* — (str) - Which class or condition every counterfactual morphs toward: a class index such as 1 when counterfactual_condition is 'class', otherwise a condition name as read from the crop file names, such as a well. Cells already in the target are left out. Empty morphs each cell to the next class or condition in sorted order. Ignored unless counterfactuals is on. Default ''. Default: `''`.
- **`counterfactual_generator`** *(optional)* — (str) - The model that draws the counterfactuals. 'autoencoder' is fast and trained to flip the classifier. 'diffusion' is trained only to draw crops of each class and redraws a partly noised crop under the target class, so a flip is not something it was optimised for; it is slower and needs a GPU, thousands of crops and hundreds of epochs; with a few hundred crops it changes the crops very little. Diffusion weights are saved as counterfactual_diffusion.pt next to the tables. Ignored unless counterfactuals is on. Default 'autoencoder'. Default: `'autoencoder'`.
- **`counterfactual_strength`** *(optional)* — (float) - Fraction of the diffusion noise schedule used for inversion, from 0 to 1. Lower values retain more of the source but may change the phenotype less. Zero leaves the crop unchanged. Used only by the diffusion counterfactual generator. Default 0.95. Default: `0.95`.
- **`counterfactual_structure_weight`** *(optional)* — (float) - Nonnegative weight of the diffusion training penalty for changes in spatial gradients at several scales. Uniform channel offsets are not penalized. This encourages layout retention but does not guarantee biological identity or phenotype success. Zero keeps denoising-only training. Used only by the diffusion counterfactual generator. Default 0. Default: `0.0`.
- **`counterfactual_checkpoint`** *(optional)* — (str) - Optional existing local diffusion checkpoint to reuse without training. Requires the diffusion generator and its expected SHA-256. The saved architecture, channels, preprocessing, classifier and condition identities must match the current run. Missing, legacy or incompatible metadata is refused; spaCR does not train a replacement. Blank keeps the existing training route. Default blank. Default: `''`.
- **`counterfactual_checkpoint_sha256`** *(optional)* — (str) - Expected 64-character hexadecimal SHA-256 of the selected local diffusion checkpoint. Required when a checkpoint is selected; ignored when the checkpoint path is blank. A matching checksum identifies the file but does not establish biological accuracy or independent validation. Default blank. Default: `''`.

</details>

<details>
<summary>Output &amp; Runtime (5 settings)</summary>

#### Output & Runtime

- **`save`** *(optional)* — (bool or list of bool) - Controls whether the current module writes its optional disk artifacts, such as masks, figures or result tables. Mask accepts a three-item list for [cell, nucleus, pathogen] independently; other modules use one boolean. Default varies by module. Default: `True`.
- **`shuffle`** *(optional)* — (bool) - Shuffle the tar dataset in the DataLoader when generating activation maps, so each batch-grid PDF contains a mixed sample rather than consecutive files from one plate or class. False preserves deterministic file order and permits direct alignment with the dataset listing. Default True. Default: `True`.
- **`batch_size`** *(optional)* — (int) - How many images are held and processed together in one pass: field stacks during normalization and Cellpose segmentation, crops per step during classifier training and activation maps. Raising it speeds runs up but increases RAM/VRAM roughly linearly; lower it on out-of-memory errors. Defaults: 50 for mask generation, 64 for training. Default: `64`.
- **`n_jobs`** *(optional)* — (int) - CPU workers for parallel stages: measurement, mask adjustment, DataLoader loading, and the sklearn/UMAP calls where -1 means every core. Raise it to shorten CPU-bound steps until RAM or disk I/O saturates. Note the measure-and-crop pipeline overrides your value with cpu_count()-4. Defaults vary by pipeline: cpu_count()-4, -1, or None. Default: `None`.
- **`ram_guard`** *(optional)* — (bool) - Keep worker pools from filling RAM. Before any module starts its workers, spaCR estimates each worker's memory from one input (a field, mask, crop batch or table) and lowers n_jobs to the count that leaves 12.5% of RAM free, printing a warning; the GUI asks first. Off keeps your n_jobs, and Measure fields still wait while free RAM is below that reserve. Default True. Default: `True`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Per-image activation overlays and class-level aggregate maps.
