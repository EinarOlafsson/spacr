# Analyze activation maps

**Purpose.** Aggregate image-attribution maps and compare attribution summaries across model classes.

**Recommended use.** Use after activation-map generation to quantify class-specific patterns of predictive evidence.

**Primary outputs.** Class-level attribution statistics and comparison figures.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.deep_spacr.analyze_activation_maps`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.analyze_activation_maps)

```python
analyze_activation_maps(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.deep_spacr.analyze_activation_maps`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.analyze_activation_maps)


#### Essential inputs

- **`model`** *(required)* — Trained classifier.
- **`images`** *(required)* — Image tensor or iterable of tensors with shape ``(C, H, W)``.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Model &amp; Data (1 settings)</summary>

#### Model & Data

- **`model_type`** *(optional)* — Optional architecture name used in diagnostic messages. Default: `None`.

</details>

<details>
<summary>Attribution Method (1 settings)</summary>

#### Attribution Method

- **`target_layer`** *(optional)* — CAM target layer. ``None`` selects the final convolutional layer. Default: `None`.

</details>

<details>
<summary>Attribution Validation (1 settings)</summary>

#### Attribution Validation

- **`sanity_check`** *(optional)* — Whether to run parameter-randomization analysis on the first image. Default: `True`.

</details>

<details>
<summary>Additional Settings (3 settings)</summary>

#### Additional Settings

- **`target`** *(optional)* — Class index to explain. ``None`` uses each image's predicted class. Default: `None`.
- **`verbose`** *(optional)* — Whether to print per-method validation results. Default: `True`.
- **`masks`** *(optional)* — Optional per-image boolean object masks used by the pointing game. Default: `None`.

</details>

<details>
<summary>Additional settings (4 settings)</summary>

#### Additional settings

- **`baseline`** *(optional)* — Replacement used for removed pixels: ``"blur"``, ``"zero"``, ``"mean"``, or ``"uniform"``. Default: `'blur'`.
- **`methods`** *(optional)* — Attribution methods from :data:`spacr.attribution.ATTRIBUTION_METHODS`. ``None`` uses ``gradcam``, ``saliency``, ``integrated_gradients``, and ``occlusion``. Default: `None`.
- **`n_steps`** *(optional)* — Number of perturbation steps for deletion and insertion curves. Default: `12`.
- **`sanity_threshold`** *(optional)* — Maximum rank correlation at which a randomized map is considered sufficiently different. Default: `0.5`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

Class-level attribution statistics and comparison figures.
