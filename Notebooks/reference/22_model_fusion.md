# Fuse compatible models

**Purpose.** Combine parameters from compatible trained image-classifier checkpoints using a specified aggregation method.

**Recommended use.** Use when models share an architecture and parameterization and their combined weights are scientifically justified.

**Primary outputs.** A fused model checkpoint.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.deep_spacr.model_fusion`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.model_fusion)

```python
model_fusion(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.deep_spacr.model_fusion`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.model_fusion)


#### Essential inputs

- **`model_paths`** *(required)* — Paths to source checkpoints (dicts or ``TorchModel`` objects).
- **`save_path`** *(required)* — Base output path; suffix ``_&lt;aggregator&gt;.pth`` is appended.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Computer Vision — Model &amp; Regularization (2 settings)</summary>

#### Computer Vision — Model & Regularization

- **`dropout_rate`** *(optional)* — Optional dropout rate. Default: `None`.
- **`use_checkpoint`** *(optional)* — Whether to enable gradient checkpointing. Default: `False`.

</details>

<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`model_name`** *(optional)* — TorchModel architecture name for the fused model. Default: `'maxvit_t'`.

</details>

<details>
<summary>Additional settings (3 settings)</summary>

#### Additional settings

- **`aggregator`** *(optional)* — Reduction over stacked weights — one of ``'mean'``, ``'geomean'``, ``'median'``, ``'sum'``, ``'max'``, ``'min'``. Default ``'mean'``. Default: `'mean'`.
- **`device`** *(optional)* — Torch device string. Default ``'cpu'``. Default: `'cpu'`.
- **`pretrained`** *(optional)* — Whether pretrained weights are expected. Default: `True`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A fused model checkpoint.
