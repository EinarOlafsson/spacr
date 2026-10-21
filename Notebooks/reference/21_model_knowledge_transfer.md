# Transfer knowledge between models

**Purpose.** Train a student image classifier using predictions from one or more teacher models.

**Recommended use.** Use when adapting validated predictive information to a related dataset or consolidating teacher models into a student architecture.

**Primary outputs.** A trained student model and recorded training provenance.

---

> Paths in this notebook are placeholders. Set `src` and any other required path to the experimental data before execution.
> spaCR writes outputs within, or immediately adjacent to, the configured source directory unless an explicit output path is set.

## 2. API entry point

This workflow calls the following public function:

- [`spacr.deep_spacr.model_knowledge_transfer`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.model_knowledge_transfer)

```python
model_knowledge_transfer(**settings)
```

Parameter definitions and defaults are generated from the same public API and are listed in the settings reference below.

## 3. Settings and API reference

This linked reference contains the full generated setting descriptions. Return to the notebook to edit essential inputs or open Advanced settings. Required inputs must be supplied; conditional inputs apply only to the indicated branch. Unchanged optional values use the installed API defaults. Descriptions and displayed defaults reflect the spaCR source used to generate this reference.

### [`spacr.deep_spacr.model_knowledge_transfer`](https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html#spacr.deep_spacr.model_knowledge_transfer)


#### Essential inputs

- **`teacher_paths`** *(required)* — Paths to the teacher checkpoints (each either a saved ``TorchModel`` or a state dict).
- **`student_save_path`** *(required)* — Destination for the trained student; the suffix ``_KD.pth`` is appended.
- **`data_loader`** *(required)* — Training DataLoader used during distillation.

<details>
<summary>Advanced settings reference — click to expand</summary>


<details>
<summary>Computer Vision — Model &amp; Regularization (2 settings)</summary>

#### Computer Vision — Model & Regularization

- **`dropout_rate`** *(optional)* — Optional dropout rate for the student. Default: `None`.
- **`use_checkpoint`** *(optional)* — Whether to enable gradient checkpointing. Default: `False`.

</details>

<details>
<summary>Computer Vision — Training &amp; Loss (1 settings)</summary>

#### Computer Vision — Training & Loss

- **`epochs`** *(optional)* — Training epochs. Default ``10``. Default: `10`.

</details>

<details>
<summary>Additional Settings (1 settings)</summary>

#### Additional Settings

- **`alpha`** *(optional)* — Weight on the true-label loss vs. distillation loss. Default ``0.5``. Default: `0.5`.

</details>

<details>
<summary>Additional settings (5 settings)</summary>

#### Additional settings

- **`device`** *(optional)* — Torch device string. Default ``'cpu'``. Default: `'cpu'`.
- **`lr`** *(optional)* — Adam learning rate. Default ``1e-4``. Default: `0.0001`.
- **`pretrained`** *(optional)* — Whether the student uses pretrained weights. Default: `True`.
- **`student_model_name`** *(optional)* — TorchModel architecture name for the student. Default ``'maxvit_t'``. Default: `'maxvit_t'`.
- **`temperature`** *(optional)* — Softmax temperature for distillation. Default ``2.0``. Default: `2.0`.

</details>

</details>

## 4. Run

Return to the notebook, configure inputs, and run its analysis cell only when ready. Long operations report progress through spaCR's logging system; set `SPACR_LOG_LEVEL=DEBUG` before starting Jupyter for more detail.

## Outputs and next steps

A trained student model and recorded training provenance.
