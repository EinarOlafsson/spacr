# Item 1003 (bug): every app's shipped defaults pass the settings validation

- **Found by:** Work Codex audit28, re-checked on nightly 04ec46d370 (2026-11-20 18:57): `set_default_settings_preprocess_generate_masks` ships `batch_size` 0 and `validate_settings(..., 'mask')` refuses it ("must be at least 1").
- **Base:** landing train 6 head 20e35b6846 (item 987b and its follow-up changed `batch_size` and `mask_eval_batch_size`). The audit28 branch (53e91b5b17) is already contained in it.

## Which side is wrong

Item 987b documents `batch_size` for mask generation as "0 or 'auto' picks the largest batch that fits the free GPU memory and RAM" (tooltip; `spacr._mask_batches._batch_size_request` accepts 0, -1, None, 'none' and 'auto'). OPS runs the same mask batch plan and ships 0 as well. So the **validator** is wrong, not the defaults. A sweep of every app found two more validator disagreements with documented defaults:

| App | Shipped value | Documented meaning | Validator said |
|---|---|---|---|
| mask, timelapse, ops | `batch_size` 0 (also 'auto') | automatic batch | "must be at least 1"; 'auto' "is a str, but int is expected" |
| analyze_plaques | `diameter` 0 | "uses the model's own scale" (tooltip; plaque.py passes None) | "must be greater than zero" |
| barcode_qc | `count_data` 'path to unique_combinations.csv' | "(str or list)" | "is a str, but list is expected" |

## Change

- `spacr.validate`: for the apps in `_AUTOMATIC_BATCH_APPS` (mask, ops) `batch_size` is checked by the mask plan's own `_batch_size_request` (negative or non-numeric values are still refused, and so is a fraction such as '60.5', which the plan would silently truncate), and its type is `(int, str)` in `_APP_TYPE_OVERRIDES` (mirrored in `spacr.cli`, where `timelapse` is the module key). `diameter` 0 is accepted for `analyze_plaques`. Every other app keeps "must be at least 1" for `batch_size` 0.
- `spacr.settings.expected_types`: `count_data` and `score_data` are `(str, list)`, as their tooltips say.
- No default, setting or user string changes; no translation task.

Three existing tests used `batch_size` 0 under the mask app as their example of an unusable value (`tests/test_validate.py`, `tests/test_cov_w2_3_validate.py`); they now use a negative value, which is still refused.

## Test

`tests/test_default_settings_validate_1003.py`: every `set_default_settings_*` and `get_*default*settings` function of `spacr.settings`, and every CLI module's shipped defaults under its own validate key, give no validation error other than a required input the user must still give (`src`, databases, tables, channels; listed in the test). Plus the batch sizes the mask plan accepts and those it must still refuse.
