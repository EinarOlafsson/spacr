# Notes from `spacr/validate.py`

Prose lifted out of `spacr/validate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (29 entries)
- [_Inventory](#_inventory) (2 entries)
- [_peek_planes](#_peek_planes) (1 entry)
- [_inventory](#_inventory) (1 entry)
- [_check_src](#_check_src) (4 entries)
- [_check_channels](#_check_channels) (2 entries)
- [coerce_expected_types](#coerce_expected_types) (3 entries)
- [_check_types](#_check_types) (2 entries)
- [_check_retired_keys](#_check_retired_keys) (3 entries)
- [_check_unknown_keys](#_check_unknown_keys) (4 entries)
- [_check_numeric_sanity](#_check_numeric_sanity) (4 entries)
- [_check_required_paths](#_check_required_paths) (5 entries)
- [_check_app_specific](#_check_app_specific) (6 entries)
- [describe_plan](#describe_plan) (1 entry)
- [_array_footprint](#_array_footprint) (1 entry)
- [describe_resources](#describe_resources) (6 entries)
- [run_preflight](#run_preflight) (1 entry)

## Module level

### lines 93-103

```python
APP_FUNCTIONS: Dict[str, str] = {
```

The names are the ``settings_type`` strings dispatched by spacr.gui_utils.run_function_gui, so a caller can pass the same key the GUI uses. Values are the function that would run.

Every app in spacr.qt.app.APPS that is not GUI-only belongs here, and tests/test_app_registry_parity.py fails when one does not. Four were missing until that test existed: `timelapse`, `motility` and `activation` had a Qt button and (for two of them) a CLI module but no entry here, so validate_settings(settings, 'timelapse') answered "unknown app" and ran the generic checks only; `invasion` had no entry in any registry outside the Qt bridge, so `spacr-run invasion` did not exist either.

### lines 110-112

```python
"classify_merged": "spacr.classify.classify",
```

One entry point over both classifier families. It calls deep_spacr or generate_ml_scores unchanged, so this validates the same settings the two original modules validate.

### lines 118-119  _(unsure)_

```python
"ops": "spacr.spacrops.ops_preprocess",
```

OPS folds onto Align & Stitch: it is stitching too, over a plate acquired in sequencing cycles.

### lines 138-142

```python
"anndata_export": "spacr.anndata_export.run_anndata_export",
```

Reads a finished measurements.db and writes one file. It has no rules of its own in _check_app_specific yet, but being named here is what stops pre-flight answering "unknown app 'anndata_export'; only the generic checks were run" -- and for an exporter the generic checks (src exists, holds a project, types are right) are the ones that matter.

### line 146  _(unsure)_

```python
APP_ALIASES: Dict[str, str] = {
```

Friendly spellings a caller (or a notebook) might reasonably use.

### lines 160-163

```python
"cellpose_all": "cellpose_masks",
```

Cellpose 4 ships one model, so "benchmark every model" had a single entrant and was cellpose_masks under another name. Aliased rather than dropped so pre-flight still recognises the old key instead of reporting it as an unknown app and running only the generic checks.

### lines 178-179

```python
pass
```

Plugin discovery records its own diagnostics; built-in validation remains useful even when third-party metadata cannot be loaded.

### lines 182-188

```python
DB_APPS = frozenset({"umap", "ml_analyze", "regression", "recruitment",
```

Apps whose ``src`` is a plate folder that must already contain measurements/measurements.db — see spacr.ml.perform_regression (``src + '/measurements/measurements.db'``), spacr.submodules .analyze_recruitment and spacr.io._read_and_join_tables. The two Toxo assays open it the same way: analyze_invasion via spacr.io._read_db and analyze_endodyogeny via spacr.io._read_and_merge_data, both on ``os.path.join(src, 'measurements/measurements.db')``.

### line 193  _(unsure)_

```python
MERGED_APPS = frozenset({"measure"})
```

Apps that read the merged/*.npy stacks produced by the mask pipeline.

### lines 196-199

```python
MASK_APPS = frozenset({"mask", "timelapse"})
```

Apps whose segmentation-channel rules are the mask pipeline's, because they run the mask pipeline: preprocess_generate_masks_timelapse is preprocess_generate_masks with tracking, and prints and returns on the same "at least one of cell_channel / nucleus_channel / ..." check.

### lines 202-205

```python
ALT_SRC_KEYS: Dict[str, str] = {
```

Apps whose input folder is not called ``src``. spacr.foreign.import_project takes ``images`` / ``masks`` / ``measurements`` — someone else's project — and writes a spaCR one to ``dst``; there is no ``src`` to check, and reporting "src is missing" for it was simply wrong.

### lines 222-225

```python
_EXT_SUFFIX = r"\.(?:tif|tiff|png|jpg|jpeg|bmp)$"
```

Mirrors spacr.utils._get_regex. Reproduced here (rather than imported) because spacr.utils pulls in torch, which would defeat the point of a one-second pre-flight check. The trailing extension group replaces the ``.{img_format}`` suffix that _get_regex interpolates.

### lines 865-866

```python
_EXPECTED_TYPE_OVERRIDES: Dict[str, Any] = {
```

Keys whose expected_types entry is narrower than the code that reads them. Enforcing the literal declaration here would reject correct settings.

### lines 868-870

```python
"src": (str, list),
```

The expected_types literal declares "src" twice; the second entry (str) shadows the first ((str, list)), but core.preprocess_generate_masks and measure.measure_crop both loop over a list of folders.

### lines 872-873  _(unsure)_

```python
"normalize": (bool, list),
```

Declared bool for the mask pipeline, but measure_crop *requires* a [lower, upper] percentile pair and refuses a bare True.

### lines 875-876  _(unsure)_

```python
"save": (bool, list),
```

core.preprocess_generate_masks expands a bool into [save]*3 itself, so either form arrives legitimately.

### lines 881-886

```python
_APP_TYPE_OVERRIDES: Dict[str, Dict[str, Any]] = {
```

``expected_types`` is one flat registry shared by every pipeline, so a key name two pipelines both use can only be declared once. ``masks`` is a bool there — the mask pipeline's "save the masks" switch — while spacr.foreign.import_project takes ``masks`` as the other lab's mask folder, or a list of them. Judging a foreign import by the mask pipeline's meaning turned a perfectly good settings file into a blocking pre-flight error.

### lines 1006-1016

```python
_APP_EXTRA_KEYS: Dict[str, frozenset] = {
```

Keys owned by a pipeline whose defaults factory lives outside spacr.settings, so `_known_setting_keys` — which is built from spacr.settings alone — cannot see them. Spelled out rather than imported because this module deliberately imports nothing heavier than spacr.settings, and spacr.foreign pulls spacr.convert with it. The list is pinned against spacr.foreign.default_settings by tests/test_app_registry_parity.py, so it cannot drift silently.

Without this, `spacr-run foreign` told the user to rename `measurements` to `measurement` — a key from a different pipeline that import_project does not read at all.

### lines 1055-1066

```python
"gradient_accumulation": "gradient_accumulation_steps",
```

ONE FILTER FOR THE PREVIEW AND THE RUN. organelle carried both a `_size` pair and an `_area` pair meaning the same thing, and they were read by DIFFERENT code: `_size` by the batch mask writer, `_area` by the shared filter the Qt live preview uses. So tuning the preview until it looked right and then pressing run applied a different filter, with nothing saying so. cell, nucleus and pathogen only ever had `_area`. ONE QUESTION, ONE ANSWER. The boolean sat beside `gradient_accumulation_steps`, and `steps = 1` already IS the off state, so the pair could disagree -- on with one step, off with eight. `settings._fold_gradient_accumulation` honours a stored `false` by collapsing the step count to 1, so a settings file in the wild keeps meaning what it meant instead of quietly starting to accumulate.

### lines 1068-1073

```python
"expected_end": "window_length",
```

RENAMED, NOT REMOVED (364). "Expected end" reads as a coordinate and the value is a LENGTH -- the parameter's own docstring had to say "window *length*, not an end coordinate", which is a name explaining itself away. `settings._fold_renamed_settings` moves an old key onto the new one before any default is filled in, so a settings file in the wild keeps working and this entry tells its owner what happened.

### lines 1075-1076

```python
"min_n": "min_observations_per_hit",
```

"min_n" is the minimum of an unnamed n. The n is OBSERVATIONS behind a hit -- wells -- which the tooltip had to spell out twice over.

### lines 1078-1079

```python
"min_cell_count": "min_cells_per_well",
```

"min_cell_count" counts cells and drops WELLS. The count is per well and the name does not say so, which is why the tooltip had to.

### lines 1081-1090

```python
"positive_control": "positive_control_id",
```

RENAMED, NOT WITHDRAWN. The setting is an IDENTIFIER looked up in `location_column`, and it sat beside three settings naming WELLS (`positive_control_wells` and friends) with nothing in the name to tell them apart. `settings._fold_renamed_settings` copies an old file's value to the new name before any default is filled in, so a settings CSV written before this keeps behaving exactly as it did.

The `generate_ml_scores` PARAMETERS of the same name are unchanged a public signature, already decoupled from the setting at four call sites that pass `pc=`/`nc=`.

### lines 1093-1096

```python
"controls": "nontargeting_control_grnas",
```

RENAMED. `controls` named non-targeting control gRNAs and shared its spelling with a figure panel key, a sweep payload field, a dependency token and a column constant -- none of which moved. See 364's classification for the site-by-site split.

### lines 1098-1102

```python
"control_wells": ("stain_baseline_wells", "analysis_excluded_wells"),
```

SPLIT, not renamed (357-Q6). It meant the invasion assay's stain baseline AND the wells Regression and sequencing drop before fitting, with different defaults and no way to set one without setting the other. `settings._fold_renamed_settings` sends an old value to BOTH, so a file written before the split behaves exactly as it did.

### lines 1104-1115

```python
"denoise": "",
```

THE EIGHT THE PACKAGE READ NOWHERE (357-Q4, answered 2026-09-09: retire all eight). Each had exactly one consumer in the generated map and in every case it was the setting's own defaults setter nothing read the value back. No replacement, so the message says the value has no effect rather than sending the reader somewhere.

TWO OF THEM DOCUMENTED A JOB THEY DID NOT DO, which is worse than a dead control and is the reason this list is worth reading: `mask_array` said it chose the labelled plane for the 'array' stream method, and `stream_dataset` takes that plane from `object_array` for both methods; `load_path_regex` said it selected already-exported crops, and nothing consults it. A user setting either got silence.

### lines 1126-1131

```python
"minimum_cell_count": "min_cells_per_well",
```

POINTS AT THE LIVE NAME, NOT AT THE ONE IT WAS MERGED INTO. This was `min_cell_count` until 2026-09-09, when that key was itself renamed to `min_cells_per_well` -- so the entry named a setting that no longer exists and sent its reader to a second dead end. A chain of renames is worse than no message: the user follows it, finds nothing, and has no reason to think the trail continues.

### lines 1158-1163

```python
"corrected_manders": "",
```

Retired 2026-09-02 with the deprecated M1_correlation_<t> /

M2_correlation_<t> columns it used to switch on. The correct coefficients it gated -- manders_m1, manders_m2 and manders_overlap_coefficient -- are now written unconditionally, so there is nothing left for it to choose and no replacement to name.

### lines 2096-2108

```python
_SIZE_BUDGET = 20000
```

The resource half of the dry-run card

WHY THIS IS SEPARATE FROM describe_plan. The plan answers "what would this do"; every number in it is a count of things that exist. This answers "can this machine finish it", and every number in it is a projection. Mixing them would let a projection be read with the confidence of a count.

THE RULE THIS MODULE FOLLOWS THROUGHOUT, AND WHICH MATTERS MOST HERE: a figure that cannot be derived is NAMED AND LEFT OUT, never guessed. A fabricated RAM ceiling that a run then sails past is worse than no card, because the user stopped watching.

## _Inventory

### lines 326-327  _(unsure)_

```python
array_planes: Optional[int] = None
```

last-axis length of one merged/ (or stack/) array: image channels plus any appended mask planes. This is what *_mask_dim indexes into.

### line 331  _(unsure)_

```python
raw_channels: Optional[int] = None
```

number of raw acquisition channels. This is what *_channel indexes into.

## _peek_planes

### line 379, trailing  _(unsure)_

```python
import numpy as np
```

local: keeps module import free of numpy's cost

## _inventory

### lines 499-500  _(unsure)_

```python
inv.raw_channels = planes
```

stack/ holds image channels only (masks are appended later, in merged/) — see spacr.io._load_and_concatenate_arrays.

## _check_src

### lines 636-637

```python
if app == "regression":
```

Regression's ``src`` is an output root, not an image or project source. Its dedicated check also handles the blank automatic value.

### lines 642-643  _(unsure)_

```python
return [Problem(ERROR, key, f"{key} is missing from the settings.", fix)]
```

spacr.core.preprocess_generate_masks raises ValueError('src is a required parameter').

### lines 680-681

```python
if not inv.merged_exists:
```

measure_crop lists settings['src'] for *.npy — an absent or empty merged/ means it silently processes zero files.

### lines 694-709

```python
problems.append(Problem(
```

A NESTED TREE IS AN IMPORT JOB, AND `consolidate` WAS THE

WRONG ANSWER TO OFFER FIRST. Measured on the two layouts consolidate's own tooltip names (instruction 375): consolidate flattens by prefixing the folder names onto the filename, and `_get_regex('cellvoyager')` then matched NONE of what it produced -- `A01_img_F001C01.tif`, `DAPI_plate1_A01_F001.tif` -- so following this advice cost a second copy of the plate and still found nothing. `spacr.image_import` reads the folder segments as part of the name and placed all 12 files of the per-well tree with well, field and channel.

consolidate is still named, second, because it is NOT retired: a per-well tree whose filenames already carry cellvoyager metadata is the one shape it parses and Import refuses rather than guesses at.

## _check_channels

### line 773, trailing  _(unsure)_

```python
continue
```

the type check reports this

### lines 826-828

```python
problems.extend(_collision_problems(settings, CHANNEL_KEYS, WARNING,
```

Collisions. Two objects segmented from the same stain is unusual but occasionally deliberate, so it is a warning; two objects reading the same *mask* plane always produces duplicate labels, so it is an error.

## coerce_expected_types

### lines 933-934

```python
continue
```

The key legitimately holds text -- a path, a regex, a model name. "30" is a name here, not a number.

### lines 936-938

```python
if bool in types:
```

BOOL BEFORE INT, because bool is a subclass of int: checking int first would turn "True" into an error and, worse, "1" into 1 for a key that wanted True.

### lines 951-953

```python
if number.is_integer():
```

EXACT ONLY. '60.0' is 60; '60.5' is not an int, and silently truncating it would change the run without saying so -- so it is left for the validator to report.

## _check_types

### lines 979-981

```python
continue
```

None means "skip this object / leave it unset" nearly everywhere in spaCR, and expected_types declares NoneType for only some of the keys that accept it, so flagging None would be pure noise.

### line 992, trailing  _(unsure)_

```python
continue
```

an int is an acceptable float

## _check_retired_keys

### lines 1186-1191

```python
from .settings import surviving_setting_name
```

NOT IN THE TABLE IS NOT THE SAME AS NOT RENAMED. A rename that belongs to a role FAMILY cannot be written out here -- 705 roles carry `_flow_threshold` -- so the literal table names at most the first spelling. It named `organelle_min_size` and said nothing about `organelleq_min_size`, which is the same setting in the slot beside it. The resolver knows the suffix rules, so ask it.

### lines 1196-1199

```python
from .object_roles import (split_role_setting,
```

WITHDRAWN FROM A WHOLE ROLE FAMILY, which the literal table cannot hold: 705 roles carry each of these, so naming the first spelling would leave every generated organelle slot silent -- the `organelleq_min_size` failure again.

### lines 1214-1216

```python
names = ", ".join(f"'{one}'" for one in replacement)
```

A SPLIT. One key that meant two things is now two keys, and both need naming: a message that offered only one of them would send half the readers to the wrong control.

## _check_unknown_keys

### line 1275  _(unsure)_

```python
continue
```

Answered by name, and better, in _check_retired_keys.

### lines 1277-1283

```python
from .object_roles import withdrawn_setting_reason
```

AND THE ROLE-FAMILY RENAMES THE LITERAL TABLE CANNOT HOLD. Without this, a legacy file gets TWO warnings for one key: the retirement message naming the real successor, and a fuzzy "did you mean" guessing at it. `difflib` matches `pathogen_Signal_to_noise` to `pathogen_signal_to_noise` at the 0.85 cutoff, so the pair differ only in confidence, and two messages about one key reads as two problems.

### lines 1288-1291

```python
continue
```

ALREADY ANSWERED BY NAME in _check_retired_keys, and better. For the withdrawn ones the fuzzy matcher is not merely redundant but WRONG: it pointed `<role>_intensity_threshold_method`, which held 'mean', at `<role>_intensity_threshold`, which holds a float.

### lines 1294-1307

```python
mine = _object_role_in(key)
```

NEVER SUGGEST A NAME FROM A DIFFERENT OBJECT ROLE, because role is exactly the axis a user cannot see they crossed. `difflib` scores on characters, and `cell_`/`nucleus_`/`pathogen_`/`organelle_` keys share every character after the prefix -- so a typo in one role's setting matches another role's at well over the 0.85 cutoff, and the message reads as helpful.

IT IS WORSE THAN SILENCE WHEN IT IS WRONG. Before the organelle preprocessing settings were declared (364), a user who worked out `remove_background_organelle` was told "did you mean 'remove_background_cell'?" -- and following that changes a DIFFERENT CHANNEL's preprocessing, quietly, on a run that then looks fine. A wrong suggestion gets FOLLOWED; silence at least gets investigated. 391 suppressed one instance of this; this is the rule behind it.

## _check_numeric_sanity

### lines 1338-1340

```python
if number is not None and (key.endswith("_diameter") or key == "diameter"):
```

Cellpose object diameters are divided into and squared by spacr.settings._get_object_settings (min = d**2/4), so zero or negative is meaningless.

### lines 1356-1357  _(unsure)_

```python
if number is not None and key == "n_jobs":
```

spacr.measure.measure_crop overrides n_jobs with cpu_count()-4, but every other pipeline passes it straight to a Pool / DataLoader.

### line 1370  _(unsure)_

```python
if number is not None and (key.endswith("_cellprob_threshold") or key in ("CP_prob", "CP_probabil...
```

cellprob_threshold is clamped to about -6..6 by Cellpose itself.

### line 1377  _(unsure)_

```python
if number is not None and (key.endswith("_flow_threshold") or key in ("FT", "flow_threshold")):
```

flow_threshold: 0 keeps only perfect masks, above ~3 keeps everything.

## _check_required_paths

### lines 1452-1453  _(unsure)_

```python
for key, label in (("grna_csv", "the gRNA barcodes"),
```

spacr.sequencing.generate_barecode_mapping reads all three CSVs to translate the row/column/gRNA barcodes it pulls out of the reads.

### lines 1462-1464

```python
for key, purpose in (("images", "the import reads their images"),
```

spacr.foreign.import_project raises ConfigurationError("import_project needs '<key>'") for each of these before it plans anything, so a pre-flight that stayed quiet about them would be worse than useless.

### lines 1478-1479  _(unsure)_

```python
problems.append(Problem(
```

import_project says so in the printed plan; saying it before the write starts is the point of a pre-flight.

### lines 1523-1525

```python
if app in ("classify", "classify_merged"):
```

BOTH SPELLINGS. `classify_merged` is the screen that took this rule's job over; without it here, scoring a dataset with no model_path fell through to the run itself.

### lines 1536-1537

```python
if not os.path.exists(custom_model):
```

spacr.spacr_cellpose prints 'Custom model not found' and returns without segmenting anything when the path is wrong.

## _check_app_specific

### lines 1614-1616

```python
if all(settings.get(k) is None for k in CHANNEL_KEYS):
```

core.preprocess_generate_masks prints 'At least one of cell_channel, nucleus_channel, pathogen_channel or organelle_channel must be defined' and returns.

### lines 1622-1643

```python
if settings.get("pathogen_channel") is not None:
```

pathogen_model: A CHECKPOINT PATH HERE IS HONOURED.

This used to warn that the setting was IGNORED, which was true while the only values anyone set were the pre-SAM toxo names. It stopped being true: `object.py` reads `pathogen_model` for `object_type == 'pathogen'` and hands it to `_resolve_cellpose_pretrained`, which returns an existing file as-is -- so a cpsam fine-tune loads. Saying "ignored" now would tell a user their working setting will be discarded.

Two things are still worth saying, and they are opposites:

a MISSING file is an ERROR, not a warning. The resolver raises FileNotFoundError rather than falling back to cpsam, deliberately -- segmenting with the wrong weights silently is worse than stopping. But it raises INSIDE the run, after the images are batched. Catching it here is the whole purpose of a validator: the same failure, before the time is spent.

a LEGACY NAME still resolves to cpsam with only a log line, so it is the case that DOES pass silently and is the one the original warning was really about.

### lines 1646-1648

```python
from .settings import CELLPOSE_MODEL_CHOICES
```

Read the dependency-light fallback used to build the settings menu. Importing the runtime resolver here would pull in torch/cv2 during a dry run, before any model is meant to load.

### line 1674  _(unsure)_

```python
normalize = settings.get("normalize")
```

measure_crop returns early on both of these.

### lines 1695-1700

```python
ratios = settings.get("dialate_png_ratios")
```

dialate_png_ratios is indexed per crop mode. A single value scalar or one-element list, which is the shipped default [0.2] now broadcasts to every mode, exactly as png_size always has, so it is no longer an error. This used to be an ERROR, which BLOCKED a run that is now correct; it was written when measure.py raised IndexError on the second mode.

### lines 1705-1707

```python
problems.append(Problem(
```

Short but not a single value: measure.py reuses the last entry for the remaining modes and says so. Worth a warning, not a refusal.

## describe_plan

### lines 1950-1951

```python
rows.append(("reads", inv.merged_dir))
```

measure_crop silently appends 'merged' when src does not end with it, so say where it would actually look.

## _array_footprint

### line 2170, trailing  _(unsure)_

```python
import numpy as np
```

local: keeps module import free of numpy's cost

## describe_resources

### line 2266

```python
read_bytes = 0
```

what would be read

### line 2322

```python
projected: Optional[int] = None
```

what would be written

### lines 2327-2329

```python
merged = per_field * fields
```

merged/ is an uncompressed .npy of the same pixels: a compressed source shrinks nothing here, and TIFF usually IS compressed, so this can exceed the input rather than match it.

### line 2333, trailing  _(unsure)_

```python
plane = footprint[0][0] * footprint[0][1] * 2
```

uint16 labels

### lines 2352-2355

```python
try:
```

Measurement converts merged signal planes to uint16. Looking only at one dtype cannot reveal whether values cross the ceiling, so the preflight performs the same plate scan the real run will use and says so before any database is opened.

### lines 2406-2408

```python
return ("Resources — nothing to project: no readable input was found "
```

A "disk free" row on its own is not a projection. Saying nothing was found is the honest answer; printing a card of zeros would read as a run that costs nothing.

## run_preflight

### lines 2447-2450

```python
try:
```

The resource card is best-effort: it stats the disk and asks torch about the GPU, and neither is worth failing a dry run over. A pre-flight that raises has denied the user the report it exists to give them.
