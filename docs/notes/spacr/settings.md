# Notes from `spacr/settings.py`

Prose lifted out of `spacr/settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [canonical_feature_selection](#canonical_feature_selection) (1 entry)
- [BarcodeSet.resolve_groups](#barcodesetresolve_groups) (1 entry)
- [barcode_set_from_settings](#barcode_set_from_settings) (3 entries)
- [Module level](#module-level) (111 entries)
- [_merge_declarations](#_merge_declarations) (2 entries)
- [_takes_an_argument](#_takes_an_argument) (1 entry)
- [set_default_settings_preprocess_generate_masks](#set_default_settings_preprocess_generate_masks) (21 entries)
- [set_default_plot_data_from_db](#set_default_plot_data_from_db) (1 entry)
- [_read_cellpose_models](#_read_cellpose_models) (1 entry)
- [cellpose_model_choices](#cellpose_model_choices) (1 entry)
- [downloaded_zoo_models](#downloaded_zoo_models) (3 entries)
- [_get_object_settings](#_get_object_settings) (4 entries)
- [set_default_umap_image_settings](#set_default_umap_image_settings) (5 entries)
- [get_measure_crop_settings](#get_measure_crop_settings) (22 entries)
- [set_default_classify](#set_default_classify) (1 entry)
- [set_default_analyze_screen](#set_default_analyze_screen) (3 entries)
- [_resolve_rename](#_resolve_rename) (2 entries)
- [surviving_setting_name](#surviving_setting_name) (1 entry)
- [_fold_renamed_settings](#_fold_renamed_settings) (2 entries)
- [set_default_train_test_model](#set_default_train_test_model) (4 entries)
- [set_generate_training_dataset_defaults](#set_generate_training_dataset_defaults) (6 entries)
- [deep_spacr_defaults](#deep_spacr_defaults) (8 entries)
- [get_analyze_recruitment_default_settings](#get_analyze_recruitment_default_settings) (1 entry)
- [get_map_barcodes_default_settings](#get_map_barcodes_default_settings) (1 entry)
- [get_train_cellpose_default_settings](#get_train_cellpose_default_settings) (1 entry)
- [set_generate_dataset_defaults](#set_generate_dataset_defaults) (1 entry)
- [_resolve_regression_analysis_choices](#_resolve_regression_analysis_choices) (4 entries)
- [_reject_a_threshold_that_cannot_mean_what_it_says](#_reject_a_threshold_that_cannot_mean_what_it_says) (3 entries)
- [get_perform_regression_default_settings](#get_perform_regression_default_settings) (49 entries)
- [_name_the_family_in_every_estimator_tooltip](#_name_the_family_in_every_estimator_tooltip) (2 entries)
- [_advanced_family_members](#_advanced_family_members) (2 entries)
- [_regroup_advanced](#_regroup_advanced) (2 entries)
- [get_setting_dependencies](#get_setting_dependencies) (17 entries)
- [get_setting_dependencies.permutation_is_certain](#get_setting_dependenciespermutation_is_certain) (1 entry)
- [get_setting_dependencies._is_nonparametric](#get_setting_dependencies_is_nonparametric) (1 entry)
- [get_setting_dependencies._level_is_read](#get_setting_dependencies_level_is_read) (1 entry)
- [parse_list](#parse_list) (1 entry)
- [check_settings](#check_settings) (7 entries)
- [set_annotate_default_settings](#set_annotate_default_settings) (4 entries)
- [set_default_generate_barecode_mapping](#set_default_generate_barecode_mapping) (2 entries)
- [get_default_generate_activation_map_settings](#get_default_generate_activation_map_settings) (1 entry)
- [get_analyze_plaque_settings](#get_analyze_plaque_settings) (5 entries)
- [set_graph_importance_defaults](#set_graph_importance_defaults) (1 entry)
- [set_analyze_invasion_defaults](#set_analyze_invasion_defaults) (2 entries)
- [get_plot_data_from_csv_default_settings](#get_plot_data_from_csv_default_settings) (1 entry)
- [get_automated_motility_assay_default_settings](#get_automated_motility_assay_default_settings) (9 entries)
- [_set_organelle_defaults](#_set_organelle_defaults) (11 entries)
- [_fold_toxoplasma, 2026-09-19](#_fold_toxoplasma-2026-09-19) (3 entries)
- [RENAMED_SETTINGS, 2026-09-19](#renamed_settings-2026-09-19) (2 entries)

## canonical_feature_selection

### line 74  _(unsure)_

```python
raise ValueError(
```

bool is an Integral, but True is not an honest spelling of channel 1.

## BarcodeSet.resolve_groups

### lines 474-478

```python
taken = {}
```

AND NO TWO BARCODES MAY LAND ON ONE GROUP. A barcode that accepts an older spelling of its group can fall back onto the group another barcode was given, and then both are handed the same captured text and counted as though they were different -- silently, because every column of the output is full.

## barcode_set_from_settings

### lines 557-562

```python
return None
```

ABSENT, BLANK, OR EMPTIED IN A PANEL ARE ONE ANSWER. A settings file written before sets existed has no key at all, a panel field cleared by hand arrives as an empty string, and a list a user emptied arrives as an empty list. All three mean the run decodes the three barcodes spaCR shipped, which is the only reading that cannot surprise somebody.

### lines 577-578  _(unsure)_

```python
entry = item
```

Already spelled out in full by whoever built it. Nothing is filled in over an explicit decision.

### lines 614-618

```python
count_columns = _SHIPPED_COUNT_COLUMNS
```

THE THREE SHIPPED BARCODES KEEP THEIR COUNTING ORDER. Counts have always been grouped by row, then column, then guide, while the reads list the column first, so taking the entry order here would reorder the rows and the header of every count table a user already has for no reason they asked for.

## Module level

### lines 623-652

```python
_DEFAULTS_REGISTRY = {}
```

The defaults seam — how a module ships settings without editing this file

Everything below this block is the settings of the modules that existed when this file was one file: ~3800 lines of defaults factories, types, tooltips and categories, all of which a new module used to have to append to. Six workstreams appending to the same file is six merge conflicts, and the file is nobody's to own.

A module registers instead, at import time, from its own file:

from spacr.settings import register_defaults

def _defaults(settings=None):

settings = dict(settings or {}) settings.setdefault("src", "") settings.setdefault("bins", 32) return settings

register_defaults(

"graph_builder", _defaults, expected_types={"bins": int}, tooltips={"bins": "(int) - Histogram bins. Default 32."}, categories={"General": ["bins"]})

The existing `set_default_*` / `get_*_settings` functions are NOT touched and NOT auto-registered: they are reached through the dispatch in `qt.screens.settings_model.resolve_default_settings`, and mirroring them here would create a second answer to "what are Mask's defaults?". This registry holds only what registers itself.

### lines 2055-2056  _(unsure)_

```python
"expected_end": "window_length",
```

"window length, not an end coordinate" is what the parameter's own docstring already had to say, which is the argument for the name.

### lines 2058-2060

```python
"min_n": "min_observations_per_hit",
```

"min_n" says the minimum of an unnamed n. The n is OBSERVATIONS wells behind a hit -- and the tooltip had to spell that out twice over ("gRNA hits need n_grna > min_n, gene hits need n_gene > min_n").

### lines 2062-2064

```python
"min_cell_count": "min_cells_per_well",
```

"min_cell_count" counts cells and drops WELLS, and the count is per well -- which the tooltip has to say ("Wells with fewer than this many cells are dropped") because the name does not.

### lines 2066-2081

```python
"positive_control": "positive_control_id",
```

THE SETTING IS AN IDENTIFIER, NOT A WELL AND NOT A CONDITION, and its own tooltip has always had to say so: "Identifier of the positive-control class. In ML screening it is the value in location_column". Three neighbouring settings name WELLS `positive_control_wells`, `negative_control_wells`, `mixed_control_wells` -- so a reader meeting `positive_control` beside them has no way to tell that this one is a value looked up in a metadata column rather than a plate address.

THE FUNCTION PARAMETERS ARE NOT RENAMED WITH IT. `generate_ml_scores` and `_resolve_controls` take `positive_control='c2'` as PUBLIC arguments, and four call sites already pass them as `pc=`/`nc=` -- so the setting name and the kwarg are already decoupled, and renaming a public signature is a break this rename does not need to make. The tooltips say which is which.

### lines 2084-2091

```python
"controls": "nontargeting_control_grnas",
```

`controls` IS A COMMON WORD DOING FOUR JOBS, and the setting is only one of them. Its own tooltip already says the true meaning "Non-targeting control gRNA identifiers" -- and `object_roles` has carried a label OVERRIDE for it since before this rename, with the reason written beside it: "`controls` names guide or gene identifiers, whereas the neighbouring control settings name wells". A setting that needs a label override to be understood is a setting whose name is wrong.

### lines 2093-2097

```python
"control_wells": ("stain_baseline_wells", "analysis_excluded_wells"),
```

A SPLIT, NOT A RENAME (357-Q6). One key meant the invasion assay's stain baseline AND the wells Regression and sequencing drop before fitting, with different defaults and no way for a user to set one without setting the other. The old value goes to BOTH new names, so a settings file written before the split behaves exactly as it did.

### lines 2099-2107

```python
"minimum_cell_count": "min_cells_per_well",
```

THE FOUR THE DOCTOR ALREADY REPORTED AND THE RUN NEVER PERFORMED. `spacr.validate.RETIRED_SETTINGS` has named all four as renames since they landed, so `spacr-doctor` said "renamed to X" about a file whose value the run then dropped on the floor and replaced with a default. Measured before adding them: each was absent here, the new key was never created, and the old key sat in the dict inert.

NOTHING CHECKED THAT THE TWO TABLES AGREED, which is how four of them accumulated. `tests/test_the_two_settings_tables_agree.py` now does.

### lines 2110-2117

```python
}
```

`organelle_min_size` / `organelle_max_size` ARE DELIBERATELY NOT HERE. They are the first spelling of a ROLE-FAMILY rename, and `object_roles.RENAMED_SETTING_SUFFIXES` already carries `min_size -> min_area` for every slot -- so a literal entry would be a second route to the same answer for `organelle` and no route at all for `organelleq`, which is the shape of the bug being fixed. Verified by deleting them: the agreement test still passes, because the suffix rule performs them.

### line 2793  _(unsure)_

```python
'permutation': 'guide_permutation',
```

Accepted spellings so a settings CSV can say either.

### lines 3892-3895

```python
"illumination_correction": bool,
```

THE FLAT-FIELD CORRECTION, declared here because Measure offers it now and Illumination still does: two modules reading one setting is exactly the case this table exists to make deterministic, rather than leaving the type to whichever module imported first.

### lines 3905-3907

```python
"dst": str,
```

Shared file/output seams used by independently registered analysis modules. Declaring them here makes tooltip/type ownership deterministic instead of whichever module happened to import first.

### lines 3918-3921

```python
"plaque_model": str,
```

Plaque analysis can optionally split a plate image into wells and use their physical diameter as a ruler. These six keys are all exposed by get_analyze_plaque_settings, so each needs the same declared contract the worker consumes rather than being dropped by check_settings.

### line 3950, trailing  _(unsure)_

```python
"timelapse_frame_limits": (list, type(None)),
```

This can be a list of lists

### line 3951  _(unsure)_

```python
"timelapse_remove_transient": bool,
```

"timelapse_frame_limits": (list, type(None)),  # This can be a list of lists

### line 3972  _(unsure)_

```python
"t_stack": bool,
```

4D (Beta)

### lines 3989-3994

```python
"cell_mask_dim": (int, type(None)),
```

None, and see the `*_chann_dim` trio further down for the argument. These three now agree with `organelle_mask_dim`, which has always declared it, and with the code that reads them: measure_crop tests every one through `is not None` and skips that object's measurements, crops and links when it is unset, so the declaration was narrower than the pipeline it describes.

### line 4005, trailing  _(unsure)_

```python
"png_size": list,
```

This can be a list of lists

### line 4017, trailing  _(unsure)_

```python
"pathogen_loc": (list, list),
```

This can be a list of lists

### line 4019, trailing  _(unsure)_

```python
"treatment_loc": (list, list),
```

This can be a list of lists

### lines 4020-4024

```python
"channel_of_interest": (int, str, list, type(None)),
```

NOT `int` ALONE. `utils.filter_dataframe_features` has always taken a list, 'morphology' and a free-text column fragment, and this declaration admitted exactly one of the four -- so three documented ways of choosing what a model trains on were unreachable from the panel and refused by `check_settings` (236 A2).

### lines 4029-4041

```python
"pathogen_limit": (bool, int),
```

Their own tooltip says "(int, bool, or None)" and four factories ship True (analyze_screen, deep_spacr, generate_training_dataset, analyze_recruitment's siblings), but they were declared plain int -- so check_settings answered those modules' OWN defaults with "Expected type int for 'nuclei_limit', but got 'True'" and gui_core's `if len(errors) > 0: return` refused to start the run.

(bool, int) rather than (bool, int, type(None)): the three-way tuple has no branch of its own and would fall into the generic loop, where bool() is tried first and bool('10') is True -- turning "keep cells with 10 or fewer nuclei" into "keep single-nucleus cells only". The (bool, int) branch already handles None, the true/false spellings and an integer, which is exactly the documented set of values.

### lines 4047-4061

```python
"background": (int, float),
```

(int, float), NOT str. Declared `str` and contradicted by everything around it: the tooltip says "(float) - Per-channel background level in raw intensity units", every factory ships a number (100, and 200 for Cellpose training and plaque analysis), and `io.py` compares it to pixels -- `np.where(image < background, 0, image)`.

THE COST WAS NOT COSMETIC. `validate._check_types` returned severity ERROR -- "background=200 is a int, but str is expected" -- for analyze_plaques, cellpose_masks and cellpose_all on their UNTOUCHED defaults, printed on every GUI run and blocking in the batch queue. And `coerce_expected_types`, whose whole job is restoring types after a CSV round trip, dutifully PRESERVED the string, so the comparison above raised UFuncTypeError on a value the user never chose.

Found by instruction 364's audit of the assay sub-modules, 2026-09-02.

### lines 4192-4193  _(unsure)_

```python
"classes": dict,
```

The class DEFINITIONS. The ordered training-folder names are `class_folder_names`; see spacr.classify_classes.

### lines 4205-4209

```python
"mixed_precision": bool,
```

`gradient_accumulation` IS NOT HERE ANY MORE. It was retired into `gradient_accumulation_steps` -- `steps = 1` already says "do not accumulate" -- and `spacr.validate.RETIRED_SETTINGS` names it, so leaving the type row behind declared it live and withdrawn at once. `_fold_gradient_accumulation` still honours a stored `false`.

### line 4219, trailing  _(unsure)_

```python
"pathogen_plate_metadata": (list, list),
```

This can be a list of lists

### line 4220, trailing  _(unsure)_

```python
"treatment_plate_metadata": (list, list),
```

This can be a list of lists

### lines 4221-4231

```python
"cell_chann_dim": (int, type(None)),
```

AN OBJECT THAT IS NOT IN THE RUN NAMES NO PLANE. Every key that points at a plane of the stack -- the raw channel an object is imaged in, the channel paired with its mask, and the plane its mask sits on -- accepts None, because a screen with no nucleus has no nucleus channel and no nucleus mask plane, and a settings file made to carry a number for one is made to make a claim about an object that is not there.

These three were the odd ones out: `organelle_chann_dim` and every `*_channel` already declared None, `analyze_recruitment` already DOCUMENTS None here ("leave it None and cells are not filtered at all"), and the declaration was the only thing still saying otherwise.

### lines 4241-4245

```python
"dependent_variable": (str, list),
```

A list fits every named response in one run, correcting each as its own multiple-testing family. The screen this was built for has two independently trained classifiers (XGBoost and MaxViT) whose agreement is the evidence, and running them as two separate jobs made that comparison a manual step.

### lines 4262-4263

```python
"p_threshold_alpha": (int, float),
```

The significance line the RUN draws, so the exported hit list and the volcano cannot be cut to two different rules (instruction 135).

### lines 4266-4269

```python
"rra_alpha": (int, float),
```

Robust rank aggregation, and the group lasso's penalty. Declared before their readers exist because a knob with no entry here is dropped by check_settings ("Warning: Key ... not found in expected types"), which is how a Tk panel once discarded the regression_type a user picked.

### lines 4273-4274

```python
"count_grna_column": str,
```

The count CSV's two variable column names, hard-coded in spacr.ml until instruction 135 B.

### lines 4280-4283

```python
"tolerance": (int, float),
```

The regression keys perform_regression indexes directly. They had no entry here at all, so the GUIs could not render them, check_settings could not coerce them out of a settings CSV and validate could not type-check them -- see get_perform_regression_default_settings.

### lines 4286-4290

```python
"score_column": str,
```

NOT a regression key any more (instruction 135 A): the regression module's duplicate of `dependent_variable` is retired in get_perform_regression_default_settings. It stays declared for `interpret_vision_model` and `hit_investigation`, where it names the CNN score column.

### lines 4292-4294

```python
"y_lims": (list, type(None)),
```

y_lims styles the volcano and the regression module no longer offers it the plot scales to the data and is rescaled on the plot. It stays declared for the plotting helpers that take it directly.

### lines 4296-4308

```python
"regression_type": (str, type(None)),
```

The model-choice keys. The first three -- regression_type, alpha and random_row_column_effects -- were categorised, tooltipped and defaulted but had NO entry here, and check_settings DROPS any key it cannot type ("Warning: Key 'regression_type' not found in expected types"), so the Tk panel discarded whichever model the user picked and get_perform_regression_default_settings then restored 'ols'. A run configured as 'mixed' fitted OLS, wrote it to results/<...>/ols/ and said nothing anywhere. The rest are the per-model knobs spacr.ml's backends read; see spacr.ml.REGRESSION_SETTINGS_USED for which type reads which.

regression_type is (str, NoneType) because None is the documented "choose from the response distribution" value; alpha is (int, float, str, NoneType) because 'auto'/None select it by cross-validation.

### lines 4310-4313

```python
"regression_backend": str,
```

WHO fits it (instruction 141). str, never None: an unset backend is 'statsmodels' by name, filled in by get_perform_regression_default_settings, so nothing downstream has to decide what None meant.

### lines 4316-4318

```python
"model_plate_position": bool,
```

Instruction 143 A: whether rowID and columnID are terms at all, which random_row_column_effects never decided -- it only chose fixed vs random for terms that were always in.

### lines 4345-4353

```python
"pipeline_style": str,
```

The three v1/v2 pipeline keys. They were defaulted by set_default_settings_preprocess_generate_masks and listed under the "Advanced" category -- so both GUIs built a widget for them -- but were declared nowhere, and an undeclared key is not merely untyped: Tk's check_settings reports "not found in expected types" and `continue`s, dropping the key from the dict it returns, and Qt's _coerce_to_expected_type hands the raw widget string through. Switching the pipeline to v2 in the panel therefore did nothing at all, and batch_fields reached the streaming loop as the string '8'.

### lines 4372-4376

```python
'sample':(int, list, type(None)),
```

(int, list or None), as the description says. It was the VALUE None, which is not a type at all: validate.validate_settings does `isinstance(value, (None,))` on it and died with "TypeError: isinstance() arg 2 must be a type" -- the preflight check crashed on any run that set 'sample'.

### lines 4671-4688

```python
'regex': str,
```

Keys their module ships AND puts a widget on, but that nothing ever declared. Undeclared is not "untyped and otherwise fine": the settings panel builds its widget map from the module's own defaults factory, so each of these became a field, and check_settings answers a field it cannot type with errors.append("Warning: Key '<k>' not found in expected types") and `continue`. gui_core.import_settings then does if len(errors) > 0: return so pressing Run did nothing at all, with the reason only in the log queue. Map Barcodes was unstartable from the Tk GUI for eleven of these at once; Training Dataset, Activation Maps, Endodyogeny, Class Proportion and Check Cellpose Models each had their own.

Every type below is the type of the value that module already ships, and tests/test_settings_full_coverage.py round-trips each factory's defaults through check_settings to prove it: a wrong type here would show up as a value that does not survive its own default.

### lines 4693-4698

```python
'barcode_set': (list, tuple, dict, type(None)),
```

A LIST OF BARCODES INSTEAD OF THREE NAMED ONES. Absent, which is what every settings file written so far has, means the three above: `barcode_set_from_settings` returns None and the run decodes exactly what it decoded before sets existed. Declared rather than merely tolerated so a panel can collect one and `check_settings` keeps the value instead of dropping it.

### lines 3610-3630

```python
'folders': (list, type(None)),
```

THE TWENTY-ONE THAT HAD PROSE AND NO TYPE (397, typed 2026-09-19). Each had a tooltip in `tooltips` and no entry here, so a user read what the setting takes and neither `check_settings`, `spacr.validate` nor the CLI could hold a value to it. Every one is read by live code -- `generate_score_heatmap`, `interpret_vision_model`, `analyze_percent_positive`, the picture panel's channel choice -- so none was a stale tooltip, and each takes the type its own tooltip states. `None` is admitted where the tooltip says "Default None". `threshold` is the widest because two apps share the name: percent-positive ships 2000, Annotate ships '' and takes quantile codes and lists. The picture channels are plain `int`, as their tooltip says: a declared `None` would turn any `*_channel` into a clearable plane box (`_is_clearable_plane_setting`), and the picture panel already offers "not drawn" through its own picker. `barcode_qc` is deliberately NOT here; `tools/build_setting_consumer_map.py` says why.

### lines 4769-4775

```python
_IMAGE_SOURCES = {
```

NOT here, deliberately: `png_type`, `size` and

`write_random_annotation_column`. They were renamed rather than retired, and each still HAS A READER -- the alias translation in spacr.training_basis and spacr.classify_classes, plus io.py's direct fallback for png_type. A setting with a reader is an alias, not a dead setting, and the registry guard is right to refuse it. They stop being OFFERED (their defaults are gone, so no control is built) while an old settings CSV still runs unchanged.

### lines 4860-4870

```python
'image_source':
```

The AI Console's own settings.

These five had no entry at all, so `install_api_tooltips` found no description, retargeted nothing onto a label and left the controls with an empty tooltip -- the only settings dialog in spaCR with no hover help on any row.

Streaming crops instead of reading them off disk (230).

### lines 4897-4899  _(unsure)_

```python
'holdout_plate':
```

Optional outlier removal before annotation.

### lines 5034-5042

```python
"image_type": "(str) - Exported crop folder to read: 'cell_png', 'nucleus_png', 'pathogen_png', o...
```

The Cells tab's picture settings (instruction 176 B).

HERE AND NOT IN A NEW TABLE. The settings window already reads this dict; seventeen of its twenty-five keys simply had no entry and so no hover help. A second table beside it would be a second answer to what "normalize" means, which is the 145 failure this tab has already made once with the channel names.

### lines 5165-5167

```python
"z_stack": '(bool) - When True, spaCR requires the array to contain an explicit z dimension and r...
```

3D (Beta)

These describe what the z plumbing does, and say plainly where it stops. A user must not read these and believe spaCR measures volumes today.

### lines 5176-5179

```python
"t_stack": '(bool) - When enabled, spaCR requires each field to be a (T, Z, Y, X) volume. Standar...
```

4D (Beta)

The time axis on top of the z axis. These say plainly where the 4-D plumbing stops, for the same reason the 3D ones above do: a user must not read them and believe spaCR tracks objects through volumes today.

### line 5614

```python
"annotation_column": "(str) - Integer column in the png_list table that stores manual class label...
```

Descriptions filled in for settings that previously had no tooltip

### lines 5638-5643

```python
'level': "(str) - Result level. For regression, 'both' writes results_grna.csv and results_gene.c...
```

ONE KEY, TWO MODULES. 'level' was the proportion plots' unit of replication long before instruction 132 gave the regression a level of its own, and this table is keyed by NAME with no module scope so the hover has to be right in both panels or it is wrong in one. Renaming either side was not available: a new tooltip key has to exist in all nine i18n catalogs, which are generated elsewhere.

### lines 5684-5691

```python
'class_balance': "(str) - Correction for imbalance among training classes. 'none' preserves sampl...
```

NOTE: a SECOND 'crop_source' description used to sit here, documenting values 'auto' / 'png' / 'merged'. Those are not what the code accepts -- crop_source.CROP_SOURCES is ('pre_generated', 'on_demand', 'generate'), with 'auto' kept only as an alias for pre_generated. Being a duplicate key in the same dict, it was silently shadowed by the correct entry further down, so the right tooltip showed by luck of ordering rather than by design. Removed; the accurate one is the only one now.

### lines 5818-5823

```python
timelapse_settings = ['fps', 'timelapse_mode', 'trackastra_model', 'trackastra_linking', 'ultrack...
```

Keys owned by the standalone Timelapse module (spacr.qt app key 'timelapse'). NOTE `timelapse` itself is NOT in this list: it lives in the "General" category because the Tk GUI reveals the "Timelapse" category only once that box is ticked (see category_dependencies), so the toggle cannot live inside the category it controls. Consumers that want "everything timelapse" should use `timelapse_settings + ['timelapse']`.

### lines 5835-5870

```python
_organelle_all_settings = [
```

How the settings panel is grouped: the Qt section boxes

(qt/screens/settings_model.SettingsWidgets.build_sections) read this map and nothing else. One entry = one heading, rendered in the order written here.

Three rules keep it usable, and tests/test_settings_categories.py enforces all three: 1. Every key produced by a module's set_default_* / get_*_settings helper appears here. An uncategorised key is not grouped at all: Tk pins it to the top of the panel as an always-visible field and Qt dumps it in the trailing "Other" section. 2. No key appears twice. A duplicate renders twice in Tk (and each copy is shown/hidden by a different heading) and is silently dropped from the second section in Qt. 3. A setting that TRIGGERS a category - see category_dependencies and category_integer_dependencies below - must live outside the category it reveals, or ticking it off hides the control that turns it back on. That is why `timelapse` sits in General and not in "Timelapse", and why organelle_channel / organelle_mask_dim sit in General and not in "Organelle". ORGANELLE, SPLIT IN TWO. Instruction 72 item 5.

One heading used to hold FIFTY-THREE settings -- the most over-configured object class in the tool, and a biologist who knew they were imaging lysosomes had to scroll past organelle_ridge_sigmas to reach the diameter.

`organelle_basic_settings` keeps only what a biologist recognises without knowing how segmentation works. Everything else is advanced. The membership is DERIVED from `organelle_types.BASIC_SETTINGS`, not typed out here, so the split cannot drift from the module that defines what basic means -- but both NAMES appear in the `categories` literal below, the way motility's two do, because that literal is where category names are declared and checked.

MOVED, NOT HIDDEN. Every advanced setting is still in the panel, still editable, still in the settings dict. A setting that leaves the UI while staying in the dict is how a run gets a value nobody can see; this project has eleven phantom settings from exactly that (instruction 61).

### line 5872  _(unsure)_

```python
"organelle_morphology", "organelle_method", "organelle_diameter",
```

what to detect

### line 5874  _(unsure)_

```python
"organelle_mask_within_cells", "organelle_rolling_ball", "organelle_rolling_ball_radius", "organe...
```

clean the image first

### line 5876  _(unsure)_

```python
"organelle_adaptive_block_size", "organelle_adaptive_offset",
```

method: adaptive

### line 5878  _(unsure)_

```python
"organelle_tophat_radius", "organelle_watershed_spots", "organelle_log_min_sigma", "organelle_log...
```

method: otsu / adaptive / log / dog (spots)

### line 5884  _(unsure)_

```python
"organelle_morph_radius", "organelle_fill_holes",
```

morphology: irregular

### line 5886  _(unsure)_

```python
"organelle_model_name", "organelle_cellprob_threshold", "organelle_flow_threshold", "organelle_re...
```

method: cellpose

### line 5888  _(unsure)_

```python
"organelle_unet_model_path", "organelle_unet_threshold",
```

method: unet

### line 5890  _(unsure)_

```python
"remove_background_organelle", "organelle_background", "organelle_signal_to_noise", "organelle_mi...
```

filter the detected objects

### line 5892  _(unsure)_

```python
"summarize_organelles_by",
```

what to write out

### line 5896  _(unsure)_

```python
_organelle_all_settings.insert(0, "organelle_type")
```

The one visible choice goes first, ahead of the six it stands in for.

### lines 5911-5915

```python
organelle_basic_settings.insert(0, NUMBER_OF_ORGANELLES)
```

HOW MANY ORGANELLES leads the category whose size it decides. It is not one of the per-slot settings -- it belongs to no slot, and the generators below skip it because it does not carry a slot's prefix -- so it is placed here rather than in `organelle_types.BASIC_SETTINGS`, which says which of ONE slot's settings a biologist meets first.

### lines 5922-5925

```python
"General": ["cell_mask_dim", "cytoplasm", "cell_chann_dim", "cell_channel", "nucleus_chann_dim", ...
```

'normalize' moved here from "Advanced". It is a top-level toggle for how every image in the run is scaled, set by seven different modules, and burying it under "rarely-touched knobs" was wrong in all of them - not least Classify, where it shapes the training set.

### lines 5928-5929

```python
"Cellpose": ["custom_model", "fill_in", "from_scratch", "n_epochs", "width_height", "target_size"...
```

How Cellpose runs, including the optional saved Cellpose checkpoint. Classify uses custom_model_path instead and never receives this key.

### lines 5939-5944

```python
"Organelle": organelle_basic_settings,
```

One heading for the whole organelle workflow, ordered the way it is set up: what to detect -> clean the image -> the knobs of the chosen organelle_method -> filter the objects -> what to summarise. The per-method blocks used to be eight separate headings gated on organelle_method; they are sub-ordered here instead, so the knobs that do not apply to your method are simply further down the list.

### lines 5952-5966

```python
"Measurements": ["save_measurements", "calculate_correlation", "spatial_measurements", "spatial_n...
```

Which objects are measured, which features are computed, and which of them survive into the analysis table. Plot-only knobs that used to live here (image_nr, dot_size, remove_image_canvas) moved to "Plot".

Three groups arrived here in the regroup:

the per-object minimum sizes and merge_edge_pathogen_cells, which only measure_crop sets. They sat under the Cell / Nucleus / Pathogen SEGMENTATION headings, so the Measure module rendered three headings holding one or two size filters each and no segmentation at all. nuclei_limit / pathogen_limit, from "Advanced". They decide whether the nucleus and pathogen tables are joined onto the object table which rows exist, not a tuning knob. parasite_table / compartment, from "Invasion Assay", which name the table and compartment the objects are read from. Leaving them there made the Replication module render a heading called "Invasion Assay".

### lines 5969-5978

```python
"Illumination Correction": ["illumination_correction", "illumination_model", "illumination_estima...
```

The flat-field correction Measure applies before it measures anything. One heading, not the four the Illumination screen splits them across: from inside a measure run they are a single decision -- correct these fields or do not -- and the estimator, the sampling and the QC are how that one decision is carried out.

Listed here rather than contributed by spacr.illumination through `register_defaults`, because a category registered at import time is in the map only for a process that imported that module, and this heading has to exist for anything that groups the Measure settings.

### lines 5981-5985

```python
"Object Crops": ["save_png", "crop_mode", "png_size", "png_channel_mapping", "png_dims", "dialate...
```

png_dims stays listed although it is no longer rendered: it has no default any more, so convert_settings_dict_for_gui never builds a widget for it, but it is still a key the pipeline reads from an older CSV and a key that falls out of `categories` altogether is one nothing can tell a user about.

### lines 5988-5995

```python
"Plate Layout & Controls": ["well_detection", "well_confidence", "well_pad", "plate_format", "wel...
```

The plate map: which wells hold which condition, which wells are the controls, and how they are labelled. Gathers the per-object condition lists that used to sit inside the Cell / Nucleus / Pathogen segmentation headings, where they had nothing to do with segmentation. ...plus how the wells are grouped for reporting: group_column / level / change_plate came from "Invasion Assay", where they were shared with the replication assay and so gave that module a heading named after an assay it does not run.

### lines 5998-6003

```python
"Training Classes": ["dataset_mode", "classes", "class_folder_names", "class_metadata", "metadata...
```

How the labelled set is assembled, in the order it is assembled: which rule defines a class -> what the classes are -> which crops -> how many -> how they are split. 'test_split' came from "Model Training": generate_training_dataset is what consumes it, writing the train/ and test/ folders before any model exists. The four metadata_item_* keys had no category at all and printed under "Other".

### line 6005  _(unsure)_

```python
"Training Classes": ["dataset_mode", "classes", "class_folder_names", "class_metadata", "metadata...
```

Which classifier model, and how it is fitted.

### lines 6007-6018

```python
"Training Classes": ["dataset_mode", "classes", "class_folder_names", "class_metadata", "metadata...
```

The classical (non-image) screen classifier fitted on measured features spacr's "Classify (ML)" module. These knobs used to be split three ways between General, Advanced and the regression heading. WHAT DEFINES A CLASS -- and nothing about where the pixels come from. `png_type`, `size` and `write_random_annotation_column` are gone from here because they are in DEAD_SETTINGS: renamed, duplicated, or replaced by the Classes dict. An old CSV still runs; the panel stops offering two controls for one thing. `write_random_annotation_column` is kept in the map although nothing offers it any more: it is an ALIAS, not a dead setting (the Classes translation still reads it), and a key that falls out of `categories` altogether is one nothing can ever group again.

### lines 6021-6028

```python
"Computer Vision Data Source": ["image_source", "image_size", "size", "train_channels", "stream_m...
```

WHERE THE PIXELS COME FROM, whichever way they are obtained: crops already on disk, cut on demand from merged, or generated first. `crop_source` decides which of these apply and greys the rest. INSTRUCTION 230 A AND B. `crop_source` becomes `image_source`; `extract_channels` is gone and `train_channels` is what the model sees; `coordinate_columns` is DERIVED from `object_array` and so is not a control at all. The regex that once stood for the three path settings is retired too -- nothing ever read it.

### lines 6030-6036

```python
"crop_source", "file_metadata", "file_type", "coordinate_columns"],
```

GROUPED BUT NOT OFFERED. These four are what `image_source` and the `object_array` derivation replaced, and they stay in the settings dict because the RUNTIME still reads them -- `crop_source.py` and the streamer both do. So they need a category (an uncategorised key renders ungrouped at the top of the panel), and `_APP_HIDDEN_KEYS` is what keeps them off the form. The same split `png_type` would have if anything still read it.

### lines 6039-6040

```python
"Computer Vision Model": ["model_type", "model_name", "init_weights", ],
```

WHICH MODEL, and how its input is scaled. A custom model path that loads supersedes model_type, so no boolean is needed to say which to believe.

### line 6043

```python
"Computer Vision Training": ["train", "test", "epochs", "learning_rate", "optimizer_type", "sched...
```

HOW IT IS FITTED: the optimisation and the loss.

### line 6045

```python
"n_top_examples"],
```

CV-ONLY by `classify.FAMILY_SETTINGS` (instruction 233).

### lines 6048-6050

```python
"Computer Vision Optimization and Regularization": ["use_checkpoint", "dropout_rate", "weight_dec...
```

WHAT KEEPS IT FROM OVERFITTING. Its own heading because these are the knobs reached for when a model has learned the training set and nothing else, which is a different question from how fast it learns.

### lines 6053-6071

```python
"Model Evaluation": ["cross_validation_enabled", "cross_validation_folds",
```

HOW IT IS JUDGED. Shared by both families: an evaluation is an evaluation, so the headings below say nothing about which family is running. INSTRUCTION 233. One list of fifteen, split BY CATEGORY -- and only two of them split by family, because the audit said so.

THE AUDIT WAS THE WORK, and it corrected the guess. `classify.py`'s FAMILY_SETTINGS is the authoritative table of what each family reads exclusively, and against it only `n_top_examples` is CV-only (the one ML-only member of the fifteen has since been retired, unread). EVERY OTHER ONE OF THE FIFTEEN IS SHARED so filing them under "Computer Vision Evaluation", which is what the names suggest, would tell the user they apply to one path when they apply to both. That is the one hard rule this item states, and the first attempt at the split broke it on eleven settings out of fifteen.

So the headings below are NEUTRAL, and the two exclusives went to the family headings that already exist.

### lines 6079-6082

```python
"Leakage Audit": ["evaluation_fail_on_leakage", "leakage_audit_train_test",
```

THE LEAKAGE AUDIT IS ITS OWN QUESTION. Four settings about whether the train and test sets share objects is not "evaluation" in the sense the rest of the list means -- it is a check on the SPLIT, and a reader looking for it under a metrics heading would not find it.

### lines 6086-6092

```python
"Machine Learning Model and Features": ["model_type_ml", "n_estimators", "test_size", "cross_vali...
```

THE FEATURE-BASED CLASSIFIER: which model, and which features it may see. Feature preparation and feature importance were two headings asking one question -- which features the model uses -- so they are one. ML-ONLY (instruction 233). `score_column` names the column generate_ml_scores writes its prediction into -- it is not read on the computer-vision path, and it was in a list a CV user was reading top to bottom.

### lines 6097-6131

```python
"Regression: Response": [
```

REGRESSION, SPLIT IN SIX.

This was one heading holding thirty-eight settings, ordered by the accident of when each was added. Reading it, you could not tell that `alpha` does nothing unless regression_type is one of four penalised families, that the nine `guide_*` keys do nothing unless the permutation test is selected, or that `agg_type=None` silently changes the unit of analysis from the well to the cell. Three settings named a threshold and none of them thresholded the same thing.

The split follows the order the questions are actually asked:

1. What am I measuring?      -> Response 2. How should it be tested?  -> Model 3. ...with which knobs?      -> Model Tuning     (per-family) 4. ...or which permutation?  -> Permutation Test (per-mode) 5. What counts as a hit?     -> Significance 6. What gets thrown away?    -> Quality Filters

MOVED, NOT REMOVED. Every one of the thirty-eight is still here, still editable, still in the settings dict, under the same key -- this is a regrouping, not a redesign. Two keys are new (`inference`, `analysis_unit`) and both are readable front ends for decisions that were previously side effects of `analysis_mode` and `agg_type`; see _resolve_regression_analysis_choices. `score_column` LEFT on 2026-08-18 (instruction 135 A). It was the regression's duplicate of `dependent_variable` -- one measurement under two names -- and it is retired in get_perform_regression_default_settings. The key lives on under "Model Evaluation" for Explain CV, which uses it for the CNN score column.

The count table's two column names sit here because this heading is where a column is NAMED: `dependent_variable` names the score table's response, and these two name the guide and the read count in the count table. The Qt regression layout shows them under "Input Tables".

### lines 6139-6154

```python
"Regression: Model": [
```

inference and regression_type lead: they decide whether anything in "Model Tuning" or "Permutation Test" does anything at all. `model_plate_position` sits immediately before `random_row_column_effects` because the two are one decision read in order: the first says whether plate row and column are in the model at all, the second says fixed or random for a term that is. Reversed, the panel offers the refinement before the question. `regression_backend` sits immediately after `regression_type` because the two are one decision read in order (instruction 141 A): the first says WHAT is fitted, the second says WHO fits it, and the second's options are greyed by the first's value. `intercept` and `intercept_value` sit after the backend and before the plate-position pair because they are still part of WHAT is fitted rather than which terms are in it: they say where the fitted line is anchored, and every coefficient below is read relative to that anchor.

### lines 6160-6162

```python
"Regression: Model Tuning": [
```

Per-family knobs. spacr.ml.REGRESSION_SETTINGS_USED says which family reads which, and a family REFUSES the ones it cannot read rather than ignoring them, so a wrong setting here is an error and not a silent no-op.

### line 6169  _(unsure)_

```python
"Regression: Permutation Test": [
```

Read only when inference resolves to the permutation test.

### lines 6171-6173

```python
"grna_statistic",
```

FIRST, because it says WHAT is measured. Everything below it says how the null is built and who is eligible, which are answers to a question this setting asks.

### lines 6182-6195

```python
"threshold_multiplier", "annotation_source",
```

`volcano` is not here because it is RETIRED (see the note at

`settings.pop('volcano', None)`): the interactive volcano filters between genes and guides by right-click now, so a setting that chose which table it drew has nothing left to choose. Leaving the name in a category made the panel offer a control with no expected_types entry and no default. `Toxoplasma` LEFT THE PANEL. `annotation_source` says everything it said and more -- an organism name, a taxon id or an accession instead of one hard-coded parasite -- and two controls for one fact is two controls that can disagree. The key is still READ: every settings file in existence carries it, and it is what `annotation_source` defaults from when a file predates the field. It is no longer OFFERED, which is the difference between migrating a setting and breaking one.

### lines 6197-6199

```python
"p_threshold_alpha", "p_threshold_kind",
```

WHAT THE RUN MEANS BY SIGNIFICANT, which the plot could previously contradict from its right-click menu (instruction 135), and the two knobs of the RRA hit caller.

### lines 6203-6205

```python
"Regression: Quality Filters": [
```

Everything that decides which rows reach the model. These were spread across the old list with the fitting knobs between them, so it was not obvious that four separate settings each drop data.

### lines 6209-6212

```python
"normalise_fraction",
```

DIRECTLY UNDER THE THRESHOLD IT DIVIDES BY. It is only meaningful in terms of what that threshold removed, so a reader who meets it anywhere else has to go and find the other one first.

### lines 6216-6219

```python
"Regression: Diagnostics": ["regression_qc"],
```

Not "was this gRNA significant" but "does this fit deserve to be believed" -- which is a different question and belongs under a heading of its own rather than beside the thresholds that decide hits. One key today; the QC suite is where further diagnostic toggles will land.

### lines 6227-6230

```python
"Replication Assay": [
```

Replication-specific vacuole assignment and scoring. The shared parasite area filters and empty-well seeding control remain listed once under "Invasion Assay"; the Qt app-specific category map presents those shared keys under Replication's Object Filtering/Scoring sections.

### lines 6254-6257

```python
"Advanced": ["resume", "strict_errors", "max_failure_rate", "queue_by_uncertainty", "queue_measur...
```

Rarely-touched knobs only. 'normalize' left for "General" and nuclei_limit / pathogen_limit for "Measurements": all three change what the run produces rather than how it is tuned, and hiding them here is what put them at the bottom of the Classify (CV) dataset settings.

### lines 6260-6262

```python
"3D Settings (Beta)": [
```

Experimental volumetric controls are deliberately split by dimensional contract. `z_axis` lives with 3D because 4D builds on the same z plan; the 4D panel contains only time-axis and inter-frame tracking controls.

### lines 6278-6296

```python
ADVANCED_UMBRELLA = 'Advanced settings'
```

ORGANELLE, SPLIT. Instruction 72 item 5.

One "Organelle" category held FIFTY-THREE settings -- the most over-configured object class in the tool. A biologist who knows they are imaging lysosomes should not have to scroll past organelle_ridge_sigmas and organelle_hysteresis_high to find the channel.

"Organelle" now keeps only what a biologist recognises without knowing how segmentation works; everything else moves to "Organelle advanced".

MOVED, NOT HIDDEN, and the distinction matters: every advanced setting is still in the panel, still editable, and still in the settings dict. A setting removed from the UI while staying in the dict is how a run gets a value nobody can see, which is exactly how this project acquired eleven phantom settings (instruction 61).

Derived from `organelle_types.BASIC_SETTINGS` rather than hand-listed, so the split cannot drift from the module that defines what "basic" means.

### lines 6298-6322

```python
ADVANCED_UMBRELLA = 'Advanced settings'
```

ADVANCED SETTINGS, GROUPED BY WHAT THEY DO. Instruction 73.

The panel groups by OBJECT: everything about cells together, everything about nuclei together. The request adds a second axis, and the reason it is worth doing is exact: `cell_min_size` and `nucleus_min_size` do the SAME THING to different objects. Filed under two headings they read as two unrelated knobs; filed under one heading they read as one decision applied four times, which is what they are.

THE STRUCTURE IS THREE LEVELS: an "Advanced settings" umbrella, one heading per family under it, and one sub-heading per object under that. `SettingsWidgets.build_sections` carries the tree; the flat `(title, rows)` pairs it has always returned are still the outer half of it, so a panel that cannot draw a tree still draws every control exactly once. `CATEGORY_PARENTS` below is what says which family sits under the umbrella; the per-object level is derived from the key prefix, so a family never has to repeat the object list.

Within each heading the keys are ORDERED BY OBJECT, so the four `*_min_size` settings sit together and read as the group they are whether or not the panel drawing them can nest.

ORGANELLE FOLDS IN rather than keeping the separate scheme instruction 72 shipped, per that instruction's item 6: one structure, not two.

### lines 6351-6359

```python
"background", "signal_to_noise",
```

`signal_to_noise`, lower-case, since b7ae412af renamed <object>_Signal_to_noise to <object>_signal_to_noise. The suffix here kept the old capitalisation and so matched nothing, which took the signal-to-noise anchor out of this family for cell, nucleus and pathogen -- the panel showed each of them a background floor and a remove-background switch and nothing else, while the docstring of the test that covers it still said three. Nothing errored: a family that matches no key just has one fewer row.

### lines 6361-6365

```python
"rolling_ball", "rolling_ball_radius", "clahe", "clahe_clip_limit",
```

What organelle can already do to its channel before anything is segmented, and cell / nucleus / pathogen cannot. Grouping does not hide the gap -- each object's sub-heading shows exactly the keys that object has -- it makes it visible, which is the first step to closing it.

### lines 6496-6498

```python
_organelle_basic_slots = [key for key in organelle_basic_settings
```

Filter ONCE rather than once per role. The predicate does not depend on `_role`, so re-testing it inside the loop re-walked both lists 364 times for an answer that could not change. Same keys, same order.

### lines 6510-6511  _(unsure)_

```python
categories['General'].append(_key)
```

Every generated slot key is declared and the base General list owns none of them; the settings contract tests pin both premises.

### lines 6524-6525  _(unsure)_

```python
category_group_dependencies = {}
```

Compatibility hook for callers that still inspect the settings dependency tables. No live setting currently gates a category by group membership.

### lines 6532-6534

```python
tuple(key for role in ORGANELLE_SLOT_ROLES
```

Both organelle categories are gated on the channel, not just the first: splitting the category would otherwise leave "Organelle advanced" showing on a run that does no organelle segmentation at all.

### lines 6540-6545

```python
setting_dependencies = {}
```

Per-setting applicability is deliberately data rather than Qt code.  Each entry is populated lazily by :func:`get_setting_dependencies`, because importing ``spacr.ml`` while it is importing this module would be circular. Predicates receive ``(current_settings, lightweight_data_context)`` and the reason callable receives the same pair.  The Tk front end can consume this table too; the first consumer is the regression SettingsBuilder.

### lines 7144-7151

```python
category_value_dependencies = {
```

Categories shown only when a setting equals a specific value.

gui_core._get_visible_categories blocks the categories of every option that does NOT match the current value, so a category listed under two or more options can never be shown. The eight per-method organelle headings are now a single "Organelle" category (ordered by method instead of gated on it), so organelle_method no longer gates anything and its map is empty. The mechanism itself is still wired up in both GUIs for the next setting that needs it.

### line 7651  _(unsure)_

```python
set_interperate_vision_model_defaults = set_interpret_vision_model_defaults
```

Backward compatibility for the misspelling published in earlier releases.

### lines 8009-8024

```python
COUNT_DEPENDENT_MEASUREMENTS = ('spatial_measurements', 'object_distances')
```

WHICH MEASUREMENTS MEAN SOMETHING FOR WHICH ORGANELLE

"How many, and how spread out" is the phenotype for a punctate or vesicular organelle and is not a question at all for a reticular one. An ER meshwork is ONE connected object filling the cell: its neighbour count is zero, its nearest-neighbour distance is undefined, and a screen that regressed on either would be regressing on whether the segmentation happened to break the network in two that day.

THE MEASUREMENTS ARE STILL COMPUTED. Nothing here switches a family off a user who wants the number gets the number, and a value that vanished without being asked to is worse than one that comes with a caveat. What this does is SAY SO, in the same voice the type preset already uses to say what it set and what is weak about it.

### lines 8281-8298

```python
from . import illumination as _illumination  # noqa: E402,F401
```

THE ILLUMINATION HELP ARRIVES WITH THE MODULE THAT OWNS IT.

`expected_types` types the nine illumination_* keys and `categories` files them under "Illumination Correction", both in the literals above -- so every session's Measure panel offers those nine controls. Their tooltips, though, are contributed by spacr.illumination through `register_defaults` at ITS import, so a session that had no other reason to import that module drew nine controls with no help beside them, and `test_every_typed_setting_has_a_tooltip` passed or failed on which files pytest was pointed at.

Importing the module here rather than copying its help text in keeps one owner for that prose: `_merge_declarations` refuses a second, different definition of a tooltip, so a copy would have to be kept in step by hand and would announce itself only as an ImportError.

LAST IN THE FILE, after every name this module defines, because registering calls back into it.

### lines 8300-8303

```python
from . import ops_settings as _ops_settings  # noqa: E402,F401
```

OPS registers here for the same reason illumination does: the tables above must exist first.

## _merge_declarations

### lines 736-742

```python
if new_category and name not in category_keys:
```

KEEP THE SNAPSHOT IN STEP. `category_keys` is built once, at import, as `list(categories.keys())` -- and a module registering through this seam adds a category AFTERWARDS. Power/Design does exactly that, so importing its screen left `category_keys` one entry short of `categories`, and `check_settings` (settings.py, "key not in category_keys") then treated that heading's own keys as unknown. Order-dependent, so it only appeared when a Qt import ran first.

### lines 746-750

```python
REGISTERED_CATEGORIES.add(name)
```

Recorded so "declared in the literal below" and "added at import by a module registering through this seam" are DISTINGUISHABLE. Without it, anything comparing the source literal against the live dict is order-dependent: it passes alone and fails after any test that imported Power/Design.

## _takes_an_argument

### line 837, trailing  _(unsure)_

```python
return True
```

not introspectable — assume the common shape

## set_default_settings_preprocess_generate_masks

### lines 886-894

```python
_fold_renamed_settings(settings)
```

THE OLD NAMES MUST MOVE BEFORE ANY DEFAULT IS FILLED IN, and this factory had no fold at all -- which is the largest half of the defect 364 recorded. Every `<role>_FT`, `<role>_CP_prob`, `<role>_Signal_to_noise` and `<role>_min_object_area` is declared here, and `_set_organelle_defaults` owns `organelle_min_area`/`_max_area`; so a settings CSV written before b7ae412af (2026-09-02) reached this function, matched nothing, and had every one of those values replaced by a default with nothing said. Measured before the fix: `cell_FT=0.42` came out as `cell_flow_threshold=100`.

### lines 896-907

```python
settings.setdefault('pipeline_style', 'v1')
```

── pipeline flavour ────────────────────────────────────────────── 'v1' — the original multi-copy chain (rename → channel folders → npy → npz → mask npy → merged/). Stable, well-tested. 'v2' — streaming pipeline (spacr.pipeline_v2). Reads originals directly, writes one npy per field to merged/ with masks appended in-place. ~60-80% less disk. Opt-in for one release, then default. Default to the v1 disk-based pipeline: it is the fully-tested path and produces the channel/stack/mask_stack folder layout the rest of spaCR (measure, annotate, downstream tools, the e2e suite) depends on. The v2 streaming pipeline (no .npz on disk) is opt-in via pipeline_style='v2' until it reproduces that layout and fixes real-data channel indexing.

### lines 909-910

```python
settings.setdefault('batch_fields', 8)
```

v2-only: how many field stacks to load into memory per Cellpose batch. Bigger = faster, more RAM.

### lines 912-913

```python
settings.setdefault('keep_npz', False)
```

v2-only: keep the in-memory NPZ batch on disk under merged/_scratch/ for debugging. Default False → NPZ never touches disk.

### lines 924-926

```python
settings.setdefault('dry_run', False)
```

Validate-only: preprocess_generate_masks runs the pre-flight checks in spacr.validate, prints the report plus the plan, and returns before any model loads or any file is written.

### lines 938-951

```python
settings.setdefault('remove_background_organelle', False)
```

DECLARED 2026-09-12 (364). `spacr/io.py` has read

`remove_background_organelle`, `organelle_background` and `organelle_signal_to_noise` since the per-channel loop was written, each through a `.get` with a fallback -- so nothing raised, nothing logged, and organelle was the ONLY object channel whose background could not be removed, because no declared setting turned it on.

THE VALUES ARE THE FALLBACKS, NOT A NEW OPINION. io.py resolved these to `settings.get('background', 100)`, `settings.get('Signal_to_noise', 10)` and `settings.get('remove_background', False)`, so declaring 100 / 10 / False changes no run that exists. Copying `remove_background_pathogen`'s True would silently start clipping every organelle channel in every settings file already written, which is a behaviour change and belongs in release notes with the maintainer's say-so, not here.

### lines 959-962

```python
settings.setdefault('cell_model_name', 'cpsam')
```

Cellpose 4 ships one stock model, so the only real choice these keys carry is "stock weights" vs "the checkpoint I trained". A legacy value in an old settings file is mapped forward by normalize_cellpose_model_name when _get_object_settings reads it.

### lines 967-971

```python
settings.setdefault('seg_qc', 'report')
```

Segmentation QC — scored on the masks the moment they exist, so a plate that segmented badly is caught here rather than after measure_crop has spent hours on it. 'report' computes, saves and prints; it never filters. The thresholds are spacr.seg_qc.QC_DEFAULTS, documented there and in the tooltips below.

### line 1011  _(unsure)_

```python
settings.setdefault('pathogen_model', None)
```

Analasys settings

### line 1033  _(unsure)_

```python
settings.setdefault('save_original_images', True)
```

Misc settings

### lines 1039-1042

```python
settings.setdefault('z_stack', False)
```

3D (Beta). Off by default and read only through spacr.zstack.plan_from_settings, which returns None whenever `z_stack` is falsy -- so with these defaults not one line of z code executes and the 2-D path is bit-identical to a run from before these keys existed.

### lines 1052-1060

```python
settings.setdefault('t_stack', False)
```

4D (Beta). The time axis on top of the z axis, read only through spacr.zstack.plan_4d_from_settings, which returns None whenever `t_stack` is falsy -- so with these defaults not one line of 4-D code executes and both the 2-D and the 3-D path stay bit-identical to a run from before these keys existed. `t_axis_order` deliberately has no usable default: (T,Z,Y,X) and (Z,T,Y,X) are both written by real microscopes and a 4-D shape cannot tell them apart, so a run that turns t_stack on without saying which it has is stopped rather than guessed at -- guessing wrong links objects across z and calls it a trajectory.

### lines 1074-1082

```python
_set_organelle_defaults(settings)
```

ORGANELLE DEFAULTS, from the one function that owns them.

Forty-odd `settings.setdefault('organelle_*', ...)` lines stood here, a second hand-written copy of `_set_organelle_defaults`. They agreed exactly and NOTHING ENFORCED THAT: a run takes whichever factory it went through and never compares the two, so a value corrected in one copy and not the other measures the same plate two ways with no error and nothing in the log. `tests/test_organelle_defaults_agree.py` was the holding pattern; deleting the duplication is the fix.

### line 1086  _(unsure)_

```python
settings.setdefault('cell_perimeter_fraction', 0)
```

merge_split

### lines 1095-1099

```python
settings.setdefault('cell_intensity_threshold', None)
```

NO DEFAULT, AND THAT IS THE ANSWER RATHER THAN AN OMISSION (391). The threshold is in RAW IMAGE UNITS, so any number here would be right for the one acquisition it was chosen on and wrong for every other exposure and gain. `None` makes the merge refuse and say what the boundaries in the field actually were, which is a number the user can then type.

### lines 1123-1127

```python
settings.setdefault('cell_max_area', 0)
```

The organelle pair is NOT set here. `_set_organelle_defaults` owns it, per organelle type, and a hand-written 0 beside it is what let the live preview and the batch run disagree in the first place: the preview read 0 and filtered nothing while the batch read the retired `organelle_min_size` of 10 and filtered.

### lines 1136-1143

```python
settings.setdefault('motility_analysis', False)
```

NOTE: `timelapse`, the `timelapse_*` knobs above and `motility_analysis` are deliberately still defaulted here even though the Mask *module* no longer surfaces them in its GUI (they moved to the standalone Timelapse and Motility Assay modules — see get_timelapse_settings and get_automated_motility_assay_default_settings). spacr.object reads settings['timelapse'] on every mask run and settings['motility_analysis'] inside the timelapse branch, and old settings CSVs still carry both, so removing the defaults would break the pipeline and every archived CSV.

### lines 1148-1151

```python
settings.setdefault('strict_errors', None)
```

Fail-loud policy. None means "not set here" and defers to the SPACR_STRICT_ERRORS environment variable, which is how a cluster turns it on for a whole batch without editing every settings file. True/False here is an explicit per-run choice and wins over the environment.

### lines 1154-1156

```python
settings.setdefault('resume', False)
```

Continue an interrupted run instead of starting over. Opt-in: spacr.resume validates what is already on disk rather than trusting it, and clears a field's existing rows before re-measuring it.

### lines 1158-1161

```python
from .illumination import illumination_settings
```

Mask estimates the same optical field as Measure but applies it only to private Cellpose inputs; the persisted stack remains raw. Call the illumination module's factory so both screens expose one vocabulary and new controls cannot land in only one of them.

### 2026-09-19, the flow thresholds (428, GitHub #123)

```python
settings.setdefault('cell_flow_threshold', 0.4)
```

`nucleus_`, `cell_` and `pathogen_flow_threshold` default to 0.4, and so does `FT` in `get_default_test_cellpose_model_settings` and `get_default_apply_cellpose_model_settings`. The maintainer chose it on 2026-09-19: it is Cellpose's own `CellposeModel.eval` default and the strictest of the options he was offered. It drops the most irregular objects (parasites included) and changes default segmentation results the most. History of this default: 100 (2024-07-18), 1.0 (4b9fef8b9, 2025-07-02), 100 again (df753075e, 2026-09-02; shipped in 1.5.0.5 to 1.5.0.8), 0.4 now. At 100 the flow check still ran but rejected practically nothing, and `spacr.validate` warned about that default on every run (GitHub #123).

`setdefault` never overwrites a value the settings already hold. So a settings file that carries 100 keeps 100, in either spelling (`cell_flow_threshold` or the pre-2026-09-02 `cell_FT`, which `_fold_renamed_settings` moves first), and gets the corrected warning from `spacr.validate._flow_threshold_problems`. Nobody's saved choice is rewritten. The shipped example pack `spaCR_settings/1_generate_masks_settings.csv` is one such file, and it keeps its 100.

`FT` was declared `int` in `expected_types`. With a 0.4 default that is a type error in `_check_types`, and `coerce_expected_types` would leave a CSV's `'0.4'` as text. It is `(int, float)` now, the same as the per-object keys.

## set_default_plot_data_from_db

### lines 1207-1219

```python
settings.setdefault('graph_type', 'jitter_box')
```

A BOX WITH JITTER, NOT A BAR WITH JITTER. Instruction 139 B, asked for on 2026-08-18: "the bargraphs with jutter plot backgrounds should be boxplots with jutter".

It is a statistical correction rather than a preference, which is why the DEFAULT moves rather than the option merely existing. A bar drawn at a mean with points behind it shows ONE number and hides the shape: two groups with the same mean and completely different spreads draw the same bar. A box shows the median, the quartiles and the whiskers, so the reader sees the distribution the points already imply -- and the jitter stays, because the box summarises and the points are the evidence.

`jitter_box` already existed as an option; only the default was wrong.

## _read_cellpose_models

### lines 1295-1296

```python
LOG.debug("Could not read the Cellpose user-model registry",
```

A malformed ~/.cellpose/models/gui_models.txt must not cost the user the stock models as well.

## cellpose_model_choices

### lines 1340-1341

```python
_CELLPOSE_MODELS_CACHE = _cpsam_first(names)
```

Cache only a real answer. A miss because Cellpose is not loaded yet must not pin the fallback for the process.

## downloaded_zoo_models

### lines 1426-1435

```python
out = []
```

include_bundled=False, AND THAT IS NOT AN OPTIMISATION. It is the flag that runs `discover_local`, which SCANDIRS resources/models and this function is called while a settings panel is being built, on the GUI thread. `tests/qt/test_preview_registry.py:: test_the_registry_never_touches_the_filesystem` exists to catch exactly that and did: the first version of this walked the disk every time a panel opened.

The remote rows carry the paths anyway, so nothing is lost: what is skipped is the walk looking for models nobody declared.

### lines 1437-1444

```python
for entry in model_zoo.catalogue(remote=True, include_bundled=False,
```

block=False, AND THAT IS THE SAME KIND OF FLAG AS THE ONE ABOVE. `remote=True` reaches `model_zoo.shared_catalogue`, which fetched the community catalogue over HTTPS -- here, on the GUI thread, while a settings panel was being built. Measured 2026-09-05 with the host non-routable: opening Mask took 32.2 s, and GNOME's "force quit" dialog is what the user saw. The community rows are taken from the cache instead; the entries this function keeps are the ones already on disk anyway.

### line 1450  _(unsure)_

```python
if path and os.path.isfile(path):
```

One stat per declared entry, not a directory walk.

## _get_object_settings

### lines 1509-1511

```python
object_settings['model_name'] = normalize_cellpose_model_name(
```

'cpsam' unless the user pointed at their own checkpoint. A legacy name from an old settings file is mapped forward here rather than carried into segmentation as if it still selected different weights.

### line 1523

```python
object_settings['diameter'] = float(settings['cell_diameter'])
```

Coerce — CSV-imported settings arrive as strings ("30.0").

### lines 1543-1546

```python
elif object_type == 'pathogen':
```

(A commented-out `use_sam_nucleus -> model_name = 'sam'` sat here. There is no model named 'sam': Cellpose 4 IS SAM and calls its one model 'cpsam', which is already what nucleus_model_name defaults to. Removed rather than left as a suggestion that would not work.)

### line 1564  _(unsure)_

```python
else:
```

(Same for the commented-out `use_sam_pathogen` branch — see above.)

## set_default_umap_image_settings

### lines 1583-1586

```python
_fold_renamed_settings(settings)
```

BEFORE THE DEFAULTS: this factory declares `reduction_method`, so it is where a file carrying the misspelt `redunction_method` has to be folded. `RETIRED_SETTINGS` has recorded that rename all along and the run never performed it.

### lines 1601-1603

```python
settings.setdefault('tsne_perplexity', 30.0)
```

Reducer-specific controls stay in the settings dict even when another reducer is selected.  Qt greys the inactive family rather than removing it, so switching methods preserves the values the user chose.

### lines 1615-1617

```python
settings.setdefault('gpu', False)
```

This is controlled by the GPU label in the action strip, not duplicated as a form row.  It is nevertheless a real run setting so notebooks, the CLI and the run journal all record the selected backend request.

### lines 1624-1633

```python
settings.setdefault('plot_cluster_grids', False)
```

OFF by default (2026-08-12, instruction 75). The cluster grid is a SECOND, montage figure emitted AFTER the embedding, so with it on the last thing an Image UMAP run put on screen -- and therefore the thing a user who steps away is left looking at -- was a sheet of cluster panels rather than the graph: "i don't want to see a grid with plots, i want the normal figure view i have in other modules ... no grid at the end, just normal behaviour". The since-removed Tk GUI always passed False here, so this made the default agree with the one surface that already had it right. The figure is one checkbox away for anyone who wants it.

### lines 1669-1671

```python
settings.setdefault('crop_source', 'auto')
```

'auto' uses the PNG crop folder when one exists and falls back to cutting crops out of merged/*.npy on demand; 'png' and 'merged' force one source. See spacr.crops.resolve_crop_source.

## get_measure_crop_settings

### lines 1685-1689

```python
_fold_renamed_settings(settings)
```

BEFORE THE ORGANELLE COUNT IS INFERRED, not merely before the defaults. A measure-crop CSV carries the organelle `_size` pair exactly as a mask CSV does, and `organelle_count` reads `organelle_*` keys to decide how many slots the file asked for -- so folding after it would count the old spellings as slots.

### lines 1691-1693

```python
_requested_organelle_count = organelle_count(settings)
```

Infer a pre-count settings file before defaults add placeholder ``organelle_*`` keys. Once those placeholders exist they cannot be distinguished from a legacy file that genuinely requested slot one.

### lines 1695-1699

```python
import ast as _ast
```

Coerce bracketed strings (e.g. channels "[0,1,2,3]" imported from a CSV) back into Python lists/tuples. The Qt drag-and-drop settings import reads CSV cells as raw strings and does not run them through check_settings(), so without this measure_crop rejects channels / crop_mode / png_size / ... as "not a list". Idempotent: values that are already lists, or ordinary strings, are left untouched.

### lines 1714-1716

```python
settings.setdefault('dry_run', False)
```

Validate-only: measure_crop runs the pre-flight checks in spacr.validate, prints the report plus the plan, and returns before the worker pool starts or anything is written to measurements.db.

### lines 1724-1728

```python
settings.setdefault('spatial_measurements', True)
```

ON BY DEFAULT since 2026-09-01, by the maintainer's decision: these are the measurements a host-pathogen screen is run FOR, and a default run that omitted them was discovered after the run, when measuring again costs twenty minutes a plate. The KD-tree and boundary pass they cost are paid for by not repeating the run.

### lines 1730-1733

```python
settings.setdefault('spatial_neighbor_radius', 50)
```

The radius is part of the COLUMN NAME (neighbors_within_50), so it is declared rather than read raw from the dict: an undeclared key has no widget and is refused by check_settings, which would leave the neighbourhood size settable only by editing the source.

### lines 1735-1745

```python
settings.setdefault('bystander_measurements', False)
```

EVERY DISTANCE WORTH MEASURING. On by default for the same reason as spatial_measurements above; it is real time on a 3-D field and that is the cheaper half of the trade. WHICH UNINFECTED CELLS ARE NEXT TO AN INFECTED ONE (instruction 388). OFF BY DEFAULT, unlike the two families above, and the difference is deliberate: those were turned on by the maintainer AFTER plates had been measured with them, and this one adds a column family nobody has seen on a real plate yet. The cost is one distance transform and one KD-tree over the cell mask, which is cheaper than either of them so the reason to leave it off is unfamiliarity, not time, and it is a one-line decision to flip once a plate has been measured with it.

### lines 1747-1751

```python
settings.setdefault('bystander_reach_in_diameters', 1.0)
```

THE REACH IS IN CELL DIAMETERS, NOT MICRONS, so it means the same thing at 20x and 63x and on a plate whose cells are simply larger. The diameter is measured from the cell mask of the field being measured; a hard-coded distance would be right for one dataset and silently wrong for the next.

### lines 1760-1765

```python
settings.setdefault('voxel_size_z_um', None)
```

Voxel geometry. Measure needs these for the same reason segmentation does: on a 3-D mask every regionprops and distance-transform call takes a spacing, and without one the z axis is treated as if a plane step were one xy pixel. Left at None a 2-D run is unaffected (spacing is not applied in 2-D at all, or *_area would silently change units) and a 3-D run stops rather than guessing.

### line 1770  _(unsure)_

```python
settings.setdefault('save_arrays', False)
```

Cropping settings

### lines 1775-1779

```python
settings.setdefault('png_channel_mapping', {'r': 2, 'g': 1, 'b': 0})
```

`png_dims` is left in place, unset, on purpose: an older settings CSV supplies it and spacr.crops.resolve_png_channel_mapping translates it. Defaulting it here as well would mean the default mapping and a default png_dims both existed, and the precedence between them would decide the colours of every crop without anyone having chosen it.

### line 1791  _(unsure)_

```python
settings.setdefault('plot',False)
```

Operational settings

### line 1795  _(unsure)_

```python
settings.setdefault('cell_mask_dim',4)
```

Object settings

### line 1803

```python
settings.setdefault('cell_max_size',None)
```

UPPER BOUNDS, off by default so no existing run changes.

### lines 1809-1820

```python
settings.setdefault('organelle_min_area', 0)
```

WHAT KIND OF ORGANELLE, on the measure side as well as the mask side. Measure has to know it to say which of its own numbers mean what they usually mean: "how many, and how spread out" is the phenotype for a punctate organelle and is a segmentation artefact for a reticular one. Defaults to 'custom', which makes no claim, so a settings file written before this existed still means exactly what it meant. THE FIRST ORGANELLE IS SEEDED BY HAND, so it has to be given the size floor the loop below gives every other slot. Without it `measure_crop` asked for a key the factory never shipped and the organelle filter silently never ran, while the form had no row to show. 0 rather than Mask's 10: Measure consumes labels that are already segmented, so it filters nothing unless asked.

### lines 1823-1826

```python
settings.setdefault(NUMBER_OF_ORGANELLES, _requested_organelle_count)
```

HOW MANY ORGANELLES THIS RUN HAS. Measure reads the same count the mask side does, because it is the same objects being measured: a run that segmented five organelles has five sets of masks to measure and a panel that offered two would leave three of them unreachable.

### lines 1832-1853

```python
settings.setdefault('cytoplasm_min_size',0)
```

NO FIXED FLOOR OF FOUR. Removed 2026-09-02 on the maintainer's instruction: "if the user chooses 2 organelles settings for 2 organells if the user chooses 100 organelles settings for 100 organells" (instruction 326).

This loop used to seed every role in `schema.ORGANELLE_ROLES` -- four of them -- with a disabled placeholder "even when number_of_organelles is zero or one", so that downstream readers could iterate a fixed schema. The cost was that a two-organelle run carried FOUR slots' keys, which is what the maintainer objected to, and it put those keys into every settings CSV, run journal and reproducibility hash.

IT WAS ALSO THE THING THAT MADE RAISING THE CEILING IMPOSSIBLE. Widening the role vocabulary so a hundred slots could be KEYED widened this loop with it, so a five-organelle run came back carrying twenty-six. The attempt on 2026-09-02 was reverted for exactly that, and this is the coupling that caused it.

The loop above already seeds `declared_organelle_roles(settings)`, which is the count the user asked for plus any slot the file already carries. A reader that needs to know which organelle tables a run has should ask that, not a constant.

### lines 1865-1868

```python
settings.setdefault('strict_errors', None)
```

Fail-loud policy. None means "not set here" and defers to the SPACR_STRICT_ERRORS environment variable, which is how a cluster turns it on for a whole batch without editing every settings file. True/False here is an explicit per-run choice and wins over the environment.

### lines 1871-1873

```python
settings.setdefault('resume', False)
```

Continue an interrupted run instead of starting over. Opt-in: spacr.resume validates what is already on disk rather than trusting it, and clears a field's existing rows before re-measuring it.

### lines 1875-1887

```python
settings.setdefault('summarize_organelles_by', 'cell')
```

Which parent compartments organelle measurements roll up into. Its absence here silenced an entire output: measure.py gates all four organelle writes on `is not None`, so a measure run wrote no organelle table at all -- even though every merged stack carries real organelle labels in `organelle_mask_dim`, and the demo ships 64 of them per field. The key was only ever defaulted in `set_default_settings_preprocess_generate_masks` (the MASK pipeline), which never reaches the measure settings, so nothing downstream could tell "the user asked for no organelle summary" from "nobody asked".

'cell' is the documented default and is what the mask pipeline already sets. Raw per-object tables are controlled solely by each slot's ``*_mask_dim``; this setting controls only the optional parent rollups.

### lines 1890-1894

```python
if settings.get('verbose'):
```

SAY WHAT THE NUMBERS WILL AND WILL NOT MEAN. Nothing is switched off: a family the organelle type makes doubtful is still measured, because a value that vanished without being asked to is worse than one that comes with a caveat. It is said out loud in the same voice the type preset uses to say what it set and what is weak about it.

### lines 1898-1918

```python
from .illumination import illumination_settings
```

Illumination / flat-field correction. It divides the microscope's uneven lighting out of every field BEFORE any intensity feature is computed, so it is a property of the measure run it changes, not of a separate one: measure_crop calls `spacr.illumination.prepare_illumination_correction(settings)` and that call reads these keys. Without them here the switch was unreachable from Measure -- the only way to throw it was another screen, which leaves process environment variables a later run silently inherits, so whether a table's intensities carried a position-dependent bias depended on what had been run in the same process beforehand.

Filled by CALLING the illumination module's own factory rather than by copying its keys, so a knob added there appears on the Measure panel without a second edit. Imported inside the function because spacr.illumination registers itself with this module at import, and a top-level import would close that loop.

`setdefault` per key, not `update`: the factory also fills `src` and `channels`, and Measure's own values for those win. The estimate reads the fields the run measures, so Illumination's three-channel default would drop the fourth channel of a four-channel measure run.

## set_default_classify

### lines 1943-1953

```python
for retired in ("location_column", "positive_control_id", "negative_control_id"):
```

`location_column`, `positive_control_id` and `negative_control_id` come from the ML factory, and the Classes dict now says the same thing better: a control well IS a class defined by a metadata column, which is exactly a row of that dict. Three settings saying it a second way were three ways for the two to disagree, and nothing said which one won.

Dropped from the MERGED module only. Classify (ML) on its own still offers them, and an old settings CSV that sets them still trains on the same wells -- spacr.classify_classes.normalize_settings turns them into class rules before anything reads them.

## set_default_analyze_screen

### lines 1965-1968

```python
_fold_renamed_settings(settings)
```

BEFORE ANY DEFAULT IS FILLED IN (364). A settings file naming a renamed key must reach the new name carrying its VALUE, and a `setdefault` that ran first would already have put the default there -- so the user's number would be silently replaced by ours.

### lines 1971-1984

```python
from .training_basis import resolve_basis
```

The shared training basis. `resolve_basis` is what keeps an older settings CSV -- which selected the basis IMPLICITLY, by whether annotation_column was set -- behaving exactly as it did.

It has to be asked BEFORE annotation_column is defaulted, and the answer has to become the default rather than 'metadata'. A plain `setdefault('dataset_mode', 'metadata')` here made that promise unkeepable: it runs before `resolve_basis` is ever consulted, and once dataset_mode is set explicitly the implicit rule cannot fire for any real run. A project whose settings named an annotation_column and no dataset_mode therefore trained on plate metadata controls instead of the user's manual annotations -- silently, reporting success, with the wrong labels. That is precisely what resolve_basis's own docstring says must not happen.

### lines 2014-2016

```python
settings.setdefault('batch_control_values', None)
```

Keep this blank so control_center follows the module's current negative_control_id value instead of silently retaining a stale 'c1' when the user changes the plate layout.

## _resolve_rename

### lines 2183-2184  _(unsure)_

```python
step += (name,)
```

Already terminal: carry it, so a split whose halves have different chain lengths does not lose the shorter one.

### lines 2193-2195

```python
return (), hops
```

A CYCLE LEAVES THE KEY ALONE rather than guessing. The key keeps its own name and the unknown-key check reports it, which is much better than migrating it somewhere arbitrary.

## surviving_setting_name

### lines 2232-2233

```python
return ()
```

Not a move. Its own fold performs it; saying otherwise here would put a boolean where an int() is waiting.

## _fold_renamed_settings

### lines 2279-2281

```python
moves = []
```

Resolve first, mutate second: `surviving_setting_name` consults `expected_types`, and editing while resolving would let one migration change another's answer.

### lines 2290-2291

```python
for _hops, old, targets in sorted(moves, key=lambda row: (row[0], row[1])):
```

NEAREST FIRST, then alphabetically so the answer cannot depend on the order the CSV happened to list its columns in.

## set_default_train_test_model

### line 2353, trailing  _(unsure)_

```python
settings.setdefault('schedule','cosine')
```

reduce_lr_on_plateau, step_lr

### line 2354, trailing  _(unsure)_

```python
settings.setdefault('loss_type','focal_loss')
```

binary_cross_entropy_with_logits

### lines 2385-2388

```python
settings.setdefault('strict_errors', None)
```

Fail-loud policy. None means "not set here" and defers to the SPACR_STRICT_ERRORS environment variable, which is how a cluster turns it on for a whole batch without editing every settings file. True/False here is an explicit per-run choice and wins over the environment.

### lines 2391-2393

```python
settings.setdefault('crop_source', 'auto')
```

'auto' uses the PNG crop folder when one exists and falls back to cutting crops out of merged/*.npy on demand; 'png' and 'merged' force one source. See spacr.crops.resolve_crop_source.

## set_generate_training_dataset_defaults

### lines 2407-2413

```python
settings.setdefault('metadata_item_1_name',None) # e.g. ['nc','pc']
```

class_metadata holds VALUES OF the class column ('columnID'), so the entries have to be well ids. It was set twice, and the first call won: ['nc','pc'] -- the CLASS NAMES from deep_spacr_defaults' 'classes' key, pasted onto the wrong setting. No columnID is ever so the shipped default selected zero crops in both classes and the second, correct assignment below was dead. Same for 'tables', set to the four object tables and then to None.

### line 2414, trailing

```python
settings.setdefault('metadata_item_1_name',None)
```

e.g. ['nc','pc']

### line 2415, trailing

```python
settings.setdefault('metadata_item_1_value',None)
```

e.g. [['c19','c2'],['c3','c4']]

### line 2416, trailing

```python
settings.setdefault('metadata_item_2_name',None)
```

e.g. ['sample1','sample2']

### line 2417, trailing

```python
settings.setdefault('metadata_item_2_value',None)
```

e.g. [['r1','r2'],['r3','r4']]

### lines 2422-2427

```python
_fold_the_classes(settings)
```

AND THEN `classes` OVERRULES BOTH (instruction 229). The Classes editor already names the column each class is defined by and the value that defines it, so `annotation_column` and `class_metadata` are two more places for the same two facts -- and nothing downstream reads both and compares them. Derived here, AFTER the defaults, so a settings file that defines no class keeps exactly what it had.

## deep_spacr_defaults

### lines 2445-2447

```python
settings.pop('custom_model', None)
```

The retired Classify boolean shared a name with Cellpose's checkpoint path. It is not a compatibility input: classifier selection is entirely determined by custom_model_path and model_type.

### lines 2450-2454

```python
if settings.get('extract_channels') and not settings.get(
```

BEFORE ANY DEFAULT LANDS (instruction 230 A). `extract_channels` is removed and `train_channels` takes its place -- the channels that matter are the ones the model sees. A settings file that set the old key and not the new one MEANT those channels, so the value is moved rather than being silently outvoted by the default filled in below.

### lines 2470-2472

```python
settings.setdefault('tables', ['cell', 'nucleus', 'pathogen',
```

THE FOUR OBJECTS spaCR MEASURES, which is what the default should always have been (instruction 230 A). `None` meant "work it out", and what it worked out was frequently nothing.

### lines 2524-2528

```python
settings.setdefault('image_source', settings.get('crop_source')
```

instruction 230 A: renamed, merged and derived

`crop_source` is IMAGE SOURCE now, and its two values are the two things a user actually chooses between. The old spellings are accepted, because every settings CSV in existence carries one.

### line 2533, trailing  _(unsure)_

```python
settings['crop_source'] = settings['image_source']
```

the old reader

### lines 2535-2538

```python
settings.setdefault('object_array', 'cell')
```

ONE PATTERN, NOT THREE. `file_metadata`, `path_string` and `file_type` between them described one thing, which is three chances to describe it inconsistently. The old three are read when present so an old CSV still loads.

### lines 2541-2543

```python
settings['coordinate_columns'] = _coordinate_columns_for(
```

DERIVED, NOT ASKED FOR. "coordinate column will always be the same so figure that out from object array" -- asking is asking the user to restate something spaCR knows, and giving them a way to get it wrong.

### line 2547

```python
settings.setdefault('stream_method', 'column')
```

instruction 230 B: the stream method and its settings

## get_analyze_recruitment_default_settings

### lines 2631-2642

```python
settings.setdefault('cell_chann_dim',3)
```

NO `*_mask_dim` HERE. Instruction 364 measured it and this closes it: `analyze_recruitment` is the only consumer of these settings and reads `cell_mask_dim`, `nucleus_mask_dim` and `pathogen_mask_dim` ZERO times, against four each for the `*_chann_dim` twins beside them. They are not referenced anywhere in `spacr/submodules.py` at all.

NOT ADDED TO `RETIRED_SETTINGS`, and that is the point of scoping this to recruitment. The same three keys are live in `get_measure_crop_settings` and `set_default_plot_merge_settings`, where they name a plane of the merged stack and are read. RETIRED_SETTINGS carries its own warning against exactly this -- naming a key there that `spacr.settings` still declares would warn a user off a setting that works.

## get_map_barcodes_default_settings

### lines 2729-2731

```python
settings.setdefault('grna', bundled_barcode_path('grna'))
```

These legacy keys are retained for older callers, but their defaults must be portable. The active Qt workflow uses the corresponding row_csv/column_csv/grna_csv keys populated from these same resources.

## get_train_cellpose_default_settings

### lines 2757-2759

```python
settings.setdefault('target_size', 1000)
```

train_cellpose and CellposeLazyDataset consume these keys directly. Keeping target_size aligned with the historical width/height default makes the defaults helper a complete runnable contract.

## set_generate_dataset_defaults

### lines 2775-2777

```python
settings.setdefault('crop_source', 'auto')
```

'auto' uses the PNG crop folder when one exists and falls back to cutting crops out of merged/*.npy on demand; 'png' and 'merged' force one source. See spacr.crops.resolve_crop_source.

## _resolve_regression_analysis_choices

### lines 2869-2870

```python
settings['agg_type'] = None
```

Per-object fitting is exactly agg_type=None. Recorded rather than silently applied, because it changes what one row of the design is.

### lines 2873-2883

```python
chosen = str(settings.get('inference', 'auto')).strip().lower()
```

AND THE INFERENCE FOLLOWS THE UNIT (219). The permutation test works well by well and 'cell' gives one row per object, so the only inference that can run here is the parametric one.

CHOSEN WHEN IT WAS NOT ASKED FOR, SAID WHEN IT WAS. Left at

'auto', this is not a conflict -- the user expressed no preference and 'auto' means "pick what the design supports", so picking it is the whole job. Set to 'nonparametric' explicitly, it IS a conflict: two deliberate choices that cannot both hold, and resolving that quietly would run something other than what was asked for.

### line 2897  _(unsure)_

```python
settings['agg_type'] = 'mean'
```

A well-level run needs a statistic. 'mean' is the historical default.

### lines 2900-2905

```python
level = str(settings.get('level', 'both')).strip().lower()
```

`level` DEFAULTS HERE TOO, not only in get_perform_regression_default_settings, because this function is the one place the analysis choices are resolved and a caller that assembled its own dict (spacr.refit, a sweep trial, a settings CSV written before 2026-08-17) reaches the fit through it. Absent means 'both', which is what every one of those older runs meant.

## _reject_a_threshold_that_cannot_mean_what_it_says

### lines 2970-2971  _(unsure)_

```python
_number('rra_alpha', 0, 1, high_open=False)
```

(0, 1]: rra_alpha=1 scores the whole ranked list, which is the degenerate but meaningful "no cut-off" end of the sweep.

### lines 2981-2983

```python
unanswered = penalty is None or (
```

'auto' and a blank both mean "cross-validate it", which is what the panel now posts for this backend -- see the group_lasso branch of `spacr.ml.regression_model`.

### lines 2987-2989

```python
settings['group_lasso_lambda'] = 'auto'
```

ONE SPELLING DOWNSTREAM. A blank cell, a missing key and the word typed in any case all mean "choose it for me", and the fit should not have to know which of the three it was handed.

## get_perform_regression_default_settings

### lines 3044-3047

```python
_fold_renamed_settings(settings)
```

BEFORE ANY DEFAULT IS FILLED IN (364). A settings file naming a renamed key must reach the new name carrying its VALUE, and a `setdefault` that ran first would already have put the default there -- so the user's number would be silently replaced by ours.

### lines 3051-3056

```python
settings.setdefault('src', '')
```

One row states one score/count relationship. Legacy score_data and count_data keys supplied by an older settings file remain in ``settings`` and are migrated by ml.normalize_regression_input_pairs; they are not defaulted into new files, which now write the explicit paired form. A blank output root places results beside the first count table. See ``spacr.ml.resolve_regression_src`` for resolution and creation rules.

### lines 3059-3062

```python
settings.setdefault('regression_panel_manifest', None)
```

Optional, programmatic publication contract. It is deliberately not a GUI category: a manifest is a structured figure specification (usually a versioned JSON file), not a value that can be edited safely in one line of a settings panel.

### lines 3064-3068

```python
settings.setdefault('analysis_mode', 'regression')
```

``regression`` preserves the historical simultaneous model.  The alternative is a within-plate marginal guide test with empirical P-values and an explicit multiple-testing family; it consumes the same score/count inputs and therefore belongs at this entry point rather than in a disconnected manuscript-only script.

### lines 3070-3099

```python
settings.setdefault('inference', 'nonparametric')
```

THE TWO CHOICES THAT DECIDE THE ANALYSIS, in the words a biologist uses.

`inference` and `analysis_unit` are the plain-language front ends for two decisions that were previously spelled as side effects of other keys: analysis_mode ('regression' vs 'guide_permutation') and agg_type (a statistic vs the None that silently switched the whole model from per-well to per-object). Both are resolved to those historical keys at the bottom of this function, so ml.perform_regression is unchanged and a settings CSV written before this still loads and runs.

inference='auto' is not a coin flip. It measures the design: a simultaneous fit needs more wells than guides, and this screen has 824 guides in 587 wells, which is rank deficient -- the fit returns a coefficient per guide that is not identifiable from the data. Auto therefore chooses the permutation test whenever the design cannot support the simultaneous model, and says so in the log.

The DEFAULT is 'parametric', not 'auto', and that is deliberate. Defaulting to 'auto' would silently switch existing settings files from the simultaneous fit to the permutation test the first time they were re-run -- a different estimand, different columns and different numbers, with nothing in the file changed. A user opts into auto; they are never moved onto it. What the default DOES do is refuse to be quiet about a design it cannot support: perform_regression prints an unmissable warning naming the counts when the parametric path is asked to fit more parameters than it has wells. See ml.resolve_auto_inference. The permutation test is the conservative default for well-level analysis. It assumes neither normal residuals nor equal variance, although it is slower and its P-value resolution is bounded by 1/(permutations + 1).

### lines 3101-3104

```python
if 'analysis_unit' not in settings:
```

Preserve the historical, public ``agg_type=None`` spelling for a per-cell analysis.  New settings use the explicit dropdown, but an old CSV must not silently become per-well merely because this readable front-end key did not exist when it was written.

### lines 3109-3112

```python
if settings['analysis_unit'] == 'cell' and not inference_was_supplied:
```

A cell-level fit cannot use the well-blocked permutation test. When the caller selected only the unit (including legacy ``agg_type=None``), let the resolver choose the compatible parametric path. An explicitly requested nonparametric analysis remains a conflict and is rejected.

### lines 3120-3128

```python
settings.setdefault('guide_nuisance_columns', ['rowID', 'columnID'])
```

ROW AND COLUMN BY DEFAULT (224). A real run came back with

Durbin-Watson 1.22 against 2 for none -- substantial positive autocorrelation in row order -- and the permutation test's whole validity rests on the residuals being exchangeable WITHIN a block. Position left in the residual is position the shuffle treats as noise.

Absent columns are dropped at the call site with a note, so a dataset without them still runs and the user is told which were not removed.

### lines 3133-3142

```python
settings.setdefault('multiple_testing_method', 'fdr_bh')
```

'none' by default: the correction to apply is a judgement about the family being tested, not something spaCR should decide silently. The dropdown offers all thirteen; picking one is a deliberate act. fdr_bh, requested 2026-08-19. The DESCRIPTION already said "fdr_bh (Benjamini--Hochberg, default)" while the code defaulted to 'none', so the documentation and the behaviour had been disagreeing: a user reading the tooltip believed their screen was corrected and it was not. 'none' also writes a q_value EQUAL to the raw p, which is what greys the volcano's adjusted axis -- so the default run could not offer the axis its own menu is built around.

### lines 3145-3162

```python
settings.setdefault('p_threshold_alpha', settings['fdr_alpha'])
```

WHAT "SIGNIFICANT" MEANT, DECIDED BY THE RUN AND NOT BY THE PICTURE.

Instruction 135: "add a setting that setts what alpha the p threshold is set at and if adjusted p or raw p is used". The volcano already offers raw-vs-adjusted on its right-click menu and the RUN had no say in it, so results_significant.csv and the figure printed beside it could be drawn to two different rules with nothing on either saying which.

THE LINE FOLLOWS THE CORRECTION LEVEL UNLESS IT IS MOVED, which is what stops this becoming a second `score_column`: two controls for one question, where the only thing the second can express is a disagreement with the first. Every existing caller moves `fdr_alpha` alone and means "call hits at this level"; a hard 0.05 here would silently ignore them.

They stay separate controls because they answer different questions `fdr_alpha` is what Benjamini-Hochberg TARGETS, an input to the procedure, and this is the level a coefficient is CALLED at. Correcting at 0.05 and reporting at 0.01 is an ordinary thing to want.

### lines 3165-3168

```python
settings.setdefault('rra_alpha', 0.25)
```

Robust rank aggregation's two knobs, declared here because the hit caller that reads them is being written in another file: a setting no defaults factory produces cannot be reached from the Tk panel, the Qt panel or `spacr-run regression`, which all build their dict here.

### lines 3174-3193

```python
settings.setdefault('nontargeting_control_grnas', ['000000'])
```

THE GENE, NOT THIRTY OF ITS GUIDES (195). Asked for 2026-08-21: "default for controlls in regression should be 000000".

This was a hand-typed list of thirty `000000_*` names, and it showed what it was: `000000_2` and `000000_7` are absent, not because those guides do not exist but because the screen somebody read them off did not have them. A library with a thirty-first non-cutting guide lost it silently, and one whose guides keep an organism prefix matched none of the thirty at all.

`spacr.control_names` resolves a GENE to every guide assigned to it, measured from the library in hand, in any of the four spellings a library writes (184). The old list still loads -- `resolve_controls` takes a mixture of genes and guides -- so a settings CSV written before this reproduces.

NOT `negative_control_id`, which stays '233460'. The two are different things: 233460 is a real gene knocked out and expected to show nothing; 000000 binds without cutting and is the empirical null every threshold is measured against.

### lines 3200-3202

```python
for _key in ('exclude_grnas', 'positive_control_wells',
```

Normalize legacy scalar values before type validation. Declaring these settings as lists gives the GUI a chip editor while settings CSVs that contain one unbracketed value continue to load.

### lines 3209-3214

```python
settings.setdefault('fraction_threshold', 0.02)
```

0.02 BY DEFAULT, at the maintainer's direction 2026-08-19. None meant "work one out", which ran `graph_sequencing_stats` on every default run the path that produced the KeyError fixed this morning -- and made the cut different from screen to screen. A stated number is reproducible and is reported in the run summary's exclusions, which now counts what it removed rather than only printing it (156).

### lines 3216-3228

```python
settings.setdefault('calibrate_fraction_threshold', False)
```

MEASURE THE THRESHOLD INSTEAD OF NAMING IT, when the plate design says which wells are pure control.

`fraction_threshold` is a number the user picks, and there is no obvious right one: too low and bleed-through gRNAs survive, too high and real ones are stripped. The control wells can answer it -- refit imaging on sequencing at each candidate cut-off and keep the one where the two agree best -- and `spacr.fraction_calibration` does that.

OFFERED, NOT DEFAULTED. Turning it on changes which gRNAs survive in every well of the screen, so it is a decision the user takes rather than one a version bump takes for them. Off, `fraction_threshold` is read exactly as before.

### lines 3230-3242

```python
for _criterion, _caption in _outlier_criteria():
```

THE FOUR OBJECT-OUTLIER FILTERS ARE NOT OFFERED HERE, and the reason is the order of the pipeline rather than a preference.

Each one excludes objects by a robust z-score over a MEASUREMENT area, or mean intensity. This module reads a score table and a count table, and the measurements are joined to the scores AFTER the fit. So at the moment these would run there is no column to take a median absolute deviation over: the control read as doing something and did nothing.

Still DEFAULTED, so a settings file that names all four loads and runs unchanged -- removing a control must not turn an old run into an error. It is the panel entry that goes, not the key.

### lines 3245-3263

```python
if 'score_column' in settings:
```

ONE COLUMN, NOT TWO (instruction 135 A). `score_column` named the column `minimum_cell_simulation` resamples to find how many objects a well needs before its mean stops moving, and it has been `setdefault('score_column', settings['dependent_variable'])` ever since the two were reconciled -- one measurement under two names. All a second control could add was a way to simulate the minimum cell count on a column the model does not fit, and then keep or drop wells on it.

MIGRATED, NOT DROPPED, the way `toxo` -> `Toxoplasma` is below: every regression settings CSV written before today carries it. A file with no `dependent_variable` gets the old value; a file whose two keys DISAGREE is told which one this run fits, because that disagreement is what changed the old run's answer and silence about it is what makes the re-run inexplicable.

THE KEY ITSELF IS NOT RETIRED. `interpret_vision_model` and

`hit_investigation` use `score_column` for the CNN score column ('cv_predictions'), so it keeps its type, its tooltip and a category there; what is retired is the REGRESSION module's duplicate.

### lines 3277-3278

```python
settings.setdefault('transform', 'log')
```

log by default: screen responses are fractions and skew hard, and the normality check fails on the raw column far more often than not.

### line 3282

```python
settings.setdefault('min_cells_per_well', 100)
```

100 cells: below that a well's score is noise dressed as a measurement.

### lines 3284-3294

```python
settings.setdefault('regression_type', 'mixed')
```

MIXED IS THE DEFAULT, and the maintainer's reason is the design rationale rather than a preference: "mixed answers the most central question best" (2026-08-17, instruction 132). The central question a CRISPR screen asks is about the GENE, and mixed is the only model here that says what a guide is -- a biological replicate of one intended perturbation, nested inside its gene -- instead of a second independent variable competing with it for the same variance.

It was 'ols' until 2026-08-17. This is a setdefault, so a settings CSV that names a type still gets that type; only a dict that never chose one moves.

### lines 3296-3306

```python
settings.setdefault('intercept', 'fitted')
```

WHAT THE INTERCEPT IS, offered rather than assumed.

'fitted' estimates it from the data, which is what every run before this key existed did and is why it is the default. The other three exist because a fitted intercept answers a question a screen does not always ask: it is the response of a well whose every predictor is at its reference level, which for a one-hot gene design is whichever gene patsy happened to drop. 'control' pins it at the negative controls, so every coefficient reads as a difference FROM the controls -- the thing a screen is usually asking about. 'zero' fits through the origin, and 'value' pins it at `intercept_value`.

### line 3308

```python
settings.setdefault('intercept_value', 0.0)
```

Read only under intercept='value'; the panel greys it otherwise.

### lines 3310-3325

```python
settings['regression_backend'] = _resolve_regression_backend(
```

WHO FITS IT, as opposed to WHAT is fitted (instruction 141 A). The two are independent: the same mixed model can be fitted by statsmodels on the CPU or by spacr.mixed_gpu on the GPU, and the answer should be the same while the time is not.

DEFAULT 'statsmodels', and this one is not a preference. Every results.csv, every volcano and every hit list this project has produced came out of statsmodels, so a default that moved would change the numbers under a user who changed nothing -- which is not a default.

NORMALISED, not just defaulted, because both GUIs render a combo's options verbatim and instruction 141 C requires each option to read '(CPU)' or '(GPU)'. So the option strings ARE the labels, the panel posts 'torch (GPU)', and this is where it becomes 'torch'. An old settings CSV has no such key at all and gets the default, which is what every one of those files meant.

### lines 3328-3333

```python
settings.setdefault('level', 'both')
```

WHICH LEVEL A FIXED-EFFECTS FIT REPORTS AT. No settings CSV written before 2026-08-17 carries this key, and every regression run before then wrote one, so the default has to be the behaviour those files meant: 'both' levels. Read only by the non-mixed families; see get_setting_dependencies, which greys the control out under 'mixed' and says why, rather than hiding it or leaving it present and inert.

### lines 3335-3377

```python
settings.setdefault('model_plate_position', False)
```

IS PLATE POSITION IN THE MODEL AT ALL (instruction 143 A: "is rowID columnID always run? this should be an opt in"). It was: prepare_formula ended with an unconditional "+ rowID + columnID", and random_row_column_effects only chose FIXED or RANDOM for terms that were already in. There are three states now -- out, fixed, random -- and this key picks the first one. Out plus random is a contradiction and is refused by ml._reconcile_random_row_column_effects, not resolved.

DEFAULT ON, AGAINST THE INSTRUCTION'S SUGGESTION OF OFF, because the instruction asked for the default to be MEASURED and the measurement says on. Fitting the maintainer's TSG101 screen (1945 rows, 610 wells) twice per level: the 35 position terms are jointly significant at F = 5.781, p = 6.71e-23 (guide) and F = 6.277, p = 2.33e-26 (gene); eight of the nine real screens on that machine reject the same null at p < 0.05. Dropping them costs 8.4 points of R2 (0.5477 -> 0.4634), inflates the residual sd 7.2% and so makes standard errors 5.5% LARGER, moves the median guide coefficient 0.271 of its standard error, and swaps named genes in and out of the exported hit list (277230 out at q 0.0394 -> 0.4071, 258462 in at q 0.1134 -> 0.0146). On synthetic data with the truth planted it loses 17% of the true hits at BH.

Default ON is wrong only on a plate with no position effect at all, and there it costs 35 parameters, 3% of the residual degrees of freedom, 1.6% on the standard errors and 0.02 hits out of 20. The two mistakes are more than an order of magnitude apart, so the default is the cheap one to be wrong about. See ml.prepare_formula for the whole measurement.

AND IT IS THE MIGRATION. Unlike `toxo` -> `Toxoplasma` or the retired `score_column` there is no old key to pop and no old value to carry across: no settings CSV written before 2026-08-18 has this key at all, and what every one of those files MEANT is plate position in the model. A setdefault of True is therefore what makes an old file still mean what it meant -- and the same setdefault is what keeps an old CSV carrying random_row_column_effects=True from landing in the refused fourth state. OFF BY DEFAULT, at the maintainer's direction 2026-08-19. Instruction 143 chose True on a measurement -- omitting a real plate-position effect cost more than carrying an absent one -- and that measurement stands as a description of the two errors. The DEFAULT is a judgement about which error to take by default across everybody's screens, and that judgement is the maintainer's. An old settings CSV without the key now means "position out" rather than "position in": stated here because the comment above says the opposite about files written before 2026-08-18, and a reader of one line without the other would be misled.

### lines 3381-3391

```python
if settings.get('regression_type') in PENALISED_REGRESSION_TYPES:
```

A PENALTY OF 1 IS NOT A DEFAULT, IT IS A FAILURE, for these families.

A fraction design's coefficients live around 1e-2, so alpha=1 shrinks every one of them to exactly zero: measured on a real screen, all 1,208 of them, and the fit then raises rather than reporting a table of zeros. A default that cannot succeed on the data the module is for is worse than no default. 'auto' cross-validates the penalty instead.

Only for the penalised families. quantile REFUSES any alpha but 1 (it uses its own `quantile` key and an alpha there means the user has confused the two), and the unpenalised families ignore it.

### lines 3394-3419

```python
if settings.get('alpha') == 1:
```

AND THE POSTED DEFAULT COUNTS AS ABSENT. `setdefault` never fired from the GUI: the panel posts every key it shows, so `alpha` arrived as the integer 1 -- the default for the families that ignore it and the penalised fit then refused, every time, on the first run a user made. Measured on the reference screen: alpha=1 shrank all 790 coefficients to exactly zero.

1 IS NOT A PENALTY ANYBODY CHOOSES for a design of fractions; it is the value the panel had lying around. Cross-validating instead is strictly better than a guaranteed refusal, and it is ANNOUNCED rather than done quietly -- a penalty chosen for the user and never named is one they cannot put in a methods section.

ANY 1, WHATEVER ITS TYPE. This used to spare a FLOAT 1.0, on the reading that an integer is the posted default and a float is a deliberate answer. That was true of the Tk panel and is not true of this one: the Qt field is a double spin box and the settings CSV it writes says `alpha,1.0`, so the rescue never fired for anyone running the current GUI. Driven on the tsg101 screen's own saved settings (236 C7), lasso and elasticnet both refused the run "shrank all 298 coefficients to exactly zero at alpha=1.0" -- from a settings file in which nobody had ever touched alpha.

Nothing is lost by dropping the escape hatch: a literal penalty of exactly 1 on a fraction-scale design is the value the guard downstream refuses anyway, and any other number is honoured.

### lines 3429-3437

```python
settings.setdefault('l1_ratio', 0.5)
```

Every knob below is read by spacr.ml.regression_model for at least one regression_type, and each one is INDEXED (settings[...]) by perform_regression, not .get()-ed: a model that reads a setting must have a default here or the module is unstartable from every entry point, which is exactly how six other keys took regression down.

The defaults match regression_model's own signature defaults, because _reject_unused_settings compares against them to tell "the user asked for this" from "the panel posted its default".

### lines 3443-3444  _(unsure)_

```python
settings.setdefault('spline_knots', 4)
```

The spline basis. Read only by regression_type 'spline', where the COVARIATES are given a basis and the guide columns are left alone.

### lines 3449-3462

```python
settings.setdefault('group_lasso_lambda', 'auto')
```

The group lasso's penalty weight. Declared here for the same reason as the RRA pair above -- the family that reads it lives in spacr/ml.py and a knob no defaults factory produces is a knob no entry point can set.

WHICH FAMILY READS IT IS NOT WRITTEN BY HAND, for any of the three. `regression_spec.REGRESSION_SETTINGS_USED` claims them for 'group_lasso' and 'rra', so `_name_the_family_in_every_estimator_tooltip` ends each tooltip with "Read by regression_type '...'", and the same table generates the rule that greys the control out. A second, hand-written list would be the one that drifted. 'auto' RATHER THAN A NUMBER. A penalty is only large or small relative to the design it is applied to, and 0.05 is nearly half of the tsg101 screen's own ceiling -- it emptied all 297 gene blocks and the run was refused, from settings nobody had touched (236 C7).

### lines 3464-3470

```python
if settings.get('group_lasso_lambda') == LEGACY_GROUP_LASSO_LAMBDA:
```

AND A SAVED 0.05 IS THE OLD DEFAULT, not an answer. It was what the panel posted for the whole of this setting's life, so every settings file written before today carries it -- and now that it is no longer the default, leaving it in place would make those files ask for a penalty under every OTHER regression type, which refuses a setting it cannot read. Converted rather than tolerated, so the old file means today what it meant when it was written: nobody chose this.

### lines 3479-3506

```python
settings.setdefault('regression_qc', True)
```

The diagnostic suite, on by default because one analysis wants it and the person running one analysis is the one who will not think to ask. It is NOT a per-family knob: every regression_type is fitted through the same `regression()` call and every one of them can be badly specified, so it is deliberately outside REGRESSION_SETTINGS_USED and is not policed by _reject_unused_settings. A parameter sweep turns it off in `parameter_sweep._trial_settings`, where a hundred trials would be ten minutes and two thousand figures nobody opens.

STILL A PARAMETER, THOUGH THE PANEL NO LONGER OFFERS ONE. Instruction 135 asks for it hard-coded True and says to check the sweep first, and the check says do not: the sweep's escape route IS this key. `parameter_sweep._trial_settings` writes `regression_qc=False` into the trial's dict, `perform_regression` re-applies THIS function to that dict (ml.py:4922) and only then reads `settings.get('regression_qc', True)`, so forcing True here would overwrite the sweep's False and put ~5.8 s and ~19 figures per trial back on all hundred of them. setdefault leaves the parameter reachable by the sweep while the GUI control goes.

THE SWEEP'S OWN ESCAPE IS HALF BROKEN ALREADY, measured the same day and reported rather than fixed here because parameter_sweep is another file: its `setdefault("regression_qc", False)` does nothing when the base dict already carries the key, and every base dict built by the Tk panel, the Qt panel or `spacr-run` carries it at True -- because all three build it HERE. Pinned by tests/test_the_regression_settings_fit_on_one_page.py. That is an argument for keeping the key, not for removing it: it is the only way the sweep has of saying no at all.

### lines 3510-3517

```python
settings.setdefault('count_grna_column', 'grna')
```

THE COUNT TABLE'S TWO COLUMNS, WHICH WERE HARD-CODED (instruction 135 B: "i think these are hardcoded and dont have a settins"). spacr.ml demands ['rowID', 'columnID', 'grna', 'count'] of the count CSV and raises "The CSV file must contain 'grna', 'count', 'rowID', and 'columnID' columns" naming the four it wants and none of the ones the file HAS, which is the failure that instruction's whole section is about. The plate coordinates are resolved elsewhere; these two are the ones a sequencing pipeline spells differently ('sgRNA', 'reads', 'n').

### lines 3520-3525

```python
settings.setdefault('independent_variable_layout', 'auto')
```

Count inputs may be one row per (well, guide) or one row per well with one guide per column. ``auto`` recognizes the former from the paired guide/value columns and otherwise uses the wide-column path. The model layout is a separate choice: a long input can be collapsed to one row per well for fixed-effects estimators, while a wide input can be melted for the historical formula and permutation paths.

### lines 3529-3547

```python
_blocks: list = []
```

sequencing.graph_sequencing_stats iterates settings['analysis_excluded_wells'] and drops those wells from the count table before it sweeps for the fraction threshold, exactly as ml.clean_controls drops filter_value from the score table. The two must name the same wells or the threshold is fitted on wells the regression never sees, so this follows filter_value. It is indexed, not .get(), and it is iterated, so None is not a legal value here.

IT WAS `control_wells` UNTIL 2026-09-09 AND THAT KEY MEANT TWO THINGS. The invasion assay used the same name for its STAIN BASELINE wells and defaulted it to None, so one settings file set both at once and a user changing one silently changed the other. Split into `analysis_excluded_wells` here and `stain_baseline_wells` there (364, 357-Q6); an old value migrates to both. AND THE THREE CONTROL BLOCKS ARE PART OF IT (221). `filter_value` gains them in `_perform_regression`, which runs after this, so deriving from `filter_value` alone left the sequencing sweep fitting its threshold on wells the regression had already dropped -- the exact divergence the comment above says must not happen.

### lines 3568-3569

```python
settings.setdefault('metadata_files', [])
```

Acquisition-specific annotations cannot have a meaningful machine-wide default. An empty list makes the optional input explicit and portable.

### lines 3571-3583

```python
settings.pop('volcano', None)
```

grna: the per-gRNA table is the one that shows whether a gene's signal is carried by every guide or by a single outlier. `volcano` IS GONE. It chose which coefficient table the volcano was drawn from -- 'gene' | 'grna' | 'all' -- before the run. The interactive volcano filters between genes and guides by right-click now (instruction 129 A), on the same fit and with no re-run, so a setting that could answer the question once was redundant.

A settings CSV that still carries it is ACCEPTED AND DROPPED rather than refused: every regression run before 2026-08-17 wrote one, and a saved settings file that suddenly fails to load is a worse outcome than a key nothing reads. Same treatment `location_column`, `positive_control_id` and `negative_control_id` already get above.

### lines 3585-3601

```python
for _retired, _forced in (('log_x', False), ('log_y', False),
```

THE REGRESSION PLOT SETTINGS STOP BEING SETTINGS (instruction 135): "Regression plot can be removed . hard code regression qc and guide permutation plot to true, logx and logy to false and x and y lim should be set automatically and can be changed on the plot."

None of the six was a question about the SCREEN. They styled the figures a run draws, which is a decision better made while looking at the figure than before the fit: x_lim/y_lims absent means the plotting code scales to the data and the interactive volcano rescales afterwards, log_x and log_y are False, guide_permutation_plot is True, and split_axis_lims was read by nothing at all.

ACCEPTED AND DROPPED rather than refused, exactly like `volcano` above: every regression settings CSV written before today carries all six, and a saved file that suddenly fails to load is a worse outcome than a key nothing reads. A value that DISAGREED with the hard-coded one changed what that run drew, so it is named on the way out instead of vanishing.

### lines 3612-3617

```python
settings.pop('split_axis_lims', None)
```

`split_axis_lims` retires with the five above but LEAVES IN SILENCE. The other five each decided something a run drew, so a value that disagreed with the hard-coded one changed a figure and is worth naming. Nothing ever read this one, so there is no behaviour to report a change in -- and an old settings CSV that carries it would otherwise keep an undeclared key in the dict every downstream reader then has to ignore.

### lines 3619-3637

```python
if 'toxo' in settings:
```

`toxo` BECAME `Toxoplasma` on 2026-08-17 (instruction 133): "change the toxo settings to Toxoplasma". An old settings CSV carries the old spelling, and dropping it would turn the annotation off without saying so -- so the value MIGRATES rather than being ignored.

POPPED, not kept alongside. Keeping both would put two controls for one question on the panel and give `toxo` a category it is no longer entitled to. Every reader goes through `ml._toxoplasma_is_on`, which accepts either spelling, so a caller that hands ml.py a raw dict with the old key still works.

ONE POP RATHER THAN AN INDEXED READ BEHIND AN `in`. The two lines meant the same thing, but the contract test that walks perform_regression and the helpers it hands the dict to (test_regression_entry_points) reads `settings['toxo']` as a key that must have a default -- it cannot see the guard -- and `toxo` is a key this function exists to REMOVE, so it never can have one. That assertion has been failing on the old spelling since the rename; taking the value out with the pop that was always coming answers it without changing what any run does.

### lines 3641-3652

```python
settings.setdefault(
```

`Toxoplasma` BECAME A FIELD on 2026-08-23: "it would be cool if that was a field that you could fill with any uniprot id or organism name and it would pull what uniprot has. Defaults to Toxo and Toxo data hardcoded as before."

The boolean is kept and still read, because every settings CSV in existence carries it and because `ml._toxoplasma_is_on` is what the bundled path asks. True means the bundled Toxoplasma tables, which is what `annotation_source` defaults to; False means no annotation at all, which is the one thing a NAME cannot express -- so it maps to the empty string and `spacr.uniprot.resolve` reads that as the bundled path being off.

### lines 3656-3658

```python
settings.setdefault('verbose', False)
```

perform_regression prints a per-stage row count and display()s the whole per-object score table under verbose, which is millions of rows on a real screen, so this pipeline is one of the False ones.

### lines 3660-3662

```python
settings.setdefault('tolerance', 0.02)
```

minimum_cell_simulation reads settings['tolerance'] and accepts an int (percent) or a float (fraction); anything else raises ValueError. 0.02 is the 2% the function's own worked example uses.

### line 3664

```python
settings.setdefault('invert_dependent_variable', False)
```

process_scores: False/0 = as measured, True/1 = 1 - x, -1 = 1 / x.

### lines 3670-3676

```python
if settings['alpha'] != 1:
```

alpha USED to double as the quantile here, which was a silent overload of a key whose tooltip, GUI label and every other regression type call a penalty weight: a settings CSV reading alpha=0.9 meant "the 90th percentile" under one regression_type and "shrink hard" under the next. The quantile now has its own key, and an alpha left over from the old spelling is refused rather than ignored - silently dropping it would fit the median and label the output 0.9.

### lines 3691-3699

```python
if str(settings.get('analysis_mode')) == 'guide_permutation':
```

ONLY WHEN THE QUANTILE FIT IS THE ONE THAT RUNS.

Under analysis_mode='guide_permutation' -- the default since 2026-08-19 -- regression_type is never read: the permutation test is the analysis. Forcing agg_type=None on its behalf left the run with one row per OBJECT feeding a test that needs one per WELL, and it died on "Phenotype/block/nuisance values are not constant within well". A setting that will not be used must not be able to break the analysis that will.

### lines 3709-3710

```python
settings['agg_type'] = None
```

Quantile regression on per-well MEANS would be the quantile of an average, which is not the quantile of the response.

### lines 3714-3737

```python
if settings['regression_type'] in ('poisson', 'horseshoe'):
```

A COUNT MODEL'S RESPONSE IS A COUNT, AND log(count) IS NOT ONE.

`poisson` and `horseshoe` are fitted as Npositive ~ ... offset(log(Ntotal)). ml.process_scores already knows this and overrides agg_type to 'count' for exactly these two families, taking the well's SUM instead of its mean -- and then applies `transform` to that sum like any other response. With the default transform='log' the integer count becomes a float, and _validate_poisson_response refuses it with "Poisson regression requires integer count data". Every entry point builds its dict here, so the effect was that neither count family could be run at all from Tk, Qt or the CLI without knowing to set transform off by hand.

The log belongs to the fraction responses it was defaulted on for ("screen responses are fractions and skew hard"); a count model already has a log LINK, so transforming the response as well would log it twice. Resolved here beside the quantile rule because it is the same kind of thing -- a model choice that decides how the response must be prepared and because doing it here fixes all three dispatchers at once.

NOTE: the narrower repair belongs in ml.process_scores, next to the count_models override it sits beside, so that a direct call to regression()/process_scores() with transform='log' is covered too. This rule makes the settings path correct; it does not make that one safe.

### lines 3746-3749

```python
settings.setdefault('strict_errors', None)
```

Fail-loud policy. None means "not set here" and defers to the SPACR_STRICT_ERRORS environment variable, which is how a cluster turns it on for a whole batch without editing every settings file. True/False here is an explicit per-run choice and wins over the environment.

## _name_the_family_in_every_estimator_tooltip

### lines 5801-5804

```python
if 'Read only by regression_type' in text:
```

ALREADY SAID IS ALREADY SAID. Some of these tooltips end with the project's existing "Read only by regression_type 'x'." convention, which answers the same question in the same place; appending a second sentence saying it again reads as a mistake.

### lines 5808-5811

```python
tooltips[key] = (
```

SHORT. The greyed-out state carries its own reason -- the dependency rule writes "not read when regression_type is 'x'" onto the disabled control -- so repeating it here spends the tooltip budget saying twice what the panel already says once.

## _advanced_family_members

### lines 6439-6442

```python
spoken_for = {k for c in _ADVANCED_REGROUP_EXEMPT
```

A key an exempt category owns is NOT a member of the family heading. Without this it would end up in both, and a duplicate renders twice in Tk (each copy shown or hidden by a different heading) and is silently dropped from the second section in Qt.

### lines 6445-6449

```python
filed = {key for members in table.values() for key in members}
```

ONE SET, BUILT ONCE. "Is this key filed anywhere?" used to be asked as a scan of every category list per candidate, which is objects x suffixes x every key in the table -- and the object list is now as long as `number_of_organelles` may go, so that product grew by more than an order of magnitude and was paid at import.

## _regroup_advanced

### lines 6471-6475

```python
out = dict(table)
```

IDENTITY IS PRESERVED FOR EVERY CATEGORY THIS DOES NOT TOUCH. Several headings are the module-level lists themselves -- `categories` ["Motility (beta)"] IS `motility_settings` -- and a test asserts that `is` relationship. Rebuilding the whole dict with fresh lists broke it for categories the regroup has no business changing.

### line 6485, trailing  _(unsure)_

```python
continue
```

untouched: keep the original list object

## get_setting_dependencies

### lines 6559-6565

```python
from .regression_spec import REGRESSION_SETTINGS_USED
```

FROM THE SPEC, NOT FROM ml. `spacr.ml` imports `spacr.plot`, which imports torch, cv2 and IPython -- so this one lookup of a dict of strings cost 2.2 seconds and 900 MB every time a settings panel was built, on the GUI thread, for every module. There is a test asserting panel-building does not import the plotting stack; it was written when that import cost 770 ms and was "the whole remaining cost of opening the first module", and torch made it four times worse.

### lines 6626-6630

```python
'grna_statistic',
```

`grna_statistic` sits FIRST in the Permutation Test category and is read only by that path -- it says WHAT is measured, and nothing measures it on a fitted run. It was the one control in that section with no rule, so a parametric run greyed its eight neighbours and left it live.

### lines 6686-6706

```python
for key in ('regression_type', 'regression_backend', 'cov_type',
```

AND THE MIRROR OF IT: the model settings are dead under permutation.

Asked on 2026-08-20 -- "if i use a mixed model an nonparametric, is that still regression or multiple linear regression" -- and the honest answer is that you do not get both. `inference='nonparametric'` selects `analysis_mode='guide_permutation'`, which NEVER CALLS `regression_model`: it is a per-guide marginal association test with Freedman-Lane permutations, and it fits no simultaneous model at all. `regression_type` is read, stored, saved into the settings CSV, and then not used.

The run summary already says so AFTERWARDS. Saying it at the point of choosing is 106's rule, and this is exactly the case for it: a user picking `mixed` here is choosing a model they will not get.

NOT UNDER 'auto', for the same reason the permutation controls stay live there: its real resolution counts guides and wells, which this cannot see, so greying a setting the run may well read is the worse error of the two. `intercept` and `intercept_value` say where a fitted line is anchored. A permutation test fits no line, so there is nothing to anchor.

### lines 6730-6748

```python
setting_dependencies['transform'] = _combined(
```

THE GROUP LASSO'S PENALTY IS DEAD UNTIL THE GROUP LASSO IS CHOSEN.

setdefault, not assignment: the loop above generates exactly this rule for every key `REGRESSION_SETTINGS_USED` claims, and the moment regression_spec lists 'group_lasso' there the generated rule -- which cannot drift from the family table or from the tooltip generator that reads the same table -- takes over and this one stops being used. A LINK-LIKE TRANSFORM IS DEAD ON A GLM, so the control says so.

A glm fits the response as measured and lets its family's own link do the transforming, so a `transform` that is itself a link -- 'log' or 'logit' -- is read and then ignored. A control the run silently discards has to say so.

ONLY the link-like transforms, and only on a glm. 'sqrt' and 'square' are not links and are applied normally, and a regression type that does not choose its own family has no conflict to resolve. Greying `transform` for those would be a control disabled for a reason that is not true of it.

### lines 6764-6769

```python
setting_dependencies['intercept_value'] = _combined(
```

THE PINNED NUMBER IS DEAD UNLESS THE INTERCEPT IS PINNED.

`intercept_value` is read by exactly one of the four modes. Under the other three the field is still shown -- so the user can see what it would be -- but greyed, with the reason naming the mode that is actually in force rather than the one that would read it.

### lines 6793-6802

```python
setting_dependencies['analysis_mode'] = rule(
```

`analysis_mode` IS DEAD WHILE `inference` IS DECIDING IT.

Instruction 134, and instruction 106's rule about how: greyed out with the reason on it, never silently inert. `inference` is the readable front end that SETS `analysis_mode` (_resolve_regression_analysis_choices), and it does so for every value except 'auto'. A user who picks 'nonparametric' and then reads a live `analysis_mode` box still saying 'regression' is looking at two controls that contradict each other, and the one they can edit is the one that loses.

### lines 6880-6882

```python
_cell_unit = lambda settings, context: str(
```

Cell-level analysis keeps one row per object, whereas the permutation test requires one row per well. Reflect that constraint in the enabled state so the GUI explains the selected inference mode before a run.

### lines 6944-6958

```python
_existing = setting_dependencies.get('guide_permutation_block')
```

THE EFFECT-SIZE CUT APPLIES TO A PERMUTATION RUN TOO, so these two are no longer greyed out under it.

The rule used to read "guide permutation uses corrected P values", and that reason is now false. It was true only because `perform_regression` RETURNED from the permutation branch about eighty lines before the block that computes the cut -- an accident of control flow, not a statement about the method. The permutation table carries a real `coefficient` (aliased from `standardized_marginal_effect`), and an effect-size cut asks how BIG an effect is, which is a separate question from how its P value was obtained.

The maintainer reported it as "why cant i see the coefficient threshold if im running nonparametric regression?", and was told the greying was correct. It was not.

### lines 6960-6987

```python
_existing = setting_dependencies.get('guide_permutation_block')
```

THE DATA-DEPENDENT HALF, and the reason `context` exists at all.

Every rule above reads only the other SETTINGS. This one reads the loaded data: `context['plate_count']` is filled by the panel from the inputs the user actually dropped in.

UNKNOWN MUST NOT GREY ANYTHING. `plate_count` is None when nothing is loaded, or when the inputs were too large to scan cheaply. A control disabled because a file was big is indistinguishable, to the person looking at it, from one disabled on purpose -- so absence of knowledge leaves the control alone.

`guide_permutation_block` names the column permutations are blocked within, and residuals are never shuffled between its levels. With one plate, blocking on the plate is the whole dataset and constrains nothing.

All three input keys are listed because a panel has whichever of them it has -- the regression screen takes pairs in one `paired_data` table (instruction 107) and has neither of the other two and _connect_setting_dependency_signals skips the ones that are absent.

ADDED TO the rule already there, never assigned over it. This dict holds ONE rule per setting, and `guide_permutation_block` already has one -- parametric inference greys every permutation control. Assigning here replaced it, so choosing parametric silently left this one field enabled among its greyed siblings. A setting is applicable only if EVERY rule that mentions it says so.

### lines 6992-6998

```python
lambda settings, context: (
```

TRUE MEANS APPLICABLE, matching every rule above -- the estimator rules return True when the setting IS read. So: enabled when the plate count is unknown, or when there is more than one plate. `context or {}` because a caller with no loaded inputs passes None, and every other predicate here tolerates that. Raising instead made this one rule the only way to crash a panel that is merely asking whether to grey a control.

### lines 7010-7025

```python
for _key in ('batch_correction', 'batch_column', 'batch_control_column',
```

AND BATCH CORRECTION, for the same reason and by the same route (instruction 135's last open line). Correcting BETWEEN batches when there is one batch removes nothing -- the run already says so out loud ("fewer than two batches means there is nothing between batches to remove, and batch_correction='none' gives an identical result") -- but saying it AFTER the run is saying it too late. The control is greyed before the user chooses.

THE SAME `_combined`, never an assignment: `batch_correction` may already carry a rule, and a setting is applicable only if EVERY rule that mentions it says so. Assigning here would silently drop the other.

And the same "unknown greys nothing": `plate_count` is None with nothing loaded or with inputs too large to scan cheaply, and a control disabled because a file was big is indistinguishable from one disabled on purpose.

### lines 7033-7035

```python
lambda settings, context: (
```

See the note on `guide_permutation_block` above: a caller with no loaded inputs passes None, and a greying question must not be the thing that raises.

### lines 7046-7059

```python
from .stream_dataset import METHOD_SETTINGS as _METHOD_SETTINGS
```

WHERE THE PIXELS COME FROM, and what each route actually reads. `image_source` chooses between reading exported crops and cutting them from merged planes, and `stream_method` chooses how the streamer finds its objects. The settings the chosen route does not read must not sit on screen asking to be filled in: a control that changes nothing is indistinguishable from one that does, and the user finds out at run time or not at all.

FROM `stream_dataset.METHOD_SETTINGS`, never a second list here. That table is what the STREAMER reads; a copy in the settings module would drift from it, and the symptom would be a live control the run ignores which is the exact failure this gate exists to prevent. The module imports numpy and pandas and nothing heavier, so this stays off the plotting stack the panel build is tested for.

### lines 7068-7071

```python
_stream_only = tuple(dict.fromkeys(
```

`load_path_regex` had a dependency rule here saying when it did not apply. The setting was retired on 2026-09-09 (357-Q4) because nothing read it in either case, and a rule explaining when an unread control is inapplicable explains nothing.

### lines 7087-7090

```python
for _key in _stream_only[1:]:
```

AND WITHIN STREAMING, ONLY THE CHOSEN METHOD'S OWN SETTINGS. An unrecognised method greys nothing: a control disabled because a settings file named a method spaCR never had is a control nobody can re-enable from the panel.

### lines 7104-7114

```python
setting_dependencies['custom_regex'] = _combined(
```

Mask input and metadata

`custom_regex` is only read by two conventions, so under the other two it is a control that changes nothing.

'auto' IS INCLUDED, and that is the part worth stating: it is not the obvious reading of "grey it out unless metadata type is custom". The description for `metadata_type` says 'auto' renames the folder to Yokogawa naming "using custom_regex when supplied, otherwise automatic detection" -- so a user on 'auto' who has a regex CAN use it, and greying the field there would take away a documented behaviour while looking like a tidy-up.

### lines 7127-7129

```python
setting_dependencies['organelle_model_name'] = _combined(
```

Organelle segmentation

Every other organelle_method is a threshold or a filter and takes no checkpoint; only cellpose loads one.

## get_setting_dependencies.permutation_is_certain

### lines 6680-6681

```python
return (inference != 'auto'
```

No usable `inference`: fall back to an explicit analysis_mode, and 'auto' is never certain.

## get_setting_dependencies._is_nonparametric

### lines 6815-6835

```python
def _is_nonparametric(settings):
```

THE LEVEL IS DEAD UNDER A MIXED MODEL, AND THE PANEL SAYS SO.

Instruction 132 A, and instruction 106's rule about how: disabled and SAYING WHY, never absent and never present-but-inert. A mixed model already answers both levels at once -- the gene is a fixed effect and each guide a random effect nested inside it -- so there is no single level left to choose, and a dropdown that still looked live would be accepting a choice nothing would honour.

random_row_column_effects IS PART OF THE CONDITION, not an afterthought. ml._reconcile_random_row_column_effects rewrites regression_type to 'mixed' before anything is fitted, so ticking that box with regression_type='ols' and level='grna' fits a mixed model and ignores the level. Reading only regression_type here would leave the control enabled for a run that cannot use it, which is the exact failure the rule exists to prevent.

HARMLESS IN THE OTHER MODULES THAT OWN A 'level'. The proportion and endodyogeny panels use the same key for 'object'/'well'/'plate' and carry no regression_type, so the predicate reads '' != 'mixed' -> True -> applicable, and their control is never greyed by this.

## get_setting_dependencies._level_is_read

### lines 6841-6846

```python
"""Whether the level setting is read at all under these settings.
```

THE PERMUTATION TEST READS IT. It fits no model, so regression_type says nothing about it and neither does random_row_column_effects both are parametric answers to a parametric question. Greying the control here left the nonparametric side with no way to ask for genes at all, because the key that gated its gene pass has no control of its own.

## parse_list

### lines 7182-7184

```python
return list(parsed_value) if len(parsed_value) > 1 else [parsed_value[0]]
```

A one-element tuple is what `(3,)` parses to; it means the same single value the user typed, so it stays one element rather than being flattened away.

## check_settings

### line 7212, trailing

```python
errors = []
```

Collect errors instead of stopping at the first one

### lines 7228-7242

```python
settings[key] = None
```

Blank means "not set", and for these keys that is a legal, shipped value: cell_/pathogen_/treatment_plate_ metadata default to None in eight factories (recruitment, invasion, replication, endodyogeny, class proportion, plot_data_from_db) and timelapse_frame_limits is declared (list, NoneType) outright.

This used to assign `parsed_value = None` and fall straight into the `else` two lines below, which raised "Expected a list ... but got NoneType" -- so every one of those modules reported errors and dropped its own default the moment it was run from the Tk panel untouched. The assignment was evidence of the intent and nothing else: no value of `value` could reach the list branch through it.

### lines 7288-7291

```python
if value is None:
```

invert_dependent_variable: False/0 = as measured, True/1 1 - x, -1 = 1 / x. The generic tuple branch at the bottom would reach bool('False') first, which is True, and silently invert every score in the screen.

### lines 7308-7312

```python
if value is None:
```

y_lims / x_lim / stain_baseline_wells / filter_min_max. The tuple branch would reach list('[0, 5]') first and hand the pipeline ['[', '0', ',', ' ', '5', ']']. literal_eval also keeps the nested form y_lims uses for a broken axis, which parse_list rejects as "mixed types".

### lines 7333-7337

```python
settings[key] = list(value) if value else None
```

Already a list: keep it. This used to call parse_list, which ast.literal_eval()s its argument and so raises on anything that is not a string -- the one branch reached by a value that is already the declared type was the one that threw the value away and logged a format error.

### lines 7345-7347

```python
if value is None or isinstance(value, bool):
```

A detector switch may be off/on OR name a model/path. bool("False") is True and str(False) is "False", so the generic tuple coercer cannot preserve this union.

### line 7391  _(unsure)_

```python
for error in errors:
```

Send all collected errors to the queue

## set_annotate_default_settings

### line 7417, trailing  _(unsure)_

```python
settings.setdefault('measurement', '')
```

'cytoplasm_channel_3_mean_intensity,pathogen_channel_3_mean_intensity')

### line 7418, trailing  _(unsure)_

```python
settings.setdefault('threshold', '')
```

'2')

### lines 7420-7434

```python
settings.setdefault('crop_source', 'png')
```

'auto' uses the PNG crop folder when one exists and falls back to cutting crops out of merged/*.npy on demand; 'png' and 'merged' force one source. See spacr.crops.resolve_crop_source. LOAD IMAGES BY DEFAULT, BY NAME (instructions 170 and 171).

This shipped 'auto', which takes the PNG folder whenever one exists the right answer for the wrong reason, and unaskable when a user wants the other. Asked 2026-08-19: "in the annotation app how do i choose to stream images from database or dataset". The answer was that you did not: the setting was here and the choice was never offered.

'png' IS "load images" and 'merged' is "stream images". The stored value does not change, so no settings file already on disk changes meaning, and `resolve_crop_source` falls back to the other route when this one's folder is absent -- saying so in its reason.

### lines 7436-7437  _(unsure)_

```python
settings.setdefault('queue_by_uncertainty', False)
```

Active-learning queue (spacr.active_learning). Off by default: it needs model scores in png_list, which only exist after Classify (CV).

## set_default_generate_barecode_mapping

### lines 7453-7455

```python
_fold_renamed_settings(settings)
```

BEFORE ANY DEFAULT IS FILLED IN. A settings file naming the old key must reach the new one carrying its VALUE, and a `setdefault` that ran first would have already put the default there.

### lines 7458-7461

```python
settings.setdefault('barcode_mismatches', 0)
```

Group names MUST be columnID / rowID (not column / row): the read processors in sequencing.py read match.group('columnID') / match.group('rowID'), so a default regex naming them column/row raised "IndexError: no such group" — the shipped default was unusable.

## get_default_generate_activation_map_settings

### lines 7504-7506

```python
settings.setdefault('smoothgrad_samples', 0)
```

Attribution methods and their analyses (spacr.attribution). The sanity check is on by default because a map that ignores the model's weights is an edge detector, not an explanation.

## get_analyze_plaque_settings

### lines 7527-7540

```python
settings.setdefault('plaque_model', 'bundled')
```

Which checkpoint segments the plaques: 'bundled' (the pre-2026 model that ships with spaCR), a model_zoo key such as 'toxoplasma_plaque_v1' (fetched from Hugging Face and checksum-verified the first time it is CHOSEN), or a path to your own.

THE DEFAULT STAYS 'bundled' ON PURPOSE, even though toxoplasma_plaque_v1 is markedly better (F1 0.856 vs 0.718 in-domain; the bundled model recalls 0.631 on the literature set, missing about a third of the plaques). Two reasons to make it a choice rather than a default: a default that downloads 1.2 GB the first time anyone runs the module is a surprise, and changing which model runs would silently change the counts in every existing pipeline that never asked for a new model. Selecting it is one setting; both of those are irreversible for someone who did not notice.

### lines 7542-7544

```python
settings.setdefault('well_detection', False)
```

False for images that each hold one plaque field (the original behaviour); True to find the wells first and analyse each separately. A path or model_zoo key selects a different detector.

### lines 7548-7549

```python
settings.setdefault('plate_format', None)
```

The ruler. Without one of these, areas stay in pixels -- which are a property of the microscope, so they cannot be pooled across scopes.

### lines 7566-7586

```python
settings.setdefault('normalize', True)
```

THE SEVEN THIS MODULE READS AND DID NOT DECLARE, added 2026-09-08.

`analyze_plaques` hands its dict to

`spacr.spacr_cellpose.identify_masks_finetune`, which reads 24 keys. Seven were declared by neither this factory nor `analyze_plaques`, and `settings['normalize']` is read unconditionally inside the per-batch loop -- so `masks=True`, THE DEFAULT, raised `KeyError: 'normalize'` on the first image. The module could not run from its own settings.

Nothing caught it because the only test driving `analyze_plaques` passes `masks=False`, which skips the mask step: the covered path was the one the user does not take.

THE VALUES ARE `get_identify_masks_finetune_default_settings`'s, unchanged -- that factory feeds the same function and is where this contract is already written down. Declared here rather than injected in `analyze_plaques` because the settings panel is built from this factory, and `normalize`, `percentiles`, `invert` and `remove_background` all change results: they should be visible and editable, not hidden defaults.

### lines 7593-7595

```python
settings.setdefault('model_name', 'cpsam')
```

Only read on the branch where `custom_model` is None, which

`analyze_plaques` never takes -- declared so the contract is complete rather than complete by luck.

## set_graph_importance_defaults

### lines 7608-7620

```python
settings.setdefault('graph_type','jitter_box')
```

A BOX WITH JITTER, NOT A BAR WITH JITTER. Instruction 139 B, asked for on 2026-08-18: "the bargraphs with jutter plot backgrounds should be boxplots with jutter".

It is a statistical correction rather than a preference, which is why the DEFAULT moves rather than the option merely existing. A bar drawn at a mean with points behind it shows ONE number and hides the shape: two groups with the same mean and completely different spreads draw the same bar. A box shows the median, the quartiles and the whiskers, so the reader sees the distribution the points already imply -- and the jitter stays, because the box summarises and the points are the evidence.

`jitter_box` already existed as an option; only the default was wrong.

## set_analyze_invasion_defaults

### lines 7661-7664

```python
_fold_renamed_settings(settings)
```

BEFORE ANY DEFAULT IS FILLED IN (364). A settings file naming a renamed key must reach the new name carrying its VALUE, and a `setdefault` that ran first would already have put the default there -- so the user's number would be silently replaced by ours.

### lines 7675-7677

```python
settings.setdefault('stain_baseline_wells', None)
```

THE STAINING CONTROLS, and this key used to be `control_wells` which Regression and sequencing also used, for the unrelated list of wells to DROP before fitting. Split on 2026-09-09 (364, 357-Q6).

## get_plot_data_from_csv_default_settings

### lines 7806-7818

```python
settings.setdefault('graph_type','jitter_box')
```

A BOX WITH JITTER, NOT A BAR WITH JITTER. Instruction 139 B, asked for on 2026-08-18: "the bargraphs with jutter plot backgrounds should be boxplots with jutter".

It is a statistical correction rather than a preference, which is why the DEFAULT moves rather than the option merely existing. A bar drawn at a mean with points behind it shows ONE number and hides the shape: two groups with the same mean and completely different spreads draw the same bar. A box shows the median, the quartiles and the whiskers, so the reader sees the distribution the points already imply -- and the jitter stays, because the box summarises and the points are the evidence.

`jitter_box` already existed as an option; only the default was wrong.

## get_automated_motility_assay_default_settings

### lines 7930-7933

```python
settings.setdefault('src', 'path')
```

array settings

`src` is the plate folder holding merged/*.npy. It used to be inherited from the mask settings this dict was merged into; the Motility Assay is now a module of its own, so it has to carry its own source folder.

### line 7944  _(unsure)_

```python
settings.setdefault('n_jobs', 8)
```

filter settings

### line 7950, trailing

```python
settings.setdefault('infection_intensity_strategy', 'xgboost')
```

'pca' | 'umap' | 'tsne' | 'histogram' | 'xgb'

### line 7951, trailing  _(unsure)_

```python
settings.setdefault('infection_intensity_mode', "relabel")
```

or 'remove'

### lines 7954-7956

```python
settings.setdefault('infection_intensity_qc_graphs', True)
```

Read by _make_intensity_motility_panel; previously undefaulted, so the standalone module had no widget for it. Exposing it lets users skip the QC plotting work on large runs.

### line 7959  _(unsure)_

```python
settings.setdefault('pixels_per_um', 1.78)
```

motility plot settings

### line 7983  _(unsure)_

```python
settings.setdefault('infection_pca_n_clusters', 2)
```

PCA / embedding-common settings

### line 8000  _(unsure)_

```python
settings.setdefault('infection_pca_tsne_search', True)
```

t-SNE

### line 8004  _(unsure)_

```python
settings.setdefault('infection_pca_tsne_perplexity', 30.0)
```

used if infection_pca_tsne_search == False

## _set_organelle_defaults

### lines 8164-8166

```python
settings.setdefault(NUMBER_OF_ORGANELLES, organelle_count(settings))
```

Read legacy intent before adding placeholder slot values. An explicit count wins; otherwise a non-empty old slot activates that slot while a genuinely empty mapping stays at zero.

### line 8169  _(unsure)_

```python
'organelle_channel': None,
```

General

### lines 8171-8173

```python
'organelle_type': DEFAULT_ORGANELLE_TYPE,
```

ONE visible choice in front of fifty-three. Defaults to 'custom', which recommends nothing: a settings file written before this existed has no opinion about it and must keep its exact meaning.

### lines 8183-8192

```python
'organelle_background': 100,
```

Preprocessing

THESE TWO BELONG HERE AND NOT IN THE GENERIC BLOCK, and the reason is worth the line: `_count_implied_by_the_slots` treats ANY non-blank `organelle_*` key as evidence that slot one is in use. Setting them unconditionally beside the cell/nucleus/pathogen defaults made every settings file infer one organelle instead of zero -- exactly the trap `get_measure_crop_settings` already warns about, "once those placeholders exist they cannot be distinguished from a legacy file that genuinely requested slot one". This function reads the count first, so a key written here cannot imply one.

### line 8201  _(unsure)_

```python
'organelle_log_min_sigma': 1,
```

Spots

### line 8211  _(unsure)_

```python
'organelle_ridge_sigmas': [1, 2, 3],
```

Network

### line 8219  _(unsure)_

```python
'organelle_unet_model_path': None,
```

U-Net

### line 8223  _(unsure)_

```python
'organelle_adaptive_block_size': 51,
```

Irregular

### line 8235  _(unsure)_

```python
'organelle_cellprob_threshold': 0.0,
```

Cellpose

### lines 8240-8259

```python
settings.update(apply_preset(settings,
```

THREE TIERS, AND THE ORDER IS THE WHOLE DESIGN:

what the USER set        wins over what the PRESET advises  wins over the bare DEFAULT

So the preset runs FIRST, against the caller's own dict, where the only keys present are the ones they chose. It fills the gaps it has an opinion about and never touches a key that is already there -- that is "preset, do not override": pick 'punctate', change organelle_method to 'adaptive', and the change sticks.

Running it after `setdefault` was the first attempt and it was wrong: the defaults had already filled organelle_method with 'otsu', the preset saw a set key, kept it, and naming a type did nothing at all. UPDATED IN PLACE, not rebound. `apply_preset` returns a NEW dict, and every caller here does `_set_organelle_defaults(settings)` without taking the return value -- so rebinding the local name silently threw forty defaults away onto a copy, and the mask panel lost every detection setting it had. Measured: 53 organelle keys became 13.

### lines 8264-8266

```python
for role in declared_organelle_roles(settings)[1:]:
```

Each secondary slot gets the same defaults and its own independent type preset. Translate only at this boundary so the preset implementation has one vocabulary and one set of tests.

## _fold_toxoplasma, 2026-09-19

```python
_fold_toxoplasma(settings)
```

`Toxoplasma` RETIRED on 2026-09-19, instruction 364, at the maintainer's decision ("Retire both", with `barcodes`). This replaces the two entries under `get_perform_regression_default_settings` anchored at `if 'toxo' in settings:` and at the `settings.setdefault(` that derived `annotation_source` from the switch. Their reasons still hold and moved into `_fold_toxoplasma`: an old file's switch is MIGRATED, never dropped, and false keeps meaning no annotation. What changed is that the switch no longer survives the migration, so the panel and the run have one control for one question.

THE ONE BEHAVIOUR THAT CHANGES, found by reading the old code rather than the survey. Before the retirement a blank `annotation_source` with no switch in the file meant the bundled tables, because the factory defaulted the switch to true and `ml._annotation_source` fell back to it. Now a blank field is the only way to say no annotation, so that file annotates nothing. A file saved from the panel carries both keys and is unaffected: a true switch beside a blank field still gives `'toxoplasma'`. The upside is the reason for it: the switch was hidden on the panel, so a panel user had no way to turn annotation off at all.

```python
return value.strip().lower() not in (
```

A STRING 'False' IS OFF. A loader that does not type its values hands this the text, and `bool('False')` is True. The code this replaces used `bool()` on the raw value, both in the regression factory and in `ml._toxoplasma_is_on`, and `toxo` was never in `expected_types`, so no type table would have caught it on the way in. The CLI's own CSV reader does type `False`; the rule here does not depend on which reader ran.

## RENAMED_SETTINGS, 2026-09-19

```python
"img_size": "crop_size",
```

RENAMED at the maintainer's decision, 2026-09-19. The proposal was `image_size`, and it could not be built as approved: `image_size` is already a live setting meaning the model's input crop (default 224), while `img_size` is how many pixels each cell is drawn at (default 200) on the Annotate screen and the Cells tab. `crop_size` is typed in `expected_types` so the fold has a live terminus, and `set_annotate_default_settings` folds before filling its defaults. The Cells tab's saved picture settings are moved by `picture_settings.drop_retired`, which asks `surviving_setting_name` rather than keeping a second table.

```python
"%s=%r is applied as %s. The setting was renamed and this "
```

THE LINE USED TO SAY "until now the value was ignored and the default used". That was true of the renames 15fa72737 repaired on 2026-09-12 and false of every rename made since, `img_size` first: its value always worked. The line now says only what is true of all of them.

