# Notes from `spacr/qt/screens/settings_model.py`

Prose lifted out of `spacr/qt/screens/settings_model.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (112 entries)
- [resolve_default_settings](#resolve_default_settings) (7 entries)
- [keys_hidden_by_their_object](#keys_hidden_by_their_object) (1 entry)
- [_categories_from_spec](#_categories_from_spec) (2 entries)
- [SettingsSection](#settingssection) (1 entry)
- [SettingsSection.__new__](#settingssection__new__) (1 entry)
- [_nest_sections](#_nest_sections) (1 entry)
- [needs_curated_layout](#needs_curated_layout) (1 entry)
- [categories_for_app](#categories_for_app) (17 entries)
- [category_tooltip](#category_tooltip) (1 entry)
- [_absorb_registered_api_modules](#_absorb_registered_api_modules) (1 entry)
- [_mapped_api_target](#_mapped_api_target) (1 entry)
- [api_docs_url](#api_docs_url) (6 entries)
- [_auto_or_number_box](#_auto_or_number_box) (1 entry)
- [_float_domain](#_float_domain) (2 entries)
- [_type_hint](#_type_hint) (1 entry)
- [language_resolved_once](#language_resolved_once) (1 entry)
- [_language_code](#_language_code) (1 entry)
- [_translated_type_hint](#_translated_type_hint) (1 entry)
- [_translated_setting_name](#_translated_setting_name) (1 entry)
- [_ApiTooltipFilter.eventFilter](#_apitooltipfiltereventfilter) (2 entries)
- [mixed_cost_note](#mixed_cost_note) (1 entry)
- [_well_keys](#_well_keys) (1 entry)
- [regression_design_scan](#regression_design_scan) (2 entries)
- [regression_model_explainer_html](#regression_model_explainer_html) (3 entries)
- [regression_model_explainer](#regression_model_explainer) (7 entries)
- [_apply_greyed_note](#_apply_greyed_note) (3 entries)
- [_note_on_label](#_note_on_label) (1 entry)
- [_clear_greyed_note](#_clear_greyed_note) (1 entry)
- [attach_api_tooltip](#attach_api_tooltip) (3 entries)
- [install_api_tooltips](#install_api_tooltips) (6 entries)
- [_unwrap_setting_label](#_unwrap_setting_label) (1 entry)
- [_setting_label_for_field](#_setting_label_for_field) (2 entries)
- [_CsvColumnField.__init__](#_csvcolumnfield__init__) (1 entry)
- [_CsvColumnField.pick](#_csvcolumnfieldpick) (1 entry)
- [_RegressionBackendField.__init__](#_regressionbackendfield__init__) (3 entries)
- [_RegressionBackendField._normalise_type](#_regressionbackendfield_normalise_type) (1 entry)
- [_RegressionBackendField.refresh](#_regressionbackendfieldrefresh) (3 entries)
- [_RegressionBackendField.availability_entries](#_regressionbackendfieldavailability_entries) (1 entry)
- [_RegressionBackendField.eventFilter](#_regressionbackendfieldeventfilter) (2 entries)
- [_RegressionBackendField._hover_popup_row](#_regressionbackendfield_hover_popup_row) (1 entry)
- [_Chip.__init__](#_chip__init__) (1 entry)
- [_ChipStrip.__init__](#_chipstrip__init__) (1 entry)
- [_ChipStrip._add_chip](#_chipstrip_add_chip) (1 entry)
- [_AlphabetSelect.__init__](#_alphabetselect__init__) (1 entry)
- [_AlphabetSelect._as_members](#_alphabetselect_as_members) (1 entry)
- [_ListEditor.__init__](#_listeditor__init__) (1 entry)
- [_ListEditor._rebuild](#_listeditor_rebuild) (1 entry)
- [_ListEditor._drop_strip](#_listeditor_drop_strip) (1 entry)
- [_ListEditor._on_footer](#_listeditor_on_footer) (1 entry)
- [_ListEditor._placeholder](#_listeditor_placeholder) (1 entry)
- [_ListEditor._as_sequence](#_listeditor_as_sequence) (1 entry)
- [list_shape_for](#list_shape_for) (1 entry)
- [SettingsWidgets.__init__](#settingswidgets__init__) (6 entries)
- [SettingsWidgets._build_sections](#settingswidgets_build_sections) (15 entries)
- [SettingsWidgets._keys_of_objects_the_run_has_no_channel_for](#settingswidgets_keys_of_objects_the_run_has_no_channel_for) (2 entries)
- [SettingsWidgets._organelle_keys_beyond](#settingswidgets_organelle_keys_beyond) (3 entries)
- [SettingsWidgets.search_text_for](#settingswidgetssearch_text_for) (1 entry)
- [SettingsWidgets.modified_keys](#settingswidgetsmodified_keys) (1 entry)
- [SettingsWidgets._widget_for](#settingswidgets_widget_for) (24 entries)
- [SettingsWidgets._coerce_to_expected_type](#settingswidgets_coerce_to_expected_type) (1 entry)
- [SettingsWidgets.collect](#settingswidgetscollect) (1 entry)
- [SettingsWidgets._refresh_analysis_unit_lock](#settingswidgets_refresh_analysis_unit_lock) (5 entries)
- [SettingsWidgets._refresh_umap_reducer_enablement](#settingswidgets_refresh_umap_reducer_enablement) (1 entry)
- [SettingsWidgets._refresh_classifier_family_enablement](#settingswidgets_refresh_classifier_family_enablement) (1 entry)
- [SettingsWidgets.refresh_training_basis_enablement](#settingswidgetsrefresh_training_basis_enablement) (1 entry)
- [SettingsWidgets._plate_context](#settingswidgets_plate_context) (2 entries)
- [SettingsWidgets._refresh_setting_dependencies](#settingswidgets_refresh_setting_dependencies) (2 entries)
- [SettingsWidgets._show_the_value_it_will_have](#settingswidgets_show_the_value_it_will_have) (1 entry)
- [SettingsWidgets._object_visibility_keys](#settingswidgets_object_visibility_keys) (2 entries)
- [SettingsWidgets.refresh_object_visibility](#settingswidgetsrefresh_object_visibility) (3 entries)
- [SettingsWidgets._slot_headings](#settingswidgets_slot_headings) (2 entries)
- [SettingsWidgets._hide_the_headings_of_slots_the_run_lacks](#settingswidgets_hide_the_headings_of_slots_the_run_lacks) (3 entries)
- [SettingsWidgets._shown_against_the_rule](#settingswidgets_shown_against_the_rule) (1 entry)
- [SettingsWidgets._set_row_visible](#settingswidgets_set_row_visible) (3 entries)
- [SettingsWidgets._connect_object_visibility_signals](#settingswidgets_connect_object_visibility_signals) (1 entry)
- [SettingsWidgets.apply_organelle_presets_from_mapping](#settingswidgetsapply_organelle_presets_from_mapping) (1 entry)
- [SettingsWidgets._read_widget](#settingswidgets_read_widget) (1 entry)
- [_sibling_label_for](#_sibling_label_for) (3 entries)
- [_sibling_label_for._named](#_sibling_label_for_named) (4 entries)
- [retarget_field_tooltips](#retarget_field_tooltips) (5 entries)

## Module level

### lines 62-65

```python
from ...organelle_types import (ALL_ORGANELLE_ROLES,
```

EVERY SLOT A FILE MAY CARRY, not the four the schema segments today. A layout that lists all of them costs nothing -- build_sections drops any key the module's settings dict does not hold -- and a layout that lists four puts slot five in the "Additional Settings" bucket nobody chose.

### lines 70-73

```python
from ...regression_spec import NO_P_VALUE_TYPES
```

Pure data, and it imports nothing -- that is the whole point of the module (see its docstring). The explainer box below reads it so that a backend joining the no-p-value set changes what the box says about correction without a second edit here.

### lines 75-76

```python
from ...schema import KEY_SEPARATOR
```

The one separator a spaCR key is built from, so the design scan below splits gRNA names the way the pipeline does rather than on a literal '_'.

### lines 301-302  _(unsure)_

```python
_APP_HIDDEN_KEYS: Dict[str, set] = {
```

Per-app category suppression. Keys not in a shown category fall into the trailing "Other" section, so the setting stays reachable — only the tab goes.

### lines 315-325

```python
"mask": {"pathogen_model"},
```

`pathogen_model` named the same thing as `pathogen_model_name` `object.py` reads the first as an override of the second -- and two controls for one value is how a user sets one and wonders why the other wins. Retired from the panel 2026-09-01 at the maintainer's request; `pathogen_model_name` is the one control, and it loads a checkpoint path exactly as the override did, because both go through `_resolve_cellpose_pretrained`.

It stays in the settings dict, and object.py still reads it, so a settings CSV written before this keeps segmenting with the model it names instead of silently falling back to cpsam.

### lines 327-330

```python
"timelapse": {"timelapse"},
```

This module IS the timelapse one. A user who turned this off would be left looking at a screen whose every remaining control is about a time dimension it had just been told to ignore -- and Mask Generation is right there for that. `resolve_default_settings` forces it True.

### lines 332-350

```python
"classify": {
```

`png_type` was one of two names for a path filter, and the one that pretended to name a file type. The Classify overhaul replaced it with `path_string` (the substring) and `file_type` (an actual extension).

It stays in the settings dict because `spacr.crop_source` still reads it as a fallback, so a settings CSV written before the split keeps working. It is not OFFERED, because offering both halves of a superseded pair is how a user sets one and wonders why the other wins. AND THE SAME RULE FOR THE 230 SUPERSESSIONS. `crop_source`, `file_metadata` and `file_type` are what `image_source` and `load_path_regex` replaced, and `coordinate_columns` is DERIVED from `object_array` -- so none of the four is a control any more.

They stay in the settings dict because the old readers still consult them as a fallback, which is what keeps a settings CSV written before the rename working. They are not OFFERED, for the reason `png_type` is not: offering both halves of a superseded pair is how a user sets one and wonders why the other wins.

### lines 355-364

```python
"class_folder_names",
```

AND THE FOLDER NAMES (instruction 229, reported again 2026-08-21: "i asked you to remove class folder names and just use the classes given in the classes setting"). The first pass only made the class field OUTRANK it, which left a control on screen that could disagree with the classes above it and lose -- a control the user can change that changes nothing.

It stays in the settings dict because dataset generation WRITES it: it records what actually went to disk, which is a different fact from what the user asked for.

### lines 371-380

```python
"class_folder_names",
```

AND THE FOLDER NAMES (instruction 229, reported again 2026-08-21: "i asked you to remove class folder names and just use the classes given in the classes setting"). The first pass only made the class field OUTRANK it, which left a control on screen that could disagree with the classes above it and lose -- a control the user can change that changes nothing.

It stays in the settings dict because dataset generation WRITES it: it records what actually went to disk, which is a different fact from what the user asked for.

### lines 383-387

```python
"umap": {"gpu", "crop_source"},
```

One action-strip GPU toggle drives both the main reducer and the search. The setting remains in _defaults and therefore in collect(); only the duplicate form control is hidden. `crop_source` reaches UMAP through the shared picture settings and is superseded there by `image_source` for the same reason as above.

### lines 389-414

```python
"regression": {
```

WHAT "REGRESSION PLOTS" AND "RUNTIME & RELIABILITY" HELD, per instruction 135. The sections are deleted from the layout above; these keys keep their values and reach the run exactly as before, they are simply not asked about.

Hidden and not dropped, deliberately, and each for its own reason:

regression_qc          `parameter_sweep` sets it False so a hundred-trial sweep does not pay ~5.8 s and 19 figures per trial. Drop the key and the sweep has nothing to set. One analysis still gets the suite, because the default is True. guide_permutation_plot hard True: the permutation run's only picture. log_x, log_y           hard False. x_lim, y_lims,         set ON the plot, where the axes being changed split_axis_lims        are visible; a number typed before the figure exists is a guess. strict_errors,         how the APPLICATION behaves on a failure, not max_failure_rate,      how this regression is fitted. The same answer verbose, random_seed,  on every module, so it is one answer in on_error*              Preferences rather than eleven in the modules.

Keys this module does not declare (`on_error`, `random_seed`, ...) are named anyway: hiding a key that is not there costs nothing, and the day a shared runtime default reaches this module it must not appear on the panel the instruction just cleared.

### lines 416-419

```python
"regression_panel_manifest",
```

Programmatic export contract for ``perform_regression``.  A mapping or manifest path belongs in a script/CLI call, not in a scalar GUI field; keeping it in the defaults makes runs reproducible, while hiding it here prevents it falling into ``Additional Settings``.

### lines 425-430

```python
"analysis_excluded_wells",
```

Regression derives this aggregate from the positive, negative, and mixed control-well settings, so it is not an independent GUI choice. The invasion assay keeps its own control under a name of its own since the 2026-09-09 split: `stain_baseline_wells` identifies wells without pre-permeabilisation stain, and this one is the list the analysis drops.

### lines 432-437

```python
"Toxoplasma",
```

SUPERSEDED BY `annotation_source`, and hidden here rather than in a second "regression" entry further up this dict -- which is where it was, and which a later key of the same name silently replaced. A dict literal keeps the last value, so `Toxoplasma` was declared hidden and then offered anyway, ungrouped, in the bucket the layouts exist to keep empty.

### lines 439-443

```python
"cell_area_outlier_mads", "nucleus_area_outlier_mads",
```

THE FOUR OBJECT-OUTLIER FILTERS. Each excludes objects by a robust z-score over a measurement, and this module joins the measurements to the scores AFTER the fit -- so at the moment they would run there is no column to take a deviation over. They are still read from a settings file; they are not offered.

### lines 451-455

```python
"mask": {"Timelapse", "Motility (beta)", "Motility Advanced (beta)"},
```

Mask no longer owns tracking or the motility assay — those are the 'timelapse' and 'motility' modules. resolve_default_settings already drops the keys so nothing spills into "Other"; this entry is the declaration of intent and keeps the tabs gone even if a future default re-introduces one of the keys.

### lines 457-458

```python
"timelapse": {"Motility (beta)", "Motility Advanced (beta)"},
```

The Timelapse module tracks objects; the motility assay is its own module and its ~50 knobs would swamp the tracking settings.

### lines 462-485

```python
OBJECT_SWITCH_SUFFIXES: Tuple[str, ...] = ("channel", "mask_dim")
```

A setting is visible when its object is in the run

WHY THIS IS A THIRD MECHANISM AND NOT ONE OF THE TWO ABOVE.

``spacr.settings.setting_dependencies`` GREYS a control and writes the reason beside it. That is right for a setting the run is about to decide for itself -- one row among a handful, where the note is the point. It is wrong here: an object a run does not segment takes forty rows with it, and forty greyed rows are not an explanation, they are the wall this exists to remove. ``_APP_HIDDEN_KEYS`` builds no widget at all. It is decided once, per MODULE, before any value exists, and it is not reversible: with no widget the value falls back to ``_defaults``, so everything the user typed into a row is gone the moment the row is hidden. "Changing a channel back must bring the old answers back with it" is exactly the thing that mechanism cannot do.

So the widget is built and kept in ``_widgets``, and its ROW is hidden. HIDDEN, NOT DELETED: ``collect()`` walks ``_widgets``, so a hidden setting is still read from its own widget, still carries what the user last typed into it, and is still written to the settings file. A settings CSV cannot lose a key because the panel was not showing it when Save was pressed.

### lines 540-542

```python
"log_min_sigma", "log_max_sigma", "log_num_sigma", "log_threshold",
```

Ring accepts 'log' and reads the LoG sigmas when it is chosen. It does NOT read the DoG pair: its 'dog' path band-passes with `ring_sigma_inner`/`_outer` instead. See `_segment_ring`.

### lines 847-849

```python
_APP_COMBO_OPTIONS: Dict[str, Dict[str, List[Any]]] = {
```

Options that are enumerations for one module but not necessarily for every setting with the same generic key.  Keeping these app-scoped avoids turning unrelated ``mode`` fields into sequencing controls.

### line 853  _(unsure)_

```python
"metric": ["euclidean"],
```

Replaced with the installed UMAP metric inventory in _widget_for.

### lines 861-863

```python
"crop_source": _CROP_SOURCE_OPTIONS,
```

'auto' is retired FROM THE PANEL and not from the code

(instruction 171): it answers "what is available here", which is not an answer to somebody asked which mode they want.

### lines 869-872

```python
"crop_source": _CROP_SOURCE_OPTIONS,
```

The choice the annotation app has always had a SETTING for and never offered -- it shipped 'auto' and took the PNG folder whenever one existed. "in the annotation app how do i choose to stream images from database or dataset" (2026-08-19).

### lines 884-886

```python
"regression_type": ["ols"],
```

Filled from the modules that own each inventory in _widget_for, so a family added to spacr.ml or a correction added to spacr.multiple_testing appears here without a second edit.

### lines 889-893

```python
"inference": [
```

(value, label) pairs for the same reason `analysis_mode` has them below: this is the choice that decides whether the family box or the permutation section is the one that does anything, and 'auto' / 'parametric' / 'nonparametric' name a statistical stance rather than the consequence a reader is choosing between.

### lines 903-912

```python
"analysis_mode": [
```

Instruction 134, asked for on 2026-08-17: "analasys mode should be a dropdown". Two valid values and it was a FREE-TEXT box, so a typo in it survived until the run had read the whole database. `_resolve_regression_analysis_choices` is what maps `inference` onto this, and it accepts exactly these two.

(value, label) PAIRS: the key is called 'guide_permutation' and the dropdown says what that IS, the same way 132's model box explains what it fits. The stored values are unchanged, so every settings file already written goes on meaning what it meant.

### lines 921-922

```python
"agg_type": ["mean", "median", "quantile", None],
```

Exactly the branches process_scores implements; anything else reaches the pipeline and is silently ignored rather than applied.

### lines 927-930

```python
"p_threshold_kind": ["adjusted", "raw"],
```

WHICH P THE SIGNIFICANCE LINE IS DRAWN ON. Two values and no third reading, so it is a closed alphabet rather than a box a user can type "adj" into. "adjusted" leads because it is the only one of the two that is evidence with hundreds of guides in the family.

### lines 938-952

```python
"classifier_family": [
```

A closed alphabet: there are two families and a typo in a free-text box would raise ClassifierFamilyError at run time, after the user had walked away.

(value, label) PAIRS, asked for on 2026-09-02: "in classify in classifier family spell out computer vision and machine learning and change machine learning to Tabular Machine Learning and cv to Computer vision (Torch)". "cv" and "ml" are abbreviations of abbreviations -- a dropdown reading `cv` / `ml` asks the user to already know which of two whole disciplines this module means, and the distinction that matters is what each one READS: one is fed object crops through Torch, the other rows of measured features. The stored values are unchanged, so every settings file already written goes on meaning what it meant, and `spacr.classify` dispatches on the same two strings.

### lines 957-959

```python
"batch_correction": _BATCH_CORRECTION_OPTIONS,
```

set_default_classify gives this screen all eight batch_* keys, so it corrects batches exactly like the other three — but it was the one app that listed no alphabet for them.

### lines 1014-1017

```python
"count_grna_column": _CsvColumnSource(("count",), "count column"),
```

The count table's own header names, which were HARD-CODED and had no setting at all until instruction 135. They fail the same way `dependent_variable` did -- inside the merge, naming a column the file has not got -- and they earn the same button.

### lines 1034-1037

```python
_UMAP_REDUCER_SETTINGS: Dict[str, set] = {
```

Settings read by exactly one Image UMAP reducer.  The controls remain in the form so switching methods preserves their values; only the inactive families are greyed.  Shared controls (random_seed and, where applicable, metric) are handled separately in _refresh_umap_reducer_enablement.

### lines 1082-1083  _(unsure)_

```python
_APP_TOOLTIP_OVERRIDES = {
```

Shared keys can have different meanings in individual modules. These overrides are applied after the global tooltip registry is loaded.

### lines 1100-1103

```python
_APP_CATEGORY_SPECS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
```

App-specific category layouts. ``@Name`` expands the corresponding legacy category; plain entries are individual setting keys. The backend settings dictionaries remain unchanged — this controls only the order and grouping in Qt, just like the Classify (CV) regroup below.

### lines 1185-1188

```python
("Labels & Classes", (
```

Category names shared with Classify (CV) wherever the two do the same job -- "Labels & Classes", "Classifier & Validation", "Runtime & Reliability". The CV layout is built inline further down; the names are what has to match, not the mechanism.

### line 1191  _(unsure)_

```python
"location_column", "positive_control_id", "negative_control_id",
```

metadata basis

### line 1193  _(unsure)_

```python
"annotation_column",
```

annotation basis

### line 1195  _(unsure)_

```python
)),
```

measurement basis

### lines 1215-1218

```python
("Plots & Heatmaps", (
```

("Output & Database", ("save_to_db",)) went with the setting on 2026-09-09 (357-Q4): `save_to_db` was read by nothing but its own defaults setter, and a heading whose only row is gone is a heading with nothing under it.

### lines 1227-1238

```python
NUMBER_OF_ORGANELLES,
```

HOW MANY ORGANELLE SLOTS, immediately before the switches of the slots it governs -- the relationship Measure states beside its mask dimensions, said here beside the channels.

AND IN THE FIRST GROUP, WHICH IS WHAT MAKES IT REACHABLE. The settings strip opens on Essentials, and essentials are the module's first group plus `_APP_ESSENTIAL_EXTRAS`. The count leads the shared "Organelle" category as well (`settings.organelle_basic_settings`), and filed only there it was in neither list: a panel drawing twenty-six organelle channel boxes offered no way to say how many there were until the user found the All settings switch.

### lines 1264-1266

```python
("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
```

The advanced half is its own heading rather than being folded into the one above, so the six settings a biologist recognises are not buried under forty-eight detection parameters. Instruction 72.

### lines 1268-1276

```python
("Image Preprocessing (per object)",
```

Instruction 73: the families that are one decision applied to several objects, grouped by what they do rather than by which object they do it to. All three nest under "Advanced settings", which is derived from the group they reference rather than restated here -- see `_shared_category_parents`.

"(per object)", NOT "Image Preprocessing": the heading above is the whole-image one, and the category-help table is keyed on the heading text, so two headings spelled alike would share one blurb.

### lines 1303-1305

```python
"number_of_organelles", "organelle_mask_dim",
```

HOW MANY SLOTS THERE ARE, before the slots themselves. It belongs to no slot, so nothing hides it, and unclaimed it landed in the bucket the layouts exist to keep empty.

### lines 1308-1314

```python
"organelle_type",
```

WHAT KIND OF ORGANELLE each slot holds. Measure needs it for the same reason mask does, and for one more: it decides whether "how many, and how spread out" is the phenotype or a segmentation artefact, which is what a measure run says out loud about its own organelle numbers. Beside the mask dimension because the two answer one question -- which plane, and what is on it.

### lines 1320-1330

```python
("Illumination Correction", (
```

Illumination correction sits between the mapping and the features because that is where it runs: it rewrites the pixels every intensity feature below is then computed from. The Illumination screen spreads these across four tabs -- correction model, field sampling, QC, failure handling -- which is the right shape when estimating a field is the whole job. Inside Measure it is one decision with its details attached, so it is one section.

`src` and `channels`, the other two keys the estimate reads, are not repeated here: Measure already offers them above, and the estimate deliberately reads the same fields the run measures.

### lines 1340-1345

```python
"spatial_measurements",
```

Instruction 71's two opt-in measurements. They were added to the measure defaults and to the shared "Measurements" category but NOT to this literal list, so the measure panel dropped them into the trailing "Additional Settings" bucket -- which is not a heading anyone chose, it is the absence of one. They extend calculate_correlation, so they sit beside it.

### lines 1347-1349

```python
"spatial_neighbor_radius",
```

The radius the neighbourhood is counted in, immediately after the switch that turns it on: it is baked into the column name, so a screen has to pick one value and keep it.

### lines 1351-1357

```python
"bystander_measurements", "bystander_reach_in_diameters",
```

Instruction 388's bystander split, and the note above applies to it word for word -- it was added to the measure defaults and to the shared "Measurements" category and landed in "Additional Settings" until it was listed HERE too. Beside the spatial pair because it answers the same kind of question about the same neighbourhood: that pair counts the neighbours, this one asks whether any of them is infected.

### lines 1361-1364

```python
"object_distances", "object_distance_maxima",
```

The spatial-distance block: how far every object is from every other, and from the intensity maxima inside it. Filed beside `radial_dist` because they answer the same kind of question, one object pair at a time instead of one radius.

### lines 1367-1370

```python
"summarize_organelles_by",
```

Not a segmentation control -- it decides which organelle summary TABLES a measure run writes, so it belongs with the other what-gets-measured settings rather than under the mask pipeline's Organelle Segmentation heading.

### lines 1400-1411

```python
NUMBER_OF_ORGANELLES,
```

HOW MANY ORGANELLE SLOTS, immediately before the switches of the slots it governs -- the relationship Measure states beside its mask dimensions, said here beside the channels.

AND IN THE FIRST GROUP, WHICH IS WHAT MAKES IT REACHABLE. The settings strip opens on Essentials, and essentials are the module's first group plus `_APP_ESSENTIAL_EXTRAS`. The count leads the shared "Organelle" category as well (`settings.organelle_basic_settings`), and filed only there it was in neither list: a panel drawing twenty-six organelle channel boxes offered no way to say how many there were until the user found the All settings switch.

### lines 1418-1424

```python
("Acquisition & Axes", (
```

`timelapse` is not offered here. This module IS the timelapse one -- turning it off would leave a screen whose every remaining control is about a time dimension it had just been told to ignore, and there is no reason a user would want that rather than opening Mask Generation. It stays in the settings dict at True (see `_ALWAYS_ON`), so a run gets what it expects and a mask-settings CSV from before the split still round-trips.

### lines 1446-1448

```python
("Organelle Segmentation (advanced)", ("@Organelle advanced",)),
```

The advanced half is its own heading rather than being folded into the one above, so the six settings a biologist recognises are not buried under forty-eight detection parameters. Instruction 72.

### lines 1450-1458

```python
("Image Preprocessing (per object)",
```

Instruction 73: the families that are one decision applied to several objects, grouped by what they do rather than by which object they do it to. All three nest under "Advanced settings", which is derived from the group they reference rather than restated here -- see `_shared_category_parents`.

"(per object)", NOT "Image Preprocessing": the heading above is the whole-image one, and the category-help table is keyed on the heading text, so two headings spelled alike would share one blurb.

### lines 1544-1551

```python
("Input Tables", ("paired_data", "metadata_files",
```

`count_grna_column` and `count_value_column` are the count CSV's own header names, which were HARD-CODED until instruction 135. They belong beside the table they name: a user who has to say what their count file calls its guide column is looking at the count file, not at the model. ``src`` is the optional result root. It remains beside the input tables because its automatic value is derived from the first count table; see ``ml.resolve_regression_src``.

### lines 1556-1574

```python
("Controls & Filters", (
```

CONTROLS AND FILTERS ARE ONE QUESTION: which rows reach the model. Asked for on 2026-08-17 -- "merge quality & filters in here. change the settings categoty to Controlls & Filters". They were two sections with the response, the estimator and the hit-calling rules between them, so it was not obvious that seven separate settings each drop data. THE THREE CONTROL BLOCKS AND THE EXCLUSION LIVE HERE, not in the trailing "additional settings" they fell into for want of being named (2026-08-21). A settings key that no panel section claims lands in the catch-all, which is where a reader looks last.

ORDER IS THE ASK: they follow `negative_control`, because they are about the same thing -- which wells and which guides are controls and the eye should not have to travel to collect them.

`control_wells` IS GONE FROM THIS PANEL. It said "these wells are controls" without saying WHICH control, and the three settings below say that. Still read by the invasion-assay panel, which has its own meaning for it.

### lines 1581-1583

```python
"calibrate_fraction_threshold",
```

DIRECTLY UNDER THE NUMBER IT REPLACES. It says "measure this from the control wells instead", so it is only readable beside the number it is an alternative to.

### lines 1594-1598

```python
("Response", (
```

WHAT IS BEING MODELLED, before HOW. The response was previously interleaved with the estimator settings under "Model & Covariates", so the two questions a user actually asks in order -- what am I measuring, and how should it be tested -- were answered in one twelve-row block.

### lines 1600-1602

```python
"dependent_variable", "invert_dependent_variable",
```

`score_column` retired with instruction 135 A: it named the same measurement as `dependent_variable` and only offered a way to disagree with it.

### lines 1606-1607

```python
("Model & Inference", (
```

`inference` leads because it decides whether "Estimator Tuning" or "Permutation Test" below is the section that does anything.

### lines 1609-1624

```python
"inference", "analysis_mode", "regression_type",
```

`level` was in no section at all, so it fell into "Additional Settings" -- the bucket this layout exists to keep empty. Asked for on 2026-08-17: "level should be in model and inference not additional settings". It is not a plate-layout setting: it decides WHICH FITS RUN, and it is greyed out under regression_type='mixed', which nests guides in genes and therefore fits both levels at once. A control whose enabled state is decided by `regression_type` belongs beside it, not three sections away. WHICH MODEL, THEN WHO FITS IT, THEN AT WHICH LEVELS. Asked for on 2026-08-18: "regression backend should be in Model and inference right after regression type". `regression_type` says WHAT is fitted and `regression_backend` says WHO fits it -- the same mixed model through statsmodels or through torch on the GPU should give the same answer and not the same runtime -- so the two belong adjacent, and `level` follows them.

### lines 1627-1631

```python
"intercept", "intercept_value",
```

WHERE THE FITTED LINE IS ANCHORED. Still part of WHAT is fitted rather than which terms are in it, so it reads with the four above: `intercept` chooses fitted, zero, control or value, and `intercept_value` is the number the last of those pins it at -- greyed for the other three.

### lines 1633-1636

```python
"model_plate_position", "random_row_column_effects",
```

`model_plate_position` decides whether rowID and columnID are in the model at all; `random_row_column_effects` then decides fixed vs random for terms that ARE in. Adjacent because setting one without seeing the other is how they end up contradicting.

### lines 1638-1649

```python
"multiple_testing_method", "fdr_alpha", "p_threshold_alpha",
```

SIGNIFICANCE MERGED IN, asked for on 2026-08-17: "significance nad hit calling is good but merge all of these settings into Model and inference". They are not a separate question -- which correction, at what level, above which effect size IS how the model's output is turned into a claim, and a user reading the model section had to scroll past three others to find out. `p_threshold_alpha` and `p_threshold_kind` are the line the plot draws significance at. The plot already had a raw/adjusted choice on its right-click menu and the RUN had no say in it, so the exported hit list and the picture could disagree about what "significant" meant. They sit with `fdr_alpha` because the three of them are one question: what counts as a hit.

### lines 1653-1660

```python
"annotation_source",
```

THE FIELD, NOT THE BOOLEAN. `annotation_source` supersedes

`Toxoplasma`: empty or 'toxoplasma' is the bundled tables exactly as the True case was, and any organism name, taxon id or accession is a UniProt lookup. The boolean stays in the settings dict, because every CSV in existence carries it and `_annotation_source` still reads it, but offering both halves of a superseded pair is how a user sets one and wonders why the other wins.

### lines 1663-1666

```python
("Estimator Tuning", (
```

The estimator-specific knobs, added by the robust and regularised fits after this layout was first written. They landed in "Additional Settings" -- the bucket a layout exists to keep empty because only the shared estimator settings above were named.

### lines 1668-1672

```python
"cov_type",
```

`cov_type` moved here from Model & Inference on 2026-08-17

"mooveCov type here". It is estimator-specific in exactly the way everything else in this section is: the penalised, robust and quantile fits have no such estimator and REFUSE it rather than quietly reporting ordinary errors under a robust label.

### lines 1678-1681

```python
"group_lasso_lambda", "rra_alpha", "rra_permutations",
```

One knob per family, filed with the rest of them:

`group_lasso_lambda` is the group lasso's penalty, and `rra_alpha`/`rra_permutations` are robust rank aggregation's cutoff and null size.

### lines 1684-1688

```python
("Permutation Test", (
```

The permutation test's own settings, previously split across three sections: its block and nuisance columns sat under the model, its permutation count and seed under estimator tuning, and its support thresholds under hit calling. Nothing here is read unless inference resolves to the nonparametric test.

### lines 1690-1694

```python
"grna_statistic",
```

FIRST: it says WHAT is measured, and everything after it says how the null is built and who is eligible -- answers to a question this setting asks. Absent from this list it fell to the trailing "Additional Settings", which is where a reader looks last.

### lines 1701-1715

```python
),
```

"REGRESSION PLOTS" AND "RUNTIME & RELIABILITY" ARE GONE, asked for on 2026-08-17: "Regression plot can be removed" and "Runtime and reliability should be removed and go to prefgerences/general".

Neither was a question about the regression. The plot section asked a user to decide the axis scaling of a figure they had not seen yet `x_lim` and `y_lims` are set on the plot now -- and the runtime section asked how the whole application handles a failure, which is the same answer for every module and belongs in Preferences.

The keys they held are not dropped, they are HIDDEN

(`_APP_HIDDEN_KEYS`): dropping `regression_qc` would take `parameter_sweep`'s `settings.setdefault("regression_qc", False)` with it, and a hundred-trial sweep would pay ~5.8 s and ~19 figures per trial for diagnostics nobody opens.

### lines 1740-1745

```python
("Channel Mapping", (
```

THE THREE `*_mask_dim` ROWS ARE GONE, with the factory keys behind them. Instruction 364: `analyze_recruitment` never read them, so half this section did nothing while its hint warned that "a wrong index here measures the wrong compartment without complaining" and their tooltips described `measure_crop`, which is where those keys are actually live.

### lines 1784-1792

```python
"cellpose_masks": (
```

the three Cellpose-facing modules

All three used to render the shared "Cellpose" category as one drop of ten to thirteen knobs. They are not one decision: the model you run, the thresholds that decide how much it finds, the geometry it sees and the background correction applied before it are four separate questions, asked at four different times. The groups below are the same four in all three modules so that moving between them is not a relearning exercise.

### lines 1851-1855

```python
"barcode_mismatches",
```

How far a read may be from a listed barcode and still be called as it -- a parsing tolerance, filed with the rest of the parse. Left out of this layout it fell into "Additional Settings", which is not a heading anyone chose; it is the absence of one.

### lines 1894-1898

```python
"power": (
```

Power / Design draws its own screen, so these groups are never a settings form. They are still the layout of record: the settings diff, the run journal and `utils.pretty_print_settings` all group by category, and fifteen keys under one "Power analysis" heading make a design change unreadable in all three.

### lines 2709-2730

```python
CATEGORY_TOOLTIPS: Dict[str, str] = {
```

Category help — one blurb per settings CATEGORY

A category is a collapsible header in a module's settings panel; the map above decides which keys land under which header. These are the blurbs the panel shows for the header itself, keyed by the title uppercased and stripped, because that is what a rendered ``Section`` has in hand.

They are deliberately NOT restatements of the heading. Someone reading "Image Preprocessing" already knows the words; what they cannot tell is what the group decides and whether today's problem lives inside it. Each entry therefore says what the settings determine and when you would open them.

``CATEGORY_TOOLTIPS_BY_APP`` overrides this table for the handful of headings that genuinely mean different things per module: "Cellpose" is a training schedule under Train Cellpose and a set of inference thresholds under Cellpose Masks, and "Runtime & Reliability" carries Timelapse's stage toggles but only ``n_jobs`` under Motility.

``app_screen`` re-exports this as ``SECTION_HINTS`` for the tests and integrations that already read it by that name.

### lines 2732-2736

```python
"OPS INPUT":
```

optical pooled screening, folded onto Align & Stitch

Nine headings, and each needs an entry here or the panel draws the generic fallback: a heading whose tooltip says nothing about the settings under it is worse than no tooltip, because the reader has already spent the hover.

### lines 2828-2831

```python
"IMAGE PREPROCESSING (PER OBJECT)":
```

PER OBJECT, and the parenthetical is load-bearing: "IMAGE

PREPROCESSING" is already this table's key for the whole-image step mask and timelapse render, and a second heading spelled the same way would silently serve that blurb instead of this one.

### lines 2839-2842

```python
"OBJECT FILTRATION (ALL OBJECTS)":
```

The workflow-ordered layouts render these under a longer title, and the tooltip table is keyed on the heading's EXACT text -- which is the trap the "Computer Vision — " prefix fell into and why instruction 73 says to write the blurbs in the same change.

### lines 2981-2987

```python
"CLASSIFIER":
```

RESTORED. These two were deleted on 2026-08-12 as unreachable, on the strength of an app list that did not include `classify_merged` -- which renders BOTH of them. The list came from `test_every_qt_section_hint_names_a_real_category`, whose own comment says it has to be exhaustive rather than representative, and it was neither. The test now includes classify_merged, so deleting a live blurb on that evidence again fails instead of shipping.

### lines 3066-3069

```python
"RESPONSE":
```

The Qt regression layout's own section names. The shared spacr.settings.categories names for the same six groups are the "REGRESSION: ..." entries below; both maps are rendered, so both need a curated blurb or the section shows the generic fallback.

### lines 3331-3334

```python
"LABELS & CLASSES":
```

Classify (ML)

Named to match Classify (CV)'s group of the same purpose. The two modules did the same job under different words, which is what made a settings CSV non-portable between them.

### lines 3362-3365

```python
"PLOTS & HEATMAPS":
```

"OUTPUT & DATABASE" went with `save_to_db` on 2026-09-09 (357-Q4). Its whole subject was that one setting -- "whether model scores are written back into the measurements database" -- and nothing read it, so the hint described a choice the run did not offer.

### line 3370

```python
"INPUT TABLES":
```

Regression

### lines 3375-3378

```python
"CONTROLS & FILTERS":
```

"CONTROLS & PLATE DESIGN", "QUALITY FILTERS" and "SIGNIFICANCE & HIT CALLING" were retired on 2026-08-17 with instruction 135: the first two merged into "CONTROLS & FILTERS" and the third into "MODEL & INFERENCE". Their hints merged with them rather than being dropped.

### lines 3387-3390

```python
"ADDITIONAL SETTINGS":
```

"MODEL & COVARIATES", "HIT CALLING & OUTLIERS" and the flat "REGRESSION" heading were retired when the regression layout was split into Response / Model & Inference / Estimator Tuning / Permutation Test / Significance & Hit Calling / Quality Filters. Their replacements are above.

### lines 3450-3455

```python
"INPUT & CHANNELS":
```

shared by the three Cellpose-facing modules

Mask, Cellpose Masks, Cellpose All and Train Cellpose ask the same four questions about a segmentation run in the same order. Naming the groups identically is the point: someone who learned them once should not have to relearn them in the next module.

### lines 3535-3537

```python
"ILLUMINATION CORRECTION":
```

Measure's single illumination heading. The Illumination screen's four tabs keep their own blurbs below; this one covers all of them, because under Measure they are one section.

### line 3598

```python
"ESTIMATOR TUNING":
```

Regression

### lines 3624-3630

```python
"POWER ANALYSIS":
```

Power / Design

"Power analysis" is the single heading `spacr/qt/screens/power.py` registers all fifteen of its keys under, which is what the settings diff and the run journal group them by when the module's own screen is not involved. The five headings below are what the layout splits it into; this entry covers the undivided one.

### lines 3666-3668

```python
"OUTPUT & RUNTIME":
```

Train Cellpose fits weights; the other three run them. "Model" therefore names the thing being produced rather than the thing being picked, which is a different sentence.

### lines 3752-3757

```python
"CHANNEL MAPPING":
```

CHANNELS ONLY, AND THE HEADING SAYS SO NOW. This category used to be "Mask & Channel Mapping" and to carry three `*_mask_dim` rows that `analyze_recruitment` never read -- instruction 364 measured zero reads against four each for their channel twins. The rows and their factory keys are gone, so both the heading and this blurb stop promising a mask index the module does not use.

### lines 4092-4098

```python
"cell_montage": "cell_montage",
```

Registered without a mapping, so their help had no API page to link to. The Volcano Explorer redraws a finished regression's coefficient table and the Parameter Sweep reads the trials of a search that already ran; each points at the module that produced what it is showing. The Cells tab -- which objects a dot on the volcano is most consistent with. Instruction 131; the answer is pure pandas in `cell_montage` and the tab only loads what it names.

### lines 4101-4106

```python
"barcode_qc": "sequencing_qc",
```

THE FOLDED MODULES. These three reached this table through

``register_app(..., api_module=...)`` -- the push half of the seam absorbed below -- so folding them into a host screen and dropping the row would take the API link out of the hover help on every one of their settings, and the folded page's help would point at the generated API index instead of at the module that does the work.

### lines 4110-4112

```python
"illumination": "illumination",
```

Illumination reached this table from its own row too, and folded into Measure. Its module is the one that estimates the flat field, so the settings on the folded page point there rather than at the index.

### lines 4115-4119

```python
"image_scatter": "qt/screens/image_scatter",
```

Image Scatter and PCA reached this table the same way, from their own rows. Both are folded onto Image UMAP now, and `unregister_app` takes a pushed entry back out with the row it came from -- so without these two lines the help on either screen falls back to the generated API index instead of the page that documents it.

### lines 4122-4125

```python
"curate": "qt/screens/curate",
```

Curate the same way, from its own row into Make Masks. Its page is the brush rather than a settings form, so nothing asks for its link today; the line is here because the alternative is that the answer silently became the generated API index the first time anything did.

### lines 4129-4134

```python
"ops": "ops_engine",
```

OPS folds onto Align & Stitch and was never given a row, so the fold reference on the API homepage listed it as bare text and its settings' help pointed at the generated index. Its entry point is `spacr.ops_engine.run_ops` -- the same module `bridge` imports to run it -- and `ops_engine` is a public module with a page of its own, so there was a page the whole time and nothing addressing it.

### lines 4462-4464

```python
POSITIVE_INTEGER_SETTINGS = frozenset({"guide_permutations"})
```

These integer settings are undefined at zero and negative values. Limiting the editor prevents the GUI from producing values rejected by the engine; pre-flight validation still handles settings loaded from external files.

### lines 4968-4985

```python
REGRESSION_LEVELS = ("both", "grna", "gene")
```

The Model & Inference explainer box (instruction 132)

"it is important for the user to know all of this."  A read-only box in the Model & Inference section that states, for the CURRENT selection, the formula that will be fitted and what it models.

It is prose, not a tooltip, because the thing it has to say does not fit in one: the default changed to `mixed`, and a mixed fit answers the gene question WELL while giving up something the previous default appeared to give -- a guide-level hit list. A user who takes the default and later goes looking for their guide p-values is exactly who this box is for, so the cost is a named section of it rather than a clause someone might not hover.

THE TEXT IS BUILT BY A PURE FUNCTION so it can be asserted without a QApplication, and so the formulas have one spelling in this file rather than one per branch of a widget callback.

### lines 5092-5095

```python
"group_lasso": "guides grouped by gene",
```

Kept short on purpose: the header renders as "MODEL: <key> -- <title>" on ONE unwrapped line, and the box does not soft-wrap, so a title long enough to pass 54 characters puts the model's own name behind a horizontal scrollbar.

### lines 5266-5276

```python
MIXED_COST_ANCHORS = (
```

What the default costs (instruction 140)

Reported 2026-08-18: "im running the mixed model now and it is taking much longer than before is that normal?" ... "it is still going, cpu at 100 percent". NOTHING WAS WRONG. MixedLM is an iterative REML optimisation and it is single-threaded, so one core at 100% for an hour is exactly what a healthy fit looks like -- and an hour of silence at 100% CPU is indistinguishable from a hang. 132 made `mixed` the DEFAULT, which means everybody pays this, so it belongs where the model is chosen.

### lines 5530-5548

```python
_MATHS_RESPONSE = {
```

The box is TYPESET, not dumped (instruction 144)

2026-08-18: "actually my main problem was it dosnt look great. i want you to use markdown and colors for negative (CANNOT) and positive (MODEL, LEVEL, ETC.) text. make the formula look better (write the math symbol version then the code version if possible) short discriptions that contain the vital information for the user and links to APIs for the different methods".

143 read the first report as "too long" and cut 2,438 characters to 892. The content is settled; what was left is that nothing was EMPHASISED, so a formula read exactly like a caveat.

ONE SOURCE, TWO LAYOUTS. Everything below composes the SAME pieces the plain renderer does -- `_MODE_TITLES`, `_MODE_NOTES`, `formula_for`, `mixed_cost_note` -- so the two cannot say different things. Only the layout is written twice, and the plain one stays because it is what a test can assert on and what a headless caller can print.

### lines 6321-6335

```python
_PERMUTATION_NOTE = (
```

The Permutation Test explainer box (instruction 135)

"Permutation test is good it just needs a text box at the top briefly explaining what it does."  ONE PARAGRAPH, and shorter than the model box above: the eight controls under it are already named for what they do, so what is missing is only the sentence that says what the test IS.

Written from `spacr.guide_permutation` rather than from the general reputation of permutation tests, which is why it says "marginal" out loud. The module's own docstring is explicit that it "does not claim to estimate a simultaneous conditional coefficient for every guide", and a user who reads "permutation test" as "the same fit, only distribution-free" would take a marginal association for a conditional one.

### lines 6346-6352

```python
"Each guide is tested independently, as a marginal association rather "
```

BOTH THE PLAIN WORDS AND THE TERM. The longhand -- "one coefficient in a design holding every guide at once" -- is what a reader who does not know the vocabulary needs; "conditional coefficients" is what a reader who does will look for, and it is the phrase `guide_freedman_lane_test`'s own docstring uses when it says the test "does not claim to estimate a simultaneous conditional coefficient". Dropping the term left the distinction true but unsearchable.

### lines 7225-7248

```python
NESTED_CAPABLE_KEYS = frozenset({
```

List / list-of-list editor

A list setting used to be a text box holding a Python literal:

class_metadata   [['c1'], ['c2']] train_channels   ['r', 'g', 'b']

which is both ugly and unforgiving -- a dropped bracket is a parse failure with no diagnosis, and `_ListEdit.get_value` silently handed the unparseable text through as a plain string. Worse, `_ListEdit` was never reached: `gui_utils.convert_settings_dict_for_gui` stringifies every list default before this module sees it (`('entry', None, str(value))`), so `isinstance(default, list)` in `_widget_for` was always False and every list setting got a `_ScalarEdit`. `collect()` then returned the raw text, because `_coerce_to_expected_type` only ever handled bool/int/float. That is how `class_metadata` reached `io.generate_training_dataset` as the *string* "[['c1'], ['c2']]" and got iterated character by character.

The widgets below replace the literal with removable chips -- one chip per value, one row per inner list -- and hand `collect()` a real Python list. The stored value is unchanged, so every settings CSV on disk still loads and every consumer reads what it always did.

### lines 7258-7259  _(unsure)_

```python
"cell_loc", "pathogen_loc", "treatment_loc", "barcode_coordinates",
```

declared ``(list, list)`` in expected_types, the in-tree marker for "this can be a list of lists"

### lines 7263-7268

```python
CHANNEL_LIST_KEYS = frozenset({
```

Channel selections that contain more than one channel use the same add/remove-chip editor as ``manders_thresholds``.  The legacy GUI converter still labels the first three as curated combos, so keep this declaration close to the list editor and let the real per-module default decide whether the setting is actually a list.  Scalar selectors such as ``cell_channel`` and ``channel_of_interest`` are intentionally absent.

### lines 7272-7275

```python
})
```

png_dims is deliberately absent: it is superseded by png_channel_mapping, which has its own three-field R/G/B editor (widgets/channel_mapping.py). Leaving it here would have offered a chip list for a setting nothing renders.

### lines 7664-7666

```python
from ..widgets.flow import FlowHost as _FlowHost, FlowLayout as _FlowLayout
```

The chip strip's wrapping row now lives beside the other widgets, because the regression results header needs the same thing. The private names stay so nothing that imported them from here has to move.

### lines 7937-7952

```python
"channel_of_interest": ((0, "Ch 0"), (1, "Ch 1"), (2, "Ch 2"),
```

WHAT THE MODEL IS ALLOWED TO LOOK AT (236 A2), asked for as "the user can train on channel_1 measurements only or morphological measurements or channel combinations, localization ... This should be straight forward and easy."

`utils.filter_dataframe_features` has always taken a list, 'morphology' and a free-text fragment. The setting declared `int`, so a spin box was all the panel could draw and three of the four documented ways of choosing a feature space were unreachable. A multi-select says the whole question in one row: light one chip for one channel, two for the combination, Shape for morphology, none for every feature.

LOCALISATION NEEDS NO CHIP. A colocalisation column names the two channels it measures and survives a request for either, so asking for channel 1 already brings channel 1's relationships with it.

### lines 7987-7989

```python
try:
```

AT IMPORT TIME, so the failure is not a missing chip style -- it is the module not importing, which takes down whatever imports it. Driven in tests/qt/test_a_theme_that_refuses_does_not_stop_an_import.py.

## resolve_default_settings

### lines 179-190

```python
_import_registered_defaults_module(app_key)
```

Modules that shipped their own defaults through the `register_defaults` seam. Consulted after plugins and before the built-in dispatch below, so a registered module is served without editing this function -- which is the whole point of the seam, and without this line every `register_defaults` call in the codebase is inert.

Import first, ask second. `register_defaults` runs at the module's own import, so the seam only answers for a module something has already imported -- and a pipeline module has no reason to be imported by the process that is merely drawing its settings panel. `register_app(..., defaults_module=...)` names it; this is what makes the panel appear instead of an empty form.

### lines 215-221

```python
s = set_default_settings_preprocess_generate_masks(settings={})
```

Timelapse tracking and the automated motility assay are first-class modules of their own now (app keys 'timelapse' / 'motility'), so the Mask module edits neither set of knobs. The keys are dropped from the *editable* dict only — preprocess_generate_masks re-applies set_default_settings_preprocess_generate_masks internally, so a Mask run still gets timelapse=False / motility_analysis=False, and a CSV driven straight through the API keeps working unchanged.

### lines 228-229  _(unsure)_

```python
s.pop("motility_analysis", None)
```

The Timelapse module tracks objects; running the assay is what the Motility Assay module is for, so its inline gate isn't offered here.

### lines 231-237

```python
s["timelapse"] = True
```

`timelapse` stays in the dict and stays True. It is not rendered as a control -- see the layout -- because this module is the timelapse one and a user turning it off here would be left with a screen of controls about a time dimension it was told to ignore. Forced rather than merely defaulted, so a settings CSV saved by an older build with `timelapse: False` cannot silently turn this module into a slower Mask Generation.

### lines 242-244

```python
s.pop("motility_analysis", None)
```

`motility_analysis` is the Mask-pipeline gate for the inline assay (spacr.object), not a knob of the assay itself — opening the Motility module *is* asking for the assay.

### lines 263-266

```python
for key in (
```

The original controls describe one lab's c1/c2/c3 plate convention. Keep them as API-compatible backend defaults, but do not expose them in the general UMAP UI. ``exclude_rows`` replaces them with rules based on the columns and values in the user's own database.

### line 296  _(unsure)_

```python
return {"src": "path to images"}
```

These are interactive apps; return minimal placeholder.

## keys_hidden_by_their_object

### lines 751-754

```python
if role == "cell":
```

CELL IS NEVER GATED. It is the object every other one is measured against, and instruction 300 explicitly superseded the earlier channel-following rule for this one family. Its plane may be empty, but the controls a fresh run needs must remain available.

## _categories_from_spec

### lines 2063-2064  _(unsure)_

```python
candidates = [key for key in source.get(token[1:], [])
```

A group reference can only mean what the shared map says it means, so it is filtered by what is actually in there.

### lines 2068-2076

```python
candidates = [token]
```

A literal key is the spec ASSERTING where that setting belongs, and it outranks the shared category map — which for Barcode QC and Illumination has never heard of their keys at all. Filtering literals by `available` sent all eleven of Barcode QC's checks to the trailing "Other" bucket, which is the exact thing the layout exists to prevent. Whether the key exists is decided at render time, where `build_sections` already drops any key that produced no widget.

## SettingsSection

### lines 2124-2140

```python
class SettingsSection(tuple):
```

THE SETTINGS TREE. Instruction 73.

The panel used to group by OBJECT and nothing else, so `cell_min_size` and `nucleus_min_size` -- one decision applied to two objects -- read as two unrelated knobs filed under two headings. The request is a second axis: group the advanced settings by WHAT THEY DO, then by which object they do it to, under one "advanced settings" umbrella.

That needs three levels, and the panel had one. `build_sections` returned List[Tuple[str, List[Tuple[str, QWidget]]]] -- a header and its rows, no third element and no recursion -- so a sub-sub-section could not be expressed at all. Widening that return type is a contract change for every module in the tool, which is why the section below is a TUPLE SUBCLASS: it still IS the pair it always was, so nothing that unpacks or `dict()`s the result has to change, and the tree hangs off attributes beside it.

## SettingsSection.__new__

### lines 2157-2158

```python
def __new__(cls, title, own_rows=(), children=()):
```

No `__slots__`: a variable-length tuple subclass cannot have one, and the four attributes below are what carries the tree.

## _nest_sections

### line 2310, trailing  _(unsure)_

```python
out.append(parent)
```

a placeholder, replaced below

## needs_curated_layout

### lines 2380-2382

```python
return False
```

An app whose defaults will not resolve has no settings panel to judge. Reporting "needs a layout" would fail the invariant test for a reason that has nothing to do with layouts.

## categories_for_app

### lines 2468-2474

```python
if app_key == "umap":
```

Map Barcodes used to relocate `n_jobs` and `test` into "Sequencing" here, so the module would stop rendering an "Advanced" tab holding one setting and a "Model Training" tab holding another. That left thirteen unrelated keys in one "Sequencing" drop; `_APP_CATEGORY_SPECS` now names all five groups the module actually has, which places those two keys — and every other one — explicitly. The relocation is not deleted behaviour, it is superseded behaviour.

### lines 2499-2503

```python
ordered = {
```

NINE groups became SIX, named for what they hold rather than for the stage of a workflow. "Validation", "Evaluation Workbench" and "Monitoring & Runtime" were three headings for one question -- how do I know whether this worked -- and nobody looking for a cross-validation setting knew which of the three to open.

### lines 2510-2518

```python
"Labels & Classes": [
```

`classes` is the heart of this module: it is the only setting that says which objects the model is being taught to tell apart. `dataset_mode` sits above it because it decides which columns the Classes editor offers.

`location_column`, `positive_control` and `negative_control` are NOT here: a control well is a class defined by a metadata column, which is exactly a row of the Classes dict, so three settings saying it a second way were three ways to disagree.

### lines 2520-2534

```python
"dataset_mode", "classes", "class_folder_names",
```

`classes` is what each class MEANS; `class_folder_names` is where its crops are written. One key used to be both. `metadata_type_by` and `measurement_rules` are GONE, not hidden. The first named the column a class is defined by, which is the Classes editor's own column field; the second was a second vocabulary for "a class is a rule about a column", written as hand-edited JSON because it had no editor. Both were answers to a question `classes` now asks once. `annotation_column` and `class_metadata` are GONE from the panel (instruction 229): the Classes editor names the column each class is defined by and the value that defines it, so a box for either was a second place to say the same thing. Both are still WRITTEN, derived from `classes`, so every consumer downstream is unchanged.

### lines 2558-2561

```python
"batch_size", "mixed_precision",
```

`gradient_accumulation` was here and is retired: the step count alone says whether to accumulate, and `steps = 1` IS the off position. A category naming a key with no type and no default draws nothing and hides the retirement.

### lines 2567-2570

```python
"cv_group_by", "holdout_plate", "nested_cv_inner_folds",
```

The plate held back from fitting, beside the folds it is the alternative to. Left out of this layout it fell into "Additional Settings", which is not a heading anyone chose; it is the absence of one.

### lines 2582-2586

```python
ordered["Model & Regularization"] = [
```

The family switch is the TOP-LEVEL choice, not one setting among ninety, so it gets its own group at the top rather than sitting inside "Model Architecture". Greying tells the user a control is inactive; it does not tell them what they are DOING, and that is what the merged module has to make obvious.

### lines 2591-2596

```python
ordered.update({
```

The control wells are NOT added back. ML used to express the metadata basis through location_column plus two control values, and Classify (CV) through class_metadata; the Classes dict now says both, as rows naming a value of a metadata column. Three settings restating one thing were three ways for it to disagree with itself.

### lines 2598-2602

```python
ordered.update({
```

The ML-only groups, appended to the CV ordering rather than duplicated: every CV key is already placed above, so this is exactly the difference between the two modules. Group names match Classify (ML)'s own, so a user moving between the three screens sees the same headings.

### lines 2610-2612

```python
"Model & Features": [
```

Preparing features, choosing a model and ranking features were three headings asking one question: which features the model uses. One heading, read top to bottom.

### lines 2621-2624

```python
ordered["Evaluation & Results"] = (
```

The heatmap is not a machine-learning setting: it is how a result is shown, and the CV family wants it just as much. So it joins the shared evaluation group rather than being prefixed onto one family.

### lines 2630-2636

```python
cv_family = "Computer Vision"
```

Rebuilt in order, because dict order IS the panel order: the family choice first, then the shared groups, then each family's own settings under a heading that names the family.

The shared groups are deliberately NOT prefixed. "Labels &

Classes" applies to both families, and prefixing it would imply it belonged to one.

### lines 2641-2643

```python
ml_groups = ("Model & Features", "Plate & Batch Correction")
```

Feature preparation and feature importance were two headings asking one question -- which features the model uses -- and "Output & Database" was one setting under a heading of its own.

### lines 2656-2660

```python
if name in ordered:
```

A rename of `label` to "Classifier & Validation" stood here, guarded on `name.startswith("ML Classifier")` so the heading would not read "Machine Learning - ML Classifier". No entry of `ml_groups` has been called that since the groups above were consolidated, so it could not run and was removed.

### lines 2663-2666

```python
for name in shared_last:
```

Shared groups that come LAST -- evaluation applies to both families, so prefixing it onto one would be a lie about who it belongs to, and putting it first would bury the settings that decide what is being trained.

### lines 2670-2677

```python
ordered = rebuilt
```

A catch-all that copied any group the five tuples above do not name stood here, and it could not run either: `ordered` is the literal a hundred lines up plus this branch's own two additions, and the tuples enumerate every one of them. It was a net against that literal growing a group nobody added to a tuple -- so the invariant it was catching is asserted directly instead, by `test_the_merged_classifier_panel_loses_no_setting_to_the_rebuild`. A silent net that nobody has ever seen fire is not evidence.

### lines 2703-2705

```python
return _drop_hidden_keys(app_key, result)
```

Last, so it catches a key wherever it ended up -- including the "Additional Settings" bucket, which is where a key goes precisely when no layout claimed it.

## category_tooltip

### lines 3895-3896  _(unsure)_

```python
return ""
```

An empty title has no fallback: "Settings that control ." is worse than nothing, and a caller passing "" wants silence.

## _absorb_registered_api_modules

### lines 4187-4189

```python
pull = getattr(app, "registered_metadata", None) if app else None
```

`getattr(..., None)`: `spacr.qt.app` may be half-built when this runs, in which case nothing has registered yet and the push half of the seam delivers every row later.

## _mapped_api_target

### lines 4271-4272

```python
segment = module[len("spacr."):].replace(".", "/")
```

"spacr.batch_correction" -> "batch_correction"; nested packages keep their path so "spacr.qt.screens.x" would be "qt/screens/x".

## api_docs_url

### lines 4360-4367

```python
chosen_by_hand = True
```

A HAND-WRITTEN TARGET IS A DECISION, and the flow fallback below must not quietly overrule it. Each of these three was checked by a person: a batch-correction setting lands on the module that IMPLEMENTS the correction rather than on whichever app displays it. Those modules have no per-setting anchor, so without this flag the fallback saw "module, no anchor" and sent all seven batch settings to the flow page -- which `test_defaults_and_gui_categories_expose_batch _correction` caught within the hour.

### lines 4377-4382

```python
module, anchor = _mapped_api_target(key, app_key)
```

Instruction 336. Before this, every row fell through to the SCREEN's module, so a setting read twelve calls down still linked to the entry point the reader was already looking at. The generated map says where the value is actually read, from an AST walk rather than the panel it is drawn on. The hand-written cases above still win: they were checked by a person and this table is mechanical.

### lines 4387-4388

```python
anchor = _module_level_anchor(app_key, module or "")
```

THE TILE'S OWN LINK. Six tiles share three module pages; this sends each to the entry point that answers for it. See `_APP_API_ANCHOR`.

### lines 4391-4407

```python
anchor = _anchor_inside(key, module)
```

A CURATED MODULE CAN STILL HAVE A PRECISE ANCHOR. The three cases above choose the module a person decided the reader should land in and then left the anchor empty, so the link went to the top of that page. The consumer map often knows the symbol INSIDE that same module, and taking it keeps the decision while adding the part the hand-written table never carried.

The KEY-ONLY map, not the per-module one, and the difference is why the obvious version of this found nothing: the per-module table is restricted to modules some app's help links to, and `batch_correction` is not one -- it implements a correction, it does not host a panel. Its row lives in the key-only map.

Six of the eight batch settings gain

`batch_correction.correction_kwargs` from this. The eight classifier-evaluation keys gain nothing and cannot: they are read in `deep_spacr`, so their curated module has no symbol to point at.

### lines 4411-4422

```python
url = f"{DOCS_SITE_BASE}/settings_flow.html#{FLOW_ANCHOR}{key}"
```

A MODULE PAGE THAT MAY NOT MENTION THE SETTING. Instruction 383: "i tested the API link for Magnefication and got a page with no mention of magnefication". It was not a wrong module -- utils IS where magnification is read -- but the only consumer is private, so there is no anchor to aim at and the reader lands at the top of 4,000 lines. 245 of 796 links are in that position.

The flow page names the setting, carries its help text and lists every function that reads it, private ones included, each linked on to its own API page. So it answers the question the reader pressed API to ask, and the module page is one click further on rather than lost.

### lines 4433-4434  _(unsure)_

```python
base, _, frag = url.partition("#")
```

The query has to precede the fragment or the browser keeps the anchor inside the query string and the page lands at the top.

## _auto_or_number_box

### lines 4477-4479

```python
box.setSpecialValueText(AUTO_TEXT)
```

Qt shows the SPECIAL TEXT in place of the minimum, so 0.0 is the spelling of "auto" and the user reaches it by winding the box down which is also where somebody hunting for a smaller penalty is heading.

## _float_domain

### lines 4545-4546  _(unsure)_

```python
text = str(_settings.tooltips.get(key, "") or "").lower()
```

`tooltips`, not `descriptions`: `descriptions` is keyed by APP, and `tooltips` is the per-setting text that states the domain.

### lines 4551-4557

```python
return 1e-6, 1.0, min(step, 0.01)
```

The setting says it lives in the unit interval. Hold the box to it: a probability the user cannot type is better than a run that dies forty seconds in having already written half a results folder. The floor is the smallest value the box can express, not 0: a setting whose own text says "between 0 and 1" is refused at 0 by the code that reads it, so a box that clamps a bad saved value to 0.0 has only moved the failure. With decimals=6 that floor is 1e-6.

## _type_hint

### line 4587, trailing  _(unsure)_

```python
s = " or ".join(dict.fromkeys(parts))
```

dedupe, keep order

## language_resolved_once

### lines 4644-4650

```python
try:
```

AND THE OTHER HALF OF THE SAME LOOKUP. The dicts above memoise what THIS module resolves; `i18n.tr` has its own path to the preference store and was still reading it once per translated string inside a scope that existed to stop exactly that. Opening both here keeps one scope for callers to reason about. Imported inside the function for the reason everything i18n is: `preferences` imports i18n, so the cycle is broken by lateness.

## _language_code

### lines 4681-4682

```python
scope = None
```

An unhashable argument cannot be cached; resolve it directly rather than refuse to answer.

## _translated_type_hint

### lines 4781-4782

```python
translated = " / ".join(tr(part, code) for part in core.split(" or "))
```

A slash is a language-neutral union separator.  Translating each atomic type avoids asking the catalog to enumerate every possible union.

## _translated_setting_name

### lines 4806-4807

```python
if source in _ROWS or source in _TERM_ROWS:
```

The compact catalog is the hand-reviewed authority for exact terms. External generated labels extend it, but never override a correction.

## _ApiTooltipFilter.eventFilter

### lines 4925-4926

```python
"""Show the API help popup instead of Qt's own tooltip.
```

Re-render on entry so a Preferences language change cannot leave a sticky popup displaying an earlier language.

### lines 4946-4947

```python
return True
```

Suppress the native tooltip: it disappears when the pointer moves toward its link, whereas HoverTooltip is intentionally clickable.

## mixed_cost_note

### lines 5322-5324

```python
return _translated_ui_text(
```

EVERY NUMBER KEPT, half the words. Instruction 143 B: "do not shorten by deleting the numbers -- they are what makes the claim checkable. Shorten by removing what does not need re-reading."

## _well_keys

### lines 5403-5413

```python
plate = str(frame.attrs.get(_FILE_PLATE, "plate1"))
```

NO PLATE COLUMN, SO THE WELLS OF THIS FILE ARE THIS FILE'S. It used to substitute the literal "plate1", which is the guess its own docstring forbids -- and it cost exactly what that sentence predicts: the example screen's four count files each name the same 384 row/column pairs, so the union across them was 384 for a 1,536-well screen. Off by the number of plates, stated confidently, on the first line of the console.

`_FILE_PLATE` is filled by the caller with the file's position, which is the same rule `load_regression_input_pairs` uses when neither side declares a plate: the pair-row order.

## regression_design_scan

### lines 5471-5482

```python
from ...tabular import read_table
```

THE ONE READER (145), and this line is why. Reading raw, the count tables of the example screen -- which spell their keys `row_name` and `column_name` -- carried no column this scan recognises, so it reported "no 'prc', 'plate_row' or 'rowID'/'columnID' column, so wells were not counted" over 642,551 rows that name 1,536 wells perfectly well.

A count of NOTHING, printed confidently, on a table that has the answer: exactly the failure instruction 145 exists to stop, and the first line of the console a user reads before a run.

`report=None`, because a column-collision note belongs to the run and not to a sizing scan the user did not ask for.

### lines 5495-5496

```python
frame.attrs[_FILE_PLATE] = f"plate{out['files']}"
```

ONE FILE IS ONE PLATE when the file does not say otherwise, which is `load_regression_input_pairs`' rule for the same question.

## regression_model_explainer_html

### lines 5909-5912

```python
recommended_label = _ink(
```

133 A. `mixed` takes its own branch above, so the flag every other backend gets from the shared path has to be added here too -- and missing it on the DEFAULT would have been the one place it mattered most.

### lines 5974-5976

```python
if key in RECOMMENDED_FOR_SCREENS:
```

133 A: say WHICH backends answer this question well, and WHY each. In `success`, the same colour "TWO MODELS, TWO TABLES" uses, because both are affirmations about the model rather than caveats about it.

### line 5986

```python
parts.append(
```

AND THE CAVEAT, because a badge without it reads as a promise.

## regression_model_explainer

### lines 6193-6194

```python
return "\n\n".join(
```

THE SAME WORDS AS THE TYPESET BOX, from the one constant, so the plain renderer and the HTML one cannot describe different runs.

### lines 6200-6202

```python
return (f"{tx('MODEL:')} {key}\n\n"
```

An unknown name is the pipeline's error to raise, with its own list of what it accepts. The box says it cannot describe the choice rather than inventing a formula for it.

### lines 6222-6226

```python
lines.append(tx(_REFUSAL_HEADING))
```

THE COST OF THE DEFAULT, in its own named section. This is the paragraph the box exists for, and the one section instruction 143 left at full length: a user who takes the default and then goes looking for guide p-values reads it exactly once, but they cannot be told to go elsewhere for it.

### lines 6230-6232

```python
lines.append(tx("Recommended for CRISPR screens").upper())
```

WHAT IT COSTS, beside "what you do not get" and for the same reason: both are things a user can only find out by having already spent the afternoon. Instruction 140.

### lines 6257-6259

```python
if chosen in ("both", "grna"):
```

ONE SENTENCE UNDER EACH FORMULA, and no blank line between them: the sentence says what a coefficient IS, which is the one thing a reader needs it for, and it belongs to the formula above it.

### lines 6294-6296

```python
source = (_NO_P_VALUE_BOTH_NOTE if chosen == "both"
```

Saying "BH-corrected" under a backend that reports no p-value would contradict this box's own WHAT ... DOES paragraph two lines above it.

### lines 6301-6305

```python
lines.append(_wrap_block(
```

ITS FIRST SENTENCE ONLY, per instruction 143. The four that followed said why pooling would be wrong and warned that a gene called by both fits is two tests of one hypothesis -- both true, both read once, and the second belongs beside the hit list where somebody is making the claim.

## _apply_greyed_note

### line 6562, trailing

```python
_clear_greyed_note(control)
```

the reason may have changed; it is named

### lines 6566-6570

```python
control.setProperty(_PENDING_NOTE_PROPERTY, note)
```

REMEMBERED ON THE CONTROL, because the label may not exist yet. The first greying pass runs while the panel is being built and the labels are decorated afterwards, so a note written only where a label would be is a note that never appears. Held here, it can be put on the label the moment there is one.

### lines 6575-6582

```python
_note_on_label(label, note)
```

ON THE LABEL, WHICH IS WHERE THE HELP ACTUALLY SHOWS. The editor is deliberately SILENT on hover -- decoration sets its display role to "metadata" and clears its tooltip so the panel does not show two tooltips for one setting -- so the note above went to a string nothing reads. EVERY greyed setting in spaCR was disabled WITHOUT saying why, which is the one thing instruction 106 asks of a greyed control, and it was invisible precisely because the reason was written where it could not be seen.

## _note_on_label

### lines 6595-6599

```python
base = _without_note(base, note)
```

A note may already be BAKED IN: the label's help was composed from the control's tooltip at decoration time, and that tooltip carried the note from the build-time greying pass. Stripped by its exact text, which is known, rather than by a pattern -- guessing where help ends and a note begins is how a restore loses a sentence.

## _clear_greyed_note

### lines 6635-6637

```python
cleaned = _without_note(
```

No backup because the note was applied before this label existed and was baked into its help by decoration. Removed by its own text.

## attach_api_tooltip

### lines 6659-6660  _(unsure)_

```python
body = str(body or "")
```

Keep an absent body absent: format_tooltip owns the localized generic fallback.  Synthesizing an English sentence here bypasses it.

### lines 6662-6666

```python
if key == "regression_type":
```

WHAT A SLOW FIT COSTS, SAID BEFORE IT IS CHOSEN. Instruction 273 section 3: the measurement already exists, and it used to reach a user only after they had started the run -- printed by the console banner, which is after the decision. `mixed_cost_note` is the one source, so the box and the banner cannot say different numbers.

### lines 6676-6677

```python
widget.setProperty("apiTooltipDescription", body)
```

Retain the old property as canonical English for integrations that read it, rather than replacing it with rendered/localized HTML.

## install_api_tooltips

### lines 6777-6782

```python
if widget.property("apiTooltipDisplayRole") == "api-link":
```

The dot this pass CREATES carries `settingKey` itself, so it is found by the sweep the next time it runs and decorated as though it were a setting — each one growing its own dot. That is what made the live-preview panel sprout duplicates every time the form was re-gated (switching Primary object from cell to nucleus). It is help, not a setting; skip it.

### lines 6790-6792

```python
if widget.isHidden():
```

Explicitly hidden controls are not settings in this popup. Decorating one would create a visible wrapper/dot with a hidden field at (0, 0), recreating the very kind of orphan overlay this helper should avoid.

### lines 6799-6816

```python
widget.setProperty("apiTooltipHtml", "")
```

A COMPOSITE FIELD IS NOT A SELF-LABELLING CONTROL, and treating it as one is what put the tooltip on the field.

Reported repeatedly, and measured on the regression panel:

THIRTY-THREE editors sit inside a composite -- a `_ScalarEdit` inside a `_CsvColumnField`, a line edit inside a chip field and the composite was landing in the branch below, which installs the hover filter on the widget itself. Qt delivers `Enter` to a parent when the pointer crosses into any of its children, so hovering the FIELD fired the help.

The branch below is right for a `QCheckBox`, which carries its own visible text and IS its own label. It is wrong for a container, which has no text and whose label is elsewhere or missing. Where there is no label to put the help on, the help goes nowhere -- a field that stays quiet is the requested behaviour, and a tooltip on the field is not a lesser version of it.

### lines 6823-6829

```python
widget.removeEventFilter(event_filter)
```

A one-widget form row (usually a Toggle/QCheckBox) carries its own visible label, so the hover help goes on its own text. Remove before installing. Qt keeps a LIST of filters and calls each installation separately, so decorating the same widget twice makes one hover emit two tooltips. `removeEventFilter` is a no-op when the filter is not installed, which makes this idempotent for free.

### lines 6845-6848

```python
label.removeEventFilter(event_filter)
```

Idempotent: this decoration pass runs again whenever the live-preview form is re-gated -- changing the primary object from cell to nucleus, for instance -- and a second installation on the same label duplicated every tooltip on the panel.

### lines 6852-6853  _(unsure)_

```python
widget.setProperty("apiTooltipDisplayRole", "metadata")
```

The editor itself remains quiet on hover. Keep its metadata so tests, integrations and a later re-parenting pass can still identify it.

## _unwrap_setting_label

### lines 6879-6884

```python
for child in candidate.findChildren(QLabel):
```

BEFORE THE HOST HAS EVER BEEN DECORATED there is no marked child to find, because `settingHelpLabel` is set by the decoration pass itself -- so the first pass over a freshly built form got the host back and treated the row as having no name at all. The first labelled child is the caption `add_row` put there, and taking it is as deterministic as the marked one: a host holds one caption.

## _setting_label_for_field

### lines 6903-6907

```python
candidate: Optional[QWidget] = field
```

A form field is often a wrapper QWidget containing an editor and a Browse button (or two numeric editors). QFormLayout only knows the wrapper, so walk the editor's parent chain before concluding that it is a label-less combined control. Otherwise its hover help ends up on the editor instead of on the form label.

### lines 6918-6919

```python
for grid in owner.findChildren(QGridLayout):
```

Hand-built search panels use compact grids rather than QFormLayout. Select the nearest widget to the field's left on the same row.

## _CsvColumnField.__init__

### lines 7124-7125  _(unsure)_

```python
self.setFocusProxy(self.edit)
```

Typing goes to the box, not to the button, when the row is tabbed into or given focus programmatically.

## _CsvColumnField.pick

### lines 7170-7173

```python
choices = columns_module.available(paths)
```

ONE read, and it is the only one. `columns.describe` and

`columns.resolve` would each re-read the headers to build the same list; the list is already here, so the near-miss below is computed from it rather than by asking the files a second time.

## _RegressionBackendField.__init__

### lines 7321-7324

```python
self.combo.addItem(label, userData=label)
```

The LABEL is the stored value (spacr.settings. _resolve_regression_backend says why), and it is kept in userData rather than read back off the text: the text carries the refusal for a disabled entry and is therefore not the value.

### lines 7330-7332

```python
self.description.setOpenExternalLinks(True)
```

CLICKABLE, which is the ask -- "linkt the the API for each". A QTextBrowser without this swallows the click and tries to navigate itself to a URL it cannot render.

### lines 7335-7337

```python
self.description.setMinimumHeight(132)
```

Seven lines at the pane's default width, measured on the real screen: enough that the selected backend's paragraph and the first of the other seven are both on screen before anyone scrolls.

## _RegressionBackendField._normalise_type

### line 7390  _(unsure)_

```python
@staticmethod
```

what is choosable, and what the box says

## _RegressionBackendField.refresh

### lines 7431-7434

```python
self.combo.setItemText(
```

THE REFUSAL IS IN THE ENTRY'S OWN TEXT. A disabled row in a dropdown is grey and silent; Qt's item tooltip is shown only while the popup is open and only under the cursor, so on its own it is a reason a user can walk straight past.

### lines 7450-7455

```python
disable_combo_row(self.combo, index,
```

NOT `setEnabled(False)` ALONE. Measured 2026-08-18: it leaves `ItemIsSelectable` set, so Qt refuses to activate the row from the popup but a model-level selection can still land on it. `disable_combo_row` clears the flag too, and keeps the tooltip -- which is what the hover panel is hung off.

### lines 7467-7471

```python
html = ("<p><b>This run will be refused.</b><br>"
```

THE SELECTION IS KEPT AND THE REFUSAL IS SHOWN. Re-pointing the setting at statsmodels here would be exactly the silent fallback instruction 141 C forbids -- and the sentence below is the one `spacr.ml._require_backend` will use if the run starts anyway, so the panel and the run say the same thing.

## _RegressionBackendField.availability_entries

### lines 7500-7518

```python
def availability_entries(self) -> List[dict]:
```

the unavailable entries explain themselves (instruction 158)

THE ROW STAYS DEAD and everything interactive lives in the hover panel. Three routes reach it and they are all here rather than in the panel, because the panel is shared with the Image UMAP and must not know what a regression backend is:

hovering a greyed row in the OPEN popup, anchored on that row; hovering the CLOSED combo while the value it holds has gone unavailable -- 141 C keeps a stale selection rather than silently re-pointing it, so this is a state a user can sit in; Shift+F1 on the combo, which is the keyboard route. It has to be explicit: the rows are disabled, so nothing about them is tabbable and no help can be inherited from them.

THE POPUP IS CLOSED THE MOMENT THE POINTER LEAVES IT. A QComboBox popup is a `Qt.Popup` with an active mouse grab, so with it still open the first click on the panel would be eaten by the grab -- the Install link would need two presses and the first would look like it did nothing.

## _RegressionBackendField.eventFilter

### lines 7545-7546

```python
return super().eventFilter(obj, event)
```

The combo's C++ half has gone. An event filter outlives the widget it watches, so this is teardown rather than never.

### line 7554  _(unsure)_

```python
self._release_popup()
```

Leaving the popup is how the pointer travels to the panel.

## _RegressionBackendField._hover_popup_row

### lines 7579-7580  _(unsure)_

```python
position = event.pos()
```

`position()` is Qt 6; `pos()` is the Qt 5 spelling. spaCR is installed against both.

## _Chip.__init__

### lines 7698-7703

```python
apply_close_mark(close, tooltip=tr("Remove {value}", value=text))
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

THE VALUE IS A VALUE. Splicing it in first asks the catalog for "Remove Cell", "Remove cytoplasm" and one key per chip anyone ever types; the caption is looked up as a template and the value put in after, so the verb translates whatever the chip holds.

## _ChipStrip.__init__

### line 7772

```python
apply_close_mark(self._drop, tooltip="Remove this group")
```

THE APPLICATION'S CLOSE MARK -- see `theme.apply_close_mark`.

## _ChipStrip._add_chip

### line 7824  _(unsure)_

```python
self._flow.removeWidget(self._entry)
```

Keep the entry field last so it always trails the chips.

## _AlphabetSelect.__init__

### lines 8042-8043  _(unsure)_

```python
button.setAccessibleName(str(value))
```

The accessible name is the value, not the label: a screen reader user is choosing 'r', and "Red" is only the gloss.

## _AlphabetSelect._as_members

### line 8101

```python
parsed = [part for part in text.replace(",", " ").split()
```

A bare "r,g" or "r g" from a hand-edited CSV.

## _ListEditor.__init__

### lines 8146-8150

```python
from ..theme import active_palette, font_px
```

font_px is used further down this method. Importing only active_palette here raised NameError out of build_sections(), and AppScreen turns that into "Failed to build settings for '<app>'" so sixteen shipped modules, mask and measure and classify among them, opened with no settings form at all.

## _ListEditor._rebuild

### lines 8240-8241

```python
strip._entry.blockSignals(True)
```

editingFinished fires while a focused QLineEdit is being torn down, which would call _commit_entry on a half-deleted strip.

## _ListEditor._drop_strip

### line 8274  _(unsure)_

```python
self._rebuild(False, [])
```

Removing the only group is how you go back to a flat list.

## _ListEditor._on_footer

### line 8289  _(unsure)_

```python
current = list(self._strips[0].values()) if self._strips else []
```

Flat -> grouped: the values already typed become the first group.

## _ListEditor._placeholder

### lines 8312-8315

```python
"""A short prompt naming the KIND of value this list takes."""
```

Short enough to survive the narrow settings column without eliding -- the point of the placeholder is to say what KIND of value belongs here, and an elided "add a whole numb…" says less than "add number".

## _ListEditor._as_sequence

### lines 8372-8373  _(unsure)_

```python
return [part.strip() for part in text.split(",") if part.strip()]
```

Not a literal: treat it as a comma-separated list, which is what a user hand-editing a settings CSV most often means.

## list_shape_for

### lines 8428-8429  _(unsure)_

```python
if flat and all(isinstance(v, str) for v in flat):
```

bool first: bool is a subclass of int, and a list of flags is not a list of numbers.

## SettingsWidgets.__init__

### lines 8475-8484

```python
from spacr.settings import organelle_slots_beyond_the_count
```

EVERY SLOT THAT CAN BE NAMED, not the four the module ships. A control that was never built cannot be revealed, so a panel whose defaults stop at `number_of_organelles` slots can only ever render that many however the count is driven -- which is exactly why raising the count to seven went on drawing the same four. The extra keys arrive with the values they would have had and their rows are hidden by `refresh_object_visibility`, so what the count changes is which of them is ON SCREEN. `number_of_organelles` itself is left at the module's own number: this widens what can be shown, not what the panel opens showing.

### lines 8488-8501

```python
current_values = {str(k): v for k, v in (current or {}).items()}
```

ONLY THE SLOTS THE COUNT ASKS FOR. Building every nameable slot and hiding the surplus cost the Mask screen 1,551 widgets where a few hundred would do, and every one of them was constructed, laid out and walked by each pass over the form before being hidden again.

`number_of_organelles` says how many a run has, and a run with none has none: the rows are built when the count is raised (see `grow_to_fit_the_organelle_count`), which is one deliberate change to one control rather than something that happens while typing. WHAT THE FORM IS BEING REBUILT FOR. A panel built from the module's shipped defaults can only ever show what a fresh run has; when the user types a nucleus channel or raises the organelle count, the form has to be built for the values ON SCREEN, not the ones the module ships. `current` is those values.

### lines 8509-8511

```python
current_names_slots = any(
```

A file written before the count existed still means what its slot keys say. Do not let the shipped count (now zero) shadow that inference when such a mapping builds a replacement form.

### lines 8522-8531

```python
from spacr.settings import expected_types
```

AND THE SLOTS THE COUNT DOES NOT ASK FOR ARE NOT BUILT AT ALL, including the first one. Its keys are in the module's shipped defaults rather than invented by the panel, so trimming the invented ones left 54 `organelle_*` settings and their categories on a form whose count says zero -- which is the thing the count is supposed to decide. AND THE WIDGETS ARE BUILT HOLDING WHAT THE USER TYPED. `deciding` settles the form's SHAPE; without this the new form arrives at the module's defaults, so a second rebuild collects a nucleus channel of None and takes the nucleus settings away again.

### lines 8543-8548

```python
self._slots_the_panel_added = {
```

WHICH KEYS THE PANEL INVENTED, and what it gave them. A settings file is not a panel: writing every slot that can be named into every CSV would bury the four a run uses. `collect` leaves these out again while they are above the count AND still hold exactly what was put here -- so a value that came from anywhere else, a user or a loaded file, is written out whatever the count is.

### lines 8553-8554  _(unsure)_

```python
self._hidden_by_the_run: set = set()
```

What the object rule decided last, and the rows watching to see that it sticks. See `_guard_hidden_rows`.

## SettingsWidgets._build_sections

### lines 8615-8621

```python
from spacr.settings_spec import convert_settings_dict_for_gui
```

`spacr.settings_spec` -- a module that imports NOTHING, which is the entire point of it existing. This function used to be reached through a module that pulled IPython, matplotlib.pyplot, cv2 and huggingface_hub in behind it: 770 ms on the GUI thread, and the whole remaining cost of opening the first module. That module is gone now, but the rule it taught is not -- import the leaf, not the package that re-exports it. See spacr/settings_spec.py.

### lines 8625-8631

```python
hidden_keys = set(_APP_HIDDEN_KEYS.get(self.app_key, frozenset()))
```

Materialize a widget per key; attach a rich HTML tooltip that ends with a compact information-icon link to the spaCR documentation. A hidden key gets no widget at all, which is what actually hides it: the trailing "Other" section below is built from `self._widgets`, so a key left out of every category still renders as long as a widget exists for it. The value stays in `self._defaults` and reaches the run unchanged.

### lines 8634-8643

```python
from PySide6.QtCore import QCoreApplication, QEventLoop
```

THE EVENT LOOP GETS A TURN EVERY SO OFTEN. A module screen builds about 1,500 widgets, which took 1.5 SECONDS OF SOLID GUI THREAD measured as zero timer ticks for the whole build, which is what "the theme freezes when I click a module" is. Qt requires widgets on the GUI thread, so this cannot move; what it can do is stop holding the thread for the entire run.

`processEvents` and not a chunked timer: the caller expects a built panel when this returns, and handing it a half-built one to be finished later would move the bug into every consumer.

### lines 8646-8650

```python
import time as _time
```

BREATHING ON TIME, NOT ON COUNT. Every 60 widgets left a worst gap of 324 ms, because the widgets are not equally expensive and a fixed count breathes at the wrong moments. A deadline gives up the thread whenever this loop has held it too long, whatever it was building.

### lines 8661-8665

```python
QCoreApplication.processEvents(
```

EXCLUDE user input. A half-built panel must not receive a click that lands on a widget which is about to move, so the backdrop repaints and the interface stays alive while the pointer and keyboard wait the extra second out.

### lines 8684-8687

```python
src_widget.value_changed.connect(self._refresh_contextual_widgets)
```

The same obligation through a different control: adding a plate changes which columns and which rows the dependent fields can offer, so the panel follows the SET as it is edited rather than only when the screen is built.

### lines 8690-8693

```python
family_widget = self._widgets.get("classifier_family")
```

The training basis changes which controls matter, so the panel has to follow it as it is changed rather than only when the screen is built. A bound method, not a lambda: see INVARIANTS 4 for what a closure connected to a Qt signal costs.

### lines 8712-8715

```python
type_widget = self._widgets.get("regression_type")
```

An entry greyed for one family is choosable for another, so the backend control has to follow `regression_type` rather than be judged once when the panel is built. Bound method, not a lambda: INVARIANTS 4.

### lines 8735-8737

```python
self._connect_setting_dependency_signals()
```

Every panel, not only the regression one it was built against. _rules_for_this_panel decides what applies here, and a panel with no gated setting connects nothing. See its docstring.

### lines 8740-8743

```python
self._connect_object_visibility_signals()
```

THE OBJECTS THIS RUN HAS. A channel that gains a number reveals its object's settings and losing it hides them again, and the type a slot is given decides which of that slot's detection settings are on screen at all. Bound method, not a lambda: INVARIANTS 4.

### line 8751  _(unsure)_

```python
cats = categories_for_app(self.app_key, get_categories())
```

Bucket into sections.

### lines 8754-8755

```python
hidden = _APP_HIDDEN_CATEGORIES.get(self.app_key, set())
```

Categories that don't apply to a given app (e.g. the classify app trains a Torch model, not Cellpose — so it gets no Cellpose tab).

### lines 8763-8766

```python
row_keys: List[str] = []
```

THE KEYS, ALONGSIDE THE ROWS. A row is `(label, widget)` and a label is a sentence for a human, so the object a row belongs to can only be read off the KEY -- the same confusion that put the plate map on nothing at all when a label was matched instead.

### line 8781  _(unsure)_

```python
remaining = [(self._label_for(k), self._widgets[k])
```

Trailing 'Other' for anything not in a category.

### lines 8787-8801

```python
if self._parent is not None:
```

ONCE THE SCREEN HAS LAID THE ROWS OUT. This hands the rows back and the screen builds each label and puts the pair into a QFormLayout afterwards -- so there is no ROW to hide yet, and hiding the field here and nothing else would leave its name behind on an empty row. Zero delay, so it lands on the next turn of the event loop, before the panel has been painted.

OWNED BY THE PANEL'S OWN WIDGET. PySide 6.6 cannot bind a Python callable through QTimer.singleShot's receiver overload, so use a real single-shot timer instead. Its QObject parent cancels the pass when a screen is destroyed inside the same turn; delete it after a successful pass so a long-lived panel does not collect dead timers. Nothing is scheduled at all without a parent -- a SettingsWidgets built with no parent is being used for its values and has no rows to lay out.

## SettingsWidgets._keys_of_objects_the_run_has_no_channel_for

### lines 8840-8843

```python
if not switches:
```

A module that owns neither spelling does not own this gate. Treating a missing ``*_channel`` as an empty channel removed Measure's real ``*_mask_dim`` control and left no way to say a nucleus or pathogen mask exists.

### lines 8853-8854

```python
continue
```

THE SWITCH ITSELF STAYS, or there is no way to say the run has this object after all.

## SettingsWidgets._organelle_keys_beyond

### lines 8885-8886

```python
continue
```

THE CONTROL ITSELF ALWAYS STAYS, or a run with no organelles would have no way to ask for one.

### line 8888  _(unsure)_

```python
owner = max((p for p in every if name.startswith(p)),
```

Longest prefix first: `organelleb_` before `organelle_`.

### lines 8894-8898

```python
beyond.add(name)
```

A SETTING ABOUT ORGANELLES IN GENERAL, which a slot prefix does not catch: `summarize_organelles_by` is one, and it kept "Organelle Segmentation (advanced)" on a form whose count said none. With no organelles there is nothing for it to be about.

## SettingsWidgets.search_text_for

### lines 8946-8958

```python
def search_text_for(self, key: str) -> str:
```

Finding a setting among the many

Mask alone renders 190 settings under thirteen collapsed headings. Someone who knows the knob exists still has to guess which heading somebody else filed it under, and someone who only knows what they want to change ("stop merging touching cells") has no entry point at all.

So the haystack is deliberately wider than the key: the description is the only part of a setting written in the language a user thinks in. Searching "gpu" has to find `n_jobs`, and "touching" has to find `merge_edge_pathogen_cells`, and neither word is in either name.

## SettingsWidgets.modified_keys

### lines 9016-9018

```python
continue
```

Rendered but not defaulted: there is nothing to differ from, so calling it modified would be an assertion the module never made.

## SettingsWidgets._widget_for

### lines 9092-9101

```python
if self.app_key == "umap" and key == "src":
```

MORE THAN ONE DATABASE (instruction 109). A screen acquired as three plates is three project folders, and `generate_image_umap` has always taken a list of them -- the panel was the half that could only express one, so the comparison the user actually wants could not be asked for from the application at all. The control adds, removes, and SAYS WHAT THE MERGE WOULD COST before anything runs.

`on_colour_by` writes into this panel's own `color_by` field, looked up when the box is ticked rather than captured now: the fields are built in one pass and `color_by` does not exist yet at this point.

### lines 9121-9125

```python
if key == "classes":
```

Not scoped to one app: every module that crops object PNGs offers this key, and all of them mean the same thing by it. `classes` is a dict of name -> {column, value}, so it gets the editor that can populate it from a column rather than a text box the user has to type JSON into.

### lines 9140-9143

```python
if key in PATH_LIST_KEYS:
```

A setting that names input files gets a file dialog and a drop target, not a box to type absolute paths into. Checked before the chip-editor and combo paths below, because several of these keys are declared ``list`` and would otherwise take the free-text route.

### lines 9155-9163

```python
source = CSV_COLUMN_SOURCES.get(self.app_key, {}).get(key)
```

A setting whose value NAMES A COLUMN of an input CSV gets the box plus a button that reads those CSVs' header row. Checked before the combo and chip paths below because these keys are declared `str` and would otherwise take the plain text box that made a typo indistinguishable from a name.

The paths are read WHEN THE BUTTON IS PRESSED, not here: the user chooses their input files after this panel is built, so a list read at construction is always the empty one.

### lines 9169-9171

```python
paths=partial(self._input_csv_paths, source.roles),
```

`partial`, not a lambda: the callable outlives this call and is read on a button press minutes later, so what it captures should be visible rather than implied.

### lines 9176-9179

```python
if key == "regression_backend":
```

WHO fits the model gets a control that can say why an option is not choosable and what each one is. A plain combo could only offer eight labels and be silent about all of it -- which is what it did until 2026-08-18. See _RegressionBackendField.

### lines 9190-9194

```python
if key == "regression_type":
```

Two inventories are owned by the modules that implement them, so the dropdown cannot list a model spaCR cannot fit or omit a correction it can apply. Both imports are cheap: regression_families reads only regression_spec, which imports nothing, and multiple_testing imports only numpy at module scope.

### lines 9196-9212

```python
kind = "combo"
```

THE SAME TABLE THE OTHER ROUTE READS -- see _regression_type_menu, which settings_spec's _regression_type_choices shares the family half of. Building a second list here out of the bare inventory is what let this panel show nineteen unlabelled names while the other route showed them grouped and explained.

'auto' is the readable spelling of the historical None, which ml.regression turns into check_distribution(response). It is normalised back to None in settings.get_perform_regression_default_settings, so the fit path is unchanged and old settings CSVs holding None still work.

A bare string and a (value, label) pair may share this list the combo builder below takes either -- and every entry here is a pair, so the stored value is what a settings CSV gets while the caption says which kind of fit it is and what it assumes.

### lines 9220-9224

```python
from spacr.hyperparam import UMAP_METRICS
```

One closed alphabet rather than a text field that accepts a typo and fails after the reducer starts.  Importing the constant does not import umap-learn (and therefore does not put a model load on the GUI thread); the runtime validator still consults the installed package.

### lines 9233-9236

```python
if key in FIXED_ALPHABETS:
```

A closed alphabet gets a control that cannot express anything outside it. Checked BEFORE the chip-editor override below, because `train_channels` is in CHANNEL_LIST_KEYS and would otherwise take the free-text path that let 'x' through.

### lines 9244-9256

```python
if key in EXCLUDE_LIST_KEYS:
```

'Exclude' names measurement columns to drop from the feature set, and there is never a reason it should be exactly one -- but spacr.settings declares it (str, None), so list_shape_for (deliberately conservative, and reading only what is declared) sent it to a plain text box. One column per run, and the SQL button overwrote whatever was already there. It gets the same chip strip as Classify (CV)'s `classes`: type a name and it becomes a chip to the right, remove them one at a time, and the SQL button beside it (COLUMN_TABLES) hands back however many columns were selected. Consumers already take either shape -- utils.filter_dataframe_ features and preprocess_data both wrap a bare str in a list -- so a settings CSV written before this still loads, and one written now still runs on the CLI.

### lines 9267-9270

```python
actual_default = self._defaults.get(key, default)
```

Unlike enumerated strings, a list remains a list in every module. The legacy converter presents channel lists and timelapse objects as dropdowns of Python literals. Render them with the same chip editor as manders_thresholds so users can add/remove arbitrary values.

### lines 9282-9285

```python
w = _ValueCombo()
```

_ValueCombo, not QComboBox: some of these lists are

(value, label) pairs, and on a plain combo `setCurrentText` takes the caption only -- so "choose ols" silently does nothing as soon as the caption stops being the value.

### lines 9287-9289

```python
w.setSizeAdjustPolicy(
```

Long inventories (notably UMAP's complete metric list) must not become the minimum width of the whole settings sidebar. The popup still shows every option; the closed control elides.

### lines 9294-9298

```python
if isinstance(opt, tuple) and len(opt) == 2:
```

A (value, label) pair shows the LABEL and stores the VALUE. Instruction 171 wants "load images" and "stream images" in those words in every panel that offers the choice, while 'png' and 'merged' go on meaning what they meant to every settings file already written.

### lines 9305-9311

```python
if key in self._defaults:
```

Pre-select the value THIS module declares, not the one hard-coded in gui_utils.convert_settings_dict_for_gui's special_cases table. That table is one row per key for the whole app, so it shipped 'resnet50' as the model_type default to Classify (which sets 'maxvit_t') and to Activation Maps (which sets 'maxvit'), and '[0,1,2,3]' as the channels default to Cellpose Masks (which sets [0, 0]).

### lines 9319-9324

```python
if default is not None and str(default) != "":
```

The default is not one of the curated options. Silently leaving index 0 selected substitutes a value the module never asked for -- the activation-map app defaults channels to [1, 2, 3] and the channel combo only lists '[0,1,2,3]', so every run started with a different channel set than the defaults declare. Offer the real default too.

### lines 9330-9334

```python
shape = list_shape_for(key, self._defaults.get(key, default))
```

A list setting gets the chip editor, not a text box holding a Python literal. The shape is decided from expected_types plus the REAL default (self._defaults), because convert_settings_dict_for_gui has already str()'d the value that arrives here as ``default``.

### lines 9344-9347

```python
if key in AUTO_OR_NUMBER_SETTINGS:
```

BY NAME, BEFORE THE TYPE SNIFF. These settings take a number or the word "auto", and the shipped default happens to be a number -- so inferring from it built a control that could not express half of what the setting accepts.

### lines 9350-9359

```python
if _is_clearable_plane_setting(key):
```

A PLANE THIS RUN MAY NOT HAVE, for the same reason and one step further: `cell_mask_dim` names a plane of the merged stack, and a screen with no nucleus has no nucleus plane. The control is otherwise chosen from the SHIPPED DEFAULT, so the three that ship a number -- cell 4, nucleus 5, pathogen 6 -- got a spin box, and a spin box has no empty state: the value could be changed but never CLEARED, and being made to name a plane for an object that is not in the run is being made to lie about it. The organelle slots ship None and have always had the box below; this is what makes the family agree.

### line 9364  _(unsure)_

```python
if isinstance(default, bool):
```

Choose widget by inferred type from the DEFAULT value

### lines 9369-9374

```python
if isinstance(default, int) and _permits_float(key):
```

THE DECLARED TYPE WINS over the default's Python type. A setting spacr.settings types as a float gets a float box even when the number it ships happens to be round -- otherwise the box silently refuses every value between the whole ones, and the setting most affected was the Cellpose flow threshold whose own tooltip names 0.4.

### lines 9379-9383

```python
minimum = (
```

Wide enough for the defaults the modules actually ship:

the replication assay's max_area is 1e9, and a +/-1e6 range silently clamped it to 1e6 -- a thousand-fold change to the largest vacuole the assay will score, applied before the user touched anything.

### line 9403  _(unsure)_

```python
w = _ScalarEdit()
```

Fallback — string or None

## SettingsWidgets._coerce_to_expected_type

### lines 9450-9455

```python
try:
```

The curated combos ('channels', 'crop_mode', 'train_channels', 'timelapse_objects', ...) offer their options as TEXT -- "['r','g','b']" -- so a list setting picked from a dropdown reached the pipeline as a string and got iterated character by character. The chip editor already returns a real list; this is the same repair for the combos.

## SettingsWidgets.collect

### lines 9496-9498

```python
for k, v in self._defaults.items():
```

Also carry over any defaults we didn't render (e.g. things not in the categories map that convert_settings_dict_for_gui also skipped).

## SettingsWidgets._refresh_analysis_unit_lock

### lines 9679-9682

```python
try:
```

EVERY SETTING ANY UNIT CONSTRAINS, so switching back to `well` releases what `cell` locked. Refreshing only the current unit's keys would leave a control greyed after the reason for it was withdrawn.

### lines 9689-9690

```python
released = set(getattr(self, "_unit_locked", set()))
```

WHAT THIS RULE ITSELF LOCKED LAST TIME. Only these are released, so a control greyed by another rule stays greyed.

### lines 9700-9707

```python
self.set_value_for_key(key, required[key])
```

SET IT, THEN GREY IT. A greyed control still showing the old value tells the user the run will use that value, and it will not -- which is worse than an editable control that disagrees, because it looks settled. `set_value_for_key`, which is the one writer -- a second way of putting a value into a widget is a second set of type rules to keep in step. It re-enters this method only for `analysis_unit` itself, which is never a required key.

### lines 9712-9718

```python
widget.setEnabled(True)
```

RELEASE ONLY WHAT THIS RULE GREYED. `analysis_mode` is also greyed by the inference rule -- it is set for you by inference='parametric' -- and a blanket setEnabled(True) here undid that, so the combo came back editable while something else was still deciding its value. Enabling a control another rule disabled is worse than leaving one greyed: the user changes it and the run ignores them.

### lines 9722-9723

```python
if hasattr(self, "_refresh_setting_dependencies"):
```

WHATEVER ELSE HAD A SAY, AFTER. The other refreshers re-assert their own greying over anything this one just released.

## SettingsWidgets._refresh_umap_reducer_enablement

### lines 9763-9765

```python
metric.setEnabled(True)
```

The projection may ignore this setting, but DBSCAN always reads it. Keep the shared metric editable instead of greying a control that can still change the result.

## SettingsWidgets._refresh_classifier_family_enablement

### lines 9791-9793

```python
return
```

An unknown family is the pipeline's error to raise, loudly, at run time. Greying on a guess would hide the control the user needs to fix it.

## SettingsWidgets.refresh_training_basis_enablement

### lines 9837-9839

```python
return
```

An unrecognised basis is the pipeline's error to raise, loudly, at run time. Greying nothing is the safe response here disabling controls on a guess would hide the one the user needs.

## SettingsWidgets._plate_context

### lines 10024-10025  _(unsure)_

```python
if sum(os.path.getsize(path) for _, path in sources) > 5_000_000:
```

A very large single-plate file should not stall the GUI merely to grey one field. Leave it unknown; the run still validates it.

### lines 10044-10045  _(unsure)_

```python
plates.add(('source', logical_index))
```

score_data[i] and count_data[i] describe the same plate; their absent IDs therefore share one fallback identity.

## SettingsWidgets._refresh_setting_dependencies

### lines 10061-10065

```python
"""Re-apply the row visibility and then grey the rows that stay.
```

THE ROWS FIRST, THEN WHICH OF THE ONES LEFT ON SCREEN ARE GREYED. This is the hook `apply_settings_dict` calls when it has finished pouring a settings file in, and a file that sets `cell_channel` has to bring the cell settings back on screen with it. A reason written beside a control on a hidden row is a reason nobody can read.

### lines 10085-10087

```python
if any('paired_data' in rule.get('sources', ())
```

Only scanned when a rule on this panel can actually read it. It opens the loaded CSVs, and doing that on every combo change of a panel with no data-dependent rule is a stall for nothing.

## SettingsWidgets._show_the_value_it_will_have

### lines 10130-10137

```python
if getattr(self, "_applying_settings", False):
```

NOT WHILE A SETTINGS FILE IS BEING POURED IN. `apply_settings_dict` sets one widget at a time, so `inference` may still hold the old value when `analysis_mode` arrives -- and forcing then would overwrite the file's value from an inference that is about to change. Caught by loading a file carrying inference='auto' and analysis_mode='guide_permutation': the mode was clobbered to 'regression' before 'auto' had landed. The refresh that runs once the whole dict is applied does the right thing.

## SettingsWidgets._object_visibility_keys

### lines 10154-10156  _(unsure)_

```python
def _object_visibility_keys(self) -> set:
```

A setting is visible when its object is in the run

### lines 10175-10177

```python
wanted.update(f"{role}_{name}"
```

The type narrows a slot, the diameter decides which way a size-split type narrows it, and the morphology is the answer for a slot left on 'custom'.

## SettingsWidgets.refresh_object_visibility

### lines 10258-10264

```python
if getattr(self, "_applying_settings", False):
```

NOT WHILE A SETTINGS FILE IS BEING POURED IN. `apply_settings_dict` sets one widget at a time, so a channel may already hold its new value while the type beside it still holds the old one; hiding rows against that half-applied panel would show a slot narrowed to the wrong morphology and then narrow it again. The bulk apply calls `_refresh_setting_dependencies` when it is finished, which is where this runs instead.

### lines 10270-10279

```python
hidden = set(hidden) | set(
```

BEFORE THE ROWS MOVE, so the guard installed below judges each row against the answer this pass is applying rather than the last one -- otherwise showing a row whose channel was just typed would look, to the guard, like something else putting a hidden row back. AND THE ROWS THE GRID SPEAKS FOR. Kept in a set of its own because this pass recomputes `hidden` from scratch every time: putting the grid's keys into `_hidden_by_the_run` would show them again on the next channel edit. Union, so a row hidden for either reason stays hidden.

### lines 10283-10286

```python
lay_out = getattr(self, "rows_are_laid_out_by", None)
```

BEFORE THE ROWS MOVE, for the other reason too: a row the screen left unbuilt because this rule hid it has to exist before the rule can show it, or `_set_row_visible` would put a bare field on screen in no layout at all.

## SettingsWidgets._slot_headings

### lines 10321-10324

```python
cached = getattr(self, "_slot_heading_cache", None)
```

AN EMPTY ANSWER IS NOT CACHED. The first pass is scheduled from `build_sections`, and on a model built for its values rather than for a screen there are no sections to find at all -- caching that would answer "no headings" for the life of the panel.

### lines 10331-10334

```python
declared = getattr(self, "_section_rows", None)
```

WHAT THE PANEL DECLARED, when it declared anything. The walk below recovers the same two facts from the rendered form, at the cost of a `findChildren` per heading; a panel that said what it was building has already answered. See :meth:`remember_section_rows`.

## SettingsWidgets._hide_the_headings_of_slots_the_run_lacks

### lines 10407-10410

```python
if not section.isHidden():
```

EVERY PASS, not only the first: the settings search puts a heading back whenever its filter is released, and a method that only hid one it had not hidden before would hide it once and never again.

### lines 10417-10419

```python
if maturity_is_visible(section.maturity()):
```

ONLY WHAT MATURITY WOULD ALSO SHOW. A heading this hid may since have been hidden again as Alpha or Beta, and putting a slot back must not overrule Preferences.

### line 10423  _(unsure)_

```python
emptied.pop(ident, None)
```

The section went away with the screen that owned it.

## SettingsWidgets._shown_against_the_rule

### lines 10459-10462

```python
return
```

NOTHING TO RE-ASSERT. The rule is applied once, when the panel is built; a row shown afterwards by the search releasing its filter is meant to stay shown. Re-queueing a pass here is what turned one keystroke into a walk of the whole form.

## SettingsWidgets._set_row_visible

### lines 10493-10494  _(unsure)_

```python
for _ in range(3):
```

Three steps is the deepest the panel nests a field: field, the button holder, the section body that owns the form.

### lines 10506-10511

```python
if widget.parentWidget() is None:
```

NOT UNTIL THE SCREEN HAS TAKEN THE WIDGET. `SettingsWidgets` is built with no parent by everything that wants the values rather than a form, and a parentless widget shown here would not be a row coming back -- it would be a window of its own, opened and painted on the next turn of the event loop, mid-construction and long after the panel that made it was finished with.

### lines 10514-10517

```python
widget.setVisible(visible)
```

There is a widget but no row yet: the screen builds the label and the form after `build_sections` hands the rows back. Hide the field so the panel is not a frame late; the scheduled pass takes the label once the row has one.

## SettingsWidgets._connect_object_visibility_signals

### lines 10550-10552

```python
recommended = self._organelle_recommendations(role)
```

A rebuilt panel can arrive already holding a preset. Remember only recommendations its widgets still equal; differing values are explicit overrides and diameter must leave them alone.

## SettingsWidgets.apply_organelle_presets_from_mapping

### lines 10707-10711

```python
if (self.set_value_for_key(key, value)
```

An imported slot may not name a channel yet, so its dependent controls are deliberately absent. Preserve the preset in the same off-form defaults that preserve explicit imported values; activating the channel later then builds the correct morphology and method.

## SettingsWidgets._read_widget

### lines 10743-10756

```python
if idx >= 0 and w.itemText(idx) == w.currentText():
```

EVERY item is added with userData=opt, including the Python None option (`addItem("None" if opt is None else str(opt), userData=opt)`). So currentData() returning None means the chosen option IS None -- not that the item carries no data. The old fallback to currentText() therefore handed back the STRING 'None', which is how every Qt run shipped strict_errors='None' and turned strict error handling silently ON, since errors.strict_errors() saw a non-None value and took bool('None') == True. cov_type and 'transform' reached statsmodels the same way.

currentText() is still right for an EDITABLE combo showing something the user typed that is not in the list -- detected by the displayed text not matching the current item's text.

## _sibling_label_for

### lines 10855-10858

```python
if field.property(DISABLED_REASON_TOOLTIP):
```

A disabled-reason tooltip is control state, not descriptive setting help.  It belongs on the disabled control by design (and is guarded by a dedicated test), so callers looking for label-paired *help* must not classify it as an ordinary field tooltip waiting to be moved.

### lines 10936-10941

```python
candidate_index = index - 1
```

The item IMMEDIATELY before it, and nothing further back. A row of several controls has labels belonging to each of them, and scanning backwards past an intervening control pairs an editor with the previous setting's name -- or, in the preview panels, with the "drop a folder here" placeholder that happens to sit first in the row.

### lines 10943-10945

```python
if layout.stretch(candidate_index) > 0:
```

A stretching label is row status or a path placeholder, not the fixed caption of the editor after it.  Measure's preview path was otherwise paired with the adjacent maximum-set spin box.

## _sibling_label_for._named

### lines 10870-10878

```python
"""The real label inside a row, unwrapping a host if there is one.
```

THE NAME MAY BE INSIDE A HOST. `Section.add_row` wraps the caption in a `SettingLabelWithInfo` whenever the row wants it right-aligned against its field, which is the settings form's normal shape -- so the layout hands back a plain `QWidget` and the `isinstance` below rejected it. Measured on Mask: 1,541 of 1,657 rows kept their help on the field for this reason alone, and 13 labels had it. `_unwrap_setting_label` is the existing answer to "what is the real label in there"; it returns the widget unchanged when there is no host to unwrap.

### lines 10887-10896

```python
if widget is None or not _widget_is_alive(widget):
```

ALIVE FIRST. This walks layout items, and a layout can hand back an item whose widget has been deleted on the C++ side -- a row that rebuilt itself, a screen torn down while a queued `_on_arrival` was still pending. Touching that wrapper is a dangling pointer, and it does not raise: a full tests/qt sweep on 2026-09-08 SEGFAULTED here, taking the whole process with it, in `retarget_field_tooltips` <- `app_screen._translate` <- `_on_arrival`. A crash on screen arrival is the worst failure mode this form has, because it takes the application rather than the tooltip.

### lines 10906-10907  _(unsure)_

```python
return None
```

Deleted between the check above and this line, which is a real ordering on a queued slot.

### lines 10909-10913

```python
try:
```

A label with a pointing hand is this repository's convention for "this text is clickable" -- AiToggleLabel, _ClearFiguresLabel, the console's copy glyph. Such a label is a CONTROL sharing the row, not the name of the editor beside it, and its own tooltip explains itself rather than its neighbour.

## retarget_field_tooltips

### lines 11001-11008

```python
event_filter = getattr(root, "_api_tooltip_filter", None)
```

Hand-built settings panels used to stop at moving the native Qt tooltip string.  That kept editors quiet, but it left those panels outside the shared tooltip contract: a native platform tooltip cannot be entered, does not share HoverTooltip's single rounded surface, and may disappear while its text is being read.  Keep one filter alive on the owning root and route every successfully paired label through the same popup used by AppScreen.  `_ApiTooltipFilter` does not require API metadata; with a plain authored tooltip it simply displays `apiTooltipHtml` verbatim.

### lines 11030-11033

```python
continue
```

Two settings cannot share one name, so this pairing is wrong. Leave the help where it is: clearing it here is how 80 settings ended up with no help anywhere, which is a worse defect than the one this pass exists to fix.

### lines 11064-11069

```python
label.setProperty("settingHelpLabel", True)
```

This widget now OWNS setting help. ``install_api_tooltips`` later discovers fields by ``settingKey``; without this mark it also rediscovers these labels as though they were editors. On a compact grid that second pass pairs each label with the label to its left, overwriting two UMAP help strings and leaving their real labels empty.

### lines 11071-11074

```python
for prop in ("settingsAppKey", "settingKey",
```

THE LABEL HAS TO CARRY THE SETTING'S IDENTITY, or the language pass cannot refresh the help it now owns: `refresh_api_tooltips` skips any widget without both of these, so a translated caption would leave the old wording on the name.

### lines 11083-11093

```python
field.setProperty("apiTooltipDisplayRole", "metadata")
```

AND THE FIELD HAS TO BE MARKED QUIET, or the move is undone the next time anything refreshes.

This is what made every previous attempt at this look fixed and then not be: the language pass runs on arrival -- a queued call, so it lands AFTER the panel is built -- walks every widget with a `settingKey`, and re-applies the html to whatever it finds. A field with no display role defaults to "tooltip" and was tipped straight back. "metadata" is the existing word for "this widget keeps the metadata but says nothing on hover", and `refresh_api_tooltips` already honours it.

## SettingsWidgets._essentials_that_follow_their_object

### added 2026-09-19 (431)

```python
_APP_ESSENTIALS_THAT_FOLLOW_THEIR_OBJECT: Dict[str, Tuple[str, ...]] = {
```

GitHub issue #120 (jak18015, 1.5.0.8, macOS): "when defining a channel number for pathogen, no pathogen segmentation list appears in the settings." Measured on nightly 5b9207b21 and on v1.5.0.8 alike by typing a pathogen channel on a built Mask screen and pressing Enter: under "All settings" Pathogen Segmentation appeared; under "Essentials", which is the level every module opens at until its user changes it, it never did. Essentials for Mask was the inputs and the workflow switches, a fixed list, so no object's segmentation was in it at all.

Now each of the four `<Object> Segmentation` categories joins Essentials for every object whose channel names a plane -- read from the widgets on each call, so a channel typed after the form opened counts. Cell included: its rows are never hidden by the object rule, but a cell channel is still the user saying "segment cells". The module-level `essential_keys()`, which the walkthrough counts as "the settings this module cannot run without", is unchanged; only the screen's model method adds these. Cost on Mask: 0.4 ms per call, against 7 ms for the static part it sits beside.

### changed 2026-09-19 (431, from review): Timelapse

```python
"timelapse": ("@Cell Segmentation", "@Nucleus Segmentation",
```

Found in review of the Mask fix. Timelapse is the only other module with `<Object> Segmentation` categories (checked across every module in `_APP_CATEGORY_SPECS`). It has the same channel switches and also opens at Essentials, and it had the same defect: typing 2 into `pathogen_channel` and pressing Enter added nothing to the visible keys and left Pathogen Segmentation hidden, while All settings showed it. Timelapse now has the same four entries. As on Mask, "Organelle Segmentation (advanced)" is left out. `test_a_pathogen_channel_brings_pathogen_segmentation_into_essentials` and `test_clearing_the_channel_takes_the_heading_away` run on both modules, and the Timelapse cases fail without this entry.

## SettingsWidgets.refresh_object_visibility

### added 2026-09-19 (431)

```python
refilter = getattr(self, "rows_are_filtered_by", None)
```

This pass shows every row its objects allow, and it ran AFTER the settings search had applied Essentials, so on Mask the first pass after a build put `resume` and `dry_run` back on the Essentials form and a committed channel could show rows the level excludes. The screen now hands in `AppScreen._refilter_the_settings_search`, called at the end of each successful pass. `_hidden_by_their_object` is kept apart from the per-object table's keys so the search strip can tell "this run has no pathogen" from "the table shows this".

## SettingsWidgets._hide_the_headings_of_slots_the_run_lacks

### added 2026-09-19 (431)

```python
def absent(role) -> bool:
```

Extended from organelle slots to nucleus and pathogen. Clearing a pathogen channel hid its rows and left "Pathogen Segmentation" on the form as a heading over nothing, at either level. The heading is hidden and recorded exactly as a slot heading is, and `AppScreen.refresh_maturity_visibility` now leaves a recorded heading hidden -- before, it re-showed any rendered heading whose maturity was visible, and the next object pass hid it again.
