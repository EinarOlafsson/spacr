# Notes from `spacr/qt/screens/app_screen.py`

Prose lifted out of `spacr/qt/screens/app_screen.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (16 entries)
- [ModuleHeader.__init__](#moduleheader__init__) (2 entries)
- [module_maturity](#module_maturity) (1 entry)
- [_absorb_registered_app_metadata](#_absorb_registered_app_metadata) (1 entry)
- [_translate_legacy_setting_keys](#_translate_legacy_setting_keys) (2 entries)
- [_LateCaptionTranslator._on_arrivals_in](#_latecaptiontranslator_on_arrivals_in) (1 entry)
- [_LateCaptionTranslator._on_arrival](#_latecaptiontranslator_on_arrival) (2 entries)
- [_LateCaptionTranslator._translate](#_latecaptiontranslator_translate) (3 entries)
- [_WrappingButtonStrip.__init__](#_wrappingbuttonstrip__init__) (1 entry)
- [AppScreen](#appscreen) (7 entries)
- [AppScreen.__init__](#appscreen__init__) (27 entries)
- [AppScreen._install_ambient](#appscreen_install_ambient) (5 entries)
- [AppScreen._discard_orphan_ambient](#appscreen_discard_orphan_ambient) (1 entry)
- [AppScreen._remove_ambient](#appscreen_remove_ambient) (1 entry)
- [AppScreen.refresh_ambient_background](#appscreenrefresh_ambient_background) (2 entries)
- [AppScreen.changeEvent](#appscreenchangeevent) (3 entries)
- [AppScreen._retheme_section_explainers](#appscreen_retheme_section_explainers) (1 entry)
- [AppScreen.page_fill](#appscreenpage_fill) (1 entry)
- [AppScreen._sync_page_palette](#appscreen_sync_page_palette) (5 entries)
- [AppScreen._clear_page_surfaces](#appscreen_clear_page_surfaces) (3 entries)
- [AppScreen._lay_out_the_settings_panel](#appscreen_lay_out_the_settings_panel) (17 entries)
- [AppScreen._mount_the_object_grid](#appscreen_mount_the_object_grid) (3 entries)
- [AppScreen.apply_object_grid_preference](#appscreenapply_object_grid_preference) (1 entry)
- [AppScreen._section_holds_anything.already_built_rows](#appscreen_section_holds_anythingalready_built_rows) (1 entry)
- [AppScreen._section_holds_anything.holds_an_active_slot](#appscreen_section_holds_anythingholds_an_active_slot) (1 entry)
- [AppScreen._restore_settings_section](#appscreen_restore_settings_section) (1 entry)
- [AppScreen._watch_the_settings_that_decide_the_form](#appscreen_watch_the_settings_that_decide_the_form) (1 entry)
- [AppScreen._rebuild_the_form](#appscreen_rebuild_the_form) (3 entries)
- [AppScreen._build_settings_section](#appscreen_build_settings_section) (15 entries)
- [AppScreen._lay_out_the_rows_that_are_back](#appscreen_lay_out_the_rows_that_are_back) (1 entry)
- [AppScreen._lay_out_one_waiting_row](#appscreen_lay_out_one_waiting_row) (1 entry)
- [AppScreen._the_rows_moved](#appscreen_the_rows_moved) (4 entries)
- [AppScreen.the_name_carries_the_help](#appscreenthe_name_carries_the_help) (2 entries)
- [AppScreen._lay_out_setting_row](#appscreen_lay_out_setting_row) (11 entries)
- [AppScreen._install_section_explainer](#appscreen_install_section_explainer) (10 entries)
- [AppScreen._refresh_model_explainer](#appscreen_refresh_model_explainer) (2 entries)
- [AppScreen._gene_tile_entry](#appscreen_gene_tile_entry) (1 entry)
- [AppScreen._with_a_plate_map](#appscreen_with_a_plate_map) (2 entries)
- [AppScreen._choose_a_model_for](#appscreen_choose_a_model_for) (1 entry)
- [AppScreen._with_a_settings_advisor](#appscreen_with_a_settings_advisor) (1 entry)
- [AppScreen.settings_for_my_data](#appscreensettings_for_my_data) (2 entries)
- [AppScreen._console_folded](#appscreen_console_folded) (2 entries)
- [AppScreen._reading_with_the_last_run](#appscreen_reading_with_the_last_run) (1 entry)
- [AppScreen._install_example_data_button](#appscreen_install_example_data_button) (4 entries)
- [AppScreen.reanchor_example_paths.rehome_text](#appscreenreanchor_example_pathsrehome_text) (3 entries)
- [AppScreen.apply_settings_that_came_with](#appscreenapply_settings_that_came_with) (1 entry)
- [AppScreen.load_the_screen_data](#appscreenload_the_screen_data) (1 entry)
- [AppScreen.load_the_measure_example](#appscreenload_the_measure_example) (1 entry)
- [AppScreen._put_the_measure_example_in_place](#appscreen_put_the_measure_example_in_place) (3 entries)
- [AppScreen.load_the_annotate_example](#appscreenload_the_annotate_example) (1 entry)
- [AppScreen._apply_the_example_settings](#appscreen_apply_the_example_settings) (2 entries)
- [AppScreen.load_the_example_images](#appscreenload_the_example_images) (1 entry)
- [AppScreen.load_the_example_images.ask](#appscreenload_the_example_imagesask) (1 entry)
- [AppScreen._put_the_example_images_in_place](#appscreen_put_the_example_images_in_place) (2 entries)
- [AppScreen.load_the_example_screen](#appscreenload_the_example_screen) (4 entries)
- [AppScreen.refresh_maturity_visibility](#appscreenrefresh_maturity_visibility) (1 entry)
- [AppScreen._install_dimension_switches](#appscreen_install_dimension_switches) (3 entries)
- [AppScreen.set_dimension](#appscreenset_dimension) (1 entry)
- [AppScreen._dimension_rows](#appscreen_dimension_rows) (1 entry)
- [AppScreen._attach_column_picker](#appscreen_attach_column_picker) (2 entries)
- [AppScreen._build_empty_state_banner](#appscreen_build_empty_state_banner) (7 entries)
- [AppScreen.choose_source_folder](#appscreenchoose_source_folder) (1 entry)
- [AppScreen._open_demos_menu](#appscreen_open_demos_menu) (1 entry)
- [AppScreen.eventFilter](#appscreeneventfilter) (7 entries)
- [AppScreen._default_hint](#appscreen_default_hint) (1 entry)
- [AppScreen._build_runtime_panel](#appscreen_build_runtime_panel) (93 entries)
- [AppScreen._remember_runtime_splitter](#appscreen_remember_runtime_splitter) (1 entry)
- [AppScreen._write_hint](#appscreen_write_hint) (4 entries)
- [AppScreen._sync_category_hint_height](#appscreen_sync_category_hint_height) (1 entry)
- [AppScreen._watch_for_late_captions](#appscreen_watch_for_late_captions) (1 entry)
- [AppScreen._wire_category_hints](#appscreen_wire_category_hints) (1 entry)
- [AppScreen.show_category_hint](#appscreenshow_category_hint) (1 entry)
- [AppScreen.hideEvent](#appscreenhideevent) (1 entry)
- [AppScreen._on_run](#appscreen_on_run) (14 entries)
- [AppScreen._announce_the_fit](#appscreen_announce_the_fit) (2 entries)
- [AppScreen._on_copy_console](#appscreen_on_copy_console) (1 entry)
- [AppScreen._on_pipeline_error](#appscreen_on_pipeline_error) (3 entries)
- [AppScreen._on_lp_switch](#appscreen_on_lp_switch) (2 entries)
- [AppScreen._on_hyperparam_switch](#appscreen_on_hyperparam_switch) (1 entry)
- [AppScreen._on_sweep_switch](#appscreen_on_sweep_switch) (1 entry)
- [AppScreen._on_umap_gpu_switch](#appscreen_on_umap_gpu_switch) (1 entry)
- [AppScreen._on_ai_switch](#appscreen_on_ai_switch) (1 entry)
- [AppScreen._wanted_provider](#appscreen_wanted_provider) (1 entry)
- [AppScreen._on_explain_error](#appscreen_on_explain_error) (1 entry)
- [AppScreen._on_file_issue](#appscreen_on_file_issue) (5 entries)
- [AppScreen._on_file_issue._file](#appscreen_on_file_issue_file) (1 entry)
- [AppScreen._on_figure_ready](#appscreen_on_figure_ready) (2 entries)
- [AppScreen.closeEvent](#appscreencloseevent) (9 entries)
- [AppScreen._on_finished](#appscreen_on_finished) (5 entries)
- [AppScreen._record_run_in_runs_tab](#appscreen_record_run_in_runs_tab) (1 entry)
- [AppScreen._pin_regression_graph](#appscreen_pin_regression_graph) (3 entries)
- [AppScreen._open_live_tile](#appscreen_open_live_tile) (3 entries)
- [AppScreen._live_tile_menu](#appscreen_live_tile_menu) (2 entries)
- [AppScreen._pinned_menu](#appscreen_pinned_menu) (1 entry)
- [AppScreen._show_publication_sheet](#appscreen_show_publication_sheet) (1 entry)
- [AppScreen._measurements_destination](#appscreen_measurements_destination) (1 entry)
- [AppScreen.open_run_beside](#appscreenopen_run_beside) (2 entries)
- [AppScreen._raise_the_results_tab](#appscreen_raise_the_results_tab) (1 entry)
- [AppScreen._on_runs_removed](#appscreen_on_runs_removed) (2 entries)
- [AppScreen._on_loaded_run_changed_refresh_tabs](#appscreen_on_loaded_run_changed_refresh_tabs) (2 entries)
- [AppScreen._on_results_tab_changed](#appscreen_on_results_tab_changed) (1 entry)
- [AppScreen._scan_source_frame](#appscreen_scan_source_frame) (2 entries)
- [AppScreen._show_trial](#appscreen_show_trial) (8 entries)
- [AppScreen._on_trial_loaded](#appscreen_on_trial_loaded) (4 entries)
- [AppScreen._pictures_from](#appscreen_pictures_from) (1 entry)
- [AppScreen._load_trial_figures](#appscreen_load_trial_figures) (1 entry)
- [AppScreen._figure_grid_menu](#appscreen_figure_grid_menu) (1 entry)
- [AppScreen._on_guide_selected](#appscreen_on_guide_selected) (1 entry)
- [AppScreen._on_pipeline_result](#appscreen_on_pipeline_result) (6 entries)
- [AppScreen._load_regression_results](#appscreen_load_regression_results) (6 entries)
- [AppScreen._clear_thread_refs](#appscreen_clear_thread_refs) (1 entry)
- [AppScreen._on_stop](#appscreen_on_stop) (2 entries)
- [AppScreen._force_stop](#appscreen_force_stop) (1 entry)
- [AppScreen._on_import_settings](#appscreen_on_import_settings) (1 entry)
- [AppScreen._load_settings_csv](#appscreen_load_settings_csv) (1 entry)
- [AppScreen.apply_settings_dict](#appscreenapply_settings_dict) (3 entries)
- [AppScreen._refresh_after_bulk_apply](#appscreen_refresh_after_bulk_apply) (1 entry)
- [AppScreen._bulk_apply_changes_form_shape](#appscreen_bulk_apply_changes_form_shape) (1 entry)
- [AppScreen._sync_folded_switches](#appscreen_sync_folded_switches) (1 entry)
- [AppScreen._migrate_control_wells](#appscreen_migrate_control_wells) (2 entries)
- [AppScreen._apply_each_setting](#appscreen_apply_each_setting) (1 entry)
- [AppScreen._apply_value](#appscreen_apply_value) (3 entries)
- [AppScreen._refresh_usage](#appscreen_refresh_usage) (1 entry)
- [AppScreen._apply_usage](#appscreen_apply_usage) (1 entry)
- [_sample_usage](#_sample_usage) (1 entry)

## Module level

### lines 160-161

```python
register_widget_qss(SETTINGS_PANEL_NAME, _settings_panel_qss, replace=True)
```

``replace=True``: this module owns the name, and a reimport must re-register rather than raise and leave every module screen unstyled.

### lines 171-194

```python
DIMENSION_TOGGLES = (
```

A SETTING APPEARS WHEN THE DATA HAS THE DIMENSION IT IS ABOUT

Two switches sit in the action row immediately left of Live: 3D and Time. Neither runs anything. Each says which dimension the plate actually has, and the settings that only mean something in that dimension appear with it. A plate of single-plane fields has no z axis, so "how far apart are two planes" is a question about nothing, and a form that asks it anyway is a form the user has to learn to ignore.

THEY ARE STATES, so they stay lit while on, exactly as the checkable fold switches on Mask Generation do. They do NOT share those switches' second half: a fold switch also sets its module's pipeline gate, because a folded module has no control for that gate anywhere else. These two do nothing to the run. `z_stack` and `t_stack` are ordinary controls INSIDE the categories they reveal, and those controls are what say whether the run uses the dimension -- so a switch that also set them would be the second source of truth this project keeps finding and removing.

HIDDEN, NOT DELETED. A hidden row keeps its widget and its value: `SettingsWidgets.collect` walks `_widgets`, never the visible rows, so a volumetric answer typed before the switch went off is still in the dict handed to the run and still written to the settings CSV.

### lines 426-428

```python
HINT_STRIP_LINES = 4
```

The hover description must never reflow the runtime controls. Four lines are enough to scan the curated setting descriptions while the full rich tooltip remains available beside the field.

### lines 473-475

```python
CATEGORY_STRIP_LINES = 3
```

The category strip sits above it and holds a shorter blurb, so three lines is enough. Fixed, for the same reason: the runtime controls above must not jump when the pointer crosses a category header.

### lines 524-533

```python
SECTION_HINTS = CATEGORY_TOOLTIPS
```

One blurb per settings CATEGORY, keyed by the uppercased category title. The table itself lives beside the category map in `settings_model`, because that is what decides which categories exist; this module only renders them. Re-exported under the historical name so integrations and tests that read `app_screen.SECTION_HINTS` keep working.

The blurbs are shown in the strip UNDER the Run / Stop actions row (see `_build_runtime_panel` and `_wire_category_hints`), not as a popup over the form: a category description is three lines long and a floating tooltip covers the very settings it is describing.

### lines 537-544

```python
COLUMN_TABLES = {
```

Settings whose VALUE is the name of a database column. Each gets a "SQL" button that opens the run's measurements.db read-only and shows what is actually in it, so a typo cannot silently create a second near-identical column. The value is the table to preselect; None lets the user choose.

dependent_variable is deliberately absent: it names a column of the score CSV, not of measurements.db, and pointing the picker at the wrong file would be worse than having no picker at all.

### lines 548-553

```python
"classes":            "png_list",
```

`classes` is a COMPOSITE, and its button goes on the column combo inside it -- see `_attach_column_picker`. It is here because it is a setting that names a column, and the whole point of this table is that such a setting should never have to be typed blind. Without it the one setting that decides every class in the module was the one setting with no way to fill it in when no table had been loaded yet.

### lines 555-556

```python
"measurement":        None,
```

custom_measurement is gone: it was collected and never read, so a SQL column picker for it offered to fill in a control that did nothing.

### lines 656-663

```python
"barcode_qc":      "Barcode QC",
```

FOLDED MODULES THAT STILL OPEN THIS SCREEN. Barcode QC, AnnData Export and Illumination Correction have no screen class of their own every knob each has is a registered settings key, so the generic form IS the module -- and all three are now pages on a host rather than tiles. They used to reach this table through `register_app(..., title=...)`; with the row gone that push never happens, and the page would be headed "Barcode_Qc" with no sentence under it.

### lines 666-667  _(unsure)_

```python
"illumination":    "Illumination Correction",
```

Not "Illumination": the module corrects the field, and the heading over its form is what says so.

### line 672  _(unsure)_

```python
APP_INTROS = {
```

Short "what this module does" blurbs shown to the right of the header.

### line 709  _(unsure)_

```python
"barcode_qc":      "Evaluate a completed barcode-mapping run using read depth per well, low-depth...
```

The two folded modules whose page is this screen — see APP_TITLES.

### lines 721-722

```python
pass
```

Discovery records individual failures. Metadata lookup must not prevent the built-in AppScreen class from importing.

### lines 1264-1266

```python
"measure": "Input & Experiment",
```

Measure's example data is the MASK OUTPUT, not raw acquisition: the merged arrays with their label masks, so Measure can be run end to end without segmenting anything first.

### lines 1268-1281

```python
"classify": "Plate Sources & Workflow",
```

Classify's example data is the MEASURE output plus real labels: 2,341 crops of which 88 are annotated. Unlabelled crops would exercise the viewer and nothing else -- a training example needs labels.

ABOVE src, in the section that names the sources, asked for on 2026-09-01. It had sat under Labels & Classes, which is where the labels it brings are configured but not where the path it sets is so the control that fills `src` was two sections away from `src`. BOTH KEYS, and `classify_merged` is the one that matters: it is the module in the Core section that a user actually opens. `classify` is not in APPS and is not folded onto any host, so it is unreachable from the UI -- which means this entry named only the dead key and the Classify screen has never shown an example-data control at all. Found on 2026-09-01 while checking which core modules have test data.

### lines 1284-1286

```python
"map_barcodes": "Sequencing Input",
```

Map Barcodes reads FASTQ, and the example is the paper's own: NCBI BioProject PRJNA1261935, the four sequenced plates. Above `src`, in the section that names it.

## ModuleHeader.__init__

### lines 394-396

```python
blurb = ApiHelpLabel(str(description), str(app_key or ""),
```

Straight onto the header row: the blurb used to share a nested row with the dot, and a container for one widget is a container for nothing.

### lines 400-401

```python
blurb.setWordWrap(False)
```

One line, flush left. The label may shrink below its ideal width so a long blurb never forces the window wider.

## module_maturity

### lines 588-592

```python
from ..widgets.fold_strip import folded_fallback
```

Every host's table, not one host's: the modules folded into the segmentation workbench keep what their tiles said in `make_masks.FOLD_FALLBACK`, and asking Map Barcodes about them answers "" -- which reads as stable and drops the beta mark off the Cellpose Workbench's settings sections.

## _absorb_registered_app_metadata

### lines 745-748

```python
pull = getattr(app, "registered_metadata", None) if app else None
```

`getattr(..., None)`: `spacr.qt.app` may be half-built when this runs (it imports the widget package before `register_app` exists), in which case there is nothing to pull and the push half of the seam delivers every row later.

## _translate_legacy_setting_keys

### lines 857-860

```python
for key in list(out):
```

AND EVERY RENAME THE VALIDATOR ALREADY KNOWS ABOUT. `RETIRED_SETTINGS` is where a rename is recorded, and it was consulted when a file was CHECKED but not when one was LOADED -- so `spacr-doctor` said "renamed to X" about the very file the panel had just dropped the value from.

### lines 865-870

```python
value = out.pop(key)
```

A SPLIT IS A TUPLE OF NAMES, NOT A NAME. `control_wells` became `stain_baseline_wells` AND `analysis_excluded_wells`, and passing the pair straight to `setdefault` stored the value under a TUPLE key -- which no widget reads, so the value was lost exactly the way this function exists to prevent. Both halves get it, which is what the old key meant: a file that set it was setting both at once.

## _LateCaptionTranslator._on_arrivals_in

### line 1001  _(unsure)_

```python
return
```

The host itself went away before the turn came round.

## _LateCaptionTranslator._on_arrival

### lines 1023-1025

```python
self._watch_pages_of(widget)
```

A PAGE STRIP. Its own captions are its tabs, and the pass sets those through `QTabWidget` methods, so the walk starts at what the strip was parented into.

### lines 1031-1032

```python
pass
```

Gone again before the turn came round. Nothing to translate is not a failure.

## _LateCaptionTranslator._translate

### lines 1071-1078

```python
retranslate_widget_tree(widget, only_new=True)
```

ONLY WHAT IS NEW. A module screen is assembled over several event turns and each large container parented in triggers another near-root pass, so the tree was being translated roughly three times over: measured at 13 passes and 23,454 widget visits for one Measure screen, of which three passes were 22,750. The stamp `retranslate_widget_tree` leaves carries the language and the catalog generation, so a language change or a newly catalogued row still reaches every widget.

### lines 1080-1086

```python
from .settings_model import retarget_field_tooltips
```

AND THE HELP GOES BACK ONTO THE NAMES. The pass above walks every widget carrying a `settingKey` and re-applies its tooltip, which is what kept putting the help back on the field: this runs on ARRIVAL, so it lands after the panel was built and after any earlier move. Doing it here, immediately after, means a row that arrives late is treated exactly like one that was there from the start.

### lines 1091-1092

```python
pass
```

The panel was closed again before the pass ran. Nothing to translate is not a failure.

## _WrappingButtonStrip.__init__

### lines 1415-1417

```python
self._gap = int(spacing)
```

``FlowLayout`` keeps its gap in a private attribute and never calls ``setSpacing``, so ``QLayout.spacing()`` would answer the style's default rather than this one. ``sizeHint`` needs the real number.

## AppScreen

### lines 1480-1482

```python
error_explain_requested = Signal(str, str)
```

Emitted when the user clicks "Explain error" with the last captured traceback + the app key so MainWindow can route to the AI Console.

### lines 1484-1486

```python
remote_submit_requested = Signal(str, dict)
```

Hand an immutable settings snapshot to the Distributed Jobs screen. MainWindow owns navigation, so the reusable screen does not reach into the application stack itself.

### lines 1489-1492

```python
_ambient = None
```

Backdrop state, declared on the class so a Qt event that arrives mid-construction (showEvent is delivered from inside a nested layout activation on some styles) finds an answer rather than an AttributeError.

### lines 7634-7642

```python
HEARTBEAT_SCHEDULE = (30, 60, 120, 300, 600, 1200, 1800, 2700, 3600)
```

A long fit says it will be long, and says where it has got to (140)

Reported 2026-08-18, twice, while a fit was running correctly: "im running the mixed model now and it is taking much longer than before is that normal?" ... "it is still going, cpu at 100 percent". AN HOUR OF SILENCE AT 100% CPU IS INDISTINGUISHABLE FROM A HANG, and that is the whole of the report -- the run was healthy and had no way to say so.

### lines 9058-9072

```python
MAX_LIVE_RUNS = 2
```

Two runs on screen at once -- deliberate, and BOUNDED (116)

"every regression run should have its own interactive volcano plot". The state half shipped in d4113297: each run keeps its level, its colouring, its axis pins, its effect cut and its selection, and gets them back. This is the other half, and the bound is the substance of it rather than a caveat on it.

WHY THE ANSWER IS NOT "N LIVE VOLCANOES". 129 measured live pyqtgraph tiles at 74.99 ms per window-drag frame against 5.19 ms for photographs, on a 16.7 ms budget. Two runs is what a comparison needs; twelve is what makes the screen unusable, and a user who discovers the bound by their machine stopping has been told nothing.

### lines 9830-9841

```python
SETTINGS_AS_TABS = frozenset()
```

EMPTY, AND THE MECHANISM STAYS. Measure was the one screen filed here, on 2026-08-19: "in measure i dont like the black categories. can we make them into measurement subtabs?" Reversed on 2026-08-23 "the measure module settings categories are for some reason in Tabs that seem like they are cut off half way when opened. please fix this, make it normal, or the same structure as the other core modules like mask with settings categories."

The set is kept rather than the code deleted: the tabs are a real answer to a real complaint about a wall of categories, and the next screen that grows one can be added here without rebuilding it. What was wrong was the tab BAR at Measure's width, not the idea.

### line 10519

```python
_WORKSPACE_PANELS = {
```

instruction 180: the screen contributes, and enrols its panels

## AppScreen.__init__

### lines 1517-1520

```python
self._ambient = None
```

Qt can deliver show/palette events from nested layout activation before this constructor reaches the backdrop section.  Keep those events from installing against a half-built widget tree; the normal install below is the single point where backdrop ownership begins.

### lines 1527-1528  _(unsure)_

```python
self._hint_map: dict = _CaptionsBuiltWhenTheyAreAskedFor(
```

widget → plain-text hint. Walking it hands over every caption, including the ones the object rule left waiting; see the class.

### line 1531, trailing  _(unsure)_

```python
self._html_tip_map: dict = {}
```

widget → HTML tooltip (sticky popup)

### lines 1532-1533

```python
self._model_explainer = None
```

The Model & Inference explainer (instruction 132). Only the regression panel builds one; every other screen leaves it None.

### lines 1535-1538

```python
self._heartbeat = None
```

THE HEARTBEAT (140). Named here rather than created lazily so a screen that never runs anything still answers `_stop_the_heartbeat` `_on_finished` calls it on every module, not only the two that start one.

### lines 1557-1560

```python
ensure_widget_qss_applied(SETTINGS_PANEL_NAME)
```

This module is imported lazily by `app.py`, long after the launch stylesheet was generated, so the block registered above is not in it. Without this the settings column opens unpanelled — see `ensure_widget_qss_applied`.

### lines 1568-1571

```python
header = ModuleHeader(
```

─── Header ─────────────────────────────────────────────────── The shared masthead — see `ModuleHeader`. This screen was where it was written and for a long time where it stayed, which is how twenty-odd screens ended up with a title at body size.

### line 1588  _(unsure)_

```python
self._settings_body = body
```

Settings panel (left)

### lines 1590-1593

```python
self._settings_panel = self._build_settings_panel()
```

THE WIDGET ACTUALLY MOUNTED, which is not `_settings_scroll`: the panel builder wraps the scroll area, so looking the scroll up in the splitter answered -1 and the rebuild returned having done nothing.

### lines 1597-1600

```python
self._form_shape_on_screen = self._form_shape()
```

Record the shape of an ordinary first-open screen too.  Rebuilt screens already get this stamp in ``MainWindow.rebuild_app_screen``; without the matching first-open stamp, merely leaving an unchanged watched field rebuilt the whole page once and stole focus.

### line 1603  _(unsure)_

```python
body.addWidget(self._build_runtime_panel())
```

Runtime panel (right)

### lines 1611-1615

```python
self._wire_live_preview_autoload()
```

The live-preview autoload watches ``src``, and can only be wired once BOTH panels exist: the settings panel owns the src field and the runtime panel owns the preview. It used to be wired from _build_empty_state_banner (inside the settings panel), where ``self._live_preview`` does not exist yet — so it never fired.

### lines 1618-1620

```python
self._wire_category_hints()
```

Same ordering constraint: the sections are built by the settings panel, the strip they describe themselves into belongs to the runtime panel, so the two can only be connected once both exist.

### lines 1623-1631

```python
self._watch_for_late_captions()
```

WHAT ARRIVES AFTER THE LANGUAGE PASS. `MainWindow` translates a screen once, when it builds it; anything parented into the screen afterwards arrives in English and nothing ever asks it again. The declared preview is exactly that -- `spacr.qt.preview_registry` installs it the first time the module is opened, which is after the pass, so on a Swedish screen its whole panel and the toggle it puts on the settings strip stay English, and so does every page a fold button opens. Watching the hosts they land in translates each new subtree as it arrives.

### lines 1634-1642

```python
self._usage_jobs = JobRunner(self, app_key=f"{self.app_key} usage",
```

Two runners, not one, and the split is deliberate. Both of these are background work — the usage poll shells out to nvidia-smi, filing an issue shells out to `gh` and then talks to api.github.com — but they run on wildly different clocks. `_refresh_usage` skips a tick while its own sample is still out, so that a machine slow enough to still be inside nvidia-smi 2 s later does not accumulate a backlog. Share a runner with the issue report and that guard also swallows every poll for the up-to-28 s an issue report can take, freezing the usage bars for the whole of it.

### line 1647  _(unsure)_

```python
self._usage_generation = 0
```

Timer to poll RAM/GPU/CPU periodically

### lines 1652-1657

```python
self._thread: Optional[QThread] = None
```

A stacked module page may be constructed hours before the user opens it.  Polling every hidden page wastes a thread and a ``nvidia-smi`` subprocess every two seconds; in a long-lived Qt process those orphan polls also made GPUtil's subprocess boundary eventually segfault.  showEvent starts the one page the user can actually see, and hideEvent stops it again.

### line 1659  _(unsure)_

```python
self._thread: Optional[QThread] = None
```

Threading state

### lines 1662-1664

```python
try:
```

Drag & drop — install a dropzone with this app's per-module handler. Universally accepts settings CSVs; folder policy is app-specific (see spacr.qt.dnd_handlers).

### lines 1672-1679

```python
if self.app_key in DNA_RAIN_APPS:
```

DNA rain backdrop (sequencing only). Sits behind every other child, takes no focus and no mouse events, and stops its timer whenever this screen is not visible, so it costs nothing while the pipeline runs on another tab. Its colour / speed / visibility / font controls live in a popover behind a DNA button beside the AI toggle — they used to be a permanent bar across the bottom of the page, which is more chrome than a backdrop is worth.

### lines 1683-1688

```python
self._clear_page_surfaces()
```

The rain is lowered behind its siblings, so it is only ever as visible as those siblings are transparent. Under dark and light every container is an opaque `bg` and it was buried completely: the animation ran, cost its frames, and reached the eye only through the few pixels of layout spacing between widgets.

### lines 1690-1693

```python
self._sync_page_palette()
```

The page colour follows whether a backdrop got installed, so it is resolved wherever that is decided -- and it has to reach QPalette.Window, not just paintEvent, or Qt's pre-paint erase still uses `bg` and flashes black. See _sync_page_palette.

### lines 1700-1711

```python
self._backdrops_ready = True
```

Ambient backdrop — the drifting blobs (or whichever theme the user picked) behind every screen that does NOT already animate something of its own. See `uses_ambient_background` for why that is one rule and not an `else` on the branch above; the two are mutually exclusive by construction, so no screen ever carries both.

Same hard-won contract as the rain: lowered behind every sibling, no focus, no mouse events, and its timer stops whenever this screen is not visible — these screens stay open while the pipeline runs on another tab, so an animation that kept ticking off-screen would cost a core for nobody.

### lines 1719-1754

```python
if self._ambient is None:
```

And unconditionally, whatever happened above. This used to run ONLY as a side effect of installing an animation — the DNA rain calls it before, `_install_ambient` after — on the reasoning that a screen with nothing behind it should be left opaque rather than transparent over emptiness.

That reasoning was wrong, and it is what made the settings half of every module screen a solid black rectangle for anybody who had turned the ambient backdrop off in Preferences: `_install_ambient` returns early when the preference is off, so the sweep never ran, so every layout container on the page kept the blanket `QWidget { background-color: bg }` — the WINDOW colour, which no page-opacity setting can reach. Measured over a probe backdrop with the preference off, the settings column, the categories, the gaps between them and the console box all read 0.000: the whole page was one opaque slab and only the cards on top of it looked deliberate.

There is never "nothing behind it". With no animation the thing behind is the window's own `bg`, which is the theme — exactly what the page is supposed to show between the floating category panels. `clear_container_surfaces` is idempotent, so the calls inside the two install paths stay where they are for their own ordering reasons.

SKIPPED ONLY WHEN THE AMBIENT INSTALL ALREADY SWEPT THIS EXACT TREE. `_install_ambient` sweeps AFTER it parents the backdrop, and nothing is added to the screen between it returning and here, so a second sweep would visit the same widgets and reach the same answer -- and it is not free: `clear_container_surfaces` walks every descendant with `findChildren`, which on the Mask screen is about 50 ms of the build. Every other route still sweeps here, and each of them needs to: the preference off, the install failed, or the DNA rain, which sweeps BEFORE it parents its widget and so leaves one container this pass is the only one to see. `_ambient` is set only by an install that got as far as its own sweep, which is what makes it the right question to ask.

### lines 1757-1760

```python
self._sync_page_palette()
```

The page colour follows whether a backdrop got installed, so it is resolved wherever that is decided -- and it has to reach QPalette.Window, not just paintEvent, or Qt's pre-paint erase still uses `bg` and flashes black. See _sync_page_palette.

### lines 1762-1765

```python
try:
```

Instruction 180: enrol with the workspace registry, so a run that finishes can record what this screen had open. LAST, and by callable, so a panel this screen may or may not build is asked for at collection time rather than captured now.

### lines 1770-1773

```python
try:
```

178 D: no overflow arrows. The two ways along a bar the ask named the wheel and the arrow keys -- were driven before this shipped and both work, which is the condition under which removing a control is safe. See `take_the_scroll_arrows_off`.

## AppScreen._install_ambient

### lines 1836-1845

```python
if not self._heavy_lock_is_free():
```

NOT WHILE SOMEBODY IS HOLDING THE HEAVY LOCK. The GPU backdrop takes HEAVY_IMPORT_LOCK to build its GL context, and the startup preloader holds that lock for a whole module import. Installing here regardless is what made opening a module soon after launch freeze the GUI thread: 83% of a measured 3148 ms block was the backdrop's constructor waiting for a lock the preloader held.

The backdrop is decoration and the module is not, so the decoration is what waits. If the lock is busy this comes back on a timer and the screen opens now, undecorated for a moment.

### lines 1858-1868

```python
window = self.window()
```

ONE BACKDROP FOR THE WINDOW, and it is not this screen's. When the central area carries one it is already behind this screen AND behind the dock beside it; building a second here puts two animations over each other, out of step, with the seam showing wherever the two containers meet.

The page surfaces are still cleared, because that is what lets the window's backdrop through this screen's opaque containers. Skipping the install and skipping the clear are different things, and skipping both is a screen with an animation behind it that nobody can see.

### lines 1883-1893

```python
self._ambient = widget
```

THE BACKDROP IS RECORDED BEFORE THE PAGE COLOUR IS RESOLVED, because `page_fill` reads `self._ambient` to decide whether this screen still has to paint a page of its own. Assigning afterwards meant the sync ran against a screen that still looked backdrop-less: it applied the flat page colour and its `AppScreen { background-color: ... }` stylesheet to a widget about to be covered by the animation, and the unconditional sync at the end of `__init__` -- by then seeing the backdrop took both straight off again. The user saw nothing for it and the tree was restyled twice, which on the Mask screen is a full re-polish of 1,538 widgets for a colour that never showed.

### lines 1895-1898

```python
self._sync_page_palette()
```

The page colour follows whether a backdrop got installed, so it is resolved wherever that is decided -- and it has to reach QPalette.Window, not just paintEvent, or Qt's pre-paint erase still uses `bg` and flashes black. See _sync_page_palette.

### lines 1904-1920

```python
try:
```

"NOT YET" IS NOT "NEVER". The spaceout backdrop refuses rather than blocking the GUI thread when a heavy import holds the lock its GL context needs, and the peek above cannot rule that out -- it is a check, not a reservation, and the preloader re-takes the lock between two imports.

Without this the refusal would land in `_ambient_applied` as an attempt already made, and the screen would stay undecorated for the life of the session because a module was opened while the preloader happened to be running. Forgetting the attempt and coming back is what the peek does for the case it can see, so it is what this does too. DEFENSIVELY, because this handler's whole job is that a backdrop can never stop a screen opening -- and "the widget module is absent" is one of the failures it is here to absorb, so asking that module to classify the failure has to tolerate its being the thing that is missing.

## AppScreen._discard_orphan_ambient

### lines 1943-1944

```python
return
```

No class, no way to recognise one — and if the import is what failed, nothing was constructed to leave behind.

## AppScreen._remove_ambient

### lines 1965-1968

```python
self._sync_page_palette()
```

`page_fill` returns a colour only while there is no backdrop, so taking the animation away is exactly the moment this screen becomes responsible for its own page. Without the repaint the Preferences toggle leaves the hole it used to leave for good.

## AppScreen.refresh_ambient_background

### lines 1984-1985

```python
self._remove_ambient()
```

Belt and braces: sequencing must not acquire one through this path either.

### lines 2006-2007

```python
return
```

Nothing changed. Re-applying would restart the animation every time the user switches back to this tab.

## AppScreen.changeEvent

### lines 2055-2061

```python
if not self._backdrops_ready:
```

A nested layout activation can deliver this while ``__init__`` is still building the two page columns.  Skipping only the ambient install is not enough: syncing the page palette here first paints a flat page for a screen that is about to gain a backdrop, then the real install has to withdraw that style again.  The finished build installs/re-themes the backdrop and syncs the page once below, so no palette work is lost by deferring the construction-time event.

### lines 2066-2069

```python
self._retheme_section_explainers()
```

THE EXPLAINER BOXES PAINT WITH PALETTE TOKENS (instruction 144), so a theme switch has to re-render them -- rich text stores the colour it was given, and re-applying the stylesheet cannot reach inside a QTextDocument's character formats.

### lines 2071-2074

```python
self._sync_page_palette()
```

The page colour is resolved at paint time from the live theme, so a theme switch has to ask for a repaint — nothing else on this screen invalidates it. The palette moves with it, or Qt keeps erasing to the OLD theme's page between the two.

## AppScreen._retheme_section_explainers

### line 2104  _(unsure)_

```python
LOG.debug("could not re-theme the %s box", title,
```

A box whose C++ side has gone, on a screen being torn down.

## AppScreen.page_fill

### lines 2190-2194

```python
if (self._ambient is not None or self._dna_rain is not None
```

A backdrop the WINDOW owns counts as "a backdrop is installed". Without this the guard that declines to build a second one would leave `_ambient` None, `page_fill` would return the flat page colour, and the screen would paint that colour straight over the window's animation -- the black slab, reported three times.

## AppScreen._sync_page_palette

### lines 2230-2235

```python
if getattr(self, "_syncing_page", False):
```

Re-entrancy guard, and it is not theoretical: `setPalette` posts a `PaletteChange`, `changeEvent` handles `PaletteChange` by calling this method, and the second call sets the palette again. That recursed until the stack ran out -- a core dump on startup, not a flicker. The flag makes the nested call a no-op; the outer one finishes the work.

### lines 2239-2241

```python
applied = getattr(self, "_page_applied", "unset")
```

Idempotent, so the re-polish a stylesheet change triggers cannot turn into a repaint loop of its own on a screen that is already showing the right colour.

### line 2250  _(unsure)_

```python
self.setAutoFillBackground(False)
```

Back to whatever the stylesheet and the app palette say.

### lines 2253-2257

```python
set_a_sheeted_widgets_own_rule(self, "")
```

NOT `setStyleSheet("")`. Under per-screen sheeting this widget may be carrying the whole window sheet, and clearing it strands the screen with no theme at all measured as `#000000` text on the dark theme, for the life of the screen. Owning no rule is what is meant here.

### lines 2264-2278

```python
set_a_sheeted_widgets_own_rule(
```

And in the screen's OWN stylesheet, which is what makes this stick. `autoFillBackground` is not ours to hold: the surface sweep and the theme passes both walk this tree setting it, screens are built and re-themed in an order that is not fixed, and more than one AppScreen is alive during startup. Whoever runs last wins, and when the loser was this method the erase went back to `bg` -- black at launch, cured by any Preferences change that re-ran the sync, black again on the next launch. Exactly the report.

A type selector, so it applies to this screen and not to the children it would otherwise cascade to: the cards and panels carry their own surface colour at the page opacity, and painting the page colour onto them would flatten the layering the scheme is built on.

## AppScreen._clear_page_surfaces

### lines 2324-2330

```python
clear_container_surfaces(self)
```

The same generic sweep every other screen uses. This used to be a hand-written list plus scroll areas and splitters, which is the shape that kept missing things on Home too: an ANONYMOUS QWidget used as a container has no QSS rule of its own, so it paints the window colour and no opacity setting can reach it. That is what left a black box under the AI chat box and a dead black rectangle between the chat and the System panel.

### lines 2340-2342

```python
getattr(self, "_actions_row", None),
```

The Run / Stop / Import / Clear strip. It has no object name of its own, so without this it takes the blanket window fill and sits as an opaque band across the backdrop.

### lines 2344-2346

```python
getattr(self, "_category_hint", None),
```

The category blurb under it. Named (so a stylesheet can reach it), which is exactly why the generic anonymous-container sweep above leaves it alone.

## AppScreen._lay_out_the_settings_panel

### lines 2375-2377

```python
scroll.setObjectName(SETTINGS_PANEL_NAME)
```

The column paints nothing — see `_settings_panel_qss`. The name is what that block keys off; without it the scroll area falls through to the blanket window fill, which is where this started.

### lines 2382-2385

```python
scroll.setMinimumWidth(280)
```

The action strip now carries GPU, search, interactive and AI toggles. Never satisfy their combined width by crushing the reducer settings into an unreadable sliver; the top-level window may grow, while the splitter remains user-resizable.

### lines 2387-2393

```python
scroll.viewport().setAutoFillBackground(False)
```

A QScrollArea's viewport auto-fills by default, and what it fills with is the WINDOW colour -- not a surface -- so no opacity setting can reach it and the settings column reads as an opaque slab over the animated backdrop. The sidebar (app.py) and Home (widgets/home.py) already say this for their own scroll areas; this one was the odd one out. The column still scrolls; it just does not paint.

### lines 2399-2402

```python
layout.setContentsMargins(0, 0, SPACING["sm"], 0)
```

No box round the categories, so the only inset is the gutter that keeps them clear of the scrollbar. The spacing below is what makes them read as separate floating panels: it is where the theme shows between one category and the next.

### lines 2406-2408

```python
self._settings_model = SettingsWidgets(
```

THE VALUES ON SCREEN DECIDE THE FORM, not the module's shipped defaults -- otherwise a nucleus channel the user has just typed would build a form that still says the run has no nucleus.

### lines 2412-2415

```python
self._rows_awaiting_layout = {}
```

``key -> the heading it belongs to``, for every row the object rule has already decided must not be on the form. `_build_settings_section` fills it and `_lay_out_the_rows_that_are_back` empties it as the rule changes its mind. Reset per panel: a screen may build a second one.

### lines 2421-2424

```python
self._settings_model.rows_are_laid_out_by = \
```

THE MODEL ASKS THE PANEL FOR A ROW IT IS ABOUT TO SHOW. The rule decides visibility; only the panel can build a row, and a rule that showed a field with no row would put a bare control in no layout at all. See `SettingsWidgets.refresh_object_visibility`.

### lines 2436-2441

```python
self._empty_state_card = self._build_empty_state_banner()
```

Empty-state banner — shown ONLY when the src widget is empty. It's a compact "Drop a plate folder here or pick a Demo dataset" card that sits above the settings form; it auto-hides as soon as the user sets src (drag/drop or typing). Users who load settings via Import don't see it a second time.

### lines 2447-2451

```python
self._discarded_settings_host = QWidget(content)
```

Sections pruned from the rendered tree can still own form rows that collect, search and a later object-visibility pass must reach.  They therefore stay in the complete section registry, but live under an explicitly hidden, screen-owned host: unlike ``setParent(None)``, that cannot turn their headers into stray top-level windows.

### lines 2457-2459

```python
self._category_blurbs = {}
```

Heading text -> the blurb its PATH resolved to, so the strip under the actions row can answer for a sub-heading whose bare word means something else somewhere else in the tool.

### lines 2461-2472

```python
self._settings_tabs = None
```

CATEGORIES AS TABS, where a screen has enough of them to be a wall. Asked for 2026-08-19: "in measure i dont like the black categories. can we make them into measurement subtabs?" -- and Measure is the screen with the most: stacked, its categories are a column of headers taller than the panel, and the expanded one paints a slab of `surface_alt` that reads as black.

THE RENDERED SECTIONS THEMSELVES ARE UNCHANGED. Each still knows its own maturity, hint and rows. ``_settings_sections`` is the complete form/search registry; ``rendered_settings_sections`` is the subset mounted in this layout. (Both retain the deepest-first order `_build_settings_section` explains.)

### lines 2487-2489

```python
for section_order, spec in enumerate(sections):
```

Map widget → plain-text hint so the bottom hint strip AND our sticky HoverTooltip can look up the description for the object under the cursor. Initialized in __init__.

### lines 2493-2506

```python
if not self._section_holds_anything(section):
```

A CATEGORY WITH NOTHING IN IT IS NOT SHOWN. Asked for 2026-08-28: "this will help not overwhelm the user."

An empty heading is worse than an absent one. It reads as a section that failed to load rather than one that does not apply, and it invites the user to expand it and find out which is the cost the request is about. Mask showed "Organelle segmentation" and "Organelle segmentation advanced" with nothing under them whenever the run had no organelles.

DECIDED HERE, over the finished section, rather than by each thing that can empty one: a heading emptied by the organelle count, by the maturity filter, or by a rule added later is the same empty heading.

### lines 2520-2525

```python
holder.viewport().setAutoFillBackground(False)
```

THE SAME VIEWPORT FILL AS THE COLUMN ABOVE. A

`QScrollArea`'s viewport auto-fills by default with the WINDOW colour rather than a surface, so no opacity preference can reach it and it reads as an opaque slab behind the settings. The column was fixed for this; the tab pages were the ones still doing it.

### lines 2530-2532

```python
section.set_expanded(True)
```

A TAB IS ALREADY THE DISCLOSURE, so the header inside it would be a second one saying the same thing. It stays expanded and keeps its hint for screen readers.

### lines 2537-2541

```python
self._settings_layout = layout
```

KEPT, so the preference can be turned on after this panel is built. Without it the only way to mount the grid was to build the panel, and the switch appeared dead until the module was reopened. `addStretch` goes in below, so the grid is inserted at the index the stretch will take rather than appended after it.

### lines 2546-2564

```python
model = getattr(self, "_settings_model", None)
```

THE OBJECT RULE, NOW THAT THE ROWS EXIST. `SettingsWidgets` cannot apply it while it is handing the rows back -- there is no ROW to hide yet, only a field, and hiding the field alone leaves its name behind on an empty row -- so it schedules the pass on a zero-delay timer instead. That timer lands on the next turn of the event loop.

WHICH IS TOO LATE FOR ANYONE WHO LOOKS FIRST. A caller that builds a panel and reads it without spinning the loop sees every gated row VISIBLE, because the only pass that would have hidden them has not run. That did not show while an unset object's keys were also being dropped from the build: rows that do not exist cannot be visible, so the deferral was invisible behind the skip. Take the skip away which is what 356 needs, see 382 -- and the deferral is the defect on its own.

Run here, synchronously, at the point the rows are on the form and the panel is about to be returned. The timer stays: it is what re-asserts the rule after a later route puts a row back, and it is cheap when there is nothing to do.

## AppScreen._mount_the_object_grid

### lines 2622-2624

```python
grid.set_app_key(self.app_key)
```

WHICH MODULE'S API THE TOOLTIPS LINK TO. The table cannot work this out -- it is a widget, not a screen -- and a guess would send a reader to another module's page.

### lines 2629-2641

```python
if len(grid.questions()) < self.MIN_GRID_QUESTIONS or not owned:
```

A TABLE HAS TO EARN ITS PLACE. The grid exists to collapse the repeated SEGMENTATION settings -- twenty-odd questions asked once per object, which as a flat form is 78 rows. A module with two shared questions has a four-row form, and replacing four rows with a table, a header, an Add button and a resize grip is a heavier control than the thing it replaces.

Regression is the case that prompted this: it shares

`area_outlier_mads` and `intensity_outlier_mads` between cell and nucleus, which is genuinely per-object and genuinely not worth a table. Classify has none at all. Measure has three questions over four objects and Mask has twenty over three, which are.

### lines 2649-2653

```python
layout.insertWidget(self._index_before_the_stretch(layout),
```

BEFORE THE TRAILING STRETCH, not after it. On the first build there is no stretch yet and this is an append; mounted later from the preference switch there is one, and appending would put the table below a spring that pushes it off the bottom of a scrolling panel.

## AppScreen.apply_object_grid_preference

### lines 2685-2687

```python
try:
```

A grid whose C++ side is gone is not a mounted grid. `deleteLater` on the section takes the table with it, and the dangling Python wrapper would otherwise read as "already mounted" forever.

## AppScreen._section_holds_anything.already_built_rows

### lines 2825-2830

```python
return list.__iter__(rows)
```

``bool``, ``len`` and normal iteration are the public completeness seam of this list and deliberately build all waiting rows.  Empty-section pruning only needs to inspect rows that are already laid out; materialising hidden object rows here defeats the caption-saving optimisation it is meant to preserve.

## AppScreen._section_holds_anything.holds_an_active_slot

### lines 2838-2844

```python
return any(
```

The organelle COUNT settles the form's shape before a channel is chosen.  Its requested slots therefore own their categories already, even though channel-gated detail rows still wait for a caption.  Counting the declarations keeps that category in the tree without iterating the lazy row list; nucleus/pathogen rows do not match this role vocabulary and remain prunable while their switches are empty.

## AppScreen._restore_settings_section

### lines 2905-2908

```python
layout = self._settings_content.layout()
```

Top-level dormant sections are the only ones moved to the host. Put one back at its declaration-order position amongst the other top-level cards. The notice stays above every category and the stretch stays below them, just as on the first build.

## AppScreen._watch_the_settings_that_decide_the_form

### lines 3052-3055

```python
switches = set(self._object_switches_on_this_form())
```

TWO SLOTS, NOT ONE. A key that adds or removes ROWS needs the form built again; a key that only decides which of the rows already on the form are SHOWN does not, and connecting both to the rebuild is what made typing a channel number reload the module.

## AppScreen._rebuild_the_form

### lines 3132-3134

```python
deferred = dict(getattr(self, "_deferred_form_values", None) or {})
```

Preserve values a bulk import supplied for controls this form has not built yet, then let later on-screen edits win for every key the current form does hold.

### lines 3145-3148

```python
merged = dict(deferred_values)
```

The deferred snapshot may be minutes old. Values supplied for controls absent from this shape stay in it; every value the live form can collect is refreshed now so next-run edits made after the import are never rolled back when the worker finishes.

### lines 3174-3177

```python
fresh._refresh_after_bulk_apply(deferred_bulk)
```

The target mapping already populated the replacement. What still needs provenance is a sparse Type preset: only the originally supplied keys tell us which recommended values were missing and which advanced values were explicit.

## AppScreen._build_settings_section

### lines 3220-3224

```python
section.setProperty("settingsCategorySource", title)
```

The category name as the layout writes it. Everything that looks a category up -- the blurb tables and the catalogs is keyed on that spelling, and the header answers with the uppercased caption it draws, so the written name is kept where `_wire_category_hints` can read it back.

### lines 3229-3237

```python
blurb = section_tooltip(self.app_key, spec)
```

The category blurb. Its primary home is the strip under the actions row (see `_wire_category_hints`); `set_hint` keeps the same text on the header for screen readers and for the beta/alpha caution note it appends. `section_tooltip` resolves a nested heading by its PATH -- "Cell" under "Object filtration" is not the "Cell" segmentation category -- and falls through to `category_tooltip` for a top-level one, which resolves the module's own override first, then the shared table, then a generic sentence, so a section is never left without text.

### lines 3240-3244

```python
self._category_blurbs.setdefault(title, blurb)
```

The strip under the actions row is fed by TITLE, because that is what a hovered header carries. A sub-heading's blurb is recorded here so the strip can answer with the one its path resolved rather than looking the bare word up again; `setdefault` leaves a top-level category owning its own name.

### lines 3246-3263

```python
declared = tuple((self._key_of_field(widget), label, widget)
```

A ROW THE RUN HAS NO OBJECT FOR COSTS NO CAPTION UNTIL IT HAS ONE.

The panel builds a control for every organelle slot that CAN be named, because a control that was never built cannot be revealed so on Mask the great majority of its controls belong to objects the run does not have, and are hidden before the panel is ever painted. Each was still given a caption and the host that right-aligns it: two widgets and two style repolishes apiece, for a caption nobody can read, on a panel where a style recalculation walks every widget alive.

THE ROW IS STILL ON THE FORM, spanning, holding its field. That is what everything that walks the form goes on finding: the row can be hidden and asked whether it is hidden, the settings search indexes it, and `SettingsWidgets` reaches it exactly as before. What waits is the CAPTION, and `_lay_out_one_waiting_row` gives the row one in place, in the same form row -- the moment the object rule says the run has that object after all.

### lines 3270-3275

```python
try:
```

WHAT THIS HEADING HOLDS, told to the model rather than left to be worked out again. `SettingsWidgets` needs the same answer to decide which slot headings belong to objects the run does not have, and it used to get it by walking every heading's form and asking each heading whether anything nests inside it -- a `findChildren` per heading, on a panel that has a heading per object slot.

### lines 3284-3288

```python
section._row_widgets = _RowsBuiltWhenTheyAreAskedFor(
```

ANYTHING THAT READS THE ROWS BACK GETS ALL OF THEM. Several checks walk `Section._row_widgets` -- the module smoke test reads every entry as a labelled setting row -- and a list that quietly held a fraction of the panel's rows would let them pass while checking almost nothing.

### lines 3294-3299

```python
section.add_prose(widget)
```

`add_prose`, for the one thing it does that `add_widget` does not: it leaves `_row_widgets` alone. A field with no caption is not yet the labelled setting row every reader of that list takes each entry to be, and `_RowsBuiltWhenTheyAreAskedFor` is what hands one over captioned -- to anybody who asks.

### lines 3303-3308

```python
for child in children:
```

THE HEADINGS BELOW THIS ONE, each a Section of its own inside this one's body. `add_prose`, not `add_widget`: the second registers what it is handed in `_row_widgets`, where every entry is taken to BE a labelled setting row by `tests/qt/test_all_module_smoke.py::_setting_row_contract`, and a heading is neither a setting nor labelled.

### lines 3311-3318

```python
if not self._section_holds_anything(nested):
```

AND A NESTED HEADING WITH NOTHING IN IT GOES TOO. Pruning only at the top left an empty sub-heading inside a category that was itself kept for its other children -- "Organelle Segmentation (advanced)" survived a run with no organelles that way. Deepest first, because this runs inside the recursion: a child is pruned before its parent is judged, so an umbrella over nothing but empty sub-headings is empty by the time the parent asks.

### lines 3320-3323

```python
section.add_prose(nested)
```

Keep the dormant heading in its intended place in the hierarchy. It remains explicitly hidden, but can be revealed in place if its object switch changes before the normal committed-value screen rebuild.

### lines 3328-3331

```python
nested.toggled.connect(
```

A HEADING OPENED FROM OUTSIDE OPENS ITS ANCESTORS. The search strip and the command palette expand the section holding a match; expanding one that sits inside a collapsed umbrella shows the user nothing at all.

### lines 3334-3351

```python
from .settings_model import has_section_explainer
```

THE EXPLAINER GOES AT THE TOP OF THE SECTION IT EXPLAINS.

Asked for on 2026-08-17: "just ad the text box i asked for (at the top)". It used to be appended to the PANE after `layout.addWidget(section)`, which put it BELOW every control it describes -- so a user read eleven settings and then found out what they were choosing between.

`Section.add_prose`, not `Section.add_widget`: the second registers the widget in `_row_widgets`, where every entry is taken to BE a labelled setting row by `tests/qt/test_all_module_smoke.py::_setting_row_contract`. A prose box is neither a setting nor labelled.

TOP LEVEL ONLY, both of these. Both tables are keyed on the heading's text alone, and a sub-heading shares its word with a category somewhere else in the tool -- "Cell" under "Object filtration" is not the Cell segmentation category.

### lines 3356-3358

```python
if depth == 0 and title == EXAMPLE_DATA_SECTIONS.get(self.app_key):
```

Place each example-data action beside the settings it populates. Regression fills paired tables; Mask Generation fills ``src`` with a downloaded example image directory.

### lines 3360-3368

```python
section.set_expanded(True)
```

OPEN, because a control nobody can see is not a control. Every settings section starts collapsed, so the test-data button was inside a hidden SectionBody on every module that has one reported on 2026-09-01 as "there is no Load test data in the classify module that i can see", and true of Mask and Measure too for anyone who did not already know where to look.

Only THIS section, and only because it carries the one action a user with no data of their own has to take first.

### lines 3380-3383

```python
self._settings_sections.append(section)
```

DEEPEST FIRST. Recorded after the children so the list a consumer scans for "which section holds this widget" answers with the innermost heading; the umbrella is an ancestor of every one of them and would otherwise always win.

## AppScreen._lay_out_the_rows_that_are_back

### lines 3438-3439  _(unsure)_

```python
self._the_rows_moved(judge_them=False)
```

NOT the object rule again: this IS the object rule, and it decides every row it has just been handed the moment this returns.

## AppScreen._lay_out_one_waiting_row

### line 3523  _(unsure)_

```python
LOG.debug("no heading left to caption %s on", key, exc_info=True)
```

The heading went away with the screen that owned it.

## AppScreen._the_rows_moved

### lines 3547-3551

```python
if all(id(pair[1]) in order for pair in rows):
```

ONLY WHEN EVERY ROW IS ONE THIS CAN PLACE. A handful of settings sit in a little holder with a button beside them, and it is the holder the row records -- so a heading with one of those cannot be ordered from the declared fields, and guessing would be worse than the order it already has.

### lines 3555-3558

```python
model = getattr(self, "_settings_model", None)
```

A ROW BUILT BECAUSE SOMETHING READ THE HEADING BACK IS NOT A ROW THE RUN HAS AN OBJECT FOR. Asking what a heading holds must not put settings for an absent object on screen, so the rule that kept them off decides them again now that they exist.

### lines 3571-3576

```python
late = getattr(self, "_captioned_late", None) or set()
```

THE CAPTION ARRIVES IN THE LANGUAGE THE PANEL WAS WRITTEN IN, which is English: every other caption on the form was translated by the pass that runs once the panel is built, and one that did not exist then would sit in English inside a translated window. The pass is idempotent, so re-running it over the headings that changed costs the rest of the panel nothing.

### lines 3594-3599

```python
bar._build_index()
```

THE STRIP INDEXES THE RENDERED FORM -- specifically, whatever widget the form holds for each row. A row that has just been captioned may hold a different one: the few settings with a button beside them sit in a holder, and the holder is what the form ends up with. Re-indexing is keyed by setting, so it replaces those entries rather than adding to them.

## AppScreen.the_name_carries_the_help

### lines 3631-3635

```python
try:
```

THE WHOLE SCREEN, not just the settings column. A module's own panels -- the regression sweep's group boxes, for one -- sit outside that scroll area and have names of their own to move the help onto; sweeping only the column left them popping from the control.

### lines 3639-3640

```python
return 0
```

Help that failed to move is a blemish, never a reason for a module not to open.

## AppScreen._lay_out_setting_row

### lines 3656-3663

```python
setting_key = self._key_of(widget)
```

THE KEY, NOT THE LABEL. `build_sections` hands out

`('Control wells', widget)` -- a title-cased sentence for a human -- and both wrappers below match on the SETTING NAME. Passing the label meant `control_wells` was never equal to 'Control wells' and the plate map (185) appeared on nothing at all. Its tests passed because they called `pick_wells_for` directly rather than driving the row the user actually sees.

### lines 3665-3677

```python
field = widget
```

A PLATE MAP BESIDE THE FIELDS THAT TAKE WELLS (185). "to the right of the field should be a button they can press that spawns a window". Only the settings whose value is ONLY wells: the picker writes the whole field, and one that overwrote `classes` or `negative_control` -- which mix wells with another vocabulary -- would destroy a value it does not understand. THE INNER FIELD IS KEPT, because everything below binds to IT and not to the row it now sits in: the tooltip is read off it, the label is bound to it, and `_widgets` still holds it. Losing this reference is what made a wrapped row lose its `settingKey` and its help the moment the wrapper stopped being a no-op.

### lines 3680-3682

```python
widget = self._with_a_settings_advisor(widget, setting_key)
```

AND THE ADVISOR BESIDE `inference` (192). "a button to the left of inference alligned with the text box to the left in model & inference".

### lines 3684-3686

```python
widget = self._with_a_model_zoo_button(widget, setting_key)
```

AND THE MODEL ZOO BESIDE A CHECKPOINT FIELD. Same wrapper shape as the two above, and the same rule: the inner field is what the panel collects from, so typing a path by hand is unchanged.

### lines 3689-3693

```python
widget.setProperty("settingKey", setting_key)
```

THE ROW IS THE FIELD AS FAR AS THE PANEL IS CONCERNED. `Section._row_widgets` records what it is handed, and the module smoke test reads `settingKey` off that -- so the holder has to carry it too, or a row with a button beside it reads as a field belonging to no setting.

### lines 3697-3699

```python
lbl_widget.setCursor(Qt.WhatsThisCursor)
```

Give the label a subtle affordance so users know it's the hover target for tooltips (fields can be focused / clicked — tooltips on labels are calmer).

### lines 3702-3713

```python
key = self._key_of_field(field)
```

MATCHED ON THE INNER FIELD, not on `widget`: `widget` may now be the row that holds it, which `_widgets` has never heard of.

THROUGH AN INDEX, not by scanning. This used to walk the whole of `_widgets` looking for the row's own field, once per row 1,538 rows against 1,538 widgets on the Mask screen, which is over a million identity comparisons to answer 1,538 questions that a dictionary answers outright. `_widget_key_index` is built once per panel and preserves the scan's answer exactly: first key wins, for the vanishingly rare case of one widget registered under two names.

### lines 3730-3736

```python
field.setProperty("apiTooltipDisplayRole", "metadata")
```

Tooltips live on the LABEL only — hovering the input field itself is left alone so focus / edit interactions aren't disturbed. Keep the field's semantic metadata for language refreshes, but mark it quiet before clearing the native tooltip. Otherwise a later object-visibility refresh restores the help on hidden fields after this row has already moved it to the label.

### lines 3740-3748

```python
field._spacr_setting_label = lbl_widget
```

SettingsWidgets may already have disabled an algorithm-specific field before this visual label exists. Bind them now and mirror the state; later reducer switches update both through the same link.

ON THE FIELD, not on the row: the reducer that later enables and disables this setting reaches it through `_widgets`, which holds the field, so a label bound to the holder would never be told.

### lines 3754-3772

```python
section.add_row(lbl_widget, widget, info_widget=None,
```

No API link dot on the settings form. It sat between the label and the field and carried a tooltip of its own, so the help popped when the pointer was over the row's right-hand side -- which reads as "the field has a tooltip", because from the user's side of the screen that is exactly what it looks like. 191 of them on the Mask form alone.

Nothing is lost but the mark: the API link is still in the label's tooltip HTML (the `href=` several tests assert on), so the reference is one hover and one click away, and the help itself is unchanged and still on the label.

The host stays, though, and is built here rather than by

`Section.add_row` (which only makes one when there is an info widget to put in it). It is what right-aligns the label against the field: dropping it left the label left-aligned and half the row's width was suddenly the page showing through rather than the category surface.

### lines 3775-3777

```python
self._attach_column_picker(field_key, field)
```

THE FIELD, for the same reason as the tooltip above: the column picker fills the setting's own widget, and handing it the row would give it something with no `setText`.

## AppScreen._install_section_explainer

### lines 3808-3813

```python
box = _ExplainerBrowser(self._retheme_section_explainers)
```

A QTextBrowser RATHER THAN A BARE QTextEdit, and the difference is the one thing instruction 144 D needs: `setOpenExternalLinks` is QTextBrowser's. A QTextEdit renders the same HTML and its links do nothing when clicked, which is worse than a module name -- a module name can at least be searched. Everything else is inherited, so read-only, selectable and the wrap mode are unchanged.

### lines 3818-3830

```python
box.setLineWrapMode(QTextEdit.WidgetWidth)
```

WRAP AT THE BOX'S OWN WIDTH. Asked for on 2026-08-18, once per box: "the text in the text box should span the width of the textbox".

It was NoWrap over text `_wrap_block` had already hard-wrapped to 54 columns, so the paragraph was 54 characters wide whatever the pane was and the right-hand side of the box sat empty. The prose is one logical line per paragraph now and this wraps it, so it reflows when the pane is resized.

A FORMULA STILL CANNOT BREAK, and that is why the minimum width below is `explainer_width()` -- the longest unbreakable line in any explainer, measured from them rather than declared. The widget is never narrow enough to wrap one.

### lines 3833-3856

```python
fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
```

MONOSPACE HAS TO COME THROUGH QSS, NOT setFont().

The theme opens with a global `QWidget { font-family: "Open Sans", ... }` rule, and in Qt a stylesheet font beats a programmatic one. Measured under the real theme: a read-only QPlainTextEdit given setFont(systemFont(FixedFont)) reports a rendered QFontInfo family of 'Open Sans' with fixedPitch False -- proportional, so the formulas do not align and the box is monospace in intention only. Setting the document's default font loses the same way, because polishing pushes the widget font back into the document. `qt/widgets/regression_results.py`'s Summary tab has the same defect; it is not this slice's file to change.

setFont stays too, so anything reading font() rather than the rendered QFontInfo still gets the fixed font.

THE RULE NAMES THE FONT AND NOTHING ELSE, deliberately. Colours are left to the app-wide stylesheet. Baking palette values in here would PIN them at construction, so the box would keep the old colours after a runtime theme switch while the screen around it re-themed. Checked on the composited screen rather than on a widget grab: a bare `box.grab()` writes the transparent page background as transparent pixels, which an image viewer shows as white -- that artefact reads exactly like white-on-white text and is not.

### lines 3863-3865

```python
box.setTextInteractionFlags(
```

Read-only, but NOT unselectable: TextSelectableByMouse and

ByKeyboard are what make Ctrl+A / Ctrl+C work in a disabled-looking box.

### lines 3868-3873

```python
advance = QFontMetrics(box.font()).horizontalAdvance("M") or 8
```

WIDE ENOUGH FOR THE WRAP COLUMN, measured from the font actually rendered rather than guessed. The settings pane opens at about 400px, which holds ~46 monospace characters -- narrower than the 62-character mixed formula, so the one line that most needs to stay intact was the first to soft-wrap. Asking for the width here lets the splitter give the pane what this content needs.

### lines 3882-3884

```python
from ..theme import active_palette
```

A STATIC BOX NEEDS NO REFRESH. Only the model box depends on the panel's current values; the permutation box says what the test does, which does not change with a setting.

### lines 3892-3895

```python
widgets = getattr(self._settings_model, "_widgets", {})
```

Follow the two settings it describes. Bound methods, not lambdas: INVARIANTS 4 is about QThread.finished specifically, but the same lifetime reasoning applies to any connection that outlives the call that made it.

### lines 3897-3899

```python
for key in ("regression_type", "level", "model_plate_position",
```

`inference` and `analysis_mode` are on the list because the box now describes the PERMUTATION path when one is chosen -- a box that did not follow them would go on explaining a model the run will not fit.

### lines 3903-3904  _(unsure)_

```python
widget = widgets.get(key)
```

`level` is a NEW setting and may not be on the panel yet; the box still has to render for the model that IS there.

### lines 3908-3913

```python
for signal_name in ("currentTextChanged", "currentIndexChanged",
```

`toggled` FIRST-CLASS, not an afterthought. Two of the four settings this box follows are `Toggle`, which is a QCheckBox and therefore has NONE of the combo/text signals below -- so no connection was made at all and the formula never moved when the plate settings did. Silent, because a `for/else` that finds nothing simply finds nothing.

## AppScreen._refresh_model_explainer

### lines 3951-3957

```python
scroll = box.verticalScrollBar().value()
```

THE PLATE SETTINGS REACH THE FORMULA. Without them the box printed `+ rowID + columnID` however they were set, so a user who turned plate position off still read it in the formula -- the display asserting something the run does not do. SELECTION KEPT ACROSS THE RE-RENDER. `setHtml` replaces the whole document, so a user part-way through dragging out a formula to paste loses it; the scroll position goes with it. Both are put back.

### lines 3959-3963

```python
box.setHtml(regression_model_explainer_html(
```

INFERENCE REACHES THE BOX (2026-08-20): "non parametric should be represented in the Text box above ... when chosen the text should explain nonparametric." Without it the box described whatever `regression_type` held, which under nonparametric is a model that is read, saved and never fitted.

## AppScreen._gene_tile_entry

### lines 3988-3991

```python
showing = getattr(gene, "feature", None)
```

NO GENE, NO TILE -- and `to_pixmap` cannot be the test for that. It renders a placeholder when nothing is selected, so a null-pixmap check never fired and the grid grew a tile saying "nothing selected" on every run. `feature()` is what the panel is showing.

## AppScreen._with_a_plate_map

### lines 4029-4034

```python
button.clicked.connect(
```

NO FIXED WIDTH (193). It was 58 px, chosen by eye against the English "Plate…" -- and every translation is longer, so the label was elided to "Plat…" in languages the author does not read. A button's width is a CONSEQUENCE of its text, never an input to it: `sizeHint()` already accounts for the label, the icon, the padding and the current font, so the layout is given that instead.

### lines 4038-4041

```python
holder._spacr_field = widget
```

THE FIELD IS STILL THE WIDGET the panel collects from. `_widgets` already holds it, and wrapping it in a row must not change which object `collect()` reads -- a picker that made the value unreadable would be worse than no picker.

## AppScreen._choose_a_model_for

### lines 4139-4140  _(unsure)_

```python
if hasattr(field, "set_value"):
```

setText for a line edit, set_value for spaCR's own path widgets the panel builds more than one shape of field for a path.

## AppScreen._with_a_settings_advisor

### lines 4172-4174

```python
holder._spacr_field = widget
```

THE FIELD IS STILL THE WIDGET the panel collects from -- the same rule the plate map follows, and for the same reason: a wrapper that made the value unreadable would be worse than no button.

## AppScreen.settings_for_my_data

### lines 4249-4253

```python
reading = self._reading_with_the_last_run(reading, values)
```

AND THE LAST RUN, IF THERE IS ONE (instruction 226). Everything above is knowable before a fit; the residuals of one that happened are what the assumptions are actually about, and a response can be skewed while the residuals are fine. An ADDITION and never a requirement: with no run this changes nothing at all.

### lines 4264-4267

```python
chosen = advise_that_runs(reading, answers).as_settings()
```

THE HEADLESS ROUTE, and it still goes through the same advisor rather than a second rule -- one advisor, whether a person or a test is asking.

## AppScreen._console_folded

### lines 4296-4297  _(unsure)_

```python
wrap.setMinimumHeight(0 if shut else 180)
```

The wrapper's minimum is the console's while it is open, and the heading's alone once it is folded.

### lines 4315-4316  _(unsure)_

```python
above = index - 1 if index > 0 else (1 if len(sizes) > 1 else -1)
```

To the pane above, which is the one holding the figures, the montage or the results -- the things a log line was crowding.

## AppScreen._reading_with_the_last_run

### lines 4353-4354

```python
return reading
```

The advisor is worth more than the extra: a run folder that cannot be read must not cost the user the advice they asked for.

## AppScreen._install_example_data_button

### lines 4392-4399

```python
row = QWidget()
```

THREE BUTTONS, ONE PER THING TO FETCH. Counts are 16 MB and scores are 19 MB; a user checking one of them should not wait for the other, and a user who wants both still presses one button.

In a ROW AT THE TOP rather than above each field. `add_prose` puts a widget above or below the section's controls, and inserting between two setting rows would put a non-setting into `_row_widgets`, which the module smoke test reads as a labelled setting row.

### lines 4420-4424

```python
feature = QPushButton("Measurements (.db)")
```

TWO THAT COST GIGABYTES ASK FIRST. Each opens a picker holding only its own kind, with what it costs said above the list. NAMED FOR THE FILE IT FETCHES. It was "Feature", which is what the tables hold rather than what the button gets you, and a user asking for "the measurements.db button" could not find it.

### lines 4447-4459

```python
section.add_prose_row("Download", row, at_top=True)
```

A LABELLED ROW, ALIGNED WITH THE SETTINGS IT FILLS. A row of buttons floating above the form reads as unrelated to it; one whose label sits in the same right-aligned column reads as part of the same form. `add_prose_row` rather than `add_row`, because `_row_widgets` is taken to hold labelled SETTINGS by the module smoke test and a row of buttons is not one.

AT THE TOP since 2026-09-02: "the input tables sould start with download buttons not end wit them". It fills the fields below it, and sitting under them read as a footer to a form it actually feeds. The whole-screen button below inserts at 0 afterwards, so the final order is: Load test data, then Download, then the fields -- broadest action first.

### lines 4462-4463  _(unsure)_

```python
button = QPushButton(tr("Load test data…"))
```

The whole-screen button keeps its place ABOVE the form: it fills both slots at once, so it belongs beside neither of them.

## AppScreen.reanchor_example_paths.rehome_text

### lines 4543-4544  _(unsure)_

```python
pure = (PureWindowsPath(stripped) if "\\" in stripped
```

Both separators: the file may have been written on Windows and read here, or the other way round.

### lines 4563-4567

```python
return str(candidate)
```

NOT conditional on the candidate existing, unlike portable_paths. Some of these are OUTPUT paths -- a measurements.db a first run has not written yet -- and requiring existence would leave the publisher's path on exactly the settings a first run needs.

### lines 4569-4571

```python
return text
```

Nothing to hang the tail on. A bare re-point at the destination would be a guess, and a wrong path that LOOKS local is worse than one that is obviously foreign, so it is left for the user to see.

## AppScreen.apply_settings_that_came_with

### lines 4614-4622

```python
try:
```

BEFORE applying, not after: the panel must never hold the publisher's path, not even for a repaint.

GUARDED SEPARATELY from the load. The surrounding handler turns any failure into "0 settings applied", so a fault in the re-homing would silently cost the user every setting the example shipped -- trading a wrong path for no configuration at all. A foreign path is the lesser failure and is at least visible, so it is what happens if this cannot run.

## AppScreen.load_the_screen_data

### line 4670  _(unsure)_

```python
was = {"measurements": ("_screen_feature_button", "Feature"),
```

The button that was pressed, so the other one does not look busy.

## AppScreen.load_the_measure_example

### lines 4754-4757

```python
merged = destination / "merged"
```

`merged/` holding at least one array is the test. The folder itself is shared with the other example sets now, so its existence says nothing about whether THIS one has been fetched -- and a cancelled download leaves it behind too.

## AppScreen._put_the_measure_example_in_place

### lines 4836-4838

```python
model = getattr(self, "_settings_model", None)
```

Through the WIDGET, the same way the mask example does it: the settings model has no setter, and writing a value the widget does not show would leave the panel disagreeing with the run.

### line 4847

```python
self.apply_settings_that_came_with(destination)
```

AND THE SETTINGS THAT CAME WITH IT, so Run is the next action.

### lines 4850-4866

```python
return {"src": self.keep_the_src_openable(destination)}
```

THE SHIPPED `src` IS ALLOWED TO WIN HERE, UNLIKE THE IMAGES ROUTE, and only a BROKEN one is taken back.

Measure's example records `src` as the `merged/` SUBFOLDER of the plate, which is more specific than the folder downloaded into and is the directory Measure must actually read `reanchor_example_paths` says in its own docstring that collapsing it to the plate root "would quietly measure the wrong directory rather than fail". So this route deliberately does NOT re-assert the destination the way the images route does (instruction 349).

What it does refuse is a value that is not a directory at all: a template token, or a publisher's path that could not be re-homed. That is the failure reported on 2026-09-02 for Mask, and nothing about it is specific to Mask -- it just showed up there first. Falling back to the folder we downloaded into is strictly better than a path that cannot be opened.

## AppScreen.load_the_annotate_example

### lines 4974-4977

```python
if (destination / "measurements" / "measurements.db").is_file():
```

THIS SET'S OWN PART IS THE TEST, not the folder. Every example now unpacks into one shared plate directory, so "the folder exists" is true as soon as any of them has been fetched -- and a cancelled download leaves the folder behind as well.

## AppScreen._apply_the_example_settings

### lines 5018-5020

```python
applied = self.apply_settings_that_came_with(destination)
```

Through the shared applier, so every module's example fills its own form the same way and there is one place that knows which file each module ships.

### lines 5028-5032

```python
return {"src": self.keep_the_src_openable(destination)}
```

THIS ROUTE NEVER WROTE `src` AT ALL, so the panel held whatever the shipped file said and the mapping below claimed the destination regardless -- a caller could believe `src` was the download folder while the form showed a path from another machine. Same guard, stated once.

## AppScreen.load_the_example_images

### lines 5053-5057

```python
plate = destination
```

THE IMAGES THEMSELVES are the test. The destination is the plate directory now and is shared with the other example sets, so "the folder is not empty" is true as soon as any of them has been fetched -- it would have skipped the download and pointed `src` at a folder with no images in it.

## AppScreen.load_the_example_images.ask

### lines 5089-5095

```python
def ask(parent, dest, on_done):
```

THE TAR, asked for here rather than made the shared default. `download_toxo_mito_demo` is driven by tests that patch the per-file worker's own helpers to prove the offline failure path stays on the GUI thread; changing what they get sent them to the network for real and aborted the process. A shared entry point's default is part of its contract with everything already calling it, so the new behaviour is requested rather than imposed.

## AppScreen._put_the_example_images_in_place

### lines 5119-5121

```python
self.apply_settings_that_came_with(images)
```

NAMING THE FILE WAS NEVER ENOUGH. It said where compatible settings were and left the user to import them, which is the work the example exists to save -- so they are applied.

### lines 5124-5144

```python
if control is not None and hasattr(control, "setText"):
```

`src` IS WRITTEN LAST, AND THAT ORDER IS THE FIX.

Reported 2026-09-02: "loade test images dosnt loade the right path into src in mask generation it loads <src>". It was written FIRST and then overwritten -- `apply_settings_that_came_with` loads the shipped CSV and applies every key in it, `src` included, so whatever the publisher recorded won. `reanchor_example_paths` re-homes a recorded ABSOLUTE path onto this folder, but a value it cannot resolve -- a template token, or a path whose folder name does not match -- is deliberately left alone by every branch of it, and then lands in the field verbatim.

THE FOLDER WE JUST UNPACKED INTO IS GROUND TRUTH, and it is known to exist. It beats anything the file can say about a machine that is not this one.

ONLY ON THIS ROUTE. Measure's shipped `src` points at a SUBFOLDER (`merged/`) and collapsing that to the plate root would quietly measure the wrong directory -- `reanchor_example_paths` says so in its own docstring. This method is the example-IMAGES route only, where `src` is the plate folder by construction.

## AppScreen.load_the_example_screen

### lines 5160-5162

```python
button = getattr(self, f"_example_{kind}_button", None) if kind else None
```

THE BUTTON THAT WAS PRESSED, so a counts fetch does not disable and relabel the "load everything" button while leaving its own looking idle.

### lines 5169-5170

```python
button.setText(tr("Fetching {count} file(s)\u2026",
```

The count is substituted AFTER the lookup, so the catalog holds a sentence rather than one sentence per possible count.

### lines 5182-5185

```python
button.setText(tr("Load test data…"))
```

Restored through `tr`: the language pass rendered this caption once, and putting the English source back would both show the wrong word and opt the button out of every later pass.

### lines 5188-5198

```python
table = self._settings_model._widgets.get("paired_data")
```

`paired_data`, NOT `count_data`/`score_data`. The regression panel holds ONE ROW PER PLATE -- its score CSV beside its count CSV and the two flat lists are the legacy shape that `_migrate_paired_data` converts. Filling the flat keys put the paths somewhere the panel does not show: measured, `collect()` came back with neither key on it.

`add_paths_for_side` is the widget's own door and it RE-PROPOSES the whole table from filename tokens on every arrival, so plate_1_unique_combinations.csv pairs itself with plate1_dv.csv whichever side arrives first.

## AppScreen.refresh_maturity_visibility

### lines 5254-5257

```python
stages = [stage for stage in ("alpha", "beta")
```

COMPOSED FROM TRANSLATED PARTS, not translated after being composed. The finished sentence names one stage or two, so it is a phrase no catalog can hold; the stage names and the sentence around them are looked up separately and joined.

## AppScreen._install_dimension_switches

### lines 5290-5291  _(unsure)_

```python
offered = {setting_dimension(key)
```

`_dimension_rows` yields only rows that HAVE a dimension, so every answer here is one of the two and none is blank.

### lines 5298-5299

```python
switch.setMinimumWidth(DIMENSION_TOGGLE_MIN_PX)
```

See DIMENSION_TOGGLE_MIN_PX: unaided, these two take 177 px off the settings column at every window narrower than about 1500.

### lines 5307-5311

```python
self.refresh_maturity_visibility()
```

THE FORM IS GATED ONLY ONCE THERE IS SOMETHING TO UNGATE IT. The settings panel is built before this row, and its first `refresh_maturity_visibility` therefore ran while no switch existed -- so nothing was hidden then, on purpose. This is the pass that hides it, now that a user can bring it back.

## AppScreen.set_dimension

### lines 5350-5351  _(unsure)_

```python
switch.setChecked(bool(on))
```

Comes back through `_on_dimension_switch`, so a programmatic move takes the path a click takes.

## AppScreen._dimension_rows

### lines 5378-5381

```python
return []
```

Every other screen keeps the form it always had. The walk is skipped rather than run and discarded: this is called from `refresh_maturity_visibility`, which every module runs while it is being built.

## AppScreen._attach_column_picker

### lines 5522-5538

```python
from .settings_model import _CsvColumnField
```

NOT BOTH BUTTONS. A field that already carries a CSV column picker has answered this question from the right file, and the SQL button beside it would answer it from the wrong one.

Asked for on 2026-08-17: "for the filter column there is an SQL buton this should be a csv buton that can read the input csvs". In the regression module `filter_column` names a column of the INPUT CSVs; the SQL picker opens the run's measurements.db, which a regression run need not even have. Two buttons on one row, offering two different column lists for one setting, is worse than either alone -- and the SQL one could no longer write into the field once the CSV field wrapped it, so it was a control that did nothing.

Detected from the WIDGET rather than from a second per-module table: whoever gave the field a CSV picker has already decided which file the setting reads, and a table here would be a second place for that decision to be made differently.

### lines 5543-5546

```python
from ..widgets.class_editor import ClassEditorWidget
```

A COMPOSITE PUTS ITS OWN BUTTON. The Classes editor's column combo is several widgets down; wrapping the composite would place the button beside the whole table rather than beside the field it fills, and could not write into it either.

## AppScreen._build_empty_state_banner

### lines 5603-5614

```python
existing = self._settings_src_path()
```

If src already points at a real path, don't show the banner. `path`, `""` and None are all placeholders the settings dicts use as "no src set yet".

ASKED OF `_settings_src_path`, NOT of `isinstance(src, QLineEdit)`. Instruction 109 gave the merging modules a `DatabaseSetWidget` for `src`, and this test read a text box that no longer existed there: `existing` stayed "" whatever was loaded, so Image UMAP showed "Point image umap at some data" over three loaded databases. `_settings_src_path` already knows how to read both shapes, and asking it is what keeps the next control that replaces a QLineEdit from reintroducing this.

### lines 5620-5625

```python
title = tr(
```

Human-friendly title varies per app; the body is the same. THE MODULE NAME IS A VALUE, NOT PART OF THE KEY: baking it into the sentence first asks the catalog for "Point measure at some data" and every other module's variant of it, none of which any catalog can hold. The sentence is looked up as a template and the name itself translated -- put in afterwards.

### lines 5630-5636

```python
try:
```

...and so does the demo. This named "Demos → Mask demo…" on every screen, so Measure, Timelapse, Classify and Sequencing all offered a dataset that opens a DIFFERENT module: following the hint on the Measure screen generates raw images, navigates away to Mask, and leaves the empty screen the user was trying to fill exactly as empty. Ask which demo lands HERE, and say nothing specific when none does.

### lines 5642-5645

```python
offer = (
```

Same reason as the title above: the clause and the sentence are each looked up on their own, so the demo's name is the only part that is interpolated and the catalog is never asked for a key that contains it.

### lines 5657-5663

```python
cta_label="Choose source data",
```

THE BUTTON DOES THE THING THE CARD IS ABOUT. It opened the

Demos menu, which is one way to get data and not the way most people arrive: somebody who already has images wanted the card to take them to their images, and instead it offered them a synthetic dataset. The demo is still offered -- in the sentence above, which names the one that lands on THIS screen -- and the button now sets the source folder.

### lines 5667-5670

```python
if isinstance(src_widget, QLineEdit):
```

Auto-hide once the user sets src -- through whichever signal the control has. A set of databases has no `textChanged`, so without this arm the card stayed on screen for the whole session however many plates were added.

### lines 5676-5679

```python
changed.connect(self._refresh_empty_state)
```

A BOUND METHOD, not a closure over `src_widget` (INVARIANTS 4), and it re-reads the control rather than being told: the signal carries no payload and the answer is "does src hold anything", which only the control knows.

## AppScreen.choose_source_folder

### lines 5770-5773

```python
setter = getattr(self._settings_model, "set_value_for_key", None)
```

THROUGH THE MODEL, not by poking the widget. `src` is a plain line edit on most screens and a list of plates on Classify, and the model is what knows the difference -- writing text into the second one would put a folder where a set of databases goes.

## AppScreen._open_demos_menu

### line 5798  _(unsure)_

```python
m.exec(mw.mapToGlobal(mw.rect().topLeft()))
```

Show the menu at the top-left of the window

## AppScreen.eventFilter

### lines 5806-5815

```python
event_type = event.type()
```

THE EVENT TYPE IS THE FIRST QUESTION, and it used to be the third. This filter is installed on every settings LABEL -- 1,538 of them on the Mask screen -- so it is handed every event those labels receive while the panel is assembling itself: polish, style change, palette change, show. Measured on a Mask build, 14,472 calls before the pointer has moved at all, each paying for two module lookups and a `QObject.property` round trip to answer a question about hovering. Nothing below this line is reachable for any other event type, so asking first is free and costs those 14,472 calls a single integer comparison instead.

### lines 5818-5832

```python
if hasattr(self, "_hint_strip") and (
```

SWALLOW THE NATIVE ONE, before the fast path below sends it on. Removing the sticky popup would otherwise just hand the job to Qt's own tooltip, which appears a moment later over the same form -- the box the maintainer asked not to see. Returning True is what stops it being drawn.

`toolTip()` is left SET on the widget: that property is what the accessibility tree reads, so suppressing the DRAWING costs no assistive text. The same trade `module_hints` makes for the sidebar, and the reason this is a suppression rather than a deletion.

A ToolTip event arrives only after Qt's hover delay, so this costs the hot path nothing: the 14,472 assembly-time events counted below are polish and style changes, never this.

### lines 5839-5841

```python
category = obj.property("settingsCategory")
```

A settings CATEGORY header writes its own strip and nothing else: it has no setting key, so falling through would blank the per-setting strip every time the pointer crossed a header.

### lines 5861-5867

```python
from ..preferences import (get_tooltips_bottom_enabled,
```

THE TWO SURFACES ARE NOW SWITCHES, instruction 371. Before this the strip always won and the popup appeared only where there was no strip -- the 2026-09-01 request, "i dont need the popup box if the tooltip is shown on the bottom of the window". 371 asks for both to be choosable, so that preference stops being wired in and becomes a cleared checkbox: `Tooltips box` off reproduces it exactly.

### lines 5881-5884

```python
self._hinted_widget = obj
```

REMEMBERED SO THE LINK HAS SOMETHING TO ACT ON. The strip outlives the hover by ten seconds (371 part 3), so by the time an Animation press arrives the pointer is long gone from the widget the animation belongs to.

### lines 5890-5900

```python
if html and (want_box or not shown_at_the_bottom):
```

ONE PLACE, NOT TWO. Asked for on 2026-09-01: "i dont need the popup box if the tooltip is shown on the bottom of the window". The strip and the popup carried the same sentence, so the popup was a second copy drawn over the form the user was reading -- the same objection that moved the module blurbs to the bottom.

The popup still appears where there is no strip to write to, so a screen without one does not silently lose its help -- and that fallback survives BOTH switches being cleared, because "no tooltips" is a choice about the two surfaces and not a request to make a screen that has neither say nothing at all.

### lines 5904-5915

```python
HoverTooltip.instance().start_hide()
```

THE STRIP IS NOT CLEARED ON LEAVE, and that is the whole point of instruction 371's third part: "for the user to be able to press the botom tooltip API link ... the last setting the mouse hovered over should be shown, not only when the mouse hovers the setting. this way the user can hover then move the mouse to the link and click it, which is otherwise not possible."

Blanking here made the link unreachable by construction: it appeared only while the pointer was on the setting, and moving toward it removed it. The hold started in `_write_hint` clears the strip instead, ten seconds later or when another setting is hovered.

## AppScreen._default_hint

### lines 5984-5988

```python
return tr("Hover any setting for details and a link to its "
```

NO LONGER NAMES THE DOT. The information dots were removed

(instruction 258, "i like simplisity!"), so half of this sentence pointed at a control that is not drawn any more -- in ten languages. The API link did not go anywhere: it is in the hover tooltip, which is what the sentence now says.

## AppScreen._build_runtime_panel

### lines 6000-6001  _(unsure)_

```python
layout.setContentsMargins(SPACING["sm"], 0, 0, 0)
```

Small left inset so the console, chat and button row sit slightly away from the container's left edge (aligned with each other).

### lines 6005-6010

```python
from ..widgets.figure_queue import FigureQueue
```

Figures card — hidden until the pipeline pushes a figure via

PipelineWorker.figure_ready. Sits ABOVE the console (like the live-preview view). The FigureQueue widget owns the thumbnail strip + zoomable enlarged view + forward/back nav + the 100-in-RAM / temp-spill memory management. See spacr.qt.widgets.figure_queue.

### lines 6015-6041

```python
self._results_panel = None
```

THE REGRESSION MODULE OPENS INTO ITS RESULTS, NOT INTO PICTURES.

A finished regression used to be a stack of matplotlib figures whose last one -- the volcano -- cost ~115 ms per redraw and made the window lag, with the numbers behind it available only in a CSV. The Results tab draws the same volcano with Qt in ~4 ms, puts the coefficient table beside it, and links the two: click a dot to select its row, select a row to identify its dot.

THE RESULTS ARE BESIDE THE FIGURES, NOT BEHIND A TAB.

"the results for the regression shoild pop up in a container to the left of the figures" and "figures should pop up in a grid above the console ... if a figure is clicked it should fill the container". Both halves have to be on screen AT ONCE: picking a row changes one point in the volcano, which nobody can see if the table and the figure are two pages of one tab stack.

|  results             |  figure grid   (pressable tiles)| |  volcano + table     |     or one figure, filling it   |

|                    console                             |

Deliberately NOT the shape the parameter search gets (116): its runs are a tab, because picking a run replaces the whole grid. One is navigation between runs, the other is reading within a run.

### lines 6043-6044

```python
self._results_page = None
```

THE SECOND RUN, when the user has deliberately asked for one, and the stills of the runs that are not live (instruction 116).

### lines 6062-6064

```python
self._results_panel.refit_requested.connect(self._on_refit)
```

RE-FIT FROM THE PLOT. The panel decides what to run; this screen owns the worker, the console and the Stop button, so it is the one that can actually start it.

### lines 6067-6070

```python
self._figure_grid = FigureGridView(self._figures_card)
```

ALL of them at once, each at its own aspect ratio. A run makes seventeen figures and they are meant to be read together -- the fraction histogram explains the volcano which one-at-a-time navigation hides.

### lines 6074-6076

```python
self._figure_grid.figure_menu_requested.connect(
```

"all gigures should be editable by right clicking". The queue owns the matplotlib objects and already knows how to build the menu, so the tile just says which one and where.

### lines 6080-6083

```python
detail = QWidget(self._figures_card)
```

A clicked tile fills the container: same widget as before, wrapped so it is a PAGE of the stack rather than the stack's own child. _on_figure_ready calls _figure_queue.show(), and a bare show() on a stacked page draws it over the grid.

### lines 6100-6105

```python
volcano_page = QWidget(self._figures_card)
```

THE REGRESSION GRAPH GETS THE BIG HALF, AND IT IS THE LIVE

ONE. "that is the slowest graph and the one i want to be interactive." Left inside the results panel it was a thumbnail above its own table, while the pipeline's DEAD copy of the same plot took a full tile on the right -- two volcanoes on screen and the big one not clickable.

### lines 6117-6123

```python
gene_split = QSplitter(Qt.Vertical, volcano_page)
```

THE GENE TILE APPEARS WITH THE GRAPH. Instruction 121:

"when a gene is clicked a tile should appear with all the information on that gene" -- appear, beside the point that was clicked. A tile behind a tab the user has to go and find is a tile they will not look at. It starts collapsed and opens itself on the first click, so an unclicked screen is all graph.

### lines 6135-6145

```python
grid_page = QWidget(self._figures_card)
```

THE FIGURE SIZE IS THE USER'S (169 B). Reported: "cant control the height of the containers in the figures container in the measurements tab. i need to be able to make each taller". The grid has had `set_target_cell_width` all along and NOTHING CALLED IT -- a setter with no caller is a control that does not exist. This is the caller.

Width, not height, because the tiles keep each figure's own aspect ratio: setting the width IS setting the height, and a separate height control would either fight the aspect or distort the figure.

### line 6169, trailing  _(unsure)_

```python
self._figures_stack.addWidget(grid_page)
```

index 0

### line 6170, trailing  _(unsure)_

```python
self._figures_stack.addWidget(detail)
```

index 1

### line 6171, trailing  _(unsure)_

```python
self._figures_stack.addWidget(volcano_page)
```

index 2

### lines 6176-6179

```python
self._figure_grid.live_tile_activated.connect(
```

EVERY LIVE TILE, not only the volcano (199). The grid emits `live_tile_activated` for all nine panels it photographs; connecting only the pinned signal is what left eight tiles clickable and inert.

### lines 6184-6190

```python
self._results_panel.table.key_selected.connect(
```

The pinned signal stays connected for the volcano's sake it is the route that already works and the one older tests name -- and `_open_live_tile` returns early on that key so the graph is not raised twice. Picking a guide raises the graph its ring was drawn on. Highlighting a point on a view nobody is looking at is the same as not highlighting it.

### lines 6194-6200

```python
from ..widgets.sweep_runs import SweepRunsPanel
```

The parameter search is the main module setup plus one extra tab for the runs -- instruction 116, corrected by the maintainer, whose exact words are in that file. Not a bespoke screen with its own copies of the table, the queue and the results panel -- this screen, with one more tab. Picking a run swaps the figures on the right, which is the substance of that request and the only part unchanged.

### lines 6204-6217

```python
self._sweep_runs.loaded_run_changed.connect(self._show_trial)
```

WHICH RUN IS LOADED IS WHICH RUN IS ON SCREEN. Instruction 157, reported 2026-08-18: "even if the ols model is marked as loaded i still see the mixed results and no summary". `loaded_run_changed` was emitted from four places in the Runs tab and connected in none, so the mark moved and the coefficients, the figures and the summary stayed on the previous run -- and the empty summary was the same fault, the panel still holding the run the user had left.

THE SAME FUNCTION AS THE CLICK, deliberately. Two entry points into two loaders is how a run that becomes loaded by FINISHING ended up on a path nothing drove. `_show_trial` returns early for the run already on screen, so the two signals the Runs tab emits together cost one load.

### lines 6219-6225

```python
self._sweep_runs.loaded_run_changed.connect(
```

AND THE TWO TABS THAT READ THE RUN'S FOLDER. Both take zero-argument providers precisely so they can be re-read the scan panel's own docstring says "the tab must not go on showing the previous run's inputs" -- but the only thing that re-read them was OPENING the tab. Change the loaded run while the Cells tab is in front and it went on showing the previous run's cells, under the new run's name.

### lines 6228-6230

```python
self._sweep_runs.runs_removed.connect(self._on_runs_removed)
```

A RUN THAT LEAVES THE TABLE LEAVES THE OTHER VIEWS

(instruction 146). The panel keeps a plot state per run and the results tab may be showing the very run being removed.

### line 6232

```python
self._sweep_runs.compare_requested.connect(
```

TWO RUNS ON SCREEN AT ONCE, deliberately (116).

### line 6235

```python
self._sweep_runs.workspace_restore_requested.connect(
```

PUT BACK WHAT THAT RUN HAD OPEN (180).

### lines 6238-6240

```python
self._sweep_runs.set_photo_provider(self.run_photograph)
```

AND SHOW THE STILL OF A RUN THAT IS NOT LIVE (116). The photograph is taken when a run beside is closed; this is where it is finally seen.

### lines 6243-6251

```python
left.addTab(self._sweep_runs, "Runs")
```

RUNS FIRST, THEN RESULTS -- instruction 128 J, asked for on 2026-08-17: "the run tab should be before the results tab and results should be shown for the chosen run". The order is the reading order of the screen: pick a run on the left tab, read that run on the next one. Results was first while it was the only thing a finished fit had to show; now that picking a run re-points it (`_show_trial`), Results is the DETAIL of whatever Runs has selected, and a detail tab ahead of the thing it details reads backwards.

### lines 6253-6257

```python
self._results_split = QSplitter(Qt.Horizontal)
```

THE RESULTS PAGE IS A SPLITTER, not the panel itself

(instruction 116). A second run opened deliberately for comparison goes in beside this one, and a tab page cannot gain a sibling. Empty until somebody asks, so a screen that never compares pays for one child widget and no layout.

### lines 6272-6275

```python
from ..widgets.measurement_scan_panel import (
```

WHICH MEASUREMENT HAS GENES WITH A CLEAR EFFECT (122). Structurally the same thing as the sweep, with the DEPENDENT VARIABLE varying instead of the settings -- so it sits beside the runs rather than in a screen of its own.

### lines 6280-6282

```python
database_provider=self._attached_database_rows,
```

THE DATABASES THE USER DROPPED ON THE INPUT TABLE. Without this the tab builds and shows nothing, which is indistinguishable from having attached none.

### lines 6284-6289

```python
destination_provider=self._measurements_destination,
```

STEP 3's ARTEFACT AND STEP 4's SETTINGS (154 F). The merged frame is written once, and each column fit is this screen's own run with one thing changed: the response. Anything else varying would make the runs incomparable, which is the whole point of putting them in one table.

### lines 6293-6295

```python
self._column_run_handles = {}
```

ONE ROW PER COLUMN, PUT UP AS EACH FIT STARTS. A queue of twelve that showed nothing until it ended would be the freeze this instruction was filed about, one screen along.

### lines 6301-6306

```python
from ..widgets.sweep_panel import SweepPanel
```

EVERY GENE AGAINST EVERY MEASUREMENT (175), beside the scan because it is the same question asked of the whole screen at once. The three providers it needs -- the merged frame, the counts and the scores -- are all already on this screen; the panel was written to take them rather than to go looking, so it stays testable without any of this.

### lines 6313-6322

```python
self._sweep_panel.finished.connect(self._keep_the_effects_grid)
```

ITS OWN SECTION, not appended to the layout: a widget added to the layout takes its height out of the others, which is how the tab came to overlap. THE GRID GOES BESIDE THE RUN (186 A). `SweepResult.effects` lived only in this panel's memory: the montage's multivariate picker takes an `effects_grid` and nothing had ever set it, so it fell back to the single-score attribution every time -- in this session and in every other. Written to the run folder, it is there for the next session too, which a panel-to-panel handover cannot manage.

### lines 6326-6328

```python
try:
```

AFTER the last section is added, not in the panel's constructor: a fold restored before the sweep section exists cannot be applied to it (169 C).

### lines 6342-6361

```python
from ..widgets.cell_montage_view import CellMontageView
```

THE CELLS BEHIND A DOT ON THE VOLCANO (131). "there should be an option to visualize the cells most likely to represent dots on the regression plot ... to show to the user in a new tab where the figures are."

A TAB, NOT A DIALOG, and it is always present. Instruction 131 C and 129 both settled that: one tab per view, named, and a tab that cannot be filled SAYS WHY rather than being absent -- which for this one is the common case, because most runs have no measurement database attached and the montage needs per-object rows.

REACHED FROM THE SELECTION THAT ALREADY EXISTS. The panel's

`table.key_selected` is the funnel every plot and the table pass through -- volcano -> table.select_key -> selection change -> re-emit -- and the gene tile is already on it. A second selection mechanism here would mean a montage of a different gene from the one the volcano is ringing, which is the plausible-and-wrong output this module is most careful about.

### lines 6368-6372

```python
cells_tab = left.addTab(self._cell_montage, "Cells")
```

THE INDEX `addTab` RETURNS, not the literal the tabs above it use. This is the last tab, so anything inserted ahead of it moves it -- and a tooltip on the wrong tab is not a visible failure, it is a sentence about the Measurements tab appearing over the Cells one.

### lines 6383-6387

```python
self._results_panel.table.keys_selected.connect(
```

AND THE WHOLE SELECTION (instruction 206). `set_coefficients` holds all of them and shows the most recent, so the count in this tab is the count on the volcano; without it a band over four guides would leave the Cells tab describing one with nothing saying the other three were picked.

### lines 6391-6394

```python
left.currentChanged.connect(self._on_results_tab_changed)
```

Read the table when the tab is OPENED, not on a timer. A sweep writes each trial as it finishes, so the answer is different every time somebody looks -- and nobody is looking while the tab is behind another one.

### lines 6397-6403

```python
left.setCurrentWidget(self._results_page)
```

AND RESULTS IS STILL WHAT OPENS. 128 J changed the ORDER of the tabs, not which one a finished regression lands on "a finished regression opens into its results, not into a run list" is the property `test_results_is_what_opens_first` has held since the Runs tab arrived, and nothing in J asks for it back. Set explicitly because QTabWidget's own default is index 0, which is now Runs.

### lines 6410-6418

```python
split.setStretchFactor(0, 1)
```

THE RESULTS SIDE IS THE WIDER ONE NOW. Asked for on 2026-08-17 -- "the left panel should be wider" -- and it is the side that grew: it holds a coefficient table with a dozen columns, the volcano, and a row of diagnostic tabs, while the grid opposite reflows to whatever it is given.

An EQUAL stretch rather than a reversed one, so a wider window still feeds both. The divider stays the user's to move; these are starting sizes, not a layout.

### lines 6421-6424

```python
left.setMinimumWidth(520)
```

Floors, not preferences -- neither side survives being handed a size hint. The results floor went up with the share: a coefficient table under 520 px shows its index and one column.

### lines 6431-6434

```python
self._grid_refresh = QTimer(self)
```

ONE REBUILD PER BURST. A pipeline streams seventeen figures in one at a time and the grid is now always on screen, so rebuilding per arrival is seventeen full relayouts. The timer collapses a burst into a single one.

### lines 6446-6452

```python
self._figure_queue.set_propagate_callback(
```

"Figure settings…" on the NON-LIVE figure holds every Image UMAP setting, live against the figure on screen (instruction 75), and a Propagate button. Propagate means the same thing here as everywhere else in the app: write the values into THIS module's settings panel, which is what the next Run reads and what is saved with the run. Wired for every module, not just UMAP -- the figure colours and text size propagate the same way.

### lines 6462-6467

```python
self._umap_explorer.set_propagate_callback(
```

The same propagate seam the Mask live preview uses, so a value tuned in the explorer's display window lands in the settings panel and is saved with the run rather than living only in the widget. The getter lets that window open showing the CURRENT run settings for the half it does not itself hold -- without it, figure size and image count open as zeros.

### lines 6473-6476

```python
self._figures_card.setMinimumHeight(
```

360 was the height of a card holding ONE figure. It now holds a coefficient table, a volcano and a grid of tiles side by side, and at 360 every one of them is a scrollbar with a sliver of content behind it -- measured on the real screen before this was raised.

### lines 6480-6482

```python
from ..widgets import ConsolePanel
```

NOT added to the layout here. It goes into the vertical splitter built below, so the figures can be dragged taller at the console's expense -- a volcano needs the room a log line does not.

### lines 6484-6488

```python
from ..widgets import ConsolePanel
```

Live-preview segmentation — Mask app only. The card + the console below live in a vertical QSplitter so the user can drag the divider up (bigger console) or down (bigger preview) depending on whether they're tuning parameters or watching a run. Non-Mask apps get the console alone.

### lines 6500-6502

```python
self._console = ConsolePanel(active_app_label=app_title,
```

`persist_key` is what lets the console remember where the user put the divider between its output box and the AI chat box, per screen: a tall chat box on Mask does not force one on Sequencing.

### lines 6507-6510

```python
from ..widgets.foldable import make_foldable
```

CLICKING "Console" FOLDS IT (instruction 228), which the maintainer asked for in those words. The minimum height has to go with it: a hidden widget contributes nothing to a layout, but a minimum on the WRAPPER would hold the strip 180px tall over nothing.

### lines 6518-6521

```python
self._live_preview = self._live_preview_card = None
```

Exactly one of these cards occupies the slot above the console. Nulled here rather than in every branch: the chain has grown to six arms and a branch that forgets one leaves a stale attribute from a previous screen.

### lines 6535-6536  _(unsure)_

```python
self._live_preview.set_propagate_callback(
```

Let the live preview push tuned settings into the main panel when its "Propagate settings" toggle is on.

### lines 6542-6545

```python
splitter.setStretchFactor(0, 3)
```

THREE PANES, THREE NUMBERS. The order after the insert is figures / preview / console, and this used to set two stretch factors and two sizes -- so the console was never given one and took whatever Qt had left, which at 1200x900 is about 200 px.

### lines 6553-6556

```python
from ..widgets.timelapse_preview import build_timelapse_preview_card
```

Timelapse takes the same slot Mask and Measure use for Live

Preview. Segmenting the sequence is the expensive half and is cached on a signature that deliberately excludes the tracking settings, so re-linking while tuning them costs nothing.

### lines 6567-6570

```python
splitter.setStretchFactor(0, 3)
```

THREE PANES, THREE NUMBERS. The order after the insert is figures / preview / console, and this used to set two stretch factors and two sizes -- so the console was never given one and took whatever Qt had left, which at 1200x900 is about 200 px.

### lines 6588-6591

```python
splitter.setStretchFactor(0, 3)
```

THREE PANES, THREE NUMBERS. The order after the insert is figures / preview / console, and this used to set two stretch factors and two sizes -- so the console was never given one and took whatever Qt had left, which at 1200x900 is about 200 px.

### lines 6608-6611

```python
splitter.setStretchFactor(0, 3)
```

THREE PANES, THREE NUMBERS. The order after the insert is figures / preview / console, and this used to set two stretch factors and two sizes -- so the console was never given one and took whatever Qt had left, which at 1200x900 is about 200 px.

### lines 6619-6621

```python
splitter = QSplitter(Qt.Vertical)
```

The regression module gets a Parameter sweep card in the same place, behind the same kind of toggle, as the Hyperparameter search the other modules have. Same feature, same shape.

### lines 6632-6635

```python
splitter.setSizes([720, 300, 220] if self._results_panel is not None
```

The figures slot carries the results table AND the figure grid on the regression screen, so it opens with room for both. The sweep card is collapsed behind a toggle until asked for, and the divider is the user's to move.

### lines 6641-6644

```python
from .hyperparam import build_hyperparam_card
```

umap / classify / ml_analyze get a Hyperparameter search card in the slot Mask and Measure use for Live Preview: same shape, same threading contract, and its Apply reuses the same route back into the settings panel.

### lines 6655-6658

```python
splitter.setStretchFactor(0, 3)
```

THREE PANES, THREE NUMBERS. The order after the insert is figures / preview / console, and this used to set two stretch factors and two sizes -- so the console was never given one and took whatever Qt had left, which at 1200x900 is about 200 px.

### lines 6666-6667

```python
splitter = QSplitter(Qt.Vertical)
```

A plain vertical splitter so the figures/console divider is draggable here too, which is where the regression module lives.

### lines 6678-6681

```python
try:
```

Route the verbose logger (if the user turned it on in

Preferences) at THIS screen's console. Only the last-focused screen receives the log stream — that's fine, users hit the console they're looking at.

### lines 6688-6690

```python
usage_card = Card(title="System", foldable=True,
```

Usage card. FOLDABLE: clicking "System" folds it, which the maintainer asked for in those words, and it sits directly under the container that wants the room.

### line 6700  _(unsure)_

```python
cpu_row = QHBoxLayout()
```

CPU row: single "CPU" bar + a toggle chevron button.

### lines 6713-6714  _(unsure)_

```python
cpu_wrap.setStyleSheet("background: transparent;")
```

Transparent so the System card surface (not the global black QWidget bg) shows behind the CPU bar + Per-core button.

### line 6719  _(unsure)_

```python
self._per_core_wrap = QWidget()
```

Per-core panel — hidden by default; one UsageBar per logical core.

### lines 6731-6733

```python
actions = QWidget()
```

Actions row. Flush-left (no extra inset) so Run / Stop / Import / Clear / Explain line up with the console, chat and System panel, which all share the runtime panel's small left inset.

### lines 6735-6738

```python
self._actions_row = actions
```

Kept so `_clear_page_surfaces` can tag it. Untagged it inherits the blanket `QWidget { background-color: bg }` rule and paints an opaque strip behind Run / Stop — a black box no opacity setting could reach, because it is the window colour rather than a surface.

### lines 6744-6752

```python
buttons = _WrappingButtonStrip(SPACING["sm"])
```

THE CAPTIONED BUTTONS WRAP; NOTHING ELSE DOES. See

`_WrappingButtonStrip` for the measurement and for why this is a sub-layout rather than a widget of its own -- in one line: the buttons stay Qt children of `_actions_row`, which is what everything that reaches into this row already depends on. Copy console and the Preferences gear enter it as ONE item, so a wrap cannot separate them (see below), and everything after `row.addStretch(1)` -- the progress bar and the switches -- stays in the horizontal row it was always in, held against the right edge by that stretch.

### lines 6792-6807

```python
from ..widgets.activity_spinner import ActivitySpinner
```

THE BACKGROUND-ACTIVITY SPINNER, BUILT HERE RATHER THAN FOUND LATER, and this is a consequence of the split above rather than a preference. `activity_spinner.attach_activity_spinner` normally installs it lazily from the global button filter, by asking `_btn_clear.parentWidget().layout().indexOf(_btn_clear)` and inserting at index + 1. Every part of that contract survives the buttons moving into a sub-layout except one: `QLayout.indexOf` does not descend into a sub-layout, so it would answer -1 and the helper would return None -- the spinner silently never installed, and in the real application only, because every test of that helper builds its own flat row and would have gone on passing.

Building it here puts it exactly where the helper would have ( immediately right of Clear console) and sets the attribute the helper checks BEFORE it touches any layout, so the lazy path finds this one and returns it instead of trying to insert a second.

### lines 6812-6813

```python
self._btn_copy_console = QPushButton("Copy console")
```

Beside Clear, because the two are the same kind of act on the same thing — and because the console is what a bug report is made of.

### lines 6820-6821

```python
from .. import iconset as _iconset_prefs
```

NOT ADDED HERE. It enters the strip welded to the Preferences gear a few lines below -- see the comment there.

### lines 6823-6828

```python
from .. import iconset as _iconset_prefs
```

Preferences, to the right of Copy console. Every module screen gets it because every module screen is somewhere a user notices the font is too small or the backdrop is costing frames -- and the alternative is the menu bar, which is a trip out of the work. Icon-only: the row is already three words wide and a gear is the one glyph nobody has to be taught.

### lines 6834-6841

```python
from ..preferences import scaled_px
```

SIZE THE GEAR, or it is Qt's 16px default forever. This project ships a 1.5 default font scale, so every label beside it renders half again as large while the icon stays 16px in a 44px button which is why it was reported as "I cannot see the gear". The icon was never missing; it was rendering at a third of the button.

Scaled with the font rather than fixed, so it keeps its proportion at every zoom level.

### lines 6850-6872

```python
copy_and_gear = QWidget()
```

COPY CONSOLE AND THE GEAR TRAVEL TOGETHER, as ONE item of the wrapping strip, and this pair exists because a wrap can otherwise separate them. The gear was asked for BY POSITION -- "to the right of Copy console" -- and `tests/qt/test_preferences_gear.py` checks exactly that, as `gear.x() > copy.x()` with both on one parent. Left as two independent items the wrap put them on different lines the moment the strip ran short: on Mask at 1400x900 the strip has 678 px and one line of buttons wants 699, so the gear went to line two and its x fell from 661 to 0. It is a 46 px icon at the end of a 605 px run of captions, so it is always the item the wrap reaches first.

The alternative was to take the gear out of the strip entirely and make it a fixed item of the row. That also satisfies the position, and was rejected on two measurements: it left the gear floating at mid-height beside a two-line stack of buttons instead of sitting among them, and its 46 px came off the strip's width at every window size -- Measure in German at 1000 px went from three lines of buttons to four, and the console under it from 208 px to 180.

An anonymous QWidget on purpose: `theme.clear_container_surfaces` tags exactly that -- a plain QWidget with no object name is scaffolding -- so this cannot become another opaque strip over the backdrop the way an untagged container does.

### lines 6881-6882  _(unsure)_

```python
from .. import iconset as _iconset
```

(The manual "Explain error" button was removed — errors now route to the AI automatically when AI is enabled; see _on_pipeline_error.)

### lines 6885-6887

```python
self._btn_file_issue = QPushButton("File as issue")
```

File as GitHub issue — same enable gate as Explain, plus the user's opt-in in AI Settings. Opens a pre-filled issue URL in the default browser; the user reviews and hits Submit.

### line 6905, trailing  _(unsure)_

```python
self._progress.setRange(0, 0)
```

indeterminate until we know

### lines 6910-6914

```python
from ..widgets import AiToggleLabel
```

Runtime-preview toggle — every app with a preview gets the same bottom-right control Mask established for Live Preview. Keeping this in the shared actions row prevents Timelapse, Motility and Measure from permanently taking half the console merely because their preview card exists.

### lines 6917-6920

```python
self._install_dimension_switches(row, AiToggleLabel)
```

3D and Time — "to the left of the Live button which sitts to the left of the AI button". Built here, before the preview toggle, so the row reads 3D · Time · Live · … · AI whether or not this module has a preview to switch on.

### line 6951  _(unsure)_

```python
if self.app_key == "mask":
```

Preserve the public name used by existing Mask integrations.

### lines 6956-6961

```python
self._ops_switch = None
```

OPS -- Mask Generation only, beside Live and the dimension switches and in the same format, as asked. Optical pooled screening is folded onto this screen: it opens as a page rather than mounting settings on this form, so it is a switch here rather than a button on the masthead strip, which carries the folds that ARE settings. See `spacr.qt.screens.mask.PAGE_FOLDS`.

### lines 6973-6974

```python
LOG.debug("Could not install the OPS switch", exc_info=True)
```

A screen without the switch is a smaller screen; an exception here would be no Mask Generation at all.

### lines 6977-6979

```python
self._gpu_switch = None
```

Image UMAP has one GPU switch for both its main run and its search. It deliberately lives in the action strip instead of being repeated in the settings form, and precedes Hyperparameter search as requested.

### lines 6994-6995

```python
if getattr(self, "_sweep", None) is not None:
```

Same slot, same behaviour, for the apps that have a hyperparameter search instead of a live preview.

### line 7003, trailing  _(unsure)_

```python
self._on_sweep_switch(False)
```

start collapsed, like the rest

### line 7011, trailing  _(unsure)_

```python
self._on_hyperparam_switch(False)
```

start collapsed, like Live

### lines 7013-7023

```python
self._interactive_switch = None
```

Interactive image-UMAP explorer — UMAP only, immediately beside AI. It starts off so ordinary runs retain the familiar static figure. Turning it on before or after a run switches the same payload to the click / image-preview / lasso / database-annotation interface.

It says "Interactive", not "Live". A LIVE view re-renders a module's own output from the current settings before a run — Mask, Timelapse, Measure and Motility, all four of which now share one contract (spacr.qt.widgets.preview_contract). This explorer is not one of those: it makes an already-computed embedding clickable, and no setting changes what it draws. One word for one thing.

### lines 7036-7041

```python
queue = getattr(self, "_figure_queue", None)
```

Clicking the STATIC figure turns Live on. The request was "i should be able to press every point" -- pressing a point on a rendered PNG means hit-testing pixels back to the embedding, a second and fragile implementation of what the explorer already does properly. So the click takes you to the view where pressing points works, instead of building that twice.

### lines 7047-7050

```python
self._ai_switch = AiToggleLabel()
```

AI toggle + provider dropdown, bottom-right of the actions row. AI switch is a plain clickable text label — white when off, accent blue when on. Chevron next to it exposes the provider picker + install/login dialog.

### lines 7055-7064

```python
self._apply_ai_default()
```

NO PROVIDER CHEVRON HERE ANY MORE. A "▾" beside the AI switch opened a provider picker on the actions row of every module, which put a PREFERENCE -- which assistant do I use -- in the place where per-run choices are made, and repeated it on each screen. It moved to Preferences → AI, where the answer is given once.

"AI assistant on at launch" IS WHAT THIS CONTROLS. The preference was written by the setup screen and read by nothing, so a user who turned it on met a grey AI switch on every module and a setting that had done nothing.

### lines 7069-7077

```python
self._category_hint_pinned = ""
```

Category strip — the settings CATEGORY blurb, immediately under the Run / Stop row. A category groups tens of settings (Organelle Segmentation groups fifty-three), so its description is a paragraph, and a paragraph-sized popup hovering over the settings panel covers the very controls it is describing. It gets a fixed region here instead: hovering a category header fills it, expanding one pins it, and it holds the pinned category while the pointer wanders back into the form. The per-setting strip below shows the setting under the cursor, so the two read as "where you are" then "what this does".

### lines 7081-7084

```python
self._category_hint.setStyleSheet("background: transparent;")
```

Named widgets keep their fill under the blanket

`QWidget { background-color: bg }` rule, and this one is a caption over the backdrop, not a surface — the same reason `cpu_wrap` above carries the declaration.

### lines 7092-7093  _(unsure)_

```python
self._hint_strip = QLabel(self._default_hint())
```

Hint strip — hover-follows caption that shows the current settings tooltip regardless of Qt HTML-tooltip rendering.

### lines 7100-7101

```python
self._hint_strip.linkActivated.connect(self._on_hint_link)
```

`linkActivated` still fires for a scheme Qt will not open, which is what makes the private href above work beside the real API URL.

## AppScreen._remember_runtime_splitter

### lines 7133-7134

```python
pass
```

A blob from an older layout restores nothing rather than raising; the default split is the right fallback.

## AppScreen._write_hint

### line 7170

```python
lines = HINT_STRIP_LINES - (1 if url else 0)
```

The link costs a line, so the body is fitted into what is left.

### lines 7177-7189

```python
animation_link = (
```

"API", not "Open spaCR API documentation". Instruction 371:

"which should also just say API". The long form repeated on every setting and the strip has four lines to spend. "Animation" BESIDE "API", which instruction 371 asks for on both surfaces: "an API link and Annimation link text ... same for the botom tooltips". Only when this setting HAS one 141 do, and a word that visibly does nothing is worse than no word, which is the rule the popup's own footer already follows.

The href is a private scheme rather than a URL. The strip has `setOpenExternalLinks(True)` for the API link, and a real scheme here would hand the animation to a browser.

### lines 7199-7200  _(unsure)_

```python
strip.setToolTip(str(text))
```

The untrimmed text stays reachable: the tooltip is what a reader who wants the rest, or a screen reader, asks for.

### lines 7202-7205

```python
self._hold_the_hint(hold)
```

`hold` IS PASSED, NOT INFERRED. Inferring it from "the text is not empty" starts the timer on the DEFAULT prompt too, and since `_release_the_hint` writes that prompt, the strip would restart its own hold forever. Only a hovered setting holds.

## AppScreen._sync_category_hint_height

### lines 7283-7285  _(unsure)_

```python
def _sync_category_hint_height(self) -> None:
```

Category help — the strip under the actions row

## AppScreen._watch_for_late_captions

### lines 7314-7325

```python
hosts = (getattr(self, "_runtime_wrap", None),
```

The runtime panel is where a preview CARD is inserted, and the body splitter is where the settings strip -- which the toggle beside it goes on -- is inserted. The strip does not exist yet: it is installed on the screen's first show, one hook before the preview, so by the time this pass runs the toggle is already inside the pane it arrived in and is translated with it. The screen itself is the third host, and it is where a fold page strip arrives: `spacr.qt.screens.map_barcodes.host_pages` wraps this screen's body in a `QTabWidget` parented HERE, and folded modules become pages on it. The outer layout is finished long before this hook runs, so watching the screen costs nothing on a module that folds nothing.

## AppScreen._wire_category_hints

### lines 7364-7369

```python
title = (section.property("settingsCategorySource")
```

THE CATEGORY AS IT IS WRITTEN, not as the header shows it. `Section.title()` answers with the uppercased caption, and a catalog keyed on "Preview & Diagnostics" has nothing under "PREVIEW & DIAGNOSTICS" — so the strip would head a translated blurb with an English title. The written name is kept on the section for exactly this, and uppercased after the lookup.

## AppScreen.show_category_hint

### lines 7404-7406

```python
heading = tr(str(title or "")).upper().strip()
```

Translated first, uppercased second. The other order asks the catalog for a caption nobody wrote and leaves an English word in bold at the head of a translated sentence.

## AppScreen.hideEvent

### lines 7449-7450  _(unsure)_

```python
"""Let the screen stop paying for things nobody can see.
```

Keep this Qt lifecycle hook out of the documented spaCR API: it is only the inverse of the showEvent timer activation above.

## AppScreen._on_run

### lines 7500-7503

```python
if self.app_key == "measure" and not self._confirm_crop_choices(
```

WHAT THE CROP SETTINGS WILL COST, said before the run rather than after it. Both of these quietly change what every downstream model sees, and neither is recoverable without measuring again -- which is twenty minutes a plate.

### lines 7511-7515

```python
from ..theme import active_palette
```

Resolve GUI colours on the GUI thread and pass plain strings to the worker. The UMAP canvas sits inside a Card, whose material is ``surface_alt`` in every theme; matching that color avoids a black/white rectangle inside dark, light, image, and glass themes. Avoid reading QApplication/QSettings from the worker.

### lines 7524-7527

```python
log_button_press(
```

Diagnostic breadcrumb — visible when the user has verbose logging on. Shows exactly which app + entry-point ran and (truncated) which settings were passed. Helps triage "Starting mask… (hangs)" reports.

### lines 7536-7539

```python
entry_name = getattr(entry, "__qualname__", repr(entry))
```

Also always print a compact one-liner into the Console so non-verbose users see the entry point name — this is what they were missing when the console just said "Starting mask…" and nothing else.

### lines 7541-7542  _(unsure)_

```python
try:
```

Tell the console which module/function this output is from so its "spaCR output — <module> — <function>" banner is accurate.

### lines 7559-7560  _(unsure)_

```python
import time as _time
```

Remember start time so _on_finished can report elapsed to the run journal + the OS notification.

### lines 7563-7566

```python
_pause_the_fractal(self)
```

WIND THE BACKDROP DOWN. Under spaceout the fractal holds nineteen Numba threads, which is exactly the machine the run wants. Stopping is better than slowing: a thinner fractal still owns the threads. The last frame stays on screen, so nothing blinks out.

### lines 7568-7576

```python
import datetime as _dt
```

EACH RUN IS ITS OWN SECTION ON THE GRID, AND A ROW IN THE RUNS TAB. Marked at the START rather than when the first figure arrives, so a run that draws nothing still appears as a section that drew nothing which is a fact worth seeing rather than a gap.

ONE LABEL FOR BOTH. The grid heading and the runs row name the same run, and two labels generated separately are two clocks: a user looking at "run 14:32:05" on the grid has to be able to find it in the table.

### lines 7579-7582

```python
source = SOURCE_REFIT if override is not None else SOURCE_RUN
```

A RE-FIT IS A RUN, AND SAYS SO. `override` is what the re-fit passes and nothing else does (see the docstring above), so this is the one place that can tell the two apart -- by the time the worker starts they are the same call.

### lines 7592-7596

```python
try:
```

The one preference the PIPELINE needs to know about, passed as an ordinary setting. The pipeline must never read QSettings -- a `from PySide6 import` in a pipeline module makes the package unimportable on a cluster -- so the GUI reads it here and a headless caller sets the same key itself.

### lines 7603-7605

```python
self._announce_the_fit(settings)
```

A LONG FIT SAYS IT WILL BE LONG (instruction 140). Before the worker is even built, so the sentence is on screen ahead of the first line the run prints.

### lines 7609-7615

```python
self._worker = worker
```

Keep a strong reference to the worker on ``self``. PySide6 does NOT keep a QObject alive through a bound-method signal connection (thread.started → worker.run), so a local-only ``worker`` can be garbage-collected before run() fires — the thread then spins its event loop forever and the pipeline never starts. Storing it here fixes an intermittent "pressed Run, nothing happens" hang.

### lines 7620-7622

```python
worker.result_ready.connect(self._on_pipeline_result)
```

THE RUN HANDS BACK ITS OWN RESULTS. Reading the CSV meant guessing which of four nested folders the run had written to, and a guess is how a screen shows last month's table or an empty one.

### lines 7626-7630

```python
self._thread.finished.connect(self._clear_thread_refs)
```

Clear our Python references only once the QThread has genuinely stopped (its event loop exited). Dropping them from _on_finished — which runs on worker.finished, before thread.quit() has taken effect — could destroy the QThread while it is still "running" ("QThread: Destroyed while thread is still running" → abort).

## AppScreen._announce_the_fit

### lines 7749-7752

```python
if self._it_will_permute(settings):
```

Explicit permutation runs use a dedicated banner because they do not fit a regression model or use ``regression_type``. Automatic inference retains the model banner until the design scan resolves the method from the guide and well counts.

### lines 7770-7772

```python
scan = dict(settings or {})
```

THE DESIGN, off the GUI thread. `submit` returns before the read starts; the sentence lands a moment after the run's own first line and says which it is.

## AppScreen._on_copy_console

### lines 7916-7922

```python
self._btn_copy_console.setText(tr("Copied"))
```

TRANSLATED AT THE MOMENT OF WRITING, all three. The language pass ran when the screen was built and does not run again, so an English literal set by a handler is English for the rest of the session -- and worse, `retranslate_widget_tree` reads a caption it did not render as data and opts the widget out of every later pass. Pressing Copy console on a Swedish screen used to leave the button reading "Copy console" for good.

## AppScreen._on_pipeline_error

### lines 7946-7949

```python
routed = False
```

Route through AI when AI is enabled with a provider AND the route-errors-through-AI preference is on (the default). The user then sees the AI's explanation + instructions; the raw traceback stays hidden (the AI still has it, so the user can ask it to show the error).

### lines 7963-7965

```python
try:
```

File-as-issue button becomes visible only when the user has opted in via AI Settings — otherwise it stays hidden so the actions row doesn't grow noise for people who don't use it.

### 2026-09-19: the failure says the report was not sent

```python
self._report_waits_for_a_click = bool(
```

GitHub #117: with "Report errors as GitHub issues" on, a user watched a Mask run fail and expected an issue to have been filed, and none was. That is deliberate. Until 807ba9e0a (2026-08-14, instruction 45) this method did file on its own: it called `_on_file_issue` as soon as the crash arrived. The consent work removed that, because the destination is the PUBLIC tracker, and a report now goes out only after the user presses Send in the editable preview. What was missing was any word of this in the console. The only sign was the "File as issue" button appearing in the row under it.

So `_on_finished` now writes one line directly under "✗ Failed": `[issue] Nothing was sent to GitHub. Reports are public, so spaCR files one only when you press File as issue and then Send report.` It is written once per failure, only when the button is on offer, and not when reporting is set to 'never', since the button then refuses to file. A run that was stopped drops the pending line, so it cannot turn up under the next failure.

Left as found and recorded in features/new/432: `ISSUE_PROMPT_ALWAYS` is still one of the first-run "One-click issue filing" choices, and since 807ba9e0a nothing reads it. 'always' and 'ask' behave the same. Whether 'always' should open the preview by itself when a run fails, or be removed, is the maintainer's call.

## AppScreen._on_lp_switch

### lines 7973-7975

```python
def _on_lp_switch(self, on: bool) -> None:
```

Opting in reveals the action; it never submits in response to the crash itself. Every report stops at an editable public-payload preview and needs a report-specific Send click.

### lines 7977-7979  _(unsure)_

```python
def _on_lp_switch(self, on: bool) -> None:
```

AI toggle + provider menu — sits in the actions row (bottom right)

## AppScreen._on_hyperparam_switch

### lines 8040-8041

```python
model = getattr(self, "_settings_model", None)
```

Seed the search space from whatever is currently in the panel, so the sweep starts from the user's settings rather than defaults.

## AppScreen._on_sweep_switch

### lines 8053-8055

```python
model = getattr(self, "_settings_model", None)
```

Seed the sweep from what is in the settings panel, so it starts from the user's inputs rather than from defaults they would have to retype.

## AppScreen._on_umap_gpu_switch

### lines 8071-8072

```python
enabled = bool(panel.request_gpu_enabled(
```

THE SWITCH IS THE ANCHOR, so the panel opens under the control the user just pressed rather than under the search panel it belongs to.

## AppScreen._on_ai_switch

### line 8161  _(unsure)_

```python
from .. import ai as ai_module
```

Auto-pick first available provider if none selected yet.

## AppScreen._wanted_provider

### lines 8187-8189

```python
try:
```

A PREFERENCE IS A WISH, NOT A GUARANTEE. The CLI it names can be uninstalled between sessions, and honouring the name regardless would route every question to something that is not there.

## AppScreen._on_explain_error

### lines 8204-8206

```python
self._console.open_error_flow(self._last_error_text, self.app_key)
```

Route the traceback into our own merged console — no more side-panel navigation. Keep the legacy signal too, for MainWindow's old dock path.

## AppScreen._on_file_issue

### lines 8230-8231

```python
from ..preferences import ISSUE_PROMPT_NEVER, get_issue_prompt_mode
```

The preview itself is the prompt and the consent boundary. The legacy mode remains respected so Preferences can revoke reporting.

### lines 8239-8240

```python
settings_snapshot: dict = {}
```

Best-effort settings snapshot from the current settings model so the issue includes what the user was trying to run.

### lines 8257-8260

```python
settings_snapshot[k] = w.get_value()
```

The chip editor is a QWidget, not a QLineEdit; a bug report that omitted every list setting was how the class_metadata crash arrived without its own value attached.

### lines 8271-8276

```python
try:
```

THE AI'S OWN ANALYSIS RIDES ALONG when spaCR AI is switched on and has already answered THIS error -- which, in the flow that files these reports, it usually has, because the console offers to explain a crash the moment it happens. Empty when the AI is off, when it has not answered, or when its last answer was about something else; see `ConsolePanel.ai_explanation_of`.

### lines 8288-8289

```python
preview = IssuePreviewDialog(
```

The console and the raw traceback go with it, so its Diagnose button can ask spaCR AI about this error and add the answer.

## AppScreen._on_file_issue._file

### lines 8300-8304

```python
"""Submit the report, returning the failure AS DATA rather than raising.
```

The failure is carried back as data rather than raised. The auto-file path used to wrap this call in `try/except` to print "[issue] auto-file failed"; once the call is asynchronous that `except` can no longer see it, and a report that silently fails to send is worse than one that fails loudly.

## AppScreen._on_figure_ready

### lines 8424-8425

```python
self._figure_queue.add_figure(
```

Keep the ordinary figure too: switching Interactive off should restore it immediately rather than requiring another UMAP run.

### lines 8448-8449  _(unsure)_

```python
self._queue_figure_grid_refresh()
```

The grid is on screen while the run streams, so it has to grow with it. Debounced, so seventeen arrivals are one relayout.

## AppScreen.closeEvent

### lines 8459-8461

```python
self._stop_the_heartbeat()
```

The heartbeat outlives the run it describes unless it is stopped: a QTimer parented to this widget keeps firing until the widget is destroyed, and its slot touches the console.

### lines 8487-8488  _(unsure)_

```python
try:
```

Stop polling before shutting the runner down, or the 2 s timer can start one more job while `shutdown` is draining the last.

### lines 8494-8497

```python
for name in ("_usage_jobs", "_jobs"):
```

The usage poll and the issue report are abandoned rather than waited for: neither writes anything a half-finished copy of would damage, and `shutdown` parks any that outlast its budget instead of terminating them mid-call.

### lines 8505-8510

```python
montage = getattr(self, "_cell_montage", None)
```

THE CELLS TAB HAS A WORKER OF ITS OWN, and for the same reason the exclusion editor below does: it is a child widget, so navigation destroying this screen never gives it a close event to shut its loader down from -- and a QThread destroyed while running aborts the process, which a seconds-long merged-source montage makes an ordinary case rather than a rare one.

### lines 8517-8521

```python
flowview = getattr(self, "_flowview_section", None)
```

Classify's FlowView footer owns a refresh timer and a graphics scene.  It is a direct child of the settings content rather than a SettingsWidgets field, so the generic settings shutdown below does not see it; close it explicitly while its Qt objects are still alive.

### lines 8528-8531

```python
self._shutdown_settings_widgets()
```

The settings panel's own background work goes with the screen. The exclusion editor reads distinct values off a worker, and it is a child widget, so navigation destroying the panel never gives it a close event of its own to shut that down from.

### lines 8533-8536

```python
try:
```

Instruction 180: a screen that is gone contributes nothing to a saved run. Withdrawn HERE and not left to the registry's own callables to fail, because a provider that raises every time is reported as a problem in every workspace document afterwards.

### line 8541  _(unsure)_

```python
fq = getattr(self, "_figure_queue", None)
```

Clean up the figure queue's temp dir if present.

### lines 8554-8559

```python
try:
```

pyqtgraph deliberately makes PlotItem/ViewBox context menus parentless top-level windows. The ordinary QWidget close cascade cannot reach them, so one closed regression screen otherwise leaves hundreds of live widgets for every later palette/style pass. Retire only menus found in this screen's own graphics scenes; a global QApplication sweep or gc.collect over live Qt wrappers is unsafe.

## AppScreen._on_finished

### lines 8602-8604

```python
self._stop_the_heartbeat()
```

BEFORE ANYTHING ELSE. A heartbeat that fires after the run has finished says "still fitting" underneath "Finished", and the last line of a console is the one a user reads.

### lines 8606-8607

```python
_resume_the_fractal(self)
```

AND GIVE THE BACKDROP ITS CORES BACK, whether the run finished or failed -- `_on_finished` is the one door both take.

### lines 8616-8619

```python
import time as _elapsed_time
```

THE RUNS TAB LEARNS HOW IT WENT. Its row said "running" from the moment the run started; leaving it there would make every finished run look like one still in flight, and picking it would be refused for a run whose results are sitting on disk.

### lines 8632-8639

```python
if (ok and not cancelled and getattr(self, "_results_panel", None)
```

NOTE: do NOT drop self._thread / self._worker here. This slot runs on worker.finished, i.e. before thread.quit() has actually stopped the QThread's event loop; releasing the last reference now can destroy the still-running QThread and abort the process. The references are cleared from _clear_thread_refs, wired to the QThread's own finished signal. A finished regression has a coefficient table on disk; open into it rather than leaving the user to find the CSV.

### lines 8648-8650

```python
try:
```

OS-level notification (libnotify / osascript / win10toast) so users don't have to sit and watch. Always safe — the notify module fails silently on any error.

## AppScreen._record_run_in_runs_tab

### lines 8664-8666  _(unsure)_

```python
def _record_run_in_runs_tab(self, label, source, settings):
```

The Runs tab: every run, not only the sweep's trials

## AppScreen._pin_regression_graph

### lines 8744-8748

```python
if frame is None or not len(frame):
```

NOTHING FITTED, NO TILE. An empty plot tile invites a click that opens an empty plot, and before a run there is nothing on the volcano to photograph anyway -- `snapshot` returns None for that too, but asking here keeps the grid from flickering a tile in and out while a run streams its first figures.

### line 8750

```python
grid.set_pinned(None, "")
```

NOTHING FITTED, NO TILES.

### lines 8754-8766

```python
from ..widgets.figure_grid_view import live_tiles_from_panels
```

EVERY LIVE PANEL, not only the volcano.

"i would like you to generate all plots with the pyqtgraph and have each represented as a tab under results" and "i would still like to retain the grid to the right ... same grid overview but pyqtgraph versions". The tabs landed first; this is the grid half.

PHOTOGRAPHS, NOT LIVE WIDGETS, and that was measured rather than assumed: per window-drag frame at 18 tiles, live pyqtgraph widgets cost 74.99 ms against 5.19 ms for pictures, on a 16.7 ms budget -- six live tiles already miss the frame. The live widget is what a tile OPENS, not what the grid holds.

## AppScreen._open_live_tile

### lines 8815-8818

```python
if str(key) == PINNED_KEY:
```

The volcano keeps its own route. It is not in the results panel's tabs at all on this screen -- it is a PAGE of the figures stack, because the gene tile goes beside it -- so the tab lookup below would correctly find nothing for it.

### lines 8820-8822

```python
return
```

Already handled: the grid emits `pinned_activated` alongside this signal for the volcano, and that connection is what raises it. Acting here too would raise it twice.

### lines 8828-8830

```python
self._console.append_notice(
```

THE TILE SAID SO RATHER THAN GOING QUIET. A key with no tab in this panel is the one case that still ends in nothing visible happening, so it is the one case that has to be said out loud.

## AppScreen._live_tile_menu

### line 8846, trailing  _(unsure)_

```python
return
```

`pinned_menu_requested` has it.

### lines 8859-8860

```python
self._pin_regression_graph()
```

The menu may have restyled the graph and the tile is a photograph of it, so the photograph has to be retaken.

## AppScreen._pinned_menu

### lines 8879-8880

```python
self._pin_regression_graph()
```

The menu may have restyled or recoloured the graph, and the tile is a photograph of it, so the photograph has to be retaken.

## AppScreen._show_publication_sheet

### lines 8912-8914

```python
self._figure_queue.add_figure(sheet.figure)
```

Into the ordinary figure queue, so it restyles, exports and saves through exactly the same path as every other figure. A bespoke viewer for one figure is a second set of those bugs.

## AppScreen._measurements_destination

### lines 8988-8996

```python
folder = _os.path.dirname(str(database))
```

THE PLATE FOLDER, which is where the score and count

CSVs of that plate sit. spaCR writes the database as ``<plate>/measurements/measurements.db``, so that is two levels up -- but only when the parent IS ``measurements``. A loose database is one level up, and assuming the deep layout for it would put the merged frame in the plate's PARENT, which on a project root is everybody's folder. The same rule `AnnotateDropHandler` follows for the same reason.

## AppScreen.open_run_beside

### lines 9112-9113

```python
self._console.append_notice(
```

ALREADY THE LIVE ONE. Opening a run beside itself is two views of one run, which is not the comparison that was asked for.

### lines 9124-9127

```python
panel = RegressionResultsPanel(self._results_split)
```

ITS OWN VOLCANO, which is the whole request. The loaded run's plot is placed externally (`external_volcano=True`, in the figures stack); this one keeps its own, so the two are on screen at the same time and each answers its own hover and its own click.

## AppScreen._raise_the_results_tab

### lines 9191-9196

```python
LOG.debug("could not raise the results tab", exc_info=True)
```

A deleted page raises RuntimeError and one of the wrong type raises TypeError. Failing to raise a tab is a blemish; raising out of the slot that tries would lose whatever called it. Covered by tests/qt/test_cov_r8_app_screen_tails.py -- the pragma here was simply wrong, not merely unexplained.

## AppScreen._on_runs_removed

### lines 9213-9215

```python
self._run_photographs.pop(os.path.abspath(folder), None)
```

A DELETED RUN TAKES ITS STILL WITH IT TOO. A photograph of a run that no longer exists is the same stale answer its plot state would have been.

### lines 9228-9233

```python
queue = getattr(self, "_figure_queue", None)
```

AND ITS FIGURES (instruction 146's last open half). The queue sections its tiles by run label, and until `forget_run` existed there was no way to drop ONE section -- `clear()` is all-or-nothing, so removing a run would have taken every other run's figures with it. A grid still showing a deleted run's tiles is the same stale answer its plot state would have been.

## AppScreen._on_loaded_run_changed_refresh_tabs

### lines 9259-9265

```python
montage = getattr(self, "_cell_montage", None)
```

ONE TRY PER CALL, and that is the whole repair as much as the names are. These were one block: `montage.clear()` raised AttributeError, so `montage.refresh()` never ran either, and both halves of the Cells tab went stale on a single typo. Reported as issue 116 -- "show the cells still not able to pull images" after re-running regression -- with the AttributeError in the attached log, logged at DEBUG where nothing showed it to the user.

### lines 9279-9282

```python
scan.refresh_databases()
```

`refresh_databases`, not `refresh`: this panel has no method by that name, so the Measurements tab never re-attached the databases when the run changed -- which is what the Cells tab then reads to find its images.

## AppScreen._on_results_tab_changed

### lines 9304-9308

```python
montage = getattr(self, "_cell_montage", None)
```

THE CELLS TAB, for the same reason as the Measurements tab and one more: the databases it needs are attached to the input table while it is behind another tab, and so is the results table it reads the fitted effect from. Nothing signals either, so opening the tab is when it can learn what it is now able to do.

## AppScreen._scan_source_frame

### lines 9433-9438

```python
folder = self._results_source_path()
```

THE RUN'S FOLDER, asked of the run. `_path` was read directly here and passed through `dirname`, which is right for the CSV a load off disk leaves behind and WRONG for the directory a live run leaves it climbed to `results/` and looked for regression_data.csv beside the other runs, where there is none. Same fault as 155 A, one view over.

### lines 9443-9444  _(unsure)_

```python
for name in ("regression_data.csv", "merged_data.csv"):
```

`regression_data.csv` is what perform_regression writes after the merge: one row per well, the guides and the response together.

## AppScreen._show_trial

### lines 9480-9484

```python
named = record.get("run")
```

NAMED THE WAY THE ROW NAMES ITSELF. The tab now holds this session's runs beside the sweep's trials, and calling an ordinary run "Trial nan" is how a mixed table stops being readable. `isinstance` rather than truthiness: a missing cell in a concatenated frame is NaN, and NaN is truthy -- it would name the run "nan" without failing.

### line 9490

```python
self._console.append_stdout(
```

Not "did not produce a regression" -- it has not finished trying.

### lines 9508-9519

```python
if folder and self._same_run_folder(panel.run_folder(), folder):
```

ALREADY ON SCREEN: SHOW IT, DO NOT RE-READ IT. The run that has just finished arrives here twice -- `_on_pipeline_result` puts its table, its fitted model, its diagnostics and its statsmodels summary in the panel straight from the run, and a moment later the Runs tab announces the same run as loaded. Re-reading the folder would replace every one of those with what could be recovered off disk: `set_frame` clears the diagnostics by design, and the summary would fall back to the saved text. The model is the better answer and this is the only place that can keep it.

It is also what makes the two signals the Runs tab emits together cost one load rather than two.

### lines 9522-9525

```python
self._console.append_stdout(
```

AND THE TAB STILL DOES NOT MOVE (190). Re-opening the run that is already on screen is the one path that never reaches `_on_trial_loaded`, so it says so here instead: the results being ready is worth announcing, being carried to them is not.

### lines 9530-9536

```python
if folder and not os.path.isdir(str(folder)):
```

A FOLDER THAT IS NAMED BUT GONE IS THE SAME ANSWER AS NO FOLDER. The record keeps whatever path the trial wrote to, and that path outlives the directory -- a cleaned scratch disk, a run copied between machines, a results tree moved. Checked HERE, where the answer is one sentence, rather than left to the off-thread load to discover: the load reports it a second later, through a different path, after the run has already been marked loaded.

### lines 9543-9546

```python
self._the_run_did_not_open(
```

AND THE MARK GOES BACK WHERE IT WAS. A run marked loaded whose results are not on screen is the disagreement of instruction 157 pointing the other way: the run the user IS looking at would then be named nowhere.

### lines 9550-9555

```python
self._pending_trial = (trial, str(folder))
```

OFF THE GUI THREAD (instruction 159). Reading a run walks its folder and parses its table, and doing that here stopped the window reported as "i tried to load another run and this seemed to hang spacr". The answer arrives at `_on_trial_loaded`, so everything that depends on SUCCESS moves there: the mark can only be rolled back once the read has actually failed, which is later than this line.

### lines 9561-9563

```python
return
```

A load is already running. The mark stays where the running load will put it; starting a second read of a different folder is how two answers arrive out of order.

## AppScreen._on_trial_loaded

### lines 9566-9569

```python
def _on_trial_loaded(self, ok: bool) -> None:
```

NOT `_raise_the_results_tab` (190). The read has only just been STARTED here, and raising the tab would move the user off whatever they were reading to watch an empty panel fill in. `_on_trial_loaded` says the run opened once it actually has.

### lines 9590-9592

```python
runs = getattr(self, "_sweep_runs", None)
```

THE LOAD REPORTED SUCCESS, so the undo it was holding is spent. A refusal arriving after this is answering an announcement that is over (instruction 159, and 157's rule about the mark).

### lines 9596-9597  _(unsure)_

```python
if not self._load_trial_figures(str(folder)):
```

Its figures too, so the grid on the right is that trial's and not whatever the last run left there.

### lines 9603-9611

```python
self._console.append_stdout(
```

THE TAB DOES NOT MOVE ON ITS OWN (190). Reported 2026-08-20: "the user should have to click the results tab to go there, no auto switching tabs." A view that moves by itself takes the user somewhere they did not ask to go and loses whatever they were reading. The results arriving is fine; being MOVED to them is not.

SO IT HAS TO SAY SO INSTEAD. Nothing raises the tab now, and a load that finished silently while the user is on another tab would look like a load that did not happen.

## AppScreen._pictures_from

### lines 9738-9741

```python
titles.append(os.path.splitext(name)[0].replace(os.sep, " / "))
```

The SUBFOLDER is part of the name now, because "residuals" under regression_qc/ and "residuals" under results/ are two different pictures and a grid captioning both the same is a grid you cannot navigate.

## AppScreen._load_trial_figures

### lines 9765-9769

```python
already = {os.path.basename(name) for name in run_names}
```

One section for both screen folders, not one each: a reader is being told "these are not this run's", and which of the two directories above the run a shared figure happens to sit in is not a distinction they can act on. `already` grows as it goes, so a name present in both folders is shown once, nearest the run.

## AppScreen._figure_grid_menu

### lines 9805-9807

```python
self._refresh_figure_grid()
```

The restyle rewrote that figure's picture; the grid is built from pictures, so it has to be rebuilt or the tile keeps showing the old one and the menu looks broken.

## AppScreen._on_guide_selected

### lines 9897-9899

```python
if not getattr(self, "_gene_opened", False) and split.sizes()[1] == 0:
```

Only the FIRST click opens it. Reasserting a size on every click would fight anyone who had dragged the tile bigger to read it, or shut to see the whole plot.

## AppScreen._on_pipeline_result

### lines 9953-9956

```python
self._update_run_in_runs_tab(
```

WHERE THIS RUN WROTE, ON THE RUN'S OWN ROW. Recorded before the results panel is even consulted, because it is what makes the Runs tab navigable: `_show_trial` opens a row by its folder, and a row with no folder is a row that can only be looked at.

### lines 9961-9966

```python
self._say_the_qc_verdict(payload)
```

THE QC VERDICT, ON SCREEN (instruction 115). The suite computes it, the manifest carries it and the report writes it to a text file that nobody opens. A run whose design is rank deficient has coefficients that are ONE of infinitely many solutions, and a screen that shows the volcano without saying so is showing a picture of an arbitrary answer.

### lines 9975-9977

```python
if folder:
```

WHERE THIS SCREEN'S LAST RUN WROTE. 142 C: a Force restart names the folders that hold whatever reached disk, so a user knows where to look rather than assuming everything is gone or everything is fine.

### lines 9982-9987

```python
try:
```

THE RUN'S OWN SETTINGS, handed over after the frame so they win over whatever `set_frame` read off disk. The shared settings/ copy is overwritten by every later run of the same screen, so on a second run the file describes the wrong one -- and a re-fit seeded from it would offer a model this table was never fitted with.

### lines 9993-9999

```python
try:
```

THE FITTED MODEL, WHICH ONLY THIS PATH HAS. The residual, scale-location and influence tabs are computed from the fit itself, and `perform_regression` hands it back here and nowhere else -- a results CSV read off disk is one row per guide and says nothing about the wells. Handed over AFTER the frame, because `set_frame` clears the diagnostics on the principle that a new table is a new fit.

### lines 10004-10005

```python
panel.set_summary(
```

The same model, the same moment: the statsmodels summary the maintainer asked for on 2026-08-17.

## AppScreen._load_regression_results

### lines 10047-10050

```python
from ..widgets.regression_results import find_results_tables
```

THE NEWEST RUN ACROSS ALL THE ROOTS, not the first root that happens to contain any results at all. `src` and the count-data folder are different places and both can hold a table, so "the first one that loads" can be last month's.

### line 10059, trailing  _(unsure)_

```python
except OSError:
```

vanished between listing and stat

### lines 10065-10068

```python
self._show_figure_grid()
```

The results are always on screen now -- they are the left half, not a tab -- so there is nothing to switch to. Show the grid rather than whichever single figure was last open, because a finished run is read as a whole.

### lines 10073-10074

```python
if ranked:
```

NOTHING LOADED, AND THAT USED TO BE SILENT: the panel sat there with its columns and no rows, which reads as a run that produced nothing.

### line 10076, trailing  _(unsure)_

```python
pass
```

panel.load already said why, on screen

### line 10078, trailing  _(unsure)_

```python
panel.load(candidates[0])
```

leaves its own reason on screen

## AppScreen._clear_thread_refs

### lines 10098-10099  _(unsure)_

```python
QTimer.singleShot(0, self._rebuild_the_form)
```

Leave the QThread.finished delivery before replacing this screen. The next event-loop turn is both safe and imperceptible.

## AppScreen._on_stop

### lines 10133-10135

```python
offer_restart=True,
```

142: the last resort, and offered from HERE rather than from the Quit dialog because this is the button somebody presses when a fit will not stop.

### lines 10153-10155

```python
self._request_cooperative_stop()
```

NOT disabled. A cooperative stop that never lands used to leave the user with no way to escalate; the button stays live so pressing it again reaches the same prompt, and the watcher asks unprompted.

## AppScreen._force_stop

### lines 10223-10225

```python
self._console.append_notice(
```

Parked: the window is usable now, and the run is still out there. Say so -- a user who is told "stopped" and then sees the file grow has been lied to.

## AppScreen._on_import_settings

### lines 10239-10242

```python
path, _ = QFileDialog.getOpenFileName(
```

A file dialog is built and executed in one expression, so the application-wide dialog pass in `spacr.qt.i18n` never sees it before it is on screen: its caption and filter are translated here instead.

## AppScreen._load_settings_csv

### lines 10294-10295

```python
if first_error is None:
```

Wrong header spelling (or an unparseable file) — remember the first complaint and try the next spelling.

## AppScreen.apply_settings_dict

### lines 10616-10618

```python
self._deferred_form_values = target
```

The run keeps its screen. Values that have no widget on the old shape wait here and are carried into the one replacement made after QThread.finished.

### lines 10632-10634

```python
if model is not None:
```

ONE WIDGET AT A TIME MEANS A HALF-APPLIED PANEL in between, and a rule that reads other settings must not act on it. See `_show_the_value_it_will_have`.

### added 2026-09-19 (364)

```python
fresh._built_for_this_bulk_apply = True
```

ONE REBUILD PER IMPORT, whatever the shape check says. The replacement screen is built from the merged values (`rebuild_app_screen(self.app_key, target)`), so rebuilding it again for the same file would build the same screen. Before this guard the recursion ended only when the fresh screen agreed with the file, and for a switch the form does not carry it never could: an old recruitment CSV with `nucleus_mask_dim` or `pathogen_mask_dim`, a Mask file on Measure, or a Measure file on Mask rebuilt the screen without end. Measured through a real `MainWindow` with Import settings and the rebuild capped at six: all four cases hit the cap in 1.5-4.9 s and ended in "Import failed"; the review of this unit let one run uncapped and killed it at 400 s. The flag is cleared in `finally`, so a later import on the same screen can still rebuild it. See the note on `_bulk_apply_changes_form_shape` for the cause, and `tests/qt/test_an_import_rebuilds_the_screen_at_most_once.py`.

## AppScreen._refresh_after_bulk_apply

### lines 10656-10658

```python
try:
```

And the two switches that decide which DIMENSIONS the form is about, for the same reason: a file that asks for a volumetric run must not land on a form that is hiding the volumetric settings.

## AppScreen._sync_folded_switches

### lines 10739-10741

```python
self._folds_last_switched_on = switched
```

Remembered so `_warn_about_moved_settings` can report what moved instead of guessing, and so a screen with no switches can say the flag was ignored rather than claiming it landed.

## AppScreen._migrate_control_wells

### line 10769, trailing  _(unsure)_

```python
return settings
```

a screen that holds the trio itself

### line 10771, trailing  _(unsure)_

```python
return settings
```

the file already says it the new way

## AppScreen._apply_each_setting

### lines 10791-10793

```python
from .settings_model import _APP_HIDDEN_KEYS
```

Preserve every known off-form value, but keep the historical return contract: only dedicated hidden controls count as exposed/applied rows.

## AppScreen._apply_value

### lines 10827-10831

```python
from .settings_model import AUTO_TEXT, _set_auto_or_number
```

A BOX THAT ALSO SAYS "auto" (181). `float("auto")` raises, and the `except` above would have swallowed it and left the control showing 1 -- the one value that cannot fit a penalised model. Two writers for one widget is why this needed saying twice; the spelling is shared so they cannot answer differently.

### lines 10842-10848

```python
index = widget.findData(val)
```

THE STORED VALUE FIRST, THEN THE TEXT. A combo built from

(value, label) pairs shows the SENTENCE and stores the KEY 'load images' for 'png' (171), 'guide permutation — test each guide on its own' for 'guide_permutation' (134) -- so matching on `itemText` alone silently ignored every settings CSV that named the key, which is every settings CSV there is. The selection simply did not move, and nothing said so.

### lines 10857-10861

```python
widget.set_value(val)
```

_ListEditor / _ListEdit / _ScalarEdit all round-trip their own value. Importing a settings CSV used to go through the plain QLineEdit branch below, which str()'d a list back into the box; the chip editor is not a QLineEdit at all, so it would have been skipped entirely.

## AppScreen._refresh_usage

### lines 10901-10902  _(unsure)_

```python
per_core = bool(self._btn_cpu_toggle.isChecked()
```

Read the toggle here: it is a widget, and the worker may not look at one.

## AppScreen._apply_usage

### lines 10915-10918

```python
if (request_generation is not None
```

A page can be hidden while the worker is sampling. hideEvent bumps the generation, invalidating only that in-flight result. Explicit refreshes remain useful on a hidden test/diagnostic screen and the next showEvent requests a fresh generation immediately.

## _sample_usage

### lines 10980-10985

```python
if _nvidia_smi_available():
```

GPUtil shells out to nvidia-smi.  Calling it when the executable does not exist is both pointless and, after hundreds of short-lived worker threads in a Qt process, has crashed in CPython's subprocess boundary (CI run 31869225004).  The cheap executable check keeps CPU-only hosts entirely outside that native boundary.  A real NVIDIA host still uses GPUtil's established parsing and reports the same values as before.

## AppScreen._refilter_the_settings_search

### added 2026-09-19 (431)

```python
self._settings_model.rows_are_filtered_by = \
```

The object rule and the settings search both decide rows, and the search has to have the last word: it is the narrower of the two. See the note on `SettingsWidgets.refresh_object_visibility`. Re-entry is refused because applying the search can lay out a waiting row, and laying one out runs the object rule, which would call this again.

## AppScreen._watch_the_settings_that_decide_the_form

### added 2026-09-19 (431)

```python
for key in object_switch_keys("cell"):
```

`cell_channel` is not a form-shaping key -- cell rows are never hidden -- so nothing watched it, and typing a cell channel under Essentials changed nothing on the form. It is watched now with the same in-place pass as the other object channels, and kept out of `_object_switches_on_this_form`, which `tests/qt/test_a_channel_number_reveals_rather_than_reloads.py` holds to be a subset of the shaping keys.

## AppScreen._wire_live_preview_naming

### added 2026-09-19 (431)

```python
timer.timeout.connect(panel.regroup_the_folder)
```

For GitHub issue #119: see the note on `LivePreviewPanel.regroup_the_folder`. Wired to `textChanged` behind a 400 ms single shot rather than to `editingFinished`, so a settings file applied programmatically regroups too, and a pattern typed a character at a time is read once.

## AppScreen._bulk_apply_changes_form_shape

### added 2026-09-19 (364)

```python
if key not in carried:
```

A switch the form neither shows nor holds cannot shape it. `_form_shaping_keys` already followed that rule for a typed edit ("A missing switch cannot shape this panel"); the bulk path looked at the file instead and took every `nucleus_*` / `pathogen_*` channel or mask-plane key in it as a switch. Instruction 364 took the three `*_mask_dim` keys off the Recruitment form on 2026-09-03 (79edbf12f), and every recruitment file saved before then carries `nucleus_mask_dim=5` and `pathogen_mask_dim=6`, so importing one compared 5 with a `current.get()` of None, asked for a rebuild, and asked again on the rebuilt screen. `cell_mask_dim` escaped only because the cell object is never a switch. The same held for any file from a module whose switches are spelt the other way: Measure has no `*_channel`, Mask has no `*_mask_dim`.

## AppScreen._file_the_report_automatically

### added 2026-09-19 (autofile-default, follow-up to 432)

```python
if fingerprint in _REPORTS_BEING_FILED:
```

Filing happens in `_on_finished`, through `_settle_the_report`, not in `_on_pipeline_error`. The console then reads "✗ Failed", then "[issue] Filing ...", then where it went, which is where 432 put the "Nothing was sent" line for 'ask'. The AI has usually not answered by then. Its analysis is attached only if it already has, and `ai_explanation_of` returns nothing for a provider that failed (432). Filing does not wait for the AI.

`_REPORTS_BEING_FILED` is module-level because two screens can fail on one crash before the first report has come back from GitHub. The ledger in `ai.settings` is written only when GitHub answers.

The report is built on the background runner, not on the GUI thread. `build_report` copies the log tail to a file, and `file_without_review` can run `gh auth token`. Only the form snapshot and the AI analysis are read on the GUI thread, because they are widget state.

## AppScreen._the_terms_allow_automatic_filing

### added 2026-09-19 (autofile-default)

```python
return not needs_agreement()
```

The maintainer tied automatic filing to the agreement: "add the user agreeing to this in the user agreement, if set to always". 'always' is the default, and the default applies to profiles that have never seen the agreement. A launch with `--no-setup` or `SPACR_NO_SETUP`, or under the offscreen platform (`setup_screen.skipped_on_purpose`), never shows the slides. A user can close the terms slide without accepting ("spaCR will present the terms again at the next startup"). A profile that accepted 4.1 accepted terms saying nothing is sent automatically. None of these has agreed, so none files automatically: the console says nothing was sent, and that filing starts once the terms are accepted. "File as issue" still works for them, through the preview.
