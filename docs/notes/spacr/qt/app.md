# Notes from `spacr/qt/app.py`

Prose lifted out of `spacr/qt/app.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (50 entries)
- [_carry_preview_state](#_carry_preview_state) (1 entry)
- [_open_at_the_measured_width](#_open_at_the_measured_width) (2 entries)
- [install_the_spaceout_fractal](#install_the_spaceout_fractal) (1 entry)
- [_UpdateWorker.run](#_updateworkerrun) (1 entry)
- [_DragsTheWindowByTheMenuBar.eventFilter](#_dragsthewindowbythemenubareventfilter) (1 entry)
- [_PipelinePreloader.start](#_pipelinepreloaderstart) (1 entry)
- [register_app](#register_app) (1 entry)
- [unregister_app](#unregister_app) (1 entry)
- [_call_screen_factory](#_call_screen_factory) (1 entry)
- [dock_rows](#dock_rows) (1 entry)
- [_declared_folds](#_declared_folds) (1 entry)
- [_declared_folds._as_string](#_declared_folds_as_string) (1 entry)
- [make_home_page](#make_home_page) (1 entry)
- [_icon_for_app](#_icon_for_app) (1 entry)
- [MainWindow.__init__](#mainwindow__init__) (23 entries)
- [MainWindow.__init__._finish_installer_onboarding](#mainwindow__init___finish_installer_onboarding) (1 entry)
- [MainWindow._install_loading_screen](#mainwindow_install_loading_screen) (1 entry)
- [MainWindow._on_preload_step](#mainwindow_on_preload_step) (2 entries)
- [MainWindow._install_fullscreen_button](#mainwindow_install_fullscreen_button) (7 entries)
- [MainWindow._close_icon](#mainwindow_close_icon) (1 entry)
- [MainWindow.eventFilter](#mainwindoweventfilter) (1 entry)
- [MainWindow.changeEvent](#mainwindowchangeevent) (1 entry)
- [MainWindow._build_menu_bar](#mainwindow_build_menu_bar) (21 entries)
- [MainWindow._build_window_menu](#mainwindow_build_window_menu) (2 entries)
- [MainWindow._menu_bar_actions](#mainwindow_menu_bar_actions) (2 entries)
- [MainWindow](#mainwindow) (2 entries)
- [MainWindow._run_e2e_chain](#mainwindow_run_e2e_chain) (2 entries)
- [MainWindow._apply_demo_to_screen](#mainwindow_apply_demo_to_screen) (3 entries)
- [MainWindow._show_about](#mainwindow_show_about) (2 entries)
- [MainWindow.refresh_theme](#mainwindowrefresh_theme) (2 entries)
- [MainWindow._refresh_demo_status_tips](#mainwindow_refresh_demo_status_tips) (1 entry)
- [MainWindow._rebuild_startup_page](#mainwindow_rebuild_startup_page) (1 entry)
- [MainWindow._on_upgrade_done](#mainwindow_on_upgrade_done) (2 entries)
- [MainWindow.closeEvent](#mainwindowcloseevent) (5 entries)
- [MainWindow._show_module_hint](#mainwindow_show_module_hint) (1 entry)
- [MainWindow._set_backdrop_blank](#mainwindow_set_backdrop_blank) (1 entry)
- [MainWindow.resume_after_restart](#mainwindowresume_after_restart) (1 entry)
- [MainWindow._install_startup_page](#mainwindow_install_startup_page) (1 entry)
- [MainWindow._show_the_screensaver](#mainwindow_show_the_screensaver) (1 entry)
- [MainWindow.rebuild_app_screen](#mainwindowrebuild_app_screen) (7 entries)
- [MainWindow._on_nav_selected](#mainwindow_on_nav_selected) (11 entries)
- [MainWindow._theme_screen](#mainwindow_theme_screen) (1 entry)
- [MainWindow._install_screen_backdrop](#mainwindow_install_screen_backdrop) (3 entries)
- [MainWindow._drop_a_redundant_screen_backdrop](#mainwindow_drop_a_redundant_screen_backdrop) (5 entries)
- [MainWindow._a_page_joined_the_stack](#mainwindow_a_page_joined_the_stack) (1 entry)
- [MainWindow.stylesheet_roots](#mainwindowstylesheet_roots) (2 entries)
- [MainWindow._backdrop_the_dock_column](#mainwindow_backdrop_the_dock_column) (2 entries)
- [MainWindow._show_preparing](#mainwindow_show_preparing) (1 entry)
- [MainWindow._build_screen_timed](#mainwindow_build_screen_timed) (1 entry)
- [MainWindow._snapshot_current_screen_settings](#mainwindow_snapshot_current_screen_settings) (2 entries)
- [MainWindow._on_train_requested](#mainwindow_on_train_requested) (2 entries)
- [_use_open_sans](#_use_open_sans) (2 entries)
- [_install_crash_dump](#_install_crash_dump) (1 entry)
- [launch](#launch) (34 entries)
- [launch._prewarm](#launch_prewarm) (2 entries)
- [launch._drain_ai](#launch_drain_ai) (3 entries)

## Module level

### lines 42-45

```python
from .app_catalog import (LazyScreenFactory, declared_for as _declared_for,
```

The declared registry rows and the stand-in that defers a screen's import until it is built. Cheap on purpose: `app_catalog` imports nothing beyond `importlib` and `inspect`, and reads `register_app` back out of this module from inside a function, so naming it here is not a cycle.

### lines 48-52

```python
from .widgets.dock import Dock
```

Nothing in this module uses a colour, a spacing or the palette API any more — the Home page, the sidebar QSS and `apply_preferences_to_app` each own their own. The import stayed behind after they moved out, and because it named `PALETTE` it fired the deprecation warning on every `import spacr.qt.app` for a value nobody read.

### lines 473-518

```python
SECTION_CORE = "Core"
```

The app registry

Two axes, and only ONE of them is a place.

An app is filed under WHAT IT DOES — Core, Data, Segmentation models, Results & QC, Toxoplasma. That is stable: Format Converter is a Data app whether or not anyone has finished it, and it is the axis the sidebar, the command palette and the Home tabs all group by.

An app is *also* staged by HOW FINISHED IT IS — alpha, beta or stable — and 22 of the 30 are not yet signed off. #16i made that second axis two extra CATEGORIES, which meant three of the five subject tabs drained to nothing and "where is the format converter" acquired two answers. #16j undoes exactly that: maturity is now drawn as the tile's HOVER COLOUR, with a legend beside the tiles, so it is visible on the same tile that is in the right category. One table (:data:`APP_STAGE`), one place per app, no second grouping.

Core            the end-to-end pipeline: images in, single-object measurements out, hits called. Data            get images and tables in, run many plates, get the numbers back out. Segmentation    build, train, pick and check the Cellpose models models        the Mask step runs. Results & QC    read what came out and decide whether to believe it. Explore         ask the numbers a question you did not plan for. Toxoplasma      parasite-specific readouts. Design          plan the next experiment before it runs.

A section holds AT MOST `MAX_APPS_PER_SECTION` apps. Past that nobody reads the row — which is exactly how "Tools" grew to sixteen entries and became unusable. If a section is full, the honest fix is a new section with a name that means something, not a longer row.

Explore and Design are the two that fix accordingly: the modules queued behind this file (Graph Builder, pivot/formula, gate editor, feature explorer, layer viewer; power/design, experiment designer) would have taken Results & QC from eight to fifteen. They are DECLARED and EMPTY — no tab, no heading, nothing drawn — until their first app registers, because a tab that opens on an empty pane is worse than no tab.

The names are as short as they can be and still mean something: they are TAB LABELS on Home, where long names would not fit on one line, and a tab that has to elide is a tab nobody can read.

### lines 629-632

```python
"results": SECTION_DATA,
```

Retired sections, aimed at where their built-ins went: report and the control chart moved from Results & QC into Data, the power calculator from Design into Data, and the feature dictionary from Explore into Tools.

### lines 850-853

```python
TILELESS_APPS = frozenset({
```

MOVED UP 2026-09-02, and the position is load-bearing:

`_refresh_sections` reads this to keep a tile-less section out of Home's tab bar, and it runs during `register_app` at import time which is before this constant's old position further down the file.

### line 871  _(unsure)_

```python
"feature_dict",
```

Reached from Help -- a user looking something up.

### line 876  _(unsure)_

```python
"investigate_hit",     # Regression, new tab
```

Reached from a button in the module they belong to.

### line 877, trailing

```python
"investigate_hit",
```

Regression, new tab

### line 878, trailing

```python
"profiler",
```

Regression, new tab

### line 879, trailing  _(unsure)_

```python
"train_compare",
```

Classify

### line 880, trailing  _(unsure)_

```python
"feature_explorer",
```

Classify

### line 881, trailing  _(unsure)_

```python
"plate_view",
```

Graph Builder

### line 882, trailing  _(unsure)_

```python
"trellis",
```

Graph Builder -- "small multiples"

### line 883, trailing  _(unsure)_

```python
"lineage",
```

Database Browser

### line 884, trailing  _(unsure)_

```python
"tabulate",
```

Database Browser

### line 885, trailing  _(unsure)_

```python
"layer_viewer",
```

QC

### line 886, trailing  _(unsure)_

```python
"control_chart",
```

QC

### line 887, trailing  _(unsure)_

```python
"outliers",
```

QC

### line 888  _(unsure)_

```python
"convert",             # Import -- Format Converter
```

Folded into Import: one module for getting data in, three ways.

### line 889, trailing  _(unsure)_

```python
"convert",
```

Import -- Format Converter

### line 890, trailing  _(unsure)_

```python
"external_masks",
```

Import -- External Masks

### lines 891-893

```python
"report",
```

Help menu entries rather than tiles. None of the six is a place to START: each one inspects or administers work that already exists, which is what a menu is for and what a tile is not.

### lines 1147-1165

```python
("mask",           "Mask",           "Generate segmentation masks for cells, nuclei, pathogens an...
```

(key, human name, description, section)

`section` is what the app IS ABOUT. How finished it is lives in APP_STAGE below and is drawn as a colour, not as a place.

NOTE: keys are load-bearing. bridge.resolve_pipeline_entry, cli.INTERACTIVE_ONLY, validate.APP_FUNCTIONS, dnd_handlers, settings_model.resolve_default_settings and saved user state all key off them. Renaming a key silently breaks those; renaming the display name or moving an app between sections is free.

Core pipeline: images in, single-object measurements out, hits called. Ctrl+1..6 map to these six before Ctrl+7..9 continue into the next apps in sidebar order. CORE IS THE PIPELINE, IN THE ORDER YOU RUN IT: mask, measure, annotate, classify, map barcodes, regression. Nothing else belongs in it -- Timelapse and the Motility assay are assays and are filed as such, and a section that lists everything is a section that sorts nothing.

### lines 1169-1178

```python
("classify_merged", "Classify",      "Train classifiers on image crops with PyTorch or on measure...
```

ONE CLASSIFY SCREEN. "Classify (CV)" and "Classify (ML)" were the two originals kept beside the merged one so a saved settings CSV would keep working -- but three entries for one job is three places to look and two of them are the same run with half the choices. Removed on 2026-08-23 at the maintainer's instruction.

THE ENTRY POINTS ARE UNTOUCHED: `deep_spacr` and `generate_ml_scores` are what `classify.classify` dispatches to, so a notebook importing either still works and a settings CSV for either still runs -- through the one screen, which reads `classifier_family`.

### lines 1189-1197

```python
("run_history",    "Run History",    "Search run settings, outputs, warnings, failures and perfor...
```

CLASSIFIER EVALUATION, EXPLAIN CV MODEL AND ACTIVATION ARE BUTTONS ON CLASSIFY. A classifier is trained on one screen and argued about on two others, so both fold onto the Classify masthead (`spacr.qt.screens.classify`) and open their own screen as a page beside the training settings. Neither has a row here any more; what each tile said is `map_barcodes.FOLD_FALLBACK`, and every table a row used to feed -- the drop handler, the API link, the header, the translated name -- names them directly instead.

### lines 1200-1207

```python
("train_compare",  "Training Runs",  "Compare training curves and settings across multiple runs",...
```

Data & batch runs: get images and tables into a spaCR project, run many plates unattended, get the numbers back out. RESULTS, NOT CORE. Core is the pipeline and its order IS the pipeline it is the category the dock opens by default and the first thing a new user reads. Training Runs compares finished runs, which is a result rather than a step, and it is ALSO a folded child of Classify, so it was on Home twice. Moved on the maintainer's instruction, 2026-09-08, with Prediction Profiler and Investigate Hit.

### lines 1210-1218

```python
("make_masks",     "Make Masks",     "Edit segmentation masks with brush, flood-fill, relabel, fi...
```

Segmentation models: build, train, pick and check the Cellpose models the Mask step runs. Not a training screen despite where it sits: MakeMasksScreen is the brush, the flood fill and the object operations, i.e. correcting a mask by hand. It carried Train Cellpose's description verbatim, which is the app directly below it. DATA, NOT MODELS. Make Masks does not train, choose or run a segmentation model: it is hand curation of masks that already exist, which is the same kind of work as the other tools filed under Data.

### lines 1220-1234

```python
("plate_view",     "Plate Viewer",   "Visualize measurements as plate heatmaps and detect edge ef...
```

THE SEGMENTATION WORKBENCH HAS NO SATELLITE TILES. Training a model, comparing two of them, browsing the zoo and curating a mask by hand are all one loop -- segment, look, correct, train, segment again -- and they were four rows the user had to leave the loop to reach. They are buttons on the Make Masks masthead now (`make_masks.FOLD_ORDER`), each opening the module's own screen as a page beside the editor.

THE KEYS ARE STILL REAL, which is the whole difficulty of dropping the rows: `spacr-run train_cellpose` runs, a settings file written for it still loads, a file dropped on the page still lands, and `spacr-run model_compare / model_zoo / curate` still say what to do instead `cli.INTERACTIVE_ONLY` holds those three sentences in its own literal now that no row carries a `cli_note=`. What went is the tile. Results & QC: look at what came out, decide whether to believe it, and hand it to someone else.

### lines 1236-1240

```python
("umap",           "Image UMAP",     "Visualize UMAP embeddings with image glyphs",              ...
```

ANNOTATOR AGREEMENT HAS NO ROW. Scoring how well two annotation passes agree is the sentence after annotating them, so it is a button on the Annotate masthead that opens its own screen, whole (`spacr.qt.screens.annotate`). `cli.INTERACTIVE_ONLY` still names it, so `spacr-run agreement` still says where to find it.

### lines 1242-1253

```python
("analyze_plaques", "Plaque Assay",  "Quantify plaque assay measurements",                       ...
```

Toxoplasma assays: parasite-specific readouts.

TIMELAPSE AND MOTILITY HAVE NO ROW. Timelapse is the mask pipeline with tracking on, so it is a switch on the Mask Generation masthead that reveals its own settings categories (`spacr.qt.screens.mask`); the Motility Assay reads finished masks and writes a measurements table, so it is a button on the Measure masthead that opens its own screen (`spacr.qt.screens.measure`). Both still run from `spacr-run`, from a settings CSV and from a chained hand-off `spacr.cli.MODULES`, `validate.APP_FUNCTIONS` and `bridge.resolve_pipeline_entry` all still know them. What went is the tile, not the module.

### lines 1261-1263

```python
STAGE_STABLE = "stable"
```

Maturity — the second axis, drawn as colour rather than as a place

### lines 1284-1288

```python
"classify_merged": STAGE_ALPHA,
```

alpha: built and reachable, not yet trusted end to end (16)

The merged Classify module is new. "stable" is the ABSENCE of a line here, so leaving it out would have claimed a maturity it has not earned -- it dispatches to two pipelines that ARE trusted, but the merged screen itself has not been run on real data yet.

### line 1303  _(unsure)_

```python
"make_masks":      STAGE_BETA,
```

beta: further along, in regular use, still not signed off

### lines 1310-1313

```python
for _row in _BUILTIN_APPS:
```

The built-ins go through the same door as everything else. Registering 34 rows one at a time on every import is what keeps `register_app` honest: an ordering or validation mistake in it shows up here, at import, rather than the first time somebody adds the 35th app.

### lines 1319-1356

```python
_SELF_REGISTERING_APPS = (
```

Apps that lived in their own module

Nothing registers here any more, and the section is kept for the note. Two pipeline modules were registered from this file rather than from themselves, because neither is a Qt module: `spacr.illumination` and `spacr.sequencing_qc` are imported into worker processes and into `spacr-run`, and neither may grow an import of PySide6 to call `register_app` at its own import. Both have since folded into the screen that runs them, and a folded module has no row.

ILLUMINATION IS A BUTTON ON MEASURE. Flat-field correction is a property of the measure run it changes rather than a run of its own: `measure_crop` calls `spacr.illumination.prepare_illumination_correction` itself, and the nine `illumination_*` keys are a settings category on Measure's own panel. The one thing that panel cannot express is estimating and QCing the field WITHOUT measuring the plate -- an hour of QC figures before a day of measuring -- so the module keeps its own settings form and Run button and opens as a page beside the measure settings (`spacr.qt.screens.measure`). `spacr-run illumination` never went through this row and is untouched.

BARCODE QC IS A BUTTON ON MAP BARCODES. A mapping run is judged by reads per well, starved wells, unmapped reads, collisions, position effects and the abundance threshold they imply, so the question "did this run work" belongs on the screen that produced the run. It folds onto the Map Barcodes masthead and opens as a page beside the mapping settings; `spacr-run barcode_qc` and the automatic call from the end of the sequencing pipeline never went through this row and are untouched.

EVERYTHING `register_app` FANS OUT DIES WITH THE ROW, so each answer has a home that outlives a tile: the entry point in `spacr.qt.bridge.resolve_pipeline_entry`, the defaults module in `settings_model._FOLDED_DEFAULTS_MODULES`, the API link in `settings_model._APP_API_MODULE`, the screen title and intro in `app_screen.APP_TITLES` and `APP_INTROS`, and the name, sentence and maturity colour the fold button carries in `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`.

### lines 1378-1386

```python
("spacr.qt.screens.data_manager", "register"),
```

THE SIX THAT USED TO ARRIVE BY ACCIDENT. Each of these registers at its own import, and each was reached only because some other screen in this table happened to import it -- Data Manager because Run Compare reads projects, Lineage because the Layer Viewer registers its companions. So the row existed exactly when the import chain that produced it did, and the moment a screen stopped being imported at launch its tile vanished with it. Named here, they are registered because somebody asked for them; the order is the order they used to arrive in, so the tiles keep the positions users know.

### lines 1395-1400

```python
("spacr.qt.screens.power", "register"),
```

The three that arrived just after the seam landed and sat finished, tested and unreachable for the same reason the first four did.

Power is the first app of the Design section, so this row is also what makes that tab appear — the section has been declared, noted and empty since the sections were named.

### lines 1402-1406

```python
("spacr.qt.screens.run_compare", "register"),
```

Run Compare registers at its own import and is named in

``spacr.qt.SELF_REGISTERING_MODULES`` too, which only runs at ``run()``. That made the row appear at launch and not under ``import spacr.qt.app``, i.e. exactly the sometimes-there row the note above is about. Both calls are idempotent.

### lines 1408-1418

```python
("spacr.qt.screens.tabulate", "register"),
```

Tabulate: finished, tested, and defining a register() that nothing called. It was held back when Explore was at the MAX_APPS_PER_SECTION ceiling of 13; it is at 8 now, so the reason has expired. Found by the README pass, which declined to advertise a screen with no tile -- which is the right instinct and also how a feature stays invisible for a fortnight.

PCA stood beside it here until it was folded onto Image UMAP: it is a button on that masthead now, opened already pointed at the same measurements database, so there is no row for this table to put in the registry.

### lines 1421-1448

```python
("spacr.qt.screens.dose_response", "register"),
```

THE THREE THE MAINTAINER COUNTED ON SCREEN AND THIS TABLE DID NOT HAVE. Asked for 2026-09-05: Home is core:6 data:6 tools:5 assays:4 and the dock's Help heading is 9. Measured from `import spacr.qt.app` alone it was 6/5/4/4 and Help 8 -- Data short `dose_response`, Tools short `gate_editor`, Help short `project_browser`.

NOT A MATURITY PROBLEM, which was the first guess and is worth writing down so it is not guessed a second time. All three declare stage='alpha', `DEFAULT_SHOW_ALPHA` is True, and `app_is_visible` already answers yes for every one of them. They were simply NOT IN THE REGISTRY: each declares its row in `app_catalog` and was named only in `spacr.qt.SELF_REGISTERING_MODULES`, which `run()` walks -- so the three rows existed in a launched GUI and nowhere else. That is exactly the sometimes-there row the note at the top of this table is about, and a line here is the fix that note prescribes.

No section and no stage moved. `SECTION_TILE_ORDER` has named `dose_response` in Data and `gate_editor` in Tools since 2026-08-31 and `_HELP_MODULES` has named `project_browser` since it was written: the filing was right all along, the registration was late.

THE OTHER FIVE LAUNCH-ONLY ROWS STAY WHERE THEY ARE. `control_chart`, `outliers`, `trellis`, `feature_explorer` and `feature_dict` are all in `TILELESS_APPS`, so not one of them changes a count on either screen and `feature_dict` is reachable from neither a tile nor `_HELP_MODULES`, so registering it here would fail `test_no_module_falls_out_of_the_dock_altogether` over a door this change was not asked to find.

### lines 1456-1461

```python
if _declared_for(_module_name) is not None:
```

THE ROW WITHOUT THE SCREEN. Every module named above declares its row in `app_catalog`, so the registry can be filled in from strings and the screen's own code — pandas, scipy, sklearn behind it — is left unimported until somebody opens the app. `register_declared` returns None for a module that declares nothing, and that module is imported the old way.

### lines 1467-1469

```python
LOG.exception("Could not register the app owned by %s", _module_name)
```

One screen's import-time bug costs that screen and nothing else. The same posture this file already takes towards plugins, for the same reason: the window still opens.

### lines 1474-1475

```python
try:
```

Plugin apps use the same registry rows and maturity annotations as built-ins. Contributions can add a key but never replace one.

### lines 1480-1483

```python
if _plugin_app.key in {row[0] for row in APPS}:
```

Recomputed per plugin, not snapshotted before the loop: the old snapshot held built-in keys only, so two plugins claiming the same key both landed in APPS and the duplicate only showed up as two identical sidebar rows.

### lines 1500-1503

```python
record_diagnostic(
```

Everything `spacr.plugins` already validates — section, stage, non-empty name — plus anything it starts allowing that this registry does not. One bad contribution is dropped; the rest of the plugins still load.

### lines 1857-1862

```python
SECTION_DATA: ("foreign", "embeddings", "run_compare",
```

`embeddings` sits beside `foreign` because both ANSWER THE SAME QUESTION -- where do the numbers come from. One imports a measured table from outside spaCR; the other makes one from the images with a self-supervised backbone. Filed under Data by its catalog row since it was written; it had no place in this table until now, which is a tile the registry drew and Home could not sort.

### lines 2034-2036

```python
_ICON_OVERRIDES = {
```

Explicit key -> icon-filename overrides for cases where the app_key doesn't match any resource filename. Add entries here rather than renaming resource files.

### lines 2038-2045

```python
"train_cellpose":  "cellpose_masks.png",
```

ONE ENTRY, and it is the only genuine borrow left.

The Cellpose Workbench is the key `train_cellpose`, and

`train_cellpose.png` is a DUMBBELL -- the training glyph. Reported 2026-09-02: "the cellpose workbench icon should be the cellpose white ico ni made, not the train icon." So this key keeps borrowing the white cell outline, and the dumbbell stays on disk for anything that really does mean "train".

### lines 2047-2095

```python
}
```

FIVE ENTRIES WERE REMOVED HERE on 2026-09-02 -- `analyze_plaques` (plaque.png), `agreement` (annotate.png), `plate_view` (map_barcodes.png), `model_compare` (mask.png) and `model_zoo` (download.png) -- for the same reason the four before them went: each has since been given ARTWORK OF ITS OWN, and it is better than what it was borrowing. `agreement.png` is two overlapping circles, which is what agreement between two annotators looks like; `model_zoo.png` is a grid of model cards rather than a download arrow; `plate_view.png` is a plate rather than a row of barcodes. An override is for an app that BORROWS another app's picture; it is not the place to record "this app has an icon".

They were invisible until now: three surfaces resolved icons WITHOUT this table, so the fold buttons and settings headings were already showing the artwork while the tiles showed the borrow. Fixing those three surfaces is what made the staleness visible, by making all five borrows take effect everywhere at once.

FOUR entries were REMOVED here — `timelapse`→run.png, `motility`→recruitment.png, `db_browser`→map_barcodes.png and `train_compare`→classify.png — because the user chose artwork for each and it is now installed as `<key>.png`, which `app_icon` finds without being told. An override is for an app that BORROWS another app's picture; it is not the place to record "this app has an icon". (`align` and `foreign` gained artwork in the same round; neither was ever in this table — `align` was in _FORCE_GLYPH below and `foreign` had nothing at all.)

`motility` is the one worth remembering. It borrowed recruitment.png, so re-skinning Recruitment silently re-skinned Motility Assay as well — the old recruitment drawing had to be kept and installed as motility.png to stop that. A borrowed icon is a coupling between two apps that nothing declares.

Of the six left, FOUR are genuine sharing and are a debt:

`train_cellpose` shows Cellpose Masks' picture, `agreement` shows Annotate's, `plate_view` shows Map Barcodes', `model_compare` shows Mask's. The other two are only renames — no app is keyed `plaque` or `download`, so `analyze_plaques` and `model_zoo` are the sole users of that artwork and share it with nobody.

queue.png / batch.png / invasion.png / replication.png are drawn for these apps and named after them, so they need no override. They used to: `queue` and `batch` BOTH aliased sequencing.png (a DNA helix, "the closest visual match for now"), which made two different apps render identically as a picture of neither. The new pair carries the distinction that matters -- queue is the same settings over many plates, batch is arbitrary module+plate combinations in sequence.

### lines 2098-2119

```python
_FORCE_GLYPH: set = set()
```

Keys that render their qtawesome glyph instead of a bundled PNG.

EMPTY, deliberately, and kept rather than deleted: it is the documented fallback for an app whose meaning no bundled artwork carries, and emptying the set is not the same as removing the escape hatch.

``align`` was the last entry. No bundled PNG read as "tiles registered into ONE canvas", so it drew ``fa5s.border-all`` — a square divided into four by its own seams. The user has since chosen ``cellpose_all_01`` for it, which is that judgement overruled by the person whose app it is; the PNG is installed as ``align.png`` and the glyph is out of the way.

WORTH KNOWING: ``cellpose_all_01.png`` was already installed as ``cellpose_all.png``, so ``align.png`` is now byte-identical to it. No two Qt tiles collide — ``cellpose_all`` is a Tk-only module and is not in :data:`APPS` — but the Tk GUI's "Cellpose All" and Qt's Align & Stitch draw the same picture, and re-inking one re-inks both. The user chose it explicitly; this is the note, not an objection.

``invasion`` left for the same reason earlier ("no bundled PNG reads as inside vs outside") once it had artwork that did.

## _carry_preview_state

### lines 93-102

```python
target._refresh_source_selectors()
```

THE SET TABLE TOO, not just the canvas. Carrying `_image` alone left the panel showing a picture above an EMPTY table and the table is how a field is chosen, so the preview looked loaded and could not be driven. Worse, `_image_path` was carried too, so the panel read as already-loaded and pressing Choose appeared to do nothing.

`_refresh_source_selectors` re-enumerates from `_image_path`, and that enumeration is cached per folder, so the table comes back without re-scanning the disk.

## _open_at_the_measured_width

### lines 181-186

```python
recorded = _get_layout_decision()
```

RESOLVED ONCE, NOT RE-DERIVED EVERY LAUNCH (instruction 359). The answer is reused only while the evidence behind it still holds the same available geometry and the same font scale. Move to another monitor or change the font size and the record stops matching, which is the point: it is a decision ABOUT those numbers, so it expires when they do.

### lines 193-196

```python
_set_layout_decision({**metrics, "width": int(wanted),
```

RECORDED WITH THE EVIDENCE, not just the answer. A width on its own cannot be checked later; a width beside the geometry and scale it came from can be, and `reason` says in words what the artifact said in numbers.

## install_the_spaceout_fractal

### lines 247-249

```python
screen.installEventFilter(_FractalFollowsItsScreen(widget, screen))
```

It follows the screen's geometry the way the ambient backdrop does; without this it keeps its first size and a resized window shows bare ground beside it.

## _UpdateWorker.run

### lines 287-288

```python
emit_safely(self.failed, self.operation, details)
```

`emit_safely`: an exception out of a QThread::run override aborts the process, and the window may be gone by now.

## _DragsTheWindowByTheMenuBar.eventFilter

### line 347, trailing  _(unsure)_

```python
return False
```

a menu, not the bare strip

## _PipelinePreloader.start

### lines 424-426

```python
self._poll = QTimer()
```

THE CALLBACKS RUN HERE, on the GUI thread. A worker that called `on_step` directly would be touching a loading screen from off the GUI thread, which is the crash this file has already had once.

## register_app

### lines 1047-1055

```python
return row
```

The cap is a design rule, not a runtime one: a violation is fixed by splitting the section, and refusing to start the app would not help anyone do that. The suite fails on it; this makes a late registration (a plugin, a lazily-imported module) visible too. The warning this used to log is gone by request. It fired on every registration past the cap, once per app, so a full section produced a stream of identical lines at launch -- and it told the reader nothing the suite does not already assert. The cap itself still stands and tests/qt/test_cov_qt_app.py still enforces it.

## unregister_app

### lines 1074-1079

```python
for module_name, attribute, field in _META_TARGETS:
```

The side tables get the row taken back out too, or a plugin that unloads leaves a title, an intro, an API link and a "GUI-only" excuse behind for an app that no longer exists — and `test_the_gui_only_list_holds_no_apps_that_no_longer_ exist` is exactly that failure. Only entries this app put there are removed: a hand-written one was not ours to drop.

## _call_screen_factory

### line 1136  _(unsure)_

```python
params = {}
```

Builtins and C callables have no introspectable signature.

## dock_rows

### lines 1607-1608  _(unsure)_

```python
rows.sort(key=lambda row: rank.get(row[3], len(rank)))
```

Stable, so the order WITHIN a section is the registry order the dock has always used; only the grouping moves.

## _declared_folds

### lines 1781-1784

```python
if isinstance(node, ast.AnnAssign):
```

ANNOTATED ASSIGNMENTS TOO. Most hosts write

`FOLDED_APPS: Tuple[str, ...] = (...)`, which is an `AnnAssign` and not an `Assign` -- reading only the latter found the host key and an empty fold list for ten of the twelve hosts.

## _declared_folds._as_string

### lines 1828-1832

```python
if isinstance(node, ast.Attribute) and _depth == 0:
```

ANOTHER MODULE'S CONSTANT, read the same way rather than by importing it: `classify` writes `activation.APP_KEY` in its fold list, and importing `activation` to learn one string is the cost this whole function exists to avoid. One level only, which is all any host uses.

## make_home_page

### lines 1978-1983

```python
apps = tiled_apps(visible_apps())
```

BOTH filters, and they are different questions. `visible_apps` drops what the maturity preference hides; `tiled_apps` drops what has been folded into a host and reached by a button instead. Home is the one surface that wants both -- the command palette and the spaCR menu want only the first, which is why the tile filter does not live in `app_is_visible`.

## _icon_for_app

### lines 2141-2142

```python
if key in _FORCE_GLYPH:
```

Keys that should use their themed qtawesome glyph rather than a bundled PNG (e.g. train_cellpose got a fresh 'brain' glyph).

## MainWindow.__init__

### lines 2393-2396

```python
from .widgets.loading_screen import splash_role
```

The compositor may map the native window before a child has drawn. Make that first backing store an opaque splash-coloured surface, so it can never expose stale desktop pixels while LoadingScreen queues its first paint.

### lines 2405-2423

```python
self.setWindowTitle("spaCR")
```

NOT `WA_OpaquePaintEvent`. That attribute is a PROMISE that the widget paints every pixel of its own rect, and Qt takes it by skipping the erase before a repaint. This window does not keep that promise: applying the application stylesheet clears `autoFillBackground` again, so by the time the window is shown it reads `autoFill=False, opaquePaint=True` -- claiming to fill while filling nothing.

What Qt then does is leave whatever was already on screen, and transparent children draw on top of it. That is the defect reported on 2026-09-05: "in the bottom left corner there is text that overlaps (new text is pasted over old text)", together with flicker on the menu bar, the status bar and the version label every text surface that sits over the animated backdrop without a ground of its own.

The splash still paints: the stylesheet fills the window, and a styled background is drawn whether or not anything claims to be opaque. What is removed is only the false promise.

### lines 2426-2430

```python
self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
```

NO TITLE BAR. Asked for on 2026-08-23: "remove the minus and x bar from the spacr window and just have an icon in the top left for true fullscreen". The window is frameless and the menu bar is what you drag it by; Quit keeps its usual shortcut, so nothing about closing depends on a button that is no longer there.

### lines 2436-2455

```python
self._stack = QStackedWidget()
```

Central layout: a row that holds an (initially empty) dock slot and the screen stack. By default the app list is a REVEAL over the stack's left edge rather than anything in that slot.

The column was 220-320 px of the 1440 a laptop has, on every screen, holding a list most sessions never touch — and it is the reason Home could not fit its five categories plus a state column without scrolling. As a drawer it costs 6 px of trigger strip and is one hover, one click, or Ctrl+Shift+A away.

The slot is what "lock the dock" fills: see :meth:`apply_dock_mode`. It exists whatever the mode, because a QMainWindow's central widget cannot be swapped without re-parenting the stack, and re-parenting a stack that already holds live screens is how a locked dock would cost you the screen you were looking at.

`self._sidebar` is the SAME `Sidebar` object it always was, only reparented: the tutorial highlights it, the command palette and the tests all reach it by that name.

### lines 2464-2471

```python
self._dock_slot.setStyleSheet(
```

Paints nothing: the dock's own rounded panel is the only surface here, and a slot with a fill of its own puts a square back behind it. PAINTS NOTHING. The application sheet already grounds the window `QMainWindow { background-color: bg }`, or the sky gradient on the picture themes -- and anything painted here covers it. That is what put an opaque slab behind the Home masthead, which had been showing the window through it.

### lines 2478-2481

```python
slot_col.setContentsMargins(0, 0, 0, 0)
```

THE GAP AROUND THE DOCK LIVES HERE. The dock widget is itself the rounded box, and a widget's own margins are inside its background, so the space that keeps the box off the window edge has to be put around it by whatever holds it.

### lines 2492-2494

```python
self._sidebar.fold_child_selected.connect(self.open_module)
```

A folded row goes through `open_module`, not `_on_nav_selected`: the key names a fold rather than a screen, and navigating to it directly would build an orphan page with no way back.

### line 2504  _(unsure)_

```python
self._screens: dict[str, QWidget] = {}
```

Register screens lazily — created on first navigation.

### line 2525  _(unsure)_

```python
status = QStatusBar()
```

Rich status bar: transient message (left) + active app + version

### lines 2534-2537

```python
status.setSizeGripEnabled(False)
```

FIXED HEIGHT, so a longer message cannot grow the bar and relay the window out under the pointer. The module hints below write into it on every hover; without this the dock flickered on Linux each time one arrived.

### lines 2542-2546

```python
try:
```

MODULE DESCRIPTIONS GO HERE, not into a popup over the grid. Asked for on 2026-09-01; Home already worked this way and the reason is on AppTile -- these blurbs run to several hundred characters, which is fine in a fixed line and wrong in a box covering what the user is reading to choose between.

### lines 2555-2556  _(unsure)_

```python
from PySide6.QtCore import QTimer
```

The AI Console now lives inside each pipeline app's Console panel (see spacr.qt.widgets.console_panel). No side-dock.

### lines 2558-2574

```python
from PySide6.QtCore import QTimer
```

Preload the heavy pipeline imports IMMEDIATELY, behind a loading screen that covers the window until they land.

They used to start on a 1500 ms timer, which put a 2.1 s freeze on a window that already looked interactive -- measured on a real launch, `spacr.core` alone is 1968 ms and the chain is 3140 ms. The delay also predates the loading screen: it existed because kicking the chain off pre-nav once caused a circular-import race in spacr.core/IPython ("partially initialized module 'IPython'"), and sleeping through it was the cheap fix. Starting after the first screen is built still satisfies that, and this call site is after it.

The imports stay on the MAIN thread. A worker races Qt's own GPU init and segfaults -- see the note on _PipelinePreloader. Blocking the loop is acceptable precisely because nothing interactive is on screen while it happens.

### lines 2580-2596

```python
from .preferences import get_preload_policy
```

LOADED WHEN CALLED (instruction 282). Preloading is off by default now, and the maintainer's own timing report is why. Measured on a real launch, the preload thread ground for TWENTY SECONDS:

spacr.core         15.6 s     spacr.deep_spacr    9.8 s torchvision         8.5 s     torch               6.8 s torch._dynamo       5.9 s     IPython             2.5 s sympy               2.9 s     torch.distributed.fsdp  1.8 s

The torch COMPILER, sympy, DISTRIBUTED TRAINING and IPython, to draw a window. Importing them ahead of first use was supposed to move the cost earlier; what it actually did was spend it while the user was trying to work, which is worse than spending it when they ask for the thing that needs it.

Nothing is lost that was not already paid: the first run of a pipeline imports what it needs, once, exactly as it would have.

### lines 2608-2610

```python
QTimer.singleShot(1500, self._preloader.start)
```

Headless, or the screen could not be built: keep the old deferred start so a test process is not made to pay 3.1 s of imports it may not need.

### line 2616  _(unsure)_

```python
try:
```

Keyboard shortcuts — Ctrl+H, Ctrl+1..9, Ctrl+K, F1/?, etc.

### lines 2624-2626

```python
self.open_module(initial_app)
```

Through `open_module`, not straight to the key: `spacr-qt timelapse` is in shell histories and scripts, and that key is a switch on Mask Generation now rather than a screen.

### lines 2629-2633

```python
self.resume_after_restart()
```

INSTRUCTION 142: come back to where the Force restart left off. After an explicit `initial_app`, because a user who named a module on the command line is asking for that module, and a saved state that overrode it would be spaCR ignoring what it was just told.

### lines 2636-2638

```python
try:
```

178 D, for the window's own tab bars as well as the screens'. A screen built later takes the arrows off itself; this covers what is already here.

### lines 2646-2647  _(unsure)_

```python
self.refresh_language()
```

Apply the persisted language after every startup widget exists. New lazy screens are translated separately when first constructed.

### lines 2650-2653

```python
try:
```

First-launch tour — coach-marks over the home layout the first time this user boots spacr. State stored in QSettings, so subsequent launches are silent. Delayed a beat so the window has time to render before the overlay attaches.

### lines 2659-2664

```python
self._tour_timer = QTimer(self)
```

Installer privacy choices precede the product tour. The native installers collect them when they have an interactive surface; an unattended package gets the same all-off page here instead. Parent the delayed callback to the window. A static singleShot outlives a window closed during its first 800 ms, then invokes the tour with a deleted C++ object on the next event-loop spin.

### lines 2606-2607

```python
self._consent_timer.start(250)
```

Neither the installer consent nor the tour in safe mode (296). They were the only two reads that reached the real preference store while safe mode started -- every other read goes through safe mode's shadow (measured 2026-09-15: `installer/consent_applied` and `onboarding/first_run_tour_seen`) -- and both open something in front of the window the user came to repair: the tour's overlay, and a modal consent page that goes on to apply installer choices and can start an account sign-in. The timers are still created, so nothing that looks for them finds them missing; the consent timer is only never started, and the tour only ever starts from it. Nothing is recorded as applied or seen, so the next ordinary start still offers both.

## MainWindow.__init__._finish_installer_onboarding

### lines 2675-2676

```python
self._tour_timer.start(500)
```

Start after the modal flow closes, so the tour never opens behind the consent/provider dialogs' nested event loops.

## MainWindow._install_loading_screen

### line 2702

```python
LOG.debug("could not install the loading screen", exc_info=True)
```

A launch must never fail for want of a splash.

## MainWindow._on_preload_step

### lines 2714-2716

```python
screen.repaint()
```

The imports block the loop, so without an explicit repaint the screen would jump from empty to full at the end and show no progress at all.

### line 2719  _(unsure)_

```python
self._loading_screen = None
```

Deleted underneath us during teardown.

## MainWindow._install_fullscreen_button

### lines 2774-2777

```python
from PySide6.QtWidgets import QHBoxLayout, QWidget
```

TOP RIGHT, minimise then full screen -- the order a title bar puts them in. Closing is not here: Quit is in the spaCR menu with its usual shortcut, and a stray click on an x mid-analysis costs more than reaching for the menu does.

### lines 2782-2786

```python
corner.setAutoFillBackground(False)
```

A plain QWidget paints its own Window palette role.  MainWindow's first-frame palette is deliberately black, so leaving this corner implicit produced one black rectangle behind the three otherwise transparent marks.  Paint no surface here: the menu bar is the title bar and must remain visible through the whole corner widget.

### lines 2814-2816

```python
close.clicked.connect(self.close)
```

THE SAME THING QUIT DOES. Not `close()` on the window -- Quit is what every other exit path goes through, and two ways of leaving that differ is how a session ends without saving something.

### lines 2821-2877

```python
corner.setStyleSheet("""
```

THE MARK CHANGES COLOUR, NOT THE PLATE BEHIND IT. The colour is painted into the glyph by `_ChromeButton` (see CHROME_HOVER): red on the x, blue on the square and on the minus. A filled rounded plate behind a 10 px mark reads as a button growing a background rather than as the mark itself lighting up, which is what was asked for. THE BAR'S OWN COLOUR, NOT `transparent`. Same defect as `QMenuBar::item` in theme.py, and reported in the same breath: "there are black boxes behind the minimize, fullscreen and close icons... the black boxes appear only after hovering".

`transparent` means paint nothing, and what is behind this corner is the WINDOW, whose palette Window role is the splash colour -- pure black. On Linux the menu bar's fill covers that; on macOS the hover repaint clears to the window first and the black arrives as a plate behind the mark. Painting the bar's colour is identical wherever transparent already worked.

TRANSPARENT, NOT "the same colour as the bar".

This used to read menu_bar_background() and paint that, so the corner could not drift from the bar it sits on. It drifted anyway, reported 2026-09-01: "the x square and minus in the top right dont always have the same background as the container".

A colour copied once at construction is a snapshot. The bar repaints for a theme change, for a palette change, and on macOS for a translucency the copied value never had -- and every one of those leaves three plates in the old colour. Matching by copying is the bug; matching by showing through cannot drift, because there is nothing to keep in step.

Safe here in a way it is NOT for the bar itself: transparent means "paint nothing", and these sit INSIDE the menu bar, which paints its own surface. The bar is a top-level surface and would show the desktop through instead.

The hover state is unaffected: it is a repaint of the GLYPH in the hover colour, never a plate behind it. TRANSPARENT, AND IT STAYS TRANSPARENT. On 2026-09-07 this was changed to paint `menu_bar_background()` on the strength of the macOS black-box report, and that was WRONG: the maintainer superseded that fix on 2026-09-01 with "the x square and minus in the top right dont always have the same background as the container, please remove or make transparent their background color if possible".

AND THE REASON IS BETTER THAN THE ONE I OVERRODE IT WITH. Painting the bar's colour here is a SNAPSHOT: the bar repaints for a theme change, a palette change, and on macOS for a translucency the copied value never had, and each of those leaves three plates in the old colour. Matching by copying is the bug. Showing through cannot drift, because there is nothing to keep in step.

See `tests/qt/test_the_window_buttons_show_the_bar_through.py`, whose `test_the_bar_colour_is_no_longer_copied_into_the_corner` exists to stop exactly the change I made.

### lines 2896-2897

```python
self._drag_from = None
```

THE MENU BAR IS THE TITLE BAR NOW. Without this the window cannot be moved at all, which is a worse trade than the bar it replaced.

### lines 2901-2904

```python
try:
```

AND THE FRAME WAS WHERE IT WAS RESIZED. Dropping the frame took the grips with it, so the window could be moved and not resized; the edges do it now, handed to the window manager so the drag behaves like every other window on the desktop.

### lines 2912-2916

```python
action = getattr(self, "_act_fullscreen", None)
```

THE ACTION THE WINDOW SUBMENU ALREADY HOLDS. A second QAction with the same F11 shortcut is an ambiguous overload, which Qt resolves by firing neither -- so the same object is registered on the window as well, which widens its context instead of competing with it.

## MainWindow._close_icon

### lines 2942-2943

```python
pad = size * CHROME_PAD
```

THE SAME PAD THE FULL-SCREEN MARK USES, so the x spans exactly the box the square spans.

## MainWindow.eventFilter

### line 3015

```python
self._drag_from = (event.globalPosition().toPoint()
```

ONLY ON EMPTY BAR. A press on a menu opens the menu.

## MainWindow.changeEvent

### lines 3045-3050

```python
try:
```

AND AGAIN ONCE THE NEW SIZE HAS ARRIVED. `changeEvent` is delivered when the STATE changes, which is before the compositor has resized the window -- so a re-lay done only here measures the old geometry and the menu still opens against the previous action rectangle. The zero-timer runs after the resize has been delivered and the layout has settled.

## MainWindow._build_menu_bar

### lines 3114-3132

```python
if sys.platform == "darwin":
```

NOT THE NATIVE macOS MENU BAR. Qt defaults this to True on darwin, which moves the whole bar up into the system strip -- and that one default is the cause of THREE separate macOS bugs at once:

1. A native menu bar DRAWS NO CORNER WIDGET. The minimise, full screen and close marks live in this bar's top-right corner (see `_install_fullscreen_button`), so on macOS they simply were not there. The window is frameless on every platform, so that left a Mac with no window buttons at all. 2. It splits the spaCR menu in two. macOS hoists Preferences, About and Quit into the application menu -- titled "Python" for an unbundled launch -- leaving a second, half-empty "spaCR" menu beside it. 3. Nothing is left in the window to DRAG. The bar is this window's title bar; in the system strip it cannot move it.

The comment that used to sit below this said the relocation "cannot be overridden". It can -- this is how -- and turning it off gives macOS the same one-menu, three-button, draggable bar as Linux.

### lines 3135-3136

```python
self._menu_drag = _DragsTheWindowByTheMenuBar(self)
```

DRAGGABLE. The window is frameless, so without this it cannot be moved: there is no title bar for the compositor to offer.

### lines 3142-3145

```python
act_home = QAction("Home", self)
```

Preferences and Quit FIRST, as asked. This ordering is now what EVERY platform sees: the native menu bar is off on macOS (above), so Qt no longer hoists Preferences and Quit into a separate application menu and the order written here is the order shown.

### lines 3158-3166

```python
self._act_quit = act_quit
```

MINIMISE AND MAXIMISE GO IN ABOVE QUIT -- but not from here. The Window submenu builds those two actions further down this method, and the SAME objects are inserted here once it has (`_lift_the_window_actions_into_the_spacr_menu`). Two QActions for one behaviour is what this file already warns about for F11: Qt resolves a duplicated shortcut by firing neither, and even without a shortcut a second object is a second enabled state to keep in step. Remembered rather than searched for, because Quit's label is translated and matching on it would break in every other language.

### lines 3170-3174

```python
self._app_actions: dict[str, QAction] = {}
```

ONE SUBMENU PER CATEGORY. Fifty-six modules in one flat list is a column taller than most screens, and reading it means reading all of it -- "the modules should be in module category dropdowns to make it more digestable". The categories are the ones Home and the dock already use, in the same order, so the three surfaces agree.

### lines 3177-3178

```python
from .widgets.fold_strip import folded_modules
```

Read once for the whole bar rather than per section: both walk the host modules, and neither answer changes while the menu is built.

### lines 3193-3195

```python
act.setProperty("moduleAppKey", key)
```

Translate the name and reviewed scientific summary as separate semantic fields; word-by-word translation of the combined text can produce misleading mixed-language help.

### lines 3205-3211

```python
host_menu = QMenu(name, self)
```

THE SECOND LEVEL, asked for on 2026-09-01. Instruction 318 folded 33 modules onto 11 mastheads and none of them appeared here at all, so finding Volcano Explorer meant knowing it lives on Regression. The host keeps its own entry as the FIRST item rather than becoming a bare container: opening the host is still what most of these menu visits want.

### lines 3228-3231

```python
sub_act.triggered.connect(
```

`open_module` resolves the folded key to its host and switches the fold on. Reused rather than reimplemented: the routing rules live in one place and the fold strip already presses this path.

### lines 3239-3247

```python
act_all = QAction("All apps", self)
```

"All apps" is NOT in the menu: a menu entry whose purpose is not obvious from its name costs attention every time it is read, and this one names a drawer most users never knew existed.

The action itself stays, registered on the window rather than on the menu. Ctrl+Shift+A is the keyboard route into the edge reveal a panel you can otherwise summon only by hovering a 6 px strip is a panel a keyboard user does not have -- and deleting the action would take the shortcut with it.

### lines 3249-3252

```python
act_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
```

MOVED OFF Ctrl+B, which was asked for as the blank-background key twice, and it was quietly given to Ctrl+Shift+B because this already held it. A shortcut somebody asks for by name and gets something else from is worse than an unfamiliar one.

### lines 3260-3263

```python
act_backdrop = QAction("Animated background", self)
```

THE BACKDROP OFF AND ON. Registered on the window like Ctrl+Shift+A above, so it works wherever focus is. It STOPS the animation rather than hiding it: a hidden backdrop that kept rendering would be the worst of both, spending the cores and showing nothing.

### lines 3276-3278

```python
act_restart = QAction("Restart the background", self)
```

RESTART THE BACKDROP. Listed here rather than left as an undocumented key: a shortcut nobody can find is one nobody uses, and this menu is where the other two live.

### lines 3291-3293

```python
act_saver = QAction("Full-screen background", self)
```

THE BACKDROP ON ITS OWN, full screen. Ctrl+Shift+F because Ctrl+F is search everywhere and F11 is the window's own full screen this is a third thing: the animation with nothing else on top.

### lines 3304-3310

```python
act_flat = QAction("Blank the background", self)
```

PAUSE AND GO FLAT. Ctrl+T stops the animation and leaves the last frame up, which is still a picture behind the work; this one also paints the ground flat, which is what "I am looking at images and want nothing behind them" actually asks for.

Ctrl+B was explicitly requested for this action. The drawer moved to Ctrl+Shift+A, which keeps both window actions keyboard-reachable.

### lines 3326-3337

```python
demo_menu = QMenu("&Demos", mb)
```

DEMOS LIVES UNDER HELP. A demo is something you reach for when you are learning what a module does, which is what the Help menu is for, and it was taking a top-level slot on a bar that has to stay short. Built here, before Help, and added to it below the submenu is the same QMenu either way. PARENTED TO THE MENU BAR, not to the window. `first_run.find_menu` which the walkthrough and the tutorial scripts both use reaches a menu through `menuBar().findChildren(QMenu)`, because walking the bar's actions returns QMenu wrappers that go stale on PySide6 6.11. A menu parented elsewhere is invisible to that lookup, so Demos would have become unfindable the moment it stopped being a top-level menu.

### lines 3361-3372

```python
act_keys = QAction("Keyboard shortcuts", self)
```

THE HOTKEY MAP, FIRST (197). Asked for 2026-08-21: "add hotkey map to help tab".

`show_cheat_sheet` has drawn this map for a long time and was reachable from exactly two places -- the `?` key, which you have to know about, and the command palette, which you have to know about. The Help menu is where a user who does NOT already know a shortcut goes to look for one, which is the entire population this screen is for.

ABOVE THE WEB LINKS because it is the only entry here that answers without a browser.

### lines 3379-3387

```python
act_setup = QAction("Set spaCR up again…", self)
```

THE SETUP SCREEN, REACHABLE AGAIN. It ran once on the first launch and then never -- and it is the only place several of these settings are explained rather than merely offered, so a user who dismissed it lost the explanation along with the questions.

IN HELP because that is where somebody goes to be told what a choice means. Preferences is where they go when they already know and want to change it; both exist, and they answer different questions.

### lines 3397-3404

```python
for key, label, tip in _HELP_MODULES:
```

THE FOUR LOOK-IT-UP MODULES (318). Each opens exactly the screen it opened from its tile; only the door changed. They are here rather than on Home because none of them is a job a user sets out to do -- they are things you consult, which is what this menu is for.

`_on_nav_selected` is the same entry point a tile click uses, so there is one path into a module and not two that can drift.

### lines 3412-3422

```python
act_tutorial = QAction("Tutorial", self)
```

NO ICON AND NO "(web)". The icon was

`SP_MessageBoxInformation`, the platform's blue circled i, which is the glyph a dialog uses to mean "here is a notice" -- next to a menu label it read as a badge rather than as an illustration of anything. That both entries carried the SAME one made it noise twice over. Where the page opens is said in the status tip, which is where a detail belongs; a label is for what the thing is.

The catalog in `spacr/qt/i18n.py` keys on the English string, so both keys moved with these labels -- renaming here alone would drop the translation in nine languages.

### lines 3455-3465

```python
self._act_preferences = act_prefs
```

Every menu action gets an EXPLICIT macOS role, and everything that is not genuinely Preferences/Quit/About gets NoRole. Left to Qt, the role is guessed from the action's TEXT, and an action whose text merely contains "settings" or "options" is silently moved out of its menu into the application menu -- which is how `recipes.MENU_ACTION_TEXT` ("Settings recipes…") ended up as the Preferences item of the macOS "python" menu while the real Preferences and Quit vanished from this one. See spacr.qt.menus.

Collected from the menu bar rather than listed by hand, so an action added later is covered without anyone remembering to.

## MainWindow._build_window_menu

### lines 3551-3554

```python
act_full = QAction("Full screen", self)
```

THE SAME ACTION THE WINDOW ITSELF CARRIES, not a second one with the same shortcut. Two distinct QActions bound to F11 on one window is an ambiguous overload and Qt then fires NEITHER, so a menu copy would have cost the key it advertises.

### lines 3571-3574

```python
act_prefs_here = QAction("Preferences…", self)
```

THE TEXT IS COPIED VERBATIM from the two actions in the spaCR menu. `spacr.qt.i18n` keys its catalog on the English string, so a copy worded differently would be a copy that stays English in the other nine languages.

## MainWindow._menu_bar_actions

### lines 3645-3647

```python
for action in list(menu.actions()) + [menu.menuAction()]:
```

The menu's OWN action too -- the one that opens it from the bar (or from a parent menu). It is an action like any other and Qt will happily give "&Options" a role if left to guess.

### lines 3651-3656

```python
if not action.text():
```

Qt's role heuristic matches on TEXT, so an action with no text has nothing to match and cannot be relocated. They turn up here because `menuAction()` CREATES one for a menu that has never been attached to a bar -- i.e. this walk can manufacture them. Skipping keeps the sweep and the audit agreeing about what exists.

## MainWindow

### lines 3673-3676

```python
DEMO_TARGETS = {
```

demos

Map each demo key to (target-app key, generator function name). Kept as a class constant so tests can introspect it without launching the file dialog.

### lines 3682-3686

```python
"timelapse": ("mask",       "generate_timelapse_demo"),
```

The timelapse demo writes a settings CSV with timelapse=True, and that key has no widget on the Mask form -- the masthead switch is its control. `AppScreen.apply_settings_dict` moves the switch from the dict it applies, so the demo lands on Mask with tracking already on and its categories already showing.

## MainWindow._run_e2e_chain

### lines 3811-3814

```python
from .settings_pack import settings_from_pack
```

MIGRATED, NOT MERGED. `settings_from_pack` reads the pack against this build's settings and reports what it could not place -- see spacr/qt/settings_pack.py for why reading a CSV straight over the defaults was the bug rather than the shortcut.

### lines 3825-3828

```python
settings = _settings_for("mask")
```

ONE STAGE. Measure and Annotate are reached from Mask

Generation once there are masks to measure; opening them now, against a dataset with no masks in it, would open two screens that can only report that there is nothing to do.

## MainWindow._apply_demo_to_screen

### line 3867  _(unsure)_

```python
if hasattr(widget, "apply_settings_dict") and layout.settings_csv:
```

AppScreen: load the CSV into its settings model

### line 3876  _(unsure)_

```python
if hasattr(widget, "_open_source"):
```

AnnotateScreen: takes a src folder directly

### line 3880  _(unsure)_

```python
if hasattr(widget, "_open_folder"):
```

MakeMasksScreen: opens a folder directly

## MainWindow._show_about

### lines 3943-3946

```python
try:
```

The PNG straight off disk, not `iconset.icon()`. That helper recolours an icon to the theme's ink so monochrome glyphs stay legible — correct for a toolbar symbol, wrong for a logo, which has its own colours and should look the same on every theme.

### lines 3981-3983

```python
license_link = (
```

The license NAME is a legal identifier and stays English; the sentence around it does not. Kept as one placeholder so a language that puts the name elsewhere in the clause can move it.

## MainWindow.refresh_theme

### lines 4039-4041

```python
try:
```

BEFORE the per-screen loop below, which reads `window_backdrop` and would otherwise reconcile every screen against the state the window had a moment ago.

### lines 4053-4056

```python
try:
```

Preferences is where the window's backdrop is turned on and off, so it is where a cached screen's record of that backdrop stops being true. Reconciling here is the only thing that clears the flag on a screen the user is not looking at.

## MainWindow._refresh_demo_status_tips

### line 4097

```python
pass
```

A deleted action during shutdown must not stop the rest.

## MainWindow._rebuild_startup_page

### lines 4116-4118

```python
try:
```

close() before deleteLater() so the outgoing page drops its subscription to the run registry now, rather than staying a live receiver until the deferred delete is flushed.

## MainWindow._on_upgrade_done

### lines 4207-4210

```python
lines = [line for line in (output or "").splitlines() if line.strip()]
```

These installs launch from a desktop entry with Terminal=false, so "check the terminal for details" named something the user could not open, and the reason was written to a stream nobody was reading. Put the tail of it in the dialog instead.

### lines 4213-4218

```python
escape = _the_missing_pip_escape(output)
```

AND THE ONE FAILURE THIS CAN ANSWER, IT ANSWERS. A venv built by `uv venv` has no pip, so an updater that reaches for pip fails before it starts -- and the fix for that cannot arrive through the updater. Here the application knows the exact command that would work, so it says it rather than leaving the user with an exit code.

## MainWindow.closeEvent

### lines 4263-4268

```python
from .screens.app_screen import AppScreen
```

Closing a parent widget does not deliver a close event to its child widgets. AppScreen.closeEvent owns cleanup that cannot be left to Qt's child-destruction cascade, notably its parentless pyqtgraph menus and background job runners. Ask each owned screen to close while it is still intact, and honour a screen that defers shutdown because one of its workers has not reached a safe boundary.

### line 4278, trailing  _(unsure)_

```python
continue
```

already deleted -- nothing left to drain

### lines 4296-4300

```python
worker = getattr(self, "_update_worker", None)
```

Help → "Check for updates…" runs its network call on a QThread parented to this window. Quitting while it's in flight destroys a live QThread, which is the same abort the console drain above exists to prevent. The updater's own socket timeouts are a few seconds, so the wait is bounded twice over.

### line 4306, trailing  _(unsure)_

```python
pass
```

already deleted — nothing left to wait for

### lines 4308-4311

```python
if event.isAccepted():
```

WITH quitOnLastWindowClosed OFF, THIS IS WHAT ENDS THE PROGRAM. Nothing else may: a figure window closing must not take the session with it, and until this closes the application stays up even with no window on screen.

## MainWindow._show_module_hint

### line 4369

```python
import logging
```

A hover handler must not take the window with it.

## MainWindow._set_backdrop_blank

### lines 4461-4464

```python
act = getattr(self, "_act_backdrop", None)
```

The ground the backdrop was covering is the theme's own window colour, so nothing has to be painted -- uncovering it is enough. Keeping the two toggles agreeing is what stops Ctrl+T from appearing to do nothing while the background is blanked.

## MainWindow.resume_after_restart

### lines 4549-4551

```python
key = self.open_module(key)
```

The record was written by whatever screen was open, under whatever key it had then; a module folded since is reopened on the host that took it over.

## MainWindow._install_startup_page

### lines 4631-4633

```python
try:
```

The hero's "All apps" button is the labelled twin of the edge reveal — a discoverable way in for anyone who never finds the hot strip, and the thing a screenshot can point at.

## MainWindow._show_the_screensaver

### lines 4662-4663  _(unsure)_

```python
self._screensaver = saver
```

HELD, or Python frees the only reference and the window closes the instant it opens.

## MainWindow.rebuild_app_screen

### lines 4827-4831

```python
previous_build_values = AppScreen.values_the_next_screen_is_built_for
```

BUILT BEFORE THE OLD ONE IS TAKEN AWAY. Removing it from the stack first drops the window to whatever is left showing -- Home -- so typing a channel value sent the user back to the start screen and then returned them, which is not a visibility toggle by any reading. The stack only ever changes once the replacement exists.

### lines 4840-4842

```python
AppScreen.values_the_next_screen_is_built_for = previous_build_values
```

ALWAYS CLEARED. Every other module open must build from the module's own defaults, and a value left here would shape the next screen somebody opened for reasons they could not see.

### lines 4862-4864

```python
try:
```

THE SHAPE IT WAS BUILT FOR, recorded before it is shown: the signals that fire as it settles would otherwise see a shape they have no record of and rebuild it again.

### lines 4873-4885

```python
_carry_preview_state(old, fresh)
```

THE LOADED PREVIEW IMAGE SURVIVES THE REBUILD.

This rebuild carries the user's VALUES across and always has. It did not carry the live preview's loaded image, and the preview lives on the screen being replaced -- so typing a channel number, which is a shaping value and therefore rebuilds, silently emptied the preview. Reported as "the images are gone every time I put in a number for an object channel", and it made the preview unusable for exactly the task it exists for: setting the channels while watching the result.

The IMAGE is carried, not the path. Re-reading from the path would be wrong for a dropped file that is not under `src` at all, and would put a disk read on the rebuild.

### lines 4896-4900

```python
if old.close() is False:
```

``deleteLater`` alone bypasses ``AppScreen.closeEvent``. Closing first retires workers, workspace providers, figure resources and parentless pyqtgraph menus. The replacement was built first so the stack never flashes Home while the comparatively expensive form is constructed.

### lines 4902-4904

```python
self._stack.removeWidget(fresh)
```

A running worker may deliberately defer close. Keep that live screen instead of destroying work in flight, and retire the unused replacement cleanly.

### lines 4913-4915

```python
fresh.register_workspace()
```

The old screen and the replacement own the same stable workspace keys. Old's close withdrew them, so publish the replacement again after teardown.

## MainWindow._on_nav_selected

### lines 4932-4935

```python
try:
```

Re-read the things that go stale while Home is off screen:

the plate queue, the run journal and the disk/GPU figures. Cheap (a JSON read and three stat calls) and only on a deliberate return to Home, not on a timer.

### lines 4941-4944

```python
self._status_app_label.setText(tr("Home"))
```

Translated here, not left raw: the startup pass renders this label once, and re-applying the English source over it both shows the wrong word and opts the label out of every later retranslation.

### lines 4954-4957

```python
self._rebuild_for_scale(key)
```

Built at a scale that no longer applies. Rebuilding here rather than at the moment the scale changed keeps the cost on the open the user is already waiting through, and it goes through the same path a shape change uses, which carries their values.

### lines 4960-4977

```python
card = self._show_preparing(key)
```

SOMETHING ON SCREEN BEFORE THE WORK STARTS. The build cannot move off the GUI thread -- Qt forbids making widgets anywhere else, and painting is the GUI thread's too, which is why the backdrop looked frozen while a module opened even though its renderer never stopped. What CAN change is that the user is looking at a module that says it is preparing, rather than at the old screen doing nothing.

THE BUILD DOES NOT YIELD. This comment used to claim it

"yields to the event loop on a 25 ms deadline, so this card animates while the widgets are made", and there is no such mechanism here: the `processEvents` in `_show_preparing` is a single paint BEFORE the work, and `_build_screen` then runs to completion. The card is drawn once and then sits still, which is better than the old screen sitting still but is not what was written. Corrected 2026-09-01 while measuring instruction 314, because a false comment is worse than no comment when someone is hunting a stall.

### lines 4984-4993

```python
try:
```

Every screen gets the same page treatment here, because this is the one place they all pass through. It cannot live in `AppScreen`: most screens are not AppScreens — Annotate, Align & Stitch, Format Converter, Import Project, Plate Queue, Batch Runner, Distributed Jobs, Database Browser, Make Masks, Model Compare, Model Zoo, Plate Viewer, Annotator Agreement, Training Runs, Classifier Evaluation, Run History and Report are plain QWidget trees, so they never got the backdrop or the surface clearing and sat as black slabs while the pipeline screens did not.

### line 4997

```python
LOG.exception("Could not theme the %s screen", key)
```

Decoration must never stop a screen from opening.

### lines 5007-5011

```python
try:
```

THE HELP GOES ON THE NAMES, for every screen and not only the ones built from settings rows. Here because this is the one place they all pass through, and after the translate above: that pass re-applies each setting's tooltip to whatever carries its key, so moving the help before it runs is undone a moment later.

### lines 5017-5018

```python
LOG.exception("Could not retarget help on the %s screen", key)
```

Help in the wrong place is a blemish, never a reason for a module not to open.

### lines 5021-5023

```python
_timing.watch_interactive(
```

Constructor return is not readiness.  The event filter records only after this page and one of its enabled controls have both painted on an event-loop turn, which is the state a user can actually operate.

### lines 5029-5032

```python
if key in self._visit_order:
```

Move this app to the end of the visit list. Revisiting an app has to count as the most recent visit — otherwise "Add current plate" on the Queue screen picks up whichever app was OPENED last rather than the one that was on screen a moment ago.

### line 5036  _(unsure)_

```python
name = tr(next((n for k, n, _d, _s in APPS if k == key), key))
```

Find nice display name

## MainWindow._theme_screen

### lines 5088-5090

```python
ensure_widget_qss_applied(root=screen)
```

A local stylesheet reaches this root and its descendants without making QApplication re-polish Home and every cached module.  The root is not in the stack yet, so these rules win the first paint.

## MainWindow._install_screen_backdrop

### lines 5125-5148

```python
from .widgets.ambient import (install_ambient,
```

THE INSTALL-SIDE HALF OF THE DEDUP, AND IT IS OFF WITH THE

OTHER HALF. This declined to build a screen's own backdrop whenever the window had one, on the "one backdrop for the window" reasoning that `_drop_a_redundant_screen_backdrop` states at length -- and that whole argument is suspended, because the window's backdrop is NOT visible through the screens above it and the result was a black page. See instruction 381, and `ded192cc4` which turned off the retire side on 2026-09-07.

LEAVING THIS ONE ON WOULD HAVE BEEN THE WORST OF BOTH: screens that never build a backdrop, deferring to a window backdrop nobody can see. `AppScreen` is unaffected either way -- it returns before this and installs its own -- so what this guard actually governed was the plain screens, which are the ones with no second chance.

Both halves go back on together, with the pixel measurement 381 asks for. NOT WHILE A HEAVY IMPORT IS RUNNING, and this runs on the GUI thread as a module is being opened -- which is precisely when the preloader is holding the lock. `AppScreen` has taken this care since instruction 315; the screens that build their own had not, so under `spaceout` they froze where `spacr` did not.

### lines 5154-5157

```python
install_ambient(
```

The spaceout fractal is installed by `install_ambient` itself (instruction 260), so this caller needs no branch: hooking the three call sites separately is what left the Home screen still showing the old artwork.

### lines 5163-5167

```python
try:
```

The peek is a check, not a reservation: the preloader re-takes the lock between two imports, so the refusal can still arrive here. It means "not yet" and not "this machine cannot", and logging it as an exception would put a traceback in the console for an ordinary click made during startup.

## MainWindow._drop_a_redundant_screen_backdrop

### lines 5250-5258

```python
screen._uses_window_backdrop = False
```

NOT A BARE RETURN, WHICH IS WHAT IT WAS AND WHAT WAS WRONG. The flag is a claim about the window as it stands NOW, and it was only ever set, never cleared. So a screen that had once shared a window backdrop kept `page_fill` returning None after the animation was switched off in Preferences, and painted the flat `bg` slab instead of the page colour -- the black page, reported three times and caught again by `tests/qt/test_page_is_never_black.py` at every zoom with `ambient_enabled=False`.

### lines 5263-5301

```python
if getattr(screen, "_surrendered_its_backdrop", False):
```

LEFT ALONE, DELIBERATELY, AND IT WAS BRIEFLY NOT.

For one day this branch set `_uses_window_backdrop = True` on the reasoning that a screen with no backdrop of its own must be deferring to the window's. That is false for the screen this matters most on: HomePage never builds one, so it had never deferred to anything -- it painted its page colour as the FLOOR beneath the window's animation. Setting the flag took that floor away, `page_fill` returned None, the page painted nothing, and `bg` showed through: a pure black home screen with the blobs theme on, reported within hours.

The flag means "this screen gave its backdrop up", which only the code below can know. A screen that never had one is not in that state and must keep painting its page.

BUT "HAS NO BACKDROP OF ITS OWN" AND "NEVER HAD ONE" ARE NOT

THE SAME QUESTION, and reading them as one left a slab on screen. A screen that surrenders its animation below has `_ambient` None for the rest of its life, so it arrives here every time afterwards -- and the reconcile at the end of `refresh_theme` CLEARS the flag whenever the window has no backdrop, while nothing ever set it again. Switch the animation off in Preferences and on again, and every module already built goes on painting its flat page colour straight over the restored animation for the rest of the session, while a module opened after the toggle is correct. Measured offscreen, dark theme, 30 % page opacity, the settings column of a module that is open throughout:

the module is opened      1.00 page, 0.70 panels the animation is off      0.00, one flat slab   (correct) the animation is on again 0.00, one flat slab   (WRONG) a module opened after     1.00 / 0.70           (correct)

So the surrender is recorded durably below, and this branch reads that record rather than the widget. HomePage never surrenders anything, so it never carries the record and the floor it paints is untouched by any of this.

### lines 5306-5309

```python
screen._uses_window_backdrop = True
```

RECORDED BEFORE THE WIDGET GOES. `page_fill` returns a flat colour whenever `_ambient` is None, so a screen that merely lost its own backdrop would paint that colour straight over the window's animation -- the black slab, reported three times.

### lines 5311-5316

```python
screen._surrendered_its_backdrop = True
```

AND RECORDED DURABLY, because the line above is a claim about the window as it stands now and gets cleared when the window's backdrop goes. This one is a claim about the SCREEN -- it gave its animation away and cannot paint a page under one again -- and it is what the branch above reads to tell this screen apart from a screen that never had a backdrop at all.

### lines 5318-5323

```python
try:
```

RETIRED HERE, NOT BY THE SCREEN. `_discard_ambient` exists on HomePage alone -- AppScreen has no such method -- so delegating to it silently did nothing for module screens while still clearing `_ambient`, which left an orphaned widget animating with no reference to it. Stopping the timer is the part that matters: an unparented AmbientWidget awaiting deleteLater still ticks.

## MainWindow._a_page_joined_the_stack

### lines 5413-5414

```python
LOG.exception("Could not mark a new page for the theme sheet")
```

Marking must never stop a screen from opening; an unmarked page is repainted by the next theme change either way.

## MainWindow.stylesheet_roots

### lines 5457-5471

```python
current = stack.currentWidget()
```

EVERY DIRECT CHILD OF THE STACK, not every indexed PAGE. The sidebar's `EdgeDrawer` is parented to the stack and is not one of its pages, so a loop over `stack.widget(i)` misses it and misses every dock row inside it -- found by the guard, which reported exactly one genuinely unsheeted widget out of 207 sampled and named its ancestry. A DIRECT CHILD IN NO LAYOUT CAN BE A WINDOW, and a sibling session was bitten by exactly that category today -- a `QDialog` parented to a panel is a direct child, is in no layout, and carries `Qt::Window`, so a sweep that re-parented such children would have swallowed a live dialog. Nothing is re-parented here: a dialog that lands in this list is merely SHEETED, which is what the event filter does to it anyway the moment it is shown. The double application is idempotent. Recorded because the category is the trap, not this use of it.

### lines 5476-5478

```python
if page is current or page.isVisible():
```

VISIBLE RATHER THAN CURRENT, for the same reason: the drawer is on screen beside the page, and "the one the stack would raise" is not the same question as "the one the user sees".

## MainWindow._backdrop_the_dock_column

### lines 5524-5529

```python
self._retire_the_dock_backdrop()
```

RETIRED HERE, NOT LEFT RUNNING. This used to be a bare

`return`, so the method only ever built. Switching the animation off in Preferences calls `refresh_theme`, which calls this -- and the backdrop went on animating until the next launch, with every screen still deferring to it. The setting appeared to do nothing.

### lines 5537-5538

```python
QTimer.singleShot(400, self._backdrop_the_dock_column)
```

The preloader is importing torch. Same answer the screens give: come back rather than build a GL context beside it.

## MainWindow._show_preparing

### lines 5646-5648

```python
from PySide6.QtCore import QCoreApplication, QEventLoop
```

ONE PAINT BEFORE THE WORK. Without this the card is created and the build starts in the same tick, so it is never drawn and the user sees the freeze it exists to replace.

## MainWindow._build_screen_timed

### lines 5723-5726

```python
from .screens.train_cellpose import CellposeWorkbenchScreen
```

WRITTEN OUT RATHER THAN CALLING THE MODULE'S OWN FACTORY:

tests/qt/test_all_module_smoke.py reads this method's bytecode for the `self._on_*` slots it wires, and a factory call hides them from it.

## MainWindow._snapshot_current_screen_settings

### lines 5797-5798  _(unsure)_

```python
from .screens.app_screen import AppScreen
```

Prefer the most-recently-viewed AppScreen — the Queue screen itself isn't one.

### lines 5802-5806

```python
for key in reversed(self._visit_order):
```

Fall back to the last non-queue AppScreen the user visited. Walk the VISIT order, not `_screens` (creation) order: a user who opens Mask, then Measure, then goes back to Mask and hits "Add current plate" means Mask — creation order would hand them Measure's settings under Mask's nose.

## MainWindow._on_train_requested

### lines 5847-5850

```python
seeder = getattr(widget, "apply_seed", None)
```

A screen that is not settings-driven says what to do with a seed itself. The Database Browser has no settings model -- it takes a database path and a table -- and without this the navigation happened and the seed was silently dropped.

### lines 5873-5876

```python
sync = getattr(widget, "_sync_folded_switches", None)
```

A folded module's pipeline GATE has no widget for the loop above to land in -- the masthead switch is its control -- so a seed that asks for tracking has to move the switch, exactly as an imported settings CSV does.

## _use_open_sans

### lines 5951-5953

```python
font.setWeight(QFont.Weight.Light if str(weight).lower() == "light"
```

QFont.Light is 300 and Normal is 400. Asked for by weight rather than by family name: "Open Sans Light" is a family on some platforms and not on others, and the weight works on both.

### lines 5956-5958

```python
existing = app.font()
```

The size the platform chose is kept: the font-scale preference is applied on top of it later, and overriding it here would silently undo that.

## _install_crash_dump

### lines 6016-6018

```python
globals()["_CRASH_DUMP_FILE"] = handle
```

Kept on the module so the handle cannot be garbage collected faulthandler writes to the file descriptor, and a closed one is a crash inside the crash handler.

## launch

### lines 6079-6082

```python
from .crash_recovery import (note_that_a_launch_began,
```

BEFORE THE BACKDROP IS BUILT. Two unclean exits in a row is treated as a pattern, and the next start is made without the one thing spaCR asks a driver to do -- because the setting that would turn it off is behind the window that never appears.

### lines 6096-6097

```python
from .setup_screen import take_the_setup_flags
```

`--no-setup` and friends are taken out first, because the next line reads argv[0] as a module name and would look one of them up.

### line 6102  _(unsure)_

```python
initial_app = argv[0] if argv else None
```

Support `spacr-qt <app>` to open directly into an app.

### lines 6105-6118

```python
_install_crash_dump()
```

A FATAL SIGNAL MUST LEAVE A STACK BEHIND.

Reported 2026-08-19 three times: a regression run closes [success] and the process is gone milliseconds later. The log ends mid-session with no shutdown lines, so it is not a clean exit; dmesg and coredumpctl have nothing; and Python prints nothing because the process dies below Python, in Qt or in a C extension. Three hypotheses were tested and eliminated against real sessions -- an off-thread plt.show(), pyplot building Qt canvases on the worker, and quitOnLastWindowClosed -- each costing a launch-and-reproduce cycle for the maintainer.

faulthandler writes the Python stack of EVERY thread on SIGSEGV, SIGABRT, SIGBUS and SIGFPE. It costs nothing until one arrives, and the next occurrence then names the frame instead of the minute.

### line 6121  _(unsure)_

```python
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
```

Enable high-DPI early.

### lines 6125-6156

```python
try:
```

PYPLOT MUST NEVER MAKE A Qt CANVAS. Set here, before any figure can exist, because switching the backend later CLOSES every open figure.

A regression runs on a JobRunner worker and draws with pyplot. Under the `qtagg` backend every `plt.figure()` on that worker builds a FigureCanvasQTAgg -- a QObject whose thread affinity is the WORKER. The main thread then renders it, and Qt answers with "QBasicTimer::start: Timers cannot be started from another thread" followed, milliseconds after `run closed [success]`, by the process going away with no Python traceback. That is the reported "it just spontaneously quit", and the `Internal C++ object (FigureCanvasQTAgg) already deleted` errors in the log are the same object seen from the other side.

`bridge` already asked for Agg with `force=False`, which does NOTHING once a backend is active -- and by the time a run starts, `qtagg` is.

NOTHING IS LOST. The two places that genuinely want a Qt canvas (`figure_queue`, `umap_explorer`) import FigureCanvasQTAgg and build it themselves on the GUI thread, which works under any global backend.

SAID IN THE ENVIRONMENT WHEN MATPLOTLIB IS NOT LOADED YET, which on a normal launch it is not: nothing imported up to this line has needed it. `matplotlib.use` can only speak to a matplotlib that exists, so calling it here used to import the whole package -- tens of milliseconds of a launch that has not yet drawn anything -- purely to set a string that MPLBACKEND sets for free, and that matplotlib reads for itself whenever it does load. Assigned rather than `setdefault`: the reason this exists is that a Qt canvas built off the GUI thread kills the process, so the choice is not the caller's to override.

And when matplotlib IS already imported the environment is too late the backend was read at its import -- so that case still forces it.

### lines 6167-6182

```python
QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)
```

EVERY DIALOG IS QT'S OWN, NOT THE DESKTOP'S. Instruction 151: a native dialog on this desktop is brokered through xdg-desktop-portal, and a brokered dialog is the tens-of-seconds stall reported as "changing the line width takes like 1 minute" -- the restyle itself measured at 0.000 s. The colour pickers were fixed one call site at a time; there are 117 QFileDialog calls across the widget package and five of them passed the option, so per-site fixing was never going to converge.

SET BEFORE THE QApplication EXISTS, which is what Qt requires of this attribute -- after construction it is ignored, silently, which would look exactly like it had worked.

The trade is real and worth stating: Qt's dialogs do not carry the desktop's bookmarks or its recent-files list. A file chooser that opens is better than a beautiful one that takes a minute, and a user who wants the native one can still set QT_QPA_PLATFORMTHEME.

### lines 6185-6201

```python
from .menus import name_the_application
```

THE APPLICATION IS NAMED BEFORE IT EXISTS.

On macOS the application menu -- the one beside the Apple logo, and the one Qt moves Preferences, Quit and About INTO -- is built while the Cocoa plugin comes up inside the QApplication constructor, from the application name and from the running bundle's CFBundleName. Naming the application on the line after that constructor is a name the menu never read: it stays "python", "spacr-qt" or "PySideApp" depending on how the launch happened, and the maintainer's report -- "preferences and quit are for some reason not in the spacr dropdown" -- is that menu being somewhere they had no reason to look.

`applicationDisplayName` is set too. It was never set at all, and it is the name Qt shows to people rather than the one it keys settings on.

Returns what actually took effect rather than what was asked for, so the launch log records the answer instead of the intention.

### lines 6207-6225

```python
if os.environ.get("SPACR_WATCH_GUI_STALLS"):
```

NAME THE CALL THAT FREEZES THE INTERFACE, when asked to. A stalled GUI thread leaves no traceback: it is not an exception and not a fault, so `faulthandler` cannot see it and the log ends mid-sentence. The only thing that can name it is a sample of the main thread's stack taken WHILE it is stuck, from another thread.

WHY IT IS HERE AND NOT IN A TOOL. `tools/watch_the_gui_thread.py` does exactly this and has to make the application itself to do it, which stopped working the day `launch` began constructing its own. Reaching in from outside was then tried two more ways and failed twice more: hosting `qt.run()` in another process makes Home time out at 30 s, and a `sitecustomize` on the benchmark worker's `PYTHONPATH` runs but its patched `QApplication` is never the one constructed. An environment flag read HERE is the one place that cannot miss.

Measured on 2026-09-11: opening Measure is 13.3 s of which a single event-loop stall is 13.0 s, and `mask` and `regression` have the same shape -- so this is not a Measure question, it is how every module opens. See instruction 380.

### lines 6232-6235

```python
try:
```

NAME THE CALLER OF AN OFF-THREAD TIMER START, because Qt will not. Its own warning has no file, no function and no thread in it, and the event arrives during a real run and is followed by the process dying -- so there is no opportunity to switch instrumentation on afterwards.

### lines 6241-6246

```python
try:
```

AND TAKE CYCLIC COLLECTION OFF THE WORKER THREADS. A collection runs on whichever thread allocated past a threshold, and it runs destructors there -- so a Cellpose pass in a preview worker would destroy some widget the GUI thread had abandoned, and Qt cannot stop that widget's timer from a foreign thread. See spacr.qt.gc_policy for the reproduction of the exact crash this prevents.

### lines 6252-6254

```python
LOG.info("application named %r (display %r); Qt reports %r / %r",
```

Logged as MEASURED rather than as intended: a name that silently failed to take looks exactly like one that worked until somebody opens the menu on a Mac, and this line is what a bug report can be read against.

### lines 6258-6263

```python
try:
```

LAPTOP MODE, decided once and SAID. It is the fallback the laptop instruction calls the fallback -- reached after the optimisations, and it turns down what is decorative rather than removing what a module does. Nothing it touches is read by a pipeline, so a run computes the same answer either way; that is what makes deciding automatically acceptable. Overridable through SPACR_LAPTOP_MODE either way.

### lines 6272-6273  _(unsure)_

```python
app.setDesktopFileName("io.github.olafssonlab.spacr")
```

Linux shells resolve dock/switcher identity through the desktop-file id (Wayland does not use setWindowIcon for that surface).

### lines 6278-6281

```python
try:
```

Lift Qt's default 256 MB QImageReader allocation limit. Large multi-panel figures rendered at high DPI decode to well over 256 MB, and hitting the limit makes QPixmap loads fail (blank figures) and the UI hang. 0 = no limit; the figure queue still caps display resolution for sanity.

### lines 6289-6292

```python
with _timing.span("fonts"):
```

Bundle Open Sans (Regular + Light + SemiBold) so the app renders the same on every OS regardless of what fonts the user has installed. Registered before applying the stylesheet so any `font-family: "Open Sans"` rule resolves.

### lines 6297-6298  _(unsure)_

```python
from .preferences import apply_preferences_to_app
```

Apply user preferences (theme + font scale) — falls back to the dark defaults on the first launch when nothing is stored yet.

### lines 6301-6305

```python
from .i18n import install_qt_translations
```

QT'S OWN WORDS. Copy, Paste, Select All, a file dialog's whole chrome and every message box's buttons come from Qt's catalogs, not from spaCR's, so they stay English until this is loaded. This one is not a dialog filter and stays here: it is read while the main window's own menus and buttons are built.

### lines 6309-6311

```python
from .logging_util import setup_logging
```

Real Python logging → rotating file + Qt signal so ConsolePanel can render records inline. Set it up before the launch breadcrumb and MainWindow construction so neither is lost.

### line 6315  _(unsure)_

```python
import logging as _lg
```

Every launch drops a timeline marker into the diagnostic log.

### lines 6325-6340

```python
app.setQuitOnLastWindowClosed(False)
```

THE APPLICATION OUTLIVES ITS WINDOWS UNTIL THE MAIN ONE CLOSES.

Qt's default is quitOnLastWindowClosed=True: the moment the number of visible top-level windows reaches zero, the event loop stops and the process exits CLEANLY -- no traceback, no core dump, the log simply ends. That is exactly the evidence for the reported "i ran it again and it just spontaneously quit": two runs closed [success], the log stops mid-session, and neither dmesg nor coredumpctl recorded anything, because nothing crashed.

A run makes and destroys top-level windows -- a figure canvas being rebuilt, a transient dialog, a progress window -- and any instant with none of them up while the main window is not counted takes the whole application with it.

So the main window decides, and it is the only thing that does.

### lines 6343-6355

```python
if not told_to_skip_setup:
```

THE FIRST RUN ASKS ITS QUESTIONS BEFORE THERE IS AN APPLICATION TO ASK THEM OVER (instruction 221, reordered).

It used to run after `win.show()` so that it had something to blur. That put a half-built main window -- wrong language, wrong theme, wrong font -- on screen for as long as the setup took, and then restyled it under the user while they were reading. The answers given here decide how the main window is BUILT, so they have to be given first. The screen carries its own backdrop and does not need a window behind it.

A launch can decline: `--no-setup`, `SPACR_NO_SETUP=1`, or an offscreen platform plugin. See `setup_screen.skipped_on_purpose`.

### lines 6358-6361

```python
from .widgets.setup_slides import open_setup_if_needed
```

THE SLIDES, not the grouped form (instruction 234). `setup_dialog` is 221's version and stays: it is the fallback nothing currently uses, and deleting it would take its tests with it while the new presentation is still settling.

### lines 6365-6373

```python
if asked is not None:
```

An answer may have changed the language, the theme or the font scale, and the main window has not been built yet -- so it is built from the new values rather than restyled into them.

ONLY WHEN THERE WERE ANSWERS. `open_setup_if_needed` returns

None when the screen was not shown -- this profile has already answered, or nobody is there to -- which is every launch after the first, and re-applying preferences nothing changed is the whole theme resolved and set on the application twice.

### lines 6377-6379

```python
LOG.debug("could not open the setup screen", exc_info=True)
```

A setup screen is not worth a launch. Every question it asks has a working default, so a user who never sees it is exactly where a user who dismissed it would be.

### lines 6388-6390

```python
benchmark_controller = _maybe_start_benchmark(app, win)
```

This is an explicitly requested, unattended acceptance run.  An instrumentation setup error must fail the worker promptly instead of opening a GUI that can sit until the driver's outer timeout.

### lines 6393-6396

```python
if win._stack.currentWidget() is win._startup:
```

Home is not ready because its constructor returned or because show() was called.  Install before show so no paint can escape the observer; the probe also requires a callback delivered after app.exec() begins and an enabled, visible control whose own paint event has completed.

### lines 6403-6405

```python
win._startup_benchmark_controller = benchmark_controller
```

Retain the optional controller with the window.  QObject parenting is sufficient for C++ lifetime, but the explicit Python reference avoids wrapper collection differences across the supported PySide releases.

### lines 6407-6413

```python
_open_at_the_measured_width(win)
```

Opens at its own size rather than maximised. Maximising assumes a desktop: over X11 forwarding, VNC or a virtual framebuffer the "available geometry" is whatever the remote session claims, which is frequently one enormous virtual desktop or a 640x480 stub, and the window arrives unusable either way. The user can still maximise it, and the 1200x720 minimum this window declares is a sane opening size on a real display.

### lines 6417-6422

```python
install_the_dialog_filters(app)
```

AND ONLY NOW THE DIALOG FILTERS. See :data:`_DIALOG_FILTERS`: they are application-wide event filters that concern dialogs alone, so every event the main window's construction delivers used to run three Python callables that could not act on it. Installed here they are in place before the event loop -- which is before any dialog can be opened and the window they cannot help build arrives 0.4 s sooner.

### lines 6425-6436

```python
try:
```

Hold Z, turn the wheel, and the text resizes under the pointer (instruction 378). Installed beside the dialog filters and for the same reason: it is application-wide, so it wants to exist before the event loop but not while the main window is still being built. Two integer comparisons per event when the key is up -- see `live_zoom`.

GUARDED, LIKE THE FILTERS ABOVE IT. `install_the_dialog_filters` wraps each installer precisely so a broken application-wide filter costs its own feature and not the launch; this one was placed beside them and given none of that, so an exception here would have taken the whole application down before the window was shown -- to lose a font gesture.

### lines 6518-6519  _(unsure)_

```python
from PySide6.QtCore import QTimer as _TimingQTimer
```

Unlike the mark above, this callback can run only after exec() has begun dispatching.  Readiness probes refuse to report before it arrives.

### lines 6525-6528

```python
try:
```

A RETURN FROM `exec` IS A CLEAN SHUTDOWN, whatever the exit code: the event loop ran and ended, which a crash never does. Clearing the count here rather than on code == 0 means a run the user quit from an error dialog still counts as "it started fine".

### lines 5573-5575

```python
threading.Thread(target=_prewarm, name="spacr-prewarm",
```

No pre-warm thread in safe mode (296: "no preloading, no background import thread"). It imports spacr.settings, settings_model and imagery off the GUI thread while the window is being built, which is concurrent start-up work that a safe start leaves out. The cost is the one the thread was added to save: the first Preferences open in a safe session pays for those imports on the GUI thread.

## launch._prewarm

### lines 6443-6447

```python
def _prewarm():
```

Pre-warm the heavy imports that a module screen needs (spacr.gui_utils pulls torch + cv2 ≈ 3-4 s; spacr.settings ≈ 1 s) in a BACKGROUND thread while the user looks at the home screen. By the time they open a module these are cached, so the module snaps open instead of freezing on the first import. Importing modules (no Qt objects) off-thread is safe.

### lines 6459-6466

```python
for mod in ("spacr.settings",
```

`spacr.qt.screens.settings_model` and `spacr.qt.imagery` were added 2026-09-09. They are not only a module screen's cost: the FIRST Preferences open in a session was measured at 902 ms against 30 ms for the second, with the animated backdrop frozen for the whole of it, and 474 ms of that 902 was settings_model alone (imagery 83, matplotlib.colors 48). Nobody reaches Preferences before the home screen exists, so this thread always wins the race it needs to win.

## launch._drain_ai

### lines 6476-6479

```python
def _drain_ai():
```

aboutToQuit fires no matter how the app exits (window closed, Ctrl+C, SIGTERM, …). Belt-and-suspenders with MainWindow's closeEvent: ensure every ConsolePanel drains its AI thread before Qt starts destroying widgets.

### lines 6487-6494

```python
try:
```

EVERY JOB RUNNER, NOT ONLY THE CONSOLES. Qt aborts the process if a running QThread is destroyed, and each runner's own `closeEvent` covers a widget being CLOSED -- not the application quitting with a job in flight, where the widget is destroyed without ever closing. Measured 2026-08-19: spaCR died immediately after every successful regression, "run closed [success]" the last line in the log and nothing after it, because the Runs tab's announce had just started a results read on a worker.

### line 6507  _(unsure)_

```python
try:
```

Also kill any subprocess still tracked by a provider
