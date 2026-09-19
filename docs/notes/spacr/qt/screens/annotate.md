# Notes from `spacr/qt/screens/annotate.py`

Prose lifted out of `spacr/qt/screens/annotate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (12 entries)
- [_vouch_later](#_vouch_later) (1 entry)
- [_ask_about_the_folder](#_ask_about_the_folder) (2 entries)
- [_qt_code_tokens](#_qt_code_tokens) (1 entry)
- [key_token](#key_token) (2 entries)
- [_PageLoadWorker](#_pageloadworker) (1 entry)
- [_PageLoadWorker.__init__](#_pageloadworker__init__) (1 entry)
- [_PageLoadWorker.run](#_pageloadworkerrun) (2 entries)
- [_RetrainWorker](#_retrainworker) (1 entry)
- [_RetrainWorker.run](#_retrainworkerrun) (3 entries)
- [_SuggestWorker](#_suggestworker) (1 entry)
- [_SuggestWorker.run](#_suggestworkerrun) (4 entries)
- [_TextReportDialog.__init__](#_textreportdialog__init__) (1 entry)
- [_Thumbnail](#_thumbnail) (1 entry)
- [_Thumbnail.__init__](#_thumbnail__init__) (2 entries)
- [_Thumbnail.set_current](#_thumbnailset_current) (1 entry)
- [_Thumbnail._stroke](#_thumbnail_stroke) (1 entry)
- [_ZoomOverlay](#_zoomoverlay) (1 entry)
- [_ZoomOverlay.paintEvent](#_zoomoverlaypaintevent) (1 entry)
- [_reanchor_png_path](#_reanchor_png_path) (2 entries)
- [_load_thumb_image_worker](#_load_thumb_image_worker) (2 entries)
- [_compute_total](#_compute_total) (3 entries)
- [_SettingsDialog.__init__](#_settingsdialog__init__) (15 entries)
- [_SettingsDialog._probe_the_source_field](#_settingsdialog_probe_the_source_field) (1 entry)
- [_SettingsDialog._load_the_example_data](#_settingsdialog_load_the_example_data) (1 entry)
- [_SettingsDialog._load_the_streaming_example](#_settingsdialog_load_the_streaming_example) (1 entry)
- [_SettingsDialog._use_the_example_data](#_settingsdialog_use_the_example_data) (2 entries)
- [_SettingsDialog._pick_src](#_settingsdialog_pick_src) (1 entry)
- [_SettingsDialog.collect](#_settingsdialogcollect) (3 entries)
- [_GenerateAnnotationDatabaseDialog._on_generate](#_generateannotationdatabasedialog_on_generate) (1 entry)
- [AnnotateScreen](#annotatescreen) (1 entry)
- [AnnotateScreen.__init__](#annotatescreen__init__) (17 entries)
- [AnnotateScreen._apply_suggested_source](#annotatescreen_apply_suggested_source) (2 entries)
- [AnnotateScreen._follow_path_probes.landed](#annotatescreen_follow_path_probeslanded) (2 entries)
- [AnnotateScreen._follow_path_probes](#annotatescreen_follow_path_probes) (1 entry)
- [AnnotateScreen._follow_path_probes.let_go](#annotatescreen_follow_path_probeslet_go) (1 entry)
- [AnnotateScreen._install_folds](#annotatescreen_install_folds) (2 entries)
- [AnnotateScreen._build_ui](#annotatescreen_build_ui) (24 entries)
- [AnnotateScreen._build_ui._append_error_and_offer_the_report](#annotatescreen_build_ui_append_error_and_offer_the_report) (1 entry)
- [AnnotateScreen._choose_the_test_data](#annotatescreen_choose_the_test_data) (1 entry)
- [AnnotateScreen._use_the_test_data](#annotatescreen_use_the_test_data) (2 entries)
- [AnnotateScreen._on_file_issue](#annotatescreen_on_file_issue) (1 entry)
- [AnnotateScreen._wanted_provider](#annotatescreen_wanted_provider) (1 entry)
- [AnnotateScreen._build_key_legend](#annotatescreen_build_key_legend) (3 entries)
- [AnnotateScreen._install_shortcuts](#annotatescreen_install_shortcuts) (1 entry)
- [AnnotateScreen._compute_grid_dims](#annotatescreen_compute_grid_dims) (1 entry)
- [AnnotateScreen._rebuild_grid](#annotatescreen_rebuild_grid) (5 entries)
- [AnnotateScreen._open_source](#annotatescreen_open_source) (7 entries)
- [AnnotateScreen._on_open_settings](#annotatescreen_on_open_settings) (2 entries)
- [AnnotateScreen._on_class_counts](#annotatescreen_on_class_counts) (2 entries)
- [AnnotateScreen._label_source](#annotatescreen_label_source) (1 entry)
- [AnnotateScreen._refresh_round_state](#annotatescreen_refresh_round_state) (1 entry)
- [AnnotateScreen._show_report](#annotatescreen_show_report) (1 entry)
- [AnnotateScreen._on_retrain](#annotatescreen_on_retrain) (1 entry)
- [AnnotateScreen._on_retrain_done](#annotatescreen_on_retrain_done) (2 entries)
- [AnnotateScreen._on_suggest_menu](#annotatescreen_on_suggest_menu) (1 entry)
- [AnnotateScreen._build_suggest_menu](#annotatescreen_build_suggest_menu) (1 entry)
- [AnnotateScreen._start_suggest](#annotatescreen_start_suggest) (4 entries)
- [AnnotateScreen._on_suggest_done](#annotatescreen_on_suggest_done) (2 entries)
- [AnnotateScreen._on_suggest_finished](#annotatescreen_on_suggest_finished) (1 entry)
- [AnnotateScreen._resolve_suggestions](#annotatescreen_resolve_suggestions) (2 entries)
- [AnnotateScreen._on_train_cv](#annotatescreen_on_train_cv) (1 entry)
- [AnnotateScreen._apply_bulk_annotation](#annotatescreen_apply_bulk_annotation) (1 entry)
- [AnnotateScreen._on_generate_annotation_db](#annotatescreen_on_generate_annotation_db) (1 entry)
- [AnnotateScreen._fold_zoom_back](#annotatescreen_fold_zoom_back) (1 entry)
- [AnnotateScreen._fit_zoom_overlay](#annotatescreen_fit_zoom_overlay) (1 entry)
- [AnnotateScreen._refresh_total](#annotatescreen_refresh_total) (3 entries)
- [AnnotateScreen._load_page](#annotatescreen_load_page) (7 entries)
- [AnnotateScreen._queue_page_load](#annotatescreen_queue_page_load) (1 entry)
- [AnnotateScreen._on_page_loaded](#annotatescreen_on_page_loaded) (2 entries)
- [AnnotateScreen._crop_source](#annotatescreen_crop_source) (1 entry)
- [AnnotateScreen._slot_is_valid](#annotatescreen_slot_is_valid) (1 entry)
- [AnnotateScreen._toggle_annotation](#annotatescreen_toggle_annotation) (1 entry)
- [AnnotateScreen._set_focus_slot](#annotatescreen_set_focus_slot) (2 entries)
- [AnnotateScreen._set_hover_slot](#annotatescreen_set_hover_slot) (1 entry)
- [AnnotateScreen.handle_key](#annotatescreenhandle_key) (2 entries)
- [AnnotateScreen._advance_after_assign](#annotatescreen_advance_after_assign) (1 entry)
- [AnnotateScreen._kbd_undo](#annotatescreen_kbd_undo) (1 entry)
- [AnnotateScreen._kbd_commit_page](#annotatescreen_kbd_commit_page) (2 entries)
- [AnnotateScreen.eventFilter](#annotatescreeneventfilter) (4 entries)
- [AnnotateScreen._detach_event_filters](#annotatescreen_detach_event_filters) (1 entry)
- [AnnotateScreen._band_event](#annotatescreen_band_event) (1 entry)
- [AnnotateScreen._apply_band](#annotatescreen_apply_band) (1 entry)
- [AnnotateScreen.closeEvent](#annotatescreencloseevent) (9 entries)

## Module level

### lines 61-76

```python
from PySide6.QtCore import (
```

PYSIDE6 BEFORE PIL, AND THE ORDER IS LOAD-BEARING. `PIL.ImageQt` resolves a Qt6 of its own at import time, and when it wins the race PySide6 then fails to load against it:

from PIL.ImageQt import ImageQt        # first from PySide6.QtCore import Qt          # ImportError: undefined symbol ZN14QObjectPrivateC2E16QtPrivate_6_11_2

Reversing the two makes it import cleanly, verified both ways in the `spacr` environment on 2026-09-04 (PySide6 6.11.2).

The running application never hit this, because `spacr.qt` has already imported PySide6 long before it reaches this screen -- which is exactly what made it invisible. What it broke was importing this module on its own: every test and tool that does so failed at the import line, and the failure names a Qt symbol rather than anything about ordering.

### lines 195-236

```python
VOUCH_TTL_S = 120.0
```

Two different questions about a folder, and why one answer cannot serve both

THE CHEAP QUESTION -- "shall I name this folder in the subtitle?" -- is what `spacr.qt.path_probe` was written for, and `path_probe.isdir` is the right way to ask it from the GUI thread: it answers from a cache and does the stat on its own thread. Naming a folder that turns out to be gone costs a label.

THE EXPENSIVE QUESTION -- "shall I open `QFileDialog` in this folder?" looks identical and is not, because the dialog STATS AND THEN LISTS its starting directory ON THE GUI THREAD. Point it at a folder on a sleeping `autofs` mount and the window locks for as long as the automount takes: twenty seconds and still counting, measured on the maintainer's machine 2026-09-04. So this question may only be answered "yes" when a real stat has come back and said so.

`path_probe` CANNOT ANSWER THE SECOND ONE, by design. `path_probe._stat_with_timeout` stops WAITING after `PROBE_TIMEOUT_S` and reports the path as PRESENT, because for the question it was written for a path drawn red on the strength of a slow mount is worse than one drawn black. Its cache therefore holds two kinds of True -- one a stat returned and one the timeout invented -- and nothing in it tells them apart. Gating the file dialog on that cache does not remove the freeze, it postpones it by the length of the timeout.

Timing the `probes.answered` emission does not recover the difference either, and the first pass at this screen tried: the cache is keyed on the path, `spacr.qt.chaining.ChainingBar.search_roots` probes `prefs.get_last_source("annotate")` -- exactly this screen's remembered source -- and `path_probe.exists` does not re-queue a key that is already in flight. So the answer this screen timed was routinely somebody else's probe, started at a moment this module never saw, and a stat that never returned was clocked at whatever was left of ITS five seconds and vouched for.

So the expensive question is asked outright, off the GUI thread, and ONLY WHAT A STAT ACTUALLY RETURNED IS KEPT. That is the rule `spacr.qt.dnd_handlers._decide` already states for the same reason ("the cache holds real answers only"), and the bounded off-thread read is the same shape as `spacr.qt.resource_cleanup._readings_within_the_budget`. Nothing below ever runs a stat on the calling thread; the GUI thread only ever reads a dict.

### lines 424-432

```python
GRID_OBJECT_NAME = "AnnotateGrid"
```

Screen chrome that must follow the theme

Both blocks below were widget-local ``setStyleSheet`` calls with raw hex in them, which is the one thing a per-widget sheet cannot do well: it beats the application sheet whatever the selector says, so it never picked up a theme change and never carried the user's page opacity. Registered blocks are re-composed from the live palette on every stylesheet build instead.

### lines 504-520

```python
BORDER_WIDTH = 2          # state ring — the thin line around every crop
```

Tile chrome

Every crop is a rounded square drawn by `_Thumbnail.paintEvent` as three concentric pieces:

┌── current ring  (white) — ONLY on the tile the next action hits │ ┌── state ring          — resting gray, or the crop's class colour │ │ ┌── the crop itself, CLIPPED to a rounded rect (a real round │ │ │   corner, not a rounded frame laid over a square image)

The two rings sit at fixed insets, so nothing moves or resizes when the cursor arrives: hover ADDS the outer ring, it never recolours the inner one, and a class colour never hides the fact that a tile is the current one. That is the whole composition rule — the two states are drawn in two different bands and cannot overwrite each other.

### line 522, trailing  _(unsure)_

```python
BORDER_WIDTH = 2
```

state ring — the thin line around every crop

### line 523, trailing  _(unsure)_

```python
HOVER_RING_WIDTH = 3
```

current-tile ring, drawn outside the state ring

### line 524, trailing  _(unsure)_

```python
TILE_INSET = HOVER_RING_WIDTH + BORDER_WIDTH
```

chrome per side, in px

### line 525, trailing  _(unsure)_

```python
TILE_RADIUS = 10
```

outer corner radius of the rounded square

### lines 528-529

```python
UNDO_LIMIT = 128
```

How many keyboard assignments can be walked back with `u`. Bounded so a long session can't grow the stack without limit.

### lines 588-595

```python
_TEXT_TOKENS = {
```

Keyboard tokens

`handle_key` is the single entry point for every keystroke so tests can drive the whole feature without synthesising Qt key events. It accepts a Qt key code, a Qt key *name* ("Left"), or a literal character ("1", "h"), and normalises all of them onto the small token vocabulary below.

### lines 597-598

```python
_TEXT_TOKENS = {
```

canonical tokens: "0".."9", "left", "right", "up", "down", "space", "backspace", "undo", "enter", "help", "escape"

### line 602  _(unsure)_

```python
"h": "left", "j": "down", "k": "up", "l": "right",
```

vi-style motion

## _vouch_later

### lines 317-319

```python
return
```

Nothing is queued behind the cap on purpose: the caller loses a head start, not a result, and a queue here would be a second backlog to reason about beside `path_probe`'s own.

## _ask_about_the_folder

### line 400, trailing  _(unsure)_

```python
_probe_isdir(text)
```

the shared cache, for everybody's subtitles

### line 401, trailing  _(unsure)_

```python
_vouch_later(text)
```

this screen's own, for its file dialogs

## _qt_code_tokens

### line 633

```python
continue
```

A binding whose enum will not convert costs that key, not the map.

## key_token

### line 674, trailing  _(unsure)_

```python
if 0x30 <= code <= 0x39:
```

Qt.Key_0 .. Qt.Key_9

### line 676, trailing  _(unsure)_

```python
if 0x41 <= code <= 0x5A:
```

Qt.Key_A .. Qt.Key_Z

## _PageLoadWorker

### line 697, trailing  _(unsure)_

```python
done = Signal(int, object)
```

(gen, list[(PIL.Image, annotation)])

## _PageLoadWorker.__init__

### lines 715-717

```python
try:
```

Whether this loader can be told to give up. The page loader can; a simpler per-row callable need not, and asking it once here keeps that decision out of the per-crop loop.

## _PageLoadWorker.run

### lines 770-772

```python
return
```

The page was abandoned mid-crop. Return WITHOUT emitting: the partial list describes a page the screen has already moved off, and the point of unwinding early was to let the thread end.

### lines 776-788

```python
try:
```

THE EMIT IS INSIDE THE GUARD, AND THAT IS THE WHOLE POINT.

`emit` and `isInterruptionRequested` are calls into this worker's C++ half, and by the time a page finishes decoding the screen may already be gone -- Qt destroys the C++ object with its parent while this thread is still in PIL. Both then raise `RuntimeError: Internal C++ object already deleted`, and raised HERE, outside any try, the exception escapes a QThread::run override: PySide6 prints "Error calling Python override of QThread::run()" and the process aborts. Caught in the full suite on 2026-08-19, mid-`Image.resize`.

Nothing is lost by swallowing it. The only thing this branch does is hand results to a screen that no longer exists.

## _RetrainWorker

### line 810, trailing  _(unsure)_

```python
done = Signal(object)
```

RoundResult

## _RetrainWorker.run

### line 844, trailing

```python
except Exception as exc:
```

surfaced, never eaten

### line 848, trailing  _(unsure)_

```python
pass
```

the screen went first; see run() above

### lines 850-852

```python
try:
```

Guarded for the reason `_PageLoadWorker.run` sets out at length: a signal emitted at a destroyed C++ object raises out of run(), and an exception out of a QThread::run override aborts the process.

## _SuggestWorker

### line 884, trailing  _(unsure)_

```python
done = Signal(object)
```

(Suggestions, written: int)

## _SuggestWorker.run

### lines 926-933

```python
resolve_suggestions(self._db_path, self._column, keep=False,
```

CLEARED BEFORE THE FIT, and this is not housekeeping. `retrain_round` takes every non-null value in the column as a class label, and `_class_value(11)` is 11 -- so a second Suggest run would fit on the FIRST run's output as two extra classes and feed the model its own opinion. It is also 379's stated rule ("re-running SUGGEST replaces the outstanding suggestions rather than adding to them"), so the two answers agree: nothing a machine proposed is ever trained on.

### lines 941-946

```python
frame = frame[frame["png_path"].isin(set(self._only))]
```

The SCOPE, applied to the proposal rather than to the fit. The model is fitted on every label either way -- narrowing the training set to one page would make a worse model to save no time at all. What "this page" narrows is which crops get written, which is the only part the reviewer has to live with.

### line 956, trailing

```python
except Exception as exc:
```

surfaced, never eaten

### line 960, trailing  _(unsure)_

```python
pass
```

the screen went first; see run() above

## _TextReportDialog.__init__

### lines 1021-1022  _(unsure)_

```python
self.setWindowFlag(Qt.Window, True)
```

A window in its own right, not a sheet stuck to the screen: it is read alongside the grid, moved, and kept open while annotating.

## _Thumbnail

### lines 1076-1078

```python
hover_changed = Signal(int, bool)
```

(slot, entered). Emitted on Enter/Leave only — never per mouse-move — so tracking the cursor across the grid costs two repaints per tile boundary crossed and nothing at all in between.

## _Thumbnail.__init__

### lines 1099-1100

```python
self._border_color = border_color or resting_border_color()
```

Colours are resolved by the screen once per grid rebuild and handed down, so the hover path never has to look up a palette.

### lines 1112-1113  _(unsure)_

```python
self.setStyleSheet("background: transparent;")
```

Transparent so the rounded tile sits cleanly on the grid canvas (no grey square peeking out at the corners).

## _Thumbnail.set_current

### lines 1150-1151  _(unsure)_

```python
self.setProperty("kbdFocused", on)
```

Mirrored onto a Qt property so QSS and tests can both see it, and so there is exactly one notion of "the current tile".

## _Thumbnail._stroke

### lines 1236-1237  _(unsure)_

```python
pen.setStyle(Qt.CustomDashLine)
```

In units of the pen width, so the dashes keep their proportions when the interface is zoomed -- see `spacr.qt.live_zoom`.

## _ZoomOverlay

### lines 1274-1276  _(unsure)_

```python
class _ZoomOverlay(QWidget):
```

The one crop the annotator wanted to look at properly

## _ZoomOverlay.paintEvent

### lines 1345-1346  _(unsure)_

```python
painter.fillRect(self.rect(), QColor(0, 0, 0, 170))
```

A scrim, not a blank: the grid stays legible behind the crop so it is obvious that nothing was navigated away from.

## _reanchor_png_path

### line 1469, trailing  _(unsure)_

```python
cand = os.path.join(root, norm[i + 1:])
```

data/.../x.png

### line 1472, trailing  _(unsure)_

```python
if norm.startswith("data/"):
```

relative-path case

## _load_thumb_image_worker

### lines 1522-1523  _(unsure)_

```python
full_img = img
```

Keep the full image for outline detection even when the display filter hides one of its channels.

### lines 1543-1545

```python
raise
```

Re-raised ahead of the blanket handler below. Swallowing it here would turn "the screen has gone, stop" back into "draw this crop without an outline" and carry on with the page.

## _compute_total

### lines 1573-1576

```python
from ... import active_learning as al
```

Order the unlabelled crops by how unsure the model is about them, so the annotator spends their time on the decision boundary. The queue is a snapshot, rebuilt on every settings apply, so crops labelled since the last rebuild drop out then rather than now.

### lines 1586-1588

```python
return {"filtered_rows": None,
```

No model scores yet is the ordinary case before a classifier has run, so fall back to page order and say why rather than showing an empty grid.

### line 1597  _(unsure)_

```python
rows = fetch_filtered_paths(
```

Cache the filtered set once so pagination + total agree

## _SettingsDialog.__init__

### lines 1628-1630

```python
from ..dialogs import detach_from_window_manager
```

A modal, transient-for dialog is ATTACHED by GNOME/Mutter: centred on the parent, undraggable, and pulling at it un-maximises the main window. See spacr.qt.dialogs.

### lines 1642-1648

```python
if settings.src:
```

WARM THE ANSWER; do not ask the filesystem from here. `Browse…` hands its starting folder to `QFileDialog`, which stats AND LISTS that folder on the GUI thread -- twenty seconds on a sleeping `autofs` share, measured 2026-09-04 -- so `_pick_src` only offers a folder a real stat has come back and confirmed. This queues that check now, on a thread of its own, so the answer is in by the time anybody reaches the button.

### lines 1651-1656

```python
self._src_edit.editingFinished.connect(self._probe_the_source_field)
```

AND AGAIN WHENEVER THE FIELD IS EDITED. `editingFinished` fires on focus-out, which for a mouse is the PRESS on Browse -- a moment before its `clicked` -- so a hand-typed local folder is normally answered in time as well, while a sleeping one is not. That asymmetry is the whole point: the picker keeps opening where the user pointed it, except when doing so would freeze the window.

### lines 1664-1681

```python
src_wrap = QWidget(); src_wrap.setLayout(src_row)
```

SOMETHING TO ANNOTATE, for a user who has not measured a plate yet. Beside Browse because it fills the same field, and because the question it answers -- "what do I point this at?" -- is asked here. TWO STRATEGIES, TWO BUTTONS. Annotating can read crops that are already on disk, or cut them from the merged arrays on demand, and the two need different halves of a plate:

crops     -> data/ and measurements/measurements.db  (282 MB) streaming -> merged/ and measurements/measurements.db (388 MB)

Both unpack into the same plate folder, so pressing both leaves a complete plate and either strategy then works. One button fetching 670 MB would make the cheaper half unavailable on its own. THE TWO EXAMPLE BUTTONS MOVED. They were here, beside the source box, spending two slots on a choice most users make once -- and naming the choice ("crops" vs "streaming") before explaining it. They are now one "Load test data" button next to Generate, which opens a chooser that can afford to describe each route properly.

### lines 1687-1689

```python
attach_column_picker(self._ann_col, self._picker_db_path, "png_list",
```

"SQL" — show what png_list already holds, so a mistyped name cannot quietly start a second annotation pass that then looks like a second annotator who agrees with nobody. Opens read-only.

### lines 1698-1708

```python
from ...crops import (LOAD_IMAGES, LOAD_IMAGES_LABEL, STREAM_IMAGES,
```

WHERE THE PICTURE COMES FROM, offered rather than inferred. The setting has always existed and was never asked about: it shipped 'auto', which takes the exported crops whenever a `data/` folder is there, so a screen holding both folders could not be told to read the arrays without editing a settings file.

The two modes are named the same way every other spaCR panel names them, and the STORED values stay 'png' and 'merged' -- no settings file written before this changes meaning. 'auto' is retired from the panel and not from the code: it answers "what is available here", which is not an answer to which mode a user wants.

### lines 1724-1726

```python
self._crop_source.setCurrentIndex(
```

EVERY SPELLING THIS SETTING HAS EVER CARRIED resolves here, so a settings file written under any of them opens on the mode it named rather than on whatever happened to be last in the list.

### lines 1757-1760

```python
from ...crops import DISPLAY_ORDERS
```

Deliberately the NEXT row, and worded to draw the distinction the one above it is about: that control says how the file was written, this one says how you want to look at it. Six orders, identity first, so the default is a no-op.

### lines 1795-1798

```python
current_primaries = str(
```

Unset means "whatever this user needs", not "RGB". The global colour-vision preference is the default, so somebody who told Preferences once that they are colour-blind finds Annotate already correct; choosing a mode here still overrides it for this session.

### lines 1825-1829

```python
self._pct_lo = QDoubleSpinBox()
```

Six decimals, set BEFORE the range and the value, matching

`percentile_pair.DECIMALS`. `settings.percentiles` can already hold 99.9999 -- it is a plain float on disk -- so the two-decimal default these carried rounded a stored value on the way IN, and the annotator then wrote the rounded one back on the next save.

### lines 1884-1895

```python
self._object_filter_fields: Dict[
```

── Which objects get an outline: six rows of two fields ─────────

One number for every colour was the complaint. Red, green and blue hold different objects, so each plane gets its own size window and its own brightness window -- six rows of two fields rather than twelve separate settings, which is the same information without a form nobody can read.

A LEGACY `object_size` IS SHOWN IN THESE FIELDS, migrated onto the three area rows by the engine, so the value a project was already filtering on is in front of the user rather than silently still in force somewhere they cannot see.

### lines 1910-1912

```python
validator = QDoubleValidator(edit)
```

A number or nothing. `filter_bound` treats anything else as no bound, and a filter that switched itself off because of a half-typed number would be silent.

### line 1927  _(unsure)_

```python
self._measurement = QLineEdit(
```

── Threshold filter (measurement > / < threshold on merged tables)

### lines 2011-2015

```python
self._norm_channels: "normalize_channels",
```

`_display_order` is deliberately NOT here. This map installs the API tooltip for a pipeline SETTING, and `display_order` is a view preference that no pipeline function takes -- listing it replaced the explanatory tooltip written above with an empty one, which is worse than having no entry at all.

### lines 2026-2030

```python
self._measurement: "measurement",
```

The twelve filter fields are deliberately NOT here. This map installs the API tooltip for a pipeline SETTING, and no pipeline function takes a per-colour window; pointing one of them at `object_max_size` would replace the sentence written above with a description of a different, single-number knob.

## _SettingsDialog._probe_the_source_field

### lines 2040-2055

```python
def _probe_the_source_field(self) -> None:
```

POLISHED BY `spacr.qt.dialogs.make_the_window_resizable`, which does this for EVERY dialog now and not only for this one.

The defect is worth keeping a note of here because this dialog is where it was measured: the detacher reads a dialog's floor on its Polish event, an event filter runs before the widget's own handler, and so the floor used to be measured before the stylesheet reached any of this dialog's 165 children. Sixteen of them change size across that boundary -- its eight QComboBoxes, 29 px in the default "Sans Serif 9" and 30 px in the stylesheet's "Open Sans".

floor read at Polish   480 x 1183   the size it re-opened at floor once on screen   512 x 1191   the size it really needs

Eight rows, eight pixels, and this dialog opened eight pixels short of its own content with a scroll bar already showing.

## _SettingsDialog._load_the_example_data

### lines 2103-2105

```python
if (destination / "measurements" / "measurements.db").is_file():
```

THIS SET'S OWN DATABASE is the test. The folder is shared with the other example sets now, so its existence says nothing -- and a cancelled download leaves it behind too.

## _SettingsDialog._load_the_streaming_example

### lines 2139-2140  _(unsure)_

```python
merged = destination / "merged"
```

`merged/` holding an array is the test, not the folder: it is shared with the crops download and with Mask's images.

## _SettingsDialog._use_the_example_data

### lines 2178-2182

```python
self._apply_example_settings(destination / "settings"
```

AND THE SETTINGS THAT CAME WITH IT. The dataset ships an

`annotate_settings.csv` describing which column holds the labels, what size the crops are and which channels they carry -- and a user who has to work that out first has done most of the work the example was meant to save.

### lines 2185-2189

```python
if hasattr(self, "_ann_col") and not self._ann_col.text().strip():
```

`infected`, NOT `annotate`, when the file did not say. The published set is labelled by a rule -- a cell is infected exactly when the pathogen table names it as a parent -- and `annotate` is deliberately empty, so opening on it would show 2,341 unlabelled crops and none of the labels the example exists to carry.

## _SettingsDialog._pick_src

### lines 2304-2309

```python
_ask_about_the_folder(d)
```

QUEUE the folder's answers now, while the dialog has just listed it and the mount is demonstrably awake, so the next press of Browse can start here. Queued, not had: this returns immediately and both stats happen on other threads. `path_probe.prime` is not what to call -- it records the `exists` question, and neither question here is that one.

## _SettingsDialog.collect

### lines 2359-2362

```python
s.display_order = str(self._display_order.currentData() or "rgb")
```

BOTH VIEW CONTROLS ARE READ BACK HERE. Neither used to be, so the two combos above were decorative: a crop drawn after choosing CMY measured pixel-for-pixel identical to the RGB one, because the settings object the loader reads still said "rgb".

### lines 2380-2383

```python
s.object_size = (0, 0)
```

The single window the twelve fields replaced. Zeroed once they have been written, or the migration would run again next time and put an old bound back into a row the user has just emptied -- which is exactly the case "empty means no bound" exists for.

### line 2385  _(unsure)_

```python
meas_txt = self._measurement.text().strip()
```

Threshold filter

## _GenerateAnnotationDatabaseDialog._on_generate

### lines 2553-2554

```python
self._status.setText(
```

NOT SILENT. A generator that writes nothing and closes looks exactly like one that worked.

## AnnotateScreen

### lines 2837-2838  _(unsure)_

```python
train_requested = Signal(str, dict)
```

Emitted with (target_app_key, seed_settings_dict); MainWindow picks this up to switch to that screen and preseed values.

## AnnotateScreen.__init__

### lines 2847-2849

```python
from ..theme import ensure_widget_qss_applied
```

This module is imported lazily, which normally means minutes after the only stylesheet that would have carried its blocks was built. A no-op when they are already in it.

### lines 2860-2863

```python
self._round_index = 0
```

── Active learning: the loop's state on this screen ─────────────── The round the next batch of labels belongs to. 0 until a source is opened; bumped by every retrain, so a label always records which model's ranking put it in front of the annotator.

### line 2876, trailing  _(unsure)_

```python
self._last_round = None
```

spacr.active_learning.RoundResult

### line 2877, trailing  _(unsure)_

```python
self._stop_verdict = None
```

spacr.active_learning.StoppingVerdict

### lines 2878-2881

```python
self._object_request = None
```

A routed ObjectRequest currently pinning the grid to a subset, and the rows it resolved to. Held separately from `_filtered_rows` because a filter/queue rebuild must not silently wipe a subset the user was sent here to look at.

### lines 2886-2890

```python
self._object_opener = self.open_object_request
```

ONE bound method, kept, so register/unregister pass the *same* object. `self.open_object_request` builds a fresh bound method on every attribute access, and LinkedSelection.unregister_object_opener is identity-checked — passing a freshly-built one withdraws nothing and the process-wide registry keeps a reference to a closed screen.

### lines 2901-2908

```python
self._settings_dialog: Optional[_SettingsDialog] = None
```

``_SettingsDialog`` contains signal/bound-method cycles.  A local variable alone does not own it after ``exec()`` returns, and cyclic GC may legally run in whichever Python thread crosses its threshold. Coverage changed that timing enough for the page QThread to collect the dialog's timer-bearing QWidget tree: Qt warned that QBasicTimer was stopped from the wrong thread and then segfaulted in the GUI dispatcher's stale timer event. Keep the Python wrapper here until Qt's GUI-thread DeferredDelete has destroyed the C++ object.

### lines 2910-2913

```python
self._total_jobs = JobRunner(self, app_key="annotate count")
```

Counting the population is database work, not widget work — see `_refresh_total`. Its own runner, separate from the page loader, because a settings apply cancels the count without disturbing the crops already on screen.

### lines 2915-2917

```python
self._resize_timer = QTimer(self)
```

A drag-resize used to launch one QThread (and one inner thread pool) per geometry event.  Debounce it and keep only the newest page request so native image/model code never overlaps with itself.

### lines 2924-2928

```python
self._focus_slot = 0
```

── The current tile ─────────────────────────────────────────────── ONE notion, shared by mouse and keyboard: `_focus_slot` is the crop the next action hits and the only tile that wears the white ring. The cursor entering a tile moves it; an arrow key moves it. There is deliberately no second "hovered tile" that could disagree.

### lines 2930-2933

```python
self._hover_slot: Optional[int] = None
```

Bookkeeping only: which tile the cursor is inside right now, or None when it is between tiles / outside the grid. Whenever it is set it equals `_focus_slot` (see `_set_hover_slot`), so the white ring never has two candidates.

### lines 2935-2937

```python
self._band = None
```

(slot, png_path, previous_value) for `u`. Bounded — a long session must not grow this without limit. Cleared on every page load since slot indices change meaning.

### lines 2948-2949  _(unsure)_

```python
self.setFocusPolicy(Qt.StrongFocus)
```

The screen itself owns keystrokes; the thumbnails are NoFocus QLabels so nothing inside the grid competes for them.

### lines 2952-2953  _(unsure)_

```python
try:
```

Drag & drop — accepts a plate folder with measurements/measurements.db (or the .db file directly).

### lines 2966-2971

```python
register_object_opener("annotate", self._object_opener)
```

Half of the object-routing contract in

`spacr.qt.linked_selection`: a scatter point and a confusion-matrix cell both want "show me exactly these crops", and neither should have to know this class exists. Withdrawn in closeEvent, passing the bound method so a second Annotate opened later keeps the registration when this one closes.

### lines 2974-2992

```python
self._follow_path_probes()
```

The remembered source is a path the USER supplied, and this used to be a bare `os.path.isdir` on it, here, in __init__ -- i.e. on the GUI thread, with the screen not yet on screen. Measured on the maintainer's machine 2026-09-04: one stat on a path under `/nas_mnt` (an autofs mount whose share was asleep) had NOT RETURNED AFTER TWENTY SECONDS. Opening Annotate after a session that last worked on the NAS therefore froze the whole application before it drew anything, with no traceback, because a stalled event loop is not a crash. The subtitle is a hint; it is not worth a single millisecond of the interface.

SUBSCRIBE FIRST, THEN ASK, and not the other way round: the probe is QUEUED by `_apply_suggested_source`, and a worker that finishes quickly emits `answered` from its own thread while `__init__` is still running. A Qt signal emitted with nothing connected to it is dropped, not buffered, so asking first would lose the only answer this path is ever given and leave the suggestion missing for the life of the screen -- the "pessimistic gate with nothing to recover it" that `path_probe.isdir` always needs a subscriber for.

### lines 2995-3002

```python
if self._suggested_source:
```

AND THE OTHER QUESTION about the same folder, which nothing above asks. `_apply_suggested_source` gates on the shared probe, which is right for a subtitle and not good enough for a file dialog; `_starting_folder` reads `_vouched_dir` instead, and if the harder question is never PUT then the answer is never there and the picker never starts in the folder the user was last working in. Queued here, at construction, so it has landed long before anybody can press the button.

## AnnotateScreen._apply_suggested_source

### lines 3031-3033

```python
return
```

A source has been opened since; `_open_source` owns the subtitle from that point and a late probe must not stamp a stale suggestion over the database the user is looking at.

### lines 3039-3042

```python
self._src_label.setProperty("i18nSkipText", True)
```

The subtitle now carries a filesystem path, which a language switch must reproduce byte-for-byte. Only the placeholder it replaced is prose, so the opt-out belongs here rather than on the label itself.

## AnnotateScreen._follow_path_probes.landed

### lines 3070-3074

```python
return
```

`closeEvent` runs nested event loops while it drains its workers, so a queued emission can still be delivered here after the screen has begun tearing itself down. Nothing on a closing screen wants a new subtitle.

### line 3079  _(unsure)_

```python
pass
```

The screen has gone; the signal outlived it.

## AnnotateScreen._follow_path_probes

### lines 3086-3101

```python
withdrawn: List[bool] = []
```

ONE withdrawal, reachable from two places. `closeEvent` is the ordinary way out, but it is not the only one: a screen can be destroyed without ever being closed -- the stack deletes it, or a test drops its last reference -- and a connection left on a process-wide signal then delivers to a widget whose C++ half is gone. `destroyed` fires while the wrapper is still usable, which is the moment to let go; the same pattern `spacr.qt.chaining.ChainingBar` uses on the same signal.

`withdrawn` is a plain cell rather than an attribute so that the second call is a no-op WITHOUT touching a half-destroyed wrapper: disconnecting an already-disconnected slot is not an exception in PySide, it is a RuntimeWarning printed from C++ that no `except` can catch. The closure holds the SIGNAL for the same reason this can run during interpreter teardown, when the module globals it would otherwise reach through have already been cleared.

## AnnotateScreen._follow_path_probes.let_go

### line 3119, trailing  _(unsure)_

```python
pass
```

the source is gone

## AnnotateScreen._install_folds

### lines 3139-3141

```python
self._fold_page_title = "Annotate"
```

What this screen's own page is called once a folded module puts a page beside it. Named here because this screen builds its own masthead and carries no registry key to be looked up by.

### line 3154

```python
self._fold_openers = openers
```

The openers outlive this call only because the screen holds them.

## AnnotateScreen._build_ui

### lines 3161-3163

```python
"""Lay out the thumbnail grid over the class and navigation rows."""
```

Resolved once here rather than imported at module scope, so the grid canvas and the tile chrome agree with the theme the user is actually running (see `tile_palette`).

### lines 3171-3174

```python
header = QWidget()
```

Header. The title and the source line stack in a column on the left; anything folded into this screen sits right-aligned past the stretch, which is where every other masthead puts its trailing controls (see `ModuleHeader.add_trailing`).

### line 3221, trailing  _(unsure)_

```python
self._btn_next.setLayoutDirection(Qt.RightToLeft)
```

icon on the right

### lines 3261-3265

```python
self._btn_suggest = QPushButton("Suggest…")
```

SUGGEST SITS NEXT TO RETRAIN because it is the same act with a different destination: Retrain re-ranks the queue with the model, Suggest writes the model's opinion down where you can accept it. A menu rather than a dialog for the same reason "Train…" has one the choice is between named things with no further settings.

### lines 3289-3300

```python
self._btn_train = QPushButton("Train…")
```

ONE TRAINING BUTTON, TWO DESTINATIONS.

"Train CV" and "Train XG" sat side by side and read as two features rather than as one decision. They are the same act -- take these annotations and train something on them -- differing only in WHAT the model looks at: the images, or the measured features. A menu says that; two buttons made the user work it out from four-letter abbreviations.

A menu rather than a dialog: the choice is between two named things with no further settings, and a modal for that is a click more than the question is worth.

### lines 3329-3331

```python
self._btn_train_cv = self._btn_train
```

KEPT AS NAMES, not as widgets. Everything that used to enable, disable or click these two buttons still has something to hold, and a caller that flips one now flips the single button they became.

### lines 3352-3357

```python
self._btn_annotate_page = QPushButton("Annotate page")
```

Page-scoped, and next to Clear column on purpose: the three differ only in how much they touch, so they belong where they can be compared. Both of these go through the same _set_annotation / _push_undo path a keystroke does, so Ctrl+Z walks back a whole page one slot at a time -- a bulk action that cannot be undone is worse than no bulk action.

### lines 3379-3383

```python
row.addStretch(1)
```

BUILD A SET TO ANNOTATE, for a user who has measured a plate and has no crops registered -- or who wants a different selection from the one Measure happened to cut. It sits on the right, past the stretch, because it is a module rather than one of the per-page actions to its left.

### lines 3385-3386

```python
self._btn_test_data = QPushButton(tr("Load test data"))
```

TO THE LEFT OF GENERATE, because fetching a set to work on comes before building one from it, and the two are the same kind of action.

### lines 3411-3416

```python
self._al_label = QLabel("")
```

The loop's one-line state, always visible: which round, how many labels, held-out accuracy, the weakest class, and whether the last stretch of labelling bought anything. A learning curve buried behind a button gets looked at once; this is what stops someone labelling a thousand crops after the curve flattened at two hundred.

### line 3426  _(unsure)_

```python
self._content_stack = QStackedWidget()
```

Content stack: empty-state until a source is opened, then grid

### lines 3448-3451

```python
self._grid_scroll.viewport().setAutoFillBackground(False)
```

THE VIEWPORT PAINTS NOTHING. The backdrop is the holder inside it, and the holder's corners are round — a viewport filled with the same grey would sit in those corners as four square nubs and the rounding would not read at all.

### lines 3456-3460

```python
self._grid_layout = QGridLayout(self._grid_holder)
```

The space BETWEEN the images, and the panel behind them: a page surface with the theme's own corner radius, styled by the registered block rather than by a widget-local sheet so it follows both the theme and the page-opacity preference. The images themselves are pixmaps and are untouched by either.

### lines 3466-3473

```python
self._grid_holder.setAutoFillBackground(False)
```

setWidget() turns autoFillBackground ON, and that is what squared the corner off. The auto-fill runs BEFORE the stylesheet's own painter and covers the whole rectangle with the palette's window brush -- which QSS has already propagated the block's `background` into. So the panel came out the right colour, the `border-radius` in the block was painted underneath it, and the backdrop read as a square slab among the rounded cards beside it. Measured at the corner; the same call is made for the viewport just above.

### lines 3475-3477

```python
self._zoom_overlay = _ZoomOverlay(self._grid_scroll.viewport())
```

Shift + left click blows one crop up to fill this container. Built here rather than on demand so it is already a child of the viewport and already above the canvas the tiles are laid out on.

### lines 3480-3481

```python
self._grid_scroll.installEventFilter(self)
```

Without these the scroll area swallows the arrow keys and scrolls instead of moving grid focus.

### lines 3488-3490

```python
self._runtime_splitter = QSplitter(Qt.Vertical, self)
```

The grid and the optional Console + AI pane share a vertical splitter.  Annotate starts grid-first; the bottom controls reveal the console on demand without opening a separate window.

### lines 3499-3503

```python
title_row = QHBoxLayout()
```

THE TITLE IS A ROW, so the two controls a console is actually for can sit on it. This screen builds its own ConsolePanel rather than using the generic module screen's, and had inherited neither -- so the one pane most likely to be holding a traceback was the one pane you could not copy or file from.

### lines 3521-3523

```python
self._btn_file_issue = QPushButton(tr("File as issue"),
```

HIDDEN UNTIL THERE IS SOMETHING TO REPORT, exactly as the module screens do it: a permanently visible "File as issue" invites reports with no traceback attached, which are the ones nobody can act on.

### lines 3543-3548

```python
_original_append_error = self._console.append_error
```

EVERY error path, not a list of them. `append_error` is the single funnel the panel documents -- a WARNING routes through it too -- so wrapping it here reveals "File as issue" whoever raised the problem, including code written after this line. Listing the call sites instead would have missed the pipeline worker, which is the one that reported this.

### lines 3573-3574  _(unsure)_

```python
bottom = QWidget(self)
```

Status and Console/AI controls stay at the bottom, matching the generic module screens.

### lines 3584-3585  _(unsure)_

```python
self._console_switch.setObjectName(CONSOLE_SWITCH_NAME)
```

Named so the registered block above can reach it: white text on the page with no plate behind it, accent-blue while the pane is open.

### lines 3587-3589

```python
self._console_switch.setProperty("i18nSkipText", True)
```

The caption ends in a state arrow, so the generic text pass would translate the composed string and drop it. This screen re-renders the caption itself from `retranslate_dynamic_content`.

### lines 3606-3609

```python
outer.addWidget(bottom)
```

NO PROVIDER CHEVRON. The generic AppScreen used to build the same one; both moved to Preferences → AI, where "which assistant do I use" is answered once instead of on the actions row of every module.

## AnnotateScreen._build_ui._append_error_and_offer_the_report

### lines 3561-3563

```python
self.note_console_error()
```

In a finally: a console that cannot draw is exactly when a user most wants the report button, and a raise here would otherwise swallow the original error.

## AnnotateScreen._choose_the_test_data

### lines 3656-3659

```python
if route == "stream":
```

WHAT COUNTS AS ALREADY HAVING IT differs by route, and the folder itself answers neither: it is shared with the other example sets and a cancelled download leaves it behind. Each route tests for its own half.

## AnnotateScreen._use_the_test_data

### lines 3713-3716

```python
self._settings.crop_source = (
```

STREAM cuts crops out of merged/*.npy as the page is drawn; LOAD reads the ones already exported under data/. Setting this is half of what the route means; leaving it would open the wrong reader on the right data.

### lines 3722-3723  _(unsure)_

```python
refresh = getattr(self, "_refresh_total", None)
```

The page count is what a new source changes first, and it is the number the user reads to know the load worked.

## AnnotateScreen._on_file_issue

### lines 3825-3827

```python
LOG.debug("file_issue signature mismatch", exc_info=True)
```

The reporter's signature is the module screens' business and has changed before. A failure to file must not take the annotation session with it, so it is reported rather than raised.

## AnnotateScreen._wanted_provider

### lines 3885-3887

```python
try:
```

A PREFERENCE IS A WISH, NOT A GUARANTEE. The CLI it names can be uninstalled between sessions, and honouring the name regardless would route every question to something that is not there.

## AnnotateScreen._build_key_legend

### lines 3924-3926

```python
from ..theme import pane_surface
```

`pane_surface`, not `PALETTE['surface']`. `tile_palette()` returns raw hex, so the 1-9 key legend stayed fully opaque whatever the page opacity said — the one strip on the annotate screen that ignored it.

### lines 3945-3947

```python
self._kbd_hint = QLabel("")
```

Transient keyboard feedback ("end of page", "nothing to undo"). Deliberately NOT the shared status label: that one is rewritten every 500 ms by the save-state timer, which would eat the message.

### lines 3957-3960

```python
self._legend_toggle.setMinimumWidth(max(28, self._legend_toggle.sizeHint().width()))
```

A CAP THAT CANNOT CUT (193). The number keeps this control compact; `sizeHint` is the floor, so a larger font or a glyph a theme renders wider grows the button rather than clipping it.

## AnnotateScreen._install_shortcuts

### lines 3983-3985

```python
"""Bind the number keys, the arrows and undo.
```

Bare arrow keys now drive grid focus (see `handle_key`), so page navigation moved to PageUp/PageDown with Alt+Arrow kept as an alias for anyone with the old muscle memory.

## AnnotateScreen._compute_grid_dims

### lines 4011-4012  _(unsure)_

```python
cols = max(1, self._settings.grid_cols or 5)
```

No viewport yet — fall back to previous values (or a sensible default of a 5x5 grid).

## AnnotateScreen._rebuild_grid

### lines 4020-4022

```python
self._fold_zoom_back()
```

A zoomed crop belongs to the grid that is about to be thrown away, and its pixmap would go on being shown over a page of different crops. Fold it back first.

### line 4024  _(unsure)_

```python
self._compute_grid_dims()
```

Recompute page-fit before we create widgets

### lines 4030-4031

```python
self._hover_slot = None
```

Every widget the cursor could have been inside is gone; keeping the index would leave a hover pointing at a tile that no longer exists.

### lines 4041-4042

```python
resting = resting_border_color()
```

One palette lookup for the whole grid — the hover path must not pay for a theme resolution on every mouse move.

### line 4055  _(unsure)_

```python
self._focus_slot = max(0, min(self._focus_slot, len(self._thumbs) - 1))
```

Widgets were just recreated — re-establish the focus marker.

## AnnotateScreen._open_source

### line 4132  _(unsure)_

```python
self._flush_pending()
```

Tear down previous worker

### lines 4139-4145

```python
_ask_about_the_folder(src)
```

Ask about the folder now, while the mount is awake -- the picker (or the settings dialog) has just listed it -- so the next press of "Source…" can start here without `QFileDialog` being the thing that finds out. Both questions, because `_starting_folder` reads the stricter one and the subtitles of other screens read the shared one. It returns immediately; both stats happen off this thread. See `_starting_folder` and `_vouched_dir`.

### line 4154  _(unsure)_

```python
self._src_label.setProperty("i18nSkipText", True)
```

From here on the subtitle is a pair of filesystem paths, not prose.

### lines 4158-4162

```python
self._content_stack.setCurrentWidget(self._grid_scroll)
```

Show the grid page FIRST so its viewport is realized, then defer the grid build + first load to the next event-loop tick. Otherwise _compute_grid_dims measures a zero-size viewport and builds a 5x5 fallback, so the first open showed only a few images that don't fill the view (until the user opened Settings, which rebuilt the grid).

### lines 4164-4165  _(unsure)_

```python
self.setFocus(Qt.OtherFocusReason)
```

Take keyboard focus so the user can start keying classes straight away without first clicking into the grid.

### lines 4167-4168  _(unsure)_

```python
self._object_request = None
```

A new source is a new loop: the round counter, the routed subset and the last verdict all belonged to the previous database.

### lines 4173-4175

```python
self._refresh_total(then=self._rebuild_and_load)
```

`_rebuild_and_load` used to be deferred a turn so the viewport was realized before the grid was sized. The count is now asynchronous, so its delivery is already a later turn and does the same job.

## AnnotateScreen._on_open_settings

### line 4195  _(unsure)_

```python
if (self._settings.src != old_src
```

Restart worker if src/col changed

### lines 4202-4204

```python
try:
```

Never drop the retained wrapper here. ``deleteLater`` is the ownership hand-off to the GUI event loop; ``destroyed`` clears it only after the QWidget tree has been deleted on that thread.

## AnnotateScreen._on_class_counts

### lines 4279-4282

```python
lines = ["Class    Count    Color"]
```

`class_counts` EXCLUDES SUGGESTIONS AT THE SOURCE since 560a34a6b, so these rows are answers and nothing here has to sort them out. This screen briefly folded the offset values back itself, which was right while the query returned them and is dead code now.

### lines 4286-4289

```python
try:
```

STILL REPORTED, because leaving them out entirely answers the question "are my classes balanced" with a number that quietly ignores a few thousand rows in the same column. Counted apart from the classes, which is the distinction that matters.

## AnnotateScreen._label_source

### lines 4304-4307  _(unsure)_

```python
def _label_source(self) -> str:
```

Active learning: retrain here, re-rank, watch the curve, know when to stop. The queue was already wired; this is the half that closes it.

## AnnotateScreen._refresh_round_state

### line 4331

```python
self._round_index = 0
```

Never let a bookkeeping read stop somebody annotating.

## AnnotateScreen._show_report

### lines 4396-4401

```python
for key, report in list(open_reports.items()):
```

Drop the husks first. `WA_DeleteOnClose` means a report the user closed has already lost its C++ half, and asking such a wrapper anything raises RuntimeError. A `destroyed` connection would clear them sooner, at the price of a closure holding this screen inside an object this screen owns; a sweep here costs nothing and keeps the reference graph a tree.

## AnnotateScreen._on_retrain

### lines 4470-4478

```python
self._flush_pending()
```

NO LONGER ASKS ABOUT OUTSTANDING SUGGESTIONS, and the reason is that the trap moved. `retrain_round` filtered them out at the source in 560a34a6b, so a fit with suggestions outstanding is simply correct now -- and a dialog offering to throw a review queue away before a Retrain is a destructive prompt with nothing behind it. The guard was right while the fit was wrong. The labels the annotator just made are the whole point of the round; a retrain that raced the save worker would fit on the state before them.

## AnnotateScreen._on_retrain_done

### lines 4511-4512  _(unsure)_

```python
self._console.append_notice("{report}\n", report=result.summary())
```

Not `text=` — ConsolePanel.append_notice forwards the mapping to `tr(core, **mapping)`, and `text` is tr's own first parameter.

### lines 4518-4520

```python
if self._object_request is None:
```

Re-rank. The round wrote fresh per-class probabilities into png_list, and build_queue prefers them, so this is where round 2 starts showing genuinely different crops.

## AnnotateScreen._on_suggest_menu

### lines 4647-4649  _(unsure)_

```python
def _on_suggest_menu(self):
```

Suggest: the model's opinion, written down where it can be rejected

## AnnotateScreen._build_suggest_menu

### lines 4705-4709

```python
caveat = menu.addAction(
```

WITHHELD, NOT DISABLED-WITH-A-TOOLTIP. There is no safe bulk accept for a model whose negatives were invented, and a greyed-out control invites the user to find out how to enable it. Rejecting in bulk stays: throwing away a ranking costs nothing.

## AnnotateScreen._start_suggest

### lines 4734-4736

```python
self._flush_pending()
```

The labels just made are the ones that most change the model; a run that raced the save worker would fit on the state before them. Same flush `_on_retrain` does, and for the same reason.

### lines 4745-4748

```python
synthetic = self._synthetic_negatives_needed()
```

THE ONE-CLASS CASE, asked for in so many words: "if only one class randomly choose the same number of images as is annotated for the other class". The count is how many the user has actually made, so it is read here rather than guessed in the worker.

### lines 4768-4772

```python
"model_type": "gradient_boosting",
```

BOOSTED TREES, which is what the request asked for by the name XGBoost. `_build_round_model` maps this to sklearn's HistGradientBoostingClassifier -- the same algorithm, already a dependency, and the one the rest of the round machinery (grouped split, model card, saved joblib) already understands.

### lines 4774-4777

```python
"balance": "downsample",
```

THE MAINTAINER'S TWO RULES, now that `retrain_round` can express them (2026-09-07). "if there is class imbalance use the class with fewer" is `balance="downsample"`; the one-class case is handled below, where the count is known.

## AnnotateScreen._on_suggest_done

### lines 4808-4812

```python
self._console.append_notice(
```

IN WORDS, BEFORE ANYTHING CAN BE ACCEPTED IN BULK. Only one class had been annotated, so the negatives this model learned from were drawn at random from the unlabelled pool: they are mostly-negative, not negative. What came back is an ORDER to review in, not a set of answers.

### line 4821  _(unsure)_

```python
self._refresh_total(then=self._load_page)
```

Re-read the page so the dashed rings appear without a navigation.

## AnnotateScreen._on_suggest_finished

### line 4845, trailing  _(unsure)_

```python
return
```

the screen went first

## AnnotateScreen._resolve_suggestions

### lines 4890-4892

```python
self._flush_pending()
```

The pending writes go first: a suggestion the annotator has just answered by hand is no longer a suggestion, and resolving before the save worker had flushed would sweep it up as one.

### line 4898  _(unsure)_

```python
self._suggestions_are_a_ranking = False
```

The ranking is gone, so the caveat that went with it is too.

## AnnotateScreen._on_train_cv

### lines 4918-4925

```python
"dataset_mode": "annotation",
```

nudge the train pipeline into the "annotation → train → apply" mode. dataset_mode is what actually selects the classes: generate_training_dataset alone left it at the Classify panel's default, 'metadata', so "Train CV" from the Annotate app built its classes from well metadata and ignored the annotations that had just been made. It used to die on the way there (KeyError: 'condition'); now that metadata mode works, leaving this unset would silently train on the wrong labels.

## AnnotateScreen._apply_bulk_annotation

### lines 5035-5037

```python
self._pending_updates.update(batch)
```

No worker means no source is open, which _on_auto_annotate already refuses -- but a bulk write that silently went nowhere would be the worst possible failure here, so it is not assumed.

## AnnotateScreen._on_generate_annotation_db

### lines 5156-5161

```python
self._settings.png_table = table
```

OPENED, not merely reported. The screen used to read `png_list` and only `png_list`, so a second generated set landed under a name it could not show -- and a user who had just asked for a set and was then shown the old one would reasonably conclude it had failed. Every engine reader takes the table now, so the new set is simply what this screen is looking at.

## AnnotateScreen._fold_zoom_back

### line 5212  _(unsure)_

```python
self.setFocus(Qt.OtherFocusReason)
```

The grid, not the overlay, is where the next keystroke belongs.

## AnnotateScreen._fit_zoom_overlay

### line 5232  _(unsure)_

```python
return
```

The viewport went away with the screen; nothing to fit.

## AnnotateScreen._refresh_total

### lines 5277-5281

```python
self._filtered_rows = self._object_rows
```

A routed request pins the population. Rebuilding the queue or the threshold filter underneath it would replace the twelve crops somebody was sent here to look at with ninety thousand, under the same "12 objects · predicted infected" heading. No I/O, so no thread: this stays synchronous.

### lines 5288-5290

```python
settings = deepcopy(self._settings)
```

Freeze the settings now. They are a mutable dataclass with mutable lists inside it, and the dialog can be reopened while the count is still running.

### lines 5293-5295

```python
self._total_jobs.cancel()
```

A newer count supersedes an older one, exactly as a newer page load supersedes an older one -- otherwise two settings applies in quick succession leave whichever count happened to finish last on screen.

## AnnotateScreen._load_page

### line 5327  _(unsure)_

```python
self._fold_zoom_back()
```

A crop blown up over the grid belongs to the page being replaced.

### lines 5333-5334

```python
self._undo_stack.clear()
```

Slot indices now mean different crops — an undo entry from the old page would write a label onto the wrong image.

### lines 5337-5339

```python
self._revalidate_hover()
```

The crops under the grid just changed. A hover recorded against the previous page is only still true if the cursor is genuinely inside that same widget.

### line 5341  _(unsure)_

```python
first = self._next_unannotated(0)
```

Park the keyboard on the first crop that still needs a label.

### lines 5344-5345  _(unsure)_

```python
for i in range(len(self._thumbs)):
```

Repaint every cell so occupancy + resting borders match the new page (cells past the end of a short last page draw nothing).

### lines 5349-5352

```python
self._page_gen += 1
```

Process the page (normalise + outline) on a worker thread so the UI stays responsive even when the recompute is slow. A generation token discards results from a page/settings change the user has since superseded.

### lines 5356-5357

```python
settings = deepcopy(self._settings)
```

Lists inside AnnotateSettings are mutable. Freeze the complete view configuration now so a Settings change cannot race the decoder.

## AnnotateScreen._queue_page_load

### lines 5367-5369

```python
if worker is not None:
```

A finished QThread is still owned here until its queued ``finished`` slot runs on the GUI thread. Replacing that reference in the small gap would let the old slot retire the new live worker.

## AnnotateScreen._on_page_loaded

### line 5424, trailing  _(unsure)_

```python
return
```

superseded by a newer load

### lines 5430-5432

```python
self._repaint_slot(i)
```

Paint from `_page_paths`, not the annotation the worker snapshotted: the user may have keyed labels in while the page was still decoding, and those are the fresher truth.

## AnnotateScreen._crop_source

### line 5469, trailing  _(unsure)_

```python
self._cropsrc = None
```

PNG path below still works

## AnnotateScreen._slot_is_valid

### lines 5494-5500

```python
def _slot_is_valid(self, slot: int) -> bool:
```

Annotation write path

`_set_annotation` is the ONE place a label is recorded. Mouse clicks (`_toggle_annotation`), keyboard assignment, clearing and undo all funnel through it, so they can never drift apart.

## AnnotateScreen._toggle_annotation

### lines 5629-5630

```python
path = path.replace("\r", r"\r").replace("\n", r"\n")
```

A filesystem path can legally contain line breaks. Escape them so every click remains one searchable console record.

## AnnotateScreen._set_focus_slot

### lines 5657-5659

```python
if self._hover_slot is not None and self._hover_slot != slot:
```

A keyboard move away from the hovered tile makes the recorded hover stale — the cursor has not moved, but it is no longer on the tile the next action hits, and only that tile may wear the ring.

### line 5669, trailing  _(unsure)_

```python
pass
```

no viewport yet — nothing to scroll into

## AnnotateScreen._set_hover_slot

### lines 5689-5690  _(unsure)_

```python
if not (0 <= slot < self._slot_count()):
```

Empty cells past the end of a short page hold no crop, so there is nothing there to be "on".

## AnnotateScreen.handle_key

### lines 5773-5774

```python
if self._legend_expanded:
```

Only meaningful while the full reference is showing; otherwise leave Escape to whatever dialog/window wants it.

### lines 5778-5780

```python
return False
```

A TOKEN THE CHAIN DOES NOT KNOW. False means "not consumed", so the key is left for Qt's default handling -- returning True here would swallow a key the window still wants.

## AnnotateScreen._advance_after_assign

### lines 5817-5818

```python
behind = sum(1 for i in range(self._focus_slot)
```

No unlabelled crop AFTER the focus. Stay put rather than silently wrapping to the top, and say which of the two situations this is.

## AnnotateScreen._kbd_undo

### lines 5881-5882

```python
if slot < len(self._page_paths) and self._page_paths[slot][0] == path:
```

Skip entries whose slot no longer holds the same crop; writing them back would label the wrong image.

## AnnotateScreen._kbd_commit_page

### line 5894, trailing  _(unsure)_

```python
self._on_next()
```

flushes pending writes, then paginates

### lines 5895-5896  _(unsure)_

```python
if self._offset == before:
```

`_load_page` clears the hint on a successful page turn, so only the "nothing more to load" case needs to say anything.

## AnnotateScreen.eventFilter

### lines 5917-5921

```python
if getattr(self, "_closing", False):
```

``drain_thread`` processes Qt events while closeEvent waits for an active page load.  The observed widgets can therefore deliver a queued event after teardown has started.  None of these interactions has meaning once the screen is closing, and touching the partly dismantled widget tree here raises from Qt's event loop.

### line 5927, trailing  _(unsure)_

```python
return False
```

not something we can reason about

### lines 5928-5929  _(unsure)_

```python
if etype == QEvent.KeyPress and event.key() == Qt.Key_Escape \
```

A zoomed crop owns Escape, and it owns it before the grid's own key handling can read the same press as "clear the selection".

### lines 5939-5941

```python
if etype == QEvent.Resize and self._zoom_is_open():
```

The overlay fills the container, so it has to follow it. A resize that left it at its old size would put the picture off-centre and move the margin a click has to land in to fold it back.

## AnnotateScreen._detach_event_filters

### line 5970  _(unsure)_

```python
pass
```

A child can already have been deleted by its Qt parent.

## AnnotateScreen._band_event

### lines 5973-5981

```python
def _band_event(self, etype, event) -> bool:
```

Rubber-band selection

A press that lands on a _Thumbnail never reaches here: the tile accepts it in its own mousePressEvent. So "press on the grid" already means "press in the space between the images", which is exactly the gesture asked for -- there is no hit-test to get wrong and no way to start a band by mis-clicking a crop.

## AnnotateScreen._apply_band

### lines 6028-6029

```python
if rect.width() < 4 and rect.height() < 4:
```

A click, not a drag. Qt sends press+release for a plain click on the background and a zero-size band would otherwise prompt for nothing.

## AnnotateScreen.closeEvent

### lines 6112-6116

```python
for report in list(getattr(self, "_reports", {}).values()):
```

Close the report windows FIRST. They are children of this screen, so Qt would take them down with it anyway -- but only after the drains below, which run an event loop, and a report left standing over a screen that is being emptied is a window onto a half-torn-down parent.

### lines 6124-6129

```python
unregister_object_opener("annotate", self._object_opener)
```

Withdraw the routing registration first, so a request arriving during teardown cannot reach a half-destroyed screen. The bound method is passed on purpose: with two Annotate screens opened in a session, this one's closeEvent runs after the other registered, and an unconditional withdrawal would leave the live screen unreachable.

### lines 6131-6134

```python
release = getattr(self, "_release_path_probe", None)
```

Same shape of problem, the other process-wide signal: a connection left on `path_probe.probes` keeps this screen reachable from an object that outlives it, and the emission would arrive at a deleted C++ widget.

### lines 6142-6143  _(unsure)_

```python
self._total_jobs.shutdown()
```

The population count is a read-only query. Abandon it: nothing it could half-finish is worth waiting for.

### lines 6147-6159

```python
retrain.requestInterruption()
```

sklearn fits and the score write-back are native/SQLite work; tearing the widget down under them is the same class of crash as the page worker below, so this waits rather than dropping the reference. It waits with a BUDGET, though, and that is the change: `wait()` with no argument is ULONG_MAX milliseconds, so a fit that wedged -- a BLAS thread spinning, an SQLite writer blocked on a lock another process holds -- hung the close *permanently*, window still on screen, nothing to click, no way out but SIGKILL. `drain_thread` waits the budget and then PARKS the thread rather than terminating it: nothing mid-write is interrupted and the last reference to a running QThread is never dropped, which is the abort this is all arranged around. The close completes either way.

### lines 6170-6172

```python
_retire(retrain)
```

Only when it really stopped. `deleteLater` on a parked, still-running QThread is the abort being avoided; the park list owns it from here.

### lines 6176-6181

```python
suggest.requestInterruption()
```

The same budgeted drain, for the same reason: a Suggest run fits a model and then WRITES to SQLite, and tearing the widget down mid-write is the crash the paragraph above is about. It has strictly more to lose than a retrain, because a half- written suggestion set is a column the annotator has to clean up by hand.

### line 6198, trailing  _(unsure)_

```python
self._page_gen += 1
```

invalidate any in-flight results

### lines 6201-6206

```python
worker.requestInterruption()
```

Cellpose/PyTorch can stay in native inference for a long time, and letting QWidget destruction continue in that window is the intermittent SIGSEGV/abort — so this waits too, and for the same reason as the retrain worker it waits a bounded time and parks what will not stop. A page that is still decoding must not be able to hold the window open forever.

## AnnotateScreen._on_file_issue

### fixed 2026-09-19, after review (the button had never filed anything)

```python
file_issue(self, {"screen": "annotate"}, body)
```

`file_issue(traceback_text, active_app, settings)` -- so that call passed the screen as the traceback, a dict as the app id and the console text as the settings, and the first thing the reporter did was `sanitize_path(<AnnotateScreen>)`. The user got `Could not file the issue: 'AnnotateScreen' object has no attribute 'replace'`. The `except TypeError` wrapped around it was guarding against a signature mismatch Python never raises here: every argument was positional and the arity was right.

It survived because the button is hidden behind `get_auto_file_issues()`, which shipped OFF. That default is now ON, so this button is part of the default experience, and `tests/qt/test_annotate_console_can_be_copied_and_reported.py` presses it.

It now builds the report from the console text with `active_app="annotate"`, shows the same `IssuePreviewDialog` the module screens show -- the button's own tooltip promises "You review it before submitting", and a press is already the affirmative act that 'always' exists to avoid asking for -- and posts from its own `JobRunner`. Not from `_total_jobs`: that one is cancelled whenever a count is restarted, and a cancelled generation is a result the runner drops. Posting stays off the GUI thread for the reason the module screens measured: up to 28 s of frozen window between `gh auth token` and `api.github.com`.
