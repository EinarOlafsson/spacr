# Notes from `spacr/qt/widgets/measure_preview.py`

Prose lifted out of `spacr/qt/widgets/measure_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_merged_source](#resolve_merged_source) (1 entry)
- [load_merged_array](#load_merged_array) (2 entries)
- [annotate_crops](#annotate_crops) (1 entry)
- [MeasurePreviewPanel](#measurepreviewpanel) (1 entry)
- [MeasurePreviewPanel.__init__](#measurepreviewpanel__init__) (4 entries)
- [MeasurePreviewPanel._build_controls](#measurepreviewpanel_build_controls) (8 entries)
- [MeasurePreviewPanel._build_ui](#measurepreviewpanel_build_ui) (4 entries)
- [MeasurePreviewPanel._on_object_changed](#measurepreviewpanel_on_object_changed) (1 entry)
- [MeasurePreviewPanel._auto_load_from_src](#measurepreviewpanel_auto_load_from_src) (1 entry)
- [MeasurePreviewPanel._on_array_loaded](#measurepreviewpanel_on_array_loaded) (1 entry)
- [MeasurePreviewPanel._on_fov_changed](#measurepreviewpanel_on_fov_changed) (1 entry)
- [MeasurePreviewPanel.settings_for_propagation](#measurepreviewpanelsettings_for_propagation) (2 entries)
- [MeasurePreviewPanel._png_channel_mapping](#measurepreviewpanel_png_channel_mapping) (1 entry)
- [MeasurePreviewPanel.apply_settings](#measurepreviewpanelapply_settings) (4 entries)
- [MeasurePreviewPanel._current_mask_dim](#measurepreviewpanel_current_mask_dim) (1 entry)
- [MeasurePreviewPanel.refresh](#measurepreviewpanelrefresh) (1 entry)
- [MeasurePreviewPanel._on_crops_ready](#measurepreviewpanel_on_crops_ready) (1 entry)
- [MeasurePreviewPanel._crop_pixmap](#measurepreviewpanel_crop_pixmap) (1 entry)
- [CropSettingsDialog.__init__](#cropsettingsdialog__init__) (1 entry)
- [CropSettingsDialog.closeEvent](#cropsettingsdialogcloseevent) (1 entry)

## resolve_merged_source

### lines 103-104  _(unsure)_

```python
for folder in (candidate / "merged", candidate):
```

`merged/` first: a run folder holds `merged/` beside `measurements/` and `qc/`, and an array loose in the run folder is not the one meant.

## load_merged_array

### lines 132-133

```python
resolved = resolve_merged_source(path)
```

A FOLDER IS RESOLVED HERE, on the worker, because the walk is a disk scan and this function exists to keep those off the GUI thread.

### line 149

```python
LOG.exception("Could not enumerate merged arrays beside %s", path)
```

A folder we cannot group is still one we can show an array from.

## annotate_crops

### lines 245-246  _(unsure)_

```python
included = nucleus is not False
```

The pipeline requires a nucleus when all companion masks exist, and additionally requires a pathogen when uninfected is off.

## MeasurePreviewPanel

### lines 435-438

```python
preview_ready = Signal(object)
```

The list of crops a pass produced, or None when it produced nothing. The other three live views have announced their result since they were written; this one was the odd column out, so nothing outside the panel could tell that a crop pass had landed.

## MeasurePreviewPanel.__init__

### lines 459-465

```python
self._jobs = JobRunner(self, threaded=threaded,
```

Reading a merged array and cropping it are both far too slow for the GUI thread: a 1024x1024x8 array off a warm SSD froze the window for 2469 ms on a drop and 1441 ms on every single spinbox step. Both now go through the runner, which also registers them so the activity spinner turns. `threaded=False` runs each job inline, emitting the same signals in the same order, so a test can drive this panel synchronously without the behaviour diverging.

### lines 471-472  _(unsure)_

```python
self._loading_fov = False
```

Guards the FOV dropdown against re-entering itself while the array it just asked for is being installed.

### lines 474-475

```python
self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
```

Bounded, reproducible sample of the folder's image sets — the dropdown never lists a whole plate. See ImageSetSampler.

### lines 481-483

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## MeasurePreviewPanel._build_controls

### lines 515-519

```python
"""Build the control row: object, crop modes and sizes."""
```

General

"experiment", not "exp". The abbreviation was the DEFAULT VALUE of the field, so it also became the experiment name of every run left untouched -- a folder called `exp` says nothing six months later. Asked for on 2026-09-01 with the `src` label.

### line 537  _(unsure)_

```python
self._save_png = Toggle(parent=self)
```

Object crops

### lines 549-562

```python
self._png_dims = ChannelMappingWidget(_default_png_mapping(), self)
```

R, G, B source channels in that order, defaulted to the shipped `png_channel_mapping` ({'r': 2, 'g': 1, 'b': 0}) rather than to "0,1,2". The text is handed straight to `crop_objects_from_array`, whose `channels` argument IS RGB order, so a default of "0,1,2" drew channel 0 RED while the run writes it BLUE -- measured on a three-channel array as preview (13, 128, 255) against run (200, 100, 10). A crop preview exists to answer "which stain lands where", and it answered with red and blue swapped. NAMED COLOUR SLOTS, not a comma list whose POSITION decides which colour a channel lands in. A positional list carries an unstated convention, and an unstated convention gets read backwards -- which is exactly how the 405 plane spent eleven days rendering red. The same editor the Measure settings page uses, so the two cannot disagree about what "channel 2" means.

### lines 568-570

```python
self._lo_pct = QDoubleSpinBox(self)
```

Six decimals, and set BEFORE the range and the value -- see the same pair in `live_preview.py`. These carried Qt's default two, which stores what it shows: 99.995 became 100.0.

### line 591  _(unsure)_

```python
self._min_sizes = {
```

Filtering

### line 601  _(unsure)_

```python
self._max_area = self._spin(0, 100_000_000, 0, parent=self)
```

Preview-only controls

### lines 606-612

```python
self._propagate_btn = QPushButton("Propagate settings", self)
```

A BUTTON, NOT A SLIDER. The Mask live settings dialog puts

"Propagate settings" in its button box as a checkable ToggleButton; this one used a Toggle, which is the sliding switch used for ordinary boolean SETTINGS. Two different controls for the same action in two dialogs that sit side by side, and the slider reads as a setting of the crop preview rather than as something that reaches out of it.

### line 621  _(unsure)_

```python
self._mask_dim = self._mask_dims["cell"]
```

Compatibility names used by integrations and older tests.

## MeasurePreviewPanel._build_ui

### lines 653-654  _(unsure)_

```python
pick_row = QHBoxLayout()
```

FOV and channel dropdowns sit immediately LEFT of the Choose control; all three wear the flat "Live toggle" look.

### lines 679-683

```python
self._paste_box = QLineEdit(self)
```

A PLACE TO PASTE. The file dialog cannot select a folder, and a run folder is what the user has in hand -- it is what `src` holds and what a Measure run is pointed at. Typing or pasting one here loads a field from its `merged/` without hunting through fifty-two arrays for one whose name says nothing about which is interesting.

### lines 702-703  _(unsure)_

```python
self._refresh_btn = self._run_btn
```

The old name for the same button. Kept so anything that reached for it by name still finds it, pointing at the one control.

### line 705  _(unsure)_

```python
self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT)
```

Same control, same place, same words as the Mask live preview.

## MeasurePreviewPanel._on_object_changed

### line 859  _(unsure)_

```python
"""Re-preview for a different object type."""
```

A previewed object should also be one of the requested crop outputs.

## MeasurePreviewPanel._auto_load_from_src

### line 966

```python
self._auto_loaded_src = text
```

Something is already on screen; do not take it away.

## MeasurePreviewPanel._on_array_loaded

### lines 1005-1006

```python
self._sampler.adopt(payload.get("directory"), sets,
```

Adopt before installing, so the enumerate() inside _refresh_source_selectors is a cache hit rather than a re-scan.

## MeasurePreviewPanel._on_fov_changed

### lines 1097-1098

```python
self.load_array_async(path, enumerate_sets=False)
```

The path came out of the sampler, so the folder is already listed; re-scanning would rediscover what is in hand.

## MeasurePreviewPanel.settings_for_propagation

### lines 1150-1154

```python
"png_channel_mapping": self._png_channel_mapping(),
```

`png_channel_mapping`, NOT the legacy `png_dims` this control used to write. `resolve_png_channel_mapping` ignores png_dims whenever a mapping is set, and Measure sets one by default so every value this control propagated was discarded by the run it was tuning.

### lines 1161-1164

```python
**{self._size_floor_key(name): int(widget.value())
```

EVERY FLOOR THE PANEL OFFERS, built from the controls the way the mask dims above are. Written out as literals this listed five of the eight, so the organelleb, organellec and organelled spin boxes could be set and were dropped on propagate.

## MeasurePreviewPanel._png_channel_mapping

### lines 1191-1192

```python
return dict(self._png_dims.get_value())
```

A colour left empty stays empty. The editor says so with "—", and the run must not put a plane back into a slot the user cleared.

## MeasurePreviewPanel.apply_settings

### lines 1232-1234

```python
try:
```

-1 is the spinbox's "Not present" and None is the settings dict's. They have to translate, or an organelle declared absent comes back pointing at channel 0.

### lines 1270-1271  _(unsure)_

```python
if "normalize" in settings:
```

`normalize` is a bool OR a [lo, hi] percentile pair, and the pair is the only place the percentiles come from.

### lines 1285-1286

```python
if "png_channel_mapping" in settings or "png_dims" in settings:
```

Through the run's own resolver, so a legacy `png_dims` settings file seeds the panel with the colours that file will produce.

### lines 1290-1295

```python
if settings.get("number_of_organelles") is not None:
```

LAST, and after the settings above have landed: the crops are cut with these values, so loading first would show the panel's defaults and then re-cut. `src` is the folder the run will read, so the preview can answer for it without being asked. HOW MANY ORGANELLE SLOTS, the same rule Mask and the Mask live preview follow: the run declares it, every panel shows that many.

## MeasurePreviewPanel._current_mask_dim

### lines 1327-1328  _(unsure)_

```python
name = "cell"
```

Cytoplasm is generated during measurement and has no stable input slice. Use cells for the preview footprint.

## MeasurePreviewPanel.refresh

### lines 1423-1424

```python
one = self.display_channel()
```

The channel dropdown is a *view* control: it does not change the png_dims that a real run would write, only what this grid shows.

## MeasurePreviewPanel._on_crops_ready

### line 1478  _(unsure)_

```python
self.preview_ready.emit(self._crops)
```

Announced like every other live view's result.

## MeasurePreviewPanel._crop_pixmap

### lines 1572-1574

```python
from ...crops import apply_display_primaries
```

A DISPLAY transform and nothing else. `crop` is the array the pipeline would write and it is not touched -- only this thumb is recoloured, and set_preview_status names the mapping.

## CropSettingsDialog.__init__

### lines 1699-1705

```python
crops_form.addRow("Match crop height to width", panel._lock_aspect)
```

A CROP BOX, NOT A GRAPH. This is neither of the two shape controls a figure has: it is not the shape of a plotted page ("Graph shape") and it is not the axis-scale lock that ties one y unit to n x units ("lock axis scales"). It is the pixel box each object is cut out into, and what the toggle does is hold that box square by carrying the width over to the height -- so it says that rather than borrowing a figure's word for a different quantity.

## CropSettingsDialog.closeEvent

### line 1786  _(unsure)_

```python
"""Remember the dialog's geometry before it goes.
```

Keep control values alive on the panel between dialog openings.
