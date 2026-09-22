# Notes from `spacr/qt/widgets/live_preview.py`

Prose lifted out of `spacr/qt/widgets/live_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (12 entries)
- [load_preview_image](#load_preview_image) (1 entry)
- [load_preview_mip](#load_preview_mip) (1 entry)
- [_to_uint8](#_to_uint8) (1 entry)
- [_labelled_boundary](#_labelled_boundary) (1 entry)
- [safe_outline_palette](#safe_outline_palette) (1 entry)
- [overlay_masks](#overlay_masks) (2 entries)
- [load_source_payload](#load_source_payload) (2 entries)
- [_PreviewWorker](#_previewworker) (2 entries)
- [_PreviewWorker.run](#_previewworkerrun) (1 entry)
- [_classical_organelle_mask](#_classical_organelle_mask) (2 entries)
- [_segment_multi](#_segment_multi) (6 entries)
- [_apply_size_filter](#_apply_size_filter) (1 entry)
- [_ZoomView](#_zoomview) (2 entries)
- [_ZoomView.__init__](#_zoomview__init__) (1 entry)
- [_ZoomView.set_pixmap](#_zoomviewset_pixmap) (1 entry)
- [_ZoomView._apply_zoom](#_zoomview_apply_zoom) (1 entry)
- [LivePreviewPanel](#livepreviewpanel) (1 entry)
- [LivePreviewPanel.__init__](#livepreviewpanel__init__) (11 entries)
- [LivePreviewPanel._stow_free_widgets](#livepreviewpanel_stow_free_widgets) (2 entries)
- [LivePreviewPanel.dropEvent](#livepreviewpaneldropevent) (1 entry)
- [LivePreviewPanel._build_ui](#livepreviewpanel_build_ui) (28 entries)
- [LivePreviewPanel._load_for_display](#livepreviewpanel_load_for_display) (1 entry)
- [LivePreviewPanel._install_loaded_image](#livepreviewpanel_install_loaded_image) (2 entries)
- [LivePreviewPanel.resizeEvent](#livepreviewpanelresizeevent) (1 entry)
- [LivePreviewPanel._refresh_source_selectors](#livepreviewpanel_refresh_source_selectors) (3 entries)
- [LivePreviewPanel._regex_config](#livepreviewpanel_regex_config) (1 entry)
- [LivePreviewPanel._populate_set_table](#livepreviewpanel_populate_set_table) (3 entries)
- [LivePreviewPanel._on_set_cell_clicked](#livepreviewpanel_on_set_cell_clicked) (2 entries)
- [LivePreviewPanel._on_mip_toggled](#livepreviewpanel_on_mip_toggled) (1 entry)
- [LivePreviewPanel._on_fov_changed](#livepreviewpanel_on_fov_changed) (2 entries)
- [LivePreviewPanel._apply_display_background](#livepreviewpanel_apply_display_background) (2 entries)
- [LivePreviewPanel.settings_for_propagation](#livepreviewpanelsettings_for_propagation) (5 entries)
- [LivePreviewPanel.apply_settings](#livepreviewpanelapply_settings) (9 entries)
- [LivePreviewPanel.run_preview](#livepreviewpanelrun_preview) (1 entry)
- [LivePreviewPanel._build_compartment_widgets._spin](#livepreviewpanel_build_compartment_widgets_spin) (1 entry)
- [LivePreviewPanel._build_compartment_widgets](#livepreviewpanel_build_compartment_widgets) (9 entries)
- [LivePreviewPanel._build_compartment_widgets._organelle_widget](#livepreviewpanel_build_compartment_widgets_organelle_widget) (2 entries)
- [LivePreviewPanel._widget_value](#livepreviewpanel_widget_value) (1 entry)
- [LivePreviewPanel._keys_whose_off_is_none](#livepreviewpanel_keys_whose_off_is_none) (1 entry)
- [LivePreviewPanel._compartment_settings](#livepreviewpanel_compartment_settings) (3 entries)
- [LivePreviewPanel._seed_organelle_column](#livepreviewpanel_seed_organelle_column) (1 entry)
- [LivePreviewPanel._apply_cycle_stop](#livepreviewpanel_apply_cycle_stop) (1 entry)
- [LivePreviewPanel._follow_object_channel](#livepreviewpanel_follow_object_channel) (1 entry)
- [LivePreviewPanel._on_primary_object_changed](#livepreviewpanel_on_primary_object_changed) (1 entry)
- [LivePreviewPanel._build_request](#livepreviewpanel_build_request) (1 entry)
- [LivePreviewPanel._refresh_canvases](#livepreviewpanel_refresh_canvases) (1 entry)
- [LivePreviewPanel._label_rgb](#livepreviewpanel_label_rgb) (2 entries)
- [LivePreviewPanel._on_model_or_object_changed](#livepreviewpanel_on_model_or_object_changed) (1 entry)
- [LivePreviewPanel._on_settings_closed](#livepreviewpanel_on_settings_closed) (1 entry)
- [LivePreviewPanel._pick_file](#livepreviewpanel_pick_file) (1 entry)
- [LivePreviewPanel._on_hover](#livepreviewpanel_on_hover) (2 entries)
- [LivePreviewPanel._on_worker_done](#livepreviewpanel_on_worker_done) (1 entry)
- [LivePreviewPanel._recompute_masks](#livepreviewpanel_recompute_masks) (1 entry)
- [LivePreviewPanel._snapshot_run](#livepreviewpanel_snapshot_run) (2 entries)
- [LivePreviewPanel._on_compare_scrub](#livepreviewpanel_on_compare_scrub) (1 entry)
- [LiveSettingsDialog.__init__](#livesettingsdialog__init__) (13 entries)
- [LiveSettingsDialog._on_propagate_toggled](#livesettingsdialog_on_propagate_toggled) (1 entry)
- [LiveSettingsDialog.refresh_visibility](#livesettingsdialogrefresh_visibility) (5 entries)
- [overlay_mask](#overlay_mask) (1 entry)

## Module level

### lines 113-116

```python
OBJECT_TYPES = ("cell", "nucleus", "cell + nucleus", "pathogen", "organelle")
```

Object types the panel understands. Order matters — it drives the order of the combo. cell/nucleus can be previewed together; pathogen and organelle are single-compartment selections whose settings panels light up when chosen.

### lines 151-152  _(unsure)_

```python
COMPARTMENTS = ("cell", "nucleus", "pathogen", "organelle")
```

The four segmentation compartments, in the left→right order their settings panels appear in the Live settings dialog.

### lines 155-157

```python
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
```

Overlay colours for individual object types. Cell = green (matches the classic v1 boundary colour), nucleus = magenta, and when both are shown together those colours read cleanly on top of most stains.

### lines 165-166  _(unsure)_

```python
RANDOM_OUTLINE_SEEDS: Dict[str, int] = {
```

Stable offsets keep the random categorical outline map distinct between compartments without making colours flicker whenever the preview refreshes.

### lines 174-178

```python
ORGANELLE_METHOD_FIELDS: Dict[Optional[str], tuple] = {
```

Per-compartment tuning settings, shown (greyed unless the compartment is the chosen object) in that compartment's panel. Each entry is ``(key_suffix, label, kind, spin_args)`` where the real setting key is ``f"{compartment}_{key_suffix}"`` and kind is one of int/float/bool/method. spin_args = (min, max, default) for int/float; ignored otherwise.

### lines 237-259

```python
COMPARTMENT_FIELDS = (
```

THE RELATIVE SCHEME IS GONE FROM THIS TABLE (391), MEASURED AGAINST WHAT THE RUN READS RATHER THAN TRIMMED BY EYE.

`spacr.object.merge_split_filter_masks` is the only reader of these per-compartment keys, and it now looks up exactly eight suffixes plus one absolute threshold. Comparing the table against it, and against `spacr.settings.set_default_settings_preprocess_generate_masks`:

LEFT (5 suffixes x 4 compartments = 20 keys that nothing read): area_multiplier, intensity_threshold_method, intensity_percentile, min_intensity_percentile, max_intensity_percentile ARRIVED (1 suffix x 4 compartments = 4 keys the run reads and the panel could not send at all): intensity_threshold

The twenty were not merely inert. Every one of them is listed in `spacr.object_roles.WITHDRAWN_SETTING_SUFFIXES`, i.e. the settings loader already tells a user who opens a saved file that these are no longer read while this panel went on offering them as live controls and writing them into the run. And the absent `intensity_threshold` was the worse half: the "Intensity merge" toggle below propagates fine, but with no threshold beside it the run has nothing to compare a shared boundary against, so it refuses to merge. The preview said "merging on"; the run merged nothing.

### lines 263-275

```python
("minimum_area_to_split",      "Minimum area to split", "int",   (0, 100_000_000, 100)),
```

BOTH RENAMED 2026-09-12 (391), KEYS AND LABELS, and both are honest corrections rather than cosmetics:

"Min object area" was NOT a minimum object area. An object smaller than it is KEPT -- it is simply never split -- so a user reading the old name would reasonably expect small objects to be discarded. "Min distance" is the minimum separation between watershed seeds and means nothing outside that algorithm.

These are written WITHOUT the leading underscore, which is how the previous rename (`b7ae412af`) missed them: a suffix substitution anchored on `_` does not see a bare suffix. Worth remembering for the next one.

### lines 278-283

```python
("perimeter_fraction",         "Perimeter fraction",    "float", (0.0, 1.0, 0.0)),
```

Defaults MUST match spacr.settings.set_default_settings_preprocess_generate_masks. They are both what the preview filters with and what the Propagate button writes into the main settings panel, so any drift silently re-tunes the real run. That is also how the five withdrawn rows named at the top of this table were caught: they had no pipeline default to match, because the pipeline had stopped shipping them.

### lines 285-296

```python
("intensity_threshold",        "Intensity threshold",   "float", (0.0, 1_000_000.0, 0.0)),
```

ONE ABSOLUTE INTENSITY, IN THE IMAGE'S OWN RAW UNITS, replacing the method dropdown and the three percentiles. It is the boundary mean two touching labels must reach before "Intensity merge" joins them.

The pipeline ships it as None -- "no number, so refuse to merge and report the boundary intensities found" -- and a spin box cannot hold None, so 0 is what says it here and `_off_as_the_run_spells_it` turns that back into None on the way out. The cost of the sentinel is that a literal threshold of 0 ("merge every pair that touches") has to be asked for as a very small positive number instead; the alternative is a preview that cannot express the run's own default, which is how the panel came to disagree with the run in the first place.

### lines 303-305

```python
OUTLINE_CHOICES = ("auto", "color (random)", "green", "magenta",
```

What the outline-colour dropdown offers, in the order it offers it. ``auto`` is one colour per compartment and ``color (random)`` one colour per object; the rest are the fixed colours in :data:`OUTLINE_COLOURS`.

### line 309  _(unsure)_

```python
VIEW_MODES = ("Overlay", "Masks", "Flows")
```

What the right-hand canvas can show.

### lines 1076-1078  _(unsure)_

```python
CLICK_SLOP_PX = 4
```

Twin zoomable views with a shared transform

## load_preview_image

### lines 313-315  _(unsure)_

```python
def load_preview_image(path: Path) -> np.ndarray:
```

Pure numpy helpers — no Qt, safe to unit-test without a display

## load_preview_mip

### lines 358-360

```python
raise ValueError(
```

A field whose planes disagree is not a stack. Showing the first plane is wrong quietly; refusing is wrong loudly, which is the one the user can act on.

## _to_uint8

### lines 410-417

```python
if img.ndim == 3 and img.shape[-1] == 1:
```

Channels-last is this module's convention everywhere else (see :func:`_select_channel` and :meth:`LivePreviewPanel._label_rgb`), so a single-channel tile collapses to grayscale and anything wider maps its first three channels onto R/G/B. This used to be gated on ``shape[-1] in (2, 3, 4)``, which sent (H, W, 1) and (H, W, 5+) tiles down the 2-D branch and returned an array with a trailing channel axis — :func:`numpy_to_qpixmap` then handed Qt a stride three times the real row length and read past the end of the buffer.

## _labelled_boundary

### lines 478-479  _(unsure)_

```python
neighbours = np.zeros_like(labels)
```

Give exterior boundary pixels the label of an adjacent object. This preserves the two-sided outline produced by the pre-existing renderer.

## safe_outline_palette

### lines 553-555

```python
return None
```

No QSettings, no Qt, or a preferences module that moved: a random colour is the historic behaviour and is never worse than crashing the renderer over a palette.

## overlay_masks

### lines 650-652

```python
LOG.debug("overlay_masks: skipping %s mask %s — image is %s",
```

A mask left over from a previously loaded image. Drawing it raised ``IndexError: boolean index did not match indexed array`` (or a broadcast ValueError) instead of simply being ignored.

### line 675  _(unsure)_

```python
b2 = boundary.copy()
```

Dilate by one pixel: OR-shift in each cardinal direction

## load_source_payload

### lines 795-798

```python
picked = sample_image_sets(
```

Open on one of the sampled sets. Whichever file sorts first is A01 field 1, which on a plate-ordered folder is exactly the corner the sample exists to stop the preview from standing in for.

### lines 805-806

```python
LOG.exception("Could not enumerate image sets under %s",
```

A folder we cannot group is still a folder we can show one image from; the panel falls back to per-file sets.

## _PreviewWorker

### line 838  _(unsure)_

```python
finished_masks = Signal(object, str, int)
```

({obj: mask, ...} or None, err, run token)

### line 840  _(unsure)_

```python
flows_ready = Signal(object, int)
```

({obj: flow_rgb} — may be empty, run token)

## _PreviewWorker.run

### lines 874-875  _(unsure)_

```python
if isinstance(res, tuple):
```

_segment_multi may return masks only (the stubbed test path) or (masks, flows). Handle both.

## _classical_organelle_mask

### lines 910-912

```python
remapped = dict(settings)
```

Slot 2's keys are `organelleb_*`; the segmentation function reads `organelle_*`. Remapped rather than passed through, so every slot previews with its own values instead of slot 1's.

### lines 918-920

```python
try:
```

Defaults for anything the panel was never given -- the classical routines index their settings directly, so a missing key is a KeyError in a worker thread rather than a preview that says what is wrong.

## _segment_multi

### lines 940-943

```python
model = preview_cellpose_model(req.model)
```

ONE constructor for every live view — see

`preview_contract.preview_cellpose_model` for why `model_type=` may never appear here. The Timelapse preview calls the same helper, so the next Cellpose API change is one fix rather than two.

### lines 952-968

```python
if req.preprocess_settings.get(f"remove_background_{obj}"):
```

Preprocess — remove background if the user opted in, doing exactly what a real run does. `spacr.io._normalize_img_batch` runs

single_channel[single_channel < background] = 0

per channel, reading `{obj}_background`. This used to subtract the background and clip at zero instead, which is a different image: thresholding leaves everything above the background where it is, subtraction shifts all of it down. And it read a plain `background` key that nothing writes -- the panel emits `{obj}_background` -- so the value was always the 100.0 default and turning the toggle on did nothing visible on any image whose real background was not near 100.

Both keys are per-object on purpose: with "cell + nucleus" selected, the two channels get their own background and their own on/off, the same way the pipeline treats them.

### lines 973-975

```python
image_2d = image_2d.copy()
```

`_select_channel` hands back a view into `req.image`. Writing through it would zero the source for every object type after this one, and for the raw pane the panel shows beside the mask.

### lines 979-981

```python
method = str(req.preprocess_settings.get(
```

NOT EVERY OBJECT IS A CELLPOSE OBJECT. An organelle whose method is anything but 'cellpose' is segmented by the classical routines the run itself uses; only 'cellpose' reaches the model below.

### lines 1002-1003  _(unsure)_

```python
try:
```

Capture the RGB flow visualisation (flows[0]) if Cellpose returned one, so the panel can show a Flows view alongside the masks.

### lines 1013-1015

```python
out[obj] = mask
```

Return the RAW (unfiltered) mask — the panel applies the per- compartment filters afterwards so the user can re-tune filters without re-running Cellpose (see LivePreviewPanel._recompute_masks).

## _apply_size_filter

### lines 1053-1058

```python
if not (min_area > 0 or max_area > 0 or remove_border):
```

THE INTENSITY PERCENTILES ARE GONE (391). The comment below used to record that this preview once defaulted them to 1/99 where the pipeline used 0/100, so the preview silently dropped the dimmest and brightest object in every field. That defect is now unreachable rather than fixed: a quantile band always removes its share, so there was no value of it that meant "do nothing" except the endpoints.

## _ZoomView

### line 1107, trailing  _(unsure)_

```python
hover_pixel = Signal(int, int)
```

(x, y) in image coords

### line 1108, trailing  _(unsure)_

```python
zoom_changed = Signal(float)
```

new scale factor

## _ZoomView.__init__

### lines 1130-1134

```python
self.horizontalScrollBar().valueChanged.connect(self._mirror_pan)
```

Panning has to be mirrored the same way zoom is. `ScrollHandDrag` moves the scroll bars rather than the transform, so `_apply_zoom` never sees a drag and the twin canvases stayed locked in scale while drifting apart in position -- zoom in, drag one, and the mask no longer sits over the cell it was drawn from.

## _ZoomView.set_pixmap

### lines 1147-1148

```python
self._user_zoomed = False
```

Fit-in-view on load, and forget any previous user zoom so the new image starts at 100 % of the canvas.

## _ZoomView._apply_zoom

### lines 1209-1212

```python
self._syncing = True
```

Guard THIS view while the peer catches up, not the peer: the flag makes _apply_zoom a no-op, so setting it on the peer meant the peer's own zoom was skipped and the twin canvases never actually tracked each other.

## LivePreviewPanel

### line 1401, trailing  _(unsure)_

```python
preview_ready = Signal(object)
```

{object_type: mask}

## LivePreviewPanel.__init__

### lines 1422-1429

```python
self._load_jobs = JobRunner(self, threaded=threaded,
```

Every preview load goes through here rather than through a QThread this file owns. That is not tidiness: `JobRunner` submits via `bridge.make_thread`, which is what puts the job in the process-wide run registry, and the registry is the *only* thing the activity spinner watches. The hand-rolled loader this replaced ran off the GUI thread perfectly well and still left the user staring at a frozen- looking window with no spinner, because nothing ever told the registry it existed.

### lines 1433-1435

```python
self._run_token: int = 0
```

Bumped whenever the run in flight is superseded (a new image, an explicit cancel). A worker's result is only accepted when the token it carries still matches.

### line 1437  _(unsure)_

```python
self._propagate_cb = None
```

Callback(dict) that pushes tuned live settings into the main panel.

### lines 1439-1440  _(unsure)_

```python
self._auto_outline_colours: Dict[str, Tuple[int, int, int]] = {}
```

One random colour per compartment for the 'auto' outline mode, re-rolled on every preview run (see _roll_auto_outline_colours).

### lines 1442-1443  _(unsure)_

```python
self._loading_fov = False
```

Guards the FOV dropdown against re-entering itself while the image it just asked for is being installed.

### lines 1445-1447

```python
self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
```

Groups the source folder into image sets from file names alone and hands out a bounded, reproducible random sample of them. Caches the enumeration per folder, so stepping through fields costs nothing.

### line 1449  _(unsure)_

```python
self._mip_enabled = False
```

Off until the user asks; only meaningful where z_count > 1.

### lines 1451-1453

```python
self._table_row = 0
```

Which cell the table is showing, so clicking a channel header keeps the field and clicking a field keeps the channel instead of resetting to the first of either.

### lines 1464-1467

```python
self.setAcceptDrops(True)
```

Accept image files dropped anywhere on the panel. QGraphicsView enables acceptDrops by default and would otherwise swallow drops over the image canvases; turning it off on the views lets the drag events propagate up to this panel's handlers.

### lines 1473-1475

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

### lines 1478-1481

```python
self._stow_free_widgets()
```

LAST, so it sees every control this constructor made. See the method: the panel's settings controls belong to the panel but are laid out by `LiveSettingsDialog`, so between dialogs they are parented here with no layout -- which is the top-left corner.

## LivePreviewPanel._stow_free_widgets

### lines 1537-1542

```python
if child.isWindow():
```

A WINDOW IS NOT A STRAY. `LiveSettingsDialog` is parented to the panel and is in no layout, but it carries Qt::Window, so the window manager places it and it never paints inside the panel at all. Stowing it would re-parent a live dialog into a hidden container, which is a worse bug than the one being fixed here.

### lines 1545-1557

```python
child.setParent(container)
```

Parented, NOT added to the container's layout: these are not meant to be seen in it, only to have a home that is never painted. Adding them would also fight the dialog, which takes them out of whatever layout holds them each time it opens. PARENT ONLY, NO setVisible(False). The container is never shown, so its children cannot paint -- that is the whole mechanism, and an explicit hide adds nothing to it. What an explicit hide DOES add is `WA_WState_ExplicitShowHide`, which makes the widget stay hidden when a layout later takes it: `LiveSettingsDialog` shows the widgets it borrows, but only the ones `_managed_widgets()` names, and the rest arrived in their rows already hidden for good. It cost one settings row its hover help, which is how it was found.

## LivePreviewPanel.dropEvent

### lines 1604-1607

```python
self.load_source_async(path)
```

Asynchronously — the drop handler must return to the event loop immediately. The decode is the small half; the expensive half is enumerating the folder the dropped file came out of, which is what froze the window for 643 ms on a 98 304-file plate.

## LivePreviewPanel._build_ui

### lines 1624-1638

```python
self._model_box = QComboBox(self)
```

HIDDEN state widgets

Every parameter widget lives here even though only a subset appears in the collapsed layout. The Live Settings dialog re-parents them into its own form when it opens, then hands them back on close so their values persist across opens. All widgets are children of `self` so they're never garbage-collected while re-parented. Read from the Cellpose API, not from a literal — see `spacr.settings.cellpose_model_menu`. It returns whatever `cellpose.models` reports plus any checkpoint the user has registered, then the accepted-but-mapped aliases cyto3/cyto2/ nuclei so a saved preview setting still loads. Those are NOT four choices: Cellpose 4 drops model_type= with a "not used in v4.0.1+" log line, so all four run the same cpsam weights. The pipeline maps them forward in settings.normalize_cellpose_model_name.

### lines 1645-1649

```python
set_translatable_items(self._object_box, OBJECT_TYPES)
```

The caption is read by a user and the entry is read by the code: every `{object}_…` setting key, the channel map handed to the worker and `_selected_object_types` are all spelled in English. Translating the caption in place used to rewrite the value with it, so a Swedish screen asked the worker to segment `cellen`.

### lines 1657-1661

```python
self._pathogen_channel = QSpinBox(self)
```

Pathogen and organelle are in OBJECT_TYPES and each has its own settings panel in the dialog, but neither had a channel control. `_build_request` built its channel map from cell and nucleus only, so `channels.get(obj, 0)` fell back to 0 and picking "pathogen" segmented the cell channel while appearing to work.

### lines 1674-1683

```python
self._flow.setRange(-1, 100); self._flow.setSingleStep(0.05)
```

UP TO 100, WHICH IS WHAT MASK SHIPS. The useful range is about 0 to 3 -- Cellpose's own default is 0.4, and that is what this box opens on for the single-object modules that also reach this panel. But Mask ships 100 per object, documented as "accepts every candidate", and a box that stops at 3 CANNOT HOLD ITS OWN SETTING: seeding the panel from Mask clamped 100 to 3 silently, and propagating handed that 3 back as if the user had chosen it.

The step stays at 0.05, so the useful end is still reachable a notch at a time; 100 is typed, not scrolled to.

### lines 1690-1691  _(unsure)_

```python
self._normalise_check = Toggle("Normalise", self)
```

Two-field percentile stretch — user asked for this shape explicitly (was a single toggle before).

### lines 1695-1709

```python
self._lo_pct = QDoubleSpinBox(self)
```

setDecimals BEFORE setRange/setValue. A QDoubleSpinBox rounds both to the precision it holds at the time, so ordering these the other way stores 100.0 for a 99.9999 default and the box looks broken rather than imprecise -- the same note make_masks.py carries.

SIX DECIMALS, not Qt's default two. Nobody chose two; it is what QDoubleSpinBox ships with and neither box ever overrode it. The cap does not merely round the DISPLAY -- the box stores what it shows, so a user typing 99.995 got 100.0, the top of the range, and the stretch they asked for silently became no stretch at all. On a 4 MP field 99.9 spares 4,000 pixels, 99.99 spares 400 and 99.999 spares 40; two decimals cannot express the difference between the last two, and those are the ones that decide whether a hot pixel pins the display range. Matches `percentile_pair.DECIMALS` rather than being a third opinion about the same quantity.

### lines 1714-1716

```python
self._lo_pct.setSingleStep(0.01)
```

The step stays coarse on purpose. A step of 1e-6 would need a million clicks to cross a percent; the fine end is typed, the coarse end is scrolled, which is what make_masks settled on.

### line 1726  _(unsure)_

```python
self._outline_colour = QComboBox(self)
```

Outline appearance

### lines 1728-1733

```python
set_translatable_items(self._outline_colour, OUTLINE_CHOICES)
```

The colour a user picks is read back out of the entry's DATA, so the caption is free to follow the language. Reading it back by text is what made these entries untranslatable before: every choice missed the colour mapping and silently fell back to the per-compartment default (green for cells) — an outline colour that could never be changed.

### lines 1735-1741

```python
self._outline_colour.setCurrentIndex(
```

Random is the default. A fixed colour is a coin flip against the image -- green outlines on a green channel are invisible exactly when you most need to see whether the mask landed -- and `auto` picks per compartment, so two touching objects of the same type share an outline and read as one. Set before the signal is connected so choosing it here does not fire a render on a panel that has no image yet.

### line 1752  _(unsure)_

```python
self._model_box.setToolTip(
```

Tooltips for the segmentation controls (type + what they do).

### lines 1793-1794  _(unsure)_

```python
for w in (self._model_box, self._object_box,
```

Keep every hidden helper widget parented but invisible so it doesn't render in the compact layout.

### lines 1804-1805  _(unsure)_

```python
pick_row = QHBoxLayout()
```

File picker row — FOV and channel dropdowns sit immediately LEFT of the Choose control, all three wearing the flat "Live toggle" look.

### lines 1813-1821

```python
self._path_label.setMinimumWidth(0)
```

Explicit rather than load-bearing: QLabel already defaults to a minimum width of 0, and a mutation confirmed removing this changes nothing. It stays as a statement that the label is MEANT to be squeezable, because the thing that actually keeps the row intact is the eliding below -- without which the label's sizeHint is the FULL path, routinely longer than the panel is wide, and the MIP toggle, both spin boxes and the Choose button are pushed past the right edge. On screen that reads as the top-left field overlapping the path, which is how it was reported.

### lines 1828-1831

```python
from .ai_toggle_label import AiToggleLabel
```

Same widget as the AI and Live switches, so the row of toggles reads as one row. Disabled until a folder is found to hold stacks: an enabled control that cannot do anything is worse than an absent one, and the tooltip says which case this folder is.

### lines 1838-1840

```python
self._max_images_box = FlatSpinBox(
```

How many images may be on screen at once. Separate from the set count: that one bounds what is *listed*, this one bounds what is *drawn*, and drawing is what costs memory and redraw time.

### lines 1871-1877

```python
self._set_table = QTableWidget(0, 0, self)
```

This button is why the field dropdown can be hidden below. It was hidden once before on the assumption the table would always be populated, and when enumeration finds no sets -- every folder whose names the configured regex does not match -- that left an empty table and no way at all to choose an image, so it went back. The answer is not a redundant dropdown but this dialog: it opens any file, grouped or not, and pins its set into the table.

### lines 1899-1902

```python
self._cycle_prev_btn = FlatButton("◀", self)
```

CYCLING THE VIEW BETWEEN THE OBJECTS BEING SEGMENTED. With

"cell + nucleus" chosen the panel runs Cellpose on two channels, and the source view could only ever show one of them -- so half of what was being tuned was never on screen.

### line 1924  _(unsure)_

```python
pick_row.addWidget(self._mip_toggle)
```

MIP sits with the set controls it applies to.

### lines 1929-1944

```python
self._offscreen_controls = QWidget(self)
```

The field and channel dropdowns are NOT added. The table picks both, and two controls for one choice can disagree. They stay constructed because apply_sample_to_combo fills one, the saved view state names a field through it, and selected_channel() reads the other. OUT OF THE PANEL, not merely hidden. Both were parented to `self` and never added to a layout, and a widget in that state occupies (0, 0) -- the top left, exactly where the loaded-path label sits. It is only setVisible(False) that keeps it off screen, and that is one stray show() away from a combo box drawn over the path, which is how this was reported twice.

They cannot simply be deleted: apply_sample_to_combo fills the fov box, the saved view state names a field through it, and selected_channel() reads the other. So they keep working and stop being able to appear, by living in a container that is never shown.

### line 1957  _(unsure)_

```python
act = QHBoxLayout()
```

Action row — Run + Live settings + status

### lines 1961-1963

```python
self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT, self)
```

Cancel sits beside Run in every live view, disabled until a pass is in flight. Before the shared contract only this panel could be cancelled at all, and only from Python.

### lines 1973-1974  _(unsure)_

```python
self._view_mode = QComboBox(self)
```

What the right-hand canvas shows: outline overlay, the raw label mask, or the Cellpose flow field.

### lines 1976-1977  _(unsure)_

```python
set_translatable_items(self._view_mode, VIEW_MODES)
```

_refresh_canvases reads the entry's data, so the caption is translated and the mode it names is not.

### line 1992  _(unsure)_

```python
canvas = QHBoxLayout()
```

Twin zoomable canvases in a synchronised pair.

### line 2017  _(unsure)_

```python
self._hover_label = QLabel("Hover over the image to inspect pixels.",
```

Pinned hover info line

### lines 2024-2026

```python
from PySide6.QtWidgets import QSlider
```

Comparison scrubber — scrub back/forth through previous preview runs to compare how different settings changed the segmentation. Hidden until at least two runs exist.

### lines 2044-2045  _(unsure)_

```python
self._live_settings_dialog: Optional["LiveSettingsDialog"] = None
```

Book-keeping for the dialog-based settings surface. Kept as a member so tests + external hooks can introspect / drive it.

## LivePreviewPanel._load_for_display

### lines 2087-2088  _(unsure)_

```python
channel = None
```

Project the channel this file belongs to, not the whole set: the view is showing one channel and the ingest projects per channel.

## LivePreviewPanel._install_loaded_image

### line 2189

```python
projected = None
```

A field that cannot be projected still has an image to show.

### lines 2193-2197

```python
self.cancel_preview()
```

A new image invalidates everything derived from the old one, including the run in flight. The raw masks and the flow images used to survive this, so the next filter change — or an in-flight preview landing a moment later — re-drew the previous image's masks over the new one and raised IndexError as soon as the two differed in size.

## LivePreviewPanel.resizeEvent

### line 2240

```python
pass
```

Cosmetic: a failure here must never stop the panel resizing.

## LivePreviewPanel._refresh_source_selectors

### lines 2261-2265

```python
channels = (int(self._image.shape[2])
```

Channel count from the array when the file holds a channel axis, and from the enumeration when it does not. One .tif per channel is the normal cellvoyager layout, and those arrays are 2-D — reading the count off shape[2] alone meant the dropdown offered nothing but "All channels" for exactly the folders that have the most channels.

### lines 2274-2276

```python
canonical = self._channel_box.currentData()
```

Keyed on the entry rather than the caption: the caption is translated, so a Swedish screen re-selecting by text would find nothing and silently drop back to "All channels".

### lines 2282-2285

```python
self._populate_set_table()
```

Both of these describe the enumeration that just ran, so they belong here rather than at the call sites: this is the one function every load path goes through, which is why hanging them off a caller left the table empty and the MIP switch dead on a real folder drop.

## LivePreviewPanel._regex_config

### line 2311, trailing  _(unsure)_

```python
for _ in range(12):
```

bounded; parents are shallow

## LivePreviewPanel._populate_set_table

### lines 2338-2340

```python
pinned = getattr(self, "_pin_path", None)
```

A set chosen through "Choose image" joins the table even when the random sample did not draw it — otherwise picking a specific field showed it once and then lost it, with no row to click back to.

### lines 2365-2367

```python
text = name if planes <= 1 else f"{name}  ({planes}z)"
```

Say the depth in the cell rather than hiding it: a field with 21 planes and one with 1 looked identical before, and only one of them is affected by the MIP switch.

### lines 2374-2375  _(unsure)_

```python
header = table.horizontalHeader()
```

Fill the width: content-width columns left the table ending mid-panel with dead space beside it.

## LivePreviewPanel._on_set_cell_clicked

### lines 2500-2506

```python
self._fov_box.blockSignals(True)
```

Load the file the cell names, rather than routing through the field dropdown. That dropdown is keyed by SET — one entry per field, at its representative channel — so asking it for channel 2 of a field it lists under channel 1 found nothing and fell through to an async reload, which is why clicking a channel header changed nothing. The cell already knows the exact file; a cell click is not a field change, it is a field-and-channel change.

### line 2511, trailing  _(unsure)_

```python
self._fov_box.setCurrentIndex(index)
```

keeps saved state honest

## LivePreviewPanel._on_mip_toggled

### lines 2548-2549  _(unsure)_

```python
pass
```

Redrawing is best-effort; the next selection change picks the new mode up regardless.

## LivePreviewPanel._on_fov_changed

### lines 2581-2582

```python
picked = self._sampler.set_for_path(path)
```

The loaded file may be a different channel of the very set the combo points at; comparing raw paths would reload it for no reason.

### lines 2592-2595

```python
self.load_source_async(path, enumerate_sets=False)
```

enumerate_sets=False: this path came *out* of the sampler, so the folder is already enumerated. Re-scanning would spend a full pass over the plate to rediscover the listing we are holding, which is the cost the sampling work in 5d5c5c92 removed.

## LivePreviewPanel._apply_display_background

### lines 2691-2694

```python
out = shown.copy()
```

`channel_view` returns a VIEW into `self._image`. Writing zeros through it would destroy the loaded image, so the next render -- and the segmentation worker, which reads the same array -- would see an image already thresholded once, again.

### lines 2699-2700

```python
if getattr(shown, "ndim", 0) != 3:
```

"All channels": each one takes its own object's background, so a composite cannot show a cleaned cell channel beside a raw nucleus.

## LivePreviewPanel.settings_for_propagation

### lines 2775-2780

```python
primary = self._primary_object()
```

The tuned value belongs to the compartment being segmented. With "nucleus" selected the panel runs Cellpose on the nucleus, so writing the result to `cell_diameter` left `nucleus_diameter` untouched and the run used neither the tuned number nor the one on screen. "cell" is the default selection, so the ordinary case is unchanged.

### lines 2784-2790

```python
f"{primary}_model_name": model,
```

AND THE COMPARTMENT'S OWN MODEL FIELD. `model_name` is the

TRAINING module's key; the Mask panel holds one checkpoint per object -- `pathogen_model_name`, `cell_model_name` and so on so propagating only `model_name` left a custom pathogen model chosen in the live preview writing to a field the Mask panel does not show. The user tuned against a zoo checkpoint, pressed Propagate, and the run still used cpsam.

### lines 2794-2796

```python
"pathogen_channel": int(self._pathogen_channel.value()),
```

Propagated like the other two, so tuning a pathogen or organelle channel here reaches the main settings panel instead of being lost when the dialog closes.

### lines 2800-2803

```python
f"{primary}_diameter": self._unclamped(
```

`_unclamped` for the three the panel seeds: a spin box that could not hold what it was given shows the clamp, and handing that back would rewrite the user's setting with the editor's limit. Untouched means unchanged.

### lines 2813-2814  _(unsure)_

```python
if hasattr(self, "_compartment_widgets"):
```

Per-compartment + common tuning settings (only present once the compartment widgets have been built).

## LivePreviewPanel.apply_settings

### lines 2871-2874

```python
try:
```

THE SLOT COUNT FIRST. Everything below reads the primary object, and with `number_of_organelles` at 2 the second slot is not offered at all until the dropdown has been rebuilt -- so seeding before this would seed a panel that cannot represent what it was given.

### lines 2879-2880

```python
for role in organelle_roles(max(1, organelle_count(settings))):
```

Each slot's channel, so switching between them shows what the run will actually use rather than slot 1's value carried across.

### lines 2924-2926

```python
for comp in COMPARTMENTS:
```

Pathogen and organelle are propagated OUT of this panel, so they are read back in as well: a round trip that drops two of its four channels is how the panel came to disagree with the run.

### lines 2928-2929  _(unsure)_

```python
key = (f"{self._active_organelle_role}_channel"
```

The organelle spinner shows the SELECTED slot, which need not be slot 1, so it is seeded from that slot's own key.

### lines 2934-2936

```python
self._seed_organelle_column(settings)
```

The organelle column, from the SELECTED slot's keys with the plain `organelle_` spelling as the fallback -- a settings file written before slots existed carries only the latter.

### lines 2940-2945

```python
if settings.get("adjust_cells") is not None:
```

SEEDED BECAUSE IT IS PROPAGATED. This toggle was written out by `settings_for_propagation` and never read in, so it always sent the unchecked box it was built with -- and Mask ships `adjust_cells` True. Propagating from an untouched preview therefore switched off the adjustment of cell masks by the nucleus and pathogen masks, without the user having touched the control that did it.

### lines 2953-2954  _(unsure)_

```python
self._normalise_check.setChecked(bool(settings["normalize"]))
```

Mask declares a bool; the crop-preview vocabulary allows a

[lo, hi] percentile pair, which is equally "normalise on".

### lines 2959-2969

```python
wanted = str(settings["model_name"])
```

NOT OFFERED IS NOT THE SAME AS NOT ACCEPTED. The live menu drops the pre-SAM spellings, because all four resolve to cpsam and offering them is four labels for one model. But a SAVED settings file naming `cyto2` still has to round-trip: dropping it here would leave the combo on whatever it happened to show, so the preview would quietly use a different model than the settings say -- which is the defect the menu change was meant to reduce, reintroduced at the other end.

The same add-if-missing rule serves a zoo checkpoint whose path was not on disk when the panel was built.

### lines 2973-2978

```python
self._model_box.addItem(wanted)
```

ADDED ONLY IF IT NAMES SOMETHING. A retired alias and a checkpoint on disk are both real answers the menu simply does not offer; a typo is not, and accepting one would put junk in the combo and preview with it. The existing contract that an unknown name is IGNORED is kept -- see test_apply_settings_ignores_none_channels_and_unknown_models.

## LivePreviewPanel.run_preview

### lines 3023-3030

```python
worker.finished.connect(self._on_worker_finished)
```

NOT worker.deleteLater. ``finished`` is emitted from inside the worker thread, so scheduling the object's C++ deletion off it hands Qt a second owner for an object Python already owns, and the two race — see the measured account in spacr.qt.bridge.make_thread (3 crashes in 8 runs of the stress harness). The relay below is a bound method rather than a lambda so Qt can see a receiving QObject with GUI-thread affinity and queues the call onto the GUI thread; a plain closure would be invoked directly on the worker thread.

## LivePreviewPanel._build_compartment_widgets._spin

### lines 3095-3100

```python
raise ValueError(kind)
```

NO `method` KIND ANY MORE (391). Its only field was

`intensity_threshold_method`, whose `mean`/`percentile` choice the run no longer makes -- see the note on COMPARTMENT_FIELDS. Left in place it would be a combo box waiting for the withdrawn setting to be re-added by someone who found the branch and assumed it had a user.

## LivePreviewPanel._build_compartment_widgets

### line 3105  _(unsure)_

```python
try:
```

Pull the informative spaCR setting descriptions for tooltips.

### lines 3111-3112  _(unsure)_

```python
self._common_widgets: Dict[str, QWidget] = {
```

Common controls — one widget each, retargeted to the chosen object at propagation time (see settings_for_propagation).

### lines 3118-3121

```python
self._common_widgets["remove_background"].toggled.connect(
```

Both reach the displayed intensity image, not only the worker, so both have to repaint. Without this the toggle looked inert until the next Run: the pixels it removes were already gone from the segmentation and still on screen.

### lines 3126-3129

```python
self._cell_channel.valueChanged.connect(self._refresh_canvases)
```

Which channel gets thresholded depends on the cell/nucleus channel indices and on which object is selected, so moving any of those has to repaint too -- otherwise pointing "cell" at a different channel leaves the cleaned pixels on the old one.

### lines 3135-3138

```python
self._object_box.currentIndexChanged.connect(
```

AND THE VIEW FOLLOWS THE OBJECT. Choosing "cell" while looking at the nucleus plane left the user tuning cell settings against a nucleus image -- the channel each object is segmented from is already stated in its spinner, so the view can simply follow it.

### lines 3145-3147

```python
_channel_spinner.valueChanged.connect(self._follow_object_channel)
```

NOT `_spin`: that name is a local widget factory further down this same method, and binding over it made the next call to it raise "QSpinBox object is not callable".

### line 3203  _(unsure)_

```python
self._adjust_cells = _spin("bool", None)
```

Cell-only extra.

### lines 3212-3218

```python
try:
```

THE PIPELINE'S OWN DEFAULT PER COMPARTMENT, not one constant for all of them. `COMPARTMENT_FIELDS` carries a single fallback per field, so when the pipeline gave `organelle_min_area` a default of 10 while cell, nucleus and pathogen kept 0, the preview went on showing 0 for the organelle -- and Propagate then wrote that 0 into the run. A preview whose defaults disagree with the run is the fault the per-object filters were unified to remove.

### lines 3240-3241  _(unsure)_

```python
for w in self._all_compartment_widgets():
```

Re-filter the cached masks live whenever any filter widget changes, so tuning updates the preview instantly (no Cellpose re-run).

## LivePreviewPanel._build_compartment_widgets._organelle_widget

### lines 3161-3164

```python
def _organelle_widget(kind, spin_args):
```

the organelle's own segmentation controls

Built here with everything else, hidden, so their values survive an open/close of the dialog the same way the compartment widgets do.

### lines 3171-3173

```python
widget = QComboBox(self)
```

Filled from LEGAL_METHODS whenever the morphology changes:

a method the morphology cannot use is not a choice, and offering it produces a preview that raises.

## LivePreviewPanel._widget_value

### lines 3279-3280

```python
return _combo_value(w)
```

The value, not the caption: a translated caption would land in the settings dict as the setting's value.

## LivePreviewPanel._keys_whose_off_is_none

### lines 3312-3317

```python
cls._OFF_IS_NONE = frozenset(
```

`_intensity_threshold` ADDED WITH THE ABSOLUTE SCHEME (391). `spacr.object.merge_split_filter_masks` defaults it to None and `spacr.utils._merge_by_intensity` reads None as "no threshold was given, so merge nothing and print the boundary intensities you found". A propagated 0 would instead read as a real threshold of zero and merge every pair that touches.

## LivePreviewPanel._compartment_settings

### lines 3353-3355

```python
prefix = (self._active_organelle_role
```

The organelle panel is ONE set of widgets serving whichever slot is selected, so its keys carry that slot's role rather than the generic "organelle" -- otherwise tuning slot 2 wrote slot 1.

### lines 3362-3368

```python
for obj in self._selected_object_types():
```

The common controls are one widget each, retargeted to whatever is selected. Written for EVERY selected object type, not just the primary: with "cell + nucleus" chosen, keying them off `_primary_object()` alone wrote `remove_background_cell` and left the nucleus channel with no key at all, so the segmentation worker which looks up `remove_background_{obj}` per object in its loop silently skipped it and the toggle appeared to do half a job.

### lines 3377-3379

```python
if self._primary_object().startswith("organelle"):
```

The organelle column, under the selected slot's prefix. Only when an organelle is what is being segmented -- otherwise a cell run would propagate a morphology and a method nothing in it reads.

## LivePreviewPanel._seed_organelle_column

### lines 3395-3397

```python
ordered = ["morphology"] + [k for k in widgets if k != "morphology"]
```

The morphology first, because it decides which methods are legal seeding the method against the previous morphology's list would drop it.

## LivePreviewPanel._apply_cycle_stop

### lines 3534-3537

```python
self._composite_roles = ()
```

A single object is the ORDINARY view, driven through the channel dropdown -- so it goes through the same path as every other channel change rather than becoming a second way to show one plane, which could then disagree with the first.

## LivePreviewPanel._follow_object_channel

### lines 3646-3650

```python
written = box.itemData(index)
```

`itemData` holds the entry AS WRITTEN, because the captions are translated -- "All channels" reads "Alla kanaler" on a Swedish screen. Falling back to the caption covers the window before `_localise_channel_combo` has run, when `populate_channel_combo` has added plain items with no data.

## LivePreviewPanel._on_primary_object_changed

### lines 3737-3739

```python
self._cycle_index = 0
```

The stops belong to the new selection, so an index into the old one means nothing -- start at the first object rather than wherever the previous cycle had got to.

## LivePreviewPanel._build_request

### lines 3761-3764

```python
merged = dict(self._settings)
```

One unified settings dict drives both background subtraction

(pre) and filtering (post): the common "remove background" "background" controls and the per-compartment filter values. No more Pre/Post checkboxes — the settings apply whenever they're set.

## LivePreviewPanel._refresh_canvases

### line 3878, trailing  _(unsure)_

```python
elif self._masks:
```

Overlay (default)

## LivePreviewPanel._label_rgb

### lines 3922-3923  _(unsure)_

```python
ids = np.unique(labels[present])
```

Same categorical map the overlay uses, so 'color (random)' means one thing in both views.

### line 3935  _(unsure)_

```python
shade = (0.5 + 0.5 * ((labels % 7) / 6.0)).astype(np.float32)
```

Vary brightness a little per label so neighbours are separable.

## LivePreviewPanel._on_model_or_object_changed

### lines 3963-3964  _(unsure)_

```python
dlg = self._live_settings_dialog
```

Kept as a hook so any observers subscribed to model/object combo changes still fire.

## LivePreviewPanel._on_settings_closed

### lines 3990-3991  _(unsure)_

```python
"""Redraw the canvases after the Live Settings dialog closes.
```

Refresh canvases in case a visual-only setting changed (e.g. outline colour) while the dialog was open.

## LivePreviewPanel._pick_file

### lines 4013-4014

```python
self._pin_path = Path(path)
```

The chosen file may live in a folder the sampler has never seen, so this one does enumerate — off the GUI thread.

## LivePreviewPanel._on_hover

### line 4028  _(unsure)_

```python
if self._image.ndim == 3:
```

Intensities across every channel

### line 4034  _(unsure)_

```python
hits = []
```

Mask hit-tests

## LivePreviewPanel._on_worker_done

### line 4077  _(unsure)_

```python
self._raw_masks = masks
```

Cache the raw masks so filters can be re-applied live, then filter.

## LivePreviewPanel._recompute_masks

### lines 4097-4099

```python
self._roll_auto_outline_colours()
```

A new run gets a new 'auto' colour. Re-rolling here rather than on every repaint keeps the outline steady while the user drags thickness or percentile sliders.

## LivePreviewPanel._snapshot_run

### line 4136

```python
if len(self._history) > 50:
```

Cap the history so memory stays bounded on long tuning sessions.

### line 4143, trailing  _(unsure)_

```python
self._compare_slider.setValue(n - 1)
```

newest

## LivePreviewPanel._on_compare_scrub

### lines 4158-4161

```python
overlay = overlay_masks(
```

The random and auto modes have to be forwarded here too. They were not, so scrubbing back through history repainted every outline in the per-compartment default — green for cells — whatever the user had chosen.

## LiveSettingsDialog.__init__

### lines 4255-4256

```python
panel._live_settings_dialog = self
```

So a morphology change can re-gate the rows: the widgets live on the panel, and it is this dialog that knows which rows they sit on.

### line 4261  _(unsure)_

```python
for w in self._managed_widgets():
```

Show the widgets we'll be adding, then re-hide them on close.

### lines 4265-4266  _(unsure)_

```python
panels_row = QHBoxLayout()
```

Row of side-by-side panels: the segmentation + common controls on the left, then one greyed-until-chosen panel per compartment to the right.

### lines 4272-4276

```python
model_row = QWidget(seg_group)
```

THE MODEL ROW CARRIES THE ZOO BUTTON, the same as every object model name on the settings panel. Without it the live view offered cpsam and nothing else, so a zoo model could be selected for the RUN and not for the PREVIEW -- which is the preview showing a different model than the run will use, while the user tunes against it.

### line 4302  _(unsure)_

```python
panel._common_widgets["signal_to_noise"].show()
```

Common controls — apply to whichever object is chosen.

### line 4311  _(unsure)_

```python
self._compartment_groupboxes: Dict[str, QGroupBox] = {}
```

One panel per compartment, greyed unless it's the chosen object.

### lines 4326-4332

```python
self._organelle_group = QGroupBox("Organelle segmentation")
```

THE ORGANELLE'S OWN COLUMN, one section wider than the rest.

Every other compartment is a Cellpose object and the generic filters are all it has. An organelle is dispatched by morphology and method into a different routine with its own half-dozen knobs, and none of those were reachable here -- so the only organelle setting that could be previewed was the Cellpose model.

### lines 4344-4345  _(unsure)_

```python
row_host = QWidget()
```

Wrap the (wide) panel row in a horizontal scroll area so it fits on screen no matter how many compartments are shown.

### lines 4354-4355  _(unsure)_

```python
buttons = QDialogButtonBox(QDialogButtonBox.Close)
```

Run button lives in the dialog so settings can be iterated without closing it — edit a value, hit Run, see the result, repeat.

### lines 4361-4363

```python
self._propagate_btn = QPushButton("Propagate settings")
```

Propagate toggle — when on (blue, like the AI / Live toggles), edits here are pushed into the main settings panel so tuning in the live preview updates the run configuration.

### lines 4376-4377  _(unsure)_

```python
panel._object_box.currentTextChanged.connect(self.refresh_visibility)
```

Re-gate the form whenever the object type or model changes, so irrelevant settings grey out live.

### lines 4382-4383  _(unsure)_

```python
self._propagate_sources = [
```

Widgets whose changes propagate to the main panel while the toggle is on — the segmentation controls plus every compartment/common knob.

### lines 4392-4395

```python
try:
```

Open wide enough to show the Segmentation panel + all four compartment panels without the user having to drag the window wider. Clamp to the available screen so it still fits on small displays (the horizontal scroll area handles any remaining overflow).

## LiveSettingsDialog._on_propagate_toggled

### line 4418, trailing  _(unsure)_

```python
self._panel.propagate_settings()
```

push current values now

## LiveSettingsDialog.refresh_visibility

### lines 4480-4484

```python
p._diameter.setEnabled(True)
```

model: cpsam uses all three. `diameter` used to be disabled here with the tooltip "Ignored by Cellpose-SAM", which was false: Cellpose 4 rescales the image by 30/diameter before it runs (see DIAMETER_TOOLTIP for the measured counts), so the UI was greying out a control that changes the result.

### lines 4497-4500

```python
ordered = list(p._selected_object_types())
```

compartment panels: show only the primary object's panel plus, for 'cell + nucleus', a secondary Nucleus panel. The other compartments' panels are hidden entirely (their settings are the same shape and only the chosen object's are relevant).

### lines 4514-4516

```python
organelle_primary = primary.startswith("organelle")
```

the organelle column: only for an organelle, and within it only the knobs the chosen morphology actually reads. Showing all twenty-one at once is a wall nobody can tune.

### lines 4530-4532

```python
p._normalise_check.setEnabled(True)
```

Normalisation is always available (independent of the Pre step and of the model, incl. cpsam). The percentile bounds only apply while normalisation is on.

### lines 4541-4542  _(unsure)_

```python
for w in (p._outline_colour, p._outline_thickness):
```

Overlay / outline knobs are always relevant (they style the overlay view), so they stay enabled.

## overlay_mask

### lines 4595-4597  _(unsure)_

```python
def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
```

Back-compat shims for callers that predate the multi-object rewrite

## LiveSettingsDialog._show_every_control_on_a_row

### added 2026-09-19 (431)

```python
self._show_every_control_on_a_row()
```

Reported by the maintainer on 2026-09-19: "in mask generation live settings, with per object settings on and pathogen chosen i dont have the option to choose pathogen channel in the live settings, only in the per object settings." Measured on a built Mask screen: the first open of Live settings showed a spin box beside "Pathogen channel"; every later open showed the caption over an empty field, and the same for "Organelle channel". `closeEvent` hides each borrowed control as it hands it back, which sets `WA_WState_ExplicitShowHide`, and a widget hidden that way stays hidden when the next dialog's form takes it. `__init__` re-showed only what `_managed_widgets()` named, and that list did not name those two. The per-object table was incidental: the defect is in the dialog, and the table was simply the only other place the pathogen channel could be set.

Two changes. `_managed_widgets()` now names both spin boxes, and so does `_propagate_sources`, so a pathogen channel changed here with Propagate on reaches the main form without waiting for some other control to move. And this sweep shows every widget on every form row, so a control added to the dialog later cannot come back hidden with no list to forget. Rows gated on purpose (the organelle morphology rows) are gated by `QFormLayout.setRowVisible` in `refresh_visibility`, which runs after this and is unaffected.

The existing check, `tests/qt/test_live_preview_channels.py::test_the_dialog_shows_a_row_for_each_channel`, read the row's CAPTION, which was there both times; it passed throughout. `tests/qt/test_live_settings_keep_every_channel_on_reopen.py` asks the spin box.

## LivePreviewPanel._set_table_columns

### added 2026-09-19 (431)

```python
columns = self._set_table_columns(sets)
```

GitHub issue #119 (jak18015, 1.5.0.8, macOS): "Columns for channels don't all show up reliably and images are all under a column called 'ch' and there are no individual channel columns". Three defects, each measured on a built Mask screen with the preview open and `src` set to a folder:

* A file whose name the naming dialect cannot read is enumerated with channel ID `""`, and the header was `f"ch {c}"` -- a column captioned "ch ". A folder of three-channel TIFFs in any naming other than the form's therefore showed every file under that one column while the channel dropdown beside it offered Ch 0, Ch 1 and Ch 2. Such a column is now captioned "image", and when no file name carries a channel at all and the loaded image has several planes on its last axis, each plane gets its own "ch N" column.
* The columns were the channel IDs of the SAMPLED sets. A channel that only some fields have appeared or not with the random draw -- "don't all show up reliably". They now come from `ImageSetSampler.channels`, which is the whole folder.
* The plane-column cap is the channel spin boxes' maximum plus one (nine): the segmentation channels cannot name a plane beyond that, and a last axis longer than that is more likely a stack read the other way round than a channel axis.

## LivePreviewPanel._open_cell

### added 2026-09-19 (431)

```python
self._open_cell(item)
```

A plane column's cell carries the plane in `_PLANE_ROLE` beside the path in `Qt.UserRole`. The file is read only when it is not the one on screen, so moving along a row of a multi-channel file changes the plane shown and reads nothing; the plane is chosen through the same display-channel dropdown a user picks from, and `_on_display_channel_changed` redraws.

## LivePreviewPanel.regroup_the_folder

### added 2026-09-19 (431)

```python
lambda: enumerate_image_sets(folder, SUPPORTED_SUFFIXES,
```

The grouping is decided when a folder loads, with the `metadata_type` and `custom_regex` the Mask form held at that moment. Loading a cellvoyager folder while the form said `cq1` and then choosing `cellvoyager` left all 92 files under one column; nothing re-read the folder until another image load missed the sampler's cache. The screen now calls this 400 ms after either setting changes (`AppScreen._wire_live_preview_naming`). The enumeration reads file names only and runs through the panel's `JobRunner`, and `adopt` files it under the dialect's cache key, so `_refresh_source_selectors` finds it rather than scanning the plate again on the GUI thread.

Seen and not changed: `load_source_payload` still enumerates on the worker with the DEFAULT dialect, so a folder opened under any other naming is scanned a second time on the GUI thread by `_refresh_source_selectors`. The result is right; only the cost is paid twice.

## LivePreviewPanel._adopt_the_regrouping

### added 2026-09-19 (431, from review)

```python
if tuple(self._regex_config()) != (meta, custom):
```

Found in review of 431. The token bumps only when the next regrouping is asked for, and the screen asks 400 ms after the naming changes. If the naming changes again and the running job finishes inside that wait, the token still matches, so the grouping read under the old naming was adopted. `_refresh_source_selectors` then read the NEW naming off the form, missed the sampler's cache and re-read the whole folder on the GUI thread, which is the cost the off-thread regroup exists to avoid. Arrow-keying through the `metadata_type` combo on a large plate does exactly that. A grouping whose naming is no longer the form's is now dropped, and the regrouping the timer asks for replaces it. Held by `test_a_regrouping_read_under_a_naming_since_changed_is_dropped`, which counts GUI-thread folder reads and fails without the check.

## first_supported_image

### added 2026-09-19 (431, #119)

```python
if name.startswith("."):
```

`._<name>.tif` sorts before every image under `str.casefold`, because `.` sorts before letters and digits. On a folder with macOS sidecars the preview's first file was therefore a sidecar. Measured on a synthetic cellvoyager folder with a sidecar beside each file: tifffile raised "not a TIFF file: header=b'\x00\x05\x16\x07'", no image loaded, and the table never filled. See the note on `enumerate_image_sets` in `preview_controls.md`. Held by `test_a_folder_on_an_exfat_drive_previews_its_images` and `test_the_listing_helpers_skip_macos_sidecars`.
