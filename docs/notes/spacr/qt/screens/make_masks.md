# Notes from `spacr/qt/screens/make_masks.py`

Prose lifted out of `spacr/qt/screens/make_masks.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (8 entries)
- [_MaskCanvas](#_maskcanvas) (3 entries)
- [_MaskCanvas.__init__](#_maskcanvas__init__) (8 entries)
- [_MaskCanvas.set_image_and_mask](#_maskcanvasset_image_and_mask) (2 entries)
- [_MaskCanvas.refresh](#_maskcanvasrefresh) (1 entry)
- [_MaskCanvas._canvas_to_image](#_maskcanvas_canvas_to_image) (3 entries)
- [_MaskCanvas.paintEvent](#_maskcanvaspaintevent) (2 entries)
- [_MaskCanvas._paint_recrop_boxes](#_maskcanvas_paint_recrop_boxes) (1 entry)
- [_MaskCanvas.mousePressEvent](#_maskcanvasmousepressevent) (3 entries)
- [_MaskCanvas.mouseMoveEvent](#_maskcanvasmousemoveevent) (3 entries)
- [_MaskCanvas.mouseReleaseEvent](#_maskcanvasmousereleaseevent) (3 entries)
- [_MaskCanvas._finish_region_gesture](#_maskcanvas_finish_region_gesture) (2 entries)
- [flow_rgb](#flow_rgb) (2 entries)
- [load_cellpose_model](#load_cellpose_model) (1 entry)
- [cellpose_detect](#cellpose_detect) (2 entries)
- [_FlowPane.__init__](#_flowpane__init__) (1 entry)
- [_FlowPane.show_rgb](#_flowpaneshow_rgb) (1 entry)
- [FoldedModulePanel.__init__](#foldedmodulepanel__init__) (2 entries)
- [NapariBridgeScreen.__init__](#naparibridgescreen__init__) (2 entries)
- [NapariBridgeScreen.open_in_napari](#naparibridgescreenopen_in_napari) (1 entry)
- [NapariBridgeScreen.take_mask_back](#naparibridgescreentake_mask_back) (2 entries)
- [MakeMasksScreen.__init__](#makemasksscreen__init__) (2 entries)
- [MakeMasksScreen._build_ui](#makemasksscreen_build_ui) (6 entries)
- [MakeMasksScreen._restate_fold_button](#makemasksscreen_restate_fold_button) (1 entry)
- [MakeMasksScreen._build_folded_screen](#makemasksscreen_build_folded_screen) (2 entries)
- [MakeMasksScreen.seed_folded](#makemasksscreenseed_folded) (1 entry)
- [MakeMasksScreen._build_tool_row](#makemasksscreen_build_tool_row) (6 entries)
- [MakeMasksScreen._on_toggle_settings](#makemasksscreen_on_toggle_settings) (1 entry)
- [MakeMasksScreen._build_tools_panel](#makemasksscreen_build_tools_panel) (5 entries)
- [MakeMasksScreen._install_shortcuts](#makemasksscreen_install_shortcuts) (1 entry)
- [MakeMasksScreen._on_wand_salvage_changed](#makemasksscreen_on_wand_salvage_changed) (1 entry)
- [MakeMasksScreen._on_undo](#makemasksscreen_on_undo) (1 entry)
- [MakeMasksScreen._on_detect_otsu](#makemasksscreen_on_detect_otsu) (1 entry)
- [MakeMasksScreen._build_view_tabs](#makemasksscreen_build_view_tabs) (1 entry)
- [MakeMasksScreen._build_cellpose_card](#makemasksscreen_build_cellpose_card) (1 entry)
- [MakeMasksScreen.run_cellpose](#makemasksscreenrun_cellpose) (2 entries)
- [MakeMasksScreen._warn](#makemasksscreen_warn) (1 entry)
- [MakeMasksScreen._should_background_load](#makemasksscreen_should_background_load) (1 entry)
- [MakeMasksScreen._handle_load_failure](#makemasksscreen_handle_load_failure) (1 entry)
- [MakeMasksScreen._apply_loaded_pair](#makemasksscreen_apply_loaded_pair) (4 entries)
- [MakeMasksScreen._open_ledger](#makemasksscreen_open_ledger) (1 entry)
- [MakeMasksScreen._on_recrop_requested](#makemasksscreen_on_recrop_requested) (1 entry)
- [MakeMasksScreen.recrop](#makemasksscreenrecrop) (3 entries)
- [MakeMasksScreen.finish_recrop](#makemasksscreenfinish_recrop) (2 entries)
- [MakeMasksScreen._on_prev](#makemasksscreen_on_prev) (1 entry)
- [MakeMasksScreen._on_next](#makemasksscreen_on_next) (1 entry)
- [MakeMasksScreen._on_stroke_started](#makemasksscreen_on_stroke_started) (1 entry)
- [MakeMasksScreen._sync_button_states](#makemasksscreen_sync_button_states) (1 entry)

## Module level

### lines 189-193

```python
"model_compare": (
```

STABLE, not alpha: `spacr.qt.maturity` promoted both at launch on the evidence in its own table, and it is the promoted stage the tile lit in. A fallback copied from `app.py`'s literal records the colour before that rewrite, which is a button lighting green-cyan where the tile it replaced lit blue.

### lines 209-211

```python
"napari_bridge": (
```

THE ONLY SOURCE, not a fallback: the bridge registered its own row until the screen folded in here, so nothing puts one in the registry any more and this is what the button reads.

### line 225  _(unsure)_

```python
_HEADLESS_PLATFORMS = ("offscreen", "minimal", "minimalegl", "vnc")
```

Qt platform plugins that have no way for a human to click a dialog button.

### lines 256-258  _(unsure)_

```python
MODE_NONE = "none"
```

Canvas — image + mask overlay with brush/erase mouse handling

### lines 295-299

```python
(MODE_DRAW,         "Draw",         "draw"),
```

THE TWO REGION TOOLS SIT BESIDE THE WAND, not after Zoom. All three answer the same question -- which pixels are one object -- where brush and erase answer it a pixel at a time, and Zoom is not a tool for changing a mask at all. Reaching the row through the fallback put them last in alphabetical order; named here they are placed.

### lines 303-306

```python
(MODE_RECROP,       "Recrop",       "recrop"),
```

RECROP IS LAST, past the tools that change a mask, because it is not one of them: every button left of it edits the field in view, and this one replaces the field in view with the several fields it should have been. Beside Divide it would read as another way to split an object.

### lines 336-340

```python
PAN_MODIFIERS = Qt.ShiftModifier | Qt.AltModifier
```

Held with the left button, these pan from ANY tool. Two of them because window managers eat one or the other: Alt+drag moves the window on most Linux desktops, and Shift+drag is taken by some tablet drivers. Whichever one survives on this machine, panning still works without putting the brush down.

### lines 1187-1189  _(unsure)_

```python
CELLPROB_THRESHOLD = 0.0
```

Cellpose-SAM: the segmentation, and its two intermediate outputs

## _MaskCanvas

### line 419, trailing  _(unsure)_

```python
stroke_started = Signal()
```

emitted just before self.mask is mutated

### line 420, trailing  _(unsure)_

```python
stroke_finished = Signal()
```

emitted after a stroke completes

### line 421, trailing  _(unsure)_

```python
zoom_changed = Signal(bool)
```

emitted with True when zoom entered / False on reset

## _MaskCanvas.__init__

### line 431, trailing  _(unsure)_

```python
self.image: Optional[np.ndarray] = None
```

uint16 grayscale

### line 432, trailing  _(unsure)_

```python
self.mask: Optional[np.ndarray] = None
```

uint8 labels

### lines 457-459

```python
follow_device_ratio(self, self.refresh)
```

The field is composited once and stays up between edits, and a window dragged to another screen fires no resize -- so the recomposite has to be asked for.

### line 468  _(unsure)_

```python
self._zoom_x0: Optional[int] = None
```

Zoom viewport in image coords; None = full-image view.

### lines 474-477

```python
self._zoom_drag_start: Optional[QPoint] = None
```

Zoom-rectangle drag state (widget-local pixel coords). The recrop box is dragged the same way and reuses them, so the two rectangle tools cannot get out of step with each other; which one is being aimed is `self.mode`.

### lines 488-490

```python
self._gesture_points: List[QPoint] = []
```

The draw outline / divide line in flight, in widget coords. Both gestures change nothing until the button comes up, so the path is collected here and converted to image pixels once, on release.

### lines 500-501  _(unsure)_

```python
self._sweeping = False
```

Right-button sweep-delete: one gesture, one undo step, one ledger entry naming every object it took out.

### line 505  _(unsure)_

```python
self._pan_from: Optional[QPoint] = None
```

Shift/Alt + left-drag pan, in widget coords.

## _MaskCanvas.set_image_and_mask

### lines 519-522

```python
self._gesture_points = []
```

A gesture belongs to the field it was started on. The arrow keys move to the next field from anywhere, including the middle of a traced outline, and the points collected on the old field name nothing on the new one.

### lines 524-525

```python
self.recrop_boxes = []
```

The boxes belong to the field they were cut out of; on the next field they would be rectangles drawn over unrelated pixels.

## _MaskCanvas.refresh

### lines 572-576

```python
pixmap = scaled_for(pixmap, self, avail_w, avail_h)
```

Composited at the panel's real pixel density. Everything below that maps a mouse position onto this picture therefore asks `logical_size`, not `pixmap.width()`: the two differ by the device pixel ratio, and a drawn outline that is out by that factor lands on the wrong object.

## _MaskCanvas._canvas_to_image

### lines 580-582  _(unsure)_

```python
def _canvas_to_image(self, x: float, y: float) -> Optional[tuple]:
```

Coordinate mapping (widget-local px  ↔  full image px)

### lines 584-585

```python
"""Widget coordinates to IMAGE pixel coordinates, or ``None``.
```

NB: QLabel.pixmap() returns a *null* QPixmap (never None) when no pixmap is set, so the emptiness test has to be isNull().

### line 612  _(unsure)_

```python
img_x = max(0, min(self.mask.shape[1] - 1, img_x))
```

Clamp to image bounds

## _MaskCanvas.paintEvent

### lines 775-777  _(unsure)_

```python
def paintEvent(self, event):
```

Painting (adds a zoom-rectangle overlay while dragging)

### lines 786-789

```python
self._paint_recrop_boxes()
```

The boxes already cut are drawn under everything else and in every mode: they are the record of what this field has already given up, and they have to be visible while the next box is being aimed as well as after the tool has been put down.

## _MaskCanvas._paint_recrop_boxes

### lines 816-821

```python
rendered = self.pixmap()
```

The pixmap is checked here rather than per box, because it is what every box is mapped through: a paint that arrives before refresh() has composited anything (a resize on a screen that has not loaded a field yet) has nothing to place a rectangle against, and boxes placed at the widget origin instead would each be a blue square over an object they name nothing about.

## _MaskCanvas.mousePressEvent

### lines 967-971

```python
self._gesture_points = [event.position().toPoint()]
```

No stroke is opened here: neither tool touches the mask until the button comes up, and an outline that encloses nothing or a line that separates nothing must leave no undo step and no ledger entry behind it — the same rule the sweep-delete follows in :meth:`_sweep_delete_at`.

### lines 994-997

```python
self._emit_stroke_end(
```

The report goes in the ledger with the click: which way the flood leaked, what tolerance the rescue settled on and whether the budget stopped it are the reasons the wand took what it took, and a mask nobody can explain is a mask nobody trusts.

### line 1005  _(unsure)_

```python
radius = self._mask_radius_for_brush()
```

Brush / erase strokes

## _MaskCanvas.mouseMoveEvent

### lines 1025-1027

```python
if (dx or dy) and self.pan_by(dx, dy):
```

Only re-anchor once the drag has actually moved the view:

discarding sub-pixel drags instead of accumulating them is what makes a slow pan at high zoom stall completely.

### lines 1041-1042

```python
self._gesture_points = [self._gesture_points[0], now]
```

A divide is one straight cut, so the drag moves the far end of the line instead of adding a bend to it.

### lines 1052-1054

```python
self._emit_stroke_start()
```

A drag that *began* outside the pixmap never fired stroke_started, so without this the resulting edit would never be pushed onto the undo history. Idempotent.

## _MaskCanvas.mouseReleaseEvent

### lines 1072-1073

```python
self._emit_stroke_end(kind="sweep_delete", target=list(labels),
```

ONE entry for the whole sweep. Six deletes in the ledger would say six decisions were made; the user made one.

### line 1084  _(unsure)_

```python
p0 = self._canvas_to_image(self._zoom_drag_start.x(),
```

Convert both endpoints to image coords and commit

### lines 1092-1095

```python
if p0 is not None and p1 is not None:
```

Handed on rather than acted on, and handed on even when it is obviously too small: the screen owns the refusal, so the user gets the same sentence for every box that will not be cut instead of silence for some of them.

## _MaskCanvas._finish_region_gesture

### lines 1138-1142

```python
image_points = [p for p in
```

Points that left the pixmap mid-drag are dropped rather than clamped to its edge, which would drag the outline onto the border of the image. A path that lost every point this way arrives as an empty list and is refused below by the same guards that refuse a click: two points do not make a cut, three do not make an outline.

### lines 1157-1159

```python
self._emit_stroke_end(
```

The ledger names both ends of the split: which object was cut and which id the piece that came off it was given, so a later reader can follow one object through the division.

## flow_rgb

### lines 1276-1280

```python
if array.ndim == 3 and array.shape[0] == 2:
```

THE VECTOR SHAPE IS TESTED FIRST. A `(2, H, W)` field also satisfies "three dimensions with at least three along the last one" whenever the image is three pixels wide or more, so testing for a picture first slices the vectors as though they were one and produces a 2-pixel-tall smear.

### lines 1282-1283  _(unsure)_

```python
dy, dx = stretch_to_uint8(array[0]), stretch_to_uint8(array[1])
```

(dY, dX) -> two colour channels plus their magnitude, each stretched on its own so a weak field is still visible.

## load_cellpose_model

### lines 1338-1340

```python
from ...accelerator import cellpose_kwargs
```

gpu= AND device= from the one resolver. Mask Generation is the module people open first, so leaving it CUDA-only while every other entry point took any accelerator was the confusing half-state.

## cellpose_detect

### lines 1387-1390

```python
try:
```

Cellpose has removed eval arguments between minor versions (4.2 has no `invert`, 3.x had no `max_size_fraction`). Offering only what THIS install accepts keeps the screen working across the versions spaCR supports instead of raising TypeError on the one it was written on.

### lines 1397-1401

```python
kwargs = {k: v for k, v in kwargs.items() if k in params}
```

Only filter against a signature that LISTS what it takes. An eval declared `(self, x, **kw)` names nothing, and filtering against it drops every setting the user chose while the run still succeeds -- the thresholds on the panel would then do nothing at all, silently.

## _FlowPane.__init__

### lines 1439-1441

```python
follow_device_ratio(self, self._rescale)
```

The flow picture is composited once per run and then left up, so a move onto a denser screen has to redraw it or it stays soft for the rest of the session.

## _FlowPane.show_rgb

### lines 1449-1451

```python
image = QImage(data.data, width, height, 3 * width,
```

The QImage borrows the buffer, so it is copied before `data` goes out of scope and the pixmap is left pointing at freed memory — which shows up as a garbled pane, not as a crash.

## FoldedModulePanel.__init__

### lines 1557-1559

```python
button.clicked.connect(
```

The bool ``clicked`` emits is swallowed here rather than in every callback: these are the host's own methods, and one that took a stray positional would fail only when pressed.

### lines 1564-1565

```python
self.buttons.setVisible(bool(self.actions))
```

An empty row would be a strip of padding under the module saying nothing; it appears the moment there is a button to put in it.

## NapariBridgeScreen.__init__

### lines 1673-1674

```python
mark_surface(self.status)
```

The log IS this screen's body — nothing is behind it — so it keeps a surface where the sweep would leave it see-through.

### lines 1677-1678  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## NapariBridgeScreen.open_in_napari

### line 1783

```python
self.say(str(exc))
```

The one refusal that is an instruction rather than an error.

## NapariBridgeScreen.take_mask_back

### lines 1815-1817

```python
self.say(str(exc))
```

Every refusal `to_spacr_mask` raises is written to be read by the person who has to act on it, so it is shown verbatim rather than replaced with a house apology.

### lines 1828-1830

```python
self._handoff = self._reloaded(result)
```

The handoff now holds what is on disk, so pressing the button twice reports "unchanged" rather than recording the same edit a second time.

## MakeMasksScreen.__init__

### line 1920  _(unsure)_

```python
try:
```

Drag & drop — accepts a folder of images to fine-tune against.

### lines 1927-1930

```python
from .settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box you type in. Built here on the field, it is moved onto the label as the last step, so every panel in the application explains itself the same way.

## MakeMasksScreen._build_ui

### lines 1942-1943  _(unsure)_

```python
self._header = ModuleHeader(
```

Masthead — the module's own name and blurb, the folder in force, and the strip of modules that fold into this one.

### lines 1952-1954

```python
self._src_label.setSizePolicy(QSizePolicy.Maximum,
```

A deep folder path must never widen the window or push the fold buttons off the end of the row: the label may shrink below its ideal width, and the tooltip carries what is cut off.

### lines 1964-1968

```python
self._tool_row = self._build_tool_row()
```

The one row of tools, across the top of the body. It is above the canvas and the settings both, so the settings toggle at its far end cannot hide the button that brings the settings back. `_tool_row` is the scroller the row rides in; the row itself is `_tool_row_layout`.

### lines 1998-2001

```python
self._settings_scroll = QScrollArea()
```

THE SETTINGS, AS ONE GROUP. Everything the settings button toggles is inside this one scroll area, so hiding them is one call and the canvas — the splitter's other child — takes the width they give up.

### lines 2012-2013  _(unsure)_

```python
self._body_stack.currentChanged.connect(self._sync_tool_row_visibility)
```

The row belongs to the editor, not to the empty state: there is nothing to brush before a folder is open.

### line 2019  _(unsure)_

```python
nav = QWidget()
```

Bottom nav bar

## MakeMasksScreen._restate_fold_button

### lines 2096-2098

```python
button.style().unpolish(button)
```

A property the stylesheet selects on is only read at polish, so a button already on screen keeps the old colour until it is polished again.

## MakeMasksScreen._build_folded_screen

### lines 2136-2138

```python
screen.compare_requested.connect(self._on_zoo_compare_requested)
```

The zoo's "compare these two" hand-off is wired by whoever hosts it. Folded, that is this screen, or the button would select two models and open nothing.

### lines 2146-2149

```python
from .app_screen import AppScreen
```

A module with no screen of its own gets the generic settings page — the same page its tile opened. Every key this screen folds today has a screen; this is what the next one gets if it does not.

## MakeMasksScreen.seed_folded

### lines 2229-2230  _(unsure)_

```python
screen.apply_settings_dict({"src": self._folder})
```

A module with no screen of its own: a settings page, whose one path is the folder this screen already has open.

## MakeMasksScreen._build_tool_row

### lines 2376-2378  _(unsure)_

```python
def _build_tool_row(self) -> QWidget:
```

The toolbar row and the settings toggle

### lines 2425-2426

```python
self._btn_recrop.setToolTip(RECROP_TOOLTIP)
```

The one tool in the row whose result is not on the canvas, so it is the one that has to say what it does before it is pressed.

### lines 2429-2431

```python
row.addWidget(Divider(Qt.Vertical))
```

Reset zoom, undo and redo ride in the same row: they are pressed between strokes, so hiding them with the settings would hide the two buttons a correction session leans on hardest.

### lines 2463-2465

```python
self._btn_settings.setChecked(True)
```

Checked before it is connected: the settings start on screen and the toggle starts lit, and neither half announces a change that did not happen.

### lines 2470-2477

```python
scroller = QScrollArea()
```

A ROW THAT CANNOT FORCE THE WINDOW WIDER THAN THE DISPLAY. Measured with every tool in it, the row asks for well over 1300px, and a layout minimum that large is not a wide toolbar — it is a window that refuses to be narrowed, so the canvas and the settings go off the right edge with it on a 1366px laptop. Inside a scroll area the row keeps its natural width and the viewport gives up first: a scrollbar on a narrow display, and on a wide one the whole set visible at once, which is the point.

### lines 2485-2488

```python
scroller.setFixedHeight(
```

The bar's own height plus room for the scrollbar that appears when it does not fit: reserved always, so the row does not grow a pixel taller the moment a tool is added and shove the canvas down with it.

## MakeMasksScreen._on_toggle_settings

### lines 2538-2541

```python
sizes = splitter.sizes()
```

A splitter that has never been laid out reports zero for everything; splitting nothing gives the panel a negative width and Qt clamps it to a pane the user cannot see. Fall back to the widths it was born with.

## MakeMasksScreen._build_tools_panel

### line 2557  _(unsure)_

```python
brush_card = Card(title="Brush")
```

Brush size slider

### lines 2644-2647

```python
runaway = QGroupBox("Trim a runaway flood")
```

The three rescues for a flood that escapes down a bright seam. Grouped and defaulted so the panel does not open as a wall of knobs: the group's own checkbox is the master switch, and the numbers under it only matter when the detector misjudges an image.

### line 2790  _(unsure)_

```python
norm_card = Card(title="Display")
```

Display card — contrast percentiles and wheel-zoom speed.

### lines 2794-2797

```python
self._norm_lo.setDecimals(PERCENTILE_DECIMALS)
```

setDecimals BEFORE setRange/setValue: a QDoubleSpinBox rounds both to the precision it has at the time, so setting 99.9999 against the default two decimals stores 100.0 and the control looks broken rather than imprecise.

### lines 2923-2927

```python
for _mode in ("replace", "merge"):
```

THE MODE IS THE ITEM'S DATA, NOT ITS LABEL. `replace` and `merge` are shown to the user and a language switch rewrites the item text in place; reading the mode back off that text would hand `engine.combine_masks` a translated word it has never heard of, so the untranslated key travels with the item instead.

## MakeMasksScreen._install_shortcuts

### lines 2967-2968  _(unsure)_

```python
QShortcut(QKeySequence("R"), self, lambda: self._set_mode(MODE_RECROP))
```

R for recrop. Free: B/E/W/D/V/Z are the other six tools and

Ctrl+S / Ctrl+Z / Ctrl+Y / Escape / the arrows are the rest.

## MakeMasksScreen._on_wand_salvage_changed

### lines 3040-3042

```python
def _on_wand_salvage_changed(self, on: bool):
```

Rescue controls. Each writes one canvas attribute; the canvas builds the dict the flood reads in wand_rescue_settings(), so a control is wired by setting the attribute it names and nothing else.

## MakeMasksScreen._on_undo

### lines 3165-3168

```python
changed = self._diff(self._canvas.mask, prev)
```

Diffed against what is ON the canvas, not against the history head: undo() has already popped, so the head IS `prev` by now and comparing the two would measure every undo as having changed nothing — which is exactly how they went unrecorded.

## MakeMasksScreen._on_detect_otsu

### lines 3303-3305

```python
self._status_label.setText(
```

Replacing with nothing would silently wipe the mask on a flat field, or on one where the minimum area rejected everything. Clearing a mask is what the Clear button is for, and it asks.

## MakeMasksScreen._build_view_tabs

### lines 3330-3332  _(unsure)_

```python
def _build_view_tabs(self) -> QTabWidget:
```

Cellpose-SAM on the open field, and its two intermediates

## MakeMasksScreen._build_cellpose_card

### lines 3398-3401

```python
for name in cellpose_model_choices():
```

THE NAME IS THE ITEM'S DATA, not its label, for the same reason the replace/merge combo carries its mode that way: a language switch rewrites item text in place, and Cellpose has never heard of a translated model name.

## MakeMasksScreen.run_cellpose

### lines 3536-3538

```python
app.processEvents()
```

The button is disabled first, so painting the status line cannot let a second click start a second run on top of this one.

### lines 3564-3565

```python
self._status_label.setText(
```

Replacing with nothing would wipe a mask the user may have spent an hour on, over a threshold that was one notch out.

## MakeMasksScreen._warn

### lines 3598-3600  _(unsure)_

```python
def _warn(self, title: str, text: str) -> None:
```

User messaging (headless-safe — see :func:`is_headless`)

## MakeMasksScreen._should_background_load

### line 3696  _(unsure)_

```python
return False
```

Let the real loader report corrupt/unreadable inputs.

## MakeMasksScreen._handle_load_failure

### lines 3771-3772

```python
self._canvas.image = None
```

Leaving the previous field visible while _current_index names the failed file would let Save write the old mask under a new filename.

## MakeMasksScreen._apply_loaded_pair

### lines 3794-3796

```python
self._recrop_children = []
```

In lockstep with the canvas clearing its own boxes: the cuts belong to the field they were made on, and carrying them onto the next one would retire the wrong file.

### lines 3798-3800

```python
self._reset_flow_panes()
```

The probability and flow panes described the LAST field's

Cellpose run; on this one they would be a picture of the wrong image with nothing on screen saying so.

### line 3802

```python
self._history.clear()
```

Reset undo history for the new image and seed with the loaded mask

### lines 3812-3813

```python
self.apply_object_filter(on_load=True)
```

Last, so its status message and its undo step sit on top of the freshly seeded history rather than being wiped by it.

## MakeMasksScreen._open_ledger

### lines 3830-3832

```python
LOG.warning("Unreadable curation ledger beside %s: %s",
```

A damaged sidecar must not cost the user the edit they are about to make. Start a fresh log, and say so rather than quietly overwriting a record nobody can read.

## MakeMasksScreen._on_recrop_requested

### lines 3841-3843  _(unsure)_

```python
def _on_recrop_requested(self, x0: int, y0: int, x1: int, y1: int) -> None:
```

Recrop — one field becoming the several fields it should have been

## MakeMasksScreen.recrop

### lines 3880-3882

```python
self._image_files.insert(
```

Straight after the field it came from, and after any sibling already cut out of it, so the children come out in the order they were drawn rather than in reverse.

### lines 3886-3888

```python
area = (box[2] - box[0]) * (box[3] - box[1])
```

On the PARENT's ledger, because this is something that was done to the parent: an area of it left. The child's own ledger says the other half of it — see :func:`mask_engine.write_recrop`.

### lines 3893-3896

```python
self._status_label.setText(
```

The object COUNT is the half of this the user cannot see: a box drawn a little too tight round two touching cells cuts both of them and writes a field with nothing in it, and the box on screen looks the same either way.

## MakeMasksScreen.finish_recrop

### lines 3920-3922

```python
if self._canvas.mask is not None:
```

The parent's mask and ledger are written before it is moved, so the record of the boxes travels into the archive with the file they were cut out of rather than being lost with the session.

### lines 3928-3929

```python
LOG.warning("Could not save %s before retiring it: %s",
```

The archive is the recovery, so a mask that will not write must not also stop the original being put somewhere safe.

## MakeMasksScreen._on_prev

### lines 3961-3963

```python
"""Go to the previous field, retiring this one if it was cut up."""
```

Leaving the field retires it if it was cut up, whichever way the user leaves: the parent must not be reachable again as though it were still a field to curate.

## MakeMasksScreen._on_next

### lines 3972-3973

```python
"""Go to the next field, retiring this one if it was cut up."""
```

A retirement has already moved the queue onto the first child, so Next has done what Next does and must not step past it.

## MakeMasksScreen._on_stroke_started

### lines 4054-4056

```python
"""Snapshot the mask before a stroke mutates it in place.
```

Brush/erase strokes mutate the mask in place; nothing to record until the stroke ends. History already has the pre-stroke mask from the previous op/load.

## MakeMasksScreen._sync_button_states

### lines 4086-4089

```python
for b in (self._btn_prev, self._btn_next, self._btn_save,
```

EVERY tool in the row, read off the row itself rather than listed here: a tool added to the mode table is disabled until a folder is open like the rest of them, without anyone having to remember this method exists.
