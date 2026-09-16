# Notes from `spacr/qt/widgets/gate_editor.py`

Prose lifted out of `spacr/qt/widgets/gate_editor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [GateCanvas.__init__](#gatecanvas__init__) (2 entries)
- [GateCanvas.decorate_axes](#gatecanvasdecorate_axes) (2 entries)
- [GateCanvas._render_volume](#gatecanvas_render_volume) (4 entries)
- [GateCanvas._draw_voxels](#gatecanvas_draw_voxels) (1 entry)
- [GateCanvas.screen_to_volume](#gatecanvasscreen_to_volume) (2 entries)
- [GateCanvas.render_now](#gatecanvasrender_now) (1 entry)
- [GateCanvas._gate_is_on_these_axes](#gatecanvas_gate_is_on_these_axes) (1 entry)
- [GateCanvas.handle_at](#gatecanvashandle_at) (1 entry)
- [GateCanvas._clear_ghost](#gatecanvas_clear_ghost) (1 entry)
- [GateCanvas._show_ghost](#gatecanvas_show_ghost) (1 entry)
- [GateCanvas._outline_pending_in_volume](#gatecanvas_outline_pending_in_volume) (1 entry)
- [GateCanvas.gate_at](#gatecanvasgate_at) (2 entries)
- [GateCanvas._volume_press](#gatecanvas_volume_press) (2 entries)
- [GateCanvas._volume_motion](#gatecanvas_volume_motion) (1 entry)
- [GateCanvas._show_volume_drag](#gatecanvas_show_volume_drag) (2 entries)
- [GateCanvas._gate_from_volume_drag](#gatecanvas_gate_from_volume_drag) (4 entries)
- [GateCanvas._depth_bounds_from_drag](#gatecanvas_depth_bounds_from_drag) (1 entry)
- [GateCanvas.anchor_plane](#gatecanvasanchor_plane) (1 entry)
- [GateCanvas._draw_anchor_aura](#gatecanvas_draw_anchor_aura) (1 entry)
- [GateCanvas._on_press](#gatecanvas_on_press) (4 entries)
- [GateCanvas._near_first_volume_vertex](#gatecanvas_near_first_volume_vertex) (1 entry)
- [GateCanvas._on_scroll](#gatecanvas_on_scroll) (1 entry)
- [GateCanvas._on_motion](#gatecanvas_on_motion) (1 entry)
- [GateCanvas._on_release](#gatecanvas_on_release) (4 entries)
- [GateCanvas._update_drag_patch](#gatecanvas_update_drag_patch) (1 entry)
- [GateCanvas.gate_from_drag](#gatecanvasgate_from_drag) (3 entries)
- [GateCanvas.close_polygon](#gatecanvasclose_polygon) (1 entry)
- [GateTree.__init__](#gatetree__init__) (3 entries)
- [GateTree._rebuild](#gatetree_rebuild) (2 entries)
- [_ClusterSettingsDialog.__init__](#_clustersettingsdialog__init__) (2 entries)
- [GateEditorPanel.__init__](#gateeditorpanel__init__) (15 entries)
- [GateEditorPanel.run_cluster](#gateeditorpanelrun_cluster) (6 entries)
- [GateEditorPanel._on_gate_edited](#gateeditorpanel_on_gate_edited) (1 entry)
- [GateEditorPanel._on_active_changed](#gateeditorpanel_on_active_changed) (1 entry)
- [GateEditorPanel.publish](#gateeditorpanelpublish) (2 entries)
- [GateEditorPanel._on_plane_picked](#gateeditorpanel_on_plane_picked) (2 entries)
- [GateEditorPanel._refresh_status](#gateeditorpanel_refresh_status) (1 entry)

## Module level

### line 113, trailing  _(unsure)_

```python
"#ff4d6d",
```

rose

### line 114, trailing  _(unsure)_

```python
"#4cc9f0",
```

cyan

### line 115, trailing  _(unsure)_

```python
"#ffb703",
```

amber

### line 116, trailing  _(unsure)_

```python
"#b388ff",
```

violet

### line 117, trailing  _(unsure)_

```python
"#06d6a0",
```

mint

### line 118, trailing  _(unsure)_

```python
"#ff8fab",
```

pink

### line 119, trailing  _(unsure)_

```python
"#8ecae6",
```

pale blue

### line 120, trailing  _(unsure)_

```python
"#f4a261",
```

sand

### line 173, trailing  _(unsure)_

```python
except Exception:
```

decoration is not load-bearing

## GateCanvas.__init__

### lines 288-290

```python
self._canvas.mpl_connect("button_release_event",
```

A spin ends here. `snap_to_axis` is read on release rather than during the drag: snapping mid-turn would fight the user's hand, and the point of snapping is only about the FINAL view.

### line 293, trailing  _(unsure)_

```python
except Exception:
```

no canvas in a bare test

## GateCanvas.decorate_axes

### lines 456-457  _(unsure)_

```python
if self._show_grid:
```

Line properties only when enabling: matplotlib warns that supplying them with False turns the grid ON, which is the opposite of asked.

### lines 463-467

```python
spec = self._spec
```

Decided by the DATA, not by the axis limits. The limits are padded outward, so a measurement whose smallest value is 1 gets a lower limit near -4 and looked non-positive -- which is why log X never applied while log Y, on a column with larger numbers and therefore proportionally smaller padding, sometimes did.

## GateCanvas._render_volume

### lines 677-678

```python
try:
```

Matplotlib's own drag-rotation is free rotation, which is what the axis lock exists to replace. Disabled so the two cannot fight.

### line 681, trailing  _(unsure)_

```python
except Exception:
```

older matplotlib

### lines 688-691

```python
self._draw_anchor_aura(ax)
```

AFTER the data and any remembered zoom have established the limits. Drawing this before scatter left a 0..1 square on measurements whose real range could be thousands, so the chosen plane was technically present and visually absent.

### lines 694-695  _(unsure)_

```python
self._axes = {(0, 0): ax}
```

Keyed like every other panel, so `panel_axes()` keeps its contract and nothing downstream has to know this one is three-dimensional.

## GateCanvas._draw_voxels

### lines 731-732

```python
sizes = 6.0 + 40.0 * (weight / weight.max())
```

Area, not radius, tracks the count: a marker whose RADIUS was the count would exaggerate a busy voxel by its square.

## GateCanvas.screen_to_volume

### lines 825-830

```python
try:
```

Perspective projection is not affine across a plane.  Invert the actual camera ray and intersect it with the selected face, so the point under the cursor is exact at every readable camera angle. The endpoint-based affine inverse below remains as a compatibility fallback for matplotlib versions whose private projection matrix moved again.

### lines 860-862

```python
anchor = list(origin)
```

Read on the SAME face as the blue aura.  Using the middle of the normal axis made the footprint land behind the plane the user had picked whenever the view was oblique.

## GateCanvas.render_now

### lines 1041-1043

```python
if self._mode in ("3D", "xD") and self._render_volume():
```

xD renders as a volume as well when it has a third component: the user picked PC1, PC2 and PC3 and got a 2D scatter, which reads as the third component having been ignored.

## GateCanvas._gate_is_on_these_axes

### line 1133  _(unsure)_

```python
return {gate.x_column, gate.y_column} <= showing
```

A box is drawn flat as its rectangle when its x and y are up.

## GateCanvas.handle_at

### lines 1254-1256

```python
ex, ey = getattr(event, "x", None), getattr(event, "y", None)
```

Pixel coordinates, via getattr: a real matplotlib event always carries them, but a synthetic one raised from code (a test, a scripted gate) need not, and no anchor is grabbable without them.

## GateCanvas._clear_ghost

### line 1276  _(unsure)_

```python
def _clear_ghost(self) -> None:
```

the shape that follows the mouse

## GateCanvas._show_ghost

### lines 1319-1322

```python
points = self._gate_points(ax, self._as_flat(gate))
```

AS FLAT, for the same reason gate_at and _outline are: the ghost previews what the drag would produce ON THIS VIEW, and a box has no flat layout until it is seen from the front. Without it a box gate was pulled with no preview of the result.

## GateCanvas._outline_pending_in_volume

### lines 1441-1442

```python
LOG.debug("could not draw the pending outline", exc_info=True)
```

An outline that cannot be painted costs the view nothing, and is not worth falling back to the flat plot that caused this.

## GateCanvas.gate_at

### lines 1481-1486

```python
if bool(self._as_flat(gate).mask(probe)[0]):
```

AS FLAT, like _outline and _handles_for above. The click is on the flat view, where a box IS its front rectangle -- and the raw box names a third measurement the two-column probe cannot carry, so mask() raised GateError, the except below swallowed it, and a gate that was drawn on screen with its corners showing could not be picked up at all.

### lines 1490-1491

```python
continue
```

A gate on columns this scatter is not showing cannot be hit-tested here, and must not stop the ones that can.

## GateCanvas._volume_press

### lines 1524-1525

```python
if self.drag_mode() == "draw":
```

The control decides.  Looking at the old 2D tool here made the new Spin button decorative because a rectangle is armed by default.

### line 1528  _(unsure)_

```python
return False
```

Click-per-vertex is handled by `_on_press`, not a drag.

## GateCanvas._volume_motion

### lines 1577-1578  _(unsure)_

```python
azimuth += dx * 0.5
```

Spinning about the vertical axis is a change of azimuth only: the horizon stays level, which is what makes it readable.

## GateCanvas._show_volume_drag

### lines 1652-1657

```python
return
```

Two of the three axes are showing the SAME measurement -- which the pickers allow, being filled from one column list with nothing excluded -- so there is no third axis to extend the footprint through and no rectangle to preview. Refused the way close_polygon already refuses it, rather than raising StopIteration out of a handler that runs on every motion event.

### lines 1663-1664  _(unsure)_

```python
order = {spec.x: 0, spec.y: 1, self._z_column: 2}
```

Drawn at BOTH ends of the depth axis, which is what the gate is: a rectangle extended all the way through the volume.

## GateCanvas._gate_from_volume_drag

### lines 1701-1703

```python
return None
```

The same measurement is on two of the three axes, so the drag describes no volume. Refused, like every other gesture the view cannot read, rather than raising out of the release handler.

### lines 1706-1710

```python
low, high = self.pending_depth()
```

THE SHAPE DROPDOWN DECIDES, on the plane the user picked. The depth bound comes from `pending_depth()` -- a slab the user drags out -- and stays None only when they have not set one, which is the full-depth case and still means what the 2D gate on that plane meant.

### lines 1716-1720

```python
u_radius = v_radius = max(u_radius, v_radius)
```

A circle is drawn round on the PLANE, so both radii are the same drag length. On two measurements with different units that is not a round shape on screen, and it is still what "circle gate" has to mean -- the alternative is a shape whose meaning changes when the axes rescale.

### lines 1738-1740

```python
bounds[depth_column] = (low, high)
```

The dragged slab wins over the "unbounded on the axis facing the viewer" default -- it is the more specific statement, and the user made it deliberately.

## GateCanvas._depth_bounds_from_drag

### line 1826

```python
if float(delta @ delta) < 9.0:
```

A click deliberately asks for the old/full-depth meaning.

## GateCanvas.anchor_plane

### lines 1920-1927

```python
if len(set(columns.values())) != 3:
```

A plane and the normal it is extended along are THREE measurements. The pickers are filled from one column list with nothing excluded, so a user can name the same one twice -- and this used to hand back a triple with a repeat in it, ('a', 'b', 'a'), which every caller then went wrong on in its own way. The aura was the visible one: its `order` map is keyed by column name, so the repeat collapsed it and every corner of the quad kept a None in the slot nothing filled. All three callers already treat None as "no plane is armed".

## GateCanvas._draw_anchor_aura

### lines 1950-1956

```python
u0, u1 = limits[axis_of[first]]
```

No KeyError is possible here and the handler that used to be around this block never ran. `anchor_plane` builds first/second/normal from exactly `spec.x`, `spec.y` and `self._z_column`, returns None if any is blank, and now returns None unless the three are DISTINCT -- so `axis_of` has an entry for each of them. Its values are 'x'/'y'/'z', which are exactly the keys of `limits`. Deleted rather than excluded from coverage: a branch that cannot be reached is dead code.

## GateCanvas._on_press

### lines 2030-2037

```python
mid_polygon = self._tool == POLYGON and bool(self._pending)
```

A press inside an existing gate MOVES it, whatever tool is armed. Checked first because "the closed gate should be draggable" has to hold without the user first disarming the tool they drew it with nobody thinks of that, and the gate then looks stuck.

The one exception is mid-polygon: there the user is placing vertices, and a vertex that happens to land inside an older gate must not drag it.

### lines 2040-2043

```python
grabbed = self.handle_at(event)
```

An anchor point is tested BEFORE the shape, because every anchor sits on or inside its own gate. Testing the shape first would mean a press on a corner moved the whole gate and resizing were unreachable.

### lines 2061-2072

```python
if event.inaxes is None or event.xdata is None or event.ydata is None:
```

From here the vertex is read off a FLAT plot, and the plot is flat whatever the mode says it is. A second copy of the volume block at the top of this method used to stand here, guarded by the MODE rather than by what is on screen -- and a 3D or xD view that fell back to the flat scatter (a third measurement the table has not got, an xD view with no third component, an empty table) is in one of those modes with an ordinary flat axes under the cursor. Every click went to the volume reader, which correctly answered that there is no volume to read, and the polygon tool placed nothing at all with no feedback. In a REAL volume `_volume_press` above has already taken the event, so the only state that block could ever reach was the one it broke.

### lines 2076-2078

```python
if len(self._pending) >= 3 and self._near_first_vertex(event, x, y):
```

Clicking the FIRST vertex again closes the shape. That is what everyone tries, and the "Close polygon" button was the only way to do it -- so a polygon looked impossible to finish.

## GateCanvas._near_first_volume_vertex

### lines 2156-2157

```python
point = self._volume_face_point(ax, *self._pending[0])
```

The SAME point `_outline_pending_in_volume` draws the marker at, so the vertex you are asked to click is the vertex that is measured.

## GateCanvas._on_scroll

### lines 2211-2213

```python
self._zoom = (zoomed(ax.get_xlim(), float(event.xdata)),
```

REMEMBERED, not just set. A redraw re-applies the computed scales after the marks are drawn, so limits set here alone are undone by the next render -- which is every gate edit.

## GateCanvas._on_motion

### lines 2252-2255

```python
self._show_ghost(self._dragged_to(event))
```

The EDIT is applied on release -- a gate is re-masked against the whole table to redraw, and doing that per mouse move makes a large frame unusable. What follows the mouse is a dashed placeholder in the shape of the gate, which costs one polygon.

## GateCanvas._on_release

### lines 2275-2277

```python
self.render_now()
```

Released off the axes, or the pull would have collapsed the shape. The gate is redrawn as it was rather than left with a ghost hanging over it.

### lines 2294-2295

```python
self.set_gates(self.gates, active=name)
```

A click, not a drag. Selecting rather than moving by zero keeps a stray click from marking the gate set dirty.

### lines 2301-2302  _(unsure)_

```python
return
```

The gate went away between press and release -- another view can remove one while a drag is in flight.

### lines 2314-2318

```python
pass
```

Matplotlib raises ValueError when an artist is already gone and NotImplementedError for containers that do not support removal. Either way the drag is over; the reference below is dropped regardless, or the next drag draws over a stale patch nothing cleans up.

## GateCanvas._update_drag_patch

### lines 2354-2355  _(unsure)_

```python
patch.set_center(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
```

Inscribed in the swept box, exactly as EllipseGate.from_drag builds it -- so the preview and the gate are the same shape.

## GateCanvas.gate_from_drag

### lines 2362-2372

```python
def gate_from_drag(self, x0: float, y0: float, x1: float, y1: float,
```

A drawn gate is TOP-LEVEL. It used to take its parent from the active gate, and drawing one selects it -- so the second gate nested inside the first, the third inside the second, and so on without anyone asking: "in the gate view it looks like the second gate is in the first and the thired gate is in the second". Worse, a nested gate is ANDed with its ancestors (`GateSet.mask` walks the path), so those gates were quietly not the shapes that were drawn.

The hierarchy itself is kept -- it round-trips through save/load and clustering still uses it -- but nesting is now something a caller asks for, never a side effect of what happens to be selected.

### lines 2385-2387

```python
return ThresholdGate(name=name, column=column, low=x0, high=x1)
```

Only the horizontal sweep is read: on a histogram the vertical axis is a count, and gating on a count is not a thing anyone means.

### lines 2399-2401

```python
return None
```

A zero-width drag would be an ellipse with a zero radius, which EllipseGate refuses. Nothing drawn is the right answer to nothing dragged.

## GateCanvas.close_polygon

### lines 2424-2425  _(unsure)_

```python
gate = PrismGate(name=name, u_column=first, v_column=second,
```

Unbounded along the normal, like every other shape drawn on the anchor plane: they said nothing about depth.

## GateTree.__init__

### lines 2485-2488

```python
header.setSectionResizeMode(0, QHeaderView.Stretch)
```

Every column visible on startup. Stretching column 0 and leaving the rest at their default width pushed n / % parent / % all off the edge of a narrow panel, so the counts -- the reason the tree has columns at all -- could not be seen until the user resized something.

### lines 2509-2511

```python
mark_surface(self.tree)
```

`GateEditorPanel` is transparent scaffolding by design (see the GraphBuilder block), so the hierarchy has nothing behind it and is the page itself.

### lines 2515-2523

```python
self._thresholds = QWidget(self)
```

Instruction 52 point 4: "the user should also be able to set thresholds for each individual gate for the measurements they are defined by". One row per measurement the SELECTED gate can take a threshold on -- which for a cylinder is its normal, and is how its height is bounded.

Rebuilt on selection rather than kept for every gate: a panel holding rows for gates nobody has selected is a panel that has to keep them in step with edits made elsewhere.

## GateTree._rebuild

### lines 2623-2625

```python
labels = [gate.name, self.UNAVAILABLE, "", ""]
```

Says so rather than vanishing: the gate keeps its row and its colour, and the count column carries the fact that this working set cannot answer it.

### lines 2633-2635

```python
item.setForeground(0, QBrush(QColor(colour)))
```

The gate's own colour, on its name. This is the half that makes colour-coding useful: a colour on the plot that is not also in the list is a colour with nothing to look it up in.

## _ClusterSettingsDialog.__init__

### lines 2862-2864

```python
from .gate_settings import GateEditorSettings
```

One place decides each default: the GateEditorSettings dataclass. Reading through getattr keeps an older saved settings object -- one written before a field existed -- from raising here.

### lines 2933-2935

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## GateEditorPanel.__init__

### lines 3023-3029

```python
continue
```

Not drag tools. The first three come from the 3D view: a shape dragged on a rotated projection has no defined extent along the axis pointing at the viewer, so offering them here would promise a gesture that cannot work. A composite is not drawn at all -- it is made from gates that already exist, in the gates panel.

### lines 3036-3038

```python
self._tool.currentIndexChanged.connect(self._on_tool_changed)
```

No `canvas.set_tool` here: the canvas is built further down and already starts on DEFAULT_TOOL. Calling it at this point read the attribute before it existed.

### lines 3043-3045

```python
self._settings_button = QPushButton("Settings", self)
```

No Close polygon button. Clicking the first vertex closes the shape, which is what everyone tries; once that worked, the button was a second way to do one thing and the only one that had to be found.

### lines 3070-3072

```python
self._mode_buttons: Dict[str, QPushButton] = {}
```

2D / 3D / xD, right of Cluster. Checkable and exclusive: the mode is one choice, and three buttons that can all be on describe a state the editor does not have.

### lines 3080-3083

```python
fit_to_text(button, padding=18)
```

Width from the TEXT, not a number. A fixed 38px clipped "2D", "3D" and "xD" on the sides, and any fixed size is a promise about a font the app does not control -- the user's theme, DPI and platform all change it.

### lines 3091-3095

```python
self._xd_button = QPushButton("xD", self)
```

OUTSIDE the exclusive group, deliberately. xD is not a third dimensionality: it says what the AXES ARE -- components rather than raw measurements -- and that is orthogonal to how many are drawn. Gating PC1 vs PC2 in 2D and PC1/PC2/PC3 in 3D are both things people want, and one exclusive group could express neither.

### lines 3107-3111

```python
volume_tools = QHBoxLayout()
```

3D has enough real controls to deserve its own row.  Putting all of them beside the ordinary tools clipped the status and the final spin-axis buttons on a normal laptop width -- the controls existed, but the user could not reach them, which is precisely how the first implementation of instruction 52 failed.

### lines 3116-3123

```python
self._plane_label = QLabel("plane", self)
```

Which axis the volume spins about. Shown only in 3D, because in 2D there is nothing to spin and a dead control is worse than no control. WHICH PLANE THE SHAPE LANDS ON, chosen rather than inferred. The first attempt read the plane off the camera angle and gave up unless the view happened to be square-on, so rotating changed what the next gate would mean. Three planes are visible in the volume; the user picks one and it stays picked.

### lines 3144-3147

```python
self._volume_shape = QComboBox(self)
```

THE SHAPE IS A DROPDOWN, which is where a user looks for one. The first attempt hid cylinder and prism from the tool picker because they are "not drag tools" -- reasoning that served the implementation and left the dropdown looking empty of 3D shapes.

### lines 3170-3172

```python
self._drag_label = QLabel("drag", self)
```

WHAT A DRAG DOES. Spinning and drawing are different gestures and were competing for the same mouse button, so a drag meant whichever the code happened to check first.

### lines 3214-3216

```python
self._status = QLabel("no gates", self)
```

No Apply button. A gate highlights its objects the moment it is shown (the tick in the gate list), so a button whose job was "now make it count" describes a step that no longer exists.

### lines 3225-3230

```python
self.body = QSplitter(Qt.Horizontal, self)
```

A SPLITTER, not a QHBoxLayout. The gate list sits between the scatter and the filter column, and in a box layout with a hard 320px cap it could not be resized at all: dragging the outer splitter moved the filter column and took the canvas AND the gate list with it as one block. Its own handle makes it independent, which is what "the gate box should be independent" asks for.

### lines 3243-3245

```python
self.tree.setMinimumWidth(220)
```

No maximum. A cap cannot be dragged past, so a gate whose name or statistics were wider than 320px had nowhere to be read. A minimum stays, so the handle cannot hide the list entirely.

### lines 3253-3254  _(unsure)_

```python
self.body.setStretchFactor(0, 1)
```

The scatter takes the slack when the panel is resized; the gate list keeps whatever width the user gave it.

### lines 3258-3260

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## GateEditorPanel.run_cluster

### lines 3353-3356

```python
spec = self.canvas.spec
```

The axes live on the SPEC. `canvas.x_column` has never existed, so getattr always returned the default and clustering refused two measurements that were plainly chosen -- "when i press cluster i get 'Clustering needs an X and a Y measurement.' when both are cohosen".

### lines 3396-3400

```python
tried = ", ".join(f"{c.eps:.3g}" for c in candidates)
```

Named rather than silently falling back to the typed eps: a walk that found nothing defensible is a result about the DATA, and clustering at the original radius anyway would present it as if the search had endorsed it.

### lines 3415-3419

```python
parent=self.canvas.active_gate)
```

A PROPERTY on the canvas, a METHOD on the tree. Calling the canvas one called its RESULT, so every cluster run that got past the dialog died on "'NoneType' object is not callable". The same mix-up is already recorded a few screens up for `gates`.

### lines 3422-3423

```python
QMessageBox.warning(self, "Could not cluster", str(exc))
```

Named, not swallowed: every one of these messages says what to change, and a silent empty result reads as a broken button.

### lines 3434-3442

```python
gates = self._gates
```

THE PANEL'S GateSet, not the canvas's. Until something calls set_gates the two are different objects, and on a fresh session nothing does -- so clusters added to the canvas's copy never reached the gate list, were never saved by screen.save_gates, were not counted in the status line, and were deleted by the next hand-drawn gate, which pushes self._gates back over the canvas. Same sequence as _on_gate_drawn, for the same reasons; tree.select is needed because tree.set_gates clears the tree and drives active back to None.

### lines 3452-3455

```python
QMessageBox.information(
```

What the walk decided, in the units the user typed in, so the number can be carried back to Gate Settings by hand. A search that silently substitutes a parameter is worse than one that never ran.

## GateEditorPanel._on_gate_edited

### lines 3470-3471  _(unsure)_

```python
gates = self.gates
```

`gates` is a PROPERTY on both this panel and the canvas. Calling it raised TypeError on every drag -- which is what the user saw.

## GateEditorPanel._on_active_changed

### lines 3518-3521

```python
if not name:
```

Choosing a gate should show you that gate. It is drawn on two named measurements, so selecting one whose axes are not on screen used to select something invisible -- and any attempt to drag it did nothing, because it was not being drawn or hit-tested there.

## GateEditorPanel.publish

### lines 3561-3574

```python
try:
```

A SELECTION, not a filter. Applying a gate highlights the objects inside it and leaves every other point on screen:

"i want it to highlight the datapoints in the gate and show the gate but also show the rest of the graph"

Filtering removed the outside rows, and the axes then rescaled to what was left -- which is what read as the plot zooming into the gate, and what moved the ground out from under the gate outline so it could not be dragged.

Narrowing the population to a gate is still a real thing to want, but it is a second, explicit act. It is not what pressing the primary button should do.

### lines 3578-3581

```python
self._status.setText(
```

A highlight needs object keys to name the rows to everyone else. A table without them cannot be published, and saying so is better than a traceback -- the gate itself is still drawn and still usable locally.

## GateEditorPanel._on_plane_picked

### lines 3617-3620

```python
"""Rotate a 3-D view to look down one axis.
```

Tick the button too. Called from its own clicked signal the button is already checked, but called programmatically -- restoring saved settings, or a test -- it is not, and the control would then show a different plane from the one the canvas is armed on.

### lines 3629-3631

```python
button = getattr(self, "_drag_buttons", {}).get("draw")
```

Picking a plane is choosing where to draw, so it arms drawing too. Making the user then find a second control to say "and now let me draw" is the kind of step that reads as the feature not working.

## GateEditorPanel._refresh_status

### lines 3716-3719

```python
active = self.tree.active_gate()
```

It used to say "drawing on inside <gate>", which was true when selecting a gate replotted its population. The plot always shows the whole table now, so saying otherwise would be a lie about the thing the user is looking at.
