"""N672 follow-up, maintainer desktop check 2026-10-12.

Two requests on top of the accepted anchor and surface editing:

* rebindable Gate Editor keys, the same as Make Masks' True 3D -- Shift+E
  Edit, Shift+S Spin, Shift+D Draw, Shift+X/Y/Z the X, Y or Z plane, or the
  spin axis when pressed within 1.5 s of Shift+S;
* an overlapping-element picker: a click on a sheet with an anchor or another
  sheet under it lists every element under the click in the graph's top-left
  corner, top layer first and in blue, and the row chosen there is the one
  that is selected, edited and moved.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

from spacr.qt import shortcuts as S
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets import gate_editor as GE
from spacr.qt.widgets.gate_editor import GateCanvas, _screen_points
from spacr.qt.widgets.gate_spec import (
    BoxGate, EditableVolumeGate, GateSet, RectGate, ThresholdGate)
from spacr.qt.widgets.graph_spec import GraphSpec

GATE_KEYS = ["Shift+E", "Shift+S", "Shift+D", "Shift+X", "Shift+Y", "Shift+Z"]


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1100, 800)
    values = np.random.default_rng(41).uniform(-1.5, 1.5, (200, 3))
    widget.set_frame(pd.DataFrame(values, columns=['a', 'b', 'c']))
    widget._x.setCurrentText('a')
    widget._y.setCurrentText('b')
    widget.gates.mode_requested.emit('3D')
    widget._z.setCurrentText('c')
    widget.show()
    QApplication.processEvents()
    return widget


def _focus(qtbot, screen):
    screen.activateWindow()
    screen.gates.canvas._canvas.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is screen)
    return screen.gates.canvas._canvas


def _box():
    return BoxGate.from_limits('box', ('a', 'b', 'c'), [(-1, 1)] * 3)


def _install(screen, gate):
    screen.gates.set_gates(GateSet([gate]))
    screen.gates.tree.select(gate.name)
    screen.gates._drag_buttons['edit'].click()
    for _ in range(3):
        QApplication.processEvents()
    screen.gates.canvas.render_now()
    return screen.gates.canvas


def _event(canvas, point, **values):
    event = dict(inaxes=canvas.axes_at(0, 0), x=float(point[0]),
                 y=float(point[1]), xdata=0., ydata=0., button=1, step=0,
                 key=None, dblclick=False)
    event.update(values)
    return SimpleNamespace(**event)


def _top_face_click(canvas):
    return _screen_points(canvas.axes_at(0, 0), [[0.3, 0.3, 1.0]])[0]


def _rows(canvas):
    rows = canvas._overlap_rows
    return [rows.item(i) for i in range(rows.count())]


# ---------------------------------------------------------------------------
# The shortcut registry
# ---------------------------------------------------------------------------

def test_gate_editor_keys_are_declared_listed_and_clash_with_nothing(qtbot):
    specs = S._screen_specs("Gate Editor")
    assert [spec.keys for spec in specs] == GATE_KEYS
    assert {spec.scope for spec in specs} == {"the Gate Editor screen"}
    assert not {"Left", "Right"} & {spec.keys for spec in specs}
    for scope in ("Annotate", "Make Masks", "Field browser"):
        assert {"Left", "Right"} <= {s.keys for s in S._screen_specs(scope)}
    masks = {s.keys for s in S._screen_specs("Make Masks")}
    assert set(GATE_KEYS) <= masks, "the same keys as Make Masks' True 3D"
    mapped = [(s.keys, s.category) for s in S.mapped()]
    assert all((keys, "Gate Editor") in mapped for keys in GATE_KEYS)
    assert S._conflicts({}, {}) == {}
    window = QMainWindow()
    qtbot.addWidget(window)
    dialog = S._KeymapDialog(window)
    qtbot.addWidget(dialog)
    listed = {(scope, spec.keys) for scope, spec in dialog._screen_rows}
    assert {("Gate Editor", keys) for keys in GATE_KEYS} <= listed
    scopes = {dialog._table.item(row, 3).text()
              for row in range(dialog._table.rowCount())}
    assert "the Gate Editor screen" in scopes
    dialog._set_screen_key("Gate Editor", "Shift+E", "Shift+D")
    assert "Edit mode" in dialog.conflict_text()
    assert "Draw" in dialog.conflict_text()
    dialog.reject()


def _expire_chord(panel):
    panel._spin_chord_from = float("-inf")


def test_mode_and_plane_keys_drive_the_buttons(qtbot, screen):
    panel = screen.gates
    target = _focus(qtbot, screen)
    qtbot.keyClick(target, Qt.Key_E, Qt.ShiftModifier)
    assert panel._drag_buttons["edit"].isChecked()
    assert panel.canvas.drag_mode() == "edit"
    qtbot.keyClick(target, Qt.Key_D, Qt.ShiftModifier)
    assert panel.canvas.drag_mode() == "draw"
    for key, axis in ((Qt.Key_X, "x"), (Qt.Key_Y, "y"), (Qt.Key_Z, "z")):
        qtbot.keyClick(target, key, Qt.ShiftModifier)
        assert panel._plane_buttons[axis].isChecked()
        assert panel.canvas.anchor_axis() == axis
    panel._spin_buttons["x"].click()
    qtbot.keyClick(target, Qt.Key_S, Qt.ShiftModifier)
    assert panel.canvas.drag_mode() == "spin"
    assert panel.canvas._spin_axis == "", "Shift+S starts from a free spin"
    _expire_chord(panel)
    qtbot.keyClick(target, Qt.Key_X, Qt.ShiftModifier)
    assert panel.canvas.anchor_axis() == "x", "the chord is over"
    assert panel.canvas._spin_axis == ""


@pytest.mark.parametrize("key, axis", [(Qt.Key_Z, "z"), (Qt.Key_X, "x"),
                                       (Qt.Key_Y, "y")])
def test_spin_chord_picks_the_axis_within_the_window(qtbot, screen, key, axis):
    panel = screen.gates
    target = _focus(qtbot, screen)
    panel._on_plane_picked("y" if axis != "y" else "x")
    plane = panel.canvas.anchor_axis()
    qtbot.keyClick(target, Qt.Key_S, Qt.ShiftModifier)
    qtbot.keyClick(target, key, Qt.ShiftModifier)
    assert panel.canvas._spin_axis == axis
    assert panel._spin_buttons[axis].isChecked()
    assert panel.canvas.drag_mode() == "spin"
    assert panel.canvas.anchor_axis() == plane, "spun, not re-planed"
    assert panel._SPIN_CHORD_SECONDS == 1.5


def test_mode_keys_follow_the_screen_focus_only(qtbot, screen):
    panel = screen.gates
    elsewhere = QWidget()
    qtbot.addWidget(elsewhere)
    elsewhere.show()
    elsewhere.activateWindow()
    elsewhere.setFocus()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is elsewhere)
    qtbot.keyClick(elsewhere, Qt.Key_E, Qt.ShiftModifier)
    assert panel.canvas.drag_mode() == "spin"


def test_keys_are_rebindable_live(qtbot, screen):
    panel = screen.gates
    window = QMainWindow()
    qtbot.addWidget(window)
    dialog = S._KeymapDialog(window)
    qtbot.addWidget(dialog)
    dialog._set_screen_key("Gate Editor", "Shift+E", "F8")
    dialog._set_screen_key("Gate Editor", "Shift+Z", "F9")
    assert not dialog.conflict_text()
    assert dialog.save()
    assert S._load_screen_keymap()["Gate Editor"] == {
        "Shift+E": "F8", "Shift+Z": "F9"}
    assert screen._spacr_screen_holders["Shift+E"].key() == QKeySequence("F8")
    target = _focus(qtbot, screen)
    qtbot.keyClick(target, Qt.Key_E, Qt.ShiftModifier)
    assert panel.canvas.drag_mode() == "spin"
    qtbot.keyClick(target, Qt.Key_F8)
    assert panel.canvas.drag_mode() == "edit"
    qtbot.keyClick(target, Qt.Key_S, Qt.ShiftModifier)
    qtbot.keyClick(target, Qt.Key_F9)
    assert panel.canvas._spin_axis == "z"
    rebuilt = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(rebuilt)
    assert rebuilt._spacr_screen_holders["Shift+E"].key() == QKeySequence("F8")
    S._save_keymap({}, {})


# ---------------------------------------------------------------------------
# The picker in 3D
# ---------------------------------------------------------------------------

def test_volume_click_lists_every_hit_top_layer_first_in_blue(screen):
    canvas = _install(screen, _box())
    click = _top_face_click(canvas)
    hits = canvas._overlap_hits_at(_event(canvas, click))
    assert len(hits) >= 2
    assert [hit.layer for hit in hits] == sorted(hit.layer for hit in hits)
    assert hits[0].layer == 0
    for before, after in zip(hits, hits[1:]):
        if before.layer == after.layer and before.kind == "anchor":
            assert after.kind == "anchor", "sheets before anchors"
    front = [hit for hit in hits if hit.kind == "sheet"]
    assert len(front) >= 2 and front[0].depth < front[-1].depth
    canvas._on_press(_event(canvas, click))
    assert canvas._overlap_panel.isVisible()
    assert canvas._overlap_panel.pos().x() < 40
    assert canvas._overlap_panel.pos().y() < 40
    rows = _rows(canvas)
    assert len(rows) == len(hits)
    blue, white = QColor(GE._TOP_LAYER_BLUE), QColor(GE._OVERLAP_INK)
    for item, hit in zip(rows, canvas._overlap):
        assert item.foreground().color() == (blue if hit.layer == 0 else white)
    assert rows[0].foreground().color() == blue
    assert rows[-1].foreground().color() == white
    assert any(item.text().startswith("Surface") for item in rows)
    canvas._on_release(_event(canvas, click))
    assert screen.gates.gates.get('box') == _box(), "a click edits nothing"


def test_choosing_a_lower_surface_makes_it_the_one_moved(screen):
    canvas = _install(screen, _box())
    screen.apply_settings(GE.replace(screen._settings, show_anchors=False))
    canvas = screen.gates.canvas
    canvas._edit_axis = 'z'
    click = _top_face_click(canvas)
    canvas._on_press(_event(canvas, click))
    canvas._on_release(_event(canvas, click))
    sheets = [i for i, hit in enumerate(canvas._overlap) if hit.kind == "sheet"]
    assert canvas._overlap_choice == sheets[0], "nearest sheet, as before"
    lower = sheets[-1]
    group = canvas._overlap[lower].ident
    canvas._overlap_rows.setCurrentRow(lower)
    assert canvas._overlap_choice == lower
    assert canvas._selected_surfaces == {group}
    canvas._on_press(_event(canvas, click))
    assert canvas._surface_drag is not None
    assert canvas._surface_drag[2] == (group,)
    assert canvas._overlap_choice == lower, "same overlap, same choice"
    ax = canvas.axes_at(0, 0)
    direction = canvas._pixels_per_unit(ax, canvas._surface_drag[4])[:, 2]
    end = np.asarray(click) + direction * .2
    canvas._on_motion(_event(canvas, end))
    canvas._on_release(_event(canvas, end))
    edited = screen.gates.gates.get('box')
    assert isinstance(edited, EditableVolumeGate)
    mesh = EditableVolumeGate.from_gate(_box())
    moved = {i for face, g in zip(mesh.faces, mesh.surface_groups)
             if g == group for i in face}
    shift = np.asarray(edited.vertices) - np.asarray(mesh.vertices)
    assert np.allclose(shift[sorted(moved), 2], .2)
    still = sorted(set(range(len(mesh.vertices))) - moved)
    assert np.allclose(shift[still], 0)


def test_anchor_rows_keyboard_and_closing(qtbot, screen):
    canvas = _install(screen, _box())
    ax = canvas.axes_at(0, 0)
    role, point = next((r, p) for r, p in canvas._volume_handles(_box())
                       if np.allclose(p, [0, 0, 1]))
    pixel = _screen_points(ax, [point])[0]
    canvas._on_press(_event(canvas, pixel))
    assert canvas._handle_drag[1] == role, "the anchor is grabbed, as before"
    canvas._on_release(_event(canvas, pixel))
    chosen = canvas._overlap[canvas._overlap_choice]
    assert chosen.kind == "anchor" and chosen.ident == role
    assert any(item.text().startswith("Anchor") for item in _rows(canvas))
    rows = canvas._overlap_rows
    start = rows.currentRow()
    step = Qt.Key_Down if start + 1 < rows.count() else Qt.Key_Up
    qtbot.keyClick(rows, step)
    assert canvas._overlap_choice == rows.currentRow() != start
    sheet = next(i for i, hit in enumerate(canvas._overlap)
                 if hit.kind == "sheet")
    canvas._choose_overlap(sheet)
    assert rows.currentRow() == sheet
    canvas._choose_overlap(len(canvas._overlap) + 3)
    assert canvas._overlap_choice == sheet
    canvas._choose_overlap(sheet)
    anchor = next(i for i, hit in enumerate(canvas._overlap)
                  if hit.kind == "anchor")
    canvas._choose_overlap(anchor)
    canvas._on_press(_event(canvas, pixel))
    assert canvas._handle_drag[1] == role
    canvas._on_release(_event(canvas, pixel))
    canvas.render_now()
    qtbot.keyClick(rows, Qt.Key_A)
    qtbot.keyClick(rows, Qt.Key_Escape)
    assert not canvas._overlap and not canvas._overlap_panel.isVisible()


def test_the_list_closes_on_empty_space_and_on_leaving_edit(screen):
    canvas = _install(screen, _box())
    click = _top_face_click(canvas)
    canvas._on_press(_event(canvas, click))
    canvas._on_release(_event(canvas, click))
    assert canvas._overlap
    canvas._on_press(_event(canvas, click, button=3))
    canvas._on_release(_event(canvas, click, button=3))
    assert canvas._overlap, "a right-drag spins and keeps the list"
    canvas._on_press(_event(canvas, click, key="control"))
    canvas._on_release(_event(canvas, click, key="control"))
    assert canvas._overlap, "Ctrl-click adds surfaces and keeps the list"
    canvas._on_press(_event(canvas, (2, 2)))
    canvas._on_release(_event(canvas, (2, 2)))
    assert not canvas._overlap and not canvas._overlap_panel.isVisible()
    canvas._on_press(_event(canvas, _top_face_click(canvas)))
    assert canvas._overlap
    screen.gates._choose_drag_mode("spin")
    assert not canvas._overlap
    screen.gates._choose_drag_mode("edit")
    canvas._on_press(_event(canvas, _top_face_click(canvas)))
    canvas._on_release(_event(canvas, _top_face_click(canvas)))
    canvas.set_mode("3D")
    assert not canvas._overlap
    canvas._on_press(_event(canvas, _top_face_click(canvas)))
    canvas._on_release(_event(canvas, _top_face_click(canvas)))
    canvas.set_gates(GateSet([_box()]), active='box')
    assert not canvas._overlap


def test_no_editable_gate_or_no_anchor_means_no_volume_list(screen):
    canvas = _install(screen, _box())
    click = _top_face_click(canvas)
    screen.gates.set_gates(GateSet([_box()]))
    canvas.set_gates(canvas.gates, active=None)
    assert canvas._volume_hits(_event(canvas, click)) == []
    canvas = _install(screen, _box())
    screen.apply_settings(GE.replace(screen._settings, show_anchors=False))
    canvas = screen.gates.canvas
    hits = canvas._volume_hits(_event(canvas, _top_face_click(canvas)))
    assert hits and {hit.kind for hit in hits} == {"sheet"}


# ---------------------------------------------------------------------------
# The picker in 2D
# ---------------------------------------------------------------------------

class _Mouse:
    def __init__(self, ax, x_data, y_data, *, button=1, key=None):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = button
        self.step = 0
        self.key = key


def _flat(qtbot, *gates):
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(700, 500)
    canvas.set_frame(pd.DataFrame({"a": [0.0, 5.0, 10.0],
                                   "b": [0.0, 5.0, 10.0]}))
    canvas.set_spec(GraphSpec(x="a", y="b"))
    canvas.set_gates(GateSet(list(gates)))
    canvas.show()
    QApplication.processEvents()
    return canvas


def _rect(name, low, high):
    return RectGate(name=name, x_column="a", y_column="b", x_low=low,
                    x_high=high, y_low=low, y_high=high)


def test_flat_overlap_lists_the_top_gate_first_and_moves_the_chosen_one(qtbot):
    canvas = _flat(qtbot, _rect("under", 1, 7), _rect("over", 3, 9))
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    assert canvas._move_name == "over", "the top gate, as before"
    canvas._on_release(_Mouse(ax, 5.0, 5.0))
    assert [item.text() for item in _rows(canvas)] == ["Region · over",
                                                       "Region · under"]
    blue, white = QColor(GE._TOP_LAYER_BLUE), QColor(GE._OVERLAP_INK)
    assert [item.foreground().color() for item in _rows(canvas)] == [blue,
                                                                     white]
    canvas._overlap_rows.setCurrentRow(1)
    edits = []
    canvas.gate_edited.connect(edits.append)
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, 5.0, 5.0))
    assert canvas._move_name == "under"
    canvas._on_release(_Mouse(ax, 6.0, 5.0))
    assert [gate.name for gate in edits] == ["under"]
    assert edits[0].x_low == pytest.approx(2.0)


def test_flat_anchor_over_a_sheet_and_single_hits(qtbot):
    canvas = _flat(qtbot, _rect("under", 1, 7), _rect("over", 3, 9))
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, 3.0, 3.0))
    assert canvas._resize is not None and canvas._resize[0] == "over"
    kinds = [(hit.kind, hit.gate) for hit in canvas._overlap]
    assert kinds[0] == ("sheet", "over") and ("anchor", "over") in kinds
    assert ("sheet", "under") in kinds
    assert kinds.index(("anchor", "over")) < kinds.index(("sheet", "under"))
    canvas._on_release(_Mouse(ax, 3.0, 3.0))
    ax = canvas.axes_at(0, 0)
    canvas.render_now()
    canvas._on_press(_Mouse(ax, 8.5, 8.5))
    assert not canvas._overlap and canvas._move_name == "over"
    canvas._on_release(_Mouse(ax, 8.5, 8.5))
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, 5.0, 5.0, key="control"))
    assert not canvas._overlap
    canvas._on_release(_Mouse(ax, 5.0, 5.0))
    canvas._show_anchors = False
    hits = canvas._flat_hits(_Mouse(canvas.axes_at(0, 0), 3.0, 3.0))
    assert {hit.kind for hit in hits} == {"sheet"}
    off = SimpleNamespace(inaxes=None, x=None, y=None, xdata=None, ydata=None)
    assert canvas._flat_hits(off) == []
    pixel_only = _Mouse(canvas.axes_at(0, 0), 5.0, 5.0)
    pixel_only.xdata = pixel_only.ydata = None
    assert canvas._flat_hits(pixel_only) == []


def test_flat_highlight_draws_sheets_anchors_and_skips_the_missing(qtbot):
    threshold = ThresholdGate(name="cut", column="a", low=2.0, high=6.0)
    canvas = _flat(qtbot, _rect("under", 1, 7), threshold, _rect("over", 3, 9))
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, 3.0, 3.0))
    canvas._on_release(_Mouse(ax, 3.0, 3.0))
    gates = {hit.gate for hit in canvas._overlap}
    assert {"over", "under", "cut"} <= gates
    before = len(canvas._artists)
    canvas._overlap.append(GE._OverlapHit("sheet", "gone", "", 9, 9., 0., 0))
    canvas._overlap.append(GE._OverlapHit("anchor", "over", "nowhere", 9, 9.,
                                          0., 1))
    canvas.render_now()
    assert len(canvas._artists) >= before
    canvas._close_overlap(redraw=False)
    assert not canvas._overlap
    canvas._close_overlap()


def test_volume_anchor_highlight_skips_rows_it_cannot_place(screen):
    canvas = _install(screen, _box())
    click = _top_face_click(canvas)
    canvas._on_press(_event(canvas, click))
    canvas._on_release(_event(canvas, click))
    canvas._overlap.append(GE._OverlapHit("anchor", "box", "anchor:none", 9,
                                          9., 0., 1))
    canvas.render_now()
    assert canvas._overlap[-1].ident == "anchor:none"


def test_flat_containment_refuses_what_it_cannot_test(qtbot):
    canvas = _flat(qtbot, _rect("only", 1, 7))
    assert canvas._flat_contains(canvas.gates.get("only"), 4.0, 4.0)
    assert not canvas._flat_contains(canvas.gates.get("only"), 9.0, 9.0)
    assert not canvas._flat_contains(SimpleNamespace(columns=()), 4.0, 4.0)

    class _Broken:
        columns = ("a", "b")

        def mask(self, _frame):
            raise ValueError("cannot be tested here")

    assert not canvas._flat_contains(_Broken(), 4.0, 4.0)


def test_a_plain_surface_grab_still_drags_without_the_list(screen):
    canvas = _install(screen, _box())
    click = _top_face_click(canvas)
    assert canvas._grab_volume_surface(_event(canvas, click))
    assert canvas._surface_drag is not None
    assert not canvas._overlap
    canvas._on_release(_event(canvas, click))
