"""``B12`` ``C7`` — the brush and the track surgery as widgets.

:mod:`tests.test_curation` covers the session with no Qt at all. What is left
here is what needs a widget: that a drag is ONE undoable stroke, that the
brush does not keep painting after the button comes up, and that a refused
join says so instead of quietly declining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.curation import CurationLog, MaskCuration, is_curated
from spacr.layers import LayerStack, Spacing
from spacr.qt import curation_tool as ct
from spacr.qt import layer_viewer as lv


def _stack(size=64, step=1.0, units="px"):
    stack = LayerStack()
    spacing = Spacing.isotropic(2, step, units=units)
    stack.add_image(np.zeros((size, size), np.uint16), name="image",
                    spacing=spacing)
    stack.add_labels(np.zeros((size, size), np.int64), name="mask",
                     spacing=spacing)
    return stack


def _canvas(qtbot, stack=None, width=200, height=200):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(width, height)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, canvas, **kwargs):
    panel = ct.BrushPanel(canvas, **kwargs)
    qtbot.addWidget(panel)
    return panel


def _tracks(spec):
    rows = []
    for track_id, frames in spec.items():
        for frame in frames:
            rows.append({"frame": frame, "track_id": track_id,
                         "original_label": 100 + frame,
                         "x": float(frame), "y": float(track_id)})
    return pd.DataFrame(rows, columns=["frame", "track_id", "original_label",
                                       "x", "y"])


# ---------------------------------------------------------------------------
# The brush tool
# ---------------------------------------------------------------------------

def test_a_click_paints_at_the_world_point_under_the_cursor(qtbot,
                                                            qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    panel.label_spin.setValue(5)
    panel.radius_spin.setValue(2.0)
    panel.start_painting()

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))

    world = canvas.canvas.world_at(45 - 1, 60 - 1)
    mask = stack["mask"]
    expected = mask.brush_index(world, radius=2.0)
    painted = np.argwhere(mask.data == 5)
    assert len(painted) == len(expected[0])
    assert set(map(tuple, painted.tolist())) == set(
        zip(*(part.tolist() for part in expected)))


def test_a_drag_is_one_stroke_and_one_undo(qtbot, qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    panel.label_spin.setValue(3)
    tool = panel.start_painting()

    tool.press(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.LeftButton))
    for x in range(41, 60):
        tool.move(canvas, canvas.canvas.world_at(50, x),
                  _Event(Qt.LeftButton, buttons=Qt.LeftButton))
    tool.release(canvas, canvas.canvas.world_at(50, 59), _Event(Qt.LeftButton))

    assert (stack["mask"].data == 3).any()
    assert len(panel.session) == 1                 # ONE stroke
    assert len(panel.session.log) == 1             # ONE ledger entry
    panel.undo()
    assert not stack["mask"].data.any()


def test_the_brush_does_not_paint_after_the_button_comes_up(qtbot,
                                                            qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    tool = panel.start_painting()

    tool.press(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.LeftButton))
    tool.release(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.LeftButton))
    before = stack["mask"].data.copy()

    # The cursor keeps travelling; nothing must follow it.
    consumed = tool.move(canvas, canvas.canvas.world_at(90, 90),
                         _Event(Qt.LeftButton, buttons=Qt.NoButton))
    assert consumed is False
    assert np.array_equal(stack["mask"].data, before)


def test_a_right_drag_erases(qtbot, qt_theme_applied):
    stack = _stack()
    stack["mask"].data[:] = 9
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    tool = panel.start_painting()

    tool.press(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.RightButton))
    tool.release(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.RightButton))

    assert (stack["mask"].data == 0).any()
    assert panel.session.log.edits[-1].target == 0


def test_the_bracket_keys_resize_the_brush(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    tool = panel.start_painting()
    before = panel.session.radius

    tool.key(canvas, _KeyEvent("]"))
    assert panel.session.radius > before
    tool.key(canvas, _KeyEvent("["))
    assert panel.session.radius == pytest.approx(before)


def test_backspace_undoes_a_stroke(qtbot, qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    tool = panel.start_painting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert stack["mask"].data.any()

    tool.key(canvas, _KeyEvent("", key=Qt.Key_Backspace))
    assert not stack["mask"].data.any()


def test_taking_the_tool_off_closes_a_stroke_rather_than_losing_it(
        qtbot, qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    tool = panel.start_painting()

    tool.press(canvas, canvas.canvas.world_at(50, 40), _Event(Qt.LeftButton))
    panel.stop_painting()          # the tool is detached mid-stroke

    assert len(panel.session.log) == 1
    assert panel.session.can_undo


def test_a_stack_with_no_mask_refuses_a_brush_rather_than_painting_nowhere(
        qtbot, qt_theme_applied):
    from spacr.layers import LayerError

    stack = LayerStack()
    stack.add_image(np.zeros((16, 16), np.uint16), name="image")
    canvas = _canvas(qtbot, stack)
    with pytest.raises(LayerError, match="labels layer"):
        ct.BrushPanel(canvas)


# ---------------------------------------------------------------------------
# The brush panel
# ---------------------------------------------------------------------------

def test_new_label_picks_one_the_mask_is_not_using(qtbot, qt_theme_applied):
    stack = _stack()
    stack["mask"].data[0, 0] = 7
    panel = _panel(qtbot, _canvas(qtbot, stack))
    assert panel.use_next_label() == 8
    assert panel.session.label == 8


def test_the_panel_shows_the_ledger_and_the_badge(qtbot, qt_theme_applied):
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas)
    assert "as the pipeline produced it" in panel.badge.text()

    panel.start_painting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.ledger.count() == 1
    assert "curated by hand" in panel.badge.text()


def test_saving_the_log_writes_it_beside_the_mask(qtbot, qt_theme_applied,
                                                  tmp_path):
    mask_path = tmp_path / "plate1_A01_1.tif"
    mask_path.write_bytes(b"")
    stack = _stack()
    canvas = _canvas(qtbot, stack)
    panel = _panel(qtbot, canvas, artifact=str(mask_path))
    panel.start_painting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))

    written = panel.save_log()
    assert written.endswith(".curation.json")
    assert is_curated(mask_path)
    assert [e.kind for e in CurationLog.read(written).edits] == ["paint"]


def test_the_undo_button_is_off_until_there_is_something_to_undo(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    assert not panel.undo_button.isEnabled()
    panel.start_painting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.undo_button.isEnabled()


# ---------------------------------------------------------------------------
# Track curation
# ---------------------------------------------------------------------------

@pytest.fixture
def track_panel(qtbot, qt_theme_applied):
    panel = ct.TrackCurationPanel(tracks=_tracks({1: [0, 1, 2, 3], 2: [6, 7]}))
    qtbot.addWidget(panel)
    return panel


def _select(panel, *track_ids):
    panel.track_list.clearSelection()
    for index in range(panel.track_list.count()):
        item = panel.track_list.item(index)
        if item.data(Qt.UserRole) in track_ids:
            item.setSelected(True)


def test_the_list_shows_every_track_with_its_span(track_panel):
    assert track_panel.track_list.count() == 2
    assert "track 1" in track_panel.track_list.item(0).text()
    assert "4 frame(s)" in track_panel.track_list.item(0).text()
    assert "0–3" in track_panel.track_list.item(0).text()


def test_joining_two_selected_tracks_leaves_the_table_consistent(track_panel):
    _select(track_panel, 1, 2)
    assert track_panel.join_selected()

    assert track_panel.session.track_ids == [1]
    assert track_panel.session.frames_of(1) == [0, 1, 2, 3, 6, 7]
    assert track_panel.session.check() == []
    assert track_panel.track_list.count() == 1


def test_a_refused_join_says_why_rather_than_declining_quietly(qtbot,
                                                               qt_theme_applied):
    panel = ct.TrackCurationPanel(tracks=_tracks({1: [0, 1, 2], 2: [2, 3]}))
    qtbot.addWidget(panel)
    _select(panel, 1, 2)

    assert panel.join_selected() is False
    assert "frame" in panel.status.text()
    assert panel.session.track_ids == [1, 2]     # untouched
    assert panel.session.check() == []


def test_splitting_gives_the_tail_a_new_track(track_panel):
    _select(track_panel, 1)
    track_panel.frame_spin.setValue(2)
    assert track_panel.split_selected()

    assert track_panel.session.frames_of(1) == [0, 1]
    assert track_panel.session.frames_of(3) == [2, 3]
    assert track_panel.session.check() == []


def test_a_split_that_would_move_nothing_says_so(track_panel):
    _select(track_panel, 1)
    track_panel.frame_spin.setValue(0)
    assert track_panel.split_selected() is False
    assert "one side empty" in track_panel.status.text()


def test_deleting_removes_the_track(track_panel):
    _select(track_panel, 2)
    assert track_panel.delete_selected()
    assert track_panel.session.track_ids == [1]
    assert track_panel.session.check() == []


def test_the_buttons_are_off_until_the_right_number_is_selected(track_panel):
    track_panel.track_list.clearSelection()
    track_panel._refresh_buttons()
    assert not track_panel.join_button.isEnabled()
    assert not track_panel.split_button.isEnabled()

    _select(track_panel, 1)
    assert track_panel.split_button.isEnabled()
    assert track_panel.delete_button.isEnabled()
    assert not track_panel.join_button.isEnabled()

    _select(track_panel, 1, 2)
    assert track_panel.join_button.isEnabled()


def test_every_operation_shows_up_in_the_panel_ledger(track_panel):
    _select(track_panel, 1, 2)
    track_panel.join_selected()
    assert track_panel.ledger.count() == 1
    assert "join" in track_panel.ledger.item(0).text()
    assert "curated by hand" in track_panel.status.text()


def test_saving_writes_the_table_and_the_ledger_together(track_panel,
                                                         tmp_path):
    _select(track_panel, 1, 2)
    track_panel.join_selected()
    target = tmp_path / "tracks" / "btrack_tracks_cell_plate1_A01_1.csv"

    written = track_panel.save(str(target))

    assert pd.read_csv(written)["track_id"].nunique() == 1
    assert is_curated(written)


def test_reopening_a_curated_table_continues_its_history(track_panel,
                                                         tmp_path, qtbot,
                                                         qt_theme_applied):
    _select(track_panel, 1, 2)
    track_panel.join_selected()
    target = str(tmp_path / "t.csv")
    track_panel.save(target)

    second = ct.TrackCurationPanel()
    qtbot.addWidget(second)
    second.load(target)

    # The earlier join is still visible, rather than a fresh ledger hiding it.
    assert [e.kind for e in second.session.log.edits] == ["join"]
    assert second.ledger.count() == 1


def test_a_csv_that_is_not_a_track_table_says_so(qtbot, qt_theme_applied,
                                                 tmp_path):
    path = tmp_path / "nope.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    panel = ct.TrackCurationPanel()
    qtbot.addWidget(panel)

    assert panel.load(str(path)) is None
    assert "track_id" in panel.status.text()


def test_a_panel_with_no_tracks_open_does_nothing_rather_than_raising(
        qtbot, qt_theme_applied):
    panel = ct.TrackCurationPanel()
    qtbot.addWidget(panel)
    assert panel.join_selected() is False
    assert panel.save() is None
    assert "Open a tracks table first" in panel.status.text()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_the_screen_puts_a_brush_over_an_opened_mask(qtbot, qt_theme_applied):
    from spacr.qt.screens.curate import CurateScreen

    screen = CurateScreen()
    qtbot.addWidget(screen)
    layer = screen.viewer.stack.add_labels(
        np.zeros((16, 16), np.int64), name="mask")
    panel = screen.attach_brush(layer, artifact="mask.tif")

    assert isinstance(panel, ct.BrushPanel)
    assert screen.brush is panel
    panel.start_painting()
    assert isinstance(screen.viewer.canvas.tool, ct.BrushTool)


def test_the_screen_says_whether_a_file_was_curated(qtbot, qt_theme_applied,
                                                    tmp_path):
    from spacr.qt.screens.curate import CurateScreen

    screen = CurateScreen()
    qtbot.addWidget(screen)
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"")
    screen._say_whether_curated(str(raw))
    assert "as the pipeline produced it" in screen.status.text()

    log = CurationLog(str(raw))
    log.append("paint", 1, n_changed=3)
    log.write_beside(raw)
    screen._say_whether_curated(str(raw))
    assert "curated by hand" in screen.status.text()


def test_the_screen_registers_itself_into_the_app_registry():
    from spacr.qt.app import APPS
    from spacr.qt.screens import curate

    curate.register()
    assert any(row[0] == curate.APP_KEY for row in APPS)
    assert curate.register() is None       # idempotent


# ---------------------------------------------------------------------------
# Stand-ins for the two Qt events the tool reads
# ---------------------------------------------------------------------------

class _Event:
    """A mouse event with just the two accessors ``BrushTool`` uses."""

    def __init__(self, button, buttons=None):
        self._button = button
        self._buttons = button if buttons is None else buttons

    def button(self):
        return self._button

    def buttons(self):
        return self._buttons


class _KeyEvent:
    def __init__(self, text, key=0):
        self._text = text
        self._key = key

    def text(self):
        return self._text

    def key(self):
        return self._key
