"""``B13`` — the mouse and the panel of a manual count.

:mod:`tests.test_counting` covers the session with no Qt at all. What is left
here is what needs a widget: that a click becomes a marker at the world point
under the cursor, that the number beside the image is read from the markers on
every model event rather than kept in a variable that can drift, and that the
export button writes the clicks rather than the total.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from spacr.counting import CountingSession
from spacr.layers import LayerError, LayerStack, Spacing
from spacr.qt import counting_tool as ct
from spacr.qt import layer_viewer as lv


def _stack(size=64, step=1.0, units="px"):
    stack = LayerStack()
    stack.add_image(np.zeros((size, size), np.uint16), name="image",
                    spacing=Spacing.isotropic(2, step, units=units))
    return stack


def _canvas(qtbot, stack=None, width=200, height=200):
    canvas = lv.LayerCanvas(stack if stack is not None else _stack())
    qtbot.addWidget(canvas)
    canvas.resize(width, height)
    canvas._ensure_canvas()
    return canvas


def _panel(qtbot, canvas, **kwargs):
    panel = ct.CountingPanel(canvas, **kwargs)
    qtbot.addWidget(panel)
    return panel


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------

def test_a_click_places_a_marker_at_the_world_point_under_the_cursor(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.start_counting()

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))

    expected = canvas.canvas.world_at(45 - 1, 60 - 1)
    placed = panel.session.layer("infected").world[0]
    assert placed[0] == pytest.approx(expected["y"])
    assert placed[1] == pytest.approx(expected["x"])


def test_clicking_the_same_marker_takes_it_away(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.start_counting()

    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 1
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 0


def test_a_right_click_only_ever_removes(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.start_counting()

    QTest.mouseClick(canvas, Qt.RightButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 0, "a right click on nothing added a marker"
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    QTest.mouseClick(canvas, Qt.RightButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.total == 0


def test_the_number_row_chooses_the_class_and_backspace_undoes(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.start_counting()

    QTest.keyClick(canvas, Qt.Key_2)
    assert panel.session.active == "uninfected"
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.counts() == {"infected": 0, "uninfected": 1}

    QTest.keyClick(canvas, Qt.Key_Backspace)
    assert panel.session.total == 0
    # A digit no class claims is left for whoever else wants it.
    assert panel.tool.key(canvas, _key_event(Qt.Key_9, "9")) is False


def _key_event(key, text):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    return QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, text)


def test_the_tool_gives_the_canvas_its_mouse_back(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    picked = []
    canvas.picked.connect(lambda *args: picked.append(args))

    panel.count_button.setChecked(True)
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert picked == []

    panel.count_button.setChecked(False)
    assert canvas.tool is None
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert len(picked) == 1


def test_a_tool_needs_a_session():
    with pytest.raises(LayerError, match="CountingSession"):
        ct.CountingTool("not a session")


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_the_tally_is_redrawn_from_the_markers_on_every_change(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    seen = []
    panel.counts_changed.connect(seen.append)

    panel.session.add({"y": 10.0, "x": 10.0})
    assert panel.class_list.item(0).text().startswith("infected [1]   1")
    assert "1 total" in panel.total.text()
    assert seen[-1] == {"infected": 1, "uninfected": 0}

    # Removed through the layer, not through the panel: still in step.
    panel.session.layer("infected").remove(0)
    assert "nothing counted yet" in panel.total.text()
    assert seen[-1] == {"infected": 0, "uninfected": 0}


def test_selecting_a_row_chooses_the_class_a_click_counts_as(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.class_list.setCurrentRow(1)
    assert panel.session.active == "uninfected"
    panel.start_counting()
    QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, QPoint(60, 45))
    assert panel.session.counts()["uninfected"] == 1


def test_the_selected_row_follows_the_session(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.session.active = "uninfected"
    panel.refresh()
    assert panel.class_list.currentRow() == 1


def test_adding_a_class_gives_it_a_row_and_a_layer(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    name = panel.add_class()
    assert name == "class 3"
    assert panel.class_list.count() == 3
    assert f"count: {name}" in canvas.stack.names
    assert panel.add_class("mitotic") == "mitotic"


def test_undo_and_clear_go_through_the_panel(qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.session.add({"y": 1.0, "x": 1.0})
    panel.session.add({"y": 2.0, "x": 2.0})

    assert panel.undo() is True
    assert panel.session.total == 1
    assert panel.clear() == 1
    assert "nothing counted yet" in panel.total.text()
    assert panel.undo() is False


def test_the_export_writes_the_clicks_not_the_total(qtbot, qt_theme_applied,
                                                    tmp_path):
    import pandas as pd

    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    written = []
    panel.exported.connect(written.append)
    panel.session.add({"y": 4.0, "x": 8.0})
    panel.session.add({"y": 16.0, "x": 32.0}, "uninfected")

    path = panel.write(str(tmp_path / "counts.csv"))
    assert written == [path]
    frame = pd.read_csv(path)
    assert len(frame) == 2
    np.testing.assert_allclose(sorted(frame["y"]), [4.0, 16.0])

    tally = panel.write(str(tmp_path / "tally.csv"), summary=True)
    assert list(pd.read_csv(tally)["count"]) == [1, 1]


def test_a_write_that_fails_says_so_instead_of_raising(qtbot,
                                                       qt_theme_applied,
                                                       tmp_path):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    assert panel.write(str(blocked / "counts.csv")) is None
    assert "Could not write" in panel.total.text()


def test_the_panel_can_be_given_a_session_that_already_has_counts(
        qtbot, qt_theme_applied):
    canvas = _canvas(qtbot)
    session = CountingSession(canvas.stack, classes=["a", "b"], size=6.0)
    session.add({"y": 1.0, "x": 1.0}, "b")
    panel = _panel(qtbot, canvas, session=session)
    assert panel.session is session
    assert panel.class_list.count() == 2
    assert "b 1 (100%)" in panel.total.text()


def test_the_field_key_reaches_the_export(qtbot, qt_theme_applied, tmp_path):
    import pandas as pd

    from spacr.layers import FieldKey

    field = FieldKey(values=dict(zip(FieldKey.columns(),
                                     ("plate1", "A", "1", "1"))))
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas, field=field)
    panel.session.add({"y": 1.0, "x": 1.0})
    frame = pd.read_csv(panel.write(str(tmp_path / "counts.csv")))
    for column, value in field.values.items():
        assert str(frame[column][0]) == str(value)


def test_the_panel_lets_go_of_the_model_when_it_closes(qtbot,
                                                       qt_theme_applied):
    canvas = _canvas(qtbot)
    panel = _panel(qtbot, canvas)
    panel.start_counting()
    panel.close()

    assert canvas.tool is None
    # Unsubscribed: a model change no longer drives a destroyed panel.
    before = panel.total.text()
    panel.session.add({"y": 1.0, "x": 1.0})
    assert panel.total.text() == before
