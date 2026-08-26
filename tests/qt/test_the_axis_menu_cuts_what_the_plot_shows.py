"""Right-clicking an AXIS of the gate editor, and everything it can refuse.

A right-click on the plotting rectangle asks "what can I do with this graph";
a right-click on an axis asks a different question -- how is this measurement
laid out, and how much of it should be drawn -- so it gets its own menu, and
:meth:`~spacr.qt.screens.gate_editor.GateEditorScreen._show_graph_menu` hands
over to it when the click landed on an axis.

Cutoffs change the VIEW and never the gates: a gate keeps the objects it
already holds whatever the axis is showing. Everything here is therefore
allowed to do nothing -- an empty axis, a cancelled dialog, text that is not a
number, a low end above the high end, a canvas that has not drawn yet -- and
each of those has to leave the plot alone and say why in the console rather
than raise out of a context-menu slot.

The menus are built against a recording stand-in rather than a real
``QMenu``: an offscreen Qt still runs ``exec``'s nested event loop with nobody
to dismiss the popup, and the run hangs for as long as anyone lets it.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.screens.gate_editor import (  # noqa: E402
    GateEditorScreen, _AxisCutoffDialog,
)
from spacr.qt.widgets.gate_canvas import AxisCutoff, CutoffError  # noqa: E402


def _objects(n=8):
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


def _axes(screen, x, y):
    """Put ``x`` and ``y`` on the two axis pickers, the way the user does."""
    for box, column in ((screen._x, x), (screen._y, y)):
        if column:
            box.setCurrentText(column)
        else:
            box.setCurrentIndex(-1)
    QApplication.processEvents()


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(_objects())
    # Flush the render this schedules while the screen is still alive: a
    # repaint that arrives after the widget is gone raises out of the Qt
    # event loop and fails whichever test happens to be running then.
    QApplication.processEvents()
    yield widget
    QApplication.processEvents()


@contextlib.contextmanager
def _no_canvas(screen):
    """The screen as it is before its canvas exists.

    Restored inside the test rather than by ``monkeypatch``: the panel closes
    its canvas on the way out, and a teardown that finds ``None`` there
    raises out of ``closeEvent``.
    """
    canvas = screen.gates.canvas
    screen.gates.canvas = None
    try:
        yield
    finally:
        screen.gates.canvas = canvas


class _Menu:
    """A QMenu that records instead of popping up."""

    built: list = []

    def __init__(self, _parent=None):
        self.rows = []
        self.at = None
        _Menu.built.append(self)

    def addSeparator(self):                      # noqa: N802 - Qt name
        self.rows.append(None)

    def addAction(self, label):                  # noqa: N802 - Qt name
        action = _Action(label)
        self.rows.append(action)
        return action

    def exec(self, point):
        self.at = point


class _Action:
    def __init__(self, label):
        self.label = label
        self.enabled = True
        self.checkable = False
        self.checked = False
        self.tooltip = ""
        self.calls = []

    def setEnabled(self, on):                    # noqa: N802 - Qt name
        self.enabled = bool(on)

    def setCheckable(self, on):                  # noqa: N802 - Qt name
        self.checkable = bool(on)

    def setChecked(self, on):                    # noqa: N802 - Qt name
        self.checked = bool(on)

    def setToolTip(self, text):                  # noqa: N802 - Qt name
        self.tooltip = text

    @property
    def triggered(self):
        return self

    def connect(self, slot):
        self.calls.append(slot)


@pytest.fixture
def recorded_menu(monkeypatch):
    import PySide6.QtWidgets as qtw

    _Menu.built = []
    monkeypatch.setattr(qtw, "QMenu", _Menu)
    return _Menu


# ---------------------------------------------------------------------------
# The cutoff dialog
# ---------------------------------------------------------------------------

def test_the_cutoff_boxes_open_showing_what_is_pinned_now(qtbot):
    dialog = _AxisCutoffDialog("X axis", "area", AxisCutoff(10.0, 80.0))
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "X axis cutoffs"
    assert dialog.values() == (10.0, 80.0)


def test_an_unpinned_end_opens_blank_and_reads_back_as_none(qtbot):
    """A blank box is a value: "let the data decide this end"."""
    dialog = _AxisCutoffDialog("Y axis", "intensity", AxisCutoff(None, 500.0))
    qtbot.addWidget(dialog)

    assert dialog._low.text() == ""
    assert dialog._high.text() == "500"
    assert dialog.values() == (None, 500.0)
    assert "intensity" in dialog._explain.text()


def test_text_that_is_not_a_number_is_refused_by_name(qtbot):
    dialog = _AxisCutoffDialog("X axis", "area", AxisCutoff())
    qtbot.addWidget(dialog)
    dialog._low.setText("about ten")

    with pytest.raises(CutoffError, match="about ten"):
        dialog.values()


# ---------------------------------------------------------------------------
# Asking for cutoffs
# ---------------------------------------------------------------------------

def _accept_with(monkeypatch, low, high):
    """Make the next cutoff dialog open with ``low``/``high`` and be accepted."""
    import spacr.qt.screens.gate_editor as GE

    real = GE._AxisCutoffDialog

    class _Answered(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._low.setText(low)
            self._high.setText(high)

        def exec(self):
            return 1

    monkeypatch.setattr(GE, "_AxisCutoffDialog", _Answered)


def test_accepting_the_dialog_applies_the_cutoffs_and_says_so(
        screen, monkeypatch):
    _axes(screen, "area", "intensity")
    _accept_with(monkeypatch, "20", "60")

    assert screen.ask_axis_cutoffs("x") == (20.0, 60.0)
    assert screen._cutoffs.get("area") == AxisCutoff(20.0, 60.0)
    assert "area shows" in screen.console.log.toPlainText()


def test_cancelling_the_dialog_changes_nothing(screen, monkeypatch):
    import spacr.qt.screens.gate_editor as GE

    _axes(screen, "area", "intensity")
    monkeypatch.setattr(GE._AxisCutoffDialog, "exec", lambda self: 0)

    assert screen.ask_axis_cutoffs("x") is None
    assert not screen._cutoffs.get("area").is_set


def test_a_cutoff_that_is_not_a_number_is_reported_not_raised(
        screen, monkeypatch):
    _axes(screen, "area", "intensity")
    _accept_with(monkeypatch, "about ten", "")

    assert screen.ask_axis_cutoffs("x") is None
    assert "Cutoffs not applied" in screen.console.log.toPlainText()
    assert not screen._cutoffs.get("area").is_set


def test_a_low_end_above_the_high_one_is_reported_not_raised(
        screen, monkeypatch):
    """Equal or crossed ends give an axis with no extent -- a blank panel."""
    _axes(screen, "area", "intensity")
    _accept_with(monkeypatch, "60", "20")

    assert screen.ask_axis_cutoffs("x") is None
    assert "Cutoffs not applied" in screen.console.log.toPlainText()
    assert not screen._cutoffs.get("area").is_set


def test_an_empty_axis_has_nothing_to_cut_off(screen, monkeypatch):
    import spacr.qt.screens.gate_editor as GE

    def _never(*_a, **_k):
        raise AssertionError("no dialog for an axis with no measurement")

    monkeypatch.setattr(GE, "_AxisCutoffDialog", _never)
    _axes(screen, "", "")

    assert screen.ask_axis_cutoffs("x") is None
    screen.set_axis_cutoffs("x", 1.0, 2.0)
    assert screen._cutoffs.columns() == ()


# ---------------------------------------------------------------------------
# Which axis was clicked
# ---------------------------------------------------------------------------

def test_a_click_with_no_canvas_lands_on_no_axis(screen):
    """A right-click can arrive before the panel has built its canvas."""
    with _no_canvas(screen):
        assert screen.axis_under(QPoint(3, 4)) is None


def test_a_figure_that_cannot_be_read_lands_on_no_axis(screen):
    """The canvas is there but its figure has gone -- still not an axis."""
    def _explode():
        raise RuntimeError("the figure has been closed")

    canvas = screen.gates.canvas
    figure = canvas.figure
    canvas.figure = _explode
    try:
        assert screen.axis_under(QPoint(3, 4)) is None
    finally:
        canvas.figure = figure


# ---------------------------------------------------------------------------
# Whether a scale can be offered
# ---------------------------------------------------------------------------

def test_an_empty_axis_offers_every_scale(screen):
    """With no measurement there is nothing to be non-positive about."""
    assert screen._axis_is_positive("") is True


def test_a_canvas_that_cannot_answer_does_not_grey_the_scales(
        screen, monkeypatch):
    """A greyed row is a claim about the data; unable-to-answer is not one."""
    def _explode(_column):
        raise RuntimeError("no frame loaded")

    monkeypatch.setattr(screen.gates.canvas, "_column_is_positive", _explode,
                        raising=False)

    assert screen._axis_is_positive("area") is True


def test_a_scale_the_settings_do_not_carry_is_left_alone(screen):
    """``z`` has no ``z_scale`` field, so choosing one writes nothing."""
    before = screen.settings()

    screen.set_axis_scale("z", "log")

    assert screen.settings() == before


# ---------------------------------------------------------------------------
# Narrowing the drawn panels
# ---------------------------------------------------------------------------

def test_panels_that_cannot_be_listed_are_not_narrowed(screen, monkeypatch):
    _axes(screen, "area", "intensity")
    screen.set_axis_cutoffs("x", 20.0, 60.0)

    def _explode():
        raise RuntimeError("the canvas has not drawn yet")

    monkeypatch.setattr(screen.gates.canvas, "panel_axes", _explode,
                        raising=False)

    screen._narrow_to_cutoffs()  # must not raise


def test_a_repaint_that_fails_does_not_lose_the_narrowing(screen, monkeypatch):
    """The limits are already on the axes; the repaint is what shows them."""
    _axes(screen, "area", "intensity")
    screen.set_axis_cutoffs("x", 20.0, 60.0)

    from matplotlib.figure import Figure

    axes = Figure().add_subplot(111)
    axes.set_xlabel("area")
    axes.plot([10.0, 80.0], [100.0, 800.0])

    monkeypatch.setattr(screen.gates.canvas, "panel_axes",
                        lambda: {"one": axes}, raising=False)

    def _explode():
        raise RuntimeError("the figure has been closed")

    monkeypatch.setattr(screen.gates.canvas, "figure", _explode,
                        raising=False)

    screen._narrow_to_cutoffs()  # must not raise

    assert axes.get_xlim() == (20.0, 60.0)


# ---------------------------------------------------------------------------
# The menu itself
# ---------------------------------------------------------------------------

def test_the_axis_menu_offers_the_scales_and_the_cutoffs(
        screen, recorded_menu):
    _axes(screen, "area", "intensity")

    screen._show_axis_menu("x", QPoint(7, 9))

    menu = recorded_menu.built[0]
    labels = [row.label for row in menu.rows if row is not None]
    assert labels[0] == "X axis: area"
    assert "linear" in labels and "log" in labels
    assert "Set cutoffs…" in labels
    assert menu.at == screen.gates.canvas.mapToGlobal(QPoint(7, 9))
    # The scale in force is ticked, and the rows that do something are wired.
    ticked = [row.label for row in menu.rows
              if row is not None and row.checkable and row.checked]
    assert ticked == ["linear"]
    assert all(row.calls for row in menu.rows
               if row is not None and row.enabled and row.label != "X axis: area"
               and not row.label.startswith("Clear cutoffs"))


def test_a_greyed_axis_row_carries_its_reason(screen, recorded_menu):
    """No cutoffs set means nothing to clear, and the menu says which column."""
    _axes(screen, "area", "intensity")

    screen._show_axis_menu("x", QPoint(0, 0))

    rows = {row.label: row for row in recorded_menu.built[0].rows
            if row is not None}
    clear = rows["Clear cutoffs"]
    assert clear.enabled is False
    assert "area" in clear.tooltip
    assert clear.calls == []


def test_a_right_click_on_an_axis_gets_the_axis_menu_not_the_plot_menu(
        screen, recorded_menu, monkeypatch):
    _axes(screen, "area", "intensity")
    monkeypatch.setattr(screen, "axis_under", lambda _point: "y")

    screen._show_graph_menu(QPoint(2, 3))

    labels = [row.label for row in recorded_menu.built[0].rows
              if row is not None]
    assert labels[0] == "Y axis: intensity"
    assert "Export gates to the database…" not in labels


def test_the_axis_menu_needs_a_canvas_to_pop_up_over(screen, recorded_menu):
    _axes(screen, "area", "intensity")

    with _no_canvas(screen):
        screen._show_axis_menu("x", QPoint(0, 0))

    assert recorded_menu.built == []
