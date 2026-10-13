"""Instruction 359 part 1: every container on the Measurements tab resizes.

Reported 2026-09-02: "in Regression's Measurements section nested sections
become illegible because their containers are too small -- every container,
subsection and sub-subsection must be height-resizable and collapsible."

THE OUTER LEVEL WAS ALREADY DONE, which is why the report is about the inner
ones. The tab's three panels became splitter children that fold in
instruction 169/186, and `test_the_measurements_tab_sections_resize.py` pins
that. What was left underneath them was a flat column: four numbered workflow
steps drawn as bold labels, with the step's controls loose beneath them, and
every tall box in those steps pinned by ``setMaximumHeight(N)`` at a literal
N chosen against a 100 % font.

THE MEASUREMENT THAT MADE THIS AN ITEM AND NOT AN OPINION, taken offscreen on
2026-09-05 with `QApplication`'s font at 9 pt and again at 18 pt, which is the
``font_scale=2`` the maintainer runs:

    box                     cap       @100 %      @200 %
    attached databases      170 px    4 rows      3 rows
    join list                72 px    4 rows      2 rows
    merge report            190 px    11 lines    5 lines
    merge evidence          220 px    12 lines    6 lines
    column picker           180 px    10 rows     5 rows
    outcomes                120 px    7 lines     3 lines

Doubling the font halved the content of every nested container on the tab,
and no drag could get it back, because a maximum height is a maximum.

`test_the_font_scale_control_shows_the_defect_this_item_was_filed_about` is
the positive control: it rebuilds the OLD shape -- a box with a hard
`setMaximumHeight` -- and asserts it goes the wrong way. Without it, the
scaling assertions here would pass on a box that simply never grew.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QPlainTextEdit, QWidget

pytestmark = pytest.mark.qt

QWIDGETSIZE_MAX = 16777215


@pytest.fixture()
def scale(monkeypatch):
    """Drive the font-scale preference the whole tab sizes itself from.

    Patched on the module rather than written to QSettings: `scaled_px` reads
    `get_font_scale` through the module globals, and a test that wrote the
    real preference would resize the machine's own spaCR.
    """
    from spacr.qt import preferences

    def at(value: float):
        monkeypatch.setattr(preferences, "get_font_scale", lambda: value)

    return at


@pytest.fixture()
def tab(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _panels(tab):
    return (tab.databases, tab.regression)


def _boxes(tab):
    """Every draggable box on the tab, by its stored key."""
    found = {}
    for panel in _panels(tab):
        found.update(panel.__dict__.get("_boxes", {}))
    return found


# --------------------------------------------------------------------------
# the sub-subsection level: the numbered steps
# --------------------------------------------------------------------------

def test_every_numbered_step_has_a_collapse_control(tab):
    """"every container, subsection and sub-subsection". The steps are the
    third level, and they had no control at all."""
    from spacr.qt.widgets.measurement_scan_panel import (WORKFLOW_STEPS,
                                                         WorkflowStep)

    steps = {}
    for panel in _panels(tab):
        steps.update(panel.steps)
    assert sorted(steps) == [number for number, _title in WORKFLOW_STEPS]
    for number, step in steps.items():
        assert isinstance(step, WorkflowStep)
        button = step.fold_button()
        assert button.isCheckable()
        # KEYBOARD REACHABLE, which is the half of "collapsible" that an
        # arrow drawn with a mouse handler does not give you.
        assert button.focusPolicy() == Qt.StrongFocus
        assert str(number) in button.accessibleName()


def test_folding_a_step_hides_its_controls_and_gives_up_its_height(qtbot,
                                                                   qapp):
    """"Collapsing one section must release its space to siblings rather than
    leave a blank fixed-height box."

    ON A PANEL OF ITS OWN, not on `tab.databases`: inside the tab that panel
    is a splitter child two containers deep, so `resize` on it is overwritten
    by the next layout pass and the heights measured here would be the
    splitter's opinion rather than the fold's.
    """
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    panel = DatabaseMergePanel(lambda: [], threaded=False)
    qtbot.addWidget(panel)
    panel.resize(700, 900)
    panel.show()
    for _ in range(6):
        qapp.processEvents()
    first, last = panel.steps[1], panel.steps[3]
    small_before, tall_before = first.height(), last.height()

    first.set_expanded(False)
    for _ in range(6):
        qapp.processEvents()

    assert not panel.heading.isVisible()
    assert first.height() < small_before
    # THE SPACE WENT SOMEWHERE, and the somewhere is the sibling that carries
    # the panel's stretch -- not a gap where step 1 used to be.
    assert last.height() > tall_before


def test_a_folded_step_can_be_opened_again(tab):
    step = tab.databases.steps[2]
    step.set_expanded(False)
    assert not step.is_expanded()
    step.set_expanded(True)
    assert step.is_expanded()
    assert tab.databases.tables_state.isVisibleTo(step)


def test_clicking_the_heading_folds_the_step(tab, qtbot):
    """A 22 px arrow beside a heading that ignores clicks is a smaller target
    than every other folding heading in the tool."""
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent

    step = tab.databases.steps[3]
    click = QMouseEvent(QEvent.MouseButtonRelease, QPointF(4, 4),
                        QPointF(4, 4), Qt.LeftButton, Qt.LeftButton,
                        Qt.NoModifier)
    assert step.eventFilter(step.label, click) is True
    assert not step.is_expanded()


def test_the_step_heading_is_still_the_workflow_step_label(tab):
    """The stylesheet selects on ``WorkflowStep`` and a reader recognises the
    numbered line. The fold arrow went BESIDE it, not instead of it."""
    for panel in _panels(tab):
        for number, step in panel.steps.items():
            assert step.label.objectName() == "WorkflowStep"
            assert step.label.text().startswith(f"{number}.")


# --------------------------------------------------------------------------
# every box is draggable, and nothing is capped any more
# --------------------------------------------------------------------------

def test_every_tall_box_has_a_drag_handle(tab):
    from spacr.qt.widgets.height_grip import HeightGrip

    boxes = _boxes(tab)
    assert set(boxes) == {"databases", "tables", "report", "evidence",
                          "columns", "outcomes"}
    for key, grip in boxes.items():
        assert isinstance(grip, HeightGrip), key


def test_the_handle_is_the_one_home_already_had(tab):
    """"reuse it rather than inventing a second one" -- the same class, moved
    to a module of its own rather than copied into this one."""
    from spacr.qt.widgets import home
    from spacr.qt.widgets.height_grip import HeightGrip

    assert home._HeightGrip is HeightGrip


def test_every_box_has_real_room_to_grow(tab):
    """A handle whose ceiling is its opening height is not a resize."""
    for key, grip in _boxes(tab).items():
        floor, ceiling = grip.bounds()
        opened = grip.target_height()
        assert floor < opened < ceiling, key
        assert grip.resize_target(ceiling) == grip._target.height(), key


def test_the_module_pins_no_height_with_a_literal_any_more(tab):
    """A SOURCE RATCHET, and it is here because the measured checks cannot
    see this one.

    ``setMaximumHeight(190)`` and a height this handle set look identical
    from outside the widget -- both report a maximum of 190 -- and
    ``setFixedHeight`` overrides a previous cap, so a leftover literal is
    invisible to every assertion above until the font changes. This is what
    goes red the moment one comes back.
    """
    import inspect

    from spacr.qt.widgets import measurement_scan_panel

    source = inspect.getsource(measurement_scan_panel)
    lines = [line for line in source.splitlines()
             if ("setMaximumHeight(" in line or "setFixedHeight(" in line)
             and not line.lstrip().startswith(("#", "*"))
             and "``" not in line]
    assert lines == [], lines


def test_dragging_a_handle_resizes_its_box(tab):
    grip = _boxes(tab)["report"]
    opened = grip.target_height()
    grip.resize_target(opened + 200)
    assert grip.target_height() == opened + 200


def test_a_box_cannot_be_dragged_to_nothing_or_off_the_page(tab):
    """The floor is what makes "cannot be made unreachable" true rather than
    merely unlikely -- the same rule the tab's outer splitter follows."""
    for key, grip in _boxes(tab).items():
        floor, ceiling = grip.bounds()
        assert grip.resize_target(-9000) == floor, key
        assert grip.resize_target(99999) == ceiling, key


def test_the_handle_resizes_from_the_keyboard(tab):
    """A pointer-only handle is not an accessible resize affordance."""
    grip = _boxes(tab)["columns"]
    grip.setFocus()
    start = grip.target_height()
    grip.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Down,
                                 Qt.NoModifier))
    assert grip.target_height() > start
    grip.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Up,
                                 Qt.NoModifier))
    assert grip.target_height() == start
    grip.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_End,
                                 Qt.NoModifier))
    assert grip.target_height() == grip.bounds()[1]


def test_a_dragged_box_can_always_be_put_back(tab):
    """"Reset/default layout must remain available so a dragged section can
    never be made permanently unreachable.\""""
    grip = _boxes(tab)["evidence"]
    opened = grip.target_height()
    grip.resize_target(grip.bounds()[0])
    assert grip.target_height() != opened
    assert grip.reset() == opened


# --------------------------------------------------------------------------
# the font scale, which is what the report was actually about
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["databases", "tables", "report", "evidence",
                                 "columns", "outcomes"])
def test_a_box_opens_twice_as_tall_at_twice_the_font(qtbot, scale, key):
    """The measured defect, keyed on the maintainer's own setting.

    Built twice rather than resized once, because "every module must initially
    open wide enough to be usable" is a claim about the FIRST paint.
    """
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    heights = {}
    for value in (1.0, 2.0):
        scale(value)
        tab = MeasurementScanPanel(threaded=False)
        qtbot.addWidget(tab)
        heights[value] = _boxes(tab)[key].target_height()
    assert heights[2.0] == pytest.approx(2 * heights[1.0], abs=2)


def test_the_font_scale_control_shows_the_defect_this_item_was_filed_about(
        qtbot, scale):
    """POSITIVE CONTROL. The old shape, rebuilt, going red the old way.

    A box pinned with ``setMaximumHeight(190)`` keeps 190 px whatever the
    font, so doubling the font takes lines OFF it -- 11 to 5, measured. The
    assertions above would pass just as happily on a box that never changed
    size at all; this is what says they are measuring the right thing.
    """
    lines = {}
    for value in (1.0, 2.0):
        scale(value)
        box = QPlainTextEdit()
        qtbot.addWidget(box)
        font = box.font()
        font.setPointSizeF(9.0 * value)
        box.setFont(font)
        box.setMaximumHeight(190)                    # the shipped behaviour
        box.resize(400, 190)
        lines[value] = box.height() // box.fontMetrics().lineSpacing()
    assert lines[2.0] < lines[1.0]


def test_a_live_font_change_moves_the_border_with_it(tab, scale):
    """"Recompute the responsive layout after a font ... change." First launch
    is not the only moment the geometry moves."""
    grip = _boxes(tab)["report"]
    opened = grip.target_height()
    scale(2.0)
    grip._follow_the_font()
    assert grip.target_height() == pytest.approx(2 * opened, abs=2)
    assert grip.bounds()[1] == pytest.approx(2 * 900, abs=2)


# --------------------------------------------------------------------------
# and the arrangement survives leaving the module
# --------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A QSettings of our own. Writing the user's own would rearrange their
    tab, and reading it would make this depend on how they left it."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    path = tmp_path / "spacr.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_a_folded_step_and_a_dragged_box_come_back(tab, store, qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    tab.databases.steps[2].set_expanded(False)
    _boxes(tab)["report"].resize_target(420)
    tab.remember_section_layout()

    stored = store.get_section_layout(tab.LAYOUT_KEY)
    assert stored["steps"]["2"] is False
    assert stored["boxes"]["report"] == 420

    fresh = MeasurementScanPanel(threaded=False)
    qtbot.addWidget(fresh)
    assert fresh.databases.steps[2].is_expanded()
    fresh.restore_section_layout()
    assert not fresh.databases.steps[2].is_expanded()
    assert _boxes(fresh)["report"].target_height() == 420


def test_a_stored_height_is_replayed_in_lines_not_in_pixels(tab, store,
                                                            qtbot, scale):
    """Dragged at 100 %, restored at 200 %, and still the same box.

    Storing device pixels would hand a user who enlarged their font the SAME
    420 px and half the lines in it -- which is the bug this whole item is
    about, reintroduced through the settings file.
    """
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    _boxes(tab)["report"].resize_target(420)
    tab.remember_section_layout()

    scale(2.0)
    fresh = MeasurementScanPanel(threaded=False)
    qtbot.addWidget(fresh)
    fresh.restore_section_layout()
    assert _boxes(fresh)["report"].target_height() == 840


def test_an_unknown_step_or_box_in_a_stored_layout_is_ignored(tab):
    """A layout written by a version with five steps must not stop this one."""
    tab.databases.set_step_folds({99: False})
    tab.databases.set_box_heights({"nothing_like_this": 300})
    assert tab.databases.step_folds() == {1: True, 2: True, 3: True}


def test_a_widget_with_no_steps_is_not_asked_for_any(tab):
    """`_step_panels` answers about the tab's own panels, and the scan panel
    is a plain QWidget -- so it must not appear."""
    assert set(tab._step_panels()) == {tab.databases, tab.regression}
    assert not isinstance(QWidget(), type(tab.databases))
