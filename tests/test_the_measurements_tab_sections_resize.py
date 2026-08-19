"""Every section of the Measurements tab has a movable border and a floor.

Reported 2026-08-19: "still cant resize the elements in the measurements tabs.
now they overlap in such a way i dont have access to some of them. make their
borders movable and make them not be able to overlap".

A QVBoxLayout gives its children whatever height it decides, so nothing could
be dragged -- and adding one more widget to it took the space out of the
others, which is what made sections unreachable.
"""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter


@pytest.fixture()
def panel(qtbot):
    from spacr.qt.widgets.measurement_scan_panel import MeasurementScanPanel

    widget = MeasurementScanPanel()
    qtbot.addWidget(widget)
    return widget


def test_the_sections_are_in_a_splitter(panel):
    assert isinstance(panel._sections, QSplitter)
    assert panel._sections.orientation() == Qt.Vertical
    assert panel._sections.count() >= 3


def test_a_section_cannot_be_collapsed_to_nothing(panel):
    """"make them not be able to overlap" -- a floor is what makes that true
    rather than merely unlikely."""
    assert panel._sections.childrenCollapsible() is False
    for index in range(panel._sections.count()):
        assert panel._sections.widget(index).minimumHeight() > 0


def test_dragging_a_border_moves_height_between_neighbours(panel):
    # A widget that was never shown has no geometry, so every size is 0 and
    # the drag has nothing to move.
    # The database and regression sections are hidden until a database is
    # attached, and a hidden splitter child holds no height -- so the drag
    # has to be tested between the ones actually on screen.
    panel.add_section(QLabel("a second visible section"))
    panel.resize(600, 900)
    panel.show()
    visible = [i for i in range(panel._sections.count())
               if panel._sections.widget(i).isVisible()]
    assert len(visible) >= 2, "nothing to drag between"
    sizes = panel._sections.sizes()
    assert sum(sizes) > 0, "the splitter was never laid out"

    first, second = visible[0], visible[1]
    moved = list(sizes)
    moved[first] += 60
    moved[second] = max(moved[second] - 60, 1)
    panel._sections.setSizes(moved)

    after = panel._sections.sizes()
    assert after[first] > sizes[first], "the border did not move"
    assert after[second] < sizes[second], "its neighbour did not give way"


def test_anything_added_becomes_its_own_section(panel):
    """A widget appended to the layout takes its height out of the others,
    which is how the tab came to overlap."""
    before = panel._sections.count()

    panel.add_section(QLabel("extra"))

    assert panel._sections.count() == before + 1
    assert panel._sections.widget(before).minimumHeight() > 0


def test_add_section_survives_being_handed_nothing(panel):
    before = panel._sections.count()

    panel.add_section(None)

    assert panel._sections.count() == before


def test_the_sweep_panel_is_a_section_and_not_a_layout_child(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    sections = [screen._scan_panel._sections.widget(i)
                for i in range(screen._scan_panel._sections.count())]
    assert screen._sweep_panel in sections
