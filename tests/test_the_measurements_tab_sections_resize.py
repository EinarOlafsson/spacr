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
    # `sections()` and not every splitter child: the last child is the layout
    # filler that makes folds collapse upward (186 C), and it is zero-minimum
    # on purpose -- it exists to give up all of its height.
    for section in panel.sections():
        assert section.minimumHeight() > 0


def test_dragging_a_border_moves_height_between_neighbours(panel):
    # A widget that was never shown has no geometry, so every size is 0 and
    # the drag has nothing to move.
    # The database and regression sections are hidden until a database is
    # attached, and a hidden splitter child holds no height -- so the drag
    # has to be tested between the ones actually on screen.
    panel.add_section(QLabel("a second visible section"))
    # AND THEY HAVE TO BE OPEN TO HAVE HEIGHT TO MOVE. A folded section is
    # pinned to its header (both minimum and maximum), which is what makes
    # a fold actually hand its space to the neighbours -- and since 176 A
    # only "Attached databases" starts open, so the drag would otherwise be
    # between two bars that cannot resize.
    for title in panel.section_titles():
        panel.set_section_expanded(title, True)
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
    before = len(panel.sections())

    panel.add_section(QLabel("extra"))

    assert len(panel.sections()) == before + 1
    assert panel.sections()[-1].minimumHeight() > 0


def test_add_section_survives_being_handed_nothing(panel):
    before = panel._sections.count()

    panel.add_section(None)

    assert panel._sections.count() == before


def test_the_sweep_panel_is_a_section_and_not_a_layout_child(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    # Each splitter child is now a CollapsibleSection wrapping the panel --
    # the panel is still a SECTION rather than a layout child, which is what
    # this test is about, so it is looked for through the wrapper.
    splitter = screen._scan_panel._sections
    contents = []
    for index in range(splitter.count()):
        child = splitter.widget(index)
        contents.append(child.content() if hasattr(child, "content") else child)
    assert screen._sweep_panel in contents


def test_every_section_can_be_folded_away(qtbot):
    """Reported 2026-08-19: "there are to many elements in the measurements tab".

    Four panels stacked is too many only when all four are open at once. The
    answer is not to remove one -- each is a step of the same workflow -- but
    to let the user fold the steps they are not on.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._scan_panel

    titles = panel.section_titles()
    assert len(titles) >= 4, titles
    assert "Gene × measurement sweep" in titles

    # WHICH ONE STARTS OPEN IS NOT THIS TEST'S BUSINESS, as it said when it
    # asserted "attached databases" -- and as of 2026-08-20 the answer is
    # NONE of them: "measurment sections should all start closed". That is
    # pinned in test_the_summary_buttons_are_not_stranded.py. What this test
    # is about is that each one CAN fold and reopen.
    for title in titles:
        panel.set_section_expanded(title, True)
        assert panel.is_section_expanded(title), f"{title} would not open"
        panel.set_section_expanded(title, False)
        assert not panel.is_section_expanded(title), f"{title} would not fold"
        panel.set_section_expanded(title, True)
        assert panel.is_section_expanded(title), f"{title} would not reopen"


def test_a_folded_section_gives_its_height_back(qtbot):
    """A fold that left the panel's height behind would not declutter anything."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.collapsible_section import CollapsibleSection

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._scan_panel
    splitter = panel._sections
    section = next(splitter.widget(i) for i in range(splitter.count())
                   if isinstance(splitter.widget(i), CollapsibleSection))

    # OPENED FIRST, because as of 2026-08-20 they all start closed
    # ("measurment sections should all start closed"). This test is about a
    # section GIVING ITS HEIGHT BACK, which needs it to have some.
    section.set_expanded(True)
    open_minimum = section.minimumHeight()
    section.set_expanded(False)
    assert section.maximumHeight() <= CollapsibleSection.FOLDED_HEIGHT
    assert section.minimumHeight() < open_minimum
    section.set_expanded(True)
    assert section.minimumHeight() == open_minimum


def test_the_header_still_says_what_was_folded(qtbot):
    """FOLDED, NOT REMOVED: a section that vanished would hide the feature."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.collapsible_section import CollapsibleSection

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    splitter = screen._scan_panel._sections
    for index in range(splitter.count()):
        child = splitter.widget(index)
        if isinstance(child, CollapsibleSection):
            child.set_expanded(False)
            assert child._header.isVisible() or not child.isVisible()
            assert child.title(), "a section folded to an unlabelled bar"
