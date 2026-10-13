"""Runs, Results, Measurements and Cells -- all four, filled, and unfolded.

Instruction 236 C8 ("THE MEASUREMENTS TAB AND THE CELL TAB") and C10 ("NO
ELEMENT OVERLAPS ANOTHER, measured on rendered widgets rather than read off
a layout").

The overlap this file guards was found by driving the real screen on a real
results table: at a 577 px panel the Results header's second combo box began
48 px inside the first, the third began 27 px inside the second, and the
third ran 32 px past the right edge. Every one of those widgets had been
correctly added, in order, to a layout that reported no error at all -- a
QHBoxLayout asked for more room than it has does not shrink its children
below their minimum, it lets them overlap. Only rendered geometry catches
that.

The measurement is over SIBLINGS. A label inside a button overlaps it by
design, and comparing across parents would report every one of those.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QAbstractButton, QAbstractSpinBox,  # noqa: E402
                               QApplication, QComboBox, QLabel, QLineEdit,
                               QTabWidget)


WATCHED = (QLabel, QAbstractButton, QComboBox, QLineEdit, QAbstractSpinBox)

#: The four tabs, in the reading order instruction 128 J settled: pick a run
#: on the left tab, read that run on the next one.
EXPECTED_TABS = ["Runs", "Results", "Measurements", "Cells"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def screen(app):
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen("regression")
    made.resize(1600, 1000)
    made.show()
    # The figures card is hidden until a run produces something. A results
    # file opened from disk is exactly that, arriving by another door.
    made._figures_card.show()
    for _ in range(40):
        app.processEvents()
    yield made
    made.close()
    for _ in range(10):
        app.processEvents()
    made.deleteLater()
    app.processEvents()


def _boxes(root):
    """Every visible leaf widget, in root coordinates."""
    seen = []
    for kind in WATCHED:
        for child in root.findChildren(kind):
            if not child.isVisible() or child.width() < 3 \
                    or child.height() < 3:
                continue
            corner = child.mapTo(root, child.rect().topLeft())
            seen.append((child, corner.x(), corner.y(),
                         child.width(), child.height()))
    return seen


def _overlaps(root):
    found, seen = [], _boxes(root)
    for index, (a, ax, ay, aw, ah) in enumerate(seen):
        for b, bx, by, bw, bh in seen[index + 1:]:
            if a.parentWidget() is not b.parentWidget():
                continue
            if a.isAncestorOf(b) or b.isAncestorOf(a):
                continue
            if ax < bx + bw and bx < ax + aw and ay < by + bh \
                    and by < ay + ah:
                found.append((type(a).__name__, a.geometry(),
                              type(b).__name__, b.geometry()))
    return found


def _unfold(page, app):
    from spacr.qt.widgets.collapsible_section import CollapsibleSection

    for section in page.findChildren(CollapsibleSection):
        try:
            if not section.is_expanded():
                section.set_expanded(True)
        except Exception:                                    # noqa: BLE001
            pass
    for _ in range(20):
        app.processEvents()


class TestTheTabs:
    def test_all_four_are_there_and_in_order(self, screen):
        tabs = screen._results_tabs
        assert [tabs.tabText(i) for i in range(tabs.count())] == EXPECTED_TABS

    def test_each_one_opens(self, screen, app):
        tabs = screen._results_tabs
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            for _ in range(20):
                app.processEvents()
            page = tabs.widget(index)
            assert not page.isHidden(), tabs.tabText(index)
            assert page.width() > 100 and page.height() > 100

    def test_every_tab_has_a_tooltip_saying_what_it_is_for(self, screen):
        """Four tabs whose names are one word each; the tooltip is where
        the difference between Results and Measurements is stated."""
        tabs = screen._results_tabs
        for index in range(tabs.count()):
            assert tabs.tabToolTip(index).strip(), tabs.tabText(index)

    def test_the_cells_tab_is_always_present(self, screen):
        """Instruction 131 C and 129 both settled it: one tab per view,
        named, and a tab that cannot be filled says why rather than being
        absent. Most runs have no measurement database attached, which is
        exactly when a missing tab would be most confusing."""
        tabs = screen._results_tabs
        assert "Cells" in [tabs.tabText(i) for i in range(tabs.count())]


class TestNothingOverlaps:
    @pytest.mark.parametrize("width", [1600, 1200, 900])
    def test_no_two_siblings_overlap_on_any_tab(self, screen, app, width):
        screen.resize(width, 1000)
        for _ in range(20):
            app.processEvents()
        tabs = screen._results_tabs
        clashes = {}
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            for _ in range(20):
                app.processEvents()
            page = tabs.widget(index)
            _unfold(page, app)
            found = _overlaps(page)
            if found:
                clashes[tabs.tabText(index)] = found[:4]
        assert not clashes, clashes

    @pytest.mark.parametrize("width", [1600, 1200, 900])
    def test_nothing_runs_off_the_right_edge(self, screen, app, width):
        """Overlap and overflow are the same failure seen twice: a row that
        cannot fit puts its last child past the edge, where it is neither
        readable nor clickable."""
        screen.resize(width, 1000)
        for _ in range(20):
            app.processEvents()
        tabs = screen._results_tabs
        past = {}
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            for _ in range(20):
                app.processEvents()
            page = tabs.widget(index)
            _unfold(page, app)
            over = [(type(w).__name__, x + wide, page.width())
                    for w, x, _y, wide, _h in _boxes(page)
                    if x + wide > page.width() + 1]
            if over:
                past[tabs.tabText(index)] = over[:4]
        assert not past, past
