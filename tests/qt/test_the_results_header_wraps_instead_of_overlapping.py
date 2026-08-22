"""The Regression results header, measured rather than read off a layout.

Instruction 236 C10: "NO ELEMENT OVERLAPS ANOTHER, measured on rendered
widgets rather than read off a layout."

WHAT WAS FOUND. The header held a run name of unpredictable length and
three combo boxes with minimum widths of 140, 120 and 120. A QHBoxLayout
asked for more room than it has does not shrink its children past their
minimum -- it lets them OVERLAP. On the real screen at a 577 px panel the
second box began 48 px inside the first, the third began 27 px inside the
second, and the third ended 32 px past the right edge of the panel.

The fix is a wrapping row, so the overflow goes onto a second line. The
test measures `mapTo` geometry at several widths, because that is the only
thing that would have caught it: every one of those widgets was correctly
added, in order, to a layout that reported no error at all.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QApplication, QComboBox,        # noqa: E402
                               QLabel, QWidget)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(app):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    made = RegressionResultsPanel()
    made.resize(1400, 700)
    made.show()
    for _ in range(10):
        app.processEvents()
    yield made
    made.close()
    made.deleteLater()
    app.processEvents()


def _rows(panel):
    """Every visible header control, as (widget, x, y, w, h) in the panel."""
    watched = [panel._run_label, panel._colour_by_label, panel._colour_by,
               panel._colour_by_2, panel._colour_by_3]
    out = []
    for widget in watched:
        if not widget.isVisible():
            continue
        corner = widget.mapTo(panel, widget.rect().topLeft())
        out.append((widget, corner.x(), corner.y(),
                    widget.width(), widget.height()))
    return out


def _overlapping(placed):
    found = []
    for i, (a, ax, ay, aw, ah) in enumerate(placed):
        for b, bx, by, bw, bh in placed[i + 1:]:
            if ax < bx + bw and bx < ax + aw and ay < by + bh \
                    and by < ay + ah:
                found.append((a, b))
    return found


NARROW = [1400, 1000, 800, 640, 520, 420]


@pytest.mark.parametrize("width", NARROW)
def test_no_two_header_controls_overlap(panel, app, width):
    panel.resize(width, 700)
    for _ in range(10):
        app.processEvents()
    clashes = _overlapping(_rows(panel))
    assert not clashes, [
        (type(a).__name__, a.geometry(), type(b).__name__, b.geometry())
        for a, b in clashes]


@pytest.mark.parametrize("width", NARROW)
def test_no_header_control_runs_off_the_panel(panel, app, width):
    """Overlap and overflow are the same failure seen twice: a row that
    cannot fit puts its last child past the edge, where it is not clickable
    and not readable."""
    panel.resize(width, 700)
    for _ in range(10):
        app.processEvents()
    past = [(type(w).__name__, x + wide, panel.width())
            for w, x, _y, wide, _h in _rows(panel)
            if x + wide > panel.width() + 1]
    assert not past, past


def test_the_row_actually_wraps_when_it_has_to(panel, app):
    """Not "it fits at 420 px" -- it fits because it is on more than one
    line. A row that reported no overlap because its controls had been
    hidden would pass the two tests above."""
    panel.resize(1400, 700)
    for _ in range(10):
        app.processEvents()
    wide_lines = {y for _w, _x, y, _wd, _h in _rows(panel)}

    panel.resize(420, 700)
    for _ in range(10):
        app.processEvents()
    narrow = _rows(panel)
    assert len(narrow) == 5, "a control disappeared instead of wrapping"
    assert len({y for _w, _x, y, _wd, _h in narrow}) > len(wide_lines)


def test_every_control_is_still_reachable_at_the_narrowest_width(panel, app):
    """A wrap that clipped a combo would take the third colour channel away
    from anyone on a small screen."""
    panel.resize(420, 700)
    for _ in range(10):
        app.processEvents()
    for widget in (panel._colour_by, panel._colour_by_2, panel._colour_by_3):
        assert widget.isVisible()
        assert widget.width() >= 40
        assert widget.height() >= 10


def test_the_shared_flow_layout_is_where_both_users_can_reach_it():
    """It was private to the settings panel's chip strip. A second caller
    that copied it would have been a second copy to fix."""
    from spacr.qt.screens.settings_model import _FlowHost, _FlowLayout
    from spacr.qt.widgets.flow import FlowHost, FlowLayout

    assert _FlowLayout is FlowLayout
    assert _FlowHost is FlowHost


def test_the_wrapping_row_reports_a_height_that_grows(app):
    """The whole mechanism: Qt only wraps if the host says its height
    depends on its width."""
    from spacr.qt.widgets.flow import FlowHost, FlowLayout

    host = FlowHost()
    flow = FlowLayout(host, spacing=4)
    for _ in range(6):
        box = QComboBox(host)
        box.setMinimumWidth(120)
        flow.addWidget(box)
    assert host.hasHeightForWidth()
    assert host.heightForWidth(200) > host.heightForWidth(1200)
