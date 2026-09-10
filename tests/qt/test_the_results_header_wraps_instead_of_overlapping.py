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

AND THEN THE MEASUREMENT ITSELF WAS WRONG. Every width below asks the panel
to resize and reads what came back -- but a top-level widget is clamped to
its own `minimumSizeHint`, which here is 503 px wide because the Annotation
check tab asks for 499 of them. So "420 px" and "1400 px" were the same
503 px panel, and with the placeholder run name on it ("No run", 40 px) the
five controls need 498 px and fit that with room over. The row was never
asked to wrap, and the test that checked it had could only ever say
`1 > 1`. The header is measured with a run name on it now, which is the
state it spends its working life in and the one element of it whose width
nobody controls.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QApplication, QComboBox,        # noqa: E402
                               QLabel, QWidget)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


#: A run name of the length these really have. The header is measured with
#: one on it, because the run name is its one element whose width nobody
#: controls -- with the placeholder ("No run", 40 px) the five controls need
#: 498 px, fit every width the panel can actually take, and never wrap, so a
#: header measured empty is a header measured in the one state that cannot
#: fail.
A_REAL_RUN_NAME = "Run: 20260819_ols_gene_pooled_plates1-4_rerun_v3"


def _settle(app):
    for _ in range(10):
        app.processEvents()


@pytest.fixture()
def panel(app):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    made = RegressionResultsPanel()
    made._run_label.setText(A_REAL_RUN_NAME)
    made.resize(1400, 700)
    made.show()
    _settle(app)
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
    _settle(app)
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
    _settle(app)
    past = [(type(w).__name__, x + wide, panel.width())
            for w, x, _y, wide, _h in _rows(panel)
            if x + wide > panel.width() + 1]
    assert not past, past


def test_the_row_actually_wraps_when_it_has_to(panel, app):
    """Not "it fits at 420 px" -- it fits because it is on more than one
    line. A row that reported no overlap because its controls had been
    hidden would pass the two tests above.

    ASKED FOR 420 PX AND MEASURED WHAT IT GOT. The panel has a floor of its
    own -- `minimumSizeHint()` is 503 px wide, which the Annotation check
    tab asks for -- and a top-level widget's `resize` is clamped to it. So
    "narrow" here is the narrowest the panel will go, not the number passed
    in, and the assertion below is that the panel really did narrow rather
    than that it reached 420.
    """
    panel.resize(1400, 700)
    _settle(app)
    row = panel._controls_row
    wide_width, wide_height = panel.width(), row.height()
    wide_lines = {y for _w, _x, y, _wd, _h in _rows(panel)}
    assert len(wide_lines) == 1, \
        "the row was already wrapped with room to spare"

    panel.resize(420, 700)
    _settle(app)
    assert panel.width() < wide_width, (
        "the panel never narrowed, so nothing here was measured twice over")
    narrow = _rows(panel)

    assert len(narrow) == 5, "a control disappeared instead of wrapping"
    assert len({y for _w, _x, y, _wd, _h in narrow}) > len(wide_lines)
    # THE HEIGHT, which is what "wrapped" means. Controls on two lines and a
    # row still one line tall would be controls drawn outside their own row.
    assert row.height() > wide_height, (row.height(), wide_height)


def test_the_row_is_taller_the_narrower_it_gets(panel, app):
    """The mechanism, asked of the header row itself rather than of one
    screen size: Qt only wraps if the row's height depends on its width, and
    the row is the widget that has to say so."""
    row = panel._controls_row

    assert row.hasHeightForWidth()
    assert row.heightForWidth(420) > row.heightForWidth(1400)


def test_every_control_is_still_reachable_at_the_narrowest_width(panel, app):
    """A wrap that clipped a combo would take the third colour channel away
    from anyone on a small screen."""
    panel.resize(420, 700)
    _settle(app)
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
