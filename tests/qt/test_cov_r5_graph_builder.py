"""Closing a graph canvas whose figure canvas has no deferred draw.

:meth:`GraphCanvas.closeEvent` cancels the canvas's pending idle draw before
Qt deletes it -- a draw that fires after deletion is a segfault, not an
exception. The cancel is guarded, and this pins both sides of the guard: the
canvas that has the call gets it, and one that has not is still closed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtGui import QCloseEvent

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets import graph_builder as gb

pytestmark = pytest.mark.qt


class _RecordingCanvas:
    """A stand-in for the figure canvas that says whether it was asked."""

    def __init__(self):
        self.cancelled = 0

    def cancel_pending_draw(self):
        self.cancelled += 1


def _closed(view) -> bool:
    """Drive ``closeEvent`` with an event that starts out refused.

    ``QWidget.closeEvent`` accepts the event, so acceptance is the proof that
    the whole handler ran rather than stopping at the canvas guard.
    """
    event = QCloseEvent()
    event.ignore()
    view.closeEvent(event)
    return event.isAccepted()


def test_a_figure_canvas_without_a_deferred_draw_is_still_closed(qtbot):
    view = gb.GraphCanvas(link=LinkedSelection(), source="cov_r5")
    qtbot.addWidget(view)

    # The canvas is built in ``_build_ui`` and nothing public replaces it, so
    # the only way to present the handler with a canvas of another shape --
    # which is the situation ``hasattr`` is there for -- is to swap it here.
    recorder = _RecordingCanvas()
    view._canvas = recorder
    assert _closed(view)
    assert recorder.cancelled == 1, "the pending draw was never cancelled"

    # A canvas with no such call (matplotlib's own, before the subclass that
    # owns its timer) must not turn closing into an AttributeError.
    view._canvas = object()
    assert _closed(view)
    assert recorder.cancelled == 1, "the replaced canvas was asked again"


# --------------------------------------------------------------------------
# Three guards in this module cannot be made to fire, and are left standing
# rather than silenced. Written down here so the next reader does not spend
# the afternoon looking for an input that reaches them.
#
# 1. `GraphCanvas._draw_mean_bar`, the false side of "elif column:" (the
#    no-spread, no-column exit). `_draw_mean_bar` has exactly one caller,
#    `_draw_bar`, which reaches it only inside
#    `if other and other in rows.columns:` -- `other` is the argument passed
#    as `column`, so it is truthy at every call.
#
# 2. `GraphCanvas._draw_distribution`, the false side of "if key in parts:"
#    over ("cbars", "cmins", "cmaxes", "cmedians"). `parts` is what
#    `ax.violinplot(..., showmedians=True)` returned, and matplotlib puts
#    all four in it: `showextrema` is left at its default True, which is
#    what the three `c*` extrema keys are conditional on, and `showmedians`
#    is passed True for `cmedians`. Measured on matplotlib 3.8.4: the keys
#    are exactly bodies, cbars, cmaxes, cmedians, cmins.
#
# 3. `GraphCanvas._draw_legend`, the false side of "if legend.get_title() is
#    not None". `Legend.get_title()` returns the legend's title Text artist,
#    which is constructed with the legend; with no title it is a Text
#    holding the empty string, never None. Measured on matplotlib 3.8.4 for
#    a legend built with `title=None` and one built with `title="x"`.
