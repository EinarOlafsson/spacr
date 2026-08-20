"""A figure whose Qt canvas is gone still renders. Instruction 166.

FROM THE MAINTAINER'S OWN LOG, ~/.spacr/logs/spacr.log, seventy times:

    figure render failed: Internal C++ object (FigureCanvasQTAgg) already
    deleted.

`savefig` renders through whatever canvas the figure currently holds, and a
figure that was ever shown in a Qt widget holds a FigureCanvasQTAgg -- which Qt
destroys with the widget. The figure itself is intact; only its painter is gone,
and every one of those seventy lines is a tile that silently did not render.
"""

import spacr


import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr.qt.widgets.figure_queue import (_retry_on_a_fresh_canvas,
                                           render_figure_to_png)


def _figure():
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


class _DeadCanvas:
    """Behaves like a Qt canvas whose C++ side has been destroyed."""

    def __getattr__(self, name):
        raise RuntimeError(
            "Internal C++ object (FigureCanvasQTAgg) already deleted.")


def test_a_figure_with_a_dead_canvas_still_renders(tmp_path):
    """THE REPORTED FAILURE, reproduced and then fixed."""
    fig = _figure()
    try:
        fig.canvas = _DeadCanvas()
        out = tmp_path / "tile.png"
        assert render_figure_to_png(fig, str(out)) is True
        assert out.exists() and out.stat().st_size > 0
    finally:
        plt.close(fig)


def test_the_retry_gives_the_figure_a_working_canvas(tmp_path):
    fig = _figure()
    try:
        fig.canvas = _DeadCanvas()
        assert _retry_on_a_fresh_canvas(fig, str(tmp_path / "a.png"),
                                        100, "#ffffff") is True
        # And the figure now has a canvas that is not the dead one.
        assert not isinstance(fig.canvas, _DeadCanvas)
    finally:
        plt.close(fig)


def test_an_ordinary_figure_is_unaffected(tmp_path):
    """The common path must not be routed through the retry."""
    fig = _figure()
    try:
        out = tmp_path / "b.png"
        assert render_figure_to_png(fig, str(out)) is True
        assert out.exists()
    finally:
        plt.close(fig)


def test_a_genuinely_unrenderable_figure_still_reports_false(tmp_path):
    """The retry must not turn a real failure into a silent success."""
    fig = _figure()
    try:
        assert render_figure_to_png(
            fig, "/nonexistent-directory-xyz/tile.png") is False
    finally:
        plt.close(fig)
