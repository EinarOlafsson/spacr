"""The figure queue's refusals: resize, restyle and the right-click menu.

*The debounced resize.* ``_rerender_for_size`` runs on every settled drag of
the figure panel and ``_on_resize_rendered`` lands the worker's answer. Both
refuse a great deal -- a container too small to draw into, a figure that will
not report its DPI, a payload that is stale or empty -- and every refusal is
there to stop a bad render from replacing a good picture.

*The restyle.* ``refresh_current_figure`` / ``refresh_figure`` are what the
"Figure settings..." dialog and the context menu call, and callers believe
the bool they return: False means "nothing changed on screen".

*The right-click menu.* ``show_figure_menu`` is the only way to restyle a
figure from the picture itself, and its ``_redraw`` closure doubles as the
swap for "Show as ..." -- which builds a NEW Figure, so anything but a bool
arriving there means "replace, do not redraw".
"""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import pytest                                                     # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets.figure_queue import FigureQueue             # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def queue(qtbot):
    """A live queue on the raster path, torn down with the test.

    The live canvas is off deliberately: it short-circuits every raster
    refusal below.
    """
    widget = FigureQueue()
    qtbot.addWidget(widget)
    widget.set_live_canvas_enabled(False)
    return widget


@pytest.fixture
def figure():
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot([0, 1], [1, 0])
    yield fig
    plt.close(fig)


# ---------------------------------------------------------------------------
# _rerender_for_size -- the resize the user drives with a mouse drag.
# ---------------------------------------------------------------------------

def test_a_container_too_small_to_draw_into_is_left_alone(queue, figure,
                                                          tmp_path):
    """Mid-layout the panel is briefly a few pixels wide.

    Re-rendering there sets the figure to a size nobody asked for and throws
    the real one away; the strip thumbnail is built from that render, so the
    user is left with a smear.
    """
    queue._figures[0] = figure
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "f.png")
    queue._jobs.submit = lambda fn, cb: None

    queue._view.resize(40, 40)
    before = tuple(figure.get_size_inches())
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) == before

    queue._view.resize(900, 700)
    queue._rerender_for_size()
    assert tuple(figure.get_size_inches()) != before, (
        "the queue no longer resizes anything, so the refusal proves nothing")


def test_a_figure_that_cannot_be_measured_costs_no_render(queue, figure,
                                                          tmp_path):
    """A figure whose canvas Qt already destroyed cannot report its size.

    Seventy of those are in the maintainer's log. Rendering one produces
    nothing useful and burns a worker per resize frame.
    """
    class _Unmeasurable:
        def get_dpi(self):
            raise RuntimeError("Internal C++ object already deleted")

    submitted = []
    queue._jobs.submit = lambda fn, cb: submitted.append(fn)
    queue._view.resize(880, 660)

    queue._figures[0] = _Unmeasurable()
    queue._current = 0
    queue._png_paths[0] = str(tmp_path / "broken.png")
    queue._rerender_for_size()
    assert submitted == [], "a figure that cannot be measured was rendered"

    queue._figures[1] = figure
    queue._current = 1
    queue._png_paths[1] = str(tmp_path / "good.png")
    queue._rerender_for_size()
    assert len(submitted) == 1, "the healthy figure was refused too"


# ---------------------------------------------------------------------------
# The right-click menu, and the closure that doubles as the figure swap.
# ---------------------------------------------------------------------------

class _StubMenu:
    def __init__(self):
        self.executed_at = []

    def exec(self, position):
        self.executed_at.append(position)


@pytest.fixture
def menu_seam(monkeypatch):
    """Capture what ``show_figure_menu`` builds, without opening a menu.

    ``menu.exec()`` spins its own modal event loop, so a real menu would hang
    the suite.
    """
    from spacr.qt.widgets import figure_settings

    built = {}

    def _build(owner, figure, on_change=None, open_settings=None):
        built["owner"] = owner
        built["figure"] = figure
        built["on_change"] = on_change
        built["open_settings"] = open_settings
        built["menu"] = _StubMenu()
        return built["menu"]

    monkeypatch.setattr(figure_settings, "build_figure_context_menu", _build)
    return built


def test_a_bool_redraws_and_a_figure_replaces(queue, figure, menu_seam):
    """"Show as ..." builds a NEW Figure, so the callback doubles as the swap.

    ``figure_settings._replot`` hands the replacement to the same ``on_change``
    a colour toggle calls with a bool. Anything that is not a bool is the new
    figure, and everything holding the old one has to be pointed at it
    together, or the tile keeps the old picture and the menu looks broken.
    """
    from PySide6.QtCore import QPoint

    replacement, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([2, 3], [3, 2])
    try:
        queue.add_figure(figure)
        queue.show_figure_menu(QPoint(0, 0), idx=0)
        redraw = menu_seam["on_change"]

        assert redraw(False) is True, "a toggle did not redraw the figure"
        assert queue.figure_for(0) is figure

        assert redraw(replacement) is True, "the new figure was not adopted"
        assert queue.figure_for(0) is replacement, (
            "the queue still holds the figure the menu replaced")
    finally:
        plt.close(replacement)


def test_figure_settings_refuses_to_open_on_nothing(queue, monkeypatch):
    """The button is reachable before a run has produced a figure, and a
    dialog built on ``None`` raises the moment it reads a facecolor -- a
    traceback in the user's face for pressing a button they were offered."""
    from spacr.qt.widgets import figure_settings

    opened = []

    class _Dialog:
        def __init__(self, fig, *args, **kwargs):
            opened.append(fig)

        def exec(self):
            return 0

    monkeypatch.setattr(figure_settings, "FigureSettingsDialog", _Dialog)

    queue._open_figure_settings()
    assert opened == [], "a settings dialog was built on no figure"

    fig, ax = plt.subplots(figsize=(2.0, 1.5))
    ax.plot([0, 1], [0, 1])
    try:
        queue.add_figure(fig)
        queue._open_figure_settings()
        assert opened == [fig], "the dialog did not open on the real figure"
    finally:
        plt.close(fig)
