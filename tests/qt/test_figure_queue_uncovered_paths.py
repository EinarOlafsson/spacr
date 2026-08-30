"""The figure queue's refusal paths, which every caller depends on silently.

:mod:`spacr.qt.widgets.figure_queue` carried fourteen ``# pragma: no cover``
markers -- more than any other module in the package. Every one of them sits on
an ``except`` clause, and item 288's argument against the marker applies with
full force here: code behind a pragma has by construction never executed under
test, so nobody has ever checked that these handlers do what their comments
claim.

They matter more than a defensive handler usually does, because of WHERE they
are. The figure queue is what a run streams its plots into. A handler that
swallowed the wrong thing here would not raise -- it would leave a figure
missing, a canvas half torn down, or a cache answering with the wrong number,
and the run would carry on and report success. Each test below therefore
asserts the FALLBACK VALUE, not merely that nothing was raised: a silent
``return`` and a silent ``return 20`` are different bugs.

Nothing here needs a display beyond the offscreen platform the suite already
uses, and no test renders at a real figure size.
"""
from __future__ import annotations

import builtins

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets import figure_queue as fq                   # noqa: E402
from spacr.qt.widgets.figure_queue import FigureQueue             # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def queue(qtbot):
    """A live queue widget, torn down with the test."""
    widget = FigureQueue()
    qtbot.addWidget(widget)
    return widget


class _RefusesAttributes:
    """An object that cannot be given an attribute.

    ``__slots__`` with no entries is the smallest thing that makes ``setattr``
    raise ``AttributeError`` for a name the class does not declare, which is
    the real shape of the failure: a caller handing in something that is not a
    Figure at all.
    """

    __slots__ = ()


# ---------------------------------------------------------------------------
# The per-figure text size, which issue #108 is about.
# ---------------------------------------------------------------------------

def test_a_text_size_is_remembered_on_a_real_figure():
    """The baseline: without this the refusal test proves nothing."""
    fig = plt.figure()
    try:
        fq.set_figure_text_size_override(fig, 14)
        assert fq.figure_text_size_override(fig) == 14
    finally:
        plt.close(fig)


@pytest.mark.parametrize("size, expected", [(14, 14), (-3, 0), (None, 0), ("9", 9)])
def test_a_remembered_text_size_is_clamped_and_coerced(size, expected):
    """``max(0, int(size or 0))`` is the contract, so pin all four of its cases.

    A negative size is not merely odd, it is a matplotlib error much later and
    far from here; zero is the documented "no override" value.
    """
    fig = plt.figure()
    try:
        fq.set_figure_text_size_override(fig, size)
        assert fq.figure_text_size_override(fig) == expected
    finally:
        plt.close(fig)


def test_an_object_that_refuses_attributes_does_not_break_the_caller():
    """The pragma at figure_queue.py:112.

    ``set_figure_text_size_override`` is called while a figure is being styled.
    Raising here would abandon the styling half-done, so the contract is that
    it returns quietly -- and the value simply is not remembered.
    """
    target = _RefusesAttributes()
    fq.set_figure_text_size_override(target, 12)          # must not raise
    assert fq.figure_text_size_override(target) == 0


def test_a_text_size_that_is_not_a_number_is_refused_quietly():
    """The same handler by its other route: ``int()`` raising ValueError."""
    fig = plt.figure()
    try:
        fq.set_figure_text_size_override(fig, "not a size")
        assert fq.figure_text_size_override(fig) == 0
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The name a figure is given in the grid captions.
# ---------------------------------------------------------------------------

def test_a_figures_own_label_names_it():
    """The baseline for the two refusals below."""
    fig = plt.figure()
    try:
        fig.set_label("a real label")
        assert fq.FigureQueue._figure_name(fig) == "a real label"
    finally:
        plt.close(fig)


def test_the_spacr_title_wins_over_matplotlibs_label():
    """Documented precedence, and the reason the function exists at all."""
    fig = plt.figure()
    try:
        fig.set_label("matplotlib's own")
        fig._spacr_title = "what spaCR called it"
        assert fq.FigureQueue._figure_name(fig) == "what spaCR called it"
    finally:
        plt.close(fig)


def test_no_figure_is_named_with_the_empty_string():
    """``None`` is a caption, not a crash: the grid still has a cell to draw."""
    assert fq.FigureQueue._figure_name(None) == ""


def test_something_that_is_not_a_figure_is_named_with_the_empty_string():
    """The pragma at figure_queue.py:1302.

    The caption is read for every item in the grid. One object that answers
    ``get_label`` with an exception must cost that one caption, not the grid.
    """

    class NotAFigure:
        _spacr_title = ""

        def get_label(self):
            raise RuntimeError("this is not a matplotlib Figure")

    assert fq.FigureQueue._figure_name(NotAFigure()) == ""


# ---------------------------------------------------------------------------
# The live canvas, and the raster it falls back to.
# ---------------------------------------------------------------------------

def test_without_a_qt_backend_the_queue_stays_on_the_raster(queue, monkeypatch):
    """The pragma at figure_queue.py:1549.

    ``show_live_canvas`` returning False is not a failure -- it is the queue
    choosing the raster path, which still shows the user their figure. The
    assertion that matters is that it returns False rather than raising, so
    the caller's ``if`` takes the other branch.
    """
    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.backends.backend_qtagg":
            raise ImportError("no Qt backend in this environment")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)
    fig = plt.figure()
    try:
        assert queue.show_live_canvas(fig) is False
    finally:
        plt.close(fig)


def test_a_queue_with_live_canvases_disabled_never_asks_for_a_backend(queue):
    """The guard ABOVE the import, which the test before it must not reach by."""
    queue._live_canvas_enabled = False
    fig = plt.figure()
    try:
        assert queue.show_live_canvas(fig) is False
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The preview size, which is asked for before the view has been laid out.
# ---------------------------------------------------------------------------

def test_a_laid_out_view_asks_for_a_size_inside_its_own_bounds(queue):
    """The baseline: the clamp is real and both ends of it are documented."""
    px = queue._preview_target_px()
    assert 600.0 <= px <= 2400.0


def test_a_view_that_cannot_be_measured_falls_back_to_the_declared_maximum(queue, monkeypatch):
    """The pragma at figure_queue.py:1714.

    Returning the maximum rather than a small default is deliberate: a preview
    drawn too small is a blurred figure the user cannot read, whereas one drawn
    too large only costs time.
    """

    class _Unmeasurable:
        def size(self):
            raise RuntimeError("this view has no window handle yet")

        def devicePixelRatioF(self):
            return 1.0

    monkeypatch.setattr(queue, "_view", _Unmeasurable())
    assert queue._preview_target_px() == float(queue.PREVIEW_MAX_PX)


# ---------------------------------------------------------------------------
# The two preference reads, which a headless process cannot satisfy.
# ---------------------------------------------------------------------------

def test_the_live_figure_cap_is_a_number_when_preferences_answer(queue):
    """The baseline for the refusal below."""
    assert isinstance(queue.live_figure_cap(), int)


def test_without_preferences_the_live_figure_cap_is_twenty(queue, monkeypatch):
    """The pragma at figure_queue.py:1944.

    Twenty is asserted by value on purpose. A cap that silently fell to zero
    would evict every figure the moment it arrived, and a run would finish
    having shown the user nothing -- which looks like a rendering bug, not a
    missing preference store.
    """
    import spacr.qt.preferences as prefs

    def refuse():
        raise RuntimeError("no QSettings in this process")

    monkeypatch.setattr(prefs, "live_figure_allowance", refuse)
    assert queue.live_figure_cap() == 20


def test_without_preferences_dynamic_figures_stay_on(queue, monkeypatch):
    """The pragma at figure_queue.py:1952.

    True, not False: an evicted figure reloading from its vector page is what
    makes eviction invisible. Defaulting the other way would turn a missing
    preference into permanently blank figures.
    """
    import spacr.qt.preferences as prefs

    def refuse():
        raise RuntimeError("no QSettings in this process")

    monkeypatch.setattr(prefs, "get_figure_dynamic", refuse)
    assert queue.dynamic_figures_enabled() is True


# ---------------------------------------------------------------------------
# Eviction, which must release the figure whatever pyplot says.
# ---------------------------------------------------------------------------

def test_eviction_reports_success_even_when_pyplot_will_not_close(queue, monkeypatch):
    """The pragma at figure_queue.py:2059.

    The close is an attempt to stop matplotlib's own registry retaining the
    figure. It is not what eviction MEANS -- the figure has already been
    spilled and dropped from the caches by the time it is tried -- so a close
    that fails must not report the eviction as not having happened. Returning
    False here would leave the caller trimming the same index forever.
    """
    fig = plt.figure()
    queue._figures[0] = fig
    monkeypatch.setattr(queue, "_spill_figure", lambda idx, figure: None)

    def refuse(_figure):
        raise RuntimeError("pyplot registry is gone")

    monkeypatch.setattr(plt, "close", refuse)
    try:
        assert queue._evict_live_figure(0) is True
        assert 0 not in queue._figures
    finally:
        monkeypatch.undo()
        plt.close(fig)
