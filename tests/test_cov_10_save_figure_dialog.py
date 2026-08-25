"""Saving a figure when the copy, the preview, or the write does not work.

The Save-figure dialog promises two things: the file gets exactly what the
preview showed, and the figure on screen is not touched. The branches here
are the ones where something between those two fails -- a figure that cannot
be copied for preview, an artist that will not take a new font size, a fast
plot with nothing drawn on it yet, a chooser the user cancels, and a
destination that cannot be written. In every one of them the dialog has to
say so and hand back an empty path, never a path to a file that is not there.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QPixmap                         # noqa: E402
from PySide6.QtWidgets import QFileDialog, QLabel         # noqa: E402

from spacr.qt.widgets import save_figure_dialog as SFD    # noqa: E402
from spacr.qt.widgets.save_figure_dialog import (         # noqa: E402
    SaveFigureDialog, style_for_file)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _figure():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], label="series")
    ax.legend()
    ax.text(0.5, 0.5, "annotation")
    ax.set_title("title")
    return fig


class _FastPlot:
    """A pyqtgraph-shaped plot whose two export methods can be told to fail."""

    def __init__(self, snapshot=None, exported=""):
        self._snapshot = snapshot
        self._exported = exported

    def export_size(self):
        return (100.0, 80.0)

    def styled_snapshot(self, width, **kwargs):
        if isinstance(self._snapshot, Exception):
            raise self._snapshot
        return self._snapshot

    def export_styled(self, path, **kwargs):
        if isinstance(self._exported, Exception):
            raise self._exported
        return self._exported


def test_styling_nothing_returns_nothing():
    """The preview path calls this with whatever the copy produced, which is
    ``None`` when the figure could not be copied. Returning ``None`` is what
    lets the dialog fall back to writing the figure as it stands."""
    assert style_for_file(None, ink="#000000") is None


def test_an_artist_that_refuses_a_font_size_does_not_stop_the_rest():
    """One artist that will not be resized must not leave the other text on
    the figure at the screen's sizes while the caller believes the whole
    figure was scaled."""
    fig = _figure()
    stubborn = fig.axes[0].title
    sizes_before = {id(t): t.get_fontsize()
                    for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize"))
                    if t is not stubborn}

    def _refuse(*args, **kwargs):
        raise RuntimeError("this artist keeps its size")

    stubborn.set_fontsize = _refuse
    styled = style_for_file(fig, font_scale=2.0)
    assert styled is fig
    grown = [t for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize"))
             if id(t) in sizes_before
             and t.get_fontsize() > sizes_before[id(t)]]
    assert grown


def test_the_ink_reaches_the_legend_and_the_free_text():
    """A legend label or an annotation left at the screen's colour is
    invisible on a printed page, and it is the part of the figure that says
    what the data means."""
    fig = _figure()
    style_for_file(fig, ink="#123456", background="#FFFFFF")
    axes = fig.axes[0]
    assert [t.get_color() for t in axes.get_legend().get_texts()] == ["#123456"]
    assert [t.get_color() for t in axes.texts] == ["#123456"]


def test_a_figure_that_cannot_be_copied_is_still_offered_as_it_stands(qtbot):
    """A figure holding a closure cannot be pickled, so there is no detached
    copy to restyle. The dialog says the file will be written exactly as the
    figure appears rather than showing a blank preview."""
    fig = _figure()
    fig._unpicklable = lambda: None
    dialog = SaveFigureDialog(fig)
    qtbot.addWidget(dialog)
    assert dialog.preview() is None
    assert dialog._canvas is None
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert any("exactly as it appears" in text for text in labels)


def test_a_fast_plot_that_cannot_be_previewed_disables_saving(qtbot):
    """A pyqtgraph plot with nothing drawn on it has no snapshot. Offering
    Save would write an empty file the user would have to discover later."""
    dialog = SaveFigureDialog(_FastPlot(snapshot=RuntimeError("nothing drawn")))
    qtbot.addWidget(dialog)
    assert dialog.preview() is None
    assert dialog._save.isEnabled() is False


def test_a_cancelled_chooser_writes_nothing(qtbot, monkeypatch):
    """Cancelling the file dialog must return an empty path, not the default
    name in the current working directory."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert dialog.save() == ""


def test_a_fast_plot_whose_export_fails_returns_no_path(qtbot, tmp_path):
    """Returning the requested path when nothing was written is how a run
    ends up with a figure list pointing at files that do not exist."""
    dialog = SaveFigureDialog(
        _FastPlot(snapshot=QPixmap(10, 10),
                  exported=RuntimeError("device is full")))
    qtbot.addWidget(dialog)
    assert dialog.save(str(tmp_path / "never_written.png")) == ""


def test_a_fast_plot_that_writes_nothing_returns_no_path(qtbot, tmp_path):
    """The exporter reports what it wrote. An empty answer means no file, and
    the dialog must not turn it into the path it was asked for."""
    dialog = SaveFigureDialog(_FastPlot(snapshot=QPixmap(10, 10), exported=""))
    qtbot.addWidget(dialog)
    assert dialog.save(str(tmp_path / "never_written.png")) == ""


def test_no_figure_at_all_writes_nothing(qtbot, tmp_path):
    """The dialog can be opened from a panel whose figure has gone. There is
    nothing to save, and the empty path is what tells the caller so."""
    dialog = SaveFigureDialog(None)
    qtbot.addWidget(dialog)
    assert dialog.save(str(tmp_path / "never_written.png")) == ""


def test_a_destination_that_cannot_be_written_returns_no_path(qtbot):
    """A folder that does not exist is the commonest cause. The failure has
    to come back as an empty path rather than as an exception out of a button
    press."""
    dialog = SaveFigureDialog(_figure())
    qtbot.addWidget(dialog)
    assert dialog.save("/no/such/folder/for/spacr/figure.png") == ""
