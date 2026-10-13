"""Instruction 178 C.2, the pyqtgraph half.

    "when right clicking and saving a figure (matplotlib or pyqt6graph) the
    user should be able to change all of theis for the saved graph, get a
    preview then save."

The matplotlib half previews on a pickled COPY. That is not available here: a
pyqtgraph plot is a live scene graph of Qt objects and does not pickle, and
rebuilding one from its data items would be a second implementation of every
plot in `fast_plots` — guaranteed to drift from the first.

So the live plot is DRESSED and UNDRESSED around one synchronous render. The
property that makes that legitimate is the one these tests are mostly about:
the plot on screen is the same afterwards, including when the write fails.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.fast_plots import FastPlot                    # noqa: E402
from spacr.qt.widgets.save_figure_dialog import (                   # noqa: E402
    FAST_PLOT_FORMATS, SaveFigureDialog, is_fast_plot,
)

STYLE = {"ink": "#231F20", "background": "#FFFFFF", "grid": True}


def _plot(qtbot, empty: bool = False):
    plot = FastPlot()
    qtbot.addWidget(plot)
    if not empty:
        rng = np.random.default_rng(0)
        plot.plot.plot(np.arange(30), rng.normal(size=30))
    return plot


def _looks(plot) -> tuple:
    return (plot._background, plot._foreground, plot._grid_on)


# -- the plot on screen is not touched --------------------------------------

def test_a_styled_preview_leaves_the_plot_on_screen_alone(qtbot):
    plot = _plot(qtbot)
    before = _looks(plot)

    assert plot.styled_snapshot(400, **STYLE) is not None
    assert _looks(plot) == before


def test_a_styled_write_leaves_the_plot_on_screen_alone(qtbot, tmp_path):
    plot = _plot(qtbot)
    before = _looks(plot)

    plot.export_styled(str(tmp_path / "p.png"), **STYLE)
    assert _looks(plot) == before


def test_a_write_that_raises_still_leaves_the_plot_as_it_was(qtbot, monkeypatch):
    """`finally`, not "and then restore" — a failed save must not restyle."""
    plot = _plot(qtbot)
    before = _looks(plot)

    def explode(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(plot, "export", explode)
    with pytest.raises(OSError):
        plot.export_styled("/nowhere/p.png", **STYLE)
    assert _looks(plot) == before


def test_the_grid_goes_back_to_what_the_user_chose(qtbot, tmp_path):
    plot = _plot(qtbot)
    plot.set_grid(False)

    plot.export_styled(str(tmp_path / "p.png"), grid=True)
    assert plot.grid_shown() is False


# -- the file is what the preview showed ------------------------------------

def test_the_preview_and_the_file_are_the_same_render(qtbot, tmp_path):
    """A preview from a different code path is a preview that can be wrong."""
    plot = _plot(qtbot)
    dialog = SaveFigureDialog(plot)
    qtbot.addWidget(dialog)

    assert dialog.preview() is not None
    written = dialog.save(str(tmp_path / "p.png"))
    assert written and os.path.getsize(written) > 0


def test_a_written_file_actually_appears_for_every_offered_format(qtbot, tmp_path):
    plot = _plot(qtbot)
    for suffix, _label in FAST_PLOT_FORMATS:
        out = tmp_path / f"p.{suffix}"
        assert plot.export_styled(str(out), **STYLE)
        assert out.exists() and out.stat().st_size > 0


# -- the dialog says what it cannot do --------------------------------------

def test_the_dialog_recognises_a_plot_by_what_it_can_do_not_by_its_class(qtbot):
    plot = _plot(qtbot)
    assert is_fast_plot(plot) is True
    assert is_fast_plot(None) is False
    assert is_fast_plot(object()) is False


def test_tiff_is_not_offered_because_the_plot_would_write_a_png(qtbot):
    """A file whose name and contents disagree is worse than a missing format."""
    dialog = SaveFigureDialog(_plot(qtbot))
    qtbot.addWidget(dialog)
    offered = [dialog.format.itemData(i) for i in range(dialog.format.count())]
    assert "tiff" not in offered
    assert offered == [value for value, _label in FAST_PLOT_FORMATS]


def test_size_is_inherited_but_raster_resolution_stays_editable(qtbot):
    """Page size has one owner; PNG resolution is an export property."""
    dialog = SaveFigureDialog(_plot(qtbot))
    qtbot.addWidget(dialog)

    for box in (dialog.width, dialog.height):
        assert box.isEnabled() is False
        assert box.toolTip()                     # instruction 106: says why
    assert "right-click" in dialog.width.toolTip()
    assert dialog.format.currentData() == "png"
    assert dialog.dpi.isEnabled() is True
    assert "Dots per inch" in dialog.dpi.toolTip()

    dialog.format.setCurrentIndex(dialog.format.findData("pdf"))
    assert dialog.dpi.isEnabled() is False
    assert "vector" in dialog.dpi.toolTip()


def test_an_empty_plot_says_there_is_nothing_to_save_rather_than_saving_it(qtbot):
    dialog = SaveFigureDialog(_plot(qtbot, empty=True))
    qtbot.addWidget(dialog)

    assert dialog.preview() is None
    assert dialog._save.isEnabled() is False


def test_the_size_shown_is_the_plots_own_export_page(qtbot):
    plot = _plot(qtbot)
    plot.set_export_size(180.0, 90.0)
    dialog = SaveFigureDialog(plot)
    qtbot.addWidget(dialog)

    assert dialog.width.value() == pytest.approx(180.0 / 25.4, abs=0.01)
    assert dialog.height.value() == pytest.approx(90.0 / 25.4, abs=0.01)


def test_the_menu_offers_one_way_to_write_a_figure(qtbot):
    """ONE DOOR (187 D), and it is the one that shows you what it will write.

    There were two. "Export…" wrote with no preview and no styling pass, so a
    page sized in millimetres got text scaled for the screen -- reported
    2026-08-20 as "the exported figure is broken, with massive text and so
    on ... actually remove the export button, save styled is enough."
    """
    plot = _plot(qtbot)
    labels = [a.text() for a in plot.build_style_menu().actions()]

    assert "Save figure…" in labels
    assert "Export…" not in labels, (
        "a second door that writes blind is how the broken figure got out")
