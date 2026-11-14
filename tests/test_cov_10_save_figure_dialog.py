"""Saving a figure whose preview cannot be built.

A figure that cannot be copied for preview is still offered as it stands, and
a fast plot with nothing drawn on it disables Save.  The empty-path outcomes
of ``save`` and ``style_for_file`` live in
``tests/qt/test_save_figure_dialog_outcomes.py``.
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

from PySide6.QtWidgets import QLabel         # noqa: E402

from spacr.qt.widgets.save_figure_dialog import (         # noqa: E402
    SaveFigureDialog)


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


def test_a_figure_that_cannot_be_copied_is_still_offered_as_it_stands(qtbot):
    """A figure holding a closure cannot be pickled, so there is no detached
    copy to restyle. The dialog explains that existing styling is retained
    while resolution, transparency and a tight crop still apply."""
    fig = _figure()
    fig._unpicklable = lambda: None
    dialog = SaveFigureDialog(fig)
    qtbot.addWidget(dialog)
    assert dialog.preview() is None
    assert dialog._canvas is None
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert any("current styling" in text and "tight crop" in text for text in labels)


def test_a_fast_plot_that_cannot_be_previewed_disables_saving(qtbot):
    """A pyqtgraph plot with nothing drawn on it has no snapshot. Offering
    Save would write an empty file the user would have to discover later."""
    dialog = SaveFigureDialog(_FastPlot(snapshot=RuntimeError("nothing drawn")))
    qtbot.addWidget(dialog)
    assert dialog.preview() is None
    assert dialog._save.isEnabled() is False
