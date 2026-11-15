"""A value and its explanation, and a page that will not be read.

Pins the Save-figure dialog's less-travelled routes: the composer that
punctuates a value and its explanation, and a fast plot whose page size
cannot be read.  The colour chooser and the graph shapes live in
``tests/qt/test_save_figure_dialog_outcomes.py``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from spacr.qt.widgets import save_figure_dialog as SFD
from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _figure(width: float = 8.0, height: float = 6.0):
    fig = plt.figure(figsize=(width, height))
    axis = fig.add_subplot()
    axis.plot([0, 1], [0, 1])
    return fig


class _FastPlot:
    """A pyqtgraph-shaped plot whose page size can be told to fail."""

    def __init__(self, size=(100.0, 80.0)):
        self._size = size

    def export_size(self):
        if isinstance(self._size, Exception):
            raise self._size
        return self._size

    def styled_snapshot(self, width, **kwargs):
        return None

    def export_styled(self, path, **kwargs):
        return path


# --------------------------------------------------------------------------
# one composer for a value and its explanation
# --------------------------------------------------------------------------

def test_a_value_with_nothing_to_explain_is_left_exactly_as_it_is():
    """One composer, so a label and the note beside it are punctuated the
    same way. A value with no sentence behind it must not gain a dash."""
    assert SFD._with_reason("300 × 200 pixels") == "300 × 200 pixels"
    assert SFD._with_reason("300 × 200 pixels", "set on the plot's menu") == \
        "300 × 200 pixels — set on the plot's menu"
    assert SFD._with_reason("", "set on the plot's menu") == \
        "set on the plot's menu"


# --------------------------------------------------------------------------
# a page that cannot be read
# --------------------------------------------------------------------------

def test_a_plot_that_will_not_give_its_page_size_shows_no_page(qtbot):
    """The millimetres are the plot's own answer. A plot that cannot give
    them leaves the row blank rather than printing a number this dialog made
    up, and the dialog still opens."""
    good = SaveFigureDialog(_FastPlot((100.0, 80.0)))
    qtbot.addWidget(good)
    assert good._page_mm() == (100.0, 80.0)
    assert "100 × 80 mm" in good._size_note.text()

    broken = SaveFigureDialog(_FastPlot(RuntimeError("no page here")))
    qtbot.addWidget(broken)
    assert broken._page_mm() == (None, None)
    assert broken._size_note.text() == ""


# --------------------------------------------------------------------------
# `_clear_holder`'s "if widget is not None" has no false side to reach, and
# is left standing rather than silenced. `self._holder` is a private layout
# and nothing outside this module names it; the only four things ever put
# into it are `addWidget` calls (the un-previewable note, the preview canvas,
# the failure label and the empty-plot label), so every item `takeAt` hands
# back owns a widget. A spacer or a nested layout would make the guard fire,
# and no code path adds one.


# ---------------------------------------------------------------------------
# The figure-integrity check on the save path (coverage ratchet, 288)
# ---------------------------------------------------------------------------

class TestTheSaveCarriesTheIntegrityStamp:
    """A figure saved here meets the rule a pipeline figure meets: with the
    integrity check on, the file carries its stamp and the provenance is
    finished beside it; a check that finds nothing, fails, or cannot finish
    costs the stamp and never the file."""

    def _integrity(self, monkeypatch, *, report, stamp, finish=None):
        from spacr import plot

        seen = {}
        monkeypatch.setattr(plot, "_figure_integrity_enabled", lambda: True)
        monkeypatch.setattr(plot, "_integrity_report",
                            lambda figure, fmt, dpi, destination: report)
        monkeypatch.setattr(plot, "_integrity_metadata",
                            lambda rep, written: stamp)

        def finished(rep, path):
            seen["finished"] = (rep, path)
            if finish is not None:
                raise finish

        monkeypatch.setattr(plot, "_finish_integrity", finished)
        return seen

    @pytest.fixture
    def dialog(self, qtbot):
        made = SaveFigureDialog(_figure())
        qtbot.addWidget(made)
        return made

    def test_a_stamped_figure_is_written_and_its_provenance_finished(
            self, dialog, tmp_path, monkeypatch):
        report = {"panels": 1}
        seen = self._integrity(monkeypatch, report=report,
                               stamp={"Subject": "spaCR integrity"})
        out = tmp_path / "stamped.pdf"
        assert dialog.save(str(out)) == str(out)
        assert out.is_file()
        assert seen["finished"] == (report, str(out))

    def test_an_empty_stamp_or_no_report_still_writes(self, dialog, tmp_path,
                                                      monkeypatch):
        self._integrity(monkeypatch, report={"panels": 1}, stamp={})
        out = tmp_path / "plain.png"
        assert dialog.save(str(out)) == str(out)
        assert dialog._integrity_stamp(dialog._source, str(out), "png") == (
            {}, {"panels": 1})
        self._integrity(monkeypatch, report=None, stamp={"x": "y"})
        assert dialog._integrity_stamp(dialog._source, str(out), "png") == (
            {}, None)

    def test_provenance_that_cannot_be_finished_keeps_the_file(
            self, dialog, tmp_path, monkeypatch):
        self._integrity(monkeypatch, report={"panels": 1}, stamp={},
                        finish=OSError("read-only folder"))
        out = tmp_path / "kept.png"
        assert dialog.save(str(out)) == str(out)
        assert out.is_file()

    def test_a_check_that_cannot_run_costs_only_the_stamp(
            self, dialog, tmp_path, monkeypatch):
        from spacr import plot

        def broken(figure, fmt, dpi):
            raise ValueError("no pixel data in this panel")

        self._integrity(monkeypatch, report=None, stamp={})
        monkeypatch.setattr(plot, "_integrity_report", broken)
        out = tmp_path / "unchecked.png"
        assert dialog._integrity_stamp(dialog._source, str(out), "png") == (
            {}, None)
        assert dialog.save(str(out)) == str(out) and out.is_file()
