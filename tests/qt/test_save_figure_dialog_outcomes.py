"""Save figure: export styling, every save that writes nothing, colours and shapes.

The Save-figure dialog promises that the file gets exactly what the preview
showed and that the figure on screen is not touched.  ``style_for_file``
scales every text artist it can and inks legends and free text; ``save``
hands back an empty path whenever no file was written (a cancelled chooser, a
fast plot that fails or writes nothing, no figure, an unwritable folder); the
background combo's chooser row resolves to a real colour or puts the combo
back; and a graph shape is written into a Matplotlib figure's height.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.
"""
from __future__ import annotations

import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import QFileDialog  # noqa: E402

from spacr.qt.widgets import colour_picker  # noqa: E402
from spacr.qt.widgets import save_figure_dialog as SFD  # noqa: E402
from spacr.qt.widgets.save_figure_dialog import (  # noqa: E402
    SaveFigureDialog, style_for_file,
)

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _figure(width=8.0, height=6.0):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], label="series")
    ax.legend()
    ax.text(0.5, 0.5, "annotation")
    ax.set_title("title")
    return fig


class _FastPlot:
    """A pyqtgraph-shaped plot whose snapshot and export can be told to fail."""

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


def _sized(fig):
    return [t for t in fig.findobj(match=lambda o: hasattr(o, "get_fontsize"))]


def test_style_for_file_scales_and_inks_every_text_it_can(subtests):
    """None passes through, a refusing artist keeps its size while the rest double, and ink reaches legend and free text."""
    with subtests.test(msg="nothing_styles_to_nothing"):
        assert style_for_file(None, ink="#000000") is None
    with subtests.test(msg="refusing_artist_does_not_stop_the_rest"):
        fig = _figure()
        stubborn = fig.axes[0].title
        before = {id(t): t.get_fontsize() for t in _sized(fig)}

        def _refuse(*_args, **_kwargs):
            raise RuntimeError("this artist keeps its size")

        stubborn.set_fontsize = _refuse
        assert style_for_file(fig, font_scale=2.0) is fig
        for text in _sized(fig):
            expected = before[id(text)] * (1 if text is stubborn else 2)
            assert text.get_fontsize() == pytest.approx(expected)
    with subtests.test(msg="ink_reaches_legend_and_free_text"):
        fig = _figure()
        style_for_file(fig, ink="#123456", background="#FFFFFF")
        axes = fig.axes[0]
        assert [t.get_color() for t in axes.get_legend().get_texts()] == ["#123456"]
        assert [t.get_color() for t in axes.texts] == ["#123456"]


def test_save_returns_an_empty_path_whenever_no_file_was_written(qtbot, monkeypatch, tmp_path, subtests):
    """Cancelled chooser, failing or silent fast plots, no figure and an unwritable folder all return ''."""
    target = str(tmp_path / "never_written.png")
    cases = {
        "cancelled_chooser": (_figure, None),
        "fast_plot_export_raises": (
            lambda: _FastPlot(QPixmap(10, 10), RuntimeError("device is full")),
            target),
        "fast_plot_writes_nothing": (lambda: _FastPlot(QPixmap(10, 10), ""),
                                     target),
        "no_figure": (lambda: None, target),
        "unwritable_folder": (_figure, "/no/such/folder/for/spacr/figure.png"),
    }
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    for case_id, (make, path) in cases.items():
        with subtests.test(msg=case_id):
            dialog = SaveFigureDialog(make())
            qtbot.addWidget(dialog)
            assert (dialog.save(path) if path else dialog.save()) == ""
    assert not (tmp_path / "never_written.png").exists()


def test_background_chooser_row_selects_adds_or_restores_a_colour(qtbot, monkeypatch, subtests):
    """The chooser row reuses a matching row case-insensitively, adds a new colour before itself, or resets on cancel."""
    def make(answer):
        asked = []

        def _pick(*args, **_kwargs):
            asked.append(args)
            return answer

        monkeypatch.setattr(colour_picker, "pick_colour", _pick)
        dialog = SaveFigureDialog(_figure())
        qtbot.addWidget(dialog)
        box = dialog.background
        return box, box.findData(SFD._CHOOSE), asked

    with subtests.test(msg="cancelled_chooser_puts_the_combo_back"):
        box, chooser, _ = make(QColor())
        box.setCurrentIndex(2)
        box.setCurrentIndex(chooser)
        assert (box.currentIndex(), box.currentData()) == (0, "")
    with subtests.test(msg="known_colour_selects_its_row_every_time"):
        box, chooser, _ = make(QColor("#ffffff"))
        before, white = box.count(), box.findData("#FFFFFF")
        for _ in range(2):
            box.setCurrentIndex(chooser)
            assert (box.count(), box.currentIndex()) == (before, white)
    with subtests.test(msg="new_colour_added_before_the_chooser_once"):
        box, chooser, _ = make(QColor("#123456"))
        before = box.count()
        box.setCurrentIndex(chooser)
        assert box.count() == before + 1
        assert box.currentData() == "#123456"
        assert box.currentIndex() == before - 1
        assert box.itemText(box.count() - 1) == SFD._CHOOSE_LABEL
        box.setCurrentIndex(box.findData(SFD._CHOOSE))
        assert (box.count(), box.currentIndex()) == (before + 1, before - 1)
    with subtests.test(msg="ordinary_row_opens_no_chooser"):
        box, chooser, asked = make(QColor("#123456"))
        box.setCurrentIndex(2)
        assert asked == []
        assert box.currentData() == "#000000"
        box.setCurrentIndex(chooser)
        assert len(asked) == 1


def test_graph_shape_is_written_into_the_height_or_falls_back_to_as_drawn(qtbot, monkeypatch, subtests):
    """Shapes set a Matplotlib figure's height from its width; without a shape table only 'as drawn' is offered."""
    with subtests.test(msg="shapes_write_the_height"):
        dialog = SaveFigureDialog(_figure(8.0, 6.0))
        qtbot.addWidget(dialog)
        assert dialog._drawn_ratio == pytest.approx(0.75)
        assert dialog.height.value() == pytest.approx(6.0)
        assert dialog._shape_ratios["square"] == 1.0
        for shape, ratio in (("square", 1.0), ("tall", 1.5), ("", 0.75)):
            dialog.graph_shape.setCurrentIndex(dialog.graph_shape.findData(shape))
            assert dialog.height.value() == pytest.approx(
                round(dialog.width.value() * ratio, 2))
    with subtests.test(msg="no_shape_table_offers_only_as_drawn"):
        monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fast_plots",
                            types.ModuleType("spacr.qt.widgets.fast_plots"))
        dialog = SaveFigureDialog(_figure())
        qtbot.addWidget(dialog)
        assert dialog.graph_shape.count() == 1
        assert dialog.graph_shape.itemData(0) == ""
        assert dialog._shape_ratios == {}
