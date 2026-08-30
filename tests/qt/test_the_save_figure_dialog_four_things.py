"""The save-figure dialog, opened from the volcano's right-click menu.

Four reports, all from one session on a real volcano:

1. "if i change graph shape to square the graph is replaced by this plot has
   noting drawn on it yet" -- the render raised, the exception went to a
   debug log nobody reads, and the dialog then blamed the plot for being
   empty. Both halves are tested here: the render no longer raises, and a
   render that does raise says so instead of calling a drawn plot blank.
2. "the DPI is still grayed out when PNG is chosen which does not make
   sense" -- the greying followed the KIND of plot rather than the FORMAT,
   which took away the one control that decides how big a raster file is.
3. "some of the options just say as shown, they should show the actual
   values, not as shown."
4. "all the settings this really needs are: background color / line width
   and color / text color / and aspect ratio: with the options square,
   vertical rectangle, horizontal rectangle."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFormLayout, QLabel

from spacr.multiple_testing import adjust_p_values
from spacr.qt.widgets.fast_plots import (CANVAS_SHAPES, FastPlot, VolcanoPlot,
                                         menu_entries)
from spacr.qt.widgets.save_figure_dialog import (EMPTY_PLOT, SaveFigureDialog,
                                                 style_for_file)


def _results(n: int = 160, seed: int = 0) -> pd.DataFrame:
    """A coefficient table shaped like the one `perform_regression` writes."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0, 1, n - 8),
                        10.0 ** (-rng.uniform(4, 8, 8))])
    q, _ = adjust_p_values(p, method="fdr_bh", alpha=0.05)
    return pd.DataFrame(
        {"feature": [f"fraction:grna[{100000 + i}_{i % 4 + 1}]"
                     for i in range(n)],
         "coefficient": rng.normal(0, 1, n),
         "p_value": p, "q_value": q,
         "multiple_testing_method": "fdr_bh"})


@pytest.fixture()
def volcano(qtbot):
    plot = VolcanoPlot()
    qtbot.addWidget(plot)
    plot.resize(900, 520)
    plot.set_results(_results())
    plot.show()
    qtbot.waitExposed(plot)
    return plot


@pytest.fixture()
def dialog(qtbot, volcano):
    made = SaveFigureDialog(volcano, parent=volcano)
    qtbot.addWidget(made)
    made.show()
    return made


def _row_labels(widget) -> list:
    """Every row name in the dialog's form, in the order they are read."""
    form = widget.findChild(QFormLayout)
    out = []
    for index in range(form.rowCount()):
        item = form.itemAt(index, QFormLayout.LabelRole)
        if item is not None and item.widget() is not None:
            out.append(item.widget().text())
    return out


def _shown(dialog) -> list:
    """What is in the preview area: pixmaps as the word, labels as text."""
    out = []
    for index in range(dialog._holder.count()):
        widget = dialog._holder.itemAt(index).widget()
        if isinstance(widget, QLabel):
            pixmap = widget.pixmap()
            out.append("pixmap" if pixmap is not None and not pixmap.isNull()
                       else widget.text())
    return out


def _choose(box, value) -> None:
    index = box.findData(value)
    assert index >= 0, f"{value!r} is not offered"
    box.setCurrentIndex(index)


# --------------------------------------------------------------------------- #
#  1. square previews and saves, and the file is square
# --------------------------------------------------------------------------- #

class TestSquareDrawsInsteadOfRaising:
    """`_dressed_for_the_file` had no `canvas_shape` keyword, so every shape
    the dialog offered raised TypeError inside the snapshot -- which the
    dialog swallowed and reported as an empty plot."""

    def test_the_snapshot_takes_the_shape_the_dialog_sends(self, volcano):
        """The exact call the dialog makes, which used to raise
        `TypeError: _dressed_for_the_file() got an unexpected keyword
        argument 'canvas_shape'`."""
        pixmap = volcano.styled_snapshot(400, canvas_shape="square")

        assert pixmap is not None
        assert pixmap.width() == pixmap.height() == 400

    def test_choosing_square_previews_a_square(self, dialog, qtbot):
        _choose(dialog.graph_shape, "square")
        qtbot.wait(10)
        preview = dialog.preview()

        assert _shown(dialog) == ["pixmap"], _shown(dialog)
        assert preview is not None and preview.width() == preview.height()

    def test_the_other_two_shapes_preview_at_their_proportion(self, dialog,
                                                              qtbot):
        for name, ratio in CANVAS_SHAPES:
            if name == "free":
                continue
            _choose(dialog.graph_shape, name)
            qtbot.wait(10)
            preview = dialog.preview()
            assert preview is not None, name
            assert preview.height() == pytest.approx(
                preview.width() * ratio, abs=2), name

    def test_the_written_png_is_square(self, dialog, tmp_path):
        _choose(dialog.graph_shape, "square")
        out = tmp_path / "volcano.png"

        assert dialog.save(str(out)) == str(out)
        image = QImage(str(out))
        assert image.width() == image.height() > 0

    def test_the_square_page_is_filled_rather_than_letterboxed(self, dialog,
                                                               tmp_path):
        """A square PAGE around an unchanged wide SCENE is not a square
        figure. Both exporters map the scene with KeepAspectRatio, so the
        plot used to sit in the top half of the file: measured before the
        fix, a 900x900 PNG whose ink stopped at row 428."""
        _choose(dialog.background, "#FFFFFF")
        _choose(dialog.graph_shape, "square")
        out = tmp_path / "filled.png"
        dialog.save(str(out))
        image = QImage(str(out))
        page = image.pixelColor(0, 0)

        inked = [y for y in range(image.height())
                 if any(image.pixelColor(x, y) != page
                        for x in range(0, image.width(), 7))]
        assert inked, "the file has nothing on it"
        assert max(inked) > image.height() * 0.9, (
            f"the ink stops at row {max(inked)} of {image.height()}")

    def test_a_square_pdf_gets_a_square_page(self, dialog, tmp_path):
        _choose(dialog.format, "pdf")
        _choose(dialog.graph_shape, "square")
        out = tmp_path / "volcano.pdf"

        assert dialog.save(str(out)) == str(out)
        assert out.stat().st_size > 0

    def test_a_square_svg_says_so_in_the_file(self, dialog, tmp_path):
        import re

        _choose(dialog.format, "svg")
        _choose(dialog.graph_shape, "square")
        out = tmp_path / "volcano.svg"
        dialog.save(str(out))
        head = out.read_text(errors="replace")[:600]
        found = re.search(r'width="([\d.]+)mm"\s+height="([\d.]+)mm"', head)

        assert found, head[:200]
        assert float(found.group(1)) == pytest.approx(float(found.group(2)),
                                                      rel=0.01)

    def test_the_plot_keeps_its_own_shape_afterwards(self, volcano, dialog,
                                                     tmp_path):
        """The styling is for the FILE. The scene is reshaped only while the
        render runs and is put back, geometry included."""
        before = volcano.plot.plotItem.geometry()
        _choose(dialog.graph_shape, "tall")
        dialog.save(str(tmp_path / "tall.png"))

        assert volcano.canvas_shape() == "free"
        assert volcano.plot.plotItem.geometry() == before


# --------------------------------------------------------------------------- #
#  1b. a preview that fails says why, and never calls a drawn plot empty
# --------------------------------------------------------------------------- #

class TestAFailedPreviewSaysWhy:

    def _break_the_render(self, plot):
        def refuse(*_args, **_kwargs):
            raise ValueError("the exporter refused this page")
        plot.styled_snapshot = refuse

    def test_the_reason_is_shown_rather_than_logged(self, volcano, dialog,
                                                    qtbot):
        self._break_the_render(volcano)
        _choose(dialog.graph_shape, "tall")
        qtbot.wait(10)

        assert dialog._trouble.isVisible()
        assert "ValueError" in dialog._trouble.text()
        assert "the exporter refused this page" in dialog._trouble.text()

    def test_a_drawn_plot_is_never_called_empty(self, volcano, dialog, qtbot):
        self._break_the_render(volcano)
        _choose(dialog.graph_shape, "tall")
        qtbot.wait(10)

        assert EMPTY_PLOT not in dialog._trouble.text()
        assert EMPTY_PLOT not in _shown(dialog)
        assert "nothing drawn" not in dialog._trouble.text().lower()

    def test_the_figure_that_did_draw_is_kept(self, volcano, dialog, qtbot):
        """Instruction 265's rule: keep the figure where possible, and never
        replace it with a wrong explanation."""
        assert _shown(dialog) == ["pixmap"]
        self._break_the_render(volcano)
        _choose(dialog.graph_shape, "tall")
        qtbot.wait(10)

        assert _shown(dialog) == ["pixmap"], (
            "the last drawing that worked was thrown away")
        assert "last one that could be drawn" in dialog._trouble.text()

    def test_saving_is_refused_while_the_settings_cannot_be_drawn(
            self, volcano, dialog, qtbot):
        self._break_the_render(volcano)
        _choose(dialog.graph_shape, "tall")
        qtbot.wait(10)

        assert dialog._save.isEnabled() is False

    def test_a_write_that_fails_says_why_too(self, volcano, dialog, tmp_path):
        def refuse(*_args, **_kwargs):
            raise OSError("read-only file system")
        volcano.export_styled = refuse

        assert dialog.save(str(tmp_path / "x.png")) == ""
        assert "read-only file system" in dialog._trouble.text()

    def test_an_empty_plot_still_says_it_is_empty(self, qtbot):
        """The sentence is reserved for the case it describes."""
        blank = FastPlot()
        qtbot.addWidget(blank)
        made = SaveFigureDialog(blank)
        qtbot.addWidget(made)

        assert _shown(made) == [EMPTY_PLOT]
        assert made._save.isEnabled() is False
        assert made._trouble.isVisible() is False


# --------------------------------------------------------------------------- #
#  2. the resolution follows the FORMAT
# --------------------------------------------------------------------------- #

class TestTheResolutionFollowsTheFormat:

    def test_png_has_a_live_resolution_on_a_pyqtgraph_plot(self, dialog):
        _choose(dialog.format, "png")

        assert dialog.dpi.isEnabled() is True

    def test_pdf_and_svg_have_none(self, dialog):
        for suffix in ("pdf", "svg"):
            _choose(dialog.format, suffix)
            assert dialog.dpi.isEnabled() is False, suffix
            assert "vector" in dialog.dpi.toolTip(), suffix

    def test_it_comes_back_when_png_is_chosen_again(self, dialog):
        _choose(dialog.format, "pdf")
        _choose(dialog.format, "png")

        assert dialog.dpi.isEnabled() is True

    def test_six_hundred_dpi_has_more_pixels_than_one_fifty(self, dialog,
                                                            tmp_path):
        _choose(dialog.graph_shape, "square")
        dialog.dpi.setValue(150)
        low = tmp_path / "low.png"
        dialog.save(str(low))

        dialog.dpi.setValue(600)
        high = tmp_path / "high.png"
        dialog.save(str(high))

        small, large = QImage(str(low)), QImage(str(high))
        assert large.width() == pytest.approx(small.width() * 4, rel=0.02)
        assert large.width() > small.width()

    def test_the_resolution_is_what_decides_the_count(self, volcano):
        """180 mm is 7.087 in, so 300 dpi is 2126 pixels across."""
        width, height = volcano.raster_pixels(900, 433)
        assert (width, height) == (900, 433), "no resolution set: keep the scene"

        volcano._export_dpi = 300
        try:
            width, _height = volcano.raster_pixels(900, 433, 180.0, None)
        finally:
            volcano._export_dpi = None
        assert width == 2126

    def test_a_vector_format_is_not_handed_a_resolution(self, dialog):
        _choose(dialog.format, "pdf")

        assert "dpi" not in dialog._for_the_file()


# --------------------------------------------------------------------------- #
#  3. an inherited row shows its value
# --------------------------------------------------------------------------- #

class TestTheInheritedRowsShowTheirValues:

    def test_the_size_row_is_named_size_and_not_a_sentence(self, dialog):
        assert "size" in _row_labels(dialog)
        assert not any(label.startswith("size —")
                       for label in _row_labels(dialog))

    def test_the_page_is_shown_in_millimetres(self, dialog):
        _choose(dialog.graph_shape, "square")

        assert "180 × 180 mm" in dialog._size_note.text()

    def test_the_page_is_shown_in_inches_too(self, dialog):
        _choose(dialog.graph_shape, "square")

        assert dialog.width.value() == pytest.approx(180.0 / 25.4, abs=0.01)
        assert dialog.height.value() == pytest.approx(180.0 / 25.4, abs=0.01)

    def test_where_it_comes_from_is_the_quieter_half(self, dialog):
        """The explanation is kept, after the number and as hover help."""
        text = dialog._size_note.text()

        assert text.index("mm") < text.index("right-click menu")
        assert dialog._size_note.toolTip()
        assert dialog._size_note.isEnabled() is False

    def test_the_resolution_row_shows_the_pixels_it_will_write(self, dialog):
        _choose(dialog.format, "png")
        _choose(dialog.graph_shape, "square")
        dialog.dpi.setValue(300)

        assert "2126 × 2126 pixels" in dialog._resolution_note.text()

    def test_the_shown_pixels_are_the_pixels_written(self, dialog, tmp_path):
        _choose(dialog.graph_shape, "wide")
        dialog.dpi.setValue(200)
        shown = dialog._resolution_note.text()
        out = tmp_path / "wide.png"
        dialog.save(str(out))
        image = QImage(str(out))

        assert f"{image.width()} × {image.height()} pixels" in shown, shown

    def test_the_count_follows_the_resolution_that_is_typed(self, dialog):
        _choose(dialog.graph_shape, "square")
        dialog.dpi.setValue(150)
        low = dialog._resolution_note.text()
        dialog.dpi.setValue(600)

        assert low != dialog._resolution_note.text()
        assert "4252 × 4252 pixels" in dialog._resolution_note.text()


# --------------------------------------------------------------------------- #
#  4. four settings, and everything dropped has a home
# --------------------------------------------------------------------------- #

class TestTheDialogOffersTheFourSettings:

    #: What the maintainer asked for, in the words the item settled on.
    WANTED = ["background colour", "line colour", "line width", "text colour",
              "graph shape"]

    #: What the file IS, as opposed to how the figure looks.
    THE_FILE = ["format", "resolution", "size"]

    def test_the_rows_are_the_four_settings_and_the_file(self, dialog):
        assert _row_labels(dialog) == self.WANTED + self.THE_FILE

    def test_the_shapes_are_the_three_the_item_names(self, dialog):
        offered = [dialog.graph_shape.itemText(i)
                   for i in range(dialog.graph_shape.count())]

        assert offered == ["as drawn", "square", "horizontal rectangle",
                           "vertical rectangle"]

    def test_no_row_says_aspect_ratio(self, dialog):
        """The word that meant two different quantities is not reintroduced,
        in a label or in a tooltip."""
        said = [label.text() for label in dialog.findChildren(QLabel)]
        tips = [w.toolTip() for w in (dialog.graph_shape, dialog.line_width,
                                      dialog.ink, dialog.background)]

        assert not [t for t in said + tips if "aspect ratio" in t.lower()]

    @pytest.mark.parametrize("gone", ["aspect", "text_px", "x_title",
                                      "y_title", "font_scale", "grid"])
    def test_the_plots_own_settings_are_not_offered_a_second_time(
            self, dialog, gone):
        assert not hasattr(dialog, gone), (
            f"{gone} is a property of the plot and belongs on its own menu")

    @pytest.mark.parametrize("entry", ["Lock axis scales", "Font size…",
                                       "Axis labels…", "Font colour…",
                                       "Grid", "Line colour…", "Line width…",
                                       "Exported page size…"])
    def test_everything_dropped_is_reachable_on_the_plots_menu(self, volcano,
                                                               entry):
        """Nothing is silently dropped: each control removed from the dialog
        is still reachable where the canvas controls already live."""
        texts = [action.text() for action in menu_entries(
            volcano.build_style_menu())]

        assert any(entry in text for text in texts), (
            f"{entry!r} is on neither the dialog nor the menu")

    def test_the_four_settings_actually_reach_the_render(self, dialog):
        _choose(dialog.ink, "#231F20")
        _choose(dialog.line_colour, "#FFFFFF")
        dialog.line_width.setValue(3.0)
        _choose(dialog.graph_shape, "square")

        assert dialog._extra_styling() == {
            "canvas_shape": "square", "line_width": 3.0,
            "line_colour": "#FFFFFF", "text_colour": "#231F20"}

    def test_an_untouched_dialog_asks_for_no_styling(self, dialog):
        """Saving is still one click for a user who wants what is on screen."""
        assert dialog._extra_styling() == {}


class TestTheTwoInksAreSeparate:
    """"line width and color / text color" -- two controls, two halves."""

    @pytest.fixture()
    def figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        made, axes = plt.subplots()
        axes.plot([1, 2, 3], [1, 4, 9])
        axes.set_xlabel("gene")
        yield made
        plt.close(made)

    def test_the_text_colour_leaves_the_spines_alone(self, figure):
        before = figure.axes[0].spines["bottom"].get_edgecolor()
        style_for_file(figure, text_colour="#231F20")

        assert figure.axes[0].xaxis.label.get_color() == "#231F20"
        assert figure.axes[0].spines["bottom"].get_edgecolor() == before

    def test_the_line_colour_leaves_the_text_alone(self, figure):
        before = figure.axes[0].xaxis.label.get_color()
        style_for_file(figure, line_colour="#FF0000")

        assert figure.axes[0].xaxis.label.get_color() == before
        assert figure.axes[0].spines["bottom"].get_edgecolor()[0] == 1.0

    def test_no_grid_control_means_the_grid_is_left_alone(self, figure):
        """The dialog no longer has a grid box, so it passes nothing -- and
        a default of False would silently turn off a grid the figure was
        drawn with."""
        figure.axes[0].grid(True)
        style_for_file(figure, text_colour="#231F20")

        assert bool(figure.axes[0].xaxis._major_tick_kw.get("gridOn"))


# ---------------------------------------------------------------------------
# choosing a colour that is not on the list
# ---------------------------------------------------------------------------

def _colour_box(dialog):
    """The first combo box that offers the "choose a colour…" entry."""
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.widgets.save_figure_dialog import _CHOOSE

    for box in dialog.findChildren(QComboBox):
        if box.findData(_CHOOSE) >= 0:
            return box
    pytest.skip("this dialog offers no colour chooser")


def test_a_chosen_colour_is_added_to_the_list_and_selected(dialog,
                                                           monkeypatch):
    """The chooser entry has to become a real, re-selectable colour.

    Leaving the box on "choose a colour…" would save the figure with whatever
    the placeholder resolves to, and the user has no way to tell that from the
    colour they picked.
    """
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker

    box = _colour_box(dialog)
    before = box.count()
    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor("#123456"))

    from spacr.qt.widgets.save_figure_dialog import _CHOOSE

    box.setCurrentIndex(box.findData(_CHOOSE))

    assert box.count() == before + 1, "the chosen colour was not kept"
    assert box.currentData() == "#123456"
    # ...and the chooser is still the last entry, so it can be used again.
    assert box.itemData(box.count() - 1) == _CHOOSE


def test_cancelling_the_colour_chooser_falls_back_to_the_first_entry(
        dialog, monkeypatch):
    """An invalid colour is what a cancelled dialog returns.

    Staying on the chooser entry would leave the form in a state that is not
    a colour at all, so the box returns to a real choice.
    """
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker
    from spacr.qt.widgets.save_figure_dialog import _CHOOSE

    box = _colour_box(dialog)
    before = box.count()
    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor())      # invalid

    box.setCurrentIndex(box.findData(_CHOOSE))

    assert box.count() == before, "a cancelled choice added an entry"
    assert box.currentIndex() == 0
    assert box.currentData() != _CHOOSE


def test_choosing_a_colour_already_on_the_list_selects_it_rather_than_adding(
        dialog, monkeypatch):
    """Two entries with the same colour is a list the user cannot read.

    The case is the whole difficulty. ``QColor.name()`` answers in lower case
    and the shipped entries are written upper, so an exact ``findData`` misses
    "white" for ``#ffffff`` and inserts a second, visually identical row --
    once for every time the user picks a colour the list already had. The
    dropdown grows without limit and every added row looks like the one above
    it.
    """
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker
    from spacr.qt.widgets.save_figure_dialog import _CHOOSE

    box = _colour_box(dialog)
    existing = next(
        (box.itemData(i) for i in range(box.count())
         if isinstance(box.itemData(i), str)
         and box.itemData(i).startswith("#")), None)
    assert existing is not None, (
        "the colour box offers no plain colour to re-select, so this "
        "dialog's list is not what this test assumes")
    before = box.count()
    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda *a, **k: QColor(existing))

    box.setCurrentIndex(box.findData(_CHOOSE))

    assert box.count() == before
    assert box.currentData() == existing


def test_the_chooser_does_not_re_enter_itself(dialog, monkeypatch):
    """``_picking`` guards a signal the chooser's own setCurrentIndex emits.

    Without it, selecting the colour re-fires currentIndexChanged, which sees
    the chooser entry again and opens a second dialog -- one click, two colour
    pickers, and the second one cancelled undoes the first.
    """
    from PySide6.QtGui import QColor
    from spacr.qt.widgets import colour_picker
    from spacr.qt.widgets.save_figure_dialog import _CHOOSE

    box = _colour_box(dialog)
    opened = []

    def once(*args, **kwargs):
        opened.append(1)
        return QColor("#abcdef")

    monkeypatch.setattr(colour_picker, "pick_colour", once)

    box.setCurrentIndex(box.findData(_CHOOSE))

    assert len(opened) == 1, f"the picker opened {len(opened)} times"
