"""Instruction 178 C.2: style a figure for the file, preview it, then write.

    "when right clicking and saving a figure (matplotlib or pyqt6graph) the
    user should be able to change all of theis for the saved graph, get a
    preview then save."

THE FIGURE ON SCREEN IS NOT TOUCHED. A saved figure is for paper and the one
on screen is for the screen (150); a save that restyled the live figure would
change what the user is reading as a side effect of writing a file, and they
would have to undo it to carry on.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from spacr.qt.widgets.save_figure_dialog import (FORMATS, INKS,
                                                 SaveFigureDialog,
                                                 copy_figure, style_for_file)


@pytest.fixture()
def figure(qtbot):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot([0, 1], [1, 0])
    axes.set_xlabel("x")
    axes.set_title("t")
    return fig


# ------------------------------------------------------------------ the copy


def test_the_preview_is_a_copy(figure, qtbot):
    copy = copy_figure(figure)

    assert copy is not None
    assert copy is not figure


def test_a_figure_that_will_not_pickle_comes_back_none(qtbot):
    """A figure holding a live canvas or a closure cannot be copied, and the
    caller offers the plain save rather than a broken preview."""
    fig = plt.figure()
    fig._not_picklable = lambda: None

    assert copy_figure(fig) is None


def test_copying_nothing_is_not_an_error():
    assert copy_figure(None) is None


# --------------------------------------------------------------- the styling


def test_the_ink_reaches_every_part_of_the_axes(figure, qtbot):
    styled = style_for_file(copy_figure(figure), ink="#231F20")
    axes = styled.axes[0]

    assert axes.xaxis.label.get_color() == "#231F20"
    assert axes.title.get_color() == "#231F20"
    assert all(s.get_edgecolor()[:3] == (0.13725490196078433,
                                         0.12156862745098039,
                                         0.12549019607843137)
               for s in axes.spines.values())


def test_no_background_means_transparent(figure, qtbot):
    styled = style_for_file(copy_figure(figure), background="")

    assert styled.patch.get_alpha() == 0.0


def test_the_size_and_resolution_are_the_file_s(figure, qtbot):
    styled = style_for_file(copy_figure(figure), width=8.0, height=5.0,
                            dpi=600)

    assert tuple(styled.get_size_inches()) == (8.0, 5.0)
    assert styled.get_dpi() == 600


# ---------------------------------------------------------------- the dialog


def test_the_screen_figure_is_never_touched(figure, qtbot):
    """THE WHOLE DESIGN."""
    before_ink = figure.axes[0].xaxis.label.get_color()
    before_size = tuple(figure.get_size_inches())

    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog.ink.setCurrentIndex(1)
    dialog.background.setCurrentIndex(1)
    dialog.width.setValue(12.0)

    assert figure.axes[0].xaxis.label.get_color() == before_ink
    assert tuple(figure.get_size_inches()) == before_size


def test_the_preview_shows_the_chosen_ink(figure, qtbot):
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog.ink.setCurrentIndex(1)          # black ink on white

    assert dialog.preview().axes[0].xaxis.label.get_color() == "#231F20"


def test_the_preview_is_rebuilt_from_the_original_each_time(figure, qtbot):
    """Styling a styled copy compounds, so going back to "as on screen"
    would not undo the first change."""
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog.ink.setCurrentIndex(1)
    dialog.ink.setCurrentIndex(0)          # back to "as on screen"

    colour = dialog.preview().axes[0].xaxis.label.get_color()
    assert colour != "#231F20"


def test_it_opens_on_as_on_screen(figure, qtbot):
    """A dialog that silently changed the default would restyle every save a
    user made out of habit."""
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)

    assert dialog.ink.currentData() == ""
    assert INKS[0][0] == ""


def test_saving_writes_the_preview(figure, qtbot, tmp_path):
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog.ink.setCurrentIndex(1)
    out = tmp_path / "figure.png"

    written = dialog.save(str(out))

    assert written == str(out)
    assert out.stat().st_size > 0


def test_every_format_is_offered():
    assert [value for value, _label in FORMATS] == ["png", "pdf", "svg",
                                                    "tiff"]


def test_saving_with_no_preview_falls_back_to_the_figure(figure, qtbot,
                                                        tmp_path):
    """A figure that would not copy is still savable -- as it appears on
    screen, which is what the plain save has always done.

    NOT DRIVEN THROUGH AN EMPTY PATH: `save("")` opens a modal
    QFileDialog and blocks forever under an offscreen platform, which is
    exactly what it did the first time this file was run.
    """
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    dialog._preview = None
    out = tmp_path / "fallback.png"

    assert dialog.save(str(out)) == str(out)
    assert out.stat().st_size > 0


def test_the_menu_offers_it(figure, qtbot):
    """A REAL PARENT, not None. `QAction(text, None)` has no owner, so Python
    collects it the moment the builder returns and the entry never reaches
    the menu -- which is what happened the first time this test ran, and it
    looks exactly like the action was never added. Every caller in the
    application passes a widget."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    owner = QWidget()
    qtbot.addWidget(owner)
    menu = build_figure_context_menu(owner, figure)
    labels = [a.text() for a in menu.actions()]

    assert "Save figure with a preview…" in labels
    assert "Save figure as…" in labels, (
        "the one-click save was replaced rather than joined")
