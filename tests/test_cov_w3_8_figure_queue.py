"""Figure-queue plumbing: styling a figure, naming its page, and the guards.

The render helpers take a real matplotlib figure and write real files into
``tmp_path``, so "did it produce a page" is answered by the page.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402

from spacr.qt.widgets import figure_queue as fq  # noqa: E402

pytestmark = pytest.mark.qt


def _figure():
    figure = Figure(figsize=(4.0, 3.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [1.0, 3.0, 2.0], label="series")
    axes.set_title("a title")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend(title="legend title")
    axes.text(0.5, 0.5, "annotation")
    figure.suptitle("figure title")
    return figure


# ---------------------------------------------------------------------------
# Every text on a figure
# ---------------------------------------------------------------------------

def test_every_text_a_figure_carries_is_found():
    figure = _figure()
    texts = {item.get_text() for item in fq.figure_text_items(figure)}
    assert {"a title", "x", "y", "series", "legend title", "annotation",
            "figure title"} <= texts


def test_a_figure_with_no_axes_still_answers():
    assert fq.figure_text_items(Figure()) == []


# ---------------------------------------------------------------------------
# The per-figure text size
# ---------------------------------------------------------------------------

def test_a_per_figure_size_is_remembered_and_cleared():
    figure = Figure()
    assert fq.figure_text_size_override(figure) == 0
    fq.set_figure_text_size_override(figure, 14)
    assert fq.figure_text_size_override(figure) == 14
    fq.set_figure_text_size_override(figure, 0)
    assert fq.figure_text_size_override(figure) == 0
    fq.set_figure_text_size_override(figure, -5)
    assert fq.figure_text_size_override(figure) == 0


def test_a_size_that_is_not_a_number_is_no_override():
    figure = Figure()
    setattr(figure, fq.FIGURE_TEXT_SIZE_ATTR, "quite large")
    assert fq.figure_text_size_override(figure) == 0


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def test_the_ink_and_the_text_are_two_colours():
    figure = _figure()
    fq._style_figure_colors(figure, "#101010", "#e0e0e0", 13, "#ff8800")
    axes = figure.axes[0]
    assert axes.spines["left"].get_edgecolor()[:3] == pytest.approx(
        (1.0, 0.53333, 0.0), abs=0.01)
    assert axes.title.get_color() == "#e0e0e0"
    assert axes.title.get_fontsize() == 13
    assert figure.patch.get_facecolor()[:3] == pytest.approx(
        (0.0627, 0.0627, 0.0627), abs=0.01)


def test_an_empty_line_colour_reuses_the_text_colour():
    figure = _figure()
    fq._style_figure_colors(figure, "#ffffff", "#000000", 0, "   ")
    axes = figure.axes[0]
    assert axes.spines["left"].get_edgecolor()[:3] == (0.0, 0.0, 0.0)


def test_the_figures_own_size_is_used_when_none_is_given():
    figure = _figure()
    fq.set_figure_text_size_override(figure, 17)
    fq._style_figure_colors(figure, "#ffffff", "#000000")
    assert figure.axes[0].title.get_fontsize() == 17


def test_a_figure_that_cannot_be_restyled_is_left_alone():
    """A half-built figure must not take the render worker down."""
    class Awkward:
        patch = None

        def get_axes(self):
            return []

    fq._style_figure_colors(Awkward(), "#ffffff", "#000000", 12, "#000000")


# ---------------------------------------------------------------------------
# The vector page beside the raster
# ---------------------------------------------------------------------------

def test_the_page_name_cannot_collide_for_a_dotted_name(tmp_path):
    assert fq._sibling_pdf(tmp_path / "fig.png") == tmp_path / "fig.pdf"
    assert fq._sibling_pdf(tmp_path / "run_2.5") == tmp_path / "run_2.5.pdf"
    assert fq._sibling_pdf(tmp_path / "run_2.6") == tmp_path / "run_2.6.pdf"
    assert fq._sibling_pdf(tmp_path / "FIG.PNG") == tmp_path / "FIG.pdf"


def test_a_rendered_figure_leaves_a_raster_and_a_vector_page(tmp_path):
    png = tmp_path / "figure.png"
    assert fq.render_figure_to_png(_figure(), str(png)) is True
    assert png.exists() and png.stat().st_size > 0
    assert fq._sibling_pdf(png).exists()


def test_a_render_falls_back_to_defaults_when_a_preference_will_not_answer(
        tmp_path, monkeypatch):
    """One unreadable preference must not cost the figure its raster.

    The DPI getter alone is broken rather than the whole module: the render
    reads the preferences twice, and blocking the import would take out the
    transparency question further down as well, which is a different branch.
    """
    from spacr.qt import preferences

    def refuse():
        raise RuntimeError("preferences are mid-migration")

    monkeypatch.setattr(preferences, "get_figure_png_dpi", refuse)
    png = tmp_path / "figure.png"
    assert fq.render_figure_to_png(_figure(), str(png)) is True
    assert png.exists() and png.stat().st_size > 0


def test_a_page_that_is_not_a_pdf_renders_to_nothing(tmp_path):
    missing = tmp_path / "absent.pdf"
    assert fq.render_pdf_to_image(str(missing), timeout_ms=500) is None
    rubbish = tmp_path / "rubbish.pdf"
    rubbish.write_bytes(b"not a pdf at all\n")
    assert fq.render_pdf_to_image(str(rubbish), timeout_ms=500) is None


# ---------------------------------------------------------------------------
# The clear control
# ---------------------------------------------------------------------------

def test_clicking_clear_flashes_and_asks_for_the_clear(qtbot):
    label = fq._ClearFiguresLabel()
    qtbot.addWidget(label)
    label.resize(120, 24)
    asked = []
    label.clicked.connect(lambda: asked.append(True))

    inside = QPointF(label.rect().center())
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, inside, inside, Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    assert asked == [True]
    assert "color:" in label.styleSheet()


def test_dragging_off_the_control_cancels_the_click(qtbot):
    label = fq._ClearFiguresLabel()
    qtbot.addWidget(label)
    label.resize(120, 24)
    asked = []
    label.clicked.connect(lambda: asked.append(True))

    outside = QPointF(label.rect().right() + 40, label.rect().bottom() + 40)
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, outside, outside, Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))
    inside = QPointF(label.rect().center())
    label.mouseReleaseEvent(QMouseEvent(
        QEvent.MouseButtonRelease, inside, inside, Qt.RightButton,
        Qt.RightButton, Qt.NoModifier))
    assert asked == []


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space])
def test_the_keyboard_reaches_the_clear_control(qtbot, key):
    label = fq._ClearFiguresLabel()
    qtbot.addWidget(label)
    asked = []
    label.clicked.connect(lambda: asked.append(True))
    label.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    assert asked == [True]


def test_another_key_is_not_a_clear(qtbot):
    label = fq._ClearFiguresLabel()
    qtbot.addWidget(label)
    asked = []
    label.clicked.connect(lambda: asked.append(True))
    label.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    assert asked == []


def test_a_palette_that_will_not_load_still_paints_the_control(qtbot,
                                                               monkeypatch):
    label = fq._ClearFiguresLabel()
    qtbot.addWidget(label)

    def refuse():
        raise RuntimeError("no theme")

    monkeypatch.setattr(fq, "active_palette", refuse)
    label._restyle()
    assert "#f85149" in label.styleSheet()
    label._flash.trigger()
    label._restyle()
    assert "#4A9EFF" in label.styleSheet()
