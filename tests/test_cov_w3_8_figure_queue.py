"""Figure-queue plumbing: styling a figure, naming its page, and the guards.

The render helpers take a real matplotlib figure and write real files into
``tmp_path``, so "did it produce a page" is answered by the page.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
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


def test_the_figures_own_size_is_used_when_none_is_given():
    figure = _figure()
    fq.set_figure_text_size_override(figure, 17)
    fq._style_figure_colors(figure, "#ffffff", "#000000")
    assert figure.axes[0].title.get_fontsize() == 17


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
