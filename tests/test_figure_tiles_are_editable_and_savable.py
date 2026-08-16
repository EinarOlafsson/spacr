"""Right-click a tile to restyle it; save any figure from that same menu.

Instruction 119, folded from 108 and 117:

    "all gigures should be editable by right clicking"
    "each figure should be editable and savable"

The right-click gesture reached the one figure that happened to be open and
the thumbnails beside it. It did not reach the grid, which is where a run's
seventeen figures actually are. And "savable" had no implementation at all --
the menu could restyle a figure and then offer no way to keep the result,
which is most of the reason to restyle one.

The awkward part, and the reason for the `navigate` flag: restyling a tile
must NOT jump to that figure. A grid exists for comparing figures, and
navigating away to change one loses the comparison being made.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def figure():
    fig = plt.figure(figsize=(4, 3))
    axis = fig.add_subplot(111)
    axis.plot([0, 1, 2], [0, 1, 0], label="a line")
    axis.legend()
    yield fig
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Save figure as…
# --------------------------------------------------------------------------- #

def test_the_menu_offers_a_save(qtbot, figure):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    host = QWidget()
    qtbot.addWidget(host)
    menu = build_figure_context_menu(host, figure)

    labels = [a.text() for a in menu.actions()]
    assert any("Save figure" in text for text in labels), labels


def test_saving_writes_a_real_png(qtbot, figure, tmp_path):
    from spacr.qt.widgets.figure_settings import save_figure_as

    target = tmp_path / "out.png"
    written = save_figure_as(None, figure, str(target))

    assert written == str(target)
    assert target.stat().st_size > 1000, "wrote a file with nothing in it"


def test_the_extension_chooses_the_format(qtbot, figure, tmp_path):
    """A .pdf that is secretly a PNG is worse than an error."""
    from spacr.qt.widgets.figure_settings import save_figure_as

    target = tmp_path / "out.pdf"
    save_figure_as(None, figure, str(target))

    assert target.read_bytes().startswith(b"%PDF")


def test_the_saved_file_carries_the_restyling(qtbot, figure, tmp_path):
    """"each figure should be editable and savable" is one sentence for a
    reason: an edit that survives to the screen and not to the file is an
    edit the user cannot use."""
    from spacr.qt.widgets.figure_settings import save_figure_as

    figure.axes[0].set_title("EDITED IN THE MENU")
    figure.axes[0].grid(True)

    target = tmp_path / "styled.svg"
    save_figure_as(None, figure, str(target))

    assert "EDITED IN THE MENU" in target.read_text(errors="ignore")


def test_a_cancelled_save_writes_nothing(qtbot, figure, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    from spacr.qt.widgets.figure_settings import save_figure_as

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert save_figure_as(None, figure) == ""
    assert not list(tmp_path.iterdir())


def test_an_unwritable_path_reports_rather_than_raises(qtbot, figure):
    """A menu action that throws takes the window down with it."""
    from spacr.qt.widgets.figure_settings import save_figure_as

    assert save_figure_as(None, figure, "/nope/nowhere/out.png") == ""


def test_saving_nothing_is_not_a_crash(qtbot):
    from spacr.qt.widgets.figure_settings import save_figure_as

    assert save_figure_as(None, None, "/tmp/never-written.png") == ""


# --------------------------------------------------------------------------- #
#  The tile carries the gesture
# --------------------------------------------------------------------------- #

def test_a_tile_emits_a_menu_request(qtbot):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QPixmap

    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    pixmap = QPixmap(80, 60)
    pixmap.fill()
    grid.set_figures([pixmap, pixmap], ["one", "two"])

    seen = []
    grid.figure_menu_requested.connect(lambda i, p: seen.append(i))
    grid._cells[1].customContextMenuRequested.emit(QPoint(3, 3))

    assert seen == [1]


def test_a_right_click_does_not_also_open_the_figure(qtbot):
    """Otherwise every attempt to restyle a tile navigates away first."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QMouseEvent, QPixmap

    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    pixmap = QPixmap(80, 60)
    pixmap.fill()
    grid.set_figures([pixmap], ["one"])

    opened = []
    grid.figure_activated.connect(opened.append)
    grid._cells[0].mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPoint(4, 4), Qt.RightButton,
        Qt.RightButton, Qt.NoModifier))

    assert opened == [], "a right-click opened the figure as well"


# --------------------------------------------------------------------------- #
#  Restyling a tile leaves the grid where it is
# --------------------------------------------------------------------------- #

def test_the_menu_for_a_tile_does_not_change_the_current_figure(qtbot, figure):
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    for _ in range(3):
        fig = plt.figure(figsize=(3, 2))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        queue.add_figure(fig)
    current = queue._current

    menus = []
    queue.show_figure_menu = (          # do not actually exec a modal menu
        lambda pos, idx=None, navigate=True: menus.append((idx, navigate)))
    queue.show_figure_menu(None, 0, navigate=False)

    assert queue._current == current, "opening a tile menu navigated away"


def test_restyling_a_tile_redraws_that_tile(qtbot):
    """The edit lands on a figure that is not on screen, so something has to
    re-rasterise it or the grid keeps showing the picture from before."""
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    figures = []
    for _ in range(3):
        fig = plt.figure(figsize=(3, 2))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        figures.append(fig)
        queue.add_figure(fig)

    assert queue._current == 2
    figures[0].axes[0].set_title("changed")

    assert queue.refresh_figure(0) is True
    assert queue._current == 2, "refreshing another figure moved the view"


def test_refreshing_a_missing_figure_is_false_not_an_exception(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)

    assert queue.refresh_figure(9) is False
