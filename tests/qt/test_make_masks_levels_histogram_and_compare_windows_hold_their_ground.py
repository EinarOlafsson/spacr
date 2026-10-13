"""Make Masks' small windows hold their ground at the edges of their input.

Pins the Levels editor, the threshold histogram, the Compare window and the
object filter list for the cases the everyday tests do not reach:

* a flat histogram puts every marker at the left edge; a right click, or a
  move or release with no marker held, moves no cutoff;
* the Levels editor ignores a cutoff typed before its histogram arrives,
  clamps one below the darkest or above the brightest pixel to 0 or 100 %,
  drops a histogram that lands after it closed, and a worker finishing after
  the window was destroyed does not raise;
* a threshold histogram opened with its snapshot in hand shows it at once,
  and closes cleanly with no run attached;
* an empty picture is an empty pixmap and a flat one is drawn, not divided
  by zero;
* the Compare window opens at its default size, and closes, when the
  preference store cannot be read or written;
* the filter list's Add does nothing with no property offered, and loading a
  saved list replaces the rows already there.
"""
from __future__ import annotations

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr.qt.screens import make_masks as mm


def _mouse(kind, x, button=Qt.LeftButton, buttons=Qt.LeftButton):
    pos = QPointF(float(x), 20.0)
    return QMouseEvent(kind, pos, pos, button, buttons, Qt.NoModifier)


# ---------------------------------------------------------------------------
# The plots
# ---------------------------------------------------------------------------

def test_a_flat_histogram_puts_every_marker_at_the_left_edge(qtbot):
    plot = mm._OtsuHistogramPlot(np.array([4.0]), np.array([7.0, 7.0]), [])
    qtbot.addWidget(plot)
    plot.resize(420, 220)
    assert plot.level_x(7.0) == 0.0
    assert plot.level_x(1000.0) == 0.0


def test_the_levels_plot_moves_a_cutoff_only_while_one_is_held(qtbot):
    plot = mm._LevelsPlot(np.ones(4), np.arange(5, dtype=float), [1.0, 3.0])
    qtbot.addWidget(plot)
    plot.resize(401, 220)
    moved = []
    plot.cutoff_changed.connect(lambda i, v: moved.append((i, v)))

    plot.mousePressEvent(_mouse(QEvent.MouseButtonPress, 100,
                                Qt.RightButton, Qt.RightButton))
    plot.mouseMoveEvent(_mouse(QEvent.MouseMove, 200))
    plot.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 200,
                                  Qt.LeftButton, Qt.NoButton))
    assert moved == []

    plot.mousePressEvent(_mouse(QEvent.MouseButtonPress, 110))
    plot.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, 120,
                                  Qt.LeftButton, Qt.NoButton))
    assert [i for i, _v in moved] == [0, 0]
    assert moved[-1][1] == pytest.approx(4.0 * 120 / (plot.width() - 1))


# ---------------------------------------------------------------------------
# The Levels editor
# ---------------------------------------------------------------------------

@pytest.fixture
def levels(qtbot):
    image = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    dialog = mm._LevelsDialog(image, (1.0, 99.0))
    qtbot.addWidget(dialog)
    yield dialog
    dialog._worker.close()


def test_levels_typed_before_the_histogram_arrive_change_nothing(levels):
    emitted = []
    levels.levels_changed.connect(lambda lo, hi: emitted.append((lo, hi)))
    levels.values = None
    drawn = list(levels.plot.levels)
    levels._choose(0, 50.0)
    levels.set_percentiles(5.0, 95.0)
    assert emitted == []
    assert levels.percentiles == (5.0, 95.0)
    assert levels.plot.levels == drawn


def test_a_cutoff_past_either_end_is_zero_or_a_hundred_percent(qtbot, levels):
    qtbot.waitUntil(lambda: levels.ready, timeout=10_000)
    emitted = []
    levels.levels_changed.connect(lambda lo, hi: emitted.append((lo, hi)))
    levels._choose(0, -5.0)
    assert emitted[-1] == (0.0, 99.0)
    levels._choose(1, 10_000.0)
    assert emitted[-1] == (0.0, 100.0)
    assert levels.white.value() == pytest.approx(100.0)


def test_a_histogram_landing_after_the_editor_closed_is_dropped(qtbot, levels):
    qtbot.waitUntil(lambda: levels.ready, timeout=10_000)
    levels.close()
    caption = levels.caption.text()
    levels._take(((np.ones(2), np.arange(3), np.arange(5.0)), None))
    assert levels.values is None
    assert levels.caption.text() == caption


def test_a_worker_finishing_after_the_editor_is_gone_does_not_raise(qtbot):
    dialog = mm._LevelsDialog(np.arange(4, dtype=np.uint16).reshape(2, 2),
                              (1.0, 99.0))
    worker = dialog._worker
    worker.close()
    deliver = dialog._deliver
    shiboken6.delete(dialog)
    assert not shiboken6.isValid(dialog)
    deliver(None, None, None)


# ---------------------------------------------------------------------------
# The threshold histogram
# ---------------------------------------------------------------------------

def test_a_histogram_opened_with_its_snapshot_shows_it_at_once(qtbot):
    dialog = mm._OtsuHistogramDialog(np.array([3.0, 5.0]),
                                     np.array([0.0, 50.0, 100.0]), [42.0],
                                     "global Otsu")
    qtbot.addWidget(dialog)
    assert dialog.ready is True
    assert dialog.progress.isHidden()
    assert "42" in dialog.caption.text()
    assert "global Otsu" in dialog.caption.text()
    assert dialog.plot.levels == [42.0]
    dialog.close()
    assert dialog.closed is True and dialog.request is None


# ---------------------------------------------------------------------------
# The Compare window
# ---------------------------------------------------------------------------

def test_an_empty_picture_is_an_empty_pixmap_and_a_flat_one_is_drawn():
    assert mm._grey_pixmap(np.zeros((0, 0)), 1.0, 99.0).isNull()
    flat = mm._grey_pixmap(np.full((6, 8), 12.0), 1.0, 99.0)
    assert (flat.width(), flat.height()) == (8, 6)
    assert flat.toImage().pixelColor(3, 3).red() == 0


def test_the_compare_window_survives_a_preference_store_it_cannot_use(
        qtbot, monkeypatch):
    from spacr.qt import preferences

    def broken(*args, **kwargs):
        raise OSError("preferences are read-only")

    monkeypatch.setattr(preferences, "get_section_layout", broken)
    monkeypatch.setattr(preferences, "set_section_layout", broken)
    assert mm._remembered_compare_size() == mm.COMPARE_DEFAULT_SIZE
    window = mm._ComparePreview(np.arange(16.0).reshape(4, 4),
                                np.arange(16.0).reshape(4, 4), "none")
    qtbot.addWidget(window)
    assert (window.width(), window.height()) == mm.COMPARE_DEFAULT_SIZE
    window.close()
    assert not window.isVisible()


# ---------------------------------------------------------------------------
# The filter list
# ---------------------------------------------------------------------------

def test_add_with_no_property_offered_adds_no_row(qtbot):
    filters = mm.ObjectFilterList()
    qtbot.addWidget(filters)
    filters.property_box.clear()
    filters._on_add()
    assert filters.filters() == []


def test_loading_a_saved_list_replaces_the_rows_already_there(qtbot):
    filters = mm.ObjectFilterList()
    qtbot.addWidget(filters)
    filters.add_filter("area", 5, 50)
    filters.add_filter("eccentricity", None, 0.9)
    assert [f["property"] for f in filters.filters()] == ["area",
                                                          "eccentricity"]
    filters.set_filters([{"property": "perimeter", "min": 3, "max": None}])
    assert [f["property"] for f in filters.filters()] == ["perimeter"]
    assert len(filters._rows) == 1
