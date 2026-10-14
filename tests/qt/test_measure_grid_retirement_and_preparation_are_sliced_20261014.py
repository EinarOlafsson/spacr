"""A redraw retires old native widgets before drawing the exact new grid."""
import weakref

import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QSpacerItem

from spacr.qt.widgets import measure_preview as MP
from tests.qt.test_measure_live_grid_off_the_gui_thread import _grid_pictures


def _panel(qtbot, threaded=True):
    panel = MP.MeasurePreviewPanel(threaded=threaded)
    qtbot.addWidget(panel)
    panel.resize(1200, 900)
    panel.show()
    return panel


def _crops():
    return [dict(crop=np.full((16, 19, 3), index * 20, np.uint8),
                 label=index + 1, area=80 + index,
                 included=index % 3 != 0,
                 category=("Uninfected" if index % 2 else "Infected"),
                 source_path=f"/synthetic/plate{index % 2}/field.npy")
            for index in range(8)]


def test_native_retirement_is_bounded_and_finishes_before_new_widgets(qtbot, monkeypatch):
    panel = _panel(qtbot)
    panel._group_cells.setChecked(False)
    retired = []
    for index in range(100):
        label = QLabel(str(index))
        label.destroyed.connect(lambda _obj=None, index=index: retired.append(index))
        panel._grid.addWidget(label, index // 8, index % 8)
    panel._grid.addItem(QSpacerItem(8, 8), 13, 0)
    panel._crops = _crops()
    scheduled = []
    monkeypatch.setattr(MP, "_after", lambda delay, owner, callback:
                        scheduled.append((delay, owner, callback)))
    placed = []
    real = panel._place_thumb

    def place(*args):
        placed.append(args[0])
        assert len(retired) == 100
        return real(*args)

    monkeypatch.setattr(panel, "_place_thumb", place)
    panel._render_grid()
    removed = 101 - panel._grid.count()
    assert 0 < removed <= 32
    assert not placed and panel._loads_in_flight
    previous = 0
    for _ in range(100):
        qtbot.wait(5)
        assert len(retired) - previous <= 32
        previous = len(retired)
        if not panel._render_plan:
            break
        delay, owner, callback = scheduled.pop(0)
        assert delay == 1 and owner is panel
        callback()
    assert not panel._render_plan
    assert placed == list(range(8))
    panel.shutdown()


def test_real_qt_continuations_destroy_old_widgets_before_insertion(qtbot, monkeypatch):
    panel = _panel(qtbot)
    panel._group_cells.setChecked(False)
    retired = []
    for index in range(100):
        label = QLabel(str(index))
        label.destroyed.connect(lambda _obj=None, index=index: retired.append(index))
        panel._grid.addWidget(label, index // 8, index % 8)
    panel._crops = _crops()
    placed = []
    real = panel._place_thumb

    def place(*args):
        assert len(retired) == 100
        placed.append(args[0])
        return real(*args)

    monkeypatch.setattr(panel, "_place_thumb", place)
    panel._render_grid()
    assert not placed and panel._render_plan
    qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    assert len(retired) == 100 and placed == list(range(8))
    assert len(_grid_pictures(panel)) == 8
    panel.shutdown()


def test_shutdown_releases_the_abandoned_render_crop_references(qtbot, monkeypatch):
    panel = _panel(qtbot)
    for index in range(100):
        panel._grid.addWidget(QLabel(str(index)), index, 0)
    panel._crops = _crops()
    crop_ref = weakref.ref(panel._crops[0]["crop"])
    scheduled = []
    monkeypatch.setattr(MP, "_after", lambda delay, owner, callback: scheduled.append(callback))
    panel._render_grid()
    assert panel._render_plan and crop_ref() is not None
    panel._crops = []
    panel.shutdown()
    assert not panel._render_plan and crop_ref() is None
    count = panel._grid.count()
    scheduled[0]()
    assert panel._grid.count() == count


@pytest.mark.parametrize("separate_sources", [False, True])
def test_grouped_headers_order_and_pixels_match_the_unchanged_inline_path(qtbot, separate_sources):
    inline = _panel(qtbot, False)
    threaded = _panel(qtbot, True)
    for panel in (inline, threaded):
        panel._group_cells.setChecked(True)
        panel._checked_sources = {"first", "second"} if separate_sources else {"first"}
        panel._crops = _crops()
        panel._render_grid()
        qtbot.waitUntil(lambda: not panel._render_plan, timeout=20000)
    left, right = _grid_pictures(inline), _grid_pictures(threaded)
    assert len(left) == len(right) == 8
    for a, b in zip(left, right):
        assert a[:2] == b[:2] and a[2] == b[2] and a[3] == b[3]
    headings = lambda p: [(p._grid.getItemPosition(i), p._grid.itemAt(i).widget().text())
                          for i in range(p._grid.count())
                          if p._grid.itemAt(i).widget().objectName() == "CropCategoryHeader"]
    assert headings(inline) == headings(threaded)
    inline.shutdown()
    threaded.shutdown()
