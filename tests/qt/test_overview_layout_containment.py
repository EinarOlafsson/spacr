"""Opening and folding Overview levels preserves native widths and runtime containment."""
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtWidgets import QScrollArea

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")


@pytest.fixture
def overview(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for field in (1, 2):
            tifffile.imwrite(folder / f"p1-{well}-f{field}-c1.tif",
                             np.arange(1200, dtype=np.uint16).reshape(30, 40) + index + field)
    screen = AppScreen("overview")
    qtbot.addWidget(screen)
    screen.resize(1500, 1000)
    screen.show()
    assert screen._settings_model.set_value_for_key("overview_max_fields", "all")
    page = screen._overview_page
    wizard = page.run({"src": str(folder), "custom_regex": PATTERN,
                       "overview_channels": "1", "overview_normalize": False})
    wizard._startup_timer.stop()
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    qtbot.waitUntil(lambda: page.browser is not None and page.browser._normalization_ready,
                    timeout=15000)
    view = page.browser
    view.fov_axis.setCurrentIndex(view.fov_axis.findData(None))
    for name in ("Console", "Actions"):
        if screen._runtime_splitter.pane(name) is not None:
            screen._runtime_splitter.set_collapsed(name, False, by_user=True)
    yield screen, view
    view.close()
    runners = (view._index_jobs, view._image_jobs, view._thumbnail_jobs,
               view._field_thumbnail_jobs, view._montage_jobs, view._export_jobs,
               view._metadata_jobs, view._measurement_jobs, view._native_jobs,
               view._image_stack_jobs, view._group_jobs, view._readout_jobs, view._levels_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    page.retire()
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)
    screen.close()


def _open_wells(view, qtbot):
    for well in WELLS:
        row, column = view._well_positions[(view.plates.currentData(), well)]
        view._well(view.wells.item(row, column))
    assert len(view._well_panels) == len(WELLS)
    assert view.fields.count() == 2
    view.fields.setCurrentRow(0)
    view._field_in(view.fields, view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None, timeout=15000)


@pytest.mark.parametrize("collapsed", [False, True], ids=["image-expanded", "image-collapsed"])
def test_right_canvas_leaves_both_native_view_columns_adjustable(overview, qtbot, collapsed):
    screen, view = overview
    _open_wells(view, qtbot)
    view._open_custom_export()
    qtbot.waitUntil(lambda: view._export_dock.isVisible() and view._splitter.width() > 0)
    ready = []
    QTimer.singleShot(0, lambda: ready.append(True))
    qtbot.waitUntil(lambda: bool(ready))
    host = view._export_splitter
    host.moveSplitter(host.closestLegalPosition(host.getRange(1)[1], 1), 1)
    assert host.sizes()[1] > 0
    view._image_panel.set_collapsed(collapsed)
    assert view._image_panel.is_collapsed() is collapsed
    assert view._image_panel.maximumWidth() == 16777215
    left, right = (view._splitter.widget(index) for index in (0, 1))
    left_minimum = max(left.minimumWidth(), left.minimumSizeHint().width())
    right_minimum = max(right.minimumWidth(), right.minimumSizeHint().width())
    low = view._splitter.closestLegalPosition(left_minimum, 1)
    high = view._splitter.closestLegalPosition(
        view._splitter.width() - view._splitter.handleWidth() - right_minimum, 1)
    assert 0 < low < high < view._splitter.width(), (low, high, view._splitter.sizes())
    view._splitter.moveSplitter(low, 1)
    first = view._splitter.sizes()
    outer = host.sizes()
    view._splitter.moveSplitter(high, 1)
    second = view._splitter.sizes()
    assert first[0] < second[0] and first[1] > second[1], (first, second)
    assert all(size > 0 for size in second)
    assert view._export_splitter.sizes() == outer
    view._image_panel.set_collapsed(not collapsed)
    view._image_panel.set_collapsed(collapsed)
    qtbot.waitUntil(lambda: view._image_panel.is_collapsed() is collapsed)
    assert view._splitter.sizes() == second


@pytest.mark.parametrize("canvas", [False, True], ids=["ordinary-overview", "right-canvas"])
def test_repeated_level_cycles_keep_runtime_console_and_actions_inside_window(overview, qtbot, canvas, qt_theme_applied):
    screen, view = overview
    if canvas:
        view._open_custom_export()
    initial_size = screen.size()
    for cycle in range(3):
        _open_wells(view, qtbot)
        view._image_panel.set_collapsed(bool(cycle % 2))
        for panel in tuple(view._well_panels):
            panel.set_collapsed(True)
            panel.set_collapsed(False)
        qtbot.waitUntil(lambda: not view._module_action_fit_timer.isActive())
        console = screen._console_wrap
        actions = screen._actions_section
        assert console.isVisible() and actions.isVisible()
        assert screen.rect().contains(console.mapTo(screen, QPoint(0, 0)))
        assert screen.rect().contains(console.mapTo(screen, console.rect().bottomRight()))
        assert screen.rect().contains(actions.mapTo(screen, actions.rect().bottomRight()))
        assert screen.size() == initial_size
        viewport = next(parent for parent in screen._figures_card.body.findChildren(QScrollArea)
                        if parent.isAncestorOf(view)).viewport()
        assert viewport.mapTo(screen, viewport.rect().bottomRight()).y() < console.mapTo(screen, QPoint()).y()
        assert viewport.height() > 0
        for panel in tuple(view._image_views):
            view._close_image_view(panel)
        for panel in tuple(view._well_panels):
            view._close_well_panel(panel)
        view._image_panel.hide()
