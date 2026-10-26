"""Round-5 Overview, M5: the hover pixel readout at the bottom left.

Item 725, maintainer feedback 2026-10-25 round 5, point 4: "when the user
hoves a picture the values for each channel should be shown in a designated
place in the botom left."
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")
HEIGHT, WIDTH = 60, 80


def _raw(channel, index=0):
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    return (yy * WIDTH + xx + 10000 * channel + 7 * index).astype(np.uint16)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for index, well in enumerate(WELLS):
        for channel in (1, 2):
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif", _raw(channel, index))
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs, view._readout_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 2), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.fov_axis.setCurrentIndex(dialog.fov_axis.findData(None))
    return dialog


def _open_image(view, qtbot, well="A01"):
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), well)]))
    qtbot.waitUntil(lambda: view.fields.count() > 0, timeout=10000)
    view.fields.setCurrentRow(0)
    view._field(view.fields.item(0))
    qtbot.waitUntil(lambda: view.view._item is not None and view._individual
                    and view.view._item.pixmap().width() == WIDTH, timeout=10000)
    qtbot.waitUntil(lambda: not view._image_jobs.is_busy() and not view._image_pending, timeout=10000)
    QApplication.processEvents()


def _move(widget, position):
    viewport = widget.viewport()
    event = QMouseEvent(QEvent.MouseMove, position, viewport.mapToGlobal(position),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(viewport, event)


def _hover_pixel(image_view, x, y):
    _move(image_view, image_view.viewportTransform().map(QPointF(x + .5, y + .5)))


def _expected(x, y, index=0):
    return [f"Channel {channel}: {int(_raw(channel, index)[y, x])}" for channel in (1, 2)]


def _shows(view, x, y, index=0):
    text = view._pixel_readout.text()
    return text.startswith(f"x {x}, y {y}") and all(value in text for value in _expected(x, y, index))


def test_the_readout_shows_every_channels_raw_value_plate_well_and_image(view, qtbot):
    _open_image(view, qtbot)
    for x, y in ((0, 0), (5, 7), (79, 59), (33, 41)):
        _hover_pixel(view.view, x, y)
        qtbot.waitUntil(lambda: _shows(view, x, y), timeout=10000)
    text = view._pixel_readout.text()
    assert "Plate p1" in text and "Well A01" in text and "Image 1" in text


def test_leaving_the_image_clears_the_readout_without_moving_the_layout(view, qtbot):
    _open_image(view, qtbot)
    height = view._pixel_readout.height()
    top = view._pixel_readout.mapTo(view, view._pixel_readout.rect().topLeft())
    _hover_pixel(view.view, 10, 12)
    qtbot.waitUntil(lambda: _shows(view, 10, 12), timeout=10000)
    QApplication.sendEvent(view.view.viewport(), QEvent(QEvent.Leave))
    assert view._pixel_readout.text() == ""
    assert view._readout_pending is None and not view._readout_timer.isActive()
    QApplication.processEvents()
    assert view._pixel_readout.height() == height
    assert view._pixel_readout.mapTo(view, view._pixel_readout.rect().topLeft()) == top


def test_the_zoomed_view_maps_the_cursor_to_the_exact_source_pixel(view, qtbot):
    _open_image(view, qtbot)
    view.view.resetTransform()
    view.view.scale(7.5, 7.5)
    view.view._user_zoomed = True
    view.view.centerOn(QPointF(52.3, 18.6))
    QApplication.processEvents()
    for x, y in ((52, 18), (50, 20), (55, 16)):
        _hover_pixel(view.view, x, y)
        qtbot.waitUntil(lambda: _shows(view, x, y), timeout=10000)
    corner = view.view.viewportTransform().map(QPointF(53.95, 17.05))
    _move(view.view, corner)
    qtbot.waitUntil(lambda: _shows(view, 53, 17), timeout=10000)
    assert view.view.transform().m11() == pytest.approx(7.5)
    text = view._pixel_readout.text()
    assert text.startswith("x 53, y 17") and all(value in text for value in _expected(53, 17))


def test_moves_are_throttled_to_one_readout_per_interval(view, qtbot, monkeypatch):
    _open_image(view, qtbot)
    calls = []
    monkeypatch.setattr(view, "_update_pixel_readout", lambda: calls.append(view._readout_pending[1]))
    view._readout_timer.timeout.disconnect()
    view._readout_timer.timeout.connect(view._update_pixel_readout)
    for x in range(20):
        _hover_pixel(view.view, x, 3)
    assert view._readout_timer.isActive() and calls == []
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=2000)
    last = view.view.viewportTransform().map(QPointF(19.5, 3.5))
    assert calls[0] == last


def test_a_stacked_image_and_a_well_tile_read_their_own_field(view, qtbot):
    view._open_image_views([field for field in view._fields if field.well == "B02"])
    panel = view._image_views[-1]
    qtbot.waitUntil(lambda: panel.view._item is not None, timeout=10000)
    panel.view.resize(400, 300)
    panel.view.fit()
    QApplication.processEvents()
    _hover_pixel(panel.view, 12, 9)
    qtbot.waitUntil(lambda: _shows(view, 12, 9, index=3), timeout=10000)
    assert "Well B02" in view._pixel_readout.text()
    _open_image(view, qtbot, "A02")
    tiles = view.fields
    item = tiles.item(0)
    qtbot.waitUntil(lambda: item.data(Qt.UserRole + 3) is not None and not item.icon().isNull(), timeout=10000)
    visible = tiles.visualItemRect(item).intersected(tiles.viewport().rect())
    assert visible.height() > 8, visible
    for point in (QPointF(visible.center().x() + .5, visible.top() + 3), QPointF(visible.center()) + QPointF(.5, .5)):
        view._clear_pixel_readout()
        _move(tiles, point)
        qtbot.waitUntil(lambda: "Well A02" in view._pixel_readout.text()
                        and "Channel 1:" in view._pixel_readout.text(), timeout=10000)
        target = view._readout_target(tiles, point)
        x, y = browser._overview_source_pixel(target[1], target[2], target[3], (HEIGHT, WIDTH))
        assert _shows(view, x, y, index=1)
        assert abs(x - WIDTH / 2) <= 2


def test_a_mosaic_maps_each_channel_tile_back_to_the_same_source_pixel(tmp_path):
    folder = tmp_path / "mosaic"
    folder.mkdir()
    for channel in range(4):
        tifffile.imwrite(folder / f"p1-A01-f1-c{channel}.tif", _raw(channel))
    fields = browser._overview_index(str(folder), PATTERN, source_scope="folder")
    rendered = browser._overview_pixels(fields[0], (0, 1, 2, 3), None, edge=400)
    tile = rendered.shape[1] // 2
    header = min(16, tile // 4)
    for index in range(4):
        left, top = (index % 2) * tile, (index // 2) * tile + header
        point = browser._overview_source_pixel(left + 10.5, top + 20.5, rendered.shape[:2],
                                               (HEIGHT, WIDTH), 4, True)
        assert point == (10, 20)
    assert browser._overview_source_pixel(5, 2, rendered.shape[:2], (HEIGHT, WIDTH), 4, True) is None
    assert browser._overview_source_pixel(10.5, 20.5, (30, 40), (HEIGHT, WIDTH)) == (21, 41)
    assert browser._overview_source_pixel(40, 5, (30, 40), (HEIGHT, WIDTH)) is None


def test_a_device_pixel_ratio_two_image_maps_logical_cursor_points_to_device_pixels(view, qtbot):
    from PySide6.QtGui import QImage, QPixmap

    _open_image(view, qtbot)
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image = QImage(pixels.data, WIDTH, HEIGHT, pixels.strides[0], QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(2.)
    view.view._item.setPixmap(pixmap)
    QApplication.processEvents()
    _move(view.view, view.view.viewportTransform().map(view.view._item.mapToScene(QPointF(15.25, 10.25))))
    qtbot.waitUntil(lambda: _shows(view, 30, 20), timeout=10000)
    text = view._pixel_readout.text()
    assert text.startswith("x 30, y 20") and all(value in text for value in _expected(30, 20))
    assert view.view._item.pixmap().devicePixelRatio() == 2.


def test_raw_values_print_exactly():
    assert browser._overview_raw_value(np.uint16(65535)) == "65535"
    assert browser._overview_raw_value(np.float32(0.125)) == "0.125"
    assert browser._overview_raw_value(np.float64(np.nan)) == "nan"
    assert browser._overview_raw_value(np.array([1, 2, 3], dtype=np.uint8)) == "(1, 2, 3)"
