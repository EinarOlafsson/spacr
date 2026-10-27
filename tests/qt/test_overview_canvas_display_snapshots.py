"""Canvas images retain their actual Overview display until explicitly updated."""
import hashlib
import json

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QGraphicsSceneContextMenuEvent, QMenu

from tests.qt.test_725_overview_export_round6 import source, view
from spacr.qt.widgets import picture_export
from spacr.qt.widgets import qc_field_browser as browser


def _pixels(composer, box):
    return picture_export._canvas_box_pixels(box, None, composer._reader())


def test_added_images_freeze_adjusted_pixels_and_context_menu_updates_only_its_target(view, qtbot, monkeypatch, tmp_path):
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths(view)}
    composer = view._open_custom_export()
    field = view._fields[0]
    image = browser._overview_canvas_image(field)
    view.normalise.setChecked(True)
    view.low.setValue(3)
    view.high.setValue(95)
    first = composer.add_images([image])[0]
    s1 = _pixels(composer, first)
    view.low.setValue(25)
    view.high.setValue(75)
    view._set_color_mode("green_magenta")
    view._channel_overrides[1] = {"ink": .25}
    view._channel_display_changed()
    second = composer.add_images([image])[0]
    s2 = _pixels(composer, second)
    assert not np.array_equal(s1, s2)
    assert np.array_equal(_pixels(composer, first), s1)
    assert composer._source_key(first) != composer._source_key(second)
    view._set_color_mode("subtractive")
    s3 = view._canvas_snapshot()(dict(first, overview_display=None), None)
    assert not np.array_equal(s2, s3)
    composer.select([second["id"]])
    menus = []
    monkeypatch.setattr(QMenu, "popup", lambda menu, *_: menus.append(menu) or menu.actions()[0].trigger())
    item = composer._items[first["id"]]
    event = QGraphicsSceneContextMenuEvent()
    event.setScreenPos(QPoint(1, 1))
    item.contextMenuEvent(event)
    assert menus[0].actions()[0].text() == browser.tr("Propagate settings")
    assert composer.selected_ids() == [first["id"]]
    assert np.array_equal(_pixels(composer, first), s3)
    assert np.array_equal(_pixels(composer, second), s2)
    view._set_color_mode("additive")
    assert np.array_equal(_pixels(composer, first), s3)
    path = tmp_path / "display-layout.json"
    composer.canvas.save(path)
    restored = picture_export._CanvasLayout.load(path)
    assert restored.pages[0][0]["overview_display"] == first["overview_display"]
    assert np.array_equal(_pixels(composer, restored.pages[0][0]), s3)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths(view)} == before


def source_paths(view):
    return sorted({browser.Path(path) for field in view._fields for _, path in field.files})


def test_live_setting_updates_all_pages_and_future_images_then_off_freezes_and_stale_preview_is_ignored(view, qtbot):
    composer = view._open_custom_export()
    image = browser._overview_canvas_image(view._fields[0])
    first = composer.add_images([image])[0]
    initial = _pixels(composer, first)
    composer.add_page()
    second = composer.add_images([image])[0]
    old_key = composer._source_key(first)
    composer.live_display.setChecked(True)
    view._set_color_mode("green_magenta")
    qtbot.waitUntil(lambda: first["overview_display"] == view._canvas_snapshot()._settings)
    assert first["overview_display"] == second["overview_display"]
    changed = _pixels(composer, first)
    assert not np.array_equal(changed, initial)
    third = composer.add_images([image])[0]
    assert np.array_equal(_pixels(composer, second), changed)
    assert np.array_equal(_pixels(composer, third), changed)
    composer._preview_ready(old_key, np.zeros((3, 3, 3), dtype=np.uint8))
    assert old_key not in composer._previews
    qtbot.waitUntil(lambda: composer._preview_jobs.active_jobs() == 0)
    QApplication.processEvents()
    item = composer._items[third["id"]]
    depth, transform = len(composer.canvas._undo), composer.view.transform()
    view._request_visuals()
    qtbot.waitUntil(lambda: not composer._display_timer.isActive())
    assert composer._items[third["id"]] is item
    assert len(composer.canvas._undo) == depth and composer.view.transform() == transform
    composer.live_display.setChecked(False)
    view.channels.setText("2")
    view._set_color_mode("subtractive")
    QApplication.processEvents()
    assert all(np.array_equal(_pixels(composer, box), changed) for box in (first, second, third))
    fourth = composer.add_images([image])[0]
    assert not np.array_equal(_pixels(composer, fourth), changed)


def test_snapshots_roundtrip_channel_keys_and_enhancement_chain(view, qtbot):
    from spacr.qt.detect_chain import Chain

    composer = view._open_custom_export()
    view._enhancement = Chain(gamma=.5)
    view._channel_overrides[1] = {"percentile_low": 20., "percentile_high": 80., "ink": .4}
    field = view._fields[0]
    view._image_overrides = {browser._overview_image_key(field): {"channels": {1: {"black": .1}}}}
    first = composer.add_images([browser._overview_canvas_image(field)])[0]
    snapshot = browser._overview_canvas_unpack(json.loads(json.dumps(first["overview_display"])))
    assert snapshot["display"]["enhancement"] == Chain(gamma=.5)
    assert snapshot["display"]["channel_percentiles"][1] == (20., 80.)
    assert browser._overview_image_key(field) in snapshot["display"]["image_overrides"]
    expected = _pixels(composer, first)
    view._enhancement = None
    view._image_overrides = {}
    view._channel_overrides = {}
    assert np.array_equal(_pixels(composer, first), expected)


def test_well_mean_snapshot_survives_browser_cancellation_and_reopens_its_own_reference(view, qtbot):
    import functools
    import threading

    event = threading.Event()
    fields = tuple(view._fields)
    channels = (1, 2)
    options = {"mode": "reference_average", "reference_kind": "reference_column",
               "reference": (fields[0].plate, 1), "percentiles": (0., 100.),
               "selection": {"time_index": 0, "z_mode": "plane", "z_index": 0,
                             "representatives": 3, "seed": 0, "generation": 4}}
    plan, context = browser._overview_prepare_normalization(fields, channels, options,
        functools.partial(browser._overview_raw_planes, edge=256), event)
    display = {"normalization": plan, "normalization_context": context, "color_mode": "green_magenta"}
    snapshot = browser._OverviewCanvasSnapshot(fields, channels, (0., 100.), display)
    field = next(field for field in fields if field.well == "A02")
    box = {"source": browser._overview_canvas_image(field)["source"], "label": field.caption}
    snapshot.capture(box)
    event.set()
    assert context._cancelled.is_set()
    assert not snapshot._context._cancelled.is_set()
    pixels = snapshot(box, None)
    assert pixels.shape == (45, 60, 3)
    reopened = json.loads(json.dumps(box))
    later = browser._OverviewCanvasSnapshot(fields, (2,), None, {"color_mode": "additive"})
    assert np.array_equal(later(reopened, None), pixels)
