"""Round-5 Overview: per-channel normalization scope (image, well, plate, all loaded plates).

Item 725, maintainer feedback 2026-10-25 round 5: "there should be normalization
options for normalizing on different levels. like the image ..., on the well
level, plate level, and all." and "the user must be able to treat each channel
individually."
"""
import numpy as np
import pytest
import tifffile

from PySide6.QtCore import Qt

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def _plate(folder, gains):
    folder.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(32 * 40, dtype=np.float64).reshape(32, 40)
    for well, gain in gains.items():
        tifffile.imwrite(folder / f"p1-{well}-f1-c1.tif", (ramp * gain).astype(np.uint16))
        tifffile.imwrite(folder / f"p1-{well}-f1-c2.tif", (ramp[::-1] * gain + 7).astype(np.uint16))
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, tmp_path):
    first = _plate(tmp_path / "plateA", {"A01": 1, "A02": 4})
    second = _plate(tmp_path / "plateB", {"A01": 10})
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": f"{first}; {second}", "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 3), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    dialog.low.setValue(0.)
    dialog.high.setValue(100.)
    return dialog


def _field(view, plate, well):
    return next(field for field in view._fields if field.plate == plate and field.well == well)


def _limits(view, plate, well):
    return view._channel_limits.get(browser._overview_image_key(_field(view, plate, well)), {})


def _scope(view, qtbot, value):
    view._set_normalization_scopes(browser._overview_channel_scopes(value))
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)


def _pixels(view, plate, well, channels=(1,)):
    options = view._display_kwargs(view=True)
    options.pop("enhancement", None)
    return browser._overview_pixels(_field(view, plate, well), channels, (0., 100.), edge=None, **options)


def test_the_scope_setting_reads_and_writes_per_channel():
    assert browser._overview_channel_scopes("plate") == {"*": "plate"}
    assert browser._overview_channel_scopes("") == {"*": "image"}
    assert browser._overview_channel_scopes("1=plate, 2=image, 3=bogus") == {"1": "plate", "2": "image", "3": "image"}
    assert browser._overview_scope_text({"1": "plate", "2": "image"}) == "1=plate, 2=image"
    assert browser._overview_scope_text({"1": "well", "2": "well"}) == "well"
    assert browser._overview_scope_of({"1": "plate"}, 2) == "image"


def test_two_wells_share_one_mapping_under_plate_scope_and_not_under_image_scope(view, qtbot):
    _scope(view, qtbot, "image")
    assert view._channel_limits == {}
    dim, bright = _pixels(view, "plateA", "A01"), _pixels(view, "plateA", "A02")
    assert (dim == bright).all()
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01") == _limits(view, "plateA", "A02")
    low, high = _limits(view, "plateA", "A01")[1]
    assert low == 0. and high == pytest.approx(4 * (32 * 40 - 1), rel=1e-3)
    dim, bright = _pixels(view, "plateA", "A01"), _pixels(view, "plateA", "A02")
    assert bright[..., 0].max() == 255 and dim[..., 0].max() == pytest.approx(255 / 4, abs=1)
    _scope(view, qtbot, "well")
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01") != _limits(view, "plateA", "A02"), timeout=15000)


def test_all_scope_spans_plates(view, qtbot):
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateB", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01") != _limits(view, "plateB", "A01")
    _scope(view, qtbot, "all")
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01") == _limits(view, "plateB", "A01")
                    == _limits(view, "plateA", "A02"), timeout=15000)
    assert _limits(view, "plateB", "A01")[1][1] == pytest.approx(10 * (32 * 40 - 1), rel=1e-3)


def test_each_channel_keeps_its_own_scope(view, qtbot):
    image_scoped = _pixels(view, "plateA", "A01", (1, 2))
    _scope(view, qtbot, "1=plate, 2=image")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert set(_limits(view, "plateA", "A01")) == {1}
    mixed = _pixels(view, "plateA", "A01", (1, 2))
    assert (mixed[..., 1] == image_scoped[..., 1]).all()
    assert not (mixed[..., 0] == image_scoped[..., 0]).all()


def test_the_scope_cache_is_reused_and_invalidated(view, qtbot, monkeypatch):
    calls = []
    real = browser._overview_scope_statistics

    def counted(groups, *args, **kwargs):
        calls.append(len(groups))
        return real(groups, *args, **kwargs)

    monkeypatch.setattr(browser, "_overview_scope_statistics", counted)
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert calls == [4]
    view._refresh_scope_limits()
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    assert calls == [4]
    view.high.setValue(50.)
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=15000)
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A02").get(1, (0, 0))[1]
                    == pytest.approx(np.percentile(np.concatenate([np.arange(32 * 40), np.arange(32 * 40) * 4.]), 50.),
                                     rel=1e-2), timeout=15000)
    view._set_load_limits(1, None)
    qtbot.waitUntil(lambda: len(calls) == 3, timeout=15000)
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01").get(1, (0, 0))[1]
                    == pytest.approx(np.percentile(np.arange(32 * 40), 50.), rel=1e-2), timeout=15000)
    _scope(view, qtbot, "image")
    assert view._channel_limits == {}


def test_thumbnails_follow_the_scope(view, qtbot):
    view.plates.setCurrentIndex(view.plates.findData("plateA"))
    item = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")])
    qtbot.waitUntil(lambda: not item.icon().isNull(), timeout=20000)
    before = item.data(Qt.UserRole + 1)
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    qtbot.waitUntil(lambda: view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).data(Qt.UserRole + 1) not in (None, before)
                    and not view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).icon().isNull(), timeout=20000)
    after = view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]).data(Qt.UserRole + 1)
    assert after is not None and after != before
    assert _limits(view, "plateA", "A01") == _limits(view, "plateA", "A02")


def test_a_scoped_channel_with_its_own_percentiles_is_stretched_at_them(view, qtbot):
    _scope(view, qtbot, "plate")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    assert _limits(view, "plateA", "A01")[1][1] == pytest.approx(4 * (32 * 40 - 1), rel=1e-3)
    view._channel_overrides[1] = {"percentile_low": 0., "percentile_high": 50.}
    view._channel_display_changed()
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    ramp = np.arange(32 * 40, dtype=np.float64)
    expected = float(np.percentile(np.concatenate([ramp, ramp * 4]), 50.))
    qtbot.waitUntil(lambda: _limits(view, "plateA", "A01")[1][1] == pytest.approx(expected, rel=1e-3),
                    timeout=15000)
    assert _limits(view, "plateA", "A01")[1] == _limits(view, "plateA", "A02")[1]
    assert _limits(view, "plateA", "A01")[2][1] == pytest.approx(4 * (32 * 40 - 1) + 7, rel=1e-3)


def test_the_canvas_snapshot_draws_the_shown_display_without_the_enhancement(view, qtbot):
    _scope(view, qtbot, "1=plate, 2=image")
    qtbot.waitUntil(lambda: bool(_limits(view, "plateA", "A01")), timeout=15000)
    view._channel_overrides[2] = {"percentile_low": 10., "percentile_high": 60.}
    view._channel_display_changed()
    view._set_channel_targets(None, (2,))
    qtbot.waitUntil(lambda: not view._metadata_jobs.is_busy(), timeout=15000)
    snapshot = view._canvas_snapshot()
    shown = view._display_kwargs(view=True)
    for key in ("channel_limits", "percentile_channels", "channel_percentiles"):
        assert snapshot._display[key] == shown[key]
    assert "enhancement" not in snapshot._display and "channel_enhancement" not in snapshot._display
    field = _field(view, "plateA", "A02")
    drawn = snapshot(browser._overview_canvas_image(field), None)
    shown.pop("enhancement", None)
    expected = browser._overview_pixels(field, snapshot._channels, snapshot._limits, edge=None, **shown)
    assert (drawn == expected[..., :3]).all()
