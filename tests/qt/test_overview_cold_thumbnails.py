"""Cold Overview thumbnails: sparse page reads, per-device read choice and a persistent plane cache.

A first open of a large plate on a share or a spinning disk read every image
whole to draw a 40-pixel thumbnail. Uncompressed pages are now read row by
row at the thumbnail stride when that is faster on the device, and decoded
preview planes persist in the user's cache folder -- never in the source.
"""
import os

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


@pytest.fixture(autouse=True)
def _isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(browser, "_OVERVIEW_READ_TIMES", {})
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    yield
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_sparse_rows_equal_the_strided_whole_page(tmp_path, byteorder):
    pixels = np.arange(700 * 650, dtype=np.uint16).reshape(700, 650)
    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, pixels, byteorder=byteorder)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        stride = browser._overview_sparse_stride(page, 64)
        assert stride == 11
        plane = browser._overview_sparse_rows(str(path), page, stride, tif.byteorder)
    np.testing.assert_array_equal(plane, pixels[::stride, ::stride])
    assert plane.dtype == np.dtype(np.uint16)


def test_compressed_or_small_strides_read_the_whole_page(tmp_path):
    pixels = np.arange(300 * 300, dtype=np.uint16).reshape(300, 300)
    tifffile.imwrite(tmp_path / "zipped.tif", pixels, compression="zlib")
    tifffile.imwrite(tmp_path / "plain.tif", pixels)
    with tifffile.TiffFile(tmp_path / "zipped.tif") as tif:
        assert browser._overview_sparse_stride(tif.pages[0], 32) == 0
    with tifffile.TiffFile(tmp_path / "plain.tif") as tif:
        assert browser._overview_sparse_stride(tif.pages[0], 200) == 0


def test_each_device_keeps_the_faster_read_after_trying_both(tmp_path):
    path = str(tmp_path / "any.tif")
    open(path, "wb").close()
    seen = []
    for _ in range(2 * browser._OVERVIEW_READ_TRIALS):
        method = browser._overview_read_method(path)
        seen.append(method)
        browser._overview_record_read(path, method, 0.01 if method == "full" else 0.05)
    assert seen.count("sparse") == seen.count("full") == browser._OVERVIEW_READ_TRIALS
    assert browser._overview_read_method(path) == "full"


def _retire_drained(view, qtbot):
    """Retire ``view`` once its jobs have finished, so no worker outlives the test."""
    from spacr.qt.bridge import prune_parked_threads

    view._cancel()
    runners = [value for value in vars(view).values() if isinstance(value, browser.JobRunner)]
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)
    view._retire()
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)


def _plate(folder, channels=(1, 2)):
    folder.mkdir()
    for well in ("A01", "A02"):
        for channel in channels:
            tifffile.imwrite(folder / f"plate1_{well}_F1_C{channel}.tif",
                             np.arange(512 * 512, dtype=np.uint16).reshape(512, 512) * channel)
    return tuple(field for field in browser._overview_index(folder) if not field.unresolved)


def test_preview_planes_persist_in_the_user_cache_and_follow_file_changes(tmp_path, monkeypatch):
    source = tmp_path / "images"
    fields = _plate(source)
    before = sorted(os.listdir(source))
    field = fields[0]
    first = browser._overview_pixels(field, (1, 2), (1., 99.), edge=64)
    folder = browser._overview_disk_cache_folder()
    stored = list(folder.glob("*/*.npy"))
    assert len(stored) == 2 and str(folder).startswith(str(tmp_path / "cache"))
    assert sorted(os.listdir(source)) == before
    browser._OVERVIEW_PLANE_CACHE.clear()
    reads = []
    real = browser._overview_raw_planes

    def counted(*args, **kwargs):
        reads.append(1)
        yield from real(*args, **kwargs)

    monkeypatch.setattr(browser, "_overview_raw_planes", counted)
    np.testing.assert_array_equal(browser._overview_pixels(field, (1, 2), (1., 99.), edge=64), first)
    assert reads == []
    changed = source / os.path.basename(field.files[0][1])
    tifffile.imwrite(changed, np.zeros((512, 512), dtype=np.uint16))
    os.utime(changed, ns=(1, 1))
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    browser._overview_pixels(field, (1, 2), (1., 99.), edge=64)
    assert reads


def test_the_disk_cache_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_OVERVIEW_DISK_CACHE", "0")
    fields = _plate(tmp_path / "images")
    browser._overview_pixels(fields[0], (1, 2), (1., 99.), edge=64)
    assert browser._overview_disk_cache_folder() is None
    assert not (tmp_path / "cache").exists()


def test_a_late_representative_result_cannot_replace_a_newer_refusal(qtbot, tmp_path):
    fields = _plate(tmp_path / "images")
    view = browser._FieldOverview(settings={"source": str(tmp_path / "images"), "channels": (1,)})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire_drained(widget, qtbot))
    view._indexed(fields)
    older = view._comparison_request()
    newer = view._comparison_request()
    assert newer == older + 1 == view._comparison_requests
    view.status.setText("No representative image can be chosen this way for these annotation values.")
    view._compare_fields(fields, "Representative images: late")
    view._comparison_requests += 1
    qtbot.waitUntil(lambda: not view._montage_jobs.is_busy(), timeout=15000)
    assert view.status.text() == "No representative image can be chosen this way for these annotation values."


def test_display_preparation_starts_after_the_rebuilt_plate_paints(qtbot, tmp_path, monkeypatch):
    fields = _plate(tmp_path / "images")
    view = browser._FieldOverview(settings={"source": str(tmp_path / "images"), "channels": (1,)})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire_drained(widget, qtbot))
    view.show()
    qtbot.waitExposed(view)
    started = []
    real = browser._FieldOverview._prepare_display

    def recorded(self):
        started.append(self._plate_painted)
        return real(self)

    monkeypatch.setattr(browser._FieldOverview, "_prepare_display", recorded)
    view._indexed(fields)
    assert started == [] and not view._normalization_ready
    qtbot.waitUntil(lambda: bool(started), timeout=5000)
    assert started[0] is True
    qtbot.waitUntil(lambda: view._normalization_ready, timeout=15000)


def test_a_plate_that_never_paints_still_prepares_its_display(qtbot, tmp_path, monkeypatch):
    fields = _plate(tmp_path / "images")
    view = browser._FieldOverview(settings={"source": str(tmp_path / "images"), "channels": (1,)})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire_drained(widget, qtbot))
    view.show()
    qtbot.waitExposed(view)
    monkeypatch.setattr(browser._FieldOverview, "_plate_painted",
                        property(lambda self: False, lambda self, value: None), raising=False)
    view._indexed(fields)
    assert not view._normalization_ready
    qtbot.waitUntil(lambda: view._normalization_ready, timeout=15000)


def test_decoded_thumbnails_install_within_a_per_turn_budget(qtbot, tmp_path, monkeypatch):
    fields = _plate(tmp_path / "images")
    view = browser._FieldOverview(settings={"source": str(tmp_path / "images"), "channels": (1,)})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire_drained(widget, qtbot))
    view.show()
    qtbot.waitExposed(view)
    view._indexed(fields)
    qtbot.waitUntil(lambda: len(view._installed_thumbnails) == 2
                    and not view._thumbnail_jobs.active_jobs(), timeout=20000)
    entries = []
    for well in ("A01", "A02"):
        item = view.wells.item(*view._well_positions[well])
        entries.append((well, item.data(browser.Qt.UserRole + 1), view._well_representatives[well],
                        np.zeros((8, 8, 3), dtype=np.uint8), "", view._plate_context(),
                        browser.device_ratio(view.wells.viewport())))
        item.setData(browser.Qt.UserRole + 1, None)
    view._installed_thumbnails.clear()
    scans = []
    real = browser._FieldOverview._submit_plate_thumbnails
    monkeypatch.setattr(browser._FieldOverview, "_submit_plate_thumbnails",
                        lambda self, *args: scans.append(args) or real(self, *args))
    monkeypatch.setattr(browser, "_OVERVIEW_INSTALL_BUDGET_S", 0.0)
    view._thumbnail_installs.extend(entries)
    view._update_visuals()
    assert scans == [] and len(view._thumbnail_installs) == 2
    view._install_pending_thumbnails()
    assert len(view._thumbnail_installs) == 2 and view._install_timer.isActive()
    monkeypatch.setattr(browser, "_OVERVIEW_INSTALL_BUDGET_S", 1.0)
    qtbot.waitUntil(lambda: not view._thumbnail_installs, timeout=5000)
    assert view._installed_thumbnails == {"A01", "A02"} and scans == []
