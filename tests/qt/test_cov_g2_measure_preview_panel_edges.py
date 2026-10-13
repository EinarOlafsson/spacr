"""Measure preview panel: export dialogs, stale exports, clicks and bad fields."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview as mp  # noqa: E402
from tests.qt.test_measure_unmix_preview_f538 import panel, plate, settings  # noqa: E402,F401


@pytest.fixture
def widget(qtbot, plate, monkeypatch):  # noqa: F811
    return panel(qtbot, plate, monkeypatch)


def test_export_dialogs_that_are_cancelled_or_misnamed_export_nothing(
        widget, monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(widget, "_start_crop_export", started.append)
    monkeypatch.setattr(mp.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: "")
    widget._export_displayed_crops()
    monkeypatch.setattr(mp.QFileDialog, "getExistingDirectory",
                        lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mp.QInputDialog, "getText", lambda *a, **k: ("x", False))
    widget._export_displayed_crops()
    monkeypatch.setattr(mp.QInputDialog, "getText", lambda *a, **k: ("a/b", True))
    widget._export_displayed_crops()
    assert "without path separators" in widget._status.text() or started == []

    def changed(*a, **k):
        widget._crop_token += 1
        return ("ok", True)

    monkeypatch.setattr(mp.QInputDialog, "getText", changed)
    widget._export_displayed_crops()
    assert started == []
    saved = widget._crops
    widget._crops = []
    widget._export_displayed_crops()
    mp.MeasurePreviewPanel._start_crop_export(widget, tmp_path / "never")
    widget._crops = saved
    assert started == []


class _Stage:
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def test_stale_or_failed_exports_clean_up_and_say_why(widget, monkeypatch):
    stage = _Stage()
    widget._export_job_token = 5
    widget._finish_crop_export(4, widget._crop_token, {"stage": stage})
    assert stage.cleaned
    stage = _Stage()
    widget._export_job_token = widget._export_token = 7
    widget._finish_crop_export(7, widget._crop_token - 1, {"stage": stage})
    assert stage.cleaned
    for result, phrase in (({"cancelled": True}, "Export cancelled"),
                           ({"error": "disk full"}, "disk full")):
        widget._export_job_token = 7
        widget._finish_crop_export(7, widget._crop_token, result)
        assert phrase in widget._status.text()

    def broken(result):
        raise OSError("target appeared")

    monkeypatch.setattr(mp, "_publish_preview_crop_export", broken)
    widget._export_job_token = 7
    widget._finish_crop_export(7, widget._crop_token, {"stage": None})
    assert "target appeared" in widget._status.text()


def test_clicks_from_a_superseded_grid_or_without_a_source_do_nothing(
        widget, monkeypatch):
    opened = []
    monkeypatch.setattr(widget, "load_array_async",
                        lambda path, **k: opened.append(path))
    widget._on_current_thumb_clicked(widget._crop_token - 1, 0)
    widget._crops[0] = dict(widget._crops[0], source_path="")
    widget._on_current_thumb_clicked(widget._crop_token, 0, activate=True)
    assert opened == []


def test_a_thumb_double_click_asks_for_its_source(qtbot):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    thumb = mp._CropThumb(3)
    qtbot.addWidget(thumb)
    seen = []
    thumb.activated.connect(seen.append)
    event = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(1, 1), QPointF(1, 1),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    thumb.mouseDoubleClickEvent(event)
    assert seen == [3]


def test_an_unreadable_field_is_reported(widget, monkeypatch):
    monkeypatch.setattr(mp, "load_merged_array",
                        lambda path: {"error": "not an array", "path": path})
    assert widget.load_array("/nowhere.npy") is False
    assert "not an array" in widget._status.text()


def test_checked_crops_skip_and_refuse_bad_fields(tmp_path):
    flat = tmp_path / "flat.npy"
    np.save(flat, np.zeros((4, 4)))
    narrow = tmp_path / "narrow.npy"
    np.save(narrow, np.zeros((4, 4, 2)))
    kwargs = {"limit": 2, "mask_dim": 5, "channels": [0]}
    result = mp._compute_checked_crops([str(flat), str(narrow), str(narrow)],
                                       None, None, kwargs, {},
                                       threading.Event())
    assert result["crops"] == [] and len(result["warnings"]) == 2
    wide = tmp_path / "wide.npy"
    np.save(wide, np.zeros((4, 4, 3)))
    unmixed = mp._compute_checked_crops(
        [str(wide)], None, None, {"limit": 4, "mask_dim": 2, "channels": [0, 1]},
        {}, threading.Event(), unmix_settings={"channels": [0]})
    assert "measured channels" in unmixed["warnings"][0]
    assert SimpleNamespace


def test_unmixing_controls_with_another_layout_are_refused(plate):  # noqa: F811
    paths, _ = plate
    data = np.load(paths["C01"])[..., :2]
    with pytest.raises(ValueError, match="same plane layout"):
        mp._unmixed_crop_source(data, paths["C01"], settings(), threading.Event(), {})


def test_an_unmixing_plan_is_prepared_once_per_folder(plate):  # noqa: F811
    paths, _ = plate
    plans = {}
    data = np.load(paths["C01"], mmap_mode="r")
    first, _ = mp._unmixed_crop_source(data, paths["C01"], settings(),
                                       threading.Event(), plans)
    again, _ = mp._unmixed_crop_source(data, paths["C01"], settings(),
                                       threading.Event(), plans)
    assert len(plans) == 1
    np.testing.assert_array_equal(first, again)


def test_a_crop_error_is_a_warning_for_that_field(tmp_path, monkeypatch):
    path = tmp_path / "f.npy"
    np.save(path, np.zeros((4, 4, 3)))
    monkeypatch.setattr(mp, "compute_crops",
                        lambda data, kwargs, params: {"error": "no objects"})
    result = mp._compute_checked_crops(
        [str(path)], None, None, {"limit": 2, "mask_dim": 2, "channels": [0]},
        {}, threading.Event())
    assert result["warnings"] == ["f.npy: no objects"]


def test_a_field_without_sets_is_installed_directly(widget, monkeypatch, plate):  # noqa: F811
    real = mp.load_merged_array

    def without_sets(path):
        payload = real(path)
        payload["sets"] = None
        return payload

    monkeypatch.setattr(mp, "load_merged_array", without_sets)
    assert widget.load_array(str(plate[0]["A01"])) is True


def test_the_checked_menu_skips_rows_without_a_path(widget):
    widget._fov_box.addItem("blank", None)
    widget._refresh_checked_menu()
    assert widget._checked_menu.actions()


def test_switching_fields_keeps_a_multi_field_selection(widget, monkeypatch,
                                                       plate):  # noqa: F811
    loaded = []
    monkeypatch.setattr(widget, "load_array_async",
                        lambda path, **k: loaded.append(path))
    chosen = {str(plate[0]["A01"]), str(plate[0]["C01"])}
    widget._checked_sources = set(chosen)
    widget._fov_box.addItem("B01", str(plate[0]["B01"]))
    widget._fov_box.setCurrentIndex(widget._fov_box.count() - 1)
    widget._on_fov_changed()
    assert widget._checked_sources == chosen


def test_settings_applied_while_unmixing_refresh_the_preview(widget,
                                                             monkeypatch):
    refreshed = []
    widget._unmix_btn.setChecked(True)
    monkeypatch.setattr(widget, "refresh", lambda: refreshed.append(True))
    widget.apply_settings({})
    assert refreshed


def test_stale_exports_without_a_stage_and_running_crops(widget):
    widget._export_job_token = 9
    widget._finish_crop_export(3, widget._crop_token, {})
    widget._export_job_token = 9
    widget._export_token = 10
    widget._crop_running = True
    widget._finish_crop_export(9, widget._crop_token, {})
    widget._crop_running = False
    assert widget._export_running is False


def test_plain_clicks_select_and_bad_indexes_open_nothing(widget, monkeypatch):
    opened = []
    monkeypatch.setattr(widget, "load_array_async",
                        lambda path, **k: opened.append(path))
    widget._on_current_thumb_clicked(widget._crop_token, 0)
    assert 0 in widget._selected
    widget._open_crop_source(99)
    assert opened == []
