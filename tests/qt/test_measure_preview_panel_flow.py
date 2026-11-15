"""The Measure crop preview: loading an array, seeding settings, the crop pass and export.

The panel reads a merged ``(H, W, C)`` array off the GUI thread, crops its
objects, files them under phenotype headings and can export the displayed
crops.  What is pinned here is everything it promises to survive -- an
unreadable ``.npy``, a stale worker result, an unreadable setting, a missing
mask slice, a cancelled export -- and the exact sentence it puts on the status
line instead of drawing nothing.  Panels are unthreaded so every job lands
before the call returns.  One general test per behaviour family, each walking
a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.widgets import measure_preview as MP  # noqa: E402

pytestmark = pytest.mark.qt

HINT = "Load a merged array first."
CATEGORIES = ["Unnucleated · Infected · Organelle n/a",
              "Nucleated · Uninfected · Organelle n/a"]


def _merged(tmp_path, name="plate1_A01_f1.npy"):
    """Cell, nucleus, pathogen and organelle planes over three image planes."""
    data = np.zeros((48, 48, 8), np.float32)
    data[..., :3] = 20
    cell = np.zeros((48, 48), np.int32)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    organelle = np.zeros_like(cell)
    cell[2:18, 2:18] = 1
    nucleus[5:10, 5:10] = 1
    organelle[11:14, 11:14] = 1
    cell[24:42, 24:42] = 2
    pathogen[28:33, 28:33] = 1
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    data[..., 7] = organelle
    path = tmp_path / name
    np.save(path, data)
    return str(path)


@pytest.fixture
def make_panel(qtbot):
    """Build a fresh unthreaded panel per case."""
    def make():
        widget = MP.MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(widget)
        return widget
    return make


def _headers_in_grid(panel):
    out = []
    for index in range(panel._grid.count()):
        widget = panel._grid.itemAt(index).widget()
        if widget is not None and widget.objectName() == "CropCategoryHeader":
            out.append(widget.text())
    return out


def test_load_merged_array_hands_back_errors_as_text(tmp_path, monkeypatch, subtests):
    """Broken files, non-3-D arrays and failed folder grouping never raise out of the loader."""
    with subtests.test(msg="not an array"):
        broken = tmp_path / "not_really.npy"
        broken.write_bytes(b"this is not a numpy file")
        payload = MP.load_merged_array(str(broken))
        assert payload["data"] is None
        assert payload["error"].startswith("Failed to load: ")

    with subtests.test(msg="not three-dimensional"):
        flat = tmp_path / "flat.npy"
        np.save(flat, np.zeros((4, 4), np.float32))
        payload = MP.load_merged_array(str(flat), enumerate_sets=False)
        assert payload["error"] == "Expected a merged (H,W,C) array; got shape (4, 4)"
        assert payload["data"] is None

    with subtests.test(msg="grouping fails, array still read"), monkeypatch.context() as m:
        def boom(directory, suffixes):
            raise OSError("the share went away mid-listing")

        m.setattr(MP, "enumerate_image_sets", boom)
        folder = tmp_path / "grouping"
        folder.mkdir()
        payload = MP.load_merged_array(_merged(folder))
        assert payload["error"] == ""
        assert payload["data"].shape == (48, 48, 8)
        assert (payload["sets"], payload["directory"], payload["channels"]) == (None, None, None)


def test_the_panel_installs_only_current_successful_loads(make_panel, tmp_path, monkeypatch, subtests):
    """Stale, failed and empty payloads leave the panel as it was and say why."""
    with subtests.test(msg="superseded result dropped"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        current = panel._data_path
        panel._on_array_loaded(panel._load_token - 1,
                               {"path": "/stale.npy", "data": np.zeros((2, 2, 2)), "error": ""})
        panel._on_array_loaded(panel._load_token, "not a payload")
        assert panel._data_path == current

    with subtests.test(msg="failed load reported"):
        panel = make_panel()
        panel._on_array_loaded(panel._load_token, {"path": "/x.npy", "data": None,
                                                   "error": "Failed to load: bad magic"})
        assert panel._status.text() == "Failed to load: bad magic"
        assert panel._data is None

    with subtests.test(msg="neither data nor error"):
        panel = make_panel()
        panel._on_array_loaded(panel._load_token, {"path": "/x.npy", "data": None, "error": ""})
        assert panel._data is None

    with subtests.test(msg="synchronous broken load"):
        panel = make_panel()
        broken = tmp_path / "broken.npy"
        broken.write_bytes(b"nope")
        assert panel.load_array(str(broken)) is False
        assert panel._status.text().startswith("Failed to load: ")
        assert panel._data is None

    with subtests.test(msg="selectors before and after a load"):
        panel = make_panel()
        panel._refresh_source_selectors()
        assert panel._fov_box.count() == 0
        assert [panel._channel_box.itemText(i)
                for i in range(panel._channel_box.count())] == ["All channels"]
        folder = tmp_path / "three_fields"
        folder.mkdir()
        paths = [_merged(folder, f"plate1_A01_f{i + 1}.npy") for i in range(3)]
        assert panel.load_array(paths[0]) is True
        assert panel._status.text() == "2 object(s) · 2 categories"
        assert panel._fov_box.count() == 3
        assert panel._channel_box.count() == 9

    with subtests.test(msg="unchanged sample cap redraws nothing"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        before = (panel._fov_box.count(), panel._status.text())
        panel._on_max_sets_changed(panel._max_sets_box.value())
        assert (panel._fov_box.count(), panel._status.text()) == before

    with subtests.test(msg="analyses that are on follow a new field and settings"), \
            monkeypatch.context() as m:
        m.setattr(preferences, "_get_show_alpha_features", lambda: True)
        panel = make_panel()
        redrawn = []
        m.setattr(panel, "_refresh_confluency", lambda: redrawn.append("confluency"))
        m.setattr(panel, "_refresh_wound", lambda: redrawn.append("wound"))
        for button in (panel._confluency_btn, panel._wound_btn):
            button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(False)
        path = tmp_path / "plate1_A01_1.npy"
        np.save(path, np.zeros((32, 32, 3), np.uint16))
        panel.load_array(str(path))
        assert redrawn[-2:] == ["confluency", "wound"]
        redrawn.clear()
        panel.apply_settings({})
        assert redrawn == ["confluency", "wound"]


class _MuteSpin(QWidget):
    """A spinbox-shaped control that publishes no ``valueChanged``."""

    def value(self):
        return 0


class _MuteToggle(QWidget):
    """A toggle-shaped control that publishes no ``toggled``."""

    def isChecked(self):
        return False


def test_apply_settings_and_propagation_keep_what_they_cannot_read(make_panel, subtests):
    """Unreadable values cost only themselves; propagation sends exactly the panel's settings."""
    with subtests.test(msg="mask dim that is not a number"):
        panel = make_panel()
        before = panel._mask_dims["cell"].value()
        panel.apply_settings({"cell_mask_dim": "the fourth one", "cell_min_size": 42})
        assert panel._mask_dims["cell"].value() == before
        assert panel._min_sizes["cell"].value() == 42

    with subtests.test(msg="unreadable percentiles still normalise"):
        panel = make_panel()
        panel._normalise.setChecked(False)
        panel.apply_settings({"normalize": ["low", "high"]})
        assert panel._normalise.isChecked() is True

    with subtests.test(msg="normalize None, False and a pair"):
        panel = make_panel()
        panel._normalise.setChecked(True)
        panel._lo_pct.setValue(2.0)
        panel._hi_pct.setValue(98.0)
        panel.apply_settings({"normalize": None, "experiment": "seeded"})
        assert (panel._normalise.isChecked(), panel._lo_pct.value(),
                panel._hi_pct.value()) == (True, 2.0, 98.0)
        assert panel._experiment.text() == "seeded"
        panel.apply_settings({"normalize": False})
        assert panel._normalise.isChecked() is False
        panel.apply_settings({"normalize": [5.0, 95.0]})
        assert (panel._normalise.isChecked(), panel._lo_pct.value(),
                panel._hi_pct.value()) == (True, 5.0, 95.0)

    with subtests.test(msg="propagation with and without a callback"):
        panel = make_panel()
        seen = []
        panel.set_propagate_callback(seen.append)
        panel.propagate_settings()
        assert seen == [panel.settings_for_propagation()]
        panel.set_propagate_callback(None)
        panel.propagate_settings()
        assert len(seen) == 1

    with subtests.test(msg="turning propagation on sends at once"):
        panel = make_panel()
        seen = []
        panel.set_propagate_callback(seen.append)
        panel._propagate_btn.setChecked(True)
        assert seen == [panel.settings_for_propagation()]
        assert seen[-1]["crop_mode"] == ["cell"]
        panel._on_setting_changed()
        assert len(seen) == 2

    with subtests.test(msg="a control with no signal does not stop the wiring"):
        panel = make_panel()
        seen = []
        panel.set_propagate_callback(seen.append)
        panel._max_area = _MuteSpin(panel)
        panel._plot = _MuteToggle(panel)
        panel._normalise.setChecked(False)
        panel._lo_pct.setEnabled(True)
        panel._connect_controls()
        assert panel._lo_pct.isEnabled() is False
        panel._propagate_btn.setChecked(True)
        before = len(seen)
        panel._max_crops.setValue(7)
        assert len(seen) > before
        assert (seen[-1]["plot"], seen[-1]["png_size"]) == (False, 224)


def test_the_preview_pass_crops_groups_and_reports(make_panel, tmp_path, monkeypatch, subtests):
    """Run/refresh says why nothing is drawn, files crops by phenotype and reports its state."""
    with subtests.test(msg="nothing loaded"):
        panel = make_panel()
        assert panel._preview_blocked_reason() == HINT
        assert panel.can_preview() is False
        panel.run_preview()
        assert panel._status.text() == HINT
        assert panel.preview_status() == HINT
        panel._status.setText("")
        panel.refresh()
        assert panel._status.text() == HINT

    with subtests.test(msg="loaded array unblocks and re-crops"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        assert panel._preview_blocked_reason() == ""
        assert panel.can_preview() is True
        landed = []
        panel.preview_ready.connect(landed.append)
        panel._crops = []
        panel.run_preview()
        assert len(landed) == 1
        assert [c["category"] for c in panel._crops] == CATEGORIES

    with subtests.test(msg="cancelling drops the pass in flight"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        panel.run_preview()
        before = list(panel._crops)
        token = panel._crop_token
        panel.cancel_preview()
        assert panel._crop_token != token
        panel._on_crops_ready(token, {"crops": [], "error": ""})
        assert panel._crops == before

    with subtests.test(msg="PNG channels outside the array"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        panel._png_dims.set_value({"r": 30, "g": 29, "b": 28})
        panel.refresh()
        assert panel._status.text() == "PNG channels do not exist in this array."

    with subtests.test(msg="no mask slice empties the grid"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        panel.refresh()
        assert len(panel._crops) == 2
        panel._object_box.setCurrentText("nucleus")
        panel._mask_dims["nucleus"].setValue(-1)
        panel.refresh()
        assert panel._crops == []
        assert "No nucleus mask slice is configured." in panel._status.text()

    for case_id, grouped, headers in [
            ("ungrouped grid", False, ["Cell  ·  1 kept  ·  1 excluded by filters"]),
            ("grouped grid", True, [
                "Nucleated · Uninfected · Organelle n/a  ·  1 kept",
                "Unnucleated · Infected · Organelle n/a  ·  0 kept  ·  1 excluded by filters"])]:
        with subtests.test(msg=case_id):
            panel = make_panel()
            panel.load_array(_merged(tmp_path))
            panel._group_cells.setChecked(grouped)
            panel._object_box.setCurrentText("cell")
            panel.refresh()
            assert _headers_in_grid(panel) == headers

    with subtests.test(msg="thumbnail recoloured, crop untouched"), monkeypatch.context() as m:
        from spacr.crops import apply_display_primaries

        panel = make_panel()
        m.setattr(type(panel), "display_primaries", lambda self: "cmy", raising=False)
        panel.load_array(_merged(tmp_path))
        assert panel._crop_display_primaries == "cmy"
        crop = np.zeros((12, 12, 3), np.uint8)
        crop[..., 0] = 200
        pixmap = panel._crop_pixmap(crop)
        expected = tuple(int(v) for v in apply_display_primaries(crop, "cmy")[0, 0])
        centre = pixmap.toImage().pixelColor(pixmap.width() // 2, pixmap.height() // 2)
        assert (pixmap.width(), pixmap.height()) == (132, 132)
        assert centre.getRgb()[:3] == expected != (200, 0, 0)
        assert crop[0, 0].tolist() == [200, 0, 0]

    with subtests.test(msg="reported parameters"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        panel.refresh()
        panel._on_thumb_clicked(0)
        values = panel.current_params()
        assert {k: values[k] for k in ("n_crops", "selected", "categories", "crop_mode",
                                       "fov", "display_channel")} == {
            "n_crops": 2, "selected": [0], "categories": CATEGORIES,
            "crop_mode": ["cell"], "fov": "plate1_A01_f1.npy", "display_channel": None}

    with subtests.test(msg="re-annotating reproduces the categories"):
        panel = make_panel()
        panel.load_array(_merged(tmp_path))
        panel.refresh()
        for entry in panel._crops:
            entry.pop("category", None)
        panel._annotate_cell_categories()
        assert [entry.get("category") for entry in panel._crops] == CATEGORIES


def _crop():
    return {"crop": np.zeros((4, 4, 3), np.uint8), "source_path": "/a.npy",
            "object_key": ("plate1_r1_c1_f1", "cell", 1), "label": 1,
            "area": 9, "bbox": [0, 0, 4, 4]}


def test_crop_export_refuses_empty_previews_and_cleans_up_when_cancelled(tmp_path, monkeypatch, subtests):
    """An empty preview is refused; a cancel before, during or between crops leaves no stage."""
    with subtests.test(msg="empty preview"):
        result = MP._write_preview_crop_export(tmp_path / "out", [], {}, "rgb", threading.Event())
        assert result == {"error": "Run a crop preview before exporting."}

    with subtests.test(msg="cancelled before writing"):
        root = tmp_path / "before"
        root.mkdir()
        cancelled = threading.Event()
        cancelled.set()
        result = MP._write_preview_crop_export(root / "out", [_crop()], {}, "rgb", cancelled)
        assert result == {"cancelled": True}
        assert list(root.iterdir()) == []

    real = MP._preview_crop_pixels
    for case_id, crops, cancel_on_call, calls_expected in [
            ("cancelled during writing", [_crop()], 1, 1),
            ("cancelled between crops", [_crop(), _crop()], 1, 1)]:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            root = tmp_path / case_id.replace(" ", "_")
            root.mkdir()
            cancelled = threading.Event()
            calls = []

            def cancelling(crop, primaries):
                calls.append(1)
                if len(calls) == cancel_on_call:
                    cancelled.set()
                return real(crop, primaries)

            m.setattr(MP, "_preview_crop_pixels", cancelling)
            result = MP._write_preview_crop_export(root / "out", crops, {}, "rgb", cancelled)
            assert result == {"cancelled": True}
            assert len(calls) == calls_expected
            assert list(root.iterdir()) == []
