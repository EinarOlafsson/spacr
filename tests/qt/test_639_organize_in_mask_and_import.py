"""Item 639: the organizer popup on Mask Generation and Import Images.

Make Masks' "Organize for Measure" popup is reused in two more modes:
``images`` (intensity images only, written as the one folder Mask
Generation reads, and ``src`` set to it) and ``import`` (images and masks,
written for Mask Generation or for Measure). Each test organises a small
synthetic set and checks the module's own preflight
(:func:`spacr.validate.validate_settings`) passes on the result.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import validate
from spacr.qt.widgets import organize_for_measure as ofm

pytestmark = pytest.mark.qt


def _tif(path: Path, array=None, shape=(12, 12)) -> Path:
    """Write a TIFF, making its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.random.default_rng(len(str(path))).integers(
            100, 4000, shape).astype(np.uint16)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _mask(i: int) -> np.ndarray:
    """A 12x12 label mask with one object."""
    array = np.zeros((12, 12), np.uint16)
    array[2:6, 2:6] = i
    return array


def _messy_images(root: Path) -> Path:
    """Two stains of three fields, nested and named unlike Yokogawa."""
    exp = root / "messy"
    for stain in ("dapi", "gfp"):
        for i in range(1, 4):
            _tif(exp / "run_01" / stain.upper() / f"sample {i} {stain}.tif")
    return exp


def _errors(problems):
    """The error-severity problems of a preflight."""
    return [p for p in problems if p.severity == validate.ERROR]


def _accept(dialog, *, yes=True):
    """Press Apply, answering every question ``yes``."""
    asked = []
    dialog.ask = lambda title, _text: asked.append(title) or yes
    dialog._on_apply()
    return asked


@pytest.fixture
def mask_screen(qtbot, qt_theme_applied):
    """A Mask Generation screen with its folds and buttons installed."""
    from spacr.qt.screens import mask as mask_module
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    mask_module.install_folds(screen)
    return screen


def test_images_mode_hides_masks_and_writes_for_mask_generation(
        qtbot, qt_theme_applied):
    dialog = ofm._dialog_for("images")()
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Organize images"
    assert dialog.add_mask_button.isHidden()
    assert dialog.layout_box.isHidden()
    assert not dialog.split_button.isHidden()
    assert dialog._target_layout == "mask"
    with pytest.raises(ValueError):
        ofm._dialog_for("sideways")


def test_measure_mode_is_unchanged(qtbot, qt_theme_applied):
    dialog = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Organize for Measure"
    assert not dialog.add_mask_button.isHidden()
    row = dialog.action_row
    assert dialog.split_button not in [row.itemAt(i).widget()
                                       for i in range(row.count())]
    assert dialog._target_layout == "measure"


def test_mask_generation_button_organizes_and_preflight_passes(
        mask_screen, tmp_path, monkeypatch):
    from spacr.qt.screens import mask as mask_module

    button = mask_screen._organize_images_button
    assert button.objectName() == "MaskOrganizeImagesButton"
    assert mask_module._install_organize_button(mask_screen) is button

    exp = _messy_images(tmp_path)
    opened = []
    real = ofm._dialog_for

    def _fill(source, host, mode):
        dialog = real(mode)(source, host)
        opened.append(dialog)
        dialog.add_column("channel")
        dialog.add_column("channel")
        dialog.add_files(0, [str(exp / "run_01" / "DAPI")])
        dialog.add_files(1, [str(exp / "run_01" / "GFP")])
        _accept(dialog)
        return dialog

    monkeypatch.setattr(ofm, "_dialog_for",
                        lambda mode: lambda s, h: _fill(s, h, mode))
    monkeypatch.setattr(ofm, "_headless", lambda: True)
    dialog = mask_module._open_organize(mask_screen)
    assert opened and dialog.plan is not None and dialog.plan.ok
    assert dialog.mode == "images" and dialog._target_layout == "mask"

    result = ofm._apply_organized(dialog.plan, "mask", log=lambda _t: None)
    dest = Path(result.dest)
    assert sorted(p.name for p in dest.glob("*.tif")) == [
        f"plate1_A0{i}_T0001F001L01A01Z01C0{c}.tif"
        for i in (1, 2, 3) for c in (1, 2)]
    assert not (dest / "merged").exists()
    assert mask_module._on_organized(mask_screen, None, result, dialog.plan,
                                     "mask")
    settings = mask_screen._settings_model.collect()
    assert settings["src"] == str(dest)
    assert settings["metadata_type"] == "cellvoyager"
    settings.update(nucleus_channel=0, cell_channel=1)
    assert not _errors(validate.validate_settings(settings, "mask"))
    settings.update(cell_channel=2)
    assert _errors(validate.validate_settings(settings, "mask")), (
        "the preflight read two channels from the organised folder")
    assert not mask_module._on_organized(mask_screen, OSError("x"), None,
                                         dialog.plan, "mask")


def test_multichannel_files_split_into_channel_columns(qtbot, qt_theme_applied,
                                                        tmp_path):
    folder = tmp_path / "stacks"
    for i in range(1, 4):
        _tif(folder / "plate" / f"well{i}.tif",
             np.random.default_rng(i).integers(0, 999, (3, 12, 12)
                                               ).astype(np.uint16))
    dialog = ofm._dialog_for("images")()
    qtbot.addWidget(dialog)
    dialog.add_column("channel")
    dialog.add_files(0, [str(folder)])
    asked = _accept(dialog)
    assert asked == ["Split multi-channel files?"]
    assert dialog.plan is None
    assert len(dialog._channel_columns()) == 3
    assert len(dialog.rows) == 3 and not dialog._incomplete_rows()
    assert (folder / "plate" / "well1.tif").is_file()
    _accept(dialog)
    assert dialog.plan is not None and dialog.plan.ok, dialog.plan
    result = ofm._apply_organized(dialog.plan, "mask", log=lambda _t: None)
    settings = {"src": result.dest, "metadata_type": "cellvoyager",
                "nucleus_channel": 0, "cell_channel": 2}
    assert not _errors(validate.validate_settings(settings, "mask"))
    assert dialog._split_multichannel() == 0


def test_split_reads_last_axis_channels_and_refuses_planes(tmp_path):
    rgb = _tif(tmp_path / "rgb.tif",
               np.zeros((12, 12, 3), np.uint8))
    planes = ofm._split_channel_file(str(rgb), str(tmp_path / "out"),
                                     str(tmp_path))
    assert [os.path.basename(p) for p in planes] == [
        "rgb_ch1.tif", "rgb_ch2.tif", "rgb_ch3.tif"]
    assert tifffile.imread(planes[0]).shape == (12, 12)
    flat = _tif(tmp_path / "flat.tif")
    assert ofm._split_channel_file(str(flat), str(tmp_path / "out")) == []
    assert ofm._split_channel_file(str(tmp_path / "gone.tif"),
                                   str(tmp_path / "out")) == []


def _import_set(root: Path) -> Path:
    """Two channels in two folders, each with Make Masks' masks/."""
    exp = root / "imp"
    for stain in ("DAPI", "GFP"):
        for i in range(1, 4):
            _tif(exp / stain / f"field{i}.tif")
            _tif(exp / stain / "masks" / f"field{i}.tif", _mask(i))
    return exp


@pytest.fixture
def import_screen(qtbot, qt_theme_applied):
    """The Import Images screen."""
    from spacr.qt.screens.image_import import ImageImportScreen

    screen = ImageImportScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


@pytest.mark.parametrize("target", ["measure", "mask"])
def test_import_organizes_images_and_masks_into_either_layout(
        import_screen, tmp_path, monkeypatch, target):
    from spacr.qt import prefs

    button = import_screen._btn_organize
    assert button.objectName() == "ImportOrganizeButton"
    exp = _import_set(tmp_path)
    real = ofm._dialog_for

    def _fill(source, host, mode):
        dialog = real(mode)(source, host)
        assert not dialog.layout_box.isHidden()
        dialog.layout_box.setCurrentIndex(dialog.layout_box.findData(target))
        dialog.add_column("channel")
        dialog.add_column("channel")
        dialog.add_files(0, [str(exp / "DAPI")])
        dialog.add_files(1, [str(exp / "GFP")])
        _accept(dialog)
        return dialog

    monkeypatch.setattr(ofm, "_dialog_for",
                        lambda mode: lambda s, h: _fill(s, h, mode))
    monkeypatch.setattr(ofm, "_headless", lambda: True)
    pushed = []
    monkeypatch.setattr(prefs, "push_recent_source",
                        lambda key, path, limit=8: pushed.append((key, path)))
    dialog = import_screen._open_organize()
    assert dialog.mode == "import" and dialog._target_layout == target
    plan = dialog.plan
    assert plan.ok and sum(1 for r in plan.rows if r.source_mask) == 6

    result = ofm._apply_organized(plan, target, log=lambda _t: None)
    assert import_screen._on_organized(None, result, plan, target)
    dest = result.dest
    if target == "measure":
        assert pushed == [("measure", dest)]
        assert len(result.merged) == 3
        assert np.load(result.merged[0]).shape == (12, 12, 4)
        settings = {"src": dest, "channels": [0, 1],
                    "nucleus_mask_dim": 2, "cell_mask_dim": 3}
        assert not _errors(validate.validate_settings(settings, "measure"))
    else:
        assert pushed == [("mask", dest)]
        names = sorted(os.listdir(os.path.join(dest, "masks")))
        assert len(names) == 6 and names[0].startswith("plate1_A01_")
        settings = ofm._mask_generation_settings(plan)
        settings.update(nucleus_channel=0, cell_channel=1)
        assert not _errors(validate.validate_settings(settings, "mask"))
    assert not import_screen._on_organized(OSError("disk"), None, plan, target)
    assert import_screen.last_error.startswith("Organizing failed")


def test_the_worker_applies_and_reports(qtbot, qt_theme_applied, tmp_path):
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    exp = _import_set(tmp_path)
    dialog = ofm._dialog_for("import")()
    qtbot.addWidget(dialog)
    dialog.add_column("channel")
    dialog.add_files(0, [str(exp / "DAPI")])
    plan = dialog.prepare_plan()
    assert plan.ok, plan.problems
    seen = []
    worker = ofm._start_organized(
        host, plan, "mask", lambda *args: seen.append(args))
    assert host._organize_job is worker
    qtbot.waitUntil(lambda: bool(seen), timeout=10000)
    error, result, _plan, layout = seen[0]
    assert error is None and layout == "mask"
    assert len(os.listdir(os.path.join(result.dest, "masks"))) == 3
