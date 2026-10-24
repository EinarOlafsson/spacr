"""Edge cases of the field browser, overview and metadata wizard.

An unreadable path in the worker loader, a failed load report, a dialog torn
down twice, a native crop for a field without a recorded shape, a stale
comparison result, and a wizard whose action fit is already running or whose
buttons carry default and flat styling.
"""
import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets import qc_field_browser as module
from spacr.qt.widgets.qc_field_browser import (
    QCFieldBrowser, QCFieldTarget, QCFieldVerdict)

pytestmark = pytest.mark.qt

PATTERN = (r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)"
           r"-t(?P<timeID>\d+)-z(?P<sliceID>\d+)-c(?P<chanID>\d+)\.tif")


def _missing_target(tmp_path):
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    return QCFieldTarget(
        field="plate1_A01_9", plate_root=str(tmp_path / "plate1"),
        merged_dir=str(merged),
        verdicts=(QCFieldVerdict("cell", "fail", ("unreadable",), "gone"),))


def test_loader_primes_only_real_paths_and_reports_unstatable_ones_absent(
        monkeypatch, tmp_path):
    from pathlib import Path

    primed = []
    monkeypatch.setattr(module.path_probe, "prime",
                        lambda path, present: primed.append((path, present)))

    def refuse(self):
        raise OSError("stale mount")

    monkeypatch.setattr(Path, "is_file", refuse)
    target = _missing_target(tmp_path)
    image = module._load_and_record_state(target, "en", ("", "/stale/field.npy"))
    assert primed == [("/stale/field.npy", False)]
    assert image.error


def test_a_failed_load_is_reported_and_a_second_teardown_is_harmless(
        qtbot, tmp_path, monkeypatch):
    browser = QCFieldBrowser([_missing_target(tmp_path)], threaded=False)
    qtbot.addWidget(browser)
    browser.show()
    browser._on_load_failed("disk went away")
    assert "disk went away" in browser._load_status.text()
    browser._on_finished(0)

    class Refusing:
        refused = 0

        def disconnect(self, *_):
            Refusing.refused += 1
            raise TypeError("already disconnected")

    class Probes:
        answered = Refusing()

    monkeypatch.setattr(module.path_probe, "probes", Probes)
    browser._on_finished(0)
    browser.close()
    assert Refusing.refused >= 2
    assert not browser._run_timer.isActive()
    assert not browser._recheck_timer.isActive()


@pytest.fixture
def source_planes(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    for well in ("A01", "B02"):
        for field in (1, 2):
            path = source / f"p1-{well}-f{field}-t1-z1-c1.tif"
            pixels = np.arange(1200, dtype=np.uint16).reshape(30, 40) + field
            if well == "A01" and field == 1:
                pixels = np.arange(1536 * 2048, dtype=np.uint16).reshape(1536, 2048)
            tifffile.imwrite(path, pixels)
    return source


def _retire(dialog, qtbot):
    dialog._cancel()
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._thumbnail_jobs,
               dialog._field_thumbnail_jobs, dialog._montage_jobs, dialog._export_jobs,
               dialog._metadata_jobs, dialog._measurement_jobs, dialog._native_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners),
                    timeout=10000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0)
    dialog.close()


@pytest.fixture
def overview(qtbot, source_planes):
    dialog = module._FieldOverview(settings={
        "source": str(source_planes), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1,), "geometry": (2, 2), "representatives": 1})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: bool(dialog._fields) and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=10000)
    return dialog


def test_native_crop_is_not_requested_for_a_field_without_a_recorded_shape(
        overview, qtbot):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    dialog.fields.setCurrentRow(0)
    qtbot.waitUntil(lambda: dialog.view._item is not None
                    and not dialog._image_jobs.is_busy())
    dialog._native_timer.stop()
    qtbot.waitUntil(lambda: not dialog._native_jobs.is_busy())
    dialog._source_shapes.clear()
    submitted = dialog._native_jobs._next_id
    dialog._load_native_detail()
    assert dialog._native_jobs._next_id == submitted
    assert not dialog._native_jobs.is_busy()
    assert not dialog._native_pending


def test_a_comparison_that_lost_ownership_is_not_shown(overview, qtbot):
    dialog = overview
    dialog._well(dialog.wells.item(*dialog._well_positions["A01"]))
    qtbot.waitUntil(lambda: not dialog._image_jobs.is_busy()
                    and not dialog._thumbnail_jobs.is_busy()
                    and not dialog._field_thumbnail_jobs.is_busy()
                    and not dialog._montage_jobs.is_busy(), timeout=10000)
    qtbot.wait(100)
    dialog.fields.item(0).setSelected(True)
    dialog.fields.item(1).setSelected(True)
    shown = []
    original = dialog._show_pixels
    dialog._show_pixels = lambda pixels: (shown.append(pixels), original(pixels))
    dialog._compare()
    dialog._comparison_signature = ("another comparison",)
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy(), timeout=10000)
    qtbot.wait(100)
    assert shown == []
    assert dialog._comparison_signature == ("another comparison",)


def test_wizard_action_fit_does_not_reenter_and_measures_styled_buttons(
        source_planes, qtbot):
    wizard = module._OverviewMetadataWizard(str(source_planes), {"custom_regex": PATTERN})
    qtbot.addWidget(wizard)
    wizard.show()

    wizard._fitting_actions = True
    wizard._fit_actions()
    assert wizard._fitting_actions is True
    wizard._fitting_actions = False

    wizard._finished = True
    wizard._fit_actions()
    assert wizard._fitting_actions is False
    wizard._finished = False

    buttons = [QPushButton("Default"), QPushButton("Auto"), QPushButton("Flat")]
    buttons[0].setDefault(True)
    buttons[1].setAutoDefault(True)
    buttons[2].setFlat(True)
    for button in buttons:
        button.setParent(wizard)
        button.show()
    wizard._action_buttons = buttons
    wizard._fit_action_buttons()
    for button in buttons:
        hint = button.sizeHint().expandedTo(button.minimumSizeHint())
        assert button.minimumWidth() > 0 and button.minimumHeight() > 0
        assert button.minimumWidth() >= min(hint.width(), button.minimumWidth())
    wizard.close()
