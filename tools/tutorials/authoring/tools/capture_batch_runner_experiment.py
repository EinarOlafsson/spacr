#!/usr/bin/env python3
"""Capture Batch Runner executing three real isolated conversion jobs at 4K."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
ORIGINALS = DATASET / "orig"
GENERATED = ROOT / "generated" / "batch_runner"
QUEUE_PATH = GENERATED / "real_conversion_queue.json"
LESSON = "37_batch"
SOURCE_FIELDS = (
    "W0127F0004T0001Z000",
    "W0315F0003T0001Z000",
    "W0025F0001T0001Z000",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_real_jobs() -> tuple[list[dict], list[Path]]:
    if GENERATED.exists():
        raise FileExistsError(
            f"tutorial output already exists; remove only {GENERATED} to rerun"
        )
    GENERATED.mkdir(parents=True)
    records: list[dict] = []
    settings_paths: list[Path] = []
    for index, prefix in enumerate(SOURCE_FIELDS, start=1):
        source = GENERATED / f"source_{index:02d}"
        destination = GENERATED / f"converted_{index:02d}"
        source.mkdir()
        files = []
        for channel in range(1, 5):
            original = ORIGINALS / f"{prefix}C{channel}.tif"
            prepared = source / original.name
            if not original.is_file():
                raise FileNotFoundError(original)
            shutil.copy2(original, prepared)
            if _sha256(original) != _sha256(prepared):
                raise RuntimeError(f"prepared source differs: {prepared}")
            files.append({
                "original": str(original),
                "prepared": str(prepared),
                "sha256": _sha256(original),
            })
        settings = {
            "src": str(source),
            "dst": str(destination),
            "layout": "auto",
            "z_handling": "keep",
            "plate_naming": "index",
            "overwrite": False,
            "map_name": "conversion_map.csv",
            "preview_only": False,
            "resume": False,
        }
        settings_path = GENERATED / f"convert_job_{index:02d}.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        settings_paths.append(settings_path)
        records.append({
            "job": index,
            "source_prefix": prefix,
            "source": str(source),
            "destination": str(destination),
            "settings": str(settings_path),
            "files": files,
        })
    return records, settings_paths


def _source_hashes(records: list[dict]) -> dict[str, str]:
    return {
        row["original"]: row["sha256"]
        for record in records for row in record["files"]
    }


def _set_job_editor(screen, module: str, settings: Path, label: str,
                    after: str = "", overrides: str = "") -> None:
    index = screen._module_combo.findData(module)
    if index < 0:
        raise RuntimeError(f"Batch Runner has no module {module!r}")
    screen._module_combo.setCurrentIndex(index)
    screen._settings_edit.setText(str(settings))
    screen._label_edit.setText(label)
    screen._depends_edit.setText(after)
    screen._overrides_edit.setText(overrides)


def _click_row(app, table, row: int):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    item = table.item(row, 0)
    if item is None:
        raise RuntimeError(f"Batch Runner has no row {row}")
    point = table.visualItemRect(item).center()
    QTest.mouseClick(table.viewport(), Qt.LeftButton, Qt.NoModifier, point)
    settle(app, 20)
    return table.viewport().mapTo(table.window(), point)


def main() -> int:
    source_records, settings_paths = _prepare_real_jobs()
    source_before = _source_hashes(source_records)

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ["PYTHONPATH"] = str(REPO) + os.pathsep + os.environ.get(
        "PYTHONPATH", ""
    )
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-batch-runner-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-batch-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog
    from spacr import batch as bt
    from spacr.qt.preferences import apply_preferences_to_app

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    nav = nav_button(window, "batch")
    if nav is None:
        raise RuntimeError("Batch Runner navigation button was not found")
    QTest.mouseClick(nav, Qt.LeftButton)
    settle(app, 45)
    screen = window._screens.get("batch")
    if screen is None:
        raise RuntimeError("Batch Runner screen was not created")
    screen._threaded = False
    if screen._runner is not None:
        raise RuntimeError("tutorial must exercise the real subprocess runner")

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav,
        "screen": screen,
        "editor": [screen._module_combo, screen._settings_edit,
                   screen._btn_pick, screen._label_edit,
                   screen._depends_edit, screen._overrides_edit,
                   screen._btn_add],
        "toolbar": [screen._btn_dup, screen._btn_remove,
                    screen._btn_up, screen._btn_down,
                    screen._btn_validate, screen._btn_load,
                    screen._btn_save],
        "run": [screen._on_error_combo, screen._threshold_spin,
                screen._btn_run, screen._btn_stop, screen._progress],
        "results": [screen._table, screen._problems_view, screen._log_view],
    })

    _set_job_editor(
        screen, "convert", settings_paths[0],
        "real field one — lossless conversion",
    )
    settle(app, 20)
    capture.save("02_add_first", {
        "editor": [screen._module_combo, screen._settings_edit,
                   screen._btn_pick, screen._label_edit,
                   screen._depends_edit, screen._overrides_edit,
                   screen._btn_add],
        "add": screen._btn_add,
    })
    QTest.mouseClick(screen._btn_add, Qt.LeftButton)
    settle(app, 30)
    if screen.queue().ids != ["convert-1"]:
        raise RuntimeError(f"first Batch job was not added: {screen.status_text()}")
    capture.save("03_first_ready", {
        "table": screen._table,
        "problems": screen._problems_view,
        "status": screen._status,
    })

    _set_job_editor(
        screen, "convert", settings_paths[1],
        "real field two — named plate",
        after="convert-1", overrides="plate_naming=name",
    )
    settle(app, 20)
    capture.save("04_add_dependent", {
        "editor": [screen._module_combo, screen._settings_edit,
                   screen._label_edit, screen._depends_edit,
                   screen._overrides_edit, screen._btn_add],
        "add": screen._btn_add,
        "table": screen._table,
    })
    QTest.mouseClick(screen._btn_add, Qt.LeftButton)
    settle(app, 30)
    if screen.queue().ids != ["convert-1", "convert-2"]:
        raise RuntimeError(f"dependent Batch job was not added: {screen.status_text()}")

    _set_job_editor(
        screen, "convert", settings_paths[2],
        "real field three — independent",
    )
    settle(app, 18)
    capture.save("05_add_independent", {
        "editor": [screen._module_combo, screen._settings_edit,
                   screen._label_edit, screen._depends_edit,
                   screen._overrides_edit, screen._btn_add],
        "add": screen._btn_add,
        "table": screen._table,
    })
    QTest.mouseClick(screen._btn_add, Qt.LeftButton)
    settle(app, 30)
    if screen.queue().ids != ["convert-1", "convert-2", "convert-3"]:
        raise RuntimeError(f"third Batch job was not added: {screen.status_text()}")
    capture.save("06_queue_ready", {
        "table": screen._table,
        "toolbar": [screen._btn_dup, screen._btn_remove,
                    screen._btn_up, screen._btn_down,
                    screen._btn_validate, screen._btn_load,
                    screen._btn_save],
        "problems": screen._problems_view,
        "status": screen._status,
    })

    row_point = _click_row(app, screen._table, 2)
    if screen.selected_job() is None or screen.selected_job().id != "convert-3":
        raise RuntimeError("could not select independent third job")
    capture.save("07_reorder", {
        "table": screen._table,
        "toolbar": [screen._btn_dup, screen._btn_remove,
                    screen._btn_up, screen._btn_down],
        "move_up": screen._btn_up,
    })
    capture.frames["07_reorder"]["selected_row"] = [
        int(row_point.x() - 12), int(row_point.y() - 12), 24, 24
    ]
    QTest.mouseClick(screen._btn_up, Qt.LeftButton)
    settle(app, 30)
    if screen.queue().ids != ["convert-1", "convert-3", "convert-2"]:
        raise RuntimeError(f"unexpected reordered queue: {screen.queue().ids}")

    capture.save("08_validate", {
        "table": screen._table,
        "toolbar": [screen._btn_dup, screen._btn_remove,
                    screen._btn_up, screen._btn_down,
                    screen._btn_validate, screen._btn_load,
                    screen._btn_save],
        "validate": screen._btn_validate,
        "problems": screen._problems_view,
    })
    QTest.mouseClick(screen._btn_validate, Qt.LeftButton)
    settle(app, 30)
    if screen.has_errors() or screen.problems_text():
        raise RuntimeError(f"valid Batch queue has problems: {screen.problems_text()}")
    capture.save("09_validated", {
        "table": screen._table,
        "run": [screen._on_error_combo, screen._threshold_spin,
                screen._btn_run, screen._btn_stop, screen._progress],
        "problems": screen._problems_view,
        "status": screen._status,
    })

    capture.save("10_save", {
        "table": screen._table,
        "toolbar": [screen._btn_validate, screen._btn_load, screen._btn_save],
        "save": screen._btn_save,
        "run": [screen._on_error_combo, screen._threshold_spin,
                screen._btn_run, screen._btn_stop],
    })
    original_dialog = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *args, **kwargs: (str(QUEUE_PATH), "Queue files (*.json)")
    )
    try:
        QTest.mouseClick(screen._btn_save, Qt.LeftButton)
    finally:
        QFileDialog.getSaveFileName = original_dialog
    settle(app, 25)
    if not QUEUE_PATH.is_file() or screen.queue_path() != str(QUEUE_PATH):
        raise RuntimeError("Batch queue was not saved")

    capture.save("11_run", {
        "table": screen._table,
        "run": [screen._on_error_combo, screen._threshold_spin,
                screen._btn_run, screen._btn_stop, screen._progress],
        "run_queue": screen._btn_run,
        "status": screen._status,
    })
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    settle(app, 70)
    result = screen.result()
    if result is None or not result.ok:
        raise RuntimeError(f"real Batch run failed: {screen.status_text()}")
    if [job.status for job in screen.queue()] != [bt.STATUS_SUCCESS] * 3:
        raise RuntimeError(
            f"unexpected Batch statuses: {[job.status for job in screen.queue()]}"
        )
    if not screen.log_text().strip():
        raise RuntimeError("selected Batch job has no real subprocess log")
    capture.save("12_result", {
        "table": screen._table,
        "progress": screen._progress,
        "summary": screen._problems_view,
        "log": screen._log_view,
        "status": screen._status,
    })
    capture.write_geometry()

    saved = bt.load_queue(str(QUEUE_PATH))
    if [job.status for job in saved] != [bt.STATUS_SUCCESS] * 3:
        raise RuntimeError("persisted Batch queue did not retain final statuses")
    outputs = []
    for record in source_records:
        destination = Path(record["destination"])
        tiffs = sorted(destination.glob("*.tif"))
        map_path = destination / "conversion_map.csv"
        if len(tiffs) != 4 or not map_path.is_file():
            raise RuntimeError(
                f"real conversion output incomplete: {destination} "
                f"({len(tiffs)} TIFFs)"
            )
        outputs.append({
            "destination": str(destination),
            "tiffs": [path.name for path in tiffs],
            "map": str(map_path),
        })

    source_after = {path: _sha256(Path(path)) for path in source_before}
    if source_before != source_after:
        raise RuntimeError("Batch Runner modified a transferred source image")
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "batch",
        "source_dataset": str(DATASET),
        "source_disclosure": (
            "Three real four-channel fields were copied byte-for-byte into "
            "isolated source folders. Batch Runner executed all three jobs "
            "through the real subprocess runner; no mocked statuses or "
            "synthetic pixels were used."
        ),
        "source_files": source_records,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "queue": str(QUEUE_PATH),
        "queue_order": saved.ids,
        "dependency": {"convert-2": ["convert-1"]},
        "on_error": "continue",
        "max_consecutive_failures": 3,
        "result": {
            "ok": result.ok,
            "statuses": {job.id: job.status for job in saved},
            "logs": {job.id: job.log_path for job in saved},
            "outputs": outputs,
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    window.close()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
