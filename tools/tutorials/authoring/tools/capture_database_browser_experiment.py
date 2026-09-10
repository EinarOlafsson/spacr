#!/usr/bin/env python3
"""Capture Database Browser against the real tutorial database at native 4K."""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
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
DATABASE = DATASET / "measurements" / "measurements.db"
GENERATED = ROOT / "generated" / "database_browser"
EXPORT = GENERATED / "cell_area_ge_10000.csv"
LESSON = "34_database"
FILTER_COLUMN = "cell_area"
FILTER_OPERATOR = ">="
FILTER_VALUE = "10000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _click_list_item(app, widget, text: str):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    matches = widget.findItems(text, Qt.MatchExactly)
    if len(matches) != 1:
        raise RuntimeError(f"expected one list item named {text!r}")
    item = matches[0]
    point = widget.visualItemRect(item).center()
    QTest.mouseClick(widget.viewport(), Qt.LeftButton, Qt.NoModifier, point)
    settle(app, 30)
    return widget.viewport().mapTo(widget.window(), point)


def main() -> int:
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)
    if GENERATED.exists():
        raise FileExistsError(
            f"tutorial output already exists; remove only {GENERATED} to rerun"
        )

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-database-browser-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-db-browser-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog
    from spacr.qt.preferences import apply_preferences_to_app

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None

    before_hash = _sha256(DATABASE)
    siblings_before = sorted(path.name for path in DATABASE.parent.iterdir())
    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    nav = nav_button(window, "db_browser")
    if nav is None:
        raise RuntimeError("Database Browser navigation button was not found")
    QTest.mouseClick(nav, Qt.LeftButton)
    settle(app, 45)
    screen = window._screens.get("db_browser")
    if screen is None:
        raise RuntimeError("Database Browser screen was not created")
    # Deterministic capture: execute the same bounded queries synchronously.
    screen._threaded = False

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav,
        "screen": screen,
        "source": [screen._path_edit, screen._btn_pick_db,
                   screen._btn_pick_src, screen._btn_open],
        "browser": [screen._table_list, screen._view],
    })

    screen._path_edit.setText(str(DATABASE))
    settle(app, 15)
    capture.save("02_open", {
        "source": [screen._path_edit, screen._btn_pick_db,
                   screen._btn_pick_src, screen._btn_open],
        "open": screen._btn_open,
    })
    QTest.mouseClick(screen._btn_open, Qt.LeftButton)
    settle(app, 50)
    if screen.database_path() != str(DATABASE):
        raise RuntimeError(f"wrong database opened: {screen.database_path()}")
    if len(screen.tables()) != 10:
        raise RuntimeError(f"expected 10 tables, got {len(screen.tables())}")
    capture.save("03_tables", {
        "database": [screen._path_edit, screen._edit_check,
                     screen._edit_note, screen._table_list],
        "tables": screen._table_list,
        "readonly": [screen._edit_check, screen._edit_note, screen._status],
    })

    cell_global = _click_list_item(app, screen._table_list, "cell")
    if screen.current_table() != "cell":
        raise RuntimeError(f"expected cell table, got {screen.current_table()}")
    if screen.row_count() != 81969 or len(screen.preview_columns()) != 218:
        raise RuntimeError(
            f"unexpected cell table: {screen.row_count()} rows, "
            f"{len(screen.preview_columns())} columns"
        )
    capture.save("04_cell_table", {
        "tables": screen._table_list,
        "preview": [screen._col_search, screen._col_count_label,
                    screen._view, screen._rows_label, screen._sort_note],
        "status": screen._status,
    })
    capture.frames["04_cell_table"]["cell_click"] = [
        int(cell_global.x() - 12), int(cell_global.y() - 12), 24, 24
    ]

    screen._col_search.setText("cell_channel_1_mean")
    settle(app, 25)
    if screen.visible_columns() != ["cell_channel_1_mean_intensity"]:
        raise RuntimeError(f"unexpected column search: {screen.visible_columns()}")
    capture.save("05_column_search", {
        "column_search": [screen._col_search, screen._col_count_label],
        "preview": screen._view,
    })

    screen._col_search.clear()
    screen._page_size_box.setValue(250)
    settle(app, 35)
    if screen.loaded_rows() != 250:
        raise RuntimeError(f"expected first 250 rows, got {screen.loaded_rows()}")
    capture.save("06_paging", {
        "paging": [screen._btn_more, screen._rows_label,
                   screen._sort_note, screen._page_size_box],
        "load_more": screen._btn_more,
        "preview": screen._view,
    })
    QTest.mouseClick(screen._btn_more, Qt.LeftButton)
    settle(app, 35)
    if screen.loaded_rows() != 500:
        raise RuntimeError(f"expected 500 loaded rows, got {screen.loaded_rows()}")
    capture.save("07_more_rows", {
        "paging": [screen._btn_more, screen._rows_label,
                   screen._sort_note, screen._page_size_box],
        "preview": screen._view,
    })

    filter_index = screen._filter_col.findText(FILTER_COLUMN)
    operator_index = screen._filter_op.findText(FILTER_OPERATOR)
    if filter_index < 0 or operator_index < 0:
        raise RuntimeError("structured area filter is unavailable")
    screen._filter_col.setCurrentIndex(filter_index)
    screen._filter_op.setCurrentIndex(operator_index)
    screen._filter_value.setText(FILTER_VALUE)
    settle(app, 20)
    capture.save("08_filter", {
        "filter": [screen._filter_col, screen._filter_op,
                   screen._filter_value, screen._raw_toggle,
                   screen._btn_apply, screen._btn_clear],
        "apply": screen._btn_apply,
    })
    QTest.mouseClick(screen._btn_apply, Qt.LeftButton)
    settle(app, 50)
    if screen.row_count() != 12907:
        raise RuntimeError(f"expected 12,907 filtered cells, got {screen.row_count()}")
    capture.save("09_filtered", {
        "preview": [screen._view, screen._rows_label, screen._sort_note],
        "filter": [screen._filter_col, screen._filter_op,
                   screen._filter_value, screen._btn_apply,
                   screen._btn_clear],
        "sql": screen._sql_label,
        "status": screen._status,
    })

    GENERATED.mkdir(parents=True)
    capture.save("10_export", {
        "filter_export": [screen._filter_col, screen._filter_op,
                          screen._filter_value, screen._btn_apply,
                          screen._btn_clear, screen._btn_export],
        "export": screen._btn_export,
        "status": screen._status,
    })
    original_dialog = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *args, **kwargs: (str(EXPORT), "CSV files (*.csv)"))
    try:
        QTest.mouseClick(screen._btn_export, Qt.LeftButton)
    finally:
        QFileDialog.getSaveFileName = original_dialog
    settle(app, 45)
    if not EXPORT.is_file():
        raise RuntimeError("filtered CSV export was not created")
    capture.save("11_exported", {
        "filter_export": [screen._filter_col, screen._filter_op,
                          screen._filter_value, screen._btn_export],
        "status": screen._status,
    })

    capture.save("12_clear", {
        "filter": [screen._filter_col, screen._filter_op,
                   screen._filter_value, screen._btn_apply,
                   screen._btn_clear],
        "clear": screen._btn_clear,
        "preview": screen._view,
    })
    QTest.mouseClick(screen._btn_clear, Qt.LeftButton)
    settle(app, 45)
    if screen.row_count() != 81969:
        raise RuntimeError(f"clear filter did not restore all rows: {screen.row_count()}")
    capture.save("13_restored", {
        "preview": [screen._view, screen._rows_label, screen._sort_note],
        "status": screen._status,
    })
    capture.write_geometry()

    with EXPORT.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        exported_rows = sum(1 for _ in reader)
    if exported_rows != 12907 or len(header) != 218:
        raise RuntimeError(
            f"expected 12,907 rows x 218 columns, got "
            f"{exported_rows} x {len(header)}"
        )
    after_hash = _sha256(DATABASE)
    siblings_after = sorted(path.name for path in DATABASE.parent.iterdir())
    if before_hash != after_hash:
        raise RuntimeError("Database Browser modified its source database")
    if siblings_before != siblings_after:
        raise RuntimeError("Database Browser left a journal or WAL beside the source")

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "db_browser",
        "source": str(DATABASE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "source_directory_before": siblings_before,
        "source_directory_after": siblings_after,
        "tables": screen.tables(),
        "table": "cell",
        "table_rows": 81969,
        "table_columns": 218,
        "page_size": 250,
        "loaded_after_more": 500,
        "column_search": {
            "query": "cell_channel_1_mean",
            "visible": ["cell_channel_1_mean_intensity"],
        },
        "filter": {
            "column": FILTER_COLUMN,
            "operator": FILTER_OPERATOR,
            "value": int(FILTER_VALUE),
            "rows": 12907,
        },
        "export": {
            "path": str(EXPORT),
            "rows": exported_rows,
            "columns": len(header),
        },
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    screen.close()
    settle(app, 20)
    window.close()
    settle(app, 20)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
