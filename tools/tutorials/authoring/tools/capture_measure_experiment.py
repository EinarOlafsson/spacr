#!/usr/bin/env python3
"""Capture the Measure tutorial from spaCR and the completed test run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import (
    CaptureSession, card_named, center_dialog, nav_button, safe_stem, settle,
    wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DEFAULT_DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
OUTPUT = ROOT / "production" / "08_measure" / "keyframes"


def apply_preview_settings(panel, settings: dict) -> None:
    panel._experiment.setText(str(settings.get("experiment", "exp")))
    panel._measurement_channels.setText(
        ",".join(str(value) for value in settings.get("channels", [0, 1, 2, 3])))
    panel._object_box.setCurrentText("cell")
    for name in ("cell", "nucleus", "pathogen", "organelle"):
        value = settings.get(f"{name}_mask_dim")
        panel._mask_dims[name].setValue(-1 if value is None else int(value))
    panel._cytoplasm.setChecked(bool(settings.get("cytoplasm", True)))
    panel._plot.setChecked(bool(settings.get("plot", False)))
    panel._test_mode.setChecked(True)
    panel._timelapse.setChecked(bool(settings.get("timelapse", False)))
    panel._save_png.setChecked(bool(settings.get("save_png", True)))
    panel._save_arrays.setChecked(bool(settings.get("save_arrays", False)))
    modes = settings.get("crop_mode", ["cell"])
    for name, widget in panel._crop_mode_checks.items():
        widget.setChecked(name in modes)
    size = settings.get("png_size", [224, 224])
    panel._crop_width.setValue(int(size[0]))
    panel._crop_height.setValue(int(size[1]))
    panel._png_dims.setText(
        ",".join(str(value) for value in settings.get("png_dims", [0, 1, 2])))
    panel._use_bbox.setChecked(bool(settings.get("use_bounding_box", False)))
    normalize = settings.get("normalize", False)
    panel._normalise.setChecked(isinstance(normalize, list))
    if isinstance(normalize, list) and len(normalize) >= 2:
        panel._lo_pct.setValue(float(normalize[0]))
        panel._hi_pct.setValue(float(normalize[1]))
    panel._normalize_by.setCurrentText(str(settings.get("normalize_by", "png")))
    panel._dilate.setChecked(bool(settings.get("dialate_pngs", False)))
    ratios = settings.get("dialate_png_ratios", [0.2])
    panel._dilate_ratio.setValue(float(ratios[0] if ratios else 0.2))
    for name, widget in panel._min_sizes.items():
        widget.setValue(int(settings.get(f"{name}_min_size", 0) or 0))
    panel._uninfected.setChecked(bool(settings.get("uninfected", False)))
    panel._merge_edge_pathogen_cells.setChecked(
        bool(settings.get("merge_edge_pathogen_cells", True)))
    panel._max_crops.setValue(40)
    panel._group_cells.setChecked(True)
    panel._refresh_control_gates()


def open_settings(app, panel, tab: int):
    from PySide6.QtWidgets import QTabWidget

    panel.open_crop_settings()
    wait_until(app, lambda: panel._crop_settings_dialog is not None and
               panel._crop_settings_dialog.isVisible(), 10, "crop settings")
    dialog = panel._crop_settings_dialog
    center_dialog(panel.window(), dialog, 1800, 1320)
    tabs = dialog.findChild(QTabWidget)
    tabs.setCurrentIndex(tab)
    settle(app, 25)
    return dialog, tabs


def close_settings(app, dialog) -> None:
    dialog.close()
    wait_until(app, lambda: not dialog.isVisible(), 10, "crop settings close")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--array", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    settings_path = dataset / "settings" / "measure_crop_settings.csv"
    test_root = dataset / "test"
    db_path = test_root / "measurements" / "measurements.db"
    arrays = sorted((test_root / "merged").glob("*.npy"))
    if not settings_path.is_file() or not db_path.is_file() or not arrays:
        raise FileNotFoundError("completed Measure test outputs are required")
    array_path = args.array.resolve() if args.array else arrays[0]

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-measure-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-measure-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QTabWidget
    from spacr.utils import load_settings
    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.dnd_handlers import MeasureDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    settings = load_settings(
        str(settings_path), setting_key="Key", setting_value="Value")
    settings.update({
        "src": str(dataset / "merged"),
        "test_mode": True,
        "test_nr": 5,
        "n_jobs": 5,
    })

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError(
            f"capture device pixel ratio is {window.devicePixelRatioF()}, "
            "expected 1.0 for a 100% scale 4K tutorial"
        )
    window._on_nav_selected("measure")
    settle(app, 90)
    screen = window._screens.get("measure")
    if screen is None:
        raise RuntimeError("Measure screen did not open")
    screen.apply_settings_dict(settings)
    panel = screen._measure_preview
    apply_preview_settings(panel, settings)
    wait_until(
        app,
        lambda: getattr(screen, "_settings_search", None) is not None
        and getattr(screen, "_recipe_button", None) is not None,
        15.0,
        "Measure settings search and Recipes controls",
    )
    settle(app, 40)

    capture = CaptureSession(app, window, OUTPUT)
    capture.save("01_console", {
        "nav": nav_button(window, "measure"),
        "console": screen._console._console_box,
        "chat": screen._console._chat_row,
        "actions": screen._actions_row,
    })

    screen._ai_menu.popup(screen._ai_menu_btn.mapToGlobal(
        QPoint(0, screen._ai_menu_btn.height())))
    wait_until(app, screen._ai_menu.isVisible, 5.0, "AI provider menu")
    capture.save("02_ai_menu", {
        "chat": screen._console._chat_row,
        "ai_controls": [screen._ai_switch, screen._ai_menu_btn],
        "ai_menu_button": screen._ai_menu_btn,
        "ai_menu": screen._ai_menu,
    }, overlays=(screen._ai_menu,))
    screen._ai_menu.hide()
    settle(app, 12)

    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
    providers_dialog = _ProvidersDialog(screen)
    center_dialog(window, providers_dialog, 1500, 1200)
    providers_dialog.show()
    settle(app, 30)
    providers_tabs = providers_dialog.findChild(QTabWidget)
    if providers_tabs is None:
        raise RuntimeError("AI providers dialog tabs are unavailable")
    capture.save("03_ai_providers", {
        "dialog": providers_dialog,
        "tabs": providers_tabs,
    }, overlays=(providers_dialog,))
    providers_tabs.setCurrentIndex(1)
    settle(app, 20)
    capture.save("04_ai_settings", {
        "dialog": providers_dialog,
        "tabs": providers_tabs,
    }, overlays=(providers_dialog,))
    providers_dialog.close()
    wait_until(app, lambda: not providers_dialog.isVisible(), 5.0,
               "AI providers dialog close")

    # Measure consumes the merged arrays created by Mask. Unlike raw CQ1
    # images, these arrays already use spaCR project metadata, so no filename
    # regex is required at this stage.
    MeasureDropHandler().apply(dataset / "merged", screen)
    settle(app, 35)
    capture.save("05_folder", {
        "src": screen._settings_model._widgets.get("src"),
        "console": screen._console._console_box,
        "settings": screen._settings_scroll,
    })

    # Restore the resolved test configuration and tour the same progressive
    # settings controls users saw in Mask.
    screen.apply_settings_dict(settings)
    search = screen._settings_search
    search.set_modified_only(False)
    search.set_level("essentials")
    screen._settings_scroll.verticalScrollBar().setValue(0)
    settle(app, 25)
    capture.save("06_essentials", {
        "settings_controls": search,
        "settings": screen._settings_scroll,
    })

    search.set_modified_only(True)
    settle(app, 25)
    capture.save("07_modified", {
        "settings_controls": search,
        "settings": screen._settings_scroll,
        "modified": search._modified,
    })

    search.set_modified_only(False)
    search.set_level("all")
    screen._settings_scroll.verticalScrollBar().setValue(0)
    settle(app, 25)
    capture.save("08_all_settings", {
        "settings_controls": search,
        "settings": screen._settings_scroll,
        "all_settings": search._disclosure,
    })

    from spacr.qt.recipes import RecipeDialog
    recipe_dialog = RecipeDialog(screen, parent=window)
    center_dialog(window, recipe_dialog, 1500, 1050)
    recipe_dialog.show()
    settle(app, 25)
    capture.save("09_recipes", {
        "recipes_button": screen._recipe_button,
        "recipes_dialog": recipe_dialog,
    }, overlays=(recipe_dialog,))
    recipe_dialog.close()
    wait_until(app, lambda: not recipe_dialog.isVisible(), 5.0,
               "Recipes dialog close")

    capture.save("10_remote_submit", {
        "remote": screen._btn_remote,
        "actions": screen._actions_row,
    })
    screen._btn_remote.click()
    wait_until(
        app,
        lambda: window._screens.get("distributed_jobs") is not None
        and window._stack.currentWidget()
        is window._screens.get("distributed_jobs"),
        15.0,
        "Distributed Jobs screen",
    )
    remote_screen = window._screens["distributed_jobs"]
    settle(app, 35)
    capture.save("11_remote_jobs", {"remote_screen": remote_screen})
    window._on_nav_selected("measure")
    wait_until(app, lambda: window._stack.currentWidget() is screen, 10.0,
               "return to Measure")
    search.set_modified_only(False)
    search.set_level("all")
    settle(app, 20)

    captured_categories = []
    for index, section in enumerate(screen._settings_sections, start=1):
        for other in screen._settings_sections:
            other.set_expanded(False)
        if not section.isVisible():
            continue
        section.set_expanded(True)
        screen._settings_scroll.ensureWidgetVisible(section.header(), 12, 12)
        settle(app, 25)
        title = section.title().title()
        stem = f"12_category_{index:02d}_{safe_stem(title)}"
        capture.save(stem, {
            "category": section,
            "category_header": section.header(),
            "category_hint": screen._category_hint,
        })
        captured_categories.append(title)
    if len(captured_categories) != 8:
        raise RuntimeError(
            f"expected 8 Measure categories, found {len(captured_categories)}"
        )
    for section in screen._settings_sections:
        section.set_expanded(False)

    # Give the real crop preview the full runtime column. The saved Measure
    # configuration has normalization off; a display-only 1/99 stretch keeps
    # the 16-bit microscopy crops legible without changing propagated values.
    screen._console_wrap.hide()
    system_card = card_named(screen, "System")
    if system_card is not None:
        system_card.hide()
    screen._preview_switch.setChecked(True)
    screen._runtime_splitter.setSizes([1000, 0])
    panel._normalise.setChecked(True)
    panel._lo_pct.setValue(1.0)
    panel._hi_pct.setValue(99.0)
    panel.load_array(str(array_path))
    wait_until(app, lambda: not panel._loads_in_flight and bool(panel._crops),
               120, "Measure crop preview")
    settle(app, 40)
    capture.save("20_cell_crops", {
        "live": screen._preview_switch,
        "source": [panel._path_label, panel._max_sets_box, panel._fov_box,
                   panel._channel_box, panel._pick_btn],
        "actions": [panel._refresh_btn, panel._settings_btn],
        "grid": panel._grid_scroll,
        "live_and_grid": [screen._preview_switch, panel._grid_scroll],
    })

    dialog, tabs = open_settings(app, panel, 0)
    capture.save("21_general", {
        "tab": tabs.currentWidget(), "tabs": tabs.tabBar(),
    }, overlays=(dialog,))
    close_settings(app, dialog)

    panel._normalise.setChecked(False)
    dialog, tabs = open_settings(app, panel, 1)
    capture.save("22_object_crops", {
        "tab": tabs.currentWidget(), "tabs": tabs.tabBar(),
    }, overlays=(dialog,))
    close_settings(app, dialog)
    panel._normalise.setChecked(True)
    wait_until(app, lambda: not panel._loads_in_flight and bool(panel._crops),
               120, "contrast-normalized crop preview")

    dialog, tabs = open_settings(app, panel, 2)
    capture.save("23_filters", {
        "tab": tabs.currentWidget(), "tabs": tabs.tabBar(),
    }, overlays=(dialog,))
    close_settings(app, dialog)

    # Channel 1 is the ER signal and downstream channel of interest.
    channel_index = panel._channel_box.findText("Ch 1")
    if channel_index < 0:
        raise RuntimeError("channel 1 is unavailable in the crop preview")
    panel._channel_box.setCurrentIndex(channel_index)
    wait_until(app, lambda: not panel._loads_in_flight and bool(panel._crops),
               120, "ER crop preview")
    capture.save("24_er_channel", {
        "channel": panel._channel_box,
        "grid": panel._grid_scroll,
    })

    panel._channel_box.setCurrentIndex(0)
    wait_until(app, lambda: not panel._loads_in_flight and bool(panel._crops),
               120, "RGB crop preview")
    capture.save("25_outputs", {
        "source": panel._path_label,
        "status": panel._status,
        "grid": panel._grid_scroll,
    })
    capture.write_geometry()

    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    row_counts = {
        table: con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in ("cell", "cytoplasm", "nucleus", "pathogen", "png_list")
    }
    run_status = dict(zip(
        (row[1] for row in con.execute('PRAGMA table_info("run_status")')),
        con.execute('SELECT * FROM "run_status"').fetchone(),
    ))
    con.close()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "settings_source": str(settings_path),
        "tutorial_overrides": {
            "src": str(dataset / "merged"), "test_mode": True,
            "test_nr": 5, "n_jobs": 5,
        },
        "preview_only_contrast": [1, 99],
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "settings_categories": captured_categories,
        "representative_array": str(array_path),
        "test_database": str(db_path),
        "row_counts": row_counts,
        "run_status": run_status,
        "crop_png_count": len(list(test_root.rglob("*.png"))),
        "captures": sorted(capture.frames),
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n")

    panel.shutdown()
    window.close()
    settle(app, 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
