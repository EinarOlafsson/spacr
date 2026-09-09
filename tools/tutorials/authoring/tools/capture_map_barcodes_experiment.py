#!/usr/bin/env python3
"""Capture Map Barcodes at native 4K with a reproducible FASTQ demo."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import (
    CaptureSession, center_dialog, nav_button, safe_stem, settle, wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DEFAULT_DATASET = ROOT / "synthetic" / "map_barcodes"
OUTPUT = ROOT / "production" / "12_map_barcodes" / "keyframes"


class MotionCaptureSession(CaptureSession):
    """Save a short native-Qt frame loop beside every sequencing keyframe."""

    def save(self, stem, widgets=None, overlays=()):
        path = super().save(stem, widgets, overlays)
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPainter

        motion_dir = self.output / "motion" / stem
        motion_dir.mkdir(parents=True, exist_ok=True)
        root_origin = self.window.mapToGlobal(QPoint(0, 0))
        for index in range(8):
            # Advancing the event loop lets the real DnaRainWidget move; no
            # synthetic glyph layer is added by the video renderer.
            settle(self.app, 5)
            pixmap = self.window.grab()
            painter = QPainter(pixmap)
            for overlay in overlays:
                if overlay is None or not overlay.isVisible():
                    continue
                position = overlay.mapToGlobal(QPoint(0, 0)) - root_origin
                painter.drawPixmap(position, overlay.grab())
            painter.end()
            frame_path = motion_dir / f"{index:02d}.png"
            if not pixmap.save(str(frame_path), "PNG"):
                raise RuntimeError(f"could not save {frame_path}")
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    required = (
        dataset / "demo_R1_001.fastq.gz",
        dataset / "demo_R2_001.fastq.gz",
        dataset / "barcodes" / "grna.csv",
        dataset / "barcodes" / "row.csv",
        dataset / "barcodes" / "column.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing synthetic sequencing inputs: {missing}")

    # The spaCR pipeline runs in a QThread. Forking a Qt process from that
    # thread can let the sequencing writer finish its files but still exit 1;
    # spawn gives its worker pool a clean interpreter, as a normal installed
    # application does on macOS and Windows.
    multiprocessing.set_start_method("spawn", force=True)

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-map-barcodes-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-map-barcodes-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QTabWidget

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.dnd_handlers import MapBarcodesDropHandler
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.settings import set_default_generate_barecode_mapping

    settings = set_default_generate_barecode_mapping(settings={})
    settings.update({
        "src": str(dataset),
        "grna_csv": str(dataset / "barcodes" / "grna.csv"),
        "row_csv": str(dataset / "barcodes" / "row.csv"),
        "column_csv": str(dataset / "barcodes" / "column.csv"),
        "mode": "paired",
        "single_direction": "R1",
        "chunk_size": 1000,
        "n_jobs": 2,
        "save_h5": False,
        "test": False,
        "fill_na": False,
    })

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")
    window._on_nav_selected("map_barcodes")
    settle(app, 90)
    screen = window._screens.get("map_barcodes")
    if screen is None:
        raise RuntimeError("Map Barcodes screen did not open")
    screen.apply_settings_dict(settings)
    wait_until(
        app,
        lambda: screen._settings_search is not None
        and screen._recipe_button is not None
        and screen._dna_rain is not None,
        15.0,
        "Map Barcodes controls",
    )
    screen._dna_rain.settings_bar.set_opacity(0.10)
    settle(app, 35)
    capture = MotionCaptureSession(app, window, OUTPUT)

    capture.save("01_console", {
        "nav": nav_button(window, "map_barcodes"),
        "console": screen._console._console_box,
        "chat": screen._console._chat_row,
        "actions": screen._actions_row,
    })

    dna_button = screen._dna_rain.settings_button
    dna_button.setChecked(True)
    wait_until(app, screen._dna_rain.settings_popover.isVisible, 5.0,
               "DNA rain settings")
    settle(app, 20)
    capture.save("02_dna_visibility", {
        "dna_button": dna_button,
        "dna_settings": screen._dna_rain.settings_bar,
        "visibility": screen._dna_rain.settings_bar._opacity,
    }, overlays=(screen._dna_rain.settings_popover,))
    dna_button.setChecked(False)
    settle(app, 15)

    screen._ai_menu.popup(screen._ai_menu_btn.mapToGlobal(
        QPoint(0, screen._ai_menu_btn.height())))
    wait_until(app, screen._ai_menu.isVisible, 5.0, "AI provider menu")
    capture.save("03_ai_menu", {
        "ai_controls": [screen._ai_switch, screen._ai_menu_btn],
        "ai_menu_button": screen._ai_menu_btn,
        "ai_menu": screen._ai_menu,
    }, overlays=(screen._ai_menu,))
    screen._ai_menu.hide()
    settle(app, 12)

    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
    providers = _ProvidersDialog(screen)
    center_dialog(window, providers, 1500, 1200)
    providers.show()
    settle(app, 25)
    tabs = providers.findChild(QTabWidget)
    capture.save("04_ai_providers", {"dialog": providers}, overlays=(providers,))
    tabs.setCurrentIndex(1)
    settle(app, 20)
    capture.save("05_ai_settings", {"dialog": providers}, overlays=(providers,))
    providers.close()

    MapBarcodesDropHandler().apply(dataset, screen)
    screen.apply_settings_dict(settings)
    settle(app, 30)
    capture.save("06_fastq_drop", {
        "src": screen._settings_model._widgets.get("src"),
        "console": screen._console._console_box,
        "settings": screen._settings_scroll,
    })

    search = screen._settings_search
    search.set_modified_only(False)
    search.set_level("essentials")
    screen._settings_scroll.verticalScrollBar().setValue(0)
    settle(app, 25)
    capture.save("07_essentials", {
        "settings_controls": search,
        "settings": screen._settings_scroll,
    })
    search.set_modified_only(True)
    settle(app, 25)
    capture.save("08_modified", {
        "settings_controls": search,
        "modified": search._modified,
        "settings": screen._settings_scroll,
    })
    search.set_modified_only(False)
    search.set_level("all")
    screen._settings_scroll.verticalScrollBar().setValue(0)
    settle(app, 25)
    capture.save("09_all_settings", {
        "settings_controls": search,
        "all_settings": search._disclosure,
        "settings": screen._settings_scroll,
    })

    from spacr.qt.recipes import RecipeDialog
    recipes = RecipeDialog(screen, parent=window)
    center_dialog(window, recipes, 1500, 1050)
    recipes.show()
    settle(app, 25)
    capture.save("10_recipes", {
        "recipes_button": screen._recipe_button,
        "dialog": recipes,
    }, overlays=(recipes,))
    recipes.close()

    capture.save("11_remote_submit", {
        "remote": screen._btn_remote,
        "actions": screen._actions_row,
    })
    screen._btn_remote.click()
    wait_until(
        app,
        lambda: window._screens.get("distributed_jobs") is not None
        and window._stack.currentWidget() is window._screens.get("distributed_jobs"),
        15.0,
        "Distributed Jobs screen",
    )
    remote = window._screens["distributed_jobs"]
    settle(app, 30)
    capture.save("12_remote_jobs", {"remote_screen": remote})
    window._on_nav_selected("map_barcodes")
    wait_until(app, lambda: window._stack.currentWidget() is screen, 10.0,
               "return to Map Barcodes")
    screen.apply_settings_dict(settings)
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
        capture.save(f"13_category_{index:02d}_{safe_stem(title)}", {
            "category": section,
            "category_header": section.header(),
            "category_hint": screen._category_hint,
        })
        captured_categories.append(title)
    if len(captured_categories) != 5:
        raise RuntimeError(
            f"expected 5 Map Barcodes categories, found {captured_categories}"
        )

    parsing = next(
        section for section in screen._settings_sections
        if "read parsing" in section.title().lower()
    )
    for section in screen._settings_sections:
        section.set_expanded(section is parsing)
    screen._settings_scroll.ensureWidgetVisible(parsing.header(), 12, 12)
    settle(app, 25)
    capture.save("18_regex_mapping", {
        "category": parsing,
        "target": screen._settings_model._widgets.get("target_sequence"),
        "regex": screen._settings_model._widgets.get("regex"),
        "offset": screen._settings_model._widgets.get("offset_start"),
        "end": screen._settings_model._widgets.get("expected_end"),
    })

    input_section = next(
        section for section in screen._settings_sections
        if "sequencing input" in section.title().lower()
    )
    for section in screen._settings_sections:
        section.set_expanded(section is input_section)
    screen._settings_scroll.ensureWidgetVisible(input_section.header(), 12, 12)
    settle(app, 20)
    capture.save("19_paired_reads", {
        "category": input_section,
        "src": screen._settings_model._widgets.get("src"),
        "mode": screen._settings_model._widgets.get("mode"),
        "direction": screen._settings_model._widgets.get("single_direction"),
    })

    capture.save("20_run", {
        "run": screen._btn_run,
        "stop": screen._btn_stop,
        "actions": screen._actions_row,
        "console": screen._console._console_box,
    })
    screen._btn_run.click()
    wait_until(app, lambda: screen._btn_stop.isEnabled(), 10.0,
               "Map Barcodes run start")
    settle(app, 35)
    capture.save("21_running", {
        "run": screen._btn_run,
        "stop": screen._btn_stop,
        "console": screen._console._console_box,
    })
    wait_until(app, lambda: screen._btn_run.isEnabled(), 120.0,
               "Map Barcodes completion")
    settle(app, 50)
    capture.save("22_output", {
        "console": screen._console._console_box,
        "actions": screen._actions_row,
    })

    capture.write_geometry()
    outputs = sorted(
        str(path.relative_to(dataset)) for path in dataset.rglob("*")
        if path.is_file() and path not in required
        and path.name != "settings_map_barcodes.csv"
    )
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "input_fastqs": [str(required[0]), str(required[1])],
        "synthetic_reads": 4800,
        "synthetic_wells": 24,
        "synthetic_grnas": 12,
        "dna_visibility": 0.10,
        "settings": settings,
        "settings_categories": captured_categories,
        "outputs": outputs,
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    screen.close()
    settle(app, 40)
    window.close()
    settle(app, 40)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
