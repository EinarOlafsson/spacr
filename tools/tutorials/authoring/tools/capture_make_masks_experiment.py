#!/usr/bin/env python3
"""Capture the native-4K Make Masks tutorial with real ER and cell masks."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import (
    CaptureSession, card_named, nav_button, settle, wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DEFAULT_DATA = ROOT / "synthetic" / "make_masks_real"
OUTPUT = ROOT / "production" / "14_make_masks" / "keyframes"


def wait_for_image(app, screen, label: str) -> None:
    wait_until(
        app,
        lambda: (
            not screen._loading
            and screen._canvas.image is not None
            and screen._canvas.mask is not None
            and screen._canvas.pixmap() is not None
            and not screen._canvas.pixmap().isNull()
        ),
        90.0,
        label,
    )
    settle(app, 35)


def representative_point(mask, *, foreground: bool) -> tuple[int, int]:
    import numpy as np

    candidates = np.argwhere(mask > 0 if foreground else mask == 0)
    if not len(candidates):
        return mask.shape[1] // 2, mask.shape[0] // 2
    # A stable point near the middle avoids edge-clipped clicks.
    y, x = candidates[len(candidates) // 2]
    return int(x), int(y)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    data = args.data.resolve()
    if not (data / "manifest.json").is_file():
        raise FileNotFoundError("run prepare_make_masks_tutorial_data.py first")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-make-masks-4k-config")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt import mask_engine as engine
    from spacr.qt.preferences import apply_preferences_to_app

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")
    window._on_nav_selected("make_masks")
    settle(app, 80)
    screen = window._screens.get("make_masks")
    if screen is None:
        raise RuntimeError("Make Masks screen did not open")

    capture = CaptureSession(app, window, OUTPUT)
    capture.save("01_empty", {
        "nav": nav_button(window, "make_masks"),
        "empty": screen._empty_state,
        "open": screen._btn_open,
    })

    screen._open_folder(str(data))
    wait_for_image(app, screen, "first real ER field")
    capture.save("02_loaded", {
        "source": screen._src_label,
        "canvas": screen._canvas,
        "tools": screen._body_splitter.widget(1),
        "navigation": [screen._btn_prev, screen._btn_next, screen._btn_save],
    })

    tools_card = card_named(screen, "Tools")
    brush_card = card_named(screen, "Brush")
    wand_card = card_named(screen, "Magic wand")
    norm_card = card_named(screen, "Normalize")
    object_card = card_named(screen, "Object operations")
    capture.save("03_tools", {
        "tools": tools_card,
        "canvas": screen._canvas,
    })

    original = screen._canvas.mask.copy()
    bg_x, bg_y = representative_point(original, foreground=False)
    fg_x, fg_y = representative_point(original, foreground=True)

    screen._set_mode("brush")
    screen._brush_slider.setValue(18)
    screen._history.push(screen._canvas.mask)
    engine.paint_line(screen._canvas.mask, bg_x - 50, bg_y, bg_x + 50, bg_y, 24, 255)
    screen._on_stroke_finished()
    screen._canvas.refresh()
    capture.save("04_brush", {
        "button": screen._btn_brush,
        "brush": brush_card,
        "canvas": screen._canvas,
    })

    screen._set_mode("erase")
    engine.paint_line(screen._canvas.mask, fg_x - 40, fg_y, fg_x + 40, fg_y, 20, 0)
    screen._on_stroke_finished()
    screen._canvas.refresh()
    capture.save("05_erase", {
        "button": screen._btn_erase,
        "brush": brush_card,
        "canvas": screen._canvas,
    })

    screen._canvas.mask = original.copy()
    screen._set_mode("erase_object")
    screen._canvas.mask = engine.erase_object_at(screen._canvas.mask, fg_x, fg_y)
    screen._on_stroke_finished()
    screen._canvas.refresh()
    capture.save("06_erase_object", {
        "button": screen._btn_del_obj,
        "canvas": screen._canvas,
    })

    screen._canvas.mask = original.copy()
    screen._set_mode("wand_add")
    screen._wand_tol.setValue(120.0)
    screen._wand_max.setValue(12000)
    screen._canvas.mask = engine.magic_wand(
        screen._canvas.image, screen._canvas.mask, bg_x, bg_y,
        screen._canvas.wand_tolerance, screen._canvas.wand_max_pixels,
        action="add",
    )
    screen._on_stroke_finished()
    screen._canvas.refresh()
    capture.save("07_wand_add", {
        "button": screen._btn_wand_add,
        "wand": wand_card,
        "canvas": screen._canvas,
    })

    wand_x, wand_y = representative_point(screen._canvas.mask, foreground=True)
    screen._set_mode("wand_erase")
    screen._canvas.mask = engine.magic_wand(
        screen._canvas.image, screen._canvas.mask, wand_x, wand_y,
        screen._canvas.wand_tolerance, screen._canvas.wand_max_pixels,
        action="erase",
    )
    screen._on_stroke_finished()
    screen._canvas.refresh()
    capture.save("08_wand_erase", {
        "button": screen._btn_wand_erase,
        "wand": wand_card,
        "canvas": screen._canvas,
    })

    height, width = screen._canvas.mask.shape
    screen._set_mode("zoom")
    screen._canvas._zoom_x0 = width // 4
    screen._canvas._zoom_y0 = height // 4
    screen._canvas._zoom_x1 = width * 3 // 4
    screen._canvas._zoom_y1 = height * 3 // 4
    screen._canvas.zoom_changed.emit(True)
    screen._canvas.refresh()
    settle(app, 25)
    capture.save("09_zoom", {
        "button": screen._btn_zoom,
        "reset": screen._btn_reset_zoom,
        "canvas": screen._canvas,
    })

    screen._norm_lo.setValue(2.0)
    screen._norm_hi.setValue(99.5)
    screen._on_normalize_changed(0.0)
    settle(app, 25)
    capture.save("10_normalize", {
        "normalize": norm_card,
        "canvas": screen._canvas,
    })

    screen._on_undo()
    capture.save("11_history", {
        "history": [screen._btn_undo, screen._btn_redo],
        "canvas": screen._canvas,
    })
    screen._on_redo()

    screen._canvas.reset_zoom()
    screen._canvas.mask = original.copy()
    screen._canvas.refresh()
    capture.save("12_object_operations", {
        "operations": object_card,
        "canvas": screen._canvas,
    })

    screen._min_area.setValue(10000)
    screen._on_remove_small()
    capture.save("13_remove_small", {
        "operations": object_card,
        "min_area": screen._min_area,
        "canvas": screen._canvas,
    })

    screen._on_save()
    settle(app, 30)
    capture.save("14_save", {
        "save": screen._btn_save,
        "status": screen._status_label,
        "canvas": screen._canvas,
    })

    screen._on_next()
    wait_for_image(app, screen, "second real ER field")
    capture.save("15_next", {
        "next": screen._btn_next,
        "navigation": [screen._btn_prev, screen._btn_next, screen._btn_save],
        "status": screen._status_label,
        "canvas": screen._canvas,
    })

    screen._on_next()
    wait_for_image(app, screen, "blank real ER field")
    screen._set_mode("brush")
    capture.save("16_blank_mask", {
        "source": screen._src_label,
        "button": screen._btn_brush,
        "brush": brush_card,
        "canvas": screen._canvas,
        "save": screen._btn_save,
    })

    capture.write_geometry()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lesson_id": "14_make_masks",
        "source": str(data),
        "image_channel": "ER (1)",
        "mask_object": "cell",
        "normalization": [2.0, 99.5],
        "minimum_area": 10000,
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    screen.close()
    settle(app, 40)
    window.close()
    settle(app, 40)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
