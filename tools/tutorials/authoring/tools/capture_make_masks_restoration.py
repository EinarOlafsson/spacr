#!/usr/bin/env python3
"""Capture native-4K Make Masks restoration scenes (item 489) on the CPU.

Records the Image enhancement card's Deep image enhancement row with the
real isolated Cellpose 3 restoration environment, forced onto the CPU
(``SPACR_DEVICE=cpu``, CUDA hidden) and with Show alpha features OFF
(CaptureSession refuses a frame otherwise). The frames are staged, not
published: the output directory is an argument so the publisher's tree is
never written.

Scenes: 01 the restoration row on the card; 02 Denoise chosen with its
model settings open; 03 the model loaded on the CPU; 04 Compare raw and
enhanced; 05 Apply, the canvas showing the restored field.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["SPACR_DEVICE"] = "cpu"

from capture_mask_experiment import (  # noqa: E402
    CaptureSession, settle, wait_until,
)
from capture_make_masks_experiment import REPO, wait_for_image  # noqa: E402

CAPTIONS = {
    "01_restoration_row": (
        "Image enhancement ends with Deep image enhancement. It restores the "
        "selected channel with a Cellpose 3 model in its own environment; "
        "the image on disk is never changed."),
    "02_denoise_settings": (
        "Choose Denoise, Deblur or One-click restoration, then open "
        "Restoration model settings. Pick weights for cells or nuclei and "
        "set the typical object diameter in pixels."),
    "03_model_ready": (
        "Load the model. On the CPU a whole field takes tens of seconds; "
        "the original intensities are kept for measurements."),
    "04_compare": (
        "Compare raw and enhanced shows both pictures side by side. The "
        "restored picture is model output, not new structure: check it "
        "before trusting masks drawn on it."),
    "05_apply": (
        "Apply uses the restored picture for display and detection only. "
        "Measurements still read the original pixels."),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve()

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication, QScrollArea

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.screens import make_masks as mm

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tools/tutorials
    from capture_policy import force_fresh_start, verify_fresh_start
    force_fresh_start()  # `spacr --fresh`, no remembered session, no crash drafts
    window = app_module.MainWindow()
    verify_fresh_start(window)
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    window._on_nav_selected("make_masks")
    settle(app, 80)
    screen = window._screens["make_masks"]
    screen._open_folder(str(data))
    wait_for_image(app, screen, "first real ER field")
    # The third field has no saved mask, so the canvas shows the picture the
    # restoration changes rather than a coloured label overlay.
    screen._on_next()
    wait_for_image(app, screen, "second real ER field")
    screen._on_next()
    wait_for_image(app, screen, "blank real ER field")

    def neutral_source():
        # 447: no host path in a frame; the neutral root the redaction
        # tool uses (tools/tutorials/redact_frame_paths.py).
        screen._src_label.setText("/data/example/make_masks  —  3 images")
        settle(app, 10)

    from spacr.qt.widgets.section import Section

    restoration = screen._restoration_controls
    card = restoration.parent()
    while card is not None and not isinstance(card, Section):
        card = card.parent()
    if card is None:
        raise RuntimeError("Image enhancement card or restoration row missing")
    card.set_expanded(True)
    settle(app, 40)

    def reveal(widget):
        parent = widget.parent()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parent()
        if parent is not None:
            parent.ensureWidgetVisible(widget, 40, 40)
        settle(app, 30)

    capture = CaptureSession(app, window, output)
    _save = capture.save

    def save(*a, **k):
        neutral_source()
        return _save(*a, **k)

    capture.save = save
    reveal(restoration)
    capture.save("01_restoration_row", {
        "card": card, "row": restoration.operation, "canvas": screen._canvas})

    restoration.operation.setCurrentIndex(restoration.operation.findData("denoise"))
    restoration.structure.setCurrentIndex(restoration.structure.findData("cyto3"))
    if hasattr(restoration.details, "set_expanded"):
        restoration.details.set_expanded(True)
    elif hasattr(restoration.details, "setExpanded"):
        restoration.details.setExpanded(True)
    settle(app, 30)
    reveal(restoration.reload)
    capture.save("02_denoise_settings", {
        "row": restoration.operation, "model": restoration.structure,
        "diameter": restoration.diameter})

    started = time.monotonic()
    restoration.reload.click()
    wait_until(app, lambda: "Loading" not in restoration.status.text() and (
        restoration._plan is not None
        or "unavailable" in restoration.status.text()), 900,
        "restoration model on the CPU")
    load_seconds = time.monotonic() - started
    if restoration._plan is None:
        raise RuntimeError(f"restoration did not load: {restoration.status.text()}")
    reveal(restoration.status)
    capture.save("03_model_ready", {"status": restoration.status,
                                    "reload": restoration.reload})

    finished = {}
    original = mm._ComparePreview._show_result

    def _record(self, enhanced, caption):
        original(self, enhanced, caption)
        finished["enhanced"] = enhanced is not None
        finished["caption"] = caption

    mm._ComparePreview._show_result = _record
    started = time.monotonic()
    screen._btn_compare.click()
    wait_until(app, lambda: "enhanced" in finished, 900, "CPU comparison")
    compare_seconds = time.monotonic() - started
    if not finished["enhanced"]:
        raise RuntimeError(f"comparison discarded: {finished['caption']}")
    dialog = screen._compare_dialog
    dialog.resize(2400, 1300)
    geo = window.geometry()
    dialog.move(geo.x() + (geo.width() - 2400) // 2,
                geo.y() + (geo.height() - 1300) // 2)
    settle(app, 40)
    dialog.fit()
    capture.save("04_compare", {"dialog": dialog}, overlays=(dialog,))
    dialog.close()
    settle(app, 20)

    said = []
    screen._canvas.status.connect(said.append)
    started = time.monotonic()
    screen._btn_apply.setChecked(True)
    wait_until(app, lambda: any("restoration finished" in text
                                or "enhancement failed" in text for text in said),
               900, "applied restoration")
    if any("enhancement failed" in text for text in said):
        raise RuntimeError(said[-1])
    settle(app, 60)
    apply_seconds = time.monotonic() - started
    capture.save("05_apply", {"apply": screen._btn_apply, "canvas": screen._canvas})

    (output / "geometry.json").write_text(json.dumps(capture.frames, indent=2) + "\n")
    (output / "scenes_en.json").write_text(json.dumps({
        "schema": 1, "item": 489, "lesson": "14_make_masks (restoration insert)",
        "captions": CAPTIONS}, indent=2, ensure_ascii=False) + "\n")
    (output / "capture_receipt.json").write_text(json.dumps({
        "schema": 1,
        "item": 489,
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_manifest": json.loads((data / "manifest.json").read_text()),
        "device": "cpu (SPACR_DEVICE=cpu, CUDA/HIP hidden)",
        "alpha_features": "off (CaptureSession.refuse_alpha_features per frame)",
        "restoration": {"operation": "denoise", "model": "cyto3",
                        "diameter_px": restoration.diameter.value(),
                        "status": restoration.status.text(),
                        "plan_device": str(getattr(restoration._plan, "device", "")),
                        "load_seconds": round(load_seconds, 2),
                        "compare_seconds": round(compare_seconds, 2),
                        "apply_seconds": round(apply_seconds, 2)},
        "frames": sorted(capture.frames),
        "claim": "UI walkthrough only; restored pixels are model output, "
                 "no accuracy or biological-improvement claim.",
    }, indent=2, ensure_ascii=False) + "\n")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
