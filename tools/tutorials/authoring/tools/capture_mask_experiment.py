#!/usr/bin/env python3
"""Capture the Mask tutorial from the real spaCR app and experiment.

The source experiment and its settings CSV are inputs only. Tutorial-specific
overrides are kept in memory and recorded in ``source_manifest.json`` beside
the captures. The tutorial deliberately runs only the ER-defined cell preview;
the settings-category tour still shows where every other workflow is configured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DEFAULT_DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
OUTPUT = ROOT / "production" / "07_mask" / "keyframes"
FRAME_SIZE = (3840, 2160)
PREVIEW_DIAMETERS = {"cell": 120.0, "nucleus": 60.0, "pathogen": 20.0,
                     "organelle": 30.0}


def object_count(mask) -> int:
    """Count labels without assuming that their ids are consecutive."""
    import numpy as np

    labels = np.unique(mask)
    return int(np.count_nonzero(labels))


def conservative_min_area(mask, target_fraction: float = 0.08,
                          maximum_fraction: float = 0.12) -> tuple[int, dict]:
    """Choose a visible but conservative small-object cutoff from ``mask``.

    The tutorial must demonstrate how the control behaves without presenting
    one dataset's pixel value as a recommendation.  Candidate cutoffs are
    measured from the current preview, and the closest one to the lower eight
    percent of object areas is accepted only when it removes no more than
    twelve percent of the labels.
    """
    import numpy as np

    labels, areas = np.unique(mask, return_counts=True)
    areas = areas[labels > 0].astype(np.int64, copy=False)
    if areas.size < 12:
        raise RuntimeError(
            "the representative Mask preview has too few objects for a "
            "minimum-area demonstration"
        )
    target = max(1, int(round(areas.size * target_fraction)))
    maximum = max(target, int(areas.size * maximum_fraction))
    candidates = []
    for threshold in np.unique(areas) + 1:
        removed = int(np.count_nonzero(areas < threshold))
        if 1 <= removed <= maximum:
            candidates.append((abs(removed - target), removed,
                               int(threshold)))
    if not candidates:
        raise RuntimeError(
            "could not derive a conservative minimum-area cutoff from the "
            "representative Mask preview"
        )
    _distance, removed, threshold = min(candidates)
    return threshold, {
        "method": "preview-derived lower-area-tail cutoff",
        "target_removed_fraction": target_fraction,
        "maximum_removed_fraction": maximum_fraction,
        "objects_measured": int(areas.size),
        "objects_below_cutoff": removed,
        "area_pixels": {
            "minimum": int(areas.min()),
            "median": float(np.median(areas)),
            "maximum": int(areas.max()),
        },
    }


def settle(app, cycles: int = 30) -> None:
    for _ in range(cycles):
        app.processEvents()


def wait_until(app, predicate, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            settle(app, 8)
            return
        time.sleep(0.025)
    raise TimeoutError(f"timed out waiting for {label}")


def nav_button(window, key: str):
    from PySide6.QtWidgets import QPushButton

    for button in window._sidebar.findChildren(QPushButton):
        if button.property("navKey") == key:
            return button
    return None


def card_named(screen, title: str):
    from PySide6.QtWidgets import QLabel
    from spacr.qt.widgets.card import Card

    for card in screen.findChildren(Card):
        headings = [label.text() for label in card.findChildren(QLabel)
                    if label.objectName() == "CardTitle"]
        if title in headings:
            return card
    return None


def union_rect(*rectangles):
    valid = [item for item in rectangles if item]
    if not valid:
        return None
    left = min(item[0] for item in valid)
    top = min(item[1] for item in valid)
    right = max(item[0] + item[2] for item in valid)
    bottom = max(item[1] + item[3] for item in valid)
    return [left, top, right - left, bottom - top]


class CaptureSession:
    def __init__(self, app, window, output: Path):
        self.app = app
        self.window = window
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames: dict[str, dict] = {}

    def rect(self, widget):
        if widget is None or not widget.isVisible():
            return None
        from PySide6.QtCore import QPoint

        preview = self.window.grab()
        scale = float(preview.devicePixelRatio())
        window_origin = self.window.mapToGlobal(QPoint(0, 0))
        origin = widget.mapToGlobal(QPoint(0, 0))
        x = int(round((origin.x() - window_origin.x()) * scale))
        y = int(round((origin.y() - window_origin.y()) * scale))
        width = int(round(widget.width() * scale))
        height = int(round(widget.height() * scale))
        left = max(0, x)
        top = max(0, y)
        right = min(preview.width(), x + width)
        bottom = min(preview.height(), y + height)
        if right <= left or bottom <= top:
            return None
        return [left, top, right - left, bottom - top]

    def save(self, stem: str, widgets: dict[str, object] | None = None,
             overlays: tuple[object, ...] = ()) -> Path:
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QPainter

        settle(self.app, 20)
        pixmap = self.window.grab()
        scale = float(pixmap.devicePixelRatio())
        root_origin = self.window.mapToGlobal(QPoint(0, 0))
        painter = QPainter(pixmap)
        for overlay in overlays:
            if overlay is None or not overlay.isVisible():
                continue
            position = overlay.mapToGlobal(QPoint(0, 0)) - root_origin
            painter.drawPixmap(position, overlay.grab())
        painter.end()
        if (pixmap.width(), pixmap.height()) != FRAME_SIZE:
            raise RuntimeError(
                f"capture is {pixmap.width()}x{pixmap.height()}, expected "
                f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}"
            )
        path = self.output / f"{stem}.png"
        if not pixmap.save(str(path), "PNG"):
            raise RuntimeError(f"could not save {path}")
        geometry = {
            # ``image`` is also a useful widget key for the source canvas.
            # Keep the frame filename under an unambiguous field so a canvas
            # rectangle cannot overwrite it in geometry.json.
            "file": path.name,
            "device_pixel_ratio": scale,
        }
        for name, widget in (widgets or {}).items():
            if isinstance(widget, (list, tuple)):
                geometry[name] = union_rect(*(self.rect(item) for item in widget))
            else:
                geometry[name] = self.rect(widget)
        self.frames[stem] = geometry
        print(path, flush=True)
        return path

    def write_geometry(self) -> None:
        path = self.output / "geometry.json"
        path.write_text(json.dumps({
            "schema": 2,
            "frame_size": list(FRAME_SIZE),
            "frames": self.frames,
        }, indent=2) + "\n")


def set_combo(combo, text: str) -> None:
    index = combo.findText(text)
    if index < 0:
        raise ValueError(f"{text!r} is not in {type(combo).__name__}")
    combo.setCurrentIndex(index)


def set_widget(widget, value) -> None:
    from PySide6.QtWidgets import QCheckBox, QComboBox

    if value is None or value == "":
        return
    if isinstance(widget, QComboBox):
        set_combo(widget, str(value))
    elif isinstance(widget, QCheckBox) or hasattr(widget, "setChecked"):
        widget.setChecked(str(value).lower() in ("true", "1", "yes"))
    elif hasattr(widget, "setValue"):
        widget.setValue(value)


def apply_object_settings(panel, settings: dict, obj: str,
                          *, unfiltered: bool = False) -> None:
    set_combo(panel._object_box, obj)
    panel._settings = dict(settings)
    panel._diameter.setValue(float(
        settings.get(f"{obj}_diameter") or PREVIEW_DIAMETERS[obj]))
    panel._flow.setValue(float(settings.get(f"{obj}_FT", 0.4)))
    panel._prob.setValue(float(settings.get(f"{obj}_CP_prob", 0.0)))
    if obj == "cell":
        panel._cell_channel.setValue(int(settings.get("cell_channel", 0)))
    elif obj == "nucleus":
        panel._nucleus_channel.setValue(int(settings.get("nucleus_channel", 0)))

    signal = settings.get(f"{obj}_Signal_to_noise", 10)
    set_widget(panel._common_widgets["signal_to_noise"], round(float(signal)))
    set_widget(panel._common_widgets["remove_background"],
               settings.get(f"remove_background_{obj}", False))
    set_widget(panel._common_widgets["background"],
               settings.get(f"{obj}_background", 100))
    for suffix, widget in panel._compartment_widgets[obj].items():
        value = settings.get(f"{obj}_{suffix}")
        if unfiltered and suffix in ("min_area", "min_intensity_percentile"):
            value = 0
        set_widget(widget, value)


def run_preview(app, panel, label: str, timeout: float = 420.0) -> None:
    panel.run_preview()
    wait_until(
        app,
        lambda: bool(panel._raw_masks) and panel._run_btn.isEnabled(),
        timeout,
        label,
    )
    wait_until(
        app,
        lambda: panel._worker is None or not panel._worker.isRunning(),
        30.0,
        f"{label} worker shutdown",
    )


def open_settings(app, panel):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QGroupBox

    panel.open_live_settings()
    wait_until(
        app,
        lambda: panel._live_settings_dialog is not None
        and panel._live_settings_dialog.isVisible(),
        10.0,
        "Live settings dialog",
    )
    dialog = panel._live_settings_dialog
    dialog.resize(1760, 900)
    top_level = panel.window()
    left = max(0, (top_level.width() - dialog.width()) // 2)
    top = max(0, (top_level.height() - dialog.height()) // 2)
    dialog.move(top_level.mapToGlobal(QPoint(left, top)))
    settle(app, 20)
    segmentation = next(
        (box for box in dialog.findChildren(QGroupBox)
         if box.title() == "Segmentation"),
        None,
    )
    return dialog, segmentation


def close_settings(app, dialog) -> None:
    dialog.close()
    wait_until(app, lambda: not dialog.isVisible(), 10.0,
               "Live settings dialog close")


def zoom_views(app, panel, factor: float = 2.6) -> None:
    panel._src_view.reset_zoom()
    panel._mask_view.reset_zoom()
    panel._src_view._apply_zoom(factor, broadcast=True)
    for view in (panel._src_view, panel._mask_view):
        view.horizontalScrollBar().setValue(
            int(view.horizontalScrollBar().maximum() * 0.48))
        view.verticalScrollBar().setValue(
            int(view.verticalScrollBar().maximum() * 0.48))
    settle(app, 20)


def center_dialog(window, dialog, width: int, height: int) -> None:
    from PySide6.QtCore import QPoint

    dialog.resize(width, height)
    dialog.move(window.mapToGlobal(QPoint(
        max(0, (window.width() - dialog.width()) // 2),
        max(0, (window.height() - dialog.height()) // 2),
    )))


def safe_stem(value: str) -> str:
    return "_".join(
        part for part in "".join(
            char.lower() if char.isalnum() else " " for char in value
        ).split()
        if part
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--field", default="W0107F0002T0001Z000")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    orig = dataset / "orig"
    settings_path = dataset / "settings" / "gen_mask_settings.csv"
    if not orig.is_dir() or not settings_path.is_file():
        raise FileNotFoundError("dataset needs orig/ and settings/gen_mask_settings.csv")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-mask-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-mask-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import (
        QApplication, QDialogButtonBox, QTabWidget,
    )
    import spacr
    spacr.__version__ = "1.5.0.4"
    from spacr.utils import load_settings
    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt import dnd_handlers
    from spacr.qt.dnd_handlers import MaskDropHandler, active_scan_jobs
    from spacr.qt import regex_detect
    from spacr.qt.preferences import apply_preferences_to_app

    settings = load_settings(
        str(settings_path), setting_key="Key", setting_value="Value")
    settings.update({
        "src": str(orig),
        "test_mode": True,
        "test_images": 5,
        "random_test": True,
        "cell_min_area": 0,
        "delete_intermediate": False,
        "keep_intermediate": True,
    })

    channel_files = {"cell": orig / f"{args.field}C2.tif"}
    missing = [str(path) for path in channel_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing representative images: " + ", ".join(missing))

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(*FRAME_SIZE)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError(
            f"capture device pixel ratio is {window.devicePixelRatioF()}, "
            "expected 1.0 for a 100% scale 4K tutorial"
        )
    window._on_nav_selected("mask")
    settle(app, 90)
    screen = window._screens.get("mask")
    if screen is None:
        raise RuntimeError("Mask screen did not open")
    screen.apply_settings_dict(settings)
    panel = screen._live_preview
    panel.apply_settings(settings)
    wait_until(
        app,
        lambda: getattr(screen, "_settings_search", None) is not None
        and getattr(screen, "_recipe_button", None) is not None,
        15.0,
        "Mask settings search and Recipes controls",
    )
    settle(app, 45)

    capture = CaptureSession(app, window, OUTPUT)
    capture.save("01_console", {
        "nav": nav_button(window, "mask"),
        "console": screen._console._console_box,
        "chat": screen._console._chat_row,
        "actions": screen._actions_row,
    })

    # The chat field can run as a local command console or route a question to
    # a configured subscription-backed AI provider. Show the actual provider
    # menu, then both tabs of the providers/settings dialog.
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

    # CQ1 filenames do not carry a plate id. spaCR can safely use its
    # documented ``plate1`` default. During automated capture, keep genuine
    # validation warnings while allowing that optional note to remain
    # non-blocking.
    validate_records = regex_detect.validate_records
    regex_detect.validate_records = lambda records, multichannel=True: [
        warning for warning in validate_records(
            records, multichannel=multichannel)
        if not str(warning).startswith("Optional: no plateID captured.")
    ]
    # The current drop report opens its own blocking confirmation editor after
    # a successful auto-detection.  Capture the same real editor explicitly
    # below, where it can be sized and recorded, so suppress only that
    # automatic modal during the background scan.  Preserve the detected
    # pattern exactly as accepting the confirmation would.
    open_regex_editor = dnd_handlers._open_regex_editor
    dnd_handlers._open_regex_editor = (
        lambda _names, initial, target, confirming=False, fallback=None:
        dnd_handlers._push_regex_to_screen(fallback or initial, target)
    )
    try:
        MaskDropHandler().apply(orig, screen)
        wait_until(app, lambda: active_scan_jobs(screen) == 0, 30.0,
                   "folder metadata scan")
    finally:
        dnd_handlers._open_regex_editor = open_regex_editor
        regex_detect.validate_records = validate_records
    settle(app, 45)

    # Show the real metadata-regex preview that users can open after a drop,
    # then save the auto-detected CQ1 pattern.  The drop report already pushed
    # this pattern into the screen; opening the dialog here makes the review
    # and Save action explicit without changing the experiment settings.
    from spacr.qt.regex_editor import RegexEditorDialog
    sample_names = sorted(
        path.name for path in orig.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )[:20]
    detected_pattern, _detected_label, detected_hits = (
        regex_detect.auto_detect_regex(sample_names)
    )
    if not detected_pattern or detected_hits != len(sample_names):
        raise RuntimeError("CQ1 metadata auto-detection did not match the sample")
    metadata_dialog = RegexEditorDialog(
        sample_names,
        initial_regex=detected_pattern,
        multichannel=True,
        parent=window,
    )
    center_dialog(window, metadata_dialog, 1500, 900)
    metadata_dialog.show()
    settle(app, 35)
    button_box = metadata_dialog.findChild(QDialogButtonBox)
    save_button = button_box.button(QDialogButtonBox.Save) if button_box else None
    if save_button is None:
        raise RuntimeError("metadata dialog Save button is unavailable")
    capture.save("05_metadata", {
        "metadata_dialog": metadata_dialog,
        "preview": metadata_dialog._preview,
        "save": save_button,
    }, overlays=(metadata_dialog,))
    save_button.click()
    wait_until(app, lambda: not metadata_dialog.isVisible(), 10.0,
               "metadata dialog save")

    # Re-apply the resolved settings after the drop report. Demonstrate the
    # progressive settings controls before opening any individual category.
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
    window._on_nav_selected("mask")
    wait_until(app, lambda: window._stack.currentWidget() is screen, 10.0,
               "return to Mask")
    search.set_modified_only(False)
    search.set_level("all")
    settle(app, 20)

    # Click every current Mask category once. Only the beginning of very long
    # categories is shown; the point is the workflow grouping, not a recital
    # of every parameter.
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
    if len(captured_categories) < 10:
        raise RuntimeError(
            f"only {len(captured_categories)} Mask categories were visible"
        )

    for section in screen._settings_sections:
        section.set_expanded(False)

    # Open the actual live preview and load the ER-only field used for cell
    # segmentation. The full application remains 4K; only the runtime console
    # and System card collapse so the paired microscopy canvases can expand.
    # The live canvases are the subject from this point onward. Collapse the
    # runtime console and System telemetry card so the splitter can give the
    # synchronized images enough vertical space for a useful 4K close-up.
    screen._console_wrap.hide()
    system_card = card_named(screen, "System")
    if system_card is not None:
        system_card.hide()
    screen._preview_switch.setChecked(True)
    screen._runtime_splitter.setSizes([1000, 0])
    panel.load_image(channel_files["cell"])
    apply_object_settings(panel, settings, "cell")
    # Preserve preprocessing from the imported experiment settings. The
    # tutorial teaches viewers to judge normalization from their own preview;
    # it must not prescribe percentiles that happened to suit this dataset.
    set_widget(panel._normalise_check, settings.get("normalize"))
    set_widget(panel._lo_pct, settings.get("lower_percentile"))
    set_widget(panel._hi_pct, settings.get("upper_percentile"))
    panel._compartment_widgets["cell"]["min_area"].setValue(0)
    settle(app, 40)
    capture.save("25_cell_ready", {
        "live": screen._preview_switch,
        "run_preview": panel._run_btn,
        "canvases": [panel._src_view, panel._mask_view],
    })

    dialog, segmentation = open_settings(app, panel)
    dialog._propagate_btn.setChecked(True)
    settle(app, 20)
    capture.save("26_live_settings", {
        "live_settings": panel._live_settings_btn,
        "segmentation": segmentation,
        "core_controls": [panel._object_box, panel._cell_channel,
                          panel._diameter, panel._flow, panel._prob],
        "run_preview": dialog._run_btn,
        "propagate": dialog._propagate_btn,
    }, overlays=(dialog,))
    close_settings(app, dialog)

    run_preview(app, panel, "cell preview")
    zoom_views(app, panel)
    capture.save("27_cell_outlines", {
        "run_preview": panel._run_btn,
        "view_mode": panel._view_mode,
        "canvases": [panel._src_view, panel._mask_view],
        "action_and_canvases": [panel._run_btn, panel._src_view,
                                panel._mask_view],
    })
    set_combo(panel._view_mode, "Flows")
    settle(app, 30)
    capture.save("28_cell_flows", {
        "view_mode": panel._view_mode,
        "canvases": [panel._src_view, panel._mask_view],
        "action_and_canvases": [panel._view_mode, panel._src_view,
                                panel._mask_view],
    })
    set_combo(panel._view_mode, "Masks")
    settle(app, 30)
    capture.save("29_cell_objects", {
        "view_mode": panel._view_mode,
        "canvases": [panel._src_view, panel._mask_view],
        "action_and_canvases": [panel._view_mode, panel._src_view,
                                panel._mask_view],
    })

    set_combo(panel._view_mode, "Overlay")
    unfiltered_count = object_count(panel._masks["cell"])
    min_area_value, min_area_selection = conservative_min_area(
        panel._masks["cell"])
    dialog, segmentation = open_settings(app, panel)
    dialog._propagate_btn.setChecked(True)
    min_area = panel._compartment_widgets["cell"]["min_area"]
    min_area.setValue(min_area_value)
    panel._recompute_masks()
    settle(app, 25)
    filtered_count = object_count(panel._masks["cell"])
    removed_fraction = (unfiltered_count - filtered_count) / unfiltered_count
    if not (0 < removed_fraction <= 0.12):
        raise RuntimeError(
            "minimum-area tutorial filter removed an unsafe fraction of "
            f"objects: {unfiltered_count} -> {filtered_count}"
        )
    propagated = screen._settings_model.collect()
    if int(propagated.get("cell_min_area", -1)) != min_area_value:
        raise RuntimeError("Live Preview minimum area was not propagated")
    if int(propagated.get("cell_channel", -1)) != int(
            settings.get("cell_channel", 0)):
        raise RuntimeError("Live Preview changed the full-run cell channel")
    capture.save("30_cell_size_filter", {
        "min_area": min_area,
        "cell_filters": dialog._compartment_groupboxes.get("cell"),
        "propagate": dialog._propagate_btn,
    }, overlays=(dialog,))
    close_settings(app, dialog)
    zoom_views(app, panel)
    capture.save("31_cell_filtered", {
        "canvases": [panel._src_view, panel._mask_view],
    })

    capture.write_geometry()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "settings_source": str(settings_path),
        "tutorial_overrides": {
            "src": str(orig),
            "test_mode": True,
            "test_images": 5,
            "random_test": True,
            "normalization": "preserved from imported experiment settings",
            "cell_min_area_initial": 0,
            "cell_min_area_filtered": min_area_value,
            "delete_intermediate": False,
            "keep_intermediate": True,
        },
        "representative_field": args.field,
        "channel_mapping": {
            "0": "nuclei",
            "1": "ER / cells / channel of interest",
            "2": "lipid droplets / organelles",
            "3": "parasites",
        },
        "live_preview_diameters": PREVIEW_DIAMETERS,
        "images": {key: str(value) for key, value in channel_files.items()},
        "settings_categories": captured_categories,
        "cell_object_counts": {
            "before_min_area_filter": unfiltered_count,
            "after_min_area_filter": filtered_count,
            "removed_fraction": removed_fraction,
        },
        "cell_min_area_selection": min_area_selection,
        "captures": sorted(capture.frames),
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    panel.shutdown()
    window.close()
    settle(app, 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
