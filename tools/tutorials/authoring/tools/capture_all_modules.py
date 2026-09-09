#!/usr/bin/env python3
"""Capture sharp 4K keyframes and widget geometry for every spaCR module."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get(
    "SPACR_REPO", "/mnt/firecuda2/codex/repo/spacr"))
CATALOG = ROOT / "catalog" / "lessons_en.json"
PRODUCTION = ROOT / "production"
CAPTURE_RELEASE_VERSION = "1.5.0.4"


# Custom workbenches do not use AppScreen's settings scroll/console layout.
# Whole-screen fallbacks made their scene boxes effectively invisible, so map
# the real input, decision, action, and result surfaces in the 3840x2160
# release capture.  These are intentionally broad enough to include labels and
# values, but never include the navigation rail or an unrelated workbench.
SEMANTIC_REGIONS = {
    "batch": {"input": [684, 350, 3132, 500], "settings": [684, 350, 3132, 500],
              "output": [684, 860, 3132, 1140]},
    "distributed_jobs": {"input": [684, 350, 3132, 430], "settings": [684, 350, 3132, 430],
                         "output": [684, 790, 3132, 1210]},
    "classifier_evaluation": {"input": [684, 350, 3132, 380], "settings": [684, 350, 3132, 380],
                              "output": [684, 740, 3132, 1260]},
    "run_history": {"input": [684, 350, 3132, 360], "settings": [684, 350, 3132, 360],
                    "run": [684, 720, 3132, 430], "output": [684, 720, 3132, 1280]},
    "curate": {"input": [684, 350, 3132, 360], "settings": [684, 350, 3132, 500],
               "output": [684, 720, 3132, 1280]},
    "data_manager": {"input": [684, 350, 3132, 350], "settings": [684, 350, 3132, 350],
                     "run": [684, 350, 3132, 350], "output": [684, 710, 3132, 1290]},
    "project_browser": {"input": [684, 350, 3132, 300], "settings": [684, 350, 3132, 300],
                       "run": [684, 650, 3132, 450], "output": [684, 650, 3132, 1350]},
    "napari_bridge": {"input": [684, 350, 3132, 380], "settings": [684, 350, 3132, 380],
                      "output": [684, 740, 3132, 1260]},
    "hit_list": {"input": [684, 350, 3132, 450], "settings": [684, 350, 3132, 450],
                 "output": [684, 810, 3132, 1190]},
    "methods_export": {"input": [684, 350, 3132, 440], "settings": [684, 350, 3132, 440],
                       "output": [684, 800, 3132, 1200]},
    "run_compare": {"input": [684, 350, 3132, 360], "settings": [684, 350, 3132, 360],
                    "output": [684, 720, 3132, 1280]},
    "control_chart": {"input": [684, 350, 800, 1650], "settings": [684, 350, 800, 1650],
                      "output": [1500, 350, 2316, 1650]},
    "pipeline_graph": {"output": [684, 722, 3132, 1346]},
    "profiler": {"output": [684, 350, 3132, 1030]},
    "qc_dashboard": {"output": [684, 782, 3132, 1216]},
    "image_scatter": {"input": [684, 350, 3132, 260], "settings": [3290, 610, 526, 1390],
                      "output": [684, 620, 2586, 1380]},
    "lineage": {"input": [684, 350, 3132, 280], "settings": [684, 350, 3132, 280],
                "output": [684, 640, 3132, 1360]},
    "layer_viewer": {"input": [684, 350, 3132, 260], "settings": [2580, 1500, 1236, 500],
                     "run": [2580, 1500, 1236, 500], "output": [684, 620, 3132, 870]},
    "graph_builder": {"input": [684, 350, 850, 1726], "settings": [684, 350, 850, 1726],
                      "output": [1550, 350, 2266, 1726]},
    "pca": {"input": [684, 350, 1608, 1100], "settings": [684, 350, 1608, 1100],
            "output": [2310, 350, 1506, 1650]},
    "tabulate": {"input": [684, 350, 700, 1650], "settings": [684, 350, 700, 1650],
                 "output": [1400, 350, 2416, 1650]},
    "feature_dict": {"input": [684, 350, 1600, 1650], "settings": [684, 350, 1600, 300],
                     "run": [2300, 350, 1516, 1650], "output": [2300, 350, 1516, 1650]},
    "trellis": {"input": [684, 350, 720, 1650], "settings": [684, 350, 720, 1650],
                "output": [1420, 350, 2396, 1650]},
    "feature_explorer": {"input": [684, 350, 3132, 330], "settings": [684, 350, 3132, 330],
                         "output": [1420, 690, 2396, 1310]},
    "outliers": {"input": [2580, 350, 1236, 1500], "settings": [2580, 350, 1236, 1500],
                 "output": [684, 350, 1880, 1650]},
    "experiment_design": {"input": [684, 350, 3132, 650], "settings": [684, 350, 3132, 650],
                          "output": [684, 1020, 3132, 980]},
    "power": {"input": [684, 350, 834, 1650], "settings": [684, 350, 834, 1650],
              "run": [684, 350, 834, 1650], "output": [1534, 350, 2282, 1650]},
    "dose_response": {"input": [684, 350, 3132, 430], "settings": [684, 350, 3132, 430],
                      "output": [684, 800, 3132, 1200]},
}


def rect_for(widget, window, scale: float, frame_size: tuple[int, int]):
    if widget is None or not widget.isVisible():
        return None
    from PySide6.QtCore import QPoint

    origin = widget.mapTo(window, QPoint(0, 0))
    x = max(0, int(round(origin.x() * scale)))
    y = max(0, int(round(origin.y() * scale)))
    width = min(frame_size[0] - x, int(round(widget.width() * scale)))
    height = min(frame_size[1] - y, int(round(widget.height() * scale)))
    if width <= 0 or height <= 0:
        return None
    return [x, y, width, height]


def nav_button(window, key: str):
    from PySide6.QtWidgets import QPushButton

    for button in window._sidebar.findChildren(QPushButton):
        if button.property("navKey") == key:
            return button
    return None


def first_primary_button(screen):
    from PySide6.QtWidgets import QPushButton

    preferred = (
        "run", "preview", "load", "open", "refresh", "analyze",
        "compare", "generate", "build", "submit", "export",
    )
    buttons = [button for button in screen.findChildren(QPushButton)
               if button.isVisible() and button.isEnabled()]
    for word in preferred:
        for button in buttons:
            if word in button.text().strip().lower():
                return button
    return next((button for button in buttons
                 if button.objectName() == "PrimaryButton"), None)


def first_scroll_area(screen):
    from PySide6.QtWidgets import QScrollArea

    candidates = [item for item in screen.findChildren(QScrollArea)
                  if item.isVisible()]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.mapTo(screen, item.rect().topLeft()).x())


def configure_mask(screen) -> None:
    settings_path = ROOT / "LIVE_IMAGE_SETTINGS.json"
    if not settings_path.exists() or not hasattr(screen, "apply_settings_dict"):
        return
    spec = json.loads(settings_path.read_text())
    settings = dict(spec["mask_settings"])
    screen.apply_settings_dict(settings)


def configure_gate_editor(screen) -> None:
    """Show the current editor doing its real job, without project data.

    An empty scatter cannot teach gating, and the optional assistant console
    consumes horizontal room that this short walkthrough never discusses.
    The deterministic two-population frame keeps the capture useful without
    prescribing any project-specific thresholds.
    """
    import numpy as np
    import pandas as pd

    from spacr.qt.widgets.gate_spec import GateSet, RectGate

    rng = np.random.default_rng(1500)
    first = rng.normal((0.0, 0.0), (0.72, 0.58), size=(420, 2))
    second = rng.normal((2.6, 2.1), (0.62, 0.68), size=(260, 2))
    points = np.vstack((first, second))
    frame = pd.DataFrame({
        "cell area": 850.0 + points[:, 0] * 140.0,
        "mean intensity": 180.0 + points[:, 1] * 32.0,
    })
    screen.set_frame(frame, label="Tutorial data")
    screen._x.setCurrentText("cell area")
    screen._y.setCurrentText("mean intensity")
    gate = RectGate(
        name="bright enlarged cells",
        x_column="cell area",
        y_column="mean intensity",
        x_low=1030.0,
        x_high=1400.0,
        y_low=215.0,
        y_high=285.0,
    )
    screen.gates.set_gates(GateSet().add(gate))
    # Keep the lesson on the core load → choose axes → draw → inspect path.
    # The narrower secondary actions and formula/filter sidebar have their own
    # workflows and otherwise squeeze the current module header and gate tree.
    for name in ("_save_filters", "_load_filters", "_annotate",
                 "_export", "_save_graph"):
        button = getattr(screen, name, None)
        if button is not None:
            button.hide()
    from PySide6.QtWidgets import QScrollArea
    for panel in screen.findChildren(QScrollArea):
        panel.hide()
    screen.console.hide()


def settle(app, cycles: int = 45) -> None:
    for _ in range(cycles):
        app.processEvents()


def _opened_fold_screen(host, key: str):
    """Return the real page opened by ``key`` on ``host``, when available."""
    for opener in getattr(host, "_fold_openers", ()):
        if getattr(opener, "key", None) == key:
            opened = getattr(opener, "window", None)
            nested = getattr(opened, "screen", None)
            return nested if nested is not None and not callable(nested) else opened
    pages = getattr(host, "_fold_pages", None)
    if pages is not None and pages.currentIndex() > 0:
        opened = pages.currentWidget()
        nested = getattr(opened, "screen", None)
        return nested if nested is not None and not callable(nested) else opened
    return None


def _reset_host_view(host) -> None:
    """Restore the host page before selecting a folded tutorial target."""
    pages = getattr(host, "_fold_pages", None)
    if pages is not None and hasattr(pages, "count"):
        while pages.count() > 1:
            index = pages.count() - 1
            page = pages.widget(index)
            pages.removeTab(index)
            if page is not None:
                page.setParent(None)
    if pages is not None and hasattr(pages, "setCurrentIndex"):
        pages.setCurrentIndex(0)
    strip = (getattr(host, "_fold_strip", None)
             or getattr(host, "_folds", None))
    for button in getattr(strip, "buttons", ()):
        if button.isCheckable() and button.isChecked():
            button.setChecked(False)
    if getattr(host, "_sweep_card", None) is not None:
        host._on_sweep_switch(False)


def open_tutorial_target(window, lesson: dict, app):
    """Open a live module, then select the lesson's consolidated workflow.

    ``app_key`` identifies the tutorial capability and remains stable in URLs
    and media directories. ``host_app_key`` is the navigation target. A fold
    button, family selector, or host toggle is then used where the capability
    is not the host's default page.
    """
    key = lesson["app_key"]
    host_key = lesson.get("host_app_key") or key
    window._on_nav_selected(host_key)
    settle(app, 55)
    host = (window._startup if host_key == "__home__"
            else window._screens.get(host_key))
    if host is None:
        raise RuntimeError(f"host screen {host_key!r} did not open")
    _reset_host_view(host)
    settle(app, 20)
    if not lesson.get("host_app_key"):
        return host

    if key in {"classify", "ml_analyze"}:
        family = "cv" if key == "classify" else "ml"
        host.apply_settings_dict({"classifier_family": family})
        settle(app, 35)
        return host
    if key == "parameter_sweep":
        host._on_sweep_switch(True)
        settle(app, 35)
        return getattr(host, "_sweep", None) or host
    if key in {"train_cellpose", "cellpose_masks"}:
        panel = host.open_folded("train_cellpose")
        workbench = host.folded_screen("train_cellpose")
        if workbench is None:
            raise RuntimeError("Cellpose Workbench did not open")
        from PySide6.QtWidgets import QTabWidget
        target = (workbench.train_screen if key == "train_cellpose"
                  else workbench.apply_screen)
        tabs = workbench.findChild(QTabWidget)
        if tabs is not None:
            tabs.setCurrentWidget(target)
        settle(app, 35)
        return target if panel is not None else workbench

    strip = (getattr(host, "_fold_strip", None)
             or getattr(host, "_folds", None))
    button = strip.button_for(key) if strip is not None else None
    if button is None:
        raise RuntimeError(
            f"folded workflow {key!r} has no button on {host_key!r}")
    button.click()
    settle(app, 55)
    if key == "hit_list":
        panel = getattr(host, "_results_panel", None)
        hits = getattr(panel, "hits", None)
        if hits is None:
            raise RuntimeError(
                "Regression did not expose its current Hits results tab"
            )
        # ``button.click()`` exercises the public route, but an offscreen Qt
        # backend can defer one of the nested tab changes until after the
        # next paint. State the two intended selections directly as well, so
        # the full-window keyframe cannot photograph Regression's Settings
        # page while returning the correct hidden Hits widget as geometry.
        figures_card = getattr(host, "_figures_card", None)
        if figures_card is not None and hasattr(figures_card, "show"):
            figures_card.show()
        raise_results = getattr(host, "_raise_the_results_tab", None)
        if callable(raise_results):
            raise_results()
        tabs = getattr(panel, "tabs", None)
        if tabs is not None and hasattr(tabs, "setCurrentWidget"):
            tabs.setCurrentWidget(hits)
        settle(app, 55)
        if hasattr(hits, "isVisible") and not hits.isVisible():
            raise RuntimeError("Regression's current Hits tab is not visible")
        return hits
    # Timelapse is a checkable category fold and therefore remains on Mask.
    if button.isCheckable():
        return host
    opened = _opened_fold_screen(host, key)
    if opened is None and hasattr(host, "folded_screen"):
        opened = host.folded_screen(key)
    return opened or host


def finish_capture_batch(failures: list[dict]) -> Path | None:
    """Write current failures, or clear an older report after full success.

    :returns: the failure-report path when ``failures`` is non-empty;
        otherwise ``None``.
    """
    failure_path = ROOT / "catalog" / "capture_failures.json"
    if failures:
        failure_path.write_text(json.dumps(failures, indent=2) + "\n")
        return failure_path
    # A report describes only the most recent complete batch. It is safe to
    # clear here because every requested capture has already been written.
    failure_path.unlink(missing_ok=True)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Capture native pixels, matching the experiment-backed lesson harnesses.
    # A 1920x1080 logical window at scale 2 produced a nominally 4K image but
    # made Home's controls twice as large as every current tutorial.
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication

    import spacr
    # Tutorial captures describe the published release even when the editable
    # nightly checkout is ahead of it. The package itself is not modified.
    spacr.__version__ = CAPTURE_RELEASE_VERSION
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    # A CPU-only tutorial capture must not poll another process's GPU. Home
    # normally samples NVML on construction and module usage bars normally
    # invoke nvidia-smi in a worker. Replace both telemetry seams before the
    # MainWindow exists; the captures label no GPU result as expected.
    import spacr.qt.widgets.home as home_widgets
    home_widgets._NVML = None
    home_widgets.SystemPanel.gpu_util = staticmethod(lambda: "CPU capture")
    home_widgets.SystemPanel.gpu_vram = staticmethod(lambda: "not sampled")
    import spacr.qt.screens.app_screen as app_screen_module
    app_screen_module._nvidia_smi_available = lambda: False
    from spacr.qt.preferences import apply_preferences_to_app

    catalog = json.loads(CATALOG.read_text())
    targets = [item for item in catalog["lessons"] if item.get("app_key")]
    if args.only:
        wanted = set(args.only)
        targets = [item for item in targets
                   if item["id"] in wanted or item["app_key"] in wanted]
        if wanted & {"05_home", "__home__"}:
            targets.insert(0, {"id": "05_home", "app_key": "__home__"})

    # A production keyframe must show the module itself, not the first-open
    # coach mark. The walkthrough remains available in the real application;
    # this only marks it seen inside the isolated capture configuration.
    from spacr.qt.walkthrough import mark_seen
    for item in targets:
        mark_seen(item.get("host_app_key") or item["app_key"])
    # The Home lesson opens Mask for its final scene even though Mask is not a
    # capture target in that run.
    mark_seen("mask")

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError(
            f"capture device pixel ratio is {window.devicePixelRatioF()}, "
            "expected native DPR 1"
        )

    failures = []
    for index, item in enumerate(targets, start=1):
        key = item["app_key"]
        host_key = item.get("host_app_key") or key
        output_dir = PRODUCTION / item["id"] / "keyframes"
        image_path = output_dir / "01_module.png"
        geometry_path = output_dir / "geometry.json"
        if image_path.exists() and geometry_path.exists() and not args.force:
            print(f"{index:02d}/{len(targets)} skip {item['id']}", flush=True)
            continue
        try:
            screen = open_tutorial_target(window, item, app)
            if key == "__home__":
                runner = getattr(screen, "_journal_jobs", None)
                if runner is not None:
                    runner.shutdown(timeout_ms=10_000)
                screen._queued.hide()
                # The shared development run journal can change while the
                # capture is being made and may contain unrelated failures.
                # Hide those two history-derived panels; the deterministic
                # System, News, and Module State panels still demonstrate the
                # status aside without presenting stale red badges as product
                # state.
                screen._recent.hide()
                screen._totals.hide()
                screen._system.refresh()
                from PySide6.QtCore import QCoreApplication, QEvent
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                settle(app, 10)
            if key == "mask":
                configure_mask(screen)
                settle(app, 45)
            if key == "gate_editor":
                configure_gate_editor(screen)
                settle(app, 70)

            pixmap = window.grab()
            scale = float(pixmap.devicePixelRatio())
            frame_size = (pixmap.width(), pixmap.height())
            if frame_size != (3840, 2160):
                raise RuntimeError(
                    f"capture is not native 4K: "
                    f"{pixmap.width()}x{pixmap.height()}"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            if not pixmap.save(str(image_path), "PNG"):
                raise RuntimeError(f"could not save {image_path}")

            settings = getattr(screen, "_settings_scroll", None) or first_scroll_area(screen)
            input_widget = getattr(screen, "_empty_state_card", None) or settings or screen
            output_widget = getattr(screen, "_console", None) or screen
            run_widget = getattr(screen, "_btn_run", None) or first_primary_button(screen)
            geometry = {
                "app_key": key,
                "host_app_key": item.get("host_app_key"),
                "frame_size": list(frame_size),
                "device_pixel_ratio": scale,
                "overview": rect_for(screen, window, scale, frame_size),
                "nav": rect_for(nav_button(window, host_key), window, scale, frame_size),
                "input": rect_for(input_widget, window, scale, frame_size),
                "settings": rect_for(settings, window, scale, frame_size),
                "run": rect_for(run_widget, window, scale, frame_size),
                "output": rect_for(output_widget, window, scale, frame_size),
            }
            semantic_run_widgets = {
                # These interactive screens update from their central panel;
                # their first primary button is only "Load table" and is not
                # the action described by the run scene.
                "graph_builder": getattr(getattr(screen, "builder", None),
                                           "canvas", None),
                "tabulate": getattr(screen, "pivot", None),
                "trellis": getattr(screen, "panel", None),
                "feature_explorer": getattr(screen, "explorer", None),
                # These two do expose a real analysis action after loading.
                "outliers": getattr(screen, "scan_button", None),
                "dose_response": getattr(screen, "fit_button", None),
            }
            semantic_run = semantic_run_widgets.get(key)
            if semantic_run is not None:
                geometry["run"] = rect_for(
                    semantic_run, window, scale, frame_size)
            if key == "gate_editor":
                # Gate Editor has no Run button: drawing on the canvas applies
                # a gate immediately. Keep each emphasis box on the controls
                # named by the matching narration instead of letting the
                # generic button heuristic land on the assistant's Run.
                overview = geometry["overview"]
                geometry["input"] = [
                    overview[0], overview[1], overview[2],
                    min(410, overview[3]),
                ]
                geometry["settings"] = rect_for(
                    screen.gates, window, scale, frame_size)
                geometry["run"] = rect_for(
                    screen.gates.canvas, window, scale, frame_size)
                geometry["output"] = rect_for(
                    screen.gates.tree, window, scale, frame_size)
            if key == "classify_merged":
                # The merged screen's useful distinction is its top-level
                # family switch.  Highlighting the whole settings scroll area
                # made the narration point at an empty drop zone and controls
                # that are not visible yet.  Keep the overview/input/run/
                # console beats, but put the decision beat precisely around
                # the label and family selector.
                console_rect = rect_for(
                    getattr(screen, "_console", None),
                    window, scale, frame_size,
                )
                panel_right = ((console_rect[0] - 20)
                               if console_rect else frame_size[0])
                family = getattr(screen, "_settings_model", None)
                family = getattr(family, "_widgets", {}).get(
                    "classifier_family")
                family_rect = rect_for(family, window, scale, frame_size)
                if family_rect:
                    x, y, width, height = family_rect
                    left = min(300, x)
                    box_x = x - left
                    geometry["settings"] = [
                        box_x,
                        max(0, y - 20),
                        min(panel_right - box_x, width + left + 20),
                        min(frame_size[1] - max(0, y - 20), height + 40),
                    ]
                input_rect = rect_for(
                    getattr(screen, "_empty_state_card", None),
                    window, scale, frame_size,
                )
                if input_rect:
                    input_rect[2] = min(
                        input_rect[2], panel_right - input_rect[0])
                    geometry["input"] = input_rect
                geometry["run"] = rect_for(
                    getattr(screen, "_btn_run", None),
                    window, scale, frame_size,
                ) or geometry["run"]
                geometry["output"] = console_rect or geometry["output"]
            if not item.get("host_app_key"):
                for role, region in SEMANTIC_REGIONS.get(key, {}).items():
                    geometry[role] = region
            if key == "__home__":
                # Home has teaching regions that are not generic module
                # controls: the category/tile library and status aside.
                geometry["modules"] = rect_for(
                    screen._tabs, window, scale, frame_size)
                geometry["status"] = rect_for(
                    screen._system.parentWidget(), window, scale, frame_size)

                # Show the one canonical performance selector in the real
                # Preferences dialog.  The former Home lesson taught system
                # telemetry but never showed the five-level choice that now
                # governs caching, workers, cleanup, and animation cost.
                from PySide6.QtGui import QColor, QPainter, QPixmap
                from PySide6.QtWidgets import QLabel, QComboBox, QTabWidget
                from spacr.qt.preferences import PreferencesDialog

                preferences = PreferencesDialog(window)
                tabs = preferences.findChild(QTabWidget, "PreferencesTabs")
                level = preferences.findChild(QComboBox, "PerformanceLevel")
                if tabs is None or level is None:
                    raise RuntimeError(
                        "Preferences has no PerformanceLevel selector"
                    )
                for tab_index in range(tabs.count()):
                    if tabs.widget(tab_index).findChild(
                            QComboBox, "PerformanceLevel") is level:
                        tabs.setCurrentIndex(tab_index)
                        break
                else:
                    raise RuntimeError(
                        "PerformanceLevel is not on a Preferences tab"
                    )
                # Capture the dialog at the size a user actually sees. A
                # full-screen 3840-pixel form stretches every field across
                # the monitor and makes its text unreadably small in video.
                # Place the real dialog over the real Home frame instead.
                preferences.resize(1200, 1200)
                preferences.show()
                settle(app, 30)
                dialog_capture = preferences.grab()
                performance = QPixmap(pixmap)
                painter = QPainter(performance)
                painter.fillRect(performance.rect(), QColor(0, 0, 0, 150))
                dialog_x = (frame_size[0] - dialog_capture.width()) // 2
                dialog_y = (frame_size[1] - dialog_capture.height()) // 2
                painter.drawPixmap(dialog_x, dialog_y, dialog_capture)
                painter.end()
                if not performance.save(
                        str(output_dir / "02_performance_4k.png"), "PNG"):
                    raise RuntimeError(
                        "could not save Home Preferences capture"
                    )
                dialog_size = (
                    dialog_capture.width(), dialog_capture.height())
                level_box = rect_for(level, preferences, 1.0, dialog_size)
                note = preferences.findChild(QLabel, "PerformanceLevelNote")
                note_box = rect_for(note, preferences, 1.0, dialog_size)
                if level_box is None or note_box is None:
                    raise RuntimeError(
                        "Performance level selector or explanation is not visible"
                    )
                left = dialog_x + max(
                    0, min(level_box[0], note_box[0]) - 420)
                top = dialog_y + max(
                    0, min(level_box[1], note_box[1]) - 40)
                right = min(
                    dialog_x + dialog_size[0],
                    dialog_x +
                    max(level_box[0] + level_box[2],
                        note_box[0] + note_box[2]) + 40,
                )
                bottom = min(
                    dialog_y + dialog_size[1],
                    dialog_y +
                    max(level_box[1] + level_box[3],
                        note_box[1] + note_box[3]) + 40,
                )
                geometry["performance"] = [
                    left, top, right - left, bottom - top,
                ]
                preferences.close()
                settle(app, 10)
                window._on_nav_selected("mask")
                settle(app, 70)
                opened_screen = window._screens.get("mask")
                if opened_screen is None:
                    raise RuntimeError("Mask screen did not open for Home capture")
                opened_header = getattr(opened_screen, "_header", None)
                if opened_header is None:
                    raise RuntimeError(
                        "Mask screen has no module header for Home emphasis"
                    )
                geometry["open_module"] = rect_for(
                    opened_header, window, scale, frame_size)
                opened = window.grab()
                if (opened.width(), opened.height()) != (3840, 2160):
                    raise RuntimeError(
                        "Home opened-module capture is not native 4K: "
                        f"{opened.width()}x{opened.height()}"
                    )
                if not opened.save(
                        str(output_dir / "03_mask_open_4k.png"), "PNG"):
                    raise RuntimeError("could not save Home opened-module capture")
                window._on_nav_selected("__home__")
                settle(app, 30)
            geometry_path.write_text(json.dumps(geometry, indent=2) + "\n")
            print(
                f"{index:02d}/{len(targets)} {item['id']} "
                f"{pixmap.width()}x{pixmap.height()}",
                flush=True,
            )
        except Exception as error:
            failures.append({"lesson": item["id"], "error": repr(error)})
            print(f"ERROR {item['id']}: {error!r}", flush=True)

    # Several current screens start bounded background scans as soon as they
    # open. Drain them before Qt destroys their QThread wrappers; otherwise a
    # successful headless capture can abort during interpreter teardown.
    from PySide6.QtCore import QThread
    home_runner = getattr(getattr(window, "_startup", None),
                          "_journal_jobs", None)
    if home_runner is not None:
        home_runner.shutdown(timeout_ms=10_000)
    for thread in window.findChildren(QThread):
        if not thread.isRunning():
            continue
        thread.requestInterruption()
        thread.quit()
        thread.wait(5_000)
    window.close()
    settle(app, 5)
    failure_path = finish_capture_batch(failures)
    if failure_path is not None:
        raise RuntimeError(f"{len(failures)} captures failed; see {failure_path}")
    # Some optional Qt screens own parentless helper threads outside the main
    # window tree. All requested files are flushed at this point; exiting
    # directly prevents PySide from destroying a still-retiring wrapper and
    # turning a successful batch into exit code 134.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
