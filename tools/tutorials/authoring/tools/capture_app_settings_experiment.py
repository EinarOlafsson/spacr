#!/usr/bin/env python3
"""Reusable native-4K capture harness for spaCR settings-driven modules."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from capture_mask_experiment import (
    CaptureSession, center_dialog, nav_button, safe_stem, settle, wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get(
    "SPACR_REPO", "/mnt/firecuda2/codex/repo/spacr"
)).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-key", required=True)
    parser.add_argument("--lesson-id", required=True)
    parser.add_argument("--settings-json", type=Path, required=True)
    parser.add_argument("--expected-categories", type=int, required=True)
    parser.add_argument("--detail-section")
    parser.add_argument("--detail-keys", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-timeout", type=float, default=180.0)
    args = parser.parse_args()
    settings = json.loads(args.settings_json.read_text())
    detail_keys = [key for key in args.detail_keys.split(",") if key]
    output = ROOT / "production" / args.lesson_id / "keyframes"

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", f"/tmp/spacr-tutorial-{args.app_key}-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/spacr-tutorial-{args.app_key}-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QApplication, QTabWidget

    import spacr
    # Release tutorial screenshots describe the current public release while
    # development continues on nightly.
    spacr.__version__ = "1.5.0.4"
    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    if args.app_key == "analyze_plaques" and not settings.get("masks", True):
        # The assay backend downloads the segmentation checkpoint before it
        # checks whether this run reuses existing masks.  This capture uses
        # deterministic precomputed masks, so the unused network download
        # would add latency without changing the real database analysis.
        import spacr.utils as spacr_utils
        spacr_utils.download_models = lambda *unused_args, **unused_kwargs: None
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
    window._on_nav_selected(args.app_key)
    settle(app, 90)
    screen = window._screens.get(args.app_key)
    if screen is None:
        raise RuntimeError(f"{args.app_key} screen did not open")
    screen.apply_settings_dict(settings)
    # A few legacy settings panels (notably Train Cellpose) consume ``src``
    # in their backend but do not render it as a form row.  A real drop keeps
    # that path as part of the run payload; mirror that state for the capture
    # rather than publishing a tutorial whose Run scene fails with KeyError.
    if settings.get("src") and "src" not in screen._settings_model._widgets:
        screen._settings_model._defaults["src"] = settings["src"]
        screen._console.append_notice(
            "[drop] {module} source folder = {src}\n",
            module=args.app_key,
            src=settings["src"],
        )
    wait_until(
        app,
        lambda: screen._settings_search is not None
        and screen._recipe_button is not None,
        15.0,
        f"{args.app_key} controls",
    )
    settle(app, 35)
    capture = CaptureSession(app, window, output)

    capture.save("01_console", {
        "nav": nav_button(window, args.app_key),
        "console": screen._console._console_box,
        "chat": screen._console._chat_row,
        "actions": screen._actions_row,
    })

    screen._ai_menu.popup(screen._ai_menu_btn.mapToGlobal(
        QPoint(0, screen._ai_menu_btn.height())))
    wait_until(app, screen._ai_menu.isVisible, 5.0, "AI provider menu")
    capture.save("02_ai_menu", {
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
    capture.save("03_ai_providers", {"dialog": providers}, overlays=(providers,))
    tabs.setCurrentIndex(1)
    settle(app, 20)
    capture.save("04_ai_settings", {"dialog": providers}, overlays=(providers,))
    providers.close()

    screen.apply_settings_dict(settings)
    settle(app, 25)
    configured_widgets = [
        screen._settings_model._widgets.get(key)
        for key in (
            "src", "paired_data", "score_data", "count_data", "metadata_files",
            "dataset", "model_path",
        )
        if screen._settings_model._widgets.get(key) is not None
    ]
    owning_sections = [
        section for section in screen._settings_sections
        if any(section.isAncestorOf(widget) for widget in configured_widgets)
    ]
    if owning_sections:
        for section in screen._settings_sections:
            section.set_expanded(section in owning_sections)
        screen._settings_scroll.ensureWidgetVisible(
            owning_sections[0].header(), 12, 12
        )
        settle(app, 25)
    capture.save("05_inputs", {
        "inputs": configured_widgets,
        "settings": screen._settings_scroll,
        "console": screen._console._console_box,
    })

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
        "modified": search._modified,
        "settings": screen._settings_scroll,
    })
    search.set_modified_only(False)
    search.set_level("all")
    screen._settings_scroll.verticalScrollBar().setValue(0)
    settle(app, 25)
    capture.save("08_all_settings", {
        "settings_controls": search,
        "all_settings": search._disclosure,
        "settings": screen._settings_scroll,
    })

    from spacr.qt.recipes import RecipeDialog
    recipes = RecipeDialog(screen, parent=window)
    center_dialog(window, recipes, 1500, 1050)
    recipes.show()
    settle(app, 25)
    capture.save("09_recipes", {
        "recipes_button": screen._recipe_button,
        "dialog": recipes,
    }, overlays=(recipes,))
    recipes.close()

    capture.save("10_remote_submit", {
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
    capture.save("11_remote_jobs", {"remote_screen": remote})
    window._on_nav_selected(args.app_key)
    wait_until(app, lambda: window._stack.currentWidget() is screen, 10.0,
               f"return to {args.app_key}")
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
        capture.save(f"12_category_{index:02d}_{safe_stem(title)}", {
            "category": section,
            "category_header": section.header(),
            "category_hint": screen._category_hint,
        })
        captured_categories.append(title)
    if len(captured_categories) != args.expected_categories:
        raise RuntimeError(
            f"expected {args.expected_categories} categories, "
            f"found {captured_categories}"
        )

    if args.detail_section:
        detail_section = next(
            section for section in screen._settings_sections
            if args.detail_section.lower() in section.title().lower()
        )
        for section in screen._settings_sections:
            section.set_expanded(section is detail_section)
        screen._settings_scroll.ensureWidgetVisible(
            detail_section.header(), 12, 12
        )
        settle(app, 25)
        detail_widgets = {
            key: screen._settings_model._widgets.get(key)
            for key in detail_keys
        }
        capture.save("20_detail", {
            "category": detail_section,
            **detail_widgets,
        })

    search_selected_params = {}

    # Timelapse and Motility have a real, bounded preview contract.  Earlier
    # captures narrated that contract while leaving the card closed.  Load the
    # deterministic tutorial sequence, run the preview, and preserve both the
    # inspected input and the resulting track/QC view.
    if args.app_key in {"timelapse", "motility"}:
        screen._preview_switch.setChecked(True)
        wait_until(
            app,
            lambda: getattr(screen, "_preview_primed", False),
            5.0,
            f"{args.app_key} preview card",
        )
        if args.app_key == "timelapse":
            panel = screen._timelapse_preview
            merged = sorted((Path(settings["src"]) / "merged").glob("*.npy"))
            if not merged or not panel.load_sequence(merged[0].parent):
                raise RuntimeError("could not load the tutorial timelapse preview")
            # Use masks derived from the same deterministic tutorial frames.
            # This exercises the real label-loading and track-linking path while
            # keeping documentation capture bounded; running Cellpose here made
            # a five-frame UI proof depend on model-download/GPU timing.
            from scipy import ndimage
            mask_stack = []
            for frame_path in merged[:8]:
                frame = np.asarray(np.load(frame_path, mmap_mode="r"))
                plane = frame[..., int(settings.get("cell_channel", 1))]
                foreground = plane > np.percentile(plane, 82.0)
                foreground = ndimage.binary_opening(foreground, iterations=1)
                labels, _ = ndimage.label(foreground)
                sizes = np.bincount(labels.ravel())
                keep = sizes >= 24
                keep[0] = False
                labels = labels * keep[labels]
                labels, _ = ndimage.label(labels > 0)
                mask_stack.append(labels.astype(np.uint16))
            mask_path = Path(settings["src"]) / "tutorial_preview_masks.npy"
            np.save(mask_path, np.stack(mask_stack))
            if not panel.load_masks(mask_path):
                raise RuntimeError("could not load tutorial-derived preview masks")
        else:
            panel = screen._motility_preview
            if not panel.load_folder(settings["src"]):
                raise RuntimeError("could not load the tutorial motility preview")
        settle(app, 35)
        preview_card = getattr(screen, screen._preview_card_attr)
        capture.save("18_preview_loaded", {
            "preview": preview_card,
            "run_preview": panel._run_btn,
        })
        panel.run_preview()
        attr = "_stats" if args.app_key == "timelapse" else "_summary"
        wait_until(
            app,
            lambda: getattr(panel, attr, None) is not None,
            args.run_timeout,
            f"{args.app_key} preview result",
        )
        settle(app, 35)
        capture.save("19_preview_result", {
            "preview": preview_card,
            "run_preview": panel._run_btn,
        })

    # Hyperparameter Search is a mini workbench, not a hidden advanced
    # setting. Capture it wherever the current module exposes it so the
    # lesson can demonstrate the search space, compact Run search action, and
    # result preview directly.
    if getattr(screen, "_hp_switch", None) is not None:
        screen._hp_switch.setChecked(True)
        wait_until(
            app,
            lambda: screen._hyperparam_card.isVisible(),
            5.0,
            f"{args.app_key} hyperparameter search card",
        )
        panel = screen._hyperparam
        if args.app_key == "umap" and args.run:
            # Four real trials are enough to demonstrate the workbench on the
            # small tutorial table without turning capture into a broad or
            # expensive search.
            panel._value_edits["n_neighbors"].setText("10, 20")
            panel._value_edits["min_dist"].setText("0.05, 0.3")
            panel._value_edits["metric"].setText("euclidean")
            panel._mode.setCurrentText("grid")
            panel._seed.setValue(42)
            panel._compact_run_btn.click()
            wait_until(
                app,
                lambda: panel._compact_stop_btn.isEnabled(),
                15.0,
                "UMAP hyperparameter search start",
            )
            wait_until(
                app,
                lambda: panel._compact_run_btn.isEnabled(),
                args.run_timeout,
                "UMAP hyperparameter search completion",
            )
            if panel.result is None:
                raise RuntimeError(
                    "UMAP hyperparameter search did not return a result")
            search_selected_params = panel.selected_params() or {}
            if not search_selected_params:
                raise RuntimeError(
                    "UMAP hyperparameter search did not select a result")
        if getattr(screen, "_runtime_splitter", None) is not None:
            screen._runtime_splitter.setSizes([1180, 420])
        settle(app, 35)
        capture.save("19_hyperparameter_search", {
            "hyperparameter_toggle": screen._hp_switch,
            "search_card": screen._hyperparam_card,
            "run_search": panel._compact_run_btn,
            "search_preview": panel._preview_stack,
        })
        if args.app_key == "umap" and args.run:
            panel.open_settings()
            wait_until(
                app,
                lambda: panel._settings_dialog is not None
                and panel._settings_dialog.isVisible(),
                5.0,
                "UMAP settings dialog",
            )
            dialog = panel._settings_dialog
            center_dialog(window, dialog, 1800, 1600)
            settle(app, 30)
            capture.save("20_hyperparameter_apply", {
                "dialog": dialog,
                "propagate": dialog._propagate,
            }, overlays=(dialog,))
            dialog._propagate.click()
            settle(app, 20)
            resolved = screen._settings_model.collect()
            for key, value in search_selected_params.items():
                if resolved.get(key) != value:
                    raise RuntimeError(
                        f"UMAP search result {key}={value!r} did not "
                        f"propagate; main form has {resolved.get(key)!r}"
                    )
            dialog.close()
            settle(app, 20)
        screen._hp_switch.setChecked(False)
        settle(app, 20)

    capture.save("21_run", {
        "run": screen._btn_run,
        "stop": screen._btn_stop,
        "actions": screen._actions_row,
        "console": screen._console._console_box,
    })
    if args.run:
        screen._btn_run.click()
        wait_until(
            app,
            lambda: screen._btn_stop.isEnabled(),
            15.0,
            f"{args.app_key} run start",
        )
        settle(app, 35)
        capture.save("22_running", {
            "run": screen._btn_run,
            "stop": screen._btn_stop,
            "console": screen._console._console_box,
        })
        wait_until(app, lambda: screen._btn_run.isEnabled(), args.run_timeout,
                   f"{args.app_key} completion")
        settle(app, 50)
        console_box = screen._console._console_box
        if hasattr(console_box, "toPlainText") and hasattr(console_box, "setPlainText"):
            console_lines = console_box.toPlainText().splitlines()
            if len(console_lines) > 46:
                console_box.setPlainText("\n".join(console_lines[-46:]))
        if hasattr(console_box, "moveCursor"):
            console_box.moveCursor(QTextCursor.End)
        if hasattr(console_box, "verticalScrollBar"):
            scroll = console_box.verticalScrollBar()
            scroll.setValue(scroll.maximum())
        console_scroll = getattr(screen._console, "_scroll", None)
        if console_scroll is not None:
            scroll = console_scroll.verticalScrollBar()
            scroll.setValue(scroll.maximum())
        settle(app, 20)
        if console_scroll is not None:
            current_stdout = getattr(screen._console, "_current_stdout", None)
            if current_stdout is not None:
                console_scroll.ensureWidgetVisible(current_stdout, 0, 0)
                block_scroll = current_stdout.verticalScrollBar()
                block_scroll.setValue(block_scroll.maximum())
            scroll = console_scroll.verticalScrollBar()
            scroll.setValue(scroll.maximum())
            settle(app, 4)
        output_widgets = {
            "console": console_box,
            "actions": screen._actions_row,
        }
        if args.app_key == "umap":
            queue = screen._figure_queue
            wait_until(app, lambda: queue._count >= 1, 20.0,
                       "Image UMAP output figure")
            if queue._count != 1:
                raise RuntimeError(
                    f"expected one Image UMAP glyph map, found {queue._count}"
                )
            queue._list.setCurrentRow(0)
            # Give the embedding the frame. The concise lesson already has a
            # separate running/console scene; this scene must make the crop
            # overlay and the underlying point cloud readable.
            screen._console.hide()
            screen._figures_card.setMinimumHeight(1400)
            settle(app, 100)
            output_widgets["explorer"] = screen._umap_explorer
        if getattr(screen, "_figures_card", None) is not None:
            output_widgets["figures"] = screen._figures_card
        capture.save("23_output", output_widgets)

    capture.write_geometry()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": args.app_key,
        "lesson_id": args.lesson_id,
        "settings_source": str(args.settings_json.resolve()),
        "settings": settings,
        "settings_categories": captured_categories,
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "run_started": bool(args.run),
        "search_selected_params": search_selected_params,
        "resolved_settings": screen._settings_model.collect(),
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
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
