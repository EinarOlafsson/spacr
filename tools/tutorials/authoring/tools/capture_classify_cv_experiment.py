#!/usr/bin/env python3
"""Capture Classify (CV) from spaCR with real c1/c2 control crops."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
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
OUTPUTS = {
    "classify": ROOT / "production" / "10_classify_cv" / "keyframes",
    "ml_analyze": ROOT / "production" / "11_classify_ml" / "keyframes",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--app-key", choices=("classify", "ml_analyze"), default="classify",
        help="Capture either the image classifier or the classical ML module.",
    )
    parser.add_argument(
        "--run-search", action="store_true",
        help="Run a bounded two-configuration grouped search for the release capture.",
    )
    args = parser.parse_args()
    app_key = args.app_key
    output = OUTPUTS[app_key]
    dataset = args.dataset.resolve()
    db_path = dataset / "measurements" / "measurements.db"
    if not db_path.is_file() or not (dataset / "data").is_dir():
        raise FileNotFoundError("Classify needs the measured tutorial project")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    control_counts = dict(con.execute(
        "SELECT rowID, COUNT(*) FROM png_list "
        "GROUP BY rowID ORDER BY rowID"
    ).fetchall())
    total_crops = con.execute("SELECT COUNT(*) FROM png_list").fetchone()[0]
    con.close()
    control_values = [name for name, count in control_counts.items() if count][:2]
    if len(control_values) != 2:
        raise RuntimeError("the tutorial project needs two populated plate columns")

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", f"/tmp/spacr-tutorial-{app_key}-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/spacr-tutorial-{app_key}-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QTabWidget

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.dnd_handlers import ClassifyDropHandler, MeasurementsDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    classify_settings = {
        "src": [str(dataset)],
        "experiment": "spaCR_Tutorial",
        "generate_training_dataset": True,
        "train": True,
        "test": True,
        "dataset_mode": "metadata",
        "classes": ["nc", "pc"],
        "class_metadata": [[control_values[0]], [control_values[1]]],
        "metadata_type_by": "rowID",
        "png_type": "cell_png",
        "crop_source": "auto",
        "size": 224,
        "test_split": 0.1,
        "balance_to_smallest": True,
        "model_type": "resnet18",
        "train_channels": ["r", "g", "b"],
        "image_size": 224,
        "normalize": True,
        "epochs": 10,
        "batch_size": 64,
        "optimizer_type": "adamw",
        "schedule": "cosine",
        "loss_type": "auto",
        "cross_validation_enabled": True,
        "cross_validation_folds": 5,
        "cv_group_by": "well",
        "classifier_evaluation": True,
        "evaluation_calibration": "temperature",
        "evaluation_bins": 10,
        "evaluation_fail_on_leakage": True,
        "generate_full_dataset": True,
        "apply_model_to_dataset": True,
        "n_top_examples": 20,
        "plot": True,
        "tensorboard": True,
        "random_seed": 42,
        "n_jobs": 8,
        "verbose": True,
    }
    ml_settings = {
        "src": str(dataset),
        "location_column": "columnID",
        "negative_control": "c1",
        "positive_control": "c2",
        "annotation_column": None,
        "channel_of_interest": 1,
        "remove_highly_correlated_features": True,
        "remove_low_variance_features": True,
        "minimum_cell_count": 25,
        "batch_correction": "control_center",
        "batch_column": "plateID",
        "batch_control_column": "columnID",
        "batch_control_values": ["c1"],
        "batch_min_samples": 3,
        "batch_missing_control": "error",
        "model_type_ml": "xgboost",
        "n_estimators": 500,
        "learning_rate": 0.01,
        "test_size": 0.2,
        "cross_validation": True,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "prune_features": True,
        "top_features": 30,
        "n_repeats": 10,
        "save_to_db": True,
        "cmap": "viridis",
        "heatmap_feature": "predictions",
        "grouping": "mean",
        "min_max": "allq",
        "verbose": True,
        "n_jobs": -1,
    }
    settings = classify_settings if app_key == "classify" else ml_settings

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")
    window._on_nav_selected(app_key)
    settle(app, 90)
    screen = window._screens.get(app_key)
    if screen is None:
        raise RuntimeError("Classify screen did not open")
    screen.apply_settings_dict(settings)
    wait_until(
        app,
        lambda: screen._settings_search is not None
        and screen._recipe_button is not None
        and screen._hyperparam is not None,
        15.0,
        "Classify controls",
    )
    settle(app, 30)
    capture = CaptureSession(app, window, output)

    capture.save("01_console", {
        "nav": nav_button(window, app_key),
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
    capture.save("03_ai_providers", {"dialog": providers},
                 overlays=(providers,))
    tabs.setCurrentIndex(1)
    settle(app, 20)
    capture.save("04_ai_settings", {"dialog": providers},
                 overlays=(providers,))
    providers.close()

    # Use the real drop path, then restore the complete tutorial mapping. The
    # first two plate columns are c1=NC and c2=PC throughout this lesson.
    if app_key == "classify":
        ClassifyDropHandler().apply(dataset, screen)
    else:
        MeasurementsDropHandler().apply(dataset, screen)
    screen.apply_settings_dict(settings)
    settle(app, 30)
    capture.save("05_folder", {
        "src": screen._settings_model._widgets.get("src"),
        "console": screen._console._console_box,
        "settings": screen._settings_scroll,
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
        and window._stack.currentWidget()
        is window._screens.get("distributed_jobs"),
        15.0,
        "Distributed Jobs screen",
    )
    remote = window._screens["distributed_jobs"]
    settle(app, 30)
    capture.save("11_remote_jobs", {"remote_screen": remote})
    window._on_nav_selected(app_key)
    wait_until(app, lambda: window._stack.currentWidget() is screen, 10.0,
               "return to Classify")
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
    expected_categories = 6 if app_key == "classify" else 8
    if len(captured_categories) != expected_categories:
        raise RuntimeError(
            f"expected {expected_categories} {app_key} categories, "
            f"found {captured_categories}"
        )

    # Return specifically to the label mapping: this is the biological choice
    # the user asked the tutorial to make explicit.
    # Both current classifiers expose the ground-truth mapping under a
    # Labels category (older ML builds called it Data & Controls).
    control_title = "labels"
    labels_section = next(
        item for item in screen._settings_sections
        if control_title in item.title().lower()
    )
    for section in screen._settings_sections:
        section.set_expanded(section is labels_section)
    screen._settings_scroll.ensureWidgetVisible(labels_section.header(), 12, 12)
    settle(app, 25)
    time.sleep(0.4)
    settle(app, 25)
    capture.save("22_control_mapping", {
        "category": labels_section,
        "classes": screen._settings_model._widgets.get(
            "classes" if app_key == "classify" else "negative_control"
        ),
        "metadata": screen._settings_model._widgets.get(
            "class_metadata" if app_key == "classify" else "positive_control"
        ),
        "metadata_column": screen._settings_model._widgets.get(
            "metadata_type_by" if app_key == "classify" else "location_column"
        ),
    })

    # Hyperparameter search evaluates grouped held-out folds.  The release
    # capture runs two deliberately tiny real configurations: enough to show
    # a completed ranked result without turning documentation capture into a
    # broad tuning exercise.
    for section in screen._settings_sections:
        section.set_expanded(False)
    system = card_named(screen, "System")
    if system is not None:
        system.hide()
    screen._hp_switch.setChecked(True)
    wait_until(app, screen._hyperparam_card.isVisible, 5.0,
               "hyperparameter search card")
    screen._runtime_splitter.setSizes([1150, 420])
    panel = screen._hyperparam
    if app_key == "classify":
        search_src = None
        if args.run_search:
            import tempfile
            from spacr.io import generate_dataset_from_lists
            from spacr.utils import correct_paths
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as search_db:
                class_paths = []
                for value in control_values:
                    paths = [row[0] for row in search_db.execute(
                        "SELECT png_path FROM png_list WHERE rowID = ? "
                        "ORDER BY plateID, columnID, fieldID, png_path LIMIT 400",
                        (value,),
                    ).fetchall()]
                    class_paths.append(correct_paths(paths, str(dataset)))
            search_dataset = tempfile.mkdtemp(
                prefix="spacr-tutorial-classify-search-")
            train_root, _test_root = generate_dataset_from_lists(
                search_dataset,
                class_data=class_paths,
                classes=["nc", "pc"],
                test_split=0.25,
                db_path=str(db_path),
                random_seed=42,
                # Each tutorial class occupies one well, so a well-grouped
                # holdout would necessarily empty one class. Fields remain
                # independent acquisition groups and provide a leakage-safe,
                # genuinely disjoint bounded search fixture.
                group_by="field",
            )
            search_src = str(Path(train_root).parent)
        def _capture_search_settings():
            current = dict(screen._settings_model.collect())
            if search_src:
                current["src"] = search_src
                current["generate_training_dataset"] = False
                current["generate_full_dataset"] = False
                current["apply_model_to_dataset"] = False
                current["plot"] = False
                current["init_weights"] = False
                current["batch_size"] = 16
                current["cv_group_by"] = "field"
            return current
        panel.set_settings_provider(_capture_search_settings)
    panel._settings_panel.setChecked(True)
    panel._mode.setCurrentText("random")
    panel._n_trials.setValue(2)
    panel._n_folds.setValue(2)
    panel._seed.setValue(42)
    if app_key == "classify":
        panel._value_edits["learning_rate"].setText("0.0003, 0.001")
        panel._value_edits["dropout_rate"].setText("0.1")
        panel._value_edits["epochs"].setText("1")
        panel._value_edits["weight_decay"].setText("0.0001")
    else:
        panel._value_edits["learning_rate"].setText("0.01, 0.1")
        panel._value_edits["n_estimators"].setText("40")
        panel._value_edits["reg_alpha"].setText("0")
        panel._value_edits["reg_lambda"].setText("1")
    if args.run_search:
        if not panel.run_search():
            raise RuntimeError(
                f"search did not start: {panel._status.text()}")
        wait_until(
            app,
            lambda: panel._worker is None,
            1200.0,
            f"{app_key} bounded hyperparameter search",
        )
        if panel.result is None or not panel.result.ranked():
            failures = [] if panel.result is None else [
                trial.error for trial in panel.result.failed]
            raise RuntimeError(
                f"search did not produce ranked trials: {panel._status.text()} "
                f"failures={failures}")
    settle(app, 30)
    capture.save("23_hyperparameter_search", {
        "hyperparameter_toggle": screen._hp_switch,
        "search": panel._settings_panel,
        "run_search": panel._run_btn,
        "folds": panel._n_folds,
        "results": panel._table,
    })

    panel._settings_btn.click()
    wait_until(
        app,
        lambda: panel._settings_dialog is not None
        and panel._settings_dialog.isVisible(),
        5.0,
        "hyperparameter search settings",
    )
    search_dialog = panel._settings_dialog
    center_dialog(window, search_dialog, 1500, 1420)
    settle(app, 30)
    capture.save("24_search_settings", {
        "settings_button": panel._settings_btn,
        "dialog": search_dialog,
        "search_space": panel._settings_panel,
        "folds": panel._n_folds,
        "criterion": panel._criterion,
    }, overlays=(search_dialog,))
    search_dialog.close()
    wait_until(app, lambda: panel._settings_dialog is None, 5.0,
               "search settings close")

    capture.save("25_run_pipeline", {
        "run": screen._btn_run,
        "stop": screen._btn_stop,
        "actions": screen._actions_row,
        "console": screen._console._console_box,
    })
    capture.write_geometry()

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "database": str(db_path),
        "total_crops": total_crops,
        "app_key": app_key,
        "control_mapping": {
            control_values[0]: "negative control",
            control_values[1]: "positive control",
        },
        "control_counts": control_counts,
        "settings": settings,
        "settings_categories": captured_categories,
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "training_started": bool(args.run_search),
        "hyperparameter_trials": (
            len(panel.result.ranked())
            if args.run_search and panel.result is not None else 0
        ),
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
