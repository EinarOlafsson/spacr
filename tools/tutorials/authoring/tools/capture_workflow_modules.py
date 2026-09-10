#!/usr/bin/env python3
"""Capture five truthful workflow states for the formerly static lessons.

The first added-module release reused one empty module screenshot for every
scene.  That proved inventory coverage, but it did not show a workflow.  This
capture loads deterministic, in-memory tutorial data where a screen exposes a
public ``set_frame``/``set_frames`` seam, uses the screen's real controls, and
records overview, input, settings, action, and result states separately.
Nothing is written into a user's project and no remote job is submitted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get(
    "SPACR_REPO", "/mnt/firecuda2/codex/repo/spacr"))
CATALOG = ROOT / "catalog" / "lessons_en.json"
PRODUCTION = ROOT / "production"

# Batch and Gate Editor already have dedicated, deeper capture harnesses.
SPECIALIZED = {"37_batch", "64_gate_editor"}


def tutorial_frame(rows: int = 192) -> pd.DataFrame:
    """Return a deterministic table shared by table-oriented workbenches."""
    rng = np.random.default_rng(15004)
    index = np.arange(rows)
    group = np.where(index % 3 == 0, "control", "treated")
    dose = np.take(np.array([0.0, 0.03, 0.1, 0.3, 1.0, 3.0]), index % 6)
    signal = 0.22 + 0.68 / (1.0 + np.exp(-(np.log10(dose + .02) + .5) * 2.2))
    return pd.DataFrame({
        "plateID": np.where(index < rows // 2, "plate1", "plate2"),
        "rowID": np.take(list("ABCDEFGH"), index % 8),
        "columnID": index % 12 + 1,
        "fieldID": index % 4 + 1,
        "condition": group,
        "dose_uM": dose,
        "cell_area": rng.normal(920, 135, rows),
        "nucleus_area": rng.normal(350, 55, rows),
        "mean_intensity": rng.normal(185, 28, rows)
        + np.where(group == "treated", 38, 0),
        "response": np.clip(signal + rng.normal(0, .035, rows), 0, 1),
        "object_count": rng.integers(75, 150, rows),
        "track_id": index // 6,
        "frame": index % 6,
        "parent_track_id": np.where(index // 6 > 0, index // 6 - 1, -1),
    })


def settle(app, cycles: int = 35) -> None:
    for _ in range(cycles):
        app.processEvents()


def visible(widget) -> bool:
    return widget is not None and hasattr(widget, "isVisible") and widget.isVisible()


def rect(widget, window, frame_size=(3840, 2160)):
    if not visible(widget):
        return None
    from PySide6.QtCore import QPoint
    origin = widget.mapTo(window, QPoint(0, 0))
    x, y = max(0, origin.x()), max(0, origin.y())
    return [x, y, min(widget.width(), frame_size[0] - x),
            min(widget.height(), frame_size[1] - y)]


def save(window, path: Path) -> None:
    pixmap = window.grab()
    if (pixmap.width(), pixmap.height()) != (3840, 2160):
        raise RuntimeError(
            f"capture is {pixmap.width()}x{pixmap.height()}, expected 3840x2160")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")


def first_button(screen, words):
    from PySide6.QtWidgets import QPushButton
    buttons = [b for b in screen.findChildren(QPushButton)
               if b.isVisible() and b.isEnabled()]
    for word in words:
        for button in buttons:
            if word in button.text().casefold():
                return button
    return buttons[0] if buttons else None


def populate(screen, key: str, frame: pd.DataFrame) -> None:
    """Use public seams first, then fill only visible tutorial inputs."""
    if key == "explain_cv":
        panel = screen.explain
        panel.database.setText(str(ROOT / "generated" / "explain_cv" /
                                   "measurements.db"))
        panel.predictions.setText(str(ROOT / "generated" / "explain_cv" /
                                      "cv_predictions.csv"))
        panel.output.setText(str(ROOT / "generated" / "explain_cv" /
                                 "surrogate_explanation"))
        panel.prediction_column.clear()
        panel.prediction_column.addItems(["prediction", "confidence"])
        panel.prediction_column.setCurrentText("prediction")
        panel.status.setText(
            "Tutorial inputs ready — plate groups remain intact for held-out validation.")
    elif key == "investigate_hit":
        panel = screen.investigate
        panel.database.setText(str(ROOT / "generated" / "investigate_hit" /
                                   "measurements.db"))
        panel.predictions.setText(str(ROOT / "generated" / "investigate_hit" /
                                      "cell_predictions.csv"))
        panel.fractions.setText(str(ROOT / "generated" / "investigate_hit" /
                                    "guide_fractions.csv"))
        panel.configure_hit(
            folder=str(ROOT / "generated" / "investigate_hit" / "regression"),
            gene="EAF1", effect=1.24, guides=("EAF1_g1", "EAF1_g2"),
            fdr=0.004, phenotype="prediction", guide_agreement=0.91,
            n_guides=2, well_support=8)
        panel.status.setText(
            "Tutorial hit ready — candidate probabilities remain separate from guide identity.")
    elif key == "volcano_explorer":
        style = screen.explorer._style
        style.base_color = "#7F8C8D"
        style.significant_color = "#C44E52"
        style.marker_edge_color = "#202124"
        style.line_color = "#5F6368"
        style.zero_line_color = "#5F6368"
        rng = np.random.default_rng(15005)
        genes = np.array([f"GENE{index:02d}" for index in range(1, 49)])
        effects = rng.normal(0.0, 0.42, len(genes))
        effects[[3, 17, 31]] = (-1.55, 1.28, 1.62)
        adjusted = np.clip(
            np.exp(-np.abs(effects) * 4.2) * rng.uniform(.015, .25, len(genes)),
            1e-7, 1.0)
        results = pd.DataFrame({
            "guide": [f"g{index:03d}" for index in range(1, len(genes) + 1)],
            "gene": genes,
            "standardized_marginal_effect": effects,
            "adjusted_p_value": adjusted,
            "minimum_wells": rng.integers(3, 9, len(genes)),
            "compartment": np.take(
                np.array(["nucleus", "cytoplasm", "pathogen"]),
                np.arange(len(genes)) % 3),
        })
        screen.explorer.set_results(results)
        screen.explorer._push_style_to_controls()
        screen.explorer.refresh()
        screen._path_label.setText("Tutorial regression results — 48 guides")
    elif key == "parameter_sweep":
        screen.destination.setText(str(ROOT / "generated" / "parameter_sweep"))
        screen.max_trials.setValue(24)
        screen.estimate()
    elif hasattr(screen, "set_frame"):
        kwargs = {"label": "Tutorial measurements"}
        try:
            screen.set_frame(frame.copy(), **kwargs)
        except TypeError:
            screen.set_frame(frame.copy())
    elif hasattr(screen, "set_frames"):
        tracks = frame[["track_id", "frame", "parent_track_id",
                        "cell_area", "mean_intensity"]].copy()
        screen.set_frames({"tracks": tracks})

    # AppScreen modules expose their real settings and console rather than a
    # public table seam.  Showing all categories and a bounded readiness note
    # is honest: no analysis is claimed to have run.
    model = getattr(screen, "_settings_model", None)
    if model is not None:
        toggle = getattr(model, "_show_all", None)
        if toggle is not None:
            toggle.setChecked(True)
    console = getattr(screen, "_console", None)
    if console is not None and hasattr(console, "append_stdout"):
        console.append_stdout(
            "Tutorial sample loaded. Review the highlighted settings before Run.\n")

    from PySide6.QtWidgets import QLineEdit
    edits = [e for e in screen.findChildren(QLineEdit)
             if e.isVisible() and e.isEnabled() and not e.text().strip()
             and "search" not in e.placeholderText().casefold()]
    if edits:
        placeholder = edits[0].placeholderText().casefold()
        if "project" in placeholder or "folder" in placeholder:
            edits[0].setText(str(ROOT / "generated" / "import_project"
                                 / "imported_spacr_project"))
        elif "database" in placeholder:
            edits[0].setText(str(ROOT / "generated" / "import_project"
                                 / "imported_spacr_project" / "measurements"
                                 / "measurements.db"))

    status = getattr(screen, "_status", None)
    if status is not None and hasattr(status, "setText"):
        status.setText("Tutorial sample ready — inspect inputs before running.")


def change_decision(screen, key: str) -> None:
    from PySide6.QtWidgets import QComboBox, QScrollArea
    if key == "volcano_explorer":
        control = screen.explorer._controls.get("color_by")
        if control is not None:
            index = control.findData("compartment")
            if index >= 0:
                control.setCurrentIndex(index)
        return
    combos = [c for c in screen.findChildren(QComboBox)
              if c.isVisible() and c.isEnabled() and c.count() > 1]
    if combos:
        combo = combos[-1]
        combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())
        combo.setFocus()
    scrolls = [s for s in screen.findChildren(QScrollArea) if s.isVisible()]
    if scrolls:
        bar = scrolls[0].verticalScrollBar()
        if bar.maximum() > 0:
            bar.setValue(min(bar.maximum(), max(1, bar.maximum() // 3)))


def run_bounded(screen, key: str, app) -> None:
    """Run only in-memory actions known to be bounded and side-effect free."""
    actions = {
        "pca": ("run pca",),
        "outliers": ("scan",),
        "dose_response": ("fit",),
        "image_scatter": ("plot",),
    }
    if key == "explain_cv":
        panel = screen.explain
        held_out = pd.DataFrame({
            "plateID": ["plate2"] * 6,
            "rowID": list("ABCDEF"),
            "columnID": [2, 4, 6, 8, 10, 12],
            "prediction": ["control", "treated"] * 3,
            "surrogate_prediction": ["control", "treated"] * 3,
        })
        result = SimpleNamespace(
            is_faithful=True, fidelity=0.92, baseline=0.55,
            importance=pd.DataFrame({
                "feature": ["mean_intensity", "cell_area", "nucleus_area"],
                "permutation_importance": [0.31, 0.14, 0.06],
            }),
            class_metrics=pd.DataFrame({
                "class": ["control", "treated"], "precision": [0.91, 0.93],
                "recall": [0.94, 0.89],
            }),
            confusion=pd.DataFrame({"actual": ["control", "treated"],
                                    "correct": [34, 33], "incorrect": [2, 3]}),
            shap_values=pd.DataFrame({
                "feature": ["mean_intensity", "cell_area"],
                "mean_abs_shap": [0.28, 0.11],
            }),
            correlated_features=pd.DataFrame({
                "feature": ["mean_intensity"], "correlated_with": ["response"],
                "correlation": [0.81],
            }),
            feature_distributions=pd.DataFrame({
                "feature": ["mean_intensity", "mean_intensity"],
                "class": ["control", "treated"], "median": [181.2, 224.7],
            }),
            held_out=held_out,
            summary=lambda: (
                "Held-out surrogate fidelity: 0.920\n"
                "Majority-class baseline: 0.550\n"
                "Interpretation allowed: fidelity exceeds the baseline.\n"
                "Grouping: plate; leakage exclusions: passed."),
        )
        panel._loaded((result, {
            "summary": "tutorial_surrogate_summary.json",
            "importance": "tutorial_permutation_importance.csv",
        }))
        panel.results.setCurrentWidget(panel.importance)
        settle(app, 40)
        return
    if key == "investigate_hit":
        panel = screen.investigate
        cells = pd.DataFrame({
            "plateID": ["plate1", "plate1", "plate2", "plate2"],
            "rowID": ["B", "D", "C", "F"], "columnID": [3, 7, 5, 9],
            "prediction": [0.94, 0.89, 0.86, 0.82],
            "candidate_rank": [1, 2, 3, 4],
            "hit_like_probability": [0.91, 0.87, 0.82, 0.78],
            "hit_like_uncertainty": [0.04, 0.05, 0.07, 0.09],
            "hit_like_call": [True, True, True, False],
            "attribution_fold": [0, 1, 0, 1],
            "target_guide_fraction": [0.72, 0.68, 0.64, 0.59],
        })
        result = SimpleNamespace(
            target_gene="EAF1", score_column="prediction",
            object_columns=("plateID", "rowID", "columnID"), cells=cells,
            wells=pd.DataFrame({
                "plateID": ["plate1", "plate2"], "independent_wells": [4, 4],
                "mean_hit_probability": [0.88, 0.80],
            }),
            guide_evidence=pd.DataFrame({
                "guide": ["EAF1_g1", "EAF1_g2"], "well_support": [4, 4],
                "effect": [1.31, 1.18], "fdr": [0.004, 0.006],
            }),
            threshold_sensitivity=pd.DataFrame({
                "threshold": [0.70, 0.80, 0.90], "candidate_cells": [19, 11, 3],
            }),
            validation={"cross_fitted": True, "independent_wells": 8,
                        "manual_annotations_overwritten": False},
            summary=lambda: (
                "Selected regression result: EAF1\n"
                "Guide agreement: 0.91 across 8 independent wells\n"
                "Candidate probabilities are cross-fitted and versioned."),
        )
        payload = {
            "result": result, "attribution_run_id": "tutorial-eaf1-v1",
            "embedding": pd.DataFrame({"cell": [1, 2, 3],
                                       "axis_1": [-0.4, 0.1, 0.6],
                                       "axis_2": [0.2, -0.3, 0.4]}),
            "gallery": cells[["plateID", "rowID", "columnID",
                              "hit_like_probability"]].copy(),
        }
        panel._loaded(payload)
        panel.tabs.setCurrentWidget(panel.cell_table)
        settle(app, 40)
        return
    if key == "volcano_explorer":
        control = screen.explorer._controls.get("color_by")
        if control is not None:
            index = control.findData("compartment")
            if index >= 0:
                control.setCurrentIndex(index)
        screen.explorer.refresh()
        settle(app, 60)
        return
    if key == "parameter_sweep":
        results = pd.DataFrame({
            "trial_id": ["trial-001", "trial-002", "trial-003", "trial-004"],
            "status": ["ok", "ok", "rejected", "ok"],
            "regression_type": ["ols", "glm", "glm", "ols"],
            "analysis_unit": ["well"] * 4,
            "multiple_testing_method": ["fdr_bh", "bonferroni", "fdr_bh", "fdr_bh"],
            "n_below_alpha": [7, 3, 0, 6],
            "positive_rank": [1, 2, None, 1],
            "seconds": [48.2, 53.1, 0.0, 46.7],
            "error_type": ["", "", "incompatible settings", ""],
        })
        screen._results = results
        screen._show(results)
        screen.status.setText(
            "4 bounded tutorial trials: 3 succeeded, 1 rejected before running.")
        settle(app, 40)
        return
    button = first_button(screen, actions.get(key, ())) if key in actions else None
    if button is not None:
        button.click()
        settle(app, 90)


def _opened_fold_screen(host, key: str):
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
    """Navigate through the current host instead of a retired Home key."""
    key = lesson["app_key"]
    host_key = lesson.get("host_app_key") or key
    window._on_nav_selected(host_key)
    settle(app, 55)
    host = window._screens.get(host_key)
    if host is None:
        raise RuntimeError(f"host screen {host_key!r} did not open")
    _reset_host_view(host)
    settle(app, 20)
    if not lesson.get("host_app_key"):
        return host
    if key == "parameter_sweep":
        host._on_sweep_switch(True)
        settle(app, 35)
        return getattr(host, "_sweep", None) or host
    strip = (getattr(host, "_fold_strip", None)
             or getattr(host, "_folds", None))
    button = strip.button_for(key) if strip is not None else None
    if button is None:
        raise RuntimeError(
            f"folded workflow {key!r} has no button on {host_key!r}")
    button.click()
    settle(app, 55)
    if button.isCheckable():
        return host
    return _opened_fold_screen(host, key) or host


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-workflows")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-workflows-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication, QScrollArea
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    # Keep the bounded capture fully CPU-only, including telemetry. Home
    # otherwise initialises NVML and visible module pages invoke nvidia-smi.
    import spacr.qt.widgets.home as home_widgets
    home_widgets._NVML = None
    home_widgets.SystemPanel.gpu_util = staticmethod(lambda: "CPU capture")
    home_widgets.SystemPanel.gpu_vram = staticmethod(lambda: "not sampled")
    import spacr.qt.screens.app_screen as app_screen_module
    app_screen_module._nvidia_smi_available = lambda: False
    from spacr.qt.walkthrough import mark_seen

    catalog = json.loads(CATALOG.read_text())
    lessons = [lesson for lesson in catalog["lessons"]
               if lesson["number"] >= 37 and lesson["id"] not in SPECIALIZED]
    if args.only:
        wanted = set(args.only)
        lessons = [lesson for lesson in lessons
                   if lesson["id"] in wanted or lesson["app_key"] in wanted]
    for lesson in lessons:
        mark_seen(lesson.get("host_app_key") or lesson["app_key"])

    app = QApplication.instance() or QApplication([])
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 60)
    frame = tutorial_frame()

    for number, lesson in enumerate(lessons, 1):
        key = lesson["app_key"]
        screen = open_tutorial_target(window, lesson, app)
        root = PRODUCTION / lesson["id"] / "keyframes"
        frames = {}

        save(window, root / "01_overview.png")
        frames["overview"] = {"file": "01_overview.png",
                              "focus": rect(screen, window)}

        populate(screen, key, frame)
        settle(app, 60)
        save(window, root / "02_input.png")
        input_widget = (getattr(screen, "_empty_state_card", None)
                        or first_button(screen, ("load", "browse", "open"))
                        or screen)
        frames["input"] = {"file": "02_input.png",
                           "focus": rect(input_widget, window)}

        change_decision(screen, key)
        settle(app, 40)
        save(window, root / "03_settings.png")
        settings = getattr(screen, "_settings_scroll", None)
        if settings is None:
            settings = next((s for s in screen.findChildren(QScrollArea)
                             if s.isVisible()), screen)
        frames["settings"] = {"file": "03_settings.png",
                              "focus": rect(settings, window)}

        action = first_button(
            screen, ("explain", "investigate", "run pca", "scan", "fit", "plot", "build", "compare",
                     "refresh", "redraw", "export", "run", "submit"))
        if action is not None:
            action.setFocus()
        settle(app, 20)
        save(window, root / "04_action.png")
        frames["run"] = {"file": "04_action.png",
                         "focus": rect(action or screen, window)}

        run_bounded(screen, key, app)
        settle(app, 60)
        save(window, root / "05_result.png")
        result = next((candidate for candidate in (
            getattr(screen, "_results_view", None),
            getattr(screen, "_table", None),
            getattr(screen, "console", None),
            getattr(screen, "_console", None),
            screen,
        ) if hasattr(candidate, "isVisible")), screen)
        frames["output"] = {"file": "05_result.png",
                            "focus": rect(result, window)}

        geometry = {
            "schema": 2,
            "frame_size": [3840, 2160],
            "device_pixel_ratio": float(window.devicePixelRatioF()),
            "frames": frames,
        }
        (root / "geometry.json").write_text(
            json.dumps(geometry, indent=2) + "\n")
        manifest = {
            "schema": 1,
            "lesson": lesson["id"],
            "capture": "deterministic in-memory workflow",
            "source_rows": len(frame),
            "no_project_writes": True,
            "states": list(frames),
        }
        (root.parent / "source_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        print(f"{number:02d}/{len(lessons)} {lesson['id']}", flush=True)

    window.close()
    settle(app, 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
