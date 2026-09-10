#!/usr/bin/env python3
"""Capture a native-4K Gate Editor walkthrough in 2D, 3D, and xD.

The source is deterministic synthetic measurement data with stable spaCR
object keys.  It exists only to make every Gate Editor mode legible without
depending on a user's project, while the plots, gates, PCA reduction, counts,
and application chrome are all produced by the real spaCR widgets.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
LESSON = "64_gate_editor"
OUTPUT = ROOT / "production" / LESSON / "keyframes"


def tutorial_measurements():
    """Return reproducible multi-feature objects with three visible groups."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(150004)
    sizes = (560, 330, 190)
    centres = np.asarray([
        [790.0, 150.0, 22.0, 0.42, 11.0, 0.90],
        [1170.0, 232.0, 88.0, 0.70, 24.0, 0.78],
        [930.0, 180.0, 148.0, 0.58, 31.0, 0.70],
    ])
    scales = np.asarray([
        [85.0, 16.0, 10.0, 0.055, 2.5, 0.028],
        [92.0, 19.0, 18.0, 0.060, 3.8, 0.035],
        [105.0, 20.0, 22.0, 0.065, 4.5, 0.045],
    ])
    groups = []
    for index, count in enumerate(sizes):
        latent = rng.normal(size=(count, 3))
        correlated = np.column_stack((
            latent[:, 0],
            0.72 * latent[:, 0] + 0.69 * latent[:, 1],
            0.35 * latent[:, 0] + 0.20 * latent[:, 1] + 0.91 * latent[:, 2],
            0.30 * latent[:, 1] + 0.95 * latent[:, 2],
            0.45 * latent[:, 0] + 0.89 * latent[:, 2],
            -0.55 * latent[:, 0] + 0.84 * latent[:, 1],
        ))
        groups.append(centres[index] + correlated * scales[index])
    values = np.vstack(groups)
    count = len(values)
    index = np.arange(count)
    frame = pd.DataFrame({
        "plateID": "tutorial_plate",
        "rowID": np.where(index < count // 2, "A", "B"),
        "columnID": 1 + (index // 180) % 6,
        "fieldID": 1 + index // 180,
        "object_label": 1 + index % 180,
        "cell area": np.clip(values[:, 0], 350.0, None),
        "nuclear intensity": np.clip(values[:, 1], 55.0, None),
        "pathogen signal": np.clip(values[:, 2], 0.0, None),
        "texture entropy": np.clip(values[:, 3], 0.05, 0.98),
        "organelle count": np.clip(np.rint(values[:, 4]), 1, None),
        "cell solidity": np.clip(values[:, 5], 0.45, 0.99),
    })
    return frame


def hide_secondary_panels(screen) -> None:
    """Keep the short lesson on the gate workspace described by narration."""
    from PySide6.QtWidgets import QScrollArea

    for name in ("_save_filters", "_load_filters", "_save_graph"):
        button = getattr(screen, name, None)
        if button is not None:
            button.hide()
    for panel in screen.findChildren(QScrollArea):
        panel.hide()
    screen.console.hide()


def main() -> int:
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-gate-editor-4k-config")
    os.environ.setdefault(
        "MPLCONFIGDIR", "/tmp/spacr-tutorial-gate-editor-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication
    import spacr

    # Capture the current published application version through the real
    # footer. Never paint over stale version text in a generated frame.
    spacr.__version__ = "1.5.0.4"
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.gate_spec import BoxGate, GateSet, RectGate

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    window._on_nav_selected("gate_editor")
    wait_until(
        app,
        lambda: window._screens.get("gate_editor") is not None,
        15.0,
        "Gate Editor screen",
    )
    screen = window._screens["gate_editor"]
    hide_secondary_panels(screen)
    # Gate names and percentages are teaching content, so reserve enough
    # width for the hierarchy instead of leaving it at its minimal 220 px.
    screen.gates.body.setSizes([2780, 700])
    settle(app, 35)

    frame = tutorial_measurements()
    screen.set_frame(
        frame,
        label=("Tutorial measurements · 1,080 objects × 6 measured "
               "features"),
    )
    screen._x.setCurrentText("cell area")
    screen._y.setCurrentText("nuclear intensity")
    two_d_gate = RectGate(
        name="high-intensity cells",
        x_column="cell area",
        y_column="nuclear intensity",
        x_low=1010.0,
        x_high=1360.0,
        y_low=194.0,
        y_high=282.0,
    )
    screen.gates.set_gates(GateSet().add(two_d_gate))
    settle(app, 45)

    capture = CaptureSession(app, window, OUTPUT)
    modes = list(screen.gates._mode_buttons.values())
    capture.save(
        "01_overview",
        {
            "nav": nav_button(window, "gate_editor"),
            "screen": screen,
            "source": screen._source,
            "axes": [screen._x, screen._y],
            "modes": modes,
            "gate_surface": [screen.gates.canvas, screen.gates.tree],
        },
    )
    capture.save(
        "02_two_d_gate",
        {
            "axes_tools": [screen._x, screen._y, screen.gates._tool,
                           screen.gates._settings_button],
            "two_d_workspace": [screen._x, screen._y, screen.gates._tool,
                                screen.gates.canvas, screen.gates.tree],
            "gate_tree": screen.gates.tree,
        },
    )

    screen._z.setCurrentText("pathogen signal")
    screen.gates._mode_buttons["3D"].click()
    three_d_gate = BoxGate(
        name="three-feature responders",
        x_column="cell area",
        y_column="nuclear intensity",
        z_column="pathogen signal",
        x_low=1020.0,
        x_high=1340.0,
        y_low=195.0,
        y_high=278.0,
        z_low=48.0,
        z_high=132.0,
    )
    screen.gates.set_gates(GateSet().add(two_d_gate).add(three_d_gate))
    settle(app, 55)
    capture.save(
        "03_three_d_volume",
        {
            "three_d_axes": [screen._x, screen._y, screen._z],
            "three_d_controls": [screen.gates._mode_buttons["3D"],
                                 screen.gates._box_gate,
                                 *screen.gates._spin_buttons.values()],
            "three_d_workspace": [screen._x, screen._y, screen._z,
                                  screen.gates._mode_buttons["3D"],
                                  screen.gates._box_gate,
                                  *screen.gates._spin_buttons.values(),
                                  screen.gates.canvas],
            "gate_tree": screen.gates.tree,
        },
    )

    # Clicking xD invokes the real reduction path. PCA is deterministic and
    # always available, so the captured PC axes and explained variance are
    # results from the tutorial measurements rather than a staged label.
    screen.gates._mode_buttons["xD"].click()
    wait_until(
        app,
        lambda: all(name in screen._frame.columns
                    for name in ("PC1", "PC2", "PC3")),
        30.0,
        "xD PCA components",
    )
    settle(app, 55)
    capture.save(
        "04_xd_projection",
        {
            "source": screen._source,
            "xd_axes": [screen._x, screen._y, screen._z],
            "xd_mode": screen.gates._mode_buttons["xD"],
            "xd_workspace": [screen._source, screen._x, screen._y, screen._z,
                             screen.gates._mode_buttons["xD"],
                             screen.gates.canvas],
            "gate_tree": screen.gates.tree,
        },
    )

    screen.gates._mode_buttons["2D"].click()
    screen._x.setCurrentText("cell area")
    screen._y.setCurrentText("nuclear intensity")
    settle(app, 45)
    capture.save(
        "05_gate_hierarchy",
        {
            "gate_tree": screen.gates.tree,
            "gate_surface": [screen.gates.canvas, screen.gates.tree],
            "status": screen.gates._status,
        },
    )
    # The hierarchy widget stretches to the bottom of the workspace, while
    # its meaningful rows are at the top. Keep the emphasis box on the
    # heading, names, and percentages instead of outlining empty space.
    hierarchy_rect = capture.frames["05_gate_hierarchy"].get("gate_tree")
    if hierarchy_rect:
        hierarchy_rect[3] = min(hierarchy_rect[3], 270)
    capture.save(
        "06_save_reuse",
        {
            "save_controls": [screen._save_gates, screen._load_gates],
            "output_controls": [screen._annotate, screen._export],
            "source": screen._source,
            "gate_tree": screen.gates.tree,
        },
    )

    # Exercise the production DBSCAN Walk on the same deterministic table.
    # This is a real bounded search over eps, not a staged status message.
    from spacr.qt.widgets.gate_spec import (
        best_cluster_candidate, cluster_gates, cluster_walk_candidates)
    candidates = cluster_walk_candidates(
        frame, "cell area", "nuclear intensity", eps=0.1,
        min_samples=10, scale=True, steps=12, method="dbscan")
    chosen = best_cluster_candidate(candidates)
    if chosen is None:
        raise RuntimeError("Gate Editor Walk found no defensible candidate")
    found = cluster_gates(
        frame, "cell area", "nuclear intensity", eps=chosen.eps,
        min_samples=10, scale=True, method="dbscan")
    if len(found) < 2:
        raise RuntimeError("Gate Editor Walk did not find two populations")
    walked = screen.gates.gates
    for gate in found:
        walked.add(gate)
    screen.gates.set_gates(walked)
    screen.gates._refresh_status()
    screen.gates._status.setText(
        f"Walk finished · eps {chosen.eps:.3g} · "
        f"{chosen.clusters} populations · {chosen.noise_fraction:.0%} outside")
    settle(app, 45)
    capture.save(
        "07_cluster_walk",
        {
            "walk_action": screen.gates._cluster,
            "walk_result": [screen.gates.canvas, screen.gates.tree,
                            screen.gates._status],
            "output_controls": [screen._annotate, screen._export],
        },
    )
    capture.write_geometry()

    gate_counts = {
        gate.name: int(screen.gates.gates.mask(screen._frame, gate.name).sum())
        for gate in screen.gates.gates.gates
    }
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "gate_editor",
        "app_version": "1.5.0.4",
        "source": "deterministic synthetic multi-feature measurements",
        "seed": 150004,
        "rows": len(frame),
        "measured_features": [
            "cell area", "nuclear intensity", "pathogen signal",
            "texture entropy", "organelle count", "cell solidity",
        ],
        "modes": ["2D", "3D", "xD"],
        "xd_method": "pca",
        "gate_counts": gate_counts,
        "cluster_walk": {
            "eps_start": 0.1,
            "steps": 12,
            "chosen_eps": chosen.eps,
            "clusters": chosen.clusters,
            "noise_fraction": chosen.noise_fraction,
        },
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    window.close()
    settle(app, 30)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
