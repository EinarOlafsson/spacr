"""A real database → Qt worker → adaptive UMAP search round trip.

The fast hyperparameter tests inject an embedder on purpose.  This integration
test is the small boundary check they cannot provide: it creates a genuine
``measurements.db``, opens the real Image UMAP screen, changes ``src`` after
the search panel is already open, and runs the real umap-learn reducer twice.
"""
from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.hyperparam import umap_available
from spacr.qt.screens.app_screen import AppScreen


def _measurements_project(root):
    measurements = root / "measurements"
    measurements.mkdir()
    rng = np.random.default_rng(7)
    n = 45
    clusters = np.repeat([0.0, 5.0, 10.0], 15)
    frame = pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{i % 5}" for i in range(n)],
        "columnID": [f"c{1 + i // 15}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": np.arange(1, n + 1),
        "cell_channel_0_area": clusters + rng.normal(0, 0.3, n),
        "cell_channel_0_mean_intensity": (
            2 * clusters + rng.normal(0, 0.4, n)),
        "cell_channel_1_mean_intensity": (
            -clusters + rng.normal(0, 0.2, n)),
    })
    with sqlite3.connect(measurements / "measurements.db") as connection:
        frame.to_sql("cell", connection, index=False)
    return root, n


def test_image_umap_adaptive_search_runs_twice_with_fresh_module_settings(
        qtbot, qt_theme_applied, tmp_path):
    available, reason = umap_available()
    if not available:
        pytest.skip(reason)
    root, n_rows = _measurements_project(tmp_path)

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()

    # Reproduce the original stale-source ordering: Search is opened first,
    # then drag/drop (represented by the same settings-model write) changes
    # the main module's src.
    screen._hp_switch.setChecked(True)
    panel = screen._hyperparam
    screen._settings_model.set_value_for_key("src", str(root))
    screen._settings_model.set_value_for_key("tables", ["cell"])
    panel._value_edits["n_neighbors"].setText("5")
    panel._value_edits["min_dist"].setText("0.1")
    panel._value_edits["metric"].setText("euclidean")
    panel._adaptive.setChecked(True)
    panel._adaptive_rounds.setText("1")

    results = []
    panel.search_finished.connect(results.append)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        for expected_count in (1, 2):
            # First use can include numba's local compilation on a clean
            # machine; that is startup work, not a hung search.
            with qtbot.waitSignal(panel.search_finished, timeout=120_000):
                assert panel.run_search(), panel._status.text()
            assert len(results) == expected_count
            assert panel._worker is None
            assert panel._run_btn.isEnabled()
            result = results[-1]
            assert result.ok
            assert len(result.successful) == 4
            assert all(
                trial.extra_metrics["embedding"].shape == (n_rows, 2)
                for trial in result.successful
            )

    assert panel._settings["src"] == str(root)
    messages = [str(item.message) for item in seen]
    assert not any("n_jobs value" in message for message in messages)
    assert not any(
        "n_neighbors is larger than the dataset size" in message
        for message in messages
    )
