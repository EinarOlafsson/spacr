"""A GUI run writes the notebook that repeats it, into its project.

The notebook holds the settings the Run button handed to the pipeline and
the call it made. These tests build a tiny replication project, run it the
way the GUI does, run the written notebook, and compare what each wrote.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AREA = 100.0
DIAMETER = 2.0 * np.sqrt(AREA / np.pi)


@pytest.fixture(autouse=True)
def _no_figure_windows(monkeypatch):
    """Keep every figure off screen and close them afterwards."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _project(root: Path) -> str:
    """A replication project: two wells, vacuoles of 1, 2, 4 and 8 parasites."""
    measurements = root / "measurements"
    measurements.mkdir(parents=True)
    pathogens, cells, label = [], [], 0
    for column, sizes in (("c1", (1, 2, 4)), ("c2", (2, 4, 8))):
        prcf = f"plate1_r1_{column}_f1"
        for cell, n in enumerate(sizes, start=1):
            for i in range(n):
                label += 1
                pathogens.append({
                    "object_label": label, "cell_id": cell,
                    "plateID": "plate1", "rowID": "r1", "columnID": column,
                    "fieldID": "f1", "prcf": prcf,
                    "pathogen_area": AREA,
                    "pathogen_equivalent_diameter_area": DIAMETER,
                    "pathogen_channel_0_centroid_weighted-0": 50.0 + cell * 400.0,
                    "pathogen_channel_0_centroid_weighted-1": 50.0 + i * 8.0,
                    "pathogen_channel_0_mean_intensity": 50.0,
                })
            cells.append({
                "object_label": cell, "plateID": "plate1", "rowID": "r1",
                "columnID": column, "fieldID": "f1", "prcf": prcf,
                "cell_area": 20000.0, "cell_channel_0_mean_intensity": 100.0,
            })
    with sqlite3.connect(measurements / "measurements.db") as con:
        pd.DataFrame(pathogens).to_sql("pathogen", con, index=False)
        pd.DataFrame(cells).to_sql("cell", con, index=False)
    return str(root)


def _gui_settings(qtbot, src):
    """The settings the replication form collects for ``src``."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt.screens.settings_model import SettingsWidgets

    host = QWidget()
    qtbot.addWidget(host)
    model = SettingsWidgets("replication", host)
    model.build_sections()
    for key, value in (("src", src), ("pathogen_types", ["dmso", "drug"]),
                       ("pathogen_plate_metadata", [["c1"], ["c2"]]),
                       ("save", True), ("verbose", False)):
        model.set_value_for_key(key, value)
    return model.collect()


def _outputs(src):
    """Every CSV the run wrote under ``results``, by relative path."""
    results = Path(src) / "results"
    return {str(path.relative_to(results)): pd.read_csv(path)
            for path in sorted(results.rglob("*.csv"))}


def _code_cells(path):
    """The source of every code cell of a notebook."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


def _same_outputs(first, second):
    """Assert two output sets hold the same files with the same tables."""
    assert sorted(first) == sorted(second)
    assert first
    for name, table in first.items():
        pd.testing.assert_frame_equal(table, second[name], check_like=True)


def test_the_notebook_holds_the_settings_and_the_run_call(tmp_path):
    """The notebook's settings evaluate to the run's, and it calls the entry."""
    from spacr.notebook_export import _write_launch_notebook

    src = _project(tmp_path / "plate1")
    settings = {"src": src, "panels": [{"name": "p", "group": "condition"}],
                "figures": None, "save": True}
    path = _write_launch_notebook("replication", settings)
    assert path.parent == Path(src) / "settings"
    cells = _code_cells(path)
    namespace = {}
    exec(cells[0], namespace)
    assert namespace["SETTINGS"] == settings
    assert "from spacr.submodules import analyze_replication" in cells[1]
    assert "coerce_expected_types(dict(SETTINGS), 'replication')" in cells[1]
    assert _write_launch_notebook("replication", settings) != path


def test_no_notebook_without_a_project_or_an_entry(tmp_path):
    """A run with no project on disk, or no entry point, writes nothing."""
    from spacr.notebook_export import _write_launch_notebook

    assert _write_launch_notebook("replication", {"src": str(tmp_path / "no")}) is None
    assert _write_launch_notebook("annotate", {"src": str(tmp_path)}) is None
    assert _write_launch_notebook("replication",
                                  {"src": str(tmp_path), "x": object()}) is None
    assert not (tmp_path / "settings").exists()


def test_the_notebook_reproduces_the_gui_run(qtbot, tmp_path):
    """Running the notebook's cells writes the tables the GUI path wrote."""
    from spacr.notebook_export import _write_launch_notebook
    from spacr.qt.bridge import resolve_pipeline_entry

    src = _project(tmp_path / "plate1")
    settings = _gui_settings(qtbot, src)
    path = _write_launch_notebook("replication", settings)
    gui_result = resolve_pipeline_entry("replication")(dict(settings))
    gui_outputs = _outputs(src)
    shutil.rmtree(Path(src) / "results")

    namespace = {}
    for source in _code_cells(path):
        exec(compile(source, str(path), "exec"), namespace)
    _same_outputs(gui_outputs, _outputs(src))
    pd.testing.assert_frame_equal(gui_result["vacuoles"],
                                  namespace["result"]["vacuoles"])


def test_the_notebook_runs_in_a_kernel(qtbot, monkeypatch, tmp_path):
    """Executed by nbclient, the notebook writes the GUI run's tables."""
    nbclient = pytest.importorskip("nbclient")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    from spacr.notebook_export import _write_launch_notebook
    from spacr.qt.bridge import resolve_pipeline_entry

    src = _project(tmp_path / "plate1")
    settings = _gui_settings(qtbot, src)
    path = _write_launch_notebook("replication", settings)
    resolve_pipeline_entry("replication")(dict(settings))
    gui_outputs = _outputs(src)
    shutil.rmtree(Path(src) / "results")

    notebook = nbformat.read(str(path), as_version=4)
    nbformat.validate(notebook)
    repo = str(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [repo] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}}).execute()
    _same_outputs(gui_outputs, _outputs(src))


@pytest.mark.usefixtures("console_info_enabled")
def test_pressing_run_writes_the_notebook_of_what_it_ran(qtbot, monkeypatch, tmp_path):
    """The Run button writes the notebook before the pipeline starts."""
    from spacr.qt import bridge
    from spacr.qt.screens.app_screen import AppScreen

    received = []
    entry = lambda key: received.append
    monkeypatch.setattr(bridge, "resolve_pipeline_entry", entry)
    monkeypatch.setattr("spacr.qt.screens.app_screen.resolve_pipeline_entry", entry)
    src = _project(tmp_path / "plate1")
    screen = AppScreen("replication")
    qtbot.addWidget(screen)
    screen._settings_model.set_value_for_key("src", src)
    screen._on_run()
    qtbot.waitUntil(lambda: screen._btn_run.isEnabled(), timeout=10000)
    path = screen._run_notebook_path
    assert path is not None and path.parent == Path(src) / "settings"
    namespace = {}
    exec(_code_cells(path)[0], namespace)
    assert received and namespace["SETTINGS"] == received[0]
