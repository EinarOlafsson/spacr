"""Numeric line exports retain x coordinates and repeated observations."""
import json
import os
import subprocess
import sys
import zipfile

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr import plot
from spacr.figures import bundle


@pytest.mark.parametrize("times", [[1.0, 3.0, 9.0, 27.0], [1.0, 1.0, 3.0, 9.0],
                                    [9.0, 1.0, 27.0, 3.0]])
def test_numeric_line_standalone_uses_original_coordinates_and_rows(
        times, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    frame = pd.DataFrame({"time": times, "signal": [2.0, 6.0, 4.0, 8.0],
                          "original_id": ["a", "b", "c", "d"]})
    figure = Figure()
    native = figure.subplots()
    native.plot(frame["time"], frame["signal"])
    bundle._register_figure_data(figure, frame, x="time", y="signal", kind="line")
    output = bundle._save_zip(figure, str(tmp_path / "line.zip"), formats=["svg"])
    with zipfile.ZipFile(output) as archive:
        source = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
        archive.extractall(tmp_path / "standalone")
    pd.testing.assert_frame_equal(source, frame)
    namespace = {"__name__": "standalone_numeric_line"}
    exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
    restored = Figure()
    axis = namespace["_draw"](restored, source, spec)
    assert len(axis.lines) == 1
    np.testing.assert_allclose(axis.lines[0].get_xdata(), times)
    np.testing.assert_allclose(axis.lines[0].get_ydata(), frame["signal"])
    edited = source.copy()
    edited.loc[edited["original_id"] == "c", "signal"] = 42.0
    altered = Figure()
    namespace["_draw"](altered, edited, spec)
    np.testing.assert_allclose(altered.axes[0].lines[0].get_ydata(), [2.0, 6.0, 42.0, 8.0])
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import sys\n"
        "class RejectSpacr:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'spacr' or fullname.startswith('spacr.'):\n"
        "            raise ImportError('standalone must not import spaCR')\n"
        "sys.meta_path.insert(0, RejectSpacr())\n")
    process = subprocess.run(
        [sys.executable, str(tmp_path / "standalone/recreate_figure.py"), "png", "svg"],
        cwd=tmp_path / "standalone",
        env=dict(os.environ, PYTHONPATH=str(guard), MPLBACKEND="Agg"),
        capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert (tmp_path / "standalone/recreated.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (tmp_path / "standalone/recreated.svg").read_text()


def test_grouped_line_keeps_categorical_positions_and_group_summaries(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    frame = pd.DataFrame({"condition": [1, 3, 9] * 12,
                          "signal": np.arange(36, dtype=float)})
    figure, _ = plot.create_grouped_plot(frame, "condition", "signal",
                                          graph_type="line", save=False)
    try:
        data, spec = bundle._figure_record(figure)
        assert spec["kind"] == "point"
        assert len(data) == len(frame)
        np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), [0, 1, 2])
        means = [np.mean(np.arange(start, 36, 3)) for start in range(3)]
        np.testing.assert_allclose(np.asarray(figure.axes[0].lines[0].get_ydata(), dtype=float), means)
    finally:
        plt.close(figure)
