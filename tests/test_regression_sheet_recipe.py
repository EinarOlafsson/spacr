"""A sheet export runs the original numerical panels without spaCR installed."""
from __future__ import annotations

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

from spacr.figures import bundle
from spacr.figures.sheet import build_sheet
from tests.test_the_figure_house_style import _results


@pytest.mark.parametrize("variant", ["full", "sparse", "custom thresholds"])
def test_standalone_sheet_preserves_numeric_panels_and_complete_source(
        variant, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    frame = _results(120)
    if variant == "sparse":
        frame = frame.drop(columns=["p_value", "q_value", "condition"])
    kwargs = {"alpha": 0.001, "effect_threshold": 0.2, "highlight": "0_0"} if variant == "custom thresholds" else {}
    sheet = build_sheet(frame, target="print", **kwargs)
    try:
        expected = [axes for axes in sheet.figure.axes if axes.has_data()]
        assert len(expected) == len(sheet.panels)
        assert len(expected) > 1
        path = bundle._save_zip(sheet.figure, str(tmp_path / "sheet.zip"), formats=["svg"])
        with zipfile.ZipFile(path) as archive:
            assert archive.read("data.csv").decode() == frame.to_csv(index=False)
            data = pd.read_csv(archive.open("data.csv"))
            spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
            destination = tmp_path / "standalone"
            archive.extractall(destination)
        blocker = tmp_path / "import_guard"
        blocker.mkdir()
        (blocker / "sitecustomize.py").write_text(
            "import sys\n"
            "class RejectSpacr:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'spacr' or fullname.startswith('spacr.'):\n"
            "            raise ImportError('standalone script must not import spaCR')\n"
            "sys.meta_path.insert(0, RejectSpacr())\n")
        environment = dict(os.environ, PYTHONPATH=str(blocker), MPLBACKEND="Agg")
        process = subprocess.run(
            [sys.executable, str(destination / "recreate_figure.py"), "png", "svg"],
            cwd=destination, env=environment, capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        assert (destination / "recreated.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in (destination / "recreated.svg").read_text()
        namespace = {"__name__": "standalone_regression_sheet"}
        exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
        restored = Figure()
        namespace["_draw"](restored, data, spec)
        actual = [axes for axes in restored.axes if axes.has_data()]
        assert len(actual) == len(expected)
        for before, after in zip(expected, actual):
            assert after.get_xlabel() == before.get_xlabel()
            assert after.get_ylabel() == before.get_ylabel()
            assert after.get_position().bounds == pytest.approx(before.get_position().bounds)
            assert len(after.patches) == len(before.patches)
            for original, recreated in zip(before.patches, after.patches):
                assert recreated.get_x() == pytest.approx(original.get_x())
                assert recreated.get_height() == pytest.approx(original.get_height())
                assert recreated.get_width() == pytest.approx(original.get_width())
            assert len(after.collections) == len(before.collections)
            for original, recreated in zip(before.collections, after.collections):
                np.testing.assert_allclose(recreated.get_offsets(), original.get_offsets(), equal_nan=True)
                np.testing.assert_allclose(recreated.get_facecolors(), original.get_facecolors(), equal_nan=True)
            assert len(after.lines) == len(before.lines)
            for original, recreated in zip(before.lines, after.lines):
                np.testing.assert_allclose(recreated.get_xdata(), original.get_xdata(), equal_nan=True)
                np.testing.assert_allclose(recreated.get_ydata(), original.get_ydata(), equal_nan=True)
        if variant != "sparse":
            qq = actual[[panel.key for panel in sheet.panels].index("qq")]
            values = np.sort(frame.loc[frame["feature"] != "Intercept", "p_value"].to_numpy())
            offsets = qq.collections[0].get_offsets()
            np.testing.assert_allclose(offsets[:, 0], -np.log10((np.arange(1, len(values) + 1) - 0.5) / len(values)))
            np.testing.assert_allclose(offsets[:, 1], -np.log10(values))
    finally:
        plt.close(sheet.figure)
