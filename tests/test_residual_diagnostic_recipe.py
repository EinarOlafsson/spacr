"""Residual export recipes reproduce the histogram, QQ and fitted-value plots."""
import json
import os
from types import SimpleNamespace
import subprocess
import sys
import zipfile

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from spacr import plot
from spacr.figures import bundle


@pytest.mark.parametrize("diagnostic", [0, 1, 2], ids=["histogram", "qq", "fitted"])
def test_residual_diagnostic_export_preserves_numeric_artists_and_standalone_cli(
        diagnostic, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    residuals = np.array([-3.2, -2.1, -1.2, -0.8, -0.2, 0.1, 0.15, 0.4, 0.6,
                          0.7, 1.1, 1.2, 1.8, 2.5, 3.1, 4.2, 6.0])
    fitted = np.linspace(2.0, 20.0, len(residuals))
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    plot._show_residules(SimpleNamespace(resid=residuals, fittedvalues=fitted))
    assert len(figures) == 3
    try:
        figure = figures[diagnostic]
        path = bundle._save_zip(figure, str(tmp_path / "diagnostic.zip"), formats=["svg"])
        with zipfile.ZipFile(path) as archive:
            frame = pd.read_csv(archive.open("data.csv"))
            spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
            archive.extractall(tmp_path / "standalone")
        np.testing.assert_allclose(frame["residual"], residuals)
        namespace = {"__name__": "standalone_residual"}
        exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
        restored = Figure()
        namespace["_draw"](restored, frame, spec)
        before, after = figure.axes[0], restored.axes[0]
        assert after.get_title() == before.get_title()
        assert after.get_xlabel() == before.get_xlabel()
        assert after.get_ylabel() == before.get_ylabel()
        if diagnostic == 0:
            counts, edges = np.histogram(residuals, bins=30)
            assert len(after.patches) == 30
            np.testing.assert_allclose([patch.get_height() for patch in after.patches], counts)
            np.testing.assert_allclose([patch.get_x() for patch in after.patches], edges[:-1])
            np.testing.assert_allclose([patch.get_width() for patch in after.patches], np.diff(edges))
        elif diagnostic == 1:
            assert len(after.lines) == 2
            expected_x = norm.ppf(np.arange(1, len(residuals) + 1) / (len(residuals) + 1))
            expected_y = (np.sort(residuals) - residuals.mean()) / residuals.std()
            np.testing.assert_allclose(after.lines[0].get_xdata(), expected_x)
            np.testing.assert_allclose(after.lines[0].get_ydata(), expected_y)
            for original, recreated in zip(before.lines, after.lines):
                np.testing.assert_allclose(recreated.get_xdata(), original.get_xdata())
                np.testing.assert_allclose(recreated.get_ydata(), original.get_ydata())
                assert recreated.get_linestyle() == original.get_linestyle()
                assert recreated.get_color() == original.get_color()
        else:
            np.testing.assert_allclose(frame["fitted"], fitted)
            assert len(after.collections) == 1
            np.testing.assert_allclose(after.collections[0].get_offsets(), np.c_[fitted, residuals])
            np.testing.assert_allclose(after.collections[0].get_sizes(), [8])
            np.testing.assert_allclose(after.collections[0].get_facecolors(), before.collections[0].get_facecolors())
            assert len(after.lines) == 1
            np.testing.assert_allclose(after.lines[0].get_ydata(), [0, 0])
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
    finally:
        for figure in figures:
            plt.close(figure)
