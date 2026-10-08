"""Single regression panels retain source rows and their actual numerical recipe."""
import json
import os
import subprocess
import sys
import zipfile

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from spacr.figures import bundle
from spacr.figures.panels import SHEET_ORDER
from spacr.figures.sheet import build_panel
from tests.test_the_figure_house_style import _results


@pytest.mark.parametrize("key,options", [
    *[(key, {}) for key in SHEET_ORDER],
    ("volcano", {"alpha": 0.001, "effect_threshold": 0.2, "highlight": "0_0", "label_top": 2}),
    ("volcano", {"baseline_kind": "controls"}),
    ("volcano", {"compartment": "Nucleus"}),
    ("effect_rank", {"alpha": 0.001, "top": 5}),
    ("p_histogram", {"bins": 13}),
])
def test_single_panel_recreation_preserves_complete_csv_and_numeric_artists(
        key, options, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr import localisation
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    monkeypatch.setattr(localisation, "table", lambda: {"0": "Nucleus", "1": "Nucleus"})
    frame = _results(120)
    figure, panel = build_panel(key, frame, target="print", **options)
    try:
        assert panel.drawn
        path = bundle._save_zip(figure, str(tmp_path / "panel.zip"), formats=["svg"])
        with zipfile.ZipFile(path) as archive:
            assert archive.read("data.csv").decode() == frame.to_csv(index=False)
            data = pd.read_csv(archive.open("data.csv"))
            spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
            if key == "volcano":
                archive.extractall(tmp_path / "standalone")
        if key == "volcano":
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
        namespace = {"__name__": "standalone_panel"}
        exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
        restored = Figure()
        namespace["_draw"](restored, data, spec)
        assert len(restored.axes) == 1
        before, after = figure.axes[0], restored.axes[0]
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
        assert [text.get_text() for text in after.texts] == [text.get_text() for text in before.texts]
        if options.get("baseline_kind") == "controls":
            shift = frame.loc[frame["condition"].isin(["nc", "control", "negative"]), "coefficient"].median()
            expected = frame.loc[frame["feature"] != "Intercept", "coefficient"].to_numpy() - shift
            plotted = np.concatenate([collection.get_offsets()[:, 0] for collection in after.collections])
            np.testing.assert_allclose(np.sort(plotted), np.sort(expected))
        if options.get("compartment"):
            assert len(after.collections[1].get_offsets()) == 8
    finally:
        plt.close(figure)
