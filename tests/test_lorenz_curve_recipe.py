"""Lorenz exports retain input provenance and reproduce cumulative shares."""
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

from spacr import plot
from spacr.figures import bundle


@pytest.fixture
def lorenz_inputs(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    frames = [pd.DataFrame({
        "guide": [f"guide{index}" for index in range(30)] + ["busy"] * 100 + ["skip"],
        "reads": list(range(1, 31)) + [100] * 100 + [900],
        "original_id": [f"first:{index}" for index in range(131)],
        "_spacr_lorenz_input": "original metadata"}),
        pd.DataFrame({"guide": ["a", "b", "c", "d", "skip"],
                      "reads": [2, 4, 8, 16, 700],
                      "original_id": [f"second:{index}" for index in range(5)],
                      "_spacr_lorenz_input": "other metadata"})]
    paths = [tmp_path / f"plate{index}.csv" for index in range(2)]
    for path, frame in zip(paths, frames):
        frame.to_csv(path, index=False)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    yield paths, frames, figures
    for figure in figures:
        plt.close(figure)


def _produce(inputs, remove_outliers):
    paths, _frames, figures = inputs
    plot.plot_lorenz_curves(paths, name_column="guide", value_column="reads",
                           remove_keys=["skip"], remove_outliers=remove_outliers,
                           x_lim=[0.1, 0.9], y_lim=[0.05, 0.95], save=False)
    return figures[-1]


@pytest.mark.parametrize("remove_outliers", [False, True])
def test_lorenz_recipe_keeps_every_original_row_and_separate_plate_identity(
        lorenz_inputs, remove_outliers, tmp_path):
    figure = _produce(lorenz_inputs, remove_outliers)
    path = bundle._save_zip(figure, str(tmp_path / "lorenz.zip"), formats=["svg"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
    expected = pd.concat(lorenz_inputs[1], ignore_index=True)
    pd.testing.assert_frame_equal(frame[expected.columns], expected)
    assert spec["kind"] == "lorenz"
    recipe = spec["lorenz"]
    assert recipe["input_column"] != "_spacr_lorenz_input"
    assert frame[recipe["input_column"]].tolist() == [0] * 131 + [1] * 5
    assert frame[recipe["row_column"]].tolist() == list(range(131)) + list(range(5))
    assert recipe["remove_keys"] == ["skip"]
    assert recipe["remove_outliers"] is remove_outliers
    assert [part["path"] for part in recipe["curves"]] == [
        str(path) for path in lorenz_inputs[0]]
    assert recipe["curves"][0]["rows"] == list(range(30 if remove_outliers else 130))
    assert recipe["curves"][1]["rows"] == list(range(4))


@pytest.mark.parametrize("remove_outliers", [False, True])
def test_lorenz_standalone_recreates_independent_shares_and_gini_without_spacr(
        lorenz_inputs, remove_outliers, tmp_path):
    figure = _produce(lorenz_inputs, remove_outliers)
    path = bundle._save_zip(figure, str(tmp_path / "lorenz.zip"), formats=["svg"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
        archive.extractall(tmp_path / "standalone")
    namespace = {"__name__": "standalone_lorenz"}
    exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
    restored = Figure()
    namespace["_draw"](restored, frame, spec)
    native_restored = Figure()
    bundle._draw(native_restored, frame, spec)
    assert len(restored.axes[0].lines) == 3
    assert len(native_restored.axes[0].lines) == 3
    first = np.array(list(range(1, 31)) + ([] if remove_outliers else [100] * 100))
    second = np.array([2, 4, 8, 16])
    for index, values in enumerate([first, second, np.concatenate([first, second])]):
        ordered = np.sort(values)
        expected = np.r_[0.0, np.cumsum(ordered) / ordered.sum()]
        x = np.linspace(0.0, 1.0, len(expected))
        gini = np.abs(values[:, None] - values[None, :]).sum() / (
            2 * len(values) * values.sum())
        before, after = figure.axes[0].lines[index], restored.axes[0].lines[index]
        np.testing.assert_allclose(after.get_xdata(), x)
        np.testing.assert_allclose(after.get_ydata(), expected)
        np.testing.assert_allclose(before.get_ydata(), expected)
        np.testing.assert_allclose(native_restored.axes[0].lines[index].get_ydata(), expected)
        prefix = f"plate {index + 1}" if index < 2 else "Combined"
        assert after.get_label() == f"{prefix} (Gini: {gini:.4f})"
        assert after.get_linestyle() == before.get_linestyle()
        assert after.get_color() == before.get_color()
    assert restored.axes[0].get_xlim() == pytest.approx([0.1, 0.9])
    assert restored.axes[0].get_ylim() == pytest.approx([0.05, 0.95])
    assert restored.axes[0].get_xlabel() == figure.axes[0].get_xlabel()
    assert restored.axes[0].get_ylabel() == figure.axes[0].get_ylabel()
    assert [text.get_text() for text in restored.axes[0].texts] == [
        text.get_text() for text in figure.axes[0].texts]
    edited = frame.copy()
    edited.loc[edited["original_id"] == "second:0", "reads"] = 200
    changed = Figure()
    namespace["_draw"](changed, edited, spec)
    changed_values = np.array([200, 4, 8, 16])
    changed_gini = np.abs(changed_values[:, None] - changed_values[None, :]).sum() / (
        2 * len(changed_values) * changed_values.sum())
    assert changed.axes[0].lines[1].get_label() == f"plate 2 (Gini: {changed_gini:.4f})"
    np.testing.assert_allclose(changed.axes[0].lines[1].get_ydata(),
                               np.r_[0.0, np.cumsum(np.sort(changed_values)) / changed_values.sum()])
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
