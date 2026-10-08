"""Gene overlap exports preserve both input tables and recreate the Venn plot."""
import json
import os
from pathlib import Path
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


@pytest.mark.parametrize("threshold", [0.1, -0.1, None])
def test_venn_export_preserves_full_input_and_recreates_overlap(
        threshold, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    inputs = [pd.DataFrame({
        "target": ["shared", "left", "duplicate", "duplicate", None, "weak"],
        "coefficient": [0.5, 0.6, -0.5, -0.5, 1.0, 0.0],
        "original_id": [f"first:{index}" for index in range(6)],
        "_spacr_venn_input": "existing metadata"}), pd.DataFrame({
        "target": ["shared", "right", "duplicate", "negative", None, "weak"],
        "coefficient": [0.7, 0.8, -0.7, -0.8, 1.0, 0.0],
        "original_id": [f"second:{index}" for index in range(6)],
        "_spacr_venn_input": "other metadata"})]
    for path, frame in zip(paths, inputs):
        frame.to_csv(path, index=False)
    expected_sets = {
        0.1: ({"shared", "left"}, {"shared", "right"}),
        -0.1: ({"duplicate"}, {"duplicate", "negative"}),
        None: ({"shared", "left", "duplicate", "weak"},
               {"shared", "right", "duplicate", "negative", "weak"})}
    left, right = expected_sets[threshold]
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    result = plot.create_venn_diagram(*paths, gene_column="target",
                                      filter_coeff=threshold, save=False)
    assert set(result["overlap"]) == left & right
    assert set(result["unique_to_file1"]) == left - right
    assert set(result["unique_to_file2"]) == right - left
    assert len(figures) == 1
    figure = figures[0]
    try:
        output = bundle._save_zip(figure, str(tmp_path / "venn.zip"), formats=["svg"])
        with zipfile.ZipFile(output) as archive:
            frame = pd.read_csv(archive.open("data.csv"))
            spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
            archive.extractall(tmp_path / "standalone")
        expected = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        pd.testing.assert_frame_equal(frame[expected.columns], expected)
        assert spec["kind"] == "venn"
        recipe = spec["venn"]
        assert recipe["gene_column"] == "target"
        assert recipe["filter_coeff"] == threshold
        assert recipe["input_column"] != "_spacr_venn_input"
        assert frame[recipe["input_column"]].tolist() == [0] * 6 + [1] * 6
        assert [part["path"] for part in recipe["inputs"]] == [str(path) for path in paths]
        namespace = {"__name__": "standalone_venn"}
        exec(compile(script, "recreate_figure.py", "exec", dont_inherit=True), namespace)
        recreated = Figure()
        native_recreated = Figure()
        namespace["_draw"](recreated, frame, spec)
        bundle._draw(native_recreated, frame, spec)
        before = figure.axes[0]
        for restored in (recreated.axes[0], native_recreated.axes[0]):
            assert restored.get_title() == before.get_title()
            assert [text.get_text() for text in restored.texts] == [text.get_text() for text in before.texts]
            assert len(restored.patches) == len(before.patches)
            for original, copied in zip(before.patches, restored.patches):
                np.testing.assert_allclose(copied.get_path().vertices, original.get_path().vertices)
                np.testing.assert_array_equal(copied.get_path().codes, original.get_path().codes)
                np.testing.assert_allclose(copied.get_facecolor(), original.get_facecolor())
                assert copied.get_alpha() == original.get_alpha()
            for original, copied in zip(before.texts, restored.texts):
                np.testing.assert_allclose(copied.get_position(), original.get_position())
                assert copied.get_fontsize() == original.get_fontsize()
                assert copied.get_color() == original.get_color()
        edited = frame.copy()
        edited.loc[edited["original_id"] == "first:0", "target"] = "new gene"
        changed = Figure()
        namespace["_draw"](changed, edited, spec)
        if threshold is not None and threshold < 0:
            assert [text.get_text() for text in changed.axes[0].texts] == [text.get_text() for text in before.texts]
        else:
            assert [text.get_text() for text in changed.axes[0].texts] != [text.get_text() for text in before.texts]
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
        plt.close(figure)
