"""Plate exports reconstruct well values, shared scales and the complete layout."""
from __future__ import annotations

import json
import zipfile

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spacr.figures import bundle
from spacr.figures.plates import build_plates


@pytest.mark.parametrize("grouping", ["mean", "sum", "count"])
def test_standalone_plate_recipe_keeps_values_missing_wells_and_shared_layout(
        grouping, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    frame = pd.DataFrame([
        {"prc": f"plate{plate}_r{row}_c{column}",
         "value": float(plate * 100 + row * 10 + column + repeat),
         "source_object_id": f"{plate}:{row}:{column}:{repeat}"}
        for plate in range(1, 5) for row in [1, 2, 4] for column in [1, 2, 6]
        for repeat in range(2)])
    frame.loc[frame["prc"] == "plate1_r1_c1", "value"] = 0.0
    frame.loc[frame["prc"] == "plate2_r2_c2", "value"] = np.nan
    frame = frame.loc[frame["source_object_id"] != "3:4:6:1"].reset_index(drop=True)
    custom = LinearSegmentedColormap.from_list("unregistered_recipe_ramp", ["#081122", "#ffdd66", "#dd3300"])
    figure, _panel = build_plates(frame, "value", grouping=grouping,
                                 min_count=2, cmap=custom, target="print")
    try:
        original = [axes for axes in figure.axes if axes.images]
        assert len(original) == 4
        path = bundle._save_zip(figure, str(tmp_path / "plates.zip"), formats=["svg"])
        with zipfile.ZipFile(path) as archive:
            assert archive.read("data.csv").decode() == frame.to_csv(index=False)
            data = pd.read_csv(archive.open("data.csv"))
            spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
        namespace = {"__name__": "standalone_plates"}
        exec(compile(script, "recreate_figure.py", "exec"), namespace)
        recreated = Figure()
        namespace["_draw"](recreated, data, spec)
        actual = [axes for axes in recreated.axes if axes.images]
        assert len(actual) == 4
        for before, after in zip(original, actual):
            np.testing.assert_allclose(np.ma.filled(after.images[0].get_array(), np.nan),
                                       np.ma.filled(before.images[0].get_array(), np.nan), equal_nan=True)
            assert after.get_title() == before.get_title()
            assert after.get_position().bounds == pytest.approx(before.get_position().bounds)
            assert after.images[0].get_clim() == before.images[0].get_clim()
            np.testing.assert_allclose(after.images[0].cmap(np.linspace(0, 1, 256)),
                                       before.images[0].cmap(np.linspace(0, 1, 256)))
            assert [tick.get_text() for tick in after.get_xticklabels()] == [
                tick.get_text() for tick in before.get_xticklabels()]
            assert [tick.get_text() for tick in after.get_yticklabels()] == [
                tick.get_text() for tick in before.get_yticklabels()]
        assert len([axes for axes in recreated.axes if axes.get_label() == "<colorbar>"]) == 1
        first = np.ma.filled(actual[0].images[0].get_array(), np.nan)
        assert first[0, 0] == (2 if grouping == "count" else 0)
        assert np.isnan(first[2, 0])
        third = np.ma.filled(actual[2].images[0].get_array(), np.nan)
        assert np.isnan(third[3, 5])
        reference = frame.loc[frame["prc"] == "plate4_r2_c6", "value"]
        expected = {"mean": reference.mean(), "sum": reference.sum(), "count": len(reference)}[grouping]
        assert actual[3].images[0].get_array()[1, 5] == pytest.approx(expected)
    finally:
        plt.close(figure)


def test_underscored_plate_identifiers_never_mix_other_plates():
    frame = pd.DataFrame({"prc": ["screen_plate_one_r1_c1", "screen_plate_two_r1_c1"],
                          "value": [0.0, 100.0]})
    figure, _panel = build_plates(frame, "value", grouping="mean", target="print")
    try:
        actual = [axes.images[0].get_array()[0, 0] for axes in figure.axes if axes.images]
        assert actual == [0.0, 100.0]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("outline", ["mark", "not_available"])
def test_plate_recipe_handles_outline_masks_and_all_unreadable_wells(outline):
    frame = pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c1", "plate1_r2_c2"],
                          "value": ["1", "3", "unreadable"], "mark": [0, 2, 1]})
    figure, _panel = build_plates(frame, "value", outline=outline, target="print")
    try:
        _data, spec = bundle._figure_record(figure)
        restored = Figure()
        bundle._draw(restored, frame, spec)
        before = next(axes for axes in figure.axes if axes.images)
        after = next(axes for axes in restored.axes if axes.images)
        np.testing.assert_allclose(np.ma.filled(after.images[0].get_array(), np.nan),
                                   np.ma.filled(before.images[0].get_array(), np.nan), equal_nan=True)
        assert after.images[0].get_array()[0, 0] == 2.0
        assert np.ma.is_masked(after.images[0].get_array()[1, 1])
        assert len(after.patches) == len(before.patches)
        for actual, original in zip(after.patches, before.patches):
            assert actual.get_xy() == original.get_xy()
            assert actual.get_width() == original.get_width()
            assert actual.get_height() == original.get_height()
    finally:
        plt.close(figure)
