"""Control plot exports retain every condition, channel and compartment."""
from __future__ import annotations

import json
import zipfile

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spacr import plot
from spacr.figures import bundle


@pytest.fixture
def controls_figure(monkeypatch, tmp_path, request):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    try:
        from PySide6.QtCore import QSettings
        from spacr.qt import preferences
    except ImportError:
        pass
    else:
        monkeypatch.setattr(
            preferences, "_settings",
            lambda: QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat))
    frame = pd.DataFrame({
        "condition": ["control"] * 4 + ["treated"] * 4,
        "source_object_id": list(range(8)),
        **{f"{compartment}_channel_{channel}_mean_intensity":
           [float(1 + channel * 7 + index * 2 + row) for row in range(8)]
           for channel in [0, 1]
           for index, compartment in enumerate(
               ["cell", "nucleus", "pathogen", "cytoplasm"])},
    })
    variant = getattr(request, "param", "")
    if variant == "paired irregular":
        frame = pd.concat([frame, frame.assign(source_object_id=frame["source_object_id"] + 8)],
                          ignore_index=True)
        frame.loc[frame["source_object_id"] == 0, "cell_channel_0_mean_intensity"] = float("nan")
        frame.loc[frame["source_object_id"] == 1, "nucleus_channel_0_mean_intensity"] = float("nan")
    if variant == "single condition":
        frame = frame.loc[frame["condition"] == "control"].reset_index(drop=True)
    if variant == "missing compartment":
        frame = frame.drop(columns=[column for column in frame.columns
                                    if column.startswith("cytoplasm_")])
    if variant == "missing channel":
        frame = frame.drop(columns=[column for column in frame.columns
                                    if "_channel_1_" in column])
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    plot._plot_controls(frame, [0], 1, figuresize=3)
    assert len(figures) == 1
    try:
        yield frame, figures[0]
    finally:
        plt.close(figures[0])


def test_standalone_controls_preserve_panel_slots_compartment_values_and_source(
        controls_figure, tmp_path):
    original_frame, figure = controls_figure
    expected = [axes for axes in figure.axes if axes.has_data()]
    assert len(expected) == 4
    path = bundle._save_zip(figure, str(tmp_path / "controls.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
    pd.testing.assert_frame_equal(frame, original_frame)
    namespace = {"__name__": "standalone_controls"}
    exec(compile(script, "recreate_figure.py", "exec"), namespace)
    restored = Figure()
    namespace["_draw"](restored, frame, spec)
    actual = [axes for axes in restored.axes if axes.has_data()]
    assert len(actual) == 4
    for source, recreated in zip(expected, actual):
        assert recreated.get_title() == source.get_title()
        assert [tick.get_text() for tick in recreated.get_xticklabels()] == [
            tick.get_text() for tick in source.get_xticklabels()]
        assert [patch.get_height() for patch in recreated.patches] == pytest.approx(
            [patch.get_height() for patch in source.patches])
        assert recreated.get_subplotspec().rowspan.start == source.get_subplotspec().rowspan.start
        assert recreated.get_subplotspec().colspan.start == source.get_subplotspec().colspan.start


def test_retyping_controls_retains_each_condition_channel_and_compartment(
        controls_figure):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.figure_settings import _retype

    _frame, figure = controls_figure
    expected = [(axes.get_title(), [tick.get_text() for tick in axes.get_xticklabels()])
                for axes in figure.axes if axes.has_data()]
    assert _retype(figure, "box")
    actual = [(axes.get_title(), [tick.get_text() for tick in axes.get_xticklabels()])
              for axes in figure.axes if axes.has_data()]
    assert actual == expected


def test_controls_statistics_identify_each_condition_and_channel(
        controls_figure, tmp_path):
    _frame, figure = controls_figure
    expected = {axes.get_title() for axes in figure.axes if axes.has_data()}
    assert len(expected) == 4
    path = bundle._save_zip(figure, str(tmp_path / "statistics.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        statistics = pd.read_csv(archive.open("statistics.csv"))
    assert "measurement" in statistics.columns
    assert set(statistics["measurement"]) == expected
    normality = statistics.loc[statistics["test_stage"] == "normality"]
    assert len(normality) == 16
    assert set(normality["groups"]) == {"cell", "nucleus", "pathogen", "cytoplasm"}
    assert set(pd.to_numeric(normality["n"])) == {4}


@pytest.mark.parametrize("controls_figure", [
    "single condition", "missing compartment", "missing channel",
], indirect=True)
def test_controls_recipe_handles_single_condition_and_incomplete_measurements(
        controls_figure, tmp_path):
    original_frame, figure = controls_figure
    expected = [axes for axes in figure.axes if axes.patches]
    path = bundle._save_zip(figure, str(tmp_path / "partial.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
    pd.testing.assert_frame_equal(frame, original_frame)
    namespace = {"__name__": "standalone_partial_controls"}
    exec(compile(script, "recreate_figure.py", "exec"), namespace)
    restored = Figure()
    namespace["_draw"](restored, frame, spec)
    actual = [axes for axes in restored.axes if axes.patches]
    assert len(actual) == len(expected)
    for source, recreated in zip(expected, actual):
        assert recreated.get_title() == source.get_title()
        assert [tick.get_text() for tick in recreated.get_xticklabels()] == [
            tick.get_text() for tick in source.get_xticklabels()]
        assert [patch.get_height() for patch in recreated.patches] == pytest.approx(
            [patch.get_height() for patch in source.patches])
        assert recreated.get_subplotspec().rowspan.start == source.get_subplotspec().rowspan.start
        assert recreated.get_subplotspec().colspan.start == source.get_subplotspec().colspan.start


@pytest.mark.parametrize("slot", [-1, 1000000, "0", 1])
def test_invalid_controls_panel_slot_preserves_the_current_figure(
        controls_figure, slot):
    frame, figure = controls_figure
    original_axes = tuple(figure.axes)
    _data, spec = bundle._figure_record(figure)
    panels = [dict(panel) for panel in spec["panels"]]
    panels[0]["slot"] = slot
    with pytest.raises(ValueError, match="multi-panel figure layout"):
        bundle._draw(figure, frame, dict(spec, panels=panels))
    assert tuple(figure.axes) == original_axes
    assert sum(bool(axes.patches) for axes in figure.axes) == 4


@pytest.mark.parametrize("controls_figure", ["paired irregular"], indirect=True)
def test_controls_paired_statistics_match_objects_after_missing_measurements(
        controls_figure, tmp_path):
    from scipy.stats import friedmanchisquare

    frame, figure = controls_figure
    _data, spec = bundle._figure_record(figure)
    spec["stats"] = {"test": "Friedman", "paired": True, "pair": "source_object_id"}
    path = bundle._save_zip(figure, str(tmp_path / "paired.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        statistics = pd.read_csv(archive.open("statistics.csv"))
    tested = statistics.loc[(statistics["test_stage"] == "omnibus")
                            & (statistics["test_name"] == "Friedman")]
    assert len(tested) == 4
    for index, panel in enumerate(spec["panels"]):
        source_rows = frame.loc[frame["condition"] == panel["where"]["condition"][0]]
        matched = source_rows.set_index("source_object_id")[panel["melt"]["columns"]].dropna()
        reference = friedmanchisquare(*(matched[column] for column in matched.columns))
        result = tested.loc[tested["panel"] == index].iloc[0]
        assert result["chosen_by"] == "auto"
        assert result["statistic"] == pytest.approx(reference.statistic)
        assert result["p_value"] == pytest.approx(reference.pvalue)
