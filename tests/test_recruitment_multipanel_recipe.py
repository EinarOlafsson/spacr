"""Exported recruitment recipes preserve every displayed measurement panel."""
from __future__ import annotations

import json
import zipfile

from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from spacr import plot
from spacr.figures import bundle


@pytest.fixture
def recruitment_figures(monkeypatch, tmp_path):
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
    columns = [f"{name}_channel_0_mean_intensity"
               for name in ("cell", "nucleus", "cytoplasm", "pathogen")]
    columns.extend(
        f"pathogen_channel_0_cytoplasm_{stat}_ratio"
        for stat in ("mean", "q75", "periphery_mean", "outside_mean", "outside_q75"))
    frame = pd.DataFrame({
        "condition": ["control"] * 4 + ["treated"] * 4,
        "pathogen": ["A", "B"] * 4,
        **{name: [float(2 + index + row) for row in range(8)]
           for index, name in enumerate(columns)},
    })
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    plot._plot_recruitment(frame, "fixture", 0, figuresize=6)
    assert len(figures) == 2
    try:
        yield frame, figures
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("figure_index, panel_count", [(0, 4), (1, 5)])
def test_standalone_recruitment_recipe_restores_every_measurement(
        recruitment_figures, tmp_path, figure_index, panel_count):
    source_frame, figures = recruitment_figures
    original = figures[figure_index]
    expected = [axes.get_ylabel() for axes in original.axes if axes.has_data()]
    assert len(expected) == panel_count
    path = bundle._save_zip(
        original, str(tmp_path / "recruitment.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        exported = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
    pd.testing.assert_frame_equal(exported, source_frame)
    namespace = {"__name__": "standalone_recruitment_recipe"}
    exec(compile(script, "recreate_figure.py", "exec"), namespace)
    recreated = Figure()
    namespace["_draw"](recreated, exported, spec)
    actual = [axes.get_ylabel() for axes in recreated.axes if axes.has_data()]
    assert actual == expected


@pytest.mark.parametrize("figure_index, panel_count", [(0, 4), (1, 5)])
def test_retyping_recruitment_keeps_every_panel_and_changes_each_graph(
        recruitment_figures, figure_index, panel_count):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.figure_settings import _retype

    _frame, figures = recruitment_figures
    figure = figures[figure_index]
    expected = [axes.get_ylabel() for axes in figure.axes if axes.has_data()]
    assert _retype(figure, "box")
    axes = [item for item in figure.axes if item.has_data()]
    assert len(axes) == panel_count
    assert [item.get_ylabel() for item in axes] == expected
    assert all(any(isinstance(patch, PathPatch) for patch in item.patches)
               for item in axes)


@pytest.mark.parametrize("figure_index, panel_count", [(0, 4), (1, 5)])
def test_exported_recruitment_statistics_identify_every_measurement(
        recruitment_figures, tmp_path, figure_index, panel_count):
    _frame, figures = recruitment_figures
    figure = figures[figure_index]
    expected = {axes.get_ylabel() for axes in figure.axes if axes.has_data()}
    assert len(expected) == panel_count
    path = bundle._save_zip(
        figure, str(tmp_path / "statistics.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        statistics = pd.read_csv(archive.open("statistics.csv"))
    assert "measurement" in statistics.columns
    assert set(statistics["measurement"].dropna()) == expected


@pytest.mark.parametrize("figure_index", [0, 1])
def test_recruitment_statistics_override_uses_each_measurements_values(
        recruitment_figures, tmp_path, figure_index):
    from scipy.stats import ttest_ind

    frame, figures = recruitment_figures
    figure = figures[figure_index]
    _data, spec = bundle._figure_record(figure)
    spec["stats"] = {"test": "Welch's t"}
    expected = {axes.get_ylabel() for axes in figure.axes if axes.has_data()}
    path = bundle._save_zip(
        figure, str(tmp_path / "override.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        table = pd.read_csv(archive.open("statistics.csv"))
    assert "measurement" in table.columns
    pairwise = table.loc[table["test_stage"] == "pairwise"]
    assert set(pairwise["measurement"]) == expected
    assert len(pairwise) == len(expected)
    for _index, row in pairwise.iterrows():
        measured = row["measurement"]
        reference = ttest_ind(
            frame.loc[frame["condition"] == "control", measured],
            frame.loc[frame["condition"] == "treated", measured],
            equal_var=False)
        assert row["test_name"] == "Welch's t"
        assert row["chosen_by"] == "user"
        assert row["statistic"] == pytest.approx(reference.statistic)
        assert row["p_value"] == pytest.approx(reference.pvalue)


@pytest.mark.parametrize("grid", [(0, 4), (1, 0), (1, 1000000), (1, 1)])
def test_invalid_panel_layout_preserves_the_original_figure(
        recruitment_figures, grid):
    frame, figures = recruitment_figures
    figure = figures[0]
    original_axes = tuple(figure.axes)
    _data, spec = bundle._figure_record(figure)
    invalid = dict(spec, grid=grid)
    with pytest.raises(ValueError, match="multi-panel figure layout"):
        bundle._draw(figure, frame, invalid)
    assert tuple(figure.axes) == original_axes
    assert all(axes.has_data() for axes in figure.axes)


@pytest.mark.parametrize("figure_index", [0, 1])
def test_exported_recipe_keeps_each_panels_edited_title_and_limits(
        recruitment_figures, tmp_path, figure_index):
    _frame, figures = recruitment_figures
    figure = figures[figure_index]
    axes = [item for item in figure.axes if item.has_data()]
    for index, item in enumerate(axes):
        item.set_title(f"Edited panel {index}")
        item.set_xlim(-0.75, 1.5 + index)
        item.set_ylim(float(index), 12.5 + index)
    _data, spec = bundle._figure_record(figure)
    spec["keep_limits"] = True
    path = bundle._save_zip(
        figure, str(tmp_path / "edited.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        saved = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
    namespace = {"__name__": "standalone_edited_recipe"}
    exec(compile(script, "recreate_figure.py", "exec"), namespace)
    recreated = Figure()
    namespace["_draw"](recreated, frame, saved)
    actual = [item for item in recreated.axes if item.has_data()]
    assert len(actual) == len(axes)
    for expected, restored in zip(axes, actual):
        assert restored.get_title() == expected.get_title()
        assert restored.get_xlim() == pytest.approx(expected.get_xlim())
        assert restored.get_ylim() == pytest.approx(expected.get_ylim())
