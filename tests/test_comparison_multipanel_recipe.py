"""Comparison ZIP recipes preserve metric families and their measurements."""
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
def comparison_figure(monkeypatch, tmp_path):
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
    records = [{
        "filename": f"field_{row}",
        **{f"{metric}_{comparison}": 0.1 + index * 0.05 + row * 0.02
           for index, metric in enumerate(
               ("jaccard", "dice", "boundary_f1", "average_precision"))
           for comparison in ("a_b", "a_c")},
    } for row in range(6)]
    for row, record in enumerate(records):
        for metric in ("jaccard", "dice", "boundary_f1", "average_precision"):
            record[f"{metric}_a_c"] += 0.04 + row * 0.01
    monkeypatch.setattr(plt, "show", lambda: None)
    figure = plot.plot_comparison_results(records)
    try:
        yield records, figure
    finally:
        plt.close(figure)


def test_standalone_comparison_recipe_restores_metric_panels_and_source_rows(
        comparison_figure, tmp_path):
    records, figure = comparison_figure
    path = bundle._save_zip(figure, str(tmp_path / "comparison.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
        spec = json.loads(archive.read("spec.json"))
        script = archive.read("recreate_figure.py")
    expected = pd.melt(pd.DataFrame(records), id_vars=["filename"],
                       var_name="metric", value_name="value")
    pd.testing.assert_frame_equal(frame, expected)
    namespace = {"__name__": "standalone_comparison"}
    exec(compile(script, "recreate_figure.py", "exec"), namespace)
    restored = Figure()
    namespace["_draw"](restored, frame, spec)
    actual = [axes for axes in restored.axes if axes.has_data()]
    assert len(actual) == 4
    for original, recreated in zip(figure.axes, actual):
        assert recreated.get_title() == original.get_title()
        assert recreated.get_ylabel() == original.get_ylabel()
        assert [tick.get_text() for tick in recreated.get_xticklabels()] == [
            tick.get_text() for tick in original.get_xticklabels()]


def test_retyping_comparison_changes_each_metric_graph_and_retains_its_categories(
        comparison_figure):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.figure_settings import _retype

    _records, figure = comparison_figure
    categories = [[tick.get_text() for tick in axes.get_xticklabels()]
                  for axes in figure.axes]
    labels = [axes.get_ylabel() for axes in figure.axes]
    assert _retype(figure, "box")
    actual = [axes for axes in figure.axes if axes.has_data()]
    assert len(actual) == 4
    assert [axes.get_ylabel() for axes in actual] == labels
    assert [[tick.get_text() for tick in axes.get_xticklabels()]
            for axes in actual] == categories
    assert all(any(isinstance(patch, PathPatch) for patch in axes.patches)
               for axes in actual)


def test_comparison_statistics_use_each_metric_family_without_cross_family_pairs(
        comparison_figure, tmp_path):
    from scipy.stats import ttest_ind

    records, figure = comparison_figure
    _frame, spec = bundle._figure_record(figure)
    spec["stats"] = {"test": "Welch's t"}
    path = bundle._save_zip(figure, str(tmp_path / "statistics.zip"), formats=["png"])
    with zipfile.ZipFile(path) as archive:
        statistics = pd.read_csv(archive.open("statistics.csv"))
    assert "measurement" in statistics.columns
    pairwise = statistics.loc[statistics["test_stage"] == "pairwise"]
    assert len(pairwise) == 4
    assert set(pairwise["measurement"]) == {axes.get_ylabel() for axes in figure.axes}
    for _index, row in pairwise.iterrows():
        first, second = row["groups"].split(" vs ", 1)
        assert first.rsplit("_a_", 1)[0] == second.rsplit("_a_", 1)[0]
        reference = ttest_ind([record[first] for record in records],
                             [record[second] for record in records], equal_var=False)
        assert row["test_name"] == "Welch's t"
        assert row["statistic"] == pytest.approx(reference.statistic)
        assert row["p_value"] == pytest.approx(reference.pvalue)
