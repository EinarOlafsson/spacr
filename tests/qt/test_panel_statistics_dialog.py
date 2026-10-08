"""Statistics controls use the measurements and axes actually plotted."""
from __future__ import annotations

import json
import zipfile

from matplotlib.figure import Figure
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare

from spacr.figures import bundle
from spacr.qt.widgets.figure_settings import _StatisticsDialog
from tests.test_controls_multipanel_recipe import controls_figure


def test_controls_dialog_tests_every_displayed_measurement(qapp, controls_figure):
    frame, figure = controls_figure
    dialog = _StatisticsDialog(figure)
    try:
        assert dialog.test.findData("Friedman") >= 0
        assert set(dialog.table["measurement"]) == {
            axes.get_title() for axes in figure.axes if axes.has_data()}
        normality = dialog.table.loc[dialog.table["test_stage"] == "normality"]
        assert len(normality) == 16
        assert set(normality["groups"]) == {"cell", "nucleus", "pathogen", "cytoplasm"}
        assert set(pd.to_numeric(normality["n"])) == {4}
        for title in dialog.table["measurement"].unique():
            assert title in dialog.report.toPlainText()
        assert dialog.pair.findData("source_object_id") >= 0
        assert not frame.empty
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("controls_figure", ["paired irregular"], indirect=True)
def test_paired_dialog_apply_export_and_recreation_match_independent_statistics(
        qapp, controls_figure, tmp_path):
    frame, figure = controls_figure
    _data, original_spec = bundle._figure_record(figure)
    dialog = _StatisticsDialog(figure)
    try:
        dialog.pair.setCurrentIndex(dialog.pair.findData("source_object_id"))
        dialog.paired.setChecked(True)
        assert dialog.test.findData("Friedman") >= 0
        dialog.test.setCurrentIndex(dialog.test.findData("Friedman"))
        for index, panel in enumerate(original_spec["panels"]):
            source = frame.loc[frame["condition"] == panel["where"]["condition"][0]]
            matched = source.set_index("source_object_id")[panel["melt"]["columns"]].dropna()
            reference = friedmanchisquare(*(matched[column] for column in matched.columns))
            result = dialog.table.loc[(dialog.table["panel"] == index)
                                      & (dialog.table["test_stage"] == "omnibus")].iloc[0]
            assert result["test_name"] == "Friedman"
            assert result["statistic"] == pytest.approx(reference.statistic)
            assert result["p_value"] == pytest.approx(reference.pvalue)
            assert result["n"] == " / ".join([str(len(matched))] * len(matched.columns))
        dialog._apply()
        first_notes = []
        for panel in figure._spacr_spec["panels"]:
            axes = figure.axes[panel["slot"]]
            notes = [text.get_text() for text in axes.texts
                     if text.get_gid() == "spacr-stats" and "Friedman" in text.get_text()]
            assert len(notes) == 1
            first_notes.append(notes)
        dialog._apply()
        assert [[text.get_text() for text in figure.axes[panel["slot"]].texts
                 if text.get_gid() == "spacr-stats" and "Friedman" in text.get_text()]
                for panel in figure._spacr_spec["panels"]] == first_notes
        path = bundle._save_zip(figure, str(tmp_path / "paired.zip"), formats=["png"])
        with zipfile.ZipFile(path) as archive:
            saved_data = pd.read_csv(archive.open("data.csv"))
            saved_stats = pd.read_csv(archive.open("statistics.csv"), keep_default_na=False)
            saved_spec = json.loads(archive.read("spec.json"))
            script = archive.read("recreate_figure.py")
        pd.testing.assert_frame_equal(saved_data, frame)
        assert saved_stats["test_name"].tolist() == dialog.table["test_name"].tolist()
        assert pd.to_numeric(saved_stats["p_value"]).tolist() == pytest.approx(
            pd.to_numeric(dialog.table["p_value"]).tolist(), nan_ok=True)
        namespace = {"__name__": "standalone_statistics"}
        exec(compile(script, "recreate_figure.py", "exec"), namespace)
        recreated = Figure()
        namespace["_draw"](recreated, saved_data, saved_spec)
        assert [[text.get_text() for text in recreated.axes[panel["slot"]].texts
                 if text.get_gid() == "spacr-stats" and "Friedman" in text.get_text()]
                for panel in saved_spec["panels"]] == first_notes
        dialog.show_on_plot.setChecked(False)
        dialog._apply()
        assert not any(artist.get_gid() == "spacr-stats"
                       for axes in figure.axes for artist in [*axes.texts, *axes.lines])
    finally:
        dialog.deleteLater()


@pytest.mark.parametrize("controls_figure", ["single condition", "missing channel"], indirect=True)
def test_apply_keeps_blank_control_slots_unannotated(qapp, controls_figure):
    _frame, figure = controls_figure
    dialog = _StatisticsDialog(figure)
    try:
        assert "panel" in dialog.table
        dialog._apply()
        for axes in figure.axes:
            if not axes.get_visible():
                assert not any(text.get_gid() == "spacr-stats" for text in axes.texts)
    finally:
        dialog.deleteLater()


def test_mixed_panel_families_offer_only_shared_overrides(qapp):
    frame = pd.DataFrame({"group": ["a"] * 4 + ["b"] * 4,
                          "x": [1., 2., 3., 4., 5., 6., 7., 8.],
                          "y": [3., 1., 4., 2., 9., 5., 8., 6.]})
    figure = Figure()
    bundle._register_figure_data(
        figure, frame, kind="box", grid=[1, 2], panels=[
            {"x": "group", "y": "y", "kind": "box", "measurement": "groups"},
            {"x": "x", "y": "y", "kind": "scatter", "measurement": "correlation"},
        ])
    bundle._draw(figure, frame, figure._spacr_spec)
    dialog = _StatisticsDialog(figure)
    try:
        assert dialog.test.count() == 1
        assert set(dialog.table["measurement"]) == {"groups", "correlation"}
        assert "correlation" in set(dialog.table["test_stage"])
        assert "normality" in set(dialog.table["test_stage"])
    finally:
        dialog.deleteLater()
