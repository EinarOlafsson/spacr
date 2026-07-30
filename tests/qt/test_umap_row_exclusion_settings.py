"""UMAP exposes general database row exclusions instead of lab plate presets."""

from __future__ import annotations

import sqlite3

import pandas as pd
from PySide6.QtCore import Qt

from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.qt.widgets.row_exclusion import RowExclusionEditor


def _measurements_source(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p2"],
        "columnID": ["c1", "c2", "c1"],
        "object_label": [1, 2, 3],
        "cell_channel_0_mean_intensity": [1.0, 2.0, 3.0],
    })
    with sqlite3.connect(measurements / "measurements.db") as connection:
        frame.to_sql("cell", connection, index=False)
    return tmp_path


def test_umap_plate_section_contains_only_general_exclusion(qtbot):
    model = SettingsWidgets("umap")
    sections = dict(model.build_sections())

    plate_labels = [label for label, _widget in
                    sections["Plate Layout & Controls"]]
    assert plate_labels == ["Exclude"]
    assert isinstance(model._widgets["exclude_rows"], RowExclusionEditor)
    for legacy in (
        "col_to_compare", "pos", "neg", "mix",
        "embedding_by_controls", "exclude_conditions",
    ):
        assert legacy not in model._widgets

    measurement_labels = [label for label, _widget in sections["Measurements"]]
    assert "Exclude features" in measurement_labels


def test_umap_display_embedding_plot_and_advanced_settings_are_panel_sections(
        qtbot):
    model = SettingsWidgets("umap")
    sections = dict(model.build_sections())

    assert {"UMAP Display", "Embedding & Clustering", "Advanced"} <= set(
        sections)
    display = {label for label, _widget in sections["UMAP Display"]}
    embedding = {
        label for label, _widget in sections["Embedding & Clustering"]}
    assert {"Point color", "Point alpha", "Plot images",
            "Plot cluster grids"} <= display
    assert {"N neighbors", "Min dist", "Clustering", "Eps",
            "Min samples"} <= embedding
    assert "https://" in model.plain_tooltip_for("plot_images")


def test_exclusion_editor_loads_columns_and_values_from_dropped_source(
    qtbot,
    tmp_path,
):
    source = _measurements_source(tmp_path)
    model = SettingsWidgets("umap")
    model.build_sections()
    assert model.set_value_for_key("tables", ["cell"])
    assert model.set_value_for_key("src", str(source))

    editor = model._widgets["exclude_rows"]
    row = editor._rows[0]
    column_index = row.column.findText("columnID")
    assert column_index >= 0
    row.column.setCurrentIndex(column_index)

    value_model = row.values.model()
    c1 = next(
        value_model.item(index)
        for index in range(value_model.rowCount())
        if value_model.item(index).text() == "c1"
    )
    c1.setCheckState(Qt.Checked)

    assert model.collect()["exclude_rows"] == {"columnID": ["c1"]}


def test_exclusion_editor_round_trips_imported_rules(qtbot, tmp_path):
    source = _measurements_source(tmp_path)
    model = SettingsWidgets("umap")
    model.build_sections()
    model.set_value_for_key("tables", ["cell"])
    model.set_value_for_key("src", str(source))

    assert model.set_value_for_key(
        "exclude_rows",
        {"columnID": ["c2"], "plateID": ["p2"]},
    )
    assert model.collect()["exclude_rows"] == {
        "columnID": ["c2"],
        "plateID": ["p2"],
    }


def test_umap_none_filter_text_collects_as_no_filter(qtbot):
    model = SettingsWidgets("umap")
    model.build_sections()

    assert model.set_value_for_key("filter_by", "None")
    assert model.collect()["filter_by"] is None
