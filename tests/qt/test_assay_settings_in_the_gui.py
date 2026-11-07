"""Recruitment and replication show every setting their notebooks use.

The panel, filter, report and QC settings of the two assays were kept in
the run settings but drawn nowhere, so a figure set made in a notebook
could not be made from the GUI. They are now grouped into sections, and the
settings that hold records (panels, background filters, figures) get a
table editor that round-trips a settings file.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _model(qtbot, app_key):
    """A built settings form for ``app_key`` and its host widget."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt.screens.settings_model import SettingsWidgets

    host = QWidget()
    qtbot.addWidget(host)
    model = SettingsWidgets(app_key, host)
    sections = model.build_sections()
    return model, {title: [label for label, _w in rows]
                   for title, rows in sections}


ASSAY_KEYS = {
    "recruitment": (
        "panels", "figures", "condition_order", "condition_labels",
        "replicate_column", "background_filters", "min_solidity",
        "merged_dir", "background_mask_channel", "report", "report_layout",
        "report_tiles_per_condition", "report_tile_size", "report_tile_seed",
        "report_tile_channels", "report_max_conditions_per_row", "well_qc",
        "well_qc_mad", "well_qc_measures", "object_qc", "object_qc_mad",
        "object_qc_measures", "qc_group", "outlier_points",
        "outlier_points_method", "representative_crops",
        "representative_crops_n", "x_axis_style", "condition_factors",
        "factor_order", "bracket_pairs", "confounder_filter",
        "max_vacuoles_per_cell", "exclude_edge_objects", "control_color_by",
    ),
    "replication": (
        "panels", "figures", "condition_order", "condition_labels",
        "replicate_column", "background_filters", "min_solidity",
        "merged_dir", "background_mask_channel", "well_qc", "well_qc_mad",
        "well_qc_measures", "object_qc", "object_qc_mad",
        "object_qc_measures", "qc_group", "outlier_points",
        "outlier_points_method", "representative_crops",
        "representative_crops_n", "x_axis_style", "condition_factors",
        "factor_order", "bracket_pairs", "bin_labels", "max_area_quantile",
        "field_qc", "exclude_edge_objects",
    ),
}

SECTIONS = {
    "recruitment": ("Quality Filters", "Panels & Figures", "Report & QC"),
    "replication": ("Quality Filters", "Panels & Figures", "Panel QC"),
}


@pytest.mark.parametrize("app_key", sorted(ASSAY_KEYS))
def test_every_assay_setting_has_a_control(qtbot, app_key):
    """No assay setting is hidden, and each one gets a control."""
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    model, sections = _model(qtbot, app_key)
    assert not set(ASSAY_KEYS[app_key]) & _APP_HIDDEN_KEYS.get(app_key, set())
    for title in SECTIONS[app_key]:
        assert title in sections, (title, list(sections))
    for key in ASSAY_KEYS[app_key]:
        assert model._widgets.get(key) is not None, key


@pytest.mark.parametrize("app_key", sorted(ASSAY_KEYS))
def test_the_structured_settings_get_their_editors(qtbot, app_key):
    """Records get a table, dict and pair settings a checked literal field."""
    from spacr.qt.screens.settings_model import _LiteralEdit, _RecordTable

    model, _sections = _model(qtbot, app_key)
    for key in ("panels", "background_filters", "figures"):
        assert isinstance(model._widgets.get(key), _RecordTable), key
    for key in ("condition_labels", "condition_factors", "bracket_pairs"):
        assert isinstance(model._widgets.get(key), _LiteralEdit), key
    combo = model._widgets.get("x_axis_style")
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "auto", "matrix", "labels"]


def test_the_panel_builder_round_trips_a_settings_file(qtbot):
    """A panel list survives the form, keys without a column included."""
    model, _sections = _model(qtbot, "recruitment")
    panels = [
        {"name": "strains_in_WT", "where": {"host_cells": "HeLa_WT"},
         "group": "pathogen", "order": ["RH", "Me49"],
         "labels": {"RH": "RH WT"}, "ymin": 0},
        {"name": "hosts", "group": "host_cells", "value": "recruitment"},
    ]
    assert model.set_value_for_key("panels", repr(panels))
    assert model.collect()["panels"] == panels
    table = model._widgets["panels"]
    assert table.table.rowCount() == 2
    assert table.problems() == []


def test_figures_are_a_mapping_and_filters_a_list(qtbot):
    """``figures`` keys on the figure name; ``background_filters`` is rows."""
    model, _sections = _model(qtbot, "replication")
    figures = {"Figure_1": {"panels": ["P01", "P02"], "ncol": 2}}
    rules = [{"channel": 2, "statistic": "percentile_25", "max_quantile": 0.98},
             {"channel": 1, "min_ratio": 1.5}]
    model.set_value_for_key("figures", figures)
    model.set_value_for_key("background_filters", rules)
    collected = model.collect()
    assert collected["figures"] == figures
    assert collected["background_filters"] == rules
    model.set_value_for_key("background_filters", None)
    assert model.collect()["background_filters"] is None


def test_a_bad_cell_is_reported_and_kept(qtbot):
    """A number column holding text is listed under the table, not dropped."""
    from spacr.qt.screens.settings_model import _BACKGROUND_FILTER_COLUMNS, _RecordTable

    table = _RecordTable(_BACKGROUND_FILTER_COLUMNS, [{"channel": 1}])
    qtbot.addWidget(table)
    changes = []
    table.value_changed.connect(lambda: changes.append(True))
    table.table.item(0, 2).setText("high")
    assert changes
    assert table.get_value() == [{"channel": 1, "min_ratio": "high"}]
    assert len(table.problems()) == 1
    assert not table.status.isHidden()
    table._on_add()
    assert table.table.rowCount() == 2
    table.table.selectRow(1)
    table._on_remove()
    assert table.table.rowCount() == 1


def test_the_literal_field_checks_what_it_is_given(qtbot):
    """Dict settings mark text; bracket_pairs also takes a word."""
    from spacr.qt.screens.settings_model import _LiteralEdit

    labels = _LiteralEdit(None, "{'a': 'A'}")
    qtbot.addWidget(labels)
    assert labels.get_value() is None and labels.is_valid()
    labels.setText("{'HeLa_ALG2': 'HeLa ALG2 KO'}")
    assert labels.get_value() == {"HeLa_ALG2": "HeLa ALG2 KO"}
    labels.setText("HeLa_ALG2: KO")
    assert not labels.is_valid() and labels.property("invalid")
    pairs = _LiteralEdit("one_factor", "", text_allowed=True)
    qtbot.addWidget(pairs)
    assert pairs.get_value() == "one_factor" and pairs.is_valid()
    pairs.set_value([["a", "b"]])
    assert pairs.get_value() == [["a", "b"]]


def test_the_new_sections_have_their_own_help(qtbot):
    """Each new section has a written blurb, not the generic fallback."""
    from spacr.qt.screens.settings_model import _category_blurb

    for app_key, titles in SECTIONS.items():
        for title in titles:
            assert _category_blurb(app_key, title), (app_key, title)
