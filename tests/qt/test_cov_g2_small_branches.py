"""Single branches in several modules that no other test happened to take."""
from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMainWindow, QMenu  # noqa: E402

from tests.qt.test_arrayed_plan_validation_f585 import (  # noqa: E402,F401
    pilot_and_plan, planner)


def test_a_pinned_empty_kind_with_columns_has_no_binding_error():
    from spacr.qt.widgets import graph_spec as gs

    spec = gs.GraphSpec(x="a", y="b", kind=gs.EMPTY)
    assert spec.binding_error({"a": gs.CONTINUOUS, "b": gs.CONTINUOUS}) == ""


def test_imagej_roi_property_lines_without_a_colon_are_skipped():
    from spacr.mask_io import _roi_props

    roi = types.SimpleNamespace(props="name: cell 4\njust text\nkind:ellipse")
    assert _roi_props(roi) == {"name": "cell 4", "kind": "ellipse"}
    assert _roi_props(types.SimpleNamespace()) == {}


def test_a_barcode_bundle_with_no_mismatches_shows_only_the_summary(tmp_path):
    from spacr.qt.screens.convert import _barcode_summary

    (tmp_path / "complete.json").write_text(json.dumps(
        {"linked_wells": 96, "mismatches": 0}))
    (tmp_path / "plate_barcode_mismatches.csv").write_text("well,barcode\n")
    text = _barcode_summary({"bundle": tmp_path})
    assert "96" in text and "\n\n" not in text
    assert _barcode_summary(None) == ""


def test_a_registry_built_while_waiting_for_the_lock_is_used(monkeypatch):
    from spacr import plugins

    built = object()

    class _Lock:
        def __enter__(self):
            monkeypatch.setattr(plugins, "_REGISTRY", built)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(plugins, "_REGISTRY", None)
    monkeypatch.setattr(plugins, "_CATALOGUE_MUTATION_LOCK", _Lock())
    monkeypatch.setattr(plugins, "reload_plugins",
                        lambda: pytest.fail("built twice"))
    assert plugins._registry() is built


def test_explicit_help_survives_a_preference_change(qtbot):
    from spacr.qt.widgets.availability_panel import AvailabilityPanel

    panel = AvailabilityPanel.instance()
    hidden = []
    panel._pinned = True
    original = panel.hide
    panel.hide = lambda: hidden.append(True)
    try:
        panel._cancel_hover()
        assert hidden == []
        panel._pinned = False
        panel._cancel_hover()
        assert hidden == [True]
    finally:
        panel.hide = original
        panel._pinned = False


def test_a_menu_that_is_on_no_menu_bar_aims_at_the_bar(qtbot, monkeypatch):
    from spacr.qt.tutorial import scripts

    window = QMainWindow()
    qtbot.addWidget(window)
    orphan = QMenu("Orphan")
    monkeypatch.setattr(scripts, "_find_menu", lambda w, title: orphan)
    monkeypatch.setattr(scripts, "_top_level_menu_containing",
                        lambda w, menu: None)
    bar, point = scripts._menu_target(window, "Orphan")
    assert bar is window.menuBar() and point is None


def test_an_older_plan_without_optional_counts_still_validates(planner,
                                                              pilot_and_plan):
    import copy

    plan = copy.deepcopy(pilot_and_plan[1])
    for key in ("cells_per_field", "cells_per_field_effective",
                "fields_per_well", "wells_per_replicate", "n_replicates",
                "n_conditions", "n_wells", "n_cells"):
        plan["variance_components"].pop(key, None)
    for row in plan["designs"]:
        row.pop("cells_per_field", None)
        row.pop("cells_per_condition", None)
    inputs, pilot, values = planner._validate_arrayed_plan(plan)[:3]
    assert inputs["effect"] == 2.0 and pilot["table"] == "cell" and values


def test_test_data_without_masks_keeps_a_destination_already_chosen(qtbot,
                                                                    tmp_path):
    from PySide6.QtWidgets import QLineEdit, QWidget

    from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget

    class _Model:
        def __init__(self):
            self._widgets = {"dst": QLineEdit("/already/here")}
            self.set = []

        def _read_widget(self, widget):
            return widget.text()

        def set_value_for_key(self, key, value):
            self.set.append((key, value))

    holder = QWidget()
    holder._settings_model = _Model()
    qtbot.addWidget(holder)
    table = ExternalMaskInputWidget(parent=holder)
    images = tmp_path / "images"
    images.mkdir()
    table._use_test_data({"images": images, "masks": {}})
    assert holder._settings_model.set == []
