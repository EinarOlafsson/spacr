"""The dilution field accepts plate mappings even with a numeric default."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QLineEdit

from spacr.qt.screens.settings_model import SettingsWidgets


def test_dilution_field_collects_csv_mapping_and_fraction(qapp, tmp_path):
    model = SettingsWidgets("analyze_plaques")
    widget = model._widget_for("entry", None, 1, "colony_dilution")
    assert isinstance(widget, QLineEdit)
    model._widgets["colony_dilution"] = widget
    assert model.collect()["colony_dilution"] == 1
    path = str(tmp_path / "plate dilutions.csv")
    for text, expected in (
        (path, path),
        ("{'plate1.png': 10000, 'plate2': 0.0001}",
         {"plate1.png": 10000, "plate2": 0.0001}),
        ("0.00000001", 0.00000001),
    ):
        widget.setText(text)
        assert model.collect()["colony_dilution"] == expected
    widget.deleteLater()


def test_unopened_dilution_control_preserves_csv_path(qapp, tmp_path):
    model = SettingsWidgets("analyze_plaques")
    path = str(tmp_path / "plate dilutions.csv")
    model._defaults["colony_dilution"] = path
    route, plan = model._route_control("entry", None, 1, "colony_dilution")
    assert route == "plain"
    model._widgets.wait_for("colony_dilution", plan)
    assert not model._widgets.is_built("colony_dilution")
    assert model.collect()["colony_dilution"] == path
    assert not model._widgets.is_built("colony_dilution")
