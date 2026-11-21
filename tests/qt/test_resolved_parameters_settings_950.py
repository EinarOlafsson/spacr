"""Saved resolution records survive real form and settings-file boundaries."""
import copy
import json

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from spacr.cli import load_settings_file, write_settings_file
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.settings import (RESOLVED_PARAMETERS_KEY, SETTINGS_SCHEMA_VERSION,
                            SETTINGS_VERSION_KEY)


class _RecordingPayload(dict):
    """Represent a runtime dict marker that saved GUI records must not retain."""


def _payload():
    """Return exact per-field resolutions with a native-scale null value."""
    return {"version": 1, "fields": {"plate1/A/1/0": {
        "normalization": {
            "source": {"path": "001.npy", "identity": "None"},
            "configuration": {"percentiles": [1.0, 99.5]},
            "value": [[0.00000000000000017, 65535.125]],
        },
        "diameter:nucleus": {
            "source": {"path": "001.npy"},
            "configuration": {"mode": "per_image"},
            "value": 26.245000000000005,
        },
        "diameter:cell": {
            "source": {"path": "001.npy"},
            "configuration": {"mode": "native"}, "value": None,
        },
    }}}


def _form(qtbot, values):
    """Build a mask form with normal controls and owned Qt lifetime."""
    parent = QWidget()
    qtbot.addWidget(parent)
    return SettingsWidgets("mask", parent, current=values)


def _refuse_warning(*args):
    """Fail an import or export that only appears successful in a dialog."""
    pytest.fail(str(args[2]))


def test_form_rebuild_keeps_plain_independent_hidden_resolutions(qtbot):
    """Rebuilt forms preserve exact data without the runtime recording marker."""
    expected = _payload()
    supplied = _RecordingPayload(copy.deepcopy(expected))
    supplied.recording = True
    supplied["fields"] = _RecordingPayload(supplied["fields"])
    form = _form(qtbot, {RESOLVED_PARAMETERS_KEY: supplied})
    returned = form.collect()
    assert returned[RESOLVED_PARAMETERS_KEY] == expected
    assert type(returned[RESOLVED_PARAMETERS_KEY]) is dict
    assert type(returned[RESOLVED_PARAMETERS_KEY]["fields"]) is dict
    assert RESOLVED_PARAMETERS_KEY not in form._widgets
    assert RESOLVED_PARAMETERS_KEY not in form._defaults
    supplied["fields"].clear()
    returned[RESOLVED_PARAMETERS_KEY]["fields"].clear()
    assert form.collect()[RESOLVED_PARAMETERS_KEY] == expected
    rebuilt = _form(qtbot, form.collect())
    assert rebuilt.collect()[RESOLVED_PARAMETERS_KEY] == expected
    assert RESOLVED_PARAMETERS_KEY not in rebuilt._widgets


@pytest.mark.parametrize("with_saved_records", [False, True],
                         ids=["fresh-file-clears", "saved-file-replaces"])
def test_real_import_distinguishes_fresh_files_from_partial_edits(
        qtbot, monkeypatch, tmp_path, with_saved_records):
    """Only an explicit fresh-file import clears prior resolved parameters."""
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    expected = _payload()
    screen.apply_settings_dict({RESOLVED_PARAMETERS_KEY: expected})
    expected["fields"].clear()
    assert screen._settings_model.collect()[RESOLVED_PARAMETERS_KEY] == _payload()
    screen.apply_settings_dict({"src": str(tmp_path / "partial-edit")})
    assert screen._settings_model.collect()[RESOLVED_PARAMETERS_KEY] == _payload()

    imported = {"src": str(tmp_path / "imported")}
    if with_saved_records:
        replacement = _payload()
        replacement["fields"]["plate1/A/1/0"]["diameter:nucleus"]["value"] = None
        imported[RESOLVED_PARAMETERS_KEY] = replacement
    destination = tmp_path / "import.csv"
    write_settings_file(imported, destination)

    def choose_import(*args, **kwargs):
        """Choose the written settings file through the normal import action."""
        return str(destination), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_import)
    monkeypatch.setattr(QMessageBox, "warning", _refuse_warning)
    screen._on_import_settings()
    collected = screen._settings_model.collect()
    assert collected["src"] == imported["src"]
    if with_saved_records:
        assert collected[RESOLVED_PARAMETERS_KEY] == replacement
    else:
        assert RESOLVED_PARAMETERS_KEY not in collected
    assert RESOLVED_PARAMETERS_KEY not in screen._settings_model._widgets


def test_real_gui_export_preserves_exact_resolutions_in_csv_and_json(
        qtbot, monkeypatch, tmp_path):
    """Normal GUI export writes the exact hidden payload in both file formats."""
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    expected = _payload()
    screen.apply_settings_dict({"src": str(tmp_path), RESOLVED_PARAMETERS_KEY: expected})
    destination = tmp_path / "export.csv"

    def choose_export(*args, **kwargs):
        """Choose an owned output path through the normal export action."""
        return str(destination), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_export)
    monkeypatch.setattr(QMessageBox, "warning", _refuse_warning)
    screen._on_export_settings()
    csv_values = load_settings_file(str(destination))
    json_values = json.loads(destination.with_suffix(".json").read_text("utf-8"))
    gui_values = AppScreen._load_settings_csv(str(destination))
    for values in (csv_values, json_values, gui_values):
        assert values[RESOLVED_PARAMETERS_KEY] == expected
        assert values[SETTINGS_VERSION_KEY] == SETTINGS_SCHEMA_VERSION
    assert screen._settings_model.collect()[RESOLVED_PARAMETERS_KEY] == expected
