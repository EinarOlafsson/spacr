"""Item 693 (issues #140, #141, #142): Export settings… on every settings form.

Every screen with "Import settings…" has "Export settings…" right after it.
Export writes what Run collects from the form to a CSV in the layout
``spacr.utils.save_settings`` writes, without starting a run or listing the
source folder, and Import reads that CSV back to the same form values. The
CSV also loads with the non-GUI loaders and resolves with ``spacr-run``.
Distributed Jobs can save the snapshot "Submit remote…" sends as the same CSV.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import pathlib
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFileDialog, QMessageBox

from spacr.qt.screens import app_screen as app_screen_module
from spacr.qt.screens.app_screen import (
    APP_TITLES,
    AppScreen,
    default_settings_csv_path,
)

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]


def _app_screen_keys():
    """Every module key the window renders as a settings form.

    The window builds an :class:`AppScreen` for each titled key that its
    screen factory does not route to a dedicated screen, and the Cellpose
    workbench builds one per tab. A key with a registered factory is a
    dedicated screen too; reading the registry from ``spacr.qt.app`` also
    fixes ``APP_TITLES``, which that import extends, whatever ran first.
    """
    from spacr.qt.app import APP_FACTORIES

    source = (ROOT / "spacr" / "qt" / "app.py").read_text(encoding="utf-8")
    body = source.split("def _build_screen_timed", 1)[1].split(
        "screen = AppScreen(app_key=key)", 1)[0]
    dedicated = set(re.findall(r'if key == "([a-z_]+)"', body))
    from spacr.qt.screens.train_cellpose import TABS

    keys = [key for key in APP_TITLES
            if key not in dedicated and key not in APP_FACTORIES]
    keys += [key for key, _label, _hint in TABS if key not in keys]
    return keys


SCREEN_KEYS = _app_screen_keys()

PLATE = "/mnt/cluster share/plate 1"


def _screen(qtbot, key):
    """One settings form, closed with the test."""
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    return screen


def _console_text(console) -> str:
    """Everything the console shows."""
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.toPlainText()
                     for block in console.findChildren(_StdoutBlock))


def _non_default(values: dict) -> dict:
    """A value unlike the form's for every setting of a plain type."""
    out = {}
    for key, value in values.items():
        if key == "src":
            out[key] = PLATE
        elif isinstance(value, bool):
            out[key] = not value
        elif isinstance(value, int):
            out[key] = value + 1
        elif isinstance(value, float):
            out[key] = round(value + 0.25, 6)
        elif (isinstance(value, list) and value
              and all(type(item) is int for item in value)):
            out[key] = value + [max(value) + 1]
    return out


def _refuse_listing(patcher):
    """Make listing a folder or starting a run fail the test."""
    def refuse(*_args, **_kwargs):
        raise AssertionError("Export listed a folder or started a run")

    for name in ("listdir", "scandir", "walk"):
        patcher.setattr(os, name, refuse)
    for name in ("glob", "iglob"):
        patcher.setattr(glob, name, refuse)
    for name in ("iterdir", "glob", "rglob"):
        patcher.setattr(pathlib.Path, name, refuse)
    patcher.setattr(app_screen_module, "resolve_pipeline_entry", refuse)
    patcher.setattr(app_screen_module, "make_thread", refuse)


def _save_to(monkeypatch, path):
    """Answer the save dialog with ``path`` and record what it offered."""
    offered = []

    def answer(_parent, _caption, default, _filter):
        offered.append(default)
        return str(path), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(answer))
    return offered


def _open(monkeypatch, path):
    """Answer the open dialog with ``path``."""
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))


def _warnings(monkeypatch):
    """Record warning boxes instead of opening them."""
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: seen.append(args)))
    return seen


def _export_round_trip(qtbot, monkeypatch, tmp_path, key):
    """Set non-default values, export, import into a fresh form.

    :returns: (values at export, values after import, CSV path, defaults).
    """
    screen = _screen(qtbot, key)
    defaults = screen._settings_model.collect()
    screen.apply_settings_dict(_non_default(defaults))
    expected = screen._settings_model.collect()
    path = tmp_path / f"{key}_settings.csv"
    _save_to(monkeypatch, path)
    with monkeypatch.context() as guard:
        _refuse_listing(guard)
        screen._on_export_settings()
    assert path.is_file()
    fresh = _screen(qtbot, key)
    _open(monkeypatch, path)
    fresh._on_import_settings()
    return expected, fresh._settings_model.collect(), path, defaults


def test_every_settings_form_is_covered():
    """The list the tests walk holds the modules the issue names."""
    assert {"mask", "measure", "classify_merged", "umap", "train_cellpose",
            "cellpose_masks"} <= set(SCREEN_KEYS)
    assert not {"annotate", "make_masks", "distributed_jobs",
                "queue"} & set(SCREEN_KEYS)


@pytest.mark.parametrize("key", SCREEN_KEYS)
def test_export_sits_right_after_import(qtbot, key):
    """Export settings… is the button after Import settings…, with help."""
    screen = _screen(qtbot, key)
    strip = screen._actions_row.layout().itemAt(0).layout()
    assert (strip.indexOf(screen._btn_export)
            == strip.indexOf(screen._btn_import) + 1)
    assert screen._btn_export.text() == "Export settings…"
    assert screen._btn_export.objectName() == screen._btn_import.objectName()
    assert "spacr-run" in screen._action_hint(screen._btn_export)


@pytest.mark.parametrize("key", SCREEN_KEYS)
def test_export_then_import_gives_back_the_same_form(
        qtbot, monkeypatch, tmp_path, key):
    """Non-default values of every type survive Export, a reset and Import."""
    expected, imported, path, defaults = _export_round_trip(
        qtbot, monkeypatch, tmp_path, key)
    if "model_metadata" in imported and "model_metadata" not in expected:
        from spacr.model_zoo import BUNDLED_REMOTE_MODELS

        assert key == "analyze_plaques"
        records = imported["model_metadata"]["settings"]
        assert records
        for name, record in records.items():
            assert record["requested_value"] == expected[name]
            published = next(row for row in BUNDLED_REMOTE_MODELS if row["key"] == expected[name])
            assert record["sha256"] == published["sha256"]
            assert record["filename"] == published["name"]
        expected = dict(expected, model_metadata=imported["model_metadata"])
    changed = {type(value).__name__ for name, value in expected.items()
               if defaults.get(name) != value}
    plain = {type(value).__name__ for value in defaults.values()
             if type(value) in (bool, int)}
    assert plain and plain <= changed
    drifted = {name: (expected.get(name), imported.get(name))
               for name in set(expected) | set(imported)
               if expected.get(name) != imported.get(name)}
    assert drifted == {}
    assert PLATE in str(expected.get("src", PLATE))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["Key", "Value"]
    assert [row[0] for row in rows[1:]] == list(expected)


def test_measure_round_trip_covers_every_value_type(
        qtbot, monkeypatch, tmp_path):
    """Measure, the screen of issue #140, carries every kind of value."""
    expected, imported, path, defaults = _export_round_trip(
        qtbot, monkeypatch, tmp_path, "measure")
    changed = {type(value).__name__ for name, value in expected.items()
               if defaults.get(name) != value}
    assert {"bool", "int", "float", "str", "list"} <= changed
    assert any(value is None for value in expected.values())
    assert imported == expected
    text = path.read_text(encoding="utf-8")
    assert f'src,{PLATE}\n' in text


def test_the_csv_loads_with_the_non_gui_loaders(qtbot, monkeypatch,
                                                tmp_path):
    """spacr.cli and spacr.utils read the export; the JSON keeps types."""
    from spacr.cli import load_settings_file
    from spacr.utils import load_settings

    expected, _imported, path, _defaults = _export_round_trip(
        qtbot, monkeypatch, tmp_path, "measure")
    for loaded in (load_settings_file(str(path)),
                   load_settings(str(path), setting_key="Key",
                                 setting_value="Value")):
        assert list(loaded) == list(expected)
        assert loaded == expected
    sibling = path.with_suffix(".json")
    assert json.loads(sibling.read_text(encoding="utf-8")) == json.loads(
        json.dumps(expected))
    assert load_settings_file(str(sibling)) == json.loads(
        json.dumps(expected))


def test_the_export_runs_with_spacr_run(qtbot, monkeypatch, tmp_path,
                                        capsys):
    """An exported Measure form passes ``spacr-run measure --dry-run``."""
    from spacr.cli import EXIT_OK, main, resolve_module, resolve_settings

    plate = tmp_path / "plate"
    (plate / "merged").mkdir(parents=True)
    np.save(plate / "merged" / "plate1_A01_1.npy",
            np.zeros((16, 16, 7), dtype=np.uint16))
    screen = _screen(qtbot, "measure")
    screen.apply_settings_dict({"src": str(plate)})
    path = tmp_path / "measure_settings.csv"
    _save_to(monkeypatch, path)
    screen._on_export_settings()
    moved = []
    resolved = resolve_settings(resolve_module("measure"), str(path), [],
                                moved)
    assert resolved["src"] == str(plate)
    assert main(["measure", "--settings", str(path), "--dry-run",
                 "--no-preflight"]) == EXIT_OK
    assert "nothing was executed" in capsys.readouterr().out


def test_export_offers_the_module_name_in_the_source_folder(
        qtbot, monkeypatch, tmp_path):
    """The dialog offers ``<module>_settings.csv`` in the src folder."""
    screen = _screen(qtbot, "measure")
    screen.apply_settings_dict({"src": str(tmp_path / "plate")})
    offered = _save_to(monkeypatch, tmp_path / "chosen")
    screen._on_export_settings()
    assert offered == [str(tmp_path / "plate" / "measure_settings.csv")]
    assert (tmp_path / "chosen.csv").is_file()
    assert (tmp_path / "chosen.json").is_file()
    assert "Saved" in _console_text(screen._console)


def test_cancelling_export_writes_nothing(qtbot, monkeypatch, tmp_path):
    """A cancelled dialog writes no file and says nothing."""
    screen = _screen(qtbot, "measure")
    before = _console_text(screen._console)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._on_export_settings()
    assert _console_text(screen._console) == before
    assert list(tmp_path.iterdir()) == []


def test_export_reports_settings_it_cannot_collect(qtbot, monkeypatch):
    """A form that cannot be collected is reported, and no dialog opens."""
    screen = _screen(qtbot, "measure")
    seen = _warnings(monkeypatch)

    def bad():
        raise ValueError("batch_size must be a number")

    monkeypatch.setattr(screen._settings_model, "collect", bad)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(
        lambda *a, **k: pytest.fail("the dialog opened")))
    screen._on_export_settings()
    assert [args[1:] for args in seen] == [
        ("Bad settings", "batch_size must be a number")]


def test_export_reports_a_file_it_cannot_write(qtbot, monkeypatch, tmp_path):
    """A failed write is a warning, not a traceback."""
    screen = _screen(qtbot, "measure")
    seen = _warnings(monkeypatch)
    _save_to(monkeypatch, tmp_path / "out.csv")

    def broken(_settings, _path):
        raise OSError("read-only share")

    monkeypatch.setattr("spacr.cli.write_settings_file", broken)
    screen._on_export_settings()
    assert [args[1:] for args in seen] == [
        ("Export failed", "read-only share")]


def test_default_path_follows_src_then_last_folder_then_home(
        monkeypatch, tmp_path):
    """Source folder first, then the last settings folder, then home."""
    monkeypatch.setitem(app_screen_module._SETTINGS_CSV_FOLDER, "last", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_settings_csv_path("mask", "") == str(
        tmp_path / "home" / "mask_settings.csv")
    assert default_settings_csv_path("classify", ["", " /p1 ", "/p2"]) == (
        os.path.join("/p1", "classify_settings.csv"))
    assert default_settings_csv_path("umap", []) == str(
        tmp_path / "home" / "umap_settings.csv")
    app_screen_module._SETTINGS_CSV_FOLDER["last"] = "/shared/configs"
    assert default_settings_csv_path("mask", None) == (
        os.path.join("/shared/configs", "mask_settings.csv"))
    assert default_settings_csv_path("mask", "/data/p") == (
        os.path.join("/data/p", "mask_settings.csv"))


def test_import_remembers_its_folder_for_the_next_export(
        qtbot, monkeypatch, tmp_path):
    """The folder a CSV was imported from is offered for the next export."""
    monkeypatch.setitem(app_screen_module._SETTINGS_CSV_FOLDER, "last", "")
    folder = tmp_path / "configs"
    folder.mkdir()
    source = folder / "old.csv"
    source.write_text("Key,Value\nbatch_size,7\n", encoding="utf-8")
    screen = _screen(qtbot, "measure")
    screen.apply_settings_dict({"src": None})
    _open(monkeypatch, source)
    screen._on_import_settings()
    assert app_screen_module._SETTINGS_CSV_FOLDER["last"] == str(folder)
    assert default_settings_csv_path("measure", "") == str(
        folder / "measure_settings.csv")


def test_an_empty_hidden_text_setting_reads_back_empty(qtbot):
    """A CSV writes '' and None alike; '' stays '' on a hidden setting."""
    screen = _screen(qtbot, "timelapse")
    model = screen._settings_model
    assert model._widgets.get("cloud_endpoint") is None
    assert model.collect()["cloud_endpoint"] == ""
    assert model.set_hidden_value("cloud_endpoint", None)
    assert model.collect()["cloud_endpoint"] == ""
    assert model.set_hidden_value("cloud_endpoint", "s3.example")
    assert model.set_hidden_value("cloud_endpoint", None)
    assert model.collect()["cloud_endpoint"] is None


@pytest.mark.parametrize("language", ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"))
def test_export_button_uses_its_current_translation(qtbot, monkeypatch, language):
    from spacr.qt.i18n import ENV_LANGUAGE, tr
    monkeypatch.setenv(ENV_LANGUAGE, language)
    screen = _screen(qtbot, "measure")
    expected = tr("Export settings…", language=language)
    assert expected != "Export settings…"
    assert screen._btn_export.text() == expected
