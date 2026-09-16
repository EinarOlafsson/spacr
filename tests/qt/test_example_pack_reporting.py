"""317: example imports explain pack migration and missing-pack defaults.

Exercise the real callers with plain stand-ins, not downloads or pipelines.
The shared reader remains real; only its schema and rename table are small.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import settings_pack
from spacr.qt.app import MainWindow
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt

_DEFAULTS = {
    "src": "",
    "cell_diameter": 10,
    "cell_flow_threshold": 1.0,
    "channels": [0],
    "verbose": False,
}


class _Console:
    """Capture the same formatted notices the example console displays."""

    def __init__(self):
        self.lines = []

    def append_notice(self, template, **values):
        self.lines.append(template.format(**values))

    def append_stdout(self, text):
        self.lines.append(text)


class _Screen:
    """Bind the real example method without building a settings form."""

    _EXAMPLE_SETTINGS_FILES = AppScreen._EXAMPLE_SETTINGS_FILES
    apply_settings_that_came_with = AppScreen.apply_settings_that_came_with
    reanchor_example_paths = staticmethod(AppScreen.reanchor_example_paths)

    def __init__(self, app_key):
        self.app_key = app_key
        self.values = dict(_DEFAULTS, cell_diameter=17, verbose=True)
        self.applied = []
        self._console = _Console()
        self._settings_model = SimpleNamespace(
            _defaults=dict(_DEFAULTS), collect=lambda: dict(self.values))

    def _load_settings_csv(self, path):
        """Keep the old caller executable without importing pipeline utils."""
        return settings_pack.read_pack(self.app_key, str(Path(path).parent))[0]

    def apply_settings_dict(self, values):
        self.applied.append(dict(values))
        self.values.update(values)
        return len(values)

    def _on_run(self):
        raise AssertionError("importing an example must not launch a pipeline")


class _Window:
    """The e2e import only needs navigation, a screen, and a status bar."""

    _run_e2e_chain = MainWindow._run_e2e_chain

    def __init__(self):
        self.screen = _Screen("mask")
        self._screens = {"mask": self.screen}
        self.opened = []
        self.messages = []

    def _on_nav_selected(self, app_key):
        self.opened.append(app_key)

    def statusBar(self):
        return self

    def showMessage(self, text, timeout):
        self.messages.append(text)


def _write_pack(directory, filename, rows):
    """Write an actual shipped-format CSV, including any malformed rows."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Key", "Value"))
        writer.writerows(rows)
    return path


@pytest.fixture
def shared_reader(monkeypatch):
    """Trace real migration while keeping registry/model imports out."""
    from spacr.qt.screens import settings_model

    monkeypatch.setattr(settings_model, "resolve_default_settings",
                        lambda app_key: dict(_DEFAULTS))
    monkeypatch.setitem(settings_pack.PACK_RENAMES, "mask",
                        {"old_flow": "cell_flow_threshold"})
    monkeypatch.setattr(settings_pack, "_package_renames", lambda key: ())
    monkeypatch.setattr(settings_pack, "_is_a_setting_somewhere",
                        lambda key: False)
    original = settings_pack.settings_from_pack
    calls = []

    def traced(app_key, pack_dir, **kwargs):
        result = original(app_key, pack_dir, **kwargs)
        calls.append((app_key, Path(pack_dir), result[1]))
        return result

    monkeypatch.setattr(settings_pack, "settings_from_pack", traced)
    return calls


def test_example_uses_shared_migration_and_reports_every_exception(
        tmp_path, shared_reader):
    """A loaded count alone hides both recoverable renames and lost rows."""
    _write_pack(tmp_path / "settings", "gen_masks_settings.csv", [
        ("cell_diameter", "41.5"),
        ("old_flow", "0.4"),
        ("channels", "[0, 2]"),
        ("a_setting_that_never_existed", "99"),
        ("unreadable_one_column_row",),
    ])
    screen = _Screen("mask")

    count = screen.apply_settings_that_came_with(tmp_path)

    assert shared_reader, "the example bypassed the migrating/reporting reader"
    report = shared_reader[-1][2]
    assert report.applied == ["cell_diameter", "channels"]
    assert report.renamed == [("old_flow", "cell_flow_threshold")]
    assert report.dropped == ["a_setting_that_never_existed"]
    assert report.malformed == 1
    assert count == 3
    assert screen.applied == [{
        "cell_diameter": 41.5, "cell_flow_threshold": 0.4, "channels": [0, 2],
    }]
    assert screen.values["verbose"] is True, "unshipped defaults reset the form"
    console = "".join(screen._console.lines)
    assert ("3 settings applied to the form from gen_masks_settings.csv; "
            "3 CSV keys accepted.") in console
    assert "Renamed 1 settings: old_flow → cell_flow_threshold" in console
    assert ("Dropped 1 unknown or retired settings: "
            "a_setting_that_never_existed") in console
    assert "Skipped 1 unreadable CSV row(s)." in console


def test_explicit_pack_beats_the_preferred_filename_from_a_users_run(
        tmp_path, shared_reader):
    """Directory priority must precede the reader's filename preference."""
    plate = tmp_path / "plate1"
    shipped = tmp_path / "shipped"
    _write_pack(shipped, "gen_masks_settings.csv", [("cell_diameter", "42")])
    _write_pack(plate / "settings", "gen_mask_settings.csv",
                [("cell_diameter", "900")])
    screen = _Screen("mask")

    assert screen.apply_settings_that_came_with(plate, pack_folder=shipped) == 1
    assert screen.values["cell_diameter"] == 42
    assert shared_reader, "explicit packs must use the shared reader too"
    assert shared_reader[-1][1] == shipped
    assert shared_reader[-1][2].source == "gen_masks_settings.csv"


@pytest.mark.parametrize("explicit_empty_pack", [False, True])
def test_plate_settings_remain_the_fallback(
        tmp_path, shared_reader, explicit_empty_pack):
    """No explicit pack, or no file for this app there, still reads the plate."""
    plate = tmp_path / "plate1"
    _write_pack(plate / "settings", "gen_mask_settings.csv",
                [("cell_diameter", "28")])
    empty = tmp_path / "empty-pack"
    empty.mkdir()
    screen = _Screen("mask")

    count = screen.apply_settings_that_came_with(
        plate, pack_folder=empty if explicit_empty_pack else None)

    assert count == 1
    assert screen.values["cell_diameter"] == 28


def test_an_existing_pack_with_only_dropped_keys_does_not_load_a_users_run(
        tmp_path, shared_reader):
    """An authoritative pack with zero applicable keys is not a missing pack."""
    plate = tmp_path / "plate1"
    shipped = tmp_path / "shipped"
    _write_pack(shipped, "gen_masks_settings.csv",
                [("a_setting_that_never_existed", "99")])
    _write_pack(plate / "settings", "gen_mask_settings.csv",
                [("cell_diameter", "900")])
    screen = _Screen("mask")

    assert screen.apply_settings_that_came_with(plate, pack_folder=shipped) == 0
    assert screen.values["cell_diameter"] == 17
    assert "a_setting_that_never_existed" not in screen.values
    assert ("Dropped 1 unknown or retired settings: "
            "a_setting_that_never_existed") in "".join(screen._console.lines)


def test_measure_pack_keeps_the_reanchored_merged_subfolder(
        tmp_path, shared_reader):
    """Passing src=plate to the shared reader would silently lose /merged."""
    plate = tmp_path / "plate1"
    (plate / "merged").mkdir(parents=True)
    foreign = "/nonexistent-spacr-example-root/publisher/plate1/merged"
    assert not Path(foreign).exists()
    _write_pack(plate / "settings", "crop_measure_settings.csv", [
        ("src", foreign), ("channels", "[1, 2]"),
    ])
    screen = _Screen("measure")

    assert screen.apply_settings_that_came_with(plate) == 2
    assert screen.values["src"] == str(plate / "merged")
    assert screen.values["channels"] == [1, 2]
    assert screen.values["verbose"] is True


def test_e2e_missing_pack_warns_and_describes_defaults(
        tmp_path, shared_reader, caplog):
    """Missing settings are usable defaults, not successfully loaded settings."""
    window = _Window()
    data = tmp_path / "plate1"

    with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
        window._run_e2e_chain(data, tmp_path / "missing-pack")

    assert window.opened == ["mask"]
    assert window.screen.applied == [dict(_DEFAULTS, src=str(data))]
    assert shared_reader[-1][2].source == ""
    warnings = [record.getMessage().lower() for record in caplog.records
                if record.name == "spacr.qt.app"
                and record.levelno >= logging.WARNING]
    assert any("settings" in text and "default" in text
               and ("missing" in text or "no " in text or "not found" in text)
               for text in warnings), "no missing-pack/defaults warning was emitted"
    assert "defaults" in window.messages[-1].lower()
    assert "loaded with its settings" not in window.messages[-1].lower()
    assert "Live Preview" in window.messages[-1]


def test_e2e_a_real_pack_is_not_misreported_as_a_defaults_fallback(
        tmp_path, shared_reader, caplog):
    """The missing-pack notice must not fire for a valid shipped filename."""
    pack = tmp_path / "pack"
    _write_pack(pack, "gen_masks_settings.csv", [("cell_diameter", "43")])
    window = _Window()
    data = tmp_path / "plate1"

    with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
        window._run_e2e_chain(data, pack)

    assert window.screen.values["cell_diameter"] == 43
    assert window.screen.values["src"] == str(data)
    assert shared_reader[-1][2].source == "gen_masks_settings.csv"
    assert not [record for record in caplog.records
                if record.name == "spacr.qt.app"
                and record.levelno >= logging.WARNING]
    assert "loaded with its settings" in window.messages[-1].lower()
    assert "defaults" not in window.messages[-1].lower()


@pytest.mark.parametrize("basis", ["metadata", "annotation"])
@pytest.mark.parametrize("explicit_classes", [False, True],
                         ids=["legacy-compound", "current-classes-win"])
def test_shared_pack_preserves_the_existing_compound_class_import(
        tmp_path, basis, explicit_classes):
    """Filtering rows must not pre-empt the existing Classes translation."""
    from spacr.qt.screens.settings_model import resolve_default_settings

    defaults = resolve_default_settings("classify")
    assert "classes" in defaults
    assert "location_column" not in defaults
    legacy = {
        "dataset_mode": basis,
        "class_folder_names": ["untreated", "treated"],
        "location_column": "columnID",
        "negative_control_id": "c1",
        "positive_control_id": "c3",
    }
    expected = {
        "untreated": {"column": "columnID", "value": "c1"},
        "treated": {"column": "columnID", "value": "c3"},
    }
    migrated_sources = {"location_column", "negative_control_id",
                        "positive_control_id"}
    if basis == "annotation":
        # This shape already reaches _migrate_control_wells because the old
        # control fields coexist with the explicitly selected annotation basis.
        legacy.update(annotation_columns=["infection"], annotation_values=[0, 1])
        expected = {
            "untreated": {"column": "infection", "value": 0},
            "treated": {"column": "infection", "value": 1},
        }
        migrated_sources = {"annotation_columns", "annotation_values"}
    if explicit_classes:
        expected = {"current": {"column": "well", "value": "A01"}}
        legacy["classes"] = expected

    # Positive evidence for the actual prior form path, not an invented
    # transformation: it reaches the real classify_classes normalizer.
    form = SimpleNamespace(_settings_model=SimpleNamespace(
        _widgets={"classes": object()}))
    previous = AppScreen._migrate_control_wells(form, dict(legacy))
    assert previous["classes"] == expected

    legacy["setting_that_never_existed_317"] = 9
    path = _write_pack(tmp_path, "classify_settings.csv", legacy.items())
    settings, report = settings_pack.settings_from_pack(
        "classify", str(path.parent), defaults=defaults)

    assert settings["classes"] == expected
    assert "setting_that_never_existed_317" not in settings
    assert "setting_that_never_existed_317" in report.dropped
    if explicit_classes:
        assert "classes" in report.applied
        assert not any(new == "classes" for _old, new in report.renamed)
    else:
        assert migrated_sources <= {
            old for old, new in report.renamed if new == "classes"}
        assert not (migrated_sources & set(report.dropped))
    assert (len(report.applied) + len(report.renamed) + len(report.dropped)
            == len(legacy))
