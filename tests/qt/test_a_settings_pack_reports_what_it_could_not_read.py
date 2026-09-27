"""A settings pack that is damaged, or read without spaCR's settings table, still loads.

Pinned behaviour of :func:`spacr.qt.settings_pack.settings_from_pack` and
its report:

* rows without a value are counted and the summary names them as
  unreadable rows;
* a file that is not UTF-8 gives the app's defaults and a logged error,
  never an exception;
* when the package settings table cannot be imported, or its lookups
  raise, an unknown key is reported as dropped rather than breaking the
  load;
* the legacy control-well class migration keeps the form's classes when it
  finds nothing to migrate or the migration raises, and wins over an
  explicitly empty ``classes`` cell;
* a ``src`` the caller supplies replaces the pack's, even on a form that
  has no ``src`` default, and is not reported as dropped.
"""
from __future__ import annotations

import logging
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt import settings_pack  # noqa: E402

pytestmark = pytest.mark.qt

_DEFAULTS = {"src": "", "cell_diameter": 10, "verbose": False}


def _pack(tmp_path, text, name="mask_settings.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_rows_without_a_value_are_named_as_unreadable(tmp_path):
    folder = _pack(tmp_path, "cell_diameter,25\nlonely_key\nverbose,true\n")

    settings, report = settings_pack.settings_from_pack(
        "mask", folder, defaults=dict(_DEFAULTS))

    assert settings["cell_diameter"] == 25 and settings["verbose"] is True
    assert report.malformed == 1
    assert report.summary().endswith("; 1 unreadable row(s).")


def test_a_pack_that_is_not_utf8_gives_the_defaults_and_logs_why(
        tmp_path, caplog):
    (tmp_path / "mask_settings.csv").write_bytes(b"cell_diameter,\xff\xfe25\n")

    with caplog.at_level(logging.ERROR, logger=settings_pack.LOG.name):
        settings, report = settings_pack.settings_from_pack(
            "mask", str(tmp_path), defaults=dict(_DEFAULTS))

    assert settings == _DEFAULTS
    assert report.applied == [] and report.source == "mask_settings.csv"
    assert "Could not read the settings pack" in caplog.text


def test_without_the_settings_table_an_unknown_key_is_simply_dropped(
        tmp_path, monkeypatch):
    folder = _pack(tmp_path, "cell_diameter,30\ntimelapse,true\n")
    monkeypatch.setitem(sys.modules, "spacr.settings", None)

    settings, report = settings_pack.settings_from_pack(
        "mask", folder, defaults=dict(_DEFAULTS))

    assert settings["cell_diameter"] == 30 and "timelapse" not in settings
    assert report.dropped == ["timelapse"] and report.elsewhere == []
    assert "dropped timelapse (this version has no such setting)" in (
        report.summary())


def test_a_settings_table_whose_lookups_raise_does_not_break_the_load(
        tmp_path, monkeypatch):
    import spacr.settings as package_settings

    class _Broken:
        def __contains__(self, key):
            raise RuntimeError("table unavailable")

    def _no_rename(key):
        raise RuntimeError("rename table unavailable")

    monkeypatch.setattr(package_settings, "surviving_setting_name", _no_rename)
    monkeypatch.setattr(package_settings, "expected_types", _Broken())
    folder = _pack(tmp_path, "cell_diameter,30\ntimelapse,true\n")

    settings, report = settings_pack.settings_from_pack(
        "mask", folder, defaults=dict(_DEFAULTS))

    assert settings["cell_diameter"] == 30
    assert report.dropped == ["timelapse"] and report.elsewhere == []


def _classify_defaults():
    from spacr.qt.screens.settings_model import resolve_default_settings

    defaults = dict(resolve_default_settings("classify"))
    assert "classes" in defaults and "location_column" not in defaults
    return defaults


def test_a_location_column_without_controls_leaves_the_classes_alone(
        tmp_path):
    defaults = _classify_defaults()
    folder = _pack(tmp_path, "location_column,columnID\n",
                   name="classify_settings.csv")

    settings, report = settings_pack.settings_from_pack(
        "classify", folder, defaults=dict(defaults))

    assert settings["classes"] == defaults["classes"]
    assert ("location_column", "classes") not in report.renamed


def test_a_migration_that_raises_keeps_the_form_s_classes(
        tmp_path, monkeypatch):
    import spacr.classify_classes as classify_classes

    def _broken(settings):
        raise ValueError("unexpected legacy shape")

    monkeypatch.setattr(classify_classes, "normalize_settings", _broken)
    defaults = _classify_defaults()
    folder = _pack(tmp_path,
                   "location_column,columnID\nnegative_control_id,c1\n"
                   "positive_control_id,c3\n",
                   name="classify_settings.csv")

    settings, report = settings_pack.settings_from_pack(
        "classify", folder, defaults=dict(defaults))

    assert settings["classes"] == defaults["classes"]
    assert not any(new == "classes" for _old, new in report.renamed)


def test_the_migrated_classes_win_over_an_empty_classes_cell(tmp_path):
    defaults = _classify_defaults()
    folder = _pack(tmp_path,
                   "classes,[]\nlocation_column,columnID\n"
                   "negative_control_id,c1\npositive_control_id,c3\n",
                   name="classify_settings.csv")

    settings, report = settings_pack.settings_from_pack(
        "classify", folder, defaults=dict(defaults))

    assert settings["classes"] == {
        "negative control": {"column": "columnID", "value": "c1"},
        "positive control": {"column": "columnID", "value": "c3"},
    }
    assert "classes" in report.applied
    assert ("location_column", "classes") in report.renamed


def test_the_caller_s_src_replaces_the_pack_s_on_a_form_without_src(
        tmp_path):
    folder = _pack(tmp_path, "src,/elsewhere/plate_1\ncell_diameter,12\n")

    settings, report = settings_pack.settings_from_pack(
        "mask", folder, src="/data/plate_9",
        defaults={"cell_diameter": 10})

    assert settings == {"cell_diameter": 12, "src": "/data/plate_9"}
    assert "src" not in report.dropped
    assert "dropped" not in report.summary()
