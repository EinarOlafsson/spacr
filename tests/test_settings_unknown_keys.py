"""Item 1002: unknown keys in a settings file warn, or are refused in strict mode."""
import json
import logging

import pytest

from spacr import cli
from spacr.cli import SettingsError, resolve_module, resolve_settings
from spacr.settings import (RECORD_ONLY_PREFIXES, RECORD_ONLY_SETTINGS,
                            expected_types, unknown_setting_keys)

MODULE = resolve_module("mask")


def _file(tmp_path, **keys):
    path = tmp_path / "s.json"
    path.write_text(json.dumps(keys))
    return str(path)


def _real_key():
    return next(k for k in cli.module_defaults(MODULE) if k in expected_types)


def test_default_warns_once_and_keeps_dict(tmp_path, caplog):
    plain = resolve_settings(MODULE, _file(tmp_path))
    with caplog.at_level(logging.WARNING):
        got = resolve_settings(MODULE, _file(tmp_path, zzz_not_a_setting=1))
    assert got == dict(plain, zzz_not_a_setting=1)
    lines = [r.getMessage() for r in caplog.records if "zzz_not_a_setting" in r.getMessage()]
    assert len(lines) == 1 and "nearest" not in lines[0]


def test_nearest_key():
    key = _real_key()
    (found,) = unknown_setting_keys({key + "x": 1})
    assert found == (key + "x", key)
    assert unknown_setting_keys({"zzz_not_a_setting": 1}) == [("zzz_not_a_setting", None)]


def test_one_line_per_key_in_moved(tmp_path):
    moved = []
    resolve_settings(MODULE, _file(tmp_path, aaa_one=1, bbb_two=2, ccc_three=3), moved=moved)
    assert sorted(p.setting for p in moved) == ["aaa_one", "bbb_two", "ccc_three"]
    assert all(p.severity == "warning" for p in moved)


@pytest.mark.parametrize("how", ["arg", "set", "file"])
def test_strict_refuses_all(tmp_path, how):
    keys = {"aaa_one": 1, "bbb_two": 2}
    kwargs, overrides = {}, []
    if how == "arg":
        kwargs["strict"] = True
    elif how == "set":
        overrides = ["settings_strict=true"]
    else:
        keys["settings_strict"] = True
    with pytest.raises(SettingsError) as err:
        resolve_settings(MODULE, _file(tmp_path, **keys), overrides, **kwargs)
    assert "aaa_one" in str(err.value) and "bbb_two" in str(err.value)


def test_strict_flag_exit_code(tmp_path, capsys):
    path = _file(tmp_path, aaa_one=1)
    assert cli.main(["mask", "--settings", path, "--strict-settings", "--dry-run"]) == cli.EXIT_USAGE
    assert "aaa_one" in capsys.readouterr().err


def test_precedence(tmp_path):
    path = _file(tmp_path, aaa_one=1, settings_strict=True)
    resolve_settings(MODULE, path, strict=False)
    with pytest.raises(SettingsError):
        resolve_settings(MODULE, path)
    resolve_settings(MODULE, path, ["settings_strict=false"])


def test_allowlist(tmp_path):
    names = ["_comment", "_comment_ops", "_meta", "_meta_run"]
    names += [n for n in RECORD_ONLY_SETTINGS if n.startswith("seg_")]
    moved = []
    resolve_settings(MODULE, _file(tmp_path, spacr_settings_version=1,
                                   **{n: "x" for n in names}), moved=moved, strict=True)
    assert moved == []
    assert [k for k, _ in unknown_setting_keys({"seg_tile_pxx": 1, "meta_run": 1})] == [
        "seg_tile_pxx", "meta_run"]


def test_control_key_not_in_resolved(tmp_path):
    assert "settings_strict" not in resolve_settings(MODULE, _file(tmp_path, settings_strict=False))


def test_validate_names_each_key_once(tmp_path, capsys):
    path = _file(tmp_path, aaa_one=1)
    cli.main(["validate", "--module", "mask", "--settings", path])
    assert capsys.readouterr().out.count("Unknown setting 'aaa_one'") == 1


def test_set_still_refuses_unknown(tmp_path):
    with pytest.raises(SettingsError, match="does not exist"):
        resolve_settings(MODULE, _file(tmp_path), ["zzz_not_a_setting=1"])


def test_allowlist_names_become_real_settings_and_leave():
    from spacr.validate import _known_setting_keys
    known = _known_setting_keys() | set(cli.module_defaults(MODULE))
    assert sorted(n for n in RECORD_ONLY_SETTINGS if n in known or n in expected_types) == []


def test_ops_phases_is_a_real_setting():
    assert "ops_phases" not in RECORD_ONLY_SETTINGS
    assert unknown_setting_keys({"ops_phases": "stitch"}, "ops") == []
    assert RECORD_ONLY_PREFIXES == ("_comment", "_meta")


class _Console:
    def __init__(self):
        self.rows = []

    def append_notice(self, source, **values):
        self.rows.append(source.format(**values).strip())


class _Screen:
    app_key = "mask"

    def __init__(self):
        self._console = _Console()


@pytest.mark.parametrize("alpha", [True, False])
def test_gui_notice_only_with_alpha(monkeypatch, alpha):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(preferences, "get_show_alpha", lambda: alpha)
    screen = _Screen()
    key = _real_key()
    AppScreen._notice_unknown_settings(screen, {key + "x": 1, "zzz_not_a_setting": 2})
    if alpha:
        assert screen._console.rows == [f"Unknown setting '{key}x' -> '{key}'",
                                        "Unknown setting 'zzz_not_a_setting'"]
    else:
        assert screen._console.rows == []
