"""The module table the CLI answers from, and the three ways it is amended.

``spacr-run`` resolves a name against one table. Plugins add rows to it, the
Qt registry adds notes about GUI-only apps to a second one, and neither is
allowed to take a built-in away: a plugin that claims ``measure`` must be
refused with a diagnostic rather than silently replacing the pipeline a user
has been running for months. The plugin SDK itself is optional, so a
checkout without it still has to answer ``--list``.

The pre-flight half is here for the same reason: ``--hash-inputs`` is a flag
whose whole job is to override what the settings file said, so what
``validate`` checks has to be the settings AFTER the flag was applied.
"""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from spacr import cli
from spacr.plugins import AppContribution


@pytest.fixture
def module_table():
    """Restore ``MODULES``/``ALIASES`` after a test amends them."""
    modules = dict(cli.MODULES)
    aliases = dict(cli.ALIASES)
    yield cli
    cli.MODULES.clear()
    cli.MODULES.update(modules)
    cli.ALIASES.clear()
    cli.ALIASES.update(aliases)


def _app(key, aliases=()):
    return AppContribution(
        key=key, name=f"{key} title", description=f"what {key} does",
        entrypoint="spacr_plugin_demo:run", defaults="spacr_plugin_demo:defaults",
        aliases=tuple(aliases), requires=("src",), writes=("a table",),
        call_style="settings", kind="analysis")


def test_without_the_plugin_sdk_the_built_in_modules_still_answer(
        module_table, monkeypatch, caplog):
    """A checkout with no plugin SDK must not lose ``--list``."""
    real_import = builtins.__import__

    def no_plugins(name, globals=None, locals=None, fromlist=(), level=0):
        if level and name == "plugins":
            raise ImportError("the plugin SDK is not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_plugins)
    before = dict(cli.MODULES)

    with caplog.at_level("ERROR", logger="spacr.cli"):
        cli._register_plugin_modules()

    assert cli.MODULES == before
    assert "plugin SDK" in caplog.text


def test_a_plugin_may_not_replace_a_built_in_module(module_table, monkeypatch):
    """The built-in wins the key, and the user is told which plugin lost."""
    import spacr.plugins as plugins

    complaints = []
    monkeypatch.setattr(plugins, "plugin_apps", lambda: (_app("measure"),))
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda key, message, *a, **k: complaints.append(
                            (key, message)))
    built_in = cli.MODULES["measure"]

    cli._register_plugin_modules()

    assert cli.MODULES["measure"] is built_in
    assert len(complaints) == 1
    assert complaints[0][0] == "measure"
    assert "collides with a built-in module" in complaints[0][1]


def test_a_plugin_app_becomes_a_module_the_cli_can_resolve(module_table,
                                                           monkeypatch):
    """A fresh key registers, and so do the friendly spellings it brings."""
    import spacr.plugins as plugins

    monkeypatch.setattr(
        plugins, "plugin_apps",
        lambda: (_app("demo_assay", aliases=("Demo-Plaques", "", "measure")),))
    monkeypatch.setattr(plugins, "record_diagnostic",
                        lambda *a, **k: None)

    cli._register_plugin_modules()

    module = cli.resolve_module("demo_assay")
    assert module is not None
    assert module.summary == "what demo_assay does"
    assert module.requires == ("src",)
    assert module.note.endswith("(analysis).")
    assert cli.ALIASES["demo_plaques"] == "demo_assay"
    assert cli.resolve_module("Demo-Plaques") is module, (
        "the plugin's own alias has to resolve")
    assert cli.resolve_module("measure").key == "measure", (
        "an alias may not shadow a built-in module either")
    assert "" not in cli.ALIASES, "a blank alias is not a name"


def test_gui_only_notes_are_read_from_a_registry_that_may_not_be_there(
        monkeypatch):
    """A cluster process that never imported Qt keeps the built-in table."""
    stub = types.ModuleType("spacr.qt.app")
    monkeypatch.setitem(sys.modules, "spacr.qt.app", stub)
    before = dict(cli.INTERACTIVE_ONLY)

    cli._absorb_registered_gui_only()

    assert cli.INTERACTIVE_ONLY == before


def test_a_per_module_type_narrowing_beats_the_declared_type():
    """``foreign``'s masks setting takes a path or a list, not any literal."""
    assert cli._allowed_types("masks", None, {"masks": str},
                              app="foreign") == (str, list)
    assert cli._allowed_types("masks", None, {"masks": str}) == (str,)


def _clean_settings(tmp_path):
    src = tmp_path / "plate"
    src.mkdir()
    path = tmp_path / "s.csv"
    path.write_text("key,value\nsrc,%s\n" % src, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("flag,expected", [
    ("--hash-inputs", True), ("--no-hash-inputs", False),
])
def test_the_hashing_flag_reaches_the_settings_the_preflight_checks(
        tmp_path, monkeypatch, flag, expected):
    """A flag that only takes effect at run time would validate a lie."""
    seen = []
    monkeypatch.setattr(cli, "_preflight",
                        lambda settings, key: seen.append(dict(settings)) or [])

    rc = cli.main(["validate", "--settings", _clean_settings(tmp_path), flag])

    assert rc == cli.EXIT_OK
    assert seen and seen[0]["hash_inputs"] is expected


def test_without_the_flag_the_settings_file_still_decides(tmp_path,
                                                          monkeypatch):
    """Neither flag given means the file's own answer stands."""
    seen = []
    monkeypatch.setattr(cli, "_preflight",
                        lambda settings, key: seen.append(dict(settings)) or [])

    rc = cli.main(["validate", "--settings", _clean_settings(tmp_path)])

    assert rc == cli.EXIT_OK
    assert "hash_inputs" not in seen[0]
