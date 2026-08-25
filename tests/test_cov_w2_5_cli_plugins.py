"""``spacr-plugins`` reports what is installed and what failed to load.

The command is the only place a user finds out that an installed plugin did
not load, so the tests below register a real plugin module through
``SPACR_PLUGIN_MODULES`` and read the command's actual output rather than
asserting against a stubbed payload. ``doctor`` additionally has to return a
non-zero exit code when anything is wrong, because that is what a CI job
checks.
"""
from __future__ import annotations

import json
import sys

import pytest

from spacr import cli_plugins, plugins


PLUGIN_SOURCE = '''
"""A tiny but valid plugin manifest used to exercise the plugin CLI."""

plugin = {
    "name": "w2_5 probe",
    "version": "9.9.9",
    "apps": [
        {
            "key": "w2_5_probe_app",
            "name": "Probe",
            "description": "Nothing but a manifest.",
            "entrypoint": "w2_5_probe_plugin:run",
            "defaults": "w2_5_probe_plugin:defaults",
        }
    ],
    "model_providers": [
        {"key": "w2_5_probe_models",
         "provider": "w2_5_probe_plugin:models"}
    ],
    "report_sections": [
        {"key": "w2_5_probe_section", "title": "Probe",
         "builder": "w2_5_probe_plugin:section"}
    ],
}


def run(settings):
    return settings


def defaults(settings=None):
    return {}


def models():
    return ()


def section(context):
    return None
'''

BROKEN_SOURCE = '''
"""A manifest that is invalid on purpose, to produce a diagnostic."""

plugin = {"name": "", "version": ""}
'''


@pytest.fixture
def plugin_registry(monkeypatch, tmp_path):
    """Install plugin modules by name and rediscover, restoring afterwards."""
    (tmp_path / "w2_5_probe_plugin.py").write_text(PLUGIN_SOURCE)
    (tmp_path / "w2_5_broken_plugin.py").write_text(BROKEN_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))

    def install(*modules):
        monkeypatch.setenv(plugins.PLUGIN_MODULES_ENV, ",".join(modules))
        for name in ("w2_5_probe_plugin", "w2_5_broken_plugin"):
            sys.modules.pop(name, None)
        return plugins.reload_plugins()

    try:
        yield install
    finally:
        monkeypatch.undo()
        for name in ("w2_5_probe_plugin", "w2_5_broken_plugin"):
            sys.modules.pop(name, None)
        plugins.reload_plugins()


def test_the_parser_defaults_to_listing():
    """No argument means ``list``, and ``--json`` is off."""
    args = cli_plugins.build_parser().parse_args([])

    assert args.command == "list"
    assert args.json is False
    assert cli_plugins.build_parser().parse_args(["doctor"]).command == "doctor"


def test_an_unknown_command_is_rejected():
    """Only the two documented commands are accepted."""
    with pytest.raises(SystemExit):
        cli_plugins.build_parser().parse_args(["undoctor"])


def test_listing_prints_every_contribution_of_a_real_plugin(
        plugin_registry, capsys):
    """Each plugin's apps, model providers and report sections are shown."""
    loaded = plugin_registry("w2_5_probe_plugin")
    assert [p.name for p in loaded] == ["w2_5 probe"]

    code = cli_plugins.main([])

    out = capsys.readouterr().out
    assert code == 0
    assert f"spaCR plugin SDK {plugins.PLUGIN_API_VERSION}" in out
    assert "w2_5 probe 9.9.9 (API 1.0)" in out
    assert "apps: w2_5_probe_app" in out
    assert "model providers: w2_5_probe_models" in out
    assert "report sections: w2_5_probe_section" in out
    assert "No plugins discovered." not in out


def test_an_empty_install_says_so_rather_than_printing_nothing(
        plugin_registry, capsys):
    """Zero plugins is a sentence, not silence."""
    plugin_registry()

    code = cli_plugins.main([])

    out = capsys.readouterr().out
    assert code == 0
    assert "No plugins discovered." in out


def test_a_plugin_that_will_not_load_is_reported_and_fails_the_doctor(
        plugin_registry, capsys):
    """``doctor`` exits non-zero and names the plugin and the exception."""
    plugin_registry("w2_5_broken_plugin")

    code = cli_plugins.main(["doctor"])

    out = capsys.readouterr().out
    assert code == 1
    assert "Diagnostics:" in out
    assert "w2_5_broken_plugin" in out
    assert "[error]" in out
    assert "ValueError" in out, "the cause belongs on the line"
    assert "No plugin errors recorded." not in out


def test_a_clean_doctor_says_there_are_no_errors(plugin_registry, capsys):
    """With nothing wrong the doctor still prints a verdict and exits zero."""
    plugin_registry("w2_5_probe_plugin")

    code = cli_plugins.main(["doctor"])

    out = capsys.readouterr().out
    assert code == 0
    assert "No plugin errors recorded." in out


def test_json_output_is_parseable_and_carries_the_same_facts(
        plugin_registry, capsys):
    """``--json`` emits one JSON document and no prose around it."""
    plugin_registry("w2_5_probe_plugin")

    code = cli_plugins.main(["list", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["sdk_version"] == plugins.PLUGIN_API_VERSION
    assert payload["apps"] == ["w2_5_probe_app"]
    assert payload["diagnostics"] == []
    only = payload["plugins"][0]
    assert only["name"] == "w2_5 probe"
    assert only["version"] == "9.9.9"
    assert only["apps"] == ["w2_5_probe_app"]
    assert only["model_providers"] == ["w2_5_probe_models"]
    assert only["report_sections"] == ["w2_5_probe_section"]


def test_a_diagnostic_without_an_exception_prints_no_dangling_dash(
        plugin_registry, capsys):
    """The exception suffix is omitted when there is nothing to append."""
    plugin_registry()
    plugins.record_diagnostic("hand-written", "something to say")

    cli_plugins.main(["list"])

    out = capsys.readouterr().out
    assert "[error] hand-written: something to say" in out
    assert "something to say —" not in out


def test_running_the_module_as_a_script_exits_with_its_return_code(
        plugin_registry, monkeypatch):
    """``python -m spacr.cli_plugins`` turns the return value into an exit."""
    import runpy

    plugin_registry("w2_5_broken_plugin")
    monkeypatch.setattr(sys, "argv", ["spacr-plugins", "doctor"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("spacr.cli_plugins", run_name="__main__")

    assert caught.value.code == 1
