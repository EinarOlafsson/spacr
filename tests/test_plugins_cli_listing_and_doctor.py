"""``spacr-plugins`` reports what is installed and what failed to load.

Real plugin modules are registered through ``SPACR_PLUGIN_MODULES`` and
rediscovered with :func:`spacr.plugins.reload_plugins`, and the command's
actual output is read: the listing, the empty-install sentence, the doctor's
verdict and exit code, the JSON payload, a hand-recorded diagnostic and the
``python -m`` entry point.
"""
from __future__ import annotations

import json
import runpy
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

_MODULES = ("w2_5_probe_plugin", "w2_5_broken_plugin")


@pytest.fixture
def plugin_registry(monkeypatch, tmp_path):
    """Install plugin modules by name and rediscover, restoring afterwards."""
    (tmp_path / "w2_5_probe_plugin.py").write_text(PLUGIN_SOURCE, encoding="utf-8")
    (tmp_path / "w2_5_broken_plugin.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    def install(*modules):
        monkeypatch.setenv(plugins.PLUGIN_MODULES_ENV, ",".join(modules))
        for name in _MODULES:
            sys.modules.pop(name, None)
        return plugins.reload_plugins()

    try:
        yield install
    finally:
        monkeypatch.undo()
        for name in _MODULES:
            sys.modules.pop(name, None)
        plugins.reload_plugins()


def test_plugin_cli_reports_contributions_diagnostics_and_exit_codes(
        subtests, plugin_registry, capsys, monkeypatch):
    """Each install state prints its exact report and returns its exit code."""
    header = f"spaCR plugin SDK {plugins.PLUGIN_API_VERSION}"
    probe_listing = [
        header,
        "w2_5 probe 9.9.9 (API 1.0)",
        "  apps: w2_5_probe_app",
        "  model providers: w2_5_probe_models",
        "  report sections: w2_5_probe_section",
    ]

    with subtests.test(msg="a real plugin lists every contribution"):
        loaded = plugin_registry("w2_5_probe_plugin")
        assert [p.name for p in loaded] == ["w2_5 probe"]
        capsys.readouterr()
        assert cli_plugins.main([]) == 0
        assert capsys.readouterr().out.splitlines() == probe_listing

    with subtests.test(msg="a clean doctor gives a verdict and exits zero"):
        assert cli_plugins.main(["doctor"]) == 0
        assert capsys.readouterr().out.splitlines() == [
            *probe_listing, "No plugin errors recorded."]

    with subtests.test(msg="json output carries the same facts"):
        assert cli_plugins.main(["list", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "sdk_version": plugins.PLUGIN_API_VERSION,
            "apps": ["w2_5_probe_app"],
            "diagnostics": [],
            "plugins": [{
                "name": "w2_5 probe", "version": "9.9.9",
                "api_version": "1.0",
                "apps": ["w2_5_probe_app"],
                "model_providers": ["w2_5_probe_models"],
                "report_sections": ["w2_5_probe_section"],
            }],
        }

    with subtests.test(msg="an empty install says so"):
        assert plugin_registry() == ()
        capsys.readouterr()
        assert cli_plugins.main([]) == 0
        assert capsys.readouterr().out.splitlines() == [
            header, "No plugins discovered."]

    with subtests.test(msg="a diagnostic without an exception has no dash"):
        plugins.record_diagnostic("hand-written", "something to say")
        assert cli_plugins.main(["list"]) == 0
        assert capsys.readouterr().out.splitlines() == [
            header, "No plugins discovered.", "Diagnostics:",
            "  [error] hand-written: something to say"]

    with subtests.test(msg="a plugin that will not load fails the doctor"):
        assert plugin_registry("w2_5_broken_plugin") == ()
        capsys.readouterr()
        assert cli_plugins.main(["doctor"]) == 1
        lines = capsys.readouterr().out.splitlines()
        assert lines[:3] == [header, "No plugins discovered.", "Diagnostics:"]
        assert lines[3:] == [
            "  [error] w2_5_broken_plugin: Could not load plugin "
            "'w2_5_broken_plugin' — "
            "ValueError('plugin name and version are required')"]
        assert "No plugin errors recorded." not in lines
        # The listing is not a doctor: it reports the same failure but exits 0.
        assert cli_plugins.main(["list"]) == 0
        assert capsys.readouterr().out.splitlines() == lines

    with subtests.test(msg="python -m turns the return code into an exit"):
        monkeypatch.setattr(sys, "argv", ["spacr-plugins", "doctor"])
        with pytest.raises(SystemExit) as caught:
            runpy.run_module("spacr.cli_plugins", run_name="__main__")
        assert caught.value.code == 1
