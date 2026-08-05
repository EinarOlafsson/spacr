"""Plugin SDK discovery and all headless extension contracts."""
from __future__ import annotations

import json
import sys
import types

import pytest

from spacr import plugins


@pytest.fixture
def example_plugin(monkeypatch):
    module = types.ModuleType("spacr_test_plugin")

    def defaults(settings=None):
        result = {"src": "", "distance_px": 3, "save_figures": True}
        result.update(settings or {})
        return result

    def run(settings):
        return settings["distance_px"]

    def validate(settings):
        from spacr.validate import Problem, WARNING
        if settings.get("distance_px", 0) > 20:
            return [Problem(
                WARNING, "distance_px", "Contact distance is unusually large.",
                "Confirm the acquisition pixel size.",
            )]
        return []

    def models():
        return [{
            "key": "contact_model",
            "name": "contact_model.CP_model",
            "kind": "cellpose",
            "source": "remote",
            "uri": "https://example.invalid/contact_model.CP_model",
        }]

    def report_section(context):
        from spacr.report import Section
        return Section(
            key="contacts",
            title="Contact sites",
            body_html=f"<p>{context.src}</p>",
            text_lines=["Contact-site summary"],
        )

    module.defaults = defaults
    module.run = run
    module.validate = validate
    module.models = models
    module.report_section = report_section
    module.plugin = {
        "name": "Test contact assay",
        "version": "0.2.0",
        "api_version": "1.0",
        "apps": [{
            "key": "contact_assay",
            "name": "Contact Assay",
            "description": "Measure organelle contact sites.",
            "entrypoint": "spacr_test_plugin:run",
            "defaults": "spacr_test_plugin:defaults",
            "kind": "assay",
            "section": "results",
            "categories": {
                "Input": ["src"],
                "Detection": ["distance_px"],
                "Output": ["save_figures"],
            },
            "tooltips": {"distance_px": "Maximum contact distance in pixels."},
            "labels": {"distance_px": "Contact distance (px)"},
            "docs_url": "https://example.invalid/contact-api",
            "aliases": ["contacts"],
            "validator": "spacr_test_plugin:validate",
            "requires": ["src — processed plate"],
            "writes": ["contact_sites.csv"],
        }],
        "model_providers": [{
            "key": "contact_models",
            "provider": "spacr_test_plugin:models",
        }],
        "report_sections": [{
            "key": "contacts",
            "title": "Contact sites",
            "builder": "spacr_test_plugin:report_section",
            "after": "statistics",
        }],
        "translations": {
            "sv": {"Contact Assay": "Kontaktanalys"},
            "de": {"Contact Assay": "Kontaktanalyse"},
        },
    }
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_test_plugin:plugin")
    plugins.reload_plugins()
    yield module
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    sys.modules.pop(module.__name__, None)
    plugins.reload_plugins()


def test_discovery_validates_and_exposes_every_contribution(example_plugin):
    discovered = plugins.discover_plugins()
    assert [plugin.name for plugin in discovered] == ["Test contact assay"]
    app = plugins.get_app("contact_assay")
    assert app is not None
    assert app.categories["Detection"] == ("distance_px",)
    assert plugins.load_object(app.entrypoint)({"distance_px": 7}) == 7
    assert [item.key for _owner, item in plugins.model_providers()] == [
        "contact_models"
    ]
    assert [item.key for _owner, item in plugins.report_sections()] == ["contacts"]
    assert plugins.diagnostics() == ()


def test_incompatible_api_and_bad_references_are_rejected():
    with pytest.raises(ValueError, match="requires SDK"):
        plugins.plugin_from_mapping({
            "name": "Future", "version": "1", "api_version": "2.0",
        })
    with pytest.raises(ValueError, match="invalid entrypoint"):
        plugins.plugin_from_mapping({
            "name": "Broken",
            "version": "1",
            "apps": [{
                "key": "broken_app",
                "name": "Broken",
                "description": "Broken on purpose.",
                "entrypoint": "not a reference",
                "defaults": "example:defaults",
            }],
        })


def test_plugin_validator_is_additive_and_never_treated_as_unknown(example_plugin):
    from spacr.validate import validate_settings
    problems = validate_settings(
        {"src": "", "distance_px": 25, "save_figures": True},
        "contact_assay",
    )
    assert any(problem.setting == "distance_px" for problem in problems)
    assert not any("unknown app" in problem.message for problem in problems)


def test_model_provider_extends_catalogue_without_network(example_plugin):
    from spacr.model_zoo import catalogue
    entries = catalogue(
        include_bundled=False, remote=False, include_plugins=True
    )
    assert [(entry.key, entry.name) for entry in entries] == [
        ("contact_model", "contact_model.CP_model")
    ]


def test_report_builder_inserts_after_requested_core_section(
    example_plugin, tmp_path,
):
    from spacr.report import collect_report
    report = collect_report(tmp_path, search_journal=False)
    keys = [section.key for section in report.sections]
    assert keys.index("contacts") == keys.index("statistics") + 1
    assert report.section("contacts").title == "Contact sites"


def test_cli_lists_plugins_and_emits_machine_readable_diagnostics(
    example_plugin, capsys,
):
    from spacr.cli_plugins import main
    assert main(["list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plugins"][0]["apps"] == ["contact_assay"]
    assert payload["diagnostics"] == []


def test_a_broken_plugin_is_isolated_and_doctor_fails(monkeypatch, capsys):
    monkeypatch.setenv(
        "SPACR_PLUGIN_MODULES",
        "does_not_exist:plugin,json:no_such_plugin_attribute",
    )
    plugins.reload_plugins()
    try:
        assert plugins.discover_plugins() == ()
        assert len(plugins.diagnostics()) == 2
        from spacr.cli_plugins import main
        assert main(["doctor"]) == 1
        assert "Diagnostics:" in capsys.readouterr().out
    finally:
        monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
        plugins.reload_plugins()
