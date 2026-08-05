"""Every rejection the plugin SDK makes, and what the caller is told.

``tests/test_plugins.py`` drives the happy path: one well-formed plugin is
discovered and every contribution reaches the app registry, the model
catalogue and the report. What it never touches is the other half of the
contract -- the twenty-odd guards that decide a third-party manifest is
malformed. Those guards are the whole reason discovery is failure-isolated,
so each one is pinned here by the error it raises AND by the message it
raises it with: a plugin author reading "invalid app key" has to be able to
tell it from "unknown section".

Discovery is also asserted to stay isolated: a plugin that collides with an
already-registered key is recorded as a diagnostic and the plugin that got
there first keeps working, rather than the whole registry failing shut.
"""
from __future__ import annotations

import sys
import types
from importlib import metadata

import pytest

from spacr import plugins


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

#: The minimum an app contribution needs to be accepted, so each test below
#: can change exactly one field and know that field is why it was rejected.
GOOD_APP = {
    "key": "contact_assay",
    "name": "Contact Assay",
    "description": "Measure organelle contact sites.",
    "entrypoint": "spacr_contract_plugin:run",
    "defaults": "spacr_contract_plugin:defaults",
}


def _manifest(**overrides):
    """A valid one-app manifest with ``overrides`` merged into the app."""
    app = dict(GOOD_APP)
    app.update(overrides)
    return {"name": "Contract plugin", "version": "0.1.0", "apps": [app]}


@pytest.fixture(autouse=True)
def _restore_registry():
    """Discovery is a process-wide cache; never leak one test into the next."""
    yield
    plugins.reload_plugins()


@pytest.fixture
def registered_module(monkeypatch):
    """Install an importable module and point discovery at an attribute of it."""
    def _install(name, **attributes):
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return _install


# --------------------------------------------------------------------------
# load_object
# --------------------------------------------------------------------------

def test_load_object_resolves_a_nested_attribute_path():
    """``module:a.b`` walks attributes, which is what screen factories use."""
    assert plugins.load_object("json:dumps")({"a": 1}) == '{"a": 1}'
    assert plugins.load_object("json:JSONDecodeError.__name__") == "JSONDecodeError"


@pytest.mark.parametrize("reference", [
    "json.dumps",          # dot instead of colon
    ":dumps",              # no module
    "json:",               # no attribute
    "json:1bad",           # attribute is not an identifier
    None,                  # not a string at all
    b"json:dumps",         # bytes are not a reference either
])
def test_load_object_refuses_anything_that_is_not_module_colon_attribute(reference):
    """A malformed reference is refused before any import is attempted."""
    with pytest.raises(ValueError, match="invalid object reference"):
        plugins.load_object(reference)


def test_load_object_names_the_reference_it_refused():
    """The author has to be able to see which of their strings was wrong."""
    with pytest.raises(ValueError) as excinfo:
        plugins.load_object("spacr_typo_here")
    assert "spacr_typo_here" in str(excinfo.value)
    assert "package.module:attribute" in str(excinfo.value)


# --------------------------------------------------------------------------
# sequence and mapping fields
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["aliases", "requires", "writes"])
def test_sequence_fields_refuse_a_bare_string(field):
    """``aliases="contacts"`` would silently become six one-character aliases."""
    with pytest.raises(TypeError, match=f"{field} must be a sequence of strings"):
        plugins.plugin_from_mapping(_manifest(**{field: "contacts"}))


@pytest.mark.parametrize("field", ["aliases", "requires", "writes"])
def test_sequence_fields_refuse_a_blank_entry(field):
    """A blank alias would register an unreachable empty key."""
    with pytest.raises(ValueError, match=f"{field} cannot contain blank values"):
        plugins.plugin_from_mapping(_manifest(**{field: ["contacts", "   "]}))


@pytest.mark.parametrize("field", ["aliases", "requires", "writes"])
def test_sequence_fields_default_to_empty_and_strip_their_entries(field):
    """``None`` means "not provided"; provided values are stripped, not trusted."""
    plugin = plugins.plugin_from_mapping(_manifest(**{field: [" contacts "]}))
    assert getattr(plugin.apps[0], field) == ("contacts",)
    omitted = plugins.plugin_from_mapping(_manifest(**{field: None}))
    assert getattr(omitted.apps[0], field) == ()


@pytest.mark.parametrize("field", ["tooltips", "labels"])
def test_string_maps_refuse_a_non_mapping(field):
    """A list of pairs is a plausible mistake and would index by position."""
    with pytest.raises(TypeError, match=f"{field} must be a mapping"):
        plugins.plugin_from_mapping(_manifest(**{field: [("distance_px", "px")]}))


@pytest.mark.parametrize("field", ["tooltips", "labels"])
def test_string_maps_coerce_keys_and_values_to_strings(field):
    """A YAML manifest can hand over ints; the GUI needs text."""
    plugin = plugins.plugin_from_mapping(_manifest(**{field: {3: 7}}))
    assert getattr(plugin.apps[0], field) == {"3": "7"}


def test_categories_must_map_a_tab_name_to_setting_keys():
    with pytest.raises(TypeError, match="categories must be a mapping"):
        plugins.plugin_from_mapping(_manifest(categories=["src"]))


def test_categories_report_which_tab_held_the_bad_keys():
    """A manifest with ten tabs must say which one is wrong."""
    with pytest.raises(TypeError, match=r"categories\['Detection'\]"):
        plugins.plugin_from_mapping(
            _manifest(categories={"Input": ["src"], "Detection": "distance_px"})
        )


# --------------------------------------------------------------------------
# app contributions
# --------------------------------------------------------------------------

def test_app_entries_must_be_contributions_or_mappings():
    with pytest.raises(TypeError, match="apps entries must be"):
        plugins.plugin_from_mapping({
            "name": "Contract plugin", "version": "0.1.0", "apps": ["contact_assay"],
        })


def test_an_already_built_contribution_is_validated_too():
    """Handing over an ``AppContribution`` does not bypass the vocabulary check."""
    bad = plugins.AppContribution(
        key="contact_assay", name="Contact Assay", description="Contacts.",
        entrypoint="spacr_contract_plugin:run",
        defaults="spacr_contract_plugin:defaults", section="nowhere",
    )
    with pytest.raises(ValueError, match="unknown section 'nowhere'"):
        plugins.plugin_from_mapping({
            "name": "Contract plugin", "version": "0.1.0", "apps": [bad],
        })


@pytest.mark.parametrize("key", [
    "Contact",        # capital
    "1contact",       # leading digit
    "c",              # single character
    "contact-assay",  # hyphen
    "",               # empty
    "x" * 65,         # longer than the 64-character limit
])
def test_app_keys_are_lowercase_identifiers_of_two_to_sixty_four_characters(key):
    with pytest.raises(ValueError, match="invalid app key"):
        plugins.plugin_from_mapping(_manifest(key=key))


def test_a_sixty_four_character_key_is_still_accepted():
    """The boundary is inclusive; 64 is legal and 65 is not."""
    key = "c" + "x" * 63
    plugin = plugins.plugin_from_mapping(_manifest(key=key))
    assert plugin.apps[0].key == key


@pytest.mark.parametrize("field", ["name", "description"])
def test_an_app_needs_a_visible_name_and_description(field):
    """Whitespace is not a name -- the launcher would show an empty tile."""
    with pytest.raises(ValueError, match="needs a name and description"):
        plugins.plugin_from_mapping(_manifest(**{field: "   "}))


@pytest.mark.parametrize("field,value,message", [
    ("section", "nowhere", "unknown section 'nowhere'"),
    ("stage", "prerelease", "unknown stage 'prerelease'"),
    ("kind", "widget", "unknown kind 'widget'"),
    ("call_style", "positional", "invalid call_style"),
])
def test_the_four_closed_vocabularies_name_the_value_they_rejected(
    field, value, message,
):
    with pytest.raises(ValueError, match=message):
        plugins.plugin_from_mapping(_manifest(**{field: value}))


@pytest.mark.parametrize("field", ["section", "stage", "kind", "call_style"])
def test_every_documented_member_of_each_vocabulary_is_accepted(field):
    """The guard must not have drifted narrower than the SDK's own constants."""
    allowed = {
        "section": plugins._SECTIONS,
        "stage": plugins._STAGES,
        "kind": plugins._KINDS,
        "call_style": plugins._CALL_STYLES,
    }[field]
    for value in sorted(allowed):
        plugin = plugins.plugin_from_mapping(_manifest(**{field: value}))
        assert getattr(plugin.apps[0], field) == value


@pytest.mark.parametrize("field", [
    "entrypoint", "defaults", "validator", "screen_factory", "drop_handler",
])
def test_every_reference_field_is_checked_and_names_itself(field):
    with pytest.raises(ValueError, match=f"invalid {field} reference"):
        plugins.plugin_from_mapping(_manifest(**{field: "not a reference"}))


@pytest.mark.parametrize("field", ["validator", "screen_factory", "drop_handler"])
def test_the_optional_reference_fields_may_be_left_empty(field):
    """Empty means "not contributed" and must not be validated as a reference."""
    plugin = plugins.plugin_from_mapping(_manifest(**{field: ""}))
    assert getattr(plugin.apps[0], field) == ""


# --------------------------------------------------------------------------
# model provider and report section contributions
# --------------------------------------------------------------------------

def test_model_provider_entries_must_be_contributions_or_mappings():
    with pytest.raises(TypeError, match="model_providers entries must be"):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1", "model_providers": ["contact_models"],
        })


@pytest.mark.parametrize("entry", [
    {"key": "Contact", "provider": "spacr_contract_plugin:models"},
    {"key": "contact_models", "provider": "spacr_contract_plugin.models"},
])
def test_a_model_provider_needs_a_valid_key_and_callable_reference(entry):
    with pytest.raises(ValueError, match="model provider needs a valid key"):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1", "model_providers": [entry],
        })


def test_report_section_entries_must_be_contributions_or_mappings():
    with pytest.raises(TypeError, match="report_sections entries must be"):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1", "report_sections": ["contacts"],
        })


@pytest.mark.parametrize("entry,message", [
    ({"key": "Contacts", "title": "Contact sites",
      "builder": "spacr_contract_plugin:section"},
     "report section needs a valid key and title"),
    ({"key": "contacts", "title": "   ",
      "builder": "spacr_contract_plugin:section"},
     "report section needs a valid key and title"),
    ({"key": "contacts", "title": "Contact sites", "builder": "not a reference"},
     "report section builder must be"),
])
def test_a_report_section_needs_a_key_a_title_and_a_builder(entry, message):
    with pytest.raises(ValueError, match=message):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1", "report_sections": [entry],
        })


def test_a_report_section_defaults_to_following_the_statistics_section():
    """The insertion point is part of the contract, not an implementation note."""
    plugin = plugins.plugin_from_mapping({
        "name": "P", "version": "1", "report_sections": [{
            "key": "contacts", "title": "Contact sites",
            "builder": "spacr_contract_plugin:section",
        }],
    })
    assert plugin.report_sections[0].after == "statistics"


# --------------------------------------------------------------------------
# the manifest as a whole
# --------------------------------------------------------------------------

def test_a_manifest_must_be_a_mapping():
    with pytest.raises(TypeError, match="plugin manifest must be a mapping"):
        plugins.plugin_from_mapping([("name", "P"), ("version", "1")])


def test_translations_must_map_language_codes_to_message_mappings():
    with pytest.raises(TypeError, match="translations must map language codes"):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1", "translations": ["sv"],
        })


def test_one_bad_language_is_named_by_its_code():
    with pytest.raises(TypeError, match=r"translations\['sv'\] must be a mapping"):
        plugins.plugin_from_mapping({
            "name": "P", "version": "1",
            "translations": {"de": {"Contact Assay": "Kontaktanalyse"},
                             "sv": ["Kontaktanalys"]},
        })


@pytest.mark.parametrize("field", ["name", "version"])
def test_a_plugin_needs_a_name_and_a_version(field):
    manifest = {"name": "P", "version": "1"}
    manifest[field] = "  "
    with pytest.raises(ValueError, match="plugin name and version are required"):
        plugins.plugin_from_mapping(manifest)


def test_a_matching_major_api_version_is_enough():
    """1.4 is accepted against SDK 1.0; only the major number is a barrier."""
    plugin = plugins.plugin_from_mapping(
        {"name": "P", "version": "1", "api_version": "1.4"}
    )
    assert plugin.api_version == "1.4"


@pytest.mark.parametrize("label,field,entries", [
    ("app", "apps", [dict(GOOD_APP), dict(GOOD_APP, name="Second")]),
    ("model provider", "model_providers", [
        {"key": "contact_models", "provider": "spacr_contract_plugin:models"},
        {"key": "contact_models", "provider": "spacr_contract_plugin:other"},
    ]),
    ("report section", "report_sections", [
        {"key": "contacts", "title": "One",
         "builder": "spacr_contract_plugin:one"},
        {"key": "contacts", "title": "Two",
         "builder": "spacr_contract_plugin:two"},
    ]),
])
def test_one_plugin_may_not_repeat_a_key_within_a_group(label, field, entries):
    """Repeating a key inside one manifest silently drops a contribution."""
    with pytest.raises(ValueError, match=f"repeats {label} key"):
        plugins.plugin_from_mapping({"name": "P", "version": "1", field: entries})


def test_the_same_key_in_two_different_groups_is_fine():
    """The namespaces are separate; only within-group collisions are errors."""
    plugin = plugins.plugin_from_mapping({
        "name": "P", "version": "1",
        "model_providers": [
            {"key": "contacts", "provider": "spacr_contract_plugin:models"},
        ],
        "report_sections": [
            {"key": "contacts", "title": "Contact sites",
             "builder": "spacr_contract_plugin:section"},
        ],
    })
    assert plugin.model_providers[0].key == "contacts"
    assert plugin.report_sections[0].key == "contacts"


def test_a_hand_built_plugin_with_non_mapping_translations_is_refused():
    """``_coerce_plugin`` validates a ready-made object, not just a mapping."""
    built = plugins.SpacrPlugin(name="P", version="1", translations=["sv"])
    with pytest.raises(TypeError, match="plugin translations must be a mapping"):
        plugins._validate_plugin(built)


# --------------------------------------------------------------------------
# what an entry point is allowed to return
# --------------------------------------------------------------------------

def test_an_entry_point_may_return_a_ready_made_plugin(monkeypatch, registered_module):
    """A ``SpacrPlugin`` instance is accepted and re-validated on the way in."""
    built = plugins.plugin_from_mapping(_manifest())
    registered_module("spacr_contract_plugin", plugin=built)
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin:plugin")
    assert [p.name for p in plugins.reload_plugins()] == ["Contract plugin"]
    assert plugins.get_app("contact_assay").name == "Contact Assay"


def test_an_entry_point_may_return_a_zero_argument_factory(
    monkeypatch, registered_module,
):
    """A factory is called once; discovery caches the result, not the callable."""
    calls = []

    def factory():
        calls.append(1)
        return _manifest()

    registered_module("spacr_contract_plugin", plugin=factory)
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin:plugin")
    plugins.reload_plugins()
    assert plugins.discover_plugins() == plugins.discover_plugins()
    assert calls == [1]


def test_an_entry_point_returning_something_else_is_isolated(
    monkeypatch, registered_module,
):
    """A string is neither a plugin nor a manifest, and must not abort spaCR."""
    registered_module("spacr_contract_plugin", plugin="Contract plugin")
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin:plugin")
    assert plugins.reload_plugins() == ()
    diagnostic, = plugins.diagnostics()
    assert diagnostic.severity == "error"
    assert "entry point must expose SpacrPlugin" in diagnostic.exception


def test_a_module_reference_without_a_colon_looks_for_a_plugin_attribute(
    monkeypatch, registered_module,
):
    """``SPACR_PLUGIN_MODULES=pkg.mod`` means ``pkg.mod:plugin``."""
    registered_module("spacr_contract_plugin", plugin=_manifest())
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin")
    assert [p.name for p in plugins.reload_plugins()] == ["Contract plugin"]


# --------------------------------------------------------------------------
# discovery: isolation and the off switch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_the_disable_switch_yields_an_empty_registry(
    monkeypatch, registered_module, value,
):
    """A user who turns plugins off gets no apps and no diagnostics either."""
    registered_module("spacr_contract_plugin", plugin=_manifest())
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin:plugin")
    monkeypatch.setenv("SPACR_DISABLE_PLUGINS", value)
    assert plugins.reload_plugins() == ()
    assert plugins.plugin_apps() == ()
    assert plugins.diagnostics() == ()
    assert plugins.get_app("contact_assay") is None


@pytest.mark.parametrize("value", ["0", "off", "", "  "])
def test_a_falsey_disable_switch_leaves_plugins_on(
    monkeypatch, registered_module, value,
):
    registered_module("spacr_contract_plugin", plugin=_manifest())
    monkeypatch.setenv("SPACR_PLUGIN_MODULES", "spacr_contract_plugin:plugin")
    monkeypatch.setenv("SPACR_DISABLE_PLUGINS", value)
    assert [p.name for p in plugins.reload_plugins()] == ["Contract plugin"]


@pytest.mark.parametrize("group,second", [
    ("apps", {"apps": [dict(GOOD_APP)]}),
    ("model_providers", {"model_providers": [
        {"key": "contact_models", "provider": "spacr_contract_plugin:models"},
    ]}),
    ("report_sections", {"report_sections": [
        {"key": "contacts", "title": "Contact sites",
         "builder": "spacr_contract_plugin:section"},
    ]}),
])
def test_a_second_plugin_reusing_a_key_loses_without_taking_the_first_down(
    monkeypatch, registered_module, group, second,
):
    """First registration wins; the loser is a diagnostic, not a crash."""
    first = {"name": "First", "version": "1"}
    first.update(second)
    registered_module("spacr_contract_plugin", plugin=first)
    registered_module(
        "spacr_contract_plugin_two",
        plugin=dict(first, name="Second"),
    )
    monkeypatch.setenv(
        "SPACR_PLUGIN_MODULES",
        "spacr_contract_plugin:plugin,spacr_contract_plugin_two:plugin",
    )
    assert [p.name for p in plugins.reload_plugins()] == ["First"]
    diagnostic, = plugins.diagnostics()
    assert diagnostic.plugin == "spacr_contract_plugin_two:plugin"
    assert "already registered" in diagnostic.exception
    # Exactly one contribution survives in the colliding group, and it is the
    # one the first plugin registered -- not a silently overwritten second.
    if group == "apps":
        assert [app.key for app in plugins.plugin_apps()] == ["contact_assay"]
        assert plugins.get_app("contact_assay").name == "Contact Assay"
    elif group == "model_providers":
        assert [(owner, item.key) for owner, item in plugins.model_providers()] == [
            ("First", "contact_models")
        ]
    else:
        assert [(owner, item.key) for owner, item in plugins.report_sections()] == [
            ("First", "contacts")
        ]


def test_a_broken_entry_point_group_is_reported_rather_than_raised(monkeypatch):
    """If importlib.metadata itself fails, spaCR still starts."""
    def explode():
        raise RuntimeError("entry point cache is corrupt")

    monkeypatch.setattr(metadata, "entry_points", explode)
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    assert plugins.reload_plugins() == ()
    diagnostic, = plugins.diagnostics()
    assert diagnostic.plugin == "entry-point discovery"
    assert "entry point cache is corrupt" in diagnostic.exception


def test_installed_entry_points_are_loaded_in_a_deterministic_order(monkeypatch):
    """Two plugins must contribute in the same order on every machine."""
    seen = []

    class _Point:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def load(self):
            seen.append(self.name)
            return {"name": self.name, "version": "1"}

    class _Points:
        def select(self, group):
            assert group == plugins.PLUGIN_ENTRY_POINT_GROUP
            return [_Point("zeta", "m:z"), _Point("alpha", "m:a")]

    monkeypatch.setattr(metadata, "entry_points", _Points)
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    assert [p.name for p in plugins.reload_plugins()] == ["alpha", "zeta"]
    assert seen == ["alpha", "zeta"]


# --------------------------------------------------------------------------
# runtime diagnostics
# --------------------------------------------------------------------------

def test_record_diagnostic_reaches_the_reader_without_raising(caplog):
    """A model provider that fails at call time is reported, not swallowed."""
    plugins.reload_plugins()
    with caplog.at_level("ERROR", logger="spacr.plugins"):
        plugins.record_diagnostic(
            "Contract plugin", "model provider failed", KeyError("uri"),
        )
    diagnostic, = plugins.diagnostics()
    assert diagnostic.plugin == "Contract plugin"
    assert diagnostic.severity == "error"
    assert diagnostic.message == "model provider failed"
    assert diagnostic.exception == "'uri'"
    assert "model provider failed" in caplog.text


def test_record_diagnostic_keeps_the_severity_it_was_given():
    """A warning must not be reported to the user as a load failure."""
    plugins.reload_plugins()
    plugins.record_diagnostic("Contract plugin", "slow catalogue", severity="warning")
    diagnostic, = plugins.diagnostics()
    assert diagnostic.severity == "warning"
    assert diagnostic.exception == ""


def test_reload_discards_diagnostics_recorded_against_the_old_registry():
    plugins.record_diagnostic("Contract plugin", "transient failure")
    assert len(plugins.diagnostics()) == 1
    plugins.reload_plugins()
    assert plugins.diagnostics() == ()
