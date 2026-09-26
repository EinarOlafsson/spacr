"""Mean-intensity bounds are live settings; old split/merge controls are inert."""

import csv

import pytest

from spacr.cli import load_settings_file
from spacr.object_roles import (
    ALL_ROLES,
    RENAMED_SETTING_SUFFIXES,
    setting_label,
    withdrawn_setting_reason,
)
from spacr.organelle_types import ALL_ORGANELLE_ROLES, organelle_number
from spacr.settings import (
    _fold_renamed_settings,
    categories,
    expected_types,
    set_default_settings_preprocess_generate_masks,
    surviving_setting_name,
    tooltips,
)
from spacr.settings_spec import convert_settings_dict_for_gui
from spacr.validate import WARNING, _check_retired_keys


REMOVED_SUFFIXES = (
    "minimum_area_to_split",
    "min_watershed_distance",
    "intensity_threshold",
    "intensity_merge",
    "intensity_split",
)
OLDER_ALIASES = (
    "min_object_area",
    "min_split_area",
    "min_distance",
    "area_multiplier",
    "intensity_threshold_method",
    "intensity_percentile",
)
PRIMARY_ROLES = ("cell", "nucleus", "pathogen", "organelle")
#: The maintainer, 2026-09-25 (item 511): Mask's per-object area and mean
#: bounds of the Cellpose objects are retired into object_filters rows.
RETIRED_BOUND_ROLES = PRIMARY_ROLES[:3]
BOUND_SUFFIXES = ("min_area", "max_area", "min_intensity", "max_intensity")


def test_bounds_are_float_controls_with_disabled_defaults_in_declared_slots():
    defaults = set_default_settings_preprocess_generate_masks({
        "verbose": False,
        "number_of_organelles": 7,
    })
    bounds = {
        f"{role}_{side}_intensity": defaults[f"{role}_{side}_intensity"]
        for role in ALL_ORGANELLE_ROLES[:7]
        for side in ("min", "max")
    }
    for role in RETIRED_BOUND_ROLES:
        for suffix in BOUND_SUFFIXES:
            assert f"{role}_{suffix}" not in defaults
            assert f"{role}_{suffix}" not in expected_types
    assert defaults["object_filters"] == {}
    spec = convert_settings_dict_for_gui(bounds)
    for key, value in bounds.items():
        assert value == 0.0
        assert type(value) is float
        assert expected_types[key] is float
        assert spec[key] == ("entry", None, 0.0)


def test_every_organelle_slot_has_bounds_beside_area_with_own_channel_help():
    filtration = categories["Object filtration"]
    positions = {key: index for index, key in enumerate(filtration)}
    assert "object_filters" in filtration
    for role in RETIRED_BOUND_ROLES:
        for suffix in BOUND_SUFFIXES:
            assert f"{role}_{suffix}" not in filtration
            assert f"{role}_{suffix}" not in tooltips
    for role in ALL_ORGANELLE_ROLES:
        ordered = [f"{role}_{suffix}" for suffix in (
            "min_area", "max_area", "min_intensity", "max_intensity")]
        first = positions[ordered[0]]
        assert filtration[first:first + len(ordered)] == ordered
        for side in ("min", "max"):
            key = f"{role}_{side}_intensity"
            assert expected_types[key] is float
            assert f"{role}_channel" in tooltips[key]
            assert "mean pixel intensity" in tooltips[key]
            assert "raw units" in tooltips[key]
            assert "0 disables" in tooltips[key]
            assert "Equality is retained" in tooltips[key]
            if role in ALL_ORGANELLE_ROLES:
                label = (f"Organelle {organelle_number(role)} — "
                         f"{side.capitalize()} intensity")
            else:
                label = f"{role.capitalize()} {side} intensity"
            assert setting_label(key) == label


def test_bounds_survive_a_hidden_organelle_slot_and_preserve_explicit_values():
    settings = set_default_settings_preprocess_generate_masks({
        "verbose": False,
        "number_of_organelles": 1,
        "cell_min_intensity": 10.25,
        "organelleq_min_intensity": 20.5,
        "organelleq_max_intensity": 40.75,
    })
    assert "cell_min_intensity" not in settings
    assert settings["object_filters"] == {"cell": [
        {"property": "intensity_mean", "min": 10.25, "max": None}]}
    assert settings["organelleq_min_intensity"] == 20.5
    assert settings["organelleq_max_intensity"] == 40.75
    assert settings["organelle_min_intensity"] == 0.0
    assert settings["organelle_max_intensity"] == 0.0
    assert settings["number_of_organelles"] == 1


def test_removed_controls_and_old_aliases_are_withdrawn_for_every_role():
    defaults = set_default_settings_preprocess_generate_masks({
        "verbose": False, "number_of_organelles": 7})
    offered = set(expected_types) | set(tooltips) | set(defaults)
    offered.update(key for members in categories.values() for key in members)
    for suffix in (*REMOVED_SUFFIXES, *OLDER_ALIASES):
        assert suffix not in RENAMED_SETTING_SUFFIXES
        for role in ALL_ROLES:
            key = f"{role}_{suffix}"
            assert key not in offered
            assert withdrawn_setting_reason(key)
            assert surviving_setting_name(key) == ()
    for role in (*PRIMARY_ROLES, "organelleb", "organelleg"):
        key = f"{role}_perimeter_fraction"
        assert defaults[key] == 0
        assert expected_types[key] is float
        assert key in categories["Object filtration"]
        assert withdrawn_setting_reason(key) is None


@pytest.mark.parametrize("header", [
    ("Key", "Value"), ("setting_key", "setting_value")])
def test_old_csv_loads_without_repurposing_split_or_merge_values(tmp_path, header):
    old_values = {
        f"{role}_{suffix}": 23
        for role in (*PRIMARY_ROLES, "organelleb", "organelleq", "organellezz")
        for suffix in (*REMOVED_SUFFIXES, *OLDER_ALIASES)
    }
    values = {
        **old_values,
        "cell_FT": 0.42,
        "organelleq_min_size": 17,
        "cell_min_intensity": 12.5,
        "organelleq_max_intensity": 82.75,
        "infection_intensity_threshold": 35.5,
        "timelapse_remove_transient": False,
        "verbose": False,
    }
    path = tmp_path / "old-settings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values.items())

    loaded = load_settings_file(str(path))
    assert loaded == values
    settings = set_default_settings_preprocess_generate_masks(loaded)
    for key, value in old_values.items():
        # Retain the key for the withdrawal diagnostic; no active control
        # receives the old value, and none of these names remains offered.
        assert settings[key] == value
        assert key not in expected_types
        assert surviving_setting_name(key) == ()
    assert "cell_min_intensity" not in settings
    assert "cell_max_intensity" not in settings
    assert settings["object_filters"] == {"cell": [
        {"property": "intensity_mean", "min": 12.5, "max": None}]}
    assert settings["organelleq_min_intensity"] == 0.0
    assert settings["organelleq_max_intensity"] == 82.75

    # Positive migration controls prove the loader still applies genuine
    # aliases while leaving the withdrawn merge/split family inert.
    assert settings["cell_flow_threshold"] == 0.42
    assert settings["organelleq_min_area"] == 17
    assert "cell_FT" not in settings
    assert "organelleq_min_size" not in settings
    assert settings["infection_intensity_threshold"] == 35.5
    assert settings["timelapse_remove_transient"] is False
    assert withdrawn_setting_reason("infection_intensity_threshold") is None
    assert withdrawn_setting_reason("timelapse_remove_transient") is None

    problems = _check_retired_keys(old_values)
    assert {problem.setting for problem in problems} == set(old_values)
    assert all(problem.severity == WARNING for problem in problems)
    assert all("read by nothing" in problem.fix for problem in problems)
    assert all("renamed" not in problem.message for problem in problems)
    folded = dict(settings)
    assert _fold_renamed_settings(folded) == settings


def test_bound_api_links_name_the_filter_reader():
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS

    for role in (*PRIMARY_ROLES, "organelleb", "organellec", "organelled"):
        for side in ("min", "max"):
            # The primary organelle generator reads its literal keys to
            # validate the bounds and enable original-channel loading. Its
            # numbered slots reach the same logic through a settings view;
            # their exact reader is the shared role-aware filter instead.
            consumer = ("generate_organelle_masks_sam" if role == "organelle"
                        else "merge_split_filter_masks")
            assert SETTING_API_TARGETS[f"{role}_{side}_intensity"] == (
                "spacr.object", consumer, True)


def test_every_preview_filter_caption_enters_the_runtime_catalog(monkeypatch):
    from importlib import import_module
    from pathlib import Path

    from spacr.qt.widgets.live_preview import COMPARTMENT_FIELDS

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    builder = import_module("build_i18n_catalogs")
    captions = {row[1] for row in COMPARTMENT_FIELDS}
    missing = captions - set(builder.canonical_sources()["ui"])
    assert not missing, f"Live Preview filter captions lack runtime sources: {missing}"
