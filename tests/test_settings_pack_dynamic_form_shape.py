"""317: pack migration must preserve values that reveal this app's own slots."""
from __future__ import annotations

import csv
from copy import deepcopy

import pytest

# Load Qt before modules that can import scientific libraries and their plugins.
pytest.importorskip("PySide6.QtCore")

from spacr.qt.screens.settings_model import resolve_default_settings
from spacr.qt.settings_pack import settings_from_pack
from spacr.settings import expected_types, surviving_setting_name


def _write_pack(tmp_path, rows):
    path = tmp_path / "gen_masks_settings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Key", "Value"))
        writer.writerows(rows)
    return tmp_path


@pytest.mark.parametrize("count_position", ["first", "last", "omitted"])
def test_same_module_slot_values_survive_the_pack_shape_change(
        tmp_path, count_position):
    """Use the real zero-slot schema, including old files with inferred count."""
    defaults = resolve_default_settings("mask")
    assert defaults["number_of_organelles"] == 0
    assert "organelleb_channel" not in defaults
    assert "organelleb_min_intensity" not in defaults
    rows = [
        ("organelleb_channel", 3),
        ("organelleb_min_intensity", 12.5),
        ("organelleb_max_intensity", 90.25),
        ("verbose", False),
    ]
    if count_position == "first":
        rows.insert(0, ("number_of_organelles", 2))
    elif count_position == "last":
        rows.append(("number_of_organelles", 2))

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["number_of_organelles"] == 2
    for key, value in rows:
        assert key in settings, f"same-module setting was discarded: {key}"
        assert settings[key] == value
    assert set(report.applied) == {key for key, _value in rows}
    assert report.renamed == []
    assert report.dropped == []
    assert report.elsewhere == []
    assert report.malformed == 0
    assert report.source == "gen_masks_settings.csv"


def test_slot_rename_reaches_the_dynamically_revealed_destination(tmp_path):
    """A valid alias must not become 'no such setting' because its slot is hidden."""
    assert surviving_setting_name("organelleb_min_size") == (
        "organelleb_min_area",)
    assert "organelleb_min_area" not in resolve_default_settings("mask")
    rows = [
        ("organelleb_min_size", 17),
        ("organelleb_channel", 3),
        ("number_of_organelles", 2),
    ]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["number_of_organelles"] == 2
    assert settings.get("organelleb_channel") == 3
    assert settings.get("organelleb_min_area") == 17
    assert "organelleb_min_size" not in settings
    assert set(report.applied) == {"organelleb_channel", "number_of_organelles"}
    assert report.renamed == [("organelleb_min_size", "organelleb_min_area")]
    assert report.dropped == []
    assert report.elsewhere == []


@pytest.mark.parametrize("primary_key", [
    "organelle_min_area", "organelle_min_size",
], ids=["current-primary", "renamed-primary"])
@pytest.mark.parametrize("secondary_value", [None, 77],
                         ids=["inherit-primary", "explicit-secondary-wins"])
def test_new_slots_inherit_the_packs_primary_values_not_unmodified_defaults(
        tmp_path, primary_key, secondary_value):
    """The pipeline clones the current primary values when revealing a slot."""
    defaults = resolve_default_settings("mask")
    assert defaults["organelle_min_area"] == 10
    assert "organelleb_min_area" not in defaults
    rows = [("number_of_organelles", 2), (primary_key, 30)]
    if secondary_value is not None:
        rows.append(("organelleb_min_area", secondary_value))

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["organelle_min_area"] == 30
    assert settings.get("organelleb_min_area") == (
        30 if secondary_value is None else secondary_value)
    assert settings["number_of_organelles"] == 2
    assert report.dropped == []
    assert report.elsewhere == []
    if primary_key == "organelle_min_size":
        assert report.renamed == [(primary_key, "organelle_min_area")]
        assert primary_key not in settings
    else:
        assert report.renamed == []


@pytest.mark.parametrize("legacy_last", [False, True],
                         ids=["legacy-first", "legacy-last"])
@pytest.mark.parametrize("legacy, current, value, stale, count, curated", [
    ("cell_FT", "cell_flow_threshold", 0.25, 75, None, False),
    ("old_flow_317", "cell_flow_threshold", 0.25, 75, None, True),
    ("organelleb_min_size", "organelleb_min_area", 31, 90, 2, False),
    ("organelle_min_size", "organelle_min_area", 31, 90, 2, False),
], ids=["package-rename", "app-rename", "secondary-slot", "primary-slot"])
def test_current_pack_spelling_wins_over_its_alias_in_either_row_order(
        tmp_path, monkeypatch, legacy_last, legacy, current, value, stale,
        count, curated):
    """Migrating an alias must not overwrite an explicitly supplied current key."""
    if curated:
        from spacr.qt.settings_pack import PACK_RENAMES

        monkeypatch.setitem(PACK_RENAMES, "mask", {legacy: current})
    else:
        assert surviving_setting_name(legacy) == (current,)
    defaults = resolve_default_settings("mask")
    assert legacy not in defaults
    if current == "organelleb_min_area":
        assert current not in defaults
    rows = [(legacy, stale), (current, value)]
    if legacy_last:
        rows.reverse()
    if count is not None:
        rows.append(("number_of_organelles", count))

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings[current] == value
    assert legacy not in settings
    if count is not None:
        assert settings["number_of_organelles"] == count
    if current == "organelle_min_area":
        # The schema-expansion prepass must choose the current value too;
        # otherwise a corrected final loop still leaves stale cloned values.
        assert settings["organelleb_min_area"] == value
    assert set(report.applied) == {
        current, *(() if count is None else ("number_of_organelles",)),
    }
    assert report.renamed == [(legacy, current)]
    assert report.dropped == []
    assert report.elsewhere == []
    assert report.malformed == 0
    # Both CSV keys are accounted for, but they name one form setting.
    assert (len(report.applied) + len(report.renamed)
            + len(report.dropped) + report.malformed) == len(rows)


def test_revealing_slots_does_not_admit_other_modules_or_unknown_settings(tmp_path):
    """Positive control: accepting every global setting is not the repair."""
    assert "learning_rate" in expected_types
    assert "learning_rate" not in resolve_default_settings("mask")
    unknown = "setting_that_never_existed_317"
    assert unknown not in expected_types
    rows = [("verbose", False), ("learning_rate", 0.001), (unknown, 7)]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["verbose"] is False
    assert "learning_rate" not in settings
    assert unknown not in settings
    assert report.applied == ["verbose"]
    assert report.renamed == []
    assert set(report.dropped) == {"learning_rate", unknown}
    assert report.elsewhere == ["learning_rate"]
    assert report.source == "gen_masks_settings.csv"
    assert report.malformed == 0


def test_caller_defaults_define_the_expandable_slot_schema_without_mutation(tmp_path):
    """A narrow caller schema must not become the global Mask schema."""
    defaults = {
        "number_of_organelles": 0,
        "organelle_channel": None,
        "organelle_min_area": 99,
        "organelle_min_intensity": 7.25,
        "channels": [0, 1],
        "verbose": False,
    }
    before = deepcopy(defaults)
    rows = [
        ("number_of_organelles", 2),
        ("organelleb_channel", 3),
        ("organelleb_min_size", 17),
        ("organelleb_min_intensity", 12.5),
        ("organelleb_max_intensity", 90.25),
        ("learning_rate", 0.001),
    ]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)), defaults=defaults)

    assert defaults == before
    assert settings is not defaults
    assert settings["number_of_organelles"] == 2
    assert settings.get("organelleb_channel") == 3
    assert settings.get("organelleb_min_area") == 17
    assert settings.get("organelleb_min_intensity") == 12.5
    assert settings["organelle_min_area"] == 99
    assert settings["organelle_min_intensity"] == 7.25
    assert settings["channels"] == [0, 1]
    assert "organelleb_max_intensity" not in settings
    assert "learning_rate" not in settings
    assert report.renamed == [("organelleb_min_size", "organelleb_min_area")]
    assert set(report.applied) == {
        "number_of_organelles", "organelleb_channel", "organelleb_min_intensity"}
    assert set(report.dropped) == {"organelleb_max_intensity", "learning_rate"}
    assert set(report.elsewhere) == set(report.dropped)


@pytest.mark.parametrize("supports_count", [False, True])
def test_caller_schema_without_slot_fields_does_not_acquire_them(
        tmp_path, supports_count):
    defaults = {"verbose": True}
    if supports_count:
        defaults["number_of_organelles"] = 0
    before = dict(defaults)
    rows = [("number_of_organelles", 2), ("organelleb_channel", 3),
            ("organelleb_min_intensity", 12.5), ("verbose", False)]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)), defaults=defaults)

    assert defaults == before
    assert settings["verbose"] is False
    assert "organelleb_channel" not in settings
    assert "organelleb_min_intensity" not in settings
    assert set(report.dropped) == {
        "organelleb_channel", "organelleb_min_intensity",
        *(() if supports_count else ("number_of_organelles",)),
    }
    if supports_count:
        assert settings["number_of_organelles"] == 2
    else:
        assert "number_of_organelles" not in settings


def test_explicitly_lowered_count_preserves_supplied_hidden_slot_values(tmp_path):
    rows = [("number_of_organelles", 0), ("organelled_channel", 5),
            ("organelled_min_intensity", 12.5)]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["number_of_organelles"] == 0
    assert settings.get("organelled_channel") == 5
    assert settings.get("organelled_min_intensity") == 12.5
    assert set(report.applied) == {key for key, _value in rows}
    assert report.dropped == []


def test_unknown_slot_suffix_does_not_infer_or_expand_a_slot(tmp_path):
    unknown = "organellezz_setting_that_never_existed_317"
    assert unknown not in expected_types
    rows = [(unknown, 3), ("verbose", False)]

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, rows)))

    assert settings["number_of_organelles"] == 0
    assert unknown not in settings
    assert "organellezz_channel" not in settings
    assert report.applied == ["verbose"]
    assert report.dropped == [unknown]
    assert report.elsewhere == []


def test_unmentioned_slot_count_keeps_the_callers_existing_value(tmp_path):
    defaults = {"number_of_organelles": 4, "organelle_channel": None,
                "verbose": True}
    before = dict(defaults)

    settings, report = settings_from_pack(
        "mask", str(_write_pack(tmp_path, [("verbose", False)])),
        defaults=defaults)

    assert defaults == before
    assert settings["number_of_organelles"] == 4
    assert report.applied == ["verbose"]
    assert report.dropped == []


def test_missing_pack_preserves_the_existing_schema_without_inventing_values(tmp_path):
    defaults = {"number_of_organelles": 4, "organelle_channel": 2,
                "verbose": True}
    before = dict(defaults)

    settings, report = settings_from_pack("mask", str(tmp_path), defaults=defaults)

    assert settings == before
    assert settings is not defaults
    assert defaults == before
    assert report.source == ""
    assert report.applied == report.renamed == report.dropped == []
