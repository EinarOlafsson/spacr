"""Unrepresentable organelle counts retain the documented existing-slot fallback."""
import json

import pytest

from spacr import organelle_types as organelles
from spacr import settings as defaults


@pytest.mark.parametrize("raw", ["inf", "-inf", "1e309", "-1e309"])
@pytest.mark.parametrize("boundary", ["direct", "json-factory"])
def test_nonfinite_count_uses_existing_slots_without_losing_answers(tmp_path, raw, boundary):
    settings = {"number_of_organelles": raw, "organelle_channel": 2,
                "organelleb_channel": 3, "organelleb_diameter": 17,
                "experiment": "preserved experiment"}
    original = dict(settings)
    if boundary == "json-factory":
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(settings), encoding="utf-8")
        encoded = path.read_bytes()
        settings = defaults.set_default_settings_preprocess_generate_masks(
            json.loads(path.read_text(encoding="utf-8")))
        assert path.read_bytes() == encoded
    assert organelles.organelle_count(settings) == 2
    assert organelles.active_organelle_roles(settings) == ("organelle", "organelleb")
    for key, value in original.items():
        assert settings[key] == value


@pytest.mark.parametrize("raw", [float("inf"), float("-inf")])
def test_typed_nonfinite_count_preserves_existing_slot_request(raw):
    settings = {"number_of_organelles": raw, "organelleb_channel": 3}
    assert organelles.organelle_count(settings) == 2
    assert settings == {"number_of_organelles": raw, "organelleb_channel": 3}


@pytest.mark.parametrize("raw,expected", [
    (None, 2), ("", 2), ("seven", 2), (float("nan"), 2),
    (2.5, 2), (organelles.MAX_ORGANELLES + 1, organelles.MAX_ORGANELLES), (-1, 0),
])
def test_existing_parse_fallback_fractional_and_clamp_semantics_are_retained(raw, expected):
    assert organelles.organelle_count({
        "number_of_organelles": raw, "organelleb_channel": 3}) == expected
