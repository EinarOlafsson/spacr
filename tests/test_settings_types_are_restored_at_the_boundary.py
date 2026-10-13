"""Text-written numbers become their declared type before a run starts.

Every number typed into a GUI field arrives as text, and so does every value
read back from a settings CSV. That produced two failures from one cause:

* three ERROR lines per run telling the user to fix, by hand, values that were
  perfectly well-formed -- ``cell_diameter='60.0'`` and its two siblings;
* the run then dying inside Cellpose on ``diameter > 0`` with
  ``TypeError: '>' not supported between instances of 'str' and 'int'``.

`expected_types` is the contract those values claim to satisfy, so converting
them to it restores what they already say they are.
"""
from __future__ import annotations

import pytest

from spacr.validate import coerce_expected_types, validate_settings


def test_the_reported_case_stops_being_reported():
    raw = {"cell_diameter": "60.0", "nucleus_diameter": "30.0",
           "pathogen_diameter": "20.0"}
    coerced = coerce_expected_types(raw, "mask")
    assert coerced == {"cell_diameter": 60, "nucleus_diameter": 30,
                       "pathogen_diameter": 20}
    complaints = [p for p in validate_settings(dict(coerced), "mask")
                  if p.setting.endswith("diameter")]
    assert complaints == []


def test_the_input_is_not_modified():
    """The caller's dict is theirs. A settings mapping is passed around and
    written back to CSV, so mutating it in place would rewrite the user's
    file as a side effect of validating it."""
    raw = {"cell_diameter": "60.0"}
    coerce_expected_types(raw, "mask")
    assert raw == {"cell_diameter": "60.0"}


def test_a_fractional_value_for_an_int_key_is_left_alone():
    """Truncating would change the run without saying so. Left as it was, so
    the validator reports it."""
    out = coerce_expected_types({"cell_diameter": "60.5"}, "mask")
    assert out["cell_diameter"] == "60.5"
    assert any(p.setting == "cell_diameter"
               for p in validate_settings(dict(out), "mask"))


def test_a_key_that_legitimately_holds_text_is_untouched():
    """"30" is a NAME for a model, not a number."""
    out = coerce_expected_types(
        {"cell_model_name": "30", "custom_regex": "12"}, "mask")
    assert out["cell_model_name"] == "30"
    assert out["custom_regex"] == "12"


@pytest.mark.parametrize("written, expected", [
    ("True", True), ("true", True), ("yes", True), ("1", True),
    ("False", False), ("false", False), ("no", False), ("0", False),
])
def test_written_booleans_are_restored(written, expected):
    """Bool is checked BEFORE int, because bool is a subclass of int: the
    other order turns "1" into the integer 1 for a key that wanted True."""
    out = coerce_expected_types({"normalize": written}, "mask")
    assert out["normalize"] is expected


def test_an_unparseable_value_is_left_for_the_validator():
    out = coerce_expected_types({"cell_diameter": "thirty"}, "mask")
    assert out["cell_diameter"] == "thirty"
    assert any(p.setting == "cell_diameter"
               for p in validate_settings(dict(out), "mask"))


def test_blank_and_unknown_keys_are_left_alone():
    out = coerce_expected_types(
        {"cell_diameter": "", "not_a_real_setting": "7"}, "mask")
    assert out["cell_diameter"] == ""
    assert out["not_a_real_setting"] == "7"


def test_values_that_are_already_typed_are_returned_unchanged():
    raw = {"cell_diameter": 60, "normalize": True, "cell_cellprob_threshold": 0.5}
    assert coerce_expected_types(raw, "mask") == raw


def test_the_bridge_hands_the_pipeline_the_coerced_dict():
    """A source check. Coercing only the copy that gets VALIDATED would
    silence the warning and leave the crash exactly where it was -- which is
    the more dangerous half of this bug, not the lesser one."""
    from pathlib import Path

    import spacr.qt.bridge as bridge

    source = Path(bridge.__file__).read_text(encoding="utf-8")
    assert "settings = coerce_expected_types(settings, app_key)" in source
