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


@pytest.mark.parametrize("key, value", [
    # Truncating would change the run without saying so. Left as it was, so
    # the validator reports it.
    ("batch_size", "60.5"),
    ("cell_diameter", "thirty"),
], ids=["fractional-value-for-an-int-key", "unparseable-value"])
def test_a_value_that_cannot_be_restored_is_left_for_the_validator(key, value):
    out = coerce_expected_types({key: value}, "mask")
    assert out[key] == value
    assert any(p.setting == key
               for p in validate_settings(dict(out), "mask"))


def test_a_fractional_diameter_is_a_diameter():
    """Item 696: an object diameter is pixels, a fraction included, and 0 is
    stored as "auto"."""
    out = coerce_expected_types({"cell_diameter": "60.5",
                                 "nucleus_diameter": "0"}, "mask")
    assert out["cell_diameter"] == 60.5
    assert out["nucleus_diameter"] == "auto"


@pytest.mark.parametrize("given", [
    # "30" is a NAME for a model, not a number.
    {"cell_model_name": "30", "custom_regex": "12"},
    {"batch_size": "", "not_a_real_setting": "7"},
], ids=["keys-that-legitimately-hold-text", "blank-and-unknown-keys"])
def test_values_that_must_not_be_coerced_are_left_alone(given):
    out = coerce_expected_types(dict(given), "mask")
    for key, value in given.items():
        assert out[key] == value


@pytest.mark.parametrize("written, expected", [
    ("True", True), ("true", True), ("yes", True), ("1", True),
    ("False", False), ("false", False), ("no", False), ("0", False),
])
def test_written_booleans_are_restored(written, expected):
    """Bool is checked BEFORE int, because bool is a subclass of int: the
    other order turns "1" into the integer 1 for a key that wanted True."""
    out = coerce_expected_types({"normalize": written}, "mask")
    assert out["normalize"] is expected


def test_a_blank_diameter_is_the_magnification_default():
    """Item 696: blank is stored as None, which the run reads as the
    magnification default."""
    out = coerce_expected_types({"cell_diameter": ""}, "mask")
    assert out["cell_diameter"] is None


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


@pytest.mark.parametrize('key, written', [('normalize', 'maybe'), ('png_size', '224')])
def test_nonconvertible_boolean_and_list_text_is_preserved_for_validation(key, written):
    supplied = {key: written}

    converted = coerce_expected_types(supplied, 'mask')
    problems = validate_settings(converted, 'mask')

    assert converted == supplied == {key: written}
    assert converted is not supplied
    assert any(problem.setting == key and problem.is_error for problem in problems)


def test_dry_plan_reports_explicit_organelle_diameter_without_writing(tmp_path):
    from spacr.validate import describe_plan

    supplied = {'src': str(tmp_path), 'organelle_channel': 2,
                'organelle_diameter': 13, 'organelle_method': 'otsu'}
    original = dict(supplied)

    plan = describe_plan(supplied, 'mask')

    assert 'organelle: channel 2, diameter 13 px, method otsu' in plan
    assert supplied == original
    assert list(tmp_path.iterdir()) == []
