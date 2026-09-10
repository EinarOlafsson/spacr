"""364: the flag and the number that already said it.

`gradient_accumulation` was a boolean beside `gradient_accumulation_steps`,
and ``steps = 1`` is the off state -- one batch per optimizer step is
ordinary training. Two keys for one decision is a pair that can disagree.

These tests are about the MIGRATION rather than the removal. Retiring the
flag on its own would silently turn accumulation on for everyone who had
written ``gradient_accumulation: false`` beside the default four steps.
"""
from __future__ import annotations

from spacr.settings import _fold_gradient_accumulation


def test_a_stored_false_collapses_the_step_count():
    """The whole reason the migration exists.

    Their training was NOT accumulating. Ignoring the flag and leaving four
    steps would start it, silently, on the next run.
    """
    settings = {"gradient_accumulation": False,
                "gradient_accumulation_steps": 4}
    _fold_gradient_accumulation(settings)
    assert settings["gradient_accumulation_steps"] == 1
    assert "gradient_accumulation" not in settings


def test_the_string_spellings_a_settings_file_carries_are_honoured():
    """A CSV or JSON settings file stores 'False', not False."""
    for spelling in ("False", "false", "0", "no", " FALSE "):
        settings = {"gradient_accumulation": spelling,
                    "gradient_accumulation_steps": 8}
        _fold_gradient_accumulation(settings)
        assert settings["gradient_accumulation_steps"] == 1, spelling


def test_a_stored_true_leaves_the_step_count_alone():
    """The steps already say how many; the flag added nothing."""
    settings = {"gradient_accumulation": True,
                "gradient_accumulation_steps": 8}
    _fold_gradient_accumulation(settings)
    assert settings["gradient_accumulation_steps"] == 8
    assert "gradient_accumulation" not in settings


def test_a_settings_file_without_the_flag_is_untouched():
    """Nothing to migrate is not an error, and must not invent a value."""
    settings = {"gradient_accumulation_steps": 4}
    _fold_gradient_accumulation(settings)
    assert settings == {"gradient_accumulation_steps": 4}


def test_the_defaults_no_longer_offer_the_flag():
    """One question, one control. The panel must not show both again."""
    from spacr.settings import set_default_train_test_model

    resolved = set_default_train_test_model({})
    assert "gradient_accumulation" not in resolved
    assert resolved["gradient_accumulation_steps"] == 4, (
        "the shipped default still accumulates over four batches, which is "
        "what the boolean's True default meant")


def test_the_retired_name_is_named_rather_than_ignored():
    """A settings file in the wild gets told, not silently overruled."""
    from spacr.validate import RETIRED_SETTINGS

    assert RETIRED_SETTINGS["gradient_accumulation"] == (
        "gradient_accumulation_steps")
