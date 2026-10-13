"""Slot N's background switch is ``remove_background_organelle_N``.

Item 76, 2026-09-30: the maintainer named the per-slot switch
``remove_background_organelle_N`` -- ``remove_background_organelle`` for the
first slot, ``remove_background_organelle_2`` ... for the rest -- following
``remove_background_<object>``. From 2026-09-21 the switches were lettered
(``remove_background_organelleb``); a file written then is folded onto the
new name. A file without any switch keeps the old shared behaviour: the
slot's channel follows the generic ``remove_background``.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.organelle_types import (
    _background_switch_key,
    _background_switch_role,
    _legacy_background_switch_role,
    organelle_role,
)


def _factory(settings):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    return set_default_settings_preprocess_generate_masks(dict(settings))


def _two_slots(**extra):
    return {"organelle_channel": 3, "organelleb_channel": 4,
            "number_of_organelles": 2, **extra}


@pytest.mark.parametrize("number, key", [
    (1, "remove_background_organelle"),
    (2, "remove_background_organelle_2"),
    (7, "remove_background_organelle_7"),
    (27, "remove_background_organelle_27"),
    (702, "remove_background_organelle_702"),
])
def test_the_key_is_numbered_and_reads_back(number, key):
    role = organelle_role(number)
    assert _background_switch_key(role) == key
    assert _background_switch_role(key) == role


@pytest.mark.parametrize("key", [
    "remove_background_organelle_0", "remove_background_organelle_1",
    "remove_background_organelle_703", "remove_background_organelle_02",
    "remove_background_organelleb", "remove_background_cell",
    "remove_background", "organelle_2_background",
])
def test_other_keys_are_not_a_numbered_switch(key):
    assert _background_switch_role(key) is None


def test_the_lettered_spelling_is_recognised_as_legacy_only():
    assert _legacy_background_switch_role(
        "remove_background_organelleb") == "organelleb"
    assert _legacy_background_switch_role(
        "remove_background_organelle") is None
    assert _legacy_background_switch_role("remove_background_cell") is None


def test_every_slot_after_the_first_is_declared_and_the_letters_are_not():
    from spacr.settings import SLOT_BACKGROUND_SWITCHES, expected_types

    assert SLOT_BACKGROUND_SWITCHES[:3] == (
        "remove_background_organelle_2", "remove_background_organelle_3",
        "remove_background_organelle_4")
    assert len(SLOT_BACKGROUND_SWITCHES) == 701
    assert expected_types["remove_background_organelle_7"] is bool
    assert "remove_background_organelleb" not in expected_types


def test_the_tooltip_names_its_own_slot_and_floor():
    from spacr.settings import tooltips

    text = tooltips["remove_background_organelle_7"]
    assert "the organelle 7 channel" in text
    assert "organelleg_background" in text


def test_a_lettered_switch_in_an_old_file_is_folded_onto_the_new_name():
    from spacr.settings import surviving_setting_name

    assert surviving_setting_name("remove_background_organelleb") == (
        "remove_background_organelle_2",)
    settings = _factory(_two_slots(remove_background_organelleb=True))
    assert settings["remove_background_organelle_2"] is True
    assert "remove_background_organelleb" not in settings


def test_the_new_name_wins_over_the_old_one_in_the_same_file():
    settings = _factory(_two_slots(remove_background_organelleb=True,
                                   remove_background_organelle_2=False))
    assert settings["remove_background_organelle_2"] is False


def test_the_doctor_reports_the_old_name_as_renamed():
    from spacr.validate import _check_retired_keys

    problems = _check_retired_keys({"remove_background_organelleb": True})
    assert len(problems) == 1
    assert "remove_background_organelle_2" in problems[0].message


@pytest.mark.parametrize("generic", [False, True])
def test_without_a_switch_a_slot_follows_the_shared_remove_background(generic):
    settings = _factory(_two_slots(remove_background=generic))
    assert settings["remove_background_organelle_2"] is generic


def test_a_disabled_slot_gets_no_switch():
    settings = _factory(_two_slots())
    assert "remove_background_organelle_3" not in settings


def _ramp():
    ramp = np.arange(1, 4097, dtype=np.float32).reshape(64, 64)
    return np.repeat(ramp[None, :, :, None], 5, axis=-1)


def _normalise(settings):
    from spacr.io import _normalize_img_batch

    stack = _ramp()
    return stack, _normalize_img_batch(
        stack=stack.copy(), channels=range(5), save_dtype=np.float32,
        settings={"lower_percentile": 2, "organelle_channel": 3,
                  "organelleb_channel": 4, "organelle_background": 300,
                  "organelleb_background": 300, **settings})


@pytest.mark.parametrize("key", [
    "remove_background_organelle_2", "remove_background_organelleb"])
def test_preprocessing_clips_slot_two_by_either_spelling(key):
    stack, on = _normalise({key: True})
    _stack, off = _normalise({})
    below = stack[0, :, :, 4] < 300
    assert np.all(on[0][below, 4] == 0)
    assert np.any(off[0][below, 4] > 0)
    np.testing.assert_array_equal(on[..., 3], off[..., 3])


def test_the_per_object_table_files_the_switch_under_its_slot():
    from spacr.object_settings_table import (_settings_key, from_table,
                                             to_table)

    flat = {"remove_background_cell": True,
            "remove_background_organelle": False,
            "remove_background_organelle_2": True,
            "remove_background_organelle_7": False}
    table = to_table(flat)
    assert table["remove_background"] == {
        "cell": True, "organelle": False, "organelleb": True,
        "organelleg": False}
    assert from_table(table) == flat
    assert _settings_key("organelleb", "remove_background") == (
        "remove_background_organelle_2")
    assert _settings_key("organelle", "remove_background") == (
        "remove_background_organelle")


def test_the_per_object_table_reads_an_unfolded_lettered_switch():
    from spacr.object_settings_table import from_table, to_table

    table = to_table({"remove_background_organelleb": True})
    assert table["remove_background"] == {"organelleb": True}
    assert from_table(table) == {"remove_background_organelle_2": True}
