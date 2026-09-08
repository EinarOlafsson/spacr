"""A setup question that silently offers one answer is a question with no point.

Reported as "there is only one option for spaCR mode, ballanced, where
there should be 3".

The cause was a wrong attribute name behind a forgiving default::

    choices_of(getattr(prefs, "VALID_SPACR_MODES", ("balanced",)))

There is no ``prefs.VALID_SPACR_MODES``. The tuple is ``SPACR_MODES``, so
the lookup found nothing, took its fallback, and the screen offered one
mode as though that were all there was. It had happened once before with
``getattr(prefs, "VALID_LANGUAGES", ("en",))`` -- which offered English
alone on a screen whose first question is the language -- and the comment
recording that fix sits directly above the line that still had it.

So this file checks the property rather than the two known cases: every
question that comes from a named inventory must offer that whole inventory.
"""
from __future__ import annotations

import pytest

from spacr.qt import preferences as prefs
from spacr.qt.setup_screen import questions


def _by_key():
    return {q[0]: q for q in questions()}


def test_spacr_mode_offers_every_performance_level():
    """FIVE LEVELS, NOT THREE POSTURES, AND THE SCREEN IS RIGHT.

    This asserted `SPACR_MODES` -- the three-value resource posture -- and
    the screen deliberately stopped offering it. `setup_screen` sets the
    reason out at length: Laptop and Workstation were settable in
    Preferences and could not be chosen on the screen whose whole job is
    choosing this once, and `spacr_mode_for_level` folds five levels onto
    three postures, so writing through `set_spacr_mode` cannot express
    either end of the scale.

    Keeping the old assertion would have meant taking two answers away
    from the user to make a test pass. The inventory is the LEVELS.
    """
    question = _by_key()["spacr_mode"]
    offered = [value for value, _caption in question[4]]

    assert sorted(offered) == sorted(prefs.PERFORMANCE_LEVELS), (
        f"the setup screen offers {sorted(offered)}, not every level")


def test_spacr_mode_is_captioned_the_way_preferences_captions_it():
    """"extra performance" is what replace('_', ' ') gives; it is not the name."""
    question = _by_key()["spacr_mode"]
    captions = dict(question[4])

    assert captions["extra_performance"] == prefs.MODE_LABELS["extra_performance"]
    assert captions["balanced"] == prefs.MODE_LABELS["balanced"]


def test_every_offered_level_can_actually_be_stored():
    """An option that raises when chosen is worse than one that is absent.

    `set_spacr_mode` REJECTS two of the five -- "unknown spaCR mode
    'laptop'" -- which is why the screen writes through
    `set_performance_level`, the setter that accepts a level and updates
    the posture behind it. Driven through the setter the screen actually
    uses, so this cannot pass while the screen raises.
    """
    before = prefs.get_performance_level()
    setter = _by_key()["spacr_mode"][3]
    try:
        for value, _caption in _by_key()["spacr_mode"][4]:
            setter(value)
            assert prefs.get_performance_level() == value
    finally:
        prefs.set_performance_level(before)


INVENTORIES = {
    "spacr_mode": "PERFORMANCE_LEVELS",
    "colour_blind": "VALID_CB_MODES",
    "issue_prompt": "ISSUE_PROMPT_MODES",
}


@pytest.mark.parametrize("key,attribute", sorted(INVENTORIES.items()))
def test_a_question_offers_its_whole_inventory(key, attribute):
    inventory = getattr(prefs, attribute)
    question = _by_key().get(key)

    assert question is not None, f"the setup screen no longer asks {key!r}"
    offered = {value for value, _caption in question[4]}
    assert offered == set(inventory), (
        f"{key} offers {sorted(offered)} of {sorted(inventory)}")


def test_the_named_inventories_exist_under_those_names():
    """The whole defect was a name that did not exist, defaulted past."""
    for attribute in INVENTORIES.values():
        assert hasattr(prefs, attribute), (
            f"preferences has no {attribute}; a getattr with a fallback "
            f"would silently shorten the question that reads it")


def test_no_multiple_choice_question_is_down_to_one_answer():
    """The shape of the bug, whatever causes it next time."""
    single = [q[0] for q in questions()
              if q[4] is not None and len(q[4]) == 1]

    assert not single, (
        f"these questions offer exactly one answer: {single}. Either the "
        f"inventory behind them is being read by the wrong name, or they "
        f"should not be questions.")
