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


def test_spacr_mode_offers_all_three():
    question = _by_key()["spacr_mode"]
    offered = [value for value, _caption in question[4]]

    assert sorted(offered) == sorted(prefs.SPACR_MODES)
    assert len(offered) == 3


def test_spacr_mode_is_captioned_the_way_preferences_captions_it():
    """"extra performance" is what replace('_', ' ') gives; it is not the name."""
    question = _by_key()["spacr_mode"]
    captions = dict(question[4])

    assert captions["extra_performance"] == prefs.MODE_LABELS["extra_performance"]
    assert captions["balanced"] == prefs.MODE_LABELS["balanced"]


def test_every_mode_can_actually_be_stored():
    """An offered value the setter rejects is worse than one not offered."""
    before = prefs.get_spacr_mode()
    try:
        for value, _caption in _by_key()["spacr_mode"][4]:
            prefs.set_spacr_mode(value)
            assert prefs.get_spacr_mode() == value
    finally:
        prefs.set_spacr_mode(before)


#: Questions whose answers come from a named inventory in `preferences`,
#: and the attribute that holds it. A question here must offer all of it.
INVENTORIES = {
    "spacr_mode": "SPACR_MODES",
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
