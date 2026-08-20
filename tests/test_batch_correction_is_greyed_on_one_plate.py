"""Instruction 135's last open line.

    "STILL OPEN: greying the control out when the score table has one plate.
     The dependency rules see only the settings dict, and how many plates a
     run has is a fact about the DATA."

The mechanism was already there — `context['plate_count']`, filled by the
panel from the inputs the user dropped in, and already used by
`guide_permutation_block`. Correcting BETWEEN batches when there is one batch
removes nothing; the run already says so, but AFTER the run is too late.

Two properties matter more than the greying itself. UNKNOWN GREYS NOTHING: a
control disabled because a file was too large to scan cheaply is
indistinguishable, to the person looking at it, from one disabled on purpose.
And the rule COMBINES rather than replaces: a setting is applicable only if
every rule that mentions it says so, and assigning over an existing rule
silently drops it.
"""
from __future__ import annotations

import pytest

from spacr.settings import get_setting_dependencies

#: Every batch control, with the correction method that READS it. Two of
#: them -- the control anchors -- are read only by `control_center`, and
#: asking about them under `combat` measures the rule that was already there
#: rather than the one added for the plate count.
BATCH_KEYS = {
    "batch_correction": "combat",
    "batch_column": "combat",
    "batch_control_column": "control_center",
    "batch_control_values": "control_center",
    "batch_covariate_column": "combat",
    "batch_combat_mean_only": "combat",
    "batch_min_samples": "combat",
    "batch_missing_control": "combat",
}


@pytest.fixture(scope="module")
def rules():
    return get_setting_dependencies()


@pytest.mark.parametrize("key,method", sorted(BATCH_KEYS.items()))
def test_one_plate_greys_every_batch_control(rules, key, method):
    assert rules[key]["predicate"]({"batch_correction": method},
                                   {"plate_count": 1}) is False


@pytest.mark.parametrize("key,method", sorted(BATCH_KEYS.items()))
def test_the_reason_says_it_would_remove_nothing(rules, key, method):
    reason = rules[key]["reason"]({"batch_correction": method},
                                  {"plate_count": 1})
    assert "one plate" in reason
    # The ACTIONABLE half: what the alternative gives you.
    assert "identical result" in reason
    assert "kept and saved" in reason


@pytest.mark.parametrize("key,method", sorted(BATCH_KEYS.items()))
def test_more_than_one_plate_leaves_them_alone(rules, key, method):
    assert rules[key]["predicate"]({"batch_correction": method},
                                   {"plate_count": 4}) is True


@pytest.mark.parametrize("key,method", sorted(BATCH_KEYS.items()))
def test_an_unknown_plate_count_greys_nothing(rules, key, method):
    """None is "nothing loaded" or "too large to scan" -- not "one plate"."""
    for context in ({"plate_count": None}, {}):
        assert rules[key]["predicate"]({"batch_correction": method},
                                       context) is True


def test_the_new_rule_did_not_replace_the_one_that_was_there(rules):
    """A setting is applicable only if EVERY rule that mentions it says so."""
    rule = rules["batch_column"]

    # Four plates, but correction switched off: still inapplicable, and for
    # ITS OWN reason rather than the plate one.
    assert rule["predicate"]({"batch_correction": "none"},
                             {"plate_count": 4}) is False
    reason = rule["reason"]({"batch_correction": "none"}, {"plate_count": 4})
    assert "batch_correction is enabled" in reason
    assert "one plate" not in reason


def test_the_rule_watches_the_inputs_that_decide_it(rules):
    for key in BATCH_KEYS:
        sources = set(rules[key].get("sources", ()))
        assert sources & {"paired_data", "score_data", "count_data"}, (
            f"{key} is not re-evaluated when the inputs change")
