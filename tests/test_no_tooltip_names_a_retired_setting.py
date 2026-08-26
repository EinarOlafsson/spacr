"""A tooltip may not describe a setting that no longer exists.

Instruction 107 replaced the two independent `score_data` / `count_data`
lists with an explicit paired table and retired `plates_score`,
`plates_count` and `plate_from_order` with them. Two tooltips were left
behind describing the retired rule: both still tell the user to pass "one
path per plate, position-aligned with plates_count / plates_score".

A STALE TOOLTIP IS WORSE THAN A MISSING ONE. A missing one says nothing; a
stale one names a control the panel does not offer, states a pairing rule
nothing applies any more, and hides `paired_data`, the control that replaced
it -- so the hover actively argues for the invisible positional contract the
paired table was built to remove.

THE GUARD IS THE RETIRED NAMES rather than those two strings, because the
failure is general: whenever a setting is removed, whatever described it is
the thing most likely to be left behind.

`OWED` IS A SHRINKING EXEMPTION, NOT AN ALLOWANCE. Both entries are real
defects, and the reason they are still here is recorded on the constant: the
English text cannot be corrected on its own, because reviewed human
translations in docs/i18n/reviewed/runtime/ are pinned to its exact bytes and
three tests refuse a reviewed record whose source has moved. Correcting the
prose and the reviewed records is one change, and it belongs to whoever holds
the catalog pipeline. `test_the_owed_list_holds_no_stale_entry` deletes the
excuse the moment the text is fixed.
"""

import pytest

from spacr.settings import (
    expected_types,
    get_perform_regression_default_settings,
    tooltips,
)

#: Retired by instruction 107 when regression inputs became a paired table.
#: Plate identity now comes from the pair -- own column, then partner, then
#: row order -- resolved in `ml.load_regression_input_pairs`.
RETIRED = ("plates_score", "plates_count", "plate_from_order")

#: Tooltips still describing the retired rule. ONLY EVER SHRINKS.
#: Blocked on the catalog rebuild, not on the wording: each of these strings
#: is the pinned source of a reviewed translation under
#: docs/i18n/reviewed/runtime/, so changing it alone turns three green tests
#: red until the reviewed records are regenerated with it.
OWED = {
    "count_data": "position-aligned with plates_count",
    "score_data": "position-aligned with plates_score",
}


@pytest.mark.parametrize("name", RETIRED)
def test_the_retired_settings_really_are_retired(name):
    """The premise of this file, asserted rather than assumed.

    If one of these came back, the guard below would be wrong to fire, so it
    states first that they are gone.
    """
    assert name not in expected_types
    assert name not in get_perform_regression_default_settings({})


@pytest.mark.parametrize("name", RETIRED)
def test_no_tooltip_names_a_retired_setting(name):
    """Nothing outside the owed list may name a key that does not exist."""
    offenders = sorted(key for key, text in tooltips.items()
                       if isinstance(text, str) and name in text
                       and key not in OWED)

    assert not offenders, (
        f"{name} was retired by instruction 107 but is still described in "
        f"the tooltips for {offenders}. Say what replaced it -- plate "
        "identity comes from the paired_data row -- rather than leaving the "
        "hover naming a control the panel does not offer.")


@pytest.mark.parametrize("key", sorted(OWED))
def test_the_owed_list_holds_no_stale_entry(key):
    """An exemption outlives its defect silently; this one cannot.

    The moment the prose is corrected, this fails and says to delete the
    entry -- so the list cannot quietly become a permanent allowance for
    text that is already right.
    """
    assert OWED[key] in tooltips[key], (
        f"{key}'s tooltip no longer says {OWED[key]!r}, so it is fixed. "
        f"Remove {key!r} from OWED; the guard above then covers it.")


def test_the_control_that_replaced_them_says_so():
    """`paired_data` has to be findable from the panel, not just from ml.py.

    It is what a user reaching for the retired keys actually wants, and the
    only place the two-axis pairing rule is stated to them.
    """
    text = tooltips["paired_data"]

    assert "score" in text and "count" in text
    assert "plateID" in text, (
        "paired_data's tooltip does not say where plate identity comes "
        "from, which is the whole reason the retired keys were retired.")
    assert "score_data" in text and "count_data" in text, (
        "paired_data's tooltip does not mention the legacy keys, so a user "
        "holding an old settings CSV cannot tell the two forms apart.")
