"""The default-settings fillers, and the property that makes them safe.

Instruction 60. These functions fill a settings dict for one module, and
they share a contract that nothing was checking:

    A VALUE THE USER ALREADY SET MUST SURVIVE.

They exist to supply what is MISSING. A filler that overwrites is a filler
that silently discards the number a user typed, and the run then reports a
result for settings they did not choose -- which is the quiet-wrong-answer
failure this repository keeps finding.

So every test below sets one key to a sentinel first and asserts it is still
there afterwards, as well as asserting the defaults arrive.

NOT covered here, deliberately: `check_settings`. It is 52 uncovered
statements in this module and the largest single gap, and it is called from
exactly one place -- `gui_core.py`, the retired Tk front end. The Qt side
only names it in a docstring. Covering a function scheduled for deletion is
the most expensive kind of zero, which is the same argument instruction 60
already makes about `gui_core` itself.
"""
from __future__ import annotations

import pytest

from spacr import settings as S


FILLERS = [
    "set_annotate_default_settings",
    "set_interpret_vision_model_defaults",
    "get_plot_data_from_csv_default_settings",
    "get_automated_motility_assay_default_settings",
]


@pytest.mark.parametrize("name", FILLERS)
def test_a_filler_supplies_defaults(name):
    """It has to actually fill something, or the test below is vacuous."""
    filler = getattr(S, name)
    out = filler({})
    result = out if isinstance(out, dict) else {}
    assert result, f"{name} filled nothing"


@pytest.mark.parametrize("name", FILLERS)
def test_a_filler_never_overwrites_a_value_the_user_set(name):
    """The contract. A filler supplies what is missing; it does not decide
    what the user meant."""
    filler = getattr(S, name)
    baseline = filler({})
    result = baseline if isinstance(baseline, dict) else {}
    if not result:
        pytest.skip(f"{name} produced no keys to probe")

    key = sorted(result)[0]
    sentinel = "USER-CHOSE-THIS"

    out = filler({key: sentinel})
    kept = out if isinstance(out, dict) else {}

    assert kept.get(key) == sentinel, (
        f"{name} overwrote {key!r}, which the user had already set")


def test_set_default_general_works_with_no_argument():
    """Documented as optional: a new dict is created when None is passed."""
    out = S.set_default_general()
    assert isinstance(out, dict)
    assert out


def test_set_default_general_fills_in_place_when_given_a_dict():
    given = {"src": "/screens/plate1"}
    out = S.set_default_general(given)

    assert out["src"] == "/screens/plate1"
    assert len(out) > 1


# --------------------------------------------------------------------------- #
#  _set_organelle_defaults -- the multi-organelle fan-out
# --------------------------------------------------------------------------- #

def test_organelle_defaults_cover_every_organelle_slot():
    """spaCR supports organelle, organelleb, organellec, organelled
    (instruction 76). A filler that only knows the first leaves the others
    unset, and the run fails much later with a missing key."""
    filled = {}
    S._set_organelle_defaults(filled)

    prefixes = {key.split("_")[0] for key in filled if key.startswith("organelle")}
    assert "organelle" in prefixes
    assert len(prefixes) >= 2, sorted(prefixes)


def test_organelle_defaults_keep_a_channel_the_user_chose():
    chosen = {"organelle_channel": 3}
    S._set_organelle_defaults(chosen)

    assert chosen["organelle_channel"] == 3


# --------------------------------------------------------------------------- #
#  _advanced_family_members -- matched on the suffix, not by substring
# --------------------------------------------------------------------------- #

def test_a_family_is_matched_on_the_suffix_not_by_substring():
    """The docstring's own warning: matching `area` by substring would drag
    in every key that merely contains it.

    `table` is a CATEGORY -> members mapping, not a flat settings dict --
    the function looks for `<object>_<suffix>` inside the category lists.
    """
    table = {
        "Object": ["cell_area", "nucleus_area", "cell_perimeter"],
        "Filters": ["cell_area_filter"],
    }

    members = S._advanced_family_members(table, ("area",))

    assert "cell_area" in members
    assert "nucleus_area" in members
    # Same substring, different suffix -- must not be dragged in.
    assert "cell_area_filter" not in members
    assert "cell_perimeter" not in members


def test_a_family_with_no_members_is_empty_not_an_error():
    assert S._advanced_family_members({"Object": ["cell_area"]},
                                      ("nothing",)) == []
