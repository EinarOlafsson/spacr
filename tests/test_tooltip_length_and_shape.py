"""Ceilings on tooltip quality, so the table cannot drift back.

The 2026-08 sweep took the settings tooltips from 37 over 600 characters to
none, and from 13 that only restated their own name to one. Those numbers
only stay where they are if something checks: the table is edited by
whoever adds a setting, and "a bit longer each time" is exactly how it got
there in the first place.

These are CEILINGS, not targets. Every one may be lowered as the remaining
work lands; none may be raised without a reason in the commit message.
"""

import pytest

from spacr.qt.screens.settings_model import get_tooltips
from spacr.settings import descriptions

#: A hover is not a manual page. Measured: the median tooltip is 341
#: characters and reads comfortably; the ones that were 800+ were archives
#: of everything ever learned about the setting.
MAX_TOOLTIP_CHARS = 600

#: Below this a tooltip restates the setting's own name and returns nothing
#: for the hover. `pc` was "(str) - Positive control identifier."
MIN_TOOLTIP_CHARS = 80

#: Settings still awaiting a stated default. Only ever revise DOWN.
MAX_WITHOUT_DEFAULT = 35


@pytest.fixture(scope="module")
def tips():
    """Setting tooltips only -- module blurbs are a different artefact and
    are legitimately longer, since they describe a whole module."""
    return {k: v for k, v in get_tooltips().items()
            if isinstance(v, str) and k not in descriptions}


def test_no_tooltip_is_an_archive(tips):
    """Length is the symptom; the cause is everything ever learned about a
    setting appended and never edited down."""
    too_long = {k: len(v) for k, v in tips.items()
                if len(v) > MAX_TOOLTIP_CHARS}
    assert not too_long, (
        f"tooltips over {MAX_TOOLTIP_CHARS} chars: {too_long}. Lead with the "
        "consequence, then the action, then the edge case.")


def test_no_tooltip_merely_restates_its_own_name(tips):
    """One known exception remains: organelle_chann_dim at 75 chars, which
    is genuinely complete and simply short."""
    too_short = {k: len(v) for k, v in tips.items()
                 if 0 < len(v) < MIN_TOOLTIP_CHARS}
    assert len(too_short) <= 1, f"tooltips under {MIN_TOOLTIP_CHARS}: {too_short}"


def test_no_tooltip_is_empty(tips):
    assert not [k for k, v in tips.items() if not v.strip()]


def test_every_tooltip_opens_with_its_type(tips):
    """The table reads as one voice: '(bool) - ...'. An entry that starts
    differently is one someone wrote to a different standard."""
    missing = [k for k, v in tips.items() if not v.lstrip().startswith("(")]
    assert not missing, f"tooltips with no leading (type): {missing}"


def test_the_missing_defaults_count_only_goes_down(tips):
    """A number a user cannot see is a number they have to guess. This is a
    ceiling on the remaining work, not an endorsement of it."""
    missing = [k for k, v in tips.items() if "efault" not in v]
    assert len(missing) <= MAX_WITHOUT_DEFAULT, (
        f"{len(missing)} tooltips state no default, above the "
        f"{MAX_WITHOUT_DEFAULT} ceiling. New settings must state theirs.")


def test_the_table_is_not_shrinking(tips):
    """A guard against 'fixing' these tests by deleting tooltips."""
    assert len(tips) > 600
