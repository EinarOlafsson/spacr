"""The preset table answers about size, about typos, and about itself.

Three questions the picker asks that the shipped presets alone do not
answer: what a split preset recommends when SIZE rather than the name
decides the detector, what happens when the name is not one it knows, and
what the explanation actually says once a preset has both filled some
settings and left others alone.

The fourth question has no shipped answer at all. Every preset in the table
ships a method that is legal for both halves of its split, so the fallback
that rewrites an illegal method is never taken by anything the repository
ships. It is exercised here against a preset built for the purpose, because
the guarantee it makes -- that a recommendation is never rejected by the
run-time validator -- is a guarantee about presets that do not exist yet.
"""

import pytest

from spacr.organelle_types import (ORGANELLE_TYPES, RING_RESOLVABLE_PX,
                                   TYPE_ORDER, known_types, resolve_type)


# ---------------------------------------------------------------------------
# size decides
# ---------------------------------------------------------------------------

def test_a_split_preset_reads_the_diameter_not_the_name():
    """`morphology_for` returns the small half below the ring threshold.

    Vesicular is the row that proves the mapping is not one-to-one: the same
    biological family is a dot at 200 nm and a ring at 2 um.
    """
    vesicular = ORGANELLE_TYPES["vesicular"]
    small, large = vesicular.size_split

    assert vesicular.morphology_for(RING_RESOLVABLE_PX - 1) == small
    assert vesicular.morphology_for(RING_RESOLVABLE_PX) == large
    assert vesicular.morphology_for(RING_RESOLVABLE_PX + 100) == large
    assert small != large


def test_an_unstated_diameter_gets_the_smaller_detector():
    """With no diameter, a split preset recommends the small half.

    Not the large one: a ring detector run on dots finds holes that are not
    there, while a spot detector run on rings finds the ring's rim, which is
    wrong but recoverable by eye.
    """
    for name in TYPE_ORDER:
        preset = ORGANELLE_TYPES[name]
        if preset.size_split is None:
            continue
        assert preset.morphology_for(None) == preset.size_split[0], name


# ---------------------------------------------------------------------------
# the picker's own list
# ---------------------------------------------------------------------------

def test_the_picker_offers_every_preset_the_table_holds():
    """`known_types` is the display order, and it is the whole table."""
    assert known_types() == TYPE_ORDER
    assert set(known_types()) == set(ORGANELLE_TYPES)
    assert known_types()[0] == "custom"


# ---------------------------------------------------------------------------
# a typo is not silently segmented
# ---------------------------------------------------------------------------

def test_a_misspelled_type_names_the_ones_that_exist():
    """An unknown `organelle_type` raises and lists the known names.

    Falling back to 'custom' would run the pipeline with different settings
    from the ones the user asked for and say nothing about it.
    """
    with pytest.raises(ValueError) as raised:
        resolve_type("mitochondria")
    message = str(raised.value)
    assert "mitochondria" in message
    for name in TYPE_ORDER:
        assert name in message


# ---------------------------------------------------------------------------
# which settings stay in front of the biologist
# ---------------------------------------------------------------------------

def test_the_plain_category_keeps_the_settings_a_biologist_recognises():
    """`is_basic` is true for the short list and false for the machinery."""
    from spacr.organelle_types import BASIC_SETTINGS, is_basic

    for setting in BASIC_SETTINGS:
        assert is_basic(setting), setting
    for setting in ("organelle_ridge_sigmas", "organelle_log_min_sigma",
                    "organelle_hysteresis_low", "organelle_morphology",
                    "organelle_method"):
        assert not is_basic(setting), setting
