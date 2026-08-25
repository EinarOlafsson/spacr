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

from spacr.object import _validate_organelle_settings
from spacr.organelle_types import (LEGAL_METHODS, ORGANELLE_TYPES,
                                   RING_RESOLVABLE_PX, TYPE_ORDER,
                                   OrganelleType, apply_preset, known_types,
                                   preset_for, resolve_type)


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


def test_a_misspelled_type_stops_apply_preset_too():
    """The whole-settings entry point raises rather than filling defaults."""
    with pytest.raises(ValueError):
        apply_preset({"organelle_type": "lysosome"})


# ---------------------------------------------------------------------------
# the method has to be legal for the morphology SIZE chose
# ---------------------------------------------------------------------------

def test_a_method_illegal_at_this_size_is_replaced_not_shipped(monkeypatch):
    """A split preset whose method suits only the small half is rewritten.

    'ridge' is legal for a network and illegal for an irregular blob. When
    the diameter tips such a preset into its large half the recommendation
    becomes the first legal method for that morphology rather than the one
    the table names, because a recommendation the run-time validator rejects
    would fail the run before an image is read.
    """
    tipping = OrganelleType(
        label="Tipping",
        members=("a thing that swells",),
        method="ridge",
        size_split=("network", "irregular"),
        params={"organelle_fill_holes": 8},
    )
    monkeypatch.setitem(ORGANELLE_TYPES, "tipping", tipping)

    small = preset_for("tipping", RING_RESOLVABLE_PX - 1)
    assert small["organelle_morphology"] == "network"
    assert small["organelle_method"] == "ridge"

    large = preset_for("tipping", RING_RESOLVABLE_PX + 1)
    assert large["organelle_morphology"] == "irregular"
    assert large["organelle_method"] != "ridge"
    assert large["organelle_method"] == LEGAL_METHODS["irregular"][0]
    # The claim that matters: the validator accepts what came out.
    assert _validate_organelle_settings("irregular",
                                        large["organelle_method"]) is None
    # The extra recommended settings survive the rewrite.
    assert large["organelle_fill_holes"] == 8


def test_a_preset_with_no_method_at_all_still_recommends_a_legal_one(
        monkeypatch):
    """An empty `method` takes the first legal one rather than an empty one."""
    silent = OrganelleType(label="Silent", members=(), method="",
                           morphology="ring")
    monkeypatch.setitem(ORGANELLE_TYPES, "silent", silent)

    out = preset_for("silent")
    assert out["organelle_method"] == LEGAL_METHODS["ring"][0]
    assert _validate_organelle_settings("ring", out["organelle_method"]) is None


# ---------------------------------------------------------------------------
# the explanation
# ---------------------------------------------------------------------------

def test_the_explanation_separates_what_it_set_from_what_you_set(capsys):
    """`apply_preset(explain=True)` prints filled values and retained ones.

    A preset that quietly kept the user's method and a preset that quietly
    overrode it look identical in the resulting settings dict; only the
    explanation tells them apart.
    """
    out = apply_preset({"organelle_type": "vesicular",
                        "organelle_diameter": 40,
                        "organelle_method": "cellpose"},
                       explain=True)
    printed = capsys.readouterr().out

    assert out["organelle_method"] == "cellpose"
    assert out["organelle_morphology"] == "ring"

    assert "Vesicular" in printed
    # the members, so the user can check the name means what they think
    assert "vacuoles" in printed
    # the size rule, with the threshold and their own diameter in it
    assert str(RING_RESOLVABLE_PX) in printed
    assert "40" in printed
    # what it filled, and what it left alone
    assert "set    organelle_morphology = 'ring'" in printed
    assert "KEPT   organelle_method = 'cellpose'" in printed
    # and the caveat, rather than hiding it
    assert "LYSOSOMES" in printed


def test_a_preset_that_decides_nothing_prints_nothing(capsys):
    """'custom' recommends nothing, so there is nothing to explain."""
    apply_preset({"organelle_type": "custom"}, explain=True)
    assert capsys.readouterr().out == ""


def test_a_fixed_morphology_preset_explains_without_a_size_rule(capsys):
    """A preset with one morphology does not print a size rule it has not."""
    apply_preset({"organelle_type": "filamentous"}, explain=True)
    printed = capsys.readouterr().out

    assert "Filamentous" in printed
    assert "microtubules" in printed
    assert "size decides" not in printed
    assert "set    organelle_morphology = 'network'" in printed
    assert "KEPT" not in printed


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
