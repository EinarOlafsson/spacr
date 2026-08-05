"""The power screen carries the two new simulator stages without losing them.

``PowerScreen.spec()`` rebuilds the DesignSpec from the form plus ``_held``, so
a field that is on neither silently reverts to its default. For an ordinary
held parameter that would be an annoyance; for these two it would mean a run
that was configured with sequencing error reloads without it and quietly
reports the wrong power. The round-trip is the contract, and this pins it.

Kept out of ``tests/qt/test_power_screen.py`` so the two files can be worked
on independently.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.power_simulate import (  # noqa: E402
    DEFAULT_MIN_CELLS_PER_WELL, DEFAULT_SEQUENCING_ERROR_RATE,
)
from spacr.qt.screens.power import _HELD_FIELDS, PowerScreen  # noqa: E402
from spacr.qt.widgets.power_design import (  # noqa: E402
    CAVEATS, DesignSpec, changes_the_number, simulator_kwargs,
)

pytestmark = pytest.mark.qt

NEW_FIELDS = ("sequencing_error_rate", "min_cells_per_well")


def test_every_spec_field_is_either_on_the_form_or_held():
    """The general rule, not just the two new fields. A DesignSpec field in
    neither place is a parameter that cannot survive a reload."""
    from dataclasses import fields

    on_the_form = {
        "n_genes", "n_grnas_per_gene", "score_per", "cells_per_well",
        "wells_per_plate", "n_plates", "constructs_per_well",
        "background_positive_rate", "effect_fold", "hit_rate",
        "reads_per_well", "n_replicates", "detection_auroc", "seed",
        "backend",
    }
    declared = {f.name for f in fields(DesignSpec)}
    unreachable = declared - on_the_form - set(_HELD_FIELDS)
    assert unreachable == set(), (
        f"these DesignSpec fields would be dropped by spec(): {unreachable}")


@pytest.mark.parametrize("field", NEW_FIELDS)
def test_the_new_stages_are_held_rather_than_dropped(field):
    assert field in _HELD_FIELDS


def test_a_spec_with_both_stages_on_round_trips_through_the_screen(qtbot):
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)

    spec = DesignSpec(
        sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
        min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL)
    screen.set_spec(spec)
    assert screen.spec() == spec
    assert screen.spec().sequencing_error_rate == \
        DEFAULT_SEQUENCING_ERROR_RATE
    assert screen.spec().min_cells_per_well == DEFAULT_MIN_CELLS_PER_WELL


def test_the_screen_still_opens_on_the_spaCRPower_baseline(qtbot):
    """Both default off. A screen whose baseline moved would make every power
    figure ever printed from it wrong."""
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.spec() == DesignSpec()
    assert screen.spec().sequencing_error_rate == 0.0
    assert screen.spec().min_cells_per_well == 0


def test_the_held_line_prints_the_values_that_were_accepted(qtbot):
    """'A parameter that only exists in a dataclass default is a parameter
    nobody knows they accepted' -- the screen's own words, applied to these."""
    screen = PowerScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.set_spec(DesignSpec(
        sequencing_error_rate=DEFAULT_SEQUENCING_ERROR_RATE,
        min_cells_per_well=DEFAULT_MIN_CELLS_PER_WELL))
    text = screen._held_note.text()
    assert f"sequencing_error_rate={DEFAULT_SEQUENCING_ERROR_RATE}" in text
    assert f"min_cells_per_well={DEFAULT_MIN_CELLS_PER_WELL}" in text


def test_both_stages_reach_the_simulator_by_its_own_parameter_names():
    kwargs = simulator_kwargs(DesignSpec(
        sequencing_error_rate=0.02, min_cells_per_well=30))
    assert kwargs["sequencing_error_rate"] == 0.02
    assert kwargs["min_cells_per_well"] == 30

    import inspect

    from spacr.power_simulate import simulate_screen

    accepted = set(inspect.signature(simulate_screen).parameters)
    assert set(kwargs).issubset(accepted), (
        "simulator_kwargs must spell every key the way the simulator does")


def test_the_caveats_are_in_the_group_that_changes_the_number():
    keys = {caveat.key for caveat in changes_the_number()}
    assert "sequencing_error_hides_untested_genes" in keys
    assert "thin_wells_count_the_same_as_full_ones" in keys

    by_key = {caveat.key: caveat for caveat in CAVEATS}
    for key in ("sequencing_error_hides_untested_genes",
                "thin_wells_count_the_same_as_full_ones"):
        detail = by_key[key].detail
        assert "set_spec(DesignSpec(" in detail, (
            f"{key} must say how to turn it on from this screen")
        assert "simulate_screen" in detail, (
            f"{key} must also name the headless API")
