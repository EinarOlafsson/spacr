"""The shared training-basis vocabulary for Classify (CV) and Classify (ML).

The two modules did the same job in different words. Six setting names were
shared out of 78 and 37, and two of those six disagreed on their DEFAULT --
``annotation_column`` was ``'test'`` in one and ``None`` in the other. Three
more pairs were the same setting under different names.

The part that mattered most was invisible: Classify (ML) chose its training
basis by asking whether ``annotation_column`` was ``None``. Nothing in the
settings panel said so, so filling in an annotation column silently stopped
the module training on plate controls.

These tests defend the two properties that make the shared vocabulary safe:
an OLD settings CSV still does exactly what it did, and an ambiguous one is
refused rather than guessed at.
"""
from __future__ import annotations

import pytest

from spacr import training_basis as tb


# ---------------------------------------------------------------------------
# Backward compatibility -- the half that protects existing projects
# ---------------------------------------------------------------------------

def test_an_old_csv_with_no_dataset_mode_keeps_its_old_behaviour():
    """The implicit rule ML used, preserved exactly.

    Every settings CSV in a user's project predates `dataset_mode`. If this
    resolved differently the run would train on different labels and report
    success either way, which is the worst shape a regression can take.
    """
    assert tb.resolve_basis({}) == "metadata"
    assert tb.resolve_basis({"annotation_column": None}) == "metadata"
    assert tb.resolve_basis({"annotation_column": ""}) == "metadata"
    assert tb.resolve_basis({"annotation_column": "test"}) == "annotation"


def test_an_explicit_dataset_mode_beats_the_implicit_rule():
    """Someone who has said what they mean must not be second-guessed."""
    settings = {"dataset_mode": "metadata", "annotation_column": "test"}
    assert tb.resolve_basis(settings) == "metadata"


@pytest.mark.parametrize("basis", tb.TRAINING_BASES)
def test_every_basis_resolves_to_itself(basis):
    assert tb.resolve_basis({"dataset_mode": basis}) == basis
    assert tb.resolve_basis({"dataset_mode": basis.upper()}) == basis


def test_an_unknown_basis_is_refused_not_guessed():
    """Falling back would train on the wrong labels and report success."""
    with pytest.raises(tb.TrainingBasisError, match="not one of"):
        tb.resolve_basis({"dataset_mode": "annotations"})     # plural typo


# ---------------------------------------------------------------------------
# The renames
# ---------------------------------------------------------------------------

def test_retired_names_are_translated_not_dropped():
    out = tb.normalize_settings(
        {"model_type_ml": "xgboost", "test_size": 0.2,
         "cross_validation": True})
    assert out["model_type"] == "xgboost"
    assert out["test_split"] == 0.2
    assert out["cross_validation_enabled"] is True
    for retired in tb.SETTING_ALIASES:
        assert retired not in out


def test_the_current_name_wins_when_both_are_present():
    """A stale alias left in the same CSV must not override a deliberate
    value -- the direction of that precedence is the whole point."""
    out = tb.normalize_settings(
        {"model_type_ml": "xgboost", "model_type": "maxvit_t"})
    assert out["model_type"] == "maxvit_t"


def test_normalize_does_not_modify_its_input():
    original = {"model_type_ml": "xgboost"}
    tb.normalize_settings(original)
    assert original == {"model_type_ml": "xgboost"}


def test_normalize_pins_the_basis_so_consumers_read_one_key():
    assert tb.normalize_settings({})["dataset_mode"] == "metadata"
    assert tb.normalize_settings(
        {"annotation_column": "test"})["dataset_mode"] == "annotation"


# ---------------------------------------------------------------------------
# What each basis owns
# ---------------------------------------------------------------------------

def test_the_bases_do_not_claim_each_others_settings():
    """Overlap would mean a control greyed out under the basis that uses it."""
    seen = {}
    for basis in tb.TRAINING_BASES:
        for key in tb.settings_for_basis(basis):
            assert key not in seen, (
                f"{key!r} is claimed by both {seen[key]!r} and {basis!r}")
            seen[key] = basis


@pytest.mark.parametrize("basis", tb.TRAINING_BASES)
def test_inapplicable_is_exactly_the_other_bases(basis):
    mine = set(tb.settings_for_basis(basis))
    theirs = set(tb.inapplicable_settings(basis))
    assert mine.isdisjoint(theirs)
    everything = {k for b in tb.TRAINING_BASES for k in tb.settings_for_basis(b)}
    assert mine | theirs == everything


def test_the_measurement_basis_owns_the_rules_the_pipeline_reads():
    """`measurement_rules` is what spacr.io and spacr.ml both consume, so a
    panel that greyed it under the measurement basis would disable the only
    control that basis has."""
    assert "measurement_rules" in tb.settings_for_basis("measurement")
    assert "measurement_rules" in tb.inapplicable_settings("metadata")
    assert "measurement_rules" in tb.inapplicable_settings("annotation")


def test_every_basis_can_describe_itself():
    for basis in tb.TRAINING_BASES:
        assert tb.describe_basis(basis)
