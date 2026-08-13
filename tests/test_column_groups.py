"""Choosing reduction columns by name, in groups. Instruction 49.

    "they should be able to chhose categories based on name lik cell, cor
     channel_1 or intensity measurements ... or morphology measurements"

The three kinds exist because a measurement table names a column three ways
at once: `cell_channel_1_mean_intensity` is a CELL measurement, a CHANNEL 1
measurement and an INTENSITY measurement, and which the user means depends on
the question they are asking.
"""
from __future__ import annotations

import pytest

from spacr.column_groups import (
    GROUP_KINDS,
    NON_FEATURE_FAMILIES,
    classify,
    columns_in,
    group_names,
    resolve,
    summarise,
)

COLUMNS = [
    "plateID", "rowID", "columnID", "object_label",
    "cell_area", "cell_perimeter",
    "cell_channel_1_mean_intensity", "cell_channel_2_mean_intensity",
    "nucleus_area", "nucleus_channel_0_mean_intensity",
    "pathogen_channel_1_mean_intensity",
]


# ---------------------------------------------------------------------------
# The three ways of naming one set
# ---------------------------------------------------------------------------

def test_the_kinds_are_object_channel_and_family():
    assert GROUP_KINDS == ("object", "channel", "family")


def test_one_column_belongs_to_one_group_of_each_kind():
    """The point of having three kinds rather than one taxonomy."""
    grouped = classify(COLUMNS)
    column = "cell_channel_1_mean_intensity"
    assert column in grouped["object"]["cell"]
    assert column in grouped["channel"]["channel_1"]
    assert column in grouped["family"]["intensity"]


def test_choosing_cell_selects_every_cell_measurement():
    chosen = resolve(COLUMNS, {"object": ["cell"]})
    assert "cell_area" in chosen
    assert "cell_channel_1_mean_intensity" in chosen
    assert "nucleus_area" not in chosen


def test_choosing_channel_1_selects_across_objects():
    """A channel is not owned by one object type."""
    chosen = resolve(COLUMNS, {"channel": ["channel_1"]})
    assert "cell_channel_1_mean_intensity" in chosen
    assert "pathogen_channel_1_mean_intensity" in chosen
    assert "cell_channel_2_mean_intensity" not in chosen


def test_choosing_a_family_selects_the_kind_of_quantity():
    chosen = resolve(COLUMNS, {"family": ["morphology"]})
    assert "cell_area" in chosen and "nucleus_area" in chosen
    assert "cell_channel_1_mean_intensity" not in chosen


def test_the_families_come_from_the_feature_dictionary():
    """Not a second taxonomy: it would disagree in the corners nobody checks."""
    from spacr.feature_dict import FEATURE_FAMILIES

    for family in group_names(COLUMNS)["family"]:
        assert family in FEATURE_FAMILIES


# ---------------------------------------------------------------------------
# Identifiers are not features
# ---------------------------------------------------------------------------

def test_identifiers_are_not_offered_as_a_group():
    """Feeding a plate id to UMAP embeds the plate -- the batch effect, not
    the biology."""
    names = group_names(COLUMNS)
    for family in NON_FEATURE_FAMILIES:
        assert family not in names["family"]


def test_an_identifier_is_never_selected_by_a_group():
    every_group = {kind: group_names(COLUMNS)[kind] for kind in GROUP_KINDS}
    chosen = resolve(COLUMNS, every_group)
    assert "plateID" not in chosen
    assert "object_label" not in chosen


def test_an_identifier_can_still_be_chosen_by_hand():
    """Not offered is not the same as forbidden."""
    assert resolve(COLUMNS, None, explicit=["plateID"]) == ["plateID"]


# ---------------------------------------------------------------------------
# Combining the two mechanisms
# ---------------------------------------------------------------------------

def test_groups_and_individual_columns_add_up():
    """"individual columns ... and ... categories" -- both, not either."""
    chosen = resolve(COLUMNS, {"family": ["intensity"]},
                     explicit=["cell_area"])
    assert "cell_area" in chosen
    assert "cell_channel_1_mean_intensity" in chosen


def test_a_column_selected_twice_appears_once():
    chosen = resolve(COLUMNS, {"object": ["cell"], "family": ["morphology"]})
    assert chosen.count("cell_area") == 1


def test_the_order_follows_the_table_not_the_clicks():
    """A UMAP whose axes depend on click order is not reproducible."""
    first = resolve(COLUMNS, {"object": ["cell"], "channel": ["channel_1"]})
    second = resolve(COLUMNS, {"channel": ["channel_1"], "object": ["cell"]})
    assert first == second
    assert first == [c for c in COLUMNS if c in set(first)]


# ---------------------------------------------------------------------------
# Refusals and edges
# ---------------------------------------------------------------------------

def test_an_unknown_kind_is_refused_by_name():
    """A typo would otherwise select nothing and read as "no such columns"."""
    with pytest.raises(KeyError, match="family"):
        columns_in(COLUMNS, "familly", "intensity")
    with pytest.raises(KeyError):
        resolve(COLUMNS, {"objects": ["cell"]})


def test_an_unknown_group_name_selects_nothing_rather_than_raising():
    """A saved selection naming a channel this table lacks still opens."""
    assert resolve(COLUMNS, {"channel": ["channel_9"]}) == []


def test_channels_sort_numerically():
    """channel_2 before channel_10, which a string sort gets wrong."""
    columns = [f"cell_channel_{i}_mean_intensity" for i in (0, 2, 10)]
    assert group_names(columns)["channel"] == [
        "channel_0", "channel_2", "channel_10"]


def test_an_empty_table_groups_into_nothing():
    assert classify([]) == {kind: {} for kind in GROUP_KINDS}
    assert resolve([], {"object": ["cell"]}) == []


def test_a_column_the_dictionary_cannot_classify_is_not_lost():
    """`unknown` is a statement about the dictionary, not about the column."""
    columns = COLUMNS + ["some_custom_feature_nobody_documented"]
    grouped = classify(columns)
    assert "unknown" in grouped["family"]


def test_the_summary_says_how_many_of_how_many():
    """400 columns and 4 look identical in a dialog until something says."""
    text = summarise(COLUMNS, {"family": ["intensity"]})
    assert "4 of 11" in text or "of 11" in text


def test_the_summary_says_so_when_nothing_is_selected():
    assert "no columns" in summarise(COLUMNS, {})


def test_the_module_needs_no_display():
    import subprocess
    import sys

    code = ("import sys, spacr.column_groups; "
            "assert not [m for m in sys.modules if m.startswith('PySide6')]")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
