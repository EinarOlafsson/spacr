"""An empty or unnameable selection is reported, never drawn as everything.

The three scopes in :mod:`spacr.well_scope` decide which objects a cell-table
plot shows. Every path that cannot produce the requested population has to
say so in words the plot can print, because a plot silently falling back to
the whole table looks exactly like a selection the user made.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.well_scope import CHOSEN, MATE, MATE_COLUMN, describe, select, wells_of


def _frame():
    return pd.DataFrame({
        "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r1_c2", "p1_r1_c2"],
        "grna": ["g1", "g2", "g3", "g4"],
        "value": [1.0, 2.0, 3.0, 4.0],
    })


# ---------------------------------------------------------------------------
# Nothing to derive wells from
# ---------------------------------------------------------------------------

def test_no_guides_selected_names_no_wells():
    """An empty selection has no wells; it does not mean every well."""
    assert wells_of(_frame(), []) == []


def test_a_table_without_a_guide_column_names_no_wells():
    """Wells come from the guides, so a table with no guides names none."""
    frame = _frame().drop(columns=["grna"])

    assert wells_of(frame, ["g1"]) == []


def test_a_table_without_a_well_column_names_no_wells():
    """The other half of the same requirement."""
    frame = _frame().drop(columns=["prc"])

    assert wells_of(frame, ["g1"]) == []


def test_wells_are_unique_in_first_occurrence_order():
    """Two selected guides in one well must not duplicate that well."""
    assert wells_of(_frame(), ["g1", "g2", "g3"]) == [
        "p1_r1_c1",
        "p1_r1_c2",
    ]


# ---------------------------------------------------------------------------
# An empty table
# ---------------------------------------------------------------------------

def test_an_empty_table_selects_nothing_and_counts_zero():
    """The report is still well-formed, so a caller can print it unguarded."""
    empty = _frame().iloc[0:0]

    out, report = select(empty, scope="wells", guides=["g1"])

    assert out is empty
    assert report["rows"] == 0
    assert report["scope"] == "wells"


def test_no_table_at_all_yields_an_empty_frame_not_none():
    """A caller that immediately plots must not be handed None."""
    out, report = select(None, scope="guides", guides=["g1"])

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 0
    assert report["rows"] == 0


def test_an_unknown_scope_is_refused_with_the_supported_values():
    """A typo cannot silently widen a plot to a different population."""
    with pytest.raises(ValueError) as excinfo:
        select(_frame(), scope="neighbouring plates", guides=["g1"])

    message = str(excinfo.value)
    assert "neighbouring plates" in message
    assert all(scope in message for scope in ("guides", "wells", "all"))


def test_well_scope_without_a_well_column_keeps_only_the_chosen_guides():
    """The fallback is narrow and explains why well-mates are unavailable."""
    frame = _frame().drop(columns=["prc"])

    out, report = select(frame, scope="wells", guides=["g2"])

    assert out["grna"].tolist() == ["g2"]
    assert report["rows"] == 1
    assert report["chosen"] == 1
    assert report["mates"] == 0
    assert "names no well" in report["note"]


# ---------------------------------------------------------------------------
# What the plot prints underneath itself
# ---------------------------------------------------------------------------

def test_a_note_is_what_gets_described_when_there_is_one():
    """When the scope could not be built, the caption says why, not a count."""
    _, report = select(_frame(), scope="guides", guides=[])

    assert "no gRNA was chosen" in describe(report)


def test_the_whole_table_is_described_by_its_size():
    """`all` has no guides or wells to talk about, only how many points."""
    _, report = select(_frame(), scope="all")

    assert describe(report) == "every datapoint in the table: 4."


def test_the_guide_scope_is_described_by_guides_and_objects():
    """The caption has to distinguish the guide scope from the well scope."""
    _, report = select(_frame(), scope="guides", guides=["g1", "g2"])

    text = describe(report)
    assert "2 objects" in text
    assert "2 chosen gRNA(s)" in text


def test_the_well_scope_is_described_by_both_sides_and_its_wells():
    """The well scope's caption exists to name the well-mates it added."""
    out, report = select(_frame(), scope="wells", guides=["g1"])

    assert list(out[MATE_COLUMN]) == [CHOSEN, MATE]
    text = describe(report)
    assert "1 objects from the chosen gRNA(s)" in text
    assert "1 of their well-mates" in text
    assert "1 well(s)" in text
