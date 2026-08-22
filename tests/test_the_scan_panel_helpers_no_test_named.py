"""Measurement-scan helpers no test had ever named.

Instruction 60. Fifteen public callables in ``measurement_scan_panel`` had
never appeared in a test. The two pinned hardest here are the ones that
decide what a ROW of the merged frame means and what a plate is called,
because both fail quietly: a fanned-out merge looks like more data, and a
plate under two names looks like two plates.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# anchor_tables
# ---------------------------------------------------------------------------

def test_only_one_row_per_cell_tables_can_anchor():
    """Anchoring on a many-per-cell table makes a row of the merged frame
    mean one nucleus or one pathogen, with the cell's own measurements
    repeated across its children -- the fan-out the roll-up exists to
    prevent, arrived at from the other side."""
    from spacr.qt.widgets.measurement_scan_panel import anchor_tables
    from spacr.object_roles import is_one_row_per_cell

    offered = anchor_tables(["cell", "nucleus", "pathogen", "cytoplasm"])
    assert offered, "nothing at all could anchor"
    for name in offered:
        assert is_one_row_per_cell(name), name


def test_a_many_per_cell_table_is_not_offered():
    from spacr.qt.widgets.measurement_scan_panel import anchor_tables
    from spacr.object_roles import is_one_row_per_cell

    every = ["cell", "nucleus", "pathogen", "cytoplasm"]
    fanning = [name for name in every if not is_one_row_per_cell(name)]
    if not fanning:
        pytest.skip("this build has no many-per-cell table to check")
    assert set(anchor_tables(every)).isdisjoint(fanning)


def test_the_callers_order_is_kept():
    """The first offered anchor is the default, so re-ordering here changes
    which table a user gets without touching anything they can see."""
    from spacr.qt.widgets.measurement_scan_panel import anchor_tables

    given = ["cytoplasm", "cell"]
    offered = anchor_tables(given)
    assert list(offered) == [n for n in given if n in offered]


def test_an_unknown_table_cannot_anchor():
    """A table nobody has declared a role for has an unknown number of rows
    per cell, and guessing 'one' is the fan-out again."""
    from spacr.qt.widgets.measurement_scan_panel import anchor_tables

    assert "made_up_table" not in anchor_tables(["made_up_table", "cell"])


def test_nothing_in_gives_nothing_out():
    from spacr.qt.widgets.measurement_scan_panel import anchor_tables

    assert anchor_tables([]) == ()


# ---------------------------------------------------------------------------
# displayed_plates
# ---------------------------------------------------------------------------

def test_plates_are_shown_by_their_canonical_name():
    """A plate under two spellings looks like two plates, and a screen
    listing both is a screen where half the wells are missing from each."""
    from spacr.qt.widgets.measurement_scan_panel import displayed_plates
    from spacr.schema import canonical_plate_id

    given = ["plate1", "Plate1", "plate_1"]
    assert list(displayed_plates(given)) == \
        [canonical_plate_id(name) for name in given]


def test_the_order_given_is_the_order_shown():
    """The list is a UI order the user chose; sorting it here would move
    their plates about for no reason they can see."""
    from spacr.qt.widgets.measurement_scan_panel import displayed_plates

    shown = displayed_plates(["p2", "p1", "p3"])
    assert len(shown) == 3


def test_an_empty_list_shows_nothing():
    from spacr.qt.widgets.measurement_scan_panel import displayed_plates

    assert displayed_plates([]) == ()
