"""A column family nothing can select is a family nothing will use.

INSTRUCTION 388's remaining work: "the column family through
``spacr/columns.py``, so regression and hit calling pick the labels up with
no changes".

`measure._with_bystanders` writes `is_bystander`, `is_distal` and
`distance_to_infected` onto the CELL table, and names them for what they say
rather than for the object they say it about -- so they carry NO OBJECT
PREFIX. `parse_column`'s structural parse looks for one, does not find it,
and returns `family="unknown"` with no object. Everything downstream that
groups by family or by object then cannot see them.

THE PRECEDENT IS ALREADY IN THE FILE, and it names the same consequence.
Embedding dimensions carry no object prefix either, and `parse_column` has a
branch for them BEFORE the structural parse, because "falling through to the
structural parse would classify every one of them as 'unknown' and hide the
whole family from the pickers". This is that, for the three columns 388 added.

NOT A NEW FAMILY. `spatial` already covers it in its own words -- "where an
object ... sits relative to ... other segmented object types ... same-type
neighbourhood and touching measurements" -- and the `KNOWN_PROPERTIES`
entries have said `"spatial"` since they were written. Nothing was reading
them.
"""
from __future__ import annotations

import pytest

from spacr import column_groups
from spacr.feature_dict import KNOWN_PROPERTIES, parse_column

BYSTANDER_COLUMNS = ("is_bystander", "is_distal", "distance_to_infected")


@pytest.mark.parametrize("column", BYSTANDER_COLUMNS)
def test_each_column_resolves_to_the_family_its_entry_already_declared(column):
    """The metadata was right; the parse was not reading it."""
    entry = parse_column(column)

    assert entry.family == "spatial", (
        f"{column} parses as family {entry.family!r}, so it is invisible to "
        "anything that groups by family")
    assert KNOWN_PROPERTIES[column].family == "spatial", (
        "the dictionary entry itself no longer says spatial, which would "
        "make the parse above right for the wrong reason")


@pytest.mark.parametrize("column", BYSTANDER_COLUMNS)
def test_each_column_belongs_to_the_cell_it_was_written_onto(column):
    """OBJECT, NOT ONLY FAMILY. `measure._with_bystanders` writes these onto
    the cell table, so a picker that offers "the cell columns" has to offer
    them. Without an object they were in no object group at all.
    """
    assert parse_column(column).object_type == "cell"


def test_the_family_is_reachable_through_the_grouping_regression_uses():
    """THE END OF THE PATH, not the middle of it.

    `column_groups.classify` is what regression selection and hit calling go
    through, so this is the assertion that corresponds to what 388 asked for.
    A test that stopped at `parse_column` would pass while the pickers stayed
    blind.
    """
    columns = list(BYSTANDER_COLUMNS) + ["cell_area",
                                         "cell_channel_1_mean_intensity"]
    groups = column_groups.classify(columns)

    assert set(groups["family"].get("spatial", [])) == set(BYSTANDER_COLUMNS)
    assert "unknown" not in groups["family"], (
        f"a family called 'unknown' is still being produced: "
        f"{groups['family'].get('unknown')}")
    for column in BYSTANDER_COLUMNS:
        assert column in groups["object"]["cell"], (
            f"{column} is in no object group, so 'the cell columns' does not "
            "offer it")


def test_a_column_that_is_genuinely_unknown_is_still_unknown():
    """THE NEGATIVE HALF. The fix is a lookup of three known names, not a
    rule that makes bare names spatial -- otherwise every future unprefixed
    column silently joins a family it was never measured into.
    """
    entry = parse_column("some_column_nobody_declared")

    assert entry.family == "unknown"
    assert entry.object_type is None
