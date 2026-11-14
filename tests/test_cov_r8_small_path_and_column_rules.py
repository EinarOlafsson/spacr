"""Two small rules: which count path to use, and how a column is prefixed.

Both are the kind of helper whose failure is silent. `refit` picking the
wrong count table produces a refit of the wrong data; `TableMerge`
prefixing a key column would break the join it exists to make.
"""
from __future__ import annotations

import pytest

from spacr.plate_measurements import OBJECT_COLUMN, TableMerge, _UNPREFIXED
from spacr.refit import _first_usable_count_path


class TestChoosingACountPath:
    """`count_data` may be one path, several, or nothing usable."""

    @pytest.mark.parametrize("count_data, expected", [
        pytest.param("/tmp/counts.csv", "/tmp/counts.csv",
                     id="a-single-path-is-taken"),
        pytest.param(["/tmp/a.csv", "/tmp/b.csv"], "/tmp/a.csv",
                     id="the-first-usable-of-several-is-taken"),
        # THE LOOP'S SECOND PASS. A settings form that offers several count
        # slots leaves the unused ones empty, so the list arriving here
        # routinely starts with blanks.
        pytest.param(["", None, "   ", "/tmp/real.csv"], "/tmp/real.csv",
                     id="empty-entries-are-skipped-to-reach-a-real-one"),
        # A path pasted with a trailing newline is still that path.
        pytest.param("  /tmp/counts.csv \n", "/tmp/counts.csv",
                     id="a-path-is-stripped-before-it-is-judged"),
        # `os.fspath` raises TypeError on an int or a dict; skipping it lets
        # a later entry answer rather than failing the whole refit.
        pytest.param([42, {"a": 1}, "/tmp/real.csv"], "/tmp/real.csv",
                     id="an-entry-that-is-not-path-like-is-skipped"),
    ])
    def test_the_first_usable_count_path_is_taken(self, count_data, expected):
        assert _first_usable_count_path(
            {"count_data": count_data}) == expected

    def test_nothing_usable_answers_none(self):
        assert _first_usable_count_path({"count_data": ["", None]}) is None
        assert _first_usable_count_path({"count_data": []}) is None
        assert _first_usable_count_path({}) is None

    def test_only_bad_entries_answer_none_rather_than_raising(self):
        assert _first_usable_count_path({"count_data": [42, object()]}) is None

    def test_a_path_like_object_is_accepted(self):
        from pathlib import Path

        assert _first_usable_count_path(
            {"count_data": Path("/tmp/counts.csv")}) == str(Path("/tmp/counts.csv"))


class TestPrefixingAMergedColumn:
    """`TableMerge.merged_column` decides what a column is called after a join.

    A measurement gets its table's prefix so two tables' `area` columns
    do not collide. A KEY must not, or the join has nothing to join on.
    """

    @pytest.fixture()
    def merge(self):
        """A TableMerge naming the `cell` table and two join keys.

        `plan` and `rows` are required and describe the merge that
        produced this record; neither is read by `merged_column`, which
        is a pure naming rule over `table` and `keys`.
        """
        return TableMerge(table="cell", plan=None, rows=0,
                          keys=("plateID", "rowID"))

    def test_a_measurement_gains_its_tables_prefix(self, merge):
        assert merge.merged_column("area") == "cell_area"

    def test_a_column_already_prefixed_is_left_alone(self, merge):
        """Otherwise a second pass would produce `cell_cell_area`."""
        assert merge.merged_column("cell_area") == "cell_area"

    def test_a_join_key_is_never_prefixed(self, merge):
        """Prefixing a key would break the join this class exists to make."""
        assert merge.merged_column("plateID") == "plateID"
        assert merge.merged_column("rowID") == "rowID"

    @pytest.mark.parametrize("name", sorted(set(_UNPREFIXED) - {OBJECT_COLUMN}))
    def test_the_shared_identity_columns_are_never_prefixed(self, merge,
                                                            name):
        """THE UNCOVERED ARM.

        These name the same thing in every table -- the plate, the well,
        the field, the source database. Prefixing them would give each
        table its own copy of an identity that is meant to be shared.
        """
        assert merge.merged_column(name) == name

    def test_the_object_column_IS_prefixed(self, merge):
        """The exception, and the reason the arm carries `!= OBJECT_COLUMN`.

        `object_label` is per-table: cell 3 and nucleus 3 are different
        objects, so the label has to say which table it came from.
        """
        assert merge.merged_column(OBJECT_COLUMN) == f"cell_{OBJECT_COLUMN}"


def test_a_byte_path_that_decodes_to_whitespace_is_skipped():
    """The loop's other way round: emptiness found only after decoding.

    `is_empty_path` recognises a blank STRING, so `"   "` never reaches
    the second check -- it is dropped at the top of the loop. A bytes
    path is not recognised there, survives `os.fspath`, and is only seen
    to be blank once `os.fsdecode` and `.strip()` have run.

    That is the arc: a candidate rejected at the BOTTOM of the loop
    rather than the top, so the loop goes round to try the next one. A
    settings file holding a byte path is unusual; a helper that returned
    a blank path because of one would be worse.
    """
    from spacr.refit import _first_usable_count_path

    assert _first_usable_count_path(
        {"count_data": [b"  ", "/tmp/real.csv"]}) == "/tmp/real.csv"


def test_a_byte_path_with_content_is_decoded_and_used():
    """And the same input shape, when it does name something."""
    from spacr.refit import _first_usable_count_path

    assert _first_usable_count_path(
        {"count_data": [b"/tmp/counts.csv"]}) == "/tmp/counts.csv"
