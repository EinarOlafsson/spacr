"""The merge offers png_list and the dependent variable (instruction 213).

Three routes, tried in order, and the order is the instruction:

  1. the ID columns
  2. the image path, split into its parts
  3. the well from the path, translated to row and column

A FALLBACK IS A FALLBACK, NOT A SECRET. Which route was used goes in the
report with the rows it matched -- a join that silently degraded is one
nobody can check, and the degradation is worth knowing because it means a
column is missing upstream.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.dependent_join import (ID_COLUMNS, ROUTES, describe, join,
                                  parts_from_path, well_to_row_and_column)


@pytest.fixture
def objects():
    return pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c2"],
        "fieldID": ["3", "3"],
        "objectID": ["7", "8"],
        "area": [10.0, 20.0],
    })


@pytest.fixture
def by_ids():
    return pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c2"],
        "fieldID": ["3", "3"],
        "objectID": ["7", "8"],
        "score": [0.9, 0.1],
    })


@pytest.fixture
def by_path():
    return pd.DataFrame({
        "png_path": ["/x/plate1_A01_3_7.png", "/x/plate1_A02_3_8.png"],
        "score": [0.9, 0.1],
    })


class TestRouteOne:

    def test_the_id_columns_join(self, objects, by_ids):
        out, report = join(objects, by_ids)
        assert report["route"] == ROUTES[0][0]
        assert out["score"].tolist() == [0.9, 0.1]

    def test_it_matched_every_row(self, objects, by_ids):
        _, report = join(objects, by_ids)
        assert report["matched"] == 2 and report["rows"] == 2

    def test_the_direct_route_is_not_called_a_fallback(self, objects,
                                                       by_ids):
        _, report = join(objects, by_ids)
        assert "fallback" not in describe(report)


class TestRouteTwo:
    """Used when a column from route 1 is missing."""

    def test_the_path_split_joins(self, objects, by_path):
        out, report = join(objects, by_path)
        assert "image path" in report["route"]
        assert out["score"].tolist() == [0.9, 0.1]

    def test_one_missing_id_column_still_joins(self, objects):
        partial = pd.DataFrame({
            "plateID": ["plate1", "plate1"],
            "rowID": ["r1", "r1"],
            "columnID": ["c1", "c2"],
            "objectID": ["7", "8"],          # fieldID is gone
            "png_path": ["/x/plate1_A01_3_7.png", "/x/plate1_A02_3_8.png"],
            "score": [0.5, 0.6],
        })
        out, report = join(objects, partial)
        assert report["matched"] == 2
        assert out["score"].tolist() == [0.5, 0.6]

    def test_the_report_says_it_was_a_fallback(self, objects, by_path):
        _, report = join(objects, by_path)
        assert "fallback" in describe(report), (
            "a join that silently degraded is one nobody can check")


class TestTheWellTranslation:
    """"with the format r1 and c2" -- the spelling png_list.rowID uses."""

    def test_a_letter_and_a_number_become_r_and_c(self):
        assert well_to_row_and_column("A01") == ("r1", "c1")
        assert well_to_row_and_column("B12") == ("r2", "c12")

    def test_a_two_letter_row_keeps_counting(self):
        assert well_to_row_and_column("AA01")[0] == "r27"

    def test_an_already_translated_well_is_left_alone(self):
        assert well_to_row_and_column("r3c11") == ("r3", "c11")

    def test_it_is_not_a_bare_number(self):
        """`spacr/predictions.py` records that png_list.rowID returns 'r1',
        and a route producing `1` would match nothing -- a failure that
        looks exactly like a screen with no overlap."""
        row, column = well_to_row_and_column("A01")
        assert row.startswith("r") and column.startswith("c")

    def test_nonsense_gives_nothing_rather_than_a_guess(self):
        assert well_to_row_and_column("???") == ("", "")


class TestThePathParts:

    def test_plate_well_field_object_all_come_back(self):
        got = parts_from_path("/crops/plate1_A01_3_7.png")
        assert got["plateID"] == "plate1"
        assert got["fieldID"] == "3"
        assert got["objectID"] == "7"
        assert got["rowID"] == "r1" and got["columnID"] == "c1"

    def test_a_path_with_no_parts_gives_an_empty_dict(self):
        assert parts_from_path("nonsense.png") == {}


class TestZeroMatchesIsAFailure:
    """"AND A ROUTE THAT MATCHES NOTHING IS A FAILURE, not an empty
    answer"."""

    def test_no_shared_columns_raises(self, objects):
        with pytest.raises(ValueError):
            join(objects, pd.DataFrame({"nope": [1]}))

    def test_matching_no_row_raises(self, objects):
        elsewhere = pd.DataFrame({
            "plateID": ["other"], "rowID": ["r9"], "columnID": ["c9"],
            "fieldID": ["9"], "objectID": ["9"], "score": [1.0]})
        with pytest.raises(ValueError):
            join(objects, elsewhere)

    def test_the_refusal_says_what_was_tried(self, objects):
        with pytest.raises(ValueError, match="could not be joined"):
            join(objects, pd.DataFrame({"nope": [1]}))

    def test_an_empty_table_raises(self, objects):
        with pytest.raises(ValueError):
            join(objects, pd.DataFrame())


class TestTheSpellingsAgree:
    """Instruction 213 C: "as the import logic dictates for standardization
    (which should be true acress the board in spacr)"."""

    def test_the_join_uses_the_imports_spelling(self):
        assert ID_COLUMNS == ("plateID", "rowID", "columnID", "fieldID",
                              "objectID")

    def test_the_names_come_from_the_schema(self):
        """Never respelled locally: a second copy of the names is the module
        that silently matches nothing, waiting to happen."""
        from spacr import schema

        assert ID_COLUMNS == (schema.PLATE_KEY, schema.ROW_KEY,
                              schema.COLUMN_KEY, schema.FIELD_KEY,
                              schema.OBJECT_KEY)

    def test_all_five_identifiers_canonicalise(self):
        """objectID was the only one of the five with no constant, which is
        how object_id came to be spelled four ways while the other four were
        normalised."""
        from spacr.schema import canonical_column_name

        assert canonical_column_name("plate_id") == "plateID"
        assert canonical_column_name("row_id") == "rowID"
        assert canonical_column_name("column_id") == "columnID"
        assert canonical_column_name("field_id") == "fieldID"
        assert canonical_column_name("object_id") == "objectID"

    def test_a_bare_object_column_is_left_alone(self):
        """In a measurement table it as often means the object TYPE as its
        number, and renaming that into an identifier would corrupt the join
        it lands in rather than merely mislabel a column."""
        from spacr.schema import canonical_column_name

        assert canonical_column_name("object") == "object"


class TestWhatItBringsAcross:

    def test_only_new_columns(self, objects, by_ids):
        out, _ = join(objects, by_ids)
        assert out["area"].tolist() == [10.0, 20.0], (
            "a column present on both sides is already what the caller has")

    def test_one_named_value_can_be_asked_for(self, objects, by_ids):
        by_ids["extra"] = [1, 2]
        out, _ = join(objects, by_ids, value="score")
        assert "score" in out.columns and "extra" not in out.columns

    def test_the_report_lists_what_was_added(self, objects, by_ids):
        _, report = join(objects, by_ids)
        assert report["added"] == ["score"]
