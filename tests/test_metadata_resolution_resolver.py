"""Table-driven tests of ``spacr.metadata_resolution.resolve_metadata_columns``.

Plate, row and column are what every downstream number is grouped by, so
the resolver never invents one.  One test walks the frames it resolves
(column map, well derivation, pseudo wells, a prompt's decision) with the
exact resulting identity columns; the other walks every refusal with the
exact exception type, missing columns and message.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import schema
from spacr.metadata_resolution import (
    MetadataDecision,
    MetadataResolutionRequired,
    clear_run_metadata_decisions,
    resolve_metadata_columns,
)

ROW, COL, PLATE = schema.ROW_KEY, schema.COLUMN_KEY, schema.PLATE_KEY


@pytest.fixture(autouse=True)
def _no_remembered_decisions():
    """Remembered prompt answers are process-wide; start and end each test clean."""
    clear_run_metadata_decisions()
    yield
    clear_run_metadata_decisions()


def test_resolve_metadata_columns_fills_only_what_is_missing(subtests):
    """Exact identity columns, derived-from and pseudo map for each resolvable frame."""
    pseudo = {"pseudo_source": "sample", "allow_pseudo": True}
    cases = [
        ("well fills the missing row, keeps the supplied column",
         pd.DataFrame({"well": ["A01", "B02"], COL: ["c9", "c9"]}),
         [ROW, COL], {"well_column": "well"},
         {ROW: ["r1", "r2"], COL: ["c9", "c9"]}, "well", None),
        ("well fills both axes",
         pd.DataFrame({"well": ["A01", "H12"]}),
         [ROW, COL], {"well_column": "well"},
         {ROW: ["r1", "r8"], COL: ["c1", "c12"]}, "well", None),
        ("pseudo wells fill the missing column only",
         pd.DataFrame({"sample": ["a", "b", "a"], ROW: ["r9"] * 3}),
         [ROW, COL], pseudo,
         {ROW: ["r9"] * 3, COL: ["c1", "c2", "c1"]}, None, ["c1", "c2"]),
        ("pseudo wells fill the missing row only; audit names both axes",
         pd.DataFrame({"sample": ["a", "b", "a"], COL: ["c7"] * 3}),
         [ROW, COL], pseudo,
         {COL: ["c7"] * 3, ROW: ["r1", "r1", "r1"]}, None, ["c1", "c2"]),
        ("column map renames its source",
         pd.DataFrame({"barcode": ["p1", "p2"]}),
         [PLATE], {"column_map": {PLATE: "barcode"}},
         {PLATE: ["p1", "p2"]}, None, None),
    ]
    for case_id, frame, required, kwargs, columns, derived, pseudo_cols in cases:
        with subtests.test(msg=case_id):
            result = resolve_metadata_columns(frame, required, **kwargs)
            for name, values in columns.items():
                assert list(result.frame[name]) == values
            assert result.derived_from_well == derived
            if pseudo_cols is None:
                assert result.pseudo_map == ()
            else:
                assert [entry[COL] for entry in result.pseudo_map] == \
                    pseudo_cols

    with subtests.test(msg="a prompt's MetadataDecision is applied"):
        seen = []

        def prompt(request):
            seen.append(request)
            return MetadataDecision(column_map={PLATE: "condition"},
                                    remember=False)

        result = resolve_metadata_columns(
            pd.DataFrame({"condition": ["a", "b"]}), [PLATE], prompt=prompt)
        assert list(result.frame[PLATE]) == ["a", "b"]
        assert result.column_map == {PLATE: "condition"}
        assert len(seen) == 1


def test_resolve_metadata_columns_refuses_to_guess_an_identity(subtests):
    """Each unresolvable frame raises its exact type with the missing columns or reason."""
    both = {ROW, COL}
    required_cases = [
        ("a mapped source that is not there",
         pd.DataFrame({"barcode": ["p1"], "value": [1.0]}), [PLATE],
         {"column_map": {PLATE: "not_a_column"}},
         (PLATE,), ("barcode", "value")),
        ("a well column that does not exist",
         pd.DataFrame({"condition": ["a", "b"]}), [ROW, COL],
         {"well_column": "well"}, both, ("condition",)),
        ("one unparseable well invalidates the derivation",
         pd.DataFrame({"well": ["A01", "sample-2"]}), [ROW, COL],
         {"well_column": "well"}, both, ("wellID",)),
        ("a pseudo source that is not a column",
         pd.DataFrame({"condition": ["a", "b"]}), [ROW, COL],
         {"pseudo_source": "treatment", "allow_pseudo": True}, both,
         ("condition",)),
        ("pseudo wells not allowed",
         pd.DataFrame({"sample": ["a", "b"]}), [ROW, COL],
         {"pseudo_source": "sample"}, both, ("sample",)),
    ]
    for case_id, frame, required, kwargs, missing, available in required_cases:
        with subtests.test(msg=case_id):
            with pytest.raises(MetadataResolutionRequired) as excinfo:
                resolve_metadata_columns(frame, required, **kwargs)
            assert set(excinfo.value.missing) == set(missing)
            assert tuple(excinfo.value.available) == available

    type_cases = [
        ("a dict is not a frame",
         lambda: resolve_metadata_columns({"plateID": ["p1"]}, [PLATE]),
         TypeError, "metadata resolver needs a DataFrame, got dict"),
        ("a prompt answering with a mapping",
         lambda: resolve_metadata_columns(
             pd.DataFrame({"condition": ["a"]}), [PLATE],
             prompt=lambda request: {"plateID": "condition"}),
         TypeError, "metadata prompt must return MetadataDecision"),
        ("a map colliding only by case",
         lambda: resolve_metadata_columns(
             pd.DataFrame({"Sample": [1], "SAMPLE": [2]}), ["sample"],
             column_map={"sample": "Sample"}),
         ValueError, "case-insensitive column collision"),
    ]
    for case_id, call, exc_type, fragment in type_cases:
        with subtests.test(msg=case_id):
            with pytest.raises(exc_type) as excinfo:
                call()
            assert fragment in str(excinfo.value)
