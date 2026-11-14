"""What :func:`spacr._lineage_measurements._measured_lineage_inputs` refuses to colour a lineage by.

A lineage is coloured by one measured feature, read row for row against the
saved field, frame and label identities.  Anything that would make that read
ambiguous or meaningless is refused with its own sentence, and the sources
are never modified by a refusal.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr import _lineage_measurements as lm
from tests.test_lineage_measurements_f537 import project  # noqa: F401


def _sql(statement):
    def edit(db):
        with sqlite3.connect(db) as con:
            con.execute(statement)
    return edit


REFUSALS = [
    ("an_identity_column_is_not_a_feature", None, "object_label",
     "lineage colour must name a numeric measurement feature"),
    ("an_empty_column_name", None, "",
     "lineage colour must name a numeric measurement feature"),
    ("a_feature_the_table_lacks", None, "cell_area",
     "lineage measurement table cell lacks identity or feature column 'cell_area'"),
    ("a_zero_label", _sql("UPDATE cell SET object_label = 0 WHERE rowid = 1"),
     "cell_intensity", "lineage measurement labels must be positive integers"),
    ("a_fractional_label", _sql("UPDATE cell SET object_label = 1.5 WHERE rowid = 1"),
     "cell_intensity", "lineage measurement labels must be positive integers"),
    ("a_repeated_object", _sql("INSERT INTO cell SELECT * FROM cell WHERE rowid = 1"),
     "cell_intensity", "lineage measurements have ambiguous duplicate object identities"),
    ("a_text_value", _sql("UPDATE cell SET cell_intensity = 'bright' WHERE rowid = 1"),
     "cell_intensity", "lineage measurement 'cell_intensity' must be numeric and finite or missing"),
    ("an_infinite_value", _sql("UPDATE cell SET cell_intensity = 9e999 WHERE rowid = 1"),
     "cell_intensity", "lineage measurement 'cell_intensity' must be numeric and finite or missing"),
    ("no_measured_row_matches", _sql("UPDATE cell SET prcf = 'plate9_r1_c1_f1_t1'"),
     "cell_intensity", "no measured objects match the saved lineage identities"),
]


def test_measured_lineage_inputs_refuse_an_ambiguous_feature(project, subtests):  # noqa: F811
    """Each malformed request raises ValueError with its exact sentence and leaves the tracks file as it was."""
    path, db = project[0], project[1]
    tracks_before = path.read_bytes()
    pristine = db.read_bytes()
    for case_id, edit, column, message in REFUSALS:
        with subtests.test(msg=case_id):
            db.write_bytes(pristine)
            if edit is not None:
                edit(db)
            with pytest.raises(ValueError) as caught:
                lm._measured_lineage_inputs(db, path, column)
            assert str(caught.value) == message
            assert path.read_bytes() == tracks_before
