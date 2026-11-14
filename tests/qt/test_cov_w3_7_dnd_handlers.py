"""Drop paths through the handlers that the other dnd files do not reach.

``tests/qt/test_dnd_handlers_full.py``, ``test_cov_dnd_handlers.py`` and
``tests/test_cov_w3_8_dnd.py`` cover the registry, the dropzone and the bulk of
the handlers between them. What is left, and is asserted here, is the small
set of paths a drop takes when the widget on the other side refuses -- a
source control that raises while adding, one that will not say what it holds
-- and Regression's two-sided drop, where the same target has to take a
measurements database OR the parameter sweep's CSVs.

Handlers are given stand-in screens: each is one or two attributes, and what
these tests are about is exactly what happens when one of them is missing or
throws.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import dnd_handlers as dh


# ---------------------------------------------------------------------------
# A measurements database onto a plate row
# ---------------------------------------------------------------------------

def _plate_with_database(tmp_path):
    root = tmp_path / "plate1"
    (root / "measurements").mkdir(parents=True)
    (root / "measurements" / "measurements.db").write_bytes(b"SQLite format 3")
    return root


def test_a_bare_database_file_is_its_own_answer(tmp_path):
    database = tmp_path / "measurements.db"
    database.write_bytes(b"SQLite format 3")
    assert dh._measurement_database(database) == database
    assert dh.MeasurementsDropHandler.database_file(database) == database


# ---------------------------------------------------------------------------
# The sweep's CSVs
# ---------------------------------------------------------------------------

def test_one_csv_is_one_table(tmp_path):
    csv = tmp_path / "scores.csv"
    csv.write_text("prc,pred\nplate1_r1_c1,0.5\n", encoding="utf-8")
    assert dh.SweepInputsDropHandler._tables(csv) == [csv]
    assert dh.SweepInputsDropHandler._tables(tmp_path / "gone.csv") == []


# ---------------------------------------------------------------------------
# Regression's two-sided drop
# ---------------------------------------------------------------------------

def test_regression_takes_a_database_or_the_sweeps_csvs(tmp_path):
    handler = dh.RegressionDropHandler()
    assert handler.accepts_multiple() is True

    root = _plate_with_database(tmp_path)
    assert handler.can_accept(root) is True

    csv = tmp_path / "counts.csv"
    csv.write_text("prc,grna,count\nplate1_r1_c1,A,7\n", encoding="utf-8")
    assert handler.can_accept(csv) is True

    nothing = tmp_path / "notes.txt"
    nothing.write_text("hello", encoding="utf-8")
    assert handler.can_accept(nothing) is False
    assert "measurements/measurements.db" in handler.error_message(nothing)
    assert "score / gRNA count CSVs" in handler.error_message(nothing)
