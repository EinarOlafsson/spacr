"""A scorecard cell that overflows to infinity must not crash the reader.

``read_scorecard`` is contracted to hand back ``([], reason)`` rather than
raise, so a corrupt card degrades into a message a screen can show. The
``n_objects`` cell is parsed as ``int(float(...))``, and the two failures that
parse reaches are not the same exception: ``float("nan")`` raises
``ValueError`` on the ``int()``, while ``float("inf")`` raises
``OverflowError``. Only the first was caught, so an infinite count escaped the
contract and reached every caller of :func:`spacr.seg_qc.read_digest`.
"""
import os

import pytest

from spacr.seg_qc import CARD_DIR, CARD_PREFIX, read_scorecard

HEADER = "field,object_type,n_objects,severity,flags,note\n"


def _card(tmp_path, n_objects):
    path = str(tmp_path / CARD_DIR / f"{CARD_PREFIX}cell.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        handle.write(HEADER)
        handle.write(f"plate1_A01_f1,cell,{n_objects},ok,,\n")
    return path


@pytest.mark.parametrize("cell", ["inf", "-inf", "1e999"])
def test_an_object_count_that_overflows_is_read_as_none_counted(tmp_path, cell):
    """An infinite count is refused as a count, not raised at the caller."""
    rows, error = read_scorecard(_card(tmp_path, cell))

    assert error == ""
    assert len(rows) == 1
    assert rows[0].n_objects == 0
    assert rows[0].field == "plate1_A01_f1"


def test_a_count_that_is_not_a_number_at_all_is_still_read_as_none_counted(
        tmp_path):
    """The nan path shares the fallback, and must keep working alongside it."""
    rows, error = read_scorecard(_card(tmp_path, "nan"))

    assert error == ""
    assert len(rows) == 1
    assert rows[0].n_objects == 0


def test_a_real_count_survives_the_widened_guard(tmp_path):
    """Widening the except must not swallow a count that parses correctly."""
    rows, error = read_scorecard(_card(tmp_path, "417"))

    assert error == ""
    assert rows[0].n_objects == 417
