"""The sweep table's numeric cells decline a comparison they cannot make.

The trials table sorts by score, and a score cell keeps its formatted display
text ("0.812") while sorting on the number behind it. Qt hands the comparison
whatever is in the other row, which is a table item for every row the sweep
wrote and can be something else entirely once anything else puts a cell in
the table.

Answering ``False`` for a comparison it cannot make is the failure that
matters here: ``sorted`` believes every answer it is given, so a cell that
guessed would produce an order nobody could account for. Returning
``NotImplemented`` lets Python raise instead.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QTableWidgetItem                   # noqa: E402

from spacr.qt.screens.hyperparam import _NumericTableItem        # noqa: E402


def test_two_score_cells_order_by_their_numbers_not_their_text(qapp):
    """"0.9" sorts above "0.10" only if the number is what is compared."""
    smaller = _NumericTableItem("0.10", 0.10)
    larger = _NumericTableItem("0.9", 0.9)

    assert smaller < larger
    assert not larger < smaller


def test_a_failed_trial_sorts_below_a_scored_one(qapp):
    """A "-" row is a hole, and a hole belongs under the results.

    It used to fall back to comparing display text, which put "-" ABOVE
    every score whichever way the column was sorted, because "-" sorts
    before "0" as a word.
    """
    scored = _NumericTableItem("0.812", 0.812)
    assert scored < QTableWidgetItem("-")

    # And the way the table actually builds both cells.
    from spacr.qt.widgets.sortable_table import table_item

    assert table_item("0.812", key=0.812) < table_item("-")
    assert not table_item("-") < table_item("0.812", key=0.812)


def test_a_comparison_with_something_that_has_no_text_is_declined(qapp):
    """Neither a number nor a guess: the cell says it cannot answer.

    Answering ``False`` would let ``sorted`` finish and report an order that
    was never computed, which is worse than the exception.
    """
    scored = _NumericTableItem("0.812", 0.812)

    assert scored.__lt__(0.5) is NotImplemented
    assert scored.__lt__(object()) is NotImplemented
    with pytest.raises(TypeError):
        scored < 0.5


def test_a_declining_cell_still_sorts_against_its_own_kind(qapp):
    """Declining one comparison does not break the table's own ordering."""
    cells = [_NumericTableItem("0.5", 0.5), _NumericTableItem("0.05", 0.05),
             _NumericTableItem("0.75", 0.75)]

    assert [cell.text() for cell in sorted(cells)] == ["0.05", "0.5", "0.75"]
