"""Well parsing accepts a list of tokens and refuses an unreadable row letter.

The picker hands `parse` a str typed by a user; the settings layer hands it a
list it has already split. Both have to work, because a list silently treated
as a string would iterate characters and select nothing.

The row-letter guard is the other half: `row_number` is the only thing that
can decide a letter pair is not a plate row, and if its failure escaped as a
bare ValueError the caller would get a message about `string.index` instead
of one naming the token.
"""
from __future__ import annotations

import pytest

from spacr import well_spec


def test_a_list_of_tokens_selects_the_same_wells_as_the_string():
    """A pre-split value must not be iterated character by character."""
    from_text = well_spec.parse("r1, c2, A03", layout=96)
    from_list = well_spec.parse(["r1", "c2", "A03"], layout=96)

    assert from_list == from_text
    assert (1, 1) in from_list, "r1 selected no wells in the first row"
    assert (1, 3) in from_list, "A03 is row A column 3"
    assert all((row, 2) in from_list for row in range(1, 9)), (
        "c2 must select every row of a 96-well plate")


def test_a_row_letter_that_is_not_a_plate_row_names_the_token(monkeypatch):
    """The message has to identify the token and the letters, because a
    ValueError out of the letter-to-number conversion says only that some
    character was not found in the alphabet."""
    def refuse(_label):
        raise ValueError("not a row letter")

    monkeypatch.setattr(well_spec, "row_number", refuse)

    with pytest.raises(well_spec.WellSpecError) as excinfo:
        well_spec.parse_one("AB12", layout=384)

    message = str(excinfo.value)
    assert "'AB12'" in message
    assert "'AB'" in message
    assert "does not name a row" in message
    assert excinfo.value.__cause__ is None, (
        "the underlying ValueError is suppressed so the message a user reads "
        "is the one about the token")
