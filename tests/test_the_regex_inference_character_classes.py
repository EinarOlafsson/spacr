"""The character class inferred for one filename slot, and the folder tally.

``_class_for`` builds the character class a slot's regex will use, from the
characters actually seen in that slot. Getting it too narrow makes the inferred
regex miss files; too wide, and it swallows the separator and merges two
slots. Both failures show up as "the regex matched nothing", far from here.
"""
from __future__ import annotations

import re

import pytest


def test_a_numeric_slot_is_just_digits():
    """The early return, which is the commonest slot by far."""
    from spacr.regex_infer import _class_for

    assert _class_for(["001", "002"], numeric=True) == r"\d+"


def test_a_letters_only_slot_asks_for_letters():
    """Arc 305 -> 307: no digit seen, so no 0-9 in the class."""
    from spacr.regex_infer import _class_for

    pattern = _class_for(["A", "B", "H"], numeric=False)

    assert "A-Za-z" in pattern
    assert "0-9" not in pattern
    assert re.fullmatch(pattern, "A") and re.fullmatch(pattern, "H")


def test_a_digits_only_slot_that_is_not_marked_numeric_asks_for_digits():
    """Arc 303 -> 305: no letter seen, so no A-Za-z in the class."""
    from spacr.regex_infer import _class_for

    pattern = _class_for(["01", "02"], numeric=False)

    assert "0-9" in pattern
    assert "A-Za-z" not in pattern


def test_a_mixed_slot_asks_for_both():
    """Both taken, which is what a well like A01 needs."""
    from spacr.regex_infer import _class_for

    pattern = _class_for(["A01", "H12"], numeric=False)

    assert "A-Za-z" in pattern and "0-9" in pattern
    assert re.fullmatch(pattern, "A01")


def test_punctuation_in_a_slot_is_escaped_into_the_class():
    """Arc 308 -> 310 taken: a hyphen or plus must not become a range.

    An unescaped '-' inside a character class is a RANGE, so a slot holding
    'A-1' would silently accept every character between two others. The escape
    is what stops an inferred regex quietly matching far more than it saw.
    """
    from spacr.regex_infer import _class_for

    pattern = _class_for(["A-1", "B-2"], numeric=False)

    assert re.fullmatch(pattern, "A-1")
    assert not re.fullmatch(pattern, "A/1")


def test_a_slot_with_nothing_in_it_falls_back_to_anything_but_a_separator():
    """Arc 308 -> 310 not taken AND no alnum: the documented fallback.

    ``[^_.]+`` is deliberately not ``.+``: the underscore and the dot are what
    divide the filename into slots, so a class that could match them would
    swallow the next slot whole.
    """
    from spacr.regex_infer import _class_for

    assert _class_for([""], numeric=False) == r"[^_.]+"
    assert _class_for([], numeric=False) == r"[^_.]+"


def test_folders_are_counted_only_for_matched_rows():
    """The tally that decides which folder a run is offered.

    An unmatched row has no parsed identity, so counting it would rank a
    folder by files the regex cannot read -- which is the opposite of what the
    count is for.
    """
    from spacr.regex_infer import structure

    preview = [
        {"matched": True, "folder": "/data/plate1"},
        {"matched": True, "folder": "/data/plate1"},
        {"matched": False, "folder": "/data/plate1"},
        {"matched": True, "folder": "/data/plate2"},
        {"matched": True, "folder": None},
        {"matched": True},
    ]

    assert structure(preview) == {"/data/plate1": 2, "/data/plate2": 1}
