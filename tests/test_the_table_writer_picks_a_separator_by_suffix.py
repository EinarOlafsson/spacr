"""Which separator a table is written with, and when the writer chooses none.

``CSV_SUFFIXES`` maps a suffix to its separator, and ``.txt`` maps to None on
purpose: a ``.txt`` table has no conventional separator, so the writer leaves
pandas' default alone rather than imposing one. Forcing a comma on a file the
user named ``.txt`` would produce a file that reads back wrongly everywhere
except spaCR.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def frame():
    return pd.DataFrame({"gene": ["a", "b"], "effect": [0.5, -0.2]})


@pytest.mark.parametrize("name, separator", [
    ("out.csv", ","),
    ("out.tsv", "\t"),
    ("out.tab", "\t"),
])
def test_a_known_suffix_chooses_its_separator(tmp_path, frame, name, separator):
    """The taken side, across every suffix the map declares."""
    from spacr.tabular import write_table

    target = write_table(frame, tmp_path / name)

    text = open(target, encoding="utf-8").read()
    assert separator in text.splitlines()[0]


def test_a_txt_table_is_left_at_the_writers_default(tmp_path, frame):
    """Arc 240 -> 242: the suffix maps to None, so no separator is imposed.

    ``.txt`` is in the map precisely so it is RECOGNISED as a table and still
    left alone. Imposing a comma would write a file that reads back wrongly
    everywhere except here, and dropping it from the map entirely would stop
    it being treated as a table at all.
    """
    from spacr.tabular import write_table

    target = write_table(frame, tmp_path / "out.txt")

    header = open(target, encoding="utf-8").read().splitlines()[0]
    assert "gene" in header and "effect" in header


def test_an_explicit_separator_from_the_caller_is_not_overridden(tmp_path,
                                                                 frame):
    """``setdefault`` and not assignment, which is what makes the map a default."""
    from spacr.tabular import write_table

    target = write_table(frame, tmp_path / "out.csv", sep=";")

    assert ";" in open(target, encoding="utf-8").read().splitlines()[0]


# ---------------------------------------------------------------------------
# table_columns — reading the header without canonicalising it
# ---------------------------------------------------------------------------

def test_the_header_can_be_read_exactly_as_written(tmp_path):
    """Arc 331 -> 335: ``canonicalise=False`` returns the file's own names.

    A caller inspecting a file the user just chose needs to show them THEIR
    column names. Canonicalising first would show names the file does not
    contain, and the user would look for a column that is not there.
    """
    from spacr.tabular import table_columns, write_table

    written = write_table(
        pd.DataFrame({"plate": ["p1"], "row": ["r1"], "value": [1.0]}),
        tmp_path / "raw.csv", canonicalise=False)

    columns = table_columns(written, canonicalise=False)

    assert list(columns) == ["plate", "row", "value"]


def test_the_header_can_be_read_canonicalised(tmp_path):
    """The taken side, which is what a reader that will join on it wants."""
    from spacr.tabular import table_columns, write_table

    written = write_table(
        pd.DataFrame({"plate": ["p1"], "row": ["r1"], "value": [1.0]}),
        tmp_path / "raw.csv", canonicalise=False)

    columns = list(table_columns(written, canonicalise=True))

    assert "value" in columns
    assert columns != ["plate", "row", "value"] or "plateID" in columns
