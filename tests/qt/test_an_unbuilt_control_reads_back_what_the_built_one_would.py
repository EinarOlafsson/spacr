"""A setting whose control is not built yet reads back what the control would.

Pinned behaviour of ``_value_a_plain_control_holds`` in
:mod:`spacr.qt.screens.settings_model` -- the value collected for a setting
in a category the user has not opened:

* a combo whose value is empty falls back to its first entry, and a combo
  with no entries to ``""``;
* an "auto" control reads ``auto`` for no value, and for a value that is not
  a number;
* a list control reads ``None`` for no value, the parsed literal for a list,
  and the text itself for something that is not a Python literal.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import (  # noqa: E402
    AUTO_TEXT,
    _value_a_plain_control_holds,
)

pytestmark = pytest.mark.qt


def _plan(control, value, items=()):
    return {"control": control, "value": value, "items": list(items)}


def test_an_empty_combo_value_falls_back_to_the_first_entry():
    items = [("Otsu", "otsu"), ("Li", "li")]

    assert _value_a_plain_control_holds(_plan("combo", "", items)) == "otsu"
    assert _value_a_plain_control_holds(_plan("combo", None, [])) == ""


@pytest.mark.parametrize("value", [None, "Auto", "not a number"])
def test_an_auto_control_without_a_number_reads_auto(value):
    assert _value_a_plain_control_holds(_plan("auto", value)) == AUTO_TEXT


class _Unparsable:
    def __repr__(self):
        return "<a value from elsewhere>"


@pytest.mark.parametrize("value, expected", [
    (None, None),
    ([1, 2, 3], [1, 2, 3]),
    (_Unparsable(), "<a value from elsewhere>"),
])
def test_a_list_control_reads_back_the_literal_or_the_text(value, expected):
    assert _value_a_plain_control_holds(_plan("list", value)) == expected
