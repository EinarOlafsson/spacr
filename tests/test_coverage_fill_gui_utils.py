"""Coverage-fill for spacr.gui_utils pure helpers (Tk stubbed by conftest)."""
from __future__ import annotations

import pytest

from spacr import gui_utils as GU


# ---------------------------------------------------------------------------
# parse_list
# ---------------------------------------------------------------------------

def test_parse_list_list():
    assert GU.parse_list("[1, 2, 3]") == [1, 2, 3]
    assert GU.parse_list("['a', 'b']") == ["a", "b"]


def test_parse_list_tuple():
    assert GU.parse_list("(1, 2)") == [1, 2]
    assert GU.parse_list("(5,)") == [5]   # single-element tuple


def test_parse_list_mixed_raises():
    with pytest.raises(ValueError):
        GU.parse_list("[1, [2], 3]")   # nested list → mixed/unsupported


def test_parse_list_not_a_list():
    with pytest.raises(ValueError):
        GU.parse_list("42")


def test_parse_list_invalid_syntax():
    with pytest.raises(ValueError):
        GU.parse_list("[1, 2,")


# ---------------------------------------------------------------------------
# convert_to_number
# ---------------------------------------------------------------------------

def test_convert_to_number():
    assert GU.convert_to_number("42") == 42
    assert GU.convert_to_number("3.14") == 3.14
    with pytest.raises(ValueError):
        GU.convert_to_number("not a number")


# ---------------------------------------------------------------------------
# convert_settings_dict_for_gui
# ---------------------------------------------------------------------------

def test_convert_settings_dict_for_gui():
    settings = {
        "metadata_type": "cellvoyager",   # special case → combo
        "verbose": True,                  # bool → check
        "diameter": 30,                   # int → entry
        "src": "/data",                   # str → entry
        "cov_type": None,                 # None (special) → combo
        "channels_list": [0, 1, 2],       # list → entry (str)
    }
    out = GU.convert_settings_dict_for_gui(settings)
    assert out["metadata_type"][0] == "combo"
    assert out["verbose"] == ("check", None, True)
    assert out["diameter"] == ("entry", None, 30)
    assert out["src"] == ("entry", None, "/data")
    assert out["channels_list"][0] == "entry"
    assert out["cov_type"][0] == "combo"
