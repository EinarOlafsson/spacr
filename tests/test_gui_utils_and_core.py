"""What the settings panel makes of a settings dict.

WHAT THIS FILE USED TO BE: the pure helpers of `spacr.gui_utils` and
`spacr.gui_core`, the Tkinter interface's two support modules. That
interface is gone and so are they, and with them `parse_list`,
`convert_to_number`, `attach_dependency_listeners`, `hide_all_settings`,
`generate_annotate_fields` and the legacy console -- every one of which
existed to drive a Tk widget and has no definition left in the tree.

`convert_settings_dict_for_gui` outlived them. It lives in
`spacr.settings_spec`, `gui_utils` only re-exported it, and it is what
decides whether a setting reaches the panel as a combo, a check or an
entry. These are the only tests holding the combo option lists to what
the pipeline actually accepts -- including the regression that
`organelle_method` must not offer 'stardist', which is the no-TensorFlow
rule -- so they stay, pointed at the real module.
"""
from __future__ import annotations

import spacr.settings_spec as GU


def test_convert_settings_dict_bool_becomes_check():
    out = GU.convert_settings_dict_for_gui({"verbose": True, "plot": False})
    assert out["verbose"][0] == "check"
    assert out["plot"][0] == "check"
    assert out["verbose"][2] is True
    assert out["plot"][2] is False


def test_convert_settings_dict_int_and_float_become_entry():
    out = GU.convert_settings_dict_for_gui({"epochs": 10, "lr": 0.001})
    assert out["epochs"][0] == "entry"
    assert out["lr"][0] == "entry"


def test_convert_settings_dict_string_becomes_entry():
    out = GU.convert_settings_dict_for_gui({"src": "/tmp/x"})
    assert out["src"][0] == "entry"
    assert out["src"][2] == "/tmp/x"


def test_convert_settings_dict_none_becomes_entry():
    out = GU.convert_settings_dict_for_gui({"custom_regex": None})
    assert out["custom_regex"][0] == "entry"


def test_convert_settings_dict_list_becomes_entry_string():
    out = GU.convert_settings_dict_for_gui({"channels": [0, 1, 2]})
    assert out["channels"][0] in ("entry", "combo")


def test_convert_settings_dict_special_case_metadata_type_is_combo():
    out = GU.convert_settings_dict_for_gui({"metadata_type": "cellvoyager"})
    kind, options, default = out["metadata_type"]
    assert kind == "combo"
    assert "cellvoyager" in options
    assert "cq1" in options


def test_convert_settings_dict_special_case_organelle_method_no_stardist():
    """Regression: the organelle_method combo must not offer 'stardist'
    (removed as part of the no-TensorFlow rule)."""
    out = GU.convert_settings_dict_for_gui({"organelle_method": "otsu"})
    kind, options, default = out["organelle_method"]
    assert "stardist" not in options


# ---------------------------------------------------------------------------
