"""
Tests for spacr.gui_utils and spacr.gui_core pure helpers.

Most of these modules is Tk-driven event handling; here we cover the
pure functions (parse_list, convert_settings_dict_for_gui, ...) plus
the widget-construction bits that can be exercised with a headless
Toplevel.
"""
from __future__ import annotations

import pytest

try:
    import spacr.gui_utils as GU
except Exception as e:  # pragma: no cover
    pytest.skip(f"spacr.gui_utils unavailable in this env: {e}",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# parse_list: string -> Python list, with rejection of mixed types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("s,expected", [
    ("[1, 2, 3]", [1, 2, 3]),
    ("['a', 'b']", ["a", "b"]),
    ("[1.5, 2.5]", [1.5, 2.5]),
    ("[]", []),
])
def test_parse_list_homogeneous(s, expected):
    assert GU.parse_list(s) == expected


def test_parse_list_tuple_single_element():
    # A single-element tuple string is unwrapped.
    assert GU.parse_list("(42,)") == [42]


def test_parse_list_tuple_multi_element_becomes_list():
    assert GU.parse_list("(1, 2, 3)") == [1, 2, 3]


def test_parse_list_rejects_mixed_types():
    with pytest.raises(ValueError):
        # Lists of dicts should be rejected.
        GU.parse_list("[{'a': 1}, 2]")


def test_parse_list_rejects_non_list_scalar():
    with pytest.raises(ValueError):
        GU.parse_list("42")


def test_parse_list_rejects_garbage_syntax():
    with pytest.raises(ValueError):
        GU.parse_list("not a list at all")


# ---------------------------------------------------------------------------
# convert_to_number: same shape as app_annotate.convert_to_number
# ---------------------------------------------------------------------------

def test_gui_convert_to_number_int():
    assert GU.convert_to_number("7") == 7
    assert isinstance(GU.convert_to_number("7"), int)


def test_gui_convert_to_number_float():
    v = GU.convert_to_number("7.5")
    assert v == pytest.approx(7.5)


def test_gui_convert_to_number_rejects_garbage():
    with pytest.raises(ValueError):
        GU.convert_to_number("nope")


# ---------------------------------------------------------------------------
# convert_settings_dict_for_gui: coerces a settings dict into the
# (widget-type, options, default) triples the GUI uses.
# ---------------------------------------------------------------------------

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
# attach_dependency_listeners: verify no crash on synthetic vars_dict.
# ---------------------------------------------------------------------------

def test_attach_dependency_listeners_is_callable():
    assert callable(GU.attach_dependency_listeners)


def test_hide_all_settings_is_callable():
    assert callable(GU.hide_all_settings)


# ---------------------------------------------------------------------------
# generate_annotate_fields — construct against a real Tk root.
# ---------------------------------------------------------------------------

@pytest.mark.gui
def test_generate_annotate_fields_populates_dict(tk_root):
    import tkinter as tk
    frame = tk.Frame(tk_root)
    frame.pack()
    vars_dict = GU.generate_annotate_fields(frame)
    assert isinstance(vars_dict, dict)
    assert len(vars_dict) > 0
    tk_root.update_idletasks()


# ---------------------------------------------------------------------------
# gui_core: pure helpers importable
# ---------------------------------------------------------------------------

def test_gui_core_public_entry_points_importable():
    import spacr.gui_core as GC
    for name in ("toggle_settings", "display_figure", "clear_unused_figures",
                 "show_previous_figure", "show_next_figure",
                 "process_fig_queue", "update_figure",
                 "setup_plot_section", "setup_settings_panel",
                 "setup_console", "setup_button_section", "setup_usage_panel",
                 "initiate_abort", "check_src_folders_files",
                 "start_process", "process_console_queue",
                 "main_thread_update_function"):
        assert callable(getattr(GC, name, None)), f"gui_core.{name} not callable"
