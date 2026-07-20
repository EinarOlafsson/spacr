"""
Tests for spacr.gui_elements — headless Tk widget construction + palette API.

Live GUI interaction is out of scope; we just verify every custom widget
constructs, is a valid tk widget, and pulls from the shared style dict.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

pytestmark = pytest.mark.gui  # tag: needs a display

# spacr.gui_elements imports pyautogui, which imports mouseinfo, which
# unconditionally opens an X display at IMPORT time and raises
# Xlib.error.DisplayConnectionError when the DISPLAY xauth cookie isn't
# available (common in subprocess pytest runs launched via coverage). Skip
# the whole file cleanly rather than crashing collection.
try:
    import spacr.gui_elements as ge
except Exception as e:  # pragma: no cover
    pytest.skip(f"spacr.gui_elements unavailable in this env: {e}",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# The style dict — palette / spacing / font hierarchy
# ---------------------------------------------------------------------------

def test_set_dark_style_is_cacheable(tk_root):
    """The second call with no side-effect args should return the cached
    dict, not rebuild it."""
    s1 = ge.set_dark_style(ttk.Style(), parent_frame=None)
    s2 = ge.set_dark_style(ttk.Style(), parent_frame=None)
    assert s1 is s2


def test_set_dark_style_returns_hex_colors(dark_style):
    for k in ("bg_color", "fg_color", "active_color", "inactive_color",
              "border_color", "muted_color",
              "success_color", "warning_color", "error_color"):
        v = dark_style[k]
        assert isinstance(v, str) and v.startswith("#") and len(v) == 7


def test_set_dark_style_extended_palette_matches_defaults(dark_style):
    """These are the values the palette-refresh commit locked in."""
    assert dark_style["bg_color"].lower() == "#0e1116"
    assert dark_style["fg_color"].lower() == "#e6edf3"
    assert dark_style["active_color"].lower() == "#4a90e2"
    assert dark_style["inactive_color"].lower() == "#1a1f27"
    assert dark_style["border_color"].lower() == "#2b3138"
    assert dark_style["muted_color"].lower() == "#8b949e"
    assert dark_style["success_color"].lower() == "#3fb950"
    assert dark_style["warning_color"].lower() == "#d29922"
    assert dark_style["error_color"].lower() == "#f85149"


def test_spacing_and_fonts_present(dark_style):
    assert dark_style["spacing"] == {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
    fs = dark_style["font_sizes"]
    assert fs["small"] < fs["body"] < fs["header"] < fs["title"]


def test_set_dark_style_custom_hex_pass_through(tk_root):
    """If the caller passes an explicit hex, it should be preserved verbatim
    rather than being remapped to the palette."""
    # Force cache invalidation by passing parent_frame.
    ge._cached_dark_style = None
    style = ge.set_dark_style(ttk.Style(), parent_frame=tk_root,
                              bg_color="#123456", fg_color="#abcdef")
    assert style["bg_color"] == "#123456"
    assert style["fg_color"] == "#abcdef"
    ge._cached_dark_style = None  # reset for other tests


# ---------------------------------------------------------------------------
# Widget construction — every spacr* widget builds & is a tk widget
# ---------------------------------------------------------------------------

def test_spacr_font_loader_ok(tk_root):
    f = ge.spacrFont("OpenSans", "Regular", font_size=12)
    assert f.get_font(size=12) is not None


def test_spacr_container_constructs(tk_root):
    w = ge.spacrContainer(tk_root, orient=tk.VERTICAL, bg="#000000")
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)


def test_spacr_entry_constructs(tk_root):
    v = tk.StringVar()
    w = ge.spacrEntry(tk_root, textvariable=v)
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)
    assert w.entry is not None
    v.set("hello")
    assert w.entry.get() == "hello"


def test_spacr_check_toggles_variable(tk_root):
    v = tk.BooleanVar(value=False)
    w = ge.spacrCheck(tk_root, text="on?", variable=v)
    tk_root.update_idletasks()
    assert v.get() is False
    w.toggle_variable(None)
    assert v.get() is True
    w.toggle_variable(None)
    assert v.get() is False


def test_spacr_combo_constructs(tk_root):
    v = tk.StringVar()
    w = ge.spacrCombo(tk_root, textvariable=v, values=["a", "b", "c"])
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)
    assert w.values == ["a", "b", "c"]


def test_spacr_dropdown_menu_constructs(tk_root):
    v = tk.StringVar(value="alpha")
    w = ge.spacrDropdownMenu(tk_root, variable=v, options=["alpha", "beta"])
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)


def test_spacr_progress_bar_constructs(tk_root):
    w = ge.spacrProgressBar(tk_root)
    tk_root.update_idletasks()
    assert isinstance(w, ttk.Progressbar)


def test_spacr_slider_constructs(tk_root):
    w = ge.spacrSlider(tk_root, from_=0, to=100)
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)


def test_spacr_frame_constructs(tk_root):
    w = ge.spacrFrame(tk_root)
    tk_root.update_idletasks()
    assert isinstance(w, ttk.Frame)


def test_spacr_label_constructs(tk_root):
    w = ge.spacrLabel(tk_root, text="hello")
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)


def test_spacr_button_constructs_and_fires_command(tk_root):
    called = []
    w = ge.spacrButton(tk_root, text="run", command=lambda: called.append(1),
                       size=40, show_text=True, animation=False)
    tk_root.update_idletasks()
    assert isinstance(w, tk.Frame)
    w.on_click(None)
    assert called == [1]


def test_spacr_button_fade_state(tk_root):
    """The palette-refresh feature: spacrButton exposes _fade_bg_to and
    handles rapid hover-in/hover-out without stacking timers."""
    btn = ge.spacrButton(tk_root, text="fade", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    assert hasattr(btn, "_fade_bg_to")
    # Kick off two fades back-to-back; the second should cancel the first.
    btn.on_enter()
    btn.on_leave()
    tk_root.update_idletasks()
    # Fill should still be a valid hex.
    fill = btn.canvas.itemcget(btn.button_bg, "fill")
    assert fill.startswith("#") and len(fill) == 7


def test_spacr_switch_toggles(tk_root):
    v = tk.BooleanVar(value=False)
    w = ge.spacrSwitch(tk_root, text="power", variable=v)
    tk_root.update_idletasks()
    # Directly flip via variable and re-render.
    v.set(True)
    w.update_switch()


def test_spacr_tooltip_constructs(tk_root):
    label = tk.Label(tk_root, text="hover me")
    label.pack()
    tk_root.update_idletasks()
    tt = ge.spacrToolTip(label, text="tip text")
    # Just verify no crash and object is initialized properly.
    assert tt.text == "tip text"


def test_spacr_divider_variants(tk_root):
    plain = ge.spacrDivider(tk_root)
    captioned = ge.spacrDivider(tk_root, text="Advanced")
    vertical = ge.spacrDivider(tk_root, orient="vertical")
    tk_root.update_idletasks()
    for w in (plain, captioned, vertical):
        assert isinstance(w, tk.Frame)
    assert captioned.text == "Advanced"
    assert vertical.orient == "vertical"


def test_spacr_divider_has_children_for_captioned(tk_root):
    """Captioned divider should have at least 3 children: left rule, label, right rule."""
    w = ge.spacrDivider(tk_root, text="Section")
    tk_root.update_idletasks()
    assert len(w.winfo_children()) >= 3
