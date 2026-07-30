"""
Coverage for the modern GUI primitives in ``spacr.gui_elements``:
``spacrToolTip``, ``spacrCard``, ``spacrToggle`` and ``spacrDivider``.

Everything here runs against a real (hidden) Tk root — the widgets are
plain Tk, so there is no way to build them without one — but no test ever
enters a main loop. The tooltip Toplevel is created and destroyed inside
the test, the toggle animation is driven by pumping ``update()`` until the
scheduled ``after`` chain drains, and the "no font loader" / "raising
after_cancel" defensive branches are reached by injecting a stub style
dict / a raising handle rather than by pinning behaviour we cannot see.
"""
from __future__ import annotations

import time
import types

import pytest

# The whole file needs a display, exactly like tests/test_gui_elements.py.
pytestmark = pytest.mark.gui

try:
    import tkinter as tk
    import spacr.gui_elements as ge
except Exception as e:  # pragma: no cover - env without a usable Tk/X
    pytest.skip(f"spacr.gui_elements unavailable in this env: {e}",
                allow_module_level=True)


@pytest.fixture(autouse=True)
def _no_leaking_figures():
    """Never let a stray matplotlib figure survive a test in this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _fake_style(**overrides):
    """A minimal style_out dict with NO font_loader.

    Used to drive the ``font_loader is None`` fallbacks in spacrCard /
    spacrToggle / spacrDivider, which the real ``set_dark_style`` never
    produces because it always loads OpenSans.
    """
    base = {
        "font_loader": None,
        "font_family": "DejaVu Sans",
        "font_size": 12,
        "font_sizes": {"small": 11, "body": 12, "header": 14, "title": 18},
        "bg_color": "#000000",
        "fg_color": "#ffffff",
        "active_color": "#007BFF",
        "inactive_color": "#2B2B2B",
        "border_color": "#2B2B2B",
        "muted_color": "#8b949e",
        "spacing": {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24},
    }
    base.update(overrides)
    return base


def _evt(x_root=100, y_root=200):
    """A duck-typed Tk event carrying only what spacrToolTip reads."""
    return types.SimpleNamespace(x_root=x_root, y_root=y_root)


# ===========================================================================
# spacrToolTip
# ===========================================================================

def test_tooltip_init_binds_enter_and_leave(tk_root):
    label = tk.Label(tk_root, text="hover me")
    tt = ge.spacrToolTip(label, "tip text")
    assert tt.widget is label
    assert tt.text == "tip text"
    assert tt.tooltip_window is None
    # Both handlers must actually be registered on the widget.
    assert label.bind("<Enter>") != ""
    assert label.bind("<Leave>") != ""


def test_tooltip_show_creates_offset_borderless_toplevel(tk_root):
    label = tk.Label(tk_root, text="hover me")
    tt = ge.spacrToolTip(label, "tip text")

    tt.show_tooltip(_evt(x_root=100, y_root=200))
    win = tt.tooltip_window
    assert isinstance(win, tk.Toplevel)
    assert win.winfo_exists()
    # Borderless + offset by (+20, +10) from the pointer.
    assert bool(win.wm_overrideredirect()) is True
    assert "+120+210" in win.wm_geometry()

    # A single themed label carrying the tooltip text.
    children = win.winfo_children()
    assert len(children) == 1
    lbl = children[0]
    assert lbl.cget("text") == "tip text"
    assert lbl.cget("relief") == "flat"
    assert int(str(lbl.cget("borderwidth"))) == 0
    # set_dark_style() re-coloured both the Toplevel and the label.
    style_out = ge.set_dark_style(ge.ttk.Style())
    assert str(lbl.cget("bg")).lower() == style_out["bg_color"].lower()
    assert str(lbl.cget("fg")).lower() == style_out["fg_color"].lower()

    tt.hide_tooltip(_evt())


def test_tooltip_hide_destroys_window_and_clears_handle(tk_root):
    label = tk.Label(tk_root, text="hover me")
    tt = ge.spacrToolTip(label, "bye")
    tt.show_tooltip(_evt())
    win = tt.tooltip_window
    assert win.winfo_exists()

    tt.hide_tooltip(_evt())
    assert tt.tooltip_window is None
    # The Toplevel is really gone, not just dereferenced.
    assert not win.winfo_exists()


def test_tooltip_hide_without_a_window_is_a_noop(tk_root):
    """The ``if self.tooltip_window`` guard: hide before any show."""
    label = tk.Label(tk_root, text="hover me")
    tt = ge.spacrToolTip(label, "nothing yet")
    assert tt.tooltip_window is None
    tt.hide_tooltip(_evt())  # must not raise
    assert tt.tooltip_window is None


def test_tooltip_show_twice_replaces_the_window(tk_root):
    label = tk.Label(tk_root, text="hover me")
    tt = ge.spacrToolTip(label, "tip")
    tt.show_tooltip(_evt(x_root=10, y_root=20))
    first = tt.tooltip_window
    tt.show_tooltip(_evt(x_root=300, y_root=400))
    second = tt.tooltip_window
    assert second is not first
    assert "+320+410" in second.wm_geometry()
    tt.hide_tooltip(_evt())
    assert tt.tooltip_window is None


# ===========================================================================
# spacrCard
# ===========================================================================

def test_card_without_title_has_only_a_body(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    card = ge.spacrCard(tk_root)
    tk_root.update_idletasks()

    assert str(card.cget("bg")).lower() == style_out["bg_color"].lower()
    assert int(str(card.cget("highlightthickness"))) == 0
    interior = card.winfo_children()[0]
    # No title bar, no divider -> the body is the only interior child.
    assert interior.winfo_children() == [card.body]
    assert int(str(interior.pack_info()["padx"])) == 0
    # Default padding key is 'md' -> 12 px on the body.
    assert int(str(card.body.pack_info()["padx"])) == style_out["spacing"]["md"]


def test_card_with_title_builds_title_bar_and_divider(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    card = ge.spacrCard(tk_root, title="Settings")
    tk_root.update_idletasks()

    interior = card.winfo_children()[0]
    title_bar, divider, body = interior.winfo_children()
    assert body is card.body
    # Title label text + muted colour.
    title_label = title_bar.winfo_children()[0]
    assert title_label.cget("text") == "Settings"
    assert str(title_label.cget("fg")).lower() == style_out["muted_color"].lower()
    assert title_label.cget("anchor") == "w"
    # 1-px divider drawn in the border colour.
    assert int(str(divider.cget("height"))) == 1
    assert str(divider.cget("bg")).lower() == style_out["border_color"].lower()
    # Title bar gets (pad, xs) vertical padding.
    pady = tuple(int(v) for v in title_bar.pack_info()["pady"])
    assert pady == (style_out["spacing"]["md"], style_out["spacing"]["xs"])


def test_card_show_border_lifts_the_outer_frame(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    card = ge.spacrCard(tk_root, title="Bordered", show_border=True)
    tk_root.update_idletasks()

    assert int(str(card.cget("highlightthickness"))) == 1
    assert str(card.cget("highlightbackground")).lower() == \
        style_out["border_color"].lower()
    # Outer frame takes the border colour, the interior stays bg_color.
    assert str(card.cget("bg")).lower() == style_out["border_color"].lower()
    interior = card.winfo_children()[0]
    assert str(interior.cget("bg")).lower() == style_out["bg_color"].lower()
    assert int(str(interior.pack_info()["padx"])) == 1
    assert int(str(interior.pack_info()["pady"])) == 1


@pytest.mark.parametrize("key,expected", [("xs", 4), ("sm", 8), ("md", 12),
                                          ("lg", 16), ("xl", 24)])
def test_card_padding_key_selects_the_spacing_value(tk_root, key, expected):
    card = ge.spacrCard(tk_root, title="T", padding=key)
    tk_root.update_idletasks()
    assert int(str(card.body.pack_info()["padx"])) == expected
    assert int(str(card.body.pack_info()["pady"])) == expected


def test_card_unknown_padding_key_falls_back_to_md(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    card = ge.spacrCard(tk_root, padding="definitely-not-a-key")
    tk_root.update_idletasks()
    assert int(str(card.body.pack_info()["padx"])) == style_out["spacing"]["md"]


def test_card_body_hosts_children(tk_root):
    card = ge.spacrCard(tk_root, title="Host")
    tk.Label(card.body, text="a").pack()
    tk.Label(card.body, text="b").pack()
    tk_root.update_idletasks()
    assert [w.cget("text") for w in card.body.winfo_children()] == ["a", "b"]


def test_card_without_font_loader_uses_a_family_tuple(tk_root, monkeypatch):
    """Fallback title font when the style dict carries no font_loader."""
    monkeypatch.setattr(ge, "set_dark_style", lambda *a, **k: _fake_style())
    card = ge.spacrCard(tk_root, title="Fallback")
    tk_root.update_idletasks()

    interior = card.winfo_children()[0]
    title_label = interior.winfo_children()[0].winfo_children()[0]
    font_spec = str(title_label.cget("font"))
    assert "DejaVu Sans" in font_spec
    assert "14" in font_spec        # font_sizes['header']
    assert "bold" in font_spec.lower()
    assert card.style_out["font_loader"] is None


def test_card_fallback_spacing_dict_supports_a_title(tk_root, monkeypatch):
    """A style dict with no 'spacing' key must still build a titled card.

    The class already declares a fallback scale for exactly this case, so
    reading a key that fallback does not define is a defect, not a
    documented requirement.
    """
    style = _fake_style()
    style.pop("spacing")
    monkeypatch.setattr(ge, "set_dark_style", lambda *a, **k: style)
    card = ge.spacrCard(tk_root, title="No spacing key")
    tk_root.update_idletasks()
    assert card.body.winfo_exists()


# ===========================================================================
# spacrToggle
# ===========================================================================

def test_toggle_builds_label_and_canvas_with_track_and_knob(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    t = ge.spacrToggle(tk_root, text="Enable")
    tk_root.update_idletasks()

    assert t._label is not None
    assert t._label.cget("text") == "Enable"
    assert t._label.cget("cursor") == "hand2"
    assert t.winfo_children() == [t._label, t._canvas]
    assert int(t._canvas.cget("width")) == t._TRACK_W
    assert int(t._canvas.cget("height")) == t._TRACK_H
    # Two canvas items: a smoothed polygon track and an oval knob.
    assert t._canvas.type(t._track) == "polygon"
    assert t._canvas.type(t._knob) == "oval"
    assert t._canvas.itemcget(t._track, "fill").lower() == \
        style_out["inactive_color"].lower()
    assert t._canvas.itemcget(t._knob, "fill").lower() == \
        style_out["fg_color"].lower()
    assert t.get() is False


def test_toggle_without_text_has_no_label(tk_root):
    t = ge.spacrToggle(tk_root)
    tk_root.update_idletasks()
    assert t._label is None
    assert t.winfo_children() == [t._canvas]


def test_toggle_without_font_loader_uses_a_family_tuple(tk_root, monkeypatch):
    monkeypatch.setattr(ge, "set_dark_style", lambda *a, **k: _fake_style())
    t = ge.spacrToggle(tk_root, text="Fallback")
    tk_root.update_idletasks()
    font_spec = str(t._label.cget("font"))
    assert "DejaVu Sans" in font_spec
    assert "12" in font_spec


def test_toggle_knob_bbox_and_track_colour_track_the_variable(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    t = ge.spacrToggle(tk_root, text="X")
    pad = (t._TRACK_H - t._KNOB_D) // 2
    off = (pad, pad, pad + t._KNOB_D, pad + t._KNOB_D)
    on = (t._TRACK_W - t._KNOB_D - pad, pad, t._TRACK_W - pad, pad + t._KNOB_D)

    assert t._knob_bbox() == off
    assert t._track_color().lower() == style_out["inactive_color"].lower()

    t.set(True)
    tk_root.update_idletasks()
    assert t.get() is True
    assert t._knob_bbox() == on
    assert t._track_color().lower() == style_out["active_color"].lower()
    # The write-trace re-drew the canvas without any animation.
    assert t._canvas.coords(t._knob) == [float(v) for v in on]
    assert t._canvas.itemcget(t._track, "fill").lower() == \
        style_out["active_color"].lower()

    t.set(0)  # truthiness is coerced to bool
    tk_root.update_idletasks()
    assert t.get() is False
    assert t._canvas.coords(t._knob) == [float(v) for v in off]


def test_toggle_external_variable_write_syncs_the_canvas(tk_root):
    v = tk.BooleanVar(value=False)
    t = ge.spacrToggle(tk_root, text="Bound", variable=v)
    assert t.variable is v
    v.set(True)
    tk_root.update_idletasks()
    assert t.get() is True
    assert t._canvas.coords(t._knob)[0] > t._TRACK_W // 2


def test_toggle_click_on_label_and_canvas_flips_the_variable(tk_root):
    t = ge.spacrToggle(tk_root, text="Click")
    t.pack()
    # Tk only delivers generated button events to a mapped window, so the
    # root has to come out of hiding for the duration of the clicks.
    tk_root.deiconify()
    tk_root.update()
    try:
        assert t._label.bind("<Button-1>") != ""
        assert t._canvas.bind("<Button-1>") != ""
        t._label.event_generate("<Button-1>", x=2, y=2)
        tk_root.update()
        assert t.get() is True
        t._canvas.event_generate("<Button-1>", x=2, y=2)
        tk_root.update()
        assert t.get() is False
    finally:
        tk_root.withdraw()


def test_toggle_command_fires_after_each_toggle(tk_root):
    calls = []
    t = ge.spacrToggle(tk_root, text="Cmd", command=lambda: calls.append(t.get()))
    t.toggle()
    t.toggle()
    tk_root.update_idletasks()
    assert calls == [True, False]
    assert t.get() is False


def test_toggle_swallows_a_raising_command_but_still_flips(tk_root):
    def boom():
        raise RuntimeError("callback exploded")

    t = ge.spacrToggle(tk_root, text="Boom", command=boom)
    t.toggle()  # must not propagate
    tk_root.update_idletasks()
    assert t.get() is True


def test_toggle_cancels_a_pending_animation_before_starting_a_new_one(tk_root):
    t = ge.spacrToggle(tk_root, text="Anim")
    t.toggle()
    pending = t._anim_id
    assert pending is not None

    cancelled = []
    real_cancel = t.after_cancel
    t.after_cancel = lambda i: (cancelled.append(i), real_cancel(i))[0]
    t.toggle()
    assert cancelled == [pending]
    assert t._anim_id is not None and t._anim_id != pending


def test_toggle_survives_an_after_cancel_that_raises(tk_root, monkeypatch):
    """Failure injection: a stale/invalid handle must not break toggling."""
    t = ge.spacrToggle(tk_root, text="Stale")
    t.toggle()
    assert t._anim_id is not None

    seen = []

    def _raising_cancel(handle):
        seen.append(handle)
        raise tk.TclError("no such after id")

    monkeypatch.setattr(t, "after_cancel", _raising_cancel)
    t.toggle()
    assert len(seen) == 1
    # The exception was swallowed and a fresh animation was scheduled.
    assert t._anim_id is not None
    assert t.get() is False


def test_toggle_animation_slides_the_knob_and_lands_exactly(tk_root):
    """Drive the real ``after`` chain to completion, not just the first tick."""
    t = ge.spacrToggle(tk_root, text="Slide")
    pad = (t._TRACK_H - t._KNOB_D) // 2
    on = (t._TRACK_W - t._KNOB_D - pad, pad, t._TRACK_W - pad, pad + t._KNOB_D)

    # Switch ON, then shove the knob back to the OFF end so _animate has a
    # real distance to cover instead of a zero-length slide.
    t.variable.set(True)
    t._canvas.coords(t._knob, pad, pad, pad + t._KNOB_D, pad + t._KNOB_D)
    tk_root.update_idletasks()

    t._animate()
    # First tick already ran synchronously and moved the knob one step.
    first_step = t._canvas.coords(t._knob)[0]
    assert pad < first_step < on[0]
    assert t._anim_id is not None

    deadline = time.time() + 5.0
    while t._anim_id is not None and time.time() < deadline:
        tk_root.update()
        time.sleep(0.005)
    tk_root.update()

    assert t._anim_id is None, "animation never finished"
    assert t._canvas.coords(t._knob) == [float(v) for v in on]


def test_toggle_round_rect_returns_a_smoothed_polygon(tk_root):
    t = ge.spacrToggle(tk_root)
    item = t._round_rect(0, 0, 20, 10, radius=5, fill="#123456", outline="")
    assert t._canvas.type(item) == "polygon"
    assert t._canvas.itemcget(item, "smooth") in ("1", "true", "bezier")
    assert t._canvas.itemcget(item, "fill") == "#123456"
    coords = t._canvas.coords(item)
    assert len(coords) == 24  # 12 (x, y) control points
    assert min(coords) == 0.0 and max(coords) == 20.0


# ===========================================================================
# spacrDivider
# ===========================================================================

def test_divider_plain_horizontal_rule(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    d = ge.spacrDivider(tk_root)
    tk_root.update_idletasks()

    assert d.text == ""
    assert d.orient == "horizontal"
    assert d.thickness == 1
    assert str(d.cget("bg")).lower() == style_out["bg_color"].lower()
    (rule,) = d.winfo_children()
    assert isinstance(rule, tk.Frame)
    assert int(str(rule.cget("height"))) == 1
    assert str(rule.cget("bg")).lower() == style_out["border_color"].lower()
    # pady=(sm, sm) collapses to the scalar sm in Tk's pack info.
    assert int(str(rule.pack_info()["pady"])) == style_out["spacing"]["sm"]
    assert int(str(rule.pack_info()["padx"])) == 0


def test_divider_vertical_ignores_text_and_uses_width(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    d = ge.spacrDivider(tk_root, text="ignored", orient="vertical", thickness=3)
    tk_root.update_idletasks()

    assert d.orient == "vertical"
    assert d.thickness == 3
    (rule,) = d.winfo_children()
    assert int(str(rule.cget("width"))) == 3
    assert str(rule.cget("bg")).lower() == style_out["border_color"].lower()
    assert rule.pack_info()["fill"] == "y"
    # The early return means no caption label was built.
    assert not any(isinstance(c, tk.Label) for c in d.winfo_children())


def test_divider_captioned_builds_left_rule_label_right_rule(tk_root):
    style_out = ge.set_dark_style(ge.ttk.Style())
    d = ge.spacrDivider(tk_root, text="Advanced", thickness=2)
    tk_root.update_idletasks()

    left, label, right = d.winfo_children()
    assert isinstance(left, tk.Frame) and isinstance(right, tk.Frame)
    assert isinstance(label, tk.Label)
    assert label.cget("text") == "Advanced"
    assert str(label.cget("fg")).lower() == style_out["muted_color"].lower()
    assert int(str(left.cget("width"))) == style_out["spacing"]["md"]
    assert int(str(left.cget("height"))) == 2
    assert int(str(right.cget("height"))) == 2
    # Grid layout: rule | caption | rule, with the trailing rule stretching.
    assert int(left.grid_info()["column"]) == 0
    assert int(label.grid_info()["column"]) == 1
    assert int(right.grid_info()["column"]) == 2
    assert int(d.grid_columnconfigure(2)["weight"]) == 1
    assert int(d.grid_columnconfigure(1)["weight"]) == 0


@pytest.mark.parametrize("given,expected", [(0, 1), (-5, 1), (1, 1), (4, 4),
                                            (2.9, 2)])
def test_divider_thickness_is_clamped_to_at_least_one(tk_root, given, expected):
    d = ge.spacrDivider(tk_root, thickness=given)
    assert d.thickness == expected
    (rule,) = d.winfo_children()
    assert int(str(rule.cget("height"))) == expected


def test_divider_without_font_loader_uses_a_family_tuple(tk_root, monkeypatch):
    monkeypatch.setattr(ge, "set_dark_style", lambda *a, **k: _fake_style())
    d = ge.spacrDivider(tk_root, text="Fallback")
    tk_root.update_idletasks()
    label = [c for c in d.winfo_children() if isinstance(c, tk.Label)][0]
    font_spec = str(label.cget("font"))
    assert "DejaVu Sans" in font_spec
    assert "11" in font_spec        # font_sizes['small']


def test_divider_captioned_falls_back_to_derived_small_font_size(tk_root,
                                                                 monkeypatch):
    """No 'font_sizes' key -> small size derived as max(font_size - 1, 9)."""
    style = _fake_style()
    style.pop("font_sizes")
    style["font_size"] = 9
    monkeypatch.setattr(ge, "set_dark_style", lambda *a, **k: style)
    d = ge.spacrDivider(tk_root, text="Derived")
    tk_root.update_idletasks()
    label = [c for c in d.winfo_children() if isinstance(c, tk.Label)][0]
    assert "9" in str(label.cget("font"))
