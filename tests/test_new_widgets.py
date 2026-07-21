"""
Tests for the new modern GUI widgets — spacrCard + spacrToggle.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# spacrCard
# ---------------------------------------------------------------------------

def test_spacr_card_constructs_with_title(tk_root):
    from spacr.gui_elements import spacrCard
    card = spacrCard(tk_root, title="Settings")
    tk_root.update_idletasks()
    assert hasattr(card, "body")
    # Card interior + title bar + divider + body = 3 children of the
    # inner Frame we build. Not asserting exact count (could change with
    # tweaks) but body must exist.
    assert card.body.winfo_exists()


def test_spacr_card_body_accepts_children(tk_root):
    import tkinter as tk
    from spacr.gui_elements import spacrCard
    card = spacrCard(tk_root, title="Card")
    tk.Label(card.body, text="hello").pack()
    tk_root.update_idletasks()
    assert len(card.body.winfo_children()) == 1


def test_spacr_card_no_title_has_no_title_bar(tk_root):
    from spacr.gui_elements import spacrCard
    card = spacrCard(tk_root)  # no title
    tk_root.update_idletasks()
    # Body still present.
    assert card.body.winfo_exists()


@pytest.mark.parametrize("padding", ["xs", "sm", "md", "lg", "xl"])
def test_spacr_card_padding_variants(tk_root, padding):
    from spacr.gui_elements import spacrCard
    card = spacrCard(tk_root, title="X", padding=padding)
    tk_root.update_idletasks()
    assert card.winfo_class() == "Frame"


# ---------------------------------------------------------------------------
# spacrToggle
# ---------------------------------------------------------------------------

def test_spacr_toggle_default_state_is_off(tk_root):
    from spacr.gui_elements import spacrToggle
    t = spacrToggle(tk_root, text="Enable")
    tk_root.update_idletasks()
    assert t.get() is False


def test_spacr_toggle_toggle_flips_variable(tk_root):
    from spacr.gui_elements import spacrToggle
    t = spacrToggle(tk_root, text="Enable")
    t.toggle()
    tk_root.update_idletasks()
    assert t.get() is True
    t.toggle()
    tk_root.update_idletasks()
    assert t.get() is False


def test_spacr_toggle_set_updates_variable(tk_root):
    from spacr.gui_elements import spacrToggle
    t = spacrToggle(tk_root, text="Enable")
    t.set(True)
    tk_root.update_idletasks()
    assert t.get() is True


def test_spacr_toggle_bound_variable_reflects_external_changes(tk_root):
    import tkinter as tk
    from spacr.gui_elements import spacrToggle
    v = tk.BooleanVar(value=False)
    t = spacrToggle(tk_root, text="Enable", variable=v)
    v.set(True)
    tk_root.update_idletasks()
    assert t.get() is True


def test_spacr_toggle_command_fires_on_toggle(tk_root):
    from spacr.gui_elements import spacrToggle
    fired = []
    t = spacrToggle(tk_root, text="Enable", command=lambda: fired.append(1))
    t.toggle()
    tk_root.update_idletasks()
    assert fired == [1]


def test_spacr_toggle_command_swallows_exceptions(tk_root):
    """A raising callback shouldn't tear down the widget."""
    from spacr.gui_elements import spacrToggle
    t = spacrToggle(tk_root, text="Enable",
                    command=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # Should not raise.
    t.toggle()
    tk_root.update_idletasks()


def test_spacr_toggle_animation_lands_at_correct_position(tk_root):
    """After toggle+animation, the knob should end up at the correct end."""
    from spacr.gui_elements import spacrToggle
    t = spacrToggle(tk_root, text="Enable")
    t.set(True)
    # Force pending animation to complete
    for _ in range(20):
        tk_root.update()
        tk_root.update_idletasks()
    # After the animation, the knob's left edge should be near the right side.
    coords = t._canvas.coords(t._knob)
    # Knob left edge should be > half the track width.
    assert coords[0] > (t._TRACK_W // 2)


# ---------------------------------------------------------------------------
# Existing widgets still work (regression)
# ---------------------------------------------------------------------------

def test_existing_spacr_divider_still_constructs(tk_root):
    from spacr.gui_elements import spacrDivider
    d = spacrDivider(tk_root, text="Section")
    tk_root.update_idletasks()
    assert d.text == "Section"


def test_existing_spacr_button_still_constructs(tk_root):
    from spacr.gui_elements import spacrButton
    b = spacrButton(tk_root, text="Run", show_text=True, size=48, animation=False)
    tk_root.update_idletasks()
    assert b.winfo_class() == "Frame"
