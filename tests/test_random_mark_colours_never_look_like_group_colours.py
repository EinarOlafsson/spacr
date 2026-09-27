"""Item 288: the "random" mark-colouring rule at its edges.

* With two colours, the seeded shuffle happens to leave them in order. A
  "random" rule that draws exactly what "group" draws would look like the
  setting did nothing, so the pair is swapped instead.
* A grouped renderer asks the user's preference for its rule. When the
  preference cannot be read it keeps the house rule, and when the palette
  resolves to no colours at all it keeps the house rule too, rather than
  drawing every group in nothing.
"""
from __future__ import annotations

import spacr.qt.preferences as preferences
from spacr import figure_style


def test_two_colours_are_swapped_rather_than_left_in_order():
    pair = ["#111111", "#222222"]
    assert figure_style._marks_coloured_by(pair, "random") == \
        ["#222222", "#111111"]
    assert figure_style._marks_coloured_by(pair, "group") == pair


def test_an_unreadable_preference_keeps_the_house_rule(monkeypatch):
    from spacr.figures.style import _group_colours

    def unreadable():
        raise OSError("settings locked")

    monkeypatch.setattr(preferences, "get_figure_style", unreadable)
    assert _group_colours(3, ["#111111", "#222222"]) is None


def test_no_colour_to_draw_with_keeps_the_house_rule(monkeypatch):
    from spacr.figures import style

    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: {"mark_colouring": "random"})
    monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                        lambda: {})
    monkeypatch.setattr(figure_style, "palette_colours", lambda name: [])
    assert style._group_colours(3, []) is None
