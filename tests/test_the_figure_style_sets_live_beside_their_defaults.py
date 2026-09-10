"""Instruction 118's standing handoff, taken.

The closed sets a style key may take were declared in
`qt/widgets/figure_settings.py` because `spacr/figure_style.py` was another
territory, and they were read off the COMMENTS beside the defaults:

    "error_bars": "sem",       # sem | sd | ci95 | none

A comment is not a contract. A value added to the renderer would have gone on
being drawn and gone on being unofferable in the panel, and nothing would have
said the two disagreed. The sets live beside the values now, and where the
values already exist somewhere — the spine presets, the line styles — the set
is DERIVED rather than copied, so adding one there offers it here.
"""
from __future__ import annotations

import pytest

from spacr.figure_style import (GENERAL_DEFAULTS, GRAPH_DEFAULTS,
                                LINE_STYLE_CHOICES, LINE_STYLE_KEYS,
                                SPINE_PRESETS, STYLE_CHOICES, style_choices)


def _every_default() -> dict:
    out = dict(GENERAL_DEFAULTS)
    for kind in GRAPH_DEFAULTS.values():
        out.update(kind)
    return out


def test_every_closed_set_contains_the_default_it_describes():
    """A default outside its own set is a panel that cannot show the default."""
    defaults = _every_default()
    for name, choices in STYLE_CHOICES.items():
        if name in defaults:
            assert defaults[name] in choices, (
                f"{name} defaults to {defaults[name]!r}, which is not in "
                f"{choices}")


def test_the_line_style_keys_all_default_to_a_line_style():
    defaults = _every_default()
    for name in LINE_STYLE_KEYS:
        assert defaults[name] in LINE_STYLE_CHOICES


def test_the_spine_set_is_derived_from_the_presets_not_copied():
    """Add a preset and it is offered, with no second edit anywhere."""
    assert style_choices("spines") == tuple(SPINE_PRESETS)


def test_one_answer_to_what_dashes_a_line_may_take():
    """Three identical tuples is three things to keep in step."""
    for name in LINE_STYLE_KEYS:
        assert style_choices(name) == LINE_STYLE_CHOICES


def test_a_key_that_is_not_closed_says_so_rather_than_guessing():
    assert style_choices("marker_size") == ()
    assert style_choices("point_alpha") == ()
    assert style_choices("nothing_at_all") == ()


def test_the_panel_reads_the_module_rather_than_its_own_copy():
    """One source; the panel's list is a fallback for an import that failed."""
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.figure_settings import style_choices_for

    for name in ("palette", "spines", "error_bars", "bins", "aspect",
                 "colormap", "format", *LINE_STYLE_KEYS):
        assert style_choices_for(name) == style_choices(name)


def test_the_comment_that_used_to_carry_the_set_is_gone():
    """It was the source of truth and it could not be tested."""
    import inspect

    from spacr import figure_style

    source = inspect.getsource(figure_style)
    assert "# sem | sd | ci95 | none" not in source
