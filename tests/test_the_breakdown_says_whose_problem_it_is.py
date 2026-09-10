"""The verdict a confusion breakdown spells out, and the theme it draws in.

``describe_breakdown``'s docstring states its purpose plainly: "The point of
this line is to stop wasted work. If all 43 errors come from well A01,
re-labelling any of them corrects nothing that will recur." The two verdicts
send the user to two different places -- the bench or the model -- so which one
prints is the whole value of the function.
"""
from __future__ import annotations

import builtins

import pandas as pd
import pytest


def _rows(wells):
    return pd.DataFrame({"well": wells, "object_label": range(len(wells))})


def test_errors_concentrated_in_one_well_are_called_a_bench_problem():
    """The concentration verdict, above both the share and the count floor.

    This is the line that stops a re-annotation queue being opened for a
    staining problem.
    """
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows(["A01"] * 9 + ["B02"]), "well")

    assert "single well" in text
    assert "staining, focus, seeding" in text
    assert "Fix it upstream" in text


def test_errors_spread_over_many_wells_are_called_the_models():
    """The other verdict, which sends the user to the model instead."""
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows([f"A{i:02d}" for i in range(10)]), "well")

    assert "the model's, not one well's" in text
    assert "staining" not in text


def test_too_few_objects_get_no_verdict_at_all():
    """The count floor: five objects cannot establish a pattern.

    A share of 100% over three objects is not concentration, and printing the
    bench verdict there would send someone to the microscope over noise.
    """
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows(["A01", "A01", "A01"]), "well")

    assert "Fix it upstream" not in text
    assert "3 object(s)" in text


def test_one_well_below_the_floor_still_gets_no_spread_verdict():
    """The ``elif len(table) > 1``: one group is not a spread either.

    Neither verdict is the honest answer for a cell with too little in it, and
    the count line still tells the reader what there was.
    """
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows(["A01", "A01"]), "well")

    assert "not one well's" not in text
    assert "Fix it upstream" not in text


def test_an_empty_cell_says_there_is_nothing_to_break_down():
    """The early return, which names the level so the message is specific."""
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows([]), "well")

    assert "nothing to break down by well" in text


def test_the_worst_group_is_named_with_its_share():
    """The first line, which every verdict is appended to."""
    from spacr.confusion import describe_breakdown

    text = describe_breakdown(_rows(["A01"] * 3 + ["B02"]), "well")

    assert "worst is A01 with 3 (75%)" in text


# ---------------------------------------------------------------------------
# figure_style.theme_ink — the colours a saved figure is drawn in
# ---------------------------------------------------------------------------

def test_no_preference_store_falls_back_to_the_print_pair(monkeypatch):
    """The import guard: headless runs have no Qt preferences.

    Every batch job takes this path, so the print pair is the usual answer
    rather than the exceptional one -- and a figure has to be drawn in
    something.
    """
    from spacr import figure_style

    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        # `from .qt.preferences import x` arrives as name='qt.preferences'
        # with level=1, so the module is in `name` and not in `fromlist`.
        if "preferences" in str(name):
            raise ImportError("no Qt preferences here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)

    assert figure_style.theme_ink() == (figure_style.PRINT_INK,
                                        figure_style.PRINT_GRID)


def test_a_theme_that_cannot_be_resolved_falls_back_too(monkeypatch):
    """The second guard: preferences import but will not answer."""
    from spacr import figure_style
    from spacr.qt import preferences

    def refuse():
        raise RuntimeError("the preference store is unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", refuse)

    assert figure_style.theme_ink() == (figure_style.PRINT_INK,
                                        figure_style.PRINT_GRID)


@pytest.mark.parametrize("theme, dark", [
    ("light", False), ("LIGHT", False), ("  light  ", False),
    ("dark", True), ("space", True), ("cell", True), ("", True), (None, True),
])
def test_every_theme_but_light_is_drawn_dark(monkeypatch, theme, dark):
    """The comparison the code's own comment insists on.

    "compare against 'light' and treat everything else as dark, because Space
    and Cell are dark themes and 'system' has already been resolved by the
    time it answers." Listing the dark ones instead would leave a theme added
    later drawn in print ink on a dark ground.
    """
    from spacr import figure_style
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: theme)

    expected = ((figure_style.DARK_INK, figure_style.DARK_GRID) if dark
                else (figure_style.PRINT_INK, figure_style.PRINT_GRID))
    assert figure_style.theme_ink() == expected
