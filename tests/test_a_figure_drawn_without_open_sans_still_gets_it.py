"""A figure drawn where Open Sans is NOT installed still comes out in it.

Item 291. The half that is easy to fake is the rcParam: a style can say
"Open Sans" and matplotlib, having no file of that name, will quietly draw in
DejaVu Sans instead -- a `findfont` line in the log is the only complaint. A
test asserting ``rcParams["font.family"] == "Open Sans"`` passes in exactly
that broken state, so every assertion here is on the font FILE matplotlib
resolved, and every one of them runs on a font manager with every Open Sans
on this machine taken out of it.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib import font_manager  # noqa: E402

import spacr  # noqa: E402
from spacr import figure_font  # noqa: E402
from spacr.style_base import (FONT_FAMILY, FigureStyle,  # noqa: E402
                              apply_page, font_rc)

#: Worked out from the package, not from the code under test, so a helper that
#: started pointing somewhere else cannot take this with it.
BUNDLED = os.path.realpath(os.path.join(
    os.path.dirname(os.path.abspath(spacr.__file__)),
    "resources", "font", "open_sans", "static"))


def _resolved(text) -> str:
    """The font file matplotlib would actually draw ``text`` with."""
    return os.path.realpath(os.fspath(
        font_manager.findfont(text.get_fontproperties())))


def _is_bundled(path: str) -> bool:
    """Whether a resolved font file is one of the package's own."""
    return os.path.commonpath([BUNDLED, path]) == BUNDLED


@pytest.fixture
def without_open_sans(monkeypatch):
    """Matplotlib as it is on a machine where Open Sans was never installed.

    Removes every Open Sans the machine has -- including any the package has
    already registered into this process -- and resets the registration latch,
    so the code under test has to do the work again from the shipped files.
    """
    manager = font_manager.fontManager
    installed = list(manager.ttflist)
    manager.ttflist = [entry for entry in installed
                       if "open sans" not in str(entry.name).lower()]
    manager._findfont_cached.cache_clear()
    monkeypatch.setattr(figure_font, "_registered", False)
    monkeypatch.setattr(figure_font, "_resolved", False)
    try:
        yield manager
    finally:
        manager.ttflist = installed
        manager._findfont_cached.cache_clear()


def test_the_bare_machine_really_is_bare(without_open_sans):
    """The control. Without the fix, this is the figure the item complains of.

    If this ever goes green-by-accident -- resolving to Open Sans with nothing
    registered -- then every other test in this file is proving nothing, so it
    is asserted rather than assumed.
    """
    assert not any("open sans" in str(e.name).lower()
                   for e in without_open_sans.ttflist)
    found = os.path.realpath(os.fspath(font_manager.findfont(
        font_manager.FontProperties(family=[FONT_FAMILY]))))
    assert not _is_bundled(found)
    assert os.path.basename(found).startswith("DejaVuSans"), found


def test_the_style_resolves_to_the_file_the_package_ships(without_open_sans):
    """`font_rc` names a family matplotlib can find: it registered it."""
    with plt.rc_context(font_rc(FigureStyle())):
        figure, axes = plt.subplots()
        try:
            axes.set_title("a title")
            figure.canvas.draw()
            found = _resolved(axes.title)
        finally:
            plt.close(figure)

    assert _is_bundled(found), (
        f"the title resolved to {found}, which is not one of the faces "
        f"spaCR ships in {BUNDLED}")
    assert os.path.basename(found).startswith("OpenSans-"), found


def test_apply_page_puts_the_shipped_face_on_text_it_did_not_draw(
        without_open_sans):
    """Drawn OUTSIDE any font context -- the labels still come out right."""
    figure, axes = plt.subplots()
    try:
        axes.plot([0, 1], [0, 1])
        axes.set_ylabel("drawn by the renderer, not by the style")
        axes.set_xticks([0.0, 0.5, 1.0])
        before = _resolved(axes.yaxis.label)
        apply_page(figure, axes,
                   FigureStyle(title="t", x_label="x", y_label="y"))
        figure.canvas.draw()
        after = {
            "title": _resolved(axes.title),
            "x label": _resolved(axes.xaxis.label),
            "y label": _resolved(axes.yaxis.label),
            "tick": _resolved(axes.get_xticklabels()[0]),
        }
    finally:
        plt.close(figure)

    assert not _is_bundled(before), (
        "the axes was already in the shipped face before apply_page ran, so "
        "this proves nothing")
    for what, found in after.items():
        assert _is_bundled(found), f"the {what} resolved to {found}"


def test_a_chosen_family_the_machine_lacks_lands_on_the_house_face(
        without_open_sans):
    """Not on DejaVu Sans. A missing choice falls back to what spaCR ships."""
    style = FigureStyle(font_family="A Face Nobody Has")
    with plt.rc_context(font_rc(style)):
        figure, axes = plt.subplots()
        try:
            axes.set_title("a title")
            figure.canvas.draw()
            found = _resolved(axes.title)
        finally:
            plt.close(figure)
    assert _is_bundled(found), found


@pytest.mark.parametrize("chosen", ["", "A Face Nobody Has"])
def test_the_volcano_renders_in_the_shipped_face(without_open_sans, chosen):
    """The whole renderer, end to end, on the machine without the font.

    Both ways in: the default, and a family the user picked that this machine
    does not have. The second is the one that separates a renderer naming
    ``style.font_family`` straight into ``rc_context`` from one that goes
    through ``font_rc`` -- the first lands such a choice on DejaVu Sans.
    """
    from spacr.volcano_style import VolcanoStyle, render_volcano

    results = pd.DataFrame({
        "standardized_marginal_effect": [0.1, -0.3, 1.2, 0.8],
        "adjusted_p_value": [0.01, 0.5, 0.001, 0.2],
        "guide": ["a", "b", "c", "d"],
    })
    style = VolcanoStyle(title="a volcano")
    if chosen:
        style.font_family = chosen
    figure, axes = render_volcano(results, style)
    try:
        panel = axes[0] if isinstance(axes, (list, tuple)) else axes
        found = {
            "title": _resolved(panel.title),
            "x label": _resolved(panel.xaxis.label),
        }
    finally:
        plt.close(figure)
    for what, path in found.items():
        assert _is_bundled(path), (
            f"the volcano's {what} resolved to {path}, not to a face spaCR "
            f"ships")


def test_nothing_in_the_log_says_the_family_was_not_found(without_open_sans,
                                                          caplog):
    """The silent fallback is not silent to matplotlib: it says so in the log.

    An empty log is the same statement as the assertions above, made from the
    other side -- matplotlib was never asked for a font it did not have.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        with plt.rc_context(font_rc(FigureStyle())):
            figure, axes = plt.subplots()
            try:
                axes.set_title("a title")
                axes.set_xlabel("x")
                figure.canvas.draw()
            finally:
                plt.close(figure)

    missing = [r.getMessage() for r in caplog.records
               if "not found" in r.getMessage()]
    assert missing == [], missing


def test_the_defaults_do_not_name_dejavu():
    """DejaVu Sans was matplotlib's fallback, never a choice spaCR made."""
    from spacr.volcano_style import FONT_FAMILIES

    assert FigureStyle().font_family == figure_font.FAMILY
    assert font_rc(FigureStyle())["font.family"][0] == figure_font.FAMILY
    assert "DejaVu Sans" not in font_rc(FigureStyle())["font.family"]

    # The explorer's combo rewrites any value it does not carry to its first
    # entry, so a default missing from this tuple would not survive the panel
    # being opened.
    assert FONT_FAMILIES[0] == figure_font.FAMILY
    assert FigureStyle().font_family in FONT_FAMILIES
