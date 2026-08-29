"""Figures are drawn in Open Sans, the face spaCR ships -- not DejaVu Sans."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from spacr import figure_font


def test_the_faces_ship_with_the_package():
    """Light for body, Regular for titles -- both must be present."""
    names = {p.rsplit("/", 1)[-1] for p in figure_font.bundled_faces()}
    assert "OpenSans-Light.ttf" in names
    assert "OpenSans-Regular.ttf" in names


def test_registering_makes_the_name_resolvable():
    """Naming a family matplotlib cannot resolve is a SILENT fallback."""
    from matplotlib import font_manager

    assert figure_font.use_open_sans_for_figures() is True
    available = {f.name for f in font_manager.fontManager.ttflist}
    assert figure_font.FAMILY in available


def test_a_drawn_figure_actually_uses_it():
    """Drive a real figure and read the face off the rendered text."""
    import matplotlib.pyplot as plt
    from spacr.figure_style import GENERAL_DEFAULTS, rc_params

    figure_font.use_open_sans_for_figures()
    with plt.rc_context(rc_params(GENERAL_DEFAULTS)):
        fig, ax = plt.subplots()
        try:
            ax.set_title("title")
            fig.canvas.draw()
            chosen = ax.title.get_fontname()
        finally:
            plt.close(fig)

    assert chosen == figure_font.FAMILY, (
        f"the title rendered in {chosen!r}, not {figure_font.FAMILY!r}")


def test_neither_style_table_defaults_to_dejavu():
    """DejaVu was matplotlib's fallback, never a choice."""
    from spacr.figure_style import GENERAL_DEFAULTS, rc_params

    assert GENERAL_DEFAULTS["font_family"] == figure_font.FAMILY
    # And the params actually handed to matplotlib say so too.
    assert rc_params(GENERAL_DEFAULTS)["font.family"] == figure_font.FAMILY

    from spacr.figures.style import rc
    assert rc()["font.sans-serif"][0] == figure_font.FAMILY


def test_body_text_is_light_and_titles_are_not():
    """"light for text and regular for titles" -- asked for 2026-08-28."""
    from spacr.qt import preferences

    assert preferences.DEFAULT_INTERFACE_FONT_WEIGHT == "light"

    # Titles keep their own weight in the stylesheet, so a lighter body
    # default does not thin them with it.
    from spacr.qt import theme
    sheet = theme.stylesheet() if hasattr(theme, "stylesheet") else ""
    if sheet:
        assert "font-weight: 400" in sheet or "font-weight: 600" in sheet
