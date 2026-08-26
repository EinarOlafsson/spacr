"""The palette chosen in Preferences has to reach a drawn figure.

Instruction 118's acceptance line is "changing the palette in Preferences
changes the next run's figures without any per-figure work". Every other
general setting travelled by being an entry in
:func:`spacr.figure_style.rc_params`; the palette was applied only by
:func:`spacr.figure_style.apply`, which nothing in the package calls, so the
one supported draw path -- :func:`spacr.figures.style.figure_style`, which
builds its overrides by diffing two ``rc_params`` dicts -- never saw it.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")


def _cycle():
    import matplotlib.pyplot as plt

    return list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def test_a_chosen_palette_colours_the_next_figure(qapp):
    """A palette set in Preferences is the colour cycle a figure is drawn in."""
    from spacr.figure_style import palette_colours
    from spacr.figures.style import figure_style
    from spacr.qt.preferences import set_figure_style

    set_figure_style({"palette": "viridis"})
    outside = _cycle()
    with figure_style("screen"):
        inside = _cycle()

    assert inside != outside, (
        "the palette preference did not reach the figure: the colour cycle "
        f"inside the house style is still {outside[:3]}")
    assert inside == palette_colours("viridis")


def test_a_per_graph_palette_beats_the_general_one(qapp):
    """A graph kind's own palette overrides the general choice for that kind."""
    from spacr.figure_style import palette_colours
    from spacr.figures.style import figure_style
    from spacr.qt.preferences import set_figure_style, set_figure_style_per_graph

    set_figure_style({"palette": "viridis"})
    set_figure_style_per_graph({"volcano": {"palette": "muted"}})

    with figure_style("screen", kind="volcano"):
        volcano = _cycle()
    with figure_style("screen", kind="histogram"):
        histogram = _cycle()

    assert volcano == palette_colours("muted")
    assert histogram == palette_colours("viridis")


def test_an_untouched_install_does_not_force_a_cycle(qapp):
    """With no preference stored, nothing overrides Matplotlib's own cycle.

    The stores hold DELTAS, so an untouched install must contribute no
    override at all -- a house style that pinned the cycle regardless would
    silently restyle every figure drawn by a caller that never asked.
    """
    from spacr.figures.style import figure_style, user_overrides

    assert "axes.prop_cycle" not in user_overrides()
    before = _cycle()
    with figure_style("screen"):
        assert _cycle() == before
