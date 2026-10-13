"""A figure with no panel is left exactly as it was.

`attach` hangs the title, caption, data and groups on the figure because the
export sees nothing else. A caller that has a figure but no panel -- a raw
matplotlib figure handed through the same queue -- must not have a label or
private attributes invented for it, because an empty title would then be
exported as if the panel had chosen it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from spacr.figures.sheet import attach                          # noqa: E402


def test_attaching_no_panel_adds_nothing_to_the_figure():
    """The figure keeps the label it already had and gains no private
    spaCR attributes, so the export can still tell 'no panel' apart from
    'a panel with an empty title'."""
    figure = plt.figure()
    figure.set_label("untouched")
    try:
        assert attach(figure, None) is None
        assert figure.get_label() == "untouched"
        for name in ("_spacr_title", "_spacr_caption", "_spacr_data",
                     "_spacr_groups"):
            assert not hasattr(figure, name), name
    finally:
        plt.close(figure)
