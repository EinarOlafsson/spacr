"""One window with every Image UMAP display setting in it.

Settled by the user: "the other settings can also be in the same settings
window even though they cannot be live applied." So the window holds both
halves and says which is which, rather than offering only the live ones.

The trap this is built around: the explorer draws thumbnails at embedding
coordinates, so a "live apply" that re-embeds moves every point and the
user loses the arrangement they were reading.
"""

from __future__ import annotations

import numpy as np
import pytest


def _payload(n: int = 6) -> dict:
    return {
        "embedding": np.column_stack([np.linspace(0, 1, n),
                                      np.linspace(1, 0, n)]),
        "labels": np.arange(n) % 3,
        "records": [{"image": None, "display_name": str(i), "db_path": None,
                     "db_png_path": None, "prcfo": f"p_A_1_f1_o{i}"}
                    for i in range(n)],
        "display": {"point_size": 26, "point_color": "cluster",
                    "point_alpha": 0.65, "outline_width": 1.0,
                    "canvas_width": 900, "sidebar_width": 280},
    }


@pytest.fixture()
def explorer(qtbot):
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)
    widget.set_payload(_payload())
    return widget


# ---------------------------------------------------------------------------
# Live apply
# ---------------------------------------------------------------------------

def test_a_live_setting_reaches_the_current_figure(explorer):
    assert explorer.apply_display({"point_size": 90})
    assert explorer.display_settings()["point_size"] == 90


def test_live_apply_never_moves_a_point(explorer):
    """The whole reason this is delicate.

    A restyle that re-embeds changes every neighbour relationship the user
    was reading. The redraw must come from the SAME embedding.
    """
    before = explorer._embedding.copy()
    explorer.apply_display({"point_size": 120, "point_alpha": 0.2,
                            "point_color": "red"})
    assert np.array_equal(before, explorer._embedding)


def test_an_unchanged_value_does_not_redraw(explorer):
    """Opening the window and pressing OK must not cost a redraw."""
    current = explorer.display_settings()
    assert not explorer.apply_display(dict(current))


def test_a_non_live_setting_is_stored_but_reports_no_redraw(explorer):
    """`figuresize` decides what gets drawn, so it needs the run that draws.

    Storing it and returning False is what lets the caller say "takes
    effect on the next run" instead of implying it already did.
    """
    assert not explorer.apply_display({"figuresize": 12.0})


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------

def test_the_window_holds_both_halves(qtbot, explorer):
    from spacr.qt.widgets.umap_explorer import UmapDisplaySettings

    dialog = UmapDisplaySettings(explorer.display_settings())
    qtbot.addWidget(dialog)

    values = dialog.values()
    assert "point_size" in values, "a live setting is missing"
    assert "figuresize" in values, (
        "the non-live settings were left out; the user asked for one window")


def test_only_the_live_half_is_offered_for_live_apply(qtbot, explorer):
    from spacr.qt.widgets.umap_explorer import UmapDisplaySettings

    dialog = UmapDisplaySettings(explorer.display_settings())
    qtbot.addWidget(dialog)

    live = dialog.live_values()
    assert "point_size" in live
    assert "figuresize" not in live, (
        "figuresize cannot be applied to a figure that is already drawn")


def test_every_live_field_is_one_the_explorer_knows(qtbot, explorer):
    """Or the window offers a control that silently does nothing."""
    from spacr.qt.widgets.umap_explorer import (ImageUmapExplorer,
                                                UmapDisplaySettings)

    dialog = UmapDisplaySettings(explorer.display_settings())
    qtbot.addWidget(dialog)
    for key in dialog.live_values():
        assert key in ImageUmapExplorer.LIVE_DISPLAY_KEYS, (
            f"{key} is offered as live but the explorer does not apply it")


def test_the_explorer_exposes_a_button_for_it(explorer):
    assert explorer._display_btn.text().startswith("Display settings")


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def test_values_propagate_through_the_registered_callback(explorer,
                                                          monkeypatch,
                                                          qtbot):
    """The same seam the Mask live preview uses.

    Without it a value tuned here lives only in the widget and is gone at
    the next run — which is the opposite of tuning it.
    """
    from spacr.qt.widgets import umap_explorer as module

    sent = {}
    explorer.set_propagate_callback(sent.update)

    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return True

        def values(self):
            return {"point_size": 44, "figuresize": 11.0}

        def live_values(self):
            return {"point_size": 44}

    monkeypatch.setattr(module, "UmapDisplaySettings", _Dialog)
    explorer.open_display_settings()

    assert sent["point_size"] == 44
    assert sent["figuresize"] == 11.0, (
        "the non-live half must propagate too, or it can never take effect")


def test_a_missing_callback_is_not_an_error(explorer, monkeypatch):
    """The explorer is usable on its own, and every test builds one that way."""
    from spacr.qt.widgets import umap_explorer as module

    class _Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return True

        def values(self):
            return {"point_size": 30}

        def live_values(self):
            return {"point_size": 30}

    monkeypatch.setattr(module, "UmapDisplaySettings", _Dialog)
    explorer.open_display_settings()
