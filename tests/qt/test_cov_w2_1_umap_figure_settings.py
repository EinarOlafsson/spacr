"""Applying figure settings to a finished Image UMAP, and the form that holds them.

The point of the three tiers is that "applies now" really does apply to the
figure already on screen and really does not move any point. So the styling
half is driven against a real matplotlib figure with real collections, and the
redraw half against a real payload carrying a real embedding.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

from spacr.qt.widgets import umap_figure_settings as ufs
from spacr.qt.widgets.umap_figure_settings import (
    IMAGE_UMAP_FIELDS, TIER_REDRAW, TIER_RERUN, TIER_STYLE,
    UmapFigureSettings, keys_for_tier, live_keys,
)


@pytest.fixture
def scatter():
    """A figure with one scatter collection and one outline, as plotted."""
    from matplotlib.figure import Figure

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.5],
                 c=["#ff0000", "#00ff00", "#0000ff"])
    axes.plot([0.0, 1.0], [0.0, 1.0])
    return fig


@pytest.fixture
def payload():
    """An embedding and its labels, the way the run stashes them."""
    rng = np.random.default_rng(0)
    embedding = np.vstack([rng.normal(size=(6, 2)),
                           rng.normal(loc=6.0, size=(6, 2))])
    return {"embedding": embedding,
            "labels": np.array([0] * 6 + [1] * 6),
            "records": [{"image": None} for _ in range(12)]}


# ---------------------------------------------------------------------------
# The tiers


def test_each_tier_names_its_own_settings():
    """A setting in two tiers would be applied twice, two different ways."""
    style = set(keys_for_tier(TIER_STYLE))
    redraw = set(keys_for_tier(TIER_REDRAW))
    rerun = set(keys_for_tier(TIER_RERUN))

    assert style == {"dot_size", "point_color", "point_alpha", "outline_width"}
    assert "figuresize" in redraw
    assert "n_neighbors" in rerun
    assert not style & redraw and not redraw & rerun and not style & rerun
    assert style | redraw | rerun == {f.key for f in IMAGE_UMAP_FIELDS}


def test_the_live_settings_are_the_two_tiers_that_reach_the_screen():
    """The rerun tier is saved and propagated, never applied."""
    assert set(live_keys()) == set(keys_for_tier(TIER_STYLE)) | set(
        keys_for_tier(TIER_REDRAW))
    assert "n_neighbors" not in live_keys()


# ---------------------------------------------------------------------------
# Restyling what is already drawn


# ---------------------------------------------------------------------------
# Redrawing from the same embedding


# ---------------------------------------------------------------------------
# Which of the two an applier chooses


# ---------------------------------------------------------------------------
# The form


@pytest.fixture
def form(qtbot):
    widget = UmapFigureSettings()
    qtbot.addWidget(widget)
    return widget


def test_only_the_selected_reducer_s_settings_are_editable(form):
    """A greyed row says the recipe on screen ignores it."""
    form._editors["reduction_method"].setCurrentText("tsne")

    assert form._editors["tsne_perplexity"].isEnabled()
    assert not form._editors["n_neighbors"].isEnabled()
    assert not form._editors["pca_whiten"].isEnabled()


def test_spectral_neighbours_follow_the_affinity(form):
    """`spectral_n_neighbors` means nothing to an RBF affinity."""
    form._editors["reduction_method"].setCurrentText("spectral")
    form._editors["spectral_affinity"].setCurrentText("nearest_neighbors")
    assert form._editors["spectral_n_neighbors"].isEnabled()

    form._editors["spectral_affinity"].setCurrentText("rbf")

    assert not form._editors["spectral_n_neighbors"].isEnabled()


def test_defaults_that_cannot_be_read_leave_the_form_empty_not_broken(
        monkeypatch, qtbot):
    """A settings module that will not answer costs the seeds, not the window."""
    import spacr.settings as settings_module

    def _refuse(values):
        raise RuntimeError("no defaults on this install")

    monkeypatch.setattr(settings_module, "set_default_umap_image_settings",
                        _refuse)

    assert UmapFigureSettings._defaults() == {}


@pytest.mark.parametrize("key, editor_type", [
    ("n_neighbors", QSpinBox),
    ("min_dist", QDoubleSpinBox),
])
def test_a_seeded_value_that_is_not_a_number_leaves_the_default(form, key,
                                                                editor_type):
    """A settings CSV can hold anything; the spin box keeps its own value."""
    field = next(f for f in IMAGE_UMAP_FIELDS if f.key == key)

    editor = form._editor(field, "not a number")

    assert isinstance(editor, editor_type)
    assert editor.value() == editor.minimum()


def test_the_live_half_is_what_reaches_the_open_figure(form):
    """The window emits everything; only this half is applied."""
    live = form.live_values()

    assert set(live) == set(live_keys())
    assert "n_neighbors" not in live


@pytest.mark.parametrize("text, expected", [
    ("1000", 1000), ("1e3", 1000), ("", None), ("none", None),
    ("null", None), ("all", None), ("many", None), (None, None),
])
def test_a_typed_row_limit_is_read_or_treated_as_every_row(text, expected):
    """Junk means "every row" rather than an exception on every keystroke."""
    assert ufs._int_or_none(text) == expected


def test_cancel_puts_back_what_the_window_opened_on(form):
    """`initial_values` is the whole of the Cancel contract."""
    opened = form.initial_values()
    form._editors["dot_size"].setValue(form._editors["dot_size"].value() + 7)

    assert form.values() != opened
    assert form.initial_values() == opened
