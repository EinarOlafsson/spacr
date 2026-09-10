"""Applying figure settings to a finished Image UMAP, and the form that holds them.

The point of the three tiers is that "applies now" really does apply to the
figure already on screen and really does not move any point. So the styling
half is driven against a real matplotlib figure with real collections, and the
redraw half against a real payload carrying a real embedding.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

from spacr.qt.widgets import umap_figure_settings as ufs
from spacr.qt.widgets.umap_figure_settings import (
    IMAGE_UMAP_FIELDS, TIER_REDRAW, TIER_RERUN, TIER_STYLE,
    UmapFigureSettings, apply_to_figure, keys_for_tier, live_keys,
    redraw_umap_figure, restyle_umap_figure,
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


def test_restyling_nothing_touches_nothing():
    """A window open with no figure behind it is an ordinary state."""
    assert restyle_umap_figure(None, {"dot_size": 20}) is False


def test_the_points_take_the_new_size_opacity_and_outline(scatter):
    """Set straight onto the artists; no point can move."""
    touched = restyle_umap_figure(scatter, {"dot_size": 33, "point_alpha": 0.25,
                                            "outline_width": 4.0})

    collection = scatter.get_axes()[0].collections[0]
    assert touched is True
    assert list(collection.get_sizes()) == [33.0]
    assert collection.get_alpha() == pytest.approx(0.25)
    assert scatter.get_axes()[0].lines[0].get_linewidth() == pytest.approx(4.0)


def test_an_opacity_outside_zero_to_one_is_clamped(scatter):
    """A typed 5 is not a transparency, and must not raise either."""
    restyle_umap_figure(scatter, {"point_alpha": 5.0})

    assert scatter.get_axes()[0].collections[0].get_alpha() == 1.0


def test_a_fixed_colour_can_be_taken_back_off(scatter):
    """Without the stash, switching back to 'cluster' leaves every point red."""
    original = scatter.get_axes()[0].collections[0].get_facecolor().copy()

    restyle_umap_figure(scatter, {"point_color": "red"})
    reddened = scatter.get_axes()[0].collections[0].get_facecolor().copy()
    restyle_umap_figure(scatter, {"point_color": "cluster"})
    restored = scatter.get_axes()[0].collections[0].get_facecolor()

    assert len(np.unique(reddened, axis=0)) == 1
    assert np.allclose(restored, original)


@pytest.mark.parametrize("method, values", [
    ("set_sizes", {"dot_size": 10}),
    ("set_alpha", {"point_alpha": 0.5}),
    ("set_facecolor", {"point_color": "red"}),
])
def test_an_artist_that_refuses_a_style_does_not_lose_the_figure(
        scatter, method, values):
    """Restyling is a convenience; a failure costs the setting, not the plot."""
    collection = scatter.get_axes()[0].collections[0]

    def _refuse(*args, **kwargs):
        raise RuntimeError("this artist is not stylable")

    setattr(collection, method, _refuse)

    assert restyle_umap_figure(scatter, values) is False


def test_an_outline_that_refuses_a_width_does_not_lose_the_figure(scatter):
    """Same posture for the cluster outlines."""
    line = scatter.get_axes()[0].lines[0]

    def _refuse(*args, **kwargs):
        raise RuntimeError("not a line any more")

    line.set_linewidth = _refuse

    assert restyle_umap_figure(scatter, {"outline_width": 2.0}) is False


# ---------------------------------------------------------------------------
# Redrawing from the same embedding


@pytest.mark.parametrize("fig, load", [
    (None, {}),
    ("figure", "not a mapping"),
])
def test_a_redraw_with_nothing_to_draw_reports_that(scatter, fig, load):
    """Neither a missing figure nor a missing payload is an error."""
    assert redraw_umap_figure(scatter if fig else None, load, {}) is False


def test_a_payload_with_no_embedding_is_not_redrawn(scatter):
    """A figure that never carried one cannot be edited from it."""
    assert redraw_umap_figure(scatter, {"embedding": []}, {}) is False
    assert redraw_umap_figure(scatter, {"embedding": np.zeros((0, 2))},
                              {}) is False


def test_a_payload_with_an_embedding_but_no_labels_is_not_redrawn(scatter):
    """Coordinates without their cluster labels are not a drawable payload."""
    axes = tuple(scatter.get_axes())

    assert redraw_umap_figure(
        scatter, {"embedding": np.zeros((12, 2))}, {}
    ) is False
    assert tuple(scatter.get_axes()) == axes


def test_labels_that_do_not_describe_the_embedding_are_refused(scatter,
                                                               payload):
    """Drawing them would colour points by somebody else's clusters."""
    payload["labels"] = np.array([0, 1])

    assert redraw_umap_figure(scatter, payload, {}) is False


def test_the_redrawn_figure_keeps_the_same_coordinates(scatter, payload):
    """"Live apply" on a projection is only honest if no point moves."""
    assert redraw_umap_figure(scatter, payload, {"figuresize": 6.0}) is True

    axes = scatter.get_axes()[0]
    drawn = np.vstack([collection.get_offsets() for collection in
                       axes.collections])
    assert np.allclose(np.sort(drawn, axis=0),
                       np.sort(payload["embedding"], axis=0))
    assert scatter.get_size_inches() == pytest.approx((6.0, 6.0))
    assert scatter._spacr_umap_payload is payload


def test_the_plot_labels_win_over_the_raw_labels(scatter, payload):
    """A run that relabelled its clusters for the figure means those."""
    payload["plot_labels"] = np.array([7] * 12)

    assert redraw_umap_figure(scatter, payload, {}) is True
    assert len(scatter.get_axes()[0].collections) >= 1


def test_a_figure_that_will_not_resize_is_still_redrawn(scatter, payload,
                                                        monkeypatch):
    """The size is decoration; the embedding is the result."""
    def _refuse(*args, **kwargs):
        raise RuntimeError("no canvas")

    monkeypatch.setattr(scatter, "set_size_inches", _refuse)

    assert redraw_umap_figure(scatter, payload, {"figuresize": 8.0}) is True


def test_an_image_overlay_that_fails_does_not_lose_the_embedding(
        scatter, payload, monkeypatch, tmp_path):
    """A montage is decoration; losing the thumbnails must not lose the plot."""
    payload["records"] = [{"image": str(tmp_path / f"{i}.png")}
                          for i in range(12)]

    def _refuse(*args, **kwargs):
        raise RuntimeError("the crops are on a disconnected share")

    monkeypatch.setattr("spacr.utils.plot_umap_images", _refuse)

    assert redraw_umap_figure(scatter, payload,
                              {"plot_images": True}) is True


def test_the_image_overlay_is_drawn_when_the_crops_are_there(scatter, payload,
                                                             monkeypatch,
                                                             tmp_path):
    """The one call that puts thumbnails on the embedding."""
    payload["records"] = [{"image": str(tmp_path / f"{i}.png")}
                          for i in range(12)]
    drawn = []
    monkeypatch.setattr("spacr.utils.plot_umap_images",
                        lambda *args, **kwargs: drawn.append(args))

    assert redraw_umap_figure(scatter, payload,
                              {"plot_images": True, "image_nr": 4}) is True
    assert len(drawn) == 1


# ---------------------------------------------------------------------------
# Which of the two an applier chooses


def test_nothing_changed_is_no_work(scatter, payload):
    """Re-rasterising an unchanged figure is the cost this avoids."""
    values = {"dot_size": 20}

    assert apply_to_figure(scatter, payload, values, previous=values) == ""


def test_a_style_change_restyles_and_a_redraw_setting_redraws(scatter,
                                                              payload):
    """The tier of the changed key decides, not the tier of every key."""
    assert apply_to_figure(scatter, payload, {"dot_size": 20},
                           previous={"dot_size": 10}) == "style"
    assert apply_to_figure(scatter, payload, {"figuresize": 7.0},
                           previous={"figuresize": 10.0}) == "redraw"


def test_a_rerun_setting_alone_changes_nothing_on_screen(scatter, payload):
    """Applying it would move every point and lose the arrangement."""
    assert apply_to_figure(scatter, payload, {"n_neighbors": 30},
                           previous={"n_neighbors": 15}) == ""


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


def test_a_missing_editor_is_skipped_rather_than_crashing_the_form(form):
    """The field table is data; a build that dropped one must not raise."""
    del form._editors["n_neighbors"]

    form._refresh_reducer_fields()

    assert "n_neighbors" not in form.values()


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


def test_a_row_limit_can_say_every_row(form):
    """A spin box has no way to say None, so the field is typed."""
    form._editors["row_limit"].setText("all")
    assert form.values()["row_limit"] is None

    form._editors["row_limit"].setText("2500")
    assert form.values()["row_limit"] == 2500


@pytest.mark.parametrize("text, expected", [
    ("1000", 1000), ("1e3", 1000), ("", None), ("none", None),
    ("null", None), ("all", None), ("many", None), (None, None),
])
def test_a_typed_row_limit_is_read_or_treated_as_every_row(text, expected):
    """Junk means "every row" rather than an exception on every keystroke."""
    assert ufs._int_or_none(text) == expected


def test_a_change_is_emitted_once_and_not_repeated(form, qtbot):
    """The window carries the whole dict, debounced, with no Apply button."""
    seen = []
    form.settings_changed.connect(seen.append)

    form._editors["dot_size"].setValue(form._editors["dot_size"].value() + 5)
    form.flush()
    form.flush()

    assert len(seen) == 1
    assert seen[0]["dot_size"] == form.values()["dot_size"]


def test_setting_a_value_back_to_what_it_was_emits_nothing(form):
    """An emission per keystroke would re-rasterise the figure for nothing."""
    seen = []
    form.settings_changed.connect(seen.append)
    original = form._editors["dot_size"].value()

    form._editors["dot_size"].setValue(original + 5)
    form._editors["dot_size"].setValue(original)
    form.flush()

    assert seen == []


def test_cancel_puts_back_what_the_window_opened_on(form):
    """`initial_values` is the whole of the Cancel contract."""
    opened = form.initial_values()
    form._editors["dot_size"].setValue(form._editors["dot_size"].value() + 7)

    assert form.values() != opened
    assert form.initial_values() == opened
