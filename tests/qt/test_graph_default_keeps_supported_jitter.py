"""293: supported jitter defaults must reach the first real chart."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def chart(qtbot, monkeypatch):
    """Build a CPU-only chart using the real saved-preference reader path."""
    from spacr.qt import preferences
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.graph_builder import GraphCanvas
    from spacr.qt.widgets.graph_spec import GraphSpec

    def build(preference, *, pinned=None, frame=None, x="group", y="value",
              **spec_options):
        """Draw the requested data shape without replacing preference logic."""
        if frame is None:
            frame = _groups(12)
        shape = ("categorical_continuous" if x and y
                 else "continuous_only")
        monkeypatch.setattr(
            preferences, "get_default_graph_type",
            lambda requested: preference if requested == shape else "",
        )
        canvas = GraphCanvas(link=LinkedSelection(), source="293_test")
        qtbot.addWidget(canvas)
        canvas.set_frame(frame)
        spec = GraphSpec(x=x, y=y, kind=pinned,
                         roles={"value": "continuous"}, **spec_options)
        canvas.set_spec(spec)
        return canvas, spec, spec.kinds_for(frame)

    return build


def _groups(size):
    """Two groups with an explicitly numeric measurement and finite values."""
    return pd.DataFrame({
        "group": ["a"] * size + ["b"] * size,
        "value": np.arange(size * 2, dtype=float),
    })


def _point_count(canvas):
    """Count finite observation marks across every displayed facet."""
    from matplotlib.collections import PathCollection

    total = 0
    for axes in canvas._figure.axes:
        for artist in axes.collections:
            if isinstance(artist, PathCollection):
                offsets = np.ma.filled(artist.get_offsets(), np.nan)
                total += int(np.isfinite(offsets).all(axis=1).sum())
    return total


def _assert_jitter_marks(canvas, kind):
    """Check actual observations and bars, not just the resolver's answer."""
    axes, = canvas._figure.axes
    assert _point_count(canvas) == 24
    assert len(axes.patches) == (2 if kind == "bar_jitter" else 0)


@pytest.mark.parametrize("kind", ["jitter", "bar_jitter"])
def test_supported_saved_default_draws_the_requested_first_chart(chart, kind):
    """A supported preference must not become a box with a false refusal."""
    canvas, spec, kinds = chart(kind)
    assert spec.resolved_kind(kinds) == kind
    assert spec._kind_note(kinds) == ""
    assert spec.describe(kinds).startswith(f"{kind} · ")
    _assert_jitter_marks(canvas, kind)


@pytest.mark.parametrize("kind", ["jitter", "bar_jitter"])
def test_explicit_jitter_pin_is_a_renderable_positive_control(chart, kind):
    """The same renderer already supports both kinds when explicitly pinned."""
    canvas, spec, kinds = chart("violin", pinned=kind)
    assert spec.resolved_kind(kinds) == kind
    assert spec._kind_note(kinds) == ""
    assert "(pinned)" in spec.describe(kinds)
    _assert_jitter_marks(canvas, kind)


def test_a_genuinely_unsupported_default_still_explains_its_fallback(chart):
    """Box+jitter has no renderer here; do not remove the capability check."""
    _canvas, spec, kinds = chart("box_jitter")
    assert spec.resolved_kind(kinds) == "box"
    note = spec._kind_note(kinds)
    assert "not a chart this builder draws" in note
    assert note in spec.describe(kinds)


def test_no_saved_default_keeps_the_existing_box_inference(chart):
    """Fixing a saved choice must not change charts with no saved choice."""
    _canvas, spec, kinds = chart("")
    assert spec.resolved_kind(kinds) == "box"
    assert spec._kind_note(kinds) == ""


def test_unsupported_default_explanation_reaches_the_visible_figure(chart):
    """An internal description is not the notice the chart's reader sees."""
    canvas, spec, kinds = chart("box_jitter")
    note = spec._kind_note(kinds)
    assert note
    assert note in canvas.notice()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("preference", ["jitter", "box", "violin"])
def test_one_numeric_column_keeps_a_drawable_shape_specific_fallback(
        chart, axis, preference):
    """These renderers require a category too; one column must not go blank."""
    bindings = {"x": None, "y": None, axis: "value"}
    canvas, spec, kinds = chart(preference, **bindings)
    assert spec.resolved_kind(kinds) == "histogram"
    assert "not a chart this builder draws" in canvas.notice()
    axes, = canvas._figure.axes
    assert axes.patches
    assert sum(patch.get_height() for patch in axes.patches) == 24


@pytest.mark.parametrize("axis", ["x", "y"])
def test_one_numeric_column_without_a_choice_keeps_its_histogram(chart, axis):
    """The one-column renderer is a positive control in both orientations."""
    bindings = {"x": None, "y": None, axis: "value"}
    canvas, spec, kinds = chart("", **bindings)
    assert spec.resolved_kind(kinds) == "histogram"
    assert spec._kind_note(kinds) == ""
    axes, = canvas._figure.axes
    assert sum(patch.get_height() for patch in axes.patches) == 24


@pytest.mark.parametrize("preference", ["box", "violin", "bar"])
def test_two_observations_draw_points_and_explain_the_saved_summary_fallback(
        chart, preference):
    """The small-data rule changes the actual picture, without creating a pin."""
    canvas, spec, _kinds = chart(preference, frame=_groups(2))
    assert _point_count(canvas) == 4
    assert not canvas._figure.axes[0].patches
    assert canvas.spec is spec
    assert canvas.spec.kind is None
    assert "smallest here has 2" in canvas.notice()
    assert "Jitter is drawn instead" in canvas.notice()
    assert "Right-click" in canvas.notice()
    assert canvas._figure.axes[0].get_ylabel() == "value"


@pytest.mark.parametrize("size", [8, 9])
def test_summary_boundary_uses_the_shared_eight_observation_rule(chart, size):
    """The boundary is inclusive; a nine-point group's chosen box survives."""
    canvas, _spec, _kinds = chart("box", frame=_groups(size))
    if size == 8:
        assert _point_count(canvas) == 16
        assert "smallest here has 8" in canvas.notice()
    else:
        assert _point_count(canvas) == 0
        assert len(canvas._figure.axes[0].patches) == 2
        assert "drawn instead" not in canvas.notice()


def test_nonfinite_values_do_not_inflate_the_observation_counts(chart):
    """Missing and infinite measurements are not evidence for a distribution."""
    finite = _groups(2)
    invalid = pd.DataFrame({
        "group": ["a"] * 9 + ["b"] * 9,
        "value": [np.nan, np.inf, -np.inf] * 6,
    })
    frame = pd.concat([finite, invalid], ignore_index=True)
    canvas, _spec, _kinds = chart("box", frame=frame)
    assert "smallest here has 2" in canvas.notice()
    assert _point_count(canvas) == 4


def test_active_filter_changes_the_group_sizes_before_the_kind_is_decided(chart):
    """Twenty-four source rows cannot justify a box for four visible rows."""
    from spacr.selection import CategoryFilter, DataFilter

    frame = _groups(12)
    frame["visible"] = (["keep"] * 2 + ["hide"] * 10) * 2
    canvas, _spec, _kinds = chart("box", frame=frame)
    assert len(canvas._figure.axes[0].patches) == 2
    canvas.link.set_filter(DataFilter().add(CategoryFilter("visible", ("keep",))))
    canvas.render_now()
    assert canvas.render_data.n_shown == 4
    assert _point_count(canvas) == 4
    assert "smallest here has 2" in canvas.notice()


def test_counts_are_per_displayed_facet_not_pooled_across_panels(chart):
    """Three observations per panel are not nine observations in each panel."""
    frame = pd.concat([_groups(3).assign(panel=label)
                       for label in ("one", "two", "three")], ignore_index=True)
    canvas, _spec, _kinds = chart("box", frame=frame, facet_col="panel")
    assert len(canvas._figure.axes) == 3
    assert _point_count(canvas) == 18
    assert "smallest here has 3" in canvas.notice()


def test_render_budget_does_not_replace_actual_group_sizes(chart):
    """A point budget of two does not turn a populated summary into a tiny one."""
    canvas, _spec, _kinds = chart("box", frame=_groups(12), point_budget=2)
    assert canvas.render_data.n_total == canvas.render_data.n_shown == 24
    assert len(canvas._figure.axes[0].patches) == 2
    assert "drawn instead" not in canvas.notice()


def test_new_data_reconsiders_a_temporary_small_sample_fallback(chart):
    """A fallback is not a user pin and must not persist over a larger table."""
    canvas, spec, _kinds = chart("box", frame=_groups(2))
    assert _point_count(canvas) == 4
    assert "smallest here has 2" in canvas.notice()
    canvas.set_frame(_groups(12))
    assert canvas.spec is spec
    assert canvas.spec.kind is None
    assert len(canvas._figure.axes[0].patches) == 2
    assert _point_count(canvas) == 0
    assert "drawn instead" not in canvas.notice()


def test_explicit_small_sample_pin_survives_redraw_and_a_new_table(chart):
    """A manual chart choice still outranks the saved starting preference."""
    canvas, spec, _kinds = chart("jitter", pinned="box", frame=_groups(2))
    for frame in (_groups(2), _groups(12)):
        canvas.set_frame(frame)
        assert canvas.spec.kind == "box"
        assert canvas.spec is spec
        assert len(canvas._figure.axes[0].patches) == 2
        assert _point_count(canvas) == 0
        assert "drawn instead" not in canvas.notice()
