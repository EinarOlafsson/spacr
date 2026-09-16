"""293: a grouped plot's saved default meets the finite data before drawing."""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pg = pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _frame(size, second_size=None):
    """Create two categorical groups with finite, nonconstant measurements."""
    second_size = size if second_size is None else second_size
    return pd.DataFrame({
        "group": ["a"] * size + ["b"] * second_size,
        "value": np.arange(size + second_size, dtype=float),
    })


def _spec(frame, *, kind=""):
    """Use the existing public spec API, with no implicit user pin."""
    from spacr.qt.widgets.grouped_plot import PlotSpec

    return PlotSpec(frame=frame, value="value", group="group", kind=kind)


def _visible_text(widget):
    """Read the actual displayed labels, not a detached caption helper."""
    from PySide6.QtWidgets import QLabel

    return "\n".join(label.text() for label in widget.findChildren(QLabel)
                     if not label.isHidden())


@pytest.fixture
def choose(monkeypatch):
    """Patch only the saved preference, keeping the real compatibility rules."""
    from spacr.qt import preferences

    def set_choice(kind):
        """Choose the starting graph for categorical-versus-numeric data."""
        monkeypatch.setattr(
            preferences, "get_default_graph_type",
            lambda shape: kind if shape == "categorical_continuous" else "",
        )

    return set_choice


@pytest.fixture
def plot(qtbot, monkeypatch):
    """Build a real software-rendered widget and observe its real mark calls."""
    from spacr.qt.widgets.grouped_plot import GroupedPlot

    monkeypatch.setitem(pg.CONFIG_OPTIONS, "useOpenGL", False)
    widget = GroupedPlot()
    qtbot.addWidget(widget)
    drawn = []
    add_group_mark = widget.add_group_mark
    depth = 0

    def record_mark(position, values, mark, **kwargs):
        """Record top-level choices while forwarding recursive composite marks."""
        nonlocal depth
        if depth == 0:
            drawn.append((mark, np.asarray(values, dtype=float).copy()))
        depth += 1
        try:
            return add_group_mark(position, values, mark, **kwargs)
        finally:
            depth -= 1

    monkeypatch.setattr(widget, "add_group_mark", record_mark)
    widget.show()
    return widget, drawn


@pytest.mark.parametrize("chosen", ["box", "violin"])
@pytest.mark.parametrize("size", [2, 8])
def test_spec_default_reconsiders_the_saved_summary_for_small_groups(
        choose, chosen, size):
    """The spec already has the data needed to reject a tiny distribution."""
    choose(chosen)
    spec = _spec(_frame(size))
    assert spec.default_kind() == "jitter"
    assert spec.kind == "", "resolving a default must not pin the input spec"


@pytest.mark.parametrize("chosen", ["box", "violin"])
@pytest.mark.parametrize("size", [2, 8])
def test_show_spec_really_draws_points_for_a_small_saved_summary(
        plot, choose, chosen, size):
    """The effective kind must reach the renderer, not just a settings label."""
    choose(chosen)
    widget, drawn = plot
    spec = _spec(_frame(size))
    assert widget.show_spec(spec) == 2
    assert widget.spec.kind == "jitter"
    assert [mark for mark, _values in drawn] == ["jitter", "jitter"]
    assert [len(values) for _mark, values in drawn] == [size, size]
    assert spec.kind == ""


@pytest.mark.parametrize("chosen", ["box", "violin"])
def test_the_displayed_caption_explains_why_the_default_was_not_drawn(
        plot, choose, chosen):
    """The fallback's reason belongs in a label the reader actually sees."""
    choose(chosen)
    widget, _drawn = plot
    widget.show_spec(_spec(_frame(2)))
    caption = _visible_text(widget)
    assert chosen.capitalize() in caption
    assert "smallest here has 2" in caption
    assert "Jitter is drawn instead" in caption
    assert "Right-click" in caption
    assert "a n=2" in caption
    assert "b n=2" in caption


@pytest.mark.parametrize("chosen", ["box", "violin"])
def test_nine_finite_observations_keep_the_saved_summary(plot, choose, chosen):
    """Nine is the first valid group size above the inclusive eight-point rule."""
    choose(chosen)
    widget, drawn = plot
    spec = _spec(_frame(9))
    assert spec.default_kind() == chosen
    assert widget.show_spec(spec) == 2
    assert widget.spec.kind == chosen
    assert [mark for mark, _values in drawn] == [chosen, chosen]
    caption = _visible_text(widget)
    assert "drawn instead" not in caption
    assert "a n=9" in caption
    assert "b n=9" in caption


def test_no_saved_preference_keeps_the_existing_box_with_points(plot, choose):
    """An absent preference must not change the established starting form."""
    choose("")
    widget, drawn = plot
    spec = _spec(_frame(2))
    assert spec.default_kind() == "box_jitter"
    assert widget.show_spec(spec) == 2
    assert widget.spec.kind == "box_jitter"
    assert [mark for mark, _values in drawn] == ["jitter_box", "jitter_box"]
    assert "drawn instead" not in _visible_text(widget)


@pytest.mark.parametrize("pinned", ["box", "violin"])
def test_an_explicit_small_sample_pin_still_overrides_the_starting_preference(
        plot, choose, pinned):
    """A named kind is a deliberate choice, not a default to reconsider."""
    choose("jitter")
    widget, drawn = plot
    spec = _spec(_frame(2), kind=pinned)
    assert widget.show_spec(spec) == 2
    assert widget.spec.kind == pinned
    assert [mark for mark, _values in drawn] == [pinned, pinned]
    assert "drawn instead" not in _visible_text(widget)


@pytest.mark.parametrize("pinned", ["box", "violin"])
def test_show_as_is_a_user_choice_and_clears_a_previous_fallback_note(
        plot, choose, pinned):
    """The existing manual-retype API must not be undone by a saved default."""
    choose("box")
    widget, drawn = plot
    widget.show_spec(_spec(_frame(2)))
    drawn.clear()
    assert widget.show_as(pinned) == 2
    assert widget.spec.kind == pinned
    assert [mark for mark, _values in drawn] == [pinned, pinned]
    assert "drawn instead" not in _visible_text(widget)
    drawn.clear()
    widget.show_spec(replace(widget.spec, frame=_frame(3)))
    assert widget.spec.kind == pinned
    assert [mark for mark, _values in drawn] == [pinned, pinned]


@pytest.mark.parametrize("chosen", ["box", "violin"])
def test_a_new_unpinned_table_reconsiders_the_temporary_fallback(
        plot, choose, chosen):
    """A temporary size-based decision must not latch onto later data."""
    choose(chosen)
    widget, drawn = plot
    widget.show_spec(_spec(_frame(2)))
    assert widget.spec.kind == "jitter"
    assert "smallest here has 2" in _visible_text(widget)
    drawn.clear()
    widget.show_spec(_spec(_frame(12)))
    assert widget.spec.kind == chosen
    assert [mark for mark, _values in drawn] == [chosen, chosen]
    assert "drawn instead" not in _visible_text(widget)
    drawn.clear()
    widget.show_spec(_spec(_frame(3)))
    assert widget.spec.kind == "jitter"
    assert [mark for mark, _values in drawn] == ["jitter", "jitter"]
    assert "smallest here has 3" in _visible_text(widget)


@pytest.mark.parametrize("chosen", ["box", "violin"])
def test_nonfinite_rows_do_not_make_a_tiny_group_look_large(
        plot, choose, chosen):
    """The choice and displayed n describe finite measurements, not CSV rows."""
    choose(chosen)
    invalid = pd.DataFrame({
        "group": ["a"] * 9 + ["b"] * 9,
        "value": [np.nan, np.inf, -np.inf] * 6,
    })
    frame = pd.concat([_frame(2), invalid], ignore_index=True)
    widget, drawn = plot
    widget.show_spec(_spec(frame))
    assert widget.spec.kind == "jitter"
    assert [mark for mark, _values in drawn] == ["jitter", "jitter"]
    caption = _visible_text(widget)
    assert "smallest here has 2" in caption
    assert "a n=2" in caption
    assert "b n=2" in caption


@pytest.mark.parametrize("chosen", ["box", "violin"])
def test_the_smallest_group_decides_not_the_total_population(
        plot, choose, chosen):
    """A large neighbouring group does not supply missing observations."""
    choose(chosen)
    widget, drawn = plot
    widget.show_spec(_spec(_frame(2, second_size=12)))
    assert widget.spec.kind == "jitter"
    assert [mark for mark, _values in drawn] == ["jitter", "jitter"]
    assert "smallest here has 2" in _visible_text(widget)
