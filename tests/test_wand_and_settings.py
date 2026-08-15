"""The magic wand, and the Gate Editor settings.

The wand half is pure geometry and needs no GUI. The settings half checks
that a setting reaches the thing it configures -- the failure mode for a
settings dialog is never that the widget is missing, it is that nothing reads
what the widget wrote.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    GateError, WandError, wand_gate, wand_select,
)


def _two_clouds(seed=0, gap=8.0, n=300):
    rng = np.random.default_rng(seed)
    return pd.concat([
        pd.DataFrame({"x": rng.normal(0.0, 0.3, n),
                      "y": rng.normal(0.0, 0.3, n)}),
        pd.DataFrame({"x": rng.normal(gap, 0.3, n),
                      "y": rng.normal(gap, 0.3, n)}),
    ], ignore_index=True)


# ---------------------------------------------------------------------------
# Growing the selection
# ---------------------------------------------------------------------------

def test_a_click_selects_the_population_it_landed_in(seed=0):
    frame = _two_clouds()
    mask = wand_select(frame, "x", "y", 0.0, 0.0)
    assert mask[:300].sum() > 250, "most of the clicked cloud was missed"
    assert not mask[300:].any(), "the wand reached the other population"


def test_the_far_population_is_selected_by_clicking_it():
    frame = _two_clouds()
    mask = wand_select(frame, "x", "y", 8.0, 8.0)
    assert mask[300:].sum() > 250
    assert not mask[:300].any()


def test_the_selection_flows_along_a_ridge_rather_than_taking_a_circle():
    """The watershed part. A bent population has to come out whole, which a
    radius around the click cannot do."""
    t = np.linspace(0.0, np.pi, 400)
    frame = pd.DataFrame({"x": np.cos(t), "y": np.sin(t)})
    mask = wand_select(frame, "x", "y", 1.0, 0.0,
                       tolerance=0.05, max_radius=10.0)
    assert mask.all(), (
        "the selection stopped partway along an unbroken arc; it is growing "
        "by distance from the click, not along neighbours")


def test_the_maximum_distance_stops_a_chain_bridging_two_populations():
    """Without a ceiling one chain of objects merges two clouds, which on a
    real scatter happens more often than not."""
    bridge = pd.DataFrame({"x": np.linspace(0.0, 8.0, 400),
                           "y": np.linspace(0.0, 8.0, 400)})
    frame = pd.concat([_two_clouds(), bridge], ignore_index=True)
    reached = wand_select(frame, "x", "y", 0.0, 0.0, max_radius=0.2)
    assert not reached[300:600].any(), "the wand crossed the bridge"

    everything = wand_select(frame, "x", "y", 0.0, 0.0, max_radius=10.0)
    assert everything[300:600].any(), (
        "with the ceiling lifted the bridge should be crossable, or this "
        "test is not testing the ceiling")


def test_a_tolerance_below_the_spacing_selects_almost_nothing():
    frame = _two_clouds()
    mask = wand_select(frame, "x", "y", 0.0, 0.0, tolerance=1e-6)
    assert mask.sum() <= 2, "objects were joined across a gap wider than the tolerance"


def test_the_seed_is_the_nearest_object_not_the_click():
    """A click landing in a gap between two objects of a cloud must still
    start inside the cloud."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
    mask = wand_select(frame, "x", "y", 0.4, 0.0, tolerance=0.6, max_radius=1.0)
    assert mask.any(), "a click that missed every object selected nothing"


def test_scaling_makes_one_tolerance_work_on_unlike_measurements():
    """Unscaled, a tolerance is a distance in whichever measurement has the
    larger numbers and the other axis is ignored."""
    frame = pd.DataFrame({"small": np.linspace(0.0, 1.0, 100),
                          "big": np.linspace(0.0, 100_000.0, 100)})
    scaled = wand_select(frame, "small", "big", 0.5, 50_000.0,
                         tolerance=0.05, max_radius=1.0, scale=True)
    assert scaled.sum() > 10, "a scaled tolerance selected almost nothing"


def test_a_constant_measurement_does_not_divide_by_nothing():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 50), "y": [3.0] * 50})
    mask = wand_select(frame, "x", "y", 0.5, 3.0, tolerance=0.1, max_radius=1.0)
    assert mask.any()


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_tolerance_or_radius_is_refused(bad):
    frame = _two_clouds()
    with pytest.raises(WandError):
        wand_select(frame, "x", "y", 0.0, 0.0, tolerance=bad)
    with pytest.raises(WandError):
        wand_select(frame, "x", "y", 0.0, 0.0, max_radius=bad)


def test_clicking_empty_space_says_what_to_change():
    frame = _two_clouds()
    with pytest.raises(WandError, match="maximum distance"):
        wand_select(frame, "x", "y", 100.0, 100.0, max_radius=0.01)


# ---------------------------------------------------------------------------
# Fitting the polygon
# ---------------------------------------------------------------------------

def test_the_wand_produces_a_polygon_gate_that_selects_its_population():
    frame = _two_clouds()
    gate = wand_gate(frame, "x", "y", 0.0, 0.0, name="blob")
    assert gate.name == "blob"
    assert len(gate.vertices) >= 3

    inside = gate.mask(frame)
    assert inside[:300].sum() > 250, "the fitted gate lost its own population"
    assert not inside[300:].any(), "the fitted gate reached the other cloud"


def test_the_gate_is_a_shape_so_it_re_applies_to_another_table():
    """A gate has to be a region, not a list of rows -- that is the whole
    difference between a lasso and a gate."""
    gate = wand_gate(_two_clouds(seed=1), "x", "y", 0.0, 0.0)
    other = _two_clouds(seed=99)
    assert gate.mask(other)[:300].sum() > 200


def test_too_few_objects_to_fit_a_polygon_says_so():
    frame = pd.DataFrame({"x": [0.0, 5.0], "y": [0.0, 5.0]})
    with pytest.raises(WandError, match="polygon needs three"):
        wand_gate(frame, "x", "y", 0.0, 0.0, tolerance=0.01, max_radius=0.05)


def test_a_straight_line_of_objects_has_no_area_to_gate():
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 40), "y": [0.0] * 40})
    with pytest.raises(WandError, match="straight line"):
        wand_gate(frame, "x", "y", 0.5, 0.0, tolerance=0.5, max_radius=2.0)


def test_a_missing_column_is_named():
    with pytest.raises(GateError, match="ghost"):
        wand_select(_two_clouds(), "x", "ghost", 0.0, 0.0)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_only_the_settings_that_cost_a_read_ask_for_one():
    """Re-reading a large table because the user nudged a colour map is the
    lag the dialog exists to remove."""
    from spacr.qt.widgets.gate_settings import GateEditorSettings

    base = GateEditorSettings()
    assert base.costs_a_reload(base.replaced(sample_fraction=0.2))
    assert base.costs_a_reload(base.replaced(max_points=1000))
    assert not base.costs_a_reload(base.replaced(colour_map="magma"))
    assert not base.costs_a_reload(base.replaced(point_size=12.0))
    assert not base.costs_a_reload(base.replaced(wand_tolerance=0.2))


def test_the_default_reads_every_row_but_draws_at_most_ten_thousand():
    """No SAMPLING by default -- a fraction of the data shown to a user who
    never asked for one is a lie about the plot. A row CAP is different: past
    ten thousand a scatter draws more markers than the screen has pixels, and
    the large-data raster takes the per-point settings with it.
    """
    from spacr.qt.widgets.gate_settings import GateEditorSettings

    settings = GateEditorSettings()
    assert settings.sample_fraction == 1.0
    assert settings.max_points == 10_000


def test_editing_the_dialog_emits_the_whole_settings_object(qtbot):
    from spacr.qt.widgets.gate_settings import (
        GateEditorSettings, GateSettingsDialog,
    )

    seen = []
    dialog = GateSettingsDialog(GateEditorSettings())
    qtbot.addWidget(dialog)
    dialog.settings_changed.connect(seen.append)

    dialog._sample.setValue(20)
    assert seen, "editing a setting emitted nothing"
    assert abs(seen[-1].sample_fraction - 0.2) < 1e-9
    assert dialog.settings().sample_fraction == seen[-1].sample_fraction


def test_the_dialog_has_a_tab_per_gating_mode(qtbot):
    from spacr.qt.widgets.gate_settings import (
        GateEditorSettings, GateSettingsDialog,
    )

    dialog = GateSettingsDialog(GateEditorSettings())
    qtbot.addWidget(dialog)
    titles = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    # xD joined them in instruction 49: it is where the column picker and the
    # projection's own hyperparameters live, and those belong beside the
    # selection they depend on rather than on the 3D tab they were squatting
    # in. The name of this test said "three" and pinned the count, so it
    # failed for the tab being ADDED rather than for anything being wrong.
    assert titles == ["General", "2D", "3D", "xD"]


def test_merge_keys_stay_in_join_order_not_click_order(qtbot):
    """A join on the same keys in a different order is the same join written
    two ways, and comparing the two would report a change that is not one."""
    from spacr.qt.widgets.gate_settings import (
        MERGE_KEYS, GateEditorSettings, GateSettingsDialog,
    )

    dialog = GateSettingsDialog(GateEditorSettings())
    qtbot.addWidget(dialog)
    dialog._merge_boxes["plateID"].setChecked(False)
    dialog._merge_boxes["plateID"].setChecked(True)
    chosen = dialog.settings().merge_keys
    assert list(chosen) == [k for k in MERGE_KEYS if k in chosen]


def test_the_canvas_takes_the_drawing_settings(qtbot):
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.gate_settings import GateEditorSettings

    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.apply_settings(GateEditorSettings().replaced(
        highlight_gated=False, gate_line_width=3.0))
    assert canvas._highlight_gated is False
    assert canvas._line_width == 3.0


def test_a_partial_settings_object_does_not_stop_the_canvas_drawing(qtbot):
    """Decoration is never load-bearing: an older saved settings set must not
    make the editor unable to draw."""
    from spacr.qt.widgets.gate_editor import GateCanvas

    class Partial:
        colour_map = "viridis"

    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.apply_settings(Partial())
    assert canvas._highlight_gated is True
