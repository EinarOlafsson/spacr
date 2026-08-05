"""The Prediction Profiler screen, driven against a real fitted model.

The screen is a view over :mod:`spacr.profiler`, so what is tested here is
the view: that the input list is the sensitivity ranking, that a slider
moves the curve, that the curve is drawn where the pure layout function says
it is, and that the two things the screen refuses to guess — the link and
the sweep range — are stated rather than assumed silently.

Every test runs ``threaded=False`` so a load is finished when the call
returns.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.profiler import FittedLinear, Profile                 # noqa: E402
from spacr.qt.screens import profiler as screen_module           # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def coefficients(tmp_path):
    """A coefficient table with a strong input, a weak one and a null."""
    folder = tmp_path / "results" / "pred" / "ols"
    folder.mkdir(parents=True)
    path = folder / "results.csv"
    pd.DataFrame({
        "feature": ["Intercept", "fraction:grna[100_1]",
                    "fraction:grna[200_1]", "fraction:grna[300_1]"],
        "coefficient": [0.5, 2.4, -0.8, 0.0],
    }).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def design():
    """A design matrix with real observed ranges."""
    rng = np.random.default_rng(3)
    n = 60
    return pd.DataFrame({
        "Intercept": np.ones(n),
        "fraction:grna[100_1]": rng.uniform(0.0, 0.4, n),
        "fraction:grna[200_1]": rng.uniform(0.0, 1.0, n),
        "fraction:grna[300_1]": rng.uniform(0.0, 1.0, n),
    })


@pytest.fixture()
def screen(qtbot, coefficients):
    """The screen, opened on the coefficient table, running inline."""
    widget = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_through_the_seam():
    from spacr.qt.app import APPS, SECTION_EXPLORE, registered_factory

    row = next((r for r in APPS if r[0] == screen_module.APP_KEY), None)
    assert row is not None, "importing the module did not register the app"
    assert row[3] == SECTION_EXPLORE
    assert registered_factory(screen_module.APP_KEY) is (
        screen_module.make_profiler_screen)
    assert screen_module.register() is False, "register() is not idempotent"


def test_the_screen_answers_spacr_run_with_a_sentence():
    from spacr import cli

    note = cli.INTERACTIVE_ONLY.get(screen_module.APP_KEY, "")
    assert len(note) >= 40
    assert "spacr.profiler.profile" in note


def test_the_screen_styles_itself_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names

    assert "ProfilerPlot" in widget_qss_names()
    assert "QScrollArea#ProfilerHeld" in stylesheet()


# ---------------------------------------------------------------------------
# Loading a model
# ---------------------------------------------------------------------------

def test_the_input_list_is_the_sensitivity_ranking(screen):
    ranked = screen.ranked_inputs()

    assert [r.variable for r in ranked][0] == "fraction:grna[100_1]"
    assert screen._inputs.topLevelItemCount() == len(ranked)
    assert screen._inputs.topLevelItem(0).text(0) == "fraction:grna[100_1]"
    assert "Intercept" not in [r.variable for r in ranked]


def test_the_strongest_input_is_selected_and_swept_on_load(screen):
    assert screen.variable() == "fraction:grna[100_1]"
    curve = screen.curve()
    assert curve is not None
    assert len(curve) == screen_module.CURVE_POINTS
    assert curve.predictions[-1] > curve.predictions[0]


def test_a_zero_coefficient_input_is_listed_and_draws_a_flat_line(screen):
    variables = [r.variable for r in screen.ranked_inputs()]
    assert "fraction:grna[300_1]" in variables, (
        "'this gRNA does nothing' is a real answer a user may want to see")

    index = variables.index("fraction:grna[300_1]")
    screen._inputs.setCurrentItem(screen._inputs.topLevelItem(index))

    curve = screen.curve()
    assert curve.span == pytest.approx(0.0)
    assert screen._canvas.points(), "a flat curve must still be drawn"


def test_selecting_another_input_re_sweeps(screen, qtbot):
    with qtbot.waitSignal(screen.profiled, timeout=1000) as caught:
        screen._inputs.setCurrentItem(screen._inputs.topLevelItem(1))

    assert caught.args[0].variable == screen.variable()
    assert screen.curve().predictions[-1] < screen.curve().predictions[0], (
        "the second input has a negative coefficient")


def test_a_live_fitted_object_can_be_profiled_without_a_file(qtbot, design):
    import statsmodels.api as sm

    y = 1.0 + 3.0 * design["fraction:grna[100_1]"]
    model = sm.OLS(y, design).fit()
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_model(model, design=design)

    assert widget.model() is model, "the screen must not re-fit"
    assert widget.variable() == "fraction:grna[100_1]"
    assert widget.curve().predictions[-1] > widget.curve().predictions[0]


def test_a_broken_coefficient_table_reports_inline_and_never_modally(
        qtbot, tmp_path):
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"term": ["a"], "beta": [1.0]}).to_csv(path, index=False)
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.load_coefficients(str(path))

    assert widget.model() is None
    assert "coefficient" in widget.last_error
    assert widget._status.property("problem") == "true"


def test_a_model_with_no_inputs_says_so(qtbot, tmp_path):
    path = tmp_path / "intercept_only.csv"
    pd.DataFrame({"feature": ["Intercept"], "coefficient": [1.0]}).to_csv(
        path, index=False)
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.load_coefficients(str(path))

    assert widget.ranked_inputs() == []
    assert "nothing to sweep" in widget._status.text()
    assert "only an intercept" in widget._status.text(), (
        "an intercept-only model was never profilable; a constant design is "
        "a different problem with a different fix")
    assert widget._canvas.curve() is None


def test_a_design_whose_inputs_never_vary_says_the_other_thing(qtbot,
                                                               coefficients):
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_coefficients(coefficients)

    widget.set_design(pd.DataFrame({
        "Intercept": [1.0, 1.0],
        "fraction:grna[100_1]": [0.3, 0.3],
        "fraction:grna[200_1]": [0.2, 0.2],
        "fraction:grna[300_1]": [0.1, 0.1]}))

    assert widget.ranked_inputs() == []
    assert "every input is constant" in widget._status.text()


# ---------------------------------------------------------------------------
# Held values
# ---------------------------------------------------------------------------

def test_every_held_input_gets_a_slider_capped_at_the_maximum(screen):
    sliders = screen._sliders

    assert set(sliders) == {r.variable for r in screen.ranked_inputs()}
    assert len(sliders) <= screen_module.MAX_SLIDERS


def test_the_slider_wall_is_capped_on_a_wide_design(qtbot, tmp_path):
    path = tmp_path / "wide.csv"
    pd.DataFrame({
        "feature": [f"fraction:grna[{i}]" for i in range(40)],
        "coefficient": list(range(1, 41))}).to_csv(path, index=False)
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.load_coefficients(str(path))

    assert len(widget.ranked_inputs()) == 40
    assert len(widget._sliders) == screen_module.MAX_SLIDERS, (
        "a scroll area with forty sliders in it is a wall, not a control")


def test_moving_a_held_slider_moves_the_whole_curve(screen):
    before = screen.curve().predictions

    screen.set_held("fraction:grna[200_1]", 1.0)

    after = screen.curve().predictions
    shift = np.asarray(after) - np.asarray(before)
    assert shift.mean() < 0, "holding a negative input higher must lower it"
    assert shift.std() == pytest.approx(0.0, abs=1e-9), (
        "a held linear input shifts the curve, it does not tilt it")


def test_the_held_value_label_follows_the_slider(screen):
    screen.set_held("fraction:grna[200_1]", 0.75)

    assert screen.held_values()["fraction:grna[200_1]"] == pytest.approx(0.75)
    assert screen._slider_labels["fraction:grna[200_1]"].text() == "0.75"


def test_the_moving_input_is_not_also_held(screen):
    moving = screen.variable()

    assert moving in screen.held_values(), "it has a slider like the rest"
    assert moving not in screen.curve().held, (
        "but the sweep must override it, not fight the slider")


def test_holding_an_input_that_has_no_control_is_refused(screen):
    with pytest.raises(KeyError):
        screen.set_held("not_an_input", 1.0)


def test_reset_puts_every_held_value_back(screen):
    screen.set_held("fraction:grna[200_1]", 1.0)

    screen._on_reset()

    assert screen.held_values()["fraction:grna[200_1]"] == pytest.approx(0.5)


def test_a_real_design_makes_the_sweep_use_observed_ranges(qtbot,
                                                           coefficients,
                                                           design):
    widget = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(widget)
    assumed = widget.curve().values

    widget.set_design(design)

    observed = widget.curve().values
    assert assumed[-1] == pytest.approx(1.0), "the assumed range is 0-1"
    assert observed[-1] == pytest.approx(
        design["fraction:grna[100_1]"].max())
    assert "assumed range" not in widget._status.text()


def test_the_assumed_range_is_stated_when_there_is_no_design(screen):
    assert "assumed range" in screen._status.text(), (
        "sweeping 0-1 without saying so implies a measurement")


# ---------------------------------------------------------------------------
# The link
# ---------------------------------------------------------------------------

def test_the_link_is_a_control_and_changes_the_scale(screen):
    identity = screen.curve().predictions

    screen._link.setCurrentText("logit")

    logit = screen.curve().predictions
    assert all(0.0 <= p <= 1.0 for p in logit)
    assert logit != identity
    assert "probability" in screen.curve().scale
    assert "probability" in screen._status.text()


def test_every_link_the_profiler_offers_is_selectable(screen):
    from spacr.profiler import LINKS

    offered = {screen._link.itemText(i) for i in range(screen._link.count())}

    assert offered == set(LINKS)
    assert screen._link.isEnabled(), (
        "a model rebuilt from coefficients does not record its link, so the "
        "user has to supply it")


def test_a_live_model_carries_its_own_link_so_the_control_is_disabled(
        qtbot, design):
    """A control that changes nothing is worse than no control."""
    import statsmodels.api as sm

    y = 1.0 + 3.0 * design["fraction:grna[100_1]"]
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_model(sm.OLS(y, design).fit(), design=design)

    assert widget._link.isEnabled() is False
    assert "carries its own link" in widget._link.toolTip()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def test_the_curve_is_drawn_inside_the_canvas_and_rises_left_to_right(screen):
    screen._canvas.resize(400, 300)

    points = screen_module.curve_points(screen.curve(), 400, 300)

    assert len(points) == screen_module.CURVE_POINTS
    xs = [x for x, _ in points]
    assert xs == sorted(xs)
    assert all(0 <= x <= 400 and 0 <= y <= 300 for x, y in points)
    assert points[-1][1] < points[0][1], (
        "a rising prediction must be drawn upwards, i.e. at a smaller y")


def test_a_flat_curve_is_drawn_down_the_middle_not_divided_by_zero():
    flat = Profile("a", (0.0, 0.5, 1.0), (2.0, 2.0, 2.0))

    points = screen_module.curve_points(flat, 400, 300)

    assert len(points) == 3
    assert all(y == pytest.approx(points[0][1]) for _x, y in points)
    assert 0 < points[0][1] < 300


def test_an_absent_or_single_point_curve_draws_nothing():
    assert screen_module.curve_points(None, 400, 300) == []
    assert screen_module.curve_points(Profile("a", (0.0,), (1.0,)),
                                      400, 300) == []


def test_a_non_finite_prediction_is_skipped_rather_than_drawn():
    curve = Profile("a", (0.0, 0.5, 1.0), (1.0, float("nan"), 3.0))

    assert screen_module.curve_points(curve, 400, 300) == []


def test_the_screen_renders_at_the_window_size(screen, qt_theme_applied):
    screen.resize(1200, 720)
    screen.show()

    frame = screen.grab()

    assert not frame.isNull()
    assert frame.width() >= 1200 and frame.height() >= 720


def test_the_empty_canvas_paints_its_explanation(qtbot, qt_theme_applied):
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.resize(1200, 720)
    widget.show()

    assert not widget.grab().isNull()
    assert widget._canvas.curve() is None


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_the_threaded_path_loads_the_same_model_and_retires(qtbot,
                                                            coefficients):
    widget = screen_module.ProfilerScreen(threaded=True)
    qtbot.addWidget(widget)

    with qtbot.waitSignal(widget.model_loaded, timeout=15000):
        widget.load_coefficients(coefficients)

    assert isinstance(widget.model(), FittedLinear)
    assert len(widget.ranked_inputs()) == 3
    qtbot.waitUntil(lambda: widget.active_jobs() == 0, timeout=15000)
    assert widget.is_busy() is False
    widget.close()


def test_the_point_count_control_changes_the_curve_resolution(screen):
    screen._points.setValue(9)

    assert len(screen.curve()) == 9
    assert math.isfinite(screen.curve().span)
