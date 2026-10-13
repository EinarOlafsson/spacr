"""The profiler screen with no model, no inputs, or an input that cannot move.

A profiler answers "what does this input do to the prediction". Every path
here is one where that question has no answer, and the screen has to say which
kind of nothing it hit: no file chosen, a model with no inputs, a sweep the
profiler refused. Drawing an empty canvas with a stale status line instead
reads as a model that predicts nothing, which is a different and much more
alarming claim.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.profiler import FittedLinear, Profile  # noqa: E402
from spacr.qt.screens import profiler as screen_module  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def coefficients(tmp_path):
    folder = tmp_path / "results" / "pred" / "ols"
    folder.mkdir(parents=True)
    path = folder / "results.csv"
    pd.DataFrame({
        "feature": ["Intercept", "fraction:grna[100_1]",
                    "fraction:grna[200_1]"],
        "coefficient": [0.5, 2.4, -0.8],
    }).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def design():
    rng = np.random.default_rng(3)
    n = 40
    return pd.DataFrame({
        "Intercept": np.ones(n),
        "fraction:grna[100_1]": rng.uniform(0.0, 0.4, n),
        "fraction:grna[200_1]": rng.uniform(0.0, 1.0, n),
    })


def _model():
    return FittedLinear(params=pd.Series({
        "Intercept": 0.5,
        "fraction:grna[100_1]": 2.4,
        "fraction:grna[200_1]": -0.8,
    }))


@pytest.fixture()
def blank(qtbot):
    """A screen nobody has given a model to."""
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# the pure layout function
# ---------------------------------------------------------------------------

def test_a_sweep_that_never_moved_is_still_drawn_across_the_canvas():
    """An input pinned at one value must not divide the plot by zero.

    ``profile`` can be handed a variable whose observed range collapsed to a
    single value -- one plate, one concentration. The curve is degenerate but
    it is a real answer, and the canvas has to place its points somewhere
    rather than raise inside ``paintEvent``.
    """
    curve = Profile(variable="dose", values=(0.5, 0.5),
                    predictions=(1.0, 2.0))

    points = screen_module.curve_points(curve, 400, 300)

    assert len(points) == 2
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in points)
    # Both sit on the left edge of the plotting area: the input did not move.
    assert points[0][0] == points[1][0] == pytest.approx(36.0)


def test_a_non_finite_prediction_is_left_off_the_line(qapp):
    """A point the model could not produce must not be drawn at an edge.

    The length check above the loop tolerates a prediction list that carries
    more entries than the sweep as long as the finite ones line up, so the
    loop cannot assume every prediction it sees is plottable. An infinity
    mapped through the y-scale lands on the frame and reads as a real
    prediction at the extreme.
    """
    curve = Profile(variable="dose", values=(0.0, 1.0),
                    predictions=(1.0, float("inf"), 2.0))

    points = screen_module.curve_points(curve, 400, 300)

    assert len(points) == 1
    assert all(math.isfinite(y) for _x, y in points)


# ---------------------------------------------------------------------------
# a screen with nothing loaded
# ---------------------------------------------------------------------------

def test_a_screen_with_no_model_has_no_design_to_sweep(blank):
    """``design()`` must answer with an empty frame, not invent columns.

    It is called before a model is loaded -- the drop handler and the status
    line both reach it -- and synthesizing a range for inputs nobody has named
    would produce a design matrix for a model that does not exist.
    """
    assert blank.model() is None
    assert blank.design().empty
    assert "Choose a regression results.csv" in blank._status.text()


def test_loading_an_empty_path_asks_for_a_file_instead_of_reading_one(blank):
    """A cleared path box is not a request to load anything.

    The box is editable and the return key loads it, so an empty box arrives
    here whenever somebody clears it and presses enter.
    """
    blank.load_coefficients("   ")

    assert blank.model() is None
    assert blank._status.text() == "Choose a coefficient table."


def test_pressing_return_in_the_path_box_loads_what_was_typed(blank,
                                                              coefficients):
    """Typing a path and pressing return is the keyboard route into a load."""
    blank._path_edit.setText(coefficients)

    blank._path_edit.returnPressed.emit()

    assert blank.model() is not None
    assert blank.ranked_inputs()


# ---------------------------------------------------------------------------
# a model handed straight in
# ---------------------------------------------------------------------------

def test_a_screen_can_be_opened_on_a_model_and_a_design_directly(qtbot,
                                                                 design):
    """The seam a caller with a live fit already in hand uses.

    Going through a file would mean writing the coefficients out and reading
    them back, which loses the model's own link and its standard errors.
    """
    widget = screen_module.ProfilerScreen(model=_model(), design=design,
                                          threaded=False)
    qtbot.addWidget(widget)

    assert widget.model() is not None
    assert [record.variable for record in widget.ranked_inputs()]
    # The supplied design is what the sweeps run over, not a synthetic one.
    assert widget.design() is design


def test_a_model_with_no_inputs_says_there_is_nothing_to_profile(qtbot):
    """No coefficients at all is a different nothing from a flat design.

    The screen has to clear the input list and the canvas as well as saying
    so: leaving the previous model's inputs listed beside an empty canvas
    invites the reader to click one.
    """
    widget = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_model(FittedLinear(params=pd.Series(dtype=float)))

    assert widget.design().empty
    assert widget._status.text() == "That model has no inputs to profile."
    assert widget._inputs.topLevelItemCount() == 0
    assert widget._canvas.curve() is None
    assert widget.ranked_inputs() == []


# ---------------------------------------------------------------------------
# held-value sliders
# ---------------------------------------------------------------------------

def test_an_input_with_an_unbounded_value_gets_the_default_sweep_range(qtbot):
    """A slider cannot span an infinity, so it falls back to the default range.

    An infinite entry in a design column is ordinary in a measurements table:
    any ratio with a zero denominator produces one. The sensitivity ranking
    sweeps robust quantiles and is unaffected, so the input is ranked and a
    slider IS built for it -- over ``min()`` and ``max()``, one of which is
    infinite. Handing that to the slider gives a control whose every position
    is the same value.
    """
    rng = np.random.default_rng(11)
    n = 100
    unbounded = rng.uniform(0.0, 1.0, n)
    unbounded[-1] = float("inf")
    frame = pd.DataFrame({
        "Intercept": np.ones(n),
        "fraction:grna[100_1]": rng.uniform(0.0, 0.4, n),
        "fraction:grna[200_1]": unbounded,
    })
    widget = screen_module.ProfilerScreen(model=_model(), design=frame,
                                          threaded=False)
    qtbot.addWidget(widget)

    ranges = widget._slider_ranges
    assert "fraction:grna[200_1]" in ranges, "the input was not ranked at all"
    for name, (low, high) in ranges.items():
        assert math.isfinite(low) and math.isfinite(high), name
        assert low < high, f"{name} got a slider that cannot move"
    assert ranges["fraction:grna[200_1]"] == screen_module.DEFAULT_RANGE


def test_a_held_input_that_cannot_move_maps_every_value_to_the_first_step(
        qtbot, design):
    """A degenerate range must not divide by zero on the way to the slider.

    ``_to_step`` runs whenever a held value is set from outside -- restoring a
    saved workspace, or a caller pinning an input -- and the range it reads
    comes from whatever design was loaded.
    """
    widget = screen_module.ProfilerScreen(model=_model(), design=design,
                                          threaded=False)
    qtbot.addWidget(widget)
    widget._slider_ranges["pinned"] = (0.25, 0.25)

    assert widget._to_step("pinned", 0.25) == 0
    assert widget._to_step("pinned", 99.0) == 0


# ---------------------------------------------------------------------------
# the link, changed without a file
# ---------------------------------------------------------------------------

def test_changing_the_link_on_a_model_with_no_file_redraws_it(qtbot, design,
                                                              monkeypatch):
    """A model handed in directly has no path to re-read, so it is redrawn.

    Doing nothing would leave the curve drawn under the previous link while
    the combo showed the new one -- a plot that disagrees with the control
    beside it.
    """
    widget = screen_module.ProfilerScreen(model=_model(), design=design,
                                          threaded=False)
    qtbot.addWidget(widget)
    assert widget._path_edit.text().strip() == ""

    redrawn: list = []
    widget.profiled.connect(redrawn.append)

    widget._on_link_changed()

    assert len(redrawn) == 1
    assert widget.curve() is not None


def test_changing_the_link_with_no_model_and_no_file_does_nothing(blank):
    """Nothing loaded means nothing to redraw and nothing to re-read."""
    blank._on_link_changed()

    assert blank.curve() is None


# ---------------------------------------------------------------------------
# a sweep the profiler refuses
# ---------------------------------------------------------------------------

def test_a_sweep_the_profiler_refuses_is_reported_and_clears_the_canvas(
        qtbot, design, monkeypatch):
    """The refusal has to reach the pane, the canvas AND ``last_error``.

    The canvas keeps painting its last curve until it is told otherwise, so a
    failure that only updated the status line would leave the previous input's
    plot on screen under the new input's name.
    """
    widget = screen_module.ProfilerScreen(model=_model(), design=design,
                                          threaded=False)
    qtbot.addWidget(widget)
    assert widget.curve() is not None

    def refuse(*args, **kwargs):
        raise ValueError("held value for 'Intercept' is not a number")

    monkeypatch.setattr(screen_module, "profile", refuse)

    widget._redraw()

    assert widget.last_error == "held value for 'Intercept' is not a number"
    assert widget._canvas.curve() is None
    assert "is not a number" in widget._canvas._message
    assert "Could not profile" in widget._status.text()
    assert "is not a number" in widget._status.text()


def test_the_curve_accessor_reports_nothing_after_a_refused_sweep(qtbot,
                                                                  design,
                                                                  monkeypatch):
    """``curve()`` says it is "the profile currently drawn", so it must be None.

    After a refusal the canvas is cleared and the pane says the sweep failed,
    but the accessor still hands back the previous input's profile. A host
    that exports "the current curve" -- the figure grid does exactly this --
    would then save a plot the screen is not showing, under the name of the
    input that failed.
    """
    widget = screen_module.ProfilerScreen(model=_model(), design=design,
                                          threaded=False)
    qtbot.addWidget(widget)

    def refuse(*args, **kwargs):
        raise ValueError("held value for 'Intercept' is not a number")

    monkeypatch.setattr(screen_module, "profile", refuse)
    widget._redraw()

    assert widget.curve() is None
