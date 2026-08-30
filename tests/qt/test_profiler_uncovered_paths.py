"""Edges of the Prediction Profiler screen that the happy path never reaches.

A design supplied before any model exists, a model that came back empty, the
browse dialog's two answers, a slider left over from a previous model, and a
status strip whose style cannot be resolved.

Every screen runs ``threaded=False`` so a load has finished when the call
returns. Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import profiler as screen_module           # noqa: E402

pytestmark = pytest.mark.qt


def _write_coefficients(folder, features, coefficients):
    """Write a coefficient table the screen can read, return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "results.csv"
    pd.DataFrame({
        "feature": features,
        "coefficient": coefficients,
    }).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def coefficients(tmp_path):
    """A table with one strong input and one weak one."""
    return _write_coefficients(
        tmp_path / "run_a" / "ols",
        ["Intercept", "fraction:grna[100_1]", "fraction:grna[200_1]"],
        [0.5, 2.4, -0.8],
    )


@pytest.fixture()
def other_coefficients(tmp_path):
    """A second table naming entirely different inputs."""
    return _write_coefficients(
        tmp_path / "run_b" / "ols",
        ["Intercept", "fraction:grna[900_9]", "fraction:grna[800_8]"],
        [0.1, 3.1, -1.2],
    )


# ---------------------------------------------------------------------------
# A design with nothing to apply it to
# ---------------------------------------------------------------------------

def test_a_design_supplied_before_any_model_draws_no_curve(qtbot):
    """set_design on an empty screen stores the design and stays empty."""
    screen = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.model() is None

    frame = pd.DataFrame({"fraction:grna[100_1]": [0.0, 0.5, 1.0]})
    screen.set_design(frame)

    assert screen.model() is None
    assert screen.curve() is None
    assert screen.ranked_inputs() == []
    # The design was kept, so the next model loaded sweeps the observed range.
    assert list(screen.design().columns) == ["fraction:grna[100_1]"]


def test_a_design_supplied_after_a_model_reprofiles_over_the_observed_range(
        qtbot, coefficients):
    """The same call, with a model present, redraws against the design."""
    screen = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(screen)
    assumed = screen.curve()
    assert assumed is not None
    assert assumed.values[-1] == pytest.approx(1.0)

    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "Intercept": np.ones(40),
        "fraction:grna[100_1]": rng.uniform(0.0, 0.25, 40),
        "fraction:grna[200_1]": rng.uniform(0.0, 0.25, 40),
    })
    screen.set_design(frame)

    observed = screen.curve()
    assert observed is not None
    assert observed.values[-1] < 0.3, (
        "the sweep should follow the design, not the assumed 0-1 range")


# ---------------------------------------------------------------------------
# A model that never arrived
# ---------------------------------------------------------------------------

def test_a_model_that_could_not_be_read_says_so_and_profiles_nothing(
        qtbot, coefficients):
    """A ``None`` model clears the screen and marks the strip as a problem."""
    screen = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(screen)
    assert screen.model() is not None

    screen.set_model(None)

    assert screen.model() is None
    assert screen._status.text() == "The model could not be read."
    assert screen._status.property("problem") == "true"


# ---------------------------------------------------------------------------
# Browsing for a table
# ---------------------------------------------------------------------------

def test_browsing_to_a_table_loads_the_model_it_describes(
        qtbot, monkeypatch, coefficients):
    """A path chosen in the file dialog is loaded like a typed one."""
    screen = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(screen)
    monkeypatch.setattr(screen_module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (coefficients, "CSV")))

    screen._browse_button.click()

    assert screen._path_edit.text() == coefficients
    assert screen.model() is not None
    assert screen.variable() == "fraction:grna[100_1]"


def test_a_cancelled_browse_leaves_the_screen_as_it_was(
        qtbot, monkeypatch, coefficients):
    """Dismissing the dialog loads nothing and keeps the model on screen."""
    screen = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(screen)
    loaded = screen.model()
    monkeypatch.setattr(screen_module.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))

    screen._browse_button.click()

    assert screen._path_edit.text() == coefficients
    assert screen.model() is loaded


# ---------------------------------------------------------------------------
# A slider that outlived its model
# ---------------------------------------------------------------------------

def test_a_slider_from_a_previous_model_moves_without_a_value_label(
        qtbot, coefficients, other_coefficients):
    """A stale slider's signal is absorbed rather than raising."""
    screen = screen_module.ProfilerScreen(coefficients=coefficients,
                                          threaded=False)
    qtbot.addWidget(screen)
    stale_name = "fraction:grna[200_1]"
    stale_slider = screen._sliders[stale_name]

    screen.load_coefficients(other_coefficients)
    assert stale_name not in screen._slider_labels, (
        "the new model's sliders replaced the old ones")

    stale_slider.setValue(screen_module.SLIDER_STEPS)

    # The signal still ran the slot; no label existed to write to.
    assert screen.held_values()[stale_name] == pytest.approx(1.0)
    assert screen.variable() == "fraction:grna[900_9]", (
        "the live model is still the one being profiled")


# ---------------------------------------------------------------------------
# A status strip with no style
# ---------------------------------------------------------------------------

def test_a_status_strip_with_no_style_still_reports_the_failure(
        qtbot, monkeypatch):
    """When the style cannot be resolved the text and flag are still set."""
    screen = screen_module.ProfilerScreen(threaded=False)
    qtbot.addWidget(screen)
    monkeypatch.setattr(type(screen._status), "style",
                        lambda self: None, raising=False)

    screen._on_job_failed("the file is not a coefficient table")

    assert screen.last_error == "the file is not a coefficient table"
    assert "not a coefficient table" in screen._status.text()
    assert screen._status.property("problem") == "true"


# ---------------------------------------------------------------------------
# The registry factory
# ---------------------------------------------------------------------------

def test_the_registry_factory_builds_an_empty_profiler_screen(qtbot):
    """The factory the app registry calls returns a usable screen."""
    widget = screen_module.make_profiler_screen("prediction_profiler")
    qtbot.addWidget(widget)

    assert isinstance(widget, screen_module.ProfilerScreen)
    assert widget.model() is None
    assert widget.curve() is None
