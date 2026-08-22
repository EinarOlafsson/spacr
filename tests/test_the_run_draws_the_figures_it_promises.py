"""Two figures the run promised and did not draw.

Reported 2026-08-21: "in the figure view i nevers ee the frna threhsold
graph i dont see the distribution graph to the right in the figure panels".

BOTH FAILED SILENTLY, and in the two ways this codebase keeps meeting: one
was behind a gate almost nobody is on the right side of, and the other
returned early on a name lookup and swallowed the reason.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest


class TestTheThresholdSweepIsAlwaysDrawn:
    """It was drawn only when `fraction_threshold is None`, and the default
    is 0.02 -- so the one case it fired in was the case nobody is in."""

    def test_the_run_draws_it_when_a_threshold_is_set(self):
        from spacr import ml

        # THE MODULE, NOT THE FUNCTION. `perform_regression` is decorated,
        # so `inspect.getsource` hands back the wrapper -- which contains
        # none of the body this is about.
        from spacr import ml

        source = pathlib.Path(ml.__file__).read_text()
        assert "_draw_the_threshold_sweep(settings, res_folder)" in source, (
            "the run still only prints a sentence explaining why there is "
            "no graph when a threshold is set")
        # And the sentence that replaced the graph is gone.
        assert "sweep graph is not drawn" not in source

    def test_the_helper_does_not_touch_the_setting(self):
        """The user's number stands; the curve says where it sits."""
        from spacr import ml

        source = inspect.getsource(ml._draw_the_threshold_sweep)
        assert "settings['fraction_threshold'] =" not in source
        assert "_AUTOMATIC_SETTINGS" not in source

    def test_it_says_where_the_chosen_value_sits(self):
        from spacr import ml

        source = inspect.getsource(ml._draw_the_threshold_sweep)
        assert "the one in force" in source

    def test_a_drawing_failure_is_not_a_run_failure(self):
        from spacr import ml

        source = inspect.getsource(ml._draw_the_threshold_sweep)
        assert "except Exception" in source
        assert "the run is unaffected" in source


class TestTheDistributionPanelFindsItsColumn:
    """`process_scores` RENAMES the response, so a straight lookup of the
    name it returns missed the untransformed frame every time."""

    @pytest.fixture
    def renamed(self):
        """What `process_scores` hands back: a differently named column."""
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "prc": [f"p1_r1_c{i}" for i in range(20)],
            "pathogen_count_mean": rng.lognormal(0.0, 1.0, 20),
        })

    def test_it_finds_the_column_under_another_name(self, renamed, capsys):
        from spacr import ml

        ml._show_response_distribution(renamed, "pathogen_count",
                                       {"transform": "log", "plot": True})
        printed = capsys.readouterr().out
        assert "was not drawn" not in printed
        assert "could not be drawn" not in printed

    def test_it_says_so_when_there_is_no_response_at_all(self, capsys):
        """A bare `pass` is why this went missing without a word."""
        from spacr import ml

        ml._show_response_distribution(
            pd.DataFrame({"prc": ["a", "b"]}), "nope",
            {"transform": "none", "plot": True})
        assert "no numeric response column" in capsys.readouterr().out

    def test_it_is_silent_when_plotting_is_off(self, renamed, capsys):
        from spacr import ml

        ml._show_response_distribution(renamed, "pathogen_count",
                                       {"transform": "log", "plot": False})
        assert capsys.readouterr().out == ""

    def test_a_failure_is_reported_rather_than_swallowed(self):
        from spacr import ml

        source = inspect.getsource(ml._show_response_distribution)
        assert "except Exception as error" in source
        assert "could not be drawn" in source
        assert "\\n        pass" not in source
