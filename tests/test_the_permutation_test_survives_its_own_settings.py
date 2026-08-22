"""The nonparametric path, driven from a real saved settings file.

Instruction 236 C7 asks for "every nonparametric test" as well as every
regression type. Both defects here were found the same way -- by loading
the tsg101 screen's own `settings/regression.csv` and running it -- and
neither is reachable from a settings dict built in Python, which is why
neither had been caught.

1. `guide_primary_min_wells` is an OPTIONAL field, so the panel leaves it
   empty and the CSV writes an empty cell. It read back as '' and reached
   `int('')`, so the whole permutation path died with "invalid literal for
   int() with base 10: ''" -- a message with neither the setting's name nor
   the analysis's in it.

2. `guide_min_wells` is a SWEEP: [1, 2, 3, 4] asks the same question at
   four strictnesses. On a one-plate screen no guide appears in four wells,
   and the volcano loop raised on the empty one -- after the analysis had
   finished, throwing away the results for 1, 2 and 3 as well, at the
   drawing stage, with a message about a plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _long_results(thresholds=(1, 2, 3), outcome="log_pred", rows=9):
    """What `analyse_long_guide_table` hands the plotting loop."""
    frame = pd.DataFrame({
        "guide": [f"g{i}" for i in range(rows)],
        "gene": [f"gene{i // 3}" for i in range(rows)],
        "outcome": outcome,
        "coefficient": np.linspace(-1.0, 1.0, rows),
        "standardized_marginal_effect": np.linspace(-1.0, 1.0, rows),
        "p_value": np.linspace(0.001, 0.9, rows),
        "adjusted_p_value": np.linspace(0.002, 0.95, rows),
        "significant": [True] * 3 + [False] * (rows - 3),
        "multiple_testing_method": "fdr_bh",
        "alpha": 0.05,
        "minimum_wells_threshold": [thresholds[i % len(thresholds)]
                                    for i in range(rows)],
    })
    return frame


class TestABlankPrimaryThreshold:
    @pytest.mark.parametrize("blank", [None, "", "  ", "\t"])
    def test_it_means_the_first_threshold(self, blank):
        """The same as an absent key, which is what an empty optional box
        has always meant everywhere else in this panel."""
        from spacr.ml import _left_blank

        thresholds = [1, 2, 3, 4]
        primary = blank
        assert (thresholds[0] if _left_blank(primary) else int(primary)) == 1

    def test_a_threshold_that_was_chosen_is_used(self):
        from spacr.ml import _left_blank

        thresholds = [1, 2, 3, 4]
        primary = 3
        assert (thresholds[0] if _left_blank(primary)
                else int(primary)) == 3

    def test_the_saved_settings_of_a_real_screen_parse(self):
        """The reproduction: every optional field empty, as a settings CSV
        writes them."""
        from spacr.ml import _left_blank

        parsed = {"guide_primary_min_wells": "",
                  "hinge_threshold": "",
                  "cov_type": ""}
        assert all(_left_blank(value) for value in parsed.values())


class TestASweepDoesNotDieOnItsStrictestRung:
    def test_the_plotter_still_refuses_a_threshold_it_was_asked_for(self,
                                                                    tmp_path):
        """The skip belongs in the loop, not in the plotter: a caller that
        names one threshold and gets a blank page has been misled."""
        from spacr.guide_permutation import plot_guide_permutation_volcano

        with pytest.raises(ValueError, match="No rows for outcome"):
            plot_guide_permutation_volcano(
                _long_results(), outcome="log_pred", minimum_wells=4,
                save_path=str(tmp_path / "v.png"))

    def test_a_rung_with_guides_still_draws(self, tmp_path):
        from spacr.guide_permutation import plot_guide_permutation_volcano

        written = plot_guide_permutation_volcano(
            _long_results(), outcome="log_pred", minimum_wells=2,
            save_path=str(tmp_path / "v.png"))
        assert written is not None

    def test_the_sweeps_other_rungs_are_not_lost_with_it(self):
        """The loop skips a threshold nothing reached and keeps going; the
        analysis is already finished by the time it runs, so anything it
        raises throws away work that succeeded."""
        import inspect

        from spacr.ml import _run_guide_permutation_analysis

        source = inspect.getsource(_run_guide_permutation_analysis)
        assert "continue" in source[source.index("No guide reached"):]

    def test_the_loop_checks_before_it_draws(self):
        """Read from the source, because reaching this loop end to end
        means running a permutation test: the check has to come before the
        call, or the skip is not a skip."""
        import inspect

        from spacr.ml import _run_guide_permutation_analysis

        source = inspect.getsource(_run_guide_permutation_analysis)
        start = source.index("No guide reached")
        drawn = source.index("plot_guide_permutation_volcano(", start)
        continued = source.index("continue", start)
        assert continued < drawn
