"""Rows with no p value are explained, not shown as an empty graph.

Reported 2026-08-21 against a run that had worked: "i ran mixed parametric
with torch and it worked! ... but with guides i see nothing in the graph"
plus three `RuntimeWarning: All-NaN slice encountered`.

NOTHING WAS BROKEN. A mixed model makes the guide a RANDOM effect, so each
guide gets a shrunken BLUP -- a prediction -- and a BLUP has no p value. A
volcano's vertical axis IS the p value, so it has nothing to draw. The run
already said so in the console; the panel the user was looking at did not,
and the concordance table warned about the absence three times as though it
were a fault.

TWO CASES, TOLD APART, because they ask different things of the reader: a
mixed fit's guide rows never had a p value and that is expected; anything
else with none is a fit that failed to produce one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestTheConcordanceStopsWarning:

    def test_an_all_nan_gene_is_answered_quietly(self):
        """The absence is the ordinary case for a mixed fit, and warning
        about it is noise that hides real warnings."""
        import warnings

        from spacr.guide_concordance import _best_p

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert np.isnan(_best_p([np.nan, np.nan, np.nan]))

    def test_it_still_finds_the_best_real_p(self):
        from spacr.guide_concordance import _best_p

        assert _best_p([np.nan, 0.03, 0.5]) == pytest.approx(0.03)

    def test_an_empty_list_is_nan(self):
        from spacr.guide_concordance import _best_p

        assert np.isnan(_best_p([]))

    def test_infinities_do_not_count_as_a_p_value(self):
        from spacr.guide_concordance import _best_p

        assert np.isnan(_best_p([np.inf, -np.inf]))


@pytest.fixture
def panel(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    said = []
    widget.say = lambda text, *a, **k: said.append(str(text))
    widget._said = said
    return widget


def _blups(n=5):
    return pd.DataFrame({
        "feature": [f"grna[{i}]" for i in range(n)],
        "coefficient": np.random.default_rng(0).normal(size=n),
        "p_value": [np.nan] * n,
        "term_type": ["random_effect_blup"] * n})


@pytest.mark.qt
class TestThePanelExplainsIt:

    def test_a_blup_table_says_why_the_volcano_is_empty(self, panel):
        panel._say_if_no_p_values(_blups())
        said = " ".join(panel._said)
        assert "BLUP" in said
        assert "random" in said.lower()

    def test_it_says_where_the_numbers_are_instead(self, panel):
        """An explanation that ends at "there is nothing" sends the user
        away from a table that has what they wanted."""
        panel._say_if_no_p_values(_blups())
        assert "table" in " ".join(panel._said).lower()

    def test_it_names_a_way_to_get_per_guide_significance(self, panel):
        panel._say_if_no_p_values(_blups())
        said = " ".join(panel._said)
        assert "nonparametric" in said or "fixed" in said

    def test_a_fit_that_simply_failed_gets_a_different_message(self, panel):
        """Expected absence and a broken fit must not read the same."""
        panel._say_if_no_p_values(_blups().assign(term_type="fixed"))
        said = " ".join(panel._said)
        assert "BLUP" not in said
        assert "did not produce" in said

    def test_a_frame_with_p_values_is_not_commented_on(self, panel):
        panel._say_if_no_p_values(_blups().assign(p_value=0.01))
        assert panel._said == []

    def test_a_frame_with_some_p_values_is_not_commented_on(self, panel):
        frame = _blups()
        frame.loc[0, "p_value"] = 0.02
        panel._say_if_no_p_values(frame)
        assert panel._said == []

    def test_a_frame_without_the_column_is_left_alone(self, panel):
        panel._say_if_no_p_values(_blups().drop(columns=["p_value"]))
        assert panel._said == []


class TestTheRefitCanReachAFixedEffectGuideFit:
    """The answer to the empty guide volcano has to be reachable from it.

    A mixed fit makes the guide a RANDOM effect, so its guide rows are
    shrunken predictions with no p value. The question that follows is "how
    do I get a p value per guide", and the answer is a fixed-effect fit at
    guide level -- which the re-fit dialog could not express, because it
    changed the model but not the LEVEL.
    """

    @staticmethod
    def _base():
        return {"count_data": ["counts.csv"], "level": "both",
                "regression_type": "mixed"}

    def test_the_level_can_be_changed(self):
        from spacr.refit import refit_settings

        settings, notes = refit_settings(self._base(), level="grna",
                                         regression_type="ols")
        assert settings["level"] == "grna"
        assert settings["regression_type"] == "ols"
        assert any("level" in note for note in notes)

    def test_an_unchanged_level_is_not_reported_as_a_change(self):
        from spacr.refit import refit_settings

        _settings, notes = refit_settings(self._base(), level="both")
        assert not any("level" in note for note in notes)

    def test_an_unknown_level_is_refused_while_the_dialog_is_open(self):
        from spacr.refit import refit_settings

        with pytest.raises(ValueError, match="level="):
            refit_settings(self._base(), level="guides")

    def test_guide_level_mixed_is_told_it_still_has_no_p_values(self):
        """The trap: asking for level='grna' does NOT turn a mixed fit's
        BLUPs into estimates, because 'mixed' makes the guide a random
        effect at every level. Without this note the re-fit returns the same
        empty volcano under a new folder name."""
        from spacr.refit import refit_settings

        _settings, notes = refit_settings(self._base(), level="grna")
        said = " ".join(notes)
        assert "BLUP" in said
        assert "fixed-effect" in said or "nonparametric" in said

    def test_a_fixed_effect_guide_fit_gets_no_such_warning(self):
        from spacr.refit import refit_settings

        _settings, notes = refit_settings(self._base(), level="grna",
                                          regression_type="ols")
        assert not any("BLUP" in note for note in notes)


@pytest.mark.qt
class TestTheDialogOffersIt:

    def test_the_level_control_exists_and_lists_every_level(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.refit_dialog import RefitDialog
        from spacr.settings import REGRESSION_LEVELS

        dialog = RefitDialog({"count_data": ["c.csv"], "level": "both",
                              "regression_type": "mixed"})
        qtbot.addWidget(dialog)

        offered = {dialog._level.itemData(i)
                   for i in range(dialog._level.count())}
        assert offered == {None, *REGRESSION_LEVELS}

    def test_it_opens_on_the_level_the_run_used(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.refit_dialog import RefitDialog

        dialog = RefitDialog({"count_data": ["c.csv"], "level": "gene",
                              "regression_type": "ols"})
        qtbot.addWidget(dialog)
        assert dialog._level.currentData() == "gene"

    def test_choosing_a_level_reaches_the_settings(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets.refit_dialog import RefitDialog

        dialog = RefitDialog({"count_data": ["c.csv"], "level": "both",
                              "regression_type": "mixed"})
        qtbot.addWidget(dialog)
        dialog._level.setCurrentIndex(dialog._level.findData("grna"))
        dialog._type.setCurrentIndex(dialog._type.findData("ols"))

        settings, _notes = dialog.settings()
        assert settings["level"] == "grna"
        assert settings["regression_type"] == "ols"
