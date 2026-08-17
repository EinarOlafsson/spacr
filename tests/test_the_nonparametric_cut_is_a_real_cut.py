"""A permutation run has an effect-size cut, and it says what it is.

Asked 2026-08-17: "why cant i see the coefficient threshold if im running
nonparametric regression?"

The answer given at the time was that greying the `threshold_method` and
`threshold_multiplier` controls under `inference='nonparametric'` is correct
design, because the permutation path calls hits on corrected P values. THAT
ANSWER IS WRONG, and these tests pin why.

A P value says an effect is distinguishable from zero. An effect-size cut
says it is big enough to be worth an experiment. The second question is about
the COEFFICIENT, and the permutation table has a real one for every guide --
`standardized_marginal_effect`, aliased to `coefficient` -- 1,726 of them on
the screen this was reported from. How the P value was obtained does not
change how wide a control guide's effect is.

Two things were missing and both are here:

  * the table carried no `condition` column, so the results panel answered
    "No control coefficients, so no effect-size cut." for every permutation
    run -- measured, in `test_a_table_without_condition_is_why_the_panel_...`;
  * `perform_regression` RETURNED from the guide-permutation branch before
    reaching the parametric branch that computes the cut, so the run computed
    none, printed none and drew none.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr.ml import _run_guide_permutation_analysis, label_control_condition


CONTROLS = [f"000000_{index}" for index in range(1, 11)]


def _screen(seed=5, hit_strength=1.0, small_strength=0.6, noise=0.25,
            wells_per_plate=48):
    """A two-plate screen with ten controls, one wide hit and one narrow one.

    Tuned deliberately: with `threshold_method='std'` and a multiplier of 3
    the cut lands at 0.34, `HIT_g1` comes out at 0.42 and `SMALL_g1` at 0.32.
    Both pass BH correction, so the narrow one is a guide the P value calls
    and the effect-size cut removes -- which is the case the whole feature
    exists for, and a fixture where the cut removed nothing would pass while
    proving nothing.
    """
    rng = np.random.default_rng(seed)
    wells = [f"p{plate}_w{well}" for plate in (1, 2)
             for well in range(wells_per_plate)]
    blocks = dict(zip(wells, np.repeat(["p1", "p2"], wells_per_plate)))
    guides = ["HIT_g1", "SMALL_g1"] + CONTROLS
    rows = []
    for well in wells:
        for guide in guides:
            rows.append({"prc": well, "grna": guide,
                         "fraction": float(rng.uniform(0.05, 0.4)),
                         "plateID": blocks[well]})
    frame = pd.DataFrame(rows)
    hit = frame.loc[frame.grna == "HIT_g1"].set_index("prc")["fraction"]
    small = frame.loc[frame.grna == "SMALL_g1"].set_index("prc")["fraction"]
    phenotype = (hit_strength * hit + small_strength * small
                 + rng.normal(0, noise, len(wells)))
    frame["score"] = frame["prc"].map(phenotype)
    return frame


def _run(tmp_path, **overrides):
    settings = {
        "guide_min_wells": [1], "guide_primary_min_wells": 1,
        "guide_permutations": 1999, "guide_permutation_seed": 3,
        "multiple_testing_method": "fdr_bh", "fdr_alpha": 0.05,
        "controls": CONTROLS, "threshold_method": "std",
        "threshold_multiplier": 3.0, "guide_permutation_plot": False,
    }
    settings.update(overrides)
    return _run_guide_permutation_analysis(
        _screen(), "score", str(tmp_path), settings)


# --------------------------------------------------------------------------- #
#  The premise: a table with no `condition` is why the panel said no
# --------------------------------------------------------------------------- #

@pytest.mark.qt
def test_a_table_without_condition_is_why_the_panel_refuses_a_cut(qtbot):
    """The panel's refusal was never about the inference being nonparametric.

    It reads `condition` to find the controls, and a table that does not
    carry the column gets one sentence: "No control coefficients". That is
    what a guide-permutation run produced, and it is indistinguishable, on
    screen, from a screen that genuinely has no controls.
    """
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{name}]" for name in CONTROLS],
        "grna": CONTROLS,
        "coefficient": np.linspace(-0.4, 0.4, len(CONTROLS)),
        "p_value": np.linspace(0.01, 0.9, len(CONTROLS)),
    })
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    panel.set_threshold_method("std")
    assert panel.status_text() == (
        "No control coefficients, so no effect-size cut.")

    # The SAME numbers, with the column the run now writes.
    labelled = frame.assign(
        condition=label_control_condition(frame["feature"], frame["grna"],
                                          controls=CONTROLS))
    panel.set_frame(labelled)
    panel.set_threshold_method("std")
    said = panel.status_text()
    assert said.startswith("Effect-size cut ")
    assert f"std of {len(CONTROLS)} controls" in said


# --------------------------------------------------------------------------- #
#  The labeller both paths share
# --------------------------------------------------------------------------- #

def test_the_labeller_marks_nc_pc_control_and_other():
    """One vocabulary for "what counts as a control", used by the parametric
    coefficient table and the permutation table alike."""
    features = pd.Series([
        "fraction:grna[000000_1]", "fraction:grna[NEG]",
        "fraction:grna[POS]", "fraction:grna[TGGT1_123]"])
    guides = pd.Series(["000000_1", "NEG", "POS", "TGGT1_123"])
    labels = label_control_condition(features, guides, nc="NEG", pc="POS",
                                     controls=["000000_1"])
    assert list(labels) == ["control", "nc", "pc", "other"]


def test_the_labeller_survives_a_control_list_that_round_tripped_as_integers():
    """A gene id typed into the GUI comes back out of a settings CSV as an
    int. The inline version this replaced compared it to a string guide and
    silently labelled nothing, which is one of the two ways `condition`
    collapses to a single value."""
    features = pd.Series(["fraction:grna[233460]", "fraction:grna[999]"])
    guides = pd.Series(["233460", "999"])
    labels = label_control_condition(features, guides, controls=[233460])
    assert list(labels) == ["control", "other"]


def test_a_control_free_screen_labels_everything_other_instead_of_raising():
    """`controls=None` is what perform_regression documents for a screen with
    no non-targeting guides. The inline version did `row['grna'] in controls`
    and raised TypeError on it."""
    features = pd.Series(["fraction:grna[a]", "fraction:grna[b]"])
    labels = label_control_condition(features, pd.Series(["a", "b"]),
                                     controls=None)
    assert list(labels) == ["other", "other"]


def test_nc_wins_over_pc_and_over_the_control_list():
    """A guide named in two places is reported once, and always the same way,
    or two panels reading the same table disagree about the null."""
    features = pd.Series(["fraction:grna[000000_1]"])
    labels = label_control_condition(features, pd.Series(["000000_1"]),
                                     nc="000000_1", pc="000000_1",
                                     controls=["000000_1"])
    assert list(labels) == ["nc"]


# --------------------------------------------------------------------------- #
#  The run computes, records and reports the cut
# --------------------------------------------------------------------------- #

def test_the_permutation_run_computes_a_cut_and_names_the_rule(tmp_path, capsys):
    """The number alone is not reportable. `coefficient_threshold` returns the
    sentence beside it and the run prints both."""
    output = _run(tmp_path)
    printed = capsys.readouterr().out

    assert output["effect_size_threshold"] > 0
    assert output["effect_size_rule"] == (
        f"3x std of {len(CONTROLS)} controls = "
        f"{output['effect_size_threshold']:.3g}")
    assert f"Effect-size cut: {output['effect_size_rule']}" in printed


def test_the_cut_is_measured_only_on_the_controls(tmp_path):
    """Measured on every guide it would be pulled up by the very hits the
    screen exists to find. `spacr.thresholds` says the controls ARE the null,
    and this is where that sentence has to be true."""
    from spacr.thresholds import coefficient_threshold

    output = _run(tmp_path)
    primary = output["primary"]
    controls = primary.loc[primary["condition"] == "control", "coefficient"]
    expected, _rule = coefficient_threshold(controls, "std", 3.0)
    assert output["effect_size_threshold"] == pytest.approx(expected)
    assert len(controls) == len(CONTROLS)


def test_every_one_of_the_seven_methods_reaches_the_permutation_run(tmp_path):
    """The run and the plot's right-click menu offer the SAME seven, because
    both go through `spacr.thresholds`. A method the run cannot honour is a
    menu entry that silently means something else."""
    from spacr.thresholds import METHODS

    seen = {}
    for index, method in enumerate(METHODS):
        output = _run(tmp_path / f"m{index}", threshold_method=method)
        seen[method] = output["effect_size_threshold"]
    assert seen["none"] is None
    assert all(value > 0 for name, value in seen.items() if name != "none")
    # `var` is in squared units, so it is a DIFFERENT number from `std` --
    # which is the whole reason spacr.thresholds calls it dimensionally odd.
    assert seen["var"] != pytest.approx(seen["std"])


def test_the_cut_is_written_into_the_results_csv_row_by_row(tmp_path):
    """A cut a reader cannot recompute from the file is a cut they cannot put
    in a methods section."""
    output = _run(tmp_path)
    table = pd.read_csv(output["paths"]["results"])
    assert table["effect_size_threshold"].tolist() == pytest.approx(
        [output["effect_size_threshold"]] * len(table))
    expected = table["coefficient"].abs() >= output["effect_size_threshold"]
    assert list(table["passes_effect_size"].astype(bool)) == list(expected)


def test_a_hit_has_to_clear_both_bars(tmp_path):
    """Corrected P below alpha AND an effect at least as wide as the cut --
    the same rule the parametric hit list applies. A guide that passes
    correction on a tiny effect is detectable, not worth an experiment."""
    output = _run(tmp_path)
    significant = output["significant"]
    primary = output["primary"]

    assert not significant.empty, "the strong hit should survive both bars"
    assert significant["significant"].all()
    assert (significant["coefficient"].abs()
            >= output["effect_size_threshold"]).all()
    # And it is a real filter here, not a no-op: the weak guide is called by
    # the correction and removed by the cut.
    called = primary.loc[primary["significant"].astype(bool), "guide"]
    assert set(called) - set(significant["guide"]) == {"SMALL_g1"}


def test_no_controls_means_no_cut_and_the_same_hits_as_before(tmp_path, capsys):
    """A screen with nothing to measure a null on gets no cut, says so, and
    calls exactly the hits the correction called. Never a silent zero, which
    would exclude every guide."""
    output = _run(tmp_path, controls=None)
    printed = capsys.readouterr().out

    assert output["effect_size_threshold"] is None
    assert "not enough to measure a spread" in output["effect_size_rule"]
    assert f"Effect-size cut: {output['effect_size_rule']}" in printed
    assert output["primary"]["passes_effect_size"].all()
    assert len(output["significant"]) == int(
        output["primary"]["significant"].sum())


def test_threshold_method_none_leaves_significance_alone(tmp_path):
    """'none' is a supported choice, not a missing setting: significance
    alone decides, and every guide keeps its row."""
    output = _run(tmp_path, threshold_method="none")
    assert output["effect_size_threshold"] is None
    assert output["effect_size_rule"] == "no effect-size cut"
    assert output["primary"]["passes_effect_size"].all()


def test_an_unsupported_method_names_the_ones_that_exist(tmp_path):
    """It refuses rather than falling back to a default the user did not ask
    for -- a run that quietly cut somewhere else is unreportable."""
    with pytest.raises(ValueError, match="Unsupported threshold method"):
        _run(tmp_path, threshold_method="vibes")


def test_the_cut_is_measured_on_the_primary_family_only(tmp_path):
    """The same guide appears once per minimum-wells threshold with an
    identical coefficient. Pooling the families would count each control up
    to four times and shrink the spread the cut is built from."""
    from spacr.thresholds import coefficient_threshold

    output = _run_guide_permutation_analysis(
        _screen(), "score", str(tmp_path), {
            "guide_min_wells": [1, 2, 3, 4], "guide_primary_min_wells": 3,
            "guide_permutations": 199, "guide_permutation_seed": 3,
            "multiple_testing_method": "fdr_bh", "fdr_alpha": 0.05,
            "controls": CONTROLS, "threshold_method": "std",
            "threshold_multiplier": 3.0, "guide_permutation_plot": False,
        })
    # `families`, not `results`. They were the same frame until 2026-08-17,
    # when `results` became the PRIMARY family only -- because handing the
    # stacked frame to the panel drew every guide once per minimum-wells
    # family, four times over, which the maintainer reported seeing on the
    # real screen. The full frame is still returned, under its own name.
    results = output["families"]
    assert results["minimum_wells_threshold"].nunique() == 4
    # And the panel's frame has each guide exactly once, which is the fix.
    assert output["results"]["guide"].is_unique

    primary_controls = results.loc[
        (results["minimum_wells_threshold"] == 3)
        & (results["condition"] == "control"), "coefficient"]
    expected, _rule = coefficient_threshold(primary_controls, "std", 3.0)
    assert output["effect_size_threshold"] == pytest.approx(expected)

    pooled = results.loc[results["condition"] == "control", "coefficient"]
    pooled_cut, _ = coefficient_threshold(pooled, "std", 3.0)
    assert len(pooled) == 4 * len(primary_controls)
    assert pooled_cut != pytest.approx(expected)


def test_the_multiplier_moves_the_cut(tmp_path):
    """It is "how many spreads wide", so doubling it has to widen the cut --
    the control that was greyed out is the one that does this."""
    narrow = _run(tmp_path / "narrow", threshold_multiplier=1.0)
    wide = _run(tmp_path / "wide", threshold_multiplier=6.0)
    assert wide["effect_size_threshold"] > narrow["effect_size_threshold"]
    assert int(wide["primary"]["passes_effect_size"].sum()) <= int(
        narrow["primary"]["passes_effect_size"].sum())


# --------------------------------------------------------------------------- #
#  The cut is DRAWN
# --------------------------------------------------------------------------- #

def test_the_drawn_lines_sit_at_plus_and_minus_the_cut(tmp_path, monkeypatch):
    """An effect-size cut is symmetric -- a large negative effect is as much a
    hit as a large positive one -- and a legend listing it twice reads as two
    different rules.

    Driven through the real plotting function: capture the axis it built and
    read the vertical lines back off it."""
    import matplotlib.pyplot as plt
    from spacr import guide_permutation as gp

    output = _run(tmp_path)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axis = original(*args, **kwargs)
        captured["axis"] = axis
        return fig, axis

    monkeypatch.setattr(plt, "subplots", spy)
    gp.plot_guide_permutation_volcano(
        output["results"], outcome="score", minimum_wells=1,
        save_path=str(tmp_path / "lines.png"),
        effect_threshold=output["effect_size_threshold"],
        effect_threshold_label=output["effect_size_rule"])

    axis = captured["axis"]
    verticals = sorted(
        line.get_xdata()[0] for line in axis.get_lines()
        if len(set(line.get_xdata())) == 1)
    cut = output["effect_size_threshold"]
    assert pytest.approx(cut) in verticals
    assert pytest.approx(-cut) in verticals
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert output["effect_size_rule"] in labels
    assert labels.count(output["effect_size_rule"]) == 1


@pytest.mark.parametrize("value", [None, 0.0, -1.0, float("nan"), "not a number"])
def test_no_line_is_drawn_where_there_is_no_cut(tmp_path, monkeypatch, value):
    """A cut at zero excludes nothing and a line at zero is the axis that is
    already there. NaN is what a threshold looks like after a round trip
    through a CSV."""
    import matplotlib.pyplot as plt
    from spacr import guide_permutation as gp

    output = _run(tmp_path)
    captured = {}
    original = plt.subplots

    def spy(*args, **kwargs):
        fig, axis = original(*args, **kwargs)
        captured["axis"] = axis
        return fig, axis

    monkeypatch.setattr(plt, "subplots", spy)
    gp.plot_guide_permutation_volcano(
        output["results"], outcome="score", minimum_wells=1,
        save_path=str(tmp_path / f"none_{abs(hash(str(value)))}.png"),
        effect_threshold=value)

    axis = captured["axis"]
    verticals = [line.get_xdata()[0] for line in axis.get_lines()
                 if len(set(line.get_xdata())) == 1]
    # Only the zero reference line the volcano always draws.
    assert verticals == [0]


def test_the_run_draws_the_cut_on_every_support_family(tmp_path):
    """Four families, four volcanoes, one cut. A family drawn without it is a
    picture whose hits cannot be read off it."""
    output = _run_guide_permutation_analysis(
        _screen(), "score", str(tmp_path), {
            "guide_min_wells": [1, 2], "guide_primary_min_wells": 1,
            "guide_permutations": 199, "guide_permutation_seed": 3,
            "multiple_testing_method": "fdr_bh", "fdr_alpha": 0.05,
            "controls": CONTROLS, "threshold_method": "std",
            "threshold_multiplier": 3.0,
        })
    for threshold in (1, 2):
        for suffix in ("pdf", "png"):
            path = output["paths"][f"plot_min_{threshold}_{suffix}"]
            assert os.path.getsize(path) > 0


# --------------------------------------------------------------------------- #
#  ... and the settings-panel control that is still greyed out
# --------------------------------------------------------------------------- #

def test_the_permutation_run_reads_both_threshold_settings():
    """It reads them, so they are not dead settings under this inference.

    This is the half of the answer that lives in the run, and it is the
    evidence that the settings-panel rule below is stale.
    """
    import inspect

    from spacr.ml import _run_guide_permutation_analysis

    source = inspect.getsource(_run_guide_permutation_analysis)
    assert "settings.get('threshold_method'" in source
    assert "settings.get('threshold_multiplier'" in source


def test_the_settings_panel_offers_the_two_threshold_controls_under_permutation():
    """The controls the maintainer went looking for and could not find.

    They grey out under `inference='nonparametric'`, which is why they were
    put on the volcano's right-click menu instead. The run now honours both,
    so greying them hides a setting the run reads -- which instruction 106
    calls out by name: a control that cannot do anything is greyed and says
    why, and a control that CAN is not.
    """
    from spacr.settings import get_setting_dependencies

    rules = get_setting_dependencies()
    active = {"inference": "nonparametric",
              "analysis_mode": "guide_permutation"}
    for key in ("threshold_method", "threshold_multiplier"):
        # NO RULE AT ALL is the correct outcome, not a rule that always says
        # yes: a setting with no applicability rule is never greyed, and
        # keeping an always-true rule would be a rule that exists to say
        # nothing. Either shape passes, so the fix is not pinned to the way
        # it happened to be made.
        rule = rules.get(key)
        assert rule is None or rule["predicate"](active, {}), (
            f"{key} is read by the guide-permutation run, so the panel must "
            f"offer it under nonparametric inference")
