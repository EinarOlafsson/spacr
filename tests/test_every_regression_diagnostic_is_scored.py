"""Instruction 115: each diagnostic carries its own verdict.

The panels and the numbers were already there. What was missing is the step a
reader was quietly expected to do twenty-three times: decide whether the
number is fine. These assert that the decision is made, that it is made in the
right DIRECTION (a diagnostic test's p is backwards), that it is calibrated so
a clean fit does not flag, and that it ends up ON the panel rather than in a
table somewhere else.
"""
import os

import numpy as np
import pandas as pd
import pytest

from spacr import regression_qc as qc

sm = pytest.importorskip("statsmodels.api")


def _design(n=400, p=6, collinear=False, heteroscedastic=False, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"g{i}" for i in range(p)])
    if collinear:
        X["g5"] = X["g0"] * 0.999 + rng.normal(0, 0.01, n)
    X.insert(0, "const", 1.0)
    mu = X.to_numpy() @ np.r_[0.2, rng.normal(0, 0.4, p)]
    noise = rng.normal(0, 0.5, n)
    if heteroscedastic:
        noise = noise * (1 + 3 * np.abs(mu))
    y = pd.Series(mu + noise)
    return sm.OLS(y, X).fit(), X, y


@pytest.fixture()
def png_preference(monkeypatch):
    monkeypatch.setattr("spacr.plot.figure_output_preferences",
                        lambda: ("png", 120))


def _report(tmp_path, model, X, y, **kwargs):
    return qc.regression_qc_report(model, X, y, str(tmp_path),
                                   regression_type="ols", verbose=False,
                                   **kwargs)


# ------------------------------------------------------- the set is complete

def test_every_panel_has_a_scorer():
    """A panel added without one goes back to being a number with no verdict."""
    assert sorted(qc._SCORERS) == sorted(qc.PANEL_ORDER)


def test_an_unscored_name_is_unknown_rather_than_an_error():
    assert qc.score_panel("no_such_panel", {}).level == "unknown"


def test_a_scorer_that_trips_does_not_take_the_fit_down():
    """A diagnostic that crashes while judging a diagnostic must not lose the
    run that already succeeded."""
    assert qc.score_panel("vif", {"max_vif": "not a number"}).level == "unknown"
    assert qc.score_panel("vif", None).level == "unknown"
    assert qc.score_panel("qq_residuals", {}).level == "unknown"


# -------------------------------------------- the direction that is easy to invert

def test_a_large_diagnostic_p_is_the_good_outcome():
    """THE CLASSIC INVERSION. The null of every one of these tests is that
    the assumption HOLDS, so a small p is evidence against the model."""
    assert qc.score_panel("scale_location", {"levene_p": 0.62}).level == "pass"
    assert qc.score_panel("scale_location", {"levene_p": 1e-9}).level == "fail"
    assert qc.score_panel("plate_effects", {"kruskal_p": 0.9}).level == "pass"
    assert qc.score_panel("plate_effects", {"kruskal_p": 1e-9}).level == "fail"


def test_the_sentence_says_which_direction_it_read():
    verdict = qc.score_panel("scale_location", {"levene_p": 0.62})
    assert "large p" in verdict.detail.lower()


# -------------------------------------------------- the conventional thresholds

@pytest.mark.parametrize("value, level", [(2.0, "pass"), (7.0, "check"),
                                          (25.0, "fail")])
def test_vif_uses_the_conventional_bands(value, level):
    assert qc.score_panel("vif", {"max_vif": value}).level == level


def test_an_aliased_predictor_fails_outright():
    """No VIF exists for it, and the fit reports its coefficient anyway."""
    verdict = qc.score_panel("vif", {"n_aliased": 2, "max_vif": 1.0})
    assert verdict.level == "fail"
    assert "linear combinations" in verdict.headline


@pytest.mark.parametrize("value, level", [(9.0, "pass"), (55.0, "check"),
                                          (400.0, "fail")])
def test_the_condition_number_uses_belsleys_bands(value, level):
    assert qc.score_panel("condition_number",
                          {"condition_number": value}).level == level


@pytest.mark.parametrize("value, level", [(0.2, "pass"), (0.7, "check"),
                                          (1.4, "fail")])
def test_cooks_distance_uses_the_half_and_one_rule(value, level):
    assert qc.score_panel("cooks_distance",
                          {"max_cooks": value, "n_above": 1}).level == level


# --------------------------------------------- calibrated against a clean fit

def test_a_spike_at_zero_is_a_pass_because_it_is_what_hits_look_like():
    """The trap: a screen with real hits has a p-value spike, and reading it
    as a fault would flag every successful screen."""
    assert qc.score_panel("p_value_histogram",
                          {"verdict": "uniform-with-spike"}).level == "pass"
    assert qc.score_panel("p_value_histogram",
                          {"verdict": "u-shaped"}).level == "fail"


def test_dffits_is_scored_on_the_fraction_not_the_maximum():
    """2*sqrt(p/n) is a SCREENING line a correct model is expected to cross.

    Measured on a clean 400-well Gaussian fit, the largest |DFFITS| is over
    twice the threshold -- so a rule read off the maximum flags a fit with no
    defect in it at all.
    """
    clean = {"n_above": 12, "n_points": 400, "threshold": 0.26,
             "max_abs_dffits": 0.7}
    assert clean["max_abs_dffits"] / clean["threshold"] > 2.0
    assert qc.score_panel("dffits", clean).level == "pass"
    assert qc.score_panel("dffits", dict(clean, n_above=100)).level == "fail"


def test_a_clean_fit_flags_nothing(tmp_path, png_preference):
    """A warning that fires on a clean fit is a warning nobody reads."""
    model, X, y = _design()
    manifest = _report(tmp_path, model, X, y, combined=False)
    flagged = [p.name for p in manifest["panels"]
               if p.verdict is not None and p.verdict.level in ("check", "fail")]
    assert flagged == [], flagged
    assert manifest["verdict_level"] == "pass"


def test_a_broken_design_is_named_panel_by_panel(tmp_path, png_preference):
    model, X, y = _design(collinear=True, heteroscedastic=True)
    manifest = _report(tmp_path, model, X, y, combined=False)
    failed = {p.name for p in manifest["panels"]
              if p.verdict is not None and p.verdict.level == "fail"}
    assert {"vif", "condition_number", "predictor_correlation"} <= failed
    assert manifest["verdict_level"] == "fail"


# ---------------------------------------------- the suite reports its worst panel

def test_the_suite_is_summarised_by_its_worst_panel():
    """Nineteen passes and one rank-deficient design is not '95% passed'."""
    verdicts = [qc.PanelVerdict("pass", "fine")] * 19
    verdicts.append(qc.PanelVerdict("fail", "the design is rank deficient"))
    assert qc.worst_verdict(verdicts).level == "fail"


def test_worst_verdict_of_nothing_is_none():
    assert qc.worst_verdict([]) is None
    assert qc.worst_verdict([None, None]) is None


# ---------------------------------------------- the verdict is WHERE the panel is

def test_the_verdict_is_drawn_on_the_panel(tmp_path, png_preference):
    """Measured on the pixels: the FAIL ink is in the badge corner."""
    Image = pytest.importorskip("PIL.Image")
    from spacr.figures.style import ROLES

    model, X, y = _design(collinear=True)
    manifest = _report(tmp_path, model, X, y,
                       panels=["vif", "qq_residuals"], combined=False)
    by_name = {p.name: p for p in manifest["panels"]}
    assert by_name["vif"].verdict.level == "fail"

    ink = tuple(int(ROLES["down"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    def badge_pixels(path):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        band = np.asarray(image.crop((0, int(height * 0.72),
                                      int(width * 0.75), height))).astype(int)
        # Anti-aliasing blends the one-pixel glyphs and border with the page,
        # so some supported renderers produce no pixels close to the literal
        # source RGB value.  Their hue still has the same two strong channel
        # separations; grayscale labels and axes have neither.
        red_over_green = max(10, (ink[0] - ink[1]) // 4)
        red_over_blue = max(10, (ink[0] - ink[2]) // 4)
        return int(((band[..., 0] - band[..., 1] > red_over_green)
                    & (band[..., 0] - band[..., 2] > red_over_blue)).sum())

    # The text and its boxed edge produce hundreds of hue-matched pixels; the
    # passing panel is exactly 0.
    assert badge_pixels(by_name["vif"].path) > 50
    # The passing panel is not stamped in the failing ink.
    assert badge_pixels(by_name["qq_residuals"].path) == 0


def test_the_combined_page_shows_the_same_verdict_not_a_second_opinion(
        tmp_path, png_preference, monkeypatch):
    """Re-scoring the redraw would let the page and the file disagree."""
    calls = []
    original = qc.score_panel

    def counted(name, stats):
        calls.append(name)
        return original(name, stats)

    monkeypatch.setattr(qc, "score_panel", counted)
    model, X, y = _design()
    _report(tmp_path, model, X, y, panels=["vif", "qq_residuals"],
            combined=True)
    assert calls == ["vif", "qq_residuals"], calls


# --------------------------------------------- the verdict travels with the run

def test_the_text_report_states_the_verdict_and_what_it_means(
        tmp_path, png_preference):
    model, X, y = _design(collinear=True)
    manifest = _report(tmp_path, model, X, y, panels=["vif"], combined=False)
    text = open(manifest["report"], encoding="utf-8").read()
    assert "verdict: FAIL" in text
    assert "means:" in text
    assert "max VIF" in text


def test_the_manifest_carries_the_verdicts_and_a_count(tmp_path,
                                                       png_preference):
    model, X, y = _design()
    manifest = _report(tmp_path, model, X, y, panels=["vif", "qq_residuals"],
                       combined=False)
    assert set(manifest["verdicts"]) == {"vif", "qq_residuals"}
    assert manifest["verdict_counts"]["pass"] == 2
    assert manifest["verdict"].level == "pass"


def test_a_skipped_panel_has_no_verdict_rather_than_a_passing_one(
        tmp_path, png_preference):
    """'The panel is not there' and 'the panel is fine' must not look alike."""
    model, X, y = _design()
    manifest = _report(tmp_path, model, X, y, panels=["roc"], combined=False)
    panel = manifest["panels"][0]
    assert panel.status == "skipped"
    assert panel.verdict is None
    assert manifest["verdict_counts"]["pass"] == 0
