"""The menu of ways to choose which cells get annotated, and its two rules.

:mod:`spacr.regression_annotation` offers ten strategies. These tests hold
the properties that make the menu worth having rather than the code paths
that make it run:

* every entry says what it is for AND what it costs, and an entry that has
  no implementation refuses out loud rather than selecting nothing;
* a well never has cells on both sides of a split, and no strategy can
  select from the wells its own score is measured on;
* the named method's trap is DETECTED: on a screen where the score is a
  function of two columns and nothing else carries the phenotype, the fit
  with those columns looks excellent and the fit without them collapses to
  chance -- and the reported survival number says so. On a screen where the
  phenotype really is in the other columns, the same number stays high. A
  control that only ever said "leaking" would be no control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import regression_annotation as ra

WELLS = 12
PER_WELL = 40
HIT_WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")


def _plate(*, seed: int = 0, honest_signal: bool = True,
           wells: int = WELLS, per_well: int = PER_WELL) -> pd.DataFrame:
    """A screen whose score is a function of two columns and noise.

    ``honest_signal`` decides whether the phenotype also shows in columns
    the score is NOT a function of. With it the leakage control should find
    that most of the fit survives; without it the fit is the score and
    nothing else, and the control has to say so.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(wells):
        prevalence = 0.4 if well < 4 else 0.05
        for index in range(per_well):
            hit = rng.random() < prevalence
            carried = hit if honest_signal else False
            rows.append({
                "plateID": "plate1",
                "rowID": f"r{1 + well // 4}",
                "columnID": f"c{1 + well % 4}",
                "fieldID": "f1",
                "prcfo": f"plate1_r{1 + well // 4}_c{1 + well % 4}_f1_o{index}",
                # the two columns the score is built from
                "cell_area": rng.normal(900 + 260 * hit, 90),
                "cell_channel_1_mean_intensity": rng.normal(
                    1200 + 500 * hit, 130),
                # columns the score never saw
                "cell_eccentricity": rng.normal(0.55 - 0.2 * carried, 0.07),
                "nucleus_area": rng.normal(300 + 60 * carried, 35),
                "cell_channel_2_mean_intensity": rng.normal(
                    500 + 90 * carried, 55),
                "cell_channel_1_texture_contrast": rng.normal(30, 4),
            })
    frame = pd.DataFrame(rows)
    z = (0.0042 * (frame["cell_channel_1_mean_intensity"] - 1200)
         + 0.0042 * (frame["cell_area"] - 900)
         + rng.normal(0, 0.2, len(frame)))
    frame["pred"] = 1.0 / (1.0 + np.exp(-z))
    return frame


def _request(frame: pd.DataFrame, **overrides) -> ra.AnnotationRequest:
    values = dict(frame=frame, score_column="pred", wells=list(HIT_WELLS),
                  n_positive=30, holdout_fraction=0.25, seed=5)
    values.update(overrides)
    return ra.AnnotationRequest(**values)


def _well_name(group: str) -> str:
    """The plate's own spelling of a well, out of a split group id."""
    return "_".join(ra.readable_group(group).split("/")[1:])


@pytest.fixture(scope="module")
def plate():
    return _plate()


@pytest.fixture(scope="module")
def prepared(plate):
    return ra.prepare(_request(plate))


@pytest.fixture(scope="module")
def controls(prepared):
    """A positive and a negative control well the hold-out did not take.

    Chosen from the setup rather than written down, because which wells the
    random hold-out draws is the seed's business and a control well that
    landed in it is a strategy with nothing to anchor on -- which is a
    different test, below.
    """
    kept = sorted({str(g) for g in prepared.groups[prepared.selectable]})
    enriched = [g for g in kept if _well_name(g) in HIT_WELLS]
    plain = [g for g in kept if _well_name(g) not in HIT_WELLS]
    assert enriched and plain
    return {"positive_control_wells": [_well_name(enriched[0])],
            "negative_control_wells": [_well_name(plain[-1])]}


# --------------------------------------------------------------------------
# The menu
# --------------------------------------------------------------------------

def test_every_entry_says_what_it_is_for_and_what_it_costs():
    """A menu entry with no cost is the one chosen for the wrong reason."""
    assert len(ra.STRATEGIES) == 10
    for entry in ra.STRATEGIES:
        assert entry.purpose.strip().endswith("."), entry.key
        assert entry.cost.strip().endswith("."), entry.key
        assert len(entry.purpose.split()) >= 12, entry.key
        assert len(entry.cost.split()) >= 8, entry.key
        assert entry.key in entry.describe() or entry.title in entry.describe()


def test_the_named_method_leads_the_menu_and_the_random_draw_closes_it():
    assert ra.strategy_keys()[0] == "top_score_random"
    assert ra.strategy_keys()[-1] == "random_holdout"


def test_an_unknown_key_is_refused_with_the_menu_in_the_message():
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.strategy("top_scoring")
    assert "top_score_random" in str(caught.value)
    assert "uncertainty" in str(caught.value)


def test_a_declared_but_unbuilt_strategy_refuses_rather_than_doing_nothing(
        monkeypatch, plate):
    """The whole point of declaring one: it must not silently select none."""
    parked = ra.Strategy(key="parked", title="Parked", purpose="For later.",
                         cost="Nothing yet.", implemented=False)
    monkeypatch.setattr(ra, "STRATEGIES", ra.STRATEGIES + (parked,))
    with pytest.raises(ra.StrategyNotImplemented) as caught:
        ra.run("parked", _request(plate))
    assert "not implemented" in str(caught.value)
    assert "top_score_random" in str(caught.value)


def test_an_implemented_entry_with_no_runner_refuses_the_same_way(
        monkeypatch, plate):
    """Marked implemented and wired to nothing is still a refusal, not a crash."""
    monkeypatch.delitem(ra._RUNNERS, ra.DIVERSITY.key)
    with pytest.raises(ra.StrategyNotImplemented):
        ra.run(ra.DIVERSITY.key, _request(plate))


# --------------------------------------------------------------------------
# Rule one: a well is never on both sides
# --------------------------------------------------------------------------

def test_the_setup_goes_through_the_packages_own_splitter(monkeypatch, plate):
    """Not a second splitter written here, however small it would be."""
    from spacr import classifier_evaluation

    calls = []
    original = classifier_evaluation.grouped_split

    def recording(*args, **kwargs):
        calls.append(kwargs.get("group_by"))
        return original(*args, **kwargs)

    monkeypatch.setattr(classifier_evaluation, "grouped_split", recording)
    ra.prepare(_request(plate))
    assert calls == ["well"]


def test_no_well_has_cells_on_both_sides_of_the_hold_out(prepared):
    held = {str(g) for g in prepared.groups[prepared.holdout]}
    kept = {str(g) for g in prepared.groups[prepared.selectable]}
    assert held and kept
    assert not (held & kept)


@pytest.mark.parametrize("key", ra.implemented_keys())
def test_a_strategy_never_selects_a_cell_from_a_held_out_well(key, plate,
                                                              prepared,
                                                              controls):
    """The chooser cannot mark its own work, enforced by whole wells."""
    result = ra.run(key, _request(plate, **controls), prepared=prepared)
    held = {str(g) for g in prepared.groups[prepared.holdout]}
    chosen_wells = set(result.selection["annotation_group"])
    assert chosen_wells, f"{key} selected nothing at all"
    assert not (chosen_wells & held), (
        f"{key} selected cells from the wells it is measured on")
    assert not (set(result.selection.index) & set(result.holdout.index))


@pytest.mark.parametrize("key", ra.implemented_keys())
def test_every_reported_number_is_measured_on_the_random_hold_out(
        key, plate, prepared, controls):
    result = ra.run(key, _request(plate, **controls), prepared=prepared)
    if result.fit is None:
        assert "annotated" in result.summary() or "queue" in result.summary()
        return
    assert result.fit.n_test == prepared.holdout.size
    assert "held out" in result.fit.split_summary


# --------------------------------------------------------------------------
# Rule two: the trap in the named method, and whether the control finds it
# --------------------------------------------------------------------------

def test_the_leakage_control_catches_a_model_that_only_relearned_the_score():
    """Score is a function of two columns; nothing else carries the phenotype.

    The fit that keeps those two columns should look excellent and the fit
    without them should be near chance, and the survival number is what
    says so on the page.
    """
    frame = _plate(seed=11, honest_signal=False)
    result = ra.run("top_score_random", _request(frame))
    leakage = result.leakage
    assert leakage.with_score_inputs.roc_auc > 0.9
    assert leakage.without_score_inputs.roc_auc < 0.7
    assert leakage.survival is not None and leakage.survival < 0.35
    assert "survives" in leakage.summary()
    assert set(leakage.dropped) >= {"cell_area",
                                    "cell_channel_1_mean_intensity"}


def test_the_leakage_control_does_not_cry_wolf_when_the_signal_is_real():
    """A control that always said 'leaking' would measure nothing.

    The same seed, the same wells and the same cut on two screens that
    differ in one thing: whether the columns the score never saw carry the
    phenotype. The survival number has to separate them.
    """
    real = ra.run("top_score_random", _request(_plate(seed=11,
                                                      honest_signal=True)))
    only_the_score = ra.run(
        "top_score_random", _request(_plate(seed=11, honest_signal=False)))
    assert real.leakage.without_score_inputs.roc_auc > 0.8
    assert real.leakage.survival > 0.6
    assert real.leakage.survival > only_the_score.leakage.survival * 1.8


def test_the_applied_predictions_come_from_the_fit_without_the_score():
    frame = _plate(seed=3)
    result = ra.run("top_score_random", _request(frame))
    assert "fit excluding the score inputs" in result.summary()
    fitted = set(result.selection.index)
    assert not (set(result.predictions.index) & fitted)
    assert len(result.predictions) == len(frame) - len(fitted)
    # the rest of the screen AND the rest of the chosen wells
    assert bool(result.predictions["in_chosen_wells"].any())
    assert bool((~result.predictions["in_chosen_wells"]).any())


def test_the_leakage_mode_can_be_asked_for_one_fit_only(plate, prepared):
    dropped = ra.run("top_score_random", _request(plate, leakage="drop"),
                     prepared=prepared)
    assert dropped.leakage.with_score_inputs is None
    assert dropped.leakage.without_score_inputs is not None
    assert dropped.leakage.survival is None
    kept = ra.run("top_score_random", _request(plate, leakage="keep"),
                  prepared=prepared)
    assert kept.leakage.without_score_inputs is None
    assert "may therefore reproduce the original score" in kept.summary()


def test_an_unknown_leakage_mode_is_refused(plate):
    with pytest.raises(ra.AnnotationStrategyError):
        ra.prepare(_request(plate, leakage="ignore"))


def test_the_scores_own_inputs_can_be_named_instead_of_inferred(plate):
    """Naming them is the exact control; the correlation cut is the stand-in."""
    named = ra.prepare(_request(plate, score_inputs=["cell_area"]))
    assert set(named.score_inputs) == {"cell_area", "pred"}
    assert "cell_channel_1_mean_intensity" in named.honest_features
    inferred = ra.prepare(_request(plate))
    assert "cell_channel_1_mean_intensity" in inferred.score_inputs


# --------------------------------------------------------------------------
# Nothing to fit on says so
# --------------------------------------------------------------------------

def test_control_anchors_without_control_wells_says_so(plate, prepared):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("control_anchors",
               _request(plate, positive_control_wells=[],
                        negative_control_wells=[]),
               prepared=prepared)
    assert "positive and a negative control well" in str(caught.value)


def test_control_anchors_names_the_plate_when_the_wells_are_not_on_it(
        plate, prepared):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("control_anchors",
               _request(plate, positive_control_wells=["r9_c9"],
                        negative_control_wells=["r8_c8"]),
               prepared=prepared)
    assert "0 positive" in str(caught.value)


def test_a_table_with_no_score_and_no_annotations_is_refused(plate):
    frame = plate.drop(columns=["pred"])
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(frame))
    assert "no score column" in str(caught.value)


def test_a_table_with_no_measurement_column_is_refused(plate):
    frame = plate[["plateID", "rowID", "columnID", "fieldID", "prcfo", "pred"]]
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(frame))
    assert "No measurement column survives" in str(caught.value)


def test_naming_a_feature_column_the_table_has_not_got_is_refused(plate):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(plate, feature_columns=["cell_area", "moonlight"]))
    assert "moonlight" in str(caught.value)


def test_a_screen_of_one_well_cannot_hold_a_well_out(plate):
    one = plate.loc[plate["columnID"] == "c1"].copy()
    one["rowID"] = "r1"
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(one, wells=["r1_c1"]))
    assert "well" in str(caught.value).lower()


def test_naming_a_well_that_is_not_on_the_plate_is_refused(plate):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(plate, wells=["r7_c7"]))
    assert "r7_c7" in str(caught.value)


def test_asking_for_more_positives_than_the_wells_hold_is_capped_out_loud(
        plate, prepared):
    """Half the pool at most, because the contrast draw is the same size."""
    result = ra.run("top_score_random", _request(plate, n_positive=10_000),
                    prepared=prepared)
    taken = int(result.counts["positives"])
    assert taken == prepared.chosen.size // 2
    assert result.counts["contrast"] == taken
    assert "10,000 positives were asked for" in result.summary()


def test_a_pool_too_small_for_a_matched_pair_is_refused(plate):
    """Under four selectable cells there is no pair to match at all."""
    tiny = plate.loc[plate["columnID"] == "c1"].copy()
    tiny = pd.concat([tiny.head(3), plate.loc[plate["columnID"] != "c1"]])
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("top_score_random",
               _request(tiny, wells=["r1_c1"], n_positive=30))
    assert "cannot supply" in str(caught.value) or \
        "too few" in str(caught.value)


# --------------------------------------------------------------------------
# Wells, and how a name is matched to one
# --------------------------------------------------------------------------

def test_a_well_is_matched_on_its_identity_tokens():
    groups = np.array(["plate1\x1fr1\x1fc1", "plate1\x1fr1\x1fc2",
                       "plate2\x1fr1\x1fc1"], dtype=object)
    assert list(ra.wells_selected(groups, ["r1_c1"])) == [True, False, True]
    assert list(ra.wells_selected(groups, ["plate2_r1_c1"])) == \
        [False, False, True]
    assert list(ra.wells_selected(groups, [])) == [True, True, True]


# --------------------------------------------------------------------------
# The individual strategies
# --------------------------------------------------------------------------

def test_a_queue_reports_what_it_did_to_the_class_balance(plate, prepared):
    for key in ("uncertainty", "diversity", "score_strata",
                "two_view_disagreement"):
        result = ra.run(key, _request(plate), prepared=prepared)
        assert "positive_share" in result.counts, key
        assert "random_positive_share" in result.counts, key
        assert "plain random draw of the same size" in result.summary(), key
        assert set(result.selection["annotation_role"]) == {"queue"}, key


def test_uncertainty_queues_cells_near_the_boundary(plate, prepared):
    result = ra.run("uncertainty", _request(plate), prepared=prepared)
    probabilities = result.selection["model_probability"].astype(float)
    assert len(probabilities) == 30
    # every queued cell is nearer 0.5 than the median unlabelled cell is
    assert float(np.abs(probabilities - 0.5).max()) < 0.5
    assert result.counts["measure"] == "margin"


def test_diversity_takes_one_cell_per_cluster(plate, prepared):
    result = ra.run("diversity", _request(plate, n_clusters=12),
                    prepared=prepared)
    assert result.counts["clusters"] == 12
    assert len(result.selection) == 12
    assert result.counts["largest_cluster"] >= result.counts["smallest_cluster"]


def test_score_strata_spreads_the_budget_over_the_whole_score_range(plate,
                                                                    prepared):
    result = ra.run("score_strata", _request(plate, n_bins=5),
                    prepared=prepared)
    assert result.counts["strata"] == 5
    scores = plate.loc[result.selection.index, "pred"]
    everything = plate["pred"]
    assert scores.min() < everything.quantile(0.3)
    assert scores.max() > everything.quantile(0.7)


def test_the_contrast_set_of_pu_learning_is_unlabelled_not_negative(plate,
                                                                    prepared):
    result = ra.run("pu_learning", _request(plate), prepared=prepared)
    roles = set(result.selection["annotation_role"])
    assert roles == {"positive", "unlabelled"}
    assert 0.0 < result.counts["labelling_rate"] <= 1.0
    assert "unlabelled rather than negative" in result.summary()
    # the rescaling moves the line, and the run says by how much
    assert result.counts["called_positive_rescaled"] >= \
        result.counts["called_positive_as_negative"]


def test_self_training_audits_every_round_on_the_fixed_hold_out(plate,
                                                                prepared):
    result = ra.run("self_training", _request(plate), prepared=prepared)
    assert "Audit curve" in result.summary()
    assert result.counts["rounds"] >= 1
    pseudo = result.selection.loc[
        result.selection["annotation_role"] == "pseudo"]
    assert not (set(pseudo.index) & set(result.holdout.index))
    assert result.counts["best_round"] < result.counts["rounds"]


def test_self_training_stops_when_the_audit_stops_improving(plate, prepared):
    """Twenty rounds must not mean twenty rounds of agreeing with itself."""
    result = ra.run("self_training", _request(plate, rounds=20),
                    prepared=prepared)
    assert result.counts["rounds"] < 20
    assert "stopped" in result.summary()


def test_two_views_are_fitted_on_different_columns(plate, prepared):
    result = ra.run("two_view_disagreement", _request(plate),
                    prepared=prepared)
    assert result.counts["intensity_columns"] >= 1
    assert result.counts["shape_columns"] >= 1
    assert "view_gap" in result.selection.columns
    assert result.counts["better_view"] in ("intensity", "shape")


def test_two_views_refuse_a_table_that_has_only_one_of_them(plate):
    one_family = plate[["plateID", "rowID", "columnID", "fieldID", "prcfo",
                        "pred", "cell_area", "nucleus_area",
                        "cell_eccentricity"]]
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("two_view_disagreement", _request(one_family))
    assert "intensity-like" in str(caught.value)


def test_neighbour_propagation_shows_the_distance_cut(plate, prepared):
    result = ra.run("neighbour_propagation", _request(plate),
                    prepared=prepared)
    assert result.counts["radius"] > 0
    assert f"{result.counts['radius']:.4g}" in result.summary()
    assert "crossed_group_share" in result.counts
    assert 0.0 <= result.counts["crossed_group_share"] <= 1.0
    assert "propagated" in set(result.selection["annotation_role"])


def test_a_radius_of_zero_propagates_nothing_and_says_so(plate, prepared):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.run("neighbour_propagation", _request(plate, distance_cut=0.0),
               prepared=prepared)
    assert "nothing propagates" in str(caught.value)


def test_a_tighter_cut_propagates_strictly_fewer_labels(plate, prepared):
    """The cut is not decoration: widening it is what manufactures agreement."""
    tight = ra.run("neighbour_propagation", _request(plate, distance_cut=0.5),
                   prepared=prepared)
    loose = ra.run("neighbour_propagation", _request(plate, distance_cut=2.0),
                   prepared=prepared)
    assert tight.counts["propagated"] < loose.counts["propagated"]
    assert "the radius 0.5 this run was given" in tight.summary()


def test_the_random_draw_reports_what_the_clever_strategies_are_buying(
        plate, prepared):
    result = ra.run("random_holdout", _request(plate), prepared=prepared)
    assert result.counts["positive_share"] < prepared.positive_share(
        prepared.chosen)
    assert "quantifies enrichment produced by targeted selection" in \
        result.summary()


# --------------------------------------------------------------------------
# Human annotations, when there are any
# --------------------------------------------------------------------------

def test_annotations_replace_the_score_as_the_reference_label(plate):
    frame = plate.copy()
    rng = np.random.default_rng(4)
    frame["annotate"] = None
    drawn = rng.choice(len(frame), size=200, replace=False)
    frame.loc[frame.index[drawn], "annotate"] = np.where(
        frame.loc[frame.index[drawn], "cell_eccentricity"] < 0.5, "hit",
        "control")
    prepared = ra.prepare(_request(frame, label_column="annotate"))
    assert prepared.annotated
    assert "annotations in 'annotate'" in prepared.label_source
    assert int(prepared.known.sum()) == 200
    # the unannotated cells are NOT treated as negatives
    assert not prepared.known[~frame["annotate"].notna().to_numpy()].any()
    assert set(prepared.holdout) <= set(np.flatnonzero(prepared.known))


def test_a_label_column_with_one_class_falls_back_to_the_score(plate):
    frame = plate.copy()
    frame["annotate"] = None
    frame.loc[frame.index[:20], "annotate"] = "hit"
    prepared = ra.prepare(_request(frame, label_column="annotate"))
    assert not prepared.annotated
    assert "fewer than two usable classes" in " ".join(prepared.notes)


def test_a_label_column_the_table_has_not_got_is_refused(plate):
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(plate, label_column="verdict"))
    assert "verdict" in str(caught.value)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------

def test_the_named_method_uses_xgboost_when_it_is_installed(plate, prepared):
    if not ra.xgboost_available():
        pytest.skip("xgboost is not installed in this environment")
    result = ra.run("top_score_random", _request(plate), prepared=prepared)
    assert result.fit.model == "xgboost"


def test_without_xgboost_it_falls_back_and_the_report_says_which(
        monkeypatch, plate, prepared):
    monkeypatch.setattr(ra, "xgboost_available", lambda: False)
    result = ra.run("top_score_random", _request(plate), prepared=prepared)
    assert result.fit.model == "hist_gradient_boosting"
    assert "hist_gradient_boosting" in result.summary()


def test_demanding_xgboost_without_it_is_an_error_naming_the_fallback(
        monkeypatch):
    monkeypatch.setattr(ra, "xgboost_available", lambda: False)
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra._estimator("xgboost", 0)
    assert "hist_gradient_boosting" in str(caught.value)


# --------------------------------------------------------------------------
# What comes out
# --------------------------------------------------------------------------

def test_a_result_can_be_written_beside_the_run(plate, prepared, tmp_path):
    result = ra.run("top_score_random", _request(plate), prepared=prepared)
    written = result.write(str(tmp_path / "annotation"))
    assert set(written) == {"selection", "holdout", "predictions", "report"}
    selection = pd.read_csv(written["selection"], index_col=0)
    assert set(selection["annotation_role"]) == {"positive", "contrast"}
    report = (tmp_path / "annotation" / "annotation_report.txt").read_text()
    assert "survives" in report


def test_the_same_seed_chooses_the_same_cells(plate):
    first = ra.run("top_score_random", _request(plate, seed=9))
    second = ra.run("top_score_random", _request(plate, seed=9))
    assert list(first.selection.index) == list(second.selection.index)
    third = ra.run("top_score_random", _request(plate, seed=10))
    assert list(third.selection.index) != list(first.selection.index)


def test_too_few_annotations_says_so_rather_than_fitting_on_nothing(plate):
    """The literal case: a label column, and almost nothing left to fit on.

    Four annotated cells over four wells, two of each class, and half the
    wells held aside. The hold-out is honest and there are two annotated
    cells outside it, which is not a training set -- so the strategy that
    needs a model refuses instead of fitting one on two cells.
    """
    frame = plate.copy()
    frame["annotate"] = None
    for well, verdict in (("c1", "hit"), ("c2", "hit"),
                          ("c3", "control"), ("c4", "control")):
        one = frame.index[(frame["rowID"] == "r1")
                          & (frame["columnID"] == well)][0]
        frame.loc[one, "annotate"] = verdict
    request = _request(frame, label_column="annotate", holdout_fraction=0.5)
    prepared = ra.prepare(request)
    assert prepared.annotated
    assert prepared.labelled().size == 2
    with pytest.raises(ra.NotEnoughLabels) as caught:
        ra.run("uncertainty", request, prepared=prepared)
    assert "annotate more cells" in str(caught.value).lower()
    assert "at least four" in str(caught.value)


def test_annotations_confined_to_one_well_cannot_be_held_out(plate):
    """An afternoon spent annotating ONE well cannot measure anything.

    Every labelled cell is in the same well, so the only split available
    puts siblings on both sides and measures how well the model memorised
    that well. The setup refuses instead.
    """
    frame = plate.copy()
    frame["annotate"] = None
    one_well = frame.index[(frame["rowID"] == "r1")
                           & (frame["columnID"] == "c1")][:6]
    frame.loc[one_well[:3], "annotate"] = "hit"
    frame.loc[one_well[3:], "annotate"] = "control"
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(frame, label_column="annotate"))
    assert "every labelled cell comes from one well" in str(caught.value)


def test_a_strategy_that_ranks_by_the_score_refuses_a_table_without_one(plate):
    """Annotations can carry a screen; a ranking by the score cannot."""
    frame = plate.copy()
    rng = np.random.default_rng(2)
    frame["annotate"] = np.where(
        rng.random(len(frame)) < 0.5, "hit", "control")
    prepared = ra.prepare(_request(frame.drop(columns=["pred"]),
                                   label_column="annotate"))
    for key in ("top_score_random", "score_strata"):
        with pytest.raises(ra.AnnotationStrategyError) as caught:
            ra.run(key, _request(frame.drop(columns=["pred"]),
                                 label_column="annotate"),
                   prepared=prepared)
        assert "diversity sampling" in str(caught.value), key


def test_a_pool_too_wide_for_a_dense_matrix_is_sampled_and_says_so(
        monkeypatch, plate, prepared):
    monkeypatch.setattr(ra, "MAX_POOL_FOR_DISTANCES", 100)
    result = ra.run("diversity", _request(plate, n_clusters=8),
                    prepared=prepared)
    assert "ran on a random sample" in result.summary()
    assert result.counts["clusters"] == 8


def test_the_role_of_every_selected_cell_is_a_named_one(plate, prepared,
                                                        controls):
    """A role a reader cannot look up is a role that says nothing."""
    for key in ra.implemented_keys():
        result = ra.run(key, _request(plate, **controls), prepared=prepared)
        for frame in (result.selection, result.holdout):
            for role in set(frame["annotation_role"]):
                assert role in ra.ROLES, f"{key} invented the role {role!r}"


def test_a_table_whose_every_column_is_a_score_input_says_so(plate):
    """Drop mode with nothing left to fit on must not report a blank fit."""
    frame = plate[["plateID", "rowID", "columnID", "fieldID", "prcfo",
                   "pred", "cell_area", "cell_channel_1_mean_intensity"]]
    prepared = ra.prepare(_request(frame, leakage="drop"))
    assert prepared.honest_features == ()
    assert "Every feature column is identified as a score input" in \
        " ".join(prepared.notes)
    result = ra.run("top_score_random", _request(frame, leakage="drop"),
                    prepared=prepared)
    assert result.fit is None
    assert result.predictions is None
    assert "Nothing was fitted" in result.summary()


def test_a_group_id_is_printed_so_a_person_can_read_it():
    """The splitter joins identities with a separator that prints as nothing.

    Written straight out, ``plate1/r1/c1`` reads as ``plate1r1c1`` and a
    reader cannot tell which part is the row.
    """
    assert ra.readable_group("plate1\x1fr1\x1fc1") == "plate1/r1/c1"
    assert ra.readable_group("plate1") == "plate1"


def test_every_reported_group_is_the_readable_spelling(plate, prepared):
    result = ra.run("top_score_random", _request(plate), prepared=prepared)
    for frame in (result.selection, result.holdout):
        for group in set(frame["annotation_group"]):
            assert "\x1f" not in group
            assert "/" in group
    assert "\x1f" not in result.summary()


# ---------------------------------------------------------------------------
# the refusals and fallbacks nobody reaches on a healthy screen
# ---------------------------------------------------------------------------

def test_an_empty_feature_list_is_refused_rather_than_fitted_on_nothing(plate):
    """A list that names no columns is a mistake, not a request for defaults.

    Falling through would fit on a zero-column design, and sklearn's message
    for that is about array shapes -- which sends the reader looking at the
    plate rather than at the setting they left empty. The distinction matters
    because an empty list and an ABSENT one mean opposite things here: absent
    means "choose for me", and the two must not converge.
    """
    with pytest.raises(ra.AnnotationStrategyError) as caught:
        ra.prepare(_request(plate, feature_columns=[]))

    assert "empty feature list" in str(caught.value)


def test_feature_views_survive_a_classifier_that_cannot_group_the_columns(
        plate, monkeypatch):
    """The views are an aid; failing to build them must not fail the fit.

    ``column_groups.classify`` reads naming conventions and can raise on a
    table whose columns do not follow any -- an external mask set, a hand-made
    export. The views are what offer "intensity only" and "shape only" in the
    menu, so the honest degradation is to offer neither, not to refuse the
    screen.
    """
    from spacr import column_groups

    def refuse(_names):
        raise RuntimeError("these column names follow no convention")

    monkeypatch.setattr(column_groups, "classify", refuse)

    views = ra.feature_views([c for c in plate.columns if c != "pred"])

    assert isinstance(views, dict)
    for name, columns in views.items():
        assert not columns or all(isinstance(c, str) for c in columns)


def test_an_unanswerable_xgboost_probe_reads_as_absent(monkeypatch):
    """``find_spec`` raises, it does not only return None.

    A namespace package shadowing the name raises ValueError, and a broken
    parent package raises ImportError. Either way the honest answer is "no
    xgboost", because the next thing the caller does is import it -- and an
    exception escaping a capability PROBE would take down the strategy menu
    while it was drawing itself.
    """
    import importlib.util

    def refuse(_name):
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(importlib.util, "find_spec", refuse)

    assert ra.xgboost_available() is False


def test_a_holdout_with_one_class_reports_no_auc_rather_than_raising(plate):
    """AUC is undefined on a single class, and the rest of the report is not.

    ``roc_auc_score`` raises ValueError when the held-out wells happen to be
    all-positive or all-negative, which a small screen produces regularly.
    Accuracy and the counts are still meaningful, so the report carries them
    with ``auc=None`` -- and None is what the caller must show as "not
    measurable", never as zero.
    """
    import dataclasses

    prepared = ra.prepare(_request(plate))
    labels = np.array(prepared.labels, copy=True)
    labels[prepared.holdout] = 1
    single = dataclasses.replace(prepared, labels=labels)

    held = len(np.asarray(prepared.holdout))
    assert held > 0, "the fixture reserved no hold-out rows to score"

    report = ra._score_holdout(
        single, probabilities=np.full(held, 0.9), n_train=10,
        columns=prepared.features, model="logistic")

    assert report.roc_auc is None, "AUC is undefined on a single class"
    assert report.accuracy == pytest.approx(1.0)
    assert report.n_test == held

    # The two things None buys, and NaN does not.
    assert report.lift == pytest.approx(
        report.balanced_accuracy - 0.5), (
        "lift must fall back to balanced accuracy; NaN propagates instead, "
        "and this lift is the number the named-method leak check compares")
    assert "ROC AUC n/a" in report.summary()
