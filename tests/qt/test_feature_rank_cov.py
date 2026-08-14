"""Feature ranking — the edges ``test_feature_explorer.py`` does not reach.

That file plants four features with separations worked out by hand and checks
the ranking gets them in the planted order. This one goes after the parts a
real measurement table hits and a four-column fixture never does:

* **which columns are even offered** — the float-that-looks-categorical that
  must not be a class column, and the discrete count that must be a feature
  even though the shared classifier calls it a category;
* the **statistics' edges** — an empty group, a group of one, a pooled SD of
  zero, all of which must produce ``nan`` rather than a number that would sort
  to the top of the table;
* what happens when the ranking statistic **cannot** be computed but the other
  three can;
* the **five-number summaries** each class carries for drawing, and the
  drawable bin range for a feature that is constant (a zero-width ``linspace``
  makes ``np.histogram`` raise, so the widening is a crash guard, not a nicety);
* the **subsample** the label-shuffling null falls back to on a big table;
* rows with **no class label**, which must not enter a score.

The final regressions cover three once-subtle label defects: unlabelled rows in
the null model, date-valued classes, and a real class named by an empty string.

No Qt, no files, no network — ``feature_rank`` is pure numpy and pandas. The
import still needs PySide6 because the column classifier it shares with the
Local Data Filter lives in a widget module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.feature_rank import (
    AUC, COHEN_D, KS, MIN_PER_CLASS, MUTUAL_INFO, NULL_MAX_ROWS,
    ExplorerError, ExplorerSpec, auc_of, candidate_features, candidate_labels,
    cohen_d_of, distributions, ks_of, mutual_info_of, rank_features,
)


# ---------------------------------------------------------------------------
# A table shaped like a spaCR measurement table, with the answer planted
# ---------------------------------------------------------------------------

@pytest.fixture
def measured() -> pd.DataFrame:
    """Twelve objects, six per condition, with every column kind in play.

    ``cell_area`` separates perfectly (AUC 1). ``pathogen_count`` separates
    partly: of the 36 ctrl/trt pairs, 31 have trt above (the ties contributing
    a half each), so AUC = 31/36 and the separation is 26/36. ``nucleus_area``
    is the same six values in both conditions, and ``plateID`` alternates, so
    both sit at exactly 0.
    """
    return pd.DataFrame({
        "condition": ["ctrl"] * 6 + ["trt"] * 6,
        "plateID": [1, 2] * 6,
        "well": [f"A{i:02d}" for i in range(1, 13)],
        "object_label": list(range(1, 13)),
        "prcfo": [f"p1_A{i:02d}_o{i}" for i in range(1, 13)],
        "cell_area": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
                      200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
        "nucleus_area": [50.0, 52.0, 54.0, 56.0, 58.0, 60.0] * 2,
        "pathogen_count": [0, 0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 4],
    })


def spec(**kwargs) -> ExplorerSpec:
    kwargs.setdefault("label", "condition")
    return ExplorerSpec(**kwargs)


def test_a_measurement_table_ranks_its_measurements_and_nothing_else(measured):
    """The whole public flow on a table with identity, key and text columns.

    Four columns are ranked and four are not, and *which* four is the point:
    ``object_label`` and ``prcfo`` identify a row rather than describe it (they
    would score 0.5 or 1.0 depending only on how the table was sorted),
    ``well`` is text, and ``condition`` is the thing being separated. They are
    absent from the ranking AND from ``skipped`` — never offered, not offered
    and dropped.
    """
    result = rank_features(measured, spec())
    assert [s.feature for s in result.scores] == [
        "cell_area", "pathogen_count", "nucleus_area", "plateID"]
    assert result.n_considered == 4
    assert result.n_rows == 12
    assert dict(result.skipped) == {}
    assert result.classes == ("ctrl", "trt")
    assert result.scores[0].score == pytest.approx(1.0)
    assert result.scores[0].higher_in == "trt"
    assert result.score_for("pathogen_count").auc == pytest.approx(31 / 36)
    assert result.score_for("pathogen_count").score == pytest.approx(26 / 36)
    assert result.score_for("nucleus_area").score == pytest.approx(0.0)
    # Six per class is above both thresholds, so the result carries no caveat.
    assert result.notice == ""
    assert not result.scores[0].is_low_n


def test_a_numeric_plate_column_is_ranked_on_purpose(measured):
    """If ``plateID`` came top, the classes are separated by batch.

    So it is ranked rather than filtered out — and here, where the plates are
    interleaved across conditions, it scores 0. A version that hid batch
    columns would make a batch effect invisible on the one screen that could
    have caught it.
    """
    result = rank_features(measured, spec())
    assert result.score_for("plateID").score == pytest.approx(0.0)
    assert result.score_for("plateID").n_by_class == {"ctrl": 6, "trt": 6}


# ---------------------------------------------------------------------------
# Which columns are offered
# ---------------------------------------------------------------------------

def test_a_coarse_float_measurement_is_not_offered_as_a_class_column(measured):
    """``eccentricity`` with six distinct values is a measurement, not a class.

    The shared column classifier calls any column with twelve or fewer
    distinct values a *category* — right for deciding between a slider and a
    tick list, wrong for "what shall I separate by". Without the float guard
    this picker would offer a continuous measurement as the class column, and
    ranking every feature against ``cell_eccentricity`` is a table of
    correlations wearing a lab coat.
    """
    frame = measured.assign(
        eccentricity=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60] * 2,
        infected=[True, True, False, False, True, False] * 2)
    labels = candidate_labels(frame)
    assert "eccentricity" not in labels
    assert "cell_area" not in labels          # float, 12 distinct
    # Text, an integer code and a boolean all are class columns.
    assert set(labels) >= {"condition", "plateID", "infected",
                           "pathogen_count"}
    # `well` is twelve distinct strings and is offered — twelve is the cap, and
    # "which well was it in" is a real thing to separate by. `prcfo` and
    # `object_label` are twelve distinct values too and are NOT, because they
    # identify the row rather than describe it.
    assert "well" in labels
    assert "object_label" not in labels and "prcfo" not in labels


def test_a_discrete_count_is_a_feature_even_though_it_looks_categorical(
        measured):
    """``pathogen_count`` runs 0–4 and is exactly what a ranking exists to find.

    ``candidate_features`` deliberately does *not* reuse the continuous/
    categorical split — a separation statistic is perfectly happy on a count —
    while still dropping the identity columns and the label itself.
    """
    frame = measured.assign(infected=[True, False] * 6)
    features = candidate_features(frame, "condition")
    assert features == ("cell_area", "infected", "nucleus_area",
                        "pathogen_count", "plateID")
    assert "object_label" not in features     # identifies rather than describes
    assert "prcfo" not in features
    assert "well" not in features             # text

    # THE LABEL SUBTRACTION, observed on a NUMERIC label. Naming `condition`
    # proved nothing: it is object dtype, so it is dropped for being
    # non-numeric whether or not the subtraction exists -- mutation-proven,
    # deleting `name == label` from candidate_features left the whole file
    # green. `pathogen_count` is numeric, so it survives everything EXCEPT
    # being named as the label.
    assert "pathogen_count" in candidate_features(frame)
    assert "pathogen_count" not in candidate_features(frame, "pathogen_count")


# ---------------------------------------------------------------------------
# The statistics where a group runs out
# ---------------------------------------------------------------------------

def test_an_empty_group_scores_nan_rather_than_a_number():
    """A missing group must not look like "no separation".

    Every one of these could plausibly return 0, 0.5 or 1 from its own
    arithmetic, and any of those would put a feature nobody measured in one
    class somewhere definite in the ranking. NaN sorts last, which is the
    only honest place for it.
    """
    empty = np.array([])
    some = np.array([1.0, 2.0, 3.0])
    assert np.isnan(auc_of(empty, some))
    assert np.isnan(auc_of(some, empty))
    assert np.isnan(ks_of(empty, some))
    assert np.isnan(mutual_info_of(empty, some))
    assert np.isnan(cohen_d_of(np.array([1.0]), some))   # n < 2 on one side


def test_cohen_d_of_two_constant_groups_is_nan_not_infinity():
    """A pooled SD of zero is a division, and infinity ranks first.

    ``|d|`` is the ranking score, so an unguarded divide would send a column
    that is the same number everywhere to the top of the table for having no
    spread at all.
    """
    assert np.isnan(cohen_d_of(np.ones(4), np.ones(4)))
    assert np.isnan(cohen_d_of(np.ones(4), np.full(4, 7.0)))
    # A real difference with real spread still works.
    assert cohen_d_of(np.array([1.0, 2.0, 3.0]),
                      np.array([2.0, 3.0, 4.0])) == pytest.approx(1.0)


def test_when_the_ranking_statistic_is_uncomputable_the_other_three_survive():
    """One object in a class kills Cohen's d, and only Cohen's d.

    The row still has to carry the AUC, the KS and the class counts, because
    "the statistic you chose cannot be computed here" is a different statement
    from "this feature does not separate", and the table has to be able to
    make the first one.
    """
    frame = pd.DataFrame({"cls": ["a", "a", "a", "b"],
                          "x": [1.0, 2.0, 3.0, 9.0]})
    result = rank_features(frame, spec(label="cls", statistic=COHEN_D))
    score = result.score_for("x")
    assert np.isnan(score.score)              # the ranking statistic
    assert score.auc == pytest.approx(1.0)    # everything else is intact
    assert score.ks == pytest.approx(1.0)
    assert score.higher_in == "b"
    assert score.n_by_class == {"a": 3, "b": 1}
    assert score.smallest_class == 1
    assert "nan" in score.describe() and "AUC 1.000" in score.describe()
    # A class of one is an anecdote and the result says so.
    assert f"the smallest class has {score.smallest_class}" in result.notice
    assert score.smallest_class < MIN_PER_CLASS


# ---------------------------------------------------------------------------
# Choosing the statistic actually changes the answer
# ---------------------------------------------------------------------------

@pytest.fixture
def bimodal() -> pd.DataFrame:
    """Two features, six objects a side, both worked out by hand.

    ``monotone`` shifts by 2: a = 1…6, b = 3…8. Of the 36 pairs b is above in
    28 (four ties counting a half each), so AUC = 28/36 and the separation is
    5/9.

    ``bimodal`` is the blind spot itself: a is six zeros, b is three −3s and
    three +3s. Every pair balances, so AUC is exactly 0.5 and both AUC and
    Cohen's d score it 0. KS sees a gap of 0.5, and mutual information — which
    is not looking for an ordering at all — recovers the class completely.
    """
    return pd.DataFrame({
        "cls": ["a"] * 6 + ["b"] * 6,
        "monotone": [1.0, 2, 3, 4, 5, 6] + [3.0, 4, 5, 6, 7, 8],
        "bimodal": [0.0] * 6 + [-3.0, -3.0, -3.0, 3.0, 3.0, 3.0],
    })


def test_each_statistic_ranks_by_itself_and_they_disagree(bimodal):
    """The reason four are offered rather than one.

    A knockdown that makes some cells bigger and some smaller is a variance
    effect. The default cannot see it — AUC 0.5, d 0.0, so it ranks last — and
    KS and mutual information both put it first. A picker whose choice did not
    reach the ranking would leave every one of these orders identical, which
    is the failure this pins: each list below is a different list.
    """
    ranked = {stat: [s.feature for s in
                     rank_features(bimodal, spec(label="cls",
                                                 statistic=stat)).scores]
              for stat in (AUC, COHEN_D, KS, MUTUAL_INFO)}
    assert ranked[AUC] == ["monotone", "bimodal"]
    assert ranked[COHEN_D] == ["monotone", "bimodal"]
    assert ranked[KS] == ["bimodal", "monotone"]
    assert ranked[MUTUAL_INFO] == ["bimodal", "monotone"]

    by_auc = rank_features(bimodal, spec(label="cls")).score_for("bimodal")
    assert by_auc.auc == pytest.approx(0.5)
    assert by_auc.score == pytest.approx(0.0)
    assert by_auc.cohen_d == pytest.approx(0.0)
    assert by_auc.ks == pytest.approx(0.5)
    assert by_auc.mutual_info == pytest.approx(1.0)
    assert by_auc.is_shape_not_shift
    assert "SHAPE, NOT SHIFT" in by_auc.describe()
    assert rank_features(bimodal, spec(label="cls")).score_for(
        "monotone").score == pytest.approx(2 * 28 / 36 - 1)
    # Ranking by KS scores the same feature 0.5 rather than 0.
    by_ks = rank_features(bimodal, spec(label="cls", statistic=KS))
    assert by_ks.score_for("bimodal").score == pytest.approx(0.5)
    # And the caveat belongs to the statistic with the blind spot: ranking by
    # AUC warns that a spread-only feature sank, ranking by KS has nothing to
    # apologise for.
    assert "spread rather than level (bimodal)" in rank_features(
        bimodal, spec(label="cls")).notice
    assert by_ks.notice == ""


def test_mutual_information_never_reports_zero_for_noise(bimodal):
    """The failure mode the docstring promises, with a number on it.

    Twenty independent draws of pure noise, twenty objects a side: the *lowest*
    binned mutual information among them still claims a fifth of the class
    label is explained. That is the small-n bias, and it is why this statistic
    produces a confident ranking of nothing on a small table — while a genuinely
    constant feature is the one case it reports as exactly zero.
    """
    rng = np.random.default_rng(11)
    noise = [mutual_info_of(rng.normal(size=20), rng.normal(size=20))
             for _ in range(20)]
    assert min(noise) > 0.1
    # NOT `max(noise) < 1.0`: the product clamps with min(1.0, ...), so only
    # an exact 1.0 could ever break that, and it restates the clamp rather
    # than the small-n bias this test is named for. Noise on 20 points must
    # stay well below a real signal.
    assert max(noise) < 0.9
    assert mutual_info_of(np.ones(6), np.ones(6)) == 0.0
    assert mutual_info_of(bimodal["bimodal"].to_numpy()[:6],
                          bimodal["bimodal"].to_numpy()[6:]) == 1.0


# ---------------------------------------------------------------------------
# What each class carries for drawing
# ---------------------------------------------------------------------------

def test_each_class_carries_its_own_five_number_summary(measured):
    """The box the panel draws, per class, computed on that class alone.

    ctrl's ``cell_area`` is 100…150: median 125, quartiles 112.5 and 137.5.
    Summarising the pooled column instead — a plausible mistake, since the
    ranking already pools for the rank test — would give both classes the same
    box and a picture that contradicts an AUC of 1.
    """
    result = rank_features(measured, spec())
    ctrl, trt = result.score_for("cell_area").summaries
    assert (ctrl.level, ctrl.n) == ("ctrl", 6)
    assert (ctrl.median, ctrl.q25, ctrl.q75) == (125.0, 112.5, 137.5)
    assert (ctrl.low, ctrl.high) == (100.0, 150.0)
    assert (trt.median, trt.low, trt.high) == (225.0, 200.0, 250.0)
    assert ctrl.describe() == "ctrl: n=6, median 125 [112.5, 137.5]"
    assert not ctrl.is_low_n                  # six is above LOW_N


def test_a_class_of_four_is_flagged_as_thin_where_it_is_drawn():
    """The low-n flag rides on the summary, not only on the score.

    The panel draws one box per class; a box built from four objects has to be
    identifiable as such at the place it is drawn.
    """
    frame = pd.DataFrame({"cls": ["a"] * 4 + ["b"] * 8,
                          "x": [1.0, 2, 3, 4] + [5.0, 6, 7, 8, 9, 10, 11, 12]})
    thin, wide = rank_features(frame, spec(label="cls")).score_for(
        "x").summaries
    assert (thin.n, wide.n) == (4, 8)
    assert thin.is_low_n and not wide.is_low_n


# ---------------------------------------------------------------------------
# The result's own reporting
# ---------------------------------------------------------------------------

def test_top_hands_back_the_first_n_without_losing_the_count(measured):
    """Truncating for display must not change what the result claims to know."""
    result = rank_features(measured, spec())
    assert [s.feature for s in result.top(2)] == ["cell_area",
                                                  "pathogen_count"]
    assert len(result.top()) == 4             # no argument means all of them
    assert result.n_considered == 4
    assert len(result) == 4


def test_the_summary_counts_the_features_it_could_not_score(measured):
    """A feature missing from a ranking looks exactly like one that ranked last.

    So the count of unscoreable columns is in the one line a user reads.
    """
    result = rank_features(measured.assign(flat=1.0, empty=np.nan), spec())
    assert set(result.skipped) == {"flat", "empty"}
    assert "2 could not be scored" in result.summary()
    assert "4 features over 12 objects" in result.summary()


def test_when_nothing_can_be_scored_the_error_names_the_reasons(measured):
    """Three broken columns, three different reasons, all in the message.

    "none of your features could be scored" without the reasons is a dead end;
    with them the user can see that one column is constant, one is empty and
    one is not in this table at all.
    """
    frame = measured.assign(flat=1.0, empty=np.nan)
    with pytest.raises(ExplorerError) as caught:
        rank_features(frame, spec(features=("flat", "empty", "absent")))
    message = str(caught.value)
    assert "none of the 3 features could be scored" in message
    assert "flat: constant" in message
    assert "empty: no finite values" in message
    assert "absent: not a column of this table" in message


def test_a_text_column_asked_for_by_name_is_skipped_not_crashed(measured):
    """The spec can name any column; a text one has to fail as a skip."""
    result = rank_features(measured, spec(features=("cell_area", "well")))
    assert [s.feature for s in result.scores] == ["cell_area"]
    assert result.skipped["well"] == "no finite values"


def test_every_refusal_names_the_column_and_the_way_out(measured):
    """Four ways to have no usable class column, four sentences that say so.

    A ranking screen with a wrong column selected is the normal state of the
    screen on the way to the right one, so each refusal has to carry the
    instruction, not only the complaint — and none of them may be a traceback.
    """
    with pytest.raises(ExplorerError, match="a separation needs two "
                                            "populations"):
        rank_features(measured, ExplorerSpec())            # no label at all
    with pytest.raises(ExplorerError, match="no column called 'absent'"):
        rank_features(measured, spec(label="absent"))
    with pytest.raises(ExplorerError, match="at least two values"):
        rank_features(measured.assign(same="x"), spec(label="same"))
    crowded = pd.DataFrame({"cls": [f"g{i}" for i in range(13)],
                            "x": np.arange(13.0)})
    with pytest.raises(ExplorerError,
                       match="Filter to the comparison you mean"):
        rank_features(crowded, spec(label="cls"))          # 13 > MAX_CLASSES
    with pytest.raises(ExplorerError, match="no continuous columns"):
        rank_features(measured[["condition", "well"]], spec())


def test_asking_the_result_for_a_feature_it_never_ranked_says_how_many(
        measured):
    """``score_for`` on a name that is neither ranked nor skipped.

    The message has to distinguish "you truncated the list" from "this column
    could not be scored", because the fix differs — raise ``top``, or look at
    the column.
    """
    result = rank_features(measured, spec(top=1))
    with pytest.raises(ExplorerError) as caught:
        result.score_for("nucleus_area")
    assert "1 of 4 features" in str(caught.value)
    assert "was skipped" not in str(caught.value)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def test_editing_a_spec_copies_it_and_re_runs_the_normalisation():
    """``with_*`` returns a new spec that has been through ``__post_init__``.

    The spec is frozen so the panel can keep a history and compare two specs
    with ``==``. An edit that wrote the field in place would corrupt the
    history; one that skipped normalisation would let an untrimmed label or an
    empty feature name through the back door.
    """
    original = spec(features=("a",))
    assert original.with_statistic(KS).statistic == KS
    assert original.statistic == AUC                    # unchanged
    assert original.with_label("  other  ").label == "other"
    assert original.with_features(["", "x", "y"]).features == ("x", "y")
    assert original.features == ("a",)
    assert original.with_statistic(AUC) == original     # value equality
    with pytest.raises(ExplorerError, match="unknown separation statistic"):
        original.with_statistic("ttest")


def test_a_spec_normalises_what_the_controls_hand_it():
    """The spinboxes and the restored session file both come through here.

    ``bins`` under two is not a histogram, a negative permutation count is not
    a null, and a ``top`` of zero is a ranking with nothing in it — the first
    two are clamped because a control can produce them by being dragged, and
    the third is refused because it can only come from a file or a caller.
    """
    lenient = ExplorerSpec(label=" cls ", bins=0, n_permutations=-5, seed=2.0)
    assert lenient.label == "cls"
    assert lenient.bins == 2
    assert lenient.n_permutations == 0
    assert isinstance(lenient.seed, int)
    with pytest.raises(ExplorerError, match="top must be at least 1, not 0"):
        ExplorerSpec(label="cls", top=0)


def test_the_summary_reports_the_null_and_the_caveats_it_collected():
    """One line under the table has to carry every qualification at once.

    A threshold from shuffling, how many features beat it, and the fact that
    two objects a side is an anecdote — a summary that dropped any of them
    would leave a screenshot of the table looking like a result.
    """
    frame = pd.DataFrame({"cls": ["a", "a", "b", "b"],
                          "x": [1.0, 2.0, 8.0, 9.0]})
    with_null = rank_features(frame, spec(label="cls", n_permutations=5,
                                          seed=2))
    line = with_null.summary()
    assert "shuffling the labels reaches" in line
    assert "feature(s) beat it" in line
    assert "anecdote with a number on it" in line
    assert with_null.null_threshold is not None

    without = rank_features(frame, spec(label="cls"))
    assert without.null_threshold is None
    # "not run" is not "passed", so the whole ranking comes back and the line
    # says the comparisons are unaccounted for.
    assert without.above_null() == without.scores
    assert "turn on the label-shuffling null" in without.summary()


def test_describe_says_what_is_about_to_be_ranked_and_by_what():
    """The line beside the button, in the two states it has."""
    assert spec().describe() == ("every continuous column split by condition, "
                                 "ranked by AUC")
    assert spec(features=("a", "b"), statistic=KS).describe() == (
        "2 features split by condition, ranked by KS")
    assert ExplorerSpec().describe().endswith(
        "split by (no class column), ranked by AUC")


def test_a_saved_spec_survives_a_key_this_version_does_not_know():
    """Restoring a session written by another build must not raise.

    ``features: null`` is what a spec saved before a feature list was chosen
    round-trips as, and an unknown key is what a newer build writes.
    """
    restored = ExplorerSpec.from_dict({"label": " cls ", "features": None,
                                       "statistic": KS, "top": 3,
                                       "from_a_later_version": 42})
    assert restored == ExplorerSpec(label="cls", statistic=KS, top=3)
    assert restored.features == ()
    assert ExplorerSpec.from_json(restored.to_json()) == restored


# ---------------------------------------------------------------------------
# The null on a table too big to shuffle whole
# ---------------------------------------------------------------------------

def test_a_big_table_shuffles_a_seeded_subsample_and_says_so():
    """Over NULL_MAX_ROWS the null is computed on a sample, which is a caveat.

    A permutation null costs a full pass per shuffle, so past 20 000 rows it
    runs on a seeded subsample — and a threshold computed on part of the table
    is a different claim from one computed on all of it, so the result says
    which it is. Same seed, same number: a ranking that moved between two
    identical runs would be unciteable.
    """
    rows = NULL_MAX_ROWS + 1
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "cls": np.where(np.arange(rows) % 2 == 0, "a", "b"),
        "x": rng.normal(size=rows),
    })
    result = rank_features(frame, spec(label="cls", n_permutations=3, seed=5))
    assert "20,000-row subsample" in result.notice
    assert result.null_threshold is not None
    again = rank_features(frame, spec(label="cls", n_permutations=3, seed=5))
    assert result.null_threshold == again.null_threshold
    # Pure noise: the one feature does not clear its own shuffle.
    assert result.above_null() == ()
    # A table that fits does not carry the caveat.
    small = frame.iloc[:100]
    assert "subsample" not in rank_features(
        small, spec(label="cls", n_permutations=3, seed=5)).notice


# ---------------------------------------------------------------------------
# Direction, with more than two classes
# ---------------------------------------------------------------------------

def test_the_class_that_separates_by_being_low_is_not_called_the_high_one():
    """One-vs-rest with the winning class *below* everything else.

    ``c`` separates perfectly and is the lowest, so the separation is 1.0 and
    the direction must point at the rest. Taking the winning level as
    "higher_in" — the obvious shortcut, since it is the level that won — would
    print exactly the wrong sentence on a perfect result.
    """
    frame = pd.DataFrame({
        "cls": ["a", "a", "b", "b", "c", "c"],
        "x": [50.0, 55.0, 51.0, 56.0, 1.0, 2.0],
    })
    score = rank_features(frame, spec(label="cls")).score_for("x")
    assert score.score == pytest.approx(1.0)
    assert score.auc == pytest.approx(0.0)    # nothing in c outranks the rest
    assert score.against == "rest"
    assert score.higher_in == "rest"
    assert "higher in rest" in score.describe()


# ---------------------------------------------------------------------------
# The drawing data
# ---------------------------------------------------------------------------

def test_a_feature_with_nothing_measured_draws_nothing(measured):
    """No finite value anywhere: empty edges and no series, not a NaN range."""
    edges, counts = distributions(measured.assign(gone=np.nan), "gone",
                                  "condition")
    assert edges.size == 0
    assert counts == {}


def test_a_constant_feature_still_gets_a_bin_range_it_can_be_drawn_in():
    """A zero-width range makes ``np.histogram`` raise, so it is widened.

    ``linspace(5, 5, 17)`` is seventeen copies of 5, and ``np.histogram`` on
    non-increasing edges is a ValueError — a constant column would take the
    panel down rather than draw a single bar. Widening by 5% of the value (or
    0.5 when the value is zero, since 5% of zero is still zero) keeps it
    drawable.
    """
    frame = pd.DataFrame({"cls": ["a", "a", "b", "b"], "flat": [5.0] * 4,
                          "zeros": [0.0] * 4})
    edges, counts = distributions(frame, "flat", "cls", bins=4)
    assert list(edges) == [5.0, 5.0625, 5.125, 5.1875, 5.25]
    assert np.all(np.diff(edges) > 0)
    assert counts["a"].tolist() == [2, 0, 0, 0]
    assert counts["b"].tolist() == [2, 0, 0, 0]
    zero_edges, zero_counts = distributions(frame, "zeros", "cls", bins=2)
    assert list(zero_edges) == [0.0, 0.25, 0.5]
    assert zero_counts["a"].sum() == 2


def test_the_drawn_bins_span_every_class_together(measured):
    """Shared edges, or the two histograms are not comparable pictures.

    Every object is drawn, including the largest: the top bin is closed at the
    right, so the 250 that defines the upper edge lands in it rather than
    falling off the chart — which is why trt is 2 and 4 rather than 3 and 3.
    """
    edges, counts = distributions(measured, "cell_area", "condition", bins=5)
    assert list(edges) == [100.0, 130.0, 160.0, 190.0, 220.0, 250.0]
    assert counts["ctrl"].tolist() == [3, 3, 0, 0, 0]
    assert counts["trt"].tolist() == [0, 0, 0, 2, 4]
    assert counts["ctrl"].sum() + counts["trt"].sum() == 12


# ---------------------------------------------------------------------------
# Rows with no class label
# ---------------------------------------------------------------------------

def test_an_object_with_no_class_enters_no_score(measured):
    """Half a plate annotated is the normal case, not the exception.

    An object with no condition cannot be on either side of a separation, so
    appending twelve of them — with feature values far outside both classes —
    must leave every score, every count and the whole order untouched.
    """
    unlabelled = measured.assign(condition=np.nan,
                                 cell_area=measured["cell_area"] * 100.0)
    both = pd.DataFrame({name: list(measured[name]) + list(unlabelled[name])
                         for name in measured.columns})
    before = rank_features(measured, spec())
    after = rank_features(both, spec())
    assert [s.feature for s in after.scores] == [s.feature
                                                 for s in before.scores]
    assert [s.score for s in after.scores] == [s.score for s in before.scores]
    assert after.score_for("cell_area").n_by_class == {"ctrl": 6, "trt": 6}
    assert after.classes == ("ctrl", "trt")


# ---------------------------------------------------------------------------
# Label regressions. Each asserts the behaviour documented by the module.
# ---------------------------------------------------------------------------

def test_rows_that_enter_no_score_must_not_move_the_null_threshold():
    """The null has to be a null of the ranking it is calibrating.

    Twelve rows are appended that are entirely empty — no class, no
    measurements, nothing that any statistic can touch. The ranking is
    byte-identical, as the test above shows. The null threshold is not:
    ``_null_threshold`` shuffles ``keys`` including the ``""`` placeholder
    that stands for "no class", so in each permutation only a fraction of the
    scored rows receive a real class label and the rest are pooled into
    "rest". The permutation therefore compares groups of sizes the real
    ranking never used, and the 95th percentile it produces is a threshold for
    a different experiment.

    Observed with this data: 0.431 with the empty rows absent, 1.000 with them
    present — so ``real``, a genuine two-SD shift, goes from beating the null
    to being discarded by it. The failure direction is data-dependent, but it
    is always a threshold nobody asked for.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"cls": ["a"] * 20 + ["b"] * 20})
    for i in range(5):
        frame[f"noise{i}"] = rng.normal(size=40)
    frame["real"] = np.concatenate([rng.normal(0.0, 1.0, 20),
                                    rng.normal(2.0, 1.0, 20)])
    padded = pd.DataFrame({name: list(values) + [np.nan] * 12
                           for name, values in frame.items()})

    settings = spec(label="cls", n_permutations=30, seed=7)
    plain = rank_features(frame, settings)
    with_empty_rows = rank_features(padded, settings)

    assert [s.score for s in with_empty_rows.scores] == [s.score
                                                         for s in plain.scores]
    assert with_empty_rows.null_threshold == pytest.approx(
        plain.null_threshold)
    assert [s.feature for s in with_empty_rows.above_null()] == ["real"]


def test_a_date_column_can_be_used_as_the_class_column():
    """``candidate_labels`` offers an acquisition date, and it cannot be used.

    ``_class_levels`` builds the per-row keys with ``frame[label].astype(str)``
    — pandas renders a column of midnight timestamps as ``'2024-01-01'`` —
    and builds the level names with ``str(v)`` over ``.unique()``, which gives
    ``'2024-01-01 00:00:00'``. The two never compare equal, so every class
    holds zero objects.

    ``rank_features`` then raises "none of the 1 features could be scored
    against 'day' ... no object with a value in 2024-01-01 00:00:00,
    2024-01-02 00:00:00", and ``distributions`` — which cannot raise — returns
    a histogram of zeros per class and the panel draws a blank chart with no
    message at all. A timestamp column that carries a time of day works, which
    is what makes this look like bad data rather than a bug.
    """
    frame = pd.DataFrame({
        "day": pd.to_datetime(["2024-01-01", "2024-01-01",
                               "2024-01-02", "2024-01-02"]),
        "x": [1.0, 2.0, 8.0, 9.0],
    })
    assert "day" in candidate_labels(frame)
    _edges, counts = distributions(frame, "x", "day")
    assert sum(int(c.sum()) for c in counts.values()) == 4
    score = rank_features(frame, spec(label="day")).score_for("x")
    assert score.score == pytest.approx(1.0)
    assert sorted(score.n_by_class.values()) == [2, 2]


def test_a_class_whose_name_is_the_empty_string_is_still_a_class():
    """The sentinel for "no label" collides with a real label.

    ``_class_levels`` marks unlabelled rows by writing ``""`` into the key
    array, and takes the level names from ``dropna().unique()`` — which keeps
    a genuine ``""``. ``rank_features`` then drops every row whose key is
    ``""`` as unlabelled, so the class has no objects left and every feature
    is skipped with the malformed reason "no object with a value in " (an
    empty class name), ending in "none of the 1 features could be scored".

    A blank cell reads back as ``''`` from any table loaded with
    ``keep_default_na=False`` or written through ``fillna('')``, and
    ``candidate_labels`` offers the column, so nothing warns first.
    """
    frame = pd.DataFrame({"cls": ["", "", "", "trt", "trt", "trt"],
                          "x": [1.0, 2.0, 3.0, 9.0, 10.0, 11.0]})
    assert "cls" in candidate_labels(frame)
    score = rank_features(frame, spec(label="cls")).score_for("x")
    assert score.score == pytest.approx(1.0)
    assert score.n_by_class == {"": 3, "trt": 3}
