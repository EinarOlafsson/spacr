"""V4 — feature ranking: four planted features with separations worked out first.

Eight objects, four in each class, and four features chosen so that every
statistic below can be computed with a pencil::

    class    perfect  partial  none  shape
    a              1        1     1   -1.0
    a              2        2     2   -0.5
    a              3        3     3    0.5
    a              4        4     4    1.0
    b              5        3     1   -4.0
    b              6        4     2   -2.0
    b              7        5     3    2.0
    b              8        6     4    4.0

* ``perfect``: every b above every a, so AUC = 1 and the separation
  ``|2·AUC − 1|`` is **1.0**. Cohen's d is 4 / sqrt(5/3) = 3.0984.
* ``partial``: counting the 16 pairs — 4 + 4 + 3.5 + 2.5 = 14 with b above a
  (the two ties contributing a half each) — AUC = 14/16 = **0.875**, so the
  separation is 0.75.
* ``none``: the two classes are identical, AUC = 0.5, separation **0**.
* ``shape``: the classes have the same median and different spreads. Every
  comparison balances, so AUC is exactly 0.5 and the separation is **0** —
  and that is the blind spot, so KS is asserted at 0.5 and the feature is
  flagged "shape, not shift".

The ranking must therefore be perfect, partial, then the two at zero — which is
the assertion that says the ranking is a ranking and not an ordering of column
names.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.feature_rank import (
    AUC, COHEN_D, KS, MUTUAL_INFO, SHAPE_NOT_SHIFT_KS, STATISTICS,
    STATISTIC_FAILURE_MODES, STATISTIC_LABELS, ExplorerError, ExplorerSpec,
    auc_of, candidate_features, candidate_labels, cohen_d_of, distributions,
    ks_of, mutual_info_of, rank_features,
)


@pytest.fixture
def planted() -> pd.DataFrame:
    return pd.DataFrame({
        "cls": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "perfect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "partial": [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0],
        "none": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        "shape": [-1.0, -0.5, 0.5, 1.0, -4.0, -2.0, 2.0, 4.0],
    })


def spec(**kwargs) -> ExplorerSpec:
    kwargs.setdefault("label", "cls")
    return ExplorerSpec(**kwargs)


# ---------------------------------------------------------------------------
# The statistics, one pair of arrays at a time
# ---------------------------------------------------------------------------

def test_auc_is_the_hand_counted_pair_fraction():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert auc_of(a, np.array([5.0, 6.0, 7.0, 8.0])) == 1.0
    assert auc_of(a, np.array([3.0, 4.0, 5.0, 6.0])) == pytest.approx(14 / 16)
    assert auc_of(a, a) == 0.5


def test_a_tie_counts_as_half():
    """Two identical values contribute exactly 0.5, not 0 or 1."""
    assert auc_of(np.array([1.0]), np.array([1.0])) == 0.5
    assert auc_of(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0.75


def test_auc_is_invariant_under_a_monotone_transform():
    """The reason it is the default: log(area) must rank like area."""
    a = np.array([1.0, 10.0, 100.0])
    b = np.array([2.0, 20.0, 5000.0])
    assert auc_of(a, b) == auc_of(np.log(a), np.log(b))
    # Cohen's d is not, which is the failure mode the docstring names.
    assert cohen_d_of(a, b) != pytest.approx(cohen_d_of(np.log(a), np.log(b)))


def test_cohen_d_is_the_pooled_standardised_difference():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    assert cohen_d_of(a, b) == pytest.approx(4.0 / math.sqrt(5.0 / 3.0))
    assert cohen_d_of(b, a) == pytest.approx(-4.0 / math.sqrt(5.0 / 3.0))


def test_ks_is_the_largest_gap_between_the_cdfs():
    a = np.array([-1.0, -0.5, 0.5, 1.0])
    b = np.array([-4.0, -2.0, 2.0, 4.0])
    # The CDFs are 0/.25/.5/.75/1 and .25/.5/.5/.5/.5/.75/1 as computed in the
    # module docstring; the biggest gap is 0.5.
    assert ks_of(a, b) == pytest.approx(0.5)
    assert ks_of(a, a) == 0.0


def test_ks_never_reports_a_gap_a_tie_created():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert ks_of(a, b) == 0.0


def test_mutual_information_is_zero_for_identical_classes_and_one_for_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert mutual_info_of(a, a.copy(), bins=4) == pytest.approx(0.0, abs=1e-9)
    assert mutual_info_of(a, a + 10.0, bins=4) == pytest.approx(1.0)


def test_mutual_information_of_a_constant_is_zero():
    ones = np.ones(5)
    assert mutual_info_of(ones, ones, bins=4) == 0.0


# ---------------------------------------------------------------------------
# The ranking
# ---------------------------------------------------------------------------

def test_the_planted_features_come_out_in_the_planted_order(planted):
    result = rank_features(planted, spec())
    assert [s.feature for s in result.scores][:2] == ["perfect", "partial"]
    assert result.scores[0].score == pytest.approx(1.0)
    assert result.scores[1].score == pytest.approx(0.75)
    assert {s.feature for s in result.scores[2:]} == {"none", "shape"}
    assert all(s.score == pytest.approx(0.0) for s in result.scores[2:])


def test_every_statistic_is_reported_for_every_feature(planted):
    score = rank_features(planted, spec()).score_for("perfect")
    assert score.auc == pytest.approx(1.0)
    assert score.cohen_d == pytest.approx(4.0 / math.sqrt(5.0 / 3.0))
    assert score.ks == pytest.approx(1.0)
    assert 0.0 <= score.mutual_info <= 1.0
    assert score.higher_in == "b"
    assert score.against == "a"


def test_the_direction_is_reported_and_correct(planted):
    flipped = planted.assign(perfect=-planted["perfect"])
    score = rank_features(flipped, spec()).score_for("perfect")
    assert score.score == pytest.approx(1.0)
    assert score.higher_in == "a"


def test_the_variance_only_feature_is_flagged_rather_than_missed(planted):
    """AUC's blind spot, reported on the row it applies to."""
    result = rank_features(planted, spec())
    shape = result.score_for("shape")
    assert shape.auc == pytest.approx(0.5)
    assert shape.score == pytest.approx(0.0)
    assert shape.ks == pytest.approx(0.5)
    assert shape.ks >= SHAPE_NOT_SHIFT_KS
    assert shape.is_shape_not_shift
    assert not result.score_for("none").is_shape_not_shift
    assert "spread rather than level" in result.notice
    assert "SHAPE, NOT SHIFT" in shape.describe()


def test_ranking_by_ks_puts_the_variance_feature_back_in_play(planted):
    """The point of offering more than one statistic."""
    by_auc = rank_features(planted, spec(statistic=AUC))
    by_ks = rank_features(planted, spec(statistic=KS))
    assert by_auc.score_for("shape").score == pytest.approx(0.0)
    assert by_ks.score_for("shape").score == pytest.approx(0.5)
    assert [s.feature for s in by_ks.scores].index("shape") < \
        [s.feature for s in by_auc.scores].index("shape")


def test_cohen_d_and_auc_disagree_about_a_tail(planted):
    """Two features, six objects a side, both worked out by hand:

    ``consistent`` shifts every object by 1 — a = 1..6, b = 2..7. AUC counts
    23.5 of 36 pairs (the five ties contributing a half each) = 0.6528, so the
    separation is 0.3056; d is 1 / sqrt(3.5) = 0.5345.

    ``tail`` moves ONE object a long way — a = six 1s, b = five 1s and a 100.
    AUC is (15 + 6) / 36 = 0.5833, a separation of 0.1667; d is 16.5 / 28.58 =
    0.5774, which is *larger* than the consistent feature's.

    So the two statistics rank them in opposite orders, and that is the point:
    d is answering "how far apart are the means in SD units", which one object
    can carry, and AUC is answering "how reliably does this order the objects",
    which it cannot.
    """
    frame = pd.DataFrame({
        "cls": ["a"] * 6 + ["b"] * 6,
        "consistent": [1.0, 2, 3, 4, 5, 6] + [2.0, 3, 4, 5, 6, 7],
        "tail": [1.0, 1, 1, 1, 1, 1] + [1.0, 1, 1, 1, 1, 100],
    })
    by_auc = rank_features(frame, spec(statistic=AUC))
    by_d = rank_features(frame, spec(statistic=COHEN_D))
    assert by_auc.scores[0].feature == "consistent"
    assert by_auc.score_for("consistent").score == pytest.approx(
        2 * 23.5 / 36 - 1)
    assert by_d.scores[0].feature == "tail"
    assert by_d.score_for("tail").score == pytest.approx(0.5774, abs=1e-4)
    assert by_d.score_for("consistent").score == pytest.approx(
        1.0 / math.sqrt(3.5))


def test_the_statistic_and_its_blind_spot_are_both_documented():
    """The 'say yours' requirement, kept where it cannot rot."""
    assert set(STATISTIC_LABELS) == set(STATISTICS)
    assert set(STATISTIC_FAILURE_MODES) == set(STATISTICS)
    assert "spread" in STATISTIC_FAILURE_MODES[AUC]
    assert "biased upward" in STATISTIC_FAILURE_MODES[MUTUAL_INFO]


def test_top_keeps_the_best_and_counts_the_rest(planted):
    result = rank_features(planted, spec(top=2))
    assert [s.feature for s in result.scores] == ["perfect", "partial"]
    assert result.n_considered == 4
    assert len(result) == 2
    assert "4 features over 8 objects" in result.summary()


def test_n_per_class_travels_with_every_score(planted):
    score = rank_features(planted, spec()).scores[0]
    assert score.n_by_class == {"a": 4, "b": 4}
    assert score.smallest_class == 4
    assert score.is_low_n                      # 4 <= LOW_N
    assert "n=4 in the smaller class" in score.describe()


def test_a_tiny_class_is_called_an_anecdote():
    frame = pd.DataFrame({"cls": ["a", "a", "a", "b"],
                          "x": [1.0, 2.0, 3.0, 9.0]})
    result = rank_features(frame, spec())
    assert "anecdote" in result.notice


# ---------------------------------------------------------------------------
# Features that cannot be scored are named, not dropped
# ---------------------------------------------------------------------------

def test_a_constant_feature_is_skipped_with_the_reason(planted):
    frame = planted.assign(flat=1.0)
    result = rank_features(frame, spec(features=("perfect", "flat")))
    assert [s.feature for s in result.scores] == ["perfect"]
    assert "constant" in result.skipped["flat"]


def test_an_all_missing_feature_is_skipped_with_the_reason(planted):
    frame = planted.assign(gone=np.nan)
    result = rank_features(frame, spec(features=("perfect", "gone")))
    assert "no finite values" in result.skipped["gone"]


def test_a_feature_measured_in_only_one_class_is_skipped(planted):
    frame = planted.assign(
        half=[1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    result = rank_features(frame, spec(features=("perfect", "half")))
    assert "no object with a value in b" in result.skipped["half"]


def test_asking_for_a_column_that_is_not_there_says_so(planted):
    result = rank_features(planted, spec(features=("perfect", "nope")))
    assert result.skipped["nope"] == "not a column of this table"


def test_score_for_a_skipped_feature_explains_rather_than_raising_blankly(
        planted):
    frame = planted.assign(flat=1.0)
    result = rank_features(frame, spec(features=("perfect", "flat")))
    with pytest.raises(ExplorerError, match="constant"):
        result.score_for("flat")


def test_no_class_column_is_a_sentence_not_a_traceback(planted):
    with pytest.raises(ExplorerError, match="which class"):
        rank_features(planted, ExplorerSpec())
    with pytest.raises(ExplorerError, match="nothing to separate"):
        rank_features(planted.assign(one="x"), spec(label="one"))
    with pytest.raises(ExplorerError, match="no column called"):
        rank_features(planted, spec(label="absent"))


def test_a_table_with_no_continuous_columns_says_so():
    frame = pd.DataFrame({"cls": ["a", "b"], "other": ["p", "q"]})
    with pytest.raises(ExplorerError, match="no continuous columns"):
        rank_features(frame, spec())


# ---------------------------------------------------------------------------
# More than two classes
# ---------------------------------------------------------------------------

def test_three_classes_are_scored_one_vs_rest_and_the_best_named():
    frame = pd.DataFrame({
        "cls": ["a", "a", "b", "b", "c", "c"],
        "x": [1.0, 2.0, 1.5, 2.5, 90.0, 95.0],
    })
    result = rank_features(frame, spec())
    score = result.score_for("x")
    assert result.classes == ("a", "b", "c")
    assert score.higher_in == "c"
    assert score.against == "rest"
    assert score.score == pytest.approx(1.0)


def test_too_many_classes_is_refused_with_the_advice():
    frame = pd.DataFrame({"cls": [f"c{i}" for i in range(20)],
                          "x": np.arange(20.0)})
    with pytest.raises(ExplorerError, match="Filter to the comparison"):
        rank_features(frame, spec())


# ---------------------------------------------------------------------------
# The multiple-comparisons null
# ---------------------------------------------------------------------------

def test_without_the_null_the_summary_says_the_comparisons_are_unaccounted(
        planted):
    result = rank_features(planted, spec())
    assert result.null_threshold is None
    assert "comparisons with one read" in result.summary()
    assert result.above_null() == result.scores


def test_the_shuffle_null_is_reproducible_and_bounds_the_noise():
    """Pure noise: the ranking's winner must not beat its own shuffle."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"cls": ["a"] * 30 + ["b"] * 30})
    for i in range(40):
        frame[f"f{i}"] = rng.normal(size=60)
    result = rank_features(frame, spec(n_permutations=30, seed=1))
    again = rank_features(frame, spec(n_permutations=30, seed=1))
    assert result.null_threshold == again.null_threshold
    assert result.null_threshold > 0.0
    # Forty noise features: the best of them does not clear the null that
    # forty noise features produce.
    assert result.above_null() == ()
    assert "by chance" not in result.summary()
    assert "shuffling the labels reaches" in result.summary()


def test_a_real_effect_clears_the_null():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"cls": ["a"] * 30 + ["b"] * 30})
    for i in range(10):
        frame[f"noise{i}"] = rng.normal(size=60)
    frame["real"] = np.concatenate([rng.normal(0, 1, 30),
                                    rng.normal(6, 1, 30)])
    result = rank_features(frame, spec(n_permutations=30, seed=1))
    assert result.scores[0].feature == "real"
    assert [s.feature for s in result.above_null()] == ["real"]


# ---------------------------------------------------------------------------
# The spec, and the drawing data
# ---------------------------------------------------------------------------

def test_the_spec_round_trips_through_json():
    original = spec(features=("a", "b"), statistic=KS, top=5,
                    n_permutations=10, seed=3)
    assert ExplorerSpec.from_json(original.to_json()) == original


def test_an_unknown_statistic_is_refused_where_it_is_written():
    with pytest.raises(ExplorerError, match="unknown separation statistic"):
        ExplorerSpec(label="cls", statistic="ttest")
    with pytest.raises(ExplorerError, match="top must be"):
        ExplorerSpec(label="cls", top=0)


def test_candidates_offer_the_right_columns(planted):
    assert candidate_labels(planted) == ("cls",)
    assert candidate_features(planted, "cls") == (
        "none", "partial", "perfect", "shape")


def test_the_drawn_histograms_share_their_bin_edges(planted):
    edges, counts = distributions(planted, "perfect", "cls", bins=4)
    assert len(edges) == 5
    assert set(counts) == {"a", "b"}
    # Shared edges, so the two class histograms are comparable: a's four
    # objects are in the low bins and b's in the high ones.
    assert counts["a"].sum() == 4 and counts["b"].sum() == 4
    assert counts["a"][-1] == 0 and counts["b"][0] == 0


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot, planted):
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget.set_frame(planted)
    return widget


def test_the_panel_ranks_and_fills_the_table(panel):
    result = panel.rank_now()
    assert result is not None
    assert panel.table.rowCount() == 4
    assert panel.table.item(0, 0).text() == "perfect"
    assert panel.table.item(0, 1).text() == "1.000"
    assert panel.table.item(0, 2).text() == "1.000"       # AUC
    assert panel.table.item(0, 4).text() == "b"           # higher in
    assert panel.table.item(0, 5).text() == "4"           # min n


def test_the_blind_spot_is_on_screen(panel):
    panel.rank_now()
    assert "cannot see" in panel._blind.text()
    assert "spread" in panel._blind.text()
    index = panel._statistic.findData(KS)
    panel._statistic.setCurrentIndex(index)
    panel.rank_now()
    assert "direction" in panel._blind.text()


def test_changing_the_statistic_re_ranks(panel):
    panel.rank_now()
    assert panel.result.scores[0].feature == "perfect"
    panel._statistic.setCurrentIndex(panel._statistic.findData(KS))
    result = panel.rank_now()
    assert result.spec.statistic == KS
    assert result.score_for("shape").score == pytest.approx(0.5)


def test_a_feature_that_cannot_be_ranked_leaves_a_message_not_a_crash(qtbot):
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"cls": ["a", "b"], "other": ["p", "q"]}))
    assert widget.rank_now() is None
    assert "no continuous columns" in widget.summary()
    widget.close()


def test_selecting_a_row_announces_the_feature(panel, qtbot):
    panel.rank_now()
    seen = []
    panel.feature_selected.connect(seen.append)
    panel.table.selectRow(1)
    assert seen and seen[-1] == "partial"
    assert panel.selected_feature() == "partial"


def test_pushing_a_spec_in_updates_the_controls(panel):
    panel.set_spec(spec(statistic=MUTUAL_INFO, top=3, n_permutations=10))
    assert panel._statistic.currentData() == MUTUAL_INFO
    assert panel._top.value() == 3
    assert panel._null.isChecked()
    assert panel.result is not None and len(panel.result) == 3


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def test_the_screen_ranks_a_computed_column_alongside_the_measured_ones(
        qtbot, planted):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    from spacr.qt.widgets.formula import ColumnFormula

    screen = FeatureExplorerScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(planted)
    screen.formulas.add_formula(ColumnFormula("doubled", "perfect * 2"))
    result = screen.explorer.rank_now()
    assert "doubled" in [s.feature for s in result.scores]
    # A monotone transform of `perfect`, so a rank statistic scores it the same.
    assert result.score_for("doubled").score == pytest.approx(
        result.score_for("perfect").score)
    screen.close()


def test_the_screen_exports_every_statistic_not_only_the_ranked_one(
        qtbot, tmp_path, planted):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen

    screen = FeatureExplorerScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(planted)
    screen.explorer.rank_now()
    path = screen.export_ranking(str(tmp_path / "ranking.csv"))
    written = pd.read_csv(path)
    assert list(written["feature"])[:2] == ["perfect", "partial"]
    assert set(written.columns) >= {"auc", "cohen_d", "ks", "mutual_info",
                                    "shape_not_shift", "min_n"}
    assert bool(written.loc[written["feature"] == "shape",
                            "shape_not_shift"].iloc[0])
    screen.close()


def test_the_ranking_follows_the_shared_filter(qtbot, planted):
    """A separation is a statement about a population, so narrowing the
    population is what the filter is for."""
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.selection import CategoryFilter, DataFilter
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen

    link = LinkedSelection()
    screen = FeatureExplorerScreen(link=link, threaded=False)
    qtbot.addWidget(screen)
    screen.set_frame(pd.concat([planted.assign(plate="p1"),
                                planted.assign(plate="p2")]))
    before = screen.explorer.rank_now()
    assert before.n_rows == 16
    link.set_filter(DataFilter([CategoryFilter("plate", ("p1",))]))
    screen._on_filter_changed()
    after = screen.explorer.rank_now()
    assert after.n_rows == 8
    screen.close()


def test_the_screen_registers_once():
    """One row in `spacr.qt.SELF_REGISTERING_MODULES` turns it on."""
    from spacr.qt import app as app_mod
    from spacr.qt.screens import feature_explorer as screen_mod

    apps = list(app_mod.APPS)
    try:
        if any(row[0] == screen_mod.APP_KEY for row in app_mod.APPS):
            assert screen_mod.register() is False
        else:
            assert screen_mod.register() is True
            assert screen_mod.register() is False
            meta = app_mod.APP_META[screen_mod.APP_KEY]
            assert meta["intro"] == screen_mod.APP_INTRO
            assert len(meta["translations"]) == 9
    finally:
        app_mod.APPS[:] = apps
        app_mod.APP_META.pop(screen_mod.APP_KEY, None)
        app_mod.APP_FACTORIES.pop(screen_mod.APP_KEY, None)
        app_mod.APP_STAGE.pop(screen_mod.APP_KEY, None)
