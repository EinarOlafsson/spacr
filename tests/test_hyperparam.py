"""Tests for :mod:`spacr.hyperparam`.

Everything here is CPU-only, offline and small. No CNN is trained and no real
UMAP is fitted: the expensive fit is always injected, because what is under
test is the *search* — the ordering, the failure bookkeeping, the early stop,
the leak guard and the honesty of the reporting — not somebody else's
optimiser.

The one test that matters most is
``TestUmapCriteria::test_two_criteria_pick_different_winners``. If that ever
goes green by accident it means the criteria stopped disagreeing, and a UMAP
search that reports a single winner has quietly become a lie.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from spacr.hyperparam import (
    APP_CRITERIA,
    DEFAULT_SPACES,
    SearchResult,
    SearchSpace,
    Trial,
    UMAP_CRITERIA,
    UMAP_MISSING_MESSAGE,
    build_folds,
    build_sklearn_model,
    classify_cv_fit_fn,
    cv_search,
    format_search,
    grid_search,
    local_direction_search,
    load_search_data,
    random_search,
    run_search_for_app,
    sklearn_cv_fit_fn,
    umap_available,
    umap_search,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def linear_fit(params):
    """Deterministic score: monotone in ``a``, mildly in ``b``."""
    return params["a"] + 0.001 * params["b"]


@pytest.fixture
def space():
    """Two parameters, six configurations."""
    return SearchSpace({"b": [1, 2], "a": [0.1, 0.5, 0.9]})


@pytest.fixture
def clustered_features():
    """Three well-separated Gaussian blobs in 5-D, plus their labels."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.35, size=(20, 5)) for c in (0.0, 6.0, 12.0)])
    y = np.repeat([0, 1, 2], 20)
    return X, y


@pytest.fixture
def tear_and_merge(clustered_features):
    """Two contrived embeddings that the two criteria rank in opposite orders.

    ``tear`` splits every true cluster in half and throws the halves apart: it
    invents no neighbours (high trustworthiness) but destroys real ones (low
    continuity). ``merge`` fuses two distinct clusters: it keeps real
    neighbours (high continuity) but invents plenty (low trustworthiness).
    """
    X, y = clustered_features
    from sklearn.decomposition import PCA
    base = PCA(n_components=2, random_state=0).fit_transform(X)
    rng = np.random.default_rng(1)

    tear = base.copy()
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        half = len(idx) // 2
        tear[idx[:half]] += np.array([0.0, 60.0])
        tear[idx[half:]] -= np.array([0.0, 60.0])

    merge = base.copy()
    merge[y == 1] = base[y == 0].mean(axis=0) + rng.normal(
        0, 0.3, size=((y == 1).sum(), 2))
    return X, y, tear, merge


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def test_grid_enumerates_the_full_product(self, space):
        grid = space.grid()
        assert len(grid) == space.size() == 6
        assert {tuple(sorted(d.items())) for d in grid} == {
            tuple(sorted({"a": a, "b": b}.items()))
            for a in (0.1, 0.5, 0.9) for b in (1, 2)}

    def test_grid_order_is_deterministic_and_name_sorted(self, space):
        """Names sort; values keep the caller's order; the last name varies
        fastest. The order is part of the contract because it fixes the
        tie-break, so it is pinned exactly."""
        assert space.names == ("a", "b")
        assert space.grid() == [
            {"a": 0.1, "b": 1}, {"a": 0.1, "b": 2},
            {"a": 0.5, "b": 1}, {"a": 0.5, "b": 2},
            {"a": 0.9, "b": 1}, {"a": 0.9, "b": 2},
        ]

    def test_grid_order_ignores_dict_insertion_order(self):
        a = SearchSpace({"z": [1, 2], "a": ["x"]})
        b = SearchSpace({"a": ["x"], "z": [1, 2]})
        assert a.grid() == b.grid()

    def test_values_are_frozen_into_tuples(self):
        values = [1, 2, 3]
        sp = SearchSpace({"k": values})
        values.append(4)
        assert sp.params["k"] == (1, 2, 3)
        assert sp.size() == 3

    def test_sample_is_deterministic_for_a_seed(self, space):
        import random
        a = [space.sample(random.Random(3)) for _ in range(1)]
        b = [space.sample(random.Random(3)) for _ in range(1)]
        assert a == b

    def test_single_point_space_is_flagged(self):
        assert SearchSpace({"a": [1]}).is_single_point
        assert not SearchSpace({"a": [1, 2]}).is_single_point

    def test_describe_names_the_size(self, space):
        assert "6 configurations" in space.describe()


class TestSearchSpaceErrors:
    def test_empty_space_is_explained(self):
        with pytest.raises(ValueError) as e:
            SearchSpace({})
        msg = str(e.value)
        assert "empty" in msg
        assert "at least one parameter" in msg
        assert "n_neighbors" in msg          # shows the caller what to write

    def test_parameter_with_no_values_is_explained(self):
        with pytest.raises(ValueError) as e:
            SearchSpace({"n_neighbors": []})
        msg = str(e.value)
        assert "'n_neighbors'" in msg
        assert "empty value list" in msg
        assert "at least one value" in msg

    def test_bare_value_instead_of_a_list_is_explained(self):
        with pytest.raises(ValueError) as e:
            SearchSpace({"n_neighbors": 15})
        msg = str(e.value)
        assert "must be a list or tuple" in msg
        assert "int" in msg
        assert "[15]" in msg                 # tells them the fix

    def test_string_value_is_not_mistaken_for_a_sequence(self):
        with pytest.raises(ValueError) as e:
            SearchSpace({"metric": "euclidean"})
        assert "list or tuple" in str(e.value)

    def test_non_mapping_is_explained(self):
        with pytest.raises(ValueError) as e:
            SearchSpace([("a", [1])])
        assert "must be a mapping" in str(e.value)

    def test_blank_parameter_name_is_explained(self):
        with pytest.raises(ValueError) as e:
            SearchSpace({"  ": [1]})
        assert "non-empty strings" in str(e.value)


# ---------------------------------------------------------------------------
# grid_search
# ---------------------------------------------------------------------------

class TestGridSearch:
    def test_runs_every_configuration_in_grid_order(self, space):
        seen = []
        grid_search(lambda p: seen.append(dict(p)) or 1.0, space)
        assert seen == space.grid()

    def test_best_is_the_top_scorer(self, space):
        r = grid_search(linear_fit, space, metric="acc")
        assert r.ok
        assert r.best.params == {"a": 0.9, "b": 2}
        assert r.metric == "acc"
        assert len(r.trials) == 6
        assert r.n_failed == 0
        assert not r.partial

    def test_lower_is_better_flips_the_winner(self, space):
        r = grid_search(linear_fit, space, metric="loss",
                        higher_is_better=False)
        assert r.best.params == {"a": 0.1, "b": 1}

    def test_ties_break_on_the_earlier_trial(self):
        sp = SearchSpace({"a": [1, 2, 3]})
        r = grid_search(lambda p: 0.5, sp)
        assert r.best.params == {"a": 1}
        assert r.best.index == 0
        assert [t.index for t in r.ranked()] == [0, 1, 2]

    def test_progress_callback_fires_per_trial(self, space):
        events = []
        grid_search(linear_fit, space,
                    on_trial=lambda t, done, total: events.append((done, total)))
        assert events == [(i, 6) for i in range(1, 7)]

    def test_extra_metrics_and_duration_are_recorded(self, space):
        r = grid_search(lambda p: (1.0, {"n_val": 7}), space)
        assert all(t.extra_metrics["n_val"] == 7 for t in r.trials)
        assert all(t.duration >= 0.0 for t in r.trials)

    def test_dict_return_is_accepted(self, space):
        r = grid_search(lambda p: {"score": 2.0, "aux": "x"}, space)
        assert r.best.score == 2.0
        assert r.best.extra_metrics == {"aux": "x"}

    def test_dict_without_score_key_fails_the_trial_not_the_sweep(self, space):
        r = grid_search(lambda p: {"aux": 1}, space)
        assert r.n_failed == 6
        assert r.best is None
        assert "without a 'score' key" in r.failed[0].error

    def test_single_point_space_says_it_is_not_a_comparison(self):
        r = grid_search(lambda p: 1.0, SearchSpace({"a": [1]}))
        joined = " ".join(r.notes)
        assert "single configuration" in joined
        assert "nothing to compare" in joined

    def test_notes_describe_the_space(self, space):
        notes = grid_search(linear_fit, space).notes
        assert notes[0].startswith("Grid search over")
        assert "6 configurations" in notes[0]


class TestFailedTrials:
    """A bad configuration loses its own trial, never the sweep."""

    def test_raising_trial_is_recorded_and_the_sweep_continues(self, space):
        def fit(p):
            if p["a"] == 0.5:
                raise RuntimeError("singular matrix")
            return p["a"]

        r = grid_search(fit, space)
        assert len(r.trials) == 6                 # every config was attempted
        assert r.n_failed == 2
        assert len(r.successful) == 4
        assert r.best.params["a"] == 0.9          # the sweep still has a winner
        failed = sorted(r.failed, key=lambda t: t.index)
        assert [t.params["a"] for t in failed] == [0.5, 0.5]
        assert all("RuntimeError: singular matrix" in t.error for t in failed)

    def test_failure_count_is_reported_in_the_notes(self, space):
        def fit(p):
            if p["a"] == 0.5:
                raise RuntimeError("boom")
            return p["a"]

        r = grid_search(fit, space)
        assert any("2 of 6 evaluated configurations failed" in n
                   for n in r.notes)
        assert any("the sweep continued" in n for n in r.notes)

    def test_all_trials_failing_reports_no_winner(self, space):
        r = grid_search(lambda p: (_ for _ in ()).throw(ValueError("nope")),
                        space)
        assert r.best is None
        assert not r.ok
        assert r.n_failed == 6
        assert any("No configuration produced a score" in n for n in r.notes)

    def test_nan_score_is_a_failure_not_a_ranking(self, space):
        r = grid_search(lambda p: float("nan") if p["a"] == 0.5 else p["a"],
                        space)
        assert r.n_failed == 2
        assert all("non-finite" in t.error for t in r.failed)
        assert all(math.isfinite(t.score) for t in r.successful)

    def test_none_score_is_a_failure(self, space):
        r = grid_search(lambda p: None, space)
        assert r.n_failed == 6
        assert "returned no score" in r.failed[0].error

    def test_non_numeric_score_is_a_failure(self, space):
        r = grid_search(lambda p: "excellent", space)
        assert r.n_failed == 6
        assert "not a number" in r.failed[0].error

    def test_a_space_with_an_invalid_value_explains_which_one(self):
        """The space is structurally fine but one value is nonsense for the
        model. That trial records the model's own complaint verbatim."""
        sp = SearchSpace({"n_estimators": [10, -5, 20]})

        def fit(p):
            if p["n_estimators"] <= 0:
                raise ValueError("n_estimators must be positive")
            return float(p["n_estimators"])

        r = grid_search(fit, sp)
        bad = [t for t in r.failed]
        assert len(bad) == 1
        assert bad[0].params == {"n_estimators": -5}
        assert "n_estimators must be positive" in bad[0].error
        assert r.best.params == {"n_estimators": 20}


class TestEarlyStop:
    def test_stopping_early_returns_the_finished_trials_marked_partial(self, space):
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 3          # allow three trials through

        r = grid_search(linear_fit, space, should_stop=stop)
        assert r.partial is True
        assert len(r.trials) == 3
        assert [t.params for t in r.trials] == space.grid()[:3]
        assert any("stopped early after 3 of 6" in n for n in r.notes)
        assert any("not a completed sweep" in n for n in r.notes)

    def test_a_partial_sweep_still_reports_a_best_so_far(self, space):
        r = grid_search(linear_fit, space, should_stop=lambda: False)
        assert not r.partial
        r2 = grid_search(linear_fit, space,
                         should_stop=lambda c={"n": 0}: c.__setitem__(
                             "n", c["n"] + 1) or c["n"] > 2)
        assert r2.partial
        assert r2.best is not None

    def test_stopping_before_the_first_trial_yields_nothing(self, space):
        r = grid_search(linear_fit, space, should_stop=lambda: True)
        assert r.partial
        assert r.trials == []
        assert r.best is None

    def test_format_search_marks_a_partial_sweep_in_the_header(self, space):
        r = grid_search(linear_fit, space,
                        should_stop=lambda c={"n": 0}: c.__setitem__(
                            "n", c["n"] + 1) or c["n"] > 2)
        assert "[PARTIAL" in format_search(r)
        assert "not a completed sweep" in format_search(r)


# ---------------------------------------------------------------------------
# random_search
# ---------------------------------------------------------------------------

class TestRandomSearch:
    def test_reproducible_under_a_seed(self):
        sp = SearchSpace({"a": list(range(20)), "b": list(range(20))})
        one = random_search(lambda p: p["a"], sp, 8, seed=42)
        two = random_search(lambda p: p["a"], sp, 8, seed=42)
        assert [t.params for t in one.trials] == [t.params for t in two.trials]
        assert one.best.params == two.best.params

    def test_a_different_seed_gives_a_different_sample(self):
        sp = SearchSpace({"a": list(range(50)), "b": list(range(50))})
        one = random_search(lambda p: p["a"], sp, 8, seed=1)
        two = random_search(lambda p: p["a"], sp, 8, seed=2)
        assert [t.params for t in one.trials] != [t.params for t in two.trials]

    def test_respects_n_trials(self):
        sp = SearchSpace({"a": list(range(20)), "b": list(range(20))})
        r = random_search(lambda p: 1.0, sp, 5, seed=0)
        assert len(r.trials) == 5

    def test_no_duplicate_configurations_by_default(self):
        sp = SearchSpace({"a": [1, 2, 3], "b": [1, 2, 3]})
        r = random_search(lambda p: 1.0, sp, 9, seed=0)
        keys = [tuple(sorted(t.params.items())) for t in r.trials]
        assert len(set(keys)) == len(keys) == 9

    def test_allow_duplicates_keeps_the_requested_count(self):
        sp = SearchSpace({"a": [1, 2]})
        r = random_search(lambda p: 1.0, sp, 6, seed=0, allow_duplicates=True)
        assert len(r.trials) == 6

    def test_a_space_smaller_than_n_trials_shrinks_and_says_so(self):
        sp = SearchSpace({"a": [1, 2, 3]})
        r = random_search(lambda p: float(p["a"]), sp, 10, seed=0)
        assert len(r.trials) == 3
        assert any("exhaustive grid, not a random sample" in n for n in r.notes)

    def test_notes_record_the_seed(self):
        sp = SearchSpace({"a": list(range(10))})
        r = random_search(lambda p: 1.0, sp, 3, seed=17)
        assert any("seed 17" in n for n in r.notes)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_n_trials_is_explained(self, bad):
        sp = SearchSpace({"a": [1, 2]})
        with pytest.raises(ValueError) as e:
            random_search(lambda p: 1.0, sp, bad)
        assert "at least 1" in str(e.value)

    def test_non_integer_n_trials_is_explained(self):
        sp = SearchSpace({"a": [1, 2]})
        with pytest.raises(ValueError) as e:
            random_search(lambda p: 1.0, sp, "many")
        assert "positive integer" in str(e.value)

    def test_early_stop_works_for_random_search_too(self):
        sp = SearchSpace({"a": list(range(20))})
        r = random_search(lambda p: 1.0, sp, 10, seed=0,
                          should_stop=lambda c={"n": 0}: c.__setitem__(
                              "n", c["n"] + 1) or c["n"] > 4)
        assert r.partial
        assert len(r.trials) == 4


# ---------------------------------------------------------------------------
# Spread + the within-noise flag
# ---------------------------------------------------------------------------

class TestSpreadReporting:
    def test_score_stats_report_the_spread(self, space):
        r = grid_search(linear_fit, space)
        stats = r.score_stats()
        assert stats["n"] == 6
        assert stats["best"] == pytest.approx(0.902)
        assert stats["worst"] == pytest.approx(0.101)
        assert stats["spread"] == pytest.approx(0.801)
        assert stats["std"] > 0

    def test_stats_are_empty_when_nothing_succeeded(self, space):
        r = grid_search(lambda p: (_ for _ in ()).throw(ValueError("x")), space)
        stats = r.score_stats()
        assert stats == {"n": 0, "best": None, "worst": None, "mean": None,
                         "std": None, "spread": None}

    def test_notes_quote_the_spread(self, space):
        r = grid_search(linear_fit, space)
        assert any("Scores across 6 successful trials span" in n
                   for n in r.notes)

    def test_within_noise_fires_when_the_leaders_are_indistinguishable(self):
        """Three configurations whose means differ by 0.001 but whose folds
        wobble by 0.1. The winner is noise, and the result says so."""
        sp = SearchSpace({"a": [1, 2, 3]})
        folds = [(np.array([0, 1]), np.array([2, 3])),
                 (np.array([2, 3]), np.array([0, 1]))]

        def fit(params, tr, va):
            base = 0.70 + 0.001 * params["a"]
            # fold-to-fold wobble of +/- 0.1, two orders above the differences
            return base + (0.1 if va[0] == 0 else -0.1)

        r = cv_search(fit, sp, labels=np.array([0, 1, 0, 1]), folds=folds,
                      metric="accuracy")
        assert r.best.params == {"a": 3}
        noise, source = r.noise_level()
        assert noise == pytest.approx(0.1)
        assert "fold-to-fold" in source
        assert r.within_noise() is True
        assert len(r.trials_within_noise()) == 3
        assert any("WITHIN NOISE" in n for n in r.notes)
        assert "did not measurably matter" in " ".join(r.notes)
        assert "WITHIN NOISE" in format_search(r)

    def test_within_noise_stays_quiet_when_the_winner_is_real(self):
        sp = SearchSpace({"a": [1, 2, 3]})
        folds = [(np.array([0, 1]), np.array([2, 3])),
                 (np.array([2, 3]), np.array([0, 1]))]

        def fit(params, tr, va):
            # Large, consistent differences; folds agree with each other.
            return 0.2 * params["a"] + (0.001 if va[0] == 0 else -0.001)

        r = cv_search(fit, sp, labels=np.array([0, 1, 0, 1]), folds=folds)
        assert r.within_noise() is False
        assert not any("WITHIN NOISE" in n for n in r.notes)
        assert "WITHIN NOISE" not in format_search(r)

    def test_without_folds_the_yardstick_is_the_across_trial_spread(self, space):
        r = grid_search(linear_fit, space)
        noise, source = r.noise_level()
        assert source == "standard deviation across trials"
        assert noise == pytest.approx(r.score_stats()["std"])

    def test_a_single_trial_has_no_noise_estimate(self):
        r = grid_search(lambda p: 1.0, SearchSpace({"a": [1]}))
        noise, source = r.noise_level()
        assert noise is None
        assert "not enough" in source
        assert r.within_noise() is False
        # With no yardstick, only the winner is claimed to be near the winner.
        assert r.trials_within_noise() == [r.best]

    def test_no_successful_trials_means_no_noise_and_no_near_misses(self):
        r = grid_search(lambda p: (_ for _ in ()).throw(ValueError("x")),
                        SearchSpace({"a": [1, 2]}))
        assert r.noise_level() == (None, "not enough successful trials to "
                                         "estimate noise")
        assert r.trials_within_noise() == []
        assert r.within_noise() is False

    def test_an_unreadable_fold_std_falls_back_to_the_trial_spread(self):
        """A fit function that reports a non-numeric ``fold_std`` must not
        crash the noise estimate — it falls back to the across-trial spread."""
        r = grid_search(lambda p: (float(p["a"]), {"fold_std": "n/a"}),
                        SearchSpace({"a": [1.0, 2.0, 3.0]}))
        noise, source = r.noise_level()
        assert source == "standard deviation across trials"
        assert noise == pytest.approx(r.score_stats()["std"])


# ---------------------------------------------------------------------------
# Cross-validated search — the leak guard and grouped folds
# ---------------------------------------------------------------------------

@pytest.fixture
def wells():
    """60 crops across 12 wells of 5, alternating classes, plus filenames."""
    n = 60
    labels = np.array([i % 2 for i in range(n)])
    groups = np.array([f"plate1_{i // 5:02d}" for i in range(n)])
    filenames = [f"plate1_W{i // 5:02d}_f1_{i}.png" for i in range(n)]
    return labels, groups, filenames


class TestGroupedFolds:
    def test_grouped_folds_are_the_default(self, wells):
        """``group_by='well'`` is the signature default, and with filenames it
        actually groups — no well straddles a split."""
        labels, _groups, filenames = wells
        import inspect
        assert inspect.signature(build_folds).parameters[
            "group_by"].default == "well"
        assert inspect.signature(cv_search).parameters[
            "group_by"].default == "well"

        folds, warnings = build_folds(labels, 3, filenames=filenames, seed=0)
        assert warnings == []
        wells_of = np.array([f.rsplit("_", 2)[0] for f in filenames])
        for train_idx, val_idx in folds:
            assert set(wells_of[train_idx]).isdisjoint(set(wells_of[val_idx]))

    def test_explicit_groups_also_keep_wells_intact(self, wells):
        labels, groups, _ = wells
        folds, _ = build_folds(labels, 3, groups=groups, seed=0)
        for train_idx, val_idx in folds:
            assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))

    def test_every_sample_is_validated_exactly_once(self, wells):
        labels, groups, _ = wells
        folds, _ = build_folds(labels, 3, groups=groups, seed=0)
        val = np.concatenate([v for _, v in folds])
        assert sorted(val.tolist()) == list(range(len(labels)))

    def test_ungrouped_folds_carry_a_loud_warning(self, wells):
        labels, _groups, _ = wells
        _folds, warnings = build_folds(labels, 3, seed=0)
        joined = " ".join(warnings)
        assert "ungrouped" in joined
        assert "optimistic" in joined

    def test_group_by_none_says_scores_are_inflated(self, wells):
        labels, groups, _ = wells
        _folds, warnings = build_folds(labels, 3, groups=groups,
                                       group_by="none", seed=0)
        assert any("inflates scores" in w for w in warnings)

    def test_folds_are_reproducible_for_a_seed(self, wells):
        labels, groups, _ = wells
        a, _ = build_folds(labels, 3, groups=groups, seed=7)
        b, _ = build_folds(labels, 3, groups=groups, seed=7)
        for (t1, v1), (t2, v2) in zip(a, b):
            assert np.array_equal(t1, t2) and np.array_equal(v1, v2)

    def test_one_fold_is_refused(self, wells):
        labels, groups, _ = wells
        with pytest.raises(ValueError) as e:
            build_folds(labels, 1, groups=groups)
        assert "at least 2" in str(e.value)
        assert "no held-out data" in str(e.value)

    def test_excluding_everything_is_refused(self, wells):
        labels, groups, _ = wells
        with pytest.raises(ValueError) as e:
            build_folds(labels, 3, groups=groups,
                        exclude=list(range(len(labels))))
        assert "Every sample was excluded" in str(e.value)

    def test_filenames_too_short_for_the_level_are_counted(self):
        """A crop filename that cannot carry a well becomes its own group, and
        the caller is told how many did — silently degrading here would mean
        every crop is its own 'well' and the grouping does nothing."""
        labels = np.array([0, 1] * 15)
        names = [f"lone{i}.png" for i in range(30)]
        _folds, warnings = build_folds(labels, 3, filenames=names, seed=0)
        assert any("did not carry a 'well' level" in w for w in warnings)
        assert any("became their own group" in w for w in warnings)

    def test_out_of_range_test_indices_are_refused(self, wells):
        labels, groups, _ = wells
        with pytest.raises(ValueError) as e:
            build_folds(labels, 3, groups=groups, exclude=[999])
        assert "must lie inside" in str(e.value)


class TestCvSearchNeverScoresOnTest:
    """The leak, constructed on purpose, and asserted not to happen."""

    def test_test_indices_never_reach_the_fit_function(self, wells):
        labels, groups, _ = wells
        test_idx = np.arange(50, 60)          # the last two wells
        seen: set = set()

        def fit(params, train_idx, val_idx):
            seen.update(int(i) for i in train_idx)
            seen.update(int(i) for i in val_idx)
            return 0.5 + 0.01 * params["a"]

        r = cv_search(fit, SearchSpace({"a": [1, 2, 3]}), labels=labels,
                      groups=groups, n_folds=3, seed=0, test_idx=test_idx)
        assert r.ok
        assert seen.isdisjoint(set(test_idx.tolist()))
        assert seen == set(range(50))         # and everything else was used

    def test_folds_exclude_test_indices(self, wells):
        labels, groups, _ = wells
        test_idx = list(range(50, 60))
        folds, _ = build_folds(labels, 3, groups=groups, seed=0,
                               exclude=test_idx)
        for train_idx, val_idx in folds:
            assert set(train_idx.tolist()).isdisjoint(test_idx)
            assert set(val_idx.tolist()).isdisjoint(test_idx)

    def test_handing_it_leaky_folds_is_refused_loudly(self, wells):
        """Construct the leak explicitly: a caller passes folds whose
        validation set is the test split. The search refuses to run."""
        labels, _groups, _ = wells
        leaky = [(np.arange(0, 50), np.arange(50, 60))]
        with pytest.raises(ValueError) as e:
            cv_search(lambda p, tr, va: 1.0, SearchSpace({"a": [1]}),
                      labels=labels, folds=leaky, test_idx=np.arange(50, 60))
        msg = str(e.value)
        assert "held-out test indices" in msg
        assert "leaks" in msg

    def test_a_leak_in_the_training_side_is_caught_too(self, wells):
        labels, _groups, _ = wells
        leaky = [(np.arange(0, 60), np.arange(0, 10))]
        with pytest.raises(ValueError) as e:
            cv_search(lambda p, tr, va: 1.0, SearchSpace({"a": [1]}),
                      labels=labels, folds=leaky, test_idx=np.arange(55, 60))
        assert "held-out test indices" in str(e.value)

    def test_notes_state_the_test_split_was_excluded(self, wells):
        labels, groups, _ = wells
        r = cv_search(lambda p, tr, va: 1.0, SearchSpace({"a": [1, 2]}),
                      labels=labels, groups=groups, n_folds=3,
                      test_idx=np.arange(50, 60))
        joined = " ".join(r.notes)
        assert "10 test samples were excluded from every fold" in joined
        assert "no configuration was selected using test data" in joined
        assert "grouped" in joined

    def test_score_is_the_mean_over_folds_with_the_fold_spread(self, wells):
        labels, groups, _ = wells
        scores = iter([0.4, 0.6, 0.5])

        def fit(params, tr, va):
            return next(scores)

        r = cv_search(fit, SearchSpace({"a": [1]}), labels=labels,
                      groups=groups, n_folds=3, seed=0)
        assert r.best.score == pytest.approx(0.5)
        assert r.best.extra_metrics["fold_scores"] == [0.4, 0.6, 0.5]
        assert r.best.extra_metrics["n_folds"] == 3
        assert r.best.extra_metrics["fold_std"] > 0

    def test_a_fold_returning_no_score_fails_that_trial_only(self, wells):
        labels, groups, _ = wells

        def fit(params, tr, va):
            return None if params["a"] == 2 else 0.5

        r = cv_search(fit, SearchSpace({"a": [1, 2, 3]}), labels=labels,
                      groups=groups, n_folds=3)
        assert r.n_failed == 1
        assert r.failed[0].params == {"a": 2}
        assert len(r.successful) == 2

    def test_random_mode_samples_configurations(self, wells):
        labels, groups, _ = wells
        r = cv_search(lambda p, tr, va: float(p["a"]),
                      SearchSpace({"a": list(range(20))}), labels=labels,
                      groups=groups, n_folds=3, n_trials=4, seed=3)
        assert len(r.trials) == 4
        assert any("Random search: 4 of 20" in n for n in r.notes)

    def test_early_stop_marks_a_cv_sweep_partial(self, wells):
        labels, groups, _ = wells
        r = cv_search(lambda p, tr, va: 1.0, SearchSpace({"a": [1, 2, 3, 4]}),
                      labels=labels, groups=groups, n_folds=3,
                      should_stop=lambda c={"n": 0}: c.__setitem__(
                          "n", c["n"] + 1) or c["n"] > 2)
        assert r.partial
        assert len(r.trials) == 2


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

class TestLocalDirectionSearch:
    def test_first_round_is_the_requested_two_by_two_matrix(self):
        seen = []

        def fit(params):
            seen.append(dict(params))
            return params["n_neighbors"] + params["min_dist"]

        result = local_direction_search(
            fit, {"n_neighbors": 5, "min_dist": 0.1}, n_trials=1)

        assert seen == [
            {"n_neighbors": 4, "min_dist": 0.05},
            {"n_neighbors": 4, "min_dist": 0.15},
            {"n_neighbors": 6, "min_dist": 0.05},
            {"n_neighbors": 6, "min_dist": 0.15},
        ]
        assert result.best.params == {
            "n_neighbors": 6, "min_dist": 0.15}

    def test_moves_to_best_corner_and_stops_when_score_no_longer_improves(self):
        def fit(params):
            n = params["n_neighbors"]
            d = params["min_dist"]
            return -((n - 6) ** 2) - ((d - 0.15) ** 2)

        result = local_direction_search(
            fit, {"n_neighbors": 5, "min_dist": 0.1}, n_trials=20)

        # Round two is centred on the first round's winner, (6, 0.15).
        second_round = [trial.params for trial in result.trials[4:8]]
        assert second_round == [
            {"n_neighbors": 5, "min_dist": 0.1},
            {"n_neighbors": 5, "min_dist": 0.2},
            {"n_neighbors": 7, "min_dist": 0.1},
            {"n_neighbors": 7, "min_dist": 0.2},
        ]
        assert len(result.trials) == 8
        assert result.best.params == {
            "n_neighbors": 6, "min_dist": 0.15}
        assert any("stopping threshold" in note for note in result.notes)

    def test_clamps_boundaries_and_never_repeats_a_configuration(self):
        result = local_direction_search(
            lambda params: -params["min_dist"],
            {"n_neighbors": 2, "min_dist": 0.0}, n_trials=12)
        keys = [
            (trial.params["n_neighbors"], trial.params["min_dist"])
            for trial in result.trials
        ]
        assert len(keys) == len(set(keys))
        assert all(n >= 2 and 0.0 <= d <= 1.0 for n, d in keys)

    def test_requires_both_single_start_coordinates(self):
        with pytest.raises(ValueError, match="min_dist"):
            local_direction_search(
                lambda params: 1.0, {"n_neighbors": 5})

    def test_blank_round_limit_defaults_to_100_but_convergence_stops_early(self):
        result = local_direction_search(
            lambda params: -abs(params["n_neighbors"] - 6),
            {"n_neighbors": 5, "min_dist": 0.1}, n_trials=None)
        assert len(result.trials) < 400
        assert any("stopping threshold" in note for note in result.notes)

    def test_minimum_improvement_can_stop_small_score_gains(self):
        result = local_direction_search(
            lambda params: params["n_neighbors"] * 0.001,
            {"n_neighbors": 5, "min_dist": 0.1},
            n_trials=20, min_improvement=0.01)
        assert len(result.trials) == 8

class TestUmapCriteria:
    def test_small_dataset_bounds_and_deduplicates_n_neighbors(self):
        X = np.arange(30, dtype=float).reshape(6, 5)
        seen = []

        def embed(features, params):
            seen.append(params["n_neighbors"])
            return features[:, :2]

        result = umap_search(
            X, SearchSpace({"n_neighbors": [5, 15, 50, 100]}),
            embed_fn=embed)

        assert seen == [5]
        assert result.space.params["n_neighbors"] == (5,)
        assert result.trials[0].params["n_neighbors"] == 5
        assert any("limited to 2…5" in note for note in result.notes)
        assert any("only once" in note for note in result.notes)

    def test_small_dataset_bounds_umaps_implicit_default(self):
        X = np.arange(30, dtype=float).reshape(6, 5)
        seen = []

        result = umap_search(
            X, SearchSpace({"min_dist": [0.1]}),
            embed_fn=lambda features, params: (
                seen.append(params["n_neighbors"]) or features[:, :2]))

        assert result.ok
        assert seen == [5]
        assert any("default n_neighbors was limited" in note
                   for note in result.notes)

    def test_adaptive_search_never_exceeds_the_dataset_limit(self):
        X = np.arange(30, dtype=float).reshape(6, 5)
        seen = []

        result = umap_search(
            X,
            SearchSpace({"n_neighbors": [1000], "min_dist": [0.1]}),
            adaptive=True, n_trials=2,
            embed_fn=lambda features, params: (
                seen.append(params["n_neighbors"]) or features[:, :2]))

        assert result.ok
        assert seen
        assert max(seen) == 5
        assert min(seen) >= 2

    def test_the_criterion_is_named_in_the_result_and_the_report(
            self, tear_and_merge):
        X, _y, tear, merge = tear_and_merge
        table = {0.0: tear, 0.5: merge}
        r = umap_search(X, SearchSpace({"min_dist": [0.0, 0.5]}),
                        metric="trustworthiness", neighbourhood_k=5,
                        embed_fn=lambda f, p: table[p["min_dist"]])
        assert r.metric == "trustworthiness"
        text = format_search(r)
        assert "criterion: trustworthiness" in text
        assert UMAP_CRITERIA["trustworthiness"][:40] in text
        assert "no ground truth" in text
        assert "not as a verdict" in text
        assert r.best.extra_metrics["criterion"] == "trustworthiness"

    def test_two_criteria_pick_different_winners(self, tear_and_merge):
        """The honesty test.

        ``tear`` invents no neighbours but destroys real ones; ``merge`` does
        the opposite. Trustworthiness prefers the first, continuity the second.
        Because both are computed for every trial, the disagreement is visible
        in a single sweep rather than hidden behind whichever one the caller
        happened to pick. If this ever stops failing to agree, reporting one
        winner has become dishonest.
        """
        X, _y, tear, merge = tear_and_merge
        table = {0.0: tear, 0.5: merge}
        embed = lambda f, p: table[p["min_dist"]]        # noqa: E731
        space = SearchSpace({"min_dist": [0.0, 0.5]})

        by_trust = umap_search(X, space, metric="trustworthiness",
                               neighbourhood_k=5, embed_fn=embed)
        by_cont = umap_search(X, space, metric="continuity",
                              neighbourhood_k=5, embed_fn=embed)

        assert by_trust.best.params == {"min_dist": 0.0}
        assert by_cont.best.params == {"min_dist": 0.5}
        assert by_trust.best.params != by_cont.best.params

        # Both numbers came out of the same single sweep, so the caller can see
        # the disagreement without re-running anything.
        winner = by_trust.best.extra_metrics
        loser = [t for t in by_trust.trials
                 if t.params == {"min_dist": 0.5}][0].extra_metrics
        assert winner["trustworthiness"] > loser["trustworthiness"]
        assert winner["continuity"] < loser["continuity"]

    def test_every_criterion_is_computed_for_every_trial(self, tear_and_merge):
        X, y, tear, merge = tear_and_merge
        table = {0.0: tear, 0.5: merge}
        r = umap_search(X, SearchSpace({"min_dist": [0.0, 0.5]}),
                        metric="silhouette", labels=y, neighbourhood_k=5,
                        embed_fn=lambda f, p: table[p["min_dist"]])
        for t in r.successful:
            assert set(t.extra_metrics) >= {
                "trustworthiness", "continuity", "silhouette"}
        assert any("re-rank the table by a different one" in n for n in r.notes)

    def test_the_caveat_is_always_attached(self, tear_and_merge):
        X, _y, tear, merge = tear_and_merge
        r = umap_search(X, SearchSpace({"min_dist": [0.0]}),
                        neighbourhood_k=5, embed_fn=lambda f, p: tear)
        assert any("no ground truth" in n for n in r.notes)
        assert any("different criterion picks a different winner" in n
                   for n in r.notes)

    def test_embeddings_are_kept_so_the_user_can_look_at_them(
            self, tear_and_merge):
        X, _y, tear, merge = tear_and_merge
        table = {0.0: tear, 0.5: merge}
        r = umap_search(X, SearchSpace({"min_dist": [0.0, 0.5]}),
                        neighbourhood_k=5,
                        embed_fn=lambda f, p: table[p["min_dist"]])
        for t in r.successful:
            assert t.extra_metrics["embedding"].shape == (60, 2)

    def test_embeddings_can_be_dropped(self, tear_and_merge):
        X, _y, tear, _merge = tear_and_merge
        r = umap_search(X, SearchSpace({"min_dist": [0.0]}),
                        neighbourhood_k=5, keep_embeddings=False,
                        embed_fn=lambda f, p: tear)
        assert "embedding" not in r.best.extra_metrics

    def test_unknown_criterion_lists_the_real_ones(self, clustered_features):
        X, _ = clustered_features
        with pytest.raises(ValueError) as e:
            umap_search(X, SearchSpace({"min_dist": [0.1]}),
                        metric="vibes", embed_fn=lambda f, p: X[:, :2])
        msg = str(e.value)
        assert "Unknown UMAP criterion" in msg
        assert "trustworthiness" in msg
        assert "changes the answer" in msg

    def test_silhouette_without_labels_is_explained(self, clustered_features):
        X, _ = clustered_features
        with pytest.raises(ValueError) as e:
            umap_search(X, SearchSpace({"min_dist": [0.1]}),
                        metric="silhouette", embed_fn=lambda f, p: X[:, :2])
        assert "needs `labels=`" in str(e.value)

    def test_a_failing_embedder_fails_only_its_own_trial(self, tear_and_merge):
        X, _y, tear, _merge = tear_and_merge

        def embed(f, p):
            if p["min_dist"] == 0.5:
                raise RuntimeError("n_neighbors larger than the dataset")
            return tear

        r = umap_search(X, SearchSpace({"min_dist": [0.0, 0.5]}),
                        neighbourhood_k=5, embed_fn=embed)
        assert r.n_failed == 1
        assert "n_neighbors larger than the dataset" in r.failed[0].error
        assert r.best is not None

    def test_silhouette_with_a_single_class_fails_the_trial_clearly(
            self, tear_and_merge):
        """Labels exist but hold one class, so silhouette cannot be computed.
        That is a failed trial with an explanation, not a crash."""
        X, _y, tear, _merge = tear_and_merge
        one_class = np.zeros(X.shape[0], dtype=int)
        r = umap_search(X, SearchSpace({"min_dist": [0.0]}),
                        metric="silhouette", labels=one_class,
                        neighbourhood_k=5, embed_fn=lambda f, p: tear)
        assert r.n_failed == 1
        assert "'silhouette' could not be computed" in r.failed[0].error
        assert "trustworthiness" in r.failed[0].error

    def test_criteria_dictionary_states_what_each_ignores(self):
        assert "says nothing about" in UMAP_CRITERIA["trustworthiness"]
        assert "says nothing about" in UMAP_CRITERIA["continuity"]
        assert "needs labels" in UMAP_CRITERIA["silhouette"]


class TestUmapMissing:
    def test_umap_absent_degrades_with_a_message_not_an_importerror(
            self, monkeypatch, clustered_features):
        """``None`` in ``sys.modules`` makes ``import umap`` raise ImportError.
        The search must catch that and hand back a readable result."""
        X, _ = clustered_features
        monkeypatch.setitem(sys.modules, "umap", None)

        available, message = umap_available()
        assert available is False
        assert message == UMAP_MISSING_MESSAGE

        r = umap_search(X, SearchSpace({"n_neighbors": [5, 15]}))
        assert isinstance(r, SearchResult)
        assert r.trials == []
        assert r.best is None
        assert r.ok is False
        assert r.notes[0] == UMAP_MISSING_MESSAGE
        assert "pip install umap-learn" in r.notes[0]
        assert "Traceback" not in r.notes[0]

    def test_the_missing_message_renders_in_the_report(
            self, monkeypatch, clustered_features):
        X, _ = clustered_features
        monkeypatch.setitem(sys.modules, "umap", None)
        text = format_search(umap_search(X, SearchSpace({"n_neighbors": [5]})))
        assert "umap-learn" in text
        assert "No trials were run." in text

    def test_an_injected_embedder_works_without_umap(
            self, monkeypatch, tear_and_merge):
        """Missing umap must not disable the code path itself — only the
        default reducer depends on it."""
        X, _y, tear, _merge = tear_and_merge
        monkeypatch.setitem(sys.modules, "umap", None)
        r = umap_search(X, SearchSpace({"min_dist": [0.0]}),
                        neighbourhood_k=5, embed_fn=lambda f, p: tear)
        assert r.ok

    def test_umap_available_is_true_when_it_imports(self):
        available, message = umap_available()
        if not available:
            pytest.skip(message)
        assert umap_available() == (True, "")


def test_lazy_module_resets_after_a_failed_import(monkeypatch):
    """A failed optional import must not poison the next successful attempt."""
    import types
    from spacr.utils import _LazyModule

    name = "_spacr_test_optional_dependency"
    proxy = _LazyModule(name)
    with pytest.raises(ModuleNotFoundError):
        proxy.answer
    assert proxy.__dict__["_module"] is None

    module = types.ModuleType(name)
    module.answer = 42
    monkeypatch.setitem(sys.modules, name, module)
    assert proxy.answer == 42

    proxy.reset()
    assert proxy.__dict__["_module"] is None


def test_lazy_module_reports_an_incompatible_installed_version(monkeypatch):
    """A stale environment should get an upgrade command, not a fit traceback."""
    import spacr.utils as U

    proxy = U._LazyModule(
        "umap.umap_",
        minimum_distribution=(
            "umap-learn", "0.5.11",
            "Older releases call scikit-learn's removed API.",
        ),
    )
    monkeypatch.setattr(U, "_distribution_version", lambda _name: "0.5.6")

    with pytest.raises(U.OptionalDependencyCompatibilityError) as exc:
        proxy.UMAP
    message = str(exc.value)
    assert "umap-learn 0.5.6" in message
    assert "0.5.11" in message
    assert "pip install --upgrade" in message
    assert proxy.__dict__["_module"] is None


# ---------------------------------------------------------------------------
# format_search
# ---------------------------------------------------------------------------

class TestFormatSearch:
    def test_report_leads_with_the_criterion(self, space):
        text = format_search(grid_search(linear_fit, space, metric="acc"))
        assert text.splitlines()[0] == "Hyperparameter search — criterion: acc"

    def test_report_lists_failures_with_their_errors(self, space):
        def fit(p):
            if p["a"] == 0.5:
                raise RuntimeError("boom")
            return p["a"]

        text = format_search(grid_search(fit, space))
        assert "Failed trials (2)" in text
        assert "RuntimeError: boom" in text

    def test_report_shows_the_spread_next_to_the_winner(self, space):
        text = format_search(grid_search(linear_fit, space))
        assert "Best:" in text
        assert "Spread over 6 successful trials" in text
        assert "Noise yardstick" in text

    def test_report_truncates_long_tables(self):
        sp = SearchSpace({"a": list(range(30))})
        text = format_search(grid_search(lambda p: float(p["a"]), sp),
                             max_rows=5)
        assert "… 25 more" in text

    def test_all_failed_report_says_so(self, space):
        text = format_search(
            grid_search(lambda p: (_ for _ in ()).throw(ValueError("x")),
                        space))
        assert "All 6 trials failed" in text

    def test_empty_result_says_nothing_ran(self):
        r = SearchResult(space=SearchSpace({"a": [1]}), metric="acc")
        assert "No trials were run." in format_search(r)


# ---------------------------------------------------------------------------
# Trial / SearchResult odds and ends
# ---------------------------------------------------------------------------

class TestResultObjects:
    def test_trial_ok_needs_both_a_score_and_no_error(self):
        assert Trial(params={}, score=1.0).ok
        assert not Trial(params={}, score=None).ok
        assert not Trial(params={}, score=1.0, error="x").ok

    def test_trial_label_is_sorted(self):
        assert Trial(params={"b": 2, "a": 1}).label() == "a=1, b=2"

    def test_as_rows_ranks_then_lists_failures(self, space):
        def fit(p):
            if p["a"] == 0.5:
                raise RuntimeError("boom")
            return p["a"]

        rows = grid_search(fit, space).as_rows()
        assert [r["rank"] for r in rows] == [1, 2, 3, 4, None, None]
        assert rows[0]["score"] >= rows[1]["score"]
        assert rows[-1]["error"].startswith("RuntimeError")

    def test_ranked_excludes_failures(self, space):
        r = grid_search(lambda p: None if p["a"] == 0.5 else p["a"], space)
        assert len(r.ranked()) == 4


# ---------------------------------------------------------------------------
# App backends
# ---------------------------------------------------------------------------

class TestSklearnBackend:
    def test_builds_the_named_estimator(self):
        model = build_sklearn_model("random_forest", {"n_estimators": 7})
        assert type(model).__name__ == "RandomForestClassifier"
        assert model.n_estimators == 7

    @pytest.mark.parametrize("model_type,expected", [
        ("random_forest", "RandomForestClassifier"),
        ("extra_trees", "ExtraTreesClassifier"),
        ("logistic_regression", "LogisticRegression"),
        ("gradient_boosting", "HistGradientBoostingClassifier"),
        ("svm", "SVC"),
        ("mlp", "MLPClassifier"),
    ])
    def test_every_always_available_backend_builds(self, model_type, expected):
        model = build_sklearn_model(
            model_type, {"n_estimators": 12, "learning_rate": 0.05,
                         "reg_lambda": 2.0, "reg_alpha": 0.5}, n_jobs=1)
        assert type(model).__name__ == expected

    @pytest.mark.parametrize("model_type,module,expected", [
        ("xgboost", "xgboost", "XGBClassifier"),
        ("lightgbm", "lightgbm", "LGBMClassifier"),
        ("catboost", "catboost", "CatBoostClassifier"),
    ])
    def test_optional_backends_build_or_say_how_to_install(
            self, model_type, module, expected):
        params = {"n_estimators": 8, "learning_rate": 0.05}
        try:
            model = build_sklearn_model(model_type, params, n_jobs=1)
        except ImportError as exc:
            assert f"pip install {module}" in str(exc)
            assert f"model_type_ml='{model_type}'" in str(exc)
        else:
            assert type(model).__name__ == expected

    @pytest.mark.parametrize("model_type,module", [
        ("xgboost", "xgboost"), ("lightgbm", "lightgbm"),
        ("catboost", "catboost")])
    def test_a_missing_optional_backend_names_the_package(
            self, monkeypatch, model_type, module):
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(ImportError) as e:
            build_sklearn_model(model_type, {"n_estimators": 8})
        assert f"pip install {module}" in str(e.value)

    def test_unknown_model_type_lists_the_supported_ones(self):
        with pytest.raises(ValueError) as e:
            build_sklearn_model("crystal_ball", {})
        assert "Unsupported model_type_ml" in str(e.value)
        assert "random_forest" in str(e.value)

    @pytest.mark.parametrize("criterion", ["accuracy", "roc_auc", "f1"])
    def test_every_criterion_scores_the_validation_rows(self, criterion):
        rng = np.random.default_rng(4)
        X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(4, 1, (30, 4))])
        y = np.repeat([0, 1], 30)
        fit = sklearn_cv_fit_fn(X, y, model_type="logistic_regression",
                                criterion=criterion, n_jobs=1)
        score, _extra = fit({}, np.arange(0, 50), np.arange(25, 35))
        assert 0.0 <= score <= 1.0

    def test_roc_auc_works_for_an_estimator_without_predict_proba(self):
        rng = np.random.default_rng(5)
        X = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(5, 1, (20, 3))])
        y = np.repeat([0, 1], 20)

        class _NoProba:
            """Decision-function-only estimator, like an uncalibrated SVM."""

            def fit(self, X, y):
                self._m = X[y == 1].mean()
                return self

            def predict(self, X):
                return (X.mean(axis=1) > self._m / 2).astype(int)

            def decision_function(self, X):
                return X.mean(axis=1)

        import spacr.hyperparam as hp
        fit = hp.sklearn_cv_fit_fn(X, y, model_type="svm",
                                   criterion="roc_auc", n_jobs=1)
        import unittest.mock as mock
        train = np.concatenate([np.arange(0, 15), np.arange(25, 40)])
        val = np.arange(15, 25)              # 5 of each class, so AUC exists
        with mock.patch.object(hp, "build_sklearn_model",
                               return_value=_NoProba()):
            score, extra = fit({}, train, val)
        assert 0.0 <= score <= 1.0
        assert extra == {"n_train": 30, "n_val": 10}

    def test_cv_fit_fn_fits_on_train_and_scores_on_val_only(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(4, 1, (30, 4))])
        y = np.repeat([0, 1], 30)
        fit = sklearn_cv_fit_fn(X, y, model_type="logistic_regression",
                                n_jobs=1)
        train_idx = np.arange(0, 50)
        val_idx = np.arange(50, 60)
        score, extra = fit({"n_estimators": 10}, train_idx, val_idx)
        assert 0.0 <= score <= 1.0
        assert extra == {"n_train": 50, "n_val": 10}

    def test_end_to_end_grouped_cv_search_over_a_real_estimator(self):
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(3, 1, (30, 4))])
        y = np.repeat([0, 1], 30)
        groups = np.array([f"w{i // 5}" for i in range(60)])
        fit = sklearn_cv_fit_fn(X, y, model_type="logistic_regression",
                                n_jobs=1)
        r = cv_search(fit, SearchSpace({"reg_lambda": [0.1, 1.0]}),
                      labels=y, groups=groups, n_folds=3, seed=0)
        assert r.ok
        assert all(0.0 <= t.score <= 1.0 for t in r.successful)
        assert r.best.extra_metrics["n_folds"] == 3


class TestClassifyBackend:
    """The deep path, exercised without training a single CNN."""

    def _fake_trainer(self, calls, scores):
        def trainer(cfg):
            calls.append(dict(cfg))
            return f"fold_{cfg['learning_rate']}.csv"
        return trainer

    def test_forces_cross_validation_and_keeps_the_group_level(self):
        import pandas as pd
        calls = []
        fit = classify_cv_fit_fn(
            {"src": "/x", "cv_group_by": "field", "epochs": 2},
            n_folds=4,
            train_fn=self._fake_trainer(calls, None),
            read_fold_csv=lambda p: pd.DataFrame(
                {"accuracy": [0.7, 0.8, 0.75, 0.72]}))
        score, extra = fit({"learning_rate": 0.001})
        assert calls[0]["cross_validation_folds"] == 4
        assert calls[0]["cv_group_by"] == "field"
        assert calls[0]["learning_rate"] == 0.001
        assert calls[0]["epochs"] == 2
        assert score == pytest.approx(0.7425)
        assert extra["n_folds"] == 4
        assert extra["fold_std"] > 0

    def test_a_dead_run_fails_that_trial_with_an_explanation(self):
        fit = classify_cv_fit_fn({"src": "/x"}, train_fn=lambda cfg: None)
        r = grid_search(fit, SearchSpace({"learning_rate": [0.1]}),
                        metric="accuracy")
        assert r.n_failed == 1
        assert "no per-fold results" in r.failed[0].error

    def test_a_missing_metric_column_is_explained(self):
        import pandas as pd
        fit = classify_cv_fit_fn(
            {"src": "/x"}, criterion="prauc",
            train_fn=lambda cfg: "x.csv",
            read_fold_csv=lambda p: pd.DataFrame({"accuracy": [0.5, 0.6]}))
        r = grid_search(fit, SearchSpace({"learning_rate": [0.1]}))
        assert "no 'prauc' column" in r.failed[0].error

    def test_fewer_than_two_folds_is_refused(self):
        with pytest.raises(ValueError) as e:
            classify_cv_fit_fn({"src": "/x"}, n_folds=1)
        assert "at least 2 folds" in str(e.value)

    def test_all_folds_reporting_nothing_usable_is_explained(self):
        import pandas as pd
        fit = classify_cv_fit_fn(
            {"src": "/x"}, n_folds=2, train_fn=lambda cfg: "x.csv",
            read_fold_csv=lambda p: pd.DataFrame(
                {"accuracy": [float("nan"), float("nan")]}))
        r = grid_search(fit, SearchSpace({"learning_rate": [0.1]}))
        assert "no fold reported a usable 'accuracy' value" in r.failed[0].error

    def test_single_fold_std_is_not_nan(self):
        import pandas as pd
        fit = classify_cv_fit_fn(
            {"src": "/x"}, n_folds=2, train_fn=lambda cfg: "x.csv",
            read_fold_csv=lambda p: pd.DataFrame({"accuracy": [0.5]}))
        _score, extra = fit({"learning_rate": 0.1})
        assert extra["fold_std"] == 0.0


class TestRunSearchForApp:
    def test_unknown_app_is_explained(self):
        with pytest.raises(ValueError) as e:
            run_search_for_app("sequencing", {}, SearchSpace({"a": [1]}))
        assert "No hyperparameter search is defined" in str(e.value)
        assert "umap" in str(e.value)

    def test_unknown_mode_is_explained(self):
        with pytest.raises(ValueError) as e:
            run_search_for_app("umap", {}, SearchSpace({"a": [1]}),
                               mode="genetic")
        assert "must be 'grid' or 'random'" in str(e.value)

    def test_criterion_must_belong_to_the_app(self):
        with pytest.raises(ValueError) as e:
            run_search_for_app("umap", {}, SearchSpace({"a": [1]}),
                               criterion="accuracy")
        assert "not available for 'umap'" in str(e.value)

    def test_classify_routes_through_the_deep_cv_backend(self, monkeypatch):
        """No CNN is trained: the fit-function factory is swapped for one that
        records the configurations it was asked to evaluate."""
        import spacr.hyperparam as hp
        calls = []
        monkeypatch.setattr(hp, "classify_cv_fit_fn",
                            lambda settings, **kw: _fixed_fit(calls))
        r = run_search_for_app(
            "classify", {"src": "/x", "cv_group_by": "well"},
            SearchSpace({"epochs": [1, 2]}), criterion="accuracy", n_folds=5)
        assert len(r.trials) == 2
        assert r.ok
        assert calls == [{"epochs": 1}, {"epochs": 2}]
        joined = " ".join(r.notes)
        assert "grouped folds" in joined
        assert "test split is never scored on" in joined
        assert "2 configurations × 5 folds" in joined

    def test_classify_random_mode_samples(self, monkeypatch):
        import spacr.hyperparam as hp
        calls = []
        monkeypatch.setattr(hp, "classify_cv_fit_fn",
                            lambda settings, **kw: _fixed_fit(calls))
        r = run_search_for_app(
            "classify", {"src": "/x"}, SearchSpace({"epochs": list(range(10))}),
            mode="random", n_trials=3, seed=5)
        assert len(r.trials) == 3

    def test_umap_path_uses_supplied_data(self, tear_and_merge, monkeypatch):
        import spacr.hyperparam as hp
        X, _y, tear, merge = tear_and_merge
        table = {0.0: tear, 0.5: merge}
        monkeypatch.setattr(hp, "umap_available", lambda: (True, ""))
        monkeypatch.setattr(
            hp, "_default_umap_embed",
            lambda feats, params, seed: table[params["min_dist"]])
        data = hp.SearchData(features=X, notes=["synthetic"])
        r = run_search_for_app(
            "umap", {"src": "/x"}, SearchSpace({"min_dist": [0.0, 0.5]}),
            criterion="continuity", data=data)
        assert r.notes[0] == "synthetic"
        assert r.ok

    def test_ml_analyze_path_runs_grouped_cv_over_a_real_estimator(self):
        """The Classify (ML) route, end to end, on injected data — grouped
        folds, no test split touched, real sklearn fits (60 rows, 4 columns)."""
        import spacr.hyperparam as hp
        rng = np.random.default_rng(2)
        X = np.vstack([rng.normal(0, 1, (30, 4)), rng.normal(3, 1, (30, 4))])
        y = np.repeat([0, 1], 30)
        groups = np.array([f"w{i // 5}" for i in range(60)])
        data = hp.SearchData(features=X, labels=y, groups=groups,
                             notes=["injected"])
        r = run_search_for_app(
            "ml_analyze", {"model_type_ml": "logistic_regression", "n_jobs": 1},
            SearchSpace({"reg_lambda": [0.1, 1.0]}), criterion="accuracy",
            n_folds=3, data=data)
        assert r.ok
        assert r.notes[0] == "injected"
        assert r.metric == "accuracy"
        assert all(t.extra_metrics["n_folds"] == 3 for t in r.successful)
        assert any("grouped" in n for n in r.notes)

    def test_ml_analyze_honours_random_mode(self):
        import spacr.hyperparam as hp
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 3))
        y = np.array([0, 1] * 20)
        groups = np.array([f"w{i // 4}" for i in range(40)])
        data = hp.SearchData(features=X, labels=y, groups=groups)
        r = run_search_for_app(
            "ml_analyze", {"model_type_ml": "logistic_regression", "n_jobs": 1},
            SearchSpace({"reg_lambda": [0.01, 0.1, 1.0, 10.0]}),
            mode="random", n_trials=2, n_folds=2, data=data)
        assert len(r.trials) == 2

    def test_loss_criterion_is_minimised_not_maximised(self, monkeypatch):
        """'loss' is the one criterion where smaller wins; the router must not
        hand it to a maximiser. No CNN is trained — the fit factory is swapped."""
        import spacr.hyperparam as hp
        monkeypatch.setattr(
            hp, "classify_cv_fit_fn",
            lambda settings, **kw: (lambda params: 1.0 / params["epochs"]))
        r = run_search_for_app(
            "classify", {"src": "/x"}, SearchSpace({"epochs": [1, 2, 4]}),
            criterion="loss")
        assert r.higher_is_better is False
        assert r.best.params == {"epochs": 4}        # smallest 1/epochs

    def test_defaults_and_criteria_tables_line_up(self):
        assert set(DEFAULT_SPACES) == set(APP_CRITERIA)
        for app, params in DEFAULT_SPACES.items():
            SearchSpace(params)              # every default space is valid
            assert APP_CRITERIA[app]


def _fixed_fit(calls):
    """A stand-in deep fit function that records calls and scores cheaply."""
    def fit(params):
        calls.append(dict(params))
        return 0.5 + 0.01 * params["epochs"], {"fold_std": 0.001, "n_folds": 5}
    return fit


@pytest.fixture
def measurements_src(tmp_path):
    """A tiny real ``measurements/measurements.db`` — 40 objects, two control
    columns, four numeric features. Small enough to read in milliseconds."""
    import sqlite3
    import pandas as pd

    (tmp_path / "measurements").mkdir()
    db = tmp_path / "measurements" / "measurements.db"
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{i % 4}" for i in range(n)],
        "columnID": ["c1"] * 20 + ["c2"] * 20,
        "fieldID": ["f1"] * n,
        "object_label": np.arange(1, n + 1),
        "cell_channel_0_mean_intensity": rng.uniform(1, 100, n),
        "cell_channel_0_area": rng.uniform(1, 100, n),
        "cell_channel_1_mean_intensity": rng.uniform(1, 100, n),
    })
    con = sqlite3.connect(db)
    try:
        frame.to_sql("cell", con, index=False)
    finally:
        con.close()
    return str(tmp_path)


class TestLoadSearchDataFromDb:
    def _settings(self, src, **kw):
        base = {"src": src, "tables": ["cell"], "filter_by": None}
        base.update(kw)
        return base

    def test_umap_gets_features_and_no_labels(self, measurements_src):
        data = load_search_data("umap", self._settings(measurements_src))
        # Provenance identifiers such as object_label are deliberately kept
        # out of the model matrix.
        assert data.features.shape == (40, 3)
        assert data.labels is None
        assert data.frame is not None

    def test_umap_csv_none_filter_uses_all_measurements(
            self, measurements_src):
        data = load_search_data(
            "umap", self._settings(measurements_src, filter_by="None"))
        assert data.features.shape == (40, 3)

    def test_umap_invalid_filter_reports_available_channels(
            self, measurements_src):
        with pytest.raises(ValueError) as error:
            load_search_data(
                "umap",
                self._settings(measurements_src, filter_by="channel_9"),
            )
        message = str(error.value)
        assert "filter_by='channel_9' matched no measurement features" in message
        assert "channel_0" in message
        assert "channel_1" in message

    def test_ml_gets_labels_from_the_control_columns(self, measurements_src):
        data = load_search_data(
            "ml_analyze",
            self._settings(measurements_src, positive_control="c2",
                           negative_control="c1", location_column="columnID"))
        assert data.features.shape == (40, 3)
        assert sorted(np.bincount(data.labels).tolist()) == [20, 20]
        assert data.groups is not None
        assert data.groups[0] == "p1_r0_c1"
        joined = " ".join(data.notes)
        assert "Labels derived from controls in 'columnID'" in joined
        assert "recognise those columns' plate position" in joined

    def test_row_limit_subsamples_and_warns(self, measurements_src):
        data = load_search_data(
            "umap", self._settings(measurements_src, row_limit=10))
        assert data.features.shape[0] == 10
        assert any("Sub-sampled to 10" in n for n in data.notes)
        assert any("rank configurations differently" in n for n in data.notes)

    def test_umap_row_exclusions_match_the_real_run(self, measurements_src):
        data = load_search_data(
            "umap",
            self._settings(
                measurements_src,
                exclude_rows={"columnID": ["c1"]},
            ),
        )
        assert data.features.shape[0] == 20
        assert set(data.frame["columnID"]) == {"c2"}
        assert any("columnID" in note and "20 row(s)" in note
                   for note in data.notes)

    def test_an_annotation_column_wins_over_the_controls(self,
                                                        measurements_src):
        import sqlite3
        import pandas as pd
        db = f"{measurements_src}/measurements/measurements.db"
        con = sqlite3.connect(db)
        try:
            frame = pd.read_sql("SELECT * FROM cell", con)
            frame["manual"] = [0, 1] * 20
            frame.to_sql("cell", con, index=False, if_exists="replace")
        finally:
            con.close()
        data = load_search_data(
            "ml_analyze",
            self._settings(measurements_src, annotation_column="manual"))
        assert sorted(np.bincount(data.labels).tolist()) == [20, 20]
        assert any("'manual' annotation column" in n for n in data.notes)

    def test_one_class_only_is_refused(self, measurements_src):
        with pytest.raises(ValueError) as e:
            load_search_data(
                "ml_analyze",
                self._settings(measurements_src, positive_control="zzz",
                               negative_control="c1",
                               location_column="columnID"))
        assert "Only one class survived" in str(e.value)

    def test_no_label_source_is_explained(self, measurements_src):
        with pytest.raises(ValueError) as e:
            load_search_data(
                "ml_analyze",
                self._settings(measurements_src, location_column="nope"))
        assert "Cannot build labels" in str(e.value)

    def test_a_table_without_well_columns_warns_about_ungrouped_folds(
            self, tmp_path):
        """Labels can still be built, but the folds will be ungrouped — and the
        notes say so rather than quietly reporting an inflated score."""
        import sqlite3
        import pandas as pd
        (tmp_path / "measurements").mkdir()
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "object_label": np.arange(20),
            "manual": [0, 1] * 10,
            "cell_area": rng.uniform(1, 100, 20),
            "cell_channel_0_mean_intensity": rng.uniform(1, 100, 20),
        })
        con = sqlite3.connect(tmp_path / "measurements" / "measurements.db")
        try:
            frame.to_sql("cell", con, index=False)
        finally:
            con.close()
        data = load_search_data("ml_analyze", {
            "src": str(tmp_path), "tables": ["cell"], "filter_by": None,
            "annotation_column": "manual"})
        assert data.groups is None
        joined = " ".join(data.notes)
        assert "no plate/row/column columns" in joined
        assert "optimistic" in joined

    def test_an_empty_database_is_explained(self, tmp_path):
        import sqlite3
        import pandas as pd
        (tmp_path / "measurements").mkdir()
        con = sqlite3.connect(tmp_path / "measurements" / "measurements.db")
        try:
            pd.DataFrame({"plateID": [], "object_label": []}).to_sql(
                "cell", con, index=False)
        finally:
            con.close()
        with pytest.raises(ValueError) as e:
            load_search_data("umap", {"src": str(tmp_path), "tables": ["cell"],
                                      "filter_by": None})
        assert "Run Measure first" in str(e.value)


class TestLoadSearchData:
    def test_missing_src_is_explained(self):
        for src in (None, "", "path", "/path/to/src"):
            with pytest.raises(ValueError) as e:
                load_search_data("umap", {"src": src})
            assert "No source folder is set" in str(e.value)

    def test_well_groups_are_built_from_plate_row_column(self):
        import pandas as pd
        from spacr.hyperparam import _well_groups
        frame = pd.DataFrame({
            "plateID": ["p1", "p1", "p1", "p2"],
            "rowID": ["A", "A", "B", "A"],
            "columnID": ["01", "01", "02", "01"],
        })
        groups, warning = _well_groups(frame)
        assert warning is None
        assert list(groups) == ["p1_A_01", "p1_A_01", "p1_B_02", "p2_A_01"]

    def test_missing_well_columns_warn_about_ungrouped_folds(self):
        import pandas as pd
        from spacr.hyperparam import _well_groups
        groups, warning = _well_groups(pd.DataFrame({"cell_area": [1, 2]}))
        assert groups is None
        assert "no plate/row/column columns" in warning
        assert "optimistic" in warning
