"""The search strategies of ``spacr.hyperparam`` and the result they return.

One table-driven test per unit, each row run as a subtest so a failing case
names itself: what a fit function may return (``_normalise_outcome``), the
grid / random runners, the ``Trial`` / ``SearchResult`` slices (Pareto front,
noise yardstick, table rows), the plain-text report, and the N-dimensional
Walk (``WalkAxis``, ``walk_search`` and its two-axis UMAP form
``local_direction_search``), including resuming a Walk from a checkpoint.
Every fit function here is a deterministic stand-in, so expected values are
exact.
"""
from __future__ import annotations

import json
import math

import pytest

from spacr import hyperparam as hp
from spacr.cancellation import PipelineCancelled
from spacr.hyperparam import SearchResult, SearchSpace, Trial, WalkAxis

_FAILED_NOTE = ("{n} of {total} evaluated configurations failed and were "
                "recorded rather than dropped; the sweep continued. See the "
                "per-trial error column.")
_NO_WINNER = ("No configuration produced a score, so there is no winner to "
              "report.")
_SINGLE = ("Only one configuration was scored, so there is nothing to "
           "compare it against — this is a single measurement, not a search.")


def _sometimes(params):
    """Score ``a`` itself, except that ``a == 2`` diverges."""
    if params["a"] == 2:
        raise RuntimeError("diverged")
    return float(params["a"])


# ===========================================================================
# _normalise_outcome
# ===========================================================================

def test_normalise_outcome_reads_every_documented_return_shape(subtests):
    """A number, a ``(score, metrics)`` pair or a dict with ``score`` become
    ``(score, extra)``; ``None`` is no score; a dict without ``score``, a
    non-number and a non-finite score are refused by type and message."""
    for cid, value, expected in [
        ("bare-number", 0.5, (0.5, {})),
        ("numeric-string", "0.5", (0.5, {})),
        ("integer", 3, (3.0, {})),
        ("dict-with-score", {"score": 0.25, "auc": 0.9}, (0.25, {"auc": 0.9})),
        ("pair", (0.7, {"n_train": 3}), (0.7, {"n_train": 3})),
        ("pair-with-none", (None, {"why": "diverged"}),
         (None, {"why": "diverged"})),
        ("dict-with-none-score", {"score": None, "x": 1}, (None, {"x": 1})),
        ("bare-none", None, (None, {})),
        ("tuple-not-a-pair", (0.1, 0.2, 0.3), "TypeError"),
    ]:
        with subtests.test(case=cid):
            if expected == "TypeError":
                with pytest.raises(TypeError, match="which is not a number"):
                    hp._normalise_outcome(value)
                continue
            assert hp._normalise_outcome(value) == expected

    for cid, value, exc, message in [
        ("dict-without-score", {"auc": 0.9}, TypeError,
         "fit function returned a dict without a 'score' key; return either a "
         "number, a (score, metrics) pair, or a dict containing 'score'."),
        ("word", "excellent", TypeError,
         "fit function returned 'excellent', which is not a number: could not "
         "convert string to float: 'excellent'"),
        ("nan", float("nan"), ValueError,
         "fit function returned a non-finite score (nan); a trial that cannot "
         "be scored is recorded as failed rather than ranked."),
        ("inf", float("inf"), ValueError,
         "fit function returned a non-finite score (inf); a trial that cannot "
         "be scored is recorded as failed rather than ranked."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(exc) as info:
                hp._normalise_outcome(value)
            assert str(info.value) == message


# ===========================================================================
# grid_search (and the shared runner behind it)
# ===========================================================================

def test_grid_search_records_progress_failures_stops_and_notes(subtests):
    """Every configuration is one trial and one progress call, failures
    included; a cancel propagates, a stop marks the result partial, and the
    notes say when the sweep had one configuration or no winner."""
    with subtests.test(case="progress-for-failed-trials-too"):
        seen = []
        result = hp.grid_search(_sometimes, SearchSpace({"a": [1, 2, 3]}),
                                on_trial=lambda t, done, total: seen.append(
                                    (done, total, t.ok)))
        assert seen == [(1, 3, True), (2, 3, False), (3, 3, True)]
        assert [t.error for t in result.trials] == [
            None, "RuntimeError: diverged", None]
        assert [t.index for t in result.trials] == [0, 1, 2]
        assert result.best.params == {"a": 3}
        assert result.notes == [
            "Grid search over a=[1, 2, 3]  (3 configurations)",
            _FAILED_NOTE.format(n=1, total=3),
            "Scores across 2 successful trials span 2 (1 to 3), standard "
            "deviation 1.",
        ]

    with subtests.test(case="cancel-is-not-a-failed-trial"):
        def _cancelled(params):
            raise PipelineCancelled("stopped")

        with pytest.raises(PipelineCancelled):
            hp.grid_search(_cancelled, SearchSpace({"a": [1, 2]}))

    with subtests.test(case="single-configuration-is-a-measurement"):
        result = hp.grid_search(lambda p: 0.5, SearchSpace({"a": [1]}))
        assert result.notes == [
            "Grid search over a=[1]  (1 configurations)",
            "The space contains a single configuration, so this evaluates one "
            "setting rather than comparing alternatives.",
            _SINGLE,
        ]

    with subtests.test(case="nothing-scored-has-no-winner"):
        def _fails(params):
            raise RuntimeError("no")

        result = hp.grid_search(_fails, SearchSpace({"a": [1, 2]}),
                                notes=["caller note"])
        assert result.best is None and result.ok is False
        assert result.notes == [
            "Grid search over a=[1, 2]  (2 configurations)", "caller note",
            _FAILED_NOTE.format(n=2, total=2), _NO_WINNER]

    with subtests.test(case="none-score-is-a-recorded-failure"):
        result = hp.grid_search(lambda p: None, SearchSpace({"a": [1]}))
        assert result.trials[0].error == (
            "fit function returned no score for this configuration")

    with subtests.test(case="should-stop-truncates-and-marks-partial"):
        calls = []

        def _stop():
            calls.append(1)
            return len(calls) > 2

        result = hp.grid_search(lambda p: float(p["a"]),
                                SearchSpace({"a": [1, 2, 3, 4]}),
                                should_stop=_stop)
        assert result.partial is True
        assert [t.params["a"] for t in result.trials] == [1, 2]
        assert result.notes[1] == (
            "Search stopped early after 2 of 4 configurations. The trials "
            "below are the ones that finished; the rest were never evaluated, "
            "so this is not a completed sweep.")

    with subtests.test(case="lower-is-better-picks-the-smallest"):
        result = hp.grid_search(lambda p: float(p["a"]),
                                SearchSpace({"a": [3, 1, 2]}),
                                higher_is_better=False, metric="loss")
        assert result.best.params == {"a": 1}
        assert result.metric == "loss" and result.higher_is_better is False

    with subtests.test(case="tie-goes-to-the-earlier-trial"):
        result = hp.grid_search(lambda p: 0.5, SearchSpace({"a": [7, 8]}))
        assert result.best.params == {"a": 7}


# ===========================================================================
# random_search
# ===========================================================================

def test_random_search_samples_reproducibly_without_repeats(subtests):
    """The same seed gives the same configurations; distinct draws by
    default, the whole space (with a note) when more are asked for than
    exist, duplicates only on request, and a grid fill-in when the sampler
    keeps drawing one point; a non-positive or non-numeric count is
    refused."""
    with subtests.test(case="reproducible-from-seed"):
        space = SearchSpace({"a": [1, 2, 3, 4], "b": [10, 20, 30]})
        first = hp.random_search(lambda p: p["a"] * p["b"], space, 4, seed=7)
        second = hp.random_search(lambda p: p["a"] * p["b"], space, 4, seed=7)
        expected = [{"a": 3, "b": 10}, {"a": 4, "b": 30}, {"a": 1, "b": 10},
                    {"a": 1, "b": 20}]
        assert [t.params for t in first.trials] == expected
        assert [t.params for t in second.trials] == expected
        assert first.notes[0] == (
            "Random search: 4 of 12 configurations sampled with seed 7 from "
            "a=[1, 2, 3, 4] × b=[10, 20, 30]  (12 configurations)")

    with subtests.test(case="never-the-same-configuration-twice"):
        result = hp.random_search(lambda p: float(p["a"]),
                                  SearchSpace({"a": [1, 2, 3]}), 3, seed=1)
        assert [t.params["a"] for t in result.trials] == [1, 3, 2]

    with subtests.test(case="more-than-exist-is-an-exhaustive-grid"):
        result = hp.random_search(lambda p: float(p["a"]),
                                  SearchSpace({"a": [1, 2]}), 10, seed=0)
        assert sorted(t.params["a"] for t in result.trials) == [1, 2]
        assert result.notes[:2] == [
            "Random search: 2 of 2 configurations sampled with seed 0 from "
            "a=[1, 2]  (2 configurations)",
            "The space holds only 2 distinct configurations, fewer than the 10 "
            "requested, so every one of them was evaluated — this is an "
            "exhaustive grid, not a random sample.",
        ]

    with subtests.test(case="duplicates-on-request"):
        result = hp.random_search(lambda p: float(p["a"]),
                                  SearchSpace({"a": [1, 2]}), 6, seed=0,
                                  allow_duplicates=True)
        assert [t.params["a"] for t in result.trials] == [2, 2, 1, 2, 2, 2]

    with subtests.test(case="numeric-string-count"):
        result = hp.random_search(lambda p: 1.0, SearchSpace({"a": [1, 2]}),
                                  "1")
        assert len(result.trials) == 1

    for cid, n_trials, message in [
        ("zero", 0, "n_trials must be at least 1, got 0. A search with no "
                    "trials has nothing to report."),
        ("negative", -3, "n_trials must be at least 1, got -3. A search with "
                         "no trials has nothing to report."),
        ("word", "many", "n_trials must be a positive integer, got 'many'."),
        ("none", None, "n_trials must be a positive integer, got None."),
    ]:
        with subtests.test(case=f"refused-{cid}"):
            with pytest.raises(ValueError) as info:
                hp.random_search(lambda p: 1.0, SearchSpace({"a": [1]}),
                                 n_trials)
            assert str(info.value) == message

    for cid, values, expected in [
        ("three-of-three", [1, 2, 3], [1, 2, 3]),
        ("three-of-four", [5, 8, 11, 14], [5, 8, 11]),
    ]:
        with subtests.test(case=f"stuck-sampler-falls-back-{cid}"):
            first = values[0]
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(SearchSpace, "sample",
                           lambda self, rng, a=first: {"a": a})
                result = hp.random_search(lambda p: float(p["a"]),
                                          SearchSpace({"a": values}), 3,
                                          seed=0)
            assert [t.params["a"] for t in result.trials] == expected


# ===========================================================================
# Trial / SearchResult
# ===========================================================================

def _t(a, score, index, error=None, **extra):
    return Trial(params={"a": a}, score=score, index=index, error=error,
                 extra_metrics=extra)


def test_search_result_ranks_fronts_and_measures_its_own_noise(subtests):
    """``label``/``ok`` per trial; the Pareto front drops dominated and
    unscorable trials and honours each objective's direction; the noise
    yardstick prefers a numeric fold spread, falls back to the across-trial
    spread, and is absent for one trial; table rows put successes first."""
    with subtests.test(case="label-and-ok"):
        trial = Trial(params={"b": "x", "a": 1}, score=0.5)
        assert trial.label() == "a=1, b='x'"
        assert trial.ok is True
        assert Trial(params={}, score=0.5, error="E").ok is False
        assert Trial(params={}).ok is False

    for cid, trials, objectives, higher, expected in [
        ("dominated-is-off-the-front",
         [_t(1, 0.9, 0, x=0.9, y=0.9), _t(2, 0.5, 1, x=0.2, y=0.2),
          _t(3, 0.4, 2, x=0.1, y=1.0)],
         {"x": True, "y": True}, True, [1, 3]),
        ("missing-or-nan-objective-is-off-the-front",
         [_t(1, 0.9, 0, x=0.9), _t(2, 0.8, 1, x=float("nan")),
          _t(3, 0.7, 2), _t(4, 0.6, 3, x="high")],
         {"x": True}, True, [1]),
        ("lower-is-better-objective",
         [_t(1, 0.1, 0, d=0.1), _t(2, 0.8, 1, d=0.8)],
         {"d": False}, False, [1]),
        ("equal-trials-both-stay",
         [_t(1, 0.5, 0, x=0.5), _t(2, 0.5, 1, x=0.5)],
         {"x": True}, True, [1, 2]),
        ("failed-trials-are-never-on-it",
         [_t(1, 0.5, 0, x=0.1), _t(2, None, 1, error="E", x=0.9)],
         {"x": True}, True, [1]),
        ("no-objectives-no-front",
         [_t(1, 0.9, 0, x=0.9)], {}, True, []),
    ]:
        with subtests.test(case=f"pareto-{cid}"):
            result = SearchResult(trials=trials, best=trials[0],
                                  higher_is_better=higher,
                                  objectives=objectives)
            assert [t.params["a"] for t in result.pareto_front()] == expected

    across = "standard deviation across trials"
    per_config = ("fold-to-fold standard deviation of the best "
                  "configuration")
    for cid, trials, best, noise, source in [
        ("fold-std-wins", [_t(1, 0.9, 0, fold_std=0.02), _t(2, 0.1, 1)], 0,
         0.02, per_config),
        ("fold-std-word-falls-back", [_t(1, 0.9, 0, fold_std="wide"),
                                      _t(2, 0.1, 1)], 0, 0.4, across),
        ("fold-std-nan-falls-back", [_t(1, 0.9, 0, fold_std=float("nan")),
                                     _t(2, 0.1, 1)], 0, 0.4, across),
        ("one-trial-has-none", [_t(1, 0.9, 0)], 0, None,
         "not enough successful trials to estimate noise"),
        ("no-best-across-trials", [_t(1, 0.9, 0, fold_std=0.02),
                                   _t(2, 0.1, 1)], None, 0.4, across),
    ]:
        with subtests.test(case=f"noise-{cid}"):
            result = SearchResult(
                trials=trials, best=None if best is None else trials[best])
            value, said = result.noise_level()
            assert said == source
            if noise is None:
                assert value is None
            else:
                assert value == pytest.approx(noise)

    with subtests.test(case="one-trial-is-its-own-tie-and-never-within-noise"):
        only = _t(1, 0.9, 0)
        result = SearchResult(trials=[only], best=only)
        assert result.trials_within_noise() == [only]
        assert result.within_noise() is False

    with subtests.test(case="within-noise-judges-the-leaders"):
        tied = [_t(1, 0.80, 0, fold_std=0.05), _t(2, 0.78, 1),
                _t(3, 0.70, 2)]
        result = SearchResult(trials=tied, best=tied[0])
        assert [t.params["a"] for t in result.trials_within_noise()] == [1, 2]
        assert result.within_noise(top_n=2) is True
        assert result.within_noise(top_n=3) is False

    for cid, higher, expected in [
        ("higher", True, {"n": 3, "best": 0.9, "worst": 0.1, "mean": 0.5,
                          "std": math.sqrt(0.32 / 3), "spread": 0.8}),
        ("lower", False, {"n": 3, "best": 0.1, "worst": 0.9, "mean": 0.5,
                          "std": math.sqrt(0.32 / 3), "spread": 0.8}),
    ]:
        with subtests.test(case=f"score-stats-{cid}"):
            trials = [_t(1, 0.9, 0), _t(2, 0.5, 1), _t(3, 0.1, 2),
                      _t(4, None, 3, error="E")]
            stats = SearchResult(trials=trials,
                                 higher_is_better=higher).score_stats()
            assert stats == pytest.approx(expected)

    with subtests.test(case="score-stats-empty"):
        assert SearchResult().score_stats() == {
            "n": 0, "best": None, "worst": None, "mean": None, "std": None,
            "spread": None}

    with subtests.test(case="rows-successes-first-then-failures"):
        rows = hp.grid_search(_sometimes, SearchSpace({"a": [1, 2, 3]}),
                              metric="accuracy").as_rows()
        assert [(r["rank"], r["index"], r["params"], r["score"], r["error"],
                 r["metric"], r["pareto"]) for r in rows] == [
            (1, 2, {"a": 3}, 3.0, None, "accuracy", False),
            (2, 0, {"a": 1}, 1.0, None, "accuracy", False),
            (None, 1, {"a": 2}, None, "RuntimeError: diverged", "accuracy",
             False),
        ]

    with subtests.test(case="rows-flag-the-front"):
        trials = [_t(1, 0.9, 0, x=0.9), _t(2, 0.5, 1, x=0.1)]
        rows = SearchResult(trials=trials, best=trials[0],
                            objectives={"x": True}).as_rows()
        assert [(r["params"]["a"], r["pareto"]) for r in rows] == [
            (1, True), (2, False)]


# ===========================================================================
# format_search
# ===========================================================================

def _ranked(scores, *, metric="score", fold_std=None, failed=(), notes=(),
            partial=False, objectives=None, extra=None):
    """A finished result with fixed durations, so the table is exact."""
    trials = [Trial(params={"depth": i + 1}, score=s, index=i, duration=0.5,
                    extra_metrics=dict({} if fold_std is None
                                       else {"fold_std": fold_std},
                                       **((extra or {}).get(i, {}))))
              for i, s in enumerate(scores)]
    for j, message in enumerate(failed):
        trials.append(Trial(params={"depth": 100 + j}, index=len(trials),
                            error=message))
    result = SearchResult(trials=trials, metric=metric, notes=list(notes),
                          partial=partial, objectives=dict(objectives or {}))
    hp._select_best(result)
    return result


def test_format_search_prints_caveats_table_and_noise_exactly(subtests):
    """The header names the criterion (and a partial sweep), criterion
    caveats and notes come before the table, failures are listed, long
    tables say what they hid, and the Pareto front, spread, noise yardstick
    and WITHIN-NOISE verdict follow the winner."""
    rule = "=" * len("Hyperparameter search — criterion: score")

    with subtests.test(case="no-trials"):
        assert hp.format_search(SearchResult(notes=["n1"])).splitlines() == [
            "Hyperparameter search — criterion: score", rule, "* n1", "",
            "No trials were run."]

    with subtests.test(case="partial-header"):
        header = hp.format_search(SearchResult(partial=True)).splitlines()[0]
        assert header == ("Hyperparameter search — criterion: score  "
                          "[PARTIAL — stopped early, not a completed sweep]")

    for cid, metric, caveat in [
        ("umap", "trustworthiness", hp.UMAP_CRITERIA["trustworthiness"]),
        ("activation", "insertion_auc",
         hp.ACTIVATION_CRITERIA["insertion_auc"]),
    ]:
        with subtests.test(case=f"criterion-caveat-{cid}"):
            lines = hp.format_search(SearchResult(metric=metric)).splitlines()
            assert lines[2:4] == [f"'{metric}' {caveat}", ""]

    with subtests.test(case="all-failed-lists-every-failure"):
        result = _ranked([], failed=["RuntimeError: a", "RuntimeError: b"])
        assert hp.format_search(result).splitlines()[3:] == [
            "All 2 trials failed:",
            "  [0] depth=100 -> RuntimeError: a",
            "  [1] depth=101 -> RuntimeError: b"]

    with subtests.test(case="ranked-table-spread-and-yardstick"):
        text = hp.format_search(_ranked([0.55, 0.6, 0.65, 0.7]))
        assert text.splitlines()[3:] == [
            "rank       score        sd     sec  params",
            "----  ----------  --------  ------  " + "-" * 30,
            "   1      0.7000         -    0.50  depth=4",
            "   2      0.6500         -    0.50  depth=3",
            "   3      0.6000         -    0.50  depth=2",
            "   4      0.5500         -    0.50  depth=1",
            "",
            "Best: depth=4  score=0.7000",
            "Spread over 4 successful trials: 0.5500 … 0.7000 (sd 0.0559)",
            "Noise yardstick: 0.0559 (standard deviation across trials)",
        ]

    with subtests.test(case="one-trial-prints-no-yardstick"):
        lines = hp.format_search(_ranked([0.4])).splitlines()
        assert lines[-2:] == ["Best: depth=1  score=0.4000",
                              "Spread over 1 successful trials: 0.4000 … "
                              "0.4000 (sd 0.0000)"]

    with subtests.test(case="failures-after-table-and-truncation"):
        result = _ranked([0.1, 0.2, 0.3, 0.4, 0.5],
                         failed=["RuntimeError: depth diverged"])
        lines = hp.format_search(result, max_rows=3).splitlines()
        assert lines[8:12] == ["      … 2 more", "", "Failed trials (1):",
                               "  [5] depth=100 -> RuntimeError: depth "
                               "diverged"]

    with subtests.test(case="within-noise-is-called-arbitrary"):
        lines = hp.format_search(
            _ranked([0.8, 0.81, 0.82], fold_std=0.05)).splitlines()
        assert lines[-2:] == [
            "Noise yardstick: 0.0500 (fold-to-fold standard deviation of the "
            "best configuration)",
            "WITHIN NOISE — 3 trials are indistinguishable from the best. "
            "Treat the winner as arbitrary."]
        assert "   1      0.8200    0.0500    0.50  depth=3" in lines

    with subtests.test(case="pareto-front-is-printed"):
        extra = {i: {"stability": 0.1 * (i + 1),
                     "cluster_structure": 0.9 - 0.1 * (i + 1)}
                 for i in range(3)}
        result = _ranked([0.5, 0.5, 0.5], extra=extra,
                         objectives={"stability": True,
                                     "cluster_structure": True})
        lines = hp.format_search(result).splitlines()
        start = lines.index("Best: depth=1  score=0.5000")
        assert lines[start + 1:start + 5] == [
            "Pareto front (3 non-dominated configuration(s)):",
            "  depth=1  [stability=0.1000, cluster_structure=0.8000]",
            "  depth=2  [stability=0.2000, cluster_structure=0.7000]",
            "  depth=3  [stability=0.3000, cluster_structure=0.6000]"]


# ===========================================================================
# WalkAxis
# ===========================================================================

def test_walk_axis_validates_and_offers_clamped_neighbours(subtests):
    """An axis refuses a blank name, a fractional/low resolution, one
    choice, no step, a non-numeric or non-positive step and an inverted
    range by message; it clamps and offers ±step (resolution 2), the centre
    too (resolution 3), rotated choices for a categorical axis, and fewer
    values at a boundary rather than duplicates."""
    for cid, kwargs, message in [
        ("blank-name", dict(name="   ", step=1.0),
         "A Walk axis needs a parameter name."),
        ("word-resolution", dict(name="n_neighbors", step=1.0,
                                 resolution="fine"),
         "Walk axis 'n_neighbors' needs a whole-number resolution."),
        ("resolution-one", dict(name="n", step=1.0, resolution=1),
         "Walk axis 'n' has resolution 1; an axis that contributes fewer than "
         "two values is not being searched at all and should be left out of "
         "the space."),
        ("one-choice", dict(name="init", choices=["pca"]),
         "Walk axis 'init' is categorical with 1 choice(s); it needs at least "
         "two."),
        ("no-step-no-choices", dict(name="d"),
         "Walk axis 'd' needs either a step (numeric) or a list of choices "
         "(categorical)."),
        ("word-step", dict(name="min_dist", step="a bit"),
         "Walk axis 'min_dist' needs a numeric step."),
        ("zero-step", dict(name="d", step=0),
         "Walk axis 'd' has step 0; a step of zero or less never leaves the "
         "centre."),
        ("inverted-range", dict(name="d", step=1, minimum=2, maximum=1),
         "Walk axis 'd' has minimum 2 above maximum 1."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                WalkAxis(**kwargs)
            assert str(info.value) == message

    n_axis = WalkAxis("n", step=5, minimum=2, maximum=40, integer=True)
    d_axis = WalkAxis("d", step=0.05, minimum=0.0, maximum=1.0)
    cat = WalkAxis("init", choices=["spectral", "random", "pca"])
    for cid, axis, centre, expected in [
        ("integer-pair", n_axis, 10, [5, 15]),
        ("integer-at-floor", n_axis, 3, [2, 8]),
        ("integer-at-ceiling", n_axis, 40, [35, 40]),
        ("integer-centre-clamped-first", n_axis, 100, [35, 40]),
        ("float-pair-rounded", d_axis, 0.1, [0.05, 0.15]),
        ("unbounded-integer", WalkAxis("n", step=2, integer=True), 3.4,
         [1, 5]),
        ("float-at-zero", d_axis, 0.0, [0.0, 0.05]),
        ("resolution-three-keeps-centre",
         WalkAxis("n", step=5, minimum=2, integer=True, resolution=3), 10,
         [5, 10, 15]),
        ("resolution-five-reaches-two-steps",
         WalkAxis("d", step=0.1, resolution=5), 1.0,
         [0.8, 0.9, 1.0, 1.1, 1.2]),
        ("pinned-axis-offers-one-value",
         WalkAxis("n", step=5, minimum=15, maximum=15, integer=True), 15,
         [15]),
        ("categorical-rotates-from-centre", cat, "random",
         ["random", "pca"]),
        ("categorical-unknown-starts-at-first", cat, "umap", ["spectral",
                                                              "random"]),
    ]:
        with subtests.test(case=f"values-{cid}"):
            assert axis.values_around(centre) == expected

    for cid, axis, value, expected in [
        ("integer-rounds", n_axis, 7.6, 8),
        ("integer-floor", n_axis, -3, 2),
        ("float-ceiling", d_axis, 1.7, 1.0),
        ("categorical-member", cat, "pca", "pca"),
        ("categorical-stranger", cat, "tsne", "spectral"),
    ]:
        with subtests.test(case=f"clamp-{cid}"):
            assert axis.clamp(value) == expected
            assert type(axis.clamp(value)) is type(expected)

    with subtests.test(case="numeric-strings-normalised"):
        axis = WalkAxis("d", step="0.5", minimum="0", maximum="2",
                        resolution="3")
        assert (axis.step, axis.minimum, axis.maximum, axis.resolution,
                axis.categorical) == (0.5, 0.0, 2.0, 3, False)
        assert cat.categorical is True and cat.choices == (
            "spectral", "random", "pca")


# ===========================================================================
# walk_search
# ===========================================================================

def _never_called(_params):
    raise AssertionError("validation must happen before the first fit")


def _axes():
    return [WalkAxis("n_neighbors", step=5.0, minimum=2.0, integer=True),
            WalkAxis("min_dist", step=0.05, minimum=0.0, maximum=1.0)]


def _store(tmp_path, *, resume=False, name="walk.json"):
    return hp._UmapCheckpoint(str(tmp_path / name), {"features": "matrix-a"},
                              resume, False)


def test_walk_search_refuses_unusable_arguments_before_fitting(subtests):
    """No axes, a repeated axis, a missing start value, a non-numeric round
    limit or threshold, zero rounds and a negative threshold are refused by
    message before any fit."""
    start = {"n_neighbors": 15, "min_dist": 0.1}
    one = [WalkAxis("n_neighbors", step=1.0)]
    for cid, axes, kwargs, begin, message in [
        ("no-axes", [], {}, start,
         "A Walk needs at least one axis; with none there is no direction to "
         "move in and the search would score the starting point for ever."),
        ("axis-twice", one + one, {}, start,
         "Walk axis 'n_neighbors' is listed twice; one parameter is one "
         "direction."),
        ("missing-start", _axes(), {}, {"n_neighbors": 15},
         "A Walk needs one starting value per searched parameter; missing "
         "['min_dist']."),
        ("word-round-limit", one, {"n_trials": "as many as it takes"}, start,
         "A Walk requires a numeric round limit and minimum improvement."),
        ("word-threshold", one, {"min_improvement": "some"}, start,
         "A Walk requires a numeric round limit and minimum improvement."),
        ("zero-rounds", one, {"n_trials": 0}, start,
         "A Walk needs a maximum of at least 1 round."),
        ("negative-threshold", one, {"min_improvement": -0.5}, start,
         "The minimum improvement must be zero or greater."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.walk_search(_never_called, begin, axes, **kwargs)
            assert str(info.value) == message


def test_walk_search_climbs_stops_and_resumes_from_its_checkpoint(
        subtests, tmp_path):
    """A walk scores the neighbourhood (never the start), moves to the round
    winner, records failures and missing scores per row, stops on no
    improvement, an all-failed round or an exhausted neighbourhood, persists
    its centre (with the two-axis legacy keys only for the UMAP pair), and on
    resume restores the stored or legacy centre and replays finished
    trials without refitting them."""
    with subtests.test(case="climbs-then-stops-on-no-improvement"):
        fitted = []

        def _fit(params):
            fitted.append((params["n_neighbors"], params["min_dist"]))
            return -abs(params["n_neighbors"] - 20) - params["min_dist"]

        result = hp.walk_search(_fit, {"n_neighbors": 10, "min_dist": 0.1,
                                       "metric": "cosine"}, _axes(),
                                n_trials=10,
                                checkpoint=_store(tmp_path / "climb"))
        assert fitted[:4] == [(5, 0.05), (5, 0.15), (15, 0.05), (15, 0.15)]
        assert fitted[4:8] == [(10, 0.0), (10, 0.1), (20, 0.0), (20, 0.1)]
        assert result.best.params == {"n_neighbors": 20, "min_dist": 0.0,
                                      "metric": "cosine"}
        assert result.best.score == 0.0
        assert len(result.trials) == 11
        assert result.notes[1] == (
            "The Walk stopped because the newest neighbourhood improved the "
            "best score by -5, not more than the 0 stopping threshold.")
        meta = json.loads((tmp_path / "climb" / "walk.json").read_text(
            encoding="utf-8"))["meta"]
        assert meta["centre"] == {"n_neighbors": 20, "min_dist": 0.0}
        assert (meta["centre_n"], meta["centre_d"]) == (20, 0.0)
        assert meta["best_score"] == 0.0

    with subtests.test(case="refused-configuration-is-a-failed-row"):
        def _fit(params):
            if params["n_neighbors"] > 10:
                raise RuntimeError("n_neighbors larger than the sample")
            return 0.5

        result = hp.walk_search(_fit, {"n_neighbors": 10, "min_dist": 0.1},
                                _axes(), n_trials=1)
        assert [(t.params["n_neighbors"], t.error) for t in result.trials] == [
            (5, None), (5, None),
            (15, "RuntimeError: n_neighbors larger than the sample"),
            (15, "RuntimeError: n_neighbors larger than the sample")]
        assert result.notes[1:3] == [
            "The Walk reached the maximum of 1 rounds.",
            _FAILED_NOTE.format(n=2, total=4)]

    with subtests.test(case="no-score-says-so-on-every-row"):
        result = hp.walk_search(lambda p: None,
                                {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                                n_trials=1)
        assert {t.error for t in result.trials} == {
            "fit function returned no score for this configuration"}
        assert len(result.trials) == 4 and result.best is None

    with subtests.test(case="all-failed-round-ends-the-walk"):
        def _fit(_params):
            raise RuntimeError("every fit fails here")

        result = hp.walk_search(_fit, {"n_neighbors": 10, "min_dist": 0.1},
                                _axes(), n_trials=5,
                                checkpoint=_store(tmp_path / "failed"))
        assert len(result.trials) == 4 and result.best is None
        assert result.notes[-1] == _NO_WINNER
        meta = json.loads((tmp_path / "failed" / "walk.json").read_text(
            encoding="utf-8"))["meta"]
        assert meta["rounds_completed"] == 1 and meta["best_score"] is None

    with subtests.test(case="pinned-axis-has-nowhere-to-go"):
        pinned = WalkAxis("n_neighbors", step=5.0, minimum=15.0, maximum=15.0,
                          integer=True)
        result = hp.walk_search(lambda p: 0.5, {"n_neighbors": 15}, [pinned],
                                n_trials=10)
        assert result.trials == [] and result.best is None
        assert result.notes[1] == (
            "Round 1 had no configuration left to try -- every neighbour of "
            "the current centre has already been scored, or the axes are all "
            "at a boundary.")

    with subtests.test(case="should-stop-marks-partial"):
        result = hp.walk_search(lambda p: 0.5,
                                {"n_neighbors": 10, "min_dist": 0.1}, _axes(),
                                n_trials=3, should_stop=lambda: True,
                                checkpoint=_store(tmp_path / "stop"))
        assert result.partial is True and result.trials == []
        assert result.notes[1] == (
            "Search stopped after 0 configurations (0 completed rounds; "
            "maximum 3 rounds).")
        stored = json.loads((tmp_path / "stop" / "walk.json").read_text(
            encoding="utf-8"))
        assert stored["status"] == "partial"

    with subtests.test(case="lower-is-better-walks-downhill"):
        result = hp.walk_search(lambda p: float(p["n_neighbors"]),
                                {"n_neighbors": 10},
                                [WalkAxis("n_neighbors", step=4, minimum=2,
                                          integer=True)],
                                n_trials=5, higher_is_better=False)
        assert [t.params["n_neighbors"] for t in result.trials] == [6, 14, 2,
                                                                  10]
        assert result.best.params == {"n_neighbors": 2}

    with subtests.test(case="one-axis-at-a-time-past-the-round-limit"):
        axes = [WalkAxis(name, step=1.0) for name in ("a", "b", "c")]
        result = hp.walk_search(lambda p: 0.5, {"a": 0, "b": 0, "c": 0}, axes,
                                n_trials=1, max_candidates_per_round=4)
        assert len(result.trials) == 6
        assert result.notes[1].startswith(
            "The full neighbourhood of these 3 axes exceeds the 4-fit round "
            "limit, so each round varies ONE axis at a time (6 fits per round "
            "instead of the product).")

    with subtests.test(case="other-axes-write-no-legacy-keys"):
        folder = tmp_path / "other"
        hp.walk_search(lambda p: 0.5, {"alpha": 2.0, "beta": 1.0},
                       [WalkAxis("alpha", step=1.0, minimum=0.0),
                        WalkAxis("beta", step=0.5, minimum=0.0)],
                       n_trials=2, checkpoint=_store(folder))
        meta = json.loads((folder / "walk.json").read_text(
            encoding="utf-8"))["meta"]
        assert meta["centre"] == {"alpha": 1.0, "beta": 0.5}
        assert "centre_n" not in meta and "centre_d" not in meta

    # -- resuming -----------------------------------------------------------
    for cid, meta, start, expected_n, expected_d in [
        ("legacy-centre-keys", {"rounds_completed": 1, "centre_n": 20,
                                "centre_d": 0.2, "best_score": 0.5},
         {"n_neighbors": 5, "min_dist": 0.05}, {15, 25}, {0.15, 0.25}),
        ("centre-naming-one-axis", {"rounds_completed": 1,
                                    "centre": {"n_neighbors": 30,
                                               "an_axis_we_dropped": 7},
                                    "best_score": 0.5},
         {"n_neighbors": 5, "min_dist": 0.5}, {25, 35}, {0.45, 0.55}),
        ("centre-not-a-mapping", {"rounds_completed": 1,
                                  "centre": "not a mapping",
                                  "best_score": 0.5},
         {"n_neighbors": 20, "min_dist": 0.5}, {15, 25}, {0.45, 0.55}),
        ("unreadable-legacy-centre", {"rounds_completed": 1,
                                      "centre_n": "twenty", "centre_d": 0.3,
                                      "best_score": 0.5},
         {"n_neighbors": 10, "min_dist": 0.1}, {5, 15}, {0.25, 0.35}),
    ]:
        with subtests.test(case=f"resume-{cid}"):
            folder = tmp_path / cid
            store = _store(folder)
            store.record(Trial(params={"n_neighbors": 99, "min_dist": 0.9},
                               score=0.5, index=0), round_index=0)
            store.finish(meta)
            seen = []
            result = hp.walk_search(
                lambda p, out=seen: out.append(dict(p)) or 0.4, start, _axes(),
                n_trials=2, checkpoint=_store(folder, resume=True))
            assert {row["n_neighbors"] for row in seen} == expected_n
            assert {round(row["min_dist"], 6) for row in seen} == expected_d
            assert result.notes[1] == (
                f"Resumed 1 completed trial(s) at Walk round 2 from "
                f"{folder / 'walk.json'}.")
            assert result.trials[0].params == {"n_neighbors": 99,
                                               "min_dist": 0.9}

    with subtests.test(case="resume-replays-finished-trials-in-order"):
        folder = tmp_path / "replay"
        store = _store(folder)
        for index, neighbours in enumerate((5, 10)):
            store.record(Trial(params={"n_neighbors": neighbours,
                                       "min_dist": 0.05},
                               score=0.1 * index, index=index), round_index=0)
        store.finish({"rounds_completed": 1,
                      "centre": {"n_neighbors": 10, "min_dist": 0.05},
                      "best_score": "not a number"})
        reported = []
        result = hp.walk_search(
            lambda p: 0.05, {"n_neighbors": 10, "min_dist": 0.05}, _axes(),
            n_trials=2,
            on_trial=lambda t, done, total: reported.append(
                (t.params["n_neighbors"], done, total)),
            checkpoint=_store(folder, resume=True))
        assert reported[:2] == [(5, 1, 8), (10, 2, 8)]
        assert [t.params["n_neighbors"] for t in result.trials[:2]] == [5, 10]
        assert len(result.trials) == 6

    with subtests.test(case="trial-scored-this-round-is-not-refitted"):
        folder = tmp_path / "midround"
        store = _store(folder)
        store.record(Trial(params={"n_neighbors": 5}, score=0.9, index=0),
                     round_index=1)
        store.finish({"rounds_completed": 1, "centre": {"n_neighbors": 10},
                      "best_score": 0.1})
        fitted, reported = [], []
        result = hp.walk_search(
            lambda p: fitted.append(dict(p)) or 0.2, {"n_neighbors": 10},
            [WalkAxis("n_neighbors", step=5.0, minimum=2.0, integer=True)],
            n_trials=2,
            on_trial=lambda t, done, total: reported.append(
                (t.params["n_neighbors"], done, total)),
            checkpoint=_store(folder, resume=True))
        assert fitted == [{"n_neighbors": 15}]
        assert [(t.params["n_neighbors"], t.score) for t in result.trials] == [
            (5, 0.9), (15, 0.2)]
        assert reported == [(5, 1, 4), (15, 2, 4)]


# ===========================================================================
# local_direction_search
# ===========================================================================

def test_local_direction_search_checks_its_two_axes_then_walks(subtests):
    """The two-axis UMAP walk refuses missing or non-numeric starts, steps,
    thresholds and round limits, a zero step, a negative threshold, a
    neighbour ceiling below two and zero rounds by message; a valid call
    walks the four diagonal corners with the frozen keys passed through and
    the ceiling respected."""
    start = {"n_neighbors": 15, "min_dist": 0.1}
    numeric = ("Local UMAP optimization requires numeric n_neighbors, "
               "min_dist, and step sizes.")
    positive = ("Local UMAP optimization steps must be positive and the "
                "minimum improvement must be zero or greater.")
    for cid, begin, kwargs, message in [
        ("missing-start", {"n_neighbors": 15}, {},
         "Local UMAP optimization needs one starting value for n_neighbors "
         "and min_dist; missing ['min_dist']."),
        ("word-n-step", start, {"n_neighbors_step": "one"}, numeric),
        ("word-d-step", start, {"min_dist_step": "small"}, numeric),
        ("word-ceiling", start, {"n_neighbors_max": "lots"}, numeric),
        ("word-start", {"n_neighbors": "fifteen", "min_dist": 0.1}, {},
         numeric),
        ("word-threshold", start, {"min_improvement": "a little"}, numeric),
        ("word-rounds", start, {"n_trials": "many"}, numeric),
        ("zero-n-step", start, {"n_neighbors_step": 0}, positive),
        ("fractional-n-step-truncates-to-zero", start,
         {"n_neighbors_step": 0.9}, positive),
        ("zero-d-step", start, {"min_dist_step": 0.0}, positive),
        ("negative-threshold", start, {"min_improvement": -1.0}, positive),
        ("ceiling-one", start, {"n_neighbors_max": 1},
         "n_neighbors_max must be at least 2."),
        ("zero-rounds", start, {"n_trials": 0},
         "Adaptive UMAP optimization needs n_trials/max rounds of at least "
         "1."),
    ]:
        with subtests.test(case=cid):
            with pytest.raises(ValueError) as info:
                hp.local_direction_search(_never_called, begin, **kwargs)
            assert str(info.value) == message

    with subtests.test(case="walks-the-diagonal-corners"):
        seen = []
        result = hp.local_direction_search(
            lambda p: seen.append(dict(p)) or 0.5,
            {"n_neighbors": 10, "min_dist": 0.1, "metric": "cosine"},
            n_trials=1, n_neighbors_step=2, n_neighbors_max=11,
            min_dist_step=0.1, metric="trustworthiness")
        assert seen == [
            {"metric": "cosine", "n_neighbors": 8, "min_dist": 0.0},
            {"metric": "cosine", "n_neighbors": 8, "min_dist": 0.2},
            {"metric": "cosine", "n_neighbors": 11, "min_dist": 0.0},
            {"metric": "cosine", "n_neighbors": 11, "min_dist": 0.2}]
        assert result.metric == "trustworthiness"
        assert result.notes[0].startswith(
            "Walk over 2 parameter(s): n_neighbors (step 2, resolution 2), "
            "min_dist (step 0.1, resolution 2).")
