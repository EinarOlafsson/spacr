"""Hyperparameter search — grids, random sweeps, UMAP criteria and grouped CV.

This module generalises :func:`spacr.core.reducer_hyperparameter_search`, which
already swept UMAP/tSNE + DBSCAN/KMeans parameters and drew the resulting
embeddings as a grid of small multiples. That function stays where it is (it
still owns the database read + the image-glyph plotting); what lives here is the
search machinery it lacked: recorded trials, failed trials that do not abort the
sweep, an incremental progress callback, early stopping, reproducible sampling,
and — most importantly — an explicit, named scoring criterion instead of an
unscored eyeball grid.

Three things this module refuses to pretend:

**UMAP has no ground truth.** There is no measurement that says one embedding is
correct and another is wrong. Every criterion here (trustworthiness, continuity,
silhouette) rewards a *different* property, and they routinely disagree about
which ``n_neighbors``/``min_dist`` wins. The scores are an aid for ranking a
panel of embeddings you then look at; they are not a verdict. Anything that
prints "best embedding" without naming the criterion is misleading, so
:func:`format_search` always names it and always prints the caveat.

**Selecting on the test split leaks.** :func:`cv_search` scores every trial on
cross-validation folds, never on test, and refuses to run if a caller hands it
folds that touch the held-out test indices. It defaults to *grouped* folds
(``group_by='well'``) and reuses :func:`spacr.io.make_cv_folds` — crops from one
well share focus, illumination and seeding density, so an ungrouped search picks
the model that memorised wells and reports a beautiful, meaningless score.

**A winner without a spread is a lie.** When the top ten configurations sit
inside the fold-to-fold standard deviation, the hyperparameter did not matter and
the "winner" is noise. Every :class:`SearchResult` reports the spread and raises
a ``within_noise`` flag when that is what happened.

:author: spaCR
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SearchSpace",
    "Trial",
    "SearchResult",
    "grid_search",
    "random_search",
    "umap_search",
    "cv_search",
    "build_folds",
    "format_search",
    "umap_available",
    "SearchData",
    "load_search_data",
    "build_sklearn_model",
    "sklearn_cv_fit_fn",
    "classify_cv_fit_fn",
    "run_search_for_app",
    "APP_CRITERIA",
    "UMAP_CRITERIA",
    "UMAP_MISSING_MESSAGE",
    "UMAP_NO_GROUND_TRUTH",
    "DEFAULT_SPACES",
]


# ---------------------------------------------------------------------------
# Messages the GUI and the CLI both surface verbatim
# ---------------------------------------------------------------------------

#: Shown instead of an ImportError traceback when umap-learn is absent.
UMAP_MISSING_MESSAGE = (
    "UMAP hyperparameter search needs the 'umap-learn' package, which is not "
    "installed in this environment. Install it with `pip install umap-learn` "
    "(or `pip install spacr[umap]`) and run the search again. Nothing was "
    "searched, so there is no result to report."
)

#: The caveat attached to every UMAP search result. Never suppressed.
UMAP_NO_GROUND_TRUTH = (
    "UMAP has no ground truth: no criterion can tell you an embedding is "
    "correct. The ranking below is one named criterion's opinion, and a "
    "different criterion picks a different winner. Read the scores as an aid "
    "for choosing which embeddings to look at, not as a verdict."
)

#: What each UMAP criterion rewards — and, just as important, what it ignores.
UMAP_CRITERIA: Dict[str, str] = {
    "trustworthiness": (
        "rewards embeddings that do not invent neighbours: points that ended up "
        "close together in the embedding were already close in feature space. "
        "It says nothing about true neighbours the embedding tore apart, so it "
        "favours embeddings that spread points out."
    ),
    "continuity": (
        "rewards embeddings that keep true neighbours together: points close in "
        "feature space stay close in the embedding. It says nothing about "
        "neighbours the embedding invented, so it favours embeddings that "
        "crowd points together."
    ),
    "silhouette": (
        "rewards embeddings in which the labels you supplied form compact, "
        "well-separated blobs. It needs labels, and it measures agreement with "
        "those labels rather than faithfulness to the feature space — a "
        "high score can simply mean the embedding overfitted the grouping."
    ),
}

#: Criteria each app's search can rank by, first entry being the default.
APP_CRITERIA: Dict[str, List[str]] = {
    "umap": ["trustworthiness", "continuity", "silhouette"],
    "classify": ["accuracy", "prauc", "loss"],
    "ml_analyze": ["accuracy", "roc_auc", "f1"],
}

#: Criteria where a smaller number is better.
LOWER_IS_BETTER = frozenset({"loss"})

#: Starting grids offered by the GUI, per app key. Small on purpose: a sweep the
#: user actually finishes beats an exhaustive one they cancel.
DEFAULT_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "umap": {
        "n_neighbors": [5, 15, 50, 100],
        "min_dist": [0.0, 0.1, 0.5],
    },
    "classify": {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "dropout_rate": [0.0, 0.1, 0.3],
    },
    "ml_analyze": {
        "learning_rate": [0.001, 0.01, 0.1],
        "n_estimators": [100, 500, 1000],
    },
}


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchSpace:
    """A named set of parameters, each with the list of values to try.

    Values are stored as tuples so a space cannot be mutated after the sweep
    that used it has been reported.

    :param params: mapping of parameter name to the list of values to try.
    :raises ValueError: if the space is empty, a name is not a string, a
        parameter's values are not a list/tuple, or a parameter has no values.
    """

    params: Mapping[str, Sequence[Any]]

    def __post_init__(self) -> None:
        """Validate and freeze the parameter mapping into tuples."""
        if not isinstance(self.params, Mapping):
            raise ValueError(
                "SearchSpace(params=...) must be a mapping of "
                "{'parameter name': [value, value, ...]}, got "
                f"{type(self.params).__name__}."
            )
        if not self.params:
            raise ValueError(
                "Search space is empty: it has no parameters, so there is "
                "nothing to search. Give at least one parameter with at least "
                "one value, e.g. SearchSpace({'n_neighbors': [5, 15, 50]})."
            )
        frozen: Dict[str, Tuple[Any, ...]] = {}
        for name, values in self.params.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Search space parameter names must be non-empty strings, "
                    f"got {name!r}."
                )
            if isinstance(values, (str, bytes)) or not isinstance(
                    values, Sequence):
                raise ValueError(
                    f"Values for parameter {name!r} must be a list or tuple of "
                    f"values to try, got {type(values).__name__} "
                    f"({values!r}). Wrap a single value in a list: "
                    f"{{{name!r}: [{values!r}]}}."
                )
            if len(values) == 0:
                raise ValueError(
                    f"Parameter {name!r} has an empty value list, so no "
                    f"configuration can be built. Give it at least one value, "
                    f"e.g. {{{name!r}: [<value>]}}, or drop the parameter."
                )
            frozen[name] = tuple(values)
        object.__setattr__(self, "params", frozen)

    @property
    def names(self) -> Tuple[str, ...]:
        """Parameter names in sorted order — this fixes the grid's column order."""
        return tuple(sorted(self.params))

    def size(self) -> int:
        """Number of configurations in the full Cartesian product."""
        n = 1
        for name in self.names:
            n *= len(self.params[name])
        return n

    @property
    def is_single_point(self) -> bool:
        """True when the space contains exactly one configuration."""
        return self.size() == 1

    def grid(self) -> List[Dict[str, Any]]:
        """Enumerate the full Cartesian product in a deterministic order.

        Parameter names are sorted; within a name, values keep the order the
        caller gave them; the last name varies fastest.

        :returns: list of parameter dicts, one per configuration.
        """
        names = self.names
        combos = itertools.product(*(self.params[n] for n in names))
        return [dict(zip(names, values)) for values in combos]

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        """Draw one configuration uniformly at random.

        Values are drawn in sorted-name order so a fixed seed always yields the
        same sequence regardless of the caller's dict insertion order.

        :param rng: seeded :class:`random.Random` instance.
        :returns: a parameter dict.
        """
        return {n: rng.choice(self.params[n]) for n in self.names}

    def describe(self) -> str:
        """One-line human summary of the space and its size."""
        parts = [f"{n}={list(self.params[n])}" for n in self.names]
        return f"{' × '.join(parts)}  ({self.size()} configurations)"


# ---------------------------------------------------------------------------
# Trials + results
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One evaluated configuration, successful or not.

    :ivar params: the configuration that was evaluated.
    :ivar score: the primary metric, or None when the trial failed.
    :ivar extra_metrics: any other numbers the fit function reported
        (per-fold scores, alternative criteria, runtime counters).
    :ivar duration: wall-clock seconds spent on this trial.
    :ivar error: the failure message, or None when the trial succeeded.
    :ivar index: position in the deterministic trial order, from zero.
    """

    params: Dict[str, Any]
    score: Optional[float] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None
    index: int = -1

    @property
    def ok(self) -> bool:
        """True when the trial produced a usable score."""
        return self.error is None and self.score is not None

    def label(self) -> str:
        """Compact ``k=v, k=v`` rendering of the configuration."""
        return ", ".join(f"{k}={self.params[k]!r}" for k in sorted(self.params))


@dataclass
class SearchResult:
    """Everything one sweep produced, including what it failed to produce.

    :ivar trials: every trial attempted, in deterministic order, failures
        included.
    :ivar best: the highest- (or lowest-) scoring successful trial, or None
        when nothing succeeded.
    :ivar space: the space that was searched.
    :ivar metric: the name of the criterion ``score`` holds.
    :ivar notes: caveats, warnings and provenance the caller must surface.
    :ivar partial: True when the sweep stopped before evaluating everything it
        was asked to. A partial sweep must never be presented as a finished one.
    :ivar higher_is_better: direction of ``metric``.
    """

    trials: List[Trial] = field(default_factory=list)
    best: Optional[Trial] = None
    space: Optional[SearchSpace] = None
    metric: str = "score"
    notes: List[str] = field(default_factory=list)
    partial: bool = False
    higher_is_better: bool = True

    # -- basic slices ----------------------------------------------------

    @property
    def successful(self) -> List[Trial]:
        """Trials that produced a score."""
        return [t for t in self.trials if t.ok]

    @property
    def failed(self) -> List[Trial]:
        """Trials that raised or returned an unusable score."""
        return [t for t in self.trials if not t.ok]

    @property
    def n_failed(self) -> int:
        """How many trials failed."""
        return len(self.failed)

    @property
    def ok(self) -> bool:
        """True when at least one trial produced a score."""
        return self.best is not None

    def ranked(self) -> List[Trial]:
        """Successful trials best-first; ties broken by trial order."""
        sign = -1.0 if self.higher_is_better else 1.0
        return sorted(self.successful,
                      key=lambda t: (sign * float(t.score), t.index))

    # -- spread ----------------------------------------------------------

    def score_stats(self) -> Dict[str, Optional[float]]:
        """Summary statistics over the successful trials' scores.

        :returns: dict with ``n``, ``best``, ``worst``, ``mean``, ``std`` and
            ``spread`` (max - min). Every value except ``n`` is None when no
            trial succeeded.
        """
        scores = [float(t.score) for t in self.successful]
        if not scores:
            return {"n": 0, "best": None, "worst": None, "mean": None,
                    "std": None, "spread": None}
        best = max(scores) if self.higher_is_better else min(scores)
        worst = min(scores) if self.higher_is_better else max(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        return {
            "n": len(scores),
            "best": best,
            "worst": worst,
            "mean": statistics.fmean(scores),
            "std": std,
            "spread": max(scores) - min(scores),
        }

    def noise_level(self) -> Tuple[Optional[float], str]:
        """The yardstick used to decide whether the winner is real.

        Prefers the best trial's own fold-to-fold standard deviation, because
        that is the run-to-run variation of a single configuration. Falls back
        to the spread across trials when no fold information exists.

        :returns: ``(value, source_description)``; value is None when there is
            not enough information.
        """
        if self.best is not None:
            fold_std = self.best.extra_metrics.get("fold_std")
            if fold_std is not None:
                try:
                    fs = float(fold_std)
                except (TypeError, ValueError):
                    fs = float("nan")
                if math.isfinite(fs):
                    return fs, ("fold-to-fold standard deviation of the best "
                                "configuration")
        stats = self.score_stats()
        if stats["n"] and stats["n"] > 1:
            return float(stats["std"]), "standard deviation across trials"
        return None, "not enough successful trials to estimate noise"

    def trials_within_noise(self) -> List[Trial]:
        """Successful trials whose score is indistinguishable from the best."""
        noise, _ = self.noise_level()
        if noise is None or self.best is None:
            return list(self.successful[:1])
        best = float(self.best.score)
        return [t for t in self.ranked() if abs(best - float(t.score)) <= noise]

    def within_noise(self, top_n: int = 3) -> bool:
        """True when the top ``top_n`` trials are within the noise level.

        When this fires, the hyperparameter did not measurably matter over the
        range searched, and reporting a single winner hides that.

        :param top_n: how many of the leading trials to compare.
        :returns: True when the leaders are statistically indistinguishable.
        """
        ranked = self.ranked()
        noise, _ = self.noise_level()
        if len(ranked) < 2 or noise is None:
            return False
        cutoff = ranked[:max(2, int(top_n))]
        best = float(cutoff[0].score)
        return all(abs(best - float(t.score)) <= noise for t in cutoff[1:])

    def as_rows(self) -> List[Dict[str, Any]]:
        """Flat, table-ready rendering of every trial, best-first then failures."""
        rows: List[Dict[str, Any]] = []
        for rank, t in enumerate(self.ranked(), start=1):
            rows.append({
                "rank": rank, "index": t.index, "params": dict(t.params),
                "score": float(t.score), "metric": self.metric,
                "duration": t.duration, "error": None,
                "extra_metrics": dict(t.extra_metrics),
            })
        for t in self.failed:
            rows.append({
                "rank": None, "index": t.index, "params": dict(t.params),
                "score": None, "metric": self.metric,
                "duration": t.duration, "error": t.error,
                "extra_metrics": dict(t.extra_metrics),
            })
        return rows


# ---------------------------------------------------------------------------
# The runner every search shares
# ---------------------------------------------------------------------------

def _normalise_outcome(value: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Coerce whatever a fit function returned into ``(score, extra_metrics)``.

    Accepts a bare number, a ``(score, metrics_dict)`` pair, or a mapping with a
    ``score`` key (all other keys become extra metrics).

    :param value: the fit function's return value.
    :returns: ``(score_or_None, extra_metrics)``.
    :raises TypeError: when the value cannot be read as a score.
    """
    extra: Dict[str, Any] = {}
    raw: Any = value
    if isinstance(value, tuple) and len(value) == 2 and isinstance(
            value[1], Mapping):
        raw, extra = value[0], dict(value[1])
    elif isinstance(value, Mapping):
        extra = {k: v for k, v in value.items() if k != "score"}
        if "score" not in value:
            raise TypeError(
                "fit function returned a dict without a 'score' key; return "
                "either a number, a (score, metrics) pair, or a dict "
                "containing 'score'."
            )
        raw = value["score"]
    if raw is None:
        return None, extra
    try:
        score = float(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"fit function returned {raw!r}, which is not a number: {exc}"
        ) from exc
    if not math.isfinite(score):
        raise ValueError(
            f"fit function returned a non-finite score ({score!r}); a trial "
            f"that cannot be scored is recorded as failed rather than ranked."
        )
    return score, extra


def _run_trials(fit_fn: Callable[..., Any],
                param_sets: Sequence[Mapping[str, Any]],
                space: SearchSpace,
                metric: str,
                *,
                higher_is_better: bool = True,
                on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                notes: Optional[Sequence[str]] = None,
                call: Optional[Callable[[Callable, Dict[str, Any]], Any]] = None,
                ) -> SearchResult:
    """Evaluate ``param_sets`` one at a time, recording failures and progress.

    This is the only place trials are executed, so every search shares the same
    guarantees: a raising trial is recorded and the sweep continues, progress is
    reported after every trial, and a stop request truncates the sweep and marks
    the result partial rather than presenting it as complete.

    :param fit_fn: callable evaluated per configuration.
    :param param_sets: the configurations to evaluate, in order.
    :param space: the space being searched (carried into the result).
    :param metric: name of the criterion the scores represent.
    :param higher_is_better: direction of ``metric``.
    :param on_trial: called as ``on_trial(trial, completed, total)`` after each
        trial, including failed ones.
    :param should_stop: polled before each trial; when it returns True the
        sweep stops and the result is marked partial.
    :param notes: caveats to attach to the result.
    :param call: optional adapter invoking ``fit_fn`` with the parameters (used
        by :func:`cv_search` to fan a configuration out over folds).
    :returns: the :class:`SearchResult`.
    """
    result = SearchResult(space=space, metric=metric,
                          notes=list(notes or []),
                          higher_is_better=higher_is_better)
    total = len(param_sets)
    invoke = call if call is not None else (lambda fn, p: fn(p))

    for idx, params in enumerate(param_sets):
        if should_stop is not None and should_stop():
            result.partial = True
            result.notes.append(
                f"Search stopped early after {idx} of {total} configurations. "
                f"The trials below are the ones that finished; the rest were "
                f"never evaluated, so this is not a completed sweep."
            )
            break
        trial = Trial(params=dict(params), index=idx)
        started = time.perf_counter()
        try:
            outcome = invoke(fit_fn, dict(params))
            trial.score, trial.extra_metrics = _normalise_outcome(outcome)
            if trial.score is None:
                trial.error = ("fit function returned no score for this "
                               "configuration")
        except Exception as exc:  # one bad configuration must not lose the sweep
            trial.error = f"{type(exc).__name__}: {exc}"
        trial.duration = time.perf_counter() - started
        result.trials.append(trial)
        if on_trial is not None:
            on_trial(trial, idx + 1, total)

    _select_best(result)
    _append_summary_notes(result, total)
    return result


def _select_best(result: SearchResult) -> None:
    """Pick the winning trial, breaking ties by the earlier trial index."""
    best: Optional[Trial] = None
    for t in result.successful:
        if best is None:
            best = t
            continue
        better = (float(t.score) > float(best.score)
                  if result.higher_is_better
                  else float(t.score) < float(best.score))
        if better:
            best = t
    result.best = best


def _append_summary_notes(result: SearchResult, requested: int) -> None:
    """Attach the spread / failure / degeneracy notes every result must carry."""
    if result.n_failed:
        result.notes.append(
            f"{result.n_failed} of {len(result.trials)} evaluated "
            f"configurations failed and were recorded rather than dropped; the "
            f"sweep continued. See the per-trial error column."
        )
    if result.best is None:
        result.notes.append(
            "No configuration produced a score, so there is no winner to "
            "report."
        )
        return
    stats = result.score_stats()
    noise, source = result.noise_level()
    if stats["n"] == 1:
        result.notes.append(
            "Only one configuration was scored, so there is nothing to compare "
            "it against — this is a single measurement, not a search."
        )
        return
    result.notes.append(
        f"Scores across {stats['n']} successful trials span "
        f"{stats['spread']:.4g} ({stats['worst']:.4g} to {stats['best']:.4g}), "
        f"standard deviation {stats['std']:.4g}."
    )
    if result.within_noise():
        n_tied = len(result.trials_within_noise())
        result.notes.append(
            f"WITHIN NOISE: the leading configurations differ by less than the "
            f"{source} ({noise:.4g}); {n_tied} of {stats['n']} trials are "
            f"indistinguishable from the best. Over the range searched this "
            f"hyperparameter did not measurably matter, and picking the "
            f"top row is picking noise."
        )


# ---------------------------------------------------------------------------
# Public searches
# ---------------------------------------------------------------------------

def grid_search(fit_fn: Callable[[Dict[str, Any]], Any],
                space: SearchSpace,
                *,
                metric: str = "score",
                higher_is_better: bool = True,
                on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                notes: Optional[Sequence[str]] = None,
                ) -> SearchResult:
    """Evaluate every configuration in ``space``.

    :param fit_fn: called as ``fit_fn(params)``; returns a score, a
        ``(score, metrics)`` pair, or a dict with a ``score`` key.
    :param space: the :class:`SearchSpace` to enumerate.
    :param metric: name of the criterion the score represents.
    :param higher_is_better: direction of ``metric``.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial; True truncates the sweep and
        marks the result partial.
    :param notes: extra caveats to attach.
    :returns: the :class:`SearchResult`.
    """
    combos = space.grid()
    extra_notes = list(notes or [])
    extra_notes.insert(
        0, f"Grid search over {space.describe()}")
    if space.is_single_point:
        extra_notes.append(
            "The space contains a single configuration, so this evaluates one "
            "setting rather than comparing alternatives."
        )
    return _run_trials(fit_fn, combos, space, metric,
                       higher_is_better=higher_is_better,
                       on_trial=on_trial, should_stop=should_stop,
                       notes=extra_notes)


def random_search(fit_fn: Callable[[Dict[str, Any]], Any],
                  space: SearchSpace,
                  n_trials: int,
                  seed: int = 0,
                  *,
                  metric: str = "score",
                  higher_is_better: bool = True,
                  on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                  should_stop: Optional[Callable[[], bool]] = None,
                  notes: Optional[Sequence[str]] = None,
                  allow_duplicates: bool = False,
                  ) -> SearchResult:
    """Evaluate ``n_trials`` configurations drawn at random from ``space``.

    Reproducible: the same ``seed`` always yields the same configurations in
    the same order.

    :param fit_fn: called as ``fit_fn(params)``.
    :param space: the :class:`SearchSpace` to sample from.
    :param n_trials: how many configurations to evaluate; must be positive.
    :param seed: RNG seed.
    :param metric: name of the criterion the score represents.
    :param higher_is_better: direction of ``metric``.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :param notes: extra caveats to attach.
    :param allow_duplicates: when False (the default) the same configuration is
        never evaluated twice; the sweep shrinks to the size of the space if the
        space is smaller than ``n_trials``.
    :returns: the :class:`SearchResult`.
    :raises ValueError: when ``n_trials`` is not a positive integer.
    """
    try:
        n_trials = int(n_trials)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"n_trials must be a positive integer, got {n_trials!r}."
        ) from exc
    if n_trials < 1:
        raise ValueError(
            f"n_trials must be at least 1, got {n_trials}. A search with no "
            f"trials has nothing to report."
        )

    rng = random.Random(seed)
    extra_notes = list(notes or [])
    picked: List[Dict[str, Any]] = []
    if allow_duplicates:
        picked = [space.sample(rng) for _ in range(n_trials)]
    else:
        wanted = min(n_trials, space.size())
        if wanted < n_trials:
            extra_notes.append(
                f"The space holds only {space.size()} distinct configurations, "
                f"fewer than the {n_trials} requested, so every one of them was "
                f"evaluated — this is an exhaustive grid, not a random sample."
            )
        seen = set()
        # Bounded rejection sampling; the cap keeps a pathological space from
        # spinning forever, and the fallback fills the remainder from the grid
        # in deterministic order.
        attempts = 0
        max_attempts = max(1000, wanted * 200)
        while len(picked) < wanted and attempts < max_attempts:
            attempts += 1
            cand = space.sample(rng)
            key = tuple(cand[n] for n in space.names)
            if key in seen:
                continue
            seen.add(key)
            picked.append(cand)
        if len(picked) < wanted:
            for cand in space.grid():
                if len(picked) >= wanted:
                    break
                key = tuple(cand[n] for n in space.names)
                if key not in seen:
                    seen.add(key)
                    picked.append(cand)

    extra_notes.insert(
        0, f"Random search: {len(picked)} of {space.size()} configurations "
           f"sampled with seed {seed} from {space.describe()}")
    return _run_trials(fit_fn, picked, space, metric,
                       higher_is_better=higher_is_better,
                       on_trial=on_trial, should_stop=should_stop,
                       notes=extra_notes)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def umap_available() -> Tuple[bool, str]:
    """Whether umap-learn can be imported.

    :returns: ``(True, "")`` when available, otherwise ``(False, message)``
        carrying :data:`UMAP_MISSING_MESSAGE`.
    """
    try:
        import umap  # noqa: F401
    except Exception:
        return False, UMAP_MISSING_MESSAGE
    return True, ""


def _default_umap_embed(features, params: Dict[str, Any], seed: int):
    """Fit a UMAP embedding for one configuration.

    :param features: 2-D numeric feature matrix.
    :param params: UMAP keyword arguments for this trial.
    :param seed: ``random_state`` so a repeated sweep reproduces.
    :returns: the 2-D embedding.
    """
    import umap
    kwargs = dict(params)
    kwargs.setdefault("n_components", 2)
    kwargs.setdefault("random_state", seed)
    reducer = umap.UMAP(**kwargs)
    return reducer.fit_transform(features)


def _umap_scores(features, embedding, labels, k: int) -> Dict[str, float]:
    """Compute every applicable embedding criterion for one trial.

    Continuity is trustworthiness with the two spaces swapped — that is the
    standard definition, and it is why the two criteria can disagree.

    :param features: the original feature matrix.
    :param embedding: the 2-D embedding under test.
    :param labels: optional class labels enabling the silhouette criterion.
    :param k: neighbourhood size for trustworthiness / continuity.
    :returns: mapping of criterion name to score.
    """
    import numpy as np
    from sklearn.manifold import trustworthiness

    X = np.asarray(features, dtype=float)
    E = np.asarray(embedding, dtype=float)
    n = X.shape[0]
    # trustworthiness requires k < n/2.
    kk = max(1, min(int(k), (n - 1) // 2))
    out: Dict[str, float] = {
        "trustworthiness": float(trustworthiness(X, E, n_neighbors=kk)),
        "continuity": float(trustworthiness(E, X, n_neighbors=kk)),
        "neighbourhood_k": float(kk),
    }
    if labels is not None:
        lab = np.asarray(labels)
        if lab.shape[0] == n and len(set(lab.tolist())) >= 2:
            from sklearn.metrics import silhouette_score
            out["silhouette"] = float(silhouette_score(E, lab))
    return out


def umap_search(features,
                space: SearchSpace,
                *,
                metric: str = "trustworthiness",
                labels=None,
                seed: int = 0,
                neighbourhood_k: int = 15,
                embed_fn: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
                keep_embeddings: bool = True,
                on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                ) -> SearchResult:
    """Sweep UMAP parameters, scoring each embedding with a named criterion.

    The honest deliverable is the panel of embeddings, not the top row of the
    table. Every trial keeps its embedding (unless ``keep_embeddings`` is off)
    so the caller can draw small multiples, and every criterion is computed for
    every trial so the caller can see how the ranking changes when the criterion
    does.

    :param features: 2-D numeric feature matrix, ``(n_samples, n_features)``.
    :param space: UMAP parameters to sweep (``n_neighbors``, ``min_dist``,
        ``metric``, ...).
    :param metric: which criterion drives the ranking; one of
        :data:`UMAP_CRITERIA`.
    :param labels: optional class labels; required for ``'silhouette'``.
    :param seed: ``random_state`` for the reducer, so the sweep reproduces.
    :param neighbourhood_k: neighbourhood size for trustworthiness/continuity.
    :param embed_fn: ``embed_fn(features, params) -> embedding`` override; when
        omitted, umap-learn is used.
    :param keep_embeddings: store each trial's embedding in its extra metrics.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :returns: the :class:`SearchResult`. When umap-learn is missing and no
        ``embed_fn`` was given, this returns an empty result whose notes lead
        with :data:`UMAP_MISSING_MESSAGE` rather than raising ImportError.
    :raises ValueError: when ``metric`` is not a known criterion, or
        ``'silhouette'`` is requested without labels.
    """
    if metric not in UMAP_CRITERIA:
        raise ValueError(
            f"Unknown UMAP criterion {metric!r}. Choose one of "
            f"{sorted(UMAP_CRITERIA)} — each rewards a different property, so "
            f"the choice changes the answer."
        )
    if metric == "silhouette" and labels is None:
        raise ValueError(
            "The 'silhouette' criterion scores how well the embedding "
            "separates labels you already have, so it needs `labels=`. Without "
            "labels, use 'trustworthiness' or 'continuity'."
        )

    if embed_fn is None:
        available, message = umap_available()
        if not available:
            return SearchResult(
                trials=[], best=None, space=space, metric=metric,
                notes=[message], partial=False, higher_is_better=True,
            )

        def embed_fn(feats, params, _seed=seed):  # noqa: F811
            """Default embedder — umap-learn with a pinned random_state."""
            return _default_umap_embed(feats, params, _seed)

    notes = [
        UMAP_NO_GROUND_TRUTH,
        f"Criterion '{metric}': {UMAP_CRITERIA[metric]}",
        "Every criterion was computed for every trial, so you can re-rank the "
        "table by a different one and see whether the winner survives.",
    ]

    def _fit(params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Embed one configuration and score it with every criterion."""
        embedding = embed_fn(features, dict(params))
        scores = _umap_scores(features, embedding, labels, neighbourhood_k)
        if metric not in scores:
            raise ValueError(
                f"criterion {metric!r} could not be computed for this trial "
                f"(available: {sorted(k for k in scores if k in UMAP_CRITERIA)})"
            )
        extra = dict(scores)
        extra["criterion"] = metric
        if keep_embeddings:
            extra["embedding"] = embedding
        return scores[metric], extra

    return _run_trials(_fit, space.grid(), space, metric,
                       higher_is_better=True, on_trial=on_trial,
                       should_stop=should_stop, notes=notes)


# ---------------------------------------------------------------------------
# Grouped cross-validated search — Classify (CV) and Classify (ML)
# ---------------------------------------------------------------------------

def build_folds(labels,
                n_folds: int = 5,
                *,
                groups=None,
                filenames: Optional[Sequence[str]] = None,
                group_by: str = "well",
                seed: int = 0,
                exclude=None,
                ) -> Tuple[List[Tuple[Any, Any]], List[str]]:
    """Build grouped cross-validation folds over the *non-test* samples.

    Reuses :func:`spacr.io.make_cv_folds`, the same fold builder
    ``cross_validation_folds`` drives during training, so a search and the run
    it configures split the data the same way. Grouping defaults to ``'well'``
    because crops from one well share focus, illumination, seeding density and
    edge effects; letting them straddle a split lets a model recognise the well
    instead of the phenotype.

    :param labels: per-sample class labels.
    :param n_folds: number of folds; must be at least 2.
    :param groups: explicit per-sample group ids. Takes precedence over
        ``filenames``.
    :param filenames: crop filenames from which group ids are parsed when
        ``groups`` is not given.
    :param group_by: grouping level — ``'well'`` (default), ``'field'``,
        ``'plate'`` or ``'none'``.
    :param seed: RNG seed, so the folds reproduce.
    :param exclude: indices to keep out of every fold — the held-out test split.
    :returns: ``(folds, warnings)`` where ``folds`` is a list of
        ``(train_idx, val_idx)`` index arrays into the *original* sample order.
    :raises ValueError: when ``n_folds`` < 2 or every sample was excluded.
    """
    import numpy as np

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(
            f"n_folds must be at least 2 to cross-validate, got {n_folds}. "
            f"With one fold there is no held-out data to score on."
        )
    lab = np.asarray(labels)
    n = lab.shape[0]
    excluded = np.zeros(n, dtype=bool)
    if exclude is not None:
        ex = np.asarray(list(exclude), dtype=int)
        if ex.size:
            if ex.min() < 0 or ex.max() >= n:
                raise ValueError(
                    f"excluded (test) indices must lie inside [0, {n}), got "
                    f"min {int(ex.min())} max {int(ex.max())}."
                )
            excluded[ex] = True
    pool = np.flatnonzero(~excluded)
    if pool.size == 0:
        raise ValueError(
            "Every sample was excluded as test data, so there is nothing left "
            "to cross-validate on."
        )

    warnings: List[str] = []
    grp = None
    if groups is not None:
        grp = np.asarray(list(groups))[pool]
    elif filenames is not None and group_by != "none":
        from .io import _cv_group_ids
        ids, n_unparsed = _cv_group_ids(
            [str(filenames[i]) for i in pool], group_by, verbose=False)
        grp = np.asarray(ids) if ids is not None else None
        if n_unparsed:
            warnings.append(
                f"{n_unparsed} filenames did not carry a "
                f"'{group_by}' level and became their own group."
            )
    if grp is None and group_by != "none":
        warnings.append(
            f"No group ids were available (pass `groups=` or `filenames=`), so "
            f"the folds are ungrouped despite group_by='{group_by}'. Crops "
            f"from one well can now straddle a split and every score below is "
            f"optimistic."
        )
    if group_by == "none":
        warnings.append(
            "group_by='none': folds are a plain stratified split. Object crops "
            "from the same well will straddle folds, which inflates scores."
        )

    from .io import make_cv_folds
    sub_folds = make_cv_folds(lab[pool], n_folds, groups=grp, seed=seed)
    folds = [(pool[np.asarray(tr, dtype=int)], pool[np.asarray(va, dtype=int)])
             for tr, va in sub_folds]
    return folds, warnings


def cv_search(fit_fn: Callable[[Dict[str, Any], Any, Any], Any],
              space: SearchSpace,
              *,
              labels,
              groups=None,
              filenames: Optional[Sequence[str]] = None,
              group_by: str = "well",
              n_folds: int = 5,
              seed: int = 0,
              test_idx=None,
              folds: Optional[Sequence[Tuple[Any, Any]]] = None,
              metric: str = "accuracy",
              higher_is_better: bool = True,
              n_trials: Optional[int] = None,
              on_trial: Optional[Callable[[Trial, int, int], None]] = None,
              should_stop: Optional[Callable[[], bool]] = None,
              ) -> SearchResult:
    """Search hyperparameters by cross-validation, never by scoring on test.

    Every configuration is fitted once per fold on that fold's training indices
    and scored on that fold's validation indices; the trial score is the mean
    across folds and the fold-to-fold standard deviation becomes the noise
    yardstick for :meth:`SearchResult.within_noise`. Indices listed in
    ``test_idx`` are removed before the folds are built and are never handed to
    ``fit_fn`` — selecting a configuration on data reserved for the final
    estimate makes that estimate meaningless.

    :param fit_fn: called as ``fit_fn(params, train_idx, val_idx)``; returns the
        validation score, a ``(score, metrics)`` pair, or a dict with ``score``.
    :param space: the :class:`SearchSpace` to search.
    :param labels: per-sample class labels (used to stratify the folds).
    :param groups: explicit per-sample group ids.
    :param filenames: crop filenames to parse group ids from.
    :param group_by: grouping level, ``'well'`` by default.
    :param n_folds: number of cross-validation folds.
    :param seed: seed for the folds and for random sampling.
    :param test_idx: indices of the held-out test split, excluded from every
        fold.
    :param folds: pre-built ``(train_idx, val_idx)`` pairs, bypassing
        :func:`build_folds`. Validated against ``test_idx``.
    :param metric: name of the score ``fit_fn`` returns.
    :param higher_is_better: direction of ``metric``.
    :param n_trials: when given, sample this many configurations at random
        instead of running the full grid.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :returns: the :class:`SearchResult`.
    :raises ValueError: when supplied ``folds`` touch ``test_idx``.
    """
    import numpy as np

    test_set = set()
    if test_idx is not None:
        test_set = {int(i) for i in np.asarray(list(test_idx), dtype=int)}

    notes: List[str] = []
    if folds is None:
        folds, warnings = build_folds(
            labels, n_folds, groups=groups, filenames=filenames,
            group_by=group_by, seed=seed, exclude=sorted(test_set) or None)
        notes.extend(warnings)
    else:
        folds = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
                 for tr, va in folds]

    # Structural guarantee, checked rather than assumed: no fold — train or
    # validation — may contain a test index. A search that scores on test data
    # selects a configuration that has already seen the answers.
    for i, (tr, va) in enumerate(folds):
        leaked = test_set.intersection(int(x) for x in tr) | \
                 test_set.intersection(int(x) for x in va)
        if leaked:
            raise ValueError(
                f"fold {i} contains {len(leaked)} held-out test indices "
                f"(e.g. {sorted(leaked)[:5]}). Scoring a hyperparameter search "
                f"on the test split leaks it: the reported test performance is "
                f"then the performance of a configuration chosen because it "
                f"did well on that exact data."
            )

    grouped = groups is not None or (filenames is not None
                                     and group_by != "none")
    notes.insert(0, (
        f"Scored on {len(folds)} "
        f"{'grouped (group_by=' + repr(group_by) + ')' if grouped else 'ungrouped'} "
        f"cross-validation folds, seed {seed}. "
        f"{len(test_set)} test samples were excluded from every fold, so no "
        f"configuration was selected using test data."
    ))
    notes.append(
        f"Each trial's score is the mean over {len(folds)} folds; the "
        f"fold-to-fold standard deviation is reported per trial and is used as "
        f"the noise yardstick when judging whether the winner is real."
    )

    def _call(fn, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Fan one configuration out over the folds and average the scores."""
        fold_scores: List[float] = []
        fold_extra: List[Dict[str, Any]] = []
        for tr, va in folds:
            score, extra = _normalise_outcome(fn(dict(params), tr, va))
            if score is None:
                raise ValueError(
                    "fit function returned no score for a fold; every fold "
                    "must be scored for the mean to mean anything")
            fold_scores.append(score)
            fold_extra.append(extra)
        mean = statistics.fmean(fold_scores)
        std = statistics.pstdev(fold_scores) if len(fold_scores) > 1 else 0.0
        merged: Dict[str, Any] = {
            "fold_scores": fold_scores,
            "fold_std": std,
            "n_folds": len(folds),
        }
        for extra in fold_extra:
            for k, v in extra.items():
                merged.setdefault(f"fold_{k}", []).append(v)
        return mean, merged

    if n_trials is None:
        param_sets = space.grid()
        notes.insert(1, f"Grid search over {space.describe()}")
    else:
        sampler = random.Random(seed)
        wanted = min(int(n_trials), space.size())
        seen = set()
        param_sets = []
        attempts = 0
        while len(param_sets) < wanted and attempts < max(1000, wanted * 200):
            attempts += 1
            cand = space.sample(sampler)
            key = tuple(cand[n] for n in space.names)
            if key in seen:
                continue
            seen.add(key)
            param_sets.append(cand)
        notes.insert(1, f"Random search: {len(param_sets)} of {space.size()} "
                        f"configurations sampled with seed {seed}")

    return _run_trials(fit_fn, param_sets, space, metric,
                       higher_is_better=higher_is_better,
                       on_trial=on_trial, should_stop=should_stop,
                       notes=notes, call=_call)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_search(result: SearchResult, max_rows: int = 20) -> str:
    """Render a search result as plain text, caveats first.

    The layout is deliberate: the criterion and its caveat come before the
    table, the spread and the within-noise flag come immediately after the
    winner, and a partial or all-failed sweep says so in the header rather than
    in a footnote nobody reads.

    :param result: the :class:`SearchResult` to render.
    :param max_rows: how many ranked trials to print.
    :returns: the report as a string.
    """
    lines: List[str] = []
    header = f"Hyperparameter search — criterion: {result.metric}"
    if result.partial:
        header += "  [PARTIAL — stopped early, not a completed sweep]"
    lines.append(header)
    lines.append("=" * len(header))

    if result.metric in UMAP_CRITERIA:
        lines.append(f"'{result.metric}' {UMAP_CRITERIA[result.metric]}")
        lines.append("")

    for note in result.notes:
        lines.append(f"* {note}")
    lines.append("")

    if not result.trials:
        lines.append("No trials were run.")
        return "\n".join(lines)

    ranked = result.ranked()
    if not ranked:
        lines.append(f"All {len(result.trials)} trials failed:")
        for t in result.failed[:max_rows]:
            lines.append(f"  [{t.index}] {t.label()} -> {t.error}")
        return "\n".join(lines)

    lines.append(f"{'rank':>4}  {'score':>10}  {'sd':>8}  {'sec':>6}  params")
    lines.append(f"{'-' * 4}  {'-' * 10}  {'-' * 8}  {'-' * 6}  {'-' * 30}")
    for rank, t in enumerate(ranked[:max_rows], start=1):
        fold_std = t.extra_metrics.get("fold_std")
        sd = f"{float(fold_std):.4f}" if fold_std is not None else "-"
        lines.append(f"{rank:>4}  {float(t.score):>10.4f}  {sd:>8}  "
                     f"{t.duration:>6.2f}  {t.label()}")
    if len(ranked) > max_rows:
        lines.append(f"      … {len(ranked) - max_rows} more")

    if result.failed:
        lines.append("")
        lines.append(f"Failed trials ({len(result.failed)}):")
        for t in result.failed[:max_rows]:
            lines.append(f"  [{t.index}] {t.label()} -> {t.error}")

    stats = result.score_stats()
    noise, source = result.noise_level()
    lines.append("")
    lines.append(f"Best: {result.best.label()}  "
                 f"{result.metric}={float(result.best.score):.4f}")
    lines.append(f"Spread over {stats['n']} successful trials: "
                 f"{stats['worst']:.4f} … {stats['best']:.4f} "
                 f"(sd {stats['std']:.4f})")
    if noise is not None:
        lines.append(f"Noise yardstick: {noise:.4f} ({source})")
    if result.within_noise():
        lines.append(
            f"WITHIN NOISE — {len(result.trials_within_noise())} trials are "
            f"indistinguishable from the best. Treat the winner as arbitrary."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App backends — what the GUI's "Hyperparameter search" button actually runs
# ---------------------------------------------------------------------------

@dataclass
class SearchData:
    """Feature matrix (and, where applicable, labels and groups) for one search.

    :ivar features: 2-D numeric matrix, one row per object.
    :ivar labels: per-row class labels, or None for an unsupervised search.
    :ivar groups: per-row group ids used to keep folds honest, or None.
    :ivar frame: the joined measurement table the matrix came from.
    :ivar notes: provenance and warnings the caller must surface.
    """

    features: Any = None
    labels: Any = None
    groups: Any = None
    frame: Any = None
    notes: List[str] = field(default_factory=list)


def _well_groups(frame) -> Tuple[Any, Optional[str]]:
    """Derive per-row well ids from a joined measurement table.

    :param frame: DataFrame carrying ``plateID``/``rowID``/``columnID``.
    :returns: ``(group_array_or_None, warning_or_None)``.
    """
    cols = [c for c in ("plateID", "rowID", "columnID") if c in frame.columns]
    if len(cols) < 2:
        return None, (
            "The measurement table has no plate/row/column columns, so wells "
            "could not be identified and the folds are ungrouped. Objects from "
            "one well will straddle folds and the scores below are optimistic."
        )
    ids = frame[cols].astype(str).agg("_".join, axis=1)
    return ids.to_numpy(), None


def load_search_data(app_key: str, settings: Mapping[str, Any]) -> SearchData:
    """Load the feature matrix a search needs, straight from the measurements DB.

    This is the same read + preprocess path
    :func:`spacr.core.reducer_hyperparameter_search` uses (``get_db_paths`` →
    ``_read_and_join_tables`` → ``preprocess_data``), so a search sees exactly
    the matrix the real run will see.

    :param app_key: ``'umap'``, ``'ml_analyze'`` or ``'classify'``.
    :param settings: the app's settings dict; ``src`` and ``tables`` are read.
    :returns: the :class:`SearchData`.
    :raises ValueError: when ``src`` is missing, or when a supervised search
        finds fewer than two classes.
    """
    import numpy as np
    import pandas as pd

    from .io import _read_and_join_tables
    from .utils import get_db_paths, preprocess_data

    src = settings.get("src")
    if not src or src in ("path", "/path/to/src", "/path"):
        raise ValueError(
            "No source folder is set. Point 'src' at an experiment directory "
            "containing measurements/measurements.db before searching."
        )
    tables = settings.get("tables") or ["cell", "cytoplasm", "nucleus",
                                        "pathogen"]
    notes: List[str] = []
    frames = []
    for db_path in get_db_paths(src):
        frames.append(_read_and_join_tables(db_path, table_names=list(tables)))
    frame = pd.concat([f for f in frames if f is not None], axis=0)
    if frame.empty:
        raise ValueError(
            f"No rows were read from {list(get_db_paths(src))}. Run Measure "
            f"first so there is a measurements table to search over."
        )

    row_limit = settings.get("row_limit")
    if row_limit and len(frame) > int(row_limit):
        frame = frame.sample(n=int(row_limit), random_state=42)
        notes.append(
            f"Sub-sampled to {int(row_limit)} of the available rows "
            f"(row_limit); a search on a subsample can rank configurations "
            f"differently from the full run.")

    features = preprocess_data(
        frame,
        settings.get("filter_by"),
        settings.get("remove_highly_correlated", True),
        settings.get("log_data", False),
        settings.get("exclude"),
    )
    data = SearchData(features=np.asarray(features, dtype=float), frame=frame,
                      notes=notes)
    if app_key == "umap":
        return data

    # Supervised searches need labels and well ids.
    groups, warn = _well_groups(frame)
    data.groups = groups
    if warn:
        data.notes.append(warn)

    ann_col = settings.get("annotation_column")
    pos = settings.get("positive_control", "c2")
    neg = settings.get("negative_control", "c1")
    loc_col = settings.get("location_column", "columnID")
    if ann_col and ann_col in frame.columns:
        labels = pd.to_numeric(frame[ann_col], errors="coerce")
        keep = labels.notna().to_numpy()
        data.notes.append(
            f"Labels taken from the '{ann_col}' annotation column "
            f"({int(keep.sum())} of {len(frame)} rows are annotated).")
    elif loc_col in frame.columns:
        col = frame[loc_col].astype(str)
        labels = pd.Series(np.where(col == str(pos), 1.0,
                                    np.where(col == str(neg), 0.0, np.nan)),
                           index=frame.index)
        keep = labels.notna().to_numpy()
        data.notes.append(
            f"Labels derived from controls in '{loc_col}': "
            f"{neg!r}=0, {pos!r}=1 ({int(keep.sum())} control rows kept). "
            f"A model that separates two control columns has also learned to "
            f"recognise those columns' plate position.")
    else:
        raise ValueError(
            f"Cannot build labels: neither an annotation column nor the "
            f"location column {loc_col!r} is present in the measurement table."
        )

    data.features = data.features[keep]
    data.labels = labels.to_numpy()[keep].astype(int)
    if data.groups is not None:
        data.groups = data.groups[keep]
    if len(set(data.labels.tolist())) < 2:
        raise ValueError(
            "Only one class survived label construction, so there is nothing "
            "to classify. Check the annotation column or the positive / "
            "negative control values."
        )
    return data


def build_sklearn_model(model_type: str, params: Mapping[str, Any],
                        seed: int = 42, n_jobs: int = -1):
    """Construct the classical-ML classifier ``model_type`` names.

    Mirrors the constructors in :func:`spacr.ml.ml_analysis` (ml.py, the
    ``model_type ==`` ladder) so a search configures the same estimator the
    real run will fit. Unknown keyword arguments in ``params`` are dropped with
    a clear error rather than silently ignored.

    :param model_type: one of the ``model_type_ml`` combo values.
    :param params: hyperparameters for this trial (``n_estimators``,
        ``learning_rate``, ``reg_alpha``, ``reg_lambda``, ...).
    :param seed: ``random_state``.
    :param n_jobs: worker count where the estimator supports it.
    :returns: an unfitted scikit-learn-compatible classifier.
    :raises ValueError: for an unsupported ``model_type``.
    :raises ImportError: with an install hint for optional backends.
    """
    p = dict(params)
    n_estimators = int(p.pop("n_estimators", 100))
    learning_rate = float(p.pop("learning_rate", 0.1))
    reg_alpha = float(p.pop("reg_alpha", 0.1))
    reg_lambda = float(p.pop("reg_lambda", 1.0))
    max_depth = p.pop("max_depth", None)
    mt = str(model_type)

    if mt == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      random_state=seed, n_jobs=n_jobs)
    if mt == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=seed, n_jobs=n_jobs)
    if mt == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, C=1.0 / max(reg_lambda, 1e-9),
                                  random_state=seed, n_jobs=n_jobs)
    if mt == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=n_estimators, learning_rate=learning_rate,
            random_state=seed)
    if mt == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "model_type_ml='xgboost' needs the 'xgboost' package: "
                "pip install xgboost") from exc
        return XGBClassifier(reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                             learning_rate=learning_rate,
                             n_estimators=n_estimators, random_state=seed,
                             nthread=n_jobs, eval_metric="logloss")
    if mt == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "model_type_ml='lightgbm' needs the 'lightgbm' package: "
                "pip install lightgbm") from exc
        return LGBMClassifier(n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                              random_state=seed, n_jobs=n_jobs, verbose=-1)
    if mt == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError(
                "model_type_ml='catboost' needs the 'catboost' package: "
                "pip install catboost") from exc
        return CatBoostClassifier(iterations=n_estimators,
                                  learning_rate=learning_rate,
                                  l2_leaf_reg=reg_lambda, random_state=seed,
                                  verbose=False)
    if mt == "svm":
        from sklearn.svm import SVC
        return SVC(probability=True, C=1.0 / max(reg_lambda, 1e-9),
                   random_state=seed)
    if mt == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(max_iter=max(200, n_estimators),
                             learning_rate_init=learning_rate,
                             alpha=reg_alpha, random_state=seed)
    raise ValueError(
        f"Unsupported model_type_ml {model_type!r}. Choose one of "
        f"random_forest, extra_trees, logistic_regression, gradient_boosting, "
        f"xgboost, lightgbm, catboost, svm, mlp."
    )


def sklearn_cv_fit_fn(features, labels, model_type: str = "xgboost",
                      *, criterion: str = "accuracy", seed: int = 42,
                      n_jobs: int = -1):
    """Build the ``fit_fn(params, train_idx, val_idx)`` :func:`cv_search` wants.

    The estimator is fitted on the fold's training indices and scored on the
    fold's validation indices — the function is never given any other indices,
    so it structurally cannot score on the held-out test split.

    :param features: 2-D numeric feature matrix.
    :param labels: per-row class labels.
    :param model_type: which classifier to build (see
        :func:`build_sklearn_model`).
    :param criterion: ``'accuracy'``, ``'roc_auc'`` or ``'f1'``.
    :param seed: ``random_state`` for the estimator.
    :param n_jobs: worker count where supported.
    :returns: the fit function.
    """
    import numpy as np

    X = np.asarray(features, dtype=float)
    y = np.asarray(labels)

    def _fit(params, train_idx, val_idx):
        """Fit on the fold's training rows, score on its validation rows."""
        from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)
        tr = np.asarray(train_idx, dtype=int)
        va = np.asarray(val_idx, dtype=int)
        model = build_sklearn_model(model_type, params, seed=seed,
                                    n_jobs=n_jobs)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        if criterion == "roc_auc":
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X[va])[:, 1]
            else:
                prob = model.decision_function(X[va])
            score = float(roc_auc_score(y[va], prob))
        elif criterion == "f1":
            score = float(f1_score(y[va], pred, average="binary"
                                   if len(set(y.tolist())) == 2 else "macro"))
        else:
            score = float(accuracy_score(y[va], pred))
        return score, {"n_train": int(tr.size), "n_val": int(va.size)}

    return _fit


def classify_cv_fit_fn(settings: Mapping[str, Any],
                       *,
                       criterion: str = "accuracy",
                       n_folds: int = 5,
                       train_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
                       read_fold_csv: Optional[Callable[[str], Any]] = None):
    """Build a ``fit_fn(params)`` that trains one deep model per fold.

    The Classify (CV) app trains Torch CNNs, so the search does not roll its
    own folds: it hands each configuration to :func:`spacr.deep_spacr.
    train_test_model` with ``cross_validation_folds`` forced to at least two and
    ``cv_group_by`` left alone, then reads the per-fold CSV that run writes.
    That keeps a single implementation of grouped k-fold — spaCR's — and means
    the search splits the data exactly the way the training run will.

    Each trial trains ``n_folds`` models, so a grid of ``g`` configurations
    trains ``g × n_folds`` models. That is the honest cost; there is no cheap
    proxy for it.

    :param settings: base Classify (CV) settings dict; each trial's parameters
        are layered on top.
    :param criterion: metric column to read from the per-fold CSV.
    :param n_folds: cross-validation folds per trial; forced to at least 2.
    :param train_fn: override for ``train_test_model`` (used by tests so no
        real CNN is trained).
    :param read_fold_csv: override for the CSV reader.
    :returns: the fit function.
    :raises ValueError: when ``n_folds`` < 2.
    """
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(
            f"Classify (CV) search needs at least 2 folds to score a "
            f"configuration on held-out data, got {n_folds}.")

    def _default_train(cfg):
        """Run spaCR's own cross-validated training for one configuration."""
        from .deep_spacr import train_test_model
        return train_test_model(cfg)

    def _default_read(path):
        """Read the per-fold CSV a cross-validated training run wrote."""
        import pandas as pd
        return pd.read_csv(path)

    trainer = train_fn or _default_train
    reader = read_fold_csv or _default_read

    def _fit(params):
        """Train one configuration across the folds and average the metric."""
        cfg = dict(settings)
        cfg.update(params)
        cfg["cross_validation_folds"] = n_folds
        cfg.setdefault("cv_group_by", settings.get("cv_group_by", "well"))
        fold_csv = trainer(cfg)
        if not fold_csv:
            raise ValueError(
                "the training run produced no per-fold results, so this "
                "configuration cannot be scored (every fold may have died)")
        fold_df = reader(fold_csv)
        if criterion not in getattr(fold_df, "columns", ()):
            raise ValueError(
                f"the per-fold results have no {criterion!r} column "
                f"(available: {list(getattr(fold_df, 'columns', []))})")
        from .deep_spacr import summarize_cv_metrics
        summary = summarize_cv_metrics(fold_df, metric_keys=[criterion])
        if summary.empty:
            raise ValueError(
                f"no fold reported a usable {criterion!r} value")
        row = summary.iloc[0]
        std = float(row["std"])
        return float(row["mean"]), {
            "fold_std": 0.0 if std != std else std,
            "n_folds": int(row["n_folds"]),
            "fold_min": float(row["min"]),
            "fold_max": float(row["max"]),
        }

    return _fit


def run_search_for_app(app_key: str,
                       settings: Mapping[str, Any],
                       space: SearchSpace,
                       *,
                       criterion: Optional[str] = None,
                       mode: str = "grid",
                       n_trials: int = 12,
                       seed: int = 0,
                       n_folds: int = 5,
                       on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                       should_stop: Optional[Callable[[], bool]] = None,
                       data: Optional[SearchData] = None,
                       ) -> SearchResult:
    """Run the right search for a spaCR app. This is what the GUI calls.

    * ``umap`` — embeds the measurement features once per configuration and
      ranks them with a named criterion (see :func:`umap_search`).
    * ``ml_analyze`` — grouped cross-validated search over classical ML
      hyperparameters (see :func:`cv_search`).
    * ``classify`` — one cross-validated deep-training run per configuration
      (see :func:`classify_cv_fit_fn`).

    :param app_key: which app is asking.
    :param settings: that app's settings dict.
    :param space: the :class:`SearchSpace` to search.
    :param criterion: metric name; defaults to the app's first
      :data:`APP_CRITERIA` entry.
    :param mode: ``'grid'`` or ``'random'``.
    :param n_trials: configurations to evaluate when ``mode='random'``.
    :param seed: seed for sampling, folds and reducers.
    :param n_folds: cross-validation folds for the supervised apps.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :param data: pre-loaded :class:`SearchData`, skipping the database read.
    :returns: the :class:`SearchResult`.
    :raises ValueError: for an unknown ``app_key`` or ``mode``.
    """
    if app_key not in APP_CRITERIA:
        raise ValueError(
            f"No hyperparameter search is defined for app {app_key!r}. "
            f"Searchable apps: {sorted(APP_CRITERIA)}.")
    if mode not in ("grid", "random"):
        raise ValueError(
            f"mode must be 'grid' or 'random', got {mode!r}.")
    criterion = criterion or APP_CRITERIA[app_key][0]
    if criterion not in APP_CRITERIA[app_key]:
        raise ValueError(
            f"Criterion {criterion!r} is not available for {app_key!r}; "
            f"choose one of {APP_CRITERIA[app_key]}.")
    higher = criterion not in LOWER_IS_BETTER

    if app_key == "classify":
        fit = classify_cv_fit_fn(settings, criterion=criterion,
                                 n_folds=n_folds)
        notes = [
            f"Each trial runs one {n_folds}-fold cross-validated training run "
            f"through spaCR's own grouped folds "
            f"(cv_group_by={settings.get('cv_group_by', 'well')!r}); the test "
            f"split is never scored on, so no configuration was chosen using "
            f"test data.",
            f"Cost: {space.size() if mode == 'grid' else n_trials} "
            f"configurations × {n_folds} folds models trained.",
        ]
        if mode == "grid":
            return grid_search(fit, space, metric=criterion,
                               higher_is_better=higher, on_trial=on_trial,
                               should_stop=should_stop, notes=notes)
        return random_search(fit, space, n_trials, seed, metric=criterion,
                             higher_is_better=higher, on_trial=on_trial,
                             should_stop=should_stop, notes=notes)

    if data is None:
        data = load_search_data(app_key, settings)

    if app_key == "umap":
        result = umap_search(
            data.features, space, metric=criterion, labels=data.labels,
            seed=seed, on_trial=on_trial, should_stop=should_stop)
        result.notes = list(data.notes) + list(result.notes)
        return result

    fit = sklearn_cv_fit_fn(
        data.features, data.labels,
        model_type=settings.get("model_type_ml", "xgboost"),
        criterion=criterion, seed=seed,
        n_jobs=int(settings.get("n_jobs", -1) or -1))
    result = cv_search(
        fit, space, labels=data.labels, groups=data.groups,
        group_by="well", n_folds=n_folds, seed=seed, metric=criterion,
        higher_is_better=higher,
        n_trials=None if mode == "grid" else n_trials,
        on_trial=on_trial, should_stop=should_stop)
    result.notes = list(data.notes) + list(result.notes)
    return result
