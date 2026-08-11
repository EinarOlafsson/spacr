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
import hashlib
import math
import os
import random
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .checkpoint import CheckpointStore, fingerprint, json_safe
from .cancellation import (
    PipelineCancelled,
    checkpoint as cancellation_checkpoint,
)

__all__ = [
    "SearchSpace",
    "Trial",
    "SearchResult",
    "grid_search",
    "random_search",
    "local_direction_search",
    "umap_search",
    "cv_search",
    "build_folds",
    "format_search",
    "umap_available",
    "umap_checkpoint_path",
    "SearchData",
    "load_search_data",
    "build_sklearn_model",
    "sklearn_cv_fit_fn",
    "classify_cv_fit_fn",
    "run_search_for_app",
    "APP_CRITERIA",
    "UMAP_CRITERIA",
    "UMAP_OBJECTIVES",
    "DEFAULT_UMAP_OBJECTIVE_WEIGHTS",
    "UMAP_MISSING_MESSAGE",
    "UMAP_NO_GROUND_TRUTH",
    "embedding_stability",
    "umap_objective_scores",
    "DEFAULT_SPACES",
    "ActivationSearchData",
    "activation_fit_fn",
    "activation_search",
    "load_activation_data",
    "ACTIVATION_CRITERIA",
    "ACTIVATION_NO_GROUND_TRUTH",
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
    "multi_objective": (
        "balances three independently reported objectives: preservation of "
        "neighbourhoods from feature space, repeat-to-repeat stability, and "
        "cluster structure. The weighted geometric mean guides the search, "
        "while the Pareto front exposes trade-offs that no single score can "
        "resolve."
    ),
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

#: Objectives used by the multi-objective UMAP mode. All are scaled to [0, 1]
#: and higher is better, making the Pareto comparison explicit and stable.
UMAP_OBJECTIVES: Dict[str, str] = {
    "neighborhood_preservation": (
        "geometric mean of trustworthiness and continuity; both invented and "
        "lost neighbours are penalized"
    ),
    "stability": (
        "mean shared-k-nearest-neighbour fraction between embeddings fitted "
        "with different reproducible seeds"
    ),
    "cluster_structure": (
        "positive silhouette structure, using supplied labels when available "
        "or the best reproducible 2–8 cluster K-means partition otherwise"
    ),
}

DEFAULT_UMAP_OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "neighborhood_preservation": 0.4,
    "stability": 0.3,
    "cluster_structure": 0.3,
}

#: The caveat attached to every Activation search result. Never suppressed.
ACTIVATION_NO_GROUND_TRUTH = (
    "Attribution has no ground truth: no measurement can tell you a saliency "
    "or CAM map is correct, because there is no correct map to compare it "
    "against. The criteria below measure three different, partly contradictory "
    "properties — whether removing the top-ranked pixels breaks the "
    "prediction, whether adding them alone restores it, and whether the peak "
    "lands on the object — and they routinely rank the same methods in "
    "different orders. Read the panel of maps; the table only chooses which "
    "ones to look at first."
)

#: What each Activation criterion rewards. Mirrors
#: :data:`spacr.attribution.CRITERION_CAVEATS`, which is the single source.
ACTIVATION_CRITERIA: Dict[str, str] = {
    "deletion_auc": (
        "area under the deletion curve — LOWER is better. Removing the pixels "
        "the map ranks highest should collapse the score immediately."
    ),
    "insertion_auc": (
        "area under the insertion curve — higher is better. The top-ranked "
        "pixels alone, on a blank background, should already recover the "
        "prediction."
    ),
    "pointing_game": (
        "fraction of images whose brightest attribution pixel falls inside the "
        "object mask — higher is better, and it says nothing about the rest of "
        "the map."
    ),
    "sanity_gap": (
        "1 - (rank correlation between the map from the trained model and the "
        "map from the same model with randomised weights) — higher is better. "
        "A method scoring near zero produces the same picture for a random "
        "model and is an edge detector, not an explanation."
    ),
}

#: Criteria each app's search can rank by, first entry being the default.
APP_CRITERIA: Dict[str, List[str]] = {
    "umap": [
        "trustworthiness", "continuity", "silhouette", "multi_objective",
    ],
    "classify": ["accuracy", "prauc", "loss"],
    "ml_analyze": ["accuracy", "roc_auc", "f1"],
    "activation": ["deletion_auc", "insertion_auc", "pointing_game",
                   "sanity_gap"],
}

#: Criteria where a smaller number is better.
LOWER_IS_BETTER = frozenset({"loss", "deletion_auc"})

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
    # One representative of each attribution family, because agreement within
    # a family is nearly worthless and disagreement across families is the
    # finding. Score-CAM and feature ablation are left out of the default
    # grid: both are an order of magnitude slower than the rest and a sweep
    # the user cancels tells them nothing.
    "activation": {
        "cam_type": ["gradcam", "gradcam_pp", "layercam", "saliency",
                     "integrated_gradients", "occlusion"],
        "smoothgrad_samples": [0, 8],
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
    :ivar objectives: objective name to direction mapping. When populated,
        :meth:`pareto_front` exposes non-dominated configurations.
    """

    trials: List[Trial] = field(default_factory=list)
    best: Optional[Trial] = None
    space: Optional[SearchSpace] = None
    metric: str = "score"
    notes: List[str] = field(default_factory=list)
    partial: bool = False
    higher_is_better: bool = True
    objectives: Dict[str, bool] = field(default_factory=dict)

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

    def pareto_front(self) -> List[Trial]:
        """Return non-dominated successful trials for declared objectives.

        A trial is dominated when another is at least as good on every
        objective and strictly better on one. The returned order follows the
        composite ranking so the GUI remains deterministic.
        """
        if not self.objectives:
            return []
        usable = []
        for trial in self.successful:
            values = {}
            for name in self.objectives:
                value = trial.extra_metrics.get(name)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    break
                if not math.isfinite(number):
                    break
                values[name] = number
            else:
                usable.append((trial, values))
        front = []
        for trial, values in usable:
            dominated = False
            for other, other_values in usable:
                if other is trial:
                    continue
                no_worse = []
                strictly_better = []
                for name, higher in self.objectives.items():
                    if higher:
                        no_worse.append(other_values[name] >= values[name])
                        strictly_better.append(
                            other_values[name] > values[name])
                    else:
                        no_worse.append(other_values[name] <= values[name])
                        strictly_better.append(
                            other_values[name] < values[name])
                if all(no_worse) and any(strictly_better):
                    dominated = True
                    break
            if not dominated:
                front.append(trial)
        indexes = {id(trial) for trial in front}
        return [trial for trial in self.ranked() if id(trial) in indexes]

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
        pareto_ids = {id(trial) for trial in self.pareto_front()}
        for rank, t in enumerate(self.ranked(), start=1):
            rows.append({
                "rank": rank, "index": t.index, "params": dict(t.params),
                "score": float(t.score), "metric": self.metric,
                "duration": t.duration, "error": None,
                "extra_metrics": dict(t.extra_metrics),
                "pareto": id(t) in pareto_ids,
            })
        for t in self.failed:
            rows.append({
                "rank": None, "index": t.index, "params": dict(t.params),
                "score": None, "metric": self.metric,
                "duration": t.duration, "error": t.error,
                "extra_metrics": dict(t.extra_metrics),
                "pareto": False,
            })
        return rows


# ---------------------------------------------------------------------------
# Persisted UMAP-search trials
# ---------------------------------------------------------------------------

def umap_checkpoint_path(settings: Mapping[str, Any]) -> Optional[str]:
    """Return the default UMAP-search checkpoint path for module settings.

    ``checkpoint_path`` wins when explicitly supplied. Otherwise the path is
    ``<project>/results/.spacr_checkpoints/umap_search.json``, with database
    and ``measurements/`` inputs normalised back to their project root.

    :param settings: Image UMAP module settings.
    :returns: absolute path, or None when no source/project can be inferred.
    """
    explicit = settings.get("checkpoint_path")
    if explicit:
        return os.path.abspath(os.path.expanduser(str(explicit)))
    source = settings.get("src")
    if isinstance(source, (list, tuple)):
        source = next((item for item in source if item), None)
    if not source:
        return None
    path = os.path.abspath(os.path.expanduser(str(source)))
    # A hand-built/test settings dict may carry a placeholder such as "/x".
    # Only explicit checkpoint_path is allowed to create a new project tree;
    # an inferred path must start from a source that actually exists.
    if not os.path.exists(path):
        return None
    if os.path.isfile(path) or path.lower().endswith((".db", ".sqlite")):
        path = os.path.dirname(path)
    if os.path.basename(path).lower() == "measurements":
        path = os.path.dirname(path)
    return os.path.join(
        path, "results", ".spacr_checkpoints", "umap_search.json")


def _array_fingerprint(value: Any) -> str:
    """Digest an array-like value without serialising it into giant JSON."""
    import numpy as np

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(array.shape)).encode("utf-8"))
    if array.dtype.hasobject:
        digest.update(fingerprint(array.tolist()).encode("ascii"))
    else:
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _trial_key(params: Mapping[str, Any]) -> str:
    """Stable id for one hyperparameter configuration."""
    return fingerprint(dict(params))


def _save_array_atomic(path: os.PathLike | str, array: Any) -> None:
    """Atomically persist one NumPy array artifact."""
    import numpy as np

    target = os.fspath(path)
    folder = os.path.dirname(target) or "."
    os.makedirs(folder, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class _UmapCheckpoint:
    """Trial/round adapter over :class:`spacr.checkpoint.CheckpointStore`."""

    def __init__(self, path: str, signature: Mapping[str, Any],
                 resume: bool, keep_embeddings: bool) -> None:
        self.store = CheckpointStore(
            path, workflow="umap_hyperparameter_search",
            signature=signature, boundary="trial", resume=resume)
        self.keep_embeddings = bool(keep_embeddings)

    @property
    def resumed(self) -> bool:
        """Whether an existing compatible checkpoint was loaded."""
        return self.store.resumed

    @property
    def state(self) -> Dict[str, Any]:
        """Adaptive-search state persisted after the last safe boundary."""
        return self.store.meta

    def load(self) -> Dict[str, Tuple[Trial, int]]:
        """Load complete trials keyed by configuration digest.

        A successful trial whose required embedding artifact is missing is
        omitted and therefore recomputed. Failed trials need no artifact.
        """
        import numpy as np

        loaded: Dict[str, Tuple[Trial, int]] = {}
        for key, raw in self.store.completed.items():
            if not isinstance(raw, Mapping):
                continue
            extra = dict(raw.get("extra_metrics") or {})
            artifact = raw.get("embedding_artifact")
            if artifact:
                artifact_path = self.store.path.parent / str(artifact)
                try:
                    extra["embedding"] = np.load(
                        artifact_path, allow_pickle=False)
                except (OSError, ValueError):
                    continue
            elif (self.keep_embeddings and raw.get("error") is None
                  and raw.get("score") is not None):
                continue
            trial = Trial(
                params=dict(raw.get("params") or {}),
                score=raw.get("score"),
                extra_metrics=extra,
                duration=float(raw.get("duration", 0.0) or 0.0),
                error=raw.get("error"),
                index=int(raw.get("index", -1) or 0),
            )
            loaded[str(key)] = (trial, int(raw.get("round", -1) or 0))
        return loaded

    def record(self, trial: Trial, *, round_index: int = -1,
               state: Optional[Mapping[str, Any]] = None) -> None:
        """Persist one completed trial and optional adaptive state."""
        extra = dict(trial.extra_metrics)
        embedding = extra.pop("embedding", None)
        payload: Dict[str, Any] = {
            "params": dict(trial.params),
            "score": trial.score,
            "extra_metrics": json_safe(extra),
            "duration": float(trial.duration),
            "error": trial.error,
            "index": int(trial.index),
            "round": int(round_index),
        }
        key = _trial_key(trial.params)
        if embedding is not None:
            artifact = self.store.artifact_path(key, ".npy")
            _save_array_atomic(artifact, embedding)
            payload["embedding_artifact"] = os.path.relpath(
                artifact, self.store.path.parent)
        self.store.mark(key, payload, meta=state)

    def update(self, state: Mapping[str, Any], *, status: str = "running") -> None:
        """Persist adaptive-round state."""
        self.store.update(meta=state, status=status)

    def finish(self, state: Optional[Mapping[str, Any]] = None) -> None:
        """Mark the search checkpoint complete."""
        self.store.finish(meta=state)


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
                prior_trials: Optional[Mapping[str, Trial]] = None,
                on_complete: Optional[Callable[[Trial], None]] = None,
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
    :param prior_trials: compatible completed trials keyed by parameter digest.
        They are replayed through ``on_trial`` and are not fitted again.
    :param on_complete: persistence callback after each newly completed trial.
    :returns: the :class:`SearchResult`.
    """
    result = SearchResult(space=space, metric=metric,
                          notes=list(notes or []),
                          higher_is_better=higher_is_better)
    total = len(param_sets)
    invoke = call if call is not None else (lambda fn, p: fn(p))
    prior = dict(prior_trials or {})

    for idx, params in enumerate(param_sets):
        cancellation_checkpoint()
        key = _trial_key(params)
        if key in prior:
            trial = prior[key]
            trial.index = idx
            result.trials.append(trial)
            if on_trial is not None:
                on_trial(trial, idx + 1, total)
            continue
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
        except PipelineCancelled:
            raise
        except Exception as exc:  # one bad configuration must not lose the sweep
            trial.error = f"{type(exc).__name__}: {exc}"
        trial.duration = time.perf_counter() - started
        result.trials.append(trial)
        if on_complete is not None:
            on_complete(trial)
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


#: The metrics umap-learn actually accepts. A typo in this field used to
#: fail deep inside the run -- after the embedding had started -- rather
#: than under the control that holds it, which is the difference between a
#: sentence and a traceback.
#:
#: Taken from umap.distances at import when it is importable, so this
#: cannot drift from the installed version; the literal is the fallback for
#: a checkout without umap-learn (the GUI must still build).
UMAP_METRICS: Tuple[str, ...] = (
    "euclidean", "manhattan", "chebyshev", "minkowski", "canberra",
    "braycurtis", "haversine", "mahalanobis", "wminkowski", "seuclidean",
    "cosine", "correlation", "hamming", "jaccard", "dice", "russellrao",
    "kulsinski", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule",
)


def umap_metrics() -> Tuple[str, ...]:
    """Every metric the INSTALLED umap-learn will accept.

    Falls back to :data:`UMAP_METRICS` when umap-learn is absent, because
    the settings panel has to build on a machine that cannot run UMAP --
    a user configuring a run on a laptop and executing it elsewhere is an
    ordinary thing to do.
    """
    try:
        from umap.distances import named_distances
    except Exception:
        return UMAP_METRICS
    names = tuple(sorted(named_distances))
    return names or UMAP_METRICS



#: Every UMAP parameter that changes the STRUCTURE of the embedding, with
#: the range UMAP itself requires. This is the space a Walk searches when
#: the user does not narrow it.
#:
#: Excluded deliberately: `random_state` and `n_epochs` change the result
#: without changing the structure being modelled, and searching them
#: rewards noise. `n_components` is here because it changes the embedding,
#: but note that anything above 2 cannot be plotted as a scatter.
UMAP_WALK_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "n_neighbors": {"step": 1.0, "minimum": 2.0, "integer": True},
    "min_dist": {"step": 0.05, "minimum": 0.0, "maximum": 1.0},
    "n_components": {"step": 1.0, "minimum": 1.0, "maximum": 10.0,
                     "integer": True},
    "metric": {"choices": None},  # filled from the installed umap-learn
    "spread": {"step": 0.25, "minimum": 0.1, "maximum": 10.0},
    "set_op_mix_ratio": {"step": 0.1, "minimum": 0.0, "maximum": 1.0},
    "local_connectivity": {"step": 1.0, "minimum": 1.0, "integer": True},
    "repulsion_strength": {"step": 0.25, "minimum": 0.0, "maximum": 10.0},
    "negative_sample_rate": {"step": 1.0, "minimum": 1.0, "maximum": 50.0,
                             "integer": True},
    "init": {"choices": ("spectral", "random", "pca")},
}


#: A Walk round that would need more fits than this stops being a
#: neighbourhood and becomes a grid search with extra steps. Ten axes at
#: resolution 2 is 1024 UMAP fits for ONE round; at a minute each that is
#: seventeen hours to take a single step. Past the cap the round falls back
#: to axis-at-a-time, which is linear in the number of axes.
MAX_WALK_CANDIDATES_PER_ROUND = 48


@dataclass
class WalkAxis:
    """One searchable direction in hyperparameter space.

    An axis is either NUMERIC -- it has a ``step`` and the walk moves along
    it by multiples of that step -- or CATEGORICAL, where ``choices`` lists
    the values and there is no direction to move in, only other values to
    try.

    :param name: the parameter name, as the fit function expects it.
    :param step: numeric axes only: how far one move goes.
    :param minimum: numeric axes only: inclusive lower clamp, or ``None``.
    :param maximum: numeric axes only: inclusive upper clamp, or ``None``.
    :param integer: numeric axes only: round candidates to whole numbers.
    :param choices: categorical axes only: the permitted values, in the
        order the walk should try them.
    :param resolution: how many values this axis contributes to one round,
        counting the centre. 2 is the classic ``±step`` pair with no centre;
        3 adds the centre back; 5 reaches two steps out. **The old 2-by-2
        search is exactly two numeric axes at resolution 2**, which is why
        that number is the default and not a special case in the code.
    """

    name: str
    step: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    integer: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    resolution: int = 2

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("A Walk axis needs a parameter name.")
        self.name = str(self.name)
        try:
            self.resolution = int(self.resolution)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Walk axis {self.name!r} needs a whole-number resolution."
            ) from exc
        if self.resolution < 2:
            raise ValueError(
                f"Walk axis {self.name!r} has resolution {self.resolution}; "
                "an axis that contributes fewer than two values is not being "
                "searched at all and should be left out of the space.")
        if self.choices is not None:
            self.choices = tuple(self.choices)
            if len(self.choices) < 2:
                raise ValueError(
                    f"Walk axis {self.name!r} is categorical with "
                    f"{len(self.choices)} choice(s); it needs at least two.")
            return
        if self.step is None:
            raise ValueError(
                f"Walk axis {self.name!r} needs either a step (numeric) or a "
                "list of choices (categorical).")
        try:
            self.step = float(self.step)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Walk axis {self.name!r} needs a numeric step.") from exc
        if self.step <= 0:
            raise ValueError(
                f"Walk axis {self.name!r} has step {self.step:g}; a step of "
                "zero or less never leaves the centre.")
        if self.minimum is not None:
            self.minimum = float(self.minimum)
        if self.maximum is not None:
            self.maximum = float(self.maximum)
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError(
                f"Walk axis {self.name!r} has minimum {self.minimum:g} above "
                f"maximum {self.maximum:g}.")

    @property
    def categorical(self) -> bool:
        """Whether this axis has choices rather than a step."""
        return self.choices is not None

    def clamp(self, value: Any) -> Any:
        """Bring ``value`` inside this axis's declared range.

        Categorical axes clamp by membership: a value that is not a choice
        becomes the first choice, because a walk that steps outside its own
        alphabet has nowhere to come back from.
        """
        if self.categorical:
            return value if value in self.choices else self.choices[0]
        number = float(value)
        if self.minimum is not None:
            number = max(self.minimum, number)
        if self.maximum is not None:
            number = min(self.maximum, number)
        if self.integer:
            number = float(int(round(number)))
            if self.minimum is not None:
                number = max(float(int(round(self.minimum))), number)
            if self.maximum is not None:
                number = min(float(int(round(self.maximum))), number)
            return int(number)
        return round(number, 12)

    def values_around(self, centre: Any) -> List[Any]:
        """The values this axis offers for one round, centred on ``centre``.

        Clamping happens here, so an axis at its boundary offers fewer
        distinct values rather than duplicates of the edge -- which is what
        lets the walk keep moving along the axes that still have room.
        """
        if self.categorical:
            current = self.clamp(centre)
            order = list(self.choices)
            start = order.index(current)
            rotated = order[start:] + order[:start]
            return _dedupe_preserving_order(rotated[:self.resolution])
        current = float(self.clamp(centre))
        half = self.resolution // 2
        multipliers = list(range(-half, half + 1))
        if self.resolution % 2 == 0:
            multipliers = [m for m in multipliers if m != 0]
        return _dedupe_preserving_order(
            [self.clamp(current + m * float(self.step)) for m in multipliers])


def _dedupe_preserving_order(values: Sequence[Any]) -> List[Any]:
    """``values`` without repeats, first occurrence winning."""
    out: List[Any] = []
    for value in values:
        if not any(existing == value and type(existing) is type(value)
                   for existing in out):
            out.append(value)
    return out


def walk_neighbourhood(axes: Sequence[WalkAxis],
                       centre: Mapping[str, Any],
                       *,
                       max_candidates: int = MAX_WALK_CANDIDATES_PER_ROUND,
                       ) -> Tuple[List[Dict[str, Any]], bool]:
    """The configurations one Walk round evaluates around ``centre``.

    The full neighbourhood is the Cartesian product of every axis's
    :meth:`WalkAxis.values_around`, minus the centre itself. With two
    numeric axes at resolution 2 that is the four diagonal corners the
    original 2-by-2 search used, which is the point: the old behaviour is
    this function's two-axis case and not a separate code path.

    The product is exponential in the number of axes, so when it exceeds
    ``max_candidates`` the round falls back to varying **one axis at a
    time** -- linear in the axis count, and still enough to choose a
    direction, at the cost of not seeing interactions between axes.

    :returns: ``(candidates, full_factorial)``. The flag is False when the
        fallback was used, and the caller is expected to say so in the
        result notes rather than quietly search less than it claimed.
    """
    axes = list(axes)
    if not axes:
        return [], True
    per_axis = {axis.name: axis.values_around(centre.get(axis.name))
                for axis in axes}
    centred = {axis.name: axis.clamp(centre.get(axis.name)) for axis in axes}

    product = 1
    for axis in axes:
        product *= max(1, len(per_axis[axis.name]))

    full_factorial = product <= max_candidates
    combos: List[Dict[str, Any]] = []
    if full_factorial:
        for combo in itertools.product(
                *(per_axis[axis.name] for axis in axes)):
            combos.append(dict(zip((axis.name for axis in axes), combo)))
    else:
        for axis in axes:
            for value in per_axis[axis.name]:
                moved = dict(centred)
                moved[axis.name] = value
                combos.append(moved)

    out: List[Dict[str, Any]] = []
    keys = set()
    centre_key = _trial_key(centred)
    for combo in combos:
        key = _trial_key(combo)
        if key == centre_key or key in keys:
            continue
        keys.add(key)
        out.append(combo)
    return out, full_factorial


def umap_walk_axes(start: Mapping[str, Any],
                   *,
                   parameters: Optional[Sequence[str]] = None,
                   steps: Optional[Mapping[str, float]] = None,
                   resolutions: Optional[Mapping[str, int]] = None,
                   n_neighbors_max: Optional[int] = None,
                   ) -> List[WalkAxis]:
    """Build Walk axes for UMAP from a starting configuration.

    ``parameters`` names which of UMAP's structural parameters take part.
    The default is the two the search has always used, so an existing call
    is unchanged; the panel passes the user's selection.

    Every axis carries the range UMAP itself requires -- ``n_neighbors``
    at least 2, ``min_dist`` within [0, 1], ``set_op_mix_ratio`` within
    [0, 1] -- because a walk is the one search that generates values that
    were never typed by anyone, and an out-of-range one fails inside the
    fit rather than at the edge.

    :param start: only its KEYS are read. Every searched name must be
        present or this raises; the starting *values* never reach the
        axes, so two different starting points build identical axes.
    :param parameters: which names take part. Empty or ``None`` falls back
        to the default pair, and a name outside
        :data:`UMAP_WALK_PARAMETERS` raises. A name listed twice is passed
        through here and only rejected later, by :func:`walk_search`.
    :param steps: per-axis step override. Discarded on the categorical
        axes (``init``, ``metric``), which move by choice rather than by
        step -- a step given for ``metric`` is even stored on the axis,
        but nothing ever reads it.
    :param resolutions: per-axis values per round, counting the centre;
        axes not named get 2. Below 2, or not a whole number, raises.
    :param n_neighbors_max: upper clamp for the ``n_neighbors`` axis only,
        and silently inert when that axis is not searched. Unvalidated
        here, unlike in :func:`local_direction_search`: below 2 it
        collides with the fixed minimum and raises from :class:`WalkAxis`,
        and a fractional cap rounds up (7.9 admits 8).
    """
    names = list(parameters) if parameters else ["n_neighbors", "min_dist"]
    steps = dict(steps or {})
    resolutions = dict(resolutions or {})
    unknown = [n for n in names if n not in UMAP_WALK_PARAMETERS]
    if unknown:
        raise ValueError(
            f"Not a searchable UMAP parameter: {sorted(unknown)}. "
            f"Searchable: {sorted(UMAP_WALK_PARAMETERS)}.")
    axes: List[WalkAxis] = []
    for name in names:
        spec = dict(UMAP_WALK_PARAMETERS[name])
        if name == "n_neighbors" and n_neighbors_max is not None:
            spec["maximum"] = float(n_neighbors_max)
        if name in steps and not spec.get("choices"):
            spec["step"] = steps[name]
        if name == "metric" and not spec.get("choices"):
            spec["choices"] = tuple(umap_metrics())
        axes.append(WalkAxis(
            name=name,
            step=spec.get("step"),
            minimum=spec.get("minimum"),
            maximum=spec.get("maximum"),
            integer=bool(spec.get("integer")),
            choices=spec.get("choices"),
            resolution=int(resolutions.get(name, 2)),
        ))
    missing = [axis.name for axis in axes if axis.name not in start]
    if missing:
        raise ValueError(
            "A Walk needs one starting value per searched parameter; "
            f"missing {sorted(missing)}.")
    return axes

def walk_search(fit_fn: Callable[[Dict[str, Any]], Any],
                start: Mapping[str, Any],
                axes: Sequence[WalkAxis],
                *,
                n_trials: Optional[int] = 100,
                min_improvement: float = 0.0,
                metric: str = "score",
                higher_is_better: bool = True,
                max_candidates_per_round: int = MAX_WALK_CANDIDATES_PER_ROUND,
                on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                notes: Optional[Sequence[str]] = None,
                checkpoint: Optional[_UmapCheckpoint] = None,
                ) -> SearchResult:
    """Walk N-dimensional hyperparameter space toward a better score.

    Each round scores the neighbourhood around the current centre -- see
    :func:`walk_neighbourhood` -- and moves to the best configuration found,
    stopping when a round fails to improve on the best score by more than
    ``min_improvement``. The starting point is a *centre*, not a trial: it
    is never fitted, because a walk asks "which way is better from here",
    and "here" is where the user already is.

    With two numeric axes at resolution 2 this is the 2-by-2 search that
    preceded it, corner for corner. It is not restricted to two: any
    parameter the fit function accepts can be an axis, and each axis
    carries its own step and resolution.

    :param start: one value per axis, plus any parameters held fixed --
        anything not named by an axis is passed to every fit unchanged.
    :param axes: the search space. Empty raises.
    :param n_trials: maximum complete rounds (100 when blank/None). NOT the
        number of fits, which is rounds times the neighbourhood size.
    :param max_candidates_per_round: past this, a round varies one axis at
        a time instead of taking the full product. Recorded in the notes.
    """
    axes = list(axes)
    if not axes:
        raise ValueError(
            "A Walk needs at least one axis; with none there is no direction "
            "to move in and the search would score the starting point for "
            "ever.")
    seen_names = set()
    for axis in axes:
        if axis.name in seen_names:
            raise ValueError(
                f"Walk axis {axis.name!r} is listed twice; one parameter is "
                "one direction.")
        seen_names.add(axis.name)
    missing = [axis.name for axis in axes if axis.name not in start]
    if missing:
        raise ValueError(
            "A Walk needs one starting value per searched parameter; "
            f"missing {sorted(missing)}.")
    try:
        max_rounds = 100 if n_trials in (None, "") else int(n_trials)
        improvement_floor = float(min_improvement)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A Walk requires a numeric round limit and minimum "
            "improvement.") from exc
    if max_rounds < 1:
        raise ValueError("A Walk needs a maximum of at least 1 round.")
    if improvement_floor < 0:
        raise ValueError(
            "The minimum improvement must be zero or greater.")

    frozen = {k: v for k, v in start.items() if k not in seen_names}
    centre = {axis.name: axis.clamp(start[axis.name]) for axis in axes}

    probe, full_factorial = walk_neighbourhood(
        axes, centre, max_candidates=max_candidates_per_round)
    per_round = max(1, len(probe))

    space = SearchSpace({
        **{name: [value] for name, value in centre.items()},
        **{key: [value] for key, value in frozen.items()},
    })
    axis_summary = ", ".join(
        f"{axis.name} (" + (
            f"{len(axis.choices)} choices" if axis.categorical
            else f"step {float(axis.step):g}")
        + f", resolution {axis.resolution})"
        for axis in axes)
    walk_notes = list(notes or []) + [
        f"Walk over {len(axes)} parameter(s): {axis_summary}. Each round "
        f"scores the neighbourhood around the current centre and moves to "
        f"the best configuration in it. The starting values define the "
        f"initial centre and are not fitted. The walk stops after at most "
        f"{max_rounds} rounds or when the best new score improves by no "
        f"more than {improvement_floor:g}."
    ]
    if not full_factorial:
        walk_notes.append(
            f"The full neighbourhood of these {len(axes)} axes exceeds the "
            f"{max_candidates_per_round}-fit round limit, so each round "
            f"varies ONE axis at a time ({per_round} fits per round instead "
            "of the product). The walk still chooses a direction; it does "
            "not see interactions between axes. Search fewer parameters, or "
            "raise the limit, to get the full neighbourhood back.")
    result = SearchResult(
        space=space, metric=metric, higher_is_better=higher_is_better,
        notes=walk_notes)

    loaded = checkpoint.load() if checkpoint is not None else {}
    state = checkpoint.state if checkpoint is not None else {}
    rounds_completed = int(state.get("rounds_completed", 0) or 0)

    def _persisted_state() -> Dict[str, Any]:
        # `centre_n`/`centre_d` are written alongside the general `centre`
        # so a checkpoint from this build stays readable by the 1.5.x
        # two-axis reader. Dropping them would make an in-flight search
        # unresumable by the version that started it.
        payload: Dict[str, Any] = {
            "rounds_completed": rounds_completed,
            "centre": dict(centre),
            "best_score": best_score,
        }
        if "n_neighbors" in centre:
            payload["centre_n"] = centre["n_neighbors"]
        if "min_dist" in centre:
            payload["centre_d"] = centre["min_dist"]
        return payload

    if checkpoint is not None and checkpoint.resumed:
        stored = state.get("centre")
        if isinstance(stored, Mapping):
            for axis in axes:
                if axis.name in stored:
                    centre[axis.name] = axis.clamp(stored[axis.name])
        else:
            # A checkpoint written before the walk went N-dimensional.
            legacy = {"n_neighbors": state.get("centre_n"),
                      "min_dist": state.get("centre_d")}
            for axis in axes:
                value = legacy.get(axis.name)
                if value is not None:
                    try:
                        centre[axis.name] = axis.clamp(value)
                    except (TypeError, ValueError):
                        pass
        result.notes.append(
            f"Resumed {len(loaded)} completed trial(s) at Walk round "
            f"{rounds_completed + 1} from {checkpoint.store.path}.")

    prior_items = [
        (trial, round_index)
        for trial, round_index in loaded.values()
        if round_index < rounds_completed
    ]
    for trial, _round in sorted(prior_items, key=lambda item: item[0].index):
        result.trials.append(trial)
        if on_trial is not None:
            on_trial(trial, len(result.trials), max_rounds * per_round)

    seen = {
        key for key, (_trial, round_index) in loaded.items()
        if round_index < rounds_completed
    }
    best_score: Optional[float] = None
    if state.get("best_score") is not None:
        try:
            best_score = float(state["best_score"])
        except (TypeError, ValueError):
            best_score = None
    stopped = False

    for _round_index in range(rounds_completed, max_rounds):
        cancellation_checkpoint()
        moves, _full = walk_neighbourhood(
            axes, centre, max_candidates=max_candidates_per_round)
        candidates = []
        candidate_keys = set()
        for move in moves:
            params = dict(frozen)
            params.update(move)
            key = _trial_key(params)
            if key not in seen and key not in candidate_keys:
                candidate_keys.add(key)
                candidates.append(params)
        if not candidates:
            result.notes.append(
                f"Round {_round_index + 1} had no configuration left to try "
                "-- every neighbour of the current centre has already been "
                "scored, or the axes are all at a boundary.")
            break
        round_trials: List[Trial] = []
        for params in candidates:
            cancellation_checkpoint()
            key = _trial_key(params)
            prior = loaded.get(key)
            if prior is not None and prior[1] == _round_index:
                trial = prior[0]
                trial.index = len(result.trials)
                result.trials.append(trial)
                round_trials.append(trial)
                seen.add(key)
                if on_trial is not None:
                    on_trial(trial, len(result.trials),
                             max_rounds * per_round)
                continue
            if should_stop is not None and should_stop():
                stopped = True
                break
            trial = Trial(params=dict(params), index=len(result.trials))
            started = time.perf_counter()
            try:
                trial.score, trial.extra_metrics = _normalise_outcome(
                    fit_fn(dict(params)))
                if trial.score is None:
                    trial.error = (
                        "fit function returned no score for this configuration")
            except PipelineCancelled:
                raise
            except Exception as exc:
                trial.error = f"{type(exc).__name__}: {exc}"
            trial.duration = time.perf_counter() - started
            result.trials.append(trial)
            round_trials.append(trial)
            seen.add(key)
            if checkpoint is not None:
                checkpoint.record(trial, round_index=_round_index,
                                  state=_persisted_state())
            if on_trial is not None:
                on_trial(trial, len(result.trials), max_rounds * per_round)
        if stopped:
            break
        rounds_completed = _round_index + 1

        successful = [trial for trial in round_trials if trial.ok]
        if not successful:
            if checkpoint is not None:
                checkpoint.update(_persisted_state())
            break
        round_best = successful[0]
        for trial in successful[1:]:
            better = (
                float(trial.score) > float(round_best.score)
                if higher_is_better
                else float(trial.score) < float(round_best.score))
            if better:
                round_best = trial
        gain = (
            float("inf") if best_score is None
            else (float(round_best.score) - best_score
                  if higher_is_better
                  else best_score - float(round_best.score))
        )
        if gain <= improvement_floor:
            result.notes.append(
                "The Walk stopped because the newest neighbourhood improved "
                f"the best score by {gain:.4g}, not more than the "
                f"{improvement_floor:g} stopping threshold.")
            if checkpoint is not None:
                checkpoint.update(_persisted_state())
            break
        best_score = float(round_best.score)
        for axis in axes:
            centre[axis.name] = axis.clamp(round_best.params[axis.name])
        if checkpoint is not None:
            checkpoint.update(_persisted_state())

    if stopped:
        result.partial = True
        result.notes.append(
            f"Search stopped after {len(result.trials)} configurations "
            f"({rounds_completed} completed rounds; maximum "
            f"{max_rounds} rounds).")
    elif rounds_completed == max_rounds:
        result.notes.append(
            f"The Walk reached the maximum of {max_rounds} rounds.")
    _select_best(result)
    _append_summary_notes(result, max_rounds * per_round)
    if checkpoint is not None:
        if stopped:
            checkpoint.update(_persisted_state(), status="partial")
        else:
            checkpoint.finish(_persisted_state())
    return result


def local_direction_search(
        fit_fn: Callable[[Dict[str, Any]], Any],
        start: Mapping[str, Any],
        *,
        n_trials: Optional[int] = 100,
        n_neighbors_step: int = 1,
        n_neighbors_max: Optional[int] = None,
        min_dist_step: float = 0.05,
        min_improvement: float = 0.0,
        metric: str = "score",
        higher_is_better: bool = True,
        on_trial: Optional[Callable[[Trial, int, int], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        notes: Optional[Sequence[str]] = None,
        checkpoint: Optional[_UmapCheckpoint] = None,
        ) -> SearchResult:
    """Walk UMAP's ``n_neighbors`` x ``min_dist`` plane by 2-by-2 rounds.

    The original two-axis Walk, kept as the name every existing caller
    uses. It is now :func:`walk_search` with two numeric axes at resolution
    2, which produces the same four diagonal corners per round; pass axes
    to that function directly to search more than these two parameters.

    :param fit_fn: called as ``fit_fn(params)`` with a clamped integer
        ``n_neighbors``, a float ``min_dist``, and every frozen key from
        ``start``. A call that raises is recorded as a failed trial; a
        round in which all four fail ends the walk with no winner.
    :param start: needs ``n_neighbors`` and ``min_dist`` -- missing or
        non-numeric raises. Out-of-range values are clamped rather than
        rejected (``n_neighbors`` up to at least 2, ``min_dist`` into
        ``[0, 1]``). Any other key is held fixed and passed to every fit
        unchanged.
    :param n_trials: maximum ROUNDS, not fits; each round costs up to
        four. Blank or ``None`` means 100. Zero, negative or non-numeric
        raises.
    :param n_neighbors_step: truncated by ``int()``, so 1.9 steps by 1 and
        anything below 1 becomes 0 and raises instead of stepping
        fractionally.
    :param n_neighbors_max: upper clamp on the ``n_neighbors`` axis.
        ``None`` leaves it unbounded and the walk climbs until the score
        stops improving. Below 2 raises.
    :param min_dist_step: step along ``min_dist``; must be strictly
        positive.
    :param min_improvement: a round must beat the running best by MORE
        than this to continue, so the default 0.0 still stops on an exact
        tie. Negative raises.
    :param metric: a label recorded on the result, nothing more. It does
        not choose a criterion -- whatever ``fit_fn`` returns is the
        score, whatever this names it.
    :param higher_is_better: the direction the walk climbs, and the
        comparison that picks the winner.
    :param on_trial: ``(trial, completed, total)`` after every trial,
        failures included. ``total`` is ``4 * n_trials``, an upper bound
        the walk normally undershoots because an already-scored
        configuration is never refitted.
    :param should_stop: polled before each new fit; True truncates the
        walk and marks the result partial.
    :param notes: caveats placed BEFORE the walk's own generated notes.
    :param checkpoint: resumes an interrupted walk -- completed trials are
        replayed without refitting and the centre, round count and best
        score are restored. ``None`` runs without persistence.
    """
    required = {"n_neighbors", "min_dist"}
    missing = required.difference(start)
    if missing:
        raise ValueError(
            "Local UMAP optimization needs one starting value for "
            f"n_neighbors and min_dist; missing {sorted(missing)}.")
    try:
        n_step = int(n_neighbors_step)
        d_step = float(min_dist_step)
        maximum_n = (
            None if n_neighbors_max is None else int(n_neighbors_max))
        int(start["n_neighbors"])
        float(start["min_dist"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Local UMAP optimization requires numeric n_neighbors, min_dist, "
            "and step sizes.") from exc
    if n_step < 1 or d_step <= 0:
        raise ValueError(
            "Local UMAP optimization steps must be positive and the minimum "
            "improvement must be zero or greater.")
    try:
        if float(min_improvement) < 0:
            raise ValueError(
                "Local UMAP optimization steps must be positive and the "
                "minimum improvement must be zero or greater.")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Local UMAP optimization requires numeric n_neighbors, min_dist, "
            "and step sizes.") from exc
    if maximum_n is not None and maximum_n < 2:
        raise ValueError("n_neighbors_max must be at least 2.")
    try:
        rounds = 100 if n_trials in (None, "") else int(n_trials)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Local UMAP optimization requires numeric n_neighbors, min_dist, "
            "and step sizes.") from exc
    if rounds < 1:
        raise ValueError(
            "Adaptive UMAP optimization needs n_trials/max rounds of at "
            "least 1.")
    axes = [
        WalkAxis("n_neighbors", step=float(n_step), minimum=2.0,
                 maximum=(None if maximum_n is None else float(maximum_n)),
                 integer=True, resolution=2),
        WalkAxis("min_dist", step=d_step, minimum=0.0, maximum=1.0,
                 resolution=2),
    ]
    return walk_search(
        fit_fn, start, axes,
        n_trials=rounds, min_improvement=min_improvement, metric=metric,
        higher_is_better=higher_is_better, on_trial=on_trial,
        should_stop=should_stop, notes=notes, checkpoint=checkpoint)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def umap_available() -> Tuple[bool, str]:
    """Whether umap-learn can be imported.

    :returns: ``(True, "")`` when available, otherwise ``(False, message)``
        carrying :data:`UMAP_MISSING_MESSAGE`.
    """
    # Through spacr.utils, never a bare `import umap`: umap's package
    # __init__ imports umap.parametric_umap -> tensorflow, and TF is not
    # a spaCR dependency. The lazy wrapper blocks it for that import.
    from .utils import umap, OptionalDependencyCompatibilityError
    try:
        umap.UMAP  # noqa: B018 - forces the deferred import
    except OptionalDependencyCompatibilityError as exc:
        return False, str(exc)
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
    from .utils import umap  # never a bare `import umap` — see umap_available
    kwargs = dict(params)
    kwargs.setdefault("n_components", 2)
    kwargs.setdefault("random_state", seed)
    reducer = umap.UMAP(**kwargs)
    # umap-learn intentionally disables parallel optimisation when a
    # random_state is supplied. That is expected for a reproducible search,
    # but it emits the same warning for every trial and buries useful output.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state.*",
            category=UserWarning,
        )
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


def embedding_stability(
    embeddings: Sequence[Any],
    *,
    neighbourhood_k: int = 15,
) -> float:
    """Measure repeat-to-repeat preservation of embedding neighbours.

    Rotation, reflection and axis scaling do not affect this measure: for
    every pair of embeddings it finds each sample's k nearest neighbours and
    averages the fraction shared by both fits.

    :param embeddings: two or more aligned ``(n_samples, n_components)``
        embeddings of the same rows.
    :param neighbourhood_k: number of neighbours compared per row.
    :returns: mean shared-neighbour fraction in ``[0, 1]``.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    arrays = [np.asarray(value, dtype=float) for value in embeddings]
    if len(arrays) < 2:
        raise ValueError("Embedding stability requires at least two repeats.")
    shape = arrays[0].shape
    if len(shape) != 2 or shape[0] < 3:
        raise ValueError(
            "Embedding stability needs 2-D embeddings with at least 3 rows.")
    if any(array.shape != shape for array in arrays):
        raise ValueError(
            "Every repeated embedding must have the same sample shape.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Repeated embeddings contain NaN or infinite values.")
    k = max(1, min(int(neighbourhood_k), shape[0] - 1))
    neighbourhoods = []
    for array in arrays:
        # Query k+1 because a row is its own nearest point, then remove it
        # explicitly rather than relying on sklearn's query-mode distinction.
        raw = NearestNeighbors(n_neighbors=k + 1).fit(array).kneighbors(
            array, return_distance=False,
        )
        cleaned = np.asarray([
            [int(value) for value in row if int(value) != int(index)][:k]
            for index, row in enumerate(raw)
        ], dtype=int)
        if cleaned.shape != (shape[0], k):
            raise RuntimeError(
                "Could not construct a complete nearest-neighbour graph for "
                "stability scoring.")
        neighbourhoods.append(cleaned)
    pair_scores = []
    for left_index in range(len(neighbourhoods) - 1):
        left = neighbourhoods[left_index]
        for right in neighbourhoods[left_index + 1:]:
            per_row = [
                len(set(left[row]).intersection(right[row])) / float(k)
                for row in range(shape[0])
            ]
            pair_scores.append(float(np.mean(per_row)))
    return float(np.mean(pair_scores))


def _cluster_structure(
    embedding: Any,
    labels: Any,
    *,
    seed: int,
) -> Tuple[float, float, str, int]:
    """Return normalized/raw silhouette and the partition provenance."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    values = np.asarray(embedding, dtype=float)
    n_samples = len(values)
    supplied = None if labels is None else np.asarray(labels)
    if (
        supplied is not None
        and supplied.shape[0] == n_samples
        and 2 <= len(np.unique(supplied)) < n_samples
    ):
        raw = float(silhouette_score(values, supplied))
        return max(0.0, min(1.0, raw)), raw, "supplied_labels", int(
            len(np.unique(supplied))
        )

    if n_samples < 4:
        raise ValueError(
            "Unsupervised cluster-structure scoring needs at least 4 rows.")
    maximum = min(8, n_samples - 1)
    best_raw = -1.0
    best_k = 0
    for n_clusters in range(2, maximum + 1):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Number of distinct clusters .*",
                category=Warning,
            )
            partition = KMeans(
                n_clusters=n_clusters,
                random_state=int(seed),
                n_init=10,
            ).fit_predict(values)
        if not 2 <= len(np.unique(partition)) < n_samples:
            continue
        raw = float(silhouette_score(values, partition))
        if raw > best_raw:
            best_raw = raw
            best_k = n_clusters
    if best_k == 0:
        return 0.0, 0.0, "no_resolved_clusters", 1
    return (
        max(0.0, min(1.0, best_raw)),
        best_raw,
        "discovered_kmeans",
        best_k,
    )


def _objective_weights(
    weights: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    """Validate and normalize multi-objective UMAP weights."""
    provided = dict(DEFAULT_UMAP_OBJECTIVE_WEIGHTS)
    if weights is not None:
        unknown = set(weights).difference(UMAP_OBJECTIVES)
        if unknown:
            raise ValueError(
                f"Unknown UMAP objective weight(s): {sorted(unknown)}. "
                f"Choose from {sorted(UMAP_OBJECTIVES)}.")
        provided.update(weights)
    normalized = {}
    for name in UMAP_OBJECTIVES:
        try:
            value = float(provided[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"UMAP objective weight {name!r} must be numeric.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"UMAP objective weight {name!r} must be finite and "
                "zero or greater.")
        normalized[name] = value
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("At least one UMAP objective weight must be positive.")
    return {name: value / total for name, value in normalized.items()}


def umap_objective_scores(
    features: Any,
    embeddings: Sequence[Any],
    *,
    labels: Any = None,
    neighbourhood_k: int = 15,
    weights: Optional[Mapping[str, Any]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Score neighborhood preservation, stability and cluster structure.

    The returned ``multi_objective`` value is a weighted geometric mean used
    to guide grid/adaptive search. The individual objective values remain the
    primary result and define :meth:`SearchResult.pareto_front`.

    :param features: read only for trustworthiness and continuity, hence
        only for ``neighborhood_preservation``. Stability and cluster
        structure come from the embeddings alone and do not move with it.
    :param embeddings: two or more fits of the same rows; fewer than two
        raises, because stability is a repeat-to-repeat measure. How they
        were produced is the caller's business -- only their count and
        geometry are used here.
    :param labels: optional, and consumed in two places: the ``silhouette``
        entry (present only when *every* repeat could compute it) and the
        cluster-structure partition. Labels whose length does not match the
        rows, or carrying fewer than two classes, are silently ignored and
        K-means discovery runs instead, so read
        ``cluster_structure_method`` rather than assuming they were used.
    :param neighbourhood_k: one value, two different caps. Trustworthiness
        and continuity clamp it to ``(n_samples - 1) // 2`` and report that
        clamped number back as ``neighbourhood_k``; stability clamps only
        to ``n_samples - 1``. A k near the sample count therefore drives
        stability to a meaningless 1.0 while the reported k still looks
        reasonable.
    :param weights: merged over :data:`DEFAULT_UMAP_OBJECTIVE_WEIGHTS` and
        then renormalized, so naming one objective does not zero the
        others -- ``{'stability': 1.0}`` ends up near 0.59, not 1.0.
        Unknown, negative, non-finite, non-numeric or all-zero raises.
    :param seed: reaches only the K-means discovery path, offset by the
        repeat index so the repeats are deliberately not identical. It has
        no effect at all when usable ``labels`` are supplied.
    :returns: the three objectives plus ``multi_objective``, the component
        ``trustworthiness``/``continuity``, the normalized weights, and the
        provenance fields ``cluster_structure_method`` and
        ``cluster_counts``.
    """
    if len(embeddings) < 2:
        raise ValueError(
            "Multi-objective UMAP search requires at least two stability "
            "repeats.")
    repeat_scores = [
        _umap_scores(features, embedding, labels, neighbourhood_k)
        for embedding in embeddings
    ]
    trust = statistics.fmean(
        score["trustworthiness"] for score in repeat_scores)
    continuity = statistics.fmean(
        score["continuity"] for score in repeat_scores)
    neighborhood = math.sqrt(max(0.0, trust) * max(0.0, continuity))
    stability = embedding_stability(
        embeddings, neighbourhood_k=neighbourhood_k,
    )
    structures = [
        _cluster_structure(embedding, labels, seed=seed + index)
        for index, embedding in enumerate(embeddings)
    ]
    structure = statistics.fmean(value[0] for value in structures)
    raw_structure = statistics.fmean(value[1] for value in structures)
    methods = sorted({value[2] for value in structures})
    cluster_counts = [value[3] for value in structures]
    normalized_weights = _objective_weights(weights)
    objectives = {
        "neighborhood_preservation": float(neighborhood),
        "stability": float(stability),
        "cluster_structure": float(structure),
    }
    # A geometric mean prevents one excellent property from fully hiding a
    # collapsed objective, while a tiny floor keeps the result finite.
    composite = math.exp(sum(
        normalized_weights[name] * math.log(max(1e-12, objectives[name]))
        for name in UMAP_OBJECTIVES
    ))
    result = {
        **objectives,
        "multi_objective": float(composite),
        "trustworthiness": float(trust),
        "continuity": float(continuity),
        "cluster_structure_raw_silhouette": float(raw_structure),
        "cluster_structure_method": "+".join(methods),
        "cluster_counts": cluster_counts,
        "stability_repeats": len(embeddings),
        "objective_weights": normalized_weights,
        "neighbourhood_k": repeat_scores[0]["neighbourhood_k"],
    }
    if all("silhouette" in score for score in repeat_scores):
        result["silhouette"] = float(statistics.fmean(
            score["silhouette"] for score in repeat_scores
        ))
    return result


def umap_search(features,
                space: SearchSpace,
                *,
                metric: str = "trustworthiness",
                labels=None,
                seed: int = 0,
                neighbourhood_k: int = 15,
                adaptive: bool = False,
                n_trials: Optional[int] = 100,
                n_neighbors_step: int = 1,
                min_dist_step: float = 0.05,
                min_improvement: float = 0.0,
                walk_parameters: Optional[Sequence[str]] = None,
                walk_resolutions: Optional[Mapping[str, int]] = None,
                walk_steps: Optional[Mapping[str, float]] = None,
                stability_repeats: int = 3,
                objective_weights: Optional[Mapping[str, Any]] = None,
                embed_fn: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
                keep_embeddings: bool = True,
                on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                should_stop: Optional[Callable[[], bool]] = None,
                checkpoint_path: Optional[str] = None,
                resume: bool = False,
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
    :param adaptive: run a Walk -- iterative local optimization from one
        starting point -- instead of a grid. See :func:`walk_search`.
    :param walk_parameters: which UMAP parameters the Walk searches.
        Defaults to n_neighbors and min_dist, the two it has always
        used; any subset of :data:`UMAP_WALK_PARAMETERS` is accepted.
    :param walk_resolutions: per-axis grid resolution, ``{name: n}``.
        2 is the classic +/-step pair. Missing axes get 2.
    :param walk_steps: per-axis step override, ``{name: size}``.
    :param n_trials: maximum complete Walk rounds; blank or
        None means 100.
    :param n_neighbors_step: local step along the n_neighbors axis.
    :param min_dist_step: local step along the min_dist axis.
    :param min_improvement: score gain required to continue after a round.
    :param stability_repeats: reproducible UMAP fits per configuration when
        ``metric='multi_objective'``; must be at least 2.
    :param objective_weights: optional weights for
        ``neighborhood_preservation``, ``stability`` and
        ``cluster_structure``. Values are normalized to sum to one.
    :param embed_fn: ``embed_fn(features, params) -> embedding`` override; when
        omitted, umap-learn is used.
    :param keep_embeddings: store each trial's embedding in its extra metrics.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :param checkpoint_path: optional atomic checkpoint JSON. Embeddings are
        stored as adjacent ``.npy`` artifacts after each completed trial.
    :param resume: load a compatible checkpoint. Input features, labels,
        search space, criterion, seed and material search settings must match.
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
    try:
        stability_repeats = int(stability_repeats)
    except (TypeError, ValueError) as exc:
        raise ValueError("stability_repeats must be a whole number.") from exc
    normalized_objective_weights = _objective_weights(objective_weights)
    if metric == "multi_objective" and stability_repeats < 2:
        raise ValueError(
            "Multi-objective UMAP search needs stability_repeats >= 2.")

    try:
        n_samples = int(features.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        try:
            n_samples = len(features)
        except TypeError as exc:
            raise ValueError(
                "UMAP features must be a 2-D array-like object.") from exc
    if n_samples < 3:
        raise ValueError(
            "UMAP hyperparameter search needs at least 3 rows after filtering; "
            f"only {n_samples} remain.")

    # umap-learn otherwise silently truncates every oversized n_neighbors
    # value to n_samples - 1. Apart from filling the terminal with warnings,
    # that can make several nominally different trials evaluate the exact same
    # embedding. Bound and de-duplicate the search before any reducer is fit so
    # the reported parameters are the parameters that were actually evaluated.
    maximum_neighbors = n_samples - 1
    bounded_params = dict(space.params)
    neighbor_note = ""
    if "n_neighbors" in bounded_params:
        requested_neighbors = list(bounded_params["n_neighbors"])
        effective_neighbors: List[int] = []
        for raw_value in requested_neighbors:
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "UMAP n_neighbors values must be whole numbers; "
                    f"got {raw_value!r}.") from exc
            value = max(2, min(maximum_neighbors, value))
            if value not in effective_neighbors:
                effective_neighbors.append(value)
        if effective_neighbors != requested_neighbors:
            neighbor_note = (
                f"n_neighbors was limited to 2…{maximum_neighbors} for the "
                f"{n_samples} available rows; duplicate effective values were "
                "evaluated only once.")
        bounded_params["n_neighbors"] = effective_neighbors
        space = SearchSpace(bounded_params)
    implicit_neighbors = min(15, maximum_neighbors)

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
    if metric == "multi_objective":
        weights_text = ", ".join(
            f"{name}={normalized_objective_weights[name]:.3g}"
            for name in UMAP_OBJECTIVES
        )
        notes.extend([
            "Multi-objective mode reports a Pareto front: every configuration "
            "on it improves at least one objective only by trading off another.",
            f"Composite search direction uses normalized weights "
            f"({weights_text}) and {stability_repeats} reproducible embedding "
            "repeats per configuration.",
        ])
    if neighbor_note:
        notes.append(neighbor_note)
    elif "n_neighbors" not in space.params and implicit_neighbors != 15:
        notes.append(
            f"UMAP's default n_neighbors was limited from 15 to "
            f"{implicit_neighbors} for the {n_samples} available rows.")

    checkpoint = None
    if checkpoint_path:
        embed_identity = {
            "module": getattr(embed_fn, "__module__", ""),
            "name": getattr(embed_fn, "__qualname__",
                            getattr(embed_fn, "__name__", type(embed_fn).__name__)),
        }
        checkpoint_signature = {
            "features": _array_fingerprint(features),
            "labels": (
                None if labels is None else _array_fingerprint(labels)),
            "space": {
                name: list(space.params[name]) for name in space.names
            },
            "metric": metric,
            "seed": int(seed),
            "neighbourhood_k": int(neighbourhood_k),
            "adaptive": bool(adaptive),
            "n_neighbors_step": int(n_neighbors_step),
            "min_dist_step": float(min_dist_step),
            "min_improvement": float(min_improvement),
            "embed_fn": embed_identity,
        }
        if metric == "multi_objective":
            checkpoint_signature["multi_objective"] = {
                "stability_repeats": int(stability_repeats),
                "objective_weights": normalized_objective_weights,
            }
        checkpoint = _UmapCheckpoint(
            os.path.abspath(os.path.expanduser(str(checkpoint_path))),
            checkpoint_signature,
            resume=bool(resume),
            keep_embeddings=keep_embeddings,
        )

    def _fit(params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Embed one configuration and score it with every criterion."""
        fit_params = dict(params)
        # A search may vary only min_dist/metric. Keep UMAP's implicit default
        # safe for a small dataset too, without adding an unsearched table
        # column to Trial.params.
        fit_params.setdefault("n_neighbors", implicit_neighbors)
        repeat_count = (
            stability_repeats if metric == "multi_objective" else 1
        )
        embeddings = []
        for repeat in range(repeat_count):
            repeat_params = dict(fit_params)
            if metric == "multi_objective":
                repeat_params["random_state"] = int(seed) + repeat
            embeddings.append(embed_fn(features, repeat_params))
        embedding = embeddings[0]
        if metric == "multi_objective":
            scores = umap_objective_scores(
                features,
                embeddings,
                labels=labels,
                neighbourhood_k=neighbourhood_k,
                weights=normalized_objective_weights,
                seed=seed,
            )
        else:
            scores = _umap_scores(
                features, embedding, labels, neighbourhood_k,
            )
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

    if adaptive:
        starts = space.grid()
        if len(starts) != 1:
            raise ValueError(
                "A Walk needs exactly one starting value for every "
                "parameter -- it is a path through the space from one point, "
                "not a grid over it. Enter a single value in each field.")
        if walk_parameters:
            # The user chose the space. Anything they named that the
            # starting point does not carry is an error there rather than
            # here, so `umap_walk_axes` is given the whole start dict.
            axes = umap_walk_axes(
                starts[0], parameters=walk_parameters,
                steps=walk_steps, resolutions=walk_resolutions,
                n_neighbors_max=maximum_neighbors)
        else:
            # The two-axis default, expressed as axes rather than as a
            # separate code path, so there is one Walk and not two.
            axes = [
                WalkAxis("n_neighbors", step=float(n_neighbors_step),
                         minimum=2.0, integer=True,
                         maximum=(None if maximum_neighbors is None
                                  else float(maximum_neighbors)),
                         resolution=int((walk_resolutions or {})
                                        .get("n_neighbors", 2))),
                WalkAxis("min_dist", step=float(min_dist_step), minimum=0.0,
                         maximum=1.0,
                         resolution=int((walk_resolutions or {})
                                        .get("min_dist", 2))),
            ]
        result = walk_search(
            _fit, starts[0], axes, n_trials=n_trials,
            min_improvement=min_improvement, metric=metric,
            higher_is_better=True, on_trial=on_trial,
            should_stop=should_stop, notes=notes, checkpoint=checkpoint)
        if metric == "multi_objective":
            result.objectives = {
                name: True for name in UMAP_OBJECTIVES
            }
        return result

    loaded = checkpoint.load() if checkpoint is not None else {}
    prior_trials = {
        key: trial for key, (trial, _round) in loaded.items()
    }
    if checkpoint is not None and checkpoint.resumed:
        notes.append(
            f"Resumed {len(prior_trials)} completed trial(s) from "
            f"{checkpoint.store.path}.")
    result = _run_trials(
        _fit, space.grid(), space, metric,
        higher_is_better=True, on_trial=on_trial,
        should_stop=should_stop, notes=notes,
        prior_trials=prior_trials,
        on_complete=(
            None if checkpoint is None
            else lambda trial: checkpoint.record(trial, round_index=-1)))
    if checkpoint is not None:
        if result.partial:
            checkpoint.update(
                {"n_trials_completed": len(result.trials)}, status="partial")
        else:
            checkpoint.finish(
                {"n_trials_completed": len(result.trials)})
    if metric == "multi_objective":
        result.objectives = {name: True for name in UMAP_OBJECTIVES}
    return result


# ---------------------------------------------------------------------------
# Activation — sweeping what a trained model is said to attend to
# ---------------------------------------------------------------------------

@dataclass
class ActivationSearchData:
    """The model and images one Activation sweep is scored on.

    :ivar model: the trained classifier, already on the right device and in
        eval mode.
    :ivar images: list of per-image tensors ``(C, H, W)``.
    :ivar masks: optional per-image boolean object masks, same spatial shape,
        enabling the pointing game. None when spaCR could not find them.
    :ivar filenames: per-image provenance for the panel labels.
    :ivar model_type: architecture name, used to make errors readable.
    :ivar notes: provenance and warnings the caller must surface.
    """

    model: Any = None
    images: List[Any] = field(default_factory=list)
    masks: Optional[List[Any]] = None
    filenames: List[str] = field(default_factory=list)
    model_type: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _activation_params(params: Mapping[str, Any]) -> Tuple[str, Dict[str, Any],
                                                           int, float]:
    """Split one trial's configuration into method, kwargs and SmoothGrad knobs.

    spaCR's Activation settings name the method ``cam_type`` and carry the
    legacy values ``'saliency_image'`` / ``'saliency_channel'``, which both mean
    the plain input-gradient saliency map; they are folded onto ``'saliency'``
    so an existing settings CSV sweeps without editing.

    :param params: one trial's parameters.
    :returns: ``(method, method_kwargs, smoothgrad_samples, smoothgrad_sigma)``.
    """
    p = dict(params)
    # Both spellings are removed whichever one supplied the value, so neither
    # can leak downstream into the attribution call as a stray keyword.
    named = p.pop("method", None)
    method = str(p.pop("cam_type", None) or named or "gradcam")
    if method in ("saliency_image", "saliency_channel"):
        method = "saliency"
    kw: Dict[str, Any] = {}
    if p.get("target_layer") not in (None, "", "None"):
        kw["layer"] = str(p["target_layer"])
    p.pop("target_layer", None)
    for src, dst in (("ig_steps", "n_steps"), ("ig_baseline", "baseline"),
                     ("occlusion_window", "window"),
                     ("occlusion_stride", "stride")):
        if p.get(src) is not None:
            kw[dst] = p[src]
        p.pop(src, None)
    n_samples = int(p.pop("smoothgrad_samples", 0) or 0)
    sigma = float(p.pop("smoothgrad_sigma", 0.15) or 0.15)
    kw.update(p)
    return method, kw, n_samples, sigma


def activation_fit_fn(data: ActivationSearchData,
                      *,
                      criterion: str = "deletion_auc",
                      n_steps: int = 12,
                      baseline: str = "blur",
                      sanity_threshold: float = 0.5,
                      run_sanity_check: bool = True,
                      keep_maps: bool = True,
                      attribute_fn: Optional[Callable[..., Any]] = None,
                      ) -> Callable[[Dict[str, Any]], Any]:
    """Build the ``fit_fn(params)`` an Activation sweep evaluates.

    Every trial attributes each image once and then measures that map four
    ways — deletion AUC, insertion AUC, the pointing game (when masks exist)
    and the randomisation sanity check — so the table can be re-ranked by any
    criterion without re-running the sweep. **All four are reported for every
    trial precisely because they disagree**; a sweep that reported only the one
    it ranked by would hide the disagreement, which is the informative part.

    :param data: the model and images to score on.
    :param criterion: which of :data:`ACTIVATION_CRITERIA` drives the ranking.
    :param n_steps: perturbation steps in the deletion / insertion curves.
    :param baseline: what removed pixels become — ``'blur'`` (least
        out-of-distribution), ``'zero'``, ``'mean'`` or ``'uniform'``.
    :param sanity_threshold: rank correlation below which a method passes the
        randomisation check.
    :param run_sanity_check: run the check on the first image only. It costs one
        extra attribution per parameterised layer, so it is the expensive part
        of a trial; turning it off removes the most valuable number here.
    :param keep_maps: keep each trial's first map so the panel can draw it.
    :param attribute_fn: override for the attribution call, used by tests.
    :returns: the fit function.
    :raises ValueError: for an unknown criterion or an empty image set.
    """
    if criterion not in ACTIVATION_CRITERIA:
        raise ValueError(
            f"Unknown Activation criterion {criterion!r}. Choose one of "
            f"{sorted(ACTIVATION_CRITERIA)} — each measures a different "
            f"property and they routinely disagree.")
    if not data.images:
        raise ValueError(
            "The Activation search has no images to score on. Point 'dataset' "
            "at a crop tar (or 'src' at an experiment with merged/*.npy) so "
            "there is something to attribute.")

    def _attribute(params: Mapping[str, Any], image):
        """Attribute one image with one trial's configuration."""
        if attribute_fn is not None:
            return attribute_fn(data.model, image, dict(params))
        from .attribution import attribute, smoothgrad
        method, kw, n_samples, sigma = _activation_params(params)
        if n_samples > 1:
            return smoothgrad(data.model, image, method, n_samples=n_samples,
                              sigma=sigma, model_type=data.model_type, **kw)
        return attribute(data.model, image, method,
                         model_type=data.model_type, **kw)

    def _fit(params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Score one configuration on every image, reporting every criterion."""
        from .attribution import (deletion_curve, insertion_curve,
                                  pointing_game_rate)

        maps = [_attribute(params, img) for img in data.images]
        deletions: List[float] = []
        insertions: List[float] = []
        flat = 0
        for att, img in zip(maps, data.images):
            deletions.append(deletion_curve(data.model, img, att,
                                            n_steps=n_steps,
                                            baseline=baseline).auc)
            insertions.append(insertion_curve(data.model, img, att,
                                              n_steps=n_steps,
                                              baseline=baseline).auc)
            flat += int(getattr(att, "is_flat", lambda: False)())

        scores: Dict[str, Any] = {
            "deletion_auc": statistics.fmean(deletions),
            "insertion_auc": statistics.fmean(insertions),
            "n_images": len(maps),
            "n_flat_maps": flat,
        }
        # The image-to-image spread of the ranked criterion is this search's
        # noise yardstick, the way fold-to-fold spread is the classifiers'. A
        # configuration that wins by less than the variation between images has
        # not won.
        per_image = {"deletion_auc": deletions,
                     "insertion_auc": insertions}.get(criterion, [])
        scores["fold_std"] = (statistics.pstdev(per_image)
                              if len(per_image) > 1 else 0.0)
        scores["deletion_std"] = (statistics.pstdev(deletions)
                                  if len(deletions) > 1 else 0.0)
        scores["insertion_std"] = (statistics.pstdev(insertions)
                                   if len(insertions) > 1 else 0.0)

        if data.masks:
            pg = pointing_game_rate([m.map for m in maps], data.masks)
            scores["pointing_game"] = float(pg["rate"])
            scores["pointing_hits"] = pg["hits"]
            scores["pointing_scored"] = pg["n"]

        if run_sanity_check:
            from .attribution import randomization_sanity_check
            method, kw, _n, _s = _activation_params(params)
            check = randomization_sanity_check(
                data.model, data.images[0], method,
                model_type=data.model_type, threshold=sanity_threshold,
                **{k: v for k, v in kw.items()
                   if k in ("layer", "n_steps", "baseline", "window",
                            "stride")})
            scores["sanity_gap"] = check.gap
            scores["sanity_similarity"] = check.final_similarity
            scores["sanity_passed"] = check.passed
            scores["sanity_verdict"] = check.verdict()

        if criterion not in scores:
            raise ValueError(
                f"criterion {criterion!r} could not be computed for this "
                f"trial: "
                + ("no object masks were available, so the pointing game has "
                   "no answer key. Rank by deletion_auc or insertion_auc, or "
                   "point 'src' at an experiment whose merged/*.npy files "
                   "carry the mask planes."
                   if criterion == "pointing_game" else
                   "the randomisation sanity check was disabled for this "
                   "sweep."))
        if keep_maps and maps:
            scores["attribution"] = maps[0]
        scores["criterion"] = criterion
        return float(scores[criterion]), scores

    return _fit


def activation_search(data: ActivationSearchData,
                      space: SearchSpace,
                      *,
                      criterion: str = "deletion_auc",
                      mode: str = "grid",
                      n_trials: int = 12,
                      seed: int = 0,
                      n_steps: int = 12,
                      baseline: str = "blur",
                      run_sanity_check: bool = True,
                      attribute_fn: Optional[Callable[..., Any]] = None,
                      on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                      should_stop: Optional[Callable[[], bool]] = None,
                      ) -> SearchResult:
    """Sweep attribution settings, scoring each with the faithfulness checks.

    The honest deliverable is the panel of maps plus the four scores per trial,
    not the top row. There is no ground truth for attribution, so this refuses
    to name a single "best": :data:`ACTIVATION_NO_GROUND_TRUTH` leads the notes,
    every criterion is computed for every trial, and the usual within-noise flag
    fires when the leaders are indistinguishable.

    :param data: model + images (+ optional masks) to score on.
    :param space: attribution parameters to sweep (``cam_type``,
        ``target_layer``, ``smoothgrad_samples``, ``smoothgrad_sigma``,
        ``occlusion_window``, ``occlusion_stride``, ``ig_steps``,
        ``ig_baseline``).
    :param criterion: which criterion ranks the trials.
    :param mode: ``'grid'`` or ``'random'``.
    :param n_trials: configurations when ``mode='random'``; maximum complete
        2-by-2 rounds when adaptive UMAP is enabled.
    :param seed: seed for random sampling.
    :param n_steps: steps in the deletion / insertion curves.
    :param baseline: removal baseline for those curves.
    :param run_sanity_check: run the randomisation check per trial.
    :param attribute_fn: override for the attribution call, used by tests.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :returns: the :class:`SearchResult`.
    :raises ValueError: for an unknown criterion or mode.
    """
    if mode not in ("grid", "random"):
        raise ValueError(f"mode must be 'grid' or 'random', got {mode!r}.")
    fit = activation_fit_fn(data, criterion=criterion, n_steps=n_steps,
                            baseline=baseline,
                            run_sanity_check=run_sanity_check,
                            attribute_fn=attribute_fn)
    higher = criterion not in LOWER_IS_BETTER
    notes = list(data.notes) + [
        ACTIVATION_NO_GROUND_TRUTH,
        f"Criterion '{criterion}': {ACTIVATION_CRITERIA[criterion]}",
        f"Scored on {len(data.images)} image(s) with {n_steps}-step deletion "
        f"and insertion curves against a {baseline!r} baseline. "
        + ("Object masks were available, so the pointing game was scored too."
           if data.masks else
           "No object masks were available, so the pointing game could not be "
           "scored for any trial."),
        "Every criterion was computed for every trial, so you can re-rank the "
        "table by a different one and see whether the winner survives — it "
        "often does not, and that is the result.",
    ]
    if not run_sanity_check:
        notes.append(
            "The model-randomisation sanity check was skipped for this sweep. "
            "Without it, a method that returns the same map for a randomised "
            "model ranks exactly like one that does not.")
    if mode == "grid":
        return grid_search(fit, space, metric=criterion,
                           higher_is_better=higher, on_trial=on_trial,
                           should_stop=should_stop, notes=notes)
    return random_search(fit, space, n_trials, seed, metric=criterion,
                         higher_is_better=higher, on_trial=on_trial,
                         should_stop=should_stop, notes=notes)


def load_activation_data(settings: Mapping[str, Any],
                         *, n_images: int = 8) -> ActivationSearchData:
    """Load the model and a handful of images an Activation sweep scores on.

    Two sources, in order of preference:

    * ``src``/``merged/*.npy`` — spaCR's own merged arrays, which carry the
      image channels *and* the object label planes in one file. Preferred
      because the object mask comes free and exactly aligned, which is what
      makes the pointing game possible at all.
    * ``dataset`` — the crop tar the Activation run itself reads. Aligned masks
      do not exist for these crops, so the pointing game is unavailable and the
      returned notes say so rather than silently dropping the criterion.

    A sweep runs every configuration over every image, so ``n_images`` is small
    on purpose: the cost is ``configurations × images × (2 curves + 1 sanity
    cascade)`` forward passes.

    :param settings: the Activation app's settings dict.
    :param n_images: how many images to score on.
    :returns: the :class:`ActivationSearchData`.
    :raises ValueError: when neither source is usable.
    """
    import glob
    import os

    import numpy as np
    import torch

    model_path = settings.get("model_path")
    if not model_path or not os.path.isfile(str(model_path)):
        raise ValueError(
            f"No trained model to explain: model_path={model_path!r} is not a "
            f"file. Point it at a model saved by Classify before searching "
            f"attribution settings.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(str(model_path), map_location=device,
                       weights_only=False)
    model.to(device)
    model.eval()

    image_size = int(settings.get("image_size", 224) or 224)
    channels = list(settings.get("channels") or [1, 2, 3])
    notes: List[str] = []

    src = settings.get("src") or ""
    merged = os.path.join(str(src), "merged") if src else ""
    npys = sorted(glob.glob(os.path.join(merged, "*.npy"))) if merged else []
    if npys:
        mask_dims = settings.get("mask_dims") or {"cell": 4, "nucleus": 5,
                                                  "pathogen": 6, "organelle": 7}
        object_type = str(settings.get("object_type", "cell"))
        mask_dim = int(mask_dims.get(object_type, 4))
        images, masks, names = [], [], []
        for path in npys[:int(n_images)]:
            arr = np.load(path)
            if arr.ndim != 3 or arr.shape[-1] <= mask_dim:
                continue
            img = np.stack([arr[..., c] for c in channels
                            if c < arr.shape[-1]], axis=0).astype(np.float32)
            span = float(img.max() - img.min())
            img = (img - float(img.min())) / (span if span > 0 else 1.0)
            tensor = torch.from_numpy(img)[None]
            mask = torch.from_numpy(
                (arr[..., mask_dim] != 0).astype(np.float32))[None, None]
            tensor = torch.nn.functional.interpolate(
                tensor, size=(image_size, image_size), mode="bilinear",
                align_corners=False)[0]
            mask = torch.nn.functional.interpolate(
                mask, size=(image_size, image_size), mode="nearest")[0, 0]
            if not bool(mask.any()):
                continue
            images.append(tensor.to(device))
            masks.append(mask.cpu().numpy() != 0)
            names.append(os.path.basename(path))
        if images:
            notes.append(
                f"Scored on {len(images)} merged array(s) from {merged}, "
                f"channels {channels}, with the '{object_type}' label plane "
                f"(index {mask_dim}) as the pointing-game answer key. The mask "
                f"is the union of every {object_type} in the field, so the "
                f"pointing game asks whether the peak landed on any object "
                f"rather than on background — not which object.")
            return ActivationSearchData(
                model=model, images=images, masks=masks, filenames=names,
                model_type=settings.get("model_type"), notes=notes)
        notes.append(
            f"{len(npys)} merged arrays were found in {merged} but none had a "
            f"usable image + mask pair, so the crop tar was used instead.")

    dataset = settings.get("dataset")
    if not dataset or not os.path.isfile(str(dataset)):
        raise ValueError(
            f"Nothing to attribute: no merged/*.npy under src={src!r} and "
            f"dataset={dataset!r} is not a file. The search needs either "
            f"spaCR's merged arrays (which also give the object masks) or the "
            f"crop tar the Activation run reads.")

    from torchvision import transforms

    from .io import TarImageDataset
    from .utils import SelectChannels

    steps = [transforms.ToTensor(),
             transforms.CenterCrop(size=(image_size, image_size))]
    if settings.get("normalize_input", True):
        steps.append(transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5)))
    steps.append(SelectChannels(channels))
    ds = TarImageDataset(str(dataset), transform=transforms.Compose(steps))
    images, names = [], []
    for i in range(min(int(n_images), len(ds))):
        img, name = ds[i]
        images.append(img.to(device))
        names.append(str(name))
    if not images:
        raise ValueError(
            f"The crop tar {dataset!r} yielded no images, so there is nothing "
            f"to attribute.")
    notes.append(
        f"Scored on {len(images)} crop(s) from {dataset}. These crops have no "
        f"aligned object mask, so the pointing game cannot be scored — point "
        f"'src' at the experiment folder whose merged/*.npy files carry the "
        f"label planes to enable it.")
    return ActivationSearchData(model=model, images=images, masks=None,
                                filenames=names,
                                model_type=settings.get("model_type"),
                                notes=notes)


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
    elif result.metric in ACTIVATION_CRITERIA:
        lines.append(f"'{result.metric}' {ACTIVATION_CRITERIA[result.metric]}")
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
    pareto = result.pareto_front()
    if pareto:
        lines.append(
            f"Pareto front ({len(pareto)} non-dominated configuration(s)):")
        for trial in pareto[:max_rows]:
            values = ", ".join(
                f"{name}={float(trial.extra_metrics[name]):.4f}"
                for name in result.objectives
            )
            lines.append(f"  {trial.label()}  [{values}]")
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
    from .batch_correction import correction_kwargs

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

    from .row_exclusions import exclude_matching_rows
    frame, exclusion_notes = exclude_matching_rows(
        frame, settings.get("exclude_rows"))
    notes.extend(exclusion_notes)

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
        **correction_kwargs(
            settings,
            default_control_column=settings.get("col_to_compare"),
            default_control_values=settings.get("neg"),
        ),
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
                                  random_state=seed)
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
                       adaptive: bool = False,
                       walk_parameters: Optional[Sequence[str]] = None,
                       walk_resolutions: Optional[Mapping[str, int]] = None,
                       walk_steps: Optional[Mapping[str, float]] = None,
                       n_neighbors_step: int = 1,
                       min_dist_step: float = 0.05,
                       min_improvement: float = 0.0,
                       stability_repeats: int = 3,
                       objective_weights: Optional[
                           Mapping[str, Any]
                       ] = None,
                       seed: int = 0,
                       n_folds: int = 5,
                       on_trial: Optional[Callable[[Trial, int, int], None]] = None,
                       should_stop: Optional[Callable[[], bool]] = None,
                       data: Optional[SearchData] = None,
                       checkpoint_path: Optional[str] = None,
                       resume: bool = False,
                       ) -> SearchResult:
    """Run the right search for a spaCR app. This is what the GUI calls.

    * ``umap`` — embeds the measurement features once per configuration and
      ranks them with a named criterion (see :func:`umap_search`).
    * ``ml_analyze`` — grouped cross-validated search over classical ML
      hyperparameters (see :func:`cv_search`).
    * ``classify`` — one cross-validated deep-training run per configuration
      (see :func:`classify_cv_fit_fn`).
    * ``activation`` — one attribution per image per configuration, scored by
      deletion AUC, insertion AUC, the pointing game and the randomisation
      sanity check (see :func:`activation_search`).

    :param app_key: which app is asking.
    :param settings: that app's settings dict.
    :param space: the :class:`SearchSpace` to search.
    :param criterion: metric name; defaults to the app's first
      :data:`APP_CRITERIA` entry.
    :param mode: ``'grid'`` or ``'random'``.
    :param n_trials: configurations to evaluate when ``mode='random'``.
    :param adaptive: for UMAP only, run a Walk from one starting point.
    :param walk_parameters: which UMAP parameters the Walk searches; the
      default is n_neighbors and min_dist. Any subset of
      :data:`UMAP_WALK_PARAMETERS`.
    :param walk_resolutions: per-axis Walk grid resolution, ``{name: n}``.
    :param walk_steps: per-axis Walk step override, ``{name: size}``.
    :param n_neighbors_step: Walk integer neighborhood increment.
    :param min_dist_step: Walk min_dist increment.
    :param min_improvement: Walk score-gain stopping threshold.
    :param stability_repeats: repeated seeded embeddings per multi-objective
        UMAP configuration.
    :param objective_weights: weights for neighborhood preservation, stability
        and cluster structure in multi-objective UMAP mode.
    :param seed: seed for sampling, folds and reducers.
    :param n_folds: cross-validation folds for the supervised apps.
    :param on_trial: progress callback ``(trial, completed, total)``.
    :param should_stop: polled before each trial.
    :param data: pre-loaded :class:`SearchData` (or
        :class:`ActivationSearchData` for ``'activation'``), skipping the
        database / model read.
    :param checkpoint_path: UMAP checkpoint path; when omitted the UMAP
        project path is derived by :func:`umap_checkpoint_path`.
    :param resume: continue a compatible UMAP search checkpoint.
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

    if app_key == "activation":
        act_data = data if isinstance(data, ActivationSearchData) else None
        if act_data is None:
            act_data = load_activation_data(settings)
        return activation_search(
            act_data, space, criterion=criterion, mode=mode,
            n_trials=n_trials, seed=seed,
            n_steps=int(settings.get("attribution_steps", 12) or 12),
            baseline=str(settings.get("attribution_baseline", "blur")
                         or "blur"),
            run_sanity_check=bool(settings.get("sanity_check", True)),
            on_trial=on_trial, should_stop=should_stop)

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
        search_checkpoint = checkpoint_path or umap_checkpoint_path(settings)
        result = umap_search(
            data.features, space, metric=criterion, labels=data.labels,
            seed=seed, adaptive=adaptive, n_trials=n_trials,
            n_neighbors_step=n_neighbors_step,
            min_dist_step=min_dist_step,
            min_improvement=min_improvement,
            walk_parameters=walk_parameters,
            walk_resolutions=walk_resolutions,
            walk_steps=walk_steps,
            stability_repeats=stability_repeats,
            objective_weights=objective_weights,
            on_trial=on_trial, should_stop=should_stop,
            checkpoint_path=search_checkpoint, resume=resume)
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
