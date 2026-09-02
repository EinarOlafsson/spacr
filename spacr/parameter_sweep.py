"""Parameter Sweep: vary regression settings and see which change the answer.

A pooled screen has no single correct analysis. It has a model family, an
aggregation rule, a unit of analysis, a set of nuisance effects, a
multiple-testing correction, and three or four filtration cutoffs -- and a hit
list that is stable across all of them is a very different claim from one that
appears under exactly one combination.

This module makes that comparison a command rather than a week. It builds a
trial list from a search space, runs each trial into its own folder, and
returns one tidy row per trial: the settings, whether it ran, how many wells
and guides survived, how many hits it called, where the named controls landed,
and how long it took.

Two things make it fast enough to be worth having:

* **Preparation is shared.** Loading 226k score rows and 642k count rows,
  aggregating per well, thresholding and joining costs about twenty seconds
  and depends only on the FILTRATION settings. Every trial sharing those
  reuses one prepared frame, so a sweep over models and corrections pays that
  cost once per filtration cell rather than once per trial.
* **A failed trial is a result.** Many combinations are illegal by
  construction -- ``quantile`` refuses ``alpha``, the penalised families
  refuse ``cov_type``, ``random_row_column_effects`` replaces the backend
  entirely. Those are recorded with their reason and the sweep continues,
  because "this combination is not allowed" is information about the design
  space, not a crash.

The search space is declared as data (:data:`DEFAULT_SWEEP_SPACE`), so adding
an axis is adding a key.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# THE HOUSE STYLE (136). `figures.style` imports matplotlib
# only inside its own functions, so naming it here costs
# nothing at import time.
from .figures.style import figure_style, theme_target
from .trial_metrics import METRIC_COLUMNS as _METRIC_COLUMNS
from .trial_metrics import summarise_trial

__all__ = [
    "DEFAULT_SWEEP_SPACE",
    "PREPARATION_KEYS",
    "SweepSpace",
    "build_trials",
    "rank_trials",
    "recommended_workers",
    "run_sweep",
    "run_sweep_parallel",
    "summarise_sweep",
]


#: Settings that change the PREPARED data rather than the model fitted to it.
#: Trials agreeing on all of these share one prepared frame. Getting this list
#: wrong is the one way this module can silently produce wrong answers -- a
#: setting that belongs here but is missing would let a trial reuse a frame
#: built under a different rule -- so it is kept beside the code that uses it
#: and asserted against the settings the preparation actually reads.
PREPARATION_KEYS: tuple[str, ...] = (
    "score_data", "count_data", "dependent_variable", "min_cell_count",
    "fraction_threshold", "agg_type", "analysis_unit", "transform",
    "invert_dependent_variable", "filter_column", "filter_value",
    "target_unique_count", "outlier_detection", "plateID",
)


#: The axes swept by default, grouped by the question each one answers.
#: Values are the complete legal inventory where one exists, so a sweep covers
#: the space rather than a sample someone once typed.
DEFAULT_SWEEP_SPACE: dict[str, list] = {
    # --- how the effect is estimated ---------------------------------------
    "regression_type": [
        "ols", "wls", "rlm", "glm", "poisson", "quasi_binomial", "beta",
        "logit", "probit", "quantile", "lasso", "ridge", "elasticnet",
    ],
    # 'auto' cross-validates the penalty. The literal 1 is spaCR's default and
    # is far larger than the scale of a fraction design -- it shrinks every
    # coefficient to exactly zero -- so sweeping only the default would report
    # the three penalised families as uniformly useless.
    "alpha": ["auto", 1],
    "inference": ["parametric", "nonparametric"],
    # --- what one observation is -------------------------------------------
    "analysis_unit": ["well", "cell"],
    "agg_type": ["mean", "median", "quantile"],
    "transform": [None, "log"],
    # --- nuisance structure: row, column and plate --------------------------
    # False keeps rowID + columnID as FIXED effects in the formula; True moves
    # them to random effects, which replaces the backend with a mixed model.
    "random_row_column_effects": [False, True],
    # The plate effect. 'none' leaves plates alone; the rest remove a
    # per-plate shift before fitting.
    "batch_correction": ["none", "center", "zscore"],
    # --- what counts as significant -----------------------------------------
    "multiple_testing_method": [
        "none", "bonferroni", "sidak", "holm", "holm_sidak",
        "simes_hochberg", "hommel", "fdr_bh", "fdr_by", "fdr_tsbh",
        "fdr_tsbky", "fdr_gbs", "storey",
    ],
    "fdr_alpha": [0.05],
    "threshold_method": ["std"],
    "threshold_multiplier": [2, 3],
    # --- filtration ----------------------------------------------------------
    "fraction_threshold": [0.02],
    "min_cell_count": [100],
    "min_n": [0],
    "outlier_detection": [False],
}


@dataclass
class SweepSpace:
    """Define variable and fixed settings for a parameter sweep.

    Parameters
    ----------
    axes : dict of str to list, optional
        Settings to vary. Each trial receives one value from every list.
        Defaults to a copy of :data:`DEFAULT_SWEEP_SPACE`.
    fixed : dict, optional
        Settings copied into every trial after the Cartesian product is
        formed and before filters are evaluated.
    filters : list of callable, optional
        Predicates called with a complete trial dictionary. A predicate
        returns ``None`` to accept a trial or a reason string to reject it.
        An empty list selects the built-in compatibility filters when trials
        are built.
    """

    axes: dict[str, list] = field(
        default_factory=lambda: dict(DEFAULT_SWEEP_SPACE))
    fixed: dict[str, Any] = field(default_factory=dict)
    #: Predicates that reject a combination before it is run. Each takes the
    #: trial dict and returns a reason string, or None to allow it.
    filters: list[Callable[[dict], str | None]] = field(default_factory=list)

    def size(self) -> int:
        total = 1
        for values in self.axes.values():
            total *= max(len(values), 1)
        return total


def _default_filters() -> list[Callable[[dict], str | None]]:
    """Combinations spaCR refuses, rejected before they cost a run.

    These are not guesses. Each mirrors a rule ``spacr.ml`` enforces, and
    dropping them here turns a sweep into thousands of identical tracebacks.
    """
    def mixed_replaces_the_backend(trial):
        """Reject random row/column effects beside a non-mixed backend."""
        if trial.get("random_row_column_effects") and \
                trial.get("regression_type") not in (None, "ols", "mixed"):
            return (f"random_row_column_effects=True fits a mixed model and "
                    f"cannot also fit {trial['regression_type']!r}")
        return None

    def aggregation_belongs_to_wells(trial):
        """Reject aggregation variants that a per-cell analysis ignores."""
        if trial.get("analysis_unit") == "cell" and \
                trial.get("agg_type") not in (None, "mean"):
            # agg_type is forced to None for a per-cell fit, so sweeping it
            # would run the same analysis several times under different labels.
            return "analysis_unit='cell' ignores agg_type"
        return None

    def quantile_needs_its_own_unit(trial):
        """Reject well-level quantile fits, which require per-object rows."""
        if trial.get("regression_type") == "quantile" and \
                trial.get("analysis_unit") == "well":
            return ("regression_type='quantile' fits per-object values, so it "
                    "forces analysis_unit='cell'")
        return None

    def permutation_ignores_the_family(trial):
        """Reject backend variants unused by nonparametric inference."""
        # The permutation test is its own estimator: it does not read
        # regression_type, and sweeping it would repeat one analysis 13 times.
        if trial.get("inference") == "nonparametric" and \
                trial.get("regression_type") not in (None, "ols"):
            return "inference='nonparametric' does not use regression_type"
        return None

    def permutation_at_cell_level_exhausts_memory(trial):
        """Reject the per-cell permutation fit measured to require 57 GiB."""
        # THE COMBINATION THAT TOOK THE MACHINE DOWN.
        #
        # The permutation test builds `x_unit.T @ permuted_outcomes` in
        # batches. At WELL level that is 606 rows and costs nothing. At CELL
        # level it is ~116,000 rows against 200,000 permutations, and a single
        # trial was measured holding 57 GB of resident memory before the host
        # ran out -- one fit, on its own, with nothing running in parallel.
        #
        # Rejected rather than merely discouraged: there is no worker count,
        # thread limit or nice level that makes it survivable, because the
        # allocation happens inside one process regardless.
        if trial.get("inference") == "nonparametric" and \
                trial.get("analysis_unit") == "cell":
            return ("inference='nonparametric' with analysis_unit='cell' "
                    "permutes ~10^5 rows and exhausts memory")
        return None

    def permutation_has_no_row_column_terms(trial):
        """Reject row/column effects that the plate-blocked permutation omits."""
        if trial.get("inference") == "nonparametric" and \
                trial.get("random_row_column_effects"):
            return ("inference='nonparametric' blocks on plate and does not "
                    "fit row/column effects")
        return None

    def penalty_belongs_to_penalised_families(trial):
        """Reject non-default alpha values for families that never read them."""
        # alpha is refused outright by every family that cannot read it, so
        # sweeping it against them would turn one axis into a wall of
        # identical rejections.
        if trial.get("alpha") not in (None, 1) and \
                trial.get("regression_type") not in (
                    "lasso", "ridge", "elasticnet", "hinge"):
            return (f"alpha is only read by the penalised families, not "
                    f"{trial.get('regression_type')!r}")
        return None

    return [mixed_replaces_the_backend, aggregation_belongs_to_wells,
            quantile_needs_its_own_unit, permutation_ignores_the_family,
            permutation_at_cell_level_exhausts_memory,
        permutation_has_no_row_column_terms,
            penalty_belongs_to_penalised_families]


def build_trials(space: SweepSpace, *, mode: str = "grid",
                 max_trials: int = 5000, seed: int = 0) -> list[dict]:
    """Enumerate accepted trials from a sweep space.

    Parameters
    ----------
    space : SweepSpace
        Variable axes, fixed settings, and optional rejection predicates.
    mode : {'grid', 'random'}, default='grid'
        ``'grid'`` visits the Cartesian product in axis order. ``'random'``
        shuffles that product without replacement before applying the limit.
    max_trials : int, default=5000
        Maximum number of accepted trials to return. Rejected combinations do
        not count toward this limit.
    seed : int, default=0
        Seed used to shuffle combinations in ``'random'`` mode. It has no
        effect in ``'grid'`` mode.

    Returns
    -------
    list of dict
        Complete setting dictionaries with one-based ``trial_id`` values.
        Fixed settings are present when filters are evaluated.

    Raises
    ------
    ValueError
        If ``mode`` is not ``'grid'`` or ``'random'``.

    Notes
    -----
    Random mode materializes the full Cartesian product before shuffling it.
    Use narrower axes when the unfiltered product is very large.
    """
    if mode not in {"grid", "random"}:
        raise ValueError("mode must be 'grid' or 'random'")
    names = list(space.axes)
    values = [space.axes[name] for name in names]
    filters = list(space.filters) or _default_filters()

    def accept(trial):
        """Return the first filter's rejection reason, or ``None``."""
        for rule in filters:
            reason = rule(trial)
            if reason:
                return reason
        return None

    combinations = itertools.product(*values)
    if mode == "random":
        combinations = list(combinations)
        random.Random(seed).shuffle(combinations)

    trials, rejected = [], []
    for combination in combinations:
        trial = dict(zip(names, combination))
        # THE FIXED VALUES ARE PART OF THE TRIAL BEFORE IT IS JUDGED.
        #
        # This used to run AFTER accept(), which meant every filter decided on
        # a half-built trial. The GUI pins each UNTICKED axis into `fixed`
        # (qt/screens/parameter_sweep.py), so the settings a user did not vary
        # were exactly the ones the filters could not see -- and
        # `permutation_at_cell_level_exhausts_memory` read `analysis_unit` as
        # None and passed.
        #
        # Reproduced with stock filters and nothing exotic: axes
        # {"inference": ["parametric", "nonparametric"]} with
        # fixed {"analysis_unit": "cell"} emitted the nonparametric x cell
        # permutation, which is the ~57 GiB run. Unticking one checkbox was
        # enough to schedule it, and the filter written to prevent exactly
        # that was already in the default list.
        trial.update(space.fixed)
        reason = accept(trial)
        if reason:
            rejected.append((trial, reason))
            continue
        trial["trial_id"] = len(trials) + 1
        trials.append(trial)
        if len(trials) >= max_trials:
            break
    return trials


def _preparation_key(settings: Mapping[str, Any]) -> str:
    """Stable identity for the prepared data a trial needs."""
    parts = []
    for key in PREPARATION_KEYS:
        value = settings.get(key)
        if isinstance(value, (list, tuple)):
            value = "|".join(str(item) for item in value)
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _named_control_rows(results: pd.DataFrame, names: Mapping[str, str]
                        ) -> dict:
    """Where each named control landed, for judging a trial's answer.

    A sweep is only interpretable against something known. The positive
    control must be recovered; a setting that loses it is not a setting worth
    using however few hits it reports.
    """
    out: dict[str, Any] = {}
    if results is None or results.empty:
        return out
    frame = results.copy()
    label_column = next(
        (c for c in ("grna", "guide", "feature", "gene") if c in frame.columns),
        None)
    effect_column = next(
        (c for c in ("coefficient", "standardized_marginal_effect", "effect")
         if c in frame.columns), None)
    q_column = next(
        (c for c in ("q_value", "adjusted_p_value") if c in frame.columns),
        None)
    p_column = next(
        (c for c in ("p_value", "permutation_p_value") if c in frame.columns),
        None)
    if label_column is None:
        return out
    labels = frame[label_column].astype(str)
    if effect_column:
        ranked = frame.assign(_abs=frame[effect_column].abs()).sort_values(
            "_abs", ascending=False).reset_index(drop=True)
        ranked_labels = ranked[label_column].astype(str)
    else:
        ranked, ranked_labels = frame, labels
    for alias, needle in names.items():
        hit = labels.str.contains(str(needle), regex=False, na=False)
        out[f"{alias}_present"] = bool(hit.any())
        if not hit.any():
            continue
        row = frame.loc[hit].iloc[0]
        if effect_column:
            position = ranked_labels.str.contains(
                str(needle), regex=False, na=False)
            # ranked_labels is a permutation of labels, so a control present
            # in labels is necessarily present here as well.
            row = ranked.loc[position].iloc[0]
            out[f"{alias}_rank"] = int(position.idxmax()) + 1
            out[f"{alias}_effect"] = float(row[effect_column])
        if q_column and pd.notna(row.get(q_column)):
            out[f"{alias}_q"] = float(row[q_column])
        if p_column and pd.notna(row.get(p_column)):
            out[f"{alias}_p"] = float(row[p_column])
    return out


def _design_summary(output: Mapping[str, Any]) -> dict:
    """How much data actually reached the fit.

    Two trials can differ by a filtration cutoff alone and end up fitting
    completely different designs, so a hit count means little without the
    size of the thing it came from. A row that carries both answers "did
    raising the cell-count threshold change the answer, or just throw data
    away?" without opening the trial folder.
    """
    summary: dict[str, Any] = {}
    if not isinstance(output, Mapping):
        return summary
    for key, column in (("n_wells", "prc"), ("n_guides", "grna")):
        frame = output.get("model_data")
        if isinstance(frame, pd.DataFrame) and column in frame.columns:
            summary[key] = int(frame[column].nunique())
    for key in ("n_wells", "n_guides", "n_cells"):
        if key not in summary and key in output:
            try:
                summary[key] = int(output[key])
            except (TypeError, ValueError):
                pass
    frame = output.get("model_data")
    if isinstance(frame, pd.DataFrame):
        summary.setdefault("n_rows_fitted", int(len(frame)))
    return summary


def correction_rows(output: Mapping[str, Any], methods: Sequence[str],
                    alpha: float = 0.05) -> list[dict]:
    """One row per correction, from ONE fit.

    A multiple-testing correction is applied to p-values that already exist;
    it does not change the model, the design, or a single coefficient. Sweeping
    it as an axis therefore refits the identical regression once per method --
    thirteen fits to obtain thirteen numbers that all come from the first one.

    On this screen that is the difference between ~24 hours and ~2, for exactly
    the same answers, which is why it is worth doing rather than clever.

    :returns: ``[{'multiple_testing_method': m, 'n_below_alpha': n, ...}, ...]``
    """
    frame = output.get("results") if isinstance(output, Mapping) else None
    if not isinstance(frame, pd.DataFrame) or "p_value" not in frame.columns:
        return [{"multiple_testing_method": m} for m in methods]

    from .multiple_testing import adjust_p_values

    p_values = pd.to_numeric(frame["p_value"], errors="coerce")
    rows = []
    for method in methods:
        row = {"multiple_testing_method": method}
        try:
            # (adjusted, reject). The reject mask is the METHOD'S OWN verdict:
            # step-down procedures like holm and hommel do not reduce to
            # "adjusted <= alpha" applied afterwards, so re-thresholding here
            # would quietly report different hits than the method called.
            adjusted, reject = adjust_p_values(
                p_values.to_numpy(), method=method, alpha=alpha)
            adjusted = pd.to_numeric(pd.Series(adjusted), errors="coerce")
            row["n_below_alpha"] = int(np.asarray(reject).sum())
            row["n_tests"] = int(p_values.notna().sum())
            row["smallest_adjusted_p"] = float(adjusted.min()) \
                if adjusted.notna().any() else float("nan")
        except Exception as error:  # noqa: BLE001 - one method must not sink the rest
            row["correction_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def _count_hits(output: Mapping[str, Any]) -> dict:
    """How many things the trial called, at whatever level it reports them."""
    counts: dict[str, Any] = {}
    for key in ("results", "significant", "primary"):
        frame = output.get(key) if isinstance(output, Mapping) else None
        if isinstance(frame, pd.DataFrame):
            counts[f"n_{key}"] = int(len(frame))
    frame = output.get("results") if isinstance(output, Mapping) else None
    if isinstance(frame, pd.DataFrame):
        for column in ("q_value", "adjusted_p_value"):
            if column in frame.columns:
                counts["n_below_alpha"] = int(
                    (pd.to_numeric(frame[column], errors="coerce") < 0.05).sum())
                break
    return counts


#: Never exceed this many workers however much memory is free. A sweep is
#: background work; it does not get to own the machine.
MAX_WORKERS = 8

#: What one trial is assumed to need when it has not been measured. Each
#: trial independently loads the score and count tables, builds the design
#: matrix and imports torch.
ASSUMED_TRIAL_GIB = 6.0

#: Share of currently-available memory a sweep may plan to use. The rest
#: remains available to the desktop and operating system.
MEMORY_BUDGET_FRACTION = 0.5


def _recommended_worker_budget(*, measured_gib=None, requested=None) -> dict:
    """Return the values used to choose a safe worker count."""
    per_trial = float(measured_gib or ASSUMED_TRIAL_GIB)
    available = None
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:  # psutil is a dependency, but be safe
        pass
    try:
        cores = max(len(os.sched_getaffinity(0)) - 2, 1)
    except AttributeError:  # non-Linux
        cores = max((os.cpu_count() or 2) - 2, 1)

    if available is None:
        workers = max(1, min(2, cores, MAX_WORKERS))
        return {
            "workers": workers,
            "available": None,
            "per_trial": per_trial,
            "budget_fraction": MEMORY_BUDGET_FRACTION,
            "requested": requested,
        }

    affordable = int((available * MEMORY_BUDGET_FRACTION) // per_trial)
    workers = max(1, min(affordable, cores, MAX_WORKERS,
                         requested or MAX_WORKERS))
    return {
        "workers": workers,
        "available": available,
        "per_trial": per_trial,
        "budget_fraction": MEMORY_BUDGET_FRACTION,
        "requested": requested,
    }


def recommended_workers(*, measured_gib=None, requested=None):
    """Choose a worker count from available memory and CPU capacity.

    Parameters
    ----------
    measured_gib : float or None, optional
        Peak resident memory for one representative trial, in GiB. ``None``
        uses :data:`ASSUMED_TRIAL_GIB`.
    requested : int or None, optional
        Preferred maximum worker count. The result is still limited by memory,
        available CPU cores, and :data:`MAX_WORKERS`.

    Returns
    -------
    workers : int
        Recommended number of worker processes, always at least one.
    reason : str
        Explanation of the memory estimate and any reduction from the
        requested count, suitable for logs. The Qt screen renders the same
        budget with localized templates.

    Notes
    -----
    The calculation budgets :data:`MEMORY_BUDGET_FRACTION` of currently
    available memory. If memory cannot be measured, at most two workers are
    recommended.
    """
    budget = _recommended_worker_budget(
        measured_gib=measured_gib,
        requested=requested,
    )
    workers = int(budget["workers"])
    available = budget["available"]
    per_trial = float(budget["per_trial"])
    if available is None:
        return workers, (f"memory could not be measured, so the sweep is "
                         f"limited to {workers} workers")

    reason = (f"{available:.0f} GiB free, ~{per_trial:.1f} GiB per trial, "
              f"budgeting {MEMORY_BUDGET_FRACTION:.0%} of free memory "
              f"-> {workers} worker{'s' if workers != 1 else ''}")
    if requested and workers < requested:
        reason += f" (you asked for {requested})"
    return workers, reason


_LAST_MEMORY_STATE: dict[str, Any] = {}


def memory_is_low(floor_gib: float = 8.0,
                  spacr_ceiling_gib: float | None = None) -> bool:
    """True when the machine or spaCR's own process tree crossed its limit.

    Checked between submissions, not only at the start: the other things on
    the machine -- an editor, the spaCR GUI, another analysis -- grow while
    the sweep runs, and the sweep must yield to them rather than race them.

    ``spacr_ceiling_gib`` is optional because the machine-wide free-memory
    floor remains the default safety policy. When a caller supplies a run
    budget, the active resource sampler's latest process-tree total makes the
    guard attributable: it can distinguish "the machine is busy" from
    "spaCR is the reason" instead of treating both as the same number.
    """
    available = None
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        pass

    own = None
    try:
        from .runctx import current_run_context

        context = current_run_context()
        sampler = getattr(context, "_resource_sampler", None)
        figure = (sampler._summary.get("last_tree_memory_bytes")
                  if sampler is not None else None)
        if figure is not None:
            own = float(figure) / (1024 ** 3)
    except Exception:                                           # noqa: BLE001
        pass

    machine_low = available is not None and available < float(floor_gib)
    spacr_low = (spacr_ceiling_gib is not None and own is not None
                 and own > float(spacr_ceiling_gib))
    _LAST_MEMORY_STATE.clear()
    _LAST_MEMORY_STATE.update({
        "available_gib": available,
        "spacr_tree_gib": own,
        "machine_low": machine_low,
        "spacr_low": spacr_low,
    })
    return bool(machine_low or spacr_low)


#: Environment variables every numerical library reads for its thread pool.
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _pin_threads(count: int = 1) -> None:
    """One compute thread per trial process. THIS IS WHY SWEEPS KILLED THE GUI.

    Measured on a real trial: torch alone defaults to 16 threads, and with
    BLAS and OpenMP on top a single trial peaks at 112 THREADS. Six workers is
    then 672 runnable threads on 32 cores -- a scheduling storm that starves
    everything else on the machine, the desktop included. It is not a memory
    problem, which is why watching free memory never predicted it and why
    lowering the worker count never fixed it: each worker was the problem.

    nice() cannot help. A niced thread still has to be scheduled, and there
    are twenty times more of them than there are cores.

    The threads buy nothing here anyway. The same trial takes 18.4 s pinned
    against 18.5 s unpinned: this workload is a few thousand rows of
    statsmodels, not a matrix multiply that parallelises. Parallelism belongs
    ACROSS trials, where it is already, not inside each one.
    """
    for name in _THREAD_VARS:
        os.environ[name] = str(count)
    # THE ENVIRONMENT ALONE IS NOT ENOUGH, and this is the part that bit.
    #
    # OpenBLAS reads OMP_NUM_THREADS once, when numpy first imports it, and
    # sizes its pool from the core count if the variable is not set yet. Any
    # module that imports numpy before this runs -- which is most of them --
    # leaves the pool at 32 threads no matter what is put in os.environ
    # afterwards. Measured: env-then-numpy gives 1 thread, numpy-then-env
    # gives 32.
    #
    # threadpool_limits resizes the LIVE pool, so it works whatever the import
    # order was. It is held open for the life of the process rather than used
    # as a context manager, because the fitting happens after this returns.
    try:
        from threadpoolctl import threadpool_limits
        global _THREAD_LIMITS
        _THREAD_LIMITS = threadpool_limits(limits=count)
    except Exception:  # threadpoolctl may be absent
        pass
    try:
        import torch
        torch.set_num_threads(count)
    except Exception:  # torch may be absent
        pass


#: Held open for the process lifetime; see :func:`_pin_threads`.
_THREAD_LIMITS = None


def be_polite() -> None:
    """Lower the current worker's CPU, I/O, and OOM-kill priority.

    The adjustments are best-effort and platform dependent. On Linux, the
    worker uses the lowest CPU and I/O priorities and sets
    ``oom_score_adj=800`` so it is preferred over interactive applications
    when the system is under memory pressure.
    """
    try:
        os.nice(19)
    except (OSError, AttributeError):
        pass
    try:
        # Linux only, and best effort: a container or a hardened kernel may
        # refuse the write, which is not a reason to fail a sweep.
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as handle:
            handle.write("800")
    except OSError:  # not Linux, or not permitted
        pass
    try:
        import subprocess
        # Linux idle I/O class: a sweep reads gigabytes of CSV and should
        # yield the disk to anything interactive.
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                       check=False, capture_output=True)
    except Exception:  # best effort
        pass


#: Requested memory ceiling for one trial. The kernel enforces it only when
#: :func:`containment_available` confirms a usable systemd user scope.
TRIAL_MEMORY_MAX = "24G"
#: Requested CPU quota for a contained trial, equivalent to four cores.
TRIAL_CPU_QUOTA = "400%"
#: Available-memory threshold below which a new trial is not started.
FREE_MEMORY_FLOOR_GB = 20.0


@lru_cache(maxsize=1)
def containment_available() -> bool:
    """Return whether a user scope accepts the trial memory limits.

    Finding ``systemd-run`` is insufficient: hosted runners and containers
    may ship the executable without a usable user manager or delegated memory
    controller. A small no-op scope verifies the properties used for a real
    trial without allocating meaningful memory.
    """
    import shutil

    if not shutil.which("systemd-run"):
        return False
    try:
        import subprocess
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet",
             "-p", "MemoryMax=64M", "-p", "MemorySwapMax=0", "true"],
            capture_output=True, timeout=10)
        return probe.returncode == 0
    except Exception:  # no user manager
        return False


def containment_note() -> str:
    """Return a user-facing summary of trial resource containment.

    Returns
    -------
    str
        Active kernel limits when containment is available, or a warning that
        explains the remaining safeguards and safer alternatives when it is
        unavailable.
    """
    if containment_available():
        return (f"Kernel containment is active for each trial: memory "
                f"{TRIAL_MEMORY_MAX}, swap disabled, and CPU quota "
                f"{TRIAL_CPU_QUOTA}. If a trial exceeds a limit, only that "
                f"trial is stopped and recorded as 'killed'; the sweep "
                f"continues.")
    return ("Kernel containment is unavailable because systemd-run --user "
            "--scope could not be started. This is common in containers and "
            "SSH sessions without a user manager. Thread limits and the "
            "free-memory check still apply, but they cannot prevent a single "
            "trial from exhausting system memory. Reduce the worker count or "
            "run the sweep from a systemd user session before using a large "
            "search space.")


def free_memory_gb() -> float:
    """Memory the kernel says is actually available, in GB.

    ``MemAvailable``, not ``MemFree``: free memory on a working machine is
    close to zero because the page cache holds the rest, and scheduling a
    trial against that number would refuse every trial on a healthy box.
    MemAvailable is the kernel's own estimate of what a new allocation could
    get without swapping.

    :returns: available memory in GB, or ``inf`` where the file does not
        exist. Infinity rather than zero deliberately -- this is a safety
        check, and a check that cannot read the machine must not become a
        check that blocks every run on it.
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:  # not Linux
        pass
    return float("inf")


def run_trial_contained(settings: Mapping[str, Any], *, trial_id=None,
                        controls: Mapping[str, str] | None = None,
                        timeout: float = 1800.0,
                        memory_max: str = TRIAL_MEMORY_MAX,
                        cpu_quota: str = TRIAL_CPU_QUOTA) -> dict:
    """Run one trial in a fresh child process and return a status row.

    When :func:`containment_available` is true, a systemd user scope enforces
    ``memory_max``, disables swap, and applies ``cpu_quota``. Otherwise the
    child still uses reduced priority and thread limits, but it has no hard
    memory ceiling; a warning is printed before it starts.

    Parameters
    ----------
    settings : mapping
        Regression settings for the child process.
    trial_id : optional
        Identifier copied into the returned row.
    controls : mapping, optional
        Named control identifiers used to summarize the result.
    timeout : float, default=1800
        Maximum child runtime in seconds.
    memory_max : str, default=TRIAL_MEMORY_MAX
        Systemd memory limit used when kernel containment is available.
    cpu_quota : str, default=TRIAL_CPU_QUOTA
        Systemd CPU quota used when kernel containment is available.

    Returns
    -------
    dict
        The child's result, or a row whose ``status`` is ``"killed"``,
        ``"timeout"``, or ``"failed"``.
    """
    import json
    import subprocess
    import sys
    import tempfile

    # The control ALIASES travel with the trial. Without them a contained row
    # loses the `{alias}_rank` columns -- and `positive_rank` is one of the
    # columns the sweep screen puts in its table, so containing a trial would
    # have quietly emptied the one column the run is judged on.
    payload = {"settings": dict(settings), "trial_id": trial_id,
               "controls": dict(controls or {})}
    folder = settings.get("src") or tempfile.gettempdir()
    os.makedirs(folder, exist_ok=True)
    settings_path = os.path.join(folder, "_trial_settings.json")
    out_path = os.path.join(folder, "_trial_result.json")
    with open(settings_path, "w") as handle:
        json.dump(payload, handle, default=str)

    child = [sys.executable, "-m", "spacr.sweep_child", settings_path, out_path]
    if containment_available():
        command = ["systemd-run", "--user", "--scope", "--quiet",
                   "-p", f"MemoryMax={memory_max}",
                   "-p", "MemorySwapMax=0",
                   "-p", f"CPUQuota={cpu_quota}",
                   "-p", "TasksMax=64",
                   "nice", "-n", "19"] + child
    else:
        # Say so rather than pretending the cap is there: an uncapped sweep is
        # a decision the user should get to make knowingly.
        print("WARNING: systemd-run is unavailable, so this trial runs "
              "WITHOUT a memory cap. A runaway fit can take the machine "
              "down. Reduce the worker count and search space, or start the "
              "sweep from a systemd user session before running large "
              "trials.")
        command = ["nice", "-n", "19"] + child

    environment = dict(os.environ)
    for name in _THREAD_VARS:
        environment[name] = "1"

    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=timeout, env=environment)
        code, tail = finished.returncode, (finished.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        code, tail = -1, "timed out"

    if os.path.exists(out_path):
        try:
            with open(out_path) as handle:
                result = json.load(handle)
        except Exception:  # truncated by a kill
            pass
        else:
            # A pool worker must carry the stamp one hop farther to the real
            # parent. A direct/main-process caller can register it now and
            # must not expose a private transport column in its public row.
            import multiprocessing
            if multiprocessing.current_process().name == "MainProcess":
                return _register_resource_workers(result)
            return result
    # No result file: the child was killed before it could write one. The cap
    # is the likeliest reason and worth naming, because "killed" and "crashed"
    # want different responses from the user.
    return {"status": "timeout" if tail == "timed out" else "killed",
            "trial_id": trial_id,
            "error_type": "MemoryMax" if code not in (0, -1) else "Timeout",
            "error": (f"the trial was killed; it may have exceeded "
                      f"MemoryMax={memory_max}. {tail}").strip()}


def _trial_settings(base_settings, trial, destination, *, qc: bool = False):
    """Build one trial's settings dict and its own output folder.

    :param qc: draw the full diagnostic suite for every trial. Off, because
        a hundred trials of it is ten minutes and two thousand files almost
        nobody opens; see the note at ``regression_qc`` below.
    """
    settings = dict(base_settings)
    settings.update({k: v for k, v in trial.items() if k != "trial_id"})
    folder = os.path.join(destination, f"trial_{trial['trial_id']:04d}")
    os.makedirs(folder, exist_ok=True)
    settings["src"] = folder
    settings.setdefault("verbose", False)
    settings.setdefault("Toxoplasma", False)
    # THE QC SUITE IS OFF FOR A SWEEP UNLESS IT IS ASKED FOR.
    #
    # It costs ~5.8 s and writes ~19 figures plus a combined PDF per fit --
    # right for one analysis, and roughly ten minutes and two thousand files
    # across a hundred trials, almost none of which anyone will open. The
    # SCALAR diagnostics that make a row judgeable are a different thing:
    # summarise_trial computes them in ~150 ms and they are unaffected by
    # this, so a sweep still sorts by control rank, inflation and R^2.
    #
    # ASSIGNED, NOT setdefault -- and this is the whole bug. `setdefault`
    # does nothing when the key is already there, and EVERY base dict reaching
    # here already carries it at True: the Tk panel, the Qt panel and
    # `spacr-run` all build theirs from
    # `get_perform_regression_default_settings`, which defaults it on.
    # Measured 2026-08-18: `_trial_settings(get_perform_regression_default_settings({}), ...)`
    # came back with regression_qc=True, so a sweep driven from the
    # application paid the full suite on every trial -- roughly ten minutes
    # and two thousand files per hundred trials -- while the comment above
    # said it did not.
    #
    # `qc` is the caller's explicit say. A sweep that wants the pictures asks
    # for them; reopening one interesting trial goes through
    # settings_for_trial, which does not pass here, so the trial you choose to
    # look at again is still fitted WITH the diagnostics.
    settings["regression_qc"] = bool(qc)
    # Write the settings the way the GUI writes them, so a trial worth a
    # second look can be opened straight in the regression module -- point it
    # at the trial folder and press run. A sweep whose interesting rows cannot
    # be reopened is only half an answer.
    try:
        from .utils import save_settings
        save_settings(dict(settings), name="regression", show=False)
    except Exception:  # never lose a trial over its record
        pass
    return settings, folder


def _execute_trial(payload):
    """Run one trial in this process and return its summary row.

    Module level and argument-only, so it can be pickled to a worker. Each
    worker imports spaCR itself: ``perform_regression`` pulls in torch and
    matplotlib, and a forked copy of those is not safe to reuse.
    """
    # SIX OR FIVE. `qc` was appended on 2026-08-18 and this tuple is an
    # internal, pickled, POSITIONAL contract -- growing it broke four
    # tests that build a payload by hand, and would break any caller
    # pickled by an older process mid-sweep. Reading it with a default
    # costs one line and makes the addition backward compatible; the
    # default is False, which is what a sweep wants.
    base_settings, trial, destination, controls, contained = payload[:5]
    qc = bool(payload[5]) if len(payload) > 5 else False
    # Before anything expensive: this work yields to the user's machine.
    be_polite()
    _pin_threads()

    # Returned through the existing future rather than a new IPC channel.
    # The parent sampler may already have seen this PID; registering the
    # creation-time stamp retroactively attaches the trial name to that
    # process and to its eventual disappearance event without a PID-reuse
    # race.
    from .fit_resources import _worker_stamp
    resource_workers = [
        _worker_stamp("parameter_sweep_pool", trial["trial_id"])
    ]

    settings, folder = _trial_settings(base_settings, trial, destination,
                                       qc=qc)
    row = {"trial_id": trial["trial_id"], "folder": folder,
           "preparation_key": _preparation_key(settings)}
    row.update({k: v for k, v in trial.items() if k != "trial_id"})
    began = time.time()

    if contained:
        # THE CAP APPLIES TO THE SWEEP THE GUI ACTUALLY RUNS.
        #
        # run_sweep has been contained by default since the kernel-cap work;
        # this path had not been, and this is the one the sweep screen calls.
        # So every guarantee that work bought -- MemoryMax, MemorySwapMax=0,
        # CPUQuota, TasksMax -- was absent from the only sweep a user starts
        # by clicking Start, and what remained was recommended_workers() and
        # the free-memory floor. Those are ACCOUNTING, and this module's own
        # history is that accounting is not containment: every previous fix
        # was a better estimate of what a trial would use, and each one was
        # wrong in a way that took the desktop with it.
        #
        # The pool worker now only waits on the child, so it holds no design
        # matrix of its own and the worker count stops being a memory
        # multiplier as well.
        child = run_trial_contained(settings, trial_id=trial["trial_id"],
                                    controls=controls)
        child_stamp = child.pop("_resource_worker", None)
        if isinstance(child_stamp, Mapping):
            resource_workers.append(dict(child_stamp))
        row.update({k: v for k, v in child.items() if k != "trial_id"})
        row["seconds"] = child.get("seconds", round(time.time() - began, 2))
        row["_resource_workers"] = resource_workers
        return row

    from .ml import perform_regression

    try:
        output = perform_regression(settings)
        row["status"] = "ok"
        if isinstance(output, Mapping):
            row.update(_count_hits(output))
            row.update(_design_summary(output))
            # THE SAME COLUMNS WHICHEVER WAY THE TRIAL RAN.
            #
            # This is the path the GUI uses (run_sweep_parallel), and it was
            # the only one that never called summarise_trial: a contained
            # trial got fit quality, residual tests, design rank and control
            # recovery, and a trial run from the sweep screen got a hit count.
            # Same sweep, same question, different table depending on which
            # entry point produced it.
            row.update(summarise_trial(output, settings))
            results = output.get("results")
            if isinstance(results, pd.DataFrame):
                row.update(_named_control_rows(results, controls))
    except BaseException as error:  # noqa: BLE001 - a failed trial is a result
        row["status"] = "failed"
        row["error_type"] = type(error).__name__
        row["error"] = str(error).splitlines()[0][:300] if str(error) else ""
        try:
            with open(os.path.join(folder, "error.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
        except OSError:
            pass
    row["seconds"] = round(time.time() - began, 2)
    row["_resource_workers"] = resource_workers
    return row


def _register_resource_workers(row: dict) -> dict:
    """Attach private child stamps to the active run, then remove them."""
    raw_stamps = row.pop("_resource_workers", [])
    stamps = (list(raw_stamps)
              if isinstance(raw_stamps, (list, tuple)) else [])
    single = row.pop("_resource_worker", None)
    if isinstance(single, Mapping):
        stamps = [*stamps, single]
    try:
        from .runctx import current_run_context

        context = current_run_context()
        if context is not None:
            for stamp in stamps:
                if isinstance(stamp, Mapping):
                    context.register_worker(stamp)
    except Exception:                                           # noqa: BLE001
        # Accounting must never change whether a trial is a result.
        pass
    return row


def run_sweep_parallel(base_settings: Mapping[str, Any], destination,
                        space: "SweepSpace | None" = None, *,
                        mode: str = "random", max_trials: int = 1000,
                        seed: int = 0,
                        controls: Mapping[str, str] | None = None,
                        n_jobs: int = 8,
                        contained: bool = True,
                        qc: bool = False,
                        progress_every: int = 25) -> pd.DataFrame:
    """Run parameter-sweep trials concurrently in spawned processes.

    Parameters
    ----------
    base_settings : mapping
        Regression settings shared by every trial, including the score and
        count inputs.
    destination : path-like
        Directory for trial folders, ``sweep_trials.json``, and the incremental
        ``sweep_results.csv`` table.
    space : SweepSpace or None, optional
        Search space. ``None`` uses :data:`DEFAULT_SWEEP_SPACE` and the built-in
        compatibility filters.
    mode : {'grid', 'random'}, default='random'
        Trial enumeration order passed to :func:`build_trials`.
    max_trials : int, default=1000
        Maximum number of accepted trials.
    seed : int, default=0
        Random-order seed used when ``mode='random'``.
    controls : mapping or None, optional
        Mapping from control aliases to identifiers. Control recovery metrics
        are added to each result row.
    n_jobs : int, default=8
        Requested pool size. :func:`recommended_workers` may reduce it based
        on memory and CPU capacity.
    contained : bool, default=True
        Run each fit in a separate child through
        :func:`run_trial_contained`. Hard memory, swap, task, and CPU limits
        are enforced only when a usable systemd user scope is available;
        otherwise the child is uncapped but uses reduced priority and thread
        limits. Set to ``False`` only when trial resource use is known.
    qc : bool, default=False
        Generate the full regression diagnostic figure suite for every trial.
    progress_every : int, default=25
        Print progress after this many completed trials. Use zero to disable
        periodic progress messages.

    Returns
    -------
    pandas.DataFrame
        One status row per trial, sorted by ``trial_id``. The same rows are
        written incrementally to ``sweep_results.csv``.

    Raises
    ------
    RuntimeError
        If called from a worker process. Scripts must call this function under
        an ``if __name__ == '__main__':`` guard.

    Notes
    -----
    The pool uses the ``spawn`` start method because fitted models may import
    torch and OpenMP runtimes. Only a bounded number of jobs are submitted at
    once, allowing new submissions to pause when available memory is low.
    """
    # A CALLER WITHOUT A MAIN GUARD FORK-BOMBS ITSELF.
    #
    # This pool spawns rather than forks (torch and OpenMP are not safe to
    # fork), and a spawned child re-imports the module it was launched from.
    # If that module is a script whose sweep call sits at top level, every
    # child starts its own sweep, which starts more children. What the user
    # sees is "BrokenProcessPool: A child process terminated abruptly", which
    # says nothing about the actual mistake.
    import multiprocessing

    if multiprocessing.current_process().name != "MainProcess":
        raise RuntimeError(
            "run_sweep_parallel was called from a worker process. The script "
            "that calls it needs an `if __name__ == \"__main__\":` guard -- "
            "without one, each spawned worker re-runs the script and starts "
            "its own sweep.")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Set here, in the parent, rather than in the worker: a spawned child
    # inherits this environment at exec, whereas a pool initializer would run
    # only AFTER the child has imported numpy and torch and they have already
    # sized their thread pools from the core count.
    _pin_threads()

    space = space or SweepSpace()
    if not space.filters:
        space.filters = _default_filters()
    controls = dict(controls or {})
    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)

    # The worker count is a REQUEST, clamped to what the machine can afford.
    # Honouring it literally is what killed the user's editor twice.
    n_jobs, reason = recommended_workers(requested=n_jobs)
    trials = build_trials(space, mode=mode, max_trials=max_trials, seed=seed)
    with open(os.path.join(destination, "sweep_trials.json"), "w",
              encoding="utf-8") as handle:
        json.dump(trials, handle, indent=2, default=str)
    print(f"[sweep] {len(trials)} trials across {n_jobs} workers ({reason})",
          flush=True)

    payloads = [(dict(base_settings), trial, destination, controls,
                 contained, qc)
                for trial in trials]
    rows: list[dict] = []
    started = time.time()
    results_path = os.path.join(destination, "sweep_results.csv")
    # 'spawn', not the default fork: perform_regression imports torch, and a
    # forked child that inherits a torch/OpenMP runtime deadlocks or segfaults
    # rather than failing cleanly.
    context = multiprocessing.get_context("spawn")
    pending = list(payloads)
    done = 0
    paused_for_memory = 0
    last_memory_state: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
        futures = {}

        def _fill():
            """Top the pool up, but never past what memory currently allows.

            Submitting all 5,000 futures at once means the pool decides when
            to start each trial and nothing can intervene. Keeping only
            n_jobs in flight is what lets the memory floor below actually
            stop the sweep growing while the user's editor is running.
            """
            nonlocal paused_for_memory, last_memory_state
            while pending and len(futures) < n_jobs:
                if memory_is_low() and futures:
                    paused_for_memory += 1
                    last_memory_state = dict(_LAST_MEMORY_STATE)
                    return
                payload = pending.pop(0)
                futures[pool.submit(_execute_trial, payload)] = \
                    payload[1]["trial_id"]

        _fill()
        while futures:
            # ``futures`` is non-empty here, so as_completed necessarily
            # yields one item. Taking exactly one lets _fill recheck memory
            # after every completion without an unreachable for-loop exit.
            future = next(as_completed(tuple(futures)))
            trial_id = futures.pop(future)
            done += 1
            try:
                rows.append(_register_resource_workers(future.result()))
            except BaseException as error:  # noqa: BLE001 - dead worker
                rows.append({"trial_id": trial_id, "status": "failed",
                             "error_type": type(error).__name__,
                             "error": str(error)[:300], "seconds": 0.0})
            if progress_every and done % progress_every == 0:
                elapsed = time.time() - started
                remaining = elapsed / done * (len(trials) - done)
                ok = sum(1 for row in rows if row.get("status") == "ok")
                note = (f", paused {paused_for_memory}x for memory"
                        if paused_for_memory else "")
                print(f"[sweep] {done}/{len(trials)} done, {ok} ok, "
                      f"{elapsed / 60:.1f} min elapsed, "
                      f"~{remaining / 60:.1f} min left{note}", flush=True)
            pd.DataFrame(rows).sort_values("trial_id").to_csv(
                results_path, index=False)
            _fill()

    if paused_for_memory:
        own = last_memory_state.get("spacr_tree_gib")
        attribution = (f"; spaCR's process tree was {own:.1f} GiB"
                       if own is not None else "")
        print(f"[sweep] held back {paused_for_memory} times because free "
              f"memory fell below the floor; the sweep yielded rather than "
              f"competing with the rest of the machine{attribution}.",
              flush=True)
    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)


def run_sweep(base_settings: Mapping[str, Any], destination,
               space: SweepSpace | None = None, *,
               mode: str = "grid", max_trials: int = 5000, seed: int = 0,
               controls: Mapping[str, str] | None = None,
               progress_every: int = 10,
               learn_from_failures: int = 2,
               corrections: Sequence[str] | None = None,
               contained: bool = True,
               qc: bool = False,
               memory_floor_gb: float = FREE_MEMORY_FLOOR_GB,
               runner: Callable | None = None) -> pd.DataFrame:
    """Run parameter-sweep trials sequentially.

    Parameters
    ----------
    base_settings : mapping
        Regression settings shared by every trial, including the score and
        count inputs.
    destination : path-like
        Directory for trial folders, ``sweep_trials.json``, and
        ``sweep_results.csv``.
    space : SweepSpace or None, optional
        Search space. ``None`` uses :data:`DEFAULT_SWEEP_SPACE` and the built-in
        compatibility filters.
    mode : {'grid', 'random'}, default='grid'
        Trial enumeration order passed to :func:`build_trials`.
    max_trials : int, default=5000
        Maximum number of accepted trials.
    seed : int, default=0
        Random-order seed used when ``mode='random'``.
    controls : mapping or None, optional
        Mapping from control aliases to identifiers. Control recovery metrics
        are added to each successful result row.
    progress_every : int, default=10
        Print and attempt an incremental CSV write after this many trials. Use
        zero to disable periodic progress messages; a CSV is still written
        after each completed in-process trial.
    learn_from_failures : int, default=2
        Skip later trials with the same model, inference, analysis-unit, and
        penalty signature after this many matching failures. Use zero to run
        every trial.
    corrections : sequence of str or None, optional
        Multiple-testing methods to apply to the p-values from one fitted
        model, producing one row per method. This option applies to the
        in-process path only; contained children return summary rows rather
        than coefficient frames.
    contained : bool, default=True
        Run each trial through :func:`run_trial_contained`. Hard resource
        limits are conditional on a usable systemd user scope; otherwise the
        child is uncapped but uses reduced priority and thread limits.
    qc : bool, default=False
        Generate the full regression diagnostic figure suite for every trial.
    memory_floor_gb : float, default=FREE_MEMORY_FLOOR_GB
        Stop starting contained trials when available memory falls below this
        threshold, in GB.
    runner : callable or None, optional
        In-process regression callable. ``None`` uses contained child trials
        when ``contained=True`` and :func:`spacr.ml.perform_regression`
        otherwise. Injected callables bypass child containment and the memory
        floor.

    Returns
    -------
    pandas.DataFrame
        Trial settings, status, timing, fit metrics, and control metrics. The
        frame is also written to ``sweep_results.csv`` as trials finish.
    """
    if runner is None and contained:
        # Each trial in its own kernel-capped process. This is the default
        # because the alternative -- trusting this module's own accounting --
        # took the user's machine down seven times, and every one of those
        # was a fix to the accounting.
        runner = None
    elif runner is None:
        from .ml import perform_regression as runner  # noqa: PLC0415
    space = space or SweepSpace()
    if not space.filters:
        space.filters = _default_filters()
    controls = dict(controls or {})
    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)

    trials = build_trials(space, mode=mode, max_trials=max_trials, seed=seed)
    manifest = os.path.join(destination, "sweep_trials.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(trials, handle, indent=2, default=str)

    rows: list[dict] = []
    started = time.time()
    # A family that cannot fit this response fails the same way every time --
    # 'poisson' needs integer counts, and a fractional score will never become
    # one. Sampling would rediscover that hundreds of times at full cost, so
    # after `learn_from_failures` identical failures the family is skipped and
    # RECORDED as skipped. The finding is kept; only the repetition is
    # dropped, and setting learn_from_failures=0 turns the shortcut off.
    exhausted: dict[tuple, dict] = {}
    for index, trial in enumerate(trials, start=1):
        signature = (trial.get("regression_type"), trial.get("inference"),
                     trial.get("analysis_unit"), trial.get("alpha"))
        known = exhausted.get(signature)
        if learn_from_failures and known and \
                known["count"] >= learn_from_failures:
            rows.append({
                "trial_id": trial["trial_id"], "folder": None,
                **{k: v for k, v in trial.items() if k != "trial_id"},
                "status": "skipped",
                "error_type": known["error_type"],
                "error": f"same failure as trial {known['first_trial']}",
                "seconds": 0.0,
            })
            continue
        # THE ONE HELPER, not a second copy of it. This branch had its own
        # inline version of _trial_settings and had drifted from it: it never
        # set `regression_qc` at all, so an in-process sweep paid the full
        # ~5.8 s diagnostic suite and ~19 figures on every trial while the
        # parallel branch was trying not to. Two copies of "build one trial's
        # settings" is how a sweep ends up meaning two different things.
        settings, folder = _trial_settings(base_settings, trial, destination,
                                           qc=qc)

        row = {"trial_id": trial["trial_id"], "folder": folder,
               "preparation_key": _preparation_key(settings)}
        row.update({k: v for k, v in trial.items() if k != "trial_id"})
        # Stop BEFORE the machine is in trouble, not once it is.
        if runner is None and contained and free_memory_gb() < memory_floor_gb:
            print(f"[sweep] stopping: {free_memory_gb():.0f} GB free is below "
                  f"the {memory_floor_gb:.0f} GB floor")
            row["status"] = "skipped"
            row["error"] = "stopped at the free-memory floor"
            rows.append(row)
            break

        began = time.time()
        output = None
        if runner is None:
            # Contained: the child returns a finished ROW, not a model.
            child = run_trial_contained(settings, trial_id=trial["trial_id"],
                                        controls=controls)
            _register_resource_workers(child)
            row.update({k: v for k, v in child.items()
                        if k not in ("trial_id",)})
            row["seconds"] = child.get("seconds", round(time.time() - began, 2))
            if row.get("status") != "ok":
                record = exhausted.setdefault(
                    signature, {"count": 0,
                                "error_type": row.get("error_type", "?"),
                                "first_trial": trial["trial_id"]})
                record["count"] += 1
            rows.append(row)
            if progress_every and index % progress_every == 0:
                try:
                    pd.DataFrame(rows).to_csv(
                        os.path.join(destination, "sweep_results.csv"),
                        index=False)
                except OSError:
                    pass
                print(f"[sweep] {index}/{len(trials)} trials "
                      f"({row.get('status')}), "
                      f"{(time.time() - started) / 60:.1f} min elapsed")
            continue

        try:
            output = runner(settings)
            row["status"] = "ok"
            if isinstance(output, Mapping):
                row.update(_count_hits(output))
                # The in-process path skipped the design summary as well as
                # every diagnostic, so an uncontained sweep could not even say
                # how many wells reached the fit.
                row.update(_design_summary(output))
                row.update(summarise_trial(output, settings))
                results = output.get("results")
                if isinstance(results, pd.DataFrame):
                    row.update(_named_control_rows(results, controls))
        except Exception as error:  # noqa: BLE001 - a failed trial is a result
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            row["error"] = str(error).splitlines()[0][:300]
            try:
                with open(os.path.join(folder, "error.txt"), "w",
                          encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
            except OSError:
                pass
            record = exhausted.setdefault(
                signature, {"count": 0, "error_type": row["error_type"],
                            "first_trial": trial["trial_id"]})
            record["count"] += 1
        row["seconds"] = round(time.time() - began, 2)
        if corrections and row.get("status") == "ok" and isinstance(output, Mapping):
            # One row per correction, all from this single fit. Each row still
            # carries every setting, so it reproduces its own regression when
            # the user opens it -- see settings_for_trial.
            for extra in correction_rows(output, corrections,
                                         alpha=float(settings.get("fdr_alpha", 0.05))):
                merged = dict(row)
                merged.update(extra)
                rows.append(merged)
        else:
            if corrections:
                row.setdefault("multiple_testing_method",
                               corrections[0] if corrections else None)
            rows.append(row)

        if progress_every and index % progress_every == 0:
            done = time.time() - started
            rate = done / index
            print(f"[sweep] {index}/{len(trials)} trials "
                  f"({row['status']}), {done / 60:.1f} min elapsed, "
                  f"~{rate * (len(trials) - index) / 60:.1f} min left",
                  flush=True)
        # Written every trial, so a sweep killed halfway still leaves a usable
        # table rather than nothing. The file is a best-effort checkpoint: a
        # full or disconnected results disk must not discard the in-memory
        # rows the sweep can still return to its caller.
        try:
            pd.DataFrame(rows).to_csv(
                os.path.join(destination, "sweep_results.csv"), index=False)
        except OSError:
            pass

    return pd.DataFrame(rows)


def rank_trials(results: pd.DataFrame, *, role: str = "positive"
                ) -> pd.DataFrame:
    """Order trials by recovery of a named control role.

    Parameters
    ----------
    results : pandas.DataFrame
        Sweep results containing ``<role>_control_percentile`` and optionally
        ``status``.
    role : {'positive', 'negative'}, default='positive'
        Control role whose percentile determines the ordering.

    Returns
    -------
    pandas.DataFrame
        Copy ordered by increasing control percentile, with missing values and
        failed trials last. The input object is returned unchanged if the
        percentile column is absent or contains no finite values.

    Notes
    -----
    Percentile is used instead of raw rank so trials that fit different
    numbers of coefficients remain comparable. Trials that did not recover
    the control remain in the table rather than being dropped.
    """
    percentile = f"{role}_control_percentile"
    if results is None or not len(results) or percentile not in results.columns:
        return results
    frame = results.copy()
    key = pd.to_numeric(frame[percentile], errors="coerce")
    if not key.notna().any():
        return results
    # NaN last, and a failed trial never outranks one that ran.
    ran = (frame["status"] == "ok") if "status" in frame.columns else True
    frame["_sort_key"] = key.where(ran, other=np.nan)
    ordered = frame.sort_values(
        "_sort_key", ascending=True, na_position="last", kind="stable")
    return ordered.drop(columns=["_sort_key"]).reset_index(drop=True)


def summarise_sweep(results: pd.DataFrame, *,
                     controls: Sequence[str] = ("gra14", "eaf1")) -> dict:
    """Summarize sweep completion, controls, and hit-count sensitivity.

    Parameters
    ----------
    results : pandas.DataFrame
        Result rows produced by :func:`run_sweep` or
        :func:`run_sweep_parallel`.
    controls : sequence of str, default=('gra14', 'eaf1')
        Control aliases whose presence and rank columns should be summarized
        when available.

    Returns
    -------
    dict
        Trial counts, elapsed minutes, failure categories, control recovery,
        and hit-count ranges. Median hit counts are also grouped by correction,
        regression family, analysis unit, and inference mode when those
        columns are present. An empty input returns ``{'trials': 0}``.

    Notes
    -----
    The summary emphasizes consistency across defensible analysis choices.
    Hit counts alone should not be used to select a model or correction.
    """
    if results.empty:
        return {"trials": 0}
    ok = results[results["status"] == "ok"]
    summary: dict[str, Any] = {
        "trials": int(len(results)),
        "succeeded": int(len(ok)),
        "failed": int((results["status"] == "failed").sum()),
        "total_minutes": round(float(results["seconds"].sum()) / 60, 1),
    }
    if "error_type" in results.columns:
        summary["failure_reasons"] = (
            results.loc[results["status"] == "failed", "error_type"]
            .value_counts().to_dict())
    for control in controls:
        column = f"{control}_present"
        if column in ok.columns and len(ok):
            summary[f"{control}_recovered_in"] = (
                f"{int(ok[column].sum())}/{len(ok)} trials")
        rank_column = f"{control}_rank"
        if rank_column in ok.columns and ok[rank_column].notna().any():
            summary[f"{control}_median_rank"] = float(
                ok[rank_column].median())
    # THE ANSWER, WHEN THE SCREEN HAS A YARDSTICK. Stated before the hit
    # counts, because a configuration that loses the positive control is not
    # improved by reporting more hits.
    if "positive_control_rank" in ok.columns and len(ok):
        found = ok[pd.to_numeric(
            ok["positive_control_rank"], errors="coerce").notna()]
        summary["positive_control_recovered_in"] = f"{len(found)}/{len(ok)} trials"
        if len(found):
            best = rank_trials(found).iloc[0]
            summary["positive_control_best_rank"] = int(
                best["positive_control_rank"])
            summary["positive_control_best_trial"] = int(best["trial_id"]) \
                if "trial_id" in best else None
            summary["positive_control_median_rank"] = float(pd.to_numeric(
                found["positive_control_rank"], errors="coerce").median())
    if "n_below_alpha" in ok.columns and len(ok):
        summary["hits_median"] = float(ok["n_below_alpha"].median())
        summary["hits_range"] = [int(ok["n_below_alpha"].min()),
                                 int(ok["n_below_alpha"].max())]
        # The spread across settings IS the result: a screen whose hit count
        # ranges from 2 to 400 depending on the correction has not been
        # analysed, it has been chosen.
        for axis in ("multiple_testing_method", "regression_type",
                     "analysis_unit", "inference"):
            if axis in ok.columns:
                summary[f"hits_by_{axis}"] = (
                    ok.groupby(axis)["n_below_alpha"]
                    .median().sort_values(ascending=False).to_dict())
    return summary


#: Columns a results row carries that describe the RUN rather than a setting.
#: Everything else in a row was a setting the trial was given, which is what
#: makes a row enough to reproduce the trial it describes.
#:
#: EVERY MEASURED COLUMN MUST BE IN HERE. The rule "anything not listed was a
#: setting" lets custom sweep axes round-trip, but also means an omitted
#: measurement would be replayed as a regression setting. Metric names come
#: from :data:`trial_metrics.METRIC_COLUMNS` so the producer and replay filter
#: stay aligned.
_BOOKKEEPING_COLUMNS = frozenset({
    "trial_id", "folder", "preparation_key", "status", "seconds",
    "error", "error_type",
}) | _METRIC_COLUMNS


def settings_for_trial(base_settings: Mapping[str, Any], row: Mapping[str, Any],
                       *, destination: str | None = None) -> dict:
    """The full settings dict that produced ``row``.

    A sweep row is not just a record of what happened -- it carries every
    setting the trial was given, which is what lets a user click a row and get
    that exact regression back rather than an approximation of it.

    Values arrive as strings when the row came from the CSV rather than from
    memory, so they are parsed back to the types spaCR expects. A setting that
    will not parse is passed through unchanged: a string that was always a
    string must survive the round trip.

    :param base_settings: the inputs the sweep ran on (score/count CSVs and
        the response column), which are not recorded per trial.
    :param row: one row of the sweep results table.
    :param destination: where to write this run's output. Defaults to the
        folder the trial originally used.
    """
    import ast

    # The control-alias columns cannot be listed in advance, because the
    # aliases are the CALLER'S: run_sweep(controls={"gra14": "239740"}) makes
    # gra14_rank, gra14_q and the rest. They are recoverable exactly, though,
    # because _named_control_rows always writes `{alias}_present` for every
    # alias whether or not it found one -- so the row names its own aliases and
    # nothing has to be guessed from suffixes. Guessing would be wrong anyway:
    # spaCR has twenty-two real settings ending in `_percentile`.
    aliases = [key[: -len("_present")] for key in row
               if isinstance(key, str) and key.endswith("_present")]
    alias_columns = {f"{alias}{suffix}" for alias in aliases
                     for suffix in ("_present", "_effect", "_rank", "_q", "_p")}

    settings = dict(base_settings)
    for key, value in row.items():
        if key in _BOOKKEEPING_COLUMNS or key in alias_columns:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # genuinely a string
        settings[key] = value

    folder = destination or row.get("folder")
    if folder and not (isinstance(folder, float) and pd.isna(folder)):
        # No mkdir here: this builds a settings dict and nothing else, so it
        # stays callable from a test, a dry run or a preview without leaving
        # directories behind. rerun_trial creates the folder it writes to.
        settings["src"] = str(folder)
    settings.setdefault("Toxoplasma", False)
    return settings


def rerun_trial(base_settings: Mapping[str, Any], row: Mapping[str, Any],
                *, destination: str | None = None) -> dict:
    """Re-run one trial and hand back its settings, output and FIGURES.

    The figures are live matplotlib Figures, not paths: a saved page cannot be
    restyled, and the point of clicking a row is to look at that condition
    properly -- change the thresholds, recolour it, fix the legend -- rather
    than to be shown a picture of it.

    Only figures this call created are returned. A screen that already has
    figures open must not have them swept up and re-attributed to a trial they
    did not come from.
    """
    import matplotlib.pyplot as plt

    settings = settings_for_trial(base_settings, row, destination=destination)
    # Plots are the entire reason for this call.
    settings["verbose"] = True
    folder = settings.get("src")
    if folder:
        os.makedirs(folder, exist_ok=True)

    before = set(plt.get_fignums())
    from .ml import perform_regression

    output = perform_regression(settings)
    # THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:
    # rcParams reach an artist when it is CREATED, so a
    # context opened after `plt.subplots` would leave the
    # spines, ticks and labels at the caller's globals.
    with figure_style(theme_target()):
        figures = [plt.figure(number) for number in plt.get_fignums()
                   if number not in before]
        return {"settings": settings, "output": output, "figures": figures}
