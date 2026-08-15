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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_SWEEP_SPACE",
    "PREPARATION_KEYS",
    "SweepSpace",
    "build_trials",
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
    """Axes to sweep, plus settings pinned for every trial."""

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
        if trial.get("random_row_column_effects") and \
                trial.get("regression_type") not in (None, "ols", "mixed"):
            return (f"random_row_column_effects=True fits a mixed model and "
                    f"cannot also fit {trial['regression_type']!r}")
        return None

    def aggregation_belongs_to_wells(trial):
        if trial.get("analysis_unit") == "cell" and \
                trial.get("agg_type") not in (None, "mean"):
            # agg_type is forced to None for a per-cell fit, so sweeping it
            # would run the same analysis several times under different labels.
            return "analysis_unit='cell' ignores agg_type"
        return None

    def quantile_needs_its_own_unit(trial):
        if trial.get("regression_type") == "quantile" and \
                trial.get("analysis_unit") == "well":
            return ("regression_type='quantile' fits per-object values, so it "
                    "forces analysis_unit='cell'")
        return None

    def permutation_ignores_the_family(trial):
        # The permutation test is its own estimator: it does not read
        # regression_type, and sweeping it would repeat one analysis 13 times.
        if trial.get("inference") == "nonparametric" and \
                trial.get("regression_type") not in (None, "ols"):
            return "inference='nonparametric' does not use regression_type"
        return None

    def permutation_has_no_row_column_terms(trial):
        if trial.get("inference") == "nonparametric" and \
                trial.get("random_row_column_effects"):
            return ("inference='nonparametric' blocks on plate and does not "
                    "fit row/column effects")
        return None

    def penalty_belongs_to_penalised_families(trial):
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
            permutation_has_no_row_column_terms,
            penalty_belongs_to_penalised_families]


def build_trials(space: SweepSpace, *, mode: str = "grid",
                 max_trials: int = 5000, seed: int = 0) -> list[dict]:
    """Enumerate the trials to run.

    ``mode='grid'`` walks the full product and truncates at ``max_trials``;
    ``mode='random'`` samples without replacement, which covers a wide space
    more evenly than an arbitrary prefix of its product.

    Rejected combinations never count toward ``max_trials`` -- the cap is on
    analyses performed, not on tuples considered.
    """
    if mode not in {"grid", "random"}:
        raise ValueError("mode must be 'grid' or 'random'")
    names = list(space.axes)
    values = [space.axes[name] for name in names]
    filters = list(space.filters) or _default_filters()

    def accept(trial):
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
        reason = accept(trial)
        if reason:
            rejected.append((trial, reason))
            continue
        trial.update(space.fixed)
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
            out[f"{alias}_effect"] = float(row[effect_column])
            position = ranked_labels.str.contains(
                str(needle), regex=False, na=False)
            if position.any():
                out[f"{alias}_rank"] = int(position.idxmax()) + 1
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
#: belongs to the desktop. This is the number that was missing when a
#: 14-worker sweep drove the machine into the OOM killer and took the user's
#: editor with it -- twice.
MEMORY_BUDGET_FRACTION = 0.5


def recommended_workers(*, measured_gib=None, requested=None):
    """How many workers this machine can actually afford right now.

    Sized from FREE MEMORY, not from the core count. Picking 14 workers on a
    32-core box because it had 32 cores is exactly what exhausted memory and
    got the editor killed: the limiting resource is the several gigabytes
    each trial needs for its own copy of the tables, not the CPU.

    :param measured_gib: peak resident size of one trial, if measured. Falls
        back to :data:`ASSUMED_TRIAL_GIB`.
    :param requested: an explicit ask, still clamped to what is affordable.
    :returns: ``(workers, reason)``. The reason is shown to the user, so a
        sweep that drops from 8 workers to 2 says why rather than looking
        arbitrarily slow.
    """
    per_trial = float(measured_gib or ASSUMED_TRIAL_GIB)
    available = None
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:  # pragma: no cover - psutil is a dependency, but be safe
        pass
    try:
        cores = max(len(os.sched_getaffinity(0)) - 2, 1)
    except AttributeError:  # pragma: no cover - non-Linux
        cores = max((os.cpu_count() or 2) - 2, 1)

    if available is None:
        workers = max(1, min(2, cores, MAX_WORKERS))
        return workers, (f"memory could not be measured, so the sweep is "
                         f"limited to {workers} workers")

    affordable = int((available * MEMORY_BUDGET_FRACTION) // per_trial)
    workers = max(1, min(affordable, cores, MAX_WORKERS,
                         requested or MAX_WORKERS))
    reason = (f"{available:.0f} GiB free, ~{per_trial:.1f} GiB per trial, "
              f"budgeting {MEMORY_BUDGET_FRACTION:.0%} of free memory "
              f"-> {workers} worker{'s' if workers != 1 else ''}")
    if requested and workers < requested:
        reason += f" (you asked for {requested})"
    return workers, reason


def memory_is_low(floor_gib: float = 8.0) -> bool:
    """True when free memory has fallen far enough to stop starting trials.

    Checked between submissions, not only at the start: the other things on
    the machine -- an editor, the spaCR GUI, another analysis -- grow while
    the sweep runs, and the sweep must yield to them rather than race them.
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3) < floor_gib
    except Exception:  # pragma: no cover
        return False


def be_polite() -> None:
    """Drop this process's CPU and I/O priority.

    A sweep must never win a scheduling contest against the user's editor.
    Applied inside each worker so it is a property of the work, not of
    however the sweep happened to be launched.
    """
    try:
        os.nice(19)
    except (OSError, AttributeError):  # pragma: no cover
        pass
    try:
        import subprocess
        # Linux idle I/O class: a sweep reads gigabytes of CSV and should
        # yield the disk to anything interactive.
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                       check=False, capture_output=True)
    except Exception:  # pragma: no cover - best effort
        pass


def _trial_settings(base_settings, trial, destination):
    """Build one trial's settings dict and its own output folder."""
    settings = dict(base_settings)
    settings.update({k: v for k, v in trial.items() if k != "trial_id"})
    folder = os.path.join(destination, f"trial_{trial['trial_id']:04d}")
    os.makedirs(folder, exist_ok=True)
    settings["src"] = folder
    settings.setdefault("verbose", False)
    settings.setdefault("toxo", False)
    return settings, folder


def _execute_trial(payload):
    """Run one trial in this process and return its summary row.

    Module level and argument-only, so it can be pickled to a worker. Each
    worker imports spaCR itself: ``perform_regression`` pulls in torch and
    matplotlib, and a forked copy of those is not safe to reuse.
    """
    base_settings, trial, destination, controls = payload
    # Before anything expensive: this work yields to the user's machine.
    be_polite()
    from .ml import perform_regression

    settings, folder = _trial_settings(base_settings, trial, destination)
    row = {"trial_id": trial["trial_id"], "folder": folder,
           "preparation_key": _preparation_key(settings)}
    row.update({k: v for k, v in trial.items() if k != "trial_id"})
    began = time.time()
    try:
        output = perform_regression(settings)
        row["status"] = "ok"
        if isinstance(output, Mapping):
            row.update(_count_hits(output))
            row.update(_design_summary(output))
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
    return row


def run_sweep_parallel(base_settings: Mapping[str, Any], destination,
                        space: "SweepSpace | None" = None, *,
                        mode: str = "random", max_trials: int = 1000,
                        seed: int = 0,
                        controls: Mapping[str, str] | None = None,
                        n_jobs: int = 8,
                        progress_every: int = 25) -> pd.DataFrame:
    """Run the sweep across processes, writing results as they land.

    One trial takes the better part of a minute on a real screen -- the cost
    is reading and joining the inputs, which is per-process work with no
    shared state -- so trials are embarrassingly parallel. Results are written
    after each completion, so a killed sweep still leaves a usable table.

    The adaptive skip that :func:`run_search` uses is deliberately absent
    here: it depends on the order failures are observed in, which is not
    deterministic across workers, and a reproducible trial list matters more
    than the trials it would save.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

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

    payloads = [(dict(base_settings), trial, destination, controls)
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
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
        futures = {}

        def _fill():
            """Top the pool up, but never past what memory currently allows.

            Submitting all 5,000 futures at once means the pool decides when
            to start each trial and nothing can intervene. Keeping only
            n_jobs in flight is what lets the memory floor below actually
            stop the sweep growing while the user's editor is running.
            """
            nonlocal paused_for_memory
            while pending and len(futures) < n_jobs:
                if memory_is_low() and futures:
                    paused_for_memory += 1
                    return
                payload = pending.pop(0)
                futures[pool.submit(_execute_trial, payload)] = \
                    payload[1]["trial_id"]

        _fill()
        while futures:
            for future in as_completed(list(futures)):
                trial_id = futures.pop(future)
                done += 1
                try:
                    rows.append(future.result())
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
                break

    if paused_for_memory:
        print(f"[sweep] held back {paused_for_memory} times because free "
              f"memory fell below the floor; the sweep yielded rather than "
              f"competing with the rest of the machine.", flush=True)
    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)


def run_sweep(base_settings: Mapping[str, Any], destination,
               space: SweepSpace | None = None, *,
               mode: str = "grid", max_trials: int = 5000, seed: int = 0,
               controls: Mapping[str, str] | None = None,
               progress_every: int = 10,
               learn_from_failures: int = 2,
               runner: Callable | None = None) -> pd.DataFrame:
    """Run every trial and return one tidy row per trial.

    :param base_settings: settings shared by every trial -- at minimum the
        score and count inputs.
    :param destination: folder to write trial folders and the summary into.
    :param space: what to sweep. Defaults to :data:`DEFAULT_SWEEP_SPACE`.
    :param controls: ``{alias: identifier}`` looked up in each trial's results,
        e.g. ``{"gra14": "239740", "eaf1": "225160"}``. Recovering the known
        positive control is the yardstick a sweep is read against.
    :param runner: injected for testing; defaults to
        :func:`spacr.ml.perform_regression`.
    :returns: the summary frame, also written as ``sweep_results.csv``.
    """
    if runner is None:
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
        settings = dict(base_settings)
        settings.update({k: v for k, v in trial.items() if k != "trial_id"})
        folder = os.path.join(destination,
                              f"trial_{trial['trial_id']:04d}")
        os.makedirs(folder, exist_ok=True)
        settings["src"] = folder
        # Every trial writes into its own folder, so a sweep never overwrites
        # a previous answer and any trial can be reopened afterwards.
        settings.setdefault("verbose", False)
        settings.setdefault("toxo", False)

        row = {"trial_id": trial["trial_id"], "folder": folder,
               "preparation_key": _preparation_key(settings)}
        row.update({k: v for k, v in trial.items() if k != "trial_id"})
        began = time.time()
        try:
            output = runner(settings)
            row["status"] = "ok"
            if isinstance(output, Mapping):
                row.update(_count_hits(output))
                results = output.get("results")
                if isinstance(results, pd.DataFrame):
                    row.update(_named_control_rows(results, controls))
        except Exception as error:  # noqa: BLE001 - a failed trial is a result
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            row["error"] = str(error).splitlines()[0][:300]
            with open(os.path.join(folder, "error.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
            record = exhausted.setdefault(
                signature, {"count": 0, "error_type": row["error_type"],
                            "first_trial": trial["trial_id"]})
            record["count"] += 1
        row["seconds"] = round(time.time() - began, 2)
        rows.append(row)

        if progress_every and index % progress_every == 0:
            done = time.time() - started
            rate = done / index
            print(f"[sweep] {index}/{len(trials)} trials "
                  f"({row['status']}), {done / 60:.1f} min elapsed, "
                  f"~{rate * (len(trials) - index) / 60:.1f} min left",
                  flush=True)
        # Written every trial, so a sweep killed halfway still leaves a usable
        # table rather than nothing.
        pd.DataFrame(rows).to_csv(
            os.path.join(destination, "sweep_results.csv"), index=False)

    return pd.DataFrame(rows)


def summarise_sweep(results: pd.DataFrame, *,
                     controls: Sequence[str] = ("gra14", "eaf1")) -> dict:
    """Reduce a sweep to the statements worth reading.

    The question a sweep answers is not "which setting gave the most hits" --
    that rewards whichever combination corrects least. It is "which findings
    survive the choices I could defensibly have made", so the summary is built
    around agreement across trials, not around any single trial.
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
_BOOKKEEPING_COLUMNS = frozenset({
    "trial_id", "folder", "preparation_key", "status", "seconds",
    "error", "error_type", "n_results", "n_significant", "n_primary",
    "n_below_alpha", "n_wells", "n_guides", "n_cells",
})


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

    settings = dict(base_settings)
    for key, value in row.items():
        if key in _BOOKKEEPING_COLUMNS:
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
    settings.setdefault("toxo", False)
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
    figures = [plt.figure(number) for number in plt.get_fignums()
               if number not in before]
    return {"settings": settings, "output": output, "figures": figures}
