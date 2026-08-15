"""Sweep regression settings and find out which ones change the answer.

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

The search space is declared as data (:data:`DEFAULT_SEARCH_SPACE`), so adding
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
    "DEFAULT_SEARCH_SPACE",
    "PREPARATION_KEYS",
    "SearchSpace",
    "build_trials",
    "run_search",
    "summarise_search",
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
DEFAULT_SEARCH_SPACE: dict[str, list] = {
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
class SearchSpace:
    """Axes to sweep, plus settings pinned for every trial."""

    axes: dict[str, list] = field(
        default_factory=lambda: dict(DEFAULT_SEARCH_SPACE))
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


def build_trials(space: SearchSpace, *, mode: str = "grid",
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


def run_search_parallel(base_settings: Mapping[str, Any], destination,
                        space: "SearchSpace | None" = None, *,
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

    space = space or SearchSpace()
    if not space.filters:
        space.filters = _default_filters()
    controls = dict(controls or {})
    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)

    trials = build_trials(space, mode=mode, max_trials=max_trials, seed=seed)
    with open(os.path.join(destination, "search_trials.json"), "w",
              encoding="utf-8") as handle:
        json.dump(trials, handle, indent=2, default=str)
    print(f"[search] {len(trials)} trials across {n_jobs} workers", flush=True)

    payloads = [(dict(base_settings), trial, destination, controls)
                for trial in trials]
    rows: list[dict] = []
    started = time.time()
    results_path = os.path.join(destination, "search_results.csv")
    # 'spawn', not the default fork: perform_regression imports torch, and a
    # forked child that inherits a torch/OpenMP runtime deadlocks or segfaults
    # rather than failing cleanly.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as pool:
        futures = {pool.submit(_execute_trial, payload): payload[1]["trial_id"]
                   for payload in payloads}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001 - a dead worker is a row
                rows.append({"trial_id": futures[future], "status": "failed",
                             "error_type": type(error).__name__,
                             "error": str(error)[:300], "seconds": 0.0})
            if progress_every and done % progress_every == 0:
                elapsed = time.time() - started
                remaining = elapsed / done * (len(trials) - done)
                ok = sum(1 for row in rows if row.get("status") == "ok")
                print(f"[search] {done}/{len(trials)} done, {ok} ok, "
                      f"{elapsed / 60:.1f} min elapsed, "
                      f"~{remaining / 60:.1f} min left", flush=True)
            pd.DataFrame(rows).sort_values("trial_id").to_csv(
                results_path, index=False)

    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)


def run_search(base_settings: Mapping[str, Any], destination,
               space: SearchSpace | None = None, *,
               mode: str = "grid", max_trials: int = 5000, seed: int = 0,
               controls: Mapping[str, str] | None = None,
               progress_every: int = 10,
               learn_from_failures: int = 2,
               runner: Callable | None = None) -> pd.DataFrame:
    """Run every trial and return one tidy row per trial.

    :param base_settings: settings shared by every trial -- at minimum the
        score and count inputs.
    :param destination: folder to write trial folders and the summary into.
    :param space: what to sweep. Defaults to :data:`DEFAULT_SEARCH_SPACE`.
    :param controls: ``{alias: identifier}`` looked up in each trial's results,
        e.g. ``{"gra14": "239740", "eaf1": "225160"}``. Recovering the known
        positive control is the yardstick a sweep is read against.
    :param runner: injected for testing; defaults to
        :func:`spacr.ml.perform_regression`.
    :returns: the summary frame, also written as ``search_results.csv``.
    """
    if runner is None:
        from .ml import perform_regression as runner  # noqa: PLC0415
    space = space or SearchSpace()
    if not space.filters:
        space.filters = _default_filters()
    controls = dict(controls or {})
    destination = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    os.makedirs(destination, exist_ok=True)

    trials = build_trials(space, mode=mode, max_trials=max_trials, seed=seed)
    manifest = os.path.join(destination, "search_trials.json")
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
            print(f"[search] {index}/{len(trials)} trials "
                  f"({row['status']}), {done / 60:.1f} min elapsed, "
                  f"~{rate * (len(trials) - index) / 60:.1f} min left",
                  flush=True)
        # Written every trial, so a sweep killed halfway still leaves a usable
        # table rather than nothing.
        pd.DataFrame(rows).to_csv(
            os.path.join(destination, "search_results.csv"), index=False)

    return pd.DataFrame(rows)


def summarise_search(results: pd.DataFrame, *,
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
