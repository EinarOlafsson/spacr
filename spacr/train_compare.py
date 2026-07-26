"""Overlay several training runs' curves and say exactly what differed between them.

Why this exists
---------------
Picking a classifier in spaCR means training it, writing down the number,
changing a knob, training it again, and then trying to remember which of the
six things you touched is the one that moved the number. The provenance diff
(:func:`spacr.run_journal.diff_runs`) answers "what changed"; this module
answers "and what did it do to the curves", on one axis, next to the diff.

What is on disk, and therefore what can be shown
------------------------------------------------
:func:`spacr.deep_spacr.train_model` evaluates train (and validation, when a
validation loader exists) at the end of every epoch and hands the metric dicts
to :func:`spacr.io._save_progress`, which appends them to two CSVs in the run's
``dst`` folder::

    <src>/model/<model_type>/<channels>/epochs_<N>/
        train.csv          # one row per epoch
        validation.csv     # one row per epoch, only when val_loaders exist
        <model_type>_epoch_<e>_channels_<ch>.pth

Columns come from :func:`spacr.deep_spacr.evaluate_model_performance`:
``epoch``, ``loss``, ``accuracy`` (and its duplicate ``Accuracy``),
``neg_accuracy``, ``pos_accuracy``, ``prauc``, ``optimal_threshold``, plus
``train_time`` on the train side and ``num_classes`` on the multiclass side.

A k-fold run (``cross_validation_folds >= 2``) trains a fresh model per fold in
``<dst>/fold_<i>/``, so it produces **k independent pairs of curves**, plus the
per-fold summary CSVs :func:`spacr.deep_spacr._cross_validate_model` writes at
the ``dst`` level.

The settings that produced a run are *not* in the run folder: ``save_settings``
writes them to ``<src>/settings/train_test_<model_type>_<epochs>.csv`` as a
``Key,Value`` CSV. :func:`load_run` walks up from the run folder to find it —
but only as far as the run's own project root, which the layout above makes
identifiable: ``<src>`` is the folder the ``model/`` tree hangs off. An
ancestor above that belongs to a different project, and reading it would report
somebody else's settings as this run's provenance, so the settings diff would
show differences nobody configured and the answer would depend on where in the
filesystem the project happened to sit. See :func:`_owns_this_run`.

Judgement calls baked in
------------------------
**Runs of different length are never aligned.** Each series is plotted over its
own epoch axis. Truncating a 60-epoch run to match a 25-epoch neighbour invents
a result ("B never got better") and padding invents a different one; both are
worse than a legend that says one run is longer.

**Train and validation are never silently mixed.** Every series label carries
run, split and fold, and :func:`plot_curves` annotates the axes when more than
one split is on it. A train curve above a validation curve is the definition of
overfitting, not evidence that one run beat another.

**k-fold runs keep their folds.** ``folds='per_fold'`` (the default) draws every
fold; ``folds='mean'`` draws the fold mean with a ±1 sd band and labels it
``mean of k folds ±sd``; ``folds='both'`` draws both. A mean curve rendered as
if it were a single run hides precisely the fold-to-fold variance k-fold was
added to expose — and because folds can stop at different epochs (early
stopping), the mean carries an ``n_folds`` column so a tail computed from two of
five folds is visible rather than implied.

**Best-epoch and last-epoch are both reported, always.** The best epoch of a
validation curve is chosen *using* that curve, so it is an optimistically biased
estimate of held-out performance; the last epoch is unbiased but may be well
past the optimum. Reporting either one alone is a way to be wrong.

**The settings diff reuses the provenance bucketing.** A flat key-by-key diff of
two real runs reports ~200 differences, of which none are decisions anybody
made. So this module reuses :func:`spacr.run_journal.values_equal` (structural
comparison, so ``"[0, 1, 2]"`` from a CSV round-trip equals ``[0, 1, 2]``) and
the same three buckets:

``changed``
    keys present in *every* run whose values differ — the knobs someone turned.
``env``
    environment / non-substantive drift: paths, timestamps, host, versions,
    worker counts, verbosity. Shown, never mixed into ``changed``.
``drift``
    keys absent from at least one run — schema drift between spaCR releases,
    summarised rather than enumerated.

Two runs whose settings match say "no differences" in so many words; an empty
table reads like a failure.

**Broken run folders are reported, not fatal.** A folder with no curves, no
settings, or a header-only (zero-epoch) log is loaded anyway, carries a note
saying what is wrong, and does not stop the other runs being compared.

This module does not import torch. It reads CSVs.

Example::

    from spacr.train_compare import find_runs, compare_runs, format_comparison

    runs = find_runs('/data/screen1/model')
    cmp = compare_runs(runs, folds='both')
    print(format_comparison(cmp, metric='accuracy'))
    fig = plot_curves(cmp, 'accuracy')

See Also:
    :func:`spacr.run_journal.diff_runs` — the two-run provenance diff whose
    bucketing this module reuses and generalises to N runs.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reused from the provenance diff rather than re-derived. ``values_equal`` is
# the whole reason the diff is usable (structural comparison across the JSON /
# CSV / live-dict round-trips a settings dict takes), and the renderers keep the
# console output of the two features identical. They are private in
# run_journal because they are not a public API; this is the same feature.
from .run_journal import (
    _drift_names,
    _read_settings_csv,
    _render_change_pair,
    _render_value,
    values_equal,
)

__all__ = [
    "TrainingRun",
    "Series",
    "Comparison",
    "load_run",
    "find_runs",
    "compare_runs",
    "plot_curves",
    "format_comparison",
    "diff_settings",
    "available_metrics",
    "render_setting_value",
    "metric_direction",
    "is_env_key",
    "FOLD_MODES",
    "SPLIT_FILES",
    "ENV_SETTING_KEYS",
    "ENV_KEY_TOKENS",
]


# ---------------------------------------------------------------------------
# On-disk layout constants
# ---------------------------------------------------------------------------

#: split name -> filename written by :func:`spacr.io._save_progress`.
SPLIT_FILES: Dict[str, str] = {"train": "train.csv", "val": "validation.csv"}

#: ``<dst>/fold_<i>/`` — one per cross-validation fold.
_FOLD_RE = re.compile(r"^fold_(\d+)$")

#: Column aliases, mirroring ``spacr.utils.suggest_training_changes``'s
#: ``_normalize_cols`` so the two readers of these CSVs agree on names.
_COLUMN_ALIASES = {
    "acc": "accuracy",
    "train_acc": "accuracy",
    "val_acc": "accuracy",
    "train_loss": "loss",
    "val_loss": "loss",
    "macro_f1": "f1_macro",
    "f1macro": "f1_macro",
    "epochs": "epoch",
    "step": "epoch",
    "learning_rate": "lr",
}

#: Columns that identify a row rather than measure anything.
_IDENTITY_COLUMNS = frozenset({"epoch", "run_id", "split", "fold", "n_folds"})

#: Settings keys that record the machine rather than a modelling decision.
#: ``dst`` is derived from ``src``/``model_type``/``channels``/``epochs``, so it
#: only ever restates a difference already reported; the rest change nothing
#: about the fitted model. Override via ``compare_runs(env_keys=...)``.
ENV_SETTING_KEYS = frozenset({
    "dst", "n_jobs", "pin_memory", "verbose", "plot", "test_mode",
})

#: ``_``-separated key tokens that mark a value as environment drift. Matched
#: token-wise, not by substring, so ``update_freq`` is not mistaken for a date.
ENV_KEY_TOKENS = frozenset({
    "time", "timestamp", "date", "host", "hostname", "user", "version",
    "git", "platform", "python", "cuda", "device", "gpu", "machine", "node",
})

#: How many parent folders :func:`load_run` climbs looking for ``settings/``.
#: Five is the deepest a real run sits below its project root
#: (``<src>/model/<model_type>/<channels>/epochs_<N>/fold_<i>``); the sixth is
#: slack. Which of those ancestors may actually be *read* is decided by
#: :func:`_owns_this_run`, not by this number.
_SETTINGS_SEARCH_DEPTH = 6

#: The directory :func:`spacr.deep_spacr.train_test_model` roots every run
#: under — ``dst = <src>/model/<model_type>/<channels>/epochs_<N>``. It is what
#: makes the project root identifiable from the run folder alone, and so what
#: bounds the settings climb; see :func:`_owns_this_run`.
_RUN_TREE_ROOT_DIR = "model"

#: Default recursion depth for :func:`find_runs`.
DEFAULT_SCAN_DEPTH = 6


def metric_direction(name: Any) -> Optional[str]:
    """Return ``'max'``, ``'min'`` or ``None`` for a metric column name.

    ``None`` means "no meaningful best" — ``optimal_threshold`` and
    ``train_time`` are recorded per epoch but neither has a direction, and
    calling the largest one "best" would be a fabrication.
    """
    n = str(name).strip().lower()
    if "loss" in n or "error" in n:
        return "min"
    for token in ("accuracy", "acc", "prauc", "auc", "f1", "precision",
                  "recall", "iou", "kappa", "dice"):
        if token in n:
            return "max"
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingRun:
    """One training run's curves, settings and complaints.

    :ivar run_id: short human id, unique within a :func:`find_runs` result.
    :ivar path: the run folder (``dst``) — the folder holding ``train.csv`` /
        ``validation.csv``, or the folder holding the ``fold_<i>/`` subfolders.
    :ivar settings: the settings dict recovered from
        ``<src>/settings/*.csv`` (or a journal ``settings.json``); ``{}`` when
        none was found, which is recorded in :attr:`notes`.
    :ivar curves: long-form per-epoch metrics with identity columns
        ``run_id``, ``split``, ``fold`` and ``epoch``, one row per logged
        epoch. Empty (with the identity columns present) when the run logged
        nothing.
    :ivar folds: fold folder names (``['fold_1', ...]``), empty for a run that
        used a single train/validation split.
    :ivar final_metrics: ``{series_label: {...}}`` — per split and fold, the
        epoch count and **both** the last-epoch and best-epoch value of every
        metric. See :func:`format_comparison` for why both.
    :ivar notes: everything wrong with, or worth knowing about, this folder.
        A run with notes is still comparable.
    :ivar settings_path: where the settings came from, or ``''``.
    :ivar manifest: a run-journal ``manifest.json`` when the folder happens to
        be one; ``{}`` for an ordinary training ``dst``.
    """
    run_id: str
    path: Path
    settings: Dict[str, Any] = field(default_factory=dict)
    curves: pd.DataFrame = field(default_factory=lambda: _empty_curves())
    folds: List[str] = field(default_factory=list)
    final_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    settings_path: str = ""
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        """Highest epoch number logged by any split/fold (0 when nothing was)."""
        if self.curves.empty:
            return 0
        return int(pd.to_numeric(self.curves["epoch"], errors="coerce").max())

    @property
    def has_curves(self) -> bool:
        return not self.curves.empty

    @property
    def is_cv(self) -> bool:
        return bool(self.folds)

    def metrics(self) -> List[str]:
        """Numeric metric columns this run actually logged, sorted."""
        return sorted(c for c in self.curves.columns
                      if c not in _IDENTITY_COLUMNS)

    def summary_line(self) -> str:
        """One line for a list widget: id, shape of the log, key settings."""
        if self.has_curves:
            shape = f"{self.n_epochs} epochs"
            if self.folds:
                shape += f" x {len(self.folds)} folds"
            splits = sorted(set(self.curves["split"]))
            shape += f" · {'+'.join(splits)}"
        else:
            shape = "no curves"
        bits = [f"{self.run_id}", shape]
        for key in ("model_type", "learning_rate", "batch_size", "loss_type"):
            if key in self.settings:
                bits.append(f"{key}={_render_value(self.settings[key], 18)}")
        if self.notes:
            bits.append(f"! {len(self.notes)} note"
                        f"{'s' if len(self.notes) > 1 else ''}")
        return " · ".join(bits)


@dataclass
class Series:
    """One line on the plot: a run, a split and (for k-fold) a fold.

    :ivar kind: ``'single'`` (one train/val split), ``'fold'`` (one fold of a
        k-fold run) or ``'mean'`` (the fold mean, with ``__sd`` columns and an
        ``n_folds`` column recording how many folds reached each epoch).
    :ivar frame: ``epoch`` plus the numeric metric columns, in epoch order and
        **at this series' own length** — never resampled onto a shared axis.
    """
    run_id: str
    split: str
    fold: str
    kind: str
    label: str
    frame: pd.DataFrame
    n_folds: int = 1

    @property
    def epochs(self) -> np.ndarray:
        return pd.to_numeric(self.frame["epoch"], errors="coerce").to_numpy()

    @property
    def n_epochs(self) -> int:
        return int(len(self.frame))

    def values(self, metric: str) -> np.ndarray:
        """This series' values for ``metric`` (empty array when absent)."""
        if metric not in self.frame.columns:
            return np.array([], dtype=float)
        return pd.to_numeric(self.frame[metric], errors="coerce").to_numpy()

    def sd(self, metric: str) -> Optional[np.ndarray]:
        """Fold-to-fold sd for a ``'mean'`` series, else ``None``."""
        col = f"{metric}__sd"
        if col not in self.frame.columns:
            return None
        return pd.to_numeric(self.frame[col], errors="coerce").to_numpy()

    def has(self, metric: str) -> bool:
        """True when this series has at least one finite value for ``metric``."""
        vals = self.values(metric)
        return bool(vals.size) and bool(np.isfinite(vals).any())

    def support(self) -> Optional[Tuple[int, int]]:
        """``(min, max)`` folds contributing to a ``'mean'`` series, else None.

        Folds stop at different epochs when early stopping fires, so the tail
        of a fold mean can be an average over two folds where the head was an
        average over five. ``min < max`` is exactly that situation.
        """
        if "n_folds" not in self.frame.columns:
            return None
        n = pd.to_numeric(self.frame["n_folds"], errors="coerce").dropna()
        return (int(n.min()), int(n.max()))

    def support_drops_at(self) -> Optional[int]:
        """First epoch where fewer than all folds contribute, or ``None``."""
        span = self.support()
        if span is None or span[0] >= span[1]:
            return None
        n = pd.to_numeric(self.frame["n_folds"], errors="coerce")
        idx = int(np.flatnonzero((n < span[1]).to_numpy())[0])
        return int(self.epochs[idx])

    def epoch_range(self) -> Tuple[int, int]:
        eps = self.epochs
        finite = eps[np.isfinite(eps)]
        if not finite.size:
            return (0, 0)
        return (int(finite.min()), int(finite.max()))

    def best(self, metric: str) -> Optional[Dict[str, Any]]:
        """``{'epoch', 'value', 'direction'}`` for the best epoch, or ``None``.

        ``None`` when the metric is absent, entirely NaN, or has no meaningful
        direction (see :func:`metric_direction`).
        """
        direction = metric_direction(metric)
        if direction is None or not self.has(metric):
            return None
        vals = self.values(metric)
        eps = self.epochs
        idx = int(np.nanargmin(vals) if direction == "min" else np.nanargmax(vals))
        return {"epoch": int(eps[idx]), "value": float(vals[idx]),
                "direction": direction}

    def last(self, metric: str) -> Optional[Dict[str, Any]]:
        """``{'epoch', 'value'}`` for the last epoch with a finite value."""
        if not self.has(metric):
            return None
        vals = self.values(metric)
        eps = self.epochs
        finite = np.flatnonzero(np.isfinite(vals))
        idx = int(finite[-1])
        return {"epoch": int(eps[idx]), "value": float(vals[idx])}


@dataclass
class Comparison:
    """The result of :func:`compare_runs` — series to plot plus the diff.

    :ivar series: every line that will be drawn, in run order.
    :ivar settings_diff: the bucketed diff (see :func:`diff_settings`).
    :ivar metrics: metric columns available on at least one series, sorted with
        the common ones first.
    :ivar problems: ``[{'run_id', 'note'}]`` — every note from every run,
        flattened so a caller can show them all in one place.
    :ivar fold_mode: which of ``per_fold`` / ``mean`` / ``both`` produced
        :attr:`series`, so the caller can state it.
    """
    runs: List[TrainingRun]
    series: List[Series]
    settings_diff: Dict[str, Any]
    metrics: List[str]
    problems: List[Dict[str, str]] = field(default_factory=list)
    fold_mode: str = "per_fold"

    def labels(self) -> List[str]:
        return [s.label for s in self.series]

    def series_for(self, label: str) -> Optional[Series]:
        """The series with this label, or ``None``."""
        for s in self.series:
            if s.label == label:
                return s
        return None

    def series_with(self, metric: str) -> List[Series]:
        """Series that actually have finite values for ``metric``."""
        return [s for s in self.series if s.has(metric)]

    def epoch_ranges(self) -> Dict[str, Tuple[int, int]]:
        return {s.label: s.epoch_range() for s in self.series}

    def lengths_differ(self) -> bool:
        """True when the series do not all cover the same epoch span."""
        spans = {s.epoch_range() for s in self.series}
        return len(spans) > 1

    def splits(self) -> List[str]:
        return sorted({s.split for s in self.series})


# ---------------------------------------------------------------------------
# Reading one run
# ---------------------------------------------------------------------------

def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", "split", "fold", "epoch"])


def _default_run_id(path: Path) -> str:
    """A readable id for a training ``dst``.

    ``.../maxvit_t/rgb/epochs_25`` is named ``epochs_25`` which collides across
    model types, so the last three components are joined when the folder looks
    like the layout ``train_test_model`` builds.
    """
    name = path.name
    if re.match(r"^epochs_\d+$", name) and len(path.parts) >= 3:
        return "/".join(path.parts[-3:])
    return name or str(path)


def _fold_dirs(path: Path) -> List[Path]:
    """``fold_<i>`` subfolders holding curves, ordered by fold number."""
    out = []
    try:
        children = list(path.iterdir())
    except OSError:
        return []
    for child in children:
        m = _FOLD_RE.match(child.name)
        if m and child.is_dir() and _has_curve_file(child):
            out.append((int(m.group(1)), child))
    return [p for _, p in sorted(out)]


def _has_curve_file(path: Path) -> bool:
    return any((path / f).is_file() for f in SPLIT_FILES.values())


def _normalize_curve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip, lowercase, alias and de-duplicate the columns of a progress CSV.

    ``evaluate_model_performance`` writes both ``accuracy`` and ``Accuracy``
    (the second is a straight copy); left alone they would show up as two
    metrics in every metric picker. Aliasing mirrors
    ``spacr.utils.suggest_training_changes``, the other reader of these files.
    """
    df = df.drop(columns=[c for c in df.columns
                          if str(c).startswith("Unnamed:")], errors="ignore")
    keep: List[Tuple[Any, str]] = []
    seen = set()
    for col in df.columns:
        low = str(col).strip().lower()
        low = _COLUMN_ALIASES.get(low, low)
        if low in seen:
            continue
        seen.add(low)
        keep.append((col, low))
    out = df[[c for c, _ in keep]].copy()
    out.columns = [low for _, low in keep]
    return out


def _read_curve_csv(path: Path) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Read one progress CSV into ``epoch`` + numeric metric columns.

    Returns ``(frame, notes)``. ``frame`` is ``None`` when the file does not
    exist; an existing file with a header and no rows returns an *empty* frame
    (not ``None``) so the caller can tell "never trained" from "not logged".
    """
    notes: List[str] = []
    if not path.is_file():
        return None, notes
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["epoch"]), [f"{path.name} is empty (0 bytes)"]
    except Exception as e:      # unreadable / malformed — say so, keep going
        return pd.DataFrame(columns=["epoch"]), [
            f"{path.name} could not be parsed ({type(e).__name__}: {e})"]

    df = _normalize_curve_columns(raw)
    if "epoch" not in df.columns:
        # A log without an epoch column can still be ordered by row; say that
        # the x axis is a row index rather than pretending it is an epoch.
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
        if len(df):
            notes.append(f"{path.name} has no 'epoch' column — using row order")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    numeric: Dict[str, pd.Series] = {}
    for col in df.columns:
        if col == "epoch":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            numeric[col] = vals
    out = pd.DataFrame({"epoch": df["epoch"], **numeric})

    bad_epoch = int(out["epoch"].isna().sum())
    if bad_epoch:
        notes.append(f"{path.name}: {bad_epoch} row(s) with an unreadable "
                     f"epoch were dropped")
        out = out[out["epoch"].notna()]
    if len(out) == 0:
        notes.append(f"{path.name} has a header but no epoch rows "
                     f"(zero-epoch log)")
        return out, notes
    if not out["epoch"].is_monotonic_increasing:
        notes.append(f"{path.name}: epoch numbers are not increasing — this "
                     f"folder was probably reused by more than one run; rows "
                     f"are shown as recorded")
    out = out.reset_index(drop=True)
    return out, notes


def _is_outside_any_project(node: Path) -> bool:
    """True when ``node`` is too high up to be a spaCR project folder.

    A hard backstop on the settings climb, underneath the ownership rule in
    :func:`_owns_this_run`: the filesystem root, the user's home and the system
    temp directory are never a project, so the walk stops there whatever the
    layout looks like.

    :param node: a candidate ancestor directory.
    :returns: True to stop climbing.
    """
    try:
        resolved = node.resolve()
    except OSError:
        return True
    if resolved == resolved.parent:        # filesystem root
        return True
    stops = {Path(tempfile.gettempdir()).resolve()}
    try:
        home = Path.home().resolve()
        stops.add(home)
        stops.add(home.parent)             # /home, /Users
    except (RuntimeError, OSError):
        pass
    return resolved in stops


def _owns_this_run(child: Path, steps: int) -> bool:
    """True when the ancestor just stepped onto could hold *this* run's settings.

    ``child`` is the directory the climb came up out of and ``steps`` is how far
    it has climbed, so the ancestor under test is ``child.parent``. Two — and
    only two — ancestors can own a run:

    * ``steps == 1``: the run folder's own parent. Whatever the layout,
      ``<x>/settings`` beside ``<x>/<run>`` is the folder
      :func:`spacr.utils.save_settings` wrote for a run whose ``dst`` is that
      run folder.
    * the project root of the training output tree, which
      :func:`spacr.deep_spacr.train_test_model` builds as
      ``<src>/model/<model_type>/<channels>/epochs_<N>`` (plus ``fold_<i>`` for
      a k-fold run). ``src`` is therefore the ancestor the climb enters by
      stepping up out of :data:`_RUN_TREE_ROOT_DIR`, and that is the *only*
      other one, because ``save_settings`` writes ``<src>/settings``.

    Everything above those is somebody else's directory. Accepting it is not a
    harmless miss: the settings diff would report differences that were never
    configured, and the reported provenance would depend on where in the
    filesystem the project happens to sit — a run under ``/data/plate1``
    picking up ``/data/settings/*.csv`` from an unrelated pipeline. No settings
    is better than someone else's.

    :param child: the directory the climb has just left.
    :param steps: 1 for the run folder's parent, 2 for its grandparent, ...
    :returns: True when ``child.parent`` may be searched for ``settings/``.
    """
    return steps == 1 or child.name == _RUN_TREE_ROOT_DIR


def _load_settings(path: Path) -> Tuple[Dict[str, Any], str, List[str]]:
    """Recover the settings that produced the run in ``path``.

    Looks, in order: a run-journal ``settings.json`` / ``settings.csv`` inside
    the folder, then ``<ancestor>/settings/*.csv`` — where
    :func:`spacr.utils.save_settings` puts a training snapshot
    (``<src>/settings/train_test_<model_type>_<epochs>.csv``) — for the
    ancestors :func:`_owns_this_run` accepts, within
    :data:`_SETTINGS_SEARCH_DEPTH` parents.
    """
    notes: List[str] = []

    j = path / "settings.json"
    if j.is_file():
        try:
            data = json.loads(j.read_text())
            if isinstance(data, dict):
                return data, str(j), notes
            notes.append(f"settings.json is a {type(data).__name__}, not an "
                         f"object — ignored")
        except Exception as e:
            notes.append(f"settings.json unreadable ({type(e).__name__})")

    c = path / "settings.csv"
    if c.is_file():
        try:
            return _read_settings_csv(c), str(c), notes
        except Exception as e:
            notes.append(f"settings.csv unreadable ({type(e).__name__})")

    model_type, epochs = _run_shape_from_path(path)
    node = path
    for steps in range(1, _SETTINGS_SEARCH_DEPTH + 1):
        node, child = node.parent, node
        if _is_outside_any_project(node):
            # Backstop: the climb has left anything that could be a spaCR
            # project at all (root / home / the temp directory).
            break
        if not _owns_this_run(child, steps):
            # An ancestor that cannot have written this run's snapshot. Keep
            # climbing — <src> is still four or five steps up a training tree —
            # but do not read what is in this one.
            continue
        sdir = node / "settings"
        if not sdir.is_dir():
            continue
        try:
            cands = sorted(p for p in sdir.glob("*.csv") if p.is_file())
        except OSError:
            continue
        picked = _pick_settings_file(cands, model_type, epochs)
        if picked is None:
            continue
        try:
            settings = _read_settings_csv(picked)
        except Exception as e:
            notes.append(f"{picked.name} unreadable ({type(e).__name__})")
            continue
        if len(cands) > 1 and not _exact_settings_name(picked, model_type, epochs):
            notes.append(
                f"settings taken from {picked.name}; {len(cands) - 1} other "
                f"settings file(s) in {sdir} could also have produced this run")
        return settings, str(picked), notes

    notes.append("no settings found (looked for settings.json/settings.csv in "
                 "the run folder, settings/*.csv beside it, and "
                 "<src>/settings/*.csv at the root of this run's model/ tree) "
                 "— this run is excluded from the settings diff")
    return {}, "", notes


def _run_shape_from_path(path: Path) -> Tuple[str, str]:
    """``(model_type, epochs)`` guessed from the ``dst`` layout, or ``('','')``."""
    m = re.match(r"^epochs_(\d+)$", path.name)
    if not m or len(path.parts) < 3:
        return "", ""
    # <src>/model/<model_type>/<channels>/epochs_<N>
    return path.parts[-3], m.group(1)


def _settings_stems(model_type: str, epochs: str) -> List[str]:
    if not model_type or not epochs:
        return []
    return [f"train_test_{model_type}_{epochs}",
            f"train_{model_type}_{epochs}",
            f"test_{model_type}_{epochs}"]


def _exact_settings_name(path: Path, model_type: str, epochs: str) -> bool:
    return path.stem in _settings_stems(model_type, epochs)


def _pick_settings_file(cands: Sequence[Path], model_type: str,
                        epochs: str) -> Optional[Path]:
    """Best settings CSV for a run, or ``None`` when none of them is plausible."""
    if not cands:
        return None
    for stem in _settings_stems(model_type, epochs):
        for p in cands:
            if p.stem == stem:
                return p
    prefixed = [p for p in cands
                if p.stem.startswith(("train_test_", "train_", "test_"))]
    pool = prefixed or list(cands)
    # Deterministic and defensible: the most recently written one.
    return max(pool, key=lambda p: (p.stat().st_mtime, p.name))


def _load_manifest(path: Path) -> Dict[str, Any]:
    mp = path / "manifest.json"
    if not mp.is_file():
        return {}
    try:
        data = json.loads(mp.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _series_label(run_id: str, split: str, fold: str, kind: str,
                  n_folds: int = 1) -> str:
    """``"run · split · fold"`` — every plotted line says all three.

    A legend entry that omits the split invites reading a train curve as a
    held-out result, and one that omits the fold invites reading one lucky
    fold as the run.
    """
    if kind == "mean":
        return f"{run_id} · {split} · mean of {n_folds} folds ±sd"
    if fold:
        pretty = fold.replace("_", " ")
        return f"{run_id} · {split} · {pretty}"
    return f"{run_id} · {split}"


def _group_notes(items: Sequence[Tuple[str, str]]) -> List[str]:
    """Collapse ``(where, message)`` pairs into one line per message."""
    order: List[str] = []
    wheres: Dict[str, List[str]] = {}
    for where, msg in items:
        if msg not in wheres:
            wheres[msg] = []
            order.append(msg)
        if where and where not in wheres[msg]:
            wheres[msg].append(where)
    out = []
    for msg in order:
        w = wheres[msg]
        out.append(f"{msg} ({', '.join(w)})" if w else msg)
    return out


def load_run(path: Any, run_id: Optional[str] = None) -> TrainingRun:
    """Load one training run folder into a :class:`TrainingRun`.

    ``path`` is the folder :func:`spacr.deep_spacr.train_model` was given as
    ``dst`` — the one holding ``train.csv`` / ``validation.csv``, or holding
    the ``fold_<i>/`` subfolders of a cross-validated run.

    A folder that is missing its curves, missing its settings or holding a
    zero-epoch log does **not** raise: the run comes back with whatever could
    be read and a :attr:`TrainingRun.notes` entry per problem, so one bad
    folder in a scan cannot stop the rest being compared.

    :param path: run folder.
    :param run_id: override the generated id.
    :returns: the loaded run.
    :raises FileNotFoundError: only when ``path`` is not a directory.
    """
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"no such run folder: {path}")

    rid = run_id or _default_run_id(p)
    settings, settings_path, notes = _load_settings(p)
    manifest = _load_manifest(p)

    fold_paths = _fold_dirs(p)
    folds = [f.name for f in fold_paths]
    sources: List[Tuple[Path, str]] = (
        [(f, f.name) for f in fold_paths] if fold_paths else [(p, "")])

    frames: List[pd.DataFrame] = []
    raw_notes: List[Tuple[str, str]] = []
    for folder, fold_name in sources:
        where = fold_name or ""
        for split, filename in SPLIT_FILES.items():
            frame, fnotes = _read_curve_csv(folder / filename)
            for n in fnotes:
                raw_notes.append((where, n))
            if frame is None:
                raw_notes.append((where, f"no {filename}"))
                continue
            if frame.empty:
                continue
            frame = frame.copy()
            frame.insert(0, "fold", fold_name)
            frame.insert(0, "split", split)
            frame.insert(0, "run_id", rid)
            frames.append(frame)

    curves = (pd.concat(frames, ignore_index=True, sort=False)
              if frames else _empty_curves())
    notes.extend(_group_notes(raw_notes))
    if curves.empty:
        notes.append("no per-epoch curves in this folder — nothing to plot "
                     "(the other runs are unaffected)")

    run = TrainingRun(run_id=rid, path=p, settings=settings, curves=curves,
                      folds=folds, notes=notes, settings_path=settings_path,
                      manifest=manifest)
    run.final_metrics = _final_metrics(run)
    return run


def _final_metrics(run: TrainingRun) -> Dict[str, Any]:
    """Per (split, fold): epoch count plus last-epoch **and** best-epoch values."""
    out: Dict[str, Any] = {}
    if run.curves.empty:
        return out
    for series in _series_from_run(run, "per_fold"):
        entry: Dict[str, Any] = {
            "run_id": series.run_id,
            "split": series.split,
            "fold": series.fold,
            "n_epochs": series.n_epochs,
            "epoch_range": series.epoch_range(),
            "last": {},
            "best": {},
        }
        for metric in series.frame.columns:
            if metric in _IDENTITY_COLUMNS or metric.endswith("__sd"):
                continue
            last = series.last(metric)
            if last is not None:
                entry["last"][metric] = last
            best = series.best(metric)
            if best is not None:
                entry["best"][metric] = best
        out[series.label] = entry
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _looks_like_run(path: Path) -> bool:
    """True when ``path`` is plausibly a training ``dst``.

    Deliberately wider than "has curves": a folder holding checkpoints but no
    ``train.csv`` is a *broken* run, and a broken run has to be discoverable in
    order to be reported. Recognised by any of: a progress CSV, a ``.pth``
    checkpoint, a cross-validation ``*_per_fold.csv``, or ``fold_<i>``
    subfolders that hold curves.
    """
    try:
        if _has_curve_file(path):
            return True
        for child in path.iterdir():
            if child.is_file():
                if child.suffix == ".pth" or child.name.endswith("_per_fold.csv"):
                    return True
            elif _FOLD_RE.match(child.name) and _has_curve_file(child):
                return True
    except OSError:
        return False
    return False


def find_runs(root: Any, max_depth: int = DEFAULT_SCAN_DEPTH,
              limit: int = 200) -> List[TrainingRun]:
    """Discover every training run under ``root``, newest folder first.

    Walks at most ``max_depth`` levels, skips hidden folders, and never returns
    a ``fold_<i>`` folder found *below* ``root`` in its own right — folds belong
    to the run above them and are loaded as part of it. Pointing ``root``
    straight at a fold folder still loads that one fold, because then it is
    what the caller asked for.

    Runs that are broken (no curves, no settings, zero-epoch log) are returned
    with their notes rather than skipped: a scan that silently drops the folder
    you were looking for is worse than one that tells you what is wrong with it.

    :param root: folder to scan (may itself be a run folder).
    :param max_depth: how deep below ``root`` to look.
    :param limit: stop after this many runs.
    :returns: loaded runs with ids made unique within the result.
    """
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"no such folder: {root}")

    found: List[Path] = []
    stack: List[Tuple[Path, int]] = [(base, 0)]
    while stack and len(found) < limit:
        node, depth = stack.pop(0)
        if _looks_like_run(node):
            found.append(node)
        if depth >= max_depth:
            continue
        try:
            children = sorted(c for c in node.iterdir() if c.is_dir())
        except OSError:
            continue
        for child in children:
            if child.name.startswith("."):
                continue
            if _FOLD_RE.match(child.name) and _looks_like_run(node):
                continue        # folds are loaded with their parent
            stack.append((child, depth + 1))

    found.sort(key=lambda p: (-_folder_mtime(p), str(p)))
    ids = _unique_ids(found)
    return [load_run(p, run_id=ids[str(p)]) for p in found]


def _folder_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _unique_ids(paths: Sequence[Path]) -> Dict[str, str]:
    """Map each path to a short id, lengthening only the ones that collide.

    Two runs of the same model trained from different dataset roots both end
    up in ``…/maxvit_t/rgb/epochs_25``; a legend with two identically-labelled
    lines is worse than a long label, so colliding ids grow one path component
    at a time until they separate.
    """
    out: Dict[str, str] = {}
    used: set = set()
    for p in paths:
        base = _default_run_id(p)
        rid = base
        depth = len(Path(base).parts)
        while rid in used and depth < len(p.parts):
            depth += 1
            rid = "/".join(p.parts[-depth:])
        # The loop bottoms out at the full path, which no two runs share, so
        # there is no further fallback to write.
        used.add(rid)
        out[str(p)] = rid
    return out


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------

def _fold_sort_key(name: Any):
    """Sort ``fold_2`` before ``fold_10`` and a fold-less run first."""
    text = str(name)
    if not text:
        return (0, 0, "")
    m = _FOLD_RE.match(text)
    return (1, int(m.group(1)), "") if m else (2, 0, text)


def _series_from_run(run: TrainingRun, fold_mode: str) -> List[Series]:
    """Build the plottable series for one run under ``fold_mode``."""
    if run.curves.empty:
        return []
    out: List[Series] = []
    metric_cols = [c for c in run.curves.columns if c not in _IDENTITY_COLUMNS]

    per_fold = fold_mode in ("per_fold", "both") or not run.folds
    want_mean = fold_mode in ("mean", "both") and bool(run.folds)

    for split in ("train", "val"):
        block = run.curves[run.curves["split"] == split]
        if block.empty:
            continue
        # Derived from the rows actually present rather than from run.folds:
        # a fold whose validation.csv is missing must not produce an empty
        # validation series, and ``fold_10`` must sort after ``fold_2``.
        fold_names = sorted(set(block["fold"]), key=_fold_sort_key)
        if per_fold:
            for fold in fold_names:
                sub = block[block["fold"] == fold]
                frame = (sub[["epoch"] + metric_cols]
                         .sort_values("epoch").reset_index(drop=True))
                kind = "fold" if fold else "single"
                out.append(Series(
                    run_id=run.run_id, split=split, fold=fold, kind=kind,
                    label=_series_label(run.run_id, split, fold, kind),
                    frame=frame))
        if want_mean and run.folds:
            mean = _fold_mean(block, metric_cols)
            if mean is not None and len(mean):
                n = int(len(fold_names))
                out.append(Series(
                    run_id=run.run_id, split=split, fold="mean", kind="mean",
                    label=_series_label(run.run_id, split, "mean", "mean", n),
                    frame=mean, n_folds=n))
    return out


def _fold_mean(block: pd.DataFrame, metric_cols: Sequence[str]
               ) -> Optional[pd.DataFrame]:
    """Mean ± sd across folds, per epoch, with the contributing fold count.

    ``ddof=1`` matches :func:`spacr.deep_spacr.summarize_cv_metrics`: the folds
    are a sample of the possible splits, not the population.

    Folds can stop at different epochs (early stopping kills one fold at 12 and
    another at 40), so the mean at a late epoch may be over two folds rather
    than five. That is recorded in ``n_folds`` per epoch rather than smoothed
    over — a mean whose support shrinks is not the same curve.
    """
    records: List[Dict[str, Any]] = []
    for epoch, sub in block.groupby("epoch", sort=True):
        row: Dict[str, Any] = {"epoch": epoch,
                               "n_folds": int(sub["fold"].nunique())}
        for metric in metric_cols:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[metric] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}__sd"] = (float(vals.std(ddof=1))
                                    if len(vals) > 1 else np.nan)
        records.append(row)
    return pd.DataFrame(records) if records else None


def available_metrics(runs: Sequence[TrainingRun],
                      folds: str = "per_fold") -> List[str]:
    """Metric columns these runs logged, common ones first.

    Lets a caller populate a metric picker before anything is compared.
    """
    series = [s for r in runs for s in _series_from_run(r, folds)]
    return _ordered_metrics(series)


def render_setting_value(value: Any, width: int = 40) -> str:
    """One-line, length-capped rendering of a settings value.

    Thin public wrapper over :func:`spacr.run_journal._render_value` so the GUI
    renders settings exactly the way the console report does.
    """
    return _render_value(value, width)


def _ordered_metrics(series: Sequence[Series]) -> List[str]:
    """Metrics available on any series, with the ones people ask for first."""
    seen: List[str] = []
    for s in series:
        for col in s.frame.columns:
            if col in _IDENTITY_COLUMNS or col.endswith("__sd"):
                continue
            if col not in seen:
                seen.append(col)
    preferred = ["accuracy", "loss", "prauc", "f1_macro",
                 "neg_accuracy", "pos_accuracy"]
    head = [m for m in preferred if m in seen]
    tail = sorted(m for m in seen if m not in head)
    return head + tail


# ---------------------------------------------------------------------------
# Settings diff — the run_journal bucketing, generalised to N runs
# ---------------------------------------------------------------------------

def is_env_key(key: Any, env_keys: Sequence[str] = ()) -> bool:
    """True when a settings key records the machine, not a modelling decision.

    Token-wise, not substring: ``start_time`` matches, ``update_freq`` does not.
    ``src`` deliberately does **not** match — a run on a different dataset is a
    real difference and belongs in ``changed``, even though it is a path.
    """
    k = str(key).strip().lower()
    if k in ENV_SETTING_KEYS or k in {str(e).lower() for e in env_keys}:
        return True
    return any(tok in ENV_KEY_TOKENS for tok in k.split("_"))


def diff_settings(runs: Sequence[TrainingRun],
                  env_keys: Sequence[str] = ()) -> Dict[str, Any]:
    """Bucket the settings differences across N runs.

    Generalises :func:`spacr.run_journal.diff_runs` from two runs to many and
    reuses its comparison (:func:`spacr.run_journal.values_equal`) and its
    reason for bucketing: a flat diff of two real runs reports ~200
    differences of which none are decisions anybody made.

    :returns: a dict with

        ``run_ids``
            the runs that contributed settings, in order.
        ``changed``
            ``[{'key', 'values': {run_id: value}}]`` for keys present in every
            contributing run whose values differ — the signal.
        ``env``
            same shape, for keys :func:`is_env_key` classifies as environment
            drift. Never merged into ``changed``.
        ``drift``
            ``[{'key', 'present': [...], 'missing': [...]}]`` — schema drift.
        ``same`` / ``shared``
            counts of unchanged and shared keys.
        ``identical``
            True when two or more runs contributed settings and nothing at all
            differs. Callers must say "no differences" rather than render an
            empty table.
        ``no_settings``
            run ids that had no settings to contribute.
        ``env_manifest``
            the manifest ``env`` diff, when the folders are run-journal runs.
    """
    with_settings = [r for r in runs if r.settings]
    ids = [r.run_id for r in with_settings]
    no_settings = [r.run_id for r in runs if not r.settings]

    out: Dict[str, Any] = {
        "run_ids": ids,
        "changed": [],
        "env": [],
        "drift": [],
        "same": 0,
        "shared": 0,
        "identical": False,
        "no_settings": no_settings,
        "env_manifest": _diff_manifest_env(runs),
    }
    if len(with_settings) < 2:
        return out

    key_sets = [set(r.settings) for r in with_settings]
    shared = set.intersection(*key_sets)
    everything = set.union(*key_sets)
    out["shared"] = len(shared)

    for key in sorted(shared):
        values = {r.run_id: r.settings[key] for r in with_settings}
        first = with_settings[0].settings[key]
        if all(values_equal(first, r.settings[key]) for r in with_settings[1:]):
            out["same"] += 1
            continue
        row = {"key": key, "values": values}
        (out["env"] if is_env_key(key, env_keys) else out["changed"]).append(row)

    for key in sorted(everything - shared):
        present = [r.run_id for r in with_settings if key in r.settings]
        missing = [r.run_id for r in with_settings if key not in r.settings]
        out["drift"].append({"key": key, "present": present,
                             "missing": missing})

    out["identical"] = not (out["changed"] or out["env"] or out["drift"])
    return out


def _diff_manifest_env(runs: Sequence[TrainingRun]) -> List[Dict[str, Any]]:
    """N-way version of :func:`spacr.run_journal._diff_env`.

    Same rule as there: when any side has no ``env`` snapshot at all, report
    nothing rather than declaring every package on the other side changed.
    """
    envs = []
    for r in runs:
        env = r.manifest.get("env") if isinstance(r.manifest, dict) else None
        if not isinstance(env, dict) or not env:
            return []
        envs.append((r.run_id, env))
    if len(envs) < 2:
        return []
    out = []
    for key in sorted(set().union(*[set(e) for _, e in envs])):
        values = {rid: env.get(key) for rid, env in envs}
        first = envs[0][1].get(key)
        if all(values_equal(first, env.get(key)) for _, env in envs[1:]):
            continue
        out.append({"key": key, "values": values})
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

#: Accepted values for ``compare_runs(folds=...)``.
FOLD_MODES = ("per_fold", "mean", "both")


def compare_runs(runs: Sequence[TrainingRun], folds: str = "per_fold",
                 env_keys: Sequence[str] = ()) -> Comparison:
    """Line up several runs' curves and diff their settings.

    :param runs: runs from :func:`load_run` / :func:`find_runs`. Runs with no
        curves are kept — their notes travel into
        :attr:`Comparison.problems` — but contribute no series.
    :param folds: how to render a cross-validated run: ``'per_fold'``
        (default; every fold gets a line), ``'mean'`` (fold mean with a ±1 sd
        band, labelled as a mean) or ``'both'``.
    :param env_keys: extra settings keys to treat as environment drift.
    :returns: a :class:`Comparison`.
    :raises ValueError: on an unknown ``folds`` mode.
    """
    if folds not in FOLD_MODES:
        raise ValueError(f"folds must be one of {FOLD_MODES}, got {folds!r}")
    runs = list(runs)

    series: List[Series] = []
    for run in runs:
        series.extend(_series_from_run(run, folds))

    problems = [{"run_id": r.run_id, "note": n} for r in runs for n in r.notes]
    return Comparison(
        runs=runs,
        series=series,
        settings_diff=diff_settings(runs, env_keys=env_keys),
        metrics=_ordered_metrics(series),
        problems=problems,
        fold_mode=folds,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

#: train dashed, validation solid — the split is readable without the legend.
_SPLIT_STYLE = {"train": "--", "val": "-"}


def _run_colours(run_ids: Sequence[str]) -> Dict[str, str]:
    from matplotlib import pyplot as plt
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["#4A9EFF"]
    return {rid: cycle[i % len(cycle)] for i, rid in enumerate(run_ids)}


def plot_curves(comparison: Comparison, metric: str = "accuracy",
                ax: Any = None, labels: Optional[Sequence[str]] = None,
                figsize: Tuple[float, float] = (9.0, 5.0),
                mark_best: bool = False, band: bool = True) -> Any:
    """Overlay every series' ``metric`` on one axes and return the figure.

    Each series is drawn **over its own epochs**. Runs of different lengths are
    not truncated to the shortest or padded to the longest: the lines simply
    end where the runs ended, and the axes annotation says so.

    Colour identifies the run, line style the split (train dashed, validation
    solid), and every legend entry spells out run · split · fold. When more
    than one split is on the axes a note says so — a train curve sitting above
    a validation curve is overfitting, not a better run.

    ``'mean'`` series (from ``compare_runs(folds='mean')``) are drawn with a ±1
    sd band; where fewer folds reach an epoch than the run has, the band is
    thinner because fewer folds contributed, which the legend states.

    :param comparison: from :func:`compare_runs`.
    :param metric: metric column to draw.
    :param ax: draw into this axes instead of making a figure (the Qt screen
        reuses one canvas).
    :param labels: only draw these series labels.
    :param figsize: figure size when ``ax`` is None.
    :param mark_best: put a marker on each series' best epoch.
    :param band: draw the ±sd band for mean series.
    :returns: the :class:`matplotlib.figure.Figure`. It carries
        ``spacr_series_by_label`` — ``{line label: Series}`` — so a click on a
        line can name its run.
    """
    from matplotlib import pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    wanted = comparison.series
    if labels is not None:
        keep = list(labels)
        wanted = [s for s in wanted if s.label in keep]
    drawn = [s for s in wanted if s.has(metric)]

    mapping: Dict[str, Series] = {}
    if not drawn:
        ax.text(0.5, 0.5,
                f"no selected run logged '{metric}'",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        fig.spacr_series_by_label = mapping
        return fig

    colours = _run_colours([r.run_id for r in comparison.runs]
                           or [s.run_id for s in drawn])
    for s in drawn:
        colour = colours.get(s.run_id, "#4A9EFF")
        style = _SPLIT_STYLE.get(s.split, "-")
        width = 2.0 if s.kind == "mean" else (1.1 if s.kind == "fold" else 1.6)
        alpha = 0.65 if s.kind == "fold" else 1.0
        eps, vals = s.epochs, s.values(metric)
        line, = ax.plot(eps, vals, style, color=colour, linewidth=width,
                        alpha=alpha, label=s.label)
        line.set_picker(6)
        mapping[s.label] = s
        if band and s.kind == "mean":
            sd = s.sd(metric)
            if sd is not None and np.isfinite(sd).any():
                ax.fill_between(eps, vals - sd, vals + sd, color=colour,
                                alpha=0.15, linewidth=0)
        if mark_best:
            best = s.best(metric)
            if best is not None:
                ax.plot([best["epoch"]], [best["value"]], "o", color=colour,
                        markersize=5, alpha=alpha)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.15)

    notes: List[str] = []
    splits = sorted({s.split for s in drawn})
    if len(splits) > 1:
        notes.append("axes mixes train (dashed) and validation (solid) — "
                     "a train curve above a validation one is overfitting, "
                     "not a better run")
    spans = {s.epoch_range() for s in drawn}
    if len(spans) > 1:
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        notes.append(f"runs cover different epoch counts ({lo}–{hi}); each "
                     f"curve is drawn to its own length")
    if any(s.kind == "mean" for s in drawn):
        notes.append("shaded band is ±1 sd across folds")
        for s in drawn:
            drop = s.support_drops_at() if s.kind == "mean" else None
            if drop is not None:
                lo, hi = s.support()
                notes.append(f"{s.label}: only {lo} of {hi} folds reach "
                             f"epoch {drop}+, so the tail averages fewer folds")

    ax.set_title(f"{metric} — {len(drawn)} series from "
                 f"{len({s.run_id for s in drawn})} run(s)")
    if notes:
        ax.text(0.0, -0.16, "\n".join(notes), transform=ax.transAxes,
                fontsize=8, va="top", ha="left", alpha=0.8)
    ax.legend(loc="best", fontsize=8, framealpha=0.6)

    fig.spacr_series_by_label = mapping
    return fig


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Four decimals, or ``nan`` — a metric that was not computed says so."""
    return "nan" if value != value else f"{float(value):.4f}"


def _table(rows: Sequence[Sequence[str]], indent: str = "  ") -> List[str]:
    """Align ``rows`` into columns. Both callers pass a header row first."""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    out = []
    for r in rows:
        cells = [str(c).ljust(widths[i]) for i, c in enumerate(r)]
        out.append(indent + "  ".join(cells).rstrip())
    return out


def format_comparison(comparison: Comparison, metric: str = "accuracy",
                      max_drift_names: int = 6) -> str:
    """Render a :class:`Comparison` as a console report.

    Ordering mirrors :func:`spacr.run_journal.format_run_diff`: the runs, then
    the curves, then the settings that changed (the signal), then environment
    drift, then schema drift reduced to one line. Problems come first, because
    a run with no curves in the list is the thing you most need to know.

    Both the last-epoch and the best-epoch value of ``metric`` are printed for
    every series, with the caveat that makes the second one interpretable.
    """
    lines: List[str] = ["Training run comparison"]

    # -- runs --------------------------------------------------------------
    for run in comparison.runs:
        head = f"  {run.run_id}"
        shape = (f"{run.n_epochs} epochs" if run.has_curves else "no curves")
        if run.folds:
            shape += f", {len(run.folds)} folds"
        lines.append(f"{head}  ({shape})")
        lines.append(f"     {run.path}")
        if run.settings_path:
            lines.append(f"     settings: {run.settings_path} "
                         f"({len(run.settings)} keys)")
        for note in run.notes:
            lines.append(f"     ! {note}")

    # -- curves ------------------------------------------------------------
    lines.append("")
    drawn = comparison.series_with(metric)
    if not drawn:
        lines.append(f"Curves: no run logged '{metric}'"
                     + (f" (available: {', '.join(comparison.metrics)})"
                        if comparison.metrics else ""))
    else:
        lines.append(f"Curves — {metric} "
                     f"({len(drawn)} series, folds={comparison.fold_mode})")
        rows = [["series", "epochs", "last", "@", "best", "@"]]
        for s in drawn:
            lo, hi = s.epoch_range()
            last = s.last(metric)
            best = s.best(metric)
            rows.append([
                s.label,
                f"{lo}–{hi}" if lo != hi else str(hi),
                _fmt(last["value"]) if last else "—",
                str(last["epoch"]) if last else "—",
                _fmt(best["value"]) if best else "—",
                str(best["epoch"]) if best else "—",
            ])
        lines.extend(_table(rows))
        if comparison.lengths_differ():
            lines.append("  Runs cover different epoch counts — each curve is "
                         "shown to its own length; none is truncated or padded.")
        if len(comparison.splits()) > 1:
            lines.append("  train and validation are both listed; compare like "
                         "with like (the split is in every label).")
        if any(s.kind == "mean" for s in drawn):
            lines.append("  'mean of k folds ±sd' rows are an average across "
                         "folds, not a single run; per-fold spread is the "
                         "point of k-fold.")
            for s in drawn:
                drop = s.support_drops_at() if s.kind == "mean" else None
                if drop is None:
                    continue
                lo, hi = s.support()
                lines.append(f"  {s.label}: from epoch {drop} only {lo} of "
                             f"{hi} folds reach that far, so the tail is a "
                             f"mean over fewer folds than the head.")
        if metric_direction(metric) is not None:
            lines.append("  'best' is the best epoch of this very curve, so on "
                         "a validation series it is an optimistically biased "
                         "estimate; 'last' is unbiased but may be past the "
                         "optimum. Both are shown for that reason.")

    # -- settings ----------------------------------------------------------
    diff = comparison.settings_diff
    lines.append("")
    ids = diff.get("run_ids") or []
    if len(ids) < 2:
        which = ", ".join(diff.get("no_settings") or []) or "these runs"
        lines.append(f"Settings: not comparable — fewer than two runs have a "
                     f"settings snapshot ({which} had none)")
    elif diff.get("identical"):
        lines.append(f"Settings: no differences — all {len(ids)} runs ran with "
                     f"identical settings ({diff.get('shared', 0)} keys "
                     f"compared, none differ)")
    else:
        changed = diff.get("changed") or []
        if changed:
            lines.append(f"Settings changed ({len(changed)} of "
                         f"{diff.get('shared', 0)} shared keys)")
            lines.extend(_value_rows(changed, ids))
        else:
            lines.append(f"Settings changed (0 of {diff.get('shared', 0)} "
                         f"shared keys) — nothing the user chose differs")

    env = diff.get("env") or []
    env_manifest = diff.get("env_manifest") or []
    lines.append("")
    if env or env_manifest:
        lines.append(f"Environment drift ({len(env) + len(env_manifest)}) — "
                     f"shown separately because none of it is a modelling "
                     f"decision")
        lines.extend(_value_rows(list(env) + list(env_manifest), ids))
    elif len(ids) >= 2:
        lines.append("Environment drift (0) — same paths, hosts and versions")

    drift = diff.get("drift") or []
    lines.append("")
    if drift:
        names = [d["key"] for d in drift]
        lines.append(f"Schema drift: {len(drift)} key(s) missing from at least "
                     f"one run")
        lines.append(f"  {_drift_names(names, max_drift_names)}")
    elif len(ids) >= 2:
        lines.append("Schema drift: none — every run records the same keys")

    if diff.get("no_settings"):
        lines.append("")
        lines.append("No settings snapshot found for: "
                     + ", ".join(diff["no_settings"]))
    return "\n".join(lines)


def _value_rows(entries: Sequence[Dict[str, Any]],
                ids: Sequence[str]) -> List[str]:
    """Render ``[{'key', 'values': {run_id: v}}]`` as aligned rows.

    For exactly two runs the two values are rendered as a pair through
    :func:`spacr.run_journal._render_change_pair`, which elides the common
    prefix — otherwise two long paths that differ in one component print the
    same 46 characters twice. Callers only ever pass a non-empty list.
    """
    rows = [["setting"] + list(ids)]
    for e in entries:
        values = e.get("values") or {}
        if len(ids) == 2:
            a, b = _render_change_pair(values.get(ids[0]), values.get(ids[1]))
            rendered = [a, b]
        else:
            rendered = [_render_value(values.get(rid), 28) for rid in ids]
        rows.append([e["key"]] + rendered)
    return _table(rows)
