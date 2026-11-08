"""A trained mitotic-cell classifier for optical pooled screens.

:func:`spacr.gene_measurement_compare.mitotic_calls` calls a cell mitotic
with a fixed rule on nuclear DNA intensity and area. On plate A of Funk et
al. 2022 the per-gene mitotic index from that rule correlates with Funk et
al.'s at Spearman rho = 0.32 only. Funk et al. trained a classifier on DNA and
tubulin features instead; this module does the same with labelled cells.

* :func:`read_mitotic_labels` reads a labelled-cell table (optionally
  through a separate crop-to-cell mapping, so labellers stay blind to well
  and gene) and turns its stages into mitotic calls.
* :func:`mitotic_features` picks the DNA (nuclear) and tubulin (cell and
  cytoplasm) features from a cell table.
* :func:`train_mitotic_classifier` fits a gradient-boosting classifier on
  labelled cells and reports grouped cross-validation, with folds split by
  well or field so that cells of one image never sit on both sides, and a
  shuffled-label control that shows what chance scores.
* :func:`sampling_weights` re-weights a label set that over-samples some
  cells (such as the rule's mitotic calls) so that training and metrics
  describe the whole screen.
* :func:`predict_mitotic` calls new cells; :func:`save_mitotic_classifier`
  and :func:`load_mitotic_classifier` keep a trained model on disk, and the
  OPS aggregate phase uses it through ``ops_mitotic_model``.
* :func:`mitotic_index_agreement` compares per-gene mitotic indices with a
  reference, such as Funk et al.'s published table, with a shuffled control.

Everything runs on the CPU. The labels must come from the user: a set of
cells annotated mitotic or not. spaCR does not ship labels or a trained
model, and it does not make labels from the rule, because a classifier
trained on the rule's calls can only learn the rule.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

#: The metrics every fold reports.
FOLD_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "roc_auc", "average_precision")

#: Label classes that count as mitotic, after :func:`_normalise_class`.
MITOTIC_CLASSES = ("prophase-prometaphase", "prophase", "prometaphase", "metaphase",
                   "anaphase-telophase", "anaphase", "telophase", "telophase-daughter",
                   "mitotic-stage-disputed", "mitotic", "mitosis")

#: Label classes that count as not mitotic.
INTERPHASE_CLASSES = ("interphase", "not mitotic", "non-mitotic", "nonmitotic")

#: Label classes left out of training: crops that show no usable cell.
EXCLUDED_CLASSES = ("artefact", "artifact", "unclear", "skip")

#: The cell key a label table is joined on.
CELL_KEY = ("plate", "well", "site", "label")

#: The format marker of a saved classifier.
MODEL_FORMAT = "spacr-ops-mitotic-classifier-1"


def _normalise_class(value: Any) -> str:
    """A label class in lower case, with ``_`` and ``/`` read as ``-``."""
    text = str(value).strip().lower().replace("_", "-").replace("/", "-")
    return " ".join(text.split())


def read_mitotic_labels(labels: Any, mapping: Any = None, *, class_column: Optional[str] = None,
                        id_column: Optional[str] = None, artefacts: str = "drop") -> pd.DataFrame:
    """Read a table of labelled cells and turn the classes into mitotic calls.

    :param labels: a CSV path or a frame with one row per labelled cell and
        a class column: ``interphase``, ``prophase-prometaphase``,
        ``metaphase``, ``anaphase-telophase``, ``telophase-daughter`` (one
        half of a telophase pair that segmentation split, counted as
        mitotic), ``mitotic-stage-disputed`` or ``artefact`` (case, ``_``
        and ``/`` do not matter).
    :param mapping: when ``labels`` names crops rather than cells, a CSV path
        or frame that maps each crop to its cell: the shared id column plus
        ``plate, well, site, label``. Keeping it separate lets the labellers
        work blind to well and gene.
    :param class_column: the class column; found among ``class_fine``,
        ``class``, ``stage``, ``phase``, ``annotation``, ``call``,
        ``label_class`` and ``class3`` when None, or ``label`` when the
        table has no cell key.
    :param id_column: the column shared by ``labels`` and ``mapping``; the
        one column both have, other than the class, when None.
    :param artefacts: ``'drop'`` leaves artefact crops out; ``'negative'``
        keeps them as not mitotic, so a classifier learns not to call them
        and its precision counts them as false calls.
    :returns: ``plate, well, site, label, stage, mitotic``: one row per
        labelled cell, ``stage`` the normalised class and ``mitotic`` True
        for every mitotic stage.
    :raises ValueError: for a class that is none of the known ones, a cell
        labelled twice with different classes, or a missing column.
    """
    table = pd.read_csv(labels) if isinstance(labels, (str, bytes)) or hasattr(
        labels, "__fspath__") else pd.DataFrame(labels).copy()
    has_key = all(c in table for c in CELL_KEY)
    if class_column is None:
        options = ["class_fine", "class", "stage", "phase", "annotation", "call",
                   "label_class", "class3"]
        if not has_key:
            options.append("label")
        class_column = next((c for c in options if c in table), None)
        if class_column is None:
            raise ValueError(f"no class column in the label table; looked for {options}")
    table = table.rename(columns={class_column: "_class"})
    if mapping is not None:
        cells = pd.read_csv(mapping) if isinstance(mapping, (str, bytes)) or hasattr(
            mapping, "__fspath__") else pd.DataFrame(mapping).copy()
        missing = [c for c in CELL_KEY if c not in cells]
        if missing:
            raise ValueError(f"the mapping lacks the cell key columns {missing}")
        if id_column is None:
            shared = [c for c in table.columns if c in cells.columns and c not in CELL_KEY]
            if len(shared) != 1:
                raise ValueError(f"cannot tell the id column shared by labels and mapping; "
                                 f"candidates {shared}; pass id_column")
            id_column = shared[0]
        table = table[[id_column, "_class"]].merge(
            cells[[id_column, *CELL_KEY]], on=id_column, how="left", validate="many_to_one")
        unmapped = int(table["plate"].isna().sum())
        if unmapped:
            raise ValueError(f"{unmapped} labelled crops have no row in the mapping")
    elif not has_key:
        raise ValueError(f"the label table needs {list(CELL_KEY)} or a mapping")
    table["stage"] = table["_class"].map(_normalise_class)
    known = set(MITOTIC_CLASSES) | set(INTERPHASE_CLASSES) | set(EXCLUDED_CLASSES)
    unknown = sorted(set(table["stage"]) - known)
    if unknown:
        raise ValueError(f"unknown label classes {unknown}")
    if artefacts not in ("drop", "negative"):
        raise ValueError(f"artefacts must be 'drop' or 'negative'; got {artefacts!r}")
    if artefacts == "drop":
        table = table[~table["stage"].isin(EXCLUDED_CLASSES)]
    table["mitotic"] = table["stage"].isin(MITOTIC_CLASSES)
    keyed = table[[*CELL_KEY, "stage", "mitotic"]].drop_duplicates()
    clash = keyed.duplicated(list(CELL_KEY), keep=False)
    if clash.any():
        raise ValueError(f"{int(clash.sum())} rows label the same cell with different classes")
    return keyed.reset_index(drop=True)


def mitotic_features(columns: Sequence[str], *, dna: str = "dapi",
                     tubulin: Optional[str] = "gfp") -> List[str]:
    """The DNA and tubulin features a mitotic classifier is trained on.

    :param columns: the cell table's columns.
    :param dna: the DNA channel name in the column names.
    :param tubulin: the tubulin channel name, or None to use DNA only.
    :returns: in table order, every nuclear shape feature, every
        ``nucleus_<dna>_*`` and ``cell_<dna>_*`` feature and, with a tubulin
        channel, every ``cell_<tubulin>_*`` and ``cytoplasm_<tubulin>_*``
        feature.
    """
    import re

    dna = dna.lower()
    prefixes = [f"nucleus_{dna}_", f"cell_{dna}_"]
    if tubulin:
        prefixes += [f"cell_{tubulin.lower()}_", f"cytoplasm_{tubulin.lower()}_"]
    names = [str(c).lower() for c in columns]
    channels = {m.group(1) for m in (re.match(r"^(?:cell|nucleus|cytoplasm)_([a-z0-9]+)_"
                                              r"mean_intensity$", n) for n in names) if m}
    out = []
    for column, name in zip(columns, names):
        if name.startswith(tuple(prefixes)):
            out.append(column)
        elif name.startswith("nucleus_") and name.split("_")[1] not in channels:
            out.append(column)
    return out


def _well_robust_z(cells: pd.DataFrame, features: Sequence[str],
                   group: Optional[str]) -> np.ndarray:
    """Every feature as a robust z against all cells of its group.

    With ``group`` None the values are returned unchanged, as float32.
    Computed one column and one group at a time, so a whole plate needs only
    the float32 result in memory.
    """
    if group is None:
        return cells[list(features)].to_numpy(np.float32)
    groups = pd.factorize(cells[group])[0]
    rows_of = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = np.empty((len(cells), len(features)), np.float32)
    for j, feature in enumerate(features):
        column = cells[feature].to_numpy(np.float64)
        for rows in rows_of:
            block = column[rows]
            median = np.nanmedian(block) if len(block) else np.nan
            mad = 1.4826 * np.nanmedian(np.abs(block - median)) if len(block) else np.nan
            out[rows, j] = (block - median) / mad if mad > 0 else np.nan
    return out


def _model(kind: str, seed: int, balanced: bool = True):
    """The classifier named ``kind``."""
    weight = "balanced" if balanced else None
    if kind == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              class_weight=weight, random_state=seed)
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight=weight))
    raise ValueError(f"model must be 'hist_gradient_boosting' or 'logistic'; got {kind!r}")


def _fit(model, x, y, weight):
    """Fit ``model``, with sample weights when given."""
    if weight is None:
        return model.fit(x, y)
    if hasattr(model, "steps"):
        return model.fit(x, y, **{f"{model.steps[-1][0]}__sample_weight": weight})
    return model.fit(x, y, sample_weight=weight)


def _fold_scores(truth: np.ndarray, probability: np.ndarray, threshold: float,
                 weight: Optional[np.ndarray] = None) -> Dict[str, float]:
    """The :data:`FOLD_METRICS` of one fold, weighted when ``weight`` is given."""
    from sklearn import metrics

    called = probability >= threshold
    two = len(np.unique(truth)) == 2
    w = {"sample_weight": weight}
    return {
        "accuracy": float(metrics.accuracy_score(truth, called, **w)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(truth, called, **w)),
        "precision": float(metrics.precision_score(truth, called, zero_division=0, **w)),
        "recall": float(metrics.recall_score(truth, called, zero_division=0, **w)),
        "f1": float(metrics.f1_score(truth, called, zero_division=0, **w)),
        "roc_auc": float(metrics.roc_auc_score(truth, probability, **w)) if two else float("nan"),
        "average_precision": float(metrics.average_precision_score(truth, probability, **w))
        if two else float("nan"),
    }


def _cross_validate(x, y, groups, kind, n_splits, seed, threshold, weight=None,
                    balanced=True):
    """Grouped cross-validation: per-fold scores and out-of-fold probabilities."""
    from sklearn.model_selection import GroupKFold

    folds = GroupKFold(n_splits=n_splits)
    probability = np.full(len(y), np.nan)
    rows = []
    for k, (train, test) in enumerate(folds.split(x, y, groups)):
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"fold {k} has only one class in its training cells; "
                             "label more cells or use fewer folds")
        model = _fit(_model(kind, seed, balanced), x[train], y[train],
                     None if weight is None else weight[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
        rows.append({"fold": k, "test_cells": int(len(test)),
                     "test_mitotic": int(y[test].sum()),
                     "test_groups": int(len(np.unique(groups[test]))),
                     **_fold_scores(y[test], probability[test], threshold,
                                    None if weight is None else weight[test])})
    return pd.DataFrame(rows), probability


def sampling_weights(strata: Sequence[Any], population_shares: Mapping[Any, float],
                     groups: Optional[Sequence[Any]] = None) -> np.ndarray:
    """Post-stratification weights that make a stratified label set stand for the screen.

    :param strata: the sampling stratum of every labelled cell, for example
        the DNA rule's call (``True``/``False``), when the label set
        over-samples rule-mitotic cells.
    :param population_shares: each stratum's share of all cells, as
        ``{stratum: share}``, or ``{(group, stratum): share}`` with
        ``groups``.
    :param groups: an optional group (the well) per labelled cell, when the
        shares differ between groups. Every group then weighs as much as
        its labelled cells.
    :returns: one weight per labelled cell, averaging 1: a cell stands for
        ``share / labelled cells of its stratum``.
    :raises ValueError: when a stratum has no share.
    """
    strata = pd.Series(list(strata))
    keys = strata if groups is None else pd.Series(list(zip(groups, strata)))
    counts = keys.map(keys.value_counts())
    missing = sorted({str(k) for k in keys if k not in population_shares})
    if missing:
        raise ValueError(f"no population share for strata {missing}")
    w = keys.map(population_shares).to_numpy(float) / counts.to_numpy(float)
    if groups is not None:
        g = pd.Series(list(groups))
        size = g.map(g.value_counts()).to_numpy(float)
        total = pd.Series(w).groupby(g.to_numpy()).transform("sum").to_numpy()
        w = w / total * size
    return w * len(w) / w.sum()


def train_mitotic_classifier(cells: pd.DataFrame, labels: Sequence[bool],
                             features: Optional[Sequence[str]] = None, *,
                             groups: Optional[Sequence[Any]] = None,
                             normalise_by: Optional[str] = "well",
                             model: str = "hist_gradient_boosting", n_splits: int = 5,
                             seed: int = 0, threshold: float = 0.5,
                             shuffled_control: bool = True,
                             sample_weight: Optional[Sequence[float]] = None,
                             balanced: Optional[bool] = None) -> Dict[str, Any]:
    """Train a mitotic classifier and validate it with grouped cross-validation.

    :param cells: one row per labelled cell, with the feature columns and the
        ``normalise_by`` column.
    :param labels: True for a mitotic cell, one per row of ``cells``.
    :param features: the columns to learn from; :func:`mitotic_features`
        of the table's columns when None.
    :param groups: the cross-validation groups, one per cell: the well, or
        the field (well and site). Cells of one group are always in the same
        fold. Required: a random split puts neighbours from the same image
        on both sides and overstates the accuracy.
    :param normalise_by: every feature is first made a robust z against all
        cells of this column's groups (the well), so the model travels
        between wells and plates; None uses the raw values.
    :param model: ``'hist_gradient_boosting'`` (the default) or
        ``'logistic'``, a simpler baseline.
    :param n_splits: the number of folds; at most the number of groups.
    :param seed: the model seed and the label-shuffle seed.
    :param threshold: the probability above which a cell is called mitotic.
    :param shuffled_control: also cross-validate on labels shuffled within
        each group, which keeps every group's mitotic share. Its scores are
        what a model that learnt nothing achieves.
    :param sample_weight: one weight per cell, used in training and in
        every metric. Pass :func:`sampling_weights` when the labelled cells
        were not drawn at random (for example half of them from the rule's
        mitotic calls); the metrics then describe the whole screen and the
        model learns the screen's mitotic share, not the label set's.
    :param balanced: weigh the two classes equally in training. None (the
        default) does so without ``sample_weight`` and not with it, since
        the weights already carry the screen's class balance.
    :returns: the trained bundle: ``model`` (fitted on every labelled cell),
        ``features`` (without those that are empty in every labelled cell,
        such as a feature whose well spread is zero; they are listed in
        ``summary["features_dropped"]``), ``normalise_by``, ``threshold``, ``folds`` (one row
        per fold), ``summary`` (mean and spread of every metric, the control
        and the label counts) and ``out_of_fold`` (each cell's held-out
        probability). Save it with :func:`save_mitotic_classifier`.
    :raises ValueError: for mismatched lengths, missing groups, one class
        only, or fewer groups than folds.
    """
    y = np.asarray(labels, bool)
    if len(y) != len(cells):
        raise ValueError(f"labels has {len(y)} entries for {len(cells)} cells")
    if groups is None:
        raise ValueError("groups is required: pass the well or field of every cell, so "
                         "cross-validation never splits one image between folds")
    groups = np.asarray(groups)
    if len(groups) != len(cells):
        raise ValueError(f"groups has {len(groups)} entries for {len(cells)} cells")
    if len(np.unique(y)) < 2:
        raise ValueError("labels must contain both mitotic and non-mitotic cells")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"{len(np.unique(groups))} groups cannot make {n_splits} folds")
    features = list(features) if features is not None else mitotic_features(cells.columns)
    if not features:
        raise ValueError("no features to learn from; pass features explicitly")
    if normalise_by is not None and normalise_by not in cells:
        raise ValueError(f"normalise_by column {normalise_by!r} is not in the cell table")
    x = _well_robust_z(cells, features, normalise_by)
    usable = np.isfinite(x).any(axis=0)
    dropped = [f for f, ok in zip(features, usable) if not ok]
    if not usable.any():
        raise ValueError("every feature is empty after normalisation")
    features = [f for f, ok in zip(features, usable) if ok]
    x = x[:, usable]
    weight = None if sample_weight is None else np.asarray(sample_weight, float)
    if weight is not None and len(weight) != len(y):
        raise ValueError(f"sample_weight has {len(weight)} entries for {len(y)} cells")
    balanced = (weight is None) if balanced is None else bool(balanced)
    folds, probability = _cross_validate(x, y, groups, model, n_splits, seed, threshold,
                                         weight, balanced)
    summary: Dict[str, Any] = {"cells": int(len(y)), "mitotic": int(y.sum()),
                               "groups": int(len(np.unique(groups))), "folds": int(n_splits),
                               "model": model, "features": len(features),
                               "features_dropped": dropped}
    for metric in FOLD_METRICS:
        summary[metric] = float(folds[metric].mean())
        summary[f"{metric}_sd"] = float(folds[metric].std(ddof=1)) if len(folds) > 1 else 0.0
    summary["majority_class_accuracy"] = float(max(y.mean(), 1 - y.mean()))
    if shuffled_control:
        rng = np.random.default_rng(seed)
        shuffled = y.copy()
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            shuffled[rows] = rng.permutation(shuffled[rows])
        control, _ = _cross_validate(x, shuffled, groups, model, n_splits, seed, threshold,
                                     weight, balanced)
        for metric in FOLD_METRICS:
            summary[f"shuffled_{metric}"] = float(control[metric].mean())
    summary["weighted"] = weight is not None
    summary["balanced_classes"] = balanced
    fitted = _fit(_model(model, seed, balanced), x, y, weight)
    return {"format": MODEL_FORMAT, "model": fitted, "features": features,
            "normalise_by": normalise_by, "threshold": float(threshold), "folds": folds,
            "summary": summary, "out_of_fold": probability}


def predict_mitotic(bundle: Mapping[str, Any], cells: pd.DataFrame, *,
                    probability: bool = False) -> pd.Series:
    """Call cells mitotic with a trained classifier.

    :param bundle: :func:`train_mitotic_classifier` or
        :func:`load_mitotic_classifier` output.
    :param cells: one row per cell with the model's features and its
        ``normalise_by`` column.
    :param probability: return the probability instead of the call.
    :returns: a Series over ``cells``: booleans, or probabilities.
    :raises ValueError: when the table lacks a feature the model needs.
    """
    missing = [c for c in bundle["features"] if c not in cells]
    group = bundle.get("normalise_by")
    if group is not None and group not in cells:
        missing.append(group)
    if missing:
        raise ValueError(f"the cell table lacks columns the mitotic model needs: {missing}")
    x = _well_robust_z(cells, bundle["features"], group)
    p = bundle["model"].predict_proba(x)[:, 1] if len(x) else np.zeros(0)
    if probability:
        return pd.Series(p, index=cells.index, name="mitotic_probability")
    return pd.Series(p >= float(bundle["threshold"]), index=cells.index, name="mitotic")


def save_mitotic_classifier(bundle: Mapping[str, Any], path: str) -> str:
    """Write a trained classifier to ``path`` (joblib).

    :param bundle: :func:`train_mitotic_classifier` output.
    :param path: the file to write.
    :returns: ``path``.
    """
    import joblib

    keep = {k: bundle[k] for k in ("format", "model", "features", "normalise_by",
                                   "threshold", "summary")}
    keep["folds"] = bundle["folds"].to_dict("records")
    joblib.dump(keep, path)
    return path


def load_mitotic_classifier(path: str) -> Dict[str, Any]:
    """Read a classifier written by :func:`save_mitotic_classifier`.

    :param path: the file.
    :returns: the bundle :func:`predict_mitotic` takes.
    :raises ValueError: when the file is not a spaCR mitotic classifier.

    A joblib file can run code when it is loaded, like any pickle; load only
    classifiers you trained or trust.
    """
    import joblib

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a spaCR mitotic classifier")
    return bundle


def mitotic_index_agreement(genes: pd.DataFrame, reference: pd.DataFrame, *,
                            min_cells: Sequence[int] = (50, 100, 200, 500),
                            shuffles: int = 200, seed: int = 2) -> pd.DataFrame:
    """Spearman correlation of per-gene mitotic indices with a reference.

    :param genes: ``gene, n_cells, mitotic_index`` (the all-cell gene table).
    :param reference: ``gene, mitotic_index`` rows, such as Funk et al.'s;
        indices above 1 are dropped as damaged rows.
    :param min_cells: the cell-count floors to report.
    :param shuffles: the shuffled-index draws of the control.
    :param seed: the shuffle seed.
    :returns: one row per floor: ``min_cells, genes, spearman,
        shuffled_mean, shuffled_p95_abs``.
    """
    from scipy.stats import spearmanr

    ref = reference[reference["mitotic_index"].notna() & (reference["mitotic_index"] <= 1)]
    merged = genes.merge(ref[["gene", "mitotic_index"]], on="gene", suffixes=("", "_reference"))
    rows = []
    for floor in min_cells:
        sub = merged[merged["n_cells"] >= floor]
        if len(sub) < 10:
            continue
        rho = spearmanr(sub["mitotic_index"], sub["mitotic_index_reference"]).correlation
        rng = np.random.default_rng(seed)
        null = [spearmanr(rng.permutation(sub["mitotic_index"].to_numpy()),
                          sub["mitotic_index_reference"]).correlation for _ in range(shuffles)]
        rows.append({"min_cells": int(floor), "genes": int(len(sub)), "spearman": float(rho),
                     "shuffled_mean": float(np.mean(null)) if null else float("nan"),
                     "shuffled_p95_abs": float(np.percentile(np.abs(null), 95))
                     if null else float("nan")})
    return pd.DataFrame(rows, columns=["min_cells", "genes", "spearman", "shuffled_mean",
                                       "shuffled_p95_abs"])
