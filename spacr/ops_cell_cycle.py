"""Cell-cycle stage, probability and verification flag of every OPS cell (item 990, alpha).

A tabular model (HistGradientBoosting or XGBoost on the DNA and microtubule features of
:func:`spacr.ops_mitotic.mitotic_features`, each a robust z against all cells of its well)
calls interphase, g2_prophase (a flag class, never counted as mitotic), prometaphase,
metaphase, ana_telo_cyto (anaphase and telophase/cytokinesis), telophase_daughters (the two
2N daughters, a stage of their own; maintainer 2026-11-20) and artefact. Interphase cells take G1, S or G2 from the field-centred DNA-content fit of
item 995 (:func:`_dna_phases`). Self-checks S1-S6 run on every plate; a stage that fails its
check collapses into its parent (:data:`PARENTS`) and is flagged ``unresolved``, on every
plate of the screen once any plate collapses it. A stage whose out-of-fold F1 on the user's
labels clears the shuffled-label control is ``verified``; every other stage is ``auto``.

The model starts from a base checkpoint (:func:`load_cell_cycle_checkpoint`, written by
:func:`train_cell_cycle_model` and :func:`save_cell_cycle_checkpoint`) and, with user
labels, is refitted on the base labels plus the screen labels weighted five times.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CLASSES = ("interphase", "g2_prophase", "prometaphase", "metaphase", "ana_telo_cyto",
           "telophase_daughters", "artefact")
MITOTIC_STAGES = ("prometaphase", "metaphase", "ana_telo_cyto", "telophase_daughters")
DAUGHTER_STAGES = ("ana_telo_cyto", "telophase_daughters")
PHASES = ("G1", "S", "G2")
PARENTS = {"prometaphase": "mitotic", "metaphase": "mitotic", "ana_telo_cyto": "mitotic",
           "telophase_daughters": "mitotic",
           "mitotic": "G2/M", "g2_prophase": "G2", "G1": "interphase", "S": "interphase",
           "G2": "interphase"}
CHECKPOINT_FORMAT = "spacr-ops-cell-cycle-1"
ARREST_GENES = ("KIF11", "TPX2", "CENPE", "KIF18A", "NUF2", "NDC80", "SPC24", "SPC25", "CKAP5",
                "CDC20", "PLK1", "AURKA", "ANAPC1", "ANAPC2", "ANAPC4", "ANAPC5", "ANAPC10",
                "ANAPC11", "CDC16", "CDC23", "CDC27")
FAMILIES = ("tabular", "cnn", "transformer")
TABULAR = ("hgb", "xgboost")
FINETUNE = ("screen", "plate", "none")
SCREEN_WEIGHT = 5.0
G2_G1_RANGE = (1.6, 2.4)
FIT_R2 = 0.9
PERMUTATIONS = 500
VERIFIED_F1 = 0.7
KEY = ["plate", "well", "site", "label"]


def collapse_stage(stage: str, failed: Sequence[str]) -> str:
    """A stage after the collapse rule: up the parent chain while the stage failed its check.

    :param stage: the called stage.
    :param failed: the stages (and parents, such as ``'mitotic'``) whose checks failed.
    :returns: the first stage up the chain of :data:`PARENTS` that did not fail.
    """
    failed = set(failed)
    while stage in failed and stage in PARENTS:
        stage = PARENTS[stage]
    return stage


def screen_collapses(per_plate: Mapping[str, Sequence[str]]) -> List[str]:
    """The stages collapsed for the whole screen: any stage a plate collapses.

    :param per_plate: the failed stages of each plate.
    :returns: their union, sorted, so every plate's tables stay comparable.
    """
    return sorted(set().union(*[set(v) for v in per_plate.values()]) if per_plate else set())


def _features(columns: Sequence[str], dna: str, mt: Optional[str]) -> List[str]:
    """The model's features: :func:`spacr.ops_mitotic.mitotic_features`."""
    from .ops_mitotic import mitotic_features

    features = mitotic_features(columns, dna=dna, tubulin=mt)
    if not features:
        raise ValueError(f"the cell tables have no feature of the DNA channel {dna!r}")
    return features


def _matrix(cells: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """The features as robust z against all cells of each well (item 856's normalisation)."""
    from .ops_mitotic import _well_robust_z

    group = "well" if "well" in cells.columns else None
    return np.nan_to_num(_well_robust_z(cells, features, group), nan=0.0)


def _learner(tabular: str, seed: int):
    """An unfitted multiclass learner: :func:`spacr.ops_mitotic._model`."""
    from .ops_mitotic import _model

    if tabular not in TABULAR:
        raise ValueError(f"ops_cell_cycle_tabular must be one of {list(TABULAR)}; got {tabular!r}")
    return _model("hist_gradient_boosting" if tabular == "hgb" else "xgboost", seed)


def _fit(tabular: str, seed: int, x: np.ndarray, y: np.ndarray,
         weight: Optional[np.ndarray] = None):
    """A fitted learner on class indices ``y``."""
    from .ops_mitotic import _fit as fit

    return fit(_learner(tabular, seed), x, y, weight)


def _probabilities(model, x: np.ndarray) -> np.ndarray:
    """Probabilities over every class of :data:`CLASSES`, zero for a class the model never saw."""
    out = np.zeros((len(x), len(CLASSES)))
    if len(x):
        out[:, np.asarray(model.classes_, int)] = model.predict_proba(x)
    return out


def _encode(labels: Sequence[Any]) -> np.ndarray:
    """Class indices of stage names; an unknown name raises ValueError."""
    index = {name: i for i, name in enumerate(CLASSES)}
    unknown = sorted({str(v) for v in labels} - set(index))
    if unknown:
        raise ValueError(f"cell-cycle labels must be one of {list(CLASSES)}; got {unknown}")
    return np.array([index[str(v)] for v in labels], int)


def _folds(groups: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> List[np.ndarray]:
    """Group-kept fold assignments: whole groups, shuffled with ``seed``, dealt round-robin."""
    names = np.unique(groups)
    order = np.random.default_rng(seed).permutation(len(names))
    fold_of = {names[j]: i % n_splits for i, j in enumerate(order)}
    return np.array([fold_of[g] for g in groups], int)


def _out_of_fold(tabular: str, seed: int, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 weight: Optional[np.ndarray] = None, scored: Optional[np.ndarray] = None,
                 n_splits: int = 5) -> np.ndarray:
    """Out-of-fold class predictions of the ``scored`` rows (all rows when None)."""
    scored = np.ones(len(y), bool) if scored is None else scored
    folds = _folds(groups, y, seed, n_splits=min(n_splits, len(np.unique(groups))))
    out = np.full(len(y), -1, int)
    for fold in np.unique(folds):
        test = (folds == fold) & scored
        train = folds != fold
        if not test.any() or len(np.unique(y[train])) < 2:
            continue
        model = _fit(tabular, seed, x[train], y[train],
                     None if weight is None else weight[train])
        out[test] = np.argmax(_probabilities(model, x[test]), axis=1)
    return out


def _per_class_f1(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """F1 of every class present in ``truth`` (rows predicted -1 are left out)."""
    keep = predicted >= 0
    out = {}
    for i, name in enumerate(CLASSES):
        t, p = truth[keep] == i, predicted[keep] == i
        if t.any():
            tp = float((t & p).sum())
            out[name] = 2 * tp / max(float(t.sum() + p.sum()), 1.0)
    return out


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    """The mean per-class F1."""
    scores = _per_class_f1(truth, predicted)
    return float(np.mean(list(scores.values()))) if scores else float("nan")


def _bootstrap_f1(truth: np.ndarray, predicted: np.ndarray, groups: np.ndarray, seed: int,
                  draws: int = 200) -> Dict[str, Tuple[float, float]]:
    """95 % field-bootstrap confidence interval of every class F1."""
    names = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in names]
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}
    for _ in range(draws):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(names), len(names))])
        for name, value in _per_class_f1(truth[rows], predicted[rows]).items():
            samples.setdefault(name, []).append(value)
    return {name: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for name, v in samples.items()}


def _shuffled_f1(tabular: str, seed: int, x, y, groups, weight, scored, permutations: int
                 ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Macro and per-class out-of-fold F1 of models trained on permuted labels.

    Permutation ``p`` shuffles the ``scored`` rows' labels (the ones judged; a base model's
    own labels stay) with seed ``seed + 1 + p`` and fits with ``seed``.
    """
    macro, per_class = [], {}
    for p in range(permutations):
        shuffled = y.copy()
        shuffled[scored] = np.random.default_rng(seed + 1 + p).permutation(y[scored])
        predicted = _out_of_fold(tabular, seed, x, shuffled, groups, weight, scored)
        macro.append(_macro_f1(y[scored], predicted[scored]))
        for name, value in _per_class_f1(y[scored], predicted[scored]).items():
            per_class.setdefault(name, []).append(value)
    return np.array(macro), {k: np.array(v) for k, v in per_class.items()}


def train_cell_cycle_model(cells: pd.DataFrame, labels: Sequence[str], *, input_mode: str = "DNA",
                           dna: str = "dapi", mt: Optional[str] = None, tabular: str = "hgb",
                           seed: int = 0, groups: Optional[Sequence[Any]] = None,
                           repeats: int = 5, permutations: int = PERMUTATIONS,
                           data_files: Sequence[str] = ()) -> Dict[str, Any]:
    """A base cell-cycle model and its card, from labelled cells (the one call that trains it).

    :param cells: the labelled cells: ``well``, the field column of ``groups`` and the
        feature columns of the cell tables.
    :param labels: one stage of :data:`CLASSES` per cell.
    :param input_mode: ``'DNA'`` or ``'DNA+MT'``.
    :param dna: the DNA channel name in the feature columns.
    :param mt: the microtubule channel name; required for ``'DNA+MT'``.
    :param tabular: ``'hgb'`` or ``'xgboost'``.
    :param seed: the seed of the folds, the shuffles and the learner.
    :param groups: the cross-validation group of every cell (fields); the ``well`` and
        ``site`` pair when None.
    :param repeats: repeats of the field-grouped 5-fold cross-validation.
    :param permutations: shuffled-label models of the control.
    :param data_files: the training data files, recorded with their SHA-256 in the card.
    :returns: ``{"model", "features", "card", "training"}``; ``training`` keeps the labelled
        rows (features and labels) that fine-tuning refits on.
    :raises ValueError: for an unknown mode or stage, or 'DNA+MT' without ``mt``.
    """
    if input_mode not in ("DNA", "DNA+MT"):
        raise ValueError(f"input_mode must be 'DNA' or 'DNA+MT'; got {input_mode!r}")
    if input_mode == "DNA+MT" and not mt:
        raise ValueError("'DNA+MT' needs the microtubule channel")
    features = _features(cells.columns, dna, mt if input_mode == "DNA+MT" else None)
    x = _matrix(cells, features)
    y = _encode(labels)
    groups = (np.asarray(groups) if groups is not None else
              (cells["well"].astype(str) + "/" + cells["site"].astype(str)).to_numpy())
    metrics = []
    for r in range(int(repeats)):
        predicted = _out_of_fold(tabular, seed + r, x, y, groups)
        metrics.append({"repeat": r, "macro_f1": _macro_f1(y, predicted),
                        "f1": _per_class_f1(y, predicted)})
    predicted = _out_of_fold(tabular, seed, x, y, groups)
    shuffled, _per = _shuffled_f1(tabular, seed, x, y, groups, None, np.ones(len(y), bool),
                                  int(permutations))
    card = {"format": CHECKPOINT_FORMAT, "family": "tabular", "tabular": tabular,
            "input": input_mode, "classes": list(CLASSES), "parents": PARENTS,
            "features": list(features), "dna": dna, "mt": mt,
            "normalisation": "robust z per well (spacr.ops_mitotic._well_robust_z)",
            "data": [{"path": str(p), "sha256": _sha256(str(p))} for p in data_files],
            "label_counts": {CLASSES[i]: int(n) for i, n in
                             zip(*np.unique(y, return_counts=True))},
            "cv": {"scheme": f"field-grouped 5-fold x {int(repeats)}", "repeats": metrics,
                   "f1": _per_class_f1(y, predicted),
                   "f1_ci95": _bootstrap_f1(y, predicted, groups, seed),
                   "shuffled_macro_f1_p975": (float(np.percentile(shuffled, 97.5))
                                              if len(shuffled) else None),
                   "permutations": int(permutations)},
            "seed": int(seed), "spacr_version": _version()}
    model = _fit(tabular, seed, x, y)
    training = pd.DataFrame(x, columns=list(features))
    training["stage"] = [CLASSES[i] for i in y]
    return {"model": model, "features": list(features), "card": card, "training": training}


def _version() -> str:
    """The running spaCR version."""
    from .version import __version__

    return str(__version__)


def _sha256(path: str) -> str:
    """The SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cell_cycle_checkpoint(bundle: Mapping[str, Any], folder: str, *,
                               base: Optional[str] = None) -> str:
    """Write a checkpoint directory ``cell_cycle_tabular_<input>_<hash12>``.

    It holds ``model.joblib`` (the learner, its features and its training rows),
    ``card.json`` and ``SHA256SUMS``; ``hash12`` is the start of the SHA-256 of
    ``model.joblib``, so two identical models get the same name.

    :param bundle: from :func:`train_cell_cycle_model` or a fine-tune.
    :param folder: where the directory is made.
    :param base: the base checkpoint's ``model.joblib`` SHA-256, recorded in the card of a
        fine-tuned checkpoint.
    :returns: the checkpoint directory.
    """
    import tempfile

    import joblib

    os.makedirs(folder, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".cell_cycle_", dir=folder)
    model_path = os.path.join(stage, "model.joblib")
    joblib.dump({"model": bundle["model"], "features": list(bundle["features"]),
                 "training": bundle.get("training")}, model_path)
    digest = _sha256(model_path)
    card = dict(bundle["card"], model_sha256=digest, base_sha256=base)
    with open(os.path.join(stage, "card.json"), "w", encoding="utf-8") as handle:
        json.dump(card, handle, indent=1, sort_keys=True, default=str)
    sums = {name: _sha256(os.path.join(stage, name)) for name in ("card.json", "model.joblib")}
    with open(os.path.join(stage, "SHA256SUMS"), "w", encoding="utf-8") as handle:
        handle.writelines(f"{value}  {name}\n" for name, value in sorted(sums.items()))
    target = os.path.join(folder, f"cell_cycle_tabular_{card['input'].replace('+', '_')}"
                                  f"_{digest[:12]}")
    if os.path.isdir(target):
        import shutil

        shutil.rmtree(stage)
    else:
        os.replace(stage, target)
    return target


def load_cell_cycle_checkpoint(path: str) -> Dict[str, Any]:
    """A checkpoint written by :func:`save_cell_cycle_checkpoint`, checked before unpickling.

    The card's ``format`` and the SHA-256 of every file in ``SHA256SUMS`` are checked first;
    only then is ``model.joblib`` loaded, with a warning, as
    :func:`spacr.ops_mitotic.load_mitotic_classifier` does: unpickling runs code, so load only
    checkpoints from the model registry or from paths you trust.

    :param path: the checkpoint directory.
    :returns: ``{"model", "features", "training", "card", "sha256"}``.
    :raises ValueError: for a missing file, a wrong format, a SHA-256 mismatch or a class
        list other than :data:`CLASSES`.
    """
    import warnings

    import joblib

    sums_path = os.path.join(path, "SHA256SUMS")
    if not os.path.isfile(sums_path):
        raise ValueError(f"{path} is not a cell-cycle checkpoint: no SHA256SUMS")
    with open(sums_path, encoding="utf-8") as handle:
        sums = dict(reversed(line.split()) for line in handle if line.strip())
    for name in ("card.json", "model.joblib"):
        target = os.path.join(path, name)
        if name not in sums or not os.path.isfile(target) or _sha256(target) != sums[name]:
            raise ValueError(f"{name} of {path} does not match its SHA-256; refusing to load")
    with open(os.path.join(path, "card.json"), encoding="utf-8") as handle:
        card = json.load(handle)
    if card.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} has format {card.get('format')!r}, not {CHECKPOINT_FORMAT!r}")
    if list(card.get("classes") or []) != list(CLASSES):
        raise ValueError(f"{path} was trained on the classes {card.get('classes')}; this spaCR calls "
                         f"{list(CLASSES)} (telophase_daughters became a stage of its own). "
                         "Train the base model again with spacr.ops_cell_cycle.train_cell_cycle_model")
    warnings.warn(f"loading the cell-cycle model {path}: unpickling runs code; load only "
                  "checkpoints you trust", stacklevel=2)
    held = joblib.load(os.path.join(path, "model.joblib"))
    return {**held, "card": card, "sha256": sums["model.joblib"]}


def fine_tune(base: Mapping[str, Any], cells: pd.DataFrame, labels: Sequence[str], *,
              seed: int = 0, tabular: Optional[str] = None) -> Dict[str, Any]:
    """The base model refitted on its own labels plus the screen's, the screen's weighted ×5.

    :param base: from :func:`load_cell_cycle_checkpoint` (its ``training`` rows are used).
    :param cells: the screen's labelled cells.
    :param labels: their stages.
    :param seed: the learner seed.
    :param tabular: the learner; the base card's when None.
    :returns: a bundle for :func:`save_cell_cycle_checkpoint`, its card recording the base.
    """
    tabular = tabular or base["card"].get("tabular", "hgb")
    features = list(base["features"])
    x_screen = _matrix(cells, features)
    y_screen = _encode(labels)
    training = base.get("training")
    x_base = training[features].to_numpy() if training is not None else np.empty((0, len(features)))
    y_base = _encode(training["stage"]) if training is not None else np.empty(0, int)
    x = np.vstack([x_base, x_screen])
    y = np.concatenate([y_base, y_screen])
    weight = np.concatenate([np.ones(len(y_base)), np.full(len(y_screen), SCREEN_WEIGHT)])
    model = _fit(tabular, seed, x, y, weight)
    card = dict(base["card"], fine_tuned=True, screen_labels=int(len(y_screen)),
                screen_weight=SCREEN_WEIGHT, seed=int(seed), tabular=tabular)
    rows = pd.DataFrame(x, columns=features)
    rows["stage"] = [CLASSES[i] for i in y]
    return {"model": model, "features": features, "card": card, "training": rows}


def predict_stages(bundle: Mapping[str, Any], cells: pd.DataFrame) -> pd.DataFrame:
    """Every class probability and the model's stage of every cell.

    :param bundle: a loaded or fitted model (``model`` and ``features``).
    :param cells: the cell table, with every feature of the model.
    :returns: on the index of ``cells``, ``p_<class>`` for each of :data:`CLASSES` and
        ``model_stage``.
    :raises ValueError: when a feature is missing.
    """
    missing = [f for f in bundle["features"] if f not in cells.columns]
    if missing:
        raise ValueError(f"the cell tables lack the model's features {missing[:5]}")
    p = _probabilities(bundle["model"], _matrix(cells, bundle["features"]))
    out = pd.DataFrame(p, index=cells.index, columns=[f"p_{c}" for c in CLASSES])
    out["model_stage"] = [CLASSES[i] for i in np.argmax(p, axis=1)]
    return out


def _dna_phases(cells: pd.DataFrame, settings: Mapping[str, Any], mitotic: np.ndarray
                ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """G1 / S / G2 of every cell from item 995's field-centred DNA fit (the one call into 995).

    :param cells: ``_dna_integrated``, ``well`` and ``site`` per cell.
    :param settings: read for ``cell_cycle_field_min_cells`` (default 50).
    :param mitotic: True for the cells the model calls mitotic, g2_prophase or artefact;
        they stay out of the field modes and the fit.
    :returns: the frame and fits of :func:`spacr.measure._cell_table_phases_measure_fc`.
    """
    from . import measure

    table = cells[["_dna_integrated", "well", "site"]].copy()
    table["mitotic"] = np.asarray(mitotic, bool)
    min_cells = int(settings.get("cell_cycle_field_min_cells") or 50)
    return measure._cell_table_phases_measure_fc(table, dna="_dna_integrated", group="well",
                                                 field="site", mitotic="mitotic",
                                                 min_cells=min_cells)


def _fit_quality(content: np.ndarray, fit) -> Tuple[float, float]:
    """``(R², G2/G1)`` of a DNA fit: the fitted density against the 100-bin histogram."""
    content = np.asarray(content, float)
    content = content[np.isfinite(content)]
    if len(content) < 10:
        return float("nan"), float(fit.g2 / fit.g1)
    edges = np.linspace(*np.percentile(content, [0.5, 99.5]), 101)
    counts, _ = np.histogram(content, edges)
    centres = (edges[:-1] + edges[1:]) / 2
    model = fit.densities(centres).sum(axis=1) * len(content) * np.diff(edges)
    total = ((counts - counts.mean()) ** 2).sum()
    r2 = 1 - ((counts - model) ** 2).sum() / total if total > 0 else float("nan")
    return float(r2), float(fit.g2 / fit.g1)


def _nuclear_diameter(cells: pd.DataFrame) -> np.ndarray:
    """Each nucleus's equivalent diameter from ``nucleus_area``."""
    return 2.0 * np.sqrt(cells["nucleus_area"].to_numpy(float) / np.pi)


def _pair_share(cells: pd.DataFrame, called: np.ndarray, two_n: np.ndarray,
                log2_dna: np.ndarray) -> float:
    """Share of 2N daughter-stage calls with a 2N daughter-stage partner nearby (S2's statistic)."""
    from scipy.spatial import cKDTree

    hits, total = 0, 0
    y, x = cells["nucleus_centroid_y"].to_numpy(float), cells["nucleus_centroid_x"].to_numpy(float)
    diameter = _nuclear_diameter(cells)
    fields = (cells["well"].astype(str) + "/" + cells["site"].astype(str)).to_numpy()
    for field in np.unique(fields[called & two_n]):
        rows = np.flatnonzero((fields == field) & called & two_n)
        total += len(rows)
        if len(rows) < 2:
            continue
        tree = cKDTree(np.c_[y[rows], x[rows]])
        for k, row in enumerate(rows):
            near = tree.query_ball_point([y[row], x[row]], 2.5 * diameter[row])
            if any(j != k and abs(log2_dna[rows[j]] - log2_dna[row]) <= 0.3 for j in near):
                hits += 1
    return hits / total if total else float("nan")


def self_checks(cells: pd.DataFrame, stages: pd.DataFrame, dna: pd.DataFrame,
                fits: Mapping[str, Any], *, card: Mapping[str, Any], seed: int = 0,
                arrest_genes: Optional[Sequence[str]] = ARREST_GENES,
                control_prefix: str = "nontargeting", permutations: int = PERMUTATIONS,
                labelled: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Checks S1-S6 of one plate.

    :param cells: the plate's cells: ``well``, ``site``, ``nucleus_area``,
        ``nucleus_centroid_y``/``_x`` (S2) and ``gene`` (S5) when present.
    :param stages: from :func:`predict_stages`.
    :param dna: the frame of :func:`_dna_phases`.
    :param fits: its fits per group.
    :param card: the model card; its ``reference`` S1 value and S4 φ, when present, tighten
        S1 and S4 as section 5 of the spec says.
    :param seed: the permutations' seed.
    :param arrest_genes: S5's knockouts with a known mitotic arrest; None skips S5.
    :param control_prefix: the non-targeting gene prefix.
    :param permutations: permutations of S2, S3 and S5.
    :param labelled: ``{"truth", "predicted", "shuffled_macro"}`` for S3; None skips it.
    :returns: one row per check: ``check``, ``stages`` (judged), ``value``, ``control``,
        ``threshold``, ``passed`` (True, False or None when skipped) and ``note``; the frame's
        ``attrs["samples"]`` keeps S2's observed share and its permutation shares for the
        report.
    """
    rng = np.random.default_rng(seed)
    stage = stages["model_stage"].to_numpy()
    reference = dict(card.get("reference") or {})
    rows = []
    samples: Dict[str, Any] = {}
    centred = dna["dna_content_centred"].to_numpy(float)
    log2_dna = np.log2(np.clip(centred, 1e-12, None))
    midpoint = np.full(len(cells), np.nan)
    for well, info in fits.items():
        fit = info["fit"]
        midpoint[(cells["well"].astype(str) == str(well)).to_numpy()] = (fit.g1 + fit.g2) / 2.0
    four_n = centred > midpoint
    condensed = np.isin(stage, ("prometaphase", "metaphase"))
    s1 = float(four_n[condensed].mean()) if condensed.any() else float("nan")
    s1_threshold = max(0.70, float(reference.get("S1", 0.70)) - 0.15)
    rows.append({"check": "S1", "stages": "prometaphase,metaphase", "value": s1,
                 "control": float(reference["S1"]) if "S1" in reference else float("nan"),
                 "threshold": s1_threshold,
                 "passed": None if not condensed.any() else bool(s1 >= s1_threshold),
                 "note": "share of prometaphase and metaphase calls above the G1/G2 midpoint"})
    late = np.isin(stage, DAUGHTER_STAGES)
    needed = {"nucleus_centroid_y", "nucleus_centroid_x", "nucleus_area"}
    if late.sum() >= 2 and needed <= set(cells.columns):
        two_n = ~four_n
        observed = _pair_share(cells, late, two_n, log2_dna)
        fields = (cells["well"].astype(str) + "/" + cells["site"].astype(str)).to_numpy()
        null = []
        for _ in range(int(permutations)):
            shuffled = late.copy()
            for field in np.unique(fields):
                idx = np.flatnonzero(fields == field)
                shuffled[idx] = late[rng.permutation(idx)]
            null.append(_pair_share(cells, shuffled, two_n, log2_dna))
        control = float(np.nanpercentile(null, 97.5))
        samples["S2"] = (observed, [float(v) for v in null])
        rows.append({"check": "S2", "stages": ",".join(DAUGHTER_STAGES), "value": observed,
                     "control": control, "threshold": control,
                     "passed": bool(observed > control), "note": "2N daughter pairs"})
    else:
        rows.append({"check": "S2", "stages": ",".join(DAUGHTER_STAGES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": "skipped: needs two daughter-stage calls and nucleus centroids"})
    if labelled is not None:
        value = _macro_f1(labelled["truth"], labelled["predicted"])
        control = float(np.percentile(labelled["shuffled_macro"], 97.5))
        rows.append({"check": "S3", "stages": ",".join(CLASSES), "value": value,
                     "control": control, "threshold": control, "passed": bool(value > control),
                     "note": "held-out macro F1 against shuffled-label models"})
    else:
        rows.append({"check": "S3", "stages": ",".join(CLASSES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": "skipped: no user labels"})
    mitotic = np.isin(stage, MITOTIC_STAGES)
    fields = (cells["well"].astype(str) + "/" + cells["site"].astype(str)).to_numpy()
    sizes = pd.Series(fields).value_counts()
    big = sizes[sizes >= 50].index
    if len(big) >= 2:
        share = np.array([mitotic[fields == f].mean() for f in big])
        n = sizes[big].to_numpy(float)
        p = float(mitotic[np.isin(fields, big)].mean())
        binomial = float(np.mean(p * (1 - p) / n)) if 0 < p < 1 else float("nan")
        phi = float(np.var(share) / binomial) if binomial and np.isfinite(binomial) else float("nan")
        limit = 2.0 * float(reference["S4"]) if "S4" in reference else float("nan")
        rows.append({"check": "S4", "stages": ",".join(MITOTIC_STAGES), "value": phi,
                     "control": float(reference.get("S4", float("nan"))), "threshold": limit,
                     "passed": (bool(phi <= limit) if np.isfinite(limit) and np.isfinite(phi)
                                else None),
                     "note": "overdispersion of the mitotic share across fields"
                             + ("" if np.isfinite(limit) else "; no base-card φ, not judged")})
    else:
        rows.append({"check": "S4", "stages": ",".join(MITOTIC_STAGES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": "skipped: fewer than two fields with 50 cells"})
    if arrest_genes and "gene" in cells.columns:
        genes = cells["gene"].astype(str).to_numpy()
        arrest = np.isin(np.char.upper(genes.astype(str)), [g.upper() for g in arrest_genes])
        control = np.char.startswith(np.char.lower(genes.astype(str)), control_prefix.lower())
        if arrest.sum() >= 100 and control.any():
            observed = float(mitotic[arrest].mean() - mitotic[control].mean())
            upper = {g.upper() for g in arrest_genes}
            others = [g for g in np.unique(genes) if g.upper() not in upper
                      and not g.lower().startswith(control_prefix.lower()) and g != "nan"]
            sizes_of = pd.Series(genes).value_counts()
            target = int(arrest.sum())
            null = []
            for _ in range(int(permutations)):
                order = rng.permutation(len(others))
                taken = np.cumsum(sizes_of.reindex([others[i] for i in order]).fillna(0)
                                  .to_numpy()) < target
                drawn = [others[i] for i in order[:int(taken.sum()) + 1]]
                rows_ = np.isin(genes, drawn)
                null.append(float(mitotic[rows_].mean() - mitotic[control].mean())
                            if rows_.any() else 0.0)
            p_value = (1 + np.sum(np.array(null) >= observed)) / (1 + len(null))
            rows.append({"check": "S5", "stages": "prometaphase,metaphase", "value": observed,
                         "control": float(np.percentile(null, 95)), "threshold": 0.05,
                         "passed": bool(p_value < 0.05),
                         "note": f"arrest knockouts vs non-targeting, permutation p = {p_value:.4f}"})
        else:
            rows.append({"check": "S5", "stages": "prometaphase,metaphase", "value": float("nan"),
                         "control": float("nan"), "threshold": float("nan"), "passed": None,
                         "note": "skipped: fewer than 100 cells of the arrest set"})
    else:
        rows.append({"check": "S5", "stages": "prometaphase,metaphase", "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": "skipped: no arrest set or no gene per cell"})
    quality = []
    for well, info in fits.items():
        here = (cells["well"].astype(str) == str(well)).to_numpy()
        quality.append(_fit_quality(centred[here], info["fit"]))
    r2 = min((q[0] for q in quality if np.isfinite(q[0])), default=float("nan"))
    ratios = [q[1] for q in quality]
    ratio_ok = all(G2_G1_RANGE[0] <= r <= G2_G1_RANGE[1] for r in ratios)
    rows.append({"check": "S6", "stages": "G1,S,G2", "value": r2,
                 "control": float(np.median(ratios)) if ratios else float("nan"),
                 "threshold": FIT_R2, "passed": bool(np.isfinite(r2) and r2 >= FIT_R2 and ratio_ok),
                 "note": f"lowest fit R2; G2/G1 {min(ratios, default=float('nan')):.3f}-"
                         f"{max(ratios, default=float('nan')):.3f} (pass {G2_G1_RANGE[0]}-"
                         f"{G2_G1_RANGE[1]})" + ("" if ratio_ok else "; ratio out of range")})
    frame = pd.DataFrame(rows)
    frame.attrs["samples"] = samples
    return frame


def _verdict(value: Any) -> Optional[bool]:
    """A check's ``passed`` as True, False or None (skipped), whatever its stored type."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


def failed_stages(checks: pd.DataFrame) -> List[str]:
    """The stages to collapse from one plate's checks (the spec's section 5 rule).

    S1 failing collapses prometaphase and metaphase, S2 ana_telo_cyto and
    telophase_daughters together (it judges their 2N pairs jointly), S3 (which judges every
    stage) each model stage one level; S1 and S2 both
    failing with S4 passing collapse to ``mitotic`` only, and S4 failing too collapses
    ``mitotic`` to ``G2/M``. S5 failing collapses prometaphase and metaphase. S6 failing on
    its R² collapses G1, S and G2; on its G2/G1 ratio only, S and G2 become ``S/G2``
    (written as the failed stage ``'S/G2'``). A skipped check collapses nothing.

    :param checks: from :func:`self_checks`.
    :returns: the failed stages, sorted.
    """
    state = {row.check: _verdict(row.passed) for row in checks.itertuples()}
    out = set()
    if state.get("S1") is False or state.get("S5") is False:
        out |= {"prometaphase", "metaphase"}
    if state.get("S2") is False:
        out |= set(DAUGHTER_STAGES)
    if state.get("S3") is False:
        out |= {*MITOTIC_STAGES, "g2_prophase"}
    if state.get("S4") is False and (state.get("S1") is False or state.get("S2") is False):
        out.add("mitotic")
    s6 = checks[checks["check"] == "S6"]
    if len(s6) and _verdict(s6["passed"].iloc[0]) is False:
        if np.isfinite(s6["value"].iloc[0]) and s6["value"].iloc[0] >= FIT_R2:
            out.add("S/G2")
        else:
            out |= {"G1", "S", "G2"}
    return sorted(out)


def final_stages(stages: pd.DataFrame, dna: pd.DataFrame, failed: Sequence[str]
                 ) -> pd.DataFrame:
    """The stage, probability and flag of every cell after the collapse rule.

    An interphase call takes G1, S or G2 from the DNA fit, with probability
    ``p_interphase`` times the phase posterior; a collapsed stage's probability is the sum over
    its children. ``'S/G2'`` in ``failed`` merges S and G2.

    :param stages: from :func:`predict_stages`.
    :param dna: the frame of :func:`_dna_phases`.
    :param failed: from :func:`failed_stages` (or the screen's union).
    :returns: ``cell_cycle_stage``, ``cell_cycle_probability``, ``cell_cycle_flag`` and
        ``dna_phase_posterior_G1|S|G2``.
    """
    failed = set(failed)
    p = {c: stages[f"p_{c}"].to_numpy(float) for c in CLASSES}
    post = {ph: dna[f"posterior_{ph}"].to_numpy(float) for ph in PHASES}
    phase = dna["phase"].astype(str).to_numpy()
    model = stages["model_stage"].to_numpy()
    stage = np.where(model == "interphase", np.where(np.isin(phase, PHASES), phase, "interphase"),
                     model).astype(object)
    probability = np.array([p[s][i] if s in p else p["interphase"][i] * post[s][i]
                            for i, s in enumerate(stage)])
    final, prob, flag = stage.copy(), probability.copy(), np.full(len(stage), "auto", object)
    p_mitotic = sum(p[s] for s in MITOTIC_STAGES)
    for i, s in enumerate(stage):
        if s in ("S", "G2") and "S/G2" in failed and "G1" not in failed:
            final[i], prob[i], flag[i] = "S/G2", p["interphase"][i] * (post["S"][i] + post["G2"][i]), "unresolved"
            continue
        up = collapse_stage(s, failed)
        if up == s:
            continue
        flag[i] = "unresolved"
        final[i] = up
        if up == "mitotic":
            prob[i] = p_mitotic[i]
        elif up == "G2/M":
            prob[i] = p_mitotic[i] + p["g2_prophase"][i] + p["interphase"][i] * post["G2"][i]
        elif up == "G2":
            prob[i] = p["g2_prophase"][i] + p["interphase"][i] * post["G2"][i]
        elif up == "interphase":
            prob[i] = p["interphase"][i]
    out = pd.DataFrame({"cell_cycle_stage": final, "cell_cycle_probability": prob,
                        "cell_cycle_flag": flag}, index=stages.index)
    for ph in PHASES:
        out[f"dna_phase_posterior_{ph}"] = post[ph]
    return out


def verified_stages(truth: np.ndarray, predicted: np.ndarray, groups: np.ndarray,
                    shuffled: Mapping[str, np.ndarray], seed: int = 0) -> List[str]:
    """The stages the user's labels verify: out-of-fold F1 ≥ 0.7 with the lower 95 % bound above
    the shuffled control's 97.5th percentile.

    :param truth: label indices of the labelled cells.
    :param predicted: their out-of-fold predictions (never the fold's own training labels).
    :param groups: their fields.
    :param shuffled: per-class F1 of the shuffled-label models.
    :param seed: the bootstrap seed.
    :returns: the verified stages, sorted.
    """
    f1 = _per_class_f1(truth, predicted)
    ci = _bootstrap_f1(truth, predicted, groups, seed)
    out = []
    for name, value in f1.items():
        control = float(np.percentile(shuffled[name], 97.5)) if name in shuffled else float("inf")
        if value >= VERIFIED_F1 and ci.get(name, (0.0, 0.0))[0] > control:
            out.append(name)
    return sorted(out)


def read_cell_cycle_labels(path: str) -> pd.DataFrame:
    """The user's labels: a CSV with ``plate, well, site, label, stage`` (and user, time).

    :param path: the labels file.
    :returns: the last label of every cell.
    :raises ValueError: when a column is missing or a stage unknown.
    """
    from .tabular import read_table

    table = read_table(str(path), canonicalise=False, report=None)
    missing = [c for c in (*KEY, "stage") if c not in table.columns]
    if missing:
        raise ValueError(f"the cell-cycle labels {path} lack {missing}")
    _encode(table["stage"])
    table["plate"] = table["plate"].astype(str)
    table["well"] = table["well"].astype(str)
    return table.drop_duplicates(KEY, keep="last").reset_index(drop=True)


def _options(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``ops_cell_cycle_*`` settings, checked."""
    mode = settings.get("ops_cell_cycle")
    family = str(settings.get("ops_cell_cycle_model_family") or "tabular").lower()
    tabular = str(settings.get("ops_cell_cycle_tabular") or "hgb").lower()
    finetune = str(settings.get("ops_cell_cycle_finetune") or "screen").lower()
    if mode not in ("DNA", "DNA+MT"):
        raise ValueError(f"ops_cell_cycle must be None, 'DNA' or 'DNA+MT'; got {mode!r}")
    if family not in FAMILIES:
        raise ValueError(f"ops_cell_cycle_model_family must be one of {list(FAMILIES)}")
    if tabular not in TABULAR:
        raise ValueError(f"ops_cell_cycle_tabular must be one of {list(TABULAR)}")
    if finetune not in FINETUNE:
        raise ValueError(f"ops_cell_cycle_finetune must be one of {list(FINETUNE)}")
    mt = settings.get("ops_cell_cycle_mt_channel") or None
    if mode == "DNA+MT" and not mt:
        raise ValueError("ops_cell_cycle 'DNA+MT' needs ops_cell_cycle_mt_channel")
    arrest = settings.get("ops_cell_cycle_arrest_genes", "default")
    if isinstance(arrest, str):
        arrest = ARREST_GENES if arrest.strip().lower() in ("default", "") else [
            g.strip() for g in arrest.split(",") if g.strip()]
    return {"mode": mode, "family": family, "tabular": tabular, "finetune": finetune,
            "dna": settings.get("ops_cell_cycle_dna_channel") or settings.get("nucleus_channel"),
            "mt": mt, "arrest": arrest, "seed": int(settings.get("ops_cell_cycle_seed") or 0),
            "base": str(settings.get("ops_cell_cycle_base_model") or ""),
            "labels": str(settings.get("ops_cell_cycle_labels") or "")}


def _html_table(frame: pd.DataFrame) -> str:
    """An HTML table of a frame, four significant digits."""
    return frame.to_html(index=False, float_format=lambda v: f"{v:.4g}", na_rep="")


def _png_data(figure) -> str:
    """A figure as an inline ``data:`` URI, without a timestamp, so reruns write the same bytes."""
    import base64
    import io

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                   metadata={"Software": None})
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _dna_figure(cells: pd.DataFrame, dna: pd.DataFrame, fits: Mapping[str, Any]) -> str:
    """Each well's DNA-content histogram with its fitted G1, S and G2 and the gates."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.style import DATA_ALPHA, SERIES, figure_style

    wells = sorted(fits)
    with figure_style("print"):
        figure = Figure(figsize=(2.2 * max(len(wells), 1), 2.0))
        axes = np.atleast_1d(figure.subplots(1, max(len(wells), 1), sharey=True))
        for ax, well in zip(axes, wells):
            fit = fits[well]["fit"]
            here = (cells["well"].astype(str) == str(well)).to_numpy()
            content = dna["dna_content_centred"].to_numpy(float)[here]
            content = content[np.isfinite(content) & (content > 0)]
            if not len(content):
                continue
            edges = np.linspace(*np.percentile(content, [0.5, 99.5]), 81)
            ax.hist(content, edges, color=SERIES[0], alpha=DATA_ALPHA, label="cells")
            centres = (edges[:-1] + edges[1:]) / 2
            density = fit.densities(centres) * len(content) * np.diff(edges)[0]
            for k, (name, colour) in enumerate(zip(("G1", "S", "G2"), SERIES[1:4])):
                ax.plot(centres, density[:, k], color=colour, lw=1.0, label=name)
            for gate in fit.gates:
                ax.axvline(gate, color="#231F20", lw=0.6, ls=(0, (2, 2)))
            ax.set_title(f"well {well}  G2/G1 {fit.g2 / fit.g1:.2f}", fontsize=8)
            ax.set_xlabel("DNA content (field-centred)", fontsize=8)
        axes[0].set_ylabel("cells", fontsize=8)
        axes[0].legend(fontsize=7, frameon=False)
        return _png_data(figure)


def _checks_figure(cells: pd.DataFrame, stages: pd.DataFrame, dna: pd.DataFrame,
                   fits: Mapping[str, Any], checks: pd.DataFrame) -> str:
    """S1 and S2 as bars with points, each against its control."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.style import condition_plot, figure_style

    centred = dna["dna_content_centred"].to_numpy(float)
    midpoint = np.array([(fits[str(w)]["fit"].g1 + fits[str(w)]["fit"].g2) / 2
                         if str(w) in fits else np.nan for w in cells["well"]])
    above = centred > midpoint
    stage = stages["model_stage"].to_numpy()
    fields = (cells["well"].astype(str) + "/" + cells["site"].astype(str)).to_numpy()
    condensed = np.isin(stage, ("prometaphase", "metaphase"))
    mitotic_share = [float(above[(fields == f) & condensed].mean()) for f in np.unique(fields)
                     if ((fields == f) & condensed).any()]
    all_share = [float(above[fields == f].mean()) for f in np.unique(fields)]
    samples = checks.attrs.get("samples", {})
    with figure_style("print"):
        figure = Figure(figsize=(4.2, 2.2))
        left, right = figure.subplots(1, 2)
        condition_plot(left, [all_share, mitotic_share], control=0)
        left.set_xticks([0, 1])
        left.set_xticklabels(["all cells\n(control)", "prometaphase\n+ metaphase"], fontsize=7)
        left.set_ylabel("share above the G1/G2 midpoint\n(per field)", fontsize=8)
        left.set_ylim(0, 1.05)
        if "S2" in samples:
            observed, null = samples["S2"]
            condition_plot(right, [null, [observed]], control=0)
            right.set_xticks([0, 1])
            right.set_xticklabels(["shuffled\n(control)", "daughter\nstages"], fontsize=7)
            right.set_ylim(0, 1.05)
            right.set_ylabel("2N calls with a 2N partner", fontsize=8)
        else:
            right.set_axis_off()
        figure.tight_layout()
        return _png_data(figure)


def _report_html(path: str, plate: str, checks: pd.DataFrame, counts: pd.Series,
                 failed: Sequence[str], notes: Sequence[str], card: Mapping[str, Any],
                 figures: Sequence[Tuple[str, str]] = ()) -> str:
    """``ops_cell_cycle_report.html``: checks with their controls, stage counts and notes."""
    import html

    table = _html_table
    f1 = pd.DataFrame([{"stage": k, "F1": v, "CI low": card.get("cv", {}).get("f1_ci95", {})
                        .get(k, [np.nan, np.nan])[0], "CI high": card.get("cv", {})
                        .get("f1_ci95", {}).get(k, [np.nan, np.nan])[1]}
                       for k, v in card.get("cv", {}).get("f1", {}).items()])
    body = [f"<h1>Cell cycle (OPS), plate {html.escape(str(plate))}</h1>",
            "<p>Stages: G1, S, G2 (DNA-content fit); g2_prophase (G2 prophase, a flag class); "
            "prometaphase; metaphase; ana_telo_cyto (anaphase, telophase and cytokinesis); "
            "artefact. CI: 95 % field-bootstrap confidence interval. NT: non-targeting.</p>",
            "<h2>Self-checks</h2>", table(checks),
            f"<p>Collapsed: {html.escape(', '.join(failed) or 'none')}</p>",
            "<h2>Cells per stage</h2>", table(counts.rename_axis("stage").reset_index(name="cells")),
            "<h2>Base model cross-validation F1</h2>", table(f1) if len(f1) else "<p>none</p>",
            "<p>Shuffled-label control, 97.5th percentile of macro F1: "
            f"{card.get('cv', {}).get('shuffled_macro_f1_p975')}</p>",
            *[f"<h2>{html.escape(title)}</h2><img src='{uri}' alt='{html.escape(title)}'>"
              for title, uri in figures],
            "<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html><meta charset='utf-8'><title>Cell cycle</title>"
                     + "".join(body))
    return path


def run_cell_cycle_phase(db: str, plate: str, wells: Sequence[str],
                         settings: Mapping[str, Any], destination: str) -> Dict[str, Any]:
    """The ``ph_cell_cycle`` phase: stages, probabilities and flags of every cell of a plate.

    Reads the per-well phenotype cell tables, calls the model (fine-tuned on the user's
    labels when given), takes G1 / S / G2 from :func:`_dna_phases`, runs
    :func:`self_checks`, collapses failed stages over every plate of the screen in ``db`` and
    writes ``ops_cell_cycle`` (one row per cell), ``ops_cell_cycle_checks`` (one row per plate
    and check) and ``ops_cell_cycle_report_<plate>.html``. With ``ops_cell_cycle_label_queue``
    above 0 it also draws the labelling queue (:func:`build_label_queue`) into
    ``cell_cycle_labels/<plate>/queue_pending.csv``.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: its wells.
    :param settings: read for every ``ops_cell_cycle_*`` setting, ``nucleus_channel``,
        ``cell_cycle_field_min_cells``, ``ops_design`` and ``ops_control_prefix``.
    :param destination: the run folder holding the cell tables.
    :returns: the phase report; ``label_queue`` names the queue folder, its size and the
        labels file, or is None.
    :raises ValueError: for a bad setting, missing cell tables or no model to start from.
    """
    from .ops_engine import _ph_cells_path, _replace_well_rows, _rows_for_wells
    from .tabular import read_table

    options = _options(settings)
    notes: List[str] = []
    if options["family"] != "tabular":
        notes.append(f"ops_cell_cycle_model_family {options['family']!r} is not available in "
                     f"this build; the tabular learner {options['tabular']!r} was used")
    if not options["base"] and not options["labels"]:
        raise ValueError("ops_cell_cycle needs a base model (ops_cell_cycle_base_model) or user "
                         "labels (ops_cell_cycle_labels); no shipped base model is registered yet")
    base = load_cell_cycle_checkpoint(options["base"]) if options["base"] else None
    frames = []
    for well in wells:
        path = _ph_cells_path(destination, well)
        if os.path.isfile(path):
            frame = read_table(path, canonicalise=False, report=None)
            frame["well"] = str(well)
            frames.append(frame)
    if not frames:
        raise ValueError(f"no phenotype cell table for wells {list(wells)}")
    cells = pd.concat(frames, ignore_index=True)
    cells["plate"] = str(plate)
    dna = str(options["dna"] or "dapi").lower()
    integrated = f"nucleus_{dna}_integrated_intensity"
    if integrated not in cells.columns:
        raise ValueError(f"the cell tables lack {integrated}, the DNA content the fit needs")
    cells["_dna_integrated"] = cells[integrated].to_numpy(float)
    _attach_genes(cells, db, plate, wells, settings, _rows_for_wells)
    labelled = None
    model = base
    if options["labels"]:
        labels = read_cell_cycle_labels(options["labels"])
        joined = cells.reset_index().merge(labels, on=KEY, how="inner")
        if len(joined) and options["finetune"] != "none":
            if base is not None:
                model = fine_tune(base, joined, joined["stage"], seed=options["seed"],
                                  tabular=options["tabular"])
            else:
                model = train_cell_cycle_model(joined, joined["stage"], input_mode=options["mode"],
                                               dna=dna, mt=options["mt"],
                                               tabular=options["tabular"], seed=options["seed"],
                                               repeats=1, permutations=0)
            labelled = _labelled_scores(model, joined, options)
    stages = predict_stages(model, cells)
    excluded = stages["model_stage"].to_numpy() != "interphase"
    dna_frame, fits = _dna_phases(cells, settings, excluded)
    fits = {str(k): v for k, v in fits.items()}
    checks = self_checks(cells, stages, dna_frame, fits, card=model["card"], seed=options["seed"],
                         arrest_genes=options["arrest"],
                         control_prefix=str(settings.get("ops_control_prefix") or "nontargeting"),
                         permutations=PERMUTATIONS, labelled=labelled)
    failed = failed_stages(checks)
    checks.insert(0, "plate", str(plate))
    checks["collapsed"] = ",".join(failed)
    _replace_plate_rows(db, "ops_cell_cycle_checks", checks, plate)
    per_plate = _collapses_in(db)
    per_plate[str(plate)] = failed
    union = screen_collapses(per_plate)
    final = final_stages(stages, dna_frame, union)
    if labelled is not None:
        verified = verified_stages(labelled["truth"], labelled["predicted"], labelled["groups"],
                                   labelled["shuffled"], options["seed"])
        final.loc[final["cell_cycle_stage"].isin(verified)
                  & (final["cell_cycle_flag"] == "auto"), "cell_cycle_flag"] = "verified"
    model_id = str(model.get("sha256") or model["card"].get("model_sha256") or "")[:12]
    table = pd.concat([cells[KEY].reset_index(drop=True), final.reset_index(drop=True),
                       stages.reset_index(drop=True)], axis=1)
    table["dna_log2_fc"] = np.log2(np.clip(dna_frame["dna_content_centred"].to_numpy(float),
                                           1e-12, None)) - np.log2(np.array(
        [fits[str(w)]["fit"].g1 if str(w) in fits else np.nan for w in cells["well"]]))
    table["model_id"] = model_id
    for well in wells:
        _replace_well_rows(db, "ops_cell_cycle", table[table["well"] == str(well)], plate, well)
    queued = None
    size = int(settings.get("ops_cell_cycle_label_queue") or 0)
    if size > 0:
        from .tabular import write_table

        folder = os.path.join(destination, "cell_cycle_labels", str(plate))
        os.makedirs(folder, exist_ok=True)
        pending = build_label_queue(table.assign(model_stage=stages["model_stage"].to_numpy()),
                                    size, options["seed"])
        write_table(pending, os.path.join(folder, "queue_pending.csv"), canonicalise=False)
        from .ops_engine import _ph_merged_root

        queued = {"folder": folder, "cells": int(len(pending)),
                  "pending": os.path.join(folder, "queue_pending.csv"),
                  "crop_root": _ph_merged_root(settings, destination),
                  "labels": options["labels"] or os.path.join(destination, "cell_cycle_labels",
                                                              "labels.csv")}
    counts = table["cell_cycle_stage"].value_counts().sort_index()
    figures = [("DNA content per well, with the fitted G1, S and G2 and the gates",
                _dna_figure(cells, dna_frame, fits)),
               ("4N in mitosis (S1) and 2N daughter pairs (S2), each against its control",
                _checks_figure(cells, stages, dna_frame, fits, checks))]
    report = _report_html(os.path.join(destination, f"ops_cell_cycle_report_{plate}.html"),
                          plate, checks, counts, union, notes, model["card"], figures)
    return {"cells": int(len(table)), "collapsed": union, "plate_collapsed": failed,
            "stages": {k: int(v) for k, v in counts.items()}, "notes": notes,
            "report": report, "model_id": model_id, "label_queue": queued}


def _labelled_scores(model: Mapping[str, Any], joined: pd.DataFrame,
                     options: Mapping[str, Any]) -> Dict[str, Any]:
    """Out-of-fold predictions and the shuffled control on the screen's labelled cells (S3)."""
    features = list(model["features"])
    x = _matrix(joined, features)
    y = _encode(joined["stage"])
    groups = (joined["well"].astype(str) + "/" + joined["site"].astype(str)).to_numpy()
    training = model.get("training")
    if training is not None and len(training) > len(y):
        x_base = training[features].to_numpy()[:len(training) - len(y)]
        y_base = _encode(training["stage"])[:len(training) - len(y)]
    else:
        x_base, y_base = np.empty((0, len(features))), np.empty(0, int)
    x_all = np.vstack([x_base, x])
    y_all = np.concatenate([y_base, y])
    g_all = np.concatenate([np.array(["base"] * len(y_base), object), groups])
    w_all = np.concatenate([np.ones(len(y_base)), np.full(len(y), SCREEN_WEIGHT)])
    scored = np.concatenate([np.zeros(len(y_base), bool), np.ones(len(y), bool)])
    predicted = _out_of_fold(options["tabular"], options["seed"], x_all, y_all, g_all, w_all,
                             scored)[scored]
    macro, per_class = _shuffled_f1(options["tabular"], options["seed"], x_all, y_all, g_all,
                                    w_all, scored, PERMUTATIONS)
    return {"truth": y, "predicted": predicted, "groups": groups, "shuffled_macro": macro,
            "shuffled": per_class}


def _attach_genes(cells: pd.DataFrame, db: str, plate: str, wells: Sequence[str],
                  settings: Mapping[str, Any], rows_for_wells) -> None:
    """Put each cell's ``gene`` on ``cells`` from the merge, the barcodes and the design (S5).

    Cells without a merge, a call or a design row get no gene; without any, S5 is skipped.
    """
    from .ops_engine import _ph_design

    design_path = settings.get("ops_design")
    merged = rows_for_wells(db, "ops_merge", plate, list(wells)) if os.path.isfile(db) else None
    barcodes = rows_for_wells(db, "ops_barcodes", plate, list(wells)) if os.path.isfile(db) else None
    if not design_path or merged is None or barcodes is None or not len(barcodes):
        return
    called = barcodes.loc[barcodes["mapped_guide"].fillna("").astype(str) != "",
                          ["well", "object_id", "mapped_guide"]]
    if not len(called):
        return
    design, _ambiguous = _ph_design(str(design_path),
                                    int(called["mapped_guide"].str.len().mode().iloc[0]))
    link = (merged[["well", "site", "label", "object_id"]]
            .merge(called, on=["well", "object_id"], how="inner")
            .merge(design.rename(columns={"prefix": "mapped_guide"}), on="mapped_guide",
                   how="inner")[["well", "site", "label", "gene"]]
            .drop_duplicates(["well", "site", "label"]))
    link["well"] = link["well"].astype(str)
    genes = cells[["well", "site", "label"]].merge(link, on=["well", "site", "label"], how="left")
    cells["gene"] = genes["gene"].to_numpy()


def _replace_plate_rows(db: str, table: str, frame: pd.DataFrame, plate: str) -> None:
    """Replace one plate's rows of a per-plate table, through :mod:`spacr.ops_store`."""
    from .ops_store import read_table, row_count, write_table

    frame = frame.assign(passed=[None if v is None else int(bool(v)) for v in frame["passed"]])
    parts = []
    if row_count(db, table) is not None:
        existing = read_table(db, table)
        parts.append(existing[existing["plate"].astype(str) != str(plate)])
    parts.append(frame)
    write_table(db, table, pd.concat(parts, ignore_index=True), if_exists="replace")


def _collapses_in(db: str) -> Dict[str, List[str]]:
    """The failed stages of every plate already in ``ops_cell_cycle_checks``."""
    from .ops_store import read_table, row_count

    if not os.path.isfile(db) or row_count(db, "ops_cell_cycle_checks") is None:
        return {}
    rows = read_table(db, "ops_cell_cycle_checks")[["plate", "collapsed"]].drop_duplicates()
    return {str(p): [s for s in str(c or "").split(",") if s and c == c]
            for p, c in zip(rows["plate"], rows["collapsed"])}


def build_label_queue(table: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """The cells to label: the most uncertain, every stage and every plate represented.

    The margin between a cell's two most probable classes ranks it
    (:func:`spacr.active_learning.rank_by_uncertainty`, ties broken with ``seed``). Every model
    stage first gets its ⌈n/10⌉ most uncertain cells, then every plate its most uncertain
    cell, then the rest fill up to ``n`` in rank order.

    :param table: ``ops_cell_cycle`` rows (``plate, well, site, label``, ``model_stage`` and
        ``p_<class>``).
    :param n: the queue size.
    :param seed: the tie-break seed.
    :returns: ``n`` rows (fewer when the table is smaller) with the keys, ``model_stage`` and
        ``rank``, in rank order.
    """
    from .active_learning import rank_by_uncertainty

    probabilities = table[[f"p_{c}" for c in CLASSES]].to_numpy(float)
    order = np.asarray(rank_by_uncertainty(probabilities, "margin", seed=int(seed)), int)
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    ranked = table.assign(rank=rank).sort_values("rank", kind="stable")
    chosen: List[int] = []
    per_stage = int(np.ceil(n / 10.0))
    for stage in CLASSES:
        chosen += list(ranked.index[ranked["model_stage"] == stage][:per_stage])
    for _plate, rows in ranked.groupby("plate", sort=True):
        if not set(rows.index) & set(chosen):
            chosen.append(rows.index[0])
    for index in ranked.index:
        if len(set(chosen)) >= n:
            break
        if index not in chosen:
            chosen.append(index)
    picked = ranked.loc[list(dict.fromkeys(chosen))].sort_values("rank", kind="stable")
    return picked.head(int(n))[[*KEY, "model_stage", "rank"]].reset_index(drop=True)


def write_label_queue(queue: pd.DataFrame, crop_root: str, folder: str, *,
                      labels: str) -> str:
    """The queue as a project Annotate opens: PNG crops and their ``png_list``.

    Each queued cell's crop is read from the crop store of the ``ph_crops`` phase
    (:func:`spacr.ops_run_infra.crop_index`) and written as an 8-bit PNG with one fixed display
    range per channel, the 0.5th to 99.5th percentile of that channel over the whole queue (no
    per-crop stretch). Annotation 1 to 7 are the stages of :data:`CLASSES`, in order.

    :param queue: from :func:`build_label_queue`.
    :param crop_root: the crop store's root (``ops_merged_root`` or ``dst_root``).
    :param folder: the project folder to make; ``measurements/measurements.db`` and
        ``data/`` go in it, with ``queue.json`` naming the classes and the labels file.
    :param labels: the labels CSV the job's answers are appended to.
    :returns: ``folder``.
    :raises ValueError: when a queued cell has no crop in the store.
    """
    from PIL import Image

    from .annotation_dataset import png_list_frame, write_png_list
    from .ops_run_infra import crop_index
    from .ops_engine import _ph_read_crop
    from .tabular import write_table

    crops = []
    for plate, rows in queue.groupby("plate", sort=False):
        index = crop_index(crop_root, str(plate), sorted(rows["well"].astype(str).unique()))
        if not len(index):
            raise ValueError(f"no crops of plate {plate} under {crop_root}; run ph_crops first")
        index = index.assign(well=[os.path.basename(os.path.dirname(s)) for s in index["shard"]])
        found = rows.astype({"well": str}).merge(index, on=["well", "site", "label"], how="left")
        if found["shard"].isna().any():
            raise ValueError(f"{int(found['shard'].isna().sum())} queued cells of plate {plate} "
                             "have no crop; run ph_crops for their wells")
        crops += [(row, _ph_read_crop(row.shard, row.offset_data, row.size)[0])
                  for row in found.itertuples()]
    stacked = [np.asarray(image, float) for _row, image in crops]
    channels = stacked[0].shape[0]
    ranges = [np.percentile(np.concatenate([s[c].ravel() for s in stacked]), [0.5, 99.5])
              for c in range(channels)]
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(folder, "measurements"), exist_ok=True)
    paths = []
    for (row, _image), image in zip(crops, stacked):
        planes = [np.clip((image[c] - lo) / max(hi - lo, 1e-9), 0, 1) for c, (lo, hi)
                  in enumerate(ranges)]
        rgb = np.stack([planes[min(i, channels - 1)] if i < channels else np.zeros_like(planes[0])
                        for i in range(3)], axis=-1)
        path = os.path.join(data, f"{row.plate}_{row.well}_s{row.site}_l{row.label}.png")
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
        paths.append(path)
    selection = pd.DataFrame({"plateID": [r.plate for r, _ in crops],
                              "rowID": [r.well for r, _ in crops],
                              "columnID": [r.well for r, _ in crops],
                              "fieldID": [r.site for r, _ in crops],
                              "objectID": [r.label for r, _ in crops]})
    write_png_list(os.path.join(folder, "measurements", "measurements.db"),
                   png_list_frame(selection, paths))
    write_table(queue.assign(png_path=paths), os.path.join(folder, "queue.csv"),
                canonicalise=False)
    with open(os.path.join(folder, "queue.json"), "w", encoding="utf-8") as handle:
        json.dump({"classes": {str(i + 1): c for i, c in enumerate(CLASSES)},
                   "labels": str(labels), "cells": int(len(queue)),
                   "ranges": [list(map(float, r)) for r in ranges]}, handle, indent=1)
    return folder


def collect_label_queue(folder: str, *, user: str = "") -> int:
    """Append the job's answers to its labels file (append-only, with user and time).

    :param folder: from :func:`write_label_queue`.
    :param user: who labelled.
    :returns: the rows appended (cells annotated 1 to 6).
    """
    import csv
    import datetime

    from .tabular import read_database, read_table

    with open(os.path.join(folder, "queue.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = read_database(os.path.join(folder, "measurements", "measurements.db"), "png_list",
                         canonicalise=False, report=None)[0][["png_path", "annotate"]]
    queue = read_table(os.path.join(folder, "queue.csv"), canonicalise=False, report=None)
    answers = queue.merge(rows, on="png_path", how="inner")
    answers = answers[answers["annotate"].notna()]
    answers["stage"] = [manifest["classes"].get(str(int(v))) for v in answers["annotate"]]
    answers = answers[answers["stage"].notna()]
    if not len(answers):
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = answers[KEY + ["stage"]].assign(user=user, time=stamp)
    path = manifest["labels"]
    new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(list(out.columns))
        writer.writerows(out.itertuples(index=False, name=None))
    return int(len(out))


def track_label_job(folder: str, cells: int):
    """Register the labelling job on Home and in the Jobs window (a Qt application must run).

    The job is titled "Cell cycle: label N cells"; it does not block closing spaCR and nothing
    stops with it. :func:`finish_label_job` retires it.

    :param folder: from :func:`write_label_queue`.
    :param cells: the queue size, for the title.
    :returns: the registry handle; its ``folder`` attribute names the queue.
    """
    from .qt.bridge import _track_external_job

    handle = _track_external_job(f"Cell cycle: label {int(cells)} cells", None,
                                 blocks_shutdown=False, stop_on_quit=False, kind="external")
    handle.folder = str(folder)
    handle.report(0, int(cells), f"queue in {folder}")
    return handle


def finish_label_job(handle, *, rerun=None, user: str = "",
                     settings: Optional[Mapping[str, Any]] = None) -> int:
    """The job is done: append its answers, refit in a new run journal, and retire the job.

    The refit runs inside its own run journal (:func:`spacr.run_journal.open_run`, app
    ``ops_cell_cycle_labels``), whose settings carry ``ops_cell_cycle_labels`` and whose inputs
    record the labels file's full SHA-256, so the journal names the exact labels the verified
    state came from. The journal folder is kept as ``handle.journal``.

    :param handle: from :func:`track_label_job`.
    :param rerun: called with the labels file after the answers are appended, for example a
        rerun of :func:`run_cell_cycle_phase` with ``ops_cell_cycle_labels`` set to it, so a
        settings-file rerun reproduces the verified state; None skips the refit.
    :param user: who labelled.
    :param settings: the run's settings, written to the journal with the labels path.
    :returns: the answers appended.
    """
    from .run_journal import open_run

    try:
        appended = collect_label_queue(handle.folder, user=user)
        with open(os.path.join(handle.folder, "queue.json"), encoding="utf-8") as stream:
            labels = json.load(stream)["labels"]
        handle.report(line=f"{appended} labels appended to {labels}")
        if appended:
            journal = {**dict(settings or {}), "ops_cell_cycle_labels": labels,
                       "hash_inputs": True}
            with open_run("ops_cell_cycle_labels", journal) as run:
                run.record_input(labels, setting_key="ops_cell_cycle_labels")
                if rerun is not None:
                    rerun(labels)
                run.set_status("success")
                handle.journal = str(run.dir)
        return appended
    finally:
        handle.retire()


def open_label_job(window, queued: Mapping[str, Any]) -> None:
    """Open a run's labelling queue in Annotate, writing its crops the first time.

    :param window: the main window (``open_module`` and its built screens).
    :param queued: the ``label_queue`` of :func:`run_cell_cycle_phase`'s report.
    """
    from .tabular import read_table

    folder = str(queued["folder"])
    if not os.path.isfile(os.path.join(folder, "queue.json")):
        pending = read_table(str(queued["pending"]), canonicalise=False, report=None)
        write_label_queue(pending, str(queued["crop_root"]), folder, labels=str(queued["labels"]))
    key = window.open_module("annotate")
    window._screens[key]._open_source(folder)
