"""Cell-cycle stage, probability and verification flag of every OPS cell (item 990, alpha).

A tabular model (HistGradientBoosting or XGBoost on the DNA and microtubule features of
:func:`spacr.ops_mitotic.mitotic_features`, each a robust z against all cells of its well)
calls interphase, g2_prophase (a flag class, never counted as mitotic), prometaphase,
metaphase, ana_telo_cyto (anaphase and telophase/cytokinesis), telophase_daughters (the two
2N daughters, a stage of their own) and artefact. Interphase cells take G1, S or G2 from the field-centred DNA-content fit of
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
STAGE_BUNDLE_FORMAT = "spacr-cellcycle-stage-model-1"
NORMALISATION = "robust z per well (spacr.ops_mitotic._well_robust_z)"
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
    from .qt.i18n import tr

    features = mitotic_features(columns, dna=dna, tubulin=mt)
    if not features:
        raise ValueError(tr("the cell tables have no feature of the DNA channel {dna}",
                            dna=repr(dna)))
    return features


def _matrix(cells: pd.DataFrame, features: Sequence[str], *,
            reference_cells: Optional[pd.DataFrame] = None) -> np.ndarray:
    """Normalize rows against complete reference wells using the shared mitotic helpers.

    :param cells: labelled or prediction rows, in the required output order.
    :param features: ordered model feature columns.
    :param reference_cells: every cell of the reference wells; None treats cells as
        the complete population. Plates are separate when both tables identify them.
    :returns: finite float32 features, with unavailable robust z values set to zero.
    :raises ValueError: when the reference omits a requested plate or well.
    """
    from .ops_mitotic import (_apply_population_normalization,
                             _fit_population_normalization, _normalization_groups,
                             _well_robust_z)
    from .qt.i18n import tr

    population = cells if reference_cells is None else reference_cells
    if "well" not in cells:
        return np.nan_to_num(_well_robust_z(cells, features, None), nan=0.0)
    if "plate" in cells and "plate" in population:
        reference_groups = _normalization_groups(population, "plate")
        groups = _normalization_groups(cells, "plate")
    else:
        reference_groups = {None: np.arange(len(population))}
        groups = {None: np.arange(len(cells))}
    out = np.empty((len(cells), len(features)), np.float32)
    for plate, rows in groups.items():
        if plate not in reference_groups:
            raise ValueError(tr("reference_cells has no full population for plate {plate}",
                                plate=repr(plate)))
        all_rows = reference_groups[plate]
        full = population if len(all_rows) == len(population) else population.iloc[all_rows]
        query = cells if len(rows) == len(cells) else cells.iloc[rows]
        reference = _fit_population_normalization(full, features, "well")
        out[rows] = _apply_population_normalization(query, reference)
    return np.nan_to_num(out, nan=0.0)


def _training_reference(cells: pd.DataFrame, reference_cells: Optional[pd.DataFrame]
                        ) -> pd.DataFrame:
    """Resolve complete reference wells, warning about the legacy population assumption.

    :param cells: labelled training rows.
    :param reference_cells: complete wells or None for a complete labelled population.
    :returns: the reference cells used by the shared normalizer.
    """
    if reference_cells is None:
        import warnings
        from .qt.i18n import tr

        warnings.warn(tr(
            "reference_cells was not supplied; training rows are used as the full "
            "normalization population. For enriched label sets, pass all cells "
            "of the reference wells in reference_cells"), UserWarning, stacklevel=3)
        return cells
    return reference_cells


def _learner(tabular: str, seed: int):
    """An unfitted multiclass learner: :func:`spacr.ops_mitotic._model`."""
    from .ops_mitotic import _model
    from .qt.i18n import tr

    if tabular not in TABULAR:
        raise ValueError(tr("ops_cell_cycle_tabular must be one of {allowed}; got {value}",
                            allowed=list(TABULAR), value=repr(tabular)))
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


class _ClassMappedModel:
    """A fitted learner whose own class indices are mapped onto :data:`CLASSES`.

    :param model: a fitted classifier with ``classes_`` and ``predict_proba``.
    :param classes: the :data:`CLASSES` index of each class the model knows.
    """

    def __init__(self, model, classes: Sequence[int]):
        """Keep the model and the :data:`CLASSES` index of each of its classes."""
        self.model = model
        self.classes_ = np.asarray(classes, int)

    def predict_proba(self, x):
        """The wrapped model's probabilities, in the columns of ``classes_``."""
        return self.model.predict_proba(x)


def _encode(labels: Sequence[Any]) -> np.ndarray:
    """Class indices of stage names; an unknown name raises ValueError."""
    from .qt.i18n import tr

    index = {name: i for i, name in enumerate(CLASSES)}
    unknown = sorted({str(v) for v in labels} - set(index))
    if unknown:
        raise ValueError(tr("cell-cycle labels must be one of {allowed}; got {unknown}",
                            allowed=list(CLASSES), unknown=unknown))
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
                           data_files: Sequence[str] = (),
                           reference_cells: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """A base cell-cycle model and its card, from labelled cells (the one call that trains it).

    :param cells: the labelled cells: ``well``, the field column of ``groups``, the feature
        columns of the cell tables and, for the plate-held-out score of two or more plates,
        ``plate``.
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
    :param reference_cells: full wells including unlabelled cells. Required for enriched
        label samples; omission treats cells as the full population with a warning.
    :returns: ``{"model", "features", "card", "training"}``; ``training`` keeps the labelled
        rows (features and labels) that fine-tuning refits on. The card's ``cv`` holds the
        field-grouped repeats, per-class F1 with field-bootstrap intervals, the shuffled-label
        control and, with two or more plates, ``plate_held_out`` (each plate predicted by a
        model trained on the others).
    :raises ValueError: for an unknown mode or stage, or 'DNA+MT' without ``mt``.
    """
    from .qt.i18n import tr

    if input_mode not in ("DNA", "DNA+MT"):
        raise ValueError(tr("input_mode must be 'DNA' or 'DNA+MT'; got {value}",
                            value=repr(input_mode)))
    if input_mode == "DNA+MT" and not mt:
        raise ValueError(tr("'DNA+MT' needs the microtubule channel"))
    features = _features(cells.columns, dna, mt if input_mode == "DNA+MT" else None)
    x = _matrix(cells, features, reference_cells=_training_reference(cells, reference_cells))
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
    plates = cells["plate"].astype(str).to_numpy() if "plate" in cells.columns else None
    held_out = None
    if plates is not None and len(np.unique(plates)) >= 2:
        by_plate = _out_of_fold(tabular, seed, x, y, plates, n_splits=len(np.unique(plates)))
        held_out = {"macro_f1": _macro_f1(y, by_plate), "f1": _per_class_f1(y, by_plate)}
    card = {"format": CHECKPOINT_FORMAT, "family": "tabular", "tabular": tabular,
            "input": input_mode, "classes": list(CLASSES), "parents": PARENTS,
            "features": list(features), "dna": dna, "mt": mt,
            "normalisation": NORMALISATION,
            "data": [{"path": str(p), "sha256": _sha256(str(p))} for p in data_files],
            "label_counts": {CLASSES[i]: int(n) for i, n in
                             zip(*np.unique(y, return_counts=True))},
            "cv": {"scheme": f"field-grouped 5-fold x {int(repeats)}", "repeats": metrics,
                   "f1": _per_class_f1(y, predicted),
                   "f1_ci95": _bootstrap_f1(y, predicted, groups, seed),
                   "shuffled_macro_f1_p975": (float(np.percentile(shuffled, 97.5))
                                              if len(shuffled) else None),
                   "plate_held_out": held_out,
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


def _checkpoint_key(bundle: Mapping[str, Any]) -> str:
    """Hash canonical prediction probes and saved labelled rows instead of joblib bytes.

    :param bundle: model, ordered features, optional labelled training rows and card.
    :returns: deterministic SHA-256 identity for the model input and predictions.
    """
    digest = hashlib.sha256()
    features = list(bundle["features"])
    header = {"input": bundle["card"]["input"], "features": features,
              "classes": list(CLASSES)}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    probe = np.random.default_rng(0).standard_normal((256, len(features))).astype(np.float32)
    probe = np.vstack([probe, np.zeros((1, len(features)), np.float32),
                       np.full((1, len(features)), -6, np.float32),
                       np.full((1, len(features)), 6, np.float32)])
    probabilities = np.asarray(_probabilities(bundle["model"], probe), dtype="<f8")
    digest.update(probabilities.tobytes(order="C"))
    training = bundle.get("training")
    if training is not None:
        rows = np.asarray(training[features], dtype="<f8")
        digest.update(json.dumps(list(rows.shape)).encode())
        digest.update(rows.tobytes(order="C"))
        digest.update(json.dumps(training["stage"].astype(str).tolist(),
                                 ensure_ascii=False, separators=(",", ":")).encode())
    return digest.hexdigest()


def save_cell_cycle_checkpoint(bundle: Mapping[str, Any], folder: str, *,
                               base: Optional[str] = None) -> str:
    """Save every model and card revision under one stable checkpoint identity.

    The directory is ``cell_cycle_tabular_<input>_<identity12>``. Its model,
    card and SHA256SUMS are staged completely before publication. Each file
    is replaced atomically, with the card before the model and checksums last;
    an interrupted publication is refused by the loader rather than using a
    stale card with a new model.

    :param bundle: fitted model, features, optional training rows and mutable card.
        The canonical checkpoint_id hashes input features, classes, fixed normalized
        probe probabilities and saved labelled rows. It is retained across saves;
        the rest of the card is excluded so updated cards replace the same path.
    :param folder: directory containing checkpoints.
    :param base: base model SHA-256 recorded for fine-tuning.
    :returns: the stable checkpoint directory, also when model bytes change.
    """
    import shutil
    import tempfile
    import joblib
    from .ops_engine import _say
    from .qt.i18n import tr

    identity = str(bundle["card"].get("checkpoint_id") or _checkpoint_key(bundle))
    if isinstance(bundle["card"], dict):
        bundle["card"]["checkpoint_id"] = identity
    card = dict(bundle["card"], checkpoint_id=identity)
    name = identity[:12]
    target = os.path.join(folder, f"cell_cycle_tabular_{card['input'].replace('+', '_')}_{name}")
    os.makedirs(folder, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=".cell_cycle_", dir=folder)
    try:
        model_path = os.path.join(stage, "model.joblib")
        joblib.dump({"model": bundle["model"], "features": list(bundle["features"]),
                     "training": bundle.get("training")}, model_path)
        card.update(model_sha256=_sha256(model_path), base_sha256=base)
        with open(os.path.join(stage, "card.json"), "w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=1, sort_keys=True, default=str)
        sums = {name: _sha256(os.path.join(stage, name))
                for name in ("card.json", "model.joblib")}
        with open(os.path.join(stage, "SHA256SUMS"), "w", encoding="utf-8") as handle:
            handle.writelines(f"{value}  {name}\n" for name, value in sorted(sums.items()))
        os.makedirs(target, exist_ok=True)
        for filename in ("card.json", "model.joblib", "SHA256SUMS"):
            os.replace(os.path.join(stage, filename), os.path.join(target, filename))
    finally:
        shutil.rmtree(stage)
    _say(tr("Cell-cycle checkpoint saved with current card: {path}", path=target))
    return target


def stage_bundle(path: str) -> Dict[str, Any]:
    """A development stage-model bundle (a ``.joblib`` dict), mapped onto :data:`CLASSES`.

    The bundle holds ``format`` ``spacr-cellcycle-stage-model-1``, a ``model``, ``features``,
    its own ``classes`` and a ``normalisation`` text. Classes that :data:`CLASSES` has and the
    bundle lacks (``telophase_daughters`` in the six-class plate A model) get probability 0.
    The bundle's own class names are matched to :data:`CLASSES` by name and the mapping is
    written in the card. The normalisation must be the per-well robust z of
    :func:`spacr.ops_mitotic._well_robust_z`, which :func:`predict_stages` applies.

    Unpickling runs code, so the bundle is only as safe as its source; there is no checksum
    file to check, so the SHA-256 of the file is recorded in the card instead.

    :param path: the ``.joblib`` file.
    :returns: ``{"model", "features", "training", "card", "sha256"}``, the shape of
        :func:`load_cell_cycle_checkpoint`.
    :raises ValueError: for another format, a class :data:`CLASSES` does not know or another
        normalisation.
    """
    import warnings

    import joblib

    from .qt.i18n import tr

    warnings.warn(tr("loading the development stage model {path}: unpickling runs code; load "
                     "only bundles you trust", path=path), stacklevel=2)
    held = joblib.load(path)
    if not isinstance(held, Mapping) or not str(held.get("format", "")).startswith(
            STAGE_BUNDLE_FORMAT):
        raise ValueError(tr("{path} is not a {format} stage-model bundle", path=path,
                            format=STAGE_BUNDLE_FORMAT))
    classes = [str(c) for c in held["classes"]]
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        raise ValueError(tr("the bundle {path} has classes {unknown} that spaCR does not call",
                            path=path, unknown=unknown))
    normalisation = str(held.get("normalisation") or "")
    if not normalisation.startswith("robust z per well"):
        raise ValueError(tr("the bundle {path} is normalised as {found}; spaCR applies the "
                            "robust z per well", path=path, found=repr(normalisation)))
    own = np.asarray(held["model"].classes_, int)
    mapped = [CLASSES.index(classes[i]) for i in own]
    digest = _sha256(path)
    features = [str(f) for f in held["features"]]
    dna = next((f.split("_")[1] for f in features if f.endswith("_mean_intensity")
                and f.startswith(("cell_", "nucleus_"))), "dapi")
    card = {"format": CHECKPOINT_FORMAT, "family": "tabular", "tabular": "hgb",
            "input": "DNA", "classes": list(CLASSES), "parents": PARENTS,
            "features": features, "dna": dna, "mt": None, "normalisation": normalisation,
            "imported_from": {"path": str(path), "sha256": digest,
                              "format": str(held["format"]), "classes": classes,
                              "class_map": {classes[i]: CLASSES[m] for i, m in zip(own, mapped)},
                              "absent_classes": [c for c in CLASSES if c not in classes],
                              "cv": held.get("cv")},
            "seed": 0, "spacr_version": _version()}
    return {"model": _ClassMappedModel(held["model"], mapped), "features": features,
            "training": None, "card": card, "sha256": digest}


def import_stage_bundle(path: str, folder: str) -> str:
    """Convert a development stage-model bundle to a release checkpoint folder.

    :param path: the ``.joblib`` bundle read by :func:`stage_bundle`.
    :param folder: the directory that receives the checkpoint.
    :returns: the checkpoint directory of :func:`save_cell_cycle_checkpoint`; load it with
        :func:`load_cell_cycle_checkpoint`.
    """
    return save_cell_cycle_checkpoint(stage_bundle(path), folder)


def load_cell_cycle_checkpoint(path: str) -> Dict[str, Any]:
    """A checkpoint written by :func:`save_cell_cycle_checkpoint`, checked before unpickling.

    A ``.joblib`` file is a development stage-model bundle and goes through
    :func:`stage_bundle` instead.

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

    from .qt.i18n import tr

    if os.path.isfile(path):
        return stage_bundle(path)
    sums_path = os.path.join(path, "SHA256SUMS")
    if not os.path.isfile(sums_path):
        raise ValueError(tr("{path} is not a cell-cycle checkpoint: no SHA256SUMS", path=path))
    with open(sums_path, encoding="utf-8") as handle:
        sums = dict(reversed(line.split()) for line in handle if line.strip())
    for name in ("card.json", "model.joblib"):
        target = os.path.join(path, name)
        if name not in sums or not os.path.isfile(target) or _sha256(target) != sums[name]:
            raise ValueError(tr("{name} of {path} does not match its SHA-256; refusing to load",
                                name=name, path=path))
    with open(os.path.join(path, "card.json"), encoding="utf-8") as handle:
        card = json.load(handle)
    if card.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(tr("{path} has format {found}, not {expected}", path=path,
                            found=repr(card.get("format")), expected=repr(CHECKPOINT_FORMAT)))
    if list(card.get("classes") or []) != list(CLASSES):
        raise ValueError(tr(
            "{path} was trained on the classes {found}; this spaCR calls {classes} "
            "(telophase_daughters became a stage of its own). Train the base model again with "
            "spacr.ops_cell_cycle.train_cell_cycle_model", path=path,
            found=card.get("classes"), classes=list(CLASSES)))
    warnings.warn(tr("loading the cell-cycle model {path}: unpickling runs code; load only "
                     "checkpoints you trust", path=path), stacklevel=2)
    held = joblib.load(os.path.join(path, "model.joblib"))
    return {**held, "card": card, "sha256": sums["model.joblib"]}


def fine_tune(base: Mapping[str, Any], cells: pd.DataFrame, labels: Sequence[str], *,
              seed: int = 0, tabular: Optional[str] = None,
              pooled: Optional[pd.DataFrame] = None,
              reference_cells: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The base model refitted on its own labels plus the screen's, the screen's weighted ×5.

    :param base: from :func:`load_cell_cycle_checkpoint` (its ``training`` rows are used), or a
        bundle with ``features``, ``card`` and no ``training`` to fit on the screen's labels
        alone.
    :param cells: this plate's labelled cells.
    :param labels: their stages.
    :param seed: the learner seed.
    :param tabular: the learner; the base card's when None.
    :param pooled: other plates' labelled rows (the model's features, already robust z, and
        ``stage``) for a fine-tune of the whole screen; None refits on this plate's alone.
    :param reference_cells: full wells including unlabelled cells, normalized by the same
        helper as prediction. Omission assumes cells is the full population and warns.
    :returns: a bundle for :func:`save_cell_cycle_checkpoint`, its card recording the base;
        ``weights`` are the sample weights, the screen rows (other plates first, this plate
        last) after the base rows.
    """
    tabular = tabular or base["card"].get("tabular", "hgb")
    features = list(base["features"])
    x_plate = _matrix(cells, features,
                      reference_cells=_training_reference(cells, reference_cells))
    y_plate = _encode(labels)
    training = base.get("training")
    x_base = training[features].to_numpy() if training is not None else np.empty((0, len(features)))
    y_base = _encode(training["stage"]) if training is not None else np.empty(0, int)
    x_pool = (pooled[features].to_numpy() if pooled is not None and len(pooled)
              else np.empty((0, len(features))))
    y_pool = (_encode(pooled["stage"]) if pooled is not None and len(pooled)
              else np.empty(0, int))
    x = np.vstack([x_base, x_pool, x_plate])
    y = np.concatenate([y_base, y_pool, y_plate])
    weight = np.concatenate([np.ones(len(y_base)),
                             np.full(len(y_pool) + len(y_plate), SCREEN_WEIGHT)])
    model = _fit(tabular, seed, x, y, weight)
    card = dict(base["card"], fine_tuned=True, screen_labels=int(len(y_pool) + len(y_plate)),
                screen_weight=SCREEN_WEIGHT, seed=int(seed), tabular=tabular)
    card.pop("checkpoint_id", None)
    rows = pd.DataFrame(x, columns=features)
    rows["stage"] = [CLASSES[i] for i in y]
    return {"model": model, "features": features, "card": card, "training": rows,
            "weights": weight}


def _labelled_rows(cells: pd.DataFrame, labels: pd.Series, features: Sequence[str],
                   plate: str, *, reference_cells: Optional[pd.DataFrame] = None
                   ) -> pd.DataFrame:
    """Labelled model rows normalized against the complete reference wells.

    :param cells: labelled cells with the original object keys.
    :param labels: their stages.
    :param features: the model feature columns.
    :param plate: the plate identifier written to the pool.
    :param reference_cells: all cells of the plate, including unlabelled cells.
    :returns: object keys, full-well robust-z features and stages.
    """
    rows = pd.DataFrame(_matrix(cells, features, reference_cells=reference_cells),
                        columns=list(features))
    for key in KEY:
        rows[key] = cells[key].to_numpy()
    rows["plate"] = str(plate)
    rows["stage"] = np.asarray(labels)
    return rows


def _screen_pool(path: str, plate: str, rows: pd.DataFrame, features: Sequence[str]
                 ) -> pd.DataFrame:
    """The screen's labelled rows of every other plate, with this plate's replaced in the pool.

    The pool file sits beside the labels file, so every plate of a screen (each its own run)
    adds to the same one; rows of another feature set are dropped.

    :returns: the other plates' rows, in plate order.
    """
    from .tabular import read_table, write_table

    pool = (read_table(path, canonicalise=False, report=None) if os.path.isfile(path)
            else pd.DataFrame(columns=[*KEY, *features, "stage"]))
    if set(features) - set(pool.columns):
        pool = pd.DataFrame(columns=[*KEY, *features, "stage"])
    pool["plate"] = pool["plate"].astype(str)
    others = pool[pool["plate"] != str(plate)]
    write_table(pd.concat([others, rows[[*KEY, *features, "stage"]]], ignore_index=True)
                .sort_values(KEY, kind="stable"), path, canonicalise=False)
    return others.sort_values(KEY, kind="stable").reset_index(drop=True)


def predict_stages(bundle: Mapping[str, Any], cells: pd.DataFrame) -> pd.DataFrame:
    """Every class probability and the model's stage of every cell.

    :param bundle: a loaded or fitted model (``model`` and ``features``).
    :param cells: the cell table, with every feature of the model.
    :returns: on the index of ``cells``, ``p_<class>`` for each of :data:`CLASSES` and
        ``model_stage``.
    :raises ValueError: when a feature is missing.
    """
    from .qt.i18n import tr

    missing = [f for f in bundle["features"] if f not in cells.columns]
    if missing:
        raise ValueError(tr("the cell tables lack the model's features {missing}",
                            missing=missing[:5]))
    recorded = str((bundle.get("card") or {}).get("normalisation") or NORMALISATION)
    if not recorded.startswith("robust z per well"):
        raise ValueError(tr("the model was trained with the normalisation {found}; only the "
                            "robust z per well is applied", found=repr(recorded)))
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
        they stay out of the field modes and the fit, as do the cells the cell table's
        DNA-rule ``mitotic`` column flags (when present), through
        :func:`spacr.measure.mitotic_holdout`.
    :returns: the frame and fits of :func:`spacr.measure._cell_table_phases_measure_fc`,
        retaining exact fit populations for a separate fixed-fit diagnostic report.
    """
    from .measure import dna_phases, mitotic_holdout

    table = cells[["_dna_integrated", "well", "site"]].copy()
    table["mitotic"] = mitotic_holdout(
        mitotic, cells["mitotic"] if "mitotic" in cells.columns else None)
    min_cells = int(settings.get("cell_cycle_field_min_cells") or 50)
    return dna_phases(table, dna="_dna_integrated", group="well", field="site",
                      mitotic="mitotic", min_cells=min_cells, capture_diagnostics=True)


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


def _s6_population_diagnostics(cells, dna, fits):
    """Score recorded fit populations and the production population using each fixed fit.

    :param cells: full OPS reference cells with their well labels.
    :param dna: current centred DNA values, in the same row order.
    :param fits: current fixed fits and optional exact fit-population records.
    :returns: diagnostic rows only; missing retained populations remain unavailable.
        No model is fitted and no production verdict is calculated or changed.
    """
    rows = []
    centred = dna["dna_content_centred"].to_numpy(float)
    wells = cells["well"].astype(str).to_numpy()
    for well, info in fits.items():
        fit = info["fit"]
        recorded = info.get("diagnostics") or {}
        populations = (("fit_mask", recorded.get("fit_mask_content")),
                       ("trimmed", recorded.get("trimmed_content")),
                       ("scored", centred[wells == str(well)]))
        bounds = recorded.get("trim_bounds", (float("nan"), float("nan")))
        for population, values in populations:
            available = values is not None
            values = np.asarray([] if values is None else values, dtype=float)
            finite = values[np.isfinite(values)]
            r2, ratio = _fit_quality(values, fit)
            rows.append({"well": str(well), "population": population,
                         "scope": "pooled" if info.get("pooled") else "well",
                         "cells": len(values) if available else None,
                         "finite_cells": len(finite) if available else None,
                         "dna_min": float(finite.min()) if len(finite) else float("nan"),
                         "dna_max": float(finite.max()) if len(finite) else float("nan"),
                         "r2": r2, "g2_g1": ratio, "available": available,
                         "trim_low": float(bounds[0]), "trim_high": float(bounds[1])})
    frame = pd.DataFrame(rows, columns=["well", "population", "scope", "cells", "finite_cells",
                                       "dna_min", "dna_max", "r2", "g2_g1", "available",
                                       "trim_low", "trim_high"])
    for name in ("cells", "finite_cells"):
        frame[name] = frame[name].astype("Int64")
    return frame


def _nuclear_diameter(cells: pd.DataFrame) -> np.ndarray:
    """Each nucleus's equivalent diameter from ``nucleus_area``."""
    return 2.0 * np.sqrt(cells["nucleus_area"].to_numpy(float) / np.pi)


def _pair_centroids(cells: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Resolve OPS profile coordinates first, followed by supported legacy cell tables."""
    for pair in (("nucleus_cy", "nucleus_cx"),
                 ("nucleus_centroid_y", "nucleus_centroid_x"),
                 ("centroid_row", "centroid_col")):
        if set(pair) <= set(cells.columns):
            return pair
    return None


def _field_rows(cells: pd.DataFrame) -> List[np.ndarray]:
    """Group field positions once in lexical order without scanning all cells per field."""
    fields = cells["well"].astype(str) + "/" + cells["site"].astype(str)
    if "plate" in cells:
        fields = cells["plate"].astype(str) + "/" + fields
    codes, names = pd.factorize(fields, sort=True)
    order = np.argsort(codes, kind="stable")
    return list(np.split(order, np.cumsum(np.bincount(codes, minlength=len(names)))[:-1]))


def _pair_neighbours(cells: pd.DataFrame, two_n: np.ndarray, log2_dna: np.ndarray,
                     groups: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Cache directed nearby 2N partners with the original diameter and DNA gates.

    :param cells: full cell table with a supported centroid pair and nucleus_area.
    :param two_n: eligibility at the G1/G2 midpoint.
    :param log2_dna: log2 centred DNA content.
    :param groups: precomputed field row indices.
    :returns: source and target positions for partners within 2.5 source diameters
        and 0.3 log2 DNA units, excluding the source itself.
    """
    from scipy.spatial import cKDTree

    points = cells[list(_pair_centroids(cells))].to_numpy(float)
    diameter = _nuclear_diameter(cells)
    valid = (two_n & np.isfinite(points).all(axis=1) & np.isfinite(log2_dna)
             & np.isfinite(diameter) & (diameter > 0))
    sources, targets = [], []
    for group in groups:
        rows = group[valid[group]]
        if len(rows) < 2:
            continue
        tree = cKDTree(points[rows])
        nearby = tree.query_ball_point(points[rows], 2.5 * diameter[rows])
        for row, neighbours in zip(rows, nearby):
            candidates = rows[np.asarray(neighbours, dtype=np.intp)]
            partners = candidates[(candidates != row)
                                  & (np.abs(log2_dna[candidates] - log2_dna[row]) <= 0.3)]
            if len(partners):
                sources.append(np.full(len(partners), row, dtype=np.intp))
                targets.append(partners)
    empty = np.empty(0, dtype=np.intp)
    return (np.concatenate(sources) if sources else empty,
            np.concatenate(targets) if targets else empty)


def _pair_share(cells: pd.DataFrame, called: np.ndarray, two_n: np.ndarray,
                log2_dna: np.ndarray, *,
                neighbours: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> float:
    """Share of 2N daughter calls with a selected eligible partner, using cached geometry.

    :param cells: the full reference population.
    :param called: the observed or within-field permuted daughter-stage labels.
    :param two_n: eligibility below the G1/G2 midpoint.
    :param log2_dna: centred DNA on the log2 scale.
    :param neighbours: precomputed directed partner graph; None builds it once.
    :returns: the original S2 share, or NaN when no selected cell is 2N.
    """
    if neighbours is None:
        neighbours = _pair_neighbours(cells, two_n, log2_dna, _field_rows(cells))
    left, right = neighbours
    selected = np.asarray(called, bool) & two_n
    total = int(selected.sum())
    counts = np.bincount(left, weights=selected[right], minlength=len(cells))
    return float(np.count_nonzero(selected & (counts > 0)) / total) if total else float("nan")


def _s5_power(baseline: float, *, enrichment: float = 3.0,
              alpha: float = 0.05, power: float = 0.80) -> Dict[str, Any]:
    """Required arrest cells for the specified one-sided proportion alternative.

    :param baseline: observed non-targeting mitotic share.
    :param enrichment: alternative mitotic share divided by baseline, above one.
    :param alpha: one-sided type-I error.
    :param power: desired probability of rejecting the null at the alternative.
    :returns: baseline, alternative, enrichment, alpha, power and required_cells;
        the last is infinite when the baseline cannot identify an alternative.
    :raises ValueError: for invalid rates, enrichment, alpha or power.
    """
    from statistics import NormalDist
    from .qt.i18n import tr

    if (not np.isfinite(baseline) or not 0 <= baseline <= 1
            or not np.isfinite(enrichment) or enrichment <= 1
            or not 0 < alpha < 0.5 or not 0.5 < power < 1):
        raise ValueError(tr("S5 needs a baseline in [0, 1], enrichment above 1, "
                            "alpha in (0, 0.5) and power in (0.5, 1)"))
    alternative = min(float(baseline * enrichment), 1.0)
    required = float("inf")
    if alternative > baseline:
        normal = NormalDist()
        numerator = (normal.inv_cdf(1 - alpha) * np.sqrt(baseline * (1 - baseline))
                     + normal.inv_cdf(power) * np.sqrt(alternative * (1 - alternative)))
        required = int(np.ceil((numerator / (alternative - baseline)) ** 2))
    return {"baseline": float(baseline), "alternative": alternative,
            "enrichment": float(enrichment), "alpha": float(alpha), "power": float(power),
            "required_cells": required}


def self_checks(cells: pd.DataFrame, stages: pd.DataFrame, dna: pd.DataFrame,
                fits: Mapping[str, Any], *, card: Mapping[str, Any], seed: int = 0,
                arrest_genes: Optional[Sequence[str]] = ARREST_GENES,
                control_prefix: str = "nontargeting", permutations: int = PERMUTATIONS,
                labelled: Optional[Mapping[str, Any]] = None,
                arrest_enrichment: float = 3.0) -> pd.DataFrame:
    """Checks S1-S6 of one plate.

    :param cells: the plate's cells: ``well``, ``site``, ``nucleus_area``,
        OPS ``nucleus_cy``/``nucleus_cx`` (S2; legacy centroid pairs also accepted)
        and ``gene`` (S5) when present.
    :param stages: from :func:`predict_stages`.
    :param dna: the frame of :func:`_dna_phases`.
    :param fits: its fits per group.
    :param card: the model card; its ``reference`` S1 value and S4 φ, when present, tighten
        S1 and S4 as section 5 of the spec says.
    :param seed: the permutations' seed.
    :param arrest_genes: S5's knockouts with a known mitotic arrest; None skips S5.
    :param control_prefix: the non-targeting gene prefix.
    :param permutations: permutations of S2, S3 and S5. The S2 null now
        permutes daughter-eligible cells within each field, preserving their
        daughter count; the previous all-rows null is replaced.
    :param labelled: ``{"truth", "predicted", "shuffled_macro"}`` for S3; None skips it.
    :param arrest_enrichment: S5 alternative relative to observed NT, default threefold;
        one-sided alpha 0.05 and power 0.80 determine its minimum arrest count.
    :returns: one row per check: ``check``, ``stages`` (judged), ``value``, ``control``,
        ``threshold``, ``passed`` (True, False or None when skipped) and ``note``; the frame's
        ``attrs["samples"]`` keeps S2's observed share and its permutation shares for the
        report. S5 also reports baseline, alternative, enrichment, alpha, power,
        required_cells, judged_cells and status (UNDERPOWERED, PASS, FAIL or SKIPPED).
    """
    from .qt.i18n import tr

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
                 "note": tr("share of prometaphase and metaphase calls above the G1/G2 midpoint")})
    late = np.isin(stage, DAUGHTER_STAGES)
    centroids = _pair_centroids(cells)
    if late.sum() >= 2 and centroids is not None and "nucleus_area" in cells:
        two_n = ~four_n
        groups = [rows[two_n[rows]] for rows in _field_rows(cells)]
        neighbours = _pair_neighbours(cells, two_n, log2_dna, groups)
        observed = _pair_share(cells, late, two_n, log2_dna, neighbours=neighbours)
        null = []
        for _ in range(int(permutations)):
            shuffled = late.copy()
            for idx in groups:
                shuffled[idx] = late[rng.permutation(idx)]
            null.append(_pair_share(cells, shuffled, two_n, log2_dna,
                                    neighbours=neighbours))
        control = float(np.nanpercentile(null, 97.5))
        samples["S2"] = (observed, [float(v) for v in null])
        rows.append({"check": "S2", "stages": ",".join(DAUGHTER_STAGES), "value": observed,
                     "control": control, "threshold": control,
                     "passed": bool(observed > control), "note": tr("2N daughter pairs")})
    else:
        reason = (tr("skipped: fewer than two daughter-stage calls") if late.sum() < 2 else
                  tr("skipped: S2 needs nucleus_area and a centroid pair: nucleus_cy/nucleus_cx, "
                     "nucleus_centroid_y/nucleus_centroid_x or centroid_row/centroid_col"))
        rows.append({"check": "S2", "stages": ",".join(DAUGHTER_STAGES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": reason})
    if labelled is not None:
        value = _macro_f1(labelled["truth"], labelled["predicted"])
        control = float(np.percentile(labelled["shuffled_macro"], 97.5))
        rows.append({"check": "S3", "stages": ",".join(CLASSES), "value": value,
                     "control": control, "threshold": control, "passed": bool(value > control),
                     "note": tr("held-out macro F1 against shuffled-label models")})
    else:
        rows.append({"check": "S3", "stages": ",".join(CLASSES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": tr("skipped: no user labels")})
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
                     "note": (tr("overdispersion of the mitotic share across fields")
                              if np.isfinite(limit) else
                              tr("overdispersion of the mitotic share across fields; no "
                                 "base-card φ, not judged"))})
    else:
        rows.append({"check": "S4", "stages": ",".join(MITOTIC_STAGES), "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "note": tr("skipped: fewer than two fields with 50 cells")})
    if arrest_genes and "gene" in cells.columns:
        genes = cells["gene"].astype(str).to_numpy()
        arrest = np.isin(np.char.upper(genes.astype(str)), [g.upper() for g in arrest_genes])
        control = np.char.startswith(np.char.lower(genes.astype(str)), control_prefix.lower())
        judged = int(arrest.sum())
        power_info = (_s5_power(float(mitotic[control].mean()), enrichment=arrest_enrichment)
                      if control.any() else {})
        if control.any() and judged < power_info["required_cells"]:
            rows.append({"check": "S5", "stages": "prometaphase,metaphase",
                         "value": float("nan"), "control": float("nan"),
                         "threshold": power_info["alpha"], "passed": None,
                         "status": "UNDERPOWERED", "judged_cells": judged, **power_info,
                         "note": tr("UNDERPOWERED: S5 has {judged} arrest cells; needs {needed} "
                                    "for {enrichment}x enrichment at alpha {alpha}, power {power}",
                                    judged=judged, needed=power_info["required_cells"],
                                    enrichment=power_info["enrichment"], alpha=power_info["alpha"],
                                    power=power_info["power"])})
        elif control.any() and judged:
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
                         "status": "PASS" if p_value < 0.05 else "FAIL",
                         "judged_cells": judged, **power_info,
                         "note": tr("arrest knockouts vs non-targeting, permutation p = {p}",
                                    p=f"{p_value:.4f}")})
        else:
            rows.append({"check": "S5", "stages": "prometaphase,metaphase", "value": float("nan"),
                         "control": float("nan"), "threshold": float("nan"), "passed": None,
                         "status": "SKIPPED", "judged_cells": judged, **power_info,
                         "note": tr("skipped: no non-targeting cells for the S5 baseline")})
    else:
        rows.append({"check": "S5", "stages": "prometaphase,metaphase", "value": float("nan"),
                     "control": float("nan"), "threshold": float("nan"), "passed": None,
                     "status": "SKIPPED", "judged_cells": 0,
                     "note": tr("skipped: no arrest set or no gene per cell")})
    quality = []
    for well, info in fits.items():
        here = (cells["well"].astype(str) == str(well)).to_numpy()
        quality.append(_fit_quality(centred[here], info["fit"]))
    r2 = min((q[0] for q in quality if np.isfinite(q[0])), default=float("nan"))
    ratios = [q[1] for q in quality]
    ratio_ok = all(G2_G1_RANGE[0] <= r <= G2_G1_RANGE[1] for r in ratios)
    span = {"low": f"{min(ratios, default=float('nan')):.3f}",
            "high": f"{max(ratios, default=float('nan')):.3f}",
            "lo": G2_G1_RANGE[0], "hi": G2_G1_RANGE[1]}
    rows.append({"check": "S6", "stages": "G1,S,G2", "value": r2,
                 "control": float(np.median(ratios)) if ratios else float("nan"),
                 "threshold": FIT_R2, "passed": bool(np.isfinite(r2) and r2 >= FIT_R2 and ratio_ok),
                 "note": (tr("lowest fit R2; G2/G1 {low}-{high} (pass {lo}-{hi})", **span)
                          if ratio_ok else
                          tr("lowest fit R2; G2/G1 {low}-{high} (pass {lo}-{hi}); ratio out of "
                             "range", **span))})
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
    from .ops_tvn import dna_posterior_column

    for ph in PHASES:
        out[dna_posterior_column(ph)] = post[ph]
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
        from .qt.i18n import tr

        raise ValueError(tr("the cell-cycle labels {path} lack {missing}", path=path,
                            missing=missing))
    _encode(table["stage"])
    table["plate"] = table["plate"].astype(str)
    table["well"] = table["well"].astype(str)
    return table.drop_duplicates(KEY, keep="last").reset_index(drop=True)


def _options(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``ops_cell_cycle_*`` settings, checked."""
    from .qt.i18n import tr

    mode = settings.get("ops_cell_cycle")
    family = str(settings.get("ops_cell_cycle_model_family") or "tabular").lower()
    tabular = str(settings.get("ops_cell_cycle_tabular") or "hgb").lower()
    finetune = str(settings.get("ops_cell_cycle_finetune") or "screen").lower()
    if mode not in ("DNA", "DNA+MT"):
        raise ValueError(tr("ops_cell_cycle must be None, 'DNA' or 'DNA+MT'; got {value}",
                            value=repr(mode)))
    if family not in FAMILIES:
        raise ValueError(tr("ops_cell_cycle_model_family must be one of {allowed}",
                            allowed=list(FAMILIES)))
    if tabular not in TABULAR:
        raise ValueError(tr("ops_cell_cycle_tabular must be one of {allowed}",
                            allowed=list(TABULAR)))
    if finetune not in FINETUNE:
        raise ValueError(tr("ops_cell_cycle_finetune must be one of {allowed}",
                            allowed=list(FINETUNE)))
    mt = settings.get("ops_cell_cycle_mt_channel") or None
    if mode == "DNA+MT" and not mt:
        raise ValueError(tr("ops_cell_cycle 'DNA+MT' needs ops_cell_cycle_mt_channel"))
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
    """A figure as an inline ``data:`` URI, without a timestamp, so reruns write the same bytes.

    Written through :func:`spacr.plot.save_figure`, the one figure writer, as a
    150 dpi PNG (the report embeds it, so the format is fixed).
    """
    import base64
    import os
    import tempfile

    from .plot import save_figure

    with tempfile.TemporaryDirectory(prefix="spacr_cell_cycle_") as folder:
        path = save_figure(figure, os.path.join(folder, "figure.png"), fmt="png", dpi=150,
                           bbox_inches="tight", metadata={"Software": None})
        with open(path, "rb") as handle:
            data = handle.read()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _dna_figure(cells: pd.DataFrame, dna: pd.DataFrame, fits: Mapping[str, Any]) -> str:
    """Each well's DNA-content histogram with its fitted G1, S and G2 and the gates."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
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
        _register_figure_data(
            figure, lambda: pd.DataFrame({
                "well": cells["well"].astype(str).to_numpy(),
                "dna_content_centred": dna["dna_content_centred"].to_numpy(float)}),
            x="dna_content_centred", hue="well", kind="histogram",
            title="DNA content per well with the fitted G1, S and G2")
        return _png_data(figure)


def _checks_figure(cells: pd.DataFrame, stages: pd.DataFrame, dna: pd.DataFrame,
                   fits: Mapping[str, Any], checks: pd.DataFrame) -> str:
    """S1 and S2 as bars with points, each against its control."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from .figures.bundle import _register_figure_data
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
        _register_figure_data(
            figure, lambda: _checks_rows(all_share, mitotic_share, samples),
            x="group", y="value", hue="check", kind="bar",
            title="Self-checks S1 and S2 against their controls")
        return _png_data(figure)


def _checks_rows(all_share, mitotic_share, samples) -> pd.DataFrame:
    """The tidy rows the self-check figure draws: one per field share or null draw."""
    rows = [("S1", "all cells (control)", float(v)) for v in all_share]
    rows += [("S1", "prometaphase + metaphase", float(v)) for v in mitotic_share]
    if "S2" in samples:
        observed, null = samples["S2"]
        rows += [("S2", "shuffled (control)", float(v)) for v in null]
        rows += [("S2", "daughter stages", float(observed))]
    return pd.DataFrame(rows, columns=["check", "group", "value"])


def _report_html(path: str, plate: str, checks: pd.DataFrame, counts: pd.Series,
                 failed: Sequence[str], notes: Sequence[str], card: Mapping[str, Any],
                 figures: Sequence[Tuple[str, str]] = ()) -> str:
    """Report checks, stage counts and notes, explaining unjudged DNA-fit stages.

    A failed S6 R² below 0.90 labels its report row and G1, S and G2 as
    not judged, with the measured R². Counts, check verdicts and collapse
    decisions remain unchanged. Optional retained populations appear in a
    separate fixed-fit diagnostic block that never changes the verdict.
    """
    import html

    from .qt.i18n import tr

    table = _html_table
    esc = html.escape
    report_checks = checks.copy()
    stage_judgements = []
    s6 = checks.loc[checks["check"] == "S6"]
    if len(s6):
        value = float(s6["value"].iloc[0])
        if _verdict(s6["passed"].iloc[0]) is False and np.isfinite(value) and value < FIT_R2:
            reason = tr("not judged (S6 R^2 {r2} < {threshold})",
                        r2=f"{value:.2f}", threshold=f"{FIT_R2:.2f}")
            report_checks.loc[s6.index, "status"] = reason
            stage_judgements = [f"<p>{esc(stage)}: {esc(reason)}</p>" for stage in PHASES]
    f1 = pd.DataFrame([{"stage": k, "F1": v, "CI low": card.get("cv", {}).get("f1_ci95", {})
                        .get(k, [np.nan, np.nan])[0], "CI high": card.get("cv", {})
                        .get("f1_ci95", {}).get(k, [np.nan, np.nan])[1]}
                       for k, v in card.get("cv", {}).get("f1", {}).items()])
    diagnostic_sections = []
    diagnostic = checks.attrs.get("s6_population_diagnostics")
    if diagnostic is not None and len(diagnostic):
        diagnostic = diagnostic.copy()
        diagnostic["population"] = diagnostic["population"].map({
            "fit_mask": tr("Fit mask"), "trimmed": tr("Trimmed fitting range"),
            "scored": tr("Production scored population")})
        diagnostic["scope"] = diagnostic["scope"].map({
            "well": tr("Well"), "pooled": tr("Pooled wells")})
        diagnostic["available"] = diagnostic["available"].map({
            True: tr("available"), False: tr("not retained")})
        diagnostic = diagnostic.rename(columns={
            "well": tr("Well"), "population": tr("Population"), "scope": tr("Fit scope"),
            "cells": tr("Selected cells"), "finite_cells": tr("Finite cells"),
            "dna_min": tr("Minimum DNA"), "dna_max": tr("Maximum DNA"),
            "r2": tr("Fit R2"), "g2_g1": tr("G2/G1 ratio"), "available": tr("Availability"),
            "trim_low": tr("Trim lower bound"), "trim_high": tr("Trim upper bound")})
        diagnostic_sections = [
            "<h2>" + esc(tr("S6 populations: diagnostic, not used for the verdict")) + "</h2>",
            "<p>" + esc(tr(
                "Each row scores the same saved fit on the labelled population; no refit is performed. "
                "R^2 uses 100 bins between the finite population 0.5th and 99.5th percentiles, "
                "with the existing finite-cell density scaling. The production verdict uses "
                "all finite scored cells.")) + "</p>", table(diagnostic)]
    body = [f"<h1>{esc(tr('Cell cycle (OPS), plate {plate}', plate=str(plate)))}</h1>",
            "<p>" + esc(tr(
                "Stages: G1, S, G2 (DNA-content fit); g2_prophase (G2 prophase, a flag class); "
                "prometaphase; metaphase; ana_telo_cyto (anaphase, telophase and cytokinesis); "
                "telophase_daughters (the two 2N daughters); artefact. CI: 95 % "
                "field-bootstrap confidence interval. NT: non-targeting.")) + "</p>",
            f"<h2>{esc(tr('Self-checks'))}</h2>", table(report_checks),
            "<p>" + esc(tr("Collapsed: {stages}", stages=", ".join(failed) or tr("none")))
            + "</p>",
            *stage_judgements,
            *diagnostic_sections,
            f"<h2>{esc(tr('Cells per stage'))}</h2>",
            table(counts.rename_axis("stage").reset_index(name="cells")),
            f"<h2>{esc(tr('Base model cross-validation F1'))}</h2>",
            table(f1) if len(f1) else f"<p>{esc(tr('none'))}</p>",
            "<p>" + esc(tr("Shuffled-label control, 97.5th percentile of macro F1: {value}",
                           value=card.get("cv", {}).get("shuffled_macro_f1_p975"))) + "</p>",
            *[f"<h2>{esc(title)}</h2><img src='{uri}' alt='{esc(title)}'>"
              for title, uri in figures],
            f"<h2>{esc(tr('Notes'))}</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes)
            + "</ul>"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html><meta charset='utf-8'><title>"
                     + esc(tr("Cell cycle")) + "</title>" + "".join(body))
    return path


def _needed_columns(path: str, options: Mapping[str, Any], base: Optional[Mapping[str, Any]]
                    ) -> List[str]:
    """The columns of one well's cell table that the phase reads: no others are loaded.

    The keys, the model features (the checkpoint's, else those the tables offer for the DNA
    and tubulin channels), the integrated DNA content, the nuclear area and the centroids
    that :func:`self_checks` uses, and the DNA-rule ``mitotic`` column when the table has one.
    """
    from .tabular import table_columns

    columns = list(table_columns(path, canonicalise=False))
    dna = str(options["dna"] or "dapi").lower()
    mt = options["mt"] if options["mode"] == "DNA+MT" else None
    features = list(base["features"]) if base is not None else _features(columns, dna, mt)
    pairs = [c for pair in (("nucleus_cy", "nucleus_cx"), ("nucleus_centroid_y",
                                                         "nucleus_centroid_x"),
                            ("centroid_row", "centroid_col")) for c in pair]
    wanted = [c for c in ("site", "label", "mitotic") if c in columns]
    return wanted + features + [f"nucleus_{dna}_integrated_intensity", "nucleus_area"] + pairs


def run_cell_cycle_phase(db: str, plate: str, wells: Sequence[str],
                         settings: Mapping[str, Any], destination: str, *,
                         permutations: int = PERMUTATIONS,
                         arrest_enrichment: float = 3.0) -> Dict[str, Any]:
    """The ``ph_cell_cycle`` phase: stages, probabilities and flags of every cell of a plate.

    A full-plate run requires at least 20 GB declared RAM in sched or a Home job
    template: six wells and 5.53 million cells took 2.2-2.6 hours at four threads,
    with peak RSS 17.6 GB; a 16 GB allocation failed. A single CP186A well took
    22-37 minutes and 2.8 GB.

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
    :param permutations: S2, S3 and S5 control draws, default 500.
    :param arrest_enrichment: the S5 power alternative, default threefold.
    :returns: the phase report; ``label_queue`` names the queue folder, its size and the
        labels file, or is None.
    :raises ValueError: for a bad setting, missing cell tables or no model to start from.
    """
    from .ops_engine import _ph_cells_path, _replace_well_rows, _rows_for_wells
    from .qt.i18n import tr
    from .tabular import read_table_columns

    options = _options(settings)
    notes: List[str] = []
    if options["family"] != "tabular":
        notes.append(tr("ops_cell_cycle_model_family {family} is not available in this build; "
                        "the tabular learner {tabular} was used",
                        family=repr(options["family"]), tabular=repr(options["tabular"])))
    if not options["base"] and not options["labels"]:
        raise ValueError(tr("ops_cell_cycle needs a base model (ops_cell_cycle_base_model) or "
                            "user labels (ops_cell_cycle_labels); no shipped base model is "
                            "registered yet"))
    base = load_cell_cycle_checkpoint(options["base"]) if options["base"] else None
    frames = []
    for well in wells:
        path = _ph_cells_path(destination, well)
        if os.path.isfile(path):
            frame = read_table_columns(path, _needed_columns(path, options, base))
            frame["well"] = str(well)
            frames.append(frame)
    if not frames:
        raise ValueError(tr("no phenotype cell table for wells {wells}", wells=list(wells)))
    cells = pd.concat(frames, ignore_index=True)
    cells["plate"] = str(plate)
    dna = str(options["dna"] or "dapi").lower()
    integrated = f"nucleus_{dna}_integrated_intensity"
    if integrated not in cells.columns:
        raise ValueError(tr("the cell tables lack {column}, the DNA content the fit needs",
                            column=integrated))
    cells["_dna_integrated"] = cells[integrated].to_numpy(float)
    _attach_genes(cells, db, plate, wells, settings, _rows_for_wells)
    labelled = None
    model = base
    if options["labels"]:
        labels = read_cell_cycle_labels(options["labels"])
        joined = cells.reset_index().merge(labels, on=KEY, how="inner")
        if len(joined) and options["finetune"] != "none":
            start = base if base is not None else {
                "features": _features(cells.columns, dna,
                                      options["mt"] if options["mode"] == "DNA+MT" else None),
                "training": None,
                "card": {"format": CHECKPOINT_FORMAT, "family": "tabular",
                         "tabular": options["tabular"], "input": options["mode"],
                         "classes": list(CLASSES), "parents": PARENTS, "dna": dna,
                         "mt": options["mt"], "seed": options["seed"],
                         "spacr_version": _version(), "trained_on": "screen labels only"}}
            pooled = None
            if options["finetune"] == "screen":
                pooled = _screen_pool(f"{options['labels']}.features.parquet", plate,
                                      _labelled_rows(joined, joined["stage"],
                                                     start["features"], plate, reference_cells=cells),
                                      start["features"])
                notes.append(tr("fine-tuned on the screen's labels: {here} of this plate and "
                                "{others} of other plates", here=len(joined),
                                others=len(pooled)))
            model = fine_tune(start, joined, joined["stage"], seed=options["seed"],
                              tabular=options["tabular"], pooled=pooled,
                              reference_cells=cells)
            labelled = _labelled_scores(model, joined, options, reference_cells=cells,
                                         permutations=permutations)
    stages = predict_stages(model, cells)
    excluded = stages["model_stage"].to_numpy() != "interphase"
    dna_frame, fits = _dna_phases(cells, settings, excluded)
    fits = {str(k): v for k, v in fits.items()}
    checks = self_checks(cells, stages, dna_frame, fits, card=model["card"], seed=options["seed"],
                         arrest_genes=options["arrest"],
                         control_prefix=str(settings.get("ops_control_prefix") or "nontargeting"),
                         permutations=permutations, labelled=labelled,
                         arrest_enrichment=arrest_enrichment)
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
                  "channels": settings.get("ops_cell_cycle_label_channels"),
                  "labels": options["labels"] or os.path.join(destination, "cell_cycle_labels",
                                                              "labels.csv")}
    counts = table["cell_cycle_stage"].value_counts().sort_index()
    figures = [(tr("DNA content per well, with the fitted G1, S and G2 and the gates"),
                _dna_figure(cells, dna_frame, fits)),
               (tr("4N in mitosis (S1) and 2N daughter pairs (S2), each against its control"),
                _checks_figure(cells, stages, dna_frame, fits, checks))]
    checks.attrs["s6_population_diagnostics"] = _s6_population_diagnostics(cells, dna_frame, fits)
    report = _report_html(os.path.join(destination, f"ops_cell_cycle_report_{plate}.html"),
                          plate, checks, counts, union, notes, model["card"], figures)
    return {"cells": int(len(table)), "collapsed": union, "plate_collapsed": failed,
            "stages": {k: int(v) for k, v in counts.items()}, "notes": notes,
            "report": report, "model_id": model_id, "label_queue": queued}


def _labelled_scores(model: Mapping[str, Any], joined: pd.DataFrame,
                     options: Mapping[str, Any], *,
                     reference_cells: Optional[pd.DataFrame] = None,
                     permutations: int = PERMUTATIONS) -> Dict[str, Any]:
    """Score screen labels using the same complete reference wells as prediction.

    :param model: fitted bundle with features and optional pooled training rows.
    :param joined: cells joined to user labels.
    :param options: learner and random seed.
    :param reference_cells: full wells including unlabelled cells.
    :param permutations: shuffled-label control draws.
    :returns: held-out predictions and shuffled-label controls for S3.
    """
    features = list(model["features"])
    x = _matrix(joined, features, reference_cells=reference_cells)
    y = _encode(joined["stage"])
    groups = (joined["well"].astype(str) + "/" + joined["site"].astype(str)).to_numpy()
    training = model.get("training")
    weights = model.get("weights")
    if training is not None and len(training) > len(y):
        x_base = training[features].to_numpy()[:len(training) - len(y)]
        y_base = _encode(training["stage"])[:len(training) - len(y)]
        w_base = (np.asarray(weights, float)[:len(training) - len(y)] if weights is not None
                  else np.ones(len(y_base)))
    else:
        x_base, y_base, w_base = np.empty((0, len(features))), np.empty(0, int), np.empty(0)
    x_all = np.vstack([x_base, x])
    y_all = np.concatenate([y_base, y])
    g_all = np.concatenate([np.array(["fixed"] * len(y_base), object), groups])
    w_all = np.concatenate([w_base, np.full(len(y), SCREEN_WEIGHT)])
    scored = np.concatenate([np.zeros(len(y_base), bool), np.ones(len(y), bool)])
    predicted = _out_of_fold(options["tabular"], options["seed"], x_all, y_all, g_all, w_all,
                             scored)[scored]
    macro, per_class = _shuffled_f1(options["tabular"], options["seed"], x_all, y_all, g_all,
                                    w_all, scored, int(permutations))
    return {"truth": y, "predicted": predicted, "groups": groups, "shuffled_macro": macro,
            "shuffled": per_class}


def _attach_genes(cells: pd.DataFrame, db: str, plate: str, wells: Sequence[str],
                  settings: Mapping[str, Any], rows_for_wells) -> None:
    """Put each cell's ``gene`` on ``cells`` from the merge, the barcodes and the design (S5).

    Cells without a merge, a call or a design row get no gene; without any, S5 is skipped.
    """
    from .ops_engine import _ph_design

    design_path = settings.get("ops_design")
    from .ops_store import _exists

    merged = rows_for_wells(db, "ops_merge", plate, list(wells)) if _exists(db) else None
    barcodes = rows_for_wells(db, "ops_barcodes", plate, list(wells)) if _exists(db) else None
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
    from .ops_store import _exists, read_table, row_count

    if not _exists(db) or row_count(db, "ops_cell_cycle_checks") is None:
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
                      labels: str, channels: Optional[Sequence[int]] = None,
                      ranges: Optional[Sequence[Sequence[float]] | Mapping[str, Sequence[float]]] = None,
                      radius: int = 30, crop_source=None) -> str:
    """Write blind stage-labelling tiles from unmasked context and the own object mask.

    A fixed 96 by 96 source window starts at the rounded table centroid minus
    48; cells whose window crosses the available field context are excluded.
    ``radius`` is a dilation of the own nucleus mask by one Euclidean disk,
    independent of that window. Funk defaults use the nucleus mask. Pixels outside
    the dilation are black, while neighbouring objects inside it remain visible.

    The panels show grey DNA, grey tubulin and their blue-DNA/green-tubulin
    composite with a zero red channel. Each panel is enlarged threefold with
    nearest neighbours, followed by a one-pixel yellow own-mask outline. Two
    four-pixel grey separators give a tile of 872 by 288 pixels. A white 10 µm
    scale bar requires explicit source pixel calibration; no physical size is
    inferred from image dimensions.

    The fixed Funk display bounds are DAPI 550–23,039 and GFP 1,084–5,002 in raw
    uint16 units. ``ranges`` overrides these bounds in selected-channel order;
    no crop or queue statistics are used. Opaque PNG names map exactly to cells
    in the separate ``queue.csv`` manifest, which must stay outside blind inputs.
    Annotation 1 to 7 follows :data:`CLASSES`.

    :param queue: unique cells from :func:`build_label_queue`, in display order;
        optional ``cy, cx`` are field coordinates and ``pixel_size_um`` supplies
        source calibration. A crop with a stored bounding box keeps its origin.
    :param crop_root: a ``ph_crops`` context-crop project, or a field-context
        folder with ``crop_index.csv`` (keys plus ``cy, cx``), ``fields/`` and
        ``masks/``. The supplied Funk plate-A layout is supported directly; its
        recorded calibration assumption is 0.33 µm per pixel. Other field
        indices supply ``field_path, mask_path, pixel_size_um`` explicitly.
    :param folder: the Annotate project folder to create.
    :param labels: the CSV to which collected answers are appended.
    :param channels: up to two zero-based crop indices, DNA first and tubulin
        second; None selects the first two available planes. Other stains can
        be selected in either slot with explicit display bounds.
    :param ranges: fixed ``(low, high)`` pairs in selected-channel order, or a
        mapping with ``dna`` and optional ``mt`` pairs; None uses the Funk bounds.
    :param radius: nonnegative nucleus-mask dilation radius in source pixels,
        30 by default, giving a 61 by 61 Euclidean disk.
    :param crop_source: optional source exposing ``get_raw(row)`` and returning
        unmasked ``(pixels (C, H, W), own_object_mask)`` through existing spaCR
        readers. Pixel calibration can be supplied as ``pixel_size_um`` or an
        existing micrometre :class:`spacr.layers.Spacing` on the source. None
        uses native OPS context crops or full fields with the nucleus mask; masked
        legacy crops cannot provide neighbour context.
    :returns: ``folder``; excluded identities and reasons are kept in ``queue.json``.
    :raises ValueError: for invalid controls, missing context or nucleus masks,
        missing physical calibration, or a queue with no complete windows.
    """
    import uuid

    from PIL import Image, ImageDraw
    from scipy import ndimage

    from .annotation_dataset import png_list_frame, write_png_list
    from .crop_store import pool_attributes, read_crop
    from .ops_run_infra import crop_index
    from .image_store import read_image
    from .ops_engine import _ph_read_masks
    from .qt.i18n import tr
    from .tabular import read_table, write_table

    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError(tr("Label queue radius must be a nonnegative integer in pixels."))
    if queue.empty or queue.duplicated(KEY).any():
        raise ValueError(tr("Select at least one cell and include each cell only once in the label queue."))
    requested = None if channels is None else list(channels)
    if requested is not None and (not requested or len(requested) > 2
            or any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) or c < 0
                   for c in requested) or len(set(requested)) != len(requested)):
        raise ValueError(tr("Select one or two distinct zero-based label queue channel indices."))
    role_ranges = isinstance(ranges, Mapping)
    if role_ranges:
        if "dna" not in ranges:
            raise ValueError(tr("A label queue range mapping needs a dna pair and an optional mt pair."))
        ranges = [ranges[name] for name in ("dna", "mt") if name in ranges]
    bounds = None if ranges is None else np.asarray(ranges, dtype=float)
    if bounds is not None and (bounds.ndim != 2 or bounds.shape[1] != 2
            or not np.isfinite(bounds).all() or not (bounds[:, 1] > bounds[:, 0]).all()):
        raise ValueError(tr("Give one finite increasing display range for each selected label queue channel."))
    sources, attributes, field_cache = {}, {}, {}
    field_index_path = os.path.join(crop_root, "crop_index.csv")
    field_index = None
    if crop_source is None and os.path.isfile(field_index_path):
        field_index = read_table(field_index_path, canonicalise=False, report=None)
        missing = set(KEY + ["cy", "cx"]) - set(field_index)
        if missing:
            raise ValueError(tr("The context crop index needs columns {columns}.", columns=sorted(missing)))
        if len(field_index[["plate", "well"]].drop_duplicates()) > 1 and not {"field_path", "mask_path"}.issubset(field_index):
            raise ValueError(tr("A multi-well context index needs explicit field_path and mask_path columns."))
    if crop_source is None:
        for plate, rows in queue.groupby("plate", sort=False):
            index = (field_index[field_index["plate"].astype(str) == str(plate)].copy()
                     if field_index is not None else crop_index(
                         crop_root, str(plate), sorted(rows["well"].astype(str).unique())))
            if not len(index):
                raise ValueError(tr("no crops of plate {plate} under {root}; run ph_crops first",
                                    plate=plate, root=crop_root))
            if "well" not in index:
                raise ValueError(tr("Legacy OPS shards contain masked crops; provide full-field or unmasked nucleus context crops."))
            if index.duplicated(["well", "site", "label"]).any():
                raise ValueError(tr("The crop index contains duplicate cells; rebuild ph_crops before labelling."))
            if field_index is not None:
                index["queue_context_format"] = "field_index"
            sources.update({(str(plate), str(row["well"]), int(row["site"]), int(row["label"])): row
                            for row in index.to_dict("records")})
            for well in rows["well"].astype(str).unique():
                attributes[(str(plate), well)] = (
                    {"pixel_size_um": 0.33} if field_index is not None and str(plate) == "LaC024A"
                    else {} if field_index is not None
                    else pool_attributes(crop_root, str(plate), well, "cell"))
    if crop_source is not None and getattr(crop_source, "context", True) is False:
        raise ValueError(tr("Stage labelling needs unmasked full-field context; enable context in the crop source."))
    data = os.path.join(folder, "data")
    os.makedirs(data, exist_ok=True)
    os.makedirs(os.path.join(folder, "measurements"), exist_ok=True)
    paths, included, excluded, calibrations = [], [], [], []
    selected = requested
    fixed = bounds
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    for row in queue.to_dict("records"):
        key = (str(row["plate"]), str(row["well"]), int(row["site"]), int(row["label"]))
        record = sources.get(key, row)
        meta = attributes.get(key[:2], {})
        if crop_source is not None:
            image, mask = crop_source.get_raw(row)
        elif key not in sources:
            raise ValueError(tr("Cell {cell} has no crop; run ph_crops for its well.", cell=key))
        elif record.get("queue_context_format") == "field_index":
            field_path = os.path.join(crop_root, str(record.get("field_path") or os.path.join(
                "fields", f"site-{key[2]}_DAPI_GFP_flatcorrected_uint16.tif")))
            mask_path = os.path.join(crop_root, str(record.get("mask_path") or os.path.join(
                "masks", f"site-{key[2]}.npz")))
            pair = (field_path, mask_path)
            if pair not in field_cache:
                field_cache[pair] = (read_image(field_path), _ph_read_masks(mask_path))
            image, masks = field_cache[pair]
            if "nuclei" not in masks:
                raise ValueError(tr("The context field needs a nuclei label mask for stage labelling."))
            mask = masks["nuclei"] == key[3]
        elif record.get("crop_format") != "ome-zarr":
            raise ValueError(tr("Legacy OPS shards contain masked crops; provide full-field or unmasked nucleus context crops."))
        else:
            options = meta.get("options", {})
            if options.get("margin_px") is None or bool(options.get("zero_outside", True)):
                raise ValueError(tr("Stage labelling needs context crops. Set crop_margin_px and keep crop_zero_outside off."))
            image, mask = read_crop(crop_root, record, context=True)
            mask = None if mask is None else mask == 2
        image = np.asarray(image)
        if image.ndim != 3 or not all(image.shape):
            raise ValueError(tr("Label queue crops must contain channels, height and width."))
        if mask is None or np.shape(mask) != image.shape[1:] or not np.any(mask):
            raise ValueError(tr("Stage labelling needs the own nucleus mask for every selected cell."))
        values = np.unique(np.asarray(mask)[np.asarray(mask) != 0])
        if len(values) != 1:
            raise ValueError(tr("Return only the selected nucleus mask; keep neighbouring objects in the image pixels."))
        nucleus = np.asarray(mask, dtype=bool)
        if all(name in record and np.isfinite(record[name]) for name in ("cy", "cx")):
            centre = (int(round(float(record["cy"]))) - int(record.get("bbox_y0", 0)),
                      int(round(float(record["cx"]))) - int(record.get("bbox_x0", 0)))
        else:
            ys, xs = np.nonzero(nucleus)
            centre = (int(round(float(ys.mean()))), int(round(float(xs.mean()))))
        top, left = centre[0] - 48, centre[1] - 48
        if top < 0 or left < 0 or top + 96 > image.shape[1] or left + 96 > image.shape[2]:
            excluded.append({**{name: row[name] for name in KEY}, "reason": "window_crosses_context_edge"})
            continue
        if selected is None:
            selected = list(range(min(2, image.shape[0])))
        if max(selected) >= image.shape[0]:
            raise ValueError(tr("A selected label queue channel is absent from a crop; check the crop channel order."))
        if fixed is None:
            if image.dtype != np.dtype(np.uint16):
                raise ValueError(tr("Funk display bounds require raw uint16 images; give explicit ranges for other pixel units."))
            fixed = np.asarray([(550, 23039), (1084, 5002)][:len(selected)], dtype=float)
        if role_ranges and len(selected) == 1:
            fixed = fixed[:1]
        if len(fixed) != len(selected):
            raise ValueError(tr("Give one finite increasing display range for each selected label queue channel."))
        pixel_size = row.get("pixel_size_um", record.get("pixel_size_um", meta.get("pixel_size_um")))
        if pixel_size is None and crop_source is not None:
            pixel_size = getattr(crop_source, "pixel_size_um", None)
            spacing = getattr(crop_source, "spacing", None)
            if pixel_size is None and spacing is not None and spacing.units in ("um", "µm") and "x" in spacing.axes:
                pixel_size = abs(float(spacing.scale[spacing.axes.index("x")]))
        if isinstance(pixel_size, (bool, np.bool_)) or not isinstance(pixel_size, (float, int, np.number)) or not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(tr("A 10 µm scale bar needs measured pixel_size_um or source Spacing in micrometres."))
        window = image[:, top:top + 96, left:left + 96]
        own = nucleus[top:top + 96, left:left + 96]
        visible = ndimage.binary_dilation(own, structure=disk, iterations=1)
        planes = []
        for channel, (lo, hi) in zip(selected, fixed):
            plane = np.clip((window[channel].astype(np.float64) - lo) / max(hi - lo, 1e-9), 0, 1)
            plane = (plane * 255 + 0.5).astype(np.uint8)
            plane[~visible] = 0
            planes.append(plane)
        dna = planes[0]
        tubulin = planes[1] if len(planes) > 1 else None
        zero = np.zeros_like(dna)
        panels = [np.repeat(dna[..., None], 3, axis=2),
                  np.repeat(tubulin[..., None], 3, axis=2) if tubulin is not None
                  else np.full((96, 96, 3), 24, np.uint8),
                  np.stack([zero, tubulin if tubulin is not None else zero, dna], axis=-1)]
        up_mask = np.repeat(np.repeat(own, 3, axis=0), 3, axis=1)
        outline = up_mask & ~ndimage.binary_erosion(up_mask, iterations=1, border_value=0)
        enlarged = []
        for slot, panel in enumerate(panels):
            panel = np.repeat(np.repeat(panel, 3, axis=0), 3, axis=1)
            if slot != 1 or tubulin is not None:
                panel[outline] = (255, 255, 0)
            enlarged.append(panel)
        separator = np.full((288, 4, 3), 90, np.uint8)
        tile = np.concatenate([enlarged[0], separator, enlarged[1], separator, enlarged[2]], axis=1)
        rendered_image = Image.fromarray(tile)
        draw = ImageDraw.Draw(rendered_image)
        if tubulin is None:
            draw.text((302, 138), tr("no tubulin channel"), fill=(150, 150, 150))
        bar = int(round(10 / float(pixel_size) * 3))
        draw.rectangle([864 - bar, 278, 864, 281], fill=(255, 255, 255))
        path = os.path.join(data, uuid.uuid4().hex + ".png")
        rendered_image.save(path)
        paths.append(path)
        included.append(row)
        calibrations.append(float(pixel_size))
    if not included:
        raise ValueError(tr("No selected cells have a complete 96-pixel context window; choose cells away from field edges."))
    rendered = pd.DataFrame(included)
    selection = pd.DataFrame({"plateID": rendered["plate"].to_numpy(),
                              "rowID": rendered["well"].to_numpy(),
                              "columnID": rendered["well"].to_numpy(),
                              "fieldID": rendered["site"].to_numpy(),
                              "objectID": rendered["label"].to_numpy()})
    write_png_list(os.path.join(folder, "measurements", "measurements.db"),
                   png_list_frame(selection, paths))
    write_table(rendered.assign(png_path=paths, image_name=[os.path.basename(p) for p in paths]),
                os.path.join(folder, "queue.csv"), canonicalise=False)
    with open(os.path.join(folder, "queue.json"), "w", encoding="utf-8") as handle:
        json.dump({"classes": {str(i + 1): c for i, c in enumerate(CLASSES)},
                   "labels": str(labels), "cells": len(included), "excluded": excluded,
                   "channels": [int(c) for c in selected], "ranges": fixed.tolist(),
                   "radius": int(radius), "window": 96, "scale": 3, "gamma": 1.0,
                   "pixel_size_um": calibrations, "scale_bar_um": 10,
                   "panels": ["DNA_grey", "tubulin_grey", "DNA_blue_tubulin_green"],
                   "object_mask": "nucleus" if crop_source is None else getattr(crop_source, "object_mask", "source_provided"),
                   "mapping": "queue.csv"}, handle, indent=1)
    return folder


def collect_label_queue(folder: str, *, user: str = "") -> int:
    """Append the job's answers to its labels file (append-only, with user and time).

    :param folder: from :func:`write_label_queue`.
    :param user: who labelled.
    :returns: the rows appended (cells annotated 1 to 7, the stages of :data:`CLASSES` as
        :func:`write_label_queue` numbers them).
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
    from .qt.i18n import tr

    handle = _track_external_job(tr("Cell cycle: label {count} cells", count=int(cells)), None,
                                 blocks_shutdown=False, stop_on_quit=False, kind="external")
    handle.folder = str(folder)
    handle.report(0, int(cells), tr("queue in {folder}", folder=folder))
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
    from .qt.i18n import tr
    from .run_journal import open_run

    try:
        appended = collect_label_queue(handle.folder, user=user)
        with open(os.path.join(handle.folder, "queue.json"), encoding="utf-8") as stream:
            labels = json.load(stream)["labels"]
        handle.report(line=tr("{count} labels appended to {path}", count=appended, path=labels))
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
        write_label_queue(pending, str(queued["crop_root"]), folder, labels=str(queued["labels"]),
                          channels=queued.get("channels"), ranges=queued.get("ranges"),
                          radius=queued.get("radius", 30), crop_source=queued.get("crop_source"))
    key = window.open_module("annotate")
    window._screens[key]._open_source(folder)
