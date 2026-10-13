"""Check which segmentation-uncertainty scores find real errors, held out.

Reads the crops ``tools/segment_uncertainty_crops.py`` cached (the ten Make
Masks Toxo PV test fields around their hand-drawn vacuoles; the r5 model
under four test-time transforms with its cell probability and flows; stock
Cellpose-SAM, the r2 fine-tune and Cellpose 3 cyto3 once each) and scores
every crop and every r5 object with each candidate reading:

* ``tta_*``: the four r5 passes through
  :func:`spacr.active_learning._segmentation_uncertainty`;
* ``ens_*``: r5 against the three other models, the same function;
* ``all_*``: the four r5 passes and the three other models together;
* ``prob_*``: the r5 cell probability (mean of the four passes), pixels
  within a margin of Cellpose's threshold away from the drawn outlines
  (``band``), and sub-threshold pixels away from anything an r5 pass drew
  (``nearmiss``);
* ``flow_*``: Cellpose's own flow error of each r5 object;
* ``shipped``: what Make Masks now computes from the r5 passes alone,
  :func:`spacr.active_learning._segmentation_uncertainty` given the passes'
  cell probability and the reference flows (its ``field``, ``area`` plus
  ``missed``; its ``objects``). It was defined after the held-out choice
  below settled on disagreement area plus near misses, so its numbers are
  post hoc, not held out, and it is kept out of every choice and fit;
* ``*_mean`` field scores (one minus mean panoptic quality of the other
  passes) against ``*_area`` ones (the fraction of foreground whose pixel
  uncertainty is at least 0.5).

Error of a crop is one minus the panoptic quality of r5 against the truth; an
object is wrong when no truth vacuole matches it at IoU 0.5.

Held out: the ten fields split into two halves of five (alternate fields in
name order), all of a field's crops on one side. A score is chosen, and a
non-negative least-squares combination of scores fitted, and the pair of
scores whose rank mean orders the crops best picked, on one half and
measured on the other, both ways; the pooled out-of-fold combination is
reported beside each fixed score on every half. Usage::

    python tools/validate_segmentation_uncertainty.py \
        --cache /mnt/wd4tb/scratch/568/cache --out result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spacr.active_learning import _segmentation_uncertainty
from spacr.scorecard import _auroc, iou_matrix, match_objects

MARGIN = 2.0
NEARMISS = -2.0
FLOW_BAD = 0.2
RING = 2
FIELD_FEATURES = ("tta_mean", "tta_area", "ens_mean", "ens_area",
                  "all_mean", "all_area", "prob_band", "prob_nearmiss",
                  "flow_mean", "flow_area", "union_area", "shipped")
COMBO = ("tta_area", "ens_area", "prob_band", "prob_nearmiss", "flow_area")
OBJECT_FEATURES = ("tta", "ens", "flow", "prob_margin", "shipped")


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    rho = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _pq(truth: np.ndarray, pred: np.ndarray) -> float:
    match = match_objects(truth, pred, threshold=0.5)
    tp = len(match.pairs)
    denominator = tp + 0.5 * ((match.n_truth - tp) + (match.n_pred - tp))
    return sum(match.ious) / denominator if denominator else 1.0


def _flow_errors(labels: np.ndarray, flow: np.ndarray) -> dict:
    """Cellpose's flow error of every object, by label."""
    from cellpose.dynamics import flow_error
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if not ids.size:
        return {}
    relabelled = np.searchsorted(np.concatenate([[0], ids]), labels)
    errors, _ = flow_error(relabelled.astype(np.int32), flow)
    return {int(i): float(e) for i, e in zip(ids, errors)}


def _crop(stored: Path):
    """Every field and object reading of one cached crop."""
    data = np.load(stored)
    cyto3 = np.load(stored.with_name(stored.stem + ".cyto3.npy"))
    truth = data["truth"]
    tta = [a.astype(np.int32) for a in data["tta_labels"]]
    reference = tta[0]
    models = [reference, data["cpsam_labels"].astype(np.int32),
              data["r2_labels"].astype(np.int32), cyto3[0].astype(np.int32)]
    u_tta = _segmentation_uncertainty(tta)
    shipped = _segmentation_uncertainty(tta, probabilities=list(
        data["tta_prob"]), vectors=data["flow"])
    u_ens = _segmentation_uncertainty(models)
    u_all = _segmentation_uncertainty(tta + models[1:])
    logit = data["tta_prob"].mean(axis=0)
    union = np.any([a > 0 for a in tta + models], axis=0)
    fore = max(int(union.sum()), 1)
    from scipy import ndimage as ndi
    from skimage.segmentation import find_boundaries
    edge = ndi.binary_dilation(find_boundaries(reference, mode="thick"),
                               iterations=RING)
    band = (np.abs(logit) < MARGIN) & ~edge
    drawn = np.any([a > 0 for a in tta], axis=0)
    nearmiss = (logit > NEARMISS) & ~ndi.binary_dilation(drawn,
                                                         iterations=RING)
    flows = _flow_errors(reference, data["flow"])
    flow_bad = np.isin(reference, [k for k, v in flows.items()
                                   if v > FLOW_BAD])
    uncertain = ((u_tta["map"] >= 0.5) | (u_ens["map"] >= 0.5) | band
                 | nearmiss | flow_bad)
    field = {
        "tta_mean": u_tta["spread"],
        "tta_area": float((u_tta["map"] >= 0.5).sum() / fore),
        "ens_mean": u_ens["spread"],
        "ens_area": float((u_ens["map"] >= 0.5).sum() / fore),
        "all_mean": u_all["spread"],
        "all_area": float((u_all["map"] >= 0.5).sum() / fore),
        "prob_band": float(band.sum() / fore),
        "prob_nearmiss": float(nearmiss.sum() / fore),
        "flow_mean": float(np.mean(list(flows.values()))) if flows else 0.0,
        "flow_area": float(flow_bad.sum() / fore),
        "union_area": float(uncertain.sum() / fore),
        "shipped": shipped["field"],
    }
    ious, _t, p_ids = iou_matrix(truth, reference)
    best = ious.max(axis=0) if ious.size else np.zeros(p_ids.size)
    best_of = dict(zip(p_ids.tolist(), best.tolist()))
    match = match_objects(truth, reference, threshold=0.5)
    matched = {int(p_ids[c]) for _r, c in match.pairs}
    objects = []
    for label, value in u_tta["objects"].items():
        inside = reference == label
        objects.append({
            "label": label, "tta": value, "ens": u_ens["objects"][label],
            "flow": flows.get(label, 0.0),
            "shipped": shipped["objects"][label],
            "prob_margin": float(-np.abs(logit[inside]).mean()),
            "wrong": label not in matched,
            "best_truth_iou": best_of.get(label, 0.0)})
    return {"crop": stored.stem, "field": stored.stem.rsplit("_", 3)[0],
            "error": 1.0 - _pq(truth, reference),
            "n_truth": int(match.n_truth), "n_pred": int(match.n_pred),
            "n_missed": int(match.n_truth - len(match.pairs)),
            "scores": field, "objects": objects}


def _queue(score, error) -> dict:
    """Spearman, and how many of the worst quarter the top quarter holds."""
    score, error = np.asarray(score, float), np.asarray(error, float)
    k = max(1, len(score) // 4)
    top = set(np.argsort(-score, kind="stable")[:k].tolist())
    worst = set(np.argsort(-error, kind="stable")[:k].tolist())
    return {"spearman": round(_spearman(score, error), 3), "n": len(score),
            "k": k, "worst_k_in_top_k": len(top & worst),
            "chance": round(k * k / len(score), 2)}


def _rank_mean(columns: np.ndarray) -> np.ndarray:
    """The mean of each column's ranks, so no score's scale dominates."""
    from scipy.stats import rankdata
    return np.mean([rankdata(c) for c in columns.T], axis=0)


def _best_pair(x_train, y_train):
    """The two scores whose rank mean best orders the training crops."""
    from itertools import combinations
    pairs = list(combinations(range(x_train.shape[1] - 1), 2))
    rho = [_spearman(_rank_mean(x_train[:, list(p)]), y_train)
           for p in pairs]
    return list(pairs[int(np.nanargmax(rho))]), float(np.nanmax(rho))


def _field_bootstrap(score, error, field, draws=2000, seed=0):
    """A 95 % interval of Spearman, resampling whole fields."""
    rng = np.random.default_rng(seed)
    names = sorted(set(field))
    field = np.asarray(field)
    rhos = []
    for _ in range(draws):
        pick = rng.choice(names, len(names))
        idx = np.concatenate([np.flatnonzero(field == f) for f in pick])
        rho = _spearman(score[idx], error[idx])
        if np.isfinite(rho):
            rhos.append(rho)
    return [round(float(v), 3) for v in np.percentile(rhos, [2.5, 97.5])]


def _standardise(train: np.ndarray, test: np.ndarray):
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def _combo_fit(x_train, y_train, x_test):
    """Least squares on standardised scores, the weights non-negative."""
    from scipy.optimize import nnls
    a, b = _standardise(x_train, x_test)
    weights, _ = nnls(a, y_train - y_train.mean())
    return b @ weights, weights


def _logistic_fit(x_train, y_train, x_test):
    from sklearn.linear_model import LogisticRegression
    a, b = _standardise(x_train, x_test)
    model = LogisticRegression(class_weight="balanced", C=1.0)
    model.fit(a, y_train)
    return model.decision_function(b), model.coef_[0]


def main() -> None:
    """Score, split, fit and write the result JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [_crop(p) for p in sorted(Path(args.cache).glob("*.npz"))]
    fields = sorted({r["field"] for r in rows})
    halves = [fields[0::2], fields[1::2]]
    error = np.array([r["error"] for r in rows])
    x = np.array([[r["scores"][f] for f in FIELD_FEATURES] for r in rows])
    combo_idx = [FIELD_FEATURES.index(f) for f in COMBO]
    objects = [dict(o, field=r["field"], crop=r["crop"])
               for r in rows for o in r["objects"]]
    ox = np.array([[o[f] for f in OBJECT_FEATURES] for o in objects])
    wrong = np.array([o["wrong"] for o in objects], dtype=int)
    best = np.array([o["best_truth_iou"] for o in objects])
    field_of = [r["field"] for r in rows]

    folds = []
    oof = np.zeros(len(rows))
    oof_pair = np.zeros(len(rows))
    oof_obj = np.zeros(len(objects))
    for side in (0, 1):
        train_f, test_f = halves[side], halves[1 - side]
        tr = np.array([r["field"] in train_f for r in rows])
        te = ~tr
        on_train = {f: _spearman(x[tr, i], error[tr])
                    for i, f in enumerate(FIELD_FEATURES[:-1])}
        chosen = max(on_train, key=lambda f: on_train[f])
        pred, weights = _combo_fit(x[tr][:, combo_idx], error[tr],
                                   x[te][:, combo_idx])
        oof[te] = pred
        pair, pair_train = _best_pair(x[tr], error[tr])
        pair_pred = _rank_mean(x[te][:, pair])
        oof_pair[te] = pair_pred / te.sum()
        otr = np.array([o["field"] in train_f for o in objects])
        ote = ~otr
        opred, ocoef = _logistic_fit(ox[otr][:, :-1], wrong[otr],
                                     ox[ote][:, :-1])
        oof_obj[ote] = opred
        folds.append({
            "train_fields": train_f, "test_fields": test_f,
            "chosen_on_train": chosen,
            "chosen_train_spearman": round(on_train[chosen], 3),
            "chosen_on_test": _queue(x[te, FIELD_FEATURES.index(chosen)],
                                     error[te]),
            "combo_weights": dict(zip(COMBO, np.round(weights, 3).tolist())),
            "combo_on_test": _queue(pred, error[te]),
            "pair_chosen_on_train": [FIELD_FEATURES[i] for i in pair],
            "pair_train_spearman": round(pair_train, 3),
            "pair_on_test": _queue(pair_pred, error[te]),
            "every_score_on_test": {
                f: _queue(x[te, i], error[te])
                for i, f in enumerate(FIELD_FEATURES)},
            "object_logistic_coef": dict(zip(OBJECT_FEATURES[:-1],
                                             np.round(ocoef, 3).tolist())),
            "object_auroc_on_test": {
                "logistic": round(float(_auroc(wrong[ote], opred)), 3),
                **{f: round(float(_auroc(wrong[ote], ox[ote, i])), 3)
                   for i, f in enumerate(OBJECT_FEATURES)}},
            "objects_test": int(ote.sum()),
            "objects_wrong_test": int(wrong[ote].sum()),
        })

    summary = {
        "crops": len(rows), "fields": len(fields),
        "objects": len(objects), "objects_wrong": int(wrong.sum()),
        "truth_objects": int(sum(r["n_truth"] for r in rows)),
        "truth_missed_by_r5": int(sum(r["n_missed"] for r in rows)),
        "heldout_combo_pooled": _queue(oof, error),
        "heldout_pair_pooled": _queue(oof_pair, error),
        "field_bootstrap_95": {
            "tta_mean": _field_bootstrap(
                x[:, FIELD_FEATURES.index("tta_mean")], error, field_of),
            "heldout_combo": _field_bootstrap(oof, error, field_of),
            "heldout_pair": _field_bootstrap(oof_pair, error, field_of)},
        "heldout_object_logistic_auroc_pooled":
            round(float(_auroc(wrong, oof_obj)), 3),
        "heldout_object_logistic_spearman_vs_1_minus_best_iou":
            round(_spearman(oof_obj, 1 - best), 3),
        "all_crops_every_score": {f: _queue(x[:, i], error)
                                  for i, f in enumerate(FIELD_FEATURES)},
        "all_objects_auroc": {
            f: round(float(_auroc(wrong, ox[:, i])), 3)
            for i, f in enumerate(OBJECT_FEATURES)},
    }
    out = {"item": 568, "device": "cpu", "reference": "cpsam_v2_toxo_r5",
           "ensemble": ["cpsam", "cpsam_v2_toxo_r2", "cellpose3:cyto3"],
           "margin_logit": MARGIN, "nearmiss_logit": NEARMISS,
           "flow_bad": FLOW_BAD, "summary": summary, "folds": folds,
           "crops_detail": [{k: v for k, v in r.items() if k != "objects"}
                            for r in rows]}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summary, "folds": [
        {k: f[k] for k in ("chosen_on_train", "chosen_on_test",
                           "combo_weights", "combo_on_test",
                           "pair_chosen_on_train", "pair_on_test",
                           "object_auroc_on_test")} for f in folds]},
        indent=1))


if __name__ == "__main__":
    main()
