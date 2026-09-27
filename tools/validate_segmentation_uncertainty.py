"""Check that test-time-augmentation uncertainty finds real segmentation errors.

Crops the ten Make Masks Toxo PV test fields around their hand-drawn
vacuoles, segments every crop under the four test-time transforms of
:func:`spacr.active_learning._tta_label_sets` with a Cellpose model on the
CPU, and asks two questions of
:func:`spacr.active_learning._segmentation_uncertainty`:

* per object: is a predicted vacuole with no ground-truth match (IoU 0.5)
  more uncertain than one with a match (AUROC), and does uncertainty rise as
  the best ground-truth IoU falls (Spearman)?
* per crop: does field uncertainty rise with the crop's real error (one
  minus panoptic quality against the truth), and does ordering the crops
  most-uncertain-first put the worst ones at the top of the queue?

The Cellpose passes are cached in ``--cache`` so the scoring can be rerun
for free. Usage::

    CUDA_VISIBLE_DEVICES='' tools/run_capped.sh 8G python \
        tools/validate_segmentation_uncertainty.py --out result.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

from spacr.active_learning import (_TTA_TRANSFORMS, _segmentation_uncertainty,
                                   _tta_label_sets)
from spacr.scorecard import _auroc, iou_matrix, match_objects

TOXO_PV = Path.home() / ".cache/spacr/example_data/make_masks_toxo_pv"
MODEL = Path("/mnt/wd4tb/af3/projects/toxoplasma_pv_model/models/"
             "cpsam_v2_toxo_r5")


def _crops(truth: np.ndarray, size: int, count: int, stride: int = 32):
    """The ``count`` non-overlapping windows holding the most truth objects."""
    height, width = truth.shape
    scored = []
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            ids = np.unique(truth[y:y + size, x:x + size])
            scored.append((int((ids != 0).sum()), y, x))
    scored.sort(reverse=True)
    chosen = []
    for n, y, x in scored:
        if len(chosen) == count or n == 0:
            break
        if all(abs(y - cy) >= size or abs(x - cx) >= size
               for _n, cy, cx in chosen):
            chosen.append((n, y, x))
    return [(y, x) for _n, y, x in chosen]


def _spearman(a, b) -> float:
    """Spearman's rho, ranks by argsort (ties are rare here)."""
    from scipy.stats import spearmanr
    rho = spearmanr(np.asarray(a), np.asarray(b)).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _pq(truth: np.ndarray, pred: np.ndarray) -> float:
    """Panoptic quality of ``pred`` against ``truth`` at IoU 0.5."""
    match = match_objects(truth, pred, threshold=0.5)
    tp = len(match.pairs)
    denominator = tp + 0.5 * ((match.n_truth - tp) + (match.n_pred - tp))
    return sum(match.ious) / denominator if denominator else 1.0


def main() -> None:
    """Segment, score and write the result JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--crops", type=int, default=3)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--cache", default="/mnt/firecuda2/codex/"
                        "scratch-docs/568/cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    model = None
    rows, objects = [], []
    started = time.time()
    for image_path in sorted(TOXO_PV.glob("*.tif")):
        truth_full = tifffile.imread(TOXO_PV / "ground_truth_masks"
                                     / image_path.name)
        image_full = tifffile.imread(image_path)
        for y, x in _crops(truth_full, args.size, args.crops):
            key = f"{image_path.stem}_{y}_{x}_{args.size}"
            stored = cache / f"{key}.npz"
            image = image_full[y:y + args.size, x:x + args.size]
            truth = truth_full[y:y + args.size, x:x + args.size]
            if stored.is_file():
                sets = list(np.load(stored)["sets"])
            else:
                if model is None:
                    import torch
                    from cellpose import models
                    torch.set_num_threads(args.threads)
                    model = models.CellposeModel(
                        gpu=False, use_bfloat16=False,
                        pretrained_model=args.model)
                t0 = time.time()
                sets = _tta_label_sets(
                    image, lambda a: model.eval([a], batch_size=4,
                                                normalize=True)[0][0])
                np.savez_compressed(stored, sets=np.stack(sets))
                print(f"{key}: {time.time() - t0:.0f} s", flush=True)
            result = _segmentation_uncertainty(sets)
            reference = sets[0]
            ious, _t, p_ids = iou_matrix(truth, reference)
            best = (ious.max(axis=0) if ious.size
                    else np.zeros(p_ids.size))
            best_of = dict(zip(p_ids.tolist(), best.tolist()))
            match = match_objects(truth, reference, threshold=0.5)
            matched = {int(p_ids[c]) for _r, c in match.pairs}
            for label, value in result["objects"].items():
                objects.append({"crop": key, "label": label,
                                "uncertainty": value,
                                "wrong": label not in matched,
                                "best_truth_iou": best_of.get(label, 0.0)})
            rows.append({"crop": key, "field": image_path.stem,
                         "uncertainty": result["field"],
                         "error": 1.0 - _pq(truth, reference),
                         "n_truth": int(match.n_truth),
                         "n_pred": int(match.n_pred),
                         "f1": match.f1 if hasattr(match, "f1") else None})

    u_obj = np.array([o["uncertainty"] for o in objects])
    wrong = np.array([o["wrong"] for o in objects], dtype=int)
    best = np.array([o["best_truth_iou"] for o in objects])
    u_crop = np.array([r["uncertainty"] for r in rows])
    error = np.array([r["error"] for r in rows])
    order = np.argsort(-u_crop, kind="stable")
    k = max(1, len(rows) // 4)
    worst = set(np.argsort(-error, kind="stable")[:k].tolist())
    found = len(worst & set(order[:k].tolist()))
    summary = {
        "objects": int(len(objects)),
        "objects_wrong_at_iou_0.5": int(wrong.sum()),
        "object_auroc_wrong": float(_auroc(wrong, u_obj)),
        "object_spearman_uncertainty_vs_1_minus_best_truth_iou":
            _spearman(u_obj, 1.0 - best),
        "object_mean_uncertainty_wrong": float(u_obj[wrong == 1].mean())
        if wrong.any() else None,
        "object_mean_uncertainty_right": float(u_obj[wrong == 0].mean())
        if (wrong == 0).any() else None,
        "crops": len(rows),
        "crop_spearman_uncertainty_vs_error": _spearman(u_crop, error),
        "queue_top_quarter_k": k,
        "queue_worst_quarter_found_in_top_quarter": found,
        "queue_expected_by_chance": round(k * k / len(rows), 2),
        "queue_mean_error_top_quarter": float(error[order[:k]].mean()),
        "queue_mean_error_all": float(error.mean()),
        "queue_mean_error_bottom_quarter": float(error[order[-k:]].mean()),
    }
    out = {"item": 568, "model": args.model, "device": "cpu",
           "transforms": list(_TTA_TRANSFORMS), "crop_size": args.size,
           "crops_per_field": args.crops, "summary": summary,
           "wall_s": round(time.time() - started, 1), "crops_detail": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
