"""Segment the Toxo PV test crops once per model and pass, for item 568.

Stage one of ``tools/validate_segmentation_uncertainty.py``: every crop is
segmented by the fine-tuned r5 model under the four test-time transforms
(labels and Cellpose cell probability, both mapped back onto the crop, and
the identity pass's flows), and once each by stock Cellpose-SAM and the
earlier r2 fine-tune. Cellpose 3 cyto3 lives in its own environment, so it
is run by ``--cyto3`` under that environment's interpreter, reading the
crops this stage saved. One ``.npz`` per crop in ``--cache``.

Usage::

    CUDA_VISIBLE_DEVICES='' tools/run_capped.sh 8G python \
        tools/segment_uncertainty_crops.py --cache /mnt/wd4tb/scratch/568/c
    ~/.spacr/backends/cellpose3/bin/python \
        tools/segment_uncertainty_crops.py --cyto3 --cache ...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

TOXO_PV = Path.home() / ".cache/spacr/example_data/make_masks_toxo_pv"
MODELS = Path("/mnt/wd4tb/af3/projects/toxoplasma_pv_model/models")
TRANSFORMS = ("identity", "flip_lr", "flip_ud", "rot90")


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


def _forward(image, name):
    if name == "flip_lr":
        return np.ascontiguousarray(image[:, ::-1])
    if name == "flip_ud":
        return np.ascontiguousarray(image[::-1, :])
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(image, 1))
    return image


def _inverse(array, name):
    if name == "flip_lr":
        return np.ascontiguousarray(array[..., :, ::-1])
    if name == "flip_ud":
        return np.ascontiguousarray(array[..., ::-1, :])
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(array, -1, axes=(-2, -1)))
    return array


def _run_cyto3(cache: Path, threads: int) -> None:
    """Segment every saved crop with Cellpose 3 cyto3 (its own env)."""
    import torch
    from cellpose import models
    torch.set_num_threads(threads)
    model = models.Cellpose(gpu=False, model_type="cyto3")
    for stored in sorted(cache.glob("*.npz")):
        out = stored.with_name(stored.stem + ".cyto3.npy")
        if out.is_file():
            continue
        image = np.load(stored)["image"]
        masks, flows, _s, diam = model.eval(image, diameter=None,
                                            channels=[0, 0])
        np.save(out, np.stack([masks.astype(np.float32),
                               flows[2].astype(np.float32)]))
        print(f"{stored.stem}: cyto3 diameter {float(diam):.1f}", flush=True)


def main() -> None:
    """Segment and cache."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--crops", type=int, default=6)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cyto3", action="store_true")
    parser.add_argument("--shard", default="0/1",
                        help="i/n: segment only every n-th field from i")
    args = parser.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if args.cyto3:
        _run_cyto3(cache, args.threads)
        return

    import tifffile
    import torch
    from cellpose import models
    torch.set_num_threads(args.threads)
    loaded = {}

    def model(name):
        if name not in loaded:
            path = "cpsam" if name == "cpsam" else str(MODELS / name)
            loaded[name] = models.CellposeModel(
                gpu=False, use_bfloat16=False, pretrained_model=path)
        return loaded[name]

    def run(name, image):
        masks, flows, _s = model(name).eval([image], batch_size=4,
                                            normalize=True)
        return (masks[0].astype(np.int32), flows[0][2].astype(np.float32),
                flows[0][1].astype(np.float32))

    shard, shards = (int(v) for v in args.shard.split("/"))
    for image_path in sorted(TOXO_PV.glob("*.tif"))[shard::shards]:
        truth_full = tifffile.imread(TOXO_PV / "ground_truth_masks"
                                     / image_path.name)
        image_full = tifffile.imread(image_path)
        for y, x in _crops(truth_full, args.size, args.crops):
            key = f"{image_path.stem}_{y}_{x}_{args.size}"
            stored = cache / f"{key}.npz"
            if stored.is_file():
                continue
            t0 = time.time()
            image = image_full[y:y + args.size, x:x + args.size]
            truth = truth_full[y:y + args.size, x:x + args.size]
            tta_labels, tta_prob = [], []
            flow = None
            for name in TRANSFORMS:
                labels, prob, dp = run("cpsam_v2_toxo_r5",
                                       _forward(image, name))
                tta_labels.append(_inverse(labels, name))
                tta_prob.append(_inverse(prob, name))
                if name == "identity":
                    flow = dp
            cpsam = run("cpsam", image)
            r2 = run("cpsam_v2_toxo_r2", image)
            np.savez_compressed(
                stored, image=image, truth=truth,
                tta_labels=np.stack(tta_labels), tta_prob=np.stack(tta_prob),
                flow=flow, cpsam_labels=cpsam[0], cpsam_prob=cpsam[1],
                r2_labels=r2[0], r2_prob=r2[1])
            print(f"{key}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
