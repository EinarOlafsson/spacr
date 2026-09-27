"""Does Noise2Void denoising raise segmentation F1 on low light? (item 557)

The ten toxo_pv fields of item 532 (curated ground truth), a 512 x 512 crop of
each (the window with the most whole ground-truth objects), photon-starved by
simulation: the signal above the field's median scaled so its 99.9th
percentile is PEAK photons, plus BG photons, Poisson noise, Gaussian read
noise READ, offset 100. N2V is trained once on the ten noisy crops (no clean
image is ever shown to it), then every strategy runs through spaCR's own Mask
generation with the 532 driver (``stage_and_preprocess`` + ``run_strategy``)
and is scored with its ``score_field``/``pooled``.

Environment: OUT (results folder), PEAK (photons at the 99.9th percentile,
default 10), EPOCHS (default 30), CHAINS (default off,n2v,clip_gauss; also
clip and n2v_clip), DEVICE (default cpu), and SPACR_BACKENDS_DIR for a
CAREamics environment outside ~/.spacr/backends. Examples::

    CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu tools/run_capped.sh 16G \\
        python tools/validate_n2v_lowlight.py
    tools/gpu_turn.sh n2v-lowlight tools/run_capped.sh 16G env DEVICE=cuda \\
        SPACR_DEVICE=cuda EPOCHS=200 python tools/validate_n2v_lowlight.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

WT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WT), str(WT / "tools")]
OUT = Path(os.environ.get("OUT", "/tmp/spacr-557-validate"))

os.environ["BENCH_LOG"] = str(OUT / "bench.log")

import benchmark_segmentation_strategies as B  # noqa: E402
import spacr._segmentation_backends as SB  # noqa: E402

PEAK, BG, READ, SEED = float(os.environ.get("PEAK", 10)), 2.0, 1.5, 0
CROP = 512
EPOCHS = int(os.environ.get("EPOCHS", 30))
CHAINS = os.environ.get("CHAINS", "off,n2v,clip_gauss").split(",")
DEVICE = os.environ.get("DEVICE", "cpu")


def best_window(gt):
    best, where = -1, (0, 0)
    for y in range(0, gt.shape[0] - CROP + 1, 128):
        for x in range(0, gt.shape[1] - CROP + 1, 128):
            win = gt[y:y + CROP, x:x + CROP]
            edge = np.unique(np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]]))
            whole = len(set(np.unique(win)) - set(edge) - {0})
            if whole > best:
                best, where = whole, (y, x)
    return where


def make_lowlight():
    rng = np.random.default_rng(SEED)
    folder = OUT / f"lowlight_p{PEAK:g}"
    (folder / "gt").mkdir(parents=True, exist_ok=True)
    pairs = []
    for image in sorted(B.TOXO_PV.glob("*.tif")):
        gt = tifffile.imread(B.TOXO_PV / "ground_truth_masks" / image.name)
        y, x = best_window(gt)
        raw = tifffile.imread(image).astype(np.float64)[y:y + CROP, x:x + CROP]
        signal = np.clip(raw - np.median(raw), 0, None)
        scale = PEAK / max(np.percentile(signal, 99.9), 1e-9)
        photons = rng.poisson(signal * scale + BG).astype(np.float64)
        noisy = np.clip(np.round(photons + rng.normal(0, READ, photons.shape) + 100), 0, 65535)
        tifffile.imwrite(folder / image.name, noisy.astype(np.uint16))
        tifffile.imwrite(folder / "gt" / image.name, gt[y:y + CROP, x:x + CROP])
        pairs.append((folder / image.name, folder / "gt" / image.name))
    return pairs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = make_lowlight()
    fields = []
    for index, (image, truth) in enumerate(pairs):
        item = B._field("lowlight", image, truth, {})
        item.well = B._well(index)
        fields.append(item)
    diameters = [f.diameter for f in fields if f.diameter]
    for f in fields:
        f.diameter = f.diameter or float(np.median(diameters))

    models = OUT / f"n2v_p{PEAK:g}_e{EPOCHS}"
    ckpt = models / "channel_0.ckpt"
    if not ckpt.is_file():
        planes = [tifffile.imread(p).astype(np.float32) for p, _ in pairs]
        record = SB._n2v_train(planes, ckpt, epochs=EPOCHS, device=DEVICE)
        record.update(fields=[p.stem for p, _ in pairs], channel=0)
        (models / "channel_0.json").write_text(json.dumps(record, indent=1))
        print("trained", {k: record[k] for k in ("epochs", "seconds", "train_loss", "val_loss")}, flush=True)
    B.CHAINS["n2v"] = {"n2v_denoise": True, "n2v_model": str(models)}
    B.CHAINS["n2v_clip"] = dict(B.CHAINS["n2v"], enhance_percentile_clip=True)
    B.CHAINS["clip"] = {"enhance_percentile_clip": True}

    results = {}
    for chain in CHAINS:
        strategy = B.Strategy(f"cpsam_{chain}", f"Cellpose-SAM, chain {chain}",
                              "general", "cpsam", chain=chain)
        root = OUT / f"run_p{PEAK:g}_e{EPOCHS}"
        start = time.perf_counter()
        result = B.run_strategy(strategy, fields, "lowlight", root, {}, DEVICE)
        result["wall_s"] = round(time.perf_counter() - start, 1)
        results[chain] = result
        print(chain, json.dumps(result["pooled"]), flush=True)
    prov = OUT / f"run_p{PEAK:g}_e{EPOCHS}" / "prep" / "n2v"
    records = sorted(prov.glob("lowlight/*/psf/segmentation_application.json"))
    if records:
        results["n2v_provenance_example"] = json.loads(records[0].read_text())["configuration"]
    path = OUT / f"results_p{PEAK:g}_e{EPOCHS}.json"
    if path.is_file():
        results = {**json.loads(path.read_text()), **results}
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
