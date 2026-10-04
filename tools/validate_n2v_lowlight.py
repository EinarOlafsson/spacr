"""Does Noise2Void denoising raise segmentation F1 on low light? (item 557)

Two data sources (DATASET):

``toxo_pv`` (default)
    The ten toxo_pv fields of item 532 (curated ground truth), a CROP x CROP
    crop of each (the window with the most whole ground-truth objects; a
    CROP at least the field size keeps whole fields), photon-starved by
    simulation: the signal above the field's median scaled so its 99.9th
    percentile is PEAK photons, plus BG photons, Poisson noise, Gaussian read
    noise READ, offset 100. NOISE=correlated adds camera-row noise on top: a
    unit Gaussian smoothed along x by a SPAN-pixel box and scaled to CORR
    photons (the pixel-to-pixel correlation plain N2V learns as signal).
``huh7``
    A genuinely low-light acquisition: the Cell Tracking Challenge
    Fluo-C2DL-Huh7 widefield movies (8-bit, background ~4 +- 4 grey levels,
    cells ~40), both sequences, every frame with a gold-truth SEG mask
    (13 frames). N2V learns from all 60 raw frames.

N2V is trained once per data set on the noisy images only (no clean image
is ever shown to it); VARIANT=struct trains structN2V (N2V2 with a
horizontal blind-spot bar STRUCT_SPAN pixels wide) instead of N2V2.
Every strategy then runs through spaCR's own Mask generation with the 532
driver (``stage_and_preprocess`` + ``run_strategy``) and is scored with its
``score_field``/``pooled``. With FLOWS=1 the Cellpose-SAM flows and cell
probability of every field are kept under OUT/flows/<tag>/<chain>/ so the
thresholds can be tuned afterwards without the GPU.

``--sweep OUT/flows/<tag>`` (SWEEP_DEVICE, default cpu; grid SWEEP_CELLPROB and SWEEP_FLOW) re-runs Cellpose's own mask step on the
kept flows over a grid of cell-probability threshold, flow threshold and
minimum object area (the Mask settings cell_cellprob_threshold,
cell_flow_threshold and the object-filter area floor), picks the best
setting per chain on one half of the fields and scores it on the other half
(two folds: alternate fields, or the two Huh7 sequences), and writes
sweep.json there: pooled held-out F1 per chain, default and tuned.

Environment: OUT (results folder), DATASET, PEAK (default 10), EPOCHS
(default 30), CROP (default 512), NOISE (shot or correlated), CORR (default
3), SPAN (default 5), VARIANT (n2v2 or struct), STRUCT_SPAN (odd, default 9),
CHAINS (default off,n2v,clip_gauss; also clip and n2v_clip), DEVICE
(default cpu), FLOWS, and SPACR_BACKENDS_DIR for a CAREamics environment
outside ~/.spacr/backends. Examples::

    CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu tools/run_capped.sh 16G \\
        python tools/validate_n2v_lowlight.py
    tools/gpu_turn.sh n2v-lowlight tools/run_capped.sh 16G env DEVICE=cuda \\
        SPACR_DEVICE=cuda EPOCHS=200 python tools/validate_n2v_lowlight.py
    CUDA_VISIBLE_DEVICES='' python tools/validate_n2v_lowlight.py \\
        --sweep OUT/flows/p10_e150
"""
import itertools
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
CROP = int(os.environ.get("CROP", 512))
EPOCHS = int(os.environ.get("EPOCHS", 30))
CHAINS = os.environ.get("CHAINS", "off,n2v,clip_gauss").split(",")
DEVICE = os.environ.get("DEVICE", "cpu")
DATASET = os.environ.get("DATASET", "toxo_pv")
NOISE = os.environ.get("NOISE", "shot")
CORR = float(os.environ.get("CORR", 3.0))
SPAN = int(os.environ.get("SPAN", 5))
VARIANT = os.environ.get("VARIANT", "n2v2")
STRUCT_SPAN = int(os.environ.get("STRUCT_SPAN", 9))
FLOWS = os.environ.get("FLOWS", "0") == "1"
PREDENOISE = os.environ.get("PREDENOISE", "")
CP3_PYTHON = os.environ.get("CP3_PYTHON", str(Path.home() / ".spacr/backends/cellpose3/bin/python"))
HUH7 = Path("/nas_mnt/data/training_data/projects/live_cell/datasets/Homo_sapiens/"
            "Masks/timelapse/Widefield_20x/Fluorescence/ctc_fluo_c2dl_huh7/"
            "unpacked/Fluo-C2DL-Huh7/Fluo-C2DL-Huh7")

CELLPROB = tuple(float(v) for v in os.environ.get("SWEEP_CELLPROB", "-3,-2,-1,0,1,2").split(","))
FLOW = tuple(float(v) for v in os.environ.get("SWEEP_FLOW", "0.4,0.6,0.8,1.0").split(","))
AREA_FRACTION = (0.0, 0.1, 0.2, 0.35, 0.5)


def data_tag():
    """The folder suffix that tells the data sets apart."""
    if DATASET == "huh7":
        return "huh7"
    tag = f"p{PEAK:g}"
    if NOISE == "correlated":
        tag += f"_corr{CORR:g}s{SPAN}"
    return tag


def model_tag():
    """The data tag plus the N2V variant and epochs."""
    variant = f"_struct{STRUCT_SPAN}" if VARIANT == "struct" else ""
    pre = f"_{PREDENOISE}" if PREDENOISE else ""
    return f"{data_tag()}{pre}{variant}_e{EPOCHS}"


def best_window(gt):
    """The CROP window holding the most whole ground-truth objects."""
    if CROP >= max(gt.shape):
        return 0, 0
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
    """Simulated low-light toxo_pv fields and their ground truth."""
    rng = np.random.default_rng(SEED)
    folder = OUT / f"lowlight_{data_tag()}"
    (folder / "gt").mkdir(parents=True, exist_ok=True)
    pairs = []
    for image in sorted(B.TOXO_PV.glob("*.tif")):
        gt = tifffile.imread(B.TOXO_PV / "ground_truth_masks" / image.name)
        y, x = best_window(gt)
        raw = tifffile.imread(image).astype(np.float64)[y:y + CROP, x:x + CROP]
        signal = np.clip(raw - np.median(raw), 0, None)
        scale = PEAK / max(np.percentile(signal, 99.9), 1e-9)
        photons = rng.poisson(signal * scale + BG).astype(np.float64)
        noisy = photons + rng.normal(0, READ, photons.shape)
        if NOISE == "correlated":
            from scipy.ndimage import uniform_filter1d
            rows = uniform_filter1d(rng.normal(0, 1, photons.shape), SPAN, axis=1)
            noisy += rows * (CORR / rows.std())
        noisy = np.clip(np.round(noisy + 100), 0, 65535)
        tifffile.imwrite(folder / image.name, noisy.astype(np.uint16))
        tifffile.imwrite(folder / "gt" / image.name, gt[y:y + CROP, x:x + CROP])
        pairs.append((folder / image.name, folder / "gt" / image.name))
    return pairs, None


def stage_huh7():
    """The Huh7 frames with gold-truth masks, and every raw frame for N2V."""
    folder = OUT / "huh7"
    (folder / "gt").mkdir(parents=True, exist_ok=True)
    pairs, planes = [], []
    for sequence in ("01", "02"):
        frames = sorted((HUH7 / sequence).glob("t*.tif"))
        planes += [tifffile.imread(f).astype(np.float32) for f in frames]
        for seg in sorted((HUH7 / f"{sequence}_GT" / "SEG").glob("man_seg*.tif")):
            frame = seg.stem.replace("man_seg", "")
            name = f"s{sequence}_t{frame}.tif"
            image = tifffile.imread(HUH7 / sequence / f"t{frame}.tif")
            tifffile.imwrite(folder / name, image.astype(np.uint16))
            tifffile.imwrite(folder / "gt" / name, tifffile.imread(seg))
            pairs.append((folder / name, folder / "gt" / name))
    return pairs, planes


_CP3_SCRIPT = """
import sys, json, numpy as np, tifffile, torch
from cellpose import denoise
torch.set_num_threads(int(sys.argv[3]))
model = denoise.DenoiseModel(model_type=sys.argv[1], gpu=False)
for src, dst, diameter in json.loads(open(sys.argv[2]).read()):
    image = tifffile.imread(src).astype(np.float32)
    out = np.asarray(model.eval(image, channels=None, channel_axis=None, diameter=diameter,
                                normalize=True, batch_size=8), dtype=np.float32)
    np.save(dst, out.reshape(image.shape))
    print(dst, flush=True)
"""


def _to_uint16(plane):
    """A float plane mapped linearly onto the full uint16 range."""
    low, high = float(plane.min()), float(plane.max())
    return np.round((plane - low) / max(high - low, 1e-12) * 65535).astype(np.uint16)


def predenoise(pairs, diameters):
    """Denoise every noisy image once, before Mask generation sees it.

    ``nlm``: scikit-image non-local means (fast mode, 5 px patches, 6 px
    search, h = 0.8 sigma, sigma from ``estimate_sigma``) on the unit-scaled
    plane. ``cp3_<model>``: Cellpose 3's restoration network (for example
    denoise_cyto3, denoise_nuclei) run in the Cellpose 3 environment
    (CP3_PYTHON) on the CPU at the field's true diameter. The result is
    rescaled linearly onto uint16 and replaces the field image; the ground
    truth is untouched.
    """
    import subprocess

    folder = OUT / f"pre_{data_tag()}_{PREDENOISE}"
    (folder / "gt").mkdir(parents=True, exist_ok=True)
    out = [(folder / image.name, gt) for image, gt in pairs]
    todo = [(src, dst, d) for (src, _), (dst, _), d in zip(pairs, out, diameters)
            if not dst.is_file()]
    if PREDENOISE == "nlm":
        from skimage.restoration import denoise_nl_means, estimate_sigma
        for src, dst, _ in todo:
            raw = tifffile.imread(src).astype(np.float64)
            unit = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-12)
            sigma = float(estimate_sigma(unit))
            plane = denoise_nl_means(unit, h=0.8 * sigma, sigma=sigma, fast_mode=True,
                                     patch_size=5, patch_distance=6)
            tifffile.imwrite(dst, _to_uint16(plane))
    elif PREDENOISE.startswith("cp3_"):
        jobs = [(str(src), str(dst.with_suffix(".npy")), float(d)) for src, dst, d in todo]
        if jobs:
            spec = folder / "jobs.json"
            spec.write_text(json.dumps(jobs))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       CELLPOSE_LOCAL_MODELS_PATH=str(Path(CP3_PYTHON).parents[1] / "models"))
            subprocess.run([CP3_PYTHON, "-c", _CP3_SCRIPT, PREDENOISE[4:], str(spec),
                            os.environ.get("CP3_THREADS", "8")], check=True, env=env)
            for _, dst, _ in todo:
                tifffile.imwrite(dst, _to_uint16(np.load(dst.with_suffix(".npy"))))
                dst.with_suffix(".npy").unlink()
    else:
        raise ValueError(f"unknown PREDENOISE {PREDENOISE!r}")
    return out


class FlowKeeper:
    """Keeps the flows Cellpose computes masks from, one entry per call."""

    def __init__(self):
        """Start with nothing kept."""
        self.kept = []

    def __enter__(self):
        """Wrap Cellpose's mask step."""
        from cellpose import models

        self.original = models.CellposeModel._compute_masks
        keeper = self

        def compute(model, shape, dP, cellprob, **kwargs):
            keeper.kept.append(dict(shape=list(shape), dP=np.asarray(dP)[:, 0],
                                    cellprob=np.asarray(cellprob)[0], **{
                                        k: kwargs.get(k) for k in (
                                            "niter", "min_size", "max_size_fraction")}))
            return keeper.original(model, shape, dP, cellprob, **kwargs)

        models.CellposeModel._compute_masks = compute
        return self

    def __exit__(self, *exc):
        """Put Cellpose's mask step back."""
        from cellpose import models

        models.CellposeModel._compute_masks = self.original


def save_flows(keeper, result, fields, chain):
    """Write one chain's kept flows next to the truth they are scored on."""
    rows = result["rows"]
    if len(keeper.kept) != len(rows):
        raise RuntimeError(f"{chain}: {len(keeper.kept)} flow sets for {len(rows)} fields")
    by_stem = {f.stem: f for f in fields}
    folder = OUT / "flows" / model_tag() / chain
    folder.mkdir(parents=True, exist_ok=True)
    for kept, row in zip(keeper.kept, rows):
        item = by_stem[row["field"]]
        np.savez_compressed(
            folder / f"{item.stem}.npz", dP=kept["dP"].astype(np.float16),
            cellprob=kept["cellprob"].astype(np.float16),
            shape=np.array(kept["shape"]), niter=kept["niter"],
            min_size=kept["min_size"], max_size_fraction=kept["max_size_fraction"],
            diameter=item.diameter, truth=item.truth, f1_50=row["f1_50"])


def main():
    """Train N2V on the noisy images, then segment and score every chain."""
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, planes = stage_huh7() if DATASET == "huh7" else make_lowlight()
    fields = []
    for index, (image, truth) in enumerate(pairs):
        item = B._field("lowlight", image, truth, {})
        item.well = B._well(index)
        fields.append(item)
    diameters = [f.diameter for f in fields if f.diameter]
    for f in fields:
        f.diameter = f.diameter or float(np.median(diameters))

    if PREDENOISE:
        pairs = predenoise(pairs, [f.diameter for f in fields])
        fields = []
        for index, (image, truth) in enumerate(pairs):
            item = B._field("lowlight", image, truth, {})
            item.well = B._well(index)
            fields.append(item)
        for f in fields:
            f.diameter = f.diameter or float(np.median(diameters))

    models = OUT / f"n2v_{model_tag()}"
    ckpt = models / "channel_0.ckpt"
    if not ckpt.is_file() and any(c.startswith("n2v") for c in CHAINS):
        planes = planes or [tifffile.imread(p).astype(np.float32) for p, _ in pairs]
        struct = dict(struct_axes="horizontal", struct_span=STRUCT_SPAN) if VARIANT == "struct" else {}
        record = SB._n2v_train(planes, ckpt, epochs=EPOCHS, device=DEVICE, **struct)
        record.update(fields=[p.stem for p, _ in pairs], channel=0)
        (models / "channel_0.json").write_text(json.dumps(record, indent=1))
        print("trained", {k: record[k] for k in ("method", "epochs", "seconds", "train_loss", "val_loss")},
              flush=True)
    n2v = "n2v_struct" if VARIANT == "struct" else "n2v"
    B.CHAINS[n2v] = {"n2v_denoise": True, "n2v_model": str(models)}
    B.CHAINS["n2v_clip"] = dict(B.CHAINS[n2v], enhance_percentile_clip=True)
    B.CHAINS["clip"] = {"enhance_percentile_clip": True}
    chains = [n2v if c == "n2v" else c for c in CHAINS]

    results = {}
    root = OUT / f"run_{model_tag()}"
    for chain in chains:
        strategy = B.Strategy(f"cpsam_{chain}", f"Cellpose-SAM, chain {chain}",
                              "general", "cpsam", chain=chain)
        start = time.perf_counter()
        keeper = FlowKeeper()
        if FLOWS:
            with keeper:
                result = B.run_strategy(strategy, fields, "lowlight", root, {}, DEVICE)
            save_flows(keeper, result, fields, chain)
        else:
            result = B.run_strategy(strategy, fields, "lowlight", root, {}, DEVICE)
        result["wall_s"] = round(time.perf_counter() - start, 1)
        results[chain] = result
        print(chain, json.dumps(result["pooled"]), flush=True)
    prov = root / "prep" / n2v
    records = sorted(prov.glob("lowlight/*/psf/segmentation_application.json"))
    if records:
        results["n2v_provenance_example"] = json.loads(records[0].read_text())["configuration"]
    path = OUT / f"results_{model_tag()}.json"
    if path.is_file():
        results = {**json.loads(path.read_text()), **results}
    path.write_text(json.dumps(results, indent=1))


def _masks_at(kept, cellprob_threshold, flow_threshold, device):
    """Cellpose's own mask step on kept flows at other thresholds."""
    from cellpose import dynamics

    shape = [int(v) for v in kept["shape"]]
    cellprob = kept["cellprob"].astype(np.float32)
    resize = None if list(cellprob.shape) == shape[1:3] else shape[1:3]
    return dynamics.resize_and_compute_masks(
        kept["dP"].astype(np.float32), cellprob, niter=int(kept["niter"]),
        cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
        resize=resize, min_size=int(kept["min_size"]),
        max_size_fraction=float(kept["max_size_fraction"]), device=device)


def _counts(truth, n_truth, masks, areas_at_least):
    """IoU 0.5 and 0.75 counts, as ``spacr.scorecard.match_objects`` finds
    them (one-to-one, maximum total IoU), for each minimum object area."""
    from scipy.optimize import linear_sum_assignment

    masks = np.asarray(masks, dtype=np.int64)
    n_pred = int(masks.max())
    table = np.bincount(truth.ravel() * (n_pred + 1) + masks.ravel(),
                        minlength=(n_truth + 1) * (n_pred + 1)).reshape(n_truth + 1, n_pred + 1)
    inter = table[1:, 1:].astype(np.float64)
    t_area, p_area = table[1:].sum(1), table[:, 1:].sum(0)
    present = p_area > 0
    union = t_area[:, None] + p_area[None, :] - inter
    ious = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    out = []
    for area in areas_at_least:
        cols = present & (p_area >= area)
        sub = ious[:, cols]
        kept = int(cols.sum())
        row = {"n_truth": n_truth}
        if sub.size:
            r, c = linear_sum_assignment(-sub)
            matched = sub[r, c]
        else:
            matched = np.zeros(0)
        for tag, threshold in (("50", 0.5), ("75", 0.75)):
            tp = int((matched >= threshold).sum())
            row.update({f"tp{tag}": tp, f"fp{tag}": kept - tp, f"fn{tag}": n_truth - tp})
        out.append(row)
    return out


def _score_grid(path, device=None):
    """Counts at IoU 0.5 and 0.75 for every grid point of one field's flows."""
    import torch

    torch.set_num_threads(int(os.environ.get("SWEEP_THREADS", 2)))
    device = device or torch.device("cpu")
    kept = dict(np.load(path, allow_pickle=False))
    truth = B._read_plane(Path(str(kept["truth"])))
    _, truth = np.unique(truth, return_inverse=True)
    truth = truth.reshape(kept["cellprob"].shape).astype(np.int64)
    n_truth = int(truth.max())
    area = np.pi * float(kept["diameter"]) ** 2 / 4
    grid = {}
    for cp, ft in itertools.product(CELLPROB, FLOW):
        masks = _masks_at(kept, cp, ft, device)
        rows = _counts(truth, n_truth, masks, [f * area for f in AREA_FRACTION])
        for fraction, row in zip(AREA_FRACTION, rows):
            grid[f"{cp:g}|{ft:g}|{fraction:g}"] = row
    return path.stem, grid


def _fold_of(stem, index):
    """Two folds: the Huh7 sequence, or alternate fields."""
    if stem.startswith("s01"):
        return 0
    if stem.startswith("s02"):
        return 1
    return index % 2


def _f1(rows, tag="50"):
    """Pooled F1, precision, recall and counts over rows of counts."""
    tp = sum(r[f"tp{tag}"] for r in rows)
    fp = sum(r[f"fp{tag}"] for r in rows)
    fn = sum(r[f"fn{tag}"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
                precision=round(p, 4), recall=round(r, 4), tp=tp, fp=fp, fn=fn)


def sweep(folder):
    """Tune thresholds per chain on one fold, score them on the other."""
    from concurrent.futures import ProcessPoolExecutor

    folder = Path(folder)
    report = {"grid": {"cellprob_threshold": CELLPROB, "flow_threshold": FLOW,
                       "min_area_fraction_of_disc": AREA_FRACTION}}
    default = "0|0.4|0"
    workers = int(os.environ.get("SWEEP_WORKERS", 4))
    device = os.environ.get("SWEEP_DEVICE", "cpu")
    for chain_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        paths = sorted(chain_dir.glob("*.npz"))
        if device == "cpu":
            with ProcessPoolExecutor(workers) as pool:
                grids = dict(pool.map(_score_grid, paths))
        else:
            import torch
            grids = dict(_score_grid(path, torch.device(device)) for path in paths)
        stems = sorted(grids)
        folds = {s: _fold_of(s, i) for i, s in enumerate(stems)}
        held_out, chosen = [], {}
        for fold in (0, 1):
            tune = [grids[s] for s in stems if folds[s] == fold]
            best = max(grids[stems[0]],
                       key=lambda k: (_f1([g[k] for g in tune])["f1"], k == default))
            chosen[f"tuned_on_fold_{fold}"] = best
            held_out += [grids[s][best] for s in stems if folds[s] != fold]
        everything = [grids[s] for s in stems]
        oracle = max(grids[stems[0]], key=lambda k: _f1([g[k] for g in everything])["f1"])
        report[chain_dir.name] = {
            "fields": len(stems),
            "default": _f1([g[default] for g in everything]),
            "default_75": _f1([g[default] for g in everything], "75"),
            "tuned_held_out": _f1(held_out),
            "tuned_held_out_75": _f1(held_out, "75"),
            "chosen": chosen,
            "in_sample_best": {"setting": oracle, **_f1([g[oracle] for g in everything])},
        }
        print(chain_dir.name, json.dumps({k: report[chain_dir.name][k] for k in (
            "default", "tuned_held_out", "chosen")}), flush=True)
    (folder / "sweep.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--sweep":
        sweep(sys.argv[2])
    else:
        main()
