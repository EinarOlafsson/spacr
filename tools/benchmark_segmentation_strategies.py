"""How good is each segmentation strategy spaCR offers? (item 532)

Every strategy is run through spaCR's own Mask generation: each field is
staged as a one-field cellvoyager plate, preprocessed by
``spacr.core.preprocess_generate_masks`` (masks off, so the enhancement chain
and the PSF step run exactly where a Mask run runs them), and the resulting
normalised ``.npz`` batches are segmented by
``spacr.object.generate_cellpose_masks_sam`` -- the function Mask calls for
the ``cell`` object -- with the strategy's model setting written the way the
Model zoo's "Use this model" writes it. The saved ``cell_mask_stack`` masks are
scored against the curated ground truth with ``spacr.scorecard``.

Objects: the Toxoplasma parasitophorous vacuole, as the ``cell`` object on
its own (only) channel. "True" diameter is the median equivalent diameter of
that field's ground-truth objects (the dataset median for a field with none).

Datasets (read from spaCR's example-data cache, never modified):

* ``toxo_pv``: the ten Make Masks test fields (item 412) with
  ``ground_truth_masks/``. They are the in-house PV models' TRAINING data.
* ``training_sample``: the PV training-dataset samples of item 450
  (``mask_datasets/toxoplasma_pv_random2`` and ``toxoplasma_pv``), the fields
  with objects plus two curated negatives, minus any field already in
  ``toxo_pv``. Also PV training data; some fields are round 6's validation
  or test fields, which ``--pv-split`` reports per field.

Timing: ``eval_s`` is the model's own ``eval`` call for one field (one field
per call), measured by wrapping ``CellposeModel.eval`` and the backend
worker's ``_RemoteBackend.eval``; ``run_s`` is the whole
``generate_cellpose_masks_sam`` call (model load, worker start, filtering,
saving) and ``prep_s`` the preprocessing of one field.

Examples::

    CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu tools/run_capped.sh 16G \\
        python tools/benchmark_segmentation_strategies.py --device cpu \\
        --out /tmp/spacr-bench-scratch/cpu --plan models
    tools/gpu_turn.sh bench-segmentation tools/run_capped.sh 16G env \\
        SPACR_DEVICE=cuda python tools/benchmark_segmentation_strategies.py \\
        --device cuda --out /tmp/spacr-bench-scratch/gpu --plan all
    python tools/benchmark_segmentation_strategies.py --report \\
        --out /tmp/spacr-bench-scratch/cpu

Results are written per (dataset, strategy) under ``<out>/results`` and a run
skips any pair already there, so an interrupted pass resumes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EXAMPLE = Path.home() / ".cache/spacr/example_data"
TOXO_PV = EXAMPLE / "make_masks_toxo_pv"
SAMPLES = (EXAMPLE / "mask_datasets/toxoplasma_pv_random2",
           EXAMPLE / "mask_datasets/toxoplasma_pv")
PV_MODELS = Path("/mnt/wd4tb/af3/projects/toxoplasma_pv_model")
PV_SPLIT = PV_MODELS / "metrics/final_r6/split.csv"
MODEL_SEARCH = (Path("/tmp/spacr-525-scratch/models"),)
NEGATIVES_PER_SAMPLE = 2
MAGNIFICATION = 20

CHAINS: Dict[str, Dict[str, object]] = {
    "off": {},
    "bg_clahe": {"enhance_background": "rolling_ball", "enhance_clahe": True},
    "clip_gauss": {"enhance_percentile_clip": True,
                   "enhance_denoise": "gaussian"},
    "log": {"enhance_log": True},
    "tv": {"enhance_denoise": "tv"},
    "deconv": {"psf_operation": "deconvolve", "psf_source": "gaussian",
               "psf_objective": "auto"},
}

PV_KEYS = {"pv_r2": "toxoplasma_pv_v1", "pv_r5": "toxoplasma_pv_v2",
           "pv_r6": "toxoplasma_pv_v3"}
BIOIMAGEIO_KEYS = {
    "dino_vitb": "bioimageio_cellposedino_vit_b_2d_microscopy_instance_segmenter",
    "dino_vitl": "bioimageio_cellposedino_vit_l_2d_microscopy_instance_segmenter",
    "oc1_p66": "bioimageio_oc1_project_66_cellpose",
}


@dataclass(frozen=True)
class Strategy:
    """One way of segmenting, as the Mask settings that express it."""

    key: str
    label: str
    family: str
    model: str
    backend: str = "cellpose"
    chain: str = "off"
    flow: float = 0.4
    diameter: str = "true"


BASE_MODELS = (
    Strategy("cpsam", "Cellpose-SAM cpsam (spaCR default)", "general", "cpsam"),
    Strategy("cpsam_v2", "Cellpose-SAM cpsam_v2", "general", "cpsam_v2"),
    Strategy("cp3_cyto", "Cellpose 3 cyto", "general", "cellpose3:cyto"),
    Strategy("cp3_cyto2", "Cellpose 3 cyto2", "general", "cellpose3:cyto2"),
    Strategy("cp3_cyto3", "Cellpose 3 cyto3", "general", "cellpose3:cyto3"),
    Strategy("cp3_nuclei", "Cellpose 3 nuclei", "general", "cellpose3:nuclei"),
    Strategy("dino_vitb", "Cellpose-DINO ViT-B (bioimage.io)", "general",
             "cellpose_dino:@dino_vitb"),
    Strategy("dino_vitl", "Cellpose-DINO ViT-L (bioimage.io)", "general",
             "cellpose_dino:@dino_vitl"),
    Strategy("oc1_p66", "bioimage.io OC1 Project 66 (Cellpose 3)", "general",
             "cellpose3:@oc1_p66"),
    Strategy("dinocell", "DINOCell", "general", "cpsam", backend="dinocell"),
    Strategy("pv_r2", "Toxoplasma PV v1 (cpsam r2)", "in_distribution",
             "@pv_r2"),
    Strategy("pv_r5", "Toxoplasma PV v2 (cpsam r5)", "in_distribution",
             "@pv_r5"),
    Strategy("pv_r6", "Toxoplasma PV v3 (cpsam r6)", "in_distribution",
             "@pv_r6"),
)

ABLATIONS = (
    ("bg_clahe", dict(chain="bg_clahe"), "background (rolling ball) + CLAHE"),
    ("clip_gauss", dict(chain="clip_gauss"), "percentile clip + Gaussian"),
    ("log", dict(chain="log"), "logarithm"),
    ("tv", dict(chain="tv"), "total-variation denoise"),
    ("deconv", dict(chain="deconv"), "PSF deconvolution, inferred optics"),
    ("flow1", dict(flow=1.0), "flow threshold 1.0"),
    ("flowoff", dict(flow=0.0), "flow threshold off (0)"),
    ("diamauto", dict(diameter="auto"), "diameter blank (auto)"),
    ("diampooled", dict(diameter="pooled"), "diameter = dataset median"),
)


@dataclass
class Field:
    """A ground-truth field and where it came from."""

    dataset: str
    stem: str
    image: str
    truth: str
    shape: List[int]
    n_truth: int
    diameter: Optional[float]
    well: str = ""
    pv_split: str = ""
    stack_name: str = ""
    extra: Dict[str, object] = field(default_factory=dict)


def _gt_diameters(mask: np.ndarray) -> List[float]:
    """Equivalent diameters of a label mask's objects."""
    from skimage.measure import regionprops

    return [float(p.equivalent_diameter_area)
            for p in regionprops(np.asarray(mask).astype(np.int32))]


def _well(index: int) -> str:
    """A unique cellvoyager well name for the index-th field."""
    return f"{'ABCDEFGHIJKLMNOP'[index // 24]}{index % 24 + 1:02d}"


def _pv_split(path: Path) -> Dict[str, str]:
    """``stem -> train/validation/test`` from round 6's split, if present."""
    if not path.is_file():
        return {}
    with path.open() as handle:
        return {row["stem"]: row["set"] for row in csv.DictReader(handle)}


def collect_fields(names: List[str], split: Dict[str, str]) -> List[Field]:
    """The fields of each named dataset, with their truth and diameter."""
    fields: List[Field] = []
    if "toxo_pv" in names:
        for image in sorted(TOXO_PV.glob("*.tif")):
            truth = TOXO_PV / "ground_truth_masks" / image.name
            fields.append(_field("toxo_pv", image, truth, split))
    if "training_sample" in names:
        seen = {f.stem for f in fields} | {p.stem for p in TOXO_PV.glob("*.tif")}
        for folder in SAMPLES:
            negatives = 0
            for image in sorted(folder.glob("*.tif")):
                if image.stem in seen:
                    continue
                item = _field("training_sample", image,
                              folder / "masks" / image.name, split)
                if item.n_truth == 0:
                    if negatives >= NEGATIVES_PER_SAMPLE or folder != SAMPLES[0]:
                        continue
                    negatives += 1
                seen.add(image.stem)
                fields.append(item)
    for dataset in {f.dataset for f in fields}:
        members = [f for f in fields if f.dataset == dataset]
        pooled = []
        for item in members:
            pooled += _gt_diameters(tifffile.imread(item.truth))
        median = float(np.median(pooled)) if pooled else 30.0
        for item in members:
            item.extra["pooled_diameter"] = round(median, 2)
            if item.diameter is None:
                item.diameter = round(median, 2)
    for index, item in enumerate(fields):
        item.well = _well(index)
    return fields


def _field(dataset: str, image: Path, truth: Path,
           split: Dict[str, str]) -> Field:
    """One field's record."""
    mask = tifffile.imread(truth)
    diameters = _gt_diameters(mask)
    return Field(dataset=dataset, stem=image.stem, image=str(image),
                 truth=str(truth), shape=list(mask.shape),
                 n_truth=len(diameters),
                 diameter=(round(float(np.median(diameters)), 2)
                           if diameters else None),
                 pv_split=split.get(image.stem, "not in round 6 split"))


def _sha256(path: Path, cache: Dict[str, str]) -> str:
    """A file's sha256, cached by path, size and mtime."""
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 22), b""):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def resolve_models(out: Path, wanted: List[str],
                   allow_fetch: bool = True) -> Dict[str, Dict[str, str]]:
    """Local, checksum-verified paths for every ``@name`` a strategy uses.

    The in-house PV checkpoints are looked for under ``PV_MODELS`` and must
    match the zoo's published sha256; bioimage.io weights are looked for in
    ``MODEL_SEARCH`` and otherwise fetched by ``spacr.model_zoo.fetch`` from
    their zoo row, which verifies the digest. With ``allow_fetch`` False a
    missing bioimage.io checkpoint is left out, and the strategies needing
    it are skipped rather than holding a GPU turn through a download.
    """
    from spacr import model_zoo as zoo

    cache_path = out / "model_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    resolved: Dict[str, Dict[str, str]] = {}
    remote = {e["key"]: e for e in zoo.BUNDLED_REMOTE_MODELS}
    for name in wanted:
        if name in PV_KEYS:
            entry = remote[PV_KEYS[name]]
            hits = sorted(PV_MODELS.rglob(entry["name"]))
            hits = [p for p in hits if p.is_file()
                    and _sha256(p, cache) == entry["sha256"]]
            if not hits:
                raise FileNotFoundError(
                    f"{entry['name']} with sha256 {entry['sha256'][:12]} not "
                    f"found under {PV_MODELS}")
            resolved[name] = dict(path=str(hits[0]), sha256=entry["sha256"],
                                  zoo_key=entry["key"], source="local, zoo sha256")
        elif name in BIOIMAGEIO_KEYS:
            payload = json.loads(
                (Path.home() / ".spacr/bioimageio_children.json").read_text())
            sizes = json.loads(
                (Path.home() / ".spacr/bioimageio_weight_sizes.json").read_text())
            rows = {r.key: r for r in zoo._bioimageio_rows(payload, sizes)}
            row = rows[BIOIMAGEIO_KEYS[name]]
            dest = out.parent / "models"
            found = None
            for folder in (dest, *MODEL_SEARCH):
                for candidate in sorted(folder.glob(f"{row.name}*")):
                    if (candidate.is_file()
                            and _sha256(candidate, cache) == row.sha256):
                        found = candidate
                        break
                if found:
                    break
            source = "cached, zoo sha256"
            if found is None and not allow_fetch:
                print(f"{name}: {row.name} not downloaded; skipped (--no-fetch)",
                      flush=True)
                continue
            if found is None:
                found = Path(zoo.fetch(row, dest))
                source = "fetched by model_zoo.fetch"
            resolved[name] = dict(path=str(found), sha256=row.sha256,
                                  zoo_key=row.key, source=source)
    out.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return resolved


def model_setting(strategy: Strategy, models: Dict[str, Dict[str, str]]) -> str:
    """The strategy's model setting with ``@name`` replaced by its path."""
    value = strategy.model
    if "@" in value:
        prefix, name = value.split("@", 1)
        value = prefix + models[name]["path"]
    return value


def base_settings(src: Path, extra: Dict[str, object]) -> Dict[str, object]:
    """Mask settings for a one-channel PV plate, the vacuole as the cell."""
    settings: Dict[str, object] = {
        "src": str(src), "channels": [0], "metadata_type": "cellvoyager",
        "custom_regex": None, "magnification": MAGNIFICATION,
        "cell_channel": 0, "nucleus_channel": None, "pathogen_channel": None,
        "organelle_channel": None, "cellpose3_add_nucleus_channel": False,
        "plot": False, "test_mode": False, "save": True, "verbose": False,
        "n_jobs": 2, "batch_size": 1, "randomize": False,
        "adjust_cells": False, "seg_qc": "off", "keep_intermediate": True,
        "timelapse": False, "resume": False,
    }
    settings.update(extra)
    return settings


def stage_and_preprocess(item: Field, chain: str, root: Path) -> Dict[str, object]:
    """Preprocess one field as its own plate; returns its npz and timing."""
    from spacr.core import preprocess_generate_masks
    from spacr.qt.synthetic import cellvoyager_filename

    plate = root / "prep" / chain / item.dataset / item.stem
    done = plate / "prep.json"
    if done.is_file():
        return json.loads(done.read_text())
    if plate.exists():
        shutil.rmtree(plate)
    plate.mkdir(parents=True)
    image = tifffile.imread(item.image)
    name = cellvoyager_filename(plate="plate1", well=item.well, field=1, chan=0)
    tifffile.imwrite(plate / name, image)
    settings = base_settings(plate, dict(CHAINS[chain], masks=False,
                                         cell_diameter=item.diameter))
    start = time.perf_counter()
    with _quiet():
        preprocess_generate_masks(settings)
    seconds = time.perf_counter() - start
    archives = sorted((plate / "masks").glob("*.npz"))
    if len(archives) != 1:
        raise RuntimeError(f"{plate}: expected one npz, found {len(archives)}")
    with np.load(archives[0]) as data:
        stack_name = str(data["filenames"][0])
        shape = list(data["data"].shape)
    record = dict(npz=str(archives[0]), stack_name=stack_name,
                  npz_shape=shape, prep_s=round(seconds, 3))
    psf_record = plate / "psf" / "segmentation_application.json"
    if psf_record.is_file():
        record["psf"] = json.loads(psf_record.read_text())
    done.write_text(json.dumps(record, indent=1))
    return record


@contextlib.contextmanager
def _quiet():
    """Send spaCR's progress prints to a log instead of the terminal."""
    log = open(os.environ.get("BENCH_LOG", os.devnull), "a")
    with contextlib.redirect_stdout(log):
        try:
            yield
        finally:
            log.close()


@contextlib.contextmanager
def eval_timer(calls: List[float]):
    """Time every outermost model ``eval`` call made inside the block.

    Cellpose's ``eval`` on a list calls itself once per image, so only the
    outermost call is recorded; nested calls would count the field twice.
    """
    from cellpose import models as cp_models
    import spacr._segmentation_backends as backends

    originals = [(cp_models.CellposeModel, "eval"),
                 (backends._RemoteBackend, "eval")]
    saved = [getattr(owner, name) for owner, name in originals]
    depth = [0]

    def wrap(inner):
        def timed(self, *args, **kwargs):
            depth[0] += 1
            start = time.perf_counter()
            try:
                return inner(self, *args, **kwargs)
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    calls.append(time.perf_counter() - start)
        return timed

    for (owner, name), inner in zip(originals, saved):
        setattr(owner, name, wrap(inner))
    try:
        yield
    finally:
        for (owner, name), inner in zip(originals, saved):
            setattr(owner, name, inner)


def score_field(truth: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    """Counts at IoU 0.5 and 0.75 and the mean matched IoU (at 0.5)."""
    from spacr.scorecard import match_objects

    out: Dict[str, object] = {}
    for threshold, tag in ((0.5, "50"), (0.75, "75")):
        match = match_objects(truth, pred, threshold)
        out[f"tp{tag}"] = match.true_positives
        out[f"fp{tag}"] = match.false_positives
        out[f"fn{tag}"] = match.false_negatives
        out[f"f1_{tag}"] = round(match.f1, 4)
        out[f"precision{tag}"] = round(match.precision, 4)
        out[f"recall{tag}"] = round(match.recall, 4)
        if tag == "50":
            out["iou_sum"] = float(np.sum(match.ious))
            out["iou_mean"] = round(float(np.mean(match.ious)), 4) if match.ious else 0.0
            out["n_truth"] = match.n_truth
            out["n_pred"] = match.n_pred
    return out


def pooled(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Pooled counts and rates over fields, plus the per-field mean F1."""
    out: Dict[str, object] = {"fields": len(rows)}
    for tag in ("50", "75"):
        tp = sum(r[f"tp{tag}"] for r in rows)
        fp = sum(r[f"fp{tag}"] for r in rows)
        fn = sum(r[f"fn{tag}"] for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.update({f"tp{tag}": tp, f"fp{tag}": fp, f"fn{tag}": fn,
                    f"precision{tag}": round(precision, 4),
                    f"recall{tag}": round(recall, 4), f"f1_{tag}": round(f1, 4)})
    positives = [r for r in rows if r["n_truth"] > 0]
    out["mean_field_f1_50"] = (round(float(np.mean([r["f1_50"] for r in positives])), 4)
                               if positives else 0.0)
    out["n_truth"] = sum(r["n_truth"] for r in rows)
    out["n_pred"] = sum(r["n_pred"] for r in rows)
    out["negative_field_fp"] = sum(r["n_pred"] for r in rows if r["n_truth"] == 0)
    out["iou_mean"] = round(sum(r["iou_sum"] for r in rows) / out["tp50"], 4) if out["tp50"] else 0.0
    evals = [r["eval_s"] for r in rows if r.get("eval_s") is not None]
    out["eval_s_median"] = round(float(np.median(evals)), 2) if evals else None
    out["eval_s_mean"] = round(float(np.mean(evals)), 2) if evals else None
    out["prep_s_mean"] = round(float(np.mean([r["prep_s"] for r in rows])), 2)
    return out


def run_strategy(strategy: Strategy, fields: List[Field], dataset: str,
                 root: Path, models: Dict[str, Dict[str, str]],
                 device: str) -> Dict[str, object]:
    """Segment one dataset with one strategy through Mask generation."""
    from spacr.object import generate_cellpose_masks_sam
    import spacr._segmentation_backends as backends

    members = [f for f in fields if f.dataset == dataset]
    run = root / "runs" / dataset / strategy.key
    if run.exists():
        shutil.rmtree(run)
    masks = run / "masks"
    masks.mkdir(parents=True)
    prep = {}
    for item in members:
        prep[item.stem] = stage_and_preprocess(item, strategy.chain, root)
        shutil.copy(prep[item.stem]["npz"], masks / f"{item.well}_{item.stem[:60]}.npz")
    by_stack = {prep[f.stem]["stack_name"]: f for f in members}

    model_value = model_setting(strategy, models)
    rows: List[Dict[str, object]] = []
    total_run = 0.0
    failure = None
    eval_calls: List[float] = []
    per_field_run: Dict[str, float] = {}
    order = sorted(masks.glob("*.npz"))
    for archive in order:
        with np.load(archive) as data:
            stack_name = str(data["filenames"][0])
        item = by_stack[stack_name]
        if strategy.diameter == "true":
            diameter = item.diameter
        elif strategy.diameter == "pooled":
            diameter = item.extra["pooled_diameter"]
        else:
            diameter = None
        extra = dict(CHAINS[strategy.chain], cell_model_name=model_value,
                     segmentation_backend=strategy.backend,
                     cell_diameter=diameter, cell_flow_threshold=strategy.flow)
        settings = base_settings(masks, extra)
        before = len(eval_calls)
        start = time.perf_counter()
        try:
            with eval_timer(eval_calls), _quiet():
                generate_cellpose_masks_sam(str(masks), settings, "cell",
                                            batch_paths=[str(archive)],
                                            run_qc=False)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            print(f"  {strategy.key} {item.stem}: FAILED {failure}", flush=True)
            break
        per_field_run[item.stem] = time.perf_counter() - start
        total_run += per_field_run[item.stem]
        field_evals = eval_calls[before:]
        saved = masks / "cell_mask_stack" / stack_name
        pred = np.load(saved)
        truth = tifffile.imread(item.truth)
        if pred.shape != truth.shape:
            raise RuntimeError(f"{item.stem}: mask {pred.shape} vs truth {truth.shape}")
        row = dict(field=item.stem, dataset=dataset, pv_split=item.pv_split,
                   diameter=diameter, eval_s=round(sum(field_evals), 3) if field_evals else None,
                   run_s=round(per_field_run[item.stem], 3),
                   prep_s=prep[item.stem]["prep_s"])
        row.update(score_field(truth.astype(np.int64), pred.astype(np.int64)))
        rows.append(row)
        print(f"  {strategy.key:>22} {item.stem[:40]:<40} gt {row['n_truth']:>4} "
              f"pred {row['n_pred']:>4} F1@.5 {row['f1_50']:.3f} "
              f"eval {row['eval_s']}s run {row['run_s']:.1f}s", flush=True)
    backends._shutdown_workers()
    result = dict(strategy=asdict(strategy), model_setting=model_value,
                  dataset=dataset, device=device, rows=rows,
                  pooled=pooled(rows) if rows else None, failure=failure,
                  run_s_total=round(total_run, 2),
                  first_field_run_s=round(per_field_run[rows[0]["field"]], 2) if rows else None)
    if strategy.chain == "deconv":
        result["psf"] = {stem: p.get("psf") for stem, p in prep.items()}
    shutil.rmtree(run, ignore_errors=True)
    return result


def best_general(results_dir: Path, dataset: str) -> Optional[str]:
    """The general model with the highest pooled F1 at IoU 0.5."""
    best, best_f1 = None, -1.0
    general = {s.key for s in BASE_MODELS if s.family == "general"}
    for path in results_dir.glob(f"{dataset}__*.json"):
        result = json.loads(path.read_text())
        key = result["strategy"]["key"]
        if key in general and result.get("pooled"):
            if result["pooled"]["f1_50"] > best_f1:
                best, best_f1 = key, result["pooled"]["f1_50"]
    return best


def ablations_for(key: str) -> List[Strategy]:
    """The enhancement, PSF, flow and diameter variants of one model."""
    base = next(s for s in BASE_MODELS if s.key == key)
    return [replace(base, key=f"{key}+{suffix}", label=f"{base.label}, {label}",
                    family="ablation", **change)
            for suffix, change, label in ABLATIONS]


def machine() -> Dict[str, object]:
    """Where the numbers were measured."""
    info: Dict[str, object] = dict(host=platform.node(), cpus=os.cpu_count(),
                                   thread_cap=os.environ.get("OMP_NUM_THREADS"),
                                   python=platform.python_version(),
                                   load_average=os.getloadavg())
    try:
        import torch
        import cellpose
        info.update(torch=torch.__version__, cellpose=getattr(cellpose, "version", ""),
                    cuda_available=torch.cuda.is_available(),
                    gpu=(torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None))
    except Exception as error:
        info["torch_error"] = str(error)
    return info


def fmt(value, digits=3):
    """A table cell."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report(out: Path) -> str:
    """Text tables of every result under ``out``."""
    results = [json.loads(p.read_text()) for p in sorted((out / "results").glob("*.json"))]
    lines: List[str] = []
    for dataset in sorted({r["dataset"] for r in results}):
        chosen = [r for r in results if r["dataset"] == dataset]
        chosen.sort(key=lambda r: -(r["pooled"] or {}).get("f1_50", -1))
        lines.append(f"== {dataset} (pooled over fields)")
        lines.append(f"{'strategy':<28}{'family':<16}{'obj':>6}{'GT':>6}"
                     f"{'P@.5':>7}{'R@.5':>7}{'F1@.5':>7}{'F1@.75':>8}"
                     f"{'mIoU':>7}{'fldF1':>7}{'eval s':>8}{'prep s':>8}")
        for r in chosen:
            p = r["pooled"]
            if not p:
                lines.append(f"{r['strategy']['key']:<28}FAILED: {r['failure']}")
                continue
            lines.append(
                f"{r['strategy']['key']:<28}{r['strategy']['family']:<16}"
                f"{p['n_pred']:>6}{p['n_truth']:>6}{fmt(p['precision50']):>7}"
                f"{fmt(p['recall50']):>7}{fmt(p['f1_50']):>7}{fmt(p['f1_75']):>8}"
                f"{fmt(p['iou_mean']):>7}{fmt(p['mean_field_f1_50']):>7}"
                f"{fmt(p['eval_s_median'], 1):>8}{fmt(p['prep_s_mean'], 1):>8}"
                + (f"  (partial: {r['failure']})" if r.get("failure") else ""))
        lines.append("")
        fields = sorted({row["field"] for r in chosen for row in r["rows"]})
        keys = [r["strategy"]["key"] for r in chosen if r["rows"]]
        lines.append(f"-- {dataset}: F1 at IoU 0.5 per field")
        lines.append(f"{'field':<42}" + "".join(f"{k[:11]:>12}" for k in keys))
        for stem in fields:
            cells = []
            for r in chosen:
                if not r["rows"]:
                    continue
                row = next((x for x in r["rows"] if x["field"] == stem), None)
                if row is None:
                    cells.append(f"{'-':>12}")
                elif row["n_truth"] == 0:
                    cells.append(f"{'FP ' + str(row['n_pred']):>12}")
                else:
                    cells.append(f"{row['f1_50']:>12.3f}")
            lines.append(f"{stem[:41]:<42}" + "".join(cells))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Run the benchmark, or print its tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("/tmp/spacr-bench-scratch/cpu"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--datasets", default="toxo_pv,training_sample")
    parser.add_argument("--plan", choices=("models", "ablations", "all"), default="all")
    parser.add_argument("--strategies", default="",
                        help="comma-separated strategy keys to run (default: the plan's)")
    parser.add_argument("--ablation-model", default="auto",
                        help="model key the ablations vary; auto = the best general "
                             "model on toxo_pv so far")
    parser.add_argument("--ablation-datasets", default="toxo_pv")
    parser.add_argument("--fields", default="",
                        help="comma-separated field stems to limit the run to (probing)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="never download weights; skip strategies whose "
                             "weights are not already here")
    parser.add_argument("--threads", type=int, default=0,
                        help="cap CPU threads (OMP/MKL and torch) in this process "
                             "and the backend workers it starts; 0 = no cap")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out
    if args.report:
        print(report(out))
        return 0
    os.environ["SPACR_DEVICE"] = args.device
    if args.threads > 0:
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = str(args.threads)
        import torch
        torch.set_num_threads(args.threads)
    os.environ.setdefault("BENCH_LOG", str(out / "spacr_output.log"))
    out.mkdir(parents=True, exist_ok=True)
    results_dir = out / "results"
    results_dir.mkdir(exist_ok=True)

    datasets = [d for d in args.datasets.split(",") if d]
    split = _pv_split(PV_SPLIT)
    fields = collect_fields(datasets, split)
    if args.fields:
        keep = set(args.fields.split(","))
        fields = [f for f in fields if f.stem in keep]
    (out / "fields.json").write_text(json.dumps([asdict(f) for f in fields], indent=1))
    print(f"{len(fields)} fields: "
          + ", ".join(f"{d} {sum(f.dataset == d for f in fields)}" for d in datasets),
          flush=True)

    only = [s for s in args.strategies.split(",") if s]
    jobs: List[tuple] = []
    if args.plan in ("models", "all"):
        for dataset in datasets:
            for strategy in BASE_MODELS:
                if not only or strategy.key in only:
                    jobs.append((strategy, dataset))
    wanted = {s.model.split("@", 1)[1] for s, _ in jobs if "@" in s.model}
    models = resolve_models(out, sorted(wanted | set(PV_KEYS)),
                            allow_fetch=not args.no_fetch)
    (out / "models.json").write_text(json.dumps(models, indent=1))

    def execute(strategy: Strategy, dataset: str) -> None:
        target = results_dir / f"{dataset}__{strategy.key}.json"
        needs = strategy.model.split("@", 1)[1] if "@" in strategy.model else None
        if needs and needs not in models:
            print(f"skip {dataset} {strategy.key} (no weights for {needs})", flush=True)
            return
        if target.is_file():
            print(f"skip {dataset} {strategy.key} (done)", flush=True)
            return
        print(f"run {dataset} {strategy.key} on {args.device}", flush=True)
        start = time.perf_counter()
        result = run_strategy(strategy, fields, dataset, out, models, args.device)
        result["wall_s"] = round(time.perf_counter() - start, 1)
        result["machine"] = machine()
        target.write_text(json.dumps(result, indent=1))
        p = result["pooled"] or {}
        print(f"done {dataset} {strategy.key}: F1@.5 {p.get('f1_50')} "
              f"F1@.75 {p.get('f1_75')} in {result['wall_s']} s", flush=True)

    for strategy, dataset in jobs:
        execute(strategy, dataset)

    if args.plan in ("ablations", "all"):
        key = (best_general(results_dir, "toxo_pv")
               if args.ablation_model == "auto" else args.ablation_model)
        if key is None:
            print("no general model result on toxo_pv yet; ablations skipped")
        else:
            print(f"ablations on {key}", flush=True)
            more = {s.model.split("@", 1)[1] for s in ablations_for(key) if "@" in s.model}
            models.update(resolve_models(out, sorted(more),
                                         allow_fetch=not args.no_fetch))
            for dataset in [d for d in args.ablation_datasets.split(",") if d]:
                for strategy in ablations_for(key):
                    if not only or strategy.key in only:
                        execute(strategy, dataset)

    if args.receipt:
        results = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
        args.receipt.write_text(json.dumps(dict(fields=[asdict(f) for f in fields],
                                                models=models, results=results),
                                           indent=1))
    print(report(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
