#!/usr/bin/env python3
"""Does an encoder trained on cells retrieve gene identity better than ImageNet?

The measurement 386 recorded and 395 asks to repeat with a microscopy encoder,
rebuilt as a committed harness (2026-09-14) because the 2026-09-12 run was
never committed. The maintainer's ruling on the old numbers was "rerun from
scratch", so every feature set -- the measured panel, the ImageNet baseline
and each candidate -- is measured here on the SAME objects and the SAME fold
indices, which are written to disk once and reused.

THE DESIGN, one pipeline for every feature set:

  objects   every crop in ``png_list`` that joins one-to-one to a ``cell`` row
            and decodes; paths carry a dead acquisition prefix and are
            rewritten onto ``--plate``. A zero-byte PNG exists on plate1, so
            existence is not accepted as readability.
  contrasts gene      c1 (SAG1) vs c2 (GRA14), the screen's positive contrast
            position  c5 vs c6, two pooled-library columns with no systematic
                      biology; must read chance for any encoder
            pairs     all 20 adjacent library-column pairs c4-c5 .. c23-c24,
                      the null DISTRIBUTION the gene contrast is read against
            nulls     labels shuffled across objects, and across wells, on
                      the gene and position contrasts
  folds     5-fold StratifiedGroupKFold grouped by well, fixed seed, computed
            once per contrast from the true labels and saved under
            ``<out>/folds``; a well is asserted never to sit in both halves
  metric    kNN (k=15), StandardScaler fit on the training half only:
            accuracy, balanced accuracy, recall per class
  inputs    one global per-channel 99th-percentile scale, computed once from
            a fixed random sample of crops and applied to every crop before
            any encoder sees it (``EmbeddingSpec.normalize`` is OFF). The
            module's own ``normalize=True`` takes the percentile over whatever
            stack it is handed, so an object's embedding would depend on the
            other crops in its batch -- and batches drawn from a sorted index
            are one well at a time.

FEATURE SETS (``--encoders``, comma-separated, or ``all``):

  panel             numeric columns of the ``cell`` table (object_label dropped)
  panel3ch          the panel without its channel_3 columns -- the encoders
                    see only image channels 0-2
  resnet18          ``spacr.embeddings`` default spec, through the module's own
                    timm path: ImageNet weights, per-channel, [0, 1] input
  resnet18_imnorm   the same backbone with ImageNet mean/std applied, to tell
                    the encoder from the module's preprocessing
  biomedclip        BiomedCLIP ViT-B/16 image tower (PubMed figures), per-channel
  openphenom        OpenPhenom CA-MAE ViT-S/16 (RxRx3 + JUMP-CP), each channel
                    encoded alone at 256 px -- the baseline's channel policy
  openphenom_joint  OpenPhenom's native mode: the three channels together,
                    channel-wise contextualised embeddings concatenated

Every encoder goes through ``spacr.embeddings.embed_array`` so the channel
policy and column naming are the module's; non-timm backbones enter through
its ``encoder=`` argument.

Usage::

    python tools/measure_encoder_retrieval.py \\
        --plate /mnt/firecuda2/methods_paper/plate1 \\
        --encoders panel,panel3ch,resnet18,biomedclip,openphenom \\
        --out /path/to/enc395

Stages run in order (``--stages index,folds,embed,evaluate``) and each is
reusable: the index and folds are refused rather than silently rebuilt when
the object list they were made for has changed, and an embedding cache is
reused only when its recorded index digest matches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sqlite3
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_K = 15
DEFAULT_FOLDS = 5
DEFAULT_SEED = 0
SCALE_SAMPLE = 2048
SCALE_PERCENTILE = 99.0

GENE_CONTRAST = ("c1", "c2")
POSITION_CONTRAST = ("c5", "c6")
PAIR_COLUMNS = tuple(f"c{i}" for i in range(4, 25))

OPENPHENOM_REPO = "recursionpharma/OpenPhenom"
OPENPHENOM_REVISION = "0f92333685f6e9f031b804c70fe246f9b05ae90d"
BIOMEDCLIP_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

PANEL_KEYS = ("object_label", "plateID", "row_name", "column_name", "fieldID",
              "prcf", "file_name", "path_name")


def log(message: str) -> None:
    """Print one timestamped line, flushed, so a tee'd log is live."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------
# contrasts
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Contrast:
    """Two plate columns compared by retrieval.

    :param name: stable id, used as the fold file name.
    :param col_a: column labelled 0.
    :param col_b: column labelled 1.
    :param role: ``gene``, ``position`` (the named control) or ``pair``.
    """

    name: str
    col_a: str
    col_b: str
    role: str


def contrasts() -> List[Contrast]:
    """The gene contrast, then every adjacent library pair.

    c5-c6 is one of the twenty pairs AND the named position control, so it is
    listed once with role ``position``; the pair summary includes it.
    """
    out = [Contrast("gene_c1_c2", *GENE_CONTRAST, "gene")]
    for a, b in zip(PAIR_COLUMNS[:-1], PAIR_COLUMNS[1:]):
        role = "position" if (a, b) == POSITION_CONTRAST else "pair"
        out.append(Contrast(f"pair_{a}_{b}", a, b, role))
    return out


# --------------------------------------------------------------------------
# stage 1: index
# --------------------------------------------------------------------------

def _connect_readonly(db: Path) -> sqlite3.Connection:
    """Open the measurement database read-only; the plate is not ours."""
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def _rewrite(path: str, plate: Path, stale_prefix: Optional[str]) -> str:
    """Map a recorded crop path onto the local plate root.

    An explicit ``stale_prefix`` wins. Otherwise the path is cut after the
    LAST ``/<plate name>/`` it contains, which is where the acquisition root
    ended; a path that already exists is kept.
    """
    if stale_prefix and path.startswith(stale_prefix):
        return str(plate / path[len(stale_prefix):].lstrip("/"))
    if os.path.exists(path):
        return path
    marker = f"/{plate.name}/"
    cut = path.rfind(marker)
    if cut < 0:
        return path
    return str(plate / path[cut + len(marker):])


def _decode(path: str) -> np.ndarray:
    """Decode one crop to ``(h, w, c)`` uint8, raising if it cannot be read."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    return array


def _readable(path: str) -> Tuple[bool, Tuple[int, ...]]:
    """Whether a crop decodes, and its shape (empty when it does not)."""
    try:
        return True, tuple(_decode(path).shape)
    except Exception:
        return False, ()


def index_digest(prcfo: Sequence[str]) -> str:
    """A digest of the object list, in order. Folds and caches are keyed by it."""
    h = hashlib.sha256()
    for key in prcfo:
        h.update(key.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def build_index(args: argparse.Namespace) -> None:
    """Join crops to cells, drop what cannot be read, save index and panel."""
    import pandas as pd

    out = Path(args.out)
    db = Path(args.db) if args.db else Path(args.plate) / "measurements" / "measurements.db"
    log(f"index: reading {db} (read-only)")
    con = _connect_readonly(db)
    try:
        png = pd.read_sql("select png_path, prcfo, row_name, column_name from png_list", con)
        cell = pd.read_sql("select * from cell", con)
    finally:
        con.close()
    report: Dict[str, object] = {"db": str(db), "png_list_rows": int(len(png)),
                                 "cell_rows": int(len(cell))}

    cell["prcfo"] = cell["prcf"] + "_o" + cell["object_label"].astype(str)
    merged = cell.merge(png[["prcfo", "png_path"]], on="prcfo", how="inner",
                        validate="one_to_one")
    report["joined"] = int(len(merged))
    merged["crop_path"] = [_rewrite(p, Path(args.plate), args.stale_prefix)
                           for p in merged["png_path"]]
    merged["well"] = merged["row_name"] + "_" + merged["column_name"]
    merged = merged.sort_values(["column_name", "row_name", "prcfo"]).reset_index(drop=True)

    if args.limit_per_well:
        merged = (merged.groupby("well", sort=False).head(args.limit_per_well)
                  .reset_index(drop=True))
        report["limit_per_well"] = int(args.limit_per_well)

    log(f"index: decoding {len(merged)} crops to check readability")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        checks = list(pool.map(_readable, merged["crop_path"]))
    ok = np.array([c[0] for c in checks], bool)
    shapes = [c[1] for c in checks if c[0]]
    expected = max(set(shapes), key=shapes.count)
    same = np.array([c[0] and c[1] == expected for c in checks], bool)
    dropped = merged.loc[~same, ["prcfo", "crop_path"]]
    report.update(crop_shape=list(expected), unreadable=int((~ok).sum()),
                  wrong_shape=int((ok & ~same).sum()),
                  dropped=dropped.to_dict("records"),
                  decode_seconds=round(time.time() - t0, 1))
    for row in report["dropped"]:
        log(f"index: SKIP {row['crop_path']} (unreadable or wrong shape)")
    merged = merged[same].reset_index(drop=True)

    panel = merged.drop(columns=[c for c in PANEL_KEYS if c in merged.columns]
                        + ["prcfo", "png_path", "crop_path", "well"])
    panel = panel.select_dtypes(include="number")
    values = panel.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise SystemExit("index: the panel has non-finite values; refusing")
    report.update(objects=int(len(merged)), wells=int(merged["well"].nunique()),
                  panel_columns=int(panel.shape[1]),
                  panel_constant=[c for c in panel.columns if panel[c].nunique() <= 1])

    digest = index_digest(merged["prcfo"].tolist())
    report["index_digest"] = digest
    out.mkdir(parents=True, exist_ok=True)
    merged[["prcfo", "well", "row_name", "column_name", "crop_path"]].to_csv(
        out / "index.csv", index=False)
    np.save(out / "panel.npy", values)
    (out / "panel_columns.txt").write_text("\n".join(panel.columns) + "\n")

    report["input_scale"] = _input_scale(merged["crop_path"].tolist(), expected, args)
    (out / "index_report.json").write_text(json.dumps(report, indent=1))
    per_col = merged.groupby("column_name").agg(objects=("prcfo", "size"),
                                                wells=("well", "nunique"))
    log(f"index: {len(merged)} objects, {report['wells']} wells, "
        f"{panel.shape[1]} panel columns, digest {digest}")
    log("index: per column objects/wells " + ", ".join(
        f"{c}={r.objects}/{r.wells}" for c, r in per_col.iterrows()))


def _input_scale(paths: List[str], shape: Tuple[int, ...],
                 args: argparse.Namespace) -> Dict[str, object]:
    """Per-channel 99th percentile over a fixed random sample of crops.

    Computed from exact integer histograms, so it is the percentile of the
    sample rather than an approximation, and the same for every encoder.
    """
    rng = np.random.default_rng(args.seed)
    take = np.sort(rng.choice(len(paths), size=min(SCALE_SAMPLE, len(paths)),
                              replace=False))
    channels = shape[2]
    hist = np.zeros((channels, 65536), np.int64)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for crop in pool.map(_decode, [paths[i] for i in take]):
            for c in range(channels):
                hist[c] += np.bincount(crop[..., c].ravel(), minlength=65536)[:65536]
    scales = []
    for c in range(channels):
        cdf = np.cumsum(hist[c]) / hist[c].sum()
        scales.append(float(max(1, int(np.searchsorted(cdf, SCALE_PERCENTILE / 100.0)))))
    log(f"index: input scale (p{SCALE_PERCENTILE:g} of {len(take)} crops) = {scales}")
    return {"percentile": SCALE_PERCENTILE, "sample": int(len(take)),
            "seed": int(args.seed), "per_channel": scales}


def load_index(out: Path):
    """The saved index, panel and report; refuses when the index is absent."""
    import pandas as pd

    if not (out / "index.csv").exists():
        raise SystemExit(f"no index under {out}; run --stages index first")
    index = pd.read_csv(out / "index.csv", dtype=str)
    report = json.loads((out / "index_report.json").read_text())
    if index_digest(index["prcfo"].tolist()) != report["index_digest"]:
        raise SystemExit("index.csv does not match index_report.json")
    return index, report


# --------------------------------------------------------------------------
# stage 2: folds
# --------------------------------------------------------------------------

def build_folds(args: argparse.Namespace) -> None:
    """Split every contrast once, grouped by well, and save the indices."""
    from sklearn.model_selection import StratifiedGroupKFold

    out = Path(args.out)
    index, report = load_index(out)
    folds_dir = out / "folds"
    folds_dir.mkdir(exist_ok=True)
    manifest_path = folds_dir / "manifest.json"
    wanted = {"index_digest": report["index_digest"], "n_splits": args.folds,
              "seed": args.seed, "grouped_by": "well",
              "splitter": "sklearn.model_selection.StratifiedGroupKFold(shuffle=True)"}
    if manifest_path.exists():
        have = json.loads(manifest_path.read_text())
        if {k: have.get(k) for k in wanted} == wanted:
            log(f"folds: reusing {folds_dir} (digest {report['index_digest']})")
            return
        if not args.refresh_folds:
            raise SystemExit(
                f"folds under {folds_dir} were made for a different index or "
                "split; pass --refresh-folds to replace them deliberately")

    column = index["column_name"].to_numpy()
    well = index["well"].to_numpy()
    summary = []
    for contrast in contrasts():
        rows = np.flatnonzero(np.isin(column, [contrast.col_a, contrast.col_b]))
        y = (column[rows] == contrast.col_b).astype(np.int8)
        groups = well[rows]
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                  random_state=args.seed)
        test_fold = np.full(len(rows), -1, np.int8)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(rows)), y, groups)):
            shared = set(groups[tr]) & set(groups[te])
            assert not shared, f"{contrast.name}: well leak in fold {f}: {shared}"
            test_fold[te] = f
        assert (test_fold >= 0).all()

        rng = np.random.default_rng([args.seed, zlib.crc32(contrast.name.encode())])
        y_object = rng.permutation(y).astype(np.int8)
        wells = np.unique(groups)
        well_label = np.array([y[groups == w][0] for w in wells], np.int8)
        permuted = dict(zip(wells, rng.permutation(well_label)))
        y_well = np.array([permuted[g] for g in groups], np.int8)

        np.savez(folds_dir / f"{contrast.name}.npz", rows=rows.astype(np.int64),
                 y=y, groups=groups.astype(str), test_fold=test_fold,
                 y_null_object=y_object, y_null_well=y_well)
        summary.append(dict(name=contrast.name, role=contrast.role,
                            col_a=contrast.col_a, col_b=contrast.col_b,
                            n=int(len(rows)), n_a=int((y == 0).sum()),
                            n_b=int((y == 1).sum()),
                            wells_a=int(len(set(groups[y == 0]))),
                            wells_b=int(len(set(groups[y == 1]))),
                            test_sizes=np.bincount(test_fold).tolist()))
    manifest_path.write_text(json.dumps({**wanted, "contrasts": summary}, indent=1))
    log(f"folds: wrote {len(summary)} contrasts to {folds_dir}")


def load_fold(out: Path, name: str) -> Dict[str, np.ndarray]:
    """One contrast's saved rows, labels, nulls and fold assignment."""
    with np.load(out / "folds" / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------
# stage 3: embed
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderDef:
    """One feature set the harness knows how to produce.

    :param name: the ``--encoders`` token.
    :param kind: ``panel`` (read from the database) or ``embedding``.
    :param policy: the ``spacr.embeddings`` channel policy it runs under.
    :param trained_on: one phrase for the results table.
    """

    name: str
    kind: str
    policy: str = ""
    trained_on: str = ""


ENCODERS: Dict[str, EncoderDef] = {e.name: e for e in (
    EncoderDef("panel", "panel", trained_on="hand-designed, all channels"),
    EncoderDef("panel3ch", "panel", trained_on="hand-designed, channels 0-2"),
    EncoderDef("resnet18", "embedding", "per_channel", "ImageNet photographs"),
    EncoderDef("resnet18_imnorm", "embedding", "per_channel",
               "ImageNet photographs, ImageNet mean/std input"),
    EncoderDef("biomedclip", "embedding", "per_channel", "PubMed figure-caption pairs"),
    EncoderDef("openphenom", "embedding", "per_channel",
               "RxRx3 + JUMP-CP Cell Painting (MAE)"),
    EncoderDef("openphenom_joint", "embedding", "project",
               "RxRx3 + JUMP-CP Cell Painting (MAE), channels jointly"),
)}

#: Named in 395 and not measurable here; recorded so the table says why.
NOT_MEASURED = {
    "subcell": "no public HuggingFace repository resolves (CZI/SubCell 404)",
    "cell_dino": "no public HuggingFace repository resolves (facebook/cell-dino 404)",
}


def _sha256(path: str) -> str:
    """Digest of a weight file on this machine, the only bytes a rerun can match."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _hf_file(repo: str, filename: str, revision: Optional[str] = None) -> str:
    """Path of a cached HuggingFace file, or '' when it is not cached."""
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(repo, filename, revision=revision)
        return found if isinstance(found, str) else ""
    except Exception:
        return ""


def _torch_batches(stack: np.ndarray, batch_size: int, device: str,
                   forward: Callable) -> np.ndarray:
    """Run ``forward`` over an ``(n, h, w, c)`` float32 stack in NCHW batches."""
    import torch

    out: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, stack.shape[0], batch_size):
            chunk = np.ascontiguousarray(
                stack[start:start + batch_size].transpose(0, 3, 1, 2))
            tensor = torch.from_numpy(chunk).to(device, non_blocking=True)
            out.append(forward(tensor).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def make_encoder(name: str, device: str, batch_size: int
                 ) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, object]]:
    """Build one encoder callable for ``embed_array(encoder=...)``, plus provenance.

    Every callable takes ``(n, 224, 224, 3)`` float32 in [0, 1] -- what
    ``embed_array`` hands over -- and returns ``(n, dims)``.
    """
    import torch

    if name == "resnet18":
        from spacr.embeddings import EmbeddingSpec, _timm_encoder, _weights_on_disk
        spec = EmbeddingSpec(backbone="resnet18", device=device,
                             batch_size=batch_size, normalize=False)
        path, digest, size = _weights_on_disk("resnet18")
        return _timm_encoder(spec), {"source": "spacr.embeddings._timm_encoder",
                                     "weights": path, "sha256": digest}

    if name == "resnet18_imnorm":
        import timm
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
        model.eval().to(device)
        cfg = model.pretrained_cfg
        mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

        def run(stack: np.ndarray) -> np.ndarray:
            """ImageNet-normalise, then the same resnet18 forward pass."""
            return _torch_batches(stack, batch_size, device,
                                  lambda x: model((x - mean) / std))
        from spacr.embeddings import _weights_on_disk
        path, digest, _ = _weights_on_disk("resnet18")
        return run, {"source": f"timm {cfg.get('hf_hub_id', 'resnet18')}",
                     "mean": cfg["mean"], "std": cfg["std"],
                     "weights": path, "sha256": digest}

    if name == "biomedclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{BIOMEDCLIP_REPO}")
        visual = model.visual.eval().to(device)
        del model
        norm = [t for t in preprocess.transforms if type(t).__name__ == "Normalize"][0]
        mean = torch.tensor(norm.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(norm.std, device=device).view(1, 3, 1, 1)

        def run(stack: np.ndarray) -> np.ndarray:
            """CLIP-normalise at the native 224 px, projected image features."""
            return _torch_batches(stack, batch_size, device,
                                  lambda x: visual((x - mean) / std))
        path = _hf_file(BIOMEDCLIP_REPO, "open_clip_pytorch_model.bin")
        return run, {"source": f"open_clip {open_clip.__version__} hf-hub:{BIOMEDCLIP_REPO}",
                     "weights": path, "sha256": _sha256(path) if path else "",
                     "features": "projected image embedding (512)"}

    if name in ("openphenom", "openphenom_joint"):
        from transformers import AutoModel
        import torch.nn.functional as F
        model = AutoModel.from_pretrained(OPENPHENOM_REPO, revision=OPENPHENOM_REVISION,
                                          trust_remote_code=True)
        model.eval().to(device)
        model.return_channelwise_embeddings = True
        joint = name == "openphenom_joint"

        def forward(x):
            """Upsample 224->256, back to the 0-255 range the model's own
            Normalizer divides, then its instance norm and encoder."""
            if not joint:
                # per-channel policy: embed_array repeats ONE plane three
                # times; OpenPhenom is channel-agnostic, so give it the plane
                # once rather than three identical channels.
                if not torch.equal(x[:, 0], x[:, 1]):
                    raise RuntimeError("openphenom per-channel adapter was "
                                       "handed three different channels")
                x = x[:, :1]
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            return model.predict(x * 255.0)

        def run(stack: np.ndarray) -> np.ndarray:
            """OpenPhenom embeddings for one stack, in batches."""
            return _torch_batches(stack, batch_size, device, forward)
        path = _hf_file(OPENPHENOM_REPO, "model.safetensors", OPENPHENOM_REVISION)
        return run, {"source": f"transformers trust_remote_code {OPENPHENOM_REPO}@{OPENPHENOM_REVISION}",
                     "weights": path, "sha256": _sha256(path) if path else "",
                     "input": "bilinear 224->256, x255, model Normalizer + InstanceNorm2d",
                     "features": "channel-wise mean-pooled patch tokens, 384 per channel"}

    raise SystemExit(f"unknown encoder {name!r}")


def embed(args: argparse.Namespace, name: str) -> None:
    """Embed every indexed crop with one encoder, streaming, resumable."""
    import torch
    from spacr.embeddings import EmbeddingSpec, embed_array

    out = Path(args.out)
    index, report = load_index(out)
    digest = report["index_digest"]
    emb_dir = out / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    meta_path = emb_dir / f"{name}.json"
    cache_path = emb_dir / f"{name}.npy"
    partial_path = emb_dir / f"{name}.partial.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("index_digest") == digest and cache_path.exists():
            log(f"embed {name}: cache is complete for digest {digest}; skipping")
            return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    definition = ENCODERS[name]
    spec = EmbeddingSpec(backbone=name, channel_policy=definition.policy,
                         channels=(0, 1, 2), batch_size=args.batch_size,
                         device=device, normalize=False)
    scale = np.asarray(report["input_scale"]["per_channel"][:3], np.float32)
    paths = index["crop_path"].tolist()
    n = len(paths)

    t_load = time.time()
    encoder, provenance = make_encoder(name, device, args.batch_size)
    load_seconds = time.time() - t_load
    log(f"embed {name}: model ready on {device} in {load_seconds:.1f}s; {n} crops")

    start, values = 0, None
    if partial_path.exists() and cache_path.exists():
        partial = json.loads(partial_path.read_text())
        if partial.get("index_digest") == digest:
            start = int(partial["done"])
            values = np.load(cache_path, mmap_mode="r+")
            log(f"embed {name}: resuming at {start}")

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    columns = None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for s in range(start, n, args.chunk):
            crops = np.stack(list(pool.map(_decode, paths[s:s + args.chunk])))
            stack = np.clip(crops[..., :3].astype(np.float32) / scale, 0.0, 1.0)
            result = embed_array(stack, spec, encoder=encoder)
            if not np.isfinite(result.values).all():
                raise SystemExit(f"embed {name}: non-finite embedding at chunk {s}")
            if values is None:
                values = np.lib.format.open_memmap(
                    cache_path, mode="w+", dtype=np.float32,
                    shape=(n, result.values.shape[1]))
            values[s:s + len(stack)] = result.values
            columns = result.columns
            done = s + len(stack)
            values.flush()
            partial_path.write_text(json.dumps({"index_digest": digest, "done": done}))
            if (s // args.chunk) % 10 == 0 or done == n:
                rate = (done - start) / max(time.time() - t0, 1e-9)
                log(f"embed {name}: {done}/{n}  {rate:.0f} crops/s")
    seconds = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if torch.cuda.is_available() and device.startswith("cuda") else 0.0)
    meta = {
        "encoder": name, "trained_on": definition.trained_on,
        "channel_policy": definition.policy, "spec_fingerprint": spec.fingerprint(),
        "dims": int(values.shape[1]),
        "first_column": columns[0] if columns else "",
        "index_digest": digest, "n_objects": n,
        "embed_seconds": round(seconds, 1), "model_load_seconds": round(load_seconds, 1),
        "crops_per_second": round((n - start) / max(seconds, 1e-9), 1),
        "resumed_from": start, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "",
        "peak_gpu_gib": round(peak, 2), "torch": torch.__version__,
        "input_scale": report["input_scale"], "provenance": provenance,
        "host": platform.node(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    del values
    meta_path.write_text(json.dumps(meta, indent=1))
    partial_path.unlink(missing_ok=True)
    log(f"embed {name}: {meta['dims']} dims in {seconds:.0f}s "
        f"({meta['crops_per_second']} crops/s, peak GPU {peak:.2f} GiB)")
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# stage 4: evaluate
# --------------------------------------------------------------------------

def features_for(out: Path, name: str, digest: str) -> np.ndarray:
    """The object-by-feature matrix for one feature set, in index order."""
    if name == "panel":
        return np.load(out / "panel.npy")
    if name == "panel3ch":
        columns = (out / "panel_columns.txt").read_text().split()
        keep = [i for i, c in enumerate(columns) if "channel_3" not in c]
        return np.load(out / "panel.npy")[:, keep]
    meta_path = out / "embeddings" / f"{name}.json"
    if not meta_path.exists():
        raise SystemExit(f"no finished embedding for {name}; run --stages embed")
    meta = json.loads(meta_path.read_text())
    if meta["index_digest"] != digest:
        raise SystemExit(f"{name}: embedding cache was made for another index")
    return np.load(out / "embeddings" / f"{name}.npy")


def knn_folds(X: np.ndarray, y: np.ndarray, test_fold: np.ndarray, k: int,
              jobs: int) -> List[Dict[str, float]]:
    """kNN over saved folds; the scaler never sees the test half."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    rows = []
    for f in range(int(test_fold.max()) + 1):
        te, tr = test_fold == f, test_fold != f
        scaler = StandardScaler().fit(X[tr])
        knn = KNeighborsClassifier(n_neighbors=k, algorithm="brute", n_jobs=jobs)
        knn.fit(scaler.transform(X[tr]), y[tr])
        pred = knn.predict(scaler.transform(X[te]))
        recall = recall_score(y[te], pred, labels=[0, 1], average=None, zero_division=0)
        rows.append(dict(fold=f, accuracy=accuracy_score(y[te], pred),
                         balanced_accuracy=balanced_accuracy_score(y[te], pred),
                         recall_a=recall[0], recall_b=recall[1],
                         majority=max(np.mean(y[te] == 0), np.mean(y[te] == 1)),
                         n_train=int(tr.sum()), n_test=int(te.sum())))
    return rows


def evaluate(args: argparse.Namespace, names: Sequence[str]) -> None:
    """Score every feature set on every contrast with the saved folds."""
    import pandas as pd

    out = Path(args.out)
    index, report = load_index(out)
    digest = report["index_digest"]
    manifest = json.loads((out / "folds" / "manifest.json").read_text())
    if manifest["index_digest"] != digest:
        raise SystemExit("folds were made for another index; rebuild them")

    fold_rows = []
    timing = {}
    for name in names:
        X = features_for(out, name, digest)
        if X.shape[0] != len(index):
            raise SystemExit(f"{name}: {X.shape[0]} rows for {len(index)} objects")
        t0 = time.time()
        for contrast in contrasts():
            fold = load_fold(out, contrast.name)
            label_sets = [("true", fold["y"])]
            if args.nulls == "all" or (args.nulls == "main" and contrast.role != "pair"):
                label_sets += [("null_object", fold["y_null_object"]),
                               ("null_well", fold["y_null_well"])]
            Xc = X[fold["rows"]]
            for labels, y in label_sets:
                for row in knn_folds(Xc, y, fold["test_fold"], args.k, args.jobs):
                    fold_rows.append(dict(encoder=name, contrast=contrast.name,
                                          role=contrast.role, col_a=contrast.col_a,
                                          col_b=contrast.col_b, labels=labels,
                                          dims=int(X.shape[1]), k=args.k, **row))
        timing[name] = round(time.time() - t0, 1)
        log(f"evaluate {name}: {X.shape[1]} dims, {timing[name]}s")

    folds_df = pd.DataFrame(fold_rows)
    tag = f"_{args.tag}" if args.tag else ""
    folds_df.to_csv(out / f"results_folds{tag}.csv", index=False)
    metrics = ["accuracy", "balanced_accuracy", "recall_a", "recall_b", "majority"]
    grouped = folds_df.groupby(["encoder", "contrast", "role", "col_a", "col_b",
                                "labels", "dims", "k"], sort=False)
    summary = grouped[metrics].mean().add_suffix("_mean").join(
        grouped[metrics].std(ddof=1).add_suffix("_sd")).join(
        grouped["n_test"].sum().rename("n_objects")).reset_index()
    summary.to_csv(out / f"results{tag}.csv", index=False)

    pairs = []
    for name in names:
        s = summary[(summary.encoder == name) & (summary.labels == "true")]
        null = s[s.role.isin(["pair", "position"])]["balanced_accuracy_mean"]
        gene = float(s[s.role == "gene"]["balanced_accuracy_mean"].iloc[0])
        pairs.append(dict(encoder=name, trained_on=ENCODERS[name].trained_on,
                          gene_balanced_accuracy=gene,
                          pairs_n=int(len(null)), pairs_mean=float(null.mean()),
                          pairs_sd=float(null.std(ddof=1)), pairs_min=float(null.min()),
                          pairs_max=float(null.max()),
                          gene_z_over_pairs=float((gene - null.mean()) / null.std(ddof=1)),
                          evaluate_seconds=timing[name]))
    pd.DataFrame(pairs).to_csv(out / f"pairs_summary{tag}.csv", index=False)
    print_table(summary, pd.DataFrame(pairs))


def print_table(summary, pairs) -> None:
    """The gene, position and null rows, then the pair distribution."""
    show = summary[summary.role != "pair"]
    print("\nencoder            contrast     labels       acc            bal.acc        recall a/b")
    for _, r in show.iterrows():
        print(f"{r.encoder:<18} {r.col_a}-{r.col_b:<8} {r.labels:<12} "
              f"{r.accuracy_mean:.3f}+/-{r.accuracy_sd:.3f}  "
              f"{r.balanced_accuracy_mean:.3f}+/-{r.balanced_accuracy_sd:.3f}  "
              f"{r.recall_a_mean:.3f}/{r.recall_b_mean:.3f}")
    print("\nencoder            gene bal.acc  20 adjacent pairs mean+/-sd [min, max]   gene z")
    for _, r in pairs.iterrows():
        print(f"{r.encoder:<18} {r.gene_balanced_accuracy:.3f}         "
              f"{r.pairs_mean:.3f}+/-{r.pairs_sd:.3f} [{r.pairs_min:.3f}, {r.pairs_max:.3f}]"
              f"          {r.gene_z_over_pairs:+.1f}")
    for name, why in NOT_MEASURED.items():
        print(f"not measured: {name} -- {why}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """The command line; ``--plate``, ``--encoders`` and ``--out`` are required."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plate", required=True, help="plate root; crops are rewritten onto it")
    p.add_argument("--encoders", required=True,
                   help=f"comma-separated, or 'all': {', '.join(ENCODERS)}")
    p.add_argument("--out", required=True, help="output directory (caches, folds, results)")
    p.add_argument("--db", help="measurement database (default <plate>/measurements/measurements.db)")
    p.add_argument("--stages", default="index,folds,embed,evaluate")
    p.add_argument("--stale-prefix", help="recorded path prefix to replace with --plate")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--nulls", choices=("main", "all", "none"), default="main",
                   help="shuffled-label nulls on gene+position (main), every pair, or none")
    p.add_argument("--device", help="torch device (default cuda when available)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512, help="crops decoded per step")
    p.add_argument("--workers", type=int, default=8, help="crop decoding threads")
    p.add_argument("--jobs", type=int, default=8, help="kNN threads")
    p.add_argument("--limit-per-well", type=int, default=0,
                   help="smoke test only: keep the first N objects of each well")
    p.add_argument("--refresh-folds", action="store_true",
                   help="replace saved folds made for a different index")
    p.add_argument("--tag", default="", help="suffix for the result files")
    args = p.parse_args(argv)
    names = list(ENCODERS) if args.encoders == "all" else [
        e.strip() for e in args.encoders.split(",") if e.strip()]
    unknown = [e for e in names if e not in ENCODERS]
    if unknown:
        p.error(f"unknown encoder(s) {unknown}; known: {', '.join(ENCODERS)}")
    args.encoder_names = names
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested stages in order."""
    args = parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if "index" in stages:
        if (Path(args.out) / "index.csv").exists() and not args.refresh_folds:
            log("index: exists; reusing (pass --refresh-folds to rebuild index and folds)")
        else:
            build_index(args)
    if "folds" in stages:
        build_folds(args)
    if "embed" in stages:
        for name in args.encoder_names:
            if ENCODERS[name].kind == "embedding":
                embed(args, name)
    if "evaluate" in stages:
        evaluate(args, args.encoder_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
