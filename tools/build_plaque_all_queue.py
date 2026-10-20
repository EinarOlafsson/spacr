"""Build one Make Masks queue holding every plaque well image, for plaque v5.

The maintainer's brief (2026-10-12): "all of the plaques, no holdout. Every
single curated and non-curated picture with plaques in one dataset." The
result is a nested curation folder that ``spacr-make-masks --folder <out>``
opens as one resumable queue::

    <out>/<stem>.<ext>          one image per unique field
    <out>/masks/<stem>.tif      its best existing mask (always a copy)
    <out>/curate_status.csv     done / skip for fields already decided
    <out>/curate_scores.csv     stem, prob -- for ``--order easy``
    <out>/curate_drafts.csv     object counts, so the first launch is quick
    <out>/manifest.csv          provenance of every field kept
    <out>/duplicates.csv        every copy that was folded into a kept field
    <out>/excluded.csv          fields left out on purpose, and why
    <out>/missing_drafts.csv    fields with no mask at all (draft them on WS)
    <out>/README.txt

Where the images come from
--------------------------
The 2026-10-08 model review (``model_review_2026-10-08``) already inventoried
every plaque well image on both machines -- Gel Doc, every curation queue,
the training sets, the NAS caches, the literature and other-pathogen archives
-- and collapsed identical pixels: 17,888 segmentation fields, each with a
toxoplasma_plaque_v3 draft. That inventory is the base. Every Home curation
source is then hashed with the same canonical pixel digest, so a field it
already holds maps onto it and anything newer is added.

De-duplication
--------------
1. Exact: the model review's canonical pixel sha256 (alternative encodings of
   the same pixels are one field).
2. Explicit: ``plaque_r4_review_dedup.csv`` (removed -> kept), recropped
   parents (``<stem>`` whose ``<stem>__rNN`` children exist, or that a ledger
   marks ``recropped``) and the Gel Doc GRA8 Plate1 wells re-imaged as
   ReimagedPlate1.
3. Near-duplicates: spaCR's 64-bit difference hash
   (``spacr.plaque_papers._image_fingerprint``: <= 4 bits, aspect within 3%)
   proposes pairs; a high-pass correlation of 64 x 64 thumbnails confirms
   them, since every well is a dark disc and look-alikes share a dHash.

In each cluster the kept field is the one carrying the latest human-curated
mask (a curated mask is tied to its own pixel grid), else an original image
rather than the upscaled copy a curation bundle holds, else the largest, else
the copy most curation queues used (a figure re-saved with detector boxes
drawn on it is used by fewer).
Curation bundles (``*_seg.npy``) are hashed on their own image, the grid the
mask was drawn on; pmc_staged and curate_single bundles hold 1200-pixel
upscales of the original wells.

Masks
-----
Human-curated (latest ``done`` across every ledger, Make Masks' keep verdicts
and the round-3 training labels) > the newest model draft (the model review's
toxoplasma_plaque_v3 drafts, then older queue drafts) > none. A later
``skip`` overrides an older ``done``, as the r5 training freeze did. Masks are
always written as new files: editing one never touches an original.

Images are hard links when the source is on the output's filesystem, else
copies; seg-bundle-only sources are written out as TIFF. Nothing under the
source folders is written.

The Gel Doc set-aside wells (plates 13 and above, Dia_Plate18 included) are
queued since 2026-10-12 by the maintainer's decision; ``--exclude-set-aside``
leaves them out. The build also writes ``drafts_request.csv`` (stem, image
path relative to af3, sha256, shape): every field still waiting for a human.

Three modes
-----------
1. Merged build (Home, CPU only, RAM-capped)::

    CUDA_VISIBLE_DEVICES='' tools/run_capped.sh 6G python tools/build_plaque_all_queue.py \\
        --out /mnt/wd4tb/af3/projects/automated_plaque_assay/curation/plaque_all_2026-10-12 \\
        --work /mnt/wd4tb/scratch/plaque_all_20261012/work

2. Drafts (workstation, GPU, through the shared queue): one label TIFF per
   requested field, ``<dir>/<stem>.tif``, plus ``drafts_run.json`` (model,
   checkpoint sha256, settings, Cellpose version). Resumable::

    tools/gpu_turn.sh plaque-all-v4 python tools/build_plaque_all_queue.py \\
        --out <merged folder> --make-drafts <dir> --request <merged folder>/drafts_request.csv \\
        --af3-root /media/carruthers/mnt1/af3 --model toxoplasma_plaque_v4 \\
        --diameter 30 --flow-threshold 0.4 --cellprob-threshold 0

3. Variant: ``<out>_<name>`` beside the merged folder, with the same images
   (hard links), the same decided fields (rows and masks copied verbatim) and
   the drafts from ``--drafts-from`` for everything else::

    python tools/build_plaque_all_queue.py --out <merged folder> --variant v4 --drafts-from <dir>
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageSequence

Image.MAX_IMAGE_PIXELS = 200_000_000

AF3 = Path("/mnt/wd4tb/af3/projects/automated_plaque_assay/curation")
LIVE = Path("/mnt/firecuda2/Claude/toxoplasma_projects/plaque_assay_model/curation")
NAS_SYNC = Path("/nas_mnt/data/sync/projects/automated_plaque_assay/curation")
NAS_DATA = Path("/nas_mnt/data/training_data/projects/automated_plaque_assay/data")
MR = AF3 / "model_review_2026-10-08"
WS_PREFIXES = (("/media/carruthers/mnt1/af3/", "/mnt/wd4tb/af3/"),
               ("/media/carruthers/mnt3/claude/", "/mnt/firecuda2/Claude/"),
               ("/nas_mnt/carruthers/sync/", "/nas_mnt/data/sync/"))
IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
SEG = "_seg.npy"
DRAFT_V3 = "toxoplasma_plaque_v3"
DRAFT_V4 = "toxoplasma_plaque_v4"
AF3_ROOT = Path("/mnt/wd4tb/af3")
SIMILAR_BITS = 4
ASPECT_TOL = 0.03
CONFIRM_ANY = 0.95
CONFIRM_SAME_FIGURE = 0.90
DOMAIN_PROB = (("gel_doc", 0.95), ("previous_curation", 0.85),
               ("previous_segmentation", 0.85), ("nas_", 0.85),
               ("permanent_holdout", 0.8), ("toxoplasma_literature", 0.75),
               ("literature_wells", 0.6), ("other_species", 0.3))
LIT_DISCARD_EXCLUDED = "done and Discard in the literature hold-out (maintainer's rescoring decision, kept)"
SET_ASIDE_EXCLUDED = "Gel Doc set-aside plate 13 and above (earlier decision, kept; includes the 5 curated Dia_Plate18 wells)"


def say(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def home_path(path: str) -> str:
    for ws, home in WS_PREFIXES:
        if path.startswith(ws):
            return home + path[len(ws):]
    return path


def wait_for_ram(min_gb: float = 8.0) -> None:
    while True:
        with open("/proc/meminfo") as handle:
            info = {line.split(":")[0]: int(line.split()[1]) for line in handle}
        avail = info["MemAvailable"] / 1024 ** 2
        if avail >= min_gb:
            return
        say(f"MemAvailable {avail:.1f} GB < {min_gb} GB; waiting 2 min")
        time.sleep(120)


def planes(x):
    x = np.asarray(x)
    if x.ndim == 2:
        yield x
    elif x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        yield x[..., 0] if x.shape[-1] == 1 else x[..., :3]
    elif x.ndim == 3 and x.shape[0] in (1, 3, 4):
        yield from planes(np.moveaxis(x, 0, -1))
    elif x.ndim >= 3:
        for y in x:
            yield from planes(y)


def canonical_pixels(x):
    """The model review's canonical form (prepare_review.py), so digests agree."""
    x = np.asarray(x)
    if x.ndim == 3 and np.array_equal(x[..., 0], x[..., 1]) and np.array_equal(x[..., 0], x[..., 2]):
        x = x[..., 0]
    if x.dtype != np.uint8 and x.size and x.min() >= 0 and x.max() <= 255 and np.equal(x, np.floor(x)).all():
        x = x.astype(np.uint8)
    return np.ascontiguousarray(x)


def digest_pixels(x) -> str:
    h = hashlib.sha256((str(x.shape) + str(x.dtype)).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def read_first_plane(path: Path):
    if path.name.endswith(SEG):
        bundle = np.load(path, allow_pickle=True).item()
        return canonical_pixels(next(planes(bundle["img"])))
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(path) as tf:
                return canonical_pixels(next(planes(tf.pages[0].asarray())))
        except (OSError, ValueError, RuntimeError):
            pass
    with Image.open(path) as im:
        frame = next(ImageSequence.Iterator(im))
        arr = np.asarray(frame.convert("RGB") if frame.mode not in ("L", "I;16", "I", "F") else frame).copy()
    return canonical_pixels(next(planes(arr)))


def read_mask(path: Path):
    if path.name.endswith(SEG):
        return np.asarray(np.load(path, allow_pickle=True).item()["masks"])
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    return np.asarray(tifffile.imread(path))


def n_objects(mask) -> int:
    values = np.unique(mask)
    return int((values != 0).sum())


def dhash(gray: np.ndarray) -> int:
    """spacr.plaque_papers._image_fingerprint's 64-bit difference hash."""
    small = np.asarray(Image.fromarray(gray).resize((9, 8), Image.BILINEAR), dtype=float)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def to_gray8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    step = max(1, int(max(x.shape[:2]) // 1024))
    x = x[::step, ::step]
    if x.ndim == 3:
        x = x[..., :3].astype(np.float32).mean(axis=2)
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, (0.5, 99.5)) if x.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def figure_key(stem: str) -> str:
    """The published figure (or plate) a well was cut from, ignoring the cropping pipeline."""
    s = re.sub(r"__[0-9a-f]{20}$", "", stem)
    s = re.sub(r"^r\d__(lit_)?[a-z_]+?__", "", s)
    s = re.sub(r"^PMC\d+__", "", s)
    return re.sub(r"\.(jpe?g|png|tiff?|gif)$", "", s.split("__")[0], flags=re.I).lower()


def highpass_feature(gray: np.ndarray, size: int = 128) -> np.ndarray:
    """Fine-scale texture at 128 x 128 (detail finer than 1/32 of the field).

    Calibrated 2026-10-12 on the dHash candidates: copies (resized, re-encoded,
    detector box drawn on) score 0.85-1.0; different wells of one figure up to
    about 0.87; different figures rarely above 0.9.
    """
    small = np.asarray(Image.fromarray(gray).resize((size, size), Image.BILINEAR), dtype=np.float32)
    blur = np.asarray(Image.fromarray(small).resize((size // 4, size // 4), Image.BILINEAR).resize((size, size), Image.BILINEAR), dtype=np.float32)
    v = (small - blur).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float16)


class HashCache:
    def __init__(self, path: Path):
        self.path = path
        self.rows = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["key"]] = row
        self.handle = open(path, "a")

    def get(self, path: Path):
        st = path.stat()
        key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
        row = self.rows.get(key)
        if row is None:
            x = read_first_plane(path)
            gray = to_gray8(x)
            row = dict(key=key, px=digest_pixels(x), shape=list(x.shape), dtype=str(x.dtype),
                       dhash=f"{dhash(gray):016x}")
            self.rows[key] = row
            self.handle.write(json.dumps(row) + "\n")
            self.handle.flush()
        return row


def parse_when(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None).isoformat(timespec="seconds")
    except ValueError:
        return text


def mtime_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def read_status(path: Path):
    rows = []
    if path and path.is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stem = (row.get("stem") or "").strip()
                if stem:
                    rows.append(dict(stem=stem, state=(row.get("state") or "").strip().lower(),
                                     n=row.get("n_objects") or "", when=parse_when(row.get("updated"))))
    return rows


def read_verdicts(path: Path):
    out = {}
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image = Path(home_path((row.get("image path") or "").strip()))
                keep = str(row.get("keep", "")).strip().lower()
                if image.name:
                    out[image.stem] = dict(keep=keep in ("true", "1", "yes", "keep"),
                                           discard=keep in ("false", "0", "no", "discard"),
                                           mask=Path(home_path((row.get("mask path") or "").strip())))
    return out


def list_fields(folder: Path, layout: str):
    """Yield (stem, image_path, mask_path) for one source folder."""
    if not folder.is_dir():
        return
    names = sorted(p for p in os.listdir(folder) if not p.startswith("."))
    if layout == "seg":
        images = {Path(n).stem: folder / n for n in names if Path(n).suffix.lower() in IMAGE_EXTS}
        for n in names:
            if n.endswith(SEG):
                stem = n[:-len(SEG)]
                yield stem, images.get(stem, folder / n), folder / n
        bundled = {n[:-len(SEG)] for n in names if n.endswith(SEG)}
        for stem, image in images.items():
            if stem not in bundled:
                yield stem, image, None
    else:
        masks = folder / "masks"
        for n in names:
            p = folder / n
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                mask = None
                for ext in (".tif", ".tiff", ".png", ".npy"):
                    if (masks / (p.stem + ext)).is_file():
                        mask = masks / (p.stem + ext)
                        break
                yield p.stem, p, mask


def sources():
    """(ledger, folder, layout, status csv or None, kind of the masks it holds)."""
    lit = AF3 / "plaque_literature_vanilla_queue_2026-09-14"
    gel = AF3 / "gel_doc_2026-10-05"
    out = [
        ("pmc_staged", LIVE / "pmc_staged", "seg", LIVE / "pmc_staged_status.csv", "queue draft"),
        ("pmc_staged_af3", AF3 / "pmc_staged", "seg", AF3 / "pmc_staged_status.csv", "queue draft"),
        ("curate_single", LIVE / "curate_single", "seg", LIVE / "curate_single_status.csv", "queue draft"),
        ("curate_single_af3", AF3 / "curate_single", "seg", AF3 / "curate_single_status.csv", "queue draft"),
        ("curate_legacy", LIVE / "curate_legacy", "seg", None, "legacy bundle, no ledger"),
        ("curate_legacy_af3", AF3 / "curate_legacy", "seg", None, "legacy bundle, no ledger"),
        ("curate_pmc", LIVE / "curate_pmc", "seg", None, "legacy bundle, no ledger"),
        ("curate_pmc_af3", AF3 / "curate_pmc", "seg", None, "legacy bundle, no ledger"),
        ("plaque_r4_review", LIVE / "plaque_r4_review", "seg", LIVE / "plaque_r4_review_status.csv", "queue draft"),
        ("plaque_r4_review_duplicates", LIVE / "plaque_r4_review_duplicates", "seg", LIVE / "plaque_r4_review_status.csv", "queue draft"),
        ("recropped_originals", LIVE / "recropped_originals", "seg", None, "queue draft"),
        ("recropped_originals_af3", AF3 / "recropped_originals", "seg", None, "queue draft"),
        ("gel_doc", gel, "seg", gel / "curate_status.csv", "queue draft"),
        ("gel_doc_replaced_by_reimaged", AF3 / "gel_doc_2026-10-05_set_aside/replaced_by_reimaged/wells", "seg", gel / "curate_status.csv", "queue draft"),
        ("gel_doc_set_aside_plate13", AF3 / "gel_doc_2026-10-05_set_aside/plate13_and_above/wells", "seg", gel / "curate_status.csv", "queue draft"),
        ("literature_holdout", lit, "nested", lit / "curate_status.csv", "vanilla cpsam seed"),
        ("r5_reseed", AF3 / "plaque_r5_reseed_2026-09-21/images", "nested", None, "toxoplasma_plaque_v2 reseed"),
        ("r5_literature_reseed", AF3 / "plaque_r5_literature_reseed_2026-09-21/images", "nested", None, "toxoplasma_plaque_v2 reseed"),
        ("training_r3", NAS_DATA / "plaque/images", "sibling", None, "training label"),
        ("training_seg_r1_r3", NAS_DATA / "_seg_training_data/images", "sibling", None, "training label"),
        ("training_literature", NAS_DATA / "plaque_literature/images", "sibling", None, "literature seed"),
    ]
    return out


def list_sibling(folder: Path):
    masks = folder.parent / "masks"
    for n in sorted(os.listdir(folder)):
        p = folder / n
        if p.suffix.lower() in IMAGE_EXTS:
            m = None
            for ext in (".tif", ".tiff", ".png"):
                if (masks / (p.stem + ext)).is_file():
                    m = masks / (p.stem + ext)
                    break
            yield p.stem, p, m


def mr_inventory(work: Path):
    """Model review segmentation records, cached as a compact CSV."""
    cache = work / "mr_records.csv"
    if not cache.exists():
        wait_for_ram(8)
        say("reading prepared_review.json (one-off)")
        data = json.loads((MR / "prepared_review.json").read_text(encoding="utf-8"))
        with open(cache, "w", newline="") as handle:
            w = csv.writer(handle)
            w.writerow(["id", "review_stem", "review_image", "domain", "eligibility", "group", "shape",
                        "pixel_sha256", "previous_state", "stem", "sources"])
            for r in data["records"]:
                if r["role"] != "segmentation":
                    continue
                srcs = sorted({home_path(a["source"]) for a in r.get("image_versions", [])}
                              | {home_path(a["source"]) for a in r.get("aliases", [])})
                w.writerow([r["id"], r["review_stem"], r["review_image"], r["domain"], r["eligibility"], r["group"],
                            "x".join(map(str, r["shape"])), r["pixel_sha256"], r.get("previous_state", ""),
                            r["stem"], "|".join(srcs)])
        del data
    rows = list(csv.DictReader(open(cache, newline="")))
    n_obj = {}
    with open(MR / "segmentation_predictions.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            n_obj[r["id"]] = int(r["n_objects"] or 0)
    for r in rows:
        r["draft_n"] = n_obj.get(r["id"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="the merged queue folder (the base folder in --variant mode)")
    ap.add_argument("--work", type=Path, help="cache folder for the merged build")
    ap.add_argument("--plan-only", action="store_true", help="write the plan and counts, not the folder")
    ap.add_argument("--exclude-set-aside", action="store_true",
                    help="leave out the Gel Doc plate 13+ set-aside wells (queued by default since 2026-10-12)")
    ap.add_argument("--variant", metavar="NAME",
                    help="build <out>_<NAME>: the same images and curated fields, drafts from --drafts-from")
    ap.add_argument("--drafts-from", type=Path, metavar="DIR", help="draft masks named <stem>.tif (variant mode)")
    ap.add_argument("--draft-model", default=DRAFT_V4, help="model that drew --drafts-from (variant mode)")
    ap.add_argument("--draft-settings", default="", help="settings it ran with (variant mode)")
    ap.add_argument("--make-drafts", type=Path, metavar="DIR",
                    help="WORKSTATION ONLY (GPU): draw a draft for every row of --request into DIR")
    ap.add_argument("--request", type=Path, help="drafts_request.csv (with --make-drafts)")
    ap.add_argument("--af3-root", type=Path, default=Path("/mnt/wd4tb/af3"),
                    help="where af3 is on this machine (/media/carruthers/mnt1/af3 on the workstation)")
    ap.add_argument("--model", default=DRAFT_V4, help="plaque model key or checkpoint (with --make-drafts)")
    ap.add_argument("--diameter", type=float, default=30.0)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.make_drafts:
        if not args.request:
            ap.error("--make-drafts needs --request")
        return make_drafts(args)
    if args.variant:
        if not args.drafts_from:
            ap.error("--variant needs --drafts-from")
        return build_variant(args)
    if not args.work:
        ap.error("--work is required for the merged build")
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    mr_rows = mr_inventory(args.work)
    say("model review fields:", len(mr_rows))
    fields = {}
    by_px = {}
    for r in mr_rows:
        f = dict(id=r["id"], stem=r["review_stem"], image=MR / r["review_image"], domain=r["domain"],
                 eligibility=r["eligibility"], group=r["group"], shape=r["shape"], px=r["pixel_sha256"],
                 previous_state=r["previous_state"], original_stem=r["stem"],
                 sources=set(filter(None, r["sources"].split("|"))), aliases=[],
                 draft=(MR / r["review_image"]).parent / "masks" / (r["review_stem"] + ".tif"),
                 draft_n=r["draft_n"], draft_model=DRAFT_V3, draft_when="2026-10-08T13:19:16",
                 origin_ledger="model_review_2026-10-08", dhash=None)
        fields[f["id"]] = f
        by_px[f["px"]] = f["id"]

    cache = HashCache(args.work / "hash_cache.jsonl")
    entries = []
    for ledger, folder, layout, status, mask_kind in sources():
        listing = list_sibling(folder) if layout == "sibling" else list_fields(folder, layout)
        count = 0
        for stem, image, mask in listing:
            count += 1
            if count % 200 == 0:
                wait_for_ram(8)
            bundle = mask if mask is not None and mask.name.endswith(SEG) else None
            row = None
            if bundle is not None:
                try:
                    row = cache.get(bundle)
                except Exception:
                    bundle = None
            if row is None:
                try:
                    row = cache.get(image)
                except Exception as exc:
                    say("unreadable", image, exc)
                    continue
            px = row["px"]
            fid = by_px.get(px)
            if bundle is not None:
                if fid is None and image != bundle and Path(image).is_file():
                    try:
                        alt = cache.get(image)
                        if alt["shape"][:2] == row["shape"][:2]:
                            fid = by_px.get(alt["px"])
                    except Exception:
                        pass
                    if fid is None:
                        image = bundle
            if fid is None:
                fid = hashlib.sha256(("segmentation" + px).encode()).hexdigest()[:20]
                safe = re.sub(r"[^\w.\-]+", "_", stem)[:120]
                fields[fid] = dict(id=fid, stem=f"{safe}__{fid}", image=image, domain=f"home:{ledger}",
                                   eligibility="new_since_2026-10-08", group="", shape="x".join(map(str, row["shape"])),
                                   px=px, previous_state="", original_stem=stem, sources=set(), aliases=[],
                                   draft=None, draft_n=None, draft_model="", draft_when="",
                                   origin_ledger=ledger, dhash=row["dhash"])
                by_px[px] = fid
            f = fields[fid]
            f["sources"].add(str(image))
            if f["dhash"] is None:
                f["dhash"] = row["dhash"]
            entries.append(dict(ledger=ledger, stem=stem, image=image, mask=mask, id=fid, mask_kind=mask_kind,
                                status=status, layout=layout))
        say(f"{ledger}: {count} fields")
    cache.handle.close()
    stem_index = collections.defaultdict(list)
    entries_by_id = collections.defaultdict(list)
    for e in entries:
        stem_index[(e["ledger"], e["stem"])].append(e)
        entries_by_id[e["id"]].append(e)

    events = collections.defaultdict(list)
    notes = collections.defaultdict(list)
    orphans = []
    seen_status = set()
    for ledger, folder, layout, status, mask_kind in sources():
        if status is None or (status, ledger) in seen_status:
            continue
        seen_status.add((status, ledger))
        for row in read_status(status):
            hits = stem_index.get((ledger, row["stem"]))
            if not hits:
                continue
            e = hits[0]
            events[e["id"]].append(dict(state=row["state"], when=row["when"], n=row["n"], ledger=ledger,
                                        stem=row["stem"], mask=e["mask"], status=str(status)))
    for (status_path, ledger) in list(seen_status):
        for row in read_status(status_path):
            if not any((lg, row["stem"]) in stem_index for (sp, lg) in seen_status if sp == status_path):
                orphans.append(dict(status=str(status_path), stem=row["stem"], state=row["state"]))

    mr_stem = {f["stem"]: f["id"] for f in fields.values()}
    for batch in sorted((NAS_SYNC / MR.name / "segmentation").glob("*/*/batch_*")):
        for row in read_status(batch / "curate_status.csv"):
            fid = mr_stem.get(row["stem"])
            if fid is None:
                orphans.append(dict(status=str(batch / "curate_status.csv"), stem=row["stem"], state=row["state"]))
                continue
            events[fid].append(dict(state=row["state"], when=row["when"], n=row["n"], ledger="model_review_nas",
                                    stem=row["stem"], mask=batch / "masks" / (row["stem"] + ".tif"),
                                    status=str(batch / "curate_status.csv")))
        verdicts = read_verdicts(batch / "csv" / "keep_discard.csv")
        ledger_rows = {r["stem"] for r in read_status(batch / "curate_status.csv")}
        when = mtime_iso(batch / "csv" / "keep_discard.csv") if verdicts else ""
        for stem, v in verdicts.items():
            fid = mr_stem.get(stem)
            if fid is None:
                continue
            notes[fid].append(f"verdict {'keep' if v['keep'] else 'discard'} (model_review_nas)")
            if v["discard"]:
                events[fid].append(dict(state="discard", when=when, n="", ledger="model_review_nas",
                                        stem=stem, mask=None, status=str(batch / "csv/keep_discard.csv")))
            elif v["keep"] and stem not in ledger_rows:
                mask = batch / "masks" / (stem + ".tif")
                events[fid].append(dict(state="done", when=mtime_iso(mask) if mask.exists() else when, n="",
                                        ledger="model_review_nas:keep_verdict", stem=stem, mask=mask,
                                        status=str(batch / "csv/keep_discard.csv")))
    lit_discards = set()
    for ledger, verdict_file in (("pmc_staged", LIVE / "pmc_staged/csv/keep_discard.csv"),
                                 ("literature_holdout", AF3 / "plaque_literature_vanilla_queue_2026-09-14/csv/keep_discard.csv")):
        for stem, v in read_verdicts(verdict_file).items():
            hits = stem_index.get((ledger, stem))
            if not hits:
                continue
            fid = hits[0]["id"]
            notes[fid].append(f"verdict {'keep' if v['keep'] else 'discard'} ({ledger})")
            if v["discard"]:
                if ledger == "literature_holdout":
                    lit_discards.add(fid)
                else:
                    events[fid].append(dict(state="discard", when=mtime_iso(verdict_file), n="", ledger=ledger,
                                            stem=stem, mask=None, status=str(verdict_file)))
    for e in entries:
        if e["ledger"] == "training_r3" and e["mask"] is not None:
            events[e["id"]].append(dict(state="done", when=mtime_iso(e["mask"]), n="", ledger="training_r3 (prebuilt-curated)",
                                        stem=e["stem"], mask=e["mask"], status=str(NAS_DATA / "plaque/curated_status.csv")))

    parent = {fid: fid for fid in fields}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    reasons = {}

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            reasons[(a, b)] = why

    by_any_stem = collections.defaultdict(set)
    for e in entries:
        by_any_stem[e["stem"]].add(e["id"])
    for f in fields.values():
        by_any_stem[f["original_stem"]].add(f["id"])
    dedup_csv = LIVE / "plaque_r4_review_dedup.csv"
    explicit = []
    if dedup_csv.exists():
        for row in csv.DictReader(open(dedup_csv, newline="")):
            for a in by_any_stem.get(row["removed"], ()):
                for b in by_any_stem.get(row["kept"], ()):
                    if a != b:
                        explicit.append((b, a, "plaque_r4_review_dedup.csv"))
    recropped_parents = set()
    for e in entries:
        if e["ledger"].startswith("recropped_originals"):
            recropped_parents.add(e["id"])
    for fid, evs in events.items():
        if any(ev["state"] == "recropped" for ev in evs):
            recropped_parents.add(fid)
    child_re = re.compile(r"^(.*)__r\d\d$")
    for stem, ids in list(by_any_stem.items()):
        m = child_re.match(stem)
        if m:
            for pid in by_any_stem.get(m.group(1), ()):
                recropped_parents.add(pid)
    for e in entries:
        if e["ledger"] == "gel_doc_replaced_by_reimaged":
            m = re.match(r"gel__GRA8__Plate1__(w\d\d)$", e["stem"])
            if m:
                for b in by_any_stem.get(f"gel__GRA8__ReimagedPlate1__{m.group(1)}", ()):
                    explicit.append((b, e["id"], "same physical well re-imaged (GRA8 Plate1 -> ReimagedPlate1)"))

    same_stem = collections.defaultdict(list)
    for f in fields.values():
        same_stem[(f["original_stem"], "x".join(f["shape"].split("x")[:2]))].append(f["id"])
    for (stem, _), members in same_stem.items():
        for other in members[1:]:
            explicit.append((members[0], other, "same stem and size, another encoding"))

    excluded = {}
    for e in entries:
        if e["ledger"] == "gel_doc_set_aside_plate13" and args.exclude_set_aside:
            excluded[e["id"]] = SET_ASIDE_EXCLUDED
    for fid in lit_discards:
        excluded[fid] = LIT_DISCARD_EXCLUDED
    for fid in recropped_parents:
        excluded.setdefault(fid, "recropped parent: its __rNN children are queued instead")

    work_ids = list(fields)
    say("thumbnail hashes for near-duplicates:", len(work_ids))
    meta_path = args.work / "thumb_meta.npz"
    dh, ar, area = {}, {}, {}
    if meta_path.exists():
        z = np.load(meta_path)
        keys = z["ids"].tolist()
        dh = dict(zip(keys, z["dhash"].tolist()))
        ar = dict(zip(keys, z["aspect"].tolist()))
        area = dict(zip(keys, z["area"].tolist()))
    todo = [fid for fid in work_ids if fid not in dh]
    for i, fid in enumerate(todo):
        if i % 500 == 0:
            wait_for_ram(8)
            say(f"  thumbnails {i}/{len(todo)}")
        try:
            gray = to_gray8(read_first_plane(Path(fields[fid]["image"])))
        except Exception as exc:
            say("unreadable", fields[fid]["image"], exc)
            excluded[fid] = f"unreadable image: {exc}"
            continue
        dh[fid] = dhash(gray)
        ar[fid] = gray.shape[1] / max(gray.shape[0], 1)
        area[fid] = int(gray.shape[0] * gray.shape[1])
    if todo:
        keys = list(dh)
        np.savez(meta_path, ids=np.array(keys), dhash=np.array([dh[i] for i in keys], dtype=np.uint64),
                 aspect=np.array([ar[i] for i in keys]), area=np.array([area[i] for i in keys]))
    ids = [fid for fid in work_ids if fid in dh and not fields[fid]["domain"].startswith("gel_doc")
           and not fields[fid]["original_stem"].startswith("gel__")]
    hashes = np.array([dh[i] for i in ids], dtype=np.uint64)
    aspects = np.array([ar[i] for i in ids])
    candidates = []
    for start in range(0, len(ids), 1000):
        bits = np.bitwise_count(hashes[start:start + 1000, None] ^ hashes[None, :])
        asp = np.abs(aspects[start:start + 1000, None] / aspects[None, :] - 1) <= ASPECT_TOL
        rows, cols = np.nonzero((bits <= SIMILAR_BITS) & asp)
        for r, c in zip(rows, cols):
            if start + r < c:
                candidates.append((ids[start + r], ids[c], int(bits[r, c])))
    say("dHash candidate pairs:", len(candidates))
    fine_path = args.work / "fine_features.npz"
    fine = {}
    if fine_path.exists():
        z = np.load(fine_path)
        fine = dict(zip(z["ids"].tolist(), z["feat"]))
    need = sorted({a for a, _, _ in candidates} | {b for _, b, _ in candidates})
    missing_fine = [fid for fid in need if fid not in fine]
    for i, fid in enumerate(missing_fine):
        if i % 500 == 0:
            wait_for_ram(8)
            say(f"  fine features {i}/{len(missing_fine)}")
        fine[fid] = highpass_feature(to_gray8(read_first_plane(Path(fields[fid]["image"]))))
    if missing_fine:
        keys = list(fine)
        np.savez(fine_path, ids=np.array(keys), feat=np.stack([fine[k] for k in keys]))
    near = []
    with open(args.work / "near_duplicate_candidates.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["a", "b", "dhash_bits", "fine_corr", "same_figure", "confirmed"])
        for a, b, bits in candidates:
            corr = float(fine[a].astype(np.float32) @ fine[b].astype(np.float32))
            same = figure_key(fields[a]["original_stem"]) == figure_key(fields[b]["original_stem"])
            ok = corr >= CONFIRM_ANY or (same and corr >= CONFIRM_SAME_FIGURE)
            w.writerow([fields[a]["stem"], fields[b]["stem"], bits, f"{corr:.4f}", same, ok])
            if ok:
                near.append((a, b, f"near-duplicate (dHash {bits} bits, fine high-pass r={corr:.3f}"
                                   f"{', same figure' if same else ''})"))
    for a, b, why in explicit:
        if a in parent and b in parent:
            union(a, b, why)
    for a, b, why in near:
        union(a, b, why)

    def resolve(fid):
        evs = sorted(events.get(fid, []), key=lambda e: (e["when"] or "", e["state"] == "done"))
        decisive = [e for e in evs if e["state"] in ("done", "skip", "discard")]
        latest = decisive[-1] if decisive else None
        dones = [e for e in evs if e["state"] == "done" and e["mask"] is not None and Path(e["mask"]).exists()]
        return latest, (dones[-1] if dones else None)

    groups = collections.defaultdict(list)
    for fid in work_ids:
        groups[find(fid)].append(fid)
    for members in groups.values():
        out_of_queue = [m for m in members if m in excluded]
        if not out_of_queue:
            continue
        first = out_of_queue[0]
        for m in members:
            if m not in excluded and resolve(m)[1] is None:
                excluded[m] = f"copy of excluded {fields[first]['stem']}: {excluded[first]}"
    clusters = collections.defaultdict(list)
    for fid in work_ids:
        if fid in excluded:
            continue
        clusters[find(fid)].append(fid)
    kept = {}
    duplicates = []
    for root, members in clusters.items():
        def rank(fid):
            latest, done = resolve(fid)
            original = not str(fields[fid]["image"]).endswith(SEG)
            return (done is not None, done["when"] if done else "", original,
                    fields[fid]["origin_ledger"] == "model_review_2026-10-08", area.get(fid, 0),
                    len(fields[fid]["sources"]), fid)
        best = max(members, key=rank)
        kept[best] = members
        for m in members:
            if m != best:
                why = reasons.get((best, m)) or reasons.get((m, best)) or "same cluster"
                duplicates.append((best, m, why))

    say("unique fields:", len(kept), "duplicates folded:", len(duplicates), "excluded:", len(excluded))

    plan = []
    for fid in sorted(kept, key=lambda i: fields[i]["stem"]):
        f = fields[fid]
        group_events = [e for m in kept[fid] for e in events.get(m, [])]
        latest, done = resolve(fid)
        if done is None:
            for m in kept[fid]:
                _, d = resolve(m)
                if d is not None:
                    notes[fid].append(f"duplicate {fields[m]['stem']} has a curated mask on another pixel grid")
        state = latest["state"] if latest else ""
        if state == "discard":
            state = "skip"
        mask_src, origin, model, curated, when, n = None, "none", "", "no", "", None
        if done is not None:
            mask_src, origin, curated, when = Path(done["mask"]), f"curated:{done['ledger']}", "yes", done["when"]
            model = "human"
        elif f["draft"] is not None and Path(f["draft"]).exists():
            mask_src, origin, model, when, n = Path(f["draft"]), "draft:model_review_2026-10-08", DRAFT_V3, f["draft_when"], f["draft_n"]
        else:
            for e in entries_by_id.get(fid, ()):
                if e["mask"] is not None and Path(e["mask"]).exists() and n_objects(read_mask(Path(e["mask"]))) > 0:
                    mask_src, origin, model = Path(e["mask"]), f"draft:{e['ledger']}", e["mask_kind"]
                    when = mtime_iso(Path(e["mask"]))
                    break
        if state == "done" and done is None:
            state = ""
            notes[fid].append("ledger says done but its mask file is missing; left uncurated")
        plan.append(dict(fid=fid, state=state, curated=curated if state != "skip" else "skip", mask_src=mask_src,
                         origin=origin, model=model, when=when, n=n, latest=latest, members=kept[fid],
                         events=group_events))
    json.dump(dict(fields=len(fields), kept=len(kept), duplicates=len(duplicates), excluded=len(excluded),
                   orphans=len(orphans)), open(args.work / "plan_summary.json", "w"), indent=1)
    with open(args.work / "orphan_status_rows.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["status", "stem", "state"])
        w.writeheader()
        w.writerows(orphans)
    if args.plan_only:
        c = collections.Counter((p["curated"], p["origin"].split(":")[0]) for p in plan)
        say("plan:", dict(c))
        say("excluded reasons:", dict(collections.Counter(excluded.values())))
        say("duplicate reasons:", dict(collections.Counter(w.split(" (")[0] for _, _, w in duplicates)))
        return 0

    out = args.out
    (out / "masks").mkdir(parents=True, exist_ok=True)
    out_dev = out.stat().st_dev
    status_rows, score_rows, draft_rows, manifest_rows, missing = [], [], [], [], []
    prob_lookup = {}
    for scores in (LIVE / "pmc_staged_scores.csv", LIVE / "curate_single_scores.csv"):
        ledger = "pmc_staged" if "pmc" in scores.name else "curate_single"
        if scores.exists():
            for row in csv.DictReader(open(scores, newline="")):
                for e in stem_index.get((ledger, row["stem"]), ()):
                    try:
                        prob_lookup[e["id"]] = float(row["plaque_prob"])
                    except (TypeError, ValueError):
                        pass
    for i, p in enumerate(plan):
        if i % 500 == 0:
            wait_for_ram(8)
            say(f"  writing {i}/{len(plan)}")
        f = fields[p["fid"]]
        src = Path(f["image"])
        if src.name.endswith(SEG):
            ext = ".tif"
        else:
            ext = src.suffix.lower()
        dest = out / (f["stem"] + ext)
        if not dest.exists():
            if src.name.endswith(SEG):
                x = read_first_plane(src)
                tifffile.imwrite(dest, x, photometric="rgb" if x.ndim == 3 else "minisblack", compression="zlib")
            elif src.stat().st_dev == out_dev:
                os.link(src, dest)
            else:
                shutil.copy2(src, dest)
        mdest = out / "masks" / (f["stem"] + ".tif")
        n = p["n"]
        if p["mask_src"] is not None:
            if not mdest.exists():
                if p["mask_src"].suffix == ".tif" and not p["mask_src"].name.endswith(SEG) and n is not None:
                    shutil.copyfile(p["mask_src"], mdest)
                else:
                    m = read_mask(p["mask_src"])
                    shp = "x".join(map(str, m.shape))
                    if m.shape[:2] != tuple(int(v) for v in f["shape"].split("x")[:2]):
                        notes[p["fid"]].append(f"mask {p['mask_src']} shape {shp} != image {f['shape']}; not used")
                        p["mask_src"], p["origin"], p["model"] = None, "none", ""
                        if p["curated"] == "yes":
                            p["curated"], p["state"] = "no", ""
                        if f["draft"] is not None and Path(f["draft"]).exists():
                            shutil.copyfile(f["draft"], mdest)
                            p["mask_src"], p["origin"], p["model"] = Path(f["draft"]), "draft:model_review_2026-10-08", DRAFT_V3
                            n = f["draft_n"]
                    else:
                        dtype = np.uint16 if m.max() < 65535 else np.uint32
                        tifffile.imwrite(mdest, m.astype(dtype), compression="zlib")
                        n = n_objects(m)
            if n is None and mdest.exists():
                n = n_objects(tifffile.imread(mdest))
        if p["mask_src"] is None:
            missing.append(dict(stem=f["stem"], image=str(dest), domain=f["domain"], source=str(src)))
            n = 0
        if p["state"] in ("done", "skip"):
            when = (p["latest"] or {}).get("when") or p["when"] or ""
            status_rows.append(dict(stem=f["stem"], state=p["state"], n_objects=n, updated=when))
        prob = prob_lookup.get(p["fid"])
        if prob is None:
            prob = next((v for k, v in DOMAIN_PROB if f["domain"].startswith(k)), 0.5)
        score_rows.append(dict(stem=f["stem"], prob=f"{prob:.4f}"))
        draft_rows.append(dict(stem=f["stem"], n_objects=n))
        dup_sources = sorted({s for m in p["members"] for s in fields[m]["sources"]} | {str(fields[m]["image"]) for m in p["members"]})
        manifest_rows.append(dict(
            stem=f["stem"], id=p["fid"], image=dest.name, domain=f["domain"],
            previous_eligibility=f["eligibility"], group=f["group"], shape=f["shape"], pixel_sha256=f["px"],
            source_path=str(src), curated=p["curated"], state=p["state"] or "pending",
            curation_date=(p["when"] if p["curated"] == "yes" else ""), mask_origin=p["origin"],
            mask_model=p["model"], mask_source=str(p["mask_src"] or ""), n_objects=n,
            n_copies=len(dup_sources), all_sources="|".join(dup_sources),
            ledger_events="|".join(sorted(f"{e['when']} {e['state']} {e['ledger']}" for e in p["events"])),
            notes="; ".join(dict.fromkeys(notes.get(p["fid"], [])))))

    def write(name, rows, fields_=None):
        with open(out / name, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=fields_ or list(rows[0]))
            w.writeheader()
            w.writerows(r for r in rows if any(str(v) for v in r.values()))

    existing = {}
    status_file = out / "curate_status.csv"
    if status_file.exists():
        for row in csv.DictReader(open(status_file, newline="")):
            existing[row["stem"]] = row
    merged = {r["stem"]: r for r in status_rows}
    for stem, row in existing.items():
        old, new = row, merged.get(stem)
        if new is None or (old.get("updated") or "") > (new.get("updated") or ""):
            merged[stem] = old
    write("curate_status.csv", sorted(merged.values(), key=lambda r: r["stem"]), ["stem", "state", "n_objects", "updated"])
    write("curate_scores.csv", score_rows, ["stem", "prob"])
    write("curate_drafts.csv", draft_rows, ["stem", "n_objects"])
    write("manifest.csv", manifest_rows)
    write("duplicates.csv", [dict(kept_stem=fields[a]["stem"], duplicate_id=b, duplicate_stem=fields[b]["stem"],
                                  duplicate_source=str(fields[b]["image"]), duplicate_all_sources="|".join(sorted(fields[b]["sources"])),
                                  duplicate_curated="yes" if resolve(b)[1] else "no", reason=why)
                             for a, b, why in duplicates] or [dict(kept_stem="", duplicate_id="", duplicate_stem="",
                                                                   duplicate_source="", duplicate_all_sources="", duplicate_curated="", reason="")])
    write("excluded.csv", [dict(id=fid, stem=fields[fid]["stem"], source=str(fields[fid]["image"]),
                                curated="yes" if resolve(fid)[1] else "no", reason=why)
                           for fid, why in sorted(excluded.items(), key=lambda kv: fields[kv[0]]["stem"])])
    write("missing_drafts.csv", missing or [dict(stem="", image="", domain="", source="")], ["stem", "image", "domain", "source"])
    summary = dict(
        total=len(manifest_rows),
        curated=sum(r["curated"] == "yes" for r in manifest_rows),
        skip=sum(r["state"] == "skip" for r in manifest_rows),
        uncurated_with_draft=sum(r["curated"] == "no" and r["mask_origin"] != "none" for r in manifest_rows),
        uncurated_without_draft=sum(r["curated"] == "no" and r["mask_origin"] == "none" for r in manifest_rows),
        duplicates=len(duplicates), excluded=len(excluded),
        by_domain=dict(collections.Counter(r["domain"] for r in manifest_rows)),
        curated_by_ledger=dict(collections.Counter(r["mask_origin"] for r in manifest_rows if r["curated"] == "yes")),
        duplicate_reasons=dict(collections.Counter(w.split(" (")[0] for _, _, w in duplicates)),
        excluded_reasons=dict(collections.Counter(excluded.values())))
    (out / "build_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    by_source = collections.Counter()
    for e in entries:
        by_source[e["ledger"]] += 1
    write_drafts_request(out, manifest_rows)
    write_readme(out, summary, by_source, len(fields))
    say(json.dumps(summary, indent=1))
    return 0


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_drafts_request(out: Path, manifest_rows) -> Path:
    """Every field still waiting for a human, for the workstation to draft."""
    status = {}
    if (out / "curate_status.csv").exists():
        for row in csv.DictReader(open(out / "curate_status.csv", newline="")):
            status[row["stem"]] = (row.get("state") or "").strip().lower()
    path = out / "drafts_request.csv"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["stem", "image", "sha256", "shape"])
        for r in manifest_rows:
            if status.get(r["stem"]) in ("done", "skip", "recropped"):
                continue
            image = out / r["image"]
            w.writerow([r["stem"], str(image.relative_to(AF3_ROOT)), file_sha256(image), r["shape"]])
    os.replace(tmp, path)
    return path


DRAFT_NAMES = ("{stem}.tif", "masks/{stem}.tif", "{stem}_cp_masks.tif", "{stem}_cp_masks.png",
               "{stem}_seg.npy", "{stem}.npy", "{stem}.png")


def find_draft(folder: Path, stem: str):
    for pattern in DRAFT_NAMES:
        p = folder / pattern.format(stem=stem)
        if p.is_file():
            return p
    return None


def build_variant(args) -> int:
    """A sibling of the merged queue: same images, same decided fields, other drafts.

    Decided fields (done or skip in the base folder's curate_status.csv) keep
    their row and their mask, copied verbatim. Every other field takes its
    mask from --drafts-from. A field already decided inside the variant keeps
    its own mask and row: nothing curated anywhere is overwritten.
    """
    base = args.out
    variant = base.parent / f"{base.name}_{args.variant}"
    (variant / "masks").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(base / "manifest.csv", newline="", encoding="utf-8")))
    base_status = {r["stem"]: r for r in csv.DictReader(open(base / "curate_status.csv", newline=""))}
    var_status = {}
    if (variant / "curate_status.csv").exists():
        var_status = {r["stem"]: r for r in csv.DictReader(open(variant / "curate_status.csv", newline=""))}
    run_info = {}
    for name in ("drafts_run.json",):
        if (args.drafts_from / name).exists():
            run_info = json.loads((args.drafts_from / name).read_text(encoding="utf-8"))
    model = run_info.get("model") or args.draft_model
    settings = run_info.get("settings_text") or args.draft_settings
    decided = {"done", "skip", "recropped"}
    status_out, drafts_out, manifest_out, missing = {}, [], [], []
    for i, r in enumerate(rows):
        if i % 1000 == 0:
            wait_for_ram(8)
            say(f"  variant {args.variant}: {i}/{len(rows)}")
        stem = r["stem"]
        src_img = base / r["image"]
        dst_img = variant / r["image"]
        if not dst_img.exists():
            if src_img.stat().st_dev == variant.stat().st_dev:
                os.link(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)
        dst_mask = variant / "masks" / f"{stem}.tif"
        own = var_status.get(stem)
        theirs = base_status.get(stem)
        row = dict(r)
        if own and (own.get("state") or "").lower() in decided:
            status_out[stem] = own
            origin, mmodel = "curated in this variant", "human"
        elif theirs and (theirs.get("state") or "").lower() in decided:
            src_mask = base / "masks" / f"{stem}.tif"
            if src_mask.exists():
                shutil.copyfile(src_mask, dst_mask)
            status_out[stem] = theirs
            origin, mmodel = r["mask_origin"] if r["curated"] == "yes" else f"base folder ({theirs['state']})", r["mask_model"]
        else:
            if own:
                status_out[stem] = own
            draft = find_draft(args.drafts_from, stem)
            origin, mmodel = "none", ""
            if dst_mask.exists():
                origin, mmodel = f"draft:{args.variant}", model
            elif draft is not None:
                m = read_mask(draft)
                h, w = (int(v) for v in r["shape"].split("x")[:2])
                if m.shape[:2] != (h, w):
                    say(f"  draft {draft} shape {m.shape} != image {r['shape']}; skipped")
                elif draft.suffix == ".tif":
                    shutil.copyfile(draft, dst_mask)
                    origin, mmodel = f"draft:{args.variant}", model
                else:
                    tifffile.imwrite(dst_mask, m.astype(np.uint16 if m.max() < 65535 else np.uint32), compression="zlib")
                    origin, mmodel = f"draft:{args.variant}", model
            if origin == "none":
                missing.append(dict(stem=stem, image=r["image"], domain=r["domain"], source=r["source_path"]))
        n = n_objects(tifffile.imread(dst_mask)) if dst_mask.exists() else 0
        drafts_out.append(dict(stem=stem, n_objects=n, model=mmodel, settings=settings if mmodel == model else "",
                               mask_origin=origin))
        row.update(mask_origin=origin, mask_model=mmodel, n_objects=n,
                   mask_source=str(base / "masks" / f"{stem}.tif") if mmodel == "human" or origin.startswith("curated") else str(args.drafts_from))
        manifest_out.append(row)

    def write(name, data, fieldnames):
        with open(variant / name, "w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)
    write("curate_status.csv", sorted(status_out.values(), key=lambda x: x["stem"]), ["stem", "state", "n_objects", "updated"])
    write("curate_drafts.csv", drafts_out, ["stem", "n_objects", "model", "settings", "mask_origin"])
    write("manifest.csv", manifest_out, list(manifest_out[0]))
    write("missing_drafts.csv", missing, ["stem", "image", "domain", "source"])
    shutil.copyfile(base / "curate_scores.csv", variant / "curate_scores.csv")
    for name in ("duplicates.csv", "excluded.csv"):
        if (base / name).exists():
            shutil.copyfile(base / name, variant / name)
    summary = dict(variant=args.variant, base=str(base), drafts_from=str(args.drafts_from), model=model,
                   settings=settings, total=len(manifest_out),
                   decided=collections.Counter((x.get("state") or "").lower() for x in status_out.values()),
                   with_variant_draft=sum(d["mask_origin"].startswith("draft:") for d in drafts_out),
                   without_mask=len(missing), drafts_run=run_info)
    (variant / "build_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    (variant / "README.txt").write_text(f"""PLAQUE ALL, {args.variant} drafts (built {dt.date.today().isoformat()})

The same {len(manifest_out)} images as {base.name} (hard links on Home; Syncthing
stores them as separate copies on the workstation), the same decided fields
(done/skip rows and their masks copied verbatim from {base.name}), and for every
other field a draft drawn by {model} with: {settings or 'see drafts_run.json'}.

OPEN IT
  spacr-make-masks --folder {variant}

  decided fields          {sum(summary['decided'].values())} ({dict(summary['decided'])})
  fields with this draft  {summary['with_variant_draft']}
  fields with no mask     {summary['without_mask']}   (missing_drafts.csv)

Curate in ONE version at a time. Rebuilding this variant (same command) carries
over everything decided in {base.name} since; a field decided here keeps its own
mask and row. curate_drafts.csv records the model and settings of every draft;
manifest.csv is {base.name}'s with the mask columns of this variant.

Rebuild: python tools/build_plaque_all_queue.py --out {base} --variant {args.variant} \\
    --drafts-from {args.drafts_from}
""", encoding="utf-8")
    say(json.dumps(summary, indent=1, default=str))
    return 0


def make_drafts(args) -> int:
    """Workstation only: draw one plaque draft per requested field, on the GPU.

    Resumable: a field whose draft already exists is not drawn again. Each
    image is checked against the sha256 in the request before it is used.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spacr import plaque, submodules

    out = args.make_drafts
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.request, newline="")))
    checkpoint = submodules._resolve_plaque_model({"plaque_model": args.model})
    model = submodules._plaque_cellpose_model(checkpoint)
    settings = {"diameter": args.diameter, "flow_threshold": args.flow_threshold, "CP_prob": args.cellprob_threshold}
    settings_text = (f"diameter={args.diameter:g}, flow_threshold={args.flow_threshold:g}, "
                     f"cellprob_threshold={args.cellprob_threshold:g} (spacr.plaque.segment_plaque_image)")
    try:
        import cellpose
        cp_version = getattr(cellpose, "__version__", "")
    except Exception:
        cp_version = ""
    info = dict(model=args.model, checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(Path(checkpoint)),
                settings=settings, settings_text=settings_text, cellpose=cp_version, request=str(args.request),
                started=dt.datetime.now().isoformat(timespec="seconds"), drawn=0, skipped_existing=0, errors=[])
    for i, r in enumerate(rows):
        dest = out / f"{r['stem']}.tif"
        if dest.exists():
            info["skipped_existing"] += 1
            continue
        image_path = args.af3_root / r["image"]
        try:
            if r.get("sha256") and file_sha256(image_path) != r["sha256"]:
                raise ValueError("image sha256 differs from the request")
            labels = plaque.segment_plaque_image(model, submodules._plaque_imread(str(image_path)), settings)
            labels = np.asarray(labels)
            tmp = dest.with_suffix(".tmp.tif")
            tifffile.imwrite(tmp, labels.astype(np.uint16 if labels.max() < 65535 else np.uint32), compression="zlib")
            os.replace(tmp, dest)
            info["drawn"] += 1
        except Exception as exc:
            info["errors"].append(dict(stem=r["stem"], error=repr(exc)))
        if i % 200 == 0:
            say(f"  drafts {i}/{len(rows)}: {info['drawn']} drawn, {len(info['errors'])} errors")
            (out / "drafts_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    info["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    (out / "drafts_run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    say(f"drafts done: {info['drawn']} drawn, {info['skipped_existing']} already there, {len(info['errors'])} errors")
    return 0 if not info["errors"] else 1


def write_readme(out: Path, s: dict, by_source, n_inventory: int) -> None:
    def table(counter):
        return "\n".join(f"  {v:>6}  {k}" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]))
    text = f"""PLAQUE ALL -- one curation queue for plaque v5 (built {dt.date.today().isoformat()})
=====================================================================

Every unique plaque well/figure image known on Home and WS, with no hold-out:
the maintainer's "all of the plaques, no holdout" set for toxoplasma_plaque v5
(trained from cpsam_v2 on everything). Built by tools/build_plaque_all_queue.py
in the spaCR repository; rebuild with the same command (curation done in this
folder is kept: existing masks are never overwritten and newer status rows win).

OPEN IT
  spacr-make-masks --folder {out}
  spacr-make-masks --folder {out} --dry-run          # counts and order, no window
  spacr-make-masks --folder {out} --limit 50          # a bounded session

  The default order is `easy` (curate_scores.csv): populated drafts first, then
  by prob (Gel Doc > earlier curation/NAS > Toxoplasma literature > literature
  wells > other species), then empty ones. Every save is recorded as done in
  curate_status.csv here, so the session resumes on either machine.

  Curated fields are already `done` (or `skip`) in curate_status.csv, so they are
  not offered by default. To revisit one, delete its row from curate_status.csv
  (it is offered again with its curated mask), or open this folder with Make
  Masks' Open folder, which shows every field.

COUNTS
  unique fields            {s['total']}
  curated (done)           {s['curated']}
  skip (unusable/discard)  {s['skip']}
  uncurated with a draft   {s['uncurated_with_draft']}
  uncurated, no mask       {s['uncurated_without_draft']}   (missing_drafts.csv: WS drafts them with toxoplasma_plaque_v4 on GPU)
  duplicates folded        {s['duplicates']}   (duplicates.csv)
  excluded                 {s['excluded']}   (excluded.csv)

  By domain (model-review domain, or home:<source> for fields new since 2026-10-08):
{table(s['by_domain'])}

  Curated masks by ledger:
{table(s['curated_by_ledger'])}

  Duplicates by reason:
{table(s['duplicate_reasons'])}

  Excluded by reason:
{table(s['excluded_reasons'])}

SOURCES MERGED ({n_inventory} distinct pixel sets before near-duplicate folding)
  model_review_2026-10-08 segmentation inventory (17,888 pixel-unique fields with
  toxoplasma_plaque_v3 drafts; it already covers Gel Doc, every curation queue,
  the training sets, NAS caches, PMC/web literature and other-pathogen archives),
  plus these Home folders, hashed with the same canonical pixel digest (files seen):
{table(by_source)}
  Curation records read: pmc_staged_status.csv (live firecuda copy preferred; the
  af3 copy is older), curate_single_status.csv, plaque_r4_review_status.csv,
  gel_doc_2026-10-05/curate_status.csv, the literature hold-out's curate_status.csv
  and keep_discard.csv, the model review's Gel Doc batches on the NAS sync copy
  (curate_status.csv and keep verdicts), and the round-3 training labels.

MASK PRIORITY
  human-curated (latest done across all ledgers; a later skip/Discard wins) >
  newest model draft (model review, toxoplasma_plaque_v3, 2026-10-08) > older
  queue drafts > none. No toxoplasma_plaque_v4 drafts existed at Home. Masks are
  copies (masks/<stem>.tif); images are hard links to the originals (af3) or copies.
  Originals are untouched.

FILES
  manifest.csv          one row per field: source path, every copy, curated yes/no,
                        curation date, mask origin, drawing model, ledger history
  duplicates.csv        copies folded into a kept field, and why
  excluded.csv          fields left out on purpose, and why
  missing_drafts.csv    fields with no mask at all
  curate_status.csv     Make Masks' resume record (stem,state,n_objects,updated)
  curate_scores.csv     stem,prob for --order easy / prob
  curate_drafts.csv     cached draft object counts
  build_summary.json    the counts above
"""
    (out / "README.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
