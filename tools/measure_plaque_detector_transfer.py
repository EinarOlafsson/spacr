#!/usr/bin/env python3
"""Does the plaque-assay WELL detector transfer from plate photographs to published figures?

Instruction 424 wants plaque measurements taken out of other people's papers,
and its first open question is the one this harness answers and nothing else:

    "The YOLO detector finds WELLS in plate photographs. Finding plaque images
    inside a published figure is a different distribution -- cropped, rescaled,
    JPEG-compressed, often greyscale. Does it transfer, or does it need its own
    training round? Measure this before building on it."

This is a TRANSFER SPIKE, not a scorecard. It samples open-access papers, pulls
their figures, runs whichever detectors the model zoo is asked for, and draws
numbered boxes on every figure so a person can look and say which boxes landed
on a plaque image and which plaque images were missed. The scoring stage reads
that hand-written verdict back and turns it into precision and recall. Nothing
here decides anything by itself: no detection is scored by another model, and
no label is inferred from a caption.

WHAT COUNTS AS A TRUE REGION. One plaque-assay image -- a whole plate, one
well, one dish, or one stained monolayer field -- is one region. A figure that
shows six wells shows six regions, because six is what a downstream measurement
would have to separate. Graphs of plaque number or area are not regions: the
module needs the IMAGE. A box is a true positive when it lands on one region
and is the first box to do so; a second box on the same region is a duplicate
and is counted against precision, because a duplicate becomes a duplicated
measurement row downstream.

WHY THE SAMPLE MUST CONTAIN NEGATIVES. Precision measured on figures that are
all plaque figures is not precision. Every figure of every sampled paper is
kept -- blots, graphs, schematics, fluorescence panels -- and the ones with no
plaque image in them are where the false boxes show up. The v4 detector's own
test set is 84 no-well images out of 129 for the same reason.

WHY THE SAMPLE MUST EXCLUDE THE TRAINING PAPERS. The current zoo detector
(``yolo_welldetect_v4.pt``) was trained on 939 reviewed PMC figures, split by
PMC article. Measuring it on a paper it trained on measures nothing, so
``--exclude-pmcids`` takes the published split file and every PMC id in it is
refused at sampling time. The older ``yolo_welldetect_v3.pt`` saw plate
photographs only, which is why it is the one that answers the transfer
question as 424 asks it; v4 answers the question a builder actually has, which
is whether the CURRENT zoo detector is good enough to build on.

LICENCE. Only the open-access subset is fetched, and ``--licences`` defaults to
the CC-BY family: the maintainer's 2026-09-19 decision is that derived
measurements may be stored for any paper but image crops only for CC-BY ones,
and this harness stores crops (the overlays) for everything it samples, so it
samples nothing else. Every record carries its licence, DOI and PMC id, and
images are deduplicated by sha256 so the same figure in a preprint and in the
published version is not counted twice.

STAGES, each writing into ``--out`` and each resumable:

    sample    Europe PMC search -> papers.json      (network, no images)
    figures   full-text XML + supplementary zip -> figures/ and figures.json
    detect    model zoo -> detections.json and overlays/
    score     labels.json (written BY A PERSON) -> the result table

Usage::

    python tools/measure_plaque_detector_transfer.py --out /tmp/spike \\
        --stages sample,figures,detect \\
        --papers 20 --exclude-pmcids split.csv \\
        --models toxoplasma_well_detector_v1,toxoplasma_well_detector_v2

    (look at overlays/*.png, write labels.json)

    python tools/measure_plaque_detector_transfer.py --out /tmp/spike \\
        --stages score --labels /tmp/spike/labels.json

CHANNEL ORDER IS PART OF THE MEASUREMENT, NOT A DETAIL. Ultralytics decodes a
file path with OpenCV and therefore trains and infers in BGR; handed a numpy
array it assumes the caller did the same. The first run of this harness passed
the RGB array it had built for the overlays, so every figure was detected with
red and blue swapped, and the result understated the detector badly. This
harness used to correct that itself, in a ``_detector_input`` of its own;
since instruction 445 the correction lives in
:func:`spacr.plaque.detect_wells`, which takes RGB and converts, so THIS FILE
MUST HAND IT RGB AND NOT CONVERT -- two conversions are the original bug
again. ``detections.json`` records ``channel_order`` as what the detector
sees, so a result file says which question it answered.

``detect`` needs ``ultralytics``, which spaCR does not install by default:
``pip install "spacr[plaque]"``. It hides the GPU from itself unless ``--gpu``
is passed -- yolo11n and yolo26n over a few hundred figures is about a minute
of CPU, and a measurement that small has no business queueing behind whatever
is training.

TWO THINGS THIS HARNESS NOW DOES TO ITSELF, BOTH BECAUSE IT DID NOT AND THAT
COST SOMEBODY SOMETHING:

* IT KEEPS ULTRALYTICS OUT OF THE USER'S CONFIGURATION. Importing
  ``ultralytics`` writes ``~/.config/Ultralytics/settings.json``, and on
  2026-09-20 that import announced "Ultralytics settings reset to default
  values" over a file dating from July, with no backup. A measurement has no
  business editing the configuration of the machine it runs on, so ``detect``
  points ``YOLO_CONFIG_DIR`` at its own output directory before the detector
  is loaded. ``--yolo-config-dir`` overrides it; pointing it at the real one
  is possible and has to be typed.
* IT RECORDS WHICH COPY OF SPACR IT MEASURED. Run as
  ``python tools/measure_plaque_detector_transfer.py``, ``sys.path[0]`` is
  ``tools/`` and not the working directory, so ``import spacr`` resolves
  through whatever the editable install points at -- which, from a worktree,
  is the other tree. The first attempt at the 2026-09-20 rerun scored the
  shared checkout while sitting in a worktree. ``detections.json`` now carries
  ``spacr_from``, so a result file says which code produced it.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import requests

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = {"User-Agent": "spacr-plaque-transfer-spike/1.0 "
                    "(https://github.com/EinarOlafsson/spacr)"}

DEFAULT_QUERY = (
    '("plaque assay" OR "plaque assays" OR "plaque formation" OR "plaque size" '
    'OR "plaque number" OR "plaquing efficiency") '
    'AND (HAS_FT:Y AND OPEN_ACCESS:Y AND IN_EPMC:Y)'
)

DEFAULT_LICENCES = ("cc by", "cc by 4.0", "cc-by", "cc0")

DEFAULT_MODELS = ("toxoplasma_well_detector_v1", "toxoplasma_well_detector_v2")

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".bmp", ".webp"}

FMT_RANK = {".tiff": 0, ".tif": 0, ".png": 1, ".webp": 2, ".jpg": 2,
            ".jpeg": 2, ".gif": 3, ".bmp": 3}

BOX_COLOURS = ((255, 40, 40), (0, 160, 255), (40, 220, 40), (255, 170, 0))

PMCID_RE = re.compile(r"PMC\d+")

#: Which rule picked the files already sitting in a figure directory.
#: Written into ``.selection`` beside them and checked before they are reused,
#: because the format-first rule this harness started with produced 200 px
#: thumbnails and a plain "there are files here already" cache would have
#: served those thumbnails to every later run.
SELECTION_RULE = "size-first-then-format-rank/1"

#: What a person is being asked when they fill in a label file.
PROTOCOL = (
    "regions: how many plaque-assay IMAGE regions the figure shows, counted "
    "from the figure itself, split into 'well' (a round culture surface "
    "photographed whole -- a well, a dish, a plate) and 'other' (a plaque "
    "image that is not round: a lawn strip, a dilution row, a stained "
    "monolayer field). A region is one surface a measurement would have to "
    "separate from its neighbours; whether plaques happen to be countable in "
    "it does not change that it is one. A GRAPH of plaque counts or areas is "
    "not a region. "
    "box_verdicts, one per box, in detections.json order: tp = lands on a "
    "region and is the first box to do so, dup = a later box on a region "
    "already claimed, fp = anything else. "
    "found: how many distinct regions of each kind the model put at least "
    "one box on.")


def _get(url: str, **kwargs: Any) -> requests.Response:
    """One polite GET with the harness user agent.

    :param url: the address to fetch.
    :param kwargs: passed to :func:`requests.get`.
    :returns: the response, whatever its status.
    """
    kwargs.setdefault("headers", UA)
    kwargs.setdefault("timeout", 120)
    return requests.get(url, **kwargs)


def search_papers(query: str, want: int, licences: Sequence[str],
                  exclude: set, seed: int, pool: int,
                  sleep: float = 0.35) -> List[Dict[str, Any]]:
    """Europe PMC hits that may be sampled, shuffled and cut to ``want``.

    The pool is read in publication order as Europe PMC returns it, then
    SHUFFLED with a fixed seed before the cut, so the sample is not the twenty
    most recent papers about plaques -- which would be twenty papers from the
    same months and, in this field, largely the same journals.

    :param query: Europe PMC query string.
    :param want: how many papers to return.
    :param licences: licence strings that may be sampled, lowercased.
    :param exclude: PMC ids that may not be sampled.
    :param seed: shuffle seed.
    :param pool: how many hits to read before shuffling.
    :param sleep: seconds between pages.
    :returns: the sampled records.
    """
    rows: List[Dict[str, Any]] = []
    cursor = "*"
    seen = 0
    while seen < pool:
        response = _get(f"{EPMC}/search", params={
            "query": query, "format": "json", "pageSize": 100,
            "cursorMark": cursor, "resultType": "core"})
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("resultList", {}).get("result", [])
        if not hits:
            break
        for hit in hits:
            seen += 1
            pmcid = hit.get("pmcid")
            licence = (hit.get("license") or "").strip().lower()
            if not pmcid or pmcid in exclude:
                continue
            if hit.get("isOpenAccess") != "Y":
                continue
            if licence not in licences:
                continue
            rows.append({
                "pmcid": pmcid,
                "pmid": hit.get("pmid"),
                "doi": hit.get("doi"),
                "title": hit.get("title"),
                "journal": ((hit.get("journalInfo") or {})
                            .get("journal", {}) or {}).get("title"),
                "year": hit.get("pubYear"),
                "licence": licence,
            })
        nxt = payload.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(sleep)
    random.Random(seed).shuffle(rows)
    return rows[:want] if want > 0 else rows


def figure_captions(pmcid: str) -> Dict[str, Dict[str, str]]:
    """Figure label and caption for each graphic file named in the JATS XML.

    :param pmcid: the article, ``PMC`` included.
    :returns: ``{graphic stem: {"label": ..., "caption": ...}}``; empty when no
        full text is deposited, which is not fatal -- the supplementary bundle
        may still carry the figures, they are simply uncaptioned.
    """
    response = _get(f"{EPMC}/{pmcid}/fullTextXML")
    if response.status_code != 200 or not response.content:
        return {}
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for fig in root.iter("fig"):
        label = (fig.findtext("label") or "").strip()
        caption_el = fig.find("caption")
        caption = ""
        if caption_el is not None:
            caption = " ".join("".join(caption_el.itertext()).split())
        for graphic in fig.iter("graphic"):
            href = (graphic.get("{http://www.w3.org/1999/xlink}href")
                    or graphic.get("href") or "")
            if href:
                out[Path(href).stem] = {"label": label, "caption": caption}
    return out


def fetch_figure_images(pmcid: str, dest: Path,
                        sleep: float = 0.4) -> List[Path]:
    """Download one article's image files, one format per figure.

    The route is the one the plaque corpus was built with: Europe PMC's
    ``supplementaryFiles`` endpoint returns a zip that holds the MAIN figures
    as well as the true supplements. The NCBI bulk OA ``.tar.gz`` and the
    europepmc.org render backend both refuse this traffic.

    Journals ship the same figure in several formats, so one file per stem
    survives -- THE BIGGEST ONE, by uncompressed bytes, with the format rank
    only breaking ties. Ranking by format alone is a trap this harness fell
    into and measured through: PLOS ships ``g001.webp`` at full size beside
    ``g001.gif`` at 200 px, so a format-first rule that had never heard of
    webp handed the detector a thumbnail of every PLOS figure. Size first is
    also what the measurement needs: the question is whether the detector
    works on a published figure, not on a preview of one.

    Files already in ``dest`` are reused only when ``.selection`` beside them
    names the rule above. Anything else is refetched and the old files are
    deleted once the new bundle has parsed, because a cache keyed on "there
    are images here" is how the format-first thumbnails survived their own fix
    -- and rerunning the ``figures`` stage into a populated directory is
    exactly what the queued retraining job asks the next person to do.

    :param pmcid: the article, ``PMC`` included.
    :param dest: directory for this article's images; created if missing.
    :param sleep: seconds to wait after a download.
    :returns: the image paths, sorted by name. Empty when the article has no
        bundle, which happens and is recorded rather than retried.
    """
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".selection"
    have = sorted(p for p in dest.glob("*") if p.suffix.lower() in IMG_EXT)
    if have and marker.is_file():
        try:
            if marker.read_text().strip() == SELECTION_RULE:
                return have
        except OSError:
            pass
    try:
        response = _get(f"{EPMC}/{pmcid}/supplementaryFiles", timeout=300)
    except requests.RequestException:
        return []
    if response.status_code != 200 or not response.content:
        return []
    out: List[Path] = []
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            best: Dict[str, Tuple[int, int, str]] = {}
            for info in bundle.infolist():
                name = info.filename
                suffix = Path(name).suffix.lower()
                if suffix not in IMG_EXT:
                    continue
                stem = Path(name).stem
                score = (-int(info.file_size), FMT_RANK[suffix])
                current = best.get(stem)
                if current is None or score < current[:2]:
                    best[stem] = (score[0], score[1], name)
            for stale in have:
                stale.unlink()
            for name in sorted(entry[2] for entry in best.values()):
                target = dest / Path(name).name
                with bundle.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                out.append(target)
    except zipfile.BadZipFile:
        return []
    marker.write_text(f"{SELECTION_RULE}\n")
    time.sleep(sleep)
    return sorted(out)


def _pixel_size(path: Path) -> Optional[List[int]]:
    """The image's width and height, or None when it will not open.

    Recorded per figure because resolution is part of the answer: a detector
    that finds nothing in a 200 px preview of a figure has not been asked the
    question this harness is asking.

    :param path: the image file.
    :returns: ``[width, height]``, or None.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            return [int(handle.width), int(handle.height)]
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    """The sha256 of a file, for the deduplication the decision asks for.

    :param path: the file to hash.
    :returns: the lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_sample(args: argparse.Namespace) -> None:
    """Choose the papers and write ``papers.json``.

    THE EXCLUSION IS RECORDED IN FULL, not as a count. Refusing the training
    articles is what makes a number measured on the current detector mean
    anything, so it is the one input a reader most needs to reproduce; a count
    beside a path to a file that can be rewritten upstream is not a record of
    it. Every excluded id is written out, with the sha256 of each file they
    were read from.

    :param args: parsed command line.
    """
    exclude = set()
    sources = []
    for path in args.exclude_pmcids or []:
        text = Path(path).read_text(errors="ignore")
        found = PMCID_RE.findall(text)
        exclude.update(found)
        sources.append({"path": str(path),
                        "sha256": sha256_file(Path(path)),
                        "pmcids_found": len(set(found))})
    licences = tuple(x.strip().lower() for x in args.licences.split(","))
    papers = search_papers(args.query, args.papers, licences, exclude,
                           args.seed, args.pool)
    out = {
        "query": args.query,
        "licences": list(licences),
        "seed": args.seed,
        "pool": args.pool,
        "excluded_pmcids": len(exclude),
        "excluded_pmcid_sources": sources,
        "excluded_pmcid_list": sorted(exclude),
        "papers": papers,
    }
    target = Path(args.out) / "papers.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(f"sampled {len(papers)} papers, {len(exclude)} PMC ids excluded "
          f"-> {target}")


def stage_figures(args: argparse.Namespace) -> None:
    """Download every figure of every sampled paper and write ``figures.json``.

    :param args: parsed command line.
    """
    root = Path(args.out)
    papers = json.loads((root / "papers.json").read_text())["papers"]
    records: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    for paper in papers:
        pmcid = paper["pmcid"]
        captions = figure_captions(pmcid)
        images = fetch_figure_images(pmcid, root / "figures" / pmcid)
        kept = 0
        for image in images:
            stem = image.stem
            meta = captions.get(stem)
            if meta is None and args.main_figures_only:
                continue
            if args.max_figures and kept >= args.max_figures:
                break
            digest = sha256_file(image)
            size = _pixel_size(image)
            duplicate_of = seen_hashes.get(digest)
            if duplicate_of is None:
                seen_hashes[digest] = f"{pmcid}/{image.name}"
            records.append({
                "key": f"{pmcid}__{image.name}",
                "pmcid": pmcid,
                "doi": paper.get("doi"),
                "licence": paper.get("licence"),
                "year": paper.get("year"),
                "journal": paper.get("journal"),
                "path": str(image.relative_to(root)),
                "fig_label": (meta or {}).get("label", ""),
                "caption": (meta or {}).get("caption", "")[:1200],
                "in_full_text_xml": meta is not None,
                "pixels": size,
                "sha256": digest,
                "duplicate_of": duplicate_of,
            })
            kept += 1
        print(f"{pmcid}: {kept} figures")
    target = root / "figures.json"
    target.write_text(json.dumps({"figures": records}, indent=2))
    print(f"{len(records)} figures from {len(papers)} papers -> {target}")


def _detector_paths(keys: Sequence[str], dest: Path) -> Dict[str, Dict[str, str]]:
    """Ask the model zoo for each detector and install it locally.

    The zoo is asked by KEY rather than by filename, which is what 424 requires
    of the module itself: a retrain lands here without a code change, and the
    entry that comes back records which weights produced a result.

    TWO NAMES ARE RECORDED AND THEY DIFFER. ``name`` is the zoo's registered
    filename -- ``yolo_welldetect_v3.pt`` -- which is how the zoo row, the
    ledger and every other record of these weights refer to them.
    ``installed_as`` is the local filename, which ``install`` versions to avoid
    clobbering an existing download, so the same checkpoint lands as
    ``yolo_welldetect_v9.pt`` in one run and ``v11`` in the next. Writing only
    the installed name left a result file whose model names contradicted the
    ledger and each other, checkable only by a reader who knew to ignore them;
    the sha256 is the identity that ties either name back to the zoo row.

    :param keys: model zoo keys, or paths to checkpoints.
    :param dest: directory to install into.
    :returns: ``{key: {"path": ..., "name": ..., "installed_as": ...,
        "sha256": ..., "kind": ...}}``.
    :raises SystemExit: when a key is not in the zoo, listing what is.
    """
    from spacr import model_zoo

    dest.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, str]] = {}
    for key in keys:
        try:
            entry = model_zoo.resolve(key)
        except Exception as exc:
            detectors = [e.key for e in model_zoo.catalogue()
                         if e.kind == "detector"]
            raise SystemExit(f"{key}: {exc}\ndetectors in the zoo: "
                             f"{', '.join(detectors)}")
        registered = entry.name
        if entry.source != "local":
            entry = model_zoo.install(entry, dest)
        out[key] = {"path": entry.path, "name": registered,
                    "installed_as": entry.name,
                    "sha256": entry.sha256, "kind": entry.kind,
                    "verified": bool(entry.verified)}
        print(f"{key}: {registered} (installed as {entry.name}) "
              f"{entry.sha256[:12]}… verified={bool(entry.verified)}")
    return out


def _load_image(path: Path) -> Any:
    """One figure as an RGB array the detector can read.

    :param path: the image file.
    :returns: an ``H x W x 3`` uint8 array, or None when the file will not
        decode -- a figure that Pillow refuses is recorded, not guessed at.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"))
    except Exception:
        return None


def _short_tag(model: str) -> str:
    """A two-character label for a model key, for drawing on a box.

    ``toxoplasma_well_detector_v1`` and ``..._v2`` share their first eleven
    characters, so the tag is taken from the END of the key, which is where a
    version sits.

    :param model: the model zoo key.
    :returns: an uppercase tag of one to three characters.
    """
    tail = model.rstrip("/").split("_")[-1]
    if re.fullmatch(r"v\d+", tail, flags=re.I):
        return tail.upper()
    return (tail[:3] or model[:3]).upper()


def _draw_overlay(image_path: Path, boxes_by_model: Dict[str, List[Dict]],
                  target: Path, max_side: int, font_path: Optional[Path]) -> None:
    """Write one figure with a model's boxes numbered on it.

    The numbers are what the hand-written label file refers to, so they are
    drawn large enough to read on a downscaled copy and placed inside the box
    rather than above it, where two stacked panels would overlap.

    ONE MODEL PER OVERLAY, which is not a style choice. Two models drawn on
    one copy hide each other: where both fire on the same panel the second
    drawn covers the first, and a box a person cannot see is a box they cannot
    judge. The caller writes one image per model into its own folder.

    :param image_path: the figure.
    :param boxes_by_model: ``{model key: [box dicts]}``, each box carrying
        ``x0``, ``y0``, ``x1`` and ``y1``.
    :param target: where to write the overlay.
    :param max_side: longest side of the written overlay in pixels.
    :param font_path: a TrueType font, or None for Pillow's default.
    """
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    scale = 1.0
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize((max(1, int(image.width * scale)),
                              max(1, int(image.height * scale))),
                             Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    width = max(2, int(min(image.size) * 0.004))
    for index, (model, boxes) in enumerate(sorted(boxes_by_model.items())):
        colour = BOX_COLOURS[index % len(BOX_COLOURS)]
        for number, box in enumerate(boxes, start=1):
            rect = [box["x0"] * scale, box["y0"] * scale,
                    box["x1"] * scale, box["y1"] * scale]
            draw.rectangle(rect, outline=colour, width=width)
            tag = f"{_short_tag(model)}-{number}"
            size = int(max(9, min(22, (rect[3] - rect[1]) * 0.4)))
            try:
                font = (ImageFont.truetype(str(font_path), size) if font_path
                        else ImageFont.load_default(size=size))
            except Exception:
                font = ImageFont.load_default()
            tag_w = size * (len(tag) * 0.62) + 4
            tag_h = size * 1.25
            inside = (rect[2] - rect[0]) > tag_w * 1.6
            top = rect[1] + width if inside else max(0.0, rect[1] - tag_h)
            left = rect[0] + (width if inside else 0)
            draw.rectangle([left, top, left + tag_w, top + tag_h], fill=colour)
            draw.text((left + 2, top + size * 0.1), tag,
                      fill=(255, 255, 255), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def _keep_ultralytics_out_of_the_user_config(args: argparse.Namespace,
                                             root: Path) -> Path:
    """Point ``YOLO_CONFIG_DIR`` somewhere this run owns.

    Ultralytics writes ``settings.json`` on import, under ``YOLO_CONFIG_DIR``
    when that is set and under the user's own ``~/.config/Ultralytics``
    otherwise. On 2026-09-20 the plain import printed "Ultralytics settings
    reset to default values" over a file that had been there since July, and
    there was no backup. The loss is small and completely avoidable, and the
    only moment at which it is avoidable is before the first import.

    This must therefore run before anything pulls ``ultralytics`` in, which
    means before ``spacr.plaque`` loads a detector.

    :param args: parsed command line; ``--yolo-config-dir`` overrides the
        default and is the way to ask for the real one back.
    :param root: the run's output directory, whose ``yolo-config`` is the
        default.
    :returns: the directory ultralytics was pointed at.
    """
    chosen = Path(args.yolo_config_dir) if args.yolo_config_dir else (
        root / "yolo-config")
    chosen.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(chosen)
    print(f"ultralytics settings -> {chosen}")
    return chosen


def stage_detect(args: argparse.Namespace) -> None:
    """Run every detector over every figure and write the overlays.

    ``spacr.plaque.detect_wells`` is the path the future module would take, so
    it is the path measured -- including its ordering and its ``Well``
    geometry. Two deliberate departures, both recorded on every box rather
    than applied silently:

    * ``min_axis_ratio`` is 0 here, so nothing is dropped before a person sees
      it, and each box carries its axis ratio and whether the shipped default
      of 0.7 would have kept it;
    * the module's loader is memoised for this process. ``_load_detector``
      builds a YOLO object per call, which over a few hundred figures is
      minutes of loading a 5 MB checkpoint again. The predict path is
      untouched.

    :param args: parsed command line.
    """
    import inspect

    root = Path(args.out)
    _keep_ultralytics_out_of_the_user_config(args, root)

    import spacr
    import spacr.plaque as plaque

    figures = json.loads((root / "figures.json").read_text())["figures"]
    models = _detector_paths([m.strip() for m in args.models.split(",") if
                              m.strip()], root / "models")
    plaque._load_detector = functools.lru_cache(maxsize=4)(
        plaque._load_detector)
    shipped_ratio = float(inspect.signature(plaque.detect_wells)
                          .parameters["min_axis_ratio"].default)
    font = Path(args.font) if args.font else None
    records: List[Dict[str, Any]] = []
    started = time.time()
    for position, figure in enumerate(figures, start=1):
        if figure.get("duplicate_of"):
            continue
        path = root / figure["path"]
        image = _load_image(path)
        if image is None:
            records.append({"key": figure["key"], "unreadable": True})
            continue
        height, width = image.shape[:2]
        boxes_by_model: Dict[str, List[Dict]] = {}
        for key, info in models.items():
            wells = plaque.detect_wells(
                image, info["path"], confidence=args.conf,
                imgsz=args.imgsz, min_axis_ratio=0.0)
            boxes = []
            for well in wells:
                box = well.as_dict()
                box["confidence"] = round(float(box["confidence"]), 4)
                box["diameter_px"] = round(float(box["diameter_px"]), 1)
                box["axis_ratio"] = round(float(box["axis_ratio"]), 3)
                box["kept_by_shipped_filter"] = (
                    well.axis_ratio >= shipped_ratio)
                boxes.append(box)
            boxes_by_model[key] = boxes
        records.append({
            "key": figure["key"], "pmcid": figure["pmcid"],
            "doi": figure.get("doi"), "fig_label": figure.get("fig_label"),
            "width": int(width), "height": int(height),
            "boxes": boxes_by_model,
        })
        for key, boxes in boxes_by_model.items():
            if boxes or args.overlay_all:
                _draw_overlay(path, {key: boxes},
                              root / "overlays" / _short_tag(key).lower()
                              / f"{figure['key']}.png",
                              args.overlay_max_side, font)
        if position % 10 == 0:
            print(f"{position}/{len(figures)} figures, "
                  f"{time.time() - started:.0f}s")
    target = root / "detections.json"
    target.write_text(json.dumps({
        "models": models, "conf": args.conf, "imgsz": args.imgsz,
        "shipped_min_axis_ratio": shipped_ratio,
        "channel_order": "bgr",
        "spacr_from": getattr(spacr, "__file__", None),
        "yolo_config_dir": os.environ.get("YOLO_CONFIG_DIR"),
        "device": "cuda" if args.gpu else "cpu",
        "seconds": round(time.time() - started, 1),
        "figures": records}, indent=2))
    counts = {key: sum(len(r.get("boxes", {}).get(key, [])) for r in records)
              for key in models}
    print(f"{len(records)} figures, boxes per model {counts} -> {target}")


def _blank_labels(root: Path) -> Dict[str, Any]:
    """The label file a person fills in, pre-filled with every box to judge.

    :param root: the output directory holding ``detections.json``.
    :returns: the label skeleton.
    """
    detections = json.loads((root / "detections.json").read_text())
    figures = []
    for record in detections["figures"]:
        if record.get("unreadable"):
            continue
        models = {}
        for key, boxes in record["boxes"].items():
            models[key] = {"box_verdicts": ["?"] * len(boxes),
                           "found": {"well": None, "other": None}}
        figures.append({"key": record["key"], "pmcid": record["pmcid"],
                        "regions": {"well": None, "other": None},
                        "models": models})
    return {"protocol": PROTOCOL, "figures": figures}


def _rates(tp: int, dup: int, fp: int, found: int,
           truth: int) -> Tuple[float, float]:
    """Precision and recall from one bucket of counts.

    :param tp: boxes that claimed a region first.
    :param dup: boxes on a region already claimed.
    :param fp: boxes on no region at all.
    :param found: regions with at least one box on them.
    :param truth: regions present.
    :returns: ``(precision, recall)``, each NaN when its denominator is 0.
    """
    boxes = tp + dup + fp
    precision = tp / boxes if boxes else float("nan")
    recall = found / truth if truth else float("nan")
    return precision, recall


def stage_score(args: argparse.Namespace) -> None:
    """Turn the hand-written labels into precision, recall and a verdict table.

    Two recalls are reported and neither is the "real" one. MICRO pools every
    region in the sample, so one 96-well plate photograph carries as much
    weight as ninety-six single-well panels -- which is correct if the goal is
    counting plaques and wrong if the goal is finding figures. MACRO averages
    the per-figure recall over figures that hold at least one region, so every
    figure counts once. They answer different questions and they disagree
    here; printing one alone would be choosing the flattering number.

    :param args: parsed command line.
    :raises SystemExit: when a label file is incomplete, naming what is
        missing. A partly judged sample scored as if it were whole is the one
        failure mode of a hand-labelled measurement.
    """
    root = Path(args.out)
    labels_path = Path(args.labels) if args.labels else root / "labels.json"
    if not labels_path.exists():
        skeleton = _blank_labels(root)
        labels_path.write_text(json.dumps(skeleton, indent=2))
        raise SystemExit(f"wrote an empty label file to {labels_path}; "
                         f"fill it in from the overlays and run score again")
    labels = json.loads(labels_path.read_text())
    detections = json.loads((root / "detections.json").read_text())
    by_key = {r["key"]: r for r in detections["figures"]}
    totals: Dict[str, Dict[str, Any]] = {}
    unjudged: List[str] = []
    figure_rows: List[Dict[str, Any]] = []
    for figure in labels["figures"]:
        key = figure["key"]
        regions = figure.get("regions") or {}
        if regions.get("well") is None or regions.get("other") is None:
            unjudged.append(f"{key}: regions")
            continue
        truth_well = int(regions["well"])
        truth_other = int(regions["other"])
        truth = truth_well + truth_other
        row: Dict[str, Any] = {"key": key, "regions": dict(regions)}
        for model, judged in figure["models"].items():
            verdicts = judged.get("box_verdicts") or []
            boxes = by_key.get(key, {}).get("boxes", {}).get(model, [])
            if len(verdicts) != len(boxes) or any(v == "?" for v in verdicts):
                unjudged.append(f"{key}: {model} box_verdicts")
                continue
            found = judged.get("found") or {}
            found_well = found.get("well")
            found_other = found.get("other")
            if found_well is None or found_other is None:
                unjudged.append(f"{key}: {model} found")
                continue
            found_well = int(found_well)
            found_other = int(found_other)
            if found_well > truth_well or found_other > truth_other:
                raise SystemExit(
                    f"{key}: {model} found more regions than the figure has "
                    f"({found_well}/{truth_well} well, "
                    f"{found_other}/{truth_other} other)")
            tp_boxes = sum(1 for v in verdicts if v == "tp")
            if found_well + found_other != tp_boxes:
                raise SystemExit(
                    f"{key}: {model} counts {found_well + found_other} regions "
                    f"found but marks {tp_boxes} boxes tp; the protocol makes "
                    f"those the same quantity -- a region is found exactly "
                    f"when a first box lands on it, and every later box is dup")
            bucket = totals.setdefault(model, {
                "tp": 0, "fp": 0, "dup": 0,
                "truth_well": 0, "truth_other": 0,
                "found_well": 0, "found_other": 0,
                "figures": 0, "region_figures": 0, "empty_figures": 0,
                "boxes_on_empty": 0, "per_figure_recall": []})
            bucket["tp"] += tp_boxes
            bucket["dup"] += sum(1 for v in verdicts if v == "dup")
            bucket["fp"] += sum(1 for v in verdicts if v == "fp")
            bucket["truth_well"] += truth_well
            bucket["truth_other"] += truth_other
            bucket["found_well"] += found_well
            bucket["found_other"] += found_other
            bucket["figures"] += 1
            if truth:
                bucket["region_figures"] += 1
                bucket["per_figure_recall"].append(
                    (found_well + found_other) / truth)
            else:
                bucket["empty_figures"] += 1
                bucket["boxes_on_empty"] += len(verdicts)
            row[model] = {"boxes": len(verdicts),
                          "tp": sum(1 for v in verdicts if v == "tp"),
                          "found_well": found_well,
                          "found_other": found_other}
        figure_rows.append(row)
    if unjudged and not args.allow_partial:
        raise SystemExit("unjudged, so nothing is scored:\n  " +
                         "\n  ".join(unjudged[:40]))
    lines = ["| model | figures | with a region | regions (well/other) | "
             "boxes | TP | dup | FP | precision | recall micro | recall macro "
             "| recall on wells | boxes on figures with no region |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary: Dict[str, Any] = {}
    for model, bucket in sorted(totals.items()):
        truth = bucket["truth_well"] + bucket["truth_other"]
        found = bucket["found_well"] + bucket["found_other"]
        precision, recall = _rates(bucket["tp"], bucket["dup"], bucket["fp"],
                                   found, truth)
        macro = (sum(bucket["per_figure_recall"])
                 / len(bucket["per_figure_recall"])
                 if bucket["per_figure_recall"] else float("nan"))
        well_recall = (bucket["found_well"] / bucket["truth_well"]
                       if bucket["truth_well"] else float("nan"))
        boxes = bucket["tp"] + bucket["dup"] + bucket["fp"]
        summary[model] = {
            "figures": bucket["figures"],
            "figures_with_a_region": bucket["region_figures"],
            "regions_well": bucket["truth_well"],
            "regions_other": bucket["truth_other"],
            "found_well": bucket["found_well"],
            "found_other": bucket["found_other"],
            "boxes": boxes, "tp": bucket["tp"], "dup": bucket["dup"],
            "fp": bucket["fp"], "precision": round(precision, 4),
            "recall_micro": round(recall, 4),
            "recall_macro": round(macro, 4),
            "recall_wells": round(well_recall, 4),
            "boxes_on_empty_figures": bucket["boxes_on_empty"],
            "empty_figures": bucket["empty_figures"],
        }
        lines.append(
            f"| {model} | {bucket['figures']} | {bucket['region_figures']} | "
            f"{bucket['truth_well']}/{bucket['truth_other']} | {boxes} | "
            f"{bucket['tp']} | {bucket['dup']} | {bucket['fp']} | "
            f"{precision:.3f} | {recall:.3f} | {macro:.3f} | "
            f"{well_recall:.3f} | {bucket['boxes_on_empty']} on "
            f"{bucket['empty_figures']} |")
    table = "\n".join(lines)
    print(table)
    if unjudged:
        print(f"\nPARTIAL: {len(unjudged)} unjudged entries were skipped")
    (root / "result.json").write_text(json.dumps({
        "summary": summary, "figures": figure_rows,
        "unjudged": unjudged, "models": detections.get("models"),
        "conf": detections.get("conf")}, indent=2))
    (root / "result.md").write_text(table + "\n")
    print(f"\n-> {root / 'result.json'} and {root / 'result.md'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Read the command line.

    :param argv: arguments, defaulting to ``sys.argv[1:]``.
    :returns: the parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True,
                        help="output directory; every stage reads and writes here")
    parser.add_argument("--stages", default="sample,figures,detect",
                        help="comma-separated: sample,figures,detect,score")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--papers", type=int, default=20)
    parser.add_argument("--pool", type=int, default=400,
                        help="hits read before the seeded shuffle")
    parser.add_argument("--seed", type=int, default=424)
    parser.add_argument("--licences", default=",".join(DEFAULT_LICENCES),
                        help="licences that may be sampled, comma-separated")
    parser.add_argument("--exclude-pmcids", action="append",
                        help="file to take PMC ids out of; repeatable. Pass "
                             "the detector's split.csv or no number here "
                             "means anything")
    parser.add_argument("--max-figures", type=int, default=0,
                        help="cap per paper; 0 keeps them all")
    parser.add_argument("--main-figures-only", action="store_true",
                        help="drop images the full-text XML does not name as "
                             "a figure graphic")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--gpu", action="store_true",
                        help="let ultralytics see the GPU. Off by default: "
                             "two yolo-n checkpoints over a few hundred "
                             "figures is about a minute of CPU, and a "
                             "measurement that small has no business "
                             "queueing behind a training run")
    parser.add_argument("--yolo-config-dir", default=None,
                        help="where ultralytics may write its settings.json. "
                             "Default <out>/yolo-config, so that importing it "
                             "cannot reset the user's own")
    parser.add_argument("--overlay-max-side", type=int, default=1400)
    parser.add_argument("--overlay-all", action="store_true",
                        help="also write overlays for figures with no box")
    parser.add_argument("--font", default=None, help="TrueType font for the "
                                                     "box numbers")
    parser.add_argument("--labels", default=None,
                        help="the hand-written label file (default "
                             "<out>/labels.json)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="score what is judged and say how much was not")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested stages in order.

    :param argv: arguments, defaulting to ``sys.argv[1:]``.
    :returns: the process exit code.
    """
    args = parse_args(argv)
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    known = {"sample": stage_sample, "figures": stage_figures,
             "detect": stage_detect, "score": stage_score}
    for stage in stages:
        if stage not in known:
            raise SystemExit(f"unknown stage {stage!r}; "
                             f"known: {', '.join(known)}")
    for stage in stages:
        print(f"== {stage}")
        known[stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
