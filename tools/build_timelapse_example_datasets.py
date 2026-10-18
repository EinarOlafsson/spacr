"""Stage Make Masks' timelapse test data: yeast, bacteria and mammalian movies.

Three public time-lapse movies with complete ground-truth instance masks in
every frame, persistent track IDs and published lineage, for the Time mode
and lineage tools of Make Masks (item 708). Each is packaged the way Make
Masks opens a timelapse: one image TIFF per frame at the top of the folder and
one uint32 mask TIFF per frame under ``ground_truth_masks/`` with the
``spacr.lineage/v1`` graph embedded in the TIFF description, so every division
of the source lineage is on the Time/lineage menu as soon as the sequence
opens.

Sources (all CC BY 4.0):

``yeast``
    YeaZ LYN-Track-and-Trace data (Zelic, Gligorovski and Rahi), Zenodo
    10.5281/zenodo.18670420: *Saccharomyces cerevisiae* colony SJR1,
    brightfield, YeaZ masks with persistent IDs and mother-bud lineage from a
    bud-neck marker.
``bacteria``
    "Tracking one-in-a-million" (Seiffarth et al.), Zenodo
    10.5281/zenodo.7260137: *Corynebacterium glutamicum* microcolony,
    sequence 00, phase contrast, manually corrected masks and CTC lineage.
``mammalian``
    Tui example data (Chang and Tsai), Zenodo 10.5281/zenodo.19026908:
    T98G glioblastoma cells migrating in an electric field, phase contrast,
    human-curated CTC masks and lineage.

The input is the CTC-like conversion of each source: ``<seq>/t###.tif``
images, ``man_track###.tif`` uint16 labels whose value is the track ID, and a
``lineage.csv`` with ``label,t_start,t_end,parent`` (0-based frames). Frames
outside the window are dropped and frames renumbered from 0; track IDs are
kept as the uint32 pixel labels.

LINEAGE CONVERSION. A CTC division ends the mother's track and starts two
daughter tracks; it becomes a 708 division at the first daughter frame, the
daughters named after the mother with ``a`` and ``b``. A budding yeast mother
keeps her track ID while the bud gets its own; 708 records that as a division
into two segments, the mother's continuation (same name, same pixel label)
and the bud (the mother's name plus a letter). A track whose parent is
outside the window, a lone daughter and any daughter beyond the second at
one frame is a founder. Every count is in ``manifest.csv``.

Output (``dest``): ``<folder>/`` per set, its ``README.md`` card and its
``.tar`` (manifest last, so a transfer that died part-way reads as absent).

Usage: python tools/build_timelapse_example_datasets.py DEST
           --source KEY=DIR [--source KEY=DIR ...] [--only KEY ...] [--upload]

``--source`` names the folder holding the converted sequence of one set.
``--upload`` publishes each archive and card to its Hugging Face dataset
repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMESPACE = uuid.UUID("5c0a7f0e-7120-4d1b-9a3e-712712712712")
REVISION = 1


@dataclass(frozen=True)
class Movie:
    """One source movie and how it is cut."""

    key: str
    repo: str
    archive: str
    folder: str
    prefix: str
    organism: str
    modality: str
    window: Tuple[int, int]
    budding: bool
    crop: Optional[Tuple[int, int, int, int]] = None
    image_glob: str = "t*.tif"
    mask_glob: str = "man_track*.tif"
    lineage: str = "lineage.csv"
    card: Dict[str, str] = field(default_factory=dict)


MOVIES: Tuple[Movie, ...] = (
    Movie(
        key="timelapse_yeast",
        repo="einarolafsson/spacr-example-timelapse-yeast",
        archive="spacr-example-timelapse-yeast.tar",
        folder="make_masks_timelapse_yeast",
        prefix="yeast_sjr1",
        organism="Saccharomyces cerevisiae",
        modality="brightfield",
        window=(100, 181),
        budding=True,
        card=dict(
            title="budding yeast colony (brightfield)",
            source="YeaZ LYN-Track-and-Trace data, colony SJR1 (budding/SJR1)",
            url="https://zenodo.org/records/18670420",
            doi="10.5281/zenodo.18670420",
            citation=(
                "Zelic M, Gligorovski V, Rahi SJ. Training and test sets of "
                "microscopy images for Zelic et al. Bioinformatics Advances "
                "2026. Zenodo (2026). doi:10.5281/zenodo.18670420"),
            shows=(
                "A growing S. cerevisiae colony imaged in brightfield. Mothers "
                "bud repeatedly; each bud is a division in the lineage menu, "
                "the mother's continuation keeping her name and label and the "
                "bud named after her with a letter. Mother-bud pairs come "
                "from a bud-neck marker, not from the masks."),
        ),
    ),
    Movie(
        key="timelapse_bacteria",
        repo="einarolafsson/spacr-example-timelapse-bacteria",
        archive="spacr-example-timelapse-bacteria.tar",
        folder="make_masks_timelapse_bacteria",
        prefix="cglutamicum_00",
        organism="Corynebacterium glutamicum ATCC 13032",
        modality="phase contrast",
        window=(440, 500),
        budding=False,
        image_glob="t*.tif",
        card=dict(
            title="Corynebacterium glutamicum microcolony (phase contrast)",
            source="Tracking one-in-a-million, ctc_format sequence 00",
            url="https://zenodo.org/records/7260137",
            doi="10.5281/zenodo.7260137",
            citation=(
                "Seiffarth J, Bloebaum L, Loeffler K, Scherr T, Gruenberger A, "
                "Scharr H, Mikut R, Noeh K. Data for - Tracking "
                "one-in-a-million: large-scale benchmark for microbial "
                "single-cell tracking with experiment-aware robustness "
                "metrics. Zenodo (2022). doi:10.5281/zenodo.7260137"),
            shows=(
                "A C. glutamicum monolayer microcolony growing in a "
                "microfluidic chamber, one frame per minute at 0.072 um per "
                "pixel. Rods divide into two daughters throughout the movie "
                "and are pushed along as the colony expands."),
        ),
    ),
    Movie(
        key="timelapse_mammalian",
        repo="einarolafsson/spacr-example-timelapse-mammalian",
        archive="spacr-example-timelapse-mammalian.tar",
        folder="make_masks_timelapse_mammalian",
        prefix="t98g",
        organism="Homo sapiens, T98G glioblastoma cells",
        modality="phase contrast",
        window=(0, 37),
        budding=False,
        card=dict(
            title="T98G glioblastoma cells in an electric field (phase contrast)",
            source="Tui example datasets, T98G_electrotaxis.zip",
            url="https://zenodo.org/records/19026908",
            doi="10.5281/zenodo.19026908",
            citation=(
                "Chang I-M, Tsai H-F. Example dataset and results for Tui, a "
                "multi-generational and expert-correctable tracker for "
                "cellular dynamics. Zenodo (2026). "
                "doi:10.5281/zenodo.19026908. Images from the Usiigaci "
                "electrotaxis movie: Tsai H-F, Gajda J, Sloan TFW, Rares A, "
                "Shen AQ. Usiigaci: instance-aware cell tracking in "
                "stain-free phase contrast microscopy enabled by machine "
                "learning. SoftwareX 9:230-237 (2019)."),
            shows=(
                "About 70 T98G cells migrating under a direct-current electric "
                "field, with mitoses whose two daughters are recorded in the "
                "lineage. Some cells move several times faster than the "
                "median cell between frames."),
        ),
    ),
)


def movie(key: str) -> Movie:
    """The :data:`MOVIES` entry called ``key``."""
    for candidate in MOVIES:
        if candidate.key == key:
            return candidate
    raise KeyError(key)


def read_lineage(path) -> Dict[int, Tuple[int, int, int]]:
    """``label -> (t_start, t_end, parent)`` from a CTC-like lineage.csv."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(row["label"]): (int(row["t_start"]), int(row["t_end"]),
                                    int(row["parent"]))
                for row in csv.DictReader(handle)}


def _suffix(index: int) -> str:
    """``a``, ``b``, ... ``z``, ``aa``: the letter of the ``index``-th bud."""
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(97 + rest) + letters
    return letters


def _uid(key: str, *parts) -> str:
    """A reproducible segment UUID, so a rebuild writes identical bytes."""
    return str(uuid.uuid5(NAMESPACE, ":".join([key, *map(str, parts)])))


def convert_lineage(key: str, present: Dict[int, List[int]],
                    lineage: Dict[int, Tuple[int, int, int]],
                    frame_paths: List[str], budding: bool):
    """Turn per-frame track presence plus CTC lineage into a 708 graph.

    :param key: the set key, seeding the reproducible UUIDs.
    :param present: track ID -> sorted window frames where it has pixels.
    :param lineage: the source ``label -> (t_start, t_end, parent)``.
    :param frame_paths: the absolute image path of each window frame.
    :param budding: mother tracks continue through their buds.
    :returns: ``(state, counts)``: the lineage dict and conversion counts.
    """
    first = {track: frames[0] for track, frames in present.items()}
    last = {track: frames[-1] for track, frames in present.items()}
    children: Dict[int, List[int]] = {}
    for track in present:
        parent = lineage.get(track, (0, 0, 0))[2]
        if parent and parent in present and first[track] > first[parent]:
            children.setdefault(parent, []).append(track)
    counts = dict(divisions=0, lone_daughters=0, extra_daughters=0,
                  late_buds=0)
    cells: Dict[str, dict] = {}
    segment_of: Dict[Tuple[int, int], str] = {}
    names: Dict[int, str] = {}
    bud_frames: Dict[int, List[Tuple[int, int]]] = {}
    accepted_parent: Dict[int, int] = {}

    for parent in sorted(children):
        kids = sorted(children[parent], key=lambda t: (first[t], t))
        if budding:
            used = set()
            for kid in kids:
                frame = first[kid]
                if frame in used or frame > last[parent]:
                    counts["late_buds" if frame > last[parent]
                           else "extra_daughters"] += 1
                    continue
                used.add(frame)
                bud_frames.setdefault(parent, []).append((frame, kid))
                accepted_parent[kid] = parent
        else:
            ending = [kid for kid in kids if first[kid] > last[parent]]
            if len(ending) < 2:
                counts["lone_daughters"] += len(ending)
                continue
            counts["extra_daughters"] += len(ending) - 2
            for kid in ending[:2]:
                accepted_parent[kid] = parent

    def segments_of(track: int):
        """``(start_frame, segment_uuid)`` per segment of ``track``."""
        starts = [first[track]] + [frame for frame, _ in bud_frames.get(track, [])]
        return [(start, _uid(key, track, start)) for start in starts]

    order = sorted(present, key=lambda t: (first[t], t))
    for track in order:
        parent = accepted_parent.get(track)
        if parent is None:
            names[track] = str(track)
    pending = list(order)
    while pending:
        progressed = []
        for track in pending:
            parent = accepted_parent.get(track)
            if track in names:
                progressed.append(track)
                continue
            if parent not in names:
                continue
            if budding:
                index = [kid for _, kid in bud_frames[parent]].index(track)
                names[track] = names[parent] + _suffix(index)
            else:
                pair = sorted((kid for kid, mother in accepted_parent.items()
                               if mother == parent), key=lambda t: (first[t], t))
                names[track] = names[parent] + _suffix(pair.index(track))
            progressed.append(track)
        if not progressed:
            raise ValueError("lineage has a cycle or a missing parent")
        pending = [track for track in pending if track not in progressed]

    for track in order:
        segs = segments_of(track)
        parent = accepted_parent.get(track)
        for index, (start, identity) in enumerate(segs):
            cell = {"name": names[track], "parent_uuid": None,
                    "born_frame": frame_paths[start], "division_frame": None,
                    "children": [], "label": int(track), "revision": REVISION}
            if index > 0:
                cell["parent_uuid"] = segs[index - 1][1]
            elif parent is not None:
                if budding:
                    mother = segments_of(parent)
                    position = [s for s, _ in mother].index(start)
                    cell["parent_uuid"] = mother[position - 1][1]
                else:
                    cell["parent_uuid"] = segments_of(parent)[-1][1]
            cells[identity] = cell
        for frame in present[track]:
            identity = [uid for start, uid in segs if start <= frame][-1]
            segment_of[(frame, track)] = identity

    for identity, cell in cells.items():
        mother = cell["parent_uuid"]
        if mother is not None:
            cells[mother]["children"].append(identity)
            cells[mother]["division_frame"] = cell["born_frame"]
    for cell in cells.values():
        if cell["children"]:
            if len(cell["children"]) != 2:
                raise ValueError("a division needs exactly two daughters")
            cell["children"].sort(key=lambda c: (cells[c]["name"], c))
    counts["divisions"] = sum(1 for cell in cells.values() if cell["children"])
    frames = {path: {} for path in frame_paths}
    for (frame, track), identity in sorted(segment_of.items()):
        frames[frame_paths[frame]][str(int(track))] = identity
    state = {"schema": "spacr.lineage/v1", "cells": cells, "frames": frames,
             "frame_revisions": {path: REVISION for path in frame_paths},
             "deleted_cells": {},
             "frame_ids": {path: str(uuid.uuid5(uuid.NAMESPACE_URL,
                                                "spacr-frame:" + path))
                           for path in frame_paths}}
    counts.update(cells=len(cells), tracks=len(present))
    return state, counts


def _write_image(path, image) -> None:
    """One frame image, zlib-compressed, grey."""
    from spacr.tiff_io import write_tiff

    write_tiff(path, image, compression="zlib")
    os.chmod(path, 0o644)


def _write_mask(folder: Path, name: str, labels, state) -> Path:
    """Write a mask through Make Masks' own saver, then zlib-compress it.

    :func:`spacr.qt.mask_engine.save_mask` decides the description (the
    ``spacr.mask/v1`` metadata with the lineage, the frame and its identity).
    The pixels are rewritten compressed with the same description, because
    uncompressed uint32 masks would make the download four times larger.
    """
    from spacr.qt import mask_engine as engine
    from spacr.tiff_io import write_tiff

    written = Path(engine.save_mask(str(folder), name, labels.astype(np.uint32),
                                    masks_dir=str(folder / "ground_truth_masks"),
                                    preserve_ids=True, lineage=state))
    with tifffile.TiffFile(written) as handle:
        description = handle.pages[0].description
        pixels = handle.pages[0].asarray()
    write_tiff(written, pixels, metadata=None, description=description,
               compression="zlib")
    os.chmod(written, 0o644)
    return written


def build_movie(spec: Movie, source: Path, dest: Path) -> Dict[str, object]:
    """Stage one set under ``dest`` and return its manifest row."""
    from spacr.qt import mask_engine as engine

    images = sorted(Path(source).glob(spec.image_glob))
    masks = sorted(Path(source).glob(spec.mask_glob))
    images = [path for path in images if not path.name.startswith("man_")]
    if len(images) != len(masks):
        raise ValueError(f"{spec.key}: {len(images)} images, {len(masks)} masks")
    low, high = spec.window
    images, masks = images[low:high], masks[low:high]
    folder = Path(dest) / spec.folder
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "ground_truth_masks").mkdir(parents=True)
    names = [f"{spec.prefix}_t{t:03d}.tif" for t in range(len(images))]
    paths = [os.path.abspath(folder / name) for name in names]
    stack = []
    present: Dict[int, List[int]] = {}
    for t, (image_path, mask_path) in enumerate(zip(images, masks)):
        image = tifffile.imread(image_path)
        labels = tifffile.imread(mask_path)
        if spec.crop is not None:
            y0, y1, x0, x1 = spec.crop
            image, labels = image[y0:y1, x0:x1], labels[y0:y1, x0:x1]
        if image.shape != labels.shape:
            raise ValueError(f"{spec.key} frame {t}: image and mask differ")
        _write_image(folder / names[t], image)
        stack.append(labels)
        for track in np.unique(labels):
            if track:
                present.setdefault(int(track), []).append(t)
    state, counts = convert_lineage(spec.key, present,
                                    read_lineage(Path(source) / spec.lineage),
                                    paths, spec.budding)
    state = engine.validate_lineage(state)
    for t, labels in enumerate(stack):
        _write_mask(folder, names[t], labels, state)
    counts_per_frame = [int(np.count_nonzero(np.unique(labels))) for labels in stack]
    digest = hashlib.sha256()
    for labels in stack:
        digest.update(labels.astype(np.uint32).tobytes())
    rows = [{"frame": t, "image": names[t],
             "ground_truth_mask": f"ground_truth_masks/{Path(names[t]).stem}.tif",
             "cells": counts_per_frame[t]} for t in range(len(names))]
    summary = {
        "key": spec.key, "organism": spec.organism, "modality": spec.modality,
        "source": spec.card["source"], "doi": spec.card["doi"],
        "licence": "CC-BY-4.0", "source_frames": f"{low}-{low + len(names) - 1}",
        "frames": len(names), "shape": "x".join(map(str, stack[0].shape)),
        "cells_min": min(counts_per_frame), "cells_max": max(counts_per_frame),
        "cells_median": float(np.median(counts_per_frame)),
        "labels_sha256": digest.hexdigest(), **counts}
    with (folder / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                         encoding="utf-8")
    return summary


CARD = """---
license: cc-by-4.0
pretty_name: spaCR example data, Make Masks timelapse ({key})
task_categories:
- image-segmentation
tags:
- microscopy
- timelapse
- cell-tracking
- lineage
- spacr
size_categories:
- n<1K
---

# spaCR example data: {title}

A time-lapse movie with ground-truth instance masks in every frame and the
published lineage, for the Time mode and lineage tools of spaCR's Make Masks
("Load test data", "{caption}"). It opens with its masks loaded; the lineage
is embedded in the mask TIFFs, so the divisions are on the lineage menu as
soon as the sequence opens.

## What it shows

{shows}

| | |
|---|---|
| organism | {organism} |
| modality | {modality} |
| frames | {frames} (source frames {source_frames}) |
| image size | {shape} pixels |
| cells per frame | {cells_min} to {cells_max} (median {cells_median:g}) |
| tracked cells | {tracks} |
| divisions | {divisions} |

## Files

- `{prefix}_t000.tif` ...: one grey image per frame (zlib TIFF).
- `ground_truth_masks/{prefix}_t000.tif` ...: one uint32 label mask per frame.
  The pixel value is the source track ID, the same in every frame; the TIFF
  description holds spaCR's `spacr.lineage/v1` graph (cell names, persistent
  segment UUIDs and divisions).
- `manifest.csv`: every frame, its mask and its cell count.
- `summary.json`: the counts above and how the lineage was converted.

## Source

{source}. {url} (DOI {doi}).

Frames outside the window are dropped and the rest renumbered from 0; the
images and the masks are otherwise unchanged. {conversion}

Built by `tools/build_timelapse_example_datasets.py` in the spaCR repository.

## Licence

CC BY 4.0, as published by the authors. Cite the source below when you use it.

## Citation

{citation}
"""

CONVERSIONS = {
    True: ("Each bud recorded in the source lineage is a division of its "
           "mother: her continuation keeps her name and label, the bud is "
           "named after her with a letter."),
    False: ("Each source division with two daughters is a division here, the "
            "daughters named after the mother with a and b. A track whose "
            "mother is outside the window, or a lone daughter, starts its own "
            "lineage."),
}

CAPTIONS = {"timelapse_yeast": "Timelapse: yeast",
            "timelapse_bacteria": "Timelapse: bacteria",
            "timelapse_mammalian": "Timelapse: mammalian"}


def card(spec: Movie, summary: Dict[str, object]) -> str:
    """The dataset card of one set."""
    return CARD.format(key=spec.key, caption=CAPTIONS[spec.key],
                       prefix=spec.prefix,
                       conversion=CONVERSIONS[spec.budding],
                       **spec.card, **{k: v for k, v in summary.items()
                                       if k not in ("source", "doi", "key")})


def build(dest: Path, sources: Dict[str, Path], only=None) -> Dict[str, Path]:
    """Stage every requested set and its archive under ``dest``.

    :param dest: output folder.
    :param sources: set key -> folder of its converted sequence.
    :param only: set keys to build; all of ``sources`` when omitted.
    :returns: set key -> archive path.
    """
    dest = Path(dest)
    archives = {}
    for spec in MOVIES:
        if spec.key not in sources or (only and spec.key not in only):
            continue
        out = dest / spec.key
        out.mkdir(parents=True, exist_ok=True)
        summary = build_movie(spec, Path(sources[spec.key]), out)
        (out / "README.md").write_text(card(spec, summary), encoding="utf-8")
        folder = out / spec.folder
        archive = out / spec.archive
        with tarfile.open(archive, "w") as tar:
            for path in sorted(folder.rglob("*")):
                if path.name != "manifest.csv":
                    info = tar.gettarinfo(str(path), path.relative_to(folder).as_posix())
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)
            info = tar.gettarinfo(str(folder / "manifest.csv"), "manifest.csv")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with (folder / "manifest.csv").open("rb") as handle:
                tar.addfile(info, handle)
        archives[spec.key] = archive
    return archives


def sha256_of(path) -> str:
    """The SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload(spec: Movie, out: Path) -> None:
    """Publish one archive and card to its dataset repository."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(spec.repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(out / "README.md"),
                    path_in_repo="README.md", repo_id=spec.repo,
                    repo_type="dataset")
    api.upload_file(path_or_fileobj=str(out / spec.archive),
                    path_in_repo=spec.archive, repo_id=spec.repo,
                    repo_type="dataset")


def main(argv=None) -> int:
    """Build, and with ``--upload`` publish, the timelapse example sets."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--source", action="append", default=[],
                        metavar="KEY=DIR")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)
    sources = dict(item.split("=", 1) for item in args.source)
    archives = build(args.dest, sources, args.only)
    for key, archive in archives.items():
        print(f"{key} {archive} {archive.stat().st_size} bytes "
              f"sha256 {sha256_of(archive)}")
        if args.upload:
            upload(movie(key), archive.parent)
            print(f"published to {movie(key).repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
