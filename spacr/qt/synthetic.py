"""
Synthetic datasets + saved settings for exercising every pipeline app.

The goal: give a developer (or a bug reporter) a one-line way to
generate a demo folder that flows cleanly through every spacr
pipeline — mask, measure, crop, classify, timelapse — plus a matching
settings CSV that plugs into the "Import settings…" button on each
app screen.

Everything is reverse-engineered from what the pipelines actually
consume:

* Filenames match the cellvoyager regex in
  `spacr.utils._get_regex('.tif', 'cellvoyager')`:
      <plateID>_<wellID>_T<timeID>F<fieldID>L<laserID>A<AID>Z<sliceID>C<chanID>.tif
* Channels are laid out in the order every mask default expects:
      C0 = nucleus, C1 = cell, C2 = pathogen, C3 = organelle
* Images are 16-bit uint16 with realistic-looking Gaussian blobs so
  the segmentation apps don't just see noise.
* Settings CSVs are written in the two-column "Key,Value" format that
  `spacr.utils.load_settings` reads. Loading via the AppScreen's
  "Import settings…" button restores every value into the form.

Public API:
    generate_mask_demo(dst, ...) -> DemoLayout
    generate_measure_demo(dst, ...) -> DemoLayout
    generate_crop_demo(dst, ...) -> DemoLayout
    generate_classify_demo(dst, ...) -> DemoLayout
    generate_timelapse_demo(dst, ...) -> DemoLayout
    save_settings_csv(dst, settings) -> Path
    demo_settings(app_key, src) -> Dict[str, Any]

CLI:
    python -m spacr.qt.synthetic mask /tmp/demo
    python -m spacr.qt.synthetic all  /tmp/demo
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


LOG = logging.getLogger("spacr.qt.synthetic")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DemoLayout:
    """What a demo generator produced. Absolute paths only."""
    src: Path
    image_dir: Path
    image_files: List[Path] = field(default_factory=list)
    mask_files: List[Path] = field(default_factory=list)
    db_path: Optional[Path] = None
    settings_csv: Optional[Path] = None
    notes: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filename builder — matches spacr's cellvoyager regex
# ---------------------------------------------------------------------------

def cellvoyager_filename(
    plate: str = "plate1",
    well: str = "A01",
    time: int = 1,
    field: int = 1,
    laser: int = 1,
    a: int = 1,
    slice_: int = 1,
    chan: int = 1,
    ext: str = "tif",
) -> str:
    """Return a filename matching:
        <plateID>_<wellID>_T<timeID>F<fieldID>L<laserID>A<AID>Z<sliceID>C<chanID>.<ext>
    """
    return (
        f"{plate}_{well}"
        f"_T{time:02d}"
        f"F{field:02d}"
        f"L{laser:02d}"
        f"A{a:02d}"
        f"Z{slice_:02d}"
        f"C{chan:02d}.{ext}"
    )


# ---------------------------------------------------------------------------
# Synthetic image content
# ---------------------------------------------------------------------------

def _synth_blob_image(
    shape: Tuple[int, int] = (256, 256),
    n_blobs: int = 12,
    blob_radius: int = 14,
    intensity: int = 40000,
    seed: int = 0,
) -> np.ndarray:
    """A uint16 image with `n_blobs` Gaussian bright spots — realistic
    enough for cellpose to find things during a smoke test."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for _ in range(n_blobs):
        cy = rng.integers(blob_radius, h - blob_radius)
        cx = rng.integers(blob_radius, w - blob_radius)
        # Slight per-blob intensity + radius jitter
        r = blob_radius * (0.7 + 0.6 * rng.random())
        peak = intensity * (0.5 + 0.8 * rng.random())
        img += peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
    img += rng.normal(500, 120, shape)     # background noise
    return np.clip(img, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Generators — one per app family
# ---------------------------------------------------------------------------

# Channel layout every mask default expects. Keys are the settings
# names in spacr.settings that pipeline functions read.
CHANNEL_LAYOUT = {
    "nucleus_channel":   0,
    "cell_channel":      1,
    "pathogen_channel":  2,
    "organelle_channel": 3,
}


def _emit_images(
    image_dir: Path,
    plate: str,
    wells: Iterable[str],
    fields: int,
    channels: Iterable[int],
    times: int = 1,
    shape: Tuple[int, int] = (256, 256),
    n_blobs: int = 12,
) -> List[Path]:
    """Write a full set of cellvoyager-named .tif files."""
    from tifffile import imwrite as tif_write
    written: List[Path] = []
    for well in wells:
        for f in range(1, fields + 1):
            for t in range(1, times + 1):
                for c in channels:
                    fn = cellvoyager_filename(
                        plate=plate, well=well,
                        time=t, field=f, chan=c,
                    )
                    # Blob density varies per channel so segmentation
                    # finds different objects: nucleus dense, cell
                    # sparser, pathogen very sparse, organelle mid.
                    density = {0: n_blobs, 1: n_blobs // 2,
                                2: n_blobs // 4, 3: n_blobs // 2}.get(
                        c, n_blobs
                    )
                    img = _synth_blob_image(
                        shape=shape, n_blobs=density,
                        seed=hash((well, f, t, c)) & 0xFFFF,
                    )
                    p = image_dir / fn
                    tif_write(p, img)
                    written.append(p)
    LOG.info("wrote %d synthetic images to %s", len(written), image_dir)
    return written


def generate_mask_demo(
    dst: Path,
    plate: str = "plate1",
    wells: Iterable[str] = ("A01", "A02"),
    fields: int = 2,
    channels: Iterable[int] = (0, 1, 2, 3),
) -> DemoLayout:
    """Populate `dst` with a folder that runs cleanly through the Mask
    app. Layout:
        dst/
          <plateID>_<wellID>_T01F<field>L01A01Z01C<chan>.tif
          settings_mask.csv
    """
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    files = _emit_images(dst, plate, wells, fields, channels)
    settings = demo_settings("mask", str(dst))
    csv_path = save_settings_csv(dst / "settings_mask.csv", settings)
    return DemoLayout(
        src=dst, image_dir=dst,
        image_files=files, settings_csv=csv_path,
        notes={"channels": list(channels)},
    )


def generate_measure_demo(dst: Path, **kw) -> DemoLayout:
    """Measure consumes what mask produces: images + `masks/`
    subfolder + a `measurements/measurements.db` scaffold. We
    pre-build the masks so a user can jump straight into Measure."""
    from tifffile import imwrite as tif_write
    layout = generate_mask_demo(dst, **kw)
    dst = layout.src
    masks_dir = dst / "masks"
    masks_dir.mkdir(exist_ok=True)
    for img in layout.image_files:
        m = np.zeros((256, 256), dtype=np.uint16)
        m[50:100, 50:100] = 1
        m[150:200, 150:200] = 2
        tif_write(masks_dir / img.name, m)
        layout.mask_files.append(masks_dir / img.name)
    (dst / "measurements").mkdir(exist_ok=True)
    db_path = dst / "measurements" / "measurements.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS "png_list" (png_path TEXT PRIMARY KEY)'
        )
    layout.db_path = db_path
    settings = demo_settings("measure", str(dst))
    layout.settings_csv = save_settings_csv(dst / "settings_measure.csv", settings)
    LOG.info("measure demo ready at %s (%d images, %d masks, DB %s)",
              dst, len(layout.image_files), len(layout.mask_files), db_path)
    return layout


def generate_crop_demo(dst: Path, **kw) -> DemoLayout:
    """Same as measure — Crop reads images + masks and writes PNG
    crops into `data/` alongside the DB."""
    layout = generate_measure_demo(dst, **kw)
    (layout.src / "data").mkdir(exist_ok=True)
    settings = demo_settings("crop", str(layout.src))
    layout.settings_csv = save_settings_csv(
        layout.src / "settings_crop.csv", settings,
    )
    return layout


def generate_classify_demo(
    dst: Path, n_crops: int = 16,
) -> DemoLayout:
    """Classify wants PNG single-object crops + a `measurements.db`
    with a `png_list` table + an `annotate` column carrying class
    labels for training/testing."""
    from PIL import Image
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "measurements").mkdir(exist_ok=True)
    (dst / "data").mkdir(exist_ok=True)
    files: List[Path] = []
    rng = np.random.default_rng(0)
    for i in range(n_crops):
        # Alternate blob patterns to give the classifier something to
        # discriminate (label 1 = dense, label 2 = sparse).
        cls = 1 if i % 2 == 0 else 2
        arr = _synth_blob_image(
            shape=(64, 64), n_blobs=8 if cls == 1 else 2,
            seed=i,
        )
        # Save as an 8-bit RGB PNG (what spacr.io stores).
        arr8 = (arr / 256).astype(np.uint8)
        rgb = np.stack([arr8, arr8, arr8], axis=-1)
        p = dst / "data" / f"crop_{i:03d}.png"
        Image.fromarray(rgb).save(p)
        files.append(p)
    db_path = dst / "measurements" / "measurements.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS "png_list" ('
            ' png_path TEXT PRIMARY KEY,'
            ' annotate INTEGER)'
        )
        conn.executemany(
            'INSERT OR REPLACE INTO "png_list" (png_path, annotate) VALUES (?, ?)',
            [(str(f), (1 if i % 2 == 0 else 2)) for i, f in enumerate(files)],
        )
    layout = DemoLayout(
        src=dst, image_dir=dst / "data",
        image_files=files, db_path=db_path,
    )
    settings = demo_settings("classify", str(dst))
    layout.settings_csv = save_settings_csv(
        dst / "settings_classify.csv", settings,
    )
    LOG.info("classify demo ready at %s (%d annotated crops)",
              dst, len(files))
    return layout


def generate_timelapse_demo(
    dst: Path,
    plate: str = "plate1",
    wells: Iterable[str] = ("A01",),
    fields: int = 1,
    times: int = 8,
    channels: Iterable[int] = (0, 1),
) -> DemoLayout:
    """Timelapse needs multi-T frames per (well, field) so tracking
    has something to lock onto. Same cellvoyager naming, just with
    T01..T<N>."""
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    files = _emit_images(
        dst, plate, wells, fields, channels, times=times,
    )
    settings = demo_settings("timelapse", str(dst))
    settings["timelapse"] = True
    settings["timelapse_frame_limits"] = [1, times]
    layout = DemoLayout(
        src=dst, image_dir=dst,
        image_files=files,
        notes={"times": times, "channels": list(channels)},
    )
    layout.settings_csv = save_settings_csv(
        dst / "settings_timelapse.csv", settings,
    )
    return layout


# ---------------------------------------------------------------------------
# Settings — reverse-engineered per app so the demo actually runs
# ---------------------------------------------------------------------------

def demo_settings(app_key: str, src: str) -> Dict[str, Any]:
    """Return a spacr settings dict tailored for the demo dataset
    generated by `generate_<app>_demo`.

    Values are the minimum needed to make the pipeline flow — real
    users will tweak thresholds + channel numbers to fit their data.
    """
    base: Dict[str, Any] = {
        "src": src,
        "metadata_type": "cellvoyager",
        "custom_regex": None,
        "channels": [0, 1, 2, 3],
        "magnification": 20,
        "plot": False,
        "test_mode": False,
    }
    if app_key == "mask":
        return {
            **base,
            **CHANNEL_LAYOUT,
            "cell_diameter": 60,
            "nucleus_diameter": 30,
            "pathogen_diameter": 20,
            "cell_background": 100,
            "cell_signal_to_noise": 10,
            "cell_CP_prob": 0.0,
            "cell_FT": 1.0,
            "cell_model_name": "cyto",
            "nucleus_model_name": "nuclei",
        }
    if app_key == "measure":
        return {
            **base,
            **CHANNEL_LAYOUT,
            "cell_min_size": 50,
            "nucleus_min_size": 25,
            "pathogen_min_size": 15,
            "save_measurements": True,
            "timelapse": False,
            "experiment": "demo",
            "cells": [1],
            "nuclei": [1],
            "pathogens": [1],
        }
    if app_key == "crop":
        return {
            **base,
            **CHANNEL_LAYOUT,
            "save_png": True,
            "png_size": 64,
            "png_dims": [0, 1, 2],
            "timelapse": False,
            "experiment": "demo",
            "cells": [1],
            "nuclei": [1],
            "pathogens": [1],
        }
    if app_key == "classify":
        return {
            "src": src,
            "annotation_column": "annotate",
            "classes": [1, 2],
            "image_size": 64,
            "batch_size": 16,
            "epochs": 2,
            "model_type": "cnn",
            "channels": [0, 1, 2],
        }
    if app_key == "timelapse":
        return {
            **base,
            **CHANNEL_LAYOUT,
            "timelapse": True,
            "timelapse_frame_limits": [1, 8],
            "cell_diameter": 60,
        }
    if app_key == "map_barcodes":
        return {
            "src": src,
            "test": False,
            "barcode_length": 24,
            "barcode_offset": 34,          # 20 prefix + 14 middle
            "chunk_size": 1000,
            "processes": 2,
            "verbose": False,
        }
    return base


# ---------------------------------------------------------------------------
# Synthetic FASTQ generator — matches EO1_R1_001.fastq.gz structure
# ---------------------------------------------------------------------------

# NovaSeq X read layout observed in EO1_R1_001.fastq.gz:
#   header: @<instr>:<run>:<flowcell>:<lane>:<tile>:<x>:<y> 1:N:0:<i7>
#   seq   : 150 bp
#   qual  : 150 bp of Illumina 1.8+ Phred+33 scores
# Every read of the real fastq carried i7 index GCTTGCGC.
FASTQ_READ_LENGTH = 150
FASTQ_INSTRUMENT  = "LH00000"
FASTQ_RUN         = 1
FASTQ_FLOWCELL    = "SYNTHFC01"
FASTQ_LANE        = 1
FASTQ_I7_INDEX    = "GCTTGCGC"

# Real spaCR reads have a fixed adapter frame around the gRNA barcode.
# The design lets barcode-mapping tests recover a known plant of
# gRNA IDs from a synthetic read pool. Layout in real reads (positions
# are approximate — spaCR's mapper doesn't rely on hard offsets):
#   [0:20]   variable prefix / stagger
#   [20:44]  gRNA barcode region (24 bp cassette)
#   [44:150] downstream constant region + fill
_ADAPTER_PREFIX  = "ATTGGCCTTGCTGTTTCCAG"           # 20 bp
_ADAPTER_MIDDLE  = "CATAGCTCTTAAAC"                 # 14 bp
_ADAPTER_SUFFIX  = ("GACGCGGCACAAACTTGAAACCCCCATTTA"
                    "CCAGAAGCTAGATCGGAAGAGCACAT"
                    "GCCTAAATTCCAGCCATGTTT")        # ~66 bp


def _phred_run(length: int, mean_q: int = 30,
                seed: int = 0) -> str:
    """Generate a Phred+33 quality string of ``length`` chars with
    Illumina-plausible variability (higher quality up front, more
    dropouts toward the end)."""
    rng = np.random.default_rng(seed)
    scores = np.clip(
        rng.normal(loc=mean_q, scale=6, size=length).round().astype(int),
        2, 40,
    )
    # Fade quality toward the tail — real reads drop below Q20 near
    # the end. Roughly halve the base quality over the last third.
    tail = int(length * 0.33)
    scores[-tail:] = np.clip(scores[-tail:] - rng.integers(4, 12, tail),
                              2, 40)
    return "".join(chr(int(q) + 33) for q in scores)


def _random_gRNA_barcode(rng: np.random.Generator) -> str:
    """Random 24-mer of A/C/G/T (uniform)."""
    return "".join(rng.choice(list("ACGT"), size=24))


def _synth_read_seq(barcode: str, rng: np.random.Generator) -> str:
    """Build one 150-bp read: prefix + middle + barcode + suffix,
    trimmed / padded to exactly 150 chars."""
    body = _ADAPTER_PREFIX + _ADAPTER_MIDDLE + barcode + _ADAPTER_SUFFIX
    if len(body) < FASTQ_READ_LENGTH:
        # Pad with random bases so downstream stats look believable
        pad = "".join(rng.choice(list("ACGT"),
                                   size=FASTQ_READ_LENGTH - len(body)))
        body = body + pad
    return body[:FASTQ_READ_LENGTH]


def _fastq_header(index: int, tile: int = 1101,
                    y: Optional[int] = None) -> str:
    """Build one @-prefixed FASTQ header matching Illumina 1.8+ format."""
    x = 1000 + (index % 9000)          # 1000..9999
    y = y if y is not None else 1000 + (index // 9000)
    return (
        f"@{FASTQ_INSTRUMENT}:{FASTQ_RUN}:{FASTQ_FLOWCELL}"
        f":{FASTQ_LANE}:{tile}:{x}:{y} 1:N:0:{FASTQ_I7_INDEX}"
    )


def generate_barcode_fasta(dst: Path, n_barcodes: int = 12,
                            prefix: str = "gRNA_",
                            seed: int = 0) -> Path:
    """Write ``n_barcodes`` synthetic gRNA barcodes to a FASTA file
    that spacr.sequencing's barcode mapper can read.

    :param dst: output file path (``.fasta``).
    :param n_barcodes: how many unique barcodes to emit.
    :param prefix: id prefix for each entry (``>gRNA_0001``, …).
    :param seed: RNG seed for reproducible barcode sets.
    :returns: the resolved ``dst`` path.
    """
    dst = Path(dst).absolute()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with open(dst, "w") as f:
        for i in range(n_barcodes):
            bc = _random_gRNA_barcode(rng)
            f.write(f">{prefix}{i+1:04d}\n{bc}\n")
    LOG.info("wrote %d synthetic gRNA barcodes → %s", n_barcodes, dst)
    return dst


def generate_synthetic_fastq(
    dst: Path,
    barcodes_fasta: Path,
    n_reads: int = 5_000,
    reads_per_barcode_min: int = 50,
    seed: int = 0,
) -> Path:
    """Write a gzip-compressed synthetic FASTQ with reads that carry
    the barcodes from ``barcodes_fasta``.

    Distribution: each barcode gets at least ``reads_per_barcode_min``
    reads; remaining budget is spread with a moderate Zipf tail so
    downstream frequency plots have realistic shape.

    :param dst: output path — appended with ``.gz`` if not present.
    :param barcodes_fasta: FASTA of gRNA barcodes (:func:`generate_barcode_fasta`).
    :param n_reads: total number of reads to emit.
    :param reads_per_barcode_min: floor of reads per barcode.
    :param seed: RNG seed for reproducible read pools.
    :returns: resolved ``.fastq.gz`` path.
    """
    import gzip
    dst = Path(dst).absolute()
    if dst.suffix != ".gz":
        dst = dst.with_suffix(dst.suffix + ".gz")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Load barcodes
    barcodes: List[str] = []
    with open(barcodes_fasta) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                barcodes.append(line)
    if not barcodes:
        raise ValueError(f"no barcodes found in {barcodes_fasta}")

    rng = np.random.default_rng(seed)
    # Allocate: N floor + zipf tail over the remaining budget
    floor = reads_per_barcode_min * len(barcodes)
    if n_reads < floor:
        counts = [max(1, n_reads // len(barcodes))] * len(barcodes)
    else:
        counts = [reads_per_barcode_min] * len(barcodes)
        remaining = n_reads - floor
        # Draw remaining barcode picks from a mild Zipf so a few
        # barcodes are heavily represented — realistic screen behaviour.
        picks = rng.zipf(1.5, size=remaining) % len(barcodes)
        for p in picks:
            counts[int(p)] += 1

    LOG.info("emitting %d reads across %d barcodes → %s",
              sum(counts), len(barcodes), dst)
    read_idx = 0
    with gzip.open(dst, "wt") as f:
        for i, (bc, n) in enumerate(zip(barcodes, counts)):
            for _ in range(n):
                header = _fastq_header(read_idx)
                seq = _synth_read_seq(bc, rng)
                qual = _phred_run(len(seq), mean_q=32, seed=read_idx)
                f.write(f"{header}\n{seq}\n+\n{qual}\n")
                read_idx += 1
    return dst


def generate_map_barcodes_demo(
    dst: Path,
    n_barcodes: int = 12,
    n_reads: int = 5_000,
    seed: int = 0,
) -> DemoLayout:
    """Populate ``dst`` with a self-contained map_barcodes demo:

    ::

        dst/
          barcodes/
            grnas.fasta         # ← N barcodes
          fastq/
            synthetic_R1.fastq.gz  # ← reads carrying those barcodes
          settings_map_barcodes.csv

    :param dst: destination folder.
    :param n_barcodes: number of unique gRNA barcodes to plant.
    :param n_reads: total number of reads to emit.
    :param seed: RNG seed for reproducibility.
    :returns: :class:`DemoLayout` describing the emitted files.
    """
    dst = Path(dst).absolute()
    (dst / "barcodes").mkdir(parents=True, exist_ok=True)
    (dst / "fastq").mkdir(parents=True, exist_ok=True)

    fasta = generate_barcode_fasta(
        dst / "barcodes" / "grnas.fasta",
        n_barcodes=n_barcodes, seed=seed,
    )
    fastq = generate_synthetic_fastq(
        dst / "fastq" / "synthetic_R1.fastq",
        barcodes_fasta=fasta, n_reads=n_reads, seed=seed,
    )
    settings = demo_settings("map_barcodes", str(dst))
    settings["fastq"] = str(fastq)
    settings["barcode_fasta"] = str(fasta)
    csv_path = save_settings_csv(dst / "settings_map_barcodes.csv", settings)
    return DemoLayout(
        src=dst, image_dir=dst / "fastq",
        image_files=[fastq], settings_csv=csv_path,
        notes={"n_reads": n_reads, "n_barcodes": n_barcodes,
                "barcodes_fasta": str(fasta)},
    )


# ---------------------------------------------------------------------------
# Settings CSV — spacr's own load_settings format
# ---------------------------------------------------------------------------

def save_settings_csv(path: Path, settings: Dict[str, Any]) -> Path:
    """Write `settings` in the two-column Key,Value format that
    `spacr.utils.load_settings` reads."""
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Key", "Value"])
        for k, v in settings.items():
            w.writerow([k, "" if v is None else str(v)])
    LOG.info("saved settings CSV → %s (%d keys)", path, len(settings))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_GENERATORS = {
    "mask":         generate_mask_demo,
    "measure":      generate_measure_demo,
    "crop":         generate_crop_demo,
    "classify":     generate_classify_demo,
    "timelapse":    generate_timelapse_demo,
    "map_barcodes": generate_map_barcodes_demo,
}


def main(argv: Optional[list[str]] = None) -> int:
    """Generate one (or every) demo dataset via the ``python -m`` CLI.

    :param argv: optional argv list; defaults to ``sys.argv[1:]``.
    :returns: process exit code (0 on success).
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m spacr.qt.synthetic",
        description="Generate a demo dataset + settings CSV for a "
                    "spacr pipeline app.",
    )
    p.add_argument(
        "app", choices=list(_GENERATORS.keys()) + ["all"],
        help="Which app's demo to generate.",
    )
    p.add_argument("dst", help="Destination folder.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.app == "all":
        for name, fn in _GENERATORS.items():
            layout = fn(Path(args.dst) / name)
            LOG.info("[%s] %s → %s", name, layout.src,
                      layout.settings_csv.name if layout.settings_csv else "-")
    else:
        layout = _GENERATORS[args.app](Path(args.dst))
        LOG.info("[%s] demo ready at %s", args.app, layout.src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
