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
        }
    if app_key == "crop":
        return {
            **base,
            **CHANNEL_LAYOUT,
            "save_png": True,
            "png_size": 64,
            "png_dims": [0, 1, 2],
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
    return base


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
    "mask":       generate_mask_demo,
    "measure":    generate_measure_demo,
    "crop":       generate_crop_demo,
    "classify":   generate_classify_demo,
    "timelapse":  generate_timelapse_demo,
}


def main(argv: Optional[list[str]] = None) -> int:
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
