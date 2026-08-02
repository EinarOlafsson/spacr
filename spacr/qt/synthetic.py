"""
Synthetic datasets + saved settings for exercising every pipeline app.

The goal: give a developer (or a bug reporter) a one-line way to
generate a demo folder that flows cleanly through every spacr
pipeline — mask, measure, crop, classify, timelapse, map_barcodes —
plus a matching settings CSV that plugs into the "Import settings…"
button on each app screen.

Everything is reverse-engineered from what the pipelines actually
consume:

* Filenames match the cellvoyager regex in
  `spacr.utils._get_regex('.tif', 'cellvoyager')`:
      <plateID>_<wellID>_T<timeID>F<fieldID>L<laserID>A<AID>Z<sliceID>C<chanID>.tif
* Channels are laid out in the order every mask default expects:
      C0 = nucleus, C1 = cell, C2 = pathogen, C3 = organelle
* Images are 16-bit uint16. Every channel of a field is drawn from
  *one shared cell layout*, so the nucleus really is inside the cell
  and the pathogen really is inside the same cell — the relationships
  measure_crop goes looking for when it links objects.
* Measure/crop demos ship the `merged/*.npy` stacks measure_crop
  actually reads (image planes first, then the label-mask planes),
  not a stand-in.
* Settings CSVs are written in the two-column "Key,Value" format that
  `spacr.utils.load_settings` reads. Loading via the AppScreen's
  "Import settings…" button restores every value into the form.

Every generator is *reproducible*: identical inputs give byte-identical
output on any machine. That is not decoration. Two people comparing
"the demo fails here" have to be looking at the same pixels, and the
previous seeding (``hash((well, field, time, chan))``) was salted by
PYTHONHASHSEED, so it changed on every interpreter start.

Public API:
    generate_mask_demo(dst, ...) -> DemoLayout
    generate_measure_demo(dst, ...) -> DemoLayout
    generate_crop_demo(dst, ...) -> DemoLayout
    generate_classify_demo(dst, ...) -> DemoLayout
    generate_timelapse_demo(dst, ...) -> DemoLayout
    generate_map_barcodes_demo(dst, ...) -> DemoLayout
    save_settings_csv(dst, settings) -> Path
    demo_settings(app_key, src, channels=None) -> Dict[str, Any]

CLI:
    python -m spacr.qt.synthetic mask /tmp/demo
    python -m spacr.qt.synthetic all  /tmp/demo
"""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LOG = logging.getLogger("spacr.qt.synthetic")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DemoLayout:
    """What a demo generator produced. Absolute paths only.

    ``merged_files`` replaces the old ``mask_files``: nothing in spaCR reads a
    folder of standalone label tiffs, and the measure/crop demos now ship the
    ``merged/*.npy`` stacks measure_crop actually opens — the label planes are
    the trailing planes of those arrays.
    """
    src: Path
    image_dir: Path
    image_files: List[Path] = field(default_factory=list)
    merged_files: List[Path] = field(default_factory=list)
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


def _stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from any mix of strings and numbers.

    ``hash()`` on a str is salted per interpreter (PYTHONHASHSEED), so the
    previous ``hash((well, field, time, chan))`` handed a *different*
    dataset to every run of the generator. crc32 over a canonical string is
    stable across processes, machines and Python versions, which is what
    "reproduce my bug report" needs.
    """
    text = "|".join(str(p) for p in parts)
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Synthetic image content
# ---------------------------------------------------------------------------

# Channel layout every mask default expects. Keys are the settings
# names in spacr.settings that pipeline functions read.
CHANNEL_LAYOUT = {
    "nucleus_channel":   0,
    "cell_channel":      1,
    "pathogen_channel":  2,
    "organelle_channel": 3,
}

#: Mask planes are appended to merged/*.npy in this order — see
#: spacr.io._load_and_concatenate_arrays, which walks
#: cell → nucleus → pathogen → organelle. `*_mask_dim` indexes that axis, so
#: the order here is what makes demo_settings' mask dims correct.
MASK_ROLE_ORDER: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "organelle")

#: Field geometry. A 4x4 lattice of cells in a 256x256 field puts 16 cells,
#: 16 nuclei and ~16 pathogens in *every* field. That number is not cosmetic:
#: spacr.seg_qc flags any field holding fewer than seg_qc_min_objects (10)
#: objects as `near_empty_field`, because a robust size statistic taken over
#: a handful of objects is one object's opinion rather than a distribution.
#: The generator this replaced emitted 12/6/3/6 blobs per channel and every
#: pathogen field came back near-empty.
FIELD_SHAPE: Tuple[int, int] = (256, 256)
CELL_GRID = 4
CELL_JITTER = 9.0

#: Sigma of the Gaussian drawn for each object, in pixels. Cellpose returns a
#: mask roughly 1.2-1.4 sigma in radius on this kind of image, so these are
#: chosen to land on the *_RADIUS_* values below — which are in turn what the
#: demo's *_diameter settings advertise.
_SIGMA_CELL = 16.0
_SIGMA_NUCLEUS = 6.5
_SIGMA_PATHOGEN = 4.0
#: Organelles are drawn *punctate*, and that is the whole point. spaCR's
#: default organelle segmenter is `organelle_morphology='spots'` +
#: `organelle_method='otsu'`, which white-top-hats the channel with disk(5)
#: before thresholding. A top hat annihilates anything wider than its
#: structuring element, so the 14-px-radius blobs this channel used to carry
#: were erased and otsu then thresholded the residual *noise*: ~230 five-pixel
#: specks per field, which seg_qc correctly failed as `over_segmented`.
#: Puncta at sigma 3.5 survive the top hat and come back at ~7.6 px across.
_SIGMA_ORGANELLE = 3.5

#: Radii of the label discs written into merged/*.npy for the measure and
#: crop demos. Nested on purpose: nucleus and pathogen both sit wholly inside
#: their cell, which is the containment measure_crop's object linking (and
#: the cytoplasm = cell - nucleus - pathogen subtraction) assumes.
_RADIUS_CELL = 20
_RADIUS_NUCLEUS = 8
_RADIUS_PATHOGEN = 5
_RADIUS_ORGANELLE = 4

#: Distance from the cell centre at which pathogens and organelles are placed.
#: Both keep the object inside the cell disc and clear of the nucleus disc.
_OFFSET_PATHOGEN = 13.0
_OFFSET_ORGANELLE = 12.0

#: What fraction of cells is infected, and how many pathogens each infected
#: cell carries (inclusive range). The *count* of infected cells is fixed
#: rather than a per-cell coin flip on purpose: seg_qc warns `near_empty_field`
#: on any field holding fewer than seg_qc_min_objects (10) objects, and a
#: Bernoulli draw over 16 cells put roughly one field in 75 below that floor.
#: A demo that is clean most of the time is not clean. 12 infected cells x 1-2
#: pathogens is 12-24 per field, never fewer than 12, and still leaves a
#: quarter of the cells uninfected — which is what the cell/pathogen link
#: measure_crop reports is about.
_INFECTED_FRACTION = 0.75
_PATHOGENS_PER_INFECTED_CELL = (1, 2)

#: Organelle puncta per cell: 64 per field.
_ORGANELLES_PER_CELL = 4

#: Background level and read noise, in raw 16-bit units. Real widefield data
#: sits on a camera offset with Poisson-ish noise; these are the numbers the
#: demo's *_background / *_Signal_to_noise settings are consistent with.
_BACKGROUND = 400
_NOISE = 90

#: Peak intensity of a drawn object before per-object jitter.
_PEAK = 42000


def _draw_spots(
    shape: Tuple[int, int],
    spots: Sequence[Tuple[float, float, float, float]],
    rng: np.random.Generator,
    background: float = _BACKGROUND,
    noise: float = _NOISE,
) -> np.ndarray:
    """Render ``(cy, cx, sigma, peak)`` Gaussians onto a noisy background.

    Each Gaussian is drawn into a local ``4*sigma`` window rather than over
    the whole meshgrid: a field now carries ~100 spots (16 cells + 16
    pathogens + 64 organelle puncta) and the whole-image version cost
    O(n_spots * h * w), which is seconds per field once the puncta arrived.
    """
    h, w = shape
    img = rng.normal(background, noise, shape).astype(np.float32)
    for cy, cx, sigma, peak in spots:
        radius = int(np.ceil(4.0 * sigma))
        y0, y1 = max(int(cy) - radius, 0), min(int(cy) + radius + 1, h)
        x0, x1 = max(int(cx) - radius, 0), min(int(cx) + radius + 1, w)
        if y0 >= y1 or x0 >= x1:
            continue
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, :]
        img[y0:y1, x0:x1] += peak * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2)
        )
    return np.clip(img, 0, 65535).astype(np.uint16)


def _paint_disc(mask: np.ndarray, cy: float, cx: float,
                radius: float, label: int) -> None:
    """Paint a filled disc of ``label`` into ``mask``, clipped at the edges."""
    h, w = mask.shape
    r = int(np.ceil(radius))
    y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, h)
    x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, w)
    if y0 >= y1 or x0 >= x1:
        return
    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    sub = mask[y0:y1, x0:x1]
    sub[inside] = label


def _synth_blob_image(
    shape: Tuple[int, int] = (256, 256),
    n_blobs: int = 12,
    blob_radius: int = 14,
    intensity: int = 40000,
    seed: int = 0,
) -> np.ndarray:
    """A uint16 image with `n_blobs` Gaussian bright spots.

    Kept for the classify demo, whose crops are single objects on a tile and
    need no cell layout. Whole *plates* come from :func:`_synth_field`.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    spots = []
    for _ in range(n_blobs):
        cy = float(rng.integers(blob_radius, max(h - blob_radius, blob_radius + 1)))
        cx = float(rng.integers(blob_radius, max(w - blob_radius, blob_radius + 1)))
        # Slight per-blob intensity + radius jitter
        r = blob_radius * (0.7 + 0.6 * rng.random())
        peak = intensity * (0.5 + 0.8 * rng.random())
        spots.append((cy, cx, r, peak))
    return _draw_spots(shape, spots, rng, background=500, noise=120)


@dataclass
class _Field:
    """One synthetic field: the acquisition channels and the truth masks.

    ``images`` is keyed by acquisition channel index (0-3, matching
    :data:`CHANNEL_LAYOUT`); ``masks`` by role name (``'cell'``,
    ``'nucleus'``, ``'pathogen'``, ``'organelle'``). They describe the same
    objects — the mask is where the Gaussians were drawn — which is what
    lets the measure demo ship a merged stack without running Cellpose.
    """
    images: Dict[int, np.ndarray]
    masks: Dict[str, np.ndarray]


def _synth_field(
    seed: int,
    channels: Sequence[int] = (0, 1, 2, 3),
    shape: Tuple[int, int] = FIELD_SHAPE,
    frame: int = 0,
) -> _Field:
    """Build one field: a lattice of cells, each with a nucleus, most with
    one or two pathogens, and each with a rosette of organelle puncta.

    :param seed: reproducible seed for the *field* — the same seed with a
        different ``frame`` returns the same cells, moved.
    :param channels: which acquisition channels to render.
    :param shape: field size in pixels.
    :param frame: zero-based timepoint. Each cell carries a constant velocity
        drawn from the field seed, so consecutive frames of a timelapse hold
        the *same* cells a few pixels away — without that a tracker has
        nothing to lock onto, which is the point of the timelapse demo.
    :returns: a :class:`_Field`.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    step_y, step_x = h / CELL_GRID, w / CELL_GRID
    wanted = {int(c) for c in channels}

    cells: List[Tuple[float, float, float]] = []   # (cy, cx, size_scale)
    margin = _RADIUS_CELL + 2
    for gy in range(CELL_GRID):
        for gx in range(CELL_GRID):
            cy = (gy + 0.5) * step_y + rng.uniform(-CELL_JITTER, CELL_JITTER)
            cx = (gx + 0.5) * step_x + rng.uniform(-CELL_JITTER, CELL_JITTER)
            vy, vx = rng.normal(0.0, 1.2, 2)
            cy = float(np.clip(cy + frame * vy, margin, h - margin))
            cx = float(np.clip(cx + frame * vx, margin, w - margin))
            # Per-cell size jitter. Without it every object has the identical
            # area, the median absolute deviation of the size distribution is
            # zero, and seg_qc's robust range collapses to a point so that
            # every object reads as a size outlier.
            cells.append((cy, cx, 0.82 + 0.36 * rng.random()))

    images: Dict[int, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    def _peak() -> float:
        return _PEAK * (0.75 + 0.45 * rng.random())

    # --- cell + nucleus: concentric, one per lattice site -----------------
    for role, sigma, radius in (
        ("cell", _SIGMA_CELL, _RADIUS_CELL),
        ("nucleus", _SIGMA_NUCLEUS, _RADIUS_NUCLEUS),
    ):
        chan = CHANNEL_LAYOUT[f"{role}_channel"]
        mask = np.zeros(shape, dtype=np.uint16)
        spots = []
        for label, (cy, cx, scale) in enumerate(cells, start=1):
            spots.append((cy, cx, sigma * scale, _peak()))
            _paint_disc(mask, cy, cx, radius * scale, label)
        masks[role] = mask
        if chan in wanted:
            images[chan] = _draw_spots(shape, spots, rng)

    # --- pathogens: inside the cell and clear of its nucleus --------------
    path_mask = np.zeros(shape, dtype=np.uint16)
    path_spots = []
    label = 0
    n_infected = int(round(_INFECTED_FRACTION * len(cells)))
    infected = set(rng.permutation(len(cells))[:n_infected].tolist())
    for index, (cy, cx, scale) in enumerate(cells):
        if index not in infected:
            continue
        n = int(rng.integers(_PATHOGENS_PER_INFECTED_CELL[0],
                             _PATHOGENS_PER_INFECTED_CELL[1] + 1))
        base_angle = rng.uniform(0, 2 * np.pi)
        for k in range(n):
            # Evenly spaced around the cell centre, so several pathogens in
            # one cell never land on top of each other and the mask holds n
            # objects rather than one peanut.
            angle = base_angle + 2 * np.pi * k / n
            py = cy + _OFFSET_PATHOGEN * scale * np.sin(angle)
            px = cx + _OFFSET_PATHOGEN * scale * np.cos(angle)
            label += 1
            path_spots.append((py, px, _SIGMA_PATHOGEN * scale, _peak()))
            _paint_disc(path_mask, py, px, _RADIUS_PATHOGEN * scale, label)
    masks["pathogen"] = path_mask
    if CHANNEL_LAYOUT["pathogen_channel"] in wanted:
        images[CHANNEL_LAYOUT["pathogen_channel"]] = _draw_spots(shape, path_spots, rng)

    # --- organelles: a rosette of puncta per cell -------------------------
    org_mask = np.zeros(shape, dtype=np.uint16)
    org_spots = []
    label = 0
    for cy, cx, scale in cells:
        base_angle = rng.uniform(0, 2 * np.pi)
        for k in range(_ORGANELLES_PER_CELL):
            angle = base_angle + 2 * np.pi * k / _ORGANELLES_PER_CELL
            oy = cy + _OFFSET_ORGANELLE * scale * np.sin(angle)
            ox = cx + _OFFSET_ORGANELLE * scale * np.cos(angle)
            label += 1
            org_spots.append((oy, ox, _SIGMA_ORGANELLE * scale, _peak()))
            _paint_disc(org_mask, oy, ox, _RADIUS_ORGANELLE * scale, label)
    masks["organelle"] = org_mask
    if CHANNEL_LAYOUT["organelle_channel"] in wanted:
        images[CHANNEL_LAYOUT["organelle_channel"]] = _draw_spots(shape, org_spots, rng)

    # Any channel the caller asked for that is not one of the four roles gets
    # plain background rather than a KeyError further down.
    for chan in sorted(wanted - set(images)):
        images[chan] = _draw_spots(shape, [], rng)

    return _Field(images=images, masks=masks)


# ---------------------------------------------------------------------------
# Generators — one per app family
# ---------------------------------------------------------------------------

def _mask_roles(channels: Sequence[int]) -> List[str]:
    """Which mask planes a plate with ``channels`` acquired would carry.

    Mirrors spacr.io._load_and_concatenate_arrays: a mask stack is appended
    only for objects whose channel was set, in :data:`MASK_ROLE_ORDER`.
    """
    wanted = {int(c) for c in channels}
    return [role for role in MASK_ROLE_ORDER
            if CHANNEL_LAYOUT[f"{role}_channel"] in wanted]


def _emit_images(
    image_dir: Path,
    plate: str,
    wells: Iterable[str],
    fields: int,
    channels: Iterable[int],
    times: int = 1,
    shape: Tuple[int, int] = FIELD_SHAPE,
) -> Tuple[List[Path], Dict[Tuple[str, int, int], _Field]]:
    """Write a full set of cellvoyager-named .tif files.

    :returns: ``(written_paths, fields)`` where ``fields`` is keyed by
        ``(well, field, time)``. The caller gets the truth masks back so the
        measure/crop demos can write a merged stack that agrees with the
        pixels, instead of inventing one.
    """
    from tifffile import imwrite as tif_write
    channels = [int(c) for c in channels]
    written: List[Path] = []
    produced: Dict[Tuple[str, int, int], _Field] = {}
    for well in wells:
        for f in range(1, fields + 1):
            # One seed per (well, field): every timepoint of that field is
            # the same cells, drifting.
            seed = _stable_seed(plate, well, f)
            for t in range(1, times + 1):
                fld = _synth_field(seed, channels=channels, shape=shape,
                                   frame=t - 1)
                produced[(well, f, t)] = fld
                for c in channels:
                    fn = cellvoyager_filename(
                        plate=plate, well=well,
                        time=t, field=f, chan=c,
                    )
                    p = image_dir / fn
                    tif_write(p, fld.images[c])
                    written.append(p)
    LOG.info("wrote %d synthetic images to %s", len(written), image_dir)
    return written, produced


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
    channels = [int(c) for c in channels]
    files, produced = _emit_images(dst, plate, wells, fields, channels)
    settings = demo_settings("mask", str(dst), channels=channels)
    csv_path = save_settings_csv(dst / "settings_mask.csv", settings)
    return DemoLayout(
        src=dst, image_dir=dst,
        image_files=files, settings_csv=csv_path,
        notes={"channels": list(channels), "plate": plate,
               "n_fields": len(produced),
               "cells_per_field": CELL_GRID * CELL_GRID},
    )


def _write_merged(
    dst: Path,
    plate: str,
    produced: Dict[Tuple[str, int, int], _Field],
    channels: Sequence[int],
) -> List[Path]:
    """Write the ``merged/*.npy`` stacks measure_crop reads.

    One array per field, shaped ``(H, W, len(channels) + n_masks)``: the
    selected image planes first, then the label-mask planes in
    :data:`MASK_ROLE_ORDER` — byte-for-byte the layout
    spacr.io._load_and_concatenate_arrays produces at the end of a Mask run,
    which is what makes ``cell_mask_dim`` and friends mean the same thing
    here as they do on a real plate.

    The filename is ``<plate>_<well>_<field>_<time>.npy``, the field name
    spaCR carries through every measurement table.
    """
    merged_dir = dst / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    roles = _mask_roles(channels)
    written: List[Path] = []
    for (well, fld_no, time), fld in sorted(produced.items()):
        planes = [fld.images[int(c)] for c in channels]
        planes += [fld.masks[role] for role in roles]
        arr = np.stack(planes, axis=-1).astype(np.uint16)
        out = merged_dir / f"{plate}_{well}_{fld_no}_{time}.npy"
        np.save(out, arr)
        written.append(out)
    LOG.info("wrote %d merged arrays (%d planes each) to %s",
             len(written), len(channels) + len(roles), merged_dir)
    return written


def generate_measure_demo(
    dst: Path,
    plate: str = "plate1",
    wells: Iterable[str] = ("A01", "A02"),
    fields: int = 2,
    channels: Iterable[int] = (0, 1, 2, 3),
) -> DemoLayout:
    """Measure consumes what Mask produces: a ``merged/`` folder of
    ``.npy`` stacks whose trailing planes are the label masks.

    We pre-build those stacks so a user can jump straight into Measure
    without a GPU. Before this, the demo wrote a ``masks/`` folder of
    per-*file* tiffs and an empty ``measurements.db`` — neither of which any
    pipeline reads — and measure's pre-flight rejected the folder outright
    with "no merged folder for measure".

    .. note::
       The organelle plane this writes is measured into nothing, and the
       defect is not in this module. Plane ``organelle_mask_dim`` of every
       merged stack carries 64 real labels, but a measure run over this folder
       writes ``cell``/``nucleus``/``pathogen``/``cytoplasm`` and **no
       organelle table at all**. Cause: every organelle write in
       ``spacr.measure._measure_crop_core`` is gated on
       ``settings.get('summarize_organelles_by') is not None``, and
       ``spacr.settings.get_measure_crop_settings`` — the defaults a measure
       run is canonicalised through — never sets that key. Only
       ``set_default_settings_preprocess_generate_masks`` (the *Mask* app)
       defaults it, to ``'cell'``. Passing ``summarize_organelles_by=['cell',
       'organelle']`` by hand makes the ``organelle`` (256 rows) and
       ``cell_organelle_summary`` (64 rows) tables appear, so the data is
       fine. The demo cannot ship the key as a workaround: it is absent from
       the Measure app's defaults, so the Qt Measure screen has no widget for
       it and ``_apply_demo_to_screen`` would drop it — a setting that is in
       the CSV, absent from the form, and silently changes what is measured.
       The fix belongs in ``spacr/settings.py`` (default it for measure) and
       ``spacr/qt/screens/settings_model.py`` (give it a widget).
    """
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    channels = [int(c) for c in channels]
    files, produced = _emit_images(dst, plate, wells, fields, channels)
    merged = _write_merged(dst, plate, produced, channels)
    settings = demo_settings("measure", str(dst), channels=channels)
    layout = DemoLayout(
        src=dst, image_dir=dst,
        image_files=files, merged_files=merged,
        settings_csv=save_settings_csv(dst / "settings_measure.csv", settings),
        notes={"channels": list(channels), "plate": plate,
               "mask_roles": _mask_roles(channels)},
    )
    LOG.info("measure demo ready at %s (%d images, %d merged stacks)",
             dst, len(layout.image_files), len(merged))
    return layout


def generate_crop_demo(
    dst: Path,
    plate: str = "plate1",
    wells: Iterable[str] = ("A01", "A02"),
    fields: int = 2,
    channels: Iterable[int] = (0, 1, 2, 3),
) -> DemoLayout:
    """Same dataset as measure — Crop is measure with ``save_png`` on, and
    writes PNG crops into per-object folders alongside the DB."""
    layout = generate_measure_demo(
        dst, plate=plate, wells=wells, fields=fields, channels=channels)
    settings = demo_settings("crop", str(layout.src),
                             channels=[int(c) for c in channels])
    layout.settings_csv = save_settings_csv(
        layout.src / "settings_crop.csv", settings,
    )
    return layout


#: Where measure_crop puts single-cell crops, and therefore where the
#: Classify/Annotate demo has to put them too. spacr.io.generate_training_dataset
#: filters png_list on ``png_path.str.contains(png_type)`` with png_type
#: 'cell_png', so a crop at ``data/crop_000.png`` is dropped before any class
#: is built — the dataset comes back with zero images and the run dies on
#: "got 0 classes".
CROP_FOLDER = "cell_png"

#: Crops per field in the classify demo. The well count matters more than the
#: crop count: spacr.io.make_validation_holdout groups folds by `cv_group_by`
#: ('well' by default) and stops with "needs at least two distinct groups".
#: That check runs on the *training* half, so the demo needs enough wells that
#: the train/test split still leaves two of them on the training side — two
#: wells total is not enough, four is.
_CROPS_PER_FIELD = 8


def generate_classify_demo(
    dst: Path, n_crops: int = 64, plate: str = "plate1",
    wells: Sequence[str] = ("A01", "A02", "A03", "A04"),
) -> DemoLayout:
    """Classify wants PNG single-object crops + a `measurements.db`
    with a `png_list` table + an `annotate` column carrying class
    labels for training/testing.

    This is a hand-built stand-in for a measured plate, not a replica of one.
    Two of the three things that matter match measure_crop; the third does
    not, and the docstring used to claim all three did.

    **Crop names match.** A real crop is ``<file_name>_<cell_id>.png`` where
    ``file_name`` is the merged stack's ``<plate>_<well>_<field>_<time>``
    (:func:`spacr.utils._generate_names`) — e.g. ``plate1_A01_1_1_1.png``.
    That is exactly what this writes, and exactly what
    ``spacr.utils._map_wells_png`` parses plate/row/column/field back out of.

    **The ``cell_png`` leaf matches.** measure_crop appends
    ``f"{crop_mode}_png/"`` to the folder, so a real cell crop does live under
    ``cell_png/`` — which is what
    :func:`spacr.io.generate_training_dataset`'s
    ``png_path.str.contains(png_type)`` filter (``png_type='cell_png'``) needs
    to see. Crops written flat as ``data/crop_000.png``, the layout this
    replaced, were all filtered away and the run died on "got 0 classes".

    **The folder above it does not match.** measure_crop buckets every crop by
    what it contains first:
    ``data/<single|multiple|no>_nucleus/<single_pathogen|multiple_pathogens|uninfected>/<plate>_<well>/cell_png/``.
    This demo writes ``data/<plate>_<well>/cell_png/`` with no bucket
    folders — nothing downstream of ``png_type`` reads them, and inventing an
    infection status per synthetic crop would be a fiction the pixels do not
    support.

    **The ``png_list`` columns do not match either.**
    :func:`spacr.utils.filepaths_to_database` writes ``png_path, file_name,
    plateID, rowID, columnID, fieldID, prcfo, cell_id``, with the tokenised
    values ``rowID='r1'``/``columnID='c1'``/``fieldID='f1'``. This table
    carries ``png_path, plateID, wellID, rowID, columnID, fieldID, timeID,
    label`` in plain form, plus the ``annotate`` column — which is the point:
    ``annotate`` is what ``dataset_mode='annotation'`` selects classes on, and
    a measure run never writes one. A human does, in the Annotate screen.

    :param dst: destination folder.
    :param n_crops: total number of crops, spread evenly over the wells.
    :param plate: plate ID baked into the crop names and png_list.
    :param wells: well IDs to spread the crops over; at least two, so the
        classifier can hold a whole well out.
    :returns: :class:`DemoLayout`.
    """
    from PIL import Image
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "measurements").mkdir(exist_ok=True)
    wells = list(wells)
    per_well = max(1, n_crops // len(wells))
    files: List[Path] = []
    rows: List[Tuple[Any, ...]] = []
    i = 0
    for well in wells:
        crop_dir = dst / "data" / f"{plate}_{well}" / CROP_FOLDER
        crop_dir.mkdir(parents=True, exist_ok=True)
        for k in range(per_well):
            # Alternate blob patterns to give the classifier something to
            # discriminate (label 1 = dense, label 2 = sparse).
            cls = 1 if k % 2 == 0 else 2
            arr = _synth_blob_image(
                shape=(64, 64), n_blobs=8 if cls == 1 else 2,
                blob_radius=6,
                seed=i,
            )
            # Save as an 8-bit RGB PNG (what spacr.io stores).
            arr8 = (arr / 256).astype(np.uint8)
            rgb = np.stack([arr8, arr8, arr8], axis=-1)
            # Fields of _CROPS_PER_FIELD objects each, so the crops carry the
            # same plate/well/field/time/label name spaCR parses metadata back
            # out of.
            field = k // _CROPS_PER_FIELD + 1
            label = k % _CROPS_PER_FIELD + 1
            p = crop_dir / f"{plate}_{well}_{field}_1_{label}.png"
            Image.fromarray(rgb).save(p)
            files.append(p)
            rows.append((str(p), plate, well, well[0], well[1:],
                         field, 1, label, cls))
            i += 1
    db_path = dst / "measurements" / "measurements.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS "png_list" ('
            ' png_path TEXT PRIMARY KEY,'
            ' plateID TEXT, wellID TEXT, rowID TEXT, columnID TEXT,'
            ' fieldID INTEGER, timeID INTEGER, label INTEGER,'
            ' annotate INTEGER)'
        )
        conn.executemany(
            'INSERT OR REPLACE INTO "png_list" ('
            ' png_path, plateID, wellID, rowID, columnID, fieldID, timeID,'
            ' label, annotate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            rows,
        )
    layout = DemoLayout(
        src=dst, image_dir=dst / "data",
        image_files=files, db_path=db_path,
        notes={"plate": plate, "wells": list(wells),
               "crops_per_well": per_well},
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
    T01..T<N>, and every frame holds the *same* cells drifting a couple of
    pixels rather than a fresh random field.

    Only the nucleus and cell channels are acquired, and the settings say so:
    the base settings used to advertise ``channels=[0,1,2,3]`` plus a
    ``pathogen_channel``/``organelle_channel`` this dataset never had, which
    pre-flight rejected with two hard errors before the run could start.

    .. note::
       The dataset and settings this writes clear pre-flight, but the
       Timelapse *pipeline* still cannot consume them, and that defect is not
       in this module. ``spacr.io._rename_and_organize_image_files`` names its
       stack files ``<plate>_<well>_<field>.npy`` when ``timelapse=True`` —
       dropping the timeID and max-projecting every timepoint of a field into
       one array — while ``spacr.io._generate_time_lists`` groups on
       ``<plate>_<well>_<field>_<time>.npy`` and skips anything with fewer
       than four underscore-separated parts. So no ``*_norm_timelapse.npz`` is
       written, no masks are generated, and ``preprocess_generate_masks``
       finally dies in ``_pivot_counts_table`` on ``no such table:
       object_counts``. Emitting the timeID in both branches (the
       non-timelapse spelling is already exactly what ``_generate_time_lists``
       parses) is the fix.
    """
    dst = Path(dst).absolute()
    dst.mkdir(parents=True, exist_ok=True)
    channels = [int(c) for c in channels]
    files, produced = _emit_images(
        dst, plate, wells, fields, channels, times=times,
    )
    settings = demo_settings("timelapse", str(dst), channels=channels)
    settings["timelapse"] = True
    # [start, end] is a *slice* of frame indices — spacr.object does
    # `stack[limits[0]:limits[1]]`. [1, times] therefore silently threw away
    # the first frame of every field; [0, times] keeps all of them.
    settings["timelapse_frame_limits"] = [0, times]
    layout = DemoLayout(
        src=dst, image_dir=dst,
        image_files=files,
        notes={"times": times, "channels": list(channels), "plate": plate},
    )
    layout.settings_csv = save_settings_csv(
        dst / "settings_timelapse.csv", settings,
    )
    return layout


# ---------------------------------------------------------------------------
# Settings — reverse-engineered per app so the demo actually runs
# ---------------------------------------------------------------------------

def _channel_settings(channels: Sequence[int]) -> Dict[str, Any]:
    """`*_channel` for the roles this dataset acquired, None for the rest.

    Naming a channel the plate does not have is a hard pre-flight error
    (``organelle_channel=3 but the dataset has only 2 channels``), so the
    roles have to follow the channel list rather than being a constant.
    """
    wanted = {int(c) for c in channels}
    return {key: (index if index in wanted else None)
            for key, index in CHANNEL_LAYOUT.items()}


def _mask_dim_settings(channels: Sequence[int]) -> Dict[str, Any]:
    """`*_mask_dim` — where each label plane lands in merged/*.npy.

    The image planes come first (one per entry of ``channels``), then one
    plane per acquired object in :data:`MASK_ROLE_ORDER`.
    """
    roles = _mask_roles(channels)
    dims: Dict[str, Any] = {f"{role}_mask_dim": None for role in MASK_ROLE_ORDER}
    for offset, role in enumerate(roles):
        dims[f"{role}_mask_dim"] = len(channels) + offset
    return dims


def demo_settings(app_key: str, src: str,
                  channels: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Return a spacr settings dict tailored for the demo dataset
    generated by `generate_<app>_demo`.

    :param app_key: which app the settings are for.
    :param src: the demo folder (for ``map_barcodes``, the demo root — its
        barcode CSVs are resolved relative to it).
    :param channels: the acquisition channels the dataset actually holds.
        Defaults to all four. Every ``*_channel`` and ``*_mask_dim`` key is
        derived from it, so a two-channel demo cannot advertise a pathogen
        channel it never acquired.

    Values are the minimum needed to make the pipeline flow — real
    users will tweak thresholds + channel numbers to fit their data.
    """
    channels = [0, 1, 2, 3] if channels is None else [int(c) for c in channels]
    base: Dict[str, Any] = {
        "src": src,
        "channels": list(channels),
        "plot": False,
        "test_mode": False,
    }
    # Only the apps that ingest raw acquisition files parse filenames or care
    # about the objective. Measure reads merged/*.npy, whose field names and
    # plane layout are already fixed, so shipping these to it would be more
    # keys the Measure screen has no widget for and measure_crop never reads —
    # accepted, dropped, and impossible to notice.
    acquisition: Dict[str, Any] = {
        "metadata_type": "cellvoyager",
        "custom_regex": None,
        "magnification": 20,
    }
    layout = _channel_settings(channels)
    if app_key == "mask":
        return {
            **base,
            **acquisition,
            **layout,
            # The demo draws cells at _RADIUS_CELL and nuclei at
            # _RADIUS_NUCLEUS; Cellpose 4 rescales by 30/diameter, so telling
            # it the truth is what puts the objects near the size cpsam was
            # trained on.
            "cell_diameter": _RADIUS_CELL * 2,
            "nucleus_diameter": _RADIUS_NUCLEUS * 2,
            "pathogen_diameter": _RADIUS_PATHOGEN * 2,
            # The camera offset the images are actually drawn on. All three
            # object channels, not just the cell: `*_background` is multiplied
            # by `*_Signal_to_noise` to set the normalisation ceiling, and a
            # demo that declares the right offset for one channel and the
            # 100 default for the other two normalises them differently for
            # no reason.
            "cell_background": _BACKGROUND,
            "nucleus_background": _BACKGROUND,
            "pathogen_background": _BACKGROUND,
            # Real key is capital-S `cell_Signal_to_noise`. The demo shipped
            # `cell_signal_to_noise` for a year: not a spaCR setting, so it
            # was accepted, ignored, and the default used instead.
            "cell_Signal_to_noise": 10,
            "nucleus_Signal_to_noise": 10,
            "pathogen_Signal_to_noise": 10,
            "cell_CP_prob": 0.0,
            "cell_FT": 1.0,
            # 'cyto' / 'nuclei' until Cellpose 4 removed them. The demo
            # settings are what a new user copies, so they name the model
            # that exists; a legacy value in a real settings file is still
            # accepted and mapped forward by
            # settings.normalize_cellpose_model_name.
            "cell_model_name": "cpsam",
            "nucleus_model_name": "cpsam",
        }
    if app_key in ("measure", "crop"):
        crop = app_key == "crop"
        # No `*_channel` here: measure_crop indexes merged/*.npy with
        # `*_mask_dim` and never reads the raw acquisition channel keys, and
        # the Qt Measure screen has no widget for them either.
        return {
            **base,
            **_mask_dim_settings(channels),
            "cell_min_size": 50,
            "nucleus_min_size": 25,
            "pathogen_min_size": 15,
            "save_measurements": True,
            "timelapse": False,
            "experiment": "demo",
            "crop_mode": ["cell"],
            "save_png": crop,
            # png_size is a [height, width] pair, not a scalar — a bare int
            # is a hard pre-flight error ("png_size=64 is a int, but list is
            # expected").
            "png_size": [64, 64],
            "png_dims": [0, 1, 2],
            # No `normalize` / `normalize_by` here, deliberately. measure_crop
            # reads `normalize` as a [low, high] percentile PAIR, but
            # spacr.settings declares it ``bool`` and the Qt Measure screen
            # therefore renders it as a Toggle: importing a demo that shipped
            # `normalize=[1, 99]` put **False** in the form
            # (`_apply_value` does `str(val).lower() in ("true","1","yes")`),
            # so the CSV on disk and the form the user is looking at disagreed
            # about how every crop is scaled. `normalize_by` alone is inert —
            # measure.py only consults it when `normalize` is a list — so
            # shipping it would be decoration. Omitting both leaves the
            # measure defaults (normalize=False, normalize_by='png'), which is
            # what the run does anyway, and nothing is silently rewritten.
            # Making [1, 99] loadable needs a real widget for the pair in
            # spacr/qt/screens/settings_model.py + a `(bool, list)` type in
            # spacr/settings.py; neither is in this module.
        }
    if app_key == "classify":
        return {
            "src": src,
            # The crops are labelled in png_list, not by well metadata, so the
            # dataset has to be built in 'annotation' mode. The shipped
            # default is 'metadata', which selects on metadata_type_by
            # ('columnID') and would build two classes out of one well.
            "dataset_mode": "annotation",
            # Only the singular key: generate_training_dataset falls back to
            # `[settings['annotation_column']]` when `annotation_columns` is
            # unset, and the plural spelling — which io.py reads — is not
            # declared in spacr.settings, so shipping it makes pre-flight warn
            # "did you mean 'annotation_column'?" on every demo load.
            "annotation_column": "annotate",
            "annotated_classes": [1, 2],
            "png_type": "cell_png",
            "file_type": "cell_png",
            "image_size": 64,
            "batch_size": 8,
            "epochs": 2,
            "test_split": 0.25,
            # 'cnn' is not a model: model_type is fed to torchvision, and the
            # GUI offers only names from that list. resnet50 is the smallest
            # of them that trains sensibly on 64 px crops.
            "model_type": "resnet50",
            # `train_channels`, not `channels`. Classify runs
            # spacr.deep_spacr.deep_spacr, which selects the crop's colour
            # planes with `settings['train_channels']` (r/g/b letters) and
            # never reads `channels` at all — `channels` is not even in
            # deep_spacr_defaults, so the Classify screen has no widget for it
            # and `_apply_demo_to_screen` dropped it on the floor. The demo's
            # crops are a greyscale plane replicated into RGB, so all three
            # planes carry signal.
            "train_channels": ["r", "g", "b"],
            # No `channel_of_interest`: it is a spacr.ml recruitment/regression
            # setting (it picks `pathogen_channel_<n>_mean_intensity` columns),
            # deep_spacr never reads it, and the Classify screen renders it as
            # a QSpinBox — so the `None` this used to ship came back from the
            # form as 3.
        }
    if app_key == "timelapse":
        return {
            **base,
            **acquisition,
            **layout,
            "timelapse": True,
            # A slice, not an inclusive 1-based range: see
            # generate_timelapse_demo, which overwrites this with the real
            # frame count.
            "timelapse_frame_limits": [0, 8],
            "timelapse_objects": ["cell"],
            # 'trackastra' is the shipped default and the better tracker, but
            # it is an optional dependency: a machine without it cannot run
            # the demo at all. 'iou' ships with spaCR, needs no tuning, and
            # is exactly right for objects that drift a couple of pixels per
            # frame — which is what this dataset is.
            "timelapse_mode": "iou",
            "cell_diameter": _RADIUS_CELL * 2,
            "nucleus_diameter": _RADIUS_NUCLEUS * 2,
        }
    if app_key == "map_barcodes":
        barcodes = os.path.join(src, BARCODE_DIRNAME)
        return {
            "src": src,
            # The three CSVs spacr.sequencing.map_sequences_to_names reads;
            # each needs 'name' and 'sequence' columns. Leaving them unset is
            # a hard pre-flight error, and the demo used to ship
            # `barcode_length` / `barcode_offset` / `processes` — none of
            # which is a spaCR setting.
            "grna_csv": os.path.join(barcodes, "grna.csv"),
            "row_csv": os.path.join(barcodes, "row.csv"),
            "column_csv": os.path.join(barcodes, "column.csv"),
            "mode": "paired",
            "single_direction": "R1",
            "target_sequence": SEQ_TARGET,
            "offset_start": SEQ_OFFSET_START,
            "expected_end": SEQ_WINDOW_LENGTH,
            "chunk_size": 1000,
            "n_jobs": 2,
            "save_h5": False,
            "test": False,
            "fill_na": False,
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

#: Sample name of the demo FASTQ pair. spacr.io.parse_gz_files groups files by
#: ``filename.split('_')`` and reads ``parts[1]`` as the read direction, so
#: the name has to be ``<sample>_R1_001.fastq.gz`` — Illumina's own
#: convention. The demo used to write ``synthetic_R1.fastq.gz``, whose
#: ``parts[1]`` is ``'R1.fastq.gz'``; parse_gz_files then returned
#: ``{'synthetic': {}}`` and generate_barecode_mapping died on ``KeyError:
#: 'R1'`` having written nothing.
FASTQ_SAMPLE = "demo"

#: Where the three barcode CSVs live inside a map_barcodes demo folder. They
#: are in a subfolder because ``src`` itself is listed flat for ``*.fastq.gz``
#: (spacr.io.parse_gz_files) — anything else in it must not look like a read
#: file, and a folder never does.
BARCODE_DIRNAME = "barcodes"

# --- the read frame the shipped barcode-mapping defaults expect ------------
#
# spacr.settings.set_default_generate_barecode_mapping anchors on
# `target_sequence`, slices `expected_end` bases starting `offset_start` from
# the anchor, and splits that window with DEFAULT_BARCODE_REGEX:
#
#   ^(?P<columnID>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<rowID>.{8}).*
#
# So the 89-base window has to be laid out exactly like this, and it is:
#
#   [ 0: 8]  column barcode        8
#   [ 8:34]  SEQ_TARGET           26   <- the anchor; supplies TGCTG…TAAAC
#   [34:55]  gRNA barcode         21
#   [55:60]  SEQ_GRNA_SUFFIX       5   <- the AACTT the regex demands
#   [60:68]  SEQ_FILL              8
#   [68:73]  SEQ_ROW_PREFIX        5   <- the AGAAG the regex demands
#   [73:81]  row barcode           8
#   [81:89]  SEQ_TAIL              8
#
# and the anchor sits `-offset_start` = 8 bases into the window, which is why
# the column barcode is exactly 8 long.
SEQ_TARGET = "TGCTGTTTCCAGCATAGCTCTTAAAC"
SEQ_OFFSET_START = -8
SEQ_WINDOW_LENGTH = 89
SEQ_GRNA_SUFFIX = "AACTT"
SEQ_FILL = "GGCACCGT"
SEQ_ROW_PREFIX = "AGAAG"
SEQ_TAIL = "CCTGATTC"
#: 20 bases of stagger before the anchor window, as real libraries carry, and
#: 41 of read after it: 20 + 89 + 41 = FASTQ_READ_LENGTH. The suffix is the
#: start of the Illumina P7 adapter, which is what a real read runs into once
#: it has passed the insert.
SEQ_READ_PREFIX = "ATTGGCCTTCAGGTACCTGA"
SEQ_READ_SUFFIX = "GATCGGAAGAGCACACGTCTGAACTCCAGTCACGCTTGCGC"

#: gRNA barcodes are 21 bases, as the bundled barcodes_grna.csv is, and row /
#: column barcodes are 8, as barcodes_row.csv and barcodes_column.csv are.
GRNA_LENGTH = 21
WELL_BARCODE_LENGTH = 8

#: A barcode containing any of these would give the regex a second place to
#: anchor, so the planted barcode would not be the one recovered. Rejecting
#: them at generation time is what makes "every planted gRNA comes back out"
#: a property the demo can be tested on.
_FORBIDDEN_MOTIFS = ("TGCTG", "TAAAC", "AACTT", "AGAAG")


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


def _random_barcode(rng: np.random.Generator, length: int) -> str:
    """Random A/C/G/T barcode carrying none of the adapter motifs."""
    while True:
        bc = "".join(rng.choice(list("ACGT"), size=length))
        if not any(motif in bc for motif in _FORBIDDEN_MOTIFS):
            return bc


def barcode_pool(n: int, length: int, seed: int = 0) -> List[str]:
    """Return ``n`` distinct synthetic barcodes of ``length`` bases.

    :param n: how many to draw.
    :param length: barcode length in bases.
    :param seed: RNG seed — the same seed always returns the same pool.
    :returns: list of ``n`` unique uppercase DNA strings.
    """
    rng = np.random.default_rng(seed)
    pool: List[str] = []
    seen = set()
    while len(pool) < n:
        bc = _random_barcode(rng, length)
        if bc in seen:
            continue
        seen.add(bc)
        pool.append(bc)
    return pool


def generate_barcode_csv(dst: Path, names: Sequence[str],
                         sequences: Sequence[str]) -> Path:
    """Write a ``name,sequence`` barcode CSV.

    This is the format spacr.sequencing.map_sequences_to_names reads — it
    requires both columns by name and rejects duplicate sequences. (The demo
    used to ship a FASTA, which that function cannot read at all.)

    :param dst: output ``.csv`` path.
    :param names: barcode names, aligned with ``sequences``.
    :param sequences: barcode sequences.
    :returns: the resolved ``dst`` path.
    :raises ValueError: when the two sequences differ in length.
    """
    if len(names) != len(sequences):
        raise ValueError(
            f"names ({len(names)}) and sequences ({len(sequences)}) "
            "must be the same length.")
    dst = Path(dst).absolute()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "sequence"])
        for name, seq in zip(names, sequences):
            w.writerow([name, seq])
    LOG.info("wrote %d barcodes → %s", len(names), dst)
    return dst


def synthetic_read(column_barcode: str, grna: str, row_barcode: str,
                   prefix: str = SEQ_READ_PREFIX) -> str:
    """Build one 150-base read carrying a (column, gRNA, row) triplet.

    The layout is the one documented above :data:`SEQ_TARGET`; a read built
    here is recovered exactly by the shipped ``regex`` / ``target_sequence``
    / ``offset_start`` / ``expected_end`` defaults.

    :param column_barcode: 8-base column barcode.
    :param grna: 21-base gRNA barcode.
    :param row_barcode: 8-base row barcode.
    :param prefix: stagger placed before the anchor window.
    :returns: a 150-base read.
    :raises ValueError: when a barcode is the wrong length — a silently
        mis-sized barcode would shift every downstream field by that many
        bases and map to nothing, which is far harder to see than a stop.
    """
    if len(column_barcode) != WELL_BARCODE_LENGTH:
        raise ValueError(
            f"column barcode must be {WELL_BARCODE_LENGTH} bases, "
            f"got {len(column_barcode)}: {column_barcode!r}")
    if len(row_barcode) != WELL_BARCODE_LENGTH:
        raise ValueError(
            f"row barcode must be {WELL_BARCODE_LENGTH} bases, "
            f"got {len(row_barcode)}: {row_barcode!r}")
    if len(grna) != GRNA_LENGTH:
        raise ValueError(
            f"gRNA barcode must be {GRNA_LENGTH} bases, "
            f"got {len(grna)}: {grna!r}")
    window = (column_barcode + SEQ_TARGET + grna + SEQ_GRNA_SUFFIX
              + SEQ_FILL + SEQ_ROW_PREFIX + row_barcode + SEQ_TAIL)
    if len(window) != SEQ_WINDOW_LENGTH:
        raise ValueError(
            f"anchor window is {len(window)} bases, expected "
            f"{SEQ_WINDOW_LENGTH}; the adapter constants no longer add up.")
    read = prefix + window + SEQ_READ_SUFFIX
    return read[:FASTQ_READ_LENGTH]


def _reverse_complement(seq: str) -> str:
    """Reverse complement of an A/C/G/T/N sequence."""
    table = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(table[b] for b in reversed(seq))


def _fastq_header(index: int, read: int = 1, tile: int = 1101,
                    y: Optional[int] = None) -> str:
    """Build one @-prefixed FASTQ header matching Illumina 1.8+ format."""
    x = 1000 + (index % 9000)          # 1000..9999
    y = y if y is not None else 1000 + (index // 9000)
    return (
        f"@{FASTQ_INSTRUMENT}:{FASTQ_RUN}:{FASTQ_FLOWCELL}"
        f":{FASTQ_LANE}:{tile}:{x}:{y} {read}:N:0:{FASTQ_I7_INDEX}"
    )


def generate_synthetic_fastq(
    dst_dir: Path,
    grnas: Sequence[str],
    rows: Sequence[str],
    columns: Sequence[str],
    n_reads: int = 5_000,
    seed: int = 0,
    sample: str = FASTQ_SAMPLE,
    paired: bool = True,
) -> List[Path]:
    """Write a gzip-compressed synthetic FASTQ pair carrying known barcodes.

    Every read is one (column, gRNA, row) triplet in the frame the shipped
    barcode-mapping defaults parse, so ``unique_combinations.csv`` comes back
    with the planted wells and guides in it. Reads are spread evenly over the
    ``rows x columns`` wells, and within a well over the gRNAs with a skew,
    because a real screen has a handful of abundant guides and a long tail.

    :param dst_dir: folder to write into.
    :param grnas: gRNA barcode sequences (21 bases each).
    :param rows: row barcode sequences (8 bases each).
    :param columns: column barcode sequences (8 bases each).
    :param n_reads: approximate total number of reads; the real total is
        rounded down to a whole number of reads per well.
    :param seed: RNG seed for reproducible read pools.
    :param sample: sample name; the files are
        ``<sample>_R1_001.fastq.gz`` (and ``_R2_``).
    :param paired: also write the R2 mate. R2 is the exact reverse
        complement of R1 — a perfectly overlapping pair — which is what
        spacr.sequencing's paired path reduces to after it
        reverse-complements R2 and takes the per-base consensus.
    :returns: the written paths, R1 first.
    :raises ValueError: when any barcode list is empty.
    """
    import gzip

    if not grnas or not rows or not columns:
        raise ValueError(
            "grnas, rows and columns must all be non-empty "
            f"(got {len(grnas)}, {len(rows)}, {len(columns)}).")

    dst_dir = Path(dst_dir).absolute()
    dst_dir.mkdir(parents=True, exist_ok=True)
    r1_path = dst_dir / f"{sample}_R1_001.fastq.gz"
    r2_path = dst_dir / f"{sample}_R2_001.fastq.gz"

    rng = np.random.default_rng(seed)
    wells = [(r, c) for r in rows for c in columns]
    per_well = max(1, n_reads // len(wells))

    r1 = gzip.open(r1_path, "wt")
    r2 = gzip.open(r2_path, "wt") if paired else None
    try:
        index = 0
        for row_bc, col_bc in wells:
            weights = rng.random(len(grnas)) + 0.08
            weights = weights / weights.sum()
            picks = rng.choice(len(grnas), size=per_well, p=weights)
            for pick in picks:
                seq = synthetic_read(col_bc, grnas[int(pick)], row_bc)
                qual = _phred_run(len(seq), mean_q=32, seed=index)
                r1.write(f"{_fastq_header(index, 1)}\n{seq}\n+\n{qual}\n")
                if r2 is not None:
                    r2.write(
                        f"{_fastq_header(index, 2)}\n"
                        f"{_reverse_complement(seq)}\n+\n{qual[::-1]}\n")
                index += 1
    finally:
        r1.close()
        if r2 is not None:
            r2.close()

    LOG.info("emitted %d reads across %d wells x %d gRNAs → %s",
             index, len(wells), len(grnas), dst_dir)
    return [r1_path, r2_path] if paired else [r1_path]


def generate_map_barcodes_demo(
    dst: Path,
    n_barcodes: int = 12,
    n_reads: int = 5_000,
    seed: int = 0,
    n_rows: int = 4,
    n_columns: int = 6,
) -> DemoLayout:
    """Populate ``dst`` with a self-contained map_barcodes demo:

    ::

        dst/
          barcodes/
            grna.csv            # ← N gRNA barcodes, name,sequence
            row.csv             # ← row (plate-row) barcodes
            column.csv          # ← column barcodes
          demo_R1_001.fastq.gz  # ← reads carrying those barcodes
          demo_R2_001.fastq.gz
          settings_map_barcodes.csv

    The FASTQs sit in ``dst`` itself, not a ``fastq/`` subfolder: the
    pipeline's ``src`` is listed *flat* for ``*.fastq.gz``
    (spacr.io.parse_gz_files), so a subfolder means zero samples found and a
    run that exits having written nothing.

    :param dst: destination folder.
    :param n_barcodes: number of unique gRNA barcodes to plant.
    :param n_reads: approximate total number of reads to emit.
    :param seed: RNG seed for reproducibility.
    :param n_rows: number of plate-row barcodes.
    :param n_columns: number of plate-column barcodes.
    :returns: :class:`DemoLayout` describing the emitted files.
    """
    dst = Path(dst).absolute()
    barcode_dir = dst / BARCODE_DIRNAME
    barcode_dir.mkdir(parents=True, exist_ok=True)

    grnas = barcode_pool(n_barcodes, GRNA_LENGTH, seed=seed)
    rows = barcode_pool(n_rows, WELL_BARCODE_LENGTH, seed=seed + 101)
    columns = barcode_pool(n_columns, WELL_BARCODE_LENGTH, seed=seed + 202)

    grna_csv = generate_barcode_csv(
        barcode_dir / "grna.csv",
        [f"gRNA_{i + 1:04d}" for i in range(len(grnas))], grnas)
    row_csv = generate_barcode_csv(
        barcode_dir / "row.csv",
        [f"r{i + 1}" for i in range(len(rows))], rows)
    column_csv = generate_barcode_csv(
        barcode_dir / "column.csv",
        [f"c{i + 1}" for i in range(len(columns))], columns)

    fastqs = generate_synthetic_fastq(
        dst, grnas=grnas, rows=rows, columns=columns,
        n_reads=n_reads, seed=seed,
    )
    settings = demo_settings("map_barcodes", str(dst))
    csv_path = save_settings_csv(dst / "settings_map_barcodes.csv", settings)
    return DemoLayout(
        src=dst, image_dir=dst,
        image_files=list(fastqs), settings_csv=csv_path,
        notes={"n_reads": n_reads, "n_barcodes": n_barcodes,
               "n_wells": n_rows * n_columns,
               "grna_csv": str(grna_csv), "row_csv": str(row_csv),
               "column_csv": str(column_csv)},
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
