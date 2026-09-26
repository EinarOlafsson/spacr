#!/usr/bin/env python
"""Build the Invasion Assay example dataset (item 463): SYNTHETIC, by decision.

No two-colour differential-staining acquisition exists on the NAS or on local
disks (the 463 item file lists where it was looked for). The maintainer chose
"Synthetic, clearly labelled": the same kind of synthetic two-colour data the
module was developed and taught on, but as IMAGES that spaCR's own Measure
turns into a real ``measurements.db``.

THE MASKS ARE THE DRAWN OBJECTS BY DEFAULT, NOT A CELLPOSE SEGMENTATION.
``--segment`` runs spaCR's Mask (Cellpose-SAM) over the images instead, and
that is the better example -- but on CPU it did not finish one plate row in
40 minutes (2026-09-21, machine load 36 on 32 cores): the 10-px parasites
make Cellpose upscale the pathogen channel threefold. The GPU was shared and
not to be used. So the published set carries the generator's own label discs
in exactly the planes Mask would have written, and Measure -- the step that
makes the database -- is spaCR's, unchanged. Rebuild with ``--segment`` on a
free GPU to replace them.

THE DESIGN OF THE TEACHING SET, KEPT. ``tools/tutorials/authoring/tools/
prepare_invasion_tutorial_data.py`` wrote the tutorial's rows directly: column
c1 a no-primary staining control whose parasites carry no outside stain, and
two conditions whose parasites are a known mixture of invaded (outside stain
absent) and attached (outside stain present). This renders that design as
pixels instead of rows, with the fractions made biological -- the inhibitor
lowers invasion rather than raising it:

    c1  staining control  no outside stain on any parasite
    c2  vehicle           70% invaded
    c3  inhibitor         35% invaded

CHANNELS (cellvoyager ``C00``..``C03``):

    0  nucleus stain            segmented as nuclei
    1  cell stain               segmented as cells
    2  total parasite stain     post-permeabilisation; segmented as pathogens
    3  outside parasite stain   pre-permeabilisation; attached parasites only

So the Invasion settings are ``total_channel = 2`` and ``outside_channel = 3``.

The field lattice, object shapes, noise model and file naming are
:mod:`spacr.qt.synthetic`'s, reused rather than copied. Every draw is seeded
from the well and field, so the output is byte-identical on any machine.

Usage (about a minute on CPU; add ``--segment`` to run Mask instead)::

    python tools/build_invasion_example_dataset.py \
        --out /mnt/wd4tb/spacr_testdata/assays/invasion
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = "einarolafsson/spacr-example-invasion"
ARCHIVE = "spacr-example-invasion.tar"
PLACEHOLDER = "<dataset>"
PLATE = "plate1"
ROWS = ("A", "B", "C", "D")
CONDITIONS = (("01", "staining control", None),
              ("02", "vehicle", 0.70),
              ("03", "inhibitor", 0.35))
FIELDS = 2
SHAPE = (512, 512)
GRID = 8
INFECTED_FRACTION = 0.75
PARASITES_PER_CELL = (1, 2)
OUTSIDE_PEAK = 30000
OUTSIDE_DIM_FRACTION = 0.1

INVASION_SETTINGS: Dict[str, object] = {
    "src": PLACEHOLDER,
    "parasite_table": "pathogen",
    "compartment": "pathogen",
    "outside_channel": 3,
    "total_channel": 2,
    "intensity_statistic": "auto",
    "background_correction": "none",
    "outside_threshold_method": "otsu",
    "outside_threshold": None,
    "stain_baseline_wells": ["c1"],
    "control_quantile": 0.99,
    "min_control_objects": 10,
    "min_parasites_per_well": 50,
    "extracellular_class": "attached",
    "cell_types": ["HeLa"],
    "cell_plate_metadata": None,
    "pathogen_types": ["vehicle", "inhibitor"],
    "pathogen_plate_metadata": [["c2"], ["c3"]],
    "treatments": None,
    "treatment_plate_metadata": None,
    "save": True,
}


def _synthetic():
    """spaCR's synthetic-image module, from the checkout this runs in."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from spacr.qt import synthetic
    return synthetic


def render_field(well: str, field: int, invaded_fraction) -> Tuple[
        Dict[int, np.ndarray], List[dict], Dict[str, np.ndarray]]:
    """Four channels of one field, the parasite truth, and the drawn masks."""
    syn = _synthetic()
    rng = np.random.default_rng(syn._stable_seed("invasion", PLATE, well, field))
    h, w = SHAPE
    step_y, step_x = h / GRID, w / GRID
    margin = syn._RADIUS_CELL + 2
    cells = []
    for gy in range(GRID):
        for gx in range(GRID):
            cy = (gy + 0.5) * step_y + rng.uniform(-syn.CELL_JITTER, syn.CELL_JITTER)
            cx = (gx + 0.5) * step_x + rng.uniform(-syn.CELL_JITTER, syn.CELL_JITTER)
            cells.append((float(np.clip(cy, margin, h - margin)),
                          float(np.clip(cx, margin, w - margin)),
                          0.82 + 0.36 * rng.random()))

    def peak():
        return syn._PEAK * (0.75 + 0.45 * rng.random())

    cell_spots = [(cy, cx, syn._SIGMA_CELL * s, peak()) for cy, cx, s in cells]
    nucleus_spots = [(cy, cx, syn._SIGMA_NUCLEUS * s, peak())
                     for cy, cx, s in cells]
    masks = {role: np.zeros(SHAPE, dtype=np.uint16)
             for role in ("cell", "nucleus", "pathogen")}
    for label, (cy, cx, s) in enumerate(cells, start=1):
        syn._paint_disc(masks["cell"], cy, cx, syn._RADIUS_CELL * s, label)
        syn._paint_disc(masks["nucleus"], cy, cx, syn._RADIUS_NUCLEUS * s,
                        label)
    parasites = []
    infected = set(rng.permutation(len(cells))[
        :int(round(INFECTED_FRACTION * len(cells)))].tolist())
    for index, (cy, cx, s) in enumerate(cells):
        if index not in infected:
            continue
        n = int(rng.integers(PARASITES_PER_CELL[0], PARASITES_PER_CELL[1] + 1))
        base = rng.uniform(0, 2 * np.pi)
        for k in range(n):
            angle = base + 2 * np.pi * k / n
            py = cy + syn._OFFSET_PATHOGEN * s * np.sin(angle)
            px = cx + syn._OFFSET_PATHOGEN * s * np.cos(angle)
            parasites.append((py, px, syn._SIGMA_PATHOGEN * s, index + 1))
            syn._paint_disc(masks["pathogen"], py, px,
                            syn._RADIUS_PATHOGEN * s, len(parasites))

    total_spots, outside_spots, truth = [], [], []
    for py, px, sigma, host in parasites:
        total_spots.append((py, px, sigma, peak()))
        if invaded_fraction is None:
            invaded = None
        else:
            invaded = bool(rng.random() < invaded_fraction)
        if invaded is False:
            level = OUTSIDE_PEAK * (0.6 + 0.8 * rng.random())
            if rng.random() < OUTSIDE_DIM_FRACTION:
                level *= 0.25
            outside_spots.append((py, px, sigma, level))
        truth.append({"well": well, "field": field,
                      "pathogen_label": len(truth) + 1, "y": round(py, 2),
                      "x": round(px, 2), "host_cell": host,
                      "state": ("no outside stain (staining control)"
                                if invaded is None else
                                "invaded" if invaded else "attached")})
    images = {
        0: syn._draw_spots(SHAPE, nucleus_spots, rng),
        1: syn._draw_spots(SHAPE, cell_spots, rng),
        2: syn._draw_spots(SHAPE, total_spots, rng),
        3: syn._draw_spots(SHAPE, outside_spots, rng),
    }
    return images, truth, masks


def write_raw(folder: Path, rows: Sequence[str] = ROWS) -> List[dict]:
    """Every field of the plate as cellvoyager-named tifs; returns the truth."""
    from tifffile import imwrite

    syn = _synthetic()
    folder.mkdir(parents=True, exist_ok=True)
    truth: List[dict] = []
    for row in rows:
        for column, _condition, fraction in CONDITIONS:
            well = f"{row}{column}"
            for field in range(1, FIELDS + 1):
                images, found, _masks = render_field(well, field, fraction)
                truth.extend(found)
                for channel, image in images.items():
                    imwrite(folder / syn.cellvoyager_filename(
                        plate=PLATE, well=well, field=field, chan=channel),
                        image)
    return truth


def write_merged(folder: Path, rows: Sequence[str] = ROWS) -> List[dict]:
    """The merged stacks Mask would write, from the drawn objects.

    Four image planes, then the cell, nucleus and pathogen label planes --
    the layout ``spacr.io._load_and_concatenate_arrays`` produces -- named
    ``<plate>_<well>_<field>_1.npy`` as a Mask run names them.
    """
    merged = folder / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    truth: List[dict] = []
    for row in rows:
        for column, _condition, fraction in CONDITIONS:
            well = f"{row}{column}"
            for field in range(1, FIELDS + 1):
                images, found, masks = render_field(well, field, fraction)
                truth.extend(found)
                stack = np.stack([images[c] for c in range(4)]
                                 + [masks["cell"], masks["nucleus"],
                                    masks["pathogen"]], axis=-1)
                np.save(merged / f"{PLATE}_{well}_{field}_1.npy",
                        stack.astype(np.uint16))
    return truth


def mask_settings(src: Path) -> dict:
    """Mask on CPU, over the three segmented channels."""
    syn = _synthetic()
    return {
        "src": str(src), "channels": [0, 1, 2, 3],
        "metadata_type": "cellvoyager", "custom_regex": None,
        "magnification": 20, "plot": False, "test_mode": False,
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": 2,
        "organelle_channel": None,
        "cell_diameter": syn._RADIUS_CELL * 2,
        "nucleus_diameter": syn._RADIUS_NUCLEUS * 2,
        "pathogen_diameter": syn._RADIUS_PATHOGEN * 2,
        "cell_background": syn._BACKGROUND,
        "nucleus_background": syn._BACKGROUND,
        "pathogen_background": syn._BACKGROUND,
        "cell_signal_to_noise": 10, "nucleus_signal_to_noise": 10,
        "pathogen_signal_to_noise": 10,
        "cell_model_name": "cpsam", "nucleus_model_name": "cpsam",
        "pathogen_model_name": "cpsam",
        "n_jobs": 8, "batch_size": 8,
    }


def measure_settings(src: Path) -> dict:
    """Measure with the object masks Mask appended after the four images."""
    return {
        "src": str(src / "merged"), "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "organelle_mask_dim": None,
        "cell_min_size": 50, "nucleus_min_size": 25, "pathogen_min_size": 15,
        "save_measurements": True, "save_png": False, "timelapse": False,
        "experiment": "invasion_example", "plot": False, "n_jobs": 8,
    }


def _drop_the_build_paths(database: Path, work: Path) -> None:
    """Make every recorded path relative to the dataset root.

    Measure records where it ran. That folder is this machine's scratch
    space and means nothing to anyone who downloads the set.
    """
    prefix = str(work).rstrip("/") + "/"
    connection = sqlite3.connect(str(database))
    tables = [r[0] for r in connection.execute(
        "select name from sqlite_master where type='table'")]
    for table in tables:
        columns = [(r[1], (r[2] or "").upper()) for r in connection.execute(
            f'pragma table_info("{table}")')]
        names = {name for name, _kind in columns}
        if {"path_name", "file_name"} <= names:
            connection.execute(
                f'update "{table}" set path_name = '
                f"'merged/' || file_name || '.npy'")
        for name, kind in columns:
            if kind in ("TEXT", ""):
                connection.execute(
                    f'update "{table}" set "{name}" = '
                    f'replace("{name}", ?, \'\') where typeof("{name}") = '
                    f"'text'", (prefix,))
                connection.execute(
                    f'update "{table}" set "{name}" = '
                    f'replace("{name}", ?, \'\') where typeof("{name}") = '
                    f"'text'", (str(work),))
    connection.commit()
    connection.execute("vacuum")
    connection.close()


def _cell(value) -> str:
    return "" if value is None else str(value)


def _anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def package(work: Path, out: Path, truth: List[dict],
            segmented: bool = False) -> Path:
    """Copy what the example ships out of the Mask/Measure work folder."""
    root = out / "dataset"
    if root.exists():
        shutil.rmtree(root)
    (root / "measurements").mkdir(parents=True)
    shutil.copy2(work / "measurements" / "measurements.db",
                 root / "measurements" / "measurements.db")
    _drop_the_build_paths(root / "measurements" / "measurements.db", work)
    shutil.copytree(work / "merged", root / "merged")
    with (root / "ground_truth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    (root / "settings").mkdir()
    with (root / "settings" / "invasion_settings.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        for key, value in INVASION_SETTINGS.items():
            writer.writerow([key, _cell(value)])

    archive = out / ARCHIVE
    members = (sorted((root / "merged").glob("*.npy"))
               + [root / "ground_truth.csv",
                  root / "measurements" / "measurements.db",
                  root / "settings" / "invasion_settings.csv"])
    with tarfile.open(archive, "w") as tar:
        for member in members:
            tar.add(member, arcname=str(member.relative_to(root)),
                    filter=_anonymous)
    size = archive.stat().st_size
    counts: Dict[str, int] = {}
    for row in truth:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    (out / "README.md").write_text(
        _card(size, _sha256(archive), counts,
              len(list((root / "merged").glob("*.npy"))), segmented),
        encoding="utf-8")
    print(f"invasion: {archive} {size / 1e6:.1f} MB; truth {counts}")
    return archive


_DRAWN_MASKS = """**The label planes are the objects the generator drew, not a
Cellpose segmentation**: Mask with Cellpose-SAM did not finish one plate row
in 40 minutes on CPU, and the GPU was not available. `--segment` on the
builder runs Mask instead, for a rebuild on a GPU."""

_SEGMENTED_MASKS = """**The label planes are spaCR's own Mask output**
(Cellpose-SAM on the nucleus, cell and total-parasite channels, run on a GPU
with `--segment`), so they carry a real segmentation's misses and merges;
`ground_truth.csv` is still the generator's list of what was drawn."""


def _card(size: int, digest: str, counts: Dict[str, int], fields: int,
          segmented: bool = False) -> str:
    masks = _SEGMENTED_MASKS if segmented else _DRAWN_MASKS
    kind = "segmented by Mask" if segmented else "drawn, see above"
    truth = "\n".join(f"| {state} | {n:,} |" for state, n in sorted(counts.items()))
    return f"""---
license: mit
pretty_name: spaCR example -- invasion (SYNTHETIC)
tags:
- microscopy
- synthetic
- spacr
size_categories:
- n<1K
---

# spaCR example data: invasion -- SYNTHETIC

**These images are synthetic. No cell, parasite or antibody was imaged.**
They exist to show spaCR's **Invasion Assay** module working end to end, and
say nothing about any real invasion experiment, stain or treatment. No real
two-colour differential-staining acquisition was available to publish.

Download it from inside spaCR with **Load test data...** in the Invasion
Assay module, or with `spacr-download invasion`. The settings file ships
beside the data, so after loading, Run is the next step.

## Size

`{ARCHIVE}`: {size:,} bytes ({size / 1e6:.0f} MB), an uncompressed tar.
SHA-256 `{digest}`.

## How it was generated

`tools/build_invasion_example_dataset.py` in the spaCR repository draws the
fields with the lattice, object shapes and noise model of `spacr.qt.synthetic`
(seeded from well and field, so reproducible byte for byte) and writes each
field as the merged stack spaCR's Mask module produces: four image planes,
then cell, nucleus and pathogen label planes. {masks} **spaCR's own Measure
module** then measured every field, so `measurements.db` holds measured
intensities, not the generator's numbers.

The design is the one the Invasion tutorial's synthetic data used: a
staining-control column and two conditions with a known mixture of invaded
and attached parasites.

| column | condition | outside stain |
|---|---|---|
| c1 | staining control | none on any parasite (the no-primary baseline) |
| c2 | vehicle | 70% of parasites invaded (no outside stain) |
| c3 | inhibitor | 35% of parasites invaded |

Rows A-D, {FIELDS} fields per well, {SHAPE[0]}x{SHAPE[1]} px, 64 host cells
per field, 75% of them carrying one or two parasites. One attached parasite
in ten is drawn at a quarter of the outside-stain brightness, so the threshold
has something to get wrong.

## Channels

| channel | stain | used as |
|---|---|---|
| 0 | nucleus | nucleus mask |
| 1 | cell | cell mask |
| 2 | total parasite stain (post-permeabilisation) | pathogen mask; `total_channel` |
| 3 | outside parasite stain (pre-permeabilisation) | `outside_channel` |

## Contents

- `measurements/measurements.db` -- Measure's tables (cell, nucleus,
  pathogen, cytoplasm).
- `merged/` -- the {fields} merged stacks: four image planes, then the cell,
  nucleus and pathogen masks ({kind}).
- `ground_truth.csv` -- every parasite the generator drew, with its position,
  host cell and state. Compare the module's calls against it:

| state | parasites drawn |
|---|---|
{truth}

- `settings/invasion_settings.csv` -- the module settings: c1 as the staining
  baseline, c2 vehicle, c3 inhibitor, channels 3 (outside) and 2 (total).
  `src` is filled in with the download location when spaCR unpacks it.
"""


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rows", default="".join(ROWS),
                        help="plate rows to draw, e.g. ABCD")
    parser.add_argument("--segment", action="store_true",
                        help="run spaCR's Mask (Cellpose-SAM) over the images "
                             "instead of using the drawn objects; slow on CPU")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    work = out / "work"
    if work.exists():
        shutil.rmtree(work)
    if args.segment:
        truth_rows = write_raw(work, rows=tuple(args.rows))
        from spacr.core import preprocess_generate_masks
        preprocess_generate_masks(mask_settings(work))
    else:
        truth_rows = write_merged(work, rows=tuple(args.rows))
    from spacr.measure import measure_crop
    measure_crop(measure_settings(work))
    package(work, out, truth_rows, segmented=args.segment)
    return 0


if __name__ == "__main__":
    sys.exit(main())
