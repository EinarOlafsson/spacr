#!/usr/bin/env python
"""Build the Replication and Recruitment example datasets (item 463).

Both assay modules read ``<src>/measurements/measurements.db`` -- the output
of Mask then Measure -- so each example is a slice of a real screen's
database: every row of a handful of control wells, plus two merged fields
from those wells so the objects can be looked at (Recruitment also draws its
outline overlays from them).

The sources are local copies of the NAS databases; this script never touches
``/nas_mnt``. Copy the database and the two fields first, through
``tools/nas_guard.sh``:

    Replication  /nas_mnt/data/MTOC_Screen/mtocScreen_20250530_111302/plate1
    Recruitment  /nas_mnt/data/THP1_screen/THP1RNF213_20250218_153752/plate1

Usage::

    python tools/build_assay_example_datasets.py replication \
        --source /mnt/wd4tb/spacr_testdata/assays/src_mtoc_plate1 \
        --out /mnt/wd4tb/spacr_testdata/assays/replication

The output folder holds the unpacked dataset, ``<name>.tar`` (uncompressed,
members relative to the dataset root, settings last) and ``README.md``, the
dataset card.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sqlite3
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PLACEHOLDER = "<dataset>"


@dataclass
class Spec:
    """What one example set is cut from, and the settings it ships."""

    key: str
    repo: str
    archive: str
    screen: str
    nas_plate: str
    tables: Tuple[str, ...]
    rows: Tuple[int, ...]
    columns: Tuple[int, ...]
    fields: Tuple[str, ...]
    settings: Dict[str, object]
    card_intro: str
    conditions: str
    extra_card: str = ""


REPLICATION = Spec(
    key="replication",
    repo="einarolafsson/spacr-example-replication",
    archive="spacr-example-replication.tar",
    screen="Toxoplasma MTOC screen, plate 1 (acquired 2025-05-30, CellVoyager, 40x water)",
    nas_plate="/nas_mnt/data/MTOC_Screen/mtocScreen_20250530_111302/plate1",
    tables=("pathogen", "cell"),
    rows=(3, 4, 5, 6, 7, 8),
    columns=(1, 2),
    fields=("plate1_H01_17_1", "plate1_G02_5_1"),
    settings={
        "src": PLACEHOLDER,
        "parasite_table": "pathogen",
        "compartment": "pathogen",
        "vacuole_key": "spatial",
        "cell_types": ["HeLa"],
        "cell_plate_metadata": None,
        "pathogen_types": ["nc", "pc"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "min_parasite_area": 400,
        "max_parasites_per_vacuole": 16,
        "require_host_cell": True,
        "save": True,
    },
    card_intro=(
        "Parasites per vacuole, for spaCR's **Replication Assay** module. "
        "Every parasite object of twelve control wells of the Toxoplasma "
        "MTOC screen, as the screen's own Mask and Measure runs segmented "
        "and measured them."),
    conditions=(
        "`nc` = column 1, `pc` = column 2: the screen's negative and "
        "positive control columns, labelled as the screen's own analysis "
        "settings label them. They are the controls of an MTOC-recruitment "
        "screen, not of a replication experiment, so no replication "
        "difference between them is claimed."),
    extra_card=(
        "The pathogen mask was segmented from the parasite stain (plane 2) "
        "and splits most vacuoles into their parasites, not always cleanly: "
        "the module groups parasites into vacuoles by clustering their "
        "centroids inside each host cell (`vacuole_key = spatial`), and "
        "its `non_power_of_two` bucket reports how often the result is off "
        "the 1-2-4-8-16 ladder. Read that bucket before reading the "
        "distribution. `min_parasite_area` is 400 px, the "
        "`pathogen_min_size` the screen's Measure run used; the table "
        "still holds the smaller objects, so lowering it is a real choice."),
)


RECRUITMENT = Spec(
    key="recruitment",
    repo="einarolafsson/spacr-example-recruitment",
    archive="spacr-example-recruitment.tar",
    screen="THP-1 RNF213 screen, plate 1 (acquired 2025-02-18, CellVoyager, 40x water)",
    nas_plate="/nas_mnt/data/THP1_screen/THP1RNF213_20250218_153752/plate1",
    tables=("cell", "nucleus", "pathogen", "cytoplasm"),
    rows=(3, 4, 5, 6, 7, 8),
    columns=(1, 2),
    fields=("PLATE1_E01_1_1", "PLATE1_E02_1_1"),
    settings={
        "src": PLACEHOLDER,
        "target": "protein",
        "cell_types": ["THP1"],
        "cell_plate_metadata": None,
        "pathogen_types": ["nc", "pc"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "channel_dims": [0, 1, 2, 3],
        "cell_chann_dim": 3,
        "nucleus_chann_dim": 0,
        "pathogen_chann_dim": 2,
        "channel_of_interest": 1,
        "plot": True,
        "plot_nr": 1,
        "plot_control": True,
        "figuresize": 10,
        "pathogen_limit": 10,
        "nuclei_limit": 1,
        "cells_per_well": 0,
        "pathogen_size_range": [0, 100000],
        "nucleus_size_range": [0, 100000],
        "cell_size_range": [0, 100000],
        "pathogen_intensity_range": [0, 100000],
        "nucleus_intensity_range": [0, 100000],
        "target_intensity_min": 1,
    },
    card_intro=(
        "Recruitment of a host protein to the Toxoplasma vacuole, for "
        "spaCR's **Recruitment** module. Every cell, nucleus, pathogen and "
        "cytoplasm object of twelve control wells of the THP-1 RNF213 "
        "screen, as the screen's own Mask and Measure runs segmented and "
        "measured them."),
    conditions=(
        "`nc` = column 1, `pc` = column 2: the screen's negative and "
        "positive control columns, as the screen's own recruitment settings "
        "(`settings/recruitment.csv` on the plate) label them."),
    extra_card=(
        "The settings are the ones the screen's recruitment analysis was run "
        "with (`recruitment.csv` in the plate's `settings/` folder), with "
        "three changes: `src` points at the download, `cell_types` is `THP1`, "
        "the cells the screen is named for (the recorded value was `Hela`), "
        "and `plot_nr` is 1 because two fields ship. The recruitment score is the pathogen-to-cytoplasm "
        "ratio of mean intensity in channel 1."),
)

SPECS = {s.key: s for s in (REPLICATION, RECRUITMENT)}

PLANES = ("0 nucleus stain (the screen's nucleus_channel)",
          "1 the screen's measured target (its channel_of_interest)",
          "2 parasite stain (the screen's pathogen_channel)",
          "3 cell stain (the screen's cell_channel)",
          "4 cell mask", "5 nucleus mask", "6 pathogen mask")


def _wells(spec: Spec) -> List[Tuple[str, str]]:
    """Every (rowID, columnID) pair the slice keeps."""
    return [(f"r{r}", f"c{c}") for c in spec.columns for r in spec.rows]


def subset_database(source: Path, target: Path, spec: Spec) -> Dict[str, int]:
    """Copy every row of the chosen wells, table by table.

    Paths recorded by the screen name the NAS; they are replaced by
    ``merged/<file_name>.npy`` so nothing in the published copy names a
    machine.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    counts: Dict[str, int] = {}
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    wells = _wells(spec)
    clause = " or ".join("(rowID = ? and columnID = ?)" for _ in wells)
    params = [value for pair in wells for value in pair]
    for table in spec.tables:
        (create,) = src.execute(
            "select sql from sqlite_master where type='table' and name=?",
            (table,)).fetchone()
        dst.execute(create)
        columns = [r[1] for r in src.execute(f'pragma table_info("{table}")')]
        rows = src.execute(
            f'select * from "{table}" where {clause}', params).fetchall()
        marks = ",".join("?" for _ in columns)
        dst.executemany(f'insert into "{table}" values ({marks})', rows)
        if "path_name" in columns and "file_name" in columns:
            dst.execute(
                f'update "{table}" set path_name = '
                f"'merged/' || file_name || '.npy'")
        counts[table] = len(rows)
    dst.commit()
    dst.execute("vacuum")
    dst.close()
    src.close()
    return counts


def _cell(value) -> str:
    """One settings value as spaCR's settings CSV writes it."""
    if value is None:
        return ""
    return str(value)


def write_settings(folder: Path, spec: Spec) -> Path:
    """``settings/<key>_settings.csv``, the name the settings pack reads."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{spec.key}_settings.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        for key, value in spec.settings.items():
            writer.writerow([key, _cell(value)])
    return path


def _anonymous(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Drop the builder's account name from each tar member."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(spec: Spec, source: Path, out: Path) -> Path:
    """Write the dataset folder, its tar and its card under ``out``."""
    root = out / "dataset"
    if root.exists():
        shutil.rmtree(root)
    counts = subset_database(source / "measurements.db",
                             root / "measurements" / "measurements.db", spec)
    wanted = {f"{r}_{c}" for r, c in _wells(spec)}
    merged = root / "merged"
    merged.mkdir(parents=True)
    for name in spec.fields:
        field_file = source / "merged" / f"{name}.npy"
        if not field_file.is_file():
            raise SystemExit(f"missing field {field_file}; copy it from "
                             f"{spec.nas_plate}/merged first")
        shutil.copy2(field_file, merged / field_file.name)
    write_settings(root / "settings", spec)

    archive = out / spec.archive
    members = (sorted(merged.glob("*.npy"))
               + [root / "measurements" / "measurements.db"]
               + sorted((root / "settings").glob("*.csv")))
    with tarfile.open(archive, "w") as tar:
        for member in members:
            tar.add(member, arcname=str(member.relative_to(root)),
                    filter=_anonymous)
    size = archive.stat().st_size
    card = out / "README.md"
    card.write_text(_card(spec, counts, size, _sha256(archive)),
                    encoding="utf-8")
    print(f"{spec.key}: {archive} {size / 1e6:.1f} MB; rows {counts}; "
          f"wells {sorted(wanted)}")
    return archive


def _card(spec: Spec, counts: Dict[str, int], size: int, digest: str) -> str:
    rows = ", ".join(f"r{r}" for r in spec.rows)
    tables = "\n".join(f"| `{t}` | {n:,} |" for t, n in counts.items())
    fields = "\n".join(f"- `merged/{f}.npy`" for f in spec.fields)
    planes = "\n".join(f"  - {p}" for p in PLANES)
    return f"""---
license: mit
pretty_name: spaCR example -- {spec.key}
tags:
- microscopy
- toxoplasma
- spacr
size_categories:
- n<1K
---

# spaCR example data: {spec.key}

{spec.card_intro}

Download it from inside spaCR with **Load test data...** in the
{spec.key.capitalize()} module, or with `spacr-download {spec.key}`. The
settings file ships beside the data, so after loading, Run is the next step.

## Size

`{spec.archive}`: {size:,} bytes ({size / 1e6:.0f} MB), an uncompressed tar.
SHA-256 `{digest}`.

## Provenance

- Screen: {spec.screen}.
- Plate folder on the lab NAS: `{spec.nas_plate}`.
- Wells: rows {rows} of columns {", ".join(f"c{c}" for c in spec.columns)}
  (twelve wells), every field of each.
- {spec.conditions}

## Contents

| table in `measurements/measurements.db` | rows |
|---|---|
{tables}

Every row of the chosen wells is kept, unfiltered; nothing is resampled.
Recorded paths that named the NAS were replaced by `merged/<file_name>.npy`.

Two merged fields, one per control column, so the objects can be inspected:

{fields}

Each is a 7-plane uint16 stack (height x width x 7):

{planes}

`settings/{spec.key}_settings.csv` holds the module settings; its `src` is
filled in with the download location when spaCR unpacks it.

## Notes

{spec.extra_card}
"""


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("key", choices=sorted(SPECS))
    parser.add_argument("--source", required=True, type=Path,
                        help="local folder with measurements.db and merged/")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if str(args.source).startswith("/nas_mnt"):
        parser.error("copy the source off the NAS first; see the docstring")
    args.out.mkdir(parents=True, exist_ok=True)
    build(SPECS[args.key], args.source, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
