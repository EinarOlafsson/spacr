"""Build, verify and pack Import's test data (ledger item 462).

The same twelve planes -- wells E01 and E02, fields 9 and 10, three channels
of the toxo_mito plate -- are written once per variant in
:data:`spacr.import_examples.IMPORT_VARIANTS`: once per filename convention
Mask's ``metadata_type`` offers, once as Zeiss CZI, once as OME-TIFF and once
as an ImageJ stack. Each variant carries the Mask run's cell / nucleus /
pathogen label images for those fields and the Measure rows of exactly the
cells in them, and a ``manifest.csv`` states what every file truly is.

Usage::

    tools/run_capped.sh 8G python tools/build_import_example.py build \\
        --plate ~/.cache/spacr/example_data/plate1 --out /tmp/import_example \\
        --samples /path/to/public/samples
    tools/run_capped.sh 8G python tools/build_import_example.py verify \\
        --out /tmp/import_example            # also writes README.md
    tools/run_capped.sh 4G python tools/build_import_example.py pack \\
        --out /tmp/import_example --tar /tmp/spacr-example-import.tar

THE WRITER AND THE READER SHARE ONE TABLE, AND THAT LIMITS WHAT THE ROUND
TRIP PROVES. Which numbers a convention counts from zero is read from
``spacr.regex_infer._METADATA_ZERO_BASED`` by both sides, and the names are
built to match the examples the convention table quotes. So a pass proves the
Import module inverts what these conventions DOCUMENT; it cannot prove the
documentation is right about a given instrument. The names were checked
against the table's examples, which were taken from real acquisitions.

:func:`synthetic_fields` builds the same shapes from random pixels, so the
test suite runs the whole round trip without the plate or the network.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import uuid
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spacr import import_examples as ix  # noqa: E402

#: ``channel number -> name``. Names sort in channel order (D < F < T), which
#: matters for the conventions that name channels instead of numbering them:
#: the importer numbers named channels in sorted order.
CHANNELS: Tuple[Tuple[int, str], ...] = ((1, "DAPI"), (2, "FITC"),
                                          (3, "TRITC"))

#: Mask classes, in the order a Mask run stacks their planes after the images.
MASK_ROLES: Tuple[str, ...] = ("cell", "nucleus", "pathogen")

#: The wells and fields taken from the plate.
WELLS: Tuple[str, ...] = ("E01", "E02")
FIELDS: Tuple[int, ...] = (9, 10)

#: Side of the square cut out of each 1994 px field. Chosen for size: twenty
#: TIFF variants of twelve planes each is most of the download.
CROP = 896

#: The spaCR metadata columns a collaborator's table would not have; the
#: variant tables drop them and identify the image by its own file name.
SPACR_KEYS = ("plateID", "rowID", "columnID", "fieldID", "prc", "prcf",
              "file_name", "path_name")


@dataclass
class FieldData:
    """One field: its planes, its masks and the measurements of its cells."""

    well: str
    field: int
    images: Dict[int, np.ndarray]
    masks: Dict[str, np.ndarray]
    cells: "object" = None
    reference: Dict[str, "object"] = dc_field(default_factory=dict)

    @property
    def row(self) -> int:
        """1-based row of the well."""
        return ord(self.well[0]) - ord("A") + 1

    @property
    def column(self) -> int:
        """1-based column of the well."""
        return int(self.well[1:])


def _drop_edge_objects(labels: np.ndarray) -> np.ndarray:
    """Zero every object the crop cut through, so no mask holds half a cell."""
    labels = labels.copy()
    edge = np.unique(np.concatenate([labels[0], labels[-1],
                                     labels[:, 0], labels[:, -1]]))
    labels[np.isin(labels, edge[edge > 0])] = 0
    return labels


def _best_window(cell_mask: np.ndarray, size: int, step: int = 64
                 ) -> Tuple[int, int]:
    """The top-left corner of the crop that keeps the most whole cells."""
    best, corner = -1, (0, 0)
    height, width = cell_mask.shape
    for top in range(0, height - size + 1, step):
        for left in range(0, width - size + 1, step):
            window = cell_mask[top:top + size, left:left + size]
            kept = np.unique(_drop_edge_objects(window))
            if kept.size - 1 > best:
                best, corner = kept.size - 1, (top, left)
    return corner


def load_toxo_fields(plate: Path, wells: Sequence[str] = WELLS,
                     fields: Sequence[int] = FIELDS, crop: int = CROP
                     ) -> List[FieldData]:
    """Read the fields out of an unpacked toxo_mito example plate.

    Raw planes come from the acquisition TIFFs at the top of ``plate`` (the
    Mask demo), masks from the Mask run's merged arrays under ``merged/``, and
    measurements from ``measurements/measurements.db`` (the Annotate example).
    """
    import pandas as pd
    import tifffile

    plate = Path(plate)
    database = plate / "measurements" / "measurements.db"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    out: List[FieldData] = []
    try:
        for well in wells:
            for number in fields:
                images = {}
                for channel, _name in CHANNELS:
                    found = sorted(plate.glob(
                        f"plate1_{well}_T0001F{number:03d}L01A??Z01"
                        f"C{channel:02d}.tif"))
                    if len(found) != 1:
                        raise FileNotFoundError(
                            f"expected one raw plane for {well} field "
                            f"{number} channel {channel}, found {found}")
                    images[channel] = tifffile.imread(str(found[0]))
                merged = np.load(plate / "merged" /
                                 f"plate1_{well}_{number}_1.npy",
                                 mmap_mode="r")
                n_images = merged.shape[-1] - len(MASK_ROLES)
                masks_full = {role: np.asarray(merged[..., n_images + i])
                              for i, role in enumerate(MASK_ROLES)}
                top, left = _best_window(masks_full["cell"], crop)
                window = (slice(top, top + crop), slice(left, left + crop))
                cut = {c: np.ascontiguousarray(a[window])
                       for c, a in images.items()}
                masks = {role: _drop_edge_objects(m[window]).astype(np.uint16)
                         for role, m in masks_full.items()}
                row = ord(well[0]) - ord("A") + 1
                prcf = f"plate1_r{row}_c{int(well[1:])}_f{number}"
                reference = {}
                for table, role in (("cell", "cell"), ("nucleus", "nucleus"),
                                    ("pathogen", "pathogen"),
                                    ("cytoplasm", "cell")):
                    frame = pd.read_sql_query(
                        f'select * from "{table}" where prcf = ?',
                        connection, params=(prcf,))
                    kept = set(np.unique(masks[role]).tolist()) - {0}
                    reference[table] = frame[frame["object_label"].astype(
                        int).isin(kept)].reset_index(drop=True)
                out.append(FieldData(well=well, field=number, images=cut,
                                     masks=masks,
                                     cells=reference["cell"],
                                     reference=reference))
    finally:
        connection.close()
    return out


def synthetic_fields(size: int = 48, seed: int = 0) -> List[FieldData]:
    """The same wells, fields and channels with random pixels, for tests."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    out = []
    for well in WELLS:
        for number in FIELDS:
            images = {c: rng.integers(0, 4000, (size, size)).astype(np.uint16)
                      for c, _n in CHANNELS}
            masks = {}
            for index, role in enumerate(MASK_ROLES):
                labels = np.zeros((size, size), np.uint16)
                for obj in range(1, 4):
                    top = 4 + (obj - 1) * 14
                    labels[top:top + 8, 6 + index * 12:14 + index * 12] = obj
                masks[role] = labels
            cells = pd.DataFrame({
                "object_label": [1, 2, 3],
                "cell_area": [64.0, 64.0, 64.0],
                "cell_channel_0_mean_intensity": rng.random(3) * 1000,
            })
            out.append(FieldData(well=well, field=number, images=images,
                                 masks=masks, cells=cells,
                                 reference={"cell": cells.assign(
                                     prcf=f"plate1_{well}_{number}")}))
    return out


def _guid(*parts) -> str:
    """A stable GUID, the way MetaXpress stamps one on every file."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(map(str, parts)))
               ).upper()


def plane_path(key: str, fd: FieldData, channel: int) -> Optional[str]:
    """Where one plane lives in ``key``'s ``plate1`` folder, or None.

    None for a variant that keeps every channel of a field in one file.
    """
    w, f, c, r, col = fd.well, fd.field, channel, fd.row, fd.column
    name = dict(CHANNELS)[c]
    short = f"{w[0]}{col}"
    scene = WELLS.index(w) + 1 if w in WELLS else 1
    return {
        "zeiss_czi": f"{w}/plate1_S{scene:05d}_T{f:05d}_C{c:05d}.czi",
        "cellvoyager": f"plate1_{w}_T0001F{f:03d}L01A01Z01C{c:02d}.tif",
        "cq1": f"W{(r - 1) * 24 + col:04d}F{f:04d}T0001Z000C{c}.tif",
        "opera_phenix": f"Images/r{r:02d}c{col:02d}f{f:02d}p01-ch{c}"
                        f"sk1fk1fl1.tiff",
        "imagexpress": f"TimePoint_1/plate1_{w}_s{f}_w{c}"
                       f"{_guid('imagexpress', w, f, c)}.TIF",
        "arrayscan": f"plate1_{w}f{f - 1:02d}d{c - 1}.TIF",
        "arrayscan_kinetic": f"plate1i3t001{w}f{f - 1:02d}d{c - 1}.TIF",
        "evos": f"scan_R_p1_z1_0_{w}f{f:02d}d{c - 1}.tif",
        "incell": f"{w[0]} - {col:02d}(fld {f} wv {name} - {name}).tif",
        "scanr": f"data/{short}--W{(r - 1) * 24 + col:05d}--P{f:05d}"
                 f"--Z00000--T00000--{name}.tif",
        "cytation": f"{short}_01_{c}_{f}_{name}_001.tif",
        "leica_matrix_screener": (
            f"slide--S00/chamber--U{col - 1:02d}--V{r - 1:02d}/"
            f"field--X{f - 1:02d}--Y00/image--L00--S00--U{col - 1:02d}--"
            f"V{r - 1:02d}--J20--E00--O00--X{f - 1:02d}--Y00--T00--Z00--"
            f"C{c - 1:02d}.ome.tif"),
        "leica_lasx_series": f"{w}/Series{f:03d}_z00_ch{c - 1:02d}.tif",
        "leica_lasx_series_time": f"{w}/Pos{f - 1:03d}_t000_z00_"
                                  f"ch{c - 1:02d}.tif",
        "nikon_nis_xy": f"{w}xy{f:02d}c{c}.tif",
        "nikon_jobs": f"Well{w}_Channel{name}_Seq{f:04d}.tif",
        "micromanager_mda": f"{w}/img_channel{c - 1:03d}_position"
                            f"{f - 1:03d}_time000000000_z000.tif",
        "zeiss_zen_split_tiles": f"{w}/plate1_S{scene:05d}_T{f:05d}"
                                 f"_C{c:05d}.tiff",
        "custom": f"toxo-plate1__{w}__site{f:02d}__{name}.tif",
        "auto": f"{w}/fov{f:02d}_ch{c}.tif",
    }.get(key)


def field_path(key: str, fd: FieldData) -> Optional[str]:
    """Where a whole field lives, for the variants that hold channels inside."""
    return {
        "ome_tiff": f"{fd.well}/field{fd.field:03d}.ome.tif",
        "tiff_stack": f"{fd.well}/field{fd.field:03d}.tif",
    }.get(key)


def mask_path(key: str, fd: FieldData) -> str:
    """Where one field's mask lives under ``masks/<object>/``.

    Named the way the variant names images -- channel 1's name for a
    per-plane convention -- so the importer parses masks with the same
    convention and pairs them by plate, well and field. A CZI variant's
    masks are TIFFs of the same name: nobody exports a label image as CZI.
    """
    path = plane_path(key, fd, 1) or field_path(key, fd)
    if path.endswith(".czi"):
        path = path[:-4] + ".tif"
    return path


def _write_tiff(path: Path, array: np.ndarray, *, ome: bool = False,
                axes: str = "YX", channel_names: Sequence[str] = ()) -> None:
    """Write one TIFF, deflate-compressed."""
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    if ome:
        metadata = {"axes": axes}
        if channel_names:
            metadata["Channel"] = {"Name": list(channel_names)}
        tifffile.imwrite(str(path), array, ome=True, metadata=metadata,
                         compression="zlib")
    elif axes == "CYX":
        tifffile.imwrite(str(path), array, imagej=True,
                         metadata={"axes": "CYX",
                                   "Labels": list(channel_names)},
                         compression="zlib")
    else:
        tifffile.imwrite(str(path), array, compression="zlib")


def _write_czi(path: Path, array: np.ndarray, channel_name: str) -> None:
    """Write one plane as a CZI, with its channel named in the metadata."""
    from pylibCZIrw import czi as pyczi

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with pyczi.create_czi(str(path), exist_ok=True) as writer:
        writer.write(data=np.ascontiguousarray(array),
                     plane={"C": 0, "Z": 0, "T": 0}, scene=0)
        writer.write_metadata(document_name=path.stem,
                              channel_names={0: channel_name})


def _write_table(path: Path, frame) -> None:
    """Write one measurement table as a one-table SQLite file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(str(path))
    try:
        frame.to_sql("cell", connection, index=False)
    finally:
        connection.close()


def write_variant(out: Path, key: str, fields: Sequence[FieldData]
                  ) -> List[Dict[str, str]]:
    """Write one generated variant; return its manifest rows."""
    import pandas as pd

    variant = ix.import_variant(key)
    base = out / "variants" / key
    rows: List[Dict[str, str]] = []
    tables = []
    ranks = {fd.field: i for i, fd in enumerate(
        sorted({fd.field: fd for fd in fields}.values(),
               key=lambda item: item.field), start=1)}
    identity = ("rank" if key in ("auto", "ome_tiff", "tiff_stack",
                                  "leica_matrix_screener") else "number")
    for fd in fields:
        expected_field = fd.field if identity == "number" else ranks[fd.field]
        common = {"variant": key, "route": variant.route,
                  "metadata_type": variant.metadata_type,
                  "well": fd.well, "source_field": str(fd.field),
                  "field": str(expected_field), "field_identity": identity,
                  "z": "1", "t": "1"}
        whole = field_path(key, fd)
        if whole is not None:
            stack = np.stack([fd.images[c] for c, _n in CHANNELS])
            names = [n for _c, n in CHANNELS]
            _write_tiff(base / "plate1" / whole, stack,
                        ome=key == "ome_tiff", axes="CYX",
                        channel_names=names)
            rows.append(dict(common, kind="image",
                             path=f"variants/{key}/plate1/{whole}",
                             channel="1;2;3", channel_name=";".join(names)))
            image_key = whole
        else:
            for channel, name in CHANNELS:
                relative = plane_path(key, fd, channel)
                target = base / "plate1" / relative
                if key == "zeiss_czi":
                    _write_czi(target, fd.images[channel], name)
                else:
                    _write_tiff(target, fd.images[channel],
                                ome=relative.endswith(".ome.tif"))
                rows.append(dict(common, kind="image",
                                 path=f"variants/{key}/plate1/{relative}",
                                 channel=str(channel), channel_name=name))
            image_key = plane_path(key, fd, 1)
        for role in MASK_ROLES:
            relative = mask_path(key, fd)
            _write_tiff(base / "masks" / role / relative, fd.masks[role],
                        ome=relative.endswith(".ome.tif"))
            rows.append(dict(common, kind="mask", object=role,
                             path=f"variants/{key}/masks/{role}/{relative}",
                             channel="", channel_name=""))
        cells = fd.cells.drop(columns=[c for c in SPACR_KEYS
                                       if c in fd.cells.columns])
        cells.insert(0, "image", image_key)
        tables.append(cells)
    _write_table(base / "measurements.db",
                 pd.concat(tables, ignore_index=True))
    rows.append({"variant": key, "route": variant.route,
                 "metadata_type": variant.metadata_type, "kind":
                 "measurements", "path": f"variants/{key}/measurements.db"})
    return rows


#: The public sample files the converter-route variants are made of:
#: ``variant -> [(source file, name in the dataset, provenance)]``. All three
#: are CC BY 4.0 on downloads.openmicroscopy.org.
SAMPLES: Dict[str, List[Tuple[str, str, str]]] = {
    "nikon_nd2": [(
        "BF007.nd2", "A01/WellA01_ChannelBF_Seq0001.nd2",
        "https://downloads.openmicroscopy.org/images/ND2/maxime/BF007.nd2 "
        "-- (c) Maxime Woringer, CC BY 4.0")],
    "leica_lif": [
        ("FRAP.lif", "A01/FRAP.lif",
         "https://downloads.openmicroscopy.org/images/Leica-LIF/seanwarren/"
         "150519_FRAP_test_ROIs_chromagreen/"
         "150519_FRAP_test_ROIs_chromagreen.lif -- (c) Sean Warren, "
         "CC BY 4.0"),
        ("PR2729.lif", "A01/PR2729_frameOrderCombinedScanTypes.lif",
         "https://downloads.openmicroscopy.org/images/Leica-LIF/michael/"
         "PR2729_frameOrderCombinedScanTypes.lif -- (c) Michael Goelzer, "
         "CC BY 4.0")],
}


def write_samples(out: Path, samples: Path) -> List[Dict[str, str]]:
    """Copy the public ND2 / LIF samples in; return their manifest rows.

    Each sample folder also gets the CC BY 4.0 text the files were published
    under (``COPYING`` beside them on downloads.openmicroscopy.org, when it is
    in ``samples``) and a ``PROVENANCE.txt`` naming author and source.
    """
    rows = []
    for key, files in SAMPLES.items():
        variant = ix.import_variant(key)
        folder = out / "variants" / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "PROVENANCE.txt").write_text(
            "Public sample files, not spaCR's. No open library writes this "
            "format, so these are redistributed under their licence, CC BY "
            "4.0, with the attribution below.\n\n" + "\n".join(
                f"{relative}: {provenance}" for _s, relative, provenance
                in files) + "\n", encoding="utf-8")
        extra = ["PROVENANCE.txt"]
        if (samples / "COPYING").is_file():
            shutil.copyfile(samples / "COPYING",
                            folder / "LICENSE-CC-BY-4.0.txt")
            extra.append("LICENSE-CC-BY-4.0.txt")
        for name in extra:
            rows.append({"variant": key, "route": variant.route,
                         "metadata_type": variant.metadata_type,
                         "kind": "licence", "path": f"variants/{key}/{name}"})
        for source, relative, provenance in files:
            target = out / "variants" / key / "plate1" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(samples / source, target)
            rows.append({"variant": key, "route": variant.route,
                         "metadata_type": variant.metadata_type,
                         "kind": "sample",
                         "path": f"variants/{key}/plate1/{relative}",
                         "well": "A01", "provenance": provenance})
    return rows


MANIFEST_COLUMNS = ("variant", "route", "metadata_type", "kind", "path",
                    "well", "source_field", "field", "field_identity",
                    "channel", "channel_name", "z", "t", "object",
                    "provenance")


def build(fields: Sequence[FieldData], out: Path,
          samples: Optional[Path] = None,
          keys: Optional[Sequence[str]] = None) -> Path:
    """Write every variant, the reference database and the manifest."""
    import pandas as pd

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows: List[Dict[str, str]] = []
    for variant in ix.IMPORT_VARIANTS:
        if keys is not None and variant.key not in keys:
            continue
        if variant.route == "import":
            rows.extend(write_variant(out, variant.key, fields))
    if samples is not None:
        rows.extend(write_samples(out, Path(samples)))
    reference = out / "reference" / "measurements.db"
    reference.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(reference))
    try:
        for table in sorted({t for fd in fields for t in fd.reference}):
            frame = pd.concat([fd.reference[table] for fd in fields
                               if table in fd.reference], ignore_index=True)
            frame.to_sql(table, connection, index=False)
    finally:
        connection.close()
    with (out / ix.MANIFEST_NAME).open("w", newline="",
                                       encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_COLUMNS})
    return out


def _check_import(root: Path, key: str, rows: List[Dict[str, str]]
                  ) -> List[str]:
    """Import one variant's plan and compare it with the manifest."""
    from spacr import foreign as fgn

    inputs = ix.variant_inputs(root, key)
    plan = fgn.plan_import(inputs["images"], inputs["masks"],
                           inputs["measurements"],
                           metadata_type=inputs["metadata_type"],
                           custom_regex=inputs["custom_regex"] or None)
    problems: List[str] = []
    problems += [f"plan error: {e}" for e in plan.errors]
    problems += [f"image plan error: {e}" for e in plan.images.errors]
    problems += [f"unreadable: {s.path}: {s.error}"
                 for s in plan.images.unreadable]
    by_path = {os.path.normpath(str(root / r["path"])): r
               for r in rows if r["kind"] == "image"}
    seen = set()
    for mapping in plan.images.mappings:
        row = by_path.get(os.path.normpath(os.path.abspath(mapping.source)))
        if row is None:
            problems.append(f"{mapping.source}: not in the manifest")
            continue
        seen.add(row["path"])
        if row["channel"] == "1;2;3":
            want_channel = mapping.plane[2] + 1
        else:
            want_channel = int(row["channel"])
        got = (mapping.well, int(mapping.field), int(mapping.channel),
               int(mapping.z), int(mapping.t))
        want = (row["well"], int(row["field"]), want_channel,
                int(row["z"]), int(row["t"]))
        if got != want:
            problems.append(f"{row['path']}: imported as well/field/channel/"
                            f"z/t {got}, truly {want}")
    missing = {r["path"] for r in by_path.values()} - seen
    problems += [f"{p}: in the manifest but not imported"
                 for p in sorted(missing)]
    n_fields = len({(r["well"], r["field"]) for r in by_path.values()})
    if len(plan.masks.fields) != n_fields:
        problems.append(f"{len(plan.masks.fields)} field(s) paired with "
                        f"masks, expected {n_fields}")
    if plan.join.rows_matched != plan.join.rows_total:
        problems.append(f"{plan.join.rows_matched} of {plan.join.rows_total}"
                        f" measurement rows matched an object")
    return problems


def _check_convert(root: Path, key: str, rows: List[Dict[str, str]]
                   ) -> List[str]:
    """Scan and plan one converter-route variant."""
    from spacr import convert as cv

    inputs = ix.variant_inputs(root, key)
    plan = cv.plan(cv.scan(inputs["images"]))
    problems = [f"plan error: {e}" for e in plan.errors]
    problems += [f"unreadable: {s.path}: {s.error}" for s in plan.unreadable]
    if not plan.mappings:
        problems.append("nothing to convert")
    for mapping in plan.mappings:
        if mapping.well != "A01":
            problems.append(f"{mapping.source}: well {mapping.well}, "
                            f"truly A01")
    return problems


def verify(root: Path) -> Dict[str, List[str]]:
    """Every variant's problems, keyed by variant. Empty lists mean it passed."""
    root = Path(root)
    rows = ix.manifest_rows(root)
    results: Dict[str, List[str]] = {}
    for key in sorted({r["variant"] for r in rows},
                      key=[v.key for v in ix.IMPORT_VARIANTS].index):
        own = [r for r in rows if r["variant"] == key]
        variant = ix.import_variant(key)
        check = _check_import if variant.route == "import" else _check_convert
        try:
            results[key] = check(root, key, own)
        except Exception as exc:                              # noqa: BLE001
            results[key] = [f"{exc.__class__.__name__}: {exc}"]
    return results


def _megabytes(folder: Path) -> float:
    """The size of everything under ``folder``, in MB."""
    return sum(p.stat().st_size for p in Path(folder).rglob("*")
               if p.is_file()) / 1e6


def write_card(out: Path, results: Dict[str, List[str]]) -> Path:
    """Write the dataset card, ``README.md``, into the set."""
    out = Path(out)
    rows = ix.manifest_rows(out)
    total = _megabytes(out)
    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "  - image-segmentation",
        "tags:",
        "  - spacr",
        "  - microscopy",
        "  - file-formats",
        "  - high-content-screening",
        "pretty_name: spaCR Import test data (every format and naming)",
        "size_categories:",
        "  - 1K<n<10K",
        "---",
        "",
        "# spaCR — Import test data",
        "",
        "The same four microscope fields written in every container format "
        "and filename convention the Import module of "
        "[spaCR](https://github.com/EinarOlafsson/spacr) reads, each with its "
        "cell, nucleus and pathogen masks and the measurements of its cells. "
        "It is the data behind **Load test data…** on the Import screen: "
        "pick a variant, and spaCR fills the screen with it and previews the "
        "import, so you can see every file land on the well, field and "
        "channel it came from.",
        "",
        f"About {total:.0f} MB in one uncompressed archive, "
        f"`{ix.IMPORT_EXAMPLE_ARCHIVE}`. One download covers every variant.",
        "",
        "## Where the fields come from",
        "",
        "Wells **E01** and **E02**, fields **9** and **10**, channels **1–3** "
        "of the toxo_mito example plate `plate1` "
        "(`einarolafsson/toxo_mito`): Toxoplasma-infected host cells, channel "
        "1 nuclei, channel 2 host cells, channel 3 parasites.",
        "",
        "* **Images** are the raw acquisition planes (16-bit, not rescaled), "
        f"cut to {CROP} × {CROP} px. The window is the one that keeps the "
        "most whole cells.",
        "* **Masks** are the cell, nucleus and pathogen label images of the "
        "Mask run on the same plate (`einarolafsson/spacr-example-measure`), "
        "cut the same way. Every object the cut went through was removed, so "
        "no mask holds half an object.",
        "* **Measurements** are the Measure rows of exactly the cells left in "
        "the masks (`einarolafsson/spacr-example-annotate`). "
        "`reference/measurements.db` has them as spaCR wrote them (cell, "
        "nucleus, pathogen and cytoplasm tables); each variant's "
        "`measurements.db` has the cell table the way a collaborator's table "
        "would arrive: spaCR's own key columns removed, and an `image` "
        "column naming that variant's own file.",
        "",
        "Channels are named **DAPI**, **FITC** and **TRITC** in the "
        "conventions that name channels rather than number them. Import "
        "numbers named channels in sorted order, and these three sort in "
        "channel order.",
        "",
        "## The variants",
        "",
        "`variants/<key>/plate1/` holds the images, "
        "`variants/<key>/masks/{cell,nucleus,pathogen}/` the masks named the "
        "same way, and `variants/<key>/measurements.db` the table.",
        "",
        "| Key | Button | Files | Naming (`metadata_type`) | Example |",
        "|---|---|---|---|---|",
    ]
    for variant in ix.IMPORT_VARIANTS:
        naming = variant.metadata_type
        if variant.custom_regex:
            naming += f" `{variant.custom_regex}`"
        example = variant.example.replace("|", "\\|")
        lines.append(f"| `{variant.key}` | {variant.label} | "
                     f"{variant.file_format} | {naming} | `{example}` |")
    lines += [
        "",
        "## What is true of every file: `manifest.csv`",
        "",
        "One row per file. `path` is relative to the unpacked "
        f"`{ix.IMPORT_EXAMPLE_FOLDER}/` folder; `well`, `field`, `channel`, "
        "`z` and `t` are what the file truly is, and `source_field` is the "
        "field number on the original plate. `field_identity` says what an "
        "import should recover: `number` means the field number itself "
        "(field 9 stays field 9); `rank` means the order within the well, "
        "for the variants whose names carry no field number (a folder "
        "layout, a Leica Matrix Screener X/Y position, a file of channels). "
        "The manifest is the last member of the archive, so a download that "
        "died part-way has none and reads as absent.",
        "",
        "## Verified",
        "",
        "Every variant was imported with spaCR's own planner and each output "
        "plane compared with the manifest: well, field, channel, z and t, "
        "every mask paired with its field, and every measurement row "
        "matched to an object in the masks.",
        "",
        "| Variant | Result |",
        "|---|---|",
    ]
    for key, problems in results.items():
        lines.append(f"| `{key}` | "
                     f"{'passed' if not problems else 'FAILED: ' + problems[0]}"
                     f" |")
    samples = {r["path"]: r.get("provenance", "") for r in rows
               if r.get("kind") == "sample"}
    lines += [
        "",
        "What a pass proves, and what it does not: the files were written to "
        "follow each convention as spaCR's convention table documents it "
        "(the same table Mask uses, including which numbers count from "
        "zero), and read back by the Import module. It shows Import inverts "
        "those conventions. It cannot show that the table is right about "
        "every instrument; conventions marked provisional in spaCR's Naming "
        "list are the ones built from few real examples.",
        "",
        "## ND2 and LIF",
        "",
        "No open library writes Nikon ND2 or Leica LIF, so those two variants "
        "are not these fields: they are public sample files from "
        "downloads.openmicroscopy.org, redistributed under their licence, "
        "CC BY 4.0 (the text is in each folder as `LICENSE-CC-BY-4.0.txt`). "
        "They carry no masks, so their buttons fill the Format Converter "
        "rather than the Import screen.",
        "",
    ]
    for path, provenance in samples.items():
        lines.append(f"* `{path}` — {provenance}")
    lines += [
        "",
        "## Using it",
        "",
        "In spaCR, open **Import** and press **Load test data…**. The first "
        "choice downloads the archive into "
        f"`~/.cache/spacr/example_data/plate1/{ix.IMPORT_EXAMPLE_FOLDER}/`; "
        "later choices open from there. From a terminal: "
        "`spacr-download import`.",
        "",
        "## Licence",
        "",
        "MIT, like spaCR's other example sets, except the two sample folders "
        "`variants/nikon_nd2` and `variants/leica_lif`, which are CC BY 4.0 "
        "by the authors named above.",
        "",
    ]
    card = out / "README.md"
    card.write_text("\n".join(lines), encoding="utf-8")
    return card


def pack(out: Path, tar_path: Path) -> int:
    """Write the uncompressed archive, manifest LAST, under the set's prefix."""
    out = Path(out)
    members = sorted(p for p in out.rglob("*") if p.is_file()
                     and p.name != ix.MANIFEST_NAME)
    members.append(out / ix.MANIFEST_NAME)
    with tarfile.open(str(tar_path), "w") as tar:
        for path in members:
            info = tar.gettarinfo(str(path), arcname=(
                f"{ix.IMPORT_EXAMPLE_FOLDER}/"
                f"{path.relative_to(out).as_posix()}"))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                tar.addfile(info, handle)
    return os.path.getsize(tar_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``build``, ``verify`` or ``pack``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "verify", "pack"))
    parser.add_argument("--plate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--crop", type=int, default=CROP)
    args = parser.parse_args(argv)
    if args.command == "build":
        fields = load_toxo_fields(args.plate, crop=args.crop)
        for fd in fields:
            print(f"{fd.well} field {fd.field}: {len(fd.cells)} cells kept, "
                  f"masks " + ", ".join(
                      f"{r}={len(np.unique(m)) - 1}"
                      for r, m in fd.masks.items()))
        build(fields, args.out, samples=args.samples)
        print(f"built {args.out}")
        return 0
    if args.command == "verify":
        results = verify(args.out)
        failed = 0
        for key, problems in results.items():
            print(f"{'PASS' if not problems else 'FAIL'}  {key}")
            for problem in problems[:10]:
                print(f"      {problem}")
            failed += bool(problems)
        write_card(args.out, results)
        print(json.dumps({"variants": len(results), "failed": failed}))
        return 1 if failed else 0
    size = pack(args.out, args.tar)
    print(f"{args.tar}: {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
