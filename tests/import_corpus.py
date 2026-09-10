"""Synthetic image trees, one per real microscope layout spaCR must import.

INSTRUCTION 363 SAYS TO BUILD THIS FIRST, and the reason is worth stating: an
import module that must accept "any number, of any formatted, images, with any
number of dimensions, with any folder or file metadata structure" has no
meaning as a requirement until "any" is written down as a list of concrete
trees. This module is that list. It is the SPECIFICATION, and
``tests/test_the_import_corpus_is_the_specification.py`` is what measures the
current code against it.

Every tree here is built from a real acquisition convention rather than
invented, because the failure this guards against is an importer that handles
the shapes someone imagined and not the ones microscopes emit.

The images are 4x4 uint16 so the whole corpus costs a few kilobytes: this
tests NAMING AND STRUCTURE, which is where the difficulty is. Pixel content is
irrelevant to every question asked of it.

WHAT EACH TREE VARIES, and none of them is exotic:

  cellvoyager      plate, well, T, field, laser, action, Z, channel, all in
                   the filename. spaCR's default and the one convention it is
                   confident about.
  cq1              the same information, a different grammar, no plate.
  harmony          Opera Phenix: r01c01f01p01-ch1sk1fk1fl1. Row and column are
                   SEPARATE, so 'A01' never appears -- a parser expecting a
                   well name finds none.
  imagexpress      Plate_A01_s1_w1: site rather than field, wavelength rather
                   than channel, and the extension is upper case.
  flat_ome         one OME-TIFF per field, dimensions INSIDE the file. Nothing
                   in the name says how many channels there are.
  per_well_folder  the well is the FOLDER, the field and channel the file.
  per_channel_folder  the channel is the FOLDER, named for the dye rather than
                   numbered -- so the channel order is alphabetical by dye
                   name unless something maps it.
  z_stack_in_file  one file per field holding Z pages.
  time_in_file     one file per field holding T pages. Byte-identical in
                   layout to the Z case: only the file's own metadata says
                   which, which is why a page index means nothing alone.
  tiled            a field split into tiles that must be stitched or tracked.

A tree is ``(name, builder)``; the builder takes a directory and returns a
:class:`CorpusTree` saying what the truth is, so a test can compare what was
parsed against what was put on disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tifffile

#: Small enough that the whole corpus is a few kilobytes.
TILE = (4, 4)


@dataclass(frozen=True)
class CorpusTree:
    """One synthetic acquisition, and the truth about what is in it.

    :param name: the convention, e.g. ``"cellvoyager"``.
    :param root: the directory the images were written under.
    :param files: every image written, relative to ``root``.
    :param truth: per relative path, the metadata that path really carries --
        ``plate``, ``well``, ``field``, ``channel`` and where present ``z``,
        ``t``, ``tile``. A key ABSENT means the layout does not encode it,
        which is different from encoding it as zero.
    :param metadata_type: the ``metadata_type`` spaCR would need, or ``""``
        where no built-in convention describes this tree. An empty string is
        the interesting case: it is the specification gap.
    :param note: what makes this tree hard, in one sentence.
    """

    name: str
    root: Path
    files: Tuple[str, ...]
    truth: Dict[str, Dict[str, object]]
    metadata_type: str
    note: str

    @property
    def wells(self) -> set:
        return {t["well"] for t in self.truth.values() if "well" in t}

    @property
    def channels(self) -> set:
        return {t["channel"] for t in self.truth.values() if "channel" in t}


def _write(path: Path, pages: int = 1, axes: str = "") -> None:
    """Write a tiny TIFF, multi-page when ``pages`` > 1.

    ``axes`` NAMES WHAT THE PAGES ARE -- ``"ZYX"``, ``"TYX"``, ``"CYX"`` --
    and is written into the file, because that is what a real microscope
    emits and it is the only thing that can tell a Z-stack from a timelapse.
    Without it tifffile reads a ``(2, 4, 4)`` array back as ONE image with two
    samples per pixel, not as two pages, so a corpus that omitted it was
    testing against files no microscope produces.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if pages <= 1:
        tifffile.imwrite(str(path), np.zeros(TILE, dtype=np.uint16))
        return
    data = np.zeros((pages, *TILE), dtype=np.uint16)
    # `photometric="minisblack"` OR THE FIRST AXIS IS READ AS SAMPLES-PER-
    # PIXEL: a (2, 4, 4) array with no hint comes back as one RGB-ish image
    # rather than two pages, which is not what any of these trees mean.
    kwargs = {"metadata": {"axes": axes or "ZYX"},
              "photometric": "minisblack"}
    if path.name.endswith((".ome.tif", ".ome.tiff")):
        kwargs["ome"] = True
    tifffile.imwrite(str(path), data, **kwargs)


def _tree(name, root, truth, metadata_type, note) -> CorpusTree:
    return CorpusTree(name=name, root=root, files=tuple(sorted(truth)),
                      truth=truth, metadata_type=metadata_type, note=note)


# --------------------------------------------------------------------------
# The builders. Two wells, two fields, two channels everywhere, so a parser
# that silently collapses one axis produces a count that is visibly wrong.
# --------------------------------------------------------------------------
WELLS = ("A01", "B02")
FIELDS = (1, 2)
CHANNELS = (1, 2)


def build_cellvoyager(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"plate1_{well}_T0001F{f:03d}L01A01Z01C{c:02d}.tif"
                _write(root / rel)
                truth[rel] = {"plate": "plate1", "well": well, "field": f,
                              "channel": c, "z": 1, "t": 1}
    return _tree("cellvoyager", root, truth, "cellvoyager",
                 "spaCR's default; every axis is in the filename.")


def build_cq1(root: Path) -> CorpusTree:
    truth = {}
    for i, well in enumerate(WELLS, start=1):
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"W{i}F{f:03d}T0001Z01C{c}.tif"
                _write(root / rel)
                truth[rel] = {"well": well, "field": f, "channel": c,
                              "z": 1, "t": 1}
    return _tree("cq1", root, truth, "cq1",
                 "No plate in the name; the well is an index, not a name.")


def build_harmony(root: Path) -> CorpusTree:
    """Opera Phenix / Harmony: row and column are separate fields."""
    truth = {}
    for row, col, well in ((1, 1, "A01"), (2, 2, "B02")):
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"r{row:02d}c{col:02d}f{f:02d}p01-ch{c}sk1fk1fl1.tiff"
                _write(root / rel)
                truth[rel] = {"well": well, "row": row, "column": col,
                              "field": f, "channel": c, "z": 1}
    return _tree("harmony", root, truth, "",
                 "Row and column are separate, so the string 'A01' never "
                 "appears; a parser expecting a well name finds none.")


def build_imagexpress(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"Plate1_{well}_s{f}_w{c}.TIF"
                _write(root / rel)
                truth[rel] = {"plate": "Plate1", "well": well, "field": f,
                              "channel": c}
    return _tree("imagexpress", root, truth, "",
                 "Site not field, wavelength not channel, upper-case "
                 "extension.")


def build_flat_ome(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            rel = f"Plate1_{well}_F{f:03d}.ome.tif"
            _write(root / rel, pages=len(CHANNELS), axes="CYX")
            truth[rel] = {"plate": "Plate1", "well": well, "field": f,
                          "channels_inside": len(CHANNELS)}
    return _tree("flat_ome", root, truth, "",
                 "Channels are pages inside the file; nothing in the name "
                 "says how many there are.")


def build_per_well_folder(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"{well}/field{f}_c{c}.tif"
                _write(root / rel)
                truth[rel] = {"well": well, "field": f, "channel": c}
    return _tree("per_well_folder", root, truth, "",
                 "The well is the FOLDER; the filename alone cannot say "
                 "which well an image is from.")


def build_per_channel_folder(root: Path) -> CorpusTree:
    truth = {}
    for dye, c in (("DAPI", 1), ("GFP", 2)):
        for well in WELLS:
            for f in FIELDS:
                rel = f"{dye}/{well}_f{f}.tif"
                _write(root / rel)
                truth[rel] = {"well": well, "field": f, "channel": c,
                              "dye": dye}
    return _tree("per_channel_folder", root, truth, "",
                 "The channel is a FOLDER named for the dye, so channel "
                 "ORDER is alphabetical unless something maps dye to index.")


def build_z_stack_in_file(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"plate1_{well}_F{f:03d}_C{c}.tif"
                _write(root / rel, pages=5, axes="ZYX")
                truth[rel] = {"plate": "plate1", "well": well, "field": f,
                              "channel": c, "z_pages": 5}
    return _tree("z_stack_in_file", root, truth, "",
                 "Five pages per file that are Z planes.")


def build_time_in_file(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for c in CHANNELS:
                rel = f"plate1_{well}_F{f:03d}_C{c}.tif"
                _write(root / rel, pages=5, axes="TYX")
                truth[rel] = {"plate": "plate1", "well": well, "field": f,
                              "channel": c, "t_pages": 5}
    return _tree("time_in_file", root, truth, "",
                 "Identical in NAME to z_stack_in_file: only the file's "
                 "own axes metadata says whether the pages are Z or T.")


def build_tiled(root: Path) -> CorpusTree:
    truth = {}
    for well in WELLS:
        for f in FIELDS:
            for tile in (1, 2, 3, 4):
                for c in CHANNELS:
                    rel = f"plate1_{well}_F{f:03d}_tile{tile:02d}_C{c}.tif"
                    _write(root / rel)
                    truth[rel] = {"plate": "plate1", "well": well, "field": f,
                                  "channel": c, "tile": tile}
    return _tree("tiled", root, truth, "",
                 "Each field is four tiles that must be stitched or tracked "
                 "as one field.")


#: Every tree, in the order a report should list them: the two spaCR claims
#: to handle first, so a regression in those is the first thing seen.
BUILDERS: Tuple[Tuple[str, Callable[[Path], CorpusTree]], ...] = (
    ("cellvoyager", build_cellvoyager),
    ("cq1", build_cq1),
    ("harmony", build_harmony),
    ("imagexpress", build_imagexpress),
    ("flat_ome", build_flat_ome),
    ("per_well_folder", build_per_well_folder),
    ("per_channel_folder", build_per_channel_folder),
    ("z_stack_in_file", build_z_stack_in_file),
    ("time_in_file", build_time_in_file),
    ("tiled", build_tiled),
)


def build_all(root: Path) -> List[CorpusTree]:
    """Write every tree under ``root``, one directory each."""
    root = Path(root)
    return [builder(root / name) for name, builder in BUILDERS]
