"""Import's test data: one set of fields, written in every format and naming.

A small set of fields from the toxo_mito example plate, converted into the
container formats spaCR reads and renamed into every filename convention
Mask's ``metadata_type`` offers, published together with their masks and a
measurements database. Importing any variant should give back exactly the
wells, fields and channels the fields started as, and the manifest says what
those are.

WHAT IS PUBLISHED. :data:`IMPORT_EXAMPLE_REPO` holds one archive. Inside it,
under ``import_example/``:

* ``variants/<key>/plate1/`` -- the SAME twelve planes (wells E01 and E02,
  fields 9 and 10, three channels, cropped from the toxo_mito plate) written
  once per variant: once per filename convention, once as Zeiss CZI, once as
  OME-TIFF and once as an ImageJ TIFF stack;
* ``variants/<key>/masks/{cell,nucleus,pathogen}/`` -- the Mask run's label
  images for exactly those fields, named the way that variant names images;
* ``variants/<key>/measurements.db`` -- the Measure rows of exactly the cells
  in those masks, keyed by that variant's own file names, as a collaborator's
  table would be;
* ``reference/measurements.db`` -- the same rows as spaCR wrote them;
* ``variants/nikon_nd2`` and ``variants/leica_lif`` -- PUBLIC SAMPLE FILES,
  because no open library writes ND2 or LIF. They are other people's images
  under CC BY 4.0, with no masks, and they go through the Format Converter;
* ``manifest.csv`` -- one row per image file stating the well, field,
  channel, z and t it truly is. The LAST member of the archive, so a transfer
  that died part-way leaves no manifest and reads as absent.

THE REGISTRY IS HERE AND NOT IN THE ARCHIVE. :data:`IMPORT_VARIANTS` is what
the Import screen offers before anything is downloaded, and what
``tools/build_import_example.py`` writes. One list for both is what keeps a
button from naming a folder the archive does not have.

This module imports nothing from Qt, so the round-trip test and the builder
can use it headless.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "IMPORT_EXAMPLE_ARCHIVE",
    "IMPORT_EXAMPLE_FOLDER",
    "IMPORT_EXAMPLE_REPO",
    "IMPORT_VARIANTS",
    "ImportVariant",
    "import_example_folder",
    "import_variant",
    "is_present",
    "manifest_rows",
    "variant_inputs",
]

#: The Hugging Face dataset repository Import's test data is published in.
IMPORT_EXAMPLE_REPO = "einarolafsson/spacr-example-import"

#: The one archive that repository ships.
IMPORT_EXAMPLE_ARCHIVE = "spacr-example-import.tar"

#: The folder every member of the archive sits under. The archive unpacks
#: into the shared example plate like the other example sets, and this prefix
#: is what keeps its files from being mistaken for that plate's own -- the
#: Mask demo counts any ``*.tif`` at the plate's top level as itself.
IMPORT_EXAMPLE_FOLDER = "import_example"

#: The file that states what every image truly is.
MANIFEST_NAME = "manifest.csv"


@dataclass(frozen=True)
class ImportVariant:
    """One way the same fields were written, and how Import reads it back.

    :param key: the folder under ``variants/`` and the chooser's route key.
    :param label: the button text.
    :param route: ``'import'`` for the Import screen itself (images, masks
        and a measurement table), ``'convert'`` for the Format Converter
        (images only).
    :param metadata_type: the filename convention to parse by, or ``'auto'``
        for folders and tokens.
    :param custom_regex: the pattern, for ``metadata_type='custom'``.
    :param file_format: what the image files are, in words.
    :param example: one image path from the variant, relative to its
        ``plate1`` folder, so the chooser can show what the names look like
        before anything is downloaded.
    :param note: anything the reader should know about this variant.
    """

    key: str
    label: str
    route: str
    metadata_type: str
    custom_regex: str
    file_format: str
    example: str
    note: str = ""

    @property
    def description(self) -> str:
        """The paragraph the chooser shows on hover."""
        where = ("the Import screen: images, cell / nucleus / pathogen masks "
                 "and a measurements table"
                 if self.route == "import" else
                 "the Format Converter: images only")
        convention = (f"Naming: {self.metadata_type}"
                      + (f" with the pattern {self.custom_regex}"
                         if self.custom_regex else ""))
        text = (f"{self.file_format}, named like {self.example}. "
                f"{convention}. Fills {where}, then previews.")
        if self.note:
            text += f"\n\n{self.note}"
        return text


_CUSTOM_REGEX = (r"(?P<plateID>[^_]+)__(?P<wellID>[A-Z]\d{2})"
                 r"__site(?P<fieldID>\d+)__(?P<chanID>[A-Za-z0-9]+)")

#: Every variant, in the order the chooser lists them. Zeiss CZI first,
#: because it is the one a demonstration starts from.
IMPORT_VARIANTS: Tuple[ImportVariant, ...] = (
    ImportVariant(
        "zeiss_czi", "Test Zeiss CZI import", "import",
        "zeiss_zen_split_tiles", "", "Zeiss CZI, one plane per file",
        "E01/plate1_S00001_T00009_C00001.czi",
        "Written with pylibCZIrw, each file carrying its channel name in the "
        "CZI metadata. The scene index is not a well name, so the well comes "
        "from the folder."),
    ImportVariant(
        "cellvoyager", "Test Yokogawa CellVoyager import", "import",
        "cellvoyager", "", "TIFF", "plate1_E01_T0001F009L01A01Z01C01.tif"),
    ImportVariant(
        "cq1", "Test Yokogawa CQ1 import", "import", "cq1", "", "TIFF",
        "W0097F0009T0001Z000C1.tif",
        "The CQ1 writes a well INDEX over 24 columns: W0097 is E01."),
    ImportVariant(
        "opera_phenix", "Test Opera Phenix import", "import",
        "opera_phenix", "", "TIFF", "Images/r05c01f09p01-ch1sk1fk1fl1.tiff",
        "Harmony writes row and column numbers, never a well name: r05c01 "
        "is E01."),
    ImportVariant(
        "imagexpress", "Test ImageXpress import", "import", "imagexpress", "",
        "TIFF", "TimePoint_1/plate1_E01_s9_w1<GUID>.TIF"),
    ImportVariant(
        "arrayscan", "Test ArrayScan import", "import", "arrayscan", "",
        "TIFF", "plate1_E01f08d0.TIF",
        "Fields and dyes count from zero: f08d0 is field 9, channel 1."),
    ImportVariant(
        "arrayscan_kinetic", "Test ArrayScan kinetic import", "import",
        "arrayscan_kinetic", "", "TIFF", "plate1i3t001E01f08d0.TIF"),
    ImportVariant(
        "evos", "Test EVOS import", "import", "evos", "", "TIFF",
        "scan_R_p1_z1_0_E01f09d0.tif"),
    ImportVariant(
        "incell", "Test IN Cell import", "import", "incell", "", "TIFF",
        "E - 01(fld 9 wv DAPI - DAPI).tif",
        "Channels are named by filter, so their order is the order of the "
        "names."),
    ImportVariant(
        "scanr", "Test ScanR import", "import", "scanr", "", "TIFF",
        "data/E1--W00097--P00009--Z00000--T00000--DAPI.tif"),
    ImportVariant(
        "cytation", "Test Cytation import", "import", "cytation", "", "TIFF",
        "E1_01_1_9_DAPI_001.tif"),
    ImportVariant(
        "leica_matrix_screener", "Test Leica Matrix Screener import",
        "import", "leica_matrix_screener", "", "OME-TIFF, one plane per file",
        "slide--S00/chamber--U00--V04/field--X08--Y00/image--L00--S00--U00--"
        "V04--J20--E00--O00--X08--Y00--T00--Z00--C00.ome.tif",
        "The Matrix Screener names a field by its X/Y position, not a number, "
        "so fields are numbered in order."),
    ImportVariant(
        "leica_lasx_series", "Test Leica LAS X series import", "import",
        "leica_lasx_series", "", "TIFF", "E01/Series009_z00_ch00.tif",
        "A series export carries no well, so the well comes from the folder."),
    ImportVariant(
        "leica_lasx_series_time", "Test Leica LAS X timelapse import",
        "import", "leica_lasx_series_time", "", "TIFF",
        "E01/Pos008_t000_z00_ch00.tif"),
    ImportVariant(
        "nikon_nis_xy", "Test Nikon NIS-Elements import", "import",
        "nikon_nis_xy", "", "TIFF", "E01xy09c1.tif"),
    ImportVariant(
        "nikon_jobs", "Test Nikon JOBS import", "import", "nikon_jobs", "",
        "TIFF", "WellE01_ChannelDAPI_Seq0009.tif"),
    ImportVariant(
        "micromanager_mda", "Test Micro-Manager import", "import",
        "micromanager_mda", "", "TIFF",
        "E01/img_channel000_position008_time000000000_z000.tif"),
    ImportVariant(
        "zeiss_zen_split_tiles", "Test Zeiss ZEN split tiles import",
        "import", "zeiss_zen_split_tiles", "", "TIFF",
        "E01/plate1_S00001_T00009_C00001.tiff"),
    ImportVariant(
        "custom", "Test custom naming import", "import", "custom",
        _CUSTOM_REGEX, "TIFF", "toxo-plate1__E01__site09__DAPI.tif"),
    ImportVariant(
        "auto", "Test folder-layout import", "import", "auto", "",
        "TIFF", "E01/fov09_ch1.tif",
        "No convention: the well is the folder and the channel is the ch "
        "token. Fields are numbered in order, because a folder layout "
        "states no field numbers to keep."),
    ImportVariant(
        "ome_tiff", "Test OME-TIFF import", "import", "auto", "",
        "OME-TIFF, three channels per file", "E01/field009.ome.tif",
        "Channels are read from inside the file."),
    ImportVariant(
        "tiff_stack", "Test TIFF stack import", "import", "auto", "",
        "ImageJ TIFF stack, three channels per file", "E01/field009.tif",
        "Channels are read from inside the file."),
    ImportVariant(
        "nikon_nd2", "Test Nikon ND2 import", "convert", "auto", "",
        "Nikon ND2 (public sample)", "A01/WellA01_ChannelBF_Seq0001.nd2",
        "No open library writes ND2, so this is a public sample file by "
        "Maxime Woringer, CC BY 4.0: one brightfield plane, renamed the way "
        "NIS-Elements JOBS names a well-plate acquisition. It has no masks, "
        "so it goes through the Format Converter."),
    ImportVariant(
        "leica_lif", "Test Leica LIF import", "convert", "auto", "",
        "Leica LIF (public samples)", "A01/FRAP.lif",
        "No open library writes LIF, so these are public sample files by "
        "Sean Warren and Michael Goelzer, CC BY 4.0: time series, a z-stack "
        "and a tile scan. They have no masks, so they go through the Format "
        "Converter."),
)


def import_variant(key: str) -> ImportVariant:
    """The variant called ``key``.

    :param key: a key from :data:`IMPORT_VARIANTS`.
    :raises KeyError: naming the keys that exist.
    """
    for candidate in IMPORT_VARIANTS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"no import test variant named {key!r}; there is "
                   f"{', '.join(v.key for v in IMPORT_VARIANTS)}")


def import_example_folder(plate_folder: Optional[Path] = None) -> Path:
    """Where the unpacked set lives.

    :param plate_folder: the shared example plate the archive unpacks into;
        :func:`spacr.example_archives.example_plate_folder` when omitted.
    :returns: ``<plate folder>/import_example``. Not created here.
    """
    if plate_folder is None:
        from .example_archives import example_plate_folder
        plate_folder = example_plate_folder()
    return Path(plate_folder) / IMPORT_EXAMPLE_FOLDER


def manifest_rows(root) -> List[Dict[str, str]]:
    """Every row of the set's ``manifest.csv``.

    :param root: the unpacked ``import_example`` folder.
    :returns: the rows, or ``[]`` when there is no readable manifest.
    """
    try:
        with (Path(root) / MANIFEST_NAME).open(
                newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def is_present(root, key: Optional[str] = None) -> bool:
    """Whether a complete copy of the set -- or of one variant -- is here.

    :param root: the unpacked ``import_example`` folder.
    :param key: one variant, or every variant when omitted.
    :returns: ``True`` only when the manifest exists and every file it lists
        for the variant(s) is on disk. A half-unpacked set reads as absent,
        which is the answer that gets it downloaded again.
    """
    root = Path(root)
    rows = [row for row in manifest_rows(root)
            if key is None or row.get("variant") == key]
    return bool(rows) and all((root / row["path"]).is_file() for row in rows)


def variant_inputs(root, key: str) -> Dict[str, object]:
    """What the chooser fills in for one variant.

    :param root: the unpacked ``import_example`` folder.
    :param key: the variant.
    :returns: ``images`` (the ``plate1`` folder), ``masks`` (``{object:
        folder}`` for the import route, empty for the converter),
        ``measurements`` (the table, or ``''``), ``metadata_type`` and
        ``custom_regex``.
    """
    variant = import_variant(key)
    folder = Path(root) / "variants" / key
    masks: Dict[str, str] = {}
    measurements = ""
    if variant.route == "import":
        for role in ("cell", "nucleus", "pathogen"):
            candidate = folder / "masks" / role
            if candidate.is_dir():
                masks[role] = str(candidate)
        table = folder / "measurements.db"
        measurements = str(table) if table.is_file() else ""
    return {
        "images": str(folder / "plate1"),
        "masks": masks,
        "measurements": measurements,
        "metadata_type": variant.metadata_type,
        "custom_regex": variant.custom_regex,
        "route": variant.route,
    }
