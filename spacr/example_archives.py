"""The published example datasets: what they are, and how to fetch them.

This module is separate from :mod:`spacr.qt.hf_download`, which owns the same
downloads with a Qt progress dialog wrapped round them, and it imports PySide6
at module scope. ``spacr-download`` has to work on a cluster login node with no
display and no Qt installed at all, so everything that is only about the data
-- which repositories publish it, what each archive is called, how a stream is
verified, how an archive is safely unpacked, and what has to be repaired on
arrival -- lives here instead. ``hf_download`` imports these names back out, so
every existing GUI caller (and every test that patches one of them on that
module) keeps working unchanged.

This module is distinct from :mod:`spacr.example_data`, which fetches the
Regression example screen's count and score CSVs from a GitHub release. That is
a different set of files, a different host and a different transport. This
module is about the ``.tar`` archives of imaging data: the Mask demo plate, the
Measure example, the Annotate/Classify example, the Replication and Recruitment
assay examples, and -- through :mod:`spacr.screen_data` -- the pieces of the
published TSG101 screen.

:data:`EXAMPLE_SETS` carries one size per set, as
:class:`spacr.screen_data.ScreenAsset` does, so the picker and the CLI can show
sizes before anything is fetched.
"""
from __future__ import annotations

import hashlib
import logging
import os
import socket
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .import_examples import IMPORT_EXAMPLE_ARCHIVE as _IMPORT_EXAMPLE_ARCHIVE
from .import_examples import IMPORT_EXAMPLE_REPO as _IMPORT_EXAMPLE_REPO

LOG = logging.getLogger("spacr.example_archives")

__all__ = [
    "ANNOTATE_EXAMPLE_REPO",
    "CONTROL_CHART_EXAMPLE_REPO",
    "DATASET_PLACEHOLDER",
    "DATASET_REPO",
    "DATASET_SUB",
    "DOSE_EXAMPLE_REPO",
    "EXAMPLE_ARCHIVES",
    "EXAMPLE_ARCHIVE_SHA256",
    "EXAMPLE_SETS",
    "ExampleSet",
    "HIT_EXAMPLE_REPO",
    "IMPORT_EXAMPLE_REPO",
    "INVASION_EXAMPLE_REPO",
    "HOST_PATHOGEN_EXAMPLE_REPO",
    "MEASURE_EXAMPLE_REPO",
    "RECRUITMENT_EXAMPLE_REPO",
    "REPLICATION_EXAMPLE_REPO",
    "OPS_EXAMPLE_REPO",
    "READS_EXAMPLE_REPO",
    "SETTINGS_REPO",
    "STITCH_EXAMPLE_REPO",
    "TEM_EXAMPLE_REPO",
    "TIMELAPSE_BACTERIA_EXAMPLE_REPO",
    "TIMELAPSE_MAMMALIAN_EXAMPLE_REPO",
    "TIMELAPSE_YEAST_EXAMPLE_REPO",
    "check_archive_sha256",
    "download_archive",
    "example_plate_folder",
    "example_set",
    "example_set_folder",
    "expand_measure_arrays",
    "explain_download_failure",
    "extract_example_archive",
    "make_the_example_paths_absolute",
]


DATASET_REPO  = "einarolafsson/toxo_mito"
DATASET_SUB   = "plate1"
SETTINGS_REPO = "einarolafsson/spacr_settings"

#: Measure's own example data: the merged arrays a Mask run produces, so
#: Measure can be exercised without segmenting anything first.
#:
#: A SEPARATE REPO from the Mask demo because it is a different artefact at a
#: different stage -- `toxo_mito` is raw acquisition, this is that plate after
#: `preprocess_generate_masks`. Sixteen fields across four wells; the wells are
#: all kept because well-level aggregation and between-condition comparison
#: are most of what Measure does after the per-object step.
MEASURE_EXAMPLE_REPO = "einarolafsson/spacr-example-measure"

#: Annotate and Classify share one example set: the crops a Measure run cut,
#: the measurements database that indexes them, and 88 real labels.
#:
#: ONE REPO, TWO SETTINGS FILES. Both modules read the same 282 MB of crops and
#: the same database; only the settings differ. Publishing it twice would
#: double the download and let the two copies drift.
ANNOTATE_EXAMPLE_REPO = "einarolafsson/spacr-example-annotate"

#: The Replication Assay's example: every parasite object of twelve control
#: wells of the Toxoplasma MTOC screen, plate 1, with two merged fields.
#:
#: ITS OWN FOLDER, like Recruitment's. Both assay sets ship
#: ``measurements/measurements.db``, and so does the Annotate set in the shared
#: plate, so unpacking either there would overwrite the other's database.
REPLICATION_EXAMPLE_REPO = "einarolafsson/spacr-example-replication"

#: The Recruitment module's example: twelve control wells of the THP-1 RNF213
#: screen, plate 1, whose own recruitment settings ship with it.
RECRUITMENT_EXAMPLE_REPO = "einarolafsson/spacr-example-recruitment"
HOST_PATHOGEN_EXAMPLE_REPO = "einarolafsson/spacr-example-host-pathogen"

#: Import's test data: the same twelve planes written in every vendor format
#: and filename convention Import reads, with their masks and measurements.
#: Described in :mod:`spacr.import_examples`, which owns the variant list.
#: Every member sits under ``import_example/``, so it unpacks into the shared
#: plate without touching the other sets' files.
IMPORT_EXAMPLE_REPO = _IMPORT_EXAMPLE_REPO

#: The Invasion Assay's example, and it is SYNTHETIC. No real two-colour
#: differential-staining acquisition was available when the set was prepared.
#: The fields are drawn, Mask segments them with Cellpose-SAM, and Measure
#: makes the database. Ground truth records the drawn objects, not the masks.
#: The versioned cache keeps earlier drawn-mask examples intact. See
#: ``tools/build_invasion_example_dataset.py``.
INVASION_EXAMPLE_REPO = "einarolafsson/spacr-example-invasion"
#: The optical pooled screen's example: two sequencing fields of one well of a
#: published screen, every cycle, and the guide library to decode them against.
#:
#: A SAMPLE OF SOMEONE ELSE'S SCREEN, NOT THE LAB'S OWN. The fields are from
#: Funk et al. 2022 (BioImage Archive S-BIAD394), byte for byte. The dataset
#: card carries the citation and the licence; the repository exists only once
#: rehosting it has been approved.
OPS_EXAMPLE_REPO = "einarolafsson/spacr-example-ops"

#: Align & Stitch's example: a 3 x 3 block of overlapping tiles from the same
#: well, cut from the same published screen as :data:`OPS_EXAMPLE_REPO`.
STITCH_EXAMPLE_REPO = "einarolafsson/spacr-example-stitch"

#: Dose-Response's example: four replicate A549 plates of one LINCS Cell
#: Painting plate map (Cell Painting Gallery cpg0004-lincs, CC0 1.0), as
#: well-level profiles, with the regression and training runs that Prediction
#: Profiler, Run Compare, Run History and Training Runs open. See
#: ``tools/build_dose_example_dataset.py``.
DOSE_EXAMPLE_REPO = "einarolafsson/spacr-example-dose"

#: Control Chart's example: every well of the 24 CPJUMP1 compound plates
#: (Cell Painting Gallery cpg0000-jump-pilot, CC0 1.0), with their DMSO and
#: positive-control wells. See ``tools/build_control_chart_example_dataset.py``.
CONTROL_CHART_EXAMPLE_REPO = "einarolafsson/spacr-example-control-chart"

#: Investigate Hit's example: 30 wells (fields 1-4) of plate 1 of the
#: published TSG101 recruitment screen (:data:`spacr.screen_data.SCREEN_REPO`)
#: with their per-cell predictions and guide counts, and a Regression run made
#: from them that names GRA14 (239740). See
#: ``tools/build_hit_example_dataset.py``.
HIT_EXAMPLE_REPO = "einarolafsson/spacr-example-hit"

#: Map Barcodes' cached reads: the first 100,000 reads of each of the eight
#: published FASTQ files of BioProject PRJNA1261935, with ENA's listing of the
#: project, so Load test data works without reaching ENA. See
#: ``tools/build_reads_example_dataset.py`` and :mod:`spacr.sra`.
READS_EXAMPLE_REPO = "einarolafsson/spacr-example-reads"

#: Make Masks' TEM test data: ten transmission electron micrographs with
#: instance masks of every mitochondrion, from EMPIAR-10982's TEM benchmark
#: (CC0 1.0; Conrad and Narayan 2023). The masks unpack into
#: ``ground_truth_masks/``, so the fields open unsegmented. See
#: ``tools/build_tem_example_dataset.py``.
TEM_EXAMPLE_REPO = "einarolafsson/spacr-example-tem"

FIBSEM_EXAMPLE_REPO = "einarolafsson/spacr-example-fibsem"

#: Make Masks' timelapse test data: one public movie each of budding yeast,
#: bacteria and mammalian cells, with ground-truth masks in every frame and
#: the published lineage embedded in the mask TIFFs, for Time mode and the
#: lineage tools. See ``tools/build_timelapse_example_datasets.py``.
TIMELAPSE_YEAST_EXAMPLE_REPO = "einarolafsson/spacr-example-timelapse-yeast"
TIMELAPSE_BACTERIA_EXAMPLE_REPO = "einarolafsson/spacr-example-timelapse-bacteria"
TIMELAPSE_MAMMALIAN_EXAMPLE_REPO = "einarolafsson/spacr-example-timelapse-mammalian"

#: The token a published settings file uses for "wherever this was unpacked".
DATASET_PLACEHOLDER = "<dataset>"


def example_plate_folder() -> Path:
    """Return the shared plate folder the plate example sets unpack into.

    ``~/.cache/spacr/example_data/plate1`` -- a real spaCR plate directory,
    holding whichever of ``merged/``, ``data/``, ``measurements/`` and
    ``settings/`` have been downloaded.

    The sets share one folder because they are used together: `data/` holds the
    crops and `measurements/measurements.db` indexes them. Each archive's
    members are relative to this folder, so the three unpack into it side by
    side and compose into a plate that Measure, Annotate and Classify can all
    be pointed at.

    The assay examples are not stages of this plate and each has a folder of
    its own beside it; see :func:`example_set_folder`.
    """
    return Path.home() / ".cache" / "spacr" / "example_data" / "plate1"


def _prepare_embeddings_example(destination=None) -> Dict[str, Path]:
    """Unpack the bundled geometric demo into its own example-cache folder."""
    folder = (Path(destination) if destination is not None else
              example_plate_folder().parent / "embeddings_synthetic_v1")
    archive = (Path(__file__).resolve().parent / "resources" / "data" /
               "embeddings_synthetic_example.tar.gz")
    extract_example_archive(archive, folder)
    return {"folder": folder, "crops": folder / "crops",
            "labels": folder / "labels.csv"}


#: The single archive each example repo ships, keyed by repo.
#:
#: ONE REQUEST INSTEAD OF THOUSANDS. The annotate set is 2,365 files; fetching
#: it a file at a time spent most of its wall clock on HTTP round trips, and
#: `snapshot_download` -- the obvious alternative -- cannot be interrupted, so
#: Cancel did nothing and quitting mid-download aborted the process.
#:
#: A tar fixes all three: one stream that can be stopped between chunks, one
#: progress figure that means something, and no per-file overhead at either
#: end. It is NOT compressed: the payloads are PNGs and .npz arrays, already
#: compressed, so gzip would cost minutes of CPU on every download to save
#: almost nothing.
EXAMPLE_ARCHIVES: Dict[str, str] = {
    DATASET_REPO: "spacr-example-mask.tar",
    MEASURE_EXAMPLE_REPO: "spacr-example-measure.tar",
    ANNOTATE_EXAMPLE_REPO: "spacr-example-annotate.tar",
    REPLICATION_EXAMPLE_REPO: "spacr-example-replication.tar",
    RECRUITMENT_EXAMPLE_REPO: "spacr-example-recruitment.tar",
    HOST_PATHOGEN_EXAMPLE_REPO: "spacr-example-host-pathogen.tar",
    IMPORT_EXAMPLE_REPO: _IMPORT_EXAMPLE_ARCHIVE,
    INVASION_EXAMPLE_REPO: "spacr-example-invasion.tar",
    OPS_EXAMPLE_REPO: "spacr-example-ops.tar",
    STITCH_EXAMPLE_REPO: "spacr-example-stitch.tar",
    DOSE_EXAMPLE_REPO: "spacr-example-dose.tar",
    CONTROL_CHART_EXAMPLE_REPO: "spacr-example-control-chart.tar",
    HIT_EXAMPLE_REPO: "spacr-example-hit.tar",
    READS_EXAMPLE_REPO: "spacr-example-reads.tar",
    TEM_EXAMPLE_REPO: "spacr-example-tem.tar",
    FIBSEM_EXAMPLE_REPO: "spacr-example-fibsem.tar",
    TIMELAPSE_YEAST_EXAMPLE_REPO: "spacr-example-timelapse-yeast.tar",
    TIMELAPSE_BACTERIA_EXAMPLE_REPO: "spacr-example-timelapse-bacteria.tar",
    TIMELAPSE_MAMMALIAN_EXAMPLE_REPO: "spacr-example-timelapse-mammalian.tar",
}

#: The SHA-256 each pinned archive must have, keyed by archive name.
#:
#: A download whose bytes hash differently is refused before anything is
#: unpacked, so a republished or corrupted archive is never mistaken for the
#: one the tests and the dataset cards describe. Archives absent here are
#: checked for length only, as they always were.
EXAMPLE_ARCHIVE_SHA256: Dict[str, str] = {
    "spacr-example-timelapse-yeast.tar": "4dd911e6d075e7cf7e2f7d750374aa9543c906119fe6f06cbcdf195afaef6de0",
    "spacr-example-timelapse-bacteria.tar": "39072dce56dff12ef73cac13700f25a32f36a9c67fa590a6079d90e53e2cc928",
    "spacr-example-timelapse-mammalian.tar": "2261aa6f2e0877fec3fab51ab9a9c075a61fe34844106a547005ef154fad7bb5",
}


def check_archive_sha256(file_name: str, digest: str) -> None:
    """Refuse an archive whose SHA-256 differs from its pin.

    :param file_name: the archive's name in its repository.
    :param digest: the hex SHA-256 of the bytes that arrived.
    :raises IOError: when :data:`EXAMPLE_ARCHIVE_SHA256` pins ``file_name``
        to another digest.
    """
    expected = EXAMPLE_ARCHIVE_SHA256.get(Path(file_name).name)
    if expected is not None and digest != expected:
        raise IOError(
            f"the download of {file_name} does not match its published "
            f"checksum (SHA-256 {digest}, expected {expected}). Nothing was "
            "unpacked.")


@dataclass(frozen=True)
class ExampleSet:
    """One published example dataset, described before it is downloaded.

    :param key: what a user types to ask for this one.
    :param repo: the Hugging Face dataset repository it is published in.
    :param summary: one line saying what the set is and what it is for.
    :param bytes: the archive's approximate size: the figure the GUI buttons
        state ("about 280 MB"), taken from the archive at publication. It is
        not checked against the download. The exact length is verified during
        the transfer, against what the server declares.
    :param markers: what the unpacked set leaves behind, as globs relative to
        the folder it unpacks into. Presence is tested on these rather than on
        the folder, because all three sets share one plate directory: the
        directory existing says nothing about which of them is in it.
    :param expands_npz: whether ``.npz`` arrays have to be written back out as
        the ``.npy`` Measure reads. See :func:`expand_measure_arrays`.
    :param in_default: whether ``spacr-download`` with no arguments fetches it.
        False for the OPS and Align & Stitch samples: together they are 0.6 GB
        of one screen's tiles, and adding them would push the default run past
        ``CONFIRM_ABOVE_BYTES``, so the default run would require confirmation.
        They are fetched when named, and by ``all``.
    :param folder: the folder under the example-data root the set unpacks
        into. ``plate1`` -- the shared plate -- for the three sets that are
        stages of one plate; a folder of its own for a set that would collide
        with them.
    """

    key: str
    repo: str
    summary: str
    bytes: int
    markers: Tuple[str, ...]
    expands_npz: bool = False
    folder: str = "plate1"
    in_default: bool = True

    @property
    def archive(self) -> str:
        """The ``.tar`` this set ships as."""
        return EXAMPLE_ARCHIVES[self.repo]

    def is_present(self, folder) -> bool:
        """Whether this set is already unpacked under ``folder``.

        Every marker has to match. A half-unpacked set -- a transfer that died
        between the images and the database -- is therefore reported as absent
        and is downloaded again.

        :param folder: the plate directory the set unpacks into.
        """
        folder = Path(folder)
        return all(any(folder.glob(pattern)) for pattern in self.markers)


#: Every example set, in the order the pipeline runs them.
#:
#: THE ORDER IS THE PIPELINE'S. Mask segments raw images into merged stacks,
#: Measure cuts crops out of those stacks and measures them, Annotate and
#: Classify label the crops. Someone downloading all three is following that
#: sequence, and a list in any other order would have to be re-sorted in the
#: reader's head. The assay sets come last: each is a measured plate that an
#: assay module reads, which is where they sit in the pipeline too.
EXAMPLE_SETS: Tuple[ExampleSet, ...] = (
    ExampleSet(
        key="import",
        repo=IMPORT_EXAMPLE_REPO,
        summary="Import test data: two wells, two fields and three channels "
                "written in 22 formats and filename conventions, with masks "
                "and measurements, plus public ND2 and LIF samples.",
        bytes=284_000_000,
        markers=("import_example/manifest.csv",),
    ),
    ExampleSet(
        key="mask",
        repo=DATASET_REPO,
        summary="Mask test data: one raw toxo_mito plate, plus the settings to "
                "segment it.",
        bytes=400_000_000,
        markers=("*.tif",),
    ),
    ExampleSet(
        key="measure",
        repo=MEASURE_EXAMPLE_REPO,
        summary="Measure example: sixteen merged fields across four wells, "
                "as a Mask run leaves them.",
        bytes=390_000_000,
        markers=("merged/*.npy",),
        expands_npz=True,
    ),
    ExampleSet(
        key="annotate",
        repo=ANNOTATE_EXAMPLE_REPO,
        summary="Annotate / Classify example: 2,341 single-cell crops, the "
                "database that indexes them, and 88 real labels.",
        bytes=280_000_000,
        markers=("measurements/measurements.db", "data"),
    ),
    ExampleSet(
        key="replication",
        repo=REPLICATION_EXAMPLE_REPO,
        summary="Replication Assay example: 11,416 parasites in twelve "
                "control wells of the MTOC screen, and its settings.",
        bytes=168_000_000,
        markers=("measurements/measurements.db",
                 "settings/replication_settings.csv"),
        folder="replication",
    ),
    ExampleSet(
        key="recruitment",
        repo=RECRUITMENT_EXAMPLE_REPO,
        summary="Recruitment example: twelve control wells of the THP-1 "
                "RNF213 screen, and the settings it was analysed with.",
        bytes=147_000_000,
        markers=("measurements/measurements.db",
                 "settings/recruitment_settings.csv"),
        folder="recruitment",
    ),
    ExampleSet(
        key="host_pathogen",
        repo=HOST_PATHOGEN_EXAMPLE_REPO,
        summary="Host–Pathogen test data: two real THP-1 RNF213 fields, "
                "164 measured hosts and 97 vacuoles, with images and masks.",
        bytes=114_000_000,
        markers=("measurements/measurements.db", "settings/host_pathogen_settings.csv",
                 "example_manifest.json", "merged/.spacr_plane_layout.json",
                 "merged/PLATE1_E01_1_1.npy", "merged/PLATE1_E02_1_1.npy"),
        folder="host_pathogen",
        in_default=False,
    ),
    ExampleSet(
        key="invasion",
        repo=INVASION_EXAMPLE_REPO,
        summary="Invasion Assay example, SYNTHETIC: two-colour images drawn "
                "by spaCR, segmented by Mask, then measured by Measure.",
        bytes=107_000_000,
        markers=("measurements/measurements.db",
                 "settings/invasion_settings.csv"),
        folder="invasion_segmented_v1",
    ),
    ExampleSet(
        key="stitch",
        repo=STITCH_EXAMPLE_REPO,
        summary="Align & Stitch example: a 3 x 3 block of overlapping 10X "
                "tiles from a published optical pooled screen.",
        bytes=197_000_000,
        markers=("manifest.csv", "tiles/*.tif"),
        folder="align_stitch",
        in_default=False,
    ),
    ExampleSet(
        key="ops",
        repo=OPS_EXAMPLE_REPO,
        summary="OPS example: two fields of a published optical pooled "
                "screen, all eleven sequencing cycles, and its guide library.",
        bytes=394_000_000,
        markers=("manifest.csv", "sequencing/c11/*.tif"),
        folder="ops_screen",
        in_default=False,
    ),
    ExampleSet(
        key="dose",
        repo=DOSE_EXAMPLE_REPO,
        summary="Dose-Response example: four replicate A549 plates of the "
                "LINCS Cell Painting set (CC0), 56 compounds at six doses, "
                "with regression and training runs made from them.",
        bytes=266_000,
        markers=("dose_plate.csv", "runs/*/results.csv"),
        folder="dose_response_lincs",
        in_default=False,
    ),
    ExampleSet(
        key="control_chart",
        repo=CONTROL_CHART_EXAMPLE_REPO,
        summary="Control Chart example: 9,131 wells of the 24 CPJUMP1 "
                "compound plates (CC0), DMSO and positive-control wells "
                "marked.",
        bytes=1_600_000,
        markers=("control_chart_wells.csv",),
        folder="control_chart_cpjump1",
        in_default=False,
    ),
    ExampleSet(
        key="hit",
        repo=HIT_EXAMPLE_REPO,
        summary="Investigate Hit example: 2,608 cells in 30 wells of the "
                "TSG101 recruitment screen, plate 1, with their predictions, "
                "guide fractions and a Regression run naming GRA14.",
        bytes=27_791_360,
        markers=("hit.json", "measurements/measurements.db"),
        folder="hit_example",
        in_default=False,
    ),
    ExampleSet(
        key="reads",
        repo=READS_EXAMPLE_REPO,
        summary="Map Barcodes test reads: the first 100,000 reads of each "
                "of the eight published FASTQ files (PRJNA1261935), so Load "
                "test data works offline.",
        bytes=27_000_000,
        markers=("manifest.json", "filereport.tsv", "*_1.fastq.gz"),
        folder="reads",
        in_default=False,
    ),
    ExampleSet(
        key="tem",
        repo=TEM_EXAMPLE_REPO,
        summary="Make Masks TEM test data: ten transmission electron "
                "micrographs with mitochondria masks, from EMPIAR-10982 "
                "(CC0).",
        bytes=31_508_480,
        markers=("manifest.csv", "tem_*.tif", "ground_truth_masks/tem_*.tif"),
        folder="make_masks_tem",
        in_default=False,
    ),
    ExampleSet(
        key="fibsem",
        repo=FIBSEM_EXAMPLE_REPO,
        summary="Make Masks FIB-SEM test data: one 128 x 256 x 256 HeLa "
                "cell volume (15 nm) with 3-D mitochondria labels, from "
                "EMPIAR-10982 (CC0).",
        bytes=25_221_120,
        markers=("manifest.csv", "hela_fibsem.tif",
                 "ground_truth_masks/hela_fibsem.tif"),
        folder="make_masks_fibsem",
        in_default=False,
    ),
    ExampleSet(
        key="timelapse_yeast",
        repo=TIMELAPSE_YEAST_EXAMPLE_REPO,
        summary="Make Masks timelapse test data, yeast: a budding "
                "S. cerevisiae colony with masks in every frame and its "
                "mother-bud lineage (CC BY 4.0).",
        bytes=69_672_960,
        markers=("manifest.csv", "ground_truth_masks/*.tif"),
        folder="make_masks_timelapse_yeast",
        in_default=False,
    ),
    ExampleSet(
        key="timelapse_bacteria",
        repo=TIMELAPSE_BACTERIA_EXAMPLE_REPO,
        summary="Make Masks timelapse test data, bacteria: a dividing "
                "C. glutamicum microcolony with masks in every frame and its "
                "lineage (CC BY 4.0).",
        bytes=55_777_280,
        markers=("manifest.csv", "ground_truth_masks/*.tif"),
        folder="make_masks_timelapse_bacteria",
        in_default=False,
    ),
    ExampleSet(
        key="timelapse_mammalian",
        repo=TIMELAPSE_MAMMALIAN_EXAMPLE_REPO,
        summary="Make Masks timelapse test data, mammalian: migrating and "
                "dividing T98G cells with masks in every frame and their "
                "lineage (CC BY 4.0).",
        bytes=78_018_560,
        markers=("manifest.csv", "ground_truth_masks/*.tif"),
        folder="make_masks_timelapse_mammalian",
        in_default=False,
    ),
)


def example_set(key: str) -> ExampleSet:
    """The set called ``key``.

    :param key: an :data:`EXAMPLE_SETS` key, such as ``import``, ``mask``,
        ``measure``, ``annotate``, ``dose``, ``control_chart`` or ``hit``.
    :raises KeyError: naming the keys that do exist. A typo therefore raises
        rather than returning ``None``.
    """
    for candidate in EXAMPLE_SETS:
        if candidate.key == key:
            return candidate
    raise KeyError(f"no example set named {key!r}; there is "
                   f"{', '.join(s.key for s in EXAMPLE_SETS)}")


def example_set_folder(key: str) -> Path:
    """Where the set called ``key`` unpacks: its folder beside the plate.

    :param key: an :data:`EXAMPLE_SETS` key.
    :returns: ``~/.cache/spacr/example_data/<folder>``. Not created here.
    :raises KeyError: as :func:`example_set` does.
    """
    return example_plate_folder().parent / example_set(key).folder


def explain_download_failure(exc: BaseException) -> str:
    """Convert a download exception into a message with cause and remedy.

    Downloading test data requires a network connection. For the ordinary
    offline case ``str(exc)`` is a nested urllib3 dump::

        (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443):
        Max retries exceeded with url: /api/datasets/... (Caused by
        NewConnectionError('<urllib3.connection.HTTPSConnection object at
        0x7e8d...>: Failed to establish a new connection: [Errno 101] Network
        is unreachable'))"), '(Request ID: 73ac20ed-...)')

    It states neither that the machine is offline nor what to do. The three
    conditions recognised are: no network, the ``huggingface_hub`` extra not
    installed, and a truncated transfer. Each gets a sentence naming the cause
    and the way out; anything else keeps its own message with the same closing
    advice attached.

    :param exc: the exception raised inside the download worker.
    :returns: a multi-line message for the failure dialog.
    """
    offline_hint = (
        "Use an already downloaded test dataset or your own local images "
        "while offline. To fetch test data, reconnect and choose Load test data again.")

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            "Downloading test data needs the 'huggingface_hub' package to "
            "list the dataset repository, and it is not installed in this "
            f"environment ({exc}).\n\n"
            "Install it with:  pip install huggingface_hub\n\n"
            + offline_hint)

    if isinstance(exc, OSError) and "Truncated download" in str(exc):
        return (
            f"{exc}\n\n"
            "The connection dropped part-way through. Nothing partial was "
            "kept, so re-running the download starts the file again.\n\n"
            + offline_hint)

    network_errors: tuple = (ConnectionError, TimeoutError, socket.gaierror)
    try:
        import requests
        network_errors += (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
    except Exception:
        pass

    if isinstance(exc, network_errors):
        return (
            "Could not reach huggingface.co, so the test dataset could "
            "not be downloaded. Check your internet connection (or your "
            "proxy settings) and try again.\n\n"
            + offline_hint)

    return f"{exc}\n\n{offline_hint}"


def _list_files(repo_id: str, subfolder: str) -> List[str]:
    """Return every file path in ``repo_id`` matching ``subfolder``.

    Empty subfolder means "top-level CSVs only" (mirrors the Tk
    downloader's behaviour for the settings pack).

    :raises ImportError: when ``huggingface_hub`` is not installed. Re-raised
        with the package named rather than letting the bare
        ``ModuleNotFoundError`` text stand on its own, because
        :func:`explain_download_failure` turns it into install instructions
        and the message is what the user reads.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError as exc:
        raise ImportError(f"huggingface_hub is not installed: {exc}") from exc
    files = list_repo_files(repo_id, repo_type="dataset")
    if subfolder:
        return [f for f in files if f.startswith(subfolder)]
    return [f for f in files if f.endswith(".csv")]


def _content_length(resp) -> Optional[int]:
    """Declared body size from the response, or None when unusable.

    Hugging Face always sends ``Content-Length`` for a resolved LFS
    object, so this doubles as the integrity check for
    :func:`_download_one`: fewer bytes on disk than advertised means the
    stream was cut short.
    """
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def download_archive(repo_id: str, file_name: str, dest_dir: Path, *,
                     progress: Optional[Callable[[int, Optional[int]], None]]
                     = None,
                     chunk_size: int = 1 << 15) -> Path:
    """Stream one file from the HF repo to ``dest_dir/basename``.

    Uses plain streamed HTTP rather than the ``hf_hub`` download machinery,
    so a one-off demo download creates no Hugging Face cache directory.

    The body lands in a sibling ``.part`` file and is only moved onto the final
    path once every advertised byte has arrived. A dropped connection therefore
    never leaves a truncated file at the destination.

    :param repo_id: the Hugging Face dataset repository.
    :param file_name: the path within it; only the basename is kept on disk.
    :param dest_dir: the folder the file lands in.
    :param progress: called as ``progress(written, expected)`` after each
        chunk, where ``expected`` is ``None`` when the server declared no
        length. A CLI draws a percentage from it; the GUI does not use it,
        because its workers emit Qt signals from inside their own loops.
    :param chunk_size: 32 KB by default. An archive of gigabytes passes a
        bigger one.
    """
    import requests
    import tempfile
    url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
             f"{file_name}?download=true")
    dst = Path(dest_dir) / Path(file_name).name
    part = None
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    expected = _content_length(resp)
    written = 0
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=dst.parent,
                                         prefix="." + dst.name + ".",
                                         suffix=".part", delete=False) as fh:
            part = Path(fh.name)
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, expected)
        if expected is not None and written != expected:
            raise IOError(
                f"Truncated download for {file_name}: wrote {written} "
                f"bytes but the server declared {expected}."
            )
        check_archive_sha256(file_name, digest.hexdigest())
        os.replace(part, dst)
    except BaseException:
        if part is not None:
            try:
                part.unlink()
            except OSError:
                pass
        raise
    return dst


#: The name the Qt downloader has always called it by.
#:
#: Kept as an alias rather than renamed at every call site because
#: ``monkeypatch.setattr(hf_download, "_download_one", ...)`` is how several
#: tests stop the demo flow reaching the network. Renaming it would leave
#: those patches naming an attribute nothing calls, and the tests would go
#: to the network for real.
_download_one = download_archive


def extract_example_archive(archive, dest) -> int:
    """Unpack a downloaded example archive under ``dest``.

    Members are extracted with ``filter="data"``. A tar can name
    ``../../etc/something`` or an absolute path, and a plain ``extractall``
    writes there, so an unfiltered unpack would let anyone who can publish to
    the repository write anywhere the user can write. The filter rejects those
    members, along with device nodes, setuid bits and symlinks pointing outside
    the tree.

    Python 3.12 and later have it built in. Older interpreters get an explicit
    check instead of a silent unfiltered unpack.

    :param archive: the ``.tar`` on disk.
    :param dest: the folder to unpack into.
    :returns: how many members were written.
    """
    import tarfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive)) as tar:
        members = tar.getmembers()
        if hasattr(tarfile, "data_filter"):
            tar.extractall(str(dest), filter="data")
        else:
            for member in members:
                name = member.name
                if name.startswith("/") or ".." in name.split("/"):
                    raise ValueError(
                        f"refusing to unpack {name!r}: it escapes the "
                        f"destination folder")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(
                        f"refusing to unpack {name!r}: it is not a plain file "
                        f"or directory")
            tar.extractall(str(dest))
    return len(members)


def expand_measure_arrays(merged: Path) -> None:
    """Write each ``.npz`` back out as the ``.npy`` Measure reads.

    The compression is a transport detail -- it halves a 700 MB download -- and
    Measure loads `.npy`. Converting on arrival keeps that entirely inside the
    downloader rather than teaching every reader about a second format.

    The ``.npz`` is removed afterwards: keeping both would double the disk cost
    of an example dataset, and nothing reads the ``.npz`` afterwards.

    This is a module function rather than a method: `after_extract` runs on the
    download thread, and calling a worker method such as
    ``_MeasureExampleWorker(dest)._expand_arrays(...)`` would construct a
    QObject on that thread, which the GUI thread may then not touch. The
    function needs no ``self``.

    :param merged: the folder the ``.npz`` arrays were unpacked into.
    """
    import numpy as np


    if not merged.is_dir():
        return
    for archive in sorted(merged.glob("*.npz")):
        target = archive.with_suffix(".npy")
        try:
            if target.is_file():
                existing = np.lib.format.open_memmap(target, mode="r")
                existing._mmap.close()
                archive.unlink(missing_ok=True)
                continue
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix=".spacr-array-", dir=merged) as staging:
                staged = Path(staging) / "array.npy"
                with np.load(archive) as bundle:
                    key = "image" if "image" in bundle else bundle.files[0]
                    with staged.open("wb") as output:
                        np.save(output, bundle[key])
                        output.flush()
                        os.fsync(output.fileno())
                os.link(staged, target)
            archive.unlink(missing_ok=True)
        except Exception:                                # noqa: BLE001
            LOG.warning("could not unpack %s", archive, exc_info=True)


def make_the_example_paths_absolute(root) -> int:
    """Rewrite a downloaded example's stored paths to its unpack location.

    A measurements database stores absolute paths to its crops, which name the
    machine that made it and resolve nowhere else. The published copy stores
    them relative to the dataset root instead, so it is portable and carries no
    account name -- and this function turns them back into absolute paths.

    The settings files are rewritten the same way: they carry
    :data:`DATASET_PLACEHOLDER` where the unpack location goes, so a user can
    press Run without first editing a path.

    Idempotent. A path that is already absolute is left alone, so running this
    twice -- a re-download over an existing copy -- does not produce
    ``/home/me/data//home/me/data/...``.

    :param root: the folder the dataset was unpacked into.
    :returns: how many values were rewritten.
    """
    from .database_concurrency import connect

    root = Path(root)
    prefix = str(root).rstrip("/\\") + os.sep
    rewritten = 0

    for database in (root / "measurements" / "measurements.db",
                     root / "measurements.db"):
        if database.is_file():
            break
    if database.is_file():
        connection = connect(database)
        try:
            tables = [r[0] for r in connection.execute(
                "select name from sqlite_master where type='table'")]
            for table in tables:
                columns = [r[1] for r in connection.execute(
                    f'PRAGMA table_info("{table}")')]
                for column in columns:
                    try:
                        cursor = connection.execute(
                            f'update "{table}" set "{column}" = ? || replace("{column}", \'/\', ?) '
                            f'where cast("{column}" as text) like \'data/%\' '
                            f'or cast("{column}" as text) like \'measurements/%\'',
                            (prefix, os.sep))
                        rewritten += cursor.rowcount or 0
                    except sqlite3.Error:
                        continue
            connection.commit()
        finally:
            connection.close()

    for settings_file in sorted((root / "settings").glob("*.csv")):
        try:
            text = settings_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if DATASET_PLACEHOLDER not in text:
            continue
        settings_file.write_text(
            text.replace(DATASET_PLACEHOLDER, str(root).rstrip("/")),
            encoding="utf-8")
        rewritten += 1

    LOG.info("example dataset at %s: %d paths made absolute", root, rewritten)
    return rewritten
