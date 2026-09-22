"""Test data for OPS and for Align & Stitch, one button away in each.

OPS uses a small sample of an optical pooled screen, and Align & Stitch uses
a small segment of the same OPS data.

TWO SAMPLES OF ONE PUBLISHED SCREEN. Both are cut from plate
``20200202_6W-LaC024A``, well A1, of Funk et al. 2022 (BioImage Archive
S-BIAD394), unmodified:

* :data:`~spacr.example_archives.STITCH_EXAMPLE_REPO` -- nine cycle-1 tiles,
  a 3 x 3 block from the middle of the well, acquired in snake-column order
  with 213 px of overlap. Align & Stitch plans and writes it as it comes.
* :data:`~spacr.example_archives.OPS_EXAMPLE_REPO` -- sites 331 and 332, every
  one of the eleven sequencing cycles, and the guide library's 11-base
  prefixes. The OPS engine stitches, segments and decodes them.

WHY THE OPS SAMPLE IS THE WELL'S LAST TWO SITES. The engine takes a well's
field count as its highest site number plus one and fits the round well to
that count, so a sample has to include the last site to keep the measured
333-field layout; see :func:`spacr.ops_engine.run_ops`. And it carries no
phenotype images, because the phenotype phase aligns anchors spread over the
whole well, which a two-field sample cannot supply.

THE SAME PATTERN AS EVERY OTHER EXAMPLE SET. One uncompressed ``.tar`` per
repository, streamed behind the shared progress dialog and unpacked through
the tar data filter by :class:`spacr.qt.hf_download._TarExampleWorker`, into a
folder of its own beside the shared example plate (:func:`example_folder`). A
cached copy is used without a request.

WHAT "FILLS THE SETTINGS" MEANS HERE. The OPS panel gets its source, output
folder, library and plate name; Align & Stitch gets its tile folder, the
block's 3 x 3 grid, its order, its overlap and a place to write. Run -- or
Plan -- is then the next action.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from ..example_archives import (EXAMPLE_ARCHIVES, OPS_EXAMPLE_REPO,
                                STITCH_EXAMPLE_REPO, example_set_folder)
from .hf_download import (_TarExampleWorker, download_toxo_mito_demo,
                          explain_download_failure)
from .i18n import tr

LOG = logging.getLogger("spacr.qt.ops_stitch_demo")

__all__ = [
    "MANIFEST_NAME",
    "OPS_PLATE",
    "STITCH_GRID",
    "STITCH_ORDER",
    "STITCH_OVERLAP",
    "example_folder",
    "install_align_button",
    "is_present",
    "listed_files",
    "load_align_test_data",
    "load_the_test_data",
    "ops_settings_for",
    "stitch_settings_for",
]

#: The file that lists what a complete copy holds, one row per file with its
#: size. It is the LAST member of each archive, so a transfer that died
#: part-way leaves no manifest and reads as absent.
MANIFEST_NAME = "manifest.csv"

#: The plate both samples were cut from. The OPS engine names its tables by
#: plate, and the folder the tiles unpack into is not called this.
OPS_PLATE = "20200202_6W-LaC024A"

#: The stitch sample's layout: three rows by three columns, the first column
#: read top to bottom and the next bottom to top, as the microscope took them.
STITCH_GRID = (3, 3)
STITCH_ORDER = "snake-column"

#: Neighbouring tiles overlap by 213 px of 1,480, which is the acquisition's
#: own raster setting (``ops_raster_overlap``). It only seeds the search.
STITCH_OVERLAP = 0.144

#: What the worker reports when the user pressed Cancel.
CANCELLED = "Cancelled by user."


def example_folder(which: str) -> Path:
    """Where the ``"ops"`` or ``"stitch"`` sample is unpacked and cached.

    :param which: ``"ops"`` or ``"stitch"``.
    :returns: the folder, beside the shared example plate. Not created here.
    """
    return example_set_folder(which)


def listed_files(folder) -> List[str]:
    """Every file the sample's manifest says a complete copy holds.

    :param folder: where the sample was unpacked.
    :returns: paths relative to ``folder``; empty when there is no readable
        manifest or a row names no file, so that an unrecognised folder is
        never mistaken for the sample.
    """
    try:
        with (Path(folder) / MANIFEST_NAME).open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []
    files: List[str] = []
    for row in rows:
        name = (row.get("path") or "").strip()
        if not name:
            return []
        files.append(name)
    return files


def is_present(folder) -> bool:
    """Whether a complete copy of a sample is unpacked in ``folder``.

    :param folder: where the sample is expected.
    :returns: True only when the manifest exists and every file it lists is
        there. A half-unpacked sample reads as absent, which is the answer
        that gets it downloaded again.
    """
    folder = Path(folder)
    files = listed_files(folder)
    return bool(files) and all((folder / name).is_file() for name in files)


def ops_settings_for(folder) -> Dict[str, object]:
    """The OPS settings that run the unpacked sample.

    :param folder: where the OPS sample was unpacked.
    :returns: the source, the output folder, the library and the plate. The
        output goes to ``ops_output`` inside the sample rather than into the
        source, which would mix the tables with the tiles.
    """
    folder = Path(folder)
    return {
        "genotype_source": str(folder / "sequencing"),
        "phenotype_source": None,
        "dst_root": str(folder / "ops_output"),
        "ops_library": str(folder / "library" / "pool10_prefixes.csv"),
        "plate": OPS_PLATE,
    }


def stitch_settings_for(folder) -> Dict[str, object]:
    """The Align & Stitch settings that plan and write the unpacked block.

    :param folder: where the stitch sample was unpacked.
    :returns: a :func:`spacr.align.default_settings`-shaped partial dict.
    """
    folder = Path(folder)
    return {
        "src": str(folder / "tiles"),
        "dst": str(folder / "stitched"),
        "grid": STITCH_GRID,
        "overlap": STITCH_OVERLAP,
        "order": STITCH_ORDER,
        "reference_channel": 0,
    }


class _OpsTarWorker(_TarExampleWorker):
    """Fetches the OPS sample."""

    repo = OPS_EXAMPLE_REPO
    archive = EXAMPLE_ARCHIVES[OPS_EXAMPLE_REPO]


class _StitchTarWorker(_TarExampleWorker):
    """Fetches the Align & Stitch sample."""

    repo = STITCH_EXAMPLE_REPO
    archive = EXAMPLE_ARCHIVES[STITCH_EXAMPLE_REPO]


_WORKERS = {"ops": _OpsTarWorker, "stitch": _StitchTarWorker}


def _download(which: str) -> Callable:
    """The downloader for one sample, called as ``(parent, dest, on_done)``."""
    titles = {
        "ops": tr("Downloading the OPS test data"),
        "stitch": tr("Downloading the Align & Stitch test data"),
    }

    def run(parent, dest, on_done) -> None:
        """Start the download behind the shared progress dialog."""
        download_toxo_mito_demo(parent, Path(dest), on_done,
                                worker_factory=_WORKERS[which],
                                title=titles[which])
    return run


def load_the_test_data(which: str, *, use: Callable[[Path], object],
                       report: Callable[[str, bool], None],
                       button: Optional[QPushButton] = None,
                       parent=None, ask=None, folder=None) -> bool:
    """Put a sample in place, downloading it first when it is not cached.

    :param which: ``"ops"`` or ``"stitch"``.
    :param use: called with the folder once a complete copy is there; it
        fills the screen's settings.
    :param report: called as ``report(text, is_error)`` with what happened.
    :param button: disabled while a download runs, and relabelled.
    :param parent: the widget the progress dialog belongs to.
    :param ask: replaces the downloader, for tests; called as
        ``ask(parent, folder, on_done)``.
    :param folder: replaces :func:`example_folder`, for tests.
    :returns: whether the sample was put in place by this call. False while a
        download is still running, and after a failure.
    """
    folder = Path(folder) if folder is not None else example_folder(which)
    if is_present(folder):
        use(folder)
        return True

    if button is not None:
        button.setEnabled(False)
        button.setText(tr("Fetching test data…"))
    report(tr("Downloading the test data into {path}", path=str(folder)),
           False)

    def _done(result, error) -> None:
        """Put the button back, then use the data or say why not."""
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Load test data…"))
        if result is None:
            if error == CANCELLED:
                report(tr("The test data download was cancelled."), False)
                return
            LOG.info("%s test data not downloaded: %s", which, error)
            report(tr("The test data could not be downloaded: {detail}",
                      detail=error or tr("unknown error")), True)
            return
        if not is_present(folder):
            report(tr("The downloaded test data is incomplete: {path}",
                      path=str(folder)), True)
            return
        use(folder)

    download = ask if ask is not None else _download(which)
    try:
        download(parent, folder, _done)
    except Exception as exc:
        LOG.warning("the %s test data download did not start", which,
                    exc_info=True)
        _done(None, explain_download_failure(exc))
    return False


def install_align_button(screen) -> QPushButton:
    """Build Align & Stitch's "Load test data…" button, wired to ``screen``.

    :param screen: the :class:`spacr.qt.screens.align.AlignScreen`. The button
        is kept on it as ``_btn_test_data``.
    :returns: the button, for the caller to place.
    """
    button = QPushButton(tr("Load test data…"), screen)
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(tr(
        "Download nine overlapping tiles from a published optical pooled "
        "screen (about 200 MB) and fill in the tile folder, grid, order and "
        "overlap, so Plan can be pressed straight away. Cached after the "
        "first download."))
    button.clicked.connect(lambda _checked=False: load_align_test_data(screen))
    screen._btn_test_data = button
    return button


def load_align_test_data(screen, *, ask=None, folder=None) -> bool:
    """Fill Align & Stitch with the stitch sample, fetching it when needed.

    :param screen: the Align & Stitch screen.
    :param ask: replaces the downloader, for tests.
    :param folder: replaces :func:`example_folder`, for tests.
    :returns: see :func:`load_the_test_data`.
    """
    def use(where: Path) -> None:
        """Apply the sample's settings and say what to press next."""
        screen.apply_settings(stitch_settings_for(where))
        screen._set_status(tr(
            "Test data ready: nine tiles in {path}. Press Plan.",
            path=str(Path(where) / "tiles")))

    def report(text: str, error: bool) -> None:
        """Show progress and failures on the screen's status line."""
        screen._set_status(text, error=error)

    return load_the_test_data(
        "stitch", use=use, report=report,
        button=getattr(screen, "_btn_test_data", None), parent=screen,
        ask=ask, folder=folder)
