"""Make Masks' test data: ten Toxoplasma vacuole fields, one button away.

Ten images of the Toxoplasma PV training set are published on Hugging Face as
a small test dataset for Make Masks, with a button that loads the dataset and
opens its first image in Make Masks.

WHAT IS PUBLISHED. :data:`MAKE_MASKS_EXAMPLE_REPO` holds ten of the lab's own
acquisitions -- no figure harvested from a paper, no held-out field -- at the
top level, their curated label masks under ``ground_truth_masks/``, a
``manifest.csv`` carrying each image's row from the in-house manifest, and one
archive of all of it. The images open UNSEGMENTED: Make Masks reads drafts from
``masks/``, never from ``ground_truth_masks/``, so the editor starts on raw
fields and the truth sits beside them.

THE SAME PATTERN AS EVERY OTHER EXAMPLE SET. One uncompressed ``.tar`` per
repository, streamed to a ``.part`` file on a worker thread behind the shared
progress dialog, unpacked through the tar data filter
(:class:`spacr.qt.hf_download._TarExampleWorker` does all of that), and cached:
:func:`is_present` is asked first, so a second press opens the folder without a
request.

ITS OWN FOLDER, NOT THE SHARED PLATE. The Measure and Annotate sets unpack
into ``example_plate_folder()`` because they compose into one plate. These ten
fields are not part of that plate, and the Mask demo decides it is present by
finding ``*.tif`` there, so unpacking them into it would make that demo think
it had been downloaded.

WHY THIS IS A MODULE AND NOT A METHOD. The screen gets one call in its
navigation row. Everything else lives here, so ``make_masks.py`` -- which
several items edit at once -- carries a two-line hook.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from .hf_download import (DownloadResult, _TarExampleWorker,
                          download_toxo_mito_demo, explain_download_failure)
from .i18n import tr

LOG = logging.getLogger("spacr.qt.make_masks_demo")

__all__ = [
    "MAKE_MASKS_EXAMPLE_ARCHIVE",
    "MAKE_MASKS_EXAMPLE_REPO",
    "download_make_masks_example",
    "install_test_data_button",
    "is_present",
    "listed_files",
    "load_the_test_data",
    "make_masks_example_folder",
    "open_the_test_data",
]

#: The Hugging Face dataset repository the ten fields are published in.
#:
#: A SIBLING OF THE OTHER EXAMPLE REPOS rather than a folder inside one of
#: them. Each example set is a separate artefact with its own card and its own
#: archive, the same way ``spacr-example-measure`` and
#: ``spacr-example-annotate`` are.
MAKE_MASKS_EXAMPLE_REPO = "einarolafsson/spacr-example-make-masks"

#: The one archive that repository ships, and the one file the button fetches.
MAKE_MASKS_EXAMPLE_ARCHIVE = "spacr-example-make-masks.tar"

#: The file in the dataset that lists what a complete copy contains. It is the
#: LAST member of the archive, so a transfer that died part-way leaves no
#: manifest and reads as absent rather than as a complete set.
MANIFEST_NAME = "manifest.csv"

#: What the worker reports when the user pressed Cancel. Told apart from a
#: failure, because a cancel the user asked for needs no warning.
CANCELLED = "Cancelled by user."


def make_masks_example_folder() -> Path:
    """Where the test data is unpacked and cached.

    ``~/.cache/spacr/example_data/make_masks_toxo_pv``, beside the shared
    example plate rather than inside it. See the module docstring for why.

    :returns: the folder. It is not created here.
    """
    return (Path.home() / ".cache" / "spacr" / "example_data"
            / "make_masks_toxo_pv")


def listed_files(folder) -> List[str]:
    """Every file the dataset's ``manifest.csv`` says a complete copy holds.

    :param folder: where the dataset was unpacked.
    :returns: paths relative to ``folder``, each image followed by its
        ground-truth mask, in manifest order. Empty when there is no readable
        manifest, or when any row names no image, so that an incomplete or
        unrecognised folder is never mistaken for the dataset.
    """
    manifest = Path(folder) / MANIFEST_NAME
    try:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []
    files: List[str] = []
    for row in rows:
        image = (row.get("image") or "").strip()
        if not image:
            return []
        files.append(image)
        truth = (row.get("ground_truth_mask") or "").strip()
        if truth:
            files.append(truth)
    return files


def is_present(folder) -> bool:
    """Whether a complete copy of the test data is unpacked in ``folder``.

    :param folder: where the dataset is expected.
    :returns: ``True`` only when the manifest exists and every image and mask
        it lists is a file. A half-unpacked set reads as absent, which is the
        answer that gets it downloaded again.
    """
    folder = Path(folder)
    files = listed_files(folder)
    return bool(files) and all((folder / name).is_file() for name in files)


class _MakeMasksTarWorker(_TarExampleWorker):
    """Fetches the Make Masks test data: ten fields and their ground truth.

    Nothing to do after extraction. The archive already is the folder Make
    Masks opens, so this is the repository, its archive and the shared
    machinery.
    """

    repo = MAKE_MASKS_EXAMPLE_REPO
    archive = MAKE_MASKS_EXAMPLE_ARCHIVE


def download_make_masks_example(parent, dest, on_done: Callable[
        [Optional[DownloadResult], str], None]) -> None:
    """Fetch the Make Masks test data behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param dest: the folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``.
        ``result`` is ``None`` on failure or cancellation, and ``error`` then
        says why.
    """
    download_toxo_mito_demo(
        parent, Path(dest), on_done,
        worker_factory=_MakeMasksTarWorker,
        title=tr("Downloading the Make Masks test data"))


def install_test_data_button(screen) -> QPushButton:
    """Build Make Masks' "Load test data…" button, wired to ``screen``.

    :param screen: the Make Masks screen the data opens in. The button is kept
        on it as ``_btn_test_data``, so a load can disable it while the
        download runs.
    :returns: the button, for the caller to place.
    """
    button = QPushButton(tr("Load test data…"), screen)
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(tr(
        "Download ten example images of Toxoplasma vacuoles (about 40 MB) and "
        "open the first one. Their curated masks come too, in the "
        "ground_truth_masks folder, which the editor does not read. Cached "
        "after the first download."))
    button.clicked.connect(lambda _checked=False: load_the_test_data(screen))
    screen._btn_test_data = button
    return button


def _say(screen, text: str) -> None:
    """Put ``text`` on the screen's status line, if it has one."""
    label = getattr(screen, "_status_label", None)
    if label is not None:
        label.setText(text)


def load_the_test_data(screen, *, ask=None, folder=None) -> bool:
    """Open the test data in Make Masks, downloading it first when needed.

    A cached copy opens at once. Otherwise the button is disabled, the status
    line says a download is running, and the download runs on a worker thread
    behind a progress dialog. When it finishes, the folder opens on its first
    image by name. A failure is reported with the reason the downloader gave,
    on the status line and, when there is a display to show one, in a warning.

    :param screen: the Make Masks screen to open the data in.
    :param ask: replaces :func:`download_make_masks_example`, for tests. It is
        called as ``ask(screen, folder, on_done)``.
    :param folder: replaces :func:`make_masks_example_folder`, for tests.
    :returns: whether this call opened the data. ``False`` while a download is
        still running, and after a failure.
    """
    folder = (Path(folder) if folder is not None
              else make_masks_example_folder())
    if is_present(folder):
        return open_the_test_data(screen, folder)

    button = getattr(screen, "_btn_test_data", None)
    if button is not None:
        button.setEnabled(False)
        button.setText(tr("Fetching test data…"))
    _say(screen, tr("Downloading the Make Masks test data"))

    def _done(result, error):
        """Put the button back, then open the data or say why it did not."""
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Load test data…"))
        if result is None:
            if error == CANCELLED:
                _say(screen, tr("The test data download was cancelled."))
                return
            LOG.info("Make Masks test data not downloaded: %s", error)
            screen._warn(
                tr("Test data not loaded"),
                tr("The test data could not be downloaded: {detail}",
                   detail=error))
            return
        open_the_test_data(screen, folder)

    download = ask if ask is not None else download_make_masks_example
    try:
        download(screen, folder, _done)
    except Exception as exc:                                 # noqa: BLE001
        LOG.warning("the Make Masks test data download did not start",
                    exc_info=True)
        _done(None, explain_download_failure(exc))
    return False


def open_the_test_data(screen, folder) -> bool:
    """Open an unpacked copy of the test data on its first image by name.

    :param screen: the Make Masks screen.
    :param folder: where the data was unpacked.
    :returns: whether the screen is now on it. An incomplete copy is reported
        rather than opened, because half the fields would pass for the whole
        set.
    """
    folder = Path(folder)
    if not is_present(folder):
        screen._warn(
            tr("Test data not loaded"),
            tr("The downloaded test data is incomplete: {path}",
               path=str(folder)))
        return False
    return bool(screen._open_folder(str(folder)))
