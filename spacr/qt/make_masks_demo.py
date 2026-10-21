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

from ..example_archives import (EXAMPLE_ARCHIVES, FIBSEM_EXAMPLE_REPO,
                                TEM_EXAMPLE_REPO,
                                TIMELAPSE_BACTERIA_EXAMPLE_REPO,
                                TIMELAPSE_MAMMALIAN_EXAMPLE_REPO,
                                TIMELAPSE_YEAST_EXAMPLE_REPO,
                                example_set_folder)
from .hf_download import (
    DownloadResult,
    _TarExampleWorker,
    download_toxo_mito_demo,
    explain_download_failure,
)
from .i18n import tr

LOG = logging.getLogger("spacr.qt.make_masks_demo")

__all__ = [
    "MAKE_MASKS_EXAMPLE_ARCHIVE",
    "MAKE_MASKS_EXAMPLE_REPO",
    "TEM_EXAMPLE_ARCHIVE",
    "TEM_EXAMPLE_REPO",
    "FIBSEM_EXAMPLE_ARCHIVE",
    "FIBSEM_EXAMPLE_REPO",
    "download_fibsem_example",
    "download_make_masks_example",
    "fibsem_example_folder",
    "load_the_fibsem_test_data",
    "download_tem_example",
    "install_test_data_button",
    "is_present",
    "listed_files",
    "load_the_tem_test_data",
    "load_the_test_data",
    "make_masks_example_folder",
    "open_the_test_data",
    "tem_example_folder",
    "TIMELAPSE_EXAMPLE_REPOS",
    "download_timelapse_example",
    "load_the_timelapse_test_data",
    "timelapse_example_folder",
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

#: The archive of the TEM test data: ten transmission electron micrographs
#: with mitochondria masks (:data:`spacr.example_archives.TEM_EXAMPLE_REPO`).
TEM_EXAMPLE_ARCHIVE = EXAMPLE_ARCHIVES[TEM_EXAMPLE_REPO]

FIBSEM_EXAMPLE_ARCHIVE = EXAMPLE_ARCHIVES[FIBSEM_EXAMPLE_REPO]

FIBSEM_VOLUME = "hela_fibsem.tif"

#: The three timelapse test movies, by kind: one public movie each of budding
#: yeast, bacteria and mammalian cells, with masks in every frame and their
#: lineage embedded in the mask TIFFs. Each kind is also the
#: ``spacr-download timelapse_<kind>`` example set.
TIMELAPSE_EXAMPLE_REPOS = {
    "yeast": TIMELAPSE_YEAST_EXAMPLE_REPO,
    "bacteria": TIMELAPSE_BACTERIA_EXAMPLE_REPO,
    "mammalian": TIMELAPSE_MAMMALIAN_EXAMPLE_REPO,
}

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


def tem_example_folder() -> Path:
    """Where the TEM test data is unpacked and cached.

    ``~/.cache/spacr/example_data/make_masks_tem``, the folder
    :data:`spacr.example_archives.EXAMPLE_SETS` names for the ``tem`` set, so
    ``spacr-download tem`` and the button share one cache.

    :returns: the folder. It is not created here.
    """
    return (Path.home() / ".cache" / "spacr" / "example_data"
            / "make_masks_tem")


def fibsem_example_folder() -> Path:
    """Where the FIB-SEM test volume is unpacked and cached.

    ``~/.cache/spacr/example_data/make_masks_fibsem``, the folder
    :data:`spacr.example_archives.EXAMPLE_SETS` names for the ``fibsem`` set,
    so ``spacr-download fibsem`` and the menu share one cache.

    :returns: the folder. It is not created here.
    """
    return (Path.home() / ".cache" / "spacr" / "example_data"
            / "make_masks_fibsem")


def timelapse_example_folder(kind: str) -> Path:
    """Where one timelapse test movie is unpacked and cached.

    ``~/.cache/spacr/example_data/make_masks_timelapse_<kind>``, the folder
    :data:`spacr.example_archives.EXAMPLE_SETS` names for the
    ``timelapse_<kind>`` set, so ``spacr-download`` and the menu share one
    cache.

    :param kind: ``yeast``, ``bacteria`` or ``mammalian``.
    :returns: the folder. It is not created here.
    :raises KeyError: for any other kind.
    """
    if kind not in TIMELAPSE_EXAMPLE_REPOS:
        raise KeyError(kind)
    return example_set_folder(f"timelapse_{kind}")


def listed_files(folder) -> List[str]:
    """Every file the dataset's ``manifest.csv`` says a complete copy holds.

    :param folder: where the dataset was unpacked.
    :returns: paths relative to ``folder``, each image followed by its ground-truth mask when the row names one, in manifest order. Empty when there is no readable manifest or any row names no image. This function lists manifest paths without checking whether those files exist.
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


class _TemTarWorker(_TarExampleWorker):
    """Fetches the TEM test data: ten micrographs and their mitochondria."""

    repo = TEM_EXAMPLE_REPO
    archive = TEM_EXAMPLE_ARCHIVE


class _FibsemTarWorker(_TarExampleWorker):
    """Fetches the FIB-SEM test volume and its mitochondria labels."""

    repo = FIBSEM_EXAMPLE_REPO
    archive = FIBSEM_EXAMPLE_ARCHIVE


def download_fibsem_example(parent, dest, on_done: Callable[
        [Optional[DownloadResult], str], None]) -> None:
    """Fetch the FIB-SEM test volume behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param dest: the folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``,
        as :func:`download_make_masks_example` calls it.
    """
    download_toxo_mito_demo(
        parent, Path(dest), on_done,
        worker_factory=_FibsemTarWorker,
        title=tr("Downloading the Make Masks FIB-SEM test data"))


def download_timelapse_example(parent, dest, on_done: Callable[
        [Optional[DownloadResult], str], None], kind: str) -> None:
    """Fetch one timelapse test movie behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param dest: the folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``,
        as :func:`download_make_masks_example` calls it.
    :param kind: ``yeast``, ``bacteria`` or ``mammalian``.
    """
    repo = TIMELAPSE_EXAMPLE_REPOS[kind]
    worker = type("_TimelapseTarWorker", (_TarExampleWorker,),
                  {"repo": repo, "archive": EXAMPLE_ARCHIVES[repo]})
    download_toxo_mito_demo(
        parent, Path(dest), on_done, worker_factory=worker,
        title=tr("Downloading the Make Masks timelapse test data"))


def download_tem_example(parent, dest, on_done: Callable[
        [Optional[DownloadResult], str], None]) -> None:
    """Fetch the TEM test data behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param dest: the folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``,
        as :func:`download_make_masks_example` calls it.
    """
    download_toxo_mito_demo(
        parent, Path(dest), on_done,
        worker_factory=_TemTarWorker,
        title=tr("Downloading the Make Masks TEM test data"))


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
    return _load_example(screen, folder, ask if ask is not None
                         else download_make_masks_example)


def load_the_tem_test_data(screen, *, ask=None, folder=None) -> bool:
    """Open the TEM test data in Make Masks, downloading it first when needed.

    The same sequence as :func:`load_the_test_data` -- cache check, button
    disabled while the download runs, the first image by name, a failure
    said with its reason -- for the ten micrographs of
    :data:`TEM_EXAMPLE_REPO`, cached in :func:`tem_example_folder`.

    :param screen: the Make Masks screen to open the data in.
    :param ask: replaces :func:`download_tem_example`, for tests.
    :param folder: replaces :func:`tem_example_folder`, for tests.
    :returns: whether this call opened the data.
    """
    folder = Path(folder) if folder is not None else tem_example_folder()
    return _load_example(screen, folder, ask if ask is not None
                         else download_tem_example)


def load_the_fibsem_test_data(screen, *, with_masks=False, ask=None,
                              folder=None) -> bool:
    """Open the FIB-SEM test volume in the 3-D mask editor.

    The same sequence as :func:`load_the_test_data`, downloading
    :data:`FIBSEM_EXAMPLE_REPO` into :func:`fibsem_example_folder` when it is
    not cached. Without masks the volume opens unsegmented, its drafts saved
    in ``masks/``. With masks, the published labels are copied once into
    ``corrected_masks/`` and open there for correction, so
    ``ground_truth_masks/`` is never overwritten.

    :param screen: the Make Masks screen whose 3-D editor opens the volume.
    :param with_masks: open the volume with its mitochondria labels loaded.
    :param ask: replaces :func:`download_fibsem_example`, for tests.
    :param folder: replaces :func:`fibsem_example_folder`, for tests.
    :returns: whether this call opened the volume.
    """
    folder = Path(folder) if folder is not None else fibsem_example_folder()
    return _load_example(screen, folder, ask if ask is not None
                         else download_fibsem_example,
                         opener=lambda target, chosen: _open_fibsem(
                             target, chosen, with_masks))


def load_the_timelapse_test_data(screen, kind: str, *, ask=None,
                                 folder=None) -> bool:
    """Open a timelapse test movie with its masks and lineage loaded.

    The same sequence as :func:`load_the_test_data`, downloading
    ``timelapse_<kind>`` into :func:`timelapse_example_folder` when it is not
    cached. The published masks are copied once into ``masks/``, the folder
    the editor reads and writes, so ``ground_truth_masks/`` is never
    overwritten. Their embedded lineage opens with the first frame, so the
    divisions are on the lineage menu and Time can be switched on at once.

    :param screen: the Make Masks screen to open the movie in.
    :param kind: ``yeast``, ``bacteria`` or ``mammalian``.
    :param ask: replaces :func:`download_timelapse_example`, for tests. It is
        called as ``ask(screen, folder, on_done)``.
    :param folder: replaces :func:`timelapse_example_folder`, for tests.
    :returns: whether this call opened the movie.
    """
    folder = (Path(folder) if folder is not None
              else timelapse_example_folder(kind))
    download = ask if ask is not None else (
        lambda parent, dest, on_done: download_timelapse_example(
            parent, dest, on_done, kind))
    return _load_example(screen, folder, download, opener=_open_timelapse)


def _open_timelapse(screen, folder) -> bool:
    """Open an unpacked timelapse movie on its first frame, masks loaded."""
    import shutil

    folder = Path(folder)
    if not is_present(folder):
        screen._warn(
            tr("Test data not loaded"),
            tr("The downloaded test data is incomplete: {path}",
               path=str(folder)))
        return False
    masks = folder / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    for truth in sorted((folder / "ground_truth_masks").glob("*.tif")):
        if not (masks / truth.name).exists():
            shutil.copyfile(truth, masks / truth.name)
    if not screen._open_folder(str(folder), masks_dir=str(masks)):
        return False
    _say(screen, tr("Timelapse test data opened with its lineage. Turn on Time "
                    "to follow cells and divisions across frames."))
    return True


def _open_fibsem(screen, folder, with_masks: bool) -> bool:
    """Open the unpacked FIB-SEM volume, with a working copy of its labels."""
    import shutil

    folder = Path(folder)
    if not is_present(folder):
        screen._warn(
            tr("Test data not loaded"),
            tr("The downloaded test data is incomplete: {path}",
               path=str(folder)))
        return False
    masks = folder / ("corrected_masks" if with_masks else "masks")
    if with_masks and not (masks / FIBSEM_VOLUME).exists():
        masks.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "ground_truth_masks" / FIBSEM_VOLUME,
                        masks / FIBSEM_VOLUME)
    return screen._open_volume_with(str(folder / FIBSEM_VOLUME),
                                    masks_dir=str(masks)) is not None


def _load_example(screen, folder: Path, download, opener=None) -> bool:
    """Open ``folder`` when complete, else download it with ``download``.

    :param screen: the Make Masks screen.
    :param folder: where the set is cached.
    :param download: called as ``download(screen, folder, on_done)``.
    :param opener: called as ``opener(screen, folder)`` to open the set;
        :func:`open_the_test_data` when omitted.
    :returns: whether this call opened the data.
    """
    opener = opener or open_the_test_data
    if is_present(folder):
        return opener(screen, folder)

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
        opener(screen, folder)

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
