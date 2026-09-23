"""Test data for the assay modules: a measured plate, one button away.

Replication, Invasion and Recruitment get test data: small datasets on
Hugging Face and a "Load test data…" button in each module. Replication and
Recruitment are slices of real screens. Invasion's is SYNTHETIC: no two-colour
differential-staining acquisition exists, so its fields and their object
masks are drawn by spaCR and then measured by the Measure module. Its tooltip
says so.

WHAT THE MODULES TAKE. All three read ``<src>/measurements/measurements.db``,
the output of Mask then Measure, so each example is a slice of a real screen's
database -- every row of twelve control wells -- with two merged fields and
the module's settings. :data:`spacr.example_archives.EXAMPLE_SETS` describes
both, so ``spacr-download replication`` fetches the same archive.

ITS OWN FOLDER PER SET. Both ship ``measurements/measurements.db``, as the
Annotate example in the shared plate does, so each unpacks beside that plate
(:func:`spacr.example_archives.example_set_folder`) rather than into it.

THE SAME PATTERN AS THE OTHER EXAMPLES. One uncompressed ``.tar`` streamed to
a ``.part`` file on a worker thread behind the shared progress dialog,
unpacked through the tar data filter, ``<dataset>`` in the settings replaced
with where it landed, and cached: a second press uses the copy on disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from ..example_archives import example_set, example_set_folder
from .hf_download import (DownloadResult, _TarExampleWorker,
                          download_toxo_mito_demo, explain_download_failure)
from .i18n import tr

LOG = logging.getLogger("spacr.qt.assay_examples")

__all__ = [
    "ASSAY_EXAMPLE_KEYS",
    "download_assay_example",
    "install_assay_example_button",
    "load_the_assay_example",
    "put_the_assay_example_in_place",
]

#: The assay modules that have published test data.
ASSAY_EXAMPLE_KEYS = ("replication", "recruitment", "invasion")


class _HostPathogenExampleWorker(_TarExampleWorker):
    """Prepare the small synthetic project offline on the existing worker thread."""

    def run(self):
        """Build atomically and report completion through the shared dialog contract."""
        from ..host_pathogen_example import build_example

        try:
            self.info.emit(tr('Preparing synthetic Host–Pathogen test data…'))
            folder = build_example(self._dest, cancelled=lambda: self._cancel,
                progress=lambda done, total: self.progress.emit(tr('Synthetic fields'), done, total))
            self.finished.emit(True, str(folder), str(folder / 'settings'), '')
        except Exception as exc:
            self.finished.emit(False, '', '', str(exc))


class _AssayTarWorker(_TarExampleWorker):
    """Fetches one assay example's archive. Nothing to do after extraction."""

    def __init__(self, dest_dir, repo: str):
        """Prepare the worker.

        :param dest_dir: the folder the archive unpacks into.
        :param repo: the dataset repository that publishes it.
        """
        super().__init__(dest_dir)
        self.repo = repo


def _title(app_key: str) -> str:
    """The progress dialog's title for ``app_key``."""
    if app_key == "replication":
        return tr("Downloading the Replication Assay test data")
    if app_key == "invasion":
        return tr("Downloading the synthetic Invasion Assay test data")
    if app_key == 'host_pathogen':
        return tr('Preparing synthetic Host–Pathogen test data')
    return tr("Downloading the Recruitment test data")


def _tooltip(app_key: str) -> str:
    """What the button says it will fetch, size first."""
    if app_key == 'host_pathogen':
        return tr('Generate about 13 MB of SYNTHETIC test data offline: four fields, '
                  '24 hosts, 28 vacuoles and individually labelled parasites. '
                  'Includes uninfected hosts, multiple vacuoles, unknown marker '
                  'states and expected results. These are drawn test images, '
                  'not biological validation. Settings are filled in; Run is next.')
    if app_key == "replication":
        return tr(
            "Download about 170 MB of test data: every parasite of twelve "
            "control wells of the Toxoplasma MTOC screen, measured, with two "
            "fields to look at. Settings are filled in, so Run is the next "
            "step. Cached after the first download.")
    if app_key == "invasion":
        return tr(
            "Download about {size} MB of SYNTHETIC test data: two-colour "
            "fields and object masks drawn by spaCR, not imaged or "
            "segmented, then measured by Measure. A staining-control column "
            "and two conditions with a known share of invaded parasites, "
            "with the "
            "truth beside them. Settings are filled in, so Run is the next "
            "step. Cached after the first download.",
            size=round(example_set("invasion").bytes / 1e6))
    return tr(
        "Download about 150 MB of test data: twelve control wells of the "
        "THP-1 RNF213 screen, measured, with two fields to look at and the "
        "settings the screen was analysed with. Run is the next step. Cached "
        "after the first download.")


def download_assay_example(parent, app_key: str, dest, on_done: Callable[
        [Optional[DownloadResult], str], None]) -> None:
    """Fetch ``app_key``'s test data behind the shared progress dialog.

    :param parent: the widget the progress dialog belongs to.
    :param app_key: ``replication``, ``recruitment`` or ``invasion``.
    :param dest: the folder the archive unpacks into.
    :param on_done: called on the GUI thread as ``on_done(result, error)``;
        ``result`` is ``None`` on failure or cancellation.
    """
    if app_key == 'host_pathogen':
        download_toxo_mito_demo(parent, Path(dest), on_done,
                                worker_factory=_HostPathogenExampleWorker,
                                title=_title(app_key))
        return
    repo = example_set(app_key).repo

    def _factory(where):
        """Build the worker for this set's repository."""
        return _AssayTarWorker(where, repo)

    download_toxo_mito_demo(parent, Path(dest), on_done,
                            worker_factory=_factory, title=_title(app_key))


def install_assay_example_button(screen, section) -> QPushButton:
    """Put the "Load test data…" button at the top of ``section``.

    :param screen: the module screen; its ``app_key`` picks the set. The
        button is kept on it as ``_assay_example_button``.
    :param section: the settings section the button goes in.
    :returns: the button.
    """
    button = QPushButton(tr("Load test data…"))
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(_tooltip(screen.app_key))
    button.clicked.connect(
        lambda _checked=False: load_the_assay_example(screen))
    screen._assay_example_button = button
    section.add_prose(button, at_top=True)
    return button


def load_the_assay_example(screen, *, ask=None, folder=None) -> Dict:
    """Fill the module in with its test data, downloading it first if needed.

    A complete cached copy is used at once. Otherwise the button is disabled
    while the download runs on a worker thread, and the settings are applied
    when it finishes.

    :param screen: the module screen.
    :param ask: replaces :func:`download_assay_example`, for tests; called as
        ``ask(screen, folder, on_done)``.
    :param folder: replaces the set's cache folder, for tests.
    :returns: ``{"src": folder}`` once the data is in place; empty while a
        download is still running, and after a failure.
    """
    key = screen.app_key
    if key == 'host_pathogen':
        from .. import host_pathogen_example as chosen

        folder = Path(folder) if folder is not None else chosen.example_folder()
    else:
        chosen = example_set(key)
        folder = Path(folder) if folder is not None else example_set_folder(key)
    if chosen.is_present(folder):
        return put_the_assay_example_in_place(screen, folder)

    button = getattr(screen, "_assay_example_button", None)
    if button is not None:
        button.setEnabled(False)
        button.setText(tr('Preparing test data…') if key == 'host_pathogen'
                       else tr("Fetching test data…"))

    placed: Dict = {}

    def _done(result, error):
        """Put the button back, then fill the settings in or say why not."""
        if button is not None:
            button.setEnabled(True)
            button.setText(tr("Load test data…"))
        if result is None:
            LOG.info("%s test data not downloaded: %s", key, error)
            screen._console.append_notice(
                "[example] the test data was not downloaded: {detail}\n",
                detail=error or "cancelled")
            return
        placed.update(put_the_assay_example_in_place(screen, folder))

    if ask is None:
        def ask(parent, dest, on_done):
            """Start this module's download."""
            download_assay_example(parent, key, dest, on_done)
    try:
        ask(screen, folder, _done)
    except Exception as exc:                                 # noqa: BLE001
        LOG.warning("the %s test data download did not start", key,
                    exc_info=True)
        _done(None, explain_download_failure(exc))
    return placed


def put_the_assay_example_in_place(screen, folder) -> Dict:
    """Apply the settings that shipped with the data and point ``src`` at it.

    :param screen: the module screen.
    :param folder: where the set was unpacked.
    :returns: ``{"src": ...}``, the value the field ends up holding.
    """
    return screen._apply_the_example_settings(Path(folder))
