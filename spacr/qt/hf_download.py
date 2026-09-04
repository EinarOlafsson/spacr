"""
Hugging Face dataset downloader with a Qt progress dialog.

Wraps the two demo repositories (``einarolafsson/toxo_mito`` for images
and ``einarolafsson/spacr_settings`` for the accompanying settings
pack) into a single ``download_toxo_mito_demo(parent, dest)`` call
that:

1. Pops a modal :class:`QProgressDialog` with per-file granularity.
2. Runs the downloads in a QThread so the UI stays responsive.
3. Reports the resulting local paths back via a completion callback.

The classic Tk downloader in :mod:`spacr.gui_utils` uses a queue-
based background thread. This module reimplements the same behaviour
using Qt's threading + signals so the Qt GUI doesn't need to spin up
a Tk mainloop just to see download progress.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel,
                               QProgressBar, QProgressDialog,
                               QPushButton, QVBoxLayout, QWidget)

# THE DATA HALF OF THIS MODULE NOW LIVES IN `spacr.example_archives`, and is
# imported back here so nothing that already calls one of these names has to
# change -- including the tests that patch `hf_download._download_one` and
# `hf_download._list_files` to keep the demo flow off the network. Those still
# name real attributes of this module, and they are still what the workers
# below resolve.
#
# It moved because `spacr-download` fetches the same datasets from a cluster
# login node, and importing this module to reach them would demand PySide6 on
# a machine with no display to give it. What is left here is Qt: the dialog,
# the threads, the signals. What left was only ever about the data.
from ..example_archives import (                              # noqa: F401
    ANNOTATE_EXAMPLE_REPO, DATASET_PLACEHOLDER, DATASET_REPO, DATASET_SUB,
    EXAMPLE_ARCHIVES, MEASURE_EXAMPLE_REPO, SETTINGS_REPO, _content_length,
    _download_one, _list_files, example_plate_folder, expand_measure_arrays,
    explain_download_failure, extract_example_archive,
    make_the_example_paths_absolute)

LOG = logging.getLogger("spacr.qt.hf_download")

#: The longest caption the progress dialog shows, used to size it once at
#: construction. Not a guess: the download reports one HuggingFace file at
#: a time and this is a real name from the toxo_mito pack, which is the
#: dataset the "load example data" button fetches.
_WIDEST_CAPTION = "Downloading plate1_A01_T0001F001L01A01Z01C01.tif"

#: Room for the dialog's frame, its margins and the progress bar's own
#: padding, on top of the caption itself.
_CAPTION_MARGIN = 96


@dataclass
class DownloadResult:
    """Outcome of one :func:`download_toxo_mito_demo` call."""
    dataset_path:  Path
    settings_path: Path


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _HFDownloadWorker(QObject):
    """Background worker that fetches both repos, emitting granular
    progress signals along the way.

    Signals:

    * ``progress(str, int, int)`` — (file_name, done_files, total_files)
    * ``info(str)`` — status message for the dialog label
    * ``finished(bool, str, str, str)`` — (ok, dataset_path, settings_path, error)
    """

    progress = Signal(str, int, int)
    info     = Signal(str)
    finished = Signal(bool, str, str, str)

    def __init__(self, dest_dir: Path):
        """Prepare the worker.

        :param dest_dir: where the download is written. Read on the worker
            thread, not in the constructor -- so a caller may hand over a
            folder that does not exist yet, and a failure to create it is
            reported through ``finished`` like every other failure rather
            than raised into the caller's event handler.
        """
        super().__init__()
        self._dest = Path(dest_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.info.emit("Listing files on Hugging Face…")
            dataset_files = _list_files(DATASET_REPO, DATASET_SUB)
            settings_files = _list_files(SETTINGS_REPO, "")

            total = len(dataset_files) + len(settings_files)
            if total == 0:
                self.finished.emit(False, "", "",
                                     "No files to download from the "
                                     "Hugging Face repositories.")
                return
            self.info.emit(f"Found {total} files to download.")

            dataset_root  = self._dest / "plate1"
            settings_root = self._dest / "settings"
            dataset_root.mkdir(parents=True, exist_ok=True)
            settings_root.mkdir(parents=True, exist_ok=True)

            done = 0
            for name in dataset_files:
                if self._cancel:
                    self.finished.emit(False, "", "", "Cancelled by user.")
                    return
                self.progress.emit(name, done, total)
                _download_one(DATASET_REPO, name, dataset_root)
                done += 1

            for name in settings_files:
                if self._cancel:
                    self.finished.emit(False, "", "", "Cancelled by user.")
                    return
                self.progress.emit(name, done, total)
                _download_one(SETTINGS_REPO, name, settings_root)
                done += 1

            self.progress.emit("done", total, total)
            self.finished.emit(True, str(dataset_root),
                                 str(settings_root), "")
        except Exception as e:
            LOG.warning("hf download failed: %s", e, exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))


# ---------------------------------------------------------------------------
# GUI-thread receiver
# ---------------------------------------------------------------------------

class _DownloadDialog(QDialog):
    """The download window: bar on top, status centred, Cancel beside it.

    A QProgressDialog was used here and its layout is not arrangeable: it puts
    the label ABOVE the bar and sizes the window from whatever caption it was
    constructed with. That caption is "Preparing…" and the window then spends
    the download showing "Downloading <filename> (3/6 files)" and a
    percentage -- so the text was clipped at the window edge, twice reported
    as "the % text is cut off". Widening it for the longest expected caption
    helped and did not fix it, because the longest caption is a FILE NAME and
    there is no longest file name.

    So the text WRAPS and is centred in the window, with the bar above it and
    Cancel to its right:

        [============ blue bar ============]
        [ spacer ][  centred status  ][Cancel]

    The left spacer is the width of the button, which is what makes the label
    centre on the WINDOW rather than on the space left over beside the button.

    Presents the parts of QProgressDialog's API the download flow uses, so the
    worker wiring did not have to change with it.
    """

    canceled = Signal()

    def __init__(self, title: str, parent=None):
        """Build the progress dialog.

        :param title: the window title -- the only thing distinguishing one
            of these dialogs from another, since the body is written by
            whichever worker is driving it.
        :param parent: parent widget; ownership only.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(520)
        outer = QVBoxLayout(self)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)      # the caption below says it all
        outer.addWidget(self._bar)

        row = QHBoxLayout()
        self._cancel = QPushButton("Cancel", self)
        self._cancel.clicked.connect(self._on_cancel)

        # The spacer matches the button, so the caption is centred on the
        # window and not on the gap beside the button.
        spacer = QWidget(self)
        spacer.setFixedWidth(self._cancel.sizeHint().width())
        row.addWidget(spacer)

        self.spacr_caption = QLabel("Preparing…", self)
        self.spacr_caption.setWordWrap(True)
        self.spacr_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.spacr_caption, 1)

        row.addWidget(self._cancel)
        outer.addLayout(row)

        self._auto_close = True
        self._cancelled = False

    # -- the QProgressDialog surface the download flow uses -----------------

    def _on_cancel(self) -> None:
        """Mark the download cancelled and tell the worker.

        The flag is set as well as the signal emitted: the worker checks it
        between chunks, and a signal alone would be missed by a worker that is
        mid-chunk when the button is pressed.
        """
        self._cancelled = True
        self.canceled.emit()

    def wasCanceled(self) -> bool:               # noqa: N802 (Qt naming)
        return self._cancelled

    def setLabelText(self, text: str) -> None:   # noqa: N802
        self.spacr_caption.setText(str(text))

    def setLabel(self, label) -> None:           # noqa: N802
        """Accepted for compatibility; this dialog owns its own label.

        Swapping the label out would drop the wrapping and the centring that
        are the whole point of this class, so the text is taken and the
        widget is not.
        """
        try:
            self.spacr_caption.setText(label.text())
        except Exception:                                    # noqa: BLE001
            pass

    def setMaximum(self, value: int) -> None:    # noqa: N802
        self._bar.setMaximum(max(1, int(value)))

    def setValue(self, value: int) -> None:      # noqa: N802
        self._bar.setValue(int(value))
        if self._auto_close and self._bar.maximum() and \
                int(value) >= self._bar.maximum():
            self.close()

    def maximum(self) -> int:
        return self._bar.maximum()

    def setAutoClose(self, on: bool) -> None:    # noqa: N802
        self._auto_close = bool(on)

    def setAutoReset(self, on: bool) -> None:    # noqa: N802
        """Accepted for compatibility. The bar is not reused after a run."""

    def setMinimumDuration(self, ms: int) -> None:   # noqa: N802
        """Accepted for compatibility. This dialog is shown when it is made."""

    def reset(self) -> None:
        self._bar.reset()


class _HFDownloadUI(QObject):
    """Receives the worker's signals **on the GUI thread**.

    This class exists purely for thread affinity. The worker is moved
    into a QThread, and Qt picks the connection type from the receiving
    *QObject's* thread — but a plain Python function is not a QObject,
    so Qt has no context to compare against and falls back to a DIRECT
    connection. Wiring the worker straight to closures therefore ran
    every handler inside the worker thread, where they drove a
    ``QProgressDialog`` (a QWidget) from off the GUI thread: Qt printed
    "QWidget::repaint: Recursive repaint detected" and then segfaulted,
    and ``QThread.wait()`` was being called by the very thread it was
    waiting on.

    Because these handlers are bound methods of a QObject created on the
    GUI thread, the connections are queued and the dialog is only ever
    touched by the thread that owns it.
    """

    def __init__(self, dlg: QProgressDialog, thread: QThread,
                 worker: "_HFDownloadWorker", parent,
                 on_done: Callable[[Optional[DownloadResult], str], None]):
        """Hold the four objects one download needs kept alive together.

        :param dlg: the progress dialog this updates and closes.
        :param thread: the worker's thread. HELD, NOT JUST USED: a QThread
            that goes out of scope while running takes the download with it.
        :param worker: the object doing the fetching, held for the same
            reason.
        :param parent: the owning widget, also kept as ``_owner`` so the
            callback can reach the screen that asked for the download.
        :param on_done: called with the result and a message when the
            download finishes, whether it succeeded or not -- the result is
            ``None`` on failure and the message says why.
        """
        super().__init__(parent)
        self._dlg = dlg
        self._thread = thread
        self._worker = worker
        self._owner = parent
        self._on_done = on_done

    @Slot(str, int, int)
    def on_progress(self, name: str, done: int, total: int) -> None:
        """Say what is being fetched, how far along, and as what percentage.

        The bar carries no text of its own -- this line is the only place a
        percentage appears, which is why it has to be here rather than left to
        `QProgressBar`'s own label. `name` is a file for the per-file workers
        and an archive for the tar ones; `done`/`total` are files in the first
        case and megabytes in the second, so the unit is not stated and the
        percentage is what both have in common.
        """
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self._dlg.setMaximum(total)
        self._dlg.setValue(done)
        percent = round(done * 100 / total)
        # The name last: it is the part that can be long, so a window too
        # narrow for all of it still shows the percentage.
        self._dlg.setLabelText(f"{percent}%  ({done}/{total})  {name}")

    @Slot(str)
    def on_info(self, msg: str) -> None:
        self._dlg.setLabelText(msg)

    @Slot(bool, str, str, str)
    def on_finished(self, ok: bool, ds: str, st: str, err: str) -> None:
        # Close the dialog *before* invoking the user callback — the
        # callback may open its own modals (Continue/Stop prompts, etc.),
        # and stacking one modal on top of another confuses Qt into the
        # "app not responding" state on Linux.
        dlg = self._dlg
        try:
            dlg.setValue(dlg.maximum())
        except Exception:
            pass
        dlg.reset()
        dlg.close()
        dlg.deleteLater()
        self._thread.quit()
        self._thread.wait(2000)
        # Drop retained refs on the owner so the QThread + dialog can
        # be garbage-collected once the download flow ends.
        for attr in ("_hf_download_thread", "_hf_download_worker",
                     "_hf_download_dialog", "_hf_download_ui"):
            try:
                delattr(self._owner, attr)
            except Exception:
                pass
        on_done = self._on_done
        self.deleteLater()
        # Defer the user callback via a 0-ms singleShot so Qt processes
        # any pending events (close event, deleteLater) before the
        # chained pipeline modals appear. This is the specific fix for
        # the "force-quit dialog after download" symptom.
        if ok:
            QTimer.singleShot(
                0,
                lambda: on_done(DownloadResult(
                    dataset_path=Path(ds),
                    settings_path=Path(st)), ""),
            )
        else:
            QTimer.singleShot(0, lambda: on_done(None, err))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

class _MeasureExampleWorker(QObject):
    """Fetch Measure's example plate and leave it in the shape Measure reads.

    Signals match :class:`_HFDownloadWorker` so the same progress dialog
    drives both.
    """

    progress = Signal(str, int, int)
    info     = Signal(str)
    finished = Signal(bool, str, str, str)

    def __init__(self, dest_dir: Path):
        """Prepare the worker.

        :param dest_dir: where the download is written. Read on the worker
            thread, not in the constructor -- so a caller may hand over a
            folder that does not exist yet, and a failure to create it is
            reported through ``finished`` like every other failure rather
            than raised into the caller's event handler.
        """
        super().__init__()
        self._dest = Path(dest_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.info.emit("Listing files on Hugging Face…")
            try:
                from huggingface_hub import list_repo_files
            except ImportError as exc:
                raise ImportError(
                    f"huggingface_hub is not installed: {exc}") from exc
            names = [f for f in list_repo_files(MEASURE_EXAMPLE_REPO,
                                                repo_type="dataset")
                     if not f.startswith(".")]
            if not names:
                self.finished.emit(False, "", "",
                                   "No files to download from "
                                   f"{MEASURE_EXAMPLE_REPO}.")
                return
            self.info.emit(f"Found {len(names)} files to download.")
            root = self._dest
            root.mkdir(parents=True, exist_ok=True)

            total = len(names)
            for done, name in enumerate(names):
                if self._cancel:
                    self.finished.emit(False, "", "", "Cancelled by user.")
                    return
                self.progress.emit(name, done, total)
                # Sub-paths are preserved: `merged/` is where Measure looks,
                # and flattening the repo would put the arrays where nothing
                # reads them.
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                _download_one(MEASURE_EXAMPLE_REPO, name, target.parent)

            self.info.emit("Unpacking the arrays…")
            expand_measure_arrays(root / "merged")
            self.progress.emit("done", total, total)
            self.finished.emit(True, str(root),
                               str(root / "settings"), "")
        except Exception as e:                               # noqa: BLE001
            LOG.warning("measure example download failed: %s", e,
                        exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))

    def _expand_arrays(self, merged: Path) -> None:
        """Deprecated shim: call :func:`expand_measure_arrays`.

        Kept because it is a method on a worker that other code may still hold,
        but it does no work of its own -- see the module function for why this
        stopped being a method at all.
        """
        expand_measure_arrays(merged)


class _TarExampleWorker(QObject):
    """Fetch one example dataset as a single archive and unpack it.

    Subclasses name the repo. Everything else -- the streaming download, the
    cancel checks, the safe extraction and the path rewrite -- is shared,
    because every example set needs all four and a second copy of any of them
    is a second place to get the extraction filter wrong.
    """

    progress = Signal(str, int, int)
    info     = Signal(str)
    finished = Signal(bool, str, str, str)

    #: Set by each subclass.
    repo: str = ""

    def after_extract(self, dest) -> None:
        """Hook for whatever one set needs after unpacking. Nothing by default."""

    def dataset_root(self, dest) -> Path:
        """What the caller is handed as "the data".

        The whole unpacked folder for most sets. The Mask demo overrides it,
        because its callers have always been given the plate directory rather
        than the folder holding it -- and changing that would move the `src`
        the example fills in.
        """
        return Path(dest)

    def __init__(self, dest_dir: Path):
        """Prepare the worker.

        :param dest_dir: where the download is written. Read on the worker
            thread, not in the constructor -- so a caller may hand over a
            folder that does not exist yet, and a failure to create it is
            reported through ``finished`` like every other failure rather
            than raised into the caller's event handler.
        """
        super().__init__()
        self._dest = Path(dest_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            import requests

            archive_name = EXAMPLE_ARCHIVES[self.repo]
            self.info.emit("Downloading the example dataset…")
            url = (f"https://huggingface.co/datasets/{self.repo}/resolve/main/"
                   f"{archive_name}?download=true")
            target = self._dest / archive_name
            self._dest.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            expected = _content_length(response)
            written = 0
            with part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if self._cancel:
                        # BETWEEN CHUNKS, so Cancel and application shutdown
                        # both take effect within a megabyte rather than after
                        # the whole set has arrived.
                        part.unlink(missing_ok=True)
                        self.finished.emit(False, "", "", "Cancelled by user.")
                        return
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
                    if expected:
                        self.progress.emit(
                            archive_name, written // (1 << 20),
                            max(1, expected // (1 << 20)))
            if expected is not None and written != expected:
                part.unlink(missing_ok=True)
                raise IOError(
                    f"the download stopped early: {written} bytes of "
                    f"{expected}. Nothing was unpacked.")
            part.replace(target)

            self.info.emit("Unpacking…")
            extract_example_archive(target, self._dest)
            # The archive is not kept: it is a second copy of everything that
            # was just written, and these sets are hundreds of megabytes.
            target.unlink(missing_ok=True)

            self.info.emit("Preparing the files…")
            # Whatever this particular set needs doing to it after unpacking.
            self.after_extract(self._dest)
            make_the_example_paths_absolute(self._dest)
            self.progress.emit("done", 1, 1)
            self.finished.emit(True, str(self.dataset_root(self._dest)),
                               str(self._dest / "settings"), "")
        except Exception as e:                               # noqa: BLE001
            LOG.warning("example download failed: %s", e, exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))


class _ChosenArchivesWorker(_TarExampleWorker):
    """Fetch a chosen LIST of archives, one after another.

    The screen is published as eight separate pieces so a user can take the
    two-gigabyte databases without the thirty gigabytes of crops. This is the
    worker behind that choice: same streaming, same cancel-between-chunks, same
    filtered extraction, run once per selected piece.
    """

    repo = ""

    def __init__(self, dest_dir, archives=(), repo: str = ""):
        """Prepare a worker for the selected archives only.

        :param dest_dir: where the archives are written; see the base worker.
        :param archives: the pieces to fetch. EMPTY IS REFUSED rather than
            treated as "all" -- this worker exists so a user can take the
            2 GB databases without the 30 GB of crops, and a default of
            everything would silently undo that choice.
        :param repo: the Hugging Face repo to pull from. Empty keeps the
            class attribute, which is what every caller uses; it is a
            parameter so a test can point the same worker at a fixture.
        """
        super().__init__(dest_dir)
        self._archives = list(archives)
        self.repo = repo or self.repo

    def run(self) -> None:
        try:
            if not self._archives:
                self.finished.emit(False, "", "", "Nothing was selected.")
                return
            done = []
            for position, archive in enumerate(self._archives, start=1):
                if self._cancel:
                    self.finished.emit(False, "", "", "Cancelled by user.")
                    return
                self.info.emit(
                    f"Downloading {archive} ({position} of "
                    f"{len(self._archives)})…")
                if not self._fetch_one(archive):
                    return                      # it emitted its own outcome
                done.append(archive)
            self.info.emit("Preparing the files…")
            make_the_example_paths_absolute(self._dest)
            self.progress.emit("done", 1, 1)
            self.finished.emit(True, str(self._dest),
                               str(self._dest / "settings"), "")
        except Exception as e:                               # noqa: BLE001
            LOG.warning("screen download failed: %s", e, exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))

    def _fetch_one(self, archive: str) -> bool:
        """Stream and unpack one archive. False when it ended the run."""
        import requests

        url = (f"https://huggingface.co/datasets/{self.repo}/resolve/main/"
               f"{archive}?download=true")
        target = self._dest / archive
        self._dest.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        expected = _content_length(response)
        written = 0
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if self._cancel:
                    part.unlink(missing_ok=True)
                    self.finished.emit(False, "", "", "Cancelled by user.")
                    return False
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if expected:
                    self.progress.emit(archive, written // (1 << 20),
                                       max(1, expected // (1 << 20)))
        if expected is not None and written != expected:
            part.unlink(missing_ok=True)
            raise IOError(
                f"{archive} stopped early: {written} bytes of {expected}. "
                f"Nothing was unpacked.")
        part.replace(target)
        extract_example_archive(target, self._dest)
        target.unlink(missing_ok=True)
        return True


def download_chosen_screen_data(parent, dest: Path, archives, repo: str,
                                on_done) -> None:
    """Fetch the chosen pieces of the published screen."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    def _factory(where):
        return _ChosenArchivesWorker(where, archives=archives, repo=repo)

    download_toxo_mito_demo(parent, dest, on_done, worker_factory=_factory,
                            title="Downloading screen data")


class _MeasureTarWorker(_TarExampleWorker):
    repo = MEASURE_EXAMPLE_REPO

    def after_extract(self, dest) -> None:
        """Write the compressed arrays back out as the ``.npy`` Measure reads.

        The compression is a TRANSPORT detail -- it halves the download -- and
        Measure loads `.npy`. Converting here keeps the second format entirely
        inside the downloader rather than teaching every reader about it.
        """
        # No worker is constructed: see expand_measure_arrays.
        expand_measure_arrays(Path(dest) / "merged")


class _AnnotateTarWorker(_TarExampleWorker):
    repo = ANNOTATE_EXAMPLE_REPO


class _MaskTarWorker(_TarExampleWorker):
    """The Mask demo, which is 210 files across two repos.

    The archive carries the settings pack under `settings/` as well, so the
    demo arrives in one request instead of 210 plus 2 -- and the two halves
    can no longer arrive out of step with each other, which they could when
    they were fetched from separate repos in separate loops.
    """

    repo = DATASET_REPO

    def dataset_root(self, dest) -> Path:
        """The plate folder itself.

        The archive's members are the plate's CONTENTS now -- the tifs at the
        top, `settings/` beside them -- so the destination is already the
        plate directory `src` should name. It used to carry a `plate1/`
        prefix, which put the images one level deeper than the other sets.
        """
        return Path(dest)


def download_annotate_example(parent, dest: Path,
                              on_done: Callable[
                                  [Optional[DownloadResult], str],
                                  None]) -> None:
    """Fetch the Annotate/Classify example set, with the shared dialog."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    download_toxo_mito_demo(
        parent, dest, on_done,
        worker_factory=_AnnotateTarWorker,
        title="Downloading spaCR annotation example data")


def download_measure_example(parent, dest: Path,
                             on_done: Callable[
                                 [Optional[DownloadResult], str],
                                 None]) -> None:
    """Fetch Measure's example plate, with the same dialog as the Mask demo.

    :param parent: any QWidget — the progress dialog parents to this.
    :param dest: local directory that will hold ``merged/`` and ``settings/``.
    :param on_done: called with ``(result, error_message)``; ``result`` is
        ``None`` on failure or cancellation.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    download_toxo_mito_demo(
        parent, dest, on_done,
        worker_factory=_MeasureTarWorker,
        title="Downloading spaCR Measure example data")


def download_toxo_mito_demo(parent,
                                dest: Path,
                                on_done: Callable[
                                    [Optional[DownloadResult], str], None],
                                *,
                                worker_factory=None,
                                title: str = "Downloading spaCR demo dataset"
                                ) -> None:
    """Kick off the demo download with a modal progress dialog.

    :param parent: any QWidget — the progress dialog parents to this.
    :param dest: local directory that will hold ``plate1/`` and
        ``settings/`` subfolders.
    :param on_done: callback fired on completion or cancellation with
        ``(result, error_message)``. ``result`` is ``None`` on failure /
        cancel; otherwise the two local paths.

    Nothing is returned — the callback carries the outcome. Errors and
    cancellations are non-fatal: they just call ``on_done`` with a
    ``None`` result and an explanatory string.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    dlg = _DownloadDialog(title, parent)
    # WIDE ENOUGH FOR WHAT IT WILL SAY, on top of the wrapping the dialog
    # already does. Widening alone never fixed this -- the longest caption is
    # a FILE NAME and there is no longest file name -- but a window sized from
    # "Preparing…" starts absurdly narrow and jumps on the first update.
    dlg.setMinimumWidth(max(
        dlg.minimumWidth(),
        dlg.spacr_caption.fontMetrics().horizontalAdvance(_WIDEST_CAPTION)
        + _CAPTION_MARGIN))
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    # AutoClose True so hitting max value closes the dialog and returns
    # control to the event loop — otherwise a stuck modal blocks the
    # main thread and Qt shows the "Application not responding" prompt.
    dlg.setAutoClose(True)
    dlg.setAutoReset(True)
    dlg.show()

    thread = QThread(parent)
    # WHICH worker, so a second dataset reuses this function's wiring rather
    # than copying it. The thread affinity, the direct-connected cancel and
    # the deliberate absence of a `deleteLater` below are all load-bearing and
    # were each arrived at from a measured crash; a second copy of them would
    # be a second place for one of them to be dropped.
    # THE DEFAULT STAYS THE PER-FILE WORKER, and the Mask demo asks for the
    # tar at its call site instead.
    #
    # Switching the default here looked tidier and broke
    # `tests/qt/test_console_thread_safety.py`, which patches `_list_files`
    # and drives this function to prove the offline failure path stays on the
    # GUI thread. The tar worker does not call `_list_files`, so the patched
    # test went to the network for real and aborted. A shared entry point's
    # default is part of its contract with everything already calling it.
    worker = (worker_factory or _HFDownloadWorker)(dest)
    worker.moveToThread(thread)

    # ``ui`` is constructed here, on the GUI thread, so every connection
    # below is a queued one — see _HFDownloadUI's docstring.
    ui = _HFDownloadUI(dlg, thread, worker, parent, on_done)
    worker.progress.connect(ui.on_progress)
    worker.info.connect(ui.on_info)
    worker.finished.connect(ui.on_finished)

    # DirectConnection is mandatory here: the worker's event loop is
    # blocked for the whole of run(), so a queued cancel would not be
    # delivered until after the download it was meant to abort had
    # already finished. cancel() only flips a bool, which is safe to do
    # from the GUI thread.
    dlg.canceled.connect(worker.cancel, Qt.DirectConnection)
    # AND QUITTING THE APPLICATION CANCELS IT TOO.
    #
    # Nothing did. A download still running when the window closed left a
    # QThread to be destroyed with its thread alive -- "QThread: Destroyed
    # while thread '' is still running", then abort -- because the finished
    # handler that quits and waits for the thread only runs if the worker
    # EMITS finished, and a worker that is still downloading never does.
    #
    # DirectConnection for the same reason the cancel above uses it: the
    # worker's event loop is blocked for the whole of run(), so a queued call
    # would be delivered after the shutdown it was meant to survive. cancel()
    # only flips a bool.
    #
    # The wait is bounded and then given up on: a shutdown that hangs on a
    # slow socket is a worse failure than the one being prevented, and the
    # loop checks its flag between files.
    try:
        from PySide6.QtCore import QCoreApplication

        application = QCoreApplication.instance()
        if application is not None:
            def _stop_before_quitting(_w=worker, _t=thread):
                try:
                    _w.cancel()
                    _t.quit()
                    _t.wait(5000)
                except Exception:                            # noqa: BLE001
                    pass

            application.aboutToQuit.connect(_stop_before_quitting,
                                            Qt.DirectConnection)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not arm the shutdown cancel", exc_info=True)
    thread.started.connect(worker.run)
    # NOTE the absence of `thread.finished.connect(worker.deleteLater)`.
    # `spacr.qt.bridge.make_thread` documents why, from a measured crash:
    # the worker's affinity is the WORKER thread, so a deferred delete is
    # posted into a loop that is stopping, and it races the GUI thread
    # dropping the object's last Python reference in `on_finished`. Two
    # owners, one object — gdb put it in
    # `QThread -> sendPostedEvents -> ~QObject`. Chaining off
    # `thread.finished` rather than `worker.finished` does not help; that
    # exact variant was measured at 2 crashes in 20 runs. The worker is a
    # Python-constructed PySide6 object, so Python already owns it: the
    # last reference (held by `_HFDownloadUI`) frees it, on the thread
    # that holds it.
    thread.start()
    # Retain references on the parent so the QThread + worker + dialog
    # aren't garbage-collected while the download is in flight.
    parent._hf_download_thread = thread
    parent._hf_download_worker = worker
    parent._hf_download_dialog = dlg
    parent._hf_download_ui = ui
