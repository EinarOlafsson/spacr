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
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QLabel, QProgressDialog

LOG = logging.getLogger("spacr.qt.hf_download")

#: The longest caption the progress dialog shows, used to size it once at
#: construction. Not a guess: the download reports one HuggingFace file at
#: a time and this is a real name from the toxo_mito pack, which is the
#: dataset the "load example data" button fetches.
_WIDEST_CAPTION = "Downloading plate1_A01_T0001F001L01A01Z01C01.tif"

#: Room for the dialog's frame, its margins and the progress bar's own
#: padding, on top of the caption itself.
_CAPTION_MARGIN = 96

# Match the classic Tk GUI's demo endpoints so users see the same
# dataset here they'd have seen in the Tk build.
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

#: The token a published settings file uses for "wherever this was unpacked".
DATASET_PLACEHOLDER = "<dataset>"


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


def explain_download_failure(exc: BaseException) -> str:
    """Turn a download exception into something a user can act on.

    This is the only demo in the Demos menu that needs the network — the six
    synthetic generators are entirely offline — so it is the only one that can
    fail for a reason outside spaCR. What the user saw before was
    ``str(exc)``, which for the ordinary offline case is a nested urllib3
    dump::

        (MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443):
        Max retries exceeded with url: /api/datasets/... (Caused by
        NewConnectionError('<urllib3.connection.HTTPSConnection object at
        0x7e8d...>: Failed to establish a new connection: [Errno 101] Network
        is unreachable'))"), '(Request ID: 73ac20ed-...)')

    — 300 characters that never say "you are offline" and never say what to do
    instead. The three conditions this actually fails on are: no network, the
    ``huggingface_hub`` extra not installed, and a truncated transfer. Each
    gets a sentence naming the cause and the way out; anything else keeps its
    own message with the same closing advice attached.

    :param exc: the exception raised inside the download worker.
    :returns: a multi-line message for the failure dialog.
    """
    offline_hint = (
        "Every other entry in the Demos menu is synthetic and runs with no "
        "network at all — use one of those to try the pipelines offline.")

    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return (
            "The real-dataset demo needs the 'huggingface_hub' package to "
            "list the demo repository, and it is not installed in this "
            f"environment ({exc}).\n\n"
            "Install it with:  pip install huggingface_hub\n\n"
            + offline_hint)

    # The truncation check comes first: `IOError` IS `OSError`, and the
    # builtin ConnectionError below is an OSError subclass, so ordering these
    # the other way round would let a half-finished transfer be reported as
    # "check your internet connection" — true but useless, because the
    # connection was fine right up to the point it was not.
    if isinstance(exc, OSError) and "Truncated download" in str(exc):
        return (
            f"{exc}\n\n"
            "The connection dropped part-way through. Nothing partial was "
            "kept, so re-running the demo starts the file again.\n\n"
            + offline_hint)

    # requests is an install-time dependency of huggingface_hub, but the
    # import is kept local so a broken environment reports the missing
    # package above rather than dying here. The builtins are in the tuple
    # too: `requests.exceptions.ConnectionError` descends from OSError, not
    # from the builtin ConnectionError, and a DNS failure raised by anything
    # other than requests (urllib, socket, huggingface_hub's own client)
    # arrives as one of these instead.
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
            "Could not reach huggingface.co, so the real demo dataset could "
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


def _download_one(repo_id: str, file_name: str, dest_dir: Path) -> Path:
    """Stream one file from the HF repo to ``dest_dir/basename``.

    Uses plain HTTP + streaming so we don't need the full ``hf_hub``
    download machinery (and its cache dir) for a one-shot demo pull.

    The body lands in a sibling ``.part`` file and is only moved onto
    the final path once every advertised byte has arrived. Writing
    straight to the destination meant a dropped connection left a
    truncated image behind that was indistinguishable from a good
    download — the next pipeline run then failed deep inside the mask
    stage instead of at the download.
    """
    import requests
    url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
             f"{file_name}?download=true")
    dst = dest_dir / Path(file_name).name
    part = dst.with_name(dst.name + ".part")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    expected = _content_length(resp)
    written = 0
    try:
        with part.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 15):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
        if expected is not None and written != expected:
            raise IOError(
                f"Truncated download for {file_name}: wrote {written} "
                f"bytes but the server declared {expected}."
            )
        os.replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return dst


# ---------------------------------------------------------------------------
# GUI-thread receiver
# ---------------------------------------------------------------------------

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
        super().__init__(parent)
        self._dlg = dlg
        self._thread = thread
        self._worker = worker
        self._owner = parent
        self._on_done = on_done

    @Slot(str, int, int)
    def on_progress(self, name: str, done: int, total: int) -> None:
        self._dlg.setMaximum(max(1, total))
        self._dlg.setValue(done)
        self._dlg.setLabelText(f"Downloading {name}\n({done}/{total} files)")

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
            self._expand_arrays(root / "merged")
            self.progress.emit("done", total, total)
            self.finished.emit(True, str(root),
                               str(root / "settings"), "")
        except Exception as e:                               # noqa: BLE001
            LOG.warning("measure example download failed: %s", e,
                        exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))

    def _expand_arrays(self, merged: Path) -> None:
        """Write each ``.npz`` back out as the ``.npy`` Measure reads.

        The compression is a TRANSPORT detail -- it halves a 700 MB download
        -- and Measure loads `.npy`. Converting on arrival keeps that entirely
        inside the downloader rather than teaching every reader about a second
        format.

        The ``.npz`` is removed afterwards: keeping both doubles the disk cost
        of an example dataset for a file nothing will open again.
        """
        import numpy as np

        if not merged.is_dir():
            return
        for archive in sorted(merged.glob("*.npz")):
            target = archive.with_suffix(".npy")
            if target.is_file():
                archive.unlink(missing_ok=True)
                continue
            try:
                with np.load(archive) as bundle:
                    # Written by the publisher under `image`; the first key is
                    # the fallback so a hand-made archive still loads.
                    key = "image" if "image" in bundle else bundle.files[0]
                    np.save(target, bundle[key])
                archive.unlink(missing_ok=True)
            except Exception:                                # noqa: BLE001
                # One bad archive must not cost the other fifteen. It is left
                # on disk, so what failed is visible rather than merely absent.
                LOG.warning("could not unpack %s", archive, exc_info=True)


def make_the_example_paths_absolute(root) -> int:
    """Point a downloaded example at where it actually landed.

    A measurements database stores ABSOLUTE paths to its crops, which name the
    machine that made it and resolve nowhere else. The published copy stores
    them relative to the dataset root instead, so it is portable and carries no
    account name -- and this is what turns them back into paths that open.

    The settings files are rewritten the same way: they carry
    :data:`DATASET_PLACEHOLDER` where the unpack location goes, so a user can
    press Run without first editing a path.

    Idempotent. A path that is already absolute is left alone, so running this
    twice -- a re-download over an existing copy -- does not produce
    ``/home/me/data//home/me/data/...``.

    :param root: the folder the dataset was unpacked into.
    :returns: how many values were rewritten.
    """
    import sqlite3

    root = Path(root)
    prefix = str(root).rstrip("/") + "/"
    rewritten = 0

    database = root / "measurements.db"
    if database.is_file():
        connection = sqlite3.connect(str(database))
        try:
            tables = [r[0] for r in connection.execute(
                "select name from sqlite_master where type='table'")]
            for table in tables:
                columns = [r[1] for r in connection.execute(
                    f'PRAGMA table_info("{table}")')]
                for column in columns:
                    try:
                        # Only the values that look like OUR relative paths.
                        # A column holding prose is untouched, and one already
                        # absolute is skipped by the same test.
                        cursor = connection.execute(
                            f'update "{table}" set "{column}" = ? || "{column}" '
                            f'where cast("{column}" as text) like \'data/%\' '
                            f'or cast("{column}" as text) like \'measurements/%\'',
                            (prefix,))
                        rewritten += cursor.rowcount or 0
                    except sqlite3.Error:
                        # A column that cannot be updated -- a generated one,
                        # or a type that will not concatenate -- is not a
                        # reason to abandon the other forty.
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


class _AnnotateExampleWorker(QObject):
    """Fetch the Annotate/Classify example set and make it usable in place."""

    progress = Signal(str, int, int)
    info     = Signal(str)
    finished = Signal(bool, str, str, str)

    def __init__(self, dest_dir: Path):
        super().__init__()
        self._dest = Path(dest_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self.info.emit("Fetching the example annotation set…")
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    f"huggingface_hub is not installed: {exc}") from exc
            # snapshot_download RATHER THAN a file at a time. This set is 2,365
            # files, and one HTTP request each -- the shape the Mask demo uses
            # for its six -- spends minutes on request overhead alone.
            #
            # The cost is granular progress: the dialog cannot show a count it
            # is not given. It says what it is doing instead, which is better
            # than a bar that moves once.
            self._dest.mkdir(parents=True, exist_ok=True)
            snapshot_download(ANNOTATE_EXAMPLE_REPO, repo_type="dataset",
                              local_dir=str(self._dest))
            if self._cancel:
                self.finished.emit(False, "", "", "Cancelled by user.")
                return
            self.info.emit("Pointing the database at its crops…")
            make_the_example_paths_absolute(self._dest)
            self.progress.emit("done", 1, 1)
            self.finished.emit(True, str(self._dest),
                               str(self._dest / "settings"), "")
        except Exception as e:                               # noqa: BLE001
            LOG.warning("annotate example download failed: %s", e,
                        exc_info=True)
            self.finished.emit(False, "", "", explain_download_failure(e))


def download_annotate_example(parent, dest: Path,
                              on_done: Callable[
                                  [Optional[DownloadResult], str],
                                  None]) -> None:
    """Fetch the Annotate/Classify example set, with the shared dialog."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    download_toxo_mito_demo(
        parent, dest, on_done,
        worker_factory=_AnnotateExampleWorker,
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
        worker_factory=_MeasureExampleWorker,
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

    dlg = QProgressDialog("Preparing…", "Cancel", 0, 1, parent)
    dlg.setWindowTitle(title)
    # SIZED FOR THE TEXT IT WILL SHOW, NOT THE TEXT IT STARTS WITH.
    # A QProgressDialog takes its width from the label it is constructed
    # with, and this one is constructed with "Preparing…" -- eleven
    # characters -- then spends the whole download showing
    # "Downloading <filename>\n(3/6 files)". Reported 2026-08-31: "the
    # text number and % test is cut off".
    #
    # A plain QLabel does not wrap, so the filename was clipped at the
    # dialog edge and the count line fell outside the dialog entirely.
    # Both are fixed by the same two changes: a label that WRAPS, and a
    # width chosen from the longest string this dialog actually shows
    # rather than from its first one.
    caption = QLabel("Preparing…", dlg)
    caption.setWordWrap(True)
    caption.setAlignment(Qt.AlignmentFlag.AlignLeft
                         | Qt.AlignmentFlag.AlignVCenter)
    dlg.setLabel(caption)
    # KEPT ON THE DIALOG. PySide6's QProgressDialog exposes setLabel() but
    # no label() getter, so the only way to reach this widget again -- to
    # measure whether its text still fits -- is to hold on to it.
    dlg.spacr_caption = caption
    dlg.setMinimumWidth(
        caption.fontMetrics().horizontalAdvance(_WIDEST_CAPTION)
        + _CAPTION_MARGIN)
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    # AutoClose True so hitting max value closes the dialog and returns
    # control to the event loop — otherwise a stuck modal blocks the
    # main thread and Qt shows the "Application not responding" prompt.
    dlg.setAutoClose(True)
    dlg.setAutoReset(True)

    thread = QThread(parent)
    # WHICH worker, so a second dataset reuses this function's wiring rather
    # than copying it. The thread affinity, the direct-connected cancel and
    # the deliberate absence of a `deleteLater` below are all load-bearing and
    # were each arrived at from a measured crash; a second copy of them would
    # be a second place for one of them to be dropped.
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
