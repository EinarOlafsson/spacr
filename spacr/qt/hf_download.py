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
from PySide6.QtWidgets import QProgressDialog

LOG = logging.getLogger("spacr.qt.hf_download")

# Match the classic Tk GUI's demo endpoints so users see the same
# dataset here they'd have seen in the Tk build.
DATASET_REPO  = "einarolafsson/toxo_mito"
DATASET_SUB   = "plate1"
SETTINGS_REPO = "einarolafsson/spacr_settings"


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
        progress(str, int, int) — (file_name, done_files, total_files)
        info(str)               — status message for the dialog label
        finished(bool, str, str, str)
            — (ok, dataset_path, settings_path, error)
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

def download_toxo_mito_demo(parent,
                                dest: Path,
                                on_done: Callable[
                                    [Optional[DownloadResult], str], None]) -> None:
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
    dlg.setWindowTitle("Downloading spaCR demo dataset")
    dlg.setMinimumDuration(0)
    dlg.setValue(0)
    # AutoClose True so hitting max value closes the dialog and returns
    # control to the event loop — otherwise a stuck modal blocks the
    # main thread and Qt shows the "Application not responding" prompt.
    dlg.setAutoClose(True)
    dlg.setAutoReset(True)

    thread = QThread(parent)
    worker = _HFDownloadWorker(dest)
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
