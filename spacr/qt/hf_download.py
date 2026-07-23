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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal
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
            self.finished.emit(False, "", "", str(e))


def _list_files(repo_id: str, subfolder: str) -> List[str]:
    """Return every file path in ``repo_id`` matching ``subfolder``.

    Empty subfolder means "top-level CSVs only" (mirrors the Tk
    downloader's behaviour for the settings pack).
    """
    from huggingface_hub import list_repo_files
    files = list_repo_files(repo_id, repo_type="dataset")
    if subfolder:
        return [f for f in files if f.startswith(subfolder)]
    return [f for f in files if f.endswith(".csv")]


def _download_one(repo_id: str, file_name: str, dest_dir: Path) -> Path:
    """Stream one file from the HF repo to ``dest_dir/basename``.

    Uses plain HTTP + streaming so we don't need the full ``hf_hub``
    download machinery (and its cache dir) for a one-shot demo pull.
    """
    import requests
    url = (f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
             f"{file_name}?download=true")
    dst = dest_dir / Path(file_name).name
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    with dst.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                fh.write(chunk)
    return dst


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

    def _on_progress(name: str, done: int, total: int) -> None:
        dlg.setMaximum(max(1, total))
        dlg.setValue(done)
        dlg.setLabelText(f"Downloading {name}\n({done}/{total} files)")

    def _on_info(msg: str) -> None:
        dlg.setLabelText(msg)

    def _on_finished(ok: bool, ds: str, st: str, err: str) -> None:
        # Close the dialog *before* invoking the user callback — the
        # callback may open its own modals (Continue/Stop prompts, etc.),
        # and stacking one modal on top of another confuses Qt into the
        # "app not responding" state on Linux.
        try:
            dlg.setValue(dlg.maximum())
        except Exception:
            pass
        dlg.reset()
        dlg.close()
        dlg.deleteLater()
        thread.quit()
        thread.wait(2000)
        # Drop retained refs on the parent so the QThread + dialog can
        # be garbage-collected once the download flow ends.
        for attr in ("_hf_download_thread", "_hf_download_worker",
                     "_hf_download_dialog"):
            try:
                delattr(parent, attr)
            except Exception:
                pass
        # Defer the user callback via a 0-ms singleShot so Qt processes
        # any pending events (close event, deleteLater) before the
        # chained pipeline modals appear. This is the specific fix for
        # the "force-quit dialog after download" symptom.
        from PySide6.QtCore import QTimer
        if ok:
            QTimer.singleShot(
                0,
                lambda: on_done(DownloadResult(
                    dataset_path=Path(ds),
                    settings_path=Path(st)), ""),
            )
        else:
            QTimer.singleShot(0, lambda: on_done(None, err))

    worker.progress.connect(_on_progress)
    worker.info.connect(_on_info)
    worker.finished.connect(_on_finished)

    dlg.canceled.connect(worker.cancel)
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.start()
    # Retain references on the parent so the QThread + worker + dialog
    # aren't garbage-collected while the download is in flight.
    parent._hf_download_thread = thread
    parent._hf_download_worker = worker
    parent._hf_download_dialog = dlg
