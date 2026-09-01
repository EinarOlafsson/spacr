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
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel,
                               QProgressBar, QProgressDialog,
                               QPushButton, QVBoxLayout, QWidget)

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


def example_plate_folder() -> Path:
    """The ONE folder every example dataset unpacks into.

    ``~/.cache/spacr/example_data/plate1`` -- a real spaCR plate directory,
    holding whichever of ``merged/``, ``data/``, ``measurements/`` and
    ``settings/`` have been downloaded.

    ONE FOLDER BECAUSE THE SETS ARE USED TOGETHER. `data/` is the crops and
    `measurements/measurements.db` is what indexes them; downloading them into
    separate trees meant the two halves of one plate could not be opened at
    once, and the user had to know which download had put what where. Each
    archive's members are relative to this folder, so the three unpack into it
    side by side and compose into a plate that Measure, Annotate and Classify
    can all be pointed at.
    """
    return Path.home() / ".cache" / "spacr" / "example_data" / "plate1"


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


def expand_measure_arrays(merged: Path) -> None:
    """Write each ``.npz`` back out as the ``.npy`` Measure reads.

    The compression is a TRANSPORT detail -- it halves a 700 MB download -- and
    Measure loads `.npy`. Converting on arrival keeps that entirely inside the
    downloader rather than teaching every reader about a second format.

    The ``.npz`` is removed afterwards: keeping both doubles the disk cost of
    an example dataset for a file nothing will open again.

    A MODULE FUNCTION, NOT A METHOD, and that is the point. `after_extract`
    runs on the download thread, and reaching this code through
    ``_MeasureExampleWorker(dest)._expand_arrays(...)`` CONSTRUCTED a QObject
    there purely to borrow a helper. `thread_guard` reported it exactly as it
    should have: the object then lived on 'Dummy-6' and every later touch from
    the GUI thread was illegal. Nothing in here ever read ``self``, so there
    was never an object to need.
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
}


def extract_example_archive(archive, dest) -> int:
    """Unpack a downloaded example archive under ``dest``.

    EXTRACTED WITH ``filter="data"``, which is the whole reason this is a
    function rather than two lines at the call site. A tar can name
    ``../../etc/something`` or an absolute path, and a plain ``extractall``
    will happily write there -- so unpacking downloaded content without a
    filter hands whoever can publish to the repo a write anywhere the user can
    write. The filter rejects those members, along with device nodes, setuid
    bits and symlinks pointing outside the tree.

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
            # No filter available: refuse anything that leaves the tree rather
            # than trusting the archive.
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

    # WHEREVER THE DATABASE IS. spaCR keeps it at `measurements/measurements.db`
    # inside a plate; the published archive used to carry it at the top. Both
    # are checked so an already-unpacked older copy is still repaired.
    for database in (root / "measurements" / "measurements.db",
                     root / "measurements.db"):
        if database.is_file():
            break
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
