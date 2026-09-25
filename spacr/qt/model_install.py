"""Install a model from inside spaCR, without freezing the window.

Two kinds of thing are installed from a click: a segmentation backend, which
is a Python package added to the running environment with ``pip`` (DINOCell
and SAMCell, :data:`spacr.model_zoo.INSTALLABLE_BACKENDS`), and a Cellpose
checkpoint from the model zoo, which is a file downloaded and verified by
:func:`spacr.model_zoo.install`. Both can take minutes, so both run off the
GUI thread: :class:`PackageInstall` runs ``pip`` as a child process watched
from the event loop, and :class:`CheckpointDownload` runs the zoo's download
on a worker thread. Each reports progress while it runs and a single result
when it ends, and a failure is a message rather than an exception, so a
failed install leaves the screen that started it working.

:class:`SegmentationBackendCombo` is the Mask module's choice of
segmentation backend: every backend is listed, one that is not installed is
greyed, and choosing it offers to install it and selects it once installed.
Make Masks offers the same backends, in the same way, in its Live magnifier
Mode box.
"""
from __future__ import annotations

import logging
import sys
from importlib import invalidate_caches
from importlib.util import find_spec
from typing import Any, Callable, List, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QComboBox, QMessageBox, QWidget

LOG = logging.getLogger("spacr.qt.model_install")

#: The grey a model that is listed but not installed is written in.
UNINSTALLED_GREY = QColor(128, 128, 128)

#: How many lines of ``pip``'s output a failure message quotes.
FAILURE_TAIL_LINES = 8

#: Installs and downloads still running, held here so that the widget that
#: started one can go -- a settings form is rebuilt, a screen is closed --
#: without Qt deleting a child process half way through ``pip`` or a thread
#: half way through a download. Each removes itself when it ends.
_RUNNING: set = set()


def backend_rows() -> List[Tuple[str, str, str, str]]:
    """Every installable segmentation backend, as the model zoo lists it.

    :returns: ``(name, label, pip requirement, import name)`` per backend,
        in the zoo's order.
    """
    from .. import model_zoo

    return [(name, label, extra, module)
            for name, (label, extra, module, _blurb)
            in model_zoo.INSTALLABLE_BACKENDS.items()]


def backend_row(name: str) -> Optional[Tuple[str, str, str, str]]:
    """The :func:`backend_rows` row for one backend.

    :param name: a backend name such as ``'samcell'``.
    :returns: the row, or ``None`` for a name the zoo does not list.
    """
    for row in backend_rows():
        if row[0] == name:
            return row
    return None


def is_importable(module: str) -> bool:
    """Whether ``module`` can be imported here, without importing it.

    :param module: the top-level import name.
    :returns: True when Python can find it.
    """
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def pip_command(requirement: str) -> List[str]:
    """The command that installs ``requirement`` into this environment.

    ``python -m pip`` with the interpreter spaCR is running on, so the
    package lands where this process imports from.

    :param requirement: a pip requirement, for example
        ``'spacr[samcell]'``.
    :returns: the program followed by its arguments.
    """
    return [sys.executable, "-m", "pip", "install", requirement]


def can_install_packages() -> bool:
    """Whether this build of spaCR can add a package to itself.

    A frozen application has no ``pip`` to run: its interpreter is the
    application binary.

    :returns: False inside a frozen build.
    """
    return not getattr(sys, "frozen", False)


def confirm_backend_install(parent: Optional[QWidget], label: str,
                            requirement: str) -> bool:
    """Ask before installing a backend, and say what the install risks.

    :param parent: the widget the question belongs to.
    :param label: the backend's name, for example ``'SAMCell'``.
    :param requirement: what is passed to ``pip install``.
    :returns: True when the user agreed.
    """
    from .i18n import tr

    if not can_install_packages():
        QMessageBox.information(
            parent, tr("Install {name}", name=label),
            tr("This build of spaCR cannot install packages into itself. "
               "Install spaCR with pip or conda to add {name}.", name=label))
        return False
    answer = QMessageBox.warning(
        parent, tr("Install {name}?", name=label),
        tr("{name} is not installed.\n\nInstalling it runs:\n"
           "    pip install \"{requirement}\"\n\ninto the environment spaCR "
           "is running in. It downloads a large package and may change the "
           "installed version of torch, which can affect Cellpose and, in the "
           "worst case, stop spaCR starting. It can take several minutes; "
           "spaCR stays usable while it runs.\n\nInstall it now?",
           name=label, requirement=requirement),
        QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
    return answer == QMessageBox.Yes


class PackageInstall(QObject):
    """One ``pip install`` in a child process, watched from the event loop.

    Nothing waits on the process: its output arrives through
    :attr:`progressed` as it is written, and :attr:`finished` fires once,
    with whether it worked and what to tell the user. A process that cannot
    start finishes at once, unsuccessfully.

    Give it no parent. It keeps itself alive while ``pip`` runs, because a
    child process deleted with its owner is killed, and ``pip`` killed half
    way through an install can leave the environment broken.

    :param requirement: what to install, for example ``'spacr[samcell]'``.
    :param parent: owner, if the caller insists on one.
    :param command: the program and arguments to run instead of
        :func:`pip_command`, for a caller that needs a different installer.
    """

    #: The last line the installer wrote.
    progressed = Signal(str)
    #: ``(worked, message)``: once, when the installer has ended.
    finished = Signal(bool, str)

    def __init__(self, requirement: str, parent: Optional[QObject] = None,
                 *, command: Optional[List[str]] = None):
        """Prepare the install without starting it.

        :param requirement: what to install.
        :param parent: owner.
        :param command: a replacement for :func:`pip_command`.
        """
        super().__init__(parent)
        self.requirement = str(requirement)
        self._command = list(command or pip_command(self.requirement))
        self._output: List[str] = []
        self._done = False
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read)
        self._process.finished.connect(self._ended)
        self._process.errorOccurred.connect(self._failed_to_start)

    def start(self) -> bool:
        """Start the installer.

        :returns: False when it could not be started, in which case
            :attr:`finished` has already said so.
        """
        program, *arguments = self._command
        _RUNNING.add(self)
        self._process.start(program, arguments)
        if not self._process.waitForStarted(5000):
            self._finish(False, self._process.errorString()
                         or "the installer did not start")
            return False
        return True

    def is_running(self) -> bool:
        """Whether the installer is still running."""
        return self._process.state() != QProcess.NotRunning

    def cancel(self) -> None:
        """Stop the installer, and report it as not having worked."""
        self._finish(False, "cancelled")
        if self.is_running():
            self._process.kill()
            self._process.waitForFinished(3000)
        _RUNNING.discard(self)

    def output(self) -> str:
        """Everything the installer has written so far."""
        return "".join(self._output)

    def _read(self) -> None:
        """Keep the new output and pass its last line on."""
        chunk = bytes(self._process.readAllStandardOutput()).decode(
            "utf-8", "replace")
        if not chunk:
            return
        self._output.append(chunk)
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            self.progressed.emit(lines[-1])

    def _ended(self, code: int, status) -> None:
        """Report the result once the process has exited."""
        self._read()
        if status == QProcess.NormalExit and code == 0:
            self._finish(True, self.output())
            return
        tail = self.output().strip().splitlines()[-FAILURE_TAIL_LINES:]
        self._finish(False, f"pip exited {code}.\n\n"
                     + "\n".join(tail or ["No output."]))

    def _failed_to_start(self, error) -> None:
        """Report a process that never ran."""
        if error == QProcess.FailedToStart:
            self._finish(False, self._process.errorString())

    def _finish(self, worked: bool, message: str) -> None:
        """Emit :attr:`finished` exactly once."""
        if not self.is_running():
            _RUNNING.discard(self)
        if self._done:
            return
        self._done = True
        if worked:
            invalidate_caches()
        self.finished.emit(bool(worked), str(message))


class _DownloadWorker(QObject):
    """Runs one zoo download on a worker thread and reports as it goes."""

    progressed = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fetch: Callable[..., Any]):
        """Hold the call that does the download.

        :param fetch: ``fetch(progress=..., cancel=...)``, returning what was
            installed.
        """
        super().__init__()
        self._fetch = fetch
        self.stop = False

    def run(self) -> None:
        """Do the download, then say how it ended."""
        try:
            result = self._fetch(
                progress=lambda got, total: self.progressed.emit(
                    int(got), int(total or 0)),
                cancel=lambda: self.stop)
        except Exception as exc:                             # noqa: BLE001
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.done.emit(result)


class CheckpointDownload(QObject):
    """Download and verify one model-zoo checkpoint off the GUI thread.

    Goes through :func:`spacr.model_zoo.install`, so the file is checked
    against its published digest before it is put where it belongs, and a
    failure leaves the destination as it was. Like :class:`PackageInstall`
    it keeps itself alive until its thread has ended, so the widget that
    started it can be deleted without taking a running thread with it.

    :param entry: the zoo's :class:`spacr.model_zoo.ModelEntry`.
    :param folder: where the checkpoint goes.
    :param parent: owner; give it none, for the reason above.
    :param unverified: accept an entry that publishes no checksum. Only for
        an entry that has none, and only after the user has said so.
    """

    #: ``(bytes so far, total bytes or 0)``.
    progressed = Signal(int, int)
    #: ``(worked, the local path or the reason it failed)``: once.
    finished = Signal(bool, str)

    def __init__(self, entry: Any, folder: str,
                 parent: Optional[QObject] = None, *,
                 unverified: bool = False):
        """Prepare the download without starting it.

        :param entry: the zoo entry.
        :param folder: destination folder.
        :param parent: owner.
        :param unverified: skip the checksum requirement.
        """
        super().__init__(parent)
        self.entry = entry
        self.folder = str(folder)
        self._unverified = bool(unverified)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_DownloadWorker] = None
        self._done = False

    def _fetch(self, *, progress, cancel):
        """The zoo call the worker thread makes."""
        from .. import model_zoo

        return model_zoo.install(self.entry, self.folder,
                                 require_checksum=not self._unverified,
                                 progress=progress, cancel=cancel)

    def start(self) -> bool:
        """Start the download on its own thread.

        :returns: True once the thread is running.
        """
        self._thread = QThread()
        self._worker = _DownloadWorker(self._fetch)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self.progressed)
        self._worker.done.connect(self._succeeded)
        self._worker.failed.connect(self._failed)
        self._thread.finished.connect(self._release)
        _RUNNING.add(self)
        self._thread.start()
        return True

    def is_running(self) -> bool:
        """Whether the download thread is still running."""
        return self._thread is not None and self._thread.isRunning()

    def wait(self, timeout_ms: int = 10_000) -> bool:
        """Block until the download thread has ended.

        :param timeout_ms: how long to wait.
        :returns: True when it has ended.
        """
        return self._thread is None or self._thread.wait(timeout_ms)

    def cancel(self) -> None:
        """Stop the download at its next chunk, and report it as cancelled.

        Returns at once: the thread ends on its own, and the partial file is
        removed by :func:`spacr.model_zoo.fetch`.
        """
        if self._worker is not None:
            self._worker.stop = True
        self._finish(False, "cancelled")

    def _succeeded(self, installed) -> None:
        """Report where the checkpoint was written."""
        self._thread.quit()
        self._finish(True, str(getattr(installed, "path", installed) or ""))

    def _failed(self, message: str) -> None:
        """Report why the download failed."""
        self._thread.quit()
        self._finish(False, message)

    def _release(self) -> None:
        """Let go of this download once its thread has ended."""
        _RUNNING.discard(self)

    def _finish(self, worked: bool, message: str) -> None:
        """Emit :attr:`finished` exactly once."""
        if self._done:
            return
        self._done = True
        self.finished.emit(bool(worked), str(message))


def human_bytes(size: Any) -> str:
    """A byte count a person can read, such as ``'1.2 GB'``.

    :param size: bytes.
    :returns: the text, or ``''`` for an unknown (zero) size.
    """
    value = float(size or 0)
    if value <= 0:
        return ""
    for unit in ("B", "kB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


class SegmentationBackendCombo(QComboBox):
    """The Mask module's ``segmentation_backend``: every backend, always.

    Cellpose is always first and always installed. DINOCell and SAMCell are
    listed whether or not their packages are here; one that is missing is
    greyed, with a tooltip saying so, and choosing it asks before running
    ``pip`` in the background. The box goes back to the backend it was on
    while the install runs, so the setting never names a backend that cannot
    load, and selects the new backend once it is installed.

    The stored value of each row is the setting's own value (``'cellpose'``,
    ``'dinocell'``, ``'samcell'``), so the settings form reads and writes it
    as it reads and writes any other dropdown.

    :param default: the value to start on.
    :param parent: parent widget.
    """

    #: ``(worked, message)`` when an install this box started has ended.
    install_finished = Signal(bool, str)

    def __init__(self, default: Any = "cellpose",
                 parent: Optional[QWidget] = None):
        """List the backends and select ``default``.

        :param default: the value to start on.
        :param parent: parent widget.
        """
        super().__init__(parent)
        self.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(12)
        self.addItem("Cellpose", "cellpose")
        self._modules = {}
        for name, label, _extra, module in backend_rows():
            self.addItem(label, name)
            self._modules[name] = module
        self.job: Optional[PackageInstall] = None
        self._installing: Optional[str] = None
        self._help = ""
        self._last = 0
        self.refresh_installed()
        self.setCurrentText(default)
        self._last = self.currentIndex()
        self.currentIndexChanged.connect(self._remember)
        self.activated.connect(self._on_activated)

    def missing(self) -> List[str]:
        """The backends listed but not installed, by name.

        Installed means "can segment now", which is either an
        environment of its own under ``~/.spacr/backends`` or -- the older
        arrangement, still honoured -- the package importable in spaCR's own
        environment. :func:`spacr._segmentation_backends._backend_state`
        answers both with file checks and no import, so this stays cheap
        enough to ask while the box is being built.
        """
        from .. import _segmentation_backends as backends

        out = []
        for name in self._modules:
            try:
                ready = backends._backend_state(name).ready
            except (KeyError, OSError, ValueError):
                ready = is_importable(self._modules[name])
            if not ready:
                out.append(name)
        return out

    def refresh_installed(self) -> None:
        """Grey the backends that are not installed, and only those."""
        from .i18n import tr

        missing = set(self.missing())
        for index in range(self.count()):
            name = self.itemData(index)
            if name in missing:
                self.setItemData(index, QBrush(UNINSTALLED_GREY),
                                 Qt.ForegroundRole)
                self.setItemData(index, tr(
                    "{name} is not installed. Choosing it offers to install "
                    "it.", name=self.itemText(index)), Qt.ToolTipRole)
            else:
                self.setItemData(index, None, Qt.ForegroundRole)
                self.setItemData(index, None, Qt.ToolTipRole)

    def setCurrentText(self, text: Any) -> None:                # noqa: N802
        """Select the row whose caption, or stored value, is ``text``.

        :param text: a caption such as ``'SAMCell'`` or a value such as
            ``'samcell'``; anything else leaves the selection alone.
        """
        wanted = "" if text is None else str(text).strip()
        index = self.findText(wanted)
        if index < 0:
            index = self.findData(wanted.lower())
        if index >= 0:
            self.setCurrentIndex(index)

    def _remember(self, index: int) -> None:
        """Keep the last installed row chosen, to go back to from a missing one.

        A click on a missing row changes the current row before ``activated``
        fires, so that row is not remembered: going back means going back to
        a backend that can run.
        """
        if index >= 0 and self.itemData(index) not in self.missing():
            self._last = index

    def _on_activated(self, index: int) -> None:
        """A person chose a row: offer the install if it is missing."""
        name = self.itemData(index)
        if name not in self.missing():
            return
        previous = self._previous_installed(index)
        self.setCurrentIndex(previous)
        self.offer_install(name)

    def _previous_installed(self, index: int) -> int:
        """The row to fall back to when ``index`` cannot be used yet."""
        missing = set(self.missing())
        for candidate in (self._last, 0):
            if candidate != index and self.itemData(candidate) not in missing:
                return candidate
        return 0

    def offer_install(self, name: str) -> bool:
        """Install backend ``name`` into an environment of its own.

        THE DESTINATION CHANGED, NOT THE GESTURE. A greyed row installs
        itself when it is chosen. It used to run
        ``pip install "spacr[<backend>]"`` against the environment spaCR is
        running in; each backend now gets an isolated environment, so the
        install goes through the Model Zoo's own dialog instead: off the GUI thread, with
        progress and Cancel, into ``~/.spacr/backends/<name>``, and spaCR's
        own environment is never changed.

        :param name: a backend name.
        :returns: True when the backend can segment afterwards.
        """
        from .widgets import model_zoo_picker

        row = backend_row(name)
        if row is None:
            return False
        worked = bool(model_zoo_picker.install_backend(self, name))
        self._installing = None
        self.refresh_installed()
        index = self.findData(name)
        if worked and index >= 0 and name not in self.missing():
            self.setCurrentIndex(index)
        self.install_finished.emit(worked, "")
        return worked


class SpotDetectorCombo(QComboBox):
    """The OPS module's ``ops_spot_detector``: spaCR's own, or SpotNet.

    spaCR's own detector is first, the default and always usable. SpotNet
    is listed whether or not it can run; when its environment or its
    DeepCell token is missing its row is disabled and its tooltip says
    which, and either way the tooltip states its non-commercial licence,
    because this box is where a person chooses it.

    :param default: the value to start on, ``'native'`` or ``'spotnet'``.
    :param parent: parent widget.
    :param readiness: ``() -> (ready, reason)``; SpotNet's own check when
        None, a stand-in in tests.
    """

    def __init__(self, default: Any = "native",
                 parent: Optional[QWidget] = None,
                 readiness: Optional[Callable[[], Tuple[bool, str]]] = None):
        """List the detectors, disable SpotNet if it cannot run, select."""
        from .i18n import tr

        super().__init__(parent)
        self._readiness = readiness
        self.addItem(tr("spaCR (native)"), "native")
        self.addItem(tr("SpotNet (DeepCell)"), "spotnet")
        self.setItemData(0, tr(
            "spaCR's own spot score, the detector this plate was validated "
            "with."), Qt.ToolTipRole)
        self.refresh()
        self.setCurrentText(default)

    def refresh(self) -> Tuple[bool, str]:
        """Enable SpotNet's row only when it can run, and say why not.

        :returns: SpotNet's ``(ready, reason)``.
        """
        from .i18n import tr

        try:
            if self._readiness is not None:
                ready, reason = self._readiness()
            else:
                from .._segmentation_backends import _spotnet_readiness
                ready, reason = _spotnet_readiness()
        except (KeyError, OSError, ValueError) as exc:
            ready, reason = False, str(exc)
        licence = tr(
            "SpotNet's models are licensed for NON-COMMERCIAL ACADEMIC USE "
            "ONLY, which is not spaCR's licence.")
        item = self.model().item(1)
        if item is not None:
            item.setEnabled(bool(ready))
        self.setItemData(1, licence if ready else f"{reason}\n\n{licence}",
                         Qt.ToolTipRole)
        if not ready:
            self.setItemData(1, QBrush(UNINSTALLED_GREY), Qt.ForegroundRole)
            if self.currentIndex() == 1:
                self.setCurrentIndex(0)
        else:
            self.setItemData(1, None, Qt.ForegroundRole)
        return bool(ready), reason

    def setCurrentText(self, text: Any) -> None:                # noqa: N802
        """Select the row whose caption or value is ``text``, if usable.

        :param text: a caption or stored value; a row that is not installed
            (disabled), or no match, leaves the selection alone.
        """
        wanted = "" if text is None else str(text).strip()
        index = self.findText(wanted)
        if index < 0:
            index = self.findData(wanted.lower())
        item = self.model().item(index) if index >= 0 else None
        if item is not None and item.isEnabled():
            self.setCurrentIndex(index)
