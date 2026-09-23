"""Install Starplast separately, then launch and monitor its desktop process."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)

from .. import _starplast as service
from .._segmentation_backends import _InstallCancelled
from .i18n import tr
from .preferences import scaled_px


class _InstallThread(QThread):
    """Run the cancellable installer without importing PyQt6 into spaCR."""

    progressed = Signal(int, int, str)

    def __init__(self, source, root, job, parent):
        """Keep the selected source and outcome until the thread has finished."""
        super().__init__(parent)
        self.source, self.root, self.job = source, root, job
        self.cancel = threading.Event()
        self.outcome = ""
        self.error = ""

    def run(self):
        """Store an outcome; Qt's finished signal delivers it after the thread exits."""
        try:
            self.job(self.source, root=self.root, cancel=self.cancel,
                     progress=lambda step, total, text: self.progressed.emit(step, total, text))
        except _InstallCancelled:
            self.outcome = "cancelled"
        except Exception as exc:
            self.outcome, self.error = "failed", str(exc)
        else:
            self.outcome = "installed"


class StarplastInstallDialog(QDialog):
    """Explain the alpha app and download size before installing in its own environment.

    :param parent: owning organism page.
    :param root: optional external-application root.
    :param job: optional installer substitute for tests.
    """

    def __init__(self, parent=None, *, root=None, job=None):
        """Build the consent and progress controls; opening the dialog installs nothing."""
        super().__init__(parent)
        self.root = service.apps_root(root)
        self.job = job or service.install_starplast
        self._thread = None
        self.installed = False
        self._closing = False
        self.setWindowTitle(tr("Install Starplast (alpha)"))
        self.resize(scaled_px(660), scaled_px(540))
        layout = QVBoxLayout(self)
        self.explanation = QLabel(tr(
            "Starplast is an alpha application for exploring the Toxoplasma knowledge map. "
            "Features and results may change. It opens in a separate window.\n\n"
            "This is a large installation: Linux x86_64 dependencies include approximately "
            "2 GB of CUDA wheels, plus the application, bundled data and other dependencies. "
            "Allow at least 12 GB of free disk space. Downloads can take several minutes.\n\n"
            "Starplast gets its own environment at {path}; spaCR's packages are not changed. "
            "The source is MIT licensed. Choose a local Git checkout if repository access "
            "is unavailable; local installs use committed files and leave the checkout untouched.",
            path=str(self.root / "starplast")), self)
        self.explanation.setWordWrap(True)
        self.explanation.setTextFormat(Qt.PlainText)
        layout.addWidget(self.explanation)
        label = QLabel(tr("Starplast source"), self)
        layout.addWidget(label)
        row = QHBoxLayout()
        self.source = QLineEdit(service.default_source(), self)
        self.source.setAccessibleName(label.text())
        label.setBuddy(self.source)
        row.addWidget(self.source, 1)
        self.browse = QPushButton(tr("Choose checkout…"), self)
        self.browse.clicked.connect(self._choose_source)
        row.addWidget(self.browse)
        layout.addLayout(row)
        self.status = QLabel(tr("Ready to install when you choose Install and open."), self)
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.progress = QProgressBar(self)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(400)
        layout.addWidget(self.details, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start_button = QPushButton(tr("Install and open"), self)
        self.cancel_button = QPushButton(tr("Cancel"), self)
        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def _choose_source(self):
        """Select a local Starplast Git checkout without changing any files."""
        path = QFileDialog.getExistingDirectory(self, tr("Choose Starplast checkout"))
        if path:
            self.source.setText(path)

    def start(self):
        """Start the install after the explicit button press, keeping Qt responsive."""
        if self._thread is not None:
            return
        self._closing = False
        self.details.clear()
        self.source.setEnabled(False)
        self.browse.setEnabled(False)
        self.start_button.setEnabled(False)
        self.cancel_button.setText(tr("Cancel"))
        self.progress.setRange(0, 0)
        self.progress.show()
        self._thread = _InstallThread(self.source.text(), self.root, self.job, self)
        self._thread.progressed.connect(self._progress)
        self._thread.finished.connect(self._finished)
        self._thread.start()

    def _progress(self, step, total, text):
        """Display command output as plain text while retaining a bounded history."""
        self.progress.setRange(0, total)
        self.progress.setValue(step)
        label, separator, output = text.partition(": ")
        self.status.setText((tr(label) + separator + output)[:300])
        self.details.appendPlainText(text)

    def _finished(self):
        """Offer retry after failure or close once installation/cancellation ends."""
        worker = self._thread
        self._thread = None
        self.progress.hide()
        self.source.setEnabled(True)
        self.browse.setEnabled(True)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText(tr("Close"))
        self.installed = worker.outcome == "installed"
        if self.installed:
            self.status.setText(tr("Starplast is installed."))
        elif worker.outcome == "cancelled":
            self.status.setText(tr("Starplast installation cancelled."))
        else:
            self.status.setText(tr("Starplast installation failed. Review the details and try again."))
            self.details.appendPlainText(worker.error)
            self.start_button.setText(tr("Try again"))
        worker.deleteLater()
        if self._closing:
            super().reject()
        elif self.installed:
            super().accept()

    def reject(self):
        """Cancel active subprocesses and defer closing until the worker has ended."""
        if self._thread is not None:
            self._closing = True
            self._thread.cancel.set()
            self.cancel_button.setEnabled(False)
            self.status.setText(tr("Cancelling Starplast installation…"))
        else:
            super().reject()

    def closeEvent(self, event):
        """Keep the dialog alive until its installer thread has stopped."""
        if self._thread is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


def open_starplast(parent=None, *, root=None):
    """Offer installation once, then launch the independent Starplast application.

    :param parent: organism page requesting the launch.
    :param root: optional external-application folder.
    :returns: child process, or None if cancelled or launch failed.
    """
    root = service.apps_root(root)
    if not service.is_installed(root):
        dialog = StarplastInstallDialog(parent, root=root)
        if dialog.exec() != QDialog.Accepted or not dialog.installed:
            return None
    try:
        process = service.launch_starplast(root=root)
    except (OSError, RuntimeError) as exc:
        QMessageBox.warning(parent, tr("Starplast could not open"), str(exc))
        return None
    timer = QTimer(QApplication.instance())
    timer.setInterval(500)

    def check():
        """Reap the detached child and show failures without closing Starplast on spaCR exit."""
        code = process.poll()
        if code is None:
            return
        timer.stop()
        timer.deleteLater()
        if code:
            QMessageBox.warning(QApplication.activeWindow(), tr("Starplast could not open"), tr(
                "Starplast exited with code {code}. See {path} for details.",
                code=code, path=str(root / "starplast-launch.log")))

    timer.timeout.connect(check)
    timer.start()
    return process
