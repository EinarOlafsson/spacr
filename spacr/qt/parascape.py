"""Install ParaScape separately, then launch and monitor its desktop process."""
from __future__ import annotations

import threading

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)

from .. import _parascape as service
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


class ParaScapeInstallDialog(QDialog):
    """Explain the alpha app and download size before installing in its own environment.

    :param parent: owning organism page.
    :param root: optional external-application root.
    :param job: optional installer substitute for tests.
    """

    def __init__(self, parent=None, *, root=None, job=None, upgrade=False):
        """Build the consent and progress controls; opening the dialog installs nothing."""
        super().__init__(parent)
        self.root = service.apps_root(root)
        self.job = job or (service.upgrade_parascape if upgrade else service.install_parascape)
        self._thread = None
        self.installed = False
        self._closing = False
        self.setWindowTitle(tr("Install ParaScape (alpha)"))
        self.resize(scaled_px(660), scaled_px(540))
        layout = QVBoxLayout(self)
        self.explanation = QLabel(tr(
            "ParaScape is an alpha application for exploring the Toxoplasma knowledge map. "
            "Features and results may change. It opens in a separate window.\n\n"
            "The latest compatible stable ParaScape release is installed from PyPI. "
            "When gene data is missing, ParaScape asks before downloading it on first open. "
            "Dependencies and gene data add to the download and disk usage. "
            "Requirements vary with the version and platform. "
            "Allow at least 12 GB of free disk space. Downloads can take several minutes.\n\n"
            "ParaScape gets its own environment at {path}; spaCR's packages are not changed. "
            "The source is MIT licensed. A local Git checkout is an optional development source; "
            "local installs use committed files and leave the checkout untouched.",
            path=str(self.root / "parascape")), self)
        self.explanation.setWordWrap(True)
        self.explanation.setTextFormat(Qt.PlainText)
        layout.addWidget(self.explanation)
        label = QLabel(tr("ParaScape source"), self)
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
        from .widgets.eliding import ProgressLine

        self.progress = ProgressLine(self, detail=False)
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
        if upgrade:
            self.setWindowTitle(tr("Upgrade ParaScape"))
            self.explanation.setText(tr(
                "Upgrade ParaScape through pip from PyPI in its separate environment at {path}. "
                "Close any running ParaScape windows before upgrading. ParaScape is in alpha; "
                "downloads and disk usage depend on the release. When the new release needs gene data, "
                "ParaScape asks before downloading it on first open. spaCR's packages are not changed.",
                path=str(self.root / "parascape")))
            self.source.setText(service.PYPI_PACKAGE)
            self.source.hide()
            self.browse.hide()
            label.hide()
            self.start_button.setText(tr("Upgrade and open"))
            self.status.setText(tr("Ready to upgrade through pip."))

    def _choose_source(self):
        """Select a local ParaScape Git checkout without changing any files."""
        path = QFileDialog.getExistingDirectory(self, tr("Choose ParaScape checkout"))
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
        if total:
            self.progress.setFormat(tr("step {step} of {steps}",
                                       step=min(step + 1, total), steps=total))
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
            self.status.setText(tr("ParaScape is installed."))
        elif worker.outcome == "cancelled":
            self.status.setText(tr("ParaScape installation cancelled."))
        else:
            self.status.setText(tr("ParaScape installation failed. Review the details and try again."))
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
            self.status.setText(tr("Cancelling ParaScape installation…"))
        else:
            super().reject()

    def closeEvent(self, event):
        """Keep the dialog alive until its installer thread has stopped.

        :param event: Qt close event; ignored while cancellation is pending.
        :returns: None.
        """
        if self._thread is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


class ParaScapeUpdateCheckDialog(QDialog):
    """Check PyPI off the GUI thread, allowing a cancelled check to open the installed app."""

    def __init__(self, parent=None, *, root=None):
        """Prepare an automatic, read-only update check."""
        super().__init__(parent)
        self.result_data = None
        self.error = ""
        self._closing = False
        self.setWindowTitle(tr("Checking ParaScape updates"))
        layout = QVBoxLayout(self)
        self.status = QLabel(tr("Checking the installed ParaScape version and PyPI…"), self)
        layout.addWidget(self.status)
        progress = QProgressBar(self)
        progress.setRange(0, 0)
        progress.setTextVisible(False)
        layout.addWidget(progress)
        self.skip = QPushButton(tr("Open without checking"), self)
        self.skip.clicked.connect(self.reject)
        layout.addWidget(self.skip)

        def check(_source, *, root, cancel, progress):
            """Store the comparison until the worker's finished signal arrives."""
            self.result_data = service.check_parascape_update(root=root, cancel=cancel)

        self._thread = _InstallThread(None, service.apps_root(root), check, self)
        self._thread.finished.connect(self._finished)
        QTimer.singleShot(0, self._thread.start)

    def _finished(self):
        """Read worker results only after the subprocess runner has stopped."""
        worker, self._thread = self._thread, None
        self.error = worker.error
        worker.deleteLater()
        if self._closing:
            super().reject()
        else:
            super().accept()

    def reject(self):
        """Cancel the check and keep the dialog alive until its worker exits."""
        if self._thread is not None:
            self._closing = True
            self._thread.cancel.set()
            self.skip.setEnabled(False)
        else:
            super().reject()

    def closeEvent(self, event):
        """Prevent destruction of an active check thread.

        :param event: Qt close event; ignored while the check is being cancelled.
        :returns: None.
        """
        if self._thread is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)


def open_parascape(parent=None, *, root=None):
    """Check for updates on every open and offer pip upgrades before launching.

    :param parent: organism page requesting the launch.
    :param root: optional external-application folder.
    :returns: child process, or None if installation or a requested upgrade is
        cancelled or unsuccessful, or the initial launch fails. Skipping or
        cancelling only the update check still opens the installed app.
    """
    root = service.apps_root(root)
    if not service.is_installed(root):
        dialog = ParaScapeInstallDialog(parent, root=root, upgrade=service._can_repair(root))
        if dialog.exec() != QDialog.Accepted or not dialog.installed:
            return None
    check = ParaScapeUpdateCheckDialog(parent, root=root)
    if check.exec() == QDialog.Accepted:
        if check.error:
            QMessageBox.information(parent, tr("ParaScape update check unavailable"), tr(
                "The update check could not finish. Opening the installed version.\n\n{error}",
                error=check.error))
        elif check.result_data and check.result_data["available"]:
            result = check.result_data
            question = QMessageBox(parent)
            question.setWindowTitle(tr("ParaScape update available"))
            question.setText(tr(
                "ParaScape {installed} is installed. Version {latest} is available on PyPI. "
                "Upgrade through pip before opening?", **result))
            upgrade = question.addButton(tr("Upgrade"), QMessageBox.AcceptRole)
            later = question.addButton(tr("Not now"), QMessageBox.RejectRole)
            question.setDefaultButton(later)
            question.exec()
            if question.clickedButton() == upgrade:
                dialog = ParaScapeInstallDialog(parent, root=root, upgrade=True)
                dialog.start()
                if dialog.exec() != QDialog.Accepted or not dialog.installed:
                    return None
    try:
        process = service.launch_parascape(root=root)
    except (OSError, RuntimeError) as exc:
        QMessageBox.warning(parent, tr("ParaScape could not open"), str(exc))
        return None
    timer = QTimer(QApplication.instance())
    timer.setInterval(500)

    def check():
        """Reap the detached child and show failures without closing ParaScape on spaCR exit."""
        code = process.poll()
        if code is None:
            return
        timer.stop()
        timer.deleteLater()
        if code:
            QMessageBox.warning(QApplication.activeWindow(), tr("ParaScape could not open"), tr(
                "ParaScape exited with code {code}. See {path} for details.",
                code=code, path=str(root / "parascape-launch.log")))

    timer.timeout.connect(check)
    timer.start()
    return process
