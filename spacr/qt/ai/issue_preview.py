"""Editable, explicitly approved preview for a public GitHub report."""
from __future__ import annotations

from typing import Mapping

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from ..i18n import tr
from ..widgets import Toggle
from .issue_report import strip_report_paths


class IssuePreviewDialog(QDialog):
    """Show the exact public payload and require a Send click."""

    def __init__(self, report: Mapping[str, str], parent=None):
        super().__init__(parent)
        self._source_body = str(report.get("body", ""))
        self._fingerprint = str(report.get("fingerprint", ""))
        self.setWindowTitle(tr("Review public GitHub report"))
        self.setMinimumSize(760, 620)
        layout = QVBoxLayout(self)

        warning = QLabel(tr(
            "Nothing has been sent. The text below will be posted to the "
            "PUBLIC spaCR GitHub repository only if you press Send report. "
            "Public issues are world-readable, indexed and mirrored; they "
            "cannot be reliably unpublished. Edit or remove anything first."
        ))
        warning.setWordWrap(True)
        layout.addWidget(warning)

        self.title_edit = QLineEdit(str(report.get("title", "")))
        self.title_edit.setPlaceholderText(tr("Issue title"))
        layout.addWidget(self.title_edit)

        self.strip_paths = Toggle(tr(
            "Remove file and folder names (recommended)"
        ))
        self.strip_paths.setChecked(True)
        self.strip_paths.toggled.connect(self._refresh_body)
        layout.addWidget(self.strip_paths)

        self.body_edit = QTextEdit()
        self.body_edit.setAcceptRichText(False)
        self.body_edit.setPlainText(strip_report_paths(self._source_body))
        layout.addWidget(self.body_edit, 1)

        buttons = QDialogButtonBox()
        send = buttons.addButton(tr("Send report"), QDialogButtonBox.AcceptRole)
        cancel = buttons.addButton(tr("Cancel"), QDialogButtonBox.RejectRole)
        send.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_body(self, strip: bool) -> None:
        self.body_edit.setPlainText(
            strip_report_paths(self._source_body) if strip else self._source_body
        )

    def approved_report(self) -> dict[str, str]:
        """Return exactly the currently displayed, editable payload."""
        return {
            "title": self.title_edit.text().strip(),
            "body": self.body_edit.toPlainText(),
            "fingerprint": self._fingerprint,
        }
