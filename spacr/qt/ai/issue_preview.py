"""Editable, explicitly approved preview for a public GitHub report."""
from __future__ import annotations

from typing import Mapping

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
)

from ..i18n import tr
from ..widgets.toggle import Toggle
from .issue_report import AI_ANALYSIS_MAX_CHARS, sanitize_path, strip_report_paths

#: How often the dialog checks whether spaCR AI has finished.
#:
#: Polled rather than wired to a signal because the console owns the AI
#: conversation and this dialog is a guest in it: an answer can arrive from a
#: stream this dialog never started -- one the user set going from the console
#: before opening the report -- and a poll notices that just as readily as it
#: notices its own.
DIAGNOSE_POLL_MS = 300

#: Given up on after this long, so the button cannot spin for ever if the
#: provider never answers and never errors.
DIAGNOSE_TIMEOUT_MS = 180_000


class IssuePreviewDialog(QDialog):
    """Show the exact public payload and require a Send click."""

    def __init__(self, report: Mapping[str, str], parent=None,
                 console=None, traceback_text: str = ""):
        super().__init__(parent)
        self._source_body = str(report.get("body", ""))
        self._fingerprint = str(report.get("fingerprint", ""))
        # The console owns the AI conversation; this dialog borrows it. Taken
        # from the parent when not passed, so the screen that opens this does
        # not have to know about the button.
        self._console = console if console is not None else getattr(
            parent, "_console", None)
        self._traceback_text = str(traceback_text or "")
        self._diagnose_elapsed = 0
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
        # ASK spaCR AI FROM HERE. The report is the moment the user is looking
        # hardest at the error, and it is also the last moment before it goes
        # somewhere public -- a diagnosis is worth having in both directions:
        # it may save the report entirely, and it makes the report better if
        # it does not.
        self.diagnose_btn = buttons.addButton(
            tr("Diagnose"), QDialogButtonBox.ActionRole)
        self.diagnose_btn.setToolTip(tr(
            "Ask spaCR AI to explain this error and add its analysis to the "
            "report."))
        self.diagnose_btn.clicked.connect(self._on_diagnose)
        send.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        layout.addWidget(buttons)
        self._diagnose_timer = QTimer(self)
        self._diagnose_timer.setInterval(DIAGNOSE_POLL_MS)
        self._diagnose_timer.timeout.connect(self._check_for_diagnosis)

    def _refresh_body(self, strip: bool) -> None:
        self.body_edit.setPlainText(
            strip_report_paths(self._source_body) if strip else self._source_body
        )

    # -- Diagnose ---------------------------------------------------------

    def _tell(self, title: str, message: str) -> None:
        """Say something to the user. One seam, so tests can listen."""
        QMessageBox.information(self, title, message)

    def _on_diagnose(self) -> None:
        """Ask spaCR AI about this error, or say why that cannot happen yet.

        The four cases, in the order they are decided:

        * an analysis is already in hand -- show it, which is the whole point
          of pressing the button a second time;
        * no provider configured -- the AI cannot be asked at all, so say so
          and point at where an account is linked;
        * a stream already running -- it may be this error or another, but
          either way a second request would queue behind it, so ask the user
          to wait rather than appearing to do nothing;
        * otherwise, ask.

        Whether the AI TOGGLE is on is deliberately not consulted. The toggle
        governs whether spaCR volunteers an explanation; pressing a button
        named Diagnose is asking for one outright.
        """
        console = self._console
        if console is None:
            self._tell(tr("Diagnose"),
                       tr("The AI console is not available from here."))
            return

        existing = self._existing_diagnosis()
        if existing:
            self._show_diagnosis(existing)
            return

        if getattr(console, "_current_provider", None) is None or \
                console._current_provider() is None:
            self._tell(
                tr("No AI account linked"),
                tr("spaCR AI needs a provider before it can explain an "
                   "error. Link one through the AI Providers dialog — the "
                   "command palette has 'Open AI Providers…' — then press "
                   "Diagnose again.\n\nThe report can be sent without a "
                   "diagnosis."))
            return

        if getattr(console, "_ai_thread", None) is not None:
            self._tell(
                tr("spaCR AI is still working"),
                tr("spaCR AI has not finished its previous answer. Wait for "
                   "it to finish, then press Diagnose again — the analysis "
                   "will be added to this report."))
            return

        if not self._traceback_text:
            self._tell(tr("Diagnose"),
                       tr("There is no error text to diagnose."))
            return

        self.diagnose_btn.setEnabled(False)
        self.diagnose_btn.setText(tr("Diagnosing…"))
        self._diagnose_elapsed = 0
        try:
            console.open_error_flow(self._traceback_text, show_raw=False)
        except Exception:                                    # noqa: BLE001
            self._end_diagnosing()
            self._tell(tr("Diagnose"),
                       tr("spaCR AI could not be started."))
            return
        self._diagnose_timer.start()

    def _existing_diagnosis(self) -> str:
        """spaCR AI's answer about THIS error, if it already has one."""
        console = self._console
        if console is None or not self._traceback_text:
            return ""
        try:
            return console.ai_explanation_of(self._traceback_text) or ""
        except Exception:                                    # noqa: BLE001
            return ""

    def _check_for_diagnosis(self) -> None:
        """Poll for the answer, and give up rather than spin for ever."""
        self._diagnose_elapsed += DIAGNOSE_POLL_MS
        answer = self._existing_diagnosis()
        if answer:
            self._end_diagnosing()
            self._show_diagnosis(answer)
            return
        console = self._console
        finished = getattr(console, "_ai_thread", None) is None
        if finished and self._diagnose_elapsed > DIAGNOSE_POLL_MS * 2:
            # The stream ended without an answer for this error -- a provider
            # error, or a reply the console did not pair with this traceback.
            self._end_diagnosing()
            self._tell(tr("Diagnose"),
                       tr("spaCR AI did not return an analysis. The console "
                          "shows what it said."))
            return
        if self._diagnose_elapsed >= DIAGNOSE_TIMEOUT_MS:
            self._end_diagnosing()
            self._tell(tr("Diagnose"),
                       tr("spaCR AI did not answer in time."))

    def _end_diagnosing(self) -> None:
        self._diagnose_timer.stop()
        self.diagnose_btn.setEnabled(True)
        self.diagnose_btn.setText(tr("Diagnose"))

    def _show_diagnosis(self, analysis: str) -> None:
        """Put the analysis in the report and scroll the user to it.

        Appended rather than rebuilt through `build_report`: by the time this
        runs the user may have edited the body, and regenerating it would
        throw their edits away to add a paragraph.
        """
        text = sanitize_path(str(analysis or "")).strip()
        if not text:
            return
        if len(text) > AI_ANALYSIS_MAX_CHARS:
            text = text[:AI_ANALYSIS_MAX_CHARS].rstrip() + "\n\n… (truncated)"
        section = (
            "\n\n<details><summary>spaCR AI's analysis of this error"
            "</summary>\n\n"
            "Generated by spaCR AI from the traceback above, unreviewed. "
            "Treat it as a lead rather than a diagnosis.\n\n"
            + text + "\n</details>\n")
        if "spaCR AI's analysis of this error" in self._source_body:
            self._scroll_to_diagnosis()
            return
        # Into the SOURCE too, so the strip toggle does not drop it:
        # `_refresh_body` rebuilds the box from `_source_body`.
        self._source_body += section
        self.body_edit.setPlainText(
            strip_report_paths(self._source_body)
            if self.strip_paths.isChecked() else self._source_body)
        self._scroll_to_diagnosis()

    def _scroll_to_diagnosis(self) -> None:
        """Move the caret and the view to the analysis."""
        document = self.body_edit.document()
        cursor = document.find("spaCR AI's analysis of this error")
        if cursor.isNull():
            return
        self.body_edit.setTextCursor(cursor)
        self.body_edit.ensureCursorVisible()

    def approved_report(self) -> dict[str, str]:
        """Return exactly the currently displayed, editable payload."""
        return {
            "title": self.title_edit.text().strip(),
            "body": self.body_edit.toPlainText(),
            "fingerprint": self._fingerprint,
        }
