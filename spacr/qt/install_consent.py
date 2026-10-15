"""Apply privacy and account choices collected by desktop installation.

Also the one-time notice that anonymised error reports are on by default
(:func:`maybe_show_anonymous_report_notice`) and the example of what such a
report holds (:class:`AnonymousReportExampleDialog`).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)

from ..install_profile import read_profile
from .i18n import tr
from .widgets.toggle import Toggle
from .prefs import _store_args


_ORG = "spacr"
_APP = "qt"
_KEY_APPLIED = "installer/consent_applied"


def _settings() -> QSettings:
    """Open spaCR's ``QSettings``.

    :returns: the settings store.
    """
    return QSettings(*_store_args(_ORG, _APP))


def _as_bool(value: Any) -> bool:
    """Read a stored value as a boolean.

    ``QSettings`` returns strings on some platforms and real booleans on
    others, so both are accepted -- anything unrecognised is ``False``,
    which is the safe answer for a consent flag.

    :param value: the stored value.
    :returns: whether it means yes.
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class InstallerConsentDialog(QDialog):
    """One first-launch page whose three optional choices start off.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        """Build the privacy and optional-account dialog.

        Sized in scaled pixels rather than raw ones: a size set from Python does
        not grow with the stylesheet's font size, and at the 200% scale the
        prose inside wrapped to more height than the window had.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("spaCR privacy and optional account setup"))
        from .preferences import scaled_px

        self.setMinimumWidth(scaled_px(640))
        layout = QVBoxLayout(self)
        explanation = QLabel(tr(
            "Crash reports go to the PUBLIC spaCR GitHub repository. They "
            "are world-readable, indexed, and cannot be reliably unpublished. "
            "A report is redacted, shown in an editable preview, and sent "
            "only when you press Send for that specific report. Account setup "
            "uses the official GitHub, Claude, Codex (GPT), and Gemini CLIs; "
            "spaCR does not store their passwords or tokens. All choices are "
            "optional and revocable in Preferences."
        ))
        explanation.setWordWrap(True)
        explanation.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Minimum)
        layout.addWidget(explanation)

        self.share_diagnostics = Toggle(tr(
            "Include redacted diagnostic logs in report previews"
        ))
        self.report_issues = Toggle(tr(
            "Enable the public GitHub issue-report action"
        ))
        self.sign_in_now = Toggle(tr(
            "Set up GitHub, Claude, GPT/Codex, and Gemini now"
        ))
        for choice in (
            self.share_diagnostics, self.report_issues, self.sign_in_now
        ):
            choice.setChecked(False)
            layout.addWidget(choice)

        buttons = QDialogButtonBox()
        save = buttons.addButton(
            tr("Save choices"), QDialogButtonBox.AcceptRole
        )
        skip = buttons.addButton(
            tr("Skip — keep all off"), QDialogButtonBox.RejectRole
        )
        save.clicked.connect(self.accept)
        skip.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def choices(self) -> dict[str, bool]:
        """Return the three explicit checkbox states."""
        return {
            "share_diagnostics": self.share_diagnostics.isChecked(),
            "report_issues": self.report_issues.isChecked(),
            "sign_in_now": self.sign_in_now.isChecked(),
        }


def apply_choices(choices: Mapping[str, Any]) -> bool:
    """Persist choices and return whether account setup was requested.

    :param choices: the consent answers, read by the keys
        ``share_diagnostics``, ``report_issues`` and ``sign_in_now``; a
        missing or unrecognised value counts as no.
    """
    from . import preferences
    from .ai import settings as ai_settings

    share = _as_bool(choices.get("share_diagnostics", False))
    reports = _as_bool(choices.get("report_issues", False))
    sign_in = _as_bool(choices.get("sign_in_now", False))
    preferences.set_share_diagnostic_logs(share)
    ai_settings.set_auto_file_issues(reports)
    preferences.set_issue_prompt_mode(
        preferences.ISSUE_PROMPT_ASK
        if reports else preferences.ISSUE_PROMPT_NEVER
    )
    return sign_in


def _open_account_setup(parent) -> None:
    """Open the AI providers dialog, for the optional account step.

    :param parent: the dialog to parent it to.
    """
    from .widgets.ai_chat_panel import _ProvidersDialog

    _ProvidersDialog(parent).exec()


def maybe_show_installer_consent(parent) -> bool:
    """Apply installer choices once, asking in-app if no page was shown.

    :param parent: the widget that parents the in-app consent dialog and, if
        sign-in was chosen, the account setup dialog.
    :returns: ``True`` when an installer profile was handled this call.
    """
    store = _settings()
    if _as_bool(store.value(_KEY_APPLIED, False)):
        return False
    profile = read_profile()
    if not profile:
        return False
    consent: Optional[Mapping[str, Any]] = profile.get("consent")
    if not isinstance(consent, Mapping):
        consent = {}

    if _as_bool(consent.get("collected", False)):
        choices = consent
    else:
        dialog = InstallerConsentDialog(parent)
        accepted = dialog.exec() == QDialog.Accepted
        choices = dialog.choices() if accepted else {}

    sign_in = apply_choices(choices)
    store.setValue(_KEY_APPLIED, True)
    store.sync()
    if sign_in:
        _open_account_setup(parent)
    return True


class AnonymousReportExampleDialog(QDialog):
    """Show exactly what an anonymised error report holds, with an example.

    The example is built by :func:`spacr.crashreport.example_public_summary`
    from a made-up error whose message names a file under the home folder,
    an e-mail address and a token, so the dialog shows each one removed.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        """Build the explanation, the example and a Close button.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        import json

        from PySide6.QtWidgets import QPlainTextEdit

        from .. import crashreport
        from .preferences import scaled_px

        self.setObjectName("AnonymousReportExample")
        self.setWindowTitle(tr("What an anonymised error report holds"))
        self.setMinimumWidth(scaled_px(560))
        layout = QVBoxLayout(self)
        explanation = QLabel(tr(
            "Each report is a public GitHub issue on EinarOlafsson/spacr, "
            "filed by spaCR's relay, so you need no GitHub account. It holds "
            "only the fields below. The message is scrubbed of paths, file "
            "names, user and host names, e-mail and IP addresses and tokens, "
            "and the call stack names files without their folders. Logs, "
            "settings, environment variables and images are never sent. The "
            "random install id is not derived from this computer; it only "
            "lets the relay share its free allowance fairly. Below is the "
            "report spaCR would send for a made-up error."
        ))
        explanation.setWordWrap(True)
        explanation.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(explanation)
        try:
            example = json.dumps(crashreport.example_public_summary(),
                                 indent=2, ensure_ascii=False)
        except Exception as exc:
            example = f"{type(exc).__name__}: {exc}"
        self.example = QPlainTextEdit(example)
        self.example.setObjectName("AnonymousReportExampleText")
        self.example.setReadOnly(True)
        self.example.setMinimumHeight(scaled_px(260))
        layout.addWidget(self.example)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AnonymousReportNotice(QDialog):
    """The one-time notice that anonymised error reports are on by default.

    Not modal: it never stands between the user and the window. Closing it
    saves the switch as it is then.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        """Build the notice, its switch and its two buttons.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        from .. import crashreport
        from .preferences import scaled_px

        self.setObjectName("AnonymousReportNotice")
        self.setWindowTitle(tr("Anonymised error reports"))
        self.setMinimumWidth(scaled_px(520))
        self.setModal(False)
        layout = QVBoxLayout(self)
        explanation = QLabel(tr(
            "spaCR sends an anonymised summary when a run fails, is refused "
            "for its settings, or spaCR crashes, so problems get fixed without you having to report "
            "them. This is on by default. The summary holds versions, the "
            "operating system, the GPU type, the module, the error type, a "
            "scrubbed message and the call stack without paths. It goes "
            "through spaCR's relay to a public GitHub issue on "
            "EinarOlafsson/spacr, up to {reports} reports a day. Each report "
            "includes an AI-written summary of the likely cause, made with "
            "your own AI when one is connected, and the AI also explains every "
            "error in the console. You can turn reports off here, or choose "
            "what is explained and sent under Preferences → AI → AI Error "
            "report.\n\n"
            "spaCR AI is free with no account, within a daily share of "
            "spaCR's free capacity: about {best} questions a day on the best "
            "model, then {light} more on a lighter model. Connect your own "
            "Claude, Codex or Gemini CLI for unlimited use.",
            reports=crashreport.REPORTS_PER_DAY,
            best=crashreport.AI_BEST_PER_DAY,
            light=crashreport.AI_LIGHT_PER_DAY,
        ))
        explanation.setWordWrap(True)
        explanation.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(explanation)
        self.send_reports = Toggle(tr("Send anonymised error reports"))
        self.send_reports.setObjectName("AnonymousReportsNoticeToggle")
        self.send_reports.setChecked(crashreport.anonymous_reports_enabled())
        layout.addWidget(self.send_reports)
        buttons = QDialogButtonBox()
        what = buttons.addButton(tr("What is sent…"),
                                 QDialogButtonBox.HelpRole)
        done = buttons.addButton(tr("OK"), QDialogButtonBox.AcceptRole)
        what.clicked.connect(
            lambda: AnonymousReportExampleDialog(self).exec())
        done.clicked.connect(self.accept)
        layout.addWidget(buttons)
        self.finished.connect(self._save_choice)

    def _save_choice(self, _result: int = 0) -> None:
        """Save the switch as it stands when the notice closes.

        :param _result: the dialog result; the switch decides, not the
            button.
        """
        from .. import crashreport

        try:
            crashreport.set_anonymous_reports_enabled(
                self.send_reports.isChecked())
        except OSError:
            pass


def maybe_show_anonymous_report_notice(parent) -> Optional[QDialog]:
    """Show the anonymised-report notice once per profile, without blocking.

    Nothing is shown while the network is forbidden
    (:func:`spacr.crashreport.network_allowed`), since no report could be
    sent; the notice then waits for a start that allows it. Waiting reports
    are also given another try, on a background thread.

    :param parent: the window that parents the notice.
    :returns: the notice, already showing, or ``None`` when it was not due.
    """
    import threading

    from .. import crashreport

    try:
        if not crashreport.network_allowed():
            return None
        if crashreport.anonymous_reports_enabled() and \
                crashreport.queued_summaries():
            threading.Thread(target=crashreport.flush_report_queue,
                             name="spacr-report-queue", daemon=True).start()
        if not crashreport.anonymous_report_notice_due():
            return None
        crashreport.mark_anonymous_report_notice_shown()
    except OSError:
        return None
    notice = AnonymousReportNotice(parent)
    notice.setAttribute(Qt.WA_DeleteOnClose, True)
    notice.show()
    return notice
