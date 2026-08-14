"""Apply privacy and account choices collected by desktop installation."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

from ..install_profile import read_profile
from .i18n import tr
from .widgets.toggle import Toggle


_ORG = "spacr"
_APP = "qt"
_KEY_APPLIED = "installer/consent_applied"


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class InstallerConsentDialog(QDialog):
    """One first-launch page whose three optional choices start off."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("spaCR privacy and optional account setup"))
        self.setMinimumWidth(640)
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
    """Persist choices and return whether account setup was requested."""
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
    from .widgets.ai_chat_panel import _ProvidersDialog

    _ProvidersDialog(parent).exec()


def maybe_show_installer_consent(parent) -> bool:
    """Apply installer choices once, asking in-app if no page was shown.

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

    # Mark before opening another modal. If that dialog or a vendor CLI fails,
    # the privacy page must not reappear and rewrite the user's choices.
    sign_in = apply_choices(choices)
    store.setValue(_KEY_APPLIED, True)
    store.sync()
    if sign_in:
        _open_account_setup(parent)
    return True
