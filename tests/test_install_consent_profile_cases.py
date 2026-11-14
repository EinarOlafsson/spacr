"""The installer's privacy profile is applied once, whatever shape it is in.

The desktop installers write ``consent`` into a JSON profile read on first
launch.  :func:`spacr.qt.install_consent.maybe_show_installer_consent` applies
choices the installer already collected without asking again, asks in-app when
the profile holds no usable answers (a hand-edited, non-mapping ``consent``
included), and does nothing for a pip install with no profile.  Every handled
profile latches, so the second launch changes nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog  # noqa: E402

from spacr.qt import install_consent as ic  # noqa: E402


class _ConsentPage:
    """Stands in for the in-app consent page; answers as the case says."""

    instances: list = []
    answer = QDialog.Accepted

    def __init__(self, parent=None):
        self.parent = parent
        type(self).instances.append(self)

    def exec(self):
        return type(self).answer

    def choices(self):
        return {"share_diagnostics": True, "report_issues": True,
                "sign_in_now": True}


class _ProvidersDialog:
    """Records any provider sign-in dialog; setup no longer opens one."""

    opened: list = []

    def __init__(self, parent=None):
        self.parent = parent

    def exec(self):
        type(self).opened.append(self.parent)
        return QDialog.Accepted


def test_maybe_show_installer_consent_applies_each_profile_shape_once(
        monkeypatch, tmp_path, qapp, subtests):
    """Each profile shape gives the exact page use, persisted choices and latch."""
    import spacr.qt.widgets.ai_chat_panel as chat_panel
    from spacr.qt import preferences
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ic, "InstallerConsentDialog", _ConsentPage)
    monkeypatch.setattr(chat_panel, "_ProvidersDialog", _ProvidersDialog)
    applied = []
    monkeypatch.setattr(preferences, "set_share_diagnostic_logs",
                        lambda on: applied.append(("share", on)))
    monkeypatch.setattr(ai_settings, "set_auto_file_issues",
                        lambda on: applied.append(("reports", on)))
    monkeypatch.setattr(preferences, "set_issue_prompt_mode",
                        lambda mode: applied.append(("prompt", mode)))
    ask, never = preferences.ISSUE_PROMPT_ASK, preferences.ISSUE_PROMPT_NEVER

    # case, profile, page answer, handled, pages shown, persisted choices
    cases = [
        ("no profile (pip install)", {}, QDialog.Accepted, False, 0, []),
        ("unreadable profile", None, QDialog.Accepted, False, 0, []),
        ("consent is not a mapping", {"consent": ["share_diagnostics"]},
         QDialog.Accepted, True, 1,
         [("share", True), ("reports", True), ("prompt", ask)]),
        ("no consent block", {"schema": 1}, QDialog.Accepted, True, 1,
         [("share", True), ("reports", True), ("prompt", ask)]),
        ("page declined", {"consent": {}}, QDialog.Rejected, True, 1,
         [("share", False), ("reports", False), ("prompt", never)]),
        ("installer collected, all off",
         {"consent": {"collected": "true", "share_diagnostics": "false",
                      "report_issues": "false", "sign_in_now": "false"}},
         QDialog.Accepted, True, 0,
         [("share", False), ("reports", False), ("prompt", never)]),
        ("installer collected, mixed",
         {"consent": {"collected": "yes", "share_diagnostics": "on",
                      "report_issues": "0"}},
         QDialog.Accepted, True, 0,
         [("share", True), ("reports", False), ("prompt", never)]),
        ("installer collected, reports on",
         {"consent": {"collected": True, "report_issues": "1"}},
         QDialog.Accepted, True, 0,
         [("share", False), ("reports", True), ("prompt", ask)]),
    ]
    for index, (case, profile, answer, handled, pages, persisted) in \
            enumerate(cases):
        with subtests.test(msg=case):
            # A fresh portable QSettings store per case: the latch starts unset.
            monkeypatch.setenv("SPACR_PORTABLE", str(tmp_path / f"p{index}"))
            monkeypatch.setenv("SPACR_HOME", str(tmp_path / f"h{index}"))
            monkeypatch.setattr(ic, "read_profile", lambda p=profile: p)
            _ConsentPage.instances = []
            _ConsentPage.answer = answer
            _ProvidersDialog.opened = []
            applied.clear()

            assert ic.maybe_show_installer_consent("parent") is handled
            assert len(_ConsentPage.instances) == pages
            assert all(page.parent == "parent"
                       for page in _ConsentPage.instances)
            assert applied == persisted
            assert _ProvidersDialog.opened == []
            # The latch: the same profile on the next launch is not applied.
            applied.clear()
            assert ic.maybe_show_installer_consent("parent") is False
            assert applied == []
            assert len(_ConsentPage.instances) == pages
            assert ic._as_bool(ic._settings().value(ic._KEY_APPLIED, False)) \
                is handled
