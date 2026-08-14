"""Desktop-installer consent remains opt-in and is applied only once."""
from __future__ import annotations


def _profile(**choices):
    return {
        "schema": 1,
        "requested_backend": "cpu",
        "active_backend": "cpu",
        "consent": {"collected": True, **choices},
    }


def test_fresh_dialog_starts_with_every_choice_off(qtbot):
    from spacr.qt.install_consent import InstallerConsentDialog

    dialog = InstallerConsentDialog()
    qtbot.addWidget(dialog)
    assert dialog.choices() == {
        "share_diagnostics": False,
        "report_issues": False,
        "sign_in_now": False,
    }


def test_collected_installer_choices_reach_runtime_preferences(
    qtbot, monkeypatch
):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import install_consent, preferences
    from spacr.qt.ai import settings as ai_settings

    parent = QWidget()
    qtbot.addWidget(parent)
    install_consent._settings().remove("installer/consent_applied")
    monkeypatch.setattr(
        install_consent,
        "read_profile",
        lambda: _profile(
            share_diagnostics=True,
            report_issues=True,
            sign_in_now=False,
        ),
    )

    assert install_consent.maybe_show_installer_consent(parent) is True
    assert preferences.get_share_diagnostic_logs() is True
    assert ai_settings.get_auto_file_issues() is True
    assert preferences.get_issue_prompt_mode() == preferences.ISSUE_PROMPT_ASK
    assert install_consent.maybe_show_installer_consent(parent) is False


def test_declining_uncollected_page_keeps_everything_off(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog, QWidget
    from spacr.qt import install_consent, preferences
    from spacr.qt.ai import settings as ai_settings

    parent = QWidget()
    qtbot.addWidget(parent)
    install_consent._settings().remove("installer/consent_applied")
    monkeypatch.setattr(
        install_consent,
        "read_profile",
        lambda: _profile(collected=False),
    )

    class Declined:
        def __init__(self, _parent): pass
        def exec(self): return QDialog.Rejected
        def choices(self): raise AssertionError("declined choices were read")

    monkeypatch.setattr(install_consent, "InstallerConsentDialog", Declined)
    assert install_consent.maybe_show_installer_consent(parent) is True
    assert preferences.get_share_diagnostic_logs() is False
    assert ai_settings.get_auto_file_issues() is False
    assert preferences.get_issue_prompt_mode() == preferences.ISSUE_PROMPT_NEVER


def test_sign_in_choice_opens_existing_provider_setup(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import install_consent

    parent = QWidget()
    qtbot.addWidget(parent)
    install_consent._settings().remove("installer/consent_applied")
    monkeypatch.setattr(
        install_consent,
        "read_profile",
        lambda: _profile(sign_in_now=True),
    )
    opened = []
    monkeypatch.setattr(
        install_consent, "_open_account_setup", lambda owner: opened.append(owner)
    )
    install_consent.maybe_show_installer_consent(parent)
    assert opened == [parent]
