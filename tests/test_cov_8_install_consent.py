"""Installer choices are applied once, whatever shape the profile is in.

The desktop installers write ``consent`` into a JSON profile that spaCR
reads on first launch. That file is produced outside the application, so
:func:`spacr.qt.install_consent.maybe_show_installer_consent` cannot assume
it holds a mapping -- a hand-edited or half-written profile has to fall back
to asking in-app rather than crashing the launch. The same function is also
the only door to the provider setup dialog, and it must mark itself applied
BEFORE opening it so a failure there cannot make the privacy page reappear.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog          # noqa: E402

from spacr.qt import install_consent as ic     # noqa: E402


class _FakeConsentDialog:
    """Stands in for the first-launch page: accepted, everything on."""

    instances = []

    def __init__(self, parent=None):
        self.parent = parent
        type(self).instances.append(self)

    def exec(self):
        return QDialog.Accepted

    def choices(self):
        return {"share_diagnostics": True, "report_issues": True,
                "sign_in_now": True}


class _FakeProvidersDialog:
    """Records that account setup was opened, and for which parent."""

    opened = []

    def __init__(self, parent=None):
        self.parent = parent

    def exec(self):
        type(self).opened.append(self.parent)
        return QDialog.Accepted


@pytest.mark.parametrize("value,expected", [
    ("yes", True), ("ON", True), (" 1 ", True), ("true", True),
    ("no", False), ("0", False), ("", False), (None, False), (2, False),
])
def test_a_written_out_flag_reads_back_as_the_boolean_it_meant(value, expected):
    """QSettings and JSON both hand back strings; both must mean the flag."""
    assert ic._as_bool(value) is expected


def test_a_flag_that_is_already_a_bool_is_passed_straight_through():
    """The string path must not be able to change a real boolean."""
    assert ic._as_bool(True) is True
    assert ic._as_bool(False) is False


@pytest.fixture
def _fakes(monkeypatch):
    """Swap both modal dialogs for recorders and clear their ledgers."""
    import spacr.qt.widgets.ai_chat_panel as chat_panel

    _FakeConsentDialog.instances = []
    _FakeProvidersDialog.opened = []
    monkeypatch.setattr(ic, "InstallerConsentDialog", _FakeConsentDialog)
    monkeypatch.setattr(chat_panel, "_ProvidersDialog", _FakeProvidersDialog)
    return _FakeConsentDialog, _FakeProvidersDialog


def test_a_profile_whose_consent_is_not_a_mapping_asks_in_app(
        _fakes, monkeypatch, qapp):
    """A malformed ``consent`` block falls back to the page, not a crash."""
    consent_dialog, providers = _fakes
    monkeypatch.setattr(ic, "read_profile",
                        lambda: {"consent": ["share_diagnostics"]})

    handled = ic.maybe_show_installer_consent(None)

    assert handled is True
    assert len(consent_dialog.instances) == 1
    assert providers.opened == [None]


def test_a_collected_profile_is_applied_without_showing_the_page(
        _fakes, monkeypatch, qapp):
    """An installer that already asked must not ask the user twice."""
    consent_dialog, providers = _fakes
    monkeypatch.setattr(ic, "read_profile", lambda: {
        "consent": {"collected": "true", "share_diagnostics": "false",
                    "report_issues": "false", "sign_in_now": "false"}})

    handled = ic.maybe_show_installer_consent(None)

    assert handled is True
    assert consent_dialog.instances == []
    assert providers.opened == []
    assert ic.maybe_show_installer_consent(None) is False


def test_no_installer_profile_means_nothing_to_apply(_fakes, monkeypatch):
    """A pip install has no profile; the launch must simply carry on."""
    monkeypatch.setattr(ic, "read_profile", lambda: {})

    assert ic.maybe_show_installer_consent(None) is False
