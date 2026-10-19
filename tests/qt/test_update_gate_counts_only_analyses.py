"""The update check waits only for work an update would break.

Reported natively on a Mac 2026-10-19 (1.5.1.5): after a fresh .pkg install
the first launch showed What's new and the installer cleanup dialog, and Help >
Check for updates then said "Finish or stop the active analyses" with nothing
running. RunRegistry.is_busy counts every handle, including read-only
housekeeping and the model downloads and package installs listed in the Jobs
window. The check now refuses for an analysis run with the existing message,
for a package install into this environment and for a model download with
their own messages, and never for housekeeping.
"""
import threading

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from spacr import updater
from spacr.qt import app as qt_app
from spacr.qt import bridge, install_consent

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qtbot, qt_theme_applied, monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / ".spacr"))
    w = qt_app.MainWindow()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def answers(monkeypatch):
    seen = {"info": [], "asked": []}
    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(QMessageBox, "information",
                        lambda _parent, _title, text, *a, **k: seen["info"].append(text))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda _parent, _title, text, *a, **k: seen["info"].append(text))

    def question(_parent, title, text, *a, **k):
        seen["asked"].append(title)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", question)
    return seen


def _check(window, answers):
    answers["info"].clear()
    answers["asked"].clear()
    window._on_update_check_done(updater.UpdateInfo("1.5.1.5", "1.5.1.6", None))
    if answers["info"]:
        return answers["info"][-1]
    return "asked" if answers["asked"] else ""


def _analysis(qtbot):
    release = threading.Event()
    started = threading.Event()

    def run(_settings):
        started.set()
        release.wait(20)

    thread, _worker = bridge.make_thread(run, {}, app_key="measure")
    thread.start()
    qtbot.waitUntil(started.is_set, timeout=10000)
    return thread, release


def test_after_a_first_launch_with_its_dialogs_only_a_real_analysis_blocks(
        window, answers, qtbot, monkeypatch):
    monkeypatch.setattr("spacr.install_profile.read_profile",
                        lambda *a, **k: {"created_utc": "2026-10-19T00:00:00Z"})
    monkeypatch.setattr(window, "_resolve_version", lambda: "1.5.1.5")
    monkeypatch.setattr(install_consent._InstallerCleanupDialog, "exec", lambda self: QDialog.Rejected)
    monkeypatch.setattr(window, "_continue_installer_onboarding", lambda: None)
    window._maybe_offer_installation_cache_review()
    qtbot.waitUntil(lambda: not getattr(window, "_cache_cleanup_busy", False), timeout=20000)
    window._show_whats_new("1.5.1.4")
    housekeeping, _worker = bridge.make_thread(lambda _s: None, {}, app_key="home journal",
                                               journal=False, user_visible=False)
    try:
        assert bridge.registry()._of_kind("housekeeping")
        assert _check(window, answers) == "asked"
    finally:
        housekeeping.start()
        qtbot.waitUntil(lambda: not bridge.registry()._of_kind("housekeeping"), timeout=10000)
    thread, release = _analysis(qtbot)
    try:
        assert "active analyses" in _check(window, answers)
    finally:
        release.set()
        thread.wait(10000)
    qtbot.waitUntil(lambda: not bridge.registry()._of_kind("analysis"), timeout=10000)
    assert _check(window, answers) == "asked"


@pytest.mark.parametrize("kind,words", [("package_install", "package is being installed"),
                                        ("model_download", "model download is in progress")])
def test_an_install_or_download_blocks_with_its_own_message(window, answers, kind, words):
    handle = bridge._track_external_job(f"{kind} job", None, kind=kind)
    try:
        message = _check(window, answers)
        assert words in message and "active analyses" not in message
    finally:
        handle.retire()
    assert _check(window, answers) == "asked"


def test_a_job_that_names_no_kind_counts_as_an_analysis(window, answers):
    handle = bridge._track_external_job("unknown job", None)
    del handle.kind
    try:
        assert "active analyses" in _check(window, answers)
    finally:
        handle.retire()
