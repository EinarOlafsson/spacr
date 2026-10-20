"""The distribution smoke on a fresh profile meets no dialog it does not expect.

The 2026-10-20 Intel Mac frozen smoke of nightly 5400e0233 failed in its
``screen`` phase with "An unexpected modal dialog blocks the installed
application", where 1.5.1.5 had passed. First-launch onboarding (installer
consent, optional cache review, report notice) opens from the start-up timers
on its own schedule, so whether a dialog was up when the smoke reached the
Measure screen depended on timing. The smoke now waits for onboarding before
it drives a screen, keeps the cache review's defaults, and names any other
modal it meets. The headless modal guard stays on for every other dialog;
only the cache review runs as a real modal, for the smoke to answer.

These tests boot the real main window the way the frozen smoke does -- a
fresh HOME, ``SPACR_DISTRIBUTION_SMOKE=1``, a frozen interpreter -- and run
the real controller up to the configured Measure screen.
"""

from __future__ import annotations

import json
import sys

import pytest

from spacr.qt import startup_benchmark as benchmark

Controller = benchmark._DistributionSmokeController

pytestmark = pytest.mark.qt


@pytest.fixture
def smoke_profile(monkeypatch, tmp_path):
    """A fresh profile and the frozen-installer smoke environment."""
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name, value in (("SPACR_DISTRIBUTION_SMOKE", "1"),
                        ("SPACR_DISTRIBUTION_KIND", "frozen"),
                        ("SPACR_NATIVE_MENU_SMOKE", "1"),
                        ("SPACR_NO_SETUP", "1"),
                        ("SPACR_DEVICE", "cpu")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    return tmp_path


def _run_to_the_configured_screen(qapp, qtbot, monkeypatch, tmp_path):
    """Drive the real controller until it has configured Measure; spy on modals."""
    import spacr.qt
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    from spacr.qt.app import MainWindow
    from spacr.qt.install_consent import _InstallerCleanupDialog
    from spacr.qt.preferences import set_preload_policy

    spacr.qt.register_self_registering_modules()
    set_preload_policy("on_demand")

    def source_scope(self):
        """Stand in for the frozen-artifact provenance check."""
        self.record["test_scope"] = "fresh-profile smoke, not artifact acceptance"

    reached = []
    advance = Controller._advance

    def stop_when_configured(self):
        """Stop before the Run click; this test is about the dialogs."""
        if self.phase == "configured":
            self.timer.stop()
            reached.append(self.phase)
            return
        advance(self)

    def answered_exec(dialog):
        """Run the cleanup review as a real modal; the smoke answers it."""
        loop = QEventLoop()
        dialog.finished.connect(loop.quit)
        dialog.open()
        loop.exec()
        return dialog.result()

    monkeypatch.setattr(_InstallerCleanupDialog, "exec", answered_exec)
    monkeypatch.setattr(Controller, "_provenance", source_scope)
    monkeypatch.setattr(Controller, "_advance", stop_when_configured)
    seen = []

    def spy():
        """Record every modal the smoke could meet, and when."""
        widget = QApplication.activeModalWidget()
        if widget is not None:
            entry = (type(widget).__name__, controller.phase)
            if entry not in seen:
                seen.append(entry)

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show()
    output = tmp_path / "receipt" / "smoke.json"
    controller = Controller(qapp, window, str(output))
    watcher = QTimer()
    watcher.setInterval(20)
    watcher.timeout.connect(spy)
    watcher.start()
    try:
        qtbot.waitUntil(lambda: bool(reached) or controller.phase == "finished",
                        timeout=240000)
    finally:
        watcher.stop()
        controller.timer.stop()
    return controller, json.loads(output.read_text(encoding="utf-8")), seen


def test_a_fresh_profile_reaches_the_configured_measure_screen_with_no_modal(
        qapp, qtbot, monkeypatch, smoke_profile):
    controller, record, seen = _run_to_the_configured_screen(
        qapp, qtbot, monkeypatch, smoke_profile)
    from PySide6.QtWidgets import QApplication

    assert record.get("status") == "running", record.get("error")
    assert "blocking_modal" not in record
    assert controller.phase == "configured"
    assert QApplication.activeModalWidget() is None
    assert {name for name, _phase in seen} <= {"_InstallerCleanupDialog"}
    assert all(phase in ("launch", "onboarding") for _name, phase in seen), seen
    assert record["installation_review"] == {"choice": "keep", "selected_count": 0}
    assert "onboarding_settled_s" in controller.record
    assert not controller.window._cache_cleanup_busy
