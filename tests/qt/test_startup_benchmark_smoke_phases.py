"""The installed-application smoke's phase machine, driven off a Mac.

``_DistributionSmokeController._advance`` walks launch -> onboarding ->
screen -> configured -> starting -> running and the later layout, review and
native-menu polls, refusing (with a named reason in the receipt) any drift,
deadline or unexpected modal.  The Mac menu poll photographs the real
Preferences dialog, waits for it to close, then quits through the menu it was
verified in.  The controller's methods are called on a stand-in that carries
exactly the attributes they read; the cleanup-review cases use the real
``_InstallerCleanupDialog`` because the smoke must never press anything but its
default Keep.
"""
from __future__ import annotations

import ctypes
import json
import sys
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import startup_benchmark as benchmark  # noqa: E402

Controller = benchmark._DistributionSmokeController

pytestmark = pytest.mark.qt


def _controller(tmp_path, **extra):
    """A stand-in with the attributes the smoke's methods read."""
    calls = []
    fake = SimpleNamespace(
        output=tmp_path / "receipt.json", record={"status": "running"},
        phase="running", calls=calls,
        timer=SimpleNamespace(stop=lambda: calls.append("timer stopped")),
        app=SimpleNamespace(exit=lambda code: calls.append(("exit", code)),
                            platformName=lambda: "xcb",
                            activeModalWidget=lambda: None))
    fake._write = lambda: Controller._write(fake)
    fake._finish = lambda code: Controller._finish(fake, code)
    fake._pipeline_failed = lambda error: Controller._pipeline_failed(fake,
                                                                      error)
    fake._attach_to_the_worker = lambda: Controller._attach_to_the_worker(fake)
    fake._keep_installer_cleanup = (
        lambda: Controller._keep_installer_cleanup(fake))
    fake._onboarding_running = lambda: Controller._onboarding_running(fake)
    fake.__dict__.update(extra)
    return fake


def _receipt(fake):
    return json.loads(fake.output.read_text(encoding="utf-8"))


class _Dialog:
    """A Preferences-window stand-in that can be photographed and rejected."""

    def __init__(self, saved=True, visible=True):
        self.saved, self.visible, self.rejected = saved, visible, False

    def isVisible(self):  # noqa: N802
        return self.visible

    def grab(self):
        return SimpleNamespace(save=lambda path: self.saved)

    def reject(self):
        self.rejected = True


def _polling(monkeypatch, tmp_path, phase, dialog=None, mode="x"):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_preferences_window_class",
                        lambda: _Dialog)
    fake = _controller(tmp_path, phase=phase,
                       record={"native_menu": {"mode": mode}},
                       _native_menu_started=benchmark.time.monotonic(),
                       _native_menu=4, _native_actions={"quit": 7})
    fake.app.activeModalWidget = lambda: dialog
    return fake


def _weak_to(modal, state):
    """A weak reference to ``modal``, another live dialog, or one now gone."""
    if state == "same":
        return weakref.ref(modal), None
    other = _Dialog()
    reference = weakref.ref(other)
    if state == "gone":
        del other
        assert reference() is None
        return reference, None
    return reference, other


def test_the_native_menu_poll_photographs_preferences_then_quits(
        monkeypatch, tmp_path, subtests):
    """Each native-menu phase waits, refuses or advances with an exact record."""
    with subtests.test(msg="the poll has a 30-second deadline"):
        fake = _polling(monkeypatch, tmp_path, "native-menu-opening")
        fake._native_menu_started -= 31
        with pytest.raises(RuntimeError,
                           match="^Native menu acceptance exceeded its deadline$"):
            Controller._poll_native_menu_check(fake)

    with subtests.test(msg="no dialog yet: wait; then photograph and close"):
        fake = _polling(monkeypatch, tmp_path, "native-menu-opening")
        Controller._poll_native_menu_check(fake)
        assert fake.phase == "native-menu-opening"
        assert fake.record["native_menu"] == {"mode": "x"}
        dialog = _Dialog()
        fake.app.activeModalWidget = lambda: dialog
        Controller._poll_native_menu_check(fake)
        assert dialog.rejected and fake.phase == "native-menu-closing"
        assert fake.record["native_menu"] == {
            "mode": "x", "preferences_opened": True, "dialog_class": "_Dialog"}
        assert fake._native_preferences_ref() is dialog
        assert fake.calls == []

    refusals = [
        ("not the preferences class", object(),
         "Native Preferences opened a different dialog"),
        ("an invisible preferences window", _Dialog(visible=False),
         "Native Preferences opened a different dialog"),
        ("a screenshot that cannot be saved", _Dialog(saved=False),
         "Could not retain the actual Preferences screenshot"),
    ]
    for case, dialog, message in refusals:
        with subtests.test(msg=case):
            fake = _polling(monkeypatch, tmp_path, "native-menu-opening",
                            dialog)
            with pytest.raises(RuntimeError) as raised:
                Controller._poll_native_menu_check(fake)
            assert str(raised.value) == message
            assert fake.phase == "native-menu-opening"
            assert "preferences_opened" not in fake.record["native_menu"]

    for visible in (False, True):
        for state in ("same", "other", "gone"):
            with subtests.test(msg=f"quit waits: visible={visible} {state}"):
                modal = _Dialog(visible=visible)
                fake = _polling(monkeypatch, tmp_path,
                                "native-menu-ready-to-quit", modal)
                fake._native_preferences_ref, _keep = _weak_to(modal, state)
                with pytest.raises(
                        RuntimeError,
                        match="^Preferences did not close before native Quit$"):
                    Controller._poll_native_menu_check(fake)
                expected = {"class_name": "_Dialog", "visible": visible,
                            "same_observed_wrapper": (
                                None if state == "gone" else state == "same")}
                assert fake.record["native_menu"]["blocking_modal"] == expected
                assert _receipt(fake)["native_menu"]["blocking_modal"] == \
                    expected
                assert fake.phase == "native-menu-ready-to-quit"
                assert "quit_dispatched" not in fake.record["native_menu"]
                assert fake.calls == []

    with subtests.test(msg="quit waits: no observed reference at all"):
        modal = _Dialog()
        fake = _polling(monkeypatch, tmp_path, "native-menu-ready-to-quit",
                        modal)
        with pytest.raises(RuntimeError):
            Controller._poll_native_menu_check(fake)
        assert fake.record["native_menu"]["blocking_modal"][
            "same_observed_wrapper"] is None

    dispatches = [
        ("system-menu-on-cocoa",
         [(4, "performActionForItemAtIndex:", None, (ctypes.c_long, 7))]),
        ("in-window-on-cocoa", ["quit"]),
    ]
    for mode, sent_expected in dispatches:
        with subtests.test(msg=f"quit through the {mode} menu"):
            fake = _polling(monkeypatch, tmp_path,
                            "native-menu-ready-to-quit", mode=mode)
            sent = []
            fake._cocoa_message = lambda *args: sent.append(args)
            fake._click_window_menu_action = sent.append
            Controller._poll_native_menu_check(fake)
            assert fake.phase == "native-menu-quitting"
            assert fake.record["native_menu"] == {"mode": mode,
                                                  "quit_dispatched": True}
            assert sent == sent_expected
            assert _receipt(fake)["native_menu"]["quit_dispatched"] is True

    with subtests.test(msg="any other menu phase waits"):
        fake = _polling(monkeypatch, tmp_path, "native-menu-closing",
                        _Dialog())
        Controller._poll_native_menu_check(fake)
        assert fake.phase == "native-menu-closing"
        assert fake.record["native_menu"] == {"mode": "x"}


def _advancing(tmp_path, phase, **extra):
    failures = []
    fake = _controller(tmp_path, phase=phase,
                       started=benchmark.time.monotonic(),
                       root=tmp_path / "experiment",
                       window=SimpleNamespace(findChildren=lambda kind: []),
                       **extra)
    fake._pipeline_failed = failures.append
    return fake, failures


def _screen(enabled=True, **extra):
    return SimpleNamespace(_btn_run=SimpleNamespace(isEnabled=lambda: enabled),
                           **extra)


def _onboarding_window(**state):
    window = SimpleNamespace(
        findChildren=lambda kind: [], isVisible=lambda: True,
        _consent_timer=SimpleNamespace(isActive=lambda: False),
        _startup_sequence_running=lambda: False, _startup_deferred=[],
        _cache_cleanup_busy=False)
    window.__dict__.update(state)
    return window


def test_advance_launches_waits_out_onboarding_and_opens_measure(
        tmp_path, qapp, subtests):
    """Launch, onboarding and screen phases each wait, advance or name the blocker."""
    with subtests.test(msg="the smoke has a ten-minute deadline"):
        fake, failures = _advancing(tmp_path, "launch")
        fake.started -= 601
        Controller._advance(fake)
        assert failures == [
            "Distribution smoke exceeded its 600-second deadline"]

    with subtests.test(msg="a stale experiment folder is refused"):
        fake, failures = _advancing(tmp_path, "launch")
        fake.root.mkdir()
        try:
            Controller._advance(fake)
        finally:
            fake.root.rmdir()
        assert failures == [
            "Smoke experiment already exists; refusing stale analysis output"]

    with subtests.test(msg="only a visible tour is skipped; hidden window waits"):
        events = []
        shown = SimpleNamespace(isVisible=lambda: True,
                                _skip_btn=SimpleNamespace(
                                    click=lambda: events.append("skipped")))
        hidden = SimpleNamespace(isVisible=lambda: False)
        fake, failures = _advancing(tmp_path, "launch")
        fake.window = SimpleNamespace(
            findChildren=lambda kind: [hidden, shown], isVisible=lambda: False)
        fake._provenance = lambda: events.append("provenance")
        fake._keep_installer_cleanup = lambda: pytest.fail(
            "no cleanup action before installed provenance")
        Controller._advance(fake)
        Controller._advance(fake)
        assert events == ["skipped", "provenance"] * 2
        assert fake.record["tours_skipped"] == 2
        assert fake.phase == "launch" and failures == []

    with subtests.test(msg="onboarding is waited out stage by stage"):
        opened = []
        window = _onboarding_window(open_module=opened.append,
                                    _cache_cleanup_busy=True)
        fake, failures = _advancing(tmp_path, "launch")
        fake.window = window
        fake._provenance = lambda: None
        Controller._advance(fake)
        assert fake.phase == "onboarding" and opened == []
        stages = [
            ("cache cleanup", {"_cache_cleanup_busy": True}),
            ("deferred start-up work", {"_startup_deferred": [object()]}),
            ("start-up sequence", {"_startup_sequence_running": lambda: True}),
            ("consent timer", {"_consent_timer": SimpleNamespace(
                isActive=lambda: True)}),
        ]
        for _stage, state in stages:
            settled = _onboarding_window(open_module=opened.append)
            window.__dict__.update(settled.__dict__)
            window.__dict__.update(state)
            Controller._advance(fake)
            assert fake.phase == "onboarding" and opened == []
        window.__dict__.update(
            _onboarding_window(open_module=opened.append).__dict__)
        Controller._advance(fake)
        assert opened == ["measure"] and fake.phase == "screen"
        assert failures == []
        assert isinstance(fake.record["onboarding_settled_s"], float)

    with subtests.test(msg="a modal left during onboarding is named"):
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(QMessageBox.Warning, "Optional cache cleanup",
                          "Cache review could not finish.")
        fake, failures = _advancing(tmp_path, "onboarding")
        fake.window = _onboarding_window(
            _cache_cleanup_busy=True,
            _cache_cleanup_error="Traceback: preview failed")
        fake.app.activeModalWidget = lambda: box
        Controller._advance(fake)
        assert len(failures) == 1
        assert failures[0].startswith(
            "An unexpected modal dialog blocks first-launch onboarding: "
            "cls='QMessageBox'")
        assert fake.record["blocking_modal"]["cache_review_error"] == (
            "Traceback: preview failed")
        assert fake.phase == "onboarding"
        box.deleteLater()

    with subtests.test(msg="the screen phase waits for an enabled Run"):
        fake, failures = _advancing(tmp_path, "screen")
        fake.window._screens = {}
        Controller._advance(fake)
        fake.window._screens = {"measure": _screen(enabled=False)}
        Controller._advance(fake)
        assert fake.phase == "screen" and failures == []
        fake.window._screens = {"measure": _screen()}
        fake.app.activeModalWidget = lambda: object()
        Controller._advance(fake)
        assert failures == ["An unexpected modal dialog blocks the installed "
                            "application: cls='object'"]
        assert fake.record["blocking_modal"] == {"cls": "object"}


def test_a_blocking_modal_is_named_in_the_failure_and_the_receipt(qtbot,
                                                                  tmp_path):
    """Class, object name, title and both texts reach the receipt and the failure."""
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(QMessageBox.Warning, "Optional cache cleanup",
                      "Cache review could not finish.")
    box.setObjectName("SomeNotice")
    box.setInformativeText("Check the log.")
    qtbot.addWidget(box)
    fake, failures = _advancing(tmp_path, "screen")
    fake.window._screens = {"measure": _screen()}
    fake.app.activeModalWidget = lambda: box
    Controller._advance(fake)
    detail = {"cls": "QMessageBox", "objectName": "SomeNotice",
              "windowTitle": "Optional cache cleanup",
              "text": "Cache review could not finish.",
              "informativeText": "Check the log."}
    if sys.platform == "darwin":
        # QMessageBox drops its window title on macOS.
        del detail["windowTitle"]
    assert fake.record["blocking_modal"] == detail
    assert _receipt(fake)["blocking_modal"] == detail
    described = ", ".join(f"{key}={value!r}" for key, value in detail.items())
    assert failures == [
        "An unexpected modal dialog blocks the installed application: "
        + described]


def _clicked_screen(**extra):
    """A Measure screen whose Run button is enabled and clicks."""
    clicks = []
    screen = _screen(
        _settings_model=SimpleNamespace(
            collect=lambda: {"src": "/smoke", "n_jobs": 1}),
        _crop_choice_warnings=lambda settings: [], **extra)
    screen._btn_run.click = lambda: clicks.append("run")
    screen.clicks = clicks
    return screen


def test_advance_configures_and_starts_the_real_run(tmp_path, subtests):
    """Drifted settings are refused; the Run is followed to its worker."""
    drift = [
        ("src", "The live Measure screen did not receive the smoke input"),
        ("jobs", "The live Measure screen did not retain its one-worker "
                 "limit"),
        ("crops", "The smoke fixture unexpectedly requires crop confirmation"),
        ("worker", "The real Run action did not start a pipeline worker"),
    ]
    for change, message in drift:
        with subtests.test(msg=f"configured screen drifted: {change}"):
            collected = {"src": "/elsewhere" if change == "src" else "/smoke",
                         "n_jobs": 4 if change == "jobs" else 1}
            screen = _clicked_screen(_worker=None)
            screen._settings_model = SimpleNamespace(
                collect=lambda c=collected: dict(c))
            screen._crop_choice_warnings = (
                lambda settings, c=change: ["crop"] if c == "crops" else [])
            fake, failures = _advancing(tmp_path, "configured", screen=screen,
                                        _expected_src="/smoke")
            Controller._advance(fake)
            assert failures == [message]
            assert fake.record["settings"] == collected
            assert screen.clicks == (["run"] if change == "worker" else [])
            assert fake.record.get("module_constructed") is (
                True if change == "worker" else None)

    with subtests.test(msg="a configured screen waits for its Run button"):
        fake, failures = _advancing(tmp_path, "configured",
                                    screen=_screen(enabled=False))
        Controller._advance(fake)
        assert fake.phase == "configured" and failures == []
        assert "settings" not in fake.record

    with subtests.test(msg="a first Run still importing its pipeline"):
        connected = []
        worker = SimpleNamespace(error=SimpleNamespace(
            connect=connected.append))
        screen = _clicked_screen(_entry_import=object(), _worker=None)
        fake, failures = _advancing(tmp_path, "configured", screen=screen,
                                    _expected_src="/smoke")
        Controller._advance(fake)
        assert fake.phase == "starting" and failures == []
        assert "real_run_clicked" not in fake.record
        Controller._advance(fake)
        assert fake.phase == "starting" and failures == []
        screen._entry_import = None
        screen._worker = worker
        Controller._advance(fake)
        assert fake.phase == "running" and failures == []
        assert fake.record["real_run_clicked"] is True
        assert connected == [fake._pipeline_failed]
        assert screen.clicks == ["run"]
        assert _receipt(fake)["real_run_clicked"] is True

    with subtests.test(msg="a warm worker goes straight to running"):
        connected = []
        worker = SimpleNamespace(error=SimpleNamespace(
            connect=connected.append))
        fake, failures = _advancing(tmp_path, "configured",
                                    screen=_clicked_screen(_worker=worker),
                                    _expected_src="/smoke")
        Controller._advance(fake)
        assert fake.phase == "running" and failures == []
        assert fake.record["module_constructed"] is True
        assert fake.record["settings"] == {"src": "/smoke", "n_jobs": 1}
        assert connected == [fake._pipeline_failed]

    with subtests.test(msg="an import that ends without a worker"):
        screen = _clicked_screen(_entry_import=object(), _worker=None)
        fake, failures = _advancing(tmp_path, "starting", screen=screen)
        Controller._advance(fake)
        assert failures == []
        screen._entry_import = None
        Controller._advance(fake)
        assert failures == [
            "The real Run action did not start a pipeline worker"]

    later = [
        ("settling-layout", "_poll_layout_check"),
        ("checking-layout-reachability", "_poll_layout_reachability"),
        ("native-menu-opening", "_poll_native_menu_check"),
        ("native-menu-quitting", "_poll_native_menu_check"),
        ("finished", None),
    ]
    for phase, method in later:
        with subtests.test(msg=f"{phase} goes to its own poll"):
            polled = []
            fake, failures = _advancing(tmp_path, phase)
            for name in ("_poll_layout_check", "_poll_layout_reachability",
                         "_poll_native_menu_check",
                         "_poll_installation_review"):
                setattr(fake, name, lambda name=name: polled.append(name))
            Controller._advance(fake)
            assert polled == ([method] if method else [])
            assert failures == []


def _cleanup_dialog(qtbot, parent, preview):
    from PySide6.QtWidgets import QApplication
    from spacr.qt.install_consent import _InstallerCleanupDialog

    dialog = _InstallerCleanupDialog(preview, parent)
    qtbot.addWidget(dialog)
    return dialog, QApplication


def test_the_cleanup_review_is_kept_and_never_bypassed(qtbot, monkeypatch,
                                                       tmp_path, subtests):
    """The smoke presses only Keep, on a later exposed tick, and no deadline skips it."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialog, QPushButton, QWidget
    from spacr import install_cleanup

    monkeypatch.setattr(install_cleanup, "apply_optional_cleanup",
                        lambda *a, **k: pytest.fail("the smoke removed files"))

    with subtests.test(msg="a late review is kept before the native menu"):
        monkeypatch.setenv("SPACR_NATIVE_MENU_SMOKE", "1")
        parent = QWidget()
        qtbot.addWidget(parent)
        parent.show()
        parent._cache_cleanup_busy = True
        parent._consent_timer = QTimer(parent)
        parent._startup_deferred = []
        parent._startup_sequence_running = lambda: False
        events = []
        fake = _controller(tmp_path, window=parent,
                           started=benchmark.time.monotonic(),
                           record={"layout": {"status": "passed"}})
        from PySide6.QtWidgets import QApplication
        fake.app.activeModalWidget = QApplication.activeModalWidget
        fake._start_native_menu_check = lambda: events.append("native menu")
        fake._poll_installation_review = (
            lambda: Controller._poll_installation_review(fake))
        Controller._finish_layout_check(fake)
        assert (events, fake.phase) == ([], "waiting-installation-review")
        assert fake.record["visual_layout_status"] == "passed"
        Controller._advance(fake)
        assert events == []
        sentinel = tmp_path / "cache bytes"
        sentinel.write_bytes(b"kept cache contents")
        preview = {"operation_id": "actual-dialog",
                   "snapshot_id": "observed-preview",
                   "folders": [{"key": "news", "path": str(tmp_path),
                                "size_bytes": 19,
                                "contents": [{"name": sentinel.name,
                                              "size_bytes": 19}],
                                "selectable": True}]}
        dialog, _app = _cleanup_dialog(qtbot, parent, preview)
        dialog.finished.connect(
            lambda _result: setattr(parent, "_cache_cleanup_busy", False))
        QTimer.singleShot(0, dialog.open)
        qtbot.waitUntil(lambda: QApplication.activeModalWidget() is dialog)
        assert dialog.findChild(QPushButton, "InstallerCleanupKeep") \
            .isDefault()
        Controller._advance(fake)
        assert dialog.isVisible() and "installation_review" not in fake.record
        assert fake._cleanup_review_seen is dialog
        qtbot.waitUntil(lambda: dialog.windowHandle().isExposed())
        Controller._advance(fake)
        assert not dialog.isVisible()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert dialog.choices()["keys"] == []
        assert sentinel.read_bytes() == b"kept cache contents"
        assert fake.record["installation_review"] == {"choice": "keep",
                                                      "selected_count": 0}
        assert events == []
        Controller._advance(fake)
        assert events == ["native menu"]

    boundaries = [
        ("native_quit", "native-menu-ready-to-quit",
         "Preferences did not close before native Quit"),
        ("expired_review", "waiting-installation-review",
         "Installation review acceptance exceeded its deadline"),
    ]
    for boundary, phase, error in boundaries:
        with subtests.test(msg=f"no bypass at {boundary}"):
            from PySide6.QtWidgets import QApplication

            window = QWidget()
            qtbot.addWidget(window)
            window.show()
            sentinel = tmp_path / "retained contents"
            sentinel.write_bytes(b"no cleanup action")
            dialog, _app = _cleanup_dialog(
                qtbot, window, {"operation_id": "boundary", "folders": []})
            clicked = []
            dialog.findChild(QPushButton, "InstallerCleanupKeep") \
                .clicked.connect(lambda: clicked.append("keep"))
            dialog.open()
            qtbot.waitUntil(lambda d=dialog:
                            QApplication.activeModalWidget() is d)
            now = benchmark.time.monotonic()
            fake = _controller(tmp_path, phase=phase, window=window,
                               started=now, _native_menu_started=now,
                               _installation_review_started=now - 31,
                               record={"native_menu": {
                                   "quit_dispatched": False}})
            fake.app.activeModalWidget = QApplication.activeModalWidget
            fake._poll_native_menu_check = (
                lambda f=fake: Controller._poll_native_menu_check(f))
            fake._poll_installation_review = (
                lambda f=fake: Controller._poll_installation_review(f))
            try:
                Controller._advance(fake)
                assert clicked == []
                assert dialog.isVisible()
                assert dialog.choices()["keys"] == []
                assert sentinel.read_bytes() == b"no cleanup action"
                assert (fake.record["status"], fake.record["error"]) == (
                    "failed", error)
                assert fake.record["native_menu"]["quit_dispatched"] is False
                assert "installation_review" not in fake.record
                assert fake.calls == ["timer stopped", ("exit", 1)]
            finally:
                dialog.reject()
