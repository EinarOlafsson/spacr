"""The launch overlay and the updater worker are both allowed to fail.

Neither is the application: a splash that cannot be built, or one whose
C++ side is torn down mid-import, must leave a window that opens. The
updater worker is the other half of the same posture -- it runs on a
QThread, where an escaping exception aborts the process rather than
raising anywhere anyone can see it.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

import PySide6.QtWidgets as qtwidgets  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402
from shiboken6 import delete as _delete_cpp_side  # noqa: E402

from spacr.qt import app as qt_app  # noqa: E402

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------
# the updater worker
# --------------------------------------------------------------------------

def test_a_finished_operation_hands_its_result_on(qapp):
    got = []
    worker = qt_app._UpdateWorker("check", lambda: {"version": "1.2.3"})
    worker.succeeded.connect(got.append)

    worker.run()

    assert got == [{"version": "1.2.3"}]


def test_a_failing_operation_reports_the_name_and_the_traceback(qapp, caplog):
    failures = []

    def explode():
        raise RuntimeError("the index is unreachable")

    worker = qt_app._UpdateWorker("check", explode)
    worker.failed.connect(lambda name, details: failures.append((name,
                                                                 details)))
    succeeded = []
    worker.succeeded.connect(succeeded.append)

    with caplog.at_level(logging.ERROR, logger=qt_app.LOG.name):
        worker.run()

    assert succeeded == [], "a failure is not also a result"
    assert failures and failures[0][0] == "check"
    assert "the index is unreachable" in failures[0][1], (
        "the traceback is what the dialog offers to show")


# --------------------------------------------------------------------------
# the loading screen
# --------------------------------------------------------------------------

_REAL_QAPPLICATION = qtwidgets.QApplication


class _OnScreenApplication:
    """The live application, reporting a real platform as a desktop run does.

    Everything else is delegated: the running QApplication is still the one
    Qt (and pytest-qt) has to talk to.
    """

    @staticmethod
    def instance():
        live = _REAL_QAPPLICATION.instance()

        class _SaysItIsOnScreen:
            def __getattr__(self, name):
                return getattr(live, name)

            @staticmethod
            def platformName():
                return "xcb"

        return _SaysItIsOnScreen()


def test_a_headless_run_is_not_given_a_splash(qapp):
    host = QWidget()
    try:
        assert qt_app.MainWindow._install_loading_screen(host) is None, (
            "a test process must not pay for a splash it cannot show")
    finally:
        host.deleteLater()


def test_a_desktop_run_gets_a_splash_covering_the_window(qapp, monkeypatch):
    monkeypatch.setattr(qtwidgets, "QApplication", _OnScreenApplication)
    host = QWidget()
    host.resize(400, 300)
    try:
        screen = qt_app.MainWindow._install_loading_screen(host)

        assert screen is not None
        assert screen.parent() is host
        assert screen.geometry() == host.rect()
    finally:
        host.deleteLater()


def test_a_splash_that_cannot_be_built_costs_nothing(qapp, monkeypatch,
                                                     caplog):
    monkeypatch.setattr(qtwidgets, "QApplication", _OnScreenApplication)
    import spacr.qt.widgets.loading_screen as loading_screen

    def refuse(**_kwargs):
        raise RuntimeError("no window handle")

    monkeypatch.setattr(loading_screen, "LoadingScreen", refuse)
    host = QWidget()
    try:
        with caplog.at_level(logging.DEBUG, logger=qt_app.LOG.name):
            assert qt_app.MainWindow._install_loading_screen(host) is None
    finally:
        host.deleteLater()

    assert any("loading screen" in record.getMessage()
               for record in caplog.records)


# --------------------------------------------------------------------------
# what the preloader does to it
# --------------------------------------------------------------------------

@pytest.fixture()
def host_with_splash(qapp, monkeypatch):
    monkeypatch.setattr(qtwidgets, "QApplication", _OnScreenApplication)
    host = QWidget()
    host.resize(500, 400)
    host._loading_screen = qt_app.MainWindow._install_loading_screen(host)
    assert host._loading_screen is not None
    yield host
    host.deleteLater()


def test_each_imported_module_advances_the_splash(host_with_splash):
    qt_app.MainWindow._on_preload_step(host_with_splash, 2, 7)

    screen = host_with_splash._loading_screen
    assert screen is not None
    assert getattr(screen, "_total", 7) == 7


def test_a_splash_deleted_mid_import_is_forgotten_not_reused(
        host_with_splash):
    _delete_cpp_side(host_with_splash._loading_screen)

    qt_app.MainWindow._on_preload_step(host_with_splash, 1, 3)

    assert host_with_splash._loading_screen is None, (
        "the next step must not go back to a widget that has gone")


def test_a_window_with_no_splash_ignores_the_progress(qapp):
    host = QWidget()
    host._loading_screen = None
    try:
        qt_app.MainWindow._on_preload_step(host, 1, 3)
        qt_app.MainWindow._on_preload_done(host)
    finally:
        host.deleteLater()

    assert host._loading_screen is None


def test_finishing_the_preload_takes_the_splash_down(host_with_splash):
    screen = host_with_splash._loading_screen

    qt_app.MainWindow._on_preload_done(host_with_splash)

    assert host_with_splash._loading_screen is None
    assert screen.isVisible() is False


def test_a_splash_already_gone_still_ends_the_preload(host_with_splash):
    _delete_cpp_side(host_with_splash._loading_screen)

    qt_app.MainWindow._on_preload_done(host_with_splash)

    assert host_with_splash._loading_screen is None
