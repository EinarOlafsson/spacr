"""`gh auth login` finishes after the setup screen is gone.

The sign-in takes as long as the user takes -- a browser, a one-time
code, a password manager -- and the setup screen is not modal about it.
So the process's ``finished`` signal routinely lands on a dialog that has
already been closed, and everything it touches by then is a deleted C++
object with a live Python wrapper in front of it.

Reported on 2026-08-23, from a real launch::

    QProcess: Destroyed while process ("gh") is still running.
    RuntimeError: libshiboken: Internal C++ object
    (PySide6.QtWidgets.QPushButton) already deleted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.setup_slides import SetupSlides, _let_go_of


@pytest.fixture
def slides(qtbot):
    made = SetupSlides()
    qtbot.addWidget(made)
    made.show()
    return made


def test_refreshing_a_closed_dialog_is_a_no_op(slides, qtbot, qapp):
    """The exact crash: `finished` arriving after the screen is gone."""
    refresh = slides._refresh_github
    slides.close()
    slides.deleteLater()
    for _ in range(3):
        qapp.processEvents()

    refresh()   # must not raise


def test_a_live_dialog_still_refreshes(slides, monkeypatch):
    """The guard must not turn the normal path into a no-op."""
    monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source", lambda: "gh")
    slides._gh_status.setText("")
    slides._refresh_github()
    assert slides._gh_status.text().strip()


def test_letting_go_disconnects_and_detaches(qtbot):
    """What the dialog's `destroyed` does to a `gh` still running.

    Disconnected FIRST: the signal points at widgets that are about to
    stop existing. Detached second, so Qt does not destroy a QProcess
    whose child is alive -- which is the "Destroyed while process is
    still running" half of the report.
    """
    from PySide6.QtCore import QObject, QProcess

    parent = QObject()
    process = QProcess(parent)
    seen = []
    process.finished.connect(lambda *_a: seen.append(True))

    _let_go_of(process)

    assert process.parent() is None
    process.finished.emit(0, QProcess.ExitStatus.NormalExit)
    assert seen == [], "the finished handler was still connected"
