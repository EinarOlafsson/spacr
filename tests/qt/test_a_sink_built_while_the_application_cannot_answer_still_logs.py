"""A console sink built while the application cannot be asked is still a sink.

43 moved ``QtLogHandler`` onto the GUI thread whichever thread builds it
(``test_a_qt_warning_reaches_the_console_once.py`` holds that half). Finding
the GUI thread means asking ``QCoreApplication`` for its instance, and the
sink is built lazily from a logging filter -- so it can be asked while the
application is being torn down, when the question itself raises. The move is
a refinement of where records are delivered; a logger that could not be built
would lose them all, and would raise out of whatever line happened to log.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_a_sink_built_while_the_application_raises_still_delivers(
        qapp, monkeypatch):
    from PySide6.QtCore import QCoreApplication

    from spacr.qt.logging_util import QtLogHandler

    def being_torn_down():
        raise RuntimeError("Internal C++ object (QApplication) already deleted.")

    monkeypatch.setattr(QCoreApplication, "instance",
                        staticmethod(being_torn_down))
    sink = QtLogHandler()
    monkeypatch.undo()

    sink.setLevel(logging.WARNING)
    out = []
    sink.record_ready.connect(lambda text, level: out.append(text))
    logger = logging.getLogger("tests.qt.a_sink_built_while_torn_down")
    logger.addHandler(sink)
    try:
        logger.warning("logged while the application could not answer")
        qapp.processEvents()
    finally:
        logger.removeHandler(sink)

    assert len(out) == 1, f"the sink delivered {len(out)} line(s), not one"
    assert "logged while the application could not answer" in out[0]
