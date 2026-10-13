"""The offscreen renderer refuses rather than rendering under no application.

:func:`spacr.figures.headless.application` asks
:func:`spacr.figures.scene.pyqtgraph_ready` whether a scene can be built, and
then takes the ``QApplication`` that answer implies. The two statements are
not one statement, so the second is checked: if there is no application after
all, the renderer says so in a sentence naming the fix instead of handing a
``None`` to ``GroupedPlot`` and crashing several frames later.
"""
from __future__ import annotations

import logging
import os

import pytest

pytest.importorskip("PySide6")

from spacr.figures import headless                              # noqa: E402


class _NoApplication:
    """A ``QApplication`` class that reports no living instance."""

    @staticmethod
    def instance():
        return None


@pytest.fixture()
def ready_but_applicationless(monkeypatch):
    """Readiness says yes; ``QApplication.instance()`` says there is none."""
    import PySide6.QtWidgets

    monkeypatch.setattr("spacr.figures.scene.pyqtgraph_ready",
                        lambda: (True, ""))
    monkeypatch.setattr(PySide6.QtWidgets, "QApplication", _NoApplication)


def test_no_living_application_is_refused_with_the_platform_advice(
        ready_but_applicationless):
    """``application()`` returns no app and the sentence that names the fix."""
    app, reason = headless.application()

    assert app is None
    assert reason == headless.NO_PLATFORM
    assert "QT_QPA_PLATFORM=offscreen" in reason


def test_rendering_without_an_application_writes_nothing_and_warns(
        tmp_path, caplog, ready_but_applicationless):
    """The refusal reaches the log and no half-written file is left behind."""
    caplog.set_level(logging.WARNING, logger="spacr.figures.headless")

    written = headless.render_offscreen(object(), str(tmp_path / "plot.png"))

    assert written is None
    assert os.listdir(str(tmp_path)) == []
    assert any("QT_QPA_PLATFORM=offscreen" in record.getMessage()
               for record in caplog.records), caplog.text


def test_a_bundle_without_an_application_writes_nothing_and_warns(
        tmp_path, caplog, ready_but_applicationless):
    """The bundle writer refuses on the same terms as the single figure."""
    caplog.set_level(logging.WARNING, logger="spacr.figures.headless")

    folder = headless.render_bundle(object(), str(tmp_path), "regression")

    assert folder is None
    assert os.listdir(str(tmp_path)) == []
    assert any("QT_QPA_PLATFORM=offscreen" in record.getMessage()
               for record in caplog.records), caplog.text
