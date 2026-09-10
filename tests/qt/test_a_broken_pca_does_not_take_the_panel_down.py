"""Three guards in the PCA panel, driven rather than assumed.

Instruction 288. All three were marked ``# pragma: no cover`` with the
word "defensive", which is the mark that most often hides either a live
path nobody tried or a dead one nobody checked. These are live:

* ``render_now`` catches anything ``_draw_arrows`` throws, because the
  arrows are a decoration on a chart that must still be drawn without
  them;
* the fit catches every non-``PCAError`` exception, so a failure inside
  ``pca()`` reaches the panel as a message instead of an exception out of
  a worker;
* ``changeEvent`` on the AI/Live label tolerates an event whose ``type()``
  raises, which is what a deleted C++ half does in PySide6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import pca_view
from spacr.qt.widgets.pca_view import PCAPanel
from spacr.qt.linked_selection import LinkedSelection


def _frame(n=30):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    })


@pytest.fixture
def panel(qtbot):
    widget = PCAPanel(link=LinkedSelection())
    qtbot.addWidget(widget)
    return widget


def test_arrows_that_fail_to_draw_do_not_take_the_chart_with_them(
        panel, monkeypatch):
    """A decoration must never be able to blank the plot."""
    panel.set_frame(_frame())
    assert panel.result is not None, "no decomposition to draw"

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the arrows could not be drawn")

    monkeypatch.setattr(type(panel.canvas), "_draw_arrows", _explode)

    panel.canvas.render_now()          # must not raise

    assert panel.result is not None, "the result was lost with the arrows"


def test_a_pca_that_fails_unexpectedly_becomes_a_message(panel, monkeypatch):
    """Not a traceback out of a worker.

    ``PCAError`` is the expected refusal and has its own arm. This drives
    the OTHER one -- anything else at all -- which is what a bug inside
    the decomposition looks like from here.
    """
    panel.set_frame(_frame())

    def _explode(*_args, **_kwargs):
        raise ZeroDivisionError("something went wrong inside pca()")

    monkeypatch.setattr(pca_view, "pca", _explode)

    panel.recompute()                  # must not raise

    assert "PCA failed" in panel.report.text(), (
        f"the failure never reached the user: {panel.report.text()!r}")
    assert "ZeroDivisionError" in panel.report.text() or \
        "something went wrong" in panel.report.text(), (
            "the message does not say what actually failed")


def test_the_expected_refusal_still_reads_differently(panel, monkeypatch):
    """So the test above is not passing on the wrong arm.

    A PCAError carries its own explanation and must NOT be dressed up as
    'PCA failed: ...' -- that prefix is what marks an unexpected fault.
    """
    from spacr.qt.widgets.pca_model import PCAError

    panel.set_frame(_frame())

    def _refuse(*_args, **_kwargs):
        raise PCAError("not enough rows to decompose")

    monkeypatch.setattr(pca_view, "pca", _refuse)

    panel.recompute()

    assert "not enough rows to decompose" in panel.report.text()
    assert "PCA failed" not in panel.report.text(), (
        "an expected refusal was reported as an unexpected fault")


def test_a_label_survives_an_event_whose_type_cannot_be_read(qtbot):
    """PySide6 raises when the C++ half of a wrapper is gone.

    ``changeEvent`` is called during teardown as well as during a
    Preferences save, so an event whose ``type()`` raises has to mean
    "restyle nothing", not an exception out of a Qt callback.
    """
    from PySide6.QtCore import QEvent

    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    label = AiToggleLabel(text="AI")
    qtbot.addWidget(label)

    class _Hostile(QEvent):
        def __init__(self, kind):
            super().__init__(kind)
            self.asked = 0

        def type(self):
            self.asked += 1
            raise RuntimeError("Signal source has been deleted")

    event = _Hostile(QEvent.StyleChange)
    before = label.text()

    label.changeEvent(event)                # must not raise

    # ASSERTED, not merely survived. "It did not raise" passes just as
    # well against a changeEvent that returns immediately and never looks
    # at the event at all.
    assert event.asked >= 1, "the event's type was never read"
    assert label.text() == before, (
        "an unreadable event restyled the label anyway")


def test_a_readable_style_change_still_restyles(qtbot, monkeypatch):
    """The other side: the hostile-event test must not pass because
    ``changeEvent`` does nothing at all."""
    from PySide6.QtCore import QEvent

    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    label = AiToggleLabel(text="AI")
    qtbot.addWidget(label)

    seen = []
    monkeypatch.setattr(type(label), "_refresh_style",
                        lambda self: seen.append(True))

    label.changeEvent(QEvent(QEvent.StyleChange))

    assert seen == [True], "a real StyleChange did not restyle the label"
