"""Qt keeps delivering events to a filter after Python has emptied it.

At startup spaCR printed this, twice:

    Error calling Python override of QObject::eventFilter():
      File ".../glass.py", line 365, in eventFilter
        if watched is self._dialog and event.type() in (
    AttributeError: '_Backdrop' object has no attribute '_dialog'

The C++ QObject outlives the Python object's ``__dict__``. During teardown
the wrapper's attributes are cleared while the filter is still installed, so
every event after that point raises AttributeError -- and Qt catches it,
prints the whole traceback, and carries on, which produces noise nobody can
act on and hides anything real underneath it.

``_DragByBackground`` was fixed this way and ``_Backdrop`` was not, which is
why the message came back naming a different class. Both are covered here,
and so is ``_fit``, which reads the same two attributes from a timer rather
than from the filter.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QDialog, QWidget

from spacr.qt.widgets import glass as G


def _emptied(obj):
    """The state Qt finds a filter in once Python has torn it down."""
    obj.__dict__.clear()
    return obj


@pytest.fixture
def dialog_and_card(qapp):
    dialog = QDialog()
    card = QWidget(dialog)
    try:
        yield dialog, card
    finally:
        dialog.deleteLater()
        qapp.processEvents()


def test_the_backdrop_filter_survives_its_own_teardown(dialog_and_card, qapp):
    dialog, card = dialog_and_card
    backdrop = _emptied(G._Backdrop(dialog, card))

    # Must not raise. Returning False is the "did not handle it" answer.
    assert backdrop.eventFilter(dialog, QEvent(QEvent.Type.Resize)) is False
    assert backdrop.eventFilter(dialog, QEvent(QEvent.Type.Show)) is False


def test_the_backdrop_fit_survives_its_own_teardown(dialog_and_card):
    """It runs from a timer as well as the filter, so it is guarded too."""
    dialog, card = dialog_and_card
    dialog.resize(400, 300)
    backdrop = G._Backdrop(dialog, card)
    placed = card.geometry()

    _emptied(backdrop)
    backdrop._fit()          # must not raise
    # And it did nothing, because there is no dialog left to measure.
    assert card.geometry() == placed


def test_the_drag_filter_survives_its_own_teardown(dialog_and_card):
    dialog, _card = dialog_and_card
    drag = _emptied(G._DragByBackground(dialog))

    assert drag.eventFilter(dialog, QEvent(QEvent.Type.MouseMove)) is False


def test_a_live_backdrop_still_fits_the_card(dialog_and_card, qapp):
    """The guard may not turn the filter off for a dialog that is still there."""
    dialog, card = dialog_and_card
    dialog.resize(400, 300)
    backdrop = G._Backdrop(dialog, card)
    dialog.resize(500, 360)
    backdrop.eventFilter(dialog, QEvent(QEvent.Type.Resize))
    qapp.processEvents()

    expected = dialog.rect().adjusted(G.INSET, G.INSET, -G.INSET, -G.INSET)
    assert card.geometry() == expected


def test_no_filter_reads_the_dialog_without_guarding_it():
    """The next filter added here has to do the same thing.

    A source check, deliberately: the two runtime tests above only cover the
    classes that exist today, and this defect has now been introduced twice
    by adding a filter that reads ``self._dialog`` straight.
    """
    import inspect

    source = inspect.getsource(G)
    for klass in ("_Backdrop", "_DragByBackground"):
        body = source.split(f"class {klass}(")[1].split("\nclass ")[0]
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "self._dialog = " in stripped:
                continue
            assert "self._dialog" not in stripped, (
                f"{klass} reads self._dialog at {stripped!r}; use "
                f"getattr(self, '_dialog', None) -- Qt delivers events to "
                f"this filter after Python has cleared its __dict__")
