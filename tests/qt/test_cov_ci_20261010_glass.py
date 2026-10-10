"""A popup backdrop that is not an ambient widget, switched off and back on.

Settings popups can carry a fractal backdrop, which pauses and resumes but
has no ``set_animating``. Pinned here:

* switching the popup backdrop off pauses and hides such a backdrop rather
  than raising because it lacks ``set_animating``;
* reopening with the same theme resumes the existing backdrop, shows it and
  sends it behind the card, without building a replacement.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

from spacr.qt.widgets import glass

pytestmark = pytest.mark.qt


class _Fractal(QWidget):
    """Stands in for a fractal backdrop: pause/resume, no set_animating."""

    def __init__(self, parent):
        super().__init__(parent)
        self.calls = []

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def lower(self):
        self.calls.append("lower")
        super().lower()


@pytest.fixture
def dialog(qtbot):
    dlg = QDialog()
    qtbot.addWidget(dlg)
    QVBoxLayout(dlg)
    return dlg


def test_turning_the_backdrop_off_pauses_a_fractal(dialog, monkeypatch):
    from spacr.qt import preferences

    backdrop = _Fractal(dialog)
    backdrop.show()
    dialog._spacr_popup_backdrop = backdrop
    monkeypatch.setattr(preferences, "get_popup_backdrop", lambda: "off")
    assert glass._install_the_backdrop(dialog) is None
    assert backdrop.calls == ["pause"]
    assert backdrop.isHidden()


def test_the_same_fractal_theme_resumes_the_existing_backdrop(dialog,
                                                              monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.widgets import ambient

    backdrop = _Fractal(dialog)
    backdrop.hide()
    dialog._spacr_popup_backdrop = backdrop
    monkeypatch.setattr(preferences, "get_popup_backdrop",
                        lambda: ambient.SPACEOUT_THEME)
    built = []
    monkeypatch.setattr(ambient, "install_ambient",
                        lambda *a, **k: built.append(k) or QWidget(dialog))
    assert glass._install_the_backdrop(dialog) is backdrop
    assert built == []
    assert backdrop.calls == ["resume", "lower"]
    assert not backdrop.isHidden()
    assert dialog._spacr_popup_backdrop is backdrop
    assert dialog.property("spacrIndependentBackdrop") is True
