"""The first-run setup dialog: its blurred backdrop, its card layout and its group headings.

The backdrop is decoration, so the same controls must be built and reachable
with it, without it, and when the card is too big for the screen and scrolls.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect                               # noqa: E402
from PySide6.QtWidgets import (QAbstractScrollArea,            # noqa: E402
                               QGraphicsBlurEffect, QGraphicsView, QLabel,
                               QWidget)

from spacr.qt import setup_screen                              # noqa: E402
from spacr.qt.widgets import setup_dialog as sd                # noqa: E402

_ALL_KEYS = {key for _heading, keys in sd.GROUPS for key in keys}


def _headings(dialog):
    return [label.text() for label in dialog.findChildren(QLabel)
            if label.text().startswith("<b>")]


def test_the_setup_dialog_with_and_without_its_backdrop(qtbot, qapp,
                                                        monkeypatch, subtests):
    """Backdrop view, card inset, plain layout, scrolling card and group headings."""
    parent = QWidget()
    parent.resize(640, 480)
    qtbot.addWidget(parent)

    with subtests.test(msg="a parent puts a blurred, frameless still behind the card"):
        dialog = sd.SetupDialog(parent)
        qtbot.addWidget(dialog)
        view = dialog._backdrop_view
        assert isinstance(view, QGraphicsView)
        assert view.frameShape() == QGraphicsView.NoFrame
        assert view.horizontalScrollBarPolicy() == view.verticalScrollBarPolicy()
        items = view.scene().items()
        assert len(items) >= 1
        effect = items[0].graphicsEffect()
        assert isinstance(effect, QGraphicsBlurEffect)
        assert effect.blurRadius() == pytest.approx(sd.BLUR)
        assert dialog.card.parent() is dialog

        dialog.show()
        qapp.processEvents()
        dialog.resize(900, 700)
        qapp.processEvents()
        assert view.geometry() == dialog.rect()
        assert dialog.card.geometry() == dialog.rect().adjusted(48, 48, -48, -48)
        dialog.close()

    with subtests.test(msg="a plain dialog resizes without touching a backdrop"):
        plain = sd.SetupDialog()
        qtbot.addWidget(plain)
        plain.show()
        qapp.processEvents()
        before = plain.card.parent()
        plain.resize(900, 700)
        qapp.processEvents()
        assert plain._backdrop_view is None
        assert plain.card.parent() is before
        asked = {q[0] for q in setup_screen.questions()}
        headings = _headings(plain)
        for heading, keys in sd.GROUPS:
            assert headings.count(f"<b>{heading}</b>") == int(
                any(key in asked for key in keys)), heading
        plain.close()

    with subtests.test(msg="a group whose questions all left is not drawn"):
        gone = ("spacr_mode", "hash_inputs", "spacr_ai", "easy_mode")
        kept = [q for q in setup_screen.questions() if q[0] not in gone]
        with monkeypatch.context() as patch:
            patch.setattr(setup_screen, "questions", lambda: kept)
            thinned = sd.SetupDialog()
        qtbot.addWidget(thinned)
        headings = _headings(thinned)
        assert "<b>How it runs</b>" not in headings
        assert "<b>How it looks</b>" in headings
        assert not set(gone) & set(thinned._editors)

    with subtests.test(msg="an oversized card without a backdrop scrolls the same controls"):
        with monkeypatch.context() as patch:
            patch.setattr(sd.SetupDialog, "_backdrop", lambda self, parent: None)
            patch.setattr(sd.SetupDialog, "screen", lambda self: SimpleNamespace(
                availableGeometry=lambda: QRect(0, 0, 360, 240)))
            small = sd.SetupDialog()
        qtbot.addWidget(small)
        assert small._backdrop_view is None
        assert small._card_scrolls
        areas = small.findChildren(QAbstractScrollArea)
        assert any(area.isAncestorOf(small.card) for area in areas)
        assert set(small._editors) == _ALL_KEYS
        for editor in small._editors.values():
            assert small.card.isAncestorOf(editor)
            assert editor.isEnabled()
