"""The field composites translucent card backgrounds off the GUI thread.

Qt paints every widget above the field on every field frame, and a rounded
translucent card background is a per-pixel alpha blend over the whole card:
3.2 megapixels a frame on Mask Generation, 5-6 ms of a loaded frame. The
backdrop now records those backgrounds into a plan, the shading thread
plays the plan into each frame together with the page under the field, and
the cards stop painting their own backgrounds while a frame composited with
the current plan is on screen.

These tests hold the mechanism on a small scene: composited and classic
frames are identical, the GUI-thread fallback is identical too, any move,
resize, show or restyle puts the cards back to painting themselves before
the next paint, and what cannot be composited is left alone.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, QRect, Qt  # noqa: E402
from PySide6.QtGui import QImage, QRegion  # noqa: E402
from PySide6.QtWidgets import QFrame, QWidget  # noqa: E402
from shiboken6 import delete, isValid  # noqa: E402

from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _cards_composited(monkeypatch):
    """Card compositing on: these tests are about it.

    Off by default since 2026-10-12 (``ambient._composite_cards_on``): on
    the workstation's GNOME capture host (2560x1440, real display) it cost
    8-10 painted fps, Home 48-50 against 57-58 and Mask 41-43 against
    51-52. The machinery stays, behind ``SPACR_COMPOSITE_CARDS=1``.
    """
    monkeypatch.setenv("SPACR_COMPOSITE_CARDS", "1")

SHEET = """
QFrame#Card { background: rgba(20, 30, 40, 150);
              border: 1px solid rgba(200, 210, 220, 90);
              border-radius: 10px; }
QFrame#Hover { background: rgba(1, 2, 3, 100); }
QFrame#Hover:hover { background: rgba(200, 0, 0, 100); }
"""


def _card(parent, rect, name="Card"):
    frame = QFrame(parent)
    frame.setObjectName(name)
    frame.setGeometry(rect)
    return frame


@pytest.fixture
def scene(qtbot):
    """A field behind two nested cards and every kind of non-card."""
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.setStyleSheet(SHEET)
    host.resize(460, 340)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    backdrop.lower()
    outer = _card(host, QRect(20, 20, 300, 200))
    inner = _card(outer, QRect(10, 10, 130, 90))
    _card(host, QRect(330, 20, 100, 120), "Hover")
    clear = QWidget(host)
    clear.setGeometry(330, 150, 100, 100)
    clear.setAttribute(Qt.WA_StyledBackground, True)
    bare = QWidget(clear)
    bare.setGeometry(0, 0, 100, 100)
    bare.setAttribute(Qt.WA_NoSystemBackground, True)
    filled = _card(host, QRect(20, 230, 140, 100), "Opaque")
    filled.setAutoFillBackground(True)
    _card(filled, QRect(5, 5, 120, 80))
    _card(host, QRect(170, 230, 40, 40))
    hidden = _card(host, QRect(220, 230, 100, 100))
    hidden.hide()
    QObject(host)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    yield host, backdrop, outer, inner
    if isValid(backdrop):
        backdrop._card_tints.invalidate()


def _settle(qtbot, backdrop):
    tints = backdrop._card_tints

    def ready():
        tints.tick(backdrop._producer_box[0])
        return tints.active

    qtbot.waitUntil(ready, timeout=10000)
    return tints


def _grab(host):
    return host.grab().toImage().convertToFormat(QImage.Format_RGB32)


def test_the_composited_frame_is_the_classic_frame(scene, qtbot):
    host, backdrop, outer, inner = scene
    tints = _settle(qtbot, backdrop)
    assert tints._cards == [outer, inner]
    assert outer.testAttribute(Qt.WA_NoSystemBackground)
    backdrop._producer_box[0].stop()
    composited = _grab(host)
    assert backdrop._last_composed[0] is tints.plan
    gui_thread = backdrop._last_composed = None
    gui_thread = _grab(host)
    tints.invalidate()
    assert not outer.testAttribute(Qt.WA_NoSystemBackground)
    classic = _grab(host)
    assert composited == classic
    assert gui_thread == classic


def test_a_setter_republishes_a_composited_frame(scene, qtbot):
    host, backdrop, outer, inner = scene
    tints = _settle(qtbot, backdrop)
    backdrop.set_time(2.0)
    producer = backdrop._producer_box[0]
    assert producer.composed[0] is tints.plan
    assert backdrop._last_composed is producer.composed


def test_a_move_resize_or_restyle_falls_back_before_the_next_paint(scene,
                                                                   qtbot):
    host, backdrop, outer, inner = scene
    tints = _settle(qtbot, backdrop)
    plan = tints.plan
    outer.move(26, 24)
    assert not tints.active and tints.plan is None
    assert not outer.testAttribute(Qt.WA_NoSystemBackground)
    assert backdrop._producer_box[0].cards is None
    tints = _settle(qtbot, backdrop)
    assert tints.plan is not plan
    assert tints.plan.items[0][2:4] == (26, 24)
    inner.resize(150, 95)
    assert not tints.active
    _settle(qtbot, backdrop)
    images = dict(tints._images)
    outer.setProperty("spacrTint", 1)
    assert not tints.active and not tints._images and images
    _settle(qtbot, backdrop)
    _card(host, QRect(220, 250, 120, 80)).show()
    assert not tints.active
    tints = _settle(qtbot, backdrop)
    assert len(tints._cards) == 3
    backdrop._producer_box[0].stop()
    composited = _grab(host)
    tints.invalidate()
    assert composited == _grab(host)


def test_a_new_page_or_background_falls_back(scene, qtbot):
    host, backdrop, outer, inner = scene
    tints = _settle(qtbot, backdrop)
    backdrop.set_backdrop(None)
    assert not tints.active
    _settle(qtbot, backdrop)
    backdrop.set_background_color("#203040")
    assert not tints.active
    _settle(qtbot, backdrop)
    backdrop.stop()
    assert not tints.active
    assert not outer.testAttribute(Qt.WA_NoSystemBackground)


def test_what_cannot_be_composited_keeps_the_classic_paint(scene, qtbot):
    host, backdrop, outer, inner = scene
    tints = backdrop._card_tints
    producer = backdrop._producer_box[0]
    tints.enabled = False
    for _ in range(5):
        tints.tick(producer)
    assert tints.plan is None and not tints.active
    tints.enabled = True
    backdrop._corner_radius = 6
    for _ in range(5):
        tints.tick(producer)
    assert tints.plan is None
    backdrop._corner_radius = 0
    tints.tick(None)
    assert tints._producer is None and tints.plan is None
    masked = _card(host, QRect(330, 255, 125, 80))
    masked.setMask(QRegion(0, 0, 120, 75))
    masked.show()
    opaque = QFrame(host)
    opaque.setGeometry(220, 230, 120, 100)
    opaque.setAttribute(Qt.WA_OpaquePaintEvent, True)
    _card(opaque, QRect(0, 0, 120, 100))
    opaque.show()
    tints = _settle(qtbot, backdrop)
    assert tints._cards == [outer, inner]


def test_a_scene_with_no_cards_never_switches(qtbot):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(200, 150)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    tints = backdrop._card_tints
    for _ in range(6):
        tints.tick(backdrop._producer_box[0])
        qtbot.wait(20)
    assert tints.plan is not None and tints.plan.items == ()
    assert not tints.active and backdrop._producer_box[0].cards is None


def test_a_field_without_a_parent_composites_nothing(qtbot):
    orphan = ambient.AmbientWidget(None, theme="blobs", fps=30)
    qtbot.addWidget(orphan)
    shown = QWidget()
    qtbot.addWidget(shown)
    shown.show()
    tints = orphan._card_tints
    assert tints.eventFilter(shown, _ShowEvent()) is False
    producer = ambient._FrameProducer(orphan.engine, orphan._engine_lock,
                                      30, (10, 10))
    tints.tick(producer)
    assert tints.plan is None


class _ShowEvent:
    @staticmethod
    def type():
        return ambient.QEvent.Show


def test_a_destroyed_field_gives_the_cards_their_backgrounds_back(scene,
                                                                 qtbot):
    host, backdrop, outer, inner = scene
    tints = _settle(qtbot, backdrop)
    assert inner.testAttribute(Qt.WA_NoSystemBackground)
    gone = _card(host, QRect(0, 0, 10, 10))
    tints._shown.append(gone)
    delete(gone)
    backdrop._producer_box[0].stop()
    delete(backdrop)
    assert isValid(inner)
    assert not inner.testAttribute(Qt.WA_NoSystemBackground)
    assert not outer.testAttribute(Qt.WA_NoSystemBackground)
