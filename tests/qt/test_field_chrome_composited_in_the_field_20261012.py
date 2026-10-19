"""The field composites field chrome and switch tracks with the cards.

Over the moving field every widget repaints on every frame. The field fade
drew the ramped container under each form field from a Python event filter
(a Python call, a painter and a blend per field per frame, about 0.6 ms of
a 2560x1440 Mask Generation frame), and each settings switch painted its
track and knob in Python. Since 2026-10-12 the chrome of every field
without the focus, and the track and knob of every switch at rest, are
played into the composited frame at the widget's place in the paint order
(``_CardTints``, ``_static_part``), and the widget leaves them out while
they are.

These tests hold that on a small scene: composited and classic frames are
identical, including a switch with a caption and fields too small to be
cards; a focus change (also into an editable combo box's editor), a
disable, a recall by the fade, a flip, a press or a silent state change of
a switch put the part back before the next paint (the silent one at that
paint); a polish or a reinstall of the fade does not paint it twice; and
what is not a field, has the focus, or has its own background under it is
left alone.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (QAbstractAnimation, QEvent, QPointF,  # noqa: E402
                            QRect, QSettings, Qt)
from PySide6.QtGui import (QFocusEvent, QImage, QMouseEvent,  # noqa: E402
                           QPainter)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame,  # noqa: E402
                               QLabel, QLineEdit, QSpinBox, QWidget)
from shiboken6 import delete, isValid  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.widgets import ambient, field_fade  # noqa: E402
from spacr.qt.widgets.toggle import Toggle  # noqa: E402

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
QFrame#ChromeCard { background: rgba(20, 30, 40, 150);
                    border: 1px solid rgba(200, 210, 220, 90);
                    border-radius: 10px; }
QLineEdit#Filled { background: rgba(200, 40, 40, 120); }
"""


@pytest.fixture
def applied(qapp, tmp_path, monkeypatch):
    """The preferences applied as the application applies them, privately."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    preferences.apply_preferences_to_app(qapp)
    assert field_fade.field_fade_enabled()
    yield qapp
    field_fade.uninstall_field_fade(qapp)
    field_fade.invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _field(kind, parent, rect, text="12.5"):
    field = kind(parent)
    field.setGeometry(rect)
    if isinstance(field, QLineEdit):
        field.setText(text)
    return field


@pytest.fixture
def scene(applied, qtbot):
    """A field behind a translucent card holding every kind of field."""
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.setStyleSheet(SHEET)
    host.resize(420, 380)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    backdrop.lower()
    card = QFrame(host)
    card.setObjectName("ChromeCard")
    card.setGeometry(20, 20, 380, 340)
    anchor = QWidget(card)
    anchor.setFocusPolicy(Qt.StrongFocus)
    anchor.setGeometry(340, 10, 20, 20)
    edit = _field(QLineEdit, card, QRect(10, 10, 320, 30))
    combo = QComboBox(card)
    combo.setEditable(True)
    combo.addItems(["cyto", "nuclei"])
    combo.setGeometry(10, 50, 320, 30)
    spin = _field(QSpinBox, card, QRect(10, 90, 320, 30))
    spin.setValue(7)
    disabled = _field(QLineEdit, card, QRect(10, 130, 320, 30), "off")
    disabled.setEnabled(False)
    small = _field(QLineEdit, card, QRect(10, 170, 60, 24), "s")
    filled = _field(QLineEdit, card, QRect(80, 170, 60, 24), "f")
    filled.setObjectName("Filled")
    filled.setAttribute(Qt.WA_StyledBackground, True)
    label = QLabel("caption", card)
    label.setGeometry(10, 200, 320, 30)
    hidden = QLineEdit(card)
    hidden.setGeometry(400, 400, 40, 20)
    switch = Toggle(parent=card)
    switch.setGeometry(10, 240, 200, 22)
    captioned = Toggle("Modified", card)
    captioned.setGeometry(10, 270, 200, 22)
    captioned.setChecked(True)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    anchor.setFocus()
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    backdrop._timer.stop()
    fields = {"edit": edit, "combo": combo, "spin": spin,
              "disabled": disabled, "small": small, "filled": filled,
              "switch": switch, "captioned": captioned, "card": card}
    yield host, backdrop, fields
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


def _same_as_classic(host, backdrop, qtbot):
    tints = _settle(qtbot, backdrop)
    backdrop._producer_box[0].stop()
    composited = _grab(host)
    assert backdrop._last_composed[0] is tints.plan
    backdrop._last_composed = None
    gui_thread = _grab(host)
    tints.invalidate()
    classic = _grab(host)
    backdrop.stop()
    backdrop.start()
    backdrop._timer.stop()
    return composited == classic and gui_thread == classic


_CHROMED = ("edit", "combo", "spin", "disabled", "small")


def test_field_chrome_and_switches_are_composited_alike(scene, qtbot):
    host, backdrop, fields = scene
    tints = _settle(qtbot, backdrop)
    lent = [fields[name] for name in _CHROMED]
    switches = [fields["switch"], fields["captioned"]]
    assert tints._parts == lent + switches
    assert all(field in field_fade._LENT for field in lent)
    assert fields["filled"] not in field_fade._LENT
    assert all(switch._lent_to is not None for switch in switches)
    assert tints._cards == [fields["card"]]
    assert len(tints.plan.items) == 8
    assert isinstance(tints.plan.items[1][0], ambient._ImagePlay)
    assert _same_as_classic(host, backdrop, qtbot)
    assert not field_fade._LENT
    assert all(switch._lent_to is None for switch in switches)


def test_a_switch_hands_its_track_back_before_it_changes(scene, qtbot):
    host, backdrop, fields = scene
    switch = fields["switch"]
    tints = _settle(qtbot, backdrop)
    switch.setChecked(True)
    assert not tints.active and switch._lent_to is None
    assert switch._spacr_static_picture() is None
    qtbot.waitUntil(
        lambda: switch._anim.state() != QAbstractAnimation.Running,
        timeout=5000)
    tints.invalidate()
    tints = _settle(qtbot, backdrop)
    assert switch in tints._parts
    assert _same_as_classic(host, backdrop, qtbot)
    tints = _settle(qtbot, backdrop)
    switch.mousePressEvent(QMouseEvent(
        QEvent.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert not tints.active
    assert switch._spacr_static_picture() is None
    switch._mouse_pressed = False
    tints = _settle(qtbot, backdrop)
    switch._knob_pos += 3.0
    assert tints.active
    switch.repaint()
    assert not tints.active and switch._lent_to is None


def test_a_focus_change_puts_the_chrome_back_before_the_next_paint(scene,
                                                                   qtbot):
    host, backdrop, fields = scene
    tints = _settle(qtbot, backdrop)
    QApplication.sendEvent(fields["card"], QFocusEvent(QEvent.FocusIn))
    assert tints.active
    QApplication.sendEvent(fields["edit"], QFocusEvent(QEvent.FocusIn))
    assert not tints.active and tints.plan is None
    assert not field_fade._LENT
    tints = _settle(qtbot, backdrop)
    QApplication.sendEvent(fields["combo"].lineEdit(),
                           QFocusEvent(QEvent.FocusOut))
    assert not tints.active
    _settle(qtbot, backdrop)
    host.activateWindow()
    fields["edit"].setFocus()
    qtbot.waitUntil(lambda: fields["edit"].hasFocus(), timeout=5000)
    tints = _settle(qtbot, backdrop)
    assert fields["edit"] not in tints._parts
    assert fields["edit"] not in field_fade._LENT
    assert fields["spin"] in field_fade._LENT
    assert _same_as_classic(host, backdrop, qtbot)


def test_a_disable_or_a_recall_puts_the_chrome_back(scene, qtbot):
    host, backdrop, fields = scene
    tints = _settle(qtbot, backdrop)
    fields["spin"].setEnabled(False)
    assert not tints.active and not field_fade._LENT
    tints = _settle(qtbot, backdrop)
    assert fields["spin"] in field_fade._LENT
    assert _same_as_classic(host, backdrop, qtbot)
    tints = _settle(qtbot, backdrop)
    field_fade.invalidate_field_fade()
    assert not tints.active and not field_fade._LENT
    tints = _settle(qtbot, backdrop)
    field_fade.uninstall_field_fade()
    assert not tints.active and not field_fade._LENT
    tints = _settle(qtbot, backdrop)
    assert tints._parts == [fields["switch"], fields["captioned"]]
    assert _same_as_classic(host, backdrop, qtbot)


def test_a_polish_or_a_reinstall_does_not_paint_it_twice(scene, qtbot):
    host, backdrop, fields = scene
    tints = _settle(qtbot, backdrop)
    assert field_fade._filter.eventFilter(fields["edit"],
                                          QEvent(QEvent.Polish)) is False
    field_fade._watch_fields(QApplication.instance())
    assert tints.active and fields["edit"] in field_fade._LENT
    assert _same_as_classic(host, backdrop, qtbot)


def test_a_destroyed_field_gives_the_chrome_back(scene, qtbot):
    host, backdrop, fields = scene
    _settle(qtbot, backdrop)
    gone = QLineEdit()
    field_fade._lend_chrome(gone, lambda: None)
    delete(gone)
    field_fade._take_back_chrome(gone)
    assert gone not in field_fade._LENT
    switch = fields["switch"]
    assert switch._lent_to is not None
    backdrop._producer_box[0].stop()
    delete(backdrop)
    assert not field_fade._LENT
    assert switch._lent_to is None
    dead = Toggle()
    delete(dead)
    ambient._take_back_part(dead)
    edit = fields["edit"]
    edit.setText("")
    image = QImage(edit.size(), QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    QApplication.processEvents()
    layer = field_fade._UNDERLAYS.get(edit)
    painter_of_the_fade = edit if layer is None else layer
    if layer is not None:
        assert layer.isVisible() and layer.geometry() == edit.geometry()
    painter_of_the_fade.render(image)
    assert any(bytes(image.constBits())), "the fade no longer paints"


def test_what_the_fade_does_not_paint_is_not_composited(scene, qtbot,
                                                        monkeypatch):
    host, backdrop, fields = scene
    assert field_fade._chrome_image(QLabel()) is None
    inner = fields["spin"].findChild(QLineEdit)
    assert field_fade._chrome_image(inner) is None
    tiny = QLineEdit()
    tiny.setFixedSize(1, 1)
    assert field_fade._chrome_image(tiny) is None
    image = QImage(4, 4, QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    field_fade.paint_field_fade(tiny, painter)
    painter.end()
    assert not any(bytes(image.constBits()))
    assert field_fade._chrome_image(fields["edit"]) is not None
    with monkeypatch.context() as patch:
        patch.setattr(field_fade, "_enabled", False)
        assert field_fade._chrome_image(fields["edit"]) is None
        patch.setattr(field_fade, "_filter", None)
        patch.setattr(field_fade, "_enabled", True)
        assert field_fade._chrome_image(fields["edit"]) is None
        field_fade._lend_chrome(tiny, lambda: None)
        assert tiny in field_fade._LENT
        field_fade._take_back_chrome(tiny)
        assert tiny not in field_fade._LENT
