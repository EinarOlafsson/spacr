"""The field's card compositing is off unless asked for.

The field could composite the translucent card backgrounds, the field chrome
and the switch tracks into its frames on the shading thread (1e51bb403,
94dd4f57a, ff1b3c1f2). Offscreen that halved the GUI-thread cost of a frame,
but on the workstation's GNOME capture host (2560x1440, a real display,
interleaved A/B, 2026-10-12) it cost 8-10 painted frames a second: Home
48-50 fps with it against 57-58 without, Mask 41-43 against 51-52. The
likely cause is the GIL, which the shading thread's picture playback and
image draws hold while the GUI thread waits. So since 2026-10-12 it is off
by default, behind the internal preference ``prefs/composite_cards`` and the
environment variable ``SPACR_COMPOSITE_CARDS=1``.

Held here: the switch's default and precedence; with it off, a field ticked,
painted and changed runs none of the compositing (no plan, no walk, no
event-hub filter, no lent chrome or switch track), and its frame is the
frame a composited field paints, which the composite tests hold equal to the
classic paint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QFrame, QLineEdit, QWidget  # noqa: E402
from shiboken6 import isValid  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.widgets import ambient, field_fade  # noqa: E402
from spacr.qt.widgets.toggle import Toggle  # noqa: E402

pytestmark = pytest.mark.qt

ENV = "SPACR_COMPOSITE_CARDS"

SHEET = """
QFrame#OffCard { background: rgba(20, 30, 40, 150);
                 border: 1px solid rgba(200, 210, 220, 90);
                 border-radius: 10px; }
"""


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private, empty preference store, and no switch in the environment."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.delenv(ENV, raising=False)
    return path


def test_the_switch_is_off_unless_the_environment_or_preference_says(
        store, monkeypatch):
    assert ambient._composite_cards_on() is False
    preferences._settings().setValue("prefs/composite_cards", True)
    assert ambient._composite_cards_on() is True
    monkeypatch.setenv(ENV, "0")
    assert ambient._composite_cards_on() is False
    preferences._settings().setValue("prefs/composite_cards", False)
    monkeypatch.setenv(ENV, " 1 ")
    assert ambient._composite_cards_on() is True
    monkeypatch.setenv(ENV, "")
    assert ambient._composite_cards_on() is False


def test_a_preference_store_that_fails_leaves_it_off(store, monkeypatch):
    def broken():
        raise RuntimeError("no store")

    monkeypatch.setattr(preferences, "_settings", broken)
    assert ambient._composite_cards_on() is False


@pytest.fixture
def applied(qapp, store):
    """The preferences applied privately, which installs the field fade."""
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    preferences.apply_preferences_to_app(qapp)
    assert field_fade.field_fade_enabled()
    yield qapp
    field_fade.uninstall_field_fade(qapp)
    field_fade.invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _scene(qtbot):
    """A field behind a translucent card with fields and switches on it."""
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.setStyleSheet(SHEET)
    host.resize(420, 300)
    backdrop = ambient.AmbientWidget(host, theme="blobs", seed=3, fps=30)
    backdrop.setGeometry(host.rect())
    backdrop.lower()
    card = QFrame(host)
    card.setObjectName("OffCard")
    card.setGeometry(20, 20, 380, 260)
    edit = QLineEdit(card)
    edit.setGeometry(10, 10, 320, 30)
    edit.setText("12.5")
    small = QLineEdit(card)
    small.setGeometry(10, 50, 60, 24)
    switch = Toggle(parent=card)
    switch.setGeometry(10, 90, 200, 22)
    captioned = Toggle("Modified", card)
    captioned.setGeometry(10, 120, 200, 22)
    captioned.setChecked(True)
    anchor = QWidget(card)
    anchor.setFocusPolicy(Qt.StrongFocus)
    anchor.setGeometry(340, 10, 20, 20)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    anchor.setFocus()
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    return host, backdrop, card, (edit, small), (switch, captioned)


def _grab(host):
    return host.grab().toImage().convertToFormat(QImage.Format_RGB32)


def _frozen_grab(backdrop, host):
    backdrop._timer.stop()
    backdrop.set_time(2.0)
    backdrop._producer_box[0].stop()
    return _grab(host)


def test_off_runs_none_of_the_compositing(applied, qtbot, monkeypatch):
    calls = []

    def record(name):
        def called(*args, **kwargs):
            calls.append(name)
        return called

    for name in ("_compose_cards", "_draw_cards", "_static_part",
                 "_lend_part", "_restore_card_backgrounds"):
        monkeypatch.setattr(ambient, name, record(name))
    for name in ("__init__", "tick", "invalidate", "eventFilter", "_rebuild",
                 "_walk"):
        monkeypatch.setattr(ambient._CardTints, name, record(name))
    monkeypatch.setattr(field_fade, "_lend_chrome", record("_lend_chrome"))

    host, backdrop, card, fields, switches = _scene(qtbot)
    assert backdrop._card_tints is None
    assert not backdrop.findChildren(ambient._CardTints)
    backdrop._timer.start()
    frames = backdrop.frames_painted
    qtbot.waitUntil(lambda: backdrop.frames_painted > frames + 10,
                    timeout=10000)
    card.move(24, 24)
    fields[0].setFocus()
    switches[0].setChecked(True)
    backdrop.set_background_color(QColor(30, 40, 50))
    backdrop.set_backdrop(None)
    backdrop.set_time(1.5)
    qtbot.wait(200)
    backdrop.stop()
    backdrop.start()
    qtbot.wait(100)
    backdrop.repaint()
    assert calls == []
    producer = backdrop._producer_box[0]
    assert producer.cards is None and producer.composed is None
    assert backdrop._last_composed is None
    assert not card.testAttribute(Qt.WA_NoSystemBackground)
    assert not any(field in field_fade._LENT for field in fields)
    assert all(switch._lent_to is None for switch in switches)
    backdrop.stop()


def test_off_paints_what_the_composited_field_paints(applied, qtbot,
                                                     monkeypatch):
    host, backdrop, *_rest = _scene(qtbot)
    assert backdrop._card_tints is None
    off = _frozen_grab(backdrop, host)
    backdrop.stop()
    host.close()

    monkeypatch.setenv(ENV, "1")
    host, backdrop, card, fields, switches = _scene(qtbot)
    tints = backdrop._card_tints
    assert tints is not None
    backdrop._timer.stop()

    def ready():
        tints.tick(backdrop._producer_box[0])
        return tints.active

    qtbot.waitUntil(ready, timeout=10000)
    assert card.testAttribute(Qt.WA_NoSystemBackground)
    assert all(switch._lent_to is not None for switch in switches)
    backdrop.set_time(2.0)
    assert backdrop._last_composed[0] is tints.plan
    backdrop._producer_box[0].stop()
    composited = _grab(host)
    assert composited == off
    if isValid(backdrop):
        tints.invalidate()
        backdrop.stop()

