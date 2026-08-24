"""One rim, whatever the card is the size of.

"the rim is to thick and bright. make the rim and window look exactly like
the setup spacr window."

The arc is a length in pixels, so on a short perimeter it covers a much
larger fraction of the rim than it does on the first-run window -- two
fifths of a small popup against a sixth of the setup card. The run reads
as a thick bright band rather than as a highlight travelling round one,
and the same setting produces two different looks.

Also here: a glassed dialog rewrites its window flags ONCE. The detach
filter used to rewrite them again on the same Polish, and a native window
recreated after the translucent one was made comes back with square
opaque corners on some window managers.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF, Qt

from spacr.qt.widgets.setup_card import SetupCard


#: (width, height) of surfaces that really carry a card.
SURFACES = (
    (980, 700),   # the first-run window, which is the reference look
    (524, 533),   # Preferences
    (420, 300),   # a small confirmation popup
    (1200, 900),  # a settings panel on a large screen
)


def test_every_card_lights_the_same_fraction_of_its_rim(qapp):
    """The fraction is the look; it is the same at every size."""
    card = SetupCard(radius=18)
    spans = [card.accent_span(QRectF(0, 0, w, h)) for w, h in SURFACES]

    assert max(spans) - min(spans) < 1e-6, dict(zip(SURFACES, spans))
    # And it is the first-run window's own value, which is what "exactly
    # like the setup spacr window" names.
    reference = card.accent_span(QRectF(0, 0, 980, 700))
    assert all(abs(span - reference) < 1e-6 for span in spans)


def test_a_tiny_card_wears_the_same_rim_as_a_huge_one(qapp):
    """Including the extremes, where the old rule broke worst: a 120 px
    popup lit two thirds of its rim and a wall-sized panel a hairline."""
    card = SetupCard(radius=18)
    small = card.accent_span(QRectF(0, 0, 120, 90))
    huge = card.accent_span(QRectF(0, 0, 2400, 1400))

    assert abs(small - huge) < 1e-6
    assert 0.04 <= small <= 0.62


def test_a_glassed_dialog_rewrites_its_flags_once(qapp, qtbot, monkeypatch,
                                                  tmp_path):
    """The detach and the frameless hint land in the SAME call."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from PySide6.QtWidgets import QDialog

    from spacr.qt.widgets import glass

    dialog = QDialog()
    qtbot.addWidget(dialog)
    calls = []
    original = QDialog.setWindowFlags

    def counted(self, flags):
        if self is dialog:
            calls.append(flags)
        original(self, flags)

    monkeypatch.setattr(QDialog, "setWindowFlags", counted)
    glass.glass(dialog)

    assert len(calls) == 1, f"{len(calls)} flag rewrites, not one"
    flags = calls[0]
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.Window
    assert dialog.property(glass.DETACHED)


def test_the_detach_filter_leaves_a_glassed_dialog_alone(qapp, qtbot):
    """Its marker is what stops the second rewrite."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QDialog

    from spacr.qt import dialogs
    from spacr.qt.widgets import glass

    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.setProperty(glass.DETACHED, True)
    before = dialog.windowFlags()

    dialogs._DetachEveryDialog().eventFilter(dialog, QEvent(QEvent.Type.Polish))

    assert dialog.windowFlags() == before
