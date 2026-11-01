"""A Previous during a Next lap runs after it, in full, the other way.

The direction of a circuit tells the user which way they went through the
slides, so one circuit is never merged with another or shortened: the Next
lap finishes its whole turn clockwise, then the Previous lap runs a whole
turn anticlockwise.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def card(qtbot):
    from spacr.qt.widgets.setup_card import SetupCard

    QApplication.instance() or QApplication([])
    widget = SetupCard(mode="glow")
    qtbot.addWidget(widget)
    widget.resize(420, 300)
    widget._aim_at_the_cursor = lambda: False
    widget._at = widget._towards = 0.25
    return widget


def _run_out(card, limit=400):
    trail = []
    for _ in range(limit):
        card._tick()
        trail.append(card._at)
        if not card.spinning:
            break
    return trail


def test_a_previous_during_a_next_lap_waits_for_it(card):
    card.circuit(clockwise=True)
    trail = [card._at]
    for _ in range(10):
        card._tick()
        trail.append(card._at)

    card.circuit(clockwise=False)
    assert card.spinning is True
    trail += _run_out(card)

    assert card.spinning is False
    turn = trail.index(max(trail))
    assert max(trail) == pytest.approx(1.25, abs=1e-6)
    assert all(b >= a for a, b in zip(trail[:turn], trail[1:turn + 1]))
    assert all(b <= a for a, b in zip(trail[turn:], trail[turn + 1:]))
    assert card._at == pytest.approx(0.25, abs=1e-6)


def test_two_nexts_run_two_whole_laps(card):
    card.circuit(clockwise=True)
    card._tick()
    card.circuit(clockwise=True)
    trail = _run_out(card)

    assert card.spinning is False
    assert card._at == pytest.approx(2.25, abs=1e-6)
    assert all(b >= a for a, b in zip(trail, trail[1:]))
