"""A card without a title still takes a title-row action, and a card whose
body waits is filled exactly once, however it comes to be shown.

``Card.add_title_action`` builds the title row on first use; a card with no
title gets a row of its own at the top holding just the action. A
``_CardBuiltWhenShown`` already marked shown while its parent was hidden,
and then revealed by the parent's show (so its own ``setVisible`` is not
called again), builds its body from ``showEvent``, and asking to build it again once it
is built does nothing.
"""
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from spacr.qt.widgets.card import Card, _CardBuiltWhenShown


def test_an_untitled_card_puts_its_action_in_a_row_at_the_top(qtbot):
    card = Card()
    qtbot.addWidget(card)
    button = QPushButton("Refresh")

    card.add_title_action(button)

    row = card._outer.itemAt(0).layout()
    assert row is card._title_row
    assert row.indexOf(button) >= 0
    assert card.title_label is None


def test_a_second_action_joins_the_same_row(qtbot):
    card = Card()
    qtbot.addWidget(card)
    first, second = QPushButton("A"), QPushButton("B")
    card.add_title_action(first)
    card.add_title_action(second)
    assert card._title_row.indexOf(first) < card._title_row.indexOf(second)


def test_a_body_revealed_by_its_parent_is_built_once(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    card = _CardBuiltWhenShown("Results")
    layout.addWidget(card)
    card.show()
    assert not card.isVisible()
    built = []

    def build():
        built.append(1)
        card.body_layout.addWidget(QLabel("filled"))

    card.build_body_when_first_shown(build)
    assert not card.body_is_built()

    host.show()
    qtbot.waitExposed(host)

    assert built == [1]
    assert card.body_is_built()
    assert card.body.findChild(QLabel).text() == "filled"
    assert card.ensure_body_built() is False
    assert built == [1]
