"""A FlowLayout's items die with their C++ objects, whoever else holds them.

A ``FlowLayout`` used as a sub-layout (the Run row's button strip) is walked
by PySide's layout-ownership glue whenever Python asks the outer box for an
item: ``row.itemAt(0)`` makes ``row`` the Python parent of every item the
strip holds. An item Qt created itself has no C++ wrapper to report its
destruction, so after Qt removed a button and deleted its item, ``row``
still kept that item's wrapper alive and registered at the freed address.
The next item allocated there was handed the stale wrapper, and when its
foreign parent went, every item of the Overview's strip read as "already
deleted" inside ``FlowLayout.minimumSize``. PySide 6.10 then left the error
set across the next virtual call and freed ``FlowLayout.setGeometry``: the
SIGSEGV in ``QWidget.render`` that crashed Qt shard 2 on 2026-11-01.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget
from shiboken6 import isValid

from spacr.qt.widgets.flow import FlowLayout

pytestmark = pytest.mark.qt


def _strip_in_a_row(qtbot, count=3):
    actions = QWidget()
    qtbot.addWidget(actions)
    row = QHBoxLayout(actions)
    strip = FlowLayout(None, spacing=4)
    row.addLayout(strip)
    buttons = [QPushButton(f"Button {n}") for n in range(count)]
    for button in buttons:
        strip.addWidget(button)
    return actions, row, strip, buttons


def test_an_item_qt_deletes_is_never_left_alive_behind_a_foreign_parent(qtbot):
    actions, row, strip, buttons = _strip_in_a_row(qtbot)
    assert row.itemAt(0) is not None
    first = strip.itemAt(0)
    assert first.widget() is buttons[0]

    elsewhere = QWidget()
    qtbot.addWidget(elsewhere)
    QVBoxLayout(elsewhere).addWidget(buttons[0])
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()

    assert strip.count() == 2
    assert not isValid(first)
    assert all(isValid(strip.itemAt(index)) for index in range(strip.count()))
    assert [strip.itemAt(index).widget() for index in range(strip.count())] == buttons[1:]


def test_items_keep_answering_after_the_outer_row_is_walked_and_buttons_move(qtbot):
    actions, row, strip, buttons = _strip_in_a_row(qtbot, count=6)
    actions.show()
    elsewhere = QWidget()
    qtbot.addWidget(elsewhere)
    target = QVBoxLayout(elsewhere)
    for button in buttons[:3]:
        assert row.itemAt(0) is not None
        target.addWidget(button)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        gc.collect()
    for _ in range(50):
        strip.addWidget(QPushButton("again"))
        assert row.itemAt(0) is not None
    QCoreApplication.processEvents()

    assert strip.count() == 53
    assert all(isValid(strip.itemAt(index)) for index in range(strip.count()))
    assert strip.minimumSize().width() > 0
    assert strip.heightForWidth(200) > 0


def test_a_python_caller_of_take_at_owns_the_item_it_took(qtbot):
    actions, row, strip, buttons = _strip_in_a_row(qtbot)
    taken = strip.takeAt(1)
    assert taken.widget() is buttons[1]
    assert strip.count() == 2
    del taken
    gc.collect()
    assert [strip.itemAt(index).widget() for index in range(strip.count())] == [buttons[0], buttons[2]]
