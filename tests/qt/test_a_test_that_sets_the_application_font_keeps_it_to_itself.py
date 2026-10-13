"""A test that changes the application font does not change it for the next.

Preferences' Save and ``MainWindow`` both set the font application-wide
(``spacr.qt.app.apply_interface_font``). On dispatch 35012948690 one Save in
``test_preferences_tabs`` left Qt shard 1's third worker drawing in Open Sans
Light, and a close-mark measurement later on that worker found its glyph too
faint to see. ``tests/qt/conftest.py`` now puts the font back; these two tests
hold it there.

The second test reads what the first one saw, so it runs only after it: alone,
or in another order, it has nothing to compare and says so.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFont  # noqa: E402

_SEEN: dict = {}


def test_a_test_may_set_the_application_font(qapp):
    _SEEN["found"] = qapp.font().toString()
    changed = QFont(qapp.font())
    changed.setWeight(QFont.Weight.Light)
    changed.setPointSizeF(max(1.0, qapp.font().pointSizeF()) + 5.0)
    qapp.setFont(changed)
    # The premise: the change took, so the next test has one to undo.
    assert qapp.font().toString() != _SEEN["found"]
    _SEEN["set"] = qapp.font().toString()


def test_the_next_test_finds_the_font_the_last_one_found(qapp):
    if "set" not in _SEEN:
        pytest.skip("runs after test_a_test_may_set_the_application_font")
    assert qapp.font().toString() == _SEEN["found"], (
        f"the previous test's font {_SEEN['set']!r} outlived it")
