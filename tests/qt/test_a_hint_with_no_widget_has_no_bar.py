"""``hint_bar_of`` answers for a control that has no window yet.

A form is built before it is put on screen, and helpers that hand a
sentence to the bar run while it is being built. Asking for the bar of
something that is not in a window has to be an answer, not a crash.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from spacr.qt.widgets.hint_bar import (HintBar, explain_through_the_bar,  # noqa: E402
                                       hint_bar_of)

pytestmark = pytest.mark.qt


def test_no_widget_at_all_means_no_bar(qapp):
    assert hint_bar_of(None) is None
    assert explain_through_the_bar(None, "anything") is False, (
        "the caller is told to leave the tooltip where it is")


def test_a_widget_in_a_window_without_a_bar_is_told_so(qapp):
    window = QWidget()
    field = QLabel("threshold", window)

    assert hint_bar_of(field) is None
    assert explain_through_the_bar(field, "what it does") is False


def test_a_widget_finds_the_bar_of_the_window_it_is_in(qapp):
    window = QWidget()
    bar = HintBar(parent=window)
    field = QLabel("threshold", window)

    assert hint_bar_of(field) is bar
    assert explain_through_the_bar(field, "what it does") is True
