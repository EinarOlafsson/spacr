"""Rows added later: nothing to add, and rows added while a drain is already queued."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import metadata_table as mt

pytestmark = pytest.mark.qt


@pytest.fixture
def panel(qtbot):
    widget = mt.MetadataTablePanel()
    qtbot.addWidget(widget)
    return widget


def _scheduled(monkeypatch):
    calls = []
    monkeypatch.setattr(mt, "_after",
                        lambda ms, owner, fn: calls.append((ms, fn)))
    return calls


def test_no_rows_to_add_later_schedules_nothing(panel, monkeypatch):
    calls = _scheduled(monkeypatch)
    panel._append_later([])
    assert calls == [] and panel._pending == []


def test_rows_added_while_a_drain_is_queued_join_it_instead_of_queueing_another(
        panel, monkeypatch):
    calls = _scheduled(monkeypatch)
    panel._append_later([{}, {}])
    panel._append_later([{}])
    assert len(panel._pending) == 3
    assert len(calls) == 1 and calls[0][0] == 0
