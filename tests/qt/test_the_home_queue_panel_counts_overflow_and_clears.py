"""The Home page's queue panel shows an overflow count and clears the queue.

Pinned behaviour of :class:`spacr.qt.widgets.home.QueuedPanel`:

* with more pending plates than it has rows for, it lists the first
  ``MAX_ROWS`` and says how many more are waiting;
* **Clear** empties the saved queue, hides the panel, reports how many went
  and announces it with ``queue_cleared``;
* a queue that cannot be opened is cleared of nothing: the panel says so by
  returning 0 and does not announce a clear.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel  # noqa: E402

from spacr.qt import plate_queue as pq  # noqa: E402
from spacr.qt.widgets.home import QueuedPanel  # noqa: E402

pytestmark = pytest.mark.qt


def _queue(tmp_path, monkeypatch, count):
    path = tmp_path / "queue.json"
    monkeypatch.setattr(pq, "_queue_path", lambda: path)
    queue = pq.PlateQueue()
    for n in range(count):
        queue.add(pq.QueueItem.build(
            "mask", {"src": f"/data/plate_{n:02d}"}, label=f"plate_{n:02d}"))
    return queue


def _texts(panel):
    return [label.text() for label in panel.findChildren(QLabel)]


def test_more_plates_than_rows_are_counted_not_listed(
        qtbot, tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, QueuedPanel.MAX_ROWS + 2)
    panel = QueuedPanel()
    qtbot.addWidget(panel)

    texts = _texts(panel)
    assert "+2 more" in texts
    assert "plate_00" in texts
    assert f"plate_{QueuedPanel.MAX_ROWS:02d}" not in texts


def test_clear_empties_the_queue_and_hides_the_panel(
        qtbot, tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, 3)
    panel = QueuedPanel()
    qtbot.addWidget(panel)
    panel.show()
    assert panel.isVisible()

    with qtbot.waitSignal(panel.queue_cleared, timeout=1000):
        removed = panel.clear_queue()

    assert removed == 3
    assert pq.PlateQueue().items() == []
    assert not panel.isVisible()


def test_a_queue_that_cannot_be_opened_clears_nothing(
        qtbot, tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, 0)
    panel = QueuedPanel()
    qtbot.addWidget(panel)

    def _explode(*_args, **_kwargs):
        raise OSError("the queue file is locked")

    monkeypatch.setattr(pq, "PlateQueue", _explode)

    with qtbot.assertNotEmitted(panel.queue_cleared):
        assert panel.clear_queue() == 0
    assert not panel.isVisible()
