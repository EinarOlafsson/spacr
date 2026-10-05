"""Jobs window behavior when registry data changes or cancellation fails."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal  # noqa: E402


class _Registry(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.handles = []

    def active(self):
        return list(self.handles)


class _Job:
    app_key = "mask"
    progress = None
    last_line = "working"
    user_visible = True

    def __init__(self):
        self.seconds = 5.0
        self.refuse_cancel = True
        self.cancel_reasons = []

    def fraction(self):
        return None

    def elapsed(self):
        return self.seconds

    def request_cancel(self, reason):
        if self.refuse_cancel:
            raise RuntimeError("job is no longer cancellable")
        self.cancel_reasons.append(reason)


@pytest.fixture
def registry(monkeypatch):
    from spacr.qt import bridge

    instance = _Registry()
    monkeypatch.setattr(bridge, "_REGISTRY", instance)
    return instance


def test_jobs_window_stays_empty_when_registry_is_unavailable(qtbot, monkeypatch):
    from spacr.qt import bridge
    from spacr.qt.widgets.activity_spinner import _JobsPanel

    def unavailable():
        raise RuntimeError("registry is unavailable")

    monkeypatch.setattr(bridge, "registry", unavailable)
    panel = _JobsPanel()
    qtbot.addWidget(panel)

    assert panel.job_count() == 0
    assert not panel._tick.isActive()
    assert panel._visible_handles() == []


def test_elapsed_column_ticks_without_rebuilding_jobs(qtbot, registry):
    from spacr.qt.widgets.activity_spinner import _JobsPanel

    job = _Job()
    registry.handles = [job]
    panel = _JobsPanel()
    qtbot.addWidget(panel)
    table = panel._table
    assert panel.job_count() == 1
    assert table.item(0, 2).text() == "0:00:05"
    row_button = table.cellWidget(0, 4)

    job.seconds = 3725.9
    panel._update_elapsed()
    assert table.item(0, 2).text() == "1:02:05"
    assert table.cellWidget(0, 4) is row_button

    job.seconds = -2.0
    panel._update_elapsed()
    assert table.item(0, 2).text() == "0:00:00"

    table.takeItem(0, 2)
    panel._update_elapsed()
    assert table.item(0, 2) is None


def test_cancel_can_be_retried_after_a_job_refuses_it(qtbot, registry):
    from spacr.qt.widgets.activity_spinner import _JobsPanel

    job = _Job()
    registry.handles = [job]
    panel = _JobsPanel()
    qtbot.addWidget(panel)
    button = panel._table.cellWidget(0, 4)

    button.click()
    assert button.isEnabled()
    assert button.text() == "Cancel"
    assert job.cancel_reasons == []

    job.refuse_cancel = False
    button.click()
    assert job.cancel_reasons == ["cancelled from the Jobs window"]
    assert not button.isEnabled()
    assert button.text() == "Cancelling…"

    panel.refresh()
    assert not panel._table.cellWidget(0, 4).isEnabled()
