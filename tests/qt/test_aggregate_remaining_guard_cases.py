"""Targeted viewer, repaint and cache failures from the combined CI report."""
from types import SimpleNamespace
import weakref

import shiboken6
from PySide6.QtWidgets import QWidget






def test_geometry_replay_can_observe_a_show_missed_during_paint(qapp):
    from spacr.qt import gil_priority as gil

    widget = QWidget()
    before = widget.geometry()
    assert widget.isHidden()
    widget.show()
    gil._hub_geometry_after(SimpleNamespace(_object_kinds=frozenset()),
                            [(widget, True, before)])
    assert not widget.isHidden()
    widget.close()




def test_a_late_render_doorbell_cannot_repaint_a_destroyed_owner(qapp):
    from spacr.qt.widgets.ambient import _repaint_if_alive

    calls = []

    class Field(QWidget):
        def repaint(self, *_args):
            calls.append("painted")

    widget = Field()
    owner = weakref.ref(widget)
    shiboken6.delete(widget)
    _repaint_if_alive(owner)
    assert calls == []


