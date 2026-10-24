"""Scene clips for drivers that call ``capture(name)`` after the actions of a scene.

``Scenes(capture, settle)`` gives ``begin(name)`` (call it before the first
action of the scene) and ``end(name, *point_at)`` (call it where the driver used
to call ``capture(name)``). In the GNOME capture host the clip is recording
during the actions, so the viewer sees the pointer go to the controls and
operate them, and ``end`` rests the pointer on the controls the narration
names. Outside the host ``begin`` does nothing and ``end`` is ``capture(name)``
exactly as before (Xvfb recordings stay reproducible).
"""
from __future__ import annotations

import capture_ui


class Scenes:
    def __init__(self, capture, settle):
        self.capture = capture
        self.settle = settle
        self.open = None

    @property
    def host(self) -> bool:
        return capture_ui.live(self.capture)

    def begin(self, name, *, lead=0.8):
        """Host: start recording ``name`` before its actions. Returns True when recording."""
        if not self.host:
            return False
        if self.open == name:
            return True
        if self.open is not None:
            self.capture.end()
            self.open = None
        if self.capture.begin(name):
            self.open = name
            capture_ui.wait(self.capture, lead, self.settle)
            return True
        return False

    def end(self, name, *point_at, dwell=1.5, desktop=False):
        """Finish the scene ``name``: rest the pointer on ``point_at`` and stop the clip."""
        if not self.host:
            self.capture(name, desktop=True) if desktop else self.capture(name)
            return
        if self.open == name:
            capture_ui.point_at(self.capture, *point_at, dwell=dwell)
            self.capture.end()
            self.open = None
        else:
            # Not begun (scene filter or a scene recorded without actions): the old whole-state clip.
            if self.open is not None:
                self.capture.end()
                self.open = None
            self.capture(name, desktop=True) if desktop else self.capture(name)
