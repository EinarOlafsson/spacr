"""Import attribution remains usable when interpreter paths cannot be read."""

import os
import sysconfig

import spacr
from spacr.qt import timing


def test_import_attribution_survives_unavailable_interpreter_paths(monkeypatch):
    monkeypatch.setattr(timing, "_SPACR_ROOT", None)
    monkeypatch.setattr(timing, "_LIBRARY_DIRS", ())
    attempts = []

    def unavailable():
        attempts.append(True)
        raise OSError("interpreter paths unavailable")

    monkeypatch.setattr(sysconfig, "get_paths", unavailable)
    own_file = os.path.join(os.path.dirname(spacr.__file__), "qt", "app.py")

    assert timing._the_spacr_frame(own_file) == "qt/app.py"
    assert attempts
    assert timing._LIBRARY_DIRS == ()
