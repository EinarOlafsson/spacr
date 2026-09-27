"""make_thread pauses a running garbage collector for the pyplot pre-import
and switches it back on afterwards.

The other half of test_bridge_uncovered_paths'
``test_the_pyplot_pre_import_leaves_the_collector_as_it_found_it``: there the
collector starts off; here it starts on, is off while pyplot is imported,
and is on again when ``make_thread`` returns -- even when the import fails.
"""
import builtins
import gc
import sys

import pytest
from PySide6.QtCore import QThread

from spacr.qt import bridge as B


@pytest.mark.parametrize("import_works", [True, False])
def test_the_collector_is_off_during_the_import_and_on_after(monkeypatch,
                                                             import_works):
    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    seen = []

    def _watched(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            seen.append(gc.isenabled())
            if not import_works:
                raise ImportError("no matplotlib in this build")
            sys.modules.setdefault("matplotlib.pyplot",
                                   type(sys)("matplotlib.pyplot"))
            return sys.modules["matplotlib"]
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _watched)
    monkeypatch.setattr(B, "_REGISTRY", B.RunRegistry())

    was_enabled = gc.isenabled()
    gc.enable()
    try:
        thread, worker = B.make_thread(lambda settings: None, {},
                                       app_key="measure")
        after = gc.isenabled()
    finally:
        if not was_enabled:
            gc.disable()
        monkeypatch.undo()

    assert seen == [False]
    assert after is True
    assert isinstance(thread, QThread) and not thread.isRunning()
    assert isinstance(worker, B.PipelineWorker)
