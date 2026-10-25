"""An Overview that is loading holds back the window's warm imports.

On Windows the Mask pipeline import (torch) and the data-library warm imports
starved the Overview's metadata scan and first browser open for the
interpreter lock, so the full-app Overview workflow overran its budget. A hold
only postpones those imports while the Overview is busy; every other path
starts them as before. The usage sampler also stops asking for nvidia-smi on
every sample.
"""
import gc

import numpy as np
import pytest
import tifffile

from spacr.qt import gil_priority


class _Owner:
    pass


@pytest.fixture(autouse=True)
def _no_holds():
    gil_priority._HOLDS.clear()
    yield
    gil_priority._HOLDS.clear()


def test_a_hold_counts_while_busy_and_ends_on_release_deletion_or_the_limit(monkeypatch):
    owner = _Owner()
    busy = {"value": True}
    gil_priority.hold_background_imports(owner, lambda _owner: busy["value"])
    assert gil_priority.background_imports_held()
    busy["value"] = False
    assert not gil_priority.background_imports_held()
    busy["value"] = True
    gil_priority.release_background_imports(owner)
    assert not gil_priority.background_imports_held()
    gil_priority.hold_background_imports(owner)
    assert gil_priority.background_imports_held()
    del owner
    gc.collect()
    assert not gil_priority.background_imports_held()
    other = _Owner()
    gil_priority.hold_background_imports(other)
    monkeypatch.setattr(gil_priority, "HOLD_LIMIT_S", 0.0)
    assert not gil_priority.background_imports_held()


def test_a_warm_import_waits_for_the_hold_and_then_starts(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import app as app_module

    window = QWidget()
    qtbot.addWidget(window)
    monkeypatch.setattr(app_module, "_HELD_IMPORT_RETRY_MS", 20)
    started = []
    owner = _Owner()
    gil_priority.hold_background_imports(owner)
    assert app_module._start_unless_held(window, lambda: started.append(1) or "thread") is None
    qtbot.wait(80)
    assert started == []
    gil_priority.release_background_imports(owner)
    qtbot.waitUntil(lambda: started == [1], timeout=2000)
    assert app_module._start_unless_held(window, lambda: "now") == "now"


def test_the_usage_sampler_asks_for_nvidia_smi_once_a_minute(monkeypatch):
    from spacr.qt.screens import app_screen

    calls = []

    def probe():
        calls.append(1)
        return False

    monkeypatch.setattr(app_screen, "_nvidia_smi_available", probe)
    app_screen._NVIDIA_SMI_KNOWN.clear()
    for _ in range(5):
        assert app_screen._nvidia_smi_known() is False
    assert calls == [1]
    monkeypatch.setattr(app_screen, "_nvidia_smi_available", lambda: True)
    assert app_screen._nvidia_smi_known() is True
    known = app_screen._NVIDIA_SMI_KNOWN["value"]
    app_screen._NVIDIA_SMI_KNOWN["value"] = (known[0], known[1], known[2] - 61)
    monkeypatch.setattr(app_screen, "_nvidia_smi_available", probe)
    assert app_screen._nvidia_smi_known() is False
    assert calls == [1, 1]


def test_a_shown_overview_holds_while_loading_and_not_once_idle_or_retired(tmp_path, qtbot):
    from spacr.qt.widgets import qc_field_browser as browser

    for well in ("A01", "A02"):
        tifffile.imwrite(tmp_path / f"plate1_{well}_F1.tif", np.arange(120, dtype=np.uint16).reshape(10, 12))
    view = browser._FieldOverview(settings={"source": str(tmp_path), "channels": (0,)})
    qtbot.addWidget(view, before_close_func=lambda widget: widget._retire())
    assert not gil_priority.background_imports_held()
    view.show()
    view._indexed(browser._overview_index(tmp_path))
    assert gil_priority.background_imports_held()
    qtbot.waitUntil(lambda: view._normalization_ready and not view._loading(), timeout=15000)
    assert not gil_priority.background_imports_held()
    view._prepare_display()
    assert gil_priority.background_imports_held()
    view.hide()
    assert not gil_priority.background_imports_held()
    view.show()
    view._retire()
    assert not gil_priority.background_imports_held()


def test_the_metadata_step_holds_until_it_finishes(tmp_path, qtbot):
    from spacr.qt.widgets import qc_field_browser as browser

    wizard = browser._OverviewMetadataWizard(str(tmp_path), {})
    qtbot.addWidget(wizard)
    wizard._startup_timer.stop()
    assert not gil_priority.background_imports_held()
    wizard.show()
    assert gil_priority.background_imports_held()
    wizard.reject()
    assert not gil_priority.background_imports_held()


def test_a_source_without_a_database_reads_no_saved_groups_and_imports_nothing(tmp_path, monkeypatch):
    import sys

    from spacr.qt.widgets import qc_field_browser as browser

    monkeypatch.setitem(sys.modules, "spacr.measure", None)
    assert browser._overview_saved_groups(str(tmp_path)) == ([], [])
    assert not (tmp_path / "measurements").exists()


def test_the_thumbnail_pool_follows_the_cpus_this_process_may_use():
    import os

    from spacr.qt.widgets import qc_field_browser as browser

    usable = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    assert browser._OVERVIEW_THUMBNAIL_WORKERS == max(2, min(8, usable))
