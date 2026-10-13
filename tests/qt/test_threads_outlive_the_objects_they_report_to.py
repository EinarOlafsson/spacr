"""A worker thread never aborts the process or raises because its object closed.

The model zoo picker owns QThreads: closing it mid-download must park a
thread that will not stop, not destroy it running. The Python worker threads
of the SAM2 seed dialog, the CSV column field, the measurement migration and
the help search emit signals of objects the user may have closed; none of them
may let a RuntimeError reach ``threading.excepthook``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

import spacr

REPO_ROOT = Path(spacr.__file__).resolve().parents[1]

PREAMBLE = f'''
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["SPACR_NO_GL"] = "1"
sys.path.insert(0, {str(REPO_ROOT)!r})
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(["zoo-close-test"])
'''


def _run(body: str, tmp_path: Path):
    env = dict(os.environ)
    env.update(SPACR_HOME=str(tmp_path / "spacr-home"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               SPACR_LOG_DIR=str(tmp_path / "logs"),
               QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="")
    script = tmp_path / "case.py"
    script.write_text(PREAMBLE + textwrap.dedent(body), encoding="utf-8")
    return subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=120, env=env)


@pytest.mark.parametrize("how", ["close", "reject", "accept", "done"])
def test_closing_the_zoo_picker_mid_download_does_not_abort(tmp_path, how):
    """The fetch ignores the stop request for longer than the wait allows."""
    out = _run(f'''
        from spacr import model_zoo
        from spacr import _segmentation_backends as backends
        from spacr.qt.widgets import model_zoo_picker as zp
        from spacr.qt.bridge import parked_thread_count

        model_zoo.shared_catalogue_is_stale = lambda: False
        model_zoo.bioimageio_entries = lambda **kw: []
        backends._probe_blockers = lambda: {{}}
        zp.ModelZooPicker.refresh = lambda self: None
        started = []

        def slow_fetch(entry, folder, **kw):
            started.append(1)
            time.sleep(5)
            return folder

        model_zoo.fetch = slow_fetch
        dialog = zp.ModelZooPicker()
        dialog.show()
        dialog._thread = QThread(dialog)
        dialog._worker = zp._DownloadWorker(object(), "/nonexistent")
        dialog._worker.moveToThread(dialog._thread)
        dialog._thread.started.connect(dialog._worker.run)
        dialog._thread.start()
        deadline = time.monotonic() + 5
        while not started and time.monotonic() < deadline:
            app.processEvents()
        assert started
        began = time.monotonic()
        getattr(dialog, {how!r})(*([0] if {how!r} == 'done' else []))
        assert time.monotonic() - began < 4, "closing waited out the download"
        assert parked_thread_count() == 1, parked_thread_count()
        dialog.deleteLater()
        app.processEvents()
        print("CASE-END", flush=True)
    ''', tmp_path)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "Destroyed while thread" not in out.stderr, out.stderr[-3000:]
    assert "CASE-END" in out.stdout


def test_a_thread_that_stops_is_not_parked(qtbot):
    from PySide6.QtCore import QThread

    from spacr.qt.bridge import parked_thread_count
    from spacr.qt.widgets.model_zoo_picker import _retire_thread

    thread = QThread()
    thread.start()
    before = parked_thread_count()
    assert _retire_thread(thread, None, 2000) is True
    assert parked_thread_count() == before
    assert _retire_thread(None, None) is True


@pytest.fixture
def hook():
    seen = []
    previous = threading.excepthook
    threading.excepthook = lambda args: seen.append(args)
    yield seen
    threading.excepthook = previous


def _delete(obj):
    from shiboken6 import Shiboken

    Shiboken.delete(obj)


def test_the_sam2_dialog_worker_survives_its_dialog(qtbot, hook):
    import numpy as np

    from spacr.qt.screens import make_masks as mm

    gate = threading.Event()

    def propagate(frames, seeds, backward=False):
        gate.wait(5)
        return np.zeros((2, 8, 8), np.uint8), None

    dialog = mm._Sam2SeedDialog(np.zeros((2, 8, 8), np.uint8),
                                propagate=propagate)
    dialog.seeds = {0: np.ones((8, 8), np.uint8)}
    dialog.run()
    worker = dialog._thread
    _delete(dialog)
    gate.set()
    worker.join(5)
    assert not worker.is_alive()
    assert hook == []


def test_the_csv_column_field_worker_survives_its_field(qtbot, hook):
    from spacr.qt.screens import settings_model as sm

    gate = threading.Event()
    field = sm._CsvColumnField(key="score_column", default="area")
    field._read_columns = lambda paths, generation: (gate.wait(5), {})[1]
    worker = field.refresh()
    _delete(field)
    gate.set()
    worker.join(5)
    assert not worker.is_alive()
    assert hook == []


def test_the_measurement_migration_survives_its_signals(qtbot, hook, monkeypatch):
    from spacr.qt.screens import settings_model as sm

    def copy(source, target, report):
        report("x")
        return ["t"]

    monkeypatch.setattr(sm, "_copy_measurement_store", copy)
    signals = sm._MeasurementMigrationSignals()
    _delete(signals)
    sm._run_measurement_migration("a", "b", "sqlite", signals)
    monkeypatch.setattr(sm, "_copy_measurement_store",
                        lambda *a: (_ for _ in ()).throw(ValueError("x")))
    signals = sm._MeasurementMigrationSignals()
    _delete(signals)
    sm._run_measurement_migration("a", "b", "sqlite", signals)
    assert hook == []


def test_the_help_search_workers_survive_their_objects(qtbot, hook, monkeypatch):
    from spacr.qt import help_search as hs

    monkeypatch.setattr(hs, "build_index", lambda: [])
    loader = hs._IndexLoader()
    _delete(loader)
    loader._work()
    reach = hs._DocsReach()
    _delete(reach)
    reach.set_state("reachable")
    assert hook == []
