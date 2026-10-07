"""A queued source load cannot become another field's primary mask."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('PySide6')
from spacr.qt.widgets.primary_mask_selector import PrimaryMaskSelector
from spacr.tiff_io import write_tiff

pytestmark = pytest.mark.qt


@pytest.fixture
def selector(qtbot):
    widget = PrimaryMaskSelector()
    qtbot.addWidget(widget)
    yield widget
    widget.shutdown()


def test_folder_advances_with_the_image_queue(qtbot, selector, tmp_path):
    primary = tmp_path / 'primary'
    primary.mkdir()
    for name, label in [('one', 7), ('two', 900)]:
        write_tiff(primary / (name + '.tif'), np.full((10, 10), label, dtype=np.uint16))
    selector.path.setText(str(primary))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'masks/one.tif')
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert np.all(selector.snapshot.labels == 7)
    selector.bind_field(tmp_path / 'two.tif', (10, 10), tmp_path / 'masks/two.tif')
    assert selector.snapshot is None
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert np.all(selector.snapshot.labels == 900)


def test_explicit_file_does_not_follow_a_different_queue_field(qtbot, selector, tmp_path):
    path = tmp_path / 'nuclei.tif'
    write_tiff(path, np.full((10, 10), 900, dtype=np.uint16))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'masks/one.tif')
    selector.path.setText(str(path))
    selector._source_changed()
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    selector.bind_field(tmp_path / 'two.tif', (10, 10), tmp_path / 'masks/two.tif')
    qtbot.waitUntil(lambda: bool(selector.error))
    assert selector.snapshot is None
    assert 'another field' in selector.error


def test_late_load_cannot_replace_the_latest_field(qtbot, selector, tmp_path, monkeypatch):
    import threading

    from spacr.qt.widgets import primary_mask_selector as module
    primary = tmp_path / 'primary'
    primary.mkdir()
    for name, label in [('one', 7), ('two', 900)]:
        write_tiff(primary / (name + '.tif'), np.full((10, 10), label, dtype=np.uint16))
    entered, finish = threading.Event(), threading.Event()
    original = module.read_primary_source
    def blocked(**request):
        if str(request['image_path']).endswith('one.tif'):
            entered.set()
            assert finish.wait(5)
        return original(**request)
    monkeypatch.setattr(module, 'read_primary_source', blocked)
    selector.path.setText(str(primary))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'masks/one.tif')
    qtbot.waitUntil(entered.is_set)
    selector.bind_field(tmp_path / 'two.tif', (10, 10), tmp_path / 'masks/two.tif')
    finish.set()
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert selector.snapshot.image_path.endswith('two.tif')
    assert np.all(selector.snapshot.labels == 900)


def test_file_and_folder_buttons_bind_actual_sources_and_cancel_keeps_selection(
        qtbot, selector, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.widgets import primary_mask_selector as module
    primary = tmp_path / 'primary'
    primary.mkdir()
    path = primary / 'one.tif'
    write_tiff(path, np.full((10, 10), 900, dtype=np.uint16))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'masks/one.tif')
    buttons = {button.text(): button for button in selector.findChildren(QPushButton)}
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *a: (str(path), ''))
    qtbot.mouseClick(buttons['File…'], Qt.LeftButton)
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert selector.snapshot.path == str(path)
    assert selector._bound_image == str(tmp_path / 'one.tif')
    monkeypatch.setattr(module.QFileDialog, 'getOpenFileName', lambda *a: ('', ''))
    qtbot.mouseClick(buttons['File…'], Qt.LeftButton)
    assert selector.path.text() == str(path)
    monkeypatch.setattr(module.QFileDialog, 'getExistingDirectory', lambda *a: str(primary))
    qtbot.mouseClick(buttons['Folder…'], Qt.LeftButton)
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert selector.path.text() == str(primary)
    monkeypatch.setattr(module.QFileDialog, 'getExistingDirectory', lambda *a: '')
    qtbot.mouseClick(buttons['Folder…'], Qt.LeftButton)
    assert selector.path.text() == str(primary)


def test_closing_during_a_load_drains_worker_and_discards_queued_source(
        qtbot, selector, tmp_path, monkeypatch):
    import threading

    from spacr.qt.widgets import primary_mask_selector as module
    entered, finish = threading.Event(), threading.Event()
    calls = []
    def blocked(**request):
        calls.append(request)
        entered.set()
        assert finish.wait(5)
        raise OSError('source disconnected')
    monkeypatch.setattr(module, 'read_primary_source', blocked)
    selector.path.setText(str(tmp_path))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'mask.tif')
    qtbot.waitUntil(entered.is_set)
    worker = selector._worker
    selector.bind_field(tmp_path / 'two.tif', (10, 10), tmp_path / 'mask2.tif')
    releaser = threading.Timer(.02, finish.set)
    releaser.start()
    selector.shutdown()
    releaser.join()
    assert not worker.isRunning()
    assert selector.snapshot is None and selector._pending is None
    qtbot.wait(10)
    assert len(calls) == 1


def test_worker_reports_decoded_count_and_a_read_error_without_escaping(qtbot, tmp_path):
    from spacr.qt.widgets.primary_mask_selector import _SourceWorker
    path = tmp_path / 'primary.tif'
    write_tiff(path, np.full((10, 10), 900, dtype=np.uint16))
    request = dict(source=path, image_path=tmp_path / 'field.tif', shape=(10, 10),
                   output_path=tmp_path / 'output.tif', bound_image=tmp_path / 'field.tif')
    worker = _SourceWorker(1, request)
    worker.run()
    assert worker.count == 1 and worker.result.labels[0, 0] == 900
    missing = _SourceWorker(2, dict(request, source=tmp_path / 'missing.tif'))
    missing.run()
    assert missing.result is None and missing.error
    worker.deleteLater()
    missing.deleteLater()


def test_saved_custom_classes_restore_with_explicit_field_binding(qtbot, selector, tmp_path):
    from spacr.qt.secondary_masks import read_primary_source
    path = tmp_path / 'one.tif'
    write_tiff(path, np.full((10, 10), 7, dtype=np.uint16))
    field, output = tmp_path / 'image.tif', tmp_path / 'masks/image.tif'
    source = read_primary_source(path, field, (10, 10), output, bound_image=field,
                                 primary_class='primary custom', secondary_class='secondary custom')
    selector.bind_field(field, (10, 10), output)
    selector.restore_source(source.provenance())
    qtbot.waitUntil(lambda: selector.snapshot is not None)
    assert selector.snapshot.primary_class == 'primary custom'
    assert selector.snapshot.secondary_class == 'secondary custom'
    assert selector.snapshot.identity == source.identity


def test_timeout_parks_an_independent_reader_without_publishing_after_close(
        qtbot, selector, tmp_path, monkeypatch):
    import threading
    import weakref

    from spacr.qt import bridge
    from spacr.qt.widgets import primary_mask_selector as module

    primary = tmp_path / 'primary'
    primary.mkdir()
    for name in ('one', 'two'):
        write_tiff(primary / (name + '.tif'), np.full((10, 10), 7, np.uint16))
    entered, finish = threading.Event(), threading.Event()
    calls, changes = [], []
    original = module.read_primary_source

    def blocked(**request):
        calls.append(request)
        entered.set()
        assert finish.wait(5)
        return original(**request)

    def immediate_timeout(worker, *, timeout_ms):
        assert timeout_ms == 5000
        return bridge.drain_thread(worker, timeout_ms=0)

    monkeypatch.setattr(module, 'read_primary_source', blocked)
    monkeypatch.setattr(module, 'drain_thread', immediate_timeout)
    selector.changed.connect(lambda: changes.append('changed'))
    selector.path.setText(str(primary))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'out/one.tif')
    qtbot.waitUntil(entered.is_set)
    worker = selector._worker
    reference = weakref.ref(worker)
    try:
        selector.bind_field(tmp_path / 'two.tif', (10, 10), tmp_path / 'out/two.tif')
        selector.shutdown()
        assert worker.isRunning() and worker.parent() is None
        assert selector._worker is None and selector._pending is None
        assert selector.snapshot is None
        assert bridge.parked_thread_count() >= 1
        change_count, status = len(changes), selector.status.text()
        finish.set()
        assert worker.wait(2000)
        qtbot.wait(10)
        assert len(calls) == 1 and len(changes) == change_count
        assert selector.snapshot is None and selector.status.text() == status
        selector.reload()
        qtbot.wait(10)
        assert len(calls) == 1 and selector._worker is None
    finally:
        finish.set()
        assert worker.wait(2000)
        bridge.prune_parked_threads()
    del worker
    assert reference() is None


def test_successful_shutdown_keeps_normal_worker_ownership(
        qtbot, selector, tmp_path, monkeypatch):
    import threading

    from spacr.qt.widgets import primary_mask_selector as module

    entered, finish = threading.Event(), threading.Event()

    def blocked(**_request):
        entered.set()
        assert finish.wait(5)
        raise OSError('source disconnected')

    monkeypatch.setattr(module, 'read_primary_source', blocked)
    selector.path.setText(str(tmp_path))
    selector.bind_field(tmp_path / 'one.tif', (10, 10), tmp_path / 'mask.tif')
    qtbot.waitUntil(entered.is_set)
    worker = selector._worker
    finish.set()
    selector.shutdown()
    assert not worker.isRunning() and worker.parent() is selector
    selector.shutdown()
    qtbot.wait(10)
    assert selector.snapshot is None and selector._worker is None


def test_native_selector_deletion_preserves_reader_past_real_shutdown_timeout(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    code = textwrap.dedent('''
        import threading
        from types import SimpleNamespace
        import numpy as np
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from spacr.qt import bridge
        from spacr.qt.widgets import primary_mask_selector as module
        try:
            import resource
        except ImportError:
            pass
        else:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        app = QApplication([])
        entered, finish = threading.Event(), threading.Event()
        def blocked(**_request):
            entered.set()
            assert finish.wait(20)
            return SimpleNamespace(labels=np.ones((10, 10), dtype=np.uint16))
        module.read_primary_source = blocked
        selector = module.PrimaryMaskSelector()
        selector.path.setText('/controlled/source')
        selector.bind_field('/controlled/one.tif', (10, 10), '/controlled/out.tif')
        assert entered.wait(2)
        worker = selector._worker
        selector.shutdown()
        assert worker.isRunning() and worker.parent() is None
        assert bridge.parked_thread_count() == 1
        selector.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(selector) and isValid(worker) and worker.isRunning()
        finish.set()
        assert worker.wait(2000)
        assert bridge.prune_parked_threads() == 0
        worker.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(worker)
        print('native-owner-deleted-reader-finished')
    ''')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen',
                       XDG_CONFIG_HOME=str(tmp_path / 'child-config'))
    root = Path(__file__).resolve().parents[2]
    environment['PYTHONPATH'] = str(root)
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                            cwd=root, env=environment, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'native-owner-deleted-reader-finished' in result.stdout
    assert 'QThread: Destroyed while thread' not in result.stderr
