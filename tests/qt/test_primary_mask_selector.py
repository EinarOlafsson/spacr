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
