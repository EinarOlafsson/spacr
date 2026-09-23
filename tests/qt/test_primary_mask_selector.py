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
