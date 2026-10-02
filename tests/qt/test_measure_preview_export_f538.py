"""Displayed-crop exports preserve pixels and never publish stale partial work."""
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacr.qt.widgets import measure_preview as mp
from tests.qt.test_measure_unmix_preview_f538 import plate, panel


def _entry():
    """Provide a nonuniform full-size crop with a stable object identity."""
    return dict(crop=np.arange(60, dtype=np.uint8).reshape(4, 5, 3),
                source_path='/source/field.npy', object_key=('/source/field.npy', 'cell', 3),
                label=3, area=20, bbox=(1, 2, 5, 7), included=True)


@pytest.mark.parametrize('unmixed', [False, True])
@pytest.mark.parametrize('primaries', ['rgb', 'cmy'])
def test_actual_button_exports_full_display_pixels_and_provenance(
        qtbot, plate, monkeypatch, tmp_path, unmixed, primaries):
    widget = panel(qtbot, plate, monkeypatch, threaded=True)
    if unmixed:
        widget._unmix_btn.click()
        qtbot.waitUntil(lambda: not widget._crop_running, timeout=10000)
    monkeypatch.setattr(widget, 'display_primaries', lambda: primaries)
    widget._render_grid()
    originals = {p: p.read_bytes() for p in plate[0].values()}
    expected = [c['crop'].copy() for c in widget._crops]
    if primaries != 'rgb':
        from spacr.crops import apply_display_primaries
        expected = [apply_display_primaries(c, primaries) for c in expected]
    destination = tmp_path / 'export'
    monkeypatch.setattr(mp.QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mp.QInputDialog, 'getText', lambda *a, **k: ('export', True))
    thread_ids = {}
    writer, publisher = mp._write_preview_crop_export, mp._publish_preview_crop_export

    def write(*args):
        """Observe the real writer's worker thread without replacing its work."""
        thread_ids['write'] = threading.get_ident()
        return writer(*args)

    def publish(*args):
        """Observe publication on the GUI thread after real staging."""
        thread_ids['publish'] = threading.get_ident()
        return publisher(*args)

    monkeypatch.setattr(mp, '_write_preview_crop_export', write)
    monkeypatch.setattr(mp, '_publish_preview_crop_export', publish)
    widget._export_btn.click()
    qtbot.waitUntil(lambda: not widget._export_running, timeout=10000)
    assert thread_ids['write'] != threading.get_ident()
    assert thread_ids['publish'] == threading.get_ident()
    manifest = json.loads((destination / 'provenance.json').read_text())
    assert manifest['crop_count'] == len(expected)
    assert manifest['display_primaries'] == primaries
    assert manifest['crop_settings']['channels'] == [0, 1, 0]
    assert manifest['checked_sources'] == [str(plate[0]['C01'])]
    assert manifest['source_files_modified'] is False
    assert manifest['batch_outputs_modified'] is False
    for record, pixels in zip(manifest['crops'], expected):
        np.testing.assert_array_equal(np.asarray(Image.open(destination / record['file'])), pixels)
        assert record['rgb_pixels_sha256'] == hashlib.sha256(pixels).hexdigest()
        assert record['shape'] == list(pixels.shape)
        assert record['object_key'][1] == 'cell'
        assert bool(record['unmixing']) is unmixed
        if unmixed:
            assert record['unmixing']['channels'] == [0, 1]
    assert all(p.read_bytes() == value for p, value in originals.items())
    assert not list(tmp_path.glob('.spacr-preview-*'))


@pytest.mark.parametrize('kind', ['empty', 'populated', 'file', 'symlink'])
def test_existing_destinations_are_never_replaced(tmp_path, kind):
    destination = tmp_path / 'export'
    if kind in {'empty', 'populated'}:
        destination.mkdir()
        if kind == 'populated':
            (destination / 'keep').write_text('original')
    elif kind == 'file':
        destination.write_text('original')
    else:
        destination.symlink_to(tmp_path / 'missing')
    result = mp._write_preview_crop_export(destination, [_entry()], {}, 'rgb', threading.Event())
    assert 'error' in result
    assert destination.is_symlink() if kind == 'symlink' else destination.exists()
    if kind == 'file':
        assert destination.read_text() == 'original'
    if kind == 'populated':
        assert (destination / 'keep').read_text() == 'original'
    assert not list(tmp_path.glob('.spacr-preview-*'))


def test_late_collision_preserves_destination_and_removes_stage(tmp_path):
    destination = tmp_path / 'export'
    result = mp._write_preview_crop_export(destination, [_entry()], {}, 'rgb', threading.Event())
    destination.mkdir()
    with pytest.raises(OSError):
        mp._publish_preview_crop_export(result)
    assert destination.is_dir() and not list(destination.iterdir())
    assert not list(tmp_path.glob('.spacr-preview-*'))


@pytest.mark.parametrize('cancel', [False, True])
def test_failure_or_cancellation_during_png_writes_leaves_no_partial_export(tmp_path, monkeypatch, cancel):
    stop = threading.Event()
    save = Image.Image.save
    calls = []

    def controlled_save(image, *args, **kwargs):
        """Interrupt between actual PNG writes to exercise staging cleanup."""
        calls.append(True)
        if len(calls) == 2:
            raise OSError('disk unavailable')
        save(image, *args, **kwargs)
        if cancel:
            stop.set()

    monkeypatch.setattr(Image.Image, 'save', controlled_save)
    result = mp._write_preview_crop_export(tmp_path / 'export', [_entry(), _entry()], {}, 'rgb', stop)
    assert result.get('cancelled') if cancel else result.get('error') == 'disk unavailable'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['cancel', 'settings', 'source', 'primaries'])
def test_completed_worker_cannot_publish_after_preview_changes(
        qtbot, plate, monkeypatch, tmp_path, change):
    widget = panel(qtbot, plate, monkeypatch, threaded=True)
    entered, release = threading.Event(), threading.Event()
    writer = mp._write_preview_crop_export

    def blocked(*args):
        """Hold a complete staged result while the GUI invalidates its snapshot."""
        result = writer(*args)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(mp, '_write_preview_crop_export', blocked)
    destination = tmp_path / 'export'
    widget._start_crop_export(destination)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert list(tmp_path.glob('.spacr-preview-*'))
        if change == 'cancel':
            widget.cancel_preview()
        elif change == 'settings':
            widget.apply_settings({'png_size': [12, 12]})
        elif change == 'source':
            widget.load_array_async(str(plate[0]['A01']))
        else:
            monkeypatch.setattr(widget, 'display_primaries', lambda: 'cmy')
            widget._render_grid()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget._export_running and not widget._crop_running,
                    timeout=10000)
    assert not destination.exists()
    assert not list(tmp_path.glob('.spacr-preview-*'))


def test_duplicate_source_basenames_keep_distinct_crops(tmp_path):
    first, second = _entry(), _entry()
    second['source_path'] = '/other/field.npy'
    second['object_key'] = ('/other/field.npy', 'cell', 3)
    second['crop'] = first['crop'] + 1
    destination = tmp_path / 'export'
    result = mp._write_preview_crop_export(destination, [first, second], {}, 'rgb', threading.Event())
    mp._publish_preview_crop_export(result)
    records = json.loads((destination / 'provenance.json').read_text())['crops']
    assert len({r['file'] for r in records}) == 2
    assert len({r['source_path'] for r in records}) == 2
    assert len({r['rgb_pixels_sha256'] for r in records}) == 2


@pytest.mark.parametrize('name,accepted', [('unused', False), ('../bad', True), ('', True)])
def test_dialog_cancel_or_invalid_name_creates_nothing(qtbot, plate, monkeypatch, tmp_path, name, accepted):
    widget = panel(qtbot, plate, monkeypatch)
    monkeypatch.setattr(mp.QFileDialog, 'getExistingDirectory', lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(mp.QInputDialog, 'getText', lambda *a, **k: (name, accepted))
    before = set(tmp_path.iterdir())
    widget._export_btn.click()
    assert not widget._export_running
    assert set(tmp_path.iterdir()) == before


def test_destination_created_inside_rename_is_not_overwritten(tmp_path, monkeypatch):
    destination = tmp_path / 'export'
    result = mp._write_preview_crop_export(destination, [_entry()], {}, 'rgb', threading.Event())
    real_qdir = mp.QDir

    class RacingDirectory:
        """Create another process's empty destination after the Python preflight."""
        def rename(self, source, target):
            """Exercise Qt's actual directory rename semantics at the race boundary."""
            Path(target).mkdir()
            return real_qdir().rename(source, target)

    monkeypatch.setattr(mp, 'QDir', RacingDirectory)
    with pytest.raises(OSError):
        mp._publish_preview_crop_export(result)
    assert destination.is_dir() and not list(destination.iterdir())
    assert not list(tmp_path.glob('.spacr-preview-*'))


def test_shutdown_discards_and_cleans_completed_stage(qtbot, plate, monkeypatch, tmp_path):
    widget = panel(qtbot, plate, monkeypatch, threaded=True)
    entered, release = threading.Event(), threading.Event()
    writer = mp._write_preview_crop_export

    def blocked(*args):
        """Retain a completed staging directory until shutdown starts."""
        result = writer(*args)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(mp, '_write_preview_crop_export', blocked)
    destination = tmp_path / 'export'
    widget._start_crop_export(destination)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    timer = threading.Timer(.05, release.set)
    timer.start()
    try:
        widget.shutdown()
    finally:
        release.set()
        timer.join()
    qtbot.waitUntil(lambda: not list(tmp_path.glob('.spacr-preview-*')), timeout=10000)
    assert not destination.exists()


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_export_action_fits_normal_preview_width(qtbot, plate, monkeypatch, theme):
    from spacr.qt import theme as themes
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
    monkeypatch.setattr("spacr.qt.preferences.get_theme", lambda: theme)
    themes.apply_qpalette(QApplication.instance(), theme)
    widget = panel(qtbot, plate, monkeypatch)
    host = QWidget()
    qtbot.addWidget(host)
    host.setObjectName("ExportCaptureHost")
    host.setAutoFillBackground(True)
    layout = QVBoxLayout(host)
    layout.addWidget(widget)
    host.setStyleSheet(themes.stylesheet(theme) +
                      "\nQWidget#ExportCaptureHost { background: " +
                      themes.palette_for(theme)["bg"] + "; }")
    host.resize(1200, 720)
    host.show()
    qtbot.waitExposed(host)
    assert widget._export_btn.isVisible()
    assert widget._export_btn.width() >= widget._export_btn.minimumSizeHint().width()
    assert widget._export_btn.geometry().right() < widget.width()
    import os
    capture = os.environ.get('SPACR_EXPORT_CAPTURE_DIR')
    if capture:
        Path(capture).mkdir(parents=True, exist_ok=True)
        host.grab().save(str(Path(capture) / f'measure-export-{theme}.png'))


def test_old_callback_cannot_retire_a_new_export(qtbot, plate, monkeypatch, tmp_path):
    widget = panel(qtbot, plate, monkeypatch, threaded=True)
    widget._start_crop_export(tmp_path / 'first')
    old_token, old_crop_token = widget._export_token, widget._crop_token
    qtbot.waitUntil(lambda: not widget._export_running, timeout=10000)
    entered, release = threading.Event(), threading.Event()
    writer = mp._write_preview_crop_export

    def blocked(*args):
        """Keep the new export active while an old completion arrives."""
        result = writer(*args)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(mp, '_write_preview_crop_export', blocked)
    widget._start_crop_export(tmp_path / 'second')
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        orphan = writer(tmp_path / 'orphan', [_entry()], {}, 'rgb', threading.Event())
        stage_path = Path(orphan['stage'].name)
        status = widget._status.text()
        widget._finish_crop_export(old_token, old_crop_token, orphan)
        assert widget._export_running
        assert not widget._export_btn.isEnabled()
        assert widget._cancel_btn.isEnabled()
        assert widget._status.text() == status
        assert not stage_path.exists()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget._export_running, timeout=10000)
    assert (tmp_path / 'second' / 'provenance.json').exists()
    assert not (tmp_path / 'orphan').exists()
