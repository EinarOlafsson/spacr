"""Actual Convert controls call protected barcode linkage on their worker."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import QSettings, QThread, Qt
from PySide6.QtWidgets import QFileDialog

from spacr.qt import preferences
from spacr.qt.screens import convert as screen_module


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Keep feature visibility and language preferences out of the user's file."""
    settings = QSettings(str(tmp_path / 'prefs.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: settings)
    preferences._set_show_alpha_features(True)
    return preferences


@pytest.fixture
def inputs(tmp_path):
    """Real source TIFFs and a local barcode CSV with one mismatched well."""
    src = tmp_path / 'raw'
    for well in ['A01', 'A02']:
        (src / well).mkdir(parents=True)
        tifffile.imwrite(src / well / 'field01_C1.tif', np.arange(64, dtype=np.uint16).reshape(8, 8))
    records = tmp_path / 'records.csv'
    records.write_text('barcode,well,strain\n0001,A01,RH\n')
    return src, tmp_path / 'converted', records


def _screen(qtbot, prefs, inputs, threaded=False):
    """Populate real optional controls without bypassing normal preview/run paths."""
    src, dst, records = inputs
    screen = screen_module.ConvertScreen(threaded=threaded)
    qtbot.addWidget(screen)
    screen.resize(1180, 860)
    screen.show()
    screen.set_source(str(src))
    screen.set_destination(str(dst))
    screen._barcode_source.setText(str(records))
    screen._barcode_assignments.setText('plate1=0001')
    return screen


def test_actual_controls_preview_and_convert_show_mismatch_report(qtbot, prefs, inputs):
    screen = _screen(qtbot, prefs, inputs)
    source_bytes = {p: p.read_bytes() for p in inputs[0].rglob('*.tif')}
    assert screen.preview() and screen.can_convert()
    assert not inputs[1].exists()
    assert screen.run_convert()
    receipt = inputs[1] / 'plate_barcode_linkage/complete.json'
    assert json.loads(receipt.read_text())['mismatches'] == 1
    summary = screen._summary.toPlainText()
    assert 'well_not_in_lims' in summary and 'A02' in summary
    assert 'plate_map_lims.csv' in summary and 'plate_barcode_mismatches.csv' in summary
    assert all(p.read_bytes() == data for p, data in source_bytes.items())


def test_alpha_gate_hides_and_restores_values(qtbot, prefs, inputs):
    prefs._set_show_alpha_features(False)
    screen = _screen(qtbot, prefs, inputs)
    assert screen._barcode_panel.isHidden()
    prefs._set_show_alpha_features(True)
    prefs._apply_alpha_widgets(screen)
    assert not screen._barcode_panel.isHidden()
    assert screen._barcode_assignments.text() == 'plate1=0001'
    prefs._set_show_alpha_features(False)
    prefs._apply_alpha_widgets(screen)
    assert screen._barcode_panel.isHidden()
    assert screen._barcode_source.text() == str(inputs[2])


def test_optional_controls_invalidate_preview_and_report_bad_assignments(qtbot, prefs, inputs):
    screen = _screen(qtbot, prefs, inputs)
    assert screen.preview()
    screen._barcode_assignments.setText('wrong=0001')
    assert not screen.can_convert()
    assert screen.preview() and not screen.can_convert()
    assert screen.preview_row_count() == 2
    assert 'every output plate' in screen._summary.toPlainText()
    assert not inputs[1].exists()


def test_file_picker_cancel_preserves_existing_csv_and_selection_invalidates_plan(
        qtbot, prefs, inputs, monkeypatch):
    screen = _screen(qtbot, prefs, inputs)
    assert screen.preview()
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: ('', ''))
    qtbot.mouseClick(screen._barcode_pick, Qt.LeftButton)
    assert screen._barcode_source.text() == str(inputs[2]) and screen.can_convert()
    other = inputs[2].with_name('other.csv')
    other.write_bytes(inputs[2].read_bytes())
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(other), ''))
    qtbot.mouseClick(screen._barcode_pick, Qt.LeftButton)
    assert screen._barcode_source.text() == str(other) and not screen.can_convert()


def test_blank_csv_preserves_ordinary_gui_conversion(qtbot, prefs, inputs):
    screen = _screen(qtbot, prefs, inputs)
    screen._barcode_source.clear()
    assert screen.preview() and screen.run_convert()
    assert not (inputs[1] / 'plate_barcode_linkage').exists()
    assert len(list(inputs[1].glob('*.tif'))) == 2


def test_worker_captures_controls_and_runs_linkage_off_gui_thread(qtbot, qapp, prefs, inputs, monkeypatch):
    screen = _screen(qtbot, prefs, inputs, threaded=True)
    observed = []
    actual = screen_module.cvt._prepare_conversion_barcodes

    def prepare(settings, *args):
        """Record worker affinity while retaining actual CSV validation."""
        observed.append((QThread.currentThread() == qapp.thread(), dict(settings)))
        return actual(settings, *args)

    monkeypatch.setattr(screen_module.cvt, '_prepare_conversion_barcodes', prepare)
    with qtbot.waitSignal(screen.job_finished, timeout=15000):
        assert screen.preview()
        assert not screen._barcode_source.isEnabled()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    with qtbot.waitSignal(screen.job_finished, timeout=15000):
        assert screen.run_convert()
        assert not screen._barcode_assignments.isEnabled()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    assert len(observed) == 2 and not any(on_gui for on_gui, _values in observed)
    assert all(values['plate_barcodes'] == 'plate1=0001' for _, values in observed)
    assert screen._barcode_source.isEnabled()
    assert 'plate_map_lims.csv' in screen._summary.toPlainText()


def test_inline_preflight_failure_preserves_existing_bundle(qtbot, prefs, inputs):
    screen = _screen(qtbot, prefs, inputs)
    bundle = inputs[1] / 'plate_barcode_linkage'
    bundle.mkdir(parents=True)
    marker = bundle / 'keep.txt'
    marker.write_text('existing')
    assert screen.preview() and not screen.can_convert()
    assert 'already exists' in screen._summary.toPlainText()
    assert marker.read_text() == 'existing'
    assert not list(inputs[1].glob('*.tif'))


def test_summary_bounds_large_mismatch_preview(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'complete.json').write_text(json.dumps({'linked_wells': 0, 'mismatches': 120}))
    pd.DataFrame({'kind': ['missing'] * 120, 'well': [f'A{i}' for i in range(120)]}).to_csv(
        bundle / 'plate_barcode_mismatches.csv', index=False)
    actual = pd.read_csv
    observed = []

    def bounded_read(*args, **kwargs):
        """Verify the limit reaches the reader, not just the rendered table."""
        observed.append(kwargs.get('nrows'))
        return actual(*args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', bounded_read)
    text = screen_module._barcode_summary({'bundle': bundle})
    assert observed == [100]
    assert 'A99' in text and 'A119' not in text
    assert '100 of 120' in text


@pytest.mark.parametrize('mode', ['light', 'dark'])
def test_capture_real_convert_controls_in_both_themes(qtbot, prefs, inputs, mode):
    """Capture actual populated widgets when the caller requests evidence images."""
    from spacr.qt.theme import apply_qpalette, palette_for, stylesheet
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    app = QApplication.instance()
    apply_qpalette(app, theme=mode)
    app.setStyleSheet(stylesheet(theme=mode))
    screen = _screen(qtbot, prefs, inputs)
    host = QWidget()
    qtbot.addWidget(host)
    host.setObjectName('F583CaptureHost')
    host.setAttribute(Qt.WA_StyledBackground, True)
    host.setStyleSheet('QWidget#F583CaptureHost { background-color: '
                      + palette_for(mode)['bg'] + '; }')
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(screen)
    host.resize(1180, 860)
    host.show()
    assert screen.preview() and screen.run_convert()
    qtbot.wait(30)
    directory = os.environ.get('SPACR_F583_CAPTURE_DIR')
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        capture = host.grab()
        assert capture.toImage().pixelColor(0, 0).alpha() == 255
        assert capture.save(str(Path(directory) / f'convert-barcodes-{mode}.png'))
    assert screen._barcode_source.width() > 200
    assert screen._barcode_assignments.width() > 200
