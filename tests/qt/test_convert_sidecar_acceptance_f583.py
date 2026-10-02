"""Real threaded Convert clicks discover barcode.txt and publish a filled map."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import QSettings, QThread, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt import preferences
from spacr.qt.screens import convert as module
from spacr.qt.theme import apply_qpalette, palette_for, stylesheet


@pytest.mark.parametrize('mode', ['light', 'dark'])
def test_threaded_buttons_discover_sidecar_fill_map_and_report_mismatch(
        qtbot, qapp, tmp_path, monkeypatch, mode):
    """Run real TIFF conversion; only preferences and thread observation are adapted."""
    prefs = QSettings(str(tmp_path / 'preferences.ini'), QSettings.IniFormat)
    monkeypatch.setattr(preferences, '_settings', lambda: prefs)
    preferences._set_show_alpha_features(True)
    src, dst = tmp_path / 'raw', tmp_path / 'converted'
    pixels = np.arange(64, dtype=np.uint16).reshape(8, 8)
    for well in ('A01', 'A02'):
        (src / well).mkdir(parents=True)
        tifffile.imwrite(src / well / 'field01_C1.tif', pixels)
    sidecar = src / 'barcode.txt'
    sidecar.write_text('0001\n')
    records = tmp_path / 'samples.csv'
    records.write_text('barcode,well,strain,compound\n0001,A01,RH,DMSO\n')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [records, *src.rglob('*')] if p.is_file()}
    observed = []
    real_prepare = module.cvt._prepare_conversion_barcodes

    def observed_prepare(settings, *args):
        observed.append((QThread.currentThread() == qapp.thread(), dict(settings)))
        return real_prepare(settings, *args)

    monkeypatch.setattr(module.cvt, '_prepare_conversion_barcodes', observed_prepare)
    apply_qpalette(qapp, theme=mode)
    qapp.setStyleSheet(stylesheet(theme=mode))
    host = QWidget()
    qtbot.addWidget(host)
    host.setObjectName('F583SidecarAcceptance')
    host.setAttribute(Qt.WA_StyledBackground, True)
    host.setStyleSheet('QWidget#F583SidecarAcceptance { background-color: '
                      + palette_for(mode)['bg'] + '; }')
    screen = module.ConvertScreen(threaded=True)
    QVBoxLayout(host).addWidget(screen)
    screen.set_source(str(src))
    screen.set_destination(str(dst))
    screen._barcode_source.setText(str(records))
    assert screen._barcode_assignments.text() == ''
    host.resize(1180, 860)
    host.show()
    with qtbot.waitSignal(screen.job_finished, timeout=30000):
        qtbot.mouseClick(screen._btn_preview, Qt.LeftButton)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    assert screen.can_convert() and screen.preview_row_count() == 2
    assert not dst.exists()
    with qtbot.waitSignal(screen.job_finished, timeout=30000):
        qtbot.mouseClick(screen._btn_convert, Qt.LeftButton)
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=10000)
    assert len(observed) == 2 and all(not on_gui for on_gui, _ in observed)
    assert all(not settings['plate_barcodes'] for _, settings in observed)
    bundle = dst / 'plate_barcode_linkage'
    receipt = json.loads((bundle / 'complete.json').read_text())
    assert receipt['complete'] and receipt['plate_barcodes'] == {'plate1': '0001'}
    assert receipt['barcode_origins']['plate1'] == [
        {'kind': 'barcode.txt', 'path': str(sidecar), 'source_plate': 'raw'}]
    assert receipt['input_sha256'][str(sidecar)] == source_hashes[str(sidecar)]
    frame = pd.read_csv(bundle / 'plate_map_lims.csv', dtype=str)
    assert frame[['plateID', 'rowID', 'columnID', 'plate_barcode', 'strain', 'compound']].values.tolist() == [
        ['plate1', 'r1', 'c1', '0001', 'RH', 'DMSO']]
    mismatches = pd.read_csv(bundle / 'plate_barcode_mismatches.csv', dtype=str)
    assert len(mismatches) == 1
    assert mismatches['well'].tolist() == ['A02']
    assert mismatches['kind'].tolist() == ['well_not_in_lims']
    assert receipt['linked_wells'] == 1 and receipt['mismatches'] == 1
    summary = screen._summary.toPlainText()
    assert 'A02' in summary and 'well_not_in_lims' in summary
    assert 'plate_map_lims.csv' in summary and 'plate_barcode_mismatches.csv' in summary
    converted = list(dst.glob('*.tif'))
    assert len(converted) == 2
    assert all(np.array_equal(np.squeeze(tifffile.imread(path)), pixels) for path in converted)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in source_hashes.items())
    evidence = os.environ.get('SPACR_F583_SIDECAR_EVIDENCE')
    if evidence:
        directory = Path(evidence) / mode
        directory.mkdir(parents=True, exist_ok=True)
        qtbot.wait(30)
        assert host.grab().save(str(directory / 'convert-sidecar.png'))
        for path in bundle.iterdir():
            if path.is_file():
                (directory / path.name).write_bytes(path.read_bytes())
        (directory / 'acceptance.json').write_text(json.dumps({
            'theme': mode, 'source_sha256': source_hashes, 'source_unchanged': True,
            'worker_calls': len(observed), 'all_work_off_gui_thread': True,
            'explicit_assignments': '', 'converted_tiffs': len(converted),
            'pixels_preserved': True, 'summary': summary,
            'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [*converted, *bundle.iterdir()] if p.is_file()},
        }, indent=2) + '\n')
