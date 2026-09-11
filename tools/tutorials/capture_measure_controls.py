"""Record only reachable Measure controls; never repair or move its crop dialog.

The main Normalize setting primes the first live preview. It is restored to
False with Live off, before any run. This does NOT certify passing True into
the batch API, and does not claim that shared setting is display-only.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np


def record_controls(app, window, screen, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from capture_settings import require_unchanged_settings

    panel = screen._measure_preview
    before = deepcopy(screen._settings_model.collect())
    if before.get('normalize') is not False or screen._preview_switch.isChecked():
        raise RuntimeError('Expected the untouched raw-crop example with Live off')
    bar = screen._settings_search
    bar.set_level('all')
    bar._input.setFocus()
    QTest.keyClicks(bar._input, 'normalize')
    settle()
    control = screen._settings_model._widgets['normalize']
    screen._settings_scroll.ensureWidgetVisible(control)
    settle()

    def visible_click(widget):
        centre = widget.mapToGlobal(widget.rect().center())
        if not widget.isVisible() or not app.primaryScreen().availableGeometry().contains(centre):
            raise RuntimeError('A requested control is not on the actual desktop')
        QTest.mouseClick(widget, Qt.LeftButton)
        settle()

    visible_click(control)
    if screen._settings_model.collect()['normalize'] is not True:
        raise RuntimeError('The main Normalize toggle did not turn on')
    capture('04_preview_normalization_setup')
    bar.set_query('')
    screen._settings_scroll.verticalScrollBar().setValue(0)
    if screen._usage_card.body.isVisible():
        visible_click(screen._usage_card.title_label)
    screen._runtime_splitter.setSizes([1400, 300])
    visible_click(screen._preview_switch)

    def ready():
        deadline = time.monotonic() + timeout
        settle(.5)
        while panel._data is None or panel._loads_in_flight:
            if time.monotonic() > deadline:
                raise TimeoutError(panel._status.text())
            settle(.2)
        settle(2)
        if not panel._crops or not panel.isVisible():
            raise RuntimeError('The live crop grid is empty or hidden')

    ready()
    # Use the visible path input to select the same published field explicitly,
    # not the random field initially sampled by the application.
    merged = Path(panel._data_path).parent
    source = merged / 'plate1_E02_1_1.npy'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    panel._paste_box.setFocus()
    QTest.keyClick(panel._paste_box, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(panel._paste_box, str(source))
    QTest.keyClick(panel._paste_box, Qt.Key_Return)
    ready()
    if Path(panel._data_path) != source or not panel._normalise.isChecked():
        raise RuntimeError('The requested field or initial preview contrast did not load')
    capture('05_live_all_channels')

    def observation():
        return {'source': str(panel._data_path), 'params': panel.current_params(),
                'objects': [{'label': int(c['label']), 'area': int(c['area']),
                             'category': c.get('category'),
                             'shape': list(c['crop'].shape),
                             'sha256': hashlib.sha256(c['crop'].tobytes()).hexdigest(),
                             'max': int(c['crop'].max()),
                             'equal_rgb': bool(np.array_equal(c['crop'][..., 0], c['crop'][..., 1])
                                               and np.array_equal(c['crop'][..., 0], c['crop'][..., 2]))}
                            for c in panel._crops]}

    original = observation()
    settings_live = deepcopy(screen._settings_model.collect())
    box = panel._channel_box
    channel_index = box.findText('Ch 0')
    if channel_index < 0:
        raise RuntimeError('Channel zero is not offered')
    box.setFocus()
    QTest.keyClick(box, Qt.Key_Home)
    for _ in range(channel_index):
        QTest.keyClick(box, Qt.Key_Down)
    ready()
    channel = observation()
    capture('06_live_channel_zero')
    require_unchanged_settings(settings_live, screen._settings_model.collect())
    QTest.keyClick(box, Qt.Key_Home)
    ready()
    restored = observation()
    if original != restored:
        raise RuntimeError('Restoring All channels did not restore exactly the same crops')
    target = merged / 'plate1_E02_9_1.npy'
    target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    fov = panel._fov_box
    index = next((i for i in range(fov.count()) if Path(str(fov.itemData(i))) == target), -1)
    if index < 0:
        raise RuntimeError('The other recorded example field is not in the actual field selector')
    fov.setFocus()
    QTest.keyClick(fov, Qt.Key_Home)
    for _ in range(index):
        QTest.keyClick(fov, Qt.Key_Down)
    ready()
    second = observation()
    capture('07_live_second_field')
    require_unchanged_settings(settings_live, screen._settings_model.collect())
    # Close Live before restoring the setting that controls the saved crops.
    visible_click(screen._preview_switch)
    bar.set_query('normalize')
    settle()
    screen._settings_scroll.ensureWidgetVisible(control)
    settle()
    visible_click(control)
    require_unchanged_settings(before, screen._settings_model.collect())
    capture('08_saved_normalization_restored')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash or hashlib.sha256(target.read_bytes()).hexdigest() != target_hash:
        raise RuntimeError('A source array changed')
    proof = {'original': original, 'single_channel': channel, 'restored': restored,
             'second_field': second, 'source_hashes': {str(source): source_hash, str(target): target_hash},
             'source_unchanged': True, 'batch_settings_restored': True,
             'batch_started': False, 'crop_dialog_opened': False,
             'application_layout_fixed': False}
    from measure_controls_evidence import check_controls
    check_controls(proof)
    proof['accepted'] = True
    write_json(captures / 'scientific_acceptance.json', proof)


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    from stage_lesson import DEFAULT_STAGE
    raise SystemExit(launch('measure', 'measure_visible_controls_v2',
                           ['--download', '--measure-preview-controls'], 300,
                           stage=DEFAULT_STAGE / 'measure_controls'))
