"""Record only reachable Measure controls; never repair or move its crop dialog.

The main Normalize setting primes the first live preview. It is restored to
False with Live off, before any run. This does NOT certify passing True into
the batch API, and does not claim that shared setting is display-only.
"""
import capture_ui
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np


def record_controls(app, window, screen, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from capture_settings import require_unchanged_settings
    import capture_ui
    host = capture_ui.live(capture)

    def search_for(text):
        """Type into the settings search (host: click, select all, type)."""
        if host:
            QTest.mouseClick(bar._input, Qt.LeftButton)
            QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
            if text:
                QTest.keyClicks(bar._input, text)
            else:
                QTest.keyClick(bar._input, Qt.Key_Backspace)
        elif text:
            bar._input.setFocus()
            QTest.keyClicks(bar._input, text)
        else:
            bar.set_query('')

    def pick(combo, index):
        if host:
            capture_ui.select_combo(capture, combo, index)
            return
        combo.setFocus()
        QTest.keyClick(combo, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(combo, Qt.Key_Down)

    def opening(name):
        """Host: start the scene clip before its actions, so the pointer is seen doing them."""
        if host and capture.begin(name):
            capture_ui.wait(capture, 0.8, settle)
            return True
        return False

    def closing(name, opened, *point_at):
        if opened:
            capture_ui.point_at(capture, *point_at, dwell=1.5)
            capture.end()
        elif not host:
            capture(name)

    panel = screen._measure_preview
    before = deepcopy(screen._settings_model.collect())
    if before.get('normalize') is not False or screen._preview_switch.isChecked():
        raise RuntimeError('Expected the untouched raw-crop example with Live off')
    bar = screen._settings_search
    if host:
        if bar.level() != 'all':
            QTest.mouseClick(bar._disclosure, Qt.LeftButton)      # Essentials -> All settings
            settle(1)
    else:
        bar.set_level('all')
    clip = opening('04_preview_normalization_setup')
    search_for('normalize')
    settle()
    control = screen._settings_model._widgets['normalize']
    capture_ui.scroll_to(capture, screen._settings_scroll, control)
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
    closing('04_preview_normalization_setup', clip, control)
    clip = opening('05_live_all_channels')
    search_for('')
    capture_ui.scroll_to_value(capture, screen._settings_scroll, 0)
    if screen._usage_card.body.isVisible():
        visible_click(screen._usage_card.title_label)
    capture_ui.splitter_sizes(capture, screen._runtime_splitter, [1400, 300])
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
    if host:
        QTest.mouseClick(panel._paste_box, Qt.LeftButton)
    else:
        panel._paste_box.setFocus()
    QTest.keyClick(panel._paste_box, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(panel._paste_box, str(source))
    QTest.keyClick(panel._paste_box, Qt.Key_Return)
    ready()
    if Path(panel._data_path) != source or not panel._normalise.isChecked():
        raise RuntimeError('The requested field or initial preview contrast did not load')
    closing('05_live_all_channels', clip, panel)

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
    clip = opening('06_live_channel_zero')
    pick(box, channel_index)
    ready()
    channel = observation()
    if clip:
        capture_ui.point_at(capture, panel, dwell=2.0)
    else:
        closing('06_live_channel_zero', False)
    require_unchanged_settings(settings_live, screen._settings_model.collect())
    if host:
        pick(box, 0)              # "All channels restores the combined view"
    else:
        QTest.keyClick(box, Qt.Key_Home)
    ready()
    if clip:
        capture.end()
    restored = observation()
    if original != restored:
        raise RuntimeError('Restoring All channels did not restore exactly the same crops')
    target = merged / 'plate1_E02_9_1.npy'
    target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    fov = panel._fov_box
    index = next((i for i in range(fov.count()) if Path(str(fov.itemData(i))) == target), -1)
    if index < 0:
        raise RuntimeError('The other recorded example field is not in the actual field selector')
    clip = opening('07_live_second_field')
    pick(fov, index)
    ready()
    second = observation()
    closing('07_live_second_field', clip, panel)
    require_unchanged_settings(settings_live, screen._settings_model.collect())
    # Checked images (612): tick two fields; the grid groups crops by field.
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMenu
    picked = []

    tries = {'n': 0}

    def tick(frame):
        menus = [m for m in app.topLevelWidgets() if isinstance(m, QMenu) and m.isVisible()]
        if len(menus) != 1:
            # A real glide to the button takes longer than a synthetic click:
            # wait for the menu to open instead of giving up.
            tries['n'] += 1
            if host and tries['n'] < 60:
                QTimer.singleShot(200, lambda: tick(frame))
                return
            picked.append('no menu')
            return
        menu = menus[0]
        if frame:
            capture(frame)
        actions = [a for a in menu.actions() if a.isCheckable() and not a.isChecked()]
        if not actions:
            picked.append('nothing to tick')
            menu.close()
            return
        picked.append(actions[0].text())
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(actions[0]).center())

    checked = panel._checked_button
    clip = opening('07c_checked_two_fields')
    for frame in ((None, None) if host else ('07b_checked_images_menu', None)):
        tries['n'] = 0
        QTimer.singleShot(900, lambda frame=frame: tick(frame))
        QTest.mouseClick(checked, Qt.LeftButton)
        settle(1.2)
        ready()
    if len(panel._checked_sources) < 2:
        raise RuntimeError(f'Checked images did not hold two fields: {picked}')
    settle(1)
    if clip:
        capture_ui.point_at(capture, panel, dwell=2.0)
    else:
        closing('07c_checked_two_fields', False)
    if host:
        # Checked images -> Uncheck all images (the menu runs its own loop).
        def uncheck_all(attempt=0):
            menus = [m for m in app.topLevelWidgets() if isinstance(m, QMenu) and m.isVisible()]
            if len(menus) != 1 and attempt < 60:
                QTimer.singleShot(200, lambda: uncheck_all(attempt + 1))
                return
            if len(menus) == 1:
                clear = next((a for a in menus[0].actions()
                              if a.text().replace('&', '').startswith('Uncheck all')), None)
                if clear is not None:
                    QTest.mouseClick(menus[0], Qt.LeftButton, pos=menus[0].actionGeometry(clear).center())
                    return
                menus[0].close()
        QTimer.singleShot(900, uncheck_all)
        QTest.mouseClick(checked, Qt.LeftButton)
        settle(1.2)
        if panel._checked_sources:
            raise RuntimeError('Uncheck all images left checked fields')
    else:
        panel._uncheck_all_sources()
    settle(.8)
    if clip:
        capture.end()
    require_unchanged_settings(settings_live, screen._settings_model.collect())
    # Close Live before restoring the setting that controls the saved crops.
    clip = opening('08_saved_normalization_restored')
    visible_click(screen._preview_switch)
    if host:
        search_for('normalize')
    else:
        bar.set_query('normalize')
    settle()
    capture_ui.scroll_to(capture, screen._settings_scroll, control)
    settle()
    visible_click(control)
    require_unchanged_settings(before, screen._settings_model.collect())
    closing('08_saved_normalization_restored', clip, control)
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
