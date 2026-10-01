"""Record the shared opening frames of the tool lessons on the current GUI.

Many lessons open on Home, on Home with the Help menu open, or on the empty
Measure or Classify host before their own tool appears. Those openings are
recorded once here, from one fresh window, so every lesson that reuses one
shows the same current layout. Nothing is run or downloaded.

The frames are written with ``capture`` like every other recording; the
rectangles of the Help actions and of the host-bar tool buttons are written
to ``openings.json`` so a lesson's spotlight can be placed on the control its
narration names.
"""
from __future__ import annotations

import time

HELP_ITEMS = (
    ('01_help_report', 'Report'),
    ('01_help_batch', 'Batch Runner'),
    ('01_help_distributed', 'Distributed Jobs'),
    ('01_help_data_manager', 'Data Manager'),
    ('01_help_project_browser', 'Project Browser'),
    ('01_help_pipeline_graph', 'Pipeline Graph'),
    ('01_help_dictionary', 'Feature Dictionary'),
)

HOSTS = (
    ('01_measure_host', 'measure'),
    ('01_classify_host', 'classify_merged'),
    ('01_image_umap_host', 'umap'),
)


def _plain(text):
    return text.replace('&', '').rstrip('.…').strip()


def record_openings(app, window, captures, capture, settle, write_json):
    from PySide6.QtCore import QPoint, QRect, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton

    def window_rect(widget, local=None):
        local = local if local is not None else widget.rect()
        top_left = widget.mapToGlobal(local.topLeft()) - window.mapToGlobal(QPoint(0, 0))
        return [top_left.x(), top_left.y(), local.width(), local.height()]

    def go_home():
        window._on_nav_selected('__home__')
        settle(1.0)
        if not window._startup.isVisible():
            raise RuntimeError('Home is not visible')

    record = {'help': {}, 'hosts': {}, 'home_tiles': {}}
    go_home()
    capture('00_home')
    for button in window.findChildren(QAbstractButton):
        key = button.property('moduleAppKey') or button.property('navKey')
        if key and button.isVisible() and window._startup.isAncestorOf(button):
            record['home_tiles'][str(key)] = window_rect(button)

    help_actions = [action for action in window.menuBar().actions()
                    if _plain(action.text()) == 'Help']
    if len(help_actions) != 1 or help_actions[0].menu() is None:
        raise RuntimeError('The current application has no unique Help menu')
    menu = help_actions[0].menu()
    for name, label in HELP_ITEMS:
        go_home()
        choices = [action for action in menu.actions()
                   if _plain(action.text()).lower() == label.lower()]
        if len(choices) != 1 or not choices[0].isEnabled():
            raise RuntimeError(f'The Help menu has no unique usable {label!r} action')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(help_actions[0]).center())
        settle(0.4)
        if not menu.isVisible():
            raise RuntimeError('The Help menu did not open')
        geometry = menu.actionGeometry(choices[0])
        QTest.mouseMove(menu, geometry.center())
        menu.setActiveAction(choices[0])
        settle(0.4)
        record['help'][name] = {'label': choices[0].text(),
                                'menu': window_rect(menu),
                                'action': window_rect(menu, QRect(geometry))}
        capture(name)
        menu.hide()
        settle(0.3)

    for name, key in HOSTS:
        go_home()
        window._on_nav_selected(key)
        deadline = time.monotonic() + 60
        while window._screens.get(key) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{key} did not open')
            settle(0.1)
        settle(2)
        screen = window._screens[key]
        buttons = {}
        for button in screen.window().findChildren(QAbstractButton):
            if not button.isVisible():
                continue
            label = button.text() or button.toolTip().split('\n')[0]
            key_name = button.property('moduleAppKey') or button.property('navKey')
            if key_name or label:
                buttons.setdefault(str(key_name or label), window_rect(button))
        record['hosts'][name] = {'module': key, 'buttons': buttons}
        capture(name)
    go_home()
    # Installer lessons end on Preferences > Performance over Home.
    from capture_home import record_performance
    record_performance(window, capture, settle)
    write_json(captures / 'openings.json', record)


def _open(window, settle, key):
    window._on_nav_selected(key)
    deadline = time.monotonic() + 60
    while window._screens.get(key) is None:
        if time.monotonic() > deadline:
            raise TimeoutError(f'{key} did not open')
        settle(0.1)
    settle(2)
    return window._screens[key]


def record_illumination_apply(app, window, stage, captures, capture, settle, write_json):
    """Measure reusing the model the Illumination lesson estimated.

    The model is estimated off camera with the lesson's own settings (the
    lesson records that run on camera), from the sixteen example fields, so
    the path set in Measure is a real saved model.
    """
    from pathlib import Path

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from spacr.illumination import illumination_settings, prepare_illumination_model

    plate = Path.home() / '.cache/spacr/example_data/plate1'
    merged = plate / 'merged'
    if not merged.is_dir():
        raise RuntimeError('The example plate has no merged folder')
    prepared = prepare_illumination_model(illumination_settings({
        'src': str(merged), 'channels': [0], 'illumination_correction': True,
        'illumination_estimator': 'polynomial', 'illumination_degree': 4,
        'illumination_per_plate': True, 'illumination_max_fields': 16,
        'illumination_dark': 0.0, 'illumination_qc': True,
        'illumination_on_missing': 'error'}), verbose=True)
    model = Path(prepared.model_path)
    if not model.is_file():
        raise RuntimeError('No illumination model was saved')
    screen = _open(window, settle, 'measure')
    values = {'src': str(plate), 'illumination_correction': True,
              'illumination_model': str(model)}
    for key, value in values.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise RuntimeError('No real Measure setting for ' + key)
        settle(.1)
    collected = screen._settings_model.collect()
    if any(collected.get(key) != value for key, value in values.items()):
        raise RuntimeError('Measure did not take the illumination settings')
    bar = screen._settings_search
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    settle(.3)
    bar._input.setFocus()
    QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(bar._input, 'illumination')
    settle(.8)
    shown = [key for key in ('illumination_correction', 'illumination_model')
             if key in bar.visible_keys()]
    if len(shown) != 2:
        raise RuntimeError('The settings search did not show the illumination settings: '
                           + repr(sorted(bar.visible_keys())[:40]) + ' level=' + str(bar.level()))
    field = screen._settings_model._widgets['illumination_model']
    screen._settings_scroll.ensureWidgetVisible(field)
    settle(.4)
    top_left = field.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
    capture('10_measure_apply_model')
    write_json(captures / 'illumination_apply.json', {
        'model': str(model), 'settings': values,
        'model_field': [top_left.x(), top_left.y(), field.width(), field.height()],
        'settings_panel': [*(lambda p: [p.x(), p.y()])(
            screen._settings_scroll.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))),
            screen._settings_scroll.width(), screen._settings_scroll.height()]})


def record_align_test_data(app, window, stage, captures, capture, settle, write_json, timeout=900):
    """Align & Stitch after its own Load test data button filled the form."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    screen = _open(window, settle, 'align')
    button = screen._btn_test_data
    if not button.isVisible() or not button.isEnabled():
        raise RuntimeError('Load test data is not usable on Align & Stitch')
    capture('02a_align_load_test_data_button')
    QTest.mouseClick(button, Qt.LeftButton)
    deadline = time.monotonic() + timeout
    while 'Press Plan' not in screen._status.text():
        if time.monotonic() > deadline:
            raise TimeoutError('The Align & Stitch test data did not load: ' + screen._status.text())
        settle(0.5)
    settle(1.5)
    top_left = button.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
    status = screen._status.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
    capture('02b_align_test_data_loaded')
    write_json(captures / 'align_test_data.json', {
        'status': screen._status.text(), 'source': screen._src_edit.text(),
        'button': [top_left.x(), top_left.y(), button.width(), button.height()],
        'status_rect': [status.x(), status.y(), screen._status.width(), screen._status.height()]})
