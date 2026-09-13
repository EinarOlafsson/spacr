"""Record Mask's actual OPS fold without starting its legacy whole-run engine."""


def record_screen(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt.screens.mask import ops_page

    buttons = [w for w in window.findChildren(QAbstractButton) if w.isVisible()
               and w.property('moduleAppKey') == 'mask' and w.objectName() == 'AppTile']
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise ValueError('Expected one enabled Mask Home tile')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(2)
    screen = window._screens.get('mask')
    if screen is None or not screen.isVisible():
        raise ValueError('The actual Home click did not open Mask')
    switch = screen._ops_switch
    if not switch.isVisible() or not switch.isEnabled() or switch.isChecked():
        raise ValueError('Expected the visible, initially closed OPS toggle')
    capture('01_mask_host')
    QTest.mouseClick(switch, Qt.LeftButton)
    settle(2)
    manager = ops_page(screen)
    if (not switch.isChecked() or manager is None or manager.page is None
            or not manager.page.isVisible() or manager.page.app_key != 'ops'):
        raise ValueError('The real OPS toggle did not open its settings page')
    capture('02_ops_settings')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Visible Home to Mask to OPS navigation only',
        'route': ['mask', 'ops'], 'gui_workflow_completed': False,
        'run_clicked': False, 'inputs_injected': False,
        'segmentation_or_decoding_performed': False,
        'app_source_modified': False, 'published': False})


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    raise SystemExit(launch('ops', 'ops_1507_gui_v2', [], 120))
