"""Record the actual Embeddings controls without inventing a crop input route."""


def record_screen(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton

    buttons = [w for w in window.findChildren(QAbstractButton) if w.isVisible()
               and w.property('moduleAppKey') == 'embeddings']
    if len(buttons) != 1 or not buttons[0].isEnabled():
        raise ValueError('Expected the actual enabled Embeddings Home tile')
    QTest.mouseClick(buttons[0], Qt.LeftButton)
    settle(2)
    screen = window._screens.get('embeddings')
    if screen is None or not screen.isVisible():
        raise ValueError('The visible Home click did not open Embeddings')
    # This is a version-specific introduction, NOT a pipeline success test.
    if screen._source.text() != 'no crops loaded' or screen._run.isEnabled():
        raise ValueError('The Embeddings input contract changed; re-author this introduction')
    capture('01_no_gui_crop_input')
    picker = screen._policy
    QTest.mouseClick(picker, Qt.LeftButton)
    settle(.3)
    if not picker.view().isVisible():
        raise ValueError('Channel policy menu did not open')
    capture('02_channel_policies', desktop=True)
    QTest.keyClick(picker.view(), Qt.Key_End)
    QTest.keyClick(picker.view(), Qt.Key_Return)
    settle(.3)
    if screen.spec().channel_policy != 'project':
        raise ValueError('Visible selection did not choose projection')
    capture('03_project_selected')
    QTest.mouseClick(picker, Qt.LeftButton)
    QTest.keyClick(picker.view(), Qt.Key_Home)
    QTest.keyClick(picker.view(), Qt.Key_Return)
    settle(.3)
    if screen.spec().channel_policy != 'per_channel' or screen._run.isEnabled():
        raise ValueError('Policy restoration or documented disabled input changed')
    capture('04_per_channel_restored')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Actual Home route and channel policy choices only',
        'gui_workflow_completed': False, 'crops_injected': False,
        'policies_selected': ['project', 'per_channel'], 'initial_source': screen._source.text(),
        'embed_enabled': screen._run.isEnabled(), 'model_started': False,
        'app_source_modified': False, 'published': False})


if __name__ == '__main__':
    from capture_barcode_saved_plots import launch
    raise SystemExit(launch('embeddings', 'embeddings_1507_gui', [], 120))
