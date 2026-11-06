"""Native Protocols model rebuilds retain valid selection ownership through retirement."""
import json
import sys
import traceback

import pytest

from PySide6.QtCore import QCoreApplication, QEvent, Qt, qInstallMessageHandler
from PySide6.QtTest import QSignalSpy
from shiboken6 import isValid

from tests.qt.test_739_navigator_protocols import _choose, _details, protocol_profile as protocol_profile
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('routes', [
    [('small_images', 'tiff_yokogawa'), ('sequencing', 'fastq_gz')],
    [('small_images', 'pdf'), ('plate_images', 'pdf')],
], ids=['simple', 'dependent'])
def test_protocol_choice_model_rebuild_and_close_keep_selection_models_attached(
        qtbot, qapp, protocol_profile, tmp_path, routes):
    from spacr.qt.widgets.home import _EasyHomePage
    warnings = []
    stage = ['construction']
    previous = None

    def message(kind, context, text):
        if 'QItemSelectionModel: Selecting when no model has been set' in text:
            warnings.append({'stage': stage[0], 'message': text,
                             'stack': traceback.format_list(traceback.extract_stack(limit=12)[:-1])})
        if previous is not None:
            previous(kind, context, text)
        else:
            sys.stderr.write(text + '\n')

    previous = qInstallMessageHandler(message)
    page = None
    try:
        page = _EasyHomePage()
        page.resize(1000, 800)
        page.show()
        qtbot.waitExposed(page)
        chosen = QSignalSpy(page.module_requested)
        stage[0] = 'enable Protocols'
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        for family, format in routes:
            stage[0] = family + ': choose family'
            _choose(qtbot, page._protocol_select, family)
            qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
            stage[0] = family + ': choose format'
            _choose(qtbot, page._protocol_select, format)
            stage[0] = family + ': rebuild dependent details'
            _details(qtbot, page)
            view = page._protocol_select.view()
            assert view.model() is page._protocol_select.model()
            assert view.selectionModel().model() is view.model()
            assert chosen.count() == 0 and page.path_edit.text() == ''
            if family == 'small_images':
                stage[0] = 'Back replaces format and retires detail models'
                qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                qapp.processEvents()
        stage[0] = 'close with inline choices open'
        page._protocol_select.showPopup()
        qtbot.waitUntil(page._protocol_select.view().isVisible)
        page.close()
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert not isValid(page)
    finally:
        if page is not None and isValid(page):
            page.close()
            page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
        qInstallMessageHandler(previous)
        (tmp_path / 'protocol-model-warnings.json').write_text(json.dumps(warnings, indent=2)+'\n')
    assert warnings == [], json.dumps(warnings, indent=2)


def test_one_native_cq1_protocol_finish_reports_its_actual_selection_owner(
        qtbot, qapp, protocol_profile, tmp_path):
    from spacr.qt.app import MainWindow
    from tests.qt.test_733_easy_mode_home import _image
    messages = []
    stage = ['MainWindow construction']
    previous = None

    def message(kind, context, text):
        if 'QItemSelectionModel: Selecting when no model has been set' in text:
            messages.append({'stage': stage[0], 'message': text,
                             'stack': traceback.format_list(traceback.extract_stack(limit=24)[:-1])})
        if previous is not None:
            previous(kind, context, text)
        else:
            sys.stderr.write(text + '\n')

    previous = qInstallMessageHandler(message)
    window = None
    try:
        image = tmp_path / 'source' / 'userW0025F0001T0001Z000C1.tif'
        image.parent.mkdir()
        _image(image)
        original = image.read_bytes()
        window = MainWindow(initial_app='__home__')
        window._tour_timer.stop()
        window._consent_timer.stop()
        window.resize(1366, 768)
        window.show()
        qtbot.waitExposed(window)
        page = window._startup
        chosen = QSignalSpy(page.module_requested)
        stage[0] = 'inspect real TIFF'
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.active_jobs(), timeout=5000)
        stage[0] = 'enable Protocols'
        qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
        stage[0] = 'choose small_images'
        _choose(qtbot, page._protocol_select, 'small_images')
        qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
        stage[0] = 'choose tiff_other'
        _choose(qtbot, page._protocol_select, 'tiff_other')
        stage[0] = 'choose cq1 and retire rebuilt metadata selector'
        _choose(qtbot, page._protocol_detail_combos['metadata_type'], 'cq1')
        stage[0] = 'final Next opens ProtocolBrowserDialog'
        qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
        dialog = window._protocol_browser_dialog
        assert dialog.isVisible() and chosen.count() == 0
        stage[0] = 'close ProtocolBrowserDialog'
        dialog.close()
        stage[0] = 'public Back rebuilds selectors'
        qtbot.mouseClick(page._protocol_back_button, Qt.LeftButton)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert image.read_bytes() == original and chosen.count() == 0
        stage[0] = 'close owning MainWindow'
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        assert not isValid(window)
    finally:
        if window is not None and isValid(window):
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            qapp.processEvents()
        qInstallMessageHandler(previous)
        (tmp_path / 'actual-selection-warning.json').write_text(json.dumps(messages, indent=2)+'\n')
    assert messages == [], json.dumps(messages, indent=2)
