"""Explicit Navigator assistance uses native controls and mocked existing transport."""
import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QComboBox, QDialog, QTextBrowser

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile
from tests.qt.test_733_easy_commands import commands as commands


class _Worker(QObject):
    chunk_ready = Signal(str)
    finished = Signal(bool, str)

    def __init__(self):
        super().__init__()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _Thread(QObject):
    def __init__(self):
        super().__init__()
        self.started = False

    def start(self):
        self.started = True

    def isRunning(self):
        return self.started


@pytest.fixture
def ai_page(qtbot, private_ai_profile, monkeypatch):
    from spacr.qt import bridge
    from spacr.qt.ai import providers, worker
    from spacr.qt.widgets.home import _EasyHomePage
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(False)
    calls, drained = [], []

    class Provider:
        def is_configured(self):
            return True

    monkeypatch.setattr(providers, 'get_provider', lambda name: Provider())
    def stream(provider, messages, **kwargs):
        thread, work = _Thread(), _Worker()
        calls.append((messages, thread, work))
        return thread, work
    monkeypatch.setattr(worker, 'make_stream_thread', stream)
    monkeypatch.setattr(bridge, 'drain_thread', lambda thread, work, **kw: drained.append((thread, work, kw)))
    page = _EasyHomePage()
    page.resize(1000, 800)
    qtbot.addWidget(page)
    page.show()
    yield page, calls, drained
    page.close()


def _enable(page):
    page._ai_provider = 'spacr_hosted'
    page.path_edit.ai.setChecked(True)
    page._toggle_navigator_ai(True)


@pytest.mark.parametrize('accept', [False, True])
def test_first_use_explains_four_providers_and_requires_acceptance(ai_page, qtbot, monkeypatch, accept):
    from spacr.qt.ai import providers
    from spacr.qt.preferences import _settings
    page, calls, drained = ai_page
    applied = []
    monkeypatch.setattr(providers, 'apply_provider_choice', applied.append)
    def choose(dialog):
        assert dialog is not None
        combo = dialog.findChild(QComboBox, 'NavigatorAiProviderChoice')
        assert {combo.itemData(i) for i in range(combo.count())} == {
            'spacr_hosted', 'claude', 'codex', 'gemini'}
        explanation = dialog.findChild(QTextBrowser).toPlainText()
        assert all(word in explanation for word in ('daily allowance', 'CLI login', 'shared', 'does not run'))
        combo.setCurrentIndex(combo.findData('codex'))
        dialog.accept() if accept else dialog.reject()
        return dialog.result()
    monkeypatch.setattr(QDialog, 'exec', choose)
    qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
    assert page.path_edit.ai.isChecked() is accept
    assert applied == (['codex'] if accept else [])
    assert _settings().value('home/navigator_ai_provider', '') == ('codex' if accept else '')
    assert not calls


def test_only_unmatched_enter_streams_to_lower_box(ai_page, qtbot):
    page, calls, drained = ai_page
    _enable(page)
    page.path_edit.setFocus()
    qtbot.keyClicks(page.path_edit, 'Explain why fluorescence becomes patchy overnight')
    qtbot.wait(60)
    assert not calls
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert len(calls) == 1 and calls[0][1].started
    assert page.path_edit.ai._busy
    work = calls[0][2]
    work.chunk_ready.emit('Helpful <b>literal</b> advice')
    qtbot.waitUntil(lambda: 'literal' in page.field.details.toPlainText())
    assert page.field.details.toPlainText() == 'Helpful <b>literal</b> advice'
    work.finished.emit(True, 'Final answer')
    qtbot.waitUntil(lambda: page._ai_thread is None)
    assert page.field.details.toPlainText() == 'Final answer'
    assert not page.path_edit.ai._busy and not page.field.cards
    page._cancel_navigator_ai()
    assert len(drained) == 1 and drained[0][2]['timeout_ms'] == 0


@pytest.mark.parametrize('value', ['Measure', 'Make Masks', 'Overview', '/tmp/unknown-images', 'field.npy'])
def test_tools_and_paths_never_enter_ai_transport(ai_page, qtbot, monkeypatch, value, commands):
    from spacr.qt.widgets import home
    page, calls, drained = ai_page
    commands._startup = page
    commands.setCentralWidget(page)
    commands.show()
    page.command_requested.connect(commands._open_easy_command)
    _enable(page)
    inspected = []
    monkeypatch.setattr(home, '_easy_inspect_path', lambda path: (
        inspected.append(path) or dict(path=path, actions=[], context={}, message='')))
    page.path_edit.setText(value)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards) or bool(inspected))
    assert not calls


@pytest.mark.parametrize('action', ['edit', 'clear', 'disable', 'hide', 'reset'])
def test_cancel_invalidates_queued_replies_without_blocking(ai_page, qtbot, action):
    page, calls, drained = ai_page
    _enable(page)
    page.path_edit.setText('Explain an unfamiliar experimental workflow')
    page._submit()
    assert len(calls) == 1
    old_generation = page._ai_generation
    work = calls[0][2]
    if action == 'edit':
        qtbot.keyClicks(page.path_edit, ' changed')
    elif action == 'clear':
        page.path_edit.selectAll()
        qtbot.keyClick(page.path_edit, Qt.Key_Backspace)
    elif action == 'disable':
        qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
    elif action == 'hide':
        page.hide()
    else:
        page._reset_for_home()
    assert work.cancelled and page._ai_generation > old_generation
    assert drained and drained[0][2]['timeout_ms'] == 0
    work.chunk_ready.emit('STALE ANSWER')
    work.finished.emit(True, 'STALE ANSWER')
    qtbot.wait(70)
    assert 'STALE' not in page.field.details.toPlainText()
    assert page._ai_thread is None and not page.path_edit.ai._busy


def test_icon_is_hollow_off_matte_on_and_busy_glow_rotates_while_typing_pulses(ai_page, qtbot):
    page, calls, drained = ai_page
    button = page.path_edit.ai
    def image():
        return button.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    off = image()
    assert any(off.pixelColor(x, y).red() > 235
               and off.pixelColor(x, y).green() > 235
               and off.pixelColor(x, y).blue() > 235
               for x in range(36) for y in range(36))
    _enable(page)
    pictures = []
    for provider in button._COLORS:
        button._provider = provider
        frame = image()
        center = button.rect().center()
        assert all(frame.pixelColor(center.x() + dx, center.y() + dy)
                   == QColor(button._COLORS[provider])
                   for dx in range(-3, 4) for dy in range(-3, 4))
        if provider == 'spacr_hosted':
            assert frame.pixelColor(center.x() + 4, center.y()) == QColor('white')
            assert frame.pixelColor(center.x() + 9, center.y()) != QColor('white')
            assert 14 <= sum(frame.pixelColor(center.x() + dx, center.y()) == QColor('white')
                             for dx in range(-10, 11)) <= 16
        pictures.append(bytes(frame.constBits()))
    assert len(set(pictures)) == 4 and bytes(off.constBits()) not in pictures
    button._last_key = button._clock.elapsed() - 1000
    button._typing('a')
    slow = button._pulse
    button._typing('ab')
    assert button._pulse > slow
    assert button._motion.isActive()
    button._last_key = button._clock.elapsed() - 100000
    button._busy = True
    button._tick()
    before = image()
    qtbot.wait(80)
    after = image()
    assert bytes(after.constBits()) != bytes(before.constBits())
    # A pixel just outside the 14 px ring, where the glow must never reach,
    # and the matte centre, measured from the button's own centre: the
    # control is a 36 px square, so fixed coordinates sat on the ring's edge.
    centre = button.rect().center()
    outside = (centre.x(), centre.y() - 16)
    assert after.pixelColor(*outside) == before.pixelColor(*outside)
    assert all(after.pixelColor(centre.x() + dx, centre.y() + dy)
               == before.pixelColor(centre.x() + dx, centre.y() + dy)
               for dx in range(-4, 5) for dy in range(-4, 5))
    assert button._motion.isActive()
    button.setChecked(False)
    assert not button._motion.isActive() and not calls


def test_saved_provider_reuses_consent_and_change_retires_old_stream(ai_page, qtbot, monkeypatch):
    from spacr.qt.ai import providers
    from spacr.qt.preferences import _settings
    page, calls, drained = ai_page
    _settings().setValue('home/navigator_ai_provider', 'claude')
    qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
    assert page._ai_provider == 'claude' and page.path_edit.ai.isChecked()
    assert page.findChild(QDialog, 'NavigatorAiProviderDialog') is None
    page.path_edit.setText('Explain a completely unfamiliar experiment')
    page._submit()
    old_worker = calls[0][2]
    applied = []
    monkeypatch.setattr(providers, 'apply_provider_choice', applied.append)
    def choose(dialog):
        combo = dialog.findChild(QComboBox, 'NavigatorAiProviderChoice')
        combo.setCurrentIndex(combo.findData('gemini'))
        return QDialog.Accepted
    monkeypatch.setattr(QDialog, 'exec', choose)
    page._choose_navigator_ai()
    assert applied == ['gemini'] and page._ai_provider == 'gemini'
    assert old_worker.cancelled and drained
    old_worker.finished.emit(True, 'STALE PROVIDER ANSWER')
    qtbot.wait(30)
    assert 'STALE PROVIDER' not in page.field.details.toPlainText()
    assert len(calls) == 1


def test_real_stream_thread_delivers_on_gui_and_retires_cleanly(
        qtbot, private_ai_profile, monkeypatch):
    import threading

    from PySide6.QtCore import QThread
    from shiboken6 import isValid

    from spacr.qt.ai import providers, worker
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    stream_threads, callback_threads, created, prompts = [], [], [], []
    gui_thread = QThread.currentThread()

    class Provider:
        def is_configured(self):
            return True

        def stream_chat(self, messages, system='', model=None):
            stream_threads.append(QThread.currentThread())
            prompts.append((messages, system))
            entered.set()
            if not release.wait(3):
                raise RuntimeError('Test did not release offline provider')
            yield 'Offline '
            yield 'thread answer'
            completed.set()

        def cancel_stream(self):
            release.set()

    monkeypatch.setattr(providers, 'get_provider', lambda name: Provider())
    monkeypatch.setattr(worker, 'question_cache_key', lambda messages: '')
    real_make = worker.make_stream_thread
    def capture(*args, **kwargs):
        pair = real_make(*args, **kwargs)
        created.append(pair)
        return pair
    monkeypatch.setattr(worker, 'make_stream_thread', capture)
    class RecordingPage(_EasyHomePage):
        def _navigator_ai_chunk(self, *args):
            callback_threads.append(QThread.currentThread())
            super()._navigator_ai_chunk(*args)

        def _navigator_ai_finished(self, *args):
            callback_threads.append(QThread.currentThread())
            super()._navigator_ai_finished(*args)

    page = RecordingPage()
    qtbot.addWidget(page)
    page.resize(1000, 800)
    page.show()
    _enable(page)
    try:
        page.path_edit.setText('Explain a novel imaging experiment in plain words')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(entered.is_set)
        assert page._ai_thread is created[0][0] and page.path_edit.ai._busy
        assert stream_threads and stream_threads[0] != gui_thread
        assert not callback_threads
        release.set()
        qtbot.waitUntil(completed.is_set)
        qtbot.waitUntil(lambda: page._ai_thread is None)
        qtbot.waitUntil(lambda: page.field.details.toPlainText() == 'Offline thread answer')
        assert len(callback_threads) == 3
        assert all(thread == gui_thread for thread in callback_threads)
        assert not page.path_edit.ai._busy
        assert len(prompts) == 1
        assert 'Do not invoke tools, run commands or analyses' in prompts[0][1]
        thread = created[0][0]
        qtbot.waitUntil(lambda: not isValid(thread) or not thread.isRunning())
        completed_answer = page.field.details.toPlainText()
        qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
        assert not page.path_edit.ai.isChecked() and not page.path_edit.ai._busy
        assert page.field.details.toPlainText() == completed_answer
        assert len(prompts) == 1 and not page._ai_retired and page._ai_worker is None
        details = page.field.details
        qtbot.mouseClick(details._back, Qt.LeftButton)
        assert details._forward.isEnabled()
        page.path_edit.setText('Measure')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: any(card._key == 'measure' for card in page.field.cards))
        assert not details._forward.isEnabled()
        assert details._stream_entry == -1 and len(prompts) == 1
        assert completed_answer not in details.toPlainText()
        page._cancel_navigator_ai()
        assert not page._ai_retired and page._ai_worker is None
        page.close()
    finally:
        release.set()
        for thread, work in created:
            if isValid(thread) and thread.isRunning():
                work.cancel()
                thread.quit()
                assert thread.wait(3000)
        page.close()


def test_preferences_none_stops_an_already_enabled_navigator(
        qtbot, private_ai_profile, monkeypatch):
    import io
    import json
    import os
    import urllib.request

    import requests
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QRadioButton
    from shiboken6 import isValid

    from spacr import crashreport
    from spacr.qt.ai import providers
    from spacr.qt.app import MainWindow
    from spacr.qt.preferences_navigation import show_tab

    preferences, terms, settings = private_ai_profile
    terms.record_agreement()
    accepted = terms.agreement_record()
    preferences.set_easy_mode(True)
    preferences.set_refresh_news(False)
    preferences.set_issue_prompt_mode(preferences.ISSUE_PROMPT_ALWAYS)
    settings.set_auto_file_issues(True)
    crashreport.set_error_report_mode('all')
    requests_seen = []

    def relay(request, timeout):
        requests_seen.append((request.full_url, json.loads(request.data)))
        response = io.BytesIO(json.dumps({
            'advice': 'Controlled Navigator reply', 'model': 'offline-test',
            'tier': 'best',
        }).encode())
        response.status = 200
        return response

    def no_external_transport(*args, **kwargs):
        raise AssertionError('Unexpected external HTTP transport')

    monkeypatch.setattr(crashreport, '_open_url', relay)
    monkeypatch.setattr(urllib.request, 'urlopen', no_external_transport)
    monkeypatch.setattr(requests.sessions.Session, 'request', no_external_transport)
    native_exec = next(original for owner, name, original in monkeypatch._setattr
                       if owner is QDialog and name == 'exec')
    network_before = os.environ.get('SPACR_NO_NETWORK')
    window = None
    dialogs = []
    try:
        monkeypatch.setenv('SPACR_NO_NETWORK', '0')
        assert crashreport.network_allowed()
        window = MainWindow()
        qtbot.addWidget(window)
        window.show()
        page = window._startup
        qtbot.waitUntil(lambda: page.path_edit.isVisible())

        def choose_hosted():
            dialog = QApplication.activeModalWidget()
            try:
                assert dialog is not None
                assert dialog.objectName() == 'NavigatorAiProviderDialog'
                combo = dialog.findChild(QComboBox, 'NavigatorAiProviderChoice')
                combo.setCurrentIndex(combo.findData(providers.HOSTED_PROVIDER_NAME))
                qtbot.mouseClick(dialog.findChild(QDialogButtonBox).button(
                    QDialogButtonBox.Ok), Qt.LeftButton)
            finally:
                if dialog is not None and isValid(dialog) and dialog.result() != QDialog.Accepted:
                    dialog.reject()

        with monkeypatch.context() as modal:
            modal.setattr(QDialog, 'exec', native_exec)
            QTimer.singleShot(0, choose_hosted)
            page.path_edit.ai.click()
        assert page.path_edit.ai.isChecked()
        assert preferences.get_preferred_provider() == providers.HOSTED_PROVIDER_NAME
        assert not requests_seen
        first = 'Explain why fluorescence becomes patchy across an unfamiliar experiment'
        page.path_edit.setText(first)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: 'Controlled Navigator reply' in page.field.details.toPlainText())
        qtbot.waitUntil(lambda: page._ai_thread is None)
        assert len(requests_seen) == 1
        assert requests_seen[0][0].endswith('/diagnose')
        assert requests_seen[0][1]['question'] == first
        assert terms.agreement_record() == accepted
        assert crashreport.error_report_mode() == 'all'

        dialog = preferences.PreferencesDialog(window)
        dialogs.append(dialog)
        qtbot.addWidget(dialog)
        assert show_tab(dialog, 'PreferencesTabAI')
        dialog.show()
        none = dialog.findChild(QRadioButton, 'AiProvider_none')
        none.click()
        assert none.isChecked()
        assert preferences.get_preferred_provider() == 'none'
        reports = dialog.findChild(QComboBox, 'AiErrorReport')
        assert reports.currentData() == 'all'
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
        assert preferences.get_preferred_provider() == 'none'
        assert providers.default_provider('none') is None
        assert crashreport.error_report_mode() == 'all'
        assert terms.agreement_record() == accepted
        assert len(requests_seen) == 1
        reopened = preferences.PreferencesDialog(window)
        dialogs.append(reopened)
        qtbot.addWidget(reopened)
        assert reopened.findChild(QRadioButton, 'AiProvider_none').isChecked()
        assert reopened.findChild(QComboBox, 'AiErrorReport').currentData() == 'all'
        reopened.close()
        page = window._startup
        second = 'Explain an unrelated unfamiliar experimental imaging question'
        page.path_edit.setText(second)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: page._ai_thread is None)
        assert len(requests_seen) == 1
        assert preferences.get_preferred_provider() == 'none'
        assert crashreport.error_report_mode() == 'all'
        assert terms.agreement_record() == accepted
    finally:
        if window is not None and isValid(window):
            page = window._startup
            threads = ([page._ai_thread] if page._ai_thread is not None else [])
            threads += [thread for thread, _, _ in page._ai_retired]
            page._cancel_navigator_ai()
            for thread in threads:
                if isValid(thread) and thread.isRunning():
                    thread.quit()
                    assert thread.wait(3000)
            for dialog in dialogs:
                if isValid(dialog):
                    dialog.close()
            window.close()
        if network_before is None:
            monkeypatch.delenv('SPACR_NO_NETWORK', raising=False)
        else:
            monkeypatch.setenv('SPACR_NO_NETWORK', network_before)


def test_preferences_none_discards_a_reply_already_in_flight(
        qtbot, private_ai_profile, monkeypatch):
    import io
    import json
    import os
    import threading
    import urllib.request

    import requests
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialogButtonBox, QRadioButton
    from shiboken6 import isValid

    from spacr import crashreport
    from spacr.qt.ai import providers
    from spacr.qt.app import MainWindow
    from spacr.qt.preferences_navigation import show_tab

    preferences, terms, settings = private_ai_profile
    terms.record_agreement()
    accepted = terms.agreement_record()
    preferences.set_easy_mode(True)
    preferences.set_refresh_news(False)
    preferences.set_issue_prompt_mode(preferences.ISSUE_PROMPT_ALWAYS)
    settings.set_auto_file_issues(True)
    crashreport.set_error_report_mode('all')
    requests_seen = []
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    thread = None

    def relay(request, timeout):
        requests_seen.append((request.full_url, json.loads(request.data)))
        entered.set()
        if not release.wait(15):
            raise RuntimeError('Held test transport was not released')
        response = io.BytesIO(json.dumps({
            'advice': 'HELD NAVIGATOR RESPONSE MUST NOT APPEAR', 'model': 'offline-test',
            'tier': 'best',
        }).encode())
        response.status = 200
        returned.set()
        return response

    def no_external_transport(*args, **kwargs):
        raise AssertionError('Unexpected external HTTP transport')

    monkeypatch.setattr(crashreport, '_open_url', relay)
    monkeypatch.setattr(urllib.request, 'urlopen', no_external_transport)
    monkeypatch.setattr(requests.sessions.Session, 'request', no_external_transport)
    native_exec = next(original for owner, name, original in monkeypatch._setattr
                       if owner is QDialog and name == 'exec')
    network_before = os.environ.get('SPACR_NO_NETWORK')
    window = None
    dialogs = []
    try:
        monkeypatch.setenv('SPACR_NO_NETWORK', '0')
        assert crashreport.network_allowed()
        window = MainWindow()
        qtbot.addWidget(window)
        window.show()
        page = window._startup
        qtbot.waitUntil(lambda: page.path_edit.isVisible())

        def choose_hosted():
            dialog = QApplication.activeModalWidget()
            try:
                assert dialog is not None
                assert dialog.objectName() == 'NavigatorAiProviderDialog'
                combo = dialog.findChild(QComboBox, 'NavigatorAiProviderChoice')
                combo.setCurrentIndex(combo.findData(providers.HOSTED_PROVIDER_NAME))
                qtbot.mouseClick(dialog.findChild(QDialogButtonBox).button(
                    QDialogButtonBox.Ok), Qt.LeftButton)
            finally:
                if dialog is not None and isValid(dialog) and dialog.result() != QDialog.Accepted:
                    dialog.reject()

        with monkeypatch.context() as modal:
            modal.setattr(QDialog, 'exec', native_exec)
            QTimer.singleShot(0, choose_hosted)
            page.path_edit.ai.click()
        assert page.path_edit.ai.isChecked()
        assert preferences.get_preferred_provider() == providers.HOSTED_PROVIDER_NAME
        assert not requests_seen
        question = 'Explain a different unfamiliar experimental imaging procedure'
        page.path_edit.setText(question)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(entered.is_set)
        thread = page._ai_thread
        assert thread is not None and thread.isRunning()
        assert page.path_edit.ai._busy
        assert len(requests_seen) == 1
        assert requests_seen[0][0].endswith('/diagnose')
        assert requests_seen[0][1]['question'] == question
        assert 'HELD NAVIGATOR RESPONSE' not in page.field.details.toPlainText()

        dialog = preferences.PreferencesDialog(window)
        dialogs.append(dialog)
        qtbot.addWidget(dialog)
        assert show_tab(dialog, 'PreferencesTabAI')
        dialog.show()
        none = dialog.findChild(QRadioButton, 'AiProvider_none')
        none.click()
        assert none.isChecked()
        assert preferences.get_preferred_provider() == 'none'
        reports = dialog.findChild(QComboBox, 'AiErrorReport')
        assert reports.currentData() == 'all'
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
        assert preferences.get_preferred_provider() == 'none'
        assert providers.default_provider('none') is None
        assert crashreport.error_report_mode() == 'all'
        assert terms.agreement_record() == accepted
        assert len(requests_seen) == 1
        reopened = preferences.PreferencesDialog(window)
        dialogs.append(reopened)
        qtbot.addWidget(reopened)
        assert reopened.findChild(QRadioButton, 'AiProvider_none').isChecked()
        assert reopened.findChild(QComboBox, 'AiErrorReport').currentData() == 'all'
        reopened.close()
        assert window._startup is page
        release.set()
        qtbot.waitUntil(returned.is_set)
        qtbot.waitUntil(lambda: page._ai_thread is None)
        qtbot.waitUntil(lambda: not isValid(thread) or not thread.isRunning())
        delivered = threading.Event()
        QTimer.singleShot(0, delivered.set)
        qtbot.waitUntil(delivered.is_set)
        assert not page._ai_reply_timer.isActive()
        assert 'HELD NAVIGATOR RESPONSE' not in page.field.details.toPlainText()
        assert not page.path_edit.ai.isChecked()
        assert len(requests_seen) == 1
        assert preferences.get_preferred_provider() == 'none'
        assert crashreport.error_report_mode() == 'all'
        assert terms.agreement_record() == accepted
    finally:
        release.set()
        if window is not None and isValid(window):
            page = window._startup
            threads = ([thread] if thread is not None else [])
            threads += [page._ai_thread] if page._ai_thread is not None else []
            threads += [thread for thread, _, _ in page._ai_retired]
            page._cancel_navigator_ai()
            for thread in threads:
                if isValid(thread) and thread.isRunning():
                    thread.quit()
                    assert thread.wait(3000)
            for dialog in dialogs:
                if isValid(dialog):
                    dialog.close()
            window.close()
        if network_before is None:
            monkeypatch.delenv('SPACR_NO_NETWORK', raising=False)
        else:
            monkeypatch.setenv('SPACR_NO_NETWORK', network_before)


@pytest.mark.parametrize('delivery', ['render', 'finished'])
def test_pending_advice_and_repeat_enter_respect_a_revoked_provider(ai_page, qtbot, private_ai_profile, delivery):
    page, calls, drained = ai_page
    preferences, _, _ = private_ai_profile
    _enable(page)
    page.path_edit.setText('Explain an unfamiliar experimental workflow')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert len(calls) == 1
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert len(calls) == 1 and 'Turn AI off to cancel' in page.field.details.toPlainText()
    work = calls[0][2]
    preferences.set_preferred_provider('none')
    if delivery == 'render':
        page._ai_text = 'Queued private advice'
        page._ai_reply_timer.start(0)
    else:
        work.finished.emit(True, 'Queued private advice')
    qtbot.waitUntil(lambda: page._ai_thread is None, timeout=5000)
    assert work.cancelled and drained and drained[0][2]['timeout_ms'] == 0
    assert not page.path_edit.ai.isChecked() and not page.path_edit.ai._busy
    assert 'Queued private advice' not in page.field.details.toPlainText()


def test_removed_provider_and_missing_cli_keep_submitted_help_local(ai_page, qtbot, private_ai_profile, monkeypatch):
    from spacr.qt.ai import providers

    page, calls, _ = ai_page
    preferences, _, _ = private_ai_profile
    _enable(page)
    preferences.set_preferred_provider('none')
    page.path_edit.setText('Explain why this unfamiliar image looks different')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert not calls and not page.path_edit.ai.isChecked()
    assert 'No provider configured' in page.field.details.toPlainText()
    preferences.set_preferred_provider('spacr_hosted')
    _enable(page)
    monkeypatch.setattr(providers, 'get_provider', lambda name: None)
    page.path_edit.setText('Explain a second unfamiliar image')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert not calls and page._ai_thread is None
    assert 'No provider configured' in page.field.details.toPlainText()


@pytest.mark.parametrize('still_running', [False, True])
def test_next_answer_retains_only_a_still_running_previous_worker(ai_page, qtbot, still_running):
    page, calls, _ = ai_page
    _enable(page)
    page.path_edit.setText('Explain an unfamiliar first image')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    first_thread, first_worker = calls[0][1:]
    first_worker.finished.emit(True, 'First complete answer')
    qtbot.waitUntil(lambda: page._ai_thread is None, timeout=5000)
    first_thread.started = still_running
    page.path_edit.setText('Explain an unfamiliar second image')
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    assert len(calls) == 2
    assert len(page._ai_retired) == int(still_running)
    assert page.path_edit.ai.isChecked()
    calls[1][2].finished.emit(True, 'Second complete answer')
    qtbot.waitUntil(lambda: page._ai_thread is None, timeout=5000)
    assert page.field.details.toPlainText() == 'Second complete answer'


@pytest.mark.parametrize('value, expected', [
    ('make mask', {'make_masks'}),
    ('SCN', {'make_masks', 'convert'}),
])
def test_native_tool_alias_and_scientific_format_remain_click_only_with_ai_on(
        ai_page, commands, qtbot, value, expected):
    page, calls, _ = ai_page
    commands._startup = page
    commands.setCentralWidget(page)
    commands.show()
    page.command_requested.connect(commands._open_easy_command)
    _enable(page)
    page.path_edit.setText(value)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)
    qtbot.waitUntil(lambda: bool(page.field.cards), timeout=5000)
    assert {card._key for card in page.field.cards} == expected
    assert not calls and not commands.opened
    if value == 'SCN':
        assert 'mask' not in {card._key for card in page.field.cards}
