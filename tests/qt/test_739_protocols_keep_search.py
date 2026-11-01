"""An unfinished protocol leaves Navigator search, answers and questions working as usual."""
from PySide6.QtCore import Qt

from spacr.qt import app
from tests.qt.test_739_navigator_protocols import _choose, _details
from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def _window(qtbot):
    """Open the real Navigator page inside the main window."""
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1366, 768)
    window.show()
    qtbot.waitExposed(window)
    return window, window._startup


def _search(qtbot, page, text):
    """Type a query like a user and submit it with Enter."""
    page.path_edit.setFocus()
    qtbot.keyClick(page.path_edit, Qt.Key_A, Qt.ControlModifier)
    qtbot.keyClicks(page.path_edit, text)
    qtbot.keyClick(page.path_edit, Qt.Key_Return)


def _keys(page):
    """Return the visible card keys in their shown order."""
    return [card._key for card in page.field.cards]


def _midway(qtbot, page):
    """Choose a family and format, stopping one step before the protocol ends."""
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'small_images')
    qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, 'tiff_other')
    _details(qtbot, page)
    assert page._protocol_step == 1 and page._protocol_next_button.isEnabled()
    return page._protocol_step, page._protocol_family, page._protocol_format, dict(page._protocol_choices)


def _state(page):
    """Read the protocol choices kept behind a search."""
    return page._protocol_step, page._protocol_family, page._protocol_format, dict(page._protocol_choices)


def test_typing_mask_mid_protocol_shows_the_same_cards_and_keeps_the_choices(qtbot, protocol_profile):
    """Mask and Make Masks appear exactly as with Protocols off; choices stay put."""
    window, page = _window(qtbot)
    try:
        _search(qtbot, page, 'mask')
        qtbot.waitUntil(lambda: 'mask' in _keys(page), timeout=5000)
        plain = _keys(page)
        assert 'make_masks' in plain
        page.path_edit.clear()
        state = _midway(qtbot, page)
        _search(qtbot, page, 'mask')
        qtbot.waitUntil(lambda: _keys(page) == plain, timeout=5000)
        assert page.path_edit.protocols.isChecked()
        assert page.field.protocol_controls.isVisible()
        assert _state(page) == state
        assert page._protocol_select.currentData() == 'tiff_other'
        for name, value in state[3].items():
            assert page._protocol_detail_combos[name].currentData() == value
        assert getattr(window, '_protocol_browser_dialog', None) is None
        qtbot.keyClicks(page.path_edit, 's')
        assert _keys(page) == plain and _state(page) == state
    finally:
        window.close()


def test_a_question_mid_protocol_reaches_local_answers_and_the_ai(qtbot, protocol_profile, monkeypatch):
    """Local replies and opted-in AI questions behave as with Protocols off."""
    from spacr.qt.ai import providers, worker

    asked = []

    class OfflineProvider:
        def is_configured(self):
            return True

        def stream_chat(self, messages, system='', model=None):
            asked.append(messages[-1]['content'])
            yield 'Offline answer about plates'

        def cancel_stream(self):
            pass

    monkeypatch.setattr(providers, 'get_provider', lambda name: OfflineProvider())
    monkeypatch.setattr(worker, 'question_cache_key', lambda messages: '')
    protocol_profile._settings().setValue('home/navigator_ai_provider', 'spacr_hosted')
    protocol_profile.set_preferred_provider('spacr_hosted')
    window, page = _window(qtbot)
    try:
        state = _midway(qtbot, page)
        _search(qtbot, page, 'What is the meaning of life')
        qtbot.waitUntil(lambda: 'Hitchhiker' in page.field.details.toPlainText(), timeout=5000)
        assert _state(page) == state
        qtbot.mouseClick(page.path_edit.ai, Qt.LeftButton)
        assert page.path_edit.ai.isChecked() and page.path_edit.protocols.isChecked()
        _search(qtbot, page, 'Explain an unfamiliar imaging experiment')
        qtbot.waitUntil(lambda: 'Offline answer about plates' in page.field.details.toPlainText(),
                        timeout=5000)
        assert asked == ['Explain an unfamiliar imaging experiment']
        assert _state(page) == state and page.field.protocol_controls.isVisible()
        qtbot.waitUntil(lambda: page._ai_thread is None, timeout=5000)
    finally:
        window.close()


def test_only_the_finished_protocol_shows_its_modules_and_new_typing_searches_again(
        qtbot, protocol_profile):
    """The last step replaces search cards; later queries and an empty field behave as stated."""
    from spacr.qt.widgets.sample_project import _protocol_result

    window, page = _window(qtbot)
    try:
        _midway(qtbot, page)
        _search(qtbot, page, 'mask')
        qtbot.waitUntil(lambda: 'make_masks' in _keys(page), timeout=5000)
        searched = _keys(page)
        qtbot.mouseClick(page._protocol_next_button, Qt.LeftButton)
        dialog = window._protocol_browser_dialog
        qtbot.waitUntil(dialog.isVisible)
        dialog.close()
        offered = [action[0] for action in _protocol_result(
            page._protocol_family, page._protocol_format, page._protocol_choices, '', {})['actions']]
        assert page._protocol_step == 2 and _keys(page) == offered and offered != searched
        assert page._protocol_result is not None
        _search(qtbot, page, 'mask')
        qtbot.waitUntil(lambda: _keys(page) == searched, timeout=5000)
        assert page._protocol_result is None and page._protocol_step == 2
        page.path_edit.clear()
        qtbot.waitUntil(lambda: _keys(page) == offered, timeout=5000)
        assert page._protocol_result is not None
    finally:
        window.close()
