"""Current runs opens the live registry panel only after a card click."""
import pytest
from PySide6.QtCore import Qt

from spacr.qt import app, bridge
from tests.qt.test_733_easy_commands import commands as commands
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('phrase', [
    'current runs', 'RUNNING RUNS!', 'running jobs', 'show running runs',
    'pågående körningar', 'laufende Aufträge', 'ejecuciones en curso',
    'execuções em curso', 'tâches en cours', '正在运行的任务',
    'चल रहे कार्य', '실행 중인 작업', 'verk í gangi',
])
def test_running_phrases_offer_a_click_only_card(commands, qtbot, monkeypatch, tmp_path, phrase):
    monkeypatch.chdir(tmp_path)
    assert app._easy_shortcut_match(phrase) == 'running'
    assert [row[0] for row in app._easy_module_search(phrase)] == ['__easy_command__:Running']
    rows = app.MainWindow._easy_navigation_suggestions(commands, phrase)
    assert [row[0] for row in rows] == ['__easy_command__:Running']
    assert not commands.opened and not commands.urls


def test_running_card_reveals_actual_live_registry_and_native_open(
        commands, qtbot, monkeypatch, tmp_path, private_ai_profile):
    monkeypatch.chdir(tmp_path)
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    page = app._NavigatorHomePage(commands)
    qtbot.addWidget(page)
    commands.setCentralWidget(page)
    commands._startup = page
    commands._easy_navigation_suggestions = lambda value: app.MainWindow._easy_navigation_suggestions(commands, value)
    page.command_requested.connect(commands._open_easy_command)
    opened = []
    page.tile_clicked.connect(opened.append)
    commands.resize(1100, 820)
    commands.show()
    aside = page._navigator_aside
    handles = []
    try:
        page._navigator_aside_split.set_collapsed('Widgets', True, by_user=False)
        aside._navigator_folders['Running'].set_shut(True, by_user=False)
        for key, visible in [('measure', True), ('mask', True), ('home journal', False)]:
            worker = bridge.PipelineWorker(lambda settings: None, {})
            worker.user_visible = visible
            handle = bridge.RunHandle(key, worker, None, parent=bridge.registry())
            handles.append(handle)
            bridge.registry().register(handle)
        assert len(aside._banners) == 2
        assert [banner._handle for banner in aside._banners] == handles[:2]
        assert not aside._running_empty.isVisible()
        page.path_edit.setText('current runs')
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        assert [card._key for card in page.field.cards] == ['__easy_command__:Running']
        assert page._navigator_aside_split.is_collapsed('Widgets')
        assert aside._navigator_folders['Running'].shut
        assert not commands.opened and not commands.urls
        card = page.field.cards[0]
        qtbot.waitUntil(lambda: not card._growing and card.isEnabled(), timeout=2500)
        card.click()
        assert not page._navigator_aside_split.is_collapsed('Widgets')
        assert not aside._navigator_folders['Running'].shut
        banner = aside._banners[0]
        qtbot.waitUntil(banner.isVisible)
        handles[0].last_line = 'Testing actual live progress'
        aside._refresh_run_banners()
        assert 'Testing actual live progress' in banner._sub.text()
        banner._btn_open.click()
        assert opened == ['measure']
        bridge.registry().unregister(handles[0])
        assert banner._handle is handles[1] and banner.isVisible()
        assert aside._banners[1]._handle is None and not aside._banners[1].isVisible()
        bridge.registry().unregister(handles[1])
        assert banner._handle is None and not banner.isVisible()
        assert aside._running_empty.isVisible()
        assert not aside._ticker.isActive()
    finally:
        for handle in handles:
            bridge.registry().unregister(handle)
        page.close()


def test_two_running_banners_keep_all_controls_inside_a_300_pixel_pane(
        qtbot, private_ai_profile):
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(300, 800)
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    aside = app._NavigatorAside(host)
    layout.addWidget(aside)
    handles = []
    try:
        for key in ('measure', 'mask'):
            worker = bridge.PipelineWorker(lambda settings: None, {})
            handle = bridge.RunHandle(key, worker, None, parent=bridge.registry())
            handle.last_line = 'A long current operation description must wrap inside the narrow panel.'
            handles.append(handle)
            bridge.registry().register(handle)
        aside._navigator_folders['Running'].set_shut(False, by_user=False)
        host.show()
        qtbot.waitUntil(lambda: all(banner.isVisible() for banner in aside._banners))
        assert host.width() == 300
        for banner in aside._banners:
            assert banner.width() <= 300
            for control in (banner._icon, banner._title, banner._sub, banner._bar,
                            banner._btn_open, banner._btn_pause, banner._btn_quit):
                rect = control.rect().translated(control.mapTo(banner, control.rect().topLeft()))
                assert control.isVisible()
                assert banner.rect().contains(rect), (control, rect, banner.rect())
        assert aside._aside_scroll.horizontalScrollBar().maximum() == 0
        assert not host.grab().isNull()
    finally:
        for handle in handles:
            bridge.registry().unregister(handle)
        aside.close()
