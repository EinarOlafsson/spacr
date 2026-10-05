"""The catalogue stays responsive and safely finishes after Preferences closes."""
import threading

import pytest
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import QDialog, QFormLayout
from shiboken6 import isValid

from spacr import plugins
from spacr.qt import preferences
from tests.test_plugin_catalogue import _catalogue


@pytest.fixture
def page(qtbot, monkeypatch, tmp_path):
    source = tmp_path / 'catalogue'
    source.mkdir()
    _catalogue(source)
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(
        str(tmp_path / 'prefs.ini'), QSettings.IniFormat))
    monkeypatch.setenv('SPACR_PLUGIN_HOME', str(tmp_path / 'plugins'))
    monkeypatch.delenv('SPACR_PLUGIN_MODULES', raising=False)
    monkeypatch.delenv('SPACR_DISABLE_PLUGINS', raising=False)
    preferences._set_plugin_catalogue(str(source))
    preferences._set_show_alpha_features(True)
    dialog = QDialog()
    qtbot.addWidget(dialog)
    result = preferences._PluginCataloguePage(QFormLayout(dialog), dialog)
    qtbot.waitUntil(lambda: result._job is None)
    result._select_key('toxo_infection')
    yield result
    if result._job is not None:
        result._job.join(5)
        assert not result._job.is_alive()


def test_install_runs_off_thread_with_live_event_loop_and_busy_guards(page, qtbot, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    actual = plugins._install_from_catalogue
    main_thread = threading.get_ident()
    calls = []

    def blocked(*args, **kwargs):
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return actual(*args, **kwargs)

    monkeypatch.setattr(plugins, '_install_from_catalogue', blocked)
    heartbeats = []
    timer = QTimer(page._dialog)
    timer.setInterval(1)
    timer.timeout.connect(lambda: heartbeats.append(True))
    timer.start()
    assert page.install_selected()
    try:
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: len(heartbeats) > 2)
        assert calls == [calls[0]] and calls[0] != main_thread
        assert not page.install_button.isEnabled()
        assert not page.uninstall_button.isEnabled()
        assert not page.source.isEnabled()
        assert not page.refresh()
        assert not page.install_selected()
        assert not page._open_selected()
        assert page.status.text() == 'Working…'
    finally:
        release.set()
    qtbot.waitUntil(lambda: page._job is None)
    assert 'Installed Toxoplasma infection assay 0.2' in page.status.text()
    assert page.selected()['installed'] == '0.2'
    assert page.open_button.isEnabled()
    assert page.source.isEnabled()
    assert page.uninstall_selected()
    qtbot.waitUntil(lambda: page._job is None)
    assert 'Uninstalled Toxoplasma infection assay' in page.status.text()
    assert page.selected()['status'] == 'available'


def test_window_destruction_does_not_destroy_or_cancel_running_install(page, qtbot, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    actual = plugins._install_from_catalogue

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return actual(*args, **kwargs)

    monkeypatch.setattr(plugins, '_install_from_catalogue', blocked)
    page._dialog.setAttribute(Qt.WA_DeleteOnClose)
    page._dialog.show()
    assert page.install_selected()
    job = page._job
    try:
        qtbot.waitUntil(started.is_set)
        page._dialog.close()
        qtbot.waitUntil(lambda: not isValid(page._dialog))
        assert job.is_alive()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not job.is_alive())
    assert job.error is None
    assert plugins._catalogue_installed()['toxo_infection']['version'] == '0.2'


def test_failed_install_restores_actions_and_reports_error(page, qtbot, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('offline wheel unavailable')

    monkeypatch.setattr(plugins, '_install_from_catalogue', fail)
    assert page.install_selected()
    qtbot.waitUntil(lambda: page._job is None)
    assert 'failed: offline wheel unavailable' in page.status.text()
    assert page.install_button.isEnabled()
    assert page.source.isEnabled()


def test_committed_install_is_not_reported_failed_when_catalogue_refresh_fails(page, qtbot, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('catalogue removed')

    monkeypatch.setattr(plugins, '_catalogue_rows', fail)
    assert page.install_selected()
    qtbot.waitUntil(lambda: page._job is None)
    assert plugins._catalogue_installed()['toxo_infection']['version'] == '0.2'
    assert 'Installed Toxoplasma infection assay 0.2' in page.status.text()
    assert 'Could not read the catalogue: catalogue removed' in page.status.text()
    assert 'failed:' not in page.status.text()
    assert page.selected()['installed'] == '0.2'
    assert page.selected()['status'] == 'installed'
    assert page.uninstall_button.isEnabled()
    assert page.open_button.isEnabled()
    assert not page.install_button.isEnabled()
    assert page.uninstall_selected()
    qtbot.waitUntil(lambda: page._job is None)
    assert 'Uninstalled Toxoplasma infection assay' in page.status.text()
    assert 'Could not read the catalogue: catalogue removed' in page.status.text()
    assert page.selected()['installed'] == ''
    assert page.selected()['status'] == 'available'
    assert page.install_button.isEnabled()
    assert not page.uninstall_button.isEnabled()
    assert not page.open_button.isEnabled()


@pytest.mark.parametrize('initial', [True, False])
def test_initial_and_explicit_listing_leave_gui_responsive(page, qtbot, monkeypatch, initial):
    started = threading.Event()
    release = threading.Event()
    actual = plugins._catalogue_rows
    main_thread = threading.get_ident()
    calls = []

    def blocked(source):
        calls.append((threading.get_ident(), source))
        started.set()
        assert release.wait(10)
        return actual(source)

    monkeypatch.setattr(plugins, '_catalogue_rows', blocked)
    if initial:
        dialog = QDialog()
        qtbot.addWidget(dialog)
        target = preferences._PluginCataloguePage(QFormLayout(dialog), dialog)
    else:
        target = page
        assert target.refresh()
    timer = QTimer(target._dialog)
    ticks = []
    timer.setInterval(1)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    try:
        qtbot.waitUntil(started.is_set)
        qtbot.waitUntil(lambda: len(ticks) > 2)
        assert calls[0][0] != main_thread
        assert calls[0][1] == target.source.text()
        assert target.table.rowCount() == 0
        assert not target.load_button.isEnabled()
        assert not target.install_button.isEnabled()
        assert not target.refresh()
        assert not target.install_selected()
    finally:
        release.set()
    qtbot.waitUntil(lambda: target._job is None)
    assert target.table.rowCount() == 2
    assert target.load_button.isEnabled()
    assert target.status.text() == '2 entries in the catalogue.'


def test_destroyed_window_cannot_receive_listing_results(page, qtbot, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    actual = plugins._catalogue_rows

    def blocked(source):
        started.set()
        assert release.wait(10)
        return actual(source)

    monkeypatch.setattr(plugins, '_catalogue_rows', blocked)
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    target = preferences._PluginCataloguePage(QFormLayout(dialog), dialog)
    dialog.show()
    job = target._job
    try:
        qtbot.waitUntil(started.is_set)
        dialog.close()
        qtbot.waitUntil(lambda: not isValid(dialog))
        assert not isValid(target._job_timer)
        assert job.is_alive()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not job.is_alive())
    assert job.error is None
    assert len(job.rows) == 2


@pytest.mark.parametrize('failed', [True, False])
def test_a_changed_source_cannot_receive_stale_listing_result(page, qtbot, monkeypatch, failed):
    started = threading.Event()
    release = threading.Event()
    actual = plugins._catalogue_rows

    def blocked(source):
        started.set()
        assert release.wait(10)
        if failed:
            raise OSError('old source unavailable')
        return actual(source)

    monkeypatch.setattr(plugins, '_catalogue_rows', blocked)
    assert page.refresh()
    try:
        qtbot.waitUntil(started.is_set)
        page.source.setText('/another/catalogue.json')
    finally:
        release.set()
    qtbot.waitUntil(lambda: page._job is None)
    assert page.table.rowCount() == 0
    assert page._rows == []
    assert page.status.text() == ''
    assert page.load_button.isEnabled()


def test_failed_listing_clears_previous_rows_and_shows_error(page, qtbot, monkeypatch):
    def fail(source):
        raise OSError('catalogue permission denied')

    assert page.table.rowCount() == 2
    monkeypatch.setattr(plugins, '_catalogue_rows', fail)
    assert page.refresh()
    qtbot.waitUntil(lambda: page._job is None)
    assert page.table.rowCount() == 0
    assert page._rows == []
    assert 'Could not read the catalogue: catalogue permission denied' == page.status.text()
    assert page.load_button.isEnabled()
    assert not page.install_button.isEnabled()


def test_catalogue_open_ignores_non_recipe_records(page, monkeypatch):
    monkeypatch.setattr(plugins, '_catalogue_installed', lambda: {
        'toxo_infection': {'kind': 'plugin'},
    })
    assert page._open_selected() is False
    assert page._dialog.isVisible() is False
    assert page.status.text() == '2 entries in the catalogue.'


@pytest.mark.parametrize('app', ['', 'unknown_desktop_module'])
def test_catalogue_open_reports_missing_recipe_destinations(
        page, monkeypatch, app):
    monkeypatch.setattr(plugins, '_catalogue_installed', lambda: {
        'toxo_infection': {'kind': 'recipe', 'app': app, 'path': ''},
    })
    assert page._open_selected() is False
    assert 'Toxoplasma infection assay failed:' in page.status.text()
    assert 'Could not apply template' in page.status.text()


def test_catalogue_open_needs_a_host_window_that_can_open_modules(
        page, monkeypatch):
    loaded = []
    monkeypatch.setattr(plugins, '_catalogue_installed', lambda: {
        'toxo_infection': {'kind': 'recipe', 'app': 'measure', 'path': 'recipe.csv'},
    })
    monkeypatch.setattr('spacr.cli.load_settings_file',
                        lambda path: loaded.append(path) or {'channels': [0]})
    assert page._open_selected() is False
    assert loaded == ['recipe.csv']
    assert 'Could not apply template' in page.status.text()
