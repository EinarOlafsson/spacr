"""First-click consent, responsive cancellation, retry and later direct launches."""
import threading

import pytest
from PySide6.QtWidgets import QDialog

from spacr.qt import parascape
from spacr.qt.screens.organism_screen import OrganismScreen


def test_parascape_is_the_first_toxoplasma_tile_and_launches_outside_registry(qtbot, monkeypatch):
    opened, navigated = [], []
    monkeypatch.setattr(parascape, 'open_parascape', lambda parent: opened.append(parent))
    page = OrganismScreen('toxoplasma')
    qtbot.addWidget(page)
    page.module_requested.connect(navigated.append)
    tile = page._tiles[0]
    assert tile.property('organismModuleKey') == 'parascape'
    assert tile.property('stage') == 'alpha'
    tile.click()
    assert opened == [page] and not navigated
    for key in ('plasmodium', 'candida'):
        other = OrganismScreen(key)
        qtbot.addWidget(other)
        assert all(tile.property('organismModuleKey') != 'parascape' for tile in other._tiles)


def test_dialog_warns_before_work_and_remains_responsive_while_cancelling(qtbot, tmp_path):
    entered = threading.Event()
    calls = []

    def job(source, *, root, progress, cancel):
        calls.append(source)
        entered.set()
        progress(1, 4, 'Waiting for cancellation')
        assert cancel.wait(5)
        raise parascape._InstallCancelled('cancelled')

    dialog = parascape.ParaScapeInstallDialog(root=tmp_path, job=job)
    qtbot.addWidget(dialog)
    dialog.show()
    assert 'alpha' in dialog.explanation.text()
    assert '12 GB' in dialog.explanation.text()
    assert 'PyPI' in dialog.explanation.text()
    assert dialog.source.text() == 'parascape'
    assert str(tmp_path/'parascape') in dialog.explanation.text()
    assert not calls
    dialog.start_button.click()
    qtbot.waitUntil(entered.is_set)
    qtbot.waitUntil(lambda: 'Waiting for cancellation' in dialog.status.text())
    assert not dialog.source.isEnabled()
    dialog.reject()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert not dialog.installed and dialog.result() == QDialog.Rejected


def test_install_failure_can_retry_and_success_closes_only_after_thread_finishes(qtbot, tmp_path):
    calls = []
    def job(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('fixture network failure')

    dialog = parascape.ParaScapeInstallDialog(root=tmp_path, job=job)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.start()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert 'fixture network failure' in dialog.details.toPlainText()
    assert dialog.start_button.isEnabled() and not dialog.installed
    dialog.start_button.click()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert dialog.installed and dialog.result() == QDialog.Accepted


def test_later_click_skips_installer_and_reports_child_failure(qtbot, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: True)
    monkeypatch.setattr(parascape, 'ParaScapeInstallDialog', lambda *a, **k: (_ for _ in ()).throw(AssertionError('reinstall')))
    monkeypatch.setattr(parascape.QMessageBox, 'warning', lambda *args: errors.append(args))
    class Child:
        def poll(self):
            return 7
    child = Child()
    monkeypatch.setattr(parascape.service, 'launch_parascape', lambda **kwargs: child)
    assert parascape.open_parascape(root=tmp_path) is child
    qtbot.waitUntil(lambda: bool(errors))
    assert '7' in errors[0][-1] and 'parascape-launch.log' in errors[0][-1]


def test_cancelled_first_click_never_launches(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: False)
    class Cancelled:
        installed = False
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            return QDialog.Rejected
    monkeypatch.setattr(parascape, 'ParaScapeInstallDialog', Cancelled)
    monkeypatch.setattr(parascape.service, 'launch_parascape', lambda **kwargs: (_ for _ in ()).throw(AssertionError('launched')))
    assert parascape.open_parascape(root=tmp_path) is None


def test_choosing_a_checkout_changes_source_only_after_a_selection(qtbot, monkeypatch, tmp_path):
    dialog = parascape.ParaScapeInstallDialog(root=tmp_path)
    qtbot.addWidget(dialog)
    before = dialog.source.text()
    monkeypatch.setattr(parascape.QFileDialog, 'getExistingDirectory', lambda *args: '')
    dialog.browse.click()
    assert dialog.source.text() == before
    monkeypatch.setattr(parascape.QFileDialog, 'getExistingDirectory', lambda *args: str(tmp_path))
    dialog.browse.click()
    assert dialog.source.text() == str(tmp_path)
    assert dialog._thread is None and not dialog.installed


def test_window_close_cancels_but_keeps_running_worker_alive_until_finished(qtbot, tmp_path):
    entered, finish = threading.Event(), threading.Event()
    calls = []

    def job(source, *, cancel, **kwargs):
        calls.append(source)
        entered.set()
        assert cancel.wait(5)
        assert finish.wait(5)
        raise parascape._InstallCancelled('cancelled')

    dialog = parascape.ParaScapeInstallDialog(root=tmp_path, job=job)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        dialog.start()
        qtbot.waitUntil(entered.is_set)
        dialog.start()
        assert len(calls) == 1
        assert not dialog.close()
        assert dialog.isVisible() and dialog._thread.isRunning()
        assert dialog._thread.cancel.is_set() and not dialog.cancel_button.isEnabled()
    finally:
        finish.set()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert not dialog.isVisible() and dialog.result() == QDialog.Rejected
    dialog.close()


@pytest.mark.parametrize('outcome', ['installed', 'cancelled', 'failed'])
def test_installer_worker_outcomes_are_safe_to_read_only_after_run(qtbot, tmp_path, outcome):
    def job(*args, **kwargs):
        kwargs['progress'](1, 2, 'status')
        if outcome == 'cancelled':
            raise parascape._InstallCancelled('cancelled')
        if outcome == 'failed':
            raise OSError('disk unavailable')

    worker = parascape._InstallThread('fixture', tmp_path, job, None)
    progress = []
    worker.progressed.connect(lambda *args: progress.append(args))
    assert worker.outcome == ''
    worker.run()
    assert worker.outcome == outcome
    assert worker.error == ('disk unavailable' if outcome == 'failed' else '')
    assert progress == [(1, 2, 'status')]
    worker.deleteLater()


def test_unsolicited_cancellation_allows_retry_without_closing_dialog(qtbot, tmp_path):
    def job(*args, **kwargs):
        raise parascape._InstallCancelled('cancelled')

    dialog = parascape.ParaScapeInstallDialog(root=tmp_path, job=job)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.start()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert dialog.isVisible() and not dialog.installed
    assert dialog.start_button.isEnabled()
    assert 'cancelled' in dialog.status.text()


def test_launch_error_is_reported_without_starting_a_poll_timer(qtbot, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: True)
    monkeypatch.setattr(parascape.QMessageBox, 'warning', lambda *args: errors.append(args))

    def launch(**kwargs):
        raise OSError('cannot start interpreter')

    monkeypatch.setattr(parascape.service, 'launch_parascape', launch)
    assert parascape.open_parascape(root=tmp_path) is None
    assert errors[0][-1] == 'cannot start interpreter'


def test_first_successful_install_launches_and_reaps_a_successful_child(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: False)
    errors = []
    monkeypatch.setattr(parascape.QMessageBox, 'warning', lambda *args: errors.append(args))

    class Installed:
        installed = True
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            return QDialog.Accepted

    class Child:
        calls = 0
        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 0

    child = Child()
    monkeypatch.setattr(parascape, 'ParaScapeInstallDialog', Installed)
    monkeypatch.setattr(parascape.service, 'launch_parascape', lambda **kwargs: child)
    assert parascape.open_parascape(root=tmp_path) is child
    qtbot.waitUntil(lambda: child.calls == 2, timeout=3000)
    qtbot.wait(600)
    assert child.calls == 2 and not errors


@pytest.fixture(autouse=True)
def default_update_check(monkeypatch, qtbot):
    monkeypatch.setattr(parascape.service, 'check_parascape_update',
                        lambda **kwargs: dict(installed='1.0', latest='1.0', available=False))
    def exec_check(dialog):
        dialog.show()
        qtbot.waitUntil(lambda: dialog._thread is None)
        return dialog.result()
    monkeypatch.setattr(parascape.ParaScapeUpdateCheckDialog, 'exec', exec_check)


@pytest.mark.parametrize('upgrade', [False, True])
def test_each_open_checks_updates_and_only_consent_runs_upgrade(qtbot, monkeypatch, tmp_path, upgrade):
    checks, upgrades, launches = [], [], []
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: True)
    def check(**kwargs):
        checks.append(1)
        return dict(installed='1.0', latest='1.1', available=True)
    monkeypatch.setattr(parascape.service, 'check_parascape_update', check)
    def prompt(box):
        assert '1.0' in box.text() and '1.1' in box.text() and 'pip' in box.text()
        box.buttons()[0 if upgrade else 1].click()
        return 0
    monkeypatch.setattr(parascape.QMessageBox, 'exec', prompt)
    def install(dialog):
        assert dialog.source.text() == 'parascape'
        assert dialog.job == parascape.service.upgrade_parascape
        upgrades.append(1)
        dialog.installed = True
        return QDialog.Accepted
    monkeypatch.setattr(parascape.ParaScapeInstallDialog, 'exec', install)
    monkeypatch.setattr(parascape.ParaScapeInstallDialog, 'start', lambda self: None)
    class Child:
        def poll(self):
            return 0
    monkeypatch.setattr(parascape.service, 'launch_parascape', lambda **kw: launches.append(Child()) or launches[-1])
    for _ in range(2):
        parascape.open_parascape(root=tmp_path)
    assert len(checks) == len(launches) == 2
    assert len(upgrades) == (2 if upgrade else 0)


def test_update_check_is_responsive_and_cancellable(qtbot, monkeypatch, tmp_path):
    entered = threading.Event()
    def check(*, cancel, **kwargs):
        entered.set()
        assert cancel.wait(5)
        raise parascape._InstallCancelled('cancelled')
    monkeypatch.setattr(parascape.service, 'check_parascape_update', check)
    dialog = parascape.ParaScapeUpdateCheckDialog(root=tmp_path)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(entered.is_set)
    dialog.skip.click()
    qtbot.waitUntil(lambda: dialog._thread is None)
    assert dialog.result() == QDialog.Rejected


def test_offline_update_check_reports_and_opens_installed_app(qtbot, monkeypatch, tmp_path):
    messages = []
    monkeypatch.setattr(parascape.service, 'is_installed', lambda root: True)
    def check(**kwargs):
        raise OSError('offline')
    monkeypatch.setattr(parascape.service, 'check_parascape_update', check)
    monkeypatch.setattr(parascape.QMessageBox, 'information', lambda *args: messages.append(args))
    class Child:
        def poll(self):
            return 0
    child = Child()
    monkeypatch.setattr(parascape.service, 'launch_parascape', lambda **kwargs: child)
    assert parascape.open_parascape(root=tmp_path) is child
    assert 'offline' in messages[0][-1]
