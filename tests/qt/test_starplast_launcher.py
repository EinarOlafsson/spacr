"""First-click consent, responsive cancellation, retry and later direct launches."""
import threading

from PySide6.QtWidgets import QDialog

from spacr.qt import starplast
from spacr.qt.screens.organism_screen import OrganismScreen


def test_starplast_is_the_first_toxoplasma_tile_and_launches_outside_registry(qtbot, monkeypatch):
    opened, navigated = [], []
    monkeypatch.setattr(starplast, 'open_starplast', lambda parent: opened.append(parent))
    page = OrganismScreen('toxoplasma')
    qtbot.addWidget(page)
    page.module_requested.connect(navigated.append)
    tile = page._tiles[0]
    assert tile.property('organismModuleKey') == 'starplast'
    assert tile.property('stage') == 'alpha'
    tile.click()
    assert opened == [page] and not navigated
    for key in ('plasmodium', 'candida'):
        other = OrganismScreen(key)
        qtbot.addWidget(other)
        assert all(tile.property('organismModuleKey') != 'starplast' for tile in other._tiles)


def test_dialog_warns_before_work_and_remains_responsive_while_cancelling(qtbot, tmp_path):
    entered = threading.Event()
    calls = []

    def job(source, *, root, progress, cancel):
        calls.append(source)
        entered.set()
        progress(1, 4, 'Waiting for cancellation')
        assert cancel.wait(5)
        raise starplast._InstallCancelled('cancelled')

    dialog = starplast.StarplastInstallDialog(root=tmp_path, job=job)
    qtbot.addWidget(dialog)
    dialog.show()
    assert 'alpha' in dialog.explanation.text()
    assert '2 GB' in dialog.explanation.text() and '12 GB' in dialog.explanation.text()
    assert str(tmp_path/'starplast') in dialog.explanation.text()
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

    dialog = starplast.StarplastInstallDialog(root=tmp_path, job=job)
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
    monkeypatch.setattr(starplast.service, 'is_installed', lambda root: True)
    monkeypatch.setattr(starplast, 'StarplastInstallDialog', lambda *a, **k: (_ for _ in ()).throw(AssertionError('reinstall')))
    monkeypatch.setattr(starplast.QMessageBox, 'warning', lambda *args: errors.append(args))
    class Child:
        def poll(self):
            return 7
    child = Child()
    monkeypatch.setattr(starplast.service, 'launch_starplast', lambda **kwargs: child)
    assert starplast.open_starplast(root=tmp_path) is child
    qtbot.waitUntil(lambda: bool(errors))
    assert '7' in errors[0][-1] and 'starplast-launch.log' in errors[0][-1]


def test_cancelled_first_click_never_launches(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(starplast.service, 'is_installed', lambda root: False)
    class Cancelled:
        installed = False
        def __init__(self, *args, **kwargs):
            pass
        def exec(self):
            return QDialog.Rejected
    monkeypatch.setattr(starplast, 'StarplastInstallDialog', Cancelled)
    monkeypatch.setattr(starplast.service, 'launch_starplast', lambda **kwargs: (_ for _ in ()).throw(AssertionError('launched')))
    assert starplast.open_starplast(root=tmp_path) is None
