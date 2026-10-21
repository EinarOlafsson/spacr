"""Startup diagnostics retain actionable failures when widgets or logs disappear."""
from types import SimpleNamespace

import pytest


def test_stack_dump_setup_can_fail_without_claiming_a_log(monkeypatch):
    from spacr.qt import startup_benchmark, verbose_logger

    def unavailable():
        raise OSError('log storage unavailable')

    monkeypatch.setenv('SPACR_DISTRIBUTION_SMOKE', '1')
    monkeypatch.setattr(verbose_logger, 'current_log_file', unavailable)
    monkeypatch.setattr(startup_benchmark, '_SMOKE_STACK_DUMP_ARMED', False)
    record = {}
    assert startup_benchmark._arm_the_smoke_stack_dump(record) is None
    assert record == {}
    assert not startup_benchmark._SMOKE_STACK_DUMP_ARMED


def test_modal_diagnostic_survives_one_deleted_qt_accessor():
    from spacr.qt import startup_benchmark

    class Modal:
        def objectName(self):
            raise RuntimeError('Qt object was deleted')

        def windowTitle(self):
            return 'Pending import'

    writes = []
    controller = SimpleNamespace(app=SimpleNamespace(activeModalWidget=Modal),
                                 record={}, _write=lambda: writes.append(True))
    message = startup_benchmark._name_the_blocking_modal(controller, 'Startup blocked')
    assert 'Pending import' in message
    assert controller.record['blocking_modal'] == {'cls': 'Modal', 'windowTitle': 'Pending import'}
    assert writes == [True]


def test_collapsed_heading_probe_failure_is_reported_as_a_layout_fault(qtbot, monkeypatch):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import startup_benchmark
    from spacr.qt.widgets import collapsible_splitter

    window = QWidget()
    qtbot.addWidget(window)

    def deleted(widget):
        raise RuntimeError('heading was deleted')

    monkeypatch.setattr(collapsible_splitter, '_collapsed_header_faults', deleted)
    assert startup_benchmark._collapsed_heading_faults_of(
        SimpleNamespace(window=window)) == [{'problem': 'collapsed-heading check failed: heading was deleted'}]


def test_startup_diagnostic_does_not_wait_forever_on_a_failed_warm_probe(monkeypatch):
    import sys
    from spacr.qt.startup_benchmark import _DistributionSmokeController

    def failed():
        raise RuntimeError('warm worker was destroyed')

    monkeypatch.setitem(sys.modules, 'spacr.qt.app', SimpleNamespace(_settings_prewarm_running=failed))
    controller = SimpleNamespace(record={})
    assert not _DistributionSmokeController._start_up_warm_running(controller)
    assert controller.record == {}


def test_failed_collapsed_heading_check_cannot_pass_visual_acceptance(monkeypatch):
    from spacr.qt import startup_benchmark

    faults = [{'problem': 'heading clipped'}]
    monkeypatch.setattr(startup_benchmark, '_collapsed_heading_faults_of', lambda controller: faults)
    writes = []
    record = {'layout': {'status': 'passed'}, 'scientific_result': {'status': 'passed'}}
    controller = SimpleNamespace(record=record, _write=lambda: writes.append(True))
    with pytest.raises(RuntimeError, match='Installed UI layout did not settle'):
        startup_benchmark._DistributionSmokeController._finish_layout_check(controller)
    assert record['layout']['status'] == record['visual_layout_status'] == 'failed'
    assert record['layout']['collapsed_heading_faults'] == faults
    assert record['scientific_result'] == {'status': 'passed'}
    assert writes == [True]


def test_keep_click_that_leaves_cleanup_open_is_a_smoke_failure(qtbot, monkeypatch):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt import install_consent, startup_benchmark

    keep = QPushButton('Keep')
    remove = QPushButton('Remove')
    for button in (keep, remove):
        qtbot.addWidget(button)
        button.show()
    keep.setDefault(True)
    remove.setEnabled(False)

    class Cleanup:
        _folder_choices = []

        def windowHandle(self):
            return SimpleNamespace(isExposed=lambda: True)

        def choices(self):
            return {'keys': []}

        def findChild(self, kind, name):
            return keep if name == 'InstallerCleanupKeep' else remove

        def isVisible(self):
            return True

    dialog = Cleanup()
    monkeypatch.setattr(install_consent, '_InstallerCleanupDialog', Cleanup)
    controller = SimpleNamespace(app=SimpleNamespace(activeModalWidget=lambda: dialog),
                                 _cleanup_review_seen=dialog)
    with pytest.raises(RuntimeError, match='Keeping installation files did not close'):
        startup_benchmark._DistributionSmokeController._keep_installer_cleanup(controller)


@pytest.mark.parametrize('script', ['spaCR.app/Contents/MacOS/other', 'not-an-app/Contents/MacOS/spaCR'])
def test_macos_launcher_lookalikes_do_not_hide_another_installation(tmp_path, monkeypatch, script):
    from spacr import doctor

    monkeypatch.setattr(doctor, 'sys', SimpleNamespace(platform='darwin'))
    assert not doctor._macos_launcher_starts(tmp_path / script, tmp_path / 'runtime')
