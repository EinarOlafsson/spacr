"""Only the replacement process gets the one-shot restart launch marker."""
import sys

import pytest

from spacr import restart_state
from spacr.qt.shutdown import restart_spacr


@pytest.fixture(autouse=True)
def private_restart_home(tmp_path, monkeypatch):
    monkeypatch.setenv('SPACR_HOME', str(tmp_path / 'private'))


@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.parametrize('name', ['spacr', 'spaceout-nav'])
def test_restart_launcher_marks_interpreter_and_bundled_spaceout_routes_once(
        monkeypatch, tmp_path, frozen, name):
    executable = str(tmp_path / 'Program Files' / name)
    monkeypatch.setattr(sys, 'executable', executable)
    monkeypatch.setattr(sys, 'frozen', frozen, raising=False)
    monkeypatch.setattr(sys, 'argv', [name, '--resume-restart', '--fresh'])
    argv = restart_state.command()
    assert argv == ([executable] if frozen else [executable, '-m', 'spacr.qt']) + ['--resume-restart']
    assert argv.count('--resume-restart') == 1
    assert '--fresh' not in argv


@pytest.mark.parametrize('module', ['', 'measure'])
def test_explicit_restart_preserves_home_or_module_settings_once_before_exit(module):
    order = []
    settings = {'src': '/project/plate1', 'cell_channel': 2}

    def launch(argv):
        assert argv[-1] == '--resume-restart'
        record = restart_state.peek()
        assert record['module'] == module and record['settings'] == settings
        assert record['run_folders'] == ['/project/results']
        order.append('launch')

    assert restart_spacr(module, settings, run_folders=['/project/results'], launcher=launch,
                         exiter=lambda code: order.append(('exit', code)))
    assert order == ['launch', ('exit', 0)]
    assert restart_state.take()['module'] == module
    assert restart_state.take() is None


def test_failed_launch_discards_only_pending_restart_and_keeps_current_process():
    exits = []

    def refuse(argv):
        assert '--resume-restart' in argv
        assert restart_state.peek()['module'] == 'measure'
        raise OSError('replacement process unavailable')

    assert not restart_spacr('measure', {'src': '/project'}, launcher=refuse, exiter=exits.append)
    assert not exits
    assert restart_state.peek() is None


def test_failed_verified_save_does_not_launch_or_exit(monkeypatch):
    monkeypatch.setattr(restart_state, 'save', lambda **kwargs: None)
    launched, exited = [], []
    assert not restart_spacr('measure', {}, launcher=launched.append, exiter=exited.append)
    assert not launched and not exited
