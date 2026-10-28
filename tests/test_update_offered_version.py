"""An accepted update must install and verify the offered version."""
import json
import os
import sys

import pytest

from spacr import updater


@pytest.mark.parametrize("uv", [None, "/private/runtime/uv"])
def test_offered_version_is_pinned_in_the_active_interpreter(monkeypatch, uv):
    monkeypatch.setattr(updater, "find_uv", lambda: uv)
    command = updater.upgrade_command(target_version="1.5.0.9")
    assert command[-1] == "spacr==1.5.0.9"
    assert sys.executable in command


@pytest.mark.parametrize("probe,success", [((0, json.dumps({'verified': True, 'version': '1.5.0.9'})), True),
                                        ((0, json.dumps({'verified': True, 'version': '1.5.0.8'})), False),
                                        ((1, "No package metadata was found"), False)])
def test_success_requires_a_fresh_verified_version(monkeypatch, probe, success):
    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "find_uv", lambda: None)
    monkeypatch.setattr(updater, "_installed_version", lambda: "1.5.0.8")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return (0, "pip completed") if len(calls) == 1 else probe
    monkeypatch.setattr(updater, "run_install_command", run)
    result = updater.run_pip_upgrade(target_version="1.5.0.9")
    assert (result[0] == 0) == success
    assert calls[1][:3] == [sys.executable, "-I", "-c"]
    if not success:
        assert "did not verify" in result[1]


def test_source_checkout_cannot_report_a_requested_release_as_installed(monkeypatch):
    monkeypatch.setattr(updater, "editable_install_location", lambda: "/source/spacr")
    monkeypatch.setattr(updater, "run_install_command",
                        lambda *args, **kwargs: pytest.fail("source checkout must remain untouched"))
    code, explanation = updater.run_pip_upgrade(target_version="1.5.0.9")
    assert code != 0
    assert "git pull" in explanation


def test_relaunch_uses_this_interpreter_and_a_surviving_directory(monkeypatch, tmp_path):
    from spacr import install_cleanup
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    monkeypatch.setattr(install_cleanup, "_spawn_detached",
                        lambda command, **kwargs: calls.append((command, kwargs)) or 123)
    assert updater.launch_updated_app() == 123
    assert calls == [([sys.executable, "-I", "-m", "spacr.qt", "--resume-restart"], {"cwd": str(tmp_path)})]


@pytest.mark.parametrize('initial_reset', [None, '0'])
def test_frozen_relaunch_uses_exact_bundle_and_preserves_saved_workspace(monkeypatch, tmp_path, initial_reset):
    from spacr import install_cleanup, restart_state

    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('SPACR_HOME', str(tmp_path / 'private preferences'))
    bundled = tmp_path / 'Program Files' / 'spaCR' / 'spacr.exe'
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b'modeled frozen executable, never executed')
    settings = {'src': str(tmp_path / 'my images'), 'cell_channel': 2}
    saved = restart_state.save(module='mask', settings=settings)
    assert saved is not None
    before = saved.read_bytes()
    if initial_reset is None:
        monkeypatch.delenv('PYINSTALLER_RESET_ENVIRONMENT', raising=False)
    else:
        monkeypatch.setenv('PYINSTALLER_RESET_ENVIRONMENT', initial_reset)
    parent_environment = dict(os.environ)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(bundled))
    calls = []
    monkeypatch.setattr(install_cleanup, '_spawn_detached',
                        lambda argv, **kwargs: calls.append((argv, kwargs)) or 456)
    assert updater.launch_updated_app() == 456
    assert calls == [([str(bundled), '--resume-restart'], {'cwd': str(tmp_path),
                     'env': {**parent_environment, 'PYINSTALLER_RESET_ENVIRONMENT': '1'}})]
    assert calls[0][1]['env'] is not os.environ
    assert dict(os.environ) == parent_environment
    assert saved.read_bytes() == before
    assert restart_state.peek()['module'] == 'mask'
    assert restart_state.peek()['settings'] == settings


@pytest.mark.parametrize('probe,success', [
    ((0, json.dumps({'verified': True, 'version': '1.5.0.9'})), True),
    ((0, json.dumps({'verified': True, 'version': '1.5.0.8'})), False),
    ((0, json.dumps({'verified': False, 'version': '1.5.0.9'})), False),
])
def test_unknown_installed_version_still_requires_exact_fresh_verification(monkeypatch, probe, success):
    from spacr import install_cleanup

    monkeypatch.setattr(sys, 'frozen', False, raising=False)
    monkeypatch.setattr(updater, 'editable_install_location', lambda: None)
    monkeypatch.setattr(updater, 'find_uv', lambda: None)
    monkeypatch.setattr(updater, '_installed_version', lambda: 'unknown-metadata')
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            assert argv == [sys.executable, '-m', 'pip', 'install', '--upgrade', 'spacr==1.5.0.9']
            return 0, 'modeled pip completed'
        assert len(calls) == 2
        assert argv == install_cleanup._runtime_identity_command(sys.executable, '1.5.0.9', prefix=sys.prefix)
        assert kwargs == {'timeout': 30}
        return probe

    monkeypatch.setattr(updater, 'run_install_command', run)
    result = updater.run_pip_upgrade(target_version='1.5.0.9')
    assert len(calls) == 2
    assert (result[0] == 0) is success
    assert 'modeled pip completed' in result[1]
    if not success:
        assert 'did not verify spaCR 1.5.0.9' in result[1]
