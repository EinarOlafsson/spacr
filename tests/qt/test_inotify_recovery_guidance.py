"""Linux watch exhaustion offers pasteable recovery without changing the system."""
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('current,expected', [('65536', 524288), ('1048576', 2097152), ('invalid', None)])
def test_inotify_guidance_is_once_only_and_never_lowers_current_limit(monkeypatch, capsys, current, expected):
    import spacr.qt as qt
    from spacr.qt import i18n

    monkeypatch.setattr(qt, '_SAID_IT_ONCE', False)
    monkeypatch.setattr(qt, 'sys', SimpleNamespace(platform='linux', stderr=sys.stderr))
    monkeypatch.setattr(i18n, 'tr', lambda source: source)
    original = Path.read_text

    def read(path, *args, **kwargs):
        return current if str(path) == '/proc/sys/fs/inotify/max_user_watches' else original(path, *args, **kwargs)

    def forbid(*args, **kwargs):
        raise AssertionError('Guidance must not run a command or change system settings')

    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(Path, 'write_text', forbid)
    monkeypatch.setattr('os.system', forbid)
    monkeypatch.setattr('subprocess.run', forbid)
    qt._explain_the_inotify_line()
    message = capsys.readouterr().err
    if expected is None:
        assert 'spacr_watch_limit=$(sysctl -n fs.inotify.max_user_watches) &&' in message
        assert 'sudo sysctl -w "fs.inotify.max_user_watches=$spacr_watch_limit"' in message
    else:
        assert ['sudo', 'sysctl', '-w', f'fs.inotify.max_user_watches={expected}'] == shlex.split(next(line.strip() for line in message.splitlines() if 'sudo sysctl -w' in line))
    line = next(line.strip() for line in message.splitlines() if 'printf' in line)
    assert shlex.split(line)[:2] == ['printf', '%s\\n']
    assert len(shlex.split(line)[1]) == 4
    assert shlex.split(line)[2] == (f'fs.inotify.max_user_watches={expected}' if expected is not None else 'fs.inotify.max_user_watches=$spacr_watch_limit')
    assert 'sudo tee /etc/sysctl.d/90-spacr-inotify.conf &&\nsudo sysctl -p /etc/sysctl.d/90-spacr-inotify.conf' in message
    assert 'Rebooting clears all watches' in message
    assert 'Restarting does not increase the configured limit' in message
    assert 'Linux-only' in message and 'not disk space' in message
    qt._explain_the_inotify_line()
    assert capsys.readouterr().err == ''


@pytest.mark.parametrize('platform', ['darwin', 'win32'])
def test_non_linux_never_receives_linux_system_commands(monkeypatch, capsys, platform):
    import spacr.qt as qt

    monkeypatch.setattr(qt, '_SAID_IT_ONCE', False)
    monkeypatch.setattr(qt, 'sys', SimpleNamespace(platform=platform, stderr=sys.stderr))
    qt._explain_the_inotify_line()
    assert capsys.readouterr().err == ''
