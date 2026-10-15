"""Run the generated uninstaller on disposable installations, retaining user files."""
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]


def _write(path, text='preserved'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def installation(tmp_path):
    home = tmp_path / 'home'
    data = home / 'xdg-share'
    root = data / 'spacr'
    launcher = home / '.local/bin/spacr'
    desktop = data / 'applications/io.github.olafssonlab.spacr.desktop'
    _write(root / 'bootstrap/uv', 'application bootstrap')
    _write(root / 'venv/lib/spacr.py', 'application code')
    _write(launcher, f'exec "{root}/venv/bin/python" -I -m spacr.qt\n')
    _write(desktop, f'Exec={root}/venv/bin/python -I -m spacr.qt\n')
    protected = [_write(root / name / 'valuable.bin') for name in
                 ('cache', 'models', 'test_data', 'outputs', 'my-experiment')]
    configured = _write(root / 'venv' / 'configured-spacr-data' / 'results.csv')
    protected.append(configured)
    other = home / '.local/share/spacr'
    _write(other / 'bootstrap/uv', 'other installation')
    _write(other / 'uninstall-spacr.sh', 'other uninstaller')
    environment = {'HOME': str(home), 'XDG_DATA_HOME': str(data),
                   'SPACR_HOME': str(configured.parent)}
    driver = tmp_path / 'driver.py'
    driver.write_text('import importlib.util,sys\n'
                      f'spec=importlib.util.spec_from_file_location("cleanup", {ic.__file__!r})\n'
                      'module=importlib.util.module_from_spec(spec)\n'
                      'sys.modules[spec.name]=module\nspec.loader.exec_module(module)\n'
                      'original=module._Machine\n'
                      f'module._Machine=lambda **kw: original(platform="linux", environ={environment!r}, fs_root={str(tmp_path)!r})\n'
                      'raise SystemExit(module._main(sys.argv[1:]))\n')
    python = _write(root / 'venv/bin/python', '#!/bin/sh\nshift 3\nexec ' +
                    shlex.quote(sys.executable) + ' -I ' + shlex.quote(str(driver)) + ' "$@"\n')
    python.chmod(0o755)
    source = (ROOT / 'packaging/online/install_spacr_unix.sh').read_text()
    heredoc = 'cat > "$uninstall_path" <<EOF\n' + source.split('cat > "$uninstall_path" <<EOF\n', 1)[1].split('\nEOF', 1)[0] + '\nEOF\n'
    script = root / 'uninstall-spacr.sh'
    generator = tmp_path / 'generate.sh'
    values = {'uninstall_path': script, 'VENV_DIR': root / 'venv', 'LAUNCHER': launcher,
              'DESKTOP_DIR': desktop.parent, 'INSTALL_ROOT': root}
    generator.write_text('\n'.join(key + '=' + shlex.quote(str(value)) for key, value in values.items()) + '\n' + heredoc)
    subprocess.run(['bash', str(generator)], check=True, capture_output=True, timeout=10)
    return root, script, launcher, desktop, protected, other, driver


def test_generated_uninstaller_removes_only_application_and_preserves_folder_inventory(installation):
    root, script, launcher, desktop, protected, other, _driver = installation
    before = {path: path.read_bytes() for path in protected}
    result = subprocess.run(['sh', str(script)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / 'bootstrap').exists()
    assert not (root / 'venv/lib/spacr.py').exists()
    assert not script.exists() and not launcher.exists() and not desktop.exists()
    assert {path: path.read_bytes() for path in protected} == before
    assert (other / 'bootstrap/uv').read_text() == 'other installation'
    assert not result.stderr
    assert 'left:' in result.stdout


def test_changed_launcher_is_not_deleted_by_unconditional_shell_commands(installation):
    _root, script, launcher, desktop, protected, other, _driver = installation
    launcher.write_text(f'exec "{other}/venv/bin/python" -I -m spacr.qt\n')
    desktop.write_text(f'Exec={other}/venv/bin/python -I -m spacr.qt\n')
    before = launcher.read_bytes(), desktop.read_bytes()
    result = subprocess.run(['sh', str(script)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (launcher.read_bytes(), desktop.read_bytes()) == before
    assert all(path.exists() for path in protected)


@pytest.mark.parametrize('reason', ['unrecognized', 'symlink', 'purge'])
def test_targeted_cli_refuses_ambiguous_or_global_destructive_request(installation, tmp_path, reason):
    root, _script, _launcher, _desktop, protected, _other, driver = installation
    requested = root
    extra = []
    if reason == 'unrecognized':
        requested = tmp_path / 'unrelated'
        _write(requested / 'user-file')
    elif reason == 'symlink':
        requested = tmp_path / 'linked-install'
        requested.symlink_to(root, target_is_directory=True)
    elif reason == 'purge':
        extra = ['--purge', '--yes']
    result = subprocess.run([sys.executable, '-I', str(driver), 'remove', '--only-root', str(requested), *extra],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert 'Targeted removal requires' in result.stdout
    assert (root / 'bootstrap/uv').exists() and all(path.exists() for path in protected)
