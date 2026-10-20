"""Windows removal uses a copied private interpreter and the shared data contract."""
from pathlib import Path
import json
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]


def _put(path, text='private runtime fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def private_runtime(tmp_path):
    root = tmp_path / 'private installation'
    base = root / 'python' / 'managed-cpython'
    _put(base / 'python.exe')
    _put(base / 'Lib/json.py')
    executable = _put(root / 'venv/Scripts/python.exe')
    work = tmp_path / 'outside-private'
    work.mkdir(mode=0o700)
    machine = ic._Machine(platform='windows', fs_root=str(tmp_path),
                          environ={'USERPROFILE': str(tmp_path / 'user')},
                          executable=str(executable), running_prefix=str(root / 'venv'))
    return root, base, work, machine


def test_only_the_installers_managed_python_and_stdlib_helper_are_copied(private_runtime):
    root, base, work, machine = private_runtime
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    plan = ic._prepare_windows_uninstall(str(root), str(work), machine=machine, base_prefix=str(base))
    assert plan['command'] == [str(work / 'runtime/python.exe'), '-I', str(work / 'install_cleanup.py'),
                               'remove', '--only-root', str(root), '--require-outside']
    assert (work / 'runtime/Lib/json.py').read_bytes() == (base / 'Lib/json.py').read_bytes()
    assert (work / 'install_cleanup.py').read_bytes() == Path(ic.__file__).read_bytes()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('reason', ['shared-python', 'wrong-interpreter', 'wrong-prefix', 'inside-folder',
                                    'nonempty-folder', 'linked-content', 'configured-data'])
def test_preparation_refuses_to_copy_unowned_or_linked_runtime(private_runtime, tmp_path, reason):
    root, base, work, machine = private_runtime
    if reason == 'shared-python':
        base = tmp_path / 'shared-python'
        _put(base / 'python.exe')
    elif reason == 'wrong-interpreter':
        machine.executable = sys.executable
    elif reason == 'wrong-prefix':
        machine.running_prefix = str(tmp_path)
    elif reason == 'inside-folder':
        work = root / 'temporary'
        work.mkdir(mode=0o700)
    elif reason == 'nonempty-folder':
        _put(work / 'preexisting-user-file')
    elif reason == 'linked-content':
        (base / 'linked-user-file').symlink_to(_put(tmp_path / 'user-file'))
    elif reason == 'configured-data':
        machine.environ['SPACR_HOME'] = str(base / 'user-data')
        _put(base / 'user-data/valuable.csv')
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        ic._prepare_windows_uninstall(str(root), str(work), machine=machine, base_prefix=str(base))
    assert not (work / 'runtime').exists()
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_outside_guard_refuses_running_environment_before_discovery_or_deletion():
    result = subprocess.run([sys.executable, '-I', ic.__file__, 'remove', '--only-root',
                             sys.prefix, '--require-outside'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert result.stdout.startswith('Uninstall requires a helper and interpreter outside')
    assert not result.stderr


def test_nsis_runs_noninteractive_outside_helper_then_removes_only_empty_root():
    source = (ROOT / 'packaging/online/spacr_online_installer.nsi').read_text(encoding="utf-8")
    uninstall = source.split('Section "Uninstall"', 1)[1]
    assert 'RMDir /r "$INSTDIR"' not in uninstall
    assert 'RMDir /r "$SMPROGRAMS' not in uninstall
    assert 'File "uninstall_spacr_windows.ps1"' in uninstall
    assert '-NoProfile -NonInteractive -ExecutionPolicy Bypass' in uninstall
    assert uninstall.index('Pop $0') < uninstall.index('Abort') < uninstall.index('RMDir "$INSTDIR"')
    script = (ROOT / 'packaging/online/uninstall_spacr_windows.ps1').read_text(encoding="utf-8")
    assert script.index('SetAccessRuleProtection($true, $false)') < script.index('prepare-windows-uninstall')
    assert script.index('prepare-windows-uninstall') < script.index('& $OutsidePython')
    assert '--only-root $InstallRoot --require-outside' in script
    assert 'Read-Host' not in script and 'Invoke-WebRequest' not in script
    assert 'Remove-Item -LiteralPath $WorkDir' in script
    assert 'Remove-Item -LiteralPath $InstallRoot' not in script


def test_normal_nsis_renderer_includes_exact_reviewed_uninstall_message_in_every_language():
    import importlib.util
    spec = importlib.util.spec_from_file_location('installer_renderer', ROOT / 'packaging/i18n/render.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    values = renderer.catalogs()
    text = renderer.render_nsis(values)
    for language, name in renderer.NSIS_LANGUAGE.items():
        source = json.loads((ROOT / 'packaging/i18n' / (language + '.json')).read_text(encoding="utf-8"))['nsis_uninstall_failed']
        expected = f'LangString SPACR_NSIS_UNINSTALL_FAILED ${{LANG_{name.upper()}}} "{renderer._nsis_escape(source)}"'
        assert expected in text
        if language != 'en':
            assert source != values['en']['nsis_uninstall_failed']
