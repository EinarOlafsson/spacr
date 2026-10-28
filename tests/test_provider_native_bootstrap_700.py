"""Vendor bootstrap plans run offline in disposable homes without Node/npm."""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr.qt.ai import cli_install, providers


@pytest.fixture
def offline_bootstrap(tmp_path, monkeypatch):
    home = tmp_path / 'profile with spaces'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.delenv('NVM_DIR', raising=False)
    monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
    programs = tmp_path / 'programs'
    programs.mkdir()
    (programs / 'bash').symlink_to('/bin/bash')
    (programs / 'sh').symlink_to('/bin/sh')
    (programs / 'mkdir').symlink_to('/bin/mkdir')
    payload = tmp_path / 'nvm-installer'
    payload.write_text('''/bin/mkdir -p "$NVM_DIR"
[ "$PROFILE" = /dev/null ] && [ "$METHOD" = script ] || exit 70
/bin/cat > "$NVM_DIR/nvm.sh" <<'NVM'
nvm() {
    printf '%s\\n' "$*" >> "$TRACE"
    if [ "$1" = "$FAIL_STAGE" ]; then return 31; fi
    if [ "$1" = install ]; then
        /bin/mkdir -p "$NVM_DIR/versions/node/v24.2.0/bin"
        /bin/cat > "$NVM_DIR/versions/node/v24.2.0/bin/node" <<'NODE'
#!/bin/sh
printf '%s\\n' 'fake Gemini via matching Node'
NODE
        /bin/cat > "$NVM_DIR/versions/node/v24.2.0/bin/npm" <<'NPM'
#!/bin/sh
printf 'npm %s\\n' "$*" >> "$TRACE"
[ "$FAIL_STAGE" != npm ] || exit 41
printf '%s\\n' '#!/usr/bin/env node' > "$NVM_DIR/versions/node/v24.2.0/bin/gemini"
/bin/chmod +x "$NVM_DIR/versions/node/v24.2.0/bin/gemini"
NPM
        /bin/chmod +x "$NVM_DIR/versions/node/v24.2.0/bin/node" "$NVM_DIR/versions/node/v24.2.0/bin/npm"
    else
        export PATH="$NVM_DIR/versions/node/v24.2.0/bin:$PATH"
    fi
}
NVM
''')
    curl = programs / 'curl'
    curl.write_text('''#!/bin/bash
printf 'download %s\\n' "$*" >> "$TRACE"
/bin/cat "$SPACR_FAKE_INSTALLER"
exit "${DOWNLOAD_STATUS:-0}"
''')
    curl.chmod(0o755)
    monkeypatch.setenv('PATH', str(programs))
    monkeypatch.setenv('TRACE', str(tmp_path / 'trace'))
    monkeypatch.setenv('SPACR_FAKE_INSTALLER', str(payload))
    monkeypatch.setenv('FAIL_STAGE', '')
    return home, Path(os.environ['TRACE'])


@pytest.mark.parametrize('name', ['codex', 'gemini'])
@pytest.mark.parametrize('platform', ['linux', 'darwin'])
def test_posix_plans_need_no_existing_node_or_npm(offline_bootstrap, name, platform):
    tool = SimpleNamespace(cli_name=name, install_methods=providers.install_methods_for(name, platform))
    plan = cli_install.plan_install(tool, platform)
    assert plan.automatic and 'npm' not in plan.method.needs
    assert plan.method is tool.install_methods[0]
    assert providers.install_hint_for(name, platform).startswith(plan.method.shown)
    if name == 'codex':
        assert plan.command == 'curl -fsSL https://chatgpt.com/codex/install.sh | sh'


@pytest.mark.parametrize('copied_hint', [False, True])
def test_gemini_bootstrap_executes_without_existing_npm_and_is_discovered_immediately(
        offline_bootstrap, monkeypatch, copied_hint):
    home, trace = offline_bootstrap
    monkeypatch.setenv('XDG_CONFIG_HOME', str(home / 'settings'))
    method = providers.install_methods_for('gemini', 'linux')[0]
    if copied_hint:
        result = subprocess.run(providers.install_hint_for('gemini', 'linux'), shell=True,
                                executable='/bin/bash', capture_output=True, text=True, timeout=10)
    else:
        result = subprocess.run(cli_install.launch_command(method, 'linux'),
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert trace.read_text().splitlines()[1:] == ['install 24', 'use 24', 'npm install -g @google/gemini-cli']
    assert not (home / '.bashrc').exists() and not (home / '.profile').exists()
    tool = SimpleNamespace(cli_name='gemini')
    location = cli_install.locate(tool, 'linux')
    assert location == str(home / '.nvm/versions/node/v24.2.0/bin/gemini')
    assert cli_install.put_on_path(location)
    assert providers._found_on_path('gemini')
    called = subprocess.run([location], capture_output=True, text=True, timeout=10)
    assert called.returncode == 0 and 'matching Node' in called.stdout


@pytest.mark.parametrize('stage', ['download', 'install', 'use', 'npm'])
def test_bootstrap_stops_after_download_node_or_package_failure(offline_bootstrap, monkeypatch, stage):
    home, trace = offline_bootstrap
    if stage == 'download':
        monkeypatch.setenv('DOWNLOAD_STATUS', '22')
    else:
        monkeypatch.setenv('FAIL_STAGE', stage)
    command = providers.install_methods_for('gemini', 'linux')[0]
    result = subprocess.run(cli_install.launch_command(command, 'linux'),
                            capture_output=True, timeout=10)
    assert result.returncode != 0
    assert not (home / '.nvm/versions/node/v24.2.0/bin/gemini').exists()
    lines = trace.read_text().splitlines()
    if stage == 'download':
        assert len(lines) == 1
    if stage in ('download', 'install', 'use'):
        assert not any(line.startswith('npm ') for line in lines)


def test_nvm_discovery_honors_custom_and_xdg_roots_with_numeric_version_order(tmp_path, monkeypatch):
    monkeypatch.setenv('NVM_DIR', str(tmp_path / 'custom nvm'))
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'xdg'))
    for root in (tmp_path / 'custom nvm', tmp_path / 'xdg/nvm'):
        for version in ('v9.0.0', 'v24.2.0', 'v24.10.0', 'invalid'):
            (root / 'versions/node' / version / 'bin').mkdir(parents=True)
    folders = cli_install._nvm_folders()
    assert [p.parent.name for p in folders] == ['v24.10.0', 'v24.2.0', 'v9.0.0'] * 2


def test_windows_codex_uses_native_powershell_and_discovers_its_user_folder(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    method = providers.install_methods_for('codex', 'win32')[0]
    assert method.needs == ('powershell.exe',)
    argv = cli_install.launch_command(method, 'win32')
    assert argv[1:4] == ['-NoProfile', '-NonInteractive', '-Command']
    assert argv[4] == "$ErrorActionPreference='Stop'; irm https://chatgpt.com/codex/install.ps1 | iex"
    monkeypatch.setattr(cli_install, '_npm_folder', lambda platform: '')
    assert str(tmp_path / 'local/Programs/OpenAI/Codex/bin') in cli_install.likely_folders('win32')
