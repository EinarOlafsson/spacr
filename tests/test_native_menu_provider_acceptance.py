"""Native acceptance jobs preserve the real artifact and exact command contracts."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_cocoa_menu_check_is_enabled_only_for_the_real_macos_artifact():
    workflow = yaml.safe_load((ROOT / '.github/workflows/frozen-installers.yml').read_text())
    steps = workflow['jobs']['smoke-frozen']['steps']
    macos = next(s for s in steps if s.get('if') == "matrix.platform == 'macos'")
    windows = next(s for s in steps if s.get('if') == "matrix.platform == 'windows'")
    assert 'SPACR_NATIVE_MENU_SMOKE=1' in macos['run']
    assert 'native_menu.mode' in macos['run'] and '= in-window-on-cocoa' in macos['run']
    assert 'native_menu.preferences_opened' in macos['run']
    assert 'native_menu.quit_observed' in macos['run']
    assert 'SPACR_NATIVE_MENU_SMOKE' not in windows['run']


def test_command_checks_require_real_smoke_receipts_and_both_windows_shells():
    workflow = yaml.safe_load((ROOT / '.github/workflows/frozen-installers.yml').read_text())
    job = workflow['jobs']['claude-native-windows']
    assert job['needs'] == 'smoke-frozen'
    assert set(job['strategy']['matrix']['shell']) == {'cmd', 'powershell51'}
    download = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/download-artifact@'))
    assert download['with']['name'] == 'legacy-smoke-windows'
    script = (ROOT / 'packaging/verify_claude_native_install.ps1').read_text()
    assert '$Hint = [string]$Proof.claude_install_hint' in script
    assert '$Proof.source_commit -ne $env:GITHUB_SHA' in script
    assert '$Proof.status -ne \'passed\'' in script
    assert "'5.1.*'" in script
    assert 'Get-Command node,npm,claude' in script
    assert 'Get-Command node,npm -ErrorAction SilentlyContinue' in script
    assert "[Environment]::GetFolderPath('UserProfile')" in script
    assert '$env:USERPROFILE =' not in script
    assert 'Fresh-host precondition failed' in script
    assert "'claude --version'" in script
    assert 'Invoke-Expression' not in script
    assert "[Environment]::SetEnvironmentVariable('Path',$BeforeUserPath,'User')" in script
    upload = next(s for s in job['steps'] if s.get('uses', '').startswith('actions/upload-artifact@'))
    assert upload['if'] == 'always()'
    assert '*.json' in upload['with']['path']
    assert 'profile' not in upload['with']['path']
