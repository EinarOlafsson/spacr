"""The legacy installer jobs must validate installed artifacts on fresh runners."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_smoke_jobs_cannot_borrow_the_build_checkout():
    yaml = pytest.importorskip('yaml')
    workflow = yaml.safe_load((ROOT / '.github/workflows/frozen-installers.yml').read_text())
    events = workflow.get('on', workflow.get(True))
    assert 'push' in events and events['push']['branches'] == ['nightly']
    assert 'workflow_dispatch' in events
    assert workflow['permissions'] == {'contents': 'read'}
    for key in ('smoke-frozen', 'smoke-debian'):
        steps = workflow['jobs'][key]['steps']
        actions = [row.get('uses', '') for row in steps]
        assert not any('checkout' in action or 'setup-python' in action for action in actions)
        assert any('download-artifact' in action for action in actions)
        commands = '\n'.join(row.get('run', '') for row in steps)
        assert 'pip install' not in commands
        assert 'empty-cwd' in commands
        assert 'smoke.json' in commands
    assert 'continue-on-error' not in workflow['jobs']['smoke-debian']
    assert workflow['jobs']['build-debian']['container'] == 'ubuntu:22.04'
    assert set(workflow['jobs']['smoke-debian']['strategy']['matrix']['container']) == {
        'ubuntu:22.04', 'debian:12'}


def test_windows_nsis_template_keeps_runtime_dollars_and_requires_actual_executable():
    source = (ROOT / 'packaging/build_windows.ps1').read_text()
    template = source.split("$Template = @'\n", 1)[1].split("\n'@", 1)[0]
    assert 'InstallDir "$PROGRAMFILES64\\spaCR"' in template
    assert 'SetOutPath "$INSTDIR"' in template
    assert 'CreateShortcut "$SMPROGRAMS\\spaCR.lnk" "$INSTDIR\\spacr.exe"' in template
    assert '-RequireInstaller' in (ROOT / '.github/workflows/frozen-installers.yml').read_text()
    assert 'if ($LASTEXITCODE -ne 0)' in source
    assert 'Join-Path $Source "spacr.exe"' in source
    assert 'RMDir /r "$INSTDIR"' not in template
    assert '@DELETE_FILES@' in template and '@DELETE_DIRECTORIES@' in template


def test_debian_xcb_shape_dependency_exists_on_builder_and_fresh_host():
    """Both clean targets failed to load Qt because libxcb-shape was absent."""
    source = (ROOT / 'packaging/build_debian.sh').read_text()
    builder = re.search(r'needed=\((.*?)\)\nmissing=', source, re.S).group(1).split()
    depends = re.search(r'^Depends: (.+)$', source, re.M).group(1)
    runtime = {value.strip().split()[0] for value in depends.split(',')}
    assert 'libxcb-shape0' in builder
    assert 'libxcb-shape0' in runtime
