"""Exercise desktop installer control flow with stub dialogs and disposable children."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'packaging/online/install_spacr_unix.sh'


@pytest.fixture
def gui(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    os.mkfifo(tmp_path / 'dialog-ready')
    stub = bindir / 'zenity'
    stub.write_text('#!' + sys.executable + '\n' + '''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['DIALOG_RECORD'], 'a') as handle: handle.write(json.dumps(args)+'\\n')
if '--list' in args:
    if os.environ['GUI_CASE']=='consent-cancel':sys.exit(1)
    print(os.environ.get('GUI_CHOICES',''))
elif '--text-info' in args and not any(a.startswith('--filename=') for a in args):
    if os.environ['GUI_CASE']=='backend-cancel':
        sys.stdin.readline()
        sys.exit(1)
    if os.environ['GUI_CASE']=='success':
        with open(os.environ['DIALOG_READY'], 'w') as ready: ready.write('ready\\n')
    for line in sys.stdin: pass
    time.sleep(30)
elif '--progress' in args:
    time.sleep(30)
''')
    stub.chmod(0o755)
    backend = tmp_path / 'backend.sh'
    backend.write_text('''#!/bin/bash
set -Eeuo pipefail
printf '%s\\n' "$@" > "$BACKEND_ARGS"
if [[ "$GUI_CASE" == backend-cancel ]]; then
    trap 'sleep .2; printf restored > "$ROLLBACK_PROOF"; exit 143' TERM
fi
printf 'backend started\\n'
if [[ "$GUI_CASE" == success ]]; then IFS= read -r ready < "$DIALOG_READY"; fi
case "$GUI_CASE" in
    failure) exit 17 ;;
    backend-cancel)
        sleep 30 &
        wait $!
        ;;
esac
printf 'backend complete\\n'
''')
    source = SCRIPT.read_text().split('# @SPACR_LINUX_GUI_BEGIN@',1)[1].split('# @SPACR_LINUX_GUI_END@',1)[0]
    harness = tmp_path / 'harness.sh'
    harness.write_text('''#!/bin/bash
set -Eeuo pipefail
spacr_say() { printf '%s' "$1"; }
''' + source + '\nINSTALL_ROOT=' + shlex.quote(str(tmp_path / 'installation')) + '''
NO_LAUNCH=1
if [[ "$GUI_CASE" == missing ]]; then
    command() { if [[ "${2:-}" == zenity ]]; then return 1; fi; builtin command "$@"; }
fi
spacr_linux_gui_install ''' + shlex.quote(str(backend)) + ''' --package-spec 'fake wheel.whl' --gui --share-diagnostics --report-issues --sign-in-now
''')
    environment = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                       XDG_CACHE_HOME=str(tmp_path / 'cache'), DIALOG_RECORD=str(tmp_path / 'dialogs.jsonl'),
                       BACKEND_ARGS=str(tmp_path / 'backend-args'), DIALOG_READY=str(tmp_path / 'dialog-ready'), ROLLBACK_PROOF=str(tmp_path / 'restored'))
    return tmp_path, harness, environment


@pytest.mark.parametrize('choices', ['', 'diagnostics|accounts'])
def test_gui_choices_default_off_and_only_current_selection_reaches_headless_child(gui, choices):
    tmp, harness, environment = gui
    result = subprocess.run(['bash', str(harness)], env=dict(environment, GUI_CASE='success', GUI_CHOICES=choices),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = (tmp / 'backend-args').read_text().splitlines()
    assert arguments[:2] == ['--package-spec', 'fake wheel.whl']
    assert '--gui' not in arguments
    assert '--consent-collected' in arguments and '--skip-system-deps' in arguments and '--no-launch' in arguments
    assert ('--share-diagnostics' in arguments) == ('diagnostics' in choices)
    assert '--report-issues' not in arguments
    assert ('--sign-in-now' in arguments) == ('accounts' in choices)
    dialogs = [json.loads(line) for line in (tmp / 'dialogs.jsonl').read_text().splitlines()]
    checklist, = [args for args in dialogs if '--checklist' in args]
    assert checklist.count('FALSE') == 3
    assert any('--text-info' in args for args in dialogs)
    logs = list((tmp / 'cache/spacr/installer').glob('gui.*/install.log'))
    assert len(logs) == 1 and 'backend complete' in logs[0].read_text()


def test_backend_failure_keeps_its_actual_exit_and_log_without_retry(gui):
    tmp, harness, environment = gui
    result = subprocess.run(['bash', str(harness)], env=dict(environment, GUI_CASE='failure'),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 17, result.stderr
    dialogs = [json.loads(line) for line in (tmp / 'dialogs.jsonl').read_text().splitlines()]
    assert any('--error' in args and '--text=gui_failed' in args for args in dialogs)
    assert any(any(arg.startswith('--filename=') for arg in args) for args in dialogs)
    assert (tmp / 'backend-args').exists()


def test_closing_progress_signals_only_own_group_and_waits_for_rollback(gui):
    tmp, harness, environment = gui
    result = subprocess.run(['bash', str(harness)], env=dict(environment, GUI_CASE='backend-cancel'),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 130, result.stderr
    assert (tmp / 'restored').read_text() == 'restored'
    assert not list((tmp / 'cache').rglob('*.fifo'))


def test_consent_cancellation_starts_no_backend_and_creates_no_runtime_folder(gui):
    tmp, harness, environment = gui
    result = subprocess.run(['bash', str(harness)], env=dict(environment, GUI_CASE='consent-cancel'),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 130
    assert not (tmp / 'backend-args').exists()
    assert not (tmp / 'cache').exists()


def test_missing_dialog_dependency_fails_before_installation_mutation(gui):
    tmp, harness, environment = gui
    result = subprocess.run(['bash', str(harness)], env=dict(environment, GUI_CASE='missing'),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 3
    assert not (tmp / 'backend-args').exists() and not (tmp / 'cache').exists()


def test_gui_reuses_existing_installer_locale_sources_and_preserves_explicit_headless_mode():
    source = SCRIPT.read_text()
    gui_source = source.split('# @SPACR_LINUX_GUI_BEGIN@',1)[1].split('# @SPACR_LINUX_GUI_END@',1)[0]
    captions = set(re.findall(r'spacr_say (\w+)', gui_source))
    for path in (ROOT / 'packaging/i18n').glob('*.json'):
        assert captions <= json.loads(path.read_text()).keys()
    route = source.split('# @SPACR_LINUX_GUI_END@',1)[1].split('require_command()',1)[0]
    assert 'SPACR_INSTALL_GUI_CHILD' in route and 'CONSENT_COLLECTED' in route
    assert 'NO_LAUNCH' in route and 'DISPLAY' in route and 'WAYLAND_DISPLAY' in route
    assert source.index('if [[ "$DRY_RUN" == "1" ]]') < source.index('# @SPACR_LINUX_GUI_BEGIN@')


@pytest.mark.parametrize('mode,expected', [
    ('desktop-x11', 'GUI'), ('desktop-wayland', 'GUI'), ('headless', 'CLI'),
    ('consent-collected', 'CLI'), ('no-launch', 'CLI'), ('child', 'CLI'),
    ('explicit-gui', 'GUI'),
])
def test_actual_desktop_routing_preserves_explicit_headless_and_child_invocations(tmp_path, mode, expected):
    source = SCRIPT.read_text()
    route = source.split('# @SPACR_LINUX_GUI_END@',1)[1].split('require_command()',1)[0]
    values = {'PLATFORM': 'linux', 'SPACR_INSTALL_GUI_CHILD': '0', 'GUI_REQUEST': '0',
              'CONSENT_COLLECTED': '0', 'NO_LAUNCH': '0', 'DISPLAY': '', 'WAYLAND_DISPLAY': '',
              'SPACR_SOURCE_DIR': str(tmp_path)}
    if mode == 'desktop-x11': values['DISPLAY'] = ':fixture'
    if mode == 'desktop-wayland': values['WAYLAND_DISPLAY'] = 'fixture'
    if mode in {'consent-collected', 'no-launch', 'child'}: values['DISPLAY'] = ':fixture'
    if mode == 'consent-collected': values['CONSENT_COLLECTED'] = '1'
    if mode == 'no-launch': values['NO_LAUNCH'] = '1'
    if mode == 'child': values['SPACR_INSTALL_GUI_CHILD'] = '1'
    if mode == 'explicit-gui': values['GUI_REQUEST'] = '1'
    harness = tmp_path / 'route.sh'
    harness.write_text('set -Eeuo pipefail\nSPACR_ORIGINAL_ARGS=()\n' +
                       '\n'.join(key + '=' + shlex.quote(value) for key,value in values.items()) +
                       '\nspacr_linux_gui_install() { printf GUI; }\n' + route + '\nprintf CLI\n')
    result = subprocess.run(['bash', str(harness)], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and result.stdout == expected, result.stderr
