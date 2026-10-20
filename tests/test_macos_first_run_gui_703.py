"""Real portable launch/cancel contracts and native GUI packaging guards."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / 'packaging/online'


@pytest.fixture
def compiler():
    value = shutil.which('cc')
    if not value:
        pytest.skip('C compiler unavailable')
    return value


@pytest.mark.parametrize('identity_exit', [0, 1, 7])
def test_packaged_launcher_verifies_actual_runtime_before_launch_or_gui(tmp_path, compiler, identity_exit):
    home = tmp_path / 'user home'
    python = home / 'Library/Application Support/spaCR/venv/bin/python'
    python.parent.mkdir(parents=True)
    support = tmp_path / 'installed support'
    support.mkdir()
    journal = tmp_path / 'verification arguments'
    python.write_text('#!/bin/sh\nif [ "$2" != "-m" ]; then\n'
                      + 'printf "%s\\n" "$@" > '+shlex.quote(str(journal))+'\n'
                      + f'exit {identity_exit}\nfi\nprintf "APPLICATION\\n"\nprintf "%s\\n" "$@"\n', encoding="utf-8")
    python.chmod(0o755)
    gui = tmp_path / 'native gui'
    gui.write_text('#!/bin/sh\nprintf "GUI\\n%s\\n" "$@"\n', encoding="utf-8")
    gui.chmod(0o755)
    binary = tmp_path / 'launcher'
    subprocess.run([compiler, '-Wall', '-Wextra', '-Werror',
                    '-DSPACR_PACKAGE_VERSION="1.5.1.5"', f'-DSPACR_SUPPORT_DIR="{support}"',
                    f'-DSPACR_GUI_EXECUTABLE="{gui}"', str(ONLINE / 'macos_launcher.c'),
                    '-o', str(binary)], check=True)
    result = subprocess.run([str(binary), '--diagnostic'], env={**os.environ, 'HOME': str(home)},
                            check=True, capture_output=True, text=True, timeout=10)
    assert journal.read_text(encoding="utf-8").splitlines() == ['-I', str(support / 'verify_runtime.py'),
                                               'verify-runtime', '--python', str(python),
                                               '--version', '1.5.1.5', '--minimum']
    assert result.stdout.splitlines() == (['APPLICATION', '-I', '-m', 'spacr.qt', '--diagnostic']
                                           if identity_exit == 0 else ['GUI', '--diagnostic'])


def test_process_group_cancel_stops_installer_child_and_runs_rollback_trap(tmp_path, compiler):
    journal = tmp_path / 'cancellation receipt'
    script = tmp_path / 'installer'
    script.write_text('#!/bin/sh\ntrap '+shlex.quote('printf restored > '+shlex.quote(str(journal))+'; exit 143')
                      + ' TERM\nsleep 30 &\nprintf "READY\\n"\nwait\n', encoding="utf-8")
    program = tmp_path / 'driver.c'
    program.write_text('''#include "macos_gui_process.h"
#include <stdio.h>
#include <unistd.h>
extern char **environ;
int main(int argc, char **argv) {
    (void)argc;
    char *args[] = {"/bin/sh", argv[1], NULL};
    struct spacr_install_process child;
    int error = spacr_install_spawn(args, environ, &child);
    if (error) return error;
    FILE *output = fdopen(child.output, "r");
    char line[32];
    if (!fgets(line, sizeof(line), output)) return 2;
    fputs(line, stdout);
    if (spacr_install_cancel(child.pid)) return 3;
    int code = spacr_install_wait(child.pid);
    fclose(output);
    printf("exit=%d\\n", code);
    return code == 143 ? 0 : 4;
}
''', encoding="utf-8")
    binary = tmp_path / 'driver'
    subprocess.run([compiler, '-Wall', '-Wextra', '-Werror', '-I', str(ONLINE),
                    str(program), str(ONLINE / 'macos_gui_process.c'), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary), str(script)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['READY', 'exit=143']
    assert journal.read_text(encoding="utf-8") == 'restored'


@pytest.mark.parametrize('installer_exit', [0, 7])
@pytest.mark.posix_only
def test_user_helper_forwards_gui_choices_without_terminal_or_hidden_relaunch(tmp_path, installer_exit):
    support = tmp_path / 'support'
    support.mkdir()
    helper = support / 'install-for-user.sh'
    shutil.copyfile(ONLINE / 'install_for_macos_user.sh', helper)
    (support / 'installer_messages.sh').write_text('spacr_say() { printf "%s\\n" "$1"; }\n', encoding="utf-8")
    journal = tmp_path / 'actual options'
    installer = support / 'install-online.sh'
    installer.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > '+shlex.quote(str(journal))
                         + f'\nexit {installer_exit}\n', encoding="utf-8")
    installer.chmod(0o755)
    runtime = tmp_path / 'runtime'
    result = subprocess.run(['/bin/sh', str(helper), '--torch-backend', 'cpu', '--share-diagnostics'],
                            input='', capture_output=True, text=True, timeout=10,
                            env={**os.environ, 'SPACR_USER_INSTALL_ROOT': str(runtime), 'SPACR_NO_RELAUNCH': '1'})
    assert result.returncode == installer_exit
    assert journal.read_text(encoding="utf-8").splitlines() == ['--platform', 'macos', '--install-root', str(runtime),
        '--no-command-launcher', '--no-launch', '--consent-collected', '--torch-backend', 'cpu', '--share-diagnostics']
    assert not (runtime / '.installing').exists()
    assert ('ready_opening' in result.stdout) == (installer_exit == 0)


def test_gui_has_default_on_consent_bounded_log_and_cpu_choice():
    gui = (ONLINE / 'macos_installer.m').read_text(encoding="utf-8")
    launcher = (ONLINE / 'macos_launcher.c').read_text(encoding="utf-8")
    builder = (ONLINE / 'build_macos_online.sh').read_text(encoding="utf-8")
    assert 'Terminal' not in launcher and 'osascript' not in launcher
    assert 'index < 3 ? NSControlStateValueOn : NSControlStateValueOff' in gui
    assert '--share-diagnostics' in gui and '--report-issues' in gui and '--sign-in-now' not in gui and 'gui_account_setup' not in gui
    assert '@"auto" : @"cpu"' in gui
    assert 'self.child > 0' in gui and 'NSTerminateCancel' in gui
    assert '1024 * 1024' in gui and 'dispatch_sync(dispatch_get_main_queue()' in gui
    assert 'spacr_install_cancel(self.child)' in gui and 'spacr_install_wait(process.pid)' in gui
    assert 'removeObjectForKey:@"SPACR_USER_INSTALL_ROOT"' in gui
    assert 'SPACR_NO_RELAUNCH' in gui and 'createsNewApplicationInstance = YES' in gui
    assert '-framework Cocoa' in builder and '-fobjc-arc' in builder
    assert '"-DSPACR_PACKAGE_VERSION=\\"$VERSION\\""' in builder
    assert 'cp spacr/install_cleanup.py "$SUPPORT/verify_runtime.py"' in builder
    assert 'installer-i18n' in builder and 'install-spacr-gui' in builder


def test_all_window_keys_have_nine_locale_translations():
    spec = importlib.util.spec_from_file_location('installer_renderer_703', ROOT / 'packaging/i18n/render.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    catalogs = renderer.catalogs()
    keys = {key for key in catalogs['en'] if key.startswith('gui_')}
    window = (ONLINE / 'macos_installer.m').read_text(encoding='utf-8')
    named = set(re.findall(r'@"(gui_[a-z_]+)"', window)) - {'gui_category_'}
    categories = re.search(r'NSArray \*keys = @\[(@"caches".*?)\];', window).group(1)
    named |= {'gui_category_' + key for key in re.findall(r'@"([a-z]+)"', categories)}
    assert keys == named and len(keys) == 29
    for language, table in catalogs.items():
        assert keys <= table.keys()
        if language != 'en':
            assert all(table[key] != catalogs['en'][key] for key in keys)


def test_nested_native_gui_is_signed_before_bundle(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('macos_signing_703', ONLINE / 'macos_signing.py')
    signing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signing)
    helper = tmp_path / 'spaCR.app/Contents/MacOS/install-spacr-gui'
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b'private native placeholder')
    monkeypatch.setattr(signing, 'signing_config', lambda: None)
    calls = []
    monkeypatch.setattr(signing, '_run', lambda args: calls.append(args))
    signing.sign_app(helper.parents[2])
    assert calls == [['codesign', '--force', '--sign', '-', str(helper)],
                     ['codesign', '--force', '--sign', '-', str(helper.parents[2])],
                     ['codesign', '--verify', '--deep', '--strict', str(helper.parents[2])]]
