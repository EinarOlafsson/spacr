"""Native historical-updater witness; run only with the installed private Python."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('state', 'gui', 'verify'))
parser.add_argument('--version')
parser.add_argument('--target-version')
parser.add_argument('--expect', choices=('missing-pip', 'success'))
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--evidence', type=Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

def sha(path):
    """Return the SHA256 of one installed source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(payload):
    """Write the current stage receipt to the explicit output path."""
    args.output.write_text(json.dumps(payload, indent=2) + '\n')

def state():
    """Measure version, source provenance, missing pip, and actual torch backend."""
    assert sys.flags.isolated, 'Use installed Python -I: no checkout/PYTHONPATH fallback'
    assert sys.prefix != sys.base_prefix, 'Expected the historical installer private venv'
    assert not os.environ.get('UV_TORCH_BACKEND'), 'Do not conceal updater backend behavior'
    distribution = md.distribution('spacr')
    version = distribution.version
    if args.version:
        assert version == args.version, (version, args.version)
    prefix = Path(sys.prefix).resolve()
    origins = {}
    for relative in ('spacr/updater.py', 'spacr/qt/app.py', 'spacr/qt/__init__.py'):
        path = Path(distribution.locate_file(relative)).resolve()
        assert path.is_relative_to(prefix), ('Package source outside installed venv', path)
        origins[relative] = {'path': str(path), 'sha256': sha(path)}
    known = json.loads(Path(__file__).with_name('historical_upgrade_sources.json').read_text())
    if version in known:
        for row in known[version]['files']:
            if row['path'] in origins:
                assert origins[row['path']]['sha256'] == row['sha256'], ('Historical wheel source drift', row['path'])
    assert importlib.util.find_spec('pip') is None, 'Missing-pip acceptance must not seed pip'
    import spacr
    import torch
    assert Path(spacr.__file__).resolve().is_relative_to(prefix)
    assert Path(torch.__file__).resolve().is_relative_to(prefix)
    return dict(version=version, prefix=str(prefix), executable=sys.executable,
        installed_sources=origins, pip_present=False, torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        torch_origin=str(Path(torch.__file__).resolve()), platform=platform.platform(),
        transport_scope='Only release checks/package downloads; no credential entry or telemetry action is driven.')

if args.mode == 'state':
    write(dict(passed=True, state=state()))
    raise SystemExit(0)

if args.mode == 'verify':
    evidence = args.evidence
    records = {name: json.loads((evidence / f'{name}.json').read_text()) for name in
        ('before', 'broken-gui', 'repaired', 'fixed-gui', 'after')}
    errors = []
    if not all(record.get('passed') is True for record in records.values()):
        errors.append('One or more stages failed')
    states = [records[name]['state'] for name in ('before', 'repaired', 'after')]
    known = json.loads(Path(__file__).with_name('historical_upgrade_sources.json').read_text())
    source_names = ('spacr/updater.py', 'spacr/qt/app.py', 'spacr/qt/__init__.py')
    for name, row in zip(('before', 'repaired', 'after'), states):
        prefix = Path(row['prefix']).resolve()
        if not Path(row['executable']).parent.resolve().is_relative_to(prefix):
            errors.append(f'{name} executable location escapes its private installation')
        if not Path(row['torch_origin']).resolve().is_relative_to(prefix):
            errors.append(f'{name} torch origin escapes its private installation')
        bindings = row['installed_sources']
        if set(bindings) != set(source_names):
            errors.append(f'{name} lacks the exact installed source provenance')
        for relative, binding in bindings.items():
            if not Path(binding['path']).resolve().is_relative_to(prefix):
                errors.append(f'{name} source origin escapes its private installation')
            if not re.fullmatch(r'[a-f0-9]{64}', binding['sha256']):
                errors.append(f'{name} source digest is malformed')
        if row['version'] in known:
            for source in known[row['version']]['files']:
                if source['path'] in source_names and bindings.get(source['path'], {}).get('sha256') != source['sha256']:
                    errors.append(f'{name} source differs from the exact historical PyPI wheel')
    if [row['version'] for row in states] != ['1.5.0.1', '1.5.0.5', args.target_version]:
        errors.append('Version transition does not match historical repair then real update')
    if len({row['prefix'] for row in states}) != 1:
        errors.append('The private installation environment changed')
    for name, version, expected_state in (('broken-gui', '1.5.0.1', states[0]), ('fixed-gui', '1.5.0.5', states[1])):
        gui = records[name]
        if gui['state']['prefix'] != states[0]['prefix'] or gui['state']['version'] != version:
            errors.append(f'{name} did not use the same version-bound private installation')
        if gui['target_version'] != args.target_version:
            errors.append(f'{name} did not observe the same public upgrade target')
        if gui['state'] != expected_state:
            errors.append(f'{name} GUI provenance differs from its independently measured state')
        if not gui.get('action_text', '').replace('&', '').startswith('Check for updates'):
            errors.append(f'{name} lacks the installed Help action witness')
        command = [expected_state['executable'], '-m', 'pip', 'install', '--upgrade', 'spacr']
        if name == 'fixed-gui':
            executable = Path(expected_state['executable'])
            uv = Path(expected_state['prefix']).parent / 'bootstrap' / ('uv.exe' if executable.suffix.lower() == '.exe' else 'uv')
            command = [str(uv), 'pip', 'install', '--upgrade', '--python', expected_state['executable'], 'spacr']
        if gui.get('actual_subprocess_commands') != [command]:
            errors.append(f'{name} lacks the actual expected updater subprocess')
        terminal = gui.get('terminal_dialog', '')
        if ((name == 'broken-gui' and not re.search(r'pip returned exit code [1-9]\d*', terminal))
                or (name == 'fixed-gui' and 'Upgrade finished. Restart spaCR to use it.' not in terminal)):
            errors.append(f'{name} lacks the required actual terminal dialog')
    if any(row['pip_present'] for row in states):
        errors.append('pip was installed, concealing the bug')
    if any(row['torch_cuda_build'] is not None or row['cuda_available'] for row in states):
        errors.append('CPU backend was not preserved by the historical repair/update route')
    failure_log = (evidence / 'broken-gui.log').read_text(errors='replace')
    if 'No module named pip' not in failure_log:
        errors.append('Actual old child-process missing-pip failure is absent')
    write(dict(passed=not errors, errors=errors, stages=records,
        scope='Released online installers and installed in-app updater only; not frozen self-update or unreleased checkout behavior.'))
    raise SystemExit(bool(errors))

assert args.expect and args.target_version
before = state()
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
import spacr.updater as updater
import spacr.qt.app as installed_app

assert sha(updater.__file__) == before['installed_sources']['spacr/updater.py']['sha256']
assert sha(installed_app.__file__) == before['installed_sources']['spacr/qt/app.py']['sha256']
commands = []
def observe(event, values):
    """Observe actual upgrade child-process arguments without replacing execution."""
    if event == 'subprocess.Popen':
        argv = values[1]
        if isinstance(argv, (list, tuple)):
            command = [os.fsdecode(value) for value in argv]
            if 'install' in command and '--upgrade' in command and 'spacr' in command:
                commands.append(command)
sys.addaudithook(observe)

started = time.monotonic()
question_seen = False
terminal = None
seen = set()
held_dialogs = []
triggered = False
receipt = dict(passed=False, state=before, target_version=args.target_version,
    expect=args.expect, dialogs=[], actual_subprocess_commands=commands,
    ui_route='Installed MainWindow Help QAction.trigger plus real QMessageBox button clicks; updater and transport functions are not replaced.')
failure_lock = threading.Lock()

def fail(message):
    """Persist failure and reap only this witness's descendants, even if Qt stalls."""
    if not failure_lock.acquire(blocking=False):
        return
    receipt.update(error=message, elapsed_seconds=time.monotonic()-started)
    write(receipt)
    print(message, file=sys.stderr, flush=True)
    try:
        import psutil
        children = psutil.Process().children(recursive=True)
        receipt['failure_cleanup_child_pids'] = [child.pid for child in children]
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=2)
        receipt['failure_cleanup_remaining_pids'] = [child.pid for child in alive]
    except Exception as cleanup_error:
        receipt['failure_cleanup_error'] = str(cleanup_error)
    write(receipt)
    os._exit(1)

def timeout():
    """Enforce the GUI deadline independently of blocked GUI callbacks."""
    fail('Installed GUI update exceeded its 25-minute deadline')

watchdog = threading.Timer(1500, timeout)
watchdog.daemon = True
watchdog.start()
try:
    app = QApplication([])
    app.setApplicationName('spaCR')
    app.setOrganizationName('Olafsson Lab')
    expected_platform = {'win32': 'windows', 'darwin': 'cocoa'}.get(sys.platform, 'xcb')
    receipt['qt_platform'] = app.platformName()
    assert app.platformName() == expected_platform, ('Expected native Qt platform', app.platformName())
    window = installed_app.MainWindow()
    window.show()
except Exception as error:
    fail(f'{type(error).__name__}: {error}')

def tick():
    """Drive the installed Help action and real modal buttons; record their outcome."""
    global question_seen, terminal, triggered
    try:
        if time.monotonic() - started > 1500:
            fail('Installed GUI update exceeded its 25-minute deadline')
        if terminal is not None:
            worker = getattr(window, '_update_worker', None)
            try:
                if worker is not None and worker.isRunning():
                    return
            except RuntimeError:
                pass
            assert question_seen and len(commands) == 1, ('Missing real update action/process witness', commands)
            command = commands[0]
            if args.expect == 'missing-pip':
                assert command == [sys.executable, '-m', 'pip', 'install', '--upgrade', 'spacr'], command
                assert re.search(r'pip returned exit code [1-9]\d*', terminal)
            else:
                uv = Path(sys.prefix).parent / 'bootstrap' / ('uv.exe' if os.name == 'nt' else 'uv')
                assert Path(command[0]).resolve() == uv.resolve(), command
                assert command[1:] == ['pip', 'install', '--upgrade', '--python', sys.executable, 'spacr'], command
                assert 'Upgrade finished. Restart spaCR to use it.' in terminal
            timer.stop()
            receipt.update(passed=True, terminal_dialog=terminal, elapsed_seconds=time.monotonic()-started)
            write(receipt)
            window.close()
            app.quit()
            return
        if not triggered:
            actions = [action for action in window.findChildren(QAction)
                       if action.text().replace('&', '').startswith('Check for updates')]
            assert len(actions) == 1 and actions[0].isEnabled(), 'Installed Help updater action unavailable'
            triggered = True
            receipt['action_text'] = actions[0].text()
            actions[0].trigger()
        for dialog in QApplication.topLevelWidgets():
            if not isinstance(dialog, QMessageBox) or not dialog.isVisible() or id(dialog) in seen:
                continue
            text = dialog.text()
            seen.add(id(dialog)); held_dialogs.append(dialog)
            title = dialog.windowTitle()
            receipt['dialogs'].append(dict(title=title, text=text))
            untitled_cocoa = sys.platform == 'darwin' and not title
            if title == 'Update available' or (untitled_cocoa and text.startswith('A new version is available.\n\n')):
                assert f"Installed: {args.version}" in text
                assert f'Latest:    {args.target_version}' in text
                question_seen = True
                assert dialog.grab().save(str(args.output.with_suffix('.prompt.png'))), 'Prompt screenshot was not saved'
                button = dialog.button(QMessageBox.StandardButton.Yes)
                assert button is not None and button.isEnabled()
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            elif title == 'Updates' or (untitled_cocoa and (
                    text.startswith('pip returned exit code ')
                    or text == 'Upgrade finished. Restart spaCR to use it.')):
                terminal = text
                assert dialog.grab().save(str(args.output.with_suffix('.result.png'))), 'Result screenshot was not saved'
                button = dialog.button(QMessageBox.StandardButton.Ok)
                assert button is not None
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
            else:
                fail('Unexpected dialog during real update: ' + dialog.windowTitle() + ': ' + text)
    except Exception as error:
        fail(f'{type(error).__name__}: {error}')

timer = QTimer()
timer.timeout.connect(tick)
timer.start(100)
try:
    exit_code = app.exec()
    if not receipt.get('passed'):
        fail('Installed GUI exited before the updater reached its required terminal state')
finally:
    watchdog.cancel()
raise SystemExit(exit_code)
