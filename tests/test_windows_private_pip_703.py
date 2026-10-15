"""Windows private-pip staging uses disposable files and explicit native-platform seams."""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile

import pytest

from spacr import install_cleanup as cleanup


USER = 'S-1-5-21-123-456-789-1001'


def acl():
    return {'owner': USER, 'user': USER, 'has_dacl': True,
            'rules': [{'sid': USER, 'type': 0, 'rights': 0x1f01ff, 'propagation': 0},
                      {'sid': 'S-1-5-18', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0}]}


def snapshot(prefix):
    return {str(path.relative_to(prefix)): path.read_bytes()
            for path in prefix.rglob('*') if path.is_file()}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    prefix = home / 'private spaCR' / 'venv'
    python = prefix / 'Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'MZ disposable modeled Windows interpreter')
    python.chmod(0o700)
    python.with_name('pythonw.exe').write_bytes(python.read_bytes())
    base = home / 'base Python'
    base.mkdir()
    base_python = base / 'python.exe'
    base_python.write_bytes(b'MZ modeled outside base interpreter')
    base_python.chmod(0o700)
    (base / 'Lib').mkdir()
    (prefix / 'pyvenv.cfg').write_text('home = ' + str(base) + '\ninclude-system-site-packages = false\n')
    package = prefix / 'Lib/site-packages/spacr'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('__version__ = "1.5.1.4"\n')
    user_file = prefix / 'unknown-user-files/keep.bin'
    user_file.parent.mkdir()
    user_file.write_bytes(b'private user data stays exact\x00\xff')
    record = cleanup.InstallRecord(kind='environment', layout='venv', platform='windows',
                                   root=str(prefix), python=str(python), version='1.5.1.4', running=True)
    machine = cleanup._Machine(platform='windows', environ={'USERPROFILE': str(home)},
                               fs_root=str(tmp_path), running_prefix=str(prefix), executable=str(python))
    monkeypatch.setattr(cleanup.sys, 'base_prefix', str(base))
    monkeypatch.setattr(cleanup.sys, '_base_executable', str(base_python), raising=False)
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: acl())
    monkeypatch.setattr(cleanup, '_retiring_runtime_identity', lambda _path: {'verified': True, 'version': '1.5.1.4'})
    return types.SimpleNamespace(prefix=prefix, python=python, base=base, base_python=base_python,
                                 record=record, machine=machine, package=package, helper=tmp_path / 'outside helper')


def plan_for(runtime):
    command, relaunch = cleanup._windows_pip_commands(runtime.record, '1.5.1.5', str(runtime.helper), runtime.machine)
    return {'records': [cleanup.asdict(runtime.record)], 'version': '1.5.1.5', 'workdir': str(runtime.helper),
            'install': command, 'relaunch': relaunch, 'fetch': None}


def test_dedicated_windows_pip_plan_can_prepare_without_touching_the_active_runtime(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    probes = []

    def probe(argv, **_kwargs):
        probes.append(argv)
        assert '-c' in argv, 'Preparation must not run a package operation'
        assert argv[0] == str(runtime.base_python)
        return 0, json.dumps({'python': argv[0], 'prefix': str(runtime.base),
                              'base_prefix': str(runtime.base), 'stdlib': str(runtime.base / 'Lib'), 'os_name': 'nt'})

    monkeypatch.setattr(cleanup, '_run', probe)
    spawned = []
    plan = cleanup.start_update_helper([runtime.record], '1.5.1.5', pid=4242,
                                      workdir=str(runtime.helper), system=runtime.machine,
                                      spawn=lambda *args: spawned.append(args))
    assert plan['error'] is None, plan['error']
    assert plan['adapter'] == 'pip-private-windows-v1'
    assert plan['command'] and len(spawned) == 1 and len(probes) == 1
    assert not Path(plan['command'][0]).resolve().is_relative_to(runtime.prefix.resolve())
    assert plan['command'][1:3] == ['-I', '-B']
    assert plan['fetch'] is None
    assert not (runtime.helper / 'approved.json').exists()
    assert snapshot(runtime.prefix) == before


@pytest.mark.parametrize('defect', ['owner', 'null_dacl', 'foreign_write', 'missing_rules', 'invalid_rule', 'propagation'])
def test_unproven_windows_acl_refuses_before_mutation(runtime, monkeypatch, defect):
    evidence = acl()
    if defect == 'owner':
        evidence['owner'] = 'S-1-5-18'
    elif defect == 'null_dacl':
        evidence['has_dacl'] = False
    elif defect == 'foreign_write':
        evidence['rules'].append({'sid': 'S-1-1-0', 'type': 0, 'rights': 2, 'propagation': 0})
    elif defect == 'missing_rules':
        evidence['rules'] = []
    elif defect == 'invalid_rule':
        evidence['rules'] = [{'sid': USER, 'type': 0, 'rights': 'all', 'propagation': 0}]
    else:
        evidence['rules'][0]['propagation'] = 4
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: evidence)
    before = snapshot(runtime.prefix)
    with pytest.raises(ValueError):
        plan_for(runtime)
    assert snapshot(runtime.prefix) == before


@pytest.mark.parametrize('rule', [
    {'sid': 'S-1-1-0', 'type': 0, 'rights': 0x120089, 'propagation': 0},
    {'sid': 'S-1-5-32-544', 'type': 0, 'rights': 0x1f01ff, 'propagation': 0},
    {'sid': 'S-1-1-0', 'type': 1, 'rights': 0x1f01ff, 'propagation': 0},
])
def test_non_writable_foreign_or_trusted_acl_rules_are_supported(runtime, monkeypatch, rule):
    evidence = acl()
    evidence['rules'].append(rule)
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda _path: evidence)
    assert plan_for(runtime)['install'][0] == str(runtime.python)


@pytest.mark.parametrize('defect', ['conda', 'system_site', 'editable', 'link', 'inside_helper', 'wrong_running', 'missing_pythonw'])
def test_unsupported_environment_is_not_replaced(runtime, defect):
    before = snapshot(runtime.prefix)
    if defect == 'conda':
        (runtime.prefix / 'conda-meta').mkdir()
    elif defect == 'system_site':
        (runtime.prefix / 'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    elif defect == 'editable':
        (runtime.prefix / 'editable.egg-link').write_text('/outside/source')
    elif defect == 'link':
        (runtime.prefix / 'outside').symlink_to(runtime.base, target_is_directory=True)
    elif defect == 'inside_helper':
        runtime.helper = runtime.prefix / 'updater'
    elif defect == 'wrong_running':
        runtime.machine.running_prefix = str(runtime.base)
    else:
        runtime.machine.executable = str(runtime.python.with_name('missing-pythonw.exe'))
    modified = snapshot(runtime.prefix)
    with pytest.raises(ValueError):
        plan_for(runtime)
    assert snapshot(runtime.prefix) == modified
    assert all(modified.get(key) == value for key, value in before.items() if key != 'pyvenv.cfg')


@pytest.mark.parametrize('field,value', [('os_name', 'posix'), ('prefix', '/foreign'), ('base_prefix', '/foreign'), ('stdlib', 'INSIDE')])
def test_external_helper_identity_must_be_verified(runtime, monkeypatch, field, value):
    runtime.helper.mkdir()
    values = {'python': str(runtime.base_python), 'prefix': str(runtime.base), 'base_prefix': str(runtime.base),
              'stdlib': str(runtime.base / 'Lib'), 'os_name': 'nt'}
    values[field] = str(runtime.prefix / 'Lib') if value == 'INSIDE' else value
    monkeypatch.setattr(cleanup, '_run', lambda *a, **kw: (0, json.dumps(values)))
    with pytest.raises(ValueError):
        cleanup._windows_pip_helper(runtime.record, str(runtime.helper / 'helper.py'), str(runtime.helper / 'plan.json'), runtime.machine)


class PackageRunner:
    def __init__(self, runtime, failure=None):
        self.runtime = runtime
        self.failure = failure
        self.commands = []

    def __call__(self, argv):
        self.commands.append(argv)
        python = Path(argv[0])
        target = python.parent.parent
        if argv[1:5] == ['-I', '-m', 'pip', 'install']:
            assert target != self.runtime.prefix
            assert 'staged' == target.name
            if self.failure == 'install':
                return 1, 'modeled package error'
            (target / 'Lib/site-packages/spacr/__init__.py').write_text('__version__ = "1.5.1.5"\n')
            return 0, 'staged package installed'
        if argv[1:] == ['-I', '-m', 'pip', 'check']:
            return (1, 'modeled dependency error') if self.failure == 'check' else (0, 'dependencies verified')
        if argv[-1] == 'import spacr.qt':
            bad = self.failure == ('final_import' if target == self.runtime.prefix else 'import')
            return (1, 'modeled import error') if bad else (0, '')
        if argv[-1] == 'import pip':
            return 0, ''
        assert '-c' in argv
        expected = json.loads(argv[-1])
        version = '1.5.1.5'
        if self.failure == 'identity' or (self.failure == 'final_identity' and target == self.runtime.prefix):
            return 1, '{}'
        if self.failure == 'wrong_version':
            version = '1.5.1.6'
        return 0, json.dumps({'verified': True, 'version': version, 'requested_version': expected['version']})


def run_transaction(runtime, monkeypatch, *, failure=None, activate=None):
    runner = PackageRunner(runtime, failure)
    monkeypatch.setattr(cleanup, '_windows_pip_scripts', lambda staged, prefix, run: {'verified': True, 'scripts': []})
    plan = plan_for(runtime)
    return cleanup._windows_pip_transaction(runtime.record, plan['version'], plan['install'],
                                            machine=runtime.machine, runner=runner, activate=activate), runner


def test_stage_only_update_retains_original_and_user_data(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    result, runner = run_transaction(runtime, monkeypatch)
    assert result[0] == '1.5.1.5'
    root = next(runtime.prefix.parent.glob('.spacr-windows-update-*'))
    assert snapshot(root / 'previous') == before
    assert (runtime.prefix / 'unknown-user-files/keep.bin').read_bytes() == before['unknown-user-files/keep.bin']
    journal = json.loads((root / 'transaction.json').read_text())
    assert journal['state'] == 'committed'
    assert not (runtime.prefix.parent / '.venv.spacr-update-lock').exists()
    assert [cmd for cmd in runner.commands if cmd[1:5] == ['-I', '-m', 'pip', 'install']][0][0] != str(runtime.python)


@pytest.mark.parametrize('failure', ['install', 'check', 'import', 'identity', 'wrong_version', 'final_identity', 'final_import', 'activation'])
def test_failure_preserves_or_restores_exact_prior_runtime(runtime, monkeypatch, failure):
    before = snapshot(runtime.prefix)

    def activate(_version):
        raise OSError('modeled relaunch failure')

    with pytest.raises((RuntimeError, ValueError, OSError)):
        run_transaction(runtime, monkeypatch, failure=failure, activate=activate if failure == 'activation' else None)
    assert snapshot(runtime.prefix) == before
    assert not (runtime.prefix.parent / '.venv.spacr-update-lock').exists()
    root = next(runtime.prefix.parent.glob('.spacr-windows-update-*'))
    assert json.loads((root / 'transaction.json').read_text())['state'] in {'staging_failed', 'rolled_back'}


def test_failed_second_rename_rolls_back_without_deleting_old_runtime(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    rename = cleanup.os.rename
    calls = []

    def fail_stage(source, target):
        calls.append((source, target))
        if Path(source).name == 'staged' and Path(target) == runtime.prefix:
            raise PermissionError('modeled Windows file lock')
        return rename(source, target)

    monkeypatch.setattr(cleanup.os, 'rename', fail_stage)
    with pytest.raises(PermissionError):
        run_transaction(runtime, monkeypatch)
    assert len(calls) == 3
    assert snapshot(runtime.prefix) == before


@pytest.mark.parametrize('when', ['after_retirement', 'activated_modified', 'backup_modified'])
def test_foreign_state_is_retained_for_review_and_never_overwritten(runtime, monkeypatch, when):
    before = snapshot(runtime.prefix)
    journal = cleanup._runtime_recovery_journal

    def interrupt(root, state):
        journal(root, state)
        if state['state'] == 'retired' and when == 'after_retirement':
            runtime.prefix.mkdir()
            (runtime.prefix / 'foreign.bin').write_bytes(b'foreign replacement')
        if state['state'] == 'verifying' and when == 'activated_modified':
            (runtime.prefix / 'foreign.bin').write_bytes(b'foreign replacement')
        if state['state'] == 'verifying' and when == 'backup_modified':
            (Path(state['rollback_runtime']) / 'foreign.bin').write_bytes(b'foreign backup modification')

    monkeypatch.setattr(cleanup, '_runtime_recovery_journal', interrupt)
    with pytest.raises(RuntimeError, match='windows_runtime_rollback_requires_review'):
        run_transaction(runtime, monkeypatch)
    root = next(runtime.prefix.parent.glob('.spacr-windows-update-*'))
    assert (runtime.prefix.parent / '.venv.spacr-update-lock').is_dir()
    assert all(snapshot(root / 'previous').get(key) == value for key, value in before.items())
    assert (runtime.prefix / 'foreign.bin').exists() if when != 'backup_modified' else (root / 'previous/foreign.bin').exists()


def test_existing_lease_is_never_replaced(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    lease = runtime.prefix.parent / '.venv.spacr-update-lock'
    lease.mkdir()
    (lease / 'owner').write_bytes(b'another process')
    with pytest.raises(FileExistsError):
        run_transaction(runtime, monkeypatch)
    assert (lease / 'owner').read_bytes() == b'another process'
    assert snapshot(runtime.prefix) == before
    assert not list(runtime.prefix.parent.glob('.spacr-windows-update-*'))


def test_downgrade_is_refused_under_lease_before_copy(runtime, monkeypatch):
    monkeypatch.setattr(cleanup, '_retiring_runtime_identity', lambda prefix: {'verified': True, 'version': '1.5.1.6'})
    with pytest.raises(ValueError, match='newer'):
        run_transaction(runtime, monkeypatch)
    assert not list(runtime.prefix.parent.glob('.spacr-windows-update-*'))


def test_cancelled_shutdown_never_stages_or_runs_pip(runtime, monkeypatch):
    plan = plan_for(runtime)
    before = snapshot(runtime.prefix)
    events = []

    class Handshake:
        def __init__(self, plan):
            pass

        def ready(self):
            events.append('ready')

        def wait_for_shutdown(self, wait):
            raise RuntimeError('cancelled')

    monkeypatch.setattr(cleanup, '_WindowsPipHandshake', Handshake)
    runner = PackageRunner(runtime)
    with pytest.raises(RuntimeError, match='cancelled'):
        cleanup._run_windows_pip_update(plan, runtime.machine, run=runner)
    assert events == ['ready']
    assert not any('-m' in cmd for cmd in runner.commands)
    assert not list(runtime.prefix.parent.glob('.spacr-windows-update-*'))
    assert snapshot(runtime.prefix) == before


@pytest.mark.parametrize('mode', ['console', 'gui', 'unowned', 'missing_record', 'ambiguous_owner'])
def test_real_distlib_windows_launcher_and_record_relocation(runtime, mode):
    stage = runtime.helper / 'staged'
    shutil.copytree(runtime.prefix, stage)
    site = stage / 'Lib/site-packages'
    metadata = site / 'spacr-1.5.1.5.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Name: spacr\nVersion: 1.5.1.5\n')
    group = 'gui_scripts' if mode == 'gui' else 'console_scripts'
    (metadata / 'entry_points.txt').write_text('[' + group + ']\nspacr = spacr.cli:main\n')
    with (metadata / 'RECORD').open('w', newline='') as stream:
        csv.writer(stream).writerow(['../../Scripts/spacr.exe' if mode != 'missing_record' else 'unrelated.py', '', ''])
    if mode == 'ambiguous_owner':
        other = site / 'foreign-1.0.dist-info'
        shutil.copytree(metadata, other)
        (other / 'METADATA').write_text('Name: foreign\nVersion: 1.0\n')
    unchanged = stage / 'Scripts/user-custom.exe'
    unchanged.write_bytes(b'MZ arbitrary preexisting user script remains exact')
    context = '''import builtins, os, sys, types
from pathlib import Path
from pip._vendor.distlib import scripts as original
native_os = types.SimpleNamespace(**vars(os))
native_os.name = 'nt'
original_import = builtins.__import__
def import_for_distlib(name, *args, **kwargs):
    return native_os if name == 'os' else original_import(name, *args, **kwargs)
module = types.ModuleType(original.__name__)
module.__file__, module.__package__ = original.__file__, original.__package__
module.__dict__['__builtins__'] = dict(vars(builtins), __import__=import_for_distlib)
exec(compile(Path(original.__file__).read_text(), original.__file__, 'exec'), module.__dict__)
sys.modules[original.__name__] = module
stage = Path(sys.argv[1])
maker = module.ScriptMaker(None, str(stage / 'Scripts'))
maker.executable = module.enquote_executable(str(stage / 'Scripts/python.exe'))
maker.variants = {''}
maker.make('spacr = spacr.cli:main', options={'gui': GUI})
'''.replace('GUI', repr(mode == 'gui'))
    if mode == 'unowned':
        context += "(stage / 'Scripts/unknown.exe').write_bytes(b'MZ' + str(stage).encode('utf-8'))\n"
    commands = []

    def native_format_runner(argv):
        commands.append(argv)
        script = argv[argv.index('-c') + 1]
        process = subprocess.run([sys.executable, '-I', '-B', '-c', context + '\n' + script, str(stage), str(runtime.prefix)],
                                 capture_output=True, text=True, timeout=30)
        return process.returncode, process.stdout + process.stderr

    if mode in {'unowned', 'missing_record', 'ambiguous_owner'}:
        with pytest.raises(ValueError, match='windows_entrypoint_verification_failed'):
            cleanup._windows_pip_scripts(str(stage), str(runtime.prefix), native_format_runner)
        return
    receipt = cleanup._windows_pip_scripts(str(stage), str(runtime.prefix), native_format_runner)
    assert receipt == {'verified': True, 'scripts': ['spacr.exe']}
    payload = (stage / 'Scripts/spacr.exe').read_bytes()
    assert payload.startswith(b'MZ')
    executable = runtime.prefix / 'Scripts' / ('pythonw.exe' if mode == 'gui' else 'python.exe')
    assert ('#!"' + str(executable) + '"\n').encode('utf-8') in payload
    assert str(stage).encode('utf-8') not in payload
    assert b'from spacr.cli import main' in zipfile.ZipFile(io.BytesIO(payload)).read('__main__.py')
    rows = list(csv.reader((metadata / 'RECORD').open(newline='')))
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode('ascii')
    assert rows == [['../../Scripts/spacr.exe', 'sha256=' + digest, str(len(payload))]]
    assert unchanged.read_bytes() == b'MZ arbitrary preexisting user script remains exact'
    assert len(commands) == 1


def test_timestamp_collision_invalidates_owned_stage_without_changing_old_file(runtime, monkeypatch):
    source = runtime.package / '__init__.py'
    os.utime(source, (1_700_000_000, 1_700_000_000))
    before = snapshot(runtime.prefix)
    original_mtime = source.stat().st_mtime_ns
    runner = PackageRunner(runtime)

    def collision(argv):
        code, output = runner(argv)
        if argv[1:5] == ['-I', '-m', 'pip', 'install']:
            staged_source = Path(argv[0]).parent.parent / 'Lib/site-packages/spacr/__init__.py'
            os.utime(staged_source, (1_700_000_000, 1_700_000_000))
        return code, output

    monkeypatch.setattr(cleanup, '_windows_pip_scripts', lambda *a: {'verified': True, 'scripts': []})
    plan = plan_for(runtime)
    cleanup._windows_pip_transaction(runtime.record, plan['version'], plan['install'], machine=runtime.machine, runner=collision)
    root = next(runtime.prefix.parent.glob('.spacr-windows-update-*'))
    assert snapshot(root / 'previous') == before
    assert (root / 'previous/Lib/site-packages/spacr/__init__.py').stat().st_mtime_ns == original_mtime
    assert int(source.stat().st_mtime) == 1_700_000_002


def handshake_plan(runtime):
    runtime.helper.mkdir()
    plan = plan_for(runtime)
    plan.update(adapter='pip-private-windows-v1', pid=4242, ticked=[],
                handshake={'schema': 1, 'token': 'a' * 64, 'expires': cleanup.time.time() + 3600})
    return plan


def test_windows_handshake_authenticates_readiness_approval_and_real_exit(runtime):
    plan = handshake_plan(runtime)
    original = snapshot(runtime.prefix)
    gui = cleanup._WindowsPipHandshake(plan)
    helper = cleanup._WindowsPipHandshake(json.loads(json.dumps(plan)))
    assert gui.status() is None
    with pytest.raises(RuntimeError, match='not ready'):
        gui.approve()
    helper.ready()
    assert gui.status()['state'] == 'ready'
    gui.approve()
    waited = []
    helper.wait_for_shutdown(wait=lambda pid: waited.append(pid) or True)
    assert waited == [4242]
    assert snapshot(runtime.prefix) == original
    assert cleanup._FrozenUpdateHandshake.ready is cleanup._WindowsPipHandshake.ready
    assert cleanup._FrozenUpdateHandshake.approve is cleanup._WindowsPipHandshake.approve
    assert cleanup._FrozenUpdateHandshake.wait_for_shutdown is cleanup._WindowsPipHandshake.wait_for_shutdown


@pytest.mark.parametrize('failure', ['cancelled', 'expired', 'foreign_binding', 'symlink', 'oversized', 'foreign_acl', 'replaced'])
def test_windows_handshake_refuses_untrusted_or_cancelled_messages(runtime, monkeypatch, failure):
    plan = handshake_plan(runtime)
    helper = cleanup._WindowsPipHandshake(plan)
    helper.ready()
    path = runtime.helper / 'readiness.json'
    if failure == 'cancelled':
        helper.approve()
        helper.cancel()
    elif failure == 'expired':
        monkeypatch.setattr(cleanup.time, 'time', lambda: plan['handshake']['expires'] + 1)
    elif failure == 'foreign_binding':
        contents = json.loads(path.read_text())
        contents['binding'] = 'foreign'
        path.write_text(json.dumps(contents))
    elif failure == 'symlink':
        outside = runtime.helper.parent / 'foreign.json'
        path.rename(outside)
        path.symlink_to(outside)
    elif failure == 'oversized':
        path.write_text(' ' * 8193)
    elif failure == 'foreign_acl':
        def foreign_acl(checked):
            evidence = acl()
            if Path(checked) == path:
                evidence['rules'].append({'sid': 'S-1-1-0', 'type': 0, 'rights': 2, 'propagation': 0})
            return evidence
        monkeypatch.setattr(cleanup, '_windows_pip_acl', foreign_acl)
    else:
        original_open = cleanup.os.open
        def replace_at_open(checked, flags, *args, **kwargs):
            if Path(checked) == path:
                path.rename(path.with_suffix('.original'))
                path.write_text(path.with_suffix('.original').read_text())
            return original_open(checked, flags, *args, **kwargs)
        monkeypatch.setattr(cleanup.os, 'open', replace_at_open)
    with pytest.raises((RuntimeError, ValueError)):
        helper.status()


def test_windows_update_uses_real_handshake_and_plan_dispatch(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    plan = handshake_plan(runtime)
    plan.update(log=str(runtime.helper / 'update.log'), schema=1)
    path = runtime.helper / 'plan.json'
    path.write_text(json.dumps(plan))
    ready = cleanup._WindowsPipHandshake.ready
    events = []

    def approve_after_ready(handshake):
        ready(handshake)
        assert snapshot(runtime.prefix) == before
        events.append('ready')
        cleanup._WindowsPipHandshake(plan).approve()

    monkeypatch.setattr(cleanup._WindowsPipHandshake, 'ready', approve_after_ready)
    monkeypatch.setattr(cleanup, '_windows_pip_scripts', lambda *args: {'verified': True, 'scripts': []})
    runner = PackageRunner(runtime)
    result = cleanup._run_plan(str(path), system=runtime.machine, run=runner,
                               wait=lambda pid: events.append('exit') or True,
                               spawn=lambda *args: events.append('relaunch'),
                               fetch=lambda *args: pytest.fail('No installer download'),
                               remove=lambda *args: pytest.fail('No removal of existing installations'))
    assert result == 0
    assert events == ['ready', 'exit', 'relaunch']
    assert snapshot(next(runtime.prefix.parent.glob('.spacr-windows-update-*')) / 'previous') == before


def test_inherit_only_foreign_write_cannot_create_writable_staged_children(runtime, monkeypatch):
    evidence = acl()
    evidence['rules'].append({'sid': 'S-1-1-0', 'type': 0, 'rights': 2, 'propagation': 2})
    monkeypatch.setattr(cleanup, '_windows_pip_acl', lambda path: evidence)
    with pytest.raises(ValueError):
        plan_for(runtime)


@pytest.mark.parametrize('foreign', [False, True])
def test_external_base_acl_is_proven_before_interpreter_execution(runtime, monkeypatch, foreign):
    runtime.helper.mkdir()
    executed = []

    def evidence(path):
        result = acl()
        if Path(path) in {runtime.base, runtime.base_python}:
            result['owner'] = 'S-1-5-32-544'
            if foreign:
                result['rules'].append({'sid': 'S-1-1-0', 'type': 0, 'rights': 2, 'propagation': 0})
        return result

    def runner(argv, **kwargs):
        executed.append(argv)
        return 0, json.dumps({'os_name': 'nt', 'python': str(runtime.base_python), 'prefix': str(runtime.base),
                              'base_prefix': str(runtime.base), 'stdlib': str(runtime.base / 'Lib')})

    monkeypatch.setattr(cleanup, '_windows_pip_acl', evidence)
    monkeypatch.setattr(cleanup, '_run', runner)
    if foreign:
        with pytest.raises(ValueError):
            cleanup._windows_pip_helper(runtime.record, str(runtime.helper / 'helper.py'), str(runtime.helper / 'plan.json'), runtime.machine)
        assert executed == []
    else:
        command, environment = cleanup._windows_pip_helper(runtime.record, str(runtime.helper / 'helper.py'), str(runtime.helper / 'plan.json'), runtime.machine)
        assert command[0] == str(runtime.base_python)
        assert len(executed) == 1
        assert 'PYTHONHOME' not in environment and 'PYTHONPATH' not in environment


@pytest.mark.parametrize('defect', ['foreign_write', 'null_dacl', 'owner', 'inherit_only', 'unreadable'])
def test_descendant_only_acl_exceptions_refuse_before_any_runtime_mutation(runtime, monkeypatch, defect):
    target = runtime.package / '__init__.py'
    before = snapshot(runtime.prefix)

    def evidence(path):
        result = acl()
        if Path(path) == target:
            if defect == 'unreadable':
                raise PermissionError('modeled protected child DACL')
            if defect == 'owner':
                result['owner'] = 'S-1-1-0'
            elif defect == 'null_dacl':
                result['has_dacl'] = False
            else:
                result['rules'].append({'sid': 'S-1-1-0', 'type': 0, 'rights': 2,
                                        'propagation': 2 if defect == 'inherit_only' else 0})
        return result

    monkeypatch.setattr(cleanup, '_windows_pip_acl', evidence)
    with pytest.raises((ValueError, PermissionError)):
        plan_for(runtime)
    assert snapshot(runtime.prefix) == before
    assert not list(runtime.prefix.parent.glob('.spacr-windows-update-*'))


def test_descendant_proof_is_not_cached(runtime, monkeypatch):
    assert cleanup._windows_pip_tree_acl(str(runtime.prefix))['entries'] > 5
    original = cleanup._windows_pip_acl

    def changed(path):
        result = original(path)
        if Path(path) == runtime.package:
            result['has_dacl'] = False
        return result

    monkeypatch.setattr(cleanup, '_windows_pip_acl', changed)
    with pytest.raises(ValueError):
        cleanup._windows_pip_tree_acl(str(runtime.prefix))


def test_descendant_reparse_attribute_is_refused_without_following_it(runtime, monkeypatch):
    original = cleanup.os.lstat
    target = runtime.package

    def reparse(path, **kwargs):
        result = original(path, **kwargs)
        if Path(path) == target:
            return types.SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(cleanup.os, 'lstat', reparse)
    with pytest.raises(ValueError):
        cleanup._windows_pip_tree_acl(str(runtime.prefix))


class NativeAclModel:
    def __init__(self, ctypes, defect=None):
        self.ctypes = ctypes
        self.defect = defect
        self.allocations = []
        self.freed = []
        self.closed = []
        self.queried = []
        self.owner = ctypes.create_string_buffer(b'owner')
        self.user = ctypes.create_string_buffer(b'user')
        self.descriptor = ctypes.create_string_buffer(b'descriptor')
        ace_type = 5 if defect == 'unknown_ace' else 0
        ace = bytes([ace_type, 0]) + (12).to_bytes(2, 'little') + (0x1f01ff).to_bytes(4, 'little') + b'SID!'
        self.acl = ctypes.create_string_buffer(b'\x02\x00' + (20).to_bytes(2, 'little') + b'\x01\x00\x00\x00' + ace)
        self.security = types.SimpleNamespace(**{name: self.function(getattr(self, name)) for name in
            ['GetNamedSecurityInfoW', 'GetAce', 'ConvertSidToStringSidW', 'OpenProcessToken', 'GetTokenInformation']})
        self.kernel = types.SimpleNamespace(**{name: self.function(getattr(self, name)) for name in
            ['GetVolumePathNameW', 'GetDriveTypeW', 'GetCurrentProcess', 'LocalFree', 'CloseHandle']})

    @staticmethod
    def function(fn):
        def call(*args):
            return fn(*args)
        return call

    def set_pointer(self, target, address):
        self.ctypes.cast(target, self.ctypes.POINTER(self.ctypes.c_void_p)).contents.value = address

    def GetVolumePathNameW(self, path, buffer, size):
        buffer.value = '\\\\server\\share\\' if self.defect == 'unc_volume' else 'C:\\'
        return 0 if self.defect == 'volume_error' else 1

    def GetDriveTypeW(self, path):
        return 4 if self.defect == 'network_drive' else 3

    def GetCurrentProcess(self):
        return 123

    def LocalFree(self, pointer):
        self.freed.append(pointer.value)

    def CloseHandle(self, token):
        self.closed.append(token.value)
        return 1

    def GetNamedSecurityInfoW(self, path, kind, flags, owner, group, dacl, sacl, descriptor):
        self.queried.append(path)
        assert kind == 1 and flags == 5 and group is None and sacl is None
        if self.defect == 'security_error':
            return 5
        self.set_pointer(owner, self.ctypes.addressof(self.owner))
        self.set_pointer(dacl, None if self.defect == 'null_dacl' else self.ctypes.addressof(self.acl))
        self.set_pointer(descriptor, self.ctypes.addressof(self.descriptor))
        return 0

    def OpenProcessToken(self, process, rights, token):
        assert rights == 8
        if self.defect == 'token_error':
            return 0
        self.set_pointer(token, 999)
        return 1

    def GetTokenInformation(self, token, kind, buffer, size, needed):
        assert token.value == 999 and kind == 1
        self.ctypes.cast(needed, self.ctypes.POINTER(self.ctypes.c_uint32)).contents.value = 16
        if buffer is None:
            return 0
        self.set_pointer(buffer, self.ctypes.addressof(self.user))
        return 0 if self.defect == 'token_read_error' else 1

    def GetAce(self, acl, index, ace):
        assert acl.value == self.ctypes.addressof(self.acl) and index == 0
        self.set_pointer(ace, self.ctypes.addressof(self.acl) + 8)
        return 0 if self.defect == 'ace_error' else 1

    def ConvertSidToStringSidW(self, sid, text):
        if self.defect == 'sid_error':
            return 0
        buffer = self.ctypes.create_unicode_buffer(USER)
        self.allocations.append(buffer)
        self.set_pointer(text, self.ctypes.addressof(buffer))
        return 1


@pytest.mark.parametrize('defect', [None, 'network_drive', 'unc_volume', 'volume_error', 'security_error',
                                  'token_error', 'token_read_error', 'sid_error', 'unknown_ace', 'ace_error', 'null_dacl'])
def test_native_acl_reader_has_fixed_volume_and_fail_closed_api_contracts(tmp_path, monkeypatch, defect):
    import ctypes

    native_os = types.SimpleNamespace(**vars(os))
    native_os.name = 'nt'
    monkeypatch.setattr(cleanup, 'os', native_os)
    model = NativeAclModel(ctypes, defect)
    monkeypatch.setattr(ctypes, 'WinDLL', lambda name, **kwargs: model.kernel if name == 'kernel32' else model.security, raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: 5, raising=False)
    if defect not in {None, 'null_dacl'}:
        with pytest.raises((ValueError, OSError)):
            cleanup._windows_pip_acl(str(tmp_path))
    else:
        result = cleanup._windows_pip_acl(str(tmp_path))
        assert result['owner'] == USER and result['user'] == USER
        assert result['has_dacl'] is (defect != 'null_dacl')
        assert len(result['rules']) == (0 if defect else 1)
    if defect not in {'network_drive', 'unc_volume', 'volume_error', 'security_error'}:
        assert ctypes.addressof(model.descriptor) in model.freed
    if defect not in {'network_drive', 'unc_volume', 'volume_error', 'security_error', 'token_error'}:
        assert model.closed == [999]


@pytest.mark.parametrize('fail_relaunch', [False, True])
def test_actual_offline_wheel_and_original_identity_probe_in_disposable_layout(runtime, tmp_path, monkeypatch, fail_relaunch):
    from tests.test_macos_runtime_transaction_703 import _wheel

    wheel = _wheel(tmp_path / 'offline-wheel', '1.5.1.5')
    old = snapshot(runtime.prefix)
    package_commands = []

    def invoke(argv):
        target = Path(argv[0]).parent.parent
        site = target / 'Lib/site-packages'
        if argv[1:5] == ['-I', '-m', 'pip', 'install']:
            assert target != runtime.prefix
            package_commands.append(argv)
            command = [sys.executable, '-I', '-B', '-m', 'pip', '--isolated', '--no-cache-dir',
                       'install', '--no-index', '--no-deps', '--no-compile', '--upgrade', '--target', str(site), str(wheel)]
        elif argv[1:] == ['-I', '-m', 'pip', 'check']:
            return 0, 'dependency-free offline fixture wheel'
        else:
            assert '-c' in argv
            probe = argv[argv.index('-c') + 1]
            shim = ('import sys\n'
                    + 'sys.path.insert(0, ' + repr(str(site)) + ')\n'
                    + 'sys.prefix=' + repr(str(target)) + '\n'
                    + 'sys.executable=' + repr(argv[0]) + '\n')
            command = [sys.executable, '-I', '-B', '-c', shim + probe, *argv[argv.index('-c') + 2:]]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr

    def relaunch(version):
        assert version == '1.5.1.5'
        if fail_relaunch:
            raise OSError('deliberately refused relaunch')

    monkeypatch.setattr(cleanup, '_windows_pip_scripts', lambda *args: {'verified': True, 'scripts': []})
    plan = plan_for(runtime)
    if fail_relaunch:
        with pytest.raises(OSError, match='refused relaunch'):
            cleanup._windows_pip_transaction(runtime.record, plan['version'], plan['install'],
                                            machine=runtime.machine, runner=invoke, activate=relaunch)
        assert snapshot(runtime.prefix) == old
    else:
        version, _ = cleanup._windows_pip_transaction(runtime.record, plan['version'], plan['install'],
                                                    machine=runtime.machine, runner=invoke, activate=relaunch)
        assert version == '1.5.1.5'
        assert snapshot(next(runtime.prefix.parent.glob('.spacr-windows-update-*')) / 'previous') == old
        assert (runtime.package / '_version.py').read_text() == '__version__ = "1.5.1.5"\n'
    assert len(package_commands) == 1
