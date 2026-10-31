"""The opt-in native replay stays source-bound and cannot replace required CI."""

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = '7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab'
DIAGNOSTIC_SOURCE = '9e7b35c8abb4cf7d72cf1fa546322472e0c18819'
ORIGINAL_BATCH_SHA256 = 'd5f347bdb25b692cf207f18526b333d01e58a952a923842881367ad77ff65826'
ACCEPTED_SERIAL_JOB_SHA256 = '4f00af6b4b9911c7250a6fd6cbc65bab9e25eeb4f8f6582e53c1b3d0152db27c'
ACCEPTED_SERIAL_STEPS_SHA256 = '89a9bd163c24a1578d1e305cef433cc4922d52b359a5ae66d3b9eef41da1c248'


def _workflow():
    """Read the dispatch definition without YAML 1.1 coercing the `on` key."""
    yaml = pytest.importorskip('yaml')
    return yaml.load((REPO / '.github/workflows/qt-native-replay.yml').read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


def _serial_workflow():
    """Read the registered serial workflow using the same YAML key policy."""
    yaml = pytest.importorskip('yaml')
    return yaml.load((REPO / '.github/workflows/qt-serial-acceptance.yml').read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)


def _stable_sha256(value):
    """Hash a parsed job or step list without depending on YAML formatting."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _evaluate_condition(expression, *, event, replay, pixel=False, safe=False):
    """Evaluate the narrow Boolean routing vocabulary used by these jobs."""
    body = expression.replace('github.event_name', repr(event))
    body = body.replace('inputs.native_batch_replay', repr(replay))
    body = body.replace('inputs.pixel_context_a744', repr(pixel))
    body = body.replace('inputs.safe_start_83ed', repr(safe))
    body = body.replace('&&', ' and ').replace('||', ' or ')
    body = re.sub(r'(?<![!=])!(?!=)', ' not ', body)
    assert re.fullmatch(r"[\w\s'\"!=()\-]+", body)
    return eval(body, {'__builtins__': {}}, {})  # noqa: S307


def test_registered_serial_route_preserves_original_job_and_isolates_replay():
    """A diagnostic dispatch cannot start or queue behind the serial lane."""
    workflow = _serial_workflow()
    trigger = workflow['on']
    assert set(trigger) == {'workflow_dispatch', 'push'}
    option = trigger['workflow_dispatch']['inputs']['native_batch_replay']
    assert option['type'] == 'boolean'
    assert option['required'] == option['default'] == 'false'
    assert trigger['push']['branches'] == ['nightly']

    serial = workflow['jobs']['serial-qt']
    original = {key: value for key, value in serial.items() if key != 'if'}
    bootstrap = {
        'name': 'Restore pinned translation review inputs',
        'if': "hashFiles('docs/translation_reviews_revision.json') != ''",
        'run': 'python tools/bootstrap_translation_reviews.py fetch',
    }
    steps = serial['steps']
    assert steps.count(bootstrap) == 1
    bootstrap_index = steps.index(bootstrap)
    assert steps[bootstrap_index - 1]['name'] == 'Install the existing hosted Qt test dependencies'
    assert steps[bootstrap_index + 1]['name'] == 'Record serial source and dependency provenance'
    original['steps'] = steps[:bootstrap_index] + steps[bootstrap_index + 1:]
    assert _stable_sha256(original) == ACCEPTED_SERIAL_JOB_SHA256
    assert _stable_sha256(original['steps']) == ACCEPTED_SERIAL_STEPS_SHA256
    diagnostic = workflow['jobs']['native-batch-replay']
    assert diagnostic['uses'] == './.github/workflows/qt-native-replay.yml'
    assert diagnostic['permissions'] == {'contents': 'read'}

    concurrency = workflow['concurrency']
    assert concurrency['cancel-in-progress'] == 'false'
    prior_group = 'n47-serial-${{ github.ref }}'
    assert concurrency['group'].startswith(prior_group)
    suffix = concurrency['group'][len(prior_group):]
    match = re.fullmatch(r'\$\{\{ (.+) \}\}', suffix)
    assert match is not None
    for event, replay, serial_expected, diagnostic_expected in (
            ('push', False, True, False),
            ('workflow_dispatch', False, True, False),
            ('workflow_dispatch', True, False, True)):
        assert _evaluate_condition(serial['if'], event=event, replay=replay) is serial_expected
        assert _evaluate_condition(diagnostic['if'], event=event,
                                   replay=replay) is diagnostic_expected
        group_suffix = _evaluate_condition(match.group(1), event=event, replay=replay)
        assert group_suffix == ('-native-batch-replay' if replay else '')


def test_native_replay_is_opt_in_source_bound_and_two_independent_original_batches():
    workflow = _workflow()
    assert set(workflow['on']) == {'workflow_call', 'workflow_dispatch'}
    assert set(workflow['jobs']) == {'replay', 'pixel-context', 'safe-start'}
    job = workflow['jobs']['replay']
    assert job['runs-on'] == 'ubuntu-24.04'
    assert job.get('permissions', workflow['permissions']) == {'contents': 'read'}
    assert job['env']['SPACR_TEST_MEMORY_GB'] == '6'
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert all(job['env'][name] == '1' for name in (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    steps = {step['name']: step for step in job['steps'] if 'name' in step}
    assert steps['Checkout pinned diagnostic collector']['with']['ref'] == DIAGNOSTIC_SOURCE
    assert steps['Checkout exact failing test source']['with']['ref'] == REPLAY_SOURCE

    selection = steps['Bind source, dependencies and the exact 32-file batch']['run']
    assert f'test "$(git rev-parse HEAD)" = "{REPLAY_SOURCE}"' in selection
    assert 'diagnostic_collector_sha=%s' in selection
    assert 'python -m pip freeze --all' in selection
    assert ORIGINAL_BATCH_SHA256 in selection
    assert "_shard(path, root, 12) == 0" in selection
    assert '_batches(selected, 32)[2]' in selection
    assert 'assert len(paths) == 32' in selection
    embedded = re.search(r"python - <<'PY'\n(.*?)\nPY", selection, re.S)
    assert embedded is not None
    ast.parse(embedded.group(1))

    replay = steps['Replay the original-order coverage batch twice independently']
    assert 'continue-on-error' not in replay
    command = replay['run']
    assert f'export GITHUB_SHA={REPLAY_SOURCE}' in command
    assert command.count('tools/run_coverage_batches.py "${batch[@]}"') == 1
    assert 'set -euo pipefail' in command
    assert 'for attempt in 1 2; do' in command
    assert 'attempt_dir="$SPACR_COVERAGE_DATA_DIR/attempt-$attempt"' in command
    assert '--data-dir "$attempt_dir"' in command
    assert 'status=started' in command and 'status=success' in command
    assert command.index('tools/run_coverage_batches.py') < command.index('status=success')
    assert 'status=failed exit=%s' in command and 'exit "$status"' in command
    assert '|| true' not in command and 'continue-on-error' not in replay
    for flag, value in (('--marker', '"not gui"'), ('--shard-index', '0'),
                        ('--shard-count', '1'), ('--batch-size', '32'),
                        ('--workers', '2'), ('--per-test-timeout', '600'),
                        ('--batch-timeout', '2700')):
        assert f'{flag} {value}' in command
    assert 'qt_serial_rss_journal' in command
    assert 'collect_qt_native_backtrace.py" --record-core-route' in command



def test_pixel_context_requires_opt_in_and_never_runs_the_historical_replay():
    """One diagnostic dispatch preserves the original replay and full CI gate."""
    workflow = _workflow()
    option = workflow['on']['workflow_dispatch']['inputs']['pixel_context_a744']
    assert option['type'] == 'boolean'
    assert option['required'] == option['default'] == 'false'
    historical, pixel = workflow['jobs']['replay'], workflow['jobs']['pixel-context']
    for event, opt_in, historical_expected, pixel_expected in (
            ('workflow_call', False, True, False),
            ('workflow_dispatch', False, True, False),
            ('workflow_dispatch', True, False, True)):
        assert _evaluate_condition(historical['if'], event=event, replay=False,
                                   pixel=opt_in) is historical_expected
        assert _evaluate_condition(pixel['if'], event=event, replay=False,
                                   pixel=opt_in) is pixel_expected
    assert pixel['runs-on'] == 'ubuntu-24.04'
    assert pixel.get('permissions', workflow['permissions']) == {'contents': 'read'}
    assert pixel['env']['CUDA_VISIBLE_DEVICES'] == ''
    steps = {step['name']: step for step in pixel['steps'] if 'name' in step}
    source = steps['Checkout immutable original application and assertions']['with']
    assert source['ref'] == 'ff1b2a487d378cfa804c3c1923fb834e0cc17dc2'
    assert source['persist-credentials'] == 'false'
    assert source['path'] == 'source'
    context = json.loads((REPO / 'tools/qt_pixel_context_a744.json').read_text(encoding="utf-8"))
    assert context['source_sha'] == source['ref']
    assert len(context['nodes']) == len(set(context['nodes'])) == 104
    assert list(dict.fromkeys(node.split('::', 1)[0] for node in context['nodes'])) == context['modules']
    assert len(context['modules']) == 6
    command = steps['One original 104-node worker context; capture only after failure']['run']
    assert 'qt_pixel_context_diagnostic.py" run' in command
    assert 'for attempt' not in command and 'continue-on-error' not in steps['One original 104-node worker context; capture only after failure']
    driver = (REPO / 'tools/qt_pixel_context_diagnostic.py').read_text(encoding="utf-8")
    assert '"actual_original_pytest_exit": original.returncode' in driver
    assert '"full_ci_acceptance": False' in driver
    assert '"retries": 0' in driver
    assert '"--max-worker-restart=0"' in driver
    assert '"-n", "1"' in driver


def test_native_replay_retains_failure_and_bounded_core_capture():
    job = _workflow()['jobs']['replay']
    steps = {step['name']: step for step in job['steps'] if 'name' in step}
    collector = steps["Recover a failed worker's bounded native evidence"]
    upload = steps['Upload native failure evidence']
    assert collector['if'] == upload['if'] == 'failure() || cancelled()'
    assert collector['continue-on-error'] == 'true'
    assert collector['timeout-minutes'] == '3'
    assert f'export GITHUB_SHA={REPLAY_SOURCE}' in collector['run']
    assert '--ordinary' in collector['run']
    assert upload['with']['retention-days'] == '3'
    assert upload['with']['if-no-files-found'] == 'ignore'
    assert upload['with']['path'].splitlines() == [
        '${{ runner.temp }}/spacr-qt-native/process-*.jsonl',
        '${{ runner.temp }}/spacr-qt-native/native-core-backtrace.txt',
        '${{ runner.temp }}/spacr-qt-native/core-route.json',
    ]
    verdict = steps['State the diagnostic-only verdict']
    assert verdict['if'] == 'always()'
    assert 'failure remains a failure' in verdict['run']
    assert job['timeout-minutes'] == '90'
    assert steps['Upload two-attempt coverage integrity data']['with']['path'] == (
        '${{ runner.temp }}/spacr-coverage-data/')


@pytest.mark.posix_only
def test_two_attempt_shell_stops_on_either_failure_and_separates_data(tmp_path):
    """A failed first or second replay keeps its failure and original receipt."""
    job = _workflow()['jobs']['replay']
    steps = {step['name']: step for step in job['steps'] if 'name' in step}
    command = steps['Replay the original-order coverage batch twice independently']['run']
    assert command.count('for attempt in 1 2; do') == 1
    loop = 'for attempt in 1 2; do' + command.split('for attempt in 1 2; do', 1)[1]
    # The excerpt starts after the step defines `batch`; bash before 4.4
    # (macOS's 3.2) rejects "${batch[@]}" of an unset array under set -u.
    script = '''set -euo pipefail
batch=(tests/stand_in.py)
python() {
  local data_dir='' arg
  for arg in "$@"; do
    if [ "$data_dir" = next ]; then data_dir="$arg"; break; fi
    if [ "$arg" = --data-dir ]; then data_dir=next; fi
  done
  test "$data_dir" != ''
  printf '%s\n' "$data_dir" >> "$RUNNER_TEMP/calls.txt"
  if [ "$data_dir" = "$FAIL_DIR" ]; then return 7; fi
}
''' + loop
    receipt = tmp_path / 'spacr-native-replay'
    receipt.mkdir()
    for failing, expected_calls, expected_status in (
            ('none', 2, 0), ('attempt-1', 1, 7), ('attempt-2', 2, 7)):
        (tmp_path / 'calls.txt').unlink(missing_ok=True)
        (receipt / 'attempts.txt').unlink(missing_ok=True)
        environment = {
            'RUNNER_TEMP': str(tmp_path),
            'SPACR_COVERAGE_DATA_DIR': str(tmp_path / 'coverage'),
            'GITHUB_SHA': REPLAY_SOURCE,
            'FAIL_DIR': str(tmp_path / 'coverage' / failing),
        }
        result = subprocess.run(['bash', '-c', script], cwd=REPO,
                                env={**os.environ, **environment},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == expected_status, result.stderr
        calls = (tmp_path / 'calls.txt').read_text(encoding="utf-8").splitlines()
        assert calls == [str(tmp_path / 'coverage' / f'attempt-{number}')
                         for number in range(1, expected_calls + 1)]
        events = (receipt / 'attempts.txt').read_text(encoding="utf-8").splitlines()
        assert events[0] == f'attempt=1 status=started source={REPLAY_SOURCE}'
        assert events.count(f'attempt=1 status=success source={REPLAY_SOURCE}') == (
            0 if failing == 'attempt-1' else 1)
        assert events.count(f'attempt=2 status=success source={REPLAY_SOURCE}') == (
            1 if failing == 'none' else 0)
        failures = [event for event in events if 'status=failed' in event]
        assert failures == ([] if failing == 'none' else [
            f'attempt={failing[-1]} status=failed exit=7 source={REPLAY_SOURCE}'])


def test_safe_start_diagnostic_preserves_default_and_pixel_modes():
    """A manual child diagnostic selects one CPU job without ordinary replay."""
    workflow = _workflow()
    option = workflow['on']['workflow_dispatch']['inputs']['safe_start_83ed']
    assert option['type'] == 'boolean'
    assert option['required'] == option['default'] == 'false'
    jobs = workflow['jobs']
    for event, pixel, safe, expected in (
            ('workflow_call', False, False, (True, False, False)),
            ('workflow_dispatch', False, False, (True, False, False)),
            ('workflow_dispatch', True, False, (False, True, False)),
            ('workflow_dispatch', False, True, (False, False, True)),
            ('workflow_dispatch', True, True, (False, False, True))):
        actual = tuple(_evaluate_condition(jobs[name]['if'], event=event,
                                          replay=False, pixel=pixel, safe=safe)
                       for name in ('replay', 'pixel-context', 'safe-start'))
        assert actual == expected
    job = jobs['safe-start']
    assert job['runs-on'] == 'ubuntu-24.04'
    assert job['permissions'] == {'contents': 'read'}
    assert 'services' not in job
    assert job['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert job['timeout-minutes'] == '65'
    steps = {step['name']: step for step in job['steps'] if 'name' in step}
    reject = steps['Reject mixed diagnostic modes']
    assert reject['env']['PIXEL_SELECTED'] == '${{ inputs.pixel_context_a744 }}'
    assert 'test "$PIXEL_SELECTED" != "true"' in reject['run']
    source = steps['Checkout immutable original application and tests']['with']
    assert source['ref'] == '83edc25dfda63e45191aa6b4b2833ddbb4b850ab'
    assert source['persist-credentials'] == 'false'
    assert source['path'] == 'source'
    context = json.loads((REPO / 'tools/safe_start_83ed_context.json').read_text(encoding="utf-8"))
    assert context['source_sha'] == source['ref']
    assert context['original_expected_case_count'] == 580
    assert len(context['batch5_files']) == len(set(context['batch5_files'])) == 32
    assert len(context['original_fixture_nodes']) == 7
    assert hashlib.sha256(('\n'.join(context['batch5_files']) + '\n').encode()).hexdigest() == context['batch5_sha256']
    command = steps['One original covered32-file context with child evidence']['run']
    assert 'safe_start_context_diagnostic.py" run' in command
    assert 'for attempt' not in command
    assert 'continue-on-error' not in steps['One original covered32-file context with child evidence']
    driver = (REPO / 'tools/safe_start_context_diagnostic.py').read_text(encoding="utf-8")
    ast.parse(driver)
    assert "'original_pytest_exit':original_exit" in driver
    assert "'full_ci_acceptance':False" in driver
    assert "'retries':0" in driver
    assert 'process.wait(timeout=2700)' in driver
    assert 'collector_integrity' in driver
    upload = steps['Preserve raw child stacks, stdout, stderr, original JSON and coverage']
    assert upload['if'] == 'always()'
    assert upload['with']['retention-days'] == '3'


def _sidecar(name):
    path = REPO / 'tools/pytest_plugins/safe_start_83ed' / (name + '.py')
    spec = importlib.util.spec_from_file_location('safe_diagnostic_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.posix_only
def test_safe_start_job_defers_runner_paths_until_runtime(tmp_path):
    """Job env cannot evaluate runner context; steps publish its actual paths."""
    job = _workflow()['jobs']['safe-start']
    assert all('${{ runner.' not in value for value in job['env'].values())
    assert 'NUMBA_CACHE_DIR' not in job['env']
    assert 'SPACR_QT_NATIVE_EVIDENCE_DIR' not in job['env']
    steps = job['steps']
    names = [step.get('name') for step in steps]
    index = names.index('Establish private diagnostic paths at runtime')
    assert index < names.index('Checkout reviewed diagnostics only')
    step = steps[index]
    assert step['working-directory'] == '${{ github.workspace }}'
    runtime_temp = tmp_path / 'runner temp'
    environment_file = tmp_path / 'github env'
    environment_file.touch()
    result = subprocess.run(['bash', '-eu', '-c', step['run']], cwd=tmp_path,
                            env={**os.environ, 'RUNNER_TEMP': str(runtime_temp),
                                 'GITHUB_ENV': str(environment_file)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert environment_file.read_text(encoding="utf-8").splitlines() == [
        f'NUMBA_CACHE_DIR={runtime_temp}/spacr-safe-start-numba',
        f'SPACR_QT_NATIVE_EVIDENCE_DIR={runtime_temp}/spacr-safe-start-native']


@pytest.mark.parametrize('outcome', ['completed', 'timeout'])
@pytest.mark.parametrize('failure', ['prepare', 'stdout', 'result', 'integrity-channel'])
def test_child_capture_failure_preserves_original_call_and_outcome(
        tmp_path, monkeypatch, outcome, failure):
    """Broken evidence storage cannot replace the original result/exception."""
    sidecar = _sidecar('safe_child_evidence')
    launcher = tmp_path / 'launch_once.py'
    launcher.write_text('original launcher', encoding="utf-8")
    monkeypatch.setenv('SPACR_SAFE_ORIGINAL_LAUNCHER_SHA',
                       hashlib.sha256(launcher.read_bytes()).hexdigest())
    monkeypatch.setenv('SPACR_SAFE_DIAGNOSTIC_DIR', str(tmp_path / 'evidence'))
    command = ['python', str(launcher), 'source', 'normal', 'repaired.json',
               'blobs', 'data_art_impulse_lens']
    original = (subprocess.CompletedProcess(command, 0, b'original-out', b'original-err')
                if outcome == 'completed' else
                subprocess.TimeoutExpired(command, 300, b'original-out', b'original-err'))
    calls = []

    def invoke(*args, **kwargs):
        calls.append((args, kwargs))
        if outcome == 'timeout':
            raise original
        return original

    def broken(*args, **kwargs):
        raise OSError('Evidence disk unavailable')

    monkeypatch.setattr(sidecar, '_original', invoke)
    if failure in ('prepare', 'integrity-channel'):
        monkeypatch.setattr(sidecar, '_prepare', broken)
    elif failure == 'stdout':
        monkeypatch.setattr(sidecar, '_save_bytes', broken)
    else:
        write_text = Path.write_text

        def result_write(path, *args, **kwargs):
            if path.name == 'result.json':
                return broken()
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'write_text', result_write)
    if failure == 'integrity-channel':
        class BrokenStderr:
            write = flush = broken
        monkeypatch.setattr(sidecar.sys, '__stderr__', BrokenStderr())
    kwargs = {'timeout': 300, 'capture_output': True}
    if outcome == 'timeout':
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            sidecar._run(command, **kwargs)
        assert caught.value is original
    else:
        assert sidecar._run(command, **kwargs) is original
    assert calls == [((command,), kwargs)]
    assert sidecar._errors


@pytest.mark.posix_only
def test_child_phase_write_failure_cannot_escape_the_profile_callback(
        tmp_path, monkeypatch):
    """A failing phase journal marks integrity without interrupting app code."""
    import atexit
    import faulthandler
    import sys
    from types import SimpleNamespace

    sidecar = _sidecar('sitecustomize')
    monkeypatch.setenv('SPACR_SAFE_DIAGNOSTIC_DIR', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['launch_once.py', 'source', 'normal',
                                     'repaired.json', 'blobs', 'data_art_impulse_lens'])
    profile = []
    monkeypatch.setattr(sys, 'setprofile', profile.append)
    monkeypatch.setattr(atexit, 'register', lambda function: None)
    monkeypatch.setattr(faulthandler, 'dump_traceback_later', lambda *args, **kwargs: None)
    coverage = SimpleNamespace(Coverage=SimpleNamespace(current=lambda: object()))
    monkeypatch.setitem(sys.modules, 'coverage', coverage)
    original_open = Path.open

    class BrokenJournal:
        def write(self, value):
            raise OSError('Journal unavailable')
        def flush(self):
            raise OSError('Journal unavailable')

    def open_journal(path, *args, **kwargs):
        return BrokenJournal() if path.name == 'phases.jsonl' else original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_journal)
    sidecar._initialize()
    frame = SimpleNamespace(f_code=SimpleNamespace(
        co_filename=os.path.abspath('launch_once.py'), co_name='_watchdog'), f_lineno=1)
    profile[-1](frame, 'call', None)
    assert len(sidecar._errors) == 2
    assert all(error['operation'] == 'phase-journal' for error in sidecar._errors)
