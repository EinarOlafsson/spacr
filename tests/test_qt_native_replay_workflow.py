"""The opt-in native replay stays source-bound and cannot replace required CI."""

import ast
import hashlib
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
ACCEPTED_SERIAL_JOB_SHA256 = '17ea2816dc74c1a0fd43cd92e9a9a649497995b3e3fb86e6c81042dfcdd15bde'
ACCEPTED_SERIAL_STEPS_SHA256 = 'b92bd79417c8328eca3233c1d51c522bec9e9aa87b9a2242f749e9ef9c5983ef'


def _workflow():
    """Read the dispatch definition without YAML 1.1 coercing the `on` key."""
    yaml = pytest.importorskip('yaml')
    return yaml.load((REPO / '.github/workflows/qt-native-replay.yml').read_text(),
                     Loader=yaml.BaseLoader)


def _serial_workflow():
    """Read the registered serial workflow using the same YAML key policy."""
    yaml = pytest.importorskip('yaml')
    return yaml.load((REPO / '.github/workflows/qt-serial-acceptance.yml').read_text(),
                     Loader=yaml.BaseLoader)


def _stable_sha256(value):
    """Hash a parsed job or step list without depending on YAML formatting."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _evaluate_condition(expression, *, event, replay):
    """Evaluate the narrow Boolean routing vocabulary used by these jobs."""
    body = expression.replace('github.event_name', repr(event))
    body = body.replace('inputs.native_batch_replay', repr(replay))
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
    assert _stable_sha256(original) == ACCEPTED_SERIAL_JOB_SHA256
    assert _stable_sha256(serial['steps']) == ACCEPTED_SERIAL_STEPS_SHA256
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
    assert set(workflow['jobs']) == {'replay'}
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


def test_two_attempt_shell_stops_on_either_failure_and_separates_data(tmp_path):
    """A failed first or second replay keeps its failure and original receipt."""
    job = _workflow()['jobs']['replay']
    steps = {step['name']: step for step in job['steps'] if 'name' in step}
    command = steps['Replay the original-order coverage batch twice independently']['run']
    assert command.count('for attempt in 1 2; do') == 1
    loop = 'for attempt in 1 2; do' + command.split('for attempt in 1 2; do', 1)[1]
    script = '''set -euo pipefail
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
        calls = (tmp_path / 'calls.txt').read_text().splitlines()
        assert calls == [str(tmp_path / 'coverage' / f'attempt-{number}')
                         for number in range(1, expected_calls + 1)]
        events = (receipt / 'attempts.txt').read_text().splitlines()
        assert events[0] == f'attempt=1 status=started source={REPLAY_SOURCE}'
        assert events.count(f'attempt=1 status=success source={REPLAY_SOURCE}') == (
            0 if failing == 'attempt-1' else 1)
        assert events.count(f'attempt=2 status=success source={REPLAY_SOURCE}') == (
            1 if failing == 'none' else 0)
        failures = [event for event in events if 'status=failed' in event]
        assert failures == ([] if failing == 'none' else [
            f'attempt={failing[-1]} status=failed exit=7 source={REPLAY_SOURCE}'])
