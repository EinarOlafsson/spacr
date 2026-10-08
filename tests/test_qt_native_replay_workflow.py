"""The opt-in native replay stays source-bound and cannot replace required CI."""

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = '7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab'
DIAGNOSTIC_SOURCE = '59a6daebab5014c3ee4094791de8ed257cd9af8d'
ORIGINAL_BATCH_SHA256 = 'd5f347bdb25b692cf207f18526b333d01e58a952a923842881367ad77ff65826'
ORIGINAL_SERIAL_JOB_SHA256 = '11555f7cf61de802a12797b7a5f7a4aad8facf55668125b5d4c46b8352d0fa8a'
ORIGINAL_SERIAL_STEPS_SHA256 = '0f6be5d7be8b0a10bd956db94255bf35d6b7587d7e75156170eb0e062f0572d3'


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
    assert _stable_sha256(original) == ORIGINAL_SERIAL_JOB_SHA256
    assert _stable_sha256(serial['steps']) == ORIGINAL_SERIAL_STEPS_SHA256
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


def test_native_replay_is_opt_in_source_bound_and_one_original_batch():
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

    replay = steps['Replay exactly one original-order coverage batch']
    assert 'continue-on-error' not in replay
    command = replay['run']
    assert f'export GITHUB_SHA={REPLAY_SOURCE}' in command
    assert 'tools/run_coverage_batches.py "${batch[@]}"' in command
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
