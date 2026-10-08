"""The opt-in native replay stays source-bound and cannot replace required CI."""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPLAY_SOURCE = '7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab'
DIAGNOSTIC_SOURCE = '59a6daebab5014c3ee4094791de8ed257cd9af8d'
ORIGINAL_BATCH_SHA256 = 'd5f347bdb25b692cf207f18526b333d01e58a952a923842881367ad77ff65826'


def _workflow():
    """Read the dispatch definition without YAML 1.1 coercing the `on` key."""
    yaml = pytest.importorskip('yaml')
    return yaml.load((REPO / '.github/workflows/qt-native-replay.yml').read_text(),
                     Loader=yaml.BaseLoader)


def test_native_replay_is_opt_in_source_bound_and_one_original_batch():
    workflow = _workflow()
    assert workflow['on'] == 'workflow_dispatch'
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
