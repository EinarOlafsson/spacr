"""A catalog prerequisite failure must never leave a runnable partial queue."""
from pathlib import Path
import json
import shlex
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_host_jobs import QueuePreflightError, validate_run, write_jobs

TUTORIALS = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path):
    repo = tmp_path / 'repo'
    tutorials = repo / 'tools/tutorials'
    (tutorials / 'lessons').mkdir(parents=True)
    host = tmp_path / 'host'
    (host / 'plans').mkdir(parents=True)
    proto = tmp_path / 'proto'
    (proto / 'idle_retest').mkdir(parents=True)
    (proto / 'idle_retest/live.sh').write_text('#!/bin/sh\n')
    queue = tmp_path / 'queue'
    queue.mkdir()
    return repo, tutorials, host, proto, queue


def put_lesson(workspace, ident, runs, *, catalog='a'):
    _repo, tutorials, host, _proto, _queue = workspace
    (tutorials / 'lessons' / f'{ident}.json').write_text(json.dumps({
        'scenes': [{'visual': 'open'}, {'visual': 'edit'}, {'visual': 'open'}],
    }))
    (host / 'plans' / f'{ident}-clips.json').write_text(json.dumps({'open': 8, 'edit': 8}))
    path = tutorials / f'host_jobs_{catalog}.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data[ident] = runs
    path.write_text(json.dumps(data))


def generate(workspace, *lessons):
    repo, _tutorials, host, proto, queue = workspace
    return write_jobs(queue, 'a', 'r-test', lessons, repo=repo, host_root=host,
                      prototype_root=proto, machine_root=host.parent)


@pytest.mark.parametrize('lesson,catalog,option', [
    ('10_classify_cv', 'a', '--classifier-existing-split'),
    ('31_external_masks', 'b', '--external-input-root'),
    ('75_regression_diagnostics', 'd', '--diagnostics-from'),
])
def test_real_catalog_capture_placeholders_block_all_jobs(workspace, lesson, catalog, option):
    runs = json.loads((TUTORIALS / f'host_jobs_{catalog}.json').read_text())[lesson]
    run = next(run for run in runs if option in run.get('capture', []))
    # Isolate the actual overlooked capture token from independent env/native prerequisites.
    put_lesson(workspace, lesson, [dict(name=run['name'], env={}, run_native=[], capture=run['capture'])])
    put_lesson(workspace, '05_valid', [dict(name='valid', capture=['--module', 'home'])])
    with pytest.raises(QueuePreflightError, match='capture: unresolved placeholder'):
        generate(workspace, '05_valid', lesson)
    assert list(workspace[-1].iterdir()) == []


@pytest.mark.parametrize('run,context', [
    ({'run_native': ['--inputs', 'missing-inputs']}, 'run_native --inputs'),
    ({'run_native': ['--ro-bind', 'missing-models=profile/.cellpose']}, 'run_native --ro-bind source'),
    ({'run_native': ['--ro-bind=missing-models=profile/.cellpose']}, 'run_native --ro-bind source'),
    ({'env': {'SPACR_TUTORIAL_MODEL_FIELDS': '/missing-tutorial-fields'}}, 'env SPACR_TUTORIAL_MODEL_FIELDS'),
    ({'capture': ['--module', 'test', '--batch-input-root', 'missing-batch']}, 'capture --batch-input-root'),
])
def test_missing_input_or_model_paths_block_generation(workspace, run, context):
    entry = dict(name='missing', env={}, run_native=[], capture=['--module', 'test'])
    entry.update(run)
    put_lesson(workspace, '20_test', [entry])
    with pytest.raises(QueuePreflightError, match=context):
        generate(workspace, '20_test')
    assert list(workspace[-1].iterdir()) == []


def test_unresolved_env_and_native_tokens_are_also_rejected(tmp_path):
    errors = validate_run({'MODEL': '<model>'}, ['--inputs', '<tree>'], ['--module', 'home'], cwd=tmp_path)
    assert any('env MODEL: unresolved placeholder' in error for error in errors)
    assert any('run_native: unresolved placeholder' in error for error in errors)


def test_existing_mapping_must_point_to_an_existing_input_tree(workspace):
    put_lesson(workspace, '19_train', [dict(name='training', run_native=[
        '--inputs', '<prepared dir containing Cellpose_training_images_masks/ extracted from example>',
    ], capture=['--module', 'train_cellpose'])])
    with pytest.raises(QueuePreflightError, match='missing local path.*cellpose_training'):
        generate(workspace, '19_train')
    assert list(workspace[-1].iterdir()) == []


def test_valid_generation_preserves_catalog_order_scene_selection_and_quoting(workspace):
    repo, _tutorials, host, _proto, _queue = workspace
    tree = repo / 'input with spaces'; tree.mkdir()
    models = repo / 'models'; models.mkdir()
    put_lesson(workspace, '20_valid', [dict(name='first', env={'FIELDS': str(tree)},
        run_native=['--inputs', str(tree), f'--ro-bind={models}=profile/.cellpose'],
        capture=['--module', 'cellpose_masks'])], catalog='a')
    put_lesson(workspace, '20_valid', [dict(name='second', capture=['--module', 'cellpose_masks'])], catalog='f')
    paths = generate(workspace, '20_valid')
    assert [p.name for p in paths] == ['a01.job', 'a02.job']
    first = shlex.split(paths[0].read_text())
    assert first[0] == f'FIELDS={tree}'
    assert first[first.index('--inputs') + 1] == str(tree)
    assert first[first.index('--clip-durations') + 1] == str(host / 'plans/20_valid-clips.json')
    assert first[first.index('--record-scenes') + 1] == 'open,edit'
    assert str(host / 'r-test-20-first') in first
    assert str(host / 'r-test-20-second') in shlex.split(paths[1].read_text())


def test_existing_queue_job_is_not_overwritten(workspace):
    put_lesson(workspace, '20_valid', [dict(name='first', capture=['--module', 'home'])])
    job = workspace[-1] / 'a01.job'; job.write_text('prior authorized command\n')
    with pytest.raises(QueuePreflightError, match='job already exists'):
        generate(workspace, '20_valid')
    assert job.read_text() == 'prior authorized command\n'


def test_explicit_launcher_is_used_without_requiring_default(workspace):
    repo, _tutorials, host, proto, queue = workspace
    launcher = proto / 'run from ch.sh'
    launcher.write_text('#!/bin/sh\n')
    (proto / 'idle_retest/live.sh').unlink()
    put_lesson(workspace, '20_valid', [dict(name='first', capture=['--module', 'home'])])
    paths = write_jobs(queue, 'a', 'r-test', ['20_valid'], repo=repo,
                       host_root=host, prototype_root=proto, launcher=launcher)
    assert shlex.split(paths[0].read_text())[0] == str(launcher)


def test_missing_explicit_launcher_prevents_all_jobs(workspace):
    repo, _tutorials, host, proto, queue = workspace
    put_lesson(workspace, '20_valid', [dict(name='first', capture=['--module', 'home'])])
    with pytest.raises(QueuePreflightError, match='capture launcher does not exist'):
        write_jobs(queue, 'a', 'r-test', ['20_valid'], repo=repo,
                   host_root=host, prototype_root=proto, launcher=proto / 'missing.sh')
    assert list(queue.iterdir()) == []
