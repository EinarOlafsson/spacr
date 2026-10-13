"""Replay isolation checks; no GUI, fit, or existing tutorial data is used."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from capture_parameter_sweep import prepare


def make_stage(root):
    inputs=root/'originals';inputs.mkdir()
    pairs=[]
    for i in range(4):
        pair={}
        for kind in ('score','count'):
            path=inputs/f'{kind}_{i}.csv';path.write_text('value\n1\n')
            pair[kind]=str(path)
        pairs.append(pair)
    capture=root/'captures/regression_release';capture.mkdir(parents=True)
    (capture/'batch_settings.json').write_text(json.dumps({'paired_data':pairs}))
    work,_,_=prepare(root)
    (work/'trials').mkdir();(work/'trials/sweep_results.csv').write_text('trial_id\n1\n2\n')
    return work


def test_existing_replay_preserves_all_bytes(tmp_path):
    work=make_stage(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    selected,settings,hashes=prepare(tmp_path,work)
    assert selected==work and len(settings['paired_data'])==4 and len(hashes)==8
    assert before=={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_existing_outside_directory_is_rejected_after_real_positive(tmp_path):
    work=make_stage(tmp_path);assert prepare(tmp_path,work)[0]==work
    outside=tmp_path/'outside';outside.mkdir();(outside/'trials').mkdir()
    (outside/'trials/sweep_results.csv').write_text('trial_id\n1\n2\n')
    # The target really exists and carries a results file: not a missing-file test.
    with pytest.raises(ValueError,match='existing private tutorial sweep'):prepare(tmp_path,outside)


def test_changed_private_input_is_rejected_not_repaired(tmp_path):
    work=make_stage(tmp_path);assert prepare(tmp_path,work)[0]==work
    path=work/'inputs/score_0.csv';path.write_text('value\n999\n')
    with pytest.raises(ValueError,match='Private sweep input differs'):prepare(tmp_path,work)
    assert path.read_text()=='value\n999\n'


def test_missing_saved_results_is_not_a_new_fit(tmp_path):
    work=make_stage(tmp_path);assert prepare(tmp_path,work)[0]==work
    (work/'trials/sweep_results.csv').unlink()
    with pytest.raises(ValueError,match='existing private tutorial sweep'):prepare(tmp_path,work)


@pytest.fixture
def relocated_stage(tmp_path):
    import shutil

    original = tmp_path / 'original'
    original.mkdir()
    make_stage(original)
    manifest = original / 'captures/regression_release/batch_settings.json'
    settings = json.loads(manifest.read_text())
    settings['src'] = str(original / 'regression_runs/completed-example')
    manifest.write_text(json.dumps(settings))
    stage = tmp_path / 'neutral-stage'
    capture = stage / 'captures/regression_release'
    capture.mkdir(parents=True)
    shutil.copy2(manifest, capture / manifest.name)
    root = stage / 'retained_inputs'
    shutil.copytree(original / 'originals', root / 'originals')
    return stage, root, original, settings


def test_relocated_sweep_uses_exact_eight_inputs_without_changing_manifest(relocated_stage):
    stage, root, original, baseline = relocated_stage
    manifest = stage / 'captures/regression_release/batch_settings.json'
    before = manifest.read_bytes()
    original_files = {path: path.read_bytes() for path in original.rglob('*') if path.is_file()}
    work, settings, hashes = prepare(stage, input_root=root)
    assert manifest.read_bytes() == before
    assert len(hashes) == 8 and len(list((work / 'inputs').iterdir())) == 8
    for before_pair, after_pair in zip(baseline['paired_data'], settings['paired_data']):
        for key in ('score', 'count'):
            copied = Path(after_pair[key])
            assert copied.is_relative_to(work / 'inputs')
            assert copied.read_bytes() == Path(before_pair[key]).read_bytes()
    assert original_files == {path: path.read_bytes() for path in original.rglob('*') if path.is_file()}


@pytest.mark.parametrize('failure', ['changed', 'missing', 'symlink', 'source-escape', 'collision', 'root-escape'])
def test_relocated_sweep_refuses_unsafe_or_changed_inputs(relocated_stage, tmp_path, failure):
    stage, root, original, settings = relocated_stage
    copied = root / 'originals/score_0.csv'
    if failure == 'changed':
        copied.write_text('changed\n')
    elif failure == 'missing':
        copied.unlink()
    elif failure == 'symlink':
        copied.unlink()
        copied.symlink_to(original / 'originals/score_0.csv')
    elif failure == 'source-escape':
        outside = tmp_path / 'outside.csv'
        outside.write_text('value\n1\n')
        settings['paired_data'][0]['score'] = str(outside)
    elif failure == 'collision':
        settings['paired_data'][1]['score'] = settings['paired_data'][0]['score']
    else:
        root = tmp_path
    manifest = stage / 'captures/regression_release/batch_settings.json'
    manifest.write_text(json.dumps(settings))
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='differs|private input root|original workspace|collide|inside the private stage'):
        prepare(stage, input_root=root)
    assert manifest.read_bytes() == before
    assert not (stage / 'sweep_runs').exists()
