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
