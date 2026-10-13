"""Regression previews describe paired tables and treat src as output only."""
from copy import deepcopy
import csv
import json
import os
import subprocess
import sys

import pytest

from spacr import validate


def test_paired_plan_preserves_rows_aliases_and_optional_database_without_scanning(monkeypatch):
    settings = dict(src='/output/run', dependent_variable='phenotype', paired_data=[
        dict(score='/inputs/s1.csv', count='/inputs/c1.csv', plate='plate1'),
        dict(score_data='/inputs/s2.csv', count_data='/inputs/c2.csv', measurements='/inputs/linked.db'),
    ])
    original = deepcopy(settings)
    monkeypatch.setattr(validate, '_inventory', lambda *args: pytest.fail('output root is not an input dataset'))
    plan = validate.describe_plan(settings, 'regression')
    for path in ('/inputs/s1.csv', '/inputs/c1.csv', '/inputs/s2.csv', '/inputs/c2.csv', '/inputs/linked.db'):
        assert path in plan
    assert 'pair 1 score' in plan and 'pair 2 count' in plan
    assert 'output root' in plan and '/output/run' in plan
    assert 'phenotype' in plan
    assert 'no image files found' not in plan and 'measurements/measurements.db' not in plan
    assert settings == original


def test_legacy_plan_exposes_positional_pairing_and_an_unmatched_row():
    plan = validate.describe_plan(dict(score_data=['s1.csv', 's2.csv'], count_data='c1.csv'), 'regression')
    assert 'paired by position' in plan
    assert 'pair 2 count' in plan
    assert 'not set' in next(line for line in plan.splitlines() if 'pair 2 count' in line)
    assert 'resolved from the paired inputs at run time' in plan


@pytest.mark.parametrize('pairs', [None, [], 'bad', ['bad']])
def test_incomplete_plan_describes_the_problem_without_crashing(pairs):
    plan = validate.describe_plan({'paired_data': pairs}, 'regression')
    assert 'invalid' in plan or 'no paired_data' in plan


def test_explicit_pairs_take_precedence_over_legacy_paths():
    plan = validate.describe_plan(dict(paired_data=[dict(score='new.csv', count='counts.csv')],
                                      score_data='old.csv'), 'regression')
    assert 'new.csv' in plan and 'old.csv' not in plan


def test_real_cli_dry_run_reads_no_tables_creates_no_output_and_imports_no_model(tmp_path):
    scores = tmp_path / 'scores.csv'
    counts = tmp_path / 'counts.csv'
    scores.write_text('plateID,rowID,columnID,pred\np1,r1,c1,0.5\n')
    counts.write_text('plateID,rowID,columnID,grna,count\np1,r1,c1,g1,50\n')
    output = tmp_path / 'new-output'
    settings_file = tmp_path / 'settings.csv'
    with settings_file.open('w', newline='') as handle:
        csv.writer(handle).writerows([['setting_key', 'setting_value'], ['src', str(output)]])
    arguments = ['regression', '--settings', str(settings_file), '--dry-run', '--set', 'src=' + str(output),
                 '--set', 'paired_data=' + json.dumps([dict(score=str(scores), count=str(counts))]),
                 '--set', 'dependent_variable=pred']
    script = (
        'import sys\nfrom spacr.cli import main\n'
        f'assert main({arguments!r}) == 0\n'
        "assert 'spacr.ml' not in sys.modules\n"
        "assert 'torch' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, '-c', script], text=True,
                            capture_output=True, timeout=45,
                            env=dict(os.environ, HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / 'config')))
    assert result.returncode == 0, result.stdout + result.stderr
    plan = result.stdout.split('Plan —', 1)[1]
    assert str(scores) in plan and str(counts) in plan
    assert 'measurements/measurements.db' not in plan
    assert not output.exists()
