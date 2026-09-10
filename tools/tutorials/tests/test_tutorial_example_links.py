"""Exercise the actual player filename filter, not a second Python regex."""
import csv
import json
from pathlib import Path
import subprocess

import pytest

REPO=Path(__file__).resolve().parents[3]
PLAYER=REPO/'docs/source/_extra/tutorials/app_v2.js'


def names(value):
    source=PLAYER.read_text()
    function=source[source.index('function tutorialExampleFiles('):source.index('\nfunction updateGuide(')]
    script=function+'\nconsole.log(JSON.stringify(tutorialExampleFiles(JSON.parse(process.argv[1]))));'
    result=subprocess.run(['node','-e',script,json.dumps(value)],text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def test_existing_lessons_without_examples_stay_unchanged():
    assert names({})==[] and names({'example_files':'not an array'})==[]


@pytest.mark.parametrize('bad',['../outside.csv','/root.csv','https://example.org/a.csv','a.csv?query',
    'a.csv#fragment','a.csv/child','<script>.csv','x'*129+'.csv',None,12])
def test_unsafe_download_name_after_real_positive(bad):
    good='SYNTHETIC_dose_response_examples.csv'
    assert names({'example_files':[good,good]})==[good]
    assert names({'example_files':[good,bad]})==[good]


def test_download_is_the_120_row_disclosed_example():
    path=REPO/'docs/source/_extra/tutorials/examples/SYNTHETIC_dose_response_examples.csv'
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==120
    assert all(row['series'].startswith('SYNTHETIC ') for row in rows)
    assert sum(float(row['concentration'])==0 for row in rows)==12
