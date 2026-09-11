from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_compare_example import relocate_record,require_same_identity


def example():
    return dict(kind='measurements-db',module='measure',status='complete',
        fingerprint_method='sha256',fingerprint='a'*64,size_bytes=100,
        project='/historical/project',path='/historical/project/measurements.db',
        extra_json='{}',run_id='real-recorded-id',spacr_version='1.5.0.5',
        created_ns=123,created_utc='historical-time',settings_json='{"normalize":false}',
        settings_hash='unchanged-settings-digest',artifact_id='original-artifact-id')


def test_relocation_changes_addresses_not_history():
    source=example();copy=deepcopy(source)
    relocated=relocate_record(source,'a'*64,100,Path('/new'),Path('/new/snapshot.db'),Path(__file__))
    assert source==copy
    assert {k for k in relocated if relocated[k]!=source[k]}=={'project','path','extra_json'}
    assert relocated['project']=='/new' and relocated['path']=='/new/snapshot.db'
    extra=json.loads(relocated['extra_json'])['tutorial_relocation']
    assert extra['original_project']==source['project'] and extra['original_path']==source['path']
    assert len(extra['original_registry_sha256'])==64


def test_stale_or_wrong_output_rejected_after_valid_transport():
    source=example()
    assert relocate_record(source,'a'*64,100,Path('/new'),Path('/new/db'),Path(__file__))['run_id']==source['run_id']
    for key,value in [('fingerprint','b'*64),('size_bytes',99),('status','partial'),
                      ('module','mask'),('kind','model-weights'),('fingerprint_method','sampled')]:
        broken=deepcopy(source);broken[key]=value
        with pytest.raises(ValueError,match='original full fingerprint'):
            relocate_record(broken,'a'*64,100,Path('/new'),Path('/new/db'),Path(__file__))


def test_equal_counts_do_not_hide_different_objects_or_fields():
    a=dict(counts={'cell':1},object_keys={'cell':[('plate1','r1','c1','f1','1')]},
           fields=[('plate1','r1','c1','f1')])
    require_same_identity(a,deepcopy(a))
    for key,value in [('object_keys',{'cell':[('plate1','r1','c1','f1','2')]}),
                      ('fields',[('plate2','r1','c1','f1')])]:
        b=deepcopy(a);b[key]=value
        with pytest.raises(ValueError,match='same object and field'):
            require_same_identity(a,b)
    b=deepcopy(a);b['counts']['cell']=2
    with pytest.raises(ValueError,match='unchanged counts'):require_same_identity(a,b)
