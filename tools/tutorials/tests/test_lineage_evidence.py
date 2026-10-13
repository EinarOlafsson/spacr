"""Positive and corrupted-input counterparts for the containment oracle."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('lineage_evidence',Path(__file__).resolve().parents[1]/'lineage_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(field,number,parent=None):
    return dict(plateID='plate1',rowID='r1',columnID='c1',fieldID=field,
                object_label=number,cell_id=parent)


@pytest.fixture
def expected():
    # Same label across fields and types, an existing same-label parent in
    # another field, a genuinely absent parent, an unassigned child, and a
    # childless root. Synthetic inputs ONLY test the independent helper.
    return m.expected_from_rows(dict(cell=[row('f1',1),row('f2',1),row('f2',2)],
        nucleus=[row('f1',1,1),row('f2',1,1),row('f3',1,1),row('f2',2,None)],
        pathogen=[row('f1',1,1),row('f2',3,9)]))


def test_positive_actual_fields_types_orphans_and_childless(expected):
    result=m.verify_forest(expected['rows'],expected['orphans'],expected)
    assert result==dict(source_counts={'cell':3,'nucleus':4,'pathogen':2},full_nodes=6,
                       attached_counts={'nucleus':2,'pathogen':1},roots=3,orphans=3,childless=1,no_pathogens=2)
    assert len(set(r['key'] for r in expected['rows']))==6
    assert expected['families']['plate1_r1_c1_f1_cell1']==[
        'plate1_r1_c1_f1_cell1','plate1_r1_c1_f1_nucleus1','plate1_r1_c1_f1_pathogen1']


@pytest.mark.parametrize('change',['key','parent','order','count','dropped'])
def test_corrupted_forest_is_rejected(expected,change):
    observed=deepcopy(expected['rows'])
    if change=='key':observed[1]['key']=observed[2]['key']
    if change=='parent':observed[1]['parent_key']='plate1_r1_c1_f2_cell1'
    if change=='order':observed.reverse()
    if change=='count':observed[0]['n_children']=0
    if change=='dropped':observed.pop()
    with pytest.raises(ValueError,match='full forest'):
        m.verify_forest(observed,expected['orphans'],expected)


@pytest.mark.parametrize('change',['dropped','invented','parent'])
def test_orphan_positive_and_absence_are_distinct(expected,change):
    observed=deepcopy(expected['orphans'])
    if change=='dropped':observed=[]
    if change=='invented':observed.append(deepcopy(observed[0]))
    if change=='parent':observed[0]['parent_id']='999'
    with pytest.raises(ValueError,match='orphan'):
        m.verify_forest(expected['rows'],observed,expected)


def test_display_cap_keeps_family_intact_and_sort_only_changes_order(expected):
    shown=[r for r in expected['rows'] if r['key']==expected['roots'][0] or r['parent_key']==expected['roots'][0]]
    result=m.verify_tree(list(reversed(shown)),expected,limit=1)
    assert result['displayed_roots']==1 and result['displayed_nodes']==3 and result['omitted_roots']==2


@pytest.mark.parametrize('change',['extra','missing','repeat','parent','type'])
def test_tree_guards(expected,change):
    observed=deepcopy(expected['rows'])
    if change=='extra':observed.append(deepcopy(observed[0]))
    if change=='missing':observed.pop()
    if change=='repeat':observed[1]=deepcopy(observed[2])
    if change=='parent':observed[1]['parent_key']='plate1_r1_c1_f2_cell1'
    if change=='type':observed[1]['table']='pathogen'
    with pytest.raises(ValueError,match='Actual tree'):
        m.verify_tree(observed,expected)


@pytest.mark.parametrize('bad',[[],['a'],['b','a'],['a','b','b']])
def test_real_selection_counterparts(bad):
    assert m.verify_selected(['a','b'],['a','b'])==['a','b']
    with pytest.raises(ValueError,match='Published'):
        m.verify_selected(bad,['a','b'])


@pytest.mark.parametrize('status,present',[('Open the Annotate screen first — it is what shows crops.',True),('Select something in the tree first.',False)])
def test_absent_opener_not_wrong_guard(status,present):
    assert not m.verify_unavailable_crop('Open the Annotate screen first — it is what shows crops.',False)['crop_opened']
    with pytest.raises(ValueError,match='unavailable-crop'):
        m.verify_unavailable_crop(status,present)
