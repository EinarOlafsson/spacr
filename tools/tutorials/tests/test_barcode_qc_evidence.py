"""Tiny independent fixtures, positive counterparts and corrupt output checks."""
import copy
import csv
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from barcode_qc_evidence import (read_counts, per_well, positions, sweep_row,
                                 compare_rows, verify_outputs, check_native_run)


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


@pytest.fixture
def source(tmp_path):
    # Duplicate entries aggregate to 3+3; the second well has one guide with
    # four reads. At exactly 0.5, all three calls survive (inclusive bound).
    path=tmp_path/'counts.csv'
    rows=[dict(row_name='r1',column_name='c1',grna_name='g1',count=1),
          dict(row_name='r1',column_name='c1',grna_name='g1',count=2),
          dict(row_name='r1',column_name='c1',grna_name='g2',count=3),
          dict(row_name='r1',column_name='c2',grna_name='g1',count=4)]
    write_csv(path,rows)
    return read_counts(path)


def test_count_identity_and_duplicate_aggregation(source):
    assert source['source_rows']==4
    assert source['total_reads']==10
    assert source['wells']=={('r1','c1'):{'g1':3,'g2':3},('r1','c2'):{'g1':4}}
    assert source['guides']=={'g1':7,'g2':3}
    assert per_well(source)==[
        dict(prc='plate1_r1_c1',plateID='plate1',rowID='r1',columnID='c1',reads=6,n_grnas=2),
        dict(prc='plate1_r1_c2',plateID='plate1',rowID='r1',columnID='c2',reads=4,n_grnas=1)]


@pytest.mark.parametrize('kind',['columns','empty','identity','zero','negative'])
def test_invalid_input_after_real_positive(source,tmp_path,kind):
    assert read_counts(source['source'])['total_reads']==10
    rows=[dict(row_name='r1',column_name='c1',grna_name='g',count=2)]
    fields=list(rows[0])
    if kind=='columns':rows=[dict(row_name='r1')];fields=list(rows[0])
    if kind=='empty':rows=[]
    if kind=='identity':rows[0]['grna_name']=' g'
    if kind=='zero':rows[0]['count']=0
    if kind=='negative':rows[0]['count']=-1
    path=tmp_path/'bad.csv';write_csv(path,rows,fields)
    with pytest.raises(ValueError):read_counts(path)


def test_position_medians_and_inclusive_ratio(source):
    result=positions(source,ratio=1.2)
    assert result==[
        dict(plateID='plate1',axis='row',label='r1',n_wells=2,median_reads=5,
             plate_median=5,ratio_to_plate=1,flagged=False),
        dict(plateID='plate1',axis='column',label='c1',n_wells=1,median_reads=6,
             plate_median=5,ratio_to_plate=1.2,flagged=True),
        dict(plateID='plate1',axis='column',label='c2',n_wells=1,median_reads=4,
             plate_median=5,ratio_to_plate=.8,flagged=True)]


def test_threshold_includes_exact_fraction_and_zero_call_wells(source):
    assert sweep_row(source,.5,target=1)==dict(threshold=.5,grnas_per_well=1.5,
        grnas_per_well_retained=1.5,wells_retained=2,well_retention=1.,
        wells_over_budget=1,collision_rate=.5,collision_rate_retained=.5,n_calls=3,reads_retained=1.)
    higher=sweep_row(source,.6,target=1)
    assert higher['grnas_per_well']==.5 and higher['grnas_per_well_retained']==1
    assert higher['wells_retained']==1 and higher['reads_retained']==.4
    empty=sweep_row(source,1.1)
    assert empty['grnas_per_well']==0 and empty['n_calls']==0
    assert math.isnan(empty['grnas_per_well_retained']) and math.isnan(empty['collision_rate_retained'])


def test_no_retained_population_refused_after_positive(source):
    assert sweep_row(source,.5)['wells_retained']==2
    with pytest.raises(ValueError,match='non-starved'):sweep_row(source,.5,starved_fraction=100)


def test_keyed_comparison_ignores_order_not_identity():
    expected=[dict(id='b',value=2,flag=True),dict(id='a',value=float('nan'),flag=False)]
    actual=[dict(id='a',value='',flag='False'),dict(id='b',value='2',flag='True')]
    assert compare_rows(actual,expected,['id'])==dict(rows=2,fields_checked=6,maximum_numeric_error=0.)


@pytest.mark.parametrize('kind',['omit','duplicate','substitute','extra','flag','text','number','missing_number'])
def test_corrupt_output_after_positive(kind):
    expected=[dict(id='a',label='first',value=2,flag=True),dict(id='b',label='second',value=3,flag=False)]
    actual=copy.deepcopy(expected);compare_rows(actual,expected,['id'])
    if kind=='omit':actual.pop()
    if kind=='duplicate':actual.append(actual[0])
    if kind=='substitute':actual[0]['id']='x'
    if kind=='extra':actual.append(dict(id='x',label='extra',value=0,flag=False))
    if kind=='flag':actual[0]['flag']=False
    if kind=='text':actual[0]['label']='wrong'
    if kind=='number':actual[0]['value']=99
    if kind=='missing_number':actual[0]['value']=None
    with pytest.raises(ValueError):compare_rows(actual,expected,['id'])


def output_fixture(folder,source):
    write_csv(folder/'reads_per_well.csv',per_well(source))
    write_csv(folder/'position_effects.csv',positions(source))
    write_csv(folder/'starved_wells.csv',[],list(per_well(source)[0]))
    write_csv(folder/'threshold_sweep.csv',[sweep_row(source,.5),sweep_row(source,.6)])


def test_saved_tables_validate_every_field_without_confusing_sweeps_with_starved(source,tmp_path):
    output_fixture(tmp_path,source)
    result=verify_outputs(tmp_path,source)
    assert result['starved_wells']==0 and result['checks']['sweep']['rows']==2
    assert result['checks']['per_well']['fields_checked']==12
    assert result['checks']['positions']['fields_checked']==24
    assert result['checks']['sweep']['fields_checked']==20
    assert result['source_unchanged'] is True


def test_empty_saved_sweep_after_positive(source,tmp_path):
    output_fixture(tmp_path,source);verify_outputs(tmp_path,source)
    write_csv(tmp_path/'threshold_sweep.csv',[],list(sweep_row(source,.5)))
    with pytest.raises(ValueError,match='No actual'):verify_outputs(tmp_path,source)


def test_changed_input_after_positive(source,tmp_path):
    output_fixture(tmp_path,source);verify_outputs(tmp_path,source)
    with Path(source['source']).open('a') as stream:stream.write('r2,c1,g1,3\n')
    with pytest.raises(ValueError,match='input changed'):verify_outputs(tmp_path,source)


@pytest.mark.parametrize('kind',['unfinished','failed','worker_error','missing_figure','hidden_card','settings_error'])
def test_native_completion_requires_visible_figures_and_clean_preflight(kind):
    run=dict(outcome=dict(finished=True,ok=True,errors=[]),gui_figure_count=2,
             figures_card_visible=True,settings_errors=[])
    check_native_run(run)
    if kind=='unfinished':run['outcome']['finished']=False
    if kind=='failed':run['outcome']['ok']=False
    if kind=='worker_error':run['outcome']['errors']=['failed']
    if kind=='missing_figure':run['gui_figure_count']=1
    if kind=='hidden_card':run['figures_card_visible']=False
    if kind=='settings_error':run['settings_errors']=['[settings] ERROR [src]']
    with pytest.raises(ValueError):check_native_run(run)
