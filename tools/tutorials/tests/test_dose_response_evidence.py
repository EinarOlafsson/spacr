"""Planted truths, positive counterparts and corrupt recording evidence."""
import copy
import csv
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dose_response_evidence import (TRUTH,REVERSAL,synthetic_rows,prepare,close,
    check_fit,check_groups,check_profile,check_wald,check_drawn_curves)


def case(group='SYNTHETIC activation'):
    truth=TRUTH[group];rows=[r for r in synthetic_rows() if r['series']==group and r['concentration']>0]
    record=dict(bottom=truth['bottom'],top=truth['top'],hill=truth['hill'],
        ec50_unconstrained=truth['ec50'],ec50=truth['ec50'],ec50_low=truth['ec50']*.97,
        ec50_high=truth['ec50']*1.03,status='fitted',bound_direction='bounded',
        n_obs=27,n_doses=9,dof=23,n_vehicle=3,n_excluded=0,vehicle_response=truth['vehicle'],
        sse=4.5,rse=math.sqrt(4.5/23),dose=[r['concentration'] for r in rows],
        response=[r['response'] for r in rows])
    if truth['ec50']>100:record.update(status='unbounded',ec50=None,ec50_low=None,ec50_high=None,bound_direction='above')
    return record


def test_generator_is_disclosed_balanced_not_random(tmp_path):
    rows=synthetic_rows();assert len(rows)==120 and rows==synthetic_rows()
    for group,truth in TRUTH.items():
        vehicle=[r['response'] for r in rows if r['series']==group and r['concentration']==0]
        assert vehicle==[truth['vehicle']-.5,truth['vehicle'],truth['vehicle']+.5]
    meta=prepare(tmp_path)
    assert meta['synthetic'] and not meta['random'] and not meta['biological_claim']
    with Path(meta['path']).open(newline='') as stream:assert len(list(csv.DictReader(stream)))==120


@pytest.mark.parametrize('group',list(TRUTH))
def test_constructed_fit_passes_every_planted_value(group):
    result=check_fit(group,case(group));assert result['source_values_checked']==54


@pytest.mark.parametrize('field',['bottom','top','hill','ec50_unconstrained','n_obs','n_doses','dof','n_vehicle',
    'n_excluded','vehicle_response','sse','rse','ec50'])
def test_wrong_native_fit_value_after_positive(field):
    record=case();check_fit('SYNTHETIC activation',record)
    record[field]+=10
    with pytest.raises(ValueError,match='planted truth'):check_fit('SYNTHETIC activation',record)


@pytest.mark.parametrize('bad',[None,float('inf'),float('nan'),-1])
def test_missing_nonfinite_wrong_number_after_positive(bad):
    close(2,2,'test')
    with pytest.raises(ValueError,match='planted truth'):close(bad,2,'test')


@pytest.mark.parametrize('field',['dose','response'])
def test_missing_source_observation_after_positive(field):
    record=case();check_fit('SYNTHETIC activation',record);record[field].pop()
    with pytest.raises(ValueError,match='source rows'):check_fit('SYNTHETIC activation',record)


@pytest.mark.parametrize('field',['dose','response'])
def test_reordered_source_observation_after_positive(field):
    record=case();check_fit('SYNTHETIC activation',record);record[field].reverse()
    with pytest.raises(ValueError,match='planted truth'):check_fit('SYNTHETIC activation',record)


@pytest.mark.parametrize('field,value',[('status','fitted'),('ec50',500),('ec50_low',400),('ec50_high',600),('bound_direction','below')])
def test_unbounded_output_cannot_become_claimed_estimate(field,value):
    group='SYNTHETIC beyond range';record=case(group);check_fit(group,record);record[field]=value
    with pytest.raises(ValueError):check_fit(group,record)


def test_fitted_status_and_interval_refusals_after_positive():
    record=case();check_fit('SYNTHETIC activation',record)
    record['status']='refused'
    with pytest.raises(ValueError,match='bracketed'):check_fit('SYNTHETIC activation',record)
    record=case();record['ec50_low']=3
    with pytest.raises(ValueError,match='inside'):check_fit('SYNTHETIC activation',record)


def groups():
    return {**{name:dict(result=case(name),error='') for name in TRUTH},
            REVERSAL:dict(result=None,error='this series is not monotone')}


@pytest.mark.parametrize('kind',['omit','substitute','fit_reversal','wrong_reason'])
def test_refused_and_unbounded_groups_stay_accounted_for(kind):
    fits=groups();check_groups(fits)
    if kind=='omit':fits.pop(REVERSAL)
    if kind=='substitute':fits['wrong group']=fits.pop(REVERSAL)
    if kind=='fit_reversal':fits[REVERSAL]['result']=case()
    if kind=='wrong_reason':fits[REVERSAL]['error']='not enough concentrations'
    with pytest.raises(ValueError):check_groups(fits)


def test_profile_endpoints_meet_declared_parameter_precision_not_an_sse_threshold():
    record=case();record.update(ec50_low=1.9445672376549057,ec50_high=2.0570129552209315)
    measured=check_profile('SYNTHETIC activation',record)
    assert measured['target_sse']==pytest.approx(5.337263017006562)
    assert measured['endpoints'][0]['independent_midpoint']==pytest.approx(1.9464029823,abs=1e-7)
    assert measured['endpoints'][1]['independent_midpoint']==pytest.approx(2.0554521741,abs=1e-7)
    record['ec50_low']*=.99
    with pytest.raises(ValueError,match='declared log-space'):check_profile('SYNTHETIC activation',record)


def test_wald_interval_uses_analytic_jacobian_in_log_space():
    record=case();record.update(ec50_low=1.9462325545312742,ec50_high=2.055252847786051)
    measured=check_wald('SYNTHETIC activation',record)
    assert measured['independent_endpoints']['ec50_low']==pytest.approx(1.9462325527,abs=1e-9)
    assert measured['independent_endpoints']['ec50_high']==pytest.approx(2.0552528497,abs=1e-9)
    record['ec50_high']+=.01
    with pytest.raises(ValueError,match='planted truth'):check_wald('SYNTHETIC activation',record)


def drawn_case():
    from matplotlib.figure import Figure
    axes=Figure().subplots();axes.set_xscale('log')
    for group,truth in TRUTH.items():
        record=case(group);axes.plot(record['dose'],record['response'],'o',label=group)
        x=np.geomspace(10**-2.5,10**2.5,200)
        y=truth['bottom']+(truth['top']-truth['bottom'])/(1+(truth['ec50']/x)**truth['hill'])
        axes.plot(x,y)
    return axes


@pytest.mark.parametrize('kind',['axis','name','point_count','point_value','curve_count','curve_value'])
def test_actual_drawn_data_cannot_be_substituted_after_positive(kind):
    axes=drawn_case();assert len(check_drawn_curves(axes))==3
    if kind=='axis':axes.set_xscale('linear')
    if kind=='name':axes.lines[0].set_label('not the source group')
    if kind=='point_count':axes.lines[0].set_data(axes.lines[0].get_xdata()[:-1],axes.lines[0].get_ydata()[:-1])
    if kind=='point_value':axes.lines[0].set_ydata(np.zeros(27))
    if kind=='curve_count':axes.lines[1].set_data(axes.lines[1].get_xdata()[:-1],axes.lines[1].get_ydata()[:-1])
    if kind=='curve_value':axes.lines[1].set_ydata(np.zeros(200))
    with pytest.raises(ValueError):check_drawn_curves(axes)
