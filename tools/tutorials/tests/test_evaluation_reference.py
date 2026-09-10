from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation_reference import binary_reference, check_numbers, verify_saved, verify_gui_matrix
from build_evaluation_example import canonical_identity
from evaluation_reference import verify_gui_predictions, verify_error_lists


def example():
    return [dict(true_label=y, predicted_label=p, prob_class_0=a, prob_class_1=b)
        for y, p, a, b in [(0, 0, .9, .1), (0, 1, .3, .7), (1, 1, .1, .9), (1, 1, .35, .65)]]


def test_independent_arithmetic_and_positive_saved_and_gui_counterparts():
    rows = example()
    result = binary_reference(rows)
    assert result['confusion'] == [[1, 1], [0, 2]]
    assert result['summary']['accuracy'] == .75
    assert result['summary']['f1_macro'] == pytest.approx((2/3 + .8)/2)
    assert result['summary']['log_loss'] == pytest.approx(-sum(map(math.log, [.9, .3, .9, .65]))/4)
    assert verify_saved(rows, result['summary'], result['calibration'])['passed']
    table = dict(columns=['true_class', 'infected_1', 'infected_2'], rows=[['infected_2','0','2'],['infected_1','1','1']])
    assert verify_gui_matrix(table, result)['cells'] == 4
    table['rows'][1][1] = '2'
    with pytest.raises(ValueError, match='numeric'):
        verify_gui_matrix(table, result)


@pytest.mark.parametrize('field', ['accuracy','f1_macro','expected_calibration_error','brier_multiclass','n'])
def test_changed_summary_is_detected_after_good_data(field):
    rows=example(); r=binary_reference(rows)
    verify_saved(rows,r['summary'],r['calibration'])
    changed=deepcopy(r['summary']); changed[field] += .1
    with pytest.raises(ValueError,match='numeric'):
        verify_saved(rows,changed,r['calibration'])


@pytest.mark.parametrize('change', ['missing','duplicate','value'])
def test_changed_calibration_is_detected_after_good_data(change):
    rows=example(); r=binary_reference(rows)
    verify_saved(rows,r['summary'],r['calibration'])
    values=deepcopy(r['calibration'])
    if change=='missing': values.pop()
    elif change=='duplicate': values.append(deepcopy(values[0]))
    else: values[0]['n'] += 1
    with pytest.raises(ValueError):
        verify_saved(rows,r['summary'],values)


@pytest.mark.parametrize('patch', [dict(true_label=2),dict(prob_class_0=-1),dict(prob_class_1=float('nan')),
                                   dict(prob_class_0=0,prob_class_1=0),dict(predicted_label=1)])
def test_bad_prediction_has_a_working_positive_counterpart(patch):
    rows=example();binary_reference(rows);rows[0].update(patch)
    expected = 'argmax' if 'predicted_label' in patch else 'Invalid binary'
    with pytest.raises(ValueError, match=expected): binary_reference(rows)


def test_database_identity_is_not_guessed_from_legacy_filename_or_traversal():
    record=dict(plateID='plate1',rowID='r5',columnID='c2',fieldID='f18',cell_id='o9',prcfo='plate1_r5_c2_f18_o9')
    assert canonical_identity(record)=='plate1_r5_c2_f18_o9'
    record['prcfo']='plate1_r5_c2_f18_o10'
    with pytest.raises(ValueError): canonical_identity(record)
    record.update(plateID='../plate1',prcfo='../plate1_r5_c2_f18_o9')
    with pytest.raises(ValueError): canonical_identity(record)


def test_formatted_gui_numbers_and_nonfinite_values():
    check_numbers({'speed':'0.9188'},{'speed':.9188034188},formatted=True)
    with pytest.raises(ValueError): check_numbers({'speed':'nan'},{'speed':.9188034188},formatted=True)


def saved_predictions():
    return [dict(basename=f'object_{i}.png', true_class='infected_1', predicted_class='infected_2',
                 confidence=str(value), well=well, true_label='0', predicted_label='1', correct='False')
            for i,value,well in [(0,.9543,'plate1_r5_c2'),(1,.63,'plate1_r5_c2'),(2,.866,'plate1_r12_c2')]]


def gui_table(rows):
    return dict(columns=list(rows[0]),rows=[[str(v) for v in r.values()] for r in rows])


def test_filter_has_exact_rows_and_restores_unchanged_table():
    rows=saved_predictions();table=gui_table(rows)
    assert verify_gui_predictions(table,rows)['checked_cells']==24
    assert verify_gui_predictions(gui_table(rows[:2]),rows,'r5_c2')['rows']==2
    assert verify_gui_predictions(gui_table(rows[1:2]),rows,'r5_c2 object_1')['rows']==1
    assert verify_gui_predictions(table,rows)['rows']==3
    with pytest.raises(ValueError,match='identities'):
        verify_gui_predictions(table,rows,'r5_c2')


@pytest.mark.parametrize('change', ['columns','duplicate','missing_cell','text','number'])
def test_prediction_table_changes_go_red_after_positive_counterpart(change):
    rows=saved_predictions();table=gui_table(rows)
    verify_gui_predictions(table,rows)
    if change=='columns': table['columns'][1]='wrong'
    elif change=='duplicate':table['rows'][1]=table['rows'][0][:]
    elif change=='missing_cell':table['rows'][0].pop()
    elif change=='text':table['rows'][0][1]='wrong'
    else:table['rows'][0][3]='0.2'
    with pytest.raises(ValueError):verify_gui_predictions(table,rows)


def test_confidence_threshold_repartitions_same_errors_without_reclassification():
    rows=saved_predictions()
    initial=dict(threshold=.75,high=['0.954  object_0.png','0.866  object_2.png'],low=['0.630  object_1.png'])
    assert verify_error_lists(initial,rows)['high']==2
    initial['threshold']=.866
    assert verify_error_lists(initial,rows)['high']==2
    raised=dict(threshold=.95,high=['0.954  object_0.png'],low=['0.630  object_1.png','0.866  object_2.png'])
    assert verify_error_lists(raised,rows)['low']==2
    raised['threshold']=1
    with pytest.raises(ValueError):verify_error_lists(raised,rows)
    raised.update(high=['(none)'],low=['0.630  object_1.png','0.866  object_2.png','0.954  object_0.png'])
    assert verify_error_lists(raised,rows)['high']==0


@pytest.mark.parametrize('change',['order','rounding','count','identity'])
def test_wrong_confidence_list_does_not_pass(change):
    rows=saved_predictions()
    s=dict(threshold=.75,high=['0.954  object_0.png','0.866  object_2.png'],low=['0.630  object_1.png'])
    verify_error_lists(s,rows)
    if change=='order':s['high'].reverse()
    elif change=='rounding':s['low']=['0.620  object_1.png']
    elif change=='count':s['high'].pop()
    else:s['low']=['0.630  unknown.png']
    with pytest.raises(ValueError):verify_error_lists(s,rows)
