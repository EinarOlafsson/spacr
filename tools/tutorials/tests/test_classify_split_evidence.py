from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classify_split_evidence import check_native_audit, check_predictions


def audit():
    return dict(passed=True, group_by='well', train_samples=32, validation_samples=32,
                critical_levels=[], hash_errors=[], unverifiable_counts={},
                overlap_counts={k: 0 for k in ('well', 'field', 'object', 'exact', 'content_sha256', 'augmentation_family')})


def predictions():
    truth = {'a.png': 0, 'b.png': 1}
    rows = [dict(filename=name, true_label=str(y), predicted_label='0',
                 prob_class_0='0.7', prob_class_1='0.3') for name, y in truth.items()]
    metric = dict(accuracy='0.5', Accuracy='0.5', f1_macro=str(1/3), class_support='[1, 1]')
    return rows, metric, truth


def test_positive_native_and_independent_counterparts():
    check_native_audit(audit(), (32, 32))
    proof = check_predictions(*predictions())
    assert proof['confusion_matrix'] == [[1, 0], [1, 0]]
    assert proof['accuracy'] == proof['majority_accuracy'] == .5


@pytest.mark.parametrize('kind', ['passed', 'group', 'count', 'well_overlap', 'unknown_identity'])
def test_named_audit_defects(kind):
    value = audit()
    if kind == 'passed':
        value['passed'] = False
    elif kind == 'group':
        value['group_by'] = 'cell'
    elif kind == 'count':
        value['train_samples'] = 31
    elif kind == 'well_overlap':
        value['overlap_counts']['well'] = 1
    else:
        value['unverifiable_counts'] = {'well': 1}
    with pytest.raises(ValueError):
        check_native_audit(value, (32, 32))


@pytest.mark.parametrize('kind', ['label', 'duplicate', 'probability', 'argmax', 'accuracy', 'f1', 'support'])
def test_named_saved_evaluation_defects(kind):
    rows, metric, truth = deepcopy(predictions())
    if kind == 'label':
        rows[0]['true_label'] = '1'
    elif kind == 'duplicate':
        rows[1]['filename'] = rows[0]['filename']
    elif kind == 'probability':
        rows[0]['prob_class_0'] = '.9'
    elif kind == 'argmax':
        rows[0]['predicted_label'] = '1'
    elif kind == 'accuracy':
        metric['accuracy'] = '.8'
    elif kind == 'f1':
        metric['f1_macro'] = '.8'
    else:
        metric['class_support'] = '[0, 2]'
    with pytest.raises(ValueError):
        check_predictions(rows, metric, truth)
