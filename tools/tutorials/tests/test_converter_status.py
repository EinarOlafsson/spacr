"""A resumed conversion can have an empty ledger, but not empty evidence."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_converter import check_converter_run_status


def entry(status='complete', run_id='first'):
    attempted = 2 if status == 'complete' else 0
    return {'run_id': run_id, 'name': 'convert_to_yokogawa_plan', 'status': status,
            'n_attempted': attempted, 'n_succeeded': attempted, 'n_failed': 0,
            'failures': [], 'success_by_stage': {'convert': 2} if attempted else {}}


def inputs(operation='convert'):
    first = entry()
    return {
        'records': [first] if operation == 'convert' else [first, entry('empty', 'second')],
        'operation': operation, 'planned_targets': ['one.tif', 'two.tif'],
        'verified_outputs': {'one.tif': {'all_pixels_exact': True},
                             'two.tif': {'all_pixels_exact': True}},
        'planned_fields': ['plate1/A01/f0001', 'plate1/A01/f0002'],
        'resumed_fields': [] if operation == 'convert' else ['plate1/A01/f0002', 'plate1/A01/f0001'],
        'n_sources': 2, 'n_written': 2 if operation == 'convert' else 0,
        'n_existing': 0 if operation == 'convert' else 2,
        'previous_records': [] if operation == 'convert' else [deepcopy(first)],
    }


@pytest.mark.parametrize('operation,status', [('convert', 'complete'), ('resume', 'empty')])
def test_exact_new_write_or_verified_resume_is_accepted_without_mutation(operation, status):
    args = inputs(operation)
    before = deepcopy(args)
    result = check_converter_run_status(**args)
    assert result['operation'] == operation
    assert result['ledger_status'] == status
    assert result['independently_verified_outputs'] == 2
    assert args == before


def test_another_resume_checks_the_new_operation_and_preserves_earlier_empty_record():
    args = inputs('resume')
    args['previous_records'] = deepcopy(args['records'])
    args['records'].append(entry('empty', 'third'))
    assert check_converter_run_status(**args)['run_id'] == 'third'


@pytest.mark.parametrize('case', ['missing', 'wrong_identity', 'not_verified'])
def test_empty_resume_requires_independently_verified_outputs(case):
    args = inputs('resume')
    if case == 'missing':
        del args['verified_outputs']['two.tif']
    elif case == 'wrong_identity':
        args['verified_outputs']['other.tif'] = args['verified_outputs'].pop('two.tif')
    else:
        args['verified_outputs']['two.tif']['all_pixels_exact'] = False
    with pytest.raises(ValueError, match='independently verified'):
        check_converter_run_status(**args)


@pytest.mark.parametrize('key,value', [('n_written', 1), ('n_existing', 1),
                                      ('resumed_fields', ['plate1/A01/f0001']),
                                      ('resumed_fields', ['plate1/A01/f0001', 'wrong']),
                                      ('resumed_fields', ['plate1/A01/f0001'] * 2)])
def test_empty_resume_requires_exact_reuse_counts_and_field_identities(key, value):
    args = inputs('resume')
    args[key] = value
    with pytest.raises(ValueError, match='verified reuse'):
        check_converter_run_status(**args)


@pytest.mark.parametrize('operation', ['convert', 'resume'])
@pytest.mark.parametrize('key,value', [('n_attempted', 1), ('n_succeeded', 1),
                                      ('n_failed', 1), ('status', 'failed'),
                                      ('failures', ['a failed source'])])
def test_ledger_counts_and_status_must_match_expected_operation(operation, key, value):
    args = inputs(operation)
    args['records'][-1][key] = value
    with pytest.raises(ValueError):
        check_converter_run_status(**args)


def test_arbitrary_empty_conversion_is_not_success():
    args = inputs()
    args['records'] = [entry('empty')]
    args['n_written'] = 0
    with pytest.raises(ValueError, match='new conversions'):
        check_converter_run_status(**args)


def test_complete_record_cannot_be_mistaken_for_the_expected_empty_resume():
    args = inputs('resume')
    args['records'][-1] = entry('complete', 'second')
    with pytest.raises(ValueError, match='verified reuse'):
        check_converter_run_status(**args)


@pytest.mark.parametrize('change', ['missing_append', 'extra_append', 'rewritten_prior', 'reused_run_id'])
def test_latest_operation_must_be_new_and_preserve_ledger_history(change):
    args = inputs('resume')
    if change == 'missing_append':
        args['records'].pop()
    elif change == 'extra_append':
        args['records'].append(entry('empty', 'third'))
    elif change == 'rewritten_prior':
        args['records'][0]['n_succeeded'] = 1
    else:
        args['records'][-1]['run_id'] = 'first'
    with pytest.raises(ValueError):
        check_converter_run_status(**args)


def test_other_workflow_cannot_supply_the_latest_status():
    args = inputs()
    args['records'][0]['name'] = 'foreign_import'
    with pytest.raises(ValueError, match='not the converter'):
        check_converter_run_status(**args)
