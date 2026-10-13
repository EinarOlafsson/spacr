"""Reference identity and count checks must fail against plausible wrong data."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('map_tutorial_data',
    Path(__file__).resolve().parents[1] / 'map_barcodes_data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


def csv(path, text):
    path.write_text(text)
    return path


def test_reverse_copy_preserves_names_and_duplicate_rows(tmp_path):
    original = csv(tmp_path / 'plain.csv', 'sequence,name\nACTG,a\nACTG,b\nACTG,b\n')
    reverse = csv(tmp_path / 'rc.csv', 'sequence,name\nCAGT,b\nCAGT,a\nCAGT,b\n')
    assert data.require_reverse_copy(original, reverse) == 3
    assert data.table(original) == {}
    csv(reverse, 'sequence,name\nCAGT,a\nCAGT,b\nCAGT,c\n')
    with pytest.raises(ValueError, match='changed'):
        data.require_reverse_copy(original, reverse)


def test_identical_duplicate_names_are_resolvable(tmp_path):
    path = csv(tmp_path / 'refs.csv', 'sequence,name\nACTG,a\nACTG,a\nGGGG,b\n')
    assert data.table(path) == {'ACTG': 'a', 'GGGG': 'b'}


def output(tmp_path):
    reference = csv(tmp_path / 'refs.csv', 'sequence,name\nACTG,a\nGGGG,b\n')
    pd.DataFrame({'column_sequence': ['ACTG', 'GGGG', 'ACTG', 'TTTT'],
                  'columnID': ['a', 'b', 'a', None]}).to_hdf(
        tmp_path / 'annotated_reads.h5', key='df')
    csv(tmp_path / 'unique_combinations.csv', 'columnID,count\na,2\nb,1\n')
    csv(tmp_path / 'qc.csv', 'columnID,total_reads\n1,4\n')
    return {'column': reference}


def test_counts_reconcile_including_unassigned_reads(tmp_path):
    proof = data.verify_counts(tmp_path, output(tmp_path), 5)
    assert (proof['extracted_rows'], proof['mapped_reads'], proof['count_rows']) == (4, 3, 2)
    assert proof['accepted'] and not proof['biological_validation_claimed']


@pytest.mark.parametrize('corruption', ['wrong_name', 'wrong_count', 'duplicate_count',
                                      'too_many_reads', 'fractional_count', 'wrong_qc'])
def test_corrupted_realistic_outputs_are_rejected(tmp_path, corruption):
    references = output(tmp_path)
    pairs = 5
    if corruption == 'wrong_name':
        frame = pd.read_hdf(tmp_path / 'annotated_reads.h5', key='df')
        frame.loc[0, 'columnID'] = 'b'
        frame.to_hdf(tmp_path / 'annotated_reads.h5', key='df', mode='w')
    elif corruption == 'wrong_count':
        csv(tmp_path / 'unique_combinations.csv', 'columnID,count\na,1\nb,2\n')
    elif corruption == 'duplicate_count':
        csv(tmp_path / 'unique_combinations.csv', 'columnID,count\na,2\nb,1\na,2\n')
    elif corruption == 'fractional_count':
        csv(tmp_path / 'unique_combinations.csv', 'columnID,count\na,2.5\nb,1\n')
    elif corruption == 'wrong_qc':
        csv(tmp_path / 'qc.csv', 'columnID,total_reads\n0,4\n')
    else:
        pairs = 2
    with pytest.raises(ValueError):
        data.verify_counts(tmp_path, references, pairs)
