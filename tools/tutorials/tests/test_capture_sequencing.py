"""A pair must contain real complete records, not just two downloaded paths."""
import gzip
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'capture_sequencing', Path(__file__).resolve().parents[1] / 'capture_sequencing.py')
recorder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder)


def pair(tmp_path, second='@read/2\nTGCA\n+\nIIII\n'):
    paths = [tmp_path / 'run_1.fastq.gz', tmp_path / 'run_2.fastq.gz']
    for path, data in zip(paths, ['@read/1\nACGT\n+\nIIII\n', second]):
        with gzip.open(path, 'wt') as handle:
            handle.write(data)
    return paths


def test_complete_mates_with_matching_ids_pass(tmp_path):
    result = recorder.inspect_pair(pair(tmp_path), 1)
    assert [row['reads'] for row in result] == [1, 1]
    assert all(row['bytes'] > 0 and len(row['sha256']) == 64 for row in result)


@pytest.mark.parametrize('second,message', [
    ('@different/2\nTGCA\n+\nIIII\n', 'identities'),
    ('@read/2\nTGCA\n+\nIII\n', 'malformed'),
    ('', 'Expected 1 reads'),
])
def test_existing_but_invalid_mates_fail(tmp_path, second, message):
    with pytest.raises(RuntimeError, match=message):
        recorder.inspect_pair(pair(tmp_path, second), 1)


def test_one_file_cannot_impersonate_both_mates(tmp_path):
    paths = pair(tmp_path)
    with pytest.raises(RuntimeError, match='distinct'):
        recorder.inspect_pair([paths[0], paths[0]], 1)


def test_matching_short_files_do_not_meet_the_requested_read_count(tmp_path):
    with pytest.raises(RuntimeError, match='Expected 2 reads, got 1'):
        recorder.inspect_pair(pair(tmp_path), 2)


def mapping_files(tmp_path, count=4):
    folder = tmp_path / 'SRR_example_paired'
    folder.mkdir()
    (folder / 'unique_combinations.csv').write_text(f'grna_name,count\nguide1,{count}\n')
    (folder / 'qc.csv').write_text('total_reads\n10\n')
    (folder / 'annotated_reads.h5').write_bytes(b'nonempty artifact fixture')
    return folder


def test_real_count_files_are_recorded_without_claiming_every_read_mapped(tmp_path):
    mapping_files(tmp_path)
    proof = recorder.inspect_mapping(tmp_path, 'SRR_example', 10)
    assert proof['accepted'] is True
    assert proof['mapped_read_count'] == 4
    assert proof['requested_read_pairs'] == 10
    assert len(proof['files']) == 3


@pytest.mark.parametrize('count', [0, 11])
def test_empty_or_impossible_count_totals_are_rejected(tmp_path, count):
    mapping_files(tmp_path, count)
    with pytest.raises(RuntimeError, match='plausible'):
        recorder.inspect_mapping(tmp_path, 'SRR_example', 10)


def test_empty_requested_artifact_is_not_a_complete_mapping(tmp_path):
    folder = mapping_files(tmp_path)
    (folder / 'annotated_reads.h5').write_bytes(b'')
    with pytest.raises(RuntimeError, match='incomplete'):
        recorder.inspect_mapping(tmp_path, 'SRR_example', 10)
