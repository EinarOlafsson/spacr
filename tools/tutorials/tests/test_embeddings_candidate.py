"""The new lesson is a measured API example, with real synchronized media."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from published_lesson import check_published_lesson  # noqa: E402

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_capture_proves_the_api_and_discloses_the_unfinished_gui():
    report = read(ROOT / 'evidence/2026-09-12_embeddings_final_checks.json')
    assert report['published'] is False
    assert report['gui_workflow_completed'] is False
    capture = report['capture']
    assert capture['accepted'] is True
    assert capture['gui']['crops_injected'] is False
    assert capture['gui']['embed_enabled'] is False
    actual = capture['terminal']['runs']
    assert [run['shape'] for run in actual] == [[16, 1536], [16, 512]]
    assert actual[0]['sources'] == actual[1]['sources']
    for run in actual:
        assert run['helper_sha256'] == sha(ROOT / 'embeddings_example.py')
        assert run['source_unchanged'] is True
        assert run['spec']['device'] == 'cpu'
        assert run['biology_validated'] is False
        assert run['stain_mapping_verified'] is False
        assert len(run['weights_sha256']) == 64
        proof = run['verification']
        assert proof['matrix_cells_checked'] == run['shape'][0] * run['shape'][1]
        assert all(proof[key] is True for key in
                   ('ordered_identities_match', 'npy_exact', 'csv_float32_exact'))


def test_embeddings_published_voices_and_heart_captions_are_the_candidates():
    """The nine-scene walkthrough replaced the 2026-09-12 recording."""
    check_published_lesson('77_embeddings', 9)
