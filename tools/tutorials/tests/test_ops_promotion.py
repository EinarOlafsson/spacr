"""OPS publication copies need both media reconciliation and real capture proof.

The expensive media verifier is replaced ONLY in these unit fixtures. No
fixture here is production evidence or a publishable tutorial candidate.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ops_promotion as promotion
from stage_lesson import write


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    lesson = {'id': '76_ops', 'number': 76, 'app_key': 'ops', 'host_app_key': 'mask',
              'scenes': [{'narration': 'Unit fixture, not actual capture.'}]}
    write(tmp_path / 'catalog/lessons_en.json', {'lessons': [lesson]})
    write(tmp_path / 'production/76_ops/lesson.en.json', lesson)
    canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    matrix = {'passed': True, 'canonical_english_sha256': canonical,
              'unique_final_tracks': 50, 'browser_reports': list(range(14))}
    calls = []

    def checked(*args):
        calls.append(args)
        return deepcopy(matrix)

    monkeypatch.setattr(promotion, 'check', checked)
    source = tmp_path / 'captured-receipt.json'
    write(source, {'unit_fixture': True})
    capture = {'accepted': True, 'gui_workflow_completed': False,
               'gui': {'accepted': True, 'run_clicked': False, 'gui_workflow_completed': False},
               'terminal': {'accepted': True, 'source_unchanged': True, 'gui_workflow_completed': False,
                            'run': {'accepted': True, 'placed': 4, 'accepted_edges': 4,
                                    'canvas': [2756, 2756], 'full_pipeline_completed': False,
                                    'segmentation_or_decoding_performed': False}},
               'source_hashes': {str(source): promotion.digest(source)}}
    target = tmp_path / 'captures/ops_1507_verified/scientific_acceptance.json'
    write(target, capture)
    assert promotion.require_recorded_ops(tmp_path, 'en', lesson) == matrix
    assert calls == [(tmp_path, '76_ops', tmp_path.parent / 'tools/render_all_voices.py')]
    return tmp_path, lesson, matrix, capture, target, source


@pytest.mark.parametrize('defect', ['track', 'browser', 'english', 'capture', 'source', 'scope'])
def test_exact_current_positive_evidence_then_each_failure(evidence, defect):
    stage, lesson, matrix, capture, target, source = evidence
    if defect == 'track':
        matrix['unique_final_tracks'] = 49
    elif defect == 'browser':
        matrix['browser_reports'].pop()
    elif defect == 'english':
        lesson['scenes'][0]['narration'] = 'Changed after verification'
    elif defect == 'capture':
        capture['gui']['run_clicked'] = True
    elif defect == 'source':
        write(source, {'altered': True})
    elif defect == 'scope':
        capture['terminal']['run']['full_pipeline_completed'] = True
    write(target, capture)
    with pytest.raises(ValueError):
        promotion.require_recorded_ops(stage, 'en', lesson)
