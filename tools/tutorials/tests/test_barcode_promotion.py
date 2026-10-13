"""Promotion requires complete media AND the exact independently checked run.

Only unit fixtures replace the heavy media/data readers. No fixture is used
as recording evidence or as a source of published tutorial results.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import barcode_promotion as promotion
from stage_lesson import write


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    lesson = {'id': '12_map_barcodes', 'number': 12, 'app_key': 'map_barcodes',
              'scenes': [{'narration': 'Unit fixture, never production evidence'}]}
    write(tmp_path / 'catalog/lessons_en.json', {'lessons': [lesson]})
    write(tmp_path / 'production/12_map_barcodes/lesson.en.json', lesson)
    canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    matrix = {'passed': True, 'unique_final_tracks': 50, 'browser_reports': list(range(14)),
              'canonical_english_sha256': canonical}
    gui = {'extracted_rows': 8611, 'mapped_reads': 7657, 'count_rows': 4099,
           'artifacts': {'counts': 'a' * 64}}
    api = {'extracted_rows': 869, 'mapped_reads': 793, 'count_rows': 535}
    capture = {'accepted': True, 'app_source_modified': False, 'gui': deepcopy(gui),
               'api': {'inputs_unchanged': True, 'gui_barcode_set_control_claimed': False,
                       'barcode_set': ['column', 'grna'], 'api_two_barcodes': deepcopy(api)}}
    search = {'no_automatic_settings_change': True, 'applied_through_visible_button': True,
              'application_functions_replaced': False}
    target = tmp_path / 'captures/map_verified/scientific_acceptance.json'
    search_path = tmp_path / 'captures/map_search_v2/barcode_search.json'
    write(target, capture); write(search_path, search)
    checks = []
    monkeypatch.setattr(promotion, 'check', lambda *args: deepcopy(matrix))

    def verified(path, references, expected):
        checks.append((Path(path), set(references), expected))
        return deepcopy(gui if expected == 10000 else api)

    monkeypatch.setattr(promotion, 'verify_counts', verified)
    assert promotion.require_recorded_map(tmp_path, 'en', lesson) == matrix
    assert [(roles, n) for _, roles, n in checks] == [
        ({'column', 'row', 'grna'}, 10000), ({'column', 'grna'}, 1000)]
    return tmp_path, lesson, matrix, capture, search, target, search_path, gui, api


@pytest.mark.parametrize('defect', ['voices', 'languages', 'english', 'substitution',
                                   'silent_apply', 'gui_editor', 'saved_count', 'saved_artifact', 'narrated_count'])
def test_successful_fixture_then_real_guard_failure(evidence, defect):
    stage, lesson, matrix, capture, search, target, search_path, gui, api = evidence
    if defect == 'voices':
        matrix['unique_final_tracks'] = 49
    elif defect == 'languages':
        matrix['browser_reports'].pop()
    elif defect == 'english':
        lesson['scenes'][0]['narration'] = 'Changed after verification'
    elif defect == 'substitution':
        search['application_functions_replaced'] = True
    elif defect == 'silent_apply':
        search['no_automatic_settings_change'] = False
    elif defect == 'gui_editor':
        capture['api']['gui_barcode_set_control_claimed'] = True
    elif defect == 'saved_count':
        gui['mapped_reads'] = 0
    elif defect == 'saved_artifact':
        gui['artifacts']['counts'] = 'b' * 64
    else:
        gui['mapped_reads'] = 7656
        capture['gui']['mapped_reads'] = 7656
    write(target, capture); write(search_path, search)
    with pytest.raises(ValueError):
        promotion.require_recorded_map(stage, 'en', lesson)


def test_metadata_without_a_recording_cannot_promote():
    with pytest.raises(ValueError, match='actual recording'):
        promotion.require_recorded_map(None, 'en', {'id': '12_map_barcodes'})
