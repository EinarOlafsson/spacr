"""Small, offline tests for the isolated tutorial refresh staging path."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'stage_lesson.py'
spec = importlib.util.spec_from_file_location('tutorial_stage', MODULE)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'refresh'
    capture = root / 'captures/home'
    capture.mkdir(parents=True)
    image = capture / 'home.png'
    Image.new('RGB', (3840, 2160)).save(image)
    import hashlib
    frames = {'home': {'image': 'home.png',
              'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
              'buttons': [
                  {'name': 'sidebar', 'nav_key': 'mask', 'rect': [0, 0, 10, 10]},
                  {'name': 'tile', 'module_key': 'mask', 'rect': [50, 20, 100, 80]},
              ]}}
    stage.write(capture / 'frames.json', frames)
    stage.write(capture / 'provenance.json', {'completed_capture': True, 'commit': 'fixture'})
    original = {'id': 'home', 'number': 1, 'scenes': [{'narration': 'Old.'}]}
    retained = {'id': 'mask', 'number': 2, 'scenes': [{'narration': 'Retained.'}]}
    stage.write(root / 'catalog/lessons_en.json', {'lessons': [original, retained]})
    changed = {'id': 'home', 'number': 1, 'scenes': [
        {'narration': 'New.', 'visual': 'home', 'focus_modules': ['mask'],
         'related_lessons': ['mask']}]}
    lesson = tmp_path / 'home.json'
    stage.write(lesson, changed)
    return root, capture, lesson, changed, retained


def test_stages_only_selected_lesson_with_measured_tile_geometry(project):
    root, capture, lesson, changed, retained = project
    stage.write(capture / 'scientific_acceptance.json', {'accepted': True})
    stage.stage_lesson(lesson, 'home', root)
    result = stage.read(root / 'catalog/lessons_en.json')
    assert result['lessons'] == [changed, retained]
    visual = stage.read(root / 'production/home/visual.json')
    assert visual['scenes'][0]['focus'] == [50, 20, 100, 80]
    assert visual['scenes'][0]['pointer'] is False
    status = stage.read(root / 'production/home/refresh-status.json')
    assert status['status'] == 'english_staged_not_publishable'
    assert status['translation_review_complete'] is False


def test_check_only_preserves_existing_catalog_and_media_metadata(project):
    root, capture, lesson, changed, retained = project
    stage.write(root / 'production/home/visual.json', {'existing': 'must survive'})
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    stage.stage_lesson(lesson, 'home', root, check_only=True)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert after == before


def test_check_only_still_rejects_changed_lesson_identity(project):
    root, capture, lesson, changed, retained = project
    changed['number'] = 77
    stage.write(lesson, changed)
    with pytest.raises(ValueError, match='retain its existing number'):
        stage.stage_lesson(lesson, 'home', root, check_only=True)


@pytest.mark.parametrize('change', [None, 'number', 'app_key', 'host_app_key', 'status', 'unknown'])
def test_can_fill_exact_published_placeholder_without_renumbering(change):
    current = [{'id': 'diagnostics', 'number': 75}]
    lesson = {'id': '77_embeddings', 'number': 77, 'app_key': 'embeddings'}
    released = [{**lesson, 'status': 'coming_soon'}]
    stage.require_append_identity(lesson, current, released)
    if change is None:
        return
    if change == 'unknown':
        released = []
    else:
        released[0][change] = 'wrong'
    with pytest.raises(ValueError, match='exact published placeholder'):
        stage.require_append_identity(lesson, current, released)


@pytest.mark.parametrize('change', [None, 'number', 'app_key', 'host_app_key', 'status', 'occupied'])
def test_fill_reserved_ops_slot_after_embeddings_was_recorded(change):
    current = [{'id': 'diagnostics', 'number': 75}, {'id': '77_embeddings', 'number': 77}]
    lesson = {'id': '76_ops', 'number': 76, 'app_key': 'ops', 'host_app_key': 'mask'}
    released = [{**lesson, 'status': 'coming_soon'}]
    stage.require_append_identity(lesson, current, released)
    if change is None:
        return
    if change == 'occupied':
        current.append({'id': 'different', 'number': 76})
    else:
        released[0][change] = 'wrong'
    with pytest.raises(ValueError, match='exact published placeholder'):
        stage.require_append_identity(lesson, current, released)


@pytest.mark.parametrize('defect', [None, 'stale', 'missing', 'outside'])
def test_visual_only_focus_preserves_lesson_and_rejects_drift(project, defect):
    import hashlib
    root, capture, lesson, changed, retained = project
    changed['scenes'][0].pop('focus_modules')
    stage.write(lesson, changed)
    canonical = hashlib.sha256(json.dumps(changed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    override = {'english_sha256': canonical, 'scenes': {'home': [50, 20, 100, 80]}}
    if defect == 'stale':
        override['english_sha256'] = 'stale'
    elif defect == 'missing':
        override['scenes'] = {}
    elif defect == 'outside':
        override['scenes']['home'] = [3839, 0, 20, 20]
    focus_map = lesson.with_name('focus.json')
    stage.write(focus_map, override)
    before = (root / 'catalog/lessons_en.json').read_bytes()
    if defect:
        with pytest.raises(ValueError):
            stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert (root / 'catalog/lessons_en.json').read_bytes() == before
    else:
        stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert stage.read(root / 'production/home/lesson.en.json') == changed
        assert stage.read(root / 'production/home/visual.json')['scenes'][0]['focus'] == [50, 20, 100, 80]


@pytest.mark.parametrize('defect', ['failed_capture', 'changed_image', 'missing_link',
                                   'missing_control', 'changed_number', 'bad_focus',
                                   'partial_pipeline', 'failed_scientific_check'])
def test_rejects_invalid_evidence_before_writing_any_catalog(project, defect):
    root, capture, lesson, changed, retained = project
    before = (root / 'catalog/lessons_en.json').read_bytes()
    if defect == 'failed_capture':
        stage.write(capture / 'provenance.json', {'completed_capture': False})
    elif defect == 'partial_pipeline':
        stage.write(capture / 'batch_acceptance.json', {'accepted': False})
    elif defect == 'failed_scientific_check':
        stage.write(capture / 'scientific_acceptance.json', {'accepted': False})
    elif defect == 'changed_image':
        Image.new('RGB', (3840, 2160), 'red').save(capture / 'home.png')
    elif defect == 'missing_link':
        changed['scenes'][0]['related_lessons'] = ['not-a-lesson']
    elif defect == 'missing_control':
        changed['scenes'][0]['focus_modules'] = ['not-a-module']
    elif defect == 'changed_number':
        changed['number'] = 77
    elif defect == 'bad_focus':
        changed['scenes'][0].pop('focus_modules')
        changed['scenes'][0]['focus'] = [3839, 0, 20, 20]
    stage.write(lesson, changed)
    with pytest.raises(ValueError):
        stage.stage_lesson(lesson, 'home', root)
    assert (root / 'catalog/lessons_en.json').read_bytes() == before
    assert not (root / 'production').exists()


@pytest.mark.parametrize('defect', [None, 'unbound', 'other_capture', 'other_frame'])
def test_recapture_focus_replaces_authored_spotlight_only_when_frame_pinned(project, defect):
    import hashlib
    root, capture, lesson, changed, retained = project
    changed['scenes'][0].pop('focus_modules')
    changed['scenes'][0]['focus'] = [0, 0, 10, 10]
    stage.write(lesson, changed)
    canonical = hashlib.sha256(json.dumps(changed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    frame = stage.read(capture / 'frames.json')['home']['sha256']
    override = {'english_sha256': canonical, 'scenes': {'home': [50, 20, 100, 80]},
                'recapture': {'capture_module': 'home', 'frames': {'home': frame}}}
    if defect == 'unbound':
        override.pop('recapture')
    elif defect == 'other_capture':
        override['recapture']['capture_module'] = 'elsewhere'
    elif defect == 'other_frame':
        override['recapture']['frames']['home'] = '0' * 64
    focus_map = lesson.with_name('focus.json')
    stage.write(focus_map, override)
    before = (root / 'catalog/lessons_en.json').read_bytes()
    if defect:
        with pytest.raises(ValueError):
            stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert (root / 'catalog/lessons_en.json').read_bytes() == before
    else:
        stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert stage.read(root / 'production/home/lesson.en.json') == changed
        assert stage.read(root / 'production/home/visual.json')['scenes'][0]['focus'] == [50, 20, 100, 80]


@pytest.fixture
def cellpose_project(project):
    import hashlib
    root, capture, lesson, changed, retained = project
    changed['id'] = '20_cellpose_masks'
    changed['scenes'][0].pop('focus_modules')
    stage.write(lesson, changed)
    stage.write(root / 'catalog/lessons_en.json', {'lessons': [changed, retained]})
    video = capture / 'home.mp4'
    video.write_bytes(b'unit fixture for a previously verified clip')
    native = capture / 'home.native.mp4'
    native.write_bytes(b'unit fixture for the verified native source')
    video_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
    receipt = capture / 'home.capture.json'
    stage.write(receipt, {'sha256': video_sha, 'full_decode_passed': True,
                         'audio_streams': 0, 'size': [3840, 2160],
                         'frames': 30, 'duration': 1.0})
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    frames = stage.read(capture / 'frames.json')
    frames['home']['composition_source'] = {'capture': str(capture), 'visual': 'home'}
    frames['home']['clip'] = {'video': video.name, 'sha256': video_sha,
                            'native_video': native.name, 'native_sha256': native_sha,
                            'receipt': receipt.name, 'receipt_sha256': receipt_sha}
    stage.write(capture / 'frames.json', frames)
    source = root / 'verified-source.json'
    stage.write(source, {'mask_verified': True})
    gate_path = capture / 'home.gate.json'
    stage.write(gate_path, {'passed': True, 'reasons': [], 'distinct_fps': 29.0,
                          'limits': {'min_fps': 27.9}, 'app_freeze_seconds': [],
                          'app_slow_seconds': []})
    proof = {'schema': 1, 'accepted': True, 'lesson': changed['id'],
             'canonical_english_sha256': hashlib.sha256(json.dumps(
                 changed, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
             'accuracy_validated': False, 'ground_truth_used': False,
             'source_hashes': {str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
                              str(gate_path): hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                              str(video): video_sha, str(native): native_sha,
                              str(receipt): receipt_sha},
             'selected_clips': [{'scene': 'home', 'capture': str(capture),
                 'video_sha256': video_sha, 'native_sha256': native_sha,
                 'gate_sha256': hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                 'decision': 'strict_pass'}]}
    stage.write(capture / 'scientific_acceptance.json', proof)
    return root, capture, lesson, changed, frames, proof, source


@pytest.mark.parametrize('defect', ['missing', 'stale_english', 'other_lesson', 'accuracy_claim',
                                   'changed_source', 'missing_source', 'empty_hashes',
                                   'missing_scene', 'duplicate_scene', 'other_clip',
                                   'other_native', 'other_origin', 'failed_gate', 'no_clip',
                                   'missing_native_hash', 'extra_frame'])
def test_cellpose_staging_requires_bound_scientific_evidence_before_catalog_writes(cellpose_project, defect):
    root, capture, lesson, changed, frames, proof, source = cellpose_project
    if defect == 'missing':
        (capture / 'scientific_acceptance.json').unlink()
    elif defect == 'stale_english':
        proof['canonical_english_sha256'] = '0' * 64
    elif defect == 'other_lesson':
        proof['lesson'] = 'home'
    elif defect == 'accuracy_claim':
        proof['accuracy_validated'] = True
    elif defect == 'changed_source':
        source.write_text('changed')
    elif defect == 'missing_source':
        source.unlink()
    elif defect == 'empty_hashes':
        proof['source_hashes'] = {}
    elif defect == 'missing_scene':
        proof['selected_clips'] = []
    elif defect == 'duplicate_scene':
        proof['selected_clips'].append(dict(proof['selected_clips'][0]))
    elif defect == 'other_clip':
        proof['selected_clips'][0]['video_sha256'] = '4' * 64
    elif defect == 'other_native':
        proof['selected_clips'][0]['native_sha256'] = '4' * 64
    elif defect == 'other_origin':
        proof['selected_clips'][0]['capture'] = str(root / 'different')
    elif defect == 'failed_gate':
        proof['selected_clips'][0]['decision'] = 'failed'
    elif defect == 'no_clip':
        frames['home'].pop('clip')
    elif defect == 'missing_native_hash':
        proof['selected_clips'][0].pop('native_sha256')
        frames['home']['clip'].pop('native_sha256')
    elif defect == 'extra_frame':
        frames['unselected'] = dict(frames['home'])
    if defect != 'missing':
        stage.write(capture / 'scientific_acceptance.json', proof)
    stage.write(capture / 'frames.json', frames)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='scientific acceptance|Scientific acceptance'):
        stage.stage_lesson(lesson, 'home', root)
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert after == before


@pytest.mark.parametrize('decision', ['strict_pass', 'approved_capture_loss_near_miss'])
def test_verified_cellpose_selection_passes_scientific_guard(cellpose_project, decision):
    import hashlib
    _root, capture, _lesson, changed, frames, proof, _source = cellpose_project
    proof['selected_clips'][0]['decision'] = decision
    if decision == 'approved_capture_loss_near_miss':
        gate_path = capture / 'home.gate.json'
        gate = stage.read(gate_path)
        gate.update(passed=False, distinct_fps=27.5, accepted_near_miss='capture loss',
                    capture_loss_seconds=[0], reasons=['choppy (capture loss): fixture'])
        stage.write(gate_path, gate)
        digest = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        proof['source_hashes'][str(gate_path)] = digest
        proof['selected_clips'][0]['gate_sha256'] = digest
    stage.write(capture / 'scientific_acceptance.json', proof)
    stage.require_cellpose_acceptance(changed, capture, frames)


@pytest.mark.parametrize('defect', ['failed', 'freeze', 'slow', 'unknown_near_miss',
                                   'below_near_floor', 'weakened_limit', 'unbound_gate'])
def test_scientific_acceptance_cannot_override_failed_or_unproven_clip_gate(cellpose_project, defect):
    import hashlib
    _root, capture, _lesson, changed, frames, proof, _source = cellpose_project
    path = capture / 'home.gate.json'
    gate = stage.read(path)
    if defect == 'failed':
        gate['passed'] = False
    elif defect == 'freeze':
        gate['app_freeze_seconds'] = [0]
    elif defect == 'slow':
        gate['app_slow_seconds'] = [0]
    elif defect == 'weakened_limit':
        gate['limits']['min_fps'] = 27.0
    elif defect in ('unknown_near_miss', 'below_near_floor'):
        gate.update(passed=False, distinct_fps=27.5, accepted_near_miss='capture loss',
                    capture_loss_seconds=[0], reasons=['choppy (capture loss): fixture'])
        proof['selected_clips'][0]['decision'] = 'approved_capture_loss_near_miss'
        if defect == 'unknown_near_miss':
            gate['app_unknown_seconds'] = [1]
        else:
            gate['distinct_fps'] = 26.96
    stage.write(path, gate)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    proof['selected_clips'][0]['gate_sha256'] = digest
    proof['source_hashes'][str(path)] = digest
    if defect == 'unbound_gate':
        del proof['source_hashes'][str(path)]
    stage.write(capture / 'scientific_acceptance.json', proof)
    with pytest.raises(ValueError, match='scientific acceptance'):
        stage.require_cellpose_acceptance(changed, capture, frames)


def test_source_bound_cellpose_capture_can_be_staged(cellpose_project):
    root, capture, lesson, changed, frames, _proof, _source = cellpose_project
    stage.stage_lesson(lesson, 'home', root)
    assert stage.read(root / 'production/20_cellpose_masks/lesson.en.json') == changed
    visual = stage.read(root / 'production/20_cellpose_masks/visual.json')
    assert visual['scenes'][0]['clip']['sha256'] == frames['home']['clip']['sha256']


@pytest.mark.parametrize('key', ['video', 'native_video', 'receipt'])
@pytest.mark.parametrize('defect', ['unbound', 'unsafe'])
def test_selected_source_clip_files_cannot_be_omitted_from_scientific_proof(cellpose_project, key, defect):
    _root, capture, _lesson, changed, frames, proof, _source = cellpose_project
    if defect == 'unbound':
        del proof['source_hashes'][str(capture / frames['home']['clip'][key])]
    else:
        frames['home']['clip'][key] = '../outside-capture'
    stage.write(capture / 'scientific_acceptance.json', proof)
    with pytest.raises(ValueError, match='scientific acceptance'):
        stage.require_cellpose_acceptance(changed, capture, frames)


@pytest.mark.parametrize('authored', ['focus_buttons', 'focus_modules'])
@pytest.mark.parametrize('defect', [None, 'unbound', 'other_english', 'other_scene',
                                   'other_capture', 'other_frame', 'outside', 'null'])
def test_frame_bound_recapture_bypasses_obsolete_authored_lookup(project, authored, defect):
    import hashlib
    root, capture, lesson, changed, retained = project
    changed['scenes'][0].pop('focus_modules')
    changed['scenes'][0][authored] = 'FoldButton' if authored == 'focus_buttons' else ['obsolete']
    stage.write(lesson, changed)
    canonical = hashlib.sha256(json.dumps(changed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    frame = stage.read(capture / 'frames.json')['home']['sha256']
    override = {'english_sha256': canonical, 'scenes': {'home': [50, 20, 100, 80]},
                'recapture': {'capture_module': 'home', 'frames': {'home': frame}}}
    if defect == 'unbound': override.pop('recapture')
    elif defect == 'other_english': override['english_sha256'] = '0' * 64
    elif defect == 'other_scene': override['scenes'] = {'wrong': [50, 20, 100, 80]}
    elif defect == 'other_capture': override['recapture']['capture_module'] = 'wrong'
    elif defect == 'other_frame': override['recapture']['frames']['home'] = '0' * 64
    elif defect == 'outside': override['scenes']['home'] = [3839, 0, 20, 20]
    elif defect == 'null': override['scenes']['home'] = None
    focus_map = lesson.with_name('recapture-focus.json')
    stage.write(focus_map, override)
    before = (root / 'catalog/lessons_en.json').read_bytes()
    if defect:
        with pytest.raises(ValueError):
            stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert (root / 'catalog/lessons_en.json').read_bytes() == before
        assert not (root / 'production').exists()
    else:
        stage.stage_lesson(lesson, 'home', root, check_only=True, focus_map=focus_map)
        assert (root / 'catalog/lessons_en.json').read_bytes() == before
        stage.stage_lesson(lesson, 'home', root, focus_map=focus_map)
        assert stage.read(root / 'production/home/lesson.en.json') == changed
        assert stage.read(root / 'production/home/visual.json')['scenes'][0]['focus'] == [50, 20, 100, 80]
