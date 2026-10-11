"""Make Masks records as native live video: prepared inputs, live pickers, exact replay."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compose_make_masks_capture as composer
from capture_make_masks import stall_milliseconds
from capture_policy import CAPTURE_BACKDROP
from run_native_capture import NEUTRAL, copy_inputs, read_only_binds


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_prepared_inputs_are_copied_byte_for_byte_with_a_manifest(tmp_path):
    source = tmp_path / 'inputs'
    (source / 'puncta/images').mkdir(parents=True)
    (source / 'puncta/images/field.tif').write_bytes(b'acquired pixels')
    (source / 'profile/.cellpose/models').mkdir(parents=True)
    (source / 'profile/.cellpose/models/cpsam').write_bytes(b'weights')
    stage = tmp_path / 'stage'
    stage.mkdir()
    manifest = copy_inputs(source, stage)
    assert manifest == {'profile/.cellpose/models/cpsam': sha(b'weights'),
                        'puncta/images/field.tif': sha(b'acquired pixels')}
    assert (stage / 'puncta/images/field.tif').read_bytes() == b'acquired pixels'
    recorded = json.loads((stage / 'inputs-manifest.json').read_text())
    assert recorded['sha256'] == manifest


@pytest.mark.parametrize('evidence', ['captures/make_masks/frames.json', 'exact-source.json'])
def test_prepared_inputs_cannot_supply_capture_evidence(tmp_path, evidence):
    source = tmp_path / 'inputs'
    (source / evidence).parent.mkdir(parents=True, exist_ok=True)
    (source / evidence).write_text('{}')
    stage = tmp_path / 'stage'
    stage.mkdir()
    with pytest.raises(ValueError):
        copy_inputs(source, stage)


def test_read_only_binds_stay_inside_the_neutral_stage(tmp_path):
    backend = tmp_path / 'cellpose3'
    backend.mkdir()
    assert read_only_binds([f'{backend}=app-state/backends/cellpose3']) == [
        '--ro-bind', str(backend), f'{NEUTRAL}/app-state/backends/cellpose3']
    for pair in (f'{backend}=/etc', f'{backend}=../outside', str(backend), f'{tmp_path}/absent=x'):
        with pytest.raises((ValueError, FileNotFoundError)):
            read_only_binds([pair])


def test_picker_guard_outlasts_a_live_clip_recorded_inside_the_picker():
    def still():
        pass

    def live():
        pass
    live.maximum_live_clip_seconds = 41.5
    assert stall_milliseconds(still) == 12000
    assert stall_milliseconds(live) == 49500


EDITOR = {'accepted': True, 'initial_objects': 63, 'steps': [{'action': 'erase_object', 'before': 63, 'after': 62}],
          'reversible_label_edits_undone_before_recrop': True, 'saved_original_foreground_unchanged': True,
          'original_images_unchanged': True, 'folded_routes_shown': ['train_cellpose'],
          'model_inference': {'model': 'cpsam', 'objects': 1},
          'recrop': {'child': '/tmp/a/child.tif', 'box': [1, 2, 3, 4], 'image_sha256': 'c' * 64, 'child_objects': 1},
          'curation_organize': {'consolidation': {'copied': ['a.tif']},
                                'organize_inputs': [{'source': '/tmp/a/x.npy', 'field': 'E01_f01', 'planes': {'ER': 1}}]}}
RESULTS = {
    'editor': EDITOR,
    'restoration': {'accepted': True, 'source_sha256': 'a' * 64, 'shape': [9, 9], 'enhanced_array_sha256': 'b' * 64,
                    'original_image_and_labels_unchanged': True, 'measurements_generated': False,
                    'plan': {'model': 'denoise_cyto3', 'weights_sha256': 'd' * 64, 'device': 'cpu', 'env': '/tmp/x'}},
    'puncta': {'accepted': True, 'image_sha256': 'e' * 64, 'parent_sha256': 'f' * 64, 'candidates': 150,
               'included': 46, 'saved_mask_sha256': '1' * 64, 'saved_csv_sha256': '2' * 64,
               'mask_outside_parent_pixels': 0, 'original_image_and_parent_unchanged': True},
    'yolo': {'accepted': True, 'classes': ['object', 'demonstration'], 'boxes': [[1, 2, 2, 5, 5]],
             'drawn': [1, 1, 1, 4, 4], 'moved': [1, 2, 2, 5, 5], 'resized': [1, 2, 2, 5, 5],
             'labels_sha256': '3' * 64, 'class_names_sha256': '4' * 64, 'negative_labels_sha256': '5' * 64,
             'independent_xywh_export_check': True, 'undo_redo_and_reload_exact': True,
             'source_images_and_masks_unchanged': True,
             'inputs': [{'source': '/tmp/a.npy', 'source_sha256': '6' * 64, 'image_sha256': '7' * 64,
                         'mask_sha256': '8' * 64, 'objects': 63, 'shape': [9, 9]}]},
}
VISUALS = {'editor': ['01_module', '03_real_labels'], 'restoration': ['restoration_04_compare'],
           'puncta': ['puncta_02_parent_and_settings'], 'yolo': ['yolo_06_drawn_box']}


def native_recording(tmp_path, key, *, clip=True, backdrop=CAPTURE_BACKDROP, result=None):
    stage = tmp_path / f'{key}-stage'
    folder = stage / 'captures/make_masks'
    folder.mkdir(parents=True)
    (stage / 'exact-source.json').write_text(json.dumps({'sha256': {'spacr/qt/screens/make_masks.py': 'same'}}))
    (folder / 'provenance.json').write_text(json.dumps({'completed_capture': True, 'module': 'make_masks',
                                                        'app_source_modified': False}))
    (folder / f'{key}_acceptance.json').write_text(json.dumps(result or RESULTS[key]))
    frames = {}
    for visual in ['00_home', *VISUALS[key]]:
        image = folder / f'{visual}.png'
        image.write_bytes(f'{key}-{visual}-pixels'.encode())
        frame = {'image': image.name, 'sha256': sha(image.read_bytes()),
                 'appearance': {'theme': 'dark', 'backdrop': backdrop},
                 'dialogs': [{'title': 'Raw and enhanced', 'rect': [10, 20, 300, 400]}]
                 if visual == 'restoration_04_compare' else []}
        if clip:
            video = folder / f'{visual}.mp4'
            video.write_bytes(f'{key}-{visual}-live-video'.encode())
            receipt = folder / f'{visual}.capture.json'
            receipt.write_text(json.dumps({'sha256': sha(video.read_bytes()), 'full_decode_passed': True,
                                           'audio_streams': 0}))
            frame['clip'] = {'video': video.name, 'sha256': sha(video.read_bytes()),
                             'receipt': receipt.name, 'receipt_sha256': sha(receipt.read_bytes())}
        frames[visual] = frame
    (folder / 'frames.json').write_text(json.dumps(frames))
    return folder


@pytest.fixture
def repository(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    (repo / 'tools/tutorials/lessons').mkdir(parents=True)
    scenes = [{'visual': visual, 'narration': visual} for key in VISUALS for visual in VISUALS[key]]
    scenes.insert(2, {'visual': '03_real_labels', 'narration': 'again', 'focus': [1, 2, 3, 4]})
    (repo / 'tools/tutorials/lessons/14_make_masks.json').write_text(json.dumps({'id': '14_make_masks', 'scenes': scenes}))
    (repo / 'features/data').mkdir(parents=True)
    for key, result in RESULTS.items():
        (repo / f'features/data/662_make_masks_current_{key}_2026-10-07.json').write_text(
            json.dumps({'accepted': True, 'native_acceptance': result}))
    monkeypatch.setattr(composer, 'REPO', repo)
    return repo


def test_native_composition_keeps_every_clip_and_replays_accepted_results(tmp_path, repository):
    roots = {key: native_recording(tmp_path, key) for key in VISUALS}
    out = tmp_path / 'lesson-stage/captures/make_masks_native'
    frames = composer.compose_native(**roots, destination=out)
    assert set(frames) == {visual for rows in VISUALS.values() for visual in rows}
    for visual, frame in frames.items():
        assert (out / frame['clip']['video']).read_bytes().endswith(b'live-video')
    focus = json.loads((out / 'focus.json').read_text())
    assert focus['scenes']['restoration_04_compare'] == [10, 20, 300, 400]
    # Like the still composer, a repeated visual keeps its last authored focus.
    assert focus['scenes']['03_real_labels'] == [1, 2, 3, 4]
    assert focus['recapture']['capture_module'] == 'make_masks_native'
    accepted = json.loads((out / 'scientific_acceptance.json').read_text())
    assert accepted['accepted'] and accepted['native_live_video']
    assert set(accepted['input_receipts']) == set(VISUALS)


@pytest.mark.parametrize('failure', ['still_frame', 'blobs', 'puncta_count', 'editor_steps', 'unaccepted'])
def test_native_composition_refuses_stills_other_backdrops_and_changed_results(tmp_path, repository, failure):
    roots = {}
    for key in VISUALS:
        result = json.loads(json.dumps(RESULTS[key]))
        if failure == 'puncta_count' and key == 'puncta':
            result['included'] = 45
        if failure == 'editor_steps' and key == 'editor':
            result['steps'][0]['after'] = 61
        if failure == 'unaccepted' and key == 'yolo':
            result['accepted'] = False
        roots[key] = native_recording(
            tmp_path, key, result=result,
            clip=not (failure == 'still_frame' and key == 'restoration'),
            backdrop='blobs' if failure == 'blobs' and key == 'puncta' else CAPTURE_BACKDROP)
    out = tmp_path / 'lesson-stage/captures/make_masks_native'
    with pytest.raises(ValueError):
        composer.compose_native(**roots, destination=out)
    assert not out.exists()


def test_recorded_paths_may_differ_but_not_results(tmp_path, repository):
    moved = json.loads(json.dumps(EDITOR))
    moved['recrop']['child'] = '/tmp/spacr-tutorial-current/make_masks_runs/x/child.tif'
    moved['curation_organize']['organize_inputs'][0]['source'] = '/tmp/spacr-tutorial-current/x.npy'
    roots = {key: native_recording(tmp_path, key, result=moved if key == 'editor' else None) for key in VISUALS}
    out = tmp_path / 'lesson-stage/captures/make_masks_native'
    composer.compose_native(**roots, destination=out)
    assert (out / 'frames.json').exists()
