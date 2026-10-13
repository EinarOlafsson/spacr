"""Only promote the two explicitly scoped, actually recorded model tutorials.

A title, a passing unit fixture or a scene list is never sufficient. Promotion
requires the complete media matrix and unchanged real capture evidence.
"""
import hashlib
import json
from pathlib import Path

from check_completed_matrix import check, digest
from stage_lesson import read

MODELS = {
    '21_model_compare': ('model_compare_1507_verified', '2026-09-12_model_compare_actual_recording.json'),
    '22_model_zoo': ('model_zoo_1507_inventory_v2', '2026-09-12_model_zoo_inventory_recording.json'),
}


def validate_scope(identity, capture):
    if identity not in MODELS or capture.get('accepted') is not True:
        raise ValueError('An actual accepted model recording is required')
    if identity == '22_model_zoo':
        if (capture.get('route') != ['make_masks', 'model_zoo']
                or capture.get('displayed_rows') != 8
                or capture.get('original_checkpoint_unchanged') is not True
                or capture.get('model_directory_unchanged') is not True
                or capture.get('checksum_state') != 'none'
                or capture.get('trained_on') != 'unknown' or capture.get('trained_by') != 'unknown'
                or any(capture.get(key) is not False for key in (
                    'checkpoint_provenance_independently_validated', 'benchmark_completed',
                    'segmentation_performed', 'model_downloaded', 'training_performed',
                    'inference_overridden', 'app_source_modified', 'published'))):
            raise ValueError('Model Zoo is the recorded inventory/provenance lesson, not a benchmark')
        return
    gui, terminal = capture['gui'], capture['terminal']
    run = terminal['run']
    if (gui.get('accepted') is not True or terminal.get('accepted') is not True
            or run.get('accepted') is not True or gui.get('route') != ['make_masks', 'model_compare']
            or gui.get('loaded_pixels_equal_originals') is not True
            or gui.get('original_inputs_unchanged') is not True
            or gui.get('compare_clicked') is not False
            or gui.get('results_injected') is not False or terminal.get('results_injected') is not False
            or run.get('source_unchanged') is not True or run.get('same_mask_control_passed') is not True
            or any(run.get(key) is not False for key in ('ground_truth_used', 'accuracy_validated',
                'different_model_weights_compared', 'gui_workflow_completed', 'inference_performed'))
            or any(item.get('gui_workflow_completed') is not False for item in (capture, gui, terminal))):
        raise ValueError('Model Compare is the saved-mask API example, not repaired GUI inference')
    counts = run['comparison']
    checked = run['independent_checks']
    if ([counts.get(key) for key in ('n_objects_a', 'n_objects_b', 'n_matched')] != [94, 95, 91]
            or checked != {'objects_a': 94, 'objects_b': 95, 'matched_pairs_pixel_checked': 91,
                           'foreground_disagreement_pixels': 9564}
            or abs(counts.get('mean_matched_iou', 0) - .9601905822488658) > 1e-12):
        raise ValueError('The recorded pixel comparison differs from the authored example')


def require_recorded_model(stage, language, lesson):
    identity = lesson['id']
    if stage is None or identity not in MODELS:
        raise ValueError('Model promotion needs actual staged media and capture proof')
    stage = Path(stage).resolve()
    root = Path(__file__).resolve().parent
    prefix = 'captions' if language in {'da', 'de', 'is', 'ko', 'nb', 'sv'} else 'lessons'
    current = [item for item in read(stage / 'catalog' / f'{prefix}_{language}.json')['lessons']
               if item['id'] == identity]
    if current != [lesson]:
        raise ValueError('The model lesson differs from its verified language catalog')
    english = read(stage / 'production' / identity / 'lesson.en.json')
    if english != read(root / 'lessons' / (identity + '.json')):
        raise ValueError('The script differs from the actual recorded example')
    matrix = check(stage, identity, stage.parent / 'tools/render_all_voices.py')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (matrix.get('passed') is not True or matrix.get('unique_final_tracks') != 50
            or matrix.get('canonical_english_sha256') != canonical
            or len(matrix.get('browser_reports', [])) != 14):
        raise ValueError('A model lesson needs all fifty verified voices and fourteen browser cases')
    folder, receipt = MODELS[identity]
    path = stage / 'captures' / folder / 'scientific_acceptance.json'
    if digest(path) != digest(root / 'evidence' / receipt):
        raise ValueError('Recorded model evidence changed after review')
    capture = read(path)
    validate_scope(identity, capture)
    if identity == '22_model_zoo':
        if digest(capture['model_path']) != capture['selected_model_fingerprint']['sha256']:
            raise ValueError('The demonstrated local checkpoint changed')
    else:
        for name, expected in capture['source_hashes'].items():
            original = Path(name).resolve(strict=True)
            if not original.is_relative_to(stage) or digest(original) != expected:
                raise ValueError('The actual model comparison capture changed or leaves staging')
        if digest(root / 'model_compare_example.py') != capture['terminal']['helper_sha256']:
            raise ValueError('The recorded API helper changed')
        figure = capture['terminal']['figure']
        if digest(figure['path']) != figure['sha256']:
            raise ValueError('The actual comparison output figure changed')
    return matrix
