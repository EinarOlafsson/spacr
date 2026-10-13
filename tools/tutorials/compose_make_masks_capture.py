"""Compose independently accepted native Make Masks frames without changing pixels.

``--native`` composes four fresh live-video recordings (run_native_capture.py
``--live-video``) instead: every lesson visual must carry its verified clip, and
each recording's own acceptance must replay the results of the accepted still
recording named by ``--reference-item``/``--reference-date`` exactly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

from compose_report_capture import _frame, _read, _same_hash
from stage_lesson import REPO, write


def compose(*, editor, restoration, puncta, destination, yolo=None, receipt_item=615,
            receipt_date='2026-10-05', copy_frames=False):
    roots = {key: Path(path).resolve() for key, path in
             [('editor', editor), ('restoration', restoration), ('puncta', puncta)]}
    if receipt_item not in (615, 662):
        raise ValueError('Use an existing independently checked receipt set')
    if receipt_date not in ('2026-10-05', '2026-10-06', '2026-10-07'):
        raise ValueError('Use an independently checked recording date')
    if yolo is not None:
        roots['yolo'] = Path(yolo).resolve()
    destination = Path(destination).resolve()
    if destination.exists() or len(set(roots.values())) != len(roots):
        raise ValueError('Preserve distinct accepted captures and use a new destination')
    hashes, sources, available, proofs = {}, [], {}, {}
    for key, root in roots.items():
        proof = _read(REPO / f'features/data/{receipt_item}_make_masks_current_{key}_{receipt_date}.json', hashes)
        if proof.get('accepted') is not True or Path(proof['capture']).resolve() != root:
            raise ValueError('The independent evidence must name this exact accepted capture')
        for path, digest in proof['source_file_sha256'].items():
            _same_hash(Path(path), digest, hashes)
        provenance = _read(root / 'provenance.json', hashes)
        if (provenance.get('completed_capture') is not True or provenance.get('module') != 'make_masks'
                or provenance.get('app_source_modified') is not False):
            raise ValueError('Every source must be a completed native Make Masks recording')
        native = _read(root / f'{key}_acceptance.json', hashes)
        if native.get('accepted') is not True or native != proof['native_acceptance']:
            raise ValueError('The actual native acceptance changed after independent checking')
        available[key] = _read(root / 'frames.json', hashes)
        if {name: row['sha256'] for name, row in available[key].items()} != proof['frame_sha256']:
            raise ValueError('The exact native frame inventory changed')
        sources.append(provenance)
        proofs[key] = proof
    if (proofs['puncta'].get('all_csv_bytes_equal') is not True
            or proofs['puncta'].get('reference_rows') != 150
            or proofs['puncta'].get('included_puncta') != 46):
        raise ValueError('The recorded puncta mask and complete CSV need the exact scientific replay')
    if receipt_item == 662:
        application = proofs['editor'].get('application_source_sha256')
        if not application or any(proof.get('application_source_sha256') != application
                                  for proof in proofs.values()):
            raise ValueError('All refreshed recordings must use the same current application source')
    if 'yolo' in proofs:
        native = proofs['yolo']['native_acceptance']
        if (native.get('source_images_and_masks_unchanged') is not True
                or native.get('demonstration_boxes_are_not_biological_ground_truth') is not True):
            raise ValueError('YOLO demonstrations must preserve acquired inputs and their stated scope')
    lesson = _read(REPO / 'tools/tutorials/lessons/14_make_masks.json', hashes)
    if yolo is not None and not any(scene['visual'].startswith('yolo_')
                                   for scene in lesson['scenes']):
        raise ValueError('Author the actual YOLO scenes before including their capture')
    frames, focus = {}, {}
    for scene in lesson['scenes']:
        visual = scene['visual']
        key = next((key for key in ('restoration', 'puncta', 'yolo')
                    if visual.startswith(key + '_')), 'editor')
        if key not in available or visual not in available[key]:
            raise ValueError(f'The lesson needs an independently accepted native frame: {visual}')
        frame = deepcopy(available[key][visual])
        path = _frame(roots[key] / frame['image'], frame['sha256'], roots[key], hashes)
        frame.update(image=os.path.relpath(path, destination), source_capture=str(roots[key]))
        frames[visual] = frame
        focus[visual] = scene.get('focus')
        if visual == 'restoration_04_compare':
            dialogs = [row for row in frame['dialogs'] if row['title'] == 'Raw and enhanced']
            if len(dialogs) != 1:
                raise ValueError('The actual restoration comparison must be visible')
            focus[visual] = dialogs[0]['rect']
        elif visual == 'puncta_02_parent_and_settings':
            focus[visual] = [345, 795, 1030, 930]
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed during verification')
    destination.mkdir(parents=True)
    if copy_frames:
        images = destination / 'native_frames'
        images.mkdir()
        for visual, frame in frames.items():
            if Path(visual).name != visual or visual in ('.', '..'):
                raise ValueError('Use a single authored visual name for each copied frame')
            original = (destination / frame['image']).resolve()
            _same_hash(original, frame['sha256'], hashes)
            copied = images / (visual + '.png')
            shutil.copyfile(original, copied)
            _same_hash(copied, frame['sha256'], hashes)
            frame.update(original_image=str(original),
                         image=os.path.relpath(copied, destination))
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[0], sources=sources,
        composition_only=True, app_source_modified=False))
    write(destination / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Current native ' + ', '.join(roots) + ' scenes; demonstration boxes are not biological ground truth',
        'input_receipts': proofs, 'source_hashes': hashes, 'published': False,
        'biological_ground_truth_claimed': False, 'frame_count': len(frames)})
    digest = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write(destination / 'focus.json', {'english_sha256': digest, 'scenes': focus,
        'recapture': {'capture_module': destination.name,
                     'frames': {key: row['sha256'] for key, row in frames.items()}}})
    print(f'Composed {len(frames)} exact current Make Masks visuals; no pixels changed or media published.')


KEYS = ('editor', 'restoration', 'puncta', 'yolo')


def _replayed(key, native, reference):
    """Name every accepted result the fresh recording must reproduce exactly."""
    def pick(row, names):
        return {name: row.get(name) for name in names}
    if key == 'puncta':
        names = ('image_sha256', 'parent_sha256', 'candidates', 'included', 'saved_mask_sha256',
                 'saved_csv_sha256', 'mask_outside_parent_pixels', 'original_image_and_parent_unchanged')
        return pick(native, names), pick(reference, names)
    if key == 'restoration':
        def values(row):
            plan = row.get('plan') or {}
            return dict(pick(row, ('source_sha256', 'shape', 'enhanced_array_sha256',
                                   'original_image_and_labels_unchanged', 'measurements_generated')),
                        model=plan.get('model'), weights_sha256=plan.get('weights_sha256'),
                        device=plan.get('device'))
        return values(native), values(reference)
    if key == 'yolo':
        def values(row):
            inputs = [pick(item, ('source_sha256', 'image_sha256', 'mask_sha256', 'objects', 'shape'))
                      for item in row.get('inputs', [])]
            return dict(pick(row, ('classes', 'boxes', 'drawn', 'moved', 'resized', 'labels_sha256',
                                   'class_names_sha256', 'negative_labels_sha256',
                                   'independent_xywh_export_check', 'undo_redo_and_reload_exact',
                                   'source_images_and_masks_unchanged')), inputs=inputs)
        return values(native), values(reference)
    def values(row):
        recrop = row.get('recrop') or {}
        organize = (row.get('curation_organize') or {})
        return dict(pick(row, ('initial_objects', 'steps', 'reversible_label_edits_undone_before_recrop',
                               'saved_original_foreground_unchanged', 'original_images_unchanged',
                               'folded_routes_shown', 'model_inference')),
                    recrop=pick(recrop, ('box', 'image_sha256', 'child_objects')),
                    consolidated=(organize.get('consolidation') or {}).get('copied'),
                    organized=[pick(item, ('field', 'planes')) for item in organize.get('organize_inputs', [])])
    return values(native), values(reference)


def compose_native(*, editor, restoration, puncta, yolo, destination,
                   reference_item=662, reference_date='2026-10-07'):
    """Compose four fresh native live-video recordings into the Make Masks lesson."""
    from capture_policy import CAPTURE_BACKDROP
    from compose_capture_set import compose as compose_set
    roots = {key: Path(path).resolve() for key, path in
             zip(KEYS, (editor, restoration, puncta, yolo))}
    destination = Path(destination).resolve()
    if destination.exists() or len(set(roots.values())) != len(roots):
        raise ValueError('Preserve distinct native recordings and use a new destination')
    hashes, receipts = {}, {}
    lesson = _read(REPO / 'tools/tutorials/lessons/14_make_masks.json', hashes)
    selections = {}
    for scene in lesson['scenes']:
        visual = scene['visual']
        key = next((key for key in ('restoration', 'puncta', 'yolo')
                    if visual.startswith(key + '_')), 'editor')
        selections.setdefault(key, set()).add(visual)
    for key, root in roots.items():
        provenance = _read(root / 'provenance.json', hashes)
        if (provenance.get('completed_capture') is not True or provenance.get('module') != 'make_masks'
                or provenance.get('app_source_modified') is not False):
            raise ValueError('Every source must be a completed native Make Masks recording')
        native = _read(root / f'{key}_acceptance.json', hashes)
        reference = _read(REPO / f'features/data/{reference_item}_make_masks_current_{key}_{reference_date}.json', hashes)
        if native.get('accepted') is not True or reference.get('accepted') is not True:
            raise ValueError(f'The {key} recording and its reference must both be accepted')
        actual, expected = _replayed(key, native, reference['native_acceptance'])
        if actual != expected:
            changed = sorted(name for name in expected if actual.get(name) != expected[name])
            raise ValueError(f'The fresh {key} recording does not replay the accepted results: {changed}')
        frames = _read(root / 'frames.json', hashes)
        missing = selections.get(key, set()) - set(frames)
        if missing:
            raise ValueError(f'The {key} recording lacks lesson visuals: {sorted(missing)}')
        for visual in selections.get(key, ()):
            frame = frames[visual]
            if not frame.get('clip'):
                raise ValueError(f'A native lesson visual has no live clip: {visual}')
            if (frame.get('appearance') or {}).get('backdrop') != CAPTURE_BACKDROP:
                raise ValueError(f'A lesson visual was not recorded on the spaCR field: {visual}')
        receipts[key] = {'capture': str(root), 'native_acceptance': native,
                         'reference': f'features/data/{reference_item}_make_masks_current_{key}_{reference_date}.json',
                         'replayed_results': actual}
    frames = compose_set([roots[key] for key in KEYS if key in selections], destination,
                         source_scenes={str(roots[key]): sorted(selections[key])
                                        for key in KEYS if key in selections})
    focus = {}
    for scene in lesson['scenes']:
        visual = scene['visual']
        focus[visual] = scene.get('focus')
        if visual == 'restoration_04_compare':
            dialogs = [row for row in frames[visual]['dialogs'] if row['title'] == 'Raw and enhanced']
            if len(dialogs) != 1:
                raise ValueError('The actual restoration comparison must be visible')
            focus[visual] = dialogs[0]['rect']
        elif visual == 'puncta_02_parent_and_settings':
            focus[visual] = [345, 795, 1030, 930]
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed during verification')
    write(destination / 'scientific_acceptance.json', {
        'accepted': True, 'native_live_video': True,
        'scope': 'Fresh native editor, restoration, puncta and YOLO live recordings; each replays its accepted reference results exactly; demonstration boxes are not biological ground truth',
        'input_receipts': receipts, 'source_hashes': hashes, 'published': False,
        'biological_ground_truth_claimed': False, 'frame_count': len(frames)})
    digest = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write(destination / 'focus.json', {'english_sha256': digest, 'scenes': focus,
        'recapture': {'capture_module': destination.name,
                      'frames': {key: row['sha256'] for key, row in frames.items()}}})
    print(f'Composed {len(frames)} native live Make Masks visuals; clips and pixels unchanged.')
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true',
                        help='Compose fresh native live-video recordings; requires --yolo and replays reference results')
    parser.add_argument('--reference-item', type=int, choices=(662,), default=662)
    parser.add_argument('--reference-date', choices=('2026-10-07',), default='2026-10-07')
    for name in ('editor', 'restoration', 'puncta', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--yolo', type=Path, help='Independently accepted native Box interaction recording')
    parser.add_argument('--receipt-item', type=int, choices=(615, 662), default=615)
    parser.add_argument('--receipt-date', choices=('2026-10-05', '2026-10-06', '2026-10-07'),
                        default='2026-10-05')
    parser.add_argument('--copy-frames', action='store_true',
                        help='Copy byte-verified native frames into the new composition directory')
    arguments = vars(parser.parse_args())
    native, item, date = arguments.pop('native'), arguments.pop('reference_item'), arguments.pop('reference_date')
    if native:
        if arguments['yolo'] is None or arguments.pop('copy_frames') or arguments.pop('receipt_item') != 615 \
                or arguments.pop('receipt_date') != '2026-10-05':
            parser.error('--native takes --editor, --restoration, --puncta, --yolo and --destination only')
        compose_native(**arguments, reference_item=item, reference_date=date)
    else:
        compose(**arguments)
