"""Compose independently accepted native Make Masks frames without changing pixels."""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('editor', 'restoration', 'puncta', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--yolo', type=Path, help='Independently accepted native Box interaction recording')
    parser.add_argument('--receipt-item', type=int, choices=(615, 662), default=615)
    parser.add_argument('--receipt-date', choices=('2026-10-05', '2026-10-06', '2026-10-07'),
                        default='2026-10-05')
    parser.add_argument('--copy-frames', action='store_true',
                        help='Copy byte-verified native frames into the new composition directory')
    compose(**vars(parser.parse_args()))
