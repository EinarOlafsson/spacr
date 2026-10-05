"""Compose current native editor, restoration and puncta frames without changing pixels."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from compose_report_capture import _frame, _read, _same_hash
from stage_lesson import REPO, write


def compose(*, editor, restoration, puncta, destination):
    roots = {key: Path(path).resolve() for key, path in
             [('editor', editor), ('restoration', restoration), ('puncta', puncta)]}
    destination = Path(destination).resolve()
    if destination.exists() or len(set(roots.values())) != 3:
        raise ValueError('Preserve three distinct accepted captures and use a new destination')
    hashes, sources, available, proofs = {}, [], {}, {}
    for key, root in roots.items():
        proof = _read(REPO / f'features/data/615_make_masks_current_{key}_2026-10-05.json', hashes)
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
    lesson = _read(REPO / 'tools/tutorials/lessons/14_make_masks.json', hashes)
    frames, focus = {}, {}
    for scene in lesson['scenes']:
        visual = scene['visual']
        key = 'restoration' if visual.startswith('restoration_') else 'puncta' if visual.startswith('puncta_') else 'editor'
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
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[0], sources=sources,
        composition_only=True, app_source_modified=False))
    write(destination / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'All current editor, CPU restoration and scientific-reference puncta scenes',
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
    compose(**vars(parser.parse_args()))
