"""Package the recorded synthetic inputs and verify the restricted round trip."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import tifffile


def verify_reopen_capture(proof):
    """Accept only the explicitly recorded close/reopen workflow, never a fix."""
    if (proof.get('explicit_reopen_between_edits') is not True
            or proof.get('unrestricted_workflow_verified') is not False
            or proof.get('same_session_second_edit_saved') is not None):
        raise ValueError('This receipt must describe only the explicit reopen workflow')
    if not (proof.get('originals_preserved') and proof.get('original_image_unchanged')):
        raise ValueError('Original source or image changed')
    checks = proof['checks']
    if checks['12d_closed_before_next_edit']['viewer_released'] is not True:
        raise ValueError('The viewer was not released between edits')
    first = checks['11_imported_and_recorded']
    second = checks['14_history_preserved_after_reopen']
    for result in (first, second):
        if result['mask']['passed'] is not True or result['mask']['pixels'] != 65536:
            raise ValueError('Every saved mask pixel must have been checked')
    one, two = first['ledger']['edits'], second['ledger']['edits']
    if len(one) != 1 or len(two) != 2 or two[0] != one[0]:
        raise ValueError('The first correction history was lost or duplicated')
    if [entry['n_changed'] for entry in two] != [2197, 1288]:
        raise ValueError('Correction counts differ from the recorded native edits')
    for entry, old, new in zip(two, (2, 4), (18, 20)):
        if entry['detail'] != {'added': [new], 'altered': [], 'removed': [old], 'via': 'napari'}:
            raise ValueError('Correction identities differ from the actual relabelling')
    return {'accepted': True, 'saved_masks_checked': 2, 'pixels_per_mask': 65536,
            'correction_entries': 2, 'unrestricted_workflow_accepted': False}


def package(source, archive, *, expected_source_sha256):
    """Copy the exact recorded source planes, refusing changed or overwritten input."""
    source, archive = Path(source), Path(archive)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_source_sha256:
        raise ValueError('Synthetic source changed from the recorded input')
    raw = np.load(source, allow_pickle=False)
    if raw.shape != (256, 256, 4) or raw.dtype != np.uint16:
        raise ValueError('Expected the original four-plane synthetic uint16 array')
    if np.unique(raw[..., 2]).tolist() != [0, *range(2, 18)]:
        raise ValueError('Expected the original sixteen synthetic labels')
    images = {'image.tif': raw[..., 1], 'mask.tif': raw[..., 2]}
    files = {}
    for name, array in images.items():
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, array)
        files[name] = buffer.getvalue()
    files['README.txt'] = (
        'SYNTHETIC Napari Bridge practice inputs -- not a biological experiment.\n'
        'Extract to a NEW folder; edits overwrite its mask.tif. Keep an untouched backup.\n'
        'In Make Masks > Napari Bridge select mask.tif, then image.tif.\n'
        'The optional napari dependency must be installed; recording used napari 0.9.1.\n'
        'Open in napari, select the mask Labels layer, choose a label and use Fill.\n'
        'The tutorial replaces label 2 with 18, undoes, repeats, then imports to spaCR.\n'
        'IMPORTANT: CLOSE and REOPEN the viewer after EACH imported edit.\n'
        'A second edit in the same viewer was not saved by the tested app version.\n'
        'This explicit workaround does not claim that defect is fixed.\n'
        'After reopening, replace label 4 with 20 and import again.\n'
        'Check mask.tif and its mask.tif.curation.json history before proceeding.\n'
        'Relabelling keeps sixteen objects: one added/removed ID is not an extra cell.\n'
        'Image intensity pixels must not change; this exercise validates no biology.\n'
    ).encode()
    manifest = dict(synthetic=True, source_sha256=digest, shape=[256, 256],
                    dtype='uint16', image_plane=1, mask_plane=2,
                    labels=[0, *range(2, 18)], biological_validation=False,
                    requires_reopen_after_import=True,
                    files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()})
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as handle:
        for name, data in files.items():
            handle.writestr('SYNTHETIC_napari/' + name, data)
    with zipfile.ZipFile(archive) as handle:
        for name, array in images.items():
            restored = tifffile.imread(io.BytesIO(handle.read('SYNTHETIC_napari/' + name)))
            if restored.dtype != array.dtype or not np.array_equal(restored, array):
                raise ValueError('Packaged input pixels changed')
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Packaging modified the original synthetic input')
    return dict(archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                bytes=archive.stat().st_size, images_verified=2, pixels_verified=131072,
                source_unchanged=True, manifest=manifest)
