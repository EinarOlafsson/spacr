from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import napari_example as helper


def receipt():
    entries = [dict(n_changed=count, detail=dict(added=[new], removed=[old], altered=[], via='napari'))
               for count, old, new in [(2197, 2, 18), (1288, 4, 20)]]
    return dict(explicit_reopen_between_edits=True, unrestricted_workflow_verified=False,
                same_session_second_edit_saved=None, originals_preserved=True,
                original_image_unchanged=True, checks={
                    '12d_closed_before_next_edit': dict(viewer_released=True),
                    '11_imported_and_recorded': dict(mask=dict(passed=True, pixels=65536),
                        ledger=dict(edits=deepcopy(entries[:1]))),
                    '14_history_preserved_after_reopen': dict(mask=dict(passed=True, pixels=65536),
                        ledger=dict(edits=deepcopy(entries)))})


def test_verified_reopen_does_not_claim_the_unrestricted_workflow():
    result = helper.verify_reopen_capture(receipt())
    assert result['accepted'] is True
    assert result['unrestricted_workflow_accepted'] is False
    assert result['pixels_per_mask'] == 65536


@pytest.mark.parametrize('key,value', [('explicit_reopen_between_edits',False),
    ('unrestricted_workflow_verified',True), ('same_session_second_edit_saved',False)])
def test_unrestricted_and_failed_receipts_are_not_relabelled(key, value):
    proof = receipt(); proof[key] = value
    with pytest.raises(ValueError, match='only the explicit reopen'):
        helper.verify_reopen_capture(proof)


@pytest.mark.parametrize('key', ['originals_preserved','original_image_unchanged'])
def test_changed_originals_rejected(key):
    proof = receipt(); proof[key] = False
    with pytest.raises(ValueError, match='Original source or image'):
        helper.verify_reopen_capture(proof)


def test_close_action_must_have_released_viewer():
    proof = receipt(); proof['checks']['12d_closed_before_next_edit']['viewer_released'] = False
    with pytest.raises(ValueError, match='not released'):
        helper.verify_reopen_capture(proof)


@pytest.mark.parametrize('case', ['pixels','history','counts','labels'])
def test_saved_pixels_and_both_history_entries_are_required(case):
    proof = receipt(); final = proof['checks']['14_history_preserved_after_reopen']
    if case == 'pixels': final['mask']['pixels'] = 65535
    if case == 'history': final['ledger']['edits'].pop(0)
    if case == 'counts': final['ledger']['edits'][1]['n_changed'] = 1
    if case == 'labels': final['ledger']['edits'][1]['detail']['added'] = [21]
    with pytest.raises(ValueError): helper.verify_reopen_capture(proof)


def source(tmp_path):
    raw = np.zeros((256,256,4), dtype=np.uint16)
    raw[...,1] = np.arange(65536,dtype=np.uint16).reshape(256,256)
    raw[:,1:17,2] = np.arange(2,18,dtype=np.uint16)
    path = tmp_path/'synthetic.npy'; np.save(path, raw)
    return path, raw, hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_preserves_the_two_actual_source_planes(tmp_path):
    path, raw, digest = source(tmp_path); archive = tmp_path/'example.zip'
    result = helper.package(path, archive, expected_source_sha256=digest)
    assert result['pixels_verified'] == 131072
    with zipfile.ZipFile(archive) as handle:
        image = tifffile.imread(io.BytesIO(handle.read('SYNTHETIC_napari/image.tif')))
        mask = tifffile.imread(io.BytesIO(handle.read('SYNTHETIC_napari/mask.tif')))
        assert np.array_equal(image, raw[:,:,1])
        assert np.array_equal(mask, raw[:,:,2])
        assert 'CLOSE and REOPEN' in handle.read('SYNTHETIC_napari/README.txt').decode()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_changed_source_is_not_packaged(tmp_path):
    path, raw, digest = source(tmp_path); raw[0,0,1] = 5; np.save(path,raw)
    with pytest.raises(ValueError, match='Synthetic source changed'):
        helper.package(path, tmp_path/'example.zip', expected_source_sha256=digest)


def test_existing_archive_is_not_overwritten(tmp_path):
    path, raw, digest = source(tmp_path); archive = tmp_path/'example.zip'
    helper.package(path, archive, expected_source_sha256=digest)
    before = archive.read_bytes()
    with pytest.raises(FileExistsError): helper.package(path,archive,expected_source_sha256=digest)
    assert archive.read_bytes() == before
