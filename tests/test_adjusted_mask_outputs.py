"""Finalized cell masks leave raw segmentation receipts and intensities intact."""
from pathlib import Path

import numpy as np
import pytest

from spacr._mask_workers import _MaskBatchLedger
from spacr.checkpoint import fingerprint
from spacr.io import _load_and_concatenate_arrays, _mask_batch_manifest
from spacr.object import _run_seg_qc
from spacr.utils import adjust_cell_masks, process_mask_file_adjust_cell


@pytest.fixture
def plate(tmp_path):
    masks = tmp_path / 'masks'
    masks.mkdir()
    stack = tmp_path / 'stack'
    stack.mkdir()
    cell = np.zeros((40, 40), np.uint16)
    cell[5:35, 5:20] = 7
    cell[5:35, 20:35] = 300
    nucleus = np.zeros_like(cell)
    nucleus[15:25, 15:25] = 1
    parasite = np.zeros_like(cell)
    intensity = np.stack([np.full_like(cell, value) for value in (11, 37)], axis=-1)
    names = ['f0.npy', 'f1.npy']
    np.savez(masks / 'batch.npz', data=np.stack([intensity] * 2), filenames=names)
    records = _mask_batch_manifest(masks)
    ledgers = {}
    for role, array in [('cell', cell), ('nucleus', nucleus), ('pathogen', parasite)]:
        ledger = _MaskBatchLedger(masks, records, role, fingerprint({'role': role}))
        ledger.output_root.mkdir()
        for name in names:
            np.save(ledger.output_root / name, array)
        ledger.mark(records[0]['path'])
        ledgers[role] = ledger
    for name in names:
        np.save(stack / name, intensity)
    return tmp_path, ledgers


def _adjust(plate, output, **kwargs):
    root, ledgers = plate
    return adjust_cell_masks(
        ledgers['pathogen'].output_root, ledgers['cell'].output_root,
        ledgers['nucleus'].output_root, output_folder=output, n_jobs=1, **kwargs)


def test_adjusted_outputs_feed_merging_and_qc_without_invalidating_raw_receipts(plate):
    root, ledgers = plate
    original = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    output = root / 'masks/adjusted_cell_mask_stack'
    _adjust(plate, output)
    for path, data in original.items():
        assert path.read_bytes() == data
    for ledger in ledgers.values():
        assert ledger.verified() == set(ledger.records)
    for name in ('f0.npy', 'f1.npy'):
        assert set(np.unique(np.load(output / name))) == {0, 1}
    _load_and_concatenate_arrays(str(root), [0, 1], 0, 1, 1, None,
                                 mask_folders={'cell': output})
    for name in ('f0.npy', 'f1.npy'):
        merged = np.load(root / 'merged' / name)
        assert merged.shape == (40, 40, 5)
        np.testing.assert_array_equal(merged[..., :2], np.load(root / 'stack' / name))
        np.testing.assert_array_equal(merged[..., 2], np.load(output / name))
    result = _run_seg_qc(str(root / 'masks'), {'seg_qc': 'report', 'verbose': False},
                         'cell', mask_folder=output)
    assert result is not None and len(result['field_qcs']) == 2
    assert all(field.n_objects == 1 for field in result['field_qcs'])
    assert Path(result['csv_path']).is_file()
    for ledger in ledgers.values():
        assert ledger.verified() == set(ledger.records)


def test_interrupted_adjustment_restarts_from_raw_masks(plate, monkeypatch):
    import spacr.utils as utils

    root, ledgers = plate
    output = root / 'masks/adjusted_cell_mask_stack'
    actual = utils.process_mask_file_adjust_cell

    def fail_second(name, *args, **kwargs):
        if name == 'f1.npy':
            raise RuntimeError('interrupted')
        return actual(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(utils, 'process_mask_file_adjust_cell', fail_second)
        with pytest.raises(RuntimeError, match='interrupted'):
            _adjust(plate, output)
    assert (output / 'f0.npy').is_file() and not (output / 'f1.npy').exists()
    for ledger in ledgers.values():
        assert ledger.verified() == set(ledger.records)
    _adjust(plate, output)
    np.testing.assert_array_equal(np.load(output / 'f0.npy'), np.load(output / 'f1.npy'))


@pytest.mark.parametrize('role', ['cell', 'nucleus', 'pathogen'])
@pytest.mark.parametrize('alias', [False, True])
def test_output_aliases_are_refused_without_changing_raw_masks(plate, role, alias):
    root, ledgers = plate
    output = ledgers[role].output_root
    if alias:
        link = root / 'alias'
        link.symlink_to(output, target_is_directory=True)
        output = link
    with pytest.raises(ValueError, match='differ from all source'):
        _adjust(plate, output)
    with pytest.raises(ValueError, match='differ from all source'):
        process_mask_file_adjust_cell('f0.npy', ledgers['pathogen'].output_root,
            ledgers['cell'].output_root, ledgers['nucleus'].output_root, output_folder=output)
    for ledger in ledgers.values():
        assert ledger.verified() == set(ledger.records)


def test_unrelated_destination_masks_are_preserved_and_refused(plate):
    root, ledgers = plate
    output = root / 'adjusted'
    output.mkdir()
    np.save(output / 'another.npy', np.ones((8, 8), np.uint16))
    before = (output / 'another.npy').read_bytes()
    with pytest.raises(ValueError, match='unrelated fields'):
        _adjust(plate, output)
    assert (output / 'another.npy').read_bytes() == before
    assert not (output / 'f0.npy').exists()


@pytest.mark.parametrize('overrides', [{'wrong': '.'}, {'cell': '/missing/adjusted-mask-folder'}])
def test_invalid_merge_override_fails_before_output_creation(plate, overrides):
    root, _ = plate
    with pytest.raises(ValueError):
        _load_and_concatenate_arrays(str(root), [0, 1], 0, 1, 1, None, mask_folders=overrides)
    assert not (root / 'merged').exists()
