"""An assigned mask worker uses the normal pipeline on only its own archives."""

from pathlib import Path

import numpy as np
import pytest

from tests.test_cellpose4_model_story import _mask_settings
from tests.test_cellpose4_model_story import sam_pipeline as sam_pipeline


def _batches(root):
    root.mkdir()
    result = []
    for index in range(3):
        path = root / f'batch{index}.npz'
        np.savez(path, data=np.ones((1, 32, 32, 2), dtype=np.float32),
                 filenames=np.array([f'plate1_A01_{index + 1}.npy']))
        result.append(path)
    return result


def test_one_model_handles_only_assigned_archives_and_reports_completed_files(
        tmp_path, monkeypatch, sam_pipeline):
    from spacr import object as objects

    src = tmp_path / 'masks'
    paths = _batches(src)
    created, done, qc = [], [], []
    constructor = objects.cp_models.CellposeModel

    def model(**kwargs):
        created.append(1)
        return constructor(**kwargs)

    def finished(path):
        index = int(Path(path).stem[-1])
        output = src / 'cell_mask_stack' / f'plate1_A01_{index + 1}.npy'
        assert np.load(output).max() == 1
        done.append(path)

    monkeypatch.setattr(objects.cp_models, 'CellposeModel', model)
    monkeypatch.setattr(objects, '_run_seg_qc', lambda *args: qc.append(args))
    objects.generate_cellpose_masks_sam(str(src), _mask_settings(src), 'cell',
        batch_paths=[paths[0], paths[2]], on_batch_done=finished, run_qc=False)
    assert len(created) == 1
    assert done == [str(paths[0]), str(paths[2])]
    assert not (src / 'cell_mask_stack' / 'plate1_A01_2.npy').exists()
    assert not qc
    assert all(path.exists() for path in paths)


def test_cancel_between_archives_keeps_completed_masks_and_skips_later_inputs(
        tmp_path, sam_pipeline):
    from spacr import object as objects
    from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token

    src = tmp_path / 'masks'
    paths = _batches(src)
    token = CancellationToken()
    done = []

    def finished(path):
        done.append(path)
        token.cancel()

    with installed_token(token), pytest.raises(PipelineCancelled):
        objects.generate_cellpose_masks_sam(str(src), _mask_settings(src), 'cell',
            batch_paths=paths, on_batch_done=finished, run_qc=False)
    assert done == [str(paths[0])]
    assert np.load(src / 'cell_mask_stack' / 'plate1_A01_1.npy').max() == 1
    assert len(list((src / 'cell_mask_stack').glob('*.npy'))) == 1
    assert all(path.exists() for path in paths)


def test_a_failed_archive_never_reports_completion(tmp_path, monkeypatch, sam_pipeline):
    from spacr import io, object as objects

    src = tmp_path / 'masks'
    paths = _batches(src)
    done = []

    def fail(*args, **kwargs):
        raise OSError('cannot save mask')

    monkeypatch.setattr(io, '_save_array_atomic', fail)
    with pytest.raises(OSError, match='cannot save mask'):
        objects.generate_cellpose_masks_sam(str(src), _mask_settings(src), 'cell',
            batch_paths=paths, on_batch_done=done.append, run_qc=False)
    assert not done


@pytest.mark.parametrize('assignment', ['duplicate', 'outside', 'hidden', 'string', 'empty'])
def test_assignment_is_checked_before_model_loading(
        tmp_path, monkeypatch, sam_pipeline, assignment):
    from spacr import object as objects

    src = tmp_path / 'masks'
    paths = _batches(src)
    outside = tmp_path / 'outside.npz'
    outside.write_bytes(paths[0].read_bytes())
    hidden = src / '.hidden.npz'
    hidden.write_bytes(paths[0].read_bytes())
    selected = {'duplicate': [paths[0], paths[0]], 'outside': [outside],
                'hidden': [hidden], 'string': str(paths[0]), 'empty': []}[assignment]
    monkeypatch.setattr(objects.cp_models, 'CellposeModel',
                        lambda **kwargs: pytest.fail('model loaded for invalid/empty assignment'))
    if assignment == 'empty':
        objects.generate_cellpose_masks_sam(str(src), _mask_settings(src), 'cell',
                                            batch_paths=selected)
    else:
        with pytest.raises(ValueError):
            objects.generate_cellpose_masks_sam(str(src), _mask_settings(src), 'cell',
                                                batch_paths=selected)
