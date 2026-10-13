"""Non-Cellpose-SAM backends get a resolved-model contract for parallel resume.

CPU only, with stand-in model files and install records: no backend
environment is built and no model is constructed.
"""

import copy
import json

import pytest

from spacr import _segmentation_backends as sb
from spacr._mask_workers import _resolved_backend_model


def _install(root, backend, record):
    env = root / backend
    env.mkdir(parents=True)
    (env / sb._MARKER).write_text(json.dumps(record))


@pytest.fixture
def root(tmp_path):
    return tmp_path / 'backends'


def test_a_stock_prefixed_model_is_signed_by_its_environment(root):
    _install(root, 'stardist', {'requirements': ['stardist==0.9.1']})
    settings = {'nucleus_model_name': 'stardist:2D_versatile_fluo'}
    before = copy.deepcopy(settings)
    _, signature, identity = _resolved_backend_model(settings, 'nucleus', root=root)
    assert settings == before
    assert identity['backend'] == 'stardist'
    assert identity['model'] == '2D_versatile_fluo'
    assert len(signature) == 64
    (root / 'stardist' / sb._MARKER).write_text(
        json.dumps({'requirements': ['stardist==0.9.2']}))
    _, changed, _ = _resolved_backend_model(settings, 'nucleus', root=root)
    assert changed != signature


def test_an_uninstalled_environment_is_refused(root):
    with pytest.raises(ValueError, match='not installed'):
        _resolved_backend_model({'cell_model_name': 'instanseg:'}, 'cell', root=root)


def test_a_model_folder_is_signed_by_its_content(tmp_path, root):
    folder = tmp_path / 'my_stardist'
    folder.mkdir()
    (folder / 'weights.h5').write_bytes(b'one')
    (folder / 'config.json').write_text('{}')
    settings = {'nucleus_model_name': f'stardist:{folder}'}
    _, first, identity = _resolved_backend_model(settings, 'nucleus', root=root)
    assert identity['path'] == str(folder.resolve())
    assert identity['bytes'] == 5
    (folder / 'weights.h5').write_bytes(b'two')
    _, second, _ = _resolved_backend_model(settings, 'nucleus', root=root)
    assert second != first


@pytest.mark.parametrize('value, backend', [
    ('cellpose3:{path}', 'cellpose3'),
    ('cellpose_dino:{path}', 'cellpose_dino'),
    ('omnipose:{path}', 'omnipose'),
])
def test_checkpoint_files_are_signed_for_every_backend(tmp_path, root, value, backend):
    path = tmp_path / 'weights.pth'
    path.write_bytes(b'weights')
    settings = {'cell_model_name': value.format(path=path)}
    _, signature, identity = _resolved_backend_model(settings, 'cell', root=root)
    assert identity['backend'] == backend
    assert identity['sha256'] and identity['bytes'] == 7
    _, again, _ = _resolved_backend_model(dict(settings, mask_gpu_indices='0,1'),
                                          'cell', root=root)
    assert again == signature


def test_missing_checkpoints_and_cellpose_sam_are_refused(tmp_path, root):
    with pytest.raises(ValueError, match='Cellpose-DINO'):
        _resolved_backend_model({'cell_model_name': 'cellpose_dino:/no/such'},
                                'cell', root=root)
    with pytest.raises(ValueError, match='_prepare_mask_model'):
        _resolved_backend_model({'cell_model_name': 'cpsam'}, 'cell', root=root)
    with pytest.raises(ValueError, match='Unsupported'):
        _resolved_backend_model({}, 'organelle', root=root)


def test_samcell_stock_weights_follow_the_backend_setting(root):
    _install(root, 'samcell', {'requirements': ['samcell==1.0']})
    _, _, identity = _resolved_backend_model(
        {'segmentation_backend': 'samcell'}, 'cell', root=root)
    assert identity == {'backend': 'samcell', 'model': 'samcell',
                        'environment': identity['environment']}
    assert len(identity['environment']) == 64


def _stub_segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
    """CPU stand-in for the generator: one mask per field, plus its bound device."""
    import os
    from pathlib import Path

    import numpy as np

    assert run_qc is False
    out = Path(src) / f'{role}_mask_stack'
    out.mkdir(exist_ok=True)
    for path in batch_paths:
        with open(Path(src).parent / 'calls.log', 'a') as log:
            log.write(f"{os.environ.get('CUDA_VISIBLE_DEVICES')} {Path(path).name}\n")
        with np.load(path, allow_pickle=False) as data:
            for plane, name in zip(data['data'], data['filenames']):
                np.save(out / str(name), (plane[..., 0] > 0).astype(np.uint16))
        on_batch_done(path)


@pytest.mark.parametrize('settings, backend', [
    ({'nucleus_model_name': 'stardist:2D_versatile_fluo'}, 'stardist'),
    ({'nucleus_model_name': 'cellpose3:nuclei'}, 'cellpose3'),
    ({'segmentation_backend': 'samcell'}, 'samcell'),
])
def test_dispatch_accepts_every_backend_with_a_contract(tmp_path, monkeypatch, capsys,
                                                       settings, backend):
    import numpy as np

    import spacr.accelerator as acc
    import spacr.object as sobj
    from spacr import _mask_workers as mw

    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'backends'
    _install(root, backend, {'requirements': [f'{backend}==1']})
    monkeypatch.setenv(sb._ROOT_ENV, str(root))
    monkeypatch.setattr(sobj, '_run_seg_qc', lambda *args, **kwargs: None)
    monkeypatch.setattr(mw, '_prepare_mask_model',
                        lambda *args: pytest.fail('Cellpose-SAM contract used'))
    masks = tmp_path / 'plate' / 'masks'
    masks.mkdir(parents=True)
    for index in range(4):
        np.savez(masks / f'batch{index}.npz', data=np.ones((2, 6, 6, 1), np.float32),
                 filenames=np.array([f'p_A0{index}_{f}.npy' for f in range(2)]))
    devices = [{'index': i, 'name': f'Card {i}', 'memory_bytes': 1 << 30, 'backend': 'cuda'}
               for i in (0, 1)]
    run = dict(settings, nucleus_channel=0, channels=[0], cell_channel=None,
               pathogen_channel=None, verbose=False, plot=False, mask_parallel=True)
    plan = mw._parallel_mask_plan(run, devices=devices)
    plan['environments'] = {i: acc._mask_worker_environment(i, backend='cuda', count=2)
                            for i in plan['devices']}

    mw._generate_masks_in_parallel(str(masks), run, 'nucleus', plan, segmenter=_stub_segment)

    calls = [line.split() for line in (masks.parent / 'calls.log').read_text().splitlines()]
    assert sorted(name for _, name in calls) == [f'batch{i}.npz' for i in range(4)]
    assert {device for device, _ in calls} == {'0', '1'}
    assert len(list((masks / 'nucleus_mask_stack').glob('*.npy'))) == 8
    ledger = json.loads((masks / '.mask-workers-nucleus.json').read_text())
    assert ledger['status'] == 'finalized' and len(ledger['completed']) == 4
    assert backend in capsys.readouterr().out


def test_a_mixed_run_routes_each_role_to_its_own_contract(root, monkeypatch):
    from spacr import _mask_workers as mw

    _install(root, 'stardist', {'requirements': ['stardist==1']})
    monkeypatch.setenv(sb._ROOT_ENV, str(root))
    seen = []
    monkeypatch.setattr(mw, '_prepare_mask_model',
                        lambda settings, role: seen.append(role) or ({}, 'sam', {}))
    settings = {'cell_model_name': 'cpsam', 'nucleus_model_name': 'stardist:'}
    assert mw._mask_model_contract(settings, 'cell')[1] == 'sam'
    _, _, identity = mw._mask_model_contract(settings, 'nucleus')
    assert seen == ['cell'] and identity['backend'] == 'stardist'
