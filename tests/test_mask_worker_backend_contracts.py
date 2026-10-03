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
