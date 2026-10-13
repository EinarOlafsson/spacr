"""Real image and checkpoint regressions for issue 73's classification inputs."""

import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from spacr import crops
from spacr.classification_pixels import (
    DECLARED_UINT8, STORED_PIL, checkpoint_policy, initialization_policy,
    loader_policy, training_preprocessing,
)


def _folder(tmp_path, pixels, fmt=3):
    folder = tmp_path / "images"
    folder.mkdir(parents=True)
    path = folder / "plate1_A01_1_1.png"
    Image.fromarray(pixels).save(path)
    if fmt is not None:
        crops.write_crop_folder_marker(folder, fmt=fmt)
    return folder, path


def _archive(folder, tmp_path):
    path = tmp_path / "images.tar"
    with tarfile.open(path, "w") as archive:
        for source in folder.iterdir():
            archive.add(source, arcname=source.name)
    return path


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("policy,expected", [
    (DECLARED_UINT8, [0, 1, 4, 128, 255]),
    (STORED_PIL, [0, 255, 255, 255, 255]),
])
def test_uint16_values_survive_folder_labelled_and_archive_loaders(tmp_path, preload, policy, expected):
    from spacr.io import NoClassDataset, spacrDataset, TarImageDataset

    pixels = np.array([[0, 256, 1024, 32768, 65535]], dtype=np.uint16)
    folder, path = _folder(tmp_path, pixels)
    flat = NoClassDataset(folder, transform=np.asarray, shuffle=False,
                          load_to_memory=preload, crop_loading_policy=policy)
    labelled = spacrDataset(tmp_path, [folder.name], transform=np.asarray,
                            shuffle=False, pin_memory=preload, crop_loading_policy=policy)
    archive = TarImageDataset(_archive(folder, tmp_path), transform=np.asarray,
                              crop_loading_policy=policy)
    for image in (flat[0][0], labelled[0][0], archive[0][0]):
        assert image.dtype == np.uint8
        assert image.shape == (1, 5, 3)
        assert image[0, :, 0].tolist() == expected
    assert np.array(Image.open(path)).tolist() == pixels.tolist()


@pytest.mark.parametrize("fmt", [None, 1, 2, 3])
@pytest.mark.parametrize("policy", [DECLARED_UINT8, STORED_PIL])
def test_declared_and_historical_channel_policies(tmp_path, fmt, policy):
    from spacr.io import NoClassDataset, spacrDataset, TarImageDataset

    pixels = np.full((4, 6, 3), [25, 80, 200], dtype=np.uint8)
    folder, path = _folder(tmp_path, pixels, fmt)
    expected = [200, 80, 25] if fmt == 2 and policy == DECLARED_UINT8 else [25, 80, 200]
    datasets = [NoClassDataset(folder, transform=np.asarray, shuffle=False, crop_loading_policy=policy),
                spacrDataset(tmp_path, [folder.name], transform=np.asarray, shuffle=False, crop_loading_policy=policy),
                TarImageDataset(_archive(folder, tmp_path), transform=np.asarray, crop_loading_policy=policy)]
    for dataset in datasets:
        assert dataset[0][0][0, 0].tolist() == expected
    if policy == DECLARED_UINT8:
        np.testing.assert_array_equal(datasets[0][0][0], crops.read_crop_png(path))


def test_exif_orientation_is_consistent_for_new_classification(tmp_path):
    from spacr.io import NoClassDataset, spacrDataset, TarImageDataset

    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    folder, path = _folder(tmp_path, pixels)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(pixels).save(path, exif=exif)
    datasets = [NoClassDataset(folder, transform=np.asarray),
                spacrDataset(tmp_path, [folder.name], transform=np.asarray),
                TarImageDataset(_archive(folder, tmp_path), transform=np.asarray)]
    for dataset in datasets:
        np.testing.assert_array_equal(dataset[0][0], np.rot90(pixels, -1))


def test_root_only_historical_training_marker_is_honored(tmp_path):
    from spacr.io import spacrDataset

    folder = tmp_path / 'train' / 'positive'
    folder.mkdir(parents=True)
    Image.new('RGB', (2, 2), (25, 80, 200)).save(folder / 'crop.png')
    crops.write_crop_folder_marker(tmp_path, fmt=2, split='train/test')
    dataset = spacrDataset(tmp_path / 'train', ['positive'], transform=np.asarray)
    assert dataset[0][0][0, 0].tolist() == [200, 80, 25]


def test_archive_resolves_nested_markers_and_interrupted_migration(tmp_path):
    from spacr.io import TarImageDataset

    root = tmp_path / 'nested'
    for fmt in (2, 3):
        folder = root / str(fmt)
        folder.mkdir(parents=True)
        for name in ('a', 'b'):
            Image.new('RGB', (2, 2), (25, 80, 200)).save(folder / (name + '.png'))
        extra = {'migration': {'from': 1, 'done_through': 'a.png'}} if fmt == 2 else {}
        crops.write_crop_folder_marker(folder, fmt=fmt, **extra)
    path = tmp_path / 'nested.tar'
    with tarfile.open(path, 'w') as archive:
        archive.add(root, arcname='nested')
    dataset = TarImageDataset(path, transform=np.asarray)
    actual = {name: image[0, 0].tolist() for image, name in dataset}
    assert actual['nested/2/a.png'] == [200, 80, 25]
    assert actual['nested/2/b.png'] == [25, 80, 200]
    assert actual['nested/3/a.png'] == [25, 80, 200]


def test_unknown_decoder_is_refused_and_missing_record_is_explicit(capsys):
    assert checkpoint_policy({}, announce=True) == STORED_PIL
    assert 'High-bit-depth crops can clip' in capsys.readouterr().out
    with pytest.raises(ValueError, match='Unsupported'):
        checkpoint_policy({'preprocessing': {'crop_loading_policy': 'future_decoder'}})


def test_training_rejects_mixed_or_misrecorded_inputs():
    declared = SimpleNamespace(crop_loading_policy=DECLARED_UINT8)
    legacy = SimpleNamespace(crop_loading_policy=STORED_PIL)
    with pytest.raises(ValueError, match='different'):
        loader_policy(SimpleNamespace(datasets=[declared, legacy]))
    with pytest.raises(ValueError, match='validation'):
        training_preprocessing(declared, legacy)
    with pytest.raises(ValueError, match='disagrees'):
        training_preprocessing(declared, None, {'crop_loading_policy': STORED_PIL})
    with pytest.raises(ValueError, match='Checkpoint'):
        training_preprocessing(declared, None, checkpoint={})


@pytest.mark.parametrize('mixed', [False, True])
def test_generated_training_and_tar_datasets_keep_source_meaning(tmp_path, mixed):
    from spacr.io import generate_dataset_from_lists, _write_crop_tar, spacrDataset, TarImageDataset

    paths = []
    originals = {}
    for i in range(8):
        fmt = 3 if mixed and i % 2 else 2
        folder = tmp_path / str(i)
        folder.mkdir()
        path = folder / f'plate1_A01_1_{i}.png'
        Image.new('RGB', (2, 2), (200, 80, 25) if fmt == 2 else (25, 80, 200)).save(path)
        crops.write_crop_folder_marker(folder, fmt=fmt)
        originals[path] = path.read_bytes()
        paths.append(str(path))
    train, test = generate_dataset_from_lists(str(tmp_path / 'dataset'), [paths], ['a'],
                                              test_split=.25, group_by='cell')
    archive = tmp_path / 'dataset.tar'
    _write_crop_tar(paths, archive)
    datasets = [spacrDataset(train, ['a'], transform=np.asarray),
                spacrDataset(test, ['a'], transform=np.asarray),
                TarImageDataset(archive, transform=np.asarray)]
    for dataset in datasets:
        for sample in dataset:
            assert sample[0][0, 0].tolist() == [25, 80, 200]
    for path, original in originals.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize('policy', [DECLARED_UINT8, STORED_PIL])
def test_augmented_training_loaders_record_their_decoder(tmp_path, policy):
    from spacr.io import generate_loaders

    for name in ('a', 'b'):
        folder = tmp_path / 'train' / name
        folder.mkdir(parents=True)
        for i in range(4):
            Image.new('RGB', (4, 4), (30, 80, 200)).save(folder / f'{i}.png')
    train, val, _ = generate_loaders(str(tmp_path), classes=['a', 'b'], n_jobs=0,
                                    image_size=4, augment=True, validation_split=.25,
                                    crop_loading_policy=policy)
    assert loader_policy(train) == loader_policy(val) == policy
    assert training_preprocessing(train, val)['crop_loading_policy'] == policy


@pytest.mark.parametrize('recorded', [None, DECLARED_UINT8])
def test_real_checkpoint_drives_identical_folder_and_tar_predictions(tmp_path, monkeypatch, recorded):
    import torch
    from spacr import deep_spacr as deep
    from spacr import torch_artifacts as artifacts

    def build_model(*args):
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1))
        with torch.no_grad():
            model[-1].weight.copy_(torch.tensor([[1., 0., -1.]]))
            model[-1].bias.zero_()
        return model

    monkeypatch.setattr(artifacts, 'build_model_from_configuration', build_model)
    monkeypatch.setattr(deep, 'pick_device', lambda **kwargs: (torch.device('cpu'), ''))
    folder, _ = _folder(tmp_path, np.full((4, 4, 3), [25, 80, 200], dtype=np.uint8), 2)
    archive = _archive(folder, tmp_path)
    model_path = tmp_path / 'model.pth'
    artifacts.save_model_artifact(build_model(), model_path,
                                  preprocessing={'crop_loading_policy': recorded} if recorded else {})
    assert initialization_policy(model_path) == (recorded or STORED_PIL)
    direct = deep.apply_model(str(folder), str(model_path), image_size=4, normalize=False, n_jobs=0)
    archived = deep.apply_model_to_tar(dict(tar_path=str(archive), model_path=str(model_path),
                                            image_size=4, normalize=False, n_jobs=0,
                                            verbose=False, batch_size=1, score_threshold=.5))
    expected = torch.sigmoid(torch.tensor((175 if recorded else -175) / 255.)).item()
    assert direct['pred'].iloc[0] == pytest.approx(expected)
    assert archived['pred'].iloc[0] == pytest.approx(expected)


def test_real_training_checkpoint_retains_decoder_on_resume(tmp_path, monkeypatch):
    import torch
    from spacr import deep_spacr as deep
    from spacr.io import generate_loaders

    def build_model(*args, **kwargs):
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
        model.num_classes = 2
        return model

    monkeypatch.setattr('spacr.utils.choose_model', build_model)
    monkeypatch.setattr(deep, 'pick_device', lambda **kwargs: (torch.device('cpu'), ''))
    for label, value in [('a', 1024), ('b', 32768)]:
        folder = tmp_path / 'train' / label
        folder.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((4, 4), value + i * 256, dtype=np.uint16)).save(folder / f'{i}.png')
    loader, _, _ = generate_loaders(str(tmp_path), classes=['a', 'b'], image_size=4, n_jobs=0)
    out = tmp_path / 'model'
    out.mkdir()
    kwargs = dict(src=str(tmp_path), dst=str(out), model_type='tiny', train_loaders=loader,
                  epochs=1, num_classes=2, schedule=None, tensorboard=False, plot=False,
                  write_card=False, n_jobs=0)
    _, path = deep.train_model(**kwargs)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['preprocessing']['crop_loading_policy'] == DECLARED_UINT8
    assert initialization_policy(path) == DECLARED_UINT8
    kwargs.update(epochs=2, resume_checkpoint=path)
    deep.train_model(**kwargs)
    payloads = [torch.load(p, map_location='cpu', weights_only=False) for p in out.glob('*.pth')]
    assert any(p['training_state']['epoch'] == 2 for p in payloads)
    assert all(p['preprocessing']['crop_loading_policy'] == DECLARED_UINT8 for p in payloads)


@pytest.mark.parametrize('mixed', [False, True])
def test_fusion_preserves_policy_and_refuses_incompatible_models(tmp_path, monkeypatch, mixed):
    import torch
    from spacr import deep_spacr as deep, torch_artifacts as artifacts

    def build(*args):
        return torch.nn.Linear(3, 2)

    monkeypatch.setattr(artifacts, 'build_model_from_configuration', build)
    paths = []
    for i in range(2):
        path = tmp_path / f'model{i}.pth'
        policy = STORED_PIL if mixed and i else DECLARED_UINT8
        artifacts.save_model_artifact(build(), path, preprocessing={'crop_loading_policy': policy})
        paths.append(str(path))
    if mixed:
        with pytest.raises(ValueError, match='crop loading policy'):
            deep.model_fusion(paths, str(tmp_path / 'combined.pth'), device='cpu')
    else:
        deep.model_fusion(paths, str(tmp_path / 'combined.pth'), device='cpu')
        assert initialization_policy(tmp_path / 'combined_mean.pth') == DECLARED_UINT8
