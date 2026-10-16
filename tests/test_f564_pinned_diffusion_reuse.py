"""Pinned local reuse mechanics with random tiny weights, never quality evidence."""
import copy
import hashlib
import os
import pickle

import pytest
import torch
from torch import nn

from spacr import attribution as cf


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        score = x.mean((1, 2, 3)) * self.scale
        return torch.stack((-score, score), 1)


@pytest.fixture(scope='module')
def fitted_generator():
    """One real tiny CPU fit establishes producer mechanics, never phenotype quality."""
    crops = torch.linspace(-1, 1, 8 * 1 * 8 * 8).reshape(8, 1, 8, 8)
    with torch.random.fork_rng(devices=[]):
        generator, history = cf._train_counterfactual_diffusion(
            Classifier(), crops, epochs=1, batch_size=8, width=8,
            timesteps=8, sample_steps=2, strength=0.5, augment=False, seed=7)
    assert len(history) == 1 and torch.isfinite(torch.tensor(history[0]['denoising']))
    return generator


@pytest.fixture
def sample(tmp_path, fitted_generator):
    model = Classifier()
    crops = torch.linspace(-1, 1, 8 * 1 * 8 * 8).reshape(8, 1, 8, 8)
    description = {'schema': 1, 'channel_ids': ['measured_plane_4'],
                   'class_labels': ['control', 'condition'],
                   'condition_key': 'class',
                   'preprocessing': {'pipeline': 'synthetic_model_input_v1',
                                     'crop_loading_policy': 'declared_synthetic_tensor_v1',
                                     'normalization': None, 'center_crop': [8, 8],
                                     'selected_channels_1based': [1],
                                     'selection_semantics': 'zero_unselected_stored_rgb_planes',
                                     'classifier_preprocessing': None}}
    contract = cf._cf_input_contract(model, crops, description, [])
    generator = copy.deepcopy(fitted_generator)
    payload = {'artifact_type': 'spacr.counterfactual_diffusion',
               'artifact_version': 1, 'architecture': 'counterfactual_ddpm_unet_v1',
               'state_dict': generator.state_dict(),
               'config': {'channels': 1, 'side': 8, 'n_classes': 2, 'width': 8,
                          'timesteps': 8, 'sample_steps': 2, 'strength': 0.5,
                          'guidance_scale': 4.0}, 'levels': [],
               'input_contract': contract,
               'provenance': {'source': 'trained_here',
                              'training_crops': 8, 'training_epochs': 1,
                              'pretraining_independence': 'unverified'}}
    path = tmp_path / 'counterfactual_diffusion.pt'
    torch.save(payload, path)
    return model, crops, description, generator.eval(), payload, path


def pinned(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_weights_roundtrip_preserves_preprocessing_and_rng(sample):
    model, crops, description, generator, payload, path = sample
    rng = torch.random.get_rng_state().clone()
    before = path.read_bytes()
    restored, sha = cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])
    assert sha == pinned(path)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert path.read_bytes() == before
    code = torch.tensor([[1., 0.], [0., 1.]])
    for actual, expected in zip(restored.state_dict().values(), generator.state_dict().values()):
        assert torch.equal(actual, expected)
    assert torch.equal(restored.encode(crops[:2]), generator.encode(crops[:2]))
    assert torch.equal(restored.decode(generator.encode(crops[:2]), code),
                       generator.decode(generator.encode(crops[:2]), code))
    assert model.training


@pytest.mark.parametrize('change', ['hash', 'legacy', 'classes', 'preprocessing',
                                    'channels', 'classifier', 'state', 'std',
                                    'config', 'architecture'])
def test_invalid_reuse_refuses_before_trainer_or_classifier_mutation(sample, monkeypatch, change):
    model, crops, description, generator, payload, path = sample
    def forbidden(*args, **kwargs):
        pytest.fail('invalid selected checkpoint reached training or scoring')
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    before_crops = crops.clone()
    original = path.read_bytes()
    expected = pinned(path)
    if change == 'hash':
        expected = '0' * 64
    elif change == 'legacy':
        del payload['artifact_type']
    elif change in ['classes', 'preprocessing', 'channels']:
        field = {'classes': 'class_labels', 'preprocessing': 'preprocessing',
                 'channels': 'channel_ids'}[change]
        description[field] = {'changed': True} if field == 'preprocessing' else ['wrong', 'order']
    elif change == 'classifier':
        with torch.no_grad():
            model.scale.add_(1)
    elif change == 'state':
        payload['state_dict']['inp.weight'] = torch.zeros(1)
    elif change == 'std':
        payload['state_dict']['std'].zero_()
    elif change == 'config':
        payload['config']['width'] = 10 ** 8
    else:
        payload['architecture'] = 'unknown'
    if change not in ['hash', 'classes', 'preprocessing', 'channels', 'classifier']:
        torch.save(payload, path)
        expected = pinned(path)
        original = path.read_bytes()
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, generator='diffusion', checkpoint=path,
                                  checkpoint_sha256=expected, input_contract=description)
    assert model.training
    assert torch.equal(crops, before_crops)
    assert path.read_bytes() == original


def test_reuse_runs_real_report_without_training_or_rewriting_selected_input(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    def forbidden(*args, **kwargs):
        pytest.fail('explicit reuse must never train')
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', forbidden)
    monkeypatch.setattr(cf, '_train_counterfactual_generator', forbidden)
    monkeypatch.setattr(cf, '_write_counterfactual_outputs', lambda *args: None)
    before = path.read_bytes()
    summary, rows, frames = cf._counterfactual_report(
        model, crops, generator='diffusion', checkpoint=path,
        checkpoint_sha256=pinned(path), input_contract=description,
        out_dir=str(path.parent), steps=3, show=1)
    assert summary['epochs'] == 0
    assert summary['fitted_crops'] == 0
    assert summary['current_reference_crops'] == 6
    assert summary['checkpoint_provenance_source'] == 'trained_here'
    assert summary['checkpoint_sha256'] == pinned(path)
    assert summary['pretraining_independence'] == 'unverified'
    assert summary['scoring_scope'] == 'descriptive_current_inputs'
    assert summary['heldout_crops'] == len(rows) == 2
    assert frames.shape == (1, 3, 1, 8, 8)
    assert path.read_bytes() == before
    assert torch.isnan(torch.tensor(summary['reconstruction_mse']))


def test_explicit_checkpoint_never_falls_back_to_autoencoder(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    monkeypatch.setattr(cf, '_train_counterfactual_generator',
                        lambda *args, **kwargs: pytest.fail('fallback training'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, checkpoint=path,
                                  checkpoint_sha256=pinned(path), input_contract=description)
    assert model.training


def test_changed_open_file_is_refused(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    original_load = torch.load
    def changed(handle, **kwargs):
        result = original_load(handle, **kwargs)
        with open(path, 'ab') as other:
            other.write(b'changed')
        return result
    monkeypatch.setattr(torch, 'load', changed)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])


def test_restricted_load_never_retries_unsafe_pickle(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    calls = []
    def reject(handle, **kwargs):
        calls.append(kwargs)
        raise pickle.UnpicklingError('unsafe object')
    monkeypatch.setattr(torch, 'load', reject)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])
    assert calls == [{'map_location': 'cpu', 'weights_only': True}]


def test_report_producer_records_actual_contract_without_training(sample, tmp_path, monkeypatch):
    model, crops, description, generator, payload, path = sample
    output = tmp_path / 'new'
    calls = []
    def ready(*args, **kwargs):
        calls.append(kwargs)
        return generator, [{'reconstruction': 0.25}]
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', ready)
    monkeypatch.setattr(cf, '_write_counterfactual_outputs', lambda *args: None)
    summary, _, _ = cf._counterfactual_report(
        model, crops, generator='diffusion', input_contract=description,
        out_dir=str(output), epochs=2, steps=3, show=0)
    made = output / 'counterfactual_diffusion.pt'
    data = torch.load(made, map_location='cpu', weights_only=True)
    assert len(calls) == 1 and calls[0]['epochs'] == 2
    assert data['input_contract'] == payload['input_contract']
    assert data['provenance']['training_crops'] == 6
    assert data['provenance']['training_epochs'] == 2
    assert summary['checkpoint_sha256'] == pinned(made)
    assert summary['epochs'] == 2
    assert data['artifact_version'] == 1


@pytest.mark.parametrize('provenance', [None, {},
    {'source': 'trained_here', 'training_crops': 6, 'training_epochs': 0,
     'pretraining_independence': 'unverified'},
    {'source': 'trained_here', 'training_crops': True, 'training_epochs': 2,
     'pretraining_independence': 'unverified'},
    {'source': 'trained_here', 'training_crops': 6, 'training_epochs': 2,
     'pretraining_independence': 'verified'},
    {'source': 'synthetic_fixture', 'purpose': 'mechanics_only',
     'training_crops': 0, 'training_epochs': 0,
     'pretraining_independence': 'unverified'}])
def test_missing_or_malformed_provenance_is_not_verified(sample, provenance):
    model, crops, description, generator, payload, path = sample
    payload['provenance'] = provenance
    torch.save(payload, path)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])


def test_nonfinite_input_metadata_uses_localized_refusal(sample):
    model, crops, description, generator, payload, path = sample
    description['preprocessing']['offset'] = float('nan')
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_input_contract(model, crops, description, [])


def test_application_forwards_exact_reuse_contract_without_reprocessing(sample, monkeypatch, tmp_path):
    from spacr.deep_spacr import _run_counterfactuals
    model, crops, description, generator, payload, path = sample
    captured = []
    def report(actual_model, actual_crops, **kwargs):
        assert actual_model is model
        assert torch.equal(actual_crops, crops)
        captured.append(kwargs)
        return {'flip_rate': 0., 'baseline_flip_rate': 0., 'monotone_fraction': 0.,
                'median_edit_l1': 0.}, [], None
    monkeypatch.setattr(cf, '_counterfactual_report', report)
    settings = {'counterfactual_generator': 'diffusion',
                'counterfactual_checkpoint': str(path),
                'counterfactual_checkpoint_sha256': pinned(path)}
    _run_counterfactuals(settings, model, [crops[:4], crops[4:]],
                        [str(i) for i in range(8)], str(tmp_path), 'cpu',
                        input_contract=description)
    assert captured[0]['input_contract'] is description
    assert captured[0]['checkpoint'] == str(path)
    assert captured[0]['checkpoint_sha256'] == pinned(path)
    assert model.training


def test_legacy_default_route_does_not_require_or_invent_reuse_metadata(sample, tmp_path, monkeypatch):
    model, crops, description, generator, payload, path = sample
    calls = []
    def trainer(*args, **kwargs):
        calls.append(kwargs)
        return generator, []
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', trainer)
    monkeypatch.setattr(cf, '_write_counterfactual_outputs', lambda *args: None)
    summary, _, _ = cf._counterfactual_report(
        model, crops, generator='diffusion', out_dir=str(tmp_path / 'legacy'),
        epochs=0, steps=3, show=0)
    saved = torch.load(tmp_path / 'legacy' / 'counterfactual_diffusion.pt',
                       weights_only=True, map_location='cpu')
    assert len(calls) == 1 and calls[0]['epochs'] == 0
    assert set(saved) == {'state_dict', 'config', 'levels'}
    assert 'checkpoint_sha256' not in summary


def test_compressed_checkpoint_refused_before_torch_load(sample, monkeypatch, tmp_path):
    import zipfile
    model, crops, description, generator, payload, path = sample
    compressed = tmp_path / 'compressed.pt'
    with zipfile.ZipFile(compressed, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.pkl', b'not a model' * 2000)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('decoded compressed archive'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(compressed, pinned(compressed), payload['input_contract'])


def test_oversized_configuration_refuses_before_reconstruction(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    payload['config']['n_classes'] = 2 ** 30
    torch.save(payload, path)
    monkeypatch.setattr(cf, '_CounterfactualDiffusion',
                        lambda *a, **k: pytest.fail('allocated unbounded generator'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])


@pytest.mark.parametrize('alias', [False, True])
def test_selected_input_cannot_be_overwritten_by_report_output(sample, monkeypatch, alias):
    model, crops, description, generator, payload, path = sample
    output = path.parent / 'counterfactual_summary.csv'
    if alias:
        os.link(path, output)
    else:
        path.rename(output)
        path = output
    before = path.read_bytes()
    monkeypatch.setattr(cf, '_cf_load_checkpoint',
                        lambda *a: pytest.fail('output collision reached checkpoint load'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, generator='diffusion', checkpoint=path,
                                  checkpoint_sha256=pinned(path), input_contract=description,
                                  out_dir=str(path.parent))
    assert path.read_bytes() == before and model.training


def test_saved_schedule_must_match_declared_configuration(sample):
    model, crops, description, generator, payload, path = sample
    payload['state_dict']['abar'][0] = 0.5
    torch.save(payload, path)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), payload['input_contract'])


@pytest.mark.parametrize('change', ['schema_bool', 'missing_pipeline', 'missing_policy',
    'crop_shape', 'zero_std', 'bad_mean', 'channel_mask', 'selection_semantics',
    'classifier_metadata'])
def test_incomplete_or_mismatched_preprocessing_refuses_before_classifier(sample, monkeypatch, change):
    model, crops, description, generator, payload, path = sample
    if change == 'schema_bool':
        description['schema'] = True
    elif change.startswith('missing_'):
        del description['preprocessing'][{'missing_pipeline': 'pipeline',
                                          'missing_policy': 'crop_loading_policy'}[change]]
    elif change == 'crop_shape':
        description['preprocessing']['center_crop'] = [4, 8]
    elif change == 'zero_std':
        description['preprocessing']['normalization'] = {'mean': [0.], 'std': [0.]}
    elif change == 'bad_mean':
        description['preprocessing']['normalization'] = {'mean': [True], 'std': [1.]}
    elif change == 'channel_mask':
        description['preprocessing']['selected_channels_1based'] = [True]
    elif change == 'selection_semantics':
        description['preprocessing']['selection_semantics'] = 'drop_and_reorder'
    else:
        description['preprocessing']['classifier_preprocessing'] = []
    monkeypatch.setattr(model, 'forward', lambda *a: pytest.fail('metadata refusal scored classifier'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_input_contract(model, crops, description, [])
    assert model.training


@pytest.mark.parametrize('shape', [(8,), (8, 8), (8, 1, 8)])
def test_selected_reuse_shape_refuses_before_legacy_channel_indexing(sample, shape):
    model, crops, description, generator, payload, path = sample
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, torch.zeros(shape), generator='diffusion',
                                  checkpoint=path, checkpoint_sha256=pinned(path),
                                  input_contract=description)
    assert model.training


@pytest.mark.parametrize('change', ['blank_pipeline', 'blank_policy', 'normalization_list'])
def test_unusable_preprocessing_refuses_before_reading_classifier_or_checkpoint(sample, monkeypatch, change):
    model, crops, description, generator, payload, path = sample
    if change == 'blank_pipeline':
        description['preprocessing']['pipeline'] = '  '
    elif change == 'blank_policy':
        description['preprocessing']['crop_loading_policy'] = ''
    else:
        description['preprocessing']['normalization'] = [0., 1.]
    original_description = copy.deepcopy(description)
    original_crops = crops.clone()
    original_checkpoint = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail('invalid preprocessing reached classifier state, checkpoint loading or training')

    monkeypatch.setattr(model, 'state_dict', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    monkeypatch.setattr(cf, '_cf_load_checkpoint', forbidden)
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', forbidden)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, generator='diffusion', checkpoint=path,
                                  checkpoint_sha256=pinned(path), input_contract=description)
    assert description == original_description
    assert torch.equal(crops, original_crops)
    assert path.read_bytes() == original_checkpoint
    assert model.training


def test_nonfinite_classifier_weights_cannot_certify_a_reuse_contract(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    with torch.no_grad():
        model.scale.fill_(float('nan'))
    original = path.read_bytes()
    original_crops = crops.clone()
    monkeypatch.setattr(cf, '_cf_load_checkpoint',
                        lambda *args: pytest.fail('nonfinite classifier reached checkpoint load'))
    monkeypatch.setattr(model, 'forward', lambda *args: pytest.fail('nonfinite classifier scored crops'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, generator='diffusion', checkpoint=path,
                                  checkpoint_sha256=pinned(path), input_contract=description)
    assert model.training and torch.isnan(model.scale)
    assert path.read_bytes() == original
    assert torch.equal(crops, original_crops)


@pytest.mark.parametrize('change', ['unknown_config_key', 'wrong_channels', 'wrong_code_levels',
                                    'nonfinite_strength', 'excessive_guidance', 'missing_state_key',
                                    'missing_config_field', 'unaligned_side'])
def test_pinned_but_incompatible_state_never_allocates_or_applies_untrusted_weights(sample, monkeypatch, change):
    model, crops, description, generator, payload, path = sample
    if change == 'unknown_config_key':
        payload['config']['device'] = 'cuda'
    elif change == 'wrong_channels':
        payload['config']['channels'] = 2
    elif change == 'wrong_code_levels':
        payload['levels'] = ['condition', 'control']
    elif change == 'nonfinite_strength':
        payload['config']['strength'] = float('nan')
    elif change == 'excessive_guidance':
        payload['config']['guidance_scale'] = 101.
    elif change == 'missing_config_field':
        del payload['config']['timesteps']
    elif change == 'unaligned_side':
        payload['config']['side'] = 6
    else:
        del payload['state_dict']['inp.weight']
    torch.save(payload, path)
    original = path.read_bytes()
    original_crops = crops.clone()
    original_classifier = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        pytest.fail('incompatible artifact reached reconstruction, weight application or training')

    if change == 'missing_state_key':
        monkeypatch.setattr(cf._CounterfactualDiffusion, 'load_state_dict', forbidden)
    else:
        monkeypatch.setattr(cf, '_CounterfactualDiffusion', forbidden)
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', forbidden)
    monkeypatch.setattr(model, 'forward', forbidden)
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._counterfactual_report(model, crops, generator='diffusion', checkpoint=path,
                                  checkpoint_sha256=pinned(path), input_contract=description)
    assert path.read_bytes() == original
    assert torch.equal(crops, original_crops)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert model.training
    assert all(torch.equal(value, original_classifier[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize('route', ['report', 'sequences'])
def test_reference_preservation_refuses_non_nchw_before_any_model_work(monkeypatch, route):
    model = Classifier()
    crops = torch.zeros(4, 8, 8)
    original = crops.clone()

    def forbidden(*args, **kwargs):
        pytest.fail('invalid preservation input reached classifier or trainer')

    monkeypatch.setattr(model, 'forward', forbidden)
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion', forbidden)
    monkeypatch.setattr(cf, '_train_counterfactual_generator', forbidden)
    with pytest.raises(ValueError, match='Reference preservation requires NCHW crop tensors'):
        if route == 'report':
            cf._counterfactual_report(model, crops, generator='diffusion', preserve_channels=[0])
        else:
            cf._counterfactual_sequences(model, object(), crops, preserve_channels=[0])
    assert model.training and torch.equal(crops, original)


def test_explicit_normalization_roundtrips_and_changed_values_refuse(sample, monkeypatch):
    model, crops, description, generator, payload, path = sample
    normalization = {'mean': [.25], 'std': [.5]}
    description['preprocessing']['normalization'] = copy.deepcopy(normalization)
    contract = cf._cf_input_contract(model, crops, description, [])
    assert contract['preprocessing']['normalization'] == normalization
    payload['input_contract'] = contract
    torch.save(payload, path)
    original = path.read_bytes()
    restored, actual_sha = cf._cf_load_checkpoint(path, pinned(path), contract)
    assert actual_sha == pinned(path)
    assert all(torch.equal(value, generator.state_dict()[key])
               for key, value in restored.state_dict().items())
    description['preprocessing']['normalization']['mean'][0] = .75
    changed = cf._cf_input_contract(model, crops, description, [])
    monkeypatch.setattr(cf, '_CounterfactualDiffusion',
                        lambda *args, **kwargs: pytest.fail('changed normalization reached reconstruction'))
    with pytest.raises(ValueError, match='counterfactual checkpoint'):
        cf._cf_load_checkpoint(path, pinned(path), changed)
    assert path.read_bytes() == original
    assert contract['preprocessing']['normalization'] == normalization
    assert model.training


def test_existing_unrelated_output_files_do_not_block_verified_reuse(sample, monkeypatch, tmp_path):
    model, crops, description, generator, payload, path = sample
    model.eval()
    output = tmp_path / 'reports'
    output.mkdir()
    sentinel = output / 'my-notes.txt'
    sentinel.write_bytes(b'unrelated user notes')
    original = path.read_bytes()
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion',
                        lambda *args, **kwargs: pytest.fail('selected reuse reached training'))
    monkeypatch.setattr(cf, '_write_counterfactual_outputs', lambda *args: None)
    summary, rows, frames = cf._counterfactual_report(
        model, crops, generator='diffusion', checkpoint=path,
        checkpoint_sha256=pinned(path), input_contract=description,
        out_dir=str(output), steps=2, show=0)
    assert summary['checkpoint_sha256'] == pinned(path)
    assert summary['fitted_crops'] == 0
    assert len(rows) == summary['heldout_crops']
    assert set(output.iterdir()) == {sentinel}
    assert sentinel.read_bytes() == b'unrelated user notes'
    assert path.read_bytes() == original
    assert not model.training


def test_first_reuse_hands_new_output_directory_to_writer_without_touching_input(sample, monkeypatch, tmp_path):
    model, crops, description, generator, payload, path = sample
    model.eval()
    output = tmp_path / 'new report directory'
    assert not output.exists()
    original = path.read_bytes()
    handed = []

    def write(directory, summary, rows, frames):
        assert directory == str(output)
        assert not output.exists()
        assert summary['fitted_crops'] == 0
        assert summary['checkpoint_sha256'] == pinned(path)
        output.mkdir()
        (output / 'writer-result.txt').write_bytes(b'owned output handoff')
        handed.append((len(rows), frames.shape))

    monkeypatch.setattr(cf, '_write_counterfactual_outputs', write)
    monkeypatch.setattr(cf, '_train_counterfactual_diffusion',
                        lambda *args, **kwargs: pytest.fail('first reuse must not train'))
    summary, rows, frames = cf._counterfactual_report(
        model, crops, generator='diffusion', checkpoint=path,
        checkpoint_sha256=pinned(path), input_contract=description,
        out_dir=str(output), steps=2, show=0)
    assert handed == [(len(rows), frames.shape)]
    assert (output / 'writer-result.txt').read_bytes() == b'owned output handoff'
    assert path.read_bytes() == original
