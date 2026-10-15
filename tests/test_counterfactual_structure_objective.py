"""The optional diffusion objective penalizes layout changes, not uniform offsets."""

import pytest
import torch
from torch import nn

from spacr import attribution


def _classifier():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2))


def _train(**options):
    x = torch.linspace(0, 1, 4 * 8 * 8).reshape(4, 1, 8, 8)
    return attribution._train_counterfactual_diffusion(
        _classifier(), x, labels=torch.tensor([0, 1, 0, 1]),
        target_probs=torch.eye(2), epochs=1,
        batch_size=4, width=8, timesteps=8, sample_steps=2, strength=.35,
        augment=False, seed=23, **options)


def test_structure_objective_trains_and_reports_its_actual_component():
    generator, history = _train(structure_weight=.4)
    assert generator.structure_weight == .4
    assert len(history) == 1
    assert history[0]['structure'] > 0
    assert history[0]['objective'] == pytest.approx(
        history[0]['denoising'] + .4 * history[0]['structure'])
    assert all(torch.isfinite(parameter).all() for parameter in generator.parameters())


def test_disabled_structure_objective_reproduces_legacy_training():
    old, old_history = _train()
    disabled, disabled_history = _train(structure_weight=0)
    assert old_history == disabled_history
    assert all(torch.equal(value, disabled.state_dict()[key])
               for key, value in old.state_dict().items())


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf')])
def test_invalid_structure_weights_refuse(weight):
    with pytest.raises(ValueError):
        _train(structure_weight=weight)


@pytest.mark.parametrize('strength', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_strength_refuses_before_sampling(strength):
    with pytest.raises(ValueError):
        attribution._CounterfactualDiffusion(1, 8, 2, width=8, strength=strength)


def test_layout_loss_ignores_uniform_channel_offsets_but_detects_movement():
    source = torch.zeros(1, 2, 16, 16)
    source[:, 0, 3:7, 4:8] = 1
    source[:, 1, 9:13, 9:13] = 2
    shifted = torch.roll(source, 3, dims=3)
    offset = source + torch.tensor([7., -3.]).reshape(1, 2, 1, 1)
    loss = attribution._cf_structure_loss
    assert float(loss(offset, source)) == pytest.approx(0, abs=1e-10)
    assert float(loss(shifted, source)) > 0
    assert float(loss(source.flip(1), source)) > 0
    assert float(loss(source, source)) == 0


def test_layout_loss_has_finite_descent_gradient_without_reference_gradient():
    source = torch.zeros(1, 1, 8, 8, requires_grad=True)
    moved = torch.eye(8).reshape(1, 1, 8, 8).requires_grad_()
    loss = attribution._cf_structure_loss(moved, source)
    loss.backward()
    gradient = moved.grad
    assert torch.isfinite(gradient).all()
    after = attribution._cf_structure_loss(moved - .1 * gradient, source)
    assert after < loss
    assert source.grad is None


@pytest.mark.parametrize('bad', [torch.zeros(1, 1, 8, 8),
                                torch.zeros(1, 2, 8, 8, dtype=torch.float64),
                                torch.full((1, 2, 8, 8), float('nan'))])
def test_layout_loss_refuses_broadcasting_dtype_changes_and_nonfinite_inputs(bad):
    with pytest.raises(ValueError):
        attribution._cf_structure_loss(bad, torch.zeros(1, 2, 8, 8))


def test_report_persists_layout_objective_and_reconstructs_safe_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(attribution, '_write_counterfactual_outputs', lambda *args: None)
    x = torch.linspace(0, 1, 8 * 8 * 8).reshape(8, 1, 8, 8)
    summary, rows, frames = attribution._counterfactual_report(
        _classifier(), x, conditions=['a', 'b'] * 4, epochs=1, steps=2,
        show=1, generator='diffusion', out_dir=str(tmp_path),
        generator_options=dict(width=8, timesteps=8, sample_steps=2,
                               strength=.35, structure_weight=.4))
    payload = torch.load(tmp_path / 'counterfactual_diffusion.pt', weights_only=True)
    rebuilt = attribution._CounterfactualDiffusion(**payload['config'])
    rebuilt.load_state_dict(payload['state_dict'], strict=True)
    assert rebuilt.structure_weight == summary['structure_weight'] == .4
    assert rebuilt.strength == .35
    assert rows and frames.shape == (1, 2, 1, 8, 8)


@pytest.mark.parametrize('kind', ['diffusion', 'autoencoder'])
def test_application_passes_diffusion_options_without_losing_zero_strength(kind, monkeypatch, tmp_path):
    from spacr.deep_spacr import _run_counterfactuals

    captured = []
    summary = dict(flip_rate=0, baseline_flip_rate=0, monotone_fraction=0, median_edit_l1=0)
    def report(*args, **kwargs):
        captured.append(kwargs)
        return summary, [], []
    monkeypatch.setattr(attribution, '_counterfactual_report', report)
    result = _run_counterfactuals(
        dict(counterfactual_generator=kind, counterfactual_strength=0,
             counterfactual_structure_weight=.4), _classifier(),
        [torch.zeros(4, 1, 8, 8)], ['a'] * 4, str(tmp_path), 'cpu')
    assert result == summary
    expected = dict(strength=0, structure_weight=.4) if kind == 'diffusion' else {}
    assert captured[0]['generator_options'] == expected


@pytest.mark.parametrize('key', ['counterfactual_strength', 'counterfactual_structure_weight'])
def test_application_leaves_invalid_value_refusal_to_translated_validators(key, tmp_path, capsys):
    from spacr.deep_spacr import _run_counterfactuals

    settings = dict(counterfactual_generator='diffusion', counterfactual_epochs=1)
    settings[key] = None
    result = _run_counterfactuals(settings, _classifier(), [torch.zeros(4, 1, 8, 8)],
                                  ['a'] * 4, str(tmp_path), 'cpu')
    assert result is None
    assert 'Counterfactuals skipped:' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
