"""Held-out checks belong in training and must not perturb optimizer updates."""
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import tifffile

from spacr import timeflows_model as tm
from spacr import timeflows_validation as validation

torch = pytest.importorskip('torch')


class Backbone(torch.nn.Module):
    channels, ps = 4, 8

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 8, stride=8)
        self.dropout = torch.nn.Dropout2d(0.25)

    def forward(self, x):
        return self.dropout(self.conv(x))


def pair(seed, shift=3):
    rng = np.random.default_rng(seed)
    labels = np.zeros((24, 28), np.int64)
    labels[5:9, 5:9] = 7
    return tm._Pair(rng.random(labels.shape, dtype=np.float32),
                    rng.random(labels.shape, dtype=np.float32), labels,
                    np.roll(labels, shift, axis=1))


def net():
    return tm.TimeflowsNet(Backbone())


def head(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if key.startswith(('head.', 'up.'))}


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_state(model, expected):
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key]), key


@pytest.mark.parametrize('representation', ['copy', 'float64', 'rgb', 'rgba', 'target'])
def test_holdout_rejects_duplicate_encoder_inputs_across_representations(representation):
    training, held = pair(1), pair(2)
    frame = training.frame_t.copy()
    if representation == 'float64':
        frame = frame.astype(np.float64)
    elif representation in ('rgb', 'rgba'):
        frame = np.stack([frame] * (3 if representation == 'rgb' else 4), axis=-1)
        if representation == 'rgba':
            frame[..., 3] = 123
    if representation == 'target':
        held.frame_t1 = frame
    else:
        held.frame_t = frame
    with pytest.raises(ValueError, match='also occurs'):
        validation.check_pair_holdout([training], [held])


def test_holdout_returns_stable_fingerprints_and_refuses_empty_validation():
    assert validation.check_pair_holdout([pair(1)], [pair(2)]) == validation.check_pair_holdout([pair(1)], [pair(2)])
    with pytest.raises(ValueError, match='held-out pair'):
        validation.check_pair_holdout([pair(1)], [])
    with pytest.raises(ValueError, match='held-out pair'):
        validation.validate_timeflows(net(), [])


def test_real_validation_reports_controls_and_restores_modes_rng_weights_and_gradients(monkeypatch):
    model = net()
    initial = head(model)
    with torch.no_grad():
        model.up.bias.add_(0.1)
    model.train()
    model.backbone.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    expected = snapshot(model)
    modes = [module.training for module in model.modules()]
    original = tm.predict_pair

    def stochastic(*args, **kwargs):
        torch.rand(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(tm, 'predict_pair', stochastic)
    rng = torch.get_rng_state().clone()
    report = validation.validate_timeflows(model, [pair(2, shift=12)], initial_head=initial)
    assert_state(model, expected)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.all(parameter.grad == 1) for parameter in model.parameters())
    arms = report['results']['overall']['arms']
    assert set(arms) == {'trained', 'initial_head', 'iou', 'zero_motion', 'oracle'}
    assert arms['oracle']['successor_accuracy'] == 1
    assert arms['zero_motion']['successor_accuracy'] == 0
    assert report['results']['motion']['at_least_1']['true_successors'] == 1
    assert report['copied_frame_control']['overall']['true_successors'] == 1


@pytest.mark.parametrize('failure_call', [1, 2, 3])
def test_failed_validation_restores_the_current_head_and_modes(monkeypatch, failure_call):
    model = net()
    initial = head(model)
    with torch.no_grad():
        model.up.bias.add_(0.1)
    expected = snapshot(model)
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    held = pair(2)
    target = tm.time_targets(held.labels_t, held.labels_t1)
    calls = []

    def predict(model, *args, **kwargs):
        model.eval()
        torch.rand(3)
        calls.append(1)
        if len(calls) == failure_call:
            raise RuntimeError('validation failed')
        return {'vector': target['vector'], 'successor': target['successor']}

    monkeypatch.setattr(tm, 'predict_pair', predict)
    with pytest.raises(RuntimeError, match='validation failed'):
        validation.validate_timeflows(model, [held], initial_head=initial)
    assert_state(model, expected)
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('malformed', ['keys', 'shape'])
def test_initial_head_snapshot_cannot_overwrite_other_model_state(malformed):
    model = net()
    initial = head(model)
    if malformed == 'keys':
        initial['backbone.conv.weight'] = model.backbone.conv.weight.detach().clone()
    else:
        initial[next(iter(initial))] = torch.zeros(1)
    expected = snapshot(model)
    with pytest.raises(ValueError, match='Initial head snapshot'):
        validation.validate_timeflows(model, [pair(2)], initial_head=initial)
    assert_state(model, expected)


def test_epoch_checks_and_controls_do_not_change_training_updates():
    torch.manual_seed(5)
    ordinary = net()
    monitored = copy.deepcopy(ordinary)
    training = [pair(1), pair(2)]
    held = [pair(3)]
    torch.manual_seed(19)
    expected = tm.train_timeflows(ordinary, training, head_steps=3, full_steps=2, seed=9)
    reports = []
    torch.manual_seed(19)
    actual = tm.train_timeflows(monitored, training, head_steps=3, full_steps=2, seed=9,
                               validation_pairs=held, on_validation=reports.append)
    assert actual == expected
    assert_state(monitored, snapshot(ordinary))
    assert [(r['stage'], r['step']) for r in reports] == [('initial', 0), ('head', 2), ('head', 3), ('full', 2)]
    assert [r['completed_epochs'] for r in reports] == [0, 1, 1, 1]
    assert reports[0]['training_loss'] is None
    assert reports[-1]['training_loss'] == actual[-1]
    assert all(r['epoch_size'] == 2 for r in reports)


def test_explicit_interval_checks_stage_ends_once_and_logs_each_report():
    reports, lines = [], []
    losses = tm.train_timeflows(net(), [pair(1)], head_steps=3, full_steps=2,
                               validation_pairs=[pair(2)], validation_every=2,
                               on_validation=reports.append, log=lines.append)
    assert len(losses) == 5
    assert [(r['stage'], r['step']) for r in reports] == [('initial', 0), ('head', 2), ('head', 3), ('full', 2)]
    logged = [json.loads(line.removeprefix('validation ')) for line in lines if line.startswith('validation ')]
    assert [(r['stage'], r['step']) for r in logged] == [(r['stage'], r['step']) for r in reports]


def test_validation_can_report_through_the_existing_log_without_a_callback():
    lines = []
    assert tm.train_timeflows(net(), [pair(1)], head_steps=0, full_steps=0,
                             validation_pairs=[pair(2)], log=lines.append) == []
    assert len(lines) == 1
    report = json.loads(lines[0].removeprefix('validation '))
    assert report['stage'] == 'initial' and report['training_loss'] is None


@pytest.mark.parametrize('device, expected_devices', [('cpu', []), ('cuda', [3]), ('cuda:2', [2])])
def test_validation_routes_rng_snapshot_without_initial_head_or_real_gpu(monkeypatch, device, expected_devices):
    seen = []
    held = pair(2)
    targets = tm.time_targets(held.labels_t, held.labels_t1)

    @contextmanager
    def fork_rng(*, devices):
        seen.append(devices)
        yield

    def predict(model, *frames, device):
        model.eval()
        return {'vector': targets['vector'], 'successor': targets['successor']}

    monkeypatch.setattr(torch.random, 'fork_rng', fork_rng)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    monkeypatch.setattr(tm, 'predict_pair', predict)
    model = net()
    report = validation.validate_timeflows(model, [held], device=device)
    assert seen == [expected_devices]
    assert set(report['results']['overall']['arms']) == {'trained', 'iou', 'zero_motion', 'oracle'}
    assert model.training
    assert report['temporal_assignment'] == validation.temporal_assignment_policy()


@pytest.mark.parametrize('interval', [0, -1, True, 0.5])
def test_invalid_validation_cadence_fails_before_an_update(interval):
    model = net()
    expected = snapshot(model)
    with pytest.raises(ValueError, match='positive integer'):
        tm.train_timeflows(model, [pair(1)], validation_pairs=[pair(2)], validation_every=interval)
    assert_state(model, expected)


def test_validation_requires_pairs_and_rejects_training_overlap_before_an_update():
    model = net()
    expected = snapshot(model)
    for kwargs in ({'validation_every': 1}, {'on_validation': lambda report: None},
                   {'validation_pairs': []}, {'validation_pairs': [pair(1)]}):
        with pytest.raises(ValueError):
            tm.train_timeflows(model, [pair(1)], **kwargs)
        assert_state(model, expected)
    with pytest.raises(ValueError, match='nonempty training'):
        tm.train_timeflows(model, [], validation_pairs=[pair(2)])


def movie(root, seed, segmentation):
    rng = np.random.default_rng(seed)
    for folder in ('01', f'01_{segmentation}/SEG', '01_GT/TRA'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for frame in range(2):
        labels = np.zeros((24, 28), np.uint16)
        labels[5:9, 5 + frame:9 + frame] = 7
        tifffile.imwrite(root / '01' / f't{frame:03}.tif', rng.integers(0, 1000, labels.shape, dtype=np.uint16))
        tifffile.imwrite(root / f'01_{segmentation}/SEG' / f'man_seg{frame:03}.tif', labels)
        tifffile.imwrite(root / '01_GT/TRA' / f'man_track{frame:03}.tif', labels)
    return root


def cli_setup(tmp_path, monkeypatch, *, duplicate=False):
    training = movie(tmp_path / 'training', 1, 'ST')
    held = movie(tmp_path / 'heldout', 1 if duplicate else 2, 'GT')
    monkeypatch.setitem(sys.modules, 'cellpose', types.SimpleNamespace(models=types.SimpleNamespace(
        CellposeModel=lambda **kwargs: types.SimpleNamespace(net=Backbone()))))
    monkeypatch.setattr(tm, 'CellposeSamFeatures', lambda backbone: backbone)
    output = tmp_path / 'result.pt'
    args = ['--movies', str(training), '--validation-movies', str(held), '--out', str(output),
            '--head-steps', '1', '--full-steps', '1', '--device', 'cpu']
    return training, held, output, args


def test_cli_runs_checks_and_persists_truth_scope_and_completion(tmp_path, monkeypatch):
    training, held, output, args = cli_setup(tmp_path, monkeypatch)
    assert tm.main(args) == 0
    assert output.exists()
    metadata = json.loads(output.with_suffix('.pt.json').read_text())
    record = metadata['validation']
    assert record['enabled'] and record['reports'] == 3
    assert record['segmentation'] == 'GT' and record['pairs'] == 1
    assert record['interval_updates'] == record['epoch_size'] == 1
    assert record['temporal_assignment'] == validation.temporal_assignment_policy()
    assert record['model_code_sha256'] == hashlib.sha256(Path(tm.__file__).read_bytes()).hexdigest()
    assert record['scoring_code_sha256'] == hashlib.sha256(Path(validation.__file__).read_bytes()).hexdigest()
    events = [json.loads(line) for line in output.with_suffix('.pt.validation.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'configuration'
    assert [(event['stage'], event['step']) for event in events[1:-1]] == [('initial', 0), ('head', 1), ('full', 1)]
    assert events[-1] == {'event': 'training_complete', 'updates': 2}
    assert events[1]['results']['overall']['arms']['oracle']['successor_accuracy'] == 1
    assert record['input_fingerprints'] == validation.check_pair_holdout(
        tm.ctc_pairs(str(training)), tm.ctc_pairs(str(held), segmentation='GT'))


def test_cli_rejects_duplicate_frames_before_writing_outputs(tmp_path, monkeypatch):
    _, _, output, args = cli_setup(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match='also occurs'):
        tm.main(args)
    assert not list(tmp_path.glob('result*'))


def test_cli_rejects_training_movie_alias_and_orphan_validation_interval(tmp_path, monkeypatch):
    training, _, output, args = cli_setup(tmp_path, monkeypatch)
    alias = tmp_path / 'alias'
    alias.symlink_to(training, target_is_directory=True)
    args[args.index('--validation-movies') + 1] = str(alias)
    with pytest.raises(SystemExit):
        tm.main(args)
    with pytest.raises(SystemExit):
        tm.main(['--movies', str(training), '--out', str(output), '--validation-every', '1'])
    assert not output.exists()


def test_cli_preserves_existing_validation_report_and_model(tmp_path, monkeypatch):
    _, _, output, args = cli_setup(tmp_path, monkeypatch)
    output.write_bytes(b'original model')
    report = output.with_suffix('.pt.validation.jsonl')
    report.write_text('original report')
    with pytest.raises(FileExistsError):
        tm.main(args)
    assert report.read_text() == 'original report'
    assert output.read_bytes() == b'original model'


def test_cli_failed_check_leaves_partial_log_without_complete_model(tmp_path, monkeypatch):
    _, _, output, args = cli_setup(tmp_path, monkeypatch)
    original = validation.validate_timeflows
    calls = []

    def validate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('heldout failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(validation, 'validate_timeflows', validate)
    with pytest.raises(RuntimeError, match='heldout failure'):
        tm.main(args)
    assert not output.exists()
    assert not output.with_suffix('.pt.json').exists()
    events = [json.loads(line) for line in output.with_suffix('.pt.validation.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events] == ['configuration', 'validation']


def test_zero_update_cli_can_record_an_initial_control_without_index_error(tmp_path, monkeypatch):
    _, _, output, args = cli_setup(tmp_path, monkeypatch)
    args[args.index('--head-steps') + 1] = '0'
    args[args.index('--full-steps') + 1] = '0'
    assert tm.main(args) == 0
    metadata = json.loads(output.with_suffix('.pt.json').read_text())
    assert metadata['final_loss'] is None and metadata['validation']['reports'] == 1


def test_ctc_loader_rejects_unknown_segmentation_source(tmp_path):
    with pytest.raises(ValueError, match='segmentation'):
        tm.ctc_pairs(str(tmp_path), segmentation='invented')
