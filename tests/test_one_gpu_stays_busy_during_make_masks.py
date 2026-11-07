"""Item 801: several mask workers share one GPU under the memory governor.

No GPU is used. Stand-in device records plan the workers, the item-493 CPU
segmenter runs inside real spawned workers, and a stand-in reader replaces
NVML for the utilisation report.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from spacr import _mask_workers as mw
from tests.test_gpu_memory_governor import card  # noqa: F401
from tests.test_mask_parallel_pipeline import (
    _calls, _plate, _segment, _settings, cpu_workers,  # noqa: F401
)

GB = 1 << 30
CARD = {'index': 0, 'name': 'Card A', 'memory_bytes': 24 * GB, 'backend': 'cuda'}


@pytest.fixture(autouse=True)
def _unrestricted_gpu_visibility(monkeypatch):
    """The stand-in cards are visible: no visibility list hides them."""
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(key, raising=False)


def test_one_worker_per_gpu_never_queries_devices(monkeypatch):
    monkeypatch.setattr(mw, '_compatible_mask_gpus', lambda: pytest.fail('queried GPUs'))
    for settings in ({}, {'mask_workers_per_gpu': 1}, {'mask_workers_per_gpu': None},
                     {'mask_workers_per_gpu': ''}):
        assert mw._parallel_mask_plan(settings) is None


@pytest.mark.parametrize('value', [0, -2, 'two', 1.5, True])
def test_an_invalid_worker_count_is_refused(value):
    with pytest.raises(ValueError, match='mask_workers_per_gpu'):
        mw._parallel_mask_plan({'mask_workers_per_gpu': value}, devices=[CARD])


def test_a_timelapse_run_is_refused_before_work():
    with pytest.raises(ValueError, match='timelapse'):
        mw._parallel_mask_plan({'mask_workers_per_gpu': 2, 'timelapse': True},
                               devices=[CARD])


def test_two_workers_share_the_one_gpu_of_a_workstation():
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 2}, devices=[CARD])
    assert plan['devices'] == ['0.1', '0.2']
    assert plan['gpus'] == {'0.1': 0, '0.2': 0}
    assert plan['names'] == {'0.1': 'Card A', '0.2': 'Card A'}
    assert plan['environments']['0.1'] == plan['environments']['0.2']
    assert plan['environments']['0.1']['CUDA_VISIBLE_DEVICES'] == '0'
    assert plan['uuids'] == {}


def test_without_mask_parallel_only_the_first_gpu_is_used():
    second = dict(CARD, index=1, name='Card B')
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 2}, devices=[CARD, second])
    assert set(plan['gpus'].values()) == {0}
    both = mw._parallel_mask_plan({'mask_workers_per_gpu': 2, 'mask_parallel': True},
                                  devices=[CARD, second])
    assert both['devices'] == ['0.1', '0.2', '1.1', '1.2']


def test_mask_parallel_on_one_gpu_still_shares_it():
    plan = mw._parallel_mask_plan({'mask_parallel': True, 'mask_workers_per_gpu': 3},
                                  devices=[dict(CARD, uuid='GPU-aaa')])
    assert plan['devices'] == ['0.1', '0.2', '0.3']
    assert plan['uuids'] == {0: 'GPU-aaa'}


def test_the_card_memory_caps_the_workers(capsys):
    small = dict(CARD, memory_bytes=8 * GB)
    assert mw._parallel_mask_plan({'mask_workers_per_gpu': 4}, devices=[small]) is None
    out = capsys.readouterr().out
    assert 'room for 1 of 4 workers' in out and 'ordinary single-device' in out
    middle = dict(CARD, memory_bytes=16 * GB)
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 4}, devices=[middle])
    assert plan['devices'] == ['0.1', '0.2']
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 4, 'gpu_memory_margin_gb': 7},
                                  devices=[middle])
    assert plan is None


def test_a_memory_fraction_caps_the_workers():
    big = dict(CARD, memory_bytes=80 * GB)
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 4, 'gpu_memory_fraction': 0.5},
                                  devices=[big])
    assert plan['devices'] == ['0.1', '0.2']
    plan = mw._parallel_mask_plan({'mask_workers_per_gpu': 4, 'gpu_memory_fraction': 0.25},
                                  devices=[big])
    assert len(plan['devices']) == 4


def test_the_worker_count_does_not_change_the_resume_signature():
    from spacr.artifacts import material_settings

    assert 'mask_workers_per_gpu' not in material_settings({'mask_workers_per_gpu': 2,
                                                            'cell_channel': 0})


def test_utilisation_is_summarised_per_gpu():
    readings = iter([{0: (95.0, 3 * GB)}, {0: (60.0, 5 * GB)}, {0: (85.0, 4 * GB)},
                     {0: (100.0, 4 * GB), 7: (1.0, 0)}])
    sampler = mw._GpuUtilisation({0: ''}, read=lambda: next(readings))
    sampler._started = None
    for _ in range(4):
        sampler._sample()
    sampler._sample()
    row = sampler.stop()[0]
    assert row['samples'] == 4
    assert row['mean_busy_percent'] == 85.0
    assert row['busy_share'] == 0.75
    assert row['peak_memory_gb'] == 5.0


def test_without_nvml_nothing_is_sampled(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mw, '_nvml_reader', lambda gpus: None)
    sampler = mw._GpuUtilisation({0: ''}).start()
    assert sampler.stop() == {}
    (tmp_path / 'plate1' / 'masks').mkdir(parents=True)
    mw._report_gpu_utilisation(str(tmp_path / 'plate1' / 'masks'), 'cell',
                               {'gpus': {'0.1': 0}}, {})
    assert 'GPU utilisation not reported' in capsys.readouterr().out
    assert not (tmp_path / 'plate1' / 'gpu_utilisation.json').exists()


def test_two_workers_on_one_gpu_match_the_serial_run_and_report_the_plate(
        tmp_path, cpu_workers, monkeypatch, capsys):
    serial = _plate(tmp_path / 'serial' / 'plate1')
    _segment(str(serial), {}, 'cell', batch_paths=sorted(serial.glob('*.npz')),
             on_batch_done=lambda path: None, run_qc=False)
    masks = _plate(tmp_path / 'shared' / 'plate1')
    settings = _settings(mask_workers_per_gpu=2)
    plan = mw._parallel_mask_plan(settings, devices=[dict(CARD, uuid='GPU-aaa')])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    plan['environments'] = {worker: dict(env) for worker in plan['devices']}
    asked = []
    monkeypatch.setattr(mw, '_SAMPLE_SECONDS', 0.01)
    monkeypatch.setattr(mw, '_nvml_reader', lambda gpus: asked.append(gpus) or (
        lambda: {0: (90.0, 6 * GB)}))

    mw._generate_masks_in_parallel(str(masks), settings, 'cell', plan, segmenter=_segment)

    serial_out = sorted((serial / 'cell_mask_stack').glob('*.npy'))
    shared_out = sorted((masks / 'cell_mask_stack').glob('*.npy'))
    assert [path.name for path in serial_out] == [path.name for path in shared_out]
    for left, right in zip(serial_out, shared_out):
        assert left.read_bytes() == right.read_bytes()
    calls = _calls(masks.parent)
    assert sorted(name for _, name in calls) == [f'batch{index}.npz' for index in range(4)]
    assert len({pid for pid, _ in calls}) == 2
    out = capsys.readouterr().out
    from spacr.qt.bridge import _mask_gpu_progress
    final = _mask_gpu_progress([line for line in out.splitlines()
                                if line.startswith('[mask GPUs]')][-1])
    assert sorted(device for device, *_ in final['workers']) == ['0.1', '0.2']
    assert final['done'] == final['total'] == 4
    assert asked == [{0: 'GPU-aaa'}]
    line = next(line for line in out.splitlines() if line.startswith('[gpu] plate plate1'))
    assert 'cell masks: GPU 0 with 2 workers busy 90% on average' in line
    assert 'at least 80% in 100%' in line and 'memory up to 6.0 GB' in line
    record = json.loads((masks.parent / 'gpu_utilisation.json').read_text(encoding='utf-8'))
    assert record['cell']['0']['workers'] == 2
    assert record['cell']['0']['mean_busy_percent'] == 90.0
    assert record['cell']['0']['samples'] >= 1


def test_a_shared_gpu_runs_cellpose_under_its_own_governor_site(tmp_path, monkeypatch, card):
    """Same masks either way; a shared GPU's calls are named mask_shared_gpu."""
    import spacr._gpu_governor as governor
    import spacr.object as O
    from tests.test_make_masks_cellpose_is_governed import _models
    from tests.test_segmentation_illumination_with_separate_preprocessing import (
        _prepared_inputs,
    )

    _models(monkeypatch, on_gpu=True)
    governed = []
    original = governor._governed_eval
    monkeypatch.setattr(governor, '_governed_eval', lambda *args, **kwargs: (
        governed.append(kwargs['site']), original(*args, **kwargs))[1])
    found = []
    for workers, site in ((1, 'mask_sam'), (2, 'mask_shared_gpu')):
        governed.clear()
        settings, src = _prepared_inputs(tmp_path / f'w{workers}', illumination=False)
        settings['mask_workers_per_gpu'] = workers
        O.generate_cellpose_masks_sam(str(Path(src) / 'masks'), settings, 'cell')
        found.append(np.load(Path(src) / 'masks' / 'cell_mask_stack' / 'plate1_A01_F001.npy'))
        assert governed == [site]
    np.testing.assert_array_equal(*found)
