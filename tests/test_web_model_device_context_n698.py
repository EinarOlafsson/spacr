"""Dynamic device cache keys with controlled factories, never GPU acceptance."""
from types import SimpleNamespace

import pytest

from spacr import accelerator, object as objects, settings, submodules
from tools.web_mask_generator import masks, plaque


@pytest.fixture
def context(monkeypatch):
    state = {'kind': 'cpu', 'device': 'cpu', 'bfloat16': False, 'usable': True}
    refreshes = []
    created = []
    def resolve(refresh=False):
        refreshes.append(refresh)
        return accelerator.Accelerator(label='controlled context', **state)
    def create(*args, **kwargs):
        model = SimpleNamespace(**state)
        created.append(model)
        return model
    monkeypatch.setattr(accelerator, 'resolve', resolve)
    monkeypatch.setattr(submodules, '_plaque_cellpose_model', create)
    monkeypatch.setattr(settings, 'set_default_settings_preprocess_generate_masks', lambda s: s)
    monkeypatch.setattr(objects, '_robustness_segmenter',
                        lambda *a: SimpleNamespace(args=[create()]))
    plaque._load_model.cache_clear()
    masks._load_mask_model.cache_clear()
    yield state, refreshes, created
    plaque._load_model.cache_clear()
    masks._load_mask_model.cache_clear()


@pytest.mark.parametrize('family', ['plaque', 'mask'])
def test_cpu_warmup_is_not_returned_in_later_gpu_context(context, family):
    state, refreshes, created = context
    load = (lambda: plaque._load_model('/operator/checkpoint', 'a' * 64)) if family == 'plaque' else (
        lambda: masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell'))
    cpu = load()
    state.update(kind='cuda', device='cuda:0', bfloat16=True)
    gpu = load()
    assert cpu is not gpu
    assert cpu.device == 'cpu' and gpu.device == 'cuda:0'
    assert len(created) == 2
    assert refreshes == [True, True]
    assert load() is gpu
    state.update(kind='cpu', device='cpu', bfloat16=False)
    assert load() is cpu


@pytest.mark.parametrize('family', ['plaque', 'mask'])
def test_same_device_checkpoint_and_digest_reuses_verified_identity(context, family):
    state, refreshes, created = context
    load = plaque._load_model if family == 'plaque' else masks._load_mask_model
    extra = () if family == 'plaque' else ('cell',)
    first = load('/operator/checkpoint', 'a' * 64, *extra)
    assert load('/operator/checkpoint', 'a' * 64, *extra) is first
    assert load('/operator/checkpoint', 'b' * 64, *extra) is not first
    assert len(created) == 2


def test_mask_roles_retain_distinct_cache_entries(context):
    cell = masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell')
    nucleus = masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'nucleus')
    assert cell is not nucleus
    assert masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell') is cell


@pytest.mark.parametrize('family', ['plaque', 'mask'])
def test_precision_context_change_does_not_reuse_previous_weights(context, family):
    state, refreshes, created = context
    state.update(kind='cuda', device='cuda:0', bfloat16=False)
    load = (lambda: plaque._load_model('/operator/checkpoint', 'a' * 64)) if family == 'plaque' else (
        lambda: masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell'))
    first = load()
    state['bfloat16'] = True
    second = load()
    assert first is not second
    assert len(created) == 2


def test_plaque_legacy_optional_digest_call_remains_cached(context):
    first = plaque._load_model('/operator/checkpoint')
    assert plaque._load_model('/operator/checkpoint') is first


@pytest.mark.parametrize('family', ['plaque', 'mask'])
def test_unusable_context_changes_actual_gpu_constructor_flag(context, family):
    state, refreshes, created = context
    state.update(kind='cuda', device='cuda:0', bfloat16=True, usable=True)
    load = (lambda: plaque._load_model('/operator/checkpoint', 'a' * 64)) if family == 'plaque' else (
        lambda: masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell'))
    first = load()
    state['usable'] = False
    second = load()
    assert first is not second
    assert first.usable and not second.usable
    assert len(created) == 2


@pytest.mark.parametrize('family', ['plaque', 'mask'])
def test_cpu_precision_probe_does_not_change_effective_constructor_precision(context, family):
    state, refreshes, created = context
    load = (lambda: plaque._load_model('/operator/checkpoint', 'a' * 64)) if family == 'plaque' else (
        lambda: masks._load_mask_model('/operator/checkpoint', 'a' * 64, 'cell'))
    first = load()
    state['bfloat16'] = True
    assert load() is first
    assert len(created) == 1
