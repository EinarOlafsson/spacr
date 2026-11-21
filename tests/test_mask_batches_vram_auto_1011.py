"""Item 1011 lane L1: VRAM-filling, even, seeded-random mask batches in the one batch plan.

CPU only. The calibration is tested on the real governor with its memory
readings replaced by a known linear model, so the fit must recover it.
"""
import math
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from spacr import _gpu_governor as G
from spacr import _mask_array_batches as array_batches
from spacr import _mask_batches as B

GIB = 2**30


# --- section 4.1: even sizes -------------------------------------------------

@pytest.mark.parametrize('n', [1, 2, 7, 50, 213, 1281, 2000])
@pytest.mark.parametrize('limit', [1, 3, 50, 256, 300])
def test_even_sizes_sum_spread_and_count(n, limit):
    """Sizes sum to n, differ by at most one, never exceed the limit, and are as few as possible."""
    sizes = B._even_sizes(n, limit)
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1
    assert max(sizes) <= limit
    assert len(sizes) == math.ceil(n / limit)


def test_even_sizes_example_from_the_spec():
    """1,281 fields at most 300 per batch give 5 batches of 256-257, not 4 x 300 + 81."""
    sizes = B._even_sizes(1281, 300)
    assert sizes == [257, 256, 256, 256, 256]
    assert B._batch_sizes(1281, 300) == [300, 300, 300, 300, 81]


def test_even_sizes_of_nothing_is_empty():
    """No fields give no batches."""
    assert B._even_sizes(0, 10) == []


def test_even_sizes_full_sweep():
    """The section 4.1 sweep: every n up to 2,000 against every limit up to 300."""
    for n in range(1, 2001, 37):
        for limit in range(1, 301, 13):
            sizes = B._even_sizes(n, limit)
            assert sum(sizes) == n and max(sizes) - min(sizes) <= 1
            assert max(sizes) <= limit and len(sizes) == math.ceil(n / limit)


# --- section 4.2: seeded order ----------------------------------------------

NAMES = [f'plate1_A01_{i:04d}.npy' for i in range(400)]


def test_field_order_is_seeded_and_ignores_global_random_state():
    """The same seed gives the same order whatever the global random state is."""
    random.seed(1)
    first = B._field_order(NAMES, {'mask_batch_seed': 7})
    random.seed(2)
    second = B._field_order(NAMES, {'mask_batch_seed': 7})
    assert first == second
    assert sorted(first) == sorted(NAMES) and first != NAMES


def test_field_order_differs_between_seeds_and_defaults_to_42():
    """Another seed gives another order; no seed means 42."""
    assert B._field_order(NAMES, {'mask_batch_seed': 1}) != B._field_order(NAMES, {'mask_batch_seed': 2})
    assert B._field_order(NAMES, {}) == B._field_order(NAMES, {'mask_batch_seed': 42})


def test_randomize_false_keeps_acquisition_order():
    """randomize=False returns the names as given, in a new list."""
    order = B._field_order(NAMES, {'randomize': False, 'mask_batch_seed': 3})
    assert order == NAMES and order is not NAMES


def test_field_order_agrees_across_processes():
    """Two interpreters with the same seed produce the same order_sha256."""
    code = ("import sys; sys.path.insert(0, %r); from spacr import _mask_batches as B; "
            "names=[f'f{i}' for i in range(300)]; "
            "print(B._order_sha256(B._field_order(names, {'mask_batch_seed': 11})))"
            % os.getcwd())
    runs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           check=True, env={**os.environ, 'PYTHONHASHSEED': str(seed)}
                           ).stdout.strip() for seed in (0, 1)}
    names = [f'f{i}' for i in range(300)]
    assert runs == {B._order_sha256(B._field_order(names, {'mask_batch_seed': 11}))}


@pytest.mark.parametrize('bad', [1.5, 'x', True])
def test_mask_batch_seed_must_be_whole(bad):
    """A seed that is not a whole number is refused."""
    with pytest.raises(ValueError, match='mask_batch_seed'):
        B._field_order(NAMES, {'mask_batch_seed': bad})


# --- section 4.5: request parsing -------------------------------------------

@pytest.mark.parametrize('value, expected', [
    ('auto_vram', 'auto_vram'), (' AUTO_VRAM ', 'auto_vram'),
    (0, None), (-1, None), (None, None), ('none', None), ('auto', None), ('whole', None),
    ('Whole', None), (64, 64), ('12', 12),
])
def test_batch_size_request(value, expected):
    """'auto_vram' is the new sentinel; 'whole' joins item 987b's automatic words."""
    assert B._batch_size_request(value) == expected


@pytest.mark.parametrize('bad', [-3, 'lots'])
def test_batch_size_request_refusals_name_auto_vram(bad):
    """The refusal lists the new choice."""
    with pytest.raises(ValueError, match='auto_vram'):
        B._batch_size_request(bad)


@pytest.mark.parametrize('value', ['auto_vram', 0])
def test_legacy_routes_get_fifty_for_either_automatic_mode(value):
    """The item 987c archive routes still get a whole number."""
    settings = {'batch_size': value}
    request = B._legacy_batch_size(settings)
    assert settings['batch_size'] == B._LEGACY_BATCH_SIZE
    assert settings['_batch_size_request'] == request


def test_resolved_tiles_reads_the_calibration_or_the_cpu_rule():
    """'auto_vram' tiles come from the calibration, else min(cap, 8); other values are unchanged."""
    auto = {'mask_eval_batch_size': 'auto_vram'}
    assert B._resolved_tiles({**auto, '_mask_calibration': {'tiles': 24}}, 5) == 24
    assert B._resolved_tiles(auto, 5) == 8
    assert B._resolved_tiles({**auto, 'mask_eval_batch_cap': 4}, 5) == 4
    assert B._resolved_tiles({'mask_eval_batch_size': 0}, 5) == 5
    assert B._resolved_tiles({'mask_eval_batch_size': 16}, 5) == 16


@pytest.mark.parametrize('bad', [0, -2, 2.5, 'many'])
def test_tile_cap_must_be_a_positive_whole_number(bad):
    """mask_eval_batch_cap below 1 or fractional is refused."""
    with pytest.raises(ValueError, match='mask_eval_batch_cap'):
        B._tile_cap({'mask_eval_batch_cap': bad})


# --- section 4.3: calibration on a known linear memory model ----------------

class _Card:
    """A fake card: reserved grows 0.3 GiB per tile + 0.5 GiB per pass; allocated 0.1 GiB per tile."""

    def __init__(self, free_gib=20.0, total_gib=24.0, held_gib=1.0):
        self.free = int(free_gib * GIB)
        self.total = int(total_gib * GIB)
        self.allocated = [int(0.8 * GIB), int(0.8 * GIB)]
        self.reserved = [int(held_gib * GIB), int(held_gib * GIB)]
        self.calls = []

    def evaluate(self, tiles):
        """One probe pass at ``tiles``: raise the peaks by the linear model, then return to rest."""
        self.calls.append(tiles)
        self.allocated[1] = max(self.allocated[1], self.allocated[0] + int(0.1 * GIB * tiles))
        self.reserved[1] = max(self.reserved[1],
                               self.reserved[0] + int(0.3 * GIB * tiles + 0.5 * GIB))


@pytest.fixture
def card(monkeypatch):
    """Point the governor's memory readings at a fake card on cuda:0."""
    fake = _Card()
    monkeypatch.setattr(G, '_model_device', lambda model: 'cuda:0')
    monkeypatch.setattr(G, '_device_index', lambda device=None: 0)
    monkeypatch.setattr(G, '_memory', lambda index: (fake.free, fake.total))
    monkeypatch.setattr(G, '_allocated', lambda index: tuple(fake.allocated))
    monkeypatch.setattr(G, '_reserved', lambda index: tuple(fake.reserved))
    monkeypatch.setattr(G, '_ledger_others', lambda index, path=None: {})
    monkeypatch.delenv(G._BUDGET_ENV, raising=False)
    G._reset()
    yield fake
    G._reset()


def _calibrate(card, settings, per_field=GIB):
    """Calibrate on the fake card with one field of ``per_field`` bytes."""
    return B._calibrate(object(), np.zeros(4, dtype=np.uint8), settings, site='mask_test',
                        evaluate=card.evaluate, per_field=per_field)


def test_calibration_recovers_the_linear_model(card):
    """per_tile is 0.3 GiB and fixed is what the process holds plus the 0.5 GiB pass cost."""
    record = _calibrate(card, {})
    assert card.calls == [2, 8] and record['probes'] == 2
    assert record['per_tile'] == pytest.approx(0.3 * GIB, rel=1e-6)
    assert record['fixed'] == pytest.approx(1.5 * GIB, rel=1e-6)
    assert record['budget_source'] == 'free'
    usable = 20 * GIB - 1.5 * GIB - G._CONTEXT_BYTES
    assert record['budget'] == pytest.approx(usable)
    assert record['tiles'] == 32
    assert record['max_fields'] == int((usable - 1.5 * GIB - 32 * 0.3 * GIB) // GIB)


def test_calibration_respects_the_cap_and_the_one_tile_floor(card):
    """The cap bounds the tile batch; a tiny budget still runs one tile."""
    assert _calibrate(card, {'mask_eval_batch_cap': 10})['tiles'] == 10
    card.free = int(2.5 * GIB)
    record = _calibrate(card, {})
    assert record['tiles'] == 1 and record['max_fields'] >= 1


def test_declared_budget_wins_over_more_free_memory(card, monkeypatch):
    """gpu_vram_budget_gb, then SPACR_GPU_VRAM_GB, sizes the batch even on a free 20 GiB card."""
    declared = _calibrate(card, {'gpu_vram_budget_gb': 6})
    assert declared['budget_source'] == 'declared'
    usable = 6 * GIB - 1.5 * GIB - G._CONTEXT_BYTES
    assert declared['budget'] == pytest.approx(usable)
    assert declared['tiles'] == int((usable - 1.5 * GIB - GIB) // (0.3 * GIB))
    monkeypatch.setenv(G._BUDGET_ENV, '6')
    assert _calibrate(card, {})['budget'] == pytest.approx(usable)
    assert _calibrate(card, {'gpu_vram_budget_gb': 8})['budget'] == pytest.approx(
        8 * GIB - 1.5 * GIB - G._CONTEXT_BYTES)


def test_other_processes_reservations_are_subtracted(card, monkeypatch):
    """What other spaCR processes announced in the ledger is not counted as free."""
    monkeypatch.setattr(G, '_ledger_others', lambda index, path=None: {
        '1:0': {'pid': 1, 'device': 0, 'used': G._CONTEXT_BYTES, 'want': G._CONTEXT_BYTES + 4 * GIB}})
    record = _calibrate(card, {})
    assert record['budget'] == pytest.approx(16 * GIB - 1.5 * GIB - G._CONTEXT_BYTES)


def test_workers_per_gpu_share_the_budget(card):
    """Two mask workers on one GPU each get half of it."""
    one = _calibrate(card, {})
    two = _calibrate(card, {'mask_workers_per_gpu': 2})
    assert two['budget'] == pytest.approx(one['budget'] / 2)


def test_a_second_probe_that_runs_out_of_memory_falls_back_to_one_probe(card, monkeypatch):
    """When t2 does not fit, per_tile = need(t1) / t1 and fixed = item 987b's model constant."""
    def evaluate(tiles):
        if tiles == 8:
            raise RuntimeError('CUDA out of memory. Tried to allocate 2 GiB')
        card.evaluate(tiles)
    monkeypatch.setattr(G, '_release', lambda: None)
    record = B._calibrate(object(), np.zeros(4, np.uint8), {}, site='mask_test',
                          evaluate=evaluate, per_field=GIB)
    assert record['probes'] == 1
    assert record['per_tile'] == pytest.approx((1.0 * GIB + 0.6 * GIB + 0.5 * GIB) / 2)
    assert record['fixed'] == pytest.approx(B._MODEL_GIB * GIB)


def test_noisy_probes_cannot_claim_a_free_tile(card):
    """A fit below a tenth of the 987b constants is floored."""
    def flat(tiles):
        card.reserved[1] = max(card.reserved[1], card.reserved[0] + int(0.5 * GIB))
    record = B._calibrate(object(), np.zeros(4, np.uint8), {}, site='mask_test',
                          evaluate=flat, per_field=GIB)
    assert record['per_tile'] >= 0.1 * B._TILE_GIB * GIB


def test_calibration_without_cuda_probes_nothing(monkeypatch):
    """CPU: no probe, tiles = min(cap, 8), no field limit from the GPU, source 'cpu'."""
    monkeypatch.setattr(G, '_model_device', lambda model: None)
    monkeypatch.setattr(G, '_device_index', lambda device=None: None)
    calls = []
    record = B._calibrate(object(), np.zeros(4, np.uint8), {'mask_eval_batch_cap': 32},
                          site='mask_test', evaluate=calls.append)
    assert calls == [] and record['probes'] == 0
    assert record['tiles'] == 8 and record['max_fields'] is None
    assert record['budget_source'] == 'cpu'


# --- governor: the declared budget ------------------------------------------

def test_governor_reads_the_declared_budget_from_settings_then_env(monkeypatch):
    """gpu_vram_budget_gb wins; an empty one falls back to SPACR_GPU_VRAM_GB; junk is ignored."""
    monkeypatch.setenv(G._BUDGET_ENV, '5.5')
    assert G._config({'gpu_vram_budget_gb': 3})['gpu_vram_budget_gb'] == 3.0
    assert G._config({'gpu_vram_budget_gb': None})['gpu_vram_budget_gb'] == 5.5
    monkeypatch.setenv(G._BUDGET_ENV, 'big')
    assert G._config({})['gpu_vram_budget_gb'] is None
    monkeypatch.delenv(G._BUDGET_ENV)
    assert G._config({'gpu_vram_budget_gb': -1})['gpu_vram_budget_gb'] is None


def test_declared_budget_lowers_the_fraction_ceiling(monkeypatch):
    """B GB on a T-byte card caps the process at min(fraction, B / T)."""
    monkeypatch.delenv(G._BUDGET_ENV, raising=False)
    monkeypatch.setattr(G, '_memory', lambda index: (20 * GIB, 24 * GIB))
    assert G._effective_fraction(0, G._config({'gpu_vram_budget_gb': 6})) == pytest.approx(0.25)
    assert G._effective_fraction(0, G._config({'gpu_vram_budget_gb': 6,
                                               'gpu_memory_fraction': 0.1})) == pytest.approx(0.1)
    assert G._effective_fraction(0, G._config({'gpu_memory_fraction': 0.5})) == 0.5
    assert G._effective_fraction(None, G._config({'gpu_vram_budget_gb': 6})) is None


def test_vram_budget_without_cuda_is_cpu():
    """No device index means no budget."""
    assert G._vram_budget(None, {}) == {'bytes': None, 'source': 'cpu', 'total': None}


def test_gpu_turn_exports_the_declared_budget_to_its_child():
    """tools/gpu_turn.sh and gpu_turn.ps1 hand --vram-gb to the job as SPACR_GPU_VRAM_GB."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'tools', 'gpu_turn.sh'), encoding='utf-8') as handle:
        assert 'SPACR_GPU_VRAM_GB="$VRAM_GB" "$@"' in handle.read()
    with open(os.path.join(root, 'tools', 'gpu_turn.ps1'), encoding='utf-8') as handle:
        assert '$env:SPACR_GPU_VRAM_GB =' in handle.read()


# --- the plan: even sizes, seed, record, resume -----------------------------

def _provisional_plan(tmp_path, n):
    """A provisional plan of ``n`` tiny fields in one batch, as the preprocessing step writes it."""
    stack = tmp_path / 'stack'
    masks = tmp_path / 'masks'
    stack.mkdir()
    masks.mkdir()
    names = [f'field_{i:03d}.npy' for i in range(n)]
    for name in names:
        np.save(stack / name, np.zeros((4, 4, 2), dtype=np.uint16))
    settings = {'randomize': True, 'mask_batch_seed': 42}
    B._write_plan(masks, stack, [names], [0, 1], settings, np.float32, request=None,
                  resolved=False)
    return masks, names


def test_auto_vram_plan_is_even_seeded_and_recorded(tmp_path):
    """13 fields at most 5 per batch give 5, 4, 4, with the seed, order hash and calibration."""
    masks, names = _provisional_plan(tmp_path, 13)
    calibration = {'tiles': 24, 'max_fields': 5, 'budget_source': 'declared', 'cap': 32,
                   'per_tile': 0.14 * GIB, 'budget': 6 * GIB, 'probes': 2}
    settings = {'batch_size': 'auto_vram', 'mask_batch_seed': 42,
                '_mask_calibration': dict(calibration)}
    lines = []
    plan = B._resolve_plan(masks, settings, free_ram=64 * GIB, log=lines.append)
    assert plan['sizes'] == [5, 4, 4] and plan['borrowed'] == []
    assert plan['batch_size_request'] == 'auto_vram' and plan['seed'] == 42
    assert plan['order_sha256'] == B._order_sha256(names)
    assert plan['calibration']['tiles'] == 24
    assert [len(batch['fields']) for batch in plan['batches']] == [5, 4, 4]
    assert lines == ['Mask batches: 13 fields → 3 batches of 4–5 (VRAM budget 6.0 GB declared; '
                     'tile batch 24 of cap 32; 0.14 GB per tile measured).']


def test_auto_vram_resume_keeps_the_plan_and_needs_no_calibration(tmp_path):
    """A resolved plan with the same seed and budget source is kept, and restores the calibration."""
    masks, _names = _provisional_plan(tmp_path, 9)
    calibration = {'tiles': 16, 'max_fields': 4, 'budget_source': 'free', 'cap': 32}
    settings = {'batch_size': 'auto_vram', '_mask_calibration': dict(calibration)}
    first = B._resolve_plan(masks, settings, free_ram=64 * GIB, log=lambda line: None)
    resumed = {'batch_size': 'auto_vram'}
    assert not B._needs_calibration(masks, resumed)
    lines = []
    again = B._resolve_plan(masks, {**resumed, '_mask_calibration': {'budget_source': 'free'}},
                            free_ram=64 * GIB, log=lines.append)
    assert lines == [] and again['sizes'] == first['sizes'] == [3, 3, 3]
    assert B._needs_calibration(masks, {'batch_size': 'auto_vram', 'mask_batch_seed': 5})
    assert not B._needs_calibration(masks, {'batch_size': 0})


def test_auto_vram_on_the_cpu_caps_batches_at_fifty(tmp_path):
    """No calibration (no CUDA): fields are RAM-limited, capped at 50, split evenly, source cpu."""
    masks, _names = _provisional_plan(tmp_path, 120)
    settings = {'batch_size': 'auto_vram'}
    lines = []
    plan = B._resolve_plan(masks, settings, free_ram=64 * GIB, log=lines.append)
    assert plan['sizes'] == [40, 40, 40]
    assert plan['calibration']['budget_source'] == 'cpu'
    assert settings['_mask_calibration']['tiles'] == 8
    assert 'no CUDA device' in lines[0]


def test_item_987b_automatic_is_unchanged(tmp_path):
    """batch_size 0 keeps full batches + a borrowed tiny remainder (987b), with no seed record."""
    masks, _names = _provisional_plan(tmp_path, 11)
    plan = B._resolve_plan(masks, {'batch_size': 5}, free_vram=None, free_ram=64 * GIB,
                           log=lambda line: None)
    assert plan['sizes'] == [5, 5, 1] and plan['borrowed'] == [2]
    assert 'seed' not in plan and 'calibration' not in plan


# --- section 4.8: OPS array batches -----------------------------------------

def test_ops_auto_vram_groups_follow_even_sizes(monkeypatch):
    """Under 'auto_vram' the resident groups are _even_sizes of the calibrated max_fields."""
    monkeypatch.setattr(B, '_store_on_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 64 * GIB)
    records = [f'site_{i}' for i in range(10)]
    settings = {'batch_size': 'auto_vram', 'mask_eval_batch_size': 'auto_vram',
                '_mask_calibration': {'max_fields': 4, 'tiles': 12, 'budget_source': 'free'}}
    seen = []
    with array_batches._array_batches(records, lambda record: (np.ones((3, 3), np.uint16), {}),
                                      settings, gpu=True) as fields:
        for record, image, _metadata, count in fields:
            seen.append((record, count))
    assert [record for record, _count in seen] == records
    assert [count for _record, count in seen] == [4] * 4 + [3] * 3 + [3] * 3


def test_ops_auto_vram_plan_records_the_seed_and_refuses_another(tmp_path, monkeypatch):
    """The persisted grouping carries the seed; a resume with another seed is refused."""
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 64 * GIB)
    path = tmp_path / 'mask_batch_plan.json'
    settings = {'batch_size': 'auto_vram', 'mask_batch_seed': 3}
    with array_batches._array_batches(['a', 'b'], lambda record: (np.ones((2, 2), np.uint8), {}),
                                      settings, plan_path=path) as fields:
        list(fields)
    assert '"mask_batch_seed": 3' in path.read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='different mask batch settings'):
        array_batches._batch_plan(path, {**settings, 'mask_batch_seed': 4})


def test_ops_fixed_request_plan_basis_is_unchanged(tmp_path):
    """A numeric request keeps the two-key basis older plans were written with."""
    _path, plan = array_batches._batch_plan(None, {'batch_size': 6, 'mask_eval_batch_size': 8})
    assert plan['settings'] == {'batch_size': 6, 'mask_eval_batch_size': 8}
