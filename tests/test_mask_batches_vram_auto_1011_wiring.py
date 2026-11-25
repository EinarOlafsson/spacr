"""Item 1011: exercise the real Mask and OPS adapters without CUDA or model downloads."""
import json

import numpy as np
import pytest

from spacr import _gpu_governor as G
from spacr import _mask_batches as B
from spacr import _mask_workers as workers
from spacr import core, io, object as objects, ops_engine as ops
from spacr.settings import set_default_settings_preprocess_generate_masks
from tests.test_mask_batches_vram_auto_1011 import card
from tests.test_mask_batch_size_auto_987b import SETTINGS, _plate
from tests.test_object_sam_masks import _install_sam_model
from tests.test_ops_engine_phenotype_coverage import _ThresholdModel


def _mask_project(tmp_path, count=7, **overrides):
    """Create a real provisional Mask plan from small uint16 source fields."""
    stack = _plate(tmp_path, count)
    for path in stack.glob('*.npy'):
        np.save(path, np.load(path).astype(np.uint16))
    settings = dict(
        SETTINGS, src=str(tmp_path), batch_size='auto_vram',
        mask_eval_batch_size='auto_vram', randomize=True,
        mask_batch_seed=17, mask_eval_batch_cap=4, verbose=False,
        save=True, magnification=20)
    settings.update(overrides)
    settings = set_default_settings_preprocess_generate_masks(settings)
    io.concatenate_and_normalize(str(stack), [0, 1], settings=settings)
    return str(tmp_path / 'masks'), settings


def _direct_eval(monkeypatch, card):
    """Keep the real calibration fitter but replace device evaluation with a measured fake."""
    calls = []

    def evaluate(model, image, *, site, settings=None, **kwargs):
        """Update fake allocator peaks and run the actual caller input/options through the model."""
        calls.append((model, image, site, dict(kwargs)))
        card.evaluate(kwargs['batch_size'])
        return model.eval(image, **kwargs)

    monkeypatch.setattr(objects, '_mask_eval', evaluate)
    return calls


def test_mask_calibrates_before_planning_with_resident_bytes_and_reuses_model(
        tmp_path, monkeypatch, card):
    """The first Mask sample uses real normalized channels; segmentation reuses its cached model."""
    masks, settings = _mask_project(tmp_path)
    holder = {'n_objects': 1}
    _install_sam_model(monkeypatch, holder)
    calls = _direct_eval(monkeypatch, card)
    monkeypatch.setattr(objects.accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    with objects._mask_model_session() as cache:
        plan = core._prepare_mask_batch_plan(masks, settings)
        calibrated_model = holder['model']
        assert plan['calibration']['probes'] == 2
        assert plan['calibration']['per_field'] == 8 * 8 * 2 * 4
        assert plan['calibration']['tiles'] == 4
        assert calls[0][3]['normalize'] is False
        assert calls[0][3]['channel_axis'] == -1
        assert calls[0][1][0].shape[-1] == 2
        assert plan.get('normalisation') == {}
        assert not settings['spacr_resolved_parameters']['fields']
        assert cache['loads'] == 1
        objects.generate_cellpose_masks_sam(masks, settings, 'cell', run_qc=False)
        assert holder['model'] is calibrated_model
        assert cache['loads'] == 1
    assert workers._MODEL_CACHE is None
    assert card.calls[:2] == [2, 4]
    for parameters in settings['spacr_resolved_parameters']['fields'].values():
        record = parameters['mask_batch']
        assert record['value']['tiles'] == 4
        assert record['configuration']['mask_batch_seed'] == 17
        assert record['configuration']['order_sha256'] == plan['order_sha256']
        assert record['value']['calibration'] == plan['calibration']


def test_mask_saved_gpu_plan_restores_calibration_without_new_probes(tmp_path, monkeypatch, card):
    """A saved GPU plan restores calibration before resolution, preserving bytes and tiles."""
    masks, settings = _mask_project(tmp_path)
    holder = {'n_objects': 1}
    _install_sam_model(monkeypatch, holder)
    _direct_eval(monkeypatch, card)
    with objects._mask_model_session():
        first = core._prepare_mask_batch_plan(masks, settings)
    settings.pop('_mask_calibration')
    before = open(B._plan_file(masks), encoding='utf-8').read()
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('resume probed'))
    second = core._prepare_mask_batch_plan(masks, settings)
    assert second['calibration'] == first['calibration'] == settings['_mask_calibration']
    assert open(B._plan_file(masks), encoding='utf-8').read() == before
    assert card.calls == [2, 4]


@pytest.mark.parametrize('original', [None, {'key': None, 'model': None, 'hits': 0, 'loads': 0}])
def test_model_session_restores_nested_and_exceptional_cache(monkeypatch, original):
    """Nested runs reuse the existing cache; exceptions always restore its original identity."""
    monkeypatch.setattr(workers, '_MODEL_CACHE', original)
    with pytest.raises(RuntimeError, match='stop'):
        with objects._mask_model_session() as active:
            with objects._mask_model_session() as nested:
                assert nested is active
            assert objects._MASK_MODEL_SESSION.get() is active
            raise RuntimeError('stop')
    assert workers._MODEL_CACHE is original


def test_coordinator_releases_calibrated_weights_before_subprocess_workers(monkeypatch):
    """The parent does not keep a second model resident while pooled workers load theirs."""
    model = object()
    cache = dict(key='identity', model=model, hits=0, loads=1)
    monkeypatch.setattr(workers, '_MODEL_CACHE', cache)
    cleared = []
    monkeypatch.setattr(B, '_empty_cache', lambda: cleared.append(True))
    with objects._mask_model_session():
        objects._release_calibration_model()
    assert cache['model'] is None and cache['key'] is None
    assert cleared == [True]


def test_ops_adapter_even_groups_journal_and_resume_keep_saved_calibration(tmp_path, monkeypatch, card):
    """OPS uses the shared resident planner, exact full-stack bytes and no resume probes."""
    calls = _direct_eval(monkeypatch, card)
    monkeypatch.setattr(B, '_store_on_gpu', lambda *a, **k: None)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_batch_seed=9, mask_eval_batch_cap=4, randomize=True)
    model = _ThresholdModel()
    stack = np.ones((8, 8, 5), np.float32)
    order = B._field_order(list(range(7)), settings)
    loaded = []

    def load(record):
        """Keep every original full stack resident while the smaller model input is probed."""
        loaded.append(record)
        return stack * (record + 1), None

    def prepare(image):
        """Match the real phenotype cell input contract with downsampling and channel stacking."""
        _dna, body, extra, _factor = ops._ph_segmentation_inputs(
            image, ['DAPI', 'AF750', 'GFP', 'CY3', 'CY5'], {'ops_ph_downsample': 2})
        return model, body, dict(extra['cell'], channel_axis=-1)

    path = tmp_path / 'mask_batch_plan.json'
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 2 * 2**30 + stack.nbytes * 6 * 3)
    with ops._ops_array_batches(order, load, settings, prepare=prepare,
                                site='ops_phenotype', gpu=True, plan_path=path) as resident:
        first = [(record, image.copy(), count) for record, image, _meta, count in resident]
    plan = json.loads(path.read_text())
    assert loaded == order
    assert plan['sizes'] == [3, 2, 2]
    assert plan['calibration']['per_field'] == stack.nbytes
    assert calls[0][1].shape == (4, 4, 2)
    assert plan['order_sha256'] == B._order_sha256(map(str, order))
    assert all(entry[2] == (3 if i < 3 else 2) for i, entry in enumerate(first))
    settings.pop('_mask_calibration')
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('resume probed'))
    with ops._ops_array_batches(order, load, settings, prepare=prepare,
                                site='ops_phenotype', gpu=True, plan_path=path) as resident:
        second = [(record, image.copy(), count) for record, image, _meta, count in resident]
    assert card.calls == [2, 4]
    assert settings['_mask_calibration'] == plan['calibration']
    assert [record for record, _image, _count in second] == order
    assert [count for _record, _image, count in second] == [entry[2] for entry in first]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left[1], right[1])
    assert len(settings['spacr_resolved_parameters']['fields']) == 7


@pytest.mark.parametrize('change', ['seed', 'fields'])
def test_mask_batch_journal_refuses_a_changed_draw(tmp_path, monkeypatch, change):
    """The existing resolved-parameter refusal protects the recorded seed and entire cohort."""
    masks, settings = _mask_project(tmp_path)
    model = _ThresholdModel()
    monkeypatch.setattr(objects, '_mask_model_setup', lambda *a, **k: (model, [0], 'cellpose'))
    plan = core._prepare_mask_batch_plan(masks, settings)
    B._load_batch(plan, 0, settings=settings)
    if change == 'seed':
        settings['mask_batch_seed'] += 1
    else:
        for value in settings['_mask_batch_membership'].values():
            value['order_sha256'] = 'changed-field-set'
    with pytest.raises(ValueError, match='source or configuration for mask_batch'):
        B._load_batch(plan, 0, settings=settings)


@pytest.mark.parametrize('batch_request, expected', [(0, 5), ('whole', 5), (2, 2)])
def test_legacy_ops_selectors_keep_original_groups_without_calibration(tmp_path, monkeypatch, batch_request, expected):
    """Legacy field and tile requests keep their existing shared grouping and tile policy."""
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('legacy probed'))
    settings = dict(batch_size=batch_request, mask_eval_batch_size=0)
    with ops._ops_array_batches(range(5), lambda index: (np.ones((2, 2)), None), settings,
                                prepare=lambda image: pytest.fail('legacy model setup'),
                                site='ops_stitched', plan_path=tmp_path / 'plan.json') as resident:
        found = [(index, count) for index, _image, _metadata, count in resident]
    assert found[0][1] == expected
    assert [index for index, _count in found] == list(range(5))
    assert objects._eval_tile_batch(settings, 5) == 5


@pytest.mark.parametrize('ram_fields', [3, 20])
def test_real_phenotype_stager_matches_seeded_segmentation_order(tmp_path, monkeypatch, ram_fields):
    """The actual phenotype caller stages and segments the same seeded field sequence."""
    import pandas as pd
    import spacr.ops_run_infra as infra

    sites = list(range(7))
    rows = pd.DataFrame({'site': sites})
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_batch_seed=17, randomize=True, mask_eval_batch_cap=4,
                    ops_ph_channels='DAPI,AF750', ops_ph_image_format='npz',
                    ops_ph_stage_dir=str(tmp_path / 'stage'), ops_ph_stage_ahead=3)
    stack = np.ones((8, 8, 2), np.float32)
    staged, loaded = [], []

    class Stager:
        """Observe the order passed to the existing staging interface."""
        ahead = 3
        peak_files = 1
        peak_bytes = stack.nbytes

        def __init__(self, folder, order, read, **options):
            """Record the complete requested staging order."""
            staged.extend(order)

        def load(self, site, mmap=False):
            """Expose the field identity through existing resident pixels."""
            loaded.append(site)
            return stack * (site + 1), []

        def release(self, site):
            """Release a staged field after segmentation."""

        def close(self):
            """Finish this synthetic staging session."""

    monkeypatch.setattr(infra, 'FieldStager', Stager)
    monkeypatch.setattr(ops, '_well_rows', lambda *args: rows)
    monkeypatch.setattr(ops, '_ph_mask_paths', lambda *args: {})
    monkeypatch.setattr(ops, '_ph_secondary_by_site', lambda *args: {})
    monkeypatch.setattr(ops, '_ph_record_missing', lambda *args: None)
    monkeypatch.setattr(ops, '_ph_transforms', lambda *args: None)
    model = _ThresholdModel()
    monkeypatch.setattr(ops, '_cellpose_model', lambda *args: model)
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 2 * 2**30 + stack.nbytes * 6 * ram_fields)
    files = {0: {site: {} for site in sites}}
    report = ops._ph_fields('unused.db', 'P1', 'A1', files, settings, False,
                            str(tmp_path), segment=True, measure=False)
    expected = B._field_order(sites, settings)
    assert staged == loaded == expected
    assert loaded != sites
    assert report['ph_segment']['segmented'] == 7
    plan = json.loads((tmp_path / 'A1' / 'phenotype_masks' / 'mask_batch_plan.json').read_text())
    assert plan['sizes'] == [3, 2, 2]
    assert settings['_mask_calibration']['budget_source'] == 'cpu'
    assert model.calls[0]['batch'] == 4


def test_cpu_mask_plan_never_evaluates_a_probe_and_caps_at_fifty(tmp_path, monkeypatch):
    """The real Mask adapter retains seeded even CPU cohorts with no model probe calls."""
    masks, settings = _mask_project(tmp_path, count=103)
    model = _ThresholdModel()
    monkeypatch.setattr(objects, '_mask_model_setup', lambda *a, **k: (model, [0], 'cellpose'))
    monkeypatch.setattr(objects, '_mask_eval', lambda *a, **k: pytest.fail('CPU probed'))
    plan = core._prepare_mask_batch_plan(masks, settings)
    assert plan['sizes'] == [35, 34, 34]
    assert plan['calibration']['probes'] == 0
    assert plan['calibration']['budget_source'] == 'cpu'
    assert plan['calibration']['per_field'] == 8 * 8 * 2 * 4
    assert objects._eval_tile_batch(settings, 35) == 4


def test_concurrent_mask_runs_reuse_models_only_within_their_own_context(monkeypatch):
    """Two real threads use the existing cache identity without sharing models across runs."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    monkeypatch.setattr(workers, '_MODEL_CACHE', None)
    barrier = Barrier(2)

    def run():
        """Build once inside one run while the other run has the same model identity."""
        with objects._mask_model_session() as cache:
            barrier.wait(timeout=5)
            first = objects._session_cached_mask_model(('cellpose', 'cpsam', {}), object)
            barrier.wait(timeout=5)
            second = objects._session_cached_mask_model(('cellpose', 'cpsam', {}), object)
            return first, second, cache['loads'], cache['hits']

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert all(first is second and loads == hits == 1 for first, second, loads, hits in results)
    assert results[0][0] is not results[1][0]
    assert workers._MODEL_CACHE is None
    assert objects._MASK_MODEL_SESSION.get() is None


@pytest.mark.parametrize('change', ['seed', 'fields'])
def test_completed_ops_run_refuses_a_changed_draw_before_loading_or_probing(tmp_path, monkeypatch, change):
    """A completed resume validates the full saved cohort even when no field needs segmentation."""
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_batch_seed=9, randomize=True)
    records = list(range(3))
    path = tmp_path / 'plan.json'
    model = _ThresholdModel()
    with ops._ops_array_batches(records, lambda record: (np.ones((4, 4)), None), settings,
                                prepare=lambda image: (model, image, {}), site='ops_phenotype',
                                plan_path=path) as resident:
        assert len(list(resident)) == 3
    original = path.read_bytes()
    settings.pop('_mask_calibration')
    if change == 'seed':
        settings['mask_batch_seed'] += 1
    else:
        records.pop()
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('completed resume probed'))
    with pytest.raises(ValueError, match='source or configuration for mask_batch'):
        with ops._ops_array_batches([], lambda record: pytest.fail('completed resume loaded'),
                                    settings, prepare=lambda image: pytest.fail('model loaded'),
                                    site='ops_phenotype', plan_path=path, order=records):
            pytest.fail('changed draw admitted')
    assert path.read_bytes() == original


def test_completed_mask_run_refuses_changed_seed_before_loading_or_probing(tmp_path, monkeypatch):
    """The Mask route validates the journal before deciding that completed masks need no work."""
    masks, settings = _mask_project(tmp_path)
    monkeypatch.setattr(objects, '_mask_model_setup',
                        lambda *a, **k: (_ThresholdModel(), [0], 'cellpose'))
    plan = core._prepare_mask_batch_plan(masks, settings)
    B._load_batch(plan, 0, settings=settings)
    original = open(B._plan_file(masks), 'rb').read()
    settings['mask_batch_seed'] += 1
    settings.pop('_mask_calibration')
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('completed resume probed'))
    with pytest.raises(ValueError, match='source or configuration for mask_batch'):
        core._prepare_mask_batch_plan(masks, settings)
    assert open(B._plan_file(masks), 'rb').read() == original


def test_stitched_windows_keep_acquisition_order_with_randomize_enabled(tmp_path, monkeypatch):
    """The actual stitched objects route retains the overlapping window sequence."""
    from tests.test_the_ops_engine_segments_with_the_validated_model import _lay_well

    canvas = np.zeros((16, 32), np.float32)
    canvas[5:8, 5:8] = canvas[5:8, 21:24] = 5000
    settings, db = _lay_well(tmp_path, monkeypatch, canvas, tile=16, step=16,
                              model=_ThresholdModel())
    settings.update(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    randomize=True, mask_batch_seed=17)
    monkeypatch.setattr(ops, '_WINDOW', 12)
    monkeypatch.setattr(ops, '_WINDOW_OVERLAP', 4)
    observed = []
    original = ops._ops_array_batches

    def batches(records, load, active, **options):
        """Observe records at the actual stitched resident-adapter boundary."""
        observed.extend(records)
        return original(records, load, active, **options)

    monkeypatch.setattr(ops, '_ops_array_batches', batches)
    index = ops._index_tiles(settings['genotype_source'])
    report = ops._objects(db, 'plate', 'A1', index['A1'], 1, settings, False)
    offsets = [window.offset() for window in observed]
    assert len(offsets) > 1 and offsets == sorted(offsets)
    assert report['windows'] == len(offsets)
    assert settings['_mask_calibration']['budget_source'] == 'cpu'


def test_real_spatial_windows_keep_acquisition_order_with_randomize_enabled(tmp_path, monkeypatch):
    """Spatial calls the shared sizing adapter without permuting its overlapping windows."""
    from tests.test_spatial_submodules import _mask_settings, xenium_slide

    source = xenium_slide(tmp_path / 'bundle')
    settings = _mask_settings(source, tmp_path / 'fields', spatial_segmenter='cellpose',
                              batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                              randomize=True, mask_batch_seed=17)
    model = _ThresholdModel()
    monkeypatch.setattr(ops, '_cellpose_model', lambda *a, **k: model)
    observed = []
    original = ops._ops_array_batches

    def batches(records, load, active, **options):
        """Observe the actual spatial field indices before shared residency."""
        observed.extend(records)
        return original(records, load, active, **options)

    monkeypatch.setattr(ops, '_ops_array_batches', batches)
    report = ops.run_spatial_masks(settings)
    assert len(observed) > 1 and observed == sorted(observed)
    assert report['objects'] > 0 and model.calls


def test_sequential_ops_sites_use_their_own_resident_calibration_and_resume(tmp_path, monkeypatch, card):
    """Stitched windows and phenotype stacks share the fitter while keeping distinct actual costs."""
    calls = _direct_eval(monkeypatch, card)
    monkeypatch.setattr(B, '_store_on_gpu', lambda *a, **k: None)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_batch_seed=9, mask_eval_batch_cap=4, randomize=False)
    model = _ThresholdModel()
    planes = np.ones((8, 8), np.float32)
    stack = np.ones((8, 8, 5), np.float32)
    sites = [('ops_stitched', planes), ('ops_phenotype', stack)]
    expected = {}
    prepared = []

    def prepare(image):
        """Keep the preparation bound to the actual input of this site."""
        prepared.append(image.shape)
        return model, image[..., 0] if image.ndim == 3 else image, {}

    for site, image in sites:
        path = tmp_path / (site + '.json')
        with ops._ops_array_batches([0, 1], lambda record: (image.copy(), None), settings,
                                    prepare=prepare, site=site, gpu=True, plan_path=path,
                                    field_id=lambda record, site=site: f'{site}_{record}') as resident:
            assert len(list(resident)) == 2
        plan = json.loads(path.read_text())
        assert plan['calibration']['per_field'] == image.nbytes
        assert settings['_mask_calibration'] == plan['calibration']
        assert settings['_mask_calibration_context'] == plan['calibration_context']
        expected[site] = plan['calibration']
    assert card.calls == [2, 4, 2, 4]
    assert prepared == [planes.shape, stack.shape]
    assert len(calls) == 4
    assert expected['ops_stitched']['per_field'] != expected['ops_phenotype']['per_field']
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('site resume probed'))
    for site, image in sites:
        with ops._ops_array_batches([0, 1], lambda record: (image.copy(), None), settings,
                                    prepare=lambda image: pytest.fail('site resume loaded model'),
                                    site=site, gpu=True, plan_path=tmp_path / (site + '.json'),
                                    field_id=lambda record, site=site: f'{site}_{record}') as resident:
            assert len(list(resident)) == 2
        assert settings['_mask_calibration'] == expected[site]
    assert card.calls == [2, 4, 2, 4]


def test_ops_journal_replay_without_external_plan_restores_matching_site(monkeypatch, card):
    """Saved journal calibration suffices for a zero-probe replay without a plan file."""
    _direct_eval(monkeypatch, card)
    monkeypatch.setattr(B, '_store_on_gpu', lambda *a, **k: None)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_eval_batch_cap=4, randomize=False)
    model = _ThresholdModel()
    image = np.ones((8, 8), np.float32)
    with ops._ops_array_batches([0], lambda record: (image.copy(), None), settings,
                                prepare=lambda image: (model, image, {}),
                                site='ops_stitched', gpu=True) as resident:
        assert len(list(resident)) == 1
    calibration = settings['_mask_calibration'].copy()
    settings.pop('_mask_calibration')
    settings.pop('_mask_calibrations')
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('journal replay probed'))
    with ops._ops_array_batches([0], lambda record: (image.copy(), None), settings,
                                prepare=lambda image: pytest.fail('journal replay loaded model'),
                                site='ops_stitched', gpu=True) as resident:
        assert len(list(resident)) == 1
    assert settings['_mask_calibration'] == calibration
    assert card.calls == [2, 4]


def test_ops_journal_records_the_actual_explicit_site_tile_override(tmp_path):
    """A path override remains effective while the resident groups use shared auto sizing."""
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_eval_batch_cap=4, ops_ph_tile_batch='2', randomize=False)
    image = np.ones((4, 4), np.float32)
    with ops._ops_array_batches([0], lambda record: (image.copy(), None), settings,
                                prepare=lambda image: (_ThresholdModel(), image, {}),
                                site='ops_phenotype', plan_path=tmp_path / 'plan.json') as resident:
        assert len(list(resident)) == 1
    assert settings['_mask_calibration']['tiles'] == 4
    assert settings['spacr_resolved_parameters']['fields']['0']['mask_batch']['value']['tiles'] == 2
    assert ops._ops_tile_batch(settings, 'ops_phenotype', 1) == 2


def test_real_mask_io_uses_repeatable_seeded_order_and_acquisition_control(tmp_path):
    """The actual Mask preprocessing writer delegates both random and control order to one helper."""
    results = []
    for index, options in enumerate([{}, {}, {'mask_batch_seed': 18}, {'randomize': False}]):
        masks, settings = _mask_project(tmp_path / str(index), count=13, **options)
        plan = B._read_plan(masks)
        names = [name for batch in plan['batches'] for name in batch['fields']]
        results.append((names, B._order_sha256(names)))
    assert results[0] == results[1]
    assert results[0] != results[2]
    assert results[3][0] == sorted(results[3][0])
    assert results[0][0] != results[3][0]


def test_calibration_probes_keep_exact_tiles_when_runtime_controller_would_shrink(monkeypatch, card):
    """The real governor must bypass fitting only for probes, retaining exact two-point tile counts."""
    model = _ThresholdModel()
    sample = np.ones((8, 8, 2), np.float32)
    fitted, fractions = [], []

    def fit(controller, requested, index, configuration):
        """Model a busy card that would shrink ordinary segmentation to one tile."""
        fitted.append(requested)
        return 1

    monkeypatch.setattr(G._BatchController, '_fit', fit)
    monkeypatch.setattr(G._BatchController, '_post', lambda *a, **k: {})
    monkeypatch.setattr(G, '_keep_headroom', lambda *a: None)
    monkeypatch.setattr(G, '_release', lambda: None)
    monkeypatch.setattr(G, '_record_path', lambda *a: None)
    monkeypatch.setattr(G, '_apply_fraction', lambda index, config: fractions.append(config.copy()))
    settings = {'gpu_adaptive_batch': True}
    for tiles in (2, 8):
        objects._mask_probe_eval(model, sample, settings, 'mask_exact_probe',
                                 {'normalize': False, 'channel_axis': -1}, tiles)
    assert [row['batch'] for row in model.calls] == [2, 8]
    assert fitted == []
    assert len(fractions) == 2
    assert all(config['gpu_adaptive_batch'] is False for config in fractions)
    assert settings['gpu_adaptive_batch'] is True


def test_second_probe_oom_reaches_the_existing_calibrator_without_smaller_retry(monkeypatch, card):
    """L1 owns the single-probe fallback when the exact eight-tile probe cannot fit."""
    from functools import partial
    from unittest.mock import Mock

    model = _ThresholdModel()
    original = model.eval
    sample = np.ones((8, 8, 2), np.float32)
    actual = []

    def evaluate(image, batch_size=8, **options):
        """Raise only on the exact eight-tile probe while measuring the two-tile call."""
        actual.append(batch_size)
        if batch_size == 8:
            raise RuntimeError('CUDA out of memory')
        card.evaluate(batch_size)
        return original(image, batch_size=batch_size, **options)

    monkeypatch.setattr(model, 'eval', Mock(side_effect=evaluate))
    monkeypatch.setattr(G._BatchController, '_fit', lambda *a: 1)
    monkeypatch.setattr(G._BatchController, '_post', lambda *a, **k: {})
    monkeypatch.setattr(G, '_keep_headroom', lambda *a: None)
    monkeypatch.setattr(G, '_release', lambda: None)
    monkeypatch.setattr(G, '_record_path', lambda *a: None)
    monkeypatch.setattr(G, '_apply_fraction', lambda *a: None)
    settings = {'mask_eval_batch_cap': 8, 'gpu_adaptive_batch': True}
    record = B._calibrate(model, sample, settings, site='mask_probe_oom',
                          evaluate=partial(objects._mask_probe_eval, model, sample, settings,
                                           'mask_probe_oom', {'normalize': False, 'channel_axis': -1}),
                          per_field=sample.nbytes)
    assert actual == [2, 8]
    assert record['probes'] == 1
    assert record['fixed'] == B._MODEL_GIB * 2**30


def test_journal_only_multi_group_membership_and_replay_ignore_changed_ram(monkeypatch):
    """A route without a permanent checkpoint still journals and replays exact shared groups."""
    image = np.ones((8, 8, 2), np.float32)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram',
                    mask_eval_batch_cap=4, randomize=False)
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 2 * 2**30 + image.nbytes * 6 * 3)
    model = _ThresholdModel()
    with ops._ops_array_batches(range(7), lambda record: (image.copy(), None), settings,
                                prepare=lambda image: (model, image[..., 0], {}),
                                site='ops_stitched') as resident:
        first = [(record, count) for record, _image, _metadata, count in resident]
    assert [count for _record, count in first] == [3, 3, 3, 2, 2, 2, 2]
    fields = settings['spacr_resolved_parameters']['fields']
    recorded = [fields[str(record)]['mask_batch']['value'] for record in range(7)]
    assert [(entry['batch'], entry['position'], entry['size']) for entry in recorded] == [
        (0, 0, 3), (0, 1, 3), (0, 2, 3), (1, 0, 2), (1, 1, 2), (2, 0, 2), (2, 1, 2)]
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 64 * 2**30)
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('journal replay probed'))
    with ops._ops_array_batches(range(7), lambda record: (image.copy(), None), settings,
                                prepare=lambda image: pytest.fail('journal replay loaded model'),
                                site='ops_stitched') as resident:
        second = [(record, count) for record, _image, _metadata, count in resident]
    assert second == first


@pytest.mark.parametrize('records', [[], [0, 1, 2, 3]])
def test_temporary_shared_plan_cleans_up_empty_and_missing_fields(tmp_path, monkeypatch, records):
    """Temporary checkpoint ownership covers empty inputs and None images without false journal rows."""
    from pathlib import Path
    import tempfile

    original = tempfile.TemporaryDirectory
    created = []

    def temporary(*args, **options):
        """Observe temporary plan ownership without changing the real cleanup behavior."""
        resource = original(*args, **options)
        if options.get('prefix') == 'spacr-mask-plan-':
            created.append(resource.name)
        return resource

    monkeypatch.setattr(tempfile, 'TemporaryDirectory', temporary)
    image = np.ones((8, 8, 2), np.float32)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram', randomize=False)
    with ops._ops_array_batches(records,
                                lambda record: (None if record % 2 == 0 else image, f'field-{record}'),
                                settings, prepare=lambda image: (_ThresholdModel(), image[..., 0], {}),
                                site='ops_stitched') as resident:
        found = [(record, data is None, meta) for record, data, meta, _count in resident]
    assert found == [(record, record % 2 == 0, f'field-{record}') for record in records]
    assert created and all(not Path(path).exists() for path in created)
    fields = settings['spacr_resolved_parameters']['fields']
    assert set(fields) == {str(record) for record in records if record % 2}
    if records:
        assert [fields[str(record)]['mask_batch']['value']['position'] for record in (1, 3)] == [0, 1]
        assert [fields[str(record)]['mask_batch']['value']['size'] for record in (1, 3)] == [2, 2]


def test_temporary_shared_plan_is_removed_on_a_segmentation_exception(monkeypatch):
    """Closing a failing resident stream also removes its temporary shared checkpoint."""
    from pathlib import Path
    import tempfile

    original = tempfile.TemporaryDirectory
    created = []

    def temporary(*args, **options):
        """Keep the real context manager while recording its owned directory."""
        resource = original(*args, **options)
        if options.get('prefix') == 'spacr-mask-plan-':
            created.append(resource.name)
        return resource

    monkeypatch.setattr(tempfile, 'TemporaryDirectory', temporary)
    image = np.ones((4, 4), np.float32)
    settings = dict(batch_size='auto_vram', mask_eval_batch_size='auto_vram', randomize=False)
    with pytest.raises(RuntimeError, match='segmentation failed'):
        with ops._ops_array_batches([0], lambda record: (image, None), settings,
                                    prepare=lambda image: (_ThresholdModel(), image, {}),
                                    site='ops_stitched') as resident:
            next(resident)
            raise RuntimeError('segmentation failed')
    assert created and all(not Path(path).exists() for path in created)


@pytest.mark.parametrize('morphology, method', [('spots', 'otsu'), ('spots', 'log'), ('network', 'ridge')])
def test_classical_only_mask_caller_never_constructs_or_probes_cellpose(tmp_path, monkeypatch, morphology, method):
    """Global auto batching must preserve real classical-only Mask runs without any model load."""
    import spacr.utils as utils

    masks, settings = _mask_project(tmp_path, count=3, cell_channel=None,
                                   nucleus_channel=None, pathogen_channel=None,
                                   organelle_channel=0, organelle_method=method,
                                   organelle_morphology=morphology, n_jobs=1)
    monkeypatch.setattr(objects, '_mask_model_setup', lambda *a, **k: pytest.fail('unused main model'))
    monkeypatch.setattr(utils, '_choose_model', lambda *a, **k: pytest.fail('unused Cellpose model'))
    monkeypatch.setattr(objects, '_load_unet_model', lambda *a, **k: pytest.fail('unused UNet model'))
    monkeypatch.setattr(objects, '_mask_probe_eval', lambda *a, **k: pytest.fail('classical probe'))
    plan = core._prepare_mask_batch_plan(masks, settings)
    objects.generate_organelle_masks_sam(masks, settings, 'organelle')
    from pathlib import Path

    assert len(list((Path(masks) / 'organelle_mask_stack').glob('*.npy'))) == 3
    assert plan['calibration']['probes'] == 0
    assert plan['calibration']['budget_source'] == 'cpu'
    assert plan['calibration']['per_field'] == 8 * 8 * 2 * 4


def test_organelle_cellpose_uses_real_preparation_options_and_the_same_loaded_model(tmp_path, monkeypatch, card):
    """Calibration and the actual organelle caller reuse one constructor and identical eval policy."""
    import spacr.utils as utils
    from tests.test_object_sam_masks import _RecordingCP

    masks, settings = _mask_project(tmp_path, count=3, cell_channel=None,
                                   nucleus_channel=None, pathogen_channel=None,
                                   organelle_channel=0, organelle_method='cellpose',
                                   organelle_rolling_ball=True, organelle_rolling_ball_radius=2,
                                   organelle_flow_threshold=0.73, organelle_cellprob_threshold=-0.4,
                                   organelle_resample=False)
    model = _RecordingCP()
    built, prepared = [], []
    original = objects._preprocess_batch

    def choose(name, device, **options):
        """Replace only the existing model constructor without changing its caller configuration."""
        built.append((name, device, options))
        return model

    def preprocess(images, active):
        """Observe real organelle preprocessing on both the probe and segmentation batch."""
        result = original(images, active)
        prepared.append(result.copy())
        return result

    monkeypatch.setattr(utils, '_choose_model', choose)
    monkeypatch.setattr(objects, '_preprocess_batch', preprocess)
    calls = _direct_eval(monkeypatch, card)
    with objects._mask_model_session() as cache:
        plan = core._prepare_mask_batch_plan(masks, settings)
        objects.generate_organelle_masks_sam(masks, settings, 'organelle')
        assert cache['loads'] == 1
    assert len(built) == 1 and built[0][2]['object_type'] == 'organelle'
    assert len(prepared) == 2 and len(calls) == 3
    assert calls[0][0] is calls[1][0] is calls[2][0] is model
    assert [call[3]['batch_size'] for call in calls] == [2, 4, 4]
    for call in calls:
        assert call[2] == 'mask_organelle'
        assert call[3]['normalize'] is False and call[3]['channel_axis'] == -1
        assert call[3]['diameter'] is None
        assert call[3]['flow_threshold'] == 0.73
        assert call[3]['cellprob_threshold'] == -0.4
        assert call[3]['resample'] is False
    np.testing.assert_array_equal(calls[0][1][0][..., 0], prepared[0][0])
    np.testing.assert_array_equal(calls[2][1][0][..., 0], prepared[1][0])
    assert plan['calibration']['per_field'] == 8 * 8 * 2 * 4


def test_unet_only_mask_plan_skips_tile_probes_and_loads_only_at_actual_segmentation(tmp_path, monkeypatch):
    """A backend without Cellpose tiles retains the shared no-probe rule and its own loader."""
    import spacr.utils as utils
    from pathlib import Path

    masks, settings = _mask_project(tmp_path, count=3, cell_channel=None,
                                   nucleus_channel=None, pathogen_channel=None,
                                   organelle_channel=0, organelle_method='unet',
                                   organelle_morphology='network')
    loaded, forwarded = [], []
    model = object()

    def load(active):
        """Represent the real native backend loader without a checkpoint or GPU."""
        loaded.append(active['organelle_method'])
        return model

    def segment(images, actual_model, active):
        """Retain the native backend call boundary without exercising its unrelated network."""
        forwarded.append(actual_model)
        return np.zeros(images.shape, np.int32)

    monkeypatch.setattr(objects, '_mask_model_setup', lambda *a, **k: pytest.fail('unused main model'))
    monkeypatch.setattr(utils, '_choose_model', lambda *a, **k: pytest.fail('unused Cellpose model'))
    monkeypatch.setattr(objects, '_mask_probe_eval', lambda *a, **k: pytest.fail('UNet tile probe'))
    monkeypatch.setattr(objects, '_load_unet_model', load)
    monkeypatch.setattr(objects, '_segment_unet', segment)
    plan = core._prepare_mask_batch_plan(masks, settings)
    assert loaded == []
    objects.generate_organelle_masks_sam(masks, settings, 'organelle')
    assert loaded == ['unet'] and forwarded == [model]
    assert len(list((Path(masks) / 'organelle_mask_stack').glob('*.npy'))) == 3
    assert plan['calibration']['probes'] == 0
    assert plan['calibration']['budget_source'] == 'cpu'


@pytest.mark.parametrize('batch_request', [0, 'whole', 2, 'auto_vram'])
def test_real_phenotype_preserves_legacy_limit_before_model_residency(tmp_path, monkeypatch, batch_request):
    """Legacy grouping sees the original free-VRAM state; calibration-active grouping sees loaded weights."""
    import pandas as pd
    from spacr import _mask_array_batches as arrays

    automatic = batch_request == 'auto_vram'
    settings = dict(batch_size=batch_request,
                    mask_eval_batch_size='auto_vram' if automatic else 0,
                    ops_ph_channels='DAPI,AF750', ops_ph_image_format='npz', randomize=False)
    sites = list(range(7))
    rows = pd.DataFrame({'site': sites})
    state = {'loaded': False}
    events = []
    model = _ThresholdModel()
    original = arrays._batch_limit

    def load_model(*args):
        """Make the real caller model factory observable without a real GPU."""
        events.append(('model',))
        state['loaded'] = True
        return model

    def limit(active, image, remaining, gpu):
        """Run the actual shared estimate with free memory dependent on weight residency."""
        events.append(('limit', state['loaded'], B._free_vram_bytes()))
        return original(active, image, remaining, gpu)

    monkeypatch.setattr(ops, '_well_rows', lambda *args: rows)
    monkeypatch.setattr(ops, '_ph_secondary_by_site', lambda *args: {})
    monkeypatch.setattr(ops, '_ph_read_field', lambda *args: np.ones((8, 8, 2), np.float32))
    monkeypatch.setattr(ops, '_ph_record_missing', lambda *args: None)
    monkeypatch.setattr(ops, '_ph_transforms', lambda *args: None)
    monkeypatch.setattr(ops, '_cellpose_model', load_model)
    monkeypatch.setattr(B, '_store_on_gpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(B, '_free_vram_bytes', lambda *args: (3 if state['loaded'] else 4) * 2**30)
    monkeypatch.setattr(B, '_model_resident', lambda: state['loaded'])
    monkeypatch.setattr(B, '_available_ram_bytes', lambda: 64 * 2**30)
    monkeypatch.setattr(arrays, '_batch_limit', limit)
    report = ops._ph_fields('unused.db', 'P1', 'A1', {0: {site: {} for site in sites}},
                            settings, True, str(tmp_path), segment=True, measure=False)
    first_limit = next(event for event in events if event[0] == 'limit')
    assert first_limit[1] is automatic
    assert first_limit[2] == (3 if automatic else 4) * 2**30
    expected = 8 if automatic else 2 if batch_request == 2 else 5
    assert model.calls[0]['batch'] == expected
    assert report['ph_segment']['segmented'] == 7
    events.clear()
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('completed resume probed'))
    again = ops._ph_fields('unused.db', 'P1', 'A1', {0: {site: {} for site in sites}},
                           settings, True, str(tmp_path), segment=True, measure=False)
    assert again['ph_segment']['segmented'] == 0
    assert events == []


@pytest.mark.parametrize('batch_request', [0, None, 'whole', 2])
def test_legacy_mask_session_keeps_original_uncached_model_construction(tmp_path, monkeypatch, batch_request):
    """Legacy plates must not retain weights that alter the next plate's free-VRAM batch estimate."""
    masks, settings = _mask_project(tmp_path, count=1, batch_size=batch_request,
                                   mask_eval_batch_size=0)
    holder = {'n_objects': 1}
    _install_sam_model(monkeypatch, holder)
    monkeypatch.setattr(workers, '_MODEL_CACHE', None)
    monkeypatch.setattr(B, '_empty_cache', lambda: pytest.fail('legacy cache release'))
    with objects._mask_model_session(settings) as cache:
        assert cache is None and objects._MASK_MODEL_SESSION.get() is None
        first = objects._mask_model_setup(settings, 'cell')[0]
        second = objects._mask_model_setup(settings, 'cell')[0]
        objects._release_calibration_model()
        assert first is not second
    assert workers._MODEL_CACHE is None


def test_mask_tile_auto_session_reuses_model_when_field_count_is_explicit(tmp_path, monkeypatch):
    """Tile calibration alone still reuses its actual model even with an explicit field batch."""
    _masks, settings = _mask_project(tmp_path, count=1, batch_size=2)
    holder = {'n_objects': 1}
    _install_sam_model(monkeypatch, holder)
    monkeypatch.setattr(workers, '_MODEL_CACHE', None)
    with objects._mask_model_session(settings) as cache:
        first = objects._mask_model_setup(settings, 'cell')[0]
        second = objects._mask_model_setup(settings, 'cell')[0]
        assert first is second and cache['loads'] == 1
    assert workers._MODEL_CACHE is None


@pytest.mark.parametrize('override', [None, ('batch_size', 50), ('mask_eval_batch_size', 2)])
def test_published_mask_settings_replay_preserves_selectors_and_resolved_groups(
        tmp_path, monkeypatch, card, override):
    """Published settings keep both selectors; replay retains exact groups and refuses overrides."""
    from spacr import run_journal as journal
    from spacr.validate import coerce_expected_types

    masks, settings = _mask_project(tmp_path)
    holder = {'n_objects': 1}
    _install_sam_model(monkeypatch, holder)
    _direct_eval(monkeypatch, card)
    monkeypatch.setattr(B, '_largest_fitting_size', lambda *a, **k: 3)
    with objects._mask_model_session(settings):
        plan = core._prepare_mask_batch_plan(masks, settings)
        objects.generate_cellpose_masks_sam(masks, settings, 'cell', run_qc=False)
    assert plan['sizes'] == [3, 2, 2]
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    run = journal.Run(app_key='mask', settings={}, dir=run_dir)
    monkeypatch.setattr(journal, 'current_run', lambda: run)
    core._save_resolved_mask_settings(settings, str(tmp_path))
    loaded = journal.load_run_settings(run_dir)
    assert settings['batch_size'] == 50
    assert loaded['batch_size'] == loaded['mask_eval_batch_size'] == 'auto_vram'
    assert not any(key.startswith('_') for key in loaded)
    replay = set_default_settings_preprocess_generate_masks(coerce_expected_types(loaded, 'mask'))
    if override is not None:
        replay[override[0]] = override[1]
    before = open(B._plan_file(masks), encoding='utf-8').read()
    probes = list(card.calls)
    monkeypatch.setattr(B, '_calibrate', lambda *a, **k: pytest.fail('public replay probed'))
    monkeypatch.setattr(B, '_largest_fitting_size', lambda *a, **k: pytest.fail('public replay regrouped'))
    monkeypatch.setattr(objects, '_mask_calibration_setup', lambda *a, **k: pytest.fail('public replay loaded model'))
    if override is not None:
        with pytest.raises(ValueError, match='configuration'):
            core._prepare_mask_batch_plan(masks, replay)
    else:
        B._legacy_batch_size(replay)
        io.concatenate_and_normalize(str(tmp_path / 'stack'), [0, 1], settings=replay)
        restored = core._prepare_mask_batch_plan(masks, replay)
        assert restored['calibration'] == plan['calibration'] == replay['_mask_calibration']
        assert restored['sizes'] == [3, 2, 2]
        assert open(B._plan_file(masks), encoding='utf-8').read() == before
        for batch in restored['batches']:
            B._load_batch(restored, batch['index'], replay)
    assert card.calls == probes
    after = json.loads(open(B._plan_file(masks), encoding='utf-8').read())
    original = json.loads(before)
    for key in ('batches', 'sizes', 'calibration', 'seed', 'order_sha256', 'normalisation'):
        assert after[key] == original[key]


def test_second_session_does_not_capture_an_in_progress_constructor_cache(monkeypatch):
    """An overlapping session must read only the restored pooled-cache identity."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr(workers, '_MODEL_CACHE', None)
    building, second_started, second_entered, release = (Event() for _ in range(4))
    caches = []

    def build():
        """Keep the first constructor's global swap visible until the second entry is attempted."""
        building.set()
        assert release.wait(5)
        return object()

    def first():
        """Start the first model construction inside an ordinary scoped session."""
        with objects._mask_model_session() as cache:
            caches.append(cache)
            return objects._session_cached_mask_model(('sam', 'fake', {}), build)

    def second():
        """Enter the second session while the first constructor holds the existing identity lock."""
        assert building.wait(5)
        second_started.set()
        with objects._mask_model_session() as cache:
            caches.append(cache)
            second_entered.set()
            return objects._session_cached_mask_model(('sam', 'fake', {}), object)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(first), pool.submit(second)
        assert second_started.wait(5)
        second_entered.wait(.2)
        release.set()
        model_a, model_b = a.result(5), b.result(5)
    assert caches[0] is not caches[1]
    assert model_a is not model_b
    assert workers._MODEL_CACHE is None


def test_legacy_coordinator_release_does_not_clear_another_run_cache(monkeypatch):
    """A disabled session follows the real parallel-role release call while another build completes."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    monkeypatch.setattr(workers, '_MODEL_CACHE', None)
    ready, built, finish, released = (Event() for _ in range(4))
    cached_model = workers._cached_model
    caches = []
    emptied = []
    monkeypatch.setattr(B, '_empty_cache', lambda: emptied.append(True))

    def held_handoff(*args):
        """Pause after real construction while the existing adapter still installs its global cache."""
        model = cached_model(*args)
        built.set()
        assert finish.wait(5)
        return model

    def ordinary_auto():
        """Construct the calibrated model after the legacy context is already open."""
        assert ready.wait(5)
        with objects._mask_model_session() as cache:
            caches.append(cache)
            model = objects._session_cached_mask_model(('sam', 'fake', {}), object)
            return cache, model

    def legacy_parallel():
        """Follow core's disabled-session context and coordinator release ordering."""
        with objects._mask_model_session({'batch_size': 0, 'mask_eval_batch_size': 0}):
            ready.set()
            assert built.wait(5)
            objects._release_calibration_model()
            released.set()

    monkeypatch.setattr(workers, '_cached_model', held_handoff)
    with ThreadPoolExecutor(max_workers=2) as pool:
        legacy, auto = pool.submit(legacy_parallel), pool.submit(ordinary_auto)
        assert built.wait(5)
        released.wait(.2)
        finish.set()
        legacy.result(5)
        cache, model = auto.result(5)
    assert cache['model'] is model
    assert emptied == []
    assert workers._MODEL_CACHE is None
