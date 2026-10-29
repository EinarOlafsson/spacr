"""Plaque controls reach original-pixel segmentation in preview and batch."""
import numpy as np
import pytest

from spacr import plaque, submodules, plaque_papers
from spacr.qt.widgets import plaque_preview
from tests.test_plaque_well_resize import _DiskModel
from tests.qt.test_plaque_assay_has_plaque_and_figure_modes import screen as screen


@pytest.mark.parametrize('diameter', [None, 0, 12, 37])
def test_resized_well_applies_diameter_and_explicit_thresholds_at_actual_resolution(diameter):
    model = _DiskModel()
    image = np.arange(17 * 31 * 3, dtype=np.uint16).reshape(17, 31, 3)
    plan = dict(side=40, max_factor=6, detector=None,
                thresholds=dict(flow_threshold=0.7, cellprob_threshold=-2, min_size=50))
    settings = dict(diameter=diameter, flow_threshold=0.2, CP_prob=1.5, minimum_object_size=0)
    labels = plaque.segment_plaque_image(model, image, settings, well_resize=plan, whole_frame=True)
    shape, options = model.calls[0]
    factor = np.sqrt(shape[-1] / 31 * shape[-2] / 17)
    assert options['diameter'] == (pytest.approx(diameter * factor) if diameter else None)
    assert options['flow_threshold'] == 0.2
    assert options['cellprob_threshold'] == 1.5
    assert options['min_size'] == 0
    assert labels.shape == (17, 31) and np.count_nonzero(labels)
    assert settings['diameter'] == diameter


def test_model_thresholds_still_supply_defaults_when_settings_are_absent():
    model = _DiskModel()
    plan = dict(side=40, max_factor=6, detector=None,
                thresholds=dict(flow_threshold=0.7, cellprob_threshold=-2, min_size=5))
    plaque.segment_plaque_image(model, np.ones((17, 31), np.uint16), {},
                                well_resize=plan, whole_frame=True)
    shape, options = model.calls[0]
    assert options['diameter'] is None
    assert options['flow_threshold'] == 0.7 and options['cellprob_threshold'] == -2
    assert options['min_size'] == pytest.approx(5 * shape[-1] / 31 * shape[-2] / 17)


def test_figure_batch_supplies_the_same_explicit_settings_to_its_segmenter(monkeypatch, tmp_path):
    received = []
    monkeypatch.setattr(plaque_papers, 'figure_folders', lambda source: [tmp_path])
    monkeypatch.setattr(submodules, '_figure_well_resize', lambda settings: None)
    def factory(path, **options):
        received.append((path, options))
        return lambda crop: (np.ones(crop.shape[:2], np.uint16), {})
    monkeypatch.setattr(plaque_papers, '_cellpose_segmenter', factory)
    def measure(*args, **kwargs):
        labels, _ = kwargs['segment'](np.zeros((17, 31, 3), np.uint8))
        assert labels.shape == (17, 31)
        return dict(figures=1, regions=1, plaques=1, database='results.db', with_ruler=1)
    monkeypatch.setattr(plaque_papers, 'measure_figure_folder', measure)
    settings = dict(src=str(tmp_path), diameter=19, flow_threshold=0.2, CP_prob=1.5,
                    minimum_object_size=3)
    submodules._analyze_plaque_figures(settings, 'chosen.pt')
    assert received[0][0] == 'chosen.pt'
    assert received[0][1]['settings'] == settings
    assert received[0][1]['min_size'] == 3


def test_changed_main_form_fields_update_preview_without_resetting_local_tuning(screen):
    screen._prime_preview()
    panel, model = screen._live_preview, screen._settings_model
    panel._diameter.setValue(23)
    panel._flow.setValue(0.19)
    model.set_value_for_key('flow_threshold', 0.61)
    model.set_value_for_key('minimum_object_size', 12)
    model.set_value_for_key('plaque_model', 'toxoplasma_plaque_v5')
    selected = panel.current_settings()
    assert selected['diameter'] == 23
    assert selected['flow_threshold'] == pytest.approx(0.61)
    assert selected['minimum_object_size'] == 12
    assert selected['plaque_model'] == 'toxoplasma_plaque_v5'
    panel._flow.setValue(0.22)
    assert panel.current_settings()['flow_threshold'] == pytest.approx(0.22)
    model.set_value_for_key('diameter', 77)
    assert panel.current_settings()['diameter'] == 77
    assert panel.current_settings()['flow_threshold'] == pytest.approx(0.22)
    written = {}
    panel.set_propagate_callback(written.update)
    panel.propagate()
    assert written['diameter'] == 77 and written['flow_threshold'] == pytest.approx(0.22)


def test_plaque_form_omits_legacy_settings_its_current_engine_does_not_consume(screen):
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    unused = {'normalize', 'channels', 'invert', 'resample', 'rescale', 'resize',
              'background', 'remove_background', 'Signal_to_noise', 'fill_in', 'batch_size'}
    assert unused <= _APP_HIDDEN_KEYS['analyze_plaques']
    assert not unused.intersection(screen._settings_model._widgets)
    assert screen._live_preview._diameter.value() == 0


def test_changed_main_source_uses_native_autoloader_once(screen, monkeypatch, tmp_path):
    screen._prime_preview()
    panel = screen._live_preview
    calls = []
    def load(source):
        calls.append(source)
        panel._src = str(source)
        return True
    monkeypatch.setattr(panel, 'load_source_async', load)
    source = str(tmp_path / 'new-plate')
    screen._settings_model.set_value_for_key('src', source)
    assert panel.current_settings()['src'] == source
    panel.current_settings()
    assert calls.count(source) == 1


def test_colony_mode_retains_functional_save_and_verbose_controls(screen):
    model = screen._settings_model
    assert {'save', 'verbose'} <= model._hidden_by_the_mode
    model.set_value_for_key('colony_counting', True)
    assert not {'save', 'verbose'}.intersection(model._hidden_by_the_mode)
    model.set_value_for_key('plaque_mode', 'figure')
    assert {'save', 'verbose'} <= model._hidden_by_the_mode
    model.set_value_for_key('plaque_mode', 'plaque')
    assert not {'save', 'verbose'}.intersection(model._hidden_by_the_mode)
    model.set_value_for_key('colony_counting', False)
    assert {'save', 'verbose'} <= model._hidden_by_the_mode


def test_immediate_preview_after_main_source_edit_never_segments_previous_image(screen, monkeypatch, tmp_path):
    screen._prime_preview()
    panel = screen._live_preview
    old_source = str(tmp_path / 'old')
    old_file = str(tmp_path / 'old' / 'previous.tif')
    panel._src = old_source
    panel._picker.blockSignals(True)
    panel._picker.addItem('previous.tif', old_file)
    panel._picker.setCurrentIndex(0)
    panel._picker.blockSignals(False)
    listings, segmentation = [], []
    monkeypatch.setattr(panel._load_jobs, 'submit', lambda *args: listings.append(args))
    monkeypatch.setattr(panel._jobs, 'submit', lambda *args: segmentation.append(args))
    screen._settings_model.set_value_for_key('src', str(tmp_path / 'new'))
    assert not panel.run_preview()
    assert listings and panel.current_path() is None
    assert not segmentation
