"""Exercise real localized graph callbacks and exported readmes without inference."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import tifffile

from spacr import __version__
from spacr.qt import i18n
from tools.web_mask_generator import app, detector, results
from tests.test_web_mask_gradio_app_n698 import demo, function


LOCALES = ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is']
ROOT = Path(__file__).parents[1]


@pytest.fixture(params=LOCALES)
def localized(request, monkeypatch):
    language = request.param
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    review = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                         '2026-10-15-web-mask-generator.json').read_text())
    return language, {row['source']: row['translation'] for row in review['records']}


def test_actual_eight_tab_graph_and_refusal_callbacks(localized, demo):
    language, translated = localized
    actual, calls = demo
    config = actual.get_config_file()
    tabs = [row['props']['label'] for row in config['components'] if row['type'] == 'tabitem']
    assert tabs == [translated.get(label, i18n.tr(label, language=language))
                    for label in [app.FAMILY_LABELS[row['family']] for row in app.model_families()]]
    callback = function(actual, 'segment_plaque')
    with pytest.raises(Exception) as error:
        callback(None, 'key', True, 0, 0, 0, [0], 0, 150, 32, False, 0.4, 0)
    assert translated['Choose an image or PDF first.'] in str(error.value)
    with pytest.raises(ValueError) as error:
        app._selection_index(0.5, 'T')
    assert str(error.value) == translated['{name} must be a finite whole number at least {minimum}.'].format(name='T', minimum=0)
    assert calls == [] and actual.is_running is False


def test_actual_mask_zip_readme_and_bytes(localized, tmp_path):
    _, translated = localized
    source = tmp_path / 'original.tif'
    image = np.arange(35, dtype=np.uint16).reshape(5, 7)
    tifffile.imwrite(source, image, metadata={'axes': 'YX'})
    before = source.read_bytes()
    labels = np.zeros((5, 7), dtype=np.uint32)
    labels[1:3, 2:4] = 5
    output = tmp_path / 'mask.zip'
    results.build_result_zip(output, input_file=source, image=image, image_axes='YX', masks=labels,
        model_id='declared_fixture', model_version='test', weights_sha256='ab' * 32,
        settings={}, unavailable={'flows': 'fixture supplies no inference', 'cell_probability': 'fixture supplies no inference'})
    key = next(key for key in translated if key.startswith('spaCR web mask result'))
    with zipfile.ZipFile(output) as archive:
        assert archive.read('README.txt').decode() == translated[key].format(id='declared_fixture', version='test', spacr=__version__) + '\n'
        assert archive.read('input/original.tif') == before
        manifest = json.loads(archive.read('manifest.json'))
        for name, metadata in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == metadata['sha256']
        assert manifest['outputs']['flows']['available'] is False
        assert manifest['outputs']['cell_probability']['calibrated'] is False
        assert 'flows.npy' not in archive.namelist()
    assert source.read_bytes() == before


def test_actual_detector_zip_readme_and_unavailable_semantics(localized, tmp_path):
    _, translated = localized
    from spacr.plaque import Well
    source = tmp_path / 'original.tif'
    image = np.arange(35, dtype=np.uint16).reshape(5, 7)
    tifffile.imwrite(source, image, metadata={'axes': 'YX'})
    before = source.read_bytes()
    output = tmp_path / 'detector.zip'
    detector.build_well_result_zip(output, input_file=source, image=image, image_axes='YX',
        wells=[Well(1, 1, 5, 4, 0.9)], model_id='declared_fixture', model_version='test',
        weights_sha256='ab' * 32, settings={})
    key = next(key for key in translated if key.startswith('spaCR well detection result'))
    with zipfile.ZipFile(output) as archive:
        assert archive.read('README.txt').decode() == translated[key] + '\n'
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['unavailable']['masks'] == translated['The well detector returns bounding boxes, not instance masks.']
        assert manifest['unavailable']['cell_probability'] == translated['Detection confidence is not a per-pixel cell probability.']
        assert 'mask.tif' not in archive.namelist() and archive.read('input/original.tif') == before
        for name, metadata in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == metadata['sha256']
    assert source.read_bytes() == before
