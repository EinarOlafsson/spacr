"""Canonical generated artifacts keep their project and native input contracts."""
import hashlib
import sqlite3
import zipfile

import numpy as np
import pytest
from PIL import Image

from spacr.qt.widgets import home
from spacr.qt.widgets.workflow_diagram import workflow_map


def _database(project, annotation_only=False):
    path = project / 'measurements' / 'measurements.db'
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        if annotation_only:
            connection.execute('CREATE TABLE image_annotations (image_id INTEGER)')
        else:
            connection.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, cell_area REAL)')
            connection.execute('CREATE TABLE png_list (png_path TEXT)')
    return path


@pytest.mark.parametrize('shape', [(6, 8, 4), (2, 6, 8, 4)])
@pytest.mark.parametrize('selection', ['project', 'folder', 'file'])
def test_merged_arrays_offer_measure_and_shared_downstream_without_pixels(
        tmp_path, monkeypatch, shape, selection):
    from spacr.qt.app import APPS, folded_children

    project = tmp_path / 'plate1'
    merged = project / 'merged'
    merged.mkdir(parents=True)
    image = merged / 'AssayPlate_A03_8_1.npy'
    np.save(image, np.zeros(shape, dtype=np.uint16))
    before = hashlib.sha256(image.read_bytes()).hexdigest()
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('No array decode during inference'))
    selected = dict(project=project, folder=merged, file=image)[selection]
    result = home._easy_inspect_path(str(selected))
    actions = {row[0]: row for row in result['actions']}
    assert result['actions'][0][0] == 'measure'
    assert actions['measure'][3] == str(project)
    assert actions['make_masks'][3] == str(image if selection == 'file' else merged)
    assert 'mask' not in actions
    native = {row[0] for row in APPS} | {key for children in folded_children().values() for key in children}
    expected = {key for key, module in workflow_map()['modules'].items()
                if set(module.get('inputs', ())) & {'measurements', 'crops'} and key in native}
    assert expected <= actions.keys()
    generated = result['context']['generated_outputs']
    assert generated['project'] == str(project)
    assert generated['artifacts'] == [dict(kind='merged', path=str(image), producer='mask')]
    assert generated['routes']['measure']['ready'] is True
    assert generated['routes']['graph_builder']['ready'] is False
    assert 'Measure' in actions['graph_builder'][2]
    assert 'graph_builder' not in result['context']['analysis_inputs']
    assert result['context']['analysis_inputs']['measure']['source'] == str(project)
    assert hashlib.sha256(image.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('shape,dtype', [((6, 8), 'uint16'), ((6, 8, 1), 'uint16'),
                                      ((2, 3, 6, 8, 4), 'uint16'), ((6, 8, 4), 'complex64')])
def test_canonical_name_alone_does_not_establish_merged_arrays(tmp_path, shape, dtype):
    merged = tmp_path / 'plate' / 'merged'
    merged.mkdir(parents=True)
    np.save(merged / 'field.npy', np.zeros(shape, dtype=dtype))
    for path in (merged, merged.parent, merged / 'field.npy'):
        result = home._easy_inspect_path(str(path))
        assert 'measure' not in {row[0] for row in result['actions']}
        assert result['context']['generated_outputs']['artifacts'] == []


def test_object_and_truncated_arrays_are_not_generated_measure_inputs(tmp_path):
    merged = tmp_path / 'plate' / 'merged'
    merged.mkdir(parents=True)
    image = merged / 'field.npy'
    np.save(image, np.zeros((4, 5, 3), dtype=object))
    assert not home._easy_inspect_path(str(image))['context']['generated_outputs']['artifacts']
    np.save(image, np.zeros((4, 5, 3), dtype=np.uint16))
    image.write_bytes(image.read_bytes()[:-10])
    assert not home._easy_inspect_path(str(image))['context']['generated_outputs']['artifacts']


def test_generated_file_inherits_only_its_own_measurement_database(tmp_path):
    project = tmp_path / 'plate1'
    merged = project / 'merged'
    merged.mkdir(parents=True)
    image = merged / 'field.npy'
    np.save(image, np.zeros((6, 8, 4), dtype=np.uint16))
    database = _database(project)
    _database(tmp_path / 'plate2')
    result = home._easy_inspect_path(str(image))
    route = result['context']['generated_outputs']['routes']['graph_builder']
    assert route['ready'] and route['source'] == str(database)
    assert result['context']['analysis_inputs']['graph_builder']['database'] == str(database)
    for key in ('activation', 'agreement', 'classifier_evaluation'):
        route = result['context']['generated_outputs']['routes'][key]
        assert route['ready'] is False and route['missing_inputs']
        assert key not in result['context']['analysis_inputs']
    assert {item['database'] for item in result['context']['analysis_inputs'].values()
            if 'database' in item} == {str(database)}


def test_annotation_database_does_not_upgrade_generated_measurement_readiness(tmp_path):
    project = tmp_path / 'plate'
    merged = project / 'merged'
    merged.mkdir(parents=True)
    image = merged / 'field.npy'
    np.save(image, np.zeros((6, 8, 4), dtype=np.uint16))
    _database(project, annotation_only=True)
    result = home._easy_inspect_path(str(image))
    assert result['context']['generated_outputs']['routes']['graph_builder']['ready'] is False
    assert 'graph_builder' not in result['context']['analysis_inputs']


def test_noncanonical_array_and_explicit_raw_tiff_do_not_inherit_project_analysis(tmp_path):
    project = tmp_path / 'plate'
    folder = project / 'raw'
    folder.mkdir(parents=True)
    array = folder / 'field.npy'
    np.save(array, np.zeros((6, 8, 4), dtype=np.uint16))
    image = folder / 'field.tif'
    Image.fromarray(np.zeros((6, 8), dtype=np.uint16)).save(image)
    _database(project)
    for selected in (array, image):
        result = home._easy_inspect_path(str(selected))
        assert not result['context']['generated_outputs']['artifacts']
        assert not {'measure', 'graph_builder', 'gate_editor'} & {row[0] for row in result['actions']}


def test_nested_crop_folder_uses_bounded_headers_and_keeps_measure_producer(tmp_path, monkeypatch):
    project = tmp_path / 'plate'
    crops = project / 'data' / 'measure_run_1234' / 'single_cell' / 'plate1_png'
    crops.mkdir(parents=True)
    image = crops / 'plate1_A01_1_1.png'
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(image)
    monkeypatch.setattr(Image.Image, 'load', lambda *args, **kwargs: pytest.fail('No PNG pixel decode'))
    for selected in (project / 'data', crops):
        result = home._easy_inspect_path(str(selected))
        generated = result['context']['generated_outputs']
        assert generated['project'] == str(project)
        assert any(item['kind'] == 'crops' and item['path'] == str(image) for item in generated['artifacts'])
        assert {'measure', 'annotate', 'classify_merged', 'umap'} <= generated['routes'].keys()


def test_numeric_mask_array_and_model_header_use_declared_output_locations(tmp_path):
    project = tmp_path / 'plate'
    masks = project / 'masks'
    masks.mkdir(parents=True)
    np.save(masks / 'field.npy', np.zeros((6, 8), dtype=np.uint16))
    result = home._easy_inspect_path(str(masks))
    assert any(item['kind'] == 'masks' for item in result['context']['generated_outputs']['artifacts'])
    model = project / 'model' / 'saved'
    model.mkdir(parents=True)
    with zipfile.ZipFile(model / 'classifier.pth', 'w') as output:
        output.writestr('archive/data.pkl', b'not executed')
    result = home._easy_inspect_path(str(project / 'model'))
    generated = result['context']['generated_outputs']
    assert generated['project'] == str(project)
    assert any(item['kind'] == 'classifier' for item in generated['artifacts'])
    assert {'classify_merged', 'activation'} <= generated['routes'].keys()
    selected_model = home._easy_inspect_path(str(model / 'classifier.pth'))
    assert selected_model['context']['generated_outputs']['project'] == str(project)
    np.save(masks / 'field.npy', np.zeros((6, 8), dtype=np.float32))
    assert not home._easy_inspect_path(str(masks))['context']['generated_outputs']['artifacts']


def test_calibrated_project_does_not_receive_ordinary_measure_from_merged(tmp_path):
    project = tmp_path / 'plate'
    merged = project / 'merged'
    merged.mkdir(parents=True)
    image = merged / 'field.npy'
    np.save(image, np.zeros((6, 8, 4), dtype=np.uint16))
    (project / 'plate_batch.json').write_text('{}')
    for selected in (project, merged, image):
        result = home._easy_inspect_path(str(selected))
        assert not result['context']['generated_outputs']['artifacts']
        assert 'measure' not in {row[0] for row in result['actions']}
