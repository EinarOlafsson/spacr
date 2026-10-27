"""Navigator suggestions require evidence inside the explicitly selected input."""
import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pytest
from PIL import Image

from spacr.qt.widgets import home


def _image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(48, dtype=np.uint16).reshape(6, 8)).save(path)
    return path


def _database(path, annotation_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        if annotation_only:
            connection.execute('CREATE TABLE image_annotations (plateID TEXT, rowID TEXT, columnID TEXT, image_id INTEGER)')
            connection.execute('CREATE TABLE image_annotations_meta (revision INTEGER)')
            connection.execute('CREATE TABLE image_annotations_history (revision INTEGER)')
        else:
            connection.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, cell_area REAL)')
            connection.execute('CREATE TABLE png_list (png_path TEXT)')
    return path


def test_tiff_folder_does_not_borrow_parent_analysis_or_promote_annotations(tmp_path, monkeypatch):
    selected = tmp_path / 'experiment' / 'raw'
    image = _image(selected / 'field.tif')
    annotations = _database(selected / 'measurements' / 'measurements.db', annotation_only=True)
    parent_database = _database(selected.parent / 'measurements' / 'measurements.db')
    sibling_database = _database(selected.parent / 'plate2' / 'measurements' / 'measurements.db')
    (selected.parent / 'counts.csv').write_text('condition,value\na,42\n')
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (image, annotations, parent_database, sibling_database)}
    entries = set(selected.rglob('*'))
    read_header = home._easy_read_header
    observed = []

    def bounded_header(path, *args, **kwargs):
        assert Path(path).is_relative_to(selected)
        observed.append(Path(path))
        return read_header(path, *args, **kwargs)

    monkeypatch.setattr(home, '_easy_read_header', bounded_header)
    result = home._easy_inspect_path(str(selected))
    assert [row[0] for row in result['actions']] == ['make_masks', 'mask']
    assert all(row[3] == str(selected) for row in result['actions'])
    assert result['context']['analysis_inputs'] == {}
    assert result['context']['plate_inputs'] == []
    assert result['context']['image_folders'] == [str(selected)]
    assert result['context']['database_schemas'][0]['annotation_only']
    assert not result['context']['database_schemas'][0]['numeric']
    assert observed and len(observed) < 32
    assert set(selected.rglob('*')) == entries
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in before.items())


def test_explicit_annotation_database_remains_browsable(tmp_path):
    database = _database(tmp_path / 'measurements' / 'measurements.db', annotation_only=True)
    result = home._easy_inspect_path(str(database))
    assert [(row[0], row[3]) for row in result['actions']] == [('db_browser', str(database))]
    assert result['context']['analysis_inputs'] == {}


@pytest.mark.parametrize('extension', ['tif', 'png'])
def test_already_readable_images_do_not_need_format_conversion(tmp_path, extension):
    image = _image(tmp_path / ('field.' + extension))
    before = image.read_bytes()
    for selected in (image, tmp_path):
        result = home._easy_inspect_path(str(selected))
        assert [row[0] for row in result['actions']] == ['make_masks', 'mask']
    assert image.read_bytes() == before


def test_container_import_stays_available_without_parent_readers(tmp_path):
    selected = tmp_path / 'selected'
    selected.mkdir()
    container = selected / 'acquisition.czi'
    container.write_bytes(b'ZISRAWFILE' + bytes(54))
    sibling = tmp_path / 'unrelated'
    sibling.mkdir()
    (sibling / 'other.lif').write_bytes(bytes(64))
    result = home._easy_inspect_path(str(selected))
    assert [(row[0], row[3]) for row in result['actions']] == [('convert', str(selected))]
    assert [row['path'] for row in result['context']['reader_candidates']] == [str(container)]


def test_sampling_stays_within_selected_master_and_its_existing_budgets(tmp_path):
    selected = tmp_path / 'master'
    selected.mkdir()
    for index in range(12):
        child = selected / str(index)
        child.mkdir()
        for number in range(24):
            (child / (str(number) + '.txt')).write_text('local')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'measurements.db').write_bytes(b'unrelated')
    files, folders, unreadable, limited = home._easy_local_sample(selected, sample_limit=20)
    assert len(files) <= 20
    assert len(folders) <= home._EASY_DIRECTORY_LIMIT
    assert all(path.is_relative_to(selected) for path in (*files, *folders))
    assert unreadable == 0 and limited


def test_selected_table_and_image_do_not_inherit_each_others_actions(tmp_path, monkeypatch):
    table = tmp_path / 'objects.csv'
    table.write_text('label,area\n1,12\n2,4\n')
    images = [_image(tmp_path / name) for name in ('scan_101_C1.tif', 'scan_101_C2.tif')]
    database = _database(tmp_path / 'measurements' / 'measurements.db')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"app_key":"mask","schema_version":1,"status":"failed"}')
    originals = {path: path.read_bytes() for path in (table, *images, database, manifest)}

    def no_pixels(*args, **kwargs):
        raise AssertionError('selected-input suggestions must not decode pixels')

    from PIL import TiffImagePlugin

    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, 'load', no_pixels)
    selected_table = home._easy_inspect_path(str(table))
    assert [row[0] for row in selected_table['actions']] == ['foreign', 'graph_builder', 'gate_editor']
    assert all(row[3] == str(table) for row in selected_table['actions'])
    assert 'regex' not in selected_table
    assert selected_table['context']['database_schemas'] == []
    assert selected_table['context']['image_folders'] == []
    assert selected_table['context']['run_evidence'] == dict(manifests=[], results=[])
    selected_image = home._easy_inspect_path(str(images[0]))
    assert [row[0] for row in selected_image['actions']] == ['make_masks', 'mask']
    assert selected_image['regex']['example_path'] in {str(image) for image in images}
    assert selected_image['context']['database_schemas'] == []
    assert selected_image['context']['run_evidence'] == dict(manifests=[], results=[])
    assert all(path.read_bytes() == content for path, content in originals.items())
