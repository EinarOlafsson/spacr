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


@pytest.mark.parametrize('bigtiff,extra_tags', [(False, 0), (True, 350)])
def test_tiff_headers_keep_unreviewed_acquisition_axes_unresolved(
        tmp_path, monkeypatch, bigtiff, extra_tags):
    import tifffile

    path = tmp_path / 'acquisition.tif'
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8)
    description = 'Acquisition notebook: ' + 'unreviewed dimensions; ' * 60
    tifffile.imwrite(
        path, pixels, bigtiff=bigtiff, metadata=None,
        description=description,
        extratags=[(65000 + index, 'B', 1, 1, False)
                   for index in range(extra_tags)])
    before = path.read_bytes()
    read_header = home._easy_read_header
    reads = []

    def bounded_header(selected, limit, offset=0):
        reads.append((limit, offset))
        return read_header(selected, limit, offset)

    def no_pixels(*args, **kwargs):
        pytest.fail('Navigator provenance must use headers, not decode TIFF pixels')

    monkeypatch.setattr(home, '_easy_read_header', bounded_header)
    monkeypatch.setattr(tifffile, 'imread', no_pixels)
    assert home._easy_file_kind(path) == 'image'
    assert sum(limit for limit, _ in reads) <= home._EASY_HEADER_LIMIT
    reads.clear()
    metadata = {}
    assert home._easy_file_kind(path, metadata) == 'image'
    assert metadata == dict(shape=[6, 8], axes='YX', axes_unresolved=True)
    assert sum(limit for limit, _ in reads) <= home._EASY_HEADER_LIMIT
    assert path.read_bytes() == before


def test_paired_masks_do_not_make_partial_merged_outputs_scientific_evidence(
        tmp_path, monkeypatch):
    import tifffile

    root = tmp_path / 'selected-plate'
    image = _image(root / 'field.tif')
    mask = root / 'field_masks.tif'
    labels = np.zeros((6, 8), dtype=np.uint32)
    labels[1:5, 2:6] = 70000
    tifffile.imwrite(mask, labels)
    before = {path: path.read_bytes() for path in (image, mask)}
    merged = root / 'merged'
    merged.mkdir()
    images = [image, mask]
    expected_pair = {str(image): str(mask)}

    def no_pixels(*args, **kwargs):
        pytest.fail('Header evidence must not load image arrays or label pixels')

    monkeypatch.setattr(np, 'load', no_pixels)
    monkeypatch.setattr(tifffile, 'imread', no_pixels)
    assert home._easy_paired_layout(images, root) == dict(
        kind='unstructured', pairs=expected_pair, project=str(root),
        filename_evidence=True)
    assert not home._easy_merged_evidence(root)

    (merged / 'notes.txt').write_text('Merge is still being prepared.')
    (merged / 'pending').mkdir()
    plane = merged / 'unmerged-plane.npy'
    np.save(plane, np.arange(48, dtype=np.uint16).reshape(6, 8))
    assert home._easy_paired_layout(images, root)['kind'] == 'unstructured'
    assert not home._easy_merged_evidence(root)

    single = merged / 'single-channel.npy'
    np.save(single, np.zeros((6, 8, 1), dtype=np.uint16))
    assert home._easy_paired_layout(images, root)['kind'] == 'spacr'
    assert not home._easy_merged_evidence(root)

    channels = merged / 'merged-channels.npy'
    np.save(channels, np.stack((np.full((6, 8), 400, dtype=np.uint32),
                               labels), axis=-1))
    single.unlink()
    assert home._easy_merged_evidence(root)
    paired = home._easy_paired_layout(images, root)
    assert paired['kind'] == 'spacr' and paired['pairs'] == expected_pair
    assert paired['project'] == str(root) and paired['filename_evidence']
    assert all(path.read_bytes() == content for path, content in before.items())


def test_generic_saved_settings_require_an_owned_manifest_before_module_preview(tmp_path):
    import json

    folder = tmp_path / 'selected-settings'
    folder.mkdir()
    settings = folder / 'settings.json'
    values = dict(src=str(tmp_path / 'unopened-input'), regression_type='ols')
    settings.write_text(json.dumps(values))
    original = settings.read_bytes()
    manifest = folder / 'manifest.json'

    missing = home._easy_inspect_path(str(settings))
    assert 'saved_settings' not in missing['context']
    assert not any(row[0] == 'regression' for row in missing['actions'])
    assert list(folder.iterdir()) == [settings]
    assert settings.read_bytes() == original

    manifest.write_text(json.dumps(dict(schema_version=1, app_key='foreign-acquisition',
                                        status='success')))
    foreign_bytes = manifest.read_bytes()
    foreign = home._easy_inspect_path(str(settings))
    assert 'saved_settings' not in foreign['context']
    assert not any(row[0] in ('regression', 'foreign-acquisition')
                   for row in foreign['actions'])
    assert manifest.read_bytes() == foreign_bytes
    assert settings.read_bytes() == original

    manifest.write_text(json.dumps(dict(schema_version=1, app_key='regression',
                                        status='success')))
    manifest_bytes = manifest.read_bytes()
    owned = home._easy_inspect_path(str(settings))
    assert owned['context']['saved_settings'] == dict(
        path=str(settings), module='regression', settings=values)
    assert owned['actions'][0][0] == 'regression'
    assert owned['actions'][0][3] == str(settings)
    assert settings.read_bytes() == original and manifest.read_bytes() == manifest_bytes
    assert not (tmp_path / 'unopened-input').exists()
    assert set(folder.iterdir()) == {settings, manifest}


def test_finished_run_distinguishes_object_results_and_empty_diagnostics(tmp_path):
    import json

    folder = tmp_path / 'finished-regression'
    folder.mkdir()
    manifest = folder / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, app_key='regression',
                                        status='success')))
    objects = folder / 'results.csv'
    objects.write_text('label,cell_area\n1,12\n2,4\n')
    coefficients = folder / 'results_gene.csv'
    coefficients.write_text('feature,coefficient\ngene_a,0.5\ngene_b,-0.25\n')
    diagnostics = folder / 'diagnostics'
    diagnostics.mkdir()
    originals = {path: path.read_bytes() for path in (manifest, objects, coefficients)}

    for extra in (None, diagnostics / 'pending'):
        if extra is not None:
            extra.mkdir()
        inspected = home._easy_inspect_path(str(folder))
        assert inspected['context']['run_evidence'] == dict(
            manifests=[], results=[dict(path=str(coefficients), folder=str(folder),
                                       diagnostics=False, diagnostics_folder=None)])
        assert 'regression_diagnostics' not in inspected['context']['analysis_inputs']
        assert all(path.read_bytes() == content for path, content in originals.items())

    output = diagnostics / 'fit_summary.csv'
    output.write_text('metric,value\nresidual_scale,0.2\n')
    output_bytes = output.read_bytes()
    inspected = home._easy_inspect_path(str(folder))
    assert inspected['context']['run_evidence']['results'] == [dict(
        path=str(coefficients), folder=str(folder), diagnostics=True,
        diagnostics_folder=str(diagnostics))]
    assert inspected['context']['analysis_inputs']['regression_diagnostics'][
        'diagnostics_folder'] == str(diagnostics)
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert output.read_bytes() == output_bytes


def test_ops_sidecars_do_not_turn_an_unrelated_numeric_table_into_a_library(tmp_path):
    import json

    root = tmp_path / 'screen'
    genotype = root / 'genotype'
    for cycle in (1, 2):
        _image(genotype / f'10X_c{cycle}_A1_CY3_Site-0.tif')
    unrelated = root / 'object-counts.csv'
    unrelated.write_text('well,count\nA01,12\nA02,8\n')
    metadata = root / 'metadata.json'
    metadata.write_text(json.dumps([dict(channel='DAPI'), dict(channel='GFP')]))
    originals = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    result = home._easy_inspect_path(str(root))
    assert result['context']['submodule_settings']['ops']['settings'] == dict(
        genotype_source=str(genotype))
    assert result['context']['metadata_files'] == [dict(
        name='metadata.json', folder=str(root), fields=[], truncated=False)]
    assert all(path.read_bytes() == content for path, content in originals.items())

    library = root / 'library.csv'
    library.write_text('prefix,sequence,gene\nACGT,ACGT,gene_a\n')
    result = home._easy_inspect_path(str(root))
    assert result['context']['submodule_settings']['ops']['settings'] == dict(
        genotype_source=str(genotype), ops_library=str(library))
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_selected_plain_tiff_does_not_borrow_sibling_filename_or_dimensions(tmp_path):
    import tifffile

    selected = tmp_path / 'overview.tif'
    tifffile.imwrite(selected, np.arange(48, dtype=np.uint16).reshape(6, 8),
                     metadata=dict(axes='YX'))
    sibling = _image(tmp_path / 'plate1_A01_T0001F001L01A01Z01C01.tif')
    originals = {path: path.read_bytes() for path in (selected, sibling)}
    result = home._easy_inspect_path(str(selected))
    proposal = result['regex']
    assert proposal['example_path'] == str(selected)
    assert proposal['required'] and proposal['groups'] == []
    assert proposal['pattern']
    assert 'metadata_preset' not in result
    assert not result['context']['dimension_routes']
    dimensions = next(row for row in result['context']['image_dimensions']
                      if row['path'] == str(selected))
    assert dimensions['axes'] == 'YX' and dimensions['z'] == dimensions['t'] == 1
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_generated_crop_depth_and_model_notes_keep_header_evidence_bounded(tmp_path):
    import zipfile

    project = tmp_path / 'plate'
    crops = project / 'data' / 'run' / 'cells' / 'plate_png'
    supported = _image(crops / 'cell.png')
    beyond_budget = _image(crops / 'extra' / 'deep_png' / 'ignored.png')
    model = project / 'model'
    model.mkdir()
    notes = model / 'training-notes.txt'
    notes.write_text('Review this checkpoint before applying it.')
    checkpoint = model / 'classifier.pth'
    with zipfile.ZipFile(checkpoint, 'w') as archive:
        archive.writestr('archive/data.pkl', b'header-only fixture; never deserialized')
    originals = {path: path.read_bytes() for path in project.rglob('*') if path.is_file()}
    result = home._easy_inspect_path(str(project))
    artifacts = result['context']['generated_outputs']['artifacts']
    assert dict(kind='crops', path=str(supported), producer='measure') in artifacts
    assert dict(kind='classifier', path=str(checkpoint), producer='classify_merged') in artifacts
    assert not any(row['path'] in (str(notes), str(beyond_budget)) for row in artifacts)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_generated_merge_bounds_partial_headers_then_verifies_selected_completion(tmp_path):
    merged = tmp_path / 'plate' / 'merged'
    merged.mkdir(parents=True)
    partials = []
    for index in range(5):
        partial = merged / f'partial-{index}.npy'
        np.save(partial, np.full((6, 8, 1), index, dtype=np.uint16))
        partials.append(partial)
    originals = {path: path.read_bytes() for path in merged.iterdir()}
    result = home._easy_inspect_path(str(merged))
    assert not result['context']['generated_outputs']['artifacts']
    assert 'measure' not in {row[0] for row in result['actions']}
    assert all(path.read_bytes() == content for path, content in originals.items())

    complete = partials.pop()
    np.save(complete, np.stack((np.ones((6, 8), dtype=np.uint16),
                               np.full((6, 8), 500, dtype=np.uint16)), axis=-1))
    completed_bytes = complete.read_bytes()
    result = home._easy_inspect_path(str(complete))
    assert result['context']['generated_outputs']['artifacts'] == [dict(
        kind='merged', path=str(complete), producer='mask')]
    assert result['context']['generated_outputs']['routes']['measure']['ready']
    assert all(path.read_bytes() == originals[path] for path in partials)
    assert complete.read_bytes() == completed_bytes


def test_calibrated_marker_without_completed_data_gives_guidance_only(tmp_path):
    marker = tmp_path / 'plate_batch.json'
    marker.write_text('{}')
    before = marker.read_bytes()
    result = home._easy_inspect_path(str(tmp_path))
    assert not result['actions']
    assert not result['context']['generated_outputs']['artifacts']
    assert 'calibrated-workflow marker' in result['message']
    assert 'completed child dataset' in result['message']
    assert marker.read_bytes() == before
    assert list(tmp_path.iterdir()) == [marker]


def test_master_regression_table_is_not_a_child_plate_generated_result(tmp_path):
    master = tmp_path / 'master'
    plate = master / 'plate1'
    database = _database(plate / 'measurements' / 'measurements.db')
    coefficients = master / 'results.csv'
    coefficients.write_text('feature,coefficient\ngene_a,0.5\ngene_b,-0.25\n')
    originals = {path: path.read_bytes() for path in (database, coefficients)}
    result = home._easy_inspect_path(str(master))
    generated = result['context']['generated_outputs']
    assert generated['project'] == str(plate)
    assert any(row['kind'] == 'measurements' and row['path'] == str(database)
               for row in generated['artifacts'])
    assert not any(row['kind'] == 'regression' for row in generated['artifacts'])
    assert result['context']['run_evidence']['results'][0]['path'] == str(coefficients)
    assert all(path.read_bytes() == content for path, content in originals.items())


def _deny_selected_home_access(monkeypatch, *, scans=(), stats=()):
    """Make only named metadata operations fail, including under root accounts."""
    original = home.os
    scans, stats = set(map(Path, scans)), set(map(Path, stats))

    class SelectedAccess:
        def __getattr__(self, name):
            return getattr(original, name)

        def scandir(self, path):
            if Path(path) in scans:
                raise PermissionError('selected directory cannot be listed')
            return original.scandir(path)

        def stat(self, path, *args, **kwargs):
            if Path(path) in stats:
                raise PermissionError('selected file metadata unavailable')
            return original.stat(path, *args, **kwargs)

    monkeypatch.setattr(home, 'os', SelectedAccess())


def _ome_input(path):
    """Write a real TIFF with bounded OME dimension metadata, not pixel stubs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.arange(48, dtype=np.uint16).reshape(6, 8)).save(
        path, tiffinfo={270: '<OME><Image><Pixels SizeT="1" SizeZ="1" SizeC="1"/></Image></OME>'})
    return path


@pytest.fixture
def spatial_input_evidence(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, project = tmp_path / 'acquisition', tmp_path / 'segmented'
    image = _ome_input(root / 'morphology_focus' / 'field.ome.tif')
    experiment = root / 'experiment.xenium'
    experiment.write_text(json.dumps({'pixel_size': .2125}), encoding='utf-8')
    transcripts = root / 'transcripts.parquet'
    pq.write_table(pa.table({'x_location': [1.0], 'y_location': [2.0],
                             'feature_name': ['gene_a']}), transcripts)
    database = _database(project / 'measurements' / 'measurements.db')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE spatial_fields (source TEXT, channels INTEGER)')
        connection.executemany('INSERT INTO spatial_fields VALUES (?, 3)', [(None,), (str(root),)])
    merged = project / 'merged' / 'field.npy'
    merged.parent.mkdir()
    np.save(merged, np.zeros((6, 8, 3), dtype=np.uint16))
    schema = dict(home._easy_database_evidence(database), path=str(database))
    return dict(root=root, project=project, image=image, database=database, merged=merged,
                schema=schema, files=[experiment, transcripts, image])


def test_analysis_discovery_refuses_a_named_execute_only_directory(tmp_path, monkeypatch):
    root = tmp_path / 'selected'
    root.mkdir()
    database = _database(root / 'measurements.db')
    before = database.read_bytes()
    details = {'unchanged': True}
    root.chmod(0o111)
    _deny_selected_home_access(monkeypatch, scans=[root])
    try:
        assert home._easy_analysis_inputs(root, details) == []
        assert details == {'unchanged': True}
        assert database.read_bytes() == before
    finally:
        root.chmod(0o700)


def test_analysis_discovery_bounds_immediate_plate_candidates_and_databases(tmp_path):
    root = tmp_path / 'selected'
    root.mkdir()
    known = {_database(root / 'measurements.db')}
    for number in range(18):
        plate = root / f'plate-{number:02d}'
        plate.mkdir()
        if number < 9:
            known.add(_database(plate / 'measurements' / 'measurements.db'))
    hidden = _database(root / '.private' / 'measurements.db')
    outside = _database(tmp_path / 'other' / 'measurements.db')
    details = {}
    found = home._easy_analysis_inputs(root, details)
    assert len(found) == 8 and len(set(found)) == 8
    assert set(found) <= known
    assert hidden not in found and outside not in found
    assert details == {'sampled_plate_candidates': 17, 'truncated': True}


@pytest.mark.parametrize('failure', ['invalid_text', 'no_library_columns', 'denied'])
def test_ops_library_refusal_preserves_both_real_acquisition_sources(tmp_path, monkeypatch, failure):
    root = tmp_path / 'screen'
    cycled = _image(root / 'genotype' / '10X_c2_A1_CY3_Site-0.tif')
    uncycled = _image(root / 'phenotype' / '20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-0.tif')
    library = root / 'library.csv'
    library.write_bytes(b'\xff,barcode\nACGT,4\n' if failure == 'invalid_text' else
                        b'gene,value\na,2\n' if failure == 'no_library_columns' else
                        b'prefix,gene\nACGT,gene_a\n')
    note = root / 'readme.txt'
    note.write_text('acquisition notes')
    deep_library = root / 'too' / 'deep' / 'library.csv'
    deep_library.parent.mkdir(parents=True)
    deep_library.write_text('prefix,gene\nACGT,gene_a\n')
    before = {path: path.read_bytes() for path in (cycled, uncycled, library, deep_library, note)}
    if failure == 'denied':
        _deny_selected_home_access(monkeypatch, stats=[library])
    hints = home._easy_specialized_inputs(root, [note, deep_library, library, cycled, uncycled],
                                           [cycled, uncycled], [])
    assert set(hints) == {'ops'}
    assert hints['ops']['source'] == str(cycled.parent)
    assert hints['ops']['settings'] == dict(genotype_source=str(cycled.parent),
                                          phenotype_source=str(uncycled.parent))
    assert all(path.read_bytes() == content for path, content in before.items())


def test_xenium_focus_listing_denial_does_not_invent_an_image(spatial_input_evidence, monkeypatch):
    case = spatial_input_evidence
    focus = case['image'].parent
    focus.chmod(0o111)
    _deny_selected_home_access(monkeypatch, scans=[focus])
    try:
        hints = home._easy_specialized_inputs(case['root'], case['files'], [case['image']], [])
        assert hints == {}
    finally:
        focus.chmod(0o700)


def test_spatial_image_stat_failure_uses_only_a_readable_morphology_fallback(
        spatial_input_evidence, monkeypatch):
    case = spatial_input_evidence
    preferred = _ome_input(case['root'] / 'morphology_focus.ome.tif')
    fallback = _ome_input(case['root'] / 'morphology_mip.ome.tif')
    _deny_selected_home_access(monkeypatch, stats=[preferred])
    before = {path: path.read_bytes() for path in [preferred, fallback, *case['files']]}
    hints = home._easy_specialized_inputs(case['root'], case['files'] + [preferred, fallback],
                                           [preferred, fallback, case['image']], [])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['image'] == str(fallback)
    assert hints['spatial_masks']['settings'] == dict(spatial_source=str(case['root']),
                                                     spatial_platform='xenium')
    assert all(path.read_bytes() == content for path, content in before.items())


def test_spatial_mapping_rechecks_the_actual_table_after_a_stale_schema_snapshot(spatial_input_evidence):
    case = spatial_input_evidence
    irrelevant = {'path': str(case['database']), 'tables': ['cell']}
    schemas = [irrelevant, case['schema']]
    before = {path: path.read_bytes() for path in [*case['files'], case['database'], case['merged']]}
    hints = home._easy_specialized_inputs(case['root'], case['files'], [case['image']], schemas)
    assert hints['spatial_map']['settings'] == dict(spatial_source=str(case['root']),
        spatial_platform='xenium', spatial_masks_dir=str(case['project']))
    assert all(path.read_bytes() == content for path, content in before.items())
    with sqlite3.connect(case['database']) as connection:
        connection.execute('DROP TABLE spatial_fields')
    after_drop = case['database'].read_bytes()
    refused = home._easy_specialized_inputs(case['root'], case['files'], [case['image']], schemas)
    assert set(refused) == {'spatial_masks'}
    assert refused['spatial_masks'] == hints['spatial_masks']
    assert case['database'].read_bytes() == after_drop


@pytest.mark.parametrize('missing', ['wrong_source', 'no_merged', 'one_channel', 'truncated', 'denied'])
def test_spatial_map_refusals_keep_the_matching_masks_route(spatial_input_evidence, monkeypatch, missing):
    case = spatial_input_evidence
    if missing == 'wrong_source':
        with sqlite3.connect(case['database']) as connection:
            connection.execute('UPDATE spatial_fields SET source = ?', (str(case['root'].parent / 'other'),))
    elif missing == 'no_merged':
        case['merged'].unlink()
    elif missing == 'one_channel':
        np.save(case['merged'], np.zeros((6, 8, 1), dtype=np.uint16))
    elif missing == 'truncated':
        case['merged'].write_bytes(b'\x93NUMPY\x01\x00')
    else:
        _deny_selected_home_access(monkeypatch, scans=[case['merged'].parent])
    before = {path: path.read_bytes() for path in case['files'] + [case['database']]}
    hints = home._easy_specialized_inputs(case['root'], case['files'], [case['image']], [case['schema']])
    assert set(hints) == {'spatial_masks'}
    assert hints['spatial_masks']['source'] == str(case['root'])
    assert hints['spatial_masks']['image'] == str(case['image'])
    assert hints['spatial_masks']['settings'] == dict(spatial_source=str(case['root']),
                                                     spatial_platform='xenium')
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('shape', [(6, 8), (6, 8, 3)])
def test_paired_images_require_a_real_three_dimensional_merged_header(tmp_path, shape):
    image, mask = _image(tmp_path / 'field.tif'), _image(tmp_path / 'masks' / 'field.tif')
    ignored = _image(tmp_path / 'outside' / 'deeper' / 'not-local.tif')
    merged = tmp_path / 'merged'
    merged.mkdir()
    (merged / 'notes.txt').write_text('not an array')
    array = merged / 'field.npy'
    np.save(array, np.zeros(shape, dtype=np.uint16))
    before = {path: path.read_bytes() for path in (image, mask, ignored, array)}
    evidence = home._easy_paired_layout([image, mask, ignored], tmp_path)
    assert evidence == dict(kind='spacr' if len(shape) == 3 else 'unstructured',
        pairs={str(image): str(mask)}, project=str(tmp_path), filename_evidence=True)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_paired_images_survive_a_named_execute_only_merged_folder(tmp_path, monkeypatch):
    image, mask = _image(tmp_path / 'field.tif'), _image(tmp_path / 'field_mask.tif')
    merged = tmp_path / 'merged'
    merged.mkdir()
    np.save(merged / 'field.npy', np.zeros((6, 8, 3), dtype=np.uint16))
    merged.chmod(0o111)
    _deny_selected_home_access(monkeypatch, scans=[merged])
    try:
        assert home._easy_paired_layout([image, mask], tmp_path) == dict(kind='unstructured',
            pairs={str(image): str(mask)}, project=str(tmp_path), filename_evidence=True)
    finally:
        merged.chmod(0o700)


@pytest.mark.parametrize('state', ['matched', 'no_bins', 'denied_bins', 'invalid_scale',
                                   'invalid_count_scale', 'missing_positions', 'denied_counts'])
def test_visium_hd_routes_use_existing_calibrated_bins_and_readable_count_evidence(tmp_path, monkeypatch, state):
    import json
    import h5py

    root = tmp_path / 'visium'
    slide = _ome_input(root / 'slide.tif')
    binned = root / 'binned_outputs'
    binned.mkdir()
    (binned / 'square_000um').mkdir()
    (binned / 'square_004um').write_text('not a bin directory')
    scales = root / 'spatial' / 'scalefactors_json.json'
    scales.parent.mkdir()
    scales.write_text(json.dumps({'microns_per_pixel': 'not numeric' if state == 'invalid_scale' else .5}))
    known = [slide, scales]
    count_folder = None
    if state != 'no_bins':
        for size in (16, 2):
            folder = binned / f'square_{size:03d}um'
            spatial = folder / 'spatial'
            spatial.mkdir(parents=True)
            scale = spatial / 'scalefactors_json.json'
            scale.write_text(json.dumps({'microns_per_pixel': 'invalid' if state in
                ('invalid_scale', 'invalid_count_scale') else .5}))
            known.append(scale)
            if size == 2:
                count_folder = folder
        if state != 'missing_positions':
            positions = count_folder / 'spatial' / 'tissue_positions.csv'
            positions.write_text('barcode,in_tissue,pxl_row_in_fullres,pxl_col_in_fullres\na,1,2,3\n')
            known.append(positions)
        counts = count_folder / 'filtered_feature_bc_matrix.h5'
        with h5py.File(counts, 'w') as handle:
            handle.create_dataset('matrix/data', data=np.array([1], dtype=np.int32))
        known.append(counts)
    project = tmp_path / 'segmented'
    database = _database(project / 'measurements.db')
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE spatial_fields (source TEXT)')
        connection.execute('INSERT INTO spatial_fields VALUES (?)', (str(root),))
    merged = project / 'merged' / 'field.npy'
    merged.parent.mkdir()
    np.save(merged, np.zeros((6, 8, 3), dtype=np.uint16))
    schema = dict(home._easy_database_evidence(database), path=str(database))
    known.extend([database, merged])
    before = {path: path.read_bytes() for path in known}
    denied = binned if state == 'denied_bins' else count_folder if state == 'denied_counts' else None
    if denied is not None:
        denied.chmod(0o111)
        _deny_selected_home_access(monkeypatch, scans=[denied])
    try:
        hints = home._easy_specialized_inputs(root, [slide, scales], [slide], [schema])
        if state == 'invalid_scale':
            assert hints == {}
        else:
            expected = dict(spatial_source=str(root), spatial_platform='visium_hd')
            if state not in ('no_bins', 'denied_bins'):
                expected['spatial_bin_um'] = 2
            assert hints['spatial_masks']['settings'] == expected
            assert hints['spatial_masks']['image'] == str(slide)
            if state == 'matched':
                assert set(hints) == {'spatial_masks', 'spatial_map'}
                assert hints['spatial_map']['settings'] == dict(expected, spatial_masks_dir=str(project))
            else:
                assert set(hints) == {'spatial_masks'}
        assert all(path.read_bytes() == content for path, content in before.items())
    finally:
        if denied is not None:
            denied.chmod(0o700)


def test_selected_tiff_replaced_during_sampling_cannot_borrow_sibling_sources(
        tmp_path, monkeypatch):
    selected = _image(tmp_path / 'selected.tif')
    sibling = _image(tmp_path / 'sibling.tiff')
    array = tmp_path / 'sibling.npy'
    np.save(array, np.ones((6, 8, 2), dtype=np.uint16))
    table = tmp_path / 'objects.csv'
    table.write_text('label,area\n1,12\n2,4\n')
    archive = tmp_path / 'archive' / selected.name
    archive.parent.mkdir()
    before = {path: path.read_bytes() for path in (selected, sibling, array, table)}
    scan = home._easy_local_sample
    replaced = []

    def replace_after_real_scan(path, *args, **kwargs):
        sampled = scan(path, *args, **kwargs)
        if not replaced:
            assert path == selected and path.is_file()
            selected.rename(archive)
            selected.mkdir()
            replaced.append(True)
        return sampled

    monkeypatch.setattr(home, '_easy_local_sample', replace_after_real_scan)
    result = home._easy_inspect_path(str(selected))
    assert replaced and selected.is_dir()
    assert archive.read_bytes() == before[selected]
    assert all(path.read_bytes() == before[path] for path in (sibling, array, table))
    foreign = [row for row in result['actions']
               if not Path(row[3]).is_relative_to(selected)]
    assert not foreign, ('Replaced selection borrowed sibling sources', foreign)
    assert result['actions'] == []
    assert result['message']
    assert 'context' not in result and 'settings_overrides' not in result


def _identity_checkpoint_files(tmp_path, image_count, auxiliary_in_input=True):
    root = tmp_path / 'input'
    selected = _image(root / 'selected.tif')
    images = [selected] + [_image(root / f'unstructured-{index:03}.tif')
                           for index in range(image_count - 1)]
    auxiliary = root if auxiliary_in_input else tmp_path / 'unrelated'
    auxiliary.mkdir(exist_ok=True)
    array = auxiliary / 'sibling.npy'
    np.save(array, np.ones((6, 8, 2), dtype=np.uint16))
    table = auxiliary / 'objects.csv'
    table.write_text('label,area\n1,12\n2,4\n')
    database = _database(auxiliary / 'measurements.db')
    archive = tmp_path / 'archive' / 'selected.tif'
    archive.parent.mkdir()
    before = {path: path.read_bytes() for path in (*images, array, table, database)}
    return root, selected, archive, before


def _observe_identity_checkpoint_decoders(monkeypatch):
    from PIL import TiffImagePlugin
    import tifffile

    calls = []
    for owner, name in ((TiffImagePlugin.TiffImageFile, 'load'),
                        (tifffile, 'imread'), (np, 'load')):
        read = getattr(owner, name)

        def observe(*args, _read=read, _name=name, **kwargs):
            calls.append(_name)
            return _read(*args, **kwargs)

        monkeypatch.setattr(owner, name, observe)
    return calls


def _assert_identity_checkpoint_refusal(result, selected, archive, before):
    from spacr.qt.i18n import tr

    assert selected.is_dir()
    assert result == {'path': str(selected), 'actions': [], 'message': tr(
        'This input could not be inspected. Check permissions and choose another path.')}
    assert 'context' not in result and 'settings_overrides' not in result
    assert all((archive if path == selected else path).read_bytes() == content
               for path, content in before.items())


def test_stable_unstructured_image_folder_uses_the_real_expanded_filename_sample(tmp_path, monkeypatch):
    root, selected, archive, before = _identity_checkpoint_files(tmp_path, 101, auxiliary_in_input=False)
    entries = set(root.rglob('*'))
    decodes = _observe_identity_checkpoint_decoders(monkeypatch)
    scan = home._easy_local_sample
    samples = []

    def observe_scan(path, sample_limit=100):
        sampled = scan(path, sample_limit)
        samples.append((path, sample_limit, sampled))
        return sampled

    monkeypatch.setattr(home, '_easy_local_sample', observe_scan)
    result = home._easy_inspect_path(str(root))
    assert [(path, limit, len(sampled[0]), sampled[3]) for path, limit, sampled in samples] == [
        (root, 100, 100, True), (root, 1000, 101, False)]
    hints = home._easy_filename_hints(samples[0][2][0])
    assert not any(hint['matched'] == hint['sampled'] and hint['sampled'] for hint in hints)
    assert [row[0] for row in result['actions']] == ['make_masks', 'mask']
    assert all(row[3] == str(root) for row in result['actions'])
    assert result['context']['filename_sample_limit'] == 1000
    assert result['context']['sampled_files'] == 101
    assert result['context']['sampled_image_filenames'] == 101
    assert result['context']['image_folders'] == [str(root)]
    assert result['context']['analysis_inputs'] == {}
    assert not result['context']['truncated']
    assert set(root.rglob('*')) == entries
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not archive.exists() and selected.is_file() and decodes == []


def test_selected_tiff_replaced_after_the_real_expanded_scan_cannot_publish_context(tmp_path, monkeypatch):
    root, selected, archive, before = _identity_checkpoint_files(tmp_path, 101)
    decodes = _observe_identity_checkpoint_decoders(monkeypatch)
    scan = home._easy_local_sample
    samples = []

    def replace_after_expanded_scan(path, sample_limit=100):
        sampled = scan(path, sample_limit)
        samples.append((path, sample_limit, len(sampled[0]), sampled[3]))
        if len(samples) == 2:
            assert path == selected and sample_limit == 1000 and selected.is_file()
            selected.rename(archive)
            selected.mkdir()
        return sampled

    monkeypatch.setattr(home, '_easy_local_sample', replace_after_expanded_scan)
    result = home._easy_inspect_path(str(selected))
    assert samples == [(selected, 100, 100, True), (selected, 1000, 104, False)]
    _assert_identity_checkpoint_refusal(result, selected, archive, before)
    assert decodes == []


def test_selected_tiff_replaced_after_real_header_classification_cannot_publish_context(tmp_path, monkeypatch):
    root, selected, archive, before = _identity_checkpoint_files(tmp_path, 2)
    decodes = _observe_identity_checkpoint_decoders(monkeypatch)
    classify = home._easy_file_kind
    replaced = []

    def replace_after_selected_kind(path, metadata=None):
        kind = classify(path, metadata)
        if path == selected and not replaced:
            assert kind == 'image' and metadata['shape'] == [6, 8]
            selected.rename(archive)
            selected.mkdir()
            replaced.append((path, kind))
        return kind

    monkeypatch.setattr(home, '_easy_file_kind', replace_after_selected_kind)
    result = home._easy_inspect_path(str(selected))
    assert replaced == [(selected, 'image')]
    _assert_identity_checkpoint_refusal(result, selected, archive, before)
    assert decodes == []
