"""Content and bounded schema discovery preserve actual module inputs."""
import gzip
import hashlib
import sqlite3

import numpy as np

from spacr.qt.widgets import home


def _measurements(path):
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, cell_area REAL)')
        connection.execute('CREATE TABLE png_list (png_path TEXT)')
    return path


def test_master_analysis_discovers_scientific_database_and_all_requested_tools(tmp_path, monkeypatch):
    project = tmp_path / 'analysis'
    plate = project / 'plate1'
    database = _measurements(plate / 'measurements' / 'measurements.db')
    with sqlite3.connect(plate / 'artifacts.db') as connection:
        connection.execute('CREATE TABLE logs (message TEXT)')
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No pixel decode')))
    result = home._easy_inspect_path(str(project))
    actions = {action[0]: action[3] for action in result['actions']}
    expected = {'db_browser', 'graph_builder', 'gate_editor', 'umap', 'regression',
                'dose_response', 'qc_dashboard', 'run_compare', 'annotate', 'classify_merged'}
    assert expected <= actions.keys()
    assert sum(action[0] == 'regression' for action in result['actions']) == 1
    for key in expected:
        assert result['context']['analysis_inputs'][key]['database'] == str(database)
    assert actions['db_browser'] == str(database)
    assert actions['qc_dashboard'] == str(plate)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_numeric_csv_and_completed_sequencing_are_distinct_from_raw_reads(tmp_path):
    table = tmp_path / 'values.csv'
    table.write_text('condition,area\ncontrol,42\ndrug,11\n')
    actions = home._easy_inspect_path(str(table))['actions']
    assert {'foreign', 'graph_builder', 'gate_editor'} <= {row[0] for row in actions}
    assert all(row[3] == str(table) for row in actions)
    counts = tmp_path / 'unique_combinations.csv'
    counts.write_text('plateID,rowID,columnID,grna,count\np1,A,1,g1,32\n')
    result = home._easy_inspect_path(str(counts))
    assert any(row[0] == 'regression' and row[3] == str(counts) for row in result['actions'])
    assert not any(row[0] == 'map_barcodes' for row in result['actions'])
    folder = tmp_path / 'completed'
    folder.mkdir()
    (folder / 'qc.csv').write_text('sample,reads\ns1,64\n')
    (folder / 'unique_combinations.csv').write_text(counts.read_text())
    assert any(row[0] == 'regression' for row in home._easy_inspect_path(str(folder))['actions'])


def test_plain_and_gzipped_fastq_content_routes_to_map_barcodes(tmp_path):
    for name, data in [('reads.fastq', b'@read1\nACGTN\n+\nIIIII\n'),
                       ('reads.fastq.gz', gzip.compress(b'@read2\nCGTAA\n+\nIIIII\n'))]:
        path = tmp_path / name
        path.write_bytes(data)
        result = home._easy_inspect_path(str(path))
        assert any(row[0] == 'map_barcodes' and row[3] == str(tmp_path) for row in result['actions'])
        assert path.read_bytes() == data
    folder_result = home._easy_inspect_path(str(tmp_path))
    assert any(row[0] == 'map_barcodes' for row in folder_result['actions'])


def test_text_csv_does_not_claim_numeric_plot_support(tmp_path):
    path = tmp_path / 'notes.csv'
    path.write_text('name,description\none,hello\ntwo,world\n')
    assert not {'graph_builder', 'gate_editor'} & {row[0] for row in home._easy_inspect_path(str(path))['actions']}


def test_multidimensional_numeric_array_is_probed_without_loading_pixels(tmp_path, monkeypatch):
    path = tmp_path / 'stack.npy'
    np.save(path, np.zeros((2, 3, 4, 5, 6), dtype=np.uint16))
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No pixel decode')))
    assert home._easy_file_kind(path) == 'array'
    result = home._easy_inspect_path(str(path))
    assert any(row[0] == 'mask' for row in result['actions'])
    assert not any(row[0] == 'make_masks' for row in result['actions'])


def test_saved_settings_are_parsed_without_application_and_keep_source(tmp_path):
    import json

    path = tmp_path / 'gen_mask_settings.json'
    values = {'src': '/original/source', 'channels': [0, 1], 'timelapse': False}
    path.write_text(json.dumps(values))
    result = home._easy_inspect_path(str(path))
    assert result['actions'][0][0] == 'mask'
    assert result['actions'][0][3] == str(path)
    assert result['context']['saved_settings'] == dict(path=str(path), module='mask', settings=values)
    assert result['context']['action_groups']['mask']['kind'] == 'settings'
    assert json.loads(path.read_text()) == values
    path = tmp_path / 'measure_crop_settings.csv'
    path.write_text('Key,Value\nsrc,/original/source\nchannels,"[0, 1]"\n')
    result = home._easy_inspect_path(str(path))
    assert result['context']['saved_settings']['module'] == 'measure'
    assert result['context']['saved_settings']['settings']['channels'] == [0, 1]
    assert not {'graph_builder', 'gate_editor', 'foreign'} & {row[0] for row in result['actions']}


def test_loose_pairs_do_not_claim_scientific_layout(tmp_path):
    from PIL import Image

    folder = tmp_path / 'loose'
    folder.mkdir()
    for name in ('field.tif', 'field_mask.tif'):
        Image.fromarray(np.ones((8, 9), dtype=np.uint16)).save(folder / name)
    result = home._easy_inspect_path(str(folder))
    assert result['context']['paired_layout']['kind'] == 'unstructured'
    assert result['context']['paired_layout']['pairs'] == {str(folder / 'field.tif'): str(folder / 'field_mask.tif')}
    assert {'make_masks', 'convert'} <= {row[0] for row in result['actions']}
    assert not {'mask', 'measure', 'qc_dashboard', 'annotate'} & {row[0] for row in result['actions']}
    (folder / 'merged').mkdir()
    np.save(folder / 'merged' / 'field.npy', np.zeros((8, 9, 3), dtype=np.uint16))
    result = home._easy_inspect_path(str(folder))
    assert result['context']['paired_layout']['kind'] == 'spacr'
    assert {'measure', 'qc_dashboard', 'annotate'} <= {row[0] for row in result['actions']}


def test_multiple_plates_keep_separate_database_identities(tmp_path):
    root = tmp_path / 'experiment'
    paths = [_measurements(root / name / 'measurements' / 'measurements.db') for name in ('plateA', 'plateB')]
    result = home._easy_inspect_path(str(root))
    assert {item['database'] for item in result['context']['plate_inputs']} == {str(path) for path in paths}
    assert len({item['identity'] for item in result['context']['plate_inputs']}) == 2


def test_explicit_header_axes_support_time_and_volume_without_pixel_reads(tmp_path, monkeypatch):
    import tifffile

    path = tmp_path / 'timed-volume.tif'
    tifffile.imwrite(path, np.zeros((2, 3, 8, 9), dtype=np.uint16), metadata={'axes': 'TZYX'})
    monkeypatch.setattr(tifffile.TiffPage, 'asarray', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No pixels')))
    result = home._easy_inspect_path(str(path))
    record = next(item for item in result['context']['image_dimensions'] if item['path'] == str(path))
    assert record['t'] == 2 and record['z'] == 3 and record['axes'] == 'TZYX'
    assert record['axes_unresolved'] is False
    actions = {row[0]: row[3] for row in result['actions']}
    assert actions['3d'] == actions['timelapse'] == str(path)
    assert result['context']['dimension_routes']['3d'] == dict(source=str(path), host='mask', dimension='z', size=3)
    assert result['context']['dimension_routes']['timelapse'] == dict(source=str(path), host='mask', dimension='t', size=2)


def test_regression_coefficients_and_failed_run_have_real_review_routes(tmp_path):
    import json

    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'results.csv').write_text('feature,coefficient\ngene_a,0.5\n')
    (folder / 'regression_data.csv').write_text('gene_a,response\n1,0.2\n')
    (folder / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'app_key': 'regression', 'status': 'failed'}))
    result = home._easy_inspect_path(str(folder))
    assert not any(row[0] == 'regression_diagnostics' for row in result['actions'])
    diagnostics = folder / 'diagnostics'
    diagnostics.mkdir()
    (diagnostics / 'residuals.csv').write_text('observed,predicted\n1,0.8\n')
    result = home._easy_inspect_path(str(folder))
    actions = {row[0]: row[3] for row in result['actions']}
    assert {'hit_list', 'profiler', 'regression_diagnostics', 'run_history', 'qc_dashboard'} <= actions.keys()
    assert actions['profiler'] == str(folder / 'results.csv')
    assert actions['hit_list'] == str(folder)
    assert actions['regression_diagnostics'] == str(diagnostics)
    assert result['context']['analysis_inputs']['regression_diagnostics']['diagnostics_folder'] == str(diagnostics)
    assert result['context']['run_evidence']['manifests'][0]['status'] == 'failed'


def test_run_settings_manifest_reuses_registered_module_without_importing_app(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    folder = tmp_path / 'saved-run'
    folder.mkdir()
    monkeypatch.setitem(sys.modules, 'spacr.qt.app', SimpleNamespace(APP_META={'regression': {}}))
    (folder / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'app_key': 'regression', 'status': 'success'}))
    settings = folder / 'settings.json'
    settings.write_text(json.dumps({'src': '/preserved/input', 'regression_type': 'ols'}))
    result = home._easy_inspect_path(str(settings))
    assert result['context']['saved_settings']['module'] == 'regression'
    assert result['actions'][0][0] == 'regression'
    assert result['actions'][0][3] == str(settings)


def test_unsafe_array_and_manifest_are_guidance_without_executing_or_allocating(tmp_path):
    import json

    path = tmp_path / 'truncated.npy'
    np.save(path, np.zeros((8, 9), dtype=np.uint16))
    path.write_bytes(path.read_bytes()[:130])
    assert home._easy_file_kind(path) == 'unsupported'
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'app_key': ['invalid'], 'status': 'failed'}))
    result = home._easy_inspect_path(str(tmp_path))
    assert not any(row[0] == 'run_history' for row in result['actions'])


def test_feature_only_database_does_not_claim_image_annotation_inputs(tmp_path):
    database = _measurements(tmp_path / 'plate' / 'measurements' / 'measurements.db')
    with sqlite3.connect(database) as connection:
        connection.execute('DROP TABLE png_list')
    result = home._easy_inspect_path(str(database))
    keys = {row[0] for row in result['actions']}
    assert {'db_browser', 'graph_builder', 'gate_editor', 'regression'} <= keys
    assert not {'annotate', 'umap'} & keys


def test_safe_2d_array_has_actual_import_but_higher_rank_keeps_native_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No pixel decode')))
    for rank in (2, 3):
        path = tmp_path / ('array' + str(rank) + '.npy')
        np.save(path, np.zeros((8, 9) if rank == 2 else (8, 9, 3), dtype=np.uint16))
        result = home._easy_inspect_path(str(path))
        keys = {row[0] for row in result['actions']}
        assert 'mask' in keys and 'make_masks' not in keys
        assert ('convert' in keys) == (rank == 2)
        record = next(item for item in result['context']['array_imports'] if item['path'] == str(path))
        assert record['supported'] == (rank == 2)
        assert result['context']['native_mask_container'] == str(path)


def test_explicit_numeric_csv_keeps_source_beside_existing_project_database(tmp_path):
    database = _measurements(tmp_path / 'plate' / 'measurements' / 'measurements.db')
    table = database.parent.parent / 'custom.csv'
    table.write_text('condition,area\ncontrol,7\ntreated,5\n')
    result = home._easy_inspect_path(str(table))
    for key in ('foreign', 'graph_builder', 'gate_editor'):
        assert next(row[3] for row in result['actions'] if row[0] == key) == str(table)
    for key in ('graph_builder', 'gate_editor'):
        assert result['context']['analysis_inputs'][key]['source'] == str(table)
        assert 'database' not in result['context']['analysis_inputs'][key]


def test_registered_pillow_rasters_offer_real_import_without_false_editor_route(tmp_path, monkeypatch):
    from PIL import Image
    from spacr.convert import _image_extensions

    for suffix in ('.gif', '.webp'):
        if suffix not in _image_extensions():
            continue
        folder = tmp_path / suffix[1:]
        folder.mkdir()
        path = folder / ('raster' + suffix)
        Image.fromarray(np.ones((8, 9, 3), dtype=np.uint8) * 40).save(path)
        before = path.read_bytes()
        with monkeypatch.context() as patch:
            patch.setattr(Image.Image, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No pixels')))
            result = home._easy_inspect_path(str(path))
        assert home._easy_file_kind(path) == 'import_image'
        assert any(row[0] == 'convert' and row[3] == str(folder) for row in result['actions'])
        assert not any(row[0] in ('make_masks', 'mask') for row in result['actions'])
        assert path.read_bytes() == before


def test_ops_structure_seeds_only_verified_sources_and_library(tmp_path):
    from PIL import Image

    root = tmp_path / 'screen'
    genotype, phenotype = root / 'genotype', root / 'phenotype'
    genotype.mkdir(parents=True)
    phenotype.mkdir()
    for cycle in (1, 2):
        Image.fromarray(np.ones((8, 9), dtype=np.uint16)).save(genotype / f'10X_c{cycle}_A1_CY3_Site-0.tif')
    Image.fromarray(np.ones((8, 9), dtype=np.uint16)).save(phenotype / '20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-0.tif')
    library = root / 'library.csv'
    library.write_text('prefix,gene\nACGT,gene_a\n')
    result = home._easy_inspect_path(str(root))
    hint = result['context']['submodule_settings']['ops']
    assert hint['settings'] == dict(genotype_source=str(genotype), phenotype_source=str(phenotype), ops_library=str(library))
    assert next(row[3] for row in result['actions'] if row[0] == 'ops') == str(genotype)
    assert 'ops_base_channels' not in hint['settings']


def test_spatial_masks_and_mapping_require_actual_source_and_merged_provenance(tmp_path):
    import json
    from PIL import Image

    root = tmp_path / 'xenium'
    focus = root / 'morphology_focus'
    focus.mkdir(parents=True)
    (root / 'experiment.xenium').write_text(json.dumps({'pixel_size': .2125}))
    (root / 'transcripts.parquet').write_bytes(b'PAR1bounded structural fixturePAR1')
    Image.fromarray(np.ones((8, 9), dtype=np.uint16)).save(focus / 'morphology_focus_0000.ome.tif')
    result = home._easy_inspect_path(str(root))
    assert result['context']['submodule_settings']['spatial_masks']['settings'] == dict(spatial_source=str(root), spatial_platform='xenium')
    assert 'spatial_map' not in result['context']['submodule_settings']
    project = root / 'segmented'
    database = _measurements(project / 'measurements' / 'measurements.db')
    (project / 'merged').mkdir()
    np.save(project / 'merged' / 'field.npy', np.zeros((8, 9, 3), dtype=np.uint16))
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE spatial_fields (source TEXT, channels INTEGER)')
        connection.execute('INSERT INTO spatial_fields VALUES (?, 1)', (str(root),))
    result = home._easy_inspect_path(str(root))
    settings = result['context']['submodule_settings']['spatial_map']['settings']
    assert settings == dict(spatial_source=str(root), spatial_platform='xenium', spatial_masks_dir=str(project))
    with sqlite3.connect(database) as connection:
        connection.execute('UPDATE spatial_fields SET source = ?', ('/other/acquisition',))
    result = home._easy_inspect_path(str(root))
    assert 'spatial_map' not in result['context']['submodule_settings']


def test_visium_hd_needs_physical_scale_and_uses_only_an_existing_bin(tmp_path):
    import json
    from PIL import Image

    root = tmp_path / 'visium'
    spatial = root / 'binned_outputs' / 'square_016um' / 'spatial'
    spatial.mkdir(parents=True)
    Image.fromarray(np.ones((8, 9), dtype=np.uint16)).save(root / 'slide.tif')
    assert 'spatial_masks' not in home._easy_inspect_path(str(root))['context']['submodule_settings']
    (spatial / 'scalefactors_json.json').write_text(json.dumps({'microns_per_pixel': .5}))
    result = home._easy_inspect_path(str(root))
    settings = result['context']['submodule_settings']['spatial_masks']['settings']
    assert settings == dict(spatial_source=str(root), spatial_platform='visium_hd', spatial_bin_um=16)


def test_pdf_header_offers_actual_plaque_assay_without_processing(tmp_path):
    path = tmp_path / 'plate.pdf'
    raw = b'%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF\n'
    path.write_bytes(raw)
    result = home._easy_inspect_path(str(path))
    assert next(row[3] for row in result['actions'] if row[0] == 'analyze_plaques') == str(path)
    assert result['context']['analysis_inputs']['analyze_plaques']['pdf'] == str(path)
    assert path.read_bytes() == raw


def test_requested_external_document_families_are_guidance_only(tmp_path):
    families = {'flowjo': ('.wsp', '.wst', '.acs', '.fcs'), 'prism': ('.pzf', '.pzfx', '.prism'),
                'chemdraw': ('.cdx', '.cdxml'), 'structure': ('.pdb', '.cif', '.mmcif', '.pdb.gz', '.mmcif.gz')}
    for family, extensions in families.items():
        for extension in extensions:
            path = tmp_path / ('document' + extension)
            path.write_bytes(b'original data preserved')
            result = home._easy_inspect_path(str(path))
            assert result['context']['unsupported_external'] == family
            assert result['actions'] == []
            assert path.read_bytes() == b'original data preserved'
            if family in ('chemdraw', 'structure'):
                assert 'numeric CSV' not in result['message']
    assert home._easy_external_format(tmp_path / 'microscopy.mrc', b'AlphaFold structure') is None
    assert home._easy_external_format(tmp_path / 'safe.npy', b'AlphaFold structure') is None
    assert home._easy_external_format(tmp_path / 'reference.txt', b'AlphaFold model') == 'structure'


def test_nonpath_external_references_are_bounded_guidance_without_reader_calls(tmp_path):
    for text, family in [('FlowJo file', 'flowjo'), ('GraphPad Prism project', 'prism'),
                         ('ChemDraw structure', 'chemdraw'), ('AlphaFold model', 'structure'),
                         ('PDB 1abc', 'structure'), (str(tmp_path / 'missing.pdb'), 'structure'),
                         ('FlowJo ' + 'reference ' * 500, 'flowjo')]:
        result = home._easy_inspect_path(text)
        assert result['actions'] == []
        assert result['context']['unsupported_external'] == family
    assert 'unsupported_external' not in home._easy_inspect_path(str(tmp_path / 'AlphaFold.npy')).get('context', {})
