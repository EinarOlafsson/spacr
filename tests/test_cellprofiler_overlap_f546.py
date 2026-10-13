"""F546: concave objects match by actual same-field pixels, without guessing."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr import _segmentation_backends as backend


def project_reply(tmp_path, cp, spacr, *, name='Cells', centres=True):
    """Build a measured ring fixture whose centre is on background."""
    merged = tmp_path / 'merged'
    merged.mkdir()
    stem = 'plate1_A01_1'
    np.save(merged / (stem + '.npy'), spacr[..., None])
    cp_path = tmp_path / 'labels.npy'
    np.save(cp_path, cp)
    columns = ['ImageNumber', 'ObjectNumber', 'AreaShape_Area']
    rows = [[1, 1, 16]]
    if centres:
        columns += ['Location_Center_X', 'Location_Center_Y']
        rows[0] += [3, 3]
    table = tmp_path / 'objects.npy'
    np.save(table, np.asarray(rows, float))
    reply = {'images': {'1': [stem + '_ch0.tif']},
             'objects': {name: {'columns': columns, 'path': str(table)}},
             'labels': {'1': {name: [str(cp_path)]}}}
    return merged, reply, {'cell_mask_dim': 0}


def ring():
    """Return a concave annulus with an empty centroid pixel."""
    image = np.zeros((7, 7), np.uint16)
    image[1:6, 1:6] = 1
    image[2:5, 2:5] = 0
    return image


@pytest.mark.parametrize('centres', [True, False])
@pytest.mark.parametrize('name', ['Cells', 'Speckles'])
def test_ring_maps_without_a_centroid_hit_and_preserves_sources(tmp_path, centres, name):
    cp = ring()
    merged, reply, settings = project_reply(tmp_path, cp, cp * 17,
                                            name=name, centres=centres)
    sources = {p: p.read_bytes() for p in tmp_path.rglob('*.npy')}
    frame = measure._cellprofiler_tables(reply, merged, settings)[
        'cellprofiler_' + name.lower()]
    assert frame['object_label'].tolist() == [17]
    assert frame['prcfo'].tolist() == ['plate1_r1_c1_f1_o17']
    assert frame['cp_AreaShape_Area'].tolist() == [16]
    assert frame['cp_object_number'].tolist() == [1]
    assert all(p.read_bytes() == data for p, data in sources.items())


@pytest.mark.parametrize('case', ['tie', 'mostly_background', 'crop', 'float',
                                  'negative', 'too_large', 'missing', 'empty',
                                  'duplicated_planes', 'overflow'])
def test_invalid_or_ambiguous_supplied_masks_never_fall_back_to_centres(tmp_path, case):
    cp = np.ones((7, 7), np.uint16)
    target = np.full((7, 7), 17, np.uint16)
    merged, reply, settings = project_reply(tmp_path, cp, target)
    label_path = Path(reply['labels']['1']['Cells'][0])
    if case == 'tie':
        cp[:] = 0
        cp[2, 2:4] = 1
        target[2, 2] = 18
        np.save(merged / 'plate1_A01_1.npy', target[..., None])
    elif case == 'mostly_background':
        target[:] = 0
        target[3, 3] = 17
        np.save(merged / 'plate1_A01_1.npy', target[..., None])
    elif case == 'crop':
        cp = cp[:3, :3]
    elif case == 'float':
        cp = cp.astype(float)
    elif case == 'negative':
        cp = -cp.astype(np.int16)
    elif case == 'overflow':
        cp = np.full_like(cp, 2**63, dtype=np.uint64)
    elif case == 'too_large':
        reply['labels']['1']['Cells'] *= 65
    elif case == 'empty':
        reply['labels']['1']['Cells'] = []
    elif case == 'duplicated_planes':
        reply['labels']['1']['Cells'] *= 2
    np.save(label_path, cp)
    if case == 'missing':
        label_path.unlink()
    frame = measure._cellprofiler_tables(reply, merged, settings)['cellprofiler_cells']
    assert len(frame) == 1
    assert pd.isna(frame['object_label'].iat[0])
    assert pd.isna(frame['prcfo'].iat[0])
    assert frame['cp_AreaShape_Area'].iat[0] == 16


def test_legacy_replies_keep_centre_matching(tmp_path):
    cp = ring()
    target = np.zeros_like(cp)
    target[3, 3] = 31
    merged, reply, settings = project_reply(tmp_path, cp, target)
    reply.pop('labels')
    frame = measure._cellprofiler_tables(reply, merged, settings)['cellprofiler_cells']
    assert frame['object_label'].tolist() == [31]


def test_overlapping_cp_objects_on_separate_planes_keep_their_ids(tmp_path):
    first = np.array([[1, 1, 0], [0, 0, 0]], np.uint16)
    second = np.array([[0, 2, 2], [0, 0, 0]], np.uint16)
    paths = []
    for i, plane in enumerate((first, second)):
        path = tmp_path / f'{i}.npy'
        np.save(path, plane)
        paths.append(str(path))
    labels = measure._cellprofiler_overlap_labels(paths, np.full((2, 3), 9))
    assert labels == {1: 9, 2: 9}


def test_multiple_image_numbers_never_reuse_another_fields_labels(tmp_path):
    cp = ring()
    merged, reply, settings = project_reply(tmp_path, cp, cp * 17)
    second = 'plate1_A02_1'
    np.save(merged / (second + '.npy'), (cp * 99)[..., None])
    reply['images']['2'] = [second + '_ch0.tif']
    reply['labels']['2'] = reply['labels']['1']
    path = reply['objects']['Cells']['path']
    rows = np.load(path)
    rows = np.vstack((rows, rows))
    rows[1, 0] = 2
    np.save(path, rows)
    frame = measure._cellprofiler_tables(reply, merged, settings)['cellprofiler_cells']
    assert frame['object_label'].tolist() == [17, 99]
    assert frame['prcfo'].tolist() == ['plate1_r1_c1_f1_o17', 'plate1_r1_c2_f1_o99']


class ObjectSet:
    """A minimal pinned CellProfiler object-set interface."""
    def __init__(self, objects):
        """Store named CP-compatible objects."""
        self.objects = objects
        self.object_names = list(objects)

    def get_objects(self, name):
        """Return an existing object without copying labels."""
        return self.objects[name]


class Objects:
    """Label planes exposed by the pinned CellProfiler Objects API."""
    def __init__(self, planes, cropped=False):
        """Keep original arrays and optional crop metadata."""
        self.planes = planes
        self.parent_image = SimpleNamespace(has_crop_mask=cropped)

    def get_labels(self):
        """Return planes paired with their CP object indices."""
        return [(plane, np.unique(plane)) for plane in self.planes]


def workspace(objects, number=1):
    """Expose only the properties read by the capture adapter."""
    return SimpleNamespace(object_set=ObjectSet(objects),
                           measurements=SimpleNamespace(image_set_number=number))


def test_capture_writes_original_planes_and_marks_crops_unusable(tmp_path):
    cp = ring()
    capture = backend._cellprofiler_capture_labels(workspace({
        'Cells': Objects([cp]), 'Cropped': Objects([cp], cropped=True)}), str(tmp_path))
    np.testing.assert_array_equal(np.load(capture['Cells'][0]), cp)
    assert capture['Cropped'] == []
    np.testing.assert_array_equal(cp, ring())


def test_capture_bounds_number_of_planes_and_does_not_leave_partial_object(tmp_path):
    cp = ring()
    capture = backend._cellprofiler_capture_labels(workspace({
        'TooMany': Objects([cp] * 65),
        'Volumetric': Objects([cp, np.zeros((2, 7, 7), np.uint16)])}), str(tmp_path))
    assert capture == {'TooMany': [], 'Volumetric': []}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('fails', [False, True])
def test_pipeline_hook_runs_modules_unchanged_and_restores_itself(tmp_path, fails):
    calls = []
    first, last = object(), object()
    ws = workspace({'Cells': Objects([ring()])})

    class Pipeline:
        """Exercise the same run_module(module, workspace) call as CP 4.2.8.1."""
        def modules(self):
            """Return the enabled modules in execution order."""
            return [first, last]

        def run_module(self, module, current):
            """Simulate an original module's successful or failed execution."""
            calls.append(module)
            if fails and module is last:
                raise RuntimeError('original failure')

        def run(self):
            """Execute modules on a field through the instance adapter."""
            for module in self.modules():
                self.run_module(module, ws)
            return 'measurements'

    pipeline = Pipeline()
    original = pipeline.run_module
    if fails:
        with pytest.raises(RuntimeError, match='original failure'):
            backend._cellprofiler_run_with_labels(pipeline, str(tmp_path))
        assert not list(tmp_path.iterdir())
    else:
        measured, labels = backend._cellprofiler_run_with_labels(pipeline, str(tmp_path))
        assert measured == 'measurements'
        assert list(labels) == ['1']
        assert len(labels['1']['Cells']) == 1
    assert calls == [first, last]
    assert pipeline.run_module == original


def test_unnamed_object_with_equally_good_roles_is_unmatched(tmp_path):
    cp = ring()
    merged, reply, settings = project_reply(tmp_path, cp, cp * 17, name='Speckles')
    np.save(merged / 'plate1_A01_1.npy', np.stack((cp * 17, cp * 18), axis=-1))
    settings['nucleus_mask_dim'] = 1
    frame = measure._cellprofiler_tables(reply, merged, settings)['cellprofiler_speckles']
    assert frame['object_label'].isna().all()
    assert frame['object_type'].isna().all()


def test_image_set_containing_multiple_field_names_cannot_borrow_one(tmp_path):
    cp = ring()
    merged, reply, settings = project_reply(tmp_path, cp, cp * 17)
    reply['images']['1'].append('plate1_A02_1_ch1.tif')
    frame = measure._cellprofiler_tables(reply, merged, settings)['cellprofiler_cells']
    assert frame['object_label'].isna().all()
    assert frame['prcfo'].isna().all()


def test_worker_capture_reaches_real_sqlite_without_changing_pipeline_or_pixels(
        tmp_path, monkeypatch):
    """Run the worker adapter with CP stand-ins, then the real Measure import."""
    import sys
    import types
    from spacr.tabular import read_table

    cp = ring()
    merged = tmp_path / 'merged'
    merged.mkdir()
    source = merged / 'plate1_A01_1.npy'
    np.save(source, np.stack((cp * 100, cp * 17), axis=-1))
    source_bytes = source.read_bytes()
    pipeline_path = tmp_path / 'example.cppipe'
    pipeline_path.write_text('original pipeline')
    database = tmp_path / 'measurements' / 'measurements.db'
    database.parent.mkdir()

    class Measurements:
        """Expose the pinned worker's measurement-reading protocol."""
        image_set_number = 1

        def has_feature(self, name, feature):
            """No failure status was recorded by this successful fixture."""
            return False

        def get_image_numbers(self):
            """The single completed field number."""
            return [1]

        def get_object_names(self):
            """Names measured by the unchanged pipeline."""
            return ['Image', 'Cells']

        def get_feature_names(self, name):
            """Return file identity or numeric object feature columns."""
            return (['FileName_DNA'] if name == 'Image' else
                    ['Number_Object_Number', 'AreaShape_Area'])

        def get_measurement(self, name, feature, number):
            """Return original object IDs and a measurement with no centroid."""
            assert number == 1
            return {'FileName_DNA': 'plate1_A01_1_ch0.tif',
                    'Number_Object_Number': np.array([1]),
                    'AreaShape_Area': np.array([16])}[feature]

    measured = Measurements()
    final_module = object()

    class Pipeline:
        """Run one original module through CellProfiler's existing hook."""
        def load(self, path):
            """Read the same user pipeline without rewriting it."""
            assert Path(path).read_text() == 'original pipeline'

        def add_pathnames_to_file_list(self, files):
            """Accept the real TIFFs exported by Measure."""
            assert files and all(Path(path).is_file() for path in files)

        def modules(self):
            """Return the enabled module list."""
            return [final_module]

        def run_module(self, module, current):
            """Populate the unchanged final workspace."""
            assert module is final_module

        def run(self):
            """Complete the field with its object labels available."""
            current = workspace({'Cells': Objects([cp])})
            current.measurements = measured
            self.run_module(final_module, current)
            return measured

    module = types.ModuleType('cellprofiler_core.pipeline')
    module.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, 'cellprofiler_core.pipeline', module)
    monkeypatch.setattr(backend, '_cellprofiler_started', lambda adapters:
        SimpleNamespace(set_default_output_directory=lambda path: None,
                        set_default_image_directory=lambda path: None))

    def runner(pipeline, files, output):
        """Use the actual worker serialization, without importing the CP runtime."""
        return backend._worker_run_cellprofiler(
            {'pipeline': pipeline, 'files': files, 'output': output}, {})

    counts = measure._run_cellprofiler_step(str(database), {
        'src': str(merged), 'cell_mask_dim': 1,
        'cellprofiler_pipeline': str(pipeline_path)}, runner=runner)
    assert counts == {'cellprofiler_cells': 1}
    frame = read_table(str(database), table='cellprofiler_cells', canonicalise=False)
    assert frame['prcfo'].tolist() == ['plate1_r1_c1_f1_o17']
    assert frame['cp_AreaShape_Area'].tolist() == [16]
    assert pipeline_path.read_text() == 'original pipeline'
    assert source.read_bytes() == source_bytes
