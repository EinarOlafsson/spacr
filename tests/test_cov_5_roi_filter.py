"""The drawn region, from the refusals through to the worker environment.

Everything here is about the failure an ROI makes possible: the region still
rasterises, Measure still finishes, and the numbers are for the wrong pixels.
So each refusal is asserted by its message, and each keep/drop decision is
asserted against a mask whose objects are placed on known sides of the line.
"""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import Canvas, Shape, ShapesLayer, Spacing


@pytest.fixture(autouse=True)
def clean_hooks():
    """Leave the hook registry and the ROI environment as they were found."""
    saved = {name: os.environ.get(name)
             for name in (mh.HOOKS_ENV_VAR, R.ROI_ENV_VAR,
                          R.ON_MISSING_ENV_VAR)}
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _rect(y0=0.0, x0=0.0, y1=9.0, x1=9.0, name='r'):
    return R.RegionOfInterest('rectangle', [[y0, x0], [y1, x1]], name=name)


def _two_object_mask():
    """A 20x20 mask: label 1 inside the top-left ROI, label 2 far outside."""
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:6, 2:6] = 1
    mask[14:18, 14:18] = 2
    return mask


def _context(mask, *, file_name='plate1_A01_F001', object_type='cell',
             settings=None, spacing=None):
    return mh.RegionContext(object_type=object_type, file_name=file_name,
                            mask=mask, settings=settings or {},
                            spacing=spacing)


# ---------------------------------------------------------------------------
# Geometry the drawing tool can hand over, and what each refusal protects
# ---------------------------------------------------------------------------

def test_an_open_shape_is_not_a_region():
    """A line names no inside, so it cannot decide which objects to keep."""
    with pytest.raises(R.RoiError, match='closed shape'):
        R.RegionOfInterest('line', [[0.0, 0.0], [4.0, 4.0]])


def test_vertices_must_be_a_table_of_points():
    """A flat list of numbers is an ambiguous pairing, not a set of points."""
    with pytest.raises(R.RoiError, match=r'\(M, 2\) array'):
        R.RegionOfInterest('polygon', [0.0, 1.0, 2.0, 3.0])


def test_a_two_point_polygon_encloses_nothing():
    """Two points are a segment; a polygon needs a third to have an area."""
    with pytest.raises(R.RoiError, match='at least three points'):
        R.RegionOfInterest('polygon', [[0.0, 0.0], [4.0, 4.0]])


def test_a_nan_vertex_is_refused():
    """A NaN corner rasterises to a region nobody drew."""
    with pytest.raises(R.RoiError, match='finite'):
        R.RegionOfInterest('rectangle', [[0.0, 0.0], [np.nan, 4.0]])


def test_to_shape_undoes_the_layers_own_offset_and_scale():
    """World vertices become data indices on the grid they will rasterise on."""
    roi = R.RegionOfInterest('rectangle', [[10.0, 20.0], [20.0, 40.0]],
                             name='colony')
    spacing = Spacing(scale=(2.0, 2.0), translate=(10.0, 10.0),
                      axes=('y', 'x'), units='um')
    shape = roi.to_shape(spacing)
    assert isinstance(shape, Shape)
    assert shape.name == 'colony'
    # (world - translate) / scale, and a rectangle is stored as four corners.
    assert shape.data.min(axis=0).tolist() == [0.0, 5.0]
    assert shape.data.max(axis=0).tolist() == [5.0, 15.0]


def test_an_roi_entry_without_vertices_names_the_missing_key():
    """A hand-edited file loses a key; the message says which one."""
    with pytest.raises(R.RoiError, match="'vertices'"):
        R.RegionOfInterest.from_dict({'kind': 'rectangle', 'name': 'r'})


# ---------------------------------------------------------------------------
# The set: what it refuses to be built from
# ---------------------------------------------------------------------------

def test_a_field_holding_a_raw_polygon_is_refused():
    """A list of points is not an ROI: it carries no kind and no units."""
    with pytest.raises(R.RoiError, match='holds list, not a RegionOfInterest'):
        R.RoiSet(fields={'f1': [[[0.0, 0.0], [4.0, 4.0]]]})


def test_one_axis_is_not_a_plane():
    """A region lies in a plane, so its axes come in a distinct pair."""
    with pytest.raises(R.RoiError, match='exactly two different'):
        R.RoiSet(axes=('y', 'y'))


def test_the_reversed_axis_pair_is_refused_not_transposed():
    """(x, y) would draw the region on its side and still look like a region."""
    with pytest.raises(R.RoiError, match='vertex order reversed'):
        R.RoiSet(axes=('x', 'y'))


def test_an_unknown_rule_is_refused():
    with pytest.raises(R.RoiError, match='unknown ROI mode'):
        R.RoiSet(mode='touching')


def test_min_overlap_is_a_fraction():
    """Zero would keep every object; above one would keep none."""
    with pytest.raises(R.RoiError, match='must be'):
        R.RoiSet(mode='overlap', min_overlap=0.0)


def test_an_unknown_object_type_would_filter_nothing():
    with pytest.raises(R.RoiError, match='unknown object type'):
        R.RoiSet(object_types=('cel',))


def test_an_unknown_on_missing_rule_is_refused():
    with pytest.raises(R.RoiError, match='unknown on_missing'):
        R.RoiSet(on_missing='skip')


def test_roi_set_length_counts_regions_across_fields():
    """Length counts drawings, not the number of covered field names."""
    roi_set = R.RoiSet(fields={
        'plate1_A01_F001': (_rect(), _rect(name='second')),
        'plate1_A01_F002': (_rect(),),
    })

    assert len(roi_set) == 3


# ---------------------------------------------------------------------------
# Which ROIs apply to which field
# ---------------------------------------------------------------------------

def test_a_fields_own_entry_wins_over_the_default():
    own, fallback = _rect(name='own'), _rect(name='any')
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (own,),
                               R.ANY_FIELD: (fallback,)})
    assert roi_set.rois_for('plate1_A01_F001.npy') == (own,)
    assert roi_set.rois_for('plate1_A01_F002.npy') == (fallback,)
    assert roi_set.covers('plate1_A01_F002.npy') is True


def test_a_set_with_no_default_says_nothing_about_an_unnamed_field():
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (_rect(),)})
    assert roi_set.rois_for('plate1_A01_F009') is None
    assert roi_set.covers('plate1_A01_F009') is False


def test_a_field_is_filed_under_its_stem_not_its_path():
    """The pipeline hands the filter a path; the ROI was filed by stem."""
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (_rect(),)})
    assert roi_set.covers('/data/masks/plate1_A01_F001.npy') is True


# ---------------------------------------------------------------------------
# Taking the drawing off a layer
# ---------------------------------------------------------------------------

def _shapes_layer():
    spacing = Spacing(scale=(1.0, 1.0), axes=('y', 'x'), units='px')
    return ShapesLayer([Shape('line', [[0.0, 0.0], [9.0, 9.0]], name='ruler'),
                        Shape('rectangle', [[0.0, 0.0], [9.0, 9.0]],
                              name='colony')],
                       name='drawn', spacing=spacing)


def test_an_open_shape_on_the_layer_is_skipped_not_refused():
    """A measuring line drawn beside the region is not part of the region."""
    roi_set = R.RoiSet.from_shapes_layer(_shapes_layer(), fields='plate1_A01_F001')
    rois = roi_set.rois_for('plate1_A01_F001')
    assert [roi.name for roi in rois] == ['colony']
    assert roi_set.units == 'px'


def test_a_layer_of_only_lines_has_no_inside():
    spacing = Spacing(scale=(1.0, 1.0), axes=('y', 'x'), units='px')
    layer = ShapesLayer([Shape('path', [[0.0, 0.0], [4.0, 4.0], [8.0, 1.0]])],
                        name='sketch', spacing=spacing)
    with pytest.raises(R.RoiError, match='no closed shape'):
        R.RoiSet.from_shapes_layer(layer)


def test_axes_the_layer_does_not_have_are_refused():
    spacing = Spacing(scale=(1.0, 1.0), axes=('y', 'x'), units='px')
    layer = ShapesLayer([Shape('rectangle', [[0.0, 0.0], [4.0, 4.0]])],
                        name='drawn', spacing=spacing)
    with pytest.raises(R.RoiError, match='do not include'):
        R.RoiSet.from_shapes_layer(layer, axes=('z', 'y'))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_a_fields_entry_that_is_not_a_mapping_is_refused(tmp_path):
    with pytest.raises(R.RoiError, match="'fields' entry must map"):
        R.RoiSet.from_dict({'fields': ['plate1_A01_F001']})


def test_a_missing_roi_file_names_the_variable_that_should_point_at_it(tmp_path):
    missing = str(tmp_path / 'nowhere' / 'roi.json')
    with pytest.raises(R.RoiError, match=R.ROI_ENV_VAR):
        R.RoiSet.load(missing)


def test_a_truncated_roi_file_is_refused(tmp_path):
    path = tmp_path / 'roi.json'
    path.write_text('{"fields": ', encoding='utf-8')
    with pytest.raises(R.RoiError, match='could not be read'):
        R.RoiSet.load(str(path))


def test_a_json_list_is_not_an_roi_file(tmp_path):
    path = tmp_path / 'roi.json'
    path.write_text('[1, 2, 3]', encoding='utf-8')
    with pytest.raises(R.RoiError, match='holds a list'):
        R.RoiSet.load(str(path))


def test_an_unwritable_destination_is_refused_here_not_in_a_worker(tmp_path):
    """A save that failed silently would leave the workers with no ROI."""
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory', encoding='utf-8')
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    with pytest.raises(R.RoiError, match='spawn'):
        roi_set.save(str(blocker / 'roi.json'))


def test_a_saved_set_round_trips(tmp_path):
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, mode='overlap',
                       min_overlap=0.25, invert=True,
                       object_types=('cell', 'nucleus'), on_missing='all')
    path = roi_set.save(str(tmp_path / 'sub' / 'roi.json'))
    back = R.RoiSet.load(path)
    assert back.mode == 'overlap' and back.min_overlap == 0.25
    assert back.invert is True and back.on_missing == 'all'
    assert back.object_types == ('cell', 'nucleus')
    assert json.loads(open(path, encoding='utf-8').read())['spacr_roi_version'] == 1


# ---------------------------------------------------------------------------
# The filter's own refusals
# ---------------------------------------------------------------------------

def test_the_filter_needs_a_set_not_a_polygon():
    with pytest.raises(R.RoiError, match='needs a RoiSet'):
        R.RoiRegionFilter(_rect())


def test_an_unknown_on_missing_override_is_refused():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    with pytest.raises(R.RoiError, match='unknown on_missing'):
        R.RoiRegionFilter(roi_set, on_missing='maybe')


# ---------------------------------------------------------------------------
# The keep/drop decision
# ---------------------------------------------------------------------------

def test_a_type_outside_object_types_is_waved_through():
    """An ROI drawn for cells must not silently cull the pathogens too."""
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, object_types=('cell',))
    flt = R.RoiRegionFilter(roi_set)
    context = _context(_two_object_mask(), object_type='pathogen')
    keep = flt(context)
    assert keep.tolist() == [True, True]
    # Waving a type through is not a decision, so it is not counted.
    assert flt.stats == {'kept': 0, 'dropped': 0, 'fields': 0}


def test_the_centroid_rule_keeps_the_object_inside_the_region():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    flt = R.RoiRegionFilter(roi_set)
    keep = flt(_context(_two_object_mask()))
    assert keep.tolist() == [True, False]
    assert flt.stats == {'kept': 1, 'dropped': 1, 'fields': 1}
    assert '1 object(s) kept' in flt.report()
    assert '1 dropped across 1 field(s)' in flt.report()


def test_inverting_measures_everything_outside_the_region():
    """"Exclude this debris" is the same drawing read the other way."""
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, invert=True)
    flt = R.RoiRegionFilter(roi_set)
    keep = flt(_context(_two_object_mask()))
    assert keep.tolist() == [False, True]


def test_the_raster_is_computed_once_for_all_five_object_types():
    """Five calls per field must not rasterise the polygon five times."""
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    flt = R.RoiRegionFilter(roi_set)
    mask = _two_object_mask()
    for object_type in ('cell', 'nucleus', 'pathogen'):
        flt(_context(mask, object_type=object_type))
    assert flt.stats['fields'] == 1
    # A different field is a different raster.
    flt(_context(mask, file_name='plate1_A01_F002'))
    assert flt.stats['fields'] == 2


def test_an_empty_field_is_neither_kept_nor_dropped():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    flt = R.RoiRegionFilter(roi_set)
    keep = flt(_context(np.zeros((20, 20), dtype=np.int32)))
    assert keep.tolist() == []
    assert flt.stats == {'kept': 0, 'dropped': 0, 'fields': 0}


def test_the_overlap_rule_judges_pixels_not_the_middle():
    """An object straddling the edge is kept only if enough of it is inside."""
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:10, 2:6] = 1          # four of its five rows fall inside
    mask[2:6, 2:6] = 2           # wholly inside
    strict = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, mode='overlap',
                      min_overlap=0.9)
    assert R.RoiRegionFilter(strict)(_context(mask)).tolist() == [False, True]
    lenient = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, mode='overlap',
                       min_overlap=0.5)
    assert R.RoiRegionFilter(lenient)(_context(mask)).tolist() == [True, True]


def test_a_two_dimensional_roi_names_a_column_through_a_stack():
    """The polygon was drawn looking down the stack, so it applies to every z."""
    mask = np.zeros((3, 20, 20), dtype=np.int32)
    mask[:, 2:6, 2:6] = 1
    mask[:, 14:18, 14:18] = 2
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, mode='overlap')
    keep = R.RoiRegionFilter(roi_set)(_context(mask))
    assert keep.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Fields the drawing says nothing about
# ---------------------------------------------------------------------------

def test_an_uncovered_field_is_refused_by_default():
    """Measuring the whole field because nothing was drawn is the silent answer."""
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (_rect(),)})
    flt = R.RoiRegionFilter(roi_set)
    with pytest.raises(R.RoiError, match='no ROI covers field'):
        flt(_context(_two_object_mask(), file_name='plate1_A01_F002'))


def test_on_missing_all_measures_the_uncovered_field_whole():
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (_rect(),)}, on_missing='all')
    flt = R.RoiRegionFilter(roi_set)
    keep = flt(_context(_two_object_mask(), file_name='plate1_A01_F002'))
    assert keep.tolist() == [True, True]
    assert flt.stats['kept'] == 2


def test_on_missing_none_skips_the_uncovered_field():
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (_rect(),)})
    flt = R.RoiRegionFilter(roi_set, on_missing='none')
    keep = flt(_context(_two_object_mask(), file_name='plate1_A01_F002'))
    assert keep.tolist() == [False, False]
    assert flt.stats['dropped'] == 2


# ---------------------------------------------------------------------------
# Units: the mismatch that draws a plausible region in the wrong place
# ---------------------------------------------------------------------------

def test_a_micron_roi_on_a_pixel_measurement_is_refused():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, units='um')
    flt = R.RoiRegionFilter(roi_set)
    with pytest.raises(R.RoiError, match="measured in 'um' but field"):
        flt(_context(_two_object_mask()))


def test_a_micron_roi_matches_a_run_that_set_the_voxel_size():
    """A 3-D run with voxel_size_xy_um is measured in µm, so a µm ROI fits."""
    mask = np.zeros((2, 20, 20), dtype=np.int32)
    mask[:, 2:6, 2:6] = 1
    mask[:, 14:18, 14:18] = 2
    # 0.5 µm per pixel: the same 10x10-pixel corner is 5 µm across.
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(y1=4.5, x1=4.5),)},
                       units='um')
    flt = R.RoiRegionFilter(roi_set)
    keep = flt(_context(mask, settings={'voxel_size_xy_um': 0.5},
                        spacing=(2.0, 0.5, 0.5)))
    assert keep.tolist() == [True, False]


def test_a_pixel_roi_is_refused_on_a_run_measured_in_microns():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}, units='px')
    flt = R.RoiRegionFilter(roi_set)
    with pytest.raises(R.RoiError, match="measured in 'px' but field"):
        flt(_context(np.zeros((2, 20, 20), dtype=np.int32) + np.eye(20, dtype=np.int32),
                     settings={'voxel_size_xy_um': 0.5},
                     spacing=(2.0, 0.5, 0.5)))


# ---------------------------------------------------------------------------
# Getting it to the workers
# ---------------------------------------------------------------------------

def test_install_without_a_saved_roi_refuses_rather_than_measuring_everything():
    os.environ.pop(R.ROI_ENV_VAR, None)
    with pytest.raises(R.RoiError, match='is not set'):
        R.install()


def test_install_reads_the_environment_and_registers_the_filter(tmp_path):
    path = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}).save(
        str(tmp_path / 'roi.json'))
    os.environ[R.ROI_ENV_VAR] = path
    os.environ[R.ON_MISSING_ENV_VAR] = 'none'
    assert R.install() == R.HOOK_NAME
    hooks = {entry.name: entry for entry in mh.region_filter_hooks()}
    assert R.HOOK_NAME in hooks
    assert hooks[R.HOOK_NAME].func.on_missing == 'none'


def test_enabling_from_a_path_loads_it_here_rather_than_in_a_worker(tmp_path, capsys):
    path = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}).save(
        str(tmp_path / 'roi.json'))
    assert R.enable_roi_filter(path) == R.HOOK_NAME
    assert os.environ[R.ROI_ENV_VAR] == path
    assert R.INSTALLER_ENTRY in os.environ[mh.HOOKS_ENV_VAR]
    out = capsys.readouterr().out
    assert 'ROI filter ENABLED' in out
    assert 'WARNING' not in out


def test_enabling_from_a_missing_path_refuses_before_the_run(tmp_path):
    with pytest.raises(R.RoiError, match='does not exist'):
        R.enable_roi_filter(str(tmp_path / 'never_written.json'))


def test_enabling_appends_to_another_extensions_hook_entry(tmp_path, monkeypatch):
    other = types.ModuleType('spacr_other_extension')
    other.install = lambda: None
    monkeypatch.setitem(sys.modules, 'spacr_other_extension', other)
    monkeypatch.setenv(mh.HOOKS_ENV_VAR, 'spacr_other_extension:install')
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    R.enable_roi_filter(roi_set, path=str(tmp_path / 'roi.json'),
                        on_missing='all', verbose=False)
    assert os.environ[mh.HOOKS_ENV_VAR] == (
        f'spacr_other_extension:install,{R.INSTALLER_ENTRY}')
    assert os.environ[R.ON_MISSING_ENV_VAR] == 'all'
    # Disabling leaves the other extension's entry behind.
    assert R.disable_roi_filter() is True
    assert os.environ[mh.HOOKS_ENV_VAR] == 'spacr_other_extension:install'
    assert R.ROI_ENV_VAR not in os.environ


def test_enabling_from_a_drawn_layer_saves_it_for_the_workers(tmp_path):
    target = tmp_path / 'roi' / 'measure_roi.json'
    R.enable_roi_filter(_shapes_layer(), path=str(target), verbose=False)
    assert target.is_file()
    assert R.RoiSet.load(str(target)).covers('anything') is True


def test_disabling_with_nothing_else_registered_clears_the_variable(tmp_path):
    R.enable_roi_filter(R.RoiSet(fields={R.ANY_FIELD: (_rect(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=False)
    assert R.disable_roi_filter() is True
    assert mh.HOOKS_ENV_VAR not in os.environ


def test_an_unregistered_filter_says_every_object_would_be_measured():
    ok, message = R.worker_delivery_status('spawn')
    assert ok is False
    assert 'not registered in this process' in message


def test_a_fork_pool_inherits_the_registry_but_would_not_survive_spawn(tmp_path):
    """The trap: it works today and fails silently on the next machine."""
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    mh.register_region_filter_hook(R.RoiRegionFilter(roi_set),
                                   name=R.HOOK_NAME, priority=R.HOOK_PRIORITY)
    os.environ.pop(mh.HOOKS_ENV_VAR, None)
    os.environ.pop(R.ROI_ENV_VAR, None)
    ok, message = R.worker_delivery_status('fork')
    assert ok is True
    assert 'would NOT survive' in message


def test_a_spawn_pool_without_the_environment_is_not_covered():
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})
    mh.register_region_filter_hook(R.RoiRegionFilter(roi_set),
                                   name=R.HOOK_NAME, priority=R.HOOK_PRIORITY)
    os.environ.pop(mh.HOOKS_ENV_VAR, None)
    os.environ.pop(R.ROI_ENV_VAR, None)
    ok, message = R.worker_delivery_status('spawn')
    assert ok is False
    assert 'every object in every field would be measured' in message


def test_the_default_start_method_is_the_one_measure_crop_will_use(monkeypatch):
    monkeypatch.setenv('SPACR_START_METHOD', 'spawn')
    mh.register_region_filter_hook(
        R.RoiRegionFilter(R.RoiSet(fields={R.ANY_FIELD: (_rect(),)})),
        name=R.HOOK_NAME, priority=R.HOOK_PRIORITY)
    os.environ.pop(mh.HOOKS_ENV_VAR, None)
    os.environ.pop(R.ROI_ENV_VAR, None)
    ok, message = R.worker_delivery_status()
    assert ok is False
    assert "'spawn'" in message
