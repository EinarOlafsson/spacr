"""``B14`` — a drawn ROI, honoured by Measure.

Three properties, and the third is the one that is silent when it breaks.

* **The geometry is right.** An ROI is stored in world coordinates, so a
  polygon drawn on a half-resolution preview must name the same region on the
  full-resolution mask. The tests below build a mask with objects at known
  places, draw an ROI over a known subset, and assert that exactly that subset
  survives — not "roughly the right number".

* **The excluded objects are never measured.** Not "measured and then dropped
  from the table": the filter runs before a single ``regionprops`` call, which
  is what makes an ROI a speed-up rather than a post-filter. A spy on
  ``spacr.measure.regionprops_table`` records every label mask that reached it
  and asserts every object in every one of them is inside the ROI.

* **It reaches the worker processes.** ``measure_crop`` measures fields in a
  pool. Under ``spawn`` a worker is a cold interpreter with an empty hook
  registry, so a filter registered through the Python API applies to *nothing*
  — the run completes and every object in the field is measured while the user
  believes only the ROI was. There is a positive test (a real ``spawn`` worker
  installs the ROI for itself and drops the right labels) and its negative
  control (a parent-only registration reaches no worker at all), because
  without the control the positive test proves nothing.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3

import numpy as np
import pytest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import LayerStack, Shape, ShapesLayer, Spacing


# ---------------------------------------------------------------------------
# fixtures — the registry and the environment must never outlive a test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_hooks():
    """Empty registries and untouched environment, before and after."""
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


#: Four discs, radius 6/8/10/12, one per quadrant of a 128 px field. The radii
#: differ so that a measured object can be identified by its area after the
#: pipeline has relabelled it.
CENTRES = ((32, 32), (32, 96), (96, 32), (96, 96))
RADII = (6, 8, 10, 12)


def disc_mask(size=128, radii=RADII, scale=1.0):
    """A label mask of four discs, label ``i + 1`` at ``CENTRES[i]``."""
    mask = np.zeros((size, size), np.uint16)
    yy, xx = np.mgrid[:size, :size]
    for i, ((cy, cx), radius) in enumerate(zip(CENTRES, radii), start=1):
        cy, cx, radius = cy * scale, cx * scale, radius * scale
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = i
    return mask


def top_half(size=128):
    """A rectangle over the top half of the field — objects 1 and 2."""
    return R.RegionOfInterest('rectangle', [[0.0, 0.0], [size / 2, size]],
                              name='top half')


def context(mask, *, object_type='cell', file_name='plate1_A01_F001',
            settings=None, spacing=None):
    """A :class:`spacr.measure_hooks.RegionContext` over ``mask``."""
    return mh.RegionContext(object_type=object_type, file_name=file_name,
                            mask=mask, settings=settings or {},
                            spacing=spacing)


# ---------------------------------------------------------------------------
# 1. the geometry
# ---------------------------------------------------------------------------

def test_the_roi_keeps_exactly_the_objects_inside_it():
    mask = disc_mask()
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (top_half(),)})
    keep = R.RoiRegionFilter(roi_set)(context(mask))
    np.testing.assert_array_equal(keep, [True, True, False, False])


def test_inverting_the_roi_keeps_exactly_the_others():
    """'Exclude this debris' is the same drawing read the other way."""
    mask = disc_mask()
    roi_set = R.RoiSet(fields={R.ANY_FIELD: (top_half(),)}, invert=True)
    keep = R.RoiRegionFilter(roi_set)(context(mask))
    np.testing.assert_array_equal(keep, [False, False, True, True])


def test_a_polygon_is_not_its_bounding_box():
    """A triangle over the top-left quadrant takes object 1 and nothing else.

    Its bounding box would take object 2 as well, so this fails if the ROI is
    ever quietly reduced to an extent.
    """
    mask = disc_mask()
    triangle = R.RegionOfInterest(
        'polygon', [[0.0, 0.0], [0.0, 70.0], [70.0, 0.0]])
    keep = R.RoiRegionFilter(R.RoiSet(fields={'*': (triangle,)}))(
        context(mask))
    np.testing.assert_array_equal(keep, [True, False, False, False])


def test_two_rois_on_one_field_are_a_union():
    mask = disc_mask()
    left = R.RegionOfInterest('rectangle', [[0.0, 0.0], [128.0, 64.0]])
    bottom_right = R.RegionOfInterest(
        'rectangle', [[64.0, 64.0], [128.0, 128.0]])
    keep = R.RoiRegionFilter(R.RoiSet(fields={'*': (left, bottom_right)}))(
        context(mask))
    np.testing.assert_array_equal(keep, [True, False, True, True])


def test_an_ellipse_excludes_the_corners_a_rectangle_would_keep():
    """The two kinds differ exactly where it matters: near the corners."""
    mask = np.zeros((64, 64), np.uint16)
    mask[30:34, 30:34] = 1   # the middle of the box
    mask[4:8, 4:8] = 2       # tucked into its top-left corner
    corners = [[0.0, 0.0], [64.0, 64.0]]
    rectangle = R.RoiSet(fields={'*': (
        R.RegionOfInterest('rectangle', corners),)})
    ellipse = R.RoiSet(fields={'*': (
        R.RegionOfInterest('ellipse', corners),)})
    np.testing.assert_array_equal(R.RoiRegionFilter(rectangle)(context(mask)),
                                  [True, True])
    np.testing.assert_array_equal(R.RoiRegionFilter(ellipse)(context(mask)),
                                  [True, False])


def test_the_overlap_rule_asks_how_much_of_the_object_is_inside():
    """A cut that halves object 1 keeps it at 40% and drops it at 60%."""
    mask = np.zeros((64, 64), np.uint16)
    yy, xx = np.mgrid[:64, :64]
    mask[(yy - 32) ** 2 + (xx - 32) ** 2 <= 10 ** 2] = 1
    # Everything at or above row 32: exactly half the disc, minus the centre
    # row, which the half-open boundary rule gives to the lower half.
    half = R.RegionOfInterest('rectangle', [[32.0, 0.0], [64.0, 64.0]])
    generous = R.RoiSet(fields={'*': (half,)}, mode='overlap',
                        min_overlap=0.4)
    strict = R.RoiSet(fields={'*': (half,)}, mode='overlap', min_overlap=0.6)
    assert R.RoiRegionFilter(generous)(context(mask)).tolist() == [True]
    assert R.RoiRegionFilter(strict)(context(mask)).tolist() == [False]


def test_the_centroid_rule_gives_a_shared_edge_to_exactly_one_roi():
    """Two ROIs meeting on an edge partition the objects rather than sharing."""
    mask = np.zeros((32, 64), np.uint16)
    mask[10:14, 30:34] = 1  # centroid at column 31.5, on the seam
    left = R.RoiSet(fields={'*': (R.RegionOfInterest(
        'rectangle', [[0.0, 0.0], [32.0, 32.0]]),)})
    right = R.RoiSet(fields={'*': (R.RegionOfInterest(
        'rectangle', [[0.0, 32.0], [32.0, 64.0]]),)})
    decisions = [R.RoiRegionFilter(s)(context(mask))[0] for s in (left, right)]
    assert sum(bool(d) for d in decisions) == 1, (
        'an object on a shared edge was counted twice or not at all')


def test_an_empty_mask_is_answered_without_asking_the_geometry():
    keep = R.RoiRegionFilter(R.RoiSet(fields={'*': (top_half(),)}))(
        context(np.zeros((16, 16), np.uint16)))
    assert keep.shape == (0,)


def test_an_object_type_the_roi_does_not_cover_is_waved_through():
    mask = disc_mask()
    roi_set = R.RoiSet(fields={'*': (top_half(),)}, object_types=('cell',))
    filt = R.RoiRegionFilter(roi_set)
    np.testing.assert_array_equal(filt(context(mask, object_type='cell')),
                                  [True, True, False, False])
    np.testing.assert_array_equal(
        filt(context(mask, object_type='nucleus')), [True] * 4)


# ---------------------------------------------------------------------------
# 2. the world — an ROI drawn on a preview names the same region at full res
# ---------------------------------------------------------------------------

def test_an_roi_drawn_on_a_half_resolution_preview_lands_on_the_full_mask():
    """The reason vertices are world coordinates and not array indices.

    The same ROI is drawn on a 2-world-units-per-pixel preview; measured
    against a full-resolution mask it must keep the same objects. Storing
    ``(row, column)`` would keep the wrong half of the field.
    """
    preview = ShapesLayer(name='roi', ndim=2,
                          spacing=Spacing(scale=(2.0, 2.0), axes=('y', 'x')))
    # Drawn in the preview's own indices: the top half of a 64-pixel preview.
    preview.add(Shape('rectangle', [[0.0, 0.0], [32.0, 64.0]]))
    roi_set = R.RoiSet.from_shapes_layer(preview)

    vertices = roi_set.fields[R.ANY_FIELD][0].vertices
    assert vertices.max() == pytest.approx(128.0), (
        'the vertices were stored in preview pixels, not world units')
    keep = R.RoiRegionFilter(roi_set)(context(disc_mask()))
    np.testing.assert_array_equal(keep, [True, True, False, False])


def test_a_translated_layer_carries_its_crop_offset_into_the_roi():
    """A tile cut out of a mosaic keeps its place in the mosaic."""
    layer = ShapesLayer(name='roi', ndim=2, spacing=Spacing(
        scale=(1.0, 1.0), translate=(1000.0, 2000.0), axes=('y', 'x')))
    layer.add(Shape('rectangle', [[0.0, 0.0], [10.0, 10.0]]))
    roi_set = R.RoiSet.from_shapes_layer(layer)
    np.testing.assert_allclose(roi_set.fields['*'][0].vertices.min(axis=0),
                               [1000.0, 2000.0])


def test_an_anisotropic_z_stack_is_filtered_in_micrometres():
    """A 3-D mask is judged in world units, not voxels.

    0.65 µm in xy is an ordinary spaCR stack; an ROI drawn in µm over the
    right-hand half of a 64-voxel field is at x > 20.8 µm, which is not where
    "half" is if the voxel size is ignored.
    """
    mask = np.zeros((4, 64, 64), np.uint16)
    mask[:, 30:34, 8:12] = 1     # x ≈ 6.5 µm — left of the ROI
    mask[:, 30:34, 50:54] = 2    # x ≈ 33.5 µm — inside it
    corners = [[0.0, 20.8], [41.6, 41.6]]
    roi_set = R.RoiSet(fields={'*': (
        R.RegionOfInterest('rectangle', corners),)}, units='um')
    keep = R.RoiRegionFilter(roi_set)(context(
        mask, settings={'voxel_size_xy_um': 0.65, 'voxel_size_z_um': 2.0},
        spacing=(2.0, 0.65, 0.65)))
    np.testing.assert_array_equal(keep, [False, True])
    # ...and the same numbers read as voxels select an empty strip of the
    # field, which is what an ignored voxel size costs.
    voxels = R.RoiSet(fields={'*': (
        R.RegionOfInterest('rectangle', corners),)})
    np.testing.assert_array_equal(
        R.RoiRegionFilter(voxels)(context(mask)), [False, False])


def test_a_2d_roi_selects_a_column_through_a_z_stack():
    """A polygon drawn looking down the stack names a column through it."""
    mask = np.zeros((5, 32, 32), np.uint16)
    mask[0:2, 4:8, 4:8] = 1      # top-left, first two slices
    mask[3:5, 24:28, 24:28] = 2  # bottom-right, last two slices
    roi = R.RegionOfInterest('rectangle', [[0.0, 0.0], [16.0, 16.0]])
    keep = R.RoiRegionFilter(R.RoiSet(fields={'*': (roi,)}))(
        context(mask, spacing=(1.0, 1.0, 1.0)))
    np.testing.assert_array_equal(keep, [True, False])


def test_mixing_micrometres_with_pixels_raises_rather_than_drawing():
    """The failure Spacing itself refuses; it must not sneak back in here."""
    roi_set = R.RoiSet(fields={'*': (top_half(),)}, units='um')
    with pytest.raises(R.RoiError, match="measured in 'um'.*measured in 'px'"):
        R.RoiRegionFilter(roi_set)(context(disc_mask()))


# ---------------------------------------------------------------------------
# 3. which fields, and what an uncovered one means
# ---------------------------------------------------------------------------

def test_a_field_of_its_own_wins_over_the_default():
    mask = disc_mask()
    roi_set = R.RoiSet(fields={
        R.ANY_FIELD: (top_half(),),
        'plate1_A01_F002': (R.RegionOfInterest(
            'rectangle', [[64.0, 0.0], [128.0, 128.0]]),)})
    filt = R.RoiRegionFilter(roi_set)
    np.testing.assert_array_equal(
        filt(context(mask, file_name='plate1_A01_F001')),
        [True, True, False, False])
    np.testing.assert_array_equal(
        filt(context(mask, file_name='plate1_A01_F002')),
        [False, False, True, True])


def test_the_field_is_matched_on_its_stem_not_its_path():
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (top_half(),)})
    filt = R.RoiRegionFilter(roi_set)
    np.testing.assert_array_equal(
        filt(context(disc_mask(),
                     file_name='/data/exp/merged/plate1_A01_F001.npy')),
        [True, True, False, False])


def test_an_uncovered_field_refuses_rather_than_measuring_everything():
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (top_half(),)})
    with pytest.raises(R.RoiError, match='no ROI covers field'):
        R.RoiRegionFilter(roi_set)(context(disc_mask(),
                                           file_name='plate1_A01_F009'))


@pytest.mark.parametrize('on_missing,expected', [('all', True), ('none', False)])
def test_an_uncovered_field_can_be_measured_whole_or_skipped(on_missing,
                                                             expected):
    roi_set = R.RoiSet(fields={'plate1_A01_F001': (top_half(),)},
                       on_missing=on_missing)
    keep = R.RoiRegionFilter(roi_set)(context(disc_mask(),
                                              file_name='other'))
    assert keep.tolist() == [expected] * 4


def test_a_field_with_an_empty_roi_list_is_covered_and_encloses_nothing():
    """Present-but-empty is not the same as absent."""
    roi_set = R.RoiSet(fields={'plate1_A01_F001': ()})
    assert roi_set.covers('plate1_A01_F001')
    keep = R.RoiRegionFilter(roi_set)(context(disc_mask()))
    np.testing.assert_array_equal(keep, [False] * 4)


# ---------------------------------------------------------------------------
# 4. persistence and validation
# ---------------------------------------------------------------------------

def test_the_roi_round_trips_through_the_file_a_worker_reads(tmp_path):
    roi_set = R.RoiSet(fields={'*': (top_half(),), 'plate1_A01_F002': ()},
                       mode='overlap', min_overlap=0.25, invert=True,
                       object_types=('cell', 'nucleus'), on_missing='all')
    path = roi_set.save(str(tmp_path / 'sub' / 'roi.json'))
    restored = R.RoiSet.load(path)

    assert restored.mode == 'overlap'
    assert restored.min_overlap == 0.25
    assert restored.invert is True
    assert restored.object_types == ('cell', 'nucleus')
    assert restored.on_missing == 'all'
    assert sorted(restored.fields) == ['*', 'plate1_A01_F002']
    np.testing.assert_allclose(restored.fields['*'][0].vertices,
                               top_half().vertices)
    # Readable by a human being who wants to know what was measured.
    assert json.loads(open(path).read())['fields']['*'][0]['kind'] == 'rectangle'


def test_an_unreadable_roi_is_refused_rather_than_ignored(tmp_path):
    with pytest.raises(R.RoiError, match='does not exist'):
        R.RoiSet.load(str(tmp_path / 'nope.json'))
    junk = tmp_path / 'junk.json'
    junk.write_text('not json')
    with pytest.raises(R.RoiError, match='could not be read'):
        R.RoiSet.load(str(junk))
    listy = tmp_path / 'list.json'
    listy.write_text('[]')
    with pytest.raises(R.RoiError, match='not an ROI'):
        R.RoiSet.load(str(listy))


def test_an_open_shape_is_not_a_region():
    with pytest.raises(R.RoiError, match='closed shape'):
        R.RegionOfInterest('path', [[0, 0], [1, 1]])
    layer = ShapesLayer(name='roi', ndim=2)
    layer.add_path([[0.0, 0.0], [5.0, 5.0]])
    with pytest.raises(R.RoiError, match='no closed shape'):
        R.RoiSet.from_shapes_layer(layer)


@pytest.mark.parametrize('kwargs,fragment', [
    ({'mode': 'nearest'}, 'unknown ROI mode'),
    ({'min_overlap': 0.0}, 'fraction'),
    ({'min_overlap': 1.5}, 'fraction'),
    ({'object_types': ('cel',)}, 'unknown object type'),
    ({'on_missing': 'maybe'}, 'unknown on_missing'),
    ({'axes': ('x', 'y')}, 'vertex order reversed'),
    ({'axes': ('y', 'y')}, 'two different axis names'),
])
def test_a_set_that_cannot_mean_what_it_says_raises(kwargs, fragment):
    with pytest.raises(R.RoiError, match=fragment):
        R.RoiSet(fields={'*': (top_half(),)}, **kwargs)


def test_a_polygon_needs_three_points_and_finite_ones():
    with pytest.raises(R.RoiError, match='three points'):
        R.RegionOfInterest('polygon', [[0, 0], [1, 1]])
    with pytest.raises(R.RoiError, match='finite'):
        R.RegionOfInterest('rectangle', [[0, 0], [np.nan, 1]])
    with pytest.raises(R.RoiError, match=r'\(M, 2\)'):
        R.RegionOfInterest('rectangle', [[0, 0, 0], [1, 1, 1]])


def test_the_description_says_what_will_be_measured():
    roi_set = R.RoiSet(fields={'*': (top_half(),)}, invert=True,
                       mode='overlap', min_overlap=0.3)
    text = roi_set.describe()
    assert 'outside' in text and '30%' in text and 'every field' in text


# ---------------------------------------------------------------------------
# 5. the whole pipeline: only the ROI's objects are measured, and only they
#    ever reach regionprops
# ---------------------------------------------------------------------------

def _merged_field(size=128):
    """A merged ``(Y, X, C)`` stack: 4 intensity channels, then the masks."""
    cell = disc_mask(size)
    nucleus = np.where(disc_mask(size, radii=(3, 4, 5, 6)) > 0, cell, 0)
    pathogen = np.where(disc_mask(size, radii=(1, 2, 2, 3)) > 0, cell, 0)
    rng = np.random.default_rng(0)
    channels = []
    for _ in range(4):
        base = rng.integers(50, 200, size=(size, size)).astype(np.uint16)
        base[cell > 0] += 3000
        channels.append(base)
    return np.stack(channels + [cell, nucleus, pathogen],
                    axis=-1).astype(np.uint16)


@pytest.fixture
def merged_project(tmp_path):
    """One merged field plus the ``measurements/`` folder the pipeline leaves."""
    merged = tmp_path / 'merged'
    merged.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)
    np.save(merged / 'plate1_A01_F001.npy', _merged_field())
    return tmp_path


def _settings(merged):
    from spacr.settings import get_measure_crop_settings
    settings = get_measure_crop_settings(settings={})
    settings.update({
        'src': str(merged / 'merged'), 'channels': [0, 1, 2, 3],
        'cell_mask_dim': 4, 'nucleus_mask_dim': 5, 'pathogen_mask_dim': 6,
        'save_measurements': True, 'save_png': False, 'save_arrays': False,
        'plot': False, 'verbose': False, 'timelapse': False,
        'crop_mode': ['cell'], 'experiment': 'exp', 'n_jobs': 1,
        'test_mode': False, 'cytoplasm': True,
    })
    return settings


def _measured_cell_areas(project):
    with sqlite3.connect(project / 'measurements' / 'measurements.db') as db:
        rows = db.execute('SELECT cell_area FROM cell').fetchall()
    return sorted(int(round(row[0])) for row in rows)


def test_only_the_objects_inside_the_roi_are_measured(merged_project):
    """The deliverable: draw a polygon, measure only inside it."""
    from spacr.measure import _measure_crop_core

    settings = _settings(merged_project)
    _measure_crop_core(0, [], 'plate1_A01_F001.npy', dict(settings))
    everything = _measured_cell_areas(merged_project)
    assert len(everything) == 4, 'the baseline run should measure four cells'

    os.remove(merged_project / 'measurements' / 'measurements.db')
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(merged_project / 'roi.json'), verbose=False)
    _measure_crop_core(0, [], 'plate1_A01_F001.npy', dict(settings))
    inside = _measured_cell_areas(merged_project)

    # Objects 1 and 2 — radius 6 and 8 — and nothing else. Identified by area
    # because the pipeline is free to relabel what it kept.
    assert len(inside) == 2
    assert inside == everything[:2], (
        f'expected the two smallest discs (the ones inside the ROI), got '
        f'{inside} out of {everything}')


def test_the_excluded_objects_never_reach_regionprops(merged_project,
                                                      monkeypatch):
    """Not measured and then dropped — never measured.

    The filter runs before any feature is computed, which is what makes an ROI
    cheap on a crowded field. Every label mask handed to ``regionprops_table``
    is recorded and every object in every one of them has to be inside the ROI.
    """
    from spacr import measure as M

    seen = []
    real = M.regionprops_table

    def spy(labels, *args, **kwargs):
        array = np.asarray(labels)
        if np.issubdtype(array.dtype, np.integer) and array.ndim == 2:
            seen.append(array.copy())
        return real(labels, *args, **kwargs)

    monkeypatch.setattr(M, 'regionprops_table', spy)
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(merged_project / 'roi.json'), verbose=False)
    M._measure_crop_core(0, [], 'plate1_A01_F001.npy',
                         dict(_settings(merged_project)))

    assert seen, 'regionprops_table was never called; the test proves nothing'
    outside = []
    for array in seen:
        if array.shape != (128, 128):
            continue  # a cropped object, already inside by construction
        for label in np.unique(array):
            if label == 0:
                continue
            rows, _cols = np.nonzero(array == label)
            if rows.mean() >= 64:
                outside.append(int(label))
    assert not outside, (
        f'objects {sorted(set(outside))} were outside the ROI and still '
        f'reached regionprops — the filter ran too late to save any work')


# ---------------------------------------------------------------------------
# 6. reaching the workers — the failure that is silent by construction
# ---------------------------------------------------------------------------

def _filter_in_this_process(payload):
    """Run in a worker: apply whatever region filters *this* process has.

    Defined at module scope because a ``spawn`` worker gets it by importing
    this module and looking it up by name; a closure would not survive.
    """
    from spacr.measure_hooks import (apply_region_filter_hooks,
                                     region_filter_hooks)
    mask, file_name = payload
    names = [entry.name for entry in region_filter_hooks()]
    _kept, dropped = apply_region_filter_hooks(
        mask, object_type='cell', file_name=file_name, settings={})
    return names, dropped


def test_the_roi_reaches_a_real_spawn_worker(tmp_path):
    """The deliverable: a cold interpreter filters the field for itself."""
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=False)

    with mp.get_context('spawn').Pool(1) as pool:
        names, dropped = pool.apply(
            _filter_in_this_process, ((disc_mask(), 'plate1_A01_F001'),))

    assert names == [R.HOOK_NAME], (
        'the spawned worker did not install the ROI filter, so every object '
        'in every field it measured would have been measured')
    assert dropped == (3, 4)


def test_a_parent_only_registration_reaches_no_spawn_worker(tmp_path):
    """The negative control, which is what makes the test above mean anything.

    Registering the filter through the Python API and nothing else is the
    natural thing to write and a silent no-op in every ``spawn`` worker. If
    this ever starts passing by accident — because something else set the
    environment — the positive test above would be proving nothing.
    """
    mh.register_region_filter_hook(
        R.RoiRegionFilter(R.RoiSet(fields={'*': (top_half(),)})),
        name=R.HOOK_NAME)

    with mp.get_context('spawn').Pool(1) as pool:
        names, dropped = pool.apply(
            _filter_in_this_process, ((disc_mask(), 'plate1_A01_F001'),))

    assert names == []
    assert dropped == ()
    # ...and the module says so, in advance, rather than letting it happen.
    ok, message = R.worker_delivery_status('spawn')
    assert ok is False
    assert 'every object in every field would be measured' in message


@pytest.fixture
def other_extension(monkeypatch):
    """A second, harmless ``SPACR_MEASURE_HOOKS`` entry to share the variable."""
    import sys
    import types

    module = types.ModuleType('spacr_roi_other_extension')
    module.install = lambda: mh.register_preprocessing_hook(
        lambda array, ctx: array, name='other-extension')
    monkeypatch.setitem(sys.modules, 'spacr_roi_other_extension', module)
    return 'spacr_roi_other_extension:install'


def test_enable_writes_the_environment_every_start_method_inherits(
        tmp_path, other_extension):
    """Three things, and the environment is the one that reaches a worker."""
    os.environ[mh.HOOKS_ENV_VAR] = other_extension
    path = str(tmp_path / 'roi.json')
    name = R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                               path=path, verbose=False)

    assert name == R.HOOK_NAME
    assert os.path.isfile(os.environ[R.ROI_ENV_VAR])
    # Appended, not assigned: another extension may already be in there.
    entries = os.environ[mh.HOOKS_ENV_VAR].split(',')
    assert entries == [other_extension, R.INSTALLER_ENTRY]
    # And the hook is tagged 'env', so the start-method warning leaves it alone.
    entry, = mh.region_filter_hooks()
    assert entry.source == 'env'
    assert mh.warn_if_hooks_will_not_reach_workers('spawn') is False


def test_disabling_leaves_another_extensions_entry_alone(tmp_path,
                                                        other_extension):
    os.environ[mh.HOOKS_ENV_VAR] = other_extension
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=False)
    assert R.disable_roi_filter() is True
    assert os.environ[mh.HOOKS_ENV_VAR] == other_extension
    assert R.ROI_ENV_VAR not in os.environ
    assert mh.region_filter_hooks() == ()


def test_disabling_the_last_entry_takes_the_variable_with_it(tmp_path):
    os.environ.pop(mh.HOOKS_ENV_VAR, None)
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=False)
    assert R.disable_roi_filter() is True
    assert mh.HOOKS_ENV_VAR not in os.environ
    # Idempotent: turning it off twice is not an error.
    assert R.disable_roi_filter() is False


def test_enabling_twice_installs_one_filter_not_two(tmp_path):
    """Re-running a GUI action must not intersect the ROI with itself."""
    for _ in range(2):
        R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                            path=str(tmp_path / 'roi.json'), verbose=False)
    assert len(mh.region_filter_hooks()) == 1


def test_a_worker_that_cannot_load_the_roi_refuses_to_measure(tmp_path):
    """Loudly: a worker that measured the whole field would say nothing."""
    os.environ.pop(R.ROI_ENV_VAR, None)
    with pytest.raises(R.RoiError, match='is not set'):
        R.install()
    os.environ[R.ROI_ENV_VAR] = str(tmp_path / 'gone.json')
    with pytest.raises(R.RoiError, match='does not exist'):
        R.install()


def test_enabling_from_a_path_fails_here_rather_than_in_a_worker(tmp_path):
    with pytest.raises(R.RoiError, match='does not exist'):
        R.enable_roi_filter(str(tmp_path / 'nope.json'), verbose=False)


def test_enabling_from_a_saved_path_installs_it(tmp_path):
    path = R.RoiSet(fields={'*': (top_half(),)}).save(
        str(tmp_path / 'roi.json'))
    R.enable_roi_filter(path, verbose=False)
    entry, = mh.region_filter_hooks()
    assert entry.name == R.HOOK_NAME
    np.testing.assert_array_equal(entry.func(context(disc_mask())),
                                  [True, True, False, False])


def test_enabling_from_a_shapes_layer_is_one_call(tmp_path):
    stack = LayerStack()
    layer = stack.add_shapes(name='ROI', ndim=2)
    layer.add_rectangle([0.0, 0.0], [64.0, 128.0])
    R.enable_roi_filter(layer, path=str(tmp_path / 'roi.json'), verbose=False)
    entry, = mh.region_filter_hooks()
    np.testing.assert_array_equal(entry.func(context(disc_mask())),
                                  [True, True, False, False])


def test_the_status_says_what_a_fork_pool_would_and_would_not_survive(tmp_path):
    ok, message = R.worker_delivery_status('spawn')
    assert ok is False and 'not registered in this process' in message

    mh.register_region_filter_hook(
        R.RoiRegionFilter(R.RoiSet(fields={'*': (top_half(),)})),
        name=R.HOOK_NAME)
    ok, message = R.worker_delivery_status('fork')
    assert ok is True and 'would NOT survive' in message

    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=False)
    ok, message = R.worker_delivery_status('spawn')
    assert ok is True and 'install it themselves' in message


def test_enable_prints_the_delivery_status_when_asked(tmp_path, capsys):
    R.enable_roi_filter(R.RoiSet(fields={'*': (top_half(),)}),
                        path=str(tmp_path / 'roi.json'), verbose=True)
    out = capsys.readouterr().out
    assert 'ROI filter ENABLED' in out and 'a ' in out


def test_the_filter_reports_what_it_did(tmp_path):
    filt = R.RoiRegionFilter(R.RoiSet(fields={'*': (top_half(),)}))
    filt(context(disc_mask()))
    filt(context(disc_mask(), object_type='nucleus'))
    assert filt.stats == {'kept': 4, 'dropped': 4, 'fields': 1}
    assert '4 object(s) kept' in filt.report()


def test_a_filter_needs_a_roi_set():
    with pytest.raises(R.RoiError, match='needs a RoiSet'):
        R.RoiRegionFilter('a polygon, honest')
    with pytest.raises(R.RoiError, match='unknown on_missing'):
        R.RoiRegionFilter(R.RoiSet(), on_missing='shrug')


def test_the_set_refuses_anything_that_is_not_a_region():
    with pytest.raises(R.RoiError, match='not a RegionOfInterest'):
        R.RoiSet(fields={'*': ('a polygon',)})
    with pytest.raises(R.RoiError, match="needs 'kind'"):
        R.RegionOfInterest.from_dict({'vertices': [[0, 0], [1, 1]]})
    with pytest.raises(R.RoiError, match='fields'):
        R.RoiSet.from_dict({'fields': 7})
