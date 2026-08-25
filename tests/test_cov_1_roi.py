"""What an ROI refuses, and what ``enable_roi_filter`` carries to the workers.

Every refusal here has the same shape: the drawn region would still produce a
region, just not the one on the screen, and Measure would report a clean run
over the wrong pixels. Raising is the only visible failure available.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import LayerError, LayerStack, Spacing


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


def _roi(name='r'):
    return R.RegionOfInterest('rectangle', [[0.0, 0.0], [64.0, 128.0]],
                              name=name)


# ---------------------------------------------------------------------------
# Geometry that encloses nothing
# ---------------------------------------------------------------------------

def test_a_rectangle_from_a_single_point_is_refused():
    """One corner is not a rectangle: it encloses no area at all.

    A click that never became a drag leaves a one-vertex shape behind. Taken
    as an ROI it rasterises to an empty region, so Measure would report zero
    objects everywhere and look like a segmentation failure.
    """
    with pytest.raises(R.RoiError, match='at least two points'):
        R.RegionOfInterest('rectangle', [[12.0, 34.0]])


def test_a_plane_needs_two_axes():
    """A one-axis spacing has no plane for a region to lie in."""
    with pytest.raises(R.RoiError, match='only 1 axis'):
        R._plane_axes(Spacing(scale=(0.65,), axes=('x',), units='um'))


def test_axes_the_layer_does_not_have_are_named_in_the_error():
    """Asking for a plane the shapes were not drawn in says so.

    The alternative is an IndexError out of the vertex slicing several frames
    later, which says nothing about the axis names that did not match.
    """
    stack = LayerStack()
    layer = stack.add_shapes(name='ROI', ndim=2)
    layer.add_rectangle([0.0, 0.0], [64.0, 128.0])

    with pytest.raises(R.RoiError) as excinfo:
        R.RoiSet.from_shapes_layer(layer, axes=('row', 'col'))

    message = str(excinfo.value)
    assert "('row', 'col')" in message
    assert str(tuple(layer.spacing.axes)) in message
    assert isinstance(excinfo.value.__cause__, LayerError)


# ---------------------------------------------------------------------------
# Describing and saving a set
# ---------------------------------------------------------------------------

def test_a_default_for_the_rest_is_named_in_the_description():
    """Named fields plus a fallback is one line, and says both halves.

    The description is what a run log and the status bar show. A set that
    covers two named fields AND everything else reads as covering only the two
    unless the fallback is mentioned.
    """
    mixed = R.RoiSet(fields={'plate1_A01_1': (_roi(),),
                             'plate1_A02_1': (_roi(),),
                             R.ANY_FIELD: (_roi(),)})
    described = mixed.describe()
    assert '2 field(s) plus a default for the rest' in described
    assert described.startswith('3 ROI(s) over ')

    only_named = R.RoiSet(fields={'plate1_A01_1': (_roi(),)})
    assert 'plus a default' not in only_named.describe()


def test_an_unwritable_roi_path_is_refused_at_the_call(tmp_path):
    """A save that cannot happen must raise where the user asked for it.

    The workers read the ROI back from this file. If saving failed quietly,
    every worker would fall back to measuring the whole field.
    """
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('this is a file')
    target = blocker / 'roi' / 'measure_roi.json'

    with pytest.raises(R.RoiError) as excinfo:
        R.RoiSet(fields={R.ANY_FIELD: (_roi(),)}).save(str(target))

    assert 'could not be written' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


# ---------------------------------------------------------------------------
# Enabling with an on_missing override
# ---------------------------------------------------------------------------

def test_an_on_missing_override_reaches_the_set_and_the_environment(tmp_path):
    """``on_missing=`` overrides the set AND is exported for the workers.

    A spawn worker rebuilds the filter from the environment alone. Overriding
    the rule only in this process would leave the workers refusing (or
    measuring) fields that the parent had decided to keep.
    """
    saved = R.RoiSet(fields={R.ANY_FIELD: (_roi(),)}, on_missing='error')
    path = str(tmp_path / 'roi' / 'measure_roi.json')

    R.enable_roi_filter(saved, path=path, on_missing='all', verbose=False)

    assert os.environ[R.ON_MISSING_ENV_VAR] == 'all'
    assert os.environ[R.ROI_ENV_VAR] == os.path.abspath(path)
    written = R.RoiSet.load(path)
    assert written.on_missing == 'all', (
        "the override has to be written into the file the workers read")
    assert saved.on_missing == 'error', "the caller's set is not mutated"

    entry, = mh.region_filter_hooks()
    assert entry.name == R.HOOK_NAME
