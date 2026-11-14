"""What an ROI refuses, and what ``enable_roi_filter`` carries to the workers.

Every refusal here has the same shape: the drawn region would still produce a
region, just not the one on the screen, and Measure would report a clean run
over the wrong pixels. Raising is the only visible failure available.
"""
from __future__ import annotations

import os

import pytest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import Spacing


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

def test_a_plane_needs_two_axes():
    """A one-axis spacing has no plane for a region to lie in."""
    with pytest.raises(R.RoiError, match='only 1 axis'):
        R._plane_axes(Spacing(scale=(0.65,), axes=('x',), units='um'))


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
    blocker.write_text('this is a file', encoding="utf-8")
    target = blocker / 'roi' / 'measure_roi.json'

    with pytest.raises(R.RoiError) as excinfo:
        R.RoiSet(fields={R.ANY_FIELD: (_roi(),)}).save(str(target))

    assert 'could not be written' in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


# ---------------------------------------------------------------------------
# Enabling with an on_missing override
# ---------------------------------------------------------------------------

