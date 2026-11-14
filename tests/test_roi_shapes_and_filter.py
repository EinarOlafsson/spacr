"""What a drawn region of interest refuses, and what ``enable_roi_filter`` exports.

Every refusal has the same reason: the drawn shape would still produce a
region, just not the one on the screen, and Measure would report a clean run
over the wrong pixels. Enabling the filter has to reach spawn workers, which
rebuild it from the environment alone, so the override and path a caller
passes must land in both the saved file and the environment.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import measure_hooks as mh
from spacr import roi as R
from spacr.layers import LayerError, LayerStack


@pytest.fixture
def clean_hooks():
    """Leave the hook registry and the ROI environment as they were found."""
    names = (mh.HOOKS_ENV_VAR, R.ROI_ENV_VAR, R.ON_MISSING_ENV_VAR)
    saved = {name: os.environ.get(name) for name in names}
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _roi(name="r"):
    return R.RegionOfInterest("rectangle", [[0.0, 0.0], [64.0, 128.0]],
                              name=name)


def test_a_shape_that_encloses_nothing_or_lies_off_its_axes_is_refused(
        subtests):
    """Each degenerate shape, or a layer read on foreign axes, is refused naming what is wrong."""
    cases = [
        ("a rectangle from a single click", "rectangle", [[12.0, 34.0]],
         "a rectangle needs at least two points, got 1"),
        ("an ellipse from a single click", "Ellipse", [[1.0, 2.0]],
         "a ellipse needs at least two points, got 1"),
        ("a two-point polygon", "polygon", [[0.0, 0.0], [1.0, 1.0]],
         "a polygon needs at least three points"),
        ("a line", "line", [[0.0, 0.0], [1.0, 1.0]],
         "an ROI is a closed shape: 'polygon', 'rectangle' or 'ellipse', not "
         "'line'."),
        ("three-dimensional points", "rectangle", [[0.0, 0.0, 0.0],
                                                   [1.0, 1.0, 1.0]],
         "an ROI needs an (M, 2) array of world points, got shape (2, 3)"),
        ("a vertex at infinity", "rectangle", [[0.0, 0.0], [np.inf, 1.0]],
         "ROI vertices must all be finite"),
    ]
    for case_id, kind, vertices, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(R.RoiError) as caught:
                R.RegionOfInterest(kind, vertices)
            assert str(caught.value).startswith(message), str(caught.value)

    with subtests.test(msg="a valid rectangle is normalised"):
        kept = R.RegionOfInterest(" Rectangle ", [[0, 0], [2, 3]], name=7)
        assert (kept.kind, kept.name) == ("rectangle", "7")
        assert kept.vertices.dtype == np.float64
        assert kept.vertices.tolist() == [[0.0, 0.0], [2.0, 3.0]]

    with subtests.test(msg="a shapes layer read on axes it does not have"):
        layer = LayerStack().add_shapes(name="ROI", ndim=2)
        layer.add_rectangle([0.0, 0.0], [64.0, 128.0])
        with pytest.raises(R.RoiError) as caught:
            R.RoiSet.from_shapes_layer(layer, axes=("row", "col"))
        assert str(caught.value) == (
            f"the shapes layer is on axes {layer.spacing.axes}, which do not "
            "include ('row', 'col')")
        assert type(caught.value.__cause__) is LayerError


def test_enable_roi_filter_exports_the_path_and_on_missing_override(
        tmp_path, clean_hooks, subtests):
    """The override reaches the saved file and the environment; the caller's set is untouched."""
    for on_missing, hooks in (("all", ""), (None, R.INSTALLER_ENTRY)):
        with subtests.test(msg=f"on_missing={on_missing!r}, hooks={hooks!r}"):
            os.environ.pop(R.ON_MISSING_ENV_VAR, None)
            os.environ[mh.HOOKS_ENV_VAR] = hooks
            mh.clear_measurement_hooks()
            saved = R.RoiSet(fields={R.ANY_FIELD: (_roi(),)},
                             on_missing="error")
            path = str(tmp_path / str(on_missing) / "roi" / "measure_roi.json")

            assert R.enable_roi_filter(saved, path=path, on_missing=on_missing,
                                       verbose=False) == R.HOOK_NAME

            assert os.environ[R.ROI_ENV_VAR] == os.path.abspath(path)
            assert os.environ.get(R.ON_MISSING_ENV_VAR) == on_missing
            assert os.environ[mh.HOOKS_ENV_VAR] == R.INSTALLER_ENTRY
            assert R.RoiSet.load(path).on_missing == (on_missing or "error")
            assert saved.on_missing == "error"
            assert [entry.name for entry in mh.region_filter_hooks()] == [
                R.HOOK_NAME]
