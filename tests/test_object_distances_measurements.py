"""Distances measured by ``spacr.object_distances`` on masks with known answers.

Every case is built from small hand-placed squares, so the expected number is
written down rather than recorded: two squares four pixels apart are four
pixels apart, a square's centre sits half its width from its own rim, and an
object with no partner of another type is infinitely far from one (``inf``),
never NaN.  NaN is reserved for "not measured" (a peakless object).
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import object_distances as od

INF = float("inf")
NAN = float("nan")


def _square(shape=(40, 40), value=1, row=10, col=10, size=6):
    """One ``size`` x ``size`` square of ``value`` at ``(row, col)``."""
    mask = np.zeros(shape, dtype=np.int32)
    mask[row:row + size, col:col + size] = value
    return mask


def _two_squares():
    """Two 6x6 squares, labels 1 and 2, with a four-pixel gap between them."""
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[10:16, 10:16] = 1
    mask[10:16, 20:26] = 2
    return mask


def _plane(*boxes, shape=(40, 40)):
    """A float plane with ``(rows, cols, value)`` boxes painted on it."""
    image = np.zeros(shape, dtype=float)
    for rows, cols, value in boxes:
        image[rows, cols] = value
    return image


def _refusing_peak_finder(monkeypatch):
    """Make skimage's peak finder raise, as it can for a degenerate object."""
    import skimage.feature

    def _explode(*args, **kwargs):
        raise ValueError("degenerate plateau")

    monkeypatch.setattr(skimage.feature, "peak_local_max", _explode)


def _assert_records(frame, expected):
    """``frame`` rows equal ``expected`` dicts exactly (NaN == NaN, inf == inf)."""
    assert len(frame) == len(expected)
    for got, want in zip(frame.to_dict("records"), expected):
        assert list(got) == list(want)
        for key, value in want.items():
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(got[key]), key
            else:
                assert got[key] == pytest.approx(value), key


def test_as_spacing_turns_a_voxel_size_into_one_value_per_axis_or_none(subtests):
    """A scalar fills every axis, a per-axis tuple is kept only at the right
    length (never padded), and no voxel size means pixel units (None)."""
    cases = [
        ("scalar", 2.5, 3, (2.5, 2.5, 2.5)),
        ("integer scalar", 2, 2, (2.0, 2.0)),
        ("per axis", (0.5, 0.25), 2, (0.5, 0.25)),
        ("too short for 3 axes", (0.5, 0.25), 3, None),
        ("too long for 1 axis", [1.0, 2.0], 1, None),
        ("pixel units", None, 2, None),
    ]
    for case_id, spacing, ndim, expected in cases:
        with subtests.test(msg=case_id):
            assert od._as_spacing(spacing, ndim) == expected


def test_sample_reads_inside_points_and_reports_outside_ones_as_infinitely_far(
        subtests):
    """Points round to their voxel; anything off the field is ``inf`` rather
    than clipped to the edge, and an all-outside set never indexes the field."""
    field = np.arange(25, dtype=float).reshape(5, 5)
    cases = [
        ("no points", np.empty((0, 2)), []),
        ("inside, rounded", np.array([[2.0, 2.0], [0.4, 3.6]]), [12.0, 4.0]),
        ("mixed", np.array([[2.0, 2.0], [99.0, 2.0], [-3.0, 1.0]]),
         [12.0, INF, INF]),
        ("all outside", np.array([[99.0, 99.0], [-4.0, -7.0]]), [INF, INF]),
        ("on the far edge", np.array([[4.0, 4.0], [5.0, 0.0]]), [24.0, INF]),
    ]
    for case_id, points, expected in cases:
        with subtests.test(msg=case_id):
            out = od._sample(field, points)
            assert out.dtype == float
            np.testing.assert_array_equal(out, np.array(expected, dtype=float))


def test_local_maxima_finds_peaks_per_object_and_never_fails_the_run(
        subtests, monkeypatch):
    """Peaks come back in whole-image coordinates; an absent label or a peak
    finder that refuses gives an empty ``(0, 2)`` array; a bounding-box window
    with the whole-image floor finds exactly the whole-image peaks."""
    mask = _square(shape=(20, 20))
    image = _plane((slice(12, 13), slice(13, 14), 5.0), shape=(20, 20))

    with subtests.test(msg="one peak"):
        np.testing.assert_array_equal(od.local_maxima(image, mask, 1),
                                      [[12.0, 13.0]])
    with subtests.test(msg="label not in mask"):
        assert od.local_maxima(image, mask, label=7).shape == (0, 2)
    with subtests.test(msg="window offsets back to whole-image indices"):
        window = (slice(10, 16), slice(10, 16))
        np.testing.assert_array_equal(
            od.local_maxima(image, mask, 1, window=window, threshold=0.0),
            [[12.0, 13.0]])

    with subtests.test(msg="windowed search equals whole-image search"):
        from scipy import ndimage
        from skimage.segmentation import expand_labels

        rng = np.random.default_rng(3)
        labels = np.zeros((120, 120), np.int32)
        for i, (r, c) in enumerate(rng.integers(5, 115, (25, 2)), 1):
            labels[r, c] = i
        labels = expand_labels(labels, 9)
        noisy = rng.gamma(2.0, 300.0, labels.shape).astype(np.float32)
        floor = float(noisy.min())
        boxes = ndimage.find_objects(labels)
        for label in np.unique(labels)[1:]:
            whole = od.local_maxima(noisy, labels, int(label))
            window = od.local_maxima(noisy, labels, int(label),
                                     window=boxes[label - 1], threshold=floor)
            np.testing.assert_array_equal(window, whole)
            assert len(whole) >= 1

    with subtests.test(msg="peak finder refuses"):
        _refusing_peak_finder(monkeypatch)
        peaks = od.local_maxima(image, mask, label=1)
        assert peaks.shape == (0, 2)


def test_between_object_types_reports_exact_geometric_distances(subtests):
    """One row per primary object with own-boundary, radial and field-edge
    distances, plus per-partner centre/surface/overlap columns; a partner of
    another shape is skipped, a missing partner is ``inf``, and an empty
    primary mask is a ``label``-only table with no rows."""
    empty = np.zeros((16, 16), dtype=np.int32)
    contained = np.zeros((40, 40), dtype=np.int32)
    contained[12:16, 12:16] = 1
    own_6 = {"label": 1, "distance_to_own_boundary": 3.0,
             "relative_radial_position": 0.0, "distance_to_field_edge": 12.5}
    cases = [
        ("square centre is half its width from its rim",
         {"cell": _square(size=6)}, [own_6]),
        ("object in the corner sits near the field edge",
         {"cell": _square(row=0, col=0, size=4)},
         [{"label": 1, "distance_to_own_boundary": 2.0,
           "relative_radial_position": 0.5, "distance_to_field_edge": 1.5}]),
        ("contained nucleus measured from the rim, not the overlap",
         {"cell": _square(size=8), "nucleus": contained},
         [{"label": 1, "distance_to_own_boundary": 4.0,
           "relative_radial_position": 0.0, "distance_to_field_edge": 13.5,
           "centre_to_nucleus_surface": 0.0,
           "surface_to_nucleus_surface": 2.0,
           "centre_to_nearest_nucleus_centre": 0.0,
           "nucleus_overlap_fraction": 16 / 64}]),
        ("no partner is infinitely far, never NaN",
         {"cell": _square(), "nucleus": np.zeros((40, 40), np.int32)},
         [{**own_6, "centre_to_nucleus_surface": INF,
           "surface_to_nucleus_surface": INF,
           "centre_to_nearest_nucleus_centre": INF,
           "nucleus_overlap_fraction": 0.0}]),
        ("two squares four pixels apart",
         {"cell": _two_squares(), "nucleus": _two_squares()},
         [{**own_6, "centre_to_nucleus_surface": 0.0,
           "surface_to_nucleus_surface": 0.0,
           "centre_to_nearest_nucleus_centre": 0.0,
           "nucleus_overlap_fraction": 1.0},
          {**own_6, "label": 2, "centre_to_nucleus_surface": 0.0,
           "surface_to_nucleus_surface": 0.0,
           "centre_to_nearest_nucleus_centre": 0.0,
           "nucleus_overlap_fraction": 1.0}]),
        ("partner of another shape is skipped, not broadcast",
         {"cell": _two_squares(), "nucleus": np.zeros((8, 8), np.int32)},
         [own_6, {**own_6, "label": 2}]),
        ("empty primary mask", {"cell": empty, "nucleus": empty}, []),
    ]
    for case_id, masks, expected in cases:
        with subtests.test(msg=case_id):
            frame = od.between_object_types(masks, primary="cell")
            _assert_records(frame, expected)
            if not expected:
                assert list(frame.columns) == ["label"]

    with subtests.test(msg="gap between the two squares, physical units"):
        cell = _square(size=6)
        nucleus = _square(size=6, col=20)
        frame = od.between_object_types({"cell": cell, "nucleus": nucleus},
                                        primary="cell", spacing=0.5)
        row = frame.iloc[0]
        assert row["surface_to_nucleus_surface"] == pytest.approx(2.5)
        assert row["centre_to_nearest_nucleus_centre"] == pytest.approx(5.0)
        assert row["distance_to_own_boundary"] == pytest.approx(1.5)


def test_maxima_distances_describes_each_objects_peaks_per_channel(
        subtests, monkeypatch):
    """Per channel: count, spread and min/mean distances from each peak to the
    own rim, the centre and every same-shaped partner type; a 2-D image is one
    channel, a missing channel is skipped, a peakless object gets NaN distances
    beside a zero count, and an empty primary gives no rows."""
    peak_00 = {"channel_0_maxima_count": 1, "channel_0_maxima_spread": 0.0,
               "channel_0_maxima_to_own_boundary_min": 3.0,
               "channel_0_maxima_to_own_boundary_mean": 3.0,
               "channel_0_maxima_to_centre_min": 2.1213203435596424,
               "channel_0_maxima_to_centre_mean": 2.1213203435596424}
    nan_0 = {"channel_0_maxima_count": 0, "channel_0_maxima_spread": 0.0,
             "channel_0_maxima_to_own_boundary_min": NAN,
             "channel_0_maxima_to_own_boundary_mean": NAN,
             "channel_0_maxima_to_centre_min": NAN,
             "channel_0_maxima_to_centre_mean": NAN}
    blob = _plane((slice(12, 14), slice(12, 14), 100.0))
    stack = np.stack([blob, _plane((slice(14, 16), slice(14, 16), 50.0))], -1)
    far = np.zeros((40, 40), np.int32)
    far[10:14, 30:34] = 1
    cell = _square(size=8)
    cases = [
        ("single plane is one channel", {"cell": cell}, blob, {},
         [{"label": 1, **peak_00}]),
        ("multichannel keeps its channel axis", {"cell": cell}, stack, {},
         [{"label": 1, **peak_00, "channel_1_maxima_count": 1,
           "channel_1_maxima_spread": 0.0,
           "channel_1_maxima_to_own_boundary_min": 4.0,
           "channel_1_maxima_to_own_boundary_mean": 4.0,
           "channel_1_maxima_to_centre_min": 0.7071067811865476,
           "channel_1_maxima_to_centre_mean": 0.7071067811865476}]),
        ("channel the stack lacks is skipped", {"cell": cell}, blob,
         {"channels": (0, 5)}, [{"label": 1, **peak_00}]),
        ("peak measured against the partner type",
         {"cell": cell, "nucleus": far},
         _plane((slice(13, 14), slice(13, 14), 100.0)), {"spacing": None},
         [{"label": 1, "channel_0_maxima_count": 1,
           "channel_0_maxima_spread": 0.0,
           "channel_0_maxima_to_own_boundary_min": 4.0,
           "channel_0_maxima_to_own_boundary_mean": 4.0,
           "channel_0_maxima_to_centre_min": 0.7071067811865476,
           "channel_0_maxima_to_centre_mean": 0.7071067811865476,
           "channel_0_maxima_to_nucleus_surface_min": 17.0,
           "channel_0_maxima_to_nucleus_surface_mean": 17.0}]),
        ("flat field: NaN distances beside a zero count",
         {"cell": cell, "nucleus": cell}, np.zeros((40, 40)), {},
         [{"label": 1, **nan_0,
           "channel_0_maxima_to_nucleus_surface_min": NAN,
           "channel_0_maxima_to_nucleus_surface_mean": NAN}]),
        ("partner of another shape adds no columns",
         {"cell": cell, "nucleus": np.zeros((7, 7), np.int32)}, blob, {},
         [{"label": 1, **peak_00}]),
        ("two peaks have a spread", {"cell": _square(size=12)},
         _plane((slice(11, 12), slice(11, 12), 9.0),
                (slice(11, 12), slice(20, 21), 9.0)), {},
         [{"label": 1, "channel_0_maxima_count": 2,
           "channel_0_maxima_spread": 9.0,
           "channel_0_maxima_to_own_boundary_min": 2.0,
           "channel_0_maxima_to_own_boundary_mean": 2.0,
           "channel_0_maxima_to_centre_min": 6.363961030678928,
           "channel_0_maxima_to_centre_mean": 6.363961030678928}]),
        ("empty primary mask", {"cell": np.zeros((16, 16), np.int32)},
         np.zeros((16, 16)), {}, []),
    ]
    for case_id, masks, images, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            frame = od.maxima_distances(masks, images, primary="cell",
                                        **kwargs)
            _assert_records(frame, expected)

    with subtests.test(msg="refusing peak finder costs only the maxima"):
        _refusing_peak_finder(monkeypatch)
        frame = od.maxima_distances(
            {"cell": cell, "nucleus": _square(size=4, row=25, col=25)},
            blob, primary="cell")
        _assert_records(frame, [{"label": 1, **nan_0,
                                 "channel_0_maxima_to_nucleus_surface_min": NAN,
                                 "channel_0_maxima_to_nucleus_surface_mean": NAN}])


def test_intensity_centre_offset_measures_polarisation_per_channel(subtests):
    """A uniform stain has offset 0, a one-sided stain moves the intensity
    centre (exact distance, scaled by voxel size), a missing channel is
    skipped, and an empty mask is a table with no rows."""
    cell = _square(size=8)
    uniform = _plane((slice(10, 18), slice(10, 18), 50.0))
    one_end = _plane((slice(10, 12), slice(10, 18), 100.0))
    cases = [
        ("uniform stain", uniform, {}, {"channel_0_intensity_centre_offset": 0.0}),
        ("one-sided stain", one_end, {},
         {"channel_0_intensity_centre_offset": 3.0}),
        ("one-sided stain in physical units", one_end, {"spacing": 2.0},
         {"channel_0_intensity_centre_offset": 6.0}),
        ("missing channel is skipped", np.stack([uniform, one_end], -1),
         {"channels": (1, 9)}, {"channel_1_intensity_centre_offset": 3.0}),
    ]
    for case_id, images, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            frame = od.intensity_centre_offset(cell, images, primary="cell",
                                               **kwargs)
            _assert_records(frame, [{"label": 1, **expected}])
    with subtests.test(msg="empty mask"):
        frame = od.intensity_centre_offset(np.zeros((16, 16), np.int32),
                                           np.zeros((16, 16)), primary="cell")
        assert list(frame.columns) == ["label"] and len(frame) == 0


def test_object_distances_joins_the_families_without_type_prefixes(subtests):
    """No field means geometry only; ``maxima=False`` drops only the maxima
    family; rows stay one per object and no column carries the ``cell_``
    prefix that ``measure`` adds later."""
    geometry = ["label", "distance_to_own_boundary",
                "relative_radial_position", "distance_to_field_edge"]
    maxima = ["channel_0_maxima_count", "channel_0_maxima_spread",
              "channel_0_maxima_to_own_boundary_min",
              "channel_0_maxima_to_own_boundary_mean",
              "channel_0_maxima_to_centre_min",
              "channel_0_maxima_to_centre_mean"]
    image = _plane((slice(12, 13), slice(12, 13), 100.0))
    cases = [
        ("no field", None, {}, geometry),
        ("everything", image, {},
         geometry + ["channel_0_intensity_centre_offset"] + maxima),
        ("maxima off", image, {"maxima": False},
         geometry + ["channel_0_intensity_centre_offset"]),
    ]
    for case_id, images, kwargs, columns in cases:
        with subtests.test(msg=case_id):
            frame = od.object_distances({"cell": _two_squares()}, images,
                                        primary="cell", **kwargs)
            assert list(frame.columns) == columns
            assert list(frame["label"]) == [1, 2]
            assert not [c for c in frame.columns if c.startswith("cell_")]
    with subtests.test(msg="joined values"):
        frame = od.object_distances({"cell": _two_squares()}, image,
                                    primary="cell")
        row1, row2 = frame.to_dict("records")
        assert row1["channel_0_intensity_centre_offset"] == pytest.approx(
            0.7071067811865476)
        assert np.isnan(row2["channel_0_intensity_centre_offset"])
        assert (row1["channel_0_maxima_count"],
                row2["channel_0_maxima_count"]) == (1, 0)
