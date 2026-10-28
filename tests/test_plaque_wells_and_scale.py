"""The well is the ruler: detection geometry, and pixels to millimetres.

A plaque area in pixels is a property of the microscope. The same plaque at two
magnifications gives two areas, and a study that pools them is comparing
optics. The well is a manufactured object of known physical size that is
already in the image, so its measured diameter converts every area into mm^2.

That makes the arithmetic here load-bearing for the science rather than
cosmetic, which is why the conversion is driven with a KNOWN answer rather
than checked for self-consistency: a scale that is wrong by a constant factor
is self-consistent and silently rescales an entire study.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import plaque
from spacr.plaque import (WELL_DIAMETERS_MM, PlaqueScale, Well, crop_well,
                          scale_from_well)


def test_a_missing_well_detector_names_the_optional_dependency(monkeypatch):
    """No detector install must produce an actionable error, not a bare import."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(ImportError, match=r'pip install "spacr\[plaque\]"') as caught:
        plaque._load_detector("wells.pt")

    assert isinstance(caught.value.__cause__, ImportError)


def test_an_installed_well_detector_receives_the_requested_weights(monkeypatch):
    """The positive import path still constructs the detector exactly once."""
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(weights)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    detector = plaque._load_detector("wells.pt")

    assert isinstance(detector, FakeYOLO)
    assert calls == ["wells.pt"]


def _capturing_detector(monkeypatch):
    """A fake detector that records the ``source`` it was handed.

    :param monkeypatch: the pytest fixture, used to replace the real loader.
    :returns: the dict that ``predict`` fills with its keyword arguments.
    """
    captured = {}

    class Detector:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(plaque, "_load_detector", lambda _weights: Detector())
    return captured


def test_the_detector_is_handed_bgr_because_that_is_what_it_was_trained_on(
    monkeypatch
):
    """The channel order the detector is fed, pinned so it cannot slip back.

    Ultralytics decodes a FILE PATH with OpenCV, so these detectors trained
    and infer in BGR; handed an ``H x W x 3`` array it converts nothing and
    assumes the caller ordered the channels itself. spaCR's house reader,
    ``cellpose.io.imread``, returns RGB, and for one release
    :func:`spacr.plaque.detect_wells` passed that straight through -- so every
    plate photograph spaCR split into wells was shown to the detector with
    red and blue exchanged.

    It is not cosmetic. Scored against the published 129-image, 297-box test
    split on 2026-09-20, RGB cost v1 (the v3 weights) mAP50 0.8888 -> 0.5877
    and recall 0.9024 -> 0.6128, and cost v2 (v4) mAP50 0.9480 -> 0.6972 and
    recall 0.9663 -> 0.2963: two thirds of the wells on a plate, lost
    silently, since a plate split into eight crops instead of twelve says
    nothing about the four.

    If this test fails, somebody is passing RGB to ultralytics again.
    """
    captured = _capturing_detector(monkeypatch)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30

    plaque.detect_wells(rgb, "wells.pt")

    source = captured["source"]
    assert source[..., 0].tolist() == [[30, 30], [30, 30]], "blue leads"
    assert source[..., 1].tolist() == [[20, 20], [20, 20]]
    assert source[..., 2].tolist() == [[10, 10], [10, 10]], "red trails"
    assert rgb[0, 0, 0] == 10, "the caller's own array is left alone"


def test_an_image_with_no_channel_order_to_get_wrong_is_passed_through(
    monkeypatch
):
    """Greyscale and four-channel arrays are handed over untouched.

    A two-dimensional array has one channel and therefore no order to swap,
    and reversing the last axis of an ``H x W x 4`` array would move the
    alpha channel to the front rather than exchange red and blue. Only a
    three-channel colour array is converted.
    """
    grey = np.arange(4, dtype=np.uint8).reshape(2, 2)
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)

    captured = _capturing_detector(monkeypatch)
    plaque.detect_wells(grey, "wells.pt")
    assert captured["source"] is grey

    captured = _capturing_detector(monkeypatch)
    plaque.detect_wells(rgba, "wells.pt")
    assert captured["source"] is rgba


def test_a_file_path_reaches_ultralytics_unchanged(monkeypatch):
    """The reference route: ultralytics decodes a path itself, in BGR.

    It is also the route every training image took, which is why passing a
    path and passing a BGR array score identically -- mAP50 0.9480 both ways
    for v2 over the published test split -- while RGB does not.
    """
    captured = _capturing_detector(monkeypatch)

    plaque.detect_wells("/plates/plate01.png", "wells.pt")

    assert captured["source"] == "/plates/plate01.png"


def test_a_square_box_gives_its_side_as_the_diameter():
    well = Well(x0=100, y0=200, x1=400, y1=500)
    assert well.width == 300 and well.height == 300
    assert well.diameter_px == 300
    assert well.axis_ratio == 1.0


def test_a_lopsided_box_is_reported_as_lopsided():
    """The honesty check on the ruler.

    A well is round, so a box far from square means the detector clipped it at
    an image edge, merged two wells, or found something else. Any of those
    makes the diameter wrong, and the diameter rescales every area in that
    well -- so a wrong one is worse than a missing one.
    """
    well = Well(x0=0, y0=0, x1=400, y1=200)
    assert well.diameter_px == 300, "the mean degrades gently"
    assert well.axis_ratio == 0.5, "and the disagreement is still visible"


def test_a_well_dict_includes_stored_and_derived_geometry():
    well = Well(1, 2, 5, 8, 0.75)

    assert well.as_dict() == {
        "x0": 1,
        "y0": 2,
        "x1": 5,
        "y1": 8,
        "confidence": 0.75,
        "diameter_px": 5.0,
        "axis_ratio": pytest.approx(2 / 3),
    }


def test_a_zero_sized_box_has_no_axis_ratio_or_physical_scale():
    """Degenerate detector output cannot become a plausible ruler."""
    well = Well(x0=4, y0=4, x1=4, y1=4)

    assert well.axis_ratio == 0.0
    assert scale_from_well(well, well_diameter_mm=10.0) is None


def test_well_detection_filters_bad_boxes_and_orders_the_rest(
    monkeypatch, caplog
):
    """Detector output is parsed, filtered, and returned in reading order."""
    class Box:
        def __init__(self, xyxy, confidence):
            self.xyxy = np.asarray([xyxy], dtype=float)
            self.conf = confidence

    boxes = [
        Box((30, 20, 40, 30), np.asarray([0.8])),
        Box((4, 4, 4, 12), np.asarray([0.7])),
        Box((0, 0, 20, 5), np.asarray([0.6])),
        Box((10, 5, 20, 15), None),
    ]
    results = [types.SimpleNamespace(boxes=None), types.SimpleNamespace(boxes=boxes)]
    calls = []

    class Detector:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return results

    monkeypatch.setattr(plaque, "_load_detector", lambda weights: Detector())

    with caplog.at_level("WARNING"):
        wells = plaque.detect_wells(
            np.zeros((40, 40), dtype=np.uint8),
            "wells.pt",
            confidence=0.4,
            imgsz=320,
            min_axis_ratio=0.7,
        )

    assert [(well.x0, well.y0, well.confidence) for well in wells] == [
        (10, 5, 1.0),
        (30, 20, 0.8),
    ]
    assert len(calls) == 1
    assert np.array_equal(
        calls[0].pop("source"), np.zeros((40, 40), dtype=np.uint8)
    )
    assert calls[0] == {"conf": 0.4, "imgsz": 320, "verbose": False}
    assert "axis ratio 0.25 is below 0.70" in caplog.text


def test_the_conversion_is_right_against_a_worked_example():
    """A 6-well well is 34.8 mm. Measured at 348 px, that is 10 px/mm.

    At 10 px/mm, one square millimetre is 100 square pixels. So a 5000-pixel
    plaque is 50 mm^2, and that number is checked against arithmetic done
    here rather than against the implementation's own output.
    """
    well = Well(x0=0, y0=0, x1=348, y1=348)
    scale = scale_from_well(well, plate_format="6-well")
    assert scale is not None
    assert scale.px_per_mm == pytest.approx(10.0)
    assert scale.area_mm2(5000) == pytest.approx(50.0)


def test_the_same_plaque_at_two_magnifications_measures_the_same():
    """The entire point, stated as a test.

    One plaque, two microscopes. The second images at twice the magnification,
    so both the well and the plaque are twice the size in pixels and the
    plaque covers four times the pixel area. In millimetres they must agree --
    if they do not, pooling plates from two scopes is meaningless.
    """
    low = Well(x0=0, y0=0, x1=348, y1=348)
    high = Well(x0=0, y0=0, x1=696, y1=696)

    low_scale = scale_from_well(low, plate_format="6-well")
    high_scale = scale_from_well(high, plate_format="6-well")

    plaque_low_px = 5000
    plaque_high_px = 5000 * 4          # twice the linear size

    assert low_scale.area_mm2(plaque_low_px) == pytest.approx(
        high_scale.area_mm2(plaque_high_px))
    assert low_scale.px_per_mm * 2 == pytest.approx(high_scale.px_per_mm)


def test_no_physical_diameter_means_no_scale_rather_than_a_guess():
    """``None``, not a default plate format.

    Assuming a format would fill the mm^2 columns with confident numbers that
    are wrong by whatever the real plate was -- and wrong by a constant factor,
    so nothing downstream would look odd.
    """
    well = Well(x0=0, y0=0, x1=348, y1=348)
    assert scale_from_well(well) is None


def test_an_unknown_plate_format_is_refused_rather_than_guessed():
    well = Well(x0=0, y0=0, x1=348, y1=348)
    with pytest.raises(KeyError, match="unknown plate format"):
        scale_from_well(well, plate_format="5-well")


def test_an_explicit_diameter_overrides_the_format():
    """A custom plate, or a dish measured by hand, must win."""
    well = Well(x0=0, y0=0, x1=100, y1=100)
    scale = scale_from_well(well, plate_format="6-well", well_diameter_mm=50.0)
    assert scale.well_diameter_mm == 50.0
    assert scale.source == "explicit"
    assert scale.px_per_mm == pytest.approx(2.0)


def test_every_declared_plate_format_produces_a_usable_scale():
    """No format may be present but unusable."""
    well = Well(x0=0, y0=0, x1=500, y1=500)
    for name in WELL_DIAMETERS_MM:
        scale = scale_from_well(well, plate_format=name)
        assert isinstance(scale, PlaqueScale)
        assert scale.px_per_mm > 0
        assert scale.source == name


def test_padding_past_the_TOP_LEFT_edge_clips_instead_of_wrapping():
    """The lower bound is the one that matters, and it is not symmetric.

    Numpy already truncates a slice that runs past the END of an array, so
    clamping the upper bound is defensive rather than load-bearing -- removing
    it does not change any result, which a mutation confirmed.

    The START is different. ``image[-15:]`` is not empty and does not raise: it
    is the LAST fifteen rows. So a well near the top-left, padded, would be cut
    from the OPPOSITE CORNER of the image -- a real crop, of real tissue, of
    the wrong well, with nothing anywhere to indicate it. Its plaques would be
    counted against a condition they did not come from.
    """
    image = np.zeros((100, 100), dtype=np.uint8)
    image[0:20, 0:20] = 1          # the well we mean
    image[80:100, 80:100] = 9      # what a negative index would take instead

    well = Well(x0=5, y0=5, x1=20, y1=20)
    crop = crop_well(image, well, pad=20)

    assert crop.shape == (40, 40), "clipped at 0, not wrapped"
    assert 9 not in np.unique(crop), (
        "the crop came from the far corner: a negative start index wrapped")
    assert 1 in np.unique(crop)


def test_cropping_past_the_bottom_right_edge_is_simply_truncated():
    """Documented rather than asserted as a guarantee of ours: this is numpy's
    behaviour, and the test exists so a reader knows the asymmetry is known."""
    image = np.zeros((100, 100), dtype=np.uint8)
    well = Well(x0=90, y0=90, x1=100, y1=100)
    assert crop_well(image, well, pad=20).shape == (30, 30)


def test_a_crop_well_inside_the_image_gets_its_padding():
    image = np.zeros((100, 100), dtype=np.uint8)
    well = Well(x0=40, y0=40, x1=60, y1=60)
    assert crop_well(image, well, pad=5).shape == (30, 30)
    assert crop_well(image, well).shape == (20, 20)


@pytest.mark.parametrize('scores,expected', [([.6, .9], .9), ([None, None], None)])
def test_duplicate_well_boxes_merge_before_physical_scaling_without_mutating_input(scores, expected):
    """Overlapping detector views are one well; separated wells stay independent."""
    from copy import deepcopy

    boxes = [dict(box=[2, 3, 22, 23], confidence=scores[0]),
             dict(box=[5, 5, 25, 25], confidence=scores[1]),
             dict(box=[50, 3, 70, 23], confidence=.8)]
    before = deepcopy(boxes)
    merged = plaque._merge_well_boxes(boxes, (40, 100))
    assert merged == [dict(box=[2, 3, 25, 25], confidence=expected),
                      dict(box=[50, 3, 70, 23], confidence=.8)]
    assert boxes == before
    assert scale_from_well(Well(2, 3, 25, 25), well_diameter_mm=6.4).px_per_mm == pytest.approx(22.5 / 6.4)


def test_a_large_centred_well_uses_the_complete_acquired_frame():
    boxes = [dict(box=[10, 10, 90, 90], confidence=.9)]
    assert plaque._merge_well_boxes(boxes, (100, 100)) == [
        dict(box=[0, 0, 100, 100], confidence=.9, whole_frame_rule=True)]
    assert boxes == [dict(box=[10, 10, 90, 90], confidence=.9)]


def test_a_small_corner_well_does_not_acquire_unrelated_frame_pixels():
    boxes = [dict(box=[0, 0, 20, 20], confidence=.9)]
    assert plaque._merge_well_boxes(boxes, (100, 100)) == boxes
    assert boxes == [dict(box=[0, 0, 20, 20], confidence=.9)]


def test_enlarging_a_single_channel_well_retains_its_acquired_channel_axis():
    image = np.arange(42, dtype=np.uint16).reshape(6, 7, 1)
    before = image.copy()
    crop, metadata = plaque._well_resize_forward(image, [1, 1, 5, 5], side=8, max_factor=2)
    assert crop.shape == (8, 8, 1)
    assert metadata['ratio'] == metadata['scale_x'] == metadata['scale_y'] == 2
    assert crop[0, 0, 0] == before[1, 1, 0]
    assert crop[-1, -1, 0] == before[4, 4, 0]
    assert np.all(np.diff(crop[..., 0].astype(np.int64), axis=0) >= 0)
    assert np.all(np.diff(crop[..., 0].astype(np.int64), axis=1) >= 0)
    crop[0, 0, 0] = 0
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('dtype', [np.uint16, np.float64])
def test_well_resize_keeps_channels_and_source_pixels_for_identity_geometry(dtype):
    """A no-resize well still owns its crop and retains the acquired channel axis."""
    image = np.arange(42, dtype=dtype).reshape(6, 7, 1)
    before = image.copy()
    crop, metadata = plaque._well_resize_forward(image, [1, 1, 5, 5], side=4, max_factor=2)
    np.testing.assert_array_equal(crop[..., 0], before[1:5, 1:5, 0])
    assert crop.shape == (4, 4, 1)
    assert metadata['ratio'] == metadata['scale_x'] == metadata['scale_y'] == 1
    crop[0, 0, 0] = 0
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('kind', ['planar-rgb', 'rgba', 'two-channel', 'invalid-volume'])
def test_acquired_plaque_layouts_are_normalized_without_rewriting_the_image(tmp_path, kind):
    import tifffile

    if kind == 'planar-rgb':
        image = np.arange(90, dtype=np.uint16).reshape(3, 5, 6)
        expected = np.moveaxis(image, 0, -1)
    elif kind == 'rgba':
        image = np.arange(120, dtype=np.uint16).reshape(5, 6, 4)
        expected = image[..., :3]
    elif kind == 'two-channel':
        image = np.arange(60, dtype=np.uint16).reshape(5, 6, 2)
        expected = image[..., 0]
    else:
        image = np.ones((2, 2, 5, 6), np.uint16)
        expected = None
    path = tmp_path / 'acquired.tif'
    tifffile.imwrite(path, image, photometric='minisblack')
    before = path.read_bytes()
    if expected is None:
        with pytest.raises(ValueError, match='Unsupported plaque image shape'):
            plaque._read_plaque_image(path)
    else:
        result = plaque._read_plaque_image(path)
        np.testing.assert_array_equal(result, expected)
        assert result.flags.c_contiguous
    assert path.read_bytes() == before


def test_declined_plaque_reader_names_the_unreadable_source(tmp_path, monkeypatch):
    import importlib

    cellpose_io = importlib.import_module('cellpose.io')

    path = tmp_path / 'unreadable.tif'
    path.write_bytes(b'preserved unreadable source')
    monkeypatch.setattr(cellpose_io, 'imread', lambda path: None)
    with pytest.raises(ValueError, match='Cannot read') as caught:
        plaque._read_plaque_image(path)
    assert str(path) in str(caught.value)
    assert path.read_bytes() == b'preserved unreadable source'


@pytest.mark.parametrize('requested,mode', [(True, 'on'), (False, 'off')])
def test_boolean_well_resize_settings_keep_their_explicit_meaning(requested, mode):
    assert plaque._well_resize_mode(requested) == mode


@pytest.mark.parametrize('outcome', ['existing', 'resolved', 'unavailable', 'lookup-error'])
def test_well_detector_resolution_honors_no_download_and_reports_unavailability(
        tmp_path, monkeypatch, outcome):
    import importlib

    zoo = importlib.import_module('spacr.model_zoo')
    weights = tmp_path / 'detector.pt'
    if outcome in ('existing', 'resolved'):
        weights.write_bytes(b'local checkpoint boundary only; not inferred')
    calls = []

    def ensure(requested, **kwargs):
        calls.append((requested, kwargs))
        if outcome == 'lookup-error':
            raise OSError('registry unavailable')
        return weights if outcome == 'resolved' else None

    monkeypatch.setattr(zoo, '_ensure_model_file', ensure)
    requested = str(weights) if outcome == 'existing' else 'requested-detector'
    result = plaque._resolve_well_resize_detector(requested, fetch=False)
    assert result == (str(weights) if outcome in ('existing', 'resolved') else None)
    if outcome == 'existing':
        assert calls == []
    else:
        assert calls == [(requested, {'kinds': ('detector',), 'download': False,
                                      'setting': 'plaque_well_resize'})]


def test_nonpositive_declared_well_geometry_refuses_before_detector_resolution(monkeypatch):
    import importlib

    zoo = importlib.import_module('spacr.model_zoo')
    monkeypatch.setattr(zoo, '_model_preprocess', lambda model: {})
    monkeypatch.setattr(plaque, '_resolve_well_resize_detector',
                        lambda *args, **kwargs: pytest.fail('bad geometry cannot resolve a detector'))
    with pytest.raises(ValueError, match='positive side'):
        plaque._well_resize_plan({'plaque_well_resize': True,
                                 'plaque_well_resize_side': 0}, fetch=False)


@pytest.mark.parametrize('colour', [False, True])
def test_nonbyte_detector_inputs_are_owned_bgr_with_bounded_levels(colour):
    image = np.arange(90 if colour else 30, dtype=np.uint16).reshape((5, 6, 3) if colour else (5, 6))
    before = image.copy()
    result = plaque._well_detector_input(image)
    assert result.dtype == np.uint8 and result.shape == (5, 6, 3)
    assert result.min() == 0 and result.max() == 255
    if colour:
        assert result[2, 3, 0] > result[2, 3, 2]
    else:
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 1], result[..., 2])
    result[:] = 0
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('backend', ['cpu-detector', 'isolated-reader'])
def test_well_detector_backends_keep_cpu_channel_and_cache_ownership(monkeypatch, backend):
    from types import SimpleNamespace
    import importlib.util
    import importlib

    plaque_papers = importlib.import_module('spacr.plaque_papers')

    image = np.zeros((5, 6, 3), np.uint8)
    image[:] = [10, 20, 30]
    before = image.copy()
    calls, loads = [], []
    actual_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec',
                        lambda name, *args, **kwargs: (None if backend == 'isolated-reader' else object())
                        if name == 'ultralytics' else actual_find(name, *args, **kwargs))

    class Detector:
        def predict(self, values, **kwargs):
            calls.append((values.copy(), kwargs))
            assert kwargs['device'] == 'cpu'
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=np.array([[1, 2, 4, 5]]), conf=np.array([.8])))]

    def load(weights):
        loads.append(weights)
        return Detector()

    def isolated(values, weights, **kwargs):
        calls.append((values.copy(), kwargs))
        return [Well(1, 2, 4, 5, .8)]

    monkeypatch.setattr(plaque, '_WELL_DETECTORS', {})
    monkeypatch.setattr(plaque, '_load_detector', load)
    monkeypatch.setattr(plaque_papers, 'reader_detect', isolated)
    for unused in range(2):
        assert plaque._raw_well_detections(image, 'local-detector') == [(1., 2., 4., 5., .8)]
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], before if backend == 'isolated-reader' else before[..., ::-1])
    assert loads == ([] if backend == 'isolated-reader' else ['local-detector'])
    np.testing.assert_array_equal(image, before)


def test_invalid_inverse_shape_and_empty_well_labels_never_modify_existing_labels():
    meta = dict(box=[1, 1, 5, 5], width=4, height=4)
    with pytest.raises(ValueError, match='does not match'):
        plaque._well_resize_inverse(np.zeros((3, 4), np.uint16), meta)
    full = np.arange(36, dtype=np.uint32).reshape(6, 6)
    before = full.copy()
    assert plaque._paste_well_labels(full, np.zeros((4, 4), np.uint16), meta) == {}
    np.testing.assert_array_equal(full, before)
    two = np.ones((4, 4, 2), np.float32)
    normalized = plaque._well_resize_intensity(two)
    assert normalized.shape == (3, 4, 4) and not normalized.any()


@pytest.mark.parametrize('flows', ['both', 'probability-only', 'rgb-only'])
def test_resized_well_flows_return_to_source_coordinates_without_overwriting_overlap(monkeypatch, flows):
    """Each pixel keeps its first owning well's labels and optional display products."""
    image = np.arange(80, dtype=np.float32).reshape(8, 10)
    before = image.copy()
    boxes = [dict(box=[0, 0, 6, 6], confidence=.9),
             dict(box=[4, 2, 10, 8], confidence=.8)]
    monkeypatch.setattr(plaque, '_well_resize_boxes', lambda *args, **kwargs: {'boxes': boxes})
    calls = []

    class Model:
        def eval(self, values, **kwargs):
            calls.append(kwargs)
            number = len(calls)
            rgb = (np.full((6, 6, 3), number * 30, np.uint8)
                   if flows != 'probability-only' else None)
            probability = (np.full((6, 6), float(number), np.float32)
                           if flows != 'rgb-only' else None)
            return np.ones((6, 6), np.uint16), [rgb, None, probability], None

    result = plaque._segment_resized_wells(Model(), image,
        dict(side=6, max_factor=2, detector='explicit-test-adapter'),
        min_size=None, want_flows=True, want_metrics=False)
    assert len(calls) == 2 and all('min_size' not in call for call in calls)
    assert result['labels'].shape == image.shape
    assert result['labels'][3, 5] == 1 and result['labels'][7, 9] == 2
    if flows == 'probability-only':
        assert result['flows']['flow_rgb'] is None
    else:
        rgb = result['flows']['flow_rgb']
        assert rgb.shape == (8, 10, 3)
        np.testing.assert_array_equal(rgb[3, 5], [30, 30, 30])
        np.testing.assert_array_equal(rgb[7, 9], [60, 60, 60])
    if flows == 'rgb-only':
        assert result['flows']['cellprob'] is None
    else:
        probability = result['flows']['cellprob']
        assert probability.shape == image.shape
        assert probability[3, 5] == 1 and probability[7, 9] == 2
    assert result['metrics'] == {}
    np.testing.assert_array_equal(image, before)
