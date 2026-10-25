"""Well resize for plaque segmentation mirrors plaque model v6's training transform.

The reference is the self-contained ``normalize.py`` the v6 training run
used. Every transform here is compared with it pixel for pixel: the scale
rule ``side:1500,maxfactor:6``, the crop and resize (area when shrinking,
bilinear when enlarging), the pixel-centre nearest inverse, the paste and the
per-crop percentile scaling. Real training fields are used when the v6 data
folder is on this machine and skipped otherwise.
"""
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from spacr import model_zoo, plaque

V6_ROOT = Path("/mnt/firecuda2/Claude/plaque-v6-20261024")
REFERENCE = V6_ROOT / "execution" / "normalize.py"
RULE = "side:1500,maxfactor:6"


def _reference():
    if not REFERENCE.is_file():
        pytest.skip("plaque v6 reference normalize.py is not on this machine")
    spec = importlib.util.spec_from_file_location("v6_normalize_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _field(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        high = np.iinfo(dtype).max
        return rng.integers(0, high, size=shape, endpoint=True).astype(dtype)
    return (rng.random(shape) * 1000).astype(dtype)


def _labels(shape, n=25, seed=1):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, np.uint32)
    for i in range(1, n + 1):
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        r = int(rng.integers(1, 9))
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        out[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = i
    return out


CASES = [
    ("shrink", (2400, 3000, 3), np.uint8, [100, 50, 3100, 2350]),
    ("enlarge_capped", (220, 210), np.uint16, [5, 3, 205, 213]),
    ("enlarge_uncapped", (600, 500, 3), np.uint8, [0, 0, 400, 500]),
    ("native", (1600, 1700), np.float64, [10, 20, 1510, 1420]),
    ("odd_dtype", (300, 340, 3), np.int32, [1, 2, 333, 299]),
]


@pytest.mark.parametrize("name,shape,dtype,box", CASES, ids=[c[0] for c in CASES])
def test_forward_inverse_paste_and_intensity_match_the_reference(name, shape, dtype, box):
    ref = _reference()
    image = _field(shape, dtype)
    h, w = shape[:2]
    box = [min(box[0], w), min(box[1], h), min(box[2], w), min(box[3], h)]
    want_crop, _, want = ref.forward(image, box, RULE)
    crop, meta = plaque._well_resize_forward(image, box, 1500, 6)
    assert crop.dtype == want_crop.dtype
    np.testing.assert_array_equal(crop, want_crop)
    for key in ("box", "width", "height", "scale_x", "scale_y", "ratio"):
        assert meta[key] == want[key]
    assert meta["ratio"] == ref.scale_for(box[2] - box[0], box[3] - box[1], RULE)
    np.testing.assert_array_equal(plaque._well_resize_intensity(crop),
                                  ref.normalize_intensity(want_crop))
    crop_labels = _labels(crop.shape[:2])
    np.testing.assert_array_equal(plaque._well_resize_inverse(crop_labels, meta),
                                  ref.inverse(crop_labels, want))
    full_ref = np.zeros((h, w), np.uint32)
    full_mine = np.zeros((h, w), np.uint32)
    full_ref[:3, :3] = 7
    full_mine[:3, :3] = 7
    ref.paste(full_ref, crop_labels, want)
    mapping = plaque._paste_well_labels(full_mine, crop_labels, meta)
    np.testing.assert_array_equal(full_mine, full_ref)
    for crop_id, full_id in mapping.items():
        assert full_id > 7
        assert crop_id in crop_labels


def test_scale_rule_shrinks_freely_and_caps_enlargement():
    assert plaque._well_resize_scale(3000, 2000, 1500, 6) == 0.5
    assert plaque._well_resize_scale(500, 300, 1500, 6) == 3.0
    assert plaque._well_resize_scale(100, 80, 1500, 6) == 6.0
    assert plaque._well_resize_scale(100, 80, 1500, 20) == 15.0
    with pytest.raises(ValueError):
        plaque._well_resize_forward(np.zeros((10, 10)), [5, 5, 5, 9], 1500, 6)


def test_inverse_is_pixel_centre_aligned_at_integer_factors():
    """A crop enlarged 4x maps back to exactly the original label image."""
    labels = _labels((90, 120), n=12)
    crop, meta = plaque._well_resize_forward(np.zeros((90, 120), np.uint8),
                                             [0, 0, 120, 90], 480, 6)
    assert meta["ratio"] == 4.0
    enlarged = np.kron(labels, np.ones((4, 4), np.uint32))
    assert enlarged.shape == (meta["height"], meta["width"])
    np.testing.assert_array_equal(plaque._well_resize_inverse(enlarged, meta), labels)


class _Host(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class _FakeBoxes:
    def __init__(self, rows):
        rows = np.asarray(rows, float).reshape(-1, 5)
        self.xyxy = rows[:, :4].view(_Host)
        self.conf = rows[:, 4].copy().view(_Host)


class _FakeDetector:
    def __init__(self, rows):
        self.rows = rows

    def predict(self, *_a, **_k):
        return [SimpleNamespace(boxes=_FakeBoxes(self.rows))]


DETECTIONS = {
    "two_wells": ((800, 1700, 3), [(40.2, 60.7, 780.4, 770.1, .9),
                                   (900.5, 40.0, 1650.3, 760.8, .8),
                                   (5, 5, 10, 10, .9),
                                   (100, 100, 700, 200, .7)]),
    "duplicates_merge": ((900, 1000, 3), [(100, 100, 600, 620, .6),
                                          (130, 120, 640, 650, .9)]),
    "whole_frame_rule": ((1000, 1050), [(80, 70, 980, 950, .9)]),
    "nothing_found": ((500, 640, 3), []),
}


@pytest.mark.parametrize("name", sorted(DETECTIONS))
def test_well_boxes_follow_the_reference_rules(name):
    ref = _reference()
    shape, rows = DETECTIONS[name]
    image = _field(shape, np.uint8, seed=3)
    want = ref.detect_wells(image, model=_FakeDetector(rows))
    got = plaque._well_resize_boxes(image, None, detections=rows)
    assert got["fallback_to_frame"] == want["fallback_to_frame"]
    assert [b["box"] for b in got["boxes"]] == [b["box"] for b in want["boxes"]]
    assert ([b.get("whole_frame_rule", False) for b in got["boxes"]]
            == [b.get("whole_frame_rule", False) for b in want["boxes"]])
    if name == "nothing_found":
        assert got["boxes"][0]["box"] == [0, 0, shape[1], shape[0]]
    if name == "two_wells":
        assert len(got["boxes"]) == 2


def test_detector_input_matches_the_reference():
    ref = _reference()
    for image in (_field((50, 60, 3), np.uint8), _field((50, 60), np.uint16),
                  _field((40, 30), np.float32)):
        np.testing.assert_array_equal(plaque._well_detector_input(image),
                                      ref.detector_input(image))


def _real_fields(limit=10):
    data = V6_ROOT / "data"
    if not (data / "geometry.json").is_file() or not (data / "fields.csv").is_file():
        pytest.skip("plaque v6 training fields are not on this machine")
    geometry = json.loads((data / "geometry.json").read_text())["fields"]
    with open(data / "fields.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    multi = [r for r in rows if len(geometry[r["id"]]["boxes"]) > 1]
    big = sorted(rows, key=lambda r: -int(r["height"]) * int(r["width"]))[:3]
    small = sorted(rows, key=lambda r: int(r["height"]) * int(r["width"]))[:3]
    picked = {r["id"]: r for r in multi + big + small + rows[::max(1, len(rows) // 4)]}
    return [(r, geometry[r["id"]]) for r in list(picked.values())[:limit]]


def test_real_fields_round_trip_exactly_like_the_reference():
    ref = _reference()
    for row, geometry in _real_fields():
        image = ref.read_image(V6_ROOT / row["image"])
        truth = ref.read_image(V6_ROOT / row["mask"]).astype(np.uint32)
        full_ref = np.zeros(truth.shape, np.uint32)
        full_mine = np.zeros(truth.shape, np.uint32)
        for record in geometry["boxes"]:
            want_crop, want_labels, want = ref.forward(image, record["box"], RULE,
                                                       labels=truth)
            crop, meta = plaque._well_resize_forward(image, record["box"], 1500, 6)
            np.testing.assert_array_equal(crop, want_crop)
            np.testing.assert_array_equal(plaque._well_resize_intensity(crop),
                                          ref.normalize_intensity(want_crop))
            ref.paste(full_ref, want_labels, want)
            plaque._paste_well_labels(full_mine, want_labels, meta)
        np.testing.assert_array_equal(full_mine, full_ref)


class _DiskModel:
    """A Cellpose stand-in: one disk of radius 10 crop pixels per well centre."""

    def __init__(self):
        self.calls = []

    def eval(self, x, **kwargs):
        self.calls.append((np.asarray(x).shape, kwargs))
        shape = x.shape[1:] if kwargs.get("channel_axis") == 0 else x.shape[:2]
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        labels = (((yy - shape[0] // 2) ** 2 + (xx - shape[1] // 2) ** 2) <= 100
                  ).astype(np.int32) * 3
        flows = [np.zeros((*shape, 3), np.uint8), None,
                 np.full(shape, 2.0, np.float32)]
        return labels, flows, None


PLAN = {"side": 1500.0, "max_factor": 6.0, "detector": "fake", "niter": 200,
        "max_size_fraction": 0.4}


def test_multi_well_image_pastes_and_relabels_each_well(monkeypatch):
    image = _field((800, 1700, 3), np.uint8)
    rows = DETECTIONS["two_wells"][1]
    monkeypatch.setattr(plaque, "_raw_well_detections", lambda *_a: rows)
    model = _DiskModel()
    labels, flows = plaque.segment_plaque_image(
        model, image, {"minimum_object_size": 4, "diameter": 30}, well_resize=PLAN,
        return_flows=True)
    boxes = plaque._well_resize_boxes(image, None, detections=rows)["boxes"]
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    for value, record in zip((1, 2), boxes):
        x0, y0, x1, y1 = record["box"]
        ys, xs = np.nonzero(labels == value)
        assert x0 <= xs.min() and xs.max() < x1 and y0 <= ys.min() and ys.max() < y1
        assert abs(ys.mean() - (y0 + y1) / 2) < 2 and abs(xs.mean() - (x0 + x1) / 2) < 2
    for (shape, kwargs), record in zip(model.calls, boxes):
        x0, y0, x1, y1 = record["box"]
        assert shape[0] == 3 and max(shape[1:]) == 1500
        assert kwargs["normalize"] is False and kwargs["diameter"] is None
        assert kwargs["channel_axis"] == 0
        assert kwargs["min_size"] == pytest.approx(
            4 * shape[2] / (x1 - x0) * shape[1] / (y1 - y0))
        assert kwargs["niter"] == int(np.ceil(200 * 1500 / max(x1 - x0, y1 - y0)))
    assert flows["cellprob"].shape == labels.shape
    assert flows["flow_rgb"].shape == (*labels.shape, 3)
    labels_m, metrics = plaque.segment_plaque_image(
        model, image, {}, well_resize=PLAN, return_metrics=True)
    np.testing.assert_array_equal(labels_m, labels)
    assert set(metrics) == {1, 2}


def test_whole_frame_skips_detection(monkeypatch):
    monkeypatch.setattr(plaque, "_raw_well_detections",
                        lambda *_a: pytest.fail("detector must not run"))
    labels = plaque.segment_plaque_image(_DiskModel(), _field((200, 300), np.uint8), {},
                                         well_resize=PLAN, whole_frame=True)
    assert labels.shape == (200, 300) and labels.max() == 1


def test_missing_detector_resizes_the_whole_frame():
    image = _field((400, 500, 3), np.uint8)
    model = _DiskModel()
    labels = plaque.segment_plaque_image(model, image, {},
                                         well_resize=dict(PLAN, detector=None))
    assert model.calls[0][0] == (3, 1200, 1500)
    assert labels.max() == 1


V6_RECORD = {"key": "toxoplasma_plaque_v6_test", "name": "cpsam_plaque_v6_test",
             "kind": "cellpose",
             "preprocess": {"well_resize": {"side": 1500, "max_factor": 6}}}


@pytest.fixture
def zoo_with_v6(monkeypatch):
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (V6_RECORD,))
    monkeypatch.setattr(plaque, "_resolve_well_resize_detector",
                        lambda requested, fetch: "detector.pt")


def test_model_metadata_turns_well_resize_on_by_default(zoo_with_v6, tmp_path):
    assert model_zoo._model_preprocess("toxoplasma_plaque_v6_test") == V6_RECORD["preprocess"]
    assert model_zoo._model_preprocess("/somewhere/cpsam_plaque_v6_test") == V6_RECORD["preprocess"]
    assert model_zoo._model_preprocess("toxoplasma_plaque_v5") == {}
    plan = plaque._well_resize_plan({"plaque_model": "toxoplasma_plaque_v6_test"})
    assert plan["side"] == 1500 and plan["max_factor"] == 6
    assert plan["detector"] == "detector.pt"
    assert plaque._well_resize_plan({"plaque_model": "toxoplasma_plaque_v5"}) is None
    assert plaque._well_resize_plan({"plaque_model": "toxoplasma_plaque_v6_test",
                                     "plaque_well_resize": "off"}) is None
    forced = plaque._well_resize_plan({"plaque_model": "toxoplasma_plaque_v5",
                                       "plaque_well_resize": "on",
                                       "plaque_well_resize_side": 1000,
                                       "plaque_well_resize_max_factor": 3})
    assert forced["side"] == 1000 and forced["max_factor"] == 3
    assert plaque._well_resize_plan({"plaque_model": "toxoplasma_plaque_v6_test"},
                                    detect=False)["detector"] is None
    weights = tmp_path / "my_v6"
    weights.write_bytes(b"x")
    (tmp_path / "my_v6.preprocess.json").write_text(json.dumps(
        {"preprocess": {"well_resize": {"side": 1200, "max_factor": 4}}}))
    plan = plaque._well_resize_plan({"plaque_model": str(weights)})
    assert plan["side"] == 1200 and plan["max_factor"] == 4
    with pytest.raises(ValueError):
        plaque._well_resize_plan({"plaque_well_resize": "sometimes"})


def test_v5_is_segmented_exactly_as_before_when_off():
    """Off, or a model without metadata, makes the legacy single call."""
    from spacr.settings import get_analyze_plaque_settings

    settings = get_analyze_plaque_settings({})
    assert settings["plaque_model"] == "toxoplasma_plaque_v5"
    assert plaque._well_resize_plan(settings) is None
    image = _field((60, 70, 3), np.uint8)
    for extra in ({}, {"plaque_well_resize": "off"}):
        model = _DiskModel()
        plaque.segment_plaque_image(model, image, dict(settings, **extra))
        assert len(model.calls) == 1
        shape, kwargs = model.calls[0]
        assert shape == image.shape
        assert kwargs == {"channel_axis": -1, "diameter": 30.0, "flow_threshold": 0.4,
                          "cellprob_threshold": 0.0, "min_size": 0.0}


def test_settings_declare_and_persist_well_resize(tmp_path):
    from spacr import settings as S
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    defaults = S.get_analyze_plaque_settings({})
    assert defaults["plaque_well_resize"] == "auto"
    assert defaults["plaque_well_resize_side"] == 1500
    assert defaults["plaque_well_resize_max_factor"] == 6.0
    kept = S.get_analyze_plaque_settings({"plaque_well_resize": "on",
                                          "plaque_well_resize_side": 1200})
    assert kept["plaque_well_resize"] == "on" and kept["plaque_well_resize_side"] == 1200
    grouped = {key for _title, keys in _APP_CATEGORY_SPECS["analyze_plaques"]
               for key in keys}
    for key in ("plaque_well_resize", "plaque_well_resize_side",
                "plaque_well_resize_max_factor"):
        assert key in S.expected_types and key in S.tooltips and key in grouped
    from spacr.utils import load_settings, save_settings

    save_settings(dict(kept, src=str(tmp_path)), name="analyze_plaques")
    loaded = load_settings(str(tmp_path / "settings" / "analyze_plaques.csv"))
    loaded = S.get_analyze_plaque_settings(loaded)
    assert loaded["plaque_well_resize"] == "on"
    assert int(loaded["plaque_well_resize_side"]) == 1200
    assert float(loaded["plaque_well_resize_max_factor"]) == 6.0


def test_model_thresholds_are_used_in_auto_only(monkeypatch):
    record = dict(V6_RECORD, preprocess={
        "well_resize": {"side": 1500, "max_factor": 6},
        "thresholds": {"cellprob": -0.5, "flow": 0.4, "min_size": 0}})
    monkeypatch.setattr(model_zoo, "BUNDLED_REMOTE_MODELS",
                        tuple(model_zoo.BUNDLED_REMOTE_MODELS) + (record,))
    monkeypatch.setattr(plaque, "_resolve_well_resize_detector",
                        lambda requested, fetch: None)
    settings = {"plaque_model": record["key"], "CP_prob": 1.0, "flow_threshold": 0.9,
                "minimum_object_size": 50}
    plan = plaque._well_resize_plan(settings)
    assert plan["thresholds"] == {"cellprob_threshold": -0.5, "flow_threshold": 0.4,
                                  "min_size": 0.0}
    model = _DiskModel()
    plaque.segment_plaque_image(model, _field((300, 300), np.uint8), settings)
    kwargs = model.calls[0][1]
    assert kwargs["cellprob_threshold"] == -0.5 and kwargs["flow_threshold"] == 0.4
    assert kwargs["min_size"] == 0.0
    forced = plaque._well_resize_plan(dict(settings, plaque_well_resize="on"))
    assert forced["thresholds"] is None
    model = _DiskModel()
    plaque.segment_plaque_image(model, _field((300, 300), np.uint8),
                                dict(settings, plaque_well_resize="on"))
    kwargs = model.calls[0][1]
    assert kwargs["cellprob_threshold"] == 1.0 and kwargs["flow_threshold"] == 0.9
    assert kwargs["min_size"] == pytest.approx(50 * 5 * 5)
    nested = dict(V6_RECORD, key="nested_v6", name="nested_v6", preprocess={
        "well_resize": {"side": 1500, "max_factor": 6, "thresholds": {"cellprob": -1}}})
    assert plaque._declared_thresholds(
        model_zoo._model_preprocess("nested_v6", records=[nested])["well_resize"]["thresholds"]
    ) == {"cellprob_threshold": -1.0}
    with pytest.raises(ValueError):
        plaque._declared_thresholds({"cellprobability": 1})


def test_equal_channel_rgb_is_segmented_as_grey():
    grey = _field((200, 240), np.uint8)
    model = _DiskModel()
    plaque.segment_plaque_image(model, np.dstack([grey] * 3), {},
                                well_resize=dict(PLAN, detector=None))
    other = _DiskModel()
    plaque.segment_plaque_image(other, grey, {}, well_resize=dict(PLAN, detector=None))
    assert model.calls[0][0] == other.calls[0][0]
