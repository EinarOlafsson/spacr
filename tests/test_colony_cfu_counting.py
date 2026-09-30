"""Colony / CFU counting on plate photos, driven with plates of KNOWN count.

A colony count becomes a CFU/mL by a multiplication, so an error in the count
is an error of the same size in the titre. The counts here are therefore
checked against synthetic plates whose colonies were placed one by one --
isolated, touching and overlapping pairs, bright on dark agar and dark on
light agar, with a lighting gradient and noise -- rather than against the
counter's own earlier output.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import plaque
from spacr.plaque import (Well, _cfu_per_ml, _colony_count_flag,
                          _count_colony_plate, _dilution_factor, _find_dish,
                          _segment_colonies)


def _plate(n_single, n_pairs=0, *, size=900, dish=0.44, radius=(6, 12),
           seed=0, dark=False, overlap=1.0, centre=None, rgb=True):
    """A plate photo with a known number of colonies.

    :param n_single: isolated colonies.
    :param n_pairs: pairs whose centres are ``2 r overlap`` apart, so 1.0
        touches and 0.85 overlaps by 15 % of a diameter.
    :param size: side of the square photo in pixels.
    :param dish: dish radius as a fraction of ``size``.
    :param radius: colony radius range in pixels.
    :param seed: the random seed.
    :param dark: dark colonies on light agar instead of bright on dark.
    :param overlap: centre spacing of a pair, in diameters.
    :param centre: the dish centre, the photo's centre when ``None``.
    :param rgb: red agar and pale colonies in three channels, else grey.
    :returns: ``(image, count, (cx, cy, R), radii)``.
    """
    rng = np.random.default_rng(seed)
    cx, cy = centre if centre is not None else (size / 2.0, size / 2.0)
    big_r = dish * size
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    agar = (170.0 if dark else 90.0) + 25.0 * (xx - cx) / big_r
    image = np.where(rr <= big_r, agar, 30.0)
    image[(rr > big_r) & (rr < big_r + 6)] = 160.0
    placed = []

    def free(x, y, r, pad):
        if np.hypot(x - cx, y - cy) > 0.9 * big_r - r:
            return False
        return all(np.hypot(x - a, y - b) > r + c + pad for a, b, c in placed)

    def put(x, y, r):
        nonlocal image
        placed.append((x, y, r))
        d = np.hypot(xx - x, yy - y)
        disc = np.clip(r + 0.5 - d, 0, 1)
        dome = 70.0 + 40.0 * np.sqrt(np.clip(1 - (d / r) ** 2, 0, 1))
        base = agar - dome if dark else agar + dome
        new = np.minimum(image, base) if dark else np.maximum(image, base)
        image = image * (1 - disc) + new * disc

    tries = pairs = singles = 0
    while pairs < n_pairs and tries < 20000:
        tries += 1
        r = rng.uniform(*radius)
        x, y = rng.uniform(cx - big_r, cx + big_r), rng.uniform(cy - big_r, cy + big_r)
        a = rng.uniform(0, 2 * np.pi)
        x2, y2 = x + 2 * r * overlap * np.cos(a), y + 2 * r * overlap * np.sin(a)
        if free(x, y, r, 2 * r + 4) and free(x2, y2, r, 2 * r + 4):
            put(x, y, r)
            put(x2, y2, r)
            pairs += 1
    while singles < n_single and tries < 60000:
        tries += 1
        r = rng.uniform(*radius)
        x, y = rng.uniform(cx - big_r, cx + big_r), rng.uniform(cy - big_r, cy + big_r)
        if free(x, y, r, 4):
            put(x, y, r)
            singles += 1
    image = np.clip(image + rng.normal(0, 4.0, image.shape), 0, 255).astype(np.uint8)
    if rgb:
        image = np.stack([image, (image * 0.6).astype(np.uint8),
                          (image * 0.5).astype(np.uint8)], axis=-1)
    return image, singles + 2 * pairs, (cx, cy, big_r), [p[2] for p in placed]


@pytest.mark.parametrize("n_single, n_pairs, overlap, dark, seed", [
    (40, 0, 1.0, False, 1),
    (100, 20, 1.0, False, 2),
    (60, 20, 0.85, False, 3),
    (250, 25, 1.0, False, 4),
    (80, 20, 0.9, True, 5),
    (10, 10, 1.0, True, 6),
])
def test_synthetic_plates_are_counted_within_two_percent(n_single, n_pairs,
                                                          overlap, dark, seed):
    image, truth, _dish, _radii = _plate(n_single, n_pairs, overlap=overlap,
                                          dark=dark, seed=seed)
    result = _count_colony_plate(image)
    count = result["summary"]["colony_count"]
    assert abs(count - truth) <= max(1, 0.02 * truth), (count, truth)
    assert result["polarity"] == ("dark" if dark else "bright")


def test_touching_pairs_are_split_into_two_colonies():
    image, truth, _dish, _radii = _plate(0, 30, overlap=1.0, seed=11)
    assert truth == 60
    assert _count_colony_plate(image)["summary"]["colony_count"] == 60

    # Pairs overlapping by 15 % of a diameter keep a shallower waist; the
    # default split depth, chosen on real plate photos where a lower one cut
    # single colonies, separates most but not all of them.
    image, truth, _dish, _radii = _plate(0, 30, overlap=0.85, seed=12)
    found = _count_colony_plate(image)["summary"]["colony_count"]
    assert 54 <= found <= 60, found


def test_an_empty_plate_counts_nothing():
    image, truth, _dish, _radii = _plate(0, 0, seed=13)
    assert truth == 0
    result = _count_colony_plate(image, settings={"colony_dilution": 100})
    assert result["summary"]["colony_count"] == 0
    assert result["summary"]["count_flag"] == "too few to count"
    assert result["summary"]["cfu_per_ml"] == 0.0


def test_the_dish_is_found_by_its_outline():
    image, _truth, (cx, cy, big_r), _radii = _plate(30, 0, seed=14)
    well, method = _find_dish(image)
    assert method == "hough"
    assert abs((well.x0 + well.x1) / 2 - cx) < 0.03 * big_r
    assert abs((well.y0 + well.y1) / 2 - cy) < 0.03 * big_r
    assert abs(well.diameter_px / 2 - big_r) < 0.05 * big_r


def test_a_dish_clipped_by_the_photo_is_still_counted():
    image, truth, _dish, _radii = _plate(40, 0, size=900, centre=(450, 420),
                                          dish=0.47, seed=15)
    result = _count_colony_plate(image)
    assert abs(result["summary"]["colony_count"] - truth) <= 2


def test_a_grey_photo_is_counted_like_a_colour_one():
    image, truth, _dish, _radii = _plate(50, 5, seed=16, rgb=False)
    assert image.ndim == 2
    count = _count_colony_plate(image)["summary"]["colony_count"]
    assert abs(count - truth) <= max(1, 0.02 * truth)


def test_a_large_photo_is_worked_on_small_and_measured_in_its_own_pixels():
    image, truth, _dish, radii = _plate(30, 0, size=900, radius=(10, 10),
                                         seed=17)
    import cv2
    big = cv2.resize(image, (2700, 2700), interpolation=cv2.INTER_CUBIC)
    result = _count_colony_plate(big)
    assert result["factor"] < 1.0
    assert abs(result["summary"]["colony_count"] - truth) <= 1
    diameters = [row["diameter_px"] for row in result["colonies"]]
    assert np.median(diameters) == pytest.approx(3 * 2 * 10.5, rel=0.12)


def test_physical_units_come_from_the_dish_diameter():
    image, truth, (cx, cy, big_r), _radii = _plate(30, 0, radius=(10, 10),
                                                    seed=18)
    well = Well(int(cx - big_r), int(cy - big_r), int(cx + big_r), int(cy + big_r))
    result = _count_colony_plate(image, well=well,
                                 settings={"well_diameter_mm": 88.0})
    summary = result["summary"]
    px_per_mm = well.diameter_px / 88.0
    assert summary["px_per_mm"] == pytest.approx(px_per_mm)
    assert summary["scale_source"] == "explicit"
    assert summary["dish_method"] == "detector"
    row = result["colonies"][0]
    assert row["diameter_mm"] == pytest.approx(row["diameter_px"] / px_per_mm)
    assert row["area_mm2"] == pytest.approx(row["area_px"] / px_per_mm ** 2)
    assert summary["median_diameter_mm"] == pytest.approx(21.0 / px_per_mm,
                                                          rel=0.1)


def test_without_a_known_size_the_areas_stay_in_pixels():
    image, _truth, _dish, _radii = _plate(20, 0, seed=19)
    summary = _count_colony_plate(image)["summary"]
    assert summary["px_per_mm"] is None and summary["scale_source"] == "unknown"
    assert summary["mean_area_mm2"] is None
    assert summary["mean_area_px"] > 0


def test_a_manual_pixel_size_overrides_the_dish():
    image, _truth, _dish, _radii = _plate(20, 0, seed=20)
    summary = _count_colony_plate(
        image, settings={"plaque_pixels_per_um": 0.01,
                         "well_diameter_mm": 88.0})["summary"]
    assert summary["px_per_mm"] == pytest.approx(10.0)
    assert summary["scale_source"] == "manual settings"


def test_cfu_per_ml_is_count_times_dilution_over_volume():
    assert _cfu_per_ml(150, 1e4, 100) == pytest.approx(1.5e7)
    assert _cfu_per_ml(150, 1e-4, 100) == pytest.approx(1.5e7)
    assert _cfu_per_ml(42, 1, 1000) == pytest.approx(42.0)
    assert _cfu_per_ml(150, None, 100) is None
    assert _cfu_per_ml(150, 1e4, 0) is None
    assert _cfu_per_ml(150, -10, 100) is None
    assert _dilution_factor("1e3") == pytest.approx(1000.0)
    assert _dilution_factor("ten") is None


def test_plates_outside_the_countable_window_are_flagged():
    assert _colony_count_flag(301) == "too many to count"
    assert _colony_count_flag(300) == "countable"
    assert _colony_count_flag(30) == "countable"
    assert _colony_count_flag(29) == "too few to count"
    assert _colony_count_flag(120, too_many=100) == "too many to count"
    assert _colony_count_flag(12, too_few=10) == "countable"
    assert _colony_count_flag(5000, too_many=None, too_few="") == "countable"


def test_the_flag_and_titre_follow_the_settings():
    image, truth, _dish, _radii = _plate(40, 0, seed=21)
    summary = _count_colony_plate(image, name="p1.png", settings={
        "colony_too_many": 35, "colony_dilution": {"p1": 1e-3},
        "colony_plated_volume_ul": 50})["summary"]
    count = summary["colony_count"]
    assert summary["count_flag"] == "too many to count"
    assert summary["dilution"] == pytest.approx(1000.0)
    assert summary["cfu_per_ml"] == pytest.approx(count * 1000 / 0.05)

    unnamed = _count_colony_plate(image, name="other.png", settings={
        "colony_dilution": {"p1": 1e-3}})["summary"]
    assert unnamed["cfu_per_ml"] is None and unnamed["dilution"] is None


def test_an_unknown_polarity_is_refused():
    image, _truth, _dish, _radii = _plate(5, 0, seed=22)
    with pytest.raises(ValueError, match="polarity"):
        _segment_colonies(image, polarity="sideways")


def test_a_larger_minimum_area_drops_the_small_colonies():
    image, truth, _dish, radii = _plate(40, 0, radius=(4, 14), seed=23)
    small = sum(r < 8 for r in radii)
    assert 0 < small < truth
    kept = _count_colony_plate(image, settings={
        "colony_min_area_px": np.pi * 8.5 ** 2})["summary"]["colony_count"]
    assert abs(kept - (truth - small)) <= 2


def test_the_run_writes_tables_and_figures_through_the_one_writers(tmp_path,
                                                                    monkeypatch):
    from cellpose import io as cp_io
    from spacr import submodules

    plates = {}
    for index, (n, pairs) in enumerate(((40, 5), (120, 10))):
        image, truth, _dish, _radii = _plate(n, pairs, seed=30 + index)
        name = f"plate{index}.png"
        cp_io.imsave(str(tmp_path / name), image)
        plates[name] = truth

    def no_model(_settings, **_kwargs):
        raise AssertionError("colony counting must not resolve a plaque model")

    monkeypatch.setattr(submodules, "_resolve_plaque_model", no_model)
    table = submodules.analyze_plaques({
        "src": str(tmp_path), "colony_counting": True, "colony_dilution": 1e4,
        "colony_plated_volume_ul": 100, "well_diameter_mm": 86.0,
        "save": True})
    out = tmp_path / "colonies"
    with sqlite3.connect(out / "colonies.db") as conn:
        per_plate = pd.read_sql("select * from per_plate", conn)
        per_colony = pd.read_sql("select * from per_colony", conn)
    assert sorted(per_plate["file"]) == sorted(plates)
    for _, row in per_plate.iterrows():
        truth = plates[row["file"]]
        assert abs(row["colony_count"] - truth) <= max(1, 0.02 * truth)
        assert row["cfu_per_ml"] == pytest.approx(row["colony_count"] * 1e4 / 0.1)
        assert row["px_per_mm"] > 0
    assert set(per_plate["count_flag"]) == {"countable"}
    assert len(per_colony) == int(per_plate["colony_count"].sum())
    assert per_colony["diameter_mm"].notna().all()
    assert len(table) == 2
    assert (out / "per_plate.csv").is_file()
    figures = sorted(p.stem for p in out.iterdir()
                     if p.suffix.lower() in (".pdf", ".png", ".svg", ".tif", ".tiff", ".jpg"))
    assert figures == ["colony_sizes", "plate0_colonies", "plate1_colonies"]


def test_figure_mode_ignores_colony_counting(tmp_path, monkeypatch):
    from spacr import submodules

    calls = []
    monkeypatch.setattr(submodules, "_resolve_plaque_model",
                        lambda settings, **kw: "model.pt")
    monkeypatch.setattr(submodules, "_analyze_plaque_figures",
                        lambda settings, model: calls.append("figure") or {})
    monkeypatch.setattr(submodules, "_analyze_colony_plates",
                        lambda settings: calls.append("colony"))
    submodules.analyze_plaques({"src": str(tmp_path), "plaque_mode": "figure",
                                "colony_counting": True})
    assert calls == ["figure"]


def test_a_missing_detector_package_falls_back_to_the_dish_outline(
        tmp_path, monkeypatch):
    from cellpose import io as cp_io
    from spacr import submodules

    image, truth, _dish, _radii = _plate(30, 0, seed=40)
    cp_io.imsave(str(tmp_path / "plate.png"), image)

    def missing(*_args, **_kwargs):
        raise ImportError("Well detection needs the 'ultralytics' package")

    monkeypatch.setattr(submodules, "_resolve_well_detector", lambda s: "w.pt")
    monkeypatch.setattr(plaque, "detect_wells", missing)
    table = submodules._analyze_colony_plates({
        "src": str(tmp_path), "save": False, "well_detection": True})
    assert table.loc[0, "dish_method"] == "hough"
    assert abs(table.loc[0, "colony_count"] - truth) <= 1


def test_detected_wells_are_counted_one_row_each(tmp_path, monkeypatch):
    from cellpose import io as cp_io
    from spacr import submodules

    left, n_left, _d, _r = _plate(20, 0, size=400, seed=41)
    right, n_right, _d, _r = _plate(35, 0, size=400, seed=42)
    cp_io.imsave(str(tmp_path / "strip.png"), np.concatenate([left, right], axis=1))
    wells = [Well(24, 24, 376, 376, 0.9), Well(424, 24, 776, 376, 0.9)]
    monkeypatch.setattr(submodules, "_resolve_well_detector", lambda s: "w.pt")
    monkeypatch.setattr(plaque, "detect_wells", lambda *a, **k: list(wells))
    table = submodules._analyze_colony_plates({
        "src": str(tmp_path), "save": False, "well_detection": True,
        "plate_format": "6-well"})
    assert list(table["well"]) == [1, 2]
    assert list(table["dish_method"]) == ["detector", "detector"]
    assert abs(table.loc[0, "colony_count"] - n_left) <= 1
    assert abs(table.loc[1, "colony_count"] - n_right) <= 1
    assert table.loc[0, "px_per_mm"] == pytest.approx(352 / 34.8)


class _Boxes:
    """The part of an ultralytics ``Boxes`` the colony detector reads."""

    def __init__(self, rows):
        rows = np.asarray(rows, np.float32).reshape(-1, 5)
        self.xyxy, self.conf = rows[:, :4], rows[:, 4]

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class _FakeColonyDetector:
    """A detector that returns fixed boxes and records how it was asked."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def predict(self, source, conf, imgsz, iou, max_det, verbose):
        self.calls.append(dict(shape=np.asarray(source).shape, conf=conf,
                               imgsz=imgsz, iou=iou, max_det=max_det))
        return [_Result([r for r in self.rows if r[4] >= conf])]


_DETECTED = [
    (100, 100, 120, 120, 0.9),
    (112, 100, 132, 120, 0.8),
    (250, 250, 260, 280, 0.6),
    (200, 200, 220, 220, 0.1),
    (2, 2, 12, 12, 0.95),
]
"""Two overlapping colonies, an elongated one, one below the default score
and one outside the dish, in the pixels of the dish crop."""


def test_a_colony_detector_counts_every_box_inside_the_dish(monkeypatch):
    fake = _FakeColonyDetector(_DETECTED)
    monkeypatch.setattr(plaque, "_load_detector", lambda weights: fake)
    image = np.full((400, 400, 3), 90, np.uint8)
    result = _count_colony_plate(
        image, well=Well(20, 20, 380, 380),
        settings={"colony_detector": "colonies.pt",
                  "plaque_pixels_per_um": 0.01})
    summary = result["summary"]
    assert summary["colony_count"] == 3
    assert summary["polarity"] == "detector"
    assert fake.calls[0]["shape"] == (360, 360, 3)
    assert fake.calls[0]["max_det"] >= 1000
    rows = sorted(result["colonies"], key=lambda r: r["centroid_x"])
    assert rows[0]["area_px"] == pytest.approx(np.pi / 4 * 20 * 20)
    assert rows[0]["diameter_px"] == pytest.approx(20)
    assert rows[0]["centroid_x"] == pytest.approx(110 + 20)
    assert rows[0]["diameter_mm"] == pytest.approx(20 / 10.0)
    assert rows[2]["eccentricity"] == pytest.approx(np.sqrt(1 - (10 / 30) ** 2))
    assert len(np.unique(result["labels"])) - 1 == 3


def test_a_grey_photo_reaches_the_detector_as_three_byte_channels(monkeypatch):
    fake = _FakeColonyDetector([(100, 100, 120, 120, 0.9)])
    monkeypatch.setattr(plaque, "_load_detector", lambda weights: fake)
    found = plaque._detect_colonies(np.full((400, 400), 3000, np.uint16),
                                    "colonies.pt", centre=(200, 200),
                                    radius=190)
    assert fake.calls[0]["shape"] == (400, 400, 3)
    assert len(found["boxes"]) == 1


def test_the_run_counts_with_the_colony_detector_when_it_is_set(
        tmp_path, monkeypatch):
    import importlib.util
    from cellpose import io as cp_io
    from spacr import submodules

    image, _truth, _dish, _radii = _plate(30, 0, size=400, seed=43)
    cp_io.imsave(str(tmp_path / "plate.png"), image)
    weights = tmp_path / "colonies.pt"
    weights.write_bytes(b"weights")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        object() if name == "ultralytics" else real_find_spec(name, *a)))
    loaded = []
    fake = _FakeColonyDetector([(150, 150, 170, 170, 0.9),
                                (200, 200, 216, 216, 0.7)])
    monkeypatch.setattr(plaque, "_load_detector",
                        lambda path: loaded.append(path) or fake)
    table = submodules._analyze_colony_plates({
        "src": str(tmp_path), "save": False,
        "colony_detector": str(weights)})
    assert loaded == [str(weights)]
    assert table.loc[0, "colony_count"] == 2
    assert table.loc[0, "polarity"] == "detector"


def test_without_ultralytics_the_run_thresholds_instead(tmp_path, monkeypatch):
    import importlib.util
    from cellpose import io as cp_io
    from spacr import submodules

    image, truth, _dish, _radii = _plate(30, 0, seed=44)
    cp_io.imsave(str(tmp_path / "plate.png"), image)
    weights = tmp_path / "colonies.pt"
    weights.write_bytes(b"weights")
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        None if name == "ultralytics" else real_find_spec(name, *a)))

    def never(path):
        raise AssertionError("the detector must not be loaded")

    monkeypatch.setattr(plaque, "_load_detector", never)
    table = submodules._analyze_colony_plates({
        "src": str(tmp_path), "save": False,
        "colony_detector": str(weights)})
    assert table.loc[0, "polarity"] in ("bright", "dark")
    assert abs(table.loc[0, "colony_count"] - truth) <= 1


def test_a_colony_detector_that_is_neither_file_nor_zoo_key_is_refused(
        tmp_path, monkeypatch):
    from spacr import model_zoo, submodules

    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [])
    with pytest.raises(ValueError, match="colony_detector="):
        submodules._analyze_colony_plates({
            "src": str(tmp_path), "save": False,
            "colony_detector": "no_such_detector"})


def test_the_published_colony_detector_is_fetched_by_its_zoo_key(
        tmp_path, monkeypatch):
    """The zoo key downloads the checkpoint once, from the dataset repo."""
    import dataclasses
    import hashlib

    from spacr import model_zoo, submodules

    record = next(r for r in model_zoo.BUNDLED_REMOTE_MODELS
                  if r["key"] == "colony_yolo11n_makrai_v1")
    entry = model_zoo._entry_from_mapping(record)
    assert entry.kind == "detector"
    assert entry.uri.startswith(
        "https://huggingface.co/datasets/einarolafsson/models/resolve/main/"
        "colony_detector/v1/colony_yolo11n_makrai_v1.pt")
    assert len(entry.sha256) == 64
    payload = b"fake colony weights"
    stand_in = dataclasses.replace(
        entry, sha256=hashlib.sha256(payload).hexdigest(), size_bytes=0)
    monkeypatch.setattr(model_zoo, "catalogue",
                        lambda remote=True: [stand_in])
    asked = []
    monkeypatch.setattr(model_zoo, "open_uri", lambda uri, **kw: (
        asked.append(uri) or iter([payload])))
    monkeypatch.setenv("HOME", str(tmp_path))
    path = submodules._resolve_detector_weights(
        "colony_yolo11n_makrai_v1", "colony_detector")
    assert asked == [entry.uri]
    assert path.startswith(str(tmp_path / ".spacr" / "models"))
    with open(path, "rb") as handle:
        assert handle.read() == payload
    assert submodules._resolve_detector_weights(path, "colony_detector") == path
    assert asked == [entry.uri]


def test_the_colony_detector_zoo_row_is_alpha():
    from spacr.settings import ALPHA_FEATURES

    assert "colony_yolo11n_makrai_v1" in ALPHA_FEATURES[542]["models"]
