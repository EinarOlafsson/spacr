"""Scratch / wound-healing closure measured in Measure.

The synthetic wounds are bands of exactly known width and area with ragged
edges, on a brightfield-like or fluorescent monolayer, closing at a known
rate; each source is held to within two per cent of the first frame's open
area and the half-closure time to within a tenth of a frame. The end-to-end
test runs ``measure_crop`` on a two-well time-lapse and reads the four
tables, the CSV exports and the figures back.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from spacr import measure
from spacr.measure import (
    _WOUND_CLOSURE_TABLE,
    _WOUND_CONDITION_TABLE,
    _WOUND_TABLE,
    _WOUND_WELL_TABLE,
    _closure_metrics,
    _measure_field_wound,
    _wound_by_well,
    _wound_closure_summary,
    _wound_condition_lookup,
    _wound_series,
    _wound_settings_check,
)

SHAPE = (384, 384)
WIDTHS = (160, 130, 100, 70, 40, 0)


def _band(width, angle=0.0, shape=SHAPE, seed=3):
    """A scratch ``width`` pixels wide with smoothly ragged edges."""
    h, w = shape
    yy, xx = np.indices(shape, dtype=float)
    a = np.deg2rad(angle)
    along = (yy - h / 2) * np.cos(a) + (xx - w / 2) * np.sin(a)
    across = -(yy - h / 2) * np.sin(a) + (xx - w / 2) * np.cos(a)
    rng = np.random.default_rng(seed)
    n = int(np.hypot(h, w)) + 4
    left = ndi.gaussian_filter1d(rng.standard_normal(n), 12) * 20
    right = ndi.gaussian_filter1d(rng.standard_normal(n), 12) * 20
    idx = np.clip((along + n / 2).astype(int), 0, n - 1)
    return (across > -width / 2 + left[idx]) & (across < width / 2 + right[idx])


def _brightfield(open_mask, seed=1):
    """Flat plastic with camera noise; cells add fine-grained texture."""
    rng = np.random.default_rng(seed)
    texture = ndi.gaussian_filter(rng.standard_normal(open_mask.shape), 1.5)
    texture /= texture.std()
    image = 1000 + rng.normal(0, 8, open_mask.shape) + (
        ~open_mask) * texture * 120
    return np.clip(image, 0, 65535).astype(np.uint16)


def _fluorescent(open_mask, seed=2):
    """Dark open plastic; a cytoplasmic stain over the monolayer."""
    rng = np.random.default_rng(seed)
    texture = ndi.gaussian_filter(rng.standard_normal(open_mask.shape), 2.0)
    texture /= texture.std()
    image = 100 + rng.normal(0, 10, open_mask.shape) + (~open_mask) * (
        600 + 120 * texture)
    return np.clip(image, 0, 65535).astype(np.uint16)


def _labels(open_mask):
    """Cell labels covering everything but the wound."""
    labels, _count = ndi.label(~open_mask)
    return labels.astype(np.uint16)


def _series(source, angle=0.0, widths=WIDTHS):
    truths = [_band(width, angle) for width in widths]
    if source == 'texture':
        planes = [_brightfield(t, seed=10 + i) for i, t in enumerate(truths)]
    elif source == 'intensity':
        planes = [_fluorescent(t, seed=10 + i) for i, t in enumerate(truths)]
    else:
        planes = [_labels(t) for t in truths]
    return truths, planes


@pytest.mark.parametrize("source,angle", [
    ("texture", 0), ("texture", 20), ("texture", 90),
    ("intensity", 0), ("intensity", 20), ("masks", 0)])
def test_known_wounds_are_measured_to_two_percent(source, angle):
    truths, planes = _series(source, angle)
    frame, status, _masks = _wound_series(
        planes, range(len(planes)), source=source)
    assert status == "ok"
    areas = np.array([t.sum() for t in truths], dtype=float)
    error = np.abs(frame["open_area_px"].to_numpy() - areas) / areas[0]
    assert error.max() <= 0.02, error
    true_relative = areas / areas[0]
    assert np.abs(frame["relative_open_area"] - true_relative).max() <= 0.02
    truth = _closure_metrics(range(len(planes)), true_relative)
    found = _closure_metrics(frame["time"], frame["relative_open_area"])
    assert abs(found["half_closure_time"] - truth["half_closure_time"]) <= 0.1
    assert found["closure_rate"] == pytest.approx(truth["closure_rate"],
                                                  rel=0.06)


def test_a_gap_narrower_than_the_texture_window_reads_as_closing():
    """Texture cannot resolve a gap much narrower than its window.

    A 15-pixel scratch with ragged edges is mostly bridged; what the
    texture source still finds open is never more than is really open.
    """
    truths, planes = _series("texture", 0, widths=(160, 15))
    frame, _status, _masks = _wound_series(planes, (0, 1), source="texture")
    assert frame["open_area_px"].iloc[1] <= truths[1].sum()


@pytest.mark.parametrize("angle", [0, 20])
def test_the_width_is_read_across_the_scratch_whatever_its_angle(angle):
    truths, planes = _series("masks", angle, widths=(120, 60))
    frame, status, _masks = _wound_series(planes, (0, 1), source="masks")
    assert status == "ok"
    for row, width in zip(frame.itertuples(), (120, 60)):
        assert abs(row.mean_width_px - width) <= 0.05 * width
        assert row.min_width_px < row.mean_width_px < row.max_width_px


def test_a_bridged_wound_has_zero_minimum_width():
    open_mask = _band(100)
    bridged = open_mask.copy()
    bridged[180:200] = False
    frame, _status, _masks = _wound_series(
        [_labels(open_mask), _labels(bridged)], (0, 1), source="masks")
    assert frame["min_width_px"].iloc[0] > 50
    assert frame["min_width_px"].iloc[1] == 0
    assert frame["n_regions"].iloc[1] == 2


def test_gaps_in_the_monolayer_beside_the_wound_are_not_wound():
    open_mask = _band(80)
    later = _band(40)
    later[20:60, 10:50] = True
    frame, _status, masks = _wound_series(
        [_labels(open_mask), _labels(later)], (0, 1), source="masks",
        keep=(1,))
    assert frame["open_area_px"].iloc[1] == pytest.approx(
        _band(40).sum(), rel=0.01)
    assert not masks[1][1][20:60, 10:50].any()


def test_cells_inside_the_wound_count_as_open():
    open_mask = _band(120)
    islands = open_mask.copy()
    islands[100:110, 185:195] = False
    frame, _status, _masks = _wound_series(
        [_labels(islands)], (0,), source="masks")
    assert frame["open_area_px"].iloc[0] == open_mask.sum()


def test_a_field_without_a_scratch_is_flagged_not_measured():
    confluent = np.zeros(SHAPE, dtype=bool)
    frame, status, _masks = _wound_series(
        [_labels(confluent)] * 2, (0, 1), source="masks")
    assert status == "no_wound"
    assert frame["relative_open_area"].isna().all()

    blob = np.zeros(SHAPE, dtype=bool)
    blob[150:230, 150:230] = True
    _frame, status, _masks = _wound_series(
        [_labels(blob)], (0,), source="masks")
    assert status == "not_a_scratch"


def test_debris_floating_in_a_fresh_wound_is_drawn_over_by_its_fronts():
    """Textured debris over a stretch of a fresh wound still counts as open."""
    open_mask = _band(120)
    image = _brightfield(open_mask).astype(float)
    rng = np.random.default_rng(5)
    debris = np.zeros(SHAPE, dtype=bool)
    debris[40:140] = open_mask[40:140]
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 1.5)
    texture /= texture.std()
    frame, status, _masks = _wound_series(
        [image + debris * texture * 120], (0,), source="texture")
    assert status == "ok"
    found = frame["open_area_px"].iloc[0] / open_mask.sum()
    assert 0.85 <= found <= 1.02, found


@pytest.mark.parametrize("noise,contrast", [(8, 40), (20, 120), (4, 300)])
def test_a_later_frame_imaged_again_keeps_its_wound(noise, contrast):
    """A later time point with other exposure or focus is recalibrated.

    With the first frame's cut, a monolayer imaged at a third of its first
    contrast reads as wound-free; the later frame's own open and covered
    levels put the cut back between them.
    """
    first, later = _band(160), _band(80)
    rng = np.random.default_rng(11)
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 1.5)
    texture /= texture.std()
    image = 1000 + rng.normal(0, noise, SHAPE) + (~later) * texture * contrast
    frame, status, _masks = _wound_series(
        [_brightfield(first, seed=10), image], (0, 1), source="texture")
    assert status == "ok"
    truth = later.sum() / first.sum()
    assert abs(frame["relative_open_area"].iloc[1] - truth) <= 0.03


def _straight(centre, half, shape=SHAPE):
    """A straight horizontal scratch, rows ``centre - half`` to ``+ half``."""
    rows = np.arange(shape[0])[:, None]
    return np.broadcast_to(np.abs(rows - centre) < half, shape).copy()


def test_a_flat_patch_of_monolayer_beside_a_closing_wound_is_not_wound():
    """Open-looking monolayer outside the first wound is not counted later.

    A later frame whose monolayer has a flat, textureless patch (glare or
    an over-exposed stretch) inside the first frame's band, but outside
    where the wound was, would otherwise add that patch to the wound.
    """
    first, later = _straight(192, 80), _straight(192, 30)
    looks_open = later.copy()
    looks_open[290:312, 60:330] = True
    frame, status, masks = _wound_series(
        [_brightfield(first, seed=10), _brightfield(looks_open, seed=11)],
        (0, 1), source="texture", keep=(1,))
    assert status == "ok"
    truth = later.sum() / first.sum()
    assert abs(frame["relative_open_area"].iloc[1] - truth) <= 0.02
    assert not masks[1][1][294:308, 80:310].any()


def test_a_later_frame_imaged_at_another_position_keeps_its_wound():
    """A scratch re-imaged off-centre is followed across the field."""
    first, later = _straight(192, 70), _straight(252, 40)
    frame, status, _masks = _wound_series(
        [_brightfield(first, seed=10), _brightfield(later, seed=11)],
        (0, 1), source="texture")
    assert status == "ok"
    truth = later.sum() / first.sum()
    assert abs(frame["relative_open_area"].iloc[1] - truth) <= 0.02


def test_closure_metrics_on_a_known_curve():
    times = np.array([0, 4, 8, 12, 16, 20])
    relative = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    widths = 400 * relative
    metrics = _closure_metrics(times, relative, widths)
    assert metrics["half_closure_time"] == pytest.approx(10.0)
    assert metrics["half_closure_reached"] == 1
    assert metrics["closure_rate"] == pytest.approx(0.05, rel=0.02)
    assert metrics["width_rate"] == pytest.approx(20.0, rel=0.02)
    assert metrics["front_speed"] == pytest.approx(10.0, rel=0.02)
    assert metrics["reopened"] == 0

    slow = _closure_metrics([0, 10, 20], [1.0, 0.8, 0.7])
    assert np.isnan(slow["half_closure_time"])
    assert slow["half_closure_reached"] == 0
    assert _closure_metrics([0, 1, 2], [1.0, 0.4, 0.9])["reopened"] == 1


def test_conditions_are_read_in_the_well_vocabulary():
    lookup = _wound_condition_lookup({"control": "c1", "drug": ["B03", "r4c7"]})
    assert lookup[(5, 1)] == "control"
    assert lookup[(2, 3)] == "drug" and lookup[(4, 7)] == "drug"
    with pytest.raises(ValueError, match="one condition only"):
        _wound_condition_lookup({"a": "c1", "b": "A01"})
    with pytest.raises(ValueError, match="wound_conditions"):
        _wound_condition_lookup({"a": "well-9"})


def test_settings_are_checked_before_a_run():
    assert _wound_settings_check({}) == "texture"
    with pytest.raises(ValueError, match="wound_source"):
        _wound_settings_check({"wound_source": "edges"})
    with pytest.raises(ValueError, match="cell_mask_dim"):
        _wound_settings_check({"wound_source": "masks",
                               "cell_mask_dim": None})


def test_wells_pool_their_fields_by_open_area():
    fields = pd.DataFrame({
        "plateID": "p1", "rowID": "r1", "columnID": "c1",
        "fieldID": ["1", "1", "2", "2"], "timeID": [0, 1, 0, 1],
        "time": [0.0, 1.0, 0.0, 1.0], "time_unit": "frame",
        "open_area_px": [300, 150, 100, 100],
        "start_open_area_px": [300, 300, 100, 100],
        "mean_width_px": [30.0, 15.0, 10.0, 10.0],
        "min_width_px": [20.0, 5.0, 8.0, 8.0],
        "mean_width_um": np.nan, "min_width_um": np.nan,
        "open_area_um2": np.nan, "status": "ok",
    })
    curves = _wound_by_well(fields)
    assert curves["relative_open_area"].tolist() == [1.0, 250 / 400]
    assert curves["min_width_px"].tolist() == [8.0, 5.0]
    summary, curves = _wound_closure_summary(curves, fields,
                                             {"control": "A01"})
    assert summary["condition"].tolist() == ["control"]
    assert summary["n_fields_ok"].iloc[0] == 2


def test_the_preview_measures_one_field_as_a_first_frame():
    open_mask = _band(100)
    data = np.stack([_brightfield(open_mask), _labels(open_mask)], axis=-1)
    row, plane, wound, status = _measure_field_wound(
        data, {"channels": [0], "wound_source": "texture",
               "voxel_size_xy_um": 0.5})
    assert status == "ok" and plane.shape == SHAPE
    assert abs(row["open_area_px"] - open_mask.sum()) <= 0.02 * open_mask.sum()
    assert row["mean_width_um"] == pytest.approx(0.5 * row["mean_width_px"])
    assert wound.dtype == bool


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0], "cell_mask_dim": 1,
        "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": True, "verbose": False, "n_jobs": 1, "timelapse": True,
        "wound_closure": True, "wound_hours_per_frame": 4.0,
        "wound_conditions": {"fast": "A01", "slow": "B02"},
    })
    settings.update(over)
    return settings


WELLS = {"plate1_A01": (150, 100, 50, 0), "plate1_B02": (150, 130, 110, 90)}


def _write_plate(root):
    """Two wells, one field each, four frames: brightfield and cell masks."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    for well, widths in WELLS.items():
        for time, width in enumerate(widths):
            open_mask = _band(width)
            stack = np.stack([_brightfield(open_mask, seed=time),
                              _labels(open_mask)], axis=-1)
            np.save(merged / f"{well}_1_{time}.npy", stack)
    return merged


def test_measure_writes_closure_per_field_well_and_condition(tmp_path):
    merged = _write_plate(tmp_path)
    measure.measure_crop(_settings(merged))

    db = tmp_path / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        fields = pd.read_sql_query(f"SELECT * FROM {_WOUND_TABLE}", conn)
        wells = pd.read_sql_query(f"SELECT * FROM {_WOUND_WELL_TABLE}", conn)
        summary = pd.read_sql_query(
            f"SELECT * FROM {_WOUND_CLOSURE_TABLE}", conn)
        conditions = pd.read_sql_query(
            f"SELECT * FROM {_WOUND_CONDITION_TABLE}", conn)

    assert len(fields) == 8 and set(fields["status"]) == {"ok"}
    assert fields["time"].max() == 12.0 and set(fields["time_unit"]) == {"h"}
    assert len(wells) == 8
    summary = summary.set_index("condition")
    fast, slow = summary.loc["fast"], summary.loc["slow"]
    assert fast["half_closure_reached"] == 1 and fast["wound_ok"] == 1
    assert fast["half_closure_time"] == pytest.approx(6.0, abs=0.6)
    assert slow["half_closure_reached"] == 0
    assert fast["closure_rate"] > slow["closure_rate"] > 0
    assert fast["width_rate"] == pytest.approx(50 / 4, rel=0.1)
    assert set(conditions["condition"]) == {"fast", "slow"}

    out = tmp_path / "results" / "wound_closure"
    assert (out / "wound_closure_per_well.csv").is_file()
    assert (out / "wound_closure_per_condition.csv").is_file()
    written = {name for name in os.listdir(out)}
    for stem in ("closure_curves", "half_closure_time",
                 "plate_plate1_half_closure"):
        assert any(name.startswith(stem) for name in written), written
    assert os.listdir(out / "fields")


def test_the_wound_settings_are_alpha():
    from spacr.settings import ALPHA_FEATURES, _is_alpha, categories

    entry = ALPHA_FEATURES[536]
    assert set(entry["settings"]) == set(categories["Wound Closure α"])
    for key in entry["settings"]:
        assert _is_alpha("settings", key)
    assert entry["widgets"] == ("MeasureWoundToggle",)
