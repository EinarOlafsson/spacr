"""Item 1035: OPS phenotype segmentation on flat-field corrected inputs.

Cellpose is replaced by the threshold model of the phenotype coverage tests
(objects are the pixels above half the image maximum), so a dim, vignetted
corner object is lost on raw pixels and found once the field is divided by
its flat-field map. Everything else is the shipped engine: the maps folder,
the option marker, the phase order and the settings replay.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import cli, ops_engine, ops_flatfield
from spacr.ops_settings import OPS_DEFAULTS
from spacr.ops_store import write_table
from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, SETTINGS_VERSION_KEY

PLATE = "P1"
SIZE = 96
OFFSET = 100.0
#: Disc centres; the last sits in the dim corner of the vignette.
CENTRES = ((48, 48), (30, 66), (66, 30), (10, 10))
SETTINGS = {"ops_ph_channels": "DAPI,AF750", "ops_ph_downsample": 1, "ops_flatfield": "none",
            "ops_ph_texture": False, "n_workers": 1, "ops_ph_image_format": "npz"}


class _ThresholdModel:
    """A stand-in for Cellpose: labels the last channel above half its maximum."""

    def __init__(self, grow=0):
        self.grow = grow

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label the bright regions; cells are grown around them."""
        plane = np.asarray(image, float)
        if plane.ndim == 3:
            plane = plane[..., -1]
        labels, _n = ndimage.label(plane > OFFSET + (plane.max() - OFFSET) / 2.0)
        if self.grow:
            from skimage.segmentation import expand_labels

            labels = expand_labels(labels, self.grow)
        return labels, None, None


@pytest.fixture
def models(monkeypatch):
    """The threshold model in place of Cellpose."""
    def fake(settings, gpu, key="cellpose_model"):
        return _ThresholdModel(grow=0 if "nucleus" in key else 4)

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)


def _flat_map(grid=12):
    """A vignette on the map grid: 1 at the centre, 0.1 in the corners."""
    yy, xx = np.mgrid[:grid, :grid] / (grid - 1) * 2 - 1
    return np.clip(1.0 - 0.55 * (yy ** 2 + xx ** 2), 0.1, None).astype(np.float32)


def _maps_folder(folder, *, pending=()):
    """A plate's maps folder as ph_flatfield writes it, some channels pending."""
    maps = {"DAPI": _flat_map(), "AF750": _flat_map()}
    method = {c: {"how": "test", "offset": OFFSET, "centre": 1.0,
                  "flat_min": float(m.min()), "flat_max": float(m.max())} for c, m in maps.items()}
    extra = {}
    if pending:
        extra["pending"] = [ops_flatfield.mark_pending(maps, method, c, "nuclear_median")
                            for c in pending]
    ops_flatfield.write_flat_maps(str(folder), PLATE, maps, method, [("A1", 0)], extra=extra)
    return folder


def _complete(folder):
    """Write the pending maps as ph_flatfield_objects does."""
    for channel in ("DAPI", "AF750"):
        ops_flatfield.complete_pending_flat(
            str(folder), PLATE, channel, _flat_map(),
            {"estimator": "nuclear_median", "offset": OFFSET})


def _field(seed=0):
    """DNA and cell planes of four equal discs, seen through the vignette."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    dapi = np.full((SIZE, SIZE), OFFSET) + rng.normal(0, 1, (SIZE, SIZE))
    body = np.full((SIZE, SIZE), OFFSET) + rng.normal(0, 1, (SIZE, SIZE))
    for cy, cx in CENTRES:
        distance = np.hypot(yy - cy, xx - cx)
        dapi[distance < 4] += 400
        body[distance < 8] += 200
    plane = ops_flatfield.flat_planes({"DAPI": _flat_map()}, (SIZE, SIZE))["DAPI"]
    seen = [OFFSET + plane * (p - OFFSET) for p in (dapi, body)]
    return [np.clip(p, 0, None).astype(np.uint16) for p in seen]


def _write_fields(folder, sites=(0, 1)):
    folder.mkdir(parents=True, exist_ok=True)
    for site in sites:
        dapi, body = _field(site)
        tifffile.imwrite(folder / f"20X_DAPI-AF750_A1_DAPI_Site-{site}.tif", dapi)
        tifffile.imwrite(folder / f"20X_DAPI-AF750_A1_AF750_Site-{site}.tif", body)
    return folder


def _placed(db, sites=(0, 1)):
    import pandas as pd

    frame = pd.DataFrame([{
        "plate": PLATE, "well": "A1", "site": site,
        "centre_y": 1000.0 + 100 * site, "centre_x": 1000.0,
        "sbs_site": 0, "is_anchor": int(site == 0)} for site in sites])
    frame = ops_engine._ph_transform_columns(frame, 0.25, 0.0, (SIZE, SIZE))
    write_table(db, "ops_phenotype", frame)


def _run(tmp_path, settings):
    tiles = _write_fields(tmp_path / "pheno")
    db = str(tmp_path / "measurements.db")
    _placed(db)
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    report = ops_engine._ph_fields(db, PLATE, "A1", index["A1"], settings, False,
                                   str(tmp_path), segment=True, measure=False)
    masks = {}
    for site in (0, 1):
        with np.load(tmp_path / "A1" / "phenotype_masks" / f"site-{site}.npz") as saved:
            masks[site] = {k: saved[k] for k in saved.files}
    marker = json.loads((tmp_path / "A1" / "phenotype_masks" / "part_options.json").read_text())
    return report, masks, marker


@pytest.fixture
def flats(monkeypatch):
    """Records whether each segmentation call was given flat-field planes."""
    seen = []
    real = ops_engine._ph_segment_field

    def spy(stack, channels, models, settings, *, flat=None, **kwargs):
        seen.append(flat)
        return real(stack, channels, models, settings, flat=flat, **kwargs)

    monkeypatch.setattr(ops_engine, "_ph_segment_field", spy)
    return seen


def test_off_segments_the_raw_pixels_with_the_old_marker(tmp_path, models, flats):
    """False: no planes reach the segmentation and the marker has no new key, maps or not."""
    _maps_folder(tmp_path / "flatfield")
    report, masks, marker = _run(tmp_path, {**SETTINGS, "ops_ph_segment_flatfield": False})
    assert flats == [None, None]
    assert "segment_flatfield" not in marker
    assert report["ph_segment"]["objects"]["paired"] == 2 * (len(CENTRES) - 1)
    assert masks[0]["nuclei"][CENTRES[-1]] == 0 and masks[0]["cells"][CENTRES[-1]] == 0


def test_off_is_the_same_as_no_maps_at_all(tmp_path, models, flats):
    """The raw path with the setting off equals the run of a folder without any maps."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _maps_folder(tmp_path / "a" / "flatfield")
    _r, off, marker_off = _run(tmp_path / "a", {**SETTINGS, "ops_ph_segment_flatfield": False})
    _r, none, marker_none = _run(tmp_path / "b", dict(SETTINGS))
    assert marker_off == marker_none
    for site in (0, 1):
        for key in off[site]:
            assert off[site][key].tobytes() == none[site][key].tobytes()


def test_on_feeds_corrected_arrays_and_recovers_the_corner_object(tmp_path, models, flats,
                                                                  monkeypatch):
    """True with complete maps: the DNA and cell planes Cellpose sees are (I - b) / flat + b."""
    _maps_folder(tmp_path / "flatfield")
    fed = []
    real = ops_engine._ops_mask_eval

    def record(model, image, settings, **keywords):
        fed.append(np.array(image, copy=True))
        return real(model, image, settings, **keywords)

    monkeypatch.setattr(ops_engine, "_ops_mask_eval", record)
    report, masks, marker = _run(tmp_path, dict(SETTINGS))
    assert all(f is not None for f in flats) and len(flats) == 2
    planes, offsets = flats[0]
    expected = [(_field(site)[0].astype(np.float32) - offsets["DAPI"]) / planes["DAPI"]
                + offsets["DAPI"] for site in (0, 1)]
    assert sum(np.array_equal(fed[0], e) for e in expected) == 1
    assert report["ph_segment"]["objects"]["paired"] == 2 * len(CENTRES)
    corner = CENTRES[-1]
    assert masks[0]["nuclei"][corner] > 0 and masks[0]["cells"][corner] > 0
    assert set(marker["segment_flatfield"]) == {"DAPI", "AF750"}


def test_raw_and_corrected_masks_are_never_mixed(tmp_path, models, flats):
    """A folder of raw masks refuses a corrected resume, and the reverse."""
    _maps_folder(tmp_path / "flatfield")
    _run(tmp_path, {**SETTINGS, "ops_ph_segment_flatfield": False})
    with pytest.raises(ValueError, match="parts made with"):
        _run(tmp_path, dict(SETTINGS))


def test_pending_maps_without_the_objects_phase_segment_raw(tmp_path, models, flats):
    _maps_folder(tmp_path / "flatfield", pending=("DAPI", "AF750"))
    _report, _masks, marker = _run(tmp_path, dict(SETTINGS))
    assert flats == [None, None]
    assert "segment_flatfield" not in marker


def _run_ops(tmp_path, monkeypatch, phases):
    """run_ops over the phenotype phases; ph_flatfield_objects completes the maps."""
    tiles = _write_fields(tmp_path / "pheno")
    db = str(tmp_path / "measurements.db")
    _placed(db)
    (tmp_path / "A1").mkdir(exist_ok=True)
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.25}}))
    built = []

    def objects(plate, wells, index, settings, destination, mask_folder="phenotype_masks"):
        folder = os.path.join(destination, "A1", mask_folder)
        built.append((mask_folder, sorted(n for n in os.listdir(folder) if n.endswith(".npz"))))
        info = json.loads((tmp_path / "flatfield" / "flat_info.json").read_text())
        if info.get("pending"):
            _complete(tmp_path / "flatfield")
        return {"built": True}

    monkeypatch.setattr(ops_engine, "_ph_flatfield_objects", objects)
    ops_engine.run_ops({**SETTINGS, "dst_root": str(tmp_path), "plate": PLATE,
                        "ops_gpu": False, "phenotype_source": str(tiles)}, phases=phases)
    return built


def test_pending_maps_are_built_from_a_raw_pass_then_segmented_corrected(tmp_path, models,
                                                                         flats, monkeypatch):
    """No maps supplied: raw into phenotype_masks_raw, the maps, then ph_segment corrected."""
    _maps_folder(tmp_path / "flatfield", pending=("DAPI", "AF750"))
    built = _run_ops(tmp_path, monkeypatch, ("ph_segment", "ph_flatfield_objects"))
    assert built == [("phenotype_masks_raw", ["site-0.npz", "site-1.npz"])]
    assert [f is None for f in flats] == [True, True, False, False]
    marker = json.loads((tmp_path / "A1" / "phenotype_masks" / "part_options.json").read_text())
    assert set(marker["segment_flatfield"]) == {"DAPI", "AF750"}
    raw = json.loads((tmp_path / "A1" / "phenotype_masks_raw" / "part_options.json").read_text())
    assert "segment_flatfield" not in raw


def test_supplied_maps_segment_once_and_corrected(tmp_path, models, flats, monkeypatch):
    """Complete maps (ops_flatfield_maps or a finished folder): one corrected pass."""
    _maps_folder(tmp_path / "flatfield")
    built = _run_ops(tmp_path, monkeypatch, ("ph_segment", "ph_flatfield_objects"))
    assert built[0][0] == "phenotype_masks"
    assert not (tmp_path / "A1" / "phenotype_masks_raw").exists()
    assert len(flats) == 2 and all(f is not None for f in flats)


def test_the_setting_off_keeps_the_old_phase_order(tmp_path, models, flats, monkeypatch):
    _maps_folder(tmp_path / "flatfield", pending=("DAPI", "AF750"))
    monkeypatch.setitem(SETTINGS, "ops_ph_segment_flatfield", False)
    built = _run_ops(tmp_path, monkeypatch, ("ph_segment", "ph_flatfield_objects"))
    assert built[0][0] == "phenotype_masks"
    assert flats == [None, None]


def _resolved(tmp_path, rows):
    path = tmp_path / "settings.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Key", "Value"])
        writer.writerows(rows)
    return cli.resolve_settings(cli.resolve_module("ops"), str(path))


def test_the_default_is_on_and_older_files_replay_raw_segmentation(tmp_path):
    assert OPS_DEFAULTS["ops_ph_segment_flatfield"] is True
    assert DEFAULT_CHANGES[974] == {"ops_ph_segment_flatfield": False}
    assert SETTINGS_SCHEMA_VERSION >= 974
    assert _resolved(tmp_path, [[SETTINGS_VERSION_KEY, 965]])["ops_ph_segment_flatfield"] is False
    assert _resolved(tmp_path, [[SETTINGS_VERSION_KEY, SETTINGS_SCHEMA_VERSION]]
                     )["ops_ph_segment_flatfield"] is True
    assert _resolved(tmp_path, [[SETTINGS_VERSION_KEY, 965], ["ops_ph_segment_flatfield", True]]
                     )["ops_ph_segment_flatfield"] is True


def test_the_setting_is_registered_with_a_short_tooltip():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_TOOLTIPS

    text = OPS_TOOLTIPS["ops_ph_segment_flatfield"]
    assert len(text) <= 600 and "974" in text
    assert "ops_ph_segment_flatfield" in OPS_CATEGORIES["Phenotype segmentation"]
    assert len(OPS_CATEGORIES["Phenotype segmentation"]) <= 20
