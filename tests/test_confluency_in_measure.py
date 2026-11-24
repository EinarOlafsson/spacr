"""Item 541: confluency in Measure, for any channel.

Confluency is the fraction of a field covered by cells. Measure computes it
from tests import store_sql
from one of three sources -- texture (brightfield, phase), intensity
(fluorescent cytoplasm or membrane stains) or the cell masks -- writes it per
field and per well into measurements.db with a monolayer QC flag, and the
infection report and any per-well table can use it as filter and
denominator.

The synthetic fields have an exactly known covered area, so each source is
held to within two percentage points of it. The real fields are the Toxo PV
Make Masks examples, whose ground-truth masks were drawn by hand, and a
plate1 field compared with its cell masks; both skip when the example data
has not been downloaded.
"""
from __future__ import annotations

import glob
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from spacr import measure
from spacr.measure import (
    _CONFLUENCY_TABLE,
    _CONFLUENCY_WELL_TABLE,
    _confluency_by_well,
    _confluency_overlay,
    _field_confluency,
    _intensity_coverage,
    _mask_coverage,
    _monolayer_qc,
    _resolve_confluency_source,
    _texture_coverage,
)

SHAPE = (384, 384)
FRACTIONS = (0.0, 0.1, 0.35, 0.6, 0.9, 1.0)
TOLERANCE = 0.02


def _truth(fraction, seed=0):
    """A monolayer with exactly ``fraction`` of the field covered."""
    if fraction <= 0:
        return np.zeros(SHAPE, dtype=bool)
    if fraction >= 1:
        return np.ones(SHAPE, dtype=bool)
    rng = np.random.default_rng(seed)
    field = ndi.gaussian_filter(rng.standard_normal(SHAPE), 20)
    return field > np.quantile(field, 1.0 - fraction)


def _brightfield(covered, seed=1):
    """Flat plastic with camera noise; cells add fine-grained texture."""
    rng = np.random.default_rng(seed)
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 1.5)
    texture /= texture.std()
    image = 1000.0 + rng.normal(0.0, 8.0, SHAPE) + covered * texture * 120.0
    return np.clip(image, 0, 65535).astype(np.uint16)


def _fluorescent(covered, seed=2):
    """Dark background; a cytoplasmic stain with uneven brightness."""
    rng = np.random.default_rng(seed)
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 2.0)
    texture /= texture.std()
    image = (100.0 + rng.normal(0.0, 10.0, SHAPE)
             + covered * (600.0 + 120.0 * texture))
    return np.clip(image, 0, 65535).astype(np.uint16)


def _labels(covered):
    """Cell labels whose union is exactly ``covered``."""
    labels, _count = ndi.label(covered)
    return labels.astype(np.uint16)


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_texture_source_recovers_known_brightfield_coverage(fraction):
    truth = _truth(fraction)
    result = _texture_coverage(_brightfield(truth))
    assert result.source == "texture"
    assert abs(result.confluency - truth.mean()) <= TOLERANCE
    assert result.covered.shape == SHAPE


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_intensity_source_recovers_known_fluorescent_coverage(fraction):
    truth = _truth(fraction)
    result = _intensity_coverage(_fluorescent(truth))
    assert result.source == "intensity"
    assert abs(result.confluency - truth.mean()) <= TOLERANCE


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_mask_source_is_the_union_of_the_cells(fraction):
    truth = _truth(fraction)
    result = _mask_coverage(_labels(truth))
    assert result.source == "masks"
    assert result.covered_px == int(truth.sum())
    assert result.confluency == pytest.approx(truth.mean())


def test_a_field_whose_pixels_do_not_split_is_decided_whole():
    """An empty field and a full monolayer both give one pixel class."""
    empty = _texture_coverage(_brightfield(np.zeros(SHAPE, dtype=bool)))
    full = _texture_coverage(_brightfield(np.ones(SHAPE, dtype=bool)))
    assert empty.uniform and empty.confluency == 0.0
    assert full.uniform and full.confluency == 1.0


def test_a_faint_phase_monolayer_is_decided_covered():
    """A monolayer whose fine texture barely exceeds the noise estimate.

    Confluent LIVECell phase fields read a texture ratio of about 2 to 3,
    bare plastic 1.1 to 1.7; the texture source must call the first
    covered and the second empty.
    """
    rng = np.random.default_rng(3)
    texture = ndi.gaussian_filter(rng.standard_normal(SHAPE), 1.0)
    texture /= texture.std()
    faint = 128.0 + rng.normal(0.0, 2.0, SHAPE) + 5.0 * texture
    bare = 128.0 + np.random.default_rng(4).normal(0.0, 2.0, SHAPE)
    covered = _texture_coverage(faint)
    empty = _texture_coverage(bare)
    assert covered.uniform and covered.confluency == 1.0
    assert empty.uniform and empty.confluency == 0.0


def test_a_z_stack_is_max_projected():
    truth = _truth(0.35)
    stack = np.stack([np.zeros(SHAPE, dtype=np.uint16), _labels(truth)])
    assert _mask_coverage(stack).confluency == pytest.approx(truth.mean())


def test_auto_uses_the_cell_masks_when_the_run_has_them():
    assert _resolve_confluency_source({"cell_mask_dim": 4}) == "masks"
    assert _resolve_confluency_source({"cell_mask_dim": None}) == "texture"
    assert _resolve_confluency_source(
        {"cell_mask_dim": 4, "confluency_source": "intensity"}) == "intensity"
    with pytest.raises(ValueError, match="cell_mask_dim"):
        _resolve_confluency_source(
            {"cell_mask_dim": None, "confluency_source": "masks"})
    with pytest.raises(ValueError, match="confluency_source"):
        _resolve_confluency_source({"confluency_source": "guess"})
    truth = _truth(0.6)
    by_masks = _field_confluency(_brightfield(truth), _labels(truth))
    by_texture = _field_confluency(_brightfield(truth))
    assert by_masks.source == "masks" and by_texture.source == "texture"


def test_the_overlay_tints_exactly_the_covered_area():
    truth = _truth(0.35)
    image = _fluorescent(truth)
    overlay = _confluency_overlay(image, truth, color=(255, 0, 0), alpha=1.0)
    assert overlay.shape == SHAPE + (3,) and overlay.dtype == np.uint8
    assert (overlay[truth][:, 1] == 0).all()
    grey = overlay[~truth]
    assert (grey[:, 0] == grey[:, 1]).all()


def _pv_field_pairs(folder):
    """(image, drawn mask) pairs the dataset's ``manifest.csv`` lists.

    The Make Masks editor works in this folder and writes recropped children
    (``<field>__r00.tif``) beside the fields, so a ``*.tif`` glob is not the
    dataset; the manifest is.
    """
    import csv

    manifest = folder / "manifest.csv"
    if not manifest.is_file():
        return []
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(folder / row["image"].strip(), folder / row["ground_truth_mask"].strip())
             for row in rows
             if (row.get("image") or "").strip()
             and (row.get("ground_truth_mask") or "").strip()]
    return sorted((image, truth) for image, truth in pairs
                  if image.is_file() and truth.is_file())


def _pv_fields(limit=4):
    folder = Path(os.environ.get("SPACR_TOXO_PV_DIR") or
                  Path.home() / ".cache/spacr/example_data/make_masks_toxo_pv")
    pairs = _pv_field_pairs(folder)[:limit]
    if not pairs:
        pytest.skip("Toxo PV example data not downloaded")
    return pairs


def test_toxo_pv_fields_come_from_the_manifest_not_editor_recrops(tmp_path):
    import tifffile

    (tmp_path / "ground_truth_masks").mkdir()
    for name in ("plate1_A02_12", "plate2_E08_13"):
        tifffile.imwrite(tmp_path / f"{name}.tif", np.zeros((4, 4), np.uint16))
        tifffile.imwrite(tmp_path / "ground_truth_masks" / f"{name}.tif",
                         np.zeros((4, 4), np.uint16))
    tifffile.imwrite(tmp_path / "plate1_A02_12__r00.tif", np.zeros((2, 2), np.uint16))
    (tmp_path / "manifest.csv").write_text(
        "name,image,ground_truth_mask\n"
        "plate1_A02_12,plate1_A02_12.tif,ground_truth_masks/plate1_A02_12.tif\n"
        "plate2_E08_13,plate2_E08_13.tif,ground_truth_masks/plate2_E08_13.tif\n",
        encoding="utf-8")
    pairs = _pv_field_pairs(tmp_path)
    assert [image.name for image, _truth in pairs] == ["plate1_A02_12.tif",
                                                       "plate2_E08_13.tif"]
    assert all(truth.parent.name == "ground_truth_masks" for _image, truth in pairs)
    assert _pv_field_pairs(tmp_path / "absent") == []


def test_real_toxo_pv_fields_match_their_hand_drawn_ground_truth():
    """Fluorescent PV stain: intensity and texture against the drawn masks.

    Otsu alone reported half the drawn area on these fields; the quarter-way
    cut brings every one within two percentage points.
    """
    import tifffile

    for path, truth in _pv_fields():
        image = tifffile.imread(path)
        drawn = tifffile.imread(truth) > 0
        by_intensity = _intensity_coverage(image).confluency
        by_texture = _texture_coverage(image).confluency
        assert abs(by_intensity - drawn.mean()) <= TOLERANCE, path
        assert abs(by_texture - drawn.mean()) <= TOLERANCE, path
        assert _mask_coverage(drawn).confluency == pytest.approx(drawn.mean())


def test_real_plate1_field_matches_its_cell_masks():
    """The cell stain of a plate1 field against the field's cell masks."""
    fields = sorted(glob.glob(str(
        Path.home() / ".cache/spacr/example_data/plate1/merged/*.npy")))[:2]
    if not fields:
        pytest.skip("plate1 example data not downloaded")
    for path in fields:
        data = np.load(path, mmap_mode="r")
        cells = np.asarray(data[..., 4])
        stain = np.asarray(data[..., 1])
        truth = _mask_coverage(cells).confluency
        assert abs(_intensity_coverage(stain).confluency - truth) <= 0.05, path
        assert abs(_texture_coverage(stain).confluency - truth) <= 0.05, path


def _settings(merged, **over):
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1],
        "cell_mask_dim": 2, "nucleus_mask_dim": None,
        "pathogen_mask_dim": None, "cell_min_size": 0,
        "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1,
        "confluency": True, "confluency_qc_threshold": 0.5,
    })
    settings.update(over)
    return settings


WELLS = {"plate1_A01": (0.9, 0.8), "plate1_B02": (0.2, 0.3)}


def _write_plate(root):
    """Two wells of two fields: brightfield, stain and cell masks."""
    merged = root / "merged"
    merged.mkdir(parents=True)
    truths = {}
    for well, fractions in WELLS.items():
        for number, fraction in enumerate(fractions, start=1):
            truth = _truth(fraction, seed=number)
            stack = np.stack([_brightfield(truth), _fluorescent(truth),
                              _labels(truth)], axis=-1)
            name = f"{well}_{number}"
            np.save(merged / f"{name}.npy", stack)
            truths[name] = truth
    return merged, truths


@pytest.mark.parametrize("source,channel", [
    ("auto", None), ("texture", 0), ("intensity", 1)])
def test_measure_writes_confluency_per_field_and_per_well(tmp_path, source,
                                                          channel):
    merged, truths = _write_plate(tmp_path)
    measure.measure_crop(_settings(
        merged, confluency_source=source, confluency_channel=channel))

    db = tmp_path / "measurements" / "measurements.db"
    with store_sql.connect(db) as conn:
        fields = pd.read_sql_query(f"SELECT * FROM {_CONFLUENCY_TABLE}", conn)
        wells = pd.read_sql_query(
            f"SELECT * FROM {_CONFLUENCY_WELL_TABLE}", conn)

    assert len(fields) == 4
    expected_source = "masks" if source == "auto" else source
    assert set(fields["confluency_source"]) == {expected_source}
    for _, row in fields.iterrows():
        truth = truths[row["file_name"]]
        assert abs(row["confluency"] - truth.mean()) <= TOLERANCE
        assert row["field_px"] == truth.size
        assert row["monolayer_ok"] == int(row["confluency"] >= 0.5)

    assert len(wells) == 2
    wells = wells.set_index("columnID")
    confluent, sparse = wells.loc["c1"], wells.loc["c2"]
    assert confluent["n_fields"] == 2 and confluent["monolayer_ok"] == 1
    assert sparse["monolayer_ok"] == 0
    for row, well in ((confluent, "plate1_A01"), (sparse, "plate1_B02")):
        covered = sum(truths[f"{well}_{n}"].sum() for n in (1, 2))
        pooled = covered / (2 * truths[f"{well}_1"].size)
        assert abs(row["confluency"] - pooled) <= TOLERANCE
        assert row["confluency_min"] <= row["confluency_mean"]


def test_the_per_well_table_pools_pixels_and_counts_fields_below_qc():
    fields = pd.DataFrame({
        "plateID": ["p", "p", "p"], "rowID": ["r1"] * 3,
        "columnID": ["c1"] * 3, "fieldID": ["f1", "f2", "f3"],
        "timeID": [None] * 3, "confluency": [1.0, 0.5, 0.0],
        "covered_px": [100, 100, 0], "field_px": [100, 200, 100],
        "confluency_qc_threshold": [0.6] * 3,
    })
    wells = _confluency_by_well(fields)
    row = wells.iloc[0]
    assert row["confluency"] == pytest.approx(0.5)
    assert row["confluency_mean"] == pytest.approx(0.5)
    assert row["fields_below_qc"] == 2
    assert row["monolayer_ok"] == 0
    assert _confluency_by_well(fields, qc_threshold=0.4).iloc[0][
        "monolayer_ok"] == 1


def test_monolayer_qc_is_a_filter_and_a_denominator_for_plaque_tables():
    """A plaque per-image table named by well, joined through ``well_of``."""
    wells = pd.DataFrame({
        "plateID": ["plate1", "plate1"], "rowID": ["r1", "r2"],
        "columnID": ["c1", "c2"], "confluency": [0.9, 0.3],
        "monolayer_ok": [1, 0],
    })
    plaques = pd.DataFrame({"file": ["plate1_A01.tif", "plate1_B02.tif",
                                     "plate1_C03.tif"],
                            "plaque_count": [18, 6, 4]})

    def well_of(row):
        from spacr.schema import parse_field_stem
        field = parse_field_stem(row["file"].replace(".tif", "_1"))
        return field.plateID, field.rowID, field.columnID

    joined = _monolayer_qc(plaques, wells, value_columns=("plaque_count",),
                          well_of=well_of)
    assert list(joined["plaque_count_per_confluency"].round(6)[:2]) == [
        20.0, 20.0]
    assert joined["monolayer_ok"].isna().iloc[2]
    kept = _monolayer_qc(plaques, wells, drop_failing=True, well_of=well_of)
    assert list(kept["file"]) == ["plate1_A01.tif"]
    stricter = _monolayer_qc(plaques, wells, well_of=well_of,
                            qc_threshold=0.95)
    assert list(stricter["monolayer_ok"][:2]) == [0, 0]


def test_infection_report_carries_confluency_and_can_drop_thin_monolayers(
        tmp_path):
    from spacr.infection import infection_report

    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    cells = pd.DataFrame({
        "object_label": [1, 2, 1, 2], "plateID": ["plate1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"], "columnID": ["c1", "c1", "c2", "c2"],
        "fieldID": ["f1"] * 4, "prcf": ["x"] * 4,
    })
    pathogens = pd.DataFrame({
        "object_label": [1, 2, 3], "plateID": ["plate1"] * 3,
        "rowID": ["r1", "r1", "r2"], "columnID": ["c1", "c1", "c2"],
        "fieldID": ["f1"] * 3, "cell_id": [1, 2, 1],
    })
    with store_sql.connect(db) as conn:
        cells.to_sql("cell", conn, index=False)
        pathogens.to_sql("pathogen", conn, index=False)

    plain = infection_report(str(db))
    assert "confluency" not in set(plain["metric"])

    for well, fraction in (("plate1_A01_1", 0.9), ("plate1_B02_1", 0.3)):
        truth = _truth(fraction)
        result = _mask_coverage(_labels(truth))
        measure._write_confluency_record(
            str(tmp_path), well, {"confluency_qc_threshold": 0.5}, result)

    report = infection_report(str(db))
    confluency = report[report["metric"] == "confluency"].set_index(
        "columnID")["value"]
    assert confluency["c1"] == pytest.approx(0.9, abs=0.01)
    per_cover = report[report["metric"] == "parasites_per_confluency"]
    per_cover = per_cover.set_index("columnID")["value"]
    assert per_cover["c1"] == pytest.approx(2 / confluency["c1"])
    assert set(report.loc[report["columnID"] == "c2", "monolayer_ok"]) == {0}

    kept = infection_report(str(db), monolayer_filter=True)
    assert set(kept["columnID"]) == {"c1"}
