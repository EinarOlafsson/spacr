"""Item 1008 part A: Measure and OPS switch the same per-cell blocks, with the same code.

``profile_features`` (Measure) and ``ops_ph_profile_features`` (OPS) accept the
blocks of :data:`spacr.measure.MEASURE_BLOCK_FAMILIES` besides the profiling
families: ``coloc_block`` switches the per-object Pearson and Manders block on,
``legacy_radial`` the six-bin radial distribution, in both apps, through
:func:`spacr.measure._intensity_measurements`. The OPS path writes Funk et al.
2022's features from Measure's own :func:`spacr.measure._add_funk_frames`, so
the cytoplasm gets them too. With every setting at its default, both outputs
are those of the base (train 7 source with item 1007 lane A).
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M
from spacr import ops_sample
from spacr.ops_settings import FINAL_RUN_FAMILIES_1008, OPS_TO_MEASURE_KEYS

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_measure_ops_features_934 import (_measured, _object_tables,  # noqa: E402
                                           _ops_against_measure, _ops_field)

#: The fingerprints of the base outputs on the 934 fixture field: the count and
#: sha256 of the column names, and the sha256 of the numeric values written
#: with 9 significant digits, row by row.
BASE = {
    "ops_default": (315, "9779178d604d142a71d1086fa6c033717aaacc5e14abfe76add619d2fd507da3",
                    "9a3305d84a5abf7d7208feceaea0f2fc4a625f74ab6648fc1e6df372b8cd3f5c"),
    "ops_profiling": (1588, "be63d33d3382b80b5f9f478e71e776a137b0755b2955e38acf620a9d3e5127db",
                      "f95ea90d119058f309370ae04e07722e802a37d7d22a518ea72b001a687eb499"),
    "measure_cell": (221, "cd9bcd120f81d9c8d3c92bfee676b1d907b412b82dc9664b9540ab4a0c54a0f5",
                     "734d8d732d197c84391fe146c3d86bb8dc85cdc8f17e385ed251aa078cabdcc5"),
    "measure_nucleus": (263, "83952045869541a534b2e902d16e253595b4c490cf3a2a0a27b4c47eb48ec479",
                        "47294900773cd406900ca296f5080ff8e49f4574d7e6af576b795ce45544b9a5"),
    "measure_cytoplasm": (129, "061d2d40629e47ea87ecd58059cc0a114a64cddb497df07c4b81e633bcf418ce",
                          "b71947bc172c85ccdace51a34fa14c8eb1a859b2708ed9c1f095600a387733b1"),
}

#: The names of the final list this branch measures: parts B
#: (``colocalisation_otsu``) and C (``cytoskeleton``) arrive separately.
KNOWN = M.PROFILE_FEATURE_FAMILIES + M.MEASURE_BLOCK_FAMILIES
AVAILABLE = [name for name in FINAL_RUN_FAMILIES_1008 if name in KNOWN]
PENDING = [name for name in FINAL_RUN_FAMILIES_1008 if name not in KNOWN]


def _fingerprint(frame):
    columns = list(frame.columns)
    values = frame.select_dtypes("number").to_numpy(np.float64)
    text = "\n".join(",".join(f"{v:.9g}" for v in row) for row in values)
    return (len(columns), hashlib.sha256("\n".join(columns).encode()).hexdigest(),
            hashlib.sha256(text.encode()).hexdigest())


def _intensity(settings, **over):
    nuclei, cells, image, _names = _ops_field()
    blank = np.zeros_like(cells)
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    settings = {**ops_sample._measurement_settings(image.shape[2], False), **settings, **over}
    return M._intensity_measurements(cells, nuclei, blank, blank, cytoplasm, image, settings,
                                     sizes=[1, 2, 3, 4, 5], periphery=False, outside=False)


def _columns(tables):
    return [set(frame.columns) for frame in tables]


def test_the_blocks_are_names_of_profile_features_and_not_of_its_presets():
    """Keep all three explicit blocks outside the profiling presets."""
    assert M.MEASURE_BLOCK_FAMILIES == ("coloc_block", "legacy_radial", "cytoskeleton")
    assert M.profile_feature_families(["legacy_radial", "edge_intensity", "coloc_block"]) == (
        "edge_intensity", "coloc_block", "legacy_radial")
    assert M.profile_feature_families("coloc_block,profiling")[-1] == "coloc_block"
    for preset in ("profiling", "all"):
        assert not set(M.profile_feature_families(preset)) & set(M.MEASURE_BLOCK_FAMILIES)
        assert M.measure_block_families(preset) == ()
    assert M.measure_block_families("legacy_radial,coloc_block") == ("coloc_block", "legacy_radial")
    with pytest.raises(ValueError, match="coloc_block"):
        M.profile_feature_families(["colocalisation_block"])


def test_coloc_block_alone_gives_the_correlation_columns_and_nothing_else():
    base = _intensity({})
    old = _intensity({"calculate_correlation": True})
    new = _intensity({"profile_features": ["coloc_block"]})
    for b, o, n in zip(_columns(base), _columns(old), _columns(new)):
        assert o == n
        added = n - b
        assert all("_correlation" in c or "manders" in c or "_overlap" in c for c in added)
    assert any(_columns(new)[0] - _columns(base)[0])
    for o, n in zip(old, new):
        pd.testing.assert_frame_equal(o, n, check_exact=True)


def test_legacy_radial_alone_gives_the_radial_dist_columns():
    base = _intensity({})
    old = _intensity({"radial_dist": True})
    new = _intensity({"profile_features": "legacy_radial"})
    assert _columns(old) == _columns(new)
    assert any("rad_dist" in c for c in _columns(new)[1] - _columns(base)[1])
    for o, n in zip(old, new):
        pd.testing.assert_frame_equal(o, n, check_exact=True)


def test_an_old_setting_and_its_block_name_compute_the_block_once(monkeypatch):
    calls = {"correlation": 0, "radial": 0}
    correlation, radial = M._calculate_correlation_object_level, M._calculate_radial_distribution

    def _count_correlation(*args, **kwargs):
        calls["correlation"] += 1
        return correlation(*args, **kwargs)

    def _count_radial(*args, **kwargs):
        calls["radial"] += 1
        return radial(*args, **kwargs)

    monkeypatch.setattr(M, "_calculate_correlation_object_level", _count_correlation)
    monkeypatch.setattr(M, "_calculate_radial_distribution", _count_radial)
    _intensity({"calculate_correlation": True, "radial_dist": True})
    once = dict(calls)
    calls.update(correlation=0, radial=0)
    _intensity({"calculate_correlation": True, "radial_dist": True,
                "profile_features": ["coloc_block", "legacy_radial"]})
    assert calls == once and once["correlation"] > 0 and once["radial"] > 0


def test_ops_switches_the_blocks_as_measure_does_and_keeps_colocalisation_as_before():
    settings = ops_sample._measurement_settings(3, False, profile_features=["coloc_block"])
    assert settings["calculate_correlation"] is False
    assert settings["profile_features"] == ["coloc_block"]
    assert ops_sample._measurement_settings(
        3, False, profile_features=["colocalisation"])["calculate_correlation"] is True
    nuclei, cells, image, names = _ops_field()
    plain = ops_sample.measure_phenotype_field(nuclei, cells, image, names, texture=False)
    block = ops_sample.measure_phenotype_field(nuclei, cells, image, names, texture=False,
                                               profile_features=["coloc_block", "legacy_radial"])
    added = set(block.columns) - set(plain.columns)
    assert any("manders" in c for c in added) and any("rad_dist" in c for c in added)
    shared = [c for c in plain.columns if c in block.columns]
    pd.testing.assert_frame_equal(plain[shared], block[shared], check_exact=True)


@pytest.mark.parametrize("key", sorted(BASE))
def test_default_outputs_are_those_of_the_base(tmp_path, key):
    nuclei, cells, image, names = _ops_field()
    if key == "ops_default":
        frame = ops_sample.measure_phenotype_field(nuclei, cells, image, names)
    elif key == "ops_profiling":
        frame = ops_sample.measure_phenotype_field(nuclei, cells, image, names,
                                                   profile_features="profiling")
    else:
        tables = _object_tables(_measured(tmp_path, "default"))
        frame = tables[key.split("_", 1)[1]].drop(columns=["path_name"])
    assert _fingerprint(frame) == BASE[key]


def test_funk_features_in_ops_only_gain_the_cytoplasm_columns():
    """The base wrote Funk's features for the nucleus and the cell with
    :func:`spacr.ops_sample.funk_object_features`; that table, rebuilt from
    the same public functions, is the new one without the cytoplasm."""
    nuclei, cells, image, names = _ops_field()
    new = ops_sample.measure_phenotype_field(nuclei, cells, image, names, funk_features=True)
    old = ops_sample.measure_phenotype_field(nuclei, cells, image, names)
    for prefix, mask in (("nucleus", nuclei), ("cell", cells)):
        funk = ops_sample.funk_object_features(mask, image, names, prefix)
        funk = funk.loc[:, [c for c in funk.columns if c == "label" or c not in old.columns]]
        old = old.merge(funk, on="label", how="left")
        old[f"{prefix}_form_factor"] = M._form_factor(
            old[f"{prefix}_area"].to_numpy(np.float64),
            old[f"{prefix}_perimeter"].to_numpy(np.float64))
    lower = [n.lower() for n in names]
    pairs = [f"{a}_{b}" for i, a in enumerate(lower) for b in lower[i + 1:]]
    cytoplasm = ({f"cytoplasm_{n}_mass_displacement" for n in lower}
                 | {f"cytoplasm_correlation_{p}" for p in pairs} | {"cytoplasm_form_factor"})
    assert set(new.columns) - set(old.columns) == cytoplasm
    assert set(old.columns) <= set(new.columns)
    for column in old.columns:
        np.testing.assert_array_equal(new[column].to_numpy(np.float64),
                                      old[column].to_numpy(np.float64))


#: Measure columns the OPS phenotype path leaves out by design: the label and
#: parent-id bookkeeping and the colocalisation block's object labels.
_MEASURE_ONLY = ("_cell_id", "label_correlation")


def _measure_named(nuclei, cells, image, names, families):
    mapped = {OPS_TO_MEASURE_KEYS[key]: value for key, value in {
        "ops_ph_texture": True, "ops_ph_funk_features": True, "ops_ph_zernike": True,
        "ops_ph_profile_features": families}.items()}
    settings = {**ops_sample._measurement_settings(
        image.shape[2], True, profile_features=families), **mapped}
    blank = np.zeros_like(cells)
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    morphology = M._morphological_measurements(cells, nuclei, blank, blank, cytoplasm,
                                                settings, zernike=True)
    intensity = M._intensity_measurements(cells, nuclei, blank, blank, cytoplasm, image,
                                          settings, sizes=[1, 2, 3, 4, 5],
                                          periphery=False, outside=False)
    lower = [n.lower() for n in names]
    named = set()
    for slot, prefix in ((0, "cell"), (1, "nucleus"), (4, "cytoplasm")):
        for frame in (morphology[slot], intensity[slot]):
            for column in frame.columns:
                if column == "label" or not pd.api.types.is_numeric_dtype(frame[column]):
                    continue
                name = M._channel_named(str(column), lower)
                named.add(name if name.startswith(f"{prefix}_") else f"{prefix}_{name}")
    return named


def test_final_list_parity_in_both_directions():
    """P1: with the complete final family list, every OPS
    column is a Measure column with the same value, and every Measure
    per-cell column of the three compartments is an OPS column."""
    pytest.importorskip("mahotas")
    assert PENDING == [] and AVAILABLE == list(FINAL_RUN_FAMILIES_1008)
    nuclei, cells, image, names = _ops_field()
    table, missing, unequal = _ops_against_measure(
        nuclei, cells, image, names, texture=True, funk=True, zernike=True,
        profile_features=AVAILABLE)
    assert missing == [] and unequal == []
    measured = _measure_named(nuclei, cells, image, names, AVAILABLE)
    absent = sorted(c for c in measured - set(table.columns)
                    if not c.endswith(_MEASURE_ONLY))
    assert absent == []
    assert any("manders" in c for c in table.columns)
    assert any("rad_dist" in c for c in table.columns)
    assert any("otsu_m1" in c for c in table.columns)
    assert any(c.startswith("cell_") and "skeleton_length" in c for c in table.columns)
    assert not any("skeleton" in c and not c.startswith("cell_") for c in table.columns)


#: The OPS phenotype path measures a float32 copy of the field and Measure
#: the stored values in float64, so a mean or a spread can differ in its last
#: float32 bit (at most 9e-8 relative on this field); two float32 rounding
#: steps bound it. Every other column is equal.
FLOAT32_ROUNDING = 2 * float(np.finfo(np.float32).eps)


def test_one_cell_has_the_same_values_in_ops_and_in_the_measure_database(tmp_path):
    """P2: Measure, run by measure_crop on a folder with the mapped settings,
    and OPS agree on every shared column of one cell, to float32 rounding."""
    pytest.importorskip("mahotas")
    nuclei, cells, image, names = _ops_field()
    table = ops_sample.measure_phenotype_field(
        nuclei, cells, image, names, texture=True, funk_features=True, zernike=True,
        profile_features=AVAILABLE)
    tables = _object_tables(_measured(
        tmp_path, "final", homogeneity=True, homogeneity_distances=[2, 4, 8],
        funk_object_features=True, zernike=True, profile_features=AVAILABLE,
        channel_names=[n.lower() for n in names]))
    label = int(table["label"].iloc[1])
    shared = exact = 0
    for prefix, frame in tables.items():
        row = frame.loc[frame["object_label"].astype(int) == label]
        assert len(row) == 1
        for column in row.columns:
            name = column if column.startswith(f"{prefix}_") else f"{prefix}_{column}"
            if name not in table.columns or not pd.api.types.is_numeric_dtype(row[column]):
                continue
            got = row[column].to_numpy(np.float64)
            want = table.loc[table["label"] == label, name].to_numpy(np.float64)
            exact += np.array_equal(got, want, equal_nan=True)
            np.testing.assert_allclose(got, want, rtol=FLOAT32_ROUNDING, atol=0, err_msg=name)
            shared += 1
    assert shared > 500 and exact > 0.9 * shared


def test_the_final_list_is_explicit_and_every_name_can_be_dropped():
    """P9: a family that fails the information test is dropped by deleting its
    name; no family needs another."""
    assert len(FINAL_RUN_FAMILIES_1008) == len(set(FINAL_RUN_FAMILIES_1008)) == 14
    assert not {"granularity", "texture_gabor", "colocalisation", "colocalisation_extra",
                "colocalisation_costes"} & set(FINAL_RUN_FAMILIES_1008)
    if PENDING:
        with pytest.raises(ValueError):
            M.profile_feature_families(FINAL_RUN_FAMILIES_1008)
    else:
        assert set(M.profile_feature_families(FINAL_RUN_FAMILIES_1008)) == set(
            FINAL_RUN_FAMILIES_1008)
    nuclei, cells, image, names = _ops_field()
    full = ops_sample.measure_phenotype_field(nuclei, cells, image, names,
                                              profile_features=AVAILABLE)
    for name in AVAILABLE:
        rest = [n for n in AVAILABLE if n != name]
        table = ops_sample.measure_phenotype_field(nuclei, cells, image, names,
                                                   profile_features=rest)
        assert set(table.columns) < set(full.columns), name


@pytest.mark.parametrize("channels, indices", [(None, (0, 1, 2)), ([1], (1,)), (["GFP"], (1,))])
def test_ops_cytoskeleton_selection_reaches_the_shared_cell_only_measurement(channels, indices):
    """Resolve selected channels through OPS and match direct per-cell skeleton values."""
    nuclei, cells, image, names = _ops_field()
    table = ops_sample.measure_phenotype_field(
        nuclei, cells, image, names, texture=False,
        profile_features=["cytoskeleton"], cytoskeleton_channels=channels)
    expected = M.measure_cytoskeleton(cells, image, list(indices))
    actual_columns = {column for column in table if "skeleton" in column}
    expected_columns = {
        f"cell_{names[index].lower()}_{stat}"
        for index in indices for stat in ("skeleton_length", "skeleton_branch_points")}
    assert actual_columns == expected_columns
    expected = expected.set_index("label").reindex(table["label"])
    for index in indices:
        for stat in ("skeleton_length", "skeleton_branch_points"):
            np.testing.assert_array_equal(
                table[f"cell_{names[index].lower()}_{stat}"],
                expected[f"channel_{index}_{stat}"])
    inactive = ops_sample.measure_phenotype_field(
        nuclei, cells, image, names, texture=False, cytoskeleton_channels=channels)
    assert not any("skeleton" in column for column in inactive)


@pytest.mark.parametrize("change", ["none", "name", "default", "type", "before", "after"])
def test_cytoskeleton_api_revision_rejects_unrecorded_source_drift(tmp_path, monkeypatch, change):
    """Bind the actual API source and reject altered keywords, defaults or hash-chain records."""
    import json
    from tests import test_documentation_i18n as owner

    root = pathlib.Path(__file__).resolve().parents[1]
    filename = "1008a2_phenotype_cytoskeleton_api_change_2026-11-21.json"
    record = json.loads((root / "tests/data/release_contracts" / filename).read_text())
    source = (root / "spacr/ops_sample.py").read_text()
    entries = json.loads((root / "tests/data/release_contracts" /
                          "ops_phenotype_path_api_arrivals_2026-11-05.json").read_text())
    if change == "name":
        source = source.replace("                            cytoskeleton_channels=None):",
                                "                            skeleton_channels=None):")
    elif change == "default":
        source = source.replace("                            cytoskeleton_channels=None):",
                                "                            cytoskeleton_channels=False):")
    elif change == "type":
        source = source.replace("granularity_subsample: float = 1.0,",
                                "granularity_subsample: float = 1,")
    elif change == "before":
        entries[record["symbol"]] += " unrecorded"
    elif change == "after":
        record["after"] += " unrecorded"
    (tmp_path / "spacr").mkdir()
    (tmp_path / "spacr/ops_sample.py").write_text(source)
    folder = tmp_path / "tests/data/release_contracts"
    folder.mkdir(parents=True)
    (folder / filename).write_text(json.dumps(record))
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    if change == "none":
        actual = owner._current_1008a2_ops_api(entries)
        assert actual == {**entries, record["symbol"]: record["after"]}
    else:
        with pytest.raises(AssertionError):
            owner._current_1008a2_ops_api(entries)
