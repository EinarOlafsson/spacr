"""Item 941: Gabor, granularity and colocalisation off by default.

The ``'profiling'`` preset of ``profile_features`` and
``ops_ph_profile_features`` no longer measures the slow families
(texture_gabor, colocalisation, colocalisation_extra, colocalisation_costes,
granularity). ``profile_optional_families`` and
``ops_ph_profile_optional_families`` add them back, and settings files
written before the change get the three families ``'profiling'`` measured
then, so their columns are unchanged. Aggregation and profile building work
on the smaller tables.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M

pytest.importorskip("mahotas")

PROFILING_BEFORE_941 = (
    "edge_intensity", "intensity_moments", "radial_distribution",
    "shape_extended", "texture_haralick", "texture_pftas", "texture_lbp",
    "texture_gabor", "periphery", "outside", "colocalisation",
    "colocalisation_extra", "neighbours",
)
SLOW = ("texture_gabor", "colocalisation", "colocalisation_extra",
        "colocalisation_costes", "granularity")
SLOW_MARKERS = ("gabor", "granularity", "manders", "pearson", "costes",
                "spearman", "icq", "rwc", "nmi", "_k1", "_k2", "puncta")


def _ops_field():
    """Two paired cells with nuclei in a 2-channel field."""
    rng = np.random.default_rng(7)
    cells = np.zeros((70, 70), np.int32)
    nuclei = np.zeros_like(cells)
    cells[5:35, 5:35] = 1
    cells[35:65, 30:65] = 2
    nuclei[12:26, 12:26] = 1
    nuclei[43:57, 40:55] = 2
    image = rng.gamma(3.0, 40.0, size=(70, 70, 2)).astype(np.float32)
    return nuclei, cells, image


def _slow(columns):
    return [c for c in columns if any(m in c.lower() for m in SLOW_MARKERS)]


def test_profiling_preset_leaves_the_slow_families_out():
    families = M.profile_feature_families("profiling")
    assert families == M.PROFILE_FEATURE_DEFAULT
    assert not set(families) & set(SLOW)
    assert set(M.PROFILE_FEATURE_OPTIONAL) == set(SLOW)
    assert set(families) | set(SLOW) == set(M.PROFILE_FEATURE_FAMILIES)
    assert M.profile_feature_families("all") == M.PROFILE_FEATURE_FAMILIES


def test_optional_families_parse_and_only_widen_profiling():
    assert M.profile_optional_families(None) == ()
    assert M.profile_optional_families(False) == ()
    assert M.profile_optional_families("") == ()
    assert M.profile_optional_families(True) == M.PROFILE_FEATURE_OPTIONAL
    assert M.profile_optional_families("all") == M.PROFILE_FEATURE_OPTIONAL
    assert M.profile_optional_families("granularity, texture_gabor") == (
        "texture_gabor", "granularity")
    with pytest.raises(ValueError, match="not an optional family"):
        M.profile_optional_families(["texture_haralick"])
    previous = ["texture_gabor", "colocalisation", "colocalisation_extra"]
    assert M.profile_feature_families("profiling", previous) == PROFILING_BEFORE_941
    assert M.profile_feature_families("profiling", True) == M.PROFILE_FEATURE_FAMILIES
    assert M.profile_feature_families(None, True) == ()
    assert M.profile_feature_families("edge_intensity", True) == ("edge_intensity",)
    assert M.profile_feature_families("all", None) == M.PROFILE_FEATURE_FAMILIES


def test_default_ops_table_has_no_slow_columns_and_flags_restore_the_previous_table():
    from spacr.ops_sample import measure_phenotype_field

    nuclei, cells, image = _ops_field()
    default = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                      profile_features="profiling")
    before = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                     profile_features=list(PROFILING_BEFORE_941))
    flagged = measure_phenotype_field(
        nuclei, cells, image, ["DAPI", "GFP"],
        profile_features=list(M.profile_feature_families(
            "profiling", ["texture_gabor", "colocalisation",
                          "colocalisation_extra"])))
    pd.testing.assert_frame_equal(flagged, before)
    assert not _slow(default.columns)
    assert set(default.columns) < set(before.columns)
    pd.testing.assert_frame_equal(before[default.columns], default)
    added = set(before.columns) - set(default.columns)
    assert added == set(_slow(before.columns))
    for prefix in ("nucleus", "cell", "cytoplasm"):
        assert f"{prefix}_dapi_gfp_manders_m1" in added
        assert f"{prefix}_dapi_gfp_rwc1" in added


def test_ops_settings_resolve_with_the_flag():
    from spacr.ops_engine import _ph_granularity_subsample, _ph_profile_features
    from spacr.ops_settings import OPS_TO_MEASURE_KEYS, ops_defaults

    defaults = ops_defaults()
    assert defaults["ops_ph_profile_optional_families"] is None
    assert OPS_TO_MEASURE_KEYS["ops_ph_profile_optional_families"] == (
        "profile_optional_families")
    on = {"ops_ph_profile_features": "profiling",
          "ops_ph_profile_optional_families": True}
    assert _ph_profile_features(on) == list(M.PROFILE_FEATURE_FAMILIES)
    assert _ph_granularity_subsample(on) == 1.0
    off = {"ops_ph_profile_features": "profiling"}
    assert _ph_profile_features(off) == list(M.PROFILE_FEATURE_DEFAULT)
    assert _ph_granularity_subsample(off) is None


def test_measure_setting_is_registered_off_and_alpha():
    from spacr.settings import (ALPHA_FEATURES, expected_types,
                                get_measure_crop_settings)

    measure = get_measure_crop_settings(settings={})
    assert measure["profile_optional_families"] is None
    assert "profile_optional_families" in expected_types
    assert set(ALPHA_FEATURES[941]["settings"]) == {
        "profile_optional_families", "ops_ph_profile_optional_families"}


@pytest.mark.parametrize("version", [0, 935])
def test_older_settings_files_keep_the_previous_families(version):
    from spacr.ops_engine import _ph_profile_features
    from spacr.settings import DEFAULT_CHANGES, _fold_older_defaults

    row = [v for v, change in DEFAULT_CHANGES.items()
           if "profile_optional_families" in change]
    assert len(row) == 1
    settings = {"profile_features": "profiling",
                "ops_ph_profile_features": "profiling"}
    applied = _fold_older_defaults(settings, version)
    assert applied["profile_optional_families"][1] == row[0]
    assert M.profile_feature_families(
        settings["profile_features"],
        settings["profile_optional_families"]) == PROFILING_BEFORE_941
    assert tuple(_ph_profile_features(settings)) == PROFILING_BEFORE_941


def test_stamped_files_and_explicit_values_take_the_new_default():
    from spacr.settings import (SETTINGS_SCHEMA_VERSION, DEFAULT_CHANGES,
                                _fold_older_defaults)

    row = max(v for v, change in DEFAULT_CHANGES.items()
              if "profile_optional_families" in change)
    assert SETTINGS_SCHEMA_VERSION >= row
    current = {"profile_features": "profiling"}
    assert "profile_optional_families" not in _fold_older_defaults(
        current, SETTINGS_SCHEMA_VERSION)
    assert M.profile_feature_families(
        current["profile_features"],
        current.get("profile_optional_families")) == M.PROFILE_FEATURE_DEFAULT
    chosen = {"profile_features": "profiling",
              "profile_optional_families": None}
    _fold_older_defaults(chosen, 0)
    assert chosen["profile_optional_families"] is None


def test_aggregation_and_profiles_tolerate_the_missing_columns():
    from spacr.embeddings import aggregate_profiles
    from spacr.merge_tables import aggregate_by_feature, summary_aggregation_plan
    from spacr.ops_engine import _ph_feature_columns
    from spacr.ops_sample import measure_phenotype_field

    nuclei, cells, image = _ops_field()
    tables = {}
    for name, families in (("default", "profiling"),
                           ("all_on", list(M.profile_feature_families(
                               "profiling", True)))):
        frames = []
        for well in ("A01", "A02", "B01", "B02"):
            table = measure_phenotype_field(nuclei, cells, image,
                                            ["DAPI", "GFP"],
                                            profile_features=families)
            table["well"] = well
            table["gene"] = "nontargeting" if well[0] == "A" else "GENE1"
            frames.append(table)
        tables[name] = pd.concat(frames, ignore_index=True)
    default, all_on = tables["default"], tables["all_on"]
    assert not _slow(default.columns)
    assert _slow(all_on.columns)

    plan_default = summary_aggregation_plan(
        default.drop(columns=["well", "gene"])).set_index("column")["method"]
    plan_all = summary_aggregation_plan(
        all_on.drop(columns=["well", "gene"])).set_index("column")["method"]
    pd.testing.assert_series_equal(plan_all[plan_default.index], plan_default)

    wells_default = aggregate_by_feature(default.drop(columns="gene"), "well")
    wells_all = aggregate_by_feature(all_on.drop(columns="gene"), "well")
    shared = [c for c in wells_default.columns]
    assert set(shared) <= set(wells_all.columns)
    pd.testing.assert_frame_equal(wells_all[shared], wells_default)

    features_default = _ph_feature_columns(default.columns)
    features_all = _ph_feature_columns(all_on.columns)
    assert set(features_default) < set(features_all)
    assert not _slow(features_default)

    profiles_default = aggregate_profiles(
        default, by=["gene"], features=features_default, well_columns=["well"])
    profiles_all = aggregate_profiles(
        all_on, by=["gene"], features=features_all, well_columns=["well"])
    common = [c for c in profiles_default.columns]
    assert set(common) <= set(profiles_all.columns)
    pd.testing.assert_frame_equal(profiles_all[common], profiles_default,
                                  check_like=True)
