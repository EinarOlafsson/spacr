"""Tests for spacr.validate — the pre-flight settings check.

Every case here is a mistake that used to cost a full GPU run to discover:
a channel index one past the end of the plate, a ``src`` pointing at the
plate folder instead of ``plate/merged``, an integer that came back from a
settings CSV as a string.

Everything runs against tiny synthetic folders on tmp_path: no network, no
GPU, no real image decoding.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr.validate import (
    ERROR,
    WARNING,
    Problem,
    describe_plan,
    format_report,
    validate_settings,
)


# ---------------------------------------------------------------------------
# synthetic datasets
# ---------------------------------------------------------------------------

def make_raw_plate(root, n_channels=3, n_fields=4):
    """A folder of CellVoyager-named raw tifs: n_fields x n_channels files."""
    plate = root / "plate1"
    plate.mkdir(parents=True, exist_ok=True)
    for field in range(1, n_fields + 1):
        for chan in range(1, n_channels + 1):
            name = f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif"
            (plate / name).write_bytes(b"")
    return str(plate)


def make_merged_plate(root, n_planes=7, n_files=3, with_db=False):
    """A plate folder with a populated ``merged/`` of (H, W, n_planes) arrays."""
    plate = root / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        np.save(str(merged / f"plate1_A01_{i}_1.npy"),
                np.zeros((8, 8, n_planes), dtype=np.uint16))
    if with_db:
        (plate / "measurements").mkdir(exist_ok=True)
        (plate / "measurements" / "measurements.db").write_bytes(b"")
    return str(plate), str(merged)


def valid_mask_settings(src):
    """A settings dict that should validate clean against a 3-channel plate."""
    return {
        "src": src,
        "metadata_type": "cellvoyager",
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": 2,
        "organelle_channel": None,
        "channels": [0, 1, 2],
        "magnification": 20,
        "batch_size": 50,
        "n_jobs": 4,
        "verbose": False,
    }


def valid_measure_settings(merged):
    """A settings dict that should validate clean against a 7-plane merged/."""
    return {
        "src": merged,
        "cell_mask_dim": 4,
        "nucleus_mask_dim": 5,
        "pathogen_mask_dim": 6,
        "cell_min_size": 0,
        "nucleus_min_size": 0,
        "pathogen_min_size": 0,
        "channels": [0, 1, 2, 3],
        "crop_mode": ["cell"],
        "save_png": True,
        "png_size": [224, 224],
        "normalize": [1, 99],
        "normalize_by": "png",
        "n_jobs": 2,
    }


def errors(problems):
    return [p for p in problems if p.severity == ERROR]


def warnings_of(problems):
    return [p for p in problems if p.severity == WARNING]


def settings_named(problems, name):
    return [p for p in problems if name in p.setting]


# ---------------------------------------------------------------------------
# the headline case: a channel index past the end of the plate
# ---------------------------------------------------------------------------

def test_organelle_channel_3_on_a_3_channel_dataset_is_an_error(tmp_path):
    """The user's own example. Three channels means valid indices are 0-2."""
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["organelle_channel"] = 3

    problems = validate_settings(settings, "mask")

    offenders = settings_named(errors(problems), "organelle_channel")
    assert offenders, f"organelle_channel=3 was not flagged: {[str(p) for p in problems]}"
    assert "3 channels" in offenders[0].message
    assert "0-2" in offenders[0].message
    assert offenders[0].fix, "an error must say what to do about it"


def test_channel_within_range_on_the_same_plate_is_clean(tmp_path):
    """Guard the other side of the boundary: index 2 of 3 is fine."""
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["organelle_channel"] = 2
    settings["pathogen_channel"] = None

    problems = validate_settings(settings, "mask")
    assert not errors(problems), [str(p) for p in problems]


def test_channel_count_read_from_a_stack_npy_when_present(tmp_path):
    """stack/*.npy holds image channels only, so its last axis is the count."""
    plate = tmp_path / "plate1"
    stack = plate / "stack"
    stack.mkdir(parents=True)
    np.save(str(stack / "plate1_A01_1_1.npy"), np.zeros((8, 8, 2), dtype=np.uint16))

    settings = {"src": str(plate), "cell_channel": 0, "nucleus_channel": 5}
    problems = validate_settings(settings, "mask")

    offenders = settings_named(errors(problems), "nucleus_channel")
    assert offenders
    assert "2 channels" in offenders[0].message


def test_negative_channel_index_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_channel"] = -1
    problems = validate_settings(settings, "mask")
    assert settings_named(errors(problems), "cell_channel")


# ---------------------------------------------------------------------------
# src
# ---------------------------------------------------------------------------

def test_missing_src_key_is_an_error():
    problems = validate_settings({"cell_channel": 0}, "mask")
    offenders = settings_named(errors(problems), "src")
    assert offenders
    assert "missing" in offenders[0].message


def test_src_that_does_not_exist_is_an_error(tmp_path):
    settings = {"src": str(tmp_path / "no_such_plate"), "cell_channel": 0}
    problems = validate_settings(settings, "mask")
    offenders = settings_named(errors(problems), "src")
    assert offenders
    assert "does not exist" in offenders[0].message


def test_placeholder_src_from_the_defaults_is_an_error():
    """``set_default_*`` ships src='path'; running that wastes a scheduler slot."""
    problems = validate_settings({"src": "path", "cell_channel": 0}, "mask")
    offenders = settings_named(errors(problems), "src")
    assert offenders
    assert "placeholder" in offenders[0].message


def test_src_that_is_a_file_not_a_folder_is_an_error(tmp_path):
    target = tmp_path / "plate.tif"
    target.write_bytes(b"")
    problems = validate_settings({"src": str(target), "cell_channel": 0}, "mask")
    assert any("folder" in p.message for p in settings_named(errors(problems), "src"))


def test_src_may_be_a_list_of_folders(tmp_path):
    first = make_raw_plate(tmp_path / "a", n_channels=3)
    settings = valid_mask_settings([first, str(tmp_path / "b" / "missing")])
    problems = validate_settings(settings, "mask")
    assert any("missing" in p.message for p in settings_named(errors(problems), "src"))


def test_empty_source_folder_for_mask_is_an_error(tmp_path):
    plate = tmp_path / "empty_plate"
    plate.mkdir()
    problems = validate_settings({"src": str(plate), "cell_channel": 0}, "mask")
    assert any("no image files" in p.message for p in errors(problems))


def test_images_that_match_no_metadata_pattern_warn_about_metadata_type(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    for i in range(3):
        (plate / f"random_name_{i}.tif").write_bytes(b"")
    settings = {"src": str(plate), "cell_channel": 0, "metadata_type": "cellvoyager"}
    problems = validate_settings(settings, "mask")
    offenders = settings_named(warnings_of(problems), "metadata_type")
    assert offenders
    assert "custom_regex" in offenders[0].fix


# ---------------------------------------------------------------------------
# measure needs merged/
# ---------------------------------------------------------------------------

def test_measure_with_no_merged_folder_is_an_error(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    settings = valid_measure_settings(str(plate))
    problems = validate_settings(settings, "measure")
    offenders = [p for p in errors(problems) if "merged" in p.message]
    assert offenders
    assert "Mask module" in offenders[0].fix


def test_measure_with_an_empty_merged_folder_is_an_error(tmp_path):
    plate = tmp_path / "plate1"
    (plate / "merged").mkdir(parents=True)
    settings = valid_measure_settings(str(plate / "merged"))
    problems = validate_settings(settings, "measure")
    offenders = [p for p in errors(problems) if "no .npy" in p.message]
    assert offenders


def test_measure_accepts_the_plate_folder_and_resolves_merged(tmp_path):
    """measure_crop appends 'merged' itself, so the plate folder is legal."""
    plate, _merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(plate)
    problems = validate_settings(settings, "measure")
    assert not errors(problems), [str(p) for p in problems]


# ---------------------------------------------------------------------------
# mask dims
# ---------------------------------------------------------------------------

def test_mask_dim_past_the_end_of_the_array_is_an_error(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["pathogen_mask_dim"] = 9

    problems = validate_settings(settings, "measure")

    offenders = settings_named(errors(problems), "pathogen_mask_dim")
    assert offenders, [str(p) for p in problems]
    assert "7 planes" in offenders[0].message
    assert "0-6" in offenders[0].message


def test_last_valid_mask_dim_is_accepted(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["pathogen_mask_dim"] = 6
    assert not errors(validate_settings(settings, "measure"))


def test_channels_list_past_the_end_of_the_array_is_an_error(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=5)
    settings = valid_measure_settings(merged)
    settings["cell_mask_dim"] = 3
    settings["nucleus_mask_dim"] = 4
    settings["pathogen_mask_dim"] = None
    settings["channels"] = [0, 1, 2, 7]

    offenders = settings_named(errors(validate_settings(settings, "measure")), "channels")
    assert offenders
    assert "7" in offenders[0].message


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------

def test_two_objects_on_the_same_channel_is_a_warning_not_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["nucleus_channel"] = 0  # same as cell_channel

    problems = validate_settings(settings, "mask")
    assert not errors(problems), [str(p) for p in problems]
    collisions = [p for p in warnings_of(problems) if "channel 0" in p.message]
    assert collisions
    assert "cell_channel" in collisions[0].setting


def test_two_objects_on_the_same_mask_plane_is_an_error(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["nucleus_mask_dim"] = 4  # same plane as the cell mask

    problems = validate_settings(settings, "measure")
    collisions = [p for p in errors(problems) if "mask plane 4" in p.message]
    assert collisions


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------

def test_str_where_int_expected_is_an_error(tmp_path):
    """The classic CSV round-trip failure: cell_mask_dim comes back as '4'."""
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["cell_min_size"] = "0"

    offenders = settings_named(errors(validate_settings(settings, "measure")), "cell_min_size")
    assert offenders
    assert "str" in offenders[0].message
    assert "int" in offenders[0].message
    assert "CSV" in offenders[0].fix


def test_bool_where_a_number_is_expected_is_a_warning(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["cell_min_size"] = True
    offenders = settings_named(warnings_of(validate_settings(settings, "measure")), "cell_min_size")
    assert offenders


def test_src_as_a_list_is_not_reported_as_a_type_error(tmp_path):
    """expected_types declares src twice and the second entry shadows (str, list)."""
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings([src])
    assert not settings_named(errors(validate_settings(settings, "mask")), "src")


def test_normalize_as_a_percentile_pair_is_not_a_type_error(tmp_path):
    """measure_crop *requires* a list here even though expected_types says bool."""
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["normalize"] = [1, 99]
    assert not settings_named(errors(validate_settings(settings, "measure")), "normalize")


# ---------------------------------------------------------------------------
# typos
# ---------------------------------------------------------------------------

def test_typo_in_a_setting_key_warns_with_the_closest_real_key(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_diamter"] = 60

    offenders = settings_named(warnings_of(validate_settings(settings, "mask")), "cell_diamter")
    assert offenders, "typo not reported"
    assert "cell_diameter" in offenders[0].message
    assert "cell_diameter" in offenders[0].fix


@pytest.mark.parametrize("typo,expected", [
    ("organele_channel", "organelle_channel"),
    ("nucleus_mask_dims", "nucleus_mask_dim"),
    ("batchsize", "batch_size"),
])
def test_typo_suggestions_name_the_intended_key(tmp_path, typo, expected):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings[typo] = 1
    offenders = settings_named(warnings_of(validate_settings(settings, "mask")), typo)
    assert offenders and expected in offenders[0].message


def test_a_key_unlike_any_setting_is_not_reported(tmp_path):
    """Newer pipelines carry keys expected_types has never heard of."""
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["mosaic_csv_out"] = "/tmp/x.csv"
    assert not settings_named(validate_settings(settings, "mask"), "mosaic_csv_out")


# ---------------------------------------------------------------------------
# numeric sanity
# ---------------------------------------------------------------------------

def test_zero_diameter_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_diameter"] = 0
    assert settings_named(errors(validate_settings(settings, "mask")), "cell_diameter")


def test_batch_size_below_one_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["batch_size"] = 0
    assert settings_named(errors(validate_settings(settings, "mask")), "batch_size")


def test_percentile_outside_0_100_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_min_intensity_percentile"] = 140
    assert settings_named(errors(validate_settings(settings, "mask")), "cell_min_intensity_percentile")


def test_inverted_percentile_pair_is_an_error(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["normalize"] = [99, 1]
    offenders = settings_named(errors(validate_settings(settings, "measure")), "normalize")
    assert offenders
    assert "[lower, upper]" in offenders[0].fix


def test_cellpose_probability_outside_its_range_is_a_warning(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_CP_prob"] = 40
    problems = validate_settings(settings, "mask")
    assert not settings_named(errors(problems), "cell_CP_prob")
    assert settings_named(warnings_of(problems), "cell_CP_prob")


def test_unusable_n_jobs_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["n_jobs"] = 0
    assert settings_named(errors(validate_settings(settings, "mask")), "n_jobs")


# ---------------------------------------------------------------------------
# app-specific rules
# ---------------------------------------------------------------------------

def test_mask_run_with_no_segmentation_channel_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    for key in ("cell_channel", "nucleus_channel", "pathogen_channel", "organelle_channel"):
        settings[key] = None
    problems = validate_settings(settings, "mask")
    assert [p for p in errors(problems) if "no segmentation channel" in p.message]


def test_measure_rejects_normalize_true(tmp_path):
    """measure_crop prints a warning and returns when normalize is a bare True."""
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["normalize"] = True
    offenders = settings_named(errors(validate_settings(settings, "measure")), "normalize")
    assert offenders


def test_measure_rejects_an_unknown_normalize_by(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["normalize_by"] = "field"
    assert settings_named(errors(validate_settings(settings, "measure")), "normalize_by")


def test_measure_rejects_an_unsupported_crop_mode(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["crop_mode"] = ["cell", "mitochondria"]
    assert settings_named(errors(validate_settings(settings, "measure")), "crop_mode")


def test_crop_mode_without_the_matching_mask_dim_is_an_error(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["crop_mode"] = ["nucleus"]
    settings["nucleus_mask_dim"] = None
    assert settings_named(errors(validate_settings(settings, "measure")), "nucleus_mask_dim")


def test_the_shipped_single_dialate_png_ratio_is_not_an_error(tmp_path):
    """[0.2] broadcasts to every crop mode, so it must not block a run.

    It used to raise IndexError on the second mode, which is what this rule
    was guarding; measure._per_crop_mode broadcasts it now.
    """
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["crop_mode"] = ["cell", "nucleus"]
    settings["dialate_pngs"] = True
    settings["dialate_png_ratios"] = [0.2]
    assert not settings_named(errors(validate_settings(settings, "measure")), "dialate_png_ratios")


def test_dialate_png_ratios_shorter_than_crop_mode_warns(tmp_path):
    """Short but not a single value: the last entry is reused -- say so."""
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    settings = valid_measure_settings(merged)
    settings["crop_mode"] = ["cell", "nucleus", "pathogen"]
    settings["dialate_pngs"] = True
    settings["dialate_png_ratios"] = [0.2, 0.3]
    problems = validate_settings(settings, "measure")
    assert not settings_named(errors(problems), "dialate_png_ratios")
    assert settings_named(warnings_of(problems), "dialate_png_ratios")


def test_sequencing_requires_the_three_barcode_csvs(tmp_path):
    src = tmp_path / "reads"
    src.mkdir()
    settings = {"src": str(src),
                "grna_csv": str(tmp_path / "nope.csv"),
                "row_csv": str(tmp_path / "nope2.csv"),
                "column_csv": str(tmp_path / "nope3.csv")}
    problems = validate_settings(settings, "sequencing")
    for key in ("grna_csv", "row_csv", "column_csv"):
        assert settings_named(errors(problems), key), f"{key} not checked"


def test_sequencing_is_clean_when_the_csvs_exist(tmp_path):
    src = tmp_path / "reads"
    src.mkdir()
    settings = {"src": str(src)}
    for key in ("grna_csv", "row_csv", "column_csv"):
        path = tmp_path / f"{key}.csv"
        path.write_text("name,sequence\na,ACGT\n")
        settings[key] = str(path)
    assert not errors(validate_settings(settings, "map_barcodes"))


def test_analysis_apps_require_measurements_db(tmp_path):
    plate = tmp_path / "plate1"
    plate.mkdir()
    problems = validate_settings({"src": str(plate)}, "umap")
    offenders = [p for p in errors(problems) if "measurements.db" in p.message]
    assert offenders
    assert "Measure module" in offenders[0].fix


def test_analysis_apps_are_clean_with_a_measurements_db(tmp_path):
    plate, _merged = make_merged_plate(tmp_path, n_planes=7, with_db=True)
    assert not errors(validate_settings({"src": plate}, "umap"))


def test_classify_without_a_model_to_score_with_is_an_error(tmp_path):
    plate, _merged = make_merged_plate(tmp_path, n_planes=7, with_db=True)
    settings = {"src": plate, "train": False, "apply_model_to_dataset": True, "model_path": ""}
    assert settings_named(errors(validate_settings(settings, "classify")), "model_path")


def test_classify_that_trains_first_needs_no_model_path(tmp_path):
    plate, _merged = make_merged_plate(tmp_path, n_planes=7, with_db=True)
    settings = {"src": plate, "train": True, "apply_model_to_dataset": True, "model_path": ""}
    assert not settings_named(errors(validate_settings(settings, "classify")), "model_path")


def test_missing_custom_cellpose_model_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["custom_model"] = str(tmp_path / "no_such_model")
    assert settings_named(errors(validate_settings(settings, "cellpose_masks")), "custom_model")


def test_organelle_unet_without_a_model_path_is_an_error(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["organelle_channel"] = 2
    settings["organelle_method"] = "unet"
    settings["organelle_unet_model_path"] = None
    assert settings_named(errors(validate_settings(settings, "mask")), "organelle_unet_model_path")


# ---------------------------------------------------------------------------
# clean runs
# ---------------------------------------------------------------------------

def test_a_fully_valid_mask_settings_dict_produces_no_errors(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=4, n_fields=6)
    settings = valid_mask_settings(src)
    settings["organelle_channel"] = 3
    settings["channels"] = [0, 1, 2, 3]
    problems = validate_settings(settings, "mask")
    assert not errors(problems), [str(p) for p in problems]


def test_a_fully_valid_measure_settings_dict_produces_no_errors(tmp_path):
    _plate, merged = make_merged_plate(tmp_path, n_planes=7)
    problems = validate_settings(valid_measure_settings(merged), "measure")
    assert not problems, [str(p) for p in problems]


@pytest.mark.parametrize("app,setter", [
    ("mask", "set_default_settings_preprocess_generate_masks"),
    ("measure", "get_measure_crop_settings"),
])
def test_shipped_defaults_validate_clean_against_real_data(tmp_path, app, setter, capsys):
    """The defaults must not themselves trip the checker."""
    import spacr.settings as S

    plate, merged = make_merged_plate(tmp_path, n_planes=7, with_db=True)
    settings = getattr(S, setter)({})
    if app == "mask":
        settings = S._set_organelle_defaults(settings)
        settings["cell_channel"] = 0
        settings["src"] = plate
        # give the mask app raw images to look at
        make_raw_plate(tmp_path / "raw", n_channels=4)
        settings["src"] = make_raw_plate(tmp_path / "raw2", n_channels=4)
    else:
        settings["src"] = merged
    capsys.readouterr()
    problems = validate_settings(settings, app)
    assert not errors(problems), [str(p) for p in problems]


# ---------------------------------------------------------------------------
# app keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias", ["measure", "measure_crop"])
def test_app_aliases_resolve_to_the_same_checks(tmp_path, alias):
    plate = tmp_path / "plate1"
    plate.mkdir()
    problems = validate_settings(valid_measure_settings(str(plate)), alias)
    assert [p for p in errors(problems) if "merged" in p.message]


def test_unknown_app_key_warns_but_still_checks(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_channel"] = 9
    problems = validate_settings(settings, "not_an_app")
    assert [p for p in warnings_of(problems) if "unknown app" in p.message]
    assert settings_named(errors(problems), "cell_channel")


def test_non_dict_settings_is_reported_not_raised():
    problems = validate_settings(["src", "path"], "mask")
    assert len(problems) == 1 and problems[0].severity == ERROR


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

def test_format_report_prints_errors_first_then_warnings():
    problems = [
        Problem(WARNING, "cell_channel", "warned about channel", "do the warning fix"),
        Problem(ERROR, "src", "src is broken", "do the error fix"),
    ]
    report = format_report(problems, {"src": "/data/plate1"}, "mask")
    assert report.index("ERRORS") < report.index("WARNINGS")
    assert report.index("src is broken") < report.index("warned about channel")


def test_format_report_includes_every_fix_line():
    problems = [
        Problem(ERROR, "organelle_channel", "index past the end", "set it to 0-2"),
        Problem(WARNING, "cell_diamter", "unknown key", "rename it to cell_diameter"),
    ]
    report = format_report(problems, {"src": "/data/plate1"}, "mask")
    assert "fix: set it to 0-2" in report
    assert "fix: rename it to cell_diameter" in report
    assert report.count("fix:") == 2


def test_format_report_header_names_the_app_and_the_source():
    report = format_report([], {"src": "/data/plate1"}, "measure")
    assert "measure" in report
    assert "/data/plate1" in report


def test_format_report_says_so_when_there_is_nothing_wrong():
    report = format_report([], {"src": "/data/plate1"}, "mask")
    assert "No problems found" in report
    assert "ERRORS" not in report


def test_format_report_counts_the_errors():
    problems = [Problem(ERROR, "a", "m", "f"), Problem(ERROR, "b", "m", "f")]
    assert "2 errors must be fixed" in format_report(problems, {}, "mask")


def test_format_report_with_only_warnings_says_the_run_would_proceed():
    report = format_report([Problem(WARNING, "a", "m", "f")], {}, "mask")
    assert "No errors" in report


def test_report_of_a_real_validation_names_the_setting_and_the_fix(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["organelle_channel"] = 3
    report = format_report(validate_settings(settings, "mask"), settings, "mask")
    assert "organelle_channel" in report
    assert "fix:" in report


def test_problem_str_carries_the_fix_on_its_own_line():
    text = str(Problem(ERROR, "src", "broken", "fix it"))
    assert text.splitlines()[-1].strip() == "fix: fix it"
    assert Problem(ERROR, "src", "broken", "fix it").is_error
    assert not Problem(WARNING, "src", "broken", "fix it").is_error


def test_problem_without_a_setting_renders_without_brackets():
    assert str(Problem(WARNING, "", "something general", "do this")).startswith("something general")


# ---------------------------------------------------------------------------
# describe_plan
# ---------------------------------------------------------------------------

def test_describe_plan_for_mask_reports_app_source_counts_channels_and_output(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3, n_fields=4)
    settings = valid_mask_settings(src)
    plan = describe_plan(settings, "mask")

    assert "spacr.core.preprocess_generate_masks" in plan
    assert src in plan
    assert "12 raw image files" in plan          # 4 fields x 3 channels
    assert "cell: channel 0" in plan
    assert "nucleus: channel 1" in plan
    assert os.path.join(src, "masks") in plan
    assert os.path.join(src, "merged") in plan
    assert "~4 fields" in plan


def test_describe_plan_for_mask_names_the_diameter_when_set(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["cell_diameter"] = 60
    assert "diameter 60 px" in describe_plan(settings, "mask")


def test_describe_plan_for_measure_reports_planes_objects_and_the_database(tmp_path):
    plate, merged = make_merged_plate(tmp_path, n_planes=7, n_files=5)
    settings = valid_measure_settings(merged)
    plan = describe_plan(settings, "measure")

    assert "spacr.measure.measure_crop" in plan
    assert "5 merged arrays" in plan
    assert "7 (indices 0-6)" in plan
    assert "cell: mask plane 4" in plan
    assert "cell PNGs at [224, 224]" in plan
    assert os.path.join(plate, "measurements", "measurements.db") in plan
    assert "~5 fields to measure" in plan


def test_describe_plan_says_where_measure_would_actually_look(tmp_path):
    plate, merged = make_merged_plate(tmp_path, n_planes=7)
    plan = describe_plan(valid_measure_settings(plate), "measure")
    assert merged in plan


def test_describe_plan_flags_test_mode(tmp_path):
    src = make_raw_plate(tmp_path, n_channels=3)
    settings = valid_mask_settings(src)
    settings["test_mode"] = True
    assert "test mode" in describe_plan(settings, "mask")


def test_describe_plan_reports_a_missing_source_instead_of_raising(tmp_path):
    plan = describe_plan({"src": str(tmp_path / "gone")}, "mask")
    assert "does not exist" in plan


def test_describe_plan_survives_a_settings_dict_with_almost_nothing_in_it():
    plan = describe_plan({}, "mask")
    assert "not set" in plan


def test_describe_plan_rejects_non_dict_settings():
    assert "not a dict" in describe_plan(["src"], "mask")


def test_describe_plan_lists_every_source_folder(tmp_path):
    a = make_raw_plate(tmp_path / "a", n_channels=3)
    b = make_raw_plate(tmp_path / "b", n_channels=3)
    plan = describe_plan(valid_mask_settings([a, b]), "mask")
    assert a in plan and b in plan


# ---------------------------------------------------------------------------
# cost: this has to be cheap enough to run before every job
# ---------------------------------------------------------------------------

def test_validate_does_not_import_torch_or_cellpose():
    """spacr.validate itself must stay light; that is its whole reason to exist."""
    import inspect

    import spacr.validate as V

    source = inspect.getsource(V)
    for banned in ("import torch", "import cellpose", "from cellpose"):
        assert banned not in source, f"spacr.validate must not {banned}"


def test_validate_reads_only_one_array_header_not_the_plate(tmp_path, monkeypatch):
    """A 200-field plate must cost the same as a 3-field one."""
    _plate, merged = make_merged_plate(tmp_path, n_planes=7, n_files=40)

    import numpy as _np

    calls = []
    real_load = _np.load

    def counting_load(path, *args, **kwargs):
        calls.append(path)
        assert kwargs.get("mmap_mode") == "r", "arrays must be memory-mapped, not read"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(_np, "load", counting_load)
    validate_settings(valid_measure_settings(merged), "measure")
    assert len(calls) == 1, f"peeked at {len(calls)} arrays, expected 1"


# ---------------------------------------------------------------------------
# expected_types must contain TYPES
# ---------------------------------------------------------------------------

def test_every_expected_type_is_actually_a_type():
    """``validate_settings`` ends in ``isinstance(value, types)``.

    Two entries used to hold the *value* ``None`` rather than ``type(None)``
    -- ``'sample': None`` and ``"x_lim": (list, None)`` -- and isinstance
    rejects that with ``TypeError: isinstance() arg 2 must be a type``. The
    preflight check crashed on any run that set ``sample``, which is the one
    place it must not: it exists to report problems, not to raise them.
    """
    from spacr.settings import expected_types

    bad = {}
    for key, declared in expected_types.items():
        for entry in (declared if isinstance(declared, tuple) else (declared,)):
            if not isinstance(entry, type):
                bad.setdefault(key, []).append(entry)
    assert not bad, bad


def test_a_declared_sample_does_not_crash_the_preflight(tmp_path):
    from spacr.validate import validate_settings

    # The call itself is the assertion: it used to raise TypeError.
    problems = validate_settings({"src": str(tmp_path), "sample": 500,
                                  "x_lim": [-1, 1]}, "classify")
    typed = [p for p in problems if p.setting in ("sample", "x_lim")]
    assert not typed, [str(p) for p in typed]


# ---------------------------------------------------------------------------
# The resource card -- the second half of the dry-run card
#
# The property under test everywhere here is HONESTY, not arithmetic. A card
# that guesses a number the run then blows past is worse than no card,
# because the user stopped watching. So each test pins either "this figure
# was measured off the machine" or "this figure was refused and named".
# ---------------------------------------------------------------------------

def _plate_with_merged(tmp_path, name="plate", fields=6, shape=(64, 64, 4),
                       dtype="uint16"):
    import numpy as np

    src = tmp_path / name
    merged = src / "merged"
    merged.mkdir(parents=True)
    for i in range(fields):
        np.save(merged / f"f{i}.npy", np.zeros(shape, dtype))
    return src


def test_the_card_measures_one_field_rather_than_assuming_it(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path, shape=(128, 96, 3), dtype="uint16")
    card = describe_resources({"src": str(src), "cell_channel": 0,
                               "n_jobs": 2}, "mask")
    # Shape and dtype come off the header of a real array, memory-mapped.
    assert "128×96×3 uint16" in card
    assert "72.0 KB" in card                     # 128*96*3*2 bytes


def test_memory_is_reported_as_a_floor_and_says_so(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path)
    card = describe_resources({"src": str(src), "cell_channel": 0,
                               "n_jobs": 4}, "mask")
    assert "at least" in card
    assert "4 worker(s) × one field each" in card
    # The floor must never be mistaken for the peak.
    assert "floor, not a peak" in card
    assert "spacr.benchmark" in card


def test_an_n_jobs_that_cannot_fit_is_called_out_with_one_that_can(tmp_path,
                                                                  monkeypatch):
    from spacr import validate as V

    src = _plate_with_merged(tmp_path, shape=(64, 64, 4))
    # 32 KB per field; pretend the machine has the reserve plus room for two.
    monkeypatch.setattr("spacr.benchmark.available_memory_bytes",
                        lambda: V._MEM_RESERVE + 64 * 64 * 4 * 2 * 2)
    card = V.describe_resources({"src": str(src), "cell_channel": 0,
                                 "n_jobs": 64}, "mask")
    assert "does not fit" in card
    assert "2 worker(s) would fit" in card


def test_an_n_jobs_that_fits_is_not_warned_about(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path, shape=(32, 32, 2))
    card = describe_resources({"src": str(src), "cell_channel": 0,
                               "n_jobs": 2}, "mask")
    assert "does not fit" not in card


def test_the_png_crop_tree_is_refused_rather_than_guessed(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path)
    card = describe_resources({"src": str(src), "cell_mask_dim": 4,
                               "save_png": True,
                               "crop_mode": ["cell", "nucleus"],
                               "n_jobs": 2}, "measure")
    assert "cannot be projected" in card
    assert "2 crop mode(s)" in card
    # ...and no total is offered for it.
    assert "would write" not in card


def test_the_mask_projection_names_its_parts_and_the_npy_trap(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path, fields=4, shape=(64, 64, 3))
    card = describe_resources({"src": str(src), "cell_channel": 0,
                               "nucleus_channel": 1, "n_jobs": 1}, "mask")
    assert "would write" in card
    assert "2 mask stack(s)" in card
    # The trap: merged/ is uncompressed, so it can be bigger than the input.
    assert "uncompressed .npy" in card


def test_not_enough_disk_is_stated_when_the_projection_exceeds_free_space(
        tmp_path, monkeypatch):
    from spacr import validate as V

    src = _plate_with_merged(tmp_path, fields=4, shape=(64, 64, 3))
    monkeypatch.setattr(V, "_free_disk", lambda path: 1024)
    card = V.describe_resources({"src": str(src), "cell_channel": 0,
                                 "n_jobs": 1}, "mask")
    assert "NOT ENOUGH DISK" in card


def test_a_source_with_nothing_readable_says_so_rather_than_printing_zeros(
        tmp_path):
    from spacr.validate import describe_resources

    empty = tmp_path / "empty"
    empty.mkdir()
    card = describe_resources({"src": str(empty)}, "mask")
    assert "nothing to project" in card
    assert "0 B" not in card


def test_settings_that_are_not_a_dict_do_not_raise():
    from spacr.validate import describe_resources

    assert "not a dict" in describe_resources(["src"], "mask")


def test_a_truncated_directory_listing_is_reported_as_a_floor(tmp_path,
                                                              monkeypatch):
    from spacr import validate as V

    src = _plate_with_merged(tmp_path, fields=4)
    monkeypatch.setattr(V, "_SIZE_BUDGET", 2)
    card = V.describe_resources({"src": str(src), "cell_channel": 0}, "mask")
    assert "at least" in card


def test_the_preflight_prints_the_card_after_the_plan(tmp_path):
    from spacr.validate import run_preflight

    src = _plate_with_merged(tmp_path)
    out = []
    run_preflight({"src": str(src), "cell_channel": 0, "n_jobs": 2}, "mask",
                  printer=out.append)
    text = "\n".join(out)
    assert text.index("Plan —") < text.index("Resources —")


def test_a_card_that_raises_does_not_take_the_preflight_with_it(tmp_path,
                                                               monkeypatch):
    """A pre-flight that raises has denied the user the report it exists for."""
    from spacr import validate as V

    def boom(settings, app_key=""):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(V, "describe_resources", boom)
    out = []
    problems = V.run_preflight({"src": str(tmp_path), "cell_channel": 0},
                               "mask", printer=out.append)
    text = "\n".join(out)
    assert "could not be projected: the disk went away" in text
    assert "Plan —" in text                  # the rest of the report survived
    assert isinstance(problems, list)


def test_the_gpu_line_is_only_offered_where_a_gpu_is_used(tmp_path):
    from spacr.validate import describe_resources

    src = _plate_with_merged(tmp_path)

    def labels(card):
        # Rows are two-space indented, then label, then two spaces.
        return {line[2:].split("  ")[0].strip()
                for line in card.splitlines() if line.startswith("  ")}

    # Measure runs no model; a VRAM figure there would be noise.
    assert "gpu" not in labels(describe_resources(
        {"src": str(src), "cell_mask_dim": 4, "n_jobs": 2}, "measure"))
    # Mask loads Cellpose, so the device is worth stating.
    assert "gpu" in labels(describe_resources(
        {"src": str(src), "cell_channel": 0, "n_jobs": 2}, "mask"))


def test_free_disk_walks_up_to_a_filesystem_that_exists(tmp_path):
    from spacr.validate import _free_disk

    # A path several levels below anything that exists still resolves.
    assert _free_disk(str(tmp_path / "a" / "b" / "c")) is not None


def test_fmt_bytes_says_unknown_rather_than_zero(tmp_path):
    from spacr.validate import _fmt_bytes

    assert _fmt_bytes(None) == "unknown"
    assert _fmt_bytes(512) == "512 B"
    assert _fmt_bytes(1024 ** 4 * 3) == "3.0 TB"
    assert _fmt_bytes(1024 ** 5) == "1024.0 TB"     # no unit beyond TB


def test_without_torch_loaded_the_card_says_it_did_not_look(tmp_path,
                                                            monkeypatch):
    """Silence would read as "no GPU", which is a different answer."""
    import sys as _sys

    from spacr.validate import describe_resources

    monkeypatch.delitem(_sys.modules, "torch", raising=False)
    card = describe_resources({"src": str(_plate_with_merged(tmp_path)),
                               "cell_channel": 0, "n_jobs": 1}, "mask")
    assert "torch is not loaded" in card


def test_a_gpu_that_cannot_be_questioned_drops_the_row(tmp_path, monkeypatch):
    """A torch whose cuda calls raise leaves no row rather than a wrong one."""
    import sys as _sys
    import types

    from spacr import validate as V

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: (_ for _ in ()).throw(RuntimeError("no ctx"))))
    monkeypatch.setitem(_sys.modules, "torch", fake)
    assert V._gpu_line() is None


def test_a_cpu_only_torch_says_segmentation_runs_on_the_cpu(monkeypatch):
    import sys as _sys
    import types

    from spacr import validate as V

    monkeypatch.setitem(_sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)))
    assert "would run on the CPU" in V._gpu_line()


def test_an_unreadable_npy_is_skipped_rather_than_fatal(tmp_path):
    from spacr.validate import _array_footprint

    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "broken.npy").write_bytes(b"not an npy at all")
    assert _array_footprint(str(merged)) is None
    assert _array_footprint(str(tmp_path / "nothing")) is None


def test_a_file_that_vanishes_mid_scan_does_not_break_the_total(tmp_path,
                                                                monkeypatch):
    from spacr import validate as V

    src = _plate_with_merged(tmp_path, fields=3)
    real = os.path.getsize

    def flaky(path):
        if path.endswith("f1.npy"):
            raise OSError("gone")
        return real(path)

    monkeypatch.setattr(os.path, "getsize", flaky)
    total, counted, truncated = V._dir_bytes(str(src / "merged"), (".npy",))
    assert counted == 2 and total > 0 and not truncated


def test_a_zero_dimensional_array_is_skipped_not_sized(tmp_path):
    """A 0-d .npy has no shape to report; the next candidate is tried."""
    import numpy as np

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "a_scalar.npy", np.array(5))
    np.save(merged / "b_real.npy", np.zeros((8, 8, 2), "uint8"))
    from spacr.validate import _array_footprint

    shape, dtype, nbytes = _array_footprint(str(merged))
    assert shape == (8, 8, 2) and dtype == "uint8" and nbytes == 128


def test_a_filesystem_that_never_answers_gives_up_rather_than_looping(
        monkeypatch, tmp_path):
    from spacr import validate as V

    def always_fails(path):
        raise OSError("no such filesystem")

    monkeypatch.setattr(V.shutil, "disk_usage", always_fails)
    assert V._free_disk(str(tmp_path / "deep" / "er")) is None
    assert V._free_disk("/") is None          # the walk-up hits the root


def test_a_working_cuda_device_is_named_with_its_free_memory(monkeypatch):
    import sys as _sys
    import types

    from spacr import validate as V

    monkeypatch.setitem(_sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda i: "Test GPU",
            mem_get_info=lambda i: (2 * 1024 ** 3, 8 * 1024 ** 3))))
    line = V._gpu_line()
    assert "Test GPU" in line and "2.0 GB free of 8.0 GB" in line
    # The fact that decides whether a VRAM figure is even relevant.
    assert "VRAM follows batch_size and not field size" in line


def test_an_unmeasurable_memory_figure_drops_the_row_rather_than_lying(
        tmp_path, monkeypatch):
    from spacr import benchmark as B
    from spacr.validate import describe_resources

    def boom():
        raise RuntimeError("/proc/meminfo is not here")

    monkeypatch.setattr(B, "available_memory_bytes", boom)
    card = describe_resources({"src": str(_plate_with_merged(tmp_path)),
                               "cell_channel": 0, "n_jobs": 2}, "mask")
    assert "memory free" not in card
    # ...and with no free figure there is nothing to compare against.
    assert "does not fit" not in card
    assert "one field" in card               # what could be measured survives
