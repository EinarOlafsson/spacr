"""Read the two vendor-documented EVOS variants without inventing a Z plane."""

import re

import pytest

from spacr.regex_infer import (
    _metadata_convention,
    _metadata_match,
    _metadata_pattern,
)

STACK_NAME = "scan_R_p2_z1_0_B03f01d0.tif"
MANUAL_NAME = "image_0002_Top Slide_R_p00_0_A01f00d0.TIF"
OPTIONAL_Z_PATTERN = (
    r"(?P<plateID>.+)_(?P<AID>[RDM])_p(?P<timeID>\d+)"
    r"(?:_z(?P<sliceID>\d+))?_(?P<laserID>\d+)_(?P<wellID>[A-Z]\d{2})"
    r"f(?P<fieldID>\d{2})d(?P<chanID>\d)\.(?i:tif)$"
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (STACK_NAME, {"plateID": "scan", "AID": "R", "timeID": "2",
                      "sliceID": "1", "laserID": "0", "wellID": "B03",
                      "fieldID": "01", "chanID": "0"}),
        (MANUAL_NAME, {"plateID": "image_0002_Top Slide", "AID": "R",
                       "timeID": "00", "sliceID": None, "laserID": "0",
                       "wellID": "A01", "fieldID": "00", "chanID": "0"}),
    ],
)
def test_vendor_literals_retain_every_observed_token(name, expected):
    pattern = re.compile(_metadata_pattern("evos", name.rsplit(".", 1)[1]))
    found = pattern.match(name)
    assert found is not None
    assert found.groupdict() == expected


def test_documented_manual_name_is_not_misidentified_as_arrayscan():
    assert _metadata_match(MANUAL_NAME, "evos") is not None
    assert _metadata_match(MANUAL_NAME, "arrayscan") is None


def test_optional_slice_reader_keeps_none_instead_of_a_synthetic_plane(tmp_path):
    from spacr.utils import _extract_filename_metadata

    grouped = _extract_filename_metadata(
        [MANUAL_NAME], str(tmp_path), re.compile(OPTIONAL_Z_PATTERN), "evos"
    )
    assert grouped == {
        ("image_0002_Top Slide", "A01", "0", "0", "0", None):
            [str(tmp_path / MANUAL_NAME)]
    }


@pytest.mark.parametrize(
    ("name", "key"),
    [(STACK_NAME, ("scan", "B03", "1", "0", "2", "1")),
     (MANUAL_NAME, ("image_0002_Top Slide", "A01", "0", "0", "0", None))],
)
def test_actual_evos_reader_keeps_known_and_absent_slices(name, key, tmp_path):
    from spacr.utils import _extract_filename_metadata

    pattern = re.compile(_metadata_pattern("evos", name.rsplit(".", 1)[1]))
    grouped = _extract_filename_metadata([name], str(tmp_path), pattern, "evos")
    assert grouped == {key: [str(tmp_path / name)]}


def test_import_metadata_retains_unknown_z_and_original_time_field():
    from spacr.convert import _parse_by_convention

    assert _metadata_match(MANUAL_NAME, "evos") == {
        "plateID": "image_0002_Top Slide", "AID": "R", "timeID": "00",
        "laserID": "0", "wellID": "A01", "fieldID": "00", "chanID": "0",
    }
    assert _parse_by_convention(MANUAL_NAME, "evos") == {
        "plate": "image_0002_Top Slide", "well": "A01", "well_token": "A01",
        "field": "0", "field_number": 0, "channel": "C1", "z": None, "t": 0,
    }


def test_single_documented_manual_tiff_reaches_actual_mask_ingest(tmp_path):
    import numpy as np
    import tifffile

    from spacr.io import _rename_and_organize_image_files

    pixels = np.arange(35, dtype=np.uint16).reshape(5, 7)
    source = tmp_path / MANUAL_NAME
    tifffile.imwrite(source, pixels)
    source_bytes = source.read_bytes()
    channels = _rename_and_organize_image_files(
        str(tmp_path), _metadata_pattern("evos", "TIF"), metadata_type="evos",
        img_format=".TIF", save_original_images=True,
    )
    assert channels == 1
    stacks = list((tmp_path / "stack").glob("*.npy"))
    assert len(stacks) == 1
    assert np.array_equal(np.load(stacks[0]), pixels[:, :, None])
    assert (tmp_path / "orig" / MANUAL_NAME).read_bytes() == source_bytes


@pytest.mark.parametrize(
    "name",
    [
        "image_0002_Top Slide_R_p00_z_0_A01f00d0.TIF",
        "image_0002_Top Slide_R_p00_z-1_0_A01f00d0.TIF",
        "image_0002_Top Slide_R_p00_z1_A01f00d0.TIF",
        "image_0002_Top Slide_R_p00_0_A01f00.TIF",
        "image_0002_Top Slide_R_p00_0_A01f00d0.TIF.bak",
        "A01f01d2.TIF",
    ],
)
def test_optional_z_does_not_admit_malformed_or_other_vendor_names(name):
    assert _metadata_match(name, "evos") is None


def test_vendor_documentation_confirms_only_the_two_recorded_literals():
    record = _metadata_convention("evos")
    assert record["examples"] == (STACK_NAME, MANUAL_NAME)
    assert record["status"] == "confirmed"
    assert "198" in record["source"] and "41" in record["source"]
    assert "save-dialog" in record["source"]
    assert "not" in record["source"] and "acquisition" in record["source"]
