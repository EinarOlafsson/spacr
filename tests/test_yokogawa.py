"""
Tests for spacr's Yokogawa image-metadata pipeline.

Exercises the regex builders and filename metadata extractor against the
two Yokogawa naming conventions the package supports:

  * CellVoyager  {plate}_{well}_T*F*L*A*Z*C*.tif
  * CQ1          W{well_id}F*T*Z*C*.tif   (well_id -> A01..P24)

These tests use ONLY the synthetic directories from conftest — no real
microscopy data required.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import spacr.utils as U


# ---------------------------------------------------------------------------
# Regex builder for both Yokogawa formats
# ---------------------------------------------------------------------------

def test_cellvoyager_regex_matches_synthetic_filenames(yokogawa_cellvoyager_dir):
    from spacr.utils import _get_regex  # not always exported; import direct
    # _get_regex signature: (metadata_type, img_format, custom_regex=None)
    regex_str = _get_regex("cellvoyager", "tif", None)
    regex = re.compile(regex_str)
    for entry in yokogawa_cellvoyager_dir["manifest"]:
        fname = os.path.basename(entry["path"])
        m = regex.match(fname)
        assert m is not None, f"CellVoyager regex failed to match {fname!r}"
        assert m.group("plateID") == entry["plate"]
        assert m.group("wellID") == entry["well"]
        assert m.group("fieldID") == entry["field"]
        assert m.group("chanID") == entry["channel"]


def test_cq1_regex_matches_synthetic_filenames(yokogawa_cq1_dir):
    from spacr.utils import _get_regex
    regex_str = _get_regex("cq1", "tif", None)
    regex = re.compile(regex_str)
    for entry in yokogawa_cq1_dir["manifest"]:
        fname = os.path.basename(entry["path"])
        m = regex.match(fname)
        assert m is not None, f"CQ1 regex failed to match {fname!r}"
        # The CQ1 regex captures wellID as an int-like string, not the A01 form.
        assert m.group("wellID") == str(entry["well_id"])
        assert m.group("fieldID") == entry["field"]
        assert m.group("chanID") == entry["channel"]


# ---------------------------------------------------------------------------
# _extract_filename_metadata groups images by (plate, well, field, channel, ...)
# ---------------------------------------------------------------------------

def test_extract_filename_metadata_cellvoyager(yokogawa_cellvoyager_dir):
    from spacr.utils import _get_regex, _extract_filename_metadata
    src = str(yokogawa_cellvoyager_dir["src"])
    regex = re.compile(_get_regex("cellvoyager", "tif", None))
    filenames = sorted(os.listdir(src))

    images_by_key = _extract_filename_metadata(
        filenames, src, regex, metadata_type="cellvoyager"
    )

    # 2 wells * 2 fields * 2 channels = 8 unique keys (single z, timepoint).
    # Each key should have exactly 1 file since the manifest has no duplicates.
    assert len(images_by_key) == 8
    for key, paths in images_by_key.items():
        # key = (plate, well, field, channel, timeID, sliceID)
        assert len(key) == 6
        assert len(paths) == 1


def test_extract_filename_metadata_cq1_converts_well_id(yokogawa_cq1_dir, capsys):
    from spacr.utils import _get_regex, _extract_filename_metadata
    src = str(yokogawa_cq1_dir["src"])
    regex = re.compile(_get_regex("cq1", "tif", None))
    filenames = sorted(os.listdir(src))

    images_by_key = _extract_filename_metadata(
        filenames, src, regex, metadata_type="cq1"
    )

    # 2 wells * 2 fields * 2 channels = 8 keys.
    assert len(images_by_key) == 8
    # The wells in the keys should be the human A01/B01 form, not raw ints.
    wells_seen = {k[1] for k in images_by_key.keys()}
    assert "A01" in wells_seen
    assert "B01" in wells_seen


# ---------------------------------------------------------------------------
# The CQ1 well-id encoder is exercised by both spacr and the manifest.
# ---------------------------------------------------------------------------

def test_cq1_well_id_boundaries():
    # 384-well grid: A01..P24
    assert U._convert_cq1_well_id(1) == "A01"
    assert U._convert_cq1_well_id(24) == "A24"
    assert U._convert_cq1_well_id(25) == "B01"
    assert U._convert_cq1_well_id(384) == "P24"


def test_cq1_manifest_well_labels_agree_with_encoder(yokogawa_cq1_dir):
    for entry in yokogawa_cq1_dir["manifest"]:
        assert U._convert_cq1_well_id(entry["well_id"]) == entry["well"]


# ---------------------------------------------------------------------------
# Filename → grouping keys stability
# ---------------------------------------------------------------------------

def test_extract_filename_metadata_no_leftover_files(yokogawa_cellvoyager_dir):
    """All manifest files should end up somewhere in the grouping — nothing
    silently dropped."""
    from spacr.utils import _get_regex, _extract_filename_metadata
    src = str(yokogawa_cellvoyager_dir["src"])
    regex = re.compile(_get_regex("cellvoyager", "tif", None))
    filenames = sorted(os.listdir(src))
    grouped = _extract_filename_metadata(filenames, src, regex, metadata_type="cellvoyager")
    all_grouped_paths = {p for paths in grouped.values() for p in paths}
    all_manifest_paths = {entry["path"] for entry in yokogawa_cellvoyager_dir["manifest"]}
    assert all_grouped_paths == all_manifest_paths
