"""The extraction preview counts what a drop would produce, without pixels.

The planners are driven over real folders of real (tiny) TIFFs so the row
count and the canonical names are the ones the extraction would actually
write, rather than the ones a stubbed walker was told to return.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt import ingest_preview as ip


@dataclass
class _Described:
    """The shape of what `spacr.qt.multi_format` reports for a container."""
    path: str
    n_fields: int = 1
    n_channels: int = 1
    n_timepoints: int = 1


def _tiny_tif(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

def test_a_container_expands_to_one_row_per_time_field_and_channel():
    rows = ip.plan_container_extraction(
        _Described("/data/movie.nd2", n_fields=2, n_channels=3,
                   n_timepoints=4))
    assert len(rows) == 2 * 3 * 4
    assert {r["well"] for r in rows} == {"plate1_A01"}
    assert rows[0]["canonical"] == "plate1_A01_T0001F001L01C01.tif"
    assert rows[-1]["canonical"] == "plate1_A01_T0004F002L01C03.tif"
    assert all(set(r) == set(ip.ROW_COLUMNS) for r in rows)


def test_a_container_that_reports_nothing_still_yields_one_plane():
    """Zero fields is what an unreadable header reports; a preview of no rows
    would say the file is empty rather than that it was not read."""
    rows = ip.plan_container_extraction(
        _Described("/data/x.czi", n_fields=0, n_channels=0, n_timepoints=0))
    assert len(rows) == 1
    assert rows[0]["field"] == rows[0]["channel"] == rows[0]["time"] == 1


def test_the_plate_and_well_a_caller_names_reach_the_canonical_name():
    rows = ip.plan_container_extraction(_Described("/data/x.lif"),
                                        plate="plate7", well="B11")
    assert rows[0]["well"] == "plate7_B11"
    assert rows[0]["canonical"].startswith("plate7_B11_T0001F001")


# --------------------------------------------------------------------------
# folders
# --------------------------------------------------------------------------

def test_a_path_that_is_not_a_folder_plans_nothing(tmp_path):
    assert ip.plan_folder_extraction(tmp_path / "absent") == []


def test_a_folder_with_no_images_plans_nothing(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing here")
    assert ip.plan_folder_extraction(tmp_path) == []


def test_every_image_under_a_dropped_folder_becomes_one_row(tmp_path):
    for well in ("A01", "A02"):
        for field in (1, 2):
            _tiny_tif(tmp_path / well / f"f{field}.tif")
    rows = ip.plan_folder_extraction(tmp_path, plate="plate3")
    assert len(rows) == 4
    assert all(r["plate"] == "plate3" for r in rows)
    assert all(r["canonical"].endswith(".tif") for r in rows)
    assert len({r["original"] for r in rows}) == 4


def test_the_preview_is_capped_without_walking_the_tree_twice(tmp_path):
    for i in range(10):
        _tiny_tif(tmp_path / "A01" / f"f{i:02d}.tif")
    rows = ip.plan_folder_extraction(tmp_path, limit=3)
    assert len(rows) == 3
    # The cap keeps the SMALLEST paths, so a preview is stable across runs.
    assert [Path(r["original"]).name for r in rows] == [
        "f00.tif", "f01.tif", "f02.tif"]


def test_no_cap_plans_the_whole_folder(tmp_path):
    for i in range(6):
        _tiny_tif(tmp_path / "A01" / f"f{i}.tif")
    assert len(ip.plan_folder_extraction(tmp_path, limit=None)) == 6


def test_a_caller_that_already_walked_the_tree_passes_its_files_in(tmp_path):
    paths = [tmp_path / "A01" / f"f{i}.tif" for i in range(3)]
    for path in paths:
        _tiny_tif(path)
    rows = ip.plan_folder_extraction(tmp_path / "absent", files=iter(paths),
                                     template=None)
    assert len(rows) == 3


def test_detection_that_already_ran_and_found_nothing_is_not_run_again(
        tmp_path, monkeypatch):
    """`template=None` means "detection ran, nothing matched" -- distinct
    from omitting it, which means "detect it here"."""
    _tiny_tif(tmp_path / "A01" / "f1.tif")
    from spacr.qt import folder_metadata as fm

    def _must_not_run(_root):
        raise AssertionError("the tree was walked a second time")

    monkeypatch.setattr(fm, "detect_folder_metadata", _must_not_run)
    rows = ip.plan_folder_extraction(tmp_path, template=None)
    assert len(rows) == 1


# --------------------------------------------------------------------------
# rows in, mappings out
# --------------------------------------------------------------------------

def test_an_edited_row_round_trips_back_into_a_mapping():
    rows = [{"original": "/data/x.tif", "plate": "plate2",
             "well": "plate2_C03", "field": "2", "channel": "3",
             "time": "4", "canonical": "plate2_C03_T0004F002L01C03.tif"}]
    mapping, = ip.rows_to_mappings(rows)
    assert mapping.plate == "plate2"
    assert (mapping.field, mapping.channel, mapping.time) == (2, 3, 4)
    assert ip.mapping_to_row(mapping) == {
        "original": "/data/x.tif", "plate": "plate2", "well": "plate2_C03",
        "field": 2, "channel": 3, "time": 4,
        "canonical": "plate2_C03_T0004F002L01C03.tif"}


def test_a_blank_index_becomes_one_rather_than_zero():
    """An emptied table cell must not produce `F000`, which no reader of the
    Yokogawa convention expects."""
    mapping, = ip.rows_to_mappings([{"original": "x", "field": "",
                                     "channel": 0, "time": None}])
    assert (mapping.field, mapping.channel, mapping.time) == (1, 1, 1)


# --------------------------------------------------------------------------
# the one-line summary
# --------------------------------------------------------------------------

def test_an_empty_preview_says_there_is_nothing_to_extract():
    assert ip.summarize_rows([]) == "no images to extract"


def test_a_single_timepoint_is_not_mentioned():
    rows = ip.plan_container_extraction(
        _Described("/data/x.nd2", n_fields=2, n_channels=2))
    said = ip.summarize_rows(rows)
    assert said == "4 images, 1 well(s), 2 field(s), 2 channel(s)"
    assert "timepoint" not in said


def test_a_time_series_says_how_many_timepoints():
    """A movie and a single snapshot summarise to the same counts otherwise,
    and the difference is the whole reason to look at the preview."""
    rows = ip.plan_container_extraction(
        _Described("/data/x.nd2", n_fields=2, n_channels=2, n_timepoints=5))
    said = ip.summarize_rows(rows)
    assert said.endswith("5 timepoint(s)")
    assert said.startswith("20 images")
