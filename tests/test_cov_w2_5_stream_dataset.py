"""A streamed dataset counts what it could not write.

A training set that is quietly short by a field is a training set built from
a different screen than the selection table describes, and nothing else in
the run would say so. Every path below that fails to produce a crop has to
land in the report's ``missing`` count or ``trouble`` list, so the tests
break the inputs one at a time and read the report.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr import stream_dataset as sd


@pytest.fixture
def merged(tmp_path):
    """Two merged fields, each with three labelled objects on the last plane."""
    folder = tmp_path / "merged"
    folder.mkdir()
    for field in (1, 2):
        stack = np.zeros((20, 20, 3), dtype=np.int32)
        stack[..., 0] = 7
        stack[2:6, 2:6, 2] = 1
        stack[8:12, 8:12, 2] = 2
        stack[14:18, 14:18, 2] = 3
        np.save(folder / f"plate1_A01_{field}_0.npy", stack)
    return str(folder)


@pytest.fixture
def selection(merged):
    """The selection table those two fields produce."""
    return sd.selection_from_arrays(merged)


def _collect(sink):
    def write(path, crop):
        sink.append((path, np.asarray(crop).shape))
    return write


# ---------------------------------------------------------------------------
# building the table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_an_empty_object_table_refuses_to_build_a_selection(frame):
    """Zero rows is an error with a reason, not an empty dataset."""
    with pytest.raises(ValueError) as caught:
        sd.selection_from_objects(frame)

    assert "the object table is empty" in str(caught.value)


def test_a_table_with_only_the_generic_object_key_still_builds():
    """``objectID`` stands in when the per-type column is absent."""
    frame = pd.DataFrame({"plateID": ["p1"] * 4, "objectID": [1, 2, 3, 4]})

    out = sd.selection_from_objects(frame, object_array="nucleus")

    assert list(out["objectID"]) == ["1", "2", "3", "4"]
    assert set(out["object_array"]) == {"nucleus"}
    assert list(out.columns) == list(sd.SELECTION_COLUMNS)


def test_a_stack_that_cannot_be_read_is_skipped_not_fatal(tmp_path, caplog):
    """One unreadable array must not stop the other fields being scanned."""
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / "plate1_A01_1_0.npy").write_bytes(b"not a numpy file")
    good = np.zeros((8, 8, 3), dtype=np.int32)
    good[1:4, 1:4, 2] = 4
    np.save(folder / "plate1_A01_2_0.npy", good)

    with caplog.at_level("DEBUG", logger="spacr.stream_dataset"):
        out = sd.selection_from_arrays(str(folder))

    assert list(out["objectID"]) == ["4"]
    assert "could not read" in caplog.text


def test_arrays_holding_only_background_are_an_error_that_says_so(tmp_path):
    """An all-zero mask is nothing to stream, and the message counts them."""
    folder = tmp_path / "merged"
    folder.mkdir()
    for field in (1, 2):
        np.save(folder / f"plate1_A01_{field}_0.npy",
                np.zeros((8, 8, 3), dtype=np.int32))

    with pytest.raises(FileNotFoundError) as caught:
        sd.selection_from_arrays(str(folder))

    assert "2 .npy stack(s)" in str(caught.value)
    assert "other than background" in str(caught.value)


def test_a_two_dimensional_array_is_its_own_mask(tmp_path):
    """A flat label image has no plane to select; it IS the plane."""
    folder = tmp_path / "merged"
    folder.mkdir()
    flat = np.zeros((8, 8), dtype=np.int32)
    flat[2:5, 2:5] = 9
    np.save(folder / "plate1_A01_1_0.npy", flat)

    out = sd.selection_from_arrays(str(folder))

    assert list(out["objectID"]) == ["9"]


# ---------------------------------------------------------------------------
# cutting
# ---------------------------------------------------------------------------

def test_a_flat_image_is_cut_as_a_single_channel():
    """A 2-D stack gains its channel axis rather than being refused."""
    stack = np.arange(144, dtype=float).reshape(12, 12)
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[3:7, 4:9] = 5

    out = sd.cut(stack, mask, 5)

    assert out.shape == (4, 5, 1)
    assert out[0, 0, 0] == stack[3, 4]


def test_only_the_named_channels_come_back():
    """A channel list narrows the crop to those planes, in that order."""
    stack = np.zeros((12, 12, 3), dtype=float)
    stack[..., 0], stack[..., 1], stack[..., 2] = 1.0, 2.0, 3.0
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[3:7, 4:9] = 5

    out = sd.cut(stack, mask, 5, channels=[2, 0])

    assert out.shape == (4, 5, 2)
    assert out[0, 0, 0] == 3.0 and out[0, 0, 1] == 1.0


# ---------------------------------------------------------------------------
# finding the stack a row came from
# ---------------------------------------------------------------------------

def test_a_folder_that_is_not_there_holds_no_stack(tmp_path):
    """A missing merged folder is None, not an OSError from listdir."""
    assert sd._stack_for(str(tmp_path / "nowhere"), "plate1_A01_1") is None


def test_an_exact_name_wins_over_a_prefix(tmp_path):
    """A field written with a trailing token must not shadow the exact file."""
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / "plate1_A01_1.npy").write_bytes(b"")
    (folder / "plate1_A01_1_0.npy").write_bytes(b"")

    found = sd._stack_for(str(folder), "plate1_A01_1")

    assert os.path.basename(found) == "plate1_A01_1.npy"


def test_a_prefix_matches_when_nothing_is_exact(tmp_path):
    """`plate_A01_1_0.npy` has a trailing token the stem does not."""
    folder = tmp_path / "merged"
    folder.mkdir()
    (folder / "plate1_A01_1_0.npy").write_bytes(b"")
    (folder / "notes.txt").write_text("ignored")

    found = sd._stack_for(str(folder), "plate1_A01_1")

    assert os.path.basename(found) == "plate1_A01_1_0.npy"
    assert sd._stack_for(str(folder), "plate9_Z99_9") is None


def test_the_stem_comes_from_the_recorded_source_when_there_is_one():
    """The parser's r1/c1 identifiers would rebuild a stem matching nothing."""
    row = {"source": "npy: plate1_A01_1_0.npy", "plateID": "plate1",
           "rowID": "r1", "columnID": "c1", "fieldID": "1"}

    assert sd._stem_of(row) == "plate1_A01_1_0"


def test_without_a_source_the_stem_is_rebuilt_from_the_identifiers():
    """An object table has no file name, so the coordinates make one."""
    row = {"source": "object table", "plateID": "plate1", "rowID": "A01",
           "columnID": "", "fieldID": "1"}

    assert sd._stem_of(row) == "plate1_A01_1"


# ---------------------------------------------------------------------------
# the streaming pass
# ---------------------------------------------------------------------------

def test_every_selected_object_is_written_once(selection, merged, tmp_path):
    """Two fields of three objects is six crops and no trouble."""
    written = []

    report = sd.stream(selection, merged, str(tmp_path / "out"),
                       write=_collect(written))

    assert report["written"] == 6
    assert report["fields"] == 2
    assert report["missing"] == 0
    assert report["trouble"] == []
    assert len(report["folders"]) == len(set(selection["split"]))
    assert all(shape[2] == 3 for _, shape in written)


def test_a_field_with_no_merged_stack_is_counted_and_named(selection, merged,
                                                           tmp_path):
    """A dataset short by a field says which field, not nothing."""
    os.remove(os.path.join(merged, "plate1_A01_2_0.npy"))
    written = []

    report = sd.stream(selection, merged, str(tmp_path / "out"),
                       write=_collect(written))

    assert report["written"] == 3
    assert report["missing"] == 3
    assert report["trouble"] == ["no merged stack for plate1_A01_2_0"]


def test_a_stack_that_will_not_load_is_counted_with_its_error(selection,
                                                              merged,
                                                              tmp_path):
    """The exception type is recorded so the cause is findable."""
    path = os.path.join(merged, "plate1_A01_2_0.npy")
    with open(path, "wb") as handle:
        handle.write(b"this is not a numpy array")
    written = []

    report = sd.stream(selection, merged, str(tmp_path / "out"),
                       write=_collect(written))

    assert report["written"] == 3
    assert report["missing"] == 3
    assert report["fields"] == 1
    assert len(report["trouble"]) == 1
    assert report["trouble"][0].startswith("plate1_A01_2_0: ")


def test_an_object_id_that_is_not_a_number_is_counted_not_dropped(
        selection, merged, tmp_path):
    """`png_list` spells it ``'o2'``; a row that cannot be parsed still counts."""
    table = selection.copy()
    table.loc[table.index[0], "objectID"] = "not an object"
    written = []

    report = sd.stream(table, merged, str(tmp_path / "out"),
                       write=_collect(written))

    assert report["written"] == 5
    assert report["missing"] == 1


def test_a_label_absent_from_the_mask_is_counted_not_dropped(selection,
                                                             merged,
                                                             tmp_path):
    """A selection naming an object the field does not hold is reported."""
    table = selection.copy()
    table.loc[table.index[0], "objectID"] = "99"
    written = []

    report = sd.stream(table, merged, str(tmp_path / "out"),
                       write=_collect(written))

    assert report["written"] == 5
    assert report["missing"] == 1


def test_the_split_decides_which_folder_a_crop_lands_in(selection, merged,
                                                        tmp_path):
    """A crop goes to the folder its selection row names."""
    written = []

    sd.stream(selection, merged, str(tmp_path / "out"),
              write=_collect(written))

    for path, _shape in written:
        assert os.path.basename(os.path.dirname(path)) in {"train", "test"}


def test_an_empty_selection_writes_nothing_and_says_nothing_went_wrong(
        merged, tmp_path):
    """No rows is an empty report, not a failure."""
    empty = pd.DataFrame(columns=list(sd.SELECTION_COLUMNS))

    report = sd.stream(empty, merged, str(tmp_path / "out"),
                       write=lambda path, crop: None)

    assert report["written"] == 0
    assert report["missing"] == 0
    assert report["trouble"] == [
        "the selection table is empty, so there is nothing to stream"]


# ---------------------------------------------------------------------------
# the whole pass
# ---------------------------------------------------------------------------

def test_the_whole_pass_records_where_the_decision_was_written(merged,
                                                              tmp_path):
    """The report names the selection table so a run can be reproduced."""
    dst = tmp_path / "dataset"

    report = sd.stream_dataset({"merged_folder": merged, "object_array": "cell",
                                "test_split": 0.5, "random_seed": 3}, str(dst))

    assert report["selection"] == str(dst / sd.SELECTION_FILE)
    assert os.path.exists(report["selection"])
    assert report["written"] == 6
    table = pd.read_csv(report["selection"])
    assert set(table["split"]) == {"train", "test"}


def test_the_merged_folder_defaults_to_one_under_the_source(tmp_path):
    """``src`` alone is enough: merged lives beside it by convention."""
    src = tmp_path / "screen"
    (src / "merged").mkdir(parents=True)
    stack = np.zeros((8, 8, 3), dtype=np.int32)
    stack[1:4, 1:4, 2] = 1
    np.save(src / "merged" / "plate1_A01_1_0.npy", stack)

    report = sd.stream_dataset({"src": str(src)}, str(tmp_path / "out"))

    assert report["written"] == 1


def test_the_two_selection_methods_read_different_settings():
    """Each method declares what it needs, and an unknown one raises."""
    assert sd.settings_for_method("column") == ("object_array",
                                                "channel_arrays")
    assert "bounding_box" in sd.settings_for_method("ARRAY")
    with pytest.raises(KeyError):
        sd.settings_for_method("telepathy")


def test_a_table_naming_no_object_refuses_with_both_column_names():
    """The message names the column it wanted and the fallback it wanted."""
    frame = pd.DataFrame({"plateID": ["p1"], "note": ["nothing usable"]})

    with pytest.raises(ValueError) as caught:
        sd.selection_from_objects(frame, object_array="pathogen")

    assert "'pathogen_id'" in str(caught.value)
    assert "'objectID'" in str(caught.value)


def test_a_folder_with_no_arrays_refuses_with_the_folder_named(tmp_path):
    """A path that holds no ``.npy`` is an error naming the path."""
    (tmp_path / "merged").mkdir()

    with pytest.raises(FileNotFoundError) as caught:
        sd.selection_from_arrays(str(tmp_path / "merged"))

    assert str(tmp_path / "merged") in str(caught.value)
    assert "nothing to stream" in str(caught.value)


def test_an_object_table_is_preferred_over_scanning_the_arrays(merged,
                                                               tmp_path):
    """When the measured table exists, the masks are not re-scanned."""
    objects = pd.DataFrame({
        "plateID": ["p1"] * 4, "rowID": ["r1"] * 4, "columnID": ["c1"] * 4,
        "fieldID": ["1"] * 4, "cell_id": [1, 2, 3, 4],
    })

    table, path = sd.build_selection(str(tmp_path / "out"), objects=objects,
                                     merged_folder=merged)

    assert len(table) == 4
    assert set(table["source"]) == {"object table"}
    assert os.path.exists(path)
