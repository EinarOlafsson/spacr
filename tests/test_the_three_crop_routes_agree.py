"""The same objects, cut three ways, must be the same images.

spaCR can produce a set of single-object crops by three routes:

* `measure_crop`, during a Measure run;
* streaming from the ARRAY, reading the object masks in the merged stacks;
* streaming from the DATABASE, reading the coordinate columns.

If they disagree, an annotation made against one set does not describe the
images a model is then trained on, and the label noise that introduces is
invisible: nothing errors, the numbers just get quietly worse.

WHAT IS AND IS NOT COMPARABLE. The database stores coordinates, so its route
can only ever produce a BOUNDING BOX. The array route has the mask and can cut
to the object. The two are therefore comparable only with
``bounding_box=True``, and this file pins that rather than leaving it to be
rediscovered.

Instruction 338. The routes share `measure._save_object_crop`, so identical
bytes are a property of the code rather than a coincidence; that sharing is
what these tests defend.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd
import pytest

from spacr.annotation_dataset import generate_annotation_dataset


@pytest.fixture
def plate(tmp_path):
    """A one-field plate with two objects, reachable by both routes."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (tmp_path / "measurements").mkdir()

    stack = np.zeros((48, 48, 4), dtype=np.uint16)
    stack[6:16, 6:16, 0] = 900          # object 1, bright in channel 0
    stack[6:16, 6:16, 3] = 1
    stack[28:40, 28:40, 1] = 700        # object 2, bright in channel 1
    stack[28:40, 28:40, 3] = 2
    np.save(merged / "plate1_A01_1_1.npy", stack)

    # The coordinate columns a Measure run would have written.
    database = tmp_path / "measurements" / "measurements.db"
    connection = sqlite3.connect(database)
    pd.DataFrame({
        "plateID": ["plate1", "plate1"],
        "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"],
        "fieldID": ["1", "1"],
        "objectID": [1, 2],
    }).to_sql("cell", connection, index=False)
    connection.commit()
    connection.close()
    return tmp_path


def _run(plate, source, dst):
    return generate_annotation_dataset({
        "src": str(plate), "stream_source": source, "object_array": "cell",
        "channel_arrays": [0, 1, 2], "bounding_box": True,
        "dst": str(dst),
    })


def _crops(folder):
    """Every written crop, keyed by file name, as bytes."""
    out = {}
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.endswith(".png"):
                out[name] = open(os.path.join(root, name), "rb").read()
    return out


def test_the_array_route_finds_both_objects(plate, tmp_path):
    report = _run(plate, "array", tmp_path / "a")
    assert report["written"] == 2, report.get("trouble")


def test_the_database_route_finds_both_objects(plate, tmp_path):
    report = _run(plate, "database", tmp_path / "d")
    assert report["written"] == 2, report.get("trouble")


def test_the_two_streaming_routes_produce_the_same_images(plate, tmp_path):
    """THE CLAIM INSTRUCTION 338 IS ABOUT.

    Compared as BYTES, not by filename: identical names with different
    contents is the failure this exists for, and it is the one that would be
    invisible everywhere else.
    """
    _run(plate, "array", tmp_path / "a")
    _run(plate, "database", tmp_path / "d")

    from_array = _crops(tmp_path / "a")
    from_database = _crops(tmp_path / "d")

    assert set(from_array) == set(from_database), (
        "the two routes selected different objects")
    differing = [n for n in from_array if from_array[n] != from_database[n]]
    assert not differing, (
        f"same objects, different pixels, from {differing}")


def test_each_route_registers_its_own_table(plate, tmp_path):
    """Two sets in one database must not collide -- the second would otherwise
    overwrite annotations already made against the first."""
    first = _run(plate, "array", tmp_path / "a")["table"]
    second = _run(plate, "database", tmp_path / "d")["table"]
    assert (first, second) == ("png_list", "png_list_2")


def test_the_registered_paths_point_at_files_that_exist(plate, tmp_path):
    """A table naming a picture that is not there is worse than no table."""
    report = _run(plate, "array", tmp_path / "a")
    connection = sqlite3.connect(plate / "measurements" / "measurements.db")
    paths = [r[0] for r in connection.execute(
        f'select png_path from "{report["table"]}"')]
    connection.close()
    assert paths
    assert all(os.path.isfile(p) for p in paths), paths


def test_filtration_reaches_both_routes(plate, tmp_path):
    """"the same filtration criteria" is half of the parity claim, so a filter
    that only one route honoured would break it silently."""
    settings = {"src": str(plate), "object_array": "cell",
                "channel_arrays": [0, 1, 2], "bounding_box": True,
                "max_objects": 1}
    from_array = generate_annotation_dataset(
        {**settings, "stream_source": "array", "dst": str(tmp_path / "a")})
    from_database = generate_annotation_dataset(
        {**settings, "stream_source": "database", "dst": str(tmp_path / "d")})
    assert from_array["written"] == from_database["written"] == 1


def test_a_filter_that_keeps_nothing_says_so(plate, tmp_path):
    report = generate_annotation_dataset({
        "src": str(plate), "stream_source": "array", "object_array": "cell",
        "dst": str(tmp_path / "a"), "max_objects": 0, "wells": ["nowhere"]})
    assert report["written"] == 0
    assert report["table"] == ""
    assert "filtered out" in " ".join(report["trouble"])


def test_a_database_with_no_such_object_says_so(plate, tmp_path):
    report = generate_annotation_dataset({
        "src": str(plate), "stream_source": "database",
        "object_array": "pathogen", "dst": str(tmp_path / "d")})
    assert report["written"] == 0
    assert "no pathogen rows" in " ".join(report["trouble"])


def test_the_database_route_is_always_a_bounding_box(plate, tmp_path):
    """It has coordinates and no mask, so asking it to cut to the object is a
    request it cannot honour -- and silently returning a box would make the
    parity comparison above meaningless."""
    import inspect

    import spacr.annotation_dataset as module

    source = inspect.getsource(module.generate_annotation_dataset)
    assert 'True if source == "database"' in source


def test_both_routes_use_measures_own_writer():
    """Two writers cannot be relied on to narrow to 8-bit, pad a two-channel
    crop, or resize identically -- and every one of those would break parity
    in a way only a byte comparison would catch."""
    import inspect

    import spacr.annotation_dataset as module

    source = inspect.getsource(module.generate_annotation_dataset)
    assert "from .measure import _save_object_crop" in source
