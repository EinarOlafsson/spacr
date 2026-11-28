"""Item 1057: the ph_measure parent writes the well table from its field parts in a bounded buffer."""
import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from spacr import tabular


def _fields(folder, count, seed=0, odd=False):
    """Write ``count`` field parts of varying size; return their paths in site order."""
    rng = np.random.default_rng(seed)
    paths = []
    for site in range(count):
        n = int(rng.integers(0, 40))
        frame = pd.DataFrame({"site": np.full(n, site, dtype="int64"),
                              "label": np.arange(n, dtype="int64"),
                              "area": rng.random(n),
                              "name": [f"c{site}_{i}" for i in range(n)]})
        if odd and site == 3:
            frame["extra"] = 1.5
        path = os.path.join(folder, f"site-{site}.parquet")
        frame.to_parquet(path, index=False)
        paths.append(path)
    return paths


def _old(paths, plate, well):
    """The concatenation the parent made before item 1057."""
    frames = [f for f in (pd.read_parquet(p) for p in paths) if len(f)]
    table = pd.concat(frames, ignore_index=True)
    table.insert(0, "well", well)
    table.insert(0, "plate", plate)
    return table


def _hash(frame):
    """A content hash of the rows, columns and dtypes."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()
                          + str(list(frame.columns)).encode()
                          + str(list(frame.dtypes)).encode()).hexdigest()


@pytest.mark.parametrize("odd", [False, True])
def test_the_streamed_table_equals_the_concatenation(tmp_path, odd):
    paths = _fields(str(tmp_path), 25, odd=odd)
    out = str(tmp_path / "well.parquet")
    report = tabular.write_table_from_parts(paths, out, leading={"plate": "P", "well": "A1"},
                                            buffer_bytes=2000)
    expected = _old(paths, "P", "A1")
    got = pd.read_parquet(out)
    assert _hash(got) == _hash(expected)
    pd.testing.assert_frame_equal(got, expected)
    assert report["rows"] == len(expected) and report["columns"] == list(expected.columns)


def test_the_buffer_holds_no_more_than_its_bound(tmp_path):
    paths = _fields(str(tmp_path), 30)
    sizes = [os.path.getsize(p) for p in paths]
    out = str(tmp_path / "well.parquet")
    small = tabular.write_table_from_parts(paths, out, buffer_bytes=1)
    assert small["peak_parts"] == 1 and small["groups"] > 5
    large = tabular.write_table_from_parts(paths, out, buffer_bytes=10 ** 9)
    assert large["peak_parts"] > small["peak_parts"] and large["groups"] == 1
    assert sizes


def test_the_order_is_the_order_of_the_parts_whatever_the_workers_finish_in(tmp_path):
    paths = _fields(str(tmp_path), 20)
    for p in reversed(paths):
        os.utime(p, None)
    out = str(tmp_path / "well.parquet")
    tabular.write_table_from_parts(paths, out, buffer_bytes=3000)
    sites = pd.read_parquet(out)["site"].drop_duplicates().tolist()
    assert sites == sorted(sites)


def test_no_rows_gives_the_empty_well_table(tmp_path):
    folder = str(tmp_path)
    path = os.path.join(folder, "s.parquet")
    pd.DataFrame({"site": [], "label": []}).to_parquet(path, index=False)
    out = str(tmp_path / "well.parquet")
    report = tabular.write_table_from_parts([path], out, leading={"plate": "P", "well": "A1"})
    assert report["rows"] == 0 and report["columns"] == ["plate", "well", "site", "label"]


def test_fields_whose_integer_columns_are_float_in_others_join_as_pandas_does(tmp_path):
    paths = []
    for site in range(12):
        n = 30 + site
        values = np.arange(n, dtype="int64") if site % 3 else np.arange(n, dtype="float64") + 0.5
        path = os.path.join(str(tmp_path), f"s{site}.parquet")
        pd.DataFrame({"site": np.full(n, site), "count": values}).to_parquet(path, index=False)
        paths.append(path)
    out = str(tmp_path / "well.parquet")
    report = tabular.write_table_from_parts(paths, out, leading={"plate": "P", "well": "A1"},
                                            buffer_bytes=1)
    assert report["groups"] == 12
    pd.testing.assert_frame_equal(pd.read_parquet(out), _old(paths, "P", "A1"))
