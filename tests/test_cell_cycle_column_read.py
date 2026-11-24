"""The cell-cycle phase reads only the columns it needs, in chunks (spec G2)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from spacr import ops_cell_cycle as cc
from spacr import ops_engine, tabular
from tests.test_ops_cell_cycle_990 import PLATE, _base, _cells


def test_column_read_equals_a_full_read_and_streams_chunks(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.standard_normal((5000, 40)), columns=[f"c{i}" for i in range(40)])
    frame["tag"] = "x"
    path = str(tmp_path / "t.parquet")
    frame.to_parquet(path, row_group_size=1000, index=False)
    seen = []
    original = pq.ParquetFile.iter_batches

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("columns"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    got = tabular.read_table_columns(path, ["c3", "tag", "c1", "absent"], chunk_rows=700)
    pd.testing.assert_frame_equal(got, tabular.read_table(path, canonicalise=False,
                                                          report=None)[["c3", "tag", "c1"]])
    assert seen == [["c3", "tag", "c1"]]


def test_phase_output_is_unchanged_by_columns_it_does_not_need(tmp_path, monkeypatch):
    base = _base(tmp_path)
    asked = []
    original = tabular.read_table_columns

    def spy(source, columns, **kwargs):
        asked.append(list(columns))
        return original(source, columns, chunk_rows=300, **kwargs)

    monkeypatch.setattr(tabular, "read_table_columns", spy)
    out = {}
    for name, junk in (("lean", 0), ("wide", 200)):
        destination = tmp_path / name
        destination.mkdir()
        for well, rows in _cells().groupby("well"):
            rows = rows.drop(columns=["well", "stage", "true_phase"])
            if junk:
                rows = pd.concat([rows, pd.DataFrame(
                    np.ones((len(rows), junk)), index=rows.index,
                    columns=[f"cell_junk_{i}" for i in range(junk)])], axis=1)
            rows.to_parquet(ops_engine._ph_cells_path(str(destination), well), index=False,
                            row_group_size=500)
        settings = {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                    "nucleus_channel": "dapi"}
        db = str(destination / "measurements.db")
        cc.run_cell_cycle_phase(db, PLATE, ["A1", "A2"], settings, str(destination),
                                permutations=20)
        from spacr.ops_store import read_table
        out[name] = read_table(db, "ops_cell_cycle")
    pd.testing.assert_frame_equal(out["lean"], out["wide"])
    assert asked and not any(c.startswith("cell_junk") for cols in asked for c in cols)
