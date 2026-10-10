"""Edge paths of the F576 Parquet store and the ML measurement-store loop."""
import json
import os
import sys
import types

import pandas as pd
import pytest

from spacr import tabular

pytest.importorskip("pyarrow")


def _store(tmp_path):
    store = str(tmp_path / "m.parquetdb")
    tabular._store_write(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
                         store, "parquet", "cell", "replace")
    return store


@pytest.mark.parametrize("name", ["..", "a/b", "a\\b", "."])
def test_invalid_parquet_table_names_are_refused(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid Parquet table name"):
        tabular._parquet_parts(str(tmp_path), name)
    with pytest.raises(ValueError, match="Invalid Parquet table name"):
        tabular._store_write(pd.DataFrame({"a": [1]}),
                             str(tmp_path / "s.parquetdb"), "parquet",
                             name, "append")


def test_windows_lock_uses_msvcrt_on_empty_and_existing_lock_file(
        tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        LK_LOCK=1, LK_UNLCK=0,
        locking=lambda fd, mode, n: calls.append((mode, n)))
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    store = str(tmp_path / "w.parquetdb")
    lock = os.path.join(store, ".spacr-write.lock")
    for _ in range(2):
        monkeypatch.setattr(os, "name", "nt")
        try:
            with tabular._parquet_store_lock(store):
                pass
        finally:
            monkeypatch.setattr(os, "name", "posix")
    assert calls == [(1, 1), (0, 1), (1, 1), (0, 1)]
    assert os.path.getsize(lock) == 1


def test_part_record_refuses_a_part_that_changes_while_read(
        tmp_path, monkeypatch):
    part = tmp_path / "p.parquet"
    part.write_bytes(b"data")
    real = os.stat
    seen = []

    def stat(path, *a, **k):
        result = real(path, *a, **k)
        if os.fspath(path) == str(part):
            seen.append(1)
            if len(seen) > 1:
                fields = list(result)
                fields[6] += 1
                return os.stat_result(fields)
        return result

    monkeypatch.setattr(os, "stat", stat)
    with pytest.raises(RuntimeError, match="changed while being read"):
        tabular._parquet_part_record(str(part))


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(active=[]),
    lambda m: m.update(retired="nope"),
    lambda m: m.update(retired=["not-a-dict"]),
])
def test_manifest_rejects_malformed_categories(tmp_path, mutate):
    store = _store(tmp_path)
    folder = os.path.join(store, "cell")
    path = os.path.join(folder, ".spacr-parts.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    mutate(manifest)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    with pytest.raises(ValueError, match="Invalid Parquet part manifest"):
        tabular._parquet_manifest(folder)


def test_limit_zero_read_returns_empty_frame_with_schema(tmp_path):
    store = _store(tmp_path)
    chunks = list(tabular._iter_chunks(store, "cell", limit=0))
    assert len(chunks) == 1
    assert chunks[0].empty
    assert list(chunks[0].columns) == ["a", "b"]


def test_regression_pairs_keep_only_present_table_names():
    from spacr.ml import normalize_regression_input_pairs

    settings = {"paired_data": [
        {"score": "s.duckdb", "score_table": "scores", "count": "c.csv",
         "count_table": None}]}
    pairs, migrated = normalize_regression_input_pairs(settings)
    assert not migrated
    assert pairs[0]["score_table"] == "scores"
    assert "count_table" not in pairs[0]


def test_generate_ml_scores_reads_a_shared_store_once(tmp_path, monkeypatch):
    from spacr import io, ml, utils

    read = []

    def fake_read(db_loc, tables, verbose, **kwargs):
        read.append(list(db_loc))
        return pd.DataFrame({"x": [1]}), None

    class Stop(BaseException):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(io, "_read_and_merge_data", fake_read)
    monkeypatch.setattr(utils, "_measurement_store_for",
                        lambda path, settings: "postgresql://shared")
    monkeypatch.setattr(utils, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(utils, "calculate_shortest_distance", stop)
    srcs = [str(tmp_path / "p1"), str(tmp_path / "p2")]
    with pytest.raises(Stop):
        ml.generate_ml_scores({"src": srcs, "measurement_backend": "postgres"})
    assert read == [["postgresql://shared"]]
