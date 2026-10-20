"""A file handed to several pair rows is parsed once, not once per row.

Reported 2026-08-19: "it looks like it is not running while it does say that
it is running. it is in any cas taking longer than it should".

It WAS running. Measured on the live process: the regression worker at 82%
CPU, `write_bytes` sitting at exactly 2,752,598,016 -- the merged frame it had
just written -- and `read_bytes` static at 1.4 MB, because it was reading that
2.75 GB file back out of the PAGE CACHE. Four times: the Measurements tab
points every pair row's score at the one merged frame, and this parsed it once
per row.
"""
import os
import tempfile

import pandas as pd
import pytest

from spacr import tabular
from spacr.ml import load_regression_input_pairs, normalize_regression_input_pairs


@pytest.fixture()
def one_score_four_counts(tmp_path):
    rows = [{"plateID": f"plate{p}", "rowID": "r1", "columnID": "c1",
             "cell_area": 100.0 + p} for p in (1, 2, 3, 4)]
    score = tmp_path / "merged_measurements.csv"
    pd.DataFrame(rows).to_csv(score, index=False)
    pairs = []
    for i in (1, 2, 3, 4):
        count = tmp_path / f"count{i}.csv"
        pd.DataFrame([{"rowID": "r1", "columnID": "c1", "grna": "g1",
                       "count": 7}]).to_csv(count, index=False)
        pairs.append({"score": str(score), "count": str(count)})
    return pairs


def test_the_shared_score_file_is_read_once(one_score_four_counts, monkeypatch):
    seen = []
    real = tabular.read_table

    def counted(path, *args, **kwargs):
        seen.append(os.path.basename(str(path)))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(tabular, "read_table", counted)

    load_regression_input_pairs(one_score_four_counts)

    merged = [name for name in seen if name == "merged_measurements.csv"]
    assert len(merged) == 1, f"parsed the 2.75 GB file {len(merged)} times"


def test_every_distinct_file_is_still_read(one_score_four_counts, monkeypatch):
    seen = []
    real = tabular.read_table
    monkeypatch.setattr(
        tabular, "read_table",
        lambda p, *a, **k: (seen.append(os.path.basename(str(p))),
                            real(p, *a, **k))[1])

    load_regression_input_pairs(one_score_four_counts)

    assert sorted(set(seen)) == ["count1.csv", "count2.csv", "count3.csv",
                                 "count4.csv", "merged_measurements.csv"]


def test_each_pair_gets_its_own_copy_to_mutate(one_score_four_counts):
    """The caller assigns plateID onto what it gets and filters it down to one
    plate. Handing out the cached frame itself would let the first pair row's
    edits reach the second."""
    counts, scores, audit = load_regression_input_pairs(one_score_four_counts)

    assert [row["plate"] for row in audit] == ["plate1", "plate2", "plate3",
                                               "plate4"]
    # Four rows, one per plate -- not the same plate four times, and not one
    # plate's rows repeated.
    assert sorted(scores["plateID"].unique()) == ["plate1", "plate2",
                                                  "plate3", "plate4"]
    assert len(scores) == 4


@pytest.mark.parametrize("backend", ["sqlite", "parquet", "duckdb"])
def test_regression_reads_and_caches_each_named_store_table(tmp_path, monkeypatch,
                                                          backend):
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    if backend == "parquet":
        pytest.importorskip("pyarrow")
    suffix = {"sqlite": ".db", "parquet": ".parquetdb", "duckdb": ".duckdb"}[backend]
    store = str(tmp_path / ("inputs" + suffix))
    metadata = {"plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"]}
    first = pd.DataFrame(metadata).assign(pred=0.15)
    second = pd.DataFrame(metadata).assign(pred=0.85)
    count = pd.DataFrame(metadata).assign(grna="g1", count=7)
    for name, frame in (("score_first", first), ("score_second", second),
                        ("counts", count)):
        tabular.write_database(frame, store, name, if_exists="replace")
    settings = {"paired_data": [
        {"score": store, "count": store, "score_table": score,
         "count_table": "counts"}
        for score in ("score_first", "score_second", "score_first")
    ]}
    pairs, migrated = normalize_regression_input_pairs(settings)
    assert migrated is False
    assert [row["score_table"] for row in pairs] == [
        "score_first", "score_second", "score_first"]
    calls = []
    original = tabular.read_table

    def counted(path, **options):
        calls.append((path, options.get("table")))
        return original(path, **options)

    monkeypatch.setattr(tabular, "read_table", counted)
    counts, scores, audit = load_regression_input_pairs(pairs)
    assert calls == [(store, "score_first"), (store, "counts"),
                     (store, "score_second")]
    assert scores["pred"].tolist() == [0.15, 0.85]
    assert counts["count"].tolist() == [7]
    assert [row["plate"] for row in audit] == ["plate1"] * 3


def test_named_database_input_rejects_an_invalid_table_before_caching(tmp_path):
    with pytest.raises(ValueError, match="Invalid table name"):
        load_regression_input_pairs([
            {"score": str(tmp_path / "inputs.duckdb"),
             "score_table": ["invalid"], "count": "counts.csv"}])


def test_postgres_named_inputs_keep_the_literal_locator(monkeypatch):
    locator = "postgresql://user:$literal@host/database"
    metadata = {"plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"]}
    frames = {"scores": pd.DataFrame(metadata).assign(pred=0.5),
              "counts": pd.DataFrame(metadata).assign(grna="g1", count=7)}
    calls = []

    def read(path, table=None):
        calls.append((path, table))
        return frames[table].copy()

    def reject_path_normalization(path):
        raise AssertionError("A PostgreSQL locator is not a filesystem path")

    monkeypatch.setenv("literal", "expanded_password")
    monkeypatch.setattr(tabular, "read_table", read)
    monkeypatch.setattr("spacr.frame_handoff.key_for", reject_path_normalization)
    counts, scores, _ = load_regression_input_pairs([
        {"score": locator, "score_table": "scores",
         "count": locator, "count_table": "counts"}])
    assert calls == [(locator, "scores"), (locator, "counts")]
    assert counts["count"].tolist() == [7]
    assert scores["pred"].tolist() == [0.5]


def test_postgres_regression_named_tables_are_read_and_cached(monkeypatch):
    locator = os.environ.get("SPACR_TEST_POSTGRES_DSN")
    if not locator:
        pytest.skip("SPACR_TEST_POSTGRES_DSN is not set")
    pytest.importorskip("psycopg")
    import uuid

    token = uuid.uuid4().hex
    first_name = "regression_first_" + token
    second_name = "regression_second_" + token
    count_name = "regression_counts_" + token
    metadata = {"plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"]}
    frames = [(first_name, pd.DataFrame(metadata).assign(pred=0.15)),
              (second_name, pd.DataFrame(metadata).assign(pred=0.85)),
              (count_name, pd.DataFrame(metadata).assign(grna="g1", count=7))]
    try:
        for name, frame in frames:
            tabular.write_database(frame, locator, name, if_exists="replace")
        pairs, migrated = normalize_regression_input_pairs({"paired_data": [
            {"score": locator, "score_table": name,
             "count": locator, "count_table": count_name}
            for name in (first_name, second_name, first_name)]})
        assert migrated is False
        calls = []
        original = tabular.read_table

        def counted(path, **options):
            calls.append((path, options.get("table")))
            return original(path, **options)

        monkeypatch.setattr(tabular, "read_table", counted)
        counts, scores, audit = load_regression_input_pairs(pairs)
        assert calls == [(locator, first_name), (locator, count_name),
                         (locator, second_name)]
        assert scores["pred"].tolist() == [0.15, 0.85]
        assert counts["count"].tolist() == [7]
        assert [row["plate"] for row in audit] == ["plate1"] * 3
    finally:
        with tabular._postgres_connect(locator) as conn:
            for name, _frame in frames:
                conn.execute("DROP TABLE IF EXISTS " + tabular._quote_identifier(name))
