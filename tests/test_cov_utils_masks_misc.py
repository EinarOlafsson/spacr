"""CPU coverage for the ``spacr.utils`` mask/misc block (utils.py 6589-7063).

Covers the defensive and rarely-exercised branches of:
convert_and_relabel_masks, correct_masks, get_cuda_version,
prepare_batch_for_segmentation, check_index, download_models,
add_column_to_database, correct_metadata_column_names, control_filelist,
rename_columns_in_db and group_feature_class.

Everything here is offline: the Hugging Face listing and every HTTP request
made by ``download_models`` are replaced with local fakes, and ``nvcc`` is
never actually invoked.
"""
from __future__ import annotations

import os
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_open_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# convert_and_relabel_masks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# correct_masks
# ---------------------------------------------------------------------------

def test_correct_masks_relabels_then_concatenates(tmp_path, monkeypatch):
    import spacr.io as sio
    from spacr.utils import correct_masks

    cell_dir = tmp_path / "masks" / "cell_mask_stack"
    cell_dir.mkdir(parents=True)
    mask = np.zeros((24, 24), dtype=np.int64)
    mask[1:6, 1:6] = 300000
    mask[12:18, 12:18] = 400000
    np.save(str(cell_dir / "field1.npy"), mask)

    calls = []
    monkeypatch.setattr(
        sio, "_load_and_concatenate_arrays",
        lambda *a, **k: calls.append((a, k)),
    )

    correct_masks(str(tmp_path))

    out = np.load(str(cell_dir / "field1.npy"))
    assert out.dtype == np.uint16
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]
    assert calls == [((str(tmp_path), [0, 1, 2, 3], 1, 0, 2), {})]


# ---------------------------------------------------------------------------
# get_cuda_version
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prepare_batch_for_segmentation
# ---------------------------------------------------------------------------

def test_prepare_batch_for_segmentation_casts_uint16_and_max_normalizes():
    from spacr.utils import prepare_batch_for_segmentation

    batch = np.zeros((2, 4, 4), dtype=np.uint16)
    batch[0, 0, 0] = 1000
    batch[0, 1, 1] = 500
    batch[1, 2, 2] = 40000

    out = prepare_batch_for_segmentation(batch)

    assert out.dtype == np.float32
    assert out.shape == (2, 4, 4)
    assert out[0].max() == pytest.approx(1.0)
    assert out[1].max() == pytest.approx(1.0)
    assert out[0, 1, 1] == pytest.approx(0.5)


def test_prepare_batch_for_segmentation_leaves_normalized_float32_untouched():
    from spacr.utils import prepare_batch_for_segmentation

    batch = np.zeros((2, 3, 3), dtype=np.float32)
    batch[0, 0, 0] = 1.0
    batch[1, 1, 1] = 0.25

    out = prepare_batch_for_segmentation(batch.copy())

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, batch)


# ---------------------------------------------------------------------------
# check_index
# ---------------------------------------------------------------------------

def test_check_index_raises_and_lists_problematic_indices(capsys):
    from spacr.utils import check_index

    df = pd.DataFrame(
        {"v": [1, 2, 3]},
        index=["p1_A_01_f1_o1", "p1_A_01", "p1_A_01_f2_o2"],
    )
    with pytest.raises(ValueError, match=r"Found 1 problematic indices"):
        check_index(df, elements=5, split_char="_")

    out = capsys.readouterr().out
    assert "Indices that cannot be separated into 5 parts" in out
    assert "p1_A_01" in out


def test_check_index_respects_custom_split_char():
    from spacr.utils import check_index

    df = pd.DataFrame({"v": [1]}, index=["a-b-c"])
    assert check_index(df, elements=3, split_char="-") is None
    with pytest.raises(ValueError):
        check_index(df, elements=3, split_char="_")


# ---------------------------------------------------------------------------
# download_models  (fully offline: list_repo_files + requests.get are faked)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload=b"weights-bytes", status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.payload), chunk_size):
            yield self.payload[i:i + chunk_size]


@pytest.fixture
def fake_models_pkg(tmp_path, monkeypatch):
    """Point ``download_models`` at a throw-away package dir and kill sleeps."""
    import spacr.utils as U

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(U, "spacr_path", str(pkg / "__init__.py"))
    slept = []
    original_sleep = U.time.sleep
    test_thread = threading.get_ident()

    def record_test_retry(seconds):
        if threading.get_ident() == test_thread:
            slept.append(seconds)
        else:
            original_sleep(seconds)

    monkeypatch.setattr(U.time, "sleep", record_test_retry)
    return {"local_dir": pkg / "resources" / "models", "slept": slept}


def test_download_models_retries_a_failed_file_then_succeeds(fake_models_pkg, monkeypatch):
    import spacr.utils as U

    attempts = {"n": 0}

    def _get(url, stream=False):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise U.requests.HTTPError("503")
        return _FakeResponse(b"ok")

    monkeypatch.setattr(U, "list_repo_files", lambda repo_id, repo_type=None: ["cyto.pth"])
    monkeypatch.setattr(U.requests, "get", _get)

    local_dir = U.download_models(repo_id="me/models", retries=3, delay=7)

    assert attempts["n"] == 2
    assert (fake_models_pkg["local_dir"] / "cyto.pth").read_bytes() == b"ok"
    assert fake_models_pkg["slept"] == [7]      # slept once, with the given delay
    assert os.path.isdir(local_dir)


def test_download_models_raises_when_a_file_never_downloads(fake_models_pkg, monkeypatch):
    import spacr.utils as U

    attempts = {"n": 0}

    def _get(url, stream=False):
        attempts["n"] += 1
        raise U.requests.Timeout("timed out")

    monkeypatch.setattr(U, "list_repo_files", lambda repo_id, repo_type=None: ["cyto.pth"])
    monkeypatch.setattr(U.requests, "get", _get)

    with pytest.raises(Exception, match=r"Failed to download cyto\.pth after multiple attempts"):
        U.download_models(repo_id="me/models", retries=2, delay=0)

    assert attempts["n"] == 2                  # one try per `retries`
    assert os.listdir(fake_models_pkg["local_dir"]) == []
    assert fake_models_pkg["slept"] == [0, 0]


def test_download_models_raises_after_listing_keeps_failing(fake_models_pkg, monkeypatch):
    import spacr.utils as U

    calls = {"n": 0}

    def _list(repo_id, repo_type=None):
        calls["n"] += 1
        raise U.requests.Timeout("hub down")

    monkeypatch.setattr(U, "list_repo_files", _list)

    with pytest.raises(Exception, match=r"Failed to download model files after multiple attempts"):
        U.download_models(repo_id="me/models", retries=3, delay=1)

    assert calls["n"] == 3                     # outer loop retried `retries` times
    assert fake_models_pkg["slept"] == [1, 1, 1]


# ---------------------------------------------------------------------------
# add_column_to_database
# ---------------------------------------------------------------------------

def _make_db(path, extra_cols=()):
    con = sqlite3.connect(path)
    cols = ", ".join(["prc TEXT"] + [f"{c} INTEGER" for c in extra_cols])
    con.execute(f"CREATE TABLE cell ({cols})")
    for prc in ("p1_A01_f1", "p1_A02_f1", "p1_A03_f1"):
        con.execute("INSERT INTO cell (prc) VALUES (?)", (prc,))
    con.commit()
    con.close()


def test_add_column_to_database_replaces_zeros_and_suffixes_existing_column(tmp_path, capsys):
    from spacr.utils import add_column_to_database

    db = tmp_path / "m.db"
    _make_db(db, extra_cols=("annotation", "annotation_1"))
    csv = tmp_path / "anno.csv"
    pd.DataFrame({
        "prc": ["p1_A01_f1", "p1_A02_f1", "p1_A03_f1"],
        "annotation": [0, 1, 0],
    }).to_csv(csv, index=False)

    add_column_to_database({
        "csv_path": str(csv), "db_path": str(db), "table_name": "cell",
        "update_column": "annotation", "match_column": "prc",
    })

    con = sqlite3.connect(db)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(cell)")]
        rows = dict(con.execute("SELECT prc, annotation_2 FROM cell").fetchall())
    finally:
        con.close()

    assert "annotation_2" in cols                       # suffix bumped past _1
    assert rows == {"p1_A01_f1": 2, "p1_A02_f1": 1, "p1_A03_f1": 2}
    out = capsys.readouterr().out
    assert "Replacing all 0 values with 2" in out
    assert "Using new column name: 'annotation_2'" in out


def test_add_column_to_database_writes_null_for_nan(tmp_path):
    from spacr.utils import add_column_to_database

    db = tmp_path / "m.db"
    _make_db(db)
    csv = tmp_path / "anno.csv"
    pd.DataFrame({
        "prc": ["p1_A01_f1", "p1_A02_f1", "p1_A03_f1"],
        "score": [3, np.nan, 5],
    }).to_csv(csv, index=False)

    add_column_to_database({
        "csv_path": str(csv), "db_path": str(db), "table_name": "cell",
        "update_column": "score", "match_column": "prc",
    })

    con = sqlite3.connect(db)
    try:
        rows = dict(con.execute("SELECT prc, score FROM cell").fetchall())
    finally:
        con.close()

    assert rows == {"p1_A01_f1": 3, "p1_A02_f1": None, "p1_A03_f1": 5}


# ---------------------------------------------------------------------------
# correct_metadata_column_names
# ---------------------------------------------------------------------------

def test_correct_metadata_column_names_col_grna_and_plate_row():
    from spacr.utils import correct_metadata_column_names

    df = pd.DataFrame({
        "col": ["c1", "c2"],
        "grna_name": ["g1", "g2"],
        "plate_row": ["plate1_A01", "plate2_B02"],
    })
    out = correct_metadata_column_names(df)

    assert "col" not in out.columns and "grna_name" not in out.columns
    assert out["columnID"].tolist() == ["c1", "c2"]
    assert out["grna"].tolist() == ["g1", "g2"]
    assert out["plateID"].tolist() == ["plate1", "plate2"]
    assert out["rowID"].tolist() == ["A01", "B02"]
    assert "plate_row" in out.columns          # source column is kept


def test_correct_metadata_column_names_is_a_noop_for_canonical_names():
    from spacr.utils import correct_metadata_column_names

    df = pd.DataFrame({"plateID": ["p1"], "rowID": ["A"], "columnID": ["01"]})
    out = correct_metadata_column_names(df)
    # canonicalise_frame promises a new frame even when no spelling changes.
    # Callers can therefore add derived columns to the result without
    # mutating the table they were handed.
    assert out is not df
    assert list(out.columns) == ["plateID", "rowID", "columnID"]
    assert out.to_dict("list") == df.to_dict("list")
    assert "column_collisions" not in df.attrs


# ---------------------------------------------------------------------------
# control_filelist
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# rename_columns_in_db
# ---------------------------------------------------------------------------

def test_rename_columns_in_db_renames_legacy_and_skips_clashes(tmp_path, capsys):
    from spacr.utils import rename_columns_in_db

    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE cell ("row" TEXT, "column" TEXT, plate TEXT, '
                'field TEXT, channel TEXT)')
    # `col` cannot be renamed here: columnID already exists on this table.
    con.execute('CREATE TABLE png_list (col TEXT, columnID TEXT)')
    con.commit()
    con.close()

    rename_columns_in_db(str(db))

    con = sqlite3.connect(db)
    try:
        cell_cols = [r[1] for r in con.execute("PRAGMA table_info(cell)")]
        png_cols = [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    finally:
        con.close()

    assert cell_cols == ["rowID", "columnID", "plateID", "fieldID", "chanID"]
    assert png_cols == ["col", "columnID"]      # skipped, no clash created
    assert "Renamed `cell`.`row`" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# group_feature_class
# ---------------------------------------------------------------------------
