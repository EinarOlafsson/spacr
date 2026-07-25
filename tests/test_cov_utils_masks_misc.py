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
import subprocess

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

def test_convert_and_relabel_masks_skips_non_int64_files(tmp_path, capsys):
    """A non-int64 npy is left byte-for-byte alone (the `continue` branch)."""
    from spacr.utils import convert_and_relabel_masks

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:6, 2:6] = 7
    np.save(str(tmp_path / "u8.npy"), mask)
    # A non-npy file must not even be looked at.
    (tmp_path / "notes.txt").write_text("ignore me")

    convert_and_relabel_masks(str(tmp_path))

    out = np.load(str(tmp_path / "u8.npy"))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, mask)          # untouched, not relabeled
    assert "Skipping u8.npy" in capsys.readouterr().out


def test_convert_and_relabel_masks_aborts_when_relabeling_exceeds_uint16(tmp_path, capsys):
    """>65535 objects after relabeling -> error printed and the file untouched."""
    from spacr.utils import convert_and_relabel_masks

    # 256 x 256 isolated single pixels = 65536 connected components, which is
    # one more than uint16 can hold.
    mask = np.zeros((512, 512), dtype=np.int64)
    mask[::2, ::2] = 1
    np.save(str(tmp_path / "huge.npy"), mask)

    convert_and_relabel_masks(str(tmp_path))

    out = np.load(str(tmp_path / "huge.npy"))
    assert out.dtype == np.int64, "file must be left as-is when relabeling fails"
    np.testing.assert_array_equal(out, mask)
    err = capsys.readouterr().out
    assert "Error: Relabeling failed for huge.npy" in err
    assert "Converted huge.npy" not in err


def test_convert_and_relabel_masks_converts_int64_to_uint16(tmp_path):
    """Happy path: int64 -> uint16 with the object count preserved."""
    from spacr.utils import convert_and_relabel_masks

    mask = np.zeros((32, 32), dtype=np.int64)
    mask[2:8, 2:8] = 70000        # label value outside the uint16 range
    mask[20:26, 20:26] = 90000
    np.save(str(tmp_path / "m.npy"), mask)

    convert_and_relabel_masks(str(tmp_path))

    out = np.load(str(tmp_path / "m.npy"))
    assert out.dtype == np.uint16
    assert sorted(np.unique(out).tolist()) == [0, 1, 2]
    # geometry preserved, only the label values were remapped
    np.testing.assert_array_equal(out != 0, mask != 0)


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

def test_get_cuda_version_parses_release_and_strips_dots(monkeypatch):
    from spacr.utils import get_cuda_version

    fake = (b"nvcc: NVIDIA (R) Cuda compiler driver\n"
            b"Cuda compilation tools, release 11.8, V11.8.89\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: fake)
    assert get_cuda_version() == "118"


def test_get_cuda_version_returns_none_without_release_token(monkeypatch):
    from spacr.utils import get_cuda_version

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"garbage output\n")
    assert get_cuda_version() is None


def test_get_cuda_version_returns_none_when_nvcc_missing(monkeypatch):
    from spacr.utils import get_cuda_version

    def _boom(*a, **k):
        raise FileNotFoundError("nvcc")

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert get_cuda_version() is None


def test_get_cuda_version_returns_none_when_nvcc_fails(monkeypatch):
    from spacr.utils import get_cuda_version

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(1, ["nvcc", "--version"])

    monkeypatch.setattr(subprocess, "check_output", _boom)
    assert get_cuda_version() is None


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
    monkeypatch.setattr(U.time, "sleep", lambda s: slept.append(s))
    return {"local_dir": pkg / "resources" / "models", "slept": slept}


def test_download_models_creates_dir_and_writes_every_file(fake_models_pkg, monkeypatch):
    import spacr.utils as U

    urls = []

    def _get(url, stream=False):
        urls.append(url)
        return _FakeResponse(b"AB" * 6000)

    monkeypatch.setattr(U, "list_repo_files",
                        lambda repo_id, repo_type=None: ["nested/cyto.pth", "nuc.pth"])
    monkeypatch.setattr(U.requests, "get", _get)

    local_dir = U.download_models(repo_id="me/models", retries=3, delay=0)

    assert local_dir == str(fake_models_pkg["local_dir"])
    assert sorted(os.listdir(local_dir)) == ["cyto.pth", "nuc.pth"]
    # the payload is streamed through in 8192-byte chunks, nothing lost
    assert (fake_models_pkg["local_dir"] / "cyto.pth").read_bytes() == b"AB" * 6000
    assert urls == [
        "https://huggingface.co/datasets/me/models/resolve/main/nested/cyto.pth?download=true",
        "https://huggingface.co/datasets/me/models/resolve/main/nuc.pth?download=true",
    ]
    assert fake_models_pkg["slept"] == []


def test_download_models_returns_early_when_models_present(fake_models_pkg, monkeypatch):
    import spacr.utils as U

    local_dir = fake_models_pkg["local_dir"]
    local_dir.mkdir(parents=True)
    (local_dir / "already.pth").write_bytes(b"x")

    def _never(*a, **k):
        raise AssertionError("must not hit the network when models exist")

    monkeypatch.setattr(U, "list_repo_files", _never)
    monkeypatch.setattr(U.requests, "get", _never)

    assert U.download_models(repo_id="me/models", retries=2, delay=0) == str(local_dir)
    assert os.listdir(local_dir) == ["already.pth"]


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
    assert out is df
    assert list(out.columns) == ["plateID", "rowID", "columnID"]


# ---------------------------------------------------------------------------
# control_filelist
# ---------------------------------------------------------------------------

def test_control_filelist_default_column_values(tmp_path):
    from spacr.utils import control_filelist

    names = ["plate1_A01_f1.tif", "plate1_B02_f1.tif", "plate1_C03_f1.tif"]
    for n in names:
        (tmp_path / n).write_text("")

    got = control_filelist(str(tmp_path))
    assert sorted(got) == ["plate1_A01_f1.tif", "plate1_B02_f1.tif"]


def test_control_filelist_rowid_mode(tmp_path):
    from spacr.utils import control_filelist

    for n in ("plate1_A01_f1.tif", "plate1_B02_f1.tif", "plate1_C03_f1.tif"):
        (tmp_path / n).write_text("")

    got = control_filelist(str(tmp_path), mode="rowID", values=["A", "C"])
    assert sorted(got) == ["plate1_A01_f1.tif", "plate1_C03_f1.tif"]


def test_control_filelist_column_values_argument(tmp_path):
    from spacr.utils import control_filelist

    for n in ("plate1_A01_f1.tif", "plate1_B02_f1.tif", "plate1_C03_f1.tif"):
        (tmp_path / n).write_text("")

    assert control_filelist(str(tmp_path), mode="columnID", values=["03"]) == \
        ["plate1_C03_f1.tif"]
    assert control_filelist(str(tmp_path), mode="columnID", values=["99"]) == []


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

def test_group_feature_class_default_compartments_and_multi_match():
    from spacr.utils import group_feature_class

    df = pd.DataFrame({
        "feature": ["cell_area",
                    "nucleus_cell_overlap",
                    "pathogen_channel_1_mean",
                    "plate_wide_measure"],
        "importance": [1.0, 2.0, 3.0, 4.0],
    })
    out = group_feature_class(df)

    assert out is df                             # mutates and returns the input
    assert out["compartment"].tolist() == [
        "cell", "cell-nucleus", "pathogen", None,
    ]


def test_group_feature_class_channel_name_fills_morphology():
    from spacr.utils import group_feature_class

    df = pd.DataFrame({
        "feature": ["cell_channel_0_mean", "cell_area", "nucleus_channel_1_mean"],
        "importance": [0.5, 0.25, 0.25],
    })
    out = group_feature_class(df, feature_groups=["channel_0", "channel_1"],
                              name="channel")

    assert out["channel"].tolist() == ["channel_0", "morphology", "channel_1"]
    assert out["channel"].isna().sum() == 0


def test_group_feature_class_custom_group_column_name():
    from spacr.utils import group_feature_class

    df = pd.DataFrame({
        "feature": ["cell_area", "blob_count"],
        "importance": [2.0, 1.0],
    })
    out = group_feature_class(df, feature_groups=["cell"], name="grp")

    assert "grp" in out.columns
    assert out["grp"].tolist() == ["cell", None]
    # unmatched features are left as NaN/None outside the 'channel' special case
    assert out["grp"].isna().sum() == 1
