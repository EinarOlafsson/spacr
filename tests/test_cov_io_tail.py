"""CPU-only coverage for the tail of :mod:`spacr.io` (``read_plot_model_stats``
onwards).

Everything in this module runs offline on synthetic SQLite databases, tiny
TIFFs and stub reader objects — no GPU, no network, no Cellpose weights.
The focus is the branches the rest of the suite never reaches:

* the checkpoint / progress writers (``_save_model``, ``_save_progress``)
* ``_read_db`` chunk assembly and its identifier guard
* every single-table path through ``_read_and_merge_data``
* the ``generate_dataset`` / ``generate_loaders`` guard rails
* every mode + helper of ``generate_training_dataset``
* the Yokogawa converters, including well overflow onto a second plate and
  the ND2 / CZI / LIF vendor branches (driven through stub readers)
* ``prepare_cellpose_dataset`` augmentation maths

"""
from __future__ import annotations

import os
import sqlite3
import tarfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io as IO  # noqa: E402
# spacr.io pulls .utils / .settings in lazily from inside the functions under
# test; import them at collection time so the heavy import chain is not
# charged to whichever test happens to run first.
import spacr.settings  # noqa: E402,F401
import spacr.utils  # noqa: E402,F401


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _headless_figures(monkeypatch):
    """No blocking windows, no figure leaks."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _png(path, rng, size=8):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _stats_frame(n=4):
    """A per-epoch stats frame with every column read_plot_model_stats plots."""
    return pd.DataFrame({
        "epoch": list(range(1, n + 1)),
        "accuracy": np.linspace(0.50, 0.90, n),
        "neg_accuracy": np.linspace(0.40, 0.80, n),
        "pos_accuracy": np.linspace(0.60, 0.95, n),
        "loss": np.linspace(1.00, 0.20, n),
        "prauc": np.linspace(0.30, 0.80, n),
        "optimal_threshold": np.linspace(0.40, 0.60, n),
    })


def _tiny_model():
    torch = pytest.importorskip("torch")
    return torch.nn.Linear(2, 2)


# ---------------------------------------------------------------------------
# read_plot_model_stats
# ---------------------------------------------------------------------------

def test_read_plot_model_stats_show_branch_draws_six_figures(tmp_path, monkeypatch):
    """save=False routes every metric through plt.show() and writes no PDF."""
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(len(plt.get_fignums())))
    train = tmp_path / "train.csv"
    val = tmp_path / "validation.csv"
    _stats_frame().to_csv(train)
    _stats_frame().to_csv(val)

    IO.read_plot_model_stats(str(train), str(val), save=False)

    # one figure shown per metric: accuracy, neg_accuracy, pos_accuracy,
    # loss, prauc, optimal_threshold
    assert len(shown) == 6
    assert all(n >= 1 for n in shown)
    assert list(tmp_path.glob("*.pdf")) == []


# ---------------------------------------------------------------------------
# _save_model
# ---------------------------------------------------------------------------

def test_save_model_defaults_and_below_every_threshold(tmp_path):
    """intermedeate_save=None/channels=None fill in defaults; a low accuracy
    saves nothing and returns None."""
    model = _tiny_model()
    out = IO._save_model(model, "cnn", {"accuracy": 0.10}, str(tmp_path),
                         epoch=3, epochs=10,
                         intermedeate_save=None, channels=None)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_save_model_uses_val_dict_not_train_dict(tmp_path):
    """A memorising train accuracy must not trigger a checkpoint; the
    validation dict is the signal."""
    model = _tiny_model()
    assert IO._save_model(model, "cnn", {"accuracy": 0.999}, str(tmp_path),
                          epoch=3, epochs=10,
                          val_dict={"accuracy": 0.10}) is None
    assert list(tmp_path.iterdir()) == []

    path = IO._save_model(model, "cnn", {"accuracy": 0.0}, str(tmp_path),
                          epoch=3, epochs=10,
                          val_dict={"accuracy": 0.995})
    assert path is not None and os.path.isfile(path)
    assert os.path.basename(path) == "cnn_epoch_3_acc_99.5000_channels_rgb.pth"


# ---------------------------------------------------------------------------
# _save_progress
# ---------------------------------------------------------------------------

def test_save_progress_appends_to_existing_csv(tmp_path):
    """A second call appends rows (header written once) instead of truncating."""
    df = _stats_frame(3)
    IO._save_progress(str(tmp_path), df, None)
    IO._save_progress(str(tmp_path), df, None)

    back = pd.read_csv(tmp_path / "train.csv", index_col=0)
    assert len(back) == 6
    assert back["epoch"].tolist() == [1, 2, 3, 1, 2, 3]
    # validation_df was None -> no validation.csv, no plotting
    assert not (tmp_path / "validation.csv").exists()


# ---------------------------------------------------------------------------
# _read_db
# ---------------------------------------------------------------------------

def _small_cell_db(path, n=6):
    df = pd.DataFrame({
        "object_label": np.arange(1, n + 1),
        "plateID": ["plate1"] * n,
        "cell_area": np.linspace(100.0, 200.0, n),
    })
    with sqlite3.connect(path) as con:
        df.to_sql("cell", con, index=False, if_exists="replace")
    return str(path)


def test_read_db_rejects_unquotable_table_name(tmp_path):
    """A table that exists but whose name is empty is refused by the
    identifier guard rather than being interpolated into SQL."""
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE "" (png_path TEXT)')
    con.execute('INSERT INTO "" VALUES (\'a.png\')')
    con.commit()
    con.close()

    with pytest.raises(ValueError, match="Invalid table name"):
        IO._read_db(str(db), [""])


def test_read_db_empty_chunk_stream_falls_back_to_limit_zero(tmp_path, monkeypatch):
    """When the chunk iterator yields nothing, the LIMIT 0 fallback still
    returns a frame carrying the table's columns."""
    db = _small_cell_db(tmp_path / "m.db")
    real = pd.read_sql_query

    def fake(sql, con, *args, **kwargs):
        if "chunksize" in kwargs:
            return iter(())
        return real(sql, con, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql_query", fake)
    [df] = IO._read_db(db, ["cell"])
    assert len(df) == 0
    assert list(df.columns) == ["object_label", "plateID", "cell_area"]


def test_read_db_concatenates_multiple_chunks(tmp_path, monkeypatch):
    """Two chunks are concatenated with ignore_index=True."""
    db = _small_cell_db(tmp_path / "m.db", n=6)
    real = pd.read_sql_query

    def fake(sql, con, *args, **kwargs):
        if "chunksize" in kwargs:
            kwargs.pop("chunksize")
            full = real(sql, con, *args, **kwargs)
            return iter([full.iloc[:2], full.iloc[2:]])
        return real(sql, con, *args, **kwargs)

    monkeypatch.setattr(pd, "read_sql_query", fake)
    [df] = IO._read_db(db, ["cell"])
    assert df["object_label"].tolist() == [1, 2, 3, 4, 5, 6]
    # ignore_index=True -> a clean RangeIndex, not 0,1,2,3,4,5 carried over
    assert list(df.index) == list(range(6))


# ---------------------------------------------------------------------------
# _read_and_merge_data
# ---------------------------------------------------------------------------

def _meta(obj_key):
    r = f"r{(obj_key % 2) + 1}"
    c = f"c{(obj_key % 2) + 1}"
    return {"plateID": "plate1", "rowID": r, "columnID": c, "fieldID": "f1",
            "prcf": f"plate1_{r}_{c}_f1"}


def _entity_frame(entity, n=6):
    """One row per object, keyed on its own object_label."""
    rows = []
    for obj in range(1, n + 1):
        row = dict(_meta(obj))
        row.update({
            "object_label": obj,
            f"{entity}_area": 100.0 + obj,
            f"{entity}_channel_0_mean_intensity": 500.0 + obj,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _child_frame(entity, cell_ids, object_labels=None):
    """Child objects (nucleus/pathogen) that point at a parent cell label."""
    rows = []
    if object_labels is None:
        object_labels = list(range(1, len(cell_ids) + 1))
    for obj, cid in zip(object_labels, cell_ids):
        row = dict(_meta(cid))
        row.update({
            "object_label": obj,
            "cell_id": cid,
            f"{entity}_area": 10.0 + obj,
            f"{entity}_channel_0_mean_intensity": 50.0 + obj,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _write_db(path, tables):
    with sqlite3.connect(path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
    return str(path)


def test_read_and_merge_cytoplasm_only(tmp_path, capsys):
    """cytoplasm without cell is grouped on its own object_label."""
    db = _write_db(tmp_path / "m.db", {"cytoplasm": _entity_frame("cytoplasm")})
    merged, obj_dfs = IO._read_and_merge_data([db], ["cytoplasm"], verbose=True)

    assert len(merged) == 6
    assert "cytoplasm_area" in merged.columns
    assert set(merged.index) == {
        f"plate1_r{(o % 2) + 1}_c{(o % 2) + 1}_f1_o{o}" for o in range(1, 7)}
    assert len(obj_dfs) == 1
    assert "cytoplasms grouped" in capsys.readouterr().out


def test_read_and_merge_nucleus_only_with_boolean_limit(tmp_path, capsys):
    """nuclei_limit=True keeps only cells that carry exactly one nucleus."""
    # cell 1 has two nuclei (labels 1 and 7), cells 2..6 have one each whose
    # nucleus label matches the parent cell label.
    nucleus = _child_frame("nucleus", [1, 1, 2, 3, 4, 5, 6],
                           object_labels=[1, 7, 2, 3, 4, 5, 6])
    db = _write_db(tmp_path / "m.db", {"nucleus": nucleus})

    merged, _ = IO._read_and_merge_data([db], ["nucleus"], verbose=True,
                                        nuclei_limit=True)
    assert len(merged) == 5
    assert not any(idx.endswith("_o1") for idx in merged.index)
    assert (merged["nucleus_prcfo_count"] == 1).all()
    assert "nucleus grouped" in capsys.readouterr().out


def test_read_and_merge_pathogen_only_with_boolean_limit(tmp_path, capsys):
    """pathogen_limit=True drops multi-pathogen cells and the surviving
    per-cell counts are rewritten as true integers."""
    pathogen = _child_frame("pathogen", [1, 1, 2, 3, 4, 5, 6],
                            object_labels=[1, 7, 2, 3, 4, 5, 6])
    db = _write_db(tmp_path / "m.db", {"pathogen": pathogen})

    merged, _ = IO._read_and_merge_data([db], ["pathogen"], verbose=True,
                                        pathogen_limit=True)
    assert len(merged) == 5
    assert not any(idx.endswith("_o1") for idx in merged.index)
    assert merged["pathogen_prcfo_count"].tolist() == [1] * 5
    assert str(merged["pathogen_prcfo_count"].dtype) == "Int64"
    assert "pathogens grouped" in capsys.readouterr().out


def test_read_and_merge_cell_plus_png_list(tmp_path, capsys):
    """png_list contributes its numeric and (pruned) non-numeric columns."""
    cell = _entity_frame("cell")
    png_rows = []
    for obj in range(1, 7):
        row = dict(_meta(obj))
        row.update({"cell_id": f"o{obj}",
                    "png_path": f"/x/data/cell_png/o{obj}.png",
                    "file_name": f"o{obj}.png",
                    "test": obj % 2})
        png_rows.append(row)
    db = _write_db(tmp_path / "m.db",
                   {"cell": cell, "png_list": pd.DataFrame(png_rows)})

    merged, _ = IO._read_and_merge_data([db], ["cell", "png_list"], verbose=True)

    assert len(merged) == 6
    assert "png_path" in merged.columns
    assert "test" in merged.columns
    # file_name / cell_id are pruned from the non-numeric png_list block
    assert "file_name" not in merged.columns
    assert merged["png_path"].str.endswith(".png").all()
    assert "png_list grouped" in capsys.readouterr().out


def test_read_and_merge_nucleus_only_keeps_rows_when_labels_differ(tmp_path):
    """A nucleus-only merge must return one row per parent cell even when the
    nucleus labels do not happen to equal the cell labels."""
    nucleus = _child_frame("nucleus", [1, 2, 3, 4, 5, 6],
                           object_labels=[11, 12, 13, 14, 15, 16])
    db = _write_db(tmp_path / "m.db", {"nucleus": nucleus})
    merged, _ = IO._read_and_merge_data([db], ["nucleus"], nuclei_limit=None)
    assert len(merged) == 6


def test_read_and_merge_change_plate_renames_plate(tmp_path):
    """change_plate=True should relabel each location as plate<idx>."""
    db = _write_db(tmp_path / "m.db", {"cell": _entity_frame("cell")})
    merged, _ = IO._read_and_merge_data([db], ["cell"], change_plate=True)
    assert set(merged["plateID"]) == {"plate1"}


# ---------------------------------------------------------------------------
# _read_mask / convert_numpy_to_tiff
# ---------------------------------------------------------------------------

def test_read_mask_upcasts_non_uint16(tmp_path):
    """A uint8 mask is promoted to uint16 by img_as_uint (values rescaled)."""
    p = tmp_path / "m.tif"
    mask = np.zeros((8, 8), np.uint8)
    mask[2:5, 2:5] = 3
    tifffile.imwrite(str(p), mask)

    out = IO._read_mask(str(p))
    assert out.dtype == np.uint16
    assert (out > 0).sum() == 9
    assert out.max() == 3 * 257  # skimage's uint8 -> uint16 scaling


def test_convert_numpy_to_tiff_honours_limit(tmp_path, rng):
    """limit stops the conversion loop early."""
    folder = tmp_path / "npys"
    folder.mkdir()
    for i in range(4):
        np.save(folder / f"a{i}.npy", rng.integers(0, 500, (4, 4)).astype(np.uint16))

    IO.convert_numpy_to_tiff(str(folder), limit=1)
    tiffs = list((folder / "tiff").glob("*.tif"))
    assert len(tiffs) == 1


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def ds_src(tmp_path, rng):
    """A plate folder with measurements.db (png_list) and the PNGs it points at."""
    def _build(name="plate1", n=6):
        src = tmp_path / name
        (src / "measurements").mkdir(parents=True)
        png_dir = src / "data" / "cell_png"
        png_dir.mkdir(parents=True)
        paths = [_png(png_dir / f"{name}_o{i + 1}.png", rng) for i in range(n)]
        with sqlite3.connect(src / "measurements" / "measurements.db") as con:
            pd.DataFrame({"png_path": paths,
                          "cell_id": [f"o{i + 1}" for i in range(n)]}
                         ).to_sql("png_list", con, index=False)
        return str(src)
    return _build


def test_generate_dataset_none_settings_and_no_images(tmp_path, monkeypatch):
    """settings=None falls back to the defaults ('path') and an empty
    png_list aborts before any tar is opened."""
    monkeypatch.chdir(tmp_path)
    meas = tmp_path / "path" / "measurements"
    meas.mkdir(parents=True)
    with sqlite3.connect(meas / "measurements.db") as con:
        con.execute("CREATE TABLE png_list (png_path TEXT)")

    with pytest.raises(RuntimeError, match="No images selected"):
        IO.generate_dataset(None)


def test_generate_dataset_rejects_non_str_non_list_src(tmp_path):
    """A src that is neither a str nor a list is refused."""
    src = tmp_path / "plate1"
    src.mkdir()
    with pytest.raises(RuntimeError, match="must be a string or list"):
        IO.generate_dataset({"src": Path(src), "file_metadata": None,
                             "experiment": "e", "sample": None})


def test_generate_dataset_sample_given_as_list(ds_src):
    """sample=[k] subsamples exactly k paths."""
    src = ds_src(n=6)
    tar = IO.generate_dataset({"src": src, "file_metadata": None,
                               "experiment": "listsample", "sample": [4]})
    with tarfile.open(tar) as t:
        assert len(t.getnames()) == 4


def test_generate_dataset_two_sources_get_combined_name(ds_src):
    """Two source roots produce a '<date>_combined_<experiment>.tar' in the
    first source's datasets/ folder."""
    a = ds_src("plateA", n=3)
    b = ds_src("plateB", n=3)
    tar = IO.generate_dataset({"src": [a, b], "file_metadata": None,
                               "experiment": "multi", "sample": None})
    assert os.path.basename(tar).endswith("_combined_multi.tar")
    assert os.path.dirname(tar) == os.path.join(a, "datasets")
    with tarfile.open(tar) as t:
        assert len(t.getnames()) == 6


def test_generate_dataset_existing_tar_is_not_overwritten(ds_src):
    """A second run with the same experiment name writes a suffixed tar."""
    src = ds_src(n=4)
    first = IO.generate_dataset({"src": src, "file_metadata": None,
                                 "experiment": "dup", "sample": None})
    second = IO.generate_dataset({"src": src, "file_metadata": None,
                                  "experiment": "dup", "sample": None})
    assert first != second
    assert os.path.isfile(first) and os.path.isfile(second)
    assert os.path.basename(second).startswith(
        os.path.basename(first)[:-len(".tar")])


# ---------------------------------------------------------------------------
# generate_loaders
# ---------------------------------------------------------------------------

def _class_tree(root, rng, classes=("nc", "pc"), n=6):
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _png(d / f"{cls}_{i}.png", rng, size=32)
    return root


def test_generate_loaders_default_classes_and_channels(tmp_path, rng):
    """classes=None / channels=None fall back to ['nc','pc'] and RGB."""
    _class_tree(tmp_path / "test", rng, n=4)
    loader, val, fig = IO.generate_loaders(str(tmp_path), mode="test",
                                           image_size=16, batch_size=2,
                                           classes=None, channels=None,
                                           n_jobs=0, verbose=True)
    assert val == [] and fig is None
    assert len(loader.dataset) == 8
    images, labels, paths = next(iter(loader))
    assert images.shape[1] == 3          # all three channels kept
    assert set(labels.tolist()) <= {0, 1}


def test_generate_loaders_invalid_mode_returns_none(tmp_path, capsys):
    assert IO.generate_loaders(str(tmp_path), mode="validate") is None
    assert "is not valid" in capsys.readouterr().out


def test_generate_loaders_missing_split_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Generate Training Data"):
        IO.generate_loaders(str(tmp_path), mode="train", n_jobs=0)


def test_generate_loaders_missing_class_folder_lists_available(tmp_path, rng):
    _class_tree(tmp_path / "train", rng, classes=("nc",), n=2)
    with pytest.raises(FileNotFoundError) as excinfo:
        IO.generate_loaders(str(tmp_path), mode="train",
                            classes=["nc", "pc"], n_jobs=0)
    msg = str(excinfo.value)
    assert "Missing:   ['pc']" in msg
    assert "Available: ['nc']" in msg


def test_generate_loaders_augment_expands_train_split(tmp_path, rng):
    """augment=True runs every train image through the 8-fold augmentation."""
    _class_tree(tmp_path / "train", rng, n=4)
    train, val, _ = IO.generate_loaders(str(tmp_path), mode="train",
                                        image_size=16, batch_size=2,
                                        classes=["nc", "pc"], n_jobs=0,
                                        validation_split=0.25, augment=True)
    # 8 images, 25% held out -> 6 train, 2 val; augmentation is 4 rotations
    # x 2 reflections = 8 copies of each train image.
    assert len(val.dataset) == 2
    assert len(train.dataset) == 6 * 8


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

N = 40


def _build_png_src(root, rng, n=N, prefix="o", extra_cols=None,
                   png_paths=None, with_condition=False):
    """<root>/measurements/measurements.db (png_list only) + the PNG crops."""
    src = Path(root)
    (src / "measurements").mkdir(parents=True, exist_ok=True)
    png_dir = src / "data" / "cell_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    real_paths = [_png(png_dir / f"{prefix}{i + 1}.png", rng) for i in range(n)]
    stored = list(real_paths) if png_paths is None else list(png_paths)

    df = pd.DataFrame({
        "png_path": stored,
        "cell_id": [f"o{i + 1}" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": [f"r{(i % 2) + 1}" for i in range(n)],
        "columnID": [f"c{(i % 2) + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "test": [1 if i % 2 == 0 else 2 for i in range(n)],
    })
    if with_condition:
        df["condition"] = ["c1" if i % 2 == 0 else "c2" for i in range(n)]
    for key, values in (extra_cols or {}).items():
        df[key] = values

    with sqlite3.connect(src / "measurements" / "measurements.db") as con:
        df.to_sql("png_list", con, index=False, if_exists="replace")
    return str(src), real_paths


def _gtd(src, **over):
    settings = {"src": src, "dataset_mode": "metadata",
                "png_type": "cell_png", "test_split": 0.2}
    settings.update(over)
    return settings


def _class_counts(train_dir, test_dir):
    return {cls: (len(os.listdir(os.path.join(train_dir, cls))),
                  len(os.listdir(os.path.join(test_dir, cls))))
            for cls in sorted(os.listdir(train_dir))}


def test_generate_training_dataset_named_metadata_rules(tmp_path, rng, monkeypatch):
    """Named rules exercise every _apply_where operator, the balancer, the
    unique-directory helper and the best-effort settings re-save."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    # a pre-existing destination forces the _<n> suffix
    os.makedirs(os.path.join(src, "datasets", "training"))

    import spacr.utils as U
    real_save = U.save_settings

    def flaky_save(settings, name="settings", show=False):
        if not show:                     # the trailing best-effort re-save
            raise RuntimeError("settings volume went away")
        return real_save(settings, name, show)

    monkeypatch.setattr(U, "save_settings", flaky_save)

    rules = [
        {"name": "colc1", "column": "columnID", "op": "==", "value": "c1"},
        {"name": "colc2", "where": [{"column": "columnID", "op": "in",
                                     "value": ["c2"]}]},
        {"name": "notr2", "where": [{"column": "rowID", "op": "notin",
                                     "value": ["r2"]}]},
        {"name": "lowtest", "where": [{"column": "test", "op": "<=", "value": 1}]},
        {"name": "hightest", "where": [{"column": "test", "op": ">", "value": 1}]},
        {"name": "everything"},          # no where at all -> all rows
    ]
    train_dir, test_dir = IO.generate_training_dataset(
        _gtd(src, metadata_rules=rules))

    assert train_dir == os.path.join(src, "datasets", "training_1", "train")
    counts = _class_counts(train_dir, test_dir)
    assert sorted(counts) == ["colc1", "colc2", "everything", "hightest",
                              "lowtest", "notr2"]
    # balanced down to the smallest class (20) then split 80/20
    assert all(counts[c] == (16, 4) for c in counts)


def test_generate_training_dataset_unknown_column_aborts(tmp_path, rng, capsys):
    """A rule naming a column that isn't in png_list selects nothing."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    out = IO.generate_training_dataset(_gtd(src, metadata_rules=[
        {"name": "ghost", "where": [{"column": "not_a_column", "op": "==",
                                     "value": 1}]},
        {"name": "badop", "where": [{"column": "columnID", "op": "~=",
                                     "value": 1}]},
    ]))
    assert out == (None, None)
    assert "No class data assembled" in capsys.readouterr().out


def test_generate_training_dataset_unnamed_rules_get_derived_names(tmp_path, rng):
    """Rules without a 'name' key are labelled '<col><op><value>'."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    train_dir, test_dir = IO.generate_training_dataset(_gtd(src, metadata_rules=[
        {"column": "columnID", "op": "==", "value": "c1"},
        {"column": "columnID", "op": "==", "value": "c2"},
    ]))
    assert sorted(os.listdir(train_dir)) == ["columnID==c1", "columnID==c2"]
    assert _class_counts(train_dir, test_dir)["columnID==c1"] == (16, 4)


def test_generate_training_dataset_metadata_mode_with_no_classes(tmp_path, rng, capsys):
    """An empty class_metadata selects nothing and aborts, saying so.

    This used to assert that the run PRINTED ``'condition' column not found``.
    That message came from a guard which detected the missing column and then
    let the next line index it anyway -- so the log said "got 0 classes" while
    the run died of ``KeyError: 'condition'`` two frames down. The column is
    now taken from ``metadata_type_by`` (``'columnID'`` by default, which this
    fixture has), and a column that really is absent raises instead of being
    announced and then used -- see
    ``test_generate_training_dataset_metadata_column_is_missing`` below and
    ``tests/test_training_dataset_metadata_column.py``.
    """
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    out = IO.generate_training_dataset(
        _gtd(src, metadata_rules=None, class_metadata=[]))
    assert out == (None, None)
    assert "No class data assembled" in capsys.readouterr().out


def test_generate_training_dataset_metadata_column_is_missing(tmp_path, rng):
    """A metadata_type_by naming a column png_list does not have raises."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    with pytest.raises(ValueError) as excinfo:
        IO.generate_training_dataset(
            _gtd(src, metadata_rules=None, metadata_type_by="condition",
                 class_metadata=["c1"]))
    assert "'condition'" in str(excinfo.value)
    assert "metadata_type_by" in str(excinfo.value)


def test_generate_training_dataset_measurement_rules(tmp_path, rng):
    """measurement mode buckets rows with numeric where-clauses."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, dataset_mode="measurement", measurement_rules=[
            {"name": "low", "where": [{"column": "test", "op": "<", "value": 2}]},
            {"name": "high", "where": [{"column": "test", "op": ">=", "value": 2}]},
        ]))
    assert sorted(os.listdir(train_dir)) == ["high", "low"]
    assert _class_counts(train_dir, test_dir) == {"high": (16, 4), "low": (16, 4)}


def test_generate_training_dataset_invalid_mode(tmp_path, rng, capsys):
    src, _ = _build_png_src(tmp_path / "plate1", rng, n=4)
    assert IO.generate_training_dataset(
        _gtd(src, dataset_mode="nonsense")) == (None, None)
    assert "Invalid dataset_mode" in capsys.readouterr().out


def test_generate_training_dataset_annotation_without_columns(tmp_path, rng):
    """No annotation columns at all -> no classes -> clean abort."""
    src, _ = _build_png_src(tmp_path / "plate1", rng, n=4)
    assert IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=[],
        annotation_column=None)) == (None, None)


def test_generate_training_dataset_annotation_random_class_written_to_db(tmp_path, rng):
    """A column with a single annotated value gains a '<col>_random' class
    sampled from the unannotated rows, and that selection is persisted."""
    flag = [1 if i < 10 else None for i in range(N)]
    src, _ = _build_png_src(tmp_path / "plate1", rng, extra_cols={"flag": flag})

    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation",
        annotation_columns=["flag", "missing_column"],
        write_random_annotation_column=True))

    assert sorted(os.listdir(train_dir)) == ["flag_1", "flag_random"]
    assert _class_counts(train_dir, test_dir) == {"flag_1": (8, 2),
                                                  "flag_random": (8, 2)}

    db = os.path.join(src, "measurements", "measurements.db")
    con = sqlite3.connect(db)
    try:
        cols = {r[1] for r in con.execute('PRAGMA table_info("png_list")')}
        assert "flag_random" in cols
        marked = con.execute(
            'SELECT COUNT(*) FROM png_list WHERE "flag_random" = 1').fetchone()[0]
        # exactly as many negatives as positives, all drawn from the
        # unannotated rows
        assert marked == 10
        overlap = con.execute(
            'SELECT COUNT(*) FROM png_list WHERE "flag_random" = 1 '
            'AND flag IS NOT NULL').fetchone()[0]
        assert overlap == 0
    finally:
        con.close()


def test_generate_training_dataset_annotation_fully_annotated_skips_random(tmp_path, rng, capsys):
    """No unannotated rows -> no random class is invented."""
    src, _ = _build_png_src(tmp_path / "plate1", rng,
                            extra_cols={"allone": [1] * N})
    train_dir, _ = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=["allone"]))
    assert os.listdir(train_dir) == ["allone_1"]
    assert "no unannotated rows available" in capsys.readouterr().out


def test_generate_training_dataset_annotation_more_positives_than_negatives(tmp_path, rng):
    """Fewer unannotated rows than positives -> take them all, then balance."""
    many = [1 if i < 30 else None for i in range(N)]
    src, _ = _build_png_src(tmp_path / "plate1", rng, extra_cols={"many": many})
    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=["many"]))
    counts = _class_counts(train_dir, test_dir)
    assert sorted(counts) == ["many_1", "many_random"]
    # balanced down to the 10 available negatives
    assert counts["many_1"] == counts["many_random"] == (8, 2)


def test_generate_training_dataset_annotation_value_filter(tmp_path, rng):
    """annotation_values restricts which observed values become classes."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    train_dir, _ = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=["test"],
        annotation_values={"test": [1]}))
    assert os.listdir(train_dir) == ["test_1"]


def test_generate_training_dataset_annotation_non_numeric_labels(tmp_path, rng):
    """String annotations that cannot be cast to int are used verbatim."""
    grades = ["A" if i % 2 == 0 else "B" for i in range(N)]
    src, _ = _build_png_src(tmp_path / "plate1", rng,
                            extra_cols={"grade": grades})
    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=["grade"]))
    assert sorted(os.listdir(train_dir)) == ["grade_A", "grade_B"]
    assert _class_counts(train_dir, test_dir)["grade_A"] == (16, 4)


def test_generate_training_dataset_annotation_positives_without_paths(tmp_path, rng, capsys):
    """Annotated rows whose png_path is NULL produce zero positives, so no
    random class is built and nothing is assembled."""
    src_dir = tmp_path / "plate1"
    # rows 0..9 have no png_path at all
    real = [str(src_dir / "data" / "cell_png" / f"o{i + 1}.png") for i in range(N)]
    paths = [None if i < 10 else real[i] for i in range(N)]
    ghost = [1 if i < 10 else None for i in range(N)]
    src, _ = _build_png_src(src_dir, rng, png_paths=paths,
                            extra_cols={"ghost": ghost})

    out = IO.generate_training_dataset(_gtd(
        src, dataset_mode="annotation", annotation_columns=["ghost"],
        png_type=""))
    assert out == (None, None)
    assert "0 rows; skipping random class" in capsys.readouterr().out


def test_generate_training_dataset_repairs_png_paths(tmp_path, rng):
    """png_paths recorded on another machine are re-rooted under src; paths
    that cannot be repaired are dropped by the png_type filter."""
    src_dir = tmp_path / "plate1"
    names = [f"o{i + 1}.png" for i in range(N)]
    stored = []
    for i, name in enumerate(names):
        if i < 10:                                    # already correct
            stored.append(str(src_dir / "data" / "cell_png" / name))
        elif i < 20:                                  # foreign absolute root
            stored.append(f"/other/machine/plate9/data/cell_png/{name}")
        elif i < 30:                                  # relative to src
            stored.append(f"data/cell_png/{name}")
        else:                                         # unrepairable
            stored.append(f"/elsewhere/{name}")
    src, _ = _build_png_src(src_dir, rng, png_paths=stored)

    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, metadata_rules=[{"name": "all"}], balance_to_smallest=False))
    n_train = len(os.listdir(os.path.join(train_dir, "all")))
    n_test = len(os.listdir(os.path.join(test_dir, "all")))
    assert n_train + n_test == 30       # the 10 unrepairable rows are gone


def test_generate_training_dataset_balance_disabled(tmp_path, rng):
    """balance_to_smallest=False keeps class sizes as-is."""
    src, _ = _build_png_src(tmp_path / "plate1", rng)
    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        src, balance_to_smallest=False, metadata_rules=[
            {"name": "half", "column": "columnID", "op": "==", "value": "c1"},
            {"name": "all"},
        ]))
    counts = _class_counts(train_dir, test_dir)
    assert counts["half"] == (16, 4)
    assert counts["all"] == (32, 8)


def test_generate_training_dataset_multiple_sources_align_by_index(tmp_path, rng, capsys):
    """Two sources with different annotation columns warn and align by index;
    the first source gets the 'training_all' destination."""
    src_a, _ = _build_png_src(tmp_path / "plateA", rng, n=20, prefix="a",
                              extra_cols={"alpha": [1] * 10 + [2] * 10,
                                          "beta": [1] * 10 + [2] * 10})
    src_b, _ = _build_png_src(tmp_path / "plateB", rng, n=20, prefix="b",
                              extra_cols={"alpha": [1] * 10 + [2] * 10})

    train_dir, test_dir = IO.generate_training_dataset(_gtd(
        [src_a, src_b], dataset_mode="annotation",
        annotation_columns=["alpha", "beta"]))

    out = capsys.readouterr().out
    assert "class name/order mismatch" in out
    assert "annotation column 'beta' not in png_list" in out
    # One combined dataset is written beside the first source.
    assert train_dir == os.path.join(
        src_a, "datasets", "training_all", "train")
    counts = _class_counts(train_dir, test_dir)
    # alpha_1/alpha_2 collected 10 from each source, beta_* only from A
    assert sorted(counts) == ["alpha_1", "alpha_2", "beta_1", "beta_2"]
    assert all(sum(v) == 10 for v in counts.values())


# ---------------------------------------------------------------------------
# training_dataset_from_annotation_metadata
# ---------------------------------------------------------------------------

@pytest.fixture
def anno_db(tmp_path, rng):
    src, _ = _build_png_src(tmp_path / "plateM", rng, n=20)
    return os.path.join(src, "measurements", "measurements.db")


def test_training_dataset_from_annotation_metadata_default_class_metadata(anno_db, tmp_path):
    """class_metadata=None defaults to ['c1','c2'] which actually matches the
    flat columnID values."""
    out = IO.training_dataset_from_annotation_metadata(
        anno_db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1, 2), metadata_type_by="columnID",
        class_metadata=None)
    assert len(out) == 2
    assert len(out[0]) == 10 and len(out[1]) == 10


def test_training_dataset_from_annotation_metadata_rowid_filter(anno_db, tmp_path):
    """Filtering by rowID keeps only the matching half of the plate."""
    out = IO.training_dataset_from_annotation_metadata(
        anno_db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1, 2), metadata_type_by="rowID",
        class_metadata=["r1"])
    # rowID r1 and test==1 select the same even rows
    assert len(out[0]) == 10 and len(out[1]) == 0


def test_training_dataset_from_annotation_metadata_bad_key(anno_db, tmp_path):
    with pytest.raises(ValueError, match="Invalid metadata_type_by"):
        IO.training_dataset_from_annotation_metadata(
            anno_db, str(tmp_path / "dst"), metadata_type_by="wellID")


def test_training_dataset_from_annotation_metadata_single_class(anno_db, tmp_path):
    """One annotated class -> a balanced 'other' class is sampled."""
    out = IO.training_dataset_from_annotation_metadata(
        anno_db, str(tmp_path / "dst"), annotation_column="test",
        annotated_classes=(1,), metadata_type_by="columnID",
        class_metadata=None)
    assert len(out) == 2
    assert len(out[0]) == len(out[1]) == 10
    assert not set(out[0]) & set(out[1])


# ---------------------------------------------------------------------------
# convert_separate_files_to_yokogawa
# ---------------------------------------------------------------------------

_SEP_REGEX = (r"W(?P<wellID>\d+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
              r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")


def _tiny_tif(path, value=1, shape=(2, 2)):
    tifffile.imwrite(str(path), np.full(shape, value, np.uint16))


def test_convert_separate_files_two_single_slice_regions(tmp_path):
    """Two source wells get two Yokogawa wells; a single slice is copied
    through without a MIP."""
    folder = tmp_path / "raw"
    folder.mkdir()
    _tiny_tif(folder / "W1_F1_T1_C1_Z1.tif", value=5)
    _tiny_tif(folder / "W2_F1_T1_C1_Z1.tif", value=9)

    IO.convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)

    produced = sorted(p.name for p in folder.glob("plate1_*.tif"))
    assert produced == ["plate1_A01_T0001F001L01C01.tif",
                        "plate1_A02_T0001F001L01C01.tif"]
    values = {int(tifffile.imread(str(folder / n)).max()) for n in produced}
    assert values == {5, 9}
    log = pd.read_csv(folder / "rename_log.csv")
    assert len(log) == 2


def test_convert_separate_files_skips_unmatched_and_wellless(tmp_path):
    """Files that don't match, and matches without a wellID, are skipped."""
    folder = tmp_path / "raw"
    folder.mkdir()
    _tiny_tif(folder / "F1.tif")
    (folder / "notes.txt").write_text("hello")

    regex = r"(W(?P<wellID>\d+))?F(?P<fieldID>\d+)\.tif"
    IO.convert_separate_files_to_yokogawa(str(folder), regex)

    assert list(folder.glob("plate*.tif")) == []
    assert (folder / "rename_log.csv").read_text().strip() == ""


def test_convert_separate_files_overflows_onto_second_plate(tmp_path):
    """The 385th region rolls over onto plate2."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for i in range(1, 386):
        _tiny_tif(folder / f"W{i}_F1_T1_C1_Z1.tif")

    IO.convert_separate_files_to_yokogawa(str(folder), _SEP_REGEX)

    assert (folder / "plate2_A01_T0001F001L01C01.tif").is_file()
    assert len(list(folder.glob("plate1_*.tif"))) == 384
    assert len(pd.read_csv(folder / "rename_log.csv")) == 385


# ---------------------------------------------------------------------------
# convert_to_yokogawa
# ---------------------------------------------------------------------------

def test_convert_to_yokogawa_overflows_onto_second_plate(tmp_path):
    """385 plain 2-D TIFFs consume a whole 384-well plate plus one."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for i in range(385):
        _tiny_tif(folder / f"img_{i:04d}.tif")

    IO.convert_to_yokogawa(str(folder))

    assert (folder / "plate2_A01_T0001F001L01C01.tif").is_file()
    assert len(pd.read_csv(folder / "rename_log.csv")) == 385


def test_convert_to_yokogawa_zstack_4d_and_unsupported(tmp_path, capsys):
    """3-D z-stacks are MIP'd, 4-D stacks are split per timepoint, and
    anything with more dimensions is reported instead of crashing."""
    folder = tmp_path / "raw"
    folder.mkdir()
    zstack = np.zeros((6, 4, 4), np.uint16)
    zstack[3] = 77
    tifffile.imwrite(str(folder / "zzz.tif"), zstack)
    tifffile.imwrite(str(folder / "ttt.tif"),
                     np.ones((2, 3, 4, 4), np.uint16))
    tifffile.imwrite(str(folder / "five.tif"),
                     np.ones((2, 2, 2, 4, 4), np.uint16))
    (folder / "broken.tif").write_bytes(b"not a tiff at all")

    IO.convert_to_yokogawa(str(folder))

    produced = sorted(p.name for p in folder.glob("plate*.tif"))
    assert len(produced) == 3                       # 1 z-stack MIP + 2 frames
    assert sum(n.endswith("T0002F001L01C01.tif") for n in produced) == 1
    mips = [tifffile.imread(str(folder / n)).max() for n in produced]
    assert 77 in mips                               # the MIP kept the bright plane
    err = capsys.readouterr().out
    assert "Unsupported TIFF dimensions" in err
    assert "Error processing standard image file broken.tif" in err
    assert len(pd.read_csv(folder / "rename_log.csv")) == 3


# --- vendor formats: ND2 / CZI / LIF, driven through stub readers ----------

class _StubND2:
    """Minimal stand-in for nd2reader.ND2Reader."""

    def __init__(self, path, fail=False):
        self.path = path
        self.fail = fail
        self.metadata = {"frames": [0], "fields_of_view": [0],
                         "z_levels": [0, 1], "channels": ["DAPI"]}

    def get_frame_2D(self, t=0, v=0, z=0, c=0):
        if self.fail:
            raise IndexError("truncated ND2")
        return np.full((4, 4), z + 1, np.uint16)


def test_convert_to_yokogawa_nd2_incomplete_frames(tmp_path, monkeypatch, capsys):
    """A frame the ND2 doesn't contain should be reported and skipped."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "movie.nd2").write_bytes(b"stub")
    monkeypatch.setattr(IO, "ND2Reader", lambda p: _StubND2(p, fail=True))

    IO.convert_to_yokogawa(str(folder))

    assert list(folder.glob("plate*.tif")) == []
    assert "incomplete data structure" in capsys.readouterr().out


def test_convert_to_yokogawa_nd2_reader_failure(tmp_path, monkeypatch, capsys):
    """A reader that blows up is caught per-file."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "movie.nd2").write_bytes(b"stub")

    def boom(_path):
        raise OSError("cannot open ND2")

    monkeypatch.setattr(IO, "ND2Reader", boom)
    IO.convert_to_yokogawa(str(folder))
    assert "Error processing ND2 file movie.nd2" in capsys.readouterr().out


def test_convert_to_yokogawa_nd2_writes_mip(tmp_path, monkeypatch):
    """A readable ND2 should be max-projected over Z and written out."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "movie.nd2").write_bytes(b"stub")
    monkeypatch.setattr(IO, "ND2Reader", lambda p: _StubND2(p))

    IO.convert_to_yokogawa(str(folder))

    out = sorted(folder.glob("plate*.tif"))
    assert len(out) == 1
    assert tifffile.imread(str(out[0])).max() == 2


class _StubCziDoc:
    def __init__(self, scenes):
        self._scenes = scenes
        self.total_bounding_box = {"T": (0, 1), "C": (0, 2), "Z": (0, 1)}

    @property
    def scenes_bounding_rectangle(self):
        return self._scenes

    def read(self, plane=None, scene=None):
        return np.full((1, 4, 4, 1), (plane["C"] + 1), np.uint16)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_convert_to_yokogawa_czi_with_scenes(tmp_path, monkeypatch):
    """Each CZI scene gets its own well and its own field index."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "img.czi").write_bytes(b"stub")
    monkeypatch.setattr(IO, "pyczi", types.SimpleNamespace(
        open_czi=lambda path: _StubCziDoc({0: (0, 0, 4, 4), 1: (4, 0, 4, 4)})))

    IO.convert_to_yokogawa(str(folder))

    produced = sorted(p.name for p in folder.glob("plate*.tif"))
    # 2 scenes x 1 T x 2 C x 1 Z, scenes take wells A02 and A03 (A01 went to
    # the source file itself)
    assert produced == [
        "plate1_A02_T0001F001L01A01Z01C01.tif",
        "plate1_A02_T0001F001L01A01Z01C02.tif",
        "plate1_A03_T0001F002L01A02Z01C01.tif",
        "plate1_A03_T0001F002L01A02Z01C02.tif",
    ]
    assert tifffile.imread(str(folder / produced[1])).max() == 2
    assert len(pd.read_csv(folder / "rename_log.csv")) == 4


def test_convert_to_yokogawa_czi_without_scenes(tmp_path, monkeypatch):
    """A CZI with no scene table falls back to a single field."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "img.czi").write_bytes(b"stub")
    monkeypatch.setattr(IO, "pyczi", types.SimpleNamespace(
        open_czi=lambda path: _StubCziDoc({})))

    IO.convert_to_yokogawa(str(folder))

    produced = sorted(p.name for p in folder.glob("plate*.tif"))
    assert produced == ["plate1_A02_T0001F001L01A01Z01C01.tif",
                        "plate1_A02_T0001F001L01A01Z01C02.tif"]


def test_convert_to_yokogawa_czi_reader_failure(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "img.czi").write_bytes(b"stub")

    def boom(path):
        raise OSError("cannot open CZI")

    monkeypatch.setattr(IO, "pyczi", types.SimpleNamespace(open_czi=boom))
    IO.convert_to_yokogawa(str(folder))
    assert "Error processing CZI file img.czi" in capsys.readouterr().out


class _StubLifImage:
    def __init__(self, drop_z=None):
        self.dims = types.SimpleNamespace(t=1, z=2, c=2)
        self.drop_z = drop_z

    def getFrame(self, z=0, t=0, c=0):
        if self.drop_z is not None and z == self.drop_z:
            raise IndexError("missing plane")
        return np.full((4, 4), z + 1, np.uint16)


def _stub_readlif(images):
    class _Reader:
        def __init__(self, path):
            self.path = path

        def getIterImage(self):
            return list(images)

    return types.SimpleNamespace(Reader=_Reader)


def test_convert_to_yokogawa_lif_mips_z_stack(tmp_path, monkeypatch):
    """Each (image, t, c) is max-projected over Z."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "stack.lif").write_bytes(b"stub")
    monkeypatch.setattr(IO, "readlif", _stub_readlif([_StubLifImage()]))

    IO.convert_to_yokogawa(str(folder))

    produced = sorted(p.name for p in folder.glob("plate*.tif"))
    assert produced == ["plate1_A01_T0001F001L01C01.tif",
                        "plate1_A01_T0001F001L01C02.tif"]
    assert tifffile.imread(str(folder / produced[0])).max() == 2


def test_convert_to_yokogawa_lif_missing_frame(tmp_path, monkeypatch, capsys):
    """A missing Z plane is skipped, the rest is still projected."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "stack.lif").write_bytes(b"stub")
    monkeypatch.setattr(IO, "readlif", _stub_readlif([_StubLifImage(drop_z=1)]))

    IO.convert_to_yokogawa(str(folder))

    produced = sorted(folder.glob("plate*.tif"))
    assert len(produced) == 2
    assert tifffile.imread(str(produced[0])).max() == 1
    assert "Missing frame" in capsys.readouterr().out


def test_convert_to_yokogawa_lif_reader_failure(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "stack.lif").write_bytes(b"stub")

    class _Boom:
        def __init__(self, path):
            raise OSError("cannot open LIF")

    monkeypatch.setattr(IO, "readlif", types.SimpleNamespace(Reader=_Boom))
    IO.convert_to_yokogawa(str(folder))
    assert "Error processing LIF file stack.lif" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset
# ---------------------------------------------------------------------------

def _cellpose_dataset(root, name, n):
    d = root / name
    m = d / "masks"
    m.mkdir(parents=True)
    for i in range(n):
        fname = f"{name}_{i}.tif"
        tifffile.imwrite(str(d / fname), np.full((8, 8), i + 1, np.uint16))
        mask = np.zeros((8, 8), np.uint16)
        mask[1:4, 1:4] = 1
        tifffile.imwrite(str(m / fname), mask)
    return d


def test_prepare_cellpose_dataset_without_datasets_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "loose.tif").write_bytes(b"x")     # no <subdir>/masks anywhere
    with pytest.raises(ValueError, match="No valid datasets"):
        IO.prepare_cellpose_dataset(str(empty), n_jobs=1)


def test_prepare_cellpose_dataset_augments_with_full_loops_and_remainder(tmp_path):
    """The small dataset is padded to the big one's size using whole
    augmentation passes plus a remainder pass."""
    root = tmp_path / "sets"
    _cellpose_dataset(root, "big", 12)
    _cellpose_dataset(root, "small", 6)

    IO.prepare_cellpose_dataset(str(root), augment_data=True,
                                train_fraction=0.75, n_jobs=1)

    out = root / "cellpose_dataset"
    train_imgs = sorted((out / "train" / "images").glob("*.tif"))
    test_imgs = sorted((out / "test" / "images").glob("*.tif"))
    # both datasets are levelled to 12 pairs -> 24 total, split 9/3 each
    assert len(train_imgs) == 18
    assert len(test_imgs) == 6
    assert len(list((out / "train" / "masks").glob("*.tif"))) == 18
    assert tifffile.imread(str(train_imgs[0])).shape == (8, 8)


def test_prepare_cellpose_dataset_default_n_jobs(tmp_path, monkeypatch):
    """n_jobs=None derives the worker count from cpu_count()."""
    monkeypatch.setattr(IO, "cpu_count", lambda: 2)
    root = tmp_path / "sets"
    _cellpose_dataset(root, "only", 4)

    IO.prepare_cellpose_dataset(str(root), augment_data=False,
                               train_fraction=0.5, n_jobs=None)

    out = root / "cellpose_dataset"
    assert len(list((out / "train" / "images").glob("*.tif"))) == 2
    assert len(list((out / "test" / "images").glob("*.tif"))) == 2
