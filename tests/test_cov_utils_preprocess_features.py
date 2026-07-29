"""CPU-only coverage for the spacr.utils "preprocess / feature filtering" block.

Covers the cluster-grid renderer (:func:`plot_grid`), the DB path helpers
(:func:`generate_path_list_from_db`, :func:`correct_paths`), the filesystem
helpers (:func:`delete_folder`, :func:`measure_test_mode`) and the feature
matrix pipeline (:func:`preprocess_data`, :func:`remove_low_variance_columns`,
:func:`remove_highly_correlated_columns`, :func:`filter_dataframe_features`)
plus the overlap-avoidance helpers.

Everything runs offline on tiny synthetic frames; matplotlib is driven under
the Agg backend and every figure is closed by the autouse fixture below.
"""
from __future__ import annotations

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_grid
# ---------------------------------------------------------------------------

def _tiny_img(val=17):
    return np.full((4, 4), val, dtype=np.uint8)


def test_plot_grid_single_cluster_string_label(capsys):
    """One cluster => the Axes object must be wrapped into a list, and a str
    cluster label must be resolved to a color by *key position*, not by index
    lookup into `colors` (which would raise on a str)."""
    from spacr.utils import plot_grid

    cluster_images = {"alpha": [_tiny_img(10), _tiny_img(200)]}
    colors = [(1.0, 0.0, 0.0, 1.0)]

    fig = plot_grid(cluster_images, colors, figuresize=2,
                    black_background=False, verbose=True)

    # verbose branch printed the resolved index
    out = capsys.readouterr().out
    assert "Lable: alpha index: 0" in out

    # exactly one cluster column, holding one inset per image
    assert len(fig.axes) == 1
    col = fig.axes[0]
    assert len(col.child_axes) == 2
    assert [int(c.images[0].get_array().flat[0]) for c in col.child_axes] == [10, 200]
    # the column background patch carries the cluster color
    assert col.patches[0].get_facecolor() == (1.0, 0.0, 0.0, 1)
    # one label text + one legend swatch patch were added to the figure
    assert [t.get_text() for t in fig.texts] == ["Cluster alpha"]
    assert len(fig.patches) == 1
    assert fig.texts[0].get_color() == "black"


def test_plot_grid_clamps_oversized_figuresize():
    """figuresize * num_clusters > 200 must be rescaled so the total figure
    width is exactly the 200 inch cap."""
    from spacr.utils import plot_grid

    cluster_images = {0: [_tiny_img()], 1: [_tiny_img()], 2: [_tiny_img()]}
    colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]

    fig = plot_grid(cluster_images, colors, figuresize=100,
                    black_background=True, verbose=False)

    w, h = fig.get_size_inches()
    assert w == pytest.approx(200.0)
    assert h == pytest.approx(200.0 / 3.0)
    # int labels index straight into `colors`
    assert len(fig.axes) == 3
    assert [tuple(ax.patches[0].get_facecolor()) for ax in fig.axes] == [
        (1.0, 0.0, 0.0, 1), (0.0, 1.0, 0.0, 1), (0.0, 0.0, 1.0, 1)]
    assert all(len(ax.child_axes) == 1 for ax in fig.axes)
    assert [t.get_text() for t in fig.texts] == [
        "Cluster 0", "Cluster 1", "Cluster 2"]
    assert fig.texts[0].get_color() == "white"


# ---------------------------------------------------------------------------
# generate_path_list_from_db
# ---------------------------------------------------------------------------

def _png_db(tmp_path, paths, name="measurements.db"):
    db_path = tmp_path / name
    con = sqlite3.connect(db_path)
    try:
        con.execute("CREATE TABLE png_list (png_path TEXT)")
        con.executemany("INSERT INTO png_list (png_path) VALUES (?)",
                        [(p,) for p in paths])
        con.commit()
    finally:
        con.close()
    return str(db_path)


def test_generate_path_list_from_db_no_metadata(tmp_path):
    from spacr.utils import generate_path_list_from_db

    paths = [f"/data/plate1/A{i:02d}_1_cell_{i}.png" for i in range(5)]
    got = generate_path_list_from_db(_png_db(tmp_path, paths), None)
    assert sorted(got) == sorted(paths)


def test_generate_path_list_from_db_single_string_metadata(tmp_path):
    from spacr.utils import generate_path_list_from_db

    paths = ["/data/p1/A01_1.png", "/data/p1/B02_1.png", "/data/p1/C03_1.png"]
    got = generate_path_list_from_db(_png_db(tmp_path, paths), "B02")
    assert got == ["/data/p1/B02_1.png"]


def test_generate_path_list_from_db_list_metadata(tmp_path):
    """A list of substrings must be OR-ed together into one LIKE query."""
    from spacr.utils import generate_path_list_from_db

    paths = ["/data/p1/A01_1.png", "/data/p1/B02_1.png",
             "/data/p1/C03_1.png", "/data/p1/D04_1.png"]
    got = generate_path_list_from_db(_png_db(tmp_path, paths), ["A01", "C03"])
    assert sorted(got) == ["/data/p1/A01_1.png", "/data/p1/C03_1.png"]


def test_generate_path_list_from_db_returns_none_on_sqlite_error(tmp_path, capsys):
    """A missing png_list table raises sqlite3.OperationalError -> None."""
    from spacr.utils import generate_path_list_from_db

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()

    got = generate_path_list_from_db(str(empty), None)
    assert got is None
    assert "Database error:" in capsys.readouterr().out


def test_generate_path_list_from_db_returns_none_on_generic_error(tmp_path, monkeypatch, capsys):
    """Any non-sqlite3 failure is swallowed by the broad handler -> None."""
    import spacr.utils as U
    from spacr.utils import generate_path_list_from_db

    def boom(*a, **k):
        raise ValueError("connect exploded")

    monkeypatch.setattr(U.sqlite3, "connect", boom)

    got = generate_path_list_from_db(str(tmp_path / "nope.db"), "A01")
    assert got is None
    out = capsys.readouterr().out
    assert "Error: connect exploded" in out
    assert "Database error:" not in out


# ---------------------------------------------------------------------------
# correct_paths
# ---------------------------------------------------------------------------

def test_correct_paths_dataframe_without_png_path_column(capsys):
    """Missing column => (df, None) and an explanatory message."""
    from spacr.utils import correct_paths

    df = pd.DataFrame({"something_else": ["a", "b"]})
    out_df, out_paths = correct_paths(df, base_path="/new/base", folder="data")

    assert out_paths is None
    assert out_df is df
    assert "No 'png_path' column found in the dataframe." in capsys.readouterr().out


def test_correct_paths_rewrites_paths_under_new_base():
    """Paths that contain '/<folder>/' get re-rooted at base_path/<folder>/."""
    from spacr.utils import correct_paths

    df = pd.DataFrame({"png_path": [
        "/old/root/data/plate1/A01_1_cell_1.png",   # rewritten
        "/old/root/data/plate1/A01_1_cell_2.png",   # rewritten
        "/old/root/misc/no_folder_here.png",        # no '/data/' -> untouched
        "/new/base/data/plate1/already_ok.png",     # already under base -> untouched
    ]})

    out_df, out_paths = correct_paths(df, base_path="/new/base", folder="data")

    assert out_paths == [
        "/new/base/data/plate1/A01_1_cell_1.png",
        "/new/base/data/plate1/A01_1_cell_2.png",
        "/old/root/misc/no_folder_here.png",
        "/new/base/data/plate1/already_ok.png",
    ]
    # the DataFrame is mutated in place with the same values
    assert out_df["png_path"].tolist() == out_paths


def test_correct_paths_list_input_returns_plain_list():
    from spacr.utils import correct_paths

    got = correct_paths(["/old/data/p1/x.png", "/old/nope/y.png"],
                        base_path="/mnt/base", folder="data")
    assert isinstance(got, list)
    assert got == ["/mnt/base/data/p1/x.png", "/old/nope/y.png"]


def test_correct_paths_honours_custom_folder_name():
    from spacr.utils import correct_paths

    got = correct_paths(["/old/root/pngs/p1/x.png"],
                        base_path="/mnt/base", folder="pngs")
    assert got == ["/mnt/base/pngs/p1/x.png"]


# ---------------------------------------------------------------------------
# delete_folder
# ---------------------------------------------------------------------------

def test_delete_folder_removes_nested_tree(tmp_path, capsys):
    from spacr.utils import delete_folder

    root = tmp_path / "victim"
    (root / "sub" / "deeper").mkdir(parents=True)
    (root / "top.txt").write_text("x")
    (root / "sub" / "mid.txt").write_text("y")
    (root / "sub" / "deeper" / "leaf.txt").write_text("z")

    assert delete_folder(str(root)) is None
    assert not root.exists()
    assert f"Folder '{root}' has been deleted." in capsys.readouterr().out


def test_delete_folder_missing_path_is_a_noop(tmp_path, capsys):
    from spacr.utils import delete_folder

    missing = tmp_path / "does_not_exist"
    delete_folder(str(missing))
    assert not missing.exists()
    assert "does not exist or is not a directory" in capsys.readouterr().out


def test_delete_folder_rejects_a_file(tmp_path, capsys):
    """An existing *file* takes the same negative branch and is NOT removed."""
    from spacr.utils import delete_folder

    f = tmp_path / "a_file.txt"
    f.write_text("keep me")
    delete_folder(str(f))
    assert f.exists() and f.read_text() == "keep me"
    assert "does not exist or is not a directory" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# measure_test_mode
# ---------------------------------------------------------------------------

def test_measure_test_mode_copies_subset_and_clears_stale_folder(tmp_path, capsys):
    from spacr.utils import measure_test_mode

    src = tmp_path / "merged"
    src.mkdir()
    for i in range(6):
        (src / f"plate1_A01_{i}.npy").write_text(str(i))

    # a stale test/merged tree that must be wiped by delete_folder first
    stale = tmp_path / "test" / "merged"
    (stale / "leftover_dir").mkdir(parents=True)
    (stale / "stale.npy").write_text("old")

    settings = {"src": str(src), "test_mode": True, "test_nr": 3}
    out = measure_test_mode(settings)

    expected = os.path.join(str(tmp_path), "test", "merged")
    assert out["src"] == expected
    copied = sorted(os.listdir(expected))
    assert len(copied) == 3
    assert "stale.npy" not in copied
    assert "leftover_dir" not in copied
    assert all(name.startswith("plate1_A01_") for name in copied)
    # originals untouched
    assert len(os.listdir(src)) == 6
    assert f"Changed source folder to {expected} for test mode" in capsys.readouterr().out


def test_measure_test_mode_src_already_test_folder(tmp_path, capsys):
    """basename == 'test' means the caller already pointed at a test folder."""
    from spacr.utils import measure_test_mode

    src = tmp_path / "test"
    src.mkdir()
    settings = {"src": str(src), "test_mode": True, "test_nr": 2}
    out = measure_test_mode(dict(settings))

    assert out["src"] == str(src)
    assert not (tmp_path / "test" / "merged").exists()
    assert f"Test mode enabled, using source folder {src}" in capsys.readouterr().out


def test_measure_test_mode_disabled_returns_settings_unchanged(tmp_path):
    from spacr.utils import measure_test_mode

    src = tmp_path / "merged"
    src.mkdir()
    settings = {"src": str(src), "test_mode": False, "test_nr": 2}
    out = measure_test_mode(settings)
    assert out["src"] == str(src)
    assert not (tmp_path / "test").exists()


# ---------------------------------------------------------------------------
# preprocess_data
# ---------------------------------------------------------------------------

def _channel_feature_df(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "nucleus_channel_1_mean_intensity": rng.normal(5.0, 1.0, n),
        "cell_channel_2_mean_intensity": rng.normal(20.0, 3.0, n),
        "cell_channel_3_mean_intensity": rng.normal(30.0, 4.0, n),
        "cell_area": rng.normal(500.0, 50.0, n),
        "object_id": np.arange(n),
        "pathogen_count": rng.integers(0, 4, n),
    })


def test_preprocess_data_filter_by_channel(capsys):
    """filter_by routes through filter_dataframe_features and keeps only the
    requested channel's features; output is z-scored."""
    from spacr.utils import preprocess_data

    df = _channel_feature_df()
    out = preprocess_data(df.copy(), filter_by=1, remove_highly_correlated=False,
                          log_data=False, exclude=None)

    assert isinstance(out, np.ndarray)
    assert out.shape == (60, 2)          # the two *_channel_1_* features
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-10)
    assert "Dropped 0 columns with NaN values" in capsys.readouterr().out


def test_preprocess_data_accepts_sqlite_numeric_text_measurements(capsys):
    """Image UMAP measurements stored as SQLite TEXT remain model features."""
    from spacr.utils import preprocess_data

    frame = pd.DataFrame({
        "cell_channel_0_mode_intensity": [
            str(value) for value in np.linspace(2.5, 25.0, 20)
        ],
        "object_label": np.arange(20),
    })

    result = preprocess_data(
        frame,
        filter_by="channel_0",
        remove_highly_correlated=False,
        log_data=False,
        exclude=None,
    )

    assert result.shape == (20, 1)
    assert np.isfinite(result).all()
    assert result.mean() == pytest.approx(0.0, abs=1e-12)
    assert "Dropped 0 columns with NaN values" in capsys.readouterr().out


def test_preprocess_data_column_list_subsets_first():
    """column_list narrows the frame before numeric selection."""
    from spacr.utils import preprocess_data

    df = _channel_feature_df()
    cols = ["cell_area", "cell_channel_2_mean_intensity"]
    out = preprocess_data(df.copy(), filter_by=None, remove_highly_correlated=False,
                          log_data=False, exclude=None, column_list=cols)

    assert out.shape == (60, 2)
    # column order is preserved, so column 0 must be the (scaled) cell_area
    expected0 = (df["cell_area"] - df["cell_area"].mean()) / df["cell_area"].std(ddof=0)
    assert np.allclose(out[:, 0], expected0.to_numpy())


def test_preprocess_data_raises_when_no_numeric_columns():
    from spacr.utils import preprocess_data

    df = pd.DataFrame({"well": ["A01", "A02", "A03"],
                       "prc": ["p1_A01_1", "p1_A02_1", "p1_A03_1"]})
    with pytest.raises(ValueError, match="No numeric columns available after filtering"):
        preprocess_data(df, filter_by=None, remove_highly_correlated=True,
                        log_data=False, exclude=None)


def test_preprocess_data_float_threshold_is_honoured():
    """A float `remove_highly_correlated` must be used verbatim -- a pair with
    r ~= 0.7 survives the default 0.95 cutoff but is pruned at 0.5."""
    from spacr.utils import preprocess_data

    rng = np.random.default_rng(11)
    a = rng.normal(0.0, 1.0, 200)
    b = 0.9 * a + rng.normal(0.0, 0.9, 200)
    c = rng.normal(0.0, 1.0, 200)
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    r = abs(np.corrcoef(a, b)[0, 1])
    assert 0.5 < r < 0.95, f"fixture correlation drifted: {r}"

    strict = preprocess_data(df.copy(), filter_by=None, remove_highly_correlated=0.5,
                             log_data=False, exclude=None)
    lenient = preprocess_data(df.copy(), filter_by=None, remove_highly_correlated=True,
                              log_data=False, exclude=None)

    assert strict.shape == (200, 2)
    assert lenient.shape == (200, 3)


def test_preprocess_data_log_transform_and_nan_fill():
    """log_data applies log(x + 1e-6); NaNs are mean-filled before scaling."""
    from spacr.utils import preprocess_data

    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "a": rng.uniform(1.0, 10.0, 30),
        "b": rng.uniform(1.0, 10.0, 30),
    })
    df.loc[0, "a"] = np.nan

    out = preprocess_data(df.copy(), filter_by=None, remove_highly_correlated=False,
                          log_data=True, exclude=None)

    assert out.shape == (30, 2)
    assert np.isfinite(out).all()
    # the imputed cell equals the column mean -> exactly 0 after standardising
    assert out[0, 0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# remove_low_variance_columns / remove_highly_correlated_columns
# ---------------------------------------------------------------------------

def test_remove_low_variance_columns_verbose_keeps_non_numeric(capsys):
    from spacr.utils import remove_low_variance_columns

    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "spread": rng.normal(0.0, 5.0, 40),
        "flat": np.full(40, 3.0),
        "well": ["A01"] * 40,
    })
    out = remove_low_variance_columns(df, threshold=0.01, verbose=True)

    assert list(out.columns) == ["spread", "well"]
    assert "Removed columns due to low variance: ['flat']" in capsys.readouterr().out


def test_remove_highly_correlated_columns_verbose(capsys):
    from spacr.utils import remove_highly_correlated_columns

    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 1.0, 50)
    df = pd.DataFrame({"a": base, "b": base * 3.0 + 1.0, "c": rng.normal(0, 1, 50)})
    out = remove_highly_correlated_columns(df, threshold=0.95, verbose=True)

    # 'b' is the later column in the upper triangle, so it is the one dropped
    assert list(out.columns) == ["a", "c"]
    assert "Removed columns due to high correlation: ['b']" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# filter_dataframe_features
# ---------------------------------------------------------------------------

def _self_pair_df(n=50, seed=13):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "cell_channel_2_mean_intensity": rng.normal(20.0, 3.0, n),
        "cell_channel_3_mean_intensity": rng.normal(30.0, 4.0, n),
        "cell_channel_4_mean_intensity": rng.normal(40.0, 5.0, n),
        "cell_area": rng.normal(500.0, 50.0, n),
        "object_id": np.arange(n),
        "pathogen_count": rng.integers(0, 4, n),
        "pathogen_pathogen": rng.normal(1.0, 0.5, n),
        "cell_cell": rng.normal(2.0, 0.5, n),
        "nucleus_nucleus": rng.normal(3.0, 0.5, n),
        "cytoplasm_cytoplasm": rng.normal(4.0, 0.5, n),
    })


def test_filter_dataframe_features_drops_self_measurement_columns(capsys):
    """The four <obj>_<obj> self-measurement columns are treated as id columns
    and removed alongside anything matching '_id' / 'count'."""
    from spacr.utils import filter_dataframe_features

    df = _self_pair_df()
    filtered, features = filter_dataframe_features(df.copy(), channel_of_interest=None,
                                                   verbose=True)

    for gone in ("pathogen_pathogen", "cell_cell", "nucleus_nucleus",
                 "cytoplasm_cytoplasm", "object_id", "pathogen_count"):
        assert gone not in filtered.columns
        assert gone not in features
    assert "cell_area" in features
    printed = capsys.readouterr().out
    assert "Columns to remove:" in printed
    assert "pathogen_pathogen" in printed
    assert "cytoplasm_cytoplasm" in printed


def test_filter_dataframe_features_channel_list(capsys):
    """A list of channels keeps every feature that mentions any of them."""
    from spacr.utils import filter_dataframe_features

    df = _self_pair_df()
    filtered, features = filter_dataframe_features(df.copy(), channel_of_interest=[1, 2],
                                                   verbose=True)

    assert sorted(features) == ["cell_channel_1_mean_intensity",
                                "cell_channel_2_mean_intensity"]
    assert filtered.shape == (50, 2)
    assert "Removed columns:" in capsys.readouterr().out


def test_filter_dataframe_features_channel_string():
    """A plain (non-'morphology') string is used as the feature substring."""
    from spacr.utils import filter_dataframe_features

    df = _self_pair_df()
    filtered, features = filter_dataframe_features(df.copy(),
                                                   channel_of_interest="channel_3")

    assert features == ["cell_channel_3_mean_intensity"]
    assert filtered.shape == (50, 1)


def test_filter_dataframe_features_exclude_str_and_list():
    from spacr.utils import filter_dataframe_features

    df = _self_pair_df()
    _, feats_list = filter_dataframe_features(df.copy(), channel_of_interest=None,
                                              exclude=["cell_area"])
    assert "cell_area" not in feats_list

    _, feats_str = filter_dataframe_features(df.copy(), channel_of_interest=None,
                                             exclude="cell_area")
    assert "cell_area" not in feats_str


def test_filter_dataframe_features_drops_nan_columns(capsys):
    from spacr.utils import filter_dataframe_features

    df = _self_pair_df()
    df["cell_channel_1_std_intensity"] = np.nan
    filtered, features = filter_dataframe_features(df, channel_of_interest="channel_1",
                                                   remove_low_variance_features=False,
                                                   remove_highly_correlated_features=False)

    assert "cell_channel_1_std_intensity" not in features
    assert features == ["cell_channel_1_mean_intensity"]
    assert "Dropped 1 columns with NaN values" in capsys.readouterr().out


def test_filter_dataframe_features_morphology():
    """'morphology' selects the shape descriptors and ignores channel columns."""
    from spacr.utils import filter_dataframe_features

    rng = np.random.default_rng(21)
    n = 40
    df = pd.DataFrame({
        "cell_area": rng.normal(500.0, 50.0, n),
        "cell_perimeter": rng.normal(90.0, 9.0, n),
        "cell_solidity": rng.uniform(0.5, 0.99, n),
        "cell_channel_1_mean_intensity": rng.normal(10.0, 2.0, n),
        "object_id": np.arange(n),
    })
    filtered, features = filter_dataframe_features(df, channel_of_interest="morphology")

    assert "cell_channel_1_mean_intensity" not in features
    assert set(features) == {"cell_area", "cell_perimeter", "cell_solidity"}
    assert filtered.shape == (n, 3)


# ---------------------------------------------------------------------------
# check_overlap / find_non_overlapping_position
# ---------------------------------------------------------------------------

def test_find_non_overlapping_position_stays_within_jitter_range():
    """A free neighbourhood yields a moved point, still inside the +/-10 box."""
    from spacr.utils import check_overlap, find_non_overlapping_position

    occupied = [(100.0, 100.0)]
    assert check_overlap((0.0, 0.0), occupied, threshold=5.0) is False

    x, y = find_non_overlapping_position(0.0, 0.0, occupied, threshold=5.0,
                                         max_attempts=10)
    assert (x, y) != (0.0, 0.0)
    assert abs(x) <= 10.0 and abs(y) <= 10.0
    assert check_overlap((x, y), occupied, threshold=5.0) is False


def test_find_non_overlapping_position_gives_up_and_returns_original():
    """The jitter is bounded to +/-10, so a threshold larger than that can
    never be satisfied and the original point comes back unchanged."""
    from spacr.utils import check_overlap, find_non_overlapping_position

    occupied = [(0.0, 0.0)]
    x, y = find_non_overlapping_position(0.0, 0.0, occupied, threshold=1000.0,
                                         max_attempts=5)
    assert (x, y) == (0.0, 0.0)
    assert check_overlap((x, y), occupied, threshold=1000.0) is True
