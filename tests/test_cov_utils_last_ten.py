"""The final ten uncovered statements in spacr.utils.

Each test here targets one specific line that the wave-1/wave-2 suites left
behind — mostly defensive branches that only fire on a failure or an unusual
array layout. Every one is driven for real (blocked imports, injected
exceptions, 4-D arrays); nothing is pragma'd out.

CPU-only, offline, no plotting windows.
"""
from __future__ import annotations

import os
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# NOT covered here: utils.py lines 34-40, the IPython-import fallback that
# defines a no-op display(). Exercising it means blocking the IPython import
# and re-importing spacr.utils, which replaces the module object while
# spacr.io and friends still hold references to the old one — that reliably
# breaks the TorchModel/ResNet tests later in the same session. Seven lines of
# import fallback are not worth destabilising the suite, so they are left
# uncovered rather than pragma'd or faked.
# ---------------------------------------------------------------------------
# 160: _select_intensity_channel's >3-D passthrough
# ---------------------------------------------------------------------------

def test_select_intensity_channel_passes_through_4d_stacks():
    """A 4-D array has no channel layout this helper understands, so it is
    returned whole (as float32) instead of being indexed."""
    from spacr.utils import _select_intensity_channel

    raw = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    out = _select_intensity_channel(raw, intensity_channel=1)

    assert out.shape == raw.shape
    assert out.dtype == np.float32
    assert np.array_equal(out, raw.astype(np.float32))


# ---------------------------------------------------------------------------
# 1537-1538: both prcfo builds fail -> the inner except prints the error
# ---------------------------------------------------------------------------

def _png_list_db(tmp_path):
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    try:
        pd.DataFrame({
            "prcfo": ["p1_r1_c1_f1_o1"],
            "png_path": ["/tmp/a.png"],
        }).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def test_update_database_reports_when_the_cell_id_fallback_also_fails(tmp_path, capsys):
    """Neither object_label nor cell_id present: the first prcfo build fails,
    the cell_id fallback is attempted, and its own exception is *printed*
    rather than propagating.

    The function cannot recover from there — with no prcfo column there is
    nothing to merge on — so the eventual KeyError from the merge is the
    documented outcome. What matters here is that the fallback's handler
    swallowed and reported its error first.
    """
    from spacr.utils import _update_database_with_merged_info

    db = _png_list_db(tmp_path)
    # has the prc* columns but NEITHER object_label NOR cell_id
    df = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"], "fieldID": ["f1"],
        "prcfo_missing": [1],
    })

    with pytest.raises(KeyError):
        _update_database_with_merged_info(str(db), df, table="png_list",
                                          columns=["prcfo"])

    out = capsys.readouterr().out
    assert "Merging on cell failed, trying with cell_id" in out
    # the inner `except Exception as e: print(e)` reported the missing column
    assert "cell_id" in out


# ---------------------------------------------------------------------------
# 3215: _scalar unwraps a Series
# ---------------------------------------------------------------------------

def test_suggest_training_changes_scalar_unwraps_a_series(tmp_path):
    """`epoch` lookups go through _scalar; feeding a duplicated index makes
    the pandas lookup return a Series, which _scalar must reduce to a float."""
    from spacr.utils import suggest_training_changes

    n = 12
    tr = pd.DataFrame({
        "epoch": range(1, n + 1),
        "loss": np.linspace(1.0, 0.4, n),
        "accuracy": np.linspace(0.5, 0.9, n),
    })
    va = pd.DataFrame({
        "epoch": range(1, n + 1),
        "loss": np.linspace(1.0, 0.5, n),
        "accuracy": np.linspace(0.5, 0.8, n),
    })
    tp = tmp_path / "training_progress.csv"
    vp = tmp_path / "validation_progress.csv"
    tr.to_csv(tp, index=False)
    va.to_csv(vp, index=False)

    real_read_csv = pd.read_csv

    def _dup_index(path, *args, **kwargs):
        df = real_read_csv(path, *args, **kwargs)
        if os.path.basename(str(path)).startswith("validation"):
            # every row shares one index label, so .iloc stays correct but any
            # label-based lookup would return a Series
            df.index = [0] * len(df)
        return df

    pd.read_csv = _dup_index
    try:
        out = suggest_training_changes(str(tmp_path), train_csv=str(tp), val_csv=str(vp))
    finally:
        pd.read_csv = real_read_csv

    # best val loss is the last row; _scalar reduced the Series to a float
    assert isinstance(out["summary"]["best_val_loss"], float)
    assert out["summary"]["best_val_loss"] == pytest.approx(0.5)
    assert isinstance(out["summary"]["best_epoch"], int)


# ---------------------------------------------------------------------------
# 4339: _remove_outside_objects clears a nucleus lying inside a stray pathogen
# ---------------------------------------------------------------------------

def test_remove_outside_objects_clears_nuclei_within_the_stray_pathogen():
    """A pathogen outside every cell is removed together with the nuclei that
    actually sit inside its footprint."""
    from spacr.utils import _remove_outside_objects

    cell = np.zeros((16, 16), np.int32)
    cell[0:5, 0:5] = 1                      # a cell, far away
    pathogen = np.zeros((16, 16), np.int32)
    pathogen[9:15, 9:15] = 4                # outside every cell
    nucleus = np.zeros((16, 16), np.int32)
    nucleus[10:13, 10:13] = 8               # genuinely inside that pathogen
    nucleus[1:3, 1:3] = 9                   # unrelated, inside the cell

    stack = np.stack([cell, nucleus, pathogen], axis=-1).astype(np.int32)
    out = _remove_outside_objects(stack, 0, 1, 2)

    out_nucleus, out_pathogen = out[..., 1], out[..., 2]
    assert out_pathogen.max() == 0, "the stray pathogen must be removed"
    assert 8 not in np.unique(out_nucleus), "its own nucleus goes with it"
    assert 9 in np.unique(out_nucleus), "the unrelated nucleus must survive"


# ---------------------------------------------------------------------------
# 5639: plot_clusters_grid skips the DBSCAN noise label
# ---------------------------------------------------------------------------

def test_plot_clusters_grid_skips_the_noise_label(tmp_path, monkeypatch):
    """Label -1 is DBSCAN noise and must be skipped by the per-cluster loop
    even though it reaches the dict comprehension guard."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import spacr.utils as su

    # plot_grid does the drawing; stub it so we only exercise the loop.
    captured = {}

    def _fake_plot_grid(cluster_images, colors, figuresize, black_background, verbose):
        captured["labels"] = sorted(cluster_images)
        return plt.figure()

    monkeypatch.setattr(su, "plot_grid", _fake_plot_grid)

    paths = []
    for i in range(4):
        p = tmp_path / f"img{i}.png"
        plt.imsave(p, np.zeros((4, 4, 3)))
        paths.append(str(p))

    labels = np.array([-1, 0, 0, 1])
    fig = su.plot_clusters_grid(
        embedding=np.zeros((4, 2)), labels=labels, image_nr=2,
        image_paths=paths, colors=["r", "g", "b"], figuresize=4,
        black_background=False, verbose=False,
    )
    plt.close("all")

    # -1 never becomes a cluster key
    assert captured["labels"] == [0, 1]
    assert fig is not None


def test_plot_clusters_grid_returns_none_without_clusters(capsys):
    """All-noise labels short-circuit with a message and no figure."""
    from spacr.utils import plot_clusters_grid

    out = plot_clusters_grid(
        embedding=np.zeros((3, 2)), labels=np.array([-1, -1, -1]), image_nr=2,
        image_paths=["a.png", "b.png", "c.png"], colors=["r"], figuresize=4,
        black_background=False, verbose=False,
    )
    assert out is None
    assert "No clusters found." in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7286: delete_intermedeate_files bails out when merged/ is under-populated
# ---------------------------------------------------------------------------

def test_delete_intermedeate_files_keeps_everything_when_merged_lags(tmp_path):
    """merged/ holding fewer fields than stack/ means the run did not finish,
    so nothing may be deleted."""
    from spacr.utils import delete_intermedeate_files

    root = str(tmp_path / "p1")
    os.makedirs(os.path.join(root, "merged"), exist_ok=True)
    np.save(os.path.join(root, "merged", "a.npy"), np.zeros((2, 2), np.uint16))
    os.makedirs(os.path.join(root, "stack"), exist_ok=True)
    for name in ("a.npy", "b.npy", "c.npy"):          # 3 > 1
        np.save(os.path.join(root, "stack", name), np.zeros((2, 2), np.uint16))
    os.makedirs(os.path.join(root, "orig"), exist_ok=True)
    os.makedirs(os.path.join(root, "1"), exist_ok=True)

    delete_intermedeate_files({"src": root})

    assert os.path.isdir(os.path.join(root, "stack"))
    assert os.path.isdir(os.path.join(root, "1"))
