"""CPU coverage for spacr.utils: the training-diagnostics advisor, pipeline
folder cleanup, DB column helpers, mask utilities and metadata correction.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


# ---------------------------------------------------------------------------
# suggest_training_changes
# ---------------------------------------------------------------------------

def _progress_csvs(dst, n=40, overfit=False, plateau=False, noisy=False,
                   rng=None):
    """Write train/validation progress CSVs in the layout _save_progress uses."""
    rng = rng or np.random.default_rng(0)
    os.makedirs(dst, exist_ok=True)
    epochs = np.arange(1, n + 1)
    train_acc = np.linspace(0.6, 0.98, n)
    if plateau:
        val_acc = np.full(n, 0.80)
        val_loss = np.full(n, 0.50)
    elif overfit:
        val_acc = np.linspace(0.60, 0.62, n)      # big train-val gap
        val_loss = np.linspace(0.6, 0.9, n)       # rising
    else:
        val_acc = np.linspace(0.58, 0.95, n)
        val_loss = np.linspace(0.7, 0.2, n)
    if noisy:
        val_loss = val_loss + rng.normal(0, 0.25, n)

    common = dict(neg_accuracy=val_acc, pos_accuracy=val_acc,
                  prauc=val_acc, optimal_threshold=np.full(n, 0.5))
    tr = pd.DataFrame(dict(epoch=epochs, accuracy=train_acc,
                           loss=np.linspace(0.7, 0.05, n), **common))
    va = pd.DataFrame(dict(epoch=epochs, accuracy=val_acc,
                           loss=val_loss, **common))
    tr.to_csv(os.path.join(dst, "train_progress.csv"), index=False)
    va.to_csv(os.path.join(dst, "validation_progress.csv"), index=False)
    return (os.path.join(dst, "train_progress.csv"),
            os.path.join(dst, "validation_progress.csv"))


def test_suggest_training_changes_healthy_run(tmp_path):
    from spacr.utils import suggest_training_changes
    tr, va = _progress_csvs(str(tmp_path))
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    assert set(("summary", "flags", "suggestions")) <= set(out)
    assert isinstance(out["suggestions"], list)


def test_suggest_training_changes_overfitting(tmp_path):
    from spacr.utils import suggest_training_changes
    tr, va = _progress_csvs(str(tmp_path), overfit=True)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    text = " ".join(out["flags"]) + " " + " ".join(out["suggestions"])
    assert text.strip(), "overfitting run produced no guidance"


def test_suggest_training_changes_plateau(tmp_path):
    from spacr.utils import suggest_training_changes
    tr, va = _progress_csvs(str(tmp_path), plateau=True)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    assert out["summary"] is not None


def test_suggest_training_changes_noisy(tmp_path):
    from spacr.utils import suggest_training_changes
    tr, va = _progress_csvs(str(tmp_path), noisy=True)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va)
    assert isinstance(out["flags"], list)


def test_suggest_training_changes_autodetects_csvs(tmp_path):
    """No explicit paths → the advisor globs them out of dst."""
    from spacr.utils import suggest_training_changes
    _progress_csvs(str(tmp_path))
    out = suggest_training_changes(str(tmp_path))
    assert "summary" in out


def test_suggest_training_changes_too_few_epochs(tmp_path):
    from spacr.utils import suggest_training_changes
    tr, va = _progress_csvs(str(tmp_path), n=4)
    out = suggest_training_changes(str(tmp_path), train_csv=tr, val_csv=va,
                                   min_epochs=10)
    assert "summary" in out


def test_suggest_training_changes_missing_csvs(tmp_path):
    from spacr.utils import suggest_training_changes
    # Unguarded, and this is the whole point of the test: an advisor pointed at
    # a directory with no progress CSVs must SAY so, not raise. The skip made
    # "handled the empty case" and "blew up on the empty case" the same result.
    out = suggest_training_changes(str(tmp_path))
    assert out is None or isinstance(out, dict)


# ---------------------------------------------------------------------------
# cleanup_pipeline_folders
# ---------------------------------------------------------------------------

def _pipeline_tree(root):
    for sub in ("stack", "masks", "orig", "merged"):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "f0.npy").write_bytes(b"x")
    return root


def test_cleanup_removes_intermediates_by_default(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    src = _pipeline_tree(tmp_path / "plate1")
    cleanup_pipeline_folders(str(src), keep_intermediate=False,
                             keep_original=False, verbose=True)
    assert (src / "merged").is_dir()
    assert not (src / "stack").exists()
    assert not (src / "masks").exists()


def test_cleanup_keeps_intermediates_when_asked(tmp_path):
    from spacr.utils import cleanup_pipeline_folders
    src = _pipeline_tree(tmp_path / "plate1")
    cleanup_pipeline_folders(str(src), keep_intermediate=True,
                             keep_original=True, verbose=False)
    assert (src / "stack").is_dir() and (src / "masks").is_dir()
    assert (src / "orig").is_dir()


def test_cleanup_refuses_without_merged(tmp_path):
    """Guard: never delete intermediates when merged/ is missing or empty."""
    from spacr.utils import cleanup_pipeline_folders
    src = _pipeline_tree(tmp_path / "plate1")
    for f in (src / "merged").iterdir():
        f.unlink()
    cleanup_pipeline_folders(str(src), keep_intermediate=False,
                             keep_original=False, verbose=False)
    assert (src / "stack").is_dir(), "intermediates deleted without merged/"


# ---------------------------------------------------------------------------
# database helpers
# ---------------------------------------------------------------------------

def test_add_column_to_database(tmp_path):
    from spacr.utils import add_column_to_database
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE png_list (prcfo TEXT, v INT)")
    con.executemany("INSERT INTO png_list VALUES (?,?)",
                    [("a", 1), ("b", 2)])
    con.commit(); con.close()
    csv = tmp_path / "new.csv"
    pd.DataFrame({"prcfo": ["a", "b"], "score": [0.1, 0.9]}).to_csv(csv, index=False)
    settings = {"csv_path": str(csv), "db_path": str(db),
                "table_name": "png_list", "update_column": "score",
                "match_column": "prcfo"}
    add_column_to_database(settings)
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    score_col = next(c for c in cols if "score" in c)
    rows = dict(con.execute(
        f'SELECT prcfo, "{score_col}" FROM png_list').fetchall())
    con.close()
    # the CSV values must actually land on the matching rows, not just the column
    assert rows == {"a": 0.1, "b": 0.9}


def test_correct_metadata_renames_columns():
    from spacr.utils import correct_metadata
    df = pd.DataFrame({"plate_name": ["p1"], "column_name": ["c1"],
                       "row_name": ["r1"], "v": [1]})
    out = correct_metadata(df)
    assert "plateID" in out.columns and "columnID" in out.columns
    assert "rowID" in out.columns


def test_map_condition():
    from spacr.utils import map_condition
    neg_lbl = map_condition("c1", neg="c1", pos="c2", mix="c3")
    pos_lbl = map_condition("c2", neg="c1", pos="c2", mix="c3")
    mix_lbl = map_condition("c3", neg="c1", pos="c2", mix="c3")
    other = map_condition("c9", neg="c1", pos="c2", mix="c3")
    # three control classes map to three distinct labels; anything else differs
    assert len({neg_lbl, pos_lbl, mix_lbl}) == 3
    assert other not in (neg_lbl, pos_lbl, mix_lbl)


# ---------------------------------------------------------------------------
# mask utilities
# ---------------------------------------------------------------------------

def test_generate_cytoplasm_mask():
    from spacr.utils import generate_cytoplasm_mask
    cell = np.zeros((20, 20), np.uint16); cell[3:17, 3:17] = 1
    nuc = np.zeros((20, 20), np.uint16); nuc[8:12, 8:12] = 1
    cyto = generate_cytoplasm_mask(nuc, cell)
    assert cyto is not None
    assert cyto[10, 10] == 0        # nucleus excluded
    assert cyto[4, 4] != 0          # rim retained


def test_fill_holes_in_mask():
    from spacr.utils import fill_holes_in_mask
    m = np.zeros((20, 20), np.uint16)
    m[3:17, 3:17] = 1
    m[9:11, 9:11] = 0               # punch a hole
    out = fill_holes_in_mask(m)
    assert out[10, 10] != 0, "hole not filled"


def test_generate_mask_random_cmap():
    from spacr.utils import _generate_mask_random_cmap
    m = np.zeros((10, 10), np.uint16); m[1:4, 1:4] = 1; m[6:9, 6:9] = 2
    cmap = _generate_mask_random_cmap(m)
    assert cmap is not None


def test_remove_outliers_by_group():
    from spacr.utils import remove_outliers_by_group
    df = pd.DataFrame({
        "g": ["a"] * 10 + ["b"] * 10,
        "v": list(np.arange(10)) + list(np.arange(10)),
    })
    df.loc[0, "v"] = 1000            # outlier in group a
    out = remove_outliers_by_group(df, group_col="g", value_col="v",
                                   method="iqr", threshold=1.5)
    assert len(out) < len(df)


def test_get_regex_variants():
    from spacr.utils import _get_regex
    for mt in ("cellvoyager", "cq1", "auto"):
        rx = _get_regex(mt, ".tif", None)
        assert isinstance(rx, str) and rx
    custom = _get_regex("custom", ".tif", r"(?P<wellID>\w+)")
    # the custom pattern is embedded (with an extension suffix appended)
    assert r"(?P<wellID>\w+)" in custom


def test_load_settings_roundtrip(tmp_path):
    from spacr.utils import load_settings
    csv = tmp_path / "s.csv"
    pd.DataFrame({"setting_key": ["a", "b"],
                  "setting_value": ["1", "True"]}).to_csv(csv, index=False)
    out = load_settings(str(csv), show=True)
    assert isinstance(out, dict) and "a" in out
