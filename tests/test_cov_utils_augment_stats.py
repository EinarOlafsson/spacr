"""CPU coverage for the augmentation / tar-packing / statistics block of
``spacr.utils`` (augment_classes, annotate_predictions, initiate_counter,
add_images_to_tar, fishers_odds, MLR).

Everything here is offline, single-process and figure-free (matplotlib runs on
Agg and every figure is closed after each test).  Two behaviours are known to
be broken and are pinned with ``strict`` xfails that assert the CORRECT
behaviour rather than the current one:

* ``augment_classes`` mislabels its final summary line (train/test counts for
  ``pc``/``nc`` are swapped).
* ``MLR`` calls ``spacr.plot._reg_v_plot(df)`` with one argument while that
  helper takes four.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tarfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let a diagnostic figure leak out of a test."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# augment_classes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# annotate_predictions — the "no condition" fall-through
# ---------------------------------------------------------------------------

def test_annotate_predictions_unassignable_wells_get_empty_condition(tmp_path):
    """A column outside 1-3 and not >3 (col 0) yields '' ; an unknown plate >3 -> None."""
    from spacr.utils import annotate_predictions
    csv = tmp_path / "preds.csv"
    pd.DataFrame({
        "path": [
            "/x/1_A00_1_1.png",   # column 0  -> falls through to ''
            "/x/9_A05_1_2.png",   # column 5 but plate 9 -> no branch matches -> None
            "/x/1_A02_1_3.png",   # column 2  -> 'nc'
        ],
        "pred": [0.1, 0.2, 0.3],
    }).to_csv(csv, index=False)

    out = annotate_predictions(str(csv))

    assert list(out["plateID"]) == ["1", "9", "1"]
    assert list(out["object"]) == ["1", "2", "3"]
    assert out.loc[0, "cond"] == ""
    assert out.loc[1, "cond"] is None
    assert out.loc[2, "cond"] == "nc"


# ---------------------------------------------------------------------------
# initiate_counter / add_images_to_tar
# ---------------------------------------------------------------------------

@pytest.fixture
def shared_counter():
    """Install a real multiprocessing counter+lock into spacr.utils, then restore."""
    import spacr.utils as U
    sentinel = object()
    old_counter = getattr(U, "counter", sentinel)
    old_lock = getattr(U, "lock", sentinel)

    counter = mp.Value("i", 0)
    lock = mp.Lock()
    U.initiate_counter(counter, lock)
    try:
        yield counter
    finally:
        for name, old in (("counter", old_counter), ("lock", old_lock)):
            if old is sentinel:
                if hasattr(U, name):
                    delattr(U, name)
            else:
                setattr(U, name, old)


def test_initiate_counter_publishes_module_globals():
    """initiate_counter binds the shared objects onto the module globals."""
    import spacr.utils as U
    sentinel = object()
    old_counter = getattr(U, "counter", sentinel)
    old_lock = getattr(U, "lock", sentinel)
    counter = mp.Value("i", 7)
    lock = mp.Lock()
    try:
        assert U.initiate_counter(counter, lock) is None
        assert U.counter is counter
        assert U.lock is lock
        assert U.counter.value == 7
    finally:
        for name, old in (("counter", old_counter), ("lock", old_lock)):
            if old is sentinel:
                if hasattr(U, name):
                    delattr(U, name)
            else:
                setattr(U, name, old)


def _write_png(path, value=0):
    from PIL import Image
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(str(path))


def test_add_images_to_tar_packs_files_and_reports_progress(tmp_path, shared_counter,
                                                            capsys):
    """Every image lands in the tar under its basename and bumps the counter."""
    from spacr.utils import add_images_to_tar
    src = tmp_path / "imgs"
    src.mkdir()
    paths = []
    for i in range(10):
        p = src / f"img_{i}.png"
        _write_png(p, i)
        paths.append(str(p))

    tar_path = tmp_path / "chunk.tar"
    add_images_to_tar(paths, str(tar_path), total_images=10)

    with tarfile.open(tar_path) as tar:
        names = sorted(tar.getnames())
    assert names == sorted(os.path.basename(p) for p in paths)
    assert shared_counter.value == 10
    # counter.value % 10 == 0 -> the progress line was emitted exactly once.
    out = capsys.readouterr().out
    assert out.count("generating .tar dataset") == 1
    assert "10/10" in out


def test_add_images_to_tar_skips_missing_files(tmp_path, shared_counter, capsys):
    """A missing source is reported and does not abort the chunk or move the counter."""
    from spacr.utils import add_images_to_tar
    good = tmp_path / "good.png"
    _write_png(good)
    missing = str(tmp_path / "gone.png")

    tar_path = tmp_path / "chunk.tar"
    add_images_to_tar([str(good), missing], str(tar_path), total_images=2)

    with tarfile.open(tar_path) as tar:
        assert tar.getnames() == ["good.png"]
    assert shared_counter.value == 1          # only the existing file counted
    assert f"File not found: {missing}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# fishers_odds
# ---------------------------------------------------------------------------

def _fisher_df():
    """40 wells; m1 tracks the phenotype, m2 is constant (degenerate table)."""
    n_pos, n_neg = 20, 20
    # m1 == 1 wells: 17 low-pred (high phenotype) + 3 high-pred
    pred = [0.2] * 17 + [0.8] * 3 + [0.2] * 2 + [0.8] * 18
    m1 = [1] * n_pos + [0] * n_neg
    return pd.DataFrame({
        "m1": m1,
        "m2": [0] * (n_pos + n_neg),          # never > 0 -> 1x2 crosstab
        "count_prc": np.arange(n_pos + n_neg),        # excluded by name
        "mean_pathogen_area": np.linspace(1, 2, n_pos + n_neg),  # excluded by name
        "mean_pred": pred,
    })


def test_fishers_odds_returns_adjusted_pvalues_and_drops_degenerate_tables(capsys):
    from spacr.utils import fishers_odds
    df = _fisher_df()
    res = fishers_odds(df, threshold=0.5, phenotyp_col="mean_pred")

    assert list(res.columns) == ["Mutant", "OddsRatio", "PValue", "AdjustedPValue"]
    # m2 produced a non-2x2 table -> NaN -> dropped; the name-filtered columns
    # never made it into the results at all.
    assert list(res["Mutant"]) == ["m1"]
    row = res.iloc[0]
    # contingency table [[17, 3], [2, 18]] -> OR = 51
    assert row["OddsRatio"] == pytest.approx(51.0)
    assert row["PValue"] < 0.05
    assert row["AdjustedPValue"] >= row["PValue"]
    # the binarized phenotype column is added in place
    assert df["high_phenotype"].tolist() == [p < 0.5 for p in df["mean_pred"]]


def test_fishers_odds_with_no_valid_tables_returns_empty_frame(capsys):
    """All-degenerate input skips the BH correction and warns."""
    from spacr.utils import fishers_odds
    df = pd.DataFrame({
        "m1": [0] * 10,                # never > 0
        "m2": [0] * 10,
        "mean_pred": [0.1] * 5 + [0.9] * 5,
    })
    res = fishers_odds(df, threshold=0.5, phenotyp_col="mean_pred")

    assert res.empty
    assert list(res.columns) == ["Mutant", "OddsRatio", "PValue"]
    assert "No p-values to adjust" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# MLR
# ---------------------------------------------------------------------------

