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

def _fill(dirpath, n, prefix):
    """Create ``n`` dummy image files in ``dirpath`` and return their names."""
    os.makedirs(dirpath, exist_ok=True)
    names = []
    for i in range(n):
        name = f"{prefix}_{i:02d}.png"
        with open(os.path.join(dirpath, name), "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        names.append(name)
    return names


def test_augment_classes_generate_only_creates_output_dirs(tmp_path):
    """generate=True/move=False only makes the two aug_* folders and returns None."""
    from spacr.utils import augment_classes
    dst = str(tmp_path)
    out = augment_classes(dst, nc=["/nope/a.png"], pc=["/nope/b.png"],
                          generate=True, move=False)
    assert out is None
    assert os.path.isdir(os.path.join(dst, "aug_nc"))
    assert os.path.isdir(os.path.join(dst, "aug_pc"))
    # The augmentation itself is guarded by `if __name__ == '__main__'`, so
    # nothing was written.
    assert os.listdir(os.path.join(dst, "aug_nc")) == []
    assert os.listdir(os.path.join(dst, "aug_pc")) == []
    # move=False -> no train/test tree at all.
    assert not os.path.exists(os.path.join(dst, "aug"))


def test_augment_classes_runs_augment_images_when_run_as_main(tmp_path, monkeypatch):
    """Under ``__name__ == '__main__'`` both classes are handed to augment_images."""
    import spacr.utils as U
    calls = []
    monkeypatch.setattr(U, "augment_images",
                        lambda file_paths, dst: calls.append((list(file_paths), dst)))
    # augment_classes reads the module-level __name__ at call time.
    monkeypatch.setattr(U, "__name__", "__main__")

    dst = str(tmp_path)
    nc = ["/src/nc1.png", "/src/nc2.png"]
    pc = ["/src/pc1.png"]
    U.augment_classes(dst, nc=nc, pc=pc, generate=True, move=False)

    assert calls == [
        (nc, os.path.join(dst, "aug_nc")),
        (pc, os.path.join(dst, "aug_pc")),
    ]


def test_augment_classes_moves_augmented_images_into_train_test(tmp_path):
    """move=True splits aug_nc/aug_pc 90/10 into aug/{train,test}/{nc,pc}."""
    from spacr.utils import augment_classes
    dst = str(tmp_path)
    nc_names = _fill(os.path.join(dst, "aug_nc"), 20, "nc")
    pc_names = _fill(os.path.join(dst, "aug_pc"), 10, "pc")

    augment_classes(dst, nc=["x"] * 20, pc=["y"] * 10, generate=False, move=True)

    aug = os.path.join(dst, "aug")
    train_nc = sorted(os.listdir(os.path.join(aug, "train", "nc")))
    test_nc = sorted(os.listdir(os.path.join(aug, "test", "nc")))
    train_pc = sorted(os.listdir(os.path.join(aug, "train", "pc")))
    test_pc = sorted(os.listdir(os.path.join(aug, "test", "pc")))

    # test_size=0.1 -> ceil(0.1 * n) go to test, the rest to train.
    assert (len(train_nc), len(test_nc)) == (18, 2)
    assert (len(train_pc), len(test_pc)) == (9, 1)
    # Nothing lost, nothing duplicated, and the sources were emptied.
    assert sorted(train_nc + test_nc) == sorted(nc_names)
    assert sorted(train_pc + test_pc) == sorted(pc_names)
    assert os.listdir(os.path.join(dst, "aug_nc")) == []
    assert os.listdir(os.path.join(dst, "aug_pc")) == []


def test_augment_classes_summary_line_labels_are_correct(tmp_path, capsys):
    """The printed summary must report each split under its own label."""
    from spacr.utils import augment_classes
    dst = str(tmp_path)
    _fill(os.path.join(dst, "aug_nc"), 20, "nc")
    _fill(os.path.join(dst, "aug_pc"), 10, "pc")

    augment_classes(dst, nc=["x"] * 20, pc=["y"] * 10, generate=False, move=True)

    out = capsys.readouterr().out
    # train nc=18, train pc=9, test nc=2, test pc=1
    assert "Train nc: 18, Train pc:9, Test nc:2, Test pc:1" in out


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

def _mlr_df(n=150):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "gene": rng.choice(["g1", "g2", "g3"], n),
        "grna": rng.choice(["r1", "r2"], n),
        "plate": rng.choice(["p1", "p2"], n),
        "row": rng.choice(["A", "B"], n),
        "column": rng.choice(["1", "2"], n),
    })
    pred = rng.normal(0.0, 0.25, n)
    pred += (df["gene"] == "g2") * (df["grna"] == "r2") * 1.5
    pred -= (df["gene"] == "g3") * (df["grna"] == "r2") * 0.8
    df["pred"] = pred
    # Two deliberate outliers so the residual / Cook's-distance filters have
    # something to find.
    df.loc[0, "pred"] = 25.0
    df.loc[1, "pred"] = -25.0
    return df


@pytest.fixture
def reg_v_plot_spy(monkeypatch):
    """Replace the volcano plot with a recorder (its real signature differs)."""
    import spacr.plot as P
    seen = []
    monkeypatch.setattr(P, "_reg_v_plot", lambda *a, **k: seen.append((a, k)))
    return seen


def test_MLR_refined_model_returns_max_interaction_effects(reg_v_plot_spy, capsys):
    from spacr.utils import MLR
    df = _mlr_df()
    max_effects, max_pvalues, model, out = MLR(df, refine_model=True)

    # With main effects present patsy names the interactions gene[T.x]:grna[T.y],
    # which is what MLR filters on.
    assert set(max_effects) == {"g2", "g3"}
    assert set(max_pvalues) == set(max_effects)
    for gene in max_effects:
        key = f"gene[T.{gene}]:grna[T.r2]"
        assert max_effects[gene] == pytest.approx(model.params[key])
        assert max_pvalues[gene] == pytest.approx(model.pvalues[key])
    # g2 was simulated with a strong positive interaction, g3 a negative one.
    assert max_effects["g2"] > 0 > max_effects["g3"]

    assert list(out.columns) == ["effect", "p"]
    assert list(out.index) == ["g2", "g3"]          # sorted by effect, descending
    assert out["effect"].is_monotonic_decreasing
    assert out.loc["g2", "p"] < 0.05

    # The refit dropped the two planted outliers.
    assert int(model.nobs) < len(df)
    printed = capsys.readouterr().out
    assert "Number of outliers detected by standardized residuals:" in printed
    assert "Durbin-Watson" in printed
    # the volcano plot got the returned frame
    assert len(reg_v_plot_spy) == 1
    pd.testing.assert_frame_equal(reg_v_plot_spy[0][0][0], out)


def test_MLR_unrefined_model_reports_interaction_effects(reg_v_plot_spy):
    from spacr.utils import MLR
    df = _mlr_df()
    max_effects, max_pvalues, model, out = MLR(df, refine_model=False)

    assert int(model.nobs) == len(df)               # no rows dropped
    assert len(max_effects) > 0
    assert set(out.columns) == {"effect", "p"}


def test_MLR_calls_reg_v_plot_with_a_usable_signature():
    from spacr.utils import MLR
    result = MLR(_mlr_df(), refine_model=False)
    assert len(result) == 4
