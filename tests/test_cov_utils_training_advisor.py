"""CPU coverage for ``spacr.utils.suggest_training_changes`` — the training
advisor that reads the train/validation progress CSVs written by
``_save_progress`` and turns them into flags + concrete suggestions.

The tests here drive the branches that a healthy run never reaches:

* the two early bail-outs (no val CSV at all / an explicit path that does not
  exist, and progress CSVs missing the required ``epoch`` / ``loss`` columns);
* the NaN guard inside the nested ``_poly_slope`` helper (a recent-loss window
  with fewer than two finite points must yield slope 0.0 instead of blowing up
  in ``np.polyfit``);
* the macro-F1 plumbing (``f1_macro`` / ``macro_f1`` columns reaching the
  summary, and the ``f1_nan_detected`` flag when a split has NaN F1);
* the column normalisation/de-duplication path (aliases, case folding, two
  original names collapsing onto one canonical name).

One behaviour is known to be broken and is pinned with a ``strict`` xfail that
asserts the CORRECT behaviour: the nested ``_scalar`` helper exists to survive
a non-unique index (``va.loc[best_val_idx, "loss"]`` coming back as a Series),
but the very next statement calls ``int()`` on ``va.loc[best_val_idx,
"epoch"]`` without the same guard, so the defence can never actually pay off.

Everything is offline, single-process and figure-free.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """The advisor is figure-free, but never let one leak if that changes."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(dst, tr, va, train_name="train_progress.csv",
           val_name="validation_progress.csv"):
    """Write the two progress frames into ``dst`` and return their paths."""
    os.makedirs(dst, exist_ok=True)
    tp = os.path.join(dst, train_name)
    vp = os.path.join(dst, val_name)
    tr.to_csv(tp, index=False)
    va.to_csv(vp, index=False)
    return tp, vp


def _frames(n=30, train_loss=None, val_loss=None, train_acc=None,
            val_acc=None, train_f1=None, val_f1=None):
    """Build a pair of well-formed progress frames with ``n`` epochs."""
    epoch = np.arange(1, n + 1)
    tr = pd.DataFrame({
        "epoch": epoch,
        "loss": np.linspace(0.70, 0.10, n) if train_loss is None else train_loss,
    })
    va = pd.DataFrame({
        "epoch": epoch,
        "loss": np.linspace(0.70, 0.20, n) if val_loss is None else val_loss,
    })
    if train_acc is not None:
        tr["accuracy"] = train_acc
    if val_acc is not None:
        va["accuracy"] = val_acc
    if train_f1 is not None:
        tr["f1_macro"] = train_f1
    if val_f1 is not None:
        va["macro_f1"] = val_f1          # alias form, normalised to f1_macro
    return tr, va


# ---------------------------------------------------------------------------
# early bail-outs
# ---------------------------------------------------------------------------

def test_missing_val_csv_is_flagged_and_returns_empty_summary(tmp_path):
    """A train CSV with no matching val CSV bails out with one flag."""
    from spacr.utils import suggest_training_changes

    tr, _ = _frames()
    tr.to_csv(tmp_path / "train_progress.csv", index=False)

    out = suggest_training_changes(str(tmp_path))

    assert out["flags"] == ["missing_val_csv"]
    assert out["summary"] == {}
    assert len(out["suggestions"]) == 1
    assert "val CSV" in out["suggestions"][0]


def test_missing_val_csv_explicit_path_that_does_not_exist(tmp_path):
    """An explicitly passed val path is existence-checked, not trusted."""
    from spacr.utils import suggest_training_changes

    tr, _ = _frames()
    train_path = tmp_path / "train_progress.csv"
    tr.to_csv(train_path, index=False)
    ghost = tmp_path / "validation_progress.csv"      # never created

    out = suggest_training_changes(str(tmp_path), train_csv=str(train_path),
                                   val_csv=str(ghost))

    assert not ghost.exists()
    assert out["flags"] == ["missing_val_csv"]
    assert out["summary"] == {}


def test_missing_train_csv_takes_priority_over_missing_val(tmp_path):
    """Empty dst: the train bail-out fires first and val is never reported."""
    from spacr.utils import suggest_training_changes

    out = suggest_training_changes(str(tmp_path))

    assert out["flags"] == ["missing_train_csv"]
    assert "missing_val_csv" not in out["flags"]
    assert out["summary"] == {}


def test_missing_loss_column_bails_out(tmp_path):
    """CSVs that log only accuracy cannot be judged: flag and return."""
    from spacr.utils import suggest_training_changes

    n = 20
    epoch = np.arange(1, n + 1)
    tr = pd.DataFrame({"epoch": epoch, "accuracy": np.linspace(0.6, 0.9, n)})
    va = pd.DataFrame({"epoch": epoch, "accuracy": np.linspace(0.6, 0.9, n)})
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert out["flags"] == ["missing_required_col:loss"]
    assert out["summary"] == {}
    assert out["suggestions"] == [
        "Progress CSVs lack 'loss'. Ensure _save_progress writes epoch and loss."
    ]


def test_missing_epoch_column_bails_out_before_loss_check(tmp_path):
    """``epoch`` is checked first, so a frame missing both reports epoch."""
    from spacr.utils import suggest_training_changes

    n = 20
    tr = pd.DataFrame({"loss": np.linspace(0.7, 0.1, n)})
    va = pd.DataFrame({"loss": np.linspace(0.7, 0.2, n)})
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert out["flags"] == ["missing_required_col:epoch"]
    assert out["summary"] == {}
    assert "epoch" in out["suggestions"][0]


def test_val_only_missing_column_still_bails_out(tmp_path):
    """The requirement is on BOTH frames — a val-only gap must bail out."""
    from spacr.utils import suggest_training_changes

    n = 20
    epoch = np.arange(1, n + 1)
    tr = pd.DataFrame({"epoch": epoch, "loss": np.linspace(0.7, 0.1, n)})
    va = pd.DataFrame({"epoch": epoch, "accuracy": np.linspace(0.6, 0.9, n)})
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert out["flags"] == ["missing_required_col:loss"]
    assert out["summary"] == {}


# ---------------------------------------------------------------------------
# _poly_slope NaN guard
# ---------------------------------------------------------------------------

def test_all_nan_recent_train_loss_yields_zero_slope(tmp_path):
    """<2 finite points in the window: slope is 0.0, np.polyfit is skipped."""
    from spacr.utils import suggest_training_changes

    n = 30
    train_loss = np.full(n, np.nan)
    train_loss[0] = 0.60                     # single finite point, outside last_k
    tr, va = _frames(n=n, train_loss=train_loss)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp,
                                   last_k=25)

    summary = out["summary"]
    assert summary["slope_train_loss_last_k"] == 0.0
    # the val window is untouched, so its slope must be a real (negative) fit —
    # proving the 0.0 above came from the NaN guard and not a global fallback.
    assert summary["slope_val_loss_last_k"] < -1e-3
    assert summary["epochs"] == n
    assert np.isnan(summary["final_metrics"]["train_loss"])


def test_single_finite_point_in_window_yields_zero_slope(tmp_path):
    """Exactly one finite value in the window is still below the fit minimum."""
    from spacr.utils import suggest_training_changes

    n = 12
    train_loss = np.full(n, np.nan)
    train_loss[-1] = 0.25                    # one finite point inside the window
    tr, va = _frames(n=n, train_loss=train_loss)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp,
                                   last_k=25)

    assert out["summary"]["slope_train_loss_last_k"] == 0.0
    assert out["summary"]["final_metrics"]["train_loss"] == pytest.approx(0.25)


def test_constant_loss_window_yields_zero_slope(tmp_path):
    """The allclose short-circuit: a flat window never reaches np.polyfit."""
    from spacr.utils import suggest_training_changes

    n = 30
    tr, va = _frames(n=n, val_loss=np.full(n, 0.42))
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert out["summary"]["slope_val_loss_last_k"] == 0.0
    assert out["summary"]["val_loss_std_last_k"] == pytest.approx(0.0)
    assert "val_plateau" in out["flags"]


# ---------------------------------------------------------------------------
# macro-F1 plumbing
# ---------------------------------------------------------------------------

def test_f1_macro_columns_reach_the_summary(tmp_path):
    """``f1_macro`` (train) and the ``macro_f1`` alias (val) both surface."""
    from spacr.utils import suggest_training_changes

    n = 30
    train_f1 = np.linspace(0.50, 0.88, n)
    val_f1 = np.linspace(0.45, 0.81, n)
    tr, va = _frames(n=n,
                     train_acc=np.linspace(0.60, 0.90, n),
                     val_acc=np.linspace(0.58, 0.86, n),
                     train_f1=train_f1, val_f1=val_f1)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    final = out["summary"]["final_metrics"]
    assert final["train_f1_macro"] == pytest.approx(train_f1[-1])
    assert final["val_f1_macro"] == pytest.approx(val_f1[-1])
    assert isinstance(final["train_f1_macro"], float)
    assert "f1_nan_detected" not in out["flags"]


def test_nan_f1_in_train_split_raises_the_flag(tmp_path):
    """>20% NaN macro-F1 on the train split is reported with advice."""
    from spacr.utils import suggest_training_changes

    n = 30
    train_f1 = np.linspace(0.50, 0.88, n)
    train_f1[::2] = np.nan                   # 50% NaN
    tr, va = _frames(n=n, train_f1=train_f1)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "f1_nan_detected" in out["flags"]
    assert any("F1(macro) shows NaN" in s for s in out["suggestions"])
    assert any("stratified sampling" in s for s in out["suggestions"])
    # the final epoch's F1 is finite, so the flag came from the NaN *fraction*
    assert not np.isnan(out["summary"]["final_metrics"]["train_f1_macro"])


def test_nan_f1_in_val_split_alone_raises_the_flag(tmp_path):
    """The val split is checked independently (alias column, all-NaN)."""
    from spacr.utils import suggest_training_changes

    n = 30
    tr, va = _frames(n=n, val_f1=np.full(n, np.nan))
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "f1_nan_detected" in out["flags"]
    assert np.isnan(out["summary"]["final_metrics"]["val_f1_macro"])
    assert "train_f1_macro" not in out["summary"]["final_metrics"]


def test_mostly_finite_f1_stays_below_the_nan_threshold(tmp_path):
    """10% NaN is under the 20% trigger — no flag, value still reported."""
    from spacr.utils import suggest_training_changes

    n = 30
    train_f1 = np.linspace(0.50, 0.88, n)
    train_f1[:3] = np.nan                    # 10% NaN
    tr, va = _frames(n=n, train_f1=train_f1)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "f1_nan_detected" not in out["flags"]
    assert out["summary"]["final_metrics"]["train_f1_macro"] == pytest.approx(
        train_f1[-1])


# ---------------------------------------------------------------------------
# column normalisation
# ---------------------------------------------------------------------------

def test_aliased_and_uppercased_columns_are_normalised(tmp_path):
    """``Epoch``/``Train_Loss``/``ACC`` are folded onto the canonical names."""
    from spacr.utils import suggest_training_changes

    n = 20
    epoch = np.arange(1, n + 1)
    tr = pd.DataFrame({"Epoch": epoch,
                       "Train_Loss": np.linspace(0.70, 0.10, n),
                       "ACC": np.linspace(0.60, 0.95, n)})
    va = pd.DataFrame({" epoch ": epoch,
                       "Val_Loss": np.linspace(0.70, 0.25, n),
                       "acc": np.linspace(0.58, 0.90, n)})
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    summary = out["summary"]
    assert summary["final_metrics"]["train_loss"] == pytest.approx(0.10)
    assert summary["final_metrics"]["val_loss"] == pytest.approx(0.25)
    assert summary["final_metrics"]["train_accuracy"] == pytest.approx(0.95)
    assert summary["best_epoch"] == n              # epoch column was understood
    assert summary["gen_gap_acc"] == pytest.approx(0.05)


def test_duplicate_columns_are_deduplicated_keeping_the_first(tmp_path):
    """Two names collapsing onto one canonical column keeps the first."""
    from spacr.utils import suggest_training_changes

    n = 20
    epoch = np.arange(1, n + 1)
    # 'loss' and 'train_loss' both normalise to 'loss'; the first must win.
    tr = pd.DataFrame({"epoch": epoch,
                       "loss": np.linspace(0.70, 0.10, n),
                       "train_loss": np.full(n, 999.0)})
    # 'loss' and 'Loss' collide on case folding alone.
    va = pd.DataFrame({"epoch": epoch,
                       "loss": np.linspace(0.70, 0.30, n),
                       "Loss": np.full(n, -999.0)})
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    final = out["summary"]["final_metrics"]
    assert final["train_loss"] == pytest.approx(0.10)
    assert final["val_loss"] == pytest.approx(0.30)
    assert out["summary"]["best_val_loss"] == pytest.approx(0.30)
    assert out["summary"]["best_epoch"] == n


def test_suggestions_are_deduplicated_and_ordered(tmp_path):
    """Overlapping heuristics must not repeat the same advice string."""
    from spacr.utils import suggest_training_changes

    n = 30
    # plateaued val loss + a wide accuracy gap => several heuristics fire.
    tr, va = _frames(n=n,
                     val_loss=np.full(n, 0.50) + np.linspace(0, 1e-5, n),
                     train_acc=np.linspace(0.80, 0.99, n),
                     val_acc=np.linspace(0.55, 0.60, n))
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "val_plateau" in out["flags"]
    assert "overfitting" in out["flags"]
    assert len(out["suggestions"]) == len(set(out["suggestions"]))
    assert out["suggestions"][0].startswith("Validation loss plateau detected")


def test_short_run_is_flagged_but_analysis_continues(tmp_path):
    """``few_epochs`` is a warning, not a bail-out: the summary is still built."""
    from spacr.utils import suggest_training_changes

    n = 6
    tr, va = _frames(n=n)
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp,
                                   min_epochs=10)

    assert "few_epochs" in out["flags"]
    assert out["suggestions"][0].startswith("Only 6 epochs logged (<10)")
    assert out["summary"]["epochs"] == n           # analysis did not stop
    assert out["summary"]["best_epoch"] == n
    assert out["summary"]["best_val_loss"] == pytest.approx(0.20)


def test_diverging_slopes_flag_overfitting_without_accuracy(tmp_path):
    """Train loss falling while val loss climbs is enough on its own."""
    from spacr.utils import suggest_training_changes

    n = 30
    tr, va = _frames(n=n, train_loss=np.linspace(0.70, 0.05, n),
                     val_loss=np.linspace(0.40, 0.90, n))
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "overfitting" in out["flags"]
    assert out["summary"]["gen_gap_acc"] is None       # no accuracy columns
    assert out["summary"]["slope_train_loss_last_k"] < 0
    assert out["summary"]["slope_val_loss_last_k"] > 0
    assert any("early stopping" in s for s in out["suggestions"])


def test_flat_losses_with_low_train_accuracy_flag_underfitting(tmp_path):
    """Neither loss moves and train accuracy is stuck below 0.70."""
    from spacr.utils import suggest_training_changes

    n = 30
    tr, va = _frames(n=n, train_loss=np.full(n, 0.90),
                     val_loss=np.full(n, 0.92),
                     train_acc=np.full(n, 0.55), val_acc=np.full(n, 0.54))
    tp, vp = _write(str(tmp_path), tr, va)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert "underfitting" in out["flags"]
    assert "overfitting" not in out["flags"]
    assert out["summary"]["final_metrics"]["train_accuracy"] == pytest.approx(0.55)
    assert out["summary"]["slope_train_loss_last_k"] == 0.0
    assert any("Underfitting signs" in s for s in out["suggestions"])


# ---------------------------------------------------------------------------
# non-unique index: the _scalar guard is dead because best_epoch is unguarded
# ---------------------------------------------------------------------------

def test_non_unique_index_still_reports_best_epoch(tmp_path, monkeypatch):
    """A duplicated index label must not break the summary.

    ``_scalar`` exists precisely so that ``va.loc[best_val_idx, "loss"]``
    coming back as a Series still yields a float; the very next statement then
    calls ``int()`` on the equally-Series ``epoch`` lookup.
    """
    from spacr.utils import suggest_training_changes

    n = 20
    tr, va = _frames(n=n)
    # duplicate the final (best) epoch row verbatim, as a resumed run that
    # re-logs its last epoch would.
    va = pd.concat([va, va.iloc[[-1]]], ignore_index=True)
    tp, vp = _write(str(tmp_path), tr, va)

    real_read_csv = pd.read_csv

    def _read_csv_non_unique(path, *args, **kwargs):
        df = real_read_csv(path, *args, **kwargs)
        if os.path.basename(str(path)).startswith("validation"):
            # the two identical last rows share one index label
            df.index = list(range(len(df) - 1)) + [len(df) - 2]
        return df

    monkeypatch.setattr(pd, "read_csv", _read_csv_non_unique)

    out = suggest_training_changes(str(tmp_path), train_csv=tp, val_csv=vp)

    assert out["summary"]["best_val_loss"] == pytest.approx(0.20)
    assert out["summary"]["best_epoch"] == n
