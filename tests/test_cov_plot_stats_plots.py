"""Coverage for the statistics/summary plotting helpers in spacr.plot.

Region under test (spacr/plot.py ~2557-2886):
    plot_histogram, plot_lorenz_curves, plot_permutation,
    plot_feature_importance, read_and_plot__vision_results,
    jitterplot_by_annotation

Everything here is CPU-only, offline and headless (Agg). The figures the
functions leave on the pyplot stack are inspected directly (bar heights,
line data, axis limits) so each test asserts a real numeric property
rather than merely touching the line.

Three latent bugs are pinned with strict xfail tests asserting the
CORRECT behaviour (never the broken one):
    * plot_lorenz_curves(remove_keys=None)  -> TypeError
    * read_and_plot__vision_results         -> os.mkdir(dst, exists=True)
    * jitterplot_by_annotation              -> stale plate_x/row_x/col_x
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _no_figure_leak():
    """No figure survives a test (and none is inherited from another)."""
    plt.close("all")
    yield
    plt.close("all")


# ===========================================================================
# helpers
# ===========================================================================

def _counts_csv(path, names, counts):
    pd.DataFrame({"grna_name": list(names), "count": list(counts)}).to_csv(
        path, index=False
    )
    return str(path)


def _gini_reference(data):
    """Independent re-implementation of the left-Riemann Gini in plot.py."""
    d = np.sort(np.asarray(data, dtype=float))
    n = len(d)
    cum = np.insert(np.cumsum(d) / d.sum(), 0, 0)
    return 1 - 2 * np.sum(cum[:-1] * np.diff(np.linspace(0, 1, n + 1)))


# ===========================================================================
# plot_histogram
# ===========================================================================

def test_plot_histogram_writes_pdf_and_labels_axes(tmp_path):
    from spacr.plot import plot_histogram

    df = pd.DataFrame({"recruitment": np.linspace(0.0, 4.0, 50)})
    plot_histogram(df, "recruitment", dst=str(tmp_path))

    out = tmp_path / "recruitment_histogram.pdf"
    assert out.is_file()
    with out.open("rb") as fh:
        assert fh.read(4) == b"%PDF"

    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Histogram of recruitment"
    assert ax.get_xlabel() == "recruitment"
    assert ax.get_ylabel() == "Frequency"
    # Every one of the 50 observations landed in some bar.
    assert sum(p.get_height() for p in ax.patches) == pytest.approx(50.0)


def test_plot_histogram_without_dst_writes_nothing(tmp_path):
    from spacr.plot import plot_histogram

    df = pd.DataFrame({"x": np.arange(20, dtype=float)})
    plot_histogram(df, "x", dst=None)

    assert list(tmp_path.iterdir()) == []
    assert plt.gcf().axes[0].get_xlabel() == "x"


# ===========================================================================
# plot_lorenz_curves
# ===========================================================================

def test_plot_lorenz_curves_two_plates_curves_and_gini(tmp_path, capsys):
    """Two plates + a combined dashed curve; Gini printed for all three."""
    from spacr.plot import plot_lorenz_curves

    equal = [7] * 10                       # perfectly even library
    skewed = list(range(1, 11))            # uneven library
    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(10)], equal)
    f2 = _counts_csv(tmp_path / "p2.csv", [f"g{i}" for i in range(10)], skewed)

    plot_lorenz_curves([f1, f2], remove_keys=["not-present"], save=False)

    ax = plt.gcf().axes[0]
    lines = ax.get_lines()
    assert len(lines) == 3  # plate 1, plate 2, combined

    for line in lines:
        y = line.get_ydata()
        x = line.get_xdata()
        assert y[0] == 0.0
        assert y[-1] == pytest.approx(1.0)
        assert np.all(np.diff(y) >= -1e-12)      # Lorenz curve is monotone
        assert x[0] == 0.0 and x[-1] == pytest.approx(1.0)
        assert len(x) == len(y)

    # 10 values per plate -> 11 points; combined has 20 values -> 21 points.
    assert len(lines[0].get_ydata()) == 11
    assert len(lines[2].get_ydata()) == 21
    assert lines[2].get_linestyle() == "--"

    # Default limits from the None-guarded defaults.
    assert ax.get_xlim() == (0.0, 1.0)
    assert ax.get_ylim() == (0.0, 1.0)
    assert ax.get_title() == "Lorenz Curves"

    out = capsys.readouterr().out
    g_equal = _gini_reference(equal)
    g_skew = _gini_reference(skewed)
    assert f"plate 1: Gini Coefficient = {g_equal:.4f}" in out
    assert f"plate 2: Gini Coefficient = {g_skew:.4f}" in out
    assert "Combined: Gini Coefficient = " in out
    # The uneven library must be scored as more unequal than the even one.
    assert g_skew > g_equal
    assert f"plate 1 (Gini: {g_equal:.4f})" == lines[0].get_label()


def test_plot_lorenz_curves_remove_keys_drops_rows(tmp_path, capsys):
    """Names listed in remove_keys never reach the Lorenz/Gini maths."""
    from spacr.plot import plot_lorenz_curves

    names = [f"g{i}" for i in range(9)] + ["control"]
    counts = [5] * 9 + [10_000]
    f1 = _counts_csv(tmp_path / "p1.csv", names, counts)

    plot_lorenz_curves([f1], remove_keys=["control"], save=False)

    lines = plt.gcf().axes[0].get_lines()
    # 9 surviving rows -> 10 Lorenz points (the 10_000 outlier is gone).
    assert len(lines[0].get_ydata()) == 10
    assert capsys.readouterr().out.count(
        f"Gini Coefficient = {_gini_reference([5] * 9):.4f}"
    ) == 2  # plate 1 and Combined are identical here


def test_plot_lorenz_curves_remove_outliers_by_wells(tmp_path):
    """remove_outliers=True drops names whose well count is a 1.5*IQR outlier."""
    from spacr.plot import plot_lorenz_curves

    names = [f"g{i}" for i in range(20)] + ["hog"] * 30
    f1 = _counts_csv(tmp_path / "p1.csv", names, np.arange(1, len(names) + 1))

    plot_lorenz_curves([f1], remove_keys=[], remove_outliers=True, save=False)
    kept = plt.gcf().axes[0].get_lines()[0].get_ydata()

    plt.close("all")
    plot_lorenz_curves([f1], remove_keys=[], remove_outliers=False, save=False)
    unfiltered = plt.gcf().axes[0].get_lines()[0].get_ydata()

    assert len(unfiltered) == 51          # all 50 rows
    assert len(kept) == 21                # 'hog' (30 wells) removed


def test_plot_lorenz_curves_saves_pdf_and_honours_limits(tmp_path, capsys):
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(12)],
                     np.arange(1, 13))

    plot_lorenz_curves([f1], remove_keys=[], x_lim=[0.2, 0.8], y_lim=[0.1, 0.9])

    saved = tmp_path / "results" / "lorenz_curve_with_gini.pdf"
    assert saved.is_file()
    with saved.open("rb") as fh:
        assert fh.read(4) == b"%PDF"
    assert f"Saved Lorenz Curve: {saved}" in capsys.readouterr().out

    ax = plt.gcf().axes[0]
    assert ax.get_xlim() == (0.2, 0.8)
    assert ax.get_ylim() == (0.1, 0.9)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: plot_lorenz_curves(remove_keys=None) iterates None -> TypeError",
)
def test_plot_lorenz_curves_default_remove_keys(tmp_path):
    """remove_keys defaults to None and must mean 'remove nothing'."""
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(6)],
                     np.arange(1, 7))
    plot_lorenz_curves([f1], save=False)
    assert len(plt.gcf().axes[0].get_lines()[0].get_ydata()) == 7


@pytest.mark.xfail(
    strict=True,
    reason="BUG: gini_coefficient uses a left-Riemann sum, so a perfectly "
           "equal distribution of n items scores 1/n instead of 0",
)
def test_plot_lorenz_curves_gini_of_equal_distribution_is_zero(tmp_path, capsys):
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(10)], [3] * 10)
    plot_lorenz_curves([f1], remove_keys=[], save=False)
    assert "plate 1: Gini Coefficient = 0.0000" in capsys.readouterr().out


# ===========================================================================
# plot_permutation / plot_feature_importance
# ===========================================================================

def test_plot_permutation_bar_widths_and_layout():
    from spacr.plot import plot_permutation

    from matplotlib.container import BarContainer, ErrorbarContainer

    means = np.array([0.05, 0.20, 0.35])
    stds = np.array([0.01, 0.02, 0.03])
    df = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "importance_mean": means,
        "importance_std": stds,
    })
    fig = plot_permutation(df)
    ax = fig.axes[0]

    assert tuple(fig.get_size_inches()) == (10.0, 8.0)  # min height clamp
    widths = [p.get_width() for p in ax.patches]
    assert widths == pytest.approx(list(means))
    assert ax.get_xlabel() == "Permutation Importance"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]

    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    errs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(bars) == 1 and len(bars[0]) == 3
    assert len(errs) == 1
    # importance_std became the horizontal error bars: mean +- std.
    segments = errs[0][2][0].get_segments()
    assert [s[0][0] for s in segments] == pytest.approx(list(means - stds))
    assert [s[1][0] for s in segments] == pytest.approx(list(means + stds))
    assert ax.xaxis.label.get_size() == pytest.approx(max(10, 12 - 3 * 0.2))


def test_plot_permutation_height_grows_with_feature_count():
    from spacr.plot import plot_permutation

    n = 60
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(n)],
        "importance_mean": np.linspace(0, 1, n),
        "importance_std": np.full(n, 0.01),
    })
    fig = plot_permutation(df)
    assert tuple(fig.get_size_inches()) == (10.0, n * 0.3)
    assert fig.axes[0].xaxis.label.get_size() == pytest.approx(10.0)
    assert len(fig.axes[0].patches) == n


def test_plot_feature_importance_bar_widths():
    from spacr.plot import plot_feature_importance

    imp = np.array([0.1, 0.4, 0.5, 0.2])
    df = pd.DataFrame({"feature": list("abcd"), "importance": imp})
    fig = plot_feature_importance(df)
    ax = fig.axes[0]

    assert [p.get_width() for p in ax.patches] == pytest.approx(list(imp))
    assert ax.get_xlabel() == "Feature Importance"
    assert tuple(fig.get_size_inches()) == (10.0, 8.0)
    assert len(ax.containers) == 1


# ===========================================================================
# read_and_plot__vision_results
# ===========================================================================

def _vision_tree(base, rows_per_model):
    """base/<epoch>/<model>_time<ts>_test_result.csv trees."""
    for epoch, (model, accs) in rows_per_model.items():
        d = base / epoch
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"accuracy": accs, "loss": np.zeros(len(accs))}).to_csv(
            d / f"{model}_time1700000000_test_result.csv", index=False
        )
    # A file that must be ignored by the *_test_result.csv filter.
    (base / "notes.csv").write_text("accuracy\n0.99\n")


@pytest.fixture
def patched_mkdir(monkeypatch):
    """Make os.mkdir tolerate the buggy `exists=True` kwarg + re-runs.

    Without this shim read_and_plot__vision_results dies on its first
    statement (see the strict-xfail test below) and none of its real
    logic can be reached.
    """
    real_mkdir = os.mkdir
    calls = []

    def shim(path, *args, **kwargs):
        calls.append(os.fspath(path))
        try:
            real_mkdir(path)
        except FileExistsError:
            pass

    monkeypatch.setattr(os, "mkdir", shim)
    return calls


def test_read_and_plot_vision_results_averages_per_model(tmp_path, patched_mkdir, capsys):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    _vision_tree(base, {
        "epoch_1": ("resnet50", [0.80, 0.90]),   # mean 0.85
        "epoch_2": ("vgg16", [0.70, 0.72]),      # mean 0.71
    })

    read_and_plot__vision_results(str(base))

    assert patched_mkdir == [str(base / "result")]

    ax = plt.gcf().axes[0]
    # Sorted ascending by the metric -> vgg16 first.
    assert [t.get_text() for t in ax.get_xticklabels()] == ["vgg16", "resnet50"]
    assert [p.get_height() for p in ax.patches] == pytest.approx([0.71, 0.85])
    assert ax.get_xlabel() == "Model"
    assert ax.get_ylabel() == "accuracy"
    assert ax.get_title() == "Average Accuracy per Model"
    assert ax.get_ylim() == (0.8, 0.9)  # default y_lim

    out = capsys.readouterr().out
    assert "resnet50" in out and "vgg16" in out


def test_read_and_plot_vision_results_custom_metric_and_ylim(tmp_path, patched_mkdir):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    d = base / "epoch_1"
    d.mkdir(parents=True)
    pd.DataFrame({"accuracy": [0.5, 0.5], "f1": [0.25, 0.75]}).to_csv(
        d / "convnext_time1_test_result.csv", index=False
    )

    read_and_plot__vision_results(str(base), y_axis="f1", y_lim=[0.0, 1.0])

    ax = plt.gcf().axes[0]
    assert [p.get_height() for p in ax.patches] == pytest.approx([0.5])
    assert ax.get_ylabel() == "f1"
    assert ax.get_ylim() == (0.0, 1.0)


def test_read_and_plot_vision_results_no_csv_files(tmp_path, patched_mkdir, capsys):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "empty"
    (base / "epoch_1").mkdir(parents=True)

    read_and_plot__vision_results(str(base))

    assert "No CSV files found in the specified directory." in capsys.readouterr().out
    assert plt.get_fignums() == []  # nothing plotted


@pytest.mark.xfail(
    strict=True,
    reason="BUG: read_and_plot__vision_results calls os.mkdir(dst, exists=True); "
           "os.mkdir takes no 'exists' kwarg -> TypeError on every call",
)
def test_read_and_plot_vision_results_creates_result_dir(tmp_path):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    _vision_tree(base, {"epoch_1": ("resnet50", [0.8, 0.9])})

    read_and_plot__vision_results(str(base))
    assert (base / "result").is_dir()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: read_and_plot__vision_results splits on a hard-coded '_time' "
           "instead of name_split, so any other name_split raises IndexError",
)
def test_read_and_plot_vision_results_honours_name_split(tmp_path, patched_mkdir):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    d = base / "epoch_1"
    d.mkdir(parents=True)
    pd.DataFrame({"accuracy": [0.85]}).to_csv(
        d / "resnet50_run7_test_result.csv", index=False
    )

    read_and_plot__vision_results(str(base), name_split="_run")

    ax = plt.gcf().axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["resnet50"]


# ===========================================================================
# jitterplot_by_annotation
# ===========================================================================
#
# The function joins two spacr.io readers and then does all its real work
# on the merged frame. The readers are replaced with deterministic fakes
# so the join + balancing + plotting logic is exercised on CPU with no DB
# engine dependency; the fakes record the arguments the function passed
# them, which is asserted below.

def _measurement_frame(legacy=True):
    """One row per object across three wells of one plate."""
    plate_col, row_col, col_col = (
        ("plate", "row", "col") if legacy else ("plateID", "rowID", "columnID")
    )
    rows = []
    layout = [("c1", 6), ("c2", 5), ("c3", 4)]
    for col, n in layout:
        for i in range(n):
            rows.append({
                "prcfo": f"plate1_r1_{col}_f1_o{i+1}",
                plate_col: "plate1",
                row_col: "r1",
                col_col: col,
                "recruitment": 1.0 + i,
                "condition": "treat" if i % 2 == 0 else "ctrl",
            })
    return pd.DataFrame(rows)


def _png_frame(legacy=True):
    """png_list-style annotation frame; only some objects are annotated."""
    plate_col, row_col, col_col = (
        ("plate", "row", "col") if legacy else ("plateID", "rowID", "columnID")
    )
    meas = _measurement_frame(legacy=legacy)
    ann = []
    for _, r in meas.iterrows():
        col = r[col_col]
        idx = int(r["prcfo"].split("_o")[1])
        if col == "c1" and idx <= 3:
            label = "pos"
        elif col == "c2" and idx <= 2:
            label = "neg"
        else:
            label = None
        ann.append({
            "prcfo": r["prcfo"],
            plate_col: r[plate_col],
            row_col: r[row_col],
            col_col: r[col_col],
            "annotation": label,
            "png_path": f"/fake/{r['prcfo']}.png",
        })
    return pd.DataFrame(ann)


@pytest.fixture
def fake_db_readers(monkeypatch):
    """Patch spacr.io._read_and_merge_data / _read_db with recording fakes."""
    import spacr.io as sio

    state = {"legacy": True, "merge_calls": [], "db_calls": []}

    def fake_merge(locs, tables, verbose=False, nuclei_limit=10,
                   pathogen_limit=10, change_plate=False):
        state["merge_calls"].append(
            {"locs": list(locs), "tables": list(tables), "verbose": verbose,
             "nuclei_limit": nuclei_limit, "pathogen_limit": pathogen_limit}
        )
        return _measurement_frame(state["legacy"]), []

    def fake_read_db(db_loc, tables):
        state["db_calls"].append({"loc": db_loc, "tables": list(tables)})
        return [_png_frame(state["legacy"])]

    monkeypatch.setattr(sio, "_read_and_merge_data", fake_merge)
    monkeypatch.setattr(sio, "_read_db", fake_read_db)
    return state


def test_jitterplot_balances_groups_and_keeps_annotated_wells(fake_db_readers, capsys):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")

    # Readers were handed the paths built from src.
    assert fake_db_readers["merge_calls"][0]["locs"] == [
        "/exp/src/measurements/measurements.db"
    ]
    assert fake_db_readers["merge_calls"][0]["tables"] == [
        "cell", "nucleus", "pathogen", "cytoplasm"
    ]
    assert fake_db_readers["merge_calls"][0]["nuclei_limit"] is True
    assert fake_db_readers["db_calls"][0]["tables"] == ["png_list"]
    assert fake_db_readers["db_calls"][0]["loc"] == \
        "/exp/src/measurements/measurements.db"

    # Well c3 has no annotated object at all -> dropped entirely.
    assert set(out["col_x"]) == {"c1", "c2"}
    # Groups: pos(3), neg(2), 'NaN'(6) inside the retained wells -> min 2.
    counts = out["annotation"].value_counts()
    assert set(counts.index) == {"pos", "neg", "NaN"}
    assert set(counts.values) == {2}
    assert len(out) == 6
    assert "Found 2 annotated images" in capsys.readouterr().out

    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "annotation"
    assert ax.get_ylabel() == "recruitment"
    assert ax.get_title() == "Jitter Plot"
    assert len(ax.collections) == 3  # one PathCollection per hue level


def test_jitterplot_is_deterministic(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    a = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    plt.close("all")
    b = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    pd.testing.assert_frame_equal(a, b)


def test_jitterplot_saves_to_output_path(tmp_path, fake_db_readers, capsys):
    from spacr.plot import jitterplot_by_annotation

    dst = tmp_path / "jitter.png"
    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        plot_title="Annotated recruitment", output_path=str(dst),
    )

    assert dst.is_file() and dst.stat().st_size > 0
    with dst.open("rb") as fh:
        assert fh.read(4) == b"\x89PNG"
    assert f"Jitter plot saved to {dst}" in capsys.readouterr().out
    assert plt.gcf().axes[0].get_title() == "Annotated recruitment"
    assert len(out) == 6


def test_jitterplot_filter_column_str(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        filter_column="condition", filter_values=["treat"],
    )
    assert set(out["condition"]) == {"treat"}
    # treat objects: c1 -> o1,o3,o5 (pos,pos,NaN), c2 -> o1,o3,o5 (neg,NaN,NaN)
    counts = out["annotation"].value_counts()
    assert set(counts.index) == {"pos", "neg", "NaN"}
    assert set(counts.values) == {1}


def test_jitterplot_filter_column_list(fake_db_readers, capsys):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        filter_column=["condition", "col_x"],
        filter_values=[["treat", "ctrl"], ["c1"]],
    )
    assert set(out["col_x"]) == {"c1"}
    # Only well c1 survives: pos x3, 'NaN' x3 -> balanced to 3 each.
    counts = out["annotation"].value_counts()
    assert set(counts.index) == {"pos", "NaN"}
    assert set(counts.values) == {3}
    assert capsys.readouterr().out.count("hello") == 2


def test_jitterplot_missing_plate_columns_raises_keyerror(monkeypatch):
    """A merged frame without plate_x/row_x/col_x must raise KeyError."""
    import spacr.io as sio
    from spacr.plot import jitterplot_by_annotation

    df = pd.DataFrame({"prcfo": ["a", "b"], "recruitment": [1.0, 2.0]})
    png = pd.DataFrame({"prcfo": ["a", "b"], "annotation": ["pos", "neg"]})
    monkeypatch.setattr(sio, "_read_and_merge_data",
                        lambda *a, **k: (df.copy(), []))
    monkeypatch.setattr(sio, "_read_db", lambda *a, **k: [png.copy()])

    with pytest.raises(KeyError, match="plate_x"):
        jitterplot_by_annotation("/exp/src", "annotation", "recruitment")


@pytest.mark.xfail(
    strict=True,
    reason="BUG: jitterplot_by_annotation requires the legacy plate_x/row_x/"
           "col_x columns; spacr.io renames these to plateID/rowID/columnID, "
           "so the function always raises KeyError on a current database",
)
def test_jitterplot_works_with_current_column_names(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    fake_db_readers["legacy"] = False
    out = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    assert len(out) == 6
