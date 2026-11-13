"""Coverage for the statistics/summary plotting helpers in spacr.plot.

Region under test (spacr/plot.py ~2557-2886):
    plot_histogram, plot_lorenz_curves, plot_permutation,
    plot_feature_importance, read_and_plot__vision_results,
    jitterplot_by_annotation

Everything here is CPU-only, offline and headless (Agg). The figures the
functions leave on the pyplot stack are inspected directly (bar heights,
line data, axis limits) so each test asserts a real numeric property
rather than merely touching the line.

Three parametrised tests carry the contracts: the summary figures, the
vision-results chart and the annotation jitter plot. Each row names a helper
that drives one call and returns what it observed, beside the value the
contract requires.
"""
from __future__ import annotations

import inspect
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

def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _counts_csv(path, names, counts):
    pd.DataFrame({"grna_name": list(names), "count": list(counts)}).to_csv(
        path, index=False
    )
    return str(path)


def _gini_reference(data):
    """Independent re-implementation of the Gini coefficient in plot.py.

    Deliberately the closed form rather than the trapezoid integration the
    implementation uses: 2*sum(i*x_i)/(n*sum(x)) - (n+1)/n.
    """
    d = np.sort(np.asarray(data, dtype=float))
    n = len(d)
    i = np.arange(1, n + 1)
    return 2 * np.sum(i * d) / (n * d.sum()) - (n + 1) / n


# ===========================================================================
# plot_histogram
# ===========================================================================

def _case_plot_histogram_writes_pdf_and_labels_axes(tmp_path):
    from spacr.plot import plot_histogram

    df = pd.DataFrame({"recruitment": np.linspace(0.0, 4.0, 50)})
    plot_histogram(df, "recruitment", dst=str(tmp_path))

    out = tmp_path / "recruitment_histogram.pdf"
    with out.open("rb") as fh:
        magic = fh.read(4)

    ax = plt.gcf().axes[0]
    # Last item: every one of the 50 observations landed in some bar.
    return (out.is_file(), magic, ax.get_title(), ax.get_xlabel(),
            ax.get_ylabel(), sum(p.get_height() for p in ax.patches))


def _case_plot_histogram_without_dst_writes_nothing(tmp_path):
    from spacr.plot import plot_histogram

    df = pd.DataFrame({"x": np.arange(20, dtype=float)})
    plot_histogram(df, "x", dst=None)

    return (list(tmp_path.iterdir()), plt.gcf().axes[0].get_xlabel())


# ===========================================================================
# plot_lorenz_curves
# ===========================================================================

def _case_plot_lorenz_curves_two_plates_curves_and_gini(tmp_path, capsys):
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

    out = capsys.readouterr().out
    g_equal = _gini_reference(equal)
    g_skew = _gini_reference(skewed)
    # The uneven library must be scored as more unequal than the even one.
    assert g_skew > g_equal
    return {
        # 10 values per plate -> 11 points; combined has 20 values -> 21.
        "points": (len(lines[0].get_ydata()), len(lines[2].get_ydata())),
        "combined_linestyle": lines[2].get_linestyle(),
        # Default limits from the None-guarded defaults.
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "title": ax.get_title(),
        "plate1_gini_printed":
            f"plate 1: Gini Coefficient = {g_equal:.4f}" in out,
        "plate2_gini_printed":
            f"plate 2: Gini Coefficient = {g_skew:.4f}" in out,
        "combined_gini_printed": "Combined: Gini Coefficient = " in out,
        "label": lines[0].get_label() == f"plate 1 (Gini: {g_equal:.4f})",
    }


def _case_plot_lorenz_curves_remove_keys_drops_rows(tmp_path, capsys):
    """Names listed in remove_keys never reach the Lorenz/Gini maths."""
    from spacr.plot import plot_lorenz_curves

    names = [f"g{i}" for i in range(9)] + ["control"]
    counts = [5] * 9 + [10_000]
    f1 = _counts_csv(tmp_path / "p1.csv", names, counts)

    plot_lorenz_curves([f1], remove_keys=["control"], save=False)

    lines = plt.gcf().axes[0].get_lines()
    # 9 surviving rows -> 10 Lorenz points (the 10_000 outlier is gone);
    # plate 1 and Combined print an identical Gini here.
    return (len(lines[0].get_ydata()),
            capsys.readouterr().out.count(
                f"Gini Coefficient = {_gini_reference([5] * 9):.4f}"))


def _case_plot_lorenz_curves_remove_outliers_by_wells(tmp_path):
    """remove_outliers=True drops names whose well count is a 1.5*IQR outlier."""
    from spacr.plot import plot_lorenz_curves

    names = [f"g{i}" for i in range(20)] + ["hog"] * 30
    f1 = _counts_csv(tmp_path / "p1.csv", names, np.arange(1, len(names) + 1))

    plot_lorenz_curves([f1], remove_keys=[], remove_outliers=True, save=False)
    kept = plt.gcf().axes[0].get_lines()[0].get_ydata()

    plt.close("all")
    plot_lorenz_curves([f1], remove_keys=[], remove_outliers=False, save=False)
    unfiltered = plt.gcf().axes[0].get_lines()[0].get_ydata()

    # all 50 rows, then 'hog' (30 wells) removed
    return (len(unfiltered), len(kept))


def _case_plot_lorenz_curves_saves_pdf_and_honours_limits(tmp_path, capsys):
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(12)],
                     np.arange(1, 13))

    plot_lorenz_curves([f1], remove_keys=[], x_lim=[0.2, 0.8], y_lim=[0.1, 0.9])

    saved = tmp_path / "results" / "lorenz_curve_with_gini.pdf"
    with saved.open("rb") as fh:
        magic = fh.read(4)

    ax = plt.gcf().axes[0]
    return (saved.is_file(), magic,
            f"Saved Lorenz Curve: {saved}" in capsys.readouterr().out,
            ax.get_xlim(), ax.get_ylim())


def _case_plot_lorenz_curves_default_remove_keys(tmp_path):
    """remove_keys defaults to None and must mean 'remove nothing'."""
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(6)],
                     np.arange(1, 7))
    plot_lorenz_curves([f1], save=False)
    return len(plt.gcf().axes[0].get_lines()[0].get_ydata())


def _case_plot_lorenz_curves_gini_of_equal_distribution_is_zero(tmp_path, capsys):
    from spacr.plot import plot_lorenz_curves

    f1 = _counts_csv(tmp_path / "p1.csv", [f"g{i}" for i in range(10)], [3] * 10)
    plot_lorenz_curves([f1], remove_keys=[], save=False)
    return "plate 1: Gini Coefficient = 0.0000" in capsys.readouterr().out


# ===========================================================================
# plot_permutation / plot_feature_importance
# ===========================================================================

def _case_plot_permutation_bar_widths_and_layout():
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

    widths = [p.get_width() for p in ax.patches]

    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    errs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(bars) == 1 and len(bars[0]) == 3
    assert len(errs) == 1
    # importance_std became the horizontal error bars: mean +- std.
    segments = errs[0][2][0].get_segments()
    return {
        "size": tuple(fig.get_size_inches()),  # min height clamp
        "widths": widths,
        "xlabel": ax.get_xlabel(),
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        "error_lows": [s[0][0] for s in segments],
        "error_highs": [s[1][0] for s in segments],
        "label_size": ax.xaxis.label.get_size(),
    }


def _case_plot_permutation_height_grows_with_feature_count():
    from spacr.plot import plot_permutation

    n = 60
    df = pd.DataFrame({
        "feature": [f"f{i}" for i in range(n)],
        "importance_mean": np.linspace(0, 1, n),
        "importance_std": np.full(n, 0.01),
    })
    fig = plot_permutation(df)
    return (tuple(fig.get_size_inches()),
            fig.axes[0].xaxis.label.get_size(), len(fig.axes[0].patches))


def _case_plot_feature_importance_bar_widths():
    from spacr.plot import plot_feature_importance

    imp = np.array([0.1, 0.4, 0.5, 0.2])
    df = pd.DataFrame({"feature": list("abcd"), "importance": imp})
    fig = plot_feature_importance(df)
    ax = fig.axes[0]

    return ([p.get_width() for p in ax.patches], ax.get_xlabel(),
            tuple(fig.get_size_inches()), len(ax.containers))


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
    (base / "notes.csv").write_text("accuracy\n0.99\n", encoding="utf-8")


@pytest.fixture
def patched_mkdir(monkeypatch):
    """Record the directory read_and_plot__vision_results creates.

    os.makedirs delegates to the module-global os.mkdir, so replacing it
    captures the path without the test having to inspect the filesystem.
    """
    real_mkdir = os.mkdir
    calls = []

    def shim(path, *args, **kwargs):
        # pathlib and lazily imported dependencies also use the process-wide
        # os.mkdir function. Only record the product directory this fixture is
        # intended to assert; still delegate every mkdir to the real function.
        if os.path.basename(os.fspath(path)) == "result":
            calls.append(os.fspath(path))
        try:
            real_mkdir(path)
        except FileExistsError:
            pass

    monkeypatch.setattr(os, "mkdir", shim)
    return calls


def _case_read_and_plot_vision_results_averages_per_model(tmp_path, patched_mkdir, capsys):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    _vision_tree(base, {
        "epoch_1": ("resnet50", [0.80, 0.90]),   # mean 0.85
        "epoch_2": ("vgg16", [0.70, 0.72]),      # mean 0.71
    })

    read_and_plot__vision_results(str(base))

    ax = plt.gcf().axes[0]
    out = capsys.readouterr().out
    return {
        "made_dirs": patched_mkdir == [str(base / "result")],
        # Sorted ascending by the metric -> vgg16 first.
        "xticks": [t.get_text() for t in ax.get_xticklabels()],
        "heights": [p.get_height() for p in ax.patches],
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
        "ylim": ax.get_ylim(),  # default y_lim
        "printed": ("resnet50" in out, "vgg16" in out),
    }


def _case_read_and_plot_vision_results_custom_metric_and_ylim(tmp_path, patched_mkdir):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    d = base / "epoch_1"
    d.mkdir(parents=True)
    pd.DataFrame({"accuracy": [0.5, 0.5], "f1": [0.25, 0.75]}).to_csv(
        d / "convnext_time1_test_result.csv", index=False
    )

    read_and_plot__vision_results(str(base), y_axis="f1", y_lim=[0.0, 1.0])

    ax = plt.gcf().axes[0]
    return ([p.get_height() for p in ax.patches], ax.get_ylabel(),
            ax.get_ylim())


def _case_read_and_plot_vision_results_no_csv_files(tmp_path, patched_mkdir, capsys):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "empty"
    (base / "epoch_1").mkdir(parents=True)

    read_and_plot__vision_results(str(base))

    return ("No CSV files found in the specified directory."
            in capsys.readouterr().out,
            plt.get_fignums())  # nothing plotted


def _case_read_and_plot_vision_results_creates_result_dir(tmp_path):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    _vision_tree(base, {"epoch_1": ("resnet50", [0.8, 0.9])})

    read_and_plot__vision_results(str(base))
    return (base / "result").is_dir()


def _case_read_and_plot_vision_results_honours_name_split(tmp_path, patched_mkdir):
    from spacr.plot import read_and_plot__vision_results

    base = tmp_path / "runs"
    d = base / "epoch_1"
    d.mkdir(parents=True)
    pd.DataFrame({"accuracy": [0.85]}).to_csv(
        d / "resnet50_run7_test_result.csv", index=False
    )

    read_and_plot__vision_results(str(base), name_split="_run")

    ax = plt.gcf().axes[0]
    return [t.get_text() for t in ax.get_xticklabels()]


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


def _case_jitterplot_balances_groups_and_keeps_annotated_wells(fake_db_readers, capsys):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")

    # Readers were handed the paths built from src.
    database = os.path.join("/exp/src", "measurements", "measurements.db")
    merge_call = fake_db_readers["merge_calls"][0]
    # Groups: pos(3), neg(2), 'NaN'(6) inside the retained wells -> min 2.
    counts = out["annotation"].value_counts()

    ax = plt.gcf().axes[0]
    return {
        "merge_locs": merge_call["locs"] == [database],
        "merge_tables": merge_call["tables"],
        "nuclei_limit": merge_call["nuclei_limit"],
        "db_tables": fake_db_readers["db_calls"][0]["tables"],
        "db_loc": fake_db_readers["db_calls"][0]["loc"] == database,
        # Well c3 has no annotated object at all -> dropped entirely.
        "wells": set(out["col_x"]),
        "groups": set(counts.index),
        "group_sizes": set(counts.values),
        "rows": len(out),
        "found_printed": "Found 2 annotated images" in capsys.readouterr().out,
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
        "hue_collections": len(ax.collections),  # one per hue level
    }


def _case_jitterplot_is_deterministic(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    a = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    plt.close("all")
    b = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    pd.testing.assert_frame_equal(a, b)
    return (len(a), len(b))


def _case_jitterplot_saves_to_output_path(tmp_path, fake_db_readers, capsys,
                                         monkeypatch):
    import spacr.plot as P
    from spacr.plot import jitterplot_by_annotation

    # ``save_figure`` writes the user's preferred figure format and rewrites the
    # extension to match; with no preference store (the case under pytest) that
    # default is PDF. This test asserts an exact ``.png`` on disk plus the PNG
    # magic bytes, so it states the preference it means.
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 200))

    dst = tmp_path / "jitter.png"
    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        plot_title="Annotated recruitment", output_path=str(dst),
    )

    assert dst.is_file() and dst.stat().st_size > 0
    with dst.open("rb") as fh:
        magic = fh.read(4)
    return (magic, f"Jitter plot saved to {dst}" in capsys.readouterr().out,
            plt.gcf().axes[0].get_title(), len(out))


def _case_jitterplot_filter_column_str(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        filter_column="condition", filter_values=["treat"],
    )
    # treat objects: c1 -> o1,o3,o5 (pos,pos,NaN), c2 -> o1,o3,o5 (neg,NaN,NaN)
    counts = out["annotation"].value_counts()
    return (set(out["condition"]), set(counts.index), set(counts.values))


def _case_jitterplot_filter_column_list(fake_db_readers, capsys):
    from spacr.plot import jitterplot_by_annotation

    out = jitterplot_by_annotation(
        "/exp/src", "annotation", "recruitment",
        filter_column=["condition", "col_x"],
        filter_values=[["treat", "ctrl"], ["c1"]],
    )
    # Only well c1 survives: pos x3, 'NaN' x3 -> balanced to 3 each.
    counts = out["annotation"].value_counts()
    return (set(out["col_x"]), set(counts.index), set(counts.values),
            capsys.readouterr().out.count("hello"))


def _case_jitterplot_missing_plate_columns_raises_keyerror(monkeypatch):
    """A merged frame without any plate/row/column columns must raise KeyError."""
    import spacr.io as sio
    from spacr.plot import jitterplot_by_annotation

    df = pd.DataFrame({"prcfo": ["a", "b"], "recruitment": [1.0, 2.0]})
    png = pd.DataFrame({"prcfo": ["a", "b"], "annotation": ["pos", "neg"]})
    monkeypatch.setattr(sio, "_read_and_merge_data",
                        lambda *a, **k: (df.copy(), []))
    monkeypatch.setattr(sio, "_read_db", lambda *a, **k: [png.copy()])

    with pytest.raises(KeyError, match="plateID") as raised:
        jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    return "plateID" in str(raised.value)


def _case_jitterplot_works_with_current_column_names(fake_db_readers):
    from spacr.plot import jitterplot_by_annotation

    fake_db_readers["legacy"] = False
    out = jitterplot_by_annotation("/exp/src", "annotation", "recruitment")
    return len(out)


_LORENZ_2 = {
    "points": (11, 21),
    "combined_linestyle": "--",
    "xlim": (0.0, 1.0),
    "ylim": (0.0, 1.0),
    "title": "Lorenz Curves",
    "plate1_gini_printed": True,
    "plate2_gini_printed": True,
    "combined_gini_printed": True,
    "label": True,
}


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_case_plot_histogram_writes_pdf_and_labels_axes,
                 (True, b"%PDF", "Histogram of recruitment", "recruitment",
                  "Frequency", pytest.approx(50.0)),
                 id="histogram-writes-pdf-and-labels-axes"),
    pytest.param(_case_plot_histogram_without_dst_writes_nothing,
                 ([], "x"), id="histogram-without-dst-writes-nothing"),
    pytest.param(_case_plot_lorenz_curves_two_plates_curves_and_gini,
                 _LORENZ_2, id="lorenz-two-plates-curves-and-gini"),
    pytest.param(_case_plot_lorenz_curves_remove_keys_drops_rows,
                 (10, 2), id="lorenz-remove-keys-drops-rows"),
    pytest.param(_case_plot_lorenz_curves_remove_outliers_by_wells,
                 (51, 21), id="lorenz-remove-outliers-by-wells"),
    pytest.param(_case_plot_lorenz_curves_saves_pdf_and_honours_limits,
                 (True, b"%PDF", True, (0.2, 0.8), (0.1, 0.9)),
                 id="lorenz-saves-pdf-and-honours-limits"),
    pytest.param(_case_plot_lorenz_curves_default_remove_keys,
                 7, id="lorenz-default-remove-keys-removes-nothing"),
    pytest.param(_case_plot_lorenz_curves_gini_of_equal_distribution_is_zero,
                 True, id="lorenz-gini-of-equal-distribution-is-zero"),
    pytest.param(_case_plot_permutation_bar_widths_and_layout,
                 {"size": (10.0, 8.0),
                  "widths": pytest.approx([0.05, 0.20, 0.35]),
                  "xlabel": "Permutation Importance",
                  "yticklabels": ["a", "b", "c"],
                  "error_lows": pytest.approx([0.04, 0.18, 0.32]),
                  "error_highs": pytest.approx([0.06, 0.22, 0.38]),
                  "label_size": pytest.approx(max(10, 12 - 3 * 0.2))},
                 id="permutation-bar-widths-and-layout"),
    pytest.param(_case_plot_permutation_height_grows_with_feature_count,
                 ((10.0, 60 * 0.3), pytest.approx(10.0), 60),
                 id="permutation-height-grows-with-feature-count"),
    pytest.param(_case_plot_feature_importance_bar_widths,
                 (pytest.approx([0.1, 0.4, 0.5, 0.2]), "Feature Importance",
                  (10.0, 8.0), 1),
                 id="feature-importance-bar-widths"),
])
def test_summary_plots_draw_what_the_data_says(request, helper, expected):
    """Histogram, Lorenz, permutation and feature-importance figures carry the
    numbers of the data handed in."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_read_and_plot_vision_results_averages_per_model,
        {"made_dirs": True, "xticks": ["vgg16", "resnet50"],
         "heights": pytest.approx([0.71, 0.85]), "xlabel": "Model",
         "ylabel": "accuracy", "title": "Average Accuracy per Model",
         "ylim": (0.8, 0.9), "printed": (True, True)},
        id="vision-averages-per-model"),
    pytest.param(
        _case_read_and_plot_vision_results_custom_metric_and_ylim,
        (pytest.approx([0.5]), "f1", (0.0, 1.0)),
        id="vision-custom-metric-and-ylim"),
    pytest.param(_case_read_and_plot_vision_results_no_csv_files,
                 (True, []), id="vision-no-csv-files-plots-nothing"),
    pytest.param(_case_read_and_plot_vision_results_creates_result_dir,
                 True, id="vision-creates-result-dir"),
    pytest.param(_case_read_and_plot_vision_results_honours_name_split,
                 ["resnet50"], id="vision-honours-name-split"),
])
def test_vision_results_chart_averages_the_metric_per_model(
        request, helper, expected):
    """The per-model chart reads only ``*_test_result.csv`` files."""
    observed = _call(request, helper)
    assert observed == expected


@pytest.mark.parametrize("helper, expected", [
    pytest.param(
        _case_jitterplot_balances_groups_and_keeps_annotated_wells,
        {"merge_locs": True,
         "merge_tables": ["cell", "nucleus", "pathogen", "cytoplasm"],
         "nuclei_limit": True, "db_tables": ["png_list"], "db_loc": True,
         "wells": {"c1", "c2"}, "groups": {"pos", "neg", "NaN"},
         "group_sizes": {2}, "rows": 6, "found_printed": True,
         "xlabel": "annotation", "ylabel": "recruitment",
         "title": "Jitter Plot", "hue_collections": 3},
        id="jitter-balances-groups-and-keeps-annotated-wells"),
    pytest.param(_case_jitterplot_is_deterministic, (6, 6),
                 id="jitter-is-deterministic"),
    pytest.param(_case_jitterplot_saves_to_output_path,
                 (b"\x89PNG", True, "Annotated recruitment", 6),
                 id="jitter-saves-to-output-path"),
    pytest.param(_case_jitterplot_filter_column_str,
                 ({"treat"}, {"pos", "neg", "NaN"}, {1}),
                 id="jitter-filter-column-string"),
    pytest.param(_case_jitterplot_filter_column_list,
                 ({"c1"}, {"pos", "NaN"}, {3}, 2),
                 id="jitter-filter-column-list"),
    pytest.param(_case_jitterplot_missing_plate_columns_raises_keyerror,
                 True, id="jitter-missing-plate-columns-raises-keyerror"),
    pytest.param(_case_jitterplot_works_with_current_column_names, 6,
                 id="jitter-works-with-current-column-names"),
])
def test_jitterplot_by_annotation_balances_and_filters_annotated_objects(
        request, helper, expected):
    """The annotation jitter plot joins the two readers, keeps annotated
    wells, balances the groups and honours the filters."""
    observed = _call(request, helper)
    assert observed == expected
