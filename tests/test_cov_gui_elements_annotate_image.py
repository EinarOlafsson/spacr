"""
CPU-only coverage tests for ``spacr.gui_elements.AnnotateApp``.

Covers the image-handling half of the class (roughly lines 4400-5100 of
``spacr/gui_elements.py``):

    * ``_normalize_filter_inputs`` / ``_apply_threshold`` /
      ``_resolve_threshold_value``  (filter coercion + quantile resolution)
    * ``prefilter_paths_annotations`` (measurement-join branch and the plain
      ``png_list`` paging branch, with and without ``image_type``)
    * ``load_images`` / ``load_single_image`` (path adjustment, missing files,
      colored borders, click bindings)
    * ``show_class_counts`` (class tally window, non-integer labels)
    * ``fill_holes`` / ``_filter_objects_by_area`` / ``outline_image``
    * ``normalize_image`` / ``filter_channels`` / ``add_colored_border``
    * ``get_on_image_click`` (annotation toggling)
    * ``update_html`` / ``clear_current_annotation``
    * two defensive branches of ``update_database_worker``

Pure-logic methods are exercised on a bare instance built with
``object.__new__`` so no display is needed; the widget-driven methods use the
``tk_root`` fixture from ``conftest.py`` which skips cleanly when no display
is available.
"""
from __future__ import annotations

import os
import queue as _queue
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a matplotlib window survive a test."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _bare_app(**attrs):
    """An AnnotateApp instance with ``__init__`` skipped.

    Only the attributes a given method actually reads are set, which keeps the
    pure-logic tests free of Tk, threads and sqlite.
    """
    from spacr.gui_elements import AnnotateApp
    app = object.__new__(AnnotateApp)
    for k, v in attrs.items():
        setattr(app, k, v)
    return app


def _rgb(arr):
    from PIL import Image
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# synthetic project: src dir + png files + measurements.db
# ---------------------------------------------------------------------------

@pytest.fixture
def annot_env(tmp_path):
    """A spacr-shaped src dir with real PNGs and a png_list table.

    ``png_list`` deliberately mixes three kinds of path so ``load_images`` can
    exercise all three of its rewriting branches:
      * 4 paths already rooted at ``src``
      * 1 path rooted elsewhere but containing ``/data/`` (rewritten to src)
      * 1 path with neither (left alone -> file does not exist)
    """
    from PIL import Image

    src = tmp_path / "proj"
    (src / "measurements").mkdir(parents=True)
    img_dir = src / "data" / "cell_png"
    img_dir.mkdir(parents=True)

    # Deterministic 40x40 RGB tiles: bright square in r+g on a dark field.
    on_disk = []
    for i in range(5):
        arr = np.zeros((40, 40, 3), dtype=np.uint8)
        arr[10:30, 10:30, 0] = 200 + i
        arr[12:28, 12:28, 1] = 180
        arr[18:22, 18:22, :] = 0          # interior hole -> fill_holes has work
        arr[:, :, 2] = 25                 # flat blue background
        p = img_dir / f"img_{i}.png"
        Image.fromarray(arr, mode="RGB").save(p)
        on_disk.append(str(p))

    db_path = src / "measurements" / "measurements.db"

    # rows 0-3 -> real src-rooted paths
    # row 4    -> foreign root but /data/ present; rewrites onto on_disk[4]
    # row 5    -> no /data/ and no file on disk
    png_paths = on_disk[:4] + [
        "/foreign/root/data/cell_png/img_4.png",
        "/nowhere/nucleus_missing.png",
    ]
    n = len(png_paths)
    png_list = pd.DataFrame({
        "png_path": png_paths,
        "prcfo": [f"plate1_A01_1_o{i + 1}" for i in range(n)],
        "cell_id": [f"o{i + 1}" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["1"] * n,
    })
    con = sqlite3.connect(db_path)
    try:
        png_list.to_sql("png_list", con, index=False)
    finally:
        con.close()

    return {
        "src": str(src),
        "db_path": str(db_path),
        "png_paths": png_paths,
        "on_disk": on_disk,
        "n": n,
    }


@pytest.fixture
def make_app(tk_root, annot_env):
    """Factory for a live AnnotateApp on a small Toplevel; always torn down."""
    import tkinter as tk

    made = []

    def _make(**kwargs):
        from spacr.gui_elements import AnnotateApp
        top = tk.Toplevel(tk_root)
        # Keep the auto-sized grid tiny: __init__ sizes the window from the
        # reported screen size, so shrink what it reports.
        top.winfo_screenwidth = lambda: 700
        top.winfo_screenheight = lambda: 520
        params = dict(
            db_path=annot_env["db_path"],
            src=annot_env["src"],
            image_size=100,
            annotation_column="annotate",
        )
        params.update(kwargs)
        app = AnnotateApp(root=top, **params)
        # Stop the status-poll from re-arming itself forever.
        app._poll_save_status = lambda: None
        app.grid_rows, app.grid_cols = 2, 3
        app.recreate_image_grid()
        made.append((app, top))
        return app

    yield _make

    for app, top in made:
        app.terminate = True
        try:
            app.update_queue.put(app.SENTINEL)
        except Exception:
            pass
        try:
            if app.db_update_thread.is_alive():
                app.db_update_thread.join(timeout=5)
        except Exception:
            pass
        try:
            top.destroy()
        except Exception:
            pass


# ===========================================================================
# _normalize_filter_inputs
# ===========================================================================

def _normalize(m, t, d):
    app = _bare_app(measurement=m, threshold=t, threshold_direction=d)
    return app, app._normalize_filter_inputs()


def test_normalize_scalar_measurement_passthrough():
    app, kind = _normalize("cell_area", 500, "higher")
    assert kind == "scalar"
    assert (app.measurement, app.threshold, app.threshold_direction) == (
        "cell_area", 500, "higher")


def test_normalize_scalar_measurement_collapses_list_threshold(capsys):
    app, kind = _normalize("cell_area", [500, 900], ["lower", "higher"])
    assert kind == "scalar"
    assert app.threshold == 500
    assert app.threshold_direction == "lower"
    out = capsys.readouterr().out
    assert "threshold is a list" in out
    assert "threshold_direction is a list" in out


def test_normalize_scalar_rejects_bad_direction():
    with pytest.raises(ValueError, match="must be 'lower' or 'higher'"):
        _normalize("cell_area", 5, "sideways")


def test_normalize_empty_measurement_list_raises():
    with pytest.raises(ValueError, match="empty list"):
        _normalize([], 5, "higher")


def test_normalize_broadcasts_scalar_threshold_and_direction(capsys):
    app, kind = _normalize(["a", "b", "c"], 7, "lower")
    assert kind == "list"
    assert app.threshold == [7, 7, 7]
    assert app.threshold_direction == ["lower"] * 3
    out = capsys.readouterr().out
    assert "broadcasting threshold to length 3" in out
    assert "broadcasting threshold_direction to length 3" in out


def test_normalize_rejects_threshold_length_mismatch():
    with pytest.raises(ValueError, match=r"len\(threshold\) = 2"):
        _normalize(["a", "b", "c"], [1, 2], "lower")


def test_normalize_rejects_direction_length_mismatch():
    with pytest.raises(ValueError, match=r"len\(threshold_direction\) = 1"):
        _normalize(["a", "b"], [1, 2], ["lower"])


def test_normalize_rejects_bad_direction_inside_list():
    with pytest.raises(ValueError, match=r"threshold_direction\[1\]"):
        _normalize(["a", "b"], [1, 2], ["lower", "diagonal"])


def test_normalize_detects_list_of_lists():
    app, kind = _normalize([["a"], ["b", "c"]], [1, 2], "higher")
    assert kind == "list_of_lists"
    assert app.measurement == [["a"], ["b", "c"]]
    assert app.threshold_direction == ["higher", "higher"]


def test_normalize_rejects_inner_list_of_wrong_length():
    with pytest.raises(ValueError, match=r"measurement\[0\] must be a list of 1 or 2"):
        _normalize([["a", "b", "c"]], [1], "higher")


def test_normalize_tuple_measurement_is_accepted():
    app, kind = _normalize(("a", "b"), (1, 2), ("lower", "higher"))
    assert kind == "list"
    assert app.measurement == ["a", "b"]
    assert app.threshold == [1, 2]


def test_normalize_rejects_non_string_non_list_measurement():
    with pytest.raises(TypeError, match="measurement must be a string or a list"):
        _normalize(5, 1, "higher")


# ===========================================================================
# _resolve_threshold_value / _apply_threshold
# ===========================================================================

def test_resolve_threshold_quantile_string():
    app = _bare_app()
    s = pd.Series(np.arange(1, 11, dtype=float))
    assert app._resolve_threshold_value("q5", s) == pytest.approx(s.quantile(0.5))
    assert app._resolve_threshold_value("q9", s) == pytest.approx(s.quantile(0.9))


def test_resolve_threshold_unknown_string_raises():
    app = _bare_app()
    with pytest.raises(ValueError, match="Unknown threshold string"):
        app._resolve_threshold_value("q42", pd.Series([1.0, 2.0]))


def test_resolve_threshold_numeric_passthrough():
    app = _bare_app()
    assert app._resolve_threshold_value(3.5, pd.Series([1.0, 2.0])) == 3.5


@pytest.mark.parametrize("direction,expected", [
    ("lower", [1, 2, 3, 4, 5]),
    ("higher", [5, 6, 7, 8, 9, 10]),
])
def test_apply_threshold_filters_rows(direction, expected, capsys):
    app = _bare_app()
    df = pd.DataFrame({"x": list(range(1, 11))})
    out = app._apply_threshold(df, "x", 5, direction)
    assert out["x"].tolist() == expected
    assert f"Filter on 'x' {direction} 5" in capsys.readouterr().out


def test_apply_threshold_accepts_quantile_string():
    app = _bare_app()
    df = pd.DataFrame({"x": np.arange(0.0, 10.0)})
    out = app._apply_threshold(df, "x", "q5", "higher")
    # q5 -> the 0.5 quantile of 0..9 == 4.5, so 5..9 survive.
    assert out["x"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


# ===========================================================================
# prefilter_paths_annotations - measurement/threshold branch
# ===========================================================================

def _measure_df(paths, with_png_path=True, with_prcfo_index=False):
    n = len(paths)
    df = pd.DataFrame({
        "prcfo": [f"plate1_A01_1_o{i + 1}" for i in range(n)],
        "cell_area": [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0][:n],
        "nucleus_area": [50.0, 100.0, 150.0, 200.0, 250.0, 300.0][:n],
    })
    if with_png_path:
        df["png_path"] = paths
    if with_prcfo_index:
        df = df.set_index("prcfo")
    return df


def _prefilter_app(annot_env, **attrs):
    base = dict(
        db_path=annot_env["db_path"],
        annotation_column="annotate",
        image_type=None,
        index=0,
        grid_rows=2,
        grid_cols=3,
        measurement=None,
        threshold=None,
        threshold_direction="higher",
    )
    base.update(attrs)
    return _bare_app(**base)


def test_prefilter_scalar_measurement_uses_existing_png_path(annot_env, monkeypatch):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: _measure_df(paths))

    app = _prefilter_app(annot_env, measurement="cell_area",
                         threshold=900, threshold_direction="higher")
    app.prefilter_paths_annotations()

    assert app._total_filtered == 4
    assert [row[0] for row in app.filtered_paths_annotations] == paths[2:]
    # annotation column was blanked before filtering
    assert all(row[1] is None for row in app.filtered_paths_annotations)


def test_prefilter_merges_png_path_when_join_omits_it(annot_env, monkeypatch):
    """The join result carries prcfo only as an index -> reset_index + merge."""
    import spacr.io as sio
    paths = annot_env["png_paths"]

    meas = _measure_df(paths, with_png_path=False, with_prcfo_index=True)
    png_list_df = pd.DataFrame({
        "prcfo": [f"plate1_A01_1_o{i + 1}" for i in range(len(paths))],
        "png_path": paths,
    }).set_index("prcfo")

    monkeypatch.setattr(sio, "_read_and_join_tables", lambda db, *a, **k: meas)
    monkeypatch.setattr(sio, "_read_db", lambda db, tables: [png_list_df])

    app = _prefilter_app(annot_env, measurement="cell_area",
                         threshold=1600, threshold_direction="lower")
    app.prefilter_paths_annotations()

    assert [row[0] for row in app.filtered_paths_annotations] == paths[:4]
    assert app._total_filtered == 4


def test_prefilter_drops_rows_without_a_png_path(annot_env, monkeypatch):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    df = _measure_df(paths)
    df.loc[0, "png_path"] = None          # dropna(subset=['png_path'])
    monkeypatch.setattr(sio, "_read_and_join_tables", lambda db, *a, **k: df)

    app = _prefilter_app(annot_env, measurement="cell_area",
                         threshold=0, threshold_direction="higher")
    app.prefilter_paths_annotations()

    assert len(app.filtered_paths_annotations) == len(paths) - 1
    assert paths[0] not in [r[0] for r in app.filtered_paths_annotations]


def test_prefilter_list_measurement_applies_each_filter(annot_env, monkeypatch, capsys):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: _measure_df(paths))

    app = _prefilter_app(annot_env,
                         measurement=["cell_area", "nucleus_area"],
                         threshold=[400, 250],
                         threshold_direction=["higher", "lower"])
    app.prefilter_paths_annotations()

    # cell_area >= 400 -> rows 1..5 ; nucleus_area <= 250 -> rows 0..4
    assert [r[0] for r in app.filtered_paths_annotations] == paths[1:5]
    out = capsys.readouterr().out
    assert "Filter on 'cell_area' higher 400" in out
    assert "Filter on 'nucleus_area' lower 250" in out


def test_prefilter_list_of_lists_builds_ratio_column(annot_env, monkeypatch, capsys):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: _measure_df(paths))

    app = _prefilter_app(annot_env,
                         measurement=[["cell_area"], ["nucleus_area", "cell_area"]],
                         threshold=[400, 0.2],
                         threshold_direction=["higher", "higher"])
    app.prefilter_paths_annotations()

    ref = _measure_df(paths)
    ref = ref[ref["cell_area"] >= 400]
    ref = ref[(ref["nucleus_area"] / ref["cell_area"]) >= 0.2]
    assert [r[0] for r in app.filtered_paths_annotations] == ref["png_path"].tolist()
    assert "ratio_1_nucleus_area_over_cell_area" in capsys.readouterr().out


def test_prefilter_measurement_branch_honours_image_type_list(annot_env, monkeypatch, capsys):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: _measure_df(paths))

    app = _prefilter_app(annot_env, measurement="cell_area",
                         threshold=0, threshold_direction="higher",
                         image_type=["cell", "img_"])
    app.prefilter_paths_annotations()

    kept = [r[0] for r in app.filtered_paths_annotations]
    assert kept == [p for p in paths if "cell" in p and "img_" in p]
    assert "/nowhere/nucleus_missing.png" not in kept
    out = capsys.readouterr().out
    assert "image_type 'cell'" in out
    assert "image_type filter: removed" in out


def test_prefilter_measurement_branch_quantile_threshold(annot_env, monkeypatch):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    monkeypatch.setattr(sio, "_read_and_join_tables",
                        lambda db, *a, **k: _measure_df(paths))

    app = _prefilter_app(annot_env, measurement="cell_area",
                         threshold="q5", threshold_direction="higher")
    app.prefilter_paths_annotations()

    ref = _measure_df(paths)
    cut = ref["cell_area"].quantile(0.5)
    assert [r[0] for r in app.filtered_paths_annotations] == \
        ref[ref["cell_area"] >= cut]["png_path"].tolist()


def test_prefilter_does_not_require_a_cell_area_column(annot_env, monkeypatch):
    import spacr.io as sio
    paths = annot_env["png_paths"]
    df = pd.DataFrame({
        "prcfo": [f"plate1_A01_1_o{i + 1}" for i in range(len(paths))],
        "png_path": paths,
        "nucleus_area": np.arange(len(paths), dtype=float) * 100.0,
    })
    monkeypatch.setattr(sio, "_read_and_join_tables", lambda db, *a, **k: df)

    app = _prefilter_app(annot_env, measurement="nucleus_area",
                         threshold=100, threshold_direction="higher")
    app.prefilter_paths_annotations()
    assert len(app.filtered_paths_annotations) == len(paths) - 1


# ===========================================================================
# prefilter_paths_annotations - plain png_list paging branch
# ===========================================================================

def test_prefilter_plain_branch_pages_and_counts(annot_env):
    app = _prefilter_app(annot_env, grid_rows=1, grid_cols=2, index=2)
    app.prefilter_paths_annotations()
    assert app._total_filtered == annot_env["n"]
    assert [r[0] for r in app.filtered_paths_annotations] == annot_env["png_paths"][2:4]


def test_prefilter_plain_branch_filters_on_image_type(annot_env):
    app = _prefilter_app(annot_env, grid_rows=2, grid_cols=3, image_type="cell")
    app.prefilter_paths_annotations()

    expected = [p for p in annot_env["png_paths"] if "cell" in p]
    assert app._total_filtered == len(expected)
    assert [r[0] for r in app.filtered_paths_annotations] == expected
    assert "/nowhere/nucleus_missing.png" not in [r[0] for r in app.filtered_paths_annotations]


# ===========================================================================
# fill_holes
# ===========================================================================

def _mask_with_holes():
    """20x20 block with one 1-px hole and one 3x3 (9-px) hole."""
    m = np.zeros((24, 24), dtype=bool)
    m[2:22, 2:22] = True
    m[5, 5] = False                      # area 1
    m[12:15, 12:15] = False              # area 9
    return m


def test_fill_holes_fills_everything_when_min_size_is_zero():
    from spacr.gui_elements import AnnotateApp
    m = _mask_with_holes()
    out = AnnotateApp.fill_holes(m, min_size=0)
    assert out.dtype == bool
    assert out[5, 5]
    assert out[13, 13]
    assert out.sum() == 20 * 20


def test_fill_holes_reopens_holes_at_or_above_min_size():
    from spacr.gui_elements import AnnotateApp
    out = AnnotateApp.fill_holes(_mask_with_holes(), min_size=5)
    assert out[5, 5], "1-px hole is below min_size and must stay filled"
    assert not out[13, 13], "9-px hole is >= min_size and must be re-opened"
    assert out.sum() == 20 * 20 - 9


def test_fill_holes_with_no_holes_returns_filled_mask():
    from spacr.gui_elements import AnnotateApp
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    out = AnnotateApp.fill_holes(m, min_size=3)
    assert np.array_equal(out, m)


def test_fill_holes_accepts_non_boolean_input():
    from spacr.gui_elements import AnnotateApp
    m = (_mask_with_holes() * 255).astype(np.uint8)
    out = AnnotateApp.fill_holes(m, min_size=0)
    assert out.dtype == bool and out.sum() == 400


# ===========================================================================
# _filter_objects_by_area
# ===========================================================================

def _two_blobs():
    m = np.zeros((30, 30), dtype=bool)
    m[2:5, 2:5] = True      # area 9
    m[10:20, 10:20] = True  # area 100
    return m


def test_filter_objects_by_area_empty_mask_short_circuits():
    from spacr.gui_elements import AnnotateApp
    m = np.zeros((8, 8), dtype=bool)
    out = AnnotateApp._filter_objects_by_area(m, min_size=1, max_size=10)
    assert out.dtype == bool and not out.any()


def test_filter_objects_by_area_drops_small_objects():
    from spacr.gui_elements import AnnotateApp
    out = AnnotateApp._filter_objects_by_area(_two_blobs(), min_size=50, max_size=0)
    assert out.sum() == 100
    assert not out[3, 3] and out[15, 15]


def test_filter_objects_by_area_drops_large_objects():
    from spacr.gui_elements import AnnotateApp
    out = AnnotateApp._filter_objects_by_area(_two_blobs(), min_size=0, max_size=50)
    assert out.sum() == 9
    assert out[3, 3] and not out[15, 15]


def test_filter_objects_by_area_guard_when_labeller_reports_no_objects(monkeypatch):
    """The ``n == 0`` guard is only reachable if ``label`` disagrees with
    ``mask.any()``; inject that disagreement so the fallback is exercised."""
    import scipy.ndimage as ndi
    from spacr.gui_elements import AnnotateApp

    m = _two_blobs()
    monkeypatch.setattr(ndi, "label", lambda a, *args, **kw: (np.zeros_like(a, dtype=int), 0))
    out = AnnotateApp._filter_objects_by_area(m, min_size=1, max_size=2)
    assert np.array_equal(out, m), "mask must pass through untouched"


def test_filter_objects_by_area_zero_bounds_keeps_all():
    from spacr.gui_elements import AnnotateApp
    m = _two_blobs()
    out = AnnotateApp._filter_objects_by_area(m, min_size=0, max_size=0)
    assert np.array_equal(out, m)


# ===========================================================================
# normalize_image
# ===========================================================================

def test_normalize_image_is_a_noop_without_channels():
    from spacr.gui_elements import AnnotateApp
    arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    out = AnnotateApp.normalize_image(_rgb(arr), (1, 99), None)
    assert np.array_equal(np.array(out), arr)


def test_normalize_image_clips_out_of_range_values():
    from spacr.gui_elements import AnnotateApp
    arr = np.array([[-40.0, 300.0], [10.0, 20.0]])
    out = np.array(AnnotateApp.normalize_image(arr, (1, 99), None))
    assert out.max() <= 255 and out.min() >= 0
    assert out[0, 0] == 0 and out[0, 1] == 255


def test_normalize_image_two_dimensional_percentile_stretch():
    from spacr.gui_elements import AnnotateApp
    arr = np.linspace(40, 90, 64).reshape(8, 8)
    out = np.array(AnnotateApp.normalize_image(arr, (1, 99), ["r"]))
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8
    # the stretch must widen the dynamic range of the original 40..90 band
    assert out.max() - out.min() > (arr.max() - arr.min())


def test_normalize_image_stretches_only_requested_channels():
    from spacr.gui_elements import AnnotateApp
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(40, 90, 64).reshape(8, 8).astype(np.uint8)
    arr[:, :, 1] = np.linspace(40, 90, 64).reshape(8, 8).astype(np.uint8)
    out = np.array(AnnotateApp.normalize_image(_rgb(arr), (1, 99), ["r"]))
    assert out[:, :, 0].max() > arr[:, :, 0].max()
    assert np.array_equal(out[:, :, 1], arr[:, :, 1]), "green must be untouched"
    assert out[:, :, 2].max() == 0


def test_normalize_image_ignores_unknown_channel_names():
    from spacr.gui_elements import AnnotateApp
    arr = np.zeros((6, 6, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(36).reshape(6, 6)
    out = np.array(AnnotateApp.normalize_image(_rgb(arr), (1, 99), ["x", "zz"]))
    assert np.array_equal(out, arr), "no valid channel name -> nothing rescaled"


# ===========================================================================
# filter_channels / add_colored_border
# ===========================================================================

def _mixed_rgb():
    arr = np.zeros((6, 6, 3), dtype=np.uint8)
    arr[:, :, 0] = 10
    arr[:, :, 1] = 20
    arr[:, :, 2] = 30
    return arr


def test_filter_channels_keeps_everything_when_unset():
    app = _bare_app(channels=None)
    arr = _mixed_rgb()
    out = np.array(app.filter_channels(_rgb(arr)))
    assert np.array_equal(out, arr)


def test_filter_channels_zeroes_unselected_channels():
    app = _bare_app(channels=["r"])
    out = np.array(app.filter_channels(_rgb(_mixed_rgb())))
    assert out[:, :, 0].max() == 10
    assert out[:, :, 1].max() == 0
    assert out[:, :, 2].max() == 0


def test_filter_channels_can_drop_the_red_channel():
    app = _bare_app(channels=["b"])
    out = np.array(app.filter_channels(_rgb(_mixed_rgb())))
    assert out[:, :, 0].max() == 0
    assert out[:, :, 1].max() == 0
    assert out[:, :, 2].max() == 30


def test_filter_channels_sanitises_messy_channel_lists():
    app = _bare_app(channels=["R", "  g ", None, ""])
    out = np.array(app.filter_channels(_rgb(_mixed_rgb())))
    assert out[:, :, 0].max() == 10
    assert out[:, :, 1].max() == 20
    assert out[:, :, 2].max() == 0


def test_add_colored_border_frames_the_image():
    app = _bare_app(fg_color="#000000")
    img = _rgb(np.full((10, 12, 3), 200, dtype=np.uint8))
    out = app.add_colored_border(img, border_width=3, border_color="#ff0000")
    assert out.size == (12 + 6, 10 + 6)
    assert out.getpixel((6, 1)) == (255, 0, 0)     # top border strip
    assert out.getpixel((6, 6)) == (200, 200, 200)  # original content
    assert out.getpixel((0, 0)) == (0, 0, 0)        # corner keeps fg_color


# ===========================================================================
# outline_image
# ===========================================================================

def _outline_source():
    """64x64 RGB with a bright red/green disk on a dark field."""
    yy, xx = np.mgrid[:64, :64]
    disk = (yy - 32) ** 2 + (xx - 32) ** 2 <= 15 ** 2
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[..., 0] = np.where(disk, 220, 15)
    arr[..., 1] = np.where(disk, 190, 10)
    arr[..., 2] = 20
    return arr


def _outline_app(**attrs):
    base = dict(outline=["r"], edge_transparency=100.0, edge_image=True,
                outline_threshold_factor=1.0)
    base.update(attrs)
    return _bare_app(**base)


def test_outline_image_returns_input_for_non_rgb():
    from PIL import Image
    app = _outline_app()
    gray = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
    assert app.outline_image(gray, gray) is gray


def test_outline_image_returns_early_when_fully_transparent():
    app = _outline_app(edge_transparency=0.0)
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr)))
    assert np.array_equal(out, arr)


def test_outline_image_returns_early_without_valid_channels():
    app = _outline_app(outline=["q"])
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr)))
    assert np.array_equal(out, arr)


def test_outline_image_blanks_channel_when_underlay_disabled_and_transparent():
    app = _outline_app(edge_image=False, edge_transparency=0.0)
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr)))
    assert out[:, :, 0].max() == 0, "red channel must be wiped"
    assert np.array_equal(out[:, :, 1], arr[:, :, 1])


def test_outline_image_draws_outline_on_selected_channels():
    app = _outline_app(outline=["r", "g"])
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr), edge_sigma=1,
                                     edge_thickness=1.0))
    assert out.shape == arr.shape and out.dtype == np.uint8
    # a fully opaque outline reaches 255 in both requested channels
    assert out[:, :, 0].max() == 255 and out[:, :, 1].max() == 255
    assert np.array_equal(out[:, :, 2], arr[:, :, 2]), "blue is untouched"
    # the underlay is preserved: the disk interior keeps its original value
    assert out[32, 32, 0] == arr[32, 32, 0]
    # and the bright ring sits on the disk boundary (radius ~15), not the centre
    ring = np.argwhere(out[:, :, 0] >= 250)
    radii = np.hypot(ring[:, 0] - 32, ring[:, 1] - 32)
    assert 10 < radii.min() and radii.max() < 20


def test_outline_image_without_underlay_keeps_base_dark_inside():
    app = _outline_app(edge_image=False)
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr), edge_thickness=1.0))
    # centre of the disk was zeroed and never re-filled from full_img
    assert out[32, 32, 0] == 0
    assert out[:, :, 0].max() > 200, "ring still drawn"


def test_outline_image_survives_bad_object_size(capsys):
    app = _outline_app()
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr), object_size=5))
    assert out.shape == arr.shape
    assert out[:, :, 0].max() > 200


def test_outline_image_applies_object_area_filter():
    """A min area larger than the disk removes the object -> no ring drawn."""
    app = _outline_app(edge_image=False)   # channel wiped, so only the ring shows
    arr = _outline_source()
    keep = np.array(app.outline_image(_rgb(arr), _rgb(arr), object_size=(10, 0)))
    drop = np.array(app.outline_image(_rgb(arr), _rgb(arr), object_size=(60000, 0)))
    assert keep[:, :, 0].max() == 255
    assert drop[:, :, 0].max() == 0, "object filtered out -> nothing to outline"


def test_outline_image_without_hole_filling_leaves_the_hole_outlined():
    arr = _outline_source()
    arr[28:36, 28:36, 0] = 0             # punch a hole in the middle of the disk
    app = _outline_app(edge_image=False)
    filled = np.array(app.outline_image(_rgb(arr), _rgb(arr), fill_holes=True))
    unfilled = np.array(app.outline_image(_rgb(arr), _rgb(arr), fill_holes=False))
    assert filled[26:38, 26:38, 0].max() == 0, "hole filled -> no inner boundary"
    assert unfilled[26:38, 26:38, 0].max() == 255, "hole keeps its own boundary"
    assert unfilled[:, :, 0].sum() > filled[:, :, 0].sum()


def test_outline_image_falls_back_when_otsu_fails(monkeypatch):
    import skimage.filters as skf

    def _boom(*a, **k):
        raise ValueError("injected otsu failure")

    monkeypatch.setattr(skf, "threshold_otsu", _boom)
    app = _outline_app(edge_image=False)
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr)))
    assert out[:, :, 0].max() == 255, "median fallback still yields a boundary"


def test_outline_image_leaves_channel_untouched_when_nothing_thresholds():
    """A huge threshold factor clamps to 255 -> empty mask -> alpha peak 0."""
    app = _outline_app(outline_threshold_factor=1e9)
    arr = _outline_source()
    out = np.array(app.outline_image(_rgb(arr), _rgb(arr)))
    assert np.array_equal(out[:, :, 0], arr[:, :, 0])


def test_outline_image_zero_thickness_skips_dilation():
    app = _outline_app(edge_image=False)
    arr = _outline_source()
    thin = np.array(app.outline_image(_rgb(arr), _rgb(arr), edge_thickness=0.0))
    thick = np.array(app.outline_image(_rgb(arr), _rgb(arr), edge_thickness=2.0))
    assert (thin[:, :, 0] > 0).sum() > 0
    assert (thick[:, :, 0] > 0).sum() > (thin[:, :, 0] > 0).sum()


# ===========================================================================
# load_single_image
# ===========================================================================

def _loader_app(annot_env, **attrs):
    base = dict(
        image_size=(32, 32),
        percentiles=(1, 99),
        normalize_channels=None,
        channels=None,
        outline=None,
        outline_sigma=1.0,
        edge_thickness=1.0,
        edge_transparency=100.0,
        edge_image=True,
        outline_threshold_factor=1.0,
        object_size=(0, 0),
    )
    base.update(attrs)
    return _bare_app(**base)


def test_load_single_image_returns_blank_for_missing_file(annot_env, capsys):
    app = _loader_app(annot_env)
    img, annotation = app.load_single_image(("/does/not/exist.png", 7))
    assert img.size == (32, 32)
    assert img.getpixel((0, 0)) == (30, 30, 30)
    assert annotation == 7
    assert "Could not find image" in capsys.readouterr().out


def test_load_single_image_resizes_and_returns_rgb(annot_env):
    app = _loader_app(annot_env, normalize_channels=["r"], channels=["r", "g"])
    img, annotation = app.load_single_image((annot_env["on_disk"][0], None))
    assert img.size == (32, 32)
    assert img.mode == "RGB"
    assert annotation is None
    arr = np.array(img)
    assert arr[:, :, 2].max() == 0, "blue filtered out"
    assert arr[:, :, 0].max() > 0


def test_load_single_image_applies_outline(annot_env):
    plain = _loader_app(annot_env, channels=["r"])
    outlined = _loader_app(annot_env, channels=["r"], outline=["g"],
                           object_size=(0, 0))
    a = np.array(plain.load_single_image((annot_env["on_disk"][1], 1))[0])
    b = np.array(outlined.load_single_image((annot_env["on_disk"][1], 1))[0])
    assert a[:, :, 1].max() == 0, "green filtered away without outlines"
    assert b[:, :, 1].max() > 0, "outline re-lights the green channel"


# ===========================================================================
# update_html
# ===========================================================================

def test_update_html_pushes_script_through_ipython_display(monkeypatch):
    import spacr.gui_elements as ge
    seen = []
    monkeypatch.setattr(ge, "display", lambda obj: seen.append(obj))
    ge.AnnotateApp.update_html("hello world")
    assert len(seen) == 1
    payload = getattr(seen[0], "data", str(seen[0]))
    assert "hello world" in payload
    assert "unique_id" in payload


# ===========================================================================
# update_database_worker defensive branches
# ===========================================================================

def test_update_database_worker_survives_pragma_failure(tmp_path, monkeypatch):
    """A driver that rejects PRAGMA must not kill the writer thread."""
    import sqlite3 as _sqlite3

    class _Cur:
        def __init__(self):
            self.sql = []
            self.closed = False

        def execute(self, sql, *a):
            self.sql.append(sql)
            raise _sqlite3.OperationalError("PRAGMA rejected")

        def close(self):
            self.closed = True

    class _Conn:
        def __init__(self):
            self.cur = _Cur()
            self.closed = False

        def cursor(self):
            return self.cur

        def commit(self):
            raise AssertionError("commit must not be reached")

        def close(self):
            self.closed = True

    conn = _Conn()
    monkeypatch.setattr(_sqlite3, "connect", lambda *a, **k: conn)

    app = _bare_app(db_path=str(tmp_path / "x.db"), terminate=False,
                    SENTINEL=object(), update_queue=_queue.Queue())
    app.update_queue.put(app.SENTINEL)
    app.update_database_worker()

    assert conn.cur.sql[0].startswith("PRAGMA journal_mode")
    assert conn.cur.closed and conn.closed
    assert app.update_queue.unfinished_tasks == 0


def test_update_database_worker_exits_on_terminate_flag(annot_env):
    """Empty queue + terminate -> the worker writes its batch then breaks out."""
    import threading

    target = annot_env["png_paths"][0]
    with sqlite3.connect(annot_env["db_path"]) as con:
        con.execute('ALTER TABLE "png_list" ADD COLUMN "annotate" INTEGER')

    app = _bare_app(db_path=annot_env["db_path"], annotation_column="annotate",
                    terminate=False, SENTINEL=object(),
                    update_queue=_queue.Queue(), worker_busy=False,
                    _unsaved_batches=1, _batch_lock=threading.Lock(),
                    _last_save_ts=None)
    app.update_queue.put({target: 1, annot_env["png_paths"][1]: None})
    app.terminate = True
    app.update_database_worker()

    with sqlite3.connect(annot_env["db_path"]) as con:
        rows = dict(con.execute('SELECT png_path, "annotate" FROM "png_list"').fetchall())
    assert rows[target] == 1
    assert rows[annot_env["png_paths"][1]] is None
    assert app.worker_busy is False
    assert app._last_save_ts is not None


# ===========================================================================
# Widget-driven paths (need a display; tk_root skips when there is none)
# ===========================================================================

def test_load_images_paints_labels_and_rewrites_paths(make_app, annot_env, capsys):
    app = make_app()
    app.measurement = None
    app.threshold = None
    app.filtered_paths_annotations = [(p, None) for p in annot_env["png_paths"]]
    app.index = 0
    app.load_images()

    # /foreign/root/data/... was rewritten onto the real file under src
    rewritten = os.path.join(annot_env["src"], "data", "cell_png", "img_4.png")
    assert app.adjusted_to_original_paths[rewritten] == "/foreign/root/data/cell_png/img_4.png"
    # one image per label of the (2 x 3) grid
    assert len(app.images) == 6
    assert set(app.images) <= set(app.labels)
    # the path that carried neither src nor /data/ was left alone and missing
    assert "Could not find image" in capsys.readouterr().out
    for lab in app.labels[:6]:
        assert lab.cget("image") != ""


def test_load_images_draws_borders_for_annotated_tiles(make_app, annot_env):
    app = make_app()
    app.measurement = None
    app.threshold = None
    app.filtered_paths_annotations = [
        (annot_env["png_paths"][0], 1),
        (annot_env["png_paths"][1], None),
    ]
    app.index = 0
    app.load_images()

    border, plain = app.labels[0], app.labels[1]
    assert app.images[border].width() == app.image_size[0] + 10
    assert app.images[plain].width() == app.image_size[0]


def test_load_images_pages_when_a_measurement_filter_is_active(make_app, annot_env):
    app = make_app()
    app.measurement = "cell_area"
    app.threshold = 0
    app.filtered_paths_annotations = [(p, None) for p in annot_env["on_disk"]] * 3
    app.index = 6
    app.load_images()
    assert len(app.images) == app.grid_rows * app.grid_cols


def test_get_on_image_click_toggles_annotation(make_app, annot_env):
    from PIL import Image
    app = make_app()
    label = app.labels[0]
    img = Image.open(annot_env["on_disk"][0]).convert("RGB").resize((60, 60))
    path = annot_env["on_disk"][0]
    handler = app.get_on_image_click(path, label, img)

    handler(types.SimpleNamespace(num=1))
    assert app.pending_updates[path] == 1
    assert app.images[label].width() == 60          # cropped 50 + 2*5 border

    handler(types.SimpleNamespace(num=1))           # same button clears it
    assert app.pending_updates[path] is None
    assert app.images[label].width() == 50          # no border re-applied

    handler(types.SimpleNamespace(num=3))           # right click -> class 2
    assert app.pending_updates[path] == 2


def test_get_on_image_click_uses_the_original_db_path(make_app, annot_env):
    from PIL import Image
    app = make_app()
    label = app.labels[0]
    img = Image.open(annot_env["on_disk"][0]).convert("RGB").resize((60, 60))
    local = os.path.join(annot_env["src"], "data", "cell_png", "img_4.png")
    original = "/foreign/root/data/cell_png/img_4.png"
    app.adjusted_to_original_paths[local] = original

    app.get_on_image_click(local, label, img)(types.SimpleNamespace(num=1))
    assert app.pending_updates == {original: 1}
    assert local not in app.pending_updates


def test_show_class_counts_lists_classes_and_skips_non_integers(make_app, annot_env):
    from tkinter import ttk
    app = make_app()
    with sqlite3.connect(annot_env["db_path"]) as con:
        con.execute('UPDATE "png_list" SET "annotate" = 1 WHERE png_path = ?',
                    (annot_env["png_paths"][0],))
        con.execute('UPDATE "png_list" SET "annotate" = 2 WHERE png_path = ?',
                    (annot_env["png_paths"][1],))
        con.execute('UPDATE "png_list" SET "annotate" = ? WHERE png_path = ?',
                    ("not_an_int", annot_env["png_paths"][2]))

    before = set(app.root.winfo_children())
    app.show_class_counts()
    new = [w for w in app.root.winfo_children() if w not in before]
    assert len(new) == 1
    win = new[0]
    assert win.title() == "Class counts (all)"

    def _walk(w):
        for c in w.winfo_children():
            yield c
            yield from _walk(c)

    trees = [w for w in _walk(win) if isinstance(w, ttk.Treeview)]
    assert len(trees) == 1
    values = [trees[0].item(i, "values") for i in trees[0].get_children()]
    assert [v[0] for v in values] == ["1", "2"], "the text label must be skipped"
    assert [v[1] for v in values] == ["1", "1"]
    assert values[0][2] == "#1f77b4" and values[1][2] == "#d62728"
    win.destroy()


def test_show_class_counts_errors_without_an_annotation_column(make_app, monkeypatch):
    import tkinter.messagebox as mb
    app = make_app()
    app.annotation_column = None
    seen = []
    monkeypatch.setattr(mb, "showerror", lambda *a, **k: seen.append(a))

    before = set(app.root.winfo_children())
    app.show_class_counts()
    assert seen and "No annotation column" in seen[0][1]
    assert set(app.root.winfo_children()) == before, "no window should open"


def test_clear_current_annotation_is_cancellable(make_app, annot_env, monkeypatch):
    import tkinter.messagebox as mb
    app = make_app()
    with sqlite3.connect(annot_env["db_path"]) as con:
        con.execute('UPDATE "png_list" SET "annotate" = 1')
    app.pending_updates = {"keep": 1}

    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: False)
    app.clear_current_annotation()

    with sqlite3.connect(annot_env["db_path"]) as con:
        vals = [r[0] for r in con.execute('SELECT "annotate" FROM "png_list"')]
    assert vals == [1] * annot_env["n"]
    assert app.pending_updates == {"keep": 1}


def test_clear_current_annotation_nulls_the_column(make_app, annot_env, monkeypatch):
    import tkinter.messagebox as mb
    app = make_app()
    with sqlite3.connect(annot_env["db_path"]) as con:
        con.execute('UPDATE "png_list" SET "annotate" = 2')
    app.pending_updates = {annot_env["png_paths"][0]: 2}
    app._unsaved_batches = 3

    monkeypatch.setattr(mb, "askyesno", lambda *a, **k: True)
    app.clear_current_annotation()

    with sqlite3.connect(annot_env["db_path"]) as con:
        vals = [r[0] for r in con.execute('SELECT "annotate" FROM "png_list"')]
    assert vals == [None] * annot_env["n"]
    assert app.pending_updates == {}
    assert app._unsaved_batches == 0
    assert len(app.filtered_paths_annotations) > 0


def test_calculate_grid_dimensions_never_returns_zero(make_app):
    app = make_app()
    app.calculate_grid_dimensions()
    assert app.grid_rows >= 1 and app.grid_cols >= 1
    assert isinstance(app.grid_rows, int) and isinstance(app.grid_cols, int)
