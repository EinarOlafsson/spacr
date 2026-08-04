"""CPU-only coverage for the Cellpose *apply* block of ``spacr.submodules``.

Covers ``apply_cellpose_model`` (and its nested ``plot_cellpose_result``),
``plot_cellpose_batch`` and ``analyze_percent_positive`` (and its nested
``translate_well_in_df`` / ``annotate_and_summarize``) -- submodules.py
lines ~455-709.

Nothing here touches a GPU or a real Cellpose network: the
``cp_models.CellposeModel`` symbol inside ``spacr.submodules`` is swapped for
a recording double that returns deterministic label masks and flow fields, so
every downstream branch (circularize / save / regionprops / CSV writing) runs
for real on tiny 64x64 arrays.  ``analyze_percent_positive`` runs against a
genuine synthetic ``measurements.db`` through ``io._read_and_merge_data``.
"""
from __future__ import annotations

import os
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from tests.cellpose_api_contract import (  # noqa: E402
    DEPRECATED_EVAL_ARGUMENTS,
    MISSING_CHANNEL_AXIS,
    configured_eval_arguments,
    emulate_pretrained_model,
    eval_arguments,
    init_arguments,
)
from tests.conftest import check_cellpose_eval_call  # noqa: E402


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _no_blocking_show(monkeypatch):
    """Never let a product-code ``plt.show()`` reach a real backend."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    return calls


IMG_SIZE = 64


# ---------------------------------------------------------------------------
# synthetic Cellpose doubles
# ---------------------------------------------------------------------------

def _label_mask(shape=(IMG_SIZE, IMG_SIZE), include_corner=True):
    """Deterministic 4-object label mask.

    Three squares sit inside the inscribed circle of the frame, one sits in
    the top-left corner *outside* it -- so ``circularize`` provably drops
    exactly one object.

    Areas: 100, 100, 64 (inside) and 25 (corner).
    """
    m = np.zeros(shape, dtype=np.int32)
    m[10:20, 10:20] = 1      # 100 px, inside the circle
    m[28:38, 28:38] = 2      # 100 px, dead centre
    m[44:52, 20:28] = 3      # 64 px, inside
    if include_corner:
        m[0:5, 0:5] = 4      # 25 px, outside the inscribed circle
    return m


def _empty_mask(shape=(IMG_SIZE, IMG_SIZE)):
    return np.zeros(shape, dtype=np.int32)


def _flow(shape=(IMG_SIZE, IMG_SIZE)):
    """Cellpose-shaped flow triple: [rgb_flow, dP, cellprob]."""
    h, w = shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    dP = np.zeros((2, h, w), dtype=np.float32)
    cellprob = np.linspace(-3.0, 3.0, h * w, dtype=np.float32).reshape(h, w)
    return [rgb, dP, cellprob]


def _install_fake_model(monkeypatch, mask_fn=None):
    """Replace ``submodules.cp_models`` with a namespace holding a fake model.

    Returns a record dict capturing constructor args and every ``eval`` call.
    """
    from spacr import submodules as SUB

    if mask_fn is None:
        mask_fn = _label_mask
    record = {"instances": [], "init_args": [], "eval_calls": [],
              "eval_configured": []}

    class _FakeCellposeModel:
        """``CellposeModel`` double declaring the installed 4.0.7 signatures.

        No ``**kwargs`` anywhere: ``submodules.apply_cellpose_model`` is a real
        call site, so an argument cellpose 4 removed raises ``TypeError`` here
        rather than being swallowed. ``eval`` returns the three values 4.0.7
        returns, which is also the arity the caller unpacks.
        """

        def __init__(self, gpu=False, pretrained_model="cpsam",
                     model_type=None, diam_mean=None, device=None, nchan=None,
                     use_bfloat16=True):
            record["init_args"].append(init_arguments(locals()))
            record["instances"].append(
                {"gpu": gpu, "pretrained_model": pretrained_model}
            )
            self.loaded_model = emulate_pretrained_model(pretrained_model,
                                                         model_type)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            # This loop deliberately leaves the axis to cellpose: it hands over
            # plain 2-D planes. Whatever arrives must still be one
            # convert_image accepts.
            check_cellpose_eval_call(x, channel_axis,
                                     require_channel_axis=False)
            bound = locals()
            record["eval_configured"].append(configured_eval_arguments(bound))
            record["eval_calls"].append({"x": list(x),
                                         **eval_arguments(bound)})
            n = len(x)
            return (
                [mask_fn() for _ in range(n)],
                [_flow() for _ in range(n)],
                [None] * n,
            )

    monkeypatch.setattr(
        SUB, "cp_models", types.SimpleNamespace(CellposeModel=_FakeCellposeModel)
    )
    monkeypatch.setattr(SUB, "_cellpose_use_gpu", lambda: True)
    return record


def _write_images(dirpath, n=3, size=IMG_SIZE, seed=7):
    """Write ``n`` small uint16 TIFFs and return their paths."""
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(n):
        arr = rng.integers(100, 4000, size=(size, size)).astype(np.uint16)
        p = os.path.join(str(dirpath), f"img_{k:02d}.tif")
        try:
            import tifffile
            tifffile.imwrite(p, arr)
        except Exception:  # pragma: no cover - tifffile is a hard dep of spacr
            from PIL import Image
            Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _apply_settings(src, **over):
    s = {
        "src": str(src),
        "model_path": "cyto_fake",
        "batch_size": 2,
        "FT": 0.4,
        "CP_probability": 0.5,
        "circularize": False,
        "save": False,
        "normalize": True,
        "percentiles": (2, 98),
        "target_size": IMG_SIZE,
    }
    s.update(over)
    return s


# ===========================================================================
# apply_cellpose_model
# ===========================================================================

def test_apply_cellpose_model_writes_measurements_and_summary(tmp_path, monkeypatch):
    """Happy path, save=False / circularize=False.

    Asserts the eval() contract, the per-object measurement rows and the
    per-image summary aggregation.
    """
    from spacr.submodules import apply_cellpose_model

    src = tmp_path / "apply"
    src.mkdir()
    _write_images(src, n=3)
    record = _install_fake_model(monkeypatch)

    settings = _apply_settings(src)
    assert apply_cellpose_model(settings) is None

    # -- the model was constructed from settings['model_path'] -------------
    assert record["instances"] == [{"gpu": True, "pretrained_model": "cyto_fake"}]

    # -- 3 images / batch_size 2 -> two eval calls of 2 and 1 images -------
    assert [len(c["x"]) for c in record["eval_calls"]] == [2, 1]
    first = record["eval_calls"][0]
    # What spaCR passes today. See
    # test_apply_cellpose_model_does_not_pass_a_dead_channels_pair below for
    # what cellpose 4 does with it.
    assert first["channels"] == [0, 0]
    assert first["normalize"] is False
    assert first["diameter"] == 30
    assert first["flow_threshold"] == settings["FT"]
    assert first["cellprob_threshold"] == settings["CP_probability"]
    assert first["rescale"] is None
    assert first["resample"] is True
    assert first["anisotropy"] is None
    assert first["min_size"] == 5
    assert first["augment"] is True
    assert first["tile_overlap"] == 0.2
    assert first["bsize"] == 224
    # the dataset hands cellpose float32 arrays resized to target_size
    for arr in first["x"]:
        assert arr.shape == (IMG_SIZE, IMG_SIZE)
        assert arr.dtype == np.float32

    results_dir = src / "results"
    meas = pd.read_csv(results_dir / "measurements.csv")
    assert list(meas.columns) == ["image", "object_id", "area"]
    # 3 images x 4 objects each
    assert len(meas) == 12
    assert set(meas["image"]) == {"img_00.tif", "img_01.tif", "img_02.tif"}
    for name, grp in meas.groupby("image"):
        assert sorted(grp["object_id"]) == [1, 2, 3, 4]
        assert sorted(grp["area"]) == [25, 64, 100, 100]

    summary = pd.read_csv(results_dir / "summary.csv")
    assert list(summary.columns) == ["image", "object_count", "average_area"]
    assert len(summary) == 3
    assert set(summary["object_count"]) == {4}
    assert np.allclose(summary["average_area"], (100 + 100 + 64 + 25) / 4)

    # save=False -> no diagnostic PNGs
    assert list(results_dir.glob("*.png")) == []
    # settings snapshot written by save_settings
    assert (src / "settings" / "apply_cellpose_model.csv").is_file()


@pytest.mark.xfail(strict=True, reason=(
    "spacr/submodules.py:621 passes channels=[0, 0] to CellposeModel.eval. "
    "cellpose 4.0.7 logs 'channels deprecated in v4.0.1+. If data contain "
    "more than 3 channels, only the first 3 channels will be used' and never "
    "reads the value, so the pair configures nothing -- it is a Cellpose 3 "
    "leftover the migration missed, and spacr.model_compare.IGNORED_ARGUMENTS "
    "already lists 'channels' as exactly this no-op. Fix: delete the "
    "channels=[0, 0] argument and select channels before handing the image "
    "over, as spacr/object.py:1913 already does."))
def test_apply_cellpose_model_does_not_pass_a_dead_channels_pair(
    tmp_path, monkeypatch, _no_blocking_show
):
    """No argument the installed cellpose silently discards may be passed.

    Passing one is not cosmetic: it reads as configuration in the settings UI
    and in the saved settings CSV, so a user tuning ``channels`` believes they
    are steering segmentation while nothing downstream changes.
    """
    from spacr.submodules import apply_cellpose_model

    src = tmp_path / "apply"
    src.mkdir()
    _write_images(src, n=1)
    record = _install_fake_model(monkeypatch)
    apply_cellpose_model(_apply_settings(src))

    dead = sorted(set(record["eval_configured"][0]) & set(DEPRECATED_EVAL_ARGUMENTS))
    assert not dead, (
        "cellpose 4 accepts and then discards: "
        + ", ".join(f"{name}={record['eval_configured'][0][name]!r}"
                    for name in dead)
    )


def test_apply_cellpose_model_circularize_drops_outside_objects_and_plots(
    tmp_path, monkeypatch, _no_blocking_show
):
    """circularize=True masks predictions to the inscribed circle and
    save=True renders one 4-panel PNG per image."""
    from spacr.submodules import apply_cellpose_model

    src = tmp_path / "apply_circ"
    src.mkdir()
    _write_images(src, n=3)
    _install_fake_model(monkeypatch)

    settings = _apply_settings(src, circularize=True, save=True)
    apply_cellpose_model(settings)

    results_dir = src / "results"
    meas = pd.read_csv(results_dir / "measurements.csv")
    # the corner object (25 px) is entirely outside the inscribed circle
    assert len(meas) == 9
    for _, grp in meas.groupby("image"):
        assert sorted(grp["area"]) == [64, 100, 100]
        assert sorted(grp["object_id"]) == [1, 2, 3]

    summary = pd.read_csv(results_dir / "summary.csv")
    assert set(summary["object_count"]) == {3}
    assert np.allclose(summary["average_area"], (100 + 100 + 64) / 3)

    # one diagnostic figure per image, named cellpose_result_<i+j>.png
    pngs = sorted(p.name for p in results_dir.glob("*.png"))
    assert pngs == [
        "cellpose_result_000.png",
        "cellpose_result_001.png",
        "cellpose_result_002.png",
    ]
    for p in results_dir.glob("*.png"):
        assert p.stat().st_size > 0
    # plot_cellpose_result called plt.show() exactly once per image
    assert len(_no_blocking_show) == 3
    # ...and closed every figure it made
    assert plt.get_fignums() == []


def test_apply_cellpose_model_handles_batch_with_no_objects(tmp_path, monkeypatch):
    """A prediction with zero objects must still produce (empty) result CSVs
    rather than blowing up the whole run."""
    from spacr.submodules import apply_cellpose_model

    src = tmp_path / "apply_empty"
    src.mkdir()
    _write_images(src, n=2)
    _install_fake_model(monkeypatch, mask_fn=_empty_mask)

    apply_cellpose_model(_apply_settings(src, batch_size=2))

    results_dir = src / "results"
    assert (results_dir / "measurements.csv").is_file()
    assert (results_dir / "summary.csv").is_file()
    assert len(pd.read_csv(results_dir / "summary.csv")) == 0


def test_apply_cellpose_model_partial_empty_prediction_is_recorded(
    tmp_path, monkeypatch
):
    """When only *some* images are empty the run completes and the empty
    image simply contributes no measurement rows."""
    from spacr.submodules import apply_cellpose_model

    src = tmp_path / "apply_mixed"
    src.mkdir()
    _write_images(src, n=2)

    seq = iter([_label_mask(), _empty_mask()])
    _install_fake_model(monkeypatch, mask_fn=lambda: next(seq))

    apply_cellpose_model(_apply_settings(src, batch_size=2))

    meas = pd.read_csv(src / "results" / "measurements.csv")
    # only the first image contributed objects
    assert set(meas["image"]) == {"img_00.tif"}
    assert len(meas) == 4
    summary = pd.read_csv(src / "results" / "summary.csv")
    assert len(summary) == 1
    assert summary.loc[0, "object_count"] == 4


# ===========================================================================
# plot_cellpose_batch
# ===========================================================================

def test_plot_cellpose_batch_builds_two_row_grid(_no_blocking_show):
    """Two rows (images / labels) with titled, axis-less panels."""
    from spacr.submodules import plot_cellpose_batch

    images = [
        np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32),
        np.zeros((32, 32), dtype=np.float32),
    ]
    labels = [_label_mask((32, 32), include_corner=False) for _ in range(2)]

    plot_cellpose_batch(images, labels)

    assert len(_no_blocking_show) == 1
    fig = plt.gcf()
    axs = fig.axes
    assert len(axs) == 4
    assert [ax.get_title() for ax in axs] == [
        "Image 1", "Image 2", "Label 1", "Label 2",
    ]
    assert all(ax.axison is False for ax in axs)
    # the top row shows the images, the bottom row the labels
    np.testing.assert_allclose(axs[0].images[0].get_array(), images[0])
    np.testing.assert_array_equal(axs[2].images[0].get_array(), labels[0])
    # labels are drawn with the random mask colormap (1 colour per label + bg)
    cmap = axs[2].images[0].get_cmap()
    n_labels = len(np.unique(np.asarray(labels))) - 1
    assert cmap.N == n_labels + 1
    assert tuple(cmap.colors[0]) == (0.0, 0.0, 0.0, 1.0)
    # images are drawn in grayscale, interpolation nearest for the masks
    assert axs[0].images[0].get_cmap().name == "gray"
    assert axs[2].images[0].get_interpolation() == "nearest"


def test_plot_cellpose_batch_single_image(_no_blocking_show):
    """A batch of one is a legal batch and must render a 2-panel figure."""
    from spacr.submodules import plot_cellpose_batch

    images = [np.zeros((16, 16), dtype=np.float32)]
    labels = [_label_mask((16, 16), include_corner=False)]

    plot_cellpose_batch(images, labels)

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Image 1", "Label 1"]


# ===========================================================================
# analyze_percent_positive
# ===========================================================================

_ROWS = ["r1", "r2"]
_COLUMNS = ["c1", "c2", "c3", "c4"]
_FIELDS = ["f1", "f2", "f3"]
_THRESHOLD = 2000.0


def _cell_frame():
    """One measurement row per (row, column, field).

    ``cell_area``   1500 for f1/f2, 500 for f3  -> filter_1 keeps f1+f2.
    ``cell_channel_1_mean_intensity``
        f1 -> 3000 (above), f3 -> 500 (below),
        f2 -> 3000 only in column c1, else 1000
        -> c1 wells are 100% positive after filtering, the rest 50%.
    """
    rows = []
    label = 0
    for r in _ROWS:
        for c in _COLUMNS:
            for f in _FIELDS:
                label += 1
                if f == "f1":
                    value = 3000.0
                elif f == "f2":
                    value = 3000.0 if c == "c1" else 1000.0
                else:
                    value = 500.0
                rows.append({
                    "object_label": label,
                    "plateID": "plate1",
                    "rowID": r,
                    "columnID": c,
                    "fieldID": f,
                    "prcf": f"plate1_{r}_{c}_{f}",
                    "cell_area": 1500.0 if f in ("f1", "f2") else 500.0,
                    "cell_channel_1_mean_intensity": value,
                })
    return pd.DataFrame(rows)


def _well_letter(row_id):
    return "ABCDEFGH"[int(row_id[1:]) - 1]


def _make_screen(tmp_path):
    """Build <src>/measurements/measurements.db + <src>/rename_log.csv."""
    src = tmp_path / "screen"
    (src / "measurements").mkdir(parents=True)
    con = sqlite3.connect(str(src / "measurements" / "measurements.db"))
    try:
        _cell_frame().to_sql("cell", con, index=False)
    finally:
        con.close()

    # rename_log: two rows per well so translate_well_in_df has duplicates
    # to collapse.
    log = []
    for r in _ROWS:
        for c in _COLUMNS:
            well = f"{_well_letter(r)}{int(c[1:]):02d}"
            for f in ("f1", "f2"):
                log.append({
                    "Original File": f"raw_{well}_{f}_ch1.tif",
                    "Renamed TIFF": f"plate1_{well}_{f}.tif",
                })
    pd.DataFrame(log).to_csv(src / "rename_log.csv", index=False)
    return src


def _pp_settings(src, **over):
    s = {
        "src": str(src),
        "tables": ["cell"],
        "filter_1": ["cell_area", 1000],
        "value_col": "cell_channel_1_mean_intensity",
        "threshold": _THRESHOLD,
    }
    s.update(over)
    return s


def _spy_on_final_merge(monkeypatch):
    """Capture the (count_df, translate_df) pair handed to the final pd.merge."""
    real_merge = pd.merge
    captured = {}

    def _spy(left, right, *args, **kwargs):
        out = real_merge(left, right, *args, **kwargs)
        if kwargs.get("on") == ["rowID", "column_name"]:
            captured["count_df"] = left.copy()
            captured["translate_df"] = right.copy()
            captured["merged"] = out.copy()
        return out

    monkeypatch.setattr(pd, "merge", _spy)
    return captured


def test_analyze_percent_positive_annotates_and_summarizes(tmp_path, monkeypatch):
    """The filter_1 + annotate_and_summarize + translate_well_in_df stages
    produce the expected per-well counts and well->row/column translation.
    """
    from spacr.submodules import analyze_percent_positive

    src = _make_screen(tmp_path)
    captured = _spy_on_final_merge(monkeypatch)

    analyze_percent_positive(_pp_settings(src))
    assert "count_df" in captured, "never reached the final merge"

    count_df = captured["count_df"]
    # 2 rows x 4 columns = 8 wells, one condition
    assert len(count_df) == 8
    assert set(count_df["condition"]) == {"none"}
    for col in ("above", "below", "total", "fraction_above", "fraction_below",
                "plateID", "rowID", "column_name"):
        assert col in count_df.columns
    # prc was split back into its three parts
    assert set(count_df["plateID"]) == {"plate1"}
    assert set(count_df["rowID"]) == set(_ROWS)
    assert set(count_df["column_name"]) == set(_COLUMNS)
    # filter_1 dropped the f3 objects -> 2 objects per well
    assert set(count_df["total"]) == {2}
    assert (count_df["above"] + count_df["below"] == count_df["total"]).all()
    np.testing.assert_allclose(
        count_df["fraction_above"] + count_df["fraction_below"], 1.0
    )
    c1 = count_df[count_df["column_name"] == "c1"]
    rest = count_df[count_df["column_name"] != "c1"]
    assert list(c1["above"]) == [2, 2] and list(c1["below"]) == [0, 0]
    np.testing.assert_allclose(c1["fraction_above"], 1.0)
    assert set(rest["above"]) == {1} and set(rest["below"]) == {1}
    np.testing.assert_allclose(rest["fraction_above"], 0.5)

    # -- translate_well_in_df ---------------------------------------------
    tdf = captured["translate_df"]
    assert len(tdf) == 8  # duplicates per plate_well collapsed
    assert set(tdf["plate_well"]) == {
        f"plate1_{_well_letter(r)}{int(c[1:]):02d}" for r in _ROWS for c in _COLUMNS
    }
    assert set(tdf["rowID"]) == {"r1", "r2"}
    assert set(tdf["column_name"]) == set(_COLUMNS)
    assert set(tdf["fieldID"]) == {"f1"}
    assert set(tdf["prc"]) == {f"p1_{r}_{c}" for r in _ROWS for c in _COLUMNS}
    row = tdf[tdf["well"] == "A03"].iloc[0]
    assert (row["plateID"], row["rowID"], row["column_name"]) == ("plate1", "r1", "c3")

    # -- the inner join lined the two tables up 1:1 ------------------------
    assert len(captured["merged"]) == 8

    # the settings snapshot is written before any of this
    assert (src / "settings" / "analyze_percent_positive.csv").is_file()


def test_analyze_percent_positive_without_filter_keeps_all_objects(
    tmp_path, monkeypatch
):
    """filter_1=None skips the filtering branch, so every field counts."""
    from spacr.submodules import analyze_percent_positive

    src = _make_screen(tmp_path)
    captured = _spy_on_final_merge(monkeypatch)

    analyze_percent_positive(_pp_settings(src, filter_1=None))
    assert "count_df" in captured, "never reached the final merge"

    count_df = captured["count_df"]
    assert len(count_df) == 8
    # all three fields survive now
    assert set(count_df["total"]) == {3}
    c1 = count_df[count_df["column_name"] == "c1"]
    rest = count_df[count_df["column_name"] != "c1"]
    assert set(c1["above"]) == {2} and set(c1["below"]) == {1}
    np.testing.assert_allclose(c1["fraction_above"], 2 / 3)
    assert set(rest["above"]) == {1} and set(rest["below"]) == {2}
    np.testing.assert_allclose(rest["fraction_below"], 2 / 3)


def test_analyze_percent_positive_returns_and_writes_result_table(tmp_path):
    """The documented return value: a per-well table written to result.csv."""
    from spacr.submodules import analyze_percent_positive

    src = _make_screen(tmp_path)
    merged = analyze_percent_positive(_pp_settings(src))

    assert isinstance(merged, pd.DataFrame)
    assert len(merged) == 8
    assert {"well", "plate_well", "fieldID", "rowID", "column_name",
            "Original File", "Renamed TIFF", "above", "below",
            "fraction_above", "fraction_below"} <= set(merged.columns)
    np.testing.assert_allclose(
        merged["fraction_above"] + merged["fraction_below"], 1.0
    )
    assert (src / "result.csv").is_file()
    assert len(pd.read_csv(src / "result.csv")) == 8
