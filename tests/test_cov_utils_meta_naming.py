"""CPU-only coverage for spacr.utils' metadata / naming / representative-image
helpers (utils.py lines ~1337-1807).

Covered here:
  * _outline_and_overlay          - contour drawing + colour overlay
  * _get_cellpose_batch_size      - the VRAM -> batch-size ladder (CUDA faked)
  * _extract_filename_metadata    - the IndexError branch for a bad regex
  * _update_database_with_merged_info - every branch incl. failure paths
  * _generate_representative_images   - end-to-end on a real synthetic sqlite DB
  * normalize_to_dtype            - unknown dtype + all-zero channel branches
  * _merge_and_save_to_database   - invalid table_type / missing columns / sqlite error
  * _map_wells / _map_wells_png   - numeric-well and malformed-filename branches

Everything runs offline on CPU: CUDA is faked with monkeypatch, the only
plotting entry point is stubbed with a real Agg figure, and all I/O goes to
tmp_path.
"""
from __future__ import annotations

import os
import re
import sqlite3
import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _outline_and_overlay
# ---------------------------------------------------------------------------

def _stack_with_masks(n_masks=1, h=48, w=48):
    """(H, W, 3 + n_masks) float image: 3 intensity channels + label masks."""
    img = np.zeros((h, w, 3 + n_masks), dtype=np.float32)
    img[..., 0] = 10.0
    img[..., 1] = 20.0
    img[..., 2] = 30.0
    for m in range(n_masks):
        mask = np.zeros((h, w), dtype=np.float32)
        # two disjoint square objects with labels 1 and 2
        mask[6 + m: 18 + m, 6: 18] = 1
        mask[28: 40, 28: 40] = 2
        img[..., 3 + m] = mask
    return img


def test_outline_and_overlay_paints_outline_colour():
    from spacr.utils import _outline_and_overlay, _gen_rgb_image

    image = _stack_with_masks(n_masks=1)
    rgb = _gen_rgb_image(image, channels=[0, 1, 2])
    color = [255, 0, 0]

    overlayed, outlines, returned_image = _outline_and_overlay(
        image, rgb, mask_dims=[3], outline_colors=[color], outline_thickness=2
    )

    assert len(outlines) == 1
    outline = outlines[0]
    assert outline.dtype == np.uint8
    assert outline.shape == image.shape[:2]
    # Both objects produced contours.
    assert outline.max() == 255
    assert outline.sum() > 0

    # Every outline pixel is repainted with the outline colour ...
    painted = outline != 0
    assert np.all(overlayed[painted] == np.array(color, dtype=np.float32))
    # ... and nothing else changed.
    assert np.array_equal(overlayed[~painted], rgb[~painted])
    # The overlay is a copy, not the input.
    assert overlayed is not rgb
    assert returned_image is image


def test_outline_and_overlay_cycles_colors_over_mask_dims():
    from spacr.utils import _outline_and_overlay, _gen_rgb_image

    image = _stack_with_masks(n_masks=2)
    rgb = _gen_rgb_image(image, channels=[0, 1, 2])

    # one colour, two mask dims -> the modulo index reuses colour 0 for dim 1
    overlayed, outlines, _ = _outline_and_overlay(
        image, rgb, mask_dims=[3, 4], outline_colors=[[7, 8, 9]], outline_thickness=1
    )

    assert len(outlines) == 2
    # the two mask dims are offset by one pixel, so the outlines differ
    assert not np.array_equal(outlines[0], outlines[1])
    union = (outlines[0] != 0) | (outlines[1] != 0)
    assert np.all(overlayed[union] == np.array([7, 8, 9], dtype=np.float32))


def test_outline_and_overlay_empty_mask_is_identity():
    from spacr.utils import _outline_and_overlay, _gen_rgb_image

    image = _stack_with_masks(n_masks=1)
    image[..., 3] = 0  # force an empty mask -> the `j == 0` skip is the only path
    rgb = _gen_rgb_image(image, channels=[0, 1, 2])

    overlayed, outlines, _ = _outline_and_overlay(
        image, rgb, mask_dims=[3], outline_colors=[[1, 2, 3]], outline_thickness=3
    )

    assert outlines[0].max() == 0
    assert np.array_equal(overlayed, rgb)


# ---------------------------------------------------------------------------
# _get_cellpose_batch_size
# ---------------------------------------------------------------------------

def _fake_cuda(monkeypatch, vram_gb, name="FakeGPU"):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    props = types.SimpleNamespace(
        total_memory=int(vram_gb * (1024 ** 3)), name=name
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: props)


@pytest.mark.parametrize(
    "vram_gb, expected",
    [(6.0, 8), (10.0, 16), (16.0, 48), (32.0, 96)],
)
def test_get_cellpose_batch_size_vram_ladder(monkeypatch, vram_gb, expected, capsys):
    from spacr.utils import _get_cellpose_batch_size

    _fake_cuda(monkeypatch, vram_gb)
    assert _get_cellpose_batch_size() == expected
    out = capsys.readouterr().out
    assert "cellpose batch size" in out
    assert f"{vram_gb:.2f} GB" in out


def test_get_cellpose_batch_size_without_cuda(monkeypatch, capsys):
    from spacr.utils import _get_cellpose_batch_size
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _get_cellpose_batch_size() == 8
    assert "CUDA is not available" in capsys.readouterr().out


def test_get_cellpose_batch_size_swallows_driver_errors(monkeypatch):
    """A driver blowing up inside get_device_properties falls back to 8."""
    from spacr.utils import _get_cellpose_batch_size
    import torch

    def boom(idx):
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert _get_cellpose_batch_size() == 8


def test_get_cellpose_batch_size_exact_24gb(monkeypatch):
    from spacr.utils import _get_cellpose_batch_size

    _fake_cuda(monkeypatch, 24.0)
    assert _get_cellpose_batch_size() in (48, 96)


# ---------------------------------------------------------------------------
# _extract_filename_metadata - IndexError branch
# ---------------------------------------------------------------------------

def test_extract_filename_metadata_missing_group_is_reported(tmp_path, capsys):
    """A regex without a wellID group makes match.group() raise IndexError."""
    from spacr.utils import _extract_filename_metadata

    rx = re.compile(r"(?P<plateID>.+)\.tif")
    out = _extract_filename_metadata(
        ["plate1_A01.tif"], str(tmp_path), rx, metadata_type="cellvoyager"
    )
    assert out == {}
    assert "Could not extract information from filename" in capsys.readouterr().out


def test_extract_filename_metadata_plateid_falls_back_to_src_basename(tmp_path):
    """No plateID group -> the source folder name is used as the plate."""
    from spacr.utils import _extract_filename_metadata

    src = tmp_path / "plateXYZ"
    src.mkdir()
    rx = re.compile(r"(?P<wellID>[A-P]\d{2})_(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
    out = _extract_filename_metadata(
        ["A01_1_2.tif"], str(src), rx, metadata_type="cellvoyager"
    )
    assert list(out.keys()) == [("plateXYZ", "A01", "1", "2", None, None)]
    assert out[("plateXYZ", "A01", "1", "2", None, None)] == [str(src / "A01_1_2.tif")]


# ---------------------------------------------------------------------------
# _update_database_with_merged_info
# ---------------------------------------------------------------------------

def _png_list_db(tmp_path, name="m.db"):
    """A DB with a png_list table keyed on prcfo."""
    db = tmp_path / name
    con = sqlite3.connect(db)
    try:
        pd.DataFrame(
            {
                "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
                "png_path": ["/a/1.png", "/a/2.png"],
            }
        ).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def _info_df(with_object_label=True, with_cell_id=False):
    df = pd.DataFrame(
        {
            "plateID": ["plate1", "plate1"],
            "rowID": ["r1", "r1"],
            "columnID": ["c1", "c1"],
            "fieldID": ["f1", "f1"],
            "pathogen": ["rh", "rh"],
            "treatment": ["cm", "cm"],
            "host_cells": ["HeLa", "HeLa"],
            "condition": ["HeLa_rh_cm", "HeLa_rh_cm"],
        }
    )
    if with_object_label:
        df["object_label"] = [1, 2]
    if with_cell_id:
        df["cell_id"] = [1, 2]
    return df


def test_update_database_with_merged_info_builds_prcfo_from_object_label(tmp_path, capsys):
    from spacr.utils import _update_database_with_merged_info

    db = _png_list_db(tmp_path)
    df = _info_df(with_object_label=True, with_cell_id=False)
    _update_database_with_merged_info(str(db), df)

    out = capsys.readouterr().out
    assert "generating prcfo columns" in out
    # object_label built prcfo successfully, so the cell_id fallback must NOT
    # run — it used to execute unconditionally and overwrite the good value.
    assert "cell_id" not in out
    assert "successfully updated" in out

    con = sqlite3.connect(db)
    try:
        got = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()
    assert list(got["condition"]) == ["HeLa_rh_cm", "HeLa_rh_cm"]
    assert set(["pathogen", "treatment", "host_cells", "condition"]).issubset(got.columns)
    assert len(got) == 2


def test_update_database_with_merged_info_falls_back_to_cell_id(tmp_path, capsys):
    """No object_label column -> the first prcfo attempt fails, cell_id wins."""
    from spacr.utils import _update_database_with_merged_info

    db = _png_list_db(tmp_path)
    df = _info_df(with_object_label=False, with_cell_id=True)
    _update_database_with_merged_info(str(db), df)

    assert "Merging on cell failed, trying with cell_id" in capsys.readouterr().out

    con = sqlite3.connect(db)
    try:
        got = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()
    assert list(got["host_cells"]) == ["HeLa", "HeLa"]


def test_update_database_with_merged_info_uses_existing_prcfo_and_custom_columns(tmp_path, capsys):
    from spacr.utils import _update_database_with_merged_info

    db = _png_list_db(tmp_path)
    df = pd.DataFrame(
        {
            "prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
            "condition": ["a", "b"],
        }
    )
    _update_database_with_merged_info(str(db), df, columns=["condition", "prcfo"])

    assert "generating prcfo columns" not in capsys.readouterr().out
    con = sqlite3.connect(db)
    try:
        got = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()
    assert list(got["condition"]) == ["a", "b"]
    assert "host_cells" not in got.columns


def test_update_database_with_merged_info_missing_table_returns_early(tmp_path, capsys):
    from spacr.utils import _update_database_with_merged_info

    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    df = _info_df()
    _update_database_with_merged_info(str(db), df, table="png_list")

    assert "Failed to read table png_list" in capsys.readouterr().out
    # nothing was created
    con = sqlite3.connect(db)
    try:
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        con.close()
    assert tables == []


def test_update_database_with_merged_info_reports_write_failure(tmp_path, monkeypatch, capsys):
    """Failure injection: to_sql blows up -> the error is caught and printed."""
    from spacr.utils import _update_database_with_merged_info

    db = _png_list_db(tmp_path)
    df = _info_df()

    def boom(self, *a, **kw):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(pd.DataFrame, "to_sql", boom)
    _update_database_with_merged_info(str(db), df)

    out = capsys.readouterr().out
    assert "Failed to update table png_list in the database" in out
    assert "disk I/O error" in out


# ---------------------------------------------------------------------------
# _generate_representative_images
# ---------------------------------------------------------------------------

N_OBJ = 8


def _entity_frame(entity, n=N_OBJ, seed=0):
    rng = np.random.default_rng(seed)
    half = n // 2
    rows = ["r1" if i < half else "r2" for i in range(n)]
    cols = {
        "object_label": list(range(1, n + 1)),
        "plateID": ["plate1"] * n,
        "rowID": rows,
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "prcf": [f"plate1_{r}_c1_f1" for r in rows],
    }
    if entity in ("nucleus", "pathogen"):
        # parent-cell link is an INTEGER label for the measurement tables
        cols["cell_id"] = list(range(1, n + 1))
    cols[f"{entity}_channel_1_mean_intensity"] = rng.uniform(100, 5000, n)
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    return pd.DataFrame(cols)


@pytest.fixture
def repr_db(tmp_path):
    """A measurements.db with the cell/cytoplasm/pathogen/png_list schema."""
    src = tmp_path / "proj"
    meas = src / "measurements"
    meas.mkdir(parents=True)
    db_path = meas / "measurements.db"

    half = N_OBJ // 2
    rows = ["r1" if i < half else "r2" for i in range(N_OBJ)]
    png_paths = [str(src / "data" / f"crop_{i + 1}.png") for i in range(N_OBJ)]

    con = sqlite3.connect(db_path)
    try:
        for entity in ("cell", "cytoplasm", "pathogen"):
            _entity_frame(entity, seed=hash(entity) % 100).to_sql(entity, con, index=False)
        pd.DataFrame(
            {
                # png_list stores the already-prefixed 'o<N>' string form
                "cell_id": [f"o{i + 1}" for i in range(N_OBJ)],
                "png_path": png_paths,
                "plateID": ["plate1"] * N_OBJ,
                "rowID": rows,
                "columnID": ["c1"] * N_OBJ,
                "fieldID": ["f1"] * N_OBJ,
                "prcfo": [
                    f"plate1_{rows[i]}_c1_f1_o{i + 1}" for i in range(N_OBJ)
                ],
            }
        ).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return {"src": src, "db_path": str(db_path), "png_paths": png_paths}


@pytest.fixture
def stub_grid_plot(monkeypatch):
    """Replace _plot_images_on_grid with a recorder returning a real Agg figure."""
    import spacr.plot

    calls = []

    def fake(image_files, channel_indices, um_per_pixel, scale_bar_length_um=5,
             fontsize=8, show_filename=True, channel_names=None, plot=False):
        calls.append({"files": list(image_files),
                      "channel_indices": list(channel_indices)})
        fig = plt.figure(figsize=(2, 2))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        return fig

    monkeypatch.setattr(spacr.plot, "_plot_images_on_grid", fake)
    return calls


def test_generate_representative_images_defaults_write_one_figure_per_channel(
    repr_db, stub_grid_plot
):
    """All-None arguments fall back to HeLa/rh/cm, pathogen:cytoplasm, 3 channels."""
    from spacr.utils import _generate_representative_images

    _generate_representative_images(
        repr_db["db_path"],
        cells=None,
        pathogens=None,
        treatments=None,
        compartments=None,
        channel_indices=None,
        channel_of_interest=1,
        nr_imgs=2,
        plot=False,
        update_db=True,
    )

    fig_dir = repr_db["src"] / "figure"
    pdfs = sorted(p.name for p in fig_dir.glob("*.pdf"))
    # the default condition is cells_pathogens_treatments = HeLa_rh_cm,
    # with 1 combined figure + one per default channel index (0, 1, 2)
    assert pdfs == [
        "measurements_proj_HeLa_rh_cm.pdf",
        "measurements_proj_channel_0_HeLa_rh_cm.pdf",
        "measurements_proj_channel_1_HeLa_rh_cm.pdf",
        "measurements_proj_channel_2_HeLa_rh_cm.pdf",
    ]
    for name in pdfs:
        assert (fig_dir / name).stat().st_size > 0

    calls = stub_grid_plot
    assert [c["channel_indices"] for c in calls] == [[0, 1, 2], [0], [1], [2]]
    # nr_imgs crops were picked from the png_list paths for every call
    assert all(len(c["files"]) == 2 for c in calls)
    assert all(set(c["files"]).issubset(set(repr_db["png_paths"])) for c in calls)
    # the same crops are reused for the per-channel figures
    assert all(c["files"] == calls[0]["files"] for c in calls)

    # update_db=True merged the annotation columns into png_list
    con = sqlite3.connect(repr_db["db_path"])
    try:
        png_list = pd.read_sql("SELECT * FROM png_list", con)
    finally:
        con.close()
    assert set(png_list["condition"]) == {"HeLa_rh_cm"}
    assert set(png_list["host_cells"]) == {"HeLa"}
    assert set(png_list["pathogen"]) == {"rh"}
    assert len(png_list) == N_OBJ


def test_generate_representative_images_all_channels_for_every_condition(
    repr_db, stub_grid_plot
):
    from spacr.utils import _generate_representative_images

    _generate_representative_images(
        repr_db["db_path"],
        cells="HeLa",
        cell_loc=None,
        pathogens=["rh"],
        pathogen_loc=None,
        treatments=["ctrl", "trt"],
        treatment_loc=[["r1"], ["r2"]],
        channel_of_interest=1,
        compartments=["pathogen", "cytoplasm"],
        nr_imgs=2,
        channel_indices=[0, 1],
        plot=False,
        update_db=False,
    )

    fig_dir = repr_db["src"] / "figure"
    pdfs = sorted(p.name for p in fig_dir.glob("*.pdf"))
    # 2 conditions x (1 combined + 2 per-channel) figures
    assert pdfs == [
        "measurements_proj_HeLa_rh_ctrl.pdf",
        "measurements_proj_HeLa_rh_trt.pdf",
        "measurements_proj_channel_0_HeLa_rh_ctrl.pdf",
        "measurements_proj_channel_0_HeLa_rh_trt.pdf",
        "measurements_proj_channel_1_HeLa_rh_ctrl.pdf",
        "measurements_proj_channel_1_HeLa_rh_trt.pdf",
    ]
    assert [c["channel_indices"] for c in stub_grid_plot] == [
        [0, 1], [0], [1], [0, 1], [0], [1],
    ]


def test_generate_representative_images_selects_condition_specific_crops(
    repr_db, stub_grid_plot
):
    """Row-mapped treatments split the objects into two condition groups."""
    from spacr.utils import _generate_representative_images

    _generate_representative_images(
        repr_db["db_path"],
        cells="HeLa",
        pathogens=["rh"],
        treatments=["ctrl", "trt"],
        treatment_loc=[["r1"], ["r2"]],
        channel_of_interest=1,
        compartments=["pathogen", "cytoplasm"],
        nr_imgs=2,
        channel_indices=[0],
        update_db=False,
    )

    calls = stub_grid_plot
    # first call of each condition block is the combined figure
    ctrl_files = set(calls[0]["files"])
    trt_files = set(calls[-1]["files"])
    # r1 objects (1..4) form 'ctrl', r2 objects (5..8) form 'trt'
    assert ctrl_files and ctrl_files.issubset(set(repr_db["png_paths"][:4]))
    assert trt_files and trt_files.issubset(set(repr_db["png_paths"][4:]))
    assert ctrl_files.isdisjoint(trt_files)

    names = sorted(p.name for p in (repr_db["src"] / "figure").glob("*.pdf"))
    assert "measurements_proj_HeLa_rh_ctrl.pdf" in names
    assert "measurements_proj_HeLa_rh_trt.pdf" in names


def test_generate_representative_images_scalar_compartment_uses_cell_area(
    repr_db, stub_grid_plot
):
    """A non-list `compartments` falls back to ranking on cell_area."""
    from spacr.utils import _generate_representative_images

    _generate_representative_images(
        repr_db["db_path"],
        cells="HeLa",
        pathogens=["rh"],
        treatments=["ctrl"],
        treatment_loc=None,
        compartments="cell",
        nr_imgs=3,
        channel_indices=[0],
        update_db=False,
    )

    fig_dir = repr_db["src"] / "figure"
    pdfs = sorted(p.name for p in fig_dir.glob("*.pdf"))
    # a single condition -> 1 combined + 1 per-channel figure
    assert pdfs == [
        "measurements_proj_HeLa_rh_ctrl.pdf",
        "measurements_proj_channel_0_HeLa_rh_ctrl.pdf",
    ]
    assert len(stub_grid_plot) == 2
    assert all(len(c["files"]) == 3 for c in stub_grid_plot)

    # update_db=False left png_list untouched
    con = sqlite3.connect(repr_db["db_path"])
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(png_list)")]
    finally:
        con.close()
    assert "condition" not in cols


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _generate_representative_images only sets 'new_measurement' when "
           "compartments is a list of >1 entries; a single-element list falls "
           "through both branches and blows up with KeyError('new_measurement').",
)
def test_generate_representative_images_single_element_compartment_list(
    repr_db, stub_grid_plot
):
    from spacr.utils import _generate_representative_images

    _generate_representative_images(
        repr_db["db_path"],
        cells="HeLa",
        pathogens=["rh"],
        treatments=["ctrl"],
        compartments=["cell"],
        nr_imgs=2,
        channel_indices=[0],
        update_db=False,
    )
    assert list((repr_db["src"] / "figure").glob("*.pdf"))


# ---------------------------------------------------------------------------
# normalize_to_dtype - remaining branches
# ---------------------------------------------------------------------------

def test_normalize_to_dtype_unknown_dtype_falls_back_to_input_range(rng):
    from spacr.utils import normalize_to_dtype

    arr = rng.integers(0, 4000, size=(16, 16, 2)).astype(np.uint16)
    out = normalize_to_dtype(arr, new_dtype="uint32")

    assert out.dtype == np.uint16
    assert out.shape == arr.shape
    # the fallback uses the *input* dtype range, so the stretch hits uint16 max
    assert out.max() == np.iinfo(np.uint16).max
    assert out.min() == 0


def test_normalize_to_dtype_all_zero_channel_uses_full_image_percentiles():
    from spacr.utils import normalize_to_dtype

    arr = np.zeros((8, 8, 2), dtype=np.uint16)
    # channel 1 is a real gradient; channel 0 stays all-zero so the
    # `non_zero_img.size == 0` fallback runs on the full image.
    arr[..., 1] = (np.arange(64).reshape(8, 8) * 100 + 1).astype(np.uint16)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = normalize_to_dtype(arr, p1=2, p2=98)

    assert out.shape == arr.shape
    assert out.dtype == np.uint16
    # the empty channel has no signal to stretch and stays flat
    assert out[..., 0].min() == out[..., 0].max()
    # the populated channel is stretched across the full uint16 range
    assert out[..., 1].max() == np.iinfo(np.uint16).max
    assert out[..., 1].min() == 0


# ---------------------------------------------------------------------------
# _merge_and_save_to_database
# ---------------------------------------------------------------------------

def _morph_intensity_frames(n=3, with_cell_id=False):
    morph = pd.DataFrame({"label": list(range(1, n + 1)),
                          "area": np.arange(n, dtype=float) + 10.0})
    if with_cell_id:
        morph["cell_id"] = list(range(1, n + 1))
    intensity = pd.DataFrame({"label": list(range(1, n + 1)),
                              "mean_intensity": np.arange(n, dtype=float) + 100.0})
    return morph, intensity


def test_merge_and_save_to_database_writes_cell_rows(tmp_path):
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    morph, intensity = _morph_intensity_frames(n=3)

    _merge_and_save_to_database(
        morph, intensity, "cell", str(src), "plate1_B03_2", "exp1", timelapse=False
    )

    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        got = pd.read_sql("SELECT * FROM cell", con)
    finally:
        con.close()
    assert len(got) == 3
    assert list(got.columns[:8]) == ["object_label", "plateID", "rowID", "columnID",
                                     "fieldID", "prcf", "file_name", "path_name"]
    assert set(got["plateID"]) == {"plate1"}
    assert set(got["rowID"]) == {"r2"}
    assert set(got["columnID"]) == {"c3"}
    assert set(got["prcf"]) == {"plate1_r2_c3_f2"}
    assert set(got["path_name"]) == {os.path.join(str(src), "plate1_B03_2.npy")}
    assert "label_list_morphology" in got.columns
    assert "label_list_intensity" in got.columns


def test_merge_and_save_to_database_nucleus_keeps_cell_id_and_timelapse(tmp_path):
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    morph, intensity = _morph_intensity_frames(n=2, with_cell_id=True)

    _merge_and_save_to_database(
        morph, intensity, "nucleus", str(src), "plate1_B03_2_7", "exp1", timelapse=True
    )

    con = sqlite3.connect(src / "measurements" / "measurements.db")
    try:
        got = pd.read_sql("SELECT * FROM nucleus", con)
    finally:
        con.close()
    assert len(got) == 2
    assert list(got.columns[:2]) == ["object_label", "cell_id"]
    assert set(got["timeID"]) == {"t7"}
    assert set(got["prcf"]) == {"plate1_r2_c3_f2_t7"}


def test_merge_and_save_to_database_rejects_unknown_table_type(tmp_path):
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    morph, intensity = _morph_intensity_frames()

    with pytest.raises(ValueError, match="Invalid table_type: organelle"):
        _merge_and_save_to_database(
            morph, intensity, "organelle", str(src), "plate1_B03_2", "exp1"
        )


def test_merge_and_save_to_database_raises_on_missing_columns(tmp_path, monkeypatch):
    """Failure injection: a merge that loses object_label must be rejected."""
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    morph, intensity = _morph_intensity_frames()

    real_merge = pd.merge

    def lossy_merge(left, right, *args, **kwargs):
        out = real_merge(left, right, *args, **kwargs)
        return out.drop(columns=["object_label"])

    monkeypatch.setattr(pd, "merge", lossy_merge)

    with pytest.raises(ValueError, match=r"Columns missing in DataFrame: \['object_label'\]"):
        _merge_and_save_to_database(
            morph, intensity, "cell", str(src), "plate1_B03_2", "exp1"
        )


def test_merge_and_save_to_database_reports_sqlite_error(tmp_path, capsys):
    """No measurements/ folder -> sqlite3 cannot open the DB; the error is caught."""
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    src.mkdir()
    morph, intensity = _morph_intensity_frames()

    _merge_and_save_to_database(
        morph, intensity, "cell", str(src), "plate1_B03_2", "exp1"
    )

    assert "SQLite error:" in capsys.readouterr().out
    assert not (src / "measurements").exists()


def test_merge_and_save_to_database_noop_on_empty_frames(tmp_path):
    from spacr.utils import _merge_and_save_to_database

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    empty = pd.DataFrame({"label": pd.Series(dtype=int)})

    _merge_and_save_to_database(
        empty, empty.copy(), "cell", str(src), "plate1_B03_2", "exp1"
    )
    assert not (src / "measurements" / "measurements.db").exists()


# ---------------------------------------------------------------------------
# _map_wells / _map_wells_png - numeric wells and malformed names
# ---------------------------------------------------------------------------

def test_map_wells_numeric_well_copies_well_into_row_and_column():
    from spacr.utils import _map_wells

    plate, row, column, field, prcf = _map_wells("plate1_12_3")
    assert (plate, row, column, field) == ("plate1", "12", "12", "f3")
    assert prcf == "plate1_12_12_f3"


def test_map_wells_png_numeric_well_copies_well_into_row_and_column():
    from spacr.utils import _map_wells_png

    plate, row, column, field, prcfo, object_id = _map_wells_png("plate1_12_3_9.png")
    assert (plate, row, column, field) == ("plate1", "12", "12", "f3")
    assert object_id == "o9"
    assert prcfo == "plate1_12_12_f3_o9"


def test_map_wells_png_malformed_name_returns_error_tuple(capsys):
    from spacr.utils import _map_wells_png

    out = _map_wells_png("garbage.png")
    assert out == ("error",) * 6
    printed = capsys.readouterr().out
    assert "Error processing filename: garbage.png" in printed


def test_map_wells_png_malformed_name_timelapse_returns_error_tuple():
    from spacr.utils import _map_wells_png

    assert _map_wells_png("plate1_A01_1.png", timelapse=True) == ("error",) * 7
