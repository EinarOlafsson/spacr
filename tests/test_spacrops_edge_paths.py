"""spacr.spacrops — defensive branches and rarely-taken paths.

Everything here drives a real code path with real inputs; where an
external condition is required (an unwritable directory, a filesystem
without symlinks, an OpenCV build without SIFT) that condition is
simulated at the boundary, never inside the function under test.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import pytest
import tifffile

from spacr.spacrops import (FOVAlignAndCropper, StitchedMultiAligner,
                            _DiskFeatureStore, align_image_to_stitch,
                            spacrStitcher, stitch_cycle_wells)
from tests.spacrops_synth import (blob_canvas, channel_variant, crop,
                                  row_of_tiles, tile_name, write_cyx,
                                  write_plane)

TILE = 384
STEP = 150
Y0 = X0 = 100


@pytest.fixture(scope="module")
def canvas():
    return blob_canvas(H=900, W=900, seed=1)


def _st(tmp_path, **kw):
    kw.setdefault("outdir", str(tmp_path / "out"))
    kw.setdefault("downsample", 0.5)
    kw.setdefault("save_qc", False)
    kw.setdefault("save_stitched_default", False)
    kw.setdefault("feature_cache_mode", "ram")
    return spacrStitcher(**kw)


def _row(a, b, dx, dy, score, theta=0.0, scale=1.0):
    return {"pathA": a, "pathB": b, "dx_px_full": str(dx), "dy_px_full": str(dy),
            "theta_deg": str(theta), "scale": str(scale), "score": str(score)}


# ===========================================================================
# construction fallbacks
# ===========================================================================

def test_sift_request_without_opencv_contrib_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delattr(cv2, "SIFT_create", raising=False)
    with pytest.raises(RuntimeError, match="opencv-contrib build not found"):
        spacrStitcher(detector="SIFT", outdir=str(tmp_path / "o"))
    with pytest.raises(RuntimeError, match="opencv-contrib build not found"):
        StitchedMultiAligner(detector="SIFT", outdir=str(tmp_path / "o"))


def test_thread_cap_failure_does_not_break_construction(tmp_path, monkeypatch):
    def boom(_n):
        raise cv2.error("setNumThreads unavailable")

    monkeypatch.setattr(cv2, "setNumThreads", boom)
    st = spacrStitcher(outdir=str(tmp_path / "o"), opencv_threads=4)
    assert st._opencv_threads == 4
    assert StitchedMultiAligner(outdir=str(tmp_path / "o2"), opencv_threads=4) is not None


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores directory permissions")
def test_store_keeps_working_when_a_corrupt_npz_cannot_be_deleted(tmp_path):
    root = tmp_path / "cache"
    store = _DiskFeatureStore(str(root))
    npz = store._npz_path("/img/a.tif")
    with open(npz, "wb") as fh:
        fh.write(b"garbage")
    os.chmod(root, 0o500)                    # deletion will fail
    try:
        assert store.get("/img/a.tif") is None
        assert os.path.exists(npz)           # still there, but reported as a miss
    finally:
        os.chmod(root, 0o700)


# ===========================================================================
# float images exercise the non-integer dtype cast
# ===========================================================================

def _float_pair(tmp_path, canvas, name="fpair"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    pa, pb = str(d / tile_name(site=1)), str(d / tile_name(site=2))
    tifffile.imwrite(pa, crop(canvas, Y0, X0, TILE).astype(np.float32))
    tifffile.imwrite(pb, crop(canvas, Y0, X0 + STEP, TILE).astype(np.float32))
    return str(d), pa, pb


def test_stitch_pair_keeps_float_images_in_float(tmp_path, canvas):
    _, pa, pb = _float_pair(tmp_path, canvas)
    st = _st(tmp_path, save_stitched_default=True, score_threshold=0.2)
    r = st.stitch_pair(pa, pb)
    img = tifffile.imread(r["stitched_full_tif"])
    assert img.dtype == np.float32            # no integer rounding applied
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + STEP]
    assert np.allclose(img[:TILE, :STEP - 1], gt[:, :STEP - 1], atol=1e-3)


def test_mosaics_keep_float_images_in_float(tmp_path, canvas):
    d = tmp_path / "ftiles"
    d.mkdir()
    for i in range(3):
        tifffile.imwrite(str(d / tile_name(site=i + 1)),
                         crop(canvas, Y0, X0 + i * STEP, TILE).astype(np.float32))
    st = _st(tmp_path)
    pairs = str(tmp_path / "pairs.csv")
    st.run_folder(str(d), pairs, max_site_gap=1, stitch=False)

    single = str(tmp_path / "m.tif")
    st.render_mosaic_from_csv(pairs, single, min_score=0.3)
    assert tifffile.imread(single).dtype == np.float32

    allc = str(tmp_path / "allc.tif")
    st.mosaic_all_channels_from_csv(pairs, allc, min_score=0.3)
    assert tifffile.imread(allc).dtype == np.float32


def test_align_keeps_float_images_in_float(tmp_path, canvas):
    d = tmp_path / "fcycles"
    d.mkdir()
    p1, p2 = str(d / "c1.tif"), str(d / "c2.tif")
    tifffile.imwrite(p1, crop(canvas, 200, 200, 512).astype(np.float32))
    tifffile.imwrite(p2, crop(canvas, 200, 230, 512).astype(np.float32))
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out, _, _ = al.align([p1, p2])
    assert tifffile.imread(out).dtype == np.float32


# ===========================================================================
# run_folder: csv directory creation, malformed scores, verbose second pass
# ===========================================================================

def test_run_folder_creates_the_csv_directory(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP)
    out = str(tmp_path / "deep" / "nested" / "pairs.csv")
    _st(tmp_path).run_folder(d, out, max_site_gap=1, stitch=False)
    assert os.path.exists(out)
    assert len(list(csv.DictReader(open(out)))) == 1


def test_run_folder_tolerates_a_non_numeric_score(tmp_path, canvas):
    """A row whose score cannot be parsed is written but never counted."""
    d = str(tmp_path / "tiles")
    tiles, _ = row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP)
    st = _st(tmp_path)
    real = st.stitch_pair

    def bad_score(*a, **k):
        row = real(*a, **k)
        if row is not None:
            row["score"] = "not-a-number"
        return row

    st.stitch_pair = bad_score
    out = str(tmp_path / "pairs.csv")
    st.run_folder(d, out, max_site_gap=1, stitch=True)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["score"] == "not-a-number"
    # the auto threshold saw no usable scores at all
    assert os.path.exists(os.path.join(st.outdir, "score_sorted_line.png"))
    # ...and the winners pass skipped the unparseable row
    assert [f for f in os.listdir(st.outdir) if f.endswith("__stitched_full.tif")] == []


def test_run_folder_verbose_reports_the_winners_pass(tmp_path, canvas, capsys):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    st = _st(tmp_path, verbose=True)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=True)
    out = capsys.readouterr().out
    assert "stitching winners second pass" in out
    assert "stitched 1/" in out or "stitched 2/" in out
    assert "Done. CSV" in out


def test_run_folder_verbose_reports_the_multichannel_mosaic(tmp_path, canvas, capsys):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP, channels=2)
    st = _st(tmp_path, verbose=True)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  mosaic=True, mosaic_out=str(tmp_path / "m.tif"),
                  mosaic_all_channels=True, mosaic_min_score=0.3)
    out = capsys.readouterr().out
    assert "rendering multi-channel mosaic" in out


# ===========================================================================
# mosaic edge pruning corner cases
# ===========================================================================

def test_edge_is_rejected_when_no_horizontal_step_was_estimated(tmp_path):
    """A rotated edge's inverse points sideways although only vertical
    neighbours exist, so there is no step_x to validate it against."""
    st = _st(tmp_path, allow_rotation=True)
    rows = [_row("/a.tif", "/b.tif", 0, 150, 0.9, theta=90.0)]
    assert st._estimate_grid_steps(rows, 0.5) == (0.0, 150.0)
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert [e[2] for e in used] == [0.9]      # only the vertical direction survived
    assert sorted(T) == ["/a.tif", "/b.tif"]


def test_edge_is_rejected_when_no_vertical_step_was_estimated(tmp_path):
    st = _st(tmp_path, allow_rotation=True)
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.9, theta=90.0)]
    assert st._estimate_grid_steps(rows, 0.5) == (150.0, 0.0)
    assert len(st._compute_mosaic_transforms(rows, 0.5)[1]) == 1


def test_vertical_edges_are_gated_on_the_estimated_row_pitch(tmp_path):
    st = _st(tmp_path)
    rows = [_row("/a.tif", "/b.tif", 0, 150, 0.9),
            _row("/b.tif", "/c.tif", 0, 150, 0.9),
            _row("/c.tif", "/d.tif", 0, 40, 0.9)]     # far off the 150 pitch
    T, used = st._compute_mosaic_transforms(rows, 0.5, step_tol_frac=0.25)
    assert "/d.tif" not in T
    assert len(used) == 2


def test_compute_mosaic_transforms_spans_a_six_tile_chain(tmp_path):
    """Exercises the union-by-rank merge of two equal-rank subtrees."""
    st = _st(tmp_path)
    p = [f"/t{i}.tif" for i in range(6)]
    rows = [_row(p[0], p[1], STEP, 0, 0.99),
            _row(p[2], p[3], STEP, 0, 0.98),
            _row(p[1], p[2], STEP, 0, 0.97),
            _row(p[3], p[4], STEP, 0, 0.96),
            _row(p[4], p[5], STEP, 0, 0.95)]
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert len(used) == 5 and len(T) == 6
    origin = T[p[0]][0, 2]
    for i in range(6):
        assert T[p[i]][0, 2] - origin == pytest.approx(i * STEP, abs=1e-3)
        assert T[p[i]][1, 2] == pytest.approx(T[p[0]][1, 2], abs=1e-3)


# ===========================================================================
# build_multichannel_mosaic_from_manifest: local reader branches
# ===========================================================================

def _hand_manifest(tmp_path, tiles, H, W, step=STEP):
    mcsv = str(tmp_path / "hand_manifest.csv")
    with open(mcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "H", "W", "M00", "M01", "M02",
                                           "M10", "M11", "M12", "canvas_x",
                                           "canvas_y", "best_pair_score"])
        w.writeheader()
        for i, p in enumerate(tiles):
            w.writerow(dict(path=p, H=H, W=W, M00=1.0, M01=0.0, M02=i * step,
                            M10=0.0, M11=1.0, M12=0.0, canvas_x=i * step,
                            canvas_y=0.0, best_pair_score=0.9))
    return mcsv


def test_manifest_builder_reads_plain_2d_tiles(tmp_path, canvas):
    d = tmp_path / "flat"
    d.mkdir()
    tiles = []
    for i in range(2):
        p = str(d / f"t{i}.tif")
        write_plane(p, crop(canvas, Y0, X0 + i * STEP, TILE))
        tiles.append(p)
    mcsv = _hand_manifest(tmp_path, tiles, TILE, TILE)
    st = _st(tmp_path)
    out = str(tmp_path / "flat.tif")
    st.build_multichannel_mosaic_from_manifest(mcsv, out, tmp_dir=str(tmp_path / "t"))
    img = tifffile.imread(out)
    plane = img if img.ndim == 2 else img[0]
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + STEP]
    assert plane.shape == gt.shape
    assert np.corrcoef(plane.ravel(), gt.ravel())[0, 1] > 0.999


def test_manifest_builder_max_projects_a_czyx_tile(tmp_path, canvas):
    d = tmp_path / "czyx"
    d.mkdir()
    tiles = []
    for i in range(2):
        p = str(d / f"t{i}.tif")
        base = crop(canvas, Y0, X0 + i * STEP, TILE)
        # (C=2, Z=3, Y, X): the brightest Z slice is the plain canvas crop
        stack = np.stack([np.stack([base // 4, base // 2, base]) for _ in range(2)])
        tifffile.imwrite(p, stack, metadata={"axes": "CZYX"})
        tiles.append(p)
    mcsv = _hand_manifest(tmp_path, tiles, TILE, TILE)
    out = str(tmp_path / "czyx.tif")
    _st(tmp_path).build_multichannel_mosaic_from_manifest(
        mcsv, out, channel_indices=[0], tmp_dir=str(tmp_path / "t"))
    img = tifffile.imread(out)
    plane = img if img.ndim == 2 else img[0]
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + STEP]
    assert np.corrcoef(plane.ravel(), gt.ravel())[0, 1] > 0.999
    assert plane.max() > (gt // 2).max()      # the max projection, not a middle slice


def test_manifest_builder_rejects_a_tile_it_cannot_reduce_to_2d(tmp_path, canvas):
    d = tmp_path / "tczyx"
    d.mkdir()
    base = crop(canvas, Y0, X0, 64)
    p = str(d / "t0.tif")
    arr = np.stack([np.stack([np.stack([base, base]) for _ in range(2)])
                    for _ in range(2)])       # (T,C,Z,Y,X)
    tifffile.imwrite(p, arr, metadata={"axes": "TCZYX"})
    mcsv = _hand_manifest(tmp_path, [p], 64, 64)
    with pytest.raises(ValueError, match="Expected 2D plane"):
        _st(tmp_path).build_multichannel_mosaic_from_manifest(
            mcsv, str(tmp_path / "x.tif"), tmp_dir=str(tmp_path / "t"))


def test_manifest_builder_full_resolution_preview(tmp_path, canvas):
    d = tmp_path / "prev"
    d.mkdir()
    p = str(d / "t0.tif")
    write_plane(p, crop(canvas, Y0, X0, TILE))
    mcsv = _hand_manifest(tmp_path, [p], TILE, TILE)
    png = str(tmp_path / "prev.png")
    _st(tmp_path).build_multichannel_mosaic_from_manifest(
        mcsv, str(tmp_path / "p.tif"), out_png=png, preview_downsample=1,
        tmp_dir=str(tmp_path / "t"))
    assert os.path.getsize(png) > 0


def test_manifest_builder_preview_of_a_flat_mosaic(tmp_path):
    d = tmp_path / "flatpix"
    d.mkdir()
    p = str(d / "t0.tif")
    write_plane(p, np.full((64, 64), 7, np.uint16))
    mcsv = _hand_manifest(tmp_path, [p], 64, 64)
    png = str(tmp_path / "flat.png")
    _st(tmp_path).build_multichannel_mosaic_from_manifest(
        mcsv, str(tmp_path / "p.tif"), out_png=png, tmp_dir=str(tmp_path / "t"))
    assert os.path.getsize(png) > 0


# ===========================================================================
# StitchedMultiAligner axis-handling branches
# ===========================================================================

def test_multialigner_normalize_guesses_axes_without_a_hint(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"))     # arr_axes AUTO
    arr = np.arange(3 * 256 * 256, dtype=np.float32).reshape(3, 256, 256)
    assert np.array_equal(al._normalize_to_yx(arr, ch=2), arr[2])


def test_multialigner_normalize_drops_surplus_time_and_channel_labels(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="TCYX")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert np.array_equal(al._normalize_to_yx(arr, ch=0), arr)


def test_multialigner_normalize_pads_channel_then_z(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="YX",
                              t_index=0, z_index=1)
    arr4 = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    assert np.array_equal(al._normalize_to_yx(arr4, ch=1), arr4[1, 0])
    arr5 = np.arange(2 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 2, 3, 4)
    assert np.array_equal(al._normalize_to_yx(arr5, ch=1), arr5[1, 1, 0])


@pytest.mark.skipif(not hasattr(cv2, "SIFT_create"), reason="no SIFT in this build")
def test_multialigner_sift_matching(tmp_path, canvas):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), detector="SIFT")
    rng = np.random.default_rng(5)
    desc = (rng.random((8, 128)).astype(np.float32) * 100)
    pts = rng.random((8, 2)).astype(np.float32) * 50
    pA, pB = al._match({"pts": pts, "desc": desc}, {"pts": pts, "desc": desc})
    assert pA.shape[0] >= 4 and np.allclose(pA, pB, atol=1e-5)

    flat = np.ones((6, 128), np.float32)       # every ratio test fails
    empty = al._match({"pts": pts[:6], "desc": flat}, {"pts": pts[:6], "desc": flat})
    assert empty[0].shape == (0, 2)


# ===========================================================================
# FOVAlignAndCropper branches
# ===========================================================================

def _mosaic_and_fov(tmp_path, canvas, fy=150, fx=180, fov=256):
    root = tmp_path / "fovcase"
    (root / "fov").mkdir(parents=True, exist_ok=True)
    mos = str(root / "mosaic.tif")
    write_cyx(mos, [crop(canvas, Y0, X0, 700)])
    write_cyx(str(root / "fov" / tile_name(site=1, mag="20X")),
              [crop(canvas, Y0 + fy, X0 + fx, fov)])
    return mos, str(root / "fov")


def test_fov_cropper_treats_a_non_positive_scale_as_one(tmp_path, canvas):
    mos, folder = _mosaic_and_fov(tmp_path, canvas)
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5)
    rows = list(csv.DictReader(open(fa.run(mos, folder, folder_image_scale=-4.0))))
    assert len(rows) == 1
    assert float(rows[0]["tx"]) == pytest.approx(180, abs=1.5)


def test_fov_cropper_skips_a_fov_whose_ransac_fails(tmp_path, canvas):
    mos, folder = _mosaic_and_fov(tmp_path, canvas)
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5)
    fa._affine_from_pts = lambda *a, **k: (None, None, 0.0)
    assert list(csv.DictReader(open(fa.run(mos, folder)))) == []


def test_fov_cropper_rotation_only_mode(tmp_path, canvas):
    mos, folder = _mosaic_and_fov(tmp_path, canvas)
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5,
                            allow_rotation=True, allow_scale=False)
    rows = list(csv.DictReader(open(fa.run(mos, folder))))
    assert len(rows) == 1
    # a rotation-enabled fit averages the residual over all inliers, so the
    # translation is a little noisier than in translation-only mode
    assert float(rows[0]["tx"]) == pytest.approx(180, abs=6.0)
    assert float(rows[0]["ty"]) == pytest.approx(150, abs=6.0)
    assert float(rows[0]["scale"]) == pytest.approx(1.0, abs=1e-4)
    assert abs(float(rows[0]["theta_deg"])) < 2.5    # near-zero, as it should be


# ===========================================================================
# stitch_cycle_wells collision / relink branches
# ===========================================================================

def _plate(tmp_path, canvas, wells=("A1",), n=2, tile=320, step=130, sub="src"):
    src = tmp_path / sub
    src.mkdir(parents=True, exist_ok=True)
    for well in wells:
        for i in range(n):
            write_cyx(str(src / f"10X_c1_{well}_Site-{i + 1}.tif"),
                      [channel_variant(crop(canvas, Y0, X0 + i * step, tile), c)
                       for c in range(2)])
    return str(src)


def _settings(src, dst, **kw):
    base = dict(src=src, dst_root=dst, verbose=False, max_site_gap=2,
                n_workers=2, downsample=0.5, plate="P1", do_nuc_stitch=False)
    base.update(kw)
    return base


def test_symlink_mode_skips_an_existing_link_when_asked(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = tmp_path / "dst"
    (dst / "_links" / "A1").mkdir(parents=True)
    (dst / "_links" / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"keep")
    res = stitch_cycle_wells(_settings(src, str(dst), do_organize=False,
                                       collision="skip"))
    assert res["organized"]["skipped"] == 1
    assert res["organized"]["linked"] == 1
    assert (dst / "_links" / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() == b"keep"


def test_symlink_mode_replaces_an_existing_link(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = str(tmp_path / "dst")
    first = stitch_cycle_wells(_settings(src, dst, do_organize=False))
    second = stitch_cycle_wells(_settings(src, dst, do_organize=False,
                                          collision="overwrite"))
    assert first["organized"]["linked"] == 2
    assert second["organized"]["linked"] == 2
    links = os.path.join(dst, "_links", "A1")
    assert len(os.listdir(links)) == 2        # relinked in place, not duplicated


def test_well_with_every_file_skipped_is_not_stitched(tmp_path, canvas):
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"
    (dst / "A1").mkdir(parents=True)
    (dst / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"occupied")
    res = stitch_cycle_wells(_settings(src, str(dst), collision="skip",
                                       do_nuc_stitch=True))
    assert res["organized"]["skipped"] == 1
    assert res["wells"] == {}                 # nothing left to stitch


def test_post_stitch_move_honours_the_collision_policy(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = tmp_path / "dst"
    (dst / "A1" / "A1").mkdir(parents=True)
    (dst / "A1" / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"already staged")
    res = stitch_cycle_wells(_settings(src, str(dst), collision="skip",
                                       do_nuc_stitch=True, mosaic=False))
    tiles = [os.path.basename(t) for t in res["wells"]["A1"]["tiles"]]
    assert tiles == ["10X_c1_A1_Site-2.tif"]   # site 1 kept its staged copy
    assert (dst / "A1" / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() == b"already staged"


def test_post_stitch_move_can_overwrite(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = tmp_path / "dst"
    (dst / "A1" / "A1").mkdir(parents=True)
    (dst / "A1" / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"stale")
    res = stitch_cycle_wells(_settings(src, str(dst), collision="overwrite",
                                       do_nuc_stitch=True, mosaic=False))
    assert len(res["wells"]["A1"]["tiles"]) == 2
    assert (dst / "A1" / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() != b"stale"


def test_post_stitch_symlink_relinks_on_a_second_run(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = str(tmp_path / "dst")
    kw = dict(do_organize=False, do_nuc_stitch=True, mosaic=False,
              collision="overwrite")
    stitch_cycle_wells(_settings(src, dst, **kw))
    res = stitch_cycle_wells(_settings(src, dst, **kw))
    tiles = res["wells"]["A1"]["tiles"]
    assert len(tiles) == 2
    assert all(os.path.islink(t) for t in tiles)
    assert all(os.path.exists(t) for t in tiles)


def test_write_mosaic_setting_currently_suppresses_the_mosaic_image(tmp_path, canvas):
    """NOTE: the flag reads inverted - ``write_mosaic=True`` produces only the
    manifest and no mosaic TIFF.  Pinned here as *current* behaviour; see the
    report accompanying these tests for the proposed fix."""
    src = _plate(tmp_path, canvas)
    dst = str(tmp_path / "dst")
    res = stitch_cycle_wells(_settings(src, dst, do_nuc_stitch=True, mosaic=True,
                                       write_mosaic=True, mosaic_min_score=0.2))
    a1 = res["wells"]["A1"]
    assert a1["mosaic_cyx"] is None            # no image written
    assert a1["mosaic_csv"] and os.path.exists(a1["mosaic_csv"])


# ===========================================================================
# align_image_to_stitch scanning branches
# ===========================================================================

def _stitched_root(tmp_path, canvas, wells=("A1",)):
    root = tmp_path / "stitched"
    for well in wells:
        d = root / well / "_stitch"
        d.mkdir(parents=True)
        write_cyx(str(d / "mosaic_allc.tif"), [crop(canvas, Y0, X0, 700)])
    return str(root)


def test_align_image_to_stitch_can_scan_non_recursively(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    (src / "deeper").mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])
    write_cyx(str(src / "deeper" / "20X_c1_A1_Site-2.tif"),
              [crop(canvas, Y0 + 40, X0 + 60, 256)])
    res = align_image_to_stitch(root, str(src), relative_scale=1.0,
                                recursive_align_src=False)
    assert len(os.listdir(res["A1"]["align_folder"])) == 1


def test_align_image_to_stitch_ignores_unparseable_names(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    src.mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])
    write_cyx(str(src / "no_metadata_here.tif"), [crop(canvas, 0, 0, 64)])
    res = align_image_to_stitch(root, str(src), relative_scale=1.0)
    assert len(os.listdir(res["A1"]["align_folder"])) == 1


def test_align_image_to_stitch_skips_a_blank_well_group(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    src.mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])
    # a regex whose well group can match empty
    res = align_image_to_stitch(root, str(src), relative_scale=1.0,
                                meta_regex=r"20X_c\d+_(?P<well>[A-H]?\d*)x?_?Site")
    # Measured: this returns ["A1"], so the old `res == {} or ...`
    # disjunction accepted both outcomes and asserted nothing.
    assert list(res) == ["A1"]


def test_align_image_to_stitch_suffixes_repeated_link_names(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    src.mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])
    align_image_to_stitch(root, str(src), relative_scale=1.0)
    align_image_to_stitch(root, str(src), relative_scale=1.0)   # links already exist
    linked = sorted(os.listdir(os.path.join(root, "_links", "align20x", "A1")))
    assert linked == ["20X_c1_A1_Site-1.tif", "20X_c1_A1_Site-1_002.tif"]


def test_align_image_to_stitch_copies_when_symlinks_are_unavailable(tmp_path, canvas,
                                                                    monkeypatch):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    src.mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])

    def no_symlinks(*a, **k):
        raise OSError("symlinks not supported on this filesystem")

    monkeypatch.setattr(os, "symlink", no_symlinks)
    res = align_image_to_stitch(root, str(src), relative_scale=1.0)
    copied = os.path.join(res["A1"]["align_folder"], "20X_c1_A1_Site-1.tif")
    assert os.path.isfile(copied) and not os.path.islink(copied)
    assert len(list(csv.DictReader(open(res["A1"]["manifest_csv"])))) == 1


# ===========================================================================
# remaining defensive paths
# ===========================================================================

def test_manifest_builder_indexes_a_stack_with_no_usable_axis_labels(tmp_path, canvas):
    """tifffile labels a bare 3-D write 'SYX'; only Y/X survive filtering, so
    the label list no longer describes the array and the plain fallback runs."""
    d = tmp_path / "bare"
    d.mkdir()
    tiles = []
    planes = {}
    for i in range(2):
        p = str(d / f"t{i}.tif")
        base = crop(canvas, Y0, X0 + i * STEP, TILE)
        stack = np.stack([base // 3, base])          # plane 1 is the bright one
        tifffile.imwrite(p, stack)                   # no axes metadata at all
        tiles.append(p)
        planes[p] = stack
    mcsv = _hand_manifest(tmp_path, tiles, TILE, TILE)
    out = str(tmp_path / "bare.tif")
    _st(tmp_path).build_multichannel_mosaic_from_manifest(
        mcsv, out, channel_indices=[1], tmp_dir=str(tmp_path / "t"))
    img = tifffile.imread(out)
    plane = img if img.ndim == 2 else img[0]
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + STEP]
    assert plane.shape == gt.shape
    assert np.corrcoef(plane.ravel(), gt.ravel())[0, 1] > 0.999
    assert plane.max() > (gt // 3).max()             # plane 1, not plane 0


def test_compute_mosaic_transforms_attaches_a_singleton_to_a_taller_tree(tmp_path):
    """The last edge merges a rank-2 tree (as src) with a fresh node."""
    st = _st(tmp_path)
    p = [f"/t{i}.tif" for i in range(5)]
    rows = [_row(p[0], p[1], STEP, 0, 0.99),
            _row(p[2], p[3], STEP, 0, 0.98),
            _row(p[1], p[2], STEP, 0, 0.97),
            _row(p[4], p[0], STEP, 0, 0.96)]
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert len(used) == 4 and len(T) == 5
    base = T[p[0]][0, 2]
    assert T[p[1]][0, 2] - base == pytest.approx(STEP, abs=1e-3)
    assert T[p[2]][0, 2] - base == pytest.approx(2 * STEP, abs=1e-3)
    assert T[p[3]][0, 2] - base == pytest.approx(3 * STEP, abs=1e-3)
    assert T[p[4]][0, 2] - base == pytest.approx(-STEP, abs=1e-3)


def test_collision_resolution_gives_up_after_ten_thousand_names(tmp_path, canvas):
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"
    well = dst / "A1"
    well.mkdir(parents=True)
    stem = "10X_c1_A1_Site-1"
    (well / f"{stem}.tif").write_bytes(b"x")
    for k in range(1, 10000):                        # every rename candidate taken
        os.close(os.open(str(well / f"{stem}_{k:03d}.tif"), os.O_CREAT | os.O_WRONLY))
    with pytest.raises(RuntimeError, match="Could not resolve collision"):
        stitch_cycle_wells(_settings(src, str(dst)))


def test_organize_survives_a_file_vanishing_mid_overwrite(tmp_path, canvas, monkeypatch):
    """The remove/move pair is racy; the FileNotFoundError guard keeps it going."""
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"
    (dst / "A1").mkdir(parents=True)
    (dst / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"stale")

    real_remove = os.remove

    def racing_remove(path, *a, **k):
        real_remove(path, *a, **k)
        raise FileNotFoundError(path)                # as if a peer removed it first

    monkeypatch.setattr(os, "remove", racing_remove)
    res = stitch_cycle_wells(_settings(src, str(dst), collision="overwrite"))
    assert res["organized"]["moved"] == 1
    assert (dst / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() != b"stale"


def test_symlinking_survives_a_link_appearing_first(tmp_path, canvas, monkeypatch):
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"

    def racing_symlink(target, link, *a, **k):
        raise FileExistsError(link)                  # a peer created it first

    monkeypatch.setattr(os, "symlink", racing_symlink)
    res = stitch_cycle_wells(_settings(src, str(dst), do_organize=False))
    assert res["organized"]["linked"] == 1
    assert res["organized"]["moved"] == 0


def test_post_stitch_move_survives_a_file_vanishing(tmp_path, canvas, monkeypatch):
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"
    (dst / "A1" / "A1").mkdir(parents=True)
    (dst / "A1" / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"stale")

    real_remove = os.remove
    seen = {"n": 0}

    def racing_remove(path, *a, **k):
        seen["n"] += 1
        real_remove(path, *a, **k)
        raise FileNotFoundError(path)

    monkeypatch.setattr(os, "remove", racing_remove)
    res = stitch_cycle_wells(_settings(src, str(dst), collision="overwrite",
                                       do_nuc_stitch=True, mosaic=False))
    assert seen["n"] >= 1
    assert len(res["wells"]["A1"]["tiles"]) == 1


def test_post_stitch_symlink_survives_a_link_appearing_first(tmp_path, canvas, monkeypatch):
    src = _plate(tmp_path, canvas, n=1)
    dst = tmp_path / "dst"
    calls = {"n": 0}
    real_symlink = os.symlink

    def racing_symlink(target, link, *a, **k):
        calls["n"] += 1
        if calls["n"] > 1:                           # only the staging pass races
            raise FileExistsError(link)
        return real_symlink(target, link, *a, **k)

    monkeypatch.setattr(os, "symlink", racing_symlink)
    res = stitch_cycle_wells(_settings(src, str(dst), do_organize=False,
                                       do_nuc_stitch=True, mosaic=False))
    assert calls["n"] >= 2
    assert len(res["wells"]["A1"]["tiles"]) == 1


def test_align_image_to_stitch_drops_files_with_an_empty_well_group(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas)
    src = tmp_path / "align20x"
    src.mkdir(parents=True)
    write_cyx(str(src / "20X_c1_A1_Site-1.tif"), [crop(canvas, Y0 + 150, X0 + 180, 256)])
    write_cyx(str(src / "20X_c1_Site-4.tif"), [crop(canvas, Y0 + 40, X0 + 60, 256)])
    res = align_image_to_stitch(
        root, str(src), relative_scale=1.0,
        meta_regex=r"20X_c\d+_(?P<well>[A-H]\d+)?_?Site[-_](?P<site>\d+)\.tif$")
    # the well-less file matched the regex but carries no well, so it is dropped
    assert list(res) == ["A1"]
    assert os.listdir(res["A1"]["align_folder"]) == ["20X_c1_A1_Site-1.tif"]
