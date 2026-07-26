"""spacr.spacrops — multi-cycle alignment, FOV cropping and the plate pipelines.

As in the stitcher tests, every image is a crop of one canvas at a known
offset so the recovered transforms can be compared with a ground truth.
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
                            align_image_to_stitch, get_preprocess_ops_settings,
                            ops_preprocess, stitch_cycle_wells)
from tests.spacrops_synth import (blob_canvas, channel_variant, crop,
                                  tile_name, write_cyx)


@pytest.fixture(scope="module")
def canvas():
    return blob_canvas(H=900, W=900, seed=2)


# ===========================================================================
# StitchedMultiAligner — construction and helpers
# ===========================================================================

def test_multialigner_rejects_an_unknown_detector(tmp_path):
    with pytest.raises(ValueError, match="detector must be"):
        StitchedMultiAligner(detector="BRISK", outdir=str(tmp_path / "o"))


def test_multialigner_stores_its_configuration(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), max_keypoints=None,
                              arr_axes="CYX", mip=True, z_index=1, t_index=2,
                              downsample=0.25)
    assert al.max_keypoints is None and al.arr_axes == "CYX"
    assert (al.mip, al.z_index, al.t_index) == (True, 1, 2)
    assert os.path.isdir(al.outdir)
    assert al._use_flann is False


@pytest.mark.skipif(not hasattr(cv2, "SIFT_create"), reason="no SIFT in this build")
def test_multialigner_sift_uses_flann(tmp_path):
    al = StitchedMultiAligner(detector="SIFT", outdir=str(tmp_path / "o"))
    assert al._use_flann and hasattr(al, "_flann")


@pytest.mark.parametrize("shape,expected", [
    ((256, 256), "YX"),
    ((3, 256, 256), "CYX"),
    ((40, 256, 256), "ZYX"),
    ((256, 256, 4), "YXC"),
    ((256, 256, 40), "YXZ"),
    ((4, 4, 4), "CYX"),
    ((3, 40, 256, 256), "CZYX"),
    ((40, 3, 256, 256), "ZCYX"),
    ((3, 4, 256, 256), "CZYX"),
    ((40, 50, 256, 256), "CZYX"),
    ((2, 3, 4, 5), "TCYX"),
    ((2, 3, 4, 256, 256), "TCZYX"),
    ((2, 40, 4, 256, 256), "TZCYX"),
    ((2, 3, 4, 5, 6), "TZCYX"),
    ((2, 3, 4, 5, 6, 7), "CZYX"),
])
def test_multialigner_guess_axes(shape, expected):
    assert StitchedMultiAligner._guess_axes_from_shape(shape) == expected


def test_multialigner_normalize_to_yx_selects_channel_and_time(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="TCYX", t_index=1)
    arr = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    assert np.array_equal(al._normalize_to_yx(arr, ch=2), arr[1, 2])


def test_multialigner_normalize_to_yx_terminates_on_surplus_zyx_labels(tmp_path):
    """Regression: this used to spin forever because neither T nor C was
    present to drop."""
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="ZYX")
    arr = np.arange(20, dtype=np.float32).reshape(4, 5)
    assert np.array_equal(al._normalize_to_yx(arr, ch=0), arr)


def test_multialigner_normalize_to_yx_mip_and_padding(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="ZYX", mip=True)
    rng = np.random.default_rng(7)
    arr = rng.random((5, 200, 200)).astype(np.float32)
    assert np.allclose(al._normalize_to_yx(arr, ch=0), arr.max(axis=0))

    al2 = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="YX", t_index=1)
    arr3 = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    assert np.array_equal(al2._normalize_to_yx(arr3, ch=0), arr3[1])


def test_multialigner_normalize_to_yx_drops_a_stray_small_axis(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="YXX")
    arr = np.arange(200 * 200 * 3, dtype=np.float32).reshape(200, 200, 3)
    assert np.array_equal(al._normalize_to_yx(arr, ch=0), arr[:, :, 0])


def test_multialigner_normalize_to_yx_rejects_a_non_2d_result(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), arr_axes="YXX")
    with pytest.raises(ValueError, match="Expected 2D YX"):
        al._normalize_to_yx(np.zeros((200, 200, 200), np.float32), ch=0)


def test_multialigner_io_helpers(tmp_path, canvas):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"))
    flat = str(tmp_path / "flat.tif")
    tifffile.imwrite(flat, crop(canvas, 0, 0, 64))
    assert al._read_plane(flat).shape == (64, 64)
    assert StitchedMultiAligner._get_channel_count_tif(flat) == 1

    cyx = str(tmp_path / "cyx.tif")
    write_cyx(cyx, [np.full((32, 32), v, np.uint16) for v in (3, 9)])
    assert StitchedMultiAligner._get_channel_count_tif(cyx) == 2
    assert al._read_plane(cyx, ch=1).mean() == pytest.approx(9)
    stack = al._read_all_channels_cyx(cyx)
    assert stack.shape == (2, 32, 32)
    assert stack[0].mean() == pytest.approx(3)


def test_multialigner_read_plane_mip(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), mip=True)
    arr = np.stack([np.full((200, 200), i, np.uint16) for i in (1, 7, 3)])
    p = str(tmp_path / "z.tif")
    tifffile.imwrite(p, arr, metadata={"axes": "ZYX"})
    assert al._read_plane(p).mean() == pytest.approx(7)


def test_multialigner_static_maths():
    assert StitchedMultiAligner._to_uint8(
        np.array([[0.0, 2.0], [4.0, 8.0]])).tolist() == [[0, 63], [127, 255]]
    assert StitchedMultiAligner._is_large_dim(128) and not StitchedMultiAligner._is_large_dim(127)

    rng = np.random.default_rng(3)
    a = (rng.random((48, 48)) * 200).astype(np.float32)
    assert StitchedMultiAligner._edge_zncc(a, a) == pytest.approx(1.0, abs=1e-3)
    tiny = np.zeros((16, 16), bool)
    tiny[0, 0] = True
    assert StitchedMultiAligner._edge_zncc(a, a, mask=tiny) == 0.0

    R = StitchedMultiAligner._closest_rotation(np.array([[2.0, 0.0], [0.0, 3.0]], np.float32))
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)
    R2 = StitchedMultiAligner._closest_rotation(np.array([[1.0, 0.0], [0.0, -1.0]], np.float32))
    assert np.linalg.det(R2) == pytest.approx(1.0, abs=1e-5)


def test_multialigner_affine_from_pts_recovers_a_translation():
    rng = np.random.default_rng(4)
    ptsB = rng.random((30, 2)).astype(np.float32) * 100
    ptsA = ptsB + np.array([5.0, -3.0], np.float32)
    M, mask, ratio = StitchedMultiAligner._affine_from_pts(ptsA, ptsB, 3.0)
    assert M[0, 2] == pytest.approx(5.0, abs=1e-2)
    assert M[1, 2] == pytest.approx(-3.0, abs=1e-2)
    assert ratio == pytest.approx(1.0)
    few = np.zeros((2, 2), np.float32)
    assert StitchedMultiAligner._affine_from_pts(few, few, 3.0) == (None, None, 0.0)
    same = np.zeros((9, 2), np.float32)
    assert StitchedMultiAligner._affine_from_pts(same, same, 3.0)[0] is None


def test_multialigner_match_needs_four_points(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"))
    small = {"pts": np.zeros((3, 2), np.float32), "desc": np.zeros((3, 32), np.uint8)}
    a, b = al._match(small, small)
    assert a.shape == (0, 2) and b.shape == (0, 2)


def test_multialigner_detect_and_describe_on_a_blank_image(tmp_path):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"))
    pts, desc = al._detect_and_describe(np.zeros((64, 64), np.uint8))
    assert pts.shape == (0, 2) and desc.shape == (0, 32)


def test_multialigner_caps_keypoints(tmp_path, canvas):
    al = StitchedMultiAligner(outdir=str(tmp_path / "o"), max_keypoints=12)
    pts, desc = al._detect_and_describe(
        StitchedMultiAligner._to_uint8(crop(canvas, 100, 100, 256).astype(np.float32)))
    assert pts.shape[0] == 12 and desc.shape[0] == 12


# ===========================================================================
# StitchedMultiAligner.align
# ===========================================================================

def _cycles(tmp_path, canvas, dy, dx, size=512, channels=2):
    d = tmp_path / "cycles"
    d.mkdir(exist_ok=True)
    ref = crop(canvas, 200, 200, size)
    mov = crop(canvas, 200 + dy, 200 + dx, size)
    p1, p2 = str(d / "cycle1.tif"), str(d / "cycle2.tif")
    write_cyx(p1, [channel_variant(ref, c) for c in range(channels)])
    write_cyx(p2, [channel_variant(mov, c) for c in range(channels)])
    return p1, p2, ref, mov


def test_align_registers_a_second_cycle_onto_the_reference(tmp_path, canvas):
    p1, p2, ref, mov = _cycles(tmp_path, canvas, dy=-25, dx=40)
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out_tif, png, csv_path = al.align([p1, p2], out_png_preview=str(tmp_path / "prev.png"))

    stack = tifffile.imread(out_tif)
    assert stack.shape == (4, 512, 512)          # 2 channels per cycle
    assert stack.dtype == np.uint16

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 4
    assert [r["output_channel"] for r in rows] == ["0", "1", "2", "3"]
    assert [r["ref"] for r in rows] == ["True", "True", "False", "False"]
    assert float(rows[0]["tx"]) == 0.0 and float(rows[0]["score"]) == 1.0
    assert float(rows[2]["tx"]) == pytest.approx(40.0, abs=1.0)
    assert float(rows[2]["ty"]) == pytest.approx(-25.0, abs=1.0)
    assert float(rows[2]["scale"]) == 1.0 and float(rows[2]["theta_deg"]) == 0.0
    assert float(rows[2]["score"]) > 0.3
    assert rows[2]["tx"] == rows[3]["tx"]         # both channels share the transform

    # the warped cycle-2 nuclei now line up with the reference
    inner = (slice(60, 450), slice(60, 450))
    aligned = stack[2][inner].astype(float)
    assert np.corrcoef(aligned.ravel(), ref[inner].astype(float).ravel())[0, 1] > 0.99
    assert np.corrcoef(mov[inner].astype(float).ravel(),
                       ref[inner].astype(float).ravel())[0, 1] < 0.5
    assert os.path.getsize(png) > 0


def test_align_defaults_its_output_paths(tmp_path, canvas):
    p1, p2, _, _ = _cycles(tmp_path, canvas, dy=0, dx=30, channels=1)
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out_tif, png, csv_path = al.align([p1, p2])
    assert out_tif == os.path.join(al.outdir, "aligned_allc.tif")
    assert csv_path == os.path.join(al.outdir, "aligned_manifest.csv")
    assert png is None
    assert os.path.exists(out_tif) and os.path.exists(csv_path)


def test_align_with_only_the_reference_copies_it_through(tmp_path, canvas):
    p1, _, ref, _ = _cycles(tmp_path, canvas, dy=0, dx=0, channels=2)
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out_tif, _, csv_path = al.align([p1])
    stack = tifffile.imread(out_tif)
    assert stack.shape[0] == 2
    assert np.array_equal(stack[0], ref)
    assert len(list(csv.DictReader(open(csv_path)))) == 2


def test_align_skips_an_image_that_cannot_be_matched(tmp_path, canvas):
    p1, p2, _, _ = _cycles(tmp_path, canvas, dy=0, dx=30, channels=1)
    blank = str(tmp_path / "cycles" / "blank.tif")
    write_cyx(blank, [np.zeros((512, 512), np.uint16)])
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out_tif, _, csv_path = al.align([p1, blank, p2])
    # the featureless image contributes no channels at all
    assert tifffile.imread(out_tif).shape[0] == 2
    rows = list(csv.DictReader(open(csv_path)))
    assert {os.path.basename(r["input_path"]) for r in rows} == {"cycle1.tif", "cycle2.tif"}


def test_align_uses_the_requested_nuclei_channel(tmp_path, canvas):
    """Channel 1 is a mirrored variant, so aligning on it gives a different
    (and here much worse) answer than aligning on channel 0."""
    p1, p2, _, _ = _cycles(tmp_path, canvas, dy=0, dx=40, channels=2)
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    _, _, c0 = al.align([p1, p2], nuclei_channel_indices=[0, 0],
                        csv_path=str(tmp_path / "c0.csv"),
                        out_tif=str(tmp_path / "a0.tif"))
    _, _, c1 = al.align([p1, p2], nuclei_channel_indices=[1, 1],
                        csv_path=str(tmp_path / "c1.csv"),
                        out_tif=str(tmp_path / "a1.tif"))
    tx0 = float(list(csv.DictReader(open(c0)))[2]["tx"])
    tx1 = float(list(csv.DictReader(open(c1)))[2]["tx"])
    assert tx0 == pytest.approx(40.0, abs=1.0)
    # the mirrored channel puts the same physical shift the other way round
    assert tx1 == pytest.approx(-40.0, abs=1.5)


def test_align_requires_matching_channel_index_lengths(tmp_path, canvas):
    p1, p2, _, _ = _cycles(tmp_path, canvas, dy=0, dx=10, channels=1)
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"))
    with pytest.raises(AssertionError, match="length must match"):
        al.align([p1, p2], nuclei_channel_indices=[0])
    with pytest.raises(AssertionError, match="at least one"):
        al.align([])


def test_align_promotes_to_the_widest_input_dtype(tmp_path, canvas):
    d = tmp_path / "mix"
    d.mkdir()
    ref = crop(canvas, 200, 200, 512)
    mov = crop(canvas, 200, 230, 512)
    p1, p2 = str(d / "a.tif"), str(d / "b.tif")
    write_cyx(p1, [(ref // 257).astype(np.uint8)])
    write_cyx(p2, [(mov // 257).astype(np.uint8)])
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5)
    out, _, _ = al.align([p1, p2], out_tif=str(tmp_path / "u8.tif"))
    assert tifffile.imread(out).dtype == np.uint8

    write_cyx(p2, [mov])                          # now uint16
    out2, _, _ = al.align([p1, p2], out_tif=str(tmp_path / "u16.tif"))
    assert tifffile.imread(out2).dtype == np.uint16


def test_align_with_rotation_allowed_recovers_the_angle(tmp_path, canvas):
    d = tmp_path / "rot"
    d.mkdir()
    base = crop(canvas, 150, 150, 620)
    M = cv2.getRotationMatrix2D((310.0, 310.0), 5.0, 1.0)
    warped = cv2.warpAffine(base.astype(np.float32), M, (620, 620))
    p1, p2 = str(d / "a.tif"), str(d / "b.tif")
    write_cyx(p1, [base[50:562, 50:562]])
    write_cyx(p2, [warped[50:562, 50:562].astype(np.uint16)])
    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5,
                              allow_rotation=True)
    _, _, csv_path = al.align([p1, p2])
    row = list(csv.DictReader(open(csv_path)))[1]
    assert float(row["theta_deg"]) == pytest.approx(5.0, abs=1.0)
    assert float(row["scale"]) == pytest.approx(1.0, abs=1e-3)


# ===========================================================================
# FOVAlignAndCropper
# ===========================================================================

def _mosaic_and_fovs(tmp_path, canvas, offsets=((150, 180), (300, 260)),
                     fov=256, mosaic_size=700, channels=1):
    root = tmp_path / "fovcase"
    (root / "fov").mkdir(parents=True, exist_ok=True)
    mos_path = str(root / "mosaic.tif")
    base = crop(canvas, 100, 100, mosaic_size)
    write_cyx(mos_path, [channel_variant(base, c) for c in range(2)])
    fovs = []
    for i, (fy, fx) in enumerate(offsets):
        p = str(root / "fov" / tile_name(site=i + 1, mag="20X"))
        sub = crop(canvas, 100 + fy, 100 + fx, fov)
        write_cyx(p, [channel_variant(sub, c) for c in range(channels)])
        fovs.append((p, fx, fy))
    return mos_path, str(root / "fov"), fovs


def test_fov_cropper_aligns_each_fov_into_the_mosaic_frame(tmp_path, canvas):
    mos, folder, fovs = _mosaic_and_fovs(tmp_path, canvas)
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5)
    csv_path = fa.run(mos, folder)
    assert csv_path == os.path.join(fa.outdir, "fov_align_manifest.csv")

    rows = {os.path.basename(r["fov_path"]): r for r in csv.DictReader(open(csv_path))}
    assert len(rows) == 2
    for p, fx, fy in fovs:
        r = rows[os.path.basename(p)]
        assert float(r["tx"]) == pytest.approx(fx, abs=1.0)
        assert float(r["ty"]) == pytest.approx(fy, abs=1.0)
        assert float(r["mosaic_x0"]) == pytest.approx(fx, abs=1.0)
        assert float(r["mosaic_y0"]) == pytest.approx(fy, abs=1.0)
        assert float(r["scale"]) == 1.0
        assert r["stitched_path"] == mos

        arr = np.load(r["npy_path"])
        # 1 FOV channel + 2 mosaic channels warped into the FOV frame
        assert arr.shape == (3, 256, 256)
        core = (slice(30, 226), slice(30, 226))
        assert np.corrcoef(arr[0][core].ravel(), arr[1][core].ravel())[0, 1] > 0.99
        # ...and the second mosaic channel is the mirrored variant, not a copy
        assert np.corrcoef(arr[0][core].ravel(), arr[2][core].ravel())[0, 1] < 0.5


def test_fov_cropper_honours_explicit_output_paths(tmp_path, canvas):
    mos, folder, _ = _mosaic_and_fovs(tmp_path, canvas, offsets=((150, 180),))
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5)
    out_csv = str(tmp_path / "custom.csv")
    npy_dir = str(tmp_path / "npys")
    assert fa.run(mos, folder, csv_path=out_csv, npy_dir=npy_dir) == out_csv
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 1
    assert os.path.dirname(rows[0]["npy_path"]) == npy_dir


def test_fov_cropper_skips_unreadable_and_featureless_files(tmp_path, canvas, capsys):
    mos, folder, _ = _mosaic_and_fovs(tmp_path, canvas, offsets=((150, 180),))
    with open(os.path.join(folder, "20X_c1_A1_Site-9_broken.tif"), "wb") as fh:
        fh.write(b"not a tiff")
    write_cyx(os.path.join(folder, "20X_c1_A1_Site-8.tif"),
              [np.zeros((256, 256), np.uint16)])
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5)
    rows = list(csv.DictReader(open(fa.run(mos, folder))))
    assert len(rows) == 1                        # only the good FOV survives
    assert "Site-1" in rows[0]["fov_path"]
    # the unreadable file is reported rather than dropped in silence
    assert "Site-9_broken.tif" in capsys.readouterr().out


def test_fov_cropper_scale_normalisation(tmp_path):
    for bad in (0.0, -3.0, None):
        fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), folder_image_scale=bad)
        assert fa.folder_image_scale == 1.0
    assert FOVAlignAndCropper(outdir=str(tmp_path / "fo"),
                              folder_image_scale=2.0).folder_image_scale == 2.0


def test_fov_cropper_lifts_the_downsampled_transform_with_the_known_scale(tmp_path, canvas):
    """A_full = s_known * A_ds and t_full = t_ds / downsample."""
    mos, folder, _ = _mosaic_and_fovs(tmp_path, canvas, offsets=((150, 180),))
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"), downsample=0.5,
                            allow_scale=True, allow_rotation=True)
    M_ds = np.array([[0.5, 0.0, 30.0], [0.0, 0.5, -12.0]], np.float32)
    fa._affine_from_pts = lambda *a, **k: (M_ds, None, 0.75)

    rows = list(csv.DictReader(open(fa.run(mos, folder, folder_image_scale=2.0))))
    assert len(rows) == 1
    assert float(rows[0]["tx"]) == pytest.approx(30.0 / 0.5)
    assert float(rows[0]["ty"]) == pytest.approx(-12.0 / 0.5)
    # linear part scaled by s_known -> 2.0 * 0.5 = 1.0
    assert float(rows[0]["scale"]) == pytest.approx(1.0, abs=1e-5)
    assert float(rows[0]["inlier_ratio"]) == pytest.approx(0.75)


def test_fov_cropper_proxies_reach_the_inner_aligner(tmp_path):
    fa = FOVAlignAndCropper(outdir=str(tmp_path / "fo"))
    assert fa._to_uint8(np.array([[0.0, 4.0]])).tolist() == [[0, 255]]
    pts, desc = fa._detect_and_describe(np.zeros((32, 32), np.uint8))
    assert pts.shape == (0, 2)
    small = {"pts": np.zeros((2, 2), np.float32), "desc": np.zeros((2, 32), np.uint8)}
    assert fa._match(small, small)[0].shape == (0, 2)
    assert fa._affine_from_pts(np.zeros((2, 2), np.float32),
                               np.zeros((2, 2), np.float32), 3.0) == (None, None, 0.0)
    R = fa._closest_rotation(np.array([[2.0, 0.0], [0.0, 2.0]], np.float32))
    assert np.allclose(R, np.eye(2), atol=1e-5)
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert fa._edge_zncc(a, a) == pytest.approx(1.0, abs=1e-3)


def test_fov_cropper_static_helpers(tmp_path):
    root = tmp_path / "imgs"
    (root / "sub").mkdir(parents=True)
    for rel in ("a.tif", "b.png", "sub/c.TIF"):
        (root / rel).write_bytes(b"x")
    assert [os.path.basename(p) for p in
            FOVAlignAndCropper._list_tifs(str(root), False, (".tif",))] == ["a.tif"]
    assert sorted(os.path.basename(p) for p in
                  FOVAlignAndCropper._list_tifs(str(root), True, (".tif",))) == ["a.tif", "c.TIF"]

    M = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -6.0]], np.float32)
    assert FOVAlignAndCropper._affine_to_3x3(M)[2].tolist() == [0.0, 0.0, 1.0]
    Mi = FOVAlignAndCropper._invert_affine(M)
    assert Mi[0, 2] == pytest.approx(-4.0, abs=1e-3)
    assert Mi[1, 2] == pytest.approx(6.0, abs=1e-3)


# ===========================================================================
# get_preprocess_ops_settings
# ===========================================================================

def test_get_preprocess_ops_settings_fills_defaults_in_place():
    settings = {}
    out = get_preprocess_ops_settings(settings)
    assert out is settings                        # mutated in place
    assert out["detector"] == "ORB"
    assert out["downsample"] == 0.5
    assert out["exts"] == [".tif", ".tiff"]
    assert out["collision"] == "rename"
    assert out["on_missing"] == "error"
    assert out["do_organize"] is True
    assert out["feature_cache_mode"] == "disk"
    assert out["mip"] is True
    assert out["src"] is None and out["dst_root"] is None
    assert "(?P<well>" in out["meta_regex"]


def test_get_preprocess_ops_settings_never_overrides_the_caller():
    out = get_preprocess_ops_settings({"detector": "SIFT", "downsample": 0.25,
                                       "src": "/data", "collision": "skip"})
    assert out["detector"] == "SIFT"
    assert out["downsample"] == 0.25
    assert out["src"] == "/data"
    assert out["collision"] == "skip"
    assert out["nfeatures"] == 8000               # untouched keys still filled


# ===========================================================================
# stitch_cycle_wells
# ===========================================================================

def _plate(tmp_path, canvas, wells=("A1", "B2"), n=2, tile=320, step=130,
           channels=2, sub="src"):
    src = tmp_path / sub
    src.mkdir(parents=True, exist_ok=True)
    for well in wells:
        for i in range(n):
            p = str(src / f"10X_c1_{well}_Site-{i + 1}.tif")
            piece = crop(canvas, 100, 100 + i * step, tile)
            write_cyx(p, [channel_variant(piece, c) for c in range(channels)])
    return str(src)


def _settings(src, dst, **kw):
    base = dict(src=src, dst_root=dst, verbose=False, mosaic=True,
                mosaic_min_score=0.2, max_site_gap=2, n_workers=2,
                downsample=0.5, plate="P1")
    base.update(kw)
    return base


def test_stitch_cycle_wells_organises_and_stitches_each_well(tmp_path, canvas):
    src = _plate(tmp_path, canvas)
    dst = str(tmp_path / "dst")
    res = stitch_cycle_wells(_settings(src, dst))

    assert set(res) == {"organized", "wells"}
    assert res["organized"]["moved"] == 4
    assert res["organized"]["linked"] == 0 and res["organized"]["skipped"] == 0
    assert sorted(res["wells"]) == ["A1", "B2"]
    assert os.listdir(src) == []                  # everything was moved out

    a1 = res["wells"]["A1"]
    assert a1["plate"] == "P1" and a1["well"] == "A1"
    assert a1["tiles_dir"] == os.path.join(dst, "A1", "A1")
    assert sorted(os.path.basename(t) for t in a1["tiles"]) == \
        ["10X_c1_A1_Site-1.tif", "10X_c1_A1_Site-2.tif"]
    assert all(os.path.exists(t) for t in a1["tiles"])
    assert os.path.basename(a1["pairwise_csv"]) == "P1_A1_pairs.csv"
    assert len(list(csv.DictReader(open(a1["pairwise_csv"])))) == 1
    assert a1["mosaic_csv"] and os.path.exists(a1["mosaic_csv"])
    assert a1["mosaic_tif"] is None               # multichannel mode
    assert a1["mosaic_cyx"] and os.path.exists(a1["mosaic_cyx"])

    mosaic = tifffile.imread(a1["mosaic_cyx"])
    assert mosaic.shape[0] == 2
    gt = canvas[100:100 + 320, 100:100 + 320 + 130]
    h, w = min(mosaic.shape[1], gt.shape[0]), min(mosaic.shape[2], gt.shape[1])
    assert np.corrcoef(mosaic[0, :h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99


def test_stitch_cycle_wells_single_channel_mode(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",), channels=1)
    dst = str(tmp_path / "dst")
    res = stitch_cycle_wells(_settings(src, dst, do_multichannel=False))
    a1 = res["wells"]["A1"]
    assert a1["mosaic_cyx"] is None
    assert a1["mosaic_tif"] and os.path.exists(a1["mosaic_tif"])
    assert a1["preview_png"] and os.path.exists(a1["preview_png"])
    assert tifffile.imread(a1["mosaic_tif"]).ndim == 2


def test_stitch_cycle_wells_requires_an_existing_src(tmp_path):
    with pytest.raises(ValueError, match="must point to an existing directory"):
        stitch_cycle_wells({"src": str(tmp_path / "nope")})
    with pytest.raises(ValueError, match="must point to an existing directory"):
        stitch_cycle_wells({})


def test_stitch_cycle_wells_with_no_matching_files(tmp_path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "readme.txt").write_text("nothing here")
    res = stitch_cycle_wells({"src": str(src), "dst_root": str(tmp_path / "d"),
                              "verbose": True})
    assert res == {"organized": {"moved": 0, "skipped": 0, "linked": 0,
                                 "by_well": {}}, "wells": {}}
    assert "no wells found" in capsys.readouterr().out


def test_stitch_cycle_wells_raises_on_an_unparseable_name(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    tifffile.imwrite(os.path.join(src, "mystery.tif"), np.zeros((8, 8), np.uint16))
    with pytest.raises(ValueError, match="Missing 'well' in filename"):
        stitch_cycle_wells(_settings(src, str(tmp_path / "dst")))


def test_stitch_cycle_wells_can_skip_unparseable_names(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    tifffile.imwrite(os.path.join(src, "mystery.tif"), np.zeros((8, 8), np.uint16))
    res = stitch_cycle_wells(_settings(src, str(tmp_path / "dst"),
                                       on_missing="skip", do_nuc_stitch=False))
    assert res["organized"]["skipped"] == 1
    assert res["organized"]["moved"] == 2
    assert res["wells"] == {}                    # stitching disabled


def test_stitch_cycle_wells_symlinks_when_not_organising(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    dst = str(tmp_path / "dst")
    res = stitch_cycle_wells(_settings(src, dst, do_organize=False))
    assert res["organized"]["moved"] == 0
    assert res["organized"]["linked"] == 2
    assert len(os.listdir(src)) == 2             # sources left in place
    links = os.path.join(dst, "_links", "A1")
    assert sorted(os.listdir(links)) == ["10X_c1_A1_Site-1.tif", "10X_c1_A1_Site-2.tif"]
    assert all(os.path.islink(os.path.join(links, f)) for f in os.listdir(links))
    # the per-well tile folder mirrors the sources as symlinks too
    for t in res["wells"]["A1"]["tiles"]:
        assert os.path.islink(t)


def test_stitch_cycle_wells_dry_run_moves_nothing(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    before = sorted(os.listdir(src))
    res = stitch_cycle_wells(_settings(src, str(tmp_path / "dst"), dry_run=True,
                                       do_nuc_stitch=False))
    assert sorted(os.listdir(src)) == before     # untouched
    assert res["organized"]["moved"] == 0
    assert len(res["organized"]["by_well"]["A1"]) == 2


def test_stitch_cycle_wells_renames_on_collision(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    dst = tmp_path / "dst"
    (dst / "A1").mkdir(parents=True)
    (dst / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"already here")
    res = stitch_cycle_wells(_settings(str(src), str(dst), do_nuc_stitch=False))
    names = sorted(os.path.basename(p) for p in res["organized"]["by_well"]["A1"])
    assert names == ["10X_c1_A1_Site-1_001.tif", "10X_c1_A1_Site-2.tif"]
    assert (dst / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() == b"already here"


def test_stitch_cycle_wells_can_skip_on_collision(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    dst = tmp_path / "dst"
    (dst / "A1").mkdir(parents=True)
    (dst / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"keep me")
    res = stitch_cycle_wells(_settings(str(src), str(dst), collision="skip",
                                       do_nuc_stitch=False))
    assert res["organized"]["skipped"] == 1
    assert res["organized"]["moved"] == 1
    assert (dst / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() == b"keep me"


def test_stitch_cycle_wells_can_overwrite_on_collision(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    dst = tmp_path / "dst"
    (dst / "A1").mkdir(parents=True)
    (dst / "A1" / "10X_c1_A1_Site-1.tif").write_bytes(b"stale")
    res = stitch_cycle_wells(_settings(str(src), str(dst), collision="overwrite",
                                       do_nuc_stitch=False))
    assert res["organized"]["moved"] == 2
    assert (dst / "A1" / "10X_c1_A1_Site-1.tif").read_bytes() != b"stale"


def test_stitch_cycle_wells_derives_and_sanitises_the_plate_id(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    dst = tmp_path / "plate 42/x"
    dst.mkdir(parents=True)
    res = stitch_cycle_wells(_settings(src, str(dst), plate=None,
                                       do_nuc_stitch=False))
    # falls back to the dst_root basename
    assert res["organized"]["moved"] == 2
    res2 = stitch_cycle_wells(_settings(_plate(tmp_path, canvas, wells=("A1",),
                                               sub="src2"),
                                        str(tmp_path / "d2"),
                                        plate="we ll/1"))
    assert res2["wells"]["A1"]["plate"] == "we_ll_1"


def test_stitch_cycle_wells_defaults_dst_root_to_src(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    res = stitch_cycle_wells({"src": src, "verbose": False, "do_nuc_stitch": False})
    assert os.path.isdir(os.path.join(src, "A1"))
    assert res["organized"]["moved"] == 2


def test_stitch_cycle_wells_non_recursive_ignores_subfolders(tmp_path, canvas):
    src = _plate(tmp_path, canvas, wells=("A1",))
    deep = os.path.join(src, "extra")
    os.makedirs(deep)
    write_cyx(os.path.join(deep, "10X_c1_C3_Site-1.tif"),
              [crop(canvas, 0, 0, 64)])
    res = stitch_cycle_wells(_settings(src, str(tmp_path / "dst"), recursive=False,
                                       do_nuc_stitch=False))
    assert sorted(res["organized"]["by_well"]) == ["A1"]


# ===========================================================================
# align_image_to_stitch
# ===========================================================================

def _stitched_root(tmp_path, canvas, layout="legacy", wells=("A1",)):
    root = tmp_path / "stitched"
    for well in wells:
        sub = "_stitch" if layout == "legacy" else "stitch"
        d = root / well / sub
        d.mkdir(parents=True)
        name = "mosaic_allc.tif" if layout == "legacy" else f"P1_{well}_mosaic_allc.tif"
        base = crop(canvas, 100, 100, 700)
        write_cyx(str(d / name), [channel_variant(base, c) for c in range(2)])
    return str(root)


def _align_src(tmp_path, canvas, wells=("A1",), offsets=((150, 180),)):
    d = tmp_path / "align20x"
    d.mkdir(parents=True, exist_ok=True)
    for well in wells:
        for i, (fy, fx) in enumerate(offsets):
            write_cyx(str(d / f"20X_c1_{well}_Site-{i + 1}.tif"),
                      [crop(canvas, 100 + fy, 100 + fx, 256)])
    return str(d)


def test_align_image_to_stitch_uses_the_legacy_layout(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas, "legacy")
    src = _align_src(tmp_path, canvas)
    res = align_image_to_stitch(root, src, relative_scale=1.0)
    assert list(res) == ["A1"]
    assert res["A1"]["mosaic"].endswith(os.path.join("_stitch", "mosaic_allc.tif"))
    rows = list(csv.DictReader(open(res["A1"]["manifest_csv"])))
    assert len(rows) == 1
    assert float(rows[0]["tx"]) == pytest.approx(180, abs=1.5)
    assert float(rows[0]["ty"]) == pytest.approx(150, abs=1.5)
    assert os.path.exists(rows[0]["npy_path"])
    assert np.load(rows[0]["npy_path"]).shape == (3, 256, 256)
    # the per-well link folder holds a symlink back to the source
    assert os.path.isdir(res["A1"]["align_folder"])
    assert len(os.listdir(res["A1"]["align_folder"])) == 1


def test_align_image_to_stitch_finds_stitch_cycle_wells_output(tmp_path, canvas):
    """Regression: the two halves of the pipeline used different folder names,
    so no well was ever aligned."""
    root = _stitched_root(tmp_path, canvas, "pipeline")
    src = _align_src(tmp_path, canvas)
    res = align_image_to_stitch(root, src, relative_scale=1.0)
    assert list(res) == ["A1"]
    assert res["A1"]["mosaic"].endswith("P1_A1_mosaic_allc.tif")
    assert len(list(csv.DictReader(open(res["A1"]["manifest_csv"])))) == 1


def test_align_image_to_stitch_requires_the_root(tmp_path):
    with pytest.raises(ValueError, match="stitch_dst_root does not exist"):
        align_image_to_stitch(str(tmp_path / "gone"), str(tmp_path))


def test_align_image_to_stitch_returns_empty_without_mosaics(tmp_path, canvas):
    root = tmp_path / "stitched"
    (root / "A1").mkdir(parents=True)
    (root / "loose.txt").write_text("x")
    assert align_image_to_stitch(str(root), _align_src(tmp_path, canvas)) == {}


def test_align_image_to_stitch_skips_wells_without_images(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas, "legacy", wells=("A1", "B2"))
    src = _align_src(tmp_path, canvas, wells=("A1",))
    res = align_image_to_stitch(root, src, relative_scale=1.0)
    assert list(res) == ["A1"]


def test_align_image_to_stitch_groups_by_well_and_sorts_by_site(tmp_path, canvas):
    root = _stitched_root(tmp_path, canvas, "legacy")
    src = _align_src(tmp_path, canvas, offsets=((150, 180), (300, 260)))
    res = align_image_to_stitch(root, src, relative_scale=1.0)
    linked = sorted(os.listdir(res["A1"]["align_folder"]))
    assert linked == ["20X_c1_A1_Site-1.tif", "20X_c1_A1_Site-2.tif"]


# ===========================================================================
# ops_preprocess
# ===========================================================================

def _geno_pheno(tmp_path, canvas, genotypes=("cycle1",)):
    geno = tmp_path / "geno"
    pheno = tmp_path / "pheno"
    pheno.mkdir(parents=True, exist_ok=True)
    for g in genotypes:
        d = geno / g
        d.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            piece = crop(canvas, 100, 100 + i * 130, 320)
            write_cyx(str(d / f"10X_c1_A1_Site-{i + 1}.tif"),
                      [channel_variant(piece, c) for c in range(2)])
    for i, (fy, fx) in enumerate(((150, 180), (40, 60))):
        write_cyx(str(pheno / f"20X_c1_A1_Site-{i + 1}.tif"),
                  [crop(canvas, 100 + fy, 100 + fx, 200)])
    return str(geno), str(pheno)


def test_ops_preprocess_stitches_then_aligns(tmp_path, canvas):
    geno, pheno = _geno_pheno(tmp_path, canvas)
    res = ops_preprocess(dict(phenotype_source=pheno, genotype_source=geno,
                              verbose=False, mosaic=True, mosaic_min_score=0.2,
                              max_site_gap=2, n_workers=2, downsample=0.5,
                              relative_scale=1.0, plate="P1"))
    assert set(res) == {"stitch", "align", "npy_out_root"}
    assert res["npy_out_root"] == os.path.join(pheno, "output")
    assert os.path.isdir(res["npy_out_root"])

    assert len(res["stitch"]) == 1
    summary = res["stitch"][0]
    assert summary["genotype_folder"].endswith("cycle1")
    assert sorted(summary["summary"]["wells"]) == ["A1"]

    assert len(res["align"]) == 1
    aligned = res["align"][0]["align"]
    assert list(aligned) == ["A1"]
    rows = list(csv.DictReader(open(aligned["A1"]["manifest_csv"])))
    assert len(rows) >= 1
    r = rows[0]
    assert float(r["tx"]) == pytest.approx(180, abs=2) or \
        float(r["tx"]) == pytest.approx(60, abs=2)
    assert os.path.exists(r["npy_path"])
    assert np.load(r["npy_path"]).shape[0] == 3      # 1 FOV + 2 mosaic channels


def test_ops_preprocess_accepts_a_list_of_genotype_folders(tmp_path, canvas):
    geno, pheno = _geno_pheno(tmp_path, canvas, genotypes=("c1", "c2"))
    folders = [os.path.join(geno, "c1"), os.path.join(geno, "c2")]
    res = ops_preprocess(dict(phenotype_source=pheno, genotype_source=folders,
                              verbose=False, mosaic=False, max_site_gap=2,
                              n_workers=2, downsample=0.5, plate="P1"))
    assert [s["genotype_folder"] for s in res["stitch"]] == folders
    assert len(res["align"]) == 2


def test_ops_preprocess_treats_a_leaf_folder_as_one_genotype(tmp_path, canvas):
    geno, pheno = _geno_pheno(tmp_path, canvas)
    leaf = os.path.join(geno, "cycle1")
    res = ops_preprocess(dict(phenotype_source=pheno, genotype_source=leaf,
                              verbose=False, mosaic=False, max_site_gap=2,
                              n_workers=2, downsample=0.5, plate="P1"))
    assert [s["genotype_folder"] for s in res["stitch"]] == [leaf]


def test_ops_preprocess_rejects_a_non_path_phenotype_source(tmp_path):
    with pytest.raises(ValueError, match="must be a path to a folder"):
        ops_preprocess({"phenotype_source": 17, "genotype_source": str(tmp_path)})


def test_ops_preprocess_rejects_a_bad_genotype_source(tmp_path):
    with pytest.raises(ValueError, match="must be a path or a list"):
        ops_preprocess({"phenotype_source": str(tmp_path), "genotype_source": 9})
