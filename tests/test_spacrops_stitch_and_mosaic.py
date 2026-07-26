"""spacr.spacrops — pairwise stitching, folder scoring and mosaic assembly.

Every tile in these tests is a crop of one large canvas taken at a known
offset, so the recovered translation, the placement of each tile in the
mosaic and the pixel content of the rendered image can all be checked
against a ground truth.
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

from spacr.spacrops import spacrStitcher
from tests.spacrops_synth import (blob_canvas, channel_variant, crop,
                                  row_of_tiles, tile_name)

TILE = 384
STEP = 150
Y0 = X0 = 100


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canvas():
    return blob_canvas(H=900, W=900, seed=1)


def _pair(tmp_path, canvas, dy, dx, tile=TILE):
    """Two tiles whose ground-truth B->A translation is (dx, dy)."""
    d = tmp_path / "pair"
    d.mkdir(exist_ok=True)
    pa = str(d / tile_name(site=1))
    pb = str(d / tile_name(site=2))
    tifffile.imwrite(pa, crop(canvas, 250, 250, tile))
    tifffile.imwrite(pb, crop(canvas, 250 + dy, 250 + dx, tile))
    return pa, pb


def _st(tmp_path, **kw):
    kw.setdefault("outdir", str(tmp_path / "out"))
    kw.setdefault("downsample", 0.5)
    kw.setdefault("save_qc", False)
    kw.setdefault("save_stitched_default", False)
    kw.setdefault("feature_cache_mode", "ram")
    return spacrStitcher(**kw)


def _row(a, b, dx, dy, score, theta=0.0, scale=1.0):
    """A pairwise-CSV row as csv.DictReader would produce it (all strings)."""
    return {"pathA": a, "pathB": b, "dx_px_full": str(dx), "dy_px_full": str(dy),
            "theta_deg": str(theta), "scale": str(scale), "score": str(score)}


def _assert_is_channel_one(plane, plane0, canvas):
    """The mosaic plane holds tile 1's *second* channel, not its first.

    Tiles land on the canvas with a sub-pixel offset, so compare by
    correlation over the strip that only the first tile covers.
    """
    reg = plane[2:TILE - 2, 2:STEP - 4].astype(float)
    ch1_gt = channel_variant(crop(canvas, Y0, X0, TILE), 1)[2:TILE - 2, 2:STEP - 4].astype(float)
    ch0_gt = crop(canvas, Y0, X0, TILE)[2:TILE - 2, 2:STEP - 4].astype(float)
    assert np.corrcoef(reg.ravel(), ch1_gt.ravel())[0, 1] > 0.99
    assert np.corrcoef(reg.ravel(), ch0_gt.ravel())[0, 1] < 0.5
    if plane0 is not None:
        other = plane0[2:TILE - 2, 2:STEP - 4].astype(float)
        assert np.corrcoef(reg.ravel(), other.ravel())[0, 1] < 0.5


# ===========================================================================
# stitch_pair — offset recovery
# ===========================================================================

@pytest.mark.parametrize("dy,dx", [(0, 150), (0, -150), (150, 0), (-150, 0), (60, 90)])
def test_stitch_pair_recovers_the_known_offset(tmp_path, canvas, dy, dx):
    pa, pb = _pair(tmp_path, canvas, dy, dx)
    r = _st(tmp_path).stitch_pair(pa, pb)
    assert r is not None
    assert r["dx_px_full"] == pytest.approx(dx, abs=0.5)
    assert r["dy_px_full"] == pytest.approx(dy, abs=0.5)
    # translation-only mode: no rotation or scale is ever reported
    assert r["theta_deg"] == 0.0 and r["scale"] == 1.0
    assert r["inlier_ratio"] > 0.4
    assert r["score"] > 0.3


def test_stitch_pair_reports_pair_metadata_and_canvas(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path).stitch_pair(pa, pb)
    assert r["pathA"] == pa and r["pathB"] == pb and r["channel_index"] == 0
    assert r["well"] == "A1" and r["siteA"] == 1 and r["siteB"] == 2
    assert r["weight"] == r["score"]
    assert r["seconds"] > 0
    # nothing was written, so the CSV canvas falls back to the DS dimensions
    assert (r["canvas_H"], r["canvas_W"]) == (TILE // 2, TILE // 2)
    assert r["stitched_full_tif"] == "" and r["qc_outline_png"] == ""
    # optional metrics stay empty unless all_scores is on
    assert r["edge_zncc_full"] == "" and r["fg_iou"] == ""


def test_stitch_pair_scores_overlap_far_above_an_unrelated_pair(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    good = _st(tmp_path).stitch_pair(pa, pb)

    other = blob_canvas(H=600, W=600, seed=99)
    pc = str(tmp_path / "pair" / tile_name(site=3))
    tifffile.imwrite(pc, crop(other, 50, 50, TILE))
    bad = _st(tmp_path).stitch_pair(pa, pc)

    assert good["score"] > 0.3
    assert bad is None or bad["score"] < 0.1


def test_stitch_pair_returns_none_when_there_are_too_few_matches(tmp_path, capsys):
    d = tmp_path / "blank"
    d.mkdir()
    pa, pb = str(d / tile_name(site=1)), str(d / tile_name(site=2))
    tifffile.imwrite(pa, np.zeros((128, 128), np.uint16))
    tifffile.imwrite(pb, np.zeros((128, 128), np.uint16))
    assert _st(tmp_path, verbose=True).stitch_pair(pa, pb) is None
    assert "<4 matches" in capsys.readouterr().out


def test_stitch_pair_returns_none_when_ransac_fails(tmp_path, canvas, capsys):
    st = _st(tmp_path, verbose=True)
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st._affine_from_pts = lambda *a, **k: (None, None, 0.0)   # RANSAC gives up
    assert st.stitch_pair(pa, pb) is None
    assert "RANSAC failed" in capsys.readouterr().out


def test_stitch_pair_falls_back_to_all_points_when_no_inliers(tmp_path, canvas):
    """inlier_mask=None must not crash the constrained recompute."""
    st = _st(tmp_path)
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    real = spacrStitcher._affine_from_pts

    def no_mask(ptsA, ptsB, thr):
        M, _, _ = real(ptsA, ptsB, thr)
        return M, None, 0.0

    st._affine_from_pts = no_mask
    r = st.stitch_pair(pa, pb)
    # with no inlier mask the translation is the mean over *every* match,
    # outliers included -- verify it is exactly that
    fA, fB = st._get_features(pa, 0), st._get_features(pb, 0)
    ptsA, ptsB = st._match(fA, fB)
    expect = (ptsA - ptsB).mean(axis=0) / st.downsample
    assert r["dx_px_full"] == pytest.approx(float(expect[0]), abs=1e-3)
    assert r["dy_px_full"] == pytest.approx(float(expect[1]), abs=1e-3)
    assert r["inlier_ratio"] == 0.0 and r["score"] == 0.0
    assert r["theta_deg"] == 0.0 and r["scale"] == 1.0


# ===========================================================================
# stitch_pair — writing the blended image
# ===========================================================================

def test_stitch_pair_writes_a_blended_image_matching_the_source_canvas(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st = _st(tmp_path, save_stitched_default=True, score_threshold=0.2)
    r = st.stitch_pair(pa, pb)

    img = tifffile.imread(r["stitched_full_tif"])
    assert img.dtype == np.uint16                       # input dtype preserved
    assert img.shape[1] == pytest.approx(TILE + STEP, abs=2)   # widened by the offset
    assert (r["canvas_W"], r["canvas_H"]) == (img.shape[1], img.shape[0])

    gt = canvas[250:250 + TILE, 250:250 + TILE + STEP]
    # the strip only tile A covers is reproduced bit-exactly
    assert np.array_equal(img[:TILE, :STEP - 1], gt[:, :STEP - 1])
    # and the whole blend tracks the original canvas
    w = min(img.shape[1], gt.shape[1])
    assert np.corrcoef(img[:TILE, :w].ravel(), gt[:, :w].ravel())[0, 1] > 0.999
    # regression: uncovered canvas must stay dark, not saturate to the dtype max
    assert int((img == 65535).sum()) == 0
    assert os.path.getsize(r["stitched_full_png"]) > 0


def test_stitch_pair_averages_the_overlap(tmp_path, canvas):
    """Where both tiles cover a pixel the result is their mean, not a sum."""
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st = _st(tmp_path, save_stitched_default=True, score_threshold=0.2)
    r = st.stitch_pair(pa, pb)
    img = tifffile.imread(r["stitched_full_tif"]).astype(np.float64)
    A = tifffile.imread(pa).astype(np.float64)
    overlap = img[:TILE, STEP + 5:TILE - 5]
    # both tiles carry (nearly) the same content there, so the mean equals A
    assert np.abs(overlap - A[:, STEP + 5:TILE - 5]).mean() < 12
    assert overlap.max() < A.max() * 1.5      # not a sum


def test_stitch_pair_below_threshold_writes_nothing(tmp_path, canvas, capsys):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st = _st(tmp_path, save_stitched_default=True, score_threshold=0.99, verbose=True)
    r = st.stitch_pair(pa, pb)
    assert r["stitched_full_tif"] == ""
    assert "no stitch" in capsys.readouterr().out
    assert not any(f.endswith("_stitched_full.tif") for f in os.listdir(st.outdir))


def test_stitch_pair_without_a_threshold_never_stitches(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st = _st(tmp_path, save_stitched_default=True, score_threshold=None)
    assert st.stitch_pair(pa, pb)["stitched_full_tif"] == ""


def test_stitch_pair_save_stitched_argument_overrides_the_default(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    st = _st(tmp_path, save_stitched_default=False, score_threshold=0.2)
    assert st.stitch_pair(pa, pb, save_stitched=True)["stitched_full_tif"] != ""


def test_stitch_pair_promotes_to_the_wider_input_dtype(tmp_path, canvas):
    d = tmp_path / "mixed"
    d.mkdir()
    pa, pb = str(d / tile_name(site=1)), str(d / tile_name(site=2))
    a = crop(canvas, 250, 250, TILE)
    b = crop(canvas, 250, 250 + STEP, TILE)
    tifffile.imwrite(pa, (a // 257).astype(np.uint8))
    tifffile.imwrite(pb, (b // 257).astype(np.uint8))
    st = _st(tmp_path, save_stitched_default=True, score_threshold=0.2)
    r = st.stitch_pair(pa, pb)
    assert tifffile.imread(r["stitched_full_tif"]).dtype == np.uint8

    tifffile.imwrite(pb, b)                              # now uint16
    st2 = _st(tmp_path, save_stitched_default=True, score_threshold=0.2,
              outdir=str(tmp_path / "out2"))
    r2 = st2.stitch_pair(pa, pb)
    assert tifffile.imread(r2["stitched_full_tif"]).dtype == np.uint16


# ===========================================================================
# stitch_pair — QC gating
# ===========================================================================

def test_stitch_pair_writes_the_qc_overlay(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path, save_qc=True).stitch_pair(pa, pb)
    assert r["qc_outline_png"].endswith("__qc_outlines.png")
    assert os.path.getsize(r["qc_outline_png"]) > 0


def test_stitch_pair_force_no_qc_suppresses_the_overlay(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path, save_qc=True).stitch_pair(pa, pb, force_no_qc=True)
    assert r["qc_outline_png"] == ""


def test_stitch_pair_qc_gate_compares_against_the_score(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    below = _st(tmp_path, save_qc=True).stitch_pair(pa, pb, qc_only_if_score_ge=0.99)
    assert below["qc_outline_png"] == ""
    above = _st(tmp_path, save_qc=True).stitch_pair(pa, pb, qc_only_if_score_ge=0.01)
    assert above["qc_outline_png"] != ""


def test_stitch_pair_malformed_qc_gate_falls_back_to_emitting_qc(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path, save_qc=True).stitch_pair(pa, pb, qc_only_if_score_ge="not-a-number")
    assert r["qc_outline_png"] != ""


# ===========================================================================
# stitch_pair — all_scores metrics
# ===========================================================================

def test_stitch_pair_all_scores_measures_the_full_res_overlap(tmp_path, canvas):
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path, all_scores=True).stitch_pair(pa, pb)
    assert r["edge_zncc_full"] > 0.2
    assert r["fg_corr"] > 0.9                       # same content in the overlap
    assert 0.0 < r["fg_iou"] < 1.0
    assert r["fg_dice"] == pytest.approx(
        2 * r["fg_iou"] / (1 + r["fg_iou"]), rel=1e-3)
    assert r["fg_xor_frac"] == pytest.approx(1.0 - r["fg_iou"], abs=1e-6)
    assert 0.0 < r["fg_xor_entropy"] <= 1.0


def test_stitch_pair_all_scores_handles_an_empty_foreground(tmp_path, canvas):
    """outline_source='none' makes every mask empty: the union is zero."""
    pa, pb = _pair(tmp_path, canvas, 0, STEP)
    r = _st(tmp_path, all_scores=True, outline_source="none").stitch_pair(pa, pb)
    assert r["fg_iou"] == 0.0 and r["fg_dice"] == 0.0
    assert r["fg_xor_frac"] == 1.0 and r["fg_xor_entropy"] == 0.0
    assert r["fg_corr"] == 0.0                       # fewer than 25 overlap pixels
    assert r["score"] == 0.0                         # DS mask is empty too
    assert r["edge_zncc_full"] > 0.2                 # still computed unmasked


# ===========================================================================
# stitch_pair — rotation / scale modes
# ===========================================================================

def _rotated_pair(tmp_path, canvas, deg, scale=1.0, tile=384):
    d = tmp_path / "rot"
    d.mkdir(exist_ok=True)
    base = crop(canvas, 200, 200, tile + 120)
    M = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), deg, scale)
    warped = cv2.warpAffine(base.astype(np.float32), M, (base.shape[1], base.shape[0]),
                            flags=cv2.INTER_LINEAR)
    pa = str(d / tile_name(site=1))
    pb = str(d / tile_name(site=2))
    tifffile.imwrite(pa, base[60:60 + tile, 60:60 + tile])
    tifffile.imwrite(pb, warped[60:60 + tile, 60:60 + tile].astype(np.uint16))
    return pa, pb


def test_stitch_pair_translation_only_zeroes_rotation(tmp_path, canvas):
    pa, pb = _rotated_pair(tmp_path, canvas, 6.0)
    r = _st(tmp_path).stitch_pair(pa, pb)
    assert r["theta_deg"] == 0.0 and r["scale"] == 1.0


def test_stitch_pair_allow_rotation_recovers_the_angle(tmp_path, canvas):
    pa, pb = _rotated_pair(tmp_path, canvas, 6.0)
    r = _st(tmp_path, allow_rotation=True).stitch_pair(pa, pb)
    # B was rotated with cv2.getRotationMatrix2D(+6), whose linear part is the
    # transpose of the maths-convention rotation; the recovered B->A transform
    # therefore reads back as +6 degrees.
    assert r["theta_deg"] == pytest.approx(6.0, abs=1.0)
    assert r["scale"] == pytest.approx(1.0, abs=1e-3)     # unit scale enforced


def test_stitch_pair_allow_scale_recovers_the_zoom(tmp_path, canvas):
    pa, pb = _rotated_pair(tmp_path, canvas, 0.0, scale=1.2)
    r = _st(tmp_path, allow_scale=True, allow_rotation=True).stitch_pair(pa, pb)
    assert r["scale"] == pytest.approx(1 / 1.2, abs=0.05)


# ===========================================================================
# run_folder
# ===========================================================================

CSV_HEADER = ["pathA", "pathB", "channel_index", "score", "inlier_ratio",
              "edge_zncc_fg", "edge_zncc_full", "fg_corr", "fg_iou", "fg_dice",
              "fg_xor_frac", "fg_xor_entropy", "dy_px_full", "dx_px_full",
              "theta_deg", "scale", "canvas_H", "canvas_W", "qc_outline_png",
              "stitched_full_tif", "stitched_full_png", "seconds", "well",
              "siteA", "siteB"]


def test_run_folder_on_an_empty_folder_writes_a_header_only_csv(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = str(tmp_path / "nested" / "pairs.csv")
    st = _st(tmp_path, verbose=True)
    assert st.run_folder(str(empty), out) == out
    with open(out) as fh:
        rdr = csv.reader(fh)
        assert next(rdr) == CSV_HEADER
        assert list(rdr) == []
    assert "No files found" in capsys.readouterr().out


def test_run_folder_scores_every_neighbouring_pair(tmp_path, canvas, capsys):
    tiles, gt = row_of_tiles(str(tmp_path / "tiles"), canvas=canvas, n=3,
                             tile=TILE, step=STEP)
    out = str(tmp_path / "pairs.csv")
    st = _st(tmp_path, verbose=True)
    st.run_folder(str(tmp_path / "tiles"), out, max_site_gap=1, stitch=False)

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2                     # (1,2) and (2,3)
    by_pair = {(os.path.basename(r["pathA"]), os.path.basename(r["pathB"])): r
               for r in rows}
    for a, b in [(0, 1), (1, 2)]:
        r = by_pair[(os.path.basename(tiles[a]), os.path.basename(tiles[b]))]
        assert float(r["dx_px_full"]) == pytest.approx(STEP, abs=0.5)
        assert abs(float(r["dy_px_full"])) < 0.5
        assert float(r["score"]) > 0.3
        assert r["well"] == "A1"
    printed = capsys.readouterr().out
    assert "candidate pairs=2" in printed
    # an auto threshold was derived and plotted
    assert os.path.exists(os.path.join(st.outdir, "score_sorted_line.png"))


def test_run_folder_same_well_only_false_compares_everything(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP, well="A1")
    row_of_tiles(d, canvas=canvas, n=1, tile=TILE, step=STEP, well="B2",
                 first_site=5)
    out = str(tmp_path / "pairs.csv")
    _st(tmp_path, verbose=True).run_folder(d, out, same_well_only=False, stitch=False)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3                     # C(3,2) - every pair, wells ignored


def test_run_folder_groups_by_well_when_same_well_only(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP, well="A1")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP, well="B2")
    out = str(tmp_path / "pairs.csv")
    _st(tmp_path).run_folder(d, out, same_well_only=True, max_site_gap=1, stitch=False)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert {r["well"] for r in rows} == {"A1", "B2"}


def test_run_folder_second_pass_stitches_the_winners(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    st = _st(tmp_path)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=True)
    stitched = [f for f in os.listdir(st.outdir) if f.endswith("__stitched_full.tif")]
    assert len(stitched) >= 1
    img = tifffile.imread(os.path.join(st.outdir, stitched[0]))
    assert img.shape[1] == TILE + STEP


def test_run_folder_accepts_a_meta_regex_override(tmp_path, canvas):
    d = tmp_path / "odd"
    d.mkdir()
    for i in range(2):
        tifffile.imwrite(str(d / f"plateX__well-A1__f{i + 1}.tif"),
                         crop(canvas, Y0, X0 + i * STEP, TILE))
    out = str(tmp_path / "pairs.csv")
    st = _st(tmp_path)
    st.run_folder(str(d), out, max_site_gap=1, stitch=False,
                  meta_regex=r"well-(?P<well>[A-H]\d+)__f(?P<site>\d+)\.tif$")
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 1 and rows[0]["well"] == "A1"
    assert float(rows[0]["dx_px_full"]) == pytest.approx(STEP, abs=0.5)


def test_run_folder_skips_pairs_that_raise(tmp_path, canvas, capsys):
    d = tmp_path / "tiles"
    row_of_tiles(str(d), canvas=canvas, n=2, tile=TILE, step=STEP)
    with open(d / tile_name(site=3), "wb") as fh:
        fh.write(b"not a tiff at all")
    out = str(tmp_path / "pairs.csv")
    st = _st(tmp_path, verbose=True)
    st.run_folder(str(d), out, max_site_gap=2, stitch=False)
    rows = list(csv.DictReader(open(out)))
    # (1,2) survives; every pair involving the broken file is dropped
    assert len(rows) == 1
    assert "failed" in capsys.readouterr().out


def test_run_folder_suppresses_qc_when_there_are_too_many_pairs(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    st = _st(tmp_path, save_qc=True)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  qc_pairs_threshold=0)
    assert [f for f in os.listdir(st.outdir) if f.endswith("qc_outlines.png")] == []


def test_run_folder_qc_gate_uses_the_threshold_when_many_pairs(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    st = _st(tmp_path, save_qc=True, score_threshold=0.3)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  qc_pairs_threshold=0, qc_only_above_threshold_when_many=True)
    qc = [f for f in os.listdir(st.outdir) if f.endswith("qc_outlines.png")]
    assert len(qc) == 2                       # both pairs clear 0.3


# ===========================================================================
# run_folder -> mosaic
# ===========================================================================

def test_run_folder_builds_a_single_channel_mosaic(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    mos = str(tmp_path / "mosaic.tif")
    st = _st(tmp_path)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  mosaic=True, mosaic_out=mos, mosaic_min_score=0.3,
                  mosaic_csv_out=str(tmp_path / "mosaic.csv"))

    img = tifffile.imread(mos)
    assert img.dtype == np.uint16
    assert img.shape[1] == pytest.approx(TILE + 2 * STEP, abs=2)
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h = min(img.shape[0], gt.shape[0])
    assert np.corrcoef(img[:h].ravel(), gt[:h].ravel())[0, 1] > 0.99
    assert os.path.exists(str(tmp_path / "mosaic.png"))

    manifest = list(csv.DictReader(open(str(tmp_path / "mosaic.csv"))))
    assert len(manifest) == 3
    xs = sorted(float(r["canvas_x"]) for r in manifest)
    assert xs[0] == pytest.approx(0.0, abs=1.0)
    assert xs[1] == pytest.approx(STEP, abs=1.0)
    assert xs[2] == pytest.approx(2 * STEP, abs=1.5)


def test_run_folder_mosaic_without_an_output_path_writes_only_the_manifest(tmp_path, canvas):
    """Regression: a None mosaic_out used to blow up in os.path.splitext."""
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP)
    mcsv = str(tmp_path / "mosaic.csv")
    st = _st(tmp_path, verbose=True)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  mosaic=True, mosaic_out=None, mosaic_min_score=0.3,
                  mosaic_csv_out=mcsv)
    assert len(list(csv.DictReader(open(mcsv)))) == 3
    assert not os.path.exists(str(tmp_path / "mosaic.tif"))


def test_run_folder_builds_a_multichannel_mosaic(tmp_path, canvas):
    d = str(tmp_path / "tiles")
    tiles, _ = row_of_tiles(d, canvas=canvas, n=3, tile=TILE, step=STEP, channels=2)
    mos = str(tmp_path / "mosaic_allc.tif")
    st = _st(tmp_path)
    st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=1, stitch=False,
                  mosaic=True, mosaic_out=mos, mosaic_all_channels=True,
                  mosaic_min_score=0.3, mosaic_csv_out=str(tmp_path / "mosaic.csv"))

    img = tifffile.imread(mos)
    assert img.ndim == 3 and img.shape[0] == 2
    gt0 = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h = min(img.shape[1], gt0.shape[0])
    w = min(img.shape[2], gt0.shape[1])
    assert np.corrcoef(img[0, :h, :w].ravel(), gt0[:h, :w].ravel())[0, 1] > 0.99
    # channel 1 carries the (different) second plane of the first tile
    _assert_is_channel_one(img[1], img[0], canvas)


# ===========================================================================
# grid-step estimation and edge pruning
# ===========================================================================

def test_estimate_grid_steps_medians_the_high_scoring_edges(tmp_path):
    st = _st(tmp_path)
    rows = [_row("a", "b", 150, 0, 0.9),
            _row("b", "c", 154, 0, 0.9),
            _row("a", "c", 0, 300, 0.9),
            _row("c", "d", 0, 296, 0.9),
            _row("d", "e", 999, 0, 0.01),        # below min_score -> ignored
            _row("e", "f", 12, 0, "")]           # blank score -> ignored
    sx, sy = st._estimate_grid_steps(rows, min_score=0.5)
    assert sx == pytest.approx(152.0)
    assert sy == pytest.approx(298.0)


def test_estimate_grid_steps_returns_zero_without_usable_edges(tmp_path):
    st = _st(tmp_path, verbose=True)
    assert st._estimate_grid_steps([_row("a", "b", 100, 100, 0.9)], 0.5) == (0.0, 0.0)


def test_compute_mosaic_transforms_lays_out_a_row_of_tiles(tmp_path):
    st = _st(tmp_path)
    rows = [_row("/t/1.tif", "/t/2.tif", STEP, 0, 0.9),
            _row("/t/2.tif", "/t/3.tif", STEP, 0, 0.8)]
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert len(used) == 2                      # a spanning tree over 3 nodes
    assert sorted(T) == ["/t/1.tif", "/t/2.tif", "/t/3.tif"]
    # tile 2 is the hub, so 1 sits STEP to its left and 3 STEP to its right
    assert T["/t/2.tif"][0, 2] == pytest.approx(0.0, abs=1e-4)
    assert T["/t/1.tif"][0, 2] == pytest.approx(-STEP, abs=1e-3)
    assert T["/t/3.tif"][0, 2] == pytest.approx(STEP, abs=1e-3)
    for M in T.values():
        assert np.allclose(M[:, :2], np.eye(2), atol=1e-4)
        assert M[1, 2] == pytest.approx(0.0, abs=1e-3)


def test_compute_mosaic_transforms_places_a_two_by_two_grid(tmp_path):
    st = _st(tmp_path)
    p = [f"/t/{i}.tif" for i in range(4)]
    rows = [_row(p[0], p[1], STEP, 0, 0.9),      # right neighbour
            _row(p[0], p[2], 0, STEP, 0.85),     # below
            _row(p[1], p[3], 0, STEP, 0.8),
            _row(p[2], p[3], STEP, 0, 0.75)]
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert len(T) == 4 and len(used) == 3
    origin = T[p[0]][:, 2]
    rel = {k: (T[k][:, 2] - origin) for k in T}
    assert rel[p[1]] == pytest.approx([STEP, 0.0], abs=1e-2)
    assert rel[p[2]] == pytest.approx([0.0, STEP], abs=1e-2)
    assert rel[p[3]] == pytest.approx([STEP, STEP], abs=1e-2)


def test_compute_mosaic_transforms_with_no_rows(tmp_path):
    assert _st(tmp_path)._compute_mosaic_transforms([], 0.5) == ({}, [])


def test_compute_mosaic_transforms_drops_diagonal_edges(tmp_path):
    st = _st(tmp_path)
    rows = [_row("/a.tif", "/b.tif", 150, 150, 0.9)]     # 45 degrees -> no bin
    T, used = st._compute_mosaic_transforms(rows, 0.5)
    assert used == []
    assert len(T) == 1                       # only the isolated root survives


def test_compute_mosaic_transforms_rejects_rotation_when_disallowed(tmp_path):
    st = _st(tmp_path, allow_rotation=False)
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.9, theta=20.0)]
    assert st._compute_mosaic_transforms(rows, 0.5, rot_tol_deg=5.0)[1] == []
    st2 = _st(tmp_path, allow_rotation=True)
    assert len(st2._compute_mosaic_transforms(rows, 0.5)[1]) == 1


def test_compute_mosaic_transforms_rejects_scale_when_disallowed(tmp_path):
    st = _st(tmp_path, allow_scale=False)
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.9, scale=1.5)]
    assert st._compute_mosaic_transforms(rows, 0.5, scale_tol=0.03)[1] == []


def test_compute_mosaic_transforms_rejects_off_grid_steps(tmp_path):
    st = _st(tmp_path, verbose=True)
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.9),
            _row("/b.tif", "/c.tif", 150, 0, 0.9),
            _row("/c.tif", "/d.tif", 40, 0, 0.9)]        # way off the 150 step
    T, used = st._compute_mosaic_transforms(rows, 0.5, step_tol_frac=0.25)
    assert "/d.tif" not in T
    assert len(used) == 2


def test_compute_mosaic_transforms_keeps_the_best_edge_per_direction(tmp_path):
    st = _st(tmp_path)
    # the same pair scored twice: only the better row may become an edge
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.5),
            _row("/a.tif", "/b.tif", 150, 0, 0.95)]
    T, used = st._compute_mosaic_transforms(rows, 0.1)
    assert [e[2] for e in used] == [0.95]
    assert sorted(T) == ["/a.tif", "/b.tif"]


def test_compute_mosaic_transforms_ignores_sub_threshold_rows(tmp_path):
    st = _st(tmp_path)
    rows = [_row("/a.tif", "/b.tif", 150, 0, 0.2)]
    T, used = st._compute_mosaic_transforms(rows, 0.9)
    assert used == [] and len(T) == 1


# ===========================================================================
# render_mosaic_from_csv
# ===========================================================================

def _pairs_csv(tmp_path, canvas, n=3, channels=1, well="A1"):
    d = str(tmp_path / "tiles")
    tiles, _ = row_of_tiles(d, canvas=canvas, n=n, tile=TILE, step=STEP,
                            channels=channels, well=well)
    out = str(tmp_path / "pairs.csv")
    st = _st(tmp_path)
    st.run_folder(d, out, max_site_gap=1, stitch=False)
    return st, out, tiles


def test_render_mosaic_from_csv_reproduces_the_canvas(tmp_path, canvas):
    st, pairs, tiles = _pairs_csv(tmp_path, canvas)
    out_tif = str(tmp_path / "m.tif")
    out_png = str(tmp_path / "m.png")
    got = st.render_mosaic_from_csv(pairs, out_tif, out_png, min_score=0.3,
                                    out_csv=str(tmp_path / "m.csv"))
    assert got == (out_tif, out_png)
    img = tifffile.imread(out_tif)
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h = min(img.shape[0], gt.shape[0])
    assert np.corrcoef(img[:h].ravel(), gt[:h].ravel())[0, 1] > 0.99
    assert os.path.getsize(out_png) > 0
    # no saturated seam where nothing is covered
    assert int((img == 65535).sum()) == 0


def test_render_mosaic_from_csv_auto_threshold(tmp_path, canvas, capsys):
    st, pairs, _ = _pairs_csv(tmp_path, canvas)
    st.verbose = True
    out_tif = str(tmp_path / "m.tif")
    st.render_mosaic_from_csv(pairs, out_tif, min_score=None)
    assert "auto min_score" in capsys.readouterr().out
    assert os.path.exists(out_tif)


def test_render_mosaic_manifest_only_mode(tmp_path, canvas):
    st, pairs, tiles = _pairs_csv(tmp_path, canvas)
    mcsv = str(tmp_path / "m.csv")
    assert st.render_mosaic_from_csv(pairs, None, "ignored.png", min_score=0.3,
                                     out_csv=mcsv) == (None, None)
    rows = list(csv.DictReader(open(mcsv)))
    assert [r["path"] for r in rows] == sorted(tiles)
    assert set(rows[0]) == {"path", "H", "W", "M00", "M01", "M02", "M10", "M11",
                            "M12", "canvas_x", "canvas_y", "best_pair_score"}
    assert int(rows[0]["H"]) == TILE and int(rows[0]["W"]) == TILE
    assert float(rows[0]["M00"]) == pytest.approx(1.0)
    assert not os.path.exists(str(tmp_path / "m.tif"))


def test_render_mosaic_requires_an_output_when_not_manifest_only(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas)
    with pytest.raises(ValueError, match="out_tif is None"):
        st.render_mosaic_from_csv(pairs, None, min_score=0.3)


def test_render_mosaic_rejects_an_empty_csv(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas)
    blank = str(tmp_path / "blank.csv")
    with open(blank, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerow({k: "" for k in CSV_HEADER})     # unusable row
    with pytest.raises(RuntimeError, match="no usable rows"):
        st.render_mosaic_from_csv(blank, str(tmp_path / "x.tif"))


def test_render_mosaic_rejects_a_fully_pruned_graph(tmp_path, canvas, monkeypatch):
    st, pairs, _ = _pairs_csv(tmp_path, canvas)
    monkeypatch.setattr(st, "_compute_mosaic_transforms", lambda *a, **k: ({}, []))
    with pytest.raises(RuntimeError, match="no nodes remained"):
        st.render_mosaic_from_csv(pairs, str(tmp_path / "x.tif"), min_score=0.3)


def test_render_mosaic_verbose_reports_the_canvas(tmp_path, canvas, capsys):
    st, pairs, _ = _pairs_csv(tmp_path, canvas)
    st.verbose = True
    st.render_mosaic_from_csv(pairs, str(tmp_path / "m.tif"), min_score=0.3)
    out = capsys.readouterr().out
    assert "nodes in mosaic: 3" in out and "canvas =" in out


# ===========================================================================
# mosaic_all_channels_from_csv
# ===========================================================================

def test_mosaic_all_channels_writes_a_cyx_stack(tmp_path, canvas):
    st, pairs, tiles = _pairs_csv(tmp_path, canvas, channels=2)
    out = str(tmp_path / "allc.tif")
    assert st.mosaic_all_channels_from_csv(pairs, out, min_score=0.3) == out
    img = tifffile.imread(out)
    assert img.shape[0] == 2
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h, w = min(img.shape[1], gt.shape[0]), min(img.shape[2], gt.shape[1])
    assert np.corrcoef(img[0, :h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99
    _assert_is_channel_one(img[1], img[0], canvas)


def test_mosaic_all_channels_honours_an_explicit_channel_order(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    out = str(tmp_path / "rev.tif")
    st.mosaic_all_channels_from_csv(pairs, out, min_score=0.3,
                                    channel_index_order=[1])
    img = tifffile.imread(out)
    assert img.shape[0] == 1
    _assert_is_channel_one(img[0], None, canvas)


def test_mosaic_all_channels_channel_count_limits_the_output(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    out = str(tmp_path / "one.tif")
    st.mosaic_all_channels_from_csv(pairs, out, min_score=0.3, channel_count=1)
    # Measured: the writer always emits (1, H, W). The old
    # `ndim == 2 or ...` arm was dead and only absorbed a surprise.
    assert tifffile.imread(out).shape[0] == 1


def test_mosaic_all_channels_rejects_a_zero_channel_count(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    with pytest.raises(ValueError, match="channel_count resolved to 0"):
        st.mosaic_all_channels_from_csv(pairs, str(tmp_path / "x.tif"),
                                        min_score=0.3, channel_count=0)


def test_mosaic_all_channels_manifest_only(tmp_path, canvas):
    st, pairs, tiles = _pairs_csv(tmp_path, canvas, channels=2)
    mcsv = str(tmp_path / "m.csv")
    assert st.mosaic_all_channels_from_csv(pairs, None, min_score=0.3,
                                           out_csv=mcsv) is None
    rows = list(csv.DictReader(open(mcsv)))
    assert [r["path"] for r in rows] == sorted(tiles)
    assert float(rows[1]["canvas_x"]) == pytest.approx(STEP, abs=1.0)


def test_mosaic_all_channels_requires_an_output_path(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    with pytest.raises(ValueError, match="out_tif is None"):
        st.mosaic_all_channels_from_csv(pairs, None, min_score=0.3)


def test_mosaic_all_channels_rejects_an_empty_csv(tmp_path, canvas):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    blank = str(tmp_path / "blank.csv")
    with open(blank, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerow({k: "" for k in CSV_HEADER})
    with pytest.raises(RuntimeError, match="no usable rows"):
        st.mosaic_all_channels_from_csv(blank, str(tmp_path / "x.tif"))


def test_mosaic_all_channels_rejects_a_fully_pruned_graph(tmp_path, canvas, monkeypatch):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    monkeypatch.setattr(st, "_compute_mosaic_transforms", lambda *a, **k: ({}, []))
    with pytest.raises(RuntimeError, match="no nodes remained"):
        st.mosaic_all_channels_from_csv(pairs, str(tmp_path / "x.tif"), min_score=0.3)


def test_mosaic_all_channels_auto_threshold(tmp_path, canvas, capsys):
    st, pairs, _ = _pairs_csv(tmp_path, canvas, channels=2)
    st.verbose = True
    st.mosaic_all_channels_from_csv(pairs, str(tmp_path / "m.tif"))
    out = capsys.readouterr().out
    assert "auto min_score" in out and "channel plan" in out


# ===========================================================================
# build_multichannel_mosaic_from_manifest
# ===========================================================================

def _manifest(tmp_path, canvas, channels=2):
    st, pairs, tiles = _pairs_csv(tmp_path, canvas, channels=channels)
    mcsv = str(tmp_path / "manifest.csv")
    st.render_mosaic_from_csv(pairs, None, min_score=0.3, out_csv=mcsv)
    return st, mcsv, tiles


def test_build_multichannel_mosaic_from_manifest_matches_the_canvas(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    out = str(tmp_path / "mc.tif")
    png = str(tmp_path / "mc.png")
    assert st.build_multichannel_mosaic_from_manifest(
        mcsv, out, out_png=png, tmp_dir=str(tmp_path / "tmp")) == out
    img = tifffile.imread(out)
    assert img.shape[0] == 2
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h, w = min(img.shape[1], gt.shape[0]), min(img.shape[2], gt.shape[1])
    assert np.corrcoef(img[0, :h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99
    assert os.path.getsize(png) > 0


def test_build_multichannel_mosaic_overwrite_blend(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    out = str(tmp_path / "ow.tif")
    st.build_multichannel_mosaic_from_manifest(mcsv, out, blend="overwrite",
                                               tmp_dir=str(tmp_path / "tmp"))
    img = tifffile.imread(out)
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + 2 * STEP]
    h, w = min(img.shape[1], gt.shape[0]), min(img.shape[2], gt.shape[1])
    assert np.corrcoef(img[0, :h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99


def test_build_multichannel_mosaic_rejects_an_unknown_blend(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    with pytest.raises(ValueError, match="blend must be"):
        st.build_multichannel_mosaic_from_manifest(mcsv, str(tmp_path / "x.tif"),
                                                   blend="average",
                                                   tmp_dir=str(tmp_path / "tmp"))


def test_build_multichannel_mosaic_selects_explicit_channels(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    out = str(tmp_path / "one.tif")
    st.build_multichannel_mosaic_from_manifest(mcsv, out, channel_indices=[1],
                                               tmp_dir=str(tmp_path / "tmp"))
    img = tifffile.imread(out)
    assert img.ndim == 2 or img.shape[0] == 1
    plane = img if img.ndim == 2 else img[0]
    _assert_is_channel_one(plane, None, canvas)


def test_build_multichannel_mosaic_requires_the_manifest_columns(tmp_path, canvas):
    st, _, _ = _manifest(tmp_path, canvas)
    bad = str(tmp_path / "bad.csv")
    with open(bad, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "H", "W"])
        w.writeheader()
        w.writerow({"path": "/x.tif", "H": 1, "W": 1})
    with pytest.raises(RuntimeError, match="missing columns"):
        st.build_multichannel_mosaic_from_manifest(bad, str(tmp_path / "x.tif"))


def test_build_multichannel_mosaic_rejects_a_manifest_of_missing_files(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    rows = list(csv.DictReader(open(mcsv)))
    gone = str(tmp_path / "gone.csv")
    with open(gone, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r["path"] = "/nowhere/" + os.path.basename(r["path"])
            w.writerow(r)
    with pytest.raises(RuntimeError, match="no usable rows"):
        st.build_multichannel_mosaic_from_manifest(gone, str(tmp_path / "x.tif"))


def test_build_multichannel_mosaic_skips_rows_with_bad_numbers(tmp_path, canvas):
    st, mcsv, tiles = _manifest(tmp_path, canvas)
    rows = list(csv.DictReader(open(mcsv)))
    rows[0]["M00"] = "not-a-float"
    patched = str(tmp_path / "patched.csv")
    with open(patched, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out = str(tmp_path / "two.tif")
    st.build_multichannel_mosaic_from_manifest(patched, out,
                                               tmp_dir=str(tmp_path / "tmp"))
    img = tifffile.imread(out)
    # one tile dropped -> the canvas is narrower than the full three-tile run
    assert img.shape[2] < TILE + 2 * STEP


def test_build_multichannel_mosaic_defaults_its_tmp_dir(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    out = str(tmp_path / "dflt.tif")
    st.build_multichannel_mosaic_from_manifest(mcsv, out)
    # the stitcher is in RAM cache mode, so the memmap lands beside the output
    assert os.path.isdir(os.path.join(os.path.dirname(out), "mosaic_tmp"))
    assert tifffile.imread(out).shape[0] == 2


def test_build_multichannel_mosaic_falls_back_when_the_memmap_fails(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas)
    ro = tmp_path / "readonly"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        out = str(tmp_path / "fb.tif")
        st.build_multichannel_mosaic_from_manifest(mcsv, out, tmp_dir=str(ro))
        assert tifffile.imread(out).shape[0] == 2
    finally:
        os.chmod(ro, 0o700)


def test_build_multichannel_mosaic_infers_the_channel_count(tmp_path, canvas):
    st, mcsv, _ = _manifest(tmp_path, canvas, channels=3)
    out = str(tmp_path / "three.tif")
    st.build_multichannel_mosaic_from_manifest(mcsv, out, tmp_dir=str(tmp_path / "t"))
    assert tifffile.imread(out).shape[0] == 3
