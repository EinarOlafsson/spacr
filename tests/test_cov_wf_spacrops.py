"""spacr.spacrops -- the quiet paths a stitching run takes when nobody is watching.

An ops run is unattended: it organizes a plate, scores thousands of tile pairs
and renders mosaics with ``verbose`` off, and the only evidence anyone sees is
what it left on disk.  Every path pinned here produces no message at all -- a
feature cache that will not load, an empty folder, a pair with nothing to
match, a dry run, an automatic threshold -- so when one of them goes wrong the
first symptom is a missing tile, or a deleted frame, weeks later.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import tifffile

from spacr import spacrops
from spacr.spacrops import (
    StitchedMultiAligner,
    _DiskFeatureStore,
    align_image_to_stitch,
    ops_preprocess,
    spacrStitcher,
    stitch_cycle_wells,
)
from tests.spacrops_synth import (
    blob_canvas,
    channel_variant,
    crop,
    row_of_tiles,
    tile_name,
    write_cyx,
    write_plane,
)

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
    kw.setdefault("verbose", False)
    return spacrStitcher(**kw)


def _pairs_csv(tmp_path, canvas, channels=1, n=2):
    """A real pairwise CSV for ``n`` overlapping tiles, plus the stitcher."""
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=n, tile=TILE, step=STEP, y0=Y0, x0=X0,
                 channels=channels)
    st = _st(tmp_path)
    return st, st.run_folder(d, str(tmp_path / "pairs.csv"), max_site_gap=2,
                             stitch=False)


def test_an_unreadable_cache_entry_is_only_deleted_when_the_bytes_are_bad(tmp_path):
    """A read error is not proof the cached features are corrupt.

    The cache is what makes re-running a plate cheap.  Dropping an entry is
    right when its bytes are damaged -- a run killed mid-write leaves a
    truncated NPZ that would poison the cache forever -- and wrong for a
    transient IO error, which on a failing disk would empty the whole cache.
    """
    store = _DiskFeatureStore(str(tmp_path / "cache"))

    # (a) bad bytes: a truncated / garbage NPZ must be dropped
    bad_src = str(tmp_path / "bad_tile.tif")
    bad_npz = store._npz_path(bad_src)
    with open(bad_npz, "wb") as fh:
        fh.write(b"PK\x03\x04 not really a zip")
    assert store.get(bad_src) is None
    assert not os.path.exists(bad_npz), "a corrupt entry must not survive"

    # (b) unreadable but not corrupt: a directory where the NPZ should be makes
    # np.load raise IsADirectoryError, which says nothing about any bytes.
    keep_src = str(tmp_path / "keep_tile.tif")
    keep_npz = store._npz_path(keep_src)
    os.makedirs(keep_npz)
    assert store.get(keep_src) is None
    assert os.path.isdir(keep_npz), "a readable-later entry was destroyed"


def test_a_pair_the_stitcher_cannot_model_is_dropped_without_a_word(tmp_path, canvas, capsys):
    """Both give-up paths stay silent unless the run asked to be told.

    run_folder scores every pair in a well; on a sparse plate most pairs do not
    overlap and some that do will not fit a model, so an ungated message would
    be thousands of lines hiding the real failures.
    """
    blank = tmp_path / "blank"
    blank.mkdir()
    ba, bb = str(blank / tile_name(site=1)), str(blank / tile_name(site=2))
    tifffile.imwrite(ba, np.zeros((128, 128), np.uint16))
    tifffile.imwrite(bb, np.zeros((128, 128), np.uint16))

    good = tmp_path / "pair"
    good.mkdir()
    pa, pb = str(good / tile_name(site=1)), str(good / tile_name(site=2))
    write_plane(pa, crop(canvas, 250, 250, TILE))
    write_plane(pb, crop(canvas, 250, 250 + STEP, TILE))

    quiet = _st(tmp_path)
    quiet._affine_from_pts = lambda *a, **k: (None, None, 0.0)   # RANSAC gives up
    assert quiet.stitch_pair(ba, bb) is None          # nothing to match
    assert quiet.stitch_pair(pa, pb) is None          # no model to fit
    assert capsys.readouterr().out == ""

    loud = _st(tmp_path, verbose=True)
    loud._affine_from_pts = lambda *a, **k: (None, None, 0.0)
    assert loud.stitch_pair(ba, bb) is None
    assert loud.stitch_pair(pa, pb) is None
    said = capsys.readouterr().out
    assert "<4 matches" in said and "RANSAC failed" in said


def test_a_folder_with_no_images_still_gets_a_pairwise_csv(tmp_path, canvas):
    """An empty scan leaves a readable CSV, not nothing at all.

    The mosaic, the alignment and the per-well summary all open this file by
    path.  A mistyped folder that wrote nothing would fail much later with a
    FileNotFoundError naming a path nobody recognises.
    """
    empty = tmp_path / "nothing"
    empty.mkdir()
    st = _st(tmp_path)
    out = str(tmp_path / "empty.csv")
    assert st.run_folder(str(empty), out, max_site_gap=2, stitch=False) == out
    with open(out) as fh:
        rdr = csv.DictReader(fh)
        assert list(rdr) == [] and "score" in rdr.fieldnames

    # the same stitcher over a folder that does hold tiles writes a data row
    tiles = str(tmp_path / "tiles")
    row_of_tiles(tiles, canvas=canvas, n=2, tile=TILE, step=STEP, y0=Y0, x0=X0)
    full = str(tmp_path / "full.csv")
    st.run_folder(tiles, full, max_site_gap=2, stitch=False)
    assert len(list(csv.DictReader(open(full)))) == 1


def test_pairing_across_wells_keeps_a_scoreless_row_out_of_the_threshold(tmp_path, canvas, capsys):
    """Every file pairs with every other, and an unscored pair is not a zero.

    ``same_well_only=False`` is how a plate whose filenames carry no usable
    well still gets stitched.  A pair that comes back without a score belongs
    in the CSV, but counting it would drag the automatic threshold down and
    let genuinely bad pairs into the mosaic.
    """
    d = str(tmp_path / "tiles")
    row_of_tiles(d, canvas=canvas, n=2, tile=TILE, step=STEP, y0=Y0, x0=X0,
                 well="A1")
    far, _ = row_of_tiles(d, canvas=canvas, n=1, tile=TILE, step=STEP,
                          y0=Y0, x0=X0 + 2 * STEP, well="B2", first_site=3)
    far_name = os.path.basename(far[0])
    st = _st(tmp_path)
    real = st.stitch_pair

    def scoreless_when_the_far_tile_is_involved(A, B, **kw):
        row = real(A, B, **kw)
        if row is not None and far_name in (os.path.basename(A), os.path.basename(B)):
            row["score"] = ""
        return row

    st.stitch_pair = scoreless_when_the_far_tile_is_involved
    out = str(tmp_path / "all_pairs.csv")
    st.run_folder(d, out, max_site_gap=8, same_well_only=False, stitch=False)

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3                        # 3 files, all against all
    blank = [r for r in rows if r["score"] == ""]
    scored = [r for r in rows if r["score"] != ""]
    assert len(blank) == 2 and len(scored) == 1
    assert float(scored[0]["score"]) > 0.0
    capsys.readouterr()

    # the same run with verbose on says how many scores the threshold saw,
    # and the two blank rows are not among them.
    st.verbose = True
    st.run_folder(d, str(tmp_path / "again.csv"), max_site_gap=8,
                  same_well_only=False, stitch=False)
    assert "auto threshold from 1 scores" in capsys.readouterr().out


def _manifest_for(tmp_path, tile_path, H, W, name="manifest.csv"):
    """A one-tile mosaic manifest placing ``tile_path`` at the origin."""
    mcsv = str(tmp_path / name)
    with open(mcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "H", "W", "M00", "M01", "M02",
                                           "M10", "M11", "M12", "canvas_x",
                                           "canvas_y", "best_pair_score"])
        w.writeheader()
        w.writerow({"path": tile_path, "H": H, "W": W,
                    "M00": 1, "M01": 0, "M02": 0,
                    "M10": 0, "M11": 1, "M12": 0,
                    "canvas_x": 0, "canvas_y": 0, "best_pair_score": 1.0})
    return mcsv


def test_the_axis_labels_on_a_tile_decide_which_plane_reaches_the_mosaic(tmp_path):
    """A tile is projected, sliced or refused according to how it is labelled.

    Confocal wells arrive as ZYX with no channel axis, plain exports with no
    axis metadata at all.  Indexing Z as if it were C loses every object that
    was sharp deeper in the stack; guessing at ten unlabelled pages puts
    arbitrary pixels in the mosaic and returns it as if it were right.
    """
    st = _st(tmp_path)

    # (a) ZYX: no C to slice, so Z is max-projected
    planes = [np.full((48, 64), 100, np.uint16) for _ in range(3)]
    planes[0][10:20, 10:20] = 5000      # only in the first plane
    planes[2][30:40, 30:40] = 4000      # only in the last
    zstack = str(tmp_path / "zstack.tif")
    tifffile.imwrite(zstack, np.stack(planes), photometric="minisblack",
                     metadata={"axes": "ZYX"})
    out = str(tmp_path / "z_mosaic.tif")
    assert st.build_multichannel_mosaic_from_manifest(
        _manifest_for(tmp_path, zstack, 48, 64), out,
        tmp_dir=str(tmp_path / "tmp")) == out
    img = np.squeeze(tifffile.imread(out))
    assert img.shape == (48, 64)
    assert np.array_equal(img, np.max(np.stack(planes), axis=0))
    assert img[15, 15] == 5000 and img[35, 35] == 4000     # both planes survive

    # (b) two unlabelled pages read as channels (page 0 is channel 0);
    # ten pages are not channels at all, and are refused rather than guessed
    two = str(tmp_path / "two.tif")
    tifffile.imwrite(two, np.stack([np.full((48, 64), 7, np.uint16),
                                    np.full((48, 64), 9, np.uint16)]))
    written = st.build_multichannel_mosaic_from_manifest(
        _manifest_for(tmp_path, two, 48, 64, name="two.csv"),
        str(tmp_path / "two_mosaic.tif"), tmp_dir=str(tmp_path / "tmp"))
    assert np.squeeze(tifffile.imread(written))[0, 0] == 7

    deep = str(tmp_path / "deep.tif")
    tifffile.imwrite(deep, np.zeros((10, 48, 64), np.uint16))
    with pytest.raises(ValueError, match=r"Expected 2D plane from deep\.tif"):
        st.build_multichannel_mosaic_from_manifest(
            _manifest_for(tmp_path, deep, 48, 64, name="deep.csv"),
            str(tmp_path / "x.tif"), tmp_dir=str(tmp_path / "tmp"))


def test_both_mosaic_renderers_pick_their_own_threshold_when_not_verbose(tmp_path, canvas):
    """The auto threshold is a decision, not a log line.

    ``min_score=None`` asks the renderer to find the elbow in the score
    distribution itself, and an unattended run leaves it there.  Both
    renderers carry their own copy of that step, and with the threshold
    mis-set the mosaic comes out one tile wide.
    """
    st, pairs = _pairs_csv(tmp_path, canvas, channels=2)
    gt = canvas[Y0:Y0 + TILE, X0:X0 + TILE + STEP]

    single = str(tmp_path / "mosaic.tif")
    st.render_mosaic_from_csv(pairs, single, min_score=None)
    img = np.squeeze(tifffile.imread(single))
    assert img.shape[1] == pytest.approx(TILE + STEP, abs=3)   # both tiles placed
    h, w = min(img.shape[0], gt.shape[0]), min(img.shape[1], gt.shape[1])
    assert np.corrcoef(img[:h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99

    allc = str(tmp_path / "mosaic_allc.tif")
    st.mosaic_all_channels_from_csv(pairs, allc, min_score=None)
    stack = tifffile.imread(allc)
    assert stack.shape[0] == 2                                  # both channels
    assert stack.shape[2] == pytest.approx(TILE + STEP, abs=3)
    h, w = min(stack.shape[1], gt.shape[0]), min(stack.shape[2], gt.shape[1])
    assert np.corrcoef(stack[0, :h, :w].ravel(), gt[:h, :w].ravel())[0, 1] > 0.99
    assert not np.array_equal(stack[0], stack[1])               # channel 1 differs


def test_a_cycle_aligns_with_scale_free_and_singleton_axes_kept(tmp_path, canvas):
    """Letting the fit find a scale must not cost the translation.

    ``allow_scale=True`` keeps the raw affine RANSAC estimated instead of the
    translation-only re-fit, which is what two cycles imaged at slightly
    different zoom need; ``squeeze_singleton=False`` keeps length-1 axes
    through the read.  Both decide where cycle 2 lands.
    """
    d = tmp_path / "cycles"
    d.mkdir()
    ref = crop(canvas, 200, 200, 448)
    mov = crop(canvas, 200 - 25, 200 + 40, 448)
    p1, p2 = str(d / "cycle1.tif"), str(d / "cycle2.tif")
    write_cyx(p1, [channel_variant(ref, c) for c in range(2)])
    write_cyx(p2, [channel_variant(mov, c) for c in range(2)])

    al = StitchedMultiAligner(outdir=str(tmp_path / "al"), downsample=0.5,
                              allow_scale=True, squeeze_singleton=False)
    out_tif, _, csv_path = al.align([p1, p2])

    rows = list(csv.DictReader(open(csv_path)))
    assert [r["ref"] for r in rows] == ["True", "True", "False", "False"]
    assert float(rows[2]["tx"]) == pytest.approx(40.0, abs=2.0)
    assert float(rows[2]["ty"]) == pytest.approx(-25.0, abs=2.0)
    assert float(rows[2]["scale"]) == pytest.approx(1.0, abs=0.01)

    stack = tifffile.imread(out_tif)
    assert stack.shape == (4, 448, 448)
    inner = (slice(60, 380), slice(60, 380))
    assert np.corrcoef(stack[2][inner].astype(float).ravel(),
                       ref[inner].astype(float).ravel())[0, 1] > 0.99
    assert np.corrcoef(mov[inner].astype(float).ravel(),
                       ref[inner].astype(float).ravel())[0, 1] < 0.6


def _plate_in(dirpath, canvas, n=1, well="A1"):
    os.makedirs(dirpath, exist_ok=True)
    made = []
    for i in range(n):
        p = os.path.join(dirpath, f"10X_c1_{well}_Site-{i + 1}.tif")
        write_plane(p, crop(canvas, Y0, X0 + i * 130, 320))
        made.append(p)
    return made


def _settings(src, dst, **kw):
    base = dict(src=str(src), dst_root=str(dst), verbose=False, max_site_gap=2,
                n_workers=1, downsample=0.5, plate="P1", do_nuc_stitch=False)
    base.update(kw)
    return base


def test_reorganising_a_plate_onto_itself_keeps_the_tiles(tmp_path, canvas):
    """Re-running the organizer in place must not eat the plate.

    ``dst_root`` defaults to ``src``, so a second run over an already organized
    plate finds each tile at exactly the destination it would move it to.
    ``collision='rename'`` deliberately gives that tile a new name, while
    ``collision='overwrite'`` has nothing to overwrite or move when source and
    destination are the same file.
    """
    kept_root = tmp_path / "kept"
    _plate_in(str(kept_root / "A1"), canvas)
    res = stitch_cycle_wells(_settings(kept_root, kept_root, collision="rename"))
    assert res["organized"]["moved"] == 1
    survivor = res["organized"]["by_well"]["A1"][0]
    assert os.path.exists(survivor) and survivor.endswith("_001.tif")

    overwrite_root = tmp_path / "overwrite"
    tile = _plate_in(str(overwrite_root / "A1"), canvas)[0]
    before = tifffile.imread(tile).copy()
    res = stitch_cycle_wells(
        _settings(overwrite_root, overwrite_root, collision="overwrite"))
    assert res["organized"]["moved"] == 0
    assert res["organized"]["by_well"]["A1"] == [tile]
    assert os.path.exists(tile)
    assert np.array_equal(tifffile.imread(tile), before)


def test_a_dry_run_leaves_the_plate_exactly_where_it_found_it(tmp_path, canvas):
    """A dry run has to be safe to point at real data.

    It is the only way to see how a plate would be grouped before committing to
    it, so it may create no link and move no file while still reporting the
    layout it would have produced.
    """
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    tiles = _plate_in(str(src), canvas, n=2)
    res = stitch_cycle_wells(_settings(src, dst, dry_run=True, do_organize=False,
                                       do_nuc_stitch=True))

    assert res["organized"]["linked"] == 0 and res["organized"]["moved"] == 0
    assert all(os.path.exists(t) for t in tiles)
    assert os.listdir(os.path.join(str(dst), "_links", "A1")) == []
    planned = res["organized"]["by_well"]["A1"]
    assert [os.path.basename(p) for p in planned] == [os.path.basename(t) for t in tiles]
    assert not any(os.path.exists(p) for p in planned)
    assert list(csv.DictReader(open(res["wells"]["A1"]["pairwise_csv"]))) == []


def test_a_recursive_align_scan_takes_the_images_and_leaves_the_rest(tmp_path, canvas):
    """A recursive scan walks into folders that hold more than images.

    Acquisition folders carry logs, previews and settings files beside the
    TIFFs.  Any of them reaching the aligner would be opened as an image and
    fail the whole well.
    """
    root = tmp_path / "stitched"
    (root / "A1" / "_stitch").mkdir(parents=True)
    write_cyx(str(root / "A1" / "_stitch" / "mosaic_allc.tif"),
              [crop(canvas, Y0, X0, 700)])

    src = tmp_path / "align20x"
    (src / "day1").mkdir(parents=True)
    write_cyx(str(src / "day1" / "20X_c1_A1_Site-1.tif"),
              [crop(canvas, Y0 + 150, X0 + 180, 256)])
    (src / "day1" / "acquisition.log").write_text("not an image\n")
    (src / "day1" / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    res = align_image_to_stitch(str(root), str(src), relative_scale=1.0)
    assert os.listdir(res["A1"]["align_folder"]) == ["20X_c1_A1_Site-1.tif"]
    rows = list(csv.DictReader(open(res["A1"]["manifest_csv"])))
    assert len(rows) == 1
    assert float(rows[0]["tx"]) == pytest.approx(180, abs=3)
    assert float(rows[0]["ty"]) == pytest.approx(150, abs=3)


def _genotype(tmp_path, canvas, name="geno"):
    d = tmp_path / name
    _plate_in(str(d), canvas, n=2)
    return str(d)


def test_ops_preprocess_writes_the_stitch_under_an_explicit_dst_root(tmp_path, canvas):
    """A named output root takes the plate, not the folder it came from.

    The default is to organize in place, which is fine on a scratch copy and
    wrong on a read-only acquisition share.  ``dst_root`` has to survive being
    copied into the per-genotype settings: dropped there, the run would
    silently rewrite the source.
    """
    geno = _genotype(tmp_path, canvas)
    pheno = tmp_path / "pheno"
    pheno.mkdir()
    dst = tmp_path / "elsewhere"
    res = ops_preprocess(dict(genotype_source=geno, phenotype_source=str(pheno),
                              dst_root=str(dst), verbose=False, plate="P1",
                              max_site_gap=2, n_workers=1, do_multichannel=False))

    assert [s["genotype_folder"] for s in res["stitch"]] == [geno]
    well = res["stitch"][0]["summary"]["wells"]["A1"]
    assert well["tiles_dir"] == os.path.join(str(dst), "A1", "A1")
    assert len(list(csv.DictReader(open(well["pairwise_csv"])))) == 1
    assert sorted(os.listdir(well["tiles_dir"])) == ["10X_c1_A1_Site-1.tif",
                                                     "10X_c1_A1_Site-2.tif"]
    assert not os.path.exists(os.path.join(geno, "A1"))
    assert res["npy_out_root"] == os.path.join(str(pheno), "output")
    assert os.path.isdir(res["npy_out_root"])


def test_ops_preprocess_still_stitches_when_the_aligner_is_not_there(tmp_path, canvas,
                                                                    monkeypatch):
    """Losing the alignment half must not lose the stitching half.

    ``align_image_to_stitch`` is looked up by name at call time so a build
    without it still runs.  The genotype mosaics and pairwise CSVs have to be
    produced and reported anyway: re-running the stitch costs hours.
    """
    geno = _genotype(tmp_path, canvas)
    pheno = tmp_path / "pheno"
    pheno.mkdir()
    monkeypatch.delattr(spacrops, "align_image_to_stitch")

    res = ops_preprocess(dict(genotype_source=geno, phenotype_source=str(pheno),
                              verbose=False, plate="P1", max_site_gap=2,
                              n_workers=1, do_multichannel=False))

    assert res["align"] == []
    well = res["stitch"][0]["summary"]["wells"]["A1"]
    assert well["tiles_dir"] == os.path.join(geno, "A1", "A1")   # in place
    assert len(list(csv.DictReader(open(well["pairwise_csv"])))) == 1
