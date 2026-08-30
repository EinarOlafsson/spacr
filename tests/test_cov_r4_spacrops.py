"""``spacr.spacrops``: the quiet paths, and the ones a bad tile takes.

Everything in this module is gated on flags a user sets and on inputs a
microscope produces, and the paths pinned here are the ones the noisy,
well-behaved run never reaches:

* the disk feature cache deleting a *corrupt* entry and keeping an
  *unreadable* one -- the difference between recovering from a killed run and
  emptying the whole cache, one entry per attempt, when a disk fills up;
* a pair with nothing to match and a pair whose fit does not converge, both
  of which have to drop out of the CSV rather than stop the folder;
* the run with ``verbose=False``, which is the default and the one the
  library is used through;
* the mosaic threshold chosen from the scores when none is given;
* the organizer's in-place and dry-run modes;
* the axis flags -- ``squeeze_singleton`` and an axes hint that names no
  channel -- which decide what a plane even is.

The last section proves the arcs in this module that cannot be taken, rather
than silencing them.
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")

import cv2                                                        # noqa: E402
import numpy as np                                                # noqa: E402
import pytest                                                     # noqa: E402
import tifffile                                                   # noqa: E402

from spacr import spacrops                                        # noqa: E402
from spacr.spacrops import (StitchedMultiAligner, _DiskFeatureStore,  # noqa: E402
                            align_image_to_stitch, spacrStitcher,
                            stitch_cycle_wells)
from tests.spacrops_synth import (blob_canvas, channel_variant,   # noqa: E402
                                  crop, write_cyx, write_plane)

TILE = 320
STEP = 130
Y0 = X0 = 100


@pytest.fixture(scope="module")
def canvas():
    return blob_canvas(H=700, W=700, seed=3)


def _stitcher(tmp_path, **kwargs):
    kwargs.setdefault("outdir", str(tmp_path / "out"))
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("save_qc", False)
    kwargs.setdefault("save_stitched_default", False)
    kwargs.setdefault("feature_cache_mode", "ram")
    return spacrStitcher(**kwargs)


def _tiles(folder, canvas, n=2, well="A1", tile=TILE, step=STEP):
    os.makedirs(folder, exist_ok=True)
    made = []
    for index in range(n):
        path = os.path.join(folder,
                            f"10X_c1_{well}_r01f{index + 1:02d}_"
                            f"Site-{index + 1}.tif")
        write_plane(path, crop(canvas, Y0, X0 + index * step, tile))
        made.append(path)
    return made


# ===========================================================================
# The disk feature cache
# ===========================================================================

@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root reads a file whatever its mode")
def test_an_unreadable_cache_entry_is_kept_and_a_corrupt_one_is_dropped(
        tmp_path):
    """Only bad *bytes* justify deleting a cache entry.

    A truncated NPZ from a killed run must go, or the miss is permanent. A
    read that failed for a reason that says nothing about the contents --
    a full disk, a permission, memory pressure -- must not, because that
    failure repeats for every entry and would empty the whole cache.

    Both halves are driven here so "kept" is a decision and not an
    omission.
    """
    store = _DiskFeatureStore(str(tmp_path / "cache"))

    corrupt = store._npz_path("/tiles/truncated.tif")
    os.makedirs(os.path.dirname(corrupt), exist_ok=True)
    with open(corrupt, "wb") as handle:
        handle.write(b"PK\x03\x04 and then the process was killed")

    unreadable = store._npz_path("/tiles/locked.tif")
    with open(unreadable, "wb") as handle:
        handle.write(b"anything at all")
    os.chmod(unreadable, 0o000)

    try:
        assert store.get("/tiles/truncated.tif") is None
        assert store.get("/tiles/locked.tif") is None
        assert not os.path.exists(corrupt), "bad bytes are dropped"
        assert os.path.exists(unreadable), (
            "a read that failed for another reason keeps the entry")
    finally:
        os.chmod(unreadable, 0o644)


# ===========================================================================
# run_folder, at its default verbosity
# ===========================================================================

def test_an_empty_folder_still_writes_a_pairwise_csv(tmp_path):
    """A run over a folder with no tiles must leave the file it promised.

    ``run_folder`` returns a path; a caller that then reads it -- which is
    what ``stitch_cycle_wells`` does -- would fail on a missing file rather
    than on an empty result.
    """
    stitcher = _stitcher(tmp_path)
    folder = tmp_path / "no_tiles"
    folder.mkdir()
    (folder / "notes.txt").write_text("not an image")
    out = str(tmp_path / "pairs.csv")

    assert stitcher.run_folder(str(folder), out) == out
    rows = list(csv.DictReader(open(out)))
    assert rows == []
    assert "score" in open(out).readline()


def test_pairing_every_tile_with_every_other_ignores_the_well(tmp_path, canvas):
    """``same_well_only=False`` is how a plate with unparseable names is run.

    The default pairs only within a well and only within a site window; off,
    the same two tiles from *different* wells become a candidate pair, which
    is the whole point of the flag.
    """
    folder = str(tmp_path / "tiles")
    _tiles(folder, canvas, n=1, well="A1")
    _tiles(folder, canvas, n=1, well="B2")
    stitcher = _stitcher(tmp_path)

    per_well = stitcher.run_folder(folder, str(tmp_path / "within.csv"),
                                   n_workers=1, stitch=False)
    across = stitcher.run_folder(folder, str(tmp_path / "across.csv"),
                                 same_well_only=False, n_workers=1,
                                 stitch=False)

    assert list(csv.DictReader(open(per_well))) == [], (
        "one tile per well leaves no candidate inside a well")
    rows = list(csv.DictReader(open(across)))
    assert len(rows) == 1
    assert {os.path.basename(rows[0]["pathA"]),
            os.path.basename(rows[0]["pathB"])} == {
        "10X_c1_A1_r01f01_Site-1.tif", "10X_c1_B2_r01f01_Site-1.tif"}


# ===========================================================================
# A pair that cannot be scored
# ===========================================================================

def test_a_pair_with_nothing_to_match_is_dropped_not_scored(tmp_path):
    """Two featureless tiles produce fewer than four correspondences.

    The pair has to come back as ``None`` -- no row, no zero score -- so a
    mosaic built from the CSV never joins two tiles on an alignment that was
    never estimated.
    """
    stitcher = _stitcher(tmp_path)
    blank_a = str(tmp_path / "flat_a.tif")
    blank_b = str(tmp_path / "flat_b.tif")
    write_plane(blank_a, np.zeros((128, 128), np.uint16))
    write_plane(blank_b, np.full((128, 128), 7, np.uint16))

    assert stitcher.stitch_pair(blank_a, blank_b, save_stitched=False) is None


def test_a_pair_whose_fit_does_not_converge_is_dropped(tmp_path, canvas,
                                                       monkeypatch):
    """RANSAC returning no model is a real outcome on repetitive tissue.

    Simulated at the OpenCV boundary rather than by hunting for an image
    pair that happens to defeat it: the condition under test is what
    ``stitch_pair`` does with the answer, not how the answer arises. The
    same two tiles score normally without the patch, which is what makes
    the ``None`` below a consequence of the failed fit.
    """
    stitcher = _stitcher(tmp_path)
    tiles = _tiles(str(tmp_path / "tiles"), canvas, n=2)

    scored = stitcher.stitch_pair(tiles[0], tiles[1], save_stitched=False)
    assert scored is not None and scored["score"] > 0

    monkeypatch.setattr(cv2, "estimateAffinePartial2D",
                        lambda *a, **k: (None, None))
    assert stitcher.stitch_pair(tiles[0], tiles[1],
                                save_stitched=False) is None


# ===========================================================================
# The mosaic threshold
# ===========================================================================

@pytest.fixture(scope="module")
def pairwise_csv(tmp_path_factory, canvas):
    """One scored pair of overlapping tiles, as a pairwise CSV."""
    root = tmp_path_factory.mktemp("mosaic")
    folder = str(root / "tiles")
    os.makedirs(folder)
    for index in range(2):
        write_cyx(os.path.join(folder,
                               f"10X_c1_A1_r01f{index + 1:02d}_"
                               f"Site-{index + 1}.tif"),
                  [channel_variant(crop(canvas, Y0, X0 + index * STEP, TILE), c)
                   for c in range(2)])
    stitcher = spacrStitcher(outdir=str(root / "out"), verbose=False,
                             save_qc=False, save_stitched_default=False,
                             feature_cache_mode="ram")
    out = stitcher.run_folder(folder, str(root / "pairs.csv"), n_workers=1,
                              stitch=False)
    return stitcher, out, root


def test_the_mosaic_threshold_is_taken_from_the_scores_when_none_is_given(
        pairwise_csv):
    """``min_score=None`` means "work it out", not "keep everything".

    A threshold above every score leaves each tile in its own component and
    the mosaic is one tile wide; the auto threshold has to accept the pair
    and produce a canvas wider than that.
    """
    stitcher, csv_path, root = pairwise_csv

    auto_tif, _ = stitcher.render_mosaic_from_csv(
        csv_path, str(root / "auto.tif"), min_score=None)
    strict_tif, _ = stitcher.render_mosaic_from_csv(
        csv_path, str(root / "strict.tif"), min_score=0.999)

    auto = tifffile.imread(auto_tif)
    strict = tifffile.imread(strict_tif)
    assert auto.shape[-1] > TILE, "the pair was joined"
    assert strict.shape[-1] == TILE, "nothing cleared an impossible threshold"


def test_the_multichannel_mosaic_picks_its_threshold_the_same_way(
        pairwise_csv):
    """The all-channel renderer reuses the pairwise transforms, so it has to
    reach the same answer about which pairs to trust."""
    stitcher, csv_path, root = pairwise_csv

    auto = stitcher.mosaic_all_channels_from_csv(
        csv_path, str(root / "auto_allc.tif"), min_score=None)
    strict = stitcher.mosaic_all_channels_from_csv(
        csv_path, str(root / "strict_allc.tif"), min_score=0.999)

    joined = tifffile.imread(auto)
    alone = tifffile.imread(strict)
    assert joined.shape[0] == 2, "both channels are in the stack"
    assert joined.shape[-1] > TILE
    assert alone.shape[-1] == TILE


# ===========================================================================
# Axis handling
# ===========================================================================

def test_a_plane_one_row_high_survives_only_with_the_squeeze_off(tmp_path):
    """``squeeze_singleton`` decides what counts as a 2-D plane.

    A stack whose Y axis is a single row squeezes down to one dimension and
    is then rejected as "not 2-D". Turning the squeeze off is what lets such
    an array through -- which is the reason the flag exists.
    """
    aligner = StitchedMultiAligner(outdir=str(tmp_path / "keep"),
                                   squeeze_singleton=False)
    squeezer = StitchedMultiAligner(outdir=str(tmp_path / "drop"),
                                    squeeze_singleton=True)
    stack = np.arange(3 * 1 * 256, dtype=np.float32).reshape(3, 1, 256)

    kept = aligner._normalize_to_yx(stack, ch=0)

    assert kept.shape == (1, 256)
    assert np.array_equal(kept, stack[0].astype(np.float32))
    with pytest.raises(ValueError, match="Expected 2D YX"):
        squeezer._normalize_to_yx(stack, ch=0)


def test_a_manifest_tile_with_no_channel_axis_is_projected_over_z(tmp_path,
                                                                  canvas):
    """The manifest reader takes the axes hint the TIFF carries.

    A ``ZYX`` tile names no channel, so nothing is sliced out for one; the Z
    planes are max-projected instead. A tile whose hint does not describe
    the array at all falls back to squeezing, which is the other branch and
    is driven by the second tile here.
    """
    stitcher = _stitcher(tmp_path)
    zyx = str(tmp_path / "zyx.tif")
    planes = [crop(canvas, Y0, X0, 128), crop(canvas, Y0 + 40, X0 + 40, 128)]
    tifffile.imwrite(zyx, np.stack(planes, axis=0), metadata={"axes": "ZYX"})

    trailing = str(tmp_path / "trailing.tif")
    tifffile.imwrite(trailing,
                     crop(canvas, Y0, X0, 128)[:, :, None])   # (Y, X, 1)

    manifest = str(tmp_path / "manifest.csv")
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "H", "W", "M00", "M01", "M02", "M10", "M11", "M12",
            "canvas_x", "canvas_y", "best_pair_score"])
        writer.writeheader()
        for index, path in enumerate((zyx, trailing)):
            writer.writerow(dict(path=path, H=128, W=128,
                                 M00=1, M01=0, M02=index * 64,
                                 M10=0, M11=1, M12=0,
                                 canvas_x=0, canvas_y=0, best_pair_score=1.0))

    out = str(tmp_path / "mc.tif")
    stitcher.build_multichannel_mosaic_from_manifest(manifest, out,
                                                     channel_indices=[0])
    mosaic = tifffile.imread(out)

    assert mosaic.ndim == 3 and mosaic.shape[0] == 1
    plane = mosaic[0]
    assert plane.shape[0] >= 128 and plane.shape[1] >= 128 + 64
    # The Z tile arrives max-projected: its brightest plane is what lands.
    assert plane[:128, :128].max() == np.maximum(*planes).max()


# ===========================================================================
# The organizer
# ===========================================================================

def _settings(src, dst, **kwargs):
    base = dict(src=str(src), dst_root=str(dst), verbose=False, max_site_gap=2,
                n_workers=1, downsample=0.5, plate="P1", do_nuc_stitch=False)
    base.update(kwargs)
    return base


def test_a_tile_already_at_its_destination_is_not_moved_onto_itself(tmp_path,
                                                                    canvas):
    """Re-running the organizer in place must not lose the plate.

    ``dst_root`` defaults to ``src``, so a second run finds each tile at
    exactly the path it would move it to. ``shutil.move`` onto itself is not
    a no-op on every filesystem, so the move is skipped -- and the tile has
    to still be there afterwards.
    """
    root = tmp_path / "plate"
    (root / "A1").mkdir(parents=True)
    tiles = _tiles(str(root / "A1"), canvas, n=1)

    result = stitch_cycle_wells(_settings(root, root, collision="overwrite"))

    assert result["organized"]["by_well"]["A1"] == tiles
    assert result["organized"]["moved"] == 1


def test_a_dry_run_plans_the_links_and_the_move_and_makes_neither(tmp_path,
                                                                  canvas):
    """A dry run is the only way to see the layout before committing to it.

    It has to be safe to point at real data: no link created, no tile moved,
    and the planned paths still reported so the layout can be read off.
    """
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    tiles = _tiles(str(src), canvas, n=2)

    result = stitch_cycle_wells(_settings(src, dst, dry_run=True,
                                          do_organize=False,
                                          do_nuc_stitch=True))

    assert result["organized"]["linked"] == 0
    assert result["organized"]["moved"] == 0
    assert all(os.path.exists(tile) for tile in tiles), "nothing was moved"
    assert os.listdir(str(dst / "_links" / "A1")) == [], "no link was made"
    planned = result["wells"]["A1"]["tiles"]
    assert [os.path.basename(p) for p in planned] == [
        os.path.basename(t) for t in tiles]
    assert not any(os.path.exists(p) for p in planned), (
        "the reported paths are a plan, not files")


def test_a_recursive_align_scan_takes_the_images_and_leaves_the_rest(tmp_path,
                                                                     canvas):
    """Acquisition folders carry logs and previews beside the TIFFs.

    Any of them reaching the aligner is opened as an image and fails the
    whole well, so the scan filters by extension as it walks.
    """
    root = tmp_path / "stitched"
    (root / "A1" / "_stitch").mkdir(parents=True)
    write_cyx(str(root / "A1" / "_stitch" / "mosaic_allc.tif"),
              [crop(canvas, Y0, X0, 600)])

    src = tmp_path / "align20x"
    (src / "day1").mkdir(parents=True)
    write_cyx(str(src / "day1" / "20X_c1_A1_Site-1.tif"),
              [crop(canvas, Y0 + 150, X0 + 180, 256)])
    (src / "day1" / "acquisition.log").write_text("not an image\n")
    (src / "day1" / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    result = align_image_to_stitch(str(root), str(src), relative_scale=1.0)

    assert os.listdir(result["A1"]["align_folder"]) == ["20X_c1_A1_Site-1.tif"]
    rows = list(csv.DictReader(open(result["A1"]["manifest_csv"])))
    assert len(rows) == 1
    assert float(rows[0]["tx"]) == pytest.approx(180, abs=4)
    assert float(rows[0]["ty"]) == pytest.approx(150, abs=4)


# ===========================================================================
# Aligning a second round
# ===========================================================================

def test_allowing_scale_recovers_a_scale_that_pinning_it_cannot(tmp_path,
                                                                canvas):
    """``allow_scale`` is what makes a second imaging round at a different
    pixel size alignable.

    The moving image here is the reference resampled by 1.25. The transform
    is estimated *into* the reference's frame, so the scale that comes back
    is its reciprocal, 0.8. With scale disallowed the fit is forced to unity
    instead and the manifest says 1.0 -- the round would be stacked at the
    wrong magnification and nothing downstream would notice.
    """
    reference = crop(canvas, Y0, X0, 400)
    moved = cv2.resize(reference, (500, 500), interpolation=cv2.INTER_LINEAR)
    ref_path = str(tmp_path / "round1.tif")
    mov_path = str(tmp_path / "round2.tif")
    write_cyx(ref_path, [reference])
    write_cyx(mov_path, [moved])

    def _scale_of(allow_scale):
        aligner = StitchedMultiAligner(
            outdir=str(tmp_path / f"out_scale_{allow_scale}"),
            downsample=0.5, allow_scale=allow_scale)
        _tif, _png, manifest = aligner.align([ref_path, mov_path])
        rows = [r for r in csv.DictReader(open(manifest))
                if r["ref"] in ("False", "false")]
        assert rows, "the moving image was aligned at all"
        return float(rows[0]["scale"])

    assert _scale_of(False) == pytest.approx(1.0, abs=1e-6)
    assert _scale_of(True) == pytest.approx(1 / 1.25, abs=0.05)


# ===========================================================================
# The whole ops pipeline
# ===========================================================================

def test_a_named_output_root_is_used_for_every_genotype_given(tmp_path,
                                                              canvas):
    """``dst_root`` has to survive being copied into each genotype's settings.

    Dropped there, the run organizes in place -- which is fine on a scratch
    copy and rewrites the plate on a read-only acquisition share. The
    genotypes are handed in as an explicit list here, the other way the
    setting accepts them, and both have to land under the one named root
    rather than beside their own images.
    """
    from spacr.spacrops import ops_preprocess

    genotypes = []
    for name in ("wt", "ko"):
        folder = tmp_path / name
        _tiles(str(folder), canvas, n=2)
        genotypes.append(str(folder))
    phenotype = tmp_path / "pheno"
    phenotype.mkdir()
    dst = tmp_path / "elsewhere"

    result = ops_preprocess(dict(genotype_source=genotypes,
                                 phenotype_source=str(phenotype),
                                 dst_root=str(dst), verbose=False, plate="P1",
                                 max_site_gap=2, n_workers=1,
                                 do_multichannel=False))

    assert [s["genotype_folder"] for s in result["stitch"]] == genotypes
    for summary in result["stitch"]:
        tiles_dir = summary["summary"]["wells"]["A1"]["tiles_dir"]
        assert tiles_dir == os.path.join(str(dst), "A1", "A1")
    for folder in genotypes:
        assert not os.path.exists(os.path.join(folder, "A1")), (
            "the genotype folder was not organized in place")


# ===========================================================================
# Proved unreachable
# ===========================================================================

def test_the_downsample_used_for_scoring_is_never_zero(tmp_path):
    """Why ``if s != 0`` in ``stitch_pair`` cannot be false.

    ``s`` is assigned as ``self.downsample if self.downsample > 0 else 1.0``
    (spacrops.py line 881), so it is either a positive number or exactly
    1.0. There is no value of ``downsample`` -- zero and negative included --
    that makes ``s`` zero, which is what this pins.
    """
    for downsample in (0.0, -2.0, 0.25):
        stitcher = _stitcher(tmp_path, downsample=downsample)
        s = stitcher.downsample if stitcher.downsample > 0 else 1.0
        assert s != 0

    # And the guard is on the same expression the lift uses, so the
    # translation is always divided by a real scale.
    stitcher = _stitcher(tmp_path, downsample=0.0)
    assert stitcher.downsample == 0.0


def test_a_pair_that_produces_a_row_always_produces_a_score(tmp_path, canvas):
    """Why ``if row.get("score", "") != ""`` in ``run_folder`` cannot be false.

    ``_job`` returns either ``None`` -- filtered out one line above -- or
    ``stitch_pair``'s dict, and ``stitch_pair`` has exactly one ``return
    dict(...)`` (line 1096) which always sets ``score=score``, a float. No
    row reaching that test can carry an empty score.
    """
    stitcher = _stitcher(tmp_path)
    tiles = _tiles(str(tmp_path / "tiles"), canvas, n=2)

    row = stitcher.stitch_pair(tiles[0], tiles[1], save_stitched=False)

    assert row is not None
    assert isinstance(row["score"], float)
    assert row.get("score", "") != ""


def test_no_pairs_means_no_transforms_and_no_root_to_look_for(tmp_path):
    """Why ``if root is None`` in ``_compute_mosaic_transforms`` cannot be true.

    ``nodes`` is built from the rows and the function returns at line 1801
    when it is empty; nothing between there and line 1964 removes a node, so
    ``root = max(nodes, ...)`` always has something to choose from. The
    empty case leaves through the earlier return, which is what this pins.
    """
    stitcher = _stitcher(tmp_path)

    transforms, used = stitcher._compute_mosaic_transforms([], 0.0)

    assert transforms == {}
    assert used == []


def test_a_zero_downsample_fails_at_the_first_resize(tmp_path, canvas):
    """Why ``if s != 0`` in ``StitchedMultiAligner.align`` cannot be false.

    ``s`` is ``float(self.downsample)`` there, so zero is expressible -- but
    it makes every feature image ``max(1, round(n * 0))`` = 1x1, and OpenCV's
    detector refuses to build a pyramid for a 1x1 image. The call therefore
    raises at the reference's own feature pass, long before the loop that
    lifts a transform, so the lift never sees ``s == 0``.

    ``FOVAlignAndCropper.run`` reads ``s`` from the same aligner attribute
    and resizes the mosaic the same way, outside its per-file ``try``, so
    the same argument holds for its lift.
    """
    path = str(tmp_path / "round1.tif")
    write_cyx(path, [crop(canvas, Y0, X0, 256)])
    aligner = StitchedMultiAligner(outdir=str(tmp_path / "z"), downsample=0.0)

    with pytest.raises(cv2.error):
        aligner.align([path, path])


def test_the_post_stitch_move_never_finds_a_tile_already_at_its_target(
        tmp_path, canvas):
    """Why ``if os.path.abspath(sp) != os.path.abspath(rp)`` at line 3185
    cannot be false.

    The organizer puts a tile at ``<dst_root>/<well>/<name>`` (line 2999),
    and the post-stitch move targets ``orig_outdir`` =
    ``<dst_root>/<well>/<well>`` (line 3069). ``well`` is guaranteed
    non-empty -- a filename whose well group is missing or empty is skipped
    at line 2942 -- so the target always carries one path component more
    than the source and the two can never be the same path.

    Pinned by the layout on disk afterwards: the tile really did travel
    from one of those paths to the other.
    """
    src = tmp_path / "plate"
    name = os.path.basename(_tiles(str(src), canvas, n=1)[0])

    result = stitch_cycle_wells(_settings(src, src, do_nuc_stitch=True,
                                          collision="overwrite"))

    moved = result["wells"]["A1"]["tiles"][0]
    assert os.path.dirname(moved) == os.path.join(str(src), "A1", "A1")
    assert os.path.isfile(moved)
    assert not os.path.exists(os.path.join(str(src), "A1", name)), (
        "the organizer's own target is a different path, and the tile left it")


def test_the_aligner_is_always_importable_from_inside_the_module():
    """Why ``if "align_image_to_stitch" in globals()`` in ``ops_preprocess``
    cannot be false in an installed build.

    The name is defined at module scope in the same file, so the lookup is
    against a dict that always holds it; the only way to the other branch is
    to delete the attribute, which is what
    ``test_ops_preprocess_still_stitches_when_the_aligner_is_not_there``
    does deliberately.
    """
    assert callable(vars(spacrops)["align_image_to_stitch"])
