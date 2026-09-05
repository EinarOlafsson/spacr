"""The OPS stitcher, measured against a plate whose geometry we chose.

WHY THIS EXISTS. ``spacr/spacrops.py`` is four thousand lines that nothing in
the package imports: no registry row, no builder, no CLI name, no settings
panel. Code with no caller has no test and no user, and two defects had been
sitting in it -- a mosaic that could not be switched on, and a flag whose two
spellings meant opposite things.

THE PLATE IS SYNTHETIC ON PURPOSE. Four sites in a 2x2 with a generous
overlap, all cut from one field, so the true offset between neighbours is a
number this file chose rather than one a human eyeballed off a screenshot.
That makes "did it stitch correctly" a comparison instead of an opinion.

A WARNING, because it cost an hour to learn: at the shipped default
``downsample=0.5`` a 256 px tile becomes 128 px and ORB finds nothing -- every
pair returns an inlier ratio near 0.04 and an offset that is pure noise, which
reads exactly like a broken stitcher and is not one. These tests give it big
tiles and no downsampling, which is what real data looks like.
"""
from __future__ import annotations

import csv
import os

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("cv2")
gaussian_filter = pytest.importorskip("scipy.ndimage").gaussian_filter

#: Tile size and the step between neighbours. The difference is the overlap,
#: and the step IS the ground truth every assertion below compares against.
TILE = 640
STEP = 420

#: Where each site sits in the 2x2, as (row, column) of ``STEP``.
SITES = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}


@pytest.fixture(scope="module")
def stitched_plate(tmp_path_factory):
    """Build a plate with known geometry, stitch it, and return the outputs.

    Module-scoped because the stitch is the expensive part and every test
    below asks a different question of the same answer.

    :param tmp_path_factory: pytest's per-module temporary directory maker.
    :returns: ``(settings_result, results_dir)``.
    """
    from spacr import spacrops

    root = tmp_path_factory.mktemp("ops_plate")
    src = root / "raw"
    src.mkdir()

    # Blobs, not white noise: ORB needs corners, and noise at this scale has
    # none that survive downsampling or repeat across two tiles.
    rng = np.random.default_rng(11)
    side = TILE + STEP
    field = np.zeros((side, side), np.float32)
    ys = rng.integers(0, side, 2600)
    xs = rng.integers(0, side, 2600)
    field[ys, xs] = rng.uniform(3000, 20000, 2600)
    field = gaussian_filter(field, 2.0)
    field = np.clip(field + rng.normal(0, 30, field.shape), 0, 65535)

    for site, (row, col) in SITES.items():
        tile = field[row * STEP:row * STEP + TILE,
                     col * STEP:col * STEP + TILE]
        for channel in (1, 2):
            # Channel 2 is a different PICTURE of the same field, not a copy:
            # identical bytes let a content-hashed feature cache collapse the
            # two, which would make a one-channel mosaic look like a bug when
            # it was deduplication.
            image = tile if channel == 1 else tile * 0.45 + 800.0
            tifffile.imwrite(
                str(src / f"10X_c{channel}_A1_Site-{site}.tif"),
                image.astype(np.uint16),
            )

    result = spacrops.stitch_cycle_wells({
        "src": str(src),
        "dst_root": str(root / "out"),
        "organize": True,
        "verbose": False,
        "downsample": 1.0,
        "nfeatures": 20000,
    })
    return result, root / "out"


def _pairs_csv(out_dir):
    """The one pairs CSV the stitch wrote, wherever it put it."""
    for dirpath, _dirs, files in os.walk(out_dir):
        for name in files:
            if name.endswith("_pairs.csv"):
                return os.path.join(dirpath, name)
    raise AssertionError(f"no pairs CSV under {out_dir}")


def test_every_pair_offset_matches_the_geometry_we_built(stitched_plate):
    """The recovered offsets must be the ones this file laid out.

    Measured when this was written: all twelve rows within 0.3 px of truth.
    The tolerance is 15 px, which is loose enough to survive a different
    OpenCV build and tight enough that a stitcher returning noise -- the
    failure mode that actually happens -- cannot pass.
    """
    _result, out_dir = stitched_plate
    rows = list(csv.DictReader(open(_pairs_csv(out_dir), encoding="utf-8")))
    assert rows, "the stitch scored no pairs at all"

    wrong = []
    for row in rows:
        a, b = int(row["siteA"]), int(row["siteB"])
        true_dy = (SITES[b][0] - SITES[a][0]) * STEP
        true_dx = (SITES[b][1] - SITES[a][1]) * STEP
        dy, dx = float(row["dy_px_full"]), float(row["dx_px_full"])
        if abs(dy - true_dy) > 15 or abs(dx - true_dx) > 15:
            wrong.append(f"site {a}->{b}: got ({dy:.1f}, {dx:.1f}), "
                         f"built ({true_dy}, {true_dx})")
    assert not wrong, "offsets do not match the plate:\n  " + "\n  ".join(wrong)


def test_a_mosaic_is_written_without_being_asked_twice(stitched_plate):
    """``stitch_cycle_wells`` promises mosaics, so it must produce one.

    Its docstring says it produces "single- or multi-channel mosaics" and it
    produced none. Two separate flags -- ``mosaic`` and ``write_mosaic`` --
    meant opposite things: `write_mosaic=True` set the output path to None,
    so the one value a reader would reach for was the one guaranteed to write
    nothing. No flag is passed here, on purpose.
    """
    _result, out_dir = stitched_plate
    mosaics = [os.path.join(dirpath, name)
               for dirpath, _dirs, files in os.walk(out_dir)
               for name in files
               if "mosaic" in name.lower() and name.endswith((".tif", ".tiff"))]
    assert mosaics, "no mosaic was written, though the docstring promises one"

    image = tifffile.imread(mosaics[0])
    height, width = image.shape[-2:]
    # The canvas is the field the tiles were cut from, give or take the
    # subpixel offsets the stitch solved for.
    assert abs(height - (TILE + STEP)) <= 8, f"canvas height {height}"
    assert abs(width - (TILE + STEP)) <= 8, f"canvas width {width}"


def test_no_tile_is_left_out_of_the_mosaic(stitched_plate):
    """Every tile must reach the manifest, or the mosaic has a hole.

    ``_pairs_by_site_window`` built ``idx_by_site[s] = i``, keeping ONE index
    per site -- but a site holds one file per channel. With the files ordered
    c1_S1, c2_S1, c1_S2, c2_S2, ... the map became {1:1, 2:3, 3:5, 4:7}: every
    candidate partner was a channel-2 file, two tiles of the same channel were
    never compared, and the c1/c2 rows of the pairs CSV were one comparison
    written twice.

    It also orphaned a tile outright. ``10X_c1_A1_Site-4.tif`` at index 6 had
    a single candidate, index 5, which fails ``j > i`` -- so it appeared in no
    pair, got no features, and reached no mosaic. A corner arrived with no
    channel-1 data and no warning, which is the failure mode a GUI button over
    this code would have shipped.
    """
    _result, out_dir = stitched_plate
    manifest = None
    for dirpath, _dirs, files in os.walk(out_dir):
        for name in files:
            if name.endswith("_mosaic.csv"):
                manifest = os.path.join(dirpath, name)
    assert manifest, "no mosaic manifest was written"

    placed = {os.path.basename(row["path"])
              for row in csv.DictReader(open(manifest, encoding="utf-8"))}
    expected = {f"10X_c{c}_A1_Site-{s}.tif" for c in (1, 2) for s in SITES}
    assert placed == expected, f"tiles missing from the mosaic: {expected - placed}"


@pytest.mark.parametrize("shape,axes,expected", [
    ((64, 64), None, 1),              # a plain 2D plane
    ((2, 64, 64), None, 2),           # a stacked 2-channel tile
    ((4, 64, 64), None, 4),
    ((5, 64, 64), "ZYX", 1),          # z-planes are NOT channels
])
def test_the_channel_count_survives_an_unnamed_axis(shape, axes, expected,
                                                    tmp_path):
    """A stacked tile declares axes ``QYX``, and Q is not nothing.

    ``_get_channel_count_tif`` filtered the declared axes down to ``TCZYX``,
    found no ``C``, and returned 1 -- without ever reaching the shape-based
    guess below it, which gets ``(2, Y, X)`` right. `tifffile` writes ``Q``,
    meaning "unspecified", for the leading axis of a plainly stacked array, so
    every genuine multi-channel tile written that way counted as one channel.
    A multi-channel mosaic then came out single-channel, silently, because the
    channel plan is ``range(min(counts))``.

    The z-stack case is why this cannot simply always guess from the shape: a
    file that DOES name its axes as ``ZYX`` has one channel, and guessing
    would call its planes channels instead.
    """
    from spacr.spacrops import spacrStitcher

    path = tmp_path / f"tile_{'x'.join(map(str, shape))}_{axes or 'auto'}.tif"
    kwargs = {"metadata": {"axes": axes}} if axes else {}
    tifffile.imwrite(str(path), np.zeros(shape, np.uint16), **kwargs)
    assert spacrStitcher._get_channel_count_tif(str(path)) == expected
