"""A4 through the engine: a phenotype acquisition placed on a stitched well.

TWO ACQUISITIONS OF ONE WELL are written to disk, from one planted field of
nuclei: a five-tile sequencing raster at 10X, and a twenty-one-field
phenotype raster at 20X whose pitch is HALF the sequencing one -- which is
the arrangement the real plate has (333 sequencing fields, 1,281 phenotype
fields, raster steps of 633.8 px against 1,267).

That halving is the whole difficulty and the reason the placement is not an
index calculation. PART 14-M measured the index calculation -- halve the
phenotype grid position about the well centre and round -- at 0.45 to 0.61 of
fields on the right sequencing tile, because at twice the tile density half
the phenotype fields sit on or near a tile boundary and a grid index cannot
see where the boundary fell. Here every field's centre is planted at a known
place, so "the right tile" is a fact rather than an opinion.

Nothing is segmented: A4 aligns on nuclear CENTRES, which is why it can run
before Phase B.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("skimage")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_layout import round_well_layout

#: The sequencing tile, and an overlap of 30 % of it -- the fraction the real
#: raster has (213 px on 1,480) and enough that every correlation strip holds
#: content its neighbour shares.
TILE = 320
OVERLAP = 96
STEP = TILE - OVERLAP
MARGIN = 20

#: The phenotype acquisition, at the plate's own geometry: 20X against 10X
#: is HALF the field linearly, and 2,960 px against 1,480 is twice the
#: pixels, so a phenotype tile covers a QUARTER of a sequencing tile's area
#: at FOUR times the linear pixel density. The raster pitch is half the
#: sequencing one, which is what puts 1,281 phenotype fields over 333
#: sequencing ones and half of them on or near a tile boundary.
#:
#: The scale `spacr.ops_phenotype` records every constant against is
#: therefore 0.25, and it is NOT the magnification ratio. Building this
#: fixture at 0.5 -- the ratio of the objectives -- made a well that no
#: microscope could take, and it agreed with an engine that had the same
#: error in it.
DENSITY = 4
PHENOTYPE_TILE = TILE * DENSITY // 2
PHENOTYPE_PITCH = STEP // 2
PHENOTYPE_FIELDS = 21
SCALE = 1.0 / DENSITY

#: Enough nuclei that one phenotype field -- a quarter of a sequencing tile,
#: about four per cent of this little well -- holds far more than the forty
#: agreeing ones `spacr.ops_phenotype.MIN_INLIERS` asks for, and far enough
#: apart that smoothing at a nucleus's own scale does not merge two peaks
#: into one.
NUCLEI = 1600
SEPARATION = 13
RADIUS = 3


def _canvas():
    """The planted well, and where every field of both rasters sits in it.

    :returns: ``(dapi, sequencing tops, phenotype centres)`` -- the nuclear
        canvas in its own pixels, ``{site: (top, left)}`` for the sequencing
        tiles and ``{site: (y, x)}`` for the phenotype field centres, both in
        canvas coordinates.
    """
    rng = np.random.default_rng(372)
    sbs = round_well_layout(5)
    rows = [row for _column, row in sbs.positions()]
    top_row = min(rows)
    height = STEP * (max(rows) - top_row) + TILE + 2 * MARGIN
    width = STEP * (sbs.columns - 1) + TILE + 2 * MARGIN
    tops = {site: ((row - top_row) * STEP + MARGIN, column * STEP + MARGIN)
            for site, (column, row) in enumerate(sbs.positions())}

    centre = sbs.site(*sbs.centre)
    middle = (tops[centre][0] + TILE / 2.0, tops[centre][1] + TILE / 2.0)
    phenotype = round_well_layout(PHENOTYPE_FIELDS)
    centres = {
        site: (middle[0] + (row - phenotype.centre[1]) * PHENOTYPE_PITCH,
               middle[1] + (column - phenotype.centre[0]) * PHENOTYPE_PITCH)
        for site, (column, row) in enumerate(phenotype.positions())}

    planted = []
    while len(planted) < NUCLEI:
        y, x = (float(v) for v in rng.uniform(0, [height, width]))
        if any(math.hypot(y - a, x - b) < SEPARATION for a, b in planted):
            continue
        planted.append((y, x))
    dapi = np.full((height, width), 220.0)
    dapi += rng.normal(0, 12, dapi.shape)
    yy, xx = np.mgrid[:height, :width]
    for y, x in planted:
        dapi[(yy - y) ** 2 + (xx - x) ** 2 <= RADIUS ** 2] += 4000.0
    return ndimage.gaussian_filter(dapi, 1.0), tops, centres


def _write(root, dapi, tops, centres):
    """Write both acquisitions the way they are named on disk.

    :param root: the run folder; ``sbs`` and ``phenotype`` are made under it.
    :param dapi: the planted canvas.
    :param tops: the sequencing tiles' top-left corners.
    :param centres: the phenotype fields' centres.
    """
    sbs = root / "sbs"
    sbs.mkdir(parents=True, exist_ok=True)
    for site, (top, left) in tops.items():
        tifffile.imwrite(sbs / f"10X_c1_A1_DAPI_Site-{site}.tif",
                         dapi[top:top + TILE, left:left + TILE]
                         .astype(np.uint16))

    zoomed = ndimage.zoom(dapi, DENSITY, order=1)
    folder = root / "phenotype"
    folder.mkdir(parents=True, exist_ok=True)
    half = PHENOTYPE_TILE // 2
    for site, (y, x) in centres.items():
        top = int(round(y * DENSITY)) - half
        left = int(round(x * DENSITY)) - half
        tile = zoomed[top:top + PHENOTYPE_TILE, left:left + PHENOTYPE_TILE]
        assert tile.shape == (PHENOTYPE_TILE, PHENOTYPE_TILE), (site, top, left)
        tifffile.imwrite(folder / f"20X_c1_A1_DAPI_Site-{site}.tif",
                         tile.astype(np.uint16))


@pytest.fixture(scope="module")
def placed(tmp_path_factory):
    """Stitch the well and place the phenotype fields on it, once.

    :param tmp_path_factory: pytest's module-scoped temporary directories.
    :returns: ``(result, tops, centres)``.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", OVERLAP)
    root = tmp_path_factory.mktemp("ops_a4")
    dapi, tops, centres = _canvas()
    _write(root, dapi, tops, centres)
    settings = {"genotype_source": str(root / "sbs"),
                "phenotype_source": str(root / "phenotype"),
                "dst_root": str(root / "out"), "plate": "synthetic",
                "ops_gpu": False, "n_workers": 1}
    result = ops_engine.run_ops(settings, phases=("stitch", "phenotype"))
    yield result, tops, centres, settings
    monkeypatch.undo()


def test_six_fields_are_aligned_and_the_rest_are_predicted(placed):
    """Six anchors, and every one of the twenty-one fields gets a centre."""
    report = placed[0]["wells"]["A1"]["phenotype"]

    assert report["anchors_used"] == 6, report.get("anchors")
    assert report["fields"] == PHENOTYPE_FIELDS
    assert report["expected_scale"] == pytest.approx(SCALE)
    for anchor in report["anchors"]:
        assert anchor["scale"] == pytest.approx(SCALE, abs=0.005)
        assert abs(anchor["degrees"]) < 1.0
        assert anchor["inliers"] >= 40


def test_every_predicted_centre_lands_on_the_planted_one(placed):
    """The six anchors predict the other fifteen to a few pixels.

    The real wells measured a median error of 1.2 px over 472 held-out
    fields and 0.5 px over 451; this fixture is smaller and noisier, so the
    bar here is the tile boundary the next test depends on, not that number.
    """
    import sqlite3

    result, _tops, centres = placed[0], placed[1], placed[2]
    with sqlite3.connect(result["db"]) as conn:
        rows = conn.execute(
            "SELECT site, centre_y, centre_x, is_anchor FROM ops_phenotype "
            "ORDER BY site").fetchall()

    assert len(rows) == PHENOTYPE_FIELDS
    errors = []
    for site, y, x, _anchor in rows:
        want = (centres[site][0] - MARGIN, centres[site][1] - MARGIN)
        errors.append(math.hypot(y - want[0], x - want[1]))
    assert max(errors) < 8.0, sorted(errors)[-4:]
    assert float(np.median(errors)) < 4.0


def test_each_field_is_given_the_tile_that_actually_covers_it(placed):
    """The tile is chosen by position, which is what halving an index cannot do."""
    import sqlite3

    result, tops, centres = placed[0], placed[1], placed[2]
    with sqlite3.connect(result["db"]) as conn:
        mapped = {site: sbs_site for site, sbs_site in conn.execute(
            "SELECT site, sbs_site FROM ops_phenotype")}

    assert result["wells"]["A1"]["phenotype"]["fields_mapped"] == \
        PHENOTYPE_FIELDS
    assert set(mapped.values()) == set(tops)

    for site, (y, x) in centres.items():
        top, left = tops[mapped[site]]
        assert top <= y < top + TILE and left <= x < left + TILE, (
            site, (y, x), mapped[site], (top, left))


def test_the_placement_is_written_once_however_often_it_runs(placed):
    """Re-running the phase replaces the well's rows rather than doubling them."""
    import pandas as pd
    import sqlite3

    result, settings = placed[0], placed[3]
    with sqlite3.connect(result["db"]) as conn:
        before = pd.read_sql_query(
            "SELECT * FROM ops_phenotype ORDER BY site", conn)
    again = ops_engine.run_ops(settings, wells=["a1"], phases=("phenotype",))
    with sqlite3.connect(again["db"]) as conn:
        after = pd.read_sql_query(
            "SELECT * FROM ops_phenotype ORDER BY site", conn)

    assert len(after) == PHENOTYPE_FIELDS
    pd.testing.assert_frame_equal(before, after)


def test_no_phenotype_folder_skips_the_phase_rather_than_failing(placed):
    """The engine still runs a sequencing-only acquisition, and says it did."""
    settings = {**placed[3]}
    settings.pop("phenotype_source")
    result = ops_engine.run_ops(settings, wells=["A1"], phases=("phenotype",))

    assert result["wells"]["A1"]["phenotype"]["skipped"].startswith(
        "no phenotype_source")


def test_a_phenotype_folder_that_is_not_one_says_so(placed, tmp_path):
    """A wrong folder is named at the top of the run, not after the stitch."""
    settings = placed[3]
    with pytest.raises(ValueError, match="phenotype_source must be"):
        ops_engine.run_ops({**settings,
                            "phenotype_source": str(tmp_path / "nowhere")},
                           wells=["A1"], phases=("phenotype",))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="named like"):
        ops_engine.run_ops({**settings, "phenotype_source": str(empty)},
                           wells=["A1"], phases=("phenotype",))


#: The plate's own file names, copied from
#: `screenA/20200202_6W-LaC024A/{sequencing,phenotype}/images/input/` on
#: 2026-09-19. They are here as LITERALS so this test needs no NAS: the
#: acquisition's naming is what the index has to survive, and the index
#: failing on it is not something a fixture built from the same assumption
#: can catch.
PLATE_NAMES = {
    "sequencing/images/input/c1/10X_c1_B1_DAPI-CY3-A594-CY5-CY7_Site-0.tif":
        ("B1", 1, 0, "DAPI-CY3-A594-CY5-CY7", "10X"),
    "sequencing/images/input/c2/10X_c2_B1_A594_Site-100.tif":
        ("B1", 2, 100, "A594", "10X"),
    "sequencing/images/input/c11/10X_c11_A2_CY7_Site-332.tif":
        ("A2", 11, 332, "CY7", "10X"),
    "phenotype/images/input/DAPI-GFP-A594-AF750/"
    "20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-0.tif":
        ("B1", 1, 0, "DAPI-GFP", "20X"),
    "phenotype/images/input/DAPI-GFP-A594-AF750/"
    "20X_DAPI-GFP-A594-AF750_B1_A594_Site-1280.tif":
        ("B1", 1, 1280, "A594", "20X"),
    "phenotype/images/input/DAPI-GFP-A594-AF750/"
    "20X_DAPI-GFP-A594-AF750_A3_AF750_Site-640.tif":
        ("A3", 1, 640, "AF750", "20X"),
}


def test_the_plates_own_names_parse_both_ways():
    """The phenotype half carries no cycle, and A4 is what consumes it.

    372 predicted `20X_c1_B1_DAPI-GFP-A594-AF750_Site-0.tif` and the plate
    has `20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-0.tif`: no cycle field,
    the well AFTER the channel set, and the file's own channels a fifth
    field. Read with the cycled pattern alone, the phenotype acquisition
    indexes ZERO tiles and A4 refuses a well whose images are all there.
    """
    import os

    from spacr.ops_engine import _match_tile

    for path, want in PLATE_NAMES.items():
        assert _match_tile(os.path.basename(path)) == want, path


def test_a_cycled_name_is_never_read_as_an_uncycled_one():
    """`10X_c2_B1_A594_Site-0.tif` satisfies both patterns; order decides.

    Read by the uncycled pattern, `c2` is the acquisition's channel set and
    every one of the eleven cycles files under cycle 1 -- one site with one
    channel instead of eleven, which decodes at chance rather than failing.
    """
    from spacr.ops_engine import (_UNCYCLED_TILE_PATTERN, _match_tile)

    name = "10X_c2_B1_A594_Site-0.tif"
    assert _UNCYCLED_TILE_PATTERN.search(name), "the trap this guards"
    assert _match_tile(name)[1] == 2


def test_a_half_downloaded_file_is_not_a_tile():
    """Thirteen `*.tif.lftp-pget-status` files sit in the phenotype folder."""
    from spacr.ops_engine import _match_tile

    assert _match_tile(
        "20X_DAPI-GFP-A594-AF750_B1_A594_Site-99.tif.lftp-pget-status") is None


def test_an_uncycled_acquisition_indexes_and_stacks_its_nuclear_plane(tmp_path):
    """The index files it under one cycle, and DAPI is plane 0 of DAPI-GFP.

    The plate's nuclear channel is not a file of its own: it is the first
    plane of a two-plane `DAPI-GFP` stack, which is the case
    `_plane_sources` exists for. Checked here on files of the plate's names
    so that the phase's one hard input is tested without the NAS.
    """
    from spacr.ops_engine import (_NUCLEAR, _index_tiles, _plane_sources)

    folder = tmp_path / "phenotype" / "images" / "input" / "DAPI-GFP-A594-AF750"
    folder.mkdir(parents=True)
    for site in range(2):
        tifffile.imwrite(
            folder / f"20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-{site}.tif",
            np.stack([np.full((8, 8), 7, np.uint16),
                      np.full((8, 8), 9, np.uint16)]))
        tifffile.imwrite(
            folder / f"20X_DAPI-GFP-A594-AF750_B1_A594_Site-{site}.tif",
            np.full((8, 8), 3, np.uint16))

    index = _index_tiles(str(tmp_path))
    assert sorted(index) == ["B1"]
    assert sorted(index["B1"]) == [1], "an uncycled acquisition has one cycle"
    assert sorted(index["B1"][1]) == [0, 1]

    sources = _plane_sources(index["B1"][1][0])
    assert set(sources) == {"DAPI", "GFP", "A594"}
    assert sources[_NUCLEAR][1] == 0, "DAPI is the first plane of DAPI-GFP"
    assert sources["GFP"][1] == 1
    assert sources["A594"][1] is None
