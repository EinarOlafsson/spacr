"""Which sequencing tile a phenotype field lies on, from six measured fields.

Instruction 372's PART 14-L, section 2c. The halving map --
``round(s_col + (col - p_col) / ratio)``, a phenotype grid index halved
about the well centre -- got the exact sequencing tile for only 0.45 to 0.61
of the real plate's aligned fields, even once the layout's heights were
right. At twice the tile density half the phenotype fields sit on or near a
sequencing tile boundary, and rounding a grid index cannot see where that
boundary actually is. Fitting the raster's six numbers on six aligned fields
and taking the nearest stitched tile got it for every one::

    A1   472 held-out fields   centre error median 1.2 px, max 4.0 px   472/472
    A2   451 held-out fields   max 4.4 px                               451/451

THE PLATE HERE IS SYNTHETIC AND SHAPED LIKE THAT ONE. Both rasters are the
ones fitted to well A1, the phenotype well's centre field sits where A1's
did against the sequencing well's, and every field carries 1 px of stage
jitter. The grid positions are built here from the measured heights rather
than taken from the layout, so a layout defect cannot hide in the truth.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spacr.ops_layout import round_well_layout
from spacr.ops_phenotype import _fit_raster, phenotype_centres, phenotype_site_map

#: The phenotype well's measured column heights (PART 14-L 2b).
MEASURED_1281 = [7, 13, 17, 21, 25, 27, 29, 31, 33, 33, 35, 35, 37, 37, 39,
                 39, 39, 41, 41, 41, 41, 41, 41, 41, 39, 39, 39, 37, 37, 35,
                 35, 33, 33, 31, 29, 27, 25, 21, 17, 13, 7]

#: The sequencing well's confirmed column heights.
HEIGHTS_333 = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
               19, 19, 17, 17, 15, 13, 9, 5]

#: ``(dy, dx)`` of one column and one row, well pixels, fitted to well A1:
#: the stitched sequencing tiles, and the 478 aligned phenotype fields.
SBS_COL, SBS_ROW = (-8.7, 1267.4), (1267.0, 9.0)
PHENOTYPE_COL, PHENOTYPE_ROW = (-4.6, 633.8), (633.5, 4.5)

#: Well-frame centre of the middle field of each acquisition -- sequencing
#: grid ``(10, -10)`` and phenotype grid ``(20, -20)`` -- as on well A1.
SBS_MIDDLE = (3186.7, 13434.2)
PHENOTYPE_MIDDLE = (3177.7, 13439.0)

#: The six anchor fields PART 14-L used: the aligned fields nearest the
#: reference implementation's own `initial_sites` for this plate.
ANCHORS = (1, 184, 548, 656, 888, 1279)

#: A sequencing tile's side, in well pixels.
TILE = 1480

#: The stage jitter every true centre carries, in pixels (A1's raster
#: residual was a median 0.7 px).
JITTER_PX = 1.0


def _snake(heights, centre_row):
    """``site -> (column, row)`` for columns centred on one row, snaked.

    :param heights: fields per column, left to right.
    :param centre_row: the row every column is centred on.
    :returns: the grid positions in acquisition order.
    """
    places = []
    for column, height in enumerate(heights):
        rows = list(range(centre_row - (height - 1) // 2,
                          centre_row + (height - 1) // 2 + 1))
        places += [(column, row) for row in (rows[::-1] if column % 2
                                             else rows)]
    return places


def _raster(places, middle, centre, col_step, row_step):
    """Well-frame centres of grid positions on a skewed raster.

    :param places: ``(column, row)`` per site.
    :param middle: the well-frame centre of grid position ``centre``.
    :param centre: the grid position ``middle`` belongs to.
    :param col_step: ``(dy, dx)`` of one column.
    :param row_step: ``(dy, dx)`` of one row.
    :returns: an ``(n, 2)`` array of ``(y, x)``.
    """
    grid = np.asarray(places, dtype=float) - np.asarray(centre, dtype=float)
    return (np.asarray(middle) + np.outer(grid[:, 0], col_step)
            + np.outer(grid[:, 1], row_step))


def _halving_pick(site):
    """The sequencing site the retired halving map chose for a field.

    :param site: the phenotype site.
    :returns: the sequencing site at ``round(10 + (col - 20) / 2)``,
        ``round(-10 + (row + 20) / 2)`` -- the expression as it shipped,
        Python's rounding included -- or None off the sequencing well.
    """
    column, row = _snake(MEASURED_1281, -20)[site]
    target = (int(round(10 + (column - 20) / 2)),
              int(round(-10 + (row + 20) / 2)))
    sbs = {place: index for index, place
           in enumerate(_snake(HEIGHTS_333, -10))}
    return sbs.get(target)


@pytest.fixture(scope="module")
def plate():
    """The synthetic plate: true centres, stitched tiles and the truth.

    :returns: a namespace of ``truth`` (1,281 true centres), ``sbs_centres``
        (``{site: (y, x)}``), ``anchors`` (the six measured centres),
        ``true_tile`` (the nearest stitched tile per field) and ``margin``
        (how much farther the second-nearest tile is, px).
    """
    tiles = _raster(_snake(HEIGHTS_333, -10), SBS_MIDDLE, (10, -10),
                    SBS_COL, SBS_ROW)
    truth = _raster(_snake(MEASURED_1281, -20), PHENOTYPE_MIDDLE, (20, -20),
                    PHENOTYPE_COL, PHENOTYPE_ROW)
    truth = truth + np.random.default_rng(0).normal(0.0, JITTER_PX,
                                                    truth.shape)
    distance = np.hypot(truth[:, None, 0] - tiles[None, :, 0],
                        truth[:, None, 1] - tiles[None, :, 1])
    ordered = np.sort(distance, axis=1)
    return SimpleNamespace(
        truth=truth,
        sbs_centres={site: (float(y), float(x))
                     for site, (y, x) in enumerate(tiles)},
        anchors={site: (float(truth[site, 0]), float(truth[site, 1]))
                 for site in ANCHORS},
        true_tile=distance.argmin(axis=1),
        margin=ordered[:, 1] - ordered[:, 0])


def _errors(plate, centres):
    """Distance of each predicted centre from the truth, site order.

    :param plate: the fixture.
    :param centres: ``{site: (y, x)}``.
    :returns: an array of pixels, one per site.
    """
    predicted = np.array([centres[site] for site in range(len(plate.truth))])
    return np.hypot(*(predicted - plate.truth).T)


# ---------------------------------------------------------------------------
# Every centre, from six
# ---------------------------------------------------------------------------

class TestPredictingEveryCentreFromSixFields:
    """The raster's six numbers, fitted on the anchors' measured centres."""

    def test_six_anchors_place_every_field_to_a_few_pixels(self, plate):
        """1,281 fields, six measured, the rest predicted."""
        centres = phenotype_centres(1281, plate.anchors)
        assert sorted(centres) == list(range(1281))
        held_out = np.delete(_errors(plate, centres), list(ANCHORS))
        assert held_out.size == 1275
        assert float(np.median(held_out)) < 2.0, np.median(held_out)
        assert float(held_out.max()) < 6.0, held_out.max()

    def test_the_fitted_steps_are_the_acquisitions(self, plate):
        """Origin, column step and row step, and the anchors' residuals."""
        _origin, col_step, row_step, residuals = _fit_raster(
            round_well_layout(1281), plate.anchors)
        assert np.allclose(col_step, PHENOTYPE_COL, atol=0.2), col_step
        assert np.allclose(row_step, PHENOTYPE_ROW, atol=0.2), row_step
        assert sorted(residuals) == sorted(ANCHORS)
        assert max(residuals.values()) < 3 * JITTER_PX

    def test_a_layout_and_its_count_give_the_same_centres(self, plate):
        """A caller holding the layout need not rebuild it from a count."""
        assert (phenotype_centres(round_well_layout(1281), plate.anchors)
                == phenotype_centres(1281, plate.anchors))

    def test_three_anchors_off_one_line_are_enough(self):
        """Exact centres, three anchors: an exact raster, every field."""
        truth = _raster(_snake(MEASURED_1281, -20), PHENOTYPE_MIDDLE,
                        (20, -20), PHENOTYPE_COL, PHENOTYPE_ROW)
        anchors = {site: tuple(truth[site]) for site in (1, 660, 1279)}
        centres = phenotype_centres(1281, anchors)
        predicted = np.array([centres[site] for site in range(1281)])
        assert np.abs(predicted - truth).max() < 1e-6

    def test_a_misplaced_anchor_shows_in_the_residual(self, plate):
        """One anchor a row out is 633 px of disagreement, and it says so."""
        anchors = dict(plate.anchors)
        y, x = anchors[548]
        anchors[548] = (y + PHENOTYPE_ROW[0], x + PHENOTYPE_ROW[1])
        *_steps, residuals = _fit_raster(round_well_layout(1281), anchors)
        assert max(residuals.values()) > 100.0

    def test_too_few_anchors_are_refused(self, plate):
        """An origin and two steps cannot come from two fields."""
        two = {site: plate.anchors[site] for site in ANCHORS[:2]}
        with pytest.raises(ValueError, match="at least three"):
            phenotype_centres(1281, two)
        with pytest.raises(ValueError, match="at least three"):
            phenotype_centres(1281, {})

    def test_anchors_on_one_line_of_the_grid_are_refused(self, plate):
        """One column fixes the row step and nothing across it."""
        layout = round_well_layout(1281)
        column = [site for site in range(1281)
                  if layout.position(site)[0] == 20][:6]
        anchors = {site: tuple(plate.truth[site]) for site in column}
        with pytest.raises(ValueError, match="one line"):
            phenotype_centres(layout, anchors)

    def test_a_centre_that_is_not_a_point_is_refused(self, plate):
        """A NaN anchor would poison every prediction silently."""
        anchors = dict(plate.anchors)
        anchors[184] = (float("nan"), 0.0)
        with pytest.raises(ValueError, match="finite"):
            phenotype_centres(1281, anchors)

    def test_an_anchor_the_well_does_not_hold_is_an_index_error(self, plate):
        """Site 1281 of a 1,281-field well names nothing."""
        anchors = dict(plate.anchors)
        anchors[1281] = (0.0, 0.0)
        with pytest.raises(IndexError, match="1281"):
            phenotype_centres(1281, anchors)


# ---------------------------------------------------------------------------
# The nearest stitched tile
# ---------------------------------------------------------------------------

class TestTheNearestStitchedTileIsTheMap:
    """The question A4 asks, answered by position rather than by index."""

    def test_every_field_lands_on_the_tile_it_lies_nearest(self, plate):
        """Every field whose nearest tile the prediction can resolve.

        A predicted centre ``e`` px from the truth cannot change the nearest
        tile while the second-nearest is more than ``2e`` farther. One field
        of this plate lies 0.1 px from equidistant between two tiles, which
        no prediction resolves; every other one is exact.
        """
        centres = phenotype_centres(1281, plate.anchors)
        errors = _errors(plate, centres)
        mapping = phenotype_site_map(1281, plate.sbs_centres, plate.anchors,
                                     tile_shape=(TILE, TILE))
        assert sorted(mapping) == list(range(1281))
        wrong = {site for site, tile in mapping.items()
                 if tile != plate.true_tile[site]}
        unresolvable = {site for site in range(1281)
                        if plate.margin[site] <= 2 * errors[site]}
        assert wrong <= unresolvable, sorted(wrong - unresolvable)
        assert len(wrong) <= 1, sorted(wrong)

    def test_the_halving_map_sent_a_boundary_field_to_the_wrong_tile(
            self, plate):
        """Site 1143: 619 px from tile 305's centre and 648 px from 304's.

        It lies in the overlap of both, and halving its row index rounds to
        304. Measured against the tree before the change, on this plate:
        `phenotype_site_map(1281, 333)` sent it to 304 and got the exact tile
        for 500 of 1,281 fields (0.390), its worst miss 1,652 px; the same
        halving on the measured heights got 726 (0.567).
        """
        assert plate.true_tile[1143] == 305
        assert plate.margin[1143] < 40.0, "not a boundary field"
        assert _halving_pick(1143) == 304
        mapping = phenotype_site_map(1281, plate.sbs_centres, plate.anchors)
        assert mapping[1143] == 305

        halving = sum(1 for site in range(1281)
                      if _halving_pick(site) == plate.true_tile[site])
        exact = sum(1 for site, tile in mapping.items()
                    if tile == plate.true_tile[site])
        assert halving == 726, halving
        assert exact >= 1280, exact

    def test_the_centre_field_maps_to_the_centre_tile(self, plate):
        """The two wells' middles are the same place."""
        phenotype, sbs = round_well_layout(1281), round_well_layout(333)
        mapping = phenotype_site_map(phenotype, plate.sbs_centres,
                                     plate.anchors)
        assert mapping[phenotype.site(*phenotype.centre)] == sbs.site(
            *sbs.centre)

    def test_a_field_inside_no_tile_is_omitted(self, plate):
        """Tiles of 900 px on a 1,267 px pitch leave gaps, and a field in
        one has no tile to be mapped to."""
        centres = phenotype_centres(1281, plate.anchors)
        tiles = np.array(list(plate.sbs_centres.values()))
        mapping = phenotype_site_map(1281, plate.sbs_centres, plate.anchors,
                                     tile_shape=(900, 900))
        covered = set()
        for site, (y, x) in centres.items():
            inside = ((np.abs(tiles[:, 0] - y) <= 450)
                      & (np.abs(tiles[:, 1] - x) <= 450))
            if inside.any():
                covered.add(site)
        assert set(mapping) == covered
        assert 0 < len(covered) < 1281, len(covered)
        for site, tile in mapping.items():
            ty, tx = plate.sbs_centres[tile]
            y, x = centres[site]
            assert abs(ty - y) <= 450 and abs(tx - x) <= 450

    def test_no_stitched_tiles_is_no_map(self, plate):
        """Nothing to land on is an empty answer, not an error."""
        assert phenotype_site_map(1281, {}, plate.anchors) == {}

    def test_a_tile_shape_that_is_not_a_size_is_refused(self, plate):
        """A tile of no height covers nothing, and would omit every field."""
        for shape in ((0, TILE), (TILE, -1), (TILE,)):
            with pytest.raises(ValueError, match="tile_shape"):
                phenotype_site_map(1281, plate.sbs_centres, plate.anchors,
                                   tile_shape=shape)

    def test_a_count_that_is_not_a_round_well_says_so(self, plate):
        """1,000 fields fit no circle, so there is no layout to fit."""
        with pytest.raises(ValueError, match="not a circle"):
            phenotype_site_map(1000, plate.sbs_centres, plate.anchors)
