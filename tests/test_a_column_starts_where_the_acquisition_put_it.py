"""The row origin of each column, solved rather than derived from a circle.

Instruction 372's PART 14-F to PART 14-I. `round_well_layout` SEARCHES for a
radius that holds exactly the acquisition's field count, and more than one
circle does; the one it found for the 41-column phenotype well is not the one
the microscope used. Measured over 188 aligned fields, inverting the fitted
raster to read each field's grid index off its own measured centre:

    column index error    median +0.000   sd 0.001
    row index error       median +0.000   sd 0.997

The column is right every time -- so the per-column HEIGHTS are right, since
a wrong height would shift every later column's index and none shifted. The
row is wrong by a whole number that is CONSTANT down a column: 0.0028 rows of
spread inside any of them. Right heights, wrong origins, and 41 integers fix
it. That cost 133 of 321 fields, because one row is 633 px.

WHY THE PLANTED TRUTH. There is no acquisition on this box to check against,
so every test here builds a well whose offsets are KNOWN, synthesises the
centres that acquisition would have produced, and asks the solver for the
integers back. The one test that is not planted is the one that matters most
for landing this safely: the digests below were taken from the tree BEFORE
the field existed, so a layout that carries no offsets has to reproduce them
exactly or the change is not the no-op it claims to be.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from spacr.ops_layout import (MEASURED_PHENOTYPE_ROW_OFFSETS, WellLayout,
                              _solve_row_offsets, round_well_layout)
from spacr.ops_phenotype import phenotype_site_map

#: The raster 372's PART 14-G fitted to the phenotype acquisition, as
#: ``(dy, dx)`` per column and per row, in well pixels. Two steps of the
#: same length at right angles, each carrying the ~4.5 px of stage skew
#: PART 13-D found on the sequencing raster.
COL_STEP = (-4.60, 633.79)
ROW_STEP = (633.52, 4.52)

#: ``site count -> (columns, site count, pair count, sha256)`` of everything
#: :class:`WellLayout` answers -- heights, site count, every position and
#: every pair -- taken from this tree on 2026-09-14 BEFORE `row_offsets`
#: existed. The digest is the whole point: it is not possible to satisfy it
#: by accident, and it covers the four values the change could have moved.
PRE_CHANGE = {
    333: (21, 333, 624,
          "3d3ebbc7d61f3cc2b3e87c3fe5925955ca51a52dec8e769eba1a070aaad76c9f"),
    1281: (41, 1281, 2480,
           "24f95d72850ae55e8401b2746541f1fc8ae883c6b923ceab27feb454c2d5276e"),
    21: (5, 21, 32,
         "f0a9448806f2475b9b519ab30ddf72a8d5ababedf9cbb7009210980961b16087"),
    5: (3, 5, 4,
        "0307b1b1bee5269ad1dfc6857f5e5fbf3d4694f1faf816e8bdcce50356a54082"),
}

#: The column heights of the two acquisitions, written out rather than
#: derived, so that a digest mismatch can be read as "which of these moved".
HEIGHTS_333 = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
               19, 19, 17, 17, 15, 13, 9, 5]
HEIGHTS_1281 = [5, 13, 19, 21, 25, 27, 29, 31, 33, 33, 35, 37, 37, 37, 39,
                39, 39, 39, 41, 41, 41, 41, 41, 39, 39, 39, 39, 37, 37, 37,
                35, 33, 33, 31, 29, 27, 25, 21, 19, 13, 5]


def _digest(layout):
    """Everything the layout answers, as one hash.

    :param layout: the well.
    :returns: the sha256 of its heights, site count, positions and pairs.
    """
    return hashlib.sha256(json.dumps({
        "columns": layout.columns,
        "radius": layout.radius,
        "centre": list(layout.centre),
        "snake": layout.snake,
        "heights": layout.heights,
        "site_count": layout.site_count,
        "positions": [list(place) for place in layout.positions()],
        "pairs": [list(pair) for pair in layout.pairs()],
    }, sort_keys=True).encode()).hexdigest()


def _acquired(layout, origin=(0.0, 0.0), col_step=COL_STEP,
              row_step=ROW_STEP, jitter=0.0, sites=None):
    """The centres an acquisition of ``layout`` would have measured.

    :param layout: the well, carrying the row offsets that are the truth.
    :param origin: where grid ``(0, 0)`` lands in the measured frame.
    :param col_step: ``(dy, dx)`` of one column.
    :param row_step: ``(dy, dx)`` of one row.
    :param jitter: a fixed sub-row wobble added to every centre, in the
        same units, standing in for the stage not being perfect.
    :param sites: which sites were placed; all of them by default.
    :returns: ``site -> (y, x)``.
    """
    chosen = range(layout.site_count) if sites is None else sites
    out = {}
    for index, site in enumerate(chosen):
        column, row = layout.position(site)
        wobble = jitter * (1 if index % 2 else -1)
        out[site] = (origin[0] + col_step[0] * column + row_step[0] * row
                     + wobble,
                     origin[1] + col_step[1] * column + row_step[1] * row
                     + wobble)
    return out


# ---------------------------------------------------------------------------
# The no-op half: a layout with no offsets is the layout this tree already had
# ---------------------------------------------------------------------------

class TestNoOffsetsMeansTodaysAnswer:
    """The test that lets this land without an acquisition to check it on."""

    @pytest.mark.parametrize("count", sorted(PRE_CHANGE))
    def test_an_unmeasured_layout_is_byte_identical_to_the_old_one(self,
                                                                   count):
        """Heights, site count, positions and pairs, all unmoved."""
        columns, sites, pairs, digest = PRE_CHANGE[count]
        layout = round_well_layout(count)
        # Non-vacuity first: a digest of nothing would match a digest of
        # nothing, so the shape behind it is asserted in the open.
        assert layout.columns == columns
        assert layout.site_count == sites == len(layout.positions())
        assert len(layout.pairs()) == pairs
        assert sum(layout.heights) == sites
        assert layout.row_offsets == ()
        assert _digest(layout) == digest, (
            f"a {count}-field well no longer answers what it answered before "
            "`row_offsets` existed")

    def test_the_two_measured_acquisitions_keep_their_heights(self):
        """Written out, so a digest failure says which end moved."""
        assert round_well_layout(333).heights == HEIGHTS_333
        assert round_well_layout(1281).heights == HEIGHTS_1281

    @pytest.mark.parametrize("count", sorted(PRE_CHANGE))
    def test_a_table_of_zeros_is_the_same_as_no_table_at_all(self, count):
        """`()` is not a special case, it is the bottom of the same range."""
        _columns, _sites, _pairs, digest = PRE_CHANGE[count]
        layout = round_well_layout(count)
        zeroed = replace(layout, row_offsets=(0,) * layout.columns)
        assert _digest(zeroed) == digest
        assert zeroed.positions() == layout.positions()
        assert zeroed.pairs() == layout.pairs()

    def test_a_short_table_leaves_the_columns_it_does_not_reach(self):
        """Padding, because `()` has to keep meaning what it meant."""
        layout = round_well_layout(333)
        partial = replace(layout, row_offsets=(1,))
        assert partial.span(0) != layout.span(0)
        for column in range(1, layout.columns):
            assert partial.span(column) == layout.span(column)

    def test_the_layout_is_still_hashable_because_its_walk_is_cached(self):
        """A list or a dict here would break `_index`'s `lru_cache`."""
        assert hash(WellLayout(row_offsets=(1, 2))) == hash(
            WellLayout(row_offsets=(1, 2)))
        assert WellLayout(row_offsets=(1, 2)) != WellLayout(row_offsets=(2, 1))
        assert isinstance(WellLayout().row_offsets, tuple)
        # And the cache answers per layout rather than per class: the same
        # well with two different tables must not share one walk.
        shifted = WellLayout(row_offsets=(3,) * 21)
        assert shifted.positions() != WellLayout().positions()

    def test_an_offset_moves_a_column_without_resizing_it(self):
        """The heights were right. Only the origins were wrong."""
        layout = round_well_layout(1281)
        shifted = replace(layout, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        assert shifted.heights == layout.heights
        assert shifted.site_count == layout.site_count
        for site in (0, 1, 640, 1280):
            assert shifted.position(site)[0] == layout.position(site)[0]
        moved = [site for site in range(layout.site_count)
                 if shifted.position(site)[1] != layout.position(site)[1]]
        assert len(moved) == 608, (
            "the 20 non-zero columns of the measured table hold 608 fields")


# ---------------------------------------------------------------------------
# The solver, against offsets that are known because they were planted
# ---------------------------------------------------------------------------

class TestSolvingTheRowOriginFromPlantedTruth:

    def test_it_recovers_the_integers_it_was_given(self):
        """Chosen columns shifted by -2..+2, on a round well."""
        plain = round_well_layout(333)
        planted = [0] * plain.columns
        for column, offset in ((1, -2), (4, +1), (9, -1), (10, +2), (17, +2)):
            planted[column] = offset
        truth = replace(plain, row_offsets=tuple(planted))
        solved = _solve_row_offsets(_acquired(truth), plain,
                                    COL_STEP, ROW_STEP)
        assert solved == tuple(planted)

    def test_it_recovers_the_measured_phenotype_table(self):
        """41 columns, 20 of them non-zero, every value in -2..+2.

        The table 372's PART 14-I measured, planted back into the well it
        came from and asked for again. It is the real shape of the problem
        -- alternating rings either side of the widest columns -- rather
        than a handful of columns chosen to be easy.
        """
        plain = round_well_layout(1281)
        truth = replace(plain, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        solved = _solve_row_offsets(_acquired(truth), plain,
                                    COL_STEP, ROW_STEP)
        assert solved == MEASURED_PHENOTYPE_ROW_OFFSETS
        assert sum(1 for value in solved if value) == 20
        assert max(abs(value) for value in solved) == 2

    def test_the_measured_frame_may_sit_anywhere(self):
        """The well frame's origin is wherever the stitch pinned site 0.

        INCLUDING HALF A ROW OFF, which is the one that needs the gauge
        rather than the pin. An origin a whole number of rows out shifts
        every column the same way and the pin takes it straight back off;
        an origin 316 px out -- half of a 633 px row -- lands every field
        exactly on a rounding boundary, and which way each one falls is
        then decided by the arithmetic rather than by the acquisition. The
        median over the placed fields takes the fraction off first, so the
        rounding happens where the fields actually are.
        """
        plain = round_well_layout(1281)
        truth = replace(plain, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        half_a_row = (0.5 * ROW_STEP[0], 0.5 * ROW_STEP[1])
        for origin in ((0.0, 0.0), (-9182.5, 4410.25), (1e5, -1e5),
                       half_a_row):
            solved = _solve_row_offsets(_acquired(truth, origin=origin),
                                        plain, COL_STEP, ROW_STEP)
            assert solved == MEASURED_PHENOTYPE_ROW_OFFSETS, origin

    def test_the_constant_is_pinned_on_columns_and_not_on_fields(self):
        """The gauge, exercised where the two ways of pinning it differ.

        Shifting every column by a row and moving the raster's origin a row
        the other way produce the very same centres, so the measurement
        fixes the table only up to a constant and something has to pin it.
        Here the nine widest columns hold 181 of the 333 fields -- a
        MAJORITY OF FIELDS in a MINORITY OF COLUMNS -- so a constant pinned
        on the field count would call those nine columns zero and report
        the other twelve as -2, which is the same well described upside
        down. The rule is the one the measured acquisition looks like: the
        largest group of COLUMNS is the one that reads zero.
        """
        plain = round_well_layout(333)
        planted = tuple(2 if 6 <= column <= 14 else 0
                        for column in range(plain.columns))
        truth = replace(plain, row_offsets=planted)
        shifted_fields = sum(height for height, offset
                             in zip(plain.heights, planted) if offset)
        assert shifted_fields == 181 > plain.site_count / 2
        assert sum(1 for offset in planted if offset) == 9 < plain.columns / 2
        solved = _solve_row_offsets(_acquired(truth), plain,
                                    COL_STEP, ROW_STEP)
        assert solved == planted

    def test_a_solve_against_a_corrected_layout_returns_that_table(self):
        """Idempotent, which is what makes it safe to re-run on a well."""
        plain = round_well_layout(1281)
        truth = replace(plain, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        assert _solve_row_offsets(_acquired(truth), truth,
                                  COL_STEP, ROW_STEP) == (
            MEASURED_PHENOTYPE_ROW_OFFSETS)

    def test_stage_wobble_under_half_a_row_does_not_move_an_integer(self):
        """A row is 633 px and the fields agree to 0.003 of one."""
        plain = round_well_layout(1281)
        truth = replace(plain, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        solved = _solve_row_offsets(_acquired(truth, jitter=120.0), plain,
                                    COL_STEP, ROW_STEP)
        assert solved == MEASURED_PHENOTYPE_ROW_OFFSETS

    def test_a_column_nothing_aligned_in_is_left_where_it_was(self):
        """Five columns produced no alignment at all, and 0 is the honest
        answer for them: register what you can, solve, place the rest."""
        plain = round_well_layout(1281)
        truth = replace(plain, row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        blind = {13, 16, 24, 27, 39}
        placed = [site for site in range(plain.site_count)
                  if plain.position(site)[0] not in blind]
        solved = _solve_row_offsets(_acquired(truth, sites=placed), plain,
                                    COL_STEP, ROW_STEP)
        for column in range(plain.columns):
            expected = (0 if column in blind
                        else MEASURED_PHENOTYPE_ROW_OFFSETS[column])
            assert solved[column] == expected, column
        # And the ones that WERE observed are still right, which is the
        # half of this that a table of all zeros would also satisfy.
        assert sum(1 for value in solved if value) == 15

    def test_a_column_with_one_surviving_field_is_not_given_a_number(self):
        """A majority of one is not a majority.

        `counts[best] * 2 > len(values)` is `1 * 2 > 1`, so the guard meant to
        stop one misplaced field deciding a column could not fire in the one
        case where a single field IS the column. The mode's whole defence --
        that a bad field gets outvoted -- needs three votes before it means
        anything.

        This is the live regime rather than a hypothetical: PART 14-G aligned
        188 of 321 fields and PART 14-I needed a wide-window run for the five
        columns nothing aligned in, so columns down to one or two survivors are
        one step from what was already measured.
        """
        plain = round_well_layout(333)
        centres = _acquired(plain)
        column_one = [site for site in range(plain.site_count)
                      if plain.position(site)[0] == 1]
        # Column 1 keeps exactly ONE field, and that field is four rows low.
        kept = column_one[0]
        centres[kept] = (centres[kept][0] + 4 * ROW_STEP[0],
                         centres[kept][1] + 4 * ROW_STEP[1])
        for site in column_one[1:]:
            centres.pop(site, None)

        solved = _solve_row_offsets(centres, plain, COL_STEP, ROW_STEP)

        assert solved[1] == plain._shift(1), (
            f"one field decided a whole column: column 1 answered {solved[1]}")

    def test_two_agreeing_fields_are_two_fields_not_a_vote(self):
        """Three is the smallest number at which a stray can be outvoted."""
        plain = round_well_layout(333)
        centres = _acquired(plain)
        column_two = [site for site in range(plain.site_count)
                      if plain.position(site)[0] == 2]
        for site in column_two[:2]:
            centres[site] = (centres[site][0] + 3 * ROW_STEP[0],
                             centres[site][1] + 3 * ROW_STEP[1])
        for site in column_two[2:]:
            centres.pop(site, None)

        solved = _solve_row_offsets(centres, plain, COL_STEP, ROW_STEP)

        assert solved[2] == plain._shift(2)

    def test_one_field_placed_against_the_wrong_neighbour_is_outvoted(self):
        """A mode, not a mean: one bad field cannot move a column.

        The stray is twenty rows out, which is what a field aligned against
        the wrong part of the well looks like rather than a wobble. Twenty
        of the column's 21 fields say +2 and one says +22, so the average
        says +3 and the vote says +2. The average is the one that would
        ship a whole column to the wrong row on the strength of a single
        bad alignment, which is the failure this module exists to undo.
        """
        plain = round_well_layout(333)
        truth = replace(plain, row_offsets=(0,) * 8 + (2,) + (0,) * 12)
        centres = _acquired(truth)
        column_eight = [site for site in range(plain.site_count)
                        if plain.position(site)[0] == 8]
        assert len(column_eight) == 21
        strayed = column_eight[0]
        centres[strayed] = (centres[strayed][0] + 20 * ROW_STEP[0],
                            centres[strayed][1] + 20 * ROW_STEP[1])
        naive_mean = (2 * 20 + 22) / 21
        assert round(naive_mean) == 3, "the mean is not being put to the test"
        solved = _solve_row_offsets(centres, plain, COL_STEP, ROW_STEP)
        assert solved[8] == 2

    def test_a_column_that_disagrees_with_itself_is_not_given_a_number(self):
        """It was not measured, and inventing an integer for it is the
        mistake this whole part is correcting."""
        plain = round_well_layout(333)
        centres = _acquired(plain)
        column_eight = [site for site in range(plain.site_count)
                        if plain.position(site)[0] == 8]
        for index, site in enumerate(column_eight):
            step = (index % 5) - 2
            centres[site] = (centres[site][0] + step * ROW_STEP[0],
                             centres[site][1] + step * ROW_STEP[1])
        solved = _solve_row_offsets(centres, plain, COL_STEP, ROW_STEP)
        assert solved[8] == 0
        assert solved == (0,) * plain.columns

    def test_a_tie_between_two_pins_goes_to_the_one_nearest_zero(self):
        """Ten columns say one thing and ten say the other.

        The two readings of that well are equally supported -- it is the
        same degeneracy as before, with nothing to break it -- so the rule
        is written down rather than left to whichever column the dict
        happened to yield first: the offsets nearest zero win. Two runs
        over one acquisition have to agree, and a tie decided by iteration
        order is a difference that would show up as a moved well.
        """
        plain = round_well_layout(333)
        planted = tuple(2 if column <= 9 else 0
                        for column in range(plain.columns))
        truth = replace(plain, row_offsets=planted)
        placed = [site for site in range(plain.site_count)
                  if plain.position(site)[0] != 20]
        observed = {plain.position(site)[0] for site in placed}
        assert sum(1 for column in observed if planted[column]) == 10
        assert sum(1 for column in observed if not planted[column]) == 10
        solved = _solve_row_offsets(_acquired(truth, sites=placed), plain,
                                    COL_STEP, ROW_STEP)
        assert solved == planted

    def test_the_order_the_fields_arrive_in_does_not_change_the_answer(self):
        """The centres are a dict, and a mode decided by insertion order
        would make the answer depend on which field was placed first.

        ASKED WHERE THE ORDER COULD DECIDE IT, which is the tie above: a
        well with a clear winner answers the same in any order whatever the
        tie-break is, so a well with a clear winner tests nothing here. Ten
        columns against ten, fed forwards and then backwards, is the case
        where "whichever came first" and "whichever is nearest zero" part
        company -- and the two runs have to agree with each other AND with
        the acquisition.
        """
        plain = round_well_layout(333)
        planted = tuple(2 if column <= 9 else 0
                        for column in range(plain.columns))
        truth = replace(plain, row_offsets=planted)
        placed = [site for site in range(plain.site_count)
                  if plain.position(site)[0] != 20]
        forwards = _acquired(truth, sites=placed)
        backwards = dict(reversed(list(forwards.items())))
        assert list(backwards) != list(forwards)
        assert (_solve_row_offsets(backwards, plain, COL_STEP, ROW_STEP)
                == _solve_row_offsets(forwards, plain, COL_STEP, ROW_STEP)
                == planted)

        # And the same on the whole phenotype well, where the winner is
        # clear and the answer must simply not move.
        big = round_well_layout(1281)
        measured = _acquired(replace(big,
                                     row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS))
        assert (_solve_row_offsets(dict(reversed(list(measured.items()))), big,
                                   COL_STEP, ROW_STEP)
                == MEASURED_PHENOTYPE_ROW_OFFSETS)

    def test_nothing_measured_leaves_the_layout_alone(self):
        plain = round_well_layout(333)
        assert _solve_row_offsets({}, plain, COL_STEP, ROW_STEP) == (
            (0,) * plain.columns)
        truth = replace(plain, row_offsets=(1,) * plain.columns)
        assert _solve_row_offsets({}, truth, COL_STEP, ROW_STEP) == (
            (1,) * plain.columns)

    def test_a_raster_with_one_axis_is_refused_rather_than_inverted(self):
        plain = round_well_layout(333)
        with pytest.raises(ValueError, match="parallel"):
            _solve_row_offsets(_acquired(plain), plain,
                               (1.0, 2.0), (2.0, 4.0))

    def test_a_centre_for_a_site_the_well_does_not_hold_says_so(self):
        plain = round_well_layout(333)
        with pytest.raises(IndexError, match="333"):
            _solve_row_offsets({333: (0.0, 0.0)}, plain, COL_STEP, ROW_STEP)


# ---------------------------------------------------------------------------
# And the correction has somewhere to go
# ---------------------------------------------------------------------------

class TestTheCorrectionReachesTheSiteMap:

    def test_no_table_is_the_map_this_tree_already_produced(self):
        """The site map is the caller that has to be unchanged too."""
        plain = phenotype_site_map(1281, 333)
        assert len(plain) == 1273
        assert phenotype_site_map(1281, 333, row_offsets=()) == plain
        assert phenotype_site_map(
            1281, 333, row_offsets=(0,) * 41) == plain

    def test_a_corrected_layout_places_fields_the_circle_refused(self):
        """The defect, end to end: solve the origins, get the fields back.

        A phenotype field whose row is two out maps to a sequencing tile
        two rows from the right one -- and near the rim that is off the
        sequencing circle entirely, so the map drops the field and A4
        never gets a window to look in.
        """
        plain_layout = round_well_layout(1281)
        truth = replace(plain_layout,
                        row_offsets=MEASURED_PHENOTYPE_ROW_OFFSETS)
        solved = _solve_row_offsets(_acquired(truth), plain_layout,
                                    COL_STEP, ROW_STEP)
        plain = phenotype_site_map(1281, 333)
        corrected = phenotype_site_map(1281, 333, row_offsets=solved)
        gained = set(corrected) - set(plain)
        assert gained == {17, 382, 898, 1263}, (
            "the corrected layout places no field the circle refused, so "
            "this test is not looking at the defect at all")
        moved = {site for site in set(plain) & set(corrected)
                 if plain[site] != corrected[site]}
        assert len(moved) == 382, (
            "382 of the 1,281 fields were being sent to the wrong sequencing "
            "tile, which is the 633 px per row that refused them")
        # AND IT HANDS BACK 20, said out loud rather than buried: a rim
        # field whose true row is two out lands off the sequencing circle
        # and this closed-form map has no tile for it. That is the honest
        # answer for a one-to-one map and it is why PART 14-C wants A4
        # windowed on the well frame rather than on a single tile.
        assert len(set(plain) - set(corrected)) == 20

    def test_a_table_of_the_wrong_length_is_a_mistake_and_not_a_partial(self):
        """41 columns want 41 integers. A short table would correct some
        columns and silently leave the rest, which is the state this is
        supposed to end."""
        with pytest.raises(ValueError, match="41"):
            phenotype_site_map(1281, 333, row_offsets=(0,) * 21)
