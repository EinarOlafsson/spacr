"""The round well's columns, from the one physical fact the circle left out.

Instruction 372's PART 14-L, section 2b. A site is acquired when the WHOLE
TILE lies inside the well circle, not when its centre does. With the half
tile ``a`` measured in pitches, a site at grid ``(dx, dy)`` from the centre
is kept iff ``(|dx| + a)**2 + (|dy| + a)**2 <= R**2``, and every column is
centred on the same row::

    sequencing 10X   a = 740 / 1267.1            = 0.5840   333 sites
    phenotype 20X    a = 1480 * 0.25069 / 633.52 = 0.5857   1,281 sites

The centre rule (``a = 0``) holds the sequencing well, which is why PART 11-B
confirmed it, and cannot produce the phenotype heights at ANY radius. The
circle it found instead had the wrong heights, and PART 14-I's 41 row
offsets were what a wrong height table looks like when a grid is sampled
only inside its columns: they fixed the interiors and left every junction
field a column off. On the real plate the shipped circle placed 257 of well
A1's 478 aligned fields at their measured grid position; the rule places 478,
and 457 of 457 on well A2 with nothing refitted.

WHY THE DIGESTS. The sequencing layout is the one every stitch runs on, and
the change must not move it. The digests below were taken on 2026-09-14 from
the tree before `row_offsets` existed and re-taken on 2026-09-15 before
`half_tile` did: they agree, so they are the centre rule's answer, and a half
tile of zero has to reproduce them byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

import spacr.ops_layout as ops_layout
from spacr.ops_layout import WellLayout, round_well_layout

#: ``site count -> (columns, site count, pair count, sha256)`` of everything
#: :class:`WellLayout` answers under the centre rule -- heights, site count,
#: every position and every pair -- taken from the tree before `half_tile`.
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

#: The sequencing well's 21 heights, confirmed by registration (PART 11-B).
HEIGHTS_333 = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21,
               19, 19, 17, 17, 15, 13, 9, 5]

#: What the centre rule's circle gave the phenotype well. Wrong.
CENTRE_RULE_1281 = [5, 13, 19, 21, 25, 27, 29, 31, 33, 33, 35, 37, 37, 37,
                    39, 39, 39, 39, 41, 41, 41, 41, 41, 39, 39, 39, 39, 37,
                    37, 37, 35, 33, 33, 31, 29, 27, 25, 21, 19, 13, 5]

#: The phenotype well's heights as the junction fields of well A1 measured
#: them (PART 14-L 2b), and as well A2 then confirmed with nothing refitted.
MEASURED_1281 = [7, 13, 17, 21, 25, 27, 29, 31, 33, 33, 35, 35, 37, 37, 39,
                 39, 39, 41, 41, 41, 41, 41, 41, 41, 39, 39, 39, 37, 37, 35,
                 35, 33, 33, 31, 29, 27, 25, 21, 17, 13, 7]

#: The two acquisitions' half tiles, in pitches, from tile size over
#: registered pitch. Neither was fitted to a height.
MEASURED_HALF_TILES = {
    "sequencing 10X": 740 / 1267.1,
    "phenotype 20X": 1480 * 0.25069 / 633.52,
}

#: EVERY FIELD OF WELL A1 THE SHIPPED CIRCLE PUT IN THE WRONG COLUMN, at the
#: ``(column, row)`` it was measured at: 21 of 478 aligned fields, clustered
#: at 14 column junctions. Measured by inverting a raster fitted to all 478
#: well-frame centres (step2_A1_sweep, _colends, _junctions); every one of
#: the 478 read back within 0.1 of a pitch of an integer position.
A1_FIELDS_A_COLUMN_OFF = {
    6: (0, -17), 18: (1, -25), 306: (12, -38), 307: (12, -37),
    343: (13, -2), 344: (13, -3), 380: (14, -39), 381: (14, -38),
    459: (16, -38), 497: (17, 0), 821: (24, -2), 822: (24, -1),
    861: (25, -39), 899: (26, -2), 900: (26, -1), 936: (27, -37),
    937: (27, -38), 973: (28, -3), 974: (28, -2), 1261: (39, -14),
    1274: (40, -23),
}

#: The same for well A2, 16 of 457, which the heights were NOT measured on.
A2_FIELDS_A_COLUMN_OFF = {
    18: (1, -25), 19: (1, -26), 306: (12, -38), 343: (13, -2),
    381: (14, -38), 458: (16, -39), 497: (17, 0), 783: (23, -40),
    821: (24, -2), 822: (24, -1), 899: (26, -2), 900: (26, -1),
    937: (27, -38), 973: (28, -3), 974: (28, -2), 1261: (39, -14),
}


def _digest(layout):
    """Everything the layout answers, as one hash.

    :param layout: the well.
    :returns: the sha256 of its shape, heights, positions and pairs -- the
        same fields the pre-change digests were taken over.
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


def _misplaced(layout, table):
    """The fields of a measured table the layout puts somewhere else.

    :param layout: the well.
    :param table: ``site -> (column, row)`` as measured.
    :returns: ``site -> (layout's position, measured position)``.
    """
    return {site: (layout.position(site), want)
            for site, want in table.items() if layout.position(site) != want}


# ---------------------------------------------------------------------------
# A half tile of zero is the centre rule, and the sequencing well does not move
# ---------------------------------------------------------------------------

class TestTheCentreRuleIsTheZeroOfTheNewTerm:
    """The half that lets the change land without moving a stitch."""

    def test_the_sequencing_well_is_byte_identical_at_the_default(self):
        """333 fields: the confirmed layout, returned as it always was."""
        columns, sites, pairs, digest = PRE_CHANGE[333]
        layout = round_well_layout(333)
        assert layout == WellLayout()
        assert layout.columns == columns
        assert layout.site_count == sites == len(layout.positions())
        assert len(layout.pairs()) == pairs
        assert layout.heights == HEIGHTS_333
        assert _digest(layout) == digest

    @pytest.mark.parametrize("count", sorted(PRE_CHANGE))
    def test_a_half_tile_of_zero_reproduces_every_old_layout(self, count):
        """Heights, site count, positions, pairs and radius, all unmoved."""
        columns, sites, pairs, digest = PRE_CHANGE[count]
        layout = round_well_layout(count, half_tile=0.0)
        # Non-vacuity first: a digest of nothing matches a digest of nothing.
        assert layout.columns == columns
        assert layout.site_count == sites == sum(layout.heights)
        assert len(layout.pairs()) == pairs
        assert layout.half_tile == 0.0
        assert _digest(layout) == digest, (
            f"a {count}-field well at half_tile=0 no longer answers what the "
            "centre rule answered")

    def test_a_half_tile_of_zero_is_the_old_span_formula_at_every_radius(self):
        """`floor(sqrt(R^2 - dx^2))` about the centre, column by column.

        The radii are accumulated the way the search accumulates them, so
        the float rounding the old layouts were built on is the rounding
        this compares against.
        """
        radius = 0.5
        while radius <= 26.0:
            width = 2 * math.floor(radius) + 1
            layout = WellLayout(columns=width, radius=radius,
                                centre=(width // 2, -(width // 2)))
            assert layout.half_tile == 0.0
            for column in range(width):
                dx = column - layout.centre[0]
                remainder = radius * radius - dx * dx
                half = math.floor(math.sqrt(remainder))
                want = (layout.centre[1] - half, layout.centre[1] + half)
                assert layout.span(column) == want, (radius, column)
            radius += 0.05


# ---------------------------------------------------------------------------
# The rule: the whole tile inside the circle
# ---------------------------------------------------------------------------

class TestTheWholeTileMustFitInsideTheWell:
    """The phenotype well, which the centre rule could not describe."""

    def test_the_phenotype_count_gives_the_measured_heights(self):
        """41 columns, 1,281 fields, the heights the junctions measured."""
        layout = round_well_layout(1281)
        assert layout.site_count == 1281
        assert layout.columns == 41
        differ = [column for column, (got, want)
                  in enumerate(zip(layout.heights, MEASURED_1281))
                  if got != want]
        assert layout.heights == MEASURED_1281, (
            f"{len(differ)} of 41 column heights differ from the measured "
            f"ones, at columns {differ}: {layout.heights}")

    def test_the_heights_are_symmetric_and_every_column_shares_one_centre_row(
            self):
        """No offsets: each column is centred on the well's own row."""
        layout = round_well_layout(1281)
        assert layout.heights == layout.heights[::-1]
        assert layout.centre == (20, -20)
        for column in range(layout.columns):
            top, bottom = layout.span(column)
            assert top + bottom == 2 * layout.centre[1], column
        assert layout.site(*layout.centre) is not None

    def test_the_layout_is_the_inequality_and_nothing_else(self):
        """Every grid position in and around the well, against the formula."""
        layout = round_well_layout(1281)
        a, radius = layout.half_tile, layout.radius
        centre_column, centre_row = layout.centre
        assert a > 0.5
        inside = 0
        for column in range(-3, layout.columns + 3):
            for row in range(centre_row - 25, centre_row + 26):
                kept = ((abs(column - centre_column) + a) ** 2
                        + (abs(row - centre_row) + a) ** 2 <= radius ** 2)
                inside += kept
                assert (layout.site(column, row) is not None) == kept, (
                    column, row)
        assert inside == 1281

    def test_the_centre_rule_cannot_produce_the_measured_heights_at_any_radius(
            self):
        """Why no radius search was ever going to find the phenotype well."""
        for thousandths in range(19000, 22001):
            layout = WellLayout(columns=41, radius=thousandths / 1000,
                                centre=(20, -20))
            assert layout.heights != MEASURED_1281, layout.radius
        assert round_well_layout(1281, half_tile=0.0).heights == (
            CENTRE_RULE_1281)

    @pytest.mark.parametrize("well,table", [
        ("A1", A1_FIELDS_A_COLUMN_OFF), ("A2", A2_FIELDS_A_COLUMN_OFF)])
    def test_the_fields_the_circle_put_a_column_off_are_where_the_plate_put_them(
            self, well, table):
        """The junction fields of the real plate, at their measured positions.

        A2's table was not used to measure anything: it is the held-out well.
        """
        wrong = _misplaced(round_well_layout(1281), table)
        assert not wrong, (
            f"well {well}: {len(wrong)} of {len(table)} junction fields are "
            f"not at their measured grid position: {wrong}")

    @pytest.mark.parametrize("table", [A1_FIELDS_A_COLUMN_OFF,
                                       A2_FIELDS_A_COLUMN_OFF])
    def test_the_centre_rule_puts_every_one_of_them_in_the_neighbouring_column(
            self, table):
        """The defect itself, kept visible: one column off, each of them."""
        circle = round_well_layout(1281, half_tile=0.0)
        for site, (column, _row) in table.items():
            assert abs(circle.position(site)[0] - column) == 1, site

    @pytest.mark.parametrize("name", sorted(MEASURED_HALF_TILES))
    def test_either_measured_half_tile_gives_both_acquisitions(self, name):
        """One rule, and neither acquisition's value moves the other well."""
        a = MEASURED_HALF_TILES[name]
        assert round_well_layout(333, half_tile=a).heights == HEIGHTS_333
        assert round_well_layout(1281, half_tile=a).heights == MEASURED_1281

    def test_the_default_half_tile_is_measured_and_inside_the_interval(self):
        """0.5857, the phenotype acquisition's, inside (0.5, 2.0).

        Across that whole open interval 333 fields give the confirmed
        heights and 1,281 give the measured ones, so the default does not
        have to be exact -- but it is a measurement and not a midpoint.
        """
        import inspect

        default = inspect.signature(round_well_layout).parameters[
            "half_tile"].default
        assert default == pytest.approx(MEASURED_HALF_TILES["phenotype 20X"],
                                        abs=5e-5)
        assert round_well_layout(1281).half_tile == default
        for a in (0.5001, 0.75, 1.0, 1.5, 1.9999):
            assert round_well_layout(333, half_tile=a).heights == HEIGHTS_333
            assert round_well_layout(1281, half_tile=a).heights == (
                MEASURED_1281), a
        for a in (0.0, 0.49, 2.01):
            try:
                heights = round_well_layout(1281, half_tile=a).heights
            except ValueError:
                continue
            assert heights != MEASURED_1281, a

    def test_a_radius_window_narrower_than_the_search_step_is_still_found(self):
        """At a = 0.5005 only 0.0001 of radius holds exactly 1,281 fields.

        A 0.05 step walks straight past a window that narrow, and "no round
        well holds this many" would then be a statement about the step. The
        count is monotone in the radius, so the window is bisected.
        """
        layout = round_well_layout(1281, half_tile=0.5005)
        assert layout.site_count == 1281
        assert layout.heights == MEASURED_1281

    def test_small_wells_keep_their_sites(self):
        """The stitch fixtures' wells: same fields, whatever the half tile."""
        for count, heights in ((21, [3, 5, 5, 5, 3]), (5, [1, 3, 1])):
            rule = round_well_layout(count)
            circle = round_well_layout(count, half_tile=0.0)
            assert rule.heights == heights
            assert rule.positions() == circle.positions()
            assert rule.pairs() == circle.pairs()
        with pytest.raises(ValueError, match="not a circle"):
            round_well_layout(4)

    def test_a_negative_half_tile_is_refused(self):
        """A tile cannot be smaller than nothing."""
        with pytest.raises(ValueError, match="half tile"):
            round_well_layout(1281, half_tile=-0.1)
        with pytest.raises(ValueError, match="half tile"):
            round_well_layout(1281, half_tile=float("nan"))

    def test_the_half_tile_is_part_of_the_layouts_identity(self):
        """The walk is cached per layout, so the term must be in the hash."""
        rule = WellLayout(columns=41, radius=20.904, centre=(20, -20),
                          half_tile=0.5857)
        circle = WellLayout(columns=41, radius=20.904, centre=(20, -20))
        assert hash(rule) == hash(WellLayout(columns=41, radius=20.904,
                                             centre=(20, -20),
                                             half_tile=0.5857))
        assert rule != circle
        assert rule.heights == MEASURED_1281
        assert rule.positions() != circle.positions()
        with pytest.raises(Exception):
            rule.half_tile = 0.0                               # type: ignore


class TestTheOffsetTableIsGone:
    """The rule replaces the table; the ledger says it must not be applied."""

    def test_the_row_offset_names_are_gone(self):
        """Nothing left for a caller to apply by mistake."""
        for name in ("MEASURED_PHENOTYPE_ROW_OFFSETS",
                     "MIN_FIELDS_FOR_A_COLUMN_VOTE",
                     "_solve_row_offsets", "_mode"):
            assert not hasattr(ops_layout, name), name
        assert "row_offsets" not in WellLayout.__dataclass_fields__
        assert not hasattr(WellLayout, "_shift")
