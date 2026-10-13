"""The A4 anchor window is sized to the seed's measured error, not to a tile's.

372, 2026-09-22, the full-plate run. Well A2's phenotype phase refused: 2 of
18 anchor candidates aligned inside a +/-1,600 px window, and a raster needs
three. Re-running the same 18 candidates at other window sizes, on the plate:

    well   +/-1,600 px   +/-800 px   +/-600 px
    A1        4 / 18      10 / 18      7 / 18
    A2     2 (1) / 18      9 / 18        --

The seed -- the fitted sequencing raster -- landed within 256 px of every
aligned centre, and a field aligned at two sizes landed within 0.8 px of
itself. So the wide window was not buying reach; it was adding sequencing
nuclei that are not under the field, among which the field's own have to be
found. These tests pin the two sides of that: the window still holds the
seed's error with room to spare, and it is no longer many fields wide.

The plate's geometry: a 2,960 px phenotype field at scale 0.25 is 740 px of
the sequencing well frame, on a 26,855 px canvas.
"""

from spacr import ops_engine

FIELD_PX = 740.0
CANVAS = (26855, 26865)
MEASURED_SEED_ERROR_PX = 256.0


def _window(seed):
    return ops_engine._anchor_window(seed, (FIELD_PX, FIELD_PX), CANVAS)


def test_a_seed_as_far_out_as_the_plate_measured_still_holds_the_field():
    """The true field lies inside the window with twice the measured error to spare."""
    true_centre = (13000.0, 13000.0)
    reach = 2.0 * MEASURED_SEED_ERROR_PX
    for dy, dx in ((reach, 0.0), (0.0, -reach), (reach / 1.5, reach / 1.5)):
        window = _window((true_centre[0] + dy, true_centre[1] + dx))
        assert window.top <= true_centre[0] - FIELD_PX / 2
        assert window.bottom >= true_centre[0] + FIELD_PX / 2
        assert window.left <= true_centre[1] - FIELD_PX / 2
        assert window.right >= true_centre[1] + FIELD_PX / 2


def test_the_window_is_not_many_fields_wide():
    """At +/-1,600 px the window was 28 fields of area and A2 refused; it is at most 12 now."""
    window = _window((13000.0, 13000.0))
    ratio = (window.height * window.width) / (FIELD_PX * FIELD_PX)
    assert ratio <= 12.0, (
        f"the anchor window is {ratio:.1f} fields of area; every extra field "
        "is sequencing nuclei the alignment has to reject")
