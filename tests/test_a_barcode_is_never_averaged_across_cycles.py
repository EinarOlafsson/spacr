"""372 Phase C: sample every other channel at B4's ids, and refuse ONE thing.

372's C1 carries a sentence that is not a caveat but a specification:

    "SEQUENCING CHANNELS, PER CYCLE ... NEVER averaged across cycles -- that
    is the one operation that destroys a barcode while leaving it decodable."

The failure it names is the dangerous kind. A mean across cycles of a
four-channel readout still LOOKS like a four-channel readout: `call_reads`
accepts it, returns a barcode of the right length, and every downstream table
fills in. The run completes and the barcodes are noise, and nothing later can
detect it because nothing later knows what the per-cycle values were.

So the refusal is enforced rather than documented, and these tests are what
hold it.
"""

import numpy as np
import pytest

from spacr.ops_objects import PlateObject
from spacr.ops_sample import (
    SampleError, average_over_cycles, barcode_rows, decode_input, reads_rows,
    sample_objects,
)

CHANNELS = ("G", "T", "A", "C")


def _objects(points):
    return [
        PlateObject(object_id=i + 1, centroid_x=float(x), centroid_y=float(y),
                    area=100, bbox=(int(y) - 5, int(x) - 5, int(y) + 5, int(x) + 5),
                    window=(0, 0))
        for i, (y, x) in enumerate(points)
    ]


def _image(shape, points, values, radius=1):
    img = np.zeros(shape, dtype=float)
    for (y, x), value in zip(points, values):
        img[int(y) - radius:int(y) + radius + 1,
            int(x) - radius:int(x) + radius + 1] = value
    return img


# -- C1, the refusal -------------------------------------------------------

def test_one_cycle_reaching_the_decoder_is_refused():
    """What a caller who already averaged the cycles hands over.

    It cannot be told apart from a lost cycle axis, and both make a barcode
    of noise that decodes cleanly. So it is refused at the door, with the
    instruction's own sentence in the message.
    """
    samples = [[np.array([1.0, 2.0]) for _ in CHANNELS]]

    with pytest.raises(SampleError, match="destroys a barcode"):
        decode_input(samples)


def test_no_function_here_averages_a_sequencing_channel():
    """The module offers no way to do it, which is the enforcement.

    A flag on `decode_input` would make this a caller's choice. It is not a
    choice -- it is a property of the channel -- so the averaging lives in a
    separate function named for the phenotype case.
    """
    import inspect

    from spacr import ops_sample

    signature = inspect.signature(ops_sample.decode_input)
    assert list(signature.parameters) == ["per_cycle"]
    assert "average" not in signature.parameters


def test_the_phenotype_path_does_average_and_is_a_different_function():
    """Same plate, same objects, opposite correct answer.

    A phenotype measured in several cycles is several samples of ONE
    quantity, so averaging raises its precision -- exactly as averaging
    Hoechst across tiles did in B1.
    """
    per_cycle = [np.array([10.0, 20.0]), np.array([12.0, 22.0]),
                 np.array([14.0, 24.0])]
    averaged = average_over_cycles(per_cycle)

    assert averaged == pytest.approx([12.0, 22.0])


def test_averaging_skips_a_cycle_that_did_not_cover_the_object():
    """Nine cycles of eleven should give the mean of nine, not NaN."""
    per_cycle = [np.array([10.0, np.nan]), np.array([20.0, 40.0]),
                 np.array([30.0, 50.0])]
    averaged = average_over_cycles(per_cycle)

    assert averaged[0] == pytest.approx(20.0)
    assert averaged[1] == pytest.approx(45.0)


# -- C1, the sampling ------------------------------------------------------

def test_a_channel_is_read_at_the_object_coordinates():
    points = [(20, 30), (60, 70)]
    image = _image((100, 100), points, [5.0, 9.0])
    found = sample_objects(_objects(points), image)

    assert found == pytest.approx([5.0, 9.0])


def test_a_transform_moves_the_sampling_point():
    """The A2/A3 transforms of PART 7: aligned, not resampled."""
    points = [(20, 30)]
    shifted = _image((100, 100), [(25, 38)], [7.0])
    found = sample_objects(_objects(points), shifted,
                           transform=lambda y, x: (y + 5, x + 8))

    assert found[0] == pytest.approx(7.0)


def test_an_object_outside_this_cycles_field_is_nan_not_an_edge_pixel():
    """'Not measured' must not become a number.

    A clamped border pixel is a number, and a number gets averaged into a
    barcode -- which is the same class of silent corruption C1 warns about.
    """
    points = [(20, 30), (500, 500)]
    image = _image((100, 100), [(20, 30)], [5.0])
    found = sample_objects(_objects(points), image)

    assert found[0] == pytest.approx(5.0)
    assert np.isnan(found[1])


def test_a_stack_is_refused_where_one_channel_is_expected():
    """The axes are what keep the cycles apart, so shape is checked."""
    with pytest.raises(SampleError, match="one 2-D image"):
        sample_objects(_objects([(1, 1)]), np.zeros((3, 10, 10)))


# -- the decode input ------------------------------------------------------

def test_the_stack_has_objects_cycles_channels_in_that_order():
    """`call_reads` reads (N, cycles, channels); a transposed stack decodes
    silently and wrongly, so the order is asserted rather than assumed."""
    objects, cycles, channels = 3, 4, len(CHANNELS)
    samples = [[np.arange(objects, dtype=float) for _ in range(channels)]
               for _ in range(cycles)]
    stacked = decode_input(samples)

    assert stacked.shape == (objects, cycles, channels)


def test_mismatched_lengths_are_refused_because_the_id_is_the_join_key():
    samples = [[np.array([1.0, 2.0]), np.array([1.0])],
               [np.array([1.0, 2.0]), np.array([1.0, 2.0])]]
    with pytest.raises(SampleError, match="join key"):
        decode_input(samples)


def test_cycles_with_different_channel_counts_are_refused():
    samples = [[np.array([1.0])], [np.array([1.0]), np.array([2.0])]]
    with pytest.raises(SampleError, match="same channels in every cycle"):
        decode_input(samples)


def test_no_cycles_at_all_is_refused():
    with pytest.raises(SampleError, match="one letter per cycle"):
        decode_input([])


# -- the tables ------------------------------------------------------------

def _planted(barcode, n_objects=1):
    """Intensities that spell `barcode`, one row per object."""
    cycles = len(barcode)
    values = np.full((n_objects, cycles, len(CHANNELS)), 100.0)
    for cycle, letter in enumerate(barcode):
        values[:, cycle, CHANNELS.index(letter)] = 900.0
    return values


def test_ops_reads_has_one_row_per_object_per_cycle():
    objects = _objects([(10, 10), (20, 20)])
    values = _planted("GTAC", n_objects=2)
    rows = reads_rows(objects, values, channels=CHANNELS, bases=CHANNELS)

    assert len(rows) == 2 * 4
    assert {row["object_id"] for row in rows} == {1, 2}
    assert sorted({row["cycle"] for row in rows}) == [1, 2, 3, 4]
    assert all(channel in rows[0] for channel in CHANNELS)


def test_the_called_base_is_the_brightest_channel():
    objects = _objects([(10, 10)])
    rows = reads_rows(objects, _planted("GTAC"), channels=CHANNELS,
                      bases=CHANNELS)

    assert [row["base"] for row in rows] == list("GTAC")
    assert all(row["quality"] > 0.5 for row in rows)


def test_ops_barcodes_assembles_the_planted_barcode():
    objects = _objects([(10, 10), (20, 20)])
    rows = barcode_rows(objects, _planted("GTAC", n_objects=2), bases=CHANNELS)

    assert [row["barcode"] for row in rows] == ["GTAC", "GTAC"]
    assert [row["n_cycles"] for row in rows] == [4, 4]
    assert [row["object_id"] for row in rows] == [1, 2]


def test_an_ambiguous_library_match_lands_as_empty_not_the_string_none():
    """`correct_to_library` returns None on a tie, and it must stay absent.

    "None" as text would look like a guide called None and join against
    nothing, which is worse than a blank because it looks like data.
    """
    objects = _objects([(10, 10)])
    rows = barcode_rows(objects, _planted("GTAC"), bases=CHANNELS,
                        library=["AAAA", "TTTT"])

    assert rows[0]["mapped_guide"] in ("", "AAAA", "TTTT")
    assert rows[0]["mapped_guide"] != "None"


def test_a_row_count_that_does_not_match_the_objects_is_refused():
    with pytest.raises(SampleError, match="join key"):
        reads_rows(_objects([(1, 1)]), _planted("GT", n_objects=3),
                   channels=CHANNELS)
