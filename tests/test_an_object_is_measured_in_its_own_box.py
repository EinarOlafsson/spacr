"""Per-object measurements are computed inside the object's own bounding box.

``spacr.measure`` reaches an object by comparing the whole field to its label,
once per object, so the intensity block cost O(objects x field) to look at
objects a few tens of pixels across. A field paid one full-field distance
transform per (cell, child) pair, one full-field dilation per object ring, and
one full-field comparison per object per colocalisation pair. Cropping each
iteration to the object's own bounding box is exact rather than approximate:
it selects the same pixels in the same C order, so every reduction over them
is unchanged to the last bit.

Both halves of that claim are held here, because either one alone is worth
nothing:

* the WORK tests watch what the expensive primitives are actually handed and
  fail the moment a whole field reaches one of them again;
* the VALUE tests re-implement the whole-field loop as an oracle inside this
  module and demand exact equality, dtype included, so a crop that quietly
  moved a number could not pass.

The three ways a crop of this shape goes wrong each have a test by name: a
radial window taken from the cell's box alone rather than its union with the
object's, which leaves an object larger than its cell with no boundary in the
window and every distance infinite; a ring window narrower than the dilation
that fills it; and a percentile batched into one call, which on a float32
image promotes the column to float64 and changes its last bits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.stats import pearsonr
from skimage.segmentation import find_boundaries

import spacr.measure as M

RING_CUT_POINTS = (5, 10, 25, 50, 75, 85, 95)
INTENSITY_PROPS = ["label", "mean_intensity", "max_intensity"]


# ---------------------------------------------------------------------------
# fixtures: a field big enough that "whole field" and "one object" differ a lot
# ---------------------------------------------------------------------------

SIDE = 256
OBJECT_SIDE = 12
OBJECT_CORNERS = ((30, 40), (60, 150), (140, 60), (180, 190), (110, 110))


def _object_field():
    """A 256-pixel-square field holding five 12-pixel-square objects."""
    mask = np.zeros((SIDE, SIDE), dtype=np.int32)
    for label, (row, col) in enumerate(OBJECT_CORNERS, start=1):
        mask[row:row + OBJECT_SIDE, col:col + OBJECT_SIDE] = label
    image = (np.random.default_rng(11).random((SIDE, SIDE)) * 4000).astype(np.uint16)
    return mask, image


def _nested_field():
    """Three cells, each holding two nuclei, in a 256-pixel-square field."""
    cell = np.zeros((SIDE, SIDE), dtype=np.int32)
    nucleus = np.zeros((SIDE, SIDE), dtype=np.int32)
    child = 0
    for label, (row, col) in enumerate(((20, 20), (20, 140), (140, 60)), start=1):
        cell[row:row + 64, col:col + 64] = label
        for offset in (8, 36):
            child += 1
            nucleus[row + offset:row + offset + 14,
                    col + offset:col + offset + 14] = child
    channels = (np.random.default_rng(12).random((SIDE, SIDE, 2)) * 4000).astype(np.uint16)
    return cell, nucleus, channels


class _WatchedImage(np.ndarray):
    """An intensity image that records the size of each array a mask selects from.

    Boolean indexing is the last step of every per-object read in
    ``measure.py``, so the size of the array it is applied to is exactly "how
    much of the field did this object touch".
    """

    def __array_finalize__(self, obj):
        self.selected_from = getattr(obj, "selected_from", None)

    def __getitem__(self, key):
        if (self.selected_from is not None and isinstance(key, np.ndarray)
                and key.dtype == bool):
            self.selected_from.append(self.size)
        return super().__getitem__(key)


def _watched(image, log):
    """``image`` as a :class:`_WatchedImage` recording into ``log``."""
    view = image.view(_WatchedImage)
    view.selected_from = log
    return view


def _assert_identical(got, expected, what):
    """Every value equal and every dtype equal, NaN counting as a match."""
    assert len(got) == len(expected), f"{what}: {len(got)} rows vs {len(expected)}"
    for index, (got_row, expected_row) in enumerate(zip(got, expected)):
        assert len(got_row) == len(expected_row), f"{what}[{index}]: width"
        for position, (a, b) in enumerate(zip(got_row, expected_row)):
            first, second = np.asarray(a), np.asarray(b)
            assert first.dtype == second.dtype, (
                f"{what}[{index}][{position}]: dtype {first.dtype} vs {second.dtype}")
            equal_nan = np.issubdtype(first.dtype, np.floating)
            assert np.array_equal(first, second, equal_nan=equal_nan), (
                f"{what}[{index}][{position}]: {a!r} vs {b!r}")


# ---------------------------------------------------------------------------
# oracles: the whole-field loops, written out so the fast paths have something
# independent to be equal to
# ---------------------------------------------------------------------------

def _outside_oracle(label_mask, image, distance=5, spacing=None):
    """The outside ring, computed over the whole field once per object."""
    stats = []
    ring_width = None if spacing is None else float(distance) * float(spacing[-1])
    for region in np.unique(label_mask)[1:]:
        region_mask = label_mask == region
        if spacing is None:
            dilated = binary_dilation(region_mask, iterations=distance)
        else:
            dilated = distance_transform_edt(~region_mask, sampling=spacing) <= ring_width
        values = image[dilated & ~region_mask]
        if values.size == 0:
            stats.append((region,) + (np.nan,) * 8)
        else:
            stats.append((region, np.mean(values),
                          *(np.percentile(values, p) for p in RING_CUT_POINTS)))
    return stats


def _periphery_oracle(label_mask, image):
    """The periphery ring, computed over the whole field once per object."""
    stats = []
    boundary = find_boundaries(label_mask)
    for region in np.unique(label_mask)[1:]:
        values = image[boundary & (label_mask == region)]
        if values.size == 0:
            stats.append((region,) + (np.nan,) * 8)
        else:
            stats.append((region, np.mean(values),
                          *(np.percentile(values, p) for p in RING_CUT_POINTS)))
    return stats


def _colocalisation_oracle(first, second, mask, thresholds):
    """Pearson and the M1/M2 pair, computed over the whole field per object."""
    rows = []
    for region in np.unique(mask)[1:]:
        selected = mask == region
        a, b = first[selected], second[selected]
        total_a, total_b = np.sum(a), np.sum(b)
        pearson = np.nan if len(a) < 2 else pearsonr(a, b)[0]
        row = {"label_correlation": region, "Pearson_correlation": pearson}
        for threshold in thresholds:
            overlap = ((a > np.percentile(a, threshold))
                       & (b > np.percentile(b, threshold)))
            row[f"M1_correlation_{threshold}"] = (
                np.sum(a[overlap]) / total_a if total_a > 0 else 0)
            row[f"M2_correlation_{threshold}"] = (
                np.sum(b[overlap]) / total_b if total_b > 0 else 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _radial_oracle(cell_mask, object_mask, channels, num_bins=6):
    """The radial profile, computed over the whole field once per pair."""
    out = {}
    for cell_label in np.unique(cell_mask)[1:]:
        cell_region = cell_mask == cell_label
        for object_label in np.unique(object_mask[cell_region]):
            if object_label == 0:
                continue
            boundary = find_boundaries(object_mask == object_label, mode="outer")
            distance_map = distance_transform_edt(~boundary)
            in_region = distance_map[cell_region]
            for channel in range(channels.shape[-1]):
                profile = np.full(num_bins, np.nan)
                plane = channels[..., channel]
                if in_region.size:
                    furthest = in_region.max()
                    if furthest <= 0:
                        profile[0] = plane[cell_region].mean()
                    else:
                        for index in range(num_bins):
                            low = index * (furthest / num_bins)
                            high = (index + 1) * (furthest / num_bins)
                            selected = cell_region & (distance_map >= low)
                            if index == num_bins - 1:
                                selected &= distance_map <= high
                            else:
                                selected &= distance_map < high
                            if selected.any():
                                profile[index] = plane[selected].mean()
                out[(cell_label, object_label, channel)] = profile
    return out


# ---------------------------------------------------------------------------
# the outside ring
# ---------------------------------------------------------------------------

def test_an_outside_ring_is_dilated_in_the_objects_own_box(monkeypatch):
    """The ring dilation sees the object's box grown by the ring width, not the field.

    ``_outside_intensity`` samples a five-pixel ring around objects that are
    tens of pixels across; dilating the whole field for each of them is the
    single largest piece of waste in the intensity block after the radial
    profile. The assertion is the exact box -- an object of side ``s`` with a
    ring of width ``d`` needs ``(s + 2d)`` pixels a side and nothing more --
    so it fails both if the crop is dropped and if it silently grows.
    """
    mask, image = _object_field()
    handed = []
    real_dilation = M.binary_dilation

    def recording(region_mask, **kwargs):
        handed.append(region_mask.shape)
        return real_dilation(region_mask, **kwargs)

    monkeypatch.setattr(M, "binary_dilation", recording)
    M._outside_intensity(mask, image, distance=5)

    assert len(handed) == len(OBJECT_CORNERS)
    expected = (OBJECT_SIDE + 10, OBJECT_SIDE + 10)
    assert set(handed) == {expected}
    assert expected[0] * expected[1] * len(handed) < mask.size


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_an_outside_ring_holds_the_values_the_whole_field_gives(dtype):
    """The cropped ring reports exactly what the whole-field ring reported.

    Guards the equality half of the crop: the ring deliberately includes
    neighbouring labels' pixels, so a window that merely looked "big enough"
    could drop a neighbour and change the mean without changing the geometry.

    Every dtype a merged stack or a preprocessing hook can produce is checked,
    values and dtypes both, because batching the seven percentile calls is
    what makes the ring cheap and the batched form of ``np.percentile``
    computes a float32 input in float64.
    """
    mask, image = _object_field()
    image = image.astype(dtype)
    _assert_identical(M._outside_intensity(mask, image, distance=5),
                      _outside_oracle(mask, image, distance=5),
                      "outside ring")


def test_an_outside_ring_keeps_its_values_on_a_spaced_volume():
    """The 3-D ring is the same sampled shell after cropping as before.

    In 3-D the ring comes from a spacing-sampled distance transform rather
    than from iterated dilation, so it needs a different per-axis margin --
    a shell 2.5 um wide reaches five voxels along a 0.5 um axis and two along
    a 2.0 um one. A single flat margin would clip the thin axis and lose the
    outer shell.
    """
    spacing = (2.0, 0.5, 0.5)
    mask = np.zeros((9, 48, 48), dtype=np.int32)
    mask[3:6, 10:16, 10:16] = 1
    mask[3:6, 30:36, 30:36] = 2
    image = (np.random.default_rng(13).random(mask.shape) * 500).astype(np.uint16)

    _assert_identical(M._outside_intensity(mask, image, distance=5, spacing=spacing),
                      _outside_oracle(mask, image, distance=5, spacing=spacing),
                      "3-D outside ring")


def test_a_ring_with_no_iteration_limit_is_measured_over_the_whole_field():
    """A non-positive ring width floods the field, and is measured that way.

    ``binary_dilation`` reads ``iterations < 1`` as "repeat until the result
    stops changing", so ``distance=0`` does not mean "no ring" -- it means
    "every pixel outside the object". A margin derived from the width would be
    zero and would crop that flood down to the object's own box, turning a
    whole-field statistic into an empty one, so no margin is applied at all.
    """
    mask, image = _object_field()
    stats = M._outside_intensity(mask, image, distance=0)

    assert len(stats) == len(OBJECT_CORNERS)
    _assert_identical(stats, _outside_oracle(mask, image, distance=0),
                      "flooded ring")
    outside_first = image[mask != 1]
    assert stats[0][1] == pytest.approx(float(np.mean(outside_first)), rel=1e-12)


def test_a_voxel_step_of_zero_leaves_the_ring_uncropped():
    """A spacing step of zero bounds no distance, so the whole volume is used.

    The per-axis margin is ``ring_width / spacing[axis]``; a zero step makes
    that unbounded, and guessing a finite margin would silently shrink the
    shell. The ring is then measured exactly as it would be with no cropping,
    which is what the whole-field oracle computes.
    """
    spacing = (1.0, 1.0, 0.0)
    mask = np.zeros((5, 24, 24), dtype=np.int32)
    mask[2, 10:13, 10:13] = 1
    image = (np.random.default_rng(14).random(mask.shape) * 500).astype(np.uint16)

    _assert_identical(M._outside_intensity(mask, image, distance=5, spacing=spacing),
                      _outside_oracle(mask, image, distance=5, spacing=spacing),
                      "unbounded ring")


# ---------------------------------------------------------------------------
# the periphery ring
# ---------------------------------------------------------------------------

def test_a_periphery_ring_reads_only_the_objects_own_box():
    """Each object's boundary pixels are gathered from its box, not the field.

    The boundary map itself is one whole-field pass and stays one; what used
    to repeat per object was the ``label_mask == region`` comparison and the
    boolean gather behind it.
    """
    mask, image = _object_field()
    selected_from = []
    M._periphery_intensity(mask, _watched(image, selected_from))

    assert len(selected_from) == len(OBJECT_CORNERS)
    assert max(selected_from) == OBJECT_SIDE * OBJECT_SIDE
    assert max(selected_from) < mask.size


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_a_periphery_ring_holds_the_values_the_whole_field_gives(dtype):
    """The cropped periphery ring reports exactly the whole-field values.

    Guards the equality half: the boundary is computed on the whole field and
    then indexed by the object's box, so an off-by-one window would drop the
    outermost boundary pixels and lower every percentile. The dtypes are
    compared as well as the values -- a float32 image batched carelessly comes
    back as float64 and its percentiles move in the last bits.
    """
    mask, image = _object_field()
    image = image.astype(dtype)
    _assert_identical(M._periphery_intensity(mask, image),
                      _periphery_oracle(mask, image),
                      "periphery ring")


# ---------------------------------------------------------------------------
# the colocalisation pair loop
# ---------------------------------------------------------------------------

def test_a_colocalisation_pair_reads_only_the_objects_own_box():
    """Both channels of a pair are gathered from the object's box.

    The pair loop runs once per (channel pair, mask, object), so a whole-field
    comparison here is multiplied by the square of the channel count.
    """
    mask, image = _object_field()
    other = (np.random.default_rng(15).random(mask.shape) * 4000).astype(np.uint16)
    selected_from = []
    M._calculate_correlation_object_level(
        _watched(image, selected_from), _watched(other, selected_from), mask,
        {"manders_thresholds": [85, 95]})

    assert selected_from, "no per-object gather was recorded"
    assert max(selected_from) == OBJECT_SIDE * OBJECT_SIDE
    assert max(selected_from) < mask.size


def test_a_colocalisation_pair_holds_the_values_the_whole_field_gives():
    """The cropped pair loop reports exactly the whole-field correlations.

    Pearson's r and the M1/M2 thresholds are order-sensitive floating-point
    reductions, so this asserts on exact equality: a window that gathered the
    same pixels in a different order would drift in the last bits.
    """
    mask, image = _object_field()
    other = (np.random.default_rng(16).random(mask.shape) * 4000).astype(np.uint16)
    thresholds = [15, 85, 95]

    got = M._calculate_correlation_object_level(
        image, other, mask, {"manders_thresholds": thresholds})
    expected = _colocalisation_oracle(image, other, mask, thresholds)

    assert list(got.columns) == list(expected.columns)
    for column in got.columns:
        assert got[column].to_numpy().dtype == expected[column].to_numpy().dtype
        assert np.array_equal(got[column].to_numpy(), expected[column].to_numpy(),
                              equal_nan=True), column


# ---------------------------------------------------------------------------
# the radial profile
# ---------------------------------------------------------------------------

def test_a_radial_profile_transforms_only_the_pairs_own_window(monkeypatch):
    """The distance transform sees one (cell, object) window, not the field.

    This is the most expensive primitive in the module: it ran once per
    (cell, child) pair over the whole field, so its cost scaled with the image
    area rather than with the objects in it.
    """
    cell, nucleus, channels = _nested_field()
    handed = []
    real_transform = M.distance_transform_edt

    def recording(image, **kwargs):
        handed.append(image.size)
        return real_transform(image, **kwargs)

    monkeypatch.setattr(M, "distance_transform_edt", recording)
    M._calculate_radial_distribution(cell, nucleus, channels, num_bins=6)

    assert len(handed) == 6                       # three cells, two nuclei each
    assert max(handed) <= (64 + 4) ** 2
    assert max(handed) < cell.size


def test_a_radial_profile_holds_the_values_the_whole_field_gives():
    """The cropped radial bins are exactly the whole-field bins.

    Each bin is a mean over a set of pixels, and a mean is order-sensitive, so
    the window has to select the same pixels in the same order rather than
    merely the same set.
    """
    cell, nucleus, channels = _nested_field()
    got = M._calculate_radial_distribution(cell, nucleus, channels, num_bins=6)
    expected = _radial_oracle(cell, nucleus, channels, num_bins=6)

    assert set(got) == set(expected)
    for key in expected:
        assert np.array_equal(got[key], expected[key], equal_nan=True), key


def test_a_radial_profile_window_covers_an_object_larger_than_its_cell():
    """A child selected by overlap can be bigger than its cell and still bin.

    Children are chosen by any overlap with the cell, so a small cell can sit
    inside a large object. A window taken from the cell's bounding box alone
    would then contain no boundary pixel of that object at all, every distance
    in it would come back infinite, and the bins would be nonsense -- which is
    why the window is the union of the two boxes.
    """
    cell = np.zeros((SIDE, SIDE), dtype=np.int32)
    cell[45:200, 45:200] = 1
    huge = np.zeros((SIDE, SIDE), dtype=np.int32)
    huge[40:220, 40:220] = 1
    channels = (np.random.default_rng(17).random((SIDE, SIDE, 1)) * 4000).astype(np.uint16)

    got = M._calculate_radial_distribution(cell, huge, channels, num_bins=6)

    assert list(got) == [(1, 1, 0)]
    # Every bin holds pixels: the cell reaches from just inside the object's
    # edge to its middle. With the object's boundary outside the window only
    # the outermost bin could fill, and it would hold the whole cell.
    assert np.isfinite(got[(1, 1, 0)]).all()
    expected = _radial_oracle(cell, huge, channels, num_bins=6)
    assert np.array_equal(got[(1, 1, 0)], expected[(1, 1, 0)], equal_nan=True)
    assert got[(1, 1, 0)][-1] != pytest.approx(float(channels[..., 0][cell == 1].mean()))


# ---------------------------------------------------------------------------
# the batched percentiles
# ---------------------------------------------------------------------------

def test_an_intensity_percentile_keeps_the_dtype_of_its_image():
    """Batching the percentile calls must not promote a float32 column.

    ``np.percentile`` given a sequence of cut points computes a float32 input
    in float64 and given one cut point computes it in float32, and the two
    disagree in the last bits. Batching all six cut points into one call is
    where the speedup comes from, so the column a float32 image produces is
    asserted to be float32 and to hold exactly the values one call per cut
    point gives -- a naive batching passes neither.
    """
    mask, image = _object_field()
    narrow = image.astype(np.float32)

    frame = M._extended_regionprops_table(mask, narrow, INTENSITY_PROPS)

    for cut_point in (5, 10, 25, 75, 85, 95):
        column = frame[f"percentile_{cut_point}"].to_numpy()
        assert column.dtype == np.float32, cut_point
        expected = np.array([np.percentile(narrow[mask == label], cut_point)
                             for label in frame["label"]], dtype=np.float32)
        assert np.array_equal(column, expected), cut_point
    assert frame["iqr_intensity"].to_numpy().dtype == np.float32


def test_the_intensity_percentiles_are_the_ones_computed_one_at_a_time():
    """A uint16 image gets the batched call and the same numbers as before.

    Integers reach float64 whichever form of ``np.percentile`` is used, so this
    is the path that actually gets batched in a measure run -- the merged
    stacks are 8- or 16-bit. The interquartile range is checked too because it
    is the other pair of calls that was collapsed into one.
    """
    mask, image = _object_field()

    frame = M._extended_regionprops_table(mask, image, INTENSITY_PROPS)

    for cut_point in (5, 10, 25, 75, 85, 95):
        column = frame[f"percentile_{cut_point}"].to_numpy()
        assert column.dtype == np.float64, cut_point
        expected = np.array([np.percentile(image[mask == label], cut_point)
                             for label in frame["label"]])
        assert np.array_equal(column, expected), cut_point

    expected_iqr = np.array([
        np.percentile(image[mask == label], 75) - np.percentile(image[mask == label], 25)
        for label in frame["label"]])
    assert np.array_equal(frame["iqr_intensity"].to_numpy(), expected_iqr)


def test_a_channels_field_percentiles_can_be_supplied_once_for_every_mask():
    """``frac_high90`` / ``frac_low10`` cut on the field, so the cut is per channel.

    The two references depend only on the intensity image, but the intensity
    block measures five masks against every channel, so computing them inside
    the per-mask call re-ravelled and re-sorted the same channel five times.
    Supplying them must give the identical frame, and must actually be used
    rather than ignored -- a reference above every pixel leaves nothing bright.
    """
    mask, image = _object_field()

    computed_here = M._extended_regionprops_table(mask, image, INTENSITY_PROPS)
    supplied = M._extended_regionprops_table(
        mask, image, INTENSITY_PROPS,
        field_percentiles=M._field_reference_percentiles(image))

    for column in ("frac_high90", "frac_low10"):
        assert np.array_equal(computed_here[column].to_numpy(),
                              supplied[column].to_numpy(), equal_nan=True), column

    unreachable = M._extended_regionprops_table(
        mask, image, INTENSITY_PROPS,
        field_percentiles=(float(image.max()) + 1.0, -1.0))
    assert (unreachable["frac_high90"].to_numpy() == 0).all()
    assert (unreachable["frac_low10"].to_numpy() == 0).all()


# ---------------------------------------------------------------------------
# masks the box list cannot describe
# ---------------------------------------------------------------------------

def test_a_non_integral_label_mask_is_still_measured():
    """Labels that are not whole numbers get no boxes and the whole field.

    ``scipy.ndimage.find_objects`` indexes boxes by integer label, so a mask
    labelled 1.5 has no box list at all. Cropping is an optimisation and never
    a filter, so such a mask is measured over the whole field and reports the
    same objects it always did -- and it must not be keyed into a box list by
    truncation either, which would hand object 1.5 the box of object 1.
    """
    mask = np.zeros((64, 64))
    mask[10:20, 10:20] = 1.5
    mask[30:40, 30:40] = 2.5
    image = (np.random.default_rng(18).random(mask.shape) * 500).astype(np.uint16)

    assert M._label_bounding_boxes(mask) == {}

    stats = M._outside_intensity(mask, image, distance=3)
    assert [row[0] for row in stats] == [1.5, 2.5]
    _assert_identical(stats, _outside_oracle(mask, image, distance=3),
                      "non-integral labels")


def test_a_label_mask_carrying_nan_still_reports_a_row_per_label():
    """A NaN in a float label mask is an empty object, not a crash.

    ``np.unique`` sorts NaN to the end and reports it as a label, and
    ``mask == nan`` selects nothing, so every per-object loop has always
    reported it as an empty object with NaN statistics. A NaN cannot be a
    dictionary key either, so the box lookup has to answer before it converts
    the label rather than after.
    """
    mask = np.zeros((48, 48))
    mask[10:20, 10:20] = 1.0
    mask[30:40, 30:40] = np.nan
    image = (np.random.default_rng(20).random(mask.shape) * 500).astype(np.uint16)

    assert M._label_bounding_boxes(mask) == {}

    stats = M._periphery_intensity(mask, image)
    assert len(stats) == 2
    assert np.isnan(stats[-1][0]) and np.isnan(stats[-1][1])
    _assert_identical(stats, _periphery_oracle(mask, image), "NaN label")

    rings = M._outside_intensity(mask, image, distance=3)
    assert len(rings) == 2
    assert np.isnan(rings[-1][0]) and np.isnan(rings[-1][1])
    _assert_identical(rings, _outside_oracle(mask, image, distance=3), "NaN label")


def test_a_sparsely_numbered_mask_is_still_measured():
    """A mask numbered far above its voxel count is measured without boxes.

    ``find_objects`` walks every label from one to the maximum, so a mask
    carrying label 10,000,000 in a 64-pixel-square field would allocate ten
    million slots to describe two objects. Above one label per voxel the box
    list costs more than the crops save, so it is not built.
    """
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[10:20, 10:20] = 10_000_000
    mask[30:40, 30:40] = 20_000_000
    image = (np.random.default_rng(19).random(mask.shape) * 500).astype(np.uint16)

    assert M._label_bounding_boxes(mask) == {}

    stats = M._periphery_intensity(mask, image)
    assert [row[0] for row in stats] == [10_000_000, 20_000_000]
    _assert_identical(stats, _periphery_oracle(mask, image), "sparse labels")


# ---------------------------------------------------------------------------
# the focus score
# ---------------------------------------------------------------------------

BLUR_SETTINGS = {
    "radial_dist": False,
    "calculate_correlation": False,
    "homogeneity": False,
    "homogeneity_distances": [8],
    "manders_thresholds": [85],
    "distance_gaussian_sigma": 0,
    "cell_mask_dim": None,
    "nucleus_mask_dim": None,
    "pathogen_mask_dim": None,
}


def test_a_blur_score_is_measured_on_the_objects_own_patch(monkeypatch):
    """The focus score is handed the object's patch, not a whole-field mask.

    ``_estimate_blur`` already cuts the object's bounding box grown by one
    pixel out of whatever it is given, so building a whole-field boolean for
    it and then letting it find the box again cost one pass over the field per
    object per channel -- and it is called for every object of every mask, so
    it is the last of the per-object whole-field passes in the intensity
    block.
    """
    cell, _nucleus, channels = _nested_field()
    empty = np.zeros_like(cell)
    handed = []
    real_blur = M._estimate_blur

    def recording(image, mask=None):
        handed.append(None if mask is None else mask.shape)
        return real_blur(image, mask=mask)

    monkeypatch.setattr(M, "_estimate_blur", recording)
    M._intensity_measurements(cell, empty, empty, empty, empty, channels,
                              dict(BLUR_SETTINGS), periphery=False, outside=False)

    assert handed, "no focus score was computed"
    assert set(handed) == {(66, 66)}          # a 64-pixel cell grown by one
    assert 66 * 66 < cell.size


def test_a_blur_score_holds_the_value_the_whole_field_mask_gives():
    """The patch scores exactly what a whole-field object mask scored.

    The Laplacian is taken on the RAW patch -- out-of-object pixels inside the
    box are deliberately not zero-filled -- so the window handed in has to be
    the same one pixel wider than the object, no more and no less. One pixel
    either way changes which neighbours the kernel reads and moves the score.
    """
    cell, _nucleus, channels = _nested_field()
    empty = np.zeros_like(cell)

    frames = M._intensity_measurements(
        cell, empty, empty, empty, empty, channels,
        dict(BLUR_SETTINGS), periphery=False, outside=False)
    measured = frames[0]

    # One 'label' column arrives per channel, so read the labels from the mask.
    labels = np.unique(cell)[1:]
    assert len(labels) == 3
    for channel in range(channels.shape[-1]):
        column = measured[f"cell_channel_{channel}_blur"].to_numpy()
        expected = np.array([
            M._estimate_blur(channels[..., channel], mask=(cell == label))
            for label in labels])
        assert column.dtype == expected.dtype
        assert np.array_equal(column, expected, equal_nan=True), channel
