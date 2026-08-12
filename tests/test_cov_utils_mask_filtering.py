"""CPU coverage for the mask-filtering block of ``spacr.utils``.

Covers ``remove_intensity_objects``, ``_filter_closest_to_stat``,
``_find_similar_sized_images``, ``_relabel_parent_with_child_labels``,
``_exclude_objects``, ``_merge_overlapping_objects``, ``_filter_object``,
``_filter_cp_masks``, ``_object_filter``, ``_get_regex``, ``_run_test_mode``
and ``_choose_model``.

Everything is synthetic, tiny and offline: the Cellpose constructors used by
``_choose_model`` are monkeypatched with recording doubles so no weights are
downloaded and no GPU is touched.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

import spacr.utils as U


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _two_object_mask(shape=(32, 32)):
    """Label mask with a small (16 px) object 1 and a big (100 px) object 2."""
    m = np.zeros(shape, dtype=np.int32)
    m[2:6, 2:6] = 1        # 16 px
    m[10:20, 10:20] = 2    # 100 px
    return m


def _intensity_batch_image(mask, values, shape=(32, 32), channels=2):
    """(H, W, C) float32 image whose channel 1 carries per-label ``values``."""
    img = np.zeros((*shape, channels), dtype=np.float32)
    for lbl, val in values.items():
        img[..., 1][mask == lbl] = val
    return img


# ---------------------------------------------------------------------------
# remove_intensity_objects
# ---------------------------------------------------------------------------

def test_remove_intensity_objects_low_mode_drops_dim_objects():
    mask = _two_object_mask()
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask == 1] = 10.0
    image[mask == 2] = 200.0

    out = U.remove_intensity_objects(image, mask.copy(), intensity_threshold=100.0,
                                     mode="low")
    assert set(np.unique(out).tolist()) == {0, 2}
    # the surviving object keeps its exact footprint
    assert int((out == 2).sum()) == 100


def test_remove_intensity_objects_high_mode_drops_bright_objects():
    mask = _two_object_mask()
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask == 1] = 10.0
    image[mask == 2] = 200.0

    out = U.remove_intensity_objects(image, mask.copy(), intensity_threshold=100.0,
                                     mode="high")
    assert set(np.unique(out).tolist()) == {0, 1}
    assert int((out == 1).sum()) == 16


def test_remove_intensity_objects_keeps_everything_when_threshold_is_extreme():
    mask = _two_object_mask()
    image = np.zeros(mask.shape, dtype=np.float32)
    image[mask > 0] = 50.0
    out = U.remove_intensity_objects(image, mask.copy(), intensity_threshold=0.0,
                                     mode="low")
    assert set(np.unique(out).tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# _filter_closest_to_stat
# ---------------------------------------------------------------------------

def test_filter_closest_to_stat_uses_mean_by_default():
    # mean of [0, 10, 11, 12, 67] == 20 -> the two closest rows are 11 and 12.
    df = pd.DataFrame({"v": [0.0, 10.0, 11.0, 12.0, 67.0],
                       "tag": list("abcde")})
    out = U._filter_closest_to_stat(df.copy(), "v", n_rows=2)
    assert "diff" not in out.columns
    assert sorted(out["v"].tolist()) == [11.0, 12.0]
    assert sorted(out["tag"].tolist()) == ["c", "d"]


def test_filter_closest_to_stat_median_differs_from_mean():
    df = pd.DataFrame({"v": [0.0, 10.0, 11.0, 12.0, 67.0]})
    mean_rows = set(U._filter_closest_to_stat(df.copy(), "v", n_rows=1)["v"])
    median_rows = set(U._filter_closest_to_stat(df.copy(), "v", n_rows=1,
                                                use_median=True)["v"])
    assert mean_rows == {12.0}      # closest to mean 20.0
    assert median_rows == {11.0}    # median is exactly 11.0


# ---------------------------------------------------------------------------
# _find_similar_sized_images
# ---------------------------------------------------------------------------

def test_find_similar_sized_images_grayscale_and_fully_padded(tmp_path):
    import cv2

    gray_paths = []
    for i in range(3):
        img = np.zeros((24, 24), dtype=np.uint8)   # 2-D -> grayscale branch
        img[6:14, 6:14] = 180                      # identical 8x8 crop
        p = str(tmp_path / f"gray{i}.png")
        assert cv2.imwrite(p, img)
        gray_paths.append(p)

    # completely padded (all-zero) image -> `coords.size == 0` -> skipped
    blank = str(tmp_path / "blank.png")
    assert cv2.imwrite(blank, np.zeros((24, 24), dtype=np.uint8))

    # one odd-sized grayscale image so the group of 3 is strictly the largest
    odd = np.zeros((24, 24), dtype=np.uint8)
    odd[2:20, 2:5] = 180
    p_odd = str(tmp_path / "odd.png")
    assert cv2.imwrite(p_odd, odd)

    group = U._find_similar_sized_images(gray_paths + [blank, p_odd])
    assert set(group) == set(gray_paths)
    assert blank not in group


def test_find_similar_sized_images_colour_images_use_per_pixel_any(tmp_path):
    import cv2

    colour_paths = []
    for i in range(3):
        img = np.zeros((24, 24, 3), dtype=np.uint8)   # 3-D -> colour branch
        img[4:16, 4:10, i % 3] = 200                  # identical 12x6 crop
        p = str(tmp_path / f"rgb{i}.png")
        assert cv2.imwrite(p, img)
        colour_paths.append(p)

    odd = np.zeros((24, 24, 3), dtype=np.uint8)
    odd[2:6, 2:22] = 200
    p_odd = str(tmp_path / "rgb_odd.png")
    assert cv2.imwrite(p_odd, odd)

    group = U._find_similar_sized_images(colour_paths + [p_odd])
    assert set(group) == set(colour_paths)


def test_find_similar_sized_images_raises_when_nothing_is_readable(tmp_path):
    # cv2.imread returns None for every path -> the size map stays empty and
    # max() over an empty sequence raises.
    missing = [str(tmp_path / "does_not_exist.png")]
    with pytest.raises(ValueError):
        U._find_similar_sized_images(missing)


# ---------------------------------------------------------------------------
# _relabel_parent_with_child_labels
# ---------------------------------------------------------------------------

def test_relabel_parent_takes_child_label_for_single_child():
    parent = np.zeros((24, 24), dtype=np.int32)
    parent[4:16, 4:16] = 1
    child = np.zeros((24, 24), dtype=np.int32)
    child[6:10, 6:10] = 5

    new_parent, new_child = U._relabel_parent_with_child_labels(parent.copy(),
                                                                child.copy())
    assert set(np.unique(new_parent).tolist()) == {0, 5}
    # parent footprint is preserved, only the label changed
    assert int((new_parent == 5).sum()) == int((parent == 1).sum())
    assert set(np.unique(new_child).tolist()) == {0, 5}


def test_relabel_parent_standardizes_multiple_children_to_first_label():
    """A parent holding two children collapses both children to the first id."""
    parent = np.zeros((24, 24), dtype=np.int32)
    parent[4:20, 4:20] = 1
    child = np.zeros((24, 24), dtype=np.int32)
    child[6:9, 6:9] = 3
    child[14:17, 14:17] = 7

    new_parent, new_child = U._relabel_parent_with_child_labels(parent.copy(),
                                                                child.copy())
    # the parent ends up carrying the LAST overlapping child label ...
    assert set(np.unique(new_parent).tolist()) == {0, 7}
    # ... while both children are standardized onto the FIRST child label.
    assert set(np.unique(new_child).tolist()) == {0, 3}
    assert int((new_child == 3).sum()) == 9 + 9


# ---------------------------------------------------------------------------
# _exclude_objects
# ---------------------------------------------------------------------------

def test_exclude_objects_drops_cells_without_a_nucleus():
    cell = np.zeros((32, 32), dtype=np.int32)
    cell[2:12, 2:12] = 1        # has a nucleus
    cell[18:28, 18:28] = 2      # no nucleus -> dropped
    nucleus = np.zeros_like(cell)
    nucleus[4:8, 4:8] = 1
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    pathogen = np.zeros_like(cell)
    pathogen[20:22, 20:22] = 1  # only inside the doomed cell

    kept, nuc, pat, cyt = U._exclude_objects(cell.copy(), nucleus.copy(),
                                             pathogen.copy(), cytoplasm.copy(),
                                             uninfected=True)
    assert set(np.unique(kept).tolist()) == {0, 1}
    # objects outside the kept cells are wiped
    assert not pat.any()
    assert nuc.any() and cyt.any()


def test_exclude_objects_infected_only_requires_a_pathogen():
    cell = np.zeros((32, 32), dtype=np.int32)
    cell[2:12, 2:12] = 1
    nucleus = np.zeros_like(cell)
    nucleus[4:8, 4:8] = 1
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    pathogen = np.zeros_like(cell)   # no pathogen anywhere

    kept, nuc, pat, cyt = U._exclude_objects(cell.copy(), nucleus.copy(),
                                             pathogen.copy(), cytoplasm.copy(),
                                             uninfected=False)
    assert not kept.any()
    assert not nuc.any() and not cyt.any() and not pat.any()


def test_exclude_objects_infected_only_keeps_fully_populated_cells():
    cell = np.zeros((32, 32), dtype=np.int32)
    cell[2:12, 2:12] = 1        # nucleus + cytoplasm + pathogen -> kept
    cell[18:28, 18:28] = 2      # nucleus + cytoplasm, no pathogen -> dropped
    nucleus = np.zeros_like(cell)
    nucleus[4:8, 4:8] = 1
    nucleus[20:24, 20:24] = 2
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    pathogen = np.zeros_like(cell)
    pathogen[9:11, 9:11] = 1

    kept, nuc, pat, cyt = U._exclude_objects(cell.copy(), nucleus.copy(),
                                             pathogen.copy(), cytoplasm.copy(),
                                             uninfected=False)
    assert set(np.unique(kept).tolist()) == {0, 1}
    assert set(np.unique(nuc).tolist()) == {0, 1}
    assert int((pat > 0).sum()) == 4


# ---------------------------------------------------------------------------
# _merge_overlapping_objects
# ---------------------------------------------------------------------------

def test_merge_overlapping_objects_trims_mask1_when_one_label_dominates():
    mask1 = np.zeros((20, 20), dtype=np.int32)
    mask1[2:12, 2:12] = 1                 # 100 px

    mask2 = np.zeros((20, 20), dtype=np.int32)
    mask2[2:12, 2:12] = 1                 # 95 px of overlap
    mask2[11, 2:7] = 2                    # 5 px of overlap -> 5 %

    out1, out2 = U._merge_overlapping_objects(mask1.copy(), mask2.copy())
    # dominant label >= 90 % -> the minority region is cut out of mask1
    assert int((out1 > 0).sum()) == 95
    assert not out1[11, 2:7].any()
    # mask2 is untouched in this branch
    assert set(np.unique(out2).tolist()) == {0, 1, 2}


def test_merge_overlapping_objects_merges_mask2_labels_when_split_is_even():
    mask1 = np.zeros((20, 20), dtype=np.int32)
    mask1[2:12, 2:12] = 1                 # 100 px

    mask2 = np.zeros((20, 20), dtype=np.int32)
    mask2[2:8, 2:12] = 1                  # 60 %
    mask2[8:12, 2:12] = 2                 # 40 %

    out1, out2 = U._merge_overlapping_objects(mask1.copy(), mask2.copy())
    # no dominant label -> mask2 labels are merged onto the first one
    assert set(np.unique(out2).tolist()) == {0, 1}
    assert int((out2 == 1).sum()) == 100
    # mask1 is left intact in this branch
    assert int((out1 > 0).sum()) == 100


def test_merge_overlapping_objects_no_op_for_single_overlap():
    mask1 = np.zeros((16, 16), dtype=np.int32)
    mask1[2:10, 2:10] = 1
    mask2 = np.zeros((16, 16), dtype=np.int32)
    mask2[2:10, 2:10] = 3

    out1, out2 = U._merge_overlapping_objects(mask1.copy(), mask2.copy())
    assert np.array_equal(out1, mask1)
    assert np.array_equal(out2, mask2)


# ---------------------------------------------------------------------------
# _filter_object
# ---------------------------------------------------------------------------

def test_filter_object_removes_objects_under_min_pixel_count():
    mask = _two_object_mask()
    out = U._filter_object(mask.copy(), min_value=20)
    assert set(np.unique(out).tolist()) == {0, 2}


# ---------------------------------------------------------------------------
# _filter_cp_masks
# ---------------------------------------------------------------------------

def _cp_inputs(masks, images=None, flows=None):
    """Wrap masks/images into the (masks, flows, batch) shape _filter_cp_masks wants."""
    n = len(masks)
    if images is None:
        images = [np.zeros((*masks[0].shape, 2), dtype=np.float32) for _ in range(n)]
    if flows is None:
        flows = [np.zeros(masks[0].shape, dtype=np.float32) for _ in range(n)]
    return masks, [flows], images


def test_filter_cp_masks_all_filters_disabled_returns_masks_unchanged():
    masks = [_two_object_mask(), _two_object_mask() * 1]
    m, f, b = _cp_inputs(masks)
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=False,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=False, merge=False,
                             batch=b, plot=False, figuresize=2)
    assert len(out) == 2
    for got, want in zip(out, masks):
        assert np.array_equal(got, want)


def test_filter_cp_masks_size_filter_drops_small_and_large_objects():
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[1:3, 1:3] = 1        # 4 px -> too small
    mask[6:12, 6:12] = 2      # 36 px -> kept
    mask[16:36, 16:36] = 3    # 400 px -> too large
    m, f, b = _cp_inputs([mask])
    out = U._filter_cp_masks(m, f, filter_size=True, filter_intensity=False,
                             minimum_size=10, maximum_size=200,
                             remove_border_objects=False, merge=False,
                             batch=b, plot=False, figuresize=2)
    assert set(np.unique(out[0]).tolist()) == {0, 2}
    assert int((out[0] == 2).sum()) == 36


def test_filter_cp_masks_merge_collapses_touching_objects():
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[4:20, 4:20] = 1
    mask[8:11, 20:23] = 2     # small satellite -> merged at threshold 0.66
    m, f, b = _cp_inputs([mask])
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=False,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=False, merge=True,
                             batch=b, plot=False, figuresize=2)
    assert U.mask_object_count(out[0]) == 1
    assert int((out[0] > 0).sum()) == 16 * 16 + 9


def test_filter_cp_masks_intensity_filter_keeps_the_bright_cluster():
    mask = _two_object_mask()
    image = _intensity_batch_image(mask, {1: 5.0, 2: 400.0})
    m, f, b = _cp_inputs([mask], images=[image])
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=True,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=False, merge=False,
                             batch=b, plot=False, figuresize=2)
    assert set(np.unique(out[0]).tolist()) == {0, 2}


def test_filter_cp_masks_intensity_filter_noop_when_clusters_are_close():
    mask = _two_object_mask()
    # centroid distance 0.1 < the 0.25 threshold -> clusters judged identical
    image = _intensity_batch_image(mask, {1: 5.0, 2: 5.1})
    m, f, b = _cp_inputs([mask], images=[image])
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=True,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=False, merge=False,
                             batch=b, plot=False, figuresize=2)
    assert set(np.unique(out[0]).tolist()) == {0, 1, 2}


def test_filter_cp_masks_intensity_filter_skips_kmeans_for_single_object():
    mask = np.zeros((32, 32), dtype=np.int32)
    mask[10:20, 10:20] = 4
    image = _intensity_batch_image(mask, {4: 123.0})
    m, f, b = _cp_inputs([mask], images=[image])
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=True,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=False, merge=False,
                             batch=b, plot=False, figuresize=2)
    # fewer than 2 samples -> KMeans is skipped and nothing is removed
    assert set(np.unique(out[0]).tolist()) == {0, 4}


def test_filter_cp_masks_removes_border_objects():
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[0:4, 0:4] = 1        # touches the border
    mask[10:16, 10:16] = 2
    m, f, b = _cp_inputs([mask])
    out = U._filter_cp_masks(m, f, filter_size=False, filter_intensity=False,
                             minimum_size=0, maximum_size=10**6,
                             remove_border_objects=True, merge=False,
                             batch=b, plot=False, figuresize=2)
    assert set(np.unique(out[0]).tolist()) == {0, 2}


def test_filter_cp_masks_plot_path_reports_counts_for_first_image(monkeypatch,
                                                                  capsys):
    """plot=True renders one figure per filtration stage, for idx == 0 only."""
    import matplotlib.pyplot as plt

    shows = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shows.append(1))

    def _mask():
        m = np.zeros((32, 32), dtype=np.int32)
        m[2:8, 2:8] = 1          # 36 px, dim
        m[12:22, 12:22] = 2      # 100 px, bright
        m[0:3, 28:31] = 3        # border object
        return m

    masks = [_mask(), _mask()]
    images = [_intensity_batch_image(mk, {1: 5.0, 2: 400.0, 3: 350.0})
              for mk in masks]
    m, f, b = _cp_inputs(masks, images=images)
    out = U._filter_cp_masks(m, f, filter_size=True, filter_intensity=True,
                             minimum_size=10, maximum_size=10**6,
                             remove_border_objects=True, merge=True,
                             batch=b, plot=True, figuresize=2)
    text = capsys.readouterr().out
    assert "Number of objects before filtration" in text
    assert "after merging adjacent objects" in text
    assert "after size filtration" in text
    assert "potential intensity clustering" in text
    assert "after removing border objects" in text
    # one figure per stage, and only for the first image in the batch
    assert len(shows) == 5
    assert len(out) == 2
    # the dim object and the border object are gone from both masks
    for got in out:
        assert set(np.unique(got).tolist()) == {0, 2}


# ---------------------------------------------------------------------------
# _object_filter
# ---------------------------------------------------------------------------

def test_object_filter_applies_size_and_intensity_bounds():
    df = pd.DataFrame({
        "cell_area": [5, 50, 500, 5000],
        "cell_channel_1_mean_intensity": [1, 100, 100, 100],
    })
    out = U._object_filter(df, object_type="cell", size_range=[10, 1000],
                           intensity_range=[10, 1000], mask_chans=[0, 1],
                           mask_chan=1)
    assert out["cell_area"].tolist() == [50, 500]


def test_object_filter_ignores_non_int_bounds_and_none_ranges():
    df = pd.DataFrame({
        "cell_area": [5, 50, 500],
        "cell_channel_1_mean_intensity": [1, 100, 100],
    })
    # floats are not ints -> both bounds are skipped
    untouched = U._object_filter(df, object_type="cell", size_range=[1.5, 2.5],
                                 intensity_range=None, mask_chans=[0, 1],
                                 mask_chan=1)
    assert len(untouched) == 3
    # a non-list size_range is ignored too
    untouched2 = U._object_filter(df, object_type="cell", size_range=(10, 100),
                                  intensity_range=(10, 100), mask_chans=[0, 1],
                                  mask_chan=1)
    assert len(untouched2) == 3


# ---------------------------------------------------------------------------
# _get_regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metadata_type,filename", [
    ("cellvoyager", "plate1_A01_T0001F001L01A01Z01C02.tif"),
    ("cq1", "W1F001T0001Z01C2.tif"),
    ("auto", "plate1_A01_T0001F001L01C02.tif"),
])
def test_get_regex_builtin_patterns_match_real_filenames(metadata_type, filename):
    rx = re.compile(U._get_regex(metadata_type, "tif"))
    match = rx.match(filename)
    assert match is not None
    assert match.group("wellID")
    assert match.group("chanID")


def test_get_regex_custom_wraps_the_user_pattern():
    rx = U._get_regex("custom", "tif", custom_regex=r"(?P<wellID>[A-H]\d{2})")
    assert rx == r"((?P<wellID>[A-H]\d{2})).tif"
    assert re.compile(rx).match("A01.tif") is not None


def test_get_regex_defaults_to_tif_when_img_format_is_none():
    rx = U._get_regex("cellvoyager", None)
    assert re.compile(rx).match("plate1_A01_T0001F001L01A01Z01C02.tif") is not None


# ---------------------------------------------------------------------------
# _run_test_mode
# ---------------------------------------------------------------------------

def test_run_test_mode_copies_selected_sets(yokogawa_cellvoyager_dir):
    src = str(yokogawa_cellvoyager_dir["src"])
    regex = U._get_regex("cellvoyager", "tif")
    out = U._run_test_mode(src, regex, timelapse=False, test_images=2,
                           random_test=True)
    assert out == os.path.join(src, "test")
    copied = sorted(os.listdir(out))
    # 2 sets x 2 channels
    assert len(copied) == 4
    # every copied file belongs to one of exactly 2 (plate, well, field) sets
    rx = re.compile(regex)
    sets = {(m.group("plateID"), m.group("wellID"), m.group("fieldID"))
            for m in (rx.match(f) for f in copied)}
    assert len(sets) == 2


def test_run_test_mode_timelapse_forces_a_single_set(yokogawa_cellvoyager_dir):
    src = str(yokogawa_cellvoyager_dir["src"])
    regex = U._get_regex("cellvoyager", "tif")
    out = U._run_test_mode(src, regex, timelapse=True, test_images=10)
    copied = os.listdir(out)
    assert len(copied) == 2      # one field, both channels


def test_run_test_mode_without_plateid_group(yokogawa_cq1_dir):
    """The cq1 regex has no plateID group -> the folder basename is used."""
    src = str(yokogawa_cq1_dir["src"])
    regex = U._get_regex("cq1", "tif")
    out = U._run_test_mode(src, regex, timelapse=False, test_images=10,
                           random_test=False)
    copied = sorted(os.listdir(out))
    # all 2 wells x 2 fields x 2 channels are selected when test_images is large
    assert len(copied) == 8
    assert all(f.startswith("W") and f.endswith(".tif") for f in copied)


def test_run_test_mode_reads_from_orig_subfolder(tmp_path):
    src = tmp_path / "plate"
    orig = src / "orig"
    orig.mkdir(parents=True)
    names = [f"plate1_A01_T0001F00{i}L01A01Z01C01.tif" for i in (1, 2)]
    for n in names:
        (orig / n).write_bytes(b"not-a-real-tif")
    # a decoy in the parent that must NOT be picked up
    (src / "plate1_B02_T0001F009L01A01Z01C01.tif").write_bytes(b"decoy")

    regex = U._get_regex("cellvoyager", "tif")
    out = U._run_test_mode(str(src), regex, test_images=10)
    assert out == os.path.join(str(src), "test")
    copied = sorted(os.listdir(out))
    assert copied == sorted(names)


def test_run_test_mode_with_no_matching_files(tmp_path):
    src = tmp_path / "empty_plate"
    src.mkdir()
    (src / "README.txt").write_text("nothing to see")
    out = U._run_test_mode(str(src), U._get_regex("cellvoyager", "tif"))
    assert os.path.isdir(out)
    assert os.listdir(out) == []

# ---------------------------------------------------------------------------
# _choose_model
# ---------------------------------------------------------------------------
#
# Cellpose 4 ships exactly one model. models.MODEL_NAMES == ['cpsam'], and
# CellposeModel accepts-and-ignores model_type= / diam_mean=, resolving an
# unknown pretrained_model to cpsam with only a log warning. So the pre-SAM
# names were never loading the model they named. _choose_model now maps every
# legacy name to cpsam explicitly and says so, instead of pretending.


class _FakeCellposeModel:
    """Records the kwargs Cellpose would have been constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_cellpose(monkeypatch):
    """Swap the Cellpose constructor for a recording double (no weights, no GPU)."""
    # The production cache is intentionally process-wide within one run.  A
    # test parameter is a new run and must not inherit an earlier parameter's
    # suppressed notice.
    U.reset_cellpose_model_reports()
    monkeypatch.setattr(U.cp_models, "CellposeModel", _FakeCellposeModel)
    yield _FakeCellposeModel
    U.reset_cellpose_model_reports()


@pytest.mark.parametrize(
    "name", ["cyto", "cyto2", "cyto3", "cyto_2", "cyto_3", "nuclei", "nucleus"]
)
def test_choose_model_maps_every_legacy_name_to_cpsam(fake_cellpose, name, capsys):
    """A pre-SAM name loads cpsam, and never passes model_type."""
    model = U._choose_model(name, device="cpu")
    assert isinstance(model, _FakeCellposeModel)
    assert model.kwargs["pretrained_model"] == "cpsam"
    assert "model_type" not in model.kwargs
    assert "diam_mean" not in model.kwargs
    # the substitution is announced rather than silent
    assert "cpsam" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["toxo_pv_lumen", "toxo_cyto"])
def test_choose_model_removed_toxo_models_fall_back_to_cpsam(fake_cellpose, name, capsys):
    """The bundled toxo checkpoints were Cellpose-3 CPnet and are gone.

    Their weights cannot load into CPSAM's Transformer (2 of 313 keys overlap),
    so the only honest behaviour is cpsam plus a clear message.
    """
    model = U._choose_model(name, device="cpu", object_type="pathogen",
                            object_settings={"diameter": 30})
    assert model.kwargs["pretrained_model"] == "cpsam"
    out = capsys.readouterr().out
    assert name in out and "cpsam" in out


def test_choose_model_sam_uses_cpsam_weights(fake_cellpose):
    model = U._choose_model("sam", device="cpu")
    assert model.kwargs["pretrained_model"] == "cpsam"


def test_choose_model_unknown_name_still_returns_cpsam(fake_cellpose):
    """An unrecognised name must not return None — that was a latent crash."""
    model = U._choose_model("no_such_model", device="cpu")
    assert isinstance(model, _FakeCellposeModel)
    assert model.kwargs["pretrained_model"] == "cpsam"


def test_choose_model_none_name_returns_cpsam(fake_cellpose):
    model = U._choose_model(None, device="cpu")
    assert model.kwargs["pretrained_model"] == "cpsam"


@pytest.mark.parametrize("restore", ["denoise", "deblur", "upsample"])
def test_choose_model_reports_and_ignores_restore_type(fake_cellpose, restore, capsys):
    """The denoise/deblur/upsample checkpoints are pre-SAM and unavailable.

    They are reported and ignored rather than silently constructing a
    CellposeDenoiseModel whose model_type Cellpose 4 would discard anyway.
    """
    model = U._choose_model("cyto", device="cpu", restore_type=restore)
    assert model.kwargs["pretrained_model"] == "cpsam"
    out = capsys.readouterr().out
    assert restore in out and "not supported" in out


def test_choose_model_passes_the_device_through(fake_cellpose):
    model = U._choose_model("cyto", device="cuda:1")
    assert model.kwargs["device"] == "cuda:1"


def test_choose_model_tolerates_missing_object_settings(fake_cellpose):
    """object_settings is optional; the removed toxo branch used to index it."""
    model = U._choose_model("cyto", device="cpu", object_type="pathogen",
                            object_settings=None)
    assert model.kwargs["pretrained_model"] == "cpsam"


# ---------------------------------------------------------------------------
# Fine-tuned checkpoints
#
# pretrained_model was hard-coded to 'cpsam', so every model produced by
# spaCR's own Train Cellpose module was discarded and the stock weights used
# instead — the trained model could never be applied to anything, and nothing
# said so.
# ---------------------------------------------------------------------------

def test_choose_model_loads_a_fine_tuned_checkpoint(fake_cellpose, tmp_path, capsys):
    ckpt = tmp_path / "my_cells_model"
    ckpt.write_bytes(b"not really weights, but it exists")

    model = U._choose_model(str(ckpt), device="cpu", object_type="cell")

    assert model.kwargs["pretrained_model"] == str(ckpt)
    assert str(ckpt) in capsys.readouterr().out


@pytest.mark.parametrize("suffix", [".pth", ".pt", ""])
def test_choose_model_loads_a_checkpoint_whatever_its_extension(
        fake_cellpose, tmp_path, suffix):
    """Cellpose checkpoints are commonly extensionless."""
    ckpt = tmp_path / f"cp_model{suffix}"
    ckpt.write_bytes(b"x")
    model = U._choose_model(str(ckpt), device="cpu")
    assert model.kwargs["pretrained_model"] == str(ckpt)


def test_choose_model_raises_on_a_missing_checkpoint_path(fake_cellpose, tmp_path):
    """Falling back to cpsam would segment with the wrong weights and say
    nothing, which is worse than stopping."""
    missing = tmp_path / "gone" / "model.pth"
    with pytest.raises(FileNotFoundError) as exc:
        U._choose_model(str(missing), device="cpu", object_type="nucleus")
    msg = str(exc.value)
    assert str(missing) in msg
    assert "nucleus" in msg
    assert "cpsam" in msg          # tells the user what to write instead


def test_choose_model_raises_on_a_missing_path_without_an_extension(
        fake_cellpose, tmp_path):
    missing = tmp_path / "models" / "trained_on_hela"
    with pytest.raises(FileNotFoundError):
        U._choose_model(str(missing), device="cpu")


def test_choose_model_bare_unknown_word_is_not_treated_as_a_path(
        fake_cellpose, capsys):
    """A typo'd model name has no separator and no extension, so it falls back
    to cpsam with a message rather than raising."""
    model = U._choose_model("cytoo", device="cpu")
    assert model.kwargs["pretrained_model"] == "cpsam"
    assert "cytoo" in capsys.readouterr().out


def test_choose_model_cpsam_by_name_does_not_touch_the_filesystem(fake_cellpose):
    model = U._choose_model("cpsam", device="cpu")
    assert model.kwargs["pretrained_model"] == "cpsam"


def test_choose_model_strips_surrounding_whitespace(fake_cellpose, tmp_path):
    """A path pasted into a settings CSV often carries a trailing space."""
    ckpt = tmp_path / "model"
    ckpt.write_bytes(b"x")
    model = U._choose_model(f"  {ckpt}  ", device="cpu")
    assert model.kwargs["pretrained_model"] == str(ckpt)
