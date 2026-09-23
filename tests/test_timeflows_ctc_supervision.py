"""Only unambiguous full masks may teach CTC motion or disappearance."""
import numpy as np
import pytest
import tifffile

from spacr import timeflows_model as tm
from tools import evaluate_timeflows as evaluator


def test_merged_objects_do_not_choose_the_largest_marker():
    segmentation = np.ones((8, 8), np.uint16)
    markers = np.zeros_like(segmentation)
    markers[1:4, 1:4] = 17
    markers[5, 5] = 93
    assert not tm.track_masks_from_ctc(segmentation, markers).any()


def test_split_objects_do_not_become_one_disconnected_track():
    segmentation = np.ones((8, 8), np.uint16)
    segmentation[:, 4:] = 2
    markers = np.zeros_like(segmentation)
    markers[1, 1] = markers[1, 5] = 17
    assert not tm.track_masks_from_ctc(segmentation, markers).any()


@pytest.mark.parametrize("convert", [tm.track_masks_from_ctc, lambda s, m: evaluator.tracked_masks(s, m)[0]])
def test_track_shared_with_merged_object_is_excluded_from_both(convert):
    segmentation = np.ones((8, 8), np.uint16)
    segmentation[:, 4:] = 2
    markers = np.zeros_like(segmentation)
    markers[1, 1] = markers[1, 5] = 17
    markers[5, 5] = 93
    assert not convert(segmentation, markers).any()


def _movie(root, target_case):
    for folder in ("01", "01_ST/SEG", "01_GT/TRA"):
        (root / folder).mkdir(parents=True)
    first = np.zeros((16, 24), np.uint16)
    first[1:5, 1:5] = 1
    first[1:5, 9:13] = 2
    first[9:13, 1:5] = 3
    marker = np.zeros_like(first)
    marker[2, 2], marker[2, 10], marker[10, 2] = 17, 31, 49
    target = first.copy()
    target_marker = marker.copy()
    target[target == 3] = 0
    target_marker[target_marker == 49] = 0
    if target_case == "missing":
        target[target == 1] = 0
    elif target_case == "merged":
        target_marker[3, 3] = 99
    elif target_case == "split":
        target[1:5, 3:5] = 4
        target_marker[2, 4] = 17
    elif target_case == "absent":
        target_marker[target_marker == 17] = 0
    for frame, seg, markers in ((0, first, marker), (1, target, target_marker), (2, first, marker)):
        for folder, prefix, array in (("01", "t", seg * 100),
                                      ("01_ST/SEG", "man_seg", seg),
                                      ("01_GT/TRA", "man_track", markers)):
            tifffile.imwrite(root / folder / f"{prefix}{frame:03d}.tif", array)


@pytest.mark.parametrize("target_case", ["missing", "merged", "split"])
def test_missing_next_full_mask_is_censored_but_true_absence_stays(tmp_path, target_case):
    _movie(tmp_path, target_case)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.tif")}
    pairs = tm.ctc_pairs(str(tmp_path))
    first = pairs[0]
    assert set(np.unique(first.labels_t)) == {0, 31, 49}
    targets = tm.time_targets(first.labels_t, first.labels_t1)
    assert targets["object_weight"].sum() == 32
    assert targets["successor"].sum() == 16
    assert targets["vector_weight"].sum() == 16
    assert all(path.read_bytes() == content for path, content in before.items())


def test_missing_track_marker_is_an_absence_in_supplied_annotation(tmp_path):
    _movie(tmp_path, "absent")
    pair = tm.ctc_pairs(str(tmp_path))[0]
    assert set(np.unique(pair.labels_t)) == {0, 17, 31, 49}
    assert set(np.unique(pair.labels_t1)) == {0, 31}


@pytest.mark.parametrize("segmentation,markers", [
    (np.ones((2, 3, 4), np.int32), np.ones((2, 3, 4), np.int32)),
    (np.ones((2, 3), np.int32), np.ones((3, 2), np.int32)),
    (np.ones((2, 3), float), np.ones((2, 3), np.int32)),
    (np.ones((2, 3), np.int32), np.ones((2, 3), float)),
    (-np.ones((2, 3), np.int32), np.ones((2, 3), np.int32)),
    (np.ones((2, 3), np.int32), -np.ones((2, 3), np.int32)),
])
def test_invalid_annotation_arrays_are_rejected(segmentation, markers):
    with pytest.raises(ValueError):
        tm.track_masks_from_ctc(segmentation, markers)


@pytest.mark.parametrize("track", [2**40 + 7, np.iinfo(np.int64).max])
def test_large_track_ids_are_preserved_without_narrowing(track):
    segmentation = np.full((8, 8), 2**63 + 11, np.uint64)
    markers = np.zeros((8, 8), np.uint64)
    markers[2, 2] = track
    output = tm.track_masks_from_ctc(segmentation, markers)
    assert output.dtype == np.int64 and (output == track).all()


def test_unrepresentable_track_ids_are_refused():
    with pytest.raises(ValueError, match="int64 capacity"):
        tm.track_masks_from_ctc(np.ones((2, 2), np.uint16),
                               np.full((2, 2), 2**63, np.uint64))


def test_exclusion_counts_include_orphan_markers_and_overlap_categories():
    segmentation = np.zeros((8, 12), np.int32)
    segmentation[:, :4] = 1
    segmentation[:, 4:8] = 2
    markers = np.zeros_like(segmentation)
    markers[1, 1], markers[1, 5], markers[5, 5], markers[1, 10] = 17, 17, 93, 104
    output, counts = tm._ctc_track_masks(segmentation, markers)
    assert not output.any()
    assert counts == {"retained_tracks": 0, "unmarked_objects": 0,
                      "multi_marker_objects": 1, "duplicate_track_objects": 2,
                      "markers_without_retained_full_mask": 3,
                      "excluded_track_ids": [17, 93, 104]}
    empty = np.zeros((0, 8), np.int32)
    assert tm.track_masks_from_ctc(empty, empty).shape == empty.shape


def test_pair_specific_censoring_does_not_change_the_previous_pairs_target(tmp_path):
    _movie(tmp_path, "valid")
    path = tmp_path / "01_ST/SEG/man_seg002.tif"
    target = tifffile.imread(path)
    target[target == 1] = 0
    tifffile.imwrite(path, target)
    previous, current = tm.ctc_pairs(str(tmp_path))
    assert 17 in previous.labels_t1
    assert 17 not in current.labels_t
    assert tm.time_targets(previous.labels_t, previous.labels_t1)["successor"].sum() == 32


@pytest.mark.parametrize("folder,prefix", [("01", "t"), ("01_ST/SEG", "man_seg"), ("01_GT/TRA", "man_track")])
def test_duplicate_frame_numbers_are_rejected_before_reading_pixels(tmp_path, monkeypatch, folder, prefix):
    _movie(tmp_path, "valid")
    (tmp_path / folder / f"{prefix}000.tiff").touch()
    monkeypatch.setattr(tifffile, "imread", lambda path: pytest.fail("must reject names before reading pixels"))
    with pytest.raises(ValueError, match="Duplicate frame 0"):
        tm.ctc_pairs(str(tmp_path))


def test_slice_mask_names_cannot_overwrite_whole_frame_annotations(tmp_path):
    _movie(tmp_path, "valid")
    (tmp_path / "01_ST/SEG/man_seg_999_000.tif").write_text("a slice, not a full frame")
    (tmp_path / "01/t_notes_001.tif").write_text("not a full frame")
    assert len(tm.ctc_pairs(str(tmp_path))) == 2


@pytest.mark.parametrize("sequence,maximum", [("../01", None), ("１２", None), ("1", None), ("01", -1)])
def test_invalid_sequence_or_pair_limit_is_rejected_before_io(tmp_path, sequence, maximum):
    with pytest.raises(ValueError):
        tm.ctc_pairs(str(tmp_path), sequence, max_pairs=maximum)
