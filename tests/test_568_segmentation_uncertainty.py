"""Segmentation uncertainty from test-time augmentation, and its queue order.

A field segmented under flips and a quarter turn by a segmenter that is sure
of itself gives four identical label images and no uncertainty; a segmenter
whose answer depends on orientation is uncertain exactly where it changes its
mind. The curation queue then offers the most uncertain fields first.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy import ndimage as ndi

from spacr.active_learning import (_TTA_TRANSFORMS, _segmentation_uncertainty,
                                   _tta_forward, _tta_inverse,
                                   _tta_label_sets)
from spacr.curation_queue import ORDERS, build_queue, order_items
from spacr.curation_queue import _load_uncertainty, _write_uncertainty


def _field() -> np.ndarray:
    image = np.zeros((40, 56), np.float32)
    image[5:15, 5:15] = 1.0
    image[25:35, 30:44] = 1.0
    return image


def _threshold(image: np.ndarray) -> np.ndarray:
    return ndi.label(np.asarray(image) > 0.5)[0]


@pytest.mark.parametrize("name", _TTA_TRANSFORMS)
@pytest.mark.parametrize("shape", [(40, 56), (40, 56, 3), (2, 40, 56)])
def test_every_transform_is_undone_on_the_labels(name, shape):
    rng = np.random.default_rng(0)
    image = rng.random(shape)
    moved = _tta_forward(image, name)
    plane = moved[0] if shape[0] == 2 else (
        moved[..., 0] if moved.ndim == 3 else moved)
    labels = (plane * 100).astype(np.int32)
    back = _tta_inverse(labels, name)
    reference = image[0] if shape[0] == 2 else (
        image[..., 0] if image.ndim == 3 else image)
    assert back.shape == reference.shape
    assert np.array_equal(back, (reference * 100).astype(np.int32))


def test_an_unknown_transform_is_refused():
    with pytest.raises(ValueError):
        _tta_forward(_field(), "rot45")


def test_a_segmenter_that_never_changes_its_mind_is_certain():
    sets = _tta_label_sets(_field(), _threshold)
    assert len(sets) == len(_TTA_TRANSFORMS)
    result = _segmentation_uncertainty(sets)
    assert result["field"] == pytest.approx(0.0)
    assert result["n_objects"] == 2
    assert set(result["objects"].values()) == {0.0}
    assert float(result["map"].max()) == 0.0


def test_uncertainty_is_high_where_the_passes_disagree():
    reference = _threshold(_field())
    small = int(reference[10, 10])
    large = int(reference[30, 37])
    sets = [reference, reference.copy(), reference.copy(), reference.copy()]
    for other in sets[1:]:
        other[other == large] = 0
    result = _segmentation_uncertainty(sets)
    assert result["objects"][small] == pytest.approx(0.0)
    assert result["objects"][large] == pytest.approx(1.0)
    assert result["map"][30, 37] > 0.5
    assert result["map"][10, 10] == 0.0
    assert 0.0 < result["field"] < 1.0


def test_an_object_only_some_orientations_find_raises_the_field_score():
    def blind_corner(image):
        labels = _threshold(image)
        labels[:20, :20] = 0
        return labels

    sets = _tta_label_sets(_field(), blind_corner)
    assert all(labels.shape == (40, 56) for labels in sets)
    assert int((sets[0][5:15, 5:15] > 0).sum()) == 0
    assert int((sets[1][5:15, 5:15] > 0).sum()) == 100
    result = _segmentation_uncertainty(sets)
    assert result["n_objects"] == 1
    assert result["field"] > 0.2
    assert result["map"][10, 10] > 0.5


def test_uncertainty_needs_two_passes_of_one_shape():
    with pytest.raises(ValueError):
        _segmentation_uncertainty([_threshold(_field())])
    with pytest.raises(ValueError):
        _segmentation_uncertainty([np.zeros((4, 4)), np.zeros((4, 5))])


def _queue_folder(tmp_path, stems):
    import imageio.v2 as imageio

    folder = tmp_path / "queue"
    (folder / "masks").mkdir(parents=True)
    for stem in stems:
        imageio.imwrite(folder / f"{stem}.tif", np.zeros((8, 8), np.uint16))
    return folder


def test_the_uncertain_order_offers_the_worst_fields_first(tmp_path):
    assert "uncertain" in ORDERS
    folder = _queue_folder(tmp_path, ["a", "b", "c", "d"])
    _write_uncertainty(folder, {"a": {"uncertainty": 0.1},
                                "b": {"uncertainty": 0.7, "n_objects": 3}})
    _write_uncertainty(folder, {"c": {"uncertainty": 0.4}})
    assert _load_uncertainty(folder) == {"a": 0.1, "b": 0.7, "c": 0.4}
    notes = []
    queue = build_queue(folder, order="uncertain", announce=notes.append)
    assert [item.stem for item in queue.items] == ["b", "c", "a", "d"]
    assert queue.effective_order == "uncertain"
    assert any("1 of 4" in note for note in notes)
    with open(folder / "curate_uncertainty.csv", newline="") as handle:
        rows = {row["stem"]: row for row in csv.DictReader(handle)}
    assert rows["b"]["n_objects"] == "3" and rows["b"]["updated"]


def test_the_uncertain_order_falls_back_to_value_and_says_so(tmp_path):
    folder = _queue_folder(tmp_path, ["a", "b"])
    notes = []
    queue = build_queue(folder, order="uncertain", announce=notes.append)
    assert queue.effective_order == "value"
    assert any("no segmentation uncertainty" in note for note in notes)
    items = list(queue.items)
    assert order_items(items, "uncertain", probs={"b": 0.9},
                       announce=notes.append)[0].stem == "b"
