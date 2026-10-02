"""ROI and mask conversion at the edges the coverage ratchet found untested.

Each test is a malformed or unusual input a user's file can hold -- labels
that are not whole numbers, a GeoJSON that is a bare geometry, an RLE whose
runs do not cover the image, a COCO file without the requested image -- and
what spacr.mask_io does with it: refuse with a sentence, or read it the way
the format means it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from spacr import mask_io


def _square(shape=(12, 12), box=(2, 7, 3, 8), label=1):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[box[0]:box[1], box[2]:box[3]] = label
    return mask


@pytest.mark.parametrize("labels,match", [
    (np.array([["a", "b"]]), "must hold integer labels"),
    (np.array([[0.5, 1.0]]), "must hold integer labels"),
    (np.array([[np.nan, 1.0]]), "must hold integer labels"),
    (np.array([[-1, 2]]), "holds negative labels"),
])
def test_labels_that_are_not_nonnegative_integers_are_refused(labels, match):
    with pytest.raises(ValueError, match=match):
        mask_io._label_array(labels, "cell")


def test_an_empty_mapping_has_no_masks_to_export():
    with pytest.raises(ValueError, match="no masks to export"):
        mask_io._object_masks({}, None)


def test_a_flat_class_map_needs_exactly_one_object_type():
    with pytest.raises(ValueError, match="exactly one"):
        mask_io._object_classes({1: "infected"}, ["cell", "nucleus"])
    assert mask_io._object_classes({1: "infected"}, ["cell"]) == {
        "cell": {1: "infected"}}


def test_an_empty_image_has_no_outline():
    assert mask_io._trace_rings(np.zeros((4, 4), dtype=bool)) == []


def test_rings_too_short_or_outside_the_image_fill_nothing():
    rr, cc = mask_io._fill_rings([[[0, 0], [1, 1]]], (5, 5))
    assert rr.size == cc.size == 0
    far = [[[50, 50], [60, 50], [60, 60], [50, 60]]]
    rr, cc = mask_io._fill_rings(far, (5, 5))
    assert rr.size == 0


@pytest.mark.parametrize("value,expected", [
    ("7", 7), (3.0, 3), ("x", None), (None, None), (2.5, None),
    (float("inf"), None)])
def test_ids_are_whole_numbers_or_nothing(value, expected):
    assert mask_io._as_int(value) == expected


def test_an_unknown_qupath_object_kind_is_refused():
    with pytest.raises(ValueError, match="object_kind must be one of"):
        mask_io.masks_to_geojson(_square(), object_kind="cell")


def test_geojson_comes_as_a_list_a_feature_or_a_bare_geometry():
    polygon = {"type": "Polygon",
               "coordinates": [[[2, 2], [6, 2], [6, 6], [2, 6], [2, 2]]]}
    feature = {"type": "Feature", "geometry": polygon, "properties": {}}
    assert mask_io._geojson_features([feature, "noise"]) == [feature]
    assert mask_io._geojson_features(feature) == [feature]
    assert mask_io._geojson_features(polygon)[0]["geometry"] is polygon
    with pytest.raises(ValueError, match="not a GeoJSON"):
        mask_io._geojson_features({"type": "Point", "coordinates": [1, 1]})


def test_only_area_geometries_have_polygons():
    ring = [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]
    assert mask_io._geometry_polygons("not a geometry") == []
    assert mask_io._geometry_polygons({"type": "Point"}) == []
    assert mask_io._geometry_polygons(
        {"type": "MultiPolygon", "coordinates": [[ring], [ring]]}) == [
            [ring], [ring]]
    assert mask_io._geometry_polygons({
        "type": "GeometryCollection",
        "geometries": [{"type": "Polygon", "coordinates": [ring]},
                       {"type": "LineString", "coordinates": ring}]}) == [
            [ring]]


def test_qupath_ids_come_from_measurements_or_the_name(tmp_path):
    """A QuPath measurement list, an object_id measurement, and a
    "<type> <id>" name each give the object its number."""
    def feature(x, props):
        return {"type": "Feature", "properties": props, "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, 1], [x + 3, 1], [x + 3, 4], [x, 4],
                             [x, 1]]]}}

    data = {"type": "FeatureCollection", "features": [
        feature(1, {"measurements": [{"name": "object_id", "value": 7},
                                     "noise"]}),
        feature(6, {"name": "cell 9"}),
        feature(11, {"measurements": {"object_id": 4}}),
    ]}
    path = tmp_path / "objects.geojson"
    path.write_text(json.dumps(data))
    masks = mask_io.geojson_to_masks(str(path), (8, 16), object_type="cell")
    cell = masks["cell"]
    assert cell[2, 2] == 7 and cell[2, 7] == 9 and cell[2, 12] == 4


def test_writing_a_roiset_over_an_old_one_replaces_it(tmp_path):
    pytest.importorskip("roifile")
    target = tmp_path / "rois" / "RoiSet.zip"
    mask_io.masks_to_roiset(_square(), target)
    first = target.stat().st_size
    mask_io.masks_to_roiset(_square(label=3), target)
    assert target.stat().st_size == first
    back = mask_io.roiset_to_masks(target, (12, 12))
    assert set(np.unique(next(iter(back.values())))) == {0, 3}


def test_rle_counts_as_bytes_or_a_list_decode_and_bad_runs_are_refused():
    binary = np.zeros((4, 5), dtype=bool)
    binary[1:3, 1:4] = True
    encoded = mask_io.rle_encode(binary)
    if isinstance(encoded["counts"], str):
        as_bytes = dict(encoded, counts=encoded["counts"].encode("ascii"))
        assert np.array_equal(mask_io.rle_decode(as_bytes), binary)
    runs = [4, 2, 2, 2, 2, 2, 6]
    flat = {"size": [4, 5], "counts": runs}
    assert mask_io.rle_decode(flat).sum() == 6
    with pytest.raises(ValueError, match="RLE runs cover"):
        mask_io.rle_decode({"size": [4, 5], "counts": [3, 2]})


def _coco(shape=(6, 6), rle=None):
    annotation = {"id": 1, "image_id": 1, "category_id": 1,
                  "segmentation": rle or [[1, 1, 4, 1, 4, 4, 1, 4]]}
    return {"images": [{"id": 1, "file_name": "plate1_A01_1.png",
                        "height": shape[0], "width": shape[1]}],
            "annotations": [annotation],
            "categories": [{"id": 1, "name": "cell"}]}


def test_a_coco_image_that_is_not_in_the_file_is_named():
    with pytest.raises(ValueError, match="no image named 'plate9.png'"):
        mask_io.coco_to_masks(_coco(), file_name="plate9.png")
    masks = mask_io.coco_to_masks(_coco(), file_name="plate1_A01_1")
    assert masks["cell"].any()


def test_an_rle_that_does_not_fit_the_image_is_refused():
    rle = {"size": [3, 3], "counts": [9]}
    with pytest.raises(ValueError, match="does not fit"):
        mask_io.coco_to_masks(_coco(rle=rle))


def test_an_unknown_roi_format_is_refused_and_json_defaults_to_coco(tmp_path):
    with pytest.raises(ValueError, match="unknown ROI format"):
        mask_io.roi_format(tmp_path / "x.zip", "svg")
    assert mask_io.roi_format(tmp_path / "x", "QuPath") == "geojson"
    written = mask_io.export_rois(_square(), tmp_path / "plate1.json")
    data = json.loads(written.read_text())
    assert "annotations" in data
    assert data["images"][0]["file_name"] == "plate1"


def test_an_empty_label_array_is_kept_and_a_string_is_not_geojson():
    assert mask_io._label_array(np.zeros((0, 0), dtype=np.int32),
                                "cell").shape == (0, 0)
    with pytest.raises(ValueError, match="not a GeoJSON"):
        mask_io._geojson_features("FeatureCollection")


def test_a_single_imagej_roi_file_is_read(tmp_path):
    roifile = pytest.importorskip("roifile")
    roi = roifile.ImagejRoi.frompoints(
        np.array([[2, 2], [6, 2], [6, 6], [2, 6]], dtype=float))
    roi.name = "cell-5"
    path = tmp_path / "one.roi"
    roi.tofile(str(path))
    masks = mask_io.roiset_to_masks(path, (10, 10))
    assert set(np.unique(masks["cell"])) == {0, 5}


def test_coco_image_names_are_read_from_a_file(tmp_path):
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(_coco()))
    assert mask_io.coco_image_names(str(path)) == ["plate1_A01_1.png"]


def test_each_roi_format_has_its_suffix_and_a_parsed_coco_lists_its_images():
    assert [mask_io.roi_suffix(f) for f in ("geojson", "imagej", "coco")] == [
        ".geojson", ".zip", ".json"]
    assert mask_io.coco_image_names(_coco()) == ["plate1_A01_1.png"]
