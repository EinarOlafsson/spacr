"""Item 545: label masks out to QuPath GeoJSON, ImageJ RoiSets and COCO, and back.

The claim under test is the strong one: an outline traced along pixel edges and
rasterised back by pixel centre reproduces the mask EXACTLY -- no boundary
tolerance -- for every format, including holes, islands inside holes, objects
in several parts, objects meeting themselves only at a corner, and objects that
touch each other. The only place a tolerance is stated is a reader that is not
spaCR's: ``pycocotools`` rasterising a COCO polygon, allowed to differ on
boundary pixels only (it measures zero here, but the test does not demand it).
"""
from __future__ import annotations

import json
import sys
import zipfile

import numpy as np
import pytest
from scipy import ndimage

from spacr import mask_io

FORMATS = (("geojson", ".geojson"), ("imagej", ".zip"), ("coco", ".json"))


def _awkward_mask() -> np.ndarray:
    """Every shape the tracing has to get right, on one field."""
    mask = np.zeros((64, 96), np.uint16)
    mask[5:25, 5:30] = 3
    mask[10:20, 10:20] = 0
    mask[13:17, 13:17] = 3
    mask[5:25, 30:45] = 7
    mask[30:36, 30:36] = 9
    mask[45:50, 45:52] = 9
    mask[50, 60] = 12
    mask[51, 61] = 12
    mask[52, 60] = 12
    mask[40:48, 5:13] = 20
    mask[40, 13] = 20
    mask[41, 14] = 20
    mask[42:48, 13:15] = 20
    mask[0:3, :] = 4
    mask[60:64, 90:96] = 65535
    return mask


def _random_labels(seed: int, shape=(96, 96), connectivity=1) -> np.ndarray:
    from skimage.measure import label

    rng = np.random.default_rng(seed)
    noise = rng.random(shape) > 0.5
    return label(noise, connectivity=connectivity).astype(np.uint16)


@pytest.mark.parametrize("fmt,suffix", FORMATS)
def test_round_trip_reproduces_the_masks_exactly(tmp_path, fmt, suffix):
    if fmt == "imagej":
        pytest.importorskip("roifile")
    cells = _awkward_mask()
    nuclei = np.where(cells == 7, 2, 0).astype(np.uint16)
    classes = {"cell": {3: "infected", 9: "dividing"}}
    path = tmp_path / f"field{suffix}"
    mask_io.export_rois({"cell": cells, "nucleus": nuclei}, path, fmt,
                        classes=classes, file_name="field.tif")
    masks, got = mask_io.import_rois(path, cells.shape, fmt,
                                     with_classes=True)
    assert sorted(masks) == ["cell", "nucleus"]
    assert np.array_equal(masks["cell"], cells)
    assert np.array_equal(masks["nucleus"], nuclei)
    assert got["cell"][3] == "infected" and got["cell"][9] == "dividing"
    assert got["cell"][7] == "cell" and got["nucleus"] == {2: "nucleus"}


@pytest.mark.parametrize("fmt,suffix", FORMATS)
@pytest.mark.parametrize("connectivity", (1, 2))
def test_random_label_fields_round_trip_exactly(tmp_path, fmt, suffix,
                                                connectivity):
    if fmt == "imagej":
        pytest.importorskip("roifile")
    labels = _random_labels(7 + connectivity, connectivity=connectivity)
    path = tmp_path / f"noise{suffix}"
    mask_io.export_rois(labels, path, fmt)
    assert np.array_equal(mask_io.import_rois(path, labels.shape)["object"],
                          labels)


def test_touching_objects_stay_separate():
    mask = np.zeros((20, 20), np.uint16)
    mask[2:10, 2:10] = 1
    mask[2:10, 10:18] = 2
    mask[10:18, 2:18] = 3
    data = mask_io.masks_to_geojson(mask)
    assert [f["properties"]["object_id"] for f in data["features"]] == [1, 2, 3]
    back = mask_io.geojson_to_masks(data, mask.shape)["object"]
    assert np.array_equal(back, mask)


def test_holes_and_parts_are_written_as_geojson_geometry():
    mask = _awkward_mask()
    features = {f["properties"]["object_id"]: f
                for f in mask_io.masks_to_geojson(mask)["features"]}
    ring = features[3]["geometry"]
    assert ring["type"] == "MultiPolygon"
    shell_with_hole, island = ring["coordinates"]
    assert len(shell_with_hole) == 2 and len(island) == 1
    assert features[9]["geometry"]["type"] == "MultiPolygon"
    assert features[7]["geometry"]["type"] == "Polygon"
    for feature in features.values():
        for polygon in (feature["geometry"]["coordinates"]
                        if feature["geometry"]["type"] == "MultiPolygon"
                        else [feature["geometry"]["coordinates"]]):
            for points in polygon:
                assert points[0] == points[-1]


def test_geojson_has_the_keys_qupath_reads():
    """QuPath's GeoJSON import: a Feature with a UUID ``id``, a geometry, and
    ``objectType``, ``classification.name``/``color`` and ``name``."""
    import uuid

    data = mask_io.masks_to_geojson(_awkward_mask(), classes={3: "infected"},
                                    object_type="cell", image_name="a.tif")
    assert data["type"] == "FeatureCollection"
    json.dumps(data)
    for feature in data["features"]:
        assert feature["type"] == "Feature"
        uuid.UUID(feature["id"])
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        props = feature["properties"]
        assert props["objectType"] in ("annotation", "detection")
        assert set(props["classification"]) == {"name", "color"}
        color = props["classification"]["color"]
        assert len(color) == 3 and all(0 <= c <= 255 for c in color)
        assert props["name"] == f"cell {props['object_id']}"
        assert props["measurements"]["object_id"] == props["object_id"]
    names = {f["properties"]["object_id"]: f["properties"]["classification"]["name"]
             for f in data["features"]}
    assert names[3] == "infected" and names[7] == "cell"
    again = mask_io.masks_to_geojson(_awkward_mask(), classes={3: "infected"},
                                     object_type="cell", image_name="a.tif")
    assert again == data
    other = mask_io.masks_to_geojson(_awkward_mask(), object_type="cell",
                                     image_name="b.tif")
    assert {f["id"] for f in other["features"]}.isdisjoint(
        f["id"] for f in data["features"])


def test_geojson_geometry_is_valid_to_shapely():
    shapely = pytest.importorskip("shapely.geometry")
    labels = _random_labels(3, connectivity=2)
    for feature in mask_io.masks_to_geojson(labels)["features"]:
        shape = shapely.shape(feature["geometry"])
        assert shape.is_valid, feature["properties"]["object_id"]
        assert shape.area == np.count_nonzero(
            labels == feature["properties"]["object_id"])


def test_a_qupath_export_without_spacr_properties_imports_by_class():
    """What QuPath itself writes: a feature list, ``name`` free text, no ids."""
    features = [
        {"type": "Feature", "id": "8d7f3c4e-0000-4000-8000-000000000001",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[2, 2], [8, 2], [8, 6], [2, 6], [2, 2]]]},
         "properties": {"objectType": "annotation",
                        "classification": {"name": "Tumor",
                                           "color": [200, 0, 0]}}},
        {"type": "Feature", "id": "8d7f3c4e-0000-4000-8000-000000000002",
         "geometry": {"type": "Polygon",
                      "coordinates": [[[10, 1.4], [14.6, 1.4], [14.6, 9],
                                       [10, 9], [10, 1.4]]]},
         "properties": {"objectType": "annotation",
                        "classification": {"name": "Stroma"}}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]},
         "properties": {}},
    ]
    masks, classes = mask_io.geojson_to_masks(features, (12, 16),
                                              with_classes=True)
    assert sorted(masks) == ["Stroma", "Tumor"]
    assert np.count_nonzero(masks["Tumor"]) == 24
    assert set(np.unique(masks["Stroma"])) == {0, 1}
    stroma = np.zeros((12, 16), bool)
    stroma[1:9, 10:15] = True
    assert np.array_equal(masks["Stroma"] > 0, stroma)
    assert classes == {"Tumor": {1: "Tumor"}, "Stroma": {1: "Stroma"}}


def test_the_roiset_is_what_roifile_reads(tmp_path):
    roifile = pytest.importorskip("roifile")
    mask = _awkward_mask()
    path = mask_io.masks_to_roiset(mask, tmp_path / "RoiSet.zip",
                                   object_type="cell", classes={3: "infected"})
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    ids = sorted(int(i) for i in np.unique(mask) if i)
    assert names == [f"cell-{i}.roi" for i in ids]
    rois = {roi.name: roi for roi in roifile.roiread(str(path))}
    assert rois["cell-7"].roitype == roifile.ROI_TYPE.TRACED
    assert rois["cell-3"].composite and rois["cell-9"].composite
    assert "class: infected" in rois["cell-3"].props
    assert "object_id: 3" in rois["cell-3"].props
    coords = rois["cell-7"].coordinates()
    assert coords.min(axis=0).tolist() == [30, 5]
    assert coords.max(axis=0).tolist() == [45, 25]
    for roi in rois.values():
        assert roi.tobytes() == roifile.ImagejRoi.frombytes(roi.tobytes()).tobytes()


def test_a_fiji_roiset_without_properties_imports_by_name(tmp_path):
    roifile = pytest.importorskip("roifile")
    square = roifile.ImagejRoi.frompoints([[1, 1], [5, 1], [5, 4], [1, 4]],
                                          name="0001-0002")
    square.roitype = roifile.ROI_TYPE.POLYGON
    rect = roifile.ImagejRoi()
    rect.roitype = roifile.ROI_TYPE.RECT
    rect.left, rect.top, rect.right, rect.bottom = 7, 2, 10, 9
    rect.name = "vacuole-5"
    line = roifile.ImagejRoi()
    line.roitype = roifile.ROI_TYPE.LINE
    line.name = "a line"
    path = tmp_path / "RoiSet.zip"
    roifile.roiwrite(str(path), [square, rect, line])
    masks = mask_io.roiset_to_masks(path, (12, 12))
    assert sorted(masks) == ["object", "vacuole"]
    assert np.count_nonzero(masks["object"] == 1) == 12
    expected = np.zeros((12, 12), bool)
    expected[2:9, 7:10] = True
    assert np.array_equal(masks["vacuole"] == 5, expected)


def test_coco_is_what_pycocotools_reads(tmp_path):
    """pycocotools may differ from spaCR only on boundary pixels (1-pixel
    tolerance, stated); RLE must decode to the object exactly."""
    pytest.importorskip("pycocotools")
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO

    labels = _awkward_mask()
    for rle in (False, True):
        data = mask_io.masks_to_coco(labels, file_name="field.tif", rle=rle,
                                     object_type="cell")
        path = tmp_path / f"coco_{rle}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        coco = COCO(str(path))
        assert coco.imgs[1]["file_name"] == "field.tif"
        for ann in coco.loadAnns(coco.getAnnIds()):
            want = labels == ann["object_id"]
            got = coco.annToMask(ann).astype(bool)
            if isinstance(ann["segmentation"], dict):
                assert np.array_equal(got, want)
                encoded = coco_mask.encode(np.asfortranarray(want.astype(np.uint8)))
                assert encoded["counts"].decode() == ann["segmentation"]["counts"]
            else:
                ring = want ^ ndimage.binary_erosion(want)
                ring |= ndimage.binary_dilation(want) & ~want
                assert not ((got ^ want) & ~ring).any()
            assert ann["area"] == int(want.sum())
            x, y, w, h = ann["bbox"]
            rows, cols = np.nonzero(want)
            assert (x, y, w, h) == (cols.min(), rows.min(),
                                    np.ptp(cols) + 1, np.ptp(rows) + 1)


def test_coco_structure_and_rle_codec():
    labels = _awkward_mask()
    data = mask_io.masks_to_coco(labels, file_name="a.tif", object_type="cell",
                                 classes={3: "infected"})
    assert set(data) >= {"images", "annotations", "categories"}
    cats = {c["id"]: (c["supercategory"], c["name"]) for c in data["categories"]}
    assert sorted(cats.values()) == [("cell", "cell"), ("cell", "infected")]
    by_id = {a["object_id"]: a for a in data["annotations"]}
    assert isinstance(by_id[3]["segmentation"], dict)
    assert isinstance(by_id[7]["segmentation"], list)
    assert len(by_id[9]["segmentation"]) == 2
    for number, ann in by_id.items():
        assert cats[ann["category_id"]][1] == ("infected" if number == 3
                                               else "cell")
    rng = np.random.default_rng(5)
    for shape in ((1, 1), (7, 3), (40, 41)):
        binary = rng.random(shape) > 0.5
        encoded = mask_io.rle_encode(binary)
        assert np.array_equal(mask_io.rle_decode(encoded), binary)
        flat = binary.ravel(order="F")
        runs, value, count = [], False, 0
        for pixel in flat:
            if pixel == value:
                count += 1
            else:
                runs.append(count)
                value, count = pixel, 1
        runs.append(count)
        plain = {"size": list(shape), "counts": runs}
        assert np.array_equal(mask_io.rle_decode(plain), binary)


def test_a_coco_dataset_of_several_images(tmp_path):
    first = _random_labels(1, shape=(30, 40))
    second = _random_labels(2, shape=(50, 20))
    data = mask_io.masks_to_coco(first, file_name="a/one.tif")
    mask_io.masks_to_coco(second, file_name="two.png", dataset=data)
    assert [i["id"] for i in data["images"]] == [1, 2]
    assert len({a["id"] for a in data["annotations"]}) == len(data["annotations"])
    path = tmp_path / "set.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert mask_io.coco_image_names(path) == ["a/one.tif", "two.png"]
    assert np.array_equal(
        mask_io.coco_to_masks(path, file_name="one")["object"], first)
    assert np.array_equal(
        mask_io.import_rois(path, file_name="two.png")["object"], second)
    with pytest.raises(ValueError, match="2 images"):
        mask_io.coco_to_masks(path)


def test_ids_that_collide_or_are_missing_are_renumbered():
    features = [
        {"type": "Feature", "properties": {"object_id": 4},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}},
        {"type": "Feature", "properties": {"object_id": 4},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[3, 0], [5, 0], [5, 2], [3, 2], [3, 0]]]}},
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon",
                      "coordinates": [[[6, 0], [8, 0], [8, 2], [6, 2], [6, 0]]]}},
    ]
    labels = mask_io.geojson_to_masks({"type": "FeatureCollection",
                                       "features": features}, (4, 10))["object"]
    assert labels[0, 0] == 4 and labels[0, 3] == 5 and labels[0, 6] == 6


def test_an_empty_mask_exports_and_imports_as_nothing(tmp_path):
    empty = np.zeros((10, 12), np.uint16)
    for fmt, suffix in FORMATS:
        if fmt == "imagej" and not _has("roifile"):
            continue
        path = mask_io.export_rois(empty, tmp_path / f"e{suffix}", fmt)
        assert mask_io.import_rois(path, empty.shape, fmt) == {}


def test_format_detection_and_bad_input(tmp_path):
    assert mask_io.roi_format("x.geojson") == "geojson"
    assert mask_io.roi_format("x_RoiSet.zip") == "imagej"
    assert mask_io.roi_format("x.json") == "coco"
    assert mask_io.roi_format("x", "qupath") == "geojson"
    geo = tmp_path / "qupath.json"
    geo.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    assert mask_io.roi_format(geo) == "geojson"
    with pytest.raises(ValueError):
        mask_io.roi_format("x.txt")
    with pytest.raises(ValueError):
        mask_io.import_rois(tmp_path / "x.geojson")
    with pytest.raises(ValueError):
        mask_io.masks_to_geojson(np.zeros((2, 2, 2), np.uint16))
    with pytest.raises(ValueError):
        mask_io.masks_to_geojson({"a": np.zeros((2, 2)), "b": np.zeros((3, 3))})


def test_imagej_without_roifile_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "roifile", None)
    with pytest.raises(ImportError, match="pip install roifile"):
        mask_io.masks_to_roiset(np.ones((3, 3), np.uint16), tmp_path / "a.zip")
    with pytest.raises(ImportError, match="pip install roifile"):
        mask_io.roiset_to_masks(tmp_path / "a.zip", (3, 3))


def test_object_polygons_outline_one_object():
    mask = _awkward_mask()
    polygons = mask_io.object_polygons(mask, 3)
    assert len(polygons) == 2 and len(polygons[0]) == 2
    assert mask_io.object_polygons(mask, 99) == []


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
