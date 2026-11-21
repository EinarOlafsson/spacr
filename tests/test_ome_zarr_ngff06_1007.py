"""Item 1007 lane A: OME-Zarr NGFF 0.6 read and opt-in write (alpha).

T1-T6, T11 and T12 of proposals/bmc_paper_items/1007_ome_zarr_ngff_06.md,
plus the plate-store and OPS wiring of ``ome_zarr_ngff_version``. T7
(registration) and T8-T10 (converter) belong to lanes C and B.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")

from spacr import ome_zarr
from spacr.layers import Spacing
from spacr.ome_zarr import (OmeZarrError, multiscale_to_ngff05,
                            multiscale_to_ngff06, read_ome_zarr,
                            transformation_matrix, write_ome_zarr)


def _attrs(path: Path) -> dict:
    """The ``attributes`` of a zarr-format 3 group."""
    return json.loads((Path(path) / "zarr.json").read_text(encoding="utf-8"))["attributes"]


def _set_attrs(path: Path, attributes: dict) -> None:
    """Replace a zarr-format 3 group's ``attributes``."""
    meta = json.loads((Path(path) / "zarr.json").read_text(encoding="utf-8"))
    meta["attributes"] = attributes
    (Path(path) / "zarr.json").write_text(json.dumps(meta), encoding="utf-8")


def _chunks_md5(root: Path) -> dict:
    """md5 of every non-metadata file under ``root``, by relative path."""
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.name != "zarr.json"}


def _cyx():
    """A small CYX image and its spacing."""
    data = np.arange(2 * 40 * 36, dtype=np.uint16).reshape(2, 40, 36)
    return data, Spacing.from_map({"y": 0.65, "x": 0.65}, units="um")


def _zyx():
    """A small ZYX stack and its anisotropic spacing."""
    data = (np.arange(4 * 24 * 20, dtype=np.uint16) % 977).reshape(4, 24, 20)
    return data, Spacing.from_map({"z": 2.0, "y": 0.5, "x": 0.5}, units="um")


@pytest.mark.parametrize("make, axes", [(_cyx, ["c", "y", "x"]), (_zyx, ["z", "y", "x"])])
def test_t1_hand_built_06_store_reads_like_its_05_twin(tmp_path, make, axes):
    """T1: a hand-written 0.6 store gives the 0.5 spacing, offset and pixels."""
    data, spacing = make()
    v05 = write_ome_zarr(tmp_path / "v05", data, axes=axes, spacing=spacing,
                         ngff_version="0.5", levels=2)
    hand = tmp_path / "hand06"
    write_ome_zarr(hand, data, axes=axes, spacing=spacing, ngff_version="0.5", levels=2)
    ms = _attrs(hand)["ome"]["multiscales"][0]
    datasets = []
    for dataset in ms["datasets"]:
        scale, translation = dataset["coordinateTransformations"]
        datasets.append({"path": dataset["path"], "coordinateTransformations": [{
            "type": "sequence", "input": {"path": dataset["path"]},
            "output": {"name": "intrinsic"},
            "transformations": [scale, translation]}]})
    _set_attrs(hand, {"ome": {"version": "0.6", "multiscales": [{
        "name": ms["name"],
        "coordinateSystems": [{"name": "intrinsic", "axes": ms["axes"]}],
        "datasets": datasets, "type": ms["type"]}]}})
    image = read_ome_zarr(hand)
    assert image.ngff_version == "0.6"
    assert image.levels == v05.levels
    assert image.axes == v05.axes
    assert image.spacing == v05.spacing
    assert [lv.translation for lv in image.levels] == [lv.translation for lv in v05.levels]
    for level in range(2):
        assert np.array_equal(image.read(level), v05.read(level))
    assert [s["name"] for s in image.coordinate_systems] == ["intrinsic"]
    assert image.transformations == ()


def test_t1_single_scale_level_with_no_translation_reads(tmp_path):
    """T1: a level written as one scale (no sequence) reads with offset 0."""
    data, spacing = _cyx()
    path = tmp_path / "s"
    write_ome_zarr(path, data, axes=["c", "y", "x"], spacing=spacing, ngff_version="0.6")
    attrs = _attrs(path)
    dataset = attrs["ome"]["multiscales"][0]["datasets"][0]
    dataset["coordinateTransformations"] = [{
        "type": "scale", "input": {"path": "0"}, "output": {"name": "physical"},
        "scale": [1.0, 0.65, 0.65]}]
    _set_attrs(path, attrs)
    image = read_ome_zarr(path)
    assert image.levels[0].scale == (1.0, 0.65, 0.65)
    assert image.levels[0].translation == (0.0, 0.0, 0.0)


def test_t2_05_and_04_reads_carry_no_06_fields(tmp_path):
    """T2: 0.4 and 0.5 reads are unchanged: no systems, no 0.6 axis fields."""
    data, spacing = _cyx()
    for version in ("0.4", "0.5"):
        path = tmp_path / version
        write_ome_zarr(path, data, axes=["c", "y", "x"], spacing=spacing,
                       ngff_version=version)
        if version == "0.5":
            attrs = _attrs(path)
            attrs["ome"]["multiscales"][0]["axes"][1]["longName"] = "rows"
            attrs["ome"]["multiscales"][0]["axes"][1]["discrete"] = False
            _set_attrs(path, attrs)
        image = read_ome_zarr(path)
        assert image.ngff_version == version
        assert image.coordinate_systems == () and image.transformations == ()
        assert all(a.discrete is None and a.long_name is None for a in image.axes)


def test_t3_06_write_layout_and_round_trip(tmp_path):
    """T3: version only at ome.version, object endpoints, identical read-back."""
    data, spacing = _cyx()
    v05 = write_ome_zarr(tmp_path / "v05", data, axes=["c", "y", "x"],
                         spacing=spacing, ngff_version="0.5", levels=2,
                         channel_names=["DAPI", "GFP"])
    v06 = write_ome_zarr(tmp_path / "v06", data, axes=["c", "y", "x"],
                         spacing=spacing, ngff_version="0.6", levels=2,
                         channel_names=["DAPI", "GFP"])
    ome = _attrs(tmp_path / "v06")["ome"]
    assert ome["version"] == "0.6"
    ms = ome["multiscales"][0]
    assert "version" not in ms and "axes" not in ms
    assert ms["coordinateSystems"] == [{"name": "physical", "axes": [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"}]}]
    for dataset in ms["datasets"]:
        (outer,) = dataset["coordinateTransformations"]
        assert outer["type"] == "sequence"
        assert outer["input"] == {"path": dataset["path"]}
        assert outer["output"] == {"name": "physical"}
        assert [t["type"] for t in outer["transformations"]] == ["scale", "translation"]
        assert all("input" not in t and "output" not in t for t in outer["transformations"])
    assert v06.levels == v05.levels and v06.axes == v05.axes
    assert v06.channel_names == v05.channel_names
    assert _chunks_md5(tmp_path / "v05") == _chunks_md5(tmp_path / "v06")
    back = multiscale_to_ngff05(ms)
    assert back.pop("name") == "v06"
    golden = dict(_attrs(tmp_path / "v05")["ome"]["multiscales"][0])
    assert golden.pop("name") == "v05"
    assert back == golden


def test_t3_06_extra_systems_and_transformations(tmp_path):
    """T3: named systems and multiscale transformations are written and kept."""
    data, spacing = _cyx()
    axes = [{"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}]
    shift = {"type": "translation", "translation": [0.0, 1.25, -3.5],
             "input": {"name": "physical"}, "output": {"name": "sbs"}}
    image = write_ome_zarr(tmp_path / "r", data, axes=["c", "y", "x"],
                           spacing=spacing, ngff_version="0.6",
                           coordinate_systems=[{"name": "sbs", "axes": axes}],
                           coordinate_transformations=[shift])
    assert [s["name"] for s in image.coordinate_systems] == ["physical", "sbs"]
    assert image.transformations == (shift,)
    assert image.levels[0].translation == (0.0, 0.0, 0.0)
    matrix = transformation_matrix(image.transformations[0], 3)
    assert matrix[1, 3] == 1.25 and matrix[2, 3] == -3.5


def test_t3_extra_metadata_needs_06_and_unique_names(tmp_path):
    """Extra systems on 0.5, or a reused system name, are refused before writing."""
    data, spacing = _cyx()
    with pytest.raises(OmeZarrError, match="0.6"):
        write_ome_zarr(tmp_path / "a", data, axes=["c", "y", "x"], spacing=spacing,
                       ngff_version="0.5", coordinate_systems=[{"name": "s", "axes": []}])
    with pytest.raises(OmeZarrError, match="unique"):
        write_ome_zarr(tmp_path / "b", data, axes=["c", "y", "x"], spacing=spacing,
                       ngff_version="0.6",
                       coordinate_systems=[{"name": "physical", "axes": []}])
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_t4_default_write_is_unchanged(tmp_path):
    """T4: the 0.5 metadata keeps its golden layout; the writer default stays 0.4."""
    data, spacing = _cyx()
    write_ome_zarr(tmp_path / "g", data, axes=["c", "y", "x"], spacing=spacing,
                   ngff_version="0.5")
    ome = _attrs(tmp_path / "g")["ome"]
    assert ome["version"] == "0.5"
    ms = ome["multiscales"][0]
    assert list(ms) == ["name", "axes", "datasets", "type", "metadata"]
    assert ms["datasets"] == [{"path": "0", "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 0.65, 0.65]},
        {"type": "translation", "translation": [0.0, 0.0, 0.0]}]}]
    assert write_ome_zarr(tmp_path / "d", data).ngff_version == "0.4"


def test_t5_transformation_conformance():
    """T5: every 0.6 matrix type reduces to the matrix it means."""
    np.testing.assert_array_equal(transformation_matrix({"type": "identity"}, 2), np.eye(3))
    scale = transformation_matrix({"type": "scale", "scale": [2.0, 3.0]}, 2)
    np.testing.assert_array_equal(scale, np.diag([2.0, 3.0, 1.0]))
    shift = transformation_matrix({"type": "translation", "translation": [1.0, -2.0]}, 2)
    np.testing.assert_array_equal(shift @ [0, 0, 1], [1.0, -2.0, 1.0])
    sequence = transformation_matrix({"type": "sequence", "transformations": [
        {"type": "scale", "scale": [2.0, 3.0]},
        {"type": "translation", "translation": [1.0, -2.0]}]}, 2)
    np.testing.assert_array_equal(sequence @ [1, 1, 1], [3.0, 1.0, 1.0])
    affine = transformation_matrix({"type": "affine", "affine": [[1, 2, 3], [4, 5, 6]]}, 2)
    np.testing.assert_array_equal(affine, [[1, 2, 3], [4, 5, 6], [0, 0, 1]])
    lifted = transformation_matrix({"type": "affine", "affine": [[1, 0, 0], [2, 3, 4], [5, 6, 7]]}, 2)
    assert lifted.shape == (4, 3)
    rotation = transformation_matrix({"type": "rotation", "rotation": [[0, -1], [1, 0]]}, 2)
    np.testing.assert_array_equal(rotation @ [1, 0, 1], [0, 1, 1])
    swap = transformation_matrix({"type": "mapAxis", "mapAxis": [1, 0]}, 2)
    np.testing.assert_array_equal(swap @ [5, 7, 1], [7, 5, 1])
    by_dim = transformation_matrix({"type": "byDimension", "transformations": [
        {"transformation": {"type": "translation", "translation": [-1.0]},
         "inputAxes": [1], "outputAxes": [1]},
        {"transformation": {"type": "scale", "scale": [2.0]},
         "inputAxes": [0], "outputAxes": [0]}]}, 2)
    np.testing.assert_array_equal(by_dim @ [3, 4, 1], [6, 3, 1])
    project = transformation_matrix({"type": "projectAxis", "createdOutputs": [0],
                                     "droppedInputs": [0]}, 3)
    np.testing.assert_array_equal(project @ [9, 2, 3, 1], [0, 2, 3, 1])
    bijection = transformation_matrix({"type": "bijection",
                                       "forward": {"type": "scale", "scale": [2.0, 2.0]},
                                       "inverse": {"type": "scale", "scale": [0.5, 0.5]}}, 2)
    np.testing.assert_array_equal(bijection, np.diag([2.0, 2.0, 1.0]))


def test_t5_affine_by_path_reads_the_stored_array(tmp_path):
    """T5: an affine stored as a Zarr array by ``path`` is read relative to the store."""
    zarr = pytest.importorskip("zarr")
    group = zarr.open_group(str(tmp_path / "store"), mode="w", zarr_format=3)
    rows = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, -1.5]])
    group.create_group("coordinateTransformations").create_array(
        "tile0", shape=rows.shape, dtype="f8", chunks=rows.shape)[...] = rows
    matrix = transformation_matrix({"type": "affine", "path": "coordinateTransformations/tile0"},
                                   2, root=tmp_path / "store")
    np.testing.assert_array_equal(matrix[:2], rows)
    with pytest.raises(OmeZarrError, match="path"):
        transformation_matrix({"type": "affine", "path": "x"}, 2)


def test_t5_unknown_type_warns_once_and_pixels_read(tmp_path):
    """T5: a field transformation is kept, warned about once, and pixels still read."""
    data, spacing = _cyx()
    path = tmp_path / "u"
    write_ome_zarr(path, data, axes=["c", "y", "x"], spacing=spacing, ngff_version="0.6")
    attrs = _attrs(path)
    ms = attrs["ome"]["multiscales"][0]
    ms["coordinateSystems"].append({"name": "warped", "axes": ms["coordinateSystems"][0]["axes"]})
    field = {"type": "displacements", "path": "coordinateTransformations/field",
             "input": {"name": "physical"}, "output": {"name": "warped"}}
    ms["coordinateTransformations"] = [field]
    _set_attrs(path, attrs)
    ome_zarr._WARNED_TRANSFORM_TYPES.discard("displacements")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = read_ome_zarr(path)
        read_ome_zarr(path)
    assert [str(w.message).count("displacements") for w in caught] == [1]
    assert first.transformations == (field,)
    assert np.array_equal(first.read(0), data)
    assert transformation_matrix(field, 3) is None


@pytest.mark.parametrize("mutate, key", [
    (lambda ms: ms["datasets"][0]["coordinateTransformations"][0].update(input="0"), "input"),
    (lambda ms: ms["datasets"][0]["coordinateTransformations"][0].update(output="physical"), "output"),
    (lambda ms: ms.update(coordinateTransformations=[{
        "type": "byDimension", "input": {"name": "physical"}, "output": {"name": "physical"},
        "transformations": [{"transformation": {"type": "identity"},
                             "input_axes": [0], "output_axes": [0]}]}]), "input_axes"),
    (lambda ms: ms.update(axes=ms.pop("coordinateSystems")[0]["axes"]), "coordinateSystems"),
    (lambda ms: ms.update(arrayCoordinateSystems=[]), "arrayCoordinateSystems"),
    (lambda ms: ms.update(coordinateTransformations=[{
        "type": "inverseOf", "input": {"name": "physical"}, "output": {"name": "physical"}}]),
     "inverseOf"),
])
def test_t6_old_spellings_name_the_key(tmp_path, mutate, key):
    """T6: a 0.6 store in a pre-0.6 spelling is refused, naming the key."""
    data, spacing = _cyx()
    path = tmp_path / "old"
    write_ome_zarr(path, data, axes=["c", "y", "x"], spacing=spacing, ngff_version="0.6")
    attrs = _attrs(path)
    mutate(attrs["ome"]["multiscales"][0])
    _set_attrs(path, attrs)
    with pytest.raises(OmeZarrError, match=key):
        read_ome_zarr(path)


def test_mappers_round_trip_and_refuse_what_they_cannot_map(tmp_path):
    """0.5 -> 0.6 -> 0.5 returns the original entry; unmappable keys are named."""
    data, spacing = _zyx()
    write_ome_zarr(tmp_path / "m", data, axes=["z", "y", "x"], spacing=spacing,
                   ngff_version="0.5", levels=3)
    ms = _attrs(tmp_path / "m")["ome"]["multiscales"][0]
    assert multiscale_to_ngff05(multiscale_to_ngff06(ms)) == ms
    single = json.loads(json.dumps(ms))
    for dataset in single["datasets"]:
        dataset["coordinateTransformations"] = dataset["coordinateTransformations"][:1]
    assert multiscale_to_ngff05(multiscale_to_ngff06(single)) == single
    for key, change in (("version", {"version": "0.5"}),
                        ("coordinateTransformations", {"coordinateTransformations": [
                            {"type": "scale", "scale": [1, 1, 1]}]})):
        with pytest.raises(OmeZarrError, match=key):
            multiscale_to_ngff06({**ms, **change})
    extra = multiscale_to_ngff06(ms)
    extra["coordinateSystems"].append({"name": "other", "axes": []})
    with pytest.raises(OmeZarrError, match="coordinate systems"):
        multiscale_to_ngff05(extra)


def test_t11_same_settings_give_identical_stores(tmp_path):
    """T11: writing the same image twice gives identical metadata and chunks."""
    data, spacing = _cyx()
    for name in ("a", "b"):
        write_ome_zarr(tmp_path / name, data, axes=["c", "y", "x"], spacing=spacing,
                       ngff_version="0.6", levels=2)
    assert _chunks_md5(tmp_path / "a") == _chunks_md5(tmp_path / "b")
    meta_a = [json.loads(p.read_text()) for p in sorted((tmp_path / "a").rglob("zarr.json"))]
    meta_b = [json.loads(p.read_text()) for p in sorted((tmp_path / "b").rglob("zarr.json"))]
    for left, right in zip(meta_a, meta_b):
        attrs_l, attrs_r = left.get("attributes", {}), right.get("attributes", {})
        for attrs in (attrs_l, attrs_r):
            for ms in attrs.get("ome", {}).get("multiscales", []):
                ms["name"] = "x"
        assert left == right


def test_t12_gate_is_open_for_06_only(tmp_path):
    """T12: 0.6 is released (2026-09-14), so it is read and written; others are not."""
    assert "0.6" in ome_zarr.SUPPORTED_NGFF_VERSIONS
    assert "0.6" in ome_zarr.WRITABLE_NGFF_VERSIONS
    data, spacing = _cyx()
    assert write_ome_zarr(tmp_path / "ok", data, spacing=spacing, axes=["c", "y", "x"],
                          ngff_version="0.6").ngff_version == "0.6"
    for version in ("0.6.dev4", "0.7", "0.9.dev2"):
        with pytest.raises(OmeZarrError, match="0.6"):
            write_ome_zarr(tmp_path / version, data, ngff_version=version)


def test_plate_store_writes_06_everywhere(tmp_path):
    """The plate store: version only at ome.version in every group, labels included."""
    from spacr import image_store

    writer = image_store.PlateWriter(tmp_path, "P1", channel_names=["a", "b"],
                                     ngff_version="0.6")
    image = np.arange(2 * 16 * 12, dtype=np.uint16).reshape(2, 16, 12)
    mask = (np.arange(16 * 12) % 5).astype(np.uint16).reshape(16, 12)
    key = writer.write_field("B03", 1, image, {"cell": mask})
    store = tmp_path / "P1.ome.zarr"
    for group in (store, store / "B", store / "B" / "3", store / key,
                  store / key / "labels", store / key / "labels" / "cell"):
        ome = _attrs(group).get("ome", {})
        if ome:
            assert ome["version"] == "0.6", group
            for sub in ("multiscales", "plate", "well", "labels", "image-label"):
                block = ome.get(sub)
                entries = block if isinstance(block, list) else [block]
                assert all(not isinstance(e, dict) or "version" not in e
                           for e in entries), (group, sub)
    assert np.array_equal(image_store.read_image(store, key=key), image)
    assert np.array_equal(image_store.read_labels(store, "cell", key=key), mask)
    assert read_ome_zarr(store / key).ngff_version == "0.6"
    with pytest.raises(ValueError, match="0.6"):
        image_store.PlateWriter(tmp_path, "P1")
    with pytest.raises(ValueError, match="0.7"):
        image_store.PlateWriter(tmp_path, "P2", ngff_version="0.7")


def test_plate_store_default_stays_05(tmp_path):
    """The plate store default is unchanged: 0.5, with image-label version."""
    from spacr import image_store

    writer = image_store.PlateWriter(tmp_path, "P1")
    key = writer.write_field("A01", 0, np.zeros((8, 8), np.uint16),
                             {"nucleus": np.ones((8, 8), np.uint16)})
    store = tmp_path / "P1.ome.zarr"
    assert _attrs(store)["ome"]["version"] == "0.5"
    assert _attrs(store / key / "labels" / "nucleus")["ome"]["image-label"] == {"version": "0.5"}


def test_ops_setting_resolves_and_is_alpha():
    """ome_zarr_ngff_version: OPS default 0.5, 0.6 accepted, others refused, alpha."""
    from spacr import ops_engine, ops_settings, settings

    assert ops_settings.OPS_DEFAULTS["ome_zarr_ngff_version"] == "0.5"
    assert ops_engine._ph_ngff_version({}) == "0.5"
    assert ops_engine._ph_ngff_version({"ome_zarr_ngff_version": "0.6"}) == "0.6"
    with pytest.raises(ValueError, match="0.5 or 0.6"):
        ops_engine._ph_ngff_version({"ome_zarr_ngff_version": "0.4"})
    assert "ome_zarr_ngff_version" in settings.ALPHA_FEATURES[1007]["settings"]
