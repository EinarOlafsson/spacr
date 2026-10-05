"""YOLO box annotations stay separate from masks and bind to source bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spacr.qt import mask_engine as engine


def _source(folder: Path, name="field.tif", content=b"original image") -> Path:
    """Create an unchanged image-byte witness for box persistence."""
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_lines_use_exclusive_edges_and_keep_overlapping_classes():
    boxes = [(2, 40, 30, 10, 10), (0, -10, 0, 25, 20),
             (1, 15, 10, 45, 30)]
    lines = engine.yolo_box_lines(boxes, 100, 50)
    assert len(lines) == 3
    assert [int(line.split()[0]) for line in lines] == [2, 0, 1]
    assert [float(v) for v in lines[0].split()[1:]] == pytest.approx(
        [0.25, 0.4, 0.3, 0.4])
    assert [float(v) for v in lines[1].split()[1:]] == pytest.approx(
        [0.125, 0.2, 0.25, 0.4])
    assert [float(v) for v in lines[2].split()[1:]] == pytest.approx(
        [0.3, 0.4, 0.3, 0.4])


@pytest.mark.parametrize("boxes,width,height", [
    ([(0, 0, 0, 1, 1)], 0, 10),
    ([(0, 0, 0, 1, 1)], 10, -1),
    ([(0, 1, 1, 1, 2)], 10, 10),
    ([(0, 11, 1, 12, 2)], 10, 10),
    ([(-1, 0, 0, 1, 1)], 10, 10),
    ([(1.5, 0, 0, 1, 1)], 10, 10),
    ([(True, 0, 0, 1, 1)], 10, 10),
    ([(0, 0, 0, float("nan"), 2)], 10, 10),
    ([(0, 0, 0, float("inf"), 2)], 10, 10),
])
def test_invalid_geometry_or_class_is_refused(boxes, width, height):
    with pytest.raises(ValueError):
        engine.yolo_box_lines(boxes, width, height)


def test_new_field_defaults_and_saved_boxes_bind_to_original_bytes(tmp_path):
    source = _source(tmp_path)
    before = source.read_bytes()
    fresh = engine.load_yolo_boxes(tmp_path, source.name, (50, 100))
    assert fresh == {"classes": ["object"], "boxes": [],
                     "source_sha256": hashlib.sha256(before).hexdigest()}
    path = engine.save_yolo_boxes(
        tmp_path, source.name, (50, 100), [(1, 40, 30, 10, 10)],
        ["cell", "nucleus"], expected_source_sha256=fresh["source_sha256"])
    assert path == tmp_path / engine.YOLO_ANNOTATIONS_NAME
    assert source.read_bytes() == before
    loaded = engine.load_yolo_boxes(tmp_path, source.name, (50, 100))
    assert loaded == {"classes": ["cell", "nucleus"],
                      "boxes": [(1, 10.0, 10.0, 40.0, 30.0)],
                      "source_sha256": fresh["source_sha256"]}
    record = json.loads(path.read_text())["images"][source.name]
    assert record["shape"] == [50, 100]
    assert record["source_sha256"] == fresh["source_sha256"]


def test_class_ids_stay_stable_across_fields_and_map_extensions(tmp_path):
    _source(tmp_path, "a.tif")
    _source(tmp_path, "nested/b.tif")
    engine.save_yolo_boxes(tmp_path, "a.tif", (10, 10),
                           [(0, 1, 1, 5, 5)], ["cell"])
    engine.save_yolo_boxes(tmp_path, "nested/b.tif", (10, 10),
                           [(1, 2, 2, 6, 6)], ["cell", "nucleus"])
    assert engine.load_yolo_boxes(tmp_path, "a.tif", (10, 10))["classes"] == [
        "cell", "nucleus"]
    original = (tmp_path / engine.YOLO_ANNOTATIONS_NAME).read_bytes()
    for classes in (["other", "nucleus"], ["cell"],
                    ["cell", "cell"]):
        with pytest.raises(ValueError):
            engine.save_yolo_boxes(tmp_path, "a.tif", (10, 10), [], classes)
    with pytest.raises(ValueError):
        engine.save_yolo_boxes(tmp_path, "a.tif", (10, 10),
                               [(2, 0, 0, 1, 1)], ["cell", "nucleus"])
    assert (tmp_path / engine.YOLO_ANNOTATIONS_NAME).read_bytes() == original


def test_changed_source_or_shape_refuses_load_and_save(tmp_path):
    source = _source(tmp_path)
    observed = engine.load_yolo_boxes(tmp_path, source.name, (10, 10))
    source.write_bytes(b"changed before first save")
    with pytest.raises(ValueError, match="changed"):
        engine.save_yolo_boxes(tmp_path, source.name, (10, 10), [], ["object"],
                               expected_source_sha256=observed["source_sha256"])
    assert not (tmp_path / engine.YOLO_ANNOTATIONS_NAME).exists()
    digest = engine.load_yolo_boxes(tmp_path, source.name, (10, 10))[
        "source_sha256"]
    engine.save_yolo_boxes(tmp_path, source.name, (10, 10), [], ["object"],
                           expected_source_sha256=digest)
    with pytest.raises(ValueError, match="shape"):
        engine.load_yolo_boxes(tmp_path, source.name, (11, 10))
    with pytest.raises(ValueError, match="shape"):
        engine.save_yolo_boxes(tmp_path, source.name, (11, 10), [], ["object"],
                               expected_source_sha256=digest)
    source.write_bytes(b"changed after save")
    with pytest.raises(ValueError, match="changed"):
        engine.load_yolo_boxes(tmp_path, source.name, (10, 10))
    with pytest.raises(ValueError, match="changed"):
        engine.save_yolo_boxes(tmp_path, source.name, (10, 10), [], ["object"])


def test_path_escape_and_corrupted_ledger_are_refused(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    _source(inside)
    outside = _source(tmp_path, "outside.tif")
    (inside / "link.tif").symlink_to(outside)
    for name in ("../outside.tif", "link.tif", str(outside)):
        with pytest.raises(ValueError):
            engine.load_yolo_boxes(inside, name, (10, 10))
    ledger = inside / engine.YOLO_ANNOTATIONS_NAME
    ledger.write_text('{"version":1,"classes":["object"],"images":{},"images":{}}')
    with pytest.raises(ValueError, match="duplicate"):
        engine.load_yolo_boxes(inside, "field.tif", (10, 10))
    ledger.write_text('{"version":1,"classes":["object"],"images":'
                      '{"../outside.tif":{"shape":[10,10],'
                      '"source_sha256":"' + "0" * 64 + '","boxes":[]}}}')
    with pytest.raises(ValueError, match="record"):
        engine.save_yolo_boxes(inside, "field.tif", (10, 10), [], ["object"])


def test_project_write_failure_keeps_previous_ledger_and_cleans_temp(tmp_path,
                                                                      monkeypatch):
    _source(tmp_path)
    ledger = engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10), [],
                                    ["object"])
    original = ledger.read_bytes()

    def fail_replace(_source_path, _target_path):
        raise OSError("replace failed")

    monkeypatch.setattr(engine.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10),
                               [(0, 1, 1, 2, 2)], ["object"])
    assert ledger.read_bytes() == original
    assert not list(tmp_path.glob(f".{ledger.name}.*.tmp"))


def test_export_writes_only_labels_and_stable_class_metadata(tmp_path):
    source = _source(tmp_path)
    before = source.read_bytes()
    labels = tmp_path / "field.txt"
    assert engine.export_yolo_boxes(labels, [(1, 0, 0, 10, 10)], (20, 20),
                                    ["cell", "nucleus"]) == str(labels)
    assert labels.read_text().splitlines() == ["1 0.25 0.25 0.5 0.5"]
    meta = tmp_path / engine.YOLO_CLASSES_NAME
    assert json.loads(meta.read_text()) == {"classes": ["cell", "nucleus"]}
    assert source.read_bytes() == before
    assert engine.export_yolo_boxes(labels, [], (20, 20),
                                    ["cell", "nucleus"]) == str(labels)
    assert labels.read_bytes() == b""
    with pytest.raises(ValueError, match="class map"):
        engine.export_yolo_boxes(labels, [], (20, 20), ["other", "nucleus"])
    assert labels.read_bytes() == b""


def test_export_rejects_invalid_boxes_without_writing(tmp_path):
    labels = tmp_path / "field.txt"
    with pytest.raises(ValueError, match="class ID"):
        engine.export_yolo_boxes(labels, [(1, 0, 0, 10, 10)], (20, 20),
                                 ["object"])
    assert not labels.exists()
    assert not (tmp_path / engine.YOLO_CLASSES_NAME).exists()


@pytest.mark.parametrize("classes", [
    [], [""], [" cell"], ["cell\nnext"], ["../cell"],
    ["cell", "CELL"],
])
def test_invalid_class_names_cannot_reach_the_ledger(tmp_path, classes):
    _source(tmp_path)
    with pytest.raises(ValueError):
        engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10), [], classes)
    assert not (tmp_path / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_export_failed_replace_keeps_existing_labels_and_cleans_temp(tmp_path,
                                                                     monkeypatch):
    labels = tmp_path / "field.txt"
    engine.export_yolo_boxes(labels, [(0, 0, 0, 10, 10)], (10, 10),
                             ["object"])
    original = labels.read_bytes()
    actual_replace = engine.os.replace

    def fail_label_replace(source_path, target_path):
        if Path(target_path) == labels:
            raise OSError("label replace failed")
        return actual_replace(source_path, target_path)

    monkeypatch.setattr(engine.os, "replace", fail_label_replace)
    with pytest.raises(OSError, match="label replace failed"):
        engine.export_yolo_boxes(labels, [], (10, 10), ["object", "nucleus"])
    assert labels.read_bytes() == original
    assert json.loads((tmp_path / engine.YOLO_CLASSES_NAME).read_text()) == {
        "classes": ["object", "nucleus"]}
    assert not list(tmp_path.glob(".field.txt.*.tmp"))


def test_a_broken_project_ledger_symlink_is_refused(tmp_path):
    _source(tmp_path)
    ledger = tmp_path / engine.YOLO_ANNOTATIONS_NAME
    ledger.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symbolic link"):
        engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10), [], ["object"])
    assert ledger.is_symlink()
