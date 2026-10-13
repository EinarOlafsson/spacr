"""Refuse damaged box metadata and failed writes without losing annotations."""

from __future__ import annotations

import json

import pytest

from spacr.qt import mask_engine as engine


@pytest.mark.parametrize("shape,boxes,classes,message", [
    (None, [], ["object"], "shape"),
    ((10, 10), None, ["object"], "sequence"),
    ((10, 10), [(0, 1, 2)], ["object"], "four corners"),
    ((10, 10), [(0, "left", 0, 2, 2)], ["object"], "finite numbers"),
    ((10, 10), [], "object", "sequence"),
    ((10, 10), [], None, "sequence"),
])
def test_invalid_annotation_inputs_leave_no_project_ledger(
        tmp_path, shape, boxes, classes, message):
    (tmp_path / "field.tif").write_bytes(b"source image")

    with pytest.raises(ValueError, match=message):
        engine.save_yolo_boxes(tmp_path, "field.tif", shape, boxes, classes)

    assert not (tmp_path / engine.YOLO_ANNOTATIONS_NAME).exists()


def test_source_folder_and_source_entry_must_be_files_of_the_right_kind(tmp_path):
    project_file = tmp_path / "project"
    project_file.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="not a directory"):
        engine.load_yolo_boxes(project_file, "field.tif", (10, 10))

    (tmp_path / "field.tif").mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        engine.load_yolo_boxes(tmp_path, "field.tif", (10, 10))
    assert not (tmp_path / engine.YOLO_ANNOTATIONS_NAME).exists()


@pytest.mark.parametrize("ledger", [
    "{",
    {"version": 2, "classes": ["object"], "images": {}},
    {"version": 1, "classes": ["object"], "images": {
        "field.tif": {"shape": [10, 10], "source_sha256": "x" * 64,
                      "boxes": []}}},
    {"version": 1, "classes": ["object"], "images": {
        "field.tif": {"shape": [10, 10], "source_sha256": "0" * 64,
                      "boxes": [[0, 8, 8, 2, 2]]}}},
])
def test_corrupt_project_ledger_is_never_replaced(tmp_path, ledger):
    (tmp_path / "field.tif").write_bytes(b"source image")
    path = tmp_path / engine.YOLO_ANNOTATIONS_NAME
    path.write_text(ledger if isinstance(ledger, str) else json.dumps(ledger))
    before = path.read_bytes()

    with pytest.raises(ValueError):
        engine.load_yolo_boxes(tmp_path, "field.tif", (10, 10))
    with pytest.raises(ValueError):
        engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10), [], ["object"])

    assert path.read_bytes() == before


def test_export_refuses_bad_paths_symlinks_and_corrupt_class_metadata(tmp_path):
    boxes = [(0, 0, 0, 10, 10)]
    classes = ["object"]
    for path in (tmp_path / "field.csv", tmp_path / "absent/field.txt"):
        with pytest.raises(ValueError, match=".txt path"):
            engine.export_yolo_boxes(path, boxes, (10, 10), classes)

    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    label = tmp_path / "field.txt"
    label.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        engine.export_yolo_boxes(label, boxes, (10, 10), classes)
    assert outside.read_bytes() == b"keep"
    label.unlink()

    metadata = tmp_path / engine.YOLO_CLASSES_NAME
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        engine.export_yolo_boxes(label, boxes, (10, 10), classes)
    assert outside.read_bytes() == b"keep"
    metadata.unlink()

    metadata.write_text('{"classes": ["object"], "unexpected": true}')
    before = metadata.read_bytes()
    with pytest.raises(ValueError, match="class metadata"):
        engine.export_yolo_boxes(label, boxes, (10, 10), classes)
    assert metadata.read_bytes() == before
    assert not label.exists()


def test_temp_creation_failure_keeps_saved_boxes(tmp_path, monkeypatch):
    (tmp_path / "field.tif").write_bytes(b"source image")
    ledger = engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10), [],
                                    ["object"])
    before = ledger.read_bytes()

    def no_temp_file(**_kwargs):
        raise OSError("scratch directory is full")

    monkeypatch.setattr(engine.tempfile, "NamedTemporaryFile", no_temp_file)
    with pytest.raises(OSError, match="scratch directory is full"):
        engine.save_yolo_boxes(tmp_path, "field.tif", (10, 10),
                               [(0, 1, 1, 2, 2)], ["object"])

    assert ledger.read_bytes() == before
    assert not list(tmp_path.glob(f".{ledger.name}.*.tmp"))
