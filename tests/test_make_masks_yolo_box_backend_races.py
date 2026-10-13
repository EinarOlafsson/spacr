"""Exercise metadata and lock races without changing annotation source files."""

from __future__ import annotations

import os

import pytest

from spacr.qt import mask_engine as engine


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO needs POSIX")
def test_metadata_changed_to_fifo_between_check_and_open_is_refused(
        tmp_path, monkeypatch):
    source = tmp_path / "field.tif"
    source.write_bytes(b"original image")
    ledger = engine.save_yolo_boxes(tmp_path, source.name, (10, 10), [],
                                    ["object"])
    original = tmp_path / "original-ledger.json"
    actual_open = os.open
    swapped = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == ledger and not swapped:
            ledger.replace(original)
            os.mkfifo(ledger)
            swapped = True
        return actual_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(engine.os, "open", swap_before_open)
    with pytest.raises(ValueError, match="regular file"):
        engine.load_yolo_boxes(tmp_path, source.name, (10, 10))

    assert swapped
    assert original.is_file()
    assert ledger.exists() and not ledger.is_file()


@pytest.mark.parametrize("replacement", [False, True])
def test_lock_cleanup_does_not_erase_a_disappeared_or_replaced_lock(
        tmp_path, monkeypatch, replacement):
    source = tmp_path / "field.tif"
    source.write_bytes(b"original image")
    ledger = tmp_path / engine.YOLO_ANNOTATIONS_NAME
    lock = tmp_path / (engine.YOLO_ANNOTATIONS_NAME + ".lock")
    actual_atomic = engine._yolo_atomic

    def change_lock_before_write(path, payload):
        assert path == ledger
        assert lock.is_file()
        lock.unlink()
        if replacement:
            lock.write_text("different writer")
        return actual_atomic(path, payload)

    monkeypatch.setattr(engine, "_yolo_atomic", change_lock_before_write)
    assert engine.save_yolo_boxes(tmp_path, source.name, (10, 10), [],
                                  ["object"]) == ledger

    assert ledger.is_file()
    if replacement:
        assert lock.read_text() == "different writer"
    else:
        assert not lock.exists()


def test_export_refuses_directory_at_label_destination(tmp_path):
    label = tmp_path / "field.txt"
    label.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        engine.export_yolo_boxes(label, [], (10, 10), ["object"])

    assert label.is_dir()
    assert not (tmp_path / engine.YOLO_CLASSES_NAME).exists()
    assert not (tmp_path / (engine.YOLO_CLASSES_NAME + ".lock")).exists()
