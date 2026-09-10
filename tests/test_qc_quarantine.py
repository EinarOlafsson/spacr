"""A quarantined field is excluded, reversible, and always auditable."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spacr import qc_quarantine


def _plate(tmp_path: Path):
    merged = tmp_path / "plate1" / "merged"
    masks = tmp_path / "plate1" / "norm_channel_stack" / "cell_mask_stack"
    merged.mkdir(parents=True)
    masks.mkdir(parents=True)
    np.save(merged / "plate1_A01_1.npy", np.arange(24).reshape(2, 3, 4))
    np.save(masks / "plate1_A01_1.npy", np.array([[0, 1], [0, 2]]))
    return merged, masks


def test_quarantine_and_restore_round_trip_the_array_and_leave_masks(tmp_path):
    merged, masks = _plate(tmp_path)
    field = "plate1_A01_1"
    original = (merged / f"{field}.npy").read_bytes()
    mask_bytes = (masks / f"{field}.npy").read_bytes()

    moved = qc_quarantine.quarantine_field(
        merged, field, flags=["cell:empty_field", "cell:empty_field"],
        who="reviewer")

    assert moved == tmp_path / "plate1" / "merged_quarantined" / f"{field}.npy"
    assert moved.read_bytes() == original
    assert not (merged / f"{field}.npy").exists()
    assert (masks / f"{field}.npy").read_bytes() == mask_bytes
    assert list(merged.glob("*.npy")) == []
    assert qc_quarantine.is_quarantined(merged, field)
    assert qc_quarantine.list_quarantined(merged) == [field]

    record_path = qc_quarantine.quarantine_record_path(moved.parent, field)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["field"] == field
    assert record["quarantined_by"] == "reviewer"
    assert record["qc_flags"] == ["cell:empty_field"]
    assert record["quarantined_at"].endswith("+00:00")
    assert record["events"][-1]["action"] == "quarantined"

    restored = qc_quarantine.restore_field(
        moved.parent, field, who="second reviewer")

    assert restored == merged / f"{field}.npy"
    assert restored.read_bytes() == original
    assert not moved.exists()
    assert (masks / f"{field}.npy").read_bytes() == mask_bytes
    assert not qc_quarantine.is_quarantined(merged, field)
    assert qc_quarantine.list_quarantined(merged) == []
    updated = json.loads(record_path.read_text(encoding="utf-8"))
    assert updated["restored_by"] == "second reviewer"
    assert [event["action"] for event in updated["events"]] == [
        "quarantined", "restored"]


def test_quarantine_never_overwrites_an_existing_copy(tmp_path):
    merged, _masks = _plate(tmp_path)
    quarantine = qc_quarantine.quarantine_dir_for(merged)
    quarantine.mkdir()
    existing = quarantine / "plate1_A01_1.npy"
    existing.write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        qc_quarantine.quarantine_field(merged, "plate1_A01_1")

    assert existing.read_bytes() == b"do not overwrite"
    assert (merged / "plate1_A01_1.npy").is_file()


def test_restore_never_overwrites_a_regenerated_active_field(tmp_path):
    merged, _masks = _plate(tmp_path)
    moved = qc_quarantine.quarantine_field(merged, "plate1_A01_1")
    (merged / "plate1_A01_1.npy").write_bytes(b"new segmentation")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        qc_quarantine.restore_field(moved.parent, "plate1_A01_1")

    assert moved.is_file()
    assert (merged / "plate1_A01_1.npy").read_bytes() == b"new segmentation"


def test_a_sidecar_write_failure_rolls_the_field_move_back(tmp_path, monkeypatch):
    merged, _masks = _plate(tmp_path)

    def fail(_path, _record):
        raise OSError("disk full")

    monkeypatch.setattr(qc_quarantine, "_write_record", fail)
    with pytest.raises(qc_quarantine.QuarantineError, match="field restored"):
        qc_quarantine.quarantine_field(merged, "plate1_A01_1")

    assert (merged / "plate1_A01_1.npy").is_file()
    assert not qc_quarantine.is_quarantined(merged, "plate1_A01_1")


def test_a_restore_ledger_failure_leaves_the_field_quarantined(
        tmp_path, monkeypatch):
    merged, _masks = _plate(tmp_path)
    moved = qc_quarantine.quarantine_field(merged, "plate1_A01_1")

    def fail(_path, _record):
        raise OSError("read-only ledger")

    monkeypatch.setattr(qc_quarantine, "_write_record", fail)
    with pytest.raises(
            qc_quarantine.QuarantineError,
            match="field remains quarantined"):
        qc_quarantine.restore_field(moved.parent, "plate1_A01_1")

    assert moved.is_file()
    assert not (merged / "plate1_A01_1.npy").exists()


@pytest.mark.parametrize("field", ["../outside", "a/b", r"a\b", "", ".npy"])
def test_a_scorecard_field_can_never_escape_the_plate(tmp_path, field):
    merged, _masks = _plate(tmp_path)
    with pytest.raises(ValueError):
        qc_quarantine.quarantine_field(merged, field)


def test_resolve_finds_active_then_quarantined_and_missing(tmp_path):
    merged, _masks = _plate(tmp_path)
    field = "plate1_A01_1"
    assert qc_quarantine.resolve_field_path(merged, field) == merged / f"{field}.npy"
    moved = qc_quarantine.quarantine_field(merged, field)
    assert qc_quarantine.resolve_field_path(merged, field) == moved
    qc_quarantine.restore_field(moved.parent, field)
    (merged / f"{field}.npy").unlink()
    assert qc_quarantine.resolve_field_path(merged, field) is None
