"""Atomic checkpoint-store contracts shared by resumable workflows."""
from __future__ import annotations

import json

import pytest

from spacr.checkpoint import (
    CheckpointError,
    CheckpointMismatch,
    CheckpointStore,
    fingerprint,
)


def test_checkpoint_marks_units_and_restores_metadata(tmp_path):
    path = tmp_path / "run.json"
    store = CheckpointStore(
        path, workflow="demo", signature={"input": "plate1"},
        boundary="field")
    store.mark("plate1/A01/f0001", {"targets": ["one.tif"]},
               meta={"last_field": 1})
    store.update(meta={"centre": [5, 0.1]}, status="partial")

    resumed = CheckpointStore(
        path, workflow="demo", signature={"input": "plate1"},
        boundary="field", resume=True)

    assert resumed.resumed
    assert resumed.status == "partial"
    assert resumed.completed["plate1/A01/f0001"]["targets"] == ["one.tif"]
    assert resumed.meta == {"last_field": 1, "centre": [5, 0.1]}


def test_checkpoint_refuses_different_inputs_or_workflow(tmp_path):
    path = tmp_path / "run.json"
    CheckpointStore(
        path, workflow="convert", signature={"src": "plate1"},
        boundary="field")

    with pytest.raises(CheckpointMismatch, match="does not match"):
        CheckpointStore(
            path, workflow="convert", signature={"src": "plate2"},
            boundary="field", resume=True)
    with pytest.raises(CheckpointMismatch, match="belongs to"):
        CheckpointStore(
            path, workflow="umap", signature={"src": "plate1"},
            boundary="trial", resume=True)


def test_corrupt_checkpoint_fails_loudly_and_is_not_overwritten(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(CheckpointError, match="could not be read"):
        CheckpointStore(
            path, workflow="demo", signature={}, boundary="field",
            resume=True)
    assert path.read_text(encoding="utf-8") == "{broken"


def test_fresh_run_replaces_stale_state_without_resume(tmp_path):
    path = tmp_path / "run.json"
    old = CheckpointStore(
        path, workflow="demo", signature={"v": 1}, boundary="field")
    old.mark("one")

    fresh = CheckpointStore(
        path, workflow="demo", signature={"v": 2}, boundary="field",
        resume=False)

    assert fresh.completed == {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["signature"] == fingerprint({"v": 2})
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_artifact_names_never_embed_the_unit_path(tmp_path):
    store = CheckpointStore(
        tmp_path / "run.json", workflow="demo", signature={},
        boundary="trial")
    artifact = store.artifact_path("../../plate/A01", ".npy")

    assert artifact.parent == store.artifact_dir
    assert artifact.suffix == ".npy"
    assert ".." not in artifact.name
