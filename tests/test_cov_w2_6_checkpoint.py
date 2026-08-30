"""A checkpoint that cannot be trusted refuses to resume rather than guess.

Everything here writes real JSON to a real directory: the point of the module
is the file on disk, and a stubbed writer would assert that the stub was
called. The failure cases are produced by damaging that file, or by making
the filesystem call underneath the write fail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spacr import checkpoint as cp


@pytest.fixture
def store(tmp_path):
    return cp.CheckpointStore(tmp_path / "run.json", workflow="umap_search",
                              signature={"src": "/data", "cells": 10},
                              boundary="trial")


# --------------------------------------------------------------------------
# json_safe
# --------------------------------------------------------------------------

def test_a_path_is_recorded_as_its_text_not_its_repr():
    """A checkpoint is meant to be inspectable without importing the workflow
    that produced it, and `PosixPath('/data')` is not a path."""
    assert cp.json_safe(Path("/data/plate1")) == "/data/plate1"


def test_a_set_is_recorded_in_a_stable_order():
    """Two runs whose settings differ only in set iteration order must
    produce the same signature, or every resume is refused."""
    once = cp.json_safe({"channels", "masks", "arrays"})
    again = cp.json_safe({"masks", "arrays", "channels"})
    assert once == again == sorted(once, key=repr)


def test_a_numpy_scalar_is_recorded_as_the_number_it_holds():
    numpy = pytest.importorskip("numpy")
    assert cp.json_safe(numpy.float32(0.5)) == 0.5
    assert cp.json_safe(numpy.int64(7)) == 7


def test_an_object_whose_item_refuses_falls_back_to_its_text():
    """`.item` is how a NumPy scalar is unwrapped without importing NumPy.
    Something else that happens to carry the name must not break a write."""

    class HasAwkwardItem:
        def item(self):
            raise TypeError("not a scalar")

        def __str__(self):
            return "awkward"

    assert cp.json_safe(HasAwkwardItem()) == "awkward"


def test_a_noncallable_item_attribute_is_not_treated_as_a_scalar_converter():
    """Metadata called ``item`` is ordinary data, not necessarily a method."""

    class HasItemMetadata:
        item = "inventory label"

        def __str__(self):
            return "labelled object"

    assert cp.json_safe(HasItemMetadata()) == "labelled object"


def test_mapping_keys_are_sorted_as_strings_so_the_digest_is_stable():
    assert list(cp.json_safe({2: "b", 10: "a", "1": "c"})) == ["1", "10", "2"]


def test_two_equivalent_signatures_share_one_digest():
    assert cp.fingerprint({"a": 1, "b": [1, 2]}) == \
        cp.fingerprint({"b": (1, 2), "a": 1})
    assert cp.fingerprint({"a": 1}) != cp.fingerprint({"a": 2})


# --------------------------------------------------------------------------
# the ordinary lifecycle
# --------------------------------------------------------------------------

def test_a_fresh_store_writes_a_document_that_names_its_boundary(tmp_path):
    store = cp.CheckpointStore(tmp_path / "run.json", workflow="mask",
                               signature="s" * 64, boundary="field")
    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["boundary"] == "field"
    assert payload["workflow"] == "mask"
    assert payload["signature"] == "s" * 64      # a 64-char digest is kept
    assert payload["status"] == "running"
    assert store.resumed is False


def test_a_completed_unit_is_on_disk_before_the_call_returns(store):
    store.mark("trial_1", {"score": 0.9}, meta={"round": 2})
    payload = json.loads(store.path.read_text())
    assert payload["completed"]["trial_1"] == {"score": 0.9}
    assert payload["meta"]["round"] == 2
    assert store.completed["trial_1"] == {"score": 0.9}
    assert store.meta == {"round": 2}


def test_a_finished_workflow_keeps_its_state_and_says_it_is_done(store):
    store.mark("trial_1", {"score": 0.9})
    store.finish(meta={"best": "trial_1"})
    assert store.status == "complete"
    assert store.meta == {"best": "trial_1"}
    assert store.completed["trial_1"] == {"score": 0.9}


def test_a_resume_with_the_same_signature_finds_the_completed_units(tmp_path):
    signature = {"src": "/data", "cells": 10}
    first = cp.CheckpointStore(tmp_path / "run.json", workflow="umap_search",
                               signature=signature, boundary="trial")
    first.mark("trial_1", {"score": 0.9})
    again = cp.CheckpointStore(tmp_path / "run.json", workflow="umap_search",
                               signature=signature, boundary="trial",
                               resume=True)
    assert again.resumed is True
    assert again.completed == {"trial_1": {"score": 0.9}}


def test_an_artifact_path_is_a_digest_not_the_unit_name(store):
    path = store.artifact_path("plate1/A01 field 3")
    assert path.parent == store.artifact_dir
    assert path.parent.is_dir()
    assert "A01" not in path.name and path.suffix == ".npy"
    assert store.artifact_path("plate1/A01 field 3") == path
    assert store.artifact_path("u", "npz").suffix == ".npz"


# --------------------------------------------------------------------------
# every way a resume is refused
# --------------------------------------------------------------------------

def _write(path, payload):
    Path(path).write_text(json.dumps(payload))


def test_a_checkpoint_from_a_future_format_is_refused_by_version(tmp_path):
    path = tmp_path / "run.json"
    _write(path, {"version": 99, "workflow": "w", "signature": "x" * 64,
                  "boundary": "field", "completed": {}, "meta": {}})
    with pytest.raises(cp.CheckpointMismatch, match="version"):
        cp.CheckpointStore(path, workflow="w", signature="x" * 64, boundary="field",
                           resume=True)


def test_a_checkpoint_from_another_workflow_is_refused_by_name(tmp_path):
    path = tmp_path / "run.json"
    _write(path, {"version": cp.CHECKPOINT_VERSION, "workflow": "measure",
                  "signature": "x" * 64, "boundary": "field",
                  "completed": {}, "meta": {}})
    with pytest.raises(cp.CheckpointMismatch, match="measure"):
        cp.CheckpointStore(path, workflow="mask", signature="x" * 64,
                           boundary="field", resume=True)


def test_changed_settings_refuse_to_be_combined_with_the_old_units(tmp_path):
    path = tmp_path / "run.json"
    cp.CheckpointStore(path, workflow="w", signature={"cells": 10},
                       boundary="field").mark("f1")
    with pytest.raises(cp.CheckpointMismatch, match="material settings"):
        cp.CheckpointStore(path, workflow="w", signature={"cells": 20},
                           boundary="field", resume=True)


def test_a_json_document_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "run.json"
    _write(path, [1, 2, 3])
    with pytest.raises(cp.CheckpointError, match="not a JSON object"):
        cp.CheckpointStore(path, workflow="w", signature="x" * 64,
                           boundary="field", resume=True)


def test_a_corrupt_checkpoint_says_to_keep_it_for_diagnosis(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{not json")
    with pytest.raises(cp.CheckpointError, match="Keep it"):
        cp.CheckpointStore(path, workflow="w", signature="x" * 64,
                           boundary="field", resume=True)


def test_a_completed_table_that_is_not_a_table_is_refused(tmp_path):
    path = tmp_path / "run.json"
    _write(path, {"version": cp.CHECKPOINT_VERSION, "workflow": "w",
                  "signature": "x" * 64, "boundary": "field",
                  "completed": ["f1"], "meta": {}})
    with pytest.raises(cp.CheckpointError, match="completed-unit table"):
        cp.CheckpointStore(path, workflow="w", signature="x" * 64,
                           boundary="field", resume=True)


def test_workflow_metadata_that_is_not_a_mapping_is_refused(tmp_path):
    path = tmp_path / "run.json"
    _write(path, {"version": cp.CHECKPOINT_VERSION, "workflow": "w",
                  "signature": "x" * 64, "boundary": "field",
                  "completed": {}, "meta": "round two"})
    with pytest.raises(cp.CheckpointError, match="invalid workflow metadata"):
        cp.CheckpointStore(path, workflow="w", signature="x" * 64,
                           boundary="field", resume=True)


# --------------------------------------------------------------------------
# a write that cannot complete stops the workflow
# --------------------------------------------------------------------------

def test_a_write_that_cannot_reach_the_disk_stops_the_workflow(store,
                                                               monkeypatch):
    """The alternative is a workflow that keeps going while believing it can
    be resumed from a checkpoint that was never written."""
    def _no_disk(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", _no_disk)
    with pytest.raises(cp.CheckpointError, match="could not be written"):
        store.mark("trial_1", {"score": 0.9})


def test_updating_meta_alone_leaves_the_status_where_it_was(store):
    """``status=None`` means "I am not saying", not "clear it".

    ``update`` is how a workflow records progress metadata mid-unit, and it
    is called far more often with meta alone than with both. Were the None
    written through, every metadata update would blank the status a resume
    reads to decide whether the run finished.
    """
    store.update(status="running")

    store.update(meta={"stage": "segmentation"})

    document = json.loads(store.path.read_text())
    assert document["status"] == "running"
    assert document["meta"]["stage"] == "segmentation"


def test_a_failed_write_leaves_no_half_written_temporary_behind(store,
                                                               monkeypatch):
    def _no_disk(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", _no_disk)
    with pytest.raises(cp.CheckpointError):
        store.update(status="stopped")
    leftovers = [p for p in store.path.parent.iterdir()
                 if p.name.endswith(".tmp")]
    assert leftovers == []
    # The previous, complete document is still readable.
    assert json.loads(store.path.read_text())["status"] == "running"


def test_a_temporary_that_cannot_be_removed_does_not_mask_the_real_failure(
        store, monkeypatch):
    """The write failure is what the user has to be told about; failing to
    tidy up after it is not."""
    def _no_disk(_fd):
        raise OSError(28, "No space left on device")

    def _cannot_unlink(_path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "fsync", _no_disk)
    monkeypatch.setattr(os, "unlink", _cannot_unlink)
    with pytest.raises(cp.CheckpointError, match="No space left"):
        store.flush()
