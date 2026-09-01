"""Direct drivers for the last small utils/timelapse coverage branches."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import spacr.timelapse as timelapse
import spacr.utils as utils


class _RecordingCellposeModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_choose_model_none_device_keeps_the_resolved_accelerator(monkeypatch):
    monkeypatch.setattr(utils.cp_models, "CellposeModel", _RecordingCellposeModel)
    monkeypatch.setattr(
        "spacr.accelerator.cellpose_kwargs",
        lambda: {"gpu": True, "device": "mps"},
    )

    model = utils._choose_model("cpsam", device=None)

    assert model.kwargs == {
        "pretrained_model": "cpsam",
        "gpu": True,
        "device": "mps",
    }


def test_reduction_rejects_gpu_for_isomap_before_backend_imports():
    with pytest.raises(
        ValueError,
        match=r"GPU acceleration is not available for isomap",
    ):
        utils.reduction_and_clustering(
            np.zeros((3, 2)),
            n_neighbors=2,
            min_dist=0.1,
            metric="euclidean",
            eps=0.5,
            min_samples=2,
            clustering="dbscan",
            reduction_method="isomap",
            prefer_gpu=True,
        )


def test_verified_delete_returns_the_matched_count():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute('CREATE TABLE cell ("fieldID" TEXT)')
        conn.executemany(
            "INSERT INTO cell VALUES (?)",
            [("f1",), ("f1",), ("f2",)],
        )

        removed = utils._verified_delete(
            conn,
            "cell",
            "s",
            's."fieldID" = ?',
            ["f1"],
            "release test rows",
        )

        assert removed == 2
        assert conn.execute(
            'SELECT COUNT(*) FROM cell WHERE "fieldID" = "f1"',
        ).fetchone() == (0,)
        assert conn.execute("SELECT COUNT(*) FROM cell").fetchone() == (1,)
    finally:
        conn.close()


def test_object_group_keys_names_the_missing_object_id():
    frame = pd.DataFrame(
        columns=["plateID", "rowID", "columnID", "fieldID"],
    )

    with pytest.raises(KeyError, match=r"cell_id"):
        timelapse._object_group_keys(frame, "cell_id")


def test_ultrack_set_assigns_a_field_and_names_a_missing_one():
    section = SimpleNamespace(max_distance=10.0)
    timelapse._ultrack_set(
        section, "max_distance", 25.0, "ultrack_max_distance",
    )
    assert section.max_distance == 25.0

    with pytest.raises(RuntimeError, match=r"ultrack_max_distance"):
        timelapse._ultrack_set(
            SimpleNamespace(),
            "max_distance",
            25.0,
            "ultrack_max_distance",
        )


def test_ultrack_converter_uses_the_legacy_name_and_rejects_no_name():
    def labels_to_edges(labels, sigma=None):
        return labels, sigma

    legacy = SimpleNamespace(labels_to_edges=labels_to_edges)
    assert timelapse._ultrack_labels_to_contours(legacy) is labels_to_edges

    with pytest.raises(
        RuntimeError,
        match=r"labels_to_contours.*labels_to_edges",
    ):
        timelapse._ultrack_labels_to_contours(SimpleNamespace())


def test_ultrack_track_kwargs_use_legacy_names_and_reject_unknown_signature():
    def legacy(config, detection=None, edges=None, images=()):
        pass

    assert timelapse._ultrack_track_kwargs(legacy) == (
        "detection",
        "edges",
        True,
    )

    def unknown(config, blobs=None, rims=None):
        pass

    with pytest.raises(RuntimeError, match=r"foreground/contours"):
        timelapse._ultrack_track_kwargs(unknown)
