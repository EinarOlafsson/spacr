"""Single branches in core modules that no other test happened to take."""
from __future__ import annotations

import sqlite3
import types

import numpy as np
import pandas as pd
import pytest


def test_crop_embeddings_from_a_frame_replace_earlier_rows(tmp_path):
    from spacr import active_learning as al

    db = str(tmp_path / "measurements.db")
    assert al._stored_embedding_frame(db) is None
    first = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert al._store_crop_embeddings(db, ["o1", "o2"], first) == 2
    second = pd.DataFrame({"a": [9.0], "b": [9.0]})
    assert al._store_crop_embeddings(db, ["o2"], second) == 2
    stored = al._stored_embedding_frame(db).set_index("prcfo")
    assert stored.loc["o2", "a"] == 9.0 and stored.loc["o1", "a"] == 1.0


def test_stored_embeddings_need_a_crop_table_that_joins(tmp_path):
    from spacr import active_learning as al

    db = str(tmp_path / "measurements.db")
    al._store_crop_embeddings(db, ["o1"], np.ones((1, 2)))
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE png_list (png_path TEXT)")
    assert al._stored_embeddings(db) is None
    with sqlite3.connect(db) as con:
        con.execute("DROP TABLE png_list")
        con.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT)")
        con.execute("INSERT INTO png_list VALUES ('x.png', 'other')")
    assert al._stored_embeddings(db) is None


def test_the_make_masks_cli_refuses_uncertainty_flags_without_uncertainty(
        tmp_path):
    from spacr import cli_make_masks

    with pytest.raises(SystemExit):
        cli_make_masks.main([str(tmp_path), "--save-uncertainty-maps"])
    with pytest.raises(SystemExit):
        cli_make_masks.main([str(tmp_path), "--uncertainty-second-model",
                             str(tmp_path / "m.pt")])


def test_windows_and_macos_always_have_a_display(monkeypatch):
    from spacr import cli_make_masks

    monkeypatch.setattr(cli_make_masks.sys, "platform", "darwin")
    assert cli_make_masks.has_display() is True


def test_a_harmony_background_that_is_not_a_mapping_has_no_mean():
    from spacr.illumination import _harmony_background_mean

    assert _harmony_background_mean([1, 2]) is None


def test_a_child_with_a_measured_column_keeps_the_count_columns_apart():
    from spacr.merge_tables import MergePolicy, roll_up

    child = pd.DataFrame({"cell_id": [1, 1], "measured": [1.0, 2.0],
                          "count": [3, 4]})
    out = roll_up(child, ["cell_id"], name="pathogen",
                  policy=MergePolicy())
    assert any("source_measured" in str(c) for c in out.columns)


def test_an_instanseg_object_channel_outside_the_stack_is_refused():
    from spacr.object import _segmentation_input_channels

    model = types.SimpleNamespace(name="instanseg")
    with pytest.raises(ValueError, match="outside the source stack"):
        _segmentation_input_channels([5], 3, model)


def test_booleans_are_not_numbers_for_sorting():
    from spacr.qt.widgets.sortable_table import numeric_value

    assert numeric_value(True) is None
    assert numeric_value(2) == 2.0


def test_settling_a_table_without_sorting_does_nothing():
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.sortable_table import _settle_sorting

    view = types.SimpleNamespace()
    assert _settle_sorting(view) is None


def test_a_cellpose3_plaque_model_given_a_flat_image_reads_it_as_is():
    from spacr import submodules

    seen = []

    class _Backend:
        def eval(self, images, diameter=None, flow_threshold=None,
                 cellprob_threshold=None):
            seen.append(images[0].shape)
            return [np.zeros((4, 4), int)], [], None

    model = submodules._Cellpose3PlaqueModel(_Backend())
    labels, flows, _ = model.eval(np.ones((4, 4)))
    assert seen == [(4, 4)] and flows == []


def test_a_cellpose3_backend_that_cannot_start_is_none(monkeypatch):
    from spacr import _segmentation_backends as sb
    from spacr import submodules

    monkeypatch.setattr(sb, "_backend_state", lambda name: types.SimpleNamespace(
        ready=True, in_process=False))

    def refuse(*a, **k):
        raise RuntimeError("no worker")

    monkeypatch.setattr(sb, "_RemoteBackend", refuse)
    assert submodules._cellpose3_plaque_backend("/models/x.pth") is None
