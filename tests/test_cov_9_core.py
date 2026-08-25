"""Branches of ``spacr.core`` that only a damaged or plural input reaches.

Four situations, each of which produces a wrong number rather than an error
when it is handled badly:

* ``test_mode`` deciding how many overlay figures to draw;
* a ``merged/`` folder the pipeline cannot list;
* two source databases that call different experiments by the same plate id;
* a screen-graph run given several source folders instead of one.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


class _Recorder:
    """Records every call and returns ``ret``; stands in for an externality."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret

    @property
    def n(self):
        return len(self.calls)


def _mask_settings(src, **over):
    settings = {
        "src": str(src),
        "metadata_type": "cellvoyager",
        "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": None,
        "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "custom_regex": None,
        "randomize": True, "examples_to_plot": 1,
    }
    settings.update(over)
    return settings


@pytest.fixture
def mask_externals(monkeypatch):
    """Stub the GPU segmentation and the disk-heavy merge/plot collaborators.

    Each one is a true externality for the orchestration under test: a
    Cellpose-SAM forward pass, the array-merging IO pass, and matplotlib
    overlay rendering.
    """
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    spies = {
        "cellpose": _Recorder(), "organelle": _Recorder(),
        "concat": _Recorder(), "overlay": _Recorder(), "arrays": _Recorder(),
        "adjust": _Recorder(), "pivot": _Recorder(), "cleanup": _Recorder(ret=[]),
    }
    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", spies["cellpose"])
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam", spies["organelle"])
    monkeypatch.setattr(sio, "_load_and_concatenate_arrays", spies["concat"])
    monkeypatch.setattr(splot, "plot_image_mask_overlay", spies["overlay"])
    monkeypatch.setattr(splot, "plot_arrays", spies["arrays"])
    monkeypatch.setattr(su, "adjust_cell_masks", spies["adjust"])
    monkeypatch.setattr(su, "_pivot_counts_table", spies["pivot"])
    monkeypatch.setattr(su, "cleanup_pipeline_folders", spies["cleanup"])
    return spies


def _run_folder(root, *, merged=0):
    src = root / "a" / "deeply" / "nested" / "project" / "plate1"
    (src / "stack").mkdir(parents=True)
    np.save(src / "stack" / "f0.npy", np.zeros((3, 8, 8), np.uint16))
    if merged:
        (src / "merged").mkdir(parents=True)
        for i in range(merged):
            np.save(src / "merged" / f"m{i}.npy", np.zeros((5, 8, 8), np.uint16))
    return src


# ---------------------------------------------------------------------------
# preprocess_generate_masks: the overlay-plot budget
# ---------------------------------------------------------------------------

def test_test_mode_counts_merged_fields_not_the_length_of_their_path(
        tmp_path, mask_externals):
    """``test_mode`` must plot one overlay per merged field, whatever the path.

    The budget is a count of fields. Deriving it from anything else -- the
    merged folder's path string is the obvious mistake -- makes the number of
    figures depend on how deeply the project happens to be nested, so the same
    run drawn from two machines produces two different sets of figures. The
    run folder here is deliberately nested far deeper than the field count.
    """
    from spacr.core import preprocess_generate_masks

    src = _run_folder(tmp_path, merged=2)
    assert len(str(src / "merged")) > 2, "the path must not equal the count"
    settings = _mask_settings(src, plot=True, test_mode=True, examples_to_plot=1)

    preprocess_generate_masks(settings)

    assert settings["examples_to_plot"] == 2
    assert mask_externals["overlay"].n == 2


def test_an_unlistable_merged_folder_is_reported_and_the_run_finishes(
        tmp_path, mask_externals, monkeypatch, capsys):
    """An overlay that cannot even be started must not brand the masks partial.

    The overlay PDFs are cosmetic; the masks are the result. When the merged
    folder cannot be listed at all -- an unreadable mount, a permission
    change mid-run -- the failure is reported and accounted for, no overlay is
    attempted, and the segmentation run is still declared complete.
    """
    import spacr.core as core
    from spacr.core import preprocess_generate_masks

    src = _run_folder(tmp_path, merged=3)
    real_listdir = os.listdir

    def _refuse_merged(path, *args, **kwargs):
        if str(path).endswith("merged"):
            raise PermissionError("merged is unreadable")
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(core.os, "listdir", _refuse_merged)
    preprocess_generate_masks(_mask_settings(src, plot=True, examples_to_plot=3))

    printed = capsys.readouterr().out
    assert "Failed to plot image mask overly." in printed
    assert "merged is unreadable" in printed
    assert mask_externals["overlay"].n == 0
    assert "Successfully completed run" in printed


# ---------------------------------------------------------------------------
# generate_image_umap: two databases that share a plate id
# ---------------------------------------------------------------------------

def _measurement_db(path, plate):
    frame = pd.DataFrame({
        "plateID": [plate] * 3,
        "rowID": [f"r{i + 1}" for i in range(3)],
        "columnID": ["c1"] * 3,
        "area": range(3),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        frame.to_sql("cell", db, index=False)
    return str(path)


class _Stop(Exception):
    """Ends the run once the branch under test has been passed."""


def test_two_umap_sources_sharing_a_plate_id_say_so_before_embedding(
        tmp_path, monkeypatch, capsys):
    """A plate id that appears in two source databases pools two experiments.

    Every per-well number and every cluster the embedding shows would then be
    computed over both runs at once with nothing on screen to say so. The
    check is advisory -- an existing caller must not stop working -- so it has
    to name the colliding plate and both source labels loudly instead.
    """
    import spacr.core as core
    import spacr.io as sio
    import spacr.utils as su

    run_a = tmp_path / "runA"
    run_b = tmp_path / "runB"
    paths = [_measurement_db(run_a / "measurements" / "measurements.db", "plate1"),
             _measurement_db(run_b / "measurements" / "measurements.db", "plate1")]

    monkeypatch.setattr(su, "get_db_paths", lambda srcs: paths)

    def _stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(sio, "open_crop_source", _stop)

    with pytest.raises(_Stop):
        core.generate_image_umap({"src": [str(run_a), str(run_b)],
                                  "plot_images": False})

    printed = capsys.readouterr().out
    assert "the same plate id appears in more than one" in printed
    assert "'plate1'" in printed
    assert "runA" in printed and "runB" in printed
    assert core.UMAP_SOURCE_COLUMN in printed


def test_an_unmergeable_source_pair_still_reaches_the_embedding(
        tmp_path, monkeypatch, capsys):
    """The collision check is advice and must never end a run by itself.

    A source database with no ``cell`` table is a legitimate shape for this
    entry point, and the plate comparison cannot be made for it. Failing there
    would stop a run that has nothing wrong with it, so the check swallows its
    own failure and the embedding continues.
    """
    import spacr.core as core
    import spacr.io as sio
    import spacr.multi_database as multi_database
    import spacr.utils as su

    run_a = tmp_path / "runA"
    run_b = tmp_path / "runB"
    paths = [_measurement_db(run_a / "measurements" / "measurements.db", "plate1"),
             _measurement_db(run_b / "measurements" / "measurements.db", "plate2")]

    monkeypatch.setattr(su, "get_db_paths", lambda srcs: paths)

    def _no_cell_table(*args, **kwargs):
        raise KeyError("cell")

    monkeypatch.setattr(multi_database, "describe_merge", _no_cell_table)

    def _stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(sio, "open_crop_source", _stop)

    with pytest.raises(_Stop):
        core.generate_image_umap({"src": [str(run_a), str(run_b)],
                                  "plot_images": False})

    assert "the same plate id appears" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generate_screen_graphs: several source folders
# ---------------------------------------------------------------------------

def test_a_list_of_screen_sources_is_read_source_by_source(tmp_path, monkeypatch):
    """Several source folders must each be read, not joined into one path.

    ``src`` accepts a string or a list. Wrapping an already-plural value would
    build ``"['/a', '/b']/measurements/measurements.db"`` -- a path that
    exists nowhere -- so the plural form has to be used as it stands.
    """
    import spacr.core as core
    import spacr.io as sio

    seen = []

    def _record(db_loc, *args, **kwargs):
        seen.append(list(db_loc))
        raise _Stop()

    monkeypatch.setattr(sio, "_read_and_merge_data", _record)

    with pytest.raises(_Stop):
        core.generate_screen_graphs({
            "src": [str(tmp_path / "plateA"), str(tmp_path / "plateB")],
            "tables": ["cell"], "nuclei_limit": 10, "pathogen_limit": 10,
        })

    assert seen == [[os.path.join(str(tmp_path / "plateA"),
                                  "measurements", "measurements.db")]]
