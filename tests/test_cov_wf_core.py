"""Branches of ``spacr.core`` that only an unusual-but-legal run reaches.

Each one decides something a user would notice and could not diagnose from
the log:

* a v2 run configured with a cell channel but no nucleus channel -- what
  Cellpose is actually given to segment on;
* mask stacks that are already complete for some object types but not others
  -- whether hours of GPU time are spent again;
* cell-mask adjustment when an organelle channel was requested but no
  organelle mask folder was ever written;
* two source databases whose plates really are different plates -- the
  collision warning must stay silent then, or it means nothing when it fires;
* ``exclude_conditions`` and ``remove_cluster_noise`` on a quiet run -- the
  rows that survive, and whether frame, embedding and crop list stay aligned.

Everything here is CPU-only, offline and deterministic: the Cellpose-SAM
forward pass, the array merge and the reducer are the only things stubbed.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


class Spy:
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
    s = {
        "src": str(src),
        "metadata_type": "cellvoyager",
        "channels": [0, 1, 2, 3],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": 2,
        "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "custom_regex": None,
        "randomize": True, "examples_to_plot": 1,
    }
    s.update(over)
    return s


@pytest.fixture
def stubs(monkeypatch):
    """Stub the GPU segmentation and the disk-heavy merge/plot collaborators."""
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    sp = {"cellpose": Spy(), "organelle": Spy(), "concat": Spy(),
          "overlay": Spy(), "arrays": Spy(), "adjust": Spy(),
          "pivot": Spy(), "cleanup": Spy(ret=[])}
    monkeypatch.setattr(sobj, "generate_cellpose_masks_sam", sp["cellpose"])
    monkeypatch.setattr(sobj, "generate_organelle_masks_sam", sp["organelle"])
    monkeypatch.setattr(sio, "_load_and_concatenate_arrays", sp["concat"])
    monkeypatch.setattr(splot, "plot_image_mask_overlay", sp["overlay"])
    monkeypatch.setattr(splot, "plot_arrays", sp["arrays"])
    monkeypatch.setattr(su, "adjust_cell_masks", sp["adjust"])
    monkeypatch.setattr(su, "_pivot_counts_table", sp["pivot"])
    monkeypatch.setattr(su, "cleanup_pipeline_folders", sp["cleanup"])
    return sp


@pytest.fixture
def run_dir(tmp_path):
    """A v1 run folder with a three-field ``stack/`` and no masks yet."""
    src = tmp_path / "plate1"
    (src / "stack").mkdir(parents=True)
    for i in range(3):
        np.save(src / "stack" / f"f{i}.npy", np.zeros((4, 8, 8), np.uint16))
    return src


def _complete_mask_stack(run_dir, name, n=3):
    """Write ``n`` mask arrays into ``<src>/masks/<name>`` -- one per field."""
    folder = run_dir / "masks" / name
    folder.mkdir(parents=True)
    for i in range(n):
        np.save(folder / f"f{i}.npy", np.ones((8, 8), np.uint16))
    return folder


# ---------------------------------------------------------------------------
# preprocess_generate_masks -- the v2 dispatch
# ---------------------------------------------------------------------------

def test_a_v2_run_without_a_nucleus_channel_segments_on_the_cell_plane_alone(
        tmp_path, monkeypatch):
    """Cellpose must be pointed at the planes that exist, not at plane 1.

    ``channels_for_cellpose`` are indices into the stack v2 just wrote, not
    into the raw file. A run configured with a cell channel and no nucleus
    channel has a one-plane stack, so appending a nucleus index anyway would
    hand Cellpose a plane that is not there (or, worse, someone else's
    channel) and segment the wrong pixels with no error to show for it.
    """
    import spacr._v1_v2_bridge as bridge
    import spacr.pipeline_v2 as pv2
    from spacr.core import preprocess_generate_masks

    run = Spy(ret={"stacks": []})
    monkeypatch.setattr(pv2, "run_v2", run)
    monkeypatch.setattr(bridge, "report_disk_savings", lambda *a, **k: None)

    src = tmp_path / "plateA"
    src.mkdir()
    preprocess_generate_masks(_mask_settings(
        src, pipeline_style="v2", cell_channel=1, nucleus_channel=None,
        pathogen_channel=None, organelle_channel=None))

    assert run.n == 1
    solo = run.calls[0][1]
    assert solo["channel_names"] == ["cell"]
    assert solo["channels_for_cellpose"] == (0,), \
        "no nucleus channel -> no second plane to segment on"

    # ...and the same dispatch WITH a nucleus channel does add it, so the
    # assertion above is about the configuration and not about a dead branch.
    run.calls.clear()
    preprocess_generate_masks(_mask_settings(
        src, pipeline_style="v2", cell_channel=1, nucleus_channel=0,
        pathogen_channel=None, organelle_channel=None))
    both = run.calls[0][1]
    assert both["channel_names"] == ["nucleus", "cell"]
    assert both["channels_for_cellpose"] == (1, 0)


def test_consolidate_keeps_each_source_in_its_own_plate(tmp_path, monkeypatch):
    """A list of plates is consolidated in order without becoming one path."""
    import spacr.core as core
    import spacr.utils as su

    roots = [tmp_path / "plate1", tmp_path / "plate2"]
    for root in roots:
        root.mkdir()
    calls = []

    def image_map(source):
        calls.append(("map", source))
        return {f"{source}/raw.tif": f"{source}/renamed.tif"}

    def consolidate(mapping, source):
        calls.append(("copy", source, mapping))
        os.makedirs(os.path.join(source, "consolidated"))

    monkeypatch.setattr(su, "generate_image_path_map", image_map)
    monkeypatch.setattr(su, "copy_images_to_consolidated", consolidate)
    settings = _mask_settings(roots[0], masks=False, consolidate=True)
    settings["src"] = [str(root) for root in roots]

    core.preprocess_generate_masks(settings)

    assert calls == [
        ("map", str(roots[0])),
        ("copy", str(roots[0]), {
            f"{roots[0]}/raw.tif": f"{roots[0]}/renamed.tif"}),
        ("map", str(roots[1])),
        ("copy", str(roots[1]), {
            f"{roots[1]}/raw.tif": f"{roots[1]}/renamed.tif"}),
    ]
    for root in roots:
        assert (root / "consolidated" / "settings").is_dir()


# ---------------------------------------------------------------------------
# preprocess_generate_masks -- the "already generated" skip, per object type
# ---------------------------------------------------------------------------

def test_only_the_object_whose_masks_are_missing_is_segmented_again(
        run_dir, stubs, capsys):
    """Re-running a plate must cost GPU time only for what is not yet there.

    ``check_mask_folder`` is consulted separately for cell, nucleus, pathogen
    and each organelle slot. If the skip were decided once -- or not at all --
    a plate that lost only its cell masks would re-segment every object type,
    turning a minutes-long repair into the hours the original run took, and
    overwriting masks a user may already have curated.
    """
    from spacr.core import preprocess_generate_masks

    for name in ("nucleus_mask_stack", "pathogen_mask_stack",
                 "organelle_mask_stack"):
        _complete_mask_stack(run_dir, name)

    preprocess_generate_masks(_mask_settings(
        run_dir, cell_channel=1, nucleus_channel=0, pathogen_channel=2,
        organelle_channel=3))

    # The one object with no mask folder is the only one segmented.
    assert [c[0][2] for c in stubs["cellpose"].calls] == ["cell"]
    assert stubs["organelle"].n == 0, \
        "a complete organelle_mask_stack must not be regenerated"
    printed = capsys.readouterr().out
    for name in ("nucleus_mask_stack", "pathogen_mask_stack",
                 "organelle_mask_stack"):
        assert f"All masks have been generated for {name}" in printed
    assert "All masks have been generated for cell_mask_stack" not in printed


def test_cell_adjustment_survives_an_organelle_channel_that_wrote_no_masks(
        run_dir, stubs):
    """A requested organelle that produced nothing must not break adjustment.

    ``adjust_cell_masks`` is handed folder paths and opens them. When
    ``organelle_channel`` is set but segmentation wrote no
    ``organelle_mask_stack`` -- an empty channel, a segmenter that found no
    objects -- passing the path anyway makes the adjustment die on a missing
    directory and the whole plate loses its reconciled cell masks. The
    existence of the folder, not the setting, decides.
    """
    from spacr.core import preprocess_generate_masks

    preprocess_generate_masks(_mask_settings(
        run_dir, adjust_cells=True, cell_channel=1, nucleus_channel=0,
        pathogen_channel=2, organelle_channel=3, n_jobs=2))

    masks = os.path.join(str(run_dir), "masks")
    assert stubs["adjust"].n == 1
    args, kwargs = stubs["adjust"].calls[0]
    assert args[0] == os.path.join(masks, "pathogen_mask_stack")
    assert args[1] == os.path.join(masks, "cell_mask_stack")
    assert args[2] == os.path.join(masks, "nucleus_mask_stack")
    assert args[3] is None, "no organelle_mask_stack on disk -> no path"
    assert kwargs["n_jobs"] == 2

    # The same settings, once the folder exists, DO pass it: the None above
    # comes from the absent folder and not from an ignored setting.
    (run_dir / "masks" / "organelle_mask_stack").mkdir(parents=True,
                                                       exist_ok=True)
    preprocess_generate_masks(_mask_settings(
        run_dir, adjust_cells=True, cell_channel=1, nucleus_channel=0,
        pathogen_channel=2, organelle_channel=3, n_jobs=2))
    assert stubs["adjust"].calls[1][0][3] == os.path.join(
        masks, "organelle_mask_stack")


# ---------------------------------------------------------------------------
# generate_image_umap -- several source databases
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """Ends the run once the branch under test has been passed."""


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


def test_two_sources_naming_different_plates_raise_no_collision_warning(
        tmp_path, monkeypatch, capsys):
    """The plate-collision warning has to stay quiet on an ordinary merge.

    Embedding two plates together is the normal reason to pass a list of
    sources. If the advisory fired whenever there was more than one database,
    every multi-plate run would print a warning that its numbers are computed
    over two experiments at once -- and the one run where that is actually
    true would be indistinguishable from the rest.
    """
    import spacr.core as core
    import spacr.io as sio
    import spacr.utils as su

    def _stop(*args, **kwargs):
        raise _Stop()

    monkeypatch.setattr(sio, "open_crop_source", _stop)

    run_a, run_b = tmp_path / "runA", tmp_path / "runB"
    distinct = [_measurement_db(run_a / "measurements" / "measurements.db", "plate1"),
                _measurement_db(run_b / "measurements" / "measurements.db", "plate2")]
    monkeypatch.setattr(su, "get_db_paths", lambda srcs: distinct)
    with pytest.raises(_Stop):
        core.generate_image_umap({"src": [str(run_a), str(run_b)],
                                  "plot_images": False})
    quiet = capsys.readouterr().out
    assert "Generating Image UMAP" in quiet, "the run really did start"
    assert "the same plate id appears" not in quiet

    # The same two sources, both calling their plate 'plate1', DO warn -- so
    # the silence above is the check passing, not the check missing.
    run_c, run_d = tmp_path / "runC", tmp_path / "runD"
    colliding = [_measurement_db(run_c / "measurements" / "measurements.db", "plate1"),
                 _measurement_db(run_d / "measurements" / "measurements.db", "plate1")]
    monkeypatch.setattr(su, "get_db_paths", lambda srcs: colliding)
    with pytest.raises(_Stop):
        core.generate_image_umap({"src": [str(run_c), str(run_d)],
                                  "plot_images": False})
    loud = capsys.readouterr().out
    assert "the same plate id appears in more than one" in loud
    assert "'plate1'" in loud


# ---------------------------------------------------------------------------
# generate_image_umap / reducer_hyperparameter_search -- row filters
# ---------------------------------------------------------------------------

def _feature_frame(n=8):
    """A measurement frame with two conditions and no crop columns at all."""
    return pd.DataFrame({
        "columnID": ["c1" if i % 2 == 0 else "c2" for i in range(n)],
        "cell_area": np.linspace(10.0, 90.0, n),
        "cell_perimeter": np.linspace(4.0, 40.0, n),
        "prcfo": [f"plate1_r1_c{(i % 2) + 1}_f1_o{i}" for i in range(n)],
    })


def _stub_umap_io(monkeypatch, frame, labels=None):
    """Point core's per-source reads at ``frame`` and record the reducer input."""
    import spacr.core as core
    import spacr.io as sio
    import spacr.utils as su

    seen = {}
    monkeypatch.setattr(su, "get_db_paths", lambda _src: [np.nan])
    monkeypatch.setattr(core, "_validate_umap_source_db", lambda *a, **k: None)
    monkeypatch.setattr(sio, "_read_and_join_tables", lambda *a, **k: frame.copy())
    monkeypatch.setattr(su, "correct_paths", lambda df, _src: (df, []))
    monkeypatch.setattr(sio, "open_crop_source", lambda *a, **k: None)

    def _cluster(numeric_data, *a, **k):
        seen["rows"] = len(numeric_data)
        lab = (np.zeros(len(numeric_data), dtype=int) if labels is None
               else np.asarray(labels))
        return np.zeros((len(numeric_data), 2)), lab, None

    monkeypatch.setattr(su, "reduction_and_clustering", _cluster)
    return seen


def _umap_settings(src, **over):
    s = {"src": str(src), "tables": ["cell"], "pos": "c2", "neg": "c1",
         "mix": "c3", "plot_images": False, "verbose": False,
         "exclude_conditions": None, "remove_cluster_noise": False,
         "n_jobs": 1, "save_figure": False,
         # the frame below carries morphology features only, and every row
         # of it is wanted: no feature filter, no row cap.
         "filter_by": None, "row_limit": None}
    s.update(over)
    return s


def test_excluded_conditions_leave_the_embedding_without_announcing_it(
        tmp_path, monkeypatch, capsys):
    """A quiet run must still exclude the rows, and a loud one must say so.

    ``exclude_conditions`` decides which wells are embedded at all, so a run
    that silently kept them would put the positive controls back into a map
    the user believes shows only the screen. ``verbose`` controls the
    reporting line and must not control the filter.
    """
    from spacr.core import generate_image_umap

    seen = _stub_umap_io(monkeypatch, _feature_frame(8))
    quiet = generate_image_umap(_umap_settings(
        tmp_path, exclude_conditions="pos", verbose=False))

    assert seen["rows"] == 4, "the four 'pos' rows were dropped before reduction"
    assert list(quiet["cond"].unique()) == ["neg"]
    assert "Excluded 4 rows" not in capsys.readouterr().out

    # verbose=True over the same data prints exactly the line withheld above.
    seen = _stub_umap_io(monkeypatch, _feature_frame(8))
    loud = generate_image_umap(_umap_settings(
        tmp_path, exclude_conditions=["pos"], verbose=True))
    assert len(loud) == 4
    assert "Excluded 4 rows after excluding: ['pos']" in capsys.readouterr().out


def test_noise_removal_keeps_the_frame_aligned_when_there_are_no_crops(
        tmp_path, monkeypatch, capsys):
    """Dropping DBSCAN noise must drop the same rows from every companion list.

    The frame, the embedding and the crop-path list are indexed positionally
    against each other. A measurement-only source has no crop list at all, and
    reaching for it would end the run with a TypeError; keeping the frame
    unfiltered instead would misalign ``all_df['cluster'] = labels`` and
    label every remaining object with its neighbour's cluster.
    """
    from spacr.core import generate_image_umap

    labels = np.array([0, 0, -1, 0, -1, 1, 1, 1])
    seen = _stub_umap_io(monkeypatch, _feature_frame(8), labels=labels)
    out = generate_image_umap(_umap_settings(
        tmp_path, remove_cluster_noise=True, plot_images=True))

    printed = capsys.readouterr().out
    assert "plotting points only" in printed, "no crop source and no png_path"
    assert seen["rows"] == 8, "all eight rows reached the reducer"
    assert len(out) == 6, "the two noise rows left the frame"
    assert list(out["cluster"]) == [0, 0, 0, 1, 1, 1]
    assert list(out["cell_area"]) == [
        pytest.approx(v) for v in _feature_frame(8)["cell_area"][labels != -1]]


def test_the_reducer_search_excludes_conditions_given_as_a_list(
        tmp_path, monkeypatch, capsys):
    """The sweep must search over the same rows the embedding would use.

    ``exclude_conditions`` accepts a string or a list; only the string form is
    wrapped. If the list form were wrapped too, the frame would be compared
    against ``[['pos']]``, nothing would match, and every hyperparameter panel
    would be drawn over the controls the user asked to leave out -- silently
    choosing parameters for the wrong data.
    """
    import spacr.utils as su
    from spacr.core import reducer_hyperparameter_search

    frame = _feature_frame(8)
    seen = {}

    def _search(numeric_data, *a, **k):
        seen["rows"] = len(numeric_data)
        return np.zeros((len(numeric_data), 2)), np.zeros(len(numeric_data), int)

    monkeypatch.setattr(su, "get_db_paths", lambda _src: ["db"])
    monkeypatch.setattr(su, "search_reduction_and_clustering", _search)
    import spacr.io as sio
    monkeypatch.setattr(sio, "_read_and_join_tables", lambda *a, **k: frame.copy())

    out = reducer_hyperparameter_search(
        settings=_umap_settings(tmp_path, exclude_conditions=["pos"],
                                verbose=False, reduction_method="umap"),
        reduction_params=[{"n_neighbors": 3}],
        dbscan_params=[{"eps": 0.5, "min_samples": 2}],
        save=False, show=False)

    assert out is None
    assert seen["rows"] == 4, "only the four 'neg' rows were swept"
    assert "Excluded 4 rows" not in capsys.readouterr().out
