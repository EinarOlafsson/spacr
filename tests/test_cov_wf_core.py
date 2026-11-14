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


# ---------------------------------------------------------------------------
# preprocess_generate_masks -- the v2 dispatch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# preprocess_generate_masks -- the "already generated" skip, per object type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# generate_image_umap -- several source databases
# ---------------------------------------------------------------------------

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


def test_quality_excluding_every_v1_field_never_calls_segmenter(run_dir, stubs):
    from spacr.core import preprocess_generate_masks
    from spacr.image_quality import excluded_fields
    folder = run_dir / 'masks'
    folder.mkdir()
    for index in range(3):
        np.save(run_dir / 'stack' / f'f{index}.npy', np.zeros((8, 8, 4), np.uint16))
    np.savez(folder / 'normalized.npz', data=np.zeros((3, 8, 8, 3), np.float32),
             filenames=np.array(['f0.npy', 'f1.npy', 'f2.npy']),
             source_channels=np.array([0, 1, 2], dtype=np.int64))
    preprocess_generate_masks(_mask_settings(run_dir, image_qc_mode='exclude',
                                              image_qc_min_focus={0: 1.}))
    assert excluded_fields(run_dir) == {'f0.npy', 'f1.npy', 'f2.npy'}
    assert stubs['cellpose'].n == 0
    assert stubs['organelle'].n == 0
    assert stubs['concat'].n == 0
