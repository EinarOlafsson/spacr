"""Item 288: three corners of the timelapse module the other tests skip.

* Timeflows tracking with no device named picks CUDA when there is one and
  the CPU otherwise -- and on the CPU loads the checkpoint in float32, the
  precision that is fast there.
* Asked to plot or save, Timeflows hands the relabelled stack and its tracks
  to the track visualiser; asked for neither, it draws nothing.
* Infection-intensity QC per plate skips a plate that has no rows, which a
  categorical plate column with an unused category produces, rather than
  running the QC on an empty table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spacr import timeflows_model as tm
from spacr import timelapse
from tests.test_timeflows_movie_tracking import _backend_inputs, _oracle


def test_no_device_named_means_the_cpu_here_and_float32(tmp_path,
                                                        monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    masks, images, net = _backend_inputs()
    checkpoint = tmp_path / "tf.pt"
    checkpoint.write_bytes(b"stand-in")
    loaded = []

    def load(path, device="cpu", precision="checkpoint"):
        loaded.append((device, precision))
        return net

    monkeypatch.setattr(tm, "_load_timeflows", load)
    monkeypatch.setattr(tm, "predict_pair", _oracle)
    timelapse._timeflows_track_cells(str(tmp_path / "masks"), "n", [], "cell",
                                     masks, images=images,
                                     model_path=str(checkpoint))
    assert loaded == [("cpu", "float32")]


def test_saving_hands_the_tracks_to_the_visualiser(tmp_path, monkeypatch):
    import spacr.plot as splot

    drawn = []
    monkeypatch.setattr(
        splot, "_visualize_and_save_timelapse_stack_with_tracks",
        lambda stack, tracks, save, src, name, plot, files, obj, mode:
        drawn.append((stack.shape, sorted(tracks["track_id"].unique()), save,
                      plot, name, obj, mode)))
    monkeypatch.setattr(tm, "predict_pair", _oracle)
    masks, images, net = _backend_inputs()
    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    timelapse._timeflows_track_cells(str(src), "plate1_A01", ["a", "b", "c"],
                                     "cell", masks, images=images, net=net,
                                     save=True)
    assert len(drawn) == 1
    shape, track_ids, save, plot, name, obj, mode = drawn[0]
    assert shape == masks.shape
    assert len(track_ids) == 2
    assert (save, plot, name, obj, mode) == (True, False, "plate1_A01",
                                             "cell", "timeflows")


def test_a_plate_with_no_rows_is_not_quality_checked(tmp_path, monkeypatch):
    seen = []

    def qc(all_df, settings, infection_col, pathogen_chan, motility_dir):
        seen.append(sorted(all_df["plateID"].astype(str).unique()))
        return all_df.copy(), infection_col

    monkeypatch.setattr(timelapse, "_infection_qc_histogram", qc)
    frame = pd.DataFrame({
        "plateID": pd.Categorical(["p1", "p1"], categories=["p1", "p2"]),
        "infected": [0, 1],
        "pathogen_channel_1_mean_intensity": [0.1, 0.9],
    })
    settings = {"infection_intensity_qc": True,
                "infection_intensity_strategy": "histogram",
                "infection_intensity_qc_scope": "plate"}
    out, column = timelapse._apply_infection_intensity_qc(
        all_df=frame, settings=settings, infection_col="infected",
        pathogen_chan=1, motility_dir=str(tmp_path / "motility"))
    assert seen == [["p1"]], "the empty p2 group never reached the QC"
    assert column == "infected"
    assert list(out["infected"]) == [0, 1]
    assert np.all(out["plateID"].astype(str) == "p1")
