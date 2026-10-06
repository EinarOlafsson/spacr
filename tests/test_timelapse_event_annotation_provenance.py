"""Detector boundaries for annotations bound to one exact tracker export."""

from __future__ import annotations

import pandas as pd
import pytest

from spacr import timelapse
from spacr.tabular import write_table


def _input(tmp_path):
    """Write one real tracker and one event table without fitting a model."""
    tracks_dir = tmp_path / "tracks"
    tracks = tracks_dir / "trackpy_tracks_cell_field.csv"
    write_table(pd.DataFrame([{"track_id": 7, "frame": 1,
                              "x": 10.0, "y": 9.0}]), tracks,
                canonicalise=False)
    annotations = tmp_path / "annotations.csv"
    row = {"field": "field", "track_id": 7, "frame": 1,
           "event": "mitosis", "object": "cell",
           "tracker_backend": "trackpy",
           "track_source_sha256": timelapse._event_source_hash(tracks)}
    return tracks_dir, tracks, annotations, row


def test_detector_reader_refuses_order_dependent_labels_at_one_observation(tmp_path):
    _tracks_dir, _tracks, annotations, row = _input(tmp_path)
    write_table(pd.DataFrame([row, dict(row, event="death")]),
                annotations, canonicalise=False)
    with pytest.raises(ValueError, match="labels one track frame more than once"):
        timelapse._event_read_annotations(str(annotations))


@pytest.mark.parametrize("change,reason", [
    (lambda row: dict(row, tracker_backend="ultrack"), "another tracker CSV"),
    (lambda row: dict(row, track_source_sha256="0" * 64),
     "another tracker CSV"),
    (lambda row: {key: value for key, value in row.items()
                  if key != "track_source_sha256"}, "incomplete tracker provenance"),
])
def test_detector_refuses_wrong_or_partial_tracker_provenance_before_model_setup(
        tmp_path, monkeypatch, change, reason):
    tracks_dir, _tracks, annotations, row = _input(tmp_path)
    write_table(pd.DataFrame([change(row)]), annotations, canonicalise=False)
    monkeypatch.setattr(timelapse, "_event_field_inputs",
                        lambda *_args: (pd.DataFrame(), None))
    model_setup = []
    monkeypatch.setattr(timelapse, "_event_cross_validate",
                        lambda *_args, **_kwargs: model_setup.append(True))
    with pytest.raises(ValueError, match=reason):
        timelapse._event_detection(str(tracks_dir), "cell", "trackpy",
                                   annotations=str(annotations), plot=False)
    assert model_setup == []


@pytest.mark.parametrize("legacy", [False, True])
def test_current_exact_tracker_and_legacy_csv_reach_the_existing_model_boundary(
        tmp_path, monkeypatch, legacy):
    tracks_dir, _tracks, annotations, row = _input(tmp_path)
    if legacy:
        row.pop("tracker_backend")
        row.pop("track_source_sha256")
    write_table(pd.DataFrame([row]), annotations, canonicalise=False)
    monkeypatch.setattr(timelapse, "_event_field_inputs",
                        lambda *_args: (pd.DataFrame(), None))
    model_setup = []

    def reached_model(*_args, **_kwargs):
        model_setup.append(True)
        raise RuntimeError("validated current tracker")

    monkeypatch.setattr(timelapse, "_event_cross_validate", reached_model)
    with pytest.raises(RuntimeError, match="validated current tracker"):
        timelapse._event_detection(str(tracks_dir), "cell", "trackpy",
                                   annotations=str(annotations), plot=False)
    assert model_setup == [True]


def test_detector_checks_each_annotated_field_against_its_own_tracker(
        tmp_path, monkeypatch):
    tracks_dir, tracks, annotations, row = _input(tmp_path)
    second = tracks.with_name("trackpy_tracks_cell_second.csv")
    second.write_bytes(tracks.read_bytes() + b"\n")
    other = dict(row, field="second",
                 track_source_sha256=timelapse._event_source_hash(second))
    write_table(pd.DataFrame([row, other]), annotations, canonicalise=False)
    monkeypatch.setattr(timelapse, "_event_field_inputs",
                        lambda *_args: (pd.DataFrame(), None))

    def reached_model(*_args, **_kwargs):
        raise RuntimeError("both tracker identities validated")

    monkeypatch.setattr(timelapse, "_event_cross_validate", reached_model)
    with pytest.raises(RuntimeError, match="both tracker identities validated"):
        timelapse._event_detection(str(tracks_dir), "cell", "trackpy",
                                   annotations=str(annotations), plot=False)
    legacy = dict(row, tracker_backend=None, track_source_sha256=None)
    write_table(pd.DataFrame([legacy, other]), annotations, canonicalise=False)
    with pytest.raises(RuntimeError, match="both tracker identities validated"):
        timelapse._event_detection(str(tracks_dir), "cell", "trackpy",
                                   annotations=str(annotations), plot=False)
    write_table(pd.DataFrame([row, dict(other, track_source_sha256="0" * 64)]),
                annotations, canonicalise=False)
    with pytest.raises(ValueError, match="another tracker CSV"):
        timelapse._event_detection(str(tracks_dir), "cell", "trackpy",
                                   annotations=str(annotations), plot=False)
