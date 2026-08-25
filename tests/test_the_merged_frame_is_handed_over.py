"""The merged frame reaches the fit without a serialisation round trip.

The Measurements queue merges every plate's measurements into one frame and
then fits response columns against it. That frame is about 2.75 GB on a
four-plate screen, and it is ALREADY IN THIS PROCESS when the fit starts:
writing it as CSV costs about 160 seconds and every parse back costs again,
all of it before the fit does any arithmetic.

So the frame is offered under the path it was written to, the artefact is
written columnar, and the loader says which of the two routes it took --
because minutes of silence between the merge finishing and the fit starting
is what makes a working run look like a hung one.
"""
import gc
import os

import numpy as np
import pandas as pd
import pytest

from spacr import frame_handoff, tabular
from spacr.ml import _stage, load_regression_input_pairs


@pytest.fixture()
def merged_and_counts(tmp_path):
    """One merged score frame for four plates, one count file per plate."""
    frame = pd.DataFrame([
        {"plateID": f"plate{p}", "rowID": "r1", "columnID": "c1",
         "cell_area": 100.0 + p}
        for p in (1, 2, 3, 4)])
    score = tmp_path / "merged_measurements.parquet"
    pairs = []
    for plate in (1, 2, 3, 4):
        count = tmp_path / f"count{plate}.csv"
        pd.DataFrame([{"rowID": "r1", "columnID": "c1", "grna": "g1",
                       "count": 7}]).to_csv(count, index=False)
        pairs.append({"score": str(score), "count": str(count)})
    return frame, str(score), pairs


def test_a_held_frame_is_never_parsed(merged_and_counts, monkeypatch):
    """The parse is the cost being removed, so a parse of the held path is a
    failure of the whole mechanism, not a slow path."""
    frame, score_path, pairs = merged_and_counts
    tabular.write_table(frame, score_path)
    frame_handoff.hold(score_path, frame)

    real = tabular.read_table

    def refuse(path, *args, **kwargs):
        if os.path.basename(str(path)).startswith("merged_measurements"):
            raise AssertionError(f"parsed {path} instead of taking the frame")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(tabular, "read_table", refuse)
    try:
        counts, scores, audit = load_regression_input_pairs(pairs)
    finally:
        frame_handoff.release(score_path)

    assert sorted(scores["plateID"].unique()) == ["plate1", "plate2",
                                                  "plate3", "plate4"]
    assert sorted(scores["cell_area"]) == [101.0, 102.0, 103.0, 104.0]
    assert len(counts) == 4
    assert [row["plate"] for row in audit] == ["plate1", "plate2", "plate3",
                                               "plate4"]


def test_the_file_is_still_read_when_nothing_was_handed_over(
        merged_and_counts, monkeypatch):
    """A caller that knows nothing about the handoff must be unaffected."""
    frame, score_path, pairs = merged_and_counts
    tabular.write_table(frame, score_path)
    frame_handoff.release()

    seen = []
    real = tabular.read_table
    monkeypatch.setattr(
        tabular, "read_table",
        lambda p, *a, **k: (seen.append(os.path.basename(str(p))),
                            real(p, *a, **k))[1])

    _counts, scores, _audit = load_regression_input_pairs(pairs)

    assert seen.count("merged_measurements.parquet") == 1
    assert len(scores) == 4


def test_the_offer_dies_with_the_producers_frame(tmp_path):
    """A strong reference here would make every merged frame in a session
    immortal -- gigabytes held by a module nobody thinks to clear."""
    path = tmp_path / "merged_measurements.parquet"
    frame = pd.DataFrame({"plateID": ["plate1"], "cell_area": [1.0]})
    frame_handoff.hold(path, frame)
    assert frame_handoff.held(path) is frame

    del frame
    gc.collect()

    assert frame_handoff.held(path) is None
    assert frame_handoff.describe(path) == ""


def test_the_held_frame_is_not_stamped_with_a_plate(tmp_path):
    """The loader assigns plateID onto a score file that names none. Writing
    that onto the frame the producer still owns would put one plate's id into
    a frame holding every plate."""
    score_path = tmp_path / "merged_measurements.parquet"
    offered = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                            "cell_area": [1.0]})
    tabular.write_table(offered, score_path)
    counts = tmp_path / "count1.csv"
    pd.DataFrame([{"plateID": "plate7", "rowID": "r1", "columnID": "c1",
                   "grna": "g1", "count": 3}]).to_csv(counts, index=False)
    frame_handoff.hold(score_path, offered)
    try:
        _counts, scores, _audit = load_regression_input_pairs(
            [{"score": str(score_path), "count": str(counts)}])
    finally:
        frame_handoff.release(score_path)

    assert list(scores["plateID"]) == ["plate7"]
    assert "plateID" not in offered.columns


def test_the_route_that_produced_the_input_is_announced(
        merged_and_counts, capsys):
    """Between the merge finishing and the fit starting the run printed
    nothing for minutes, which is why a running fit was reported as dead."""
    frame, score_path, pairs = merged_and_counts
    tabular.write_table(frame, score_path)

    load_regression_input_pairs(pairs)
    read_from_disk = capsys.readouterr().out
    assert "Reading merged_measurements.parquet" in read_from_disk
    assert "MB" in read_from_disk
    assert "4 rows in" in read_from_disk

    frame_handoff.hold(score_path, frame)
    try:
        load_regression_input_pairs(pairs)
    finally:
        frame_handoff.release(score_path)
    handed_over = capsys.readouterr().out
    assert "handed over in memory, not parsed" in handed_over
    assert "Reading merged_measurements.parquet" not in handed_over


def test_every_stage_of_the_fit_says_it_started(capsys):
    """A step that prints nothing for minutes cannot be told from a hung one."""
    settings = {}
    _stage(settings, "fitting the model")

    assert "Regression: fitting the model" in capsys.readouterr().out
    assert settings["_regression_stage"] == "fitting the model"


def test_staging_writes_a_columnar_artefact_that_reads_back(tmp_path):
    """The artefact is kept so a user can open it and so every fit of a queue
    reads the same numbers; only its format changes."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(2000, 12)),
                         columns=[f"feature_{i}" for i in range(12)])
    frame["plateID"] = "plate1"

    path = frame_handoff.stage(frame, tmp_path, "merged_measurements")

    assert path.endswith(".parquet")
    assert frame_handoff.held(path) is frame
    back = tabular.read_table(path)
    assert list(back.columns) == list(frame.columns)
    assert np.allclose(back["feature_0"], frame["feature_0"])

    as_csv = tmp_path / "same.csv"
    frame.to_csv(as_csv, index=False)
    assert os.path.getsize(path) < os.path.getsize(as_csv), (
        "a columnar artefact that is not smaller than the CSV it replaces "
        "has not paid for the change")
    frame_handoff.release(path)


def test_staging_falls_back_to_csv_without_a_parquet_engine(tmp_path,
                                                            monkeypatch):
    """An environment with no Parquet engine still gets an artefact, and is
    told which format it got rather than being failed."""
    monkeypatch.setattr(frame_handoff, "_columnar_engine", lambda: None)
    said = []
    frame = pd.DataFrame({"plateID": ["plate1"], "cell_area": [2.0]})

    path = frame_handoff.stage(frame, tmp_path, "merged_measurements",
                               report=said.append)

    assert path.endswith(".csv")
    assert "no Parquet engine installed" in said[0]
    assert tabular.read_table(path)["cell_area"].tolist() == [2.0]
    frame_handoff.release(path)


def test_a_frame_offered_for_nothing_is_refused(tmp_path):
    """Holding None would answer a later reader with a frame that is not one."""
    with pytest.raises(ValueError, match="no frame"):
        frame_handoff.hold(tmp_path / "m.parquet", None)
    with pytest.raises(ValueError, match="no frame"):
        frame_handoff.stage(None, tmp_path, "merged_measurements")


def test_releasing_says_how_many_offers_it_withdrew(tmp_path):
    """A producer that has finished can make the fallback to the file
    deterministic instead of waiting for the frame to be collected."""
    first = pd.DataFrame({"a": [1]})
    second = pd.DataFrame({"a": [2]})
    frame_handoff.hold(tmp_path / "one.parquet", first)
    frame_handoff.hold(tmp_path / "two.parquet", second)

    assert frame_handoff.release(tmp_path / "one.parquet") == 1
    assert frame_handoff.release(tmp_path / "one.parquet") == 0
    assert frame_handoff.held(tmp_path / "two.parquet") is second
    assert frame_handoff.release() >= 1
    assert frame_handoff.held(tmp_path / "two.parquet") is None


def test_a_path_that_is_not_one_is_not_an_offer():
    """`held` is asked about whatever a pair row carries, which need not be a
    path at all; the answer is the same as for a path nobody offered."""
    assert frame_handoff.held(object()) is None
