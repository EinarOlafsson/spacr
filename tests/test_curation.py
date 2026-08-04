"""``B12`` ``C7`` — hand corrections, and the record of them.

Two claims carry the weight. That a brush stroke changes *exactly* the
elements a world-space ball covers — which on an anisotropic stack is not the
same set as a ball in array elements, and getting it wrong repaints slices the
user never saw. And that a join or a split leaves the track table consistent,
where consistent has a definition: one row per (track, frame).

The third, quieter claim is that nothing can be edited without leaving a
ledger entry, because that is what makes a curated dataset distinguishable
from a raw one.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr.curation import (CurationError, CurationLog, MaskCuration,
                            TrackCuration, is_curated, log_path_for)
from spacr.layers import LabelsLayer, Spacing


# ---------------------------------------------------------------------------
# The brush covers exactly the ball, under anisotropic spacing
# ---------------------------------------------------------------------------

def _stack_layer(shape=(9, 9, 9), z=2.0, xy=0.5):
    """A 3-D mask with real spaCR anisotropy: 2 µm z-steps, 0.5 µm pixels."""
    return LabelsLayer(np.zeros(shape, dtype=np.int64), name="mask",
                       spacing=Spacing.from_map({"z": z, "y": xy, "x": xy},
                                                units="um"))


def test_a_brush_covers_a_ball_in_world_units_not_in_elements():
    layer = _stack_layer()
    centre = {"z": 8.0, "y": 2.0, "x": 2.0}     # element (4, 4, 4)

    index = layer.brush_index(centre, radius=1.4)

    # 1.4 µm reaches 2 elements in y/x (0.5 µm each) and NOT ONE in z
    # (2 µm each) — the whole point of a world-space brush.
    zs, ys, xs = index
    assert set(zs.tolist()) == {4}
    assert set(ys.tolist()) == {2, 3, 4, 5, 6}
    assert set(xs.tolist()) == {2, 3, 4, 5, 6}


def test_a_brush_in_elements_would_have_been_a_cylinder():
    """The bug the world-space brush exists to avoid, stated as a contrast."""
    layer = _stack_layer()
    index = layer.brush_index({"z": 8.0, "y": 2.0, "x": 2.0}, radius=1.4)
    covered = set(zip(*(part.tolist() for part in index)))

    # A 2-element brush would have covered (2,4,4) and (6,4,4); 1.4 µm does
    # not, because those are 4 µm away.
    assert (2, 4, 4) not in covered
    assert (6, 4, 4) not in covered


def test_a_stroke_changes_exactly_the_elements_the_ball_covers():
    layer = _stack_layer()
    session = MaskCuration(layer, artifact="mask.tif")
    centre = {"z": 8.0, "y": 2.0, "x": 2.0}
    expected = layer.brush_index(centre, radius=1.4)

    changed = session.paint(centre, label=7, radius=1.4)

    painted = np.argwhere(layer.data == 7)
    assert changed == len(expected[0]) == len(painted)
    assert set(map(tuple, painted.tolist())) == set(
        zip(*(part.tolist() for part in expected)))
    # And nothing else moved.
    assert int(np.count_nonzero(layer.data)) == changed


def test_the_ball_is_a_ball_and_not_the_box_around_it():
    layer = _stack_layer(shape=(1, 21, 21), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.paint({"z": 0.0, "y": 10.0, "x": 10.0}, label=3, radius=5.0)

    painted = np.argwhere(layer.data == 3)
    distances = np.hypot(painted[:, 1] - 10.0, painted[:, 2] - 10.0)
    assert distances.max() <= 5.0 + 1e-9
    # A 5-unit box would be 11x11 = 121 elements; a disc is fewer.
    assert 0 < len(painted) < 121


def test_a_brush_off_the_edge_of_the_grid_paints_nothing_rather_than_raising():
    layer = _stack_layer()
    session = MaskCuration(layer)
    assert session.paint({"z": 500.0, "y": 500.0, "x": 500.0}, label=1) == 0
    assert not layer.data.any()


def test_a_brush_that_hangs_over_the_edge_paints_the_part_that_is_on_it():
    layer = _stack_layer(shape=(1, 11, 11), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    changed = session.paint({"z": 0.0, "y": 0.0, "x": 0.0}, label=4, radius=2.0)
    assert changed > 0
    assert layer.data[0, 0, 0] == 4
    assert changed == int(np.count_nonzero(layer.data))


def test_painting_the_label_that_is_already_there_changes_nothing(qtbot=None):
    layer = _stack_layer(shape=(1, 5, 5), z=2.0, xy=1.0)
    layer.data[:] = 6
    session = MaskCuration(layer)
    assert session.paint({"z": 0.0, "y": 2.0, "x": 2.0}, label=6,
                         radius=1.0) == 0
    assert len(session.log) == 0     # and nothing is recorded


def test_erasing_paints_background():
    layer = _stack_layer(shape=(1, 5, 5), z=2.0, xy=1.0)
    layer.data[:] = 6
    session = MaskCuration(layer)
    session.erase({"z": 0.0, "y": 2.0, "x": 2.0}, radius=1.0)
    assert layer.data[0, 2, 2] == 0


# ---------------------------------------------------------------------------
# One drag is one undo
# ---------------------------------------------------------------------------

def test_a_stroke_is_one_undo_however_many_dabs_it_took():
    layer = _stack_layer(shape=(1, 21, 21), z=2.0, xy=1.0)
    session = MaskCuration(layer)

    session.begin_stroke()
    for x in range(4, 16):
        session.paint({"z": 0.0, "y": 10.0, "x": float(x)}, label=2,
                      radius=1.0)
    session.end_stroke()

    assert len(session) == 1              # one stroke in the history
    assert layer.data.any()
    session.undo()
    assert not layer.data.any()


def test_undo_restores_the_labels_that_were_there_not_one_flat_value():
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    layer.data[0, :, :4] = 11
    layer.data[0, :, 4:] = 22
    before = layer.data.copy()
    session = MaskCuration(layer)

    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=99, radius=3.0)
    assert (layer.data == 99).any()

    session.undo()
    # Both original labels are back, in the right places — an undo that
    # restored "the" previous label would have flattened them into one.
    assert np.array_equal(layer.data, before)


def test_overlapping_dabs_undo_in_the_reverse_order_they_were_laid():
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    before = layer.data.copy()
    session = MaskCuration(layer)

    session.begin_stroke()
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=1, radius=2.0)
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=2, radius=2.0)
    session.end_stroke()
    session.undo()

    assert np.array_equal(layer.data, before)


def test_a_bare_dab_is_its_own_stroke_and_undoable():
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=5, radius=1.0)
    assert session.can_undo
    session.undo()
    assert not layer.data.any()


def test_a_stroke_that_changed_nothing_is_not_recorded():
    layer = _stack_layer(shape=(1, 5, 5), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.begin_stroke()
    assert session.end_stroke() is None
    assert len(session.log) == 0


def test_undo_with_nothing_to_undo_is_not_an_error():
    session = MaskCuration(_stack_layer(shape=(1, 3, 3), z=2.0, xy=1.0))
    assert session.undo() is None


def test_the_history_is_bounded_so_a_long_session_cannot_grow_forever():
    layer = _stack_layer(shape=(1, 41, 41), z=2.0, xy=1.0)
    session = MaskCuration(layer, history=3)
    for x in range(10):
        session.paint({"z": 0.0, "y": 20.0, "x": float(x * 4)}, label=x + 1,
                      radius=1.0)
    assert len(session) == 3
    # The LEDGER, unlike the undo history, keeps everything.
    assert len(session.log) == 10


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_every_stroke_lands_in_the_ledger_with_what_it_replaced():
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    layer.data[0, :, :] = 3
    session = MaskCuration(layer, artifact="mask.tif")
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=8, radius=2.0)

    edit = session.log.edits[-1]
    assert edit.kind == "paint"
    assert edit.target == 8
    assert edit.n_changed > 0
    assert edit.detail["replaced"] == [3]
    assert edit.detail["radius"] == 2.0


def test_the_ledger_records_what_was_painted_not_what_the_controls_hold_after():
    """A provenance record must state what HAPPENED.

    Reading the session's current label when the stroke closed made the
    ledger a confident, plausible, false statement whenever a caller passed
    an explicit label or moved the control mid-session.
    """
    layer = _stack_layer(shape=(1, 15, 15), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.label, session.radius = 1, 3.0

    session.paint({"z": 0.0, "y": 7.0, "x": 7.0}, label=42, radius=2.0)
    session.label, session.radius = 5, 9.0      # the controls moved after

    edit = session.log.edits[-1]
    assert edit.target == 42
    assert edit.detail["radius"] == 2.0


def test_a_stroke_of_two_labels_records_both_rather_than_the_last_one():
    layer = _stack_layer(shape=(1, 21, 21), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.begin_stroke()
    session.paint({"z": 0.0, "y": 5.0, "x": 5.0}, label=3, radius=1.0)
    session.paint({"z": 0.0, "y": 15.0, "x": 15.0}, label=4, radius=1.0)
    edit = session.end_stroke()
    assert edit.target == [3, 4]


def test_an_undo_is_appended_rather_than_erasing_the_paint_it_undid():
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    session = MaskCuration(layer)
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=1, radius=2.0)
    session.undo()

    kinds = [edit.kind for edit in session.log.edits]
    assert kinds == ["paint", "undo"]


def test_the_ledger_is_written_beside_the_artefact_and_read_back(tmp_path):
    mask = tmp_path / "plate1_A01_1.tif"
    mask.write_bytes(b"")
    layer = _stack_layer(shape=(1, 9, 9), z=2.0, xy=1.0)
    session = MaskCuration(layer, artifact=str(mask))
    session.paint({"z": 0.0, "y": 4.0, "x": 4.0}, label=1, radius=2.0)

    written = session.save_log()
    assert written == log_path_for(mask)
    reloaded = CurationLog.read(written)
    assert [e.kind for e in reloaded.edits] == ["paint"]
    assert reloaded.edits[0].n_changed == session.log.edits[0].n_changed


def test_a_ledger_is_valid_json_with_a_schema_version(tmp_path):
    log = CurationLog("mask.tif")
    log.append("paint", 3, n_changed=12, radius=2.0)
    path = log.write(tmp_path / "x.curation.json")
    data = json.loads((tmp_path / "x.curation.json").read_text())
    assert data["schema_version"] == 1
    assert data["edits"][0]["kind"] == "paint"
    assert path.endswith("x.curation.json")


def test_a_file_with_no_ledger_is_raw_and_one_with_edits_is_curated(tmp_path):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"")
    assert not is_curated(mask)

    log = CurationLog(str(mask))
    log.append("paint", 1, n_changed=5)
    log.write_beside(mask)
    assert is_curated(mask)


def test_an_empty_ledger_does_not_make_a_dataset_curated(tmp_path):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"")
    CurationLog(str(mask)).write_beside(mask)
    # Opened the brush, painted nothing: still raw.
    assert not is_curated(mask)


def test_an_unreadable_ledger_is_treated_as_curated_not_as_raw(tmp_path):
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"")
    with open(log_path_for(mask), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    # Suspicion, not certification.
    assert is_curated(mask)


def test_the_ledger_says_in_words_that_the_data_was_touched():
    log = CurationLog("mask.tif")
    assert "as the pipeline produced it" in log.describe()
    log.append("paint", 1, n_changed=5)
    assert "curated by hand" in log.describe()


def test_two_artefacts_in_one_folder_get_their_own_ledgers(tmp_path):
    assert log_path_for(tmp_path / "m.tif") != log_path_for(tmp_path / "m.npy")


# ---------------------------------------------------------------------------
# Track curation
# ---------------------------------------------------------------------------

def _tracks(spec):
    """``{track_id: [frames]}`` → a table with the canonical columns."""
    rows = []
    for track_id, frames in spec.items():
        for frame in frames:
            rows.append({"frame": frame, "track_id": track_id,
                         "original_label": 100 + frame,
                         "x": float(frame), "y": float(track_id)})
    return pd.DataFrame(rows, columns=["frame", "track_id", "original_label",
                                       "x", "y"])


def test_a_table_without_the_key_columns_is_refused():
    with pytest.raises(CurationError, match="track_id"):
        TrackCuration(pd.DataFrame({"frame": [0, 1]}))


def test_the_caller_s_frame_is_not_edited_underneath_them():
    frame = _tracks({1: [0, 1], 2: [2, 3]})
    before = frame.copy()
    TrackCuration(frame).join(1, 2)
    pd.testing.assert_frame_equal(frame, before)


# -- join --------------------------------------------------------------------

def test_joining_makes_the_second_track_a_continuation_of_the_first():
    session = TrackCuration(_tracks({1: [0, 1, 2], 2: [5, 6]}))
    edit = session.join(1, 2)

    assert session.track_ids == [1]
    assert session.frames_of(1) == [0, 1, 2, 5, 6]
    assert session.check() == []
    assert edit.kind == "join"
    assert edit.n_changed == 2
    assert edit.detail["absorbed"] == 2


def test_joining_tracks_that_overlap_in_time_is_refused():
    session = TrackCuration(_tracks({1: [0, 1, 2], 2: [2, 3]}))
    with pytest.raises(CurationError, match="frame"):
        session.join(1, 2)
    # And the table is untouched — refused, not half-done.
    assert session.track_ids == [1, 2]
    assert session.check() == []


def test_joining_a_track_to_itself_is_refused():
    session = TrackCuration(_tracks({1: [0, 1]}))
    with pytest.raises(CurationError, match="itself"):
        session.join(1, 1)


def test_joining_an_unknown_track_says_which_ones_exist():
    session = TrackCuration(_tracks({1: [0], 2: [1]}))
    with pytest.raises(CurationError, match="no track 99"):
        session.join(1, 99)


# -- split -------------------------------------------------------------------

def test_splitting_gives_the_tail_a_new_id_and_leaves_the_head_alone():
    session = TrackCuration(_tracks({1: [0, 1, 2, 3]}))
    edit = session.split(1, 2)

    assert session.frames_of(1) == [0, 1]
    new_id = edit.detail["new_track"]
    assert session.frames_of(new_id) == [2, 3]
    assert session.check() == []
    assert edit.n_changed == 2


def test_a_split_id_is_above_every_existing_one_rather_than_in_a_gap():
    session = TrackCuration(_tracks({1: [0, 1], 7: [0, 1]}))
    edit = session.split(1, 1)
    assert edit.detail["new_track"] == 8


def test_splitting_where_it_would_move_nothing_is_refused():
    session = TrackCuration(_tracks({1: [2, 3, 4]}))
    with pytest.raises(CurationError, match="one side empty"):
        session.split(1, 2)          # everything is already >= 2
    with pytest.raises(CurationError, match="one side empty"):
        session.split(1, 99)         # nothing is >= 99
    assert session.frames_of(1) == [2, 3, 4]


def test_a_split_then_a_join_gets_back_to_where_it_started():
    session = TrackCuration(_tracks({1: [0, 1, 2, 3]}))
    new_id = session.split(1, 2).detail["new_track"]
    session.join(1, new_id)

    assert session.track_ids == [1]
    assert session.frames_of(1) == [0, 1, 2, 3]
    assert session.check() == []
    assert [e.kind for e in session.log.edits] == ["split", "join"]


# -- delete ------------------------------------------------------------------

def test_deleting_removes_the_rows_and_records_what_went():
    session = TrackCuration(_tracks({1: [0, 1], 2: [0, 1, 2]}))
    edit = session.delete(2)

    assert session.track_ids == [1]
    assert len(session.tracks) == 2
    assert edit.n_changed == 3
    assert edit.detail["frames"] == [0, 1, 2]


def test_deleting_an_unknown_track_is_refused():
    session = TrackCuration(_tracks({1: [0]}))
    with pytest.raises(CurationError, match="no track"):
        session.delete(9)


# -- consistency -------------------------------------------------------------

def test_a_table_that_arrived_broken_is_shown_to_be_broken_not_hidden():
    broken = pd.concat([_tracks({1: [0, 1]}), _tracks({1: [1]})])
    problems = TrackCuration(broken).check()
    assert problems and "two places at one time" in problems[0]


def test_every_operation_leaves_the_table_consistent():
    session = TrackCuration(_tracks({1: [0, 1, 2, 3], 2: [6, 7], 3: [0, 1]}))
    new_id = session.split(1, 2).detail["new_track"]
    session.join(new_id, 2)
    session.delete(3)
    assert session.check() == []
    assert session.frames_of(1) == [0, 1]
    assert session.frames_of(new_id) == [2, 3, 6, 7]


def test_the_curated_table_comes_back_sorted_by_track_then_frame():
    session = TrackCuration(_tracks({2: [3, 1], 1: [2, 0]}))
    out = session.to_frame()
    assert list(zip(out["track_id"], out["frame"])) == [
        (1, 0), (1, 2), (2, 1), (2, 3)]


# -- persistence -------------------------------------------------------------

def test_saving_writes_the_table_and_its_ledger_together(tmp_path):
    session = TrackCuration(_tracks({1: [0, 1], 2: [4, 5]}))
    session.join(1, 2)
    target = tmp_path / "tracks" / "btrack_tracks_cell_plate1_A01_1.csv"

    written = session.save(target)

    assert pd.read_csv(written)["track_id"].tolist() == [1, 1, 1, 1]
    assert is_curated(written)
    reloaded = CurationLog.read_beside(written)
    assert [e.kind for e in reloaded.edits] == ["join"]


def test_the_ledger_json_survives_numpy_track_ids(tmp_path):
    frame = _tracks({1: [0, 1], 2: [4, 5]})
    frame["track_id"] = frame["track_id"].astype(np.int64)
    session = TrackCuration(frame)
    session.join(np.int64(1), np.int64(2))
    written = session.save(tmp_path / "t.csv")
    # json.dump would have raised on a numpy scalar; it did not.
    assert json.loads(open(log_path_for(written)).read())["edits"]


def test_a_curated_table_says_so_in_words():
    session = TrackCuration(_tracks({1: [0, 1], 2: [4, 5]}))
    assert "consistent" in session.describe()
    assert "as the pipeline produced it" in session.describe()
    session.join(1, 2)
    assert "curated by hand" in session.describe()
