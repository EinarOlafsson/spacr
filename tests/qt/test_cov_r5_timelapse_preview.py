"""Timelapse preview: the branches the panel takes when things are missing.

Pins the halves of ``timelapse_preview`` that only run when something the
happy path always supplies is *absent* — a movie panel that was never
attached, a sequence that was never loaded, a sampler that has never
enumerated, a memory-map that is already open, a sibling field that fails
or is cancelled — plus the movie-field queue's cache, trim and
generation-guard decisions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import timelapse_preview as TP


H = W = 8


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def _field_folder(parent: Path, name: str, value: int = 1,
                  frames: int = 2) -> Path:
    folder = parent / name
    folder.mkdir()
    for frame in range(frames):
        np.save(folder / f"frame_{frame}.npy",
                np.full((H, W), value + frame, dtype=np.uint16))
    return folder


def _blob_masks(n: int = 2) -> np.ndarray:
    masks = np.zeros((n, H, W), dtype=np.int32)
    masks[:, 2:5, 2:5] = 1
    return masks


def _stub_segmenter(monkeypatch):
    """Segment every frame into the same square, without Cellpose."""
    def segment(image, _params):
        mask = np.zeros(np.asarray(image).shape[:2], dtype=np.int32)
        mask[2:5, 2:5] = 1
        return mask
    monkeypatch.setattr(TP, "segment_frame", segment)
    monkeypatch.setattr(TP, "link_tracks", lambda *_a, **_k: None)


def _panel(qtbot, monkeypatch):
    _stub_segmenter(monkeypatch)
    panel = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


# ---------------------------------------------------------------------------
# track_stats / render_frame without coordinates
# ---------------------------------------------------------------------------

def test_a_track_table_without_xy_still_scores_length_but_not_swaps():
    """Displacement indicators need x/y; the length indicators do not."""
    without = pd.DataFrame({"frame": [0, 1, 0], "track_id": [1, 1, 2]})
    with_xy = without.assign(x=[0.0, 40.0, 0.0], y=[0.0, 0.0, 0.0])

    bare = TP.track_stats(without, n_frames=2, min_length=2,
                          displacement_limit=5.0)
    full = TP.track_stats(with_xy, n_frames=2, min_length=2,
                          displacement_limit=5.0)

    assert bare.n_tracks == full.n_tracks == 2
    assert bare.suspicious_jumps == 0 and bare.max_step == 0.0
    assert full.suspicious_jumps == 1 and full.max_step == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# cancellation plumbing
# ---------------------------------------------------------------------------

def test_a_missing_qthread_reads_as_interrupted(monkeypatch):
    """Off a QThread the flag cannot be read, so the field must not proceed."""
    assert TP.movie_worker_interrupted() is False

    class NoThread:
        @staticmethod
        def currentThread():
            raise RuntimeError("no QThread for this thread")

    monkeypatch.setattr(TP, "QThread", NoThread)

    assert TP.movie_worker_interrupted() is True


def test_reading_a_sequence_stops_at_the_frame_the_cancel_arrives_on(tmp_path):
    """``_read_sequence_frames`` checks between every frame, not just once."""
    folder = _field_folder(tmp_path, "field", value=1, frames=3)
    seq = TP.FrameSequence.open(folder, max_frames=3)

    whole = TP._read_sequence_frames(seq)
    assert whole.shape == (3, H, W)
    assert whole[:, 0, 0].tolist() == [1, 2, 3]

    reads = []

    def cancel_after_two():
        reads.append(len(reads))
        return len(reads) > 2

    with pytest.raises(TP.MovieFieldCancelled):
        TP._read_sequence_frames(seq, cancelled=cancel_after_two)


def test_frames_that_segment_to_different_shapes_are_refused(tmp_path,
                                                             monkeypatch):
    """A mixed-shape folder is not one field of view, and says so."""
    folder = _field_folder(tmp_path, "mixed", value=1)
    shapes = [(H, W), (H + 2, W)]

    def segment(_image, _params):
        return np.zeros(shapes.pop(0), dtype=np.int32)

    monkeypatch.setattr(TP, "segment_frame", segment)
    seq = TP.FrameSequence.open(folder, max_frames=2)

    with pytest.raises(ValueError) as excinfo:
        TP._read_and_segment_sequence(seq, {})

    assert "not a single field of view" in str(excinfo.value)


# ---------------------------------------------------------------------------
# run_preview_pass / build_movie_field / movie_field_payload
# ---------------------------------------------------------------------------


def test_a_movie_field_reuses_cached_masks_and_checks_their_frame_count(
        tmp_path, monkeypatch):
    """Cached masks skip segmentation, but only if they still fit the field."""
    folder = _field_folder(tmp_path, "field", value=2)
    segmented = []

    def segment(image, _params):
        segmented.append(int(np.asarray(image)[0, 0]))
        return np.zeros((H, W), dtype=np.int32)

    monkeypatch.setattr(TP, "segment_frame", segment)
    monkeypatch.setattr(TP, "link_tracks", lambda *_a, **_k: None)

    reused = TP.build_movie_field(
        folder, max_frames=2, seg={"channel": 0}, track={},
        cached_masks=_blob_masks())

    assert segmented == [], "cached masks were re-segmented"
    assert reused["segmented"] is False
    assert reused["images"][:, 0, 0].tolist() == [2, 3]

    payload = TP.movie_field_payload(
        path=folder, max_frames=2, seg={"channel": 0}, track={},
        cached_masks=_blob_masks(3))

    assert payload["source"] == str(folder)
    assert "cached masks have 3 frames" in payload["error"]
    assert "cancelled" not in payload


# ---------------------------------------------------------------------------
# panel: shutdown / propagation / playback guards
# ---------------------------------------------------------------------------

def test_shutdown_skips_a_runner_that_was_never_built(qtbot, monkeypatch):
    """``shutdown`` is the teardown path; a half-built panel must survive it."""
    panel = _panel(qtbot, monkeypatch)
    asked = []
    monkeypatch.setattr(panel._jobs, "shutdown",
                        lambda *_a, **_k: asked.append("_jobs"))
    movie_jobs = panel._movie_jobs
    monkeypatch.setattr(movie_jobs, "shutdown",
                        lambda *_a, **_k: asked.append("_movie_jobs"))

    panel._movie_jobs = None
    panel.shutdown()
    assert asked == ["_jobs"]

    panel._movie_jobs = movie_jobs
    panel.shutdown()
    assert asked == ["_jobs", "_jobs", "_movie_jobs"]


def test_propagation_is_silent_until_a_callback_is_registered(qtbot,
                                                              monkeypatch):
    """The toggle and the button both go through one unwired-safe path."""
    panel = _panel(qtbot, monkeypatch)
    seen = []

    panel.propagate_settings()
    panel._on_propagate_toggled(False)
    assert seen == []

    panel.set_propagate_callback(seen.append)
    panel._on_propagate_toggled(False)
    assert seen == [], "un-ticking the toggle propagated"

    panel._on_propagate_toggled(True)
    assert len(seen) == 1
    assert seen[0]["timelapse_mode"] == panel._mode_box.currentText()


def test_stopping_playback_before_the_button_exists_still_stops_the_timer(
        qtbot, monkeypatch):
    """``_stop_playback`` runs during teardown, so it may outlive its button."""
    panel = _panel(qtbot, monkeypatch)
    panel._frame_slider.setMaximum(4)

    panel._toggle_playback()
    assert panel._play_timer.isActive()
    assert panel._play_btn.text() == "Pause"

    panel._stop_playback()
    assert not panel._play_timer.isActive()
    assert panel._play_btn.text() == "Play"

    panel._toggle_playback()
    button = panel._play_btn
    # The guard exists for the window between the button going away and the
    # timer being stopped; nothing public removes it, so remove it here.
    del panel._play_btn
    panel._stop_playback()

    assert not panel._play_timer.isActive()
    assert button.text() == "Pause", "the guard reached the removed button"
    panel._play_btn = button


# ---------------------------------------------------------------------------
# panel: selectors without a sequence
# ---------------------------------------------------------------------------

def test_the_selectors_refresh_before_any_sequence_is_loaded(qtbot, tmp_path,
                                                             monkeypatch):
    """Nothing to sample yet, so only the channel dropdown is rebuilt."""
    panel = _panel(qtbot, monkeypatch)

    panel._refresh_source_selectors()
    assert panel._fov_box.count() == 0
    assert panel.sample_note() == ""

    _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)

    assert panel._fov_box.count() >= 1
    assert panel._channel_box.count() >= 1


def test_a_new_cap_only_restates_the_sample_once_there_is_one(qtbot, tmp_path,
                                                              monkeypatch):
    """With no sequence there is no sample sentence to put in the status bar."""
    panel = _panel(qtbot, monkeypatch)
    idle = panel._status.text()

    panel._on_max_sets_changed(7)
    assert panel._sampler.max_sets == 7
    assert panel._status.text() == idle
    assert panel.sample_note() == ""

    for name in ("field_a", "field_b", "field_c"):
        _field_folder(tmp_path, name)
    assert panel.load_sequence(tmp_path / "field_b")

    panel._on_max_sets_changed(2)

    note = panel.sample_note()
    assert note and panel._status.text() == note[:1].upper() + note[1:]


# ---------------------------------------------------------------------------
# panel: movie sources
# ---------------------------------------------------------------------------

def test_the_movie_has_no_sources_until_a_sequence_is_loaded(qtbot, tmp_path,
                                                             monkeypatch):
    panel = _panel(qtbot, monkeypatch)

    assert panel._movie_source_paths() == []

    _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)

    assert panel._movie_source_paths()[0] == str(selected)


def test_a_listing_entry_with_no_file_is_skipped_not_fatal(qtbot, tmp_path,
                                                           monkeypatch):
    """One unusable sampler entry must not cost the whole sibling listing."""
    from spacr.qt.widgets.preview_controls import ImageSet

    panel = _panel(qtbot, monkeypatch)
    sibling = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)

    # `_sampler` is the only route to `sets`, and nothing public can put a
    # channel-less set in it; an ImageSet with no files raises from `path()`.
    broken = ImageSet(key=("", "", "gone"), directory=str(tmp_path),
                      channels={})
    panel._sampler._sets = [broken] + list(panel._sampler.sets)

    paths = panel._movie_source_paths()

    assert paths[0] == str(selected)
    assert str(sibling) in paths
    assert not any("gone" in path for path in paths)


def test_an_unenumerated_sampler_falls_back_to_listing_the_siblings(
        qtbot, tmp_path, monkeypatch):
    """The synchronous caller never fills the sampler, so the movie re-lists."""
    from spacr.qt.widgets.preview_controls import ImageSetSampler

    panel = _panel(qtbot, monkeypatch)
    sibling = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)
    assert panel._sampler.sets, "the fixture never enumerated in the first place"

    panel._sampler = ImageSetSampler(10)
    paths = panel._movie_source_paths()

    assert paths == [str(selected), str(sibling)]


# ---------------------------------------------------------------------------
# panel: no movie panel attached
# ---------------------------------------------------------------------------


def test_cancelling_a_pending_field_without_a_runner_still_forgets_it(
        qtbot, monkeypatch):
    panel = _panel(qtbot, monkeypatch)
    cancels = []
    runner = panel._movie_jobs
    monkeypatch.setattr(runner, "cancel", lambda: cancels.append(1))

    panel._movie_jobs = None
    panel._movie_pending_path = "/somewhere/field_a"
    panel._movie_pending_key = ("seg", "track")
    generation = panel._movie_generation
    panel._cancel_pending_movie_field()

    assert panel._movie_pending_path is None
    assert panel._movie_pending_key is None
    assert panel._movie_generation == generation + 1
    assert cancels == []

    panel._movie_jobs = runner
    panel._cancel_pending_movie_field()
    assert cancels == [1]


# ---------------------------------------------------------------------------
# panel: freeze / worker result
# ---------------------------------------------------------------------------

def test_a_list_valued_setting_is_frozen_element_by_element(qtbot,
                                                            monkeypatch):
    """Frame limits arrive as a list, which is not hashable as a key."""
    panel = _panel(qtbot, monkeypatch)

    frozen = panel._freeze_movie_value(
        {"limits": [0, 12], "nested": [{"a": [1, 2]}], "n": 3})

    assert frozen == (("limits", (0, 12)),
                      ("n", 3),
                      ("nested", ((("a", (1, 2)),),)))
    assert hash(frozen)
