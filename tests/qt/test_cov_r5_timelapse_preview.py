"""Timelapse preview: the branches the panel takes when things are missing.

Pins the halves of ``timelapse_preview`` that only run when something the
happy path always supplies is *absent* — a movie panel that was never
attached, a sequence that was never loaded, a sampler that has never
enumerated, a memory-map that is already open, a sibling field that fails
or is cancelled — plus the movie-field queue's cache, trim and
generation-guard decisions.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import timelapse_preview as TP


H = W = 8
N_FRAMES = 3


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


def _seed_preview(panel, masks=None):
    """Give the panel the state a finished preview pass would have left."""
    masks = _blob_masks() if masks is None else masks
    panel._masks = masks
    panel._tracked = masks * 5
    panel._tracks = None
    panel._movie_images = np.ones_like(masks, dtype=np.uint16)
    return masks


# ---------------------------------------------------------------------------
# FrameSequence: the second read of a memory-mapped stack
# ---------------------------------------------------------------------------

def test_a_memory_mapped_stack_is_mapped_once_and_reused(tmp_path):
    """The map is opened on the first page read and kept for every later one."""
    import tifffile

    path = tmp_path / "stack.tif"
    # One page holding a 3-D array — spelled out so the fixture cannot drift
    # into the multi-page (page-addressable) kind, which never memory-maps.
    tifffile.imwrite(str(path),
                     np.arange(N_FRAMES * H * W, dtype=np.uint16)
                     .reshape(N_FRAMES, H, W),
                     photometric="rgb", planarconfig="separate")
    seq = TP.FrameSequence.open(str(path), max_frames=N_FRAMES)
    assert seq.kind == "tiffmm", "the fixture stopped being a single-page stack"

    maps = []
    real_memmap = tifffile.memmap

    def counting_memmap(*args, **kwargs):
        maps.append(args)
        return real_memmap(*args, **kwargs)

    tifffile.memmap = counting_memmap
    try:
        first = seq.frame(0)
        second = seq.frame(2)
    finally:
        tifffile.memmap = real_memmap

    assert len(maps) == 1, "the stack was memory-mapped again for frame 2"
    assert first[0, 0] == 0
    assert second[0, 0] == np.uint16(2 * H * W)


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


def test_a_track_whose_points_are_all_nan_draws_no_head_marker():
    """Every coordinate is dropped, so there is nothing left to mark."""
    image = np.zeros((H, W), dtype=np.uint16)
    nan_track = pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1],
        "x": [np.nan, np.nan], "y": [np.nan, np.nan]})
    real_track = nan_track.assign(x=[2.0, 4.0], y=[2.0, 4.0])

    blank = TP.render_frame(image, tracks=nan_track, frame=1)
    drawn = TP.render_frame(image, tracks=real_track, frame=1)

    assert blank.max() == 0, "an all-NaN track painted something"
    assert drawn.max() > 0 and tuple(drawn[4, 4]) == TP.track_colour(1)


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

def test_a_pass_that_wants_images_reads_each_frame_exactly_once(tmp_path,
                                                                monkeypatch):
    """``include_images`` segments and keeps the raw frames in one read."""
    folder = _field_folder(tmp_path, "field", value=4)
    seq = TP.FrameSequence.open(folder, max_frames=2)
    _stub_segmenter(monkeypatch)

    result = TP.run_preview_pass(TP.TimelapseRequest(
        sequence=seq, seg={}, track={}, include_images=True))

    assert result["images"][:, 0, 0].tolist() == [4, 5]
    assert result["segmented"] is True
    assert seq.read_count == 2, "the frames were read a second time"


def test_cached_masks_still_get_their_raw_frames_read_for_the_movie(tmp_path,
                                                                    monkeypatch):
    """A re-link keeps the masks but the movie still needs the images."""
    folder = _field_folder(tmp_path, "field", value=9)
    seq = TP.FrameSequence.open(folder, max_frames=2)
    monkeypatch.setattr(TP, "link_tracks", lambda *_a, **_k: None)
    masks = _blob_masks()

    result = TP.run_preview_pass(TP.TimelapseRequest(
        sequence=seq, cached_masks=masks, seg={}, track={},
        include_images=True))

    assert result["masks_built"] is False
    assert result["segmented"] is False
    assert result["images"][:, 0, 0].tolist() == [9, 10]


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

def test_without_a_movie_panel_nothing_is_desired_and_nothing_is_published(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    panel = _panel(qtbot, monkeypatch)
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)
    _seed_preview(panel)
    panel._movie_sources = [str(selected)]
    panel._movie_fields[str(selected)] = {
        "source": str(selected), "title": selected.name,
        "images": panel._masks, "masks": panel._masks,
        "labels": panel._tracked, "tracks": None, "channel": 0,
        "_seg_key": panel._movie_seg_key, "_track_key": panel._movie_track_key,
        "_needs_refresh": False}

    assert panel._desired_movie_sources() == []
    panel._present_movie_fields()          # no panel: publishes nothing

    movie = TimelapseMoviePanel()
    qtbot.addWidget(movie)
    panel._movie_panel = movie
    assert panel._desired_movie_sources() == [str(selected)]
    panel._present_movie_fields()
    assert len(movie.movies()) == 1


def test_resetting_movie_fields_without_a_runner_leaves_the_panel_alone(
        qtbot, tmp_path, monkeypatch):
    """Cancelling is best-effort; clearing the panel is opt-in."""
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    panel = _panel(qtbot, monkeypatch)
    movie = TimelapseMoviePanel()
    qtbot.addWidget(movie)
    panel.attach_movie_panel(movie)
    selected = _field_folder(tmp_path, "field_b")
    assert panel.load_sequence(selected)
    _seed_preview(panel)
    panel._push_to_movie()
    assert len(movie.movies()) == 1

    runner = panel._movie_jobs
    panel._movie_jobs = None
    generation = panel._movie_generation
    panel._reset_movie_fields()

    assert panel._movie_generation == generation + 1
    assert panel._movie_fields == {}
    assert len(movie.movies()) == 1, "an opt-out reset still cleared the panel"

    panel._movie_jobs = runner
    panel._reset_movie_fields(clear_panel=True)
    assert movie.movies() == []


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


def test_a_result_with_no_signature_is_not_added_to_the_mask_cache(
        qtbot, monkeypatch):
    """Only a pass that declared a segmentation signature may cache masks."""
    panel = _panel(qtbot, monkeypatch)
    masks = _blob_masks()

    panel._pending_signature = None
    panel._on_worker_done(
        {"masks": masks, "tracks": None, "masks_built": True, "images": None},
        "")
    assert panel._mask_cache == {}
    assert panel._status.text().startswith("Masks built + linked")

    panel._pending_signature = ("sig", 1)
    panel._on_worker_done(
        {"masks": masks, "tracks": None, "masks_built": False, "images": None},
        "")
    assert list(panel._mask_cache) == [("sig", 1)]
    assert panel._status.text().startswith("Re-linked (cached masks)")


# ---------------------------------------------------------------------------
# panel: the movie-field queue
# ---------------------------------------------------------------------------

def _movie_setup(qtbot, tmp_path, monkeypatch, *, max_fields: int = 2):
    """A panel with a movie panel, a loaded field, a sibling, and a recorder."""
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    sibling = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        masks = _blob_masks()
        return {
            "source": str(kwargs["path"]),
            "title": Path(kwargs["path"]).name,
            "images": np.zeros_like(masks, dtype=np.uint16),
            "masks": masks,
            "labels": masks * 3,
            "tracks": None,
            "channel": 0,
            "segmented": kwargs["cached_masks"] is None,
        }

    monkeypatch.setattr(TP, "movie_field_payload", build)
    panel = _panel(qtbot, monkeypatch)
    movie = TimelapseMoviePanel()
    qtbot.addWidget(movie)
    movie.set_max_fields(max_fields)
    panel.attach_movie_panel(movie)
    assert panel.load_sequence(selected)
    _seed_preview(panel)
    return panel, movie, selected, sibling, calls


def test_fields_and_failures_outside_the_cap_are_forgotten(qtbot, tmp_path,
                                                           monkeypatch):
    """Lowering the cap must not leave arrays or verdicts for dropped fields."""
    panel, _movie, selected, _sibling, _calls = _movie_setup(
        qtbot, tmp_path, monkeypatch, max_fields=1)
    panel._push_to_movie()
    wanted = (panel._movie_seg_key, panel._movie_track_key)
    assert panel._desired_movie_sources() == [str(selected)]

    panel._movie_fields["/gone/field_z"] = {"_seg_key": None}
    panel._movie_failures["/gone/field_z"] = wanted
    panel._movie_failures[str(selected)] = wanted

    panel._refresh_movie_targets()

    assert "/gone/field_z" not in panel._movie_fields
    assert "/gone/field_z" not in panel._movie_failures
    assert str(selected) in panel._movie_failures, "a live verdict was dropped"
    assert str(selected) in panel._movie_fields


def test_one_field_at_a_time_a_pending_one_blocks_the_next(qtbot, tmp_path,
                                                           monkeypatch):
    """The queue is serialized: nothing is submitted while a field is in flight."""
    panel, _movie, selected, sibling, calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    assert [str(call["path"]) for call in calls] == [str(sibling)]
    calls.clear()

    panel._movie_fields.pop(str(sibling))
    panel._movie_pending_path = str(sibling)
    panel._movie_pending_key = (panel._movie_seg_key, panel._movie_track_key)
    panel._refresh_movie_targets()
    assert calls == [], "a second field was submitted alongside a pending one"

    panel._movie_pending_path = None
    panel._refresh_movie_targets()
    assert [str(call["path"]) for call in calls] == [str(sibling)]
    assert str(selected) in panel._movie_fields


def test_a_field_that_failed_is_not_retried_under_the_same_settings(
        qtbot, tmp_path, monkeypatch):
    """A broken sibling is asked once per settings identity, not in a loop."""
    panel, _movie, _selected, sibling, calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    monkeypatch.setattr(
        TP, "movie_field_payload",
        lambda **kwargs: calls.append(kwargs) or {
            "error": "unreadable", "source": str(kwargs["path"])})

    panel._push_to_movie()

    assert [str(call["path"]) for call in calls] == [str(sibling)]
    assert panel._movie_failures[str(sibling)] == (panel._movie_seg_key,
                                                   panel._movie_track_key)
    assert f"Movie field {sibling.name} failed: unreadable" in \
        panel._status.text()

    panel._refresh_movie_targets()
    assert len(calls) == 1, "the failed field was retried under the same key"

    panel._movie_failures.clear()
    panel._refresh_movie_targets()
    assert len(calls) == 2


def test_a_tracking_change_relinks_a_sibling_from_its_cached_masks(
        qtbot, tmp_path, monkeypatch):
    """Re-linking must not re-segment: the sibling's masks are handed back."""
    panel, _movie, _selected, sibling, calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    assert calls[-1]["cached_masks"] is None
    first_masks = panel._movie_fields[str(sibling)]["masks"]
    failures_before = dict(panel._movie_failures)
    calls.clear()

    panel._displacement.setValue(panel._displacement.value() + 5)
    panel._push_to_movie()

    assert [str(call["path"]) for call in calls] == [str(sibling)]
    assert calls[0]["cached_masks"] is first_masks
    assert failures_before == {}


def test_a_segmentation_change_throws_the_cached_sibling_masks_away(
        qtbot, tmp_path, monkeypatch):
    """A new mask identity invalidates every field, not just the linking."""
    panel, _movie, _selected, sibling, calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    panel._movie_failures["/gone/field_z"] = ("stale",)
    calls.clear()

    panel._diameter.setValue(panel._diameter.value() + 7)
    panel._push_to_movie()

    assert [str(call["path"]) for call in calls] == [str(sibling)]
    assert calls[0]["cached_masks"] is None, "stale masks were re-linked"
    assert panel._movie_failures == {}


def test_a_result_from_a_superseded_generation_is_dropped(qtbot, tmp_path,
                                                          monkeypatch):
    """A field whose queue was cancelled must not install itself late."""
    panel, _movie, _selected, sibling, _calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    panel._movie_fields.pop(str(sibling))
    key = (panel._movie_seg_key, panel._movie_track_key)
    payload = {"source": str(sibling), "title": sibling.name,
               "images": _blob_masks(), "masks": _blob_masks(),
               "labels": _blob_masks(), "tracks": None, "channel": 0}
    panel._movie_pending_path = str(sibling)

    panel._on_movie_field_done(str(sibling), panel._movie_generation + 1,
                               key, dict(payload))

    assert str(sibling) not in panel._movie_fields
    assert panel._movie_pending_path == str(sibling), "pending was cleared"

    panel._on_movie_field_done(str(sibling), panel._movie_generation, key,
                               dict(payload))
    assert str(sibling) in panel._movie_fields


def test_a_cancelled_field_frees_the_queue_without_being_recorded(
        qtbot, tmp_path, monkeypatch):
    """Cancellation is not a failure: the field is neither kept nor blamed."""
    panel, _movie, _selected, sibling, calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    panel._movie_fields.pop(str(sibling))
    key = (panel._movie_seg_key, panel._movie_track_key)
    calls.clear()

    panel._on_movie_field_done(str(sibling), panel._movie_generation, key,
                               {"cancelled": True, "source": str(sibling)})

    assert str(sibling) not in panel._movie_failures
    assert panel._movie_pending_path is None
    # Freeing the queue means the field is immediately asked for again.
    assert [str(call["path"]) for call in calls] == [str(sibling)]
    assert str(sibling) in panel._movie_fields


def test_a_field_whose_frames_are_not_in_hand_is_flagged_for_a_reread(
        qtbot, tmp_path, monkeypatch):
    """Masks stand in for the movie until the raw frames have been read."""
    panel, _movie, selected, _sibling, _calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    # The placeholder entry is deliberately short-lived: the very next
    # `_refresh_movie_targets` sees `_needs_refresh` and re-reads the field
    # on the movie worker, so the queue is stubbed out to look at it.
    monkeypatch.setattr(panel, "_refresh_movie_targets", lambda: None)
    masks = _seed_preview(panel)
    panel._movie_images = None

    panel._push_to_movie()

    entry = panel._movie_fields[str(selected)]
    assert entry["images"] is masks
    assert entry["_needs_refresh"] is True

    images = np.full_like(masks, 4, dtype=np.uint16)
    panel._movie_images = images
    panel._push_to_movie()

    entry = panel._movie_fields[str(selected)]
    assert entry["images"] is images
    assert entry["_needs_refresh"] is False


def test_attaching_a_second_movie_panel_unhooks_the_first(qtbot, tmp_path,
                                                          monkeypatch):
    """The old panel's cap must stop driving this preview's queue."""
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    panel, first, _selected, _sibling, _calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    panel._push_to_movie()
    refreshes = []
    monkeypatch.setattr(panel, "_refresh_movie_targets",
                        lambda: refreshes.append(1))

    second = TimelapseMoviePanel()
    qtbot.addWidget(second)
    panel.attach_movie_panel(second)
    refreshes.clear()

    first.set_max_fields(1)
    assert refreshes == [], "the detached panel still drives the queue"

    second.set_max_fields(1)
    assert refreshes == [1]


def test_reattaching_the_same_movie_panel_connects_it_only_once(qtbot,
                                                                tmp_path,
                                                                monkeypatch):
    """A screen that re-wires on every show must not multiply the connections."""
    panel, movie, _selected, _sibling, _calls = _movie_setup(
        qtbot, tmp_path, monkeypatch)
    refreshes = []
    monkeypatch.setattr(panel, "_refresh_movie_targets",
                        lambda: refreshes.append(1))

    panel.attach_movie_panel(movie)
    panel.attach_movie_panel(movie)
    refreshes.clear()

    movie.set_max_fields(1)

    assert refreshes == [1], "the cap signal reached the panel twice"


def test_a_movie_panel_that_was_destroyed_is_replaced_anyway(qtbot,
                                                             monkeypatch):
    """A screen torn down before its preview leaves a dangling movie panel.

    Reading a signal off a destroyed QObject raises, and that must not stop
    the preview adopting its new movie panel.
    """
    import shiboken6
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    panel = _panel(qtbot, monkeypatch)
    # Deliberately not handed to qtbot: this one is destroyed by hand.
    first = TimelapseMoviePanel()
    panel.attach_movie_panel(first)
    shiboken6.delete(first)
    assert not shiboken6.isValid(first)

    second = TimelapseMoviePanel()
    qtbot.addWidget(second)
    panel.attach_movie_panel(second)

    refreshes = []
    monkeypatch.setattr(panel, "_refresh_movie_targets",
                        lambda: refreshes.append(1))
    second.set_max_fields(1)
    assert refreshes == [1]
    assert panel._movie_panel is second
