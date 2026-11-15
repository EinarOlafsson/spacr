"""The timelapse movie: one field's playback, the multi-field panel and its queue.

The movie exists for watching a track break, so a field with nothing to play
must say so and stop its timer.  The tracking preview feeds the movie panel one
sibling field at a time through a cached, generation-guarded queue; every
queue decision (cap, pending field, failure memory, mask reuse, stale results,
cancellation) is driven here with ``movie_field_payload`` replaced by a
recorder, and every case asserts exactly which fields were asked for.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import timelapse_preview as TP  # noqa: E402
from spacr.qt.widgets.timelapse_movie import FovMovie, TimelapseMoviePanel  # noqa: E402

pytestmark = pytest.mark.qt

H = W = 8


def _field(frames: int = 4, size: int = 32) -> dict:
    """One blob crossing the field, with a track that follows it."""
    images = np.zeros((frames, size, size), np.uint16)
    labels = np.zeros((frames, size, size), np.int32)
    for t in range(frames):
        images[t, 8:16, 4 + 3 * t:12 + 3 * t] = 900
        labels[t, 8:16, 4 + 3 * t:12 + 3 * t] = 1
    tracks = pd.DataFrame([{"frame": t, "x": 8 + 3 * t, "y": 12,
                            "track_id": 1} for t in range(frames)])
    return {"images": images, "labels": labels, "tracks": tracks}


def _field_folder(parent: Path, name: str, value: int = 1, frames: int = 2) -> Path:
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


def _preview(qtbot, monkeypatch):
    """A preview panel whose segmenter and linker never touch Cellpose."""
    def segment(image, _params):
        mask = np.zeros(np.asarray(image).shape[:2], dtype=np.int32)
        mask[2:5, 2:5] = 1
        return mask

    monkeypatch.setattr(TP, "segment_frame", segment)
    monkeypatch.setattr(TP, "link_tracks", lambda *_a, **_k: None)
    panel = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    return panel


def _seed_preview(panel, masks=None):
    """The state a finished preview pass leaves behind."""
    masks = _blob_masks() if masks is None else masks
    panel._masks = masks
    panel._tracked = masks * 5
    panel._tracks = None
    panel._movie_images = np.ones_like(masks, dtype=np.uint16)
    return masks


def _movie_panel(qtbot, max_fields=None):
    movie = TimelapseMoviePanel()
    qtbot.addWidget(movie)
    if max_fields is not None:
        movie.set_max_fields(max_fields)
    return movie


def _movie_setup(qtbot, tmp_path, monkeypatch, *, max_fields: int = 2):
    """A preview with a movie panel, a loaded field, one sibling and a recorder."""
    sibling = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        masks = _blob_masks()
        return {"source": str(kwargs["path"]), "title": Path(kwargs["path"]).name,
                "images": np.zeros_like(masks, dtype=np.uint16), "masks": masks,
                "labels": masks * 3, "tracks": None, "channel": 0,
                "segmented": kwargs["cached_masks"] is None}

    monkeypatch.setattr(TP, "movie_field_payload", build)
    panel = _preview(qtbot, monkeypatch)
    movie = _movie_panel(qtbot, max_fields)
    panel.attach_movie_panel(movie)
    assert panel.load_sequence(selected)
    _seed_preview(panel)
    return panel, movie, selected, sibling, calls


def _paths(calls):
    return [str(call["path"]) for call in calls]


def test_a_field_movie_says_when_there_is_nothing_to_play(qtbot, monkeypatch,
                                                          subtests):
    """No frames, an unrenderable frame, or frames that vanish mid-play leave a sentence and a stopped timer."""
    from spacr.qt.widgets import timelapse_preview

    def movie():
        widget = FovMovie(title="field 1")
        qtbot.addWidget(widget)
        widget.resize(400, 320)
        return widget

    with subtests.test(msg="unbound"):
        widget = movie()
        widget.set_sequence(**_field())
        widget.set_sequence(None)
        assert (widget.frame_count(), widget._rendered(0), widget._canvas.text(),
                widget._counter.text(), widget._strip._cells) == (
            0, None, "No frames loaded.", "0 / 0", [])
    with subtests.test(msg="unrenderable-frames-left-out"):
        widget = movie()

        def refuse(*_a, **_k):
            raise ValueError("labels do not match the image")

        with monkeypatch.context() as patch:
            patch.setattr(timelapse_preview, "render_frame", refuse)
            widget.set_sequence(**_field())
            assert widget.frame_count() == 4
            assert widget._rendered(0) is None
            assert widget._strip._cells == []
            assert widget._canvas.text() == "No frames loaded."
    with subtests.test(msg="timer-stops-when-frames-go-away"):
        widget = movie()
        widget.set_sequence(**_field())
        widget.play()
        assert widget._timer.isActive()
        widget.set_sequence(**_field(frames=1))
        widget._advance()
        assert (widget._timer.isActive(), widget._play.text(), widget._frame) == (
            False, "Play", 0)


def test_the_movie_panel_drops_surplus_fields_and_plays_them_all_together(
        qtbot, subtests):
    """A shorter field list releases movies; Play all starts and stops every field."""
    with subtests.test(msg="fewer-fields"):
        panel = _movie_panel(qtbot)
        panel.set_fields([dict(title="a", **_field()), dict(title="b", **_field())])
        assert len(panel.movies()) == 2
        panel.movies()[1].play()
        panel.set_fields([dict(title="a", **_field())])
        assert [m._title.text() for m in panel.movies()] == ["a"]
    with subtests.test(msg="play-all"):
        panel = _movie_panel(qtbot)
        panel.set_fields([dict(title="a", **_field()), dict(title="b", **_field())])
        panel._toggle_all()
        assert [m._timer.isActive() for m in panel.movies()] == [True, True]
        assert panel._play_all.text() == "Pause all"
        panel._toggle_all()
        assert [m._timer.isActive() for m in panel.movies()] == [False, False]
        assert panel._play_all.text() == "Play all"


def test_the_preview_publishes_and_resets_movie_fields_only_with_a_panel(
        qtbot, tmp_path, monkeypatch, subtests):
    """No panel means nothing desired or published; a reset without a runner keeps the panel unless asked."""
    with subtests.test(msg="no-movie-panel"):
        panel = _preview(qtbot, monkeypatch)
        (tmp_path / "unattached").mkdir()
        selected = _field_folder(tmp_path / "unattached", "field_b")
        assert panel.load_sequence(selected)
        _seed_preview(panel)
        panel._movie_sources = [str(selected)]
        panel._movie_fields[str(selected)] = {
            "source": str(selected), "title": selected.name,
            "images": panel._masks, "masks": panel._masks,
            "labels": panel._tracked, "tracks": None, "channel": 0,
            "_seg_key": panel._movie_seg_key,
            "_track_key": panel._movie_track_key, "_needs_refresh": False}
        assert panel._desired_movie_sources() == []
        panel._present_movie_fields()
        movie = _movie_panel(qtbot)
        panel._movie_panel = movie
        assert panel._desired_movie_sources() == [str(selected)]
        panel._present_movie_fields()
        assert len(movie.movies()) == 1
    with subtests.test(msg="reset-without-a-runner"):
        panel = _preview(qtbot, monkeypatch)
        movie = _movie_panel(qtbot)
        panel.attach_movie_panel(movie)
        (tmp_path / "reset").mkdir()
        assert panel.load_sequence(_field_folder(tmp_path / "reset", "field_b"))
        _seed_preview(panel)
        panel._push_to_movie()
        assert len(movie.movies()) == 1
        runner, generation = panel._movie_jobs, panel._movie_generation
        panel._movie_jobs = None
        panel._reset_movie_fields()
        assert (panel._movie_generation, panel._movie_fields,
                len(movie.movies())) == (generation + 1, {}, 1)
        panel._movie_jobs = runner
        panel._reset_movie_fields(clear_panel=True)
        assert movie.movies() == []


def test_the_movie_field_queue_asks_for_each_sibling_once_per_settings(
        qtbot, tmp_path, monkeypatch, subtests):
    """One field in flight, failures remembered per key, masks reused on re-link and dropped on re-segment."""
    with subtests.test(msg="outside-the-cap-is-forgotten"):
        (tmp_path / "cap").mkdir()
        panel, _m, selected, _s, _c = _movie_setup(
            qtbot, tmp_path / "cap", monkeypatch, max_fields=1)
        panel._push_to_movie()
        wanted = (panel._movie_seg_key, panel._movie_track_key)
        assert panel._desired_movie_sources() == [str(selected)]
        panel._movie_fields["/gone/field_z"] = {"_seg_key": None}
        panel._movie_failures["/gone/field_z"] = wanted
        panel._movie_failures[str(selected)] = wanted
        panel._refresh_movie_targets()
        assert set(panel._movie_fields) == {str(selected)}
        assert panel._movie_failures == {str(selected): wanted}
    with subtests.test(msg="a-pending-field-blocks-the-next"):
        (tmp_path / "pending").mkdir()
        panel, _m, selected, sibling, calls = _movie_setup(
            qtbot, tmp_path / "pending", monkeypatch)
        panel._push_to_movie()
        assert _paths(calls) == [str(sibling)]
        calls.clear()
        panel._movie_fields.pop(str(sibling))
        panel._movie_pending_path = str(sibling)
        panel._movie_pending_key = (panel._movie_seg_key, panel._movie_track_key)
        panel._refresh_movie_targets()
        assert calls == []
        panel._movie_pending_path = None
        panel._refresh_movie_targets()
        assert _paths(calls) == [str(sibling)]
        assert set(panel._movie_fields) == {str(selected), str(sibling)}
    with subtests.test(msg="a-failure-is-not-retried-under-the-same-key"):
        (tmp_path / "failed").mkdir()
        panel, _m, _sel, sibling, calls = _movie_setup(
            qtbot, tmp_path / "failed", monkeypatch)
        monkeypatch.setattr(TP, "movie_field_payload", lambda **kw: calls.append(
            kw) or {"error": "unreadable", "source": str(kw["path"])})
        panel._push_to_movie()
        assert _paths(calls) == [str(sibling)]
        assert panel._movie_failures == {
            str(sibling): (panel._movie_seg_key, panel._movie_track_key)}
        assert f"Movie field {sibling.name} failed: unreadable" in \
            panel._status.text()
        panel._refresh_movie_targets()
        assert len(calls) == 1
        panel._movie_failures.clear()
        panel._refresh_movie_targets()
        assert len(calls) == 2
    with subtests.test(msg="a-tracking-change-relinks-cached-masks"):
        (tmp_path / "relink").mkdir()
        panel, _m, _sel, sibling, calls = _movie_setup(
            qtbot, tmp_path / "relink", monkeypatch)
        panel._push_to_movie()
        assert calls[-1]["cached_masks"] is None
        first_masks = panel._movie_fields[str(sibling)]["masks"]
        calls.clear()
        panel._displacement.setValue(panel._displacement.value() + 5)
        panel._push_to_movie()
        assert _paths(calls) == [str(sibling)]
        assert calls[0]["cached_masks"] is first_masks
        assert panel._movie_failures == {}
    with subtests.test(msg="a-segmentation-change-drops-cached-masks"):
        (tmp_path / "reseg").mkdir()
        panel, _m, _sel, sibling, calls = _movie_setup(
            qtbot, tmp_path / "reseg", monkeypatch)
        panel._push_to_movie()
        panel._movie_failures["/gone/field_z"] = ("stale",)
        calls.clear()
        panel._diameter.setValue(panel._diameter.value() + 7)
        panel._push_to_movie()
        assert _paths(calls) == [str(sibling)]
        assert calls[0]["cached_masks"] is None
        assert panel._movie_failures == {}


def test_movie_field_results_are_kept_only_for_the_current_generation(
        qtbot, tmp_path, monkeypatch, subtests):
    """A superseded result is dropped, a cancel frees the queue unblamed, masks stand in until frames arrive."""
    payload_for = lambda sibling: {  # noqa: E731
        "source": str(sibling), "title": sibling.name, "images": _blob_masks(),
        "masks": _blob_masks(), "labels": _blob_masks(), "tracks": None,
        "channel": 0}
    with subtests.test(msg="superseded-generation-dropped"):
        (tmp_path / "gen").mkdir()
        panel, _m, _sel, sibling, _calls = _movie_setup(
            qtbot, tmp_path / "gen", monkeypatch)
        panel._push_to_movie()
        panel._movie_fields.pop(str(sibling))
        key = (panel._movie_seg_key, panel._movie_track_key)
        panel._movie_pending_path = str(sibling)
        panel._on_movie_field_done(str(sibling), panel._movie_generation + 1,
                                   key, payload_for(sibling))
        assert str(sibling) not in panel._movie_fields
        assert panel._movie_pending_path == str(sibling)
        panel._on_movie_field_done(str(sibling), panel._movie_generation, key,
                                   payload_for(sibling))
        assert str(sibling) in panel._movie_fields
    with subtests.test(msg="cancelled-field-frees-the-queue"):
        (tmp_path / "cancel").mkdir()
        panel, _m, _sel, sibling, calls = _movie_setup(
            qtbot, tmp_path / "cancel", monkeypatch)
        panel._push_to_movie()
        panel._movie_fields.pop(str(sibling))
        key = (panel._movie_seg_key, panel._movie_track_key)
        calls.clear()
        panel._on_movie_field_done(str(sibling), panel._movie_generation, key,
                                   {"cancelled": True, "source": str(sibling)})
        assert str(sibling) not in panel._movie_failures
        assert panel._movie_pending_path is None
        assert _paths(calls) == [str(sibling)]
        assert str(sibling) in panel._movie_fields
    with subtests.test(msg="frames-not-in-hand-flagged-for-a-reread"):
        (tmp_path / "reread").mkdir()
        panel, _m, selected, _sib, _calls = _movie_setup(
            qtbot, tmp_path / "reread", monkeypatch)
        monkeypatch.setattr(panel, "_refresh_movie_targets", lambda: None)
        masks = _seed_preview(panel)
        panel._movie_images = None
        panel._push_to_movie()
        entry = panel._movie_fields[str(selected)]
        assert (entry["images"] is masks, entry["_needs_refresh"]) == (True, True)
        images = np.full_like(masks, 4, dtype=np.uint16)
        panel._movie_images = images
        panel._push_to_movie()
        entry = panel._movie_fields[str(selected)]
        assert (entry["images"] is images, entry["_needs_refresh"]) == (True, False)


def test_attaching_a_movie_panel_rewires_its_cap_signal_exactly_once(
        qtbot, tmp_path, monkeypatch, subtests):
    """A replaced panel stops driving the queue, a re-attached one connects once, a destroyed one is replaced."""
    import shiboken6

    def recorder(panel):
        refreshes = []
        monkeypatch.setattr(panel, "_refresh_movie_targets",
                            lambda: refreshes.append(1))
        return refreshes

    with subtests.test(msg="second-panel-unhooks-the-first"):
        (tmp_path / "second").mkdir()
        panel, first, _sel, _sib, _calls = _movie_setup(
            qtbot, tmp_path / "second", monkeypatch)
        panel._push_to_movie()
        refreshes = recorder(panel)
        second = _movie_panel(qtbot)
        panel.attach_movie_panel(second)
        refreshes.clear()
        first.set_max_fields(1)
        assert refreshes == []
        second.set_max_fields(1)
        assert refreshes == [1]
    with subtests.test(msg="same-panel-connects-once"):
        (tmp_path / "same").mkdir()
        panel, movie, _sel, _sib, _calls = _movie_setup(
            qtbot, tmp_path / "same", monkeypatch)
        refreshes = recorder(panel)
        panel.attach_movie_panel(movie)
        panel.attach_movie_panel(movie)
        refreshes.clear()
        movie.set_max_fields(1)
        assert refreshes == [1]
    with subtests.test(msg="destroyed-panel-replaced"):
        panel = _preview(qtbot, monkeypatch)
        first = TimelapseMoviePanel()        # destroyed by hand, not by qtbot
        panel.attach_movie_panel(first)
        shiboken6.delete(first)
        assert not shiboken6.isValid(first)
        second = _movie_panel(qtbot)
        panel.attach_movie_panel(second)
        refreshes = recorder(panel)
        second.set_max_fields(1)
        assert (refreshes, panel._movie_panel is second) == ([1], True)
