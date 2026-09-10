"""The Timelapse movie receives real sibling fields, not one repeated view."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _field_folder(parent: Path, name: str, value: int = 1) -> Path:
    folder = parent / name
    folder.mkdir()
    for frame in range(2):
        image = np.full((8, 8), value + frame, dtype=np.uint16)
        np.save(folder / f"frame_{frame}.npy", image)
    return folder


def _tracked_masks() -> np.ndarray:
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[:, 2:5, 2:5] = 1
    return masks


def _fake_field(path, **_kwargs):
    masks = _tracked_masks()
    return {
        "source": str(path),
        "title": Path(path).name,
        "images": np.zeros_like(masks, dtype=np.uint16),
        "masks": masks,
        "labels": masks * 9,
        "tracks": None,
        "channel": 0,
        "segmented": True,
    }


def _seed_primary(panel) -> None:
    masks = _tracked_masks()
    panel._masks = masks
    panel._tracked = masks * 7
    panel._tracks = None
    panel._movie_images = np.ones_like(masks, dtype=np.uint16)


def test_a_movie_field_is_read_once_and_relabelled_by_its_own_tracks(
        tmp_path, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as preview

    source = _field_folder(tmp_path, "field_b", value=3)
    seen = []

    def segment(image, _params):
        seen.append(int(image[0, 0]))
        mask = np.zeros(image.shape, dtype=np.int32)
        mask[2:5, 2:5] = 1
        return mask

    tracks = pd.DataFrame({
        "frame": [0, 1],
        "original_label": [1, 1],
        "track_id": [7, 7],
        "x": [3.0, 3.0],
        "y": [3.0, 3.0],
    })
    monkeypatch.setattr(preview, "segment_frame", segment)
    monkeypatch.setattr(preview, "link_tracks", lambda *_a, **_k: tracks)

    result = preview.build_movie_field(
        source, max_frames=2, seg={"channel": 0}, track={"mode": "iou"})

    assert seen == [3, 4]
    assert result["images"][:, 0, 0].tolist() == [3, 4]
    assert set(np.unique(result["labels"])) == {0, 7}
    assert result["segmented"] is True


def test_cancellation_is_checked_between_expensive_frames(tmp_path,
                                                           monkeypatch):
    from spacr.qt.widgets import timelapse_preview as preview

    source = _field_folder(tmp_path, "field", value=1)
    segmented = []

    def segment(image, _params):
        segmented.append(int(image[0, 0]))
        return np.zeros(image.shape, dtype=np.int32)

    monkeypatch.setattr(preview, "segment_frame", segment)
    result = preview.movie_field_payload(
        path=source,
        max_frames=2,
        seg={"channel": 0},
        track={"mode": "iou"},
        cancelled=lambda: bool(segmented),
    )

    assert result == {"cancelled": True, "source": str(source)}
    assert segmented == [1]


def test_the_selected_field_is_first_and_siblings_stream_in_order(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as preview
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    first = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    third = _field_folder(tmp_path, "field_c")
    calls = []

    def build(**kwargs):
        calls.append(str(kwargs["path"]))
        return _fake_field(kwargs["path"])

    monkeypatch.setattr(preview, "movie_field_payload", build)
    panel = preview.TimelapsePreviewPanel(threaded=False)
    movie = TimelapseMoviePanel()
    qtbot.addWidget(panel)
    qtbot.addWidget(movie)
    movie.set_max_fields(3)
    panel.attach_movie_panel(movie)
    assert panel.load_sequence(selected)
    _seed_primary(panel)

    panel._push_to_movie()

    assert [item._title.text() for item in movie.movies()] == [
        selected.name, first.name, third.name]
    assert calls == [str(first), str(third)]


def test_raising_the_cap_keeps_the_first_field_and_loads_only_the_next(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as preview
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    first = _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    calls = []
    monkeypatch.setattr(
        preview, "movie_field_payload",
        lambda **kwargs: calls.append(str(kwargs["path"]))
        or _fake_field(kwargs["path"]))
    panel = preview.TimelapsePreviewPanel(threaded=False)
    movie = TimelapseMoviePanel()
    qtbot.addWidget(panel)
    qtbot.addWidget(movie)
    movie.set_max_fields(1)
    panel.attach_movie_panel(movie)
    assert panel.load_sequence(selected)
    _seed_primary(panel)
    panel._push_to_movie()
    primary = movie.movies()[0]

    movie.set_max_fields(2)

    assert calls == [str(first)]
    assert movie.movies()[0] is primary
    assert [item._title.text() for item in movie.movies()] == [
        selected.name, first.name]


def test_lowering_the_cap_interrupts_the_surplus_worker(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as preview
    from spacr.qt.widgets.timelapse_movie import TimelapseMoviePanel

    _field_folder(tmp_path, "field_a")
    selected = _field_folder(tmp_path, "field_b")
    started = threading.Event()
    interrupted = threading.Event()
    worker_threads = []

    def slow_field(*_args, cancelled=None, **_kwargs):
        worker_threads.append(threading.get_ident())
        started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if cancelled is not None and cancelled():
                interrupted.set()
                raise preview.MovieFieldCancelled("cancelled")
            time.sleep(0.005)
        raise AssertionError("the lower cap never interrupted this field")

    monkeypatch.setattr(preview, "build_movie_field", slow_field)
    panel = preview.TimelapsePreviewPanel(threaded=True)
    movie = TimelapseMoviePanel()
    qtbot.addWidget(panel)
    qtbot.addWidget(movie)
    movie.set_max_fields(2)
    panel.attach_movie_panel(movie)
    assert panel.load_sequence(selected)
    _seed_primary(panel)
    main_thread = threading.get_ident()
    panel._push_to_movie()
    qtbot.waitUntil(started.is_set, timeout=3000)

    movie.set_max_fields(1)
    qtbot.waitUntil(interrupted.is_set, timeout=3000)

    assert worker_threads and worker_threads[0] != main_thread
    assert panel._movie_pending_path is None
    assert list(panel._movie_fields) == [str(selected)]
    assert len(movie.movies()) == 1
    panel.shutdown()
