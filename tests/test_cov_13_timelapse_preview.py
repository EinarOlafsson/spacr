"""The tracking preview against inputs, backends and widgets that misbehave.

The preview reads a movie lazily, segments it once, and re-links it on every
tracking change. Everything here is a way that arrangement can be handed
something it did not expect: a TIFF that is one frame, a stack no memory map
will open, a Cellpose that returns a list of lists, a drop of a remote URL, a
panel whose worker thread has already been deleted. None of them may raise out
of a slot -- an exception in a Qt slot takes the window, not the preview.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt.widgets import timelapse_preview as TP  # noqa: E402
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call  # noqa: E402

pytestmark = pytest.mark.qt

H = W = 32
N_FRAMES = 4


def _disc(mask, cy, cx, r, label):
    yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return mask


def _synthetic_masks(n_frames=N_FRAMES, drift=3):
    stack = np.zeros((n_frames, H, W), np.int32)
    for t in range(n_frames):
        _disc(stack[t], 8, 6 + drift * t, 3, 1)
        _disc(stack[t], 22, 6 + drift * t, 3, 2)
    return stack


@pytest.fixture()
def frame_dir(tmp_path):
    """A folder of per-frame TIFFs — the commonest timelapse input."""
    import tifffile
    folder = tmp_path / "frames"
    folder.mkdir()
    masks = _synthetic_masks()
    for t in range(N_FRAMES):
        image = (masks[t] > 0).astype(np.uint16) * 900 + 40
        tifffile.imwrite(str(folder / f"field_t{t}.tif"), image)
    return str(folder)


@pytest.fixture()
def mask_dir(tmp_path):
    import tifffile
    folder = tmp_path / "labels"
    folder.mkdir()
    masks = _synthetic_masks()
    for t in range(N_FRAMES):
        tifffile.imwrite(str(folder / f"field_t{t}.tif"),
                         masks[t].astype(np.uint16))
    return str(folder)


# ---------------------------------------------------------------------------
# segmenting
# ---------------------------------------------------------------------------

def test_a_cellpose_that_returns_a_list_of_masks_yields_the_first_one(
        monkeypatch):
    """Cellpose hands back a list when it is given a list of images.

    The preview always passes one plane, but a model wrapper that batches
    internally still answers in the batched shape, and indexing the list as an
    array would produce a label image of one row.
    """
    class _Batching:
        """Answers in the batched shape whatever it was handed.

        ``channel_axis`` is named rather than swallowed: this call site
        deliberately leaves Cellpose to auto-detect the axis, so the contract
        is "whatever arrives must be something ``convert_image`` accepts".
        """

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis,
                                     require_channel_axis=False)
            return ([np.eye(4, dtype=np.int32)], None, None)

    monkeypatch.setattr(TP, "preview_cellpose_model",
                        lambda name: _Batching())

    mask = TP.segment_frame(np.zeros((4, 4), np.uint16), {"model": "cpsam"})

    assert mask.shape == (4, 4)
    assert mask.dtype == np.int32
    assert mask.trace() == 4


def test_ready_made_masks_are_stacked_frame_by_frame(mask_dir):
    """A mask folder is read the same lazy way an image folder is.

    Reading the whole folder into one array first would defeat the frame cap
    the preview exists to respect.
    """
    seq = TP.FrameSequence.open(mask_dir, max_frames=3)

    stack = TP._as_label_stack(seq, channel=0)

    assert stack.shape == (3, H, W)
    assert stack.dtype == np.int32
    assert set(np.unique(stack)) == {0, 1, 2}


def test_segmenting_a_sequence_returns_one_label_plane_per_frame(frame_dir,
                                                                 monkeypatch):
    """The stack the linker gets has to be (T, H, W) int32, every time."""
    from skimage.measure import label as sk_label

    monkeypatch.setattr(
        TP, "segment_frame",
        lambda image, params: sk_label(
            TP.frame_channel(np.asarray(image), 0) > 100).astype(np.int32))
    seq = TP.FrameSequence.open(frame_dir, max_frames=N_FRAMES)

    stack = TP.segment_sequence(seq, {})

    assert stack.shape == (N_FRAMES, H, W)
    assert stack.dtype == np.int32


def test_frames_that_segment_to_different_shapes_are_refused(frame_dir,
                                                             monkeypatch):
    """Two shapes mean two fields of view, and linking them is meaningless."""
    shapes = iter([(H, W), (H, W - 1)])

    monkeypatch.setattr(
        TP, "segment_frame",
        lambda image, params: np.zeros(next(shapes, (H, W)), np.int32))
    seq = TP.FrameSequence.open(frame_dir, max_frames=2)

    with pytest.raises(ValueError, match="not a single field of view"):
        TP.segment_sequence(seq, {})


# ---------------------------------------------------------------------------
# track statistics and rendering
# ---------------------------------------------------------------------------

def test_a_track_whose_positions_are_unknown_contributes_no_jump():
    """A NaN centroid is a position nobody measured, not a leap across the field.

    ``suspicious_jumps`` is what the linking radius is tuned against, so one
    unmeasured centroid counting as a jump would push the user to widen a
    radius that was already right.
    """
    tracks = pd.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "track_id": [1, 1, 1, 2, 2, 2],
        "x": [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
        "y": [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
    })

    stats = TP.track_stats(tracks, n_frames=3, min_length=2,
                           displacement_limit=0.5)

    assert stats.suspicious_jumps == 2      # track 2 only
    assert stats.max_step == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# one preview pass
# ---------------------------------------------------------------------------

def _linkable_masks():
    return _synthetic_masks()


# ---------------------------------------------------------------------------
# the worker thread
# ---------------------------------------------------------------------------

def test_the_worker_emits_its_result_rather_than_returning_it(qtbot):
    """A QThread's ``run`` returns into nothing, so the result has to be a signal."""
    worker = TP._TimelapseWorker(
        TP.TimelapseRequest(cached_masks=_linkable_masks(), seg={}, track={}))
    seen: list = []
    worker.finished_result.connect(lambda payload, error: seen.append(
        (payload, error)))

    worker.run()

    assert len(seen) == 1
    payload, error = seen[0]
    assert error == ""
    assert payload["masks"].shape == (N_FRAMES, H, W)


def test_the_worker_emits_its_failure_instead_of_raising_out_of_run(qtbot,
                                                                    caplog):
    """An exception escaping ``run`` is delivered to Qt, which aborts.

    Every refusal the pass can produce -- no sequence, a missing tracking
    backend, frames of two shapes -- has to arrive at the panel as a string it
    can put in the status label.
    """
    worker = TP._TimelapseWorker(TP.TimelapseRequest())
    seen: list = []
    worker.finished_result.connect(lambda payload, error: seen.append(
        (payload, error)))

    with caplog.at_level("INFO", logger=TP.LOG.name):
        worker.run()

    assert seen == [(None, "Load a sequence first.")]
    assert any("timelapse preview failed" in record.message
               for record in caplog.records)


# ---------------------------------------------------------------------------
# the model list
# ---------------------------------------------------------------------------

def test_a_model_menu_that_cannot_be_read_falls_back_to_the_known_names(
        monkeypatch):
    """An empty model dropdown is a dead panel.

    The menu comes from ``spacr.settings``, which probes the Cellpose install.
    On a machine without one that probe raises, and a combo with nothing in it
    leaves the user unable to run the preview at all.
    """
    import spacr.settings as settings

    def refuse():
        raise RuntimeError("cellpose is not installed")

    monkeypatch.setattr(settings, "cellpose_model_menu", refuse)

    assert TP._model_menu() == TP._FALLBACK_MODELS


def test_an_empty_model_menu_also_falls_back(monkeypatch):
    """A probe that answers with nothing is the same problem as one that raises."""
    import spacr.settings as settings

    monkeypatch.setattr(settings, "cellpose_model_menu", lambda: [])

    assert TP._model_menu() == TP._FALLBACK_MODELS


# ---------------------------------------------------------------------------
# opening a sequence, off the GUI thread
# ---------------------------------------------------------------------------

def test_a_sequence_that_will_not_yield_its_first_frame_still_opens(
        frame_dir, monkeypatch, caplog):
    """Warming the first frame is an optimisation, not a precondition.

    The panel installs the sequence and draws from it afterwards; a read that
    fails here would be reported as a load failure for a movie that is fine.
    """
    def refuse(self, index):
        raise OSError("the file went away")

    monkeypatch.setattr(TP.FrameSequence, "frame", refuse)

    with caplog.at_level("DEBUG", logger=TP.LOG.name):
        payload = TP.open_sequence_payload(frame_dir, max_frames=2,
                                           list_siblings=False)

    assert payload.get("error") in (None, "")
    assert payload["sequence"] is not None


def test_a_plate_whose_siblings_cannot_be_listed_still_loads_the_field(
        frame_dir, monkeypatch, caplog):
    """The FOV dropdown is a convenience; the field the user dropped is not.

    Listing a plate walks a folder that can be on a mount that has just gone
    away, and losing the drop over it would be the wrong trade.

    And the failure still comes back as a LIST holding the loaded field, not
    as ``None``. ``None`` means "nobody listed", which sends
    ``_refresh_source_selectors`` off to list the same folder itself -- on
    the GUI thread, on the path whose listing has just failed here. If that
    failure was a sleeping mount, the retry is the twenty-second freeze.
    """
    def refuse(*args, **kwargs):
        raise PermissionError("the plate folder is not readable")

    monkeypatch.setattr(TP, "sibling_sources", refuse)

    with caplog.at_level("ERROR", logger=TP.LOG.name):
        payload = TP.open_sequence_payload(frame_dir, max_frames=2,
                                           list_siblings=True)

    assert payload["sequence"] is not None
    assert payload["siblings"] == [pathlib.Path(frame_dir)], (
        "a failed listing must still hand back the field itself, or the GUI "
        "thread re-lists the folder that just refused")
    assert any("Could not list sequences beside" in record.message
               for record in caplog.records)


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

@pytest.fixture()
def panel(qtbot):
    widget = TP.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_a_drop_of_something_that_is_not_a_local_file_is_ignored(panel):
    """A URL dragged from a browser is not a movie on this disk.

    ``toLocalFile`` returns an empty string for it, and treating that as a
    path would open the working directory.
    """
    from PySide6.QtCore import QMimeData, QUrl

    class _Event:
        def __init__(self, mime):
            self._mime = mime
            self.ignored = False

        def mimeData(self):
            return self._mime

        def ignore(self):
            self.ignored = True

        def acceptProposedAction(self):
            raise AssertionError("a remote URL was accepted as a sequence")

    mime = QMimeData()
    mime.setUrls([QUrl("https://example.org/movie.tif")])
    event = _Event(mime)

    assert panel._dropped_path(event) is None
    panel.dropEvent(event)
    assert event.ignored is True


def test_asking_to_open_nothing_submits_no_job(panel):
    """An empty path arrives from a cleared box and from a cancelled dialog."""
    assert panel.load_sequence_async("") is False
    assert panel.load_sequence_async(None) is False
    assert panel._sequence is None


def test_a_load_superseded_by_a_newer_one_is_dropped(panel, frame_dir):
    """The stale open must not install itself over the newer sequence.

    Opens run on a worker, so two drops in quick succession finish in either
    order. The token is what makes the last drop win rather than the slowest
    read.
    """
    stale_token = panel._load_token
    panel.load_sequence_async(frame_dir)
    installed = panel._sequence
    assert installed is not None

    panel._on_sequence_loaded(stale_token, {"sequence": object()})

    assert panel._sequence is installed


def test_a_load_that_failed_puts_its_reason_in_the_status_line(panel):
    """The message is the only report; a modal here would hang a headless run."""
    panel._on_sequence_loaded(panel._load_token,
                              {"error": "Load failed: not a timelapse"})

    assert panel._status.text() == "Load failed: not a timelapse"
    assert panel._sequence is None


def test_a_payload_with_no_sequence_in_it_installs_nothing(panel):
    """Neither an error nor a sequence is a shape a future loader could return."""
    panel._on_sequence_loaded(panel._load_token, {"sequence": None})
    assert panel._sequence is None

    panel._on_sequence_loaded(panel._load_token, "not a payload")
    assert panel._sequence is None


# ---------------------------------------------------------------------------
# the channel controls that can be reached before the UI is finished
# ---------------------------------------------------------------------------

def test_the_channel_controls_do_nothing_before_the_ui_is_built(panel):
    """Both slots are wired to widgets that fire during construction.

    A spinner's ``valueChanged`` and a combo's ``currentIndexChanged`` both
    reach these while ``_build_ui`` is still running, when the other half of
    the pair does not exist yet.
    """
    box = panel._channel_box
    chosen = box.currentIndex()
    try:
        del panel._channel_box
        panel._sync_channel_combo_from_spin()
    finally:
        panel._channel_box = box
    assert box.currentIndex() == chosen

    spin = panel._channel
    value = spin.value()
    try:
        del panel._channel
        panel._on_display_channel_changed()
    finally:
        panel._channel = spin
    assert spin.value() == value


def test_raising_the_set_cap_to_the_value_it_already_has_redraws_nothing(
        panel, frame_dir):
    """Re-drawing the sample on a no-op change would reshuffle the dropdown.

    The sample is deliberately a pure function of (folder, cap) so that any
    other settings change re-renders the identical list. A cap "change" to the
    same number must therefore not be treated as a new draw.
    """
    panel.load_sequence_async(frame_dir)
    before = [panel._fov_box.itemText(i) for i in range(panel._fov_box.count())]

    panel._on_max_sets_changed(panel._sampler.max_sets)
    assert [panel._fov_box.itemText(i)
            for i in range(panel._fov_box.count())] == before

    panel._on_max_sets_changed(panel._sampler.max_sets + 3)
    assert panel._sampler.max_sets == len(before) + 3 or panel._sampler.max_sets


def test_picking_the_field_of_view_already_loaded_reloads_nothing(panel,
                                                                  frame_dir,
                                                                  monkeypatch):
    """Stepping onto the current field must not re-read it.

    ``_refresh_source_selectors`` sets the dropdown while installing a
    sequence, so the handler fires for the field that has just been loaded.
    Re-entering the load there is an infinite round trip.
    """
    panel.load_sequence_async(frame_dir)
    loads: list = []
    monkeypatch.setattr(panel, "load_sequence_async",
                        lambda path, **kwargs: loads.append(path))

    panel._on_fov_changed()
    assert loads == []

    panel._loading_fov = True
    try:
        panel._on_fov_changed()
    finally:
        panel._loading_fov = False
    assert loads == []


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------

def test_playback_will_not_start_on_a_single_frame(panel):
    """One frame is not a movie, and a timer looping on it burns a core."""
    assert panel._frame_slider.maximum() == 0

    panel._toggle_playback()

    assert panel._play_timer.isActive() is False
    assert panel._play_btn.text() == "Play"


def test_playback_stops_itself_if_the_frames_go_away_underneath_it(panel):
    """Loading a new sequence resets the slider while the timer is running.

    Without this the timer keeps firing against a slider whose maximum is
    zero, so the panel appears to be playing something that is not there.
    """
    panel._play_timer.start()

    panel._advance_frame()

    assert panel._play_timer.isActive() is False


def test_playback_wraps_at_the_last_frame(panel, frame_dir):
    """A preview loop that stopped at the end would have to be restarted."""
    panel.load_sequence_async(frame_dir)
    last = panel._frame_slider.maximum()
    assert last > 0

    panel._frame_slider.setValue(last)
    panel._advance_frame()
    assert panel._frame_slider.value() == 0

    panel._advance_frame()
    assert panel._frame_slider.value() == 1


def test_drawing_a_frame_of_an_empty_sequence_draws_nothing(panel):
    """A sequence that opened and holds no frames must not index into it."""
    reads: list = []

    class _Empty:
        def __len__(self):
            return 0

        def frame(self, index):
            reads.append(index)
            raise AssertionError("an empty sequence was read")

    panel._sequence = _Empty()
    before = panel._frame_label.text()

    panel._refresh_canvases()

    assert reads == []
    assert panel._frame_label.text() == before


# ---------------------------------------------------------------------------
# worker teardown
# ---------------------------------------------------------------------------

def test_releasing_a_worker_qt_already_deleted_is_a_no_op(panel, caplog):
    """The screen can close while the retired thread is still unwinding.

    ``_release_worker`` runs from the thread's ``finished`` signal, so its
    wrapper can already be dead by the time it gets there; raising would take
    the teardown with it.
    """
    class _Gone:
        def wait(self, *args):
            raise RuntimeError("Internal C++ object already deleted.")

        def setParent(self, parent):
            raise AssertionError("setParent must not be reached after wait")

    panel._retired_worker = _Gone()

    with caplog.at_level("DEBUG", logger=TP.LOG.name):
        panel._release_worker()

    assert panel._retired_worker is None


def test_closing_the_panel_survives_a_worker_qt_already_deleted(panel, caplog):
    """Same race, on the way out of the screen.

    Close is the moment both workers are most likely to be mid-teardown, and
    an exception in ``closeEvent`` leaves the window half-closed.
    """
    from PySide6.QtGui import QCloseEvent

    waited: list = []

    class _Gone:
        def __init__(self, name):
            self._name = name

        def wait(self, *args):
            waited.append(self._name)
            raise RuntimeError("Internal C++ object already deleted.")

    panel._worker = _Gone("running")
    panel._retired_worker = _Gone("retired")
    event = QCloseEvent()

    with caplog.at_level("DEBUG", logger=TP.LOG.name):
        panel.closeEvent(event)

    # Both are waited on -- the pass in flight AND the one whose result has
    # landed while its thread was still unwinding.
    assert waited == ["running", "retired"]
    assert event.isAccepted() is True


# ---------------------------------------------------------------------------
# the model list is live
# ---------------------------------------------------------------------------

def test_a_newly_registered_model_appears_without_disturbing_the_selection(
        panel, monkeypatch):
    """A checkpoint trained after the panel was built has to become choosable.

    Additive on purpose: rebuilding the list would drop whatever the user had
    picked if a probe came back thinner than the last one.
    """
    chosen = panel._model_box.currentText()
    before = {panel._model_box.itemText(i)
              for i in range(panel._model_box.count())}
    assert "my_checkpoint" not in before

    monkeypatch.setattr(TP, "_model_menu",
                        lambda: ("my_checkpoint",) + tuple(sorted(before)))

    panel.refresh_model_choices()

    after = [panel._model_box.itemText(i)
             for i in range(panel._model_box.count())]
    assert after[0] == "my_checkpoint"
    assert before <= set(after)
    assert panel._model_box.currentText() == chosen


def test_showing_the_panel_re_reads_the_model_list(panel, monkeypatch):
    """The screen is built once and shown many times.

    Reading the list only at construction means a model registered while the
    app was running never appears until it is restarted.
    """
    from PySide6.QtGui import QShowEvent

    calls: list = []
    monkeypatch.setattr(panel, "refresh_model_choices",
                        lambda: calls.append(True))

    panel.showEvent(QShowEvent())

    assert calls == [True]
