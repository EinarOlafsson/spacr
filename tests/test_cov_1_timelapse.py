"""btrack tracking, driven end to end against a stand-in for the library.

btrack is an optional dependency and is not installed here, so the whole of
:func:`spacr.timelapse._btrack_track_cells` past its import guard has never
run. The stand-in below is not a mock of spaCR's own logic: it segments the
real masks with ``regionprops`` and links objects by label, so every number
that reaches the track table -- the centroids, the merge back onto the
original labels, the relabelled stack -- is computed by the product from real
image data. What is faked is only btrack's API surface and its native solver.

The cases are the ones a real acquisition produces: a batch where nothing was
segmented, a name the well parser cannot read, a solver that stalls, and an
older btrack whose ``track()`` does not take ``tracking_updates``.

``tests/test_cov_timelapse_btrack.py`` asserts the same behaviours against the
REAL library. It is marked ``network`` and needs ``spacr[btrack]`` installed,
so on a machine without the optional backend it skips in full and this
function goes untested there -- which is every machine that does not opt in.
These run everywhere.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from skimage.measure import regionprops

import spacr.timelapse as TL


# ---------------------------------------------------------------------------
# The stand-in
# ---------------------------------------------------------------------------

class _Object:
    """One segmented object at one timepoint, as btrack reports them."""

    def __init__(self, t, x, y, label):
        self.t, self.x, self.y, self.z, self.label = t, x, y, 0.0, label


class _Track:
    """One linked trajectory: the arrays btrack's Tracklet exposes."""

    def __init__(self, track_id, objects):
        self.ID = track_id
        self.t = [o.t for o in objects]
        self.x = [o.x for o in objects]
        self.y = [o.y for o in objects]
        self.z = [o.z for o in objects]


@pytest.fixture
def fake_btrack(monkeypatch):
    """Install a btrack whose tracker links objects by label.

    Returns the recorder: ``calls`` holds what spaCR asked the library to do,
    and the knobs make the library behave like an older or a stalling one.
    """
    state = {
        "calls": [],
        "trackers": [],
        "track_type_error": False,
        "optimize_error": None,
        "config": "/fake/cell_config.json",
    }

    def _segmentation_to_objects(masks, properties=(), num_workers=1):
        state["calls"].append(("segment", tuple(properties), num_workers))
        objects = []
        for t, frame in enumerate(np.asarray(masks)):
            for region in regionprops(frame):
                y, x = region.centroid
                objects.append(_Object(t=t, x=x, y=y, label=region.label))
        return objects

    class _Tracker:
        """``btrack.BayesianTracker`` as spaCR drives it."""

        def __init__(self):
            self.appended = []
            self.tracks = []
            self.update_method = None
            self.max_search_radius = None
            self.features = None
            self.volume = None
            self.tracking_updates = None
            self.optimized = []
            state["trackers"].append(self)

        def __enter__(self):
            state["calls"].append(("enter", None, None))
            return self

        def __exit__(self, *exc):
            state["calls"].append(("exit", None, None))
            return False

        def configure(self, config):
            state["calls"].append(("configure", config, None))

        def append(self, objects):
            self.appended.extend(objects)

        def track(self, tracking_updates=None, step_size=None):
            if tracking_updates is not None and state["track_type_error"]:
                raise TypeError("track() got an unexpected keyword argument "
                                "'tracking_updates'")
            state["calls"].append(("track", tracking_updates, step_size))
            by_label = {}
            for obj in self.appended:
                by_label.setdefault(obj.label, []).append(obj)
            self.tracks = [_Track(label, objects)
                           for label, objects in sorted(by_label.items())]

        def optimize(self, backend=None, options=None):
            self.optimized.append({"backend": backend, "options": options})
            state["calls"].append(("optimize", backend, options))
            if state["optimize_error"] is not None:
                raise state["optimize_error"]

    btrack = types.ModuleType("btrack")
    utils = types.ModuleType("btrack.utils")
    datasets = types.ModuleType("btrack.datasets")
    constants = types.ModuleType("btrack.constants")
    utils.segmentation_to_objects = _segmentation_to_objects
    datasets.cell_config = lambda: state["config"]
    constants.BayesianUpdates = types.SimpleNamespace(APPROXIMATE="approximate",
                                                      EXACT="exact")
    btrack.utils = utils
    btrack.datasets = datasets
    btrack.constants = constants
    btrack.BayesianTracker = _Tracker
    for name, module in (("btrack", btrack), ("btrack.utils", utils),
                         ("btrack.datasets", datasets),
                         ("btrack.constants", constants)):
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture
def drawn(monkeypatch):
    """Record the calls to the timelapse overlay writer instead of drawing."""
    import spacr.plot as PL
    calls = []
    monkeypatch.setattr(PL, "_visualize_and_save_timelapse_stack_with_tracks",
                        lambda *args, **kwargs: calls.append(args))
    return calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def two_blobs(n_frames=3, size=32, drift=0):
    """``(T, Y, X)`` masks holding two labelled blobs per frame."""
    stack = np.zeros((n_frames, size, size), dtype=np.uint16)
    for t in range(n_frames):
        shift = t * drift
        stack[t, 4 + shift:9 + shift, 4:9] = 1
        stack[t, 20:25, 20 + shift:25 + shift] = 2
    return stack


def _run(src, masks, name="plate1_A01_1", **over):
    kwargs = dict(src=str(src), name=name, batch_filenames=[], mode="btrack",
                  object_type="cell", plot=False, save=False,
                  timelapse_remove_transient=False, masks_3D=masks,
                  radius=100, n_jobs=1)
    kwargs.update(over)
    return TL._btrack_track_cells(**kwargs)


def _tracks_csv(tmp_path, object_type="cell", name="plate1_A01_1"):
    return tmp_path / "tracks" / f"btrack_tracks_{object_type}_{name}.csv"


# ---------------------------------------------------------------------------
# What is not a stack of frames
# ---------------------------------------------------------------------------

def test_an_empty_batch_of_frames_is_refused(tmp_path, fake_btrack):
    """A batch with no frames cannot be tracked, and says so."""
    with pytest.raises(ValueError, match="nothing to track"):
        _run(tmp_path / "stack", [])


def test_frames_of_different_shapes_are_refused(tmp_path, fake_btrack):
    """Frames of two sizes cannot be one acquisition.

    Stacking them would either raise deep inside numpy or, worse, broadcast
    into an array that is not the images that were segmented.
    """
    masks = [np.zeros((32, 32), np.uint16), np.zeros((16, 16), np.uint16)]
    with pytest.raises(ValueError, match="same shape"):
        _run(tmp_path / "stack", masks)


def test_a_single_frame_is_not_a_timelapse(tmp_path, fake_btrack):
    """A 2-D mask has no time axis; the message names the shape it got."""
    with pytest.raises(ValueError, match=r"must be 3D"):
        _run(tmp_path / "stack", np.zeros((32, 32), np.uint16))


# ---------------------------------------------------------------------------
# Nothing segmented
# ---------------------------------------------------------------------------

def test_a_batch_with_nothing_segmented_never_loads_the_native_tracker(
        tmp_path, fake_btrack, drawn):
    """No objects has a complete answer without btrack's native solver.

    Constructing ``BayesianTracker`` loads ``libtracker``, which can fail on a
    machine whose libstdc++ is older than the wheel. An empty batch is
    deterministic without it: an empty track table and an all-zero stack.
    """
    src = tmp_path / "stack"
    masks = np.zeros((3, 16, 16), dtype=np.uint16)

    out = _run(src, masks, plot=True)

    assert fake_btrack["trackers"] == [], "no tracker was constructed"
    assert [call[0] for call in fake_btrack["calls"]] == ["segment"]
    assert np.asarray(out).shape == (3, 16, 16)
    assert not np.asarray(out).any()

    table = pd.read_csv(_tracks_csv(tmp_path))
    assert len(table) == 0
    assert list(table.columns) == [
        "track_id", "frame", "x", "y", "original_label", "file_name",
        "plateID", "rowID", "columnID", "fieldID", "prcf", "wellID"]
    assert len(drawn) == 1, "the empty batch is still drawn when asked"


# ---------------------------------------------------------------------------
# The tracked path
# ---------------------------------------------------------------------------

def test_a_tracked_batch_relabels_its_masks_with_the_track_ids(tmp_path,
                                                               fake_btrack,
                                                               drawn):
    """Every object carries its track id, and the table names its well.

    The relabelled stack is what the rest of the pipeline measures, so a
    label that does not follow the same cell through the frames is a
    measurement of two different cells averaged together.
    """
    src = tmp_path / "stack"
    masks = two_blobs(n_frames=3, drift=1)

    out = _run(src, masks, plot=True, save=True,
               batch_filenames=[f"plate1_A01_1_t{t}.npy" for t in range(3)])

    out = np.asarray(out)
    assert out.shape == masks.shape
    # Two objects per frame, and the same two ids in every frame.
    per_frame = [sorted(set(np.unique(frame).tolist()) - {0}) for frame in out]
    assert per_frame == [[1, 2]] * 3

    table = pd.read_csv(_tracks_csv(tmp_path))
    assert len(table) == 6
    assert sorted(table["track_id"].unique()) == [1, 2]
    assert set(table["frame"]) == {0, 1, 2}
    assert set(table["original_label"]) == {1, 2}
    assert set(table["plateID"]) == {"plate1"}
    assert set(table["rowID"]) == {"r1"}
    assert set(table["columnID"]) == {"c1"}
    assert set(table["wellID"]) == {"A01"}, (
        "wellID is composed from the parsed row and column, not re-split")
    assert len(drawn) == 1


def test_the_tracker_is_configured_the_way_the_pipeline_needs_it(tmp_path,
                                                                 fake_btrack):
    """The volume, the search radius and the update method all reach btrack.

    A tracker whose volume does not match the image links objects across an
    edge it should not, and the approximate update is what keeps a full plate
    tractable.
    """
    src = tmp_path / "stack"
    _run(src, two_blobs(size=48), radius=None)

    tracker, = fake_btrack["trackers"]
    assert tracker.volume == ((0, 48), (0, 48))
    assert tracker.update_method == "approximate"
    assert tracker.max_search_radius == 48 // 20, (
        "radius=None asks for one twentieth of the image width")
    assert tracker.features == ["area", "major_axis_length",
                                "minor_axis_length", "orientation", "solidity"]
    assert ("configure", "/fake/cell_config.json", None) in fake_btrack["calls"]
    assert ("track", ["motion", "visual"], None) in fake_btrack["calls"]


def test_an_older_btrack_is_driven_through_its_own_api(tmp_path, fake_btrack):
    """``track(tracking_updates=...)`` is new; the fallback still tracks.

    An older btrack takes the updates as an attribute and a step size. Without
    the fallback the whole batch would die on a TypeError from the library.
    """
    fake_btrack["track_type_error"] = True
    src = tmp_path / "stack"

    out = _run(src, two_blobs())

    tracker, = fake_btrack["trackers"]
    assert tracker.tracking_updates == ["MOTION", "VISUAL"]
    assert ("track", None, 100) in fake_btrack["calls"]
    assert sorted(set(np.unique(np.asarray(out)).tolist()) - {0}) == [1, 2]


# ---------------------------------------------------------------------------
# The optimiser
# ---------------------------------------------------------------------------

def test_the_glpk_limits_the_caller_set_are_passed_through(tmp_path,
                                                           fake_btrack):
    """A time limit in seconds becomes GLPK's tm_lim in milliseconds."""
    _run(tmp_path / "stack", two_blobs(),
         optimizer_time_limit_s=1.5, optimizer_mip_gap=0.05)

    tracker, = fake_btrack["trackers"]
    assert tracker.optimized == [{
        "backend": "glpk",
        "options": {"options": {"tm_lim": 1500, "mip_gap": 0.05}}}]


def test_no_limits_means_glpk_s_own_defaults(tmp_path, fake_btrack):
    """With nothing to pass, the optimiser is called with no options at all.

    An empty options dict is not the same request as no options: it is what a
    caller sends when it wants the defaults, and building one would put spaCR
    between the user and GLPK's own tuning.
    """
    _run(tmp_path / "stack", two_blobs(),
         optimizer_time_limit_s=None, optimizer_mip_gap=None)

    tracker, = fake_btrack["trackers"]
    assert tracker.optimized == [{"backend": "glpk", "options": None}]


def test_a_stalling_optimiser_leaves_the_tracks_it_already_had(tmp_path,
                                                               fake_btrack,
                                                               caplog):
    """GLPK failing is a warning and the pre-optimisation tracks, not a crash.

    The tracks exist before the global optimisation; losing the batch because
    the solver misbehaved would throw away work that is already good enough to
    measure.
    """
    fake_btrack["optimize_error"] = RuntimeError("glpk gave up")

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(tmp_path / "stack", two_blobs())

    assert sorted(set(np.unique(np.asarray(out)).tolist()) - {0}) == [1, 2]
    assert any("optimisation failed" in record.message
               for record in caplog.records)


def test_a_batch_too_big_to_optimise_says_so_and_skips_it(tmp_path,
                                                          fake_btrack,
                                                          caplog):
    """Past the object ceiling the global optimisation is not attempted.

    Left to run, GLPK on a plate-sized problem is the difference between a run
    that finishes overnight and one that does not finish.
    """
    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        _run(tmp_path / "stack", two_blobs(),
             max_objects_for_optimization=2)

    tracker, = fake_btrack["trackers"]
    assert tracker.optimized == []
    assert any("Skipping btrack global optimisation" in record.message
               for record in caplog.records)


def test_optimisation_can_be_turned_off_outright(tmp_path, fake_btrack):
    """``run_optimization=False`` is honoured however small the batch is."""
    _run(tmp_path / "stack", two_blobs(), run_optimization=False)

    tracker, = fake_btrack["trackers"]
    assert tracker.optimized == []
    assert tracker.tracks, "tracking itself still happened"


# ---------------------------------------------------------------------------
# Tracks that do not survive
# ---------------------------------------------------------------------------

def test_removing_transients_can_empty_the_table_without_emptying_the_batch(
        tmp_path, fake_btrack, caplog):
    """Every track shorter than the batch goes; the stack comes back unlabelled.

    ``timelapse_remove_transient`` keeps only objects seen in every frame. A
    batch where nothing persists then has no tracks at all, and the answer is
    an empty table and a zeroed stack -- not a KeyError on a frame with no
    columns.
    """
    # One blob in the first frame only: it cannot survive a 3-frame minimum.
    masks = np.zeros((3, 32, 32), dtype=np.uint16)
    masks[0, 4:9, 4:9] = 1

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(tmp_path / "stack", masks, timelapse_remove_transient=True)

    assert not np.asarray(out).any()
    table = pd.read_csv(_tracks_csv(tmp_path))
    assert len(table) == 0
    assert {"track_id", "wellID", "original_label"} <= set(table.columns)
    assert any("no usable tracks" in record.message
               for record in caplog.records)


def test_a_name_the_well_parser_chokes_on_still_writes_the_tracks(
        tmp_path, fake_btrack, monkeypatch, caplog):
    """An unreadable batch name costs the well columns, not the batch.

    The track ids and the relabelled masks do not depend on the name at all,
    so a name the parser cannot take apart leaves the identity columns unset
    and everything else intact.
    """
    import spacr.utils as U

    def _boom(file_name, timelapse=False):
        raise IndexError("list index out of range")

    monkeypatch.setattr(U, "_map_wells", _boom)

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(tmp_path / "stack", two_blobs(), name="oddname")

    assert sorted(set(np.unique(np.asarray(out)).tolist()) - {0}) == [1, 2]
    table = pd.read_csv(_tracks_csv(tmp_path, name="oddname"))
    assert len(table) == 6
    assert "wellID" not in table.columns
    assert any("Failed to parse plate, well, field" in record.message
               for record in caplog.records)


def test_a_list_of_frames_is_stacked_before_tracking(tmp_path, fake_btrack):
    """Cellpose hands its masks over as a list of 2-D frames.

    That is the shape the mask generators actually call this with, so the list
    has to become a ``(T, Y, X)`` stack here rather than at the first numpy
    operation that happens to broadcast.
    """
    stack = two_blobs(n_frames=3)
    as_list = [frame for frame in stack]

    out = _run(tmp_path / "stack", as_list)

    assert np.asarray(out).shape == stack.shape
    table = pd.read_csv(_tracks_csv(tmp_path))
    assert sorted(table["track_id"].unique()) == [1, 2]


# ---------------------------------------------------------------------------
# Peak ids
# ---------------------------------------------------------------------------

def test_a_peak_id_that_is_not_an_object_key_is_named_in_the_refusal():
    """An id with no object in it is refused, and the refusal quotes it.

    The legacy spelling -- a prcf plus a bare object number -- is still read.
    Anything else is a key this module cannot honestly take apart, and a
    positional split would invent a plate and a well for it instead.
    """
    from spacr import schema

    ids = pd.DataFrame({"ID": ["plate1_r1_c1_f1_o7", "plate1_r1_c1_f1_7"]})
    out = TL._explode_peak_ids(ids, "test")
    assert list(out["object_number"]) == ["o7", "o7"], (
        "the legacy bare-object spelling reads the same as the current one")
    assert list(out["plateID"]) == ["plate1", "plate1"]

    with pytest.raises(schema.KeyParseError) as excinfo:
        TL._explode_peak_ids(pd.DataFrame({"ID": ["junk_5"]}), "test")

    assert "junk_5" in str(excinfo.value)


# ---------------------------------------------------------------------------
# numpy 1
# ---------------------------------------------------------------------------

def test_the_module_still_imports_where_trapezoid_is_called_trapz(monkeypatch):
    """On numpy 1 the integrator is ``trapz``; the module must still load.

    This is the one module in spaCR whose import fails outright when the
    numpy-2 rename is not handled -- there is no lazy path to fall back on,
    so every timelapse feature disappears with it.
    """
    import importlib.util

    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", lambda y, x=None, dx=1.0: 0.0,
                        raising=False)

    # Named inside the package so its relative imports still resolve.
    name = "spacr._cov1_timelapse_numpy1"
    spec = importlib.util.spec_from_file_location(name, TL.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert module.trapz is np.trapz
        assert callable(module._btrack_track_cells)
    finally:
        sys.modules.pop(name, None)
