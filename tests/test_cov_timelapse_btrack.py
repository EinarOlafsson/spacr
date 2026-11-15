"""CPU-only coverage for :func:`spacr.timelapse._btrack_track_cells`.

The happy paths run the *real* btrack tracker on tiny synthetic label
stacks (4 frames of 64x64, four blobs drifting one pixel per frame). btrack
fetches its example config registry on first use, so this module belongs to
the network suite and skips cleanly when that endpoint is unavailable.

Everything that cannot be provoked with real data — the older-btrack
``TypeError`` fallback, a GLPK optimiser that blows up, the object-count
guard, the well-metadata parse failure — is driven through a fake
``btrack.BayesianTracker`` that records exactly how the product code
configured it, so the assertions are about the tracker contract rather than
about "did it run".
"""
from __future__ import annotations

import functools
import inspect
import logging
import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", autouse=True)
def _require_btrack_registry():
    """The btrack dataset module performs this download during first import."""
    pytest.importorskip(
        "btrack",
        reason="optional btrack backend is not installed (spacr[btrack])",
    )
    from tests.resource_capabilities import endpoint_available

    registry = (
        "https://raw.githubusercontent.com/"
        "lowe-lab-ucl/btrack-examples/main/registry.txt"
    )
    if not endpoint_available(registry):
        pytest.skip("btrack example registry is unreachable")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

FEATURES = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "solidity",
]

CENTRES = [(16, 16), (16, 44), (44, 16), (44, 44)]


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let a figure survive a test (rule: MPLBACKEND=Agg + close all)."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _moving_masks(n_frames=4, size=64, radius=6, centres=CENTRES,
                  transient_label=False):
    """Label stack (T, Y, X) with one blob per centre drifting +1 px per frame.

    ``transient_label`` adds an extra object that only exists in frame 0, which
    is what the transient-track filter is supposed to throw away.
    """
    yy, xx = np.mgrid[:size, :size]
    stack = np.zeros((n_frames, size, size), dtype=np.int32)
    for t in range(n_frames):
        for lbl, (cy, cx) in enumerate(centres, start=1):
            stack[t][(yy - (cy + t)) ** 2 + (xx - (cx + t)) ** 2 <= radius ** 2] = lbl
    if transient_label:
        extra = len(centres) + 1
        stack[0][(yy - 30) ** 2 + (xx - 30) ** 2 <= 3 ** 2] = extra
    return stack


class _FakeTrack:
    """Minimal stand-in for :class:`btrack.btypes.Tracklet`."""

    def __init__(self, ID, t, x, y, z):  # noqa: N803 - mirrors btrack's attr name
        self.ID = ID
        self.t = list(t)
        self.x = list(x)
        self.y = list(y)
        self.z = list(z)


def _tracks_from_masks(masks, factor=10, only_first_frame_for=()):
    """Build fake tracks whose (frame, x, y) exactly match the mask centroids.

    Track ``ID`` is ``factor * original_label`` so a relabelled mask proves
    which track table the product code actually used.
    """
    from spacr.timelapse import _prepare_for_tracking

    objects_df = _prepare_for_tracking(np.asarray(masks))
    tracks = []
    for label, grp in objects_df.groupby("original_label"):
        grp = grp.sort_values("frame")
        if label in only_first_frame_for:
            grp = grp.iloc[:1]
        tracks.append(
            _FakeTrack(
                ID=int(label) * factor,
                t=[int(v) for v in grp["frame"]],
                x=[float(v) for v in grp["x"]],
                y=[float(v) for v in grp["y"]],
                z=[0.0] * len(grp),
            )
        )
    return tracks


class _FakeTracker:
    """Records everything ``_btrack_track_cells`` does to a BayesianTracker."""

    def __init__(self, rec):
        self._rec = rec
        rec["instance"] = self
        self._tracks = []
        self.update_method = None
        self.max_search_radius = None
        self.features = None
        self.tracking_updates = None
        self.volume = None

    # context-manager protocol used by the product code
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._rec["closed"] = True
        return False

    def configure(self, config_file):
        self._rec["config"] = config_file

    def append(self, objects):
        self._rec["n_appended"] = len(objects)

    def track(self, **kwargs):
        self._rec["track_calls"].append(dict(kwargs))
        self._tracks = list(self._rec["tracks"])
        if self._rec["track_typeerror"] and "tracking_updates" in kwargs:
            raise TypeError(
                "track() got an unexpected keyword argument 'tracking_updates'"
            )

    @property
    def tracks(self):
        return self._tracks

    def optimize(self, **kwargs):
        self._rec["optimize_calls"].append(dict(kwargs))
        if self._rec["optimize_error"] is not None:
            raise self._rec["optimize_error"]
        if self._rec["optimized_tracks"] is not None:
            self._tracks = list(self._rec["optimized_tracks"])


def _install_fake_tracker(monkeypatch, tracks, *, optimized_tracks=None,
                          track_typeerror=False, optimize_error=None):
    import btrack

    rec = {
        "tracks": tracks,
        "optimized_tracks": optimized_tracks,
        "track_typeerror": track_typeerror,
        "optimize_error": optimize_error,
        "track_calls": [],
        "optimize_calls": [],
        "instance": None,
        "closed": False,
        "config": None,
        "n_appended": None,
    }
    monkeypatch.setattr(btrack, "BayesianTracker", lambda: _FakeTracker(rec))
    return rec


def _run(src_dir, masks, **overrides):
    """Call ``_btrack_track_cells`` with sane, fast defaults."""
    from spacr.timelapse import _btrack_track_cells

    n_frames = len(masks)
    kwargs = dict(
        src=str(src_dir / "masks"),
        name="plate1_A01_1",
        batch_filenames=[f"plate1_A01_1_t{i}.tif" for i in range(n_frames)],
        object_type="cell",
        plot=False,
        save=False,
        masks_3D=masks,
        mode="btrack",
        timelapse_remove_transient=False,
        radius=20,
        n_jobs=1,
        optimizer_time_limit_s=5,
        optimizer_mip_gap=0.01,
        run_optimization=True,
        max_objects_for_optimization=None,
    )
    kwargs.update(overrides)
    return _btrack_track_cells(**kwargs)


def _tracks_csv(src_dir, object_type="cell", name="plate1_A01_1"):
    return src_dir / "tracks" / f"btrack_tracks_{object_type}_{name}.csv"


@pytest.fixture
def src_dir(tmp_path):
    """A project-ish directory; the product writes tracks/ next to masks/."""
    (tmp_path / "masks").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# real-btrack happy paths
# ---------------------------------------------------------------------------

def _case_btrack_tracks_moving_cells_end_to_end(src_dir):
    """Four drifting blobs keep one label per object across all four frames."""
    observed = {}
    masks = _moving_masks()
    out = _run(src_dir, masks)

    observed['isinstance(out, list)'] = bool(isinstance(out, list))
    observed['len(out)'] = len(out)
    observed['all((f.shape == (64, 64) and f.dtype == masks.dtype for f in out))'] = bool(all((f.shape == (64, 64) and f.dtype == masks.dtype for f in out)))

    # every frame carries exactly the same four track ids
    label_sets = [set(np.unique(f)) - {0} for f in out]
    observed['all((len(s) == 4 for s in label_sets))'] = bool(all((len(s) == 4 for s in label_sets)))
    observed['label_sets[0] == label_sets[1] == label_sets[2] == label_sets[3]'] = bool(label_sets[0] == label_sets[1] == label_sets[2] == label_sets[3])

    # and each blob keeps its id while it drifts
    for cy, cx in CENTRES:
        ids = {int(out[t][cy + t, cx + t]) for t in range(4)}
        assert len(ids) == 1 and 0 not in ids

    csv = _tracks_csv(src_dir)
    observed['csv.exists()'] = bool(csv.exists())
    df = pd.read_csv(csv)
    observed['len(df)'] = len(df)
    observed["set(df['frame'])"] = set(df['frame'])
    observed["df['track_id'].nunique()"] = df['track_id'].nunique()
    observed["set(df['original_label'])"] = set(df['original_label'])
    return observed


_EXPECT_btrack_tracks_moving_cells_end_to_end = {
    'isinstance(out, list)': True,
    'len(out)': 4,
    'all((f.shape == (64, 64) and f.dtype == masks.dtype for f in out))': True,
    'all((len(s) == 4 for s in label_sets))': True,
    'label_sets[0] == label_sets[1] == label_sets[2] == label_sets[3]': True,
    'csv.exists()': True,
    'len(df)': 16,
    "set(df['frame'])": {0, 1, 2, 3},
    "df['track_id'].nunique()": 4,
    "set(df['original_label'])": {1, 2, 3, 4},
}


def _case_btrack_writes_well_metadata_parsed_from_name(src_dir):
    """The npz batch name is decomposed into plate/row/column/field/well."""
    observed = {}
    masks = _moving_masks(n_frames=3)
    _run(src_dir, masks)

    df = pd.read_csv(_tracks_csv(src_dir))
    observed["set(df['file_name'])"] = set(df['file_name'])
    observed["set(df['plateID'])"] = set(df['plateID'])
    observed["set(df['rowID'])"] = set(df['rowID'])
    observed["set(df['columnID'])"] = set(df['columnID'])
    observed["set(df['fieldID'])"] = set(df['fieldID'])
    observed["set(df['prcf'])"] = set(df['prcf'])
    observed["set(df['wellID'])"] = set(df['wellID'])
    return observed


_EXPECT_btrack_writes_well_metadata_parsed_from_name = {
    "set(df['file_name'])": {'plate1_A01_1'},
    "set(df['plateID'])": {'plate1'},
    "set(df['rowID'])": {'r1'},
    "set(df['columnID'])": {'c1'},
    "set(df['fieldID'])": {'f1'},
    "set(df['prcf'])": {'plate1_r1_c1_f1'},
    "set(df['wellID'])": {'A01'},
}


def _case_btrack_accepts_a_list_of_2d_masks(src_dir, tmp_path):
    """A list of 2-D frames is stacked and gives the same answer as an ndarray."""
    observed = {}
    masks = _moving_masks(n_frames=3)
    from_array = _run(src_dir, masks)

    other = tmp_path / "other"
    (other / "masks").mkdir(parents=True)
    from_list = _run(other, [masks[t] for t in range(masks.shape[0])])

    observed['len(from_list) == len(from_array) == 3'] = bool(len(from_list) == len(from_array) == 3)
    for a, b in zip(from_array, from_list):
        np.testing.assert_array_equal(a, b)
    return observed


_EXPECT_btrack_accepts_a_list_of_2d_masks = {
    'len(from_list) == len(from_array) == 3': True,
}


def _bad_masks(kind):
    return {
        "empty-list": [],
        "ragged-list": [np.zeros((8, 8), dtype=np.int32), np.zeros((8, 9), dtype=np.int32)],
        "2d-array": np.zeros((8, 8), np.int32),
        "4d-array": np.zeros((2, 3, 4, 5), np.int32),
    }[kind]


def _case_btrack_rejects_masks_that_are_not_a_uniform_time_stack(src_dir, kind, message):
    with pytest.raises(ValueError, match=message) as excinfo:
        _run(src_dir, _bad_masks(kind))
    return {"error": str(excinfo.value)}


_REJECT_MESSAGES = {
    "empty-list": "masks_3D is an empty list; nothing to track.",
    "ragged-list": "All masks must have the same shape; got shapes={(8, 8), (8, 9)}.",
    "2d-array": "masks_3D must be 3D (T, Y, X); got shape (8, 8)",
    "4d-array": "masks_3D must be 3D (T, Y, X); got shape (2, 3, 4, 5)",
}


# ---------------------------------------------------------------------------
# tracker configuration (fake BayesianTracker)
# ---------------------------------------------------------------------------

def _case_btrack_configures_tracker_from_arguments(src_dir, monkeypatch):
    observed = {}
    from btrack import datasets as btrack_datasets
    from btrack.constants import BayesianUpdates

    masks = _moving_masks()
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))
    out = _run(src_dir, masks, radius=37)

    tracker = rec["instance"]
    observed["rec['config'] == btrack_datasets.cell_config()"] = bool(rec['config'] == btrack_datasets.cell_config())
    observed['tracker.update_method == BayesianUpdates.APPROXIMATE'] = bool(tracker.update_method == BayesianUpdates.APPROXIMATE)
    observed['tracker.max_search_radius'] = tracker.max_search_radius
    observed['tracker.features'] = tracker.features
    observed['tracker.volume'] = tracker.volume
    observed["rec['n_appended']"] = rec['n_appended']
    observed["rec['track_calls']"] = rec['track_calls']
    observed["rec['closed'] is True"] = bool(rec['closed'] is True)

    # track ids came from the fake tracks: label L -> 10 * L
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_configures_tracker_from_arguments = {
    "rec['config'] == btrack_datasets.cell_config()": True,
    'tracker.update_method == BayesianUpdates.APPROXIMATE': True,
    'tracker.max_search_radius': 37,
    'tracker.features': FEATURES,
    'tracker.volume': ((0, 64), (0, 64)),
    "rec['n_appended']": 16,
    "rec['track_calls']": [{'tracking_updates': ['motion', 'visual']}],
    "rec['closed'] is True": True,
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
}


def _case_btrack_derives_radius_from_image_width(src_dir, monkeypatch, size,
                                                expected_radius):
    """radius=None -> width/20, floored at 1 for tiny images."""
    observed = {}
    centres = CENTRES if size == 64 else [(8, 8)]
    masks = _moving_masks(n_frames=2, size=size, radius=2, centres=centres)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, radius=None)

    observed["rec['instance'].max_search_radius == expected_radius"] = bool(rec['instance'].max_search_radius == expected_radius)
    observed["rec['instance'].volume == ((0, size), (0, size))"] = bool(rec['instance'].volume == ((0, size), (0, size)))
    return observed


_EXPECT_btrack_derives_radius_from_image_width = {
    "rec['instance'].max_search_radius == expected_radius": True,
    "rec['instance'].volume == ((0, size), (0, size))": True,
}


def _case_btrack_falls_back_to_step_size_track_on_typeerror(src_dir, monkeypatch):
    """Older btrack: track(tracking_updates=...) -> TypeError -> attribute + step_size."""
    observed = {}
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch, _tracks_from_masks(masks), track_typeerror=True
    )

    out = _run(src_dir, masks)

    observed["rec['track_calls']"] = rec['track_calls']
    observed["rec['instance'].tracking_updates"] = rec['instance'].tracking_updates
    # the fallback still produced usable tracks
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_falls_back_to_step_size_track_on_typeerror = {
    "rec['track_calls']": [{'tracking_updates': ['motion', 'visual']}, {'step_size': 100}],
    "rec['instance'].tracking_updates": ['MOTION', 'VISUAL'],
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
}


# ---------------------------------------------------------------------------
# global optimisation branches
# ---------------------------------------------------------------------------

def _case_btrack_passes_glpk_options_and_uses_optimised_tracks(src_dir, monkeypatch):
    observed = {}
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    out = _run(src_dir, masks, optimizer_time_limit_s=2.5, optimizer_mip_gap=0.05)

    observed["rec['optimize_calls']"] = rec['optimize_calls']
    # post-optimisation tracks are the ones written out
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_passes_glpk_options_and_uses_optimised_tracks = {
    "rec['optimize_calls']": [{'backend': 'glpk', 'options': {'options': {'tm_lim': 2500, 'mip_gap': 0.05}}}],
    'set(np.unique(out[0]))': {0, 100, 200, 300, 400},
}


def _case_btrack_optimises_with_default_options_when_none_given(src_dir, monkeypatch):
    observed = {}
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, optimizer_time_limit_s=None, optimizer_mip_gap=None)

    observed["rec['optimize_calls']"] = rec['optimize_calls']
    return observed


_EXPECT_btrack_optimises_with_default_options_when_none_given = {
    "rec['optimize_calls']": [{'backend': 'glpk'}],
}


def _case_btrack_ignores_non_positive_optimiser_settings(src_dir, monkeypatch,
                                                        time_limit, mip_gap):
    """Zero time-limit / MIP gap are treated as 'unset', not as literal zeros."""
    masks = _moving_masks(n_frames=2)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, optimizer_time_limit_s=time_limit, optimizer_mip_gap=mip_gap)

    return {"optimize_calls": rec["optimize_calls"]}


def _case_btrack_skips_optimisation_when_disabled(src_dir, monkeypatch):
    observed = {}
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    out = _run(src_dir, masks, run_optimization=False)

    observed["rec['optimize_calls']"] = rec['optimize_calls']
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_skips_optimisation_when_disabled = {
    "rec['optimize_calls']": [],
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
}


def _case_btrack_skips_optimisation_for_too_many_objects(src_dir, monkeypatch, caplog):
    observed = {}
    masks = _moving_masks(n_frames=3)  # 12 real objects
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks, max_objects_for_optimization=5)

    observed["rec['optimize_calls']"] = rec['optimize_calls']
    observed["any(('Skipping btrack global optimisation' in r.getMessage() for r in caplog.records))"] = bool(any(('Skipping btrack global optimisation' in r.getMessage() for r in caplog.records)))
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_skips_optimisation_for_too_many_objects = {
    "rec['optimize_calls']": [],
    "any(('Skipping btrack global optimisation' in r.getMessage() for r in caplog.records))": True,
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
}


def _case_btrack_falls_back_to_raw_tracks_when_optimiser_fails(src_dir, monkeypatch,
                                                              caplog):
    observed = {}
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
        optimize_error=RuntimeError("GLPK stalled"),
    )

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks)

    observed["len(rec['optimize_calls'])"] = len(rec['optimize_calls'])
    observed["any(('global optimisation failed' in r.getMessage() for r in caplog.records))"] = bool(any(('global optimisation failed' in r.getMessage() for r in caplog.records)))
    # pre-optimisation tracks survive the failure
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    observed['_tracks_csv(src_dir).exists()'] = bool(_tracks_csv(src_dir).exists())
    return observed


_EXPECT_btrack_falls_back_to_raw_tracks_when_optimiser_fails = {
    "len(rec['optimize_calls'])": 1,
    "any(('global optimisation failed' in r.getMessage() for r in caplog.records))": True,
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
    '_tracks_csv(src_dir).exists()': True,
}


# ---------------------------------------------------------------------------
# transient-track filtering
# ---------------------------------------------------------------------------

def _case_btrack_transient_filter_drops_single_frame_tracks(src_dir, monkeypatch,
                                                           remove_transient,
                                                           expect_transient):
    """A track that only exists in frame 0 survives only when filtering is off."""
    observed = {}
    masks = _moving_masks(n_frames=3, transient_label=True)
    tracks = _tracks_from_masks(masks, factor=10, only_first_frame_for=(5,))
    _install_fake_tracker(monkeypatch, tracks)

    out = _run(src_dir, masks, timelapse_remove_transient=remove_transient)

    frame0 = set(np.unique(out[0]))
    observed['{0, 10, 20, 30, 40} <= frame0'] = bool({0, 10, 20, 30, 40} <= frame0)
    observed['(50 in frame0) is expect_transient'] = bool((50 in frame0) is expect_transient)

    df = pd.read_csv(_tracks_csv(src_dir))
    observed["(50 in set(df['track_id'])) is expect_transient"] = bool((50 in set(df['track_id'])) is expect_transient)
    observed["set(df['frame'])"] = set(df['frame'])
    return observed


_EXPECT_btrack_transient_filter_drops_single_frame_tracks = {
    '{0, 10, 20, 30, 40} <= frame0': True,
    '(50 in frame0) is expect_transient': True,
    "(50 in set(df['track_id'])) is expect_transient": True,
    "set(df['frame'])": {0, 1, 2},
}


# ---------------------------------------------------------------------------
# well-metadata parsing failure
# ---------------------------------------------------------------------------

def _case_btrack_survives_unparsable_well_metadata(src_dir, monkeypatch, caplog):
    """An IndexError out of _map_wells is logged and the CSV is still written."""
    observed = {}
    import spacr.utils

    def _boom(file_name, timelapse=False):
        raise IndexError(f"cannot parse {file_name}")

    monkeypatch.setattr(spacr.utils, "_map_wells", _boom)

    masks = _moving_masks(n_frames=3)
    _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks)

    observed["any(('Failed to parse plate, well, field' in r.getMessage() for r in caplog.records))"] = bool(any(('Failed to parse plate, well, field' in r.getMessage() for r in caplog.records)))
    df = pd.read_csv(_tracks_csv(src_dir))
    observed["'plateID' not in df.columns"] = bool('plateID' not in df.columns)
    observed["'wellID' not in df.columns"] = bool('wellID' not in df.columns)
    observed["set(df['track_id'])"] = set(df['track_id'])
    observed['set(np.unique(out[0]))'] = set(np.unique(out[0]))
    return observed


_EXPECT_btrack_survives_unparsable_well_metadata = {
    "any(('Failed to parse plate, well, field' in r.getMessage() for r in caplog.records))": True,
    "'plateID' not in df.columns": True,
    "'wellID' not in df.columns": True,
    "set(df['track_id'])": {10, 20, 30, 40},
    'set(np.unique(out[0]))': {0, 10, 20, 30, 40},
}


# ---------------------------------------------------------------------------
# visualisation hand-off
# ---------------------------------------------------------------------------

def _case_btrack_hands_masks_and_tracks_to_the_visualiser(src_dir, monkeypatch,
                                                         plot, save, expect_call):
    observed = {}
    import spacr.plot

    calls = []

    def _fake_visualise(masks, tracks_df, save_, src_, name_, plot_, filenames,
                        object_type_, mode_):
        calls.append(dict(masks=masks, tracks_df=tracks_df, save=save_, src=src_,
                          name=name_, plot=plot_, filenames=filenames,
                          object_type=object_type_, mode=mode_))

    monkeypatch.setattr(
        spacr.plot, "_visualize_and_save_timelapse_stack_with_tracks", _fake_visualise
    )

    masks = _moving_masks(n_frames=3)
    _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))
    _run(src_dir, masks, plot=plot, save=save, object_type="nucleus")

    observed['len(calls) == (1 if expect_call else 0)'] = bool(len(calls) == (1 if expect_call else 0))
    if expect_call:
        call = calls[0]
        assert isinstance(call["masks"], np.ndarray)
        assert call["masks"].shape == masks.shape
        assert set(np.unique(call["masks"][0])) == {0, 10, 20, 30, 40}
        assert list(call["tracks_df"].columns[:5]) == [
            "track_id", "frame", "x", "y", "original_label"
        ]
        assert call["save"] is save and call["plot"] is plot
        assert call["object_type"] == "nucleus"
        assert call["mode"] == "btrack"
        assert call["filenames"] == [f"plate1_A01_1_t{i}.tif" for i in range(3)]
        assert os.path.basename(call["src"]) == "masks"
    return observed


_EXPECT_btrack_hands_masks_and_tracks_to_the_visualiser = {
    'len(calls) == (1 if expect_call else 0)': True,
}


# ---------------------------------------------------------------------------
# empty-input regressions (both were real crashes, now fixed)
# ---------------------------------------------------------------------------

def _case_btrack_with_no_objects_returns_zeroed_stack(src_dir):
    observed = {}
    masks = np.zeros((3, 32, 32), dtype=np.int32)
    out = _run(src_dir, masks, radius=5)

    observed['len(out)'] = len(out)
    observed['all((int(np.count_nonzero(f)) == 0 for f in out))'] = bool(all((int(np.count_nonzero(f)) == 0 for f in out)))
    return observed


_EXPECT_btrack_with_no_objects_returns_zeroed_stack = {
    'len(out)': 3,
    'all((int(np.count_nonzero(f)) == 0 for f in out))': True,
}


def _case_btrack_all_tracks_transient_returns_zeroed_stack(src_dir, monkeypatch):
    observed = {}
    masks = _moving_masks(n_frames=3)
    # every fake track lives for a single frame -> all shorter than n_frames
    tracks = _tracks_from_masks(masks, factor=10, only_first_frame_for=(1, 2, 3, 4))
    _install_fake_tracker(monkeypatch, tracks)

    out = _run(src_dir, masks, timelapse_remove_transient=True)

    observed['len(out)'] = len(out)
    observed['all((int(np.count_nonzero(f)) == 0 for f in out))'] = bool(all((int(np.count_nonzero(f)) == 0 for f in out)))
    observed['_tracks_csv(src_dir).exists()'] = bool(_tracks_csv(src_dir).exists())
    return observed


_EXPECT_btrack_all_tracks_transient_returns_zeroed_stack = {
    'len(out)': 3,
    'all((int(np.count_nonzero(f)) == 0 for f in out))': True,
    '_tracks_csv(src_dir).exists()': True,
}


# ---------------------------------------------------------------------------
# behaviour tests: one per public-function contract, one parametrised row per case
# ---------------------------------------------------------------------------

def _run_case(case, request):
    """Call a case helper with the fixtures named in its signature."""
    bound = getattr(case, "keywords", {})
    wanted = [n for n in inspect.signature(case).parameters if n not in bound]
    return case(**{n: request.getfixturevalue(n) for n in wanted})


_P = functools.partial


INPUT_CASES = [
    pytest.param(_case_btrack_tracks_moving_cells_end_to_end, _EXPECT_btrack_tracks_moving_cells_end_to_end, id="tracks-moving-cells-end-to-end"),
    pytest.param(_case_btrack_writes_well_metadata_parsed_from_name, _EXPECT_btrack_writes_well_metadata_parsed_from_name, id="writes-well-metadata-parsed-from-name"),
    pytest.param(_case_btrack_accepts_a_list_of_2d_masks, _EXPECT_btrack_accepts_a_list_of_2d_masks, id="accepts-a-list-of-2d-masks"),
    pytest.param(_P(_case_btrack_rejects_masks_that_are_not_a_uniform_time_stack, kind='empty-list', message='empty list'), {'error': _REJECT_MESSAGES['empty-list']}, id="rejects-masks-that-are-not-a-uniform-time-stack-empty-list"),
    pytest.param(_P(_case_btrack_rejects_masks_that_are_not_a_uniform_time_stack, kind='ragged-list', message='same shape'), {'error': _REJECT_MESSAGES['ragged-list']}, id="rejects-masks-that-are-not-a-uniform-time-stack-ragged-list"),
    pytest.param(_P(_case_btrack_rejects_masks_that_are_not_a_uniform_time_stack, kind='2d-array', message='must be 3D'), {'error': _REJECT_MESSAGES['2d-array']}, id="rejects-masks-that-are-not-a-uniform-time-stack-2d-array"),
    pytest.param(_P(_case_btrack_rejects_masks_that_are_not_a_uniform_time_stack, kind='4d-array', message='must be 3D'), {'error': _REJECT_MESSAGES['4d-array']}, id="rejects-masks-that-are-not-a-uniform-time-stack-4d-array"),
    pytest.param(_case_btrack_with_no_objects_returns_zeroed_stack, _EXPECT_btrack_with_no_objects_returns_zeroed_stack, id="with-no-objects-returns-zeroed-stack"),
]


@pytest.mark.parametrize("case, expected", INPUT_CASES)
def test_btrack_tracks_a_uniform_mask_stack_end_to_end_and_rejects_anything_else(case, expected, request):
    """Real btrack: ids stay per object, CSV and well metadata, list input, input validation, empty stack."""
    observed = _run_case(case, request)
    assert observed == expected


CONFIGURATION_CASES = [
    pytest.param(_case_btrack_configures_tracker_from_arguments, _EXPECT_btrack_configures_tracker_from_arguments, id="configures-tracker-from-arguments"),
    pytest.param(_P(_case_btrack_derives_radius_from_image_width, size=64, expected_radius=3), _EXPECT_btrack_derives_radius_from_image_width, id="derives-radius-from-image-width-64px"),
    pytest.param(_P(_case_btrack_derives_radius_from_image_width, size=16, expected_radius=1), _EXPECT_btrack_derives_radius_from_image_width, id="derives-radius-from-image-width-16px"),
    pytest.param(_case_btrack_falls_back_to_step_size_track_on_typeerror, _EXPECT_btrack_falls_back_to_step_size_track_on_typeerror, id="falls-back-to-step-size-track-on-typeerror"),
    pytest.param(_case_btrack_passes_glpk_options_and_uses_optimised_tracks, _EXPECT_btrack_passes_glpk_options_and_uses_optimised_tracks, id="passes-glpk-options-and-uses-optimised-tracks"),
    pytest.param(_case_btrack_optimises_with_default_options_when_none_given, _EXPECT_btrack_optimises_with_default_options_when_none_given, id="optimises-with-default-options-when-none-given"),
    pytest.param(_P(_case_btrack_ignores_non_positive_optimiser_settings, time_limit=0, mip_gap=0.01), {'optimize_calls': [{'backend': 'glpk', 'options': {'options': {'mip_gap': 0.01}}}]}, id="ignores-non-positive-optimiser-settings-zero-time-limit"),
    pytest.param(_P(_case_btrack_ignores_non_positive_optimiser_settings, time_limit=2, mip_gap=0), {'optimize_calls': [{'backend': 'glpk', 'options': {'options': {'tm_lim': 2000}}}]}, id="ignores-non-positive-optimiser-settings-zero-mip-gap"),
    pytest.param(_P(_case_btrack_ignores_non_positive_optimiser_settings, time_limit=0, mip_gap=0), {'optimize_calls': [{'backend': 'glpk'}]}, id="ignores-non-positive-optimiser-settings-both-zero"),
    pytest.param(_case_btrack_skips_optimisation_when_disabled, _EXPECT_btrack_skips_optimisation_when_disabled, id="skips-optimisation-when-disabled"),
    pytest.param(_case_btrack_skips_optimisation_for_too_many_objects, _EXPECT_btrack_skips_optimisation_for_too_many_objects, id="skips-optimisation-for-too-many-objects"),
    pytest.param(_case_btrack_falls_back_to_raw_tracks_when_optimiser_fails, _EXPECT_btrack_falls_back_to_raw_tracks_when_optimiser_fails, id="falls-back-to-raw-tracks-when-optimiser-fails"),
]


@pytest.mark.parametrize("case, expected", CONFIGURATION_CASES)
def test_btrack_configures_the_tracker_and_the_global_optimiser_from_its_arguments(case, expected, request):
    """Fake tracker: radius, features, volume, update method, TypeError fallback, GLPK options, skip and failure paths."""
    observed = _run_case(case, request)
    assert observed == expected


OUTPUT_CASES = [
    pytest.param(_P(_case_btrack_transient_filter_drops_single_frame_tracks, remove_transient=True, expect_transient=False), _EXPECT_btrack_transient_filter_drops_single_frame_tracks, id="transient-filter-drops-single-frame-tracks-removed"),
    pytest.param(_P(_case_btrack_transient_filter_drops_single_frame_tracks, remove_transient=False, expect_transient=True), _EXPECT_btrack_transient_filter_drops_single_frame_tracks, id="transient-filter-drops-single-frame-tracks-kept"),
    pytest.param(_case_btrack_survives_unparsable_well_metadata, _EXPECT_btrack_survives_unparsable_well_metadata, id="survives-unparsable-well-metadata"),
    pytest.param(_P(_case_btrack_hands_masks_and_tracks_to_the_visualiser, plot=False, save=False, expect_call=False), _EXPECT_btrack_hands_masks_and_tracks_to_the_visualiser, id="hands-masks-and-tracks-to-the-visualiser-neither"),
    pytest.param(_P(_case_btrack_hands_masks_and_tracks_to_the_visualiser, plot=True, save=False, expect_call=True), _EXPECT_btrack_hands_masks_and_tracks_to_the_visualiser, id="hands-masks-and-tracks-to-the-visualiser-plot"),
    pytest.param(_P(_case_btrack_hands_masks_and_tracks_to_the_visualiser, plot=False, save=True, expect_call=True), _EXPECT_btrack_hands_masks_and_tracks_to_the_visualiser, id="hands-masks-and-tracks-to-the-visualiser-save"),
    pytest.param(_P(_case_btrack_hands_masks_and_tracks_to_the_visualiser, plot=True, save=True, expect_call=True), _EXPECT_btrack_hands_masks_and_tracks_to_the_visualiser, id="hands-masks-and-tracks-to-the-visualiser-plot-and-save"),
    pytest.param(_case_btrack_all_tracks_transient_returns_zeroed_stack, _EXPECT_btrack_all_tracks_transient_returns_zeroed_stack, id="all-tracks-transient-returns-zeroed-stack"),
]


@pytest.mark.parametrize("case, expected", OUTPUT_CASES)
def test_btrack_writes_the_tracks_csv_filters_transient_tracks_and_hands_results_to_the_visualiser(case, expected, request):
    """Output side: transient filtering, metadata parse failure, visualiser hand-off, all-transient result."""
    observed = _run_case(case, request)
    assert observed == expected
