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

def test_btrack_tracks_moving_cells_end_to_end(src_dir):
    """Four drifting blobs keep one label per object across all four frames."""
    masks = _moving_masks()
    out = _run(src_dir, masks)

    assert isinstance(out, list)
    assert len(out) == 4
    assert all(f.shape == (64, 64) and f.dtype == masks.dtype for f in out)

    # every frame carries exactly the same four track ids
    label_sets = [set(np.unique(f)) - {0} for f in out]
    assert all(len(s) == 4 for s in label_sets)
    assert label_sets[0] == label_sets[1] == label_sets[2] == label_sets[3]

    # and each blob keeps its id while it drifts
    for cy, cx in CENTRES:
        ids = {int(out[t][cy + t, cx + t]) for t in range(4)}
        assert len(ids) == 1 and 0 not in ids

    csv = _tracks_csv(src_dir)
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 16  # 4 objects x 4 frames
    assert set(df["frame"]) == {0, 1, 2, 3}
    assert df["track_id"].nunique() == 4
    assert set(df["original_label"]) == {1, 2, 3, 4}


def test_btrack_writes_well_metadata_parsed_from_name(src_dir):
    """The npz batch name is decomposed into plate/row/column/field/well."""
    masks = _moving_masks(n_frames=3)
    _run(src_dir, masks)

    df = pd.read_csv(_tracks_csv(src_dir))
    assert set(df["file_name"]) == {"plate1_A01_1"}
    assert set(df["plateID"]) == {"plate1"}
    assert set(df["rowID"]) == {"r1"}
    assert set(df["columnID"]) == {"c1"}
    assert set(df["fieldID"]) == {"f1"}
    assert set(df["prcf"]) == {"plate1_r1_c1_f1"}
    assert set(df["wellID"]) == {"A01"}


def test_btrack_accepts_a_list_of_2d_masks(src_dir, tmp_path):
    """A list of 2-D frames is stacked and gives the same answer as an ndarray."""
    masks = _moving_masks(n_frames=3)
    from_array = _run(src_dir, masks)

    other = tmp_path / "other"
    (other / "masks").mkdir(parents=True)
    from_list = _run(other, [masks[t] for t in range(masks.shape[0])])

    assert len(from_list) == len(from_array) == 3
    for a, b in zip(from_array, from_list):
        np.testing.assert_array_equal(a, b)


def test_btrack_rejects_empty_mask_list(src_dir):
    with pytest.raises(ValueError, match="empty list"):
        _run(src_dir, [])


def test_btrack_rejects_ragged_mask_list(src_dir):
    ragged = [np.zeros((8, 8), dtype=np.int32), np.zeros((8, 9), dtype=np.int32)]
    with pytest.raises(ValueError, match="same shape"):
        _run(src_dir, ragged)


@pytest.mark.parametrize("bad", [np.zeros((8, 8), np.int32),
                                 np.zeros((2, 3, 4, 5), np.int32)])
def test_btrack_rejects_non_3d_masks(src_dir, bad):
    with pytest.raises(ValueError, match="must be 3D"):
        _run(src_dir, bad)


# ---------------------------------------------------------------------------
# tracker configuration (fake BayesianTracker)
# ---------------------------------------------------------------------------

def test_btrack_configures_tracker_from_arguments(src_dir, monkeypatch):
    from btrack import datasets as btrack_datasets
    from btrack.constants import BayesianUpdates

    masks = _moving_masks()
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))
    out = _run(src_dir, masks, radius=37)

    tracker = rec["instance"]
    assert rec["config"] == btrack_datasets.cell_config()
    assert tracker.update_method == BayesianUpdates.APPROXIMATE
    assert tracker.max_search_radius == 37
    assert tracker.features == FEATURES
    assert tracker.volume == ((0, 64), (0, 64))
    assert rec["n_appended"] == 16  # 4 blobs x 4 frames of real segmentation
    assert rec["track_calls"] == [{"tracking_updates": ["motion", "visual"]}]
    assert rec["closed"] is True

    # track ids came from the fake tracks: label L -> 10 * L
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}


@pytest.mark.parametrize("size,expected_radius", [(64, 3), (16, 1)])
def test_btrack_derives_radius_from_image_width(src_dir, monkeypatch, size,
                                                expected_radius):
    """radius=None -> width/20, floored at 1 for tiny images."""
    centres = CENTRES if size == 64 else [(8, 8)]
    masks = _moving_masks(n_frames=2, size=size, radius=2, centres=centres)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, radius=None)

    assert rec["instance"].max_search_radius == expected_radius
    assert rec["instance"].volume == ((0, size), (0, size))


def test_btrack_falls_back_to_step_size_track_on_typeerror(src_dir, monkeypatch):
    """Older btrack: track(tracking_updates=...) -> TypeError -> attribute + step_size."""
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch, _tracks_from_masks(masks), track_typeerror=True
    )

    out = _run(src_dir, masks)

    assert rec["track_calls"] == [
        {"tracking_updates": ["motion", "visual"]},
        {"step_size": 100},
    ]
    assert rec["instance"].tracking_updates == ["MOTION", "VISUAL"]
    # the fallback still produced usable tracks
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}


# ---------------------------------------------------------------------------
# global optimisation branches
# ---------------------------------------------------------------------------

def test_btrack_passes_glpk_options_and_uses_optimised_tracks(src_dir, monkeypatch):
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    out = _run(src_dir, masks, optimizer_time_limit_s=2.5, optimizer_mip_gap=0.05)

    assert rec["optimize_calls"] == [
        {"backend": "glpk",
         "options": {"options": {"tm_lim": 2500, "mip_gap": 0.05}}}
    ]
    # post-optimisation tracks are the ones written out
    assert set(np.unique(out[0])) == {0, 100, 200, 300, 400}


def test_btrack_optimises_with_default_options_when_none_given(src_dir, monkeypatch):
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, optimizer_time_limit_s=None, optimizer_mip_gap=None)

    assert rec["optimize_calls"] == [{"backend": "glpk"}]


@pytest.mark.parametrize(
    "time_limit,mip_gap,expected",
    [
        (0, 0.01, {"mip_gap": 0.01}),
        (2, 0, {"tm_lim": 2000}),
        (0, 0, None),
    ],
)
def test_btrack_ignores_non_positive_optimiser_settings(src_dir, monkeypatch,
                                                        time_limit, mip_gap,
                                                        expected):
    """Zero time-limit / MIP gap are treated as 'unset', not as literal zeros."""
    masks = _moving_masks(n_frames=2)
    rec = _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    _run(src_dir, masks, optimizer_time_limit_s=time_limit, optimizer_mip_gap=mip_gap)

    if expected is None:
        assert rec["optimize_calls"] == [{"backend": "glpk"}]
    else:
        assert rec["optimize_calls"] == [
            {"backend": "glpk", "options": {"options": expected}}
        ]


def test_btrack_skips_optimisation_when_disabled(src_dir, monkeypatch):
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    out = _run(src_dir, masks, run_optimization=False)

    assert rec["optimize_calls"] == []
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}


def test_btrack_skips_optimisation_for_too_many_objects(src_dir, monkeypatch, caplog):
    masks = _moving_masks(n_frames=3)  # 12 real objects
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
    )

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks, max_objects_for_optimization=5)

    assert rec["optimize_calls"] == []
    assert any("Skipping btrack global optimisation" in r.getMessage()
               for r in caplog.records)
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}


def test_btrack_falls_back_to_raw_tracks_when_optimiser_fails(src_dir, monkeypatch,
                                                              caplog):
    masks = _moving_masks(n_frames=3)
    rec = _install_fake_tracker(
        monkeypatch,
        _tracks_from_masks(masks, factor=10),
        optimized_tracks=_tracks_from_masks(masks, factor=100),
        optimize_error=RuntimeError("GLPK stalled"),
    )

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks)

    assert len(rec["optimize_calls"]) == 1
    assert any("global optimisation failed" in r.getMessage()
               for r in caplog.records)
    # pre-optimisation tracks survive the failure
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}
    assert _tracks_csv(src_dir).exists()


# ---------------------------------------------------------------------------
# transient-track filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("remove_transient,expect_transient", [(True, False),
                                                               (False, True)])
def test_btrack_transient_filter_drops_single_frame_tracks(src_dir, monkeypatch,
                                                           remove_transient,
                                                           expect_transient):
    """A track that only exists in frame 0 survives only when filtering is off."""
    masks = _moving_masks(n_frames=3, transient_label=True)
    tracks = _tracks_from_masks(masks, factor=10, only_first_frame_for=(5,))
    _install_fake_tracker(monkeypatch, tracks)

    out = _run(src_dir, masks, timelapse_remove_transient=remove_transient)

    frame0 = set(np.unique(out[0]))
    assert {0, 10, 20, 30, 40} <= frame0
    assert (50 in frame0) is expect_transient

    df = pd.read_csv(_tracks_csv(src_dir))
    assert (50 in set(df["track_id"])) is expect_transient
    assert set(df["frame"]) == {0, 1, 2}


# ---------------------------------------------------------------------------
# well-metadata parsing failure
# ---------------------------------------------------------------------------

def test_btrack_survives_unparsable_well_metadata(src_dir, monkeypatch, caplog):
    """An IndexError out of _map_wells is logged and the CSV is still written."""
    import spacr.utils

    def _boom(file_name, timelapse=False):
        raise IndexError(f"cannot parse {file_name}")

    monkeypatch.setattr(spacr.utils, "_map_wells", _boom)

    masks = _moving_masks(n_frames=3)
    _install_fake_tracker(monkeypatch, _tracks_from_masks(masks))

    with caplog.at_level(logging.WARNING, logger="spacr.timelapse"):
        out = _run(src_dir, masks)

    assert any("Failed to parse plate, well, field" in r.getMessage()
               for r in caplog.records)
    df = pd.read_csv(_tracks_csv(src_dir))
    assert "plateID" not in df.columns
    assert "wellID" not in df.columns
    assert set(df["track_id"]) == {10, 20, 30, 40}
    assert set(np.unique(out[0])) == {0, 10, 20, 30, 40}


# ---------------------------------------------------------------------------
# visualisation hand-off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plot,save,expect_call", [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_btrack_hands_masks_and_tracks_to_the_visualiser(src_dir, monkeypatch,
                                                         plot, save, expect_call):
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

    assert len(calls) == (1 if expect_call else 0)
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


# ---------------------------------------------------------------------------
# empty-input regressions (both were real crashes, now fixed)
# ---------------------------------------------------------------------------

def test_btrack_with_no_objects_returns_zeroed_stack(src_dir):
    masks = np.zeros((3, 32, 32), dtype=np.int32)
    out = _run(src_dir, masks, radius=5)

    assert len(out) == 3
    assert all(int(np.count_nonzero(f)) == 0 for f in out)


def test_btrack_all_tracks_transient_returns_zeroed_stack(src_dir, monkeypatch):
    masks = _moving_masks(n_frames=3)
    # every fake track lives for a single frame -> all shorter than n_frames
    tracks = _tracks_from_masks(masks, factor=10, only_first_frame_for=(1, 2, 3, 4))
    _install_fake_tracker(monkeypatch, tracks)

    out = _run(src_dir, masks, timelapse_remove_transient=True)

    assert len(out) == 3
    assert all(int(np.count_nonzero(f)) == 0 for f in out)
    assert _tracks_csv(src_dir).exists()
