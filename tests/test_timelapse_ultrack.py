"""CPU-only tests for the Ultrack tracking backend in spacr.timelapse.

Ultrack is an OPTIONAL dependency: it brings an ILP solver and a database
backend, and a real solve writes a sqlite store to disk. Nothing here installs
it, downloads anything, or runs a solver — the package is injected into
``sys.modules`` as a stub, exactly as the sibling ``test_timelapse_trackastra``
suite does, which lets these tests pin the part spaCR actually owns: the
adapter contract. Specifically that

  * the tracks table matches the layout trackpy/btrack/trackastra emit, because
    the track visualiser and the motility assay both read it,
  * ids are consistent across frames and the centroids come from the relabelled
    stack rather than from Ultrack's own coordinate table,
  * the temporary data store is gone when the adapter returns, and the user's
    run folder gains the tracks CSV and nothing else,
  * degenerate inputs (single frame, empty masks, shape mismatch) are handled
    rather than reaching Ultrack,
  * Ultrack's renamed entry points (labels_to_edges/labels_to_contours,
    detection+edges/foreground+contours) are both driven from one adapter,
  * and a missing package produces an actionable message instead of a bare
    ImportError from three frames down.

That last one matters: this repo has a documented habit of failures that look
like "no data", and an optional-dependency ImportError is exactly that shape.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# stub ultrack
# ---------------------------------------------------------------------------

def _disc(frame, cy, cx, label, r=4):
    """Stamp a filled disc of `label` into `frame`."""
    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    frame[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = label
    return frame


def _moving_stack(n_frames=4, size=48, n_objects=2):
    """A (T, Y, X) label stack whose objects drift right by 3 px per frame.

    Labels are deliberately SHUFFLED per frame — object 1 is not label 1 in
    every frame — so a test that passes cannot be passing by accident of the
    labels already being consistent.
    """
    masks = np.zeros((n_frames, size, size), dtype=np.uint16)
    for t in range(n_frames):
        for i in range(n_objects):
            label = (i + t) % n_objects + 1
            _disc(masks[t], cy=12 + i * 18, cx=10 + 3 * t, label=label)
    return masks


def _consistent_stack(n_frames=3, size=48):
    """What a working tracker returns: the same id on the same object."""
    tracked = np.zeros((n_frames, size, size), dtype=np.uint16)
    for t in range(n_frames):
        _disc(tracked[t], cy=12, cx=10 + 3 * t, label=1)
        _disc(tracked[t], cy=30, cx=10 + 3 * t, label=2)
    return tracked


class _Section:
    """Stand-in for one pydantic section of Ultrack's MainConfig."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _install_stub_ultrack(monkeypatch, relabelled=None, record=None,
                          legacy=False, drop_fields=(), broken_track=False,
                          contours_name="labels_to_contours"):
    """Inject a fake `ultrack` package.

    :param relabelled: the stack ``tracks_to_zarr`` hands back; this is what the
        adapter must derive its tracks table from.
    :param record: dict that collects what the adapter did to Ultrack.
    :param legacy: expose the pre-rename ``track(detection=, edges=)`` signature.
    :param drop_fields: config field names to delete, to exercise version drift.
    :param broken_track: expose a ``track()`` matching neither generation.
    :param contours_name: which of the two converter names ``ultrack.utils``
        exposes.
    """
    pkg = types.ModuleType("ultrack")
    utils_mod = types.ModuleType("ultrack.utils")

    class _MainConfig:
        def __init__(self):
            self.data_config = _Section(working_dir=None, database="sqlite")
            self.segmentation_config = _Section(n_workers=1, min_area=100)
            self.linking_config = _Section(max_distance=15.0, n_workers=1)
            self.tracking_config = _Section(appear_weight=-0.001,
                                            disappear_weight=-0.001,
                                            division_weight=-0.001)
            for field in drop_fields:
                for section in (self.data_config, self.segmentation_config,
                                self.linking_config, self.tracking_config):
                    section.__dict__.pop(field, None)
            if record is not None:
                record["config"] = self

    def _labels_to_contours(labels, sigma=None):
        if record is not None:
            record["sigma"] = sigma
            record["n_label_stacks"] = len(labels)
            record["labels_shape"] = np.asarray(labels[0]).shape
        arr = np.asarray(labels[0])
        return (arr > 0), (arr > 0).astype(np.float32)

    def _note_track(config, images):
        if record is None:
            return
        work_dir = config.data_config.working_dir
        record["working_dir"] = str(work_dir)
        # the store must exist *while* Ultrack runs, and be gone afterwards
        record["working_dir_existed"] = os.path.isdir(str(work_dir))
        record["max_distance"] = config.linking_config.max_distance
        record["division_weight"] = config.tracking_config.division_weight
        record["n_workers"] = (config.segmentation_config.n_workers,
                               config.linking_config.n_workers)
        record["images"] = None if not len(images) else np.asarray(images[0]).shape
        # Ultrack really would write its database here; prove we cleaned it up.
        with open(os.path.join(str(work_dir), "data.db"), "wb") as fh:
            fh.write(b"stub")

    if broken_track:
        def _track(config, blobs=None, rims=None):
            raise AssertionError("adapter must not call an unsupported track()")
    elif legacy:
        def _track(config, detection=None, edges=None, images=()):
            if record is not None:
                record["kwargs"] = ("detection", "edges")
                record["foreground_shape"] = np.asarray(detection).shape
            _note_track(config, images)
    else:
        def _track(config, foreground=None, contours=None, images=()):
            if record is not None:
                record["kwargs"] = ("foreground", "contours")
                record["foreground_shape"] = np.asarray(foreground).shape
            _note_track(config, images)

    def _to_tracks_layer(config, include_parents=True):
        # Ultrack's own table carries coordinates the adapter must NOT trust:
        # these are deliberately wrong so a test can prove x/y came from the
        # exported segmentation instead.
        table = pd.DataFrame({"track_id": [1], "t": [0], "y": [-999.0],
                              "x": [-999.0], "parent_track_id": [-1]})
        if record is not None:
            record["tracks_layer"] = True
        return table, {}

    def _tracks_to_zarr(config, tracks_df, store_or_path=None):
        if record is not None:
            record["zarr_store"] = store_or_path
        return relabelled if relabelled is not None else np.zeros((2, 8, 8), np.uint16)

    pkg.MainConfig = _MainConfig
    pkg.track = _track
    pkg.to_tracks_layer = _to_tracks_layer
    pkg.tracks_to_zarr = _tracks_to_zarr
    setattr(utils_mod, contours_name, _labels_to_contours)
    pkg.utils = utils_mod
    monkeypatch.setitem(sys.modules, "ultrack", pkg)
    monkeypatch.setitem(sys.modules, "ultrack.utils", utils_mod)
    return pkg


@pytest.fixture(autouse=True)
def _no_figures():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _run_dir(tmp_path):
    """`src` points at a batch inside the run folder; tracks land beside it."""
    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    return src


# ---------------------------------------------------------------------------
# missing dependency
# ---------------------------------------------------------------------------

def test_missing_ultrack_raises_an_actionable_runtimeerror(monkeypatch, tmp_path):
    """A bare ImportError would read like 'no data'. It must name the fix."""
    from spacr.timelapse import _ultrack_track_cells

    # make the import fail even if ultrack is installed in the env
    monkeypatch.setitem(sys.modules, "ultrack", None)
    monkeypatch.setitem(sys.modules, "ultrack.utils", None)

    with pytest.raises(RuntimeError) as exc:
        _ultrack_track_cells(
            src=str(tmp_path / "run" / "x"), name="b1", batch_filenames=[],
            object_type="cell", masks=_moving_stack())

    msg = str(exc.value)
    assert "pip install spacr[ultrack]" in msg
    # naming the alternatives is the point: the user has three working backends
    for alt in ("trackastra", "trackpy", "btrack", "iou"):
        assert alt in msg


# ---------------------------------------------------------------------------
# adapter contract
# ---------------------------------------------------------------------------

def test_tracks_table_matches_the_trackpy_btrack_layout(monkeypatch, tmp_path):
    """The visualiser and motility assay read frame/track_id/x/y."""
    from spacr.timelapse import _ultrack_track_cells

    masks = _moving_stack(n_frames=3, n_objects=2)
    tracked = _consistent_stack(n_frames=3)
    _install_stub_ultrack(monkeypatch, relabelled=tracked)

    src = _run_dir(tmp_path)
    out = _ultrack_track_cells(
        src=str(src), name="b1", batch_filenames=["a.tif", "b.tif", "c.tif"],
        object_type="cell", masks=masks, images=masks.astype(np.float32))

    csv = tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b1.csv"
    assert csv.exists(), "tracks CSV was not written"
    df = pd.read_csv(csv)
    assert list(df.columns) == ["frame", "track_id", "original_label", "x", "y"]
    assert len(out) == masks.shape[0]   # _masks_to_masks_stack returns a list


def test_tracks_table_columns_agree_with_the_trackastra_backend(monkeypatch, tmp_path):
    """Both stack-returning backends must emit an identical layout.

    The motility assay reads whichever CSV exists, so a column that drifted
    between the two backends would be a silent, mode-dependent failure.
    """
    from spacr.timelapse import (_relabelled_stack_to_tracks_df,
                                 _ultrack_track_cells)

    tracked = _consistent_stack(n_frames=3)
    _install_stub_ultrack(monkeypatch, relabelled=tracked)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=3))

    mine = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b.csv")
    theirs = _relabelled_stack_to_tracks_df(tracked)
    assert list(mine.columns) == list(theirs.columns)
    pd.testing.assert_frame_equal(mine, theirs, check_dtype=False)


def test_ids_are_consistent_across_frames(monkeypatch, tmp_path):
    from spacr.timelapse import _ultrack_track_cells

    tracked = _consistent_stack(n_frames=3)
    _install_stub_ultrack(monkeypatch, relabelled=tracked)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=3))

    df = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b.csv")
    assert sorted(df["track_id"].unique().tolist()) == [1, 2]
    assert df.groupby("track_id")["frame"].nunique().tolist() == [3, 3]


def test_centroids_come_from_the_relabelled_stack_not_ultracks_table(monkeypatch, tmp_path):
    """Ultrack's to_tracks_layer coordinates can disagree with the export.

    The stub returns x=y=-999 from to_tracks_layer; if those ever reached the
    CSV the visualiser would plot tracks off-canvas.
    """
    from spacr.timelapse import _ultrack_track_cells

    tracked = _consistent_stack(n_frames=3)
    _install_stub_ultrack(monkeypatch, relabelled=tracked)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=3))

    df = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b.csv")
    assert (df[["x", "y"]] >= 0).all().all(), "Ultrack's own coordinates leaked through"
    # objects drift right by 3 px per frame in the relabelled stack
    xs = df[df["track_id"] == 1].sort_values("frame")["x"].tolist()
    assert xs[1] - xs[0] == pytest.approx(3.0, abs=0.6)
    ys = df[df["track_id"] == 1]["y"].tolist()
    assert all(y == pytest.approx(12.0, abs=0.6) for y in ys)


def test_labels_are_fed_in_rather_than_a_user_supplied_contour_map(monkeypatch, tmp_path):
    """The user has labels, not contours; the adapter must do the conversion."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    masks = _moving_stack(n_frames=3)
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(3), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=masks)

    assert rec["n_label_stacks"] == 1
    assert rec["labels_shape"] == masks.shape
    assert rec["foreground_shape"] == masks.shape


def test_images_are_forwarded_for_appearance_features(monkeypatch, tmp_path):
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    masks = _moving_stack(n_frames=2)
    imgs = (masks > 0).astype(np.float32) * 1234.0
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=masks, images=imgs)
    assert rec["images"] == imgs.shape


def test_no_images_means_geometry_only(monkeypatch, tmp_path):
    """images=None must not crash, and must not fabricate an intensity stack."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=2),
                         images=None)
    assert rec["images"] is None


# ---------------------------------------------------------------------------
# the knobs spaCR surfaces
# ---------------------------------------------------------------------------

def test_the_surfaced_knobs_reach_ultracks_config(monkeypatch, tmp_path):
    """A setting that never lands on the config is a lie in the GUI."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=2),
                         max_distance=42.5, division_weight=-2.5,
                         contour_sigma=3.0, n_workers=4)

    assert rec["max_distance"] == pytest.approx(42.5)
    assert rec["division_weight"] == pytest.approx(-2.5)
    assert rec["n_workers"] == (4, 4)
    assert rec["sigma"] == pytest.approx(3.0)


def test_zero_contour_sigma_means_no_smoothing(monkeypatch, tmp_path):
    """Ultrack spells 'no smoothing' as None; the GUI has no tri-state float."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=2),
                         contour_sigma=0.0)
    assert rec["sigma"] is None


def test_a_config_field_that_moved_names_the_spacr_setting(monkeypatch, tmp_path):
    """Ultrack renames config fields between releases and spaCR pins no version.

    A raw pydantic error would name only Ultrack's field, leaving the user with
    no idea which knob in the GUI to change.
    """
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2),
                          drop_fields=("max_distance",))
    src = _run_dir(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                             object_type="cell", masks=_moving_stack(n_frames=2))
    msg = str(exc.value)
    assert "ultrack_max_distance" in msg
    assert "pip install spacr[ultrack]" in msg


# ---------------------------------------------------------------------------
# version drift in Ultrack's own API
# ---------------------------------------------------------------------------

def test_the_pre_rename_detection_edges_signature_still_works(monkeypatch, tmp_path):
    """Older ultrack took detection=/edges=, newer takes foreground=/contours=."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2), record=rec,
                          legacy=True, contours_name="labels_to_edges")
    src = _run_dir(tmp_path)
    out = _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                               object_type="cell", masks=_moving_stack(n_frames=2))
    assert rec["kwargs"] == ("detection", "edges")
    assert len(out) == 2


def test_an_unrecognisable_track_signature_is_reported_not_guessed(monkeypatch, tmp_path):
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2),
                          broken_track=True)
    src = _run_dir(tmp_path)
    with pytest.raises(RuntimeError, match="foreground/contours"):
        _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                             object_type="cell", masks=_moving_stack(n_frames=2))


def test_a_missing_labels_converter_is_reported(monkeypatch, tmp_path):
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2),
                          contours_name="something_else")
    src = _run_dir(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                             object_type="cell", masks=_moving_stack(n_frames=2))
    assert "labels_to_contours" in str(exc.value)
    assert "labels_to_edges" in str(exc.value)


# ---------------------------------------------------------------------------
# the temporary data store
# ---------------------------------------------------------------------------

def test_the_data_store_lives_in_a_temp_dir_that_is_cleaned_up(monkeypatch, tmp_path):
    """A stray sqlite database in the user's output folder is unacceptable."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(3), record=rec)
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=3))

    work_dir = rec["working_dir"]
    assert rec["working_dir_existed"], "the store must exist while Ultrack runs"
    assert not os.path.isdir(work_dir), f"temporary store left behind at {work_dir}"
    assert str(tmp_path) not in work_dir, "the store must not sit in the run folder"


def test_nothing_but_the_tracks_csv_is_written_into_the_run_folder(monkeypatch, tmp_path):
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(3))
    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=_moving_stack(n_frames=3))

    written = sorted(
        os.path.relpath(os.path.join(root, f), str(tmp_path))
        for root, _dirs, files in os.walk(str(tmp_path)) for f in files)
    assert written == [os.path.join("run", "tracks", "ultrack_tracks_cell_b.csv")]


def test_the_store_is_cleaned_up_even_when_the_solve_raises(monkeypatch, tmp_path):
    """A failed solve must not leave a database behind either."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    pkg = _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(2),
                                record=rec)
    real_to_zarr = pkg.tracks_to_zarr

    def _boom(config, tracks_df, store_or_path=None):
        real_to_zarr(config, tracks_df, store_or_path)
        raise ValueError("solver blew up")

    pkg.tracks_to_zarr = _boom
    src = _run_dir(tmp_path)
    with pytest.raises(ValueError, match="solver blew up"):
        _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                             object_type="cell", masks=_moving_stack(n_frames=2))
    assert not os.path.isdir(rec["working_dir"])


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------

def test_single_frame_returns_without_constructing_ultrack(monkeypatch, tmp_path, capsys):
    """One frame has nothing to link — don't pay for a solve."""
    from spacr.timelapse import _ultrack_track_cells

    rec = {}
    _install_stub_ultrack(monkeypatch, record=rec)
    masks = _moving_stack(n_frames=1)
    src = _run_dir(tmp_path)
    out = _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                               object_type="cell", masks=masks)
    assert rec == {}, "Ultrack should not have been constructed"
    assert "nothing to link" in capsys.readouterr().out
    assert len(out) == 1


def test_non_3d_masks_are_rejected(monkeypatch, tmp_path):
    """A 2D mask would index as a stack of rows and fail deep inside Ultrack."""
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch)
    with pytest.raises(ValueError, match=r"\(T, Y, X\)"):
        _ultrack_track_cells(src=str(tmp_path / "r" / "b"), name="b",
                             batch_filenames=[], object_type="cell",
                             masks=np.zeros((8, 8), np.uint16))


def test_mismatched_image_shape_is_rejected(monkeypatch, tmp_path):
    """A silent shape mismatch would give Ultrack misaligned appearance."""
    from spacr.timelapse import _ultrack_track_cells

    _install_stub_ultrack(monkeypatch)
    masks = _moving_stack(n_frames=2, size=48)
    with pytest.raises(ValueError, match="does not match"):
        _ultrack_track_cells(src=str(tmp_path / "r" / "b"), name="b",
                             batch_filenames=[], object_type="cell",
                             masks=masks, images=np.zeros((2, 16, 16), np.float32))


def test_empty_masks_produce_an_empty_tracks_table(monkeypatch, tmp_path):
    """No objects must not raise; it should yield an empty, correct table."""
    from spacr.timelapse import _ultrack_track_cells

    blank = np.zeros((3, 32, 32), dtype=np.uint16)
    _install_stub_ultrack(monkeypatch, relabelled=blank)
    src = _run_dir(tmp_path)
    out = _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                               object_type="cell", masks=blank)
    df = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b.csv")
    assert df.empty
    assert list(df.columns) == ["frame", "track_id", "original_label", "x", "y"]
    assert len(out) == 3


def test_remove_transient_drops_tracks_absent_from_some_frame(monkeypatch, tmp_path, capsys):
    """A track seen in 2 of 3 frames goes when timelapse_remove_transient is on."""
    from spacr.timelapse import _ultrack_track_cells

    tracked = np.zeros((3, 48, 48), dtype=np.uint16)
    for t in range(3):
        _disc(tracked[t], cy=12, cx=10 + 3 * t, label=1)     # all three frames
    for t in range(2):
        _disc(tracked[t], cy=32, cx=10 + 3 * t, label=2)     # only two
    _install_stub_ultrack(monkeypatch, relabelled=tracked)

    src = _run_dir(tmp_path)
    _ultrack_track_cells(src=str(src), name="b", batch_filenames=[],
                         object_type="cell", masks=tracked,
                         timelapse_remove_transient=True)
    df = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b.csv")
    assert df["track_id"].unique().tolist() == [1]
    assert "Removed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_trackastra_remains_the_default_timelapse_mode():
    """Ultrack is added alongside trackastra, not in place of it."""
    from spacr.settings import set_default_settings_preprocess_generate_masks as defaults
    s = defaults({"src": "x"})
    assert s["timelapse_mode"] == "trackastra"


def test_ultrack_settings_have_defaults():
    from spacr.settings import set_default_settings_preprocess_generate_masks as defaults
    s = defaults({"src": "x"})
    assert s["ultrack_max_distance"] == 25.0
    assert s["ultrack_division_weight"] == -0.1
    assert s["ultrack_contour_sigma"] == 0.0
    assert s["ultrack_n_workers"] == 1


def test_ultrack_settings_are_typed_and_categorised():
    from spacr.settings import categories, expected_types
    floats = ("ultrack_max_distance", "ultrack_division_weight", "ultrack_contour_sigma")
    for k in floats:
        assert expected_types.get(k) is float
    assert expected_types.get("ultrack_n_workers") is int
    for k in floats + ("ultrack_n_workers",):
        assert k in categories["Timelapse"]


def test_ultrack_settings_have_tooltips_meeting_the_house_bar():
    """The tooltip gate is enforced repo-wide; assert it here too.

    A knob whose tooltip only restates its name is worse than no knob: the user
    changes it blind.
    """
    import re
    from spacr.settings import tooltips

    for k in ("ultrack_max_distance", "ultrack_division_weight",
              "ultrack_contour_sigma", "ultrack_n_workers"):
        text = tooltips[k]
        m = re.match(r"^\((?P<type>[^)]+)\)\s*-\s*(?P<body>.+)$", text.strip(), re.S)
        assert m, f"{k} tooltip has no (type) prefix"
        assert len(m.group("body").split()) >= 15, f"{k} tooltip is thin"
        assert "\n" not in text and "`" not in text


def test_timelapse_mode_tooltip_lists_ultrack():
    """The combo offers it, so the tooltip has to say what it is."""
    from spacr.settings import tooltips
    assert "ultrack" in tooltips["timelapse_mode"]


def test_gui_offers_ultrack_in_the_timelapse_mode_combo():
    """A backend the combo cannot select is a backend nobody can run.

    This used to grep ``inspect.getsource(spacr.gui_utils)`` for the dict
    literal. The literal moved to ``spacr.settings_spec`` (an import-cost
    change: the Qt interface wants the widget spec without Tk's 770 ms of
    imports) and ``gui_utils`` re-exports the function, so the grep went
    looking for text that had left the file while the combo itself was fine.
    Calling the function instead of reading the source asks the question the
    GUI actually asks, and survives the next move.
    """
    import spacr.gui_utils as G

    spec = G.convert_settings_dict_for_gui({"timelapse_mode": "trackastra"})
    assert "timelapse_mode" in spec, "timelapse_mode has no widget spec"
    kind, options, default = spec["timelapse_mode"]
    assert kind == "combo"
    assert "ultrack" in options
    assert set(options) == {"trackastra", "ultrack", "trackpy", "iou", "btrack"}
    assert default == "trackastra", "the default must not change"


def test_object_dispatch_imports_the_ultrack_backend():
    """object.py must be able to reach it, or the mode is unreachable."""
    import inspect
    import spacr.object as O
    src = inspect.getsource(O)
    assert "_ultrack_track_cells" in src
    assert "timelapse_mode == 'ultrack'" in src


def test_setup_declares_the_ultrack_extra():
    """The error message promises `pip install spacr[ultrack]`; it must exist."""
    import pathlib
    import re
    import spacr

    setup_py = pathlib.Path(spacr.__file__).resolve().parent.parent / "setup.py"
    if not setup_py.exists():
        pytest.skip("running against an installed spacr without setup.py")
    text = setup_py.read_text()
    assert re.search(r"'ultrack':\s*\[", text), "extras_require['ultrack'] missing"
