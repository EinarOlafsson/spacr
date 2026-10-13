"""The corners of :mod:`spacr.object` that a resumed or hand-edited run hits.

Three things happen here that a first, fully-specified run never sees:

* an older scikit-image without ``max_size`` -- spaCR's own size thresholds
  have to keep their ``<`` boundary through the fallback call;
* settings that already carry a recorded ``cellpose_<role>_channel``, or a
  ``<role>_channel`` that is not a number at all, which is what a settings CSV
  edited by hand hands the mask generators;
* the 4-D (Beta) path's per-frame notes, which are the only report the user
  gets of what the time-series segmenter decided.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

import spacr.object as O

# The Cellpose doubles are shared with the other object coverage suites so the
# call contract they assert against cannot drift apart from this one.
from tests.test_cov_object_masks_sam import (  # noqa: F401  (pytest fixtures)
    _close_figures,
    fake_model,
    force_cpu,
)
from tests.test_cov_object_cellpose_masks import fake_cellpose  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_npz(src: Path, n=2, h=32, w=32, c=3, name="batch1.npz"):
    """Write a pre-batched .npz whose channels are told apart by their content.

    Channel ``k`` is dominated by a bright band on row block ``k``, so the
    plane a segmenter was handed can be identified after normalisation.
    """
    src.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    data = rng.integers(100, 400, size=(n, h, w, c)).astype(np.uint16)
    for k in range(c):
        data[:, 4 * k:4 * k + 3, :, k] = 4000
    filenames = np.array([f"plate1_A01_{i + 1}.npy" for i in range(n)])
    np.savez(src / name, data=data, filenames=filenames)
    return data, [str(f) for f in filenames]


def _which_channel(plane, stack_image):
    """Return the stack channel index whose pattern ``plane`` reproduces."""
    flat = np.asarray(plane, dtype=float).ravel()
    scores = [abs(np.corrcoef(flat, stack_image[..., k].ravel().astype(float))[0, 1])
              for k in range(stack_image.shape[-1])]
    return int(np.argmax(scores)), scores


def _sam_settings(src, **over):
    s = {
        "src": str(src),
        "cell_channel": 0,
        "nucleus_channel": 1,
        "pathogen_channel": None,
        "magnification": 20,
        "batch_size": 50,
        "verbose": False,
        "plot": False,
        "save": True,
        "timelapse": False,
        "n_jobs": 1,
        "cell_min_split_area": 0,
        "nucleus_min_split_area": 0,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# scikit-image 0.25 fallback
# ---------------------------------------------------------------------------

def _old_remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """scikit-image 0.25's ``remove_small_objects``: no ``max_size`` keyword."""
    from skimage.measure import label as sk_label
    labelled = sk_label(np.asarray(ar), connectivity=connectivity)
    keep = np.zeros(np.asarray(ar).shape, dtype=bool)
    for value in np.unique(labelled):
        if value == 0:
            continue
        component = labelled == value
        if component.sum() >= min_size:
            keep |= component
    return keep


def _old_remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    """scikit-image 0.25's ``remove_small_holes``: no ``max_size`` keyword."""
    filled = np.asarray(ar).copy()
    holes = ~np.asarray(ar)
    from skimage.measure import label as sk_label
    labelled = sk_label(holes, connectivity=connectivity)
    for value in np.unique(labelled):
        if value == 0:
            continue
        component = labelled == value
        if component.sum() < area_threshold:
            filled |= component
    return filled


def test_the_old_skimage_call_keeps_the_same_size_boundary(monkeypatch):
    """On 0.25 the fallback must pass min_size, not the decremented threshold.

    0.26 removes objects strictly SMALLER than ``max_size`` + 1, so spaCR
    passes ``min_size - 1``. Handing that same number to the old keyword would
    delete every object of exactly ``min_size`` px -- objects the user asked to
    keep.
    """
    seen = {}

    def _recording(ar, min_size=64, connectivity=1, *, out=None):
        seen["min_size"] = min_size
        return _old_remove_small_objects(ar, min_size, connectivity, out=out)

    monkeypatch.setattr(O, "remove_small_objects", _recording)

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True       # area 4
    mask[5, 5:7] = True         # area 2
    result = O._remove_objects_smaller_than(mask, 4)

    assert seen["min_size"] == 4
    assert result[1:3, 1:3].all(), "an object of exactly min_size is kept"
    assert not result[5, 5:7].any()


def test_the_old_skimage_hole_call_keeps_the_same_area_boundary(monkeypatch):
    """The hole filler's fallback keeps a hole of exactly area_threshold."""
    seen = {}

    def _recording(ar, area_threshold=64, connectivity=1, *, out=None):
        seen["area_threshold"] = area_threshold
        return _old_remove_small_holes(ar, area_threshold, connectivity, out=out)

    monkeypatch.setattr(O, "remove_small_holes", _recording)

    mask = np.ones((10, 10), dtype=bool)
    mask[2:4, 2:4] = False      # area 4
    mask[7, 7:9] = False        # area 2
    result = O._fill_holes_smaller_than(mask, 4)

    assert seen["area_threshold"] == 4
    assert not result[2:4, 2:4].any(), "a hole of exactly the threshold stays"
    assert result[7, 7:9].all()


# ---------------------------------------------------------------------------
# The recorded-channel fallback in both Cellpose generators
# ---------------------------------------------------------------------------

def test_sam_keeps_a_recorded_channel_position(tmp_path, fake_model):
    """The recorded ``cellpose_<role>_channel`` positions are used as written.

    They are the positions ``io.preprocess_img_data`` actually gave each role
    on the merged stack. The fallback below them only guesses those positions
    for a resumed run that has none; when they ARE present, recomputing them
    would segment a different plane than the one the stack was built for. The
    pair recorded here is deliberately not the one role order would produce,
    so a recomputation is visible.
    """
    src = tmp_path / "stack"
    data, _ = _write_npz(src, c=2)

    settings = _sam_settings(src, cellpose_cell_channel=0,
                             cellpose_nucleus_channel=1)
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    model = fake_model["model"]
    image = model.eval_inputs[0][0]
    first, scores = _which_channel(image[..., 0], data[0])
    second, _ = _which_channel(image[..., 1], data[0])
    assert (first, second) == (0, 1), (
        f"cellpose was handed stack planes {(first, second)} instead of the "
        f"recorded (0, 1) (correlations for plane 0: {scores})")


def test_sam_survives_a_channel_that_is_not_a_number(tmp_path, fake_model):
    """A non-numeric ``<role>_channel`` is skipped, not fatal.

    Settings CSVs are edited by hand, and an organelle channel left as text
    used to take the whole mask run down before the first field was read.
    """
    src = tmp_path / "stack"
    _write_npz(src, c=3)

    settings = _sam_settings(src, organelle_channel="not-a-channel")
    assert O.generate_cellpose_masks_sam(str(src), settings, "cell") is None

    written = sorted(p.name for p in (src / "cell_mask_stack").iterdir())
    assert written == ["plate1_A01_1.npy", "plate1_A01_2.npy"]
    assert "organelle" not in fake_model["model"].eval_kwargs[0]


def test_the_chosen_model_generator_keeps_a_recorded_channel_position(
        tmp_path, fake_cellpose):
    """``generate_cellpose_masks`` resolves channels exactly like its SAM sibling.

    The two generators read the same settings; if only one honoured the
    recorded positions, the same run would segment different planes depending
    on which model the user picked.
    """
    src = tmp_path / "stack"
    data, _ = _write_npz(src, c=2)

    settings = _sam_settings(src, cellpose_nucleus_channel=1,
                             cellpose_cell_channel=0, filter=False,
                             seg_qc="off")
    O.generate_cellpose_masks(str(src), settings, "nucleus")

    model = fake_cellpose["model"]
    index, scores = _which_channel(model.eval_inputs[0][0][..., 0], data[0])
    assert index == 1, (
        f"the nucleus plane came from stack channel {index}, not the recorded "
        f"position 1 (correlations {scores})")


def test_the_chosen_model_generator_survives_a_text_channel(tmp_path,
                                                            fake_cellpose):
    """A non-numeric channel is skipped here too, and the run finishes."""
    src = tmp_path / "stack"
    _write_npz(src, c=3)

    settings = _sam_settings(src, organelle_channel="", filter=False,
                             seg_qc="off")
    assert O.generate_cellpose_masks(str(src), settings, "nucleus") is None
    written = sorted(p.name for p in (src / "nucleus_mask_stack").iterdir())
    assert written == ["plate1_A01_1.npy", "plate1_A01_2.npy"]


# ---------------------------------------------------------------------------
# 4-D notes
# ---------------------------------------------------------------------------

def test_the_time_series_segmenter_reports_its_notes(tmp_path, fake_model,
                                                     monkeypatch, capsys):
    """Verbose 4-D runs print the acquisition's notes and each frame's own.

    The notes are how the user learns what the 4-D path decided -- which axis
    it read as time, whether a frame was projected. Printed per acquisition
    AND per frame, because a single bad frame is otherwise invisible.
    """
    src = tmp_path / "stack"
    _write_npz(src, n=2, c=2)
    monkeypatch.setattr(O, "display", lambda *a, **k: None)

    class _Result:
        """Stand-in for the TStackResult the 4-D segmenter returns."""

        def __init__(self):
            self.notes = ["t_axis=0 read as time"]
            self.z_results = [
                type("Z", (), {"notes": ["frame 1 projected"]})(),
                type("Z", (), {"notes": ["frame 2 projected"]})(),
            ]

    result = _Result()

    def _fake_t(acquisition, model, t_plan, eval_kwargs):
        frames = np.asarray(acquisition)
        masks = [np.zeros(frames.shape[1:3], dtype=np.uint16)
                 for _ in range(frames.shape[0])]
        for mask in masks:
            mask[2:8, 2:8] = 1
        return masks, result, None

    monkeypatch.setattr(O, "_segment_timepoints_with_t", _fake_t)

    settings = _sam_settings(src, verbose=True, t_stack=True,
                             t_axis_order="TYX")
    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    out = capsys.readouterr().out
    assert "[4D] batch1: t_axis=0 read as time" in out
    assert "[4D] plate1_A01_1.npy: frame 1 projected" in out
    assert "[4D] plate1_A01_2.npy: frame 2 projected" in out


# ---------------------------------------------------------------------------
# Organelle generator
# ---------------------------------------------------------------------------

def test_the_organelle_generator_uses_the_recorded_plane(tmp_path):
    """A recorded ``cellpose_organelle_channel`` picks the plane to segment.

    Without it the raw ``organelle_channel`` is mapped through role order; with
    it, the position the writer used wins. Segmenting the wrong plane produces
    masks that look plausible and measure the wrong organelle.
    """
    from spacr.object import generate_organelle_masks_sam

    src = tmp_path / "masks"
    src.mkdir(parents=True)
    h = w = 64
    yy, xx = np.mgrid[:h, :w]
    blobs = np.full((h, w), 100, dtype=np.uint16)
    for cy, cx in ((16, 16), (16, 48), (48, 16), (48, 48)):
        blobs[(yy - cy) ** 2 + (xx - cx) ** 2 <= 9] = 3000
    flat = np.full((h, w), 100, dtype=np.uint16)
    # Plane 0 holds the organelles; plane 1 is featureless.
    data = np.stack([np.stack([blobs, flat], axis=-1)], axis=0)
    np.savez(src / "stack1.npz", data=data,
             filenames=np.array(["plate1_A01_1.npy"]))

    settings = {
        "verbose": False, "save": True, "plot": False, "batch_size": 4,
        "n_jobs": 1, "nucleus_channel": None, "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": 1,              # the raw channel...
        "cellpose_organelle_channel": 0,     # ...recorded at position 0
        "organelle_morphology": "spots",
        "organelle_method": "otsu",
        "organelle_model_name": "cpsam",
    }
    generate_organelle_masks_sam(str(src), settings, "organelle")

    mask = np.load(src / "organelle_mask_stack" / "plate1_A01_1.npy")
    labels = set(np.unique(mask).tolist()) - {0}
    assert len(labels) == 4, (
        f"expected the four blobs on plane 0, got {len(labels)} objects -- "
        "the featureless plane was segmented instead")
